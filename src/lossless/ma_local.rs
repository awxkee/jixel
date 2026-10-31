/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! Per-group local learned-tree training and encoding.

use super::frame::keep_smaller_writer;
use super::groups::{DeepLzScratchPool, group_lz_threads};
use super::lz77::{
    ConstantTail, LzToken, build_lz_pixel_code_tails, estimate_coded_bits,
    estimate_literal_and_run_bits_single, lz77_compress_for_speed_with_depth, lz77_run_count,
    write_lz_section_tail, write_tree_lz77,
};
use super::ma_encoding::tokenize_ma_rect;
use super::ma_learning::{finish_ma_candidate, ma_leaf_price, ma_learn_params, with_ma_prune};
use super::ma_sampling::{ma_channel_sample_count, sample_channel_ma};
use super::ma_source::{
    LearnedCandidate, MA_MAX_LEAVES, MA_MIN_NODE_SAMPLES, MaPlacement, MaSource, MaTransform,
    ac_stream_id, with_ma_pixels,
};
use super::ma_streams::{
    LZ_LITERAL_ALTERNATIVE_MAX_RATIO, VARIANT_WRITE_TOLERANCE, priced_tree_streams,
};
use super::tokens::RawTokens;
use super::{GroupLayout, lz77};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;
use crate::ma_tree::{MaSamples, learn_ma_tree_in_place};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpParams, write_wp_header};

/// Samples the local learner draws per group (a full 1024-px RGB group is
/// sampled at stride 3, a 256-px one fully). Denser than the global learn:
/// the payoff of local trees is finding more of the near-deterministic
/// contexts of text and graphics, which the global stride starves (the IDE
/// screenshot: local trees at the global stride are neutral, at stride 7
/// −4%, at stride 3 −8%; the deterministic-node shortcut in the learner
/// keeps the dense learn affordable).
const LOCAL_TREE_TARGET_SAMPLES: usize = 1 << 20;

/// Groups learned concurrently; each lane holds one group's samples
/// (~110 B per sample, ~115 MiB at the sample target). Beyond four lanes the
/// learners' own row-chunk and per-property parallelism already occupy the
/// workers.
const LOCAL_TREE_LEARN_LANES: usize = 4;

/// Local trees are tried when the global tree grew to at least half its
/// leaf budget and still codes the frame under `LOCAL_TREE_MAX_BITS_PER_VALUE`
/// bits per value: many contexts, most of them near-deterministic, is the
/// profile that keeps gaining from denser sampling (IDE screenshot 0.30 b/v
/// at 4095 leaves: −8%, its patched frame 0.27 b/v at 3773 leaves; a
/// Buddhabrot render 0.74: −1%; photos 2.2–2.8: −1..0%; a mostly flat
/// screenshot at 743 leaves: +0.6%; every attempt costs ~2× encode time).
pub(super) const LOCAL_TREE_MIN_GLOBAL_LEAVES: usize = MA_MAX_LEAVES / 2;

pub(super) const LOCAL_TREE_MAX_BITS_PER_VALUE: f64 = 0.5;

/// Local per-group learned trees: every AC group section carries its own
/// tree, histograms and context map (`use_global_tree = 0`), learned from
/// that group's pixels alone, and the DC-global section declares no tree.
/// This is libjxl's streaming layout for large images. Groups with distinct
/// content (code text next to UI chrome) each get a full leaf budget and
/// their own cluster budget without spending splits on the stream id, and
/// each is learned from a denser sample of its own pixels
/// (`LOCAL_TREE_TARGET_SAMPLES`), which is where the gain comes from: at the
/// global learn's stride local trees are neutral. Every group picks literal
/// or LZ77 tokens on its own code. Returns the sections in codestream
/// order, or `None` when the source has a global stream.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_local_tree_sections(
    speed: crate::Speed,
    source: &MaSource<'_>,
    placement: &[MaPlacement],
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
    num_dc_groups: usize,
    min_symbol: u32,
    wp_params: WpParams,
    use_wp: bool,
    dictionary: Option<&BitWriter>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<Vec<BitWriter>> {
    let gdim = layout.dim();
    let xsize_groups = xsize.div_ceil(gdim);
    let ysize_groups = ysize.div_ceil(gdim);
    let num_ac_groups = xsize_groups * ysize_groups;
    if num_ac_groups < 2 || placement.iter().any(|p| p.global) {
        return None;
    }
    let target = LOCAL_TREE_TARGET_SAMPLES;
    let max_candidates = source.max_candidates();
    let leaf_offsets = matches!(source.transform, MaTransform::Rct(_));
    let group_rect = |group_index: usize| {
        let x0 = (group_index % xsize_groups) * gdim;
        let y0 = (group_index / xsize_groups) * gdim;
        (x0, y0)
    };

    // Sample and learn each group in its own lane (the learner also
    // parallelizes inside); a lane holds one group's sample set at a time.
    let cands: Vec<LearnedCandidate> = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        LOCAL_TREE_LEARN_LANES,
        |group_index, scratch| {
            let (x0, y0) = group_rect(group_index);
            let stream = ac_stream_id(num_dc_groups, group_index);
            let sampled = || {
                placement
                    .iter()
                    .filter(|p| source.channels[p.channel].constant.is_none())
            };
            let group_values: usize = sampled()
                .map(|p| {
                    let ch = &source.channels[p.channel];
                    gdim.min(ch.w - x0) * gdim.min(ch.h - y0)
                })
                .sum();
            let mut stride = group_values.div_ceil(target).max(1);
            if stride > 1 && stride.is_multiple_of(2) {
                stride += 1;
            }
            let capacity = sampled()
                .map(|p| {
                    let ch = &source.channels[p.channel];
                    ma_channel_sample_count(gdim.min(ch.w - x0), gdim.min(ch.h - y0), stride)
                })
                .sum();
            let mut samples = MaSamples::with_capacity(capacity);
            for p in sampled() {
                let ch = &source.channels[p.channel];
                let gw = gdim.min(ch.w - x0);
                let gh = gdim.min(ch.h - y0);
                let refs = source.ref_planes(placement, p, x0, y0);
                with_ma_pixels!(ch, |pixels| sample_channel_ma(
                    |y| &pixels[(y0 + y) * ch.w + x0..][..gw],
                    gw,
                    gh,
                    p.chan_id,
                    stream,
                    &refs,
                    wp_params,
                    use_wp,
                    stride,
                    |props, tok, val| samples.push_val(props, tok, val),
                ));
            }
            // Too few samples to learn from: the root alone (one predictor).
            let max_leaves = if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
                1
            } else {
                source.max_leaves()
            };
            // Offsets inspect every fourth row in the original order. Keep
            // only those rows; the dense learner can partition the full set
            // in place instead of allocating a second full set and indices.
            let offsets = (leaf_offsets && max_leaves > 1).then(|| MaSamples {
                props: samples.props.sampled(4),
                tok: samples.tok.iter().step_by(4).copied().collect(),
                val: Vec::new(),
            });
            let params = with_ma_prune(
                ma_learn_params(
                    min_symbol,
                    max_leaves,
                    max_candidates,
                    stride as f64,
                    use_wp,
                    !leaf_offsets,
                ),
                stride as f64,
            );
            let leaf_price = ma_leaf_price(&params, stride as f64);
            let tree = learn_ma_tree_in_place(&mut samples, params, pool, scratch);
            drop(samples);
            let mut cand = finish_ma_candidate(tree, stride as f64, leaf_price);
            if let Some(offsets) = offsets {
                cand.set_leaf_offsets(&offsets, 1);
            }
            cand
        },
    );

    let constant = source.constant_placement(placement);
    let cands: Vec<LearnedCandidate> = match constant {
        Some((p, value)) => cands
            .iter()
            .map(|cand| cand.with_constant_channel(p.chan_id, value))
            .collect(),
        None => cands,
    };

    // Tokenize every group through its own tree, then code each group on
    // its own histograms. Retain raw tokens only for active coding lanes.
    let cands_ref = &cands;
    let deep_lz_max_ratio = source.deep_lz_max_ratio();
    let deep_lz = DeepLzScratchPool::new(group_lz_threads(speed, pool));
    let group_sections: Vec<BitWriter> = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(speed, pool),
        |group_index, scratch| {
            let (x0, y0) = group_rect(group_index);
            let num_values = placement
                .iter()
                .map(|p| {
                    let ch = &source.channels[p.channel];
                    gdim.min(ch.w - x0) * gdim.min(ch.h - y0)
                })
                .sum();
            let mut toks = RawTokens::with_capacity(num_values);
            let mut tail = ConstantTail::NONE;
            for p in placement {
                let ch = &source.channels[p.channel];
                if ch.constant.is_some() {
                    tail = cands_ref[group_index]
                        .constant_tail(gdim.min(ch.w - x0), gdim.min(ch.h - y0));
                    continue;
                }
                tokenize_ma_rect(
                    source,
                    placement,
                    p,
                    ac_stream_id(num_dc_groups, group_index),
                    x0,
                    y0,
                    gdim.min(ch.w - x0),
                    gdim.min(ch.h - y0),
                    &cands_ref[group_index],
                    wp_params,
                    use_wp,
                    &mut toks,
                );
            }
            let cand = &cands_ref[group_index];
            match &toks {
                RawTokens::Compact(tokens) => write_local_tree_group(
                    speed,
                    tokens,
                    tail,
                    cand,
                    min_symbol,
                    wp_params,
                    deep_lz_max_ratio,
                    &deep_lz,
                    scratch,
                ),
                RawTokens::Wide(tokens) => write_local_tree_group(
                    speed,
                    tokens,
                    tail,
                    cand,
                    min_symbol,
                    wp_params,
                    deep_lz_max_ratio,
                    &deep_lz,
                    scratch,
                ),
            }
        },
    );

    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();
    if let Some(dictionary) = dictionary {
        sections[0].append(dictionary);
    }
    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 0); // has_tree = 0
    sections[0].write(1, 0); // use_global_tree = 0 (no channel is coded here)
    write_wp_header(wp_params, &mut sections[0]);
    source.write_transforms(&mut sections[0]);
    // libjxl skips an empty stream's tree; jxl-oxide parses one.
    let empty_code = crate::modular::build_pixel_code(&[Token::new(0, 0)], scratch);
    crate::modular::write_tree_and_pixel_histograms(&empty_code, scratch, &mut sections[0]);
    sections[0].zero_pad_to_byte();
    for section in sections[1..num_dc_groups + 2].iter_mut() {
        section.write(1, 0); // use_global_tree = 0
        write_wp_header(wp_params, section);
        section.write(2, 0); // no transforms
        section.zero_pad_to_byte();
    }
    for (group_index, section) in group_sections.into_iter().enumerate() {
        sections[2 + num_dc_groups + group_index] = section;
    }
    Some(sections)
}

/// One AC group section under its local tree: GroupHeader, tree,
/// histograms and tokens. The literal and LZ77 variants are ordered by their
/// order-0 estimates and a later one is written only when its code-based
/// estimate is within `VARIANT_WRITE_TOLERANCE`, as on the global path.
fn write_local_tree_group<T: lz77::LiteralToken>(
    speed: crate::Speed,
    tokens: &[T],
    tail: ConstantTail,
    cand: &LearnedCandidate,
    min_symbol: u32,
    wp_params: WpParams,
    deep_lz_max_ratio: f64,
    deep_lz: &DeepLzScratchPool,
    scratch: &mut CoderScratch,
) -> BitWriter {
    let num_ctx = cand.num_ctx;
    let distance_ctx = num_ctx;
    let run_count = lz77_run_count(tokens);
    let mut best: Option<BitWriter> = None;
    let mut best_estimate = f64::INFINITY;
    let deep = |scratch: &mut CoderScratch| -> Vec<LzToken> {
        deep_lz.with_depth(|depth| {
            lz77_compress_for_speed_with_depth(
                tokens,
                distance_ctx,
                speed,
                depth,
                Some(run_count),
                scratch,
            )
        })
    };
    if (run_count as f64) >= LZ_LITERAL_ALTERNATIVE_MAX_RATIO * tokens.len() as f64 {
        let lz = deep(scratch);
        write_local_tree_variant(
            speed,
            &lz,
            tail,
            cand,
            min_symbol,
            wp_params,
            scratch,
            &mut best,
            &mut best_estimate,
        );
    } else {
        let (e_lit, e_run) =
            estimate_literal_and_run_bits_single(tokens, num_ctx as usize + 1, min_symbol);
        if e_run > e_lit * deep_lz_max_ratio {
            write_local_tree_variant(
                speed,
                tokens,
                tail,
                cand,
                min_symbol,
                wp_params,
                scratch,
                &mut best,
                &mut best_estimate,
            );
        } else if e_run < e_lit {
            let lz = deep(scratch);
            write_local_tree_variant(
                speed,
                &lz,
                tail,
                cand,
                min_symbol,
                wp_params,
                scratch,
                &mut best,
                &mut best_estimate,
            );
            write_local_tree_variant(
                speed,
                tokens,
                tail,
                cand,
                min_symbol,
                wp_params,
                scratch,
                &mut best,
                &mut best_estimate,
            );
        } else {
            write_local_tree_variant(
                speed,
                tokens,
                tail,
                cand,
                min_symbol,
                wp_params,
                scratch,
                &mut best,
                &mut best_estimate,
            );
            let lz = deep(scratch);
            write_local_tree_variant(
                speed,
                &lz,
                tail,
                cand,
                min_symbol,
                wp_params,
                scratch,
                &mut best,
                &mut best_estimate,
            );
        }
    }
    if speed.effort().lz_priced_search() {
        let (e_lit, _) =
            estimate_literal_and_run_bits_single(tokens, num_ctx as usize + 1, min_symbol);
        if let Some(streams) = priced_tree_streams(
            speed,
            &[tokens],
            e_lit,
            num_ctx as usize + 1,
            min_symbol,
            &ThreadPool::new_lossless(1),
            scratch,
        ) {
            write_local_tree_variant(
                speed,
                &streams[0],
                tail,
                cand,
                min_symbol,
                wp_params,
                scratch,
                &mut best,
                &mut best_estimate,
            );
        }
    }
    best.expect("local-tree group section")
}

fn write_local_tree_variant<T: lz77::LzTokenSource>(
    speed: crate::Speed,
    stream: &[T],
    tail: ConstantTail,
    cand: &LearnedCandidate,
    min_symbol: u32,
    wp_params: WpParams,
    scratch: &mut CoderScratch,
    best: &mut Option<BitWriter>,
    best_estimate: &mut f64,
) {
    let num_ctx = cand.num_ctx;
    let distance_ctx = num_ctx;
    let code = build_lz_pixel_code_tails(
        speed.effort().lz_exhaustive_entropy(),
        &[stream],
        &[tail],
        num_ctx as usize,
        min_symbol,
        true,
        true,
        None,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let estimate = estimate_coded_bits(&[stream], distance_ctx, &code, min_symbol, None);
    if estimate >= *best_estimate * VARIANT_WRITE_TOLERANCE {
        return;
    }
    *best_estimate = best_estimate.min(estimate);
    let mut section = BitWriter::new();
    section.write(1, 0); // use_global_tree = 0
    write_wp_header(wp_params, &mut section);
    section.write(2, 0); // no transforms
    write_tree_lz77(
        &cand.tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section_tail(stream, tail, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();
    keep_smaller_writer(best, section);
}
