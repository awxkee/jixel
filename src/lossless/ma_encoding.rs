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

//! Learned-tree frame and section encoding.

use super::frame::keep_smaller_writer;
use super::groups::group_lz_threads;
use super::header::{write_frame_header_modular_kind, write_patch_dictionary, write_toc_entry};
use super::lz77::{
    ConstantTail, build_lz_pixel_code_tails, estimate_coded_bits, write_lz_section_tail,
    write_tree_lz77,
};
use super::ma_local::{
    LOCAL_TREE_MAX_BITS_PER_VALUE, LOCAL_TREE_MIN_GLOBAL_LEAVES, write_local_tree_sections,
};
use super::ma_lookup::MaLookup;
use super::ma_sampling::tokenize_channel_ma;
use super::ma_source::{
    LearnedCandidate, MaChannel, MaPlacement, MaSource, ac_stream_id, with_ma_pixels,
};
use super::ma_streams::{TreeStreams, VARIANT_WRITE_TOLERANCE, choose_tree_streams};
use super::palette::write_palette_transform;
use super::tokens::{RawTokenStreams, RawTokens};
use super::{GroupLayout, lz77};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpParams, write_wp_header};

// Compile both representations once; use the compact path only when every
// stream fits. The wide fallback retains the same stream order and values.
macro_rules! with_raw_streams {
    ($raw:expr, |$streams:ident| $body:block) => {
        match RawTokenStreams::new($raw) {
            RawTokenStreams::Compact($streams) => $body,
            RawTokenStreams::Wide($streams) => $body,
        }
    };
}

/// Write the complete learned-tree frame under `layout`: a local tree in the
/// single section when the frame is one group, otherwise a global tree in the
/// DC-global section (followed by any global-stream channels) with every AC
/// group routed through it (fresh WP state and group-local coordinates per
/// group, matching the decoder). Returns the candidate's estimated fractional
/// saving over the flat path.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_learned_tree_frame(
    speed: crate::Speed,
    source: &MaSource<'_>,
    has_alpha: bool,
    frame_kind: ModularFrameKind<'_>,
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    use_wp: bool,
    cand: &LearnedCandidate,
    local_trees: bool,
    writer: &mut BitWriter,
) -> f64 {
    let gdim = layout.dim();
    let xsize_groups = xsize.div_ceil(gdim);
    let ysize_groups = ysize.div_ceil(gdim);
    let num_ac_groups = xsize_groups * ysize_groups;
    let num_dc_groups = xsize.div_ceil(layout.lf_dim()) * ysize.div_ceil(layout.lf_dim());
    let single_group = num_ac_groups == 1;
    let placement = source.placement(layout, single_group);
    let constant = source.constant_placement(&placement);
    let wrapped;
    let cand = match constant {
        Some((p, value)) => {
            wrapped = cand.with_constant_channel(p.chan_id, value);
            &wrapped
        }
        None => cand,
    };
    let num_ctx = cand.num_ctx;
    let distance_ctx = num_ctx;
    let tokenize = |p: &MaPlacement,
                    stream: i32,
                    x0: usize,
                    y0: usize,
                    w: usize,
                    h: usize,
                    out: &mut RawTokens| {
        tokenize_ma_rect(
            source, &placement, p, stream, x0, y0, w, h, cand, wp_params, use_wp, out,
        );
    };

    write_frame_header_modular_kind(has_alpha, frame_kind, layout, writer);
    // A patched frame's dictionary opens its first section, before the
    // dequant/tree bits, exactly as on the flat path.
    let dictionary: Option<BitWriter> = match frame_kind {
        ModularFrameKind::Patched { references, .. } => {
            let mut w = BitWriter::new();
            write_patch_dictionary(references, has_alpha, scratch, &mut w);
            Some(w)
        }
        _ => None,
    };
    let dictionary = dictionary.as_ref();

    if single_group {
        let channel_tokens = pool.steal_map(scratch, placement.len(), |i, _scratch| {
            let p = &placement[i];
            let ch = &source.channels[p.channel];
            if ch.constant.is_some() {
                return RawTokens::with_capacity(0);
            }
            let mut tokens = RawTokens::with_capacity(ch.w * ch.h);
            tokenize(p, 0, 0, 0, ch.w, ch.h, &mut tokens);
            tokens
        });
        let tail = constant.map_or(ConstantTail::NONE, |(p, _)| {
            let ch = &source.channels[p.channel];
            cand.constant_tail(ch.w, ch.h)
        });
        let mut section = with_raw_streams!(channel_tokens, |channel_tokens| {
            let mut tokens = Vec::with_capacity(source.total_values());
            for channel in channel_tokens {
                tokens.extend(channel);
            }

            let variants = choose_tree_streams(
                speed,
                vec![tokens],
                distance_ctx,
                num_ctx as usize + 1,
                min_symbol,
                source.deep_lz_max_ratio(),
                pool,
                scratch,
            );
            let mut best: Option<BitWriter> = None;
            let mut best_estimate = f64::INFINITY;
            for variant in variants {
                match variant {
                    TreeStreams::Literals(streams) => {
                        let stream = streams.into_iter().next().expect("one stream");
                        write_learned_single_variant(
                            speed,
                            &stream,
                            tail,
                            source,
                            cand,
                            min_symbol,
                            wp_params,
                            dictionary,
                            pool,
                            scratch,
                            &mut best,
                            &mut best_estimate,
                        );
                    }
                    TreeStreams::Lz(streams) => {
                        let stream = streams.into_iter().next().expect("one stream");
                        write_learned_single_variant(
                            speed,
                            &stream,
                            tail,
                            source,
                            cand,
                            min_symbol,
                            wp_params,
                            dictionary,
                            pool,
                            scratch,
                            &mut best,
                            &mut best_estimate,
                        );
                    }
                }
            }
            best.expect("learned-tree section")
        });

        writer.write(1, 0);
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(std::slice::from_mut(&mut section));
        writer.zero_pad_to_byte();
        return cand.estimated_savings;
    }

    // Global stream: channels that fit a group, coded in the DC-global section.
    let mut global_tokens = RawTokens::with_capacity(0);
    for p in placement.iter().filter(|p| p.global) {
        let ch = &source.channels[p.channel];
        if ch.constant.is_some() {
            continue;
        }
        tokenize(p, 0, 0, 0, ch.w, ch.h, &mut global_tokens);
    }
    let global_tail = constant
        .filter(|(p, _)| p.global)
        .map_or(ConstantTail::NONE, |(p, _)| {
            let ch = &source.channels[p.channel];
            cand.constant_tail(ch.w, ch.h)
        });
    let has_global_stream = !global_tokens.is_empty() || global_tail.count > 0;

    let group_placement: Vec<&MaPlacement> = placement
        .iter()
        .filter(|p| !p.global && source.channels[p.channel].constant.is_none())
        .collect();
    let group_constant = constant.filter(|(p, _)| !p.global);
    let group_tail = |group_index: usize| -> ConstantTail {
        if source.group_override(group_index).is_some() {
            return ConstantTail::NONE;
        }
        group_constant.map_or(ConstantTail::NONE, |(p, _)| {
            let ch = &source.channels[p.channel];
            let x0 = (group_index % xsize_groups) * gdim;
            let y0 = (group_index / xsize_groups) * gdim;
            cand.constant_tail(gdim.min(ch.w - x0), gdim.min(ch.h - y0))
        })
    };
    let group_tokens: Vec<RawTokens> = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(speed, pool),
        |group_index, _scratch| {
            let x0 = (group_index % xsize_groups) * gdim;
            let y0 = (group_index / xsize_groups) * gdim;
            if let Some(over) = source.group_override(group_index) {
                let num_values = over.channels.iter().map(|ch| ch.w * ch.h).sum();
                let mut toks = RawTokens::with_capacity(num_values);
                for (chan_id, ch) in over.channels.iter().enumerate() {
                    tokenize_ma_group_channel(
                        ch,
                        chan_id as u32,
                        ac_stream_id(num_dc_groups, group_index),
                        cand,
                        wp_params,
                        use_wp,
                        &mut toks,
                    );
                }
                return toks;
            }
            let num_values = group_placement
                .iter()
                .map(|p| {
                    let ch = &source.channels[p.channel];
                    gdim.min(ch.w - x0) * gdim.min(ch.h - y0)
                })
                .sum();
            let mut toks = RawTokens::with_capacity(num_values);
            for p in &group_placement {
                let ch = &source.channels[p.channel];
                let gw = gdim.min(ch.w - x0);
                let gh = gdim.min(ch.h - y0);
                tokenize(
                    p,
                    ac_stream_id(num_dc_groups, group_index),
                    x0,
                    y0,
                    gw,
                    gh,
                    &mut toks,
                );
            }
            toks
        },
    );
    // Stream 0 is the global stream (possibly empty), then one per AC group.
    let mut all_tokens: Vec<RawTokens> = Vec::with_capacity(1 + num_ac_groups);
    all_tokens.push(global_tokens);
    all_tokens.extend(group_tokens);
    let tails: Vec<ConstantTail> = std::iter::once(global_tail)
        .chain((0..num_ac_groups).map(group_tail))
        .collect();
    let mut sections = with_raw_streams!(all_tokens, |all_tokens| {
        let variants = choose_tree_streams(
            speed,
            all_tokens,
            distance_ctx,
            num_ctx as usize + 1,
            min_symbol,
            source.deep_lz_max_ratio(),
            pool,
            scratch,
        );
        let mut best: Option<Vec<BitWriter>> = None;
        let mut best_bits = usize::MAX;
        let mut best_estimate = f64::INFINITY;
        for variant in variants {
            match variant {
                TreeStreams::Literals(streams) => {
                    write_learned_grouped_variant(
                        speed,
                        &streams,
                        &tails,
                        source,
                        cand,
                        min_symbol,
                        wp_params,
                        pool,
                        scratch,
                        num_ac_groups,
                        num_dc_groups,
                        dictionary,
                        &mut best,
                        &mut best_bits,
                        &mut best_estimate,
                    );
                }
                TreeStreams::Lz(streams) => {
                    write_learned_grouped_variant(
                        speed,
                        &streams,
                        &tails,
                        source,
                        cand,
                        min_symbol,
                        wp_params,
                        pool,
                        scratch,
                        num_ac_groups,
                        num_dc_groups,
                        dictionary,
                        &mut best,
                        &mut best_bits,
                        &mut best_estimate,
                    );
                }
            }
        }
        best.expect("learned-tree sections")
    });

    // Local per-group trees (libjxl's streaming layout) compete on size.
    let local_gate = local_trees
        && cand.num_ctx as usize >= LOCAL_TREE_MIN_GLOBAL_LEAVES
        && cand.est_real <= LOCAL_TREE_MAX_BITS_PER_VALUE * source.total_values() as f64;
    if !has_global_stream
        && local_gate
        && let Some(local) = write_local_tree_sections(
            speed,
            source,
            &placement,
            xsize,
            ysize,
            layout,
            num_dc_groups,
            min_symbol,
            wp_params,
            use_wp,
            dictionary,
            pool,
            scratch,
        )
    {
        let global_bits: usize = sections.iter().map(BitWriter::bits_written).sum();
        let local_bits: usize = local.iter().map(BitWriter::bits_written).sum();
        if local_bits < global_bits {
            sections = local;
        }
    }

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    cand.estimated_savings
}

/// Tokenize the `w` x `h` rectangle at (`x0`, `y0`) of placement `p`'s
/// channel through `cand`'s tree. Placement is in source-channel order; its
/// channel ids already account for the independent global and group streams
/// (including palettes). The compiled lookup folds the channel and stream
/// ids, so it is built per (channel, stream): cheap next to tokenizing the
/// stream itself.
/// Tokenize one palettised group's own channel (whole channel, no reference
/// properties, channel ids numbered inside that group's stream).
#[allow(clippy::too_many_arguments)]
fn tokenize_ma_group_channel(
    ch: &MaChannel<'_>,
    chan_id: u32,
    stream: i32,
    cand: &LearnedCandidate,
    wp_params: WpParams,
    use_wp: bool,
    out: &mut RawTokens,
) {
    let lookup = MaLookup::new(&cand.tree, &cand.leaf_ctx, chan_id, stream);
    with_ma_pixels!(ch, |pixels| tokenize_channel_ma(
        |y| &pixels[y * ch.w..][..ch.w],
        ch.w,
        ch.h,
        chan_id,
        stream,
        &[],
        wp_params,
        use_wp,
        &lookup,
        &cand.leaf_offset,
        out,
    ));
}

pub(super) fn tokenize_ma_rect(
    source: &MaSource<'_>,
    placement: &[MaPlacement],
    p: &MaPlacement,
    stream: i32,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    cand: &LearnedCandidate,
    wp_params: WpParams,
    use_wp: bool,
    out: &mut RawTokens,
) {
    let ch = &source.channels[p.channel];
    debug_assert!(ch.constant.is_none(), "constant channels are tails");
    let lookup = MaLookup::new(&cand.tree, &cand.leaf_ctx, p.chan_id, stream);
    let refs = source.ref_planes(placement, p, x0, y0);
    with_ma_pixels!(ch, |pixels| tokenize_channel_ma(
        |y| &pixels[(y0 + y) * ch.w + x0..][..w],
        w,
        h,
        p.chan_id,
        stream,
        &refs,
        wp_params,
        use_wp,
        &lookup,
        &cand.leaf_offset,
        out,
    ));
}

#[allow(clippy::too_many_arguments)]
fn write_learned_single_variant<T: lz77::LzTokenSource>(
    speed: crate::Speed,
    stream: &[T],
    tail: ConstantTail,
    source: &MaSource<'_>,
    cand: &LearnedCandidate,
    min_symbol: u32,
    wp_params: WpParams,
    dictionary: Option<&BitWriter>,
    pool: &ThreadPool,
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
        Some(pool),
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let estimate = estimate_coded_bits(&[stream], distance_ctx, &code, min_symbol, Some(pool));
    if estimate >= *best_estimate * VARIANT_WRITE_TOLERANCE {
        return;
    }
    *best_estimate = (*best_estimate).min(estimate);
    let mut body = BitWriter::new();
    if let Some(dictionary) = dictionary {
        body.append(dictionary);
    }
    body.write(1, 1); // dc_quant all_default = 1
    body.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    body.write(1, 0); // use_global_tree = 0
    write_wp_header(wp_params, &mut body);
    source.write_transforms(&mut body);
    write_tree_lz77(
        &cand.tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut body,
    );
    write_lz_section_tail(stream, tail, distance_ctx, &code, min_symbol, &mut body);
    body.zero_pad_to_byte();
    keep_smaller_writer(best, body);
}

#[allow(clippy::too_many_arguments)]
fn write_learned_grouped_variant<T: lz77::LzTokenSource>(
    speed: crate::Speed,
    streams: &[Vec<T>],
    tails: &[ConstantTail],
    source: &MaSource<'_>,
    cand: &LearnedCandidate,
    min_symbol: u32,
    wp_params: WpParams,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    num_ac_groups: usize,
    num_dc_groups: usize,
    dictionary: Option<&BitWriter>,
    best: &mut Option<Vec<BitWriter>>,
    best_bits: &mut usize,
    best_estimate: &mut f64,
) {
    let num_ctx = cand.num_ctx;
    let distance_ctx = num_ctx;
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let slices: Vec<&[T]> = streams.iter().map(Vec::as_slice).collect();
    let code = build_lz_pixel_code_tails(
        speed.effort().lz_exhaustive_entropy(),
        &slices,
        tails,
        num_ctx as usize,
        min_symbol,
        true,
        true,
        Some(pool),
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let estimate = estimate_coded_bits(&slices, distance_ctx, &code, min_symbol, Some(pool));
    if estimate >= *best_estimate * VARIANT_WRITE_TOLERANCE {
        return;
    }
    *best_estimate = (*best_estimate).min(estimate);
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    if let Some(dictionary) = dictionary {
        sections[0].append(dictionary);
    }
    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(
        &cand.tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    write_wp_header(wp_params, &mut sections[0]);
    source.write_transforms(&mut sections[0]);
    // An empty global stream still carries its ANS state: jxl-oxide reads it.
    write_lz_section_tail(
        &streams[0],
        tails[0],
        distance_ctx,
        &code,
        min_symbol,
        &mut sections[0],
    );
    sections[0].zero_pad_to_byte();

    for section in sections[1..num_dc_groups + 1].iter_mut() {
        section.write(1, 1);
        write_wp_header(wp_params, section);
        section.write(2, 0);
        section.zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    write_wp_header(wp_params, &mut sections[ac_global_idx]);
    sections[ac_global_idx].zero_pad_to_byte();

    // The code borrows caller scratch; scratch-free maps let the same pool
    // write the independent group sections without aliasing that storage.
    let group_sections = pool.map(num_ac_groups, |group| {
        let mut section = BitWriter::new();
        section.write(1, 1);
        write_wp_header(wp_params, &mut section);
        // A palettised group declares its own Palette transform here.
        match source.group_override(group) {
            Some(over) => write_palette_transform(over.num_c, over.nb_colors, &mut section),
            None => section.write(2, 0),
        }
        write_lz_section_tail(
            &streams[1 + group],
            tails[1 + group],
            distance_ctx,
            &code,
            min_symbol,
            &mut section,
        );
        section.zero_pad_to_byte();
        section
    });
    for (group_index, section) in group_sections.into_iter().enumerate() {
        sections[2 + num_dc_groups + group_index] = section;
    }
    let bits: usize = sections.iter().map(BitWriter::bits_written).sum();
    if bits < *best_bits {
        *best_bits = bits;
        *best = Some(sections);
    }
}
