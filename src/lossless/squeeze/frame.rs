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

//! Lossy modular frame headers, DC previews, and group encoding.

use super::super::groups::{group_lz_threads, write_lz_groups};
use super::super::header::{write_toc_entry, write_u64};
use super::super::lz77::{
    LZ77_MIN_SYMBOL, LzToken, LzTokenSource, build_lz_pixel_code, build_lz_pixel_code_opts,
    lz77_compress_for_speed, write_lz_section, write_tree_lz77,
};
use super::super::ma_sampling::{MaRefPlane, sample_channel_ma};
use super::super::ma_source::{MA_MIN_NODE_SAMPLES, MA_PHOTO_CANDIDATES, MA_SPLIT_COST_BITS};
use super::super::predictor::{
    PREDICTOR_GRADIENT, SqueezePredictorCost, channel_to_context, choose_predictor_for_plane,
    fixed_predictor, selected_grad_pack_interior_fn, tokenize_plane_rows,
};
use super::super::tree::build_balanced_tree_tokens_mul;
use super::super::{GROUP_DIM, GroupLayout, LF_GROUP_DIM};
use super::forest::{
    LM_BUDGET_CAP, LM_BUDGET_DIV, LM_SAFE_PRED_MASK, LM_SAMPLE_TARGET, emit_lm_forest,
    scale_ma_value_props, tokenize_channel_ma_mult,
};
use super::quantize::{
    LOSSY_MODULAR_XYB_SCALE, LmPlainInput, LmQuantized, lm_quantize, lm_quantize_plain,
    lm_quantize_xyb,
};
use super::{LM_MAX_CONTEXTS, for_each_squeeze_group_crop};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;
use crate::ma_tree::{LearnedTree, MA_REF_CHANNELS, MaLearnParams, MaSamples, learn_ma_tree};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Regular is_last modular frame inside the lossy (`xyb_encoded`) codestream.
/// Identical to `write_frame_header_modular_flags` except the do_YCbCr bit is
/// absent: with xyb_encoded set the color transform is implicitly XYB.
pub(super) fn write_frame_header_modular_xyb_regular(epf: (u32, f32), w: &mut BitWriter) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(2, 0b00); // upsampling = 1
    w.write(2, GroupLayout::DEFAULT.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    w.write(1, 1); // is_last
    w.write(2, 0b00); // name length = 0
    write_lm_loop_filter(epf, w);
    w.write(2, 0b00); // no FH extensions
}

/// Modular DC frame (kDCFrame, dc_level 1) of the lossy codestream: like
/// the regular header up to the passes, then the level, no loop filters.
fn write_frame_header_modular_xyb_dc(w: &mut BitWriter) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b01); // DC frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(2, 0b00); // upsampling = 1
    w.write(2, GroupLayout::DEFAULT.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(2, 0b00); // dc_level = 1
    w.write(2, 0b00); // name length = 0
    write_lm_loop_filter((0, 0.0), w);
    w.write(2, 0b00); // no FH extensions
}

/// Loop-filter field of the modular frame header: no gaborish, EPF at the
/// arm's schedule (all defaults except the modular sigma).
fn write_lm_loop_filter(epf: (u32, f32), w: &mut BitWriter) {
    let (iters, sigma) = epf;
    w.write(1, 0); // loop_filter not all_default
    w.write(1, 0); // no gaborish
    w.write(2, iters as u64); // EPF iters
    if iters > 0 {
        w.write(1, 0); // epf_weight_custom = false
        w.write(1, 0); // epf_sigma_custom = false
        w.write(16, crate::util::f32_to_f16_bits(sigma) as u64); // epf_sigma_for_modular
    }
    w.write(2, 0b00); // no LF extensions
}

/// Custom dc_quant field: the decoder multiplies stored X/Y/B channels by these
/// steps to recover XYB floats.
pub(super) fn write_lossy_modular_dc_quant(w: &mut BitWriter) {
    use crate::util::f32_to_f16_bits;
    w.write(1, 0); // dc_quant not all_default
    for scale in LOSSY_MODULAR_XYB_SCALE {
        let step_x128 = 128.0 / scale;
        w.write(16, f32_to_f16_bits(step_x128) as u64);
    }
}

/// Regular single-pass modular frame of a non-XYB codestream, no loop
/// filters (same shape as the lossless header plus the YCbCr signaling).
pub(super) fn write_frame_header_modular_plain(
    ycbcr: bool,
    extra: usize,
    epf: (u32, f32),
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(1, u64::from(ycbcr)); // do_YCbCr
    if ycbcr {
        for _ in 0..3 {
            w.write(2, 0); // chroma subsampling 4:4:4
        }
    }
    w.write(2, 0b00); // upsampling = 1
    for _ in 0..extra {
        w.write(2, 0b00); // ec_upsampling = 1
    }
    w.write(2, GroupLayout::DEFAULT.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    for _ in 0..extra {
        w.write(2, 0b00); // extra-channel blending = Replace
    }
    w.write(1, 1); // is_last
    w.write(2, 0b00); // name length = 0
    write_lm_loop_filter(epf, w);
    w.write(2, 0b00); // no FH extensions
}

/// Lossy modular frame: the XYB image on the fixed integer lattice, squeezed,
/// with each pyramid level's residuals rounded to distance-scaled multiples,
/// then coded through the lossless squeeze machinery. `effective_distance` is
/// the arm's already-calibrated quantizer distance (see
/// `lm_calibrated_distance` and the auto gate). Returns the original frame
/// size in bits for the Auto gate, or None when this path is unsupported.
pub(crate) fn encode_frame_lossy_modular_squeeze(
    xyb: &crate::image::Image3F,
    effective_distance: f32,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    match lm_quantize(xyb, effective_distance) {
        Some(quantized) => encode_lm_frame(quantized, speed, pool, scratch, writer),
        None => None,
    }
}

/// DC-frame residual step at the finest squeeze level, in VarDCT DC steps,
/// per stored color [Y, X, B-Y].
const DC_FRAME_GAIN: [f32; 3] = [1.5, 2.5, 1.5];

/// Step ratio from one squeeze level to the next coarser one.
const DC_FRAME_LEVEL_RATIO: f32 = 0.6;

/// Progressive DC: the 1:8 DC image of a VarDCT frame as its squeezed DC
/// frame (frame type kDCFrame, dc_level 1), which the VarDCT frame then
/// references instead of coding its own DC. Full-resolution chroma, no loop
/// filters. `levels` keeps that many of the finest squeeze levels (clamped
/// to at least one; `None` squeezes down to the 8x8 preview). Returns false
/// when the image is too small to squeeze.
pub(crate) fn encode_dc_frame(
    dc: &crate::image::Image3F,
    vardct_steps: [f32; 3],
    levels: Option<u32>,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    // Stored color [Y, X, B-Y] -> the VarDCT DC step in lattice units.
    let unit = [
        vardct_steps[1] * LOSSY_MODULAR_XYB_SCALE[1],
        vardct_steps[0] * LOSSY_MODULAR_XYB_SCALE[0],
        vardct_steps[2] * LOSSY_MODULAR_XYB_SCALE[2],
    ];
    let q_of = |color: usize, hshift: i32, vshift: i32| -> i32 {
        let level = (hshift + vshift - 1).max(0);
        let q = unit[color] * DC_FRAME_GAIN[color] * DC_FRAME_LEVEL_RATIO.powi(level);
        (q.round() as i32).max(1)
    };
    let levels = levels.map_or(usize::MAX, |n| n.max(1) as usize);
    let steps = crate::squeeze::squeeze_steps_levels(dc.xsize(), dc.ysize(), 3, levels);
    match lm_quantize_xyb(dc, steps, q_of) {
        Some(quantized) => encode_lm_frame(
            LmQuantized {
                dc_frame: true,
                ..quantized
            },
            speed,
            pool,
            scratch,
            writer,
        )
        .is_some(),
        None => false,
    }
}

/// Whether [`encode_dc_frame`] can code a DC image of this size.
pub(crate) fn dc_frame_supported(xsize: usize, ysize: usize) -> bool {
    !crate::squeeze::default_squeeze_steps(xsize, ysize, 3).is_empty()
}

/// Lossy modular frame of a non-XYB codestream: the integer planes in
/// `input` squeezed and quantized at the plain-layout schedule (see
/// `lm_plain_distance`). Returns the original frame size in bits for the Auto
/// gate, or None when this path is unsupported.
pub(crate) fn encode_frame_lossy_modular_plain(
    input: &LmPlainInput,
    xsize: usize,
    ysize: usize,
    effective_distance: f32,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    match lm_quantize_plain(input, xsize, ysize, effective_distance) {
        Some(quantized) => encode_lm_frame(quantized, speed, pool, scratch, writer),
        None => None,
    }
}

/// Reuse fine-channel trees and local-node-id tokens while the coarse
/// representation changes. Keys retain original channel and stream identity.
struct LmTreeCache {
    original_len: usize,
    averages: usize,
    trees: Vec<[Option<LearnedTree>; 3]>,
    tokens: Vec<[LmCachedTokens; 3]>,
    token_bytes: usize,
}

#[derive(Default)]
struct LmCachedTokens {
    blocks: Vec<Option<Vec<Token>>>,
    reserved_bytes: usize,
}

struct LmPendingTokens {
    key: (usize, usize),
    group: usize,
    tokens: Vec<Token>,
}

const LM_TOKEN_CACHE_BYTES: usize = 64 * 1024 * 1024;

impl LmTreeCache {
    fn new(q: &LmQuantized) -> Self {
        Self {
            original_len: q.channels.len(),
            averages: q.steps.last().map_or(0, |step| step.num_c),
            trees: (0..q.channels.len()).map(|_| [None, None, None]).collect(),
            tokens: (0..q.channels.len())
                .map(|_| std::array::from_fn(|_| LmCachedTokens::default()))
                .collect(),
            token_bytes: 0,
        }
    }

    fn update_tree(&mut self, key: (usize, usize), tree: &LearnedTree) {
        let (channel, stream) = key;
        if self.trees[channel][stream]
            .as_ref()
            .is_none_or(|old| old.nodes != tree.nodes)
        {
            let old = std::mem::take(&mut self.tokens[channel][stream]);
            self.token_bytes -= old.reserved_bytes;
        }
        self.trees[channel][stream] = Some(tree.clone());
    }

    fn reserve_tokens(&mut self, key: (usize, usize), samples: usize, groups: usize) {
        let (channel, stream) = key;
        if !self.tokens[channel][stream].blocks.is_empty() {
            return;
        }
        let Some(bytes) = samples.checked_mul(std::mem::size_of::<Token>()) else {
            return;
        };
        if bytes > LM_TOKEN_CACHE_BYTES - self.token_bytes {
            return;
        }
        self.tokens[channel][stream] = LmCachedTokens {
            blocks: (0..groups).map(|_| None).collect(),
            reserved_bytes: bytes,
        };
        self.token_bytes += bytes;
    }

    fn block(&self, key: (usize, usize), group: usize) -> Option<&[Token]> {
        self.tokens[key.0][key.1].blocks.get(group)?.as_deref()
    }

    fn admitted(&self, key: (usize, usize)) -> bool {
        !self.tokens[key.0][key.1].blocks.is_empty()
    }

    fn retain(&mut self, pending: LmPendingTokens) {
        let entry = &mut self.tokens[pending.key.0][pending.key.1];
        if let Some(block) = entry.blocks.get_mut(pending.group) {
            debug_assert!(block.is_none());
            *block = Some(pending.tokens);
        }
    }

    fn key(&self, q: &LmQuantized, channel: usize, stream: usize) -> Option<(usize, usize)> {
        // The merged averages change on every fold. DC-frame global trees
        // also see previous same-shaped channels, whose reference set changes.
        if channel < self.averages || (q.dc_frame && stream == 0) {
            return None;
        }
        let original = channel + self.original_len - q.channels.len();
        debug_assert!(original < self.original_len);
        Some((original, stream))
    }
}

/// Race exact decoded representations, including the frame header, padding
/// and TOC. The original size is retained for the caller's Auto-arm decision.
fn encode_lm_frame(
    mut quantized: LmQuantized,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    // Screen the changed coarse prefix before allocating a fine-band cache
    // or rebuilding whole-frame entropy tables. Tiny expected savings do not
    // justify an O(pixels) trial. Fast presets and DC frames also stay direct.
    let Some(folded_steps) = super::fold_cost::select_fold(&quantized, speed, scratch) else {
        let start = writer.bits_written();
        encode_lm_frame_impl(&quantized, speed, pool, scratch, writer, None);
        return Some(writer.bits_written() - start);
    };
    let prefix = writer.clone();
    let mut cache = LmTreeCache::new(&quantized);
    let mut best = prefix.clone();
    encode_lm_frame_impl(
        &quantized,
        speed,
        pool,
        scratch,
        &mut best,
        Some(&mut cache),
    );
    let original_bits = best.bits_written() - prefix.bits_written();
    for _ in 0..folded_steps {
        if !quantized.fold_last() {
            *writer = best;
            return Some(original_bits);
        }
    }
    // Even exhaustive effort serializes at most one shortlisted candidate.
    let mut candidate = prefix.clone();
    encode_lm_frame_impl(
        &quantized,
        speed,
        pool,
        scratch,
        &mut candidate,
        Some(&mut cache),
    );
    if candidate.bits_written() < best.bits_written() {
        best = candidate;
    }
    *writer = best;
    Some(original_bits)
}

/// Code one quantized pyramid as a complete modular frame (header, TOC and
/// sections). Fine-channel trees and local tokens can be reused across candidates.
fn encode_lm_frame_impl(
    quantized: &LmQuantized,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
    mut cache: Option<&mut LmTreeCache>,
) {
    let channels = &quantized.channels;
    let quants = &quantized.quants;
    let steps = &quantized.steps;
    let layout = quantized.layout;
    let epf = quantized.epf;
    let dc_frame = quantized.dc_frame;
    let (xsize, ysize) = (quantized.xsize, quantized.ysize);
    let nb = channels.len();

    // Same literal-vs-LZ77 guard as the atlas path, from the actual quantized
    // channel range.
    let max_abs = channels
        .iter()
        .flat_map(|ch| ch.data.iter())
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    let value_bits = 33 - max_abs.saturating_mul(2).max(1).leading_zeros();
    let min_symbol = if value_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * value_bits + 24
    };

    let use_wp = true;
    let grad_pack_fn = selected_grad_pack_interior_fn();
    let xsize_groups = xsize.div_ceil(GROUP_DIM);
    let ysize_groups = ysize.div_ceil(GROUP_DIM);
    let num_ac_groups = xsize_groups * ysize_groups;
    let xsize_dc_groups = xsize.div_ceil(LF_GROUP_DIM);
    let ysize_dc_groups = ysize.div_ceil(LF_GROUP_DIM);
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;

    if num_ac_groups == 1 && !dc_frame {
        // Single-group frame: one section carrying everything.
        let predictors: Vec<u32> = if speed.effort().predictor_search {
            let quants_ref = &quants;
            pool.steal_map(scratch, nb, |c, _scratch| {
                let ch = &channels[c];
                let data = &ch.data;
                let w = ch.w;
                let get = move |gx: usize, gy: usize| data[gy * w + gx];
                if quants_ref[c] > 1 {
                    let mut costs = SqueezePredictorCost::with_zero();
                    costs.add_crop(get, ch.w, ch.h, use_wp);
                    costs.safe_predictor()
                } else {
                    choose_predictor_for_plane(get, ch.w, ch.h, use_wp)
                }
            })
        } else {
            (0..nb)
                .map(|i| {
                    if quants[i] > 1 {
                        PREDICTOR_GRADIENT
                    } else {
                        fixed_predictor(use_wp)
                    }
                })
                .collect()
        };
        let channel_tokens = pool.steal_map(scratch, nb, |c, scratch| {
            let ch = &channels[c];
            let mut tokens = Vec::with_capacity(ch.w * ch.h);
            tokenize_plane_rows(
                channel_to_context(c, nb),
                |y| &ch.data[y * ch.w..][..ch.w],
                ch.w,
                ch.h,
                predictors[c],
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
            tokens
        });
        let mut tokens: Vec<Token> = Vec::new();
        for channel in channel_tokens {
            tokens.extend(channel);
        }

        if dc_frame {
            write_frame_header_modular_xyb_dc(writer);
        } else {
            layout.write_frame_header(epf, writer);
        }
        let mut section = BitWriter::new();
        layout.write_dc_quant(&mut section);
        section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
        section.write(1, 0); // use_global_tree = 0
        section.write(1, 1); // wp_default = 1
        layout.write_transforms(steps, &mut section);

        let distance_ctx = nb as u32;
        let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
        let code = build_lz_pixel_code(
            speed.effort().lz_exhaustive_entropy(),
            std::iter::once(lz_tokens.as_slice()),
            nb,
            min_symbol,
            speed.effort().lz_refined_entropy,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        let leaves: Vec<(u32, u32)> = predictors
            .iter()
            .copied()
            .zip(quants.iter().copied())
            .collect();
        write_tree_lz77(
            &build_balanced_tree_tokens_mul(&leaves),
            &code,
            min_symbol,
            &mut scratch.huffman_pool,
            &mut section,
        );
        write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
        section.zero_pad_to_byte();

        writer.write(1, 0); // no permutation
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(std::slice::from_mut(&mut section));
        writer.zero_pad_to_byte();
        return;
    }

    // Multi-group. Group streams key the tree on the WITHIN-group channel
    // slot while the global stream keys on the real channel index, and the
    // slot->channel map differs between them; with per-channel multipliers the
    // two sides can no longer share leaves the way the lossless path does. The
    // global tree therefore splits first on the static stream-id property
    // (property 1: 0 = LfGlobal, >0 = any group stream) and hangs a separate
    // channel chain off each side.
    let split = channels
        .iter()
        .position(|c| c.w > GROUP_DIM || c.h > GROUP_DIM)
        .unwrap_or(nb);
    // Group slots: the DC-group-eligible (min shift >= 3, above ~2048px) and
    // AC-group-eligible channels, each in the group scan order. A channel's
    // within-group slot is its rank in its list (full-frame channels never
    // produce an empty crop, so ranks are group-invariant).
    let dc_slot_channels: Vec<usize> = (split..nb)
        .filter(|&c| channels[c].hshift.min(channels[c].vshift) >= 3)
        .collect();
    let ac_slot_channels: Vec<usize> = (split..nb)
        .filter(|&c| {
            let msh = channels[c].hshift.min(channels[c].vshift);
            (0..=2).contains(&msh)
        })
        .collect();
    let n_dc_slots = dc_slot_channels.len();
    let n_ac_slots = ac_slot_channels.len();
    let _ = n_ac_slots;
    let n_tags = split + n_dc_slots + n_ac_slots;
    let dc_tag_base = split;
    let ac_tag_base = split + n_dc_slots;

    // Tags: [0, split) global-stream channels, then DC slots, then AC slots.
    let tag_channel: Vec<usize> = (0..split)
        .chain(dc_slot_channels.iter().copied())
        .chain(ac_slot_channels.iter().copied())
        .collect();
    let tag_quants: Vec<u32> = tag_channel.iter().map(|&c| quants[c]).collect();
    // The DC frame's global-stream channels also see the two previous
    // same-shaped channels (decoder reference properties, full scale).
    let full_scale: Vec<Vec<i32>> = if dc_frame {
        channels[..split]
            .iter()
            .zip(quants)
            .map(|(ch, &q)| ch.data.iter().map(|&v| v * q as i32).collect())
            .collect()
    } else {
        Vec::new()
    };
    let refs_of = |c: usize| -> Vec<MaRefPlane<'_>> {
        if full_scale.is_empty() {
            return Vec::new();
        }
        let ch = &channels[c];
        (0..c)
            .rev()
            .filter(|&j| {
                let o = &channels[j];
                (o.w, o.h, o.hshift, o.vshift) == (ch.w, ch.h, ch.hshift, ch.vshift)
            })
            .take(MA_REF_CHANNELS)
            .map(|j| MaRefPlane {
                pixels: &full_scale[j],
                stride: channels[j].w,
                x0: 0,
                y0: 0,
            })
            .collect()
    };

    let tag_keys: Vec<Option<(usize, usize)>> = tag_channel
        .iter()
        .enumerate()
        .map(|(t, &c)| {
            let stream = if t < split {
                0
            } else if t < ac_tag_base {
                1
            } else {
                2
            };
            cache
                .as_deref()
                .and_then(|cache| cache.key(quantized, c, stream))
        })
        .collect();
    let mut cached: Vec<Option<LearnedTree>> = tag_keys
        .iter()
        .map(|key| key.and_then(|(c, stream)| cache.as_deref()?.trees[c][stream].clone()))
        .collect();
    let sample_tag = |t: usize| {
        let ch = &channels[tag_channel[t]];
        let px = ch.w * ch.h;
        let mut stride = px.div_ceil(LM_SAMPLE_TARGET).max(1);
        if stride > 1 && stride.is_multiple_of(2) {
            stride += 1;
        }
        let chan = if t < split {
            t
        } else if t < ac_tag_base {
            t - dc_tag_base
        } else {
            t - ac_tag_base
        } as u32;
        let mut samples = MaSamples::with_capacity(px / stride + 1);
        let data = &ch.data;
        let w = ch.w;
        let refs = if t < split { refs_of(t) } else { Vec::new() };
        sample_channel_ma(
            |y| &data[y * w..][..w],
            ch.w,
            ch.h,
            chan,
            0,
            &refs,
            WpParams::DEFAULT,
            false,
            stride,
            |props, tok, _| samples.push(props, tok),
        );
        scale_ma_value_props(&mut samples, tag_quants[t] as i32);
        samples
    };
    let learned = {
        // Unchanged fine tags retain their trees. Generate their samples only
        // if the joint context budget later requires re-learning one of them.
        let mut tag_samples: Vec<Option<MaSamples>> =
            pool.steal_map(scratch, n_tags, |t, _scratch| {
                if cached[t].is_some() {
                    None
                } else {
                    Some(sample_tag(t))
                }
            });

        // Learn one subtree per tag (sequential: the learner parallelizes
        // internally). Leaf budget follows the tag's pixel share; a learned tree
        // must beat its own flat estimate plus serialization/context overhead or
        // it collapses to a single leaf. Budgets are jointly rescaled so the
        // total leaf count can never exceed the LZ77 context bound (large images
        // at low distance otherwise overflow it).
        let budgets: Vec<usize> = (0..n_tags)
            .map(|t| {
                let ch = &channels[tag_channel[t]];
                ((ch.w * ch.h) / LM_BUDGET_DIV).clamp(1, LM_BUDGET_CAP)
            })
            .collect();
        // Large images at low distance can push the SUM of actual leaves past the
        // LZ77 context bound; only then scale the budgets down and re-learn the
        // offenders (budgets are upper bounds — most trees collapse well below
        // them, so up-front rescaling would perturb every ordinary encode).
        let budget_cap = LM_MAX_CONTEXTS - 64;
        let mut learned: Vec<LearnedTree> = Vec::with_capacity(n_tags);
        for t in 0..n_tags {
            if let Some(tree) = cached[t].take() {
                learned.push(tree);
                continue;
            }
            let ch = &channels[tag_channel[t]];
            let px = ch.w * ch.h;
            let mut stride = px.div_ceil(LM_SAMPLE_TARGET).max(1);
            if stride > 1 && stride.is_multiple_of(2) {
                stride += 1;
            }
            let params = |max_leaves: usize| MaLearnParams {
                alphabet: min_symbol as usize,
                max_leaves,
                split_cost_bits: MA_SPLIT_COST_BITS / stride as f32,
                min_node: MA_MIN_NODE_SAMPLES,
                allow_wp: false,
                allowed_preds: LM_SAFE_PRED_MASK,
                side_preds: 1,
                max_candidates: MA_PHOTO_CANDIDATES,
                modal_offsets: false,
                prune_leaf_bits: 0.0,
            };
            let samples = tag_samples[t].as_ref().expect("uncached tag was sampled");
            if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
                learned.push(learn_ma_tree(samples, params(1), pool, scratch));
                continue;
            }
            let budget = budgets[t];
            let tree = learn_ma_tree(samples, params(budget), pool, scratch);
            let n_leaves = tree.nodes.len().div_ceil(2);
            let overhead_bits = tree.nodes.len() as f64 * 10.0 + n_leaves as f64 * 200.0;
            let est_real = tree.est_bits * stride as f64 + overhead_bits;
            let flat_real = tree.flat_bits * stride as f64;
            if est_real < flat_real {
                learned.push(tree);
            } else {
                learned.push(learn_ma_tree(samples, params(1), pool, scratch));
            }
        }
        let leaves_of = |t: &LearnedTree| t.nodes.len().div_ceil(2);
        let mut total_leaves: usize = learned.iter().map(leaves_of).sum();
        while total_leaves > budget_cap {
            let (worst, _) = learned
                .iter()
                .enumerate()
                .max_by_key(|(_, t)| leaves_of(t))
                .unwrap();
            let target = (leaves_of(&learned[worst]) / 2).max(1);
            let ch = &channels[tag_channel[worst]];
            let px = ch.w * ch.h;
            let mut stride = px.div_ceil(LM_SAMPLE_TARGET).max(1);
            if stride > 1 && stride.is_multiple_of(2) {
                stride += 1;
            }
            let params = MaLearnParams {
                alphabet: min_symbol as usize,
                max_leaves: target,
                split_cost_bits: MA_SPLIT_COST_BITS / stride as f32,
                min_node: MA_MIN_NODE_SAMPLES,
                allow_wp: false,
                allowed_preds: LM_SAFE_PRED_MASK,
                side_preds: 1,
                max_candidates: MA_PHOTO_CANDIDATES,
                modal_offsets: false,
                prune_leaf_bits: 0.0,
            };
            let samples = tag_samples[worst].get_or_insert_with(|| sample_tag(worst));
            learned[worst] = learn_ma_tree(samples, params, pool, scratch);
            total_leaves = learned.iter().map(leaves_of).sum();
        }
        if let Some(cache) = cache.as_deref_mut() {
            for (tree, key) in learned.iter().zip(&tag_keys) {
                if let Some(key) = *key {
                    cache.update_tree(key, tree);
                }
            }
        }
        learned
    };

    let (tree_tokens, leaf_ctx, tag_base, num_ctx) =
        emit_lm_forest(&learned, &tag_quants, split, n_dc_slots, num_dc_groups);
    let num_ctx = num_ctx as usize;
    let distance_ctx = num_ctx as u32;

    if let Some(cache) = cache.as_deref_mut() {
        let mut tag_pixels = vec![0usize; n_tags];
        for c in 0..split {
            tag_pixels[c] = channels[c].data.len();
        }
        for (gdim, xgroups, groups, minsh, maxsh, base) in [
            (
                LF_GROUP_DIM,
                xsize_dc_groups,
                num_dc_groups,
                3,
                1000,
                dc_tag_base,
            ),
            (GROUP_DIM, xsize_groups, num_ac_groups, 0, 2, ac_tag_base),
        ] {
            for group in 0..groups {
                for_each_squeeze_group_crop(
                    channels,
                    split,
                    gdim,
                    group % xgroups,
                    group / xgroups,
                    minsh,
                    maxsh,
                    |within, _c, _x, _y, width, height| tag_pixels[base + within] += width * height,
                );
            }
        }
        for (t, key) in tag_keys.iter().enumerate() {
            if let Some(key) = *key {
                let groups = [1, num_dc_groups, num_ac_groups][key.1];
                cache.reserve_tokens(key, tag_pixels[t], groups);
            }
        }
    }
    // Local leaf-node ids survive forest routing/context renumbering. The
    // stored value is the signed residual, independent of its entropy code.
    let local_leaf_ctx: Vec<Vec<u32>> = learned
        .iter()
        .map(|tree| (0..tree.nodes.len() as u32).collect())
        .collect();
    let (global_lz, dc_group_lz, ac_group_lz, pending) = {
        let cache_ref = cache.as_deref();
        let tokenize_block = |tag: usize,
                              channel: usize,
                              chan: u32,
                              x0: usize,
                              y0: usize,
                              width: usize,
                              height: usize,
                              group: usize,
                              refs: &[MaRefPlane<'_>],
                              out: &mut Vec<LzToken>,
                              pending: &mut Vec<LmPendingTokens>| {
            let key = tag_keys[tag];
            let append = |tokens: &[Token], out: &mut Vec<LzToken>| {
                let base = tag_base[tag] as usize;
                out.extend(tokens.iter().map(|token| {
                    Token::new(leaf_ctx[base + token.context as usize], token.value).as_lz()
                }));
            };
            if let Some(tokens) = key.and_then(|key| cache_ref?.block(key, group)) {
                append(tokens, out);
                return;
            }
            let ch = &channels[channel];
            let mut tokens = Vec::with_capacity(width * height);
            tokenize_channel_ma_mult(
                |y| &ch.data[(y0 + y) * ch.w + x0..][..width],
                width,
                height,
                chan,
                0,
                tag_quants[tag] as i32,
                &learned[tag],
                &local_leaf_ctx[tag],
                0,
                refs,
                &mut tokens,
            );
            append(&tokens, out);
            if let Some(key) = key.filter(|&key| cache_ref.is_some_and(|cache| cache.admitted(key)))
            {
                pending.push(LmPendingTokens { key, group, tokens });
            }
        };
        let global_channels = pool.steal_map(scratch, split, |c, _scratch| {
            let ch = &channels[c];
            let mut out = Vec::with_capacity(ch.w * ch.h);
            let mut pending = Vec::new();
            tokenize_block(
                c,
                c,
                c as u32,
                0,
                0,
                ch.w,
                ch.h,
                0,
                &refs_of(c),
                &mut out,
                &mut pending,
            );
            (out, pending)
        });
        let mut global_lz = Vec::new();
        let mut pending = Vec::new();
        for (tokens, blocks) in global_channels {
            global_lz.extend(tokens);
            pending.extend(blocks);
        }
        // Literal-only: shared per-pixel contexts make zero literals near
        // free. Cached blocks only remap their local leaf ids into this forest.
        let crop_group = |gdim: usize,
                          gx: usize,
                          gy: usize,
                          minsh: i32,
                          maxsh: i32,
                          slot_tag_base: usize,
                          group: usize| {
            let mut out = Vec::new();
            let mut pending = Vec::new();
            for_each_squeeze_group_crop(
                channels,
                split,
                gdim,
                gx,
                gy,
                minsh,
                maxsh,
                |within, channel, x0, y0, width, height| {
                    tokenize_block(
                        slot_tag_base + within,
                        channel,
                        within as u32,
                        x0,
                        y0,
                        width,
                        height,
                        group,
                        &[],
                        &mut out,
                        &mut pending,
                    );
                },
            );
            (out, pending)
        };
        let dc_groups = pool.steal_map_with_threads(
            scratch,
            num_dc_groups,
            group_lz_threads(speed, pool),
            |group, _scratch| {
                crop_group(
                    LF_GROUP_DIM,
                    group % xsize_dc_groups,
                    group / xsize_dc_groups,
                    3,
                    1000,
                    dc_tag_base,
                    group,
                )
            },
        );
        let ac_groups = pool.steal_map_with_threads(
            scratch,
            num_ac_groups,
            group_lz_threads(speed, pool),
            |group, _scratch| {
                crop_group(
                    GROUP_DIM,
                    group % xsize_groups,
                    group / xsize_groups,
                    0,
                    2,
                    ac_tag_base,
                    group,
                )
            },
        );
        let mut dc_group_lz = Vec::with_capacity(num_dc_groups);
        for (tokens, blocks) in dc_groups {
            dc_group_lz.push(tokens);
            pending.extend(blocks);
        }
        let mut ac_group_lz = Vec::with_capacity(num_ac_groups);
        for (tokens, blocks) in ac_groups {
            ac_group_lz.push(tokens);
            pending.extend(blocks);
        }
        (global_lz, dc_group_lz, ac_group_lz, pending)
    };
    if let Some(cache) = cache {
        for block in pending {
            cache.retain(block);
        }
    }

    let code = build_lz_pixel_code_opts(
        speed.effort().lz_exhaustive_entropy(),
        std::iter::once(global_lz.as_slice())
            .chain(dc_group_lz.iter().map(Vec::as_slice))
            .chain(ac_group_lz.iter().map(Vec::as_slice)),
        num_ctx,
        min_symbol,
        speed.effort().lz_refined_entropy,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    if dc_frame {
        write_frame_header_modular_xyb_dc(writer);
    } else {
        layout.write_frame_header(epf, writer);
    }

    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // ----- Section 0: LfGlobal = global tree + global modular image -----
    layout.write_dc_quant(&mut sections[0]);
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(
        &tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree = 1
    sections[0].write(1, 1); // wp_default = 1
    layout.write_transforms(steps, &mut sections[0]);
    write_lz_section(
        &global_lz,
        distance_ctx,
        &code,
        min_symbol,
        &mut sections[0],
    );
    sections[0].zero_pad_to_byte();

    // ----- DC groups -----
    write_lz_groups(
        &dc_group_lz,
        &[],
        &code,
        distance_ctx,
        min_symbol,
        WpParams::DEFAULT,
        pool,
        &mut sections[1..num_dc_groups + 1],
    );

    // ----- AC global: trivial -----
    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    // ----- AC groups -----
    write_lz_groups(
        &ac_group_lz,
        &[],
        &code,
        distance_ctx,
        min_symbol,
        WpParams::DEFAULT,
        pool,
        &mut sections[2 + num_dc_groups..],
    );

    if num_ac_groups == 1 {
        // One TOC entry: every channel lives in the LfGlobal stream and the
        // group streams are empty, so the decoder reads nothing past it.
        debug_assert!(dc_group_lz.iter().chain(&ac_group_lz).all(Vec::is_empty));
        sections.truncate(1);
    }
    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
}

#[cfg(test)]
mod representation_race_tests {
    use super::*;
    use crate::coding::CodingTransform;

    #[test]
    fn race_preserves_prefix_and_reports_original_size_at_every_alignment() {
        let pool = ThreadPool::new(1);
        let mut scratch = CoderScratch::default();
        for (w, h, flat) in [
            (37usize, 29usize, false),
            (513, 17, false),
            (128, 128, true),
        ] {
            let input = LmPlainInput {
                color: (0..3)
                    .map(|c| {
                        (0..w * h)
                            .map(|i| {
                                if flat {
                                    (12000 + c * 8000) as i32
                                } else {
                                    ((i * 523 + (i / w) * 311 + c * 997) % 65536) as i32
                                }
                            })
                            .collect()
                    })
                    .collect(),
                extra: Vec::new(),
                max: 65535.0,
                coding: CodingTransform::Rgb,
            };
            for alignment in 0..8 {
                let quantized = lm_quantize_plain(&input, w, h, 6.0).unwrap();
                if flat {
                    // Ensure the alignment test reaches the full trial as
                    // well as the coarse screen's direct-encoding fallback.
                    assert_eq!(
                        super::super::fold_cost::select_fold(
                            &quantized,
                            crate::Speed::Slow,
                            &mut scratch,
                        ),
                        Some(4),
                    );
                }
                let mut prefix = BitWriter::new();
                prefix.write(8, 0xa5);
                prefix.write(alignment, (1u64 << alignment) - 1);
                let prefix_bits = prefix.bits_written();
                let mut original = prefix.clone();
                encode_lm_frame_impl(
                    &quantized,
                    crate::Speed::Slow,
                    &pool,
                    &mut scratch,
                    &mut original,
                    None,
                );
                let mut raced = prefix;
                let baseline_bits = encode_lm_frame(
                    quantized,
                    crate::Speed::Slow,
                    &pool,
                    &mut scratch,
                    &mut raced,
                )
                .unwrap();
                assert_eq!(baseline_bits, original.bits_written() - prefix_bits);
                assert!(raced.bits_written() <= original.bits_written());
                let bytes = raced.into_bytes();
                assert_eq!(bytes[0], 0xa5);
                let mask = (1u8 << alignment) - 1;
                assert_eq!(bytes[1] & mask, mask);
            }
        }
    }

    #[test]
    fn cached_local_tokens_match_fresh_encoding_after_forest_changes() {
        let (w, h) = (513, 17);
        let input = LmPlainInput {
            color: (0..3)
                .map(|c| {
                    (0..w * h)
                        .map(|i| ((i * 523 + (i / w) * 311 + c * 997) % 65536) as i32)
                        .collect()
                })
                .collect(),
            extra: Vec::new(),
            max: 65535.0,
            coding: CodingTransform::Rgb,
        };
        let mut quantized = lm_quantize_plain(&input, w, h, 6.0).unwrap();
        let pool = ThreadPool::new(1);
        let mut scratch = CoderScratch::default();
        let mut cache = LmTreeCache::new(&quantized);
        encode_lm_frame_impl(
            &quantized,
            crate::Speed::Slow,
            &pool,
            &mut scratch,
            &mut BitWriter::new(),
            Some(&mut cache),
        );
        assert!(cache.token_bytes > 0 && cache.token_bytes <= LM_TOKEN_CACHE_BYTES);
        for _ in 0..4 {
            assert!(quantized.fold_last());
        }
        let mut reused = BitWriter::new();
        encode_lm_frame_impl(
            &quantized,
            crate::Speed::Slow,
            &pool,
            &mut scratch,
            &mut reused,
            Some(&mut cache),
        );
        let mut fresh = BitWriter::new();
        encode_lm_frame_impl(
            &quantized,
            crate::Speed::Slow,
            &pool,
            &mut scratch,
            &mut fresh,
            None,
        );
        assert_eq!(reused.into_bytes(), fresh.into_bytes());
    }

    #[test]
    fn token_blocks_are_invalidated_when_their_tree_changes() {
        use crate::ma_tree::MaNode;
        let mut cache = LmTreeCache {
            original_len: 1,
            averages: 0,
            trees: vec![[None, None, None]],
            tokens: vec![std::array::from_fn(|_| LmCachedTokens::default())],
            token_bytes: 0,
        };
        let mut tree = LearnedTree {
            nodes: vec![MaNode::Leaf { pred: 0, offset: 0 }],
            est_bits: 0.0,
            flat_bits: 0.0,
        };
        let key = (0, 2);
        cache.update_tree(key, &tree);
        cache.reserve_tokens(key, 2, 1);
        cache.retain(LmPendingTokens {
            key,
            group: 0,
            tokens: vec![Token::new(0, 17), Token::new(0, 23)],
        });
        tree.est_bits = 123.0;
        cache.update_tree(key, &tree);
        assert_eq!(cache.block(key, 0).unwrap()[1].value, 23);
        tree.nodes[0] = MaNode::Leaf { pred: 7, offset: 0 };
        cache.update_tree(key, &tree);
        assert!(cache.block(key, 0).is_none());
        assert_eq!(cache.token_bytes, 0);
        cache.reserve_tokens(
            key,
            LM_TOKEN_CACHE_BYTES / std::mem::size_of::<Token>() + 1,
            1,
        );
        assert!(!cache.admitted(key));
        assert_eq!(cache.token_bytes, 0);
    }
}
