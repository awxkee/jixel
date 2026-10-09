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

//! Progressive lossless and quantized lossy modular Squeeze encoding.

use super::lz77::{
    LZ77_MIN_SYMBOL, LzToken, build_lz_pixel_code, build_lz_pixel_code_opts,
    lz77_compress_for_speed, lz77_compress_for_speed_with_depth, lz77_literals,
    write_local_tree_lz77, write_lz_section, write_tree_lz77,
};
use super::predictor::{
    GradPackInteriorFn, PREDICTOR_GRADIENT, SqueezePredictorCost, channel_to_context,
    choose_predictor_for_plane, fixed_predictor, predictor_value, selected_grad_pack_interior_fn,
    tokenize_plane_rows,
};
use super::rct::write_rct_transform;
use super::{
    DeepLzScratchPool, GROUP_DIM, GroupLayout, LF_GROUP_DIM, MA_MIN_NODE_SAMPLES,
    MA_PHOTO_CANDIDATES, MA_SPLIT_COST_BITS, build_balanced_tree_tokens_mul, group_lz_threads,
    push_leaf_mul, push_split, sample_channel_ma, walk_channel_ma, write_frame_header_modular,
    write_lz_groups, write_toc_entry, write_u64,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS};

/// Lossy-modular context budget: the pre-4096 scratch bound, kept so the
/// lossless leaf cap can grow without perturbing this path's trees.
const LM_MAX_CONTEXTS: usize = 1024;
const _: () = assert!(LM_MAX_CONTEXTS <= LZ77_MAX_CONTEXTS);
use crate::encode_image::AlphaPlane;
use crate::entropy::{Token, pack_signed};
use crate::image::Image3Si;
use crate::ma_tree::{LearnedTree, MaLearnParams, MaNode, MaSamples, learn_ma_tree};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Transform list for RCT + a Squeeze with the given step sequence.
/// Byte-exact to libjxl Transform/SqueezeParams field encoding.
fn write_modular_transforms_rct_squeeze(steps: &[crate::squeeze::SqueezeStep], w: &mut BitWriter) {
    // transforms count = 2 -> selector 2 + Bits(4)=0.
    w.write(2, 0b10);
    w.write(4, 0);
    write_rct_transform(6, w);
    write_squeeze_transform_entry(steps, w);
}

/// Transform list holding a single Squeeze (XYB channels are already
/// decorrelated, so no RCT precedes it).
fn write_modular_transforms_squeeze_only(steps: &[crate::squeeze::SqueezeStep], w: &mut BitWriter) {
    w.write(2, 0b01); // transforms count = 1 (Val(1))
    write_squeeze_transform_entry(steps, w);
}

/// One Squeeze transform entry with an explicit step sequence.
fn write_squeeze_transform_entry(steps: &[crate::squeeze::SqueezeStep], w: &mut BitWriter) {
    w.write(2, 0b10); // id = kSqueeze
    // num_squeezes: U32(Val0, BitsOffset(4,1), BitsOffset(6,9), BitsOffset(8,41)).
    let n = steps.len() as u32;
    if n == 0 {
        w.write(2, 0b00);
    } else if n <= 16 {
        w.write(2, 0b01);
        w.write(4, (n - 1) as u64);
    } else if n <= 72 {
        w.write(2, 0b10);
        w.write(6, (n - 9) as u64);
    } else {
        w.write(2, 0b11);
        w.write(8, (n - 41) as u64);
    }
    for s in steps {
        w.write(1, if s.horizontal { 1 } else { 0 });
        w.write(1, if s.in_place { 1 } else { 0 });
        // begin_c: U32(Bits3, ...): for small values use selector 0 = Bits(3).
        debug_assert!(s.begin_c < 8);
        w.write(2, 0b00);
        w.write(3, s.begin_c as u64);
        // num_c: U32(Val1,Val2,Val3,BitsOffset(4,4)).
        match s.num_c {
            1 => w.write(2, 0b00),
            2 => w.write(2, 0b01),
            3 => w.write(2, 0b10),
            n => {
                w.write(2, 0b11);
                w.write(4, (n - 4) as u64);
            }
        }
    }
}

/// Progressive-lossless single-group path. Applies the alternating Squeeze
/// pyramid to the RCT'd channels, then tokenizes the resulting channels with
/// fixed Weighted prediction in Fast mode or adaptive prediction in Slow mode.
/// The decoder reconstructs the input exactly through inverse-Squeeze + inverse-RCT.
pub(super) fn encode_squeeze_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    use_wp: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
    // Lift the 3 RCT'd planes (and alpha, if present) into channels.
    let num_c = if alpha.is_some() { 4 } else { 3 };
    let mut channels: Vec<Channel> = Vec::with_capacity(num_c);
    for c in 0..3usize {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            let row = linear.plane_row(c, y);
            ch.data[y * xsize..y * xsize + xsize].copy_from_slice(&row[..xsize]);
        }
        channels.push(ch);
    }
    if let Some(a) = alpha {
        let mut ch = Channel::new(xsize, ysize);
        match a {
            AlphaPlane::U8(data) => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    for (dst, &src) in row[..xsize].iter_mut().zip(src_row.iter()) {
                        *dst = src as i32;
                    }
                }
            }
            AlphaPlane::U16 { data, bits: _ } => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    for (dst, &src) in row[..xsize].iter_mut().zip(src_row.iter()) {
                        *dst = src as i32;
                    }
                }
            }
            AlphaPlane::F32(data) => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    row[..xsize].copy_from_slice(&src_row[..xsize]);
                }
            }
        }
        channels.push(ch);
    }
    // Apply the alternating H/V pyramid over all channels (explicit sequence).
    let steps = default_squeeze_steps(xsize, ysize, num_c);
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();

    // Slow searches the useful general and directional predictors per channel;
    // Fast avoids the analysis pass and uses fixed Weighted prediction.
    let predictors: Vec<u32> = if speed.effort().predictor_search {
        pool.steal_map(scratch, channels.len(), |c, _scratch| {
            let ch = &channels[c];
            let data = &ch.data;
            let w = ch.w;
            let get = move |gx: usize, gy: usize| data[gy * w + gx];
            choose_predictor_for_plane(get, ch.w, ch.h, use_wp)
        })
    } else {
        vec![fixed_predictor(use_wp); nb]
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

    write_frame_header_modular(alpha.is_some(), GroupLayout::DEFAULT, writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut section);

    let distance_ctx = nb as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy,
        std::iter::once(lz_tokens.as_slice()),
        nb,
        min_symbol,
        speed.effort().lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_local_tree_lz77(
        &predictors,
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
}

#[allow(clippy::too_many_arguments)]
fn for_each_squeeze_group_crop(
    channels: &[crate::squeeze::Channel],
    split: usize,
    gdim: usize,
    gx: usize,
    gy: usize,
    minsh: i32,
    maxsh: i32,
    mut visit: impl FnMut(usize, usize, usize, usize, usize, usize),
) {
    let mut within = 0usize;
    for (c, ch) in channels.iter().enumerate().skip(split) {
        let msh = ch.hshift.min(ch.vshift);
        if msh < minsh || msh > maxsh {
            continue;
        }
        let hs = ch.hshift as usize;
        let vs = ch.vshift as usize;
        let rx0 = (gx * gdim) >> hs;
        let ry0 = (gy * gdim) >> vs;
        if rx0 >= ch.w || ry0 >= ch.h {
            continue;
        }
        let rw = (gdim >> hs).min(ch.w - rx0);
        let rh = (gdim >> vs).min(ch.h - ry0);
        if rw == 0 || rh == 0 {
            continue;
        }
        visit(within, c, rx0, ry0, rw, rh);
        within += 1;
    }
}

/// Stage-2 progressive lossless, multi-group (image > one AC group).
///
/// Applies the Squeeze pyramid to the whole frame, then splits channels per the
/// JXL rule: channels that fit a group form the global modular image (LfGlobal /
/// section 0); larger channels are partitioned into AC groups, each carrying the
/// `frame_rect >> shift` crop of every large channel. A single global tree (chain
/// on the channel property) serves all streams; the within-group channel index is
/// what the tree sees, so group crops share contexts with the leading globals.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_squeeze_multigroup(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    xsize_groups: usize,
    ysize_groups: usize,
    xsize_dc_groups: usize,
    ysize_dc_groups: usize,
    num_dc_groups: usize,
    num_ac_groups: usize,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    use_wp: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
    let num_c = if alpha.is_some() { 4 } else { 3 };
    let mut channels: Vec<Channel> = Vec::with_capacity(num_c);
    for c in 0..3usize {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            let row = linear.plane_row(c, y);
            ch.data[y * xsize..y * xsize + xsize].copy_from_slice(&row[..xsize]);
        }
        channels.push(ch);
    }
    if let Some(a) = alpha {
        let mut ch = Channel::new(xsize, ysize);
        for (i, value) in ch.data.iter_mut().enumerate() {
            *value = a.get_i32(i);
        }
        channels.push(ch);
    }
    let steps = default_squeeze_steps(xsize, ysize, num_c);
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();

    // Channel -> stream split (JXL rule): the leading run of channels that fit a
    // group is the global modular image; the first channel still larger than a
    // group starts the suffix that is partitioned into AC groups.
    let split = channels
        .iter()
        .position(|c| c.w > GROUP_DIM || c.h > GROUP_DIM)
        .unwrap_or(nb);

    let predictors: Vec<u32> = if speed.effort().predictor_search {
        // The MA tree sees a channel's position inside each modular sub-image,
        // not its index in `channels`. Pool costs by that decoder-visible slot.
        let mut costs: Vec<SqueezePredictorCost> =
            (0..nb).map(|_| SqueezePredictorCost::default()).collect();
        for (cost, ch) in costs[..split].iter_mut().zip(&channels[..split]) {
            cost.add_crop(|x, y| ch.data[y * ch.w + x], ch.w, ch.h, use_wp);
        }
        for gy in 0..ysize_dc_groups {
            for gx in 0..xsize_dc_groups {
                for_each_squeeze_group_crop(
                    &channels,
                    split,
                    LF_GROUP_DIM,
                    gx,
                    gy,
                    3,
                    1000,
                    |within, c, x0, y0, w, h| {
                        let ch = &channels[c];
                        costs[within].add_crop(
                            |x, y| ch.data[(y0 + y) * ch.w + x0 + x],
                            w,
                            h,
                            use_wp,
                        );
                    },
                );
            }
        }
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                for_each_squeeze_group_crop(
                    &channels,
                    split,
                    GROUP_DIM,
                    gx,
                    gy,
                    0,
                    2,
                    |within, c, x0, y0, w, h| {
                        let ch = &channels[c];
                        costs[within].add_crop(
                            |x, y| ch.data[(y0 + y) * ch.w + x0 + x],
                            w,
                            h,
                            use_wp,
                        );
                    },
                );
            }
        }
        costs.iter().map(|c| c.predictor(use_wp)).collect()
    } else {
        vec![fixed_predictor(use_wp); nb]
    };

    let distance_ctx = nb as u32;

    // Global stream: the small channels [0, split), whole.
    let global_channel_tokens = pool.steal_map(scratch, split, |c, scratch| {
        let ch = &channels[c];
        let ctx = channel_to_context(c, nb);
        let mut tokens = Vec::with_capacity(ch.w * ch.h);
        tokenize_plane_rows(
            ctx,
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
    let mut global_tokens: Vec<Token> = Vec::new();
    for channel in global_channel_tokens {
        global_tokens.extend(channel);
    }
    let global_lz = lz77_compress_for_speed(&global_tokens, distance_ctx, speed, scratch);

    // One group's worth of cropped large-channel tokens. `gdim` is the group's
    // frame-space size (GROUP_DIM for AC, LF_GROUP_DIM for DC); a channel is
    // included when min(hshift,vshift) is inside [minsh,maxsh], and contributes
    // its `(group_rect >> shift)` crop. The decoder rebuilds the same scan, so
    // the within-group index (sequential over non-empty crops) is exactly the
    // `chan` property the global tree keys on.
    let (dc_group_lz, ac_group_lz) = {
        let deep_lz = speed
            .effort()
            .lz_deep
            .then(|| DeepLzScratchPool::new(group_lz_threads(speed, pool)));
        let crop_group = |gdim: usize,
                          gx: usize,
                          gy: usize,
                          minsh: i32,
                          maxsh: i32,
                          scratch: &mut CoderScratch|
         -> Vec<LzToken> {
            let mut gtok: Vec<Token> = Vec::new();
            for_each_squeeze_group_crop(
                &channels,
                split,
                gdim,
                gx,
                gy,
                minsh,
                maxsh,
                |within, c, rx0, ry0, rw, rh| {
                    let ch = &channels[c];
                    let ctx = channel_to_context(within, nb);
                    let pred = predictors[within];
                    tokenize_plane_rows(
                        ctx,
                        |y| &ch.data[(ry0 + y) * ch.w + rx0..][..rw],
                        rw,
                        rh,
                        pred,
                        grad_pack_fn,
                        &mut scratch.gradient,
                        &mut gtok,
                    );
                },
            );
            if let Some(deep_lz) = &deep_lz {
                deep_lz.with_depth(|depth| {
                    lz77_compress_for_speed_with_depth(
                        &gtok,
                        distance_ctx,
                        speed,
                        depth,
                        None,
                        scratch,
                    )
                })
            } else {
                lz77_compress_for_speed(&gtok, distance_ctx, speed, scratch)
            }
        };

        // DC (LF) groups carry the deeply-squeezed large channels (min shift >= 3),
        // partitioned into LF_GROUP_DIM rects. Empty unless the image is large enough
        // that a >=3x-squeezed channel still exceeds a group (dimension > ~2048).
        let dc_group_lz = pool.steal_map_with_threads(
            scratch,
            num_dc_groups,
            group_lz_threads(speed, pool),
            |group_index, scratch| {
                let gx = group_index % xsize_dc_groups;
                let gy = group_index / xsize_dc_groups;
                crop_group(LF_GROUP_DIM, gx, gy, 3, 1000, scratch)
            },
        );

        // AC groups carry the shallow large channels (min shift <= 2), in GROUP_DIM rects.
        let ac_group_lz = pool.steal_map_with_threads(
            scratch,
            num_ac_groups,
            group_lz_threads(speed, pool),
            |group_index, scratch| {
                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                crop_group(GROUP_DIM, gx, gy, 0, 2, scratch)
            },
        );
        (dc_group_lz, ac_group_lz)
    };

    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy,
        std::iter::once(global_lz.as_slice())
            .chain(dc_group_lz.iter().map(Vec::as_slice))
            .chain(ac_group_lz.iter().map(Vec::as_slice)),
        nb,
        min_symbol,
        speed.effort().lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    write_frame_header_modular(alpha.is_some(), GroupLayout::DEFAULT, writer);

    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // ----- Section 0: LfGlobal = global tree + global modular image -----
    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_local_tree_lz77(
        &predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree = 1
    sections[0].write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut sections[0]);
    write_lz_section(
        &global_lz,
        distance_ctx,
        &code,
        min_symbol,
        &mut sections[0],
    );
    sections[0].zero_pad_to_byte();

    // ----- DC groups: GroupHeader + any min-shift>=3 large-channel crops -----
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

    // ----- AC groups: GroupHeader + the cropped large-channel tokens -----
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

    // TOC + sections.
    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}

// ===================== Lossy modular (quantized Squeeze) =====================

/// cjxl's per-squeeze-level quantization steps for XYB modular, indexed by the
/// stored-channel color [Y, X, B-Y] and the (capped) squeeze depth. Values are
/// integer lattice steps relative to the `LOSSY_MODULAR_XYB_SCALE` scaling.
static SQUEEZE_XYB_QTABLE: [[f32; 16]; 3] = [
    [
        163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02,
        0.01, 0.005,
    ], // Y
    [
        1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5,
    ], // X
    [
        2048.0, 1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5,
        0.5,
    ], // B-Y
];
static SQUEEZE_QUALITY_FACTOR_XYB: f32 = 4.8;
/// Per-color multipliers on the cjxl base rows (stored order [Y, X, B-Y]),
/// Optuna study lossy_modular_v3 2026-09-02 (129 trials, crops+Kodak,
/// holdout-validated): B-Y much finer — coarse B-Y measurably desaturates
/// yellows (mean +2/255 blue shift) and SS2 punishes it; X detail is cheap to
/// discard; Y slightly coarser pays for the chroma.
static LM_ROW_MUL: [f32; 3] = [1.265, 1.704, 0.546];
/// XYB float -> integer scale factors in X, Y, B order. Signaled to the decoder
/// as custom dc_quant steps (their reciprocals); powers of two stay F16-exact.
static LOSSY_MODULAR_XYB_SCALE: [f32; 3] = [65536.0, 4096.0, 4096.0];

/// Quantization step for one squeezed channel (cjxl's schedule, including the
/// float->int truncation).
fn lossy_modular_q(color: usize, hshift: i32, vshift: i32, distance: f32) -> i32 {
    let mut shift = (hshift + vshift).min(16);
    if shift > 0 {
        shift -= 1;
    }
    let quantizer = 0.25f32 * distance;
    let q = quantizer
        * SQUEEZE_QUALITY_FACTOR_XYB
        * SQUEEZE_XYB_QTABLE[color][shift as usize]
        * LM_ROW_MUL[color];
    (q as i32).max(1)
}

/// Decoder predictors that are exactly scale-equivariant over an exact
/// division (safe under a leaf multiplier): Zero, Left, Top, Select, Gradient
/// and the plain neighbor predictors. Weighted and the averages are not.
/// Map the caller's distance to the modular arm's quantizer distance so both
/// arms land on comparable SS2 at the same `d` (fitted on the 8-image Kodak
/// pilot; the byte-only auto gate depends on this). Piecewise linear through
/// the measured knots, flat past the last one.
pub(crate) fn lm_calibrated_distance(d: f32) -> f32 {
    static KNOTS: [(f32, f32); 6] = [
        (1.0, 1.324),
        (1.5, 1.851),
        (2.0, 2.259),
        (3.0, 2.466),
        (4.0, 2.504),
        (5.0, 2.350),
    ];
    let knots = KNOTS;
    let k = if d <= knots[0].0 {
        knots[0].1
    } else if d >= knots[knots.len() - 1].0 {
        knots[knots.len() - 1].1
    } else {
        let mut k = knots[0].1;
        for w in knots.array_windows::<2>() {
            let [(d0, k0), (d1, k1)] = [w[0], w[1]];
            if d >= d0 && d <= d1 {
                k = k0 + (k1 - k0) * (d - d0) / (d1 - d0);
                break;
            }
        }
        k
    };
    d * k
}

/// Per-tag MA sample budget; more samples plateaued (<=0.6%) in the 2026-09
/// knob sweeps, as did larger leaf budgets and split-candidate counts.
const LM_SAMPLE_TARGET: usize = 1 << 16;
/// Learned-subtree leaf budget: pixels/div, clamped.
const LM_BUDGET_DIV: usize = 1024;
const LM_BUDGET_CAP: usize = 256;

const LM_SAFE_PRED_MASK: u16 =
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 9);

/// Rescale the value-derived MA properties (libjxl ids 4..=14) by `q`:
/// sampling ran on the divided channel, but the decoder computes properties on
/// the full-scale reconstruction.
fn scale_ma_value_props(samples: &mut MaSamples, q: i32) {
    if q <= 1 {
        return;
    }
    for p in samples.props.iter_mut() {
        for v in &mut p[4..=14] {
            *v *= q;
        }
    }
}

/// Tokenize a (divided) channel rectangle through a learned tree whose splits
/// live in the full-scale property domain. `base` is the tag's node offset in
/// the merged forest arena; `leaf_ctx` is arena-indexed.
#[allow(clippy::too_many_arguments)]
fn tokenize_channel_ma_mult<'a>(
    get_row: impl Fn(usize) -> &'a [i32],
    gw: usize,
    gh: usize,
    chan: u32,
    q: i32,
    tree: &LearnedTree,
    leaf_ctx: &[u32],
    base: u32,
    out: &mut Vec<Token>,
) {
    walk_channel_ma(
        get_row,
        gw,
        gh,
        chan,
        0,
        &[],
        WpParams::DEFAULT,
        false,
        |_x, _y, v, p, n, _wp_pred| {
            let (node, pred) = if q > 1 {
                let mut sp = *p;
                for s in &mut sp[4..=14] {
                    *s *= q;
                }
                tree.lookup(&sp)
            } else {
                tree.lookup(p)
            };
            let pv = predictor_value(pred, n, 0);
            out.push(Token::new(
                leaf_ctx[(base + node) as usize],
                pack_signed((v - pv) as i32),
            ));
        },
    );
}

/// Merge the per-tag learned subtrees under the stream/channel routing
/// skeleton and BFS-emit the whole forest, stamping every leaf with its tag's
/// residual multiplier. The skeleton splits on the static stream-id property:
/// 0 = LfGlobal stream, [1+ndc, 2ndc] = DC-group streams, and AC-group
/// streams above both; each side hangs a channel / slot chain. Tags are
/// ordered [global channels][DC slots][AC slots]. Returns the tree tokens,
/// the arena-indexed leaf context table, each tag's arena base, and the
/// context count.
fn emit_lm_forest(
    trees: &[LearnedTree],
    tag_mults: &[u32],
    split: usize,
    n_dc_slots: usize,
    num_dc_groups: usize,
) -> (Vec<Token>, Vec<u32>, Vec<u32>, u32) {
    use std::collections::VecDeque;
    let n_tags = trees.len();
    let n_ac_slots = n_tags - split - n_dc_slots;
    debug_assert!(split > 0 && n_ac_slots > 0);
    let mut nodes: Vec<MaNode> = Vec::new();
    let mut mult_of: Vec<u32> = Vec::new();
    let mut tag_base = vec![0u32; n_tags];

    let append_tree = |nodes: &mut Vec<MaNode>, mult_of: &mut Vec<u32>, t: usize| -> u32 {
        let base = nodes.len() as u32;
        for n in &trees[t].nodes {
            nodes.push(match *n {
                MaNode::Split { prop, val, gt, le } => MaNode::Split {
                    prop,
                    val,
                    gt: gt + base,
                    le: le + base,
                },
                MaNode::Leaf { pred } => MaNode::Leaf { pred },
            });
            mult_of.push(tag_mults[t]);
        }
        base
    };
    // Per-side chains over property 0 (the decoder-visible channel value:
    // channel index in the global stream, within-group slot in group streams).
    let build_chain = |nodes: &mut Vec<MaNode>,
                       mult_of: &mut Vec<u32>,
                       tag_base: &mut [u32],
                       lo: usize,
                       n: usize|
     -> u32 {
        let base0 = append_tree(nodes, mult_of, lo);
        tag_base[lo] = base0;
        let mut acc = base0;
        for k in 1..n {
            let sub = append_tree(nodes, mult_of, lo + k);
            tag_base[lo + k] = sub;
            nodes.push(MaNode::Split {
                prop: 0,
                val: (k - 1) as i32,
                gt: sub,
                le: acc,
            });
            mult_of.push(1);
            acc = (nodes.len() - 1) as u32;
        }
        acc
    };
    let global_root = build_chain(&mut nodes, &mut mult_of, &mut tag_base, 0, split);
    let ac_root = build_chain(
        &mut nodes,
        &mut mult_of,
        &mut tag_base,
        split + n_dc_slots,
        n_ac_slots,
    );
    let low_root = if n_dc_slots > 0 {
        // DC streams exist: separate them from the global stream below the
        // AC threshold.
        let dc_root = build_chain(&mut nodes, &mut mult_of, &mut tag_base, split, n_dc_slots);
        nodes.push(MaNode::Split {
            prop: 1,
            val: 0,
            gt: dc_root,
            le: global_root,
        });
        mult_of.push(1);
        (nodes.len() - 1) as u32
    } else {
        global_root
    };
    // AC stream ids start at 1 + 3*ndc + kNumQuantTables, always above
    // 2*ndc; global (0) and DC ([1+ndc, 2ndc]) stay at or below it.
    nodes.push(MaNode::Split {
        prop: 1,
        val: (2 * num_dc_groups) as i32,
        gt: ac_root,
        le: low_root,
    });
    mult_of.push(1);
    let root = (nodes.len() - 1) as u32;

    // BFS emission (matches libjxl's FIFO tree decode).
    let mut tokens = Vec::new();
    let mut leaf_ctx = vec![u32::MAX; nodes.len()];
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(root);
    let mut ctx = 0u32;
    while let Some(i) = queue.pop_front() {
        match nodes[i as usize] {
            MaNode::Split { prop, val, gt, le } => {
                push_split(&mut tokens, prop as u32, val);
                queue.push_back(gt);
                queue.push_back(le);
            }
            MaNode::Leaf { pred } => {
                push_leaf_mul(&mut tokens, pred, mult_of[i as usize]);
                leaf_ctx[i as usize] = ctx;
                ctx += 1;
            }
        }
    }
    (tokens, leaf_ctx, tag_base, ctx)
}

/// Quantize every sample to its nearest multiple of `q` (ties away from zero)
/// and store the DIVIDED value. The channel is then coded at 1/q scale and the
/// signaled tree-leaf multiplier `q` restores the scale in the decoder.
fn quantize_channel_divide(ch: &mut crate::squeeze::Channel, q: i32) {
    if q <= 1 {
        return;
    }
    for v in ch.data.iter_mut() {
        let a = *v;
        *v = if a < 0 {
            -((-a + q / 2) / q)
        } else {
            (a + q / 2) / q
        };
    }
}

/// Regular is_last modular frame inside the lossy (`xyb_encoded`) codestream.
/// Identical to `write_frame_header_modular_flags` except the do_YCbCr bit is
/// absent: with xyb_encoded set the color transform is implicitly XYB.
fn write_frame_header_modular_xyb_regular(epf: (u32, f32), w: &mut BitWriter) {
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

/// Decoder-side edge-preserving filter of the lossy-modular frame
/// (iterations, `epf_sigma_for_modular`), sigma linear in the arm's
/// effective distance.
fn lm_epf_schedule(layout: LmLayout, effective_distance: f32) -> (u32, f32) {
    let eff = effective_distance;
    match layout {
        LmLayout::Xyb => (if eff < 5.0 { 1 } else { 2 }, 0.69 + 0.25 * eff),
        LmLayout::Plain { rct: true, .. } => (2, 1.5 + 0.29 * eff),
        LmLayout::Plain { .. } => (if eff < 10.0 { 1 } else { 2 }, 1.6 + 0.055 * eff),
    }
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
fn write_lossy_modular_dc_quant(w: &mut BitWriter) {
    use crate::util::f32_to_f16_bits;
    w.write(1, 0); // dc_quant not all_default
    for scale in LOSSY_MODULAR_XYB_SCALE {
        let step_x128 = 128.0 / scale;
        w.write(16, f32_to_f16_bits(step_x128) as u64);
    }
}

/// How the arm's modular image maps onto the frame: the XYB lattice, or the
/// image's own integer samples (a non-XYB frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LmLayout {
    /// [Y, X, B-Y] on the `LOSSY_MODULAR_XYB_SCALE` lattice, signaled
    /// through custom dc_quant steps.
    Xyb,
    /// The samples in bit-depth units; the decoder divides by the sample
    /// range. `ycbcr` sets the frame's do_YCbCr bit, `rct` codes the three
    /// color channels through the YCoCg RCT, `extra` counts the extra
    /// channels appended after the color ones.
    Plain {
        ycbcr: bool,
        rct: bool,
        extra: usize,
    },
}

impl LmLayout {
    fn write_frame_header(self, epf: (u32, f32), w: &mut BitWriter) {
        match self {
            Self::Xyb => write_frame_header_modular_xyb_regular(epf, w),
            Self::Plain { ycbcr, extra, .. } => {
                write_frame_header_modular_plain(ycbcr, extra, epf, w);
            }
        }
    }

    /// The LfGlobal dc_quant field.
    fn write_dc_quant(self, w: &mut BitWriter) {
        match self {
            Self::Xyb => write_lossy_modular_dc_quant(w),
            // Unused by the decoder outside XYB.
            Self::Plain { .. } => w.write(1, 1),
        }
    }

    fn write_transforms(self, steps: &[crate::squeeze::SqueezeStep], w: &mut BitWriter) {
        match self {
            Self::Plain { rct: true, .. } => write_modular_transforms_rct_squeeze(steps, w),
            _ => write_modular_transforms_squeeze_only(steps, w),
        }
    }
}

/// Regular single-pass modular frame of a non-XYB codestream, no loop
/// filters (same shape as the lossless header plus the YCbCr signaling).
fn write_frame_header_modular_plain(ycbcr: bool, extra: usize, epf: (u32, f32), w: &mut BitWriter) {
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

/// The lossy-modular arm's quantized state: divided channels, per-channel
/// steps, and the squeeze script that produced them.
pub(crate) struct LmQuantized {
    channels: Vec<crate::squeeze::Channel>,
    quants: Vec<u32>,
    steps: Vec<crate::squeeze::SqueezeStep>,
    layout: LmLayout,
    /// Decoder EPF (iterations, modular sigma) for the frame header.
    epf: (u32, f32),
    xsize: usize,
    ysize: usize,
}

/// Quantization role of one channel of a plain-layout pyramid.
#[derive(Clone, Copy, PartialEq)]
enum LmRole {
    Luma,
    Chroma,
    /// Extra channel at its own (luma-row) distance; 0 = lossless.
    Extra(f32),
}

/// Chroma step relative to luma in the plain layouts: Cb/Cr, and the YCoCg
/// differences of RGB (twice their range). Slow Kodak sweeps (01/02/05/20,
/// d 1-4): YCbCr flat over 2-3, RGB flat over 3-6.
fn lm_plain_chroma_mul(coding: crate::coding::CodingTransform) -> f32 {
    match coding {
        crate::coding::CodingTransform::Rgb => 4.0,
        _ => 2.5,
    }
}

/// Map the caller's distance to a plain-layout quantizer distance so the arm
/// lands on the VarDCT arm's SSIMULACRA2 at the same `d` (same purpose as
/// [`lm_calibrated_distance`]; fitted at Slow with the chroma multipliers
/// above and the frame's EPF). Piecewise linear, flat past the ends.
pub(crate) fn lm_plain_distance(coding: crate::coding::CodingTransform, d: f32) -> f32 {
    let knots: &[(f32, f32)] = match coding {
        crate::coding::CodingTransform::Rgb => &[
            (1.0, 1.527),
            (1.5, 2.099),
            (2.0, 2.599),
            (3.0, 2.701),
            (4.0, 2.745),
            (5.0, 2.641),
        ],
        _ => &[
            (1.0, 2.383),
            (1.5, 3.324),
            (2.0, 3.905),
            (3.0, 3.952),
            (4.0, 3.895),
            (5.0, 3.766),
        ],
    };
    d * interpolate_knots(knots, d)
}

/// Piecewise-linear lookup through `knots` (sorted by x), flat outside.
fn interpolate_knots(knots: &[(f32, f32)], x: f32) -> f32 {
    let (first, last) = (knots[0], knots[knots.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    for w in knots.windows(2) {
        let [(x0, y0), (x1, y1)] = [w[0], w[1]];
        if x >= x0 && x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    last.1
}

/// Luma-row quantization step of a channel in bit-depth units: the arm's
/// schedule scaled from the XYB luma lattice to `max`.
fn lossy_modular_q_plain(distance: f32, hshift: i32, vshift: i32, max: f32, mul: f32) -> i32 {
    let mut shift = (hshift + vshift).min(16);
    if shift > 0 {
        shift -= 1;
    }
    let unit = max / LOSSY_MODULAR_XYB_SCALE[1];
    let q = 0.25 * distance * SQUEEZE_QUALITY_FACTOR_XYB * SQUEEZE_XYB_QTABLE[0][shift as usize];
    ((q * unit * mul) as i32).max(1)
}

/// The non-XYB frame's integer planes for the arm: `color` holds one (gray)
/// or three channels in the coded plane order, `extra` the extra channels
/// with their squeeze distances (0 = lossless), all in units of `max`.
pub(crate) struct LmPlainInput {
    pub color: Vec<Vec<i32>>,
    pub extra: Vec<(Vec<i32>, f32)>,
    pub max: f32,
    pub coding: crate::coding::CodingTransform,
}

/// Build and quantize a plain-layout pyramid at `effective_distance`
/// (already calibrated). None when the image is too small to squeeze.
pub(crate) fn lm_quantize_plain(
    input: &LmPlainInput,
    xsize: usize,
    ysize: usize,
    effective_distance: f32,
) -> Option<LmQuantized> {
    use crate::coding::CodingTransform;
    use crate::squeeze::{
        Channel, apply_step_forward, default_squeeze_steps, lossy_squeeze_steps_chroma,
    };
    let num_color = input.color.len();
    debug_assert!(num_color == 1 || num_color == 3);
    let num_c = num_color + input.extra.len();
    if default_squeeze_steps(xsize, ysize, num_c).is_empty() {
        return None;
    }
    let n = xsize * ysize;
    let mut planes: Vec<Vec<i32>> = Vec::with_capacity(num_c);
    let mut roles: Vec<LmRole> = Vec::with_capacity(num_c);
    let (ycbcr, rct, chroma): (bool, bool, Vec<usize>) = match (num_color, input.coding) {
        (3, CodingTransform::YCbCr) => (true, false, vec![0, 2]),
        (3, _) => (false, true, vec![1, 2]),
        _ => (false, false, Vec::new()),
    };
    if rct {
        let [r, g, b] = [&input.color[0], &input.color[1], &input.color[2]];
        let mut y = vec![0i32; n];
        let mut co = vec![0i32; n];
        let mut cg = vec![0i32; n];
        for i in 0..n {
            (y[i], co[i], cg[i]) = crate::lossless::forward_ycocg(r[i], g[i], b[i]);
        }
        planes.extend([y, co, cg]);
    } else {
        planes.extend(input.color.iter().map(|p| p[..n].to_vec()));
    }
    roles.extend((0..num_color).map(|c| {
        if chroma.contains(&c) {
            LmRole::Chroma
        } else {
            LmRole::Luma
        }
    }));
    for (plane, distance) in &input.extra {
        planes.push(plane[..n].to_vec());
        roles.push(LmRole::Extra(*distance));
    }
    let steps = lossy_squeeze_steps_chroma(xsize, ysize, num_c, &chroma);
    let n_pre: usize = steps.iter().filter(|s| !s.in_place).map(|s| s.num_c).sum();
    let mut channels: Vec<Channel> = planes
        .into_iter()
        .map(|data| Channel {
            data,
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        })
        .collect();
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();
    let mut quants: Vec<u32> = Vec::with_capacity(nb);
    for (i, ch) in channels.iter_mut().enumerate() {
        let role = if i >= nb - n_pre {
            LmRole::Chroma
        } else {
            roles[i % num_c]
        };
        let q = match role {
            LmRole::Luma => {
                lossy_modular_q_plain(effective_distance, ch.hshift, ch.vshift, input.max, 1.0)
            }
            LmRole::Chroma => lossy_modular_q_plain(
                effective_distance,
                ch.hshift,
                ch.vshift,
                input.max,
                lm_plain_chroma_mul(input.coding),
            ),
            LmRole::Extra(d) if d > 0.0 => {
                lossy_modular_q_plain(d, ch.hshift, ch.vshift, input.max, 1.0)
            }
            LmRole::Extra(_) => 1,
        };
        quantize_channel_divide(ch, q);
        quants.push(q as u32);
    }
    let layout = LmLayout::Plain {
        ycbcr,
        rct,
        extra: input.extra.len(),
    };
    Some(LmQuantized {
        channels,
        quants,
        steps,
        layout,
        epf: lm_epf_schedule(layout, effective_distance),
        xsize,
        ysize,
    })
}

/// Build and quantize the arm's channel pyramid at `effective_distance`
/// (already calibrated by the caller). None when the image is too small.
pub(crate) fn lm_quantize(
    xyb: &crate::image::Image3F,
    effective_distance: f32,
) -> Option<LmQuantized> {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps, lossy_squeeze_steps};
    let xsize = xyb.xsize();
    let ysize = xyb.ysize();
    let num_c = 3usize;
    if default_squeeze_steps(xsize, ysize, num_c).is_empty() {
        // Tiny image: no squeeze means "lossy" would be plain color
        // quantization at the wrong scale. Not this arm's job.
        return None;
    }
    let steps = lossy_squeeze_steps(xsize, ysize, num_c);
    // With the chroma pre-squeeze, the last four channels are its residuals
    // (mapped to the chroma quantization row, like cjxl); everything before
    // them stays aligned to `num_c` blocks.
    let chroma_pre = steps.len() >= 2 && !steps[0].in_place;

    // XYB floats onto the integer lattice, stored as [Y, X, B-Y].
    let n = xsize * ysize;
    let input = [
        &xyb.plane_data(0)[..n],
        &xyb.plane_data(1)[..n],
        &xyb.plane_data(2)[..n],
    ];
    let mut planes = [vec![0i32; n], vec![0i32; n], vec![0i32; n]];
    {
        let [dst_y, dst_x, dst_b] = &mut planes;
        let scales = [
            LOSSY_MODULAR_XYB_SCALE[0],
            LOSSY_MODULAR_XYB_SCALE[1],
            LOSSY_MODULAR_XYB_SCALE[2],
        ];
        unsafe {
            crate::xyb::selected_quantize_xyb_channels_fn()(input, [dst_y, dst_x, dst_b], scales);
        }
    }
    let mut channels: Vec<Channel> = planes
        .into_iter()
        .map(|data| Channel {
            data,
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        })
        .collect();
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    // Quantize every channel on its (color, depth) schedule. Channels are
    // stored divided by their step; the tree leaf multiplier restores the
    // scale. The decoder's full-scale predictions stay consistent because
    // every predictor used on a q>1 channel is exactly scale-equivariant over
    // the divided values.
    let nb = channels.len();
    let color_of = |i: usize| {
        if chroma_pre && i >= nb - 4 {
            1 // trailing chroma pre-squeeze residuals -> chroma row
        } else {
            i % num_c
        }
    };
    let mut quants: Vec<u32> = Vec::with_capacity(nb);
    for (i, ch) in channels.iter_mut().enumerate() {
        let q = lossy_modular_q(color_of(i), ch.hshift, ch.vshift, effective_distance);
        quantize_channel_divide(ch, q);
        quants.push(q as u32);
    }
    Some(LmQuantized {
        channels,
        quants,
        steps,
        layout: LmLayout::Xyb,
        epf: lm_epf_schedule(LmLayout::Xyb, effective_distance),
        xsize,
        ysize,
    })
}

/// Lossy modular frame: the XYB image on the fixed integer lattice, squeezed,
/// with each pyramid level's residuals rounded to distance-scaled multiples,
/// then coded through the lossless squeeze machinery. `effective_distance` is
/// the arm's already-calibrated quantizer distance (see
/// `lm_calibrated_distance` and the auto gate). Returns false when the frame
/// cannot take this path (caller keeps the VarDCT arm).
pub(crate) fn encode_frame_lossy_modular_squeeze(
    xyb: &crate::image::Image3F,
    effective_distance: f32,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    match lm_quantize(xyb, effective_distance) {
        Some(quantized) => encode_lm_frame(quantized, speed, pool, scratch, writer),
        None => false,
    }
}

/// Lossy modular frame of a non-XYB codestream: the integer planes in
/// `input` squeezed and quantized at the plain-layout schedule (see
/// `lm_plain_distance`). Returns false when the frame cannot take this path.
pub(crate) fn encode_frame_lossy_modular_plain(
    input: &LmPlainInput,
    xsize: usize,
    ysize: usize,
    effective_distance: f32,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    match lm_quantize_plain(input, xsize, ysize, effective_distance) {
        Some(quantized) => encode_lm_frame(quantized, speed, pool, scratch, writer),
        None => false,
    }
}

/// Code one quantized pyramid as a complete modular frame (header, TOC and
/// sections).
fn encode_lm_frame(
    quantized: LmQuantized,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    let LmQuantized {
        channels,
        quants,
        steps,
        layout,
        epf,
        xsize,
        ysize,
    } = quantized;
    let nb = channels.len();

    // Same literal-vs-LZ77 guard as the atlas path, from the actual quantized
    // channel range.
    let max_abs = channels
        .iter()
        .flat_map(|ch| ch.data.iter())
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    let value_bits = 33 - (2 * max_abs).max(1).leading_zeros();
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

    if num_ac_groups == 1 {
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

        layout.write_frame_header(epf, writer);
        let mut section = BitWriter::new();
        layout.write_dc_quant(&mut section);
        section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
        section.write(1, 0); // use_global_tree = 0
        section.write(1, 1); // wp_default = 1
        layout.write_transforms(&steps, &mut section);

        let distance_ctx = nb as u32;
        let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
        let code = build_lz_pixel_code(
            speed.effort().lz_exhaustive_entropy,
            std::iter::once(lz_tokens.as_slice()),
            nb,
            min_symbol,
            speed.effort().lz_refined_entropy,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        let leaves: Vec<(u32, u32)> = predictors.iter().copied().zip(quants).collect();
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
        return true;
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

    let learned = {
        // Per-tag MA samples: divided values, decoder-visible channel value as the
        // chan property, properties rescaled to the full-scale domain afterwards.
        let tag_samples: Vec<MaSamples> = pool.steal_map(scratch, n_tags, |t, _scratch| {
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
            sample_channel_ma(
                |y| &data[y * w..][..w],
                ch.w,
                ch.h,
                chan,
                0,
                &[],
                WpParams::DEFAULT,
                false,
                stride,
                |props, tok| samples.push(props, tok),
            );
            scale_ma_value_props(&mut samples, tag_quants[t] as i32);
            samples
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
            };
            let samples = &tag_samples[t];
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
            };
            learned[worst] = learn_ma_tree(&tag_samples[worst], params, pool, scratch);
            total_leaves = learned.iter().map(leaves_of).sum();
        }
        learned
    };

    let (tree_tokens, leaf_ctx, tag_base, num_ctx) =
        emit_lm_forest(&learned, &tag_quants, split, n_dc_slots, num_dc_groups);
    let num_ctx = num_ctx as usize;
    let distance_ctx = num_ctx as u32;

    let global_channel_tokens = pool.steal_map(scratch, split, |c, _scratch| {
        let ch = &channels[c];
        let data = &ch.data;
        let w = ch.w;
        let get_row = |y| &data[y * w..][..w];
        let mut tokens = Vec::with_capacity(ch.w * ch.h);
        tokenize_channel_ma_mult(
            get_row,
            ch.w,
            ch.h,
            c as u32,
            tag_quants[c] as i32,
            &learned[c],
            &leaf_ctx,
            tag_base[c],
            &mut tokens,
        );
        tokens
    });
    let mut global_tokens: Vec<Token> = Vec::new();
    for channel in global_channel_tokens {
        global_tokens.extend(channel);
    }
    // Literal-only: with per-pixel tree contexts and ANS-clustered histograms,
    // zero literals are near-free and LZ77 matches cannot pay for their
    // length+distance symbols (context-equal matching measured +10%,
    // value-only matching +4..12%). The LZ77 layer stays declared in the
    // header; no matches are ever emitted.
    let global_lz = lz77_literals(&global_tokens);
    let crop_group = |gdim: usize,
                      gx: usize,
                      gy: usize,
                      minsh: i32,
                      maxsh: i32,
                      slot_tag_base: usize,
                      _scratch: &mut CoderScratch|
     -> Vec<LzToken> {
        let mut gtok: Vec<Token> = Vec::new();
        for_each_squeeze_group_crop(
            &channels,
            split,
            gdim,
            gx,
            gy,
            minsh,
            maxsh,
            |within, c, rx0, ry0, rw, rh| {
                let ch = &channels[c];
                let tag = slot_tag_base + within;
                let data = &ch.data;
                let w = ch.w;
                let get_row = |y| &data[(ry0 + y) * w + rx0..][..rw];
                tokenize_channel_ma_mult(
                    get_row,
                    rw,
                    rh,
                    within as u32,
                    tag_quants[tag] as i32,
                    &learned[tag],
                    &leaf_ctx,
                    tag_base[tag],
                    &mut gtok,
                );
            },
        );
        lz77_literals(&gtok)
    };

    let dc_group_lz = pool.steal_map_with_threads(
        scratch,
        num_dc_groups,
        group_lz_threads(speed, pool),
        |group_index, scratch| {
            let gx = group_index % xsize_dc_groups;
            let gy = group_index / xsize_dc_groups;
            crop_group(LF_GROUP_DIM, gx, gy, 3, 1000, dc_tag_base, scratch)
        },
    );
    let ac_group_lz = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(speed, pool),
        |group_index, scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            crop_group(GROUP_DIM, gx, gy, 0, 2, ac_tag_base, scratch)
        },
    );

    let code = build_lz_pixel_code_opts(
        speed.effort().lz_exhaustive_entropy,
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

    layout.write_frame_header(epf, writer);

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
    layout.write_transforms(&steps, &mut sections[0]);
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

    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}

// ============ Lossy extra channel inside a VarDCT frame (squeeze) ============

/// Group sizes of the VarDCT frames this channel rides in.
const VARDCT_GROUP_DIM: usize = 256;
const VARDCT_LF_GROUP_DIM: usize = 2048;

/// One extra channel squeezed and quantized like the lossy-modular arm's luma
/// row, coded in the modular streams of a VarDCT frame. The frame's global
/// tree belongs to the VarDCT DC, so every section carries a local tree: one
/// leaf per channel, its quantization step as the residual multiplier.
pub(crate) struct ExtraSqueeze {
    channels: Vec<crate::squeeze::Channel>,
    quants: Vec<u32>,
    steps: Vec<crate::squeeze::SqueezeStep>,
    /// Channels before `split` live in the global stream.
    split: usize,
    min_symbol: u32,
    /// Learn a context tree per stream (decoder MA-tree walk) instead of the
    /// fixed one-Gradient-leaf-per-channel tree.
    learned_trees: bool,
}

impl ExtraSqueeze {
    /// `distance` is in luma-quantizer units; `max` is the channel's largest
    /// sample. None when the plane is too small to squeeze.
    pub(crate) fn new(
        plane: &AlphaPlane,
        xsize: usize,
        ysize: usize,
        distance: f32,
        learned_trees: bool,
    ) -> Option<Self> {
        use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
        let steps = default_squeeze_steps(xsize, ysize, 1);
        if steps.is_empty() {
            return None;
        }
        let max = match plane {
            AlphaPlane::U8(_) => 255.0f32,
            AlphaPlane::U16 { bits, .. } => ((1u32 << *bits) - 1) as f32,
            AlphaPlane::F32(_) => return None,
        };
        let data: Vec<i32> = (0..xsize * ysize).map(|i| plane.get_i32(i)).collect();
        let mut channels = vec![Channel {
            data,
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        }];
        for s in &steps {
            apply_step_forward(&mut channels, s);
        }
        // The arm's luma schedule is in units of 1/4096 of the sample range.
        let unit = max / LOSSY_MODULAR_XYB_SCALE[1];
        let quants: Vec<u32> = channels
            .iter_mut()
            .map(|ch| {
                let mut shift = (ch.hshift + ch.vshift).min(16);
                if shift > 0 {
                    shift -= 1;
                }
                let q = 0.25
                    * distance
                    * SQUEEZE_QUALITY_FACTOR_XYB
                    * SQUEEZE_XYB_QTABLE[0][shift as usize]
                    * unit;
                let q = (q as i32).max(1);
                quantize_channel_divide(ch, q);
                q as u32
            })
            .collect();
        let split = channels
            .iter()
            .position(|c| c.w > VARDCT_GROUP_DIM || c.h > VARDCT_GROUP_DIM)
            .unwrap_or(channels.len());
        let max_abs = channels
            .iter()
            .flat_map(|ch| ch.data.iter())
            .map(|v| v.unsigned_abs())
            .max()
            .unwrap_or(0);
        let value_bits = 33 - (2 * max_abs).max(1).leading_zeros();
        let min_symbol = if value_bits <= 13 {
            LZ77_MIN_SYMBOL
        } else {
            4 * value_bits + 24
        };
        Some(Self {
            channels,
            quants,
            steps,
            split,
            min_symbol,
            learned_trees,
        })
    }

    /// The GlobalModular image: its header (with the squeeze transform) and
    /// the channels that fit one group.
    pub(crate) fn write_global(&self, scratch: &mut CoderScratch, w: &mut BitWriter) {
        let crops: Vec<(usize, usize, usize, usize, usize)> = (0..self.split)
            .map(|c| (c, 0, 0, self.channels[c].w, self.channels[c].h))
            .collect();
        w.write(1, 0); // use_global_tree = 0
        w.write(1, 1); // wp_header all_default
        write_modular_transforms_squeeze_only(&self.steps, w);
        self.write_crops(&crops, scratch, w);
    }

    /// The channels of LF group (`gx`, `gy`): minimum shift 3 and up.
    pub(crate) fn write_lf_group(
        &self,
        gx: usize,
        gy: usize,
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        self.write_group(VARDCT_LF_GROUP_DIM, gx, gy, 3, 1000, scratch, w);
    }

    /// The channels of AC group (`gx`, `gy`): minimum shift 0..=2.
    pub(crate) fn write_ac_group(
        &self,
        gx: usize,
        gy: usize,
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        self.write_group(VARDCT_GROUP_DIM, gx, gy, 0, 2, scratch, w);
    }

    #[allow(clippy::too_many_arguments)]
    fn write_group(
        &self,
        gdim: usize,
        gx: usize,
        gy: usize,
        minsh: i32,
        maxsh: i32,
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        let mut crops = Vec::new();
        for_each_squeeze_group_crop(
            &self.channels,
            self.split,
            gdim,
            gx,
            gy,
            minsh,
            maxsh,
            |_within, c, rx0, ry0, rw, rh| crops.push((c, rx0, ry0, rw, rh)),
        );
        // A group with no channel in its shift range reads nothing.
        if crops.is_empty() {
            return;
        }
        w.write(1, 0); // use_global_tree = 0
        w.write(1, 1); // wp_header all_default
        w.write(2, 0); // no transforms
        self.write_crops(&crops, scratch, w);
    }

    /// Local tree + entropy code + tokens of one stream's channel crops: the
    /// fixed tree, or with learned trees whichever of the two is smaller.
    fn write_crops(
        &self,
        crops: &[(usize, usize, usize, usize, usize)],
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        if crops.is_empty() {
            return;
        }
        let mut fixed = BitWriter::new();
        self.write_crops_fixed(crops, scratch, &mut fixed);
        if self.learned_trees {
            let mut learned = BitWriter::new();
            self.write_crops_learned(crops, scratch, &mut learned);
            if learned.bits_written() < fixed.bits_written() {
                w.append(&learned);
                return;
            }
        }
        w.append(&fixed);
    }

    /// One learned subtree per channel slot, chained on property 0 with the
    /// slot's step stamped on its leaves (same scheme as the lossy-modular
    /// arm, restricted to scale-equivariant predictors).
    fn write_crops_learned(
        &self,
        crops: &[(usize, usize, usize, usize, usize)],
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        let pool = ThreadPool::new(1);
        let n = crops.len();
        let mut trees: Vec<LearnedTree> = Vec::with_capacity(n);
        for (slot, &(c, rx0, ry0, rw, rh)) in crops.iter().enumerate() {
            let ch = &self.channels[c];
            let (data, stride) = (&ch.data, ch.w);
            let px = rw * rh;
            let mut step = px.div_ceil(LM_SAMPLE_TARGET).max(1);
            if step > 1 && step.is_multiple_of(2) {
                step += 1;
            }
            let mut samples = MaSamples::with_capacity(px / step + 1);
            sample_channel_ma(
                |y| &data[(ry0 + y) * stride + rx0..][..rw],
                rw,
                rh,
                slot as u32,
                0,
                &[],
                WpParams::DEFAULT,
                false,
                step,
                |props, tok| samples.push(props, tok),
            );
            scale_ma_value_props(&mut samples, self.quants[c] as i32);
            let params = |max_leaves: usize| MaLearnParams {
                alphabet: self.min_symbol as usize,
                max_leaves,
                split_cost_bits: MA_SPLIT_COST_BITS / step as f32,
                min_node: MA_MIN_NODE_SAMPLES,
                allow_wp: false,
                allowed_preds: LM_SAFE_PRED_MASK,
                side_preds: 1,
                max_candidates: MA_PHOTO_CANDIDATES,
            };
            let budget = (px / LM_BUDGET_DIV).clamp(1, LM_BUDGET_CAP);
            let tree = if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
                learn_ma_tree(&samples, params(1), &pool, scratch)
            } else {
                let tree = learn_ma_tree(&samples, params(budget), &pool, scratch);
                let leaves = tree.nodes.len().div_ceil(2);
                let overhead = tree.nodes.len() as f64 * 10.0 + leaves as f64 * 200.0;
                if tree.est_bits * step as f64 + overhead < tree.flat_bits * step as f64 {
                    tree
                } else {
                    learn_ma_tree(&samples, params(1), &pool, scratch)
                }
            };
            trees.push(tree);
        }
        let mults: Vec<u32> = crops.iter().map(|&(c, ..)| self.quants[c]).collect();
        let (tree_tokens, leaf_ctx, bases, num_ctx) = emit_slot_forest(&trees, &mults);
        let mut tokens: Vec<Token> = Vec::new();
        for (slot, &(c, rx0, ry0, rw, rh)) in crops.iter().enumerate() {
            let ch = &self.channels[c];
            let (data, stride) = (&ch.data, ch.w);
            tokenize_channel_ma_mult(
                |y| &data[(ry0 + y) * stride + rx0..][..rw],
                rw,
                rh,
                slot as u32,
                self.quants[c] as i32,
                &trees[slot],
                &leaf_ctx,
                bases[slot],
                &mut tokens,
            );
        }
        let lz = lz77_literals(&tokens);
        let code = build_lz_pixel_code_opts(
            false,
            std::iter::once(lz.as_slice()),
            num_ctx as usize,
            self.min_symbol,
            true,
            true,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        write_tree_lz77(
            &tree_tokens,
            &code,
            self.min_symbol,
            &mut scratch.huffman_pool,
            w,
        );
        write_lz_section(&lz, num_ctx, &code, self.min_symbol, w);
    }

    fn write_crops_fixed(
        &self,
        crops: &[(usize, usize, usize, usize, usize)],
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        let n = crops.len();
        let grad_pack_fn = selected_grad_pack_interior_fn();
        let mut tokens: Vec<Token> = Vec::new();
        for (slot, &(c, rx0, ry0, rw, rh)) in crops.iter().enumerate() {
            let ch = &self.channels[c];
            let data = &ch.data;
            let stride = ch.w;
            tokenize_plane_rows(
                channel_to_context(slot, n),
                |y| &data[(ry0 + y) * stride + rx0..][..rw],
                rw,
                rh,
                PREDICTOR_GRADIENT,
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
        }
        let lz = lz77_literals(&tokens);
        let code = build_lz_pixel_code(
            false,
            std::iter::once(lz.as_slice()),
            n,
            self.min_symbol,
            true,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        let leaves: Vec<(u32, u32)> = crops
            .iter()
            .map(|&(c, ..)| (PREDICTOR_GRADIENT, self.quants[c]))
            .collect();
        write_tree_lz77(
            &build_balanced_tree_tokens_mul(&leaves),
            &code,
            self.min_symbol,
            &mut scratch.huffman_pool,
            w,
        );
        write_lz_section(&lz, n as u32, &code, self.min_symbol, w);
    }
}

/// Chain per-slot learned subtrees on property 0 (the within-stream channel
/// index) and BFS-emit them, each leaf carrying its slot's multiplier.
/// Returns the tree tokens, the arena-indexed leaf contexts, each slot's
/// arena base and the context count.
fn emit_slot_forest(trees: &[LearnedTree], mults: &[u32]) -> (Vec<Token>, Vec<u32>, Vec<u32>, u32) {
    use std::collections::VecDeque;
    let mut nodes: Vec<MaNode> = Vec::new();
    let mut mult_of: Vec<u32> = Vec::new();
    let mut bases = vec![0u32; trees.len()];
    let mut root = 0u32;
    for (slot, tree) in trees.iter().enumerate() {
        let base = nodes.len() as u32;
        bases[slot] = base;
        for n in &tree.nodes {
            nodes.push(match *n {
                MaNode::Split { prop, val, gt, le } => MaNode::Split {
                    prop,
                    val,
                    gt: gt + base,
                    le: le + base,
                },
                MaNode::Leaf { pred } => MaNode::Leaf { pred },
            });
            mult_of.push(mults[slot]);
        }
        if slot == 0 {
            root = base;
        } else {
            nodes.push(MaNode::Split {
                prop: 0,
                val: (slot - 1) as i32,
                gt: base,
                le: root,
            });
            mult_of.push(1);
            root = (nodes.len() - 1) as u32;
        }
    }
    let mut tokens = Vec::new();
    let mut leaf_ctx = vec![u32::MAX; nodes.len()];
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(root);
    let mut ctx = 0u32;
    while let Some(i) = queue.pop_front() {
        match nodes[i as usize] {
            MaNode::Split { prop, val, gt, le } => {
                push_split(&mut tokens, prop as u32, val);
                queue.push_back(gt);
                queue.push_back(le);
            }
            MaNode::Leaf { pred } => {
                push_leaf_mul(&mut tokens, pred, mult_of[i as usize]);
                leaf_ctx[i as usize] = ctx;
                ctx += 1;
            }
        }
    }
    (tokens, leaf_ctx, bases, ctx)
}

#[cfg(test)]
mod lm_plain_tests {
    use super::*;
    use crate::coding::CodingTransform;

    fn ramp(w: usize, h: usize, seed: i32) -> Vec<i32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                ((x * 523 + y * 311 + seed * 97) % 65536).max(0)
            })
            .collect()
    }

    fn input(coding: CodingTransform, color: usize, extra: Vec<(Vec<i32>, f32)>) -> LmPlainInput {
        LmPlainInput {
            color: (0..color).map(|c| ramp(64, 48, c as i32)).collect(),
            extra,
            max: 65535.0,
            coding,
        }
    }

    #[test]
    fn ycbcr_layout_pre_squeezes_the_outer_planes() {
        let q =
            lm_quantize_plain(&input(CodingTransform::YCbCr, 3, Vec::new()), 64, 48, 8.0).unwrap();
        assert_eq!(
            q.layout,
            LmLayout::Plain {
                ycbcr: true,
                rct: false,
                extra: 0
            }
        );
        let pre: Vec<(usize, usize)> = q
            .steps
            .iter()
            .filter(|s| !s.in_place)
            .map(|s| (s.begin_c, s.num_c))
            .collect();
        assert_eq!(pre, [(0, 1), (0, 1), (2, 1), (2, 1)]);
        assert!(q.steps.iter().filter(|s| s.in_place).all(|s| s.num_c == 3));
        // The chroma pre-squeeze residuals come last and sit on the coarser
        // chroma schedule.
        let nb = q.channels.len();
        assert!(q.quants[nb - 1] > q.quants[nb - 1 - 4]);
        assert!(q.quants.iter().all(|&v| v >= 1));
    }

    #[test]
    fn rgb_layout_uses_the_rct_and_gray_neither() {
        let q =
            lm_quantize_plain(&input(CodingTransform::Rgb, 3, Vec::new()), 64, 48, 8.0).unwrap();
        assert_eq!(
            q.layout,
            LmLayout::Plain {
                ycbcr: false,
                rct: true,
                extra: 0
            }
        );
        let pre: Vec<usize> = q
            .steps
            .iter()
            .filter(|s| !s.in_place)
            .map(|s| s.begin_c)
            .collect();
        assert_eq!(pre, [1, 1, 2, 2]);
        let g =
            lm_quantize_plain(&input(CodingTransform::Rgb, 1, Vec::new()), 64, 48, 8.0).unwrap();
        assert_eq!(
            g.layout,
            LmLayout::Plain {
                ycbcr: false,
                rct: false,
                extra: 0
            }
        );
        assert!(g.steps.iter().all(|s| s.in_place && s.num_c == 1));
    }

    #[test]
    fn extra_channels_follow_their_own_distance() {
        let lossless = (ramp(64, 48, 7), 0.0);
        let lossy = (ramp(64, 48, 9), 40.0);
        let q = lm_quantize_plain(
            &input(CodingTransform::YCbCr, 3, vec![lossless, lossy]),
            64,
            48,
            8.0,
        )
        .unwrap();
        assert_eq!(
            q.layout,
            LmLayout::Plain {
                ycbcr: true,
                rct: false,
                extra: 2
            }
        );
        assert!(q.steps.iter().filter(|s| s.in_place).all(|s| s.num_c == 5));
        let nb = q.channels.len();
        // Channel i % 5: 3 = the lossless extra (step 1 everywhere), 4 = the
        // lossy one (coarser than luma at the same level).
        for i in 0..nb - 4 {
            match i % 5 {
                3 => assert_eq!(q.quants[i], 1, "lossless extra at {i}"),
                4 => assert!(q.quants[i] >= q.quants[i - 3], "lossy extra at {i}"),
                _ => {}
            }
        }
        // Tiny frames have no pyramid.
        assert!(
            lm_quantize_plain(&input(CodingTransform::YCbCr, 3, Vec::new()), 8, 8, 8.0).is_none()
        );
    }

    #[test]
    fn plain_distance_is_monotonic_and_flat_outside_the_knots() {
        for coding in [CodingTransform::YCbCr, CodingTransform::Rgb] {
            let mut last = 0.0;
            for d in [0.1, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 10.0] {
                let e = lm_plain_distance(coding, d);
                assert!(e > last, "{coding:?} d={d}");
                assert!(
                    e > d,
                    "{coding:?} d={d}: the arm quantizes coarser than its distance"
                );
                last = e;
            }
            assert_eq!(
                lm_plain_distance(coding, 0.5) * 2.0,
                lm_plain_distance(coding, 1.0)
            );
        }
        assert_eq!(interpolate_knots(&[(1.0, 2.0), (3.0, 4.0)], 2.0), 3.0);
    }
}
