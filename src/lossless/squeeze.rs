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

//! Progressive lossless modular Squeeze frames and transforms.

mod extra;
#[cfg(test)]
mod extra_squeeze_tests;
mod fold_cost;
mod forest;
mod frame;
#[cfg(test)]
mod lm_plain_tests;
mod plane;
mod quantize;

pub(crate) use extra::ExtraSqueeze;
pub(crate) use frame::{
    dc_frame_supported, encode_dc_frame, encode_frame_lossy_modular_plain,
    encode_frame_lossy_modular_squeeze,
};
pub(crate) use plane::{GlobalPlaneTree, StreamCrop, learn_global_plane_tree};
pub(crate) use quantize::{LmPlainInput, lm_calibrated_distance, lm_plain_distance};

use super::groups::{DeepLzScratchPool, group_lz_threads, write_lz_groups};
use super::header::{write_frame_header_modular_kind, write_toc_entry};
use super::lz77::{
    LzToken, build_lz_pixel_code, lz77_compress_for_speed, lz77_compress_for_speed_with_depth,
    write_local_tree_lz77, write_lz_section,
};
use super::predictor::{
    GradPackInteriorFn, SqueezePredictorCost, channel_to_context, choose_predictor_for_plane,
    fixed_predictor, tokenize_plane_rows,
};
use super::rct::write_rct_transform;
use super::{GROUP_DIM, GroupLayout, LF_GROUP_DIM};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS};
use crate::encode_image::AlphaPlane;
use crate::entropy::Token;
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Lossy-modular context budget: the pre-4096 scratch bound, kept so the
/// lossless leaf cap can grow without perturbing this path's trees.
const LM_MAX_CONTEXTS: usize = 1024;

const _: () = assert!(LM_MAX_CONTEXTS <= LZ77_MAX_CONTEXTS);

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
    frame_kind: ModularFrameKind<'_>,
    layout: GroupLayout,
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

    write_frame_header_modular_kind(alpha.is_some(), frame_kind, layout, writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut section);

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
    frame_kind: ModularFrameKind<'_>,
    layout: GroupLayout,
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
        speed.effort().lz_exhaustive_entropy(),
        std::iter::once(global_lz.as_slice())
            .chain(dc_group_lz.iter().map(Vec::as_slice))
            .chain(ac_group_lz.iter().map(Vec::as_slice)),
        nb,
        min_symbol,
        speed.effort().lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    write_frame_header_modular_kind(alpha.is_some(), frame_kind, layout, writer);

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
