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

use crate::ac_context::K_COMPACT_BLOCK_CONTEXT_MAP;
use crate::bit_writer::BitWriter;
use crate::dc_group_data::DcGroupData;
use crate::enc_group::write_ac_group;
use crate::enc_xyb::to_xyb;
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    EntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code, write_token,
};
use crate::image::{Image3B, Image3F, Rect};
use crate::quant_weights::DequantMatrices;
use crate::static_entropy_codes::{
    K_AC_CONTEXT_MAP, K_AC_PREFIX_CODES, K_CONTEXT_TREE_TOKENS, K_DC_CONTEXT_MAP,
    K_DC_PREFIX_CODES, K_GRADIENT_CONTEXT_LUT,
};

const K_BLOCK_DIM: usize = 8;
const K_TILE_DIM: usize = 64;
const K_GROUP_DIM: usize = 256;
const K_DC_GROUP_DIM: usize = 2048;
const K_GROUP_DIM_IN_BLOCKS: usize = 32; // = K_GROUP_DIM / K_BLOCK_DIM
const K_TILE_DIM_IN_BLOCKS: usize = 8; // = K_TILE_DIM / K_BLOCK_DIM
const K_NUM_TREE_CONTEXTS: usize = 6;

const K_GRAD_RANGE_MID: i64 = 512;
const K_GRAD_RANGE_MIN: i64 = 0;
const K_GRAD_RANGE_MAX: i64 = 1023;

// -----------------------------------------------------------------------------
// Image dimensions.
// -----------------------------------------------------------------------------

#[allow(dead_code)]
struct ImageDim {
    xsize: usize,
    ysize: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_groups: usize,
    ysize_groups: usize,
    xsize_dc_groups: usize,
    ysize_dc_groups: usize,
    num_groups: usize,
    num_dc_groups: usize,
}

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

impl ImageDim {
    fn new(xsize: usize, ysize: usize) -> Self {
        let xsize_blocks = div_ceil(xsize, K_BLOCK_DIM);
        let ysize_blocks = div_ceil(ysize, K_BLOCK_DIM);
        let xsize_groups = div_ceil(xsize, K_GROUP_DIM);
        let ysize_groups = div_ceil(ysize, K_GROUP_DIM);
        let xsize_dc_groups = div_ceil(xsize, K_DC_GROUP_DIM);
        let ysize_dc_groups = div_ceil(ysize, K_DC_GROUP_DIM);
        Self {
            xsize,
            ysize,
            xsize_blocks,
            ysize_blocks,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_groups: xsize_groups * ysize_groups,
            num_dc_groups: xsize_dc_groups * ysize_dc_groups,
        }
    }
}

// -----------------------------------------------------------------------------
// Distance-derived parameters.
// -----------------------------------------------------------------------------

struct DistanceParams {
    distance: f32,
    global_scale: i32,
    quant_dc: i32,
    scale: f32,
    scale_dc: f32,
    x_qm_scale: u32,
    epf_iters: u32,
    gab_enabled: bool,
}

fn clamp1<T: PartialOrd>(v: T, lo: T, hi: T) -> T {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn quant_dc(distance: f32) -> f32 {
    let k_dc_quant_pow = 0.57f32;
    let k_dc_quant = 1.12f32;
    let k_dc_mul = 2.9f32;
    let effective = k_dc_mul * (distance / k_dc_mul).powf(k_dc_quant_pow);
    let effective = clamp1(effective, 0.5 * distance, distance);
    (k_dc_quant / effective).min(50.0)
}

fn compute_distance_params(distance: f32) -> DistanceParams {
    const K_GLOBAL_SCALE_DENOM: i32 = 1 << 16;
    const K_GLOBAL_SCALE_NUMERATOR: i32 = 4096;
    const K_AC_QUANT: f32 = 0.8;
    const K_QUANT_FIELD_TARGET: f32 = 5.0;

    let qdc = quant_dc(distance);
    let mut scale = K_GLOBAL_SCALE_DENOM as f32 * K_AC_QUANT / (distance * K_QUANT_FIELD_TARGET);
    scale = clamp1(scale, 1.0, (1 << 15) as f32);
    let scaled_quant_dc = (qdc * K_GLOBAL_SCALE_NUMERATOR as f32 * 1.6) as i32;
    let global_scale = clamp1(scale as i32, 1, scaled_quant_dc);
    let scale_f = global_scale as f32 / K_GLOBAL_SCALE_DENOM as f32;
    let qd = ((qdc / scale_f) + 0.5) as i32;
    let qd = clamp1(qd, 1, 1 << 16);
    let scale_dc = qd as f32 * scale_f;

    let mut x_qm_scale: u32 = 2;
    if distance > 1.25 {
        x_qm_scale += 1;
    }
    if distance > 9.0 {
        x_qm_scale += 1;
    }
    if distance < 0.299 {
        x_qm_scale += 1;
    }

    let mut epf_iters: u32 = 0;
    for t in [0.7f32, 1.5, 4.0] {
        if distance >= t {
            epf_iters += 1;
        }
    }

    DistanceParams {
        distance,
        global_scale,
        quant_dc: qd,
        scale: scale_f,
        scale_dc,
        x_qm_scale,
        epf_iters,
        // Always-on for lossy. libjxl default. Costs ~0.05 dB worth of file
        // size but recovers high-frequency sharpness lost in quantization.
        gab_enabled: true,
    }
}

// -----------------------------------------------------------------------------
// DC tokens, AC metadata tokens, context tree tokens.
// -----------------------------------------------------------------------------

#[inline]
fn clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let mn = n.min(w);
    let mx = n.max(w);
    let g = (n as i64 + w as i64 - l as i64) as i32;
    g.clamp(mn, mx)
}

/// Emit DC tokens for one DC group (in channel order Y, X, B).
fn write_dc_tokens(dc_data: &DcGroupData, dc_code: &EntropyCode, writer: &mut BitWriter) {
    for c in [1usize, 0, 2] {
        let plane = dc_data.quant_dc.plane(c);
        let ysize = plane.ysize();
        let xsize = plane.xsize();
        for y in 0..ysize {
            for x in 0..xsize {
                let qrow_here = plane.row(y)[x] as i64;
                let row_above = if y > 0 { Some(plane.row(y - 1)) } else { None };
                // libjxl-tiny formulas:
                //   left = x ? qrow[x-1] : y ? qrow_above[x] : 0
                //   top  = y ? qrow_above[x] : left
                //   topleft = (x && y) ? qrow_above[x-1] : left
                let left: i64 = if x > 0 {
                    plane.row(y)[x - 1] as i64
                } else if let Some(rt) = row_above {
                    rt[x] as i64
                } else {
                    0
                };
                let top: i64 = match row_above {
                    Some(rt) => rt[x] as i64,
                    None => left,
                };
                let topleft: i64 = if x > 0 && y > 0 {
                    row_above.unwrap()[x - 1] as i64
                } else {
                    left
                };
                let guess = clamped_gradient(top as i32, left as i32, topleft as i32);
                let grad_prop = clamp1(
                    K_GRAD_RANGE_MID + top + left - topleft,
                    K_GRAD_RANGE_MIN,
                    K_GRAD_RANGE_MAX,
                );
                let residual = qrow_here as i32 - guess;
                let ctx_id = K_GRADIENT_CONTEXT_LUT[grad_prop as usize] as u32;
                write_token(Token::new(ctx_id, pack_signed(residual)), dc_code, writer);
            }
        }
    }
}

/// AC metadata: ytox/ytob CfL maps (all 0), AC strategy (all 0 = DCT-8x8),
/// quant field residuals (all 0), and EPF (token (0, PackSigned(4)) per block).
///
/// In libjxl-tiny ALL four sub-streams use the same shared dc_code.
fn write_ac_metadata_tokens(dc_data: &DcGroupData, dc_code: &EntropyCode, writer: &mut BitWriter) {
    let xsize_blocks = dc_data.ac_strategy.xsize();
    let ysize_blocks = dc_data.ac_strategy.ysize();
    let xtiles = dc_data.ytox_map.xsize();
    let ytiles = dc_data.ytox_map.ysize();

    // --- (a) YtoX and YtoB tokens, with gradient prediction ---
    for c in 0..2usize {
        let cfl_map = if c == 0 {
            &dc_data.ytox_map
        } else {
            &dc_data.ytob_map
        };
        for y in 0..ytiles {
            for x in 0..xtiles {
                let here: i64 = cfl_map.row(y)[x] as i64;
                let row_above = if y > 0 {
                    Some(cfl_map.row(y - 1))
                } else {
                    None
                };
                let left: i64 = if x > 0 {
                    cfl_map.row(y)[x - 1] as i64
                } else if let Some(rt) = row_above {
                    rt[x] as i64
                } else {
                    0
                };
                let top: i64 = match row_above {
                    Some(rt) => rt[x] as i64,
                    None => left,
                };
                let topleft: i64 = if x > 0 && y > 0 {
                    row_above.unwrap()[x - 1] as i64
                } else {
                    left
                };
                let guess = clamped_gradient(top as i32, left as i32, topleft as i32);
                let residual = here as i32 - guess;
                let ctx_id = 2u32 - c as u32;
                write_token(Token::new(ctx_id, pack_signed(residual)), dc_code, writer);
            }
        }
    }

    // --- (b) AC strategy tokens — emitted once per FIRST block ---
    let mut left: i32 = 0;
    for y in 0..ysize_blocks {
        for x in 0..xsize_blocks {
            if !dc_data.ac_strategy.is_first_block(x, y) {
                continue;
            }
            let cur = dc_data.ac_strategy.strategy_code(x, y) as i32;
            let ctx_id = if left > 11 {
                7
            } else if left > 5 {
                8
            } else if left > 3 {
                9
            } else {
                10
            } as u32;
            write_token(Token::new(ctx_id, pack_signed(cur)), dc_code, writer);
            left = cur;
        }
    }
    // --- (c) Quant field residuals — emitted once per FIRST block ---
    // Initial `left` = strategy_code(0, 0) of first block (always a first block).
    let mut left: i32 = dc_data.ac_strategy.strategy_code(0, 0) as i32;
    for y in 0..ysize_blocks {
        let row_qf = dc_data.raw_quant_field.row(y);
        for x in 0..xsize_blocks {
            if !dc_data.ac_strategy.is_first_block(x, y) {
                continue;
            }
            let cur: i32 = row_qf[x] as i32 - 1;
            let residual: i32 = cur - left;
            let ctx_id = if left > 11 {
                3
            } else if left > 5 {
                4
            } else if left > 3 {
                5
            } else {
                6
            } as u32;
            write_token(Token::new(ctx_id, pack_signed(residual)), dc_code, writer);
            left = cur;
        }
    }
    // --- (d) EPF tokens: (0, PackSigned(4)) per block ---
    let nblocks = xsize_blocks * ysize_blocks;
    for _ in 0..nblocks {
        write_token(Token::new(0, pack_signed(4)), dc_code, writer);
    }
}

/// Build and emit the context tree.
fn write_context_tree(num_dc_groups: usize, writer: &mut BitWriter) {
    // Build tokens with the patched value at index 1.
    let mut tokens: Vec<Token> = Vec::with_capacity(K_CONTEXT_TREE_TOKENS.len());
    for (i, &(ctx, val)) in K_CONTEXT_TREE_TOKENS.iter().enumerate() {
        let v = if i == 1 {
            pack_signed(1 + num_dc_groups as i32)
        } else {
            val
        };
        tokens.push(Token::new(ctx, v));
    }
    // OptimizeEntropyCode clusters the K_NUM_TREE_CONTEXTS=6 contexts.
    let code = optimize_entropy_code(&tokens, K_NUM_TREE_CONTEXTS);
    let code_ref = code.as_ref();

    writer.write(1, 1); // not an empty tree
    writer.write(1, 0); // no lz77
    write_entropy_code(&code_ref, writer);
    for t in &tokens {
        write_token(*t, &code_ref, writer);
    }
}

// -----------------------------------------------------------------------------
// Frame header, quant scales, DC global, AC global, DC group, TOC.
// -----------------------------------------------------------------------------

fn write_frame_header(
    x_qm_scale: u32,
    epf_iters: u32,
    gab_enabled: bool,
    has_alpha: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // not all default
    w.write(2, 0); // regular frame
    w.write(1, 0); // vardct
    w.write(2, 2); // flags selector bits (17 .. 272)
    w.write(8, 111); // skip adaptive dc flag (128)
    w.write(2, 0); // no upsampling

    // Per-extra-channel upsampling. Same u2S(1,2,4,8) code, default 1:
    // selector "00" gives 1, written once per extra channel.
    if has_alpha {
        w.write(2, 0); // ec_upsampling[0] = 1
    }

    w.write(3, x_qm_scale as u64);
    w.write(3, 2); // b_qm_scale
    w.write(2, 0); // one pass
    w.write(1, 0); // no custom frame size or origin

    // Color-channel BlendingInfo: mode=Replace, full_frame=true means
    // source/alpha_channel/clamp are all omitted. The two zero bits select
    // BlendingMode::Replace.
    w.write(2, 0); // color blend mode = Replace

    // Per-extra-channel BlendingInfo. With num_extra_channels=1, BlendingInfo
    // for the alpha channel has:
    //   mode = Replace (2 bits = 00)
    //   alpha_channel: NOT written (mode != Blend/AWA)
    //   clamp: NOT written (mode != Blend/AWA/Mul)
    //   source: NOT written (full_frame && Replace)
    if has_alpha {
        w.write(2, 0); // ec_blending_info[0].mode = Replace
    }

    w.write(1, 1); // last frame
    w.write(2, 0); // no name
    if epf_iters == 2 && gab_enabled {
        w.write(1, 1); // default loop filter (gab=1, epf=2)
    } else {
        w.write(1, 0); // not default
        if gab_enabled {
            w.write(1, 1); // gaborish enabled
            w.write(1, 0); // gab_custom = false (use defaults)
        } else {
            w.write(1, 0); // no gaborish
        }
        w.write(2, epf_iters as u64);
        if epf_iters > 0 {
            w.write(1, 0); // default epf sharpness
            w.write(1, 0); // default epf weights
            w.write(1, 0); // default epf sigma
        }
        w.write(2, 0); // no loop filter extensions
    }
    w.write(2, 0); // no frame header extensions
}

fn write_quant_scales(global_scale: i32, quant_dc: i32, w: &mut BitWriter) {
    if global_scale < 2049 {
        w.write(2, 0);
        w.write(11, (global_scale - 1) as u64);
    } else if global_scale < 4097 {
        w.write(2, 1);
        w.write(11, (global_scale - 2049) as u64);
    } else if global_scale < 8193 {
        w.write(2, 2);
        w.write(12, (global_scale - 4097) as u64);
    } else {
        w.write(2, 3);
        w.write(16, (global_scale - 8193) as u64);
    }
    if quant_dc == 16 {
        w.write(2, 0);
    } else if quant_dc < 33 {
        w.write(2, 1);
        w.write(5, (quant_dc - 1) as u64);
    } else if quant_dc < 257 {
        w.write(2, 2);
        w.write(8, (quant_dc - 1) as u64);
    } else {
        w.write(2, 3);
        w.write(16, (quant_dc - 1) as u64);
    }
}

fn write_dc_global(
    distp: &DistanceParams,
    num_dc_groups: usize,
    dc_code: &EntropyCode,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    w: &mut BitWriter,
) {
    w.write(1, 1); // default dequant DC
    write_quant_scales(distp.global_scale, distp.quant_dc, w);
    w.write(1, 0); // non-default BlockCtxMap
    w.write(16, 0); // no dc ctx, no qft

    // WriteContextMap with kCompactBlockContextMap (only context map, no prefix codes).
    {
        // Empty prefix-codes slice; WriteContextMap builds its own.
        let empty_codes: [crate::entropy::PrefixCode; 0] = [];
        let cm_entropy = EntropyCode {
            context_map: &K_COMPACT_BLOCK_CONTEXT_MAP,
            num_contexts: K_COMPACT_BLOCK_CONTEXT_MAP.len(),
            prefix_codes: &empty_codes,
            num_prefix_codes: 0,
            orig_context_map: None,
            orig_num_contexts: 0,
        };
        crate::entropy::write_context_map(&cm_entropy, w);
    }

    w.write(1, 1); // default DC clamp (= ColorCorrelationParams.all_default = true)

    // Global tree.
    // write_context_tree emits "have_tree=1 + Histograms::decode (tree's own entropy code)
    // + tree tokens". The TREE'S PIXEL HISTOGRAMS are then written as the next two
    // bits + entropy code (DC entropy code, since it's the global tree used for DC).
    write_context_tree(num_dc_groups, w);
    w.write(1, 0); // no lz77 (for the global tree's pixel histograms = dc_code)

    // Then the static DC entropy code: this is the global tree's pixel histograms.
    write_entropy_code(dc_code, w);

    // FullModularImage::read happens HERE in the decoder. If we declared an alpha
    // extra channel, write its GroupHeader + local tree + pixel data now.
    if let Some(alpha_plane) = alpha {
        crate::modular::write_global_alpha_modular(alpha_plane, xsize, ysize, w);
    }
}

fn write_ac_global(num_groups: usize, ac_code: &EntropyCode, w: &mut BitWriter) {
    w.write(1, 1); // all default quant matrices
    if num_groups > 1 {
        // CeilLog2Nonzero(num_groups)
        let bits = 32
            - (num_groups as u32).leading_zeros()
            - if num_groups.is_power_of_two() { 1 } else { 0 };
        if bits != 0 {
            w.write(bits as usize, 0);
        }
    }
    w.write(2, 3);
    w.write(13, 0); // all default coeff order
    w.write(1, 0); // no lz77
    write_entropy_code(ac_code, w);
}

fn write_dc_group(dc_data: &DcGroupData, dc_code: &EntropyCode, w: &mut BitWriter) {
    w.write(2, 0); // extra_dc_precision
    w.write(4, 3); // use global tree, default wp, no transforms
    write_dc_tokens(dc_data, dc_code, w);

    let num_blocks = dc_data.ac_strategy.xsize() * dc_data.ac_strategy.ysize();
    let num_ac_blocks = dc_data.ac_strategy.count_first_blocks();
    // CeilLog2Nonzero(num_blocks)
    let nb_bits = if num_blocks <= 1 {
        0
    } else {
        32 - (num_blocks as u32).leading_zeros() as usize
            - if num_blocks.is_power_of_two() { 1 } else { 0 }
    };
    if nb_bits != 0 {
        w.write(nb_bits, (num_ac_blocks - 1) as u64);
    }
    w.write(4, 3);
    write_ac_metadata_tokens(dc_data, dc_code, w);
}

fn write_toc(sizes: &[usize], w: &mut BitWriter) {
    w.write(1, 0); // no permutation
    w.zero_pad_to_byte();
    let k_bits = [10usize, 14, 22, 30];
    for &s in sizes {
        let mut offset: usize = 0;
        let mut ok = false;
        for (i, &b) in k_bits.iter().enumerate() {
            if s < offset + (1usize << b) {
                w.write(2, i as u64);
                w.write(b, (s - offset) as u64);
                ok = true;
                break;
            }
            offset += 1usize << b;
        }
        assert!(ok, "section size {} too large for TOC", s);
    }
    w.zero_pad_to_byte();
}

fn combine_sections(sections: &mut Vec<BitWriter>, writer: &mut BitWriter) {
    if sections.len() == 4 {
        // Single AC group case: concat sections 1..4 (bitwise) into section 0.
        let tail: Vec<BitWriter> = sections.drain(1..).collect();
        for s in &tail {
            sections[0].append(s);
        }
    }
    let sizes: Vec<usize> = sections
        .iter()
        .map(|s| s.bits_written().div_ceil(8))
        .collect();
    write_toc(&sizes, writer);
    // After write_toc, writer is byte-aligned.
    writer.append_byte_aligned(sections);
}

pub(crate) fn encode_frame(
    distance: f32,
    linear: &Image3F,
    alpha: Option<&AlphaPlane>,
    writer: &mut BitWriter,
) {
    let dim = ImageDim::new(linear.xsize(), linear.ysize());
    let distp = compute_distance_params(distance);
    let matrices = DequantMatrices::new();
    let dc_code = EntropyCode::new(&K_DC_CONTEXT_MAP, &K_DC_PREFIX_CODES);
    let ac_code = EntropyCode::new(&K_AC_CONTEXT_MAP, &K_AC_PREFIX_CODES);

    let mut opsin = linear.clone();
    to_xyb(&mut opsin);

    // Forward gaborish. Decoder runs the inverse on the way out; the pair
    // forms a pre/de-emphasis around the lossy round-trip.
    if distp.gab_enabled {
        crate::gaborish::gaborish_inverse(&mut opsin, 0.990_851_1);
    }

    // Section layout:
    //   0                                  : DC global
    //   1..1+num_dc_groups                 : DC groups
    //   1+num_dc_groups                    : AC global
    //   2+num_dc_groups..                  : AC groups
    let num_sections = 2 + dim.num_dc_groups + dim.num_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // Process DC groups.
    for dc_gy in 0..dim.ysize_dc_groups {
        for dc_gx in 0..dim.xsize_dc_groups {
            process_dc_group(
                linear,
                &opsin,
                &dim,
                &distp,
                &matrices,
                &dc_code,
                &ac_code,
                dc_gx,
                dc_gy,
                alpha,
                &mut sections,
            );
        }
    }

    // Globals (after AC groups so we have the data; semantically order independent).
    write_dc_global(
        &distp,
        dim.num_dc_groups,
        &dc_code,
        alpha,
        dim.xsize,
        dim.ysize,
        &mut sections[0],
    );
    write_ac_global(
        dim.num_groups,
        &ac_code,
        &mut sections[1 + dim.num_dc_groups],
    );

    // Frame header into writer, then TOC + sections.
    write_frame_header(
        distp.x_qm_scale,
        distp.epf_iters,
        distp.gab_enabled,
        alpha.is_some(),
        writer,
    );
    combine_sections(&mut sections, writer);
}

#[allow(clippy::too_many_arguments)]
fn process_dc_group(
    linear: &Image3F,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    matrices: &DequantMatrices,
    dc_code: &EntropyCode,
    ac_code: &EntropyCode,
    dc_gx: usize,
    dc_gy: usize,
    alpha: Option<&AlphaPlane>,
    sections: &mut [BitWriter],
) {
    // DC group rect in pixels (clamped to image bounds).
    let dc_group_x0 = dc_gx * K_DC_GROUP_DIM;
    let dc_group_y0 = dc_gy * K_DC_GROUP_DIM;
    let dc_group_xsize = K_DC_GROUP_DIM.min(dim.xsize.saturating_sub(dc_group_x0));
    let dc_group_ysize = K_DC_GROUP_DIM.min(dim.ysize.saturating_sub(dc_group_y0));
    let dc_group_xsize_blocks = div_ceil(dc_group_xsize, K_BLOCK_DIM);
    let dc_group_ysize_blocks = div_ceil(dc_group_ysize, K_BLOCK_DIM);
    let dc_group_xsize_groups = div_ceil(dc_group_xsize, K_GROUP_DIM);
    let dc_group_ysize_groups = div_ceil(dc_group_ysize, K_GROUP_DIM);

    let mut dc_data = DcGroupData::new(dc_group_xsize_blocks, dc_group_ysize_blocks);

    // Bogus adaptive quantization: fill raw_quant_field with per-block multipliers
    // derived from local luma gradient magnitude in `linear`. Without this, every
    // block uses quant=1 (the flat libjxl effort-1 behaviour).
    crate::adaptive_quant::fill_quant_field(
        linear,
        &mut dc_data.raw_quant_field,
        dc_group_x0,
        dc_group_y0,
    );

    // Block selection: choose DCT16X8 / DCT8X16 for super-blocks where they
    // beat per-block DCT8. AdjustQuantField propagates max-quant across
    // covered blocks of each chosen multi-block transform.
    // Uses opsin (XYB, post-gaborish) so cost estimates use the right scale.
    crate::enc_ac_strategy::fill_ac_strategy(
        opsin,
        distp.distance,
        matrices,
        &mut dc_data.raw_quant_field,
        &mut dc_data.ac_strategy,
    );

    // For each AC group within this DC group.
    let num_groups_here = dc_group_xsize_groups * dc_group_ysize_groups;
    for gix in 0..num_groups_here {
        let gx = gix % dc_group_xsize_groups;
        let gy = gix / dc_group_xsize_groups;
        let image_gx = dc_gx * (K_DC_GROUP_DIM / K_GROUP_DIM) + gx;
        let image_gy = dc_gy * (K_DC_GROUP_DIM / K_GROUP_DIM) + gy;
        let ac_group_idx = 2 + dim.num_dc_groups + image_gy * dim.xsize_groups + image_gx;

        // Group rect in pixels (clamped).
        let group_x0 = image_gx * K_GROUP_DIM;
        let group_y0 = image_gy * K_GROUP_DIM;
        let group_xsize = K_GROUP_DIM.min(dim.xsize.saturating_sub(group_x0));
        let group_ysize = K_GROUP_DIM.min(dim.ysize.saturating_sub(group_y0));
        let group_ysize_tiles = div_ceil(group_ysize, K_TILE_DIM);

        // num_nzeros is per AC group, 32x32 blocks.
        let mut num_nzeros = Image3B::new(K_GROUP_DIM_IN_BLOCKS, K_GROUP_DIM_IN_BLOCKS);

        // Process stripe by stripe.
        for ty in 0..group_ysize_tiles {
            let stripe_x0 = group_x0;
            let stripe_y0 = group_y0 + ty * K_TILE_DIM;
            let stripe_xsize = group_xsize;
            let stripe_ysize = K_TILE_DIM.min(dim.ysize.saturating_sub(stripe_y0));
            let stripe_xsize_padded = div_ceil(stripe_xsize, K_BLOCK_DIM) * K_BLOCK_DIM;
            let stripe_ysize_padded = div_ceil(stripe_ysize, K_BLOCK_DIM) * K_BLOCK_DIM;

            // Build the stripe image (XYB).
            let stripe = build_stripe(
                opsin,
                stripe_x0,
                stripe_y0,
                stripe_xsize,
                stripe_ysize,
                stripe_xsize_padded,
                stripe_ysize_padded,
            );

            // stripe_brect is the block-rectangle of this stripe WITHIN THE DC GROUP.
            let stripe_brect_x0 = gx * K_GROUP_DIM_IN_BLOCKS;
            let stripe_brect_y0 = gy * K_GROUP_DIM_IN_BLOCKS + ty * K_TILE_DIM_IN_BLOCKS;
            let stripe_brect = Rect::new(
                stripe_brect_x0,
                stripe_brect_y0,
                stripe_xsize_padded / K_BLOCK_DIM,
                stripe_ysize_padded / K_BLOCK_DIM,
            );

            write_ac_group(
                &stripe,
                stripe_brect,
                matrices,
                distp.scale,
                distp.scale_dc,
                distp.x_qm_scale,
                &mut dc_data,
                ac_code,
                &mut num_nzeros,
                &mut sections[ac_group_idx],
            );
        }

        // Per-AC-group modular alpha subbitstream, written AFTER all AC
        // coefficient stripes so it appears after the coefficients in the
        // bitstream, matching what the decoder expects.  No-op for small images
        // (those already have all alpha data in the LfGlobal section).
        if let Some(alpha_plane) = alpha {
            let abs_group_id = image_gy * dim.xsize_groups + image_gx;
            crate::modular::write_ac_group_alpha(
                alpha_plane,
                dim.xsize,
                dim.ysize,
                group_x0,
                group_y0,
                group_xsize,
                group_ysize,
                abs_group_id,
                dim.num_dc_groups,
                dim.num_groups,
                &mut sections[ac_group_idx],
            );
        }
    }

    // DC group section.
    let dc_group_idx = 1 + dc_gy * dim.xsize_dc_groups + dc_gx;
    write_dc_group(&dc_data, dc_code, &mut sections[dc_group_idx]);
}

/// Carve a stripe out of the (already-XYB-converted, gaborized) opsin image,
/// padding to whole blocks by edge-replication.
fn build_stripe(
    opsin: &Image3F,
    x0: usize,
    y0: usize,
    xsize: usize,
    ysize: usize,
    xsize_padded: usize,
    ysize_padded: usize,
) -> Image3F {
    let mut stripe = Image3F::new(xsize_padded, ysize_padded);
    for c in 0..3 {
        // Copy actual content.
        for y in 0..ysize {
            let src_row = opsin.plane_row(c, y0 + y);
            let dst_row = stripe.plane_row_mut(c, y);
            dst_row[..xsize].copy_from_slice(&src_row[x0..x0 + xsize]);
            let last = dst_row[xsize - 1];
            for x in xsize..xsize_padded {
                dst_row[x] = last;
            }
        }
        // Replicate bottom row.
        for y in ysize..ysize_padded {
            let (src, dst) = stripe.plane_mut(c).two_rows_mut_safe(ysize - 1, y);
            dst.copy_from_slice(src);
        }
    }
    stripe
}
