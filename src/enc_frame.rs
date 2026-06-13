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

use crate::ac_context::{K_COMPACT_BLOCK_CONTEXT_MAP, K_NUM_AC_CONTEXTS};
use crate::bit_writer::BitWriter;
use crate::dc_group_data::{DcGroupData, STRATEGY_DCT, STRATEGY_DCT4X4};
use crate::enc_group::write_ac_group;
use crate::enc_xyb::to_xyb;
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    EntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code, write_token,
};
use crate::image::{Image3B, Image3F, Image3S, Rect};
use crate::quant_weights::DequantMatrices;
use crate::static_entropy_codes::{
    K_CONTEXT_TREE_TOKENS, K_GRADIENT_CONTEXT_LUT, K_NUM_DC_CONTEXTS,
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

impl ImageDim {
    fn new(xsize: usize, ysize: usize) -> Self {
        let xsize_blocks = xsize.div_ceil(K_BLOCK_DIM);
        let ysize_blocks = ysize.div_ceil(K_BLOCK_DIM);
        let xsize_groups = xsize.div_ceil(K_GROUP_DIM);
        let ysize_groups = ysize.div_ceil(K_GROUP_DIM);
        let xsize_dc_groups = xsize.div_ceil(K_DC_GROUP_DIM);
        let ysize_dc_groups = ysize.div_ceil(K_DC_GROUP_DIM);
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

fn quant_dc(distance: f32) -> f32 {
    let k_dc_quant_pow = 0.57f32;
    let k_dc_quant = 1.12f32;
    let k_dc_mul = 2.9f32;
    let effective = k_dc_mul * (distance / k_dc_mul).powf(k_dc_quant_pow);
    let effective = f32::clamp(effective, 0.5 * distance, distance);
    (k_dc_quant / effective).min(50.0)
}

fn compute_distance_params(distance: f32) -> DistanceParams {
    const K_GLOBAL_SCALE_DENOM: i32 = 1 << 16;
    const K_GLOBAL_SCALE_NUMERATOR: i32 = 4096;
    const K_AC_QUANT: f32 = 0.8;
    const K_QUANT_FIELD_TARGET: f32 = 5.0;

    let qdc = quant_dc(distance);
    let mut scale = K_GLOBAL_SCALE_DENOM as f32 * K_AC_QUANT / (distance * K_QUANT_FIELD_TARGET);
    scale = f32::clamp(scale, 1.0, (1 << 15) as f32);
    let scaled_quant_dc = (qdc * K_GLOBAL_SCALE_NUMERATOR as f32 * 1.6) as i32;
    let global_scale = i32::clamp(scale as i32, 1, scaled_quant_dc);
    let scale_f = global_scale as f32 / K_GLOBAL_SCALE_DENOM as f32;
    let qd = ((qdc / scale_f) + 0.5) as i32;
    let qd = i32::clamp(qd, 1, 1 << 16);
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
        gab_enabled: distance > 1.412,
    }
}

#[inline]
fn clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let mn = n.min(w);
    let mx = n.max(w);
    let g = (n as i64 + w as i64 - l as i64) as i32;
    g.clamp(mn, mx)
}

/// Emit DC tokens for one DC group (in channel order Y, X, B).
/// Same as write_dc_tokens, but returns the tokens instead of writing them.
/// Use this to build adaptive entropy codes from the actual token distribution
/// before committing the bit pattern.
fn collect_dc_tokens(dc_data: &DcGroupData) -> Vec<Token> {
    let mut tokens = Vec::new();

    for c in [1usize, 0, 2] {
        let plane = dc_data.quant_dc.plane(c);
        let ysize = plane.ysize();
        let xsize = plane.xsize();
        for y in 0..ysize {
            let grow_row = plane.row(y);
            for (x, &qrow_here) in grow_row[..xsize].iter().enumerate() {
                let row_above = if y > 0 { Some(plane.row(y - 1)) } else { None };
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
                let grad_prop = i64::clamp(
                    K_GRAD_RANGE_MID + top + left - topleft,
                    K_GRAD_RANGE_MIN,
                    K_GRAD_RANGE_MAX,
                );
                let residual = qrow_here as i32 - guess;
                let ctx_id = K_GRADIENT_CONTEXT_LUT[grad_prop as usize] as u32;
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
            }
        }
    }
    tokens
}

/// AC metadata: ytox/ytob CfL maps (all 0), AC strategy (all 0 = DCT-8x8),
/// quant field residuals (all 0), and EPF (token (0, PackSigned(4)) per block).
///
/// In libjxl-tiny ALL four sub-streams use the same shared dc_code.
/// Same as write_ac_metadata_tokens, but returns the tokens. Mirror of
/// collect_dc_tokens for the AC metadata (YtoX/B, ACS, QF, EPF).
fn collect_ac_metadata_tokens(dc_data: &DcGroupData) -> Vec<Token> {
    let mut tokens = Vec::new();
    let xsize_blocks = dc_data.ac_strategy.xsize();
    let ysize_blocks = dc_data.ac_strategy.ysize();
    let xtiles = dc_data.ytox_map.xsize();
    let ytiles = dc_data.ytox_map.ysize();

    // (a) YtoX and YtoB tokens with gradient prediction.
    for c in 0..2usize {
        let cfl_map = if c == 0 {
            &dc_data.ytox_map
        } else {
            &dc_data.ytob_map
        };
        for y in 0..ytiles {
            let cfl_row = cfl_map.row(y);
            for (x, &here) in cfl_row[..xtiles].iter().enumerate() {
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
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
            }
        }
    }

    // (b) AC strategy tokens.
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
            tokens.push(Token::new(ctx_id, pack_signed(cur)));
            left = cur;
        }
    }
    // (c) QF residuals.
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
            tokens.push(Token::new(ctx_id, pack_signed(residual)));
            left = cur;
        }
    }
    // (d) EPF tokens.
    let nblocks = xsize_blocks * ysize_blocks;
    for _ in 0..nblocks {
        tokens.push(Token::new(0, pack_signed(4)));
    }
    tokens
}

/// Real prefix-code bit cost of a DC group's AC-metadata token stream under a
/// freshly optimized entropy code. Used by the DCT4X4 activation gate to weigh
/// the meta-stream cost of the selected DCT4X4 blocks against their RD benefit.
fn meta_entropy_cost(dc_data: &DcGroupData) -> u64 {
    let toks = collect_ac_metadata_tokens(dc_data);
    let code_owned = optimize_entropy_code(&toks, K_NUM_DC_CONTEXTS);
    let code = code_owned.as_ref();
    let mut bits = 0u64;
    for t in &toks {
        let (tok, nbits, _b) = crate::entropy::uint_encode(t.value);
        let pc = &code.prefix_codes[code.context_map[t.context as usize] as usize];
        bits += if pc.single_symbol {
            nbits as u64
        } else {
            pc.depths[tok as usize] as u64 + nbits as u64
        };
    }
    bits
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

fn write_frame_header(
    x_qm_scale: u32,
    epf_iters: u32,
    gab_enabled: bool,
    has_alpha: bool,
    coeff_shifts: &[u32],
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
    // Passes bundle (jxl-frame header.rs:127-132):
    //   num_passes: U32(1,2,3,4+u(3))   default 1
    //   if num_passes != 1:
    //     num_ds:    U32(0,1,2,3+u(1))  -> 0 (no downsampling)
    //     shift:     Vec[u(2)] len (num_passes-1)
    //     downsample/last_pass: Vec len num_ds (empty here)
    // For VarDCT AC the per-pass coeff_shift used by the decoder is
    // passes.shift[pass]; the last pass implicitly has shift 0, so we only emit
    // shifts for passes 0..num_passes-1 and require coeff_shifts.last()==0.
    let num_passes = coeff_shifts.len();
    debug_assert!((1..=11).contains(&num_passes), "num_passes out of range");
    debug_assert!(num_passes == 1 || coeff_shifts[num_passes - 1] == 0);
    if num_passes == 1 {
        w.write(2, 0); // num_passes = 1 (U32 selector 0)
    } else {
        // U32(1,2,3,4+u(3)): selectors 0->1, 1->2, 2->3, 3->4+u(3).
        match num_passes {
            2 => w.write(2, 1),
            3 => w.write(2, 2),
            n => {
                w.write(2, 3);
                w.write(3, (n - 4) as u64);
            }
        }
        w.write(2, 0); // num_ds = 0 (U32 selector 0)
        for &s in &coeff_shifts[..num_passes - 1] {
            w.write(2, s as u64); // shift[p] = coeff_shift of pass p (u(2))
        }
    }
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
        let empty_freqs: [Vec<u16>; 0] = [];
        let empty_syms: [Vec<crate::entropy::AnsEncSymbolInfo>; 0] = [];
        let cm_entropy = EntropyCode {
            context_map: &K_COMPACT_BLOCK_CONTEXT_MAP,
            num_contexts: K_COMPACT_BLOCK_CONTEXT_MAP.len(),
            prefix_codes: &empty_codes,
            num_prefix_codes: 0,
            orig_context_map: None,
            orig_num_contexts: 0,
            use_prefix_code: true,
            ans_freqs: &empty_freqs,
            ans_symbols: &empty_syms,
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

fn write_ac_global(
    num_groups: usize,
    ac_codes: &[crate::entropy::OwnedEntropyCode],
    lz_code: &crate::entropy::OwnedEntropyCode,
    use_lz77: bool,
    w: &mut BitWriter,
) {
    w.write(1, 1);
    if num_groups > 1 {
        let bits = 32
            - (num_groups as u32).leading_zeros()
            - if num_groups.is_power_of_two() { 1 } else { 0 };
        if bits != 0 {
            w.write(bits as usize, 0);
        }
    }
    // HfGlobal parses `num_passes` HfPass blocks (jxl-frame hf_global.rs:57-59),
    // each = used_orders(U32 sel 3 + u(13)=0 -> natural order) + hf_dist entropy
    // code. Each pass gets its own code (ac_codes[p]); the single-pass LZ77 path
    // instead writes the LZ code in its one HfPass.
    for code in ac_codes {
        w.write(2, 3);
        w.write(13, 0);
        if use_lz77 {
            crate::enc_lz77_ac::write_ac_lz_header_and_code(lz_code, w);
        } else {
            w.write(1, 0);
            write_entropy_code(&code.as_ref(), w);
        }
    }
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
    coeff_shifts: &[u32],
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let dim = ImageDim::new(linear.xsize(), linear.ysize());
    let distp = compute_distance_params(distance);
    let matrices = DequantMatrices::new();

    // Progressive lossy splits each quantized AC coeff across `num_passes`
    // passes by a decreasing per-pass shift (last = 0). The decoder reconstructs
    // C = sum_p (sent_p << shift_p) (jxl-vardct hf_coeff.rs:185,191).
    let num_passes = coeff_shifts.len();

    let mut opsin = linear.clone();
    to_xyb(&mut opsin);
    if distp.gab_enabled {
        crate::gaborish::gaborish_inverse(&mut opsin, 0.990_851_1);
    }

    let num_sections = 2 + dim.num_dc_groups + num_passes * dim.num_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // Phase 1: process each DC group, computing dc_data and collecting
    // per-AC-group token vectors. Nothing is written yet. Each DC group is
    // fully independent (reads `opsin` read-only, produces owned results), so
    // for multi-group images we farm groups out across threads. For a single
    // DC group (images <= 2048 px), the threads instead parallelize the
    // AC-strategy selection *inside* `process_dc_group`. Either way the result
    // sequence is deterministic, so output is bit-identical to single-threaded.
    let mut dc_datas: Vec<DcGroupData> = Vec::with_capacity(dim.num_dc_groups);
    let mut all_pending: Vec<PendingAcGroup> = Vec::new();

    let group_coords: Vec<(usize, usize)> = (0..dim.ysize_dc_groups)
        .flat_map(|gy| (0..dim.xsize_dc_groups).map(move |gx| (gx, gy)))
        .collect();

    if num_threads > 1 && group_coords.len() > 1 {
        // Multi-group: parallelize across DC groups; selection inside stays
        // serial (inner thread budget = 1) to avoid oversubscription.
        let nthreads = num_threads.min(group_coords.len());
        let chunk_size = group_coords.len().div_ceil(nthreads);
        let chunks: Vec<&[(usize, usize)]> = group_coords.chunks(chunk_size).collect();
        let results: Vec<Vec<(DcGroupData, Vec<PendingAcGroup>)>> = std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    s.spawn(|| {
                        chunk
                            .iter()
                            .map(|&(dc_gx, dc_gy)| {
                                process_dc_group(
                                    linear,
                                    &opsin,
                                    &dim,
                                    &distp,
                                    &matrices,
                                    dc_gx,
                                    dc_gy,
                                    num_passes,
                                    coeff_shifts,
                                    1,
                                )
                            })
                            .collect()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for chunk_result in results {
            for (dc_data, mut pending) in chunk_result {
                dc_datas.push(dc_data);
                all_pending.append(&mut pending);
            }
        }
    } else {
        // Single group (or single-threaded): selection parallelizes internally.
        for &(dc_gx, dc_gy) in &group_coords {
            let (dc_data, mut pending) = process_dc_group(
                linear,
                &opsin,
                &dim,
                &distp,
                &matrices,
                dc_gx,
                dc_gy,
                num_passes,
                coeff_shifts,
                num_threads,
            );
            dc_datas.push(dc_data);
            all_pending.append(&mut pending);
        }
    }

    // Phase 2: build adaptive DC entropy code from all DC + AC-metadata tokens.
    let mut dc_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(dim.num_dc_groups);
    let mut meta_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(dim.num_dc_groups);
    let mut all_dc_tokens: Vec<Token> = Vec::new();
    for dc_data in &dc_datas {
        let dc_t = collect_dc_tokens(dc_data);
        let mt_t = collect_ac_metadata_tokens(dc_data);
        all_dc_tokens.extend_from_slice(&dc_t);
        all_dc_tokens.extend_from_slice(&mt_t);
        dc_tokens_per_group.push(dc_t);
        meta_tokens_per_group.push(mt_t);
    }
    let dc_code_owned = optimize_entropy_code(&all_dc_tokens, K_NUM_DC_CONTEXTS);
    let dc_code = dc_code_owned.as_ref();

    let ac_num_contexts = K_NUM_AC_CONTEXTS + 1;

    // Per-pass aggregated tokens -> per-pass entropy code. Pass 0 (coarse) and
    // the residual pass(es) have very different token distributions, so a single
    // shared code is wasteful; each HfPass gets a code built from its own tokens.
    let mut pass_tokens_agg: Vec<Vec<Token>> = vec![Vec::new(); num_passes];
    for pg in &all_pending {
        for (p, pass_tokens) in pg.tokens.iter().enumerate() {
            pass_tokens_agg[p].extend_from_slice(pass_tokens);
        }
    }
    let ac_code_per_pass: Vec<crate::entropy::OwnedEntropyCode> = pass_tokens_agg
        .iter()
        .map(|toks| crate::entropy::optimize_entropy_code_ac(toks, K_NUM_AC_CONTEXTS))
        .collect();

    // LZ77 path is single-pass only for now: it compresses one token stream per
    // group. Multi-pass uses the per-pass plain codes.
    let mut ac_lz_per_group: Vec<Vec<crate::enc_lz77_ac::AcLz>> = Vec::new();
    let ac_lz_code_owned;
    let use_lz77;
    if num_passes == 1 {
        ac_lz_per_group = Vec::with_capacity(all_pending.len());
        for pg in &all_pending {
            ac_lz_per_group.push(crate::enc_lz77_ac::lz77_compress_ac(&pg.tokens[0]));
        }
        ac_lz_code_owned = crate::enc_lz77_ac::build_ac_lz_code(&ac_lz_per_group, ac_num_contexts);
        let lz_bits = crate::enc_lz77_ac::estimate_ac_lz_bits(
            &ac_lz_per_group,
            &ac_lz_code_owned,
            ac_num_contexts,
        );
        let plain_bits =
            crate::enc_lz77_ac::estimate_ac_plain_bits(&pass_tokens_agg[0], &ac_code_per_pass[0]);
        // Require a real margin to cover the LZ77 header + distance-context cost.
        use_lz77 = lz_bits + 512 < plain_bits;
    } else {
        ac_lz_code_owned = crate::enc_lz77_ac::build_ac_lz_code(&ac_lz_per_group, ac_num_contexts);
        use_lz77 = false;
    }

    // Phase 4: write DC global with adaptive DC code.
    write_dc_global(
        &distp,
        dim.num_dc_groups,
        &dc_code,
        alpha,
        dim.xsize,
        dim.ysize,
        &mut sections[0],
    );

    // Phase 5: write each DC group section with adaptive DC code.
    for (i, dc_data) in dc_datas.iter().enumerate() {
        let dc_group_idx = 1 + i;
        let w = &mut sections[dc_group_idx];
        w.write(2, 0); // extra_dc_precision
        w.write(4, 3); // use global tree, default wp, no transforms
        for t in &dc_tokens_per_group[i] {
            write_token(*t, &dc_code, w);
        }
        let num_blocks = dc_data.ac_strategy.xsize() * dc_data.ac_strategy.ysize();
        let num_ac_blocks = dc_data.ac_strategy.count_first_blocks();
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
        for t in &meta_tokens_per_group[i] {
            write_token(*t, &dc_code, w);
        }
    }

    // Phase 6: AC global. One HfPass per pass (each with its own code), or a
    // single HfPass carrying the LZ77 code in the single-pass case.
    write_ac_global(
        dim.num_groups,
        &ac_code_per_pass,
        &ac_lz_code_owned,
        use_lz77,
        &mut sections[1 + dim.num_dc_groups],
    );

    // Phase 7: write each (pass, group) AC section. Section index for
    // (pass, group) = 2 + num_dc_groups + pass*num_groups + group_idx
    // (jxl-frame toc.rs:196-200). With LZ77 (single-pass only) we emit the
    // compressed stream; otherwise raw tokens via the shared plain code.
    for (i, pg) in all_pending.iter().enumerate() {
        for (pass, pass_tokens) in pg.tokens.iter().enumerate() {
            let section_idx = 2 + dim.num_dc_groups + pass * dim.num_groups + pg.group_idx;
            let w = &mut sections[section_idx];
            if use_lz77 {
                for t in &ac_lz_per_group[i] {
                    crate::enc_lz77_ac::write_ac_lz(*t, &ac_lz_code_owned, ac_num_contexts, w);
                }
            } else {
                let code_ref = ac_code_per_pass[pass].as_ref();
                if code_ref.use_prefix_code {
                    for t in pass_tokens {
                        write_token(*t, &code_ref, w);
                    }
                } else {
                    // rANS: the whole group's tokens are encoded as one LIFO unit.
                    crate::entropy::write_ans_tokens(
                        pass_tokens,
                        code_ref.context_map,
                        code_ref.ans_symbols,
                        w,
                    );
                }
            }
        }
    }
    // Modular alpha: extra-channel modular data is decoded in the last pass
    // (the only pass whose modular sub-image shift range is set when num_ds=0,
    // jxl-frame lib.rs:101-108 / pass_group.rs:93). Write it into that section.
    if let Some(alpha_plane) = alpha {
        let last_pass = num_passes - 1;
        for image_gy in 0..dim.ysize_groups {
            for image_gx in 0..dim.xsize_groups {
                let group_x0 = image_gx * K_GROUP_DIM;
                let group_y0 = image_gy * K_GROUP_DIM;
                let group_xsize = K_GROUP_DIM.min(dim.xsize.saturating_sub(group_x0));
                let group_ysize = K_GROUP_DIM.min(dim.ysize.saturating_sub(group_y0));
                let abs_group_id = image_gy * dim.xsize_groups + image_gx;
                let ac_group_idx =
                    2 + dim.num_dc_groups + last_pass * dim.num_groups + abs_group_id;
                crate::modular::write_ac_group_alpha(
                    alpha_plane,
                    dim.xsize,
                    dim.ysize,
                    group_x0,
                    group_y0,
                    group_xsize,
                    group_ysize,
                    &mut sections[ac_group_idx],
                );
            }
        }
    }

    write_frame_header(
        distp.x_qm_scale,
        distp.epf_iters,
        distp.gab_enabled,
        alpha.is_some(),
        coeff_shifts,
        writer,
    );
    combine_sections(&mut sections, writer);
}

/// Per-AC-group buffered tokens. For progressive (multi-pass) encoding the
/// quantized AC coefficients of each block are split across passes; `tokens`
/// holds one token stream per pass. `group_idx` is the raster group index
/// (0..num_groups); the section for (pass, group) is
/// `2 + num_dc_groups + pass*num_groups + group_idx`.
pub(crate) struct PendingAcGroup {
    pub group_idx: usize,
    pub tokens: Vec<Vec<Token>>,
}

#[allow(clippy::too_many_arguments)]
fn process_dc_group(
    _linear: &Image3F,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    matrices: &DequantMatrices,
    dc_gx: usize,
    dc_gy: usize,
    num_passes: usize,
    coeff_shifts: &[u32],
    num_threads: usize,
) -> (DcGroupData, Vec<PendingAcGroup>) {
    // DC group rect in pixels (clamped to image bounds).
    let dc_group_x0 = dc_gx * K_DC_GROUP_DIM;
    let dc_group_y0 = dc_gy * K_DC_GROUP_DIM;
    let dc_group_xsize = K_DC_GROUP_DIM.min(dim.xsize.saturating_sub(dc_group_x0));
    let dc_group_ysize = K_DC_GROUP_DIM.min(dim.ysize.saturating_sub(dc_group_y0));
    let dc_group_xsize_blocks = dc_group_xsize.div_ceil(K_BLOCK_DIM);
    let dc_group_ysize_blocks = dc_group_ysize.div_ceil(K_BLOCK_DIM);
    let dc_group_xsize_groups = dc_group_xsize.div_ceil(K_GROUP_DIM);
    let dc_group_ysize_groups = dc_group_ysize.div_ceil(K_GROUP_DIM);

    let mut dc_data = DcGroupData::new(dc_group_xsize_blocks, dc_group_ysize_blocks);

    crate::adaptive_quant::fill_quant_field(
        opsin,
        &mut dc_data.raw_quant_field,
        dc_group_x0,
        dc_group_y0,
        distp.distance,
        1.0 / distp.scale,
    );
    dc_data.dct4x4_benefit = crate::enc_ac_strategy::fill_ac_strategy(
        opsin,
        dc_group_x0,
        dc_group_y0,
        distp.distance,
        distp.scale,
        distp.x_qm_scale,
        matrices,
        &mut dc_data.raw_quant_field,
        &mut dc_data.ac_strategy,
        num_threads,
    );
    // Per-tile CfL: find optimal Y→X and Y→B slopes per 64×64 tile, written
    // into ytox_map/ytob_map and applied during DCT in write_ac_group.
    crate::enc_color_correlation::fill_cmap(
        opsin,
        matrices,
        dc_group_x0 / K_BLOCK_DIM,
        dc_group_y0 / K_BLOCK_DIM,
        dc_group_xsize_blocks,
        dc_group_ysize_blocks,
        &mut dc_data.ytox_map,
        &mut dc_data.ytob_map,
    );

    // DCT4X4 activation gate. `fill_ac_strategy` greedily commits every block
    // where DCT4X4 wins the per-block RD comparison, but a sparse set of DCT4X4
    // blocks can disrupt the prefix-code clustering of the (otherwise nearly
    // free) AC-strategy meta stream — most visibly the constant EPF context,
    // which loses its zero-length codeword. Measure the *real* meta-token cost
    // with the selected DCT4X4 blocks vs with them reverted to DCT8, and keep
    // DCT4X4 only when the accumulated RD benefit covers the meta-cost increase
    // (1 bit of rate contributes RD_LAMBDA to the cost). Done before
    // `write_ac_group`, so the reverted strategy drives coefficient computation.
    {
        let mut positions: Vec<(usize, usize)> = Vec::new();
        for y in 0..dc_data.ac_strategy.ysize() {
            for x in 0..dc_data.ac_strategy.xsize() {
                if dc_data.ac_strategy.is_first_block(x, y)
                    && dc_data.ac_strategy.raw_strategy(x, y) == STRATEGY_DCT4X4
                {
                    positions.push((x, y));
                }
            }
        }
        if !positions.is_empty() {
            let cost_with = meta_entropy_cost(&dc_data);
            for &(x, y) in &positions {
                dc_data.ac_strategy.set_first(x, y, STRATEGY_DCT);
            }
            let cost_without = meta_entropy_cost(&dc_data);
            let meta_delta = cost_with.saturating_sub(cost_without) as f32;
            if dc_data.dct4x4_benefit > crate::enc_ac_strategy::RD_LAMBDA * meta_delta {
                // Worth it: restore the DCT4X4 selection.
                for &(x, y) in &positions {
                    dc_data.ac_strategy.set_first(x, y, STRATEGY_DCT4X4);
                }
            }
            // else: leave reverted to DCT8.
        }
    }

    let num_groups_here = dc_group_xsize_groups * dc_group_ysize_groups;
    let mut pending: Vec<PendingAcGroup> = Vec::with_capacity(num_groups_here);

    // Each AC group is independent: it reads `dc_data` (strategy / quant / cmap)
    // read-only and writes only its own 32×32-block DC region. Transforms never
    // cross a group boundary (`can_place_strategy`), so the regions are disjoint
    // and merge deterministically — output is identical to single-threaded.
    let merge_quant_dc = |dc: &mut DcGroupData, gx: usize, gy: usize, local: &Image3S| {
        let ox = gx * K_GROUP_DIM_IN_BLOCKS;
        let oy = gy * K_GROUP_DIM_IN_BLOCKS;
        let gwb = local.xsize();
        let ghb = local.ysize();
        for c in 0..3 {
            for ly in 0..ghb {
                let src = local.plane_row(c, ly);
                dc.quant_dc.plane_row_mut(c, oy + ly)[ox..ox + gwb].copy_from_slice(&src[..gwb]);
            }
        }
    };

    if num_threads > 1 && num_groups_here > 1 {
        let nthreads = num_threads.min(num_groups_here);
        let chunk_size = num_groups_here.div_ceil(nthreads);
        let index_chunks: Vec<Vec<usize>> = (0..num_groups_here)
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();
        let dc_ref = &dc_data;
        let results: Vec<Vec<(usize, PendingAcGroup, Image3S)>> = std::thread::scope(|s| {
            let handles: Vec<_> = index_chunks
                .into_iter()
                .map(|chunk| {
                    s.spawn(move || {
                        chunk
                            .into_iter()
                            .map(|gix| {
                                let gx = gix % dc_group_xsize_groups;
                                let gy = gix / dc_group_xsize_groups;
                                let (p, local) = process_ac_group(
                                    opsin,
                                    dim,
                                    distp,
                                    matrices,
                                    dc_ref,
                                    num_passes,
                                    coeff_shifts,
                                    dc_gx,
                                    dc_gy,
                                    gx,
                                    gy,
                                );
                                (gix, p, local)
                            })
                            .collect()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // Chunks are contiguous index ranges, so flattening preserves gix order.
        for chunk_result in results {
            for (gix, p, local) in chunk_result {
                let gx = gix % dc_group_xsize_groups;
                let gy = gix / dc_group_xsize_groups;
                merge_quant_dc(&mut dc_data, gx, gy, &local);
                pending.push(p);
            }
        }
    } else {
        for gix in 0..num_groups_here {
            let gx = gix % dc_group_xsize_groups;
            let gy = gix / dc_group_xsize_groups;
            let (p, local) = process_ac_group(
                opsin,
                dim,
                distp,
                matrices,
                &dc_data,
                num_passes,
                coeff_shifts,
                dc_gx,
                dc_gy,
                gx,
                gy,
            );
            merge_quant_dc(&mut dc_data, gx, gy, &local);
            pending.push(p);
        }
    }
    (dc_data, pending)
}

/// Encode a single AC group: build its tile stripes, quantize and tokenize,
/// and place its DC coefficients into a returned group-local `quant_dc`
/// (origin-relative, merged by the caller). Reads `dc_data` read-only.
#[allow(clippy::too_many_arguments)]
fn process_ac_group(
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    matrices: &DequantMatrices,
    dc_data: &DcGroupData,
    num_passes: usize,
    coeff_shifts: &[u32],
    dc_gx: usize,
    dc_gy: usize,
    gx: usize,
    gy: usize,
) -> (PendingAcGroup, Image3S) {
    let image_gx = dc_gx * (K_DC_GROUP_DIM / K_GROUP_DIM) + gx;
    let image_gy = dc_gy * (K_DC_GROUP_DIM / K_GROUP_DIM) + gy;
    let group_x0 = image_gx * K_GROUP_DIM;
    let group_y0 = image_gy * K_GROUP_DIM;
    let group_xsize = K_GROUP_DIM.min(dim.xsize.saturating_sub(group_x0));
    let group_ysize = K_GROUP_DIM.min(dim.ysize.saturating_sub(group_y0));
    let group_ysize_tiles = group_ysize.div_ceil(K_TILE_DIM);
    let gwb = group_xsize.div_ceil(K_BLOCK_DIM);
    let ghb = group_ysize.div_ceil(K_BLOCK_DIM);
    let qorigin_x = gx * K_GROUP_DIM_IN_BLOCKS;
    let qorigin_y = gy * K_GROUP_DIM_IN_BLOCKS;

    let mut local_quant_dc = Image3S::new(gwb, ghb);
    let mut num_nzeros: Vec<Image3B> = (0..num_passes)
        .map(|_| Image3B::new(K_GROUP_DIM_IN_BLOCKS, K_GROUP_DIM_IN_BLOCKS))
        .collect();
    let mut tokens: Vec<Vec<Token>> = (0..num_passes)
        .map(|_| Vec::with_capacity(K_GROUP_DIM_IN_BLOCKS * K_GROUP_DIM_IN_BLOCKS * 4))
        .collect();

    for ty in 0..group_ysize_tiles {
        let stripe_x0 = group_x0;
        let stripe_y0 = group_y0 + ty * K_TILE_DIM;
        let stripe_xsize = group_xsize;
        let stripe_ysize = K_TILE_DIM.min(dim.ysize.saturating_sub(stripe_y0));
        let stripe_xsize_padded = stripe_xsize.div_ceil(K_BLOCK_DIM) * K_BLOCK_DIM;
        let stripe_ysize_padded = stripe_ysize.div_ceil(K_BLOCK_DIM) * K_BLOCK_DIM;

        let stripe = build_stripe(
            opsin,
            stripe_x0,
            stripe_y0,
            stripe_xsize,
            stripe_ysize,
            stripe_xsize_padded,
            stripe_ysize_padded,
        );

        let stripe_brect = Rect::new(
            qorigin_x,
            qorigin_y + ty * K_TILE_DIM_IN_BLOCKS,
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
            dc_data,
            &mut local_quant_dc,
            qorigin_x,
            qorigin_y,
            &mut num_nzeros,
            coeff_shifts,
            &mut tokens,
        );
    }

    (
        PendingAcGroup {
            group_idx: image_gy * dim.xsize_groups + image_gx,
            tokens,
        },
        local_quant_dc,
    )
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
            let (data, padding) = dst_row.split_at_mut(xsize);
            data.copy_from_slice(&src_row[x0..x0 + xsize]);
            let last = *data.last().unwrap();
            padding[..xsize_padded - xsize].fill(last);
        }
        // Replicate bottom row.
        for y in ysize..ysize_padded {
            let (src, dst) = stripe.plane_mut(c).two_rows_mut_safe(ysize - 1, y);
            dst.copy_from_slice(src);
        }
    }
    stripe
}
