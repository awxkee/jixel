/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

use crate::adaptive_quant::{AQ_MAP_SCRATCH, K_AC_QUANT};
use core::arch::wasm32::*;

const MATCH_GAMMA_OFFSET: f32 = 0.019;
const MASKING_SQRT_MUL_V: f32 = 145487.346437769899962;

#[inline]
#[target_feature(enable = "simd128")]
fn load4s(s: &[f32], i: usize) -> v128 {
    unsafe { v128_load(s[i..].as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
fn store4(v: v128, s: &mut [f32], i: usize) {
    unsafe {
        v128_store(s[i..].as_mut_ptr().cast(), v);
    }
}

/// `a * b + c`.
///
/// WASM SIMD128 does not expose a portable fused f32x4 FMA intrinsic, so this
/// intentionally maps to mul+add while preserving the same call shape as the
/// NEON/SSE/AVX backends.
#[inline]
#[target_feature(enable = "simd128")]
fn vmlaf(a: v128, b: v128, c: v128) -> v128 {
    f32x4_add(c, f32x4_mul(a, b))
}

#[inline]
#[target_feature(enable = "simd128")]
fn pairwise_add_f32x4(a: v128, b: v128) -> v128 {
    let even = i8x16_shuffle::<
        0,
        1,
        2,
        3, // a0
        8,
        9,
        10,
        11, // a2
        16,
        17,
        18,
        19, // b0
        24,
        25,
        26,
        27, // b2
    >(a, b);
    let odd = i8x16_shuffle::<
        4,
        5,
        6,
        7, // a1
        12,
        13,
        14,
        15, // a3
        20,
        21,
        22,
        23, // b1
        28,
        29,
        30,
        31, // b3
    >(a, b);
    f32x4_add(even, odd)
}

#[inline]
#[target_feature(enable = "simd128")]
fn hsum4x4(s0: v128, s1: v128, s2: v128, s3: v128) -> v128 {
    let p01 = pairwise_add_f32x4(s0, s1);
    let p23 = pairwise_add_f32x4(s2, s3);
    pairwise_add_f32x4(p01, p23)
}

#[inline]
#[target_feature(enable = "simd128")]
fn uzp1_f32x4(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<
        0,
        1,
        2,
        3, // a0
        8,
        9,
        10,
        11, // a2
        16,
        17,
        18,
        19, // b0
        24,
        25,
        26,
        27, // b2
    >(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
fn uzp2_f32x4(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<
        4,
        5,
        6,
        7, // a1
        12,
        13,
        14,
        15, // a3
        20,
        21,
        22,
        23, // b1
        28,
        29,
        30,
        31, // b3
    >(a, b)
}

/// Vectorised `ratio_cubic_to_simple_gamma`.
#[inline]
#[target_feature(enable = "simd128")]
fn ratio_cubic_x4(v: v128, invert: bool) -> v128 {
    const K_SG_MUL: f32 = 226.77216153508914;
    const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
    const K_LOG2: f32 = 0.693147181;
    const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_LOG2;
    const K_SG_V_OFFSET: f32 = 7.7825991679894591;
    let k_epsilon = 1e-2f32;
    let k_num_mul = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    let k_v_offset = K_SG_V_OFFSET * K_LOG2 + k_epsilon;
    let k_den_mul = K_LOG2 * K_SG_MUL;

    let v = f32x4_max(v, f32x4_splat(0.0));
    let v2 = f32x4_mul(v, v);
    let num = vmlaf(f32x4_splat(k_num_mul), v2, f32x4_splat(k_epsilon));
    let den = vmlaf(
        f32x4_mul(f32x4_splat(k_den_mul), v),
        v2,
        f32x4_splat(k_v_offset),
    );
    if invert {
        f32x4_div(num, den)
    } else {
        f32x4_div(den, num)
    }
}

/// Vectorised `masking_sqrt`.
#[inline]
#[target_feature(enable = "simd128")]
fn masking_sqrt_x4(v: v128) -> v128 {
    let k_log_offset = 27.505837037000106f32;
    let inner = vmlaf(
        v,
        f32x4_splat(MASKING_SQRT_MUL_V),
        f32x4_splat(k_log_offset),
    );
    f32x4_mul(f32x4_splat(0.25), f32x4_sqrt(inner))
}

#[inline]
#[target_feature(enable = "simd128")]
fn dirty_log2f_x4(d: v128) -> v128 {
    let one = f32x4_splat(1.0);

    // Same range reduction as scalar dirty_log2f(): reduce into
    // [sqrt(2)/2, sqrt(2)] and keep the extracted exponent in n.
    let mut ix = d;
    ix = i32x4_add(ix, i32x4_splat((0x3f80_0000u32 - 0x3f35_04f3u32) as i32));
    let n = i32x4_sub(u32x4_shr(ix, 23), i32x4_splat(0x7f));
    ix = i32x4_add(
        v128_and(ix, i32x4_splat(0x007f_ffff)),
        i32x4_splat(0x3f35_04f3),
    );

    let a = ix;
    let x = f32x4_div(f32x4_sub(a, one), f32x4_add(a, one));
    let x2 = f32x4_mul(x, x);

    let mut u = f32x4_splat(0.4121985850084821691);
    u = vmlaf(u, x2, f32x4_splat(0.5770780163490337802));
    u = vmlaf(u, x2, f32x4_splat(0.9617966939259845749));

    let n = f32x4_convert_i32x4(n);
    let base = vmlaf(x, f32x4_splat(2.8853900817779268), n);
    vmlaf(f32x4_mul(x2, x), u, base)
}

#[inline]
#[target_feature(enable = "simd128")]
fn compute_mask_x4(out_val: v128) -> v128 {
    let k_base = f32x4_splat(-0.7647);
    let k_mul4 = f32x4_splat(9.4708735624378946);
    let k_mul2 = f32x4_splat(17.35036561631863);
    let k_offset2 = f32x4_splat(302.59587815579727);
    let k_mul3 = f32x4_splat(6.7943250517376494);
    let k_offset3 = f32x4_splat(3.7179635626140772);
    let k_offset4 = f32x4_splat(0.25 * 3.7179635626140772);
    let k_mul0 = f32x4_splat(0.80061762862741759);
    let one = f32x4_splat(1.0);

    let v1 = f32x4_max(f32x4_mul(out_val, k_mul0), f32x4_splat(1e-3));
    let v1_sq = f32x4_mul(v1, v1);
    let v2 = f32x4_div(one, f32x4_add(v1, k_offset2));
    let v3 = f32x4_div(one, f32x4_add(v1_sq, k_offset3));
    let v4 = f32x4_div(one, f32x4_add(v1_sq, k_offset4));

    vmlaf(k_mul4, v4, vmlaf(k_mul2, v2, vmlaf(k_mul3, v3, k_base)))
}

#[inline]
#[target_feature(enable = "simd128")]
fn gamma_row_sum_x4(row_x: &[f32], row_y: &[f32], base: usize) -> v128 {
    let bias = f32x4_splat(0.16);
    let half = f32x4_splat(0.5);

    let x0 = load4s(row_x, base);
    let y0 = f32x4_add(load4s(row_y, base), bias);
    let r0 = f32x4_sub(y0, x0);
    let g0 = f32x4_add(y0, x0);
    let sum0 = f32x4_mul(
        half,
        f32x4_add(ratio_cubic_x4(r0, true), ratio_cubic_x4(g0, true)),
    );

    let x1 = load4s(row_x, base + 4);
    let y1 = f32x4_add(load4s(row_y, base + 4), bias);
    let r1 = f32x4_sub(y1, x1);
    let g1 = f32x4_add(y1, x1);
    let sum1 = f32x4_mul(
        half,
        f32x4_add(ratio_cubic_x4(r1, true), ratio_cubic_x4(g1, true)),
    );

    f32x4_add(sum0, sum1)
}

#[inline]
#[target_feature(enable = "simd128")]
fn gamma_modulation_blocks4_x4(
    x: usize,
    y: usize,
    xyb: &crate::image::Image3F,
    out_val: v128,
) -> v128 {
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut acc2 = f32x4_splat(0.0);
    let mut acc3 = f32x4_splat(0.0);

    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        acc0 = f32x4_add(acc0, gamma_row_sum_x4(row_x, row_y, x));
        acc1 = f32x4_add(acc1, gamma_row_sum_x4(row_x, row_y, x + 8));
        acc2 = f32x4_add(acc2, gamma_row_sum_x4(row_x, row_y, x + 16));
        acc3 = f32x4_add(acc3, gamma_row_sum_x4(row_x, row_y, x + 24));
    }

    let overall = f32x4_mul(hsum4x4(acc0, acc1, acc2, acc3), f32x4_splat(1.0 / 64.0));
    vmlaf(
        f32x4_splat(0.1005613337192697),
        dirty_log2f_x4(overall),
        out_val,
    )
}

#[inline]
#[target_feature(enable = "simd128")]
fn hf_row_sum_x4(
    row: &[f32],
    row_next: &[f32],
    base: usize,
    valmin_y: v128,
    right_tail_mask: v128,
    has_vertical: bool,
) -> v128 {
    let p0 = load4s(row, base);
    let p1 = load4s(row, base + 4);

    let right0 = f32x4_min(f32x4_abs(f32x4_sub(p0, load4s(row, base + 1))), valmin_y);
    let right1 = f32x4_mul(
        f32x4_min(f32x4_abs(f32x4_sub(p1, load4s(row, base + 5))), valmin_y),
        right_tail_mask,
    );
    let mut sum = f32x4_add(right0, right1);

    if has_vertical {
        let down0 = f32x4_min(f32x4_abs(f32x4_sub(p0, load4s(row_next, base))), valmin_y);
        let down1 = f32x4_min(
            f32x4_abs(f32x4_sub(p1, load4s(row_next, base + 4))),
            valmin_y,
        );
        sum = f32x4_add(sum, f32x4_add(down0, down1));
    }

    sum
}

#[inline]
#[target_feature(enable = "simd128")]
fn hf_modulation_blocks4_direct_x4(
    x: usize,
    y: usize,
    xyb_y: &crate::image::Image3F,
    out_val: v128,
    strength: f32,
) -> v128 {
    let valmin_y = f32x4_splat(0.0206);

    let mut s0 = f32x4_splat(0.0);
    let mut s1 = f32x4_splat(0.0);
    let mut s2 = f32x4_splat(0.0);
    let mut s3 = f32x4_splat(0.0);

    for dy in 0..8 {
        let row = xyb_y.plane_row(1, y + dy);
        let row_next = if dy == 7 {
            row
        } else {
            xyb_y.plane_row(1, y + dy + 1)
        };

        s0 = f32x4_add(
            s0,
            hf_row_sum_x4(row, row_next, x, valmin_y, f32x4_splat(1.0), dy != 7),
        );
        s1 = f32x4_add(
            s1,
            hf_row_sum_x4(row, row_next, x + 8, valmin_y, f32x4_splat(1.0), dy != 7),
        );
        s2 = f32x4_add(
            s2,
            hf_row_sum_x4(row, row_next, x + 16, valmin_y, f32x4_splat(1.0), dy != 7),
        );
        s3 = f32x4_add(
            s3,
            hf_row_sum_x4(row, row_next, x + 24, valmin_y, f32x4_splat(1.0), dy != 7),
        );
    }

    let out = hsum4x4(s0, s1, s2, s3);
    vmlaf(
        out,
        f32x4_splat(-0.38 * strength),
        f32x4_add(out_val, f32x4_splat(0.42)),
    )
}

#[inline]
#[target_feature(enable = "simd128")]
fn blue_row_sum_x4(row_x: &[f32], row_y: &[f32], row_b: &[f32], base: usize) -> v128 {
    let k_limit = f32x4_splat(0.010474084867598155);
    let k_offset = f32x4_splat(0.0031994768654636393);
    let zero = f32x4_splat(0.0);

    let x0 = load4s(row_x, base);
    let y0 = load4s(row_y, base);
    let b0 = load4s(row_b, base);
    let y_eff0 = f32x4_add(f32x4_add(y0, k_offset), f32x4_abs(x0));
    let s0 = f32x4_min(f32x4_max(f32x4_sub(b0, y_eff0), zero), k_limit);

    let x1 = load4s(row_x, base + 4);
    let y1 = load4s(row_y, base + 4);
    let b1 = load4s(row_b, base + 4);
    let y_eff1 = f32x4_add(f32x4_add(y1, k_offset), f32x4_abs(x1));
    let s1 = f32x4_min(f32x4_max(f32x4_sub(b1, y_eff1), zero), k_limit);

    f32x4_add(s0, s1)
}

#[inline]
#[target_feature(enable = "simd128")]
fn blue_modulation_blocks4_x4(
    x: usize,
    y: usize,
    xyb: &crate::image::Image3F,
    out_val: v128,
) -> v128 {
    let mut s0 = f32x4_splat(0.0);
    let mut s1 = f32x4_splat(0.0);
    let mut s2 = f32x4_splat(0.0);
    let mut s3 = f32x4_splat(0.0);

    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        let row_b = xyb.plane_row(2, y + dy);

        s0 = f32x4_add(s0, blue_row_sum_x4(row_x, row_y, row_b, x));
        s1 = f32x4_add(s1, blue_row_sum_x4(row_x, row_y, row_b, x + 8));
        s2 = f32x4_add(s2, blue_row_sum_x4(row_x, row_y, row_b, x + 16));
        s3 = f32x4_add(s3, blue_row_sum_x4(row_x, row_y, row_b, x + 24));
    }

    const K_LIMIT: f32 = 0.010474084867598155;
    const K_MAX_LIMIT: f32 = 15.463398341612438 * K_LIMIT;
    const SCALE: f32 = 0.90590804735610064;

    let mut sums = hsum4x4(s0, s1, s2, s3);

    let flip_mask = f32x4_ge(sums, f32x4_splat(32.0 * K_LIMIT));
    let flipped = f32x4_sub(f32x4_splat(64.0 * K_LIMIT), sums);
    sums = v128_bitselect(flipped, sums, flip_mask);
    sums = f32x4_min(sums, f32x4_splat(K_MAX_LIMIT));

    vmlaf(sums, f32x4_splat(SCALE), out_val)
}

pub(crate) const EXP2_P0: f32 = 1.00000011920928955078125_f32;
pub(crate) const EXP2_P1: f32 = 0.69314706325531005859375_f32;
pub(crate) const EXP2_P2: f32 = 0.24022041261196136474609375_f32;
pub(crate) const EXP2_P3: f32 = 5.550567805767059326171875e-2_f32;
pub(crate) const EXP2_P4: f32 = 9.678089059889316558837890625e-3_f32;
pub(crate) const EXP2_P5: f32 = 1.33218802511692047119140625e-3_f32;

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) fn vpow2ifq_s32(q: v128) -> v128 {
    i32x4_shl(i32x4_add(q, i32x4_splat(0x7f)), 23)
}

#[inline]
#[target_feature(enable = "simd128")]
fn round_ties_away_f32x4(v: v128) -> v128 {
    let half = f32x4_splat(0.5);
    let zero = f32x4_splat(0.0);
    let positive = f32x4_ge(v, zero);
    let qp = f32x4_floor(f32x4_add(v, half));
    let qn = f32x4_ceil(f32x4_sub(v, half));
    v128_bitselect(qp, qn, positive)
}

#[inline]
#[target_feature(enable = "simd128")]
fn fast_exp2_x4(v: v128) -> v128 {
    // exp2(x) = 2^q * 2^r
    // q = round(x), r = x - q, so r is approximately in [-0.5, 0.5].
    let qf = round_ties_away_f32x4(v);
    let q = i32x4_trunc_sat_f32x4(qf);
    let r = f32x4_sub(v, qf);

    let mut p = f32x4_splat(EXP2_P5);
    p = vmlaf(p, r, f32x4_splat(EXP2_P4));
    p = vmlaf(p, r, f32x4_splat(EXP2_P3));
    p = vmlaf(p, r, f32x4_splat(EXP2_P2));
    p = vmlaf(p, r, f32x4_splat(EXP2_P1));
    p = vmlaf(p, r, f32x4_splat(EXP2_P0));

    let scale = vpow2ifq_s32(q);
    f32x4_mul(p, scale)
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_quant_u8x4(qf_row: &mut [u8], bx: usize, qi: v128) {
    qf_row[bx] = i32x4_extract_lane::<0>(qi).clamp(1, 255) as u8;
    qf_row[bx + 1] = i32x4_extract_lane::<1>(qi).clamp(1, 255) as u8;
    qf_row[bx + 2] = i32x4_extract_lane::<2>(qi).clamp(1, 255) as u8;
    qf_row[bx + 3] = i32x4_extract_lane::<3>(qi).clamp(1, 255) as u8;
}

#[inline]
fn write_quant_scalar_block(
    opsin: &crate::image::Image3F,
    qf_out: &mut u8,
    aq: f32,
    px: usize,
    py: usize,
    img_xsize: usize,
    img_ysize: usize,
    mul: f32,
    add: f32,
    inv_scale: f32,
    hf_strength: f32,
) {
    if px >= img_xsize || py >= img_ysize {
        *qf_out = 1;
        return;
    }

    let bx_px = px.min(img_xsize.saturating_sub(8));
    let by_px = py.min(img_ysize.saturating_sub(8));
    let mask_val = crate::adaptive_quant::compute_mask(aq);
    let mask_val = crate::adaptive_quant::gamma_modulation(bx_px, by_px, opsin, mask_val);
    let out_val = crate::adaptive_quant::hf_modulation(bx_px, by_px, opsin, mask_val, hf_strength);
    let out_val = out_val.min(crate::adaptive_quant::blue_modulation(
        bx_px, by_px, opsin, mask_val,
    ));
    let qf = crate::adaptive_quant::fast_exp2(out_val * 1.442695041) * mul + add;
    let qi = crate::dct::fmla(qf, inv_scale, 0.5) as i32;
    *qf_out = qi.clamp(1, 255) as u8;
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
fn write_quant_row_wasm(
    opsin: &crate::image::Image3F,
    aq_row: &[f32],
    qf_row: &mut [u8],
    x0: usize,
    py: usize,
    img_xsize: usize,
    img_ysize: usize,
    mul: f32,
    add: f32,
    inv_scale: f32,
    hf_strength: f32,
) {
    let xsize_blocks = aq_row.len();
    debug_assert!(qf_row.len() >= xsize_blocks);

    if py >= img_ysize {
        qf_row[..xsize_blocks].fill(1);
        return;
    }

    let valid_blocks = if x0 >= img_xsize {
        0
    } else {
        ((img_xsize - x0 + 7) >> 3).min(xsize_blocks)
    };

    let mut bx = 0usize;
    let full_y = py + 8 <= img_ysize;
    let exp_mul = f32x4_splat(1.442695041);
    let mul_v = f32x4_splat(mul);
    let add_v = f32x4_splat(add);
    let inv_scale_v = f32x4_splat(inv_scale);
    let half = f32x4_splat(0.5);

    // Fully interior groups of four adjacent 8x8 blocks. The +33 requirement is
    // for the HF right-difference load of the fourth block, whose masked-out lane
    // would otherwise read one float past the image at exact right edges.
    while bx + 4 <= valid_blocks {
        let px = x0 + bx * 8;
        if !(full_y && px + 33 <= img_xsize) {
            break;
        }

        let aq = load4s(aq_row, bx);
        let mask_val = compute_mask_x4(aq);
        let mask_val = gamma_modulation_blocks4_x4(px, py, opsin, mask_val);
        let hf = hf_modulation_blocks4_direct_x4(px, py, opsin, mask_val, hf_strength);
        let blue = blue_modulation_blocks4_x4(px, py, opsin, mask_val);
        let out_val = f32x4_min(hf, blue);
        let qf = f32x4_add(
            f32x4_mul(fast_exp2_x4(f32x4_mul(out_val, exp_mul)), mul_v),
            add_v,
        );
        let qi = i32x4_trunc_sat_f32x4(vmlaf(qf, inv_scale_v, half));
        store_quant_u8x4(qf_row, bx, qi);

        bx += 4;
    }

    for (rel_bx, (qf_out, &aq)) in qf_row[bx..valid_blocks]
        .iter_mut()
        .zip(aq_row[bx..valid_blocks].iter())
        .enumerate()
    {
        let bx = bx + rel_bx;
        write_quant_scalar_block(
            opsin,
            qf_out,
            aq,
            x0 + bx * 8,
            py,
            img_xsize,
            img_ysize,
            mul,
            add,
            inv_scale,
            hf_strength,
        );
    }

    qf_row[valid_blocks..xsize_blocks].fill(1);
}

#[inline]
#[target_feature(enable = "simd128")]
fn stage1_diff_x4(
    row_y: &[f32],
    row_y1: &[f32],
    row_y2: &[f32],
    gx: usize,
    offset: v128,
    quarter: v128,
    limit: v128,
) -> v128 {
    let cy = load4s(row_y, gx);
    let ly = load4s(row_y, gx - 1);
    let ry = load4s(row_y, gx + 1);
    let uy = load4s(row_y1, gx);
    let dy = load4s(row_y2, gx);
    let base_y = f32x4_mul(quarter, f32x4_add(f32x4_add(f32x4_add(dy, uy), ly), ry));
    let gammac = ratio_cubic_x4(f32x4_add(cy, offset), false);
    let dyv = f32x4_mul(gammac, f32x4_sub(cy, base_y));
    let diff = f32x4_min(f32x4_mul(dyv, dyv), limit);
    masking_sqrt_x4(diff)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn scalar_stage1_diff_pixel(
    row_y: &[f32],
    row_y1: &[f32],
    row_y2: &[f32],
    x0: usize,
    img_xsize: usize,
    rx: usize,
) -> f32 {
    let clampx = |x: isize| -> usize { x.max(0).min(img_xsize as isize - 1) as usize };
    let gx = x0 + rx;
    let gx_c = clampx(gx as isize);
    let gx1 = clampx(gx as isize - 1);
    let gx2 = clampx(gx as isize + 1);

    let in_y = row_y[gx_c];
    let base = 0.25 * (row_y2[gx_c] + row_y1[gx_c] + row_y[gx1] + row_y[gx2]);
    let gammac =
        crate::adaptive_quant::ratio_cubic_to_simple_gamma(in_y + MATCH_GAMMA_OFFSET, false);

    let mut diff = gammac * (in_y - base);
    diff *= diff;
    if diff >= 0.2 {
        diff = 0.2;
    }
    crate::adaptive_quant::masking_sqrt(diff)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stage1_pre_scalar_pixel(
    rows: &[(&[f32], &[f32], &[f32]); 4],
    x0: usize,
    img_xsize: usize,
    rx: usize,
) -> f32 {
    // Match the old row_acc order as closely as possible:
    //
    //   row_acc[x] = (((r0[x] + r1[x]) + r2[x]) + r3[x])
    //   pre[px] = (row_acc[x + 0] + row_acc[x + 1]
    //            + row_acc[x + 2] + row_acc[x + 3]) * 0.25
    //
    let mut sum = 0.0f32;
    for dx in 0..4 {
        let mut col = 0.0f32;
        for &(row_y, row_y1, row_y2) in rows.iter() {
            col += scalar_stage1_diff_pixel(row_y, row_y1, row_y2, x0, img_xsize, rx + dx);
        }
        sum += col;
    }
    sum * 0.25
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
fn stage1_fused_4rows_to_pre(
    opsin: &crate::image::Image3F,
    x0: usize,
    y0: usize,
    ry: usize,
    img_xsize: usize,
    img_ysize: usize,
    pre_w: usize,
    prow: &mut [f32],
) {
    debug_assert!(prow.len() >= pre_w);

    if pre_w == 0 {
        return;
    }

    let clampy = |y: isize| -> usize { y.max(0).min(img_ysize as isize - 1) as usize };

    let rows: [(&[f32], &[f32], &[f32]); 4] = core::array::from_fn(|dy| {
        let gy = y0 + ry + dy;
        let gy_c = clampy(gy as isize);
        let gy1 = clampy(gy as isize - 1);
        let gy2 = clampy(gy as isize + 1);
        (
            opsin.plane_row(1, gy_c),
            opsin.plane_row(1, gy1),
            opsin.plane_row(1, gy2),
        )
    });

    let offset = f32x4_splat(MATCH_GAMMA_OFFSET);
    let quarter = f32x4_splat(0.25);
    let limit = f32x4_splat(0.2);
    let zero = f32x4_splat(0.0);

    let mut px = 0usize;
    while px < pre_w {
        let gx = x0 + px * 4;

        // Four output pre pixels consume sixteen source pixels:
        //
        //   gx + 0..3, gx + 4..7, gx + 8..11, gx + 12..15
        //
        // Each Stage-1 vector also reads gx-1 and gx+1. Therefore, the last
        // vector reads up to gx+16, so the fully-vectorized path requires:
        //
        //   gx >= 1 && gx + 16 < img_xsize
        //
        if px + 4 <= pre_w && gx >= 1 && gx + 17 <= img_xsize {
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;

            for &(row_y, row_y1, row_y2) in rows.iter() {
                c0 = f32x4_add(
                    c0,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx, offset, quarter, limit),
                );
                c1 = f32x4_add(
                    c1,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx + 4, offset, quarter, limit),
                );
                c2 = f32x4_add(
                    c2,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx + 8, offset, quarter, limit),
                );
                c3 = f32x4_add(
                    c3,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx + 12, offset, quarter, limit),
                );
            }

            store4(f32x4_mul(hsum4x4(c0, c1, c2, c3), quarter), prow, px);

            px += 4;
        } else {
            prow[px] = stage1_pre_scalar_pixel(&rows, x0, img_xsize, px * 4);
            px += 1;
        }
    }
}

/// Scalar fuzzy-erosion value for one `pre` pixel (row edges). Byte-identical to
/// the scalar Stage-2 inner body.
#[inline]
fn scalar_px(
    rowt: &[f32],
    row: &[f32],
    rowb: &[f32],
    pre_w: usize,
    fx: usize,
    kmul: &[f32; 4],
) -> f32 {
    let xm1 = if fx >= 1 { fx - 1 } else { fx };
    let xp1 = if fx + 1 < pre_w { fx + 1 } else { fx };
    let mut mins = [row[fx], row[xm1], row[xp1], rowt[xm1]];
    crate::adaptive_quant::sort4(&mut mins);
    crate::adaptive_quant::store_min4(rowt[fx], &mut mins);
    crate::adaptive_quant::store_min4(rowt[xp1], &mut mins);
    crate::adaptive_quant::store_min4(rowb[xm1], &mut mins);
    crate::adaptive_quant::store_min4(rowb[fx], &mut mins);
    crate::adaptive_quant::store_min4(rowb[xp1], &mut mins);
    kmul[0] * mins[0] + kmul[1] * mins[1] + kmul[2] * mins[2] + kmul[3] * mins[3]
}

#[inline]
#[target_feature(enable = "simd128")]
fn cs_f32x4(a: &mut v128, b: &mut v128) {
    let lo = f32x4_min(*a, *b);
    let hi = f32x4_max(*a, *b);
    *a = lo;
    *b = hi;
}

#[inline]
#[target_feature(enable = "simd128")]
fn sort4_f32x4(m0: &mut v128, m1: &mut v128, m2: &mut v128, m3: &mut v128) {
    cs_f32x4(m0, m1);
    cs_f32x4(m2, m3);
    cs_f32x4(m0, m2);
    cs_f32x4(m1, m3);
    cs_f32x4(m1, m2);
}

#[inline]
#[target_feature(enable = "simd128")]
fn insert_min4_f32x4(m0: &mut v128, m1: &mut v128, m2: &mut v128, m3: &mut v128, v: v128) {
    let n0 = f32x4_min(*m0, v);
    let mut t = f32x4_max(*m0, v);
    *m0 = n0;

    let n1 = f32x4_min(*m1, t);
    t = f32x4_max(*m1, t);
    *m1 = n1;

    let n2 = f32x4_min(*m2, t);
    t = f32x4_max(*m2, t);
    *m2 = n2;

    *m3 = f32x4_min(*m3, t);
}

#[inline]
#[target_feature(enable = "simd128")]
fn fuzzy_erosion_x4(
    rowt: &[f32],
    row: &[f32],
    rowb: &[f32],
    fx: usize,
    k0: v128,
    k1: v128,
    k2: v128,
    k3: v128,
) -> v128 {
    // First 4 candidates.
    let mut m0 = load4s(rowt, fx - 1);
    let mut m1 = load4s(rowt, fx);
    let mut m2 = load4s(rowt, fx + 1);
    let mut m3 = load4s(row, fx - 1);

    // Sort first 4 candidates ascending, per lane.
    sort4_f32x4(&mut m0, &mut m1, &mut m2, &mut m3);

    // Insert remaining 5 candidates, keeping only the 4 smallest.
    insert_min4_f32x4(&mut m0, &mut m1, &mut m2, &mut m3, load4s(row, fx));
    insert_min4_f32x4(&mut m0, &mut m1, &mut m2, &mut m3, load4s(row, fx + 1));
    insert_min4_f32x4(&mut m0, &mut m1, &mut m2, &mut m3, load4s(rowb, fx - 1));
    insert_min4_f32x4(&mut m0, &mut m1, &mut m2, &mut m3, load4s(rowb, fx));
    insert_min4_f32x4(&mut m0, &mut m1, &mut m2, &mut m3, load4s(rowb, fx + 1));

    let mut v = f32x4_mul(k0, m0);
    v = vmlaf(k1, m1, v);
    v = vmlaf(k2, m2, v);
    v = vmlaf(k3, m3, v);
    v
}

#[inline]
#[target_feature(enable = "simd128")]
fn fuzzy_erosion_row_to_aq<const SET_MODE: bool>(
    rowt: &[f32],
    row: &[f32],
    rowb: &[f32],
    pre_w: usize,
    kmul: &[f32; 4],
    aq_row: &mut [f32],
) {
    debug_assert_eq!(pre_w, aq_row.len() * 2);

    if pre_w == 0 {
        return;
    }

    let k0 = f32x4_splat(kmul[0]);
    let k1 = f32x4_splat(kmul[1]);
    let k2 = f32x4_splat(kmul[2]);
    let k3 = f32x4_splat(kmul[3]);

    // First pair contains fx=0, which needs scalar edge handling.
    let first0 = scalar_px(rowt, row, rowb, pre_w, 0, kmul);
    let first1 = scalar_px(rowt, row, rowb, pre_w, 1, kmul);

    if SET_MODE {
        aq_row[0] = first0 + first1;
    } else {
        aq_row[0] += first0;
        aq_row[0] += first1;
    }

    let mut fx = 2usize;
    let mut out_x = 1usize;

    // Process 8 fuzzy pixels -> 4 AQ pixels.
    //
    // v0 = [f2, f3, f4, f5]
    // v1 = [f6, f7, f8, f9]
    //
    // even = [f2, f4, f6, f8]
    // odd  = [f3, f5, f7, f9]
    //
    // aq += even; aq += odd preserves original fx order per pair.
    while fx + 9 <= pre_w {
        let v0 = fuzzy_erosion_x4(rowt, row, rowb, fx, k0, k1, k2, k3);
        let v1 = fuzzy_erosion_x4(rowt, row, rowb, fx + 4, k0, k1, k2, k3);

        let even = uzp1_f32x4(v0, v1);
        let odd = uzp2_f32x4(v0, v1);

        if SET_MODE {
            store4(f32x4_add(even, odd), aq_row, out_x);
        } else {
            let acc = load4s(aq_row, out_x);
            let acc = f32x4_add(acc, even);
            let acc = f32x4_add(acc, odd);
            store4(acc, aq_row, out_x);
        }

        fx += 8;
        out_x += 4;
    }

    // Scalar tail, still pairwise.
    while fx + 1 < pre_w {
        let a = scalar_px(rowt, row, rowb, pre_w, fx, kmul);
        let b = scalar_px(rowt, row, rowb, pre_w, fx + 1, kmul);

        if SET_MODE {
            aq_row[out_x] = a + b;
        } else {
            aq_row[out_x] += a;
            aq_row[out_x] += b;
        }

        fx += 2;
        out_x += 1;
    }

    debug_assert_eq!(fx, pre_w);
    debug_assert_eq!(out_x, aq_row.len());
}

#[target_feature(enable = "simd128")]
pub(crate) fn fill_quant_field(
    opsin: &crate::image::Image3F,
    raw_quant_field: &mut crate::image::ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
    inv_scale: f32,
) {
    AQ_MAP_SCRATCH.with_borrow_mut(|scratch| {
        let xsize_blocks = raw_quant_field.xsize();
        let ysize_blocks = raw_quant_field.ysize();
        let img_xsize = opsin.xsize();
        let img_ysize = opsin.ysize();

        let scale = K_AC_QUANT / distance;

        let region_px_w = xsize_blocks * 8;
        let region_px_h = ysize_blocks * 8;

        // ---- Stage 1: per-pixel masking pre-pass.
        let pre_w = region_px_w / 4;
        let pre_h = region_px_h / 4;

        let total_secondary = pre_w * pre_h;
        if scratch.secondary.len() < total_secondary {
            scratch.secondary.resize(total_secondary, 0.);
        }
        let pre = &mut scratch.secondary[..total_secondary];

        for out_y in 0..pre_h {
            let ry = out_y * 4;
            let prow = &mut pre[out_y * pre_w..out_y * pre_w + pre_w];
            stage1_fused_4rows_to_pre(opsin, x0, y0, ry, img_xsize, img_ysize, pre_w, prow);
        }

        // ---- Stage 2: FuzzyErosion, then 2x downsample into block-resolution aq_map.
        let fe_mul = if distance < 2.0 {
            (2.0 - distance) * 0.5
        } else {
            0.0
        };
        let fe_base = [0.125f32, 0.1, 0.09, 0.06];
        let fe_add = [0.0f32, -0.1, -0.09, -0.06];
        let mut kmul = [0.0f32; 4];
        let mut norm_sum = 0.0f32;
        for i in 0..4 {
            kmul[i] = fe_base[i] + fe_mul * fe_add[i];
            norm_sum += kmul[i];
        }
        let k_total = 0.29959705784054957f32;
        for w in &mut kmul {
            *w *= k_total / norm_sum;
        }
        if scratch.aq_map.len() < xsize_blocks * ysize_blocks {
            scratch.aq_map.resize(xsize_blocks * ysize_blocks, 0.);
        }
        let aq_map = &mut scratch.aq_map[..xsize_blocks * ysize_blocks];
        for fy in 0..pre_h {
            let ym1 = if fy >= 1 { fy - 1 } else { fy };
            let yp1 = if fy + 1 < pre_h { fy + 1 } else { fy };

            let rowt = &pre[ym1 * pre_w..ym1 * pre_w + pre_w];
            let row = &pre[fy * pre_w..fy * pre_w + pre_w];
            let rowb = &pre[yp1 * pre_w..yp1 * pre_w + pre_w];

            let out_y = fy >> 1;
            let aq_row = &mut aq_map[out_y * xsize_blocks..out_y * xsize_blocks + xsize_blocks];

            if (fy & 1) == 0 {
                fuzzy_erosion_row_to_aq::<true>(rowt, row, rowb, pre_w, &kmul, aq_row);
            } else {
                fuzzy_erosion_row_to_aq::<false>(rowt, row, rowb, pre_w, &kmul, aq_row);
            }
        }

        // ---- Stage 3: per-block modulations + integer quant field.
        let base_level = 0.48 * scale;
        let k_dampen_ramp_start = 2.0f32;
        let k_dampen_ramp_end = 8.0f32;
        let mut dampen = 1.0f32;
        if distance >= k_dampen_ramp_start {
            dampen = 1.0
                - ((distance - k_dampen_ramp_start) / (k_dampen_ramp_end - k_dampen_ramp_start));
            if dampen < 0.0 {
                dampen = 0.0;
            }
        }
        let mul = scale * dampen;
        let add = (1.0 - dampen) * base_level;
        let hf_strength = crate::adaptive_quant::hf_modulation_strength(distance);

        for by in 0..ysize_blocks {
            let py = y0 + by * 8;
            let aq_row = &aq_map[by * xsize_blocks..by * xsize_blocks + xsize_blocks];
            let qf_row = raw_quant_field.row_mut(by);
            write_quant_row_wasm(
                opsin,
                aq_row,
                qf_row,
                x0,
                py,
                img_xsize,
                img_ysize,
                mul,
                add,
                inv_scale,
                hf_strength,
            );
        }
    });
}
