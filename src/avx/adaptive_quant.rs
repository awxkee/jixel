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

use crate::adaptive_quant::K_AC_QUANT;
use std::arch::x86_64::*;

const MATCH_GAMMA_OFFSET: f32 = 0.019;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn load8s(s: &[f32], i: usize) -> __m256 {
    unsafe { _mm256_loadu_ps(s[i..].as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn store8(v: __m256, s: &mut [f32], i: usize) {
    unsafe {
        _mm256_storeu_ps(s[i..].as_mut_ptr(), v);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn load4s(s: &[f32], i: usize) -> __m128 {
    unsafe { _mm_loadu_ps(s[i..].as_ptr()) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn store4(v: __m128, s: &mut [f32], i: usize) {
    unsafe {
        _mm_storeu_ps(s[i..].as_mut_ptr(), v);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
fn abs_ps(v: __m256) -> __m256 {
    _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), v)
}

/// `a * b + c`, true fused multiply-add.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn mlaf(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(a, b, c)
}

/// Vectorised `ratio_cubic_to_simple_gamma`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn ratio_cubic_x8(v: __m256, invert: bool) -> __m256 {
    const K_SG_MUL: f32 = 226.77216153508914;
    const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
    const K_LOG2: f32 = 0.693147181;
    const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_LOG2;
    const K_SG_V_OFFSET: f32 = 7.7825991679894591;
    let k_epsilon = 1e-2f32;
    let k_num_mul = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    let k_v_offset = K_SG_V_OFFSET * K_LOG2 + k_epsilon;
    let k_den_mul = K_LOG2 * K_SG_MUL;

    let v = _mm256_max_ps(v, _mm256_setzero_ps());
    let v2 = _mm256_mul_ps(v, v);
    let num = mlaf(_mm256_set1_ps(k_num_mul), v2, _mm256_set1_ps(k_epsilon));
    let den = mlaf(
        _mm256_mul_ps(_mm256_set1_ps(k_den_mul), v),
        v2,
        _mm256_set1_ps(k_v_offset),
    );
    if invert {
        _mm256_div_ps(num, den)
    } else {
        _mm256_div_ps(den, num)
    }
}

const MASKING_SQRT_MUL_V: f32 = 145487.346437769899962;

/// Vectorised `masking_sqrt`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn masking_sqrt_x8(v: __m256) -> __m256 {
    let k_log_offset = 27.505837037000106f32;
    let inner = mlaf(
        v,
        _mm256_set1_ps(MASKING_SQRT_MUL_V),
        _mm256_set1_ps(k_log_offset),
    );
    _mm256_mul_ps(_mm256_set1_ps(0.25), _mm256_sqrt_ps(inner))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dirty_log2f_x8(d: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);

    // Same range reduction as scalar dirty_log2f(): reduce into
    // [sqrt(2)/2, sqrt(2)] and keep the extracted exponent in n.
    let mut ix = _mm256_castps_si256(d);
    ix = _mm256_add_epi32(
        ix,
        _mm256_set1_epi32((0x3f800000u32 - 0x3f3504f3u32) as i32),
    );
    let n = _mm256_sub_epi32(_mm256_srli_epi32::<23>(ix), _mm256_set1_epi32(0x7f));
    ix = _mm256_add_epi32(
        _mm256_and_si256(ix, _mm256_set1_epi32(0x007fffff)),
        _mm256_set1_epi32(0x3f3504f3),
    );

    let a = _mm256_castsi256_ps(ix);
    let x = _mm256_div_ps(_mm256_sub_ps(a, one), _mm256_add_ps(a, one));
    let x2 = _mm256_mul_ps(x, x);

    let mut u = _mm256_set1_ps(0.4121985850084821691);
    u = mlaf(u, x2, _mm256_set1_ps(0.5770780163490337802));
    u = mlaf(u, x2, _mm256_set1_ps(0.9617966939259845749));

    let n = _mm256_cvtepi32_ps(n);
    let base = mlaf(x, _mm256_set1_ps(2.8853900817779268), n);
    mlaf(_mm256_mul_ps(x2, x), u, base)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn compute_mask_x8(out_val: __m256) -> __m256 {
    let k_base = _mm256_set1_ps(-0.7647);
    let k_mul4 = _mm256_set1_ps(9.4708735624378946);
    let k_mul2 = _mm256_set1_ps(17.35036561631863);
    let k_offset2 = _mm256_set1_ps(302.59587815579727);
    let k_mul3 = _mm256_set1_ps(6.7943250517376494);
    let k_offset3 = _mm256_set1_ps(3.7179635626140772);
    let k_offset4 = _mm256_set1_ps(0.25 * 3.7179635626140772);
    let k_mul0 = _mm256_set1_ps(0.80061762862741759);
    let one = _mm256_set1_ps(1.0);

    let v1 = _mm256_max_ps(_mm256_mul_ps(out_val, k_mul0), _mm256_set1_ps(1e-3));
    let v1_sq = _mm256_mul_ps(v1, v1);
    let v2 = _mm256_div_ps(one, _mm256_add_ps(v1, k_offset2));
    let v3 = _mm256_div_ps(one, _mm256_add_ps(v1_sq, k_offset3));
    let v4 = _mm256_div_ps(one, _mm256_add_ps(v1_sq, k_offset4));

    mlaf(k_mul4, v4, mlaf(k_mul2, v2, mlaf(k_mul3, v3, k_base)))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn gamma_row_sum_x8(row_x: &[f32], row_y: &[f32], base: usize) -> __m256 {
    let bias = _mm256_set1_ps(0.16);
    let half = _mm256_set1_ps(0.5);

    let x = load8s(row_x, base);
    let y = _mm256_add_ps(load8s(row_y, base), bias);
    let r = _mm256_sub_ps(y, x);
    let g = _mm256_add_ps(y, x);
    _mm256_mul_ps(
        half,
        _mm256_add_ps(ratio_cubic_x8(r, true), ratio_cubic_x8(g, true)),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
fn hsum8_vectors(
    s0: __m256,
    s1: __m256,
    s2: __m256,
    s3: __m256,
    s4: __m256,
    s5: __m256,
    s6: __m256,
    s7: __m256,
) -> __m256 {
    let p01 = _mm256_hadd_ps(s0, s1);
    let p23 = _mm256_hadd_ps(s2, s3);
    let p45 = _mm256_hadd_ps(s4, s5);
    let p67 = _mm256_hadd_ps(s6, s7);

    let q0123 = _mm256_hadd_ps(p01, p23);
    let q4567 = _mm256_hadd_ps(p45, p67);

    let low_halves = _mm256_permute2f128_ps::<0x20>(q0123, q4567);
    let high_halves = _mm256_permute2f128_ps::<0x31>(q0123, q4567);
    _mm256_add_ps(low_halves, high_halves)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn gamma_modulation_blocks8_x8(
    x: usize,
    y: usize,
    xyb: &crate::image::Image3F,
    out_val: __m256,
) -> __m256 {
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut s4 = _mm256_setzero_ps();
    let mut s5 = _mm256_setzero_ps();
    let mut s6 = _mm256_setzero_ps();
    let mut s7 = _mm256_setzero_ps();

    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        s0 = _mm256_add_ps(s0, gamma_row_sum_x8(row_x, row_y, x));
        s1 = _mm256_add_ps(s1, gamma_row_sum_x8(row_x, row_y, x + 8));
        s2 = _mm256_add_ps(s2, gamma_row_sum_x8(row_x, row_y, x + 16));
        s3 = _mm256_add_ps(s3, gamma_row_sum_x8(row_x, row_y, x + 24));
        s4 = _mm256_add_ps(s4, gamma_row_sum_x8(row_x, row_y, x + 32));
        s5 = _mm256_add_ps(s5, gamma_row_sum_x8(row_x, row_y, x + 40));
        s6 = _mm256_add_ps(s6, gamma_row_sum_x8(row_x, row_y, x + 48));
        s7 = _mm256_add_ps(s7, gamma_row_sum_x8(row_x, row_y, x + 56));
    }

    let overall = _mm256_mul_ps(
        hsum8_vectors(s0, s1, s2, s3, s4, s5, s6, s7),
        _mm256_set1_ps(1.0 / 64.0),
    );
    mlaf(
        _mm256_set1_ps(0.1005613337192697),
        dirty_log2f_x8(overall),
        out_val,
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn hf_row_sum_x8(
    row: &[f32],
    row_next: &[f32],
    base: usize,
    valmin_y: __m256,
    right_tail_mask: __m256,
    has_vertical: bool,
) -> __m256 {
    let p = load8s(row, base);
    let right = _mm256_mul_ps(
        _mm256_min_ps(abs_ps(_mm256_sub_ps(p, load8s(row, base + 1))), valmin_y),
        right_tail_mask,
    );
    let mut sum = right;

    if has_vertical {
        let down = _mm256_min_ps(abs_ps(_mm256_sub_ps(p, load8s(row_next, base))), valmin_y);
        sum = _mm256_add_ps(sum, down);
    }

    sum
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn hf_modulation_blocks8_direct_x8(
    x: usize,
    y: usize,
    xyb_y: &crate::image::Image3F,
    out_val: __m256,
    strength: f32,
) -> __m256 {
    let valmin_y = _mm256_set1_ps(0.0206);
    let right_mask = _mm256_setr_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0);

    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut s4 = _mm256_setzero_ps();
    let mut s5 = _mm256_setzero_ps();
    let mut s6 = _mm256_setzero_ps();
    let mut s7 = _mm256_setzero_ps();

    for dy in 0..8 {
        let row = xyb_y.plane_row(1, y + dy);
        let row_next = if dy == 7 {
            row
        } else {
            xyb_y.plane_row(1, y + dy + 1)
        };
        let has_vertical = dy != 7;

        s0 = _mm256_add_ps(
            s0,
            hf_row_sum_x8(row, row_next, x, valmin_y, right_mask, has_vertical),
        );
        s1 = _mm256_add_ps(
            s1,
            hf_row_sum_x8(row, row_next, x + 8, valmin_y, right_mask, has_vertical),
        );
        s2 = _mm256_add_ps(
            s2,
            hf_row_sum_x8(row, row_next, x + 16, valmin_y, right_mask, has_vertical),
        );
        s3 = _mm256_add_ps(
            s3,
            hf_row_sum_x8(row, row_next, x + 24, valmin_y, right_mask, has_vertical),
        );
        s4 = _mm256_add_ps(
            s4,
            hf_row_sum_x8(row, row_next, x + 32, valmin_y, right_mask, has_vertical),
        );
        s5 = _mm256_add_ps(
            s5,
            hf_row_sum_x8(row, row_next, x + 40, valmin_y, right_mask, has_vertical),
        );
        s6 = _mm256_add_ps(
            s6,
            hf_row_sum_x8(row, row_next, x + 48, valmin_y, right_mask, has_vertical),
        );
        s7 = _mm256_add_ps(
            s7,
            hf_row_sum_x8(row, row_next, x + 56, valmin_y, right_mask, has_vertical),
        );
    }

    let sums = hsum8_vectors(s0, s1, s2, s3, s4, s5, s6, s7);
    mlaf(
        sums,
        _mm256_set1_ps(-0.38 * strength),
        _mm256_add_ps(out_val, _mm256_set1_ps(0.42)),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn blue_row_sum_x8(row_x: &[f32], row_y: &[f32], row_b: &[f32], base: usize) -> __m256 {
    let k_limit = _mm256_set1_ps(0.010474084867598155);
    let k_offset = _mm256_set1_ps(0.0031994768654636393);
    let zero = _mm256_setzero_ps();

    let x = load8s(row_x, base);
    let y = load8s(row_y, base);
    let b = load8s(row_b, base);
    let y_eff = _mm256_add_ps(_mm256_add_ps(y, k_offset), abs_ps(x));
    _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(b, y_eff), zero), k_limit)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn blue_modulation_blocks8_x8(
    x: usize,
    y: usize,
    xyb: &crate::image::Image3F,
    out_val: __m256,
) -> __m256 {
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut s4 = _mm256_setzero_ps();
    let mut s5 = _mm256_setzero_ps();
    let mut s6 = _mm256_setzero_ps();
    let mut s7 = _mm256_setzero_ps();

    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        let row_b = xyb.plane_row(2, y + dy);

        s0 = _mm256_add_ps(s0, blue_row_sum_x8(row_x, row_y, row_b, x));
        s1 = _mm256_add_ps(s1, blue_row_sum_x8(row_x, row_y, row_b, x + 8));
        s2 = _mm256_add_ps(s2, blue_row_sum_x8(row_x, row_y, row_b, x + 16));
        s3 = _mm256_add_ps(s3, blue_row_sum_x8(row_x, row_y, row_b, x + 24));
        s4 = _mm256_add_ps(s4, blue_row_sum_x8(row_x, row_y, row_b, x + 32));
        s5 = _mm256_add_ps(s5, blue_row_sum_x8(row_x, row_y, row_b, x + 40));
        s6 = _mm256_add_ps(s6, blue_row_sum_x8(row_x, row_y, row_b, x + 48));
        s7 = _mm256_add_ps(s7, blue_row_sum_x8(row_x, row_y, row_b, x + 56));
    }

    const K_LIMIT: f32 = 0.010474084867598155;
    const K_MAX_LIMIT: f32 = 15.463398341612438 * K_LIMIT;
    const SCALE: f32 = 0.90590804735610064;

    let mut sums = hsum8_vectors(s0, s1, s2, s3, s4, s5, s6, s7);
    let flip_mask = _mm256_cmp_ps::<_CMP_GE_OQ>(sums, _mm256_set1_ps(32.0 * K_LIMIT));
    let flipped = _mm256_sub_ps(_mm256_set1_ps(64.0 * K_LIMIT), sums);
    sums = _mm256_blendv_ps(sums, flipped, flip_mask);
    sums = _mm256_min_ps(sums, _mm256_set1_ps(K_MAX_LIMIT));

    mlaf(sums, _mm256_set1_ps(SCALE), out_val)
}

pub(crate) const EXP2_P0: f32 = 1.00000011920928955078125_f32;
pub(crate) const EXP2_P1: f32 = 0.69314706325531005859375_f32;
pub(crate) const EXP2_P2: f32 = 0.24022041261196136474609375_f32;
pub(crate) const EXP2_P3: f32 = 5.550567805767059326171875e-2_f32;
pub(crate) const EXP2_P4: f32 = 9.678089059889316558837890625e-3_f32;
pub(crate) const EXP2_P5: f32 = 1.33218802511692047119140625e-3_f32;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn pow2if_s32(q: __m256i) -> __m256i {
    _mm256_slli_epi32::<23>(_mm256_add_epi32(q, _mm256_set1_epi32(0x7f)))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn fast_exp2_x8(v: __m256) -> __m256 {
    // exp2(x) = 2^q * 2^r, q = round(x), r = x - q.
    let q = _mm256_cvtps_epi32(v);
    let qf = _mm256_cvtepi32_ps(q);
    let r = _mm256_sub_ps(v, qf);

    // Horner: coeff + p*r. Keep the FMA order explicit: p*r + coeff.
    let mut p = _mm256_set1_ps(EXP2_P5);
    p = mlaf(p, r, _mm256_set1_ps(EXP2_P4));
    p = mlaf(p, r, _mm256_set1_ps(EXP2_P3));
    p = mlaf(p, r, _mm256_set1_ps(EXP2_P2));
    p = mlaf(p, r, _mm256_set1_ps(EXP2_P1));
    p = mlaf(p, r, _mm256_set1_ps(EXP2_P0));

    let scale = _mm256_castsi256_ps(pow2if_s32(q));
    _mm256_mul_ps(p, scale)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn store_quant_u8x8(qf_row: &mut [u8], bx: usize, qi: __m256i) {
    let qi = _mm256_max_epi32(
        _mm256_set1_epi32(1),
        _mm256_min_epi32(qi, _mm256_set1_epi32(255)),
    );
    let zero = _mm256_setzero_si256();
    let i16 = _mm256_packus_epi32(qi, zero);
    // packus_epi32 works independently in 128-bit lanes:
    //   [q0 q1 q2 q3 0 0 0 0 | q4 q5 q6 q7 0 0 0 0]
    // Move q4..q7 next to q0..q3 before packing to u8.
    let i16 = _mm256_permute4x64_epi64::<0xd8>(i16);
    let i8 = _mm256_packus_epi16(i16, zero);
    unsafe {
        _mm_storel_epi64(
            qf_row.as_mut_ptr().add(bx).cast(),
            _mm256_castsi256_si128(i8),
        );
    }
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

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
fn write_quant_row_avx2(
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
    let xsize_blocks = aq_row.len().min(qf_row.len());

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
    let exp_mul = _mm256_set1_ps(1.442695041);
    let mul_v = _mm256_set1_ps(mul);
    let add_v = _mm256_set1_ps(add);
    let inv_scale_v = _mm256_set1_ps(inv_scale);
    let half = _mm256_set1_ps(0.5);

    // Fully interior groups of eight adjacent 8x8 blocks. The +65 requirement is
    // for the HF right-difference load of the eighth block, whose masked-out lane
    // would otherwise read one float past the image at exact right edges.
    while bx + 8 <= valid_blocks {
        let px = x0 + bx * 8;
        if !(full_y && px + 65 <= img_xsize) {
            break;
        }

        let aq = load8s(aq_row, bx);
        let mask_val = compute_mask_x8(aq);
        let mask_val = gamma_modulation_blocks8_x8(px, py, opsin, mask_val);
        let hf = hf_modulation_blocks8_direct_x8(px, py, opsin, mask_val, hf_strength);
        let blue = blue_modulation_blocks8_x8(px, py, opsin, mask_val);
        let out_val = _mm256_min_ps(hf, blue);
        let qf = mlaf(fast_exp2_x8(_mm256_mul_ps(out_val, exp_mul)), mul_v, add_v);
        let qi_f = mlaf(qf, inv_scale_v, half);
        let qi = _mm256_cvttps_epi32(qi_f);
        store_quant_u8x8(qf_row, bx, qi);

        bx += 8;
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
#[target_feature(enable = "avx2,fma")]
fn stage1_diff_x8(
    row_y: &[f32],
    row_y1: &[f32],
    row_y2: &[f32],
    gx: usize,
    offset: __m256,
    quarter: __m256,
    limit: __m256,
) -> __m256 {
    let cy = load8s(row_y, gx);
    let ly = load8s(row_y, gx - 1);
    let ry = load8s(row_y, gx + 1);
    let uy = load8s(row_y1, gx);
    let dy = load8s(row_y2, gx);
    let base_y = _mm256_mul_ps(
        quarter,
        _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(dy, uy), ly), ry),
    );
    let gammac = ratio_cubic_x8(_mm256_add_ps(cy, offset), false);
    let dyv = _mm256_mul_ps(gammac, _mm256_sub_ps(cy, base_y));
    let diff = _mm256_min_ps(_mm256_mul_ps(dyv, dyv), limit);
    masking_sqrt_x8(diff)
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
    // row_acc[x] = (((r0[x] + r1[x]) + r2[x]) + r3[x])
    // pre[px] = (row_acc[x + 0] + ... + row_acc[x + 3]) * 0.25
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

#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum4_groups_from_4x8(c0: __m256, c1: __m256, c2: __m256, c3: __m256) -> __m256 {
    // Input:
    //   c0 = pixels  0..7, c1 =  8..15, c2 = 16..23, c3 = 24..31.
    // Return:
    //   [sum 0..3, sum 4..7, sum 8..11, ..., sum 28..31].
    let p01 = _mm256_hadd_ps(c0, c1);
    let p23 = _mm256_hadd_ps(c2, c3);
    let h = _mm256_hadd_ps(p01, p23);
    _mm256_permutevar8x32_ps(h, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
}

#[target_feature(enable = "avx2,fma")]
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

    let offset = _mm256_set1_ps(MATCH_GAMMA_OFFSET);
    let quarter = _mm256_set1_ps(0.25);
    let limit = _mm256_set1_ps(0.2);
    let zero = _mm256_setzero_ps();

    let mut px = 0usize;
    while px < pre_w {
        let gx = x0 + px * 4;

        // Eight output pre pixels consume thirty-two source pixels. Each
        // Stage-1 vector also reads gx-1 and gx+1. Therefore, the last vector
        // reads up to gx+32, so the fully-vectorized path requires:
        //   gx >= 1 && gx + 32 < img_xsize
        if px + 8 <= pre_w && gx >= 1 && gx + 33 <= img_xsize {
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;

            for &(row_y, row_y1, row_y2) in rows.iter() {
                c0 = _mm256_add_ps(
                    c0,
                    stage1_diff_x8(row_y, row_y1, row_y2, gx, offset, quarter, limit),
                );
                c1 = _mm256_add_ps(
                    c1,
                    stage1_diff_x8(row_y, row_y1, row_y2, gx + 8, offset, quarter, limit),
                );
                c2 = _mm256_add_ps(
                    c2,
                    stage1_diff_x8(row_y, row_y1, row_y2, gx + 16, offset, quarter, limit),
                );
                c3 = _mm256_add_ps(
                    c3,
                    stage1_diff_x8(row_y, row_y1, row_y2, gx + 24, offset, quarter, limit),
                );
            }

            let sum = hsum4_groups_from_4x8(c0, c1, c2, c3);
            store8(_mm256_mul_ps(sum, quarter), prow, px);
            px += 8;
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
#[target_feature(enable = "avx2,fma")]
fn cs_f32x8(a: &mut __m256, b: &mut __m256) {
    let lo = _mm256_min_ps(*a, *b);
    let hi = _mm256_max_ps(*a, *b);
    *a = lo;
    *b = hi;
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn sort4_f32x8(m0: &mut __m256, m1: &mut __m256, m2: &mut __m256, m3: &mut __m256) {
    cs_f32x8(m0, m1);
    cs_f32x8(m2, m3);
    cs_f32x8(m0, m2);
    cs_f32x8(m1, m3);
    cs_f32x8(m1, m2);
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn insert_min4_f32x8(
    m0: &mut __m256,
    m1: &mut __m256,
    m2: &mut __m256,
    m3: &mut __m256,
    v: __m256,
) {
    let n0 = _mm256_min_ps(*m0, v);
    let mut t = _mm256_max_ps(*m0, v);
    *m0 = n0;

    let n1 = _mm256_min_ps(*m1, t);
    t = _mm256_max_ps(*m1, t);
    *m1 = n1;

    let n2 = _mm256_min_ps(*m2, t);
    t = _mm256_max_ps(*m2, t);
    *m2 = n2;

    *m3 = _mm256_min_ps(*m3, t);
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn fuzzy_erosion_x8(
    rowt: &[f32],
    row: &[f32],
    rowb: &[f32],
    fx: usize,
    k0: __m256,
    k1: __m256,
    k2: __m256,
    k3: __m256,
) -> __m256 {
    // First 4 candidates.
    let mut m0 = load8s(rowt, fx - 1);
    let mut m1 = load8s(rowt, fx);
    let mut m2 = load8s(rowt, fx + 1);
    let mut m3 = load8s(row, fx - 1);

    // Sort first 4 candidates ascending, per lane.
    sort4_f32x8(&mut m0, &mut m1, &mut m2, &mut m3);

    // Insert remaining 5 candidates, keeping only the 4 smallest.
    insert_min4_f32x8(&mut m0, &mut m1, &mut m2, &mut m3, load8s(row, fx));
    insert_min4_f32x8(&mut m0, &mut m1, &mut m2, &mut m3, load8s(row, fx + 1));
    insert_min4_f32x8(&mut m0, &mut m1, &mut m2, &mut m3, load8s(rowb, fx - 1));
    insert_min4_f32x8(&mut m0, &mut m1, &mut m2, &mut m3, load8s(rowb, fx));
    insert_min4_f32x8(&mut m0, &mut m1, &mut m2, &mut m3, load8s(rowb, fx + 1));

    let mut v = _mm256_mul_ps(k0, m0);
    v = mlaf(k1, m1, v);
    v = mlaf(k2, m2, v);
    v = mlaf(k3, m3, v);
    v
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn pair_sums4_from8(v: __m256) -> __m128 {
    // v = [f0 f1 f2 f3 f4 f5 f6 f7]
    // return [f0+f1, f2+f3, f4+f5, f6+f7]
    let p = _mm256_hadd_ps(v, v);
    let p = _mm256_permutevar8x32_ps(p, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    _mm256_castps256_ps128(p)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
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

    let k0 = _mm256_set1_ps(kmul[0]);
    let k1 = _mm256_set1_ps(kmul[1]);
    let k2 = _mm256_set1_ps(kmul[2]);
    let k3 = _mm256_set1_ps(kmul[3]);

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
    while fx + 9 <= pre_w {
        let v = fuzzy_erosion_x8(rowt, row, rowb, fx, k0, k1, k2, k3);
        let pairs = pair_sums4_from8(v);

        if SET_MODE {
            store4(pairs, aq_row, out_x);
        } else {
            let acc = load4s(aq_row, out_x);
            store4(_mm_add_ps(acc, pairs), aq_row, out_x);
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

#[target_feature(enable = "avx2,fma")]
pub(crate) fn fill_quant_field(
    scratch: &mut crate::adaptive_quant::AqMapScratch,
    opsin: &crate::image::Image3F,
    raw_quant_field: &mut crate::image::ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
    inv_scale: f32,
) {
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
    let dampen = crate::adaptive_quant::aq_dampen(distance);
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;
    let hf_strength = crate::adaptive_quant::hf_modulation_strength(distance);

    for by in 0..ysize_blocks {
        let py = y0 + by * 8;
        let aq_row = &aq_map[by * xsize_blocks..by * xsize_blocks + xsize_blocks];
        let qf_row = raw_quant_field.row_mut(by);
        write_quant_row_avx2(
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
}
