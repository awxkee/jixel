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
use std::arch::aarch64::*;

const MATCH_GAMMA_OFFSET: f32 = 0.019;

#[inline]
#[target_feature(enable = "neon")]
fn load4s(s: &[f32], i: usize) -> float32x4_t {
    unsafe { vld1q_f32(s[i..].as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store4(v: float32x4_t, s: &mut [f32], i: usize) {
    unsafe {
        vst1q_f32(s[i..].as_mut_ptr(), v);
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
fn hsum(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
fn abs_ps(v: float32x4_t) -> float32x4_t {
    vabsq_f32(v)
}

/// `a * b + c`, true fused multiply-add.
#[inline]
#[target_feature(enable = "neon")]
fn vmlaf(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vfmaq_f32(c, a, b)
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
fn neg_s32(n: int32x4_t) -> int32x4_t {
    vnegq_s32(n)
}

/// Vectorised `ratio_cubic_to_simple_gamma`.
#[inline]
#[target_feature(enable = "neon")]
fn ratio_cubic_x4<const INVERT: bool>(v: float32x4_t) -> float32x4_t {
    const K_SG_MUL: f32 = 226.77216153508914;
    const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
    const K_LOG2: f32 = 0.693147181;
    const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_LOG2;
    const K_SG_V_OFFSET: f32 = 7.7825991679894591;
    let k_epsilon = 1e-2f32;
    let k_num_mul = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    let k_v_offset = K_SG_V_OFFSET * K_LOG2 + k_epsilon;
    let k_den_mul = K_LOG2 * K_SG_MUL;

    let v = vmaxq_f32(v, vdupq_n_f32(0.0));
    let v2 = vmulq_f32(v, v);
    let num = vmlaf(vdupq_n_f32(k_num_mul), v2, vdupq_n_f32(k_epsilon));
    let den = vmlaf(
        vmulq_f32(vdupq_n_f32(k_den_mul), v),
        v2,
        vdupq_n_f32(k_v_offset),
    );
    if INVERT {
        vdivq_f32(num, den)
    } else {
        vdivq_f32(den, num)
    }
}

const MASKING_SQRT_MUL_V: f32 = 145487.346437769899962; //(211.66567973503678f32 * 1e8f32).sqrt();

/// Vectorised `masking_sqrt`.
#[inline]
#[target_feature(enable = "neon")]
fn masking_sqrt_x4(v: float32x4_t) -> float32x4_t {
    let k_log_offset = 27.505837037000106f32;
    let inner = vmlaf(
        v,
        vdupq_n_f32(MASKING_SQRT_MUL_V),
        vdupq_n_f32(k_log_offset),
    );
    vmulq_f32(vdupq_n_f32(0.25), vsqrtq_f32(inner))
}

#[inline]
#[target_feature(enable = "neon")]
fn dirty_log2f_x4(d: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);

    // Same range reduction as scalar dirty_log2f(): reduce into
    // [sqrt(2)/2, sqrt(2)] and keep the extracted exponent in n.
    let mut ix = vreinterpretq_u32_f32(d);
    ix = vaddq_u32(ix, vdupq_n_u32(0x3f800000u32 - 0x3f3504f3u32));
    let n = vreinterpretq_s32_u32(vsubq_u32(vshrq_n_u32(ix, 23), vdupq_n_u32(0x7f)));
    ix = vaddq_u32(
        vandq_u32(ix, vdupq_n_u32(0x007fffff)),
        vdupq_n_u32(0x3f3504f3),
    );

    let a = vreinterpretq_f32_u32(ix);
    let x = vdivq_f32(vsubq_f32(a, one), vaddq_f32(a, one));
    let x2 = vmulq_f32(x, x);

    let mut u = vdupq_n_f32(0.4121985850084821691);
    u = vmlaf(u, x2, vdupq_n_f32(0.5770780163490337802));
    u = vmlaf(u, x2, vdupq_n_f32(0.9617966939259845749));

    let n = vcvtq_f32_s32(n);
    let base = vmlaf(x, vdupq_n_f32(2.8853900817779268), n);
    vmlaf(vmulq_f32(x2, x), u, base)
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
fn dirty_log1pf_x4(d: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let a = vaddq_f32(d, one);
    let ix = vreinterpretq_u32_f32(a);
    let exponent = vandq_u32(ix, vdupq_n_u32(0x7f80_0000));
    let n = vreinterpretq_s32_u32(vsubq_u32(vshrq_n_u32(exponent, 23), vdupq_n_u32(0x7f)));
    let mantissa = vreinterpretq_f32_u32(vorrq_u32(
        vandq_u32(ix, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f80_0000),
    ));
    let reduced = vsubq_f32(mantissa, one);
    let t = vbslq_f32(vceqq_s32(n, vdupq_n_s32(0)), d, reduced);

    let mut p = vdupq_n_f32(0.014539075084030628);
    p = vmlaf(p, t, vdupq_n_f32(-0.0675969123840332));
    p = vmlaf(p, t, vdupq_n_f32(0.15056970715522766));
    p = vmlaf(p, t, vdupq_n_f32(-0.23573730885982513));
    p = vmlaf(p, t, vdupq_n_f32(0.33125850558280945));
    p = vmlaf(p, t, vdupq_n_f32(-0.4998837411403656));
    p = vmlaf(p, t, vdupq_n_f32(0.999998927116394));

    let n = vcvtq_f32_s32(n);
    vmlaf(n, vdupq_n_f32(0.6931471805599453), vmulq_f32(t, p))
}

#[inline]
#[target_feature(enable = "neon")]
fn compute_mask_x4(out_val: float32x4_t) -> float32x4_t {
    let k_base = vdupq_n_f32(-0.7647);
    let k_mul4 = vdupq_n_f32(9.4708735624378946);
    let k_mul2 = vdupq_n_f32(17.35036561631863);
    let k_offset2 = vdupq_n_f32(302.59587815579727);
    let k_mul3 = vdupq_n_f32(6.7943250517376494);
    let k_offset3 = vdupq_n_f32(3.7179635626140772);
    let k_offset4 = vdupq_n_f32(0.25 * 3.7179635626140772);
    let k_mul0 = vdupq_n_f32(0.80061762862741759);
    let one = vdupq_n_f32(1.0);

    let v1 = vmaxq_f32(vmulq_f32(out_val, k_mul0), vdupq_n_f32(1e-3));
    let v1_sq = vmulq_f32(v1, v1);
    let v2 = vdivq_f32(one, vaddq_f32(v1, k_offset2));
    let v3 = vdivq_f32(one, vaddq_f32(v1_sq, k_offset3));
    let v4 = vdivq_f32(one, vaddq_f32(v1_sq, k_offset4));

    vmlaf(k_mul4, v4, vmlaf(k_mul2, v2, vmlaf(k_mul3, v3, k_base)))
}

#[inline]
#[target_feature(enable = "neon")]
fn gamma_row_sum_x4(row_x: &[f32], row_y: &[f32], base: usize) -> float32x4_t {
    let bias = vdupq_n_f32(0.16);
    let half = vdupq_n_f32(0.5);

    let x0 = load4s(row_x, base);
    let y0 = vaddq_f32(load4s(row_y, base), bias);
    let r0 = vsubq_f32(y0, x0);
    let g0 = vaddq_f32(y0, x0);
    let sum0 = vmulq_f32(
        half,
        vaddq_f32(ratio_cubic_x4::<true>(r0), ratio_cubic_x4::<true>(g0)),
    );

    let x1 = load4s(row_x, base + 4);
    let y1 = vaddq_f32(load4s(row_y, base + 4), bias);
    let r1 = vsubq_f32(y1, x1);
    let g1 = vaddq_f32(y1, x1);
    let sum1 = vmulq_f32(
        half,
        vaddq_f32(ratio_cubic_x4::<true>(r1), ratio_cubic_x4::<true>(g1)),
    );

    vaddq_f32(sum0, sum1)
}

#[inline]
#[target_feature(enable = "neon")]
fn gamma_modulation_blocks4_x4(
    x: usize,
    y: usize,
    xyb: &crate::image::Image3F,
    out_val: float32x4_t,
) -> float32x4_t {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        acc0 = vaddq_f32(acc0, gamma_row_sum_x4(row_x, row_y, x));
        acc1 = vaddq_f32(acc1, gamma_row_sum_x4(row_x, row_y, x + 8));
        acc2 = vaddq_f32(acc2, gamma_row_sum_x4(row_x, row_y, x + 16));
        acc3 = vaddq_f32(acc3, gamma_row_sum_x4(row_x, row_y, x + 24));
    }

    let partial01 = vpaddq_f32(acc0, acc1);
    let partial23 = vpaddq_f32(acc2, acc3);
    let overall = vmulq_n_f32(vpaddq_f32(partial01, partial23), 1.0 / 64.0);

    vmlaf(
        vdupq_n_f32(0.1005613337192697),
        dirty_log2f_x4(overall),
        out_val,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn hf_row_sum_x4(
    row: &[f32],
    row_next: &[f32],
    base: usize,
    valmin_y: float32x4_t,
    right_tail_mask: float32x4_t,
    has_vertical: bool,
) -> float32x4_t {
    let p0 = load4s(row, base);
    let p1 = load4s(row, base + 4);

    let right0 = vminq_f32(vabsq_f32(vsubq_f32(p0, load4s(row, base + 1))), valmin_y);
    let right1 = vmulq_f32(
        vminq_f32(vabsq_f32(vsubq_f32(p1, load4s(row, base + 5))), valmin_y),
        right_tail_mask,
    );
    let mut sum = vaddq_f32(right0, right1);

    if has_vertical {
        let down0 = vminq_f32(vabsq_f32(vsubq_f32(p0, load4s(row_next, base))), valmin_y);
        let down1 = vminq_f32(
            vabsq_f32(vsubq_f32(p1, load4s(row_next, base + 4))),
            valmin_y,
        );
        sum = vaddq_f32(sum, vaddq_f32(down0, down1));
    }

    sum
}

#[inline]
#[target_feature(enable = "neon")]
fn hf_modulation_blocks4_direct_x4(
    x: usize,
    y: usize,
    xyb_y: &crate::image::Image3F,
    out_val: float32x4_t,
    strength: f32,
) -> float32x4_t {
    let valmin_y = vdupq_n_f32(0.0206);

    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let mut s3 = vdupq_n_f32(0.0);

    for dy in 0..8 {
        let row = xyb_y.plane_row(1, y + dy);
        let row_next = if dy == 7 {
            row
        } else {
            xyb_y.plane_row(1, y + dy + 1)
        };

        s0 = vaddq_f32(
            s0,
            hf_row_sum_x4(row, row_next, x, valmin_y, vdupq_n_f32(1.0), dy != 7),
        );
        s1 = vaddq_f32(
            s1,
            hf_row_sum_x4(row, row_next, x + 8, valmin_y, vdupq_n_f32(1.0), dy != 7),
        );
        s2 = vaddq_f32(
            s2,
            hf_row_sum_x4(row, row_next, x + 16, valmin_y, vdupq_n_f32(1.0), dy != 7),
        );
        s3 = vaddq_f32(
            s3,
            hf_row_sum_x4(row, row_next, x + 24, valmin_y, vdupq_n_f32(1.0), dy != 7),
        );
    }

    let out = vpaddq_f32(vpaddq_f32(s0, s1), vpaddq_f32(s2, s3));

    vmlaf(
        out,
        vdupq_n_f32(-0.38 * strength),
        vaddq_f32(out_val, vdupq_n_f32(0.42)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn blue_row_sum_x4(row_x: &[f32], row_y: &[f32], row_b: &[f32], base: usize) -> float32x4_t {
    let k_limit = vdupq_n_f32(0.010474084867598155);
    let k_offset = vdupq_n_f32(0.0031994768654636393);
    let zero = vdupq_n_f32(0.0);

    let x0 = load4s(row_x, base);
    let y0 = load4s(row_y, base);
    let b0 = load4s(row_b, base);
    let y_eff0 = vaddq_f32(vaddq_f32(y0, k_offset), vabsq_f32(x0));
    let s0 = vminq_f32(vmaxq_f32(vsubq_f32(b0, y_eff0), zero), k_limit);

    let x1 = load4s(row_x, base + 4);
    let y1 = load4s(row_y, base + 4);
    let b1 = load4s(row_b, base + 4);
    let y_eff1 = vaddq_f32(vaddq_f32(y1, k_offset), vabsq_f32(x1));
    let s1 = vminq_f32(vmaxq_f32(vsubq_f32(b1, y_eff1), zero), k_limit);

    vaddq_f32(s0, s1)
}

#[inline]
#[target_feature(enable = "neon")]
fn blue_modulation_blocks4_x4(
    x: usize,
    y: usize,
    xyb: &crate::image::Image3F,
    out_val: float32x4_t,
) -> float32x4_t {
    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let mut s3 = vdupq_n_f32(0.0);

    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        let row_b = xyb.plane_row(2, y + dy);

        s0 = vaddq_f32(s0, blue_row_sum_x4(row_x, row_y, row_b, x));
        s1 = vaddq_f32(s1, blue_row_sum_x4(row_x, row_y, row_b, x + 8));
        s2 = vaddq_f32(s2, blue_row_sum_x4(row_x, row_y, row_b, x + 16));
        s3 = vaddq_f32(s3, blue_row_sum_x4(row_x, row_y, row_b, x + 24));
    }

    const K_LIMIT: f32 = 0.010474084867598155;
    const K_MAX_LIMIT: f32 = 15.463398341612438 * K_LIMIT;
    let scale: f32 = 0.90590804735610064;

    // Convert:
    //
    //   s0 = [a0, a1, a2, a3]
    //   s1 = [b0, b1, b2, b3]
    //   s2 = [c0, c1, c2, c3]
    //   s3 = [d0, d1, d2, d3]
    //
    // into:
    //
    //   sums = [
    //     a0+a1+a2+a3,
    //     b0+b1+b2+b3,
    //     c0+c1+c2+c3,
    //     d0+d1+d2+d3,
    //   ]
    //
    let p01 = vpaddq_f32(s0, s1);
    let p23 = vpaddq_f32(s2, s3);
    let mut sums = vpaddq_f32(p01, p23);

    // if v >= 32.0 * K_LIMIT {
    //     v = 64.0 * K_LIMIT - v;
    // }
    let flip_mask = vcgeq_f32(sums, vdupq_n_f32(32.0 * K_LIMIT));
    let flipped = vsubq_f32(vdupq_n_f32(64.0 * K_LIMIT), sums);
    sums = vbslq_f32(flip_mask, flipped, sums);

    // if v >= K_MAX_LIMIT {
    //     v = K_MAX_LIMIT;
    // }
    sums = vminq_f32(sums, vdupq_n_f32(K_MAX_LIMIT));

    // out_val + sums * scale
    vfmaq_f32(out_val, sums, vdupq_n_f32(scale))
}

pub(crate) const EXP2_P0: f32 = 1.00000011920928955078125_f32;
pub(crate) const EXP2_P1: f32 = 0.69314706325531005859375_f32;
pub(crate) const EXP2_P2: f32 = 0.24022041261196136474609375_f32;
pub(crate) const EXP2_P3: f32 = 5.550567805767059326171875e-2_f32;
pub(crate) const EXP2_P4: f32 = 9.678089059889316558837890625e-3_f32;
pub(crate) const EXP2_P5: f32 = 1.33218802511692047119140625e-3_f32;

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn vpow2ifq_s32(q: int32x4_t) -> int32x4_t {
    vshlq_n_s32::<23>(vaddq_s32(q, vdupq_n_s32(0x7f)))
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn fast_exp2_x4(v: float32x4_t) -> float32x4_t {
    // exp2(x) = 2^q * 2^r
    // q = round(x), r = x - q, so r is approximately in [-0.5, 0.5].
    let q = vcvtaq_s32_f32(v);
    let qf = vcvtq_f32_s32(q);
    let r = vsubq_f32(v, qf);

    let mut p = vdupq_n_f32(EXP2_P5);
    p = vfmaq_f32(vdupq_n_f32(EXP2_P4), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP2_P3), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP2_P2), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP2_P1), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP2_P0), p, r);

    let scale = vreinterpretq_f32_s32(vpow2ifq_s32(q));
    vmulq_f32(p, scale)
}

#[inline]
#[target_feature(enable = "neon")]
fn store_quant_u8x4(qf_row: &mut [u8], bx: usize, qi: int32x4_t) {
    let qi = vmaxq_s32(vdupq_n_s32(1), vminq_s32(qi, vdupq_n_s32(255)));
    let qi16 = vqmovun_s32(qi);
    let qi8 = vqmovn_u16(vcombine_u16(qi16, vdup_n_u16(0)));
    unsafe {
        vst1_lane_u32::<0>(
            qf_row.as_mut_ptr().add(bx).cast::<u32>(),
            vreinterpret_u32_u8(qi8),
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

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn write_quant_row_neon(
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
    let exp_mul = vdupq_n_f32(1.442695041);
    let mul_v = vdupq_n_f32(mul);
    let add_v = vdupq_n_f32(add);
    let inv_scale_v = vdupq_n_f32(inv_scale);
    let half = vdupq_n_f32(0.5);

    // Fully interior groups of four adjacent 8x8 blocks. The +33 requirement is
    // for the HF right-difference load of the fourth block, whose masked-out lane
    // would otherwise read one byte/float past the image at exact right edges.
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
        let out_val = vminq_f32(hf, blue);
        let qf = vaddq_f32(
            vmulq_f32(fast_exp2_x4(vmulq_f32(out_val, exp_mul)), mul_v),
            add_v,
        );
        let qi = vcvtq_s32_f32(vmlaf(qf, inv_scale_v, half));
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
#[target_feature(enable = "neon")]
fn stage1_diff_x4(
    row_y: &[f32],
    row_y1: &[f32],
    row_y2: &[f32],
    gx: usize,
    offset: float32x4_t,
    quarter: float32x4_t,
    limit: float32x4_t,
) -> float32x4_t {
    let cy = load4s(row_y, gx);
    let ly = load4s(row_y, gx - 1);
    let ry = load4s(row_y, gx + 1);
    let uy = load4s(row_y1, gx);
    let dy = load4s(row_y2, gx);
    let base_y = vmulq_f32(quarter, vaddq_f32(vaddq_f32(vaddq_f32(dy, uy), ly), ry));
    let gammac = ratio_cubic_x4::<false>(vaddq_f32(cy, offset));
    let dyv = vmulq_f32(gammac, vsubq_f32(cy, base_y));
    let diff = vminq_f32(vmulq_f32(dyv, dyv), limit);
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
        crate::adaptive_quant::ratio_cubic_to_simple_gamma::<false>(in_y + MATCH_GAMMA_OFFSET);

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

#[target_feature(enable = "neon")]
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

    let offset = vdupq_n_f32(MATCH_GAMMA_OFFSET);
    let quarter = vdupq_n_f32(0.25);
    let limit = vdupq_n_f32(0.2);
    let zero = vdupq_n_f32(0.0);

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
                c0 = vaddq_f32(
                    c0,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx, offset, quarter, limit),
                );
                c1 = vaddq_f32(
                    c1,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx + 4, offset, quarter, limit),
                );
                c2 = vaddq_f32(
                    c2,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx + 8, offset, quarter, limit),
                );
                c3 = vaddq_f32(
                    c3,
                    stage1_diff_x4(row_y, row_y1, row_y2, gx + 12, offset, quarter, limit),
                );
            }

            let p01 = vpaddq_f32(c0, c1);
            let p23 = vpaddq_f32(c2, c3);
            let sum = vpaddq_f32(p01, p23);
            store4(vmulq_f32(sum, quarter), prow, px);

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
#[target_feature(enable = "neon")]
fn cs_f32x4(a: &mut float32x4_t, b: &mut float32x4_t) {
    let lo = vminq_f32(*a, *b);
    let hi = vmaxq_f32(*a, *b);
    *a = lo;
    *b = hi;
}

#[inline]
#[target_feature(enable = "neon")]
fn sort4_f32x4(
    m0: &mut float32x4_t,
    m1: &mut float32x4_t,
    m2: &mut float32x4_t,
    m3: &mut float32x4_t,
) {
    cs_f32x4(m0, m1);
    cs_f32x4(m2, m3);
    cs_f32x4(m0, m2);
    cs_f32x4(m1, m3);
    cs_f32x4(m1, m2);
}

#[inline]
#[target_feature(enable = "neon")]
fn insert_min4_f32x4(
    m0: &mut float32x4_t,
    m1: &mut float32x4_t,
    m2: &mut float32x4_t,
    m3: &mut float32x4_t,
    v: float32x4_t,
) {
    let n0 = vminq_f32(*m0, v);
    let mut t = vmaxq_f32(*m0, v);
    *m0 = n0;

    let n1 = vminq_f32(*m1, t);
    t = vmaxq_f32(*m1, t);
    *m1 = n1;

    let n2 = vminq_f32(*m2, t);
    t = vmaxq_f32(*m2, t);
    *m2 = n2;

    *m3 = vminq_f32(*m3, t);
}

#[inline]
#[target_feature(enable = "neon")]
fn fuzzy_erosion_x4(
    rowt: &[f32],
    row: &[f32],
    rowb: &[f32],
    fx: usize,
    k0: float32x4_t,
    k1: float32x4_t,
    k2: float32x4_t,
    k3: float32x4_t,
) -> float32x4_t {
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

    let mut v = vmulq_f32(k0, m0);
    v = vfmaq_f32(v, k1, m1);
    v = vfmaq_f32(v, k2, m2);
    v = vfmaq_f32(v, k3, m3);
    v
}

#[inline]
#[target_feature(enable = "neon")]
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

    let k0 = vdupq_n_f32(kmul[0]);
    let k1 = vdupq_n_f32(kmul[1]);
    let k2 = vdupq_n_f32(kmul[2]);
    let k3 = vdupq_n_f32(kmul[3]);

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

        let even = vuzp1q_f32(v0, v1);
        let odd = vuzp2q_f32(v0, v1);

        if SET_MODE {
            store4(vaddq_f32(even, odd), aq_row, out_x);
        } else {
            let acc = load4s(aq_row, out_x);
            let acc = vaddq_f32(acc, even);
            let acc = vaddq_f32(acc, odd);
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

/// SIMD implementation of the three reductions used by
/// `apply_chroma_hf_protection`. Each pass keeps its vector accumulators live
/// across the complete block and reduces only once. Partial right/bottom
/// blocks use scalar tails.
#[target_feature(enable = "neon")]
pub(crate) fn chroma_hf_stats_neon(
    opsin: &crate::image::Image3F,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) -> crate::adaptive_quant::ChromaHfStats {
    let mut out = crate::adaptive_quant::ChromaHfStats::default();

    // Pass 1: opponent magnitude.
    let mut magnitude = vdupq_n_f32(0.0);
    for y in py..py + h {
        let xr = opsin.plane_row(0, y);
        let yr = opsin.plane_row(1, y);
        let br = opsin.plane_row(2, y);
        let mut dx = 0;
        while dx + 4 <= w {
            let x = px + dx;
            let xv = load4s(xr, x);
            let cb = vsubq_f32(load4s(br, x), load4s(yr, x));
            magnitude = vaddq_f32(magnitude, vaddq_f32(vabsq_f32(xv), vabsq_f32(cb)));
            dx += 4;
        }
        let xr = &xr[px + dx..px + w];
        let yr = &yr[px + dx..px + w];
        let br = &br[px + dx..px + w];
        for ((&x, &y), &b) in xr.iter().zip(yr).zip(br) {
            out.chroma_sum += x.abs() + (b - y).abs();
        }
    }
    out.chroma_sum += hsum(magnitude);

    // Pass 2: horizontal edges.
    if w > 1 {
        let width = w - 1;
        let mut chroma4 = vdupq_n_f32(0.0);
        let mut luma4 = vdupq_n_f32(0.0);
        for y in py..py + h {
            let xr = opsin.plane_row(0, y);
            let yr = opsin.plane_row(1, y);
            let br = opsin.plane_row(2, y);
            let mut dx = 0;
            while dx + 4 <= width {
                let x = px + dx;
                let x0 = load4s(xr, x);
                let x1 = load4s(xr, x + 1);
                let y0 = load4s(yr, x);
                let y1 = load4s(yr, x + 1);
                let cb0 = vsubq_f32(load4s(br, x), y0);
                let cb1 = vsubq_f32(load4s(br, x + 1), y1);
                let chroma =
                    vaddq_f32(vabsq_f32(vsubq_f32(x1, x0)), vabsq_f32(vsubq_f32(cb1, cb0)));
                chroma4 = vaddq_f32(chroma4, chroma);
                luma4 = vaddq_f32(luma4, vabsq_f32(vsubq_f32(y1, y0)));
                dx += 4;
            }
            let start = px + dx;
            let end = px + width;
            let x_pairs = xr[start..end].iter().zip(&xr[start + 1..end + 1]);
            let y_pairs = yr[start..end].iter().zip(&yr[start + 1..end + 1]);
            let b_pairs = br[start..end].iter().zip(&br[start + 1..end + 1]);
            for (((&x0, &x1), (&y0, &y1)), (&b0, &b1)) in x_pairs.zip(y_pairs).zip(b_pairs) {
                out.chroma_grad += (x1 - x0).abs() + ((b1 - y1) - (b0 - y0)).abs();
                out.luma_grad += (y1 - y0).abs();
            }
        }
        out.chroma_grad += hsum(chroma4);
        out.luma_grad += hsum(luma4);
        out.edges += width * h;
    }

    // Pass 3: vertical edges.
    if h > 1 {
        let mut chroma4 = vdupq_n_f32(0.0);
        let mut luma4 = vdupq_n_f32(0.0);
        for y in py..py + h - 1 {
            let xr = opsin.plane_row(0, y);
            let yr = opsin.plane_row(1, y);
            let br = opsin.plane_row(2, y);
            let nxr = opsin.plane_row(0, y + 1);
            let nyr = opsin.plane_row(1, y + 1);
            let nbr = opsin.plane_row(2, y + 1);
            let mut dx = 0;
            while dx + 4 <= w {
                let x = px + dx;
                let x0 = load4s(xr, x);
                let x1 = load4s(nxr, x);
                let y0 = load4s(yr, x);
                let y1 = load4s(nyr, x);
                let cb0 = vsubq_f32(load4s(br, x), y0);
                let cb1 = vsubq_f32(load4s(nbr, x), y1);
                let chroma =
                    vaddq_f32(vabsq_f32(vsubq_f32(x1, x0)), vabsq_f32(vsubq_f32(cb1, cb0)));
                chroma4 = vaddq_f32(chroma4, chroma);
                luma4 = vaddq_f32(luma4, vabsq_f32(vsubq_f32(y1, y0)));
                dx += 4;
            }
            let start = px + dx;
            let end = px + w;
            let x_pairs = xr[start..end].iter().zip(&nxr[start..end]);
            let y_pairs = yr[start..end].iter().zip(&nyr[start..end]);
            let b_pairs = br[start..end].iter().zip(&nbr[start..end]);
            for (((&x0, &x1), (&y0, &y1)), (&b0, &b1)) in x_pairs.zip(y_pairs).zip(b_pairs) {
                out.chroma_grad += (x1 - x0).abs() + ((b1 - y1) - (b0 - y0)).abs();
                out.luma_grad += (y1 - y0).abs();
            }
        }
        out.chroma_grad += hsum(chroma4);
        out.luma_grad += hsum(luma4);
        out.edges += w * (h - 1);
    }

    out
}

#[target_feature(enable = "neon")]
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
        write_quant_row_neon(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_quant::dirty_log1pf;

    #[test]
    fn dirty_log1pf_x4_matches_scalar_and_ln_1p() {
        // NEON path only runs where the intrinsics are available.
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let inputs = [0.0f32, 1e-4, 0.5, 1.0, 42.0, 256.0, 1.0e5, 5.0e6];
        for chunk in inputs.chunks(4) {
            let mut buf = [0f32; 4];
            buf[..chunk.len()].copy_from_slice(chunk);
            let out = unsafe {
                let v = vld1q_f32(buf.as_ptr());
                let r = dirty_log1pf_x4(v);
                let mut o = [0f32; 4];
                vst1q_f32(o.as_mut_ptr(), r);
                o
            };
            for (i, &x) in chunk.iter().enumerate() {
                // Lane-for-lane identical to the scalar reference.
                assert_eq!(out[i], dirty_log1pf(x), "lane mismatch at x={x}");
                // And close to the true ln(1+x): absolute error is tiny everywhere.
                let want = x.ln_1p();
                assert!(
                    (out[i] - want).abs() < 1e-5,
                    "ln_1p({x}) abs err {}",
                    (out[i] - want).abs()
                );
            }
        }
    }
}
