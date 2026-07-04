/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::enc_xyb::*;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn halley_cbrt(x: float32x4_t, a: float32x4_t) -> float32x4_t {
    let tx = vmulq_f32(vmulq_f32(x, x), x);
    let num = vfmaq_n_f32(tx, a, 2.0);
    let den = vfmaq_n_f32(a, tx, 2.0);
    vmulq_f32(x, vdivq_f32(num, den))
}

#[inline]
#[target_feature(enable = "neon")]
fn integer_pow_1_3(hx: uint32x4_t) -> uint32x4_t {
    let scale = vdupq_n_u32(341);
    let hi = vshrq_n_u64::<10>(vmull_high_u32(hx, scale));
    let lo = vshrq_n_u64::<10>(vmull_u32(vget_low_u32(hx), vget_low_u32(scale)));
    vcombine_u32(vmovn_u64(lo), vmovn_u64(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn cbrt_seed_positive_f32(x: float32x4_t) -> float32x4_t {
    let hx = vreinterpretq_u32_f32(x);
    let hx = vaddq_u32(integer_pow_1_3(hx), vdupq_n_u32(709958130));
    vreinterpretq_f32_u32(hx)
}

#[inline]
#[target_feature(enable = "neon")]
fn vcbrtq_fast3_positive_f32(
    a0: float32x4_t,
    a1: float32x4_t,
    a2: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let mut x0 = cbrt_seed_positive_f32(a0);
    let mut x1 = cbrt_seed_positive_f32(a1);
    let mut x2 = cbrt_seed_positive_f32(a2);

    x0 = halley_cbrt(x0, a0);
    x1 = halley_cbrt(x1, a1);
    x2 = halley_cbrt(x2, a2);

    x0 = halley_cbrt(x0, a0);
    x1 = halley_cbrt(x1, a1);
    x2 = halley_cbrt(x2, a2);

    (x0, x1, x2)
}

#[inline]
#[target_feature(enable = "neon")]
fn rgb_to_xyb_f32x4_neon(
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let bias = vdupq_n_f32(OPSIN_BIAS);

    let mut mixed0 = vfmaq_n_f32(bias, b, M02);
    mixed0 = vfmaq_n_f32(mixed0, g, M01);
    mixed0 = vfmaq_n_f32(mixed0, r, M00);

    let mut mixed1 = vfmaq_n_f32(bias, b, M12);
    mixed1 = vfmaq_n_f32(mixed1, g, M11);
    mixed1 = vfmaq_n_f32(mixed1, r, M10);

    let mut mixed2 = vfmaq_n_f32(bias, b, M22);
    mixed2 = vfmaq_n_f32(mixed2, g, M21);
    mixed2 = vfmaq_n_f32(mixed2, r, M20);

    let zero = vdupq_n_f32(0.0);
    mixed0 = vmaxq_f32(mixed0, zero);
    mixed1 = vmaxq_f32(mixed1, zero);
    mixed2 = vmaxq_f32(mixed2, zero);

    let (tm0, tm1, tm2) = vcbrtq_fast3_positive_f32(mixed0, mixed1, mixed2);

    let neg_bias = vdupq_n_f32(NEG_BIAS_CBRT);
    let tm0 = vaddq_f32(tm0, neg_bias);
    let tm1 = vaddq_f32(tm1, neg_bias);
    let tm2 = vaddq_f32(tm2, neg_bias);

    let half = vdupq_n_f32(0.5);

    let x = vmulq_f32(vsubq_f32(tm0, tm1), half);
    let y = vmulq_f32(vaddq_f32(tm0, tm1), half);

    (x, y, tm2)
}

/// Transform one row-band in place.
#[target_feature(enable = "neon")]
pub(crate) fn to_xyb_neon_band(band: [&mut [f32]; 3], w: usize) {
    let [rp, gp, bp] = band;
    for ((r_row, g_row), b_row) in rp
        .chunks_exact_mut(w)
        .zip(gp.chunks_exact_mut(w))
        .zip(bp.chunks_exact_mut(w))
    {
        let (r_chunks, r_tail) = r_row.as_chunks_mut::<4>();
        let (g_chunks, g_tail) = g_row.as_chunks_mut::<4>();
        let (b_chunks, b_tail) = b_row.as_chunks_mut::<4>();

        for ((r4, g4), b4) in r_chunks
            .iter_mut()
            .zip(g_chunks.iter_mut())
            .zip(b_chunks.iter_mut())
        {
            let r = unsafe { vld1q_f32(r4.as_ptr()) };
            let g = unsafe { vld1q_f32(g4.as_ptr()) };
            let b = unsafe { vld1q_f32(b4.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_neon(r, g, b);

            unsafe {
                vst1q_f32(r4.as_mut_ptr(), xv);
                vst1q_f32(g4.as_mut_ptr(), yv);
                vst1q_f32(b4.as_mut_ptr(), bv);
            }
        }

        if !r_tail.is_empty() {
            let mut r4: [f32; 4] = [0.; 4];
            let mut g4: [f32; 4] = [0.; 4];
            let mut b4: [f32; 4] = [0.; 4];
            r4[..r_tail.len()].copy_from_slice(r_tail);
            g4[..g_tail.len()].copy_from_slice(g_tail);
            b4[..b_tail.len()].copy_from_slice(b_tail);
            let r = unsafe { vld1q_f32(r4.as_ptr()) };
            let g = unsafe { vld1q_f32(g4.as_ptr()) };
            let b = unsafe { vld1q_f32(b4.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_neon(r, g, b);

            unsafe {
                vst1q_f32(r4.as_mut_ptr(), xv);
                vst1q_f32(g4.as_mut_ptr(), yv);
                vst1q_f32(b4.as_mut_ptr(), bv);
            }

            r_tail.copy_from_slice(&r4[..r_tail.len()]);
            g_tail.copy_from_slice(&g4[..g_tail.len()]);
            b_tail.copy_from_slice(&b4[..b_tail.len()]);
        }
    }
}
