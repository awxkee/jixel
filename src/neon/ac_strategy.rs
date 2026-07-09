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
use std::arch::aarch64::*;

/// NEON implementation of `sse_and_rate_scalar`.
///
/// # Safety
/// Caller must ensure `neon` is available (checked via [`supported`]); on
/// AArch64 NEON is part of the baseline ISA. All slice accesses are
/// bounds-validated against `width*height`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn sse_and_rate_neon(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    _rate_log2_lut: &crate::enc_ac_strategy::RateLog2Lut,
    thr: &[f32; 4],
) -> (f32, usize, f32) {
    let n = width * height;
    assert!(coeff.len() >= n && inv_matrix.len() >= n);
    debug_assert!(width.is_multiple_of(4) && half.is_multiple_of(4));

    let qs = vdupq_n_f32(q_scaled);
    let mut sse_acc = vdupq_n_f32(0.0);
    let mut nzeros = 0usize;

    let mut mag_acc = vdupq_n_f32(0.0);
    let zero = vdupq_n_f32(0.0);
    let all_active = vdupq_n_u32(u32::MAX);
    let lane_ids = unsafe { vld1q_u32([0u32, 1, 2, 3].as_ptr()) };

    for (y, (coeffs, inv_matrix)) in coeff
        .chunks_exact(width)
        .zip(inv_matrix.chunks_exact(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];

        for (x0, (coeff, inv_matrix)) in coeffs
            .as_chunks::<4>()
            .0
            .iter()
            .zip(inv_matrix.as_chunks::<4>().0.iter())
            .enumerate()
        {
            let x = x0 * 4;
            let threshold = if x >= half { thr_hi } else { thr_lo };
            let thrv = vdupq_n_f32(threshold);

            let cv = unsafe { vld1q_f32(coeff.as_ptr()) };
            let mv = unsafe { vld1q_f32(inv_matrix.as_ptr()) };

            let a = vmulq_f32(vmulq_f32(mv, qs), cv);
            let absa = vabsq_f32(a);

            let keep = vcgeq_f32(absa, thrv);
            let rounded = vrndnq_f32(a);

            let q = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(rounded), keep));

            let d = vsubq_f32(a, q);
            let d2 = vmulq_f32(d, d);

            let active = if y < cy && x < cx {
                let lane_x = vaddq_u32(vdupq_n_u32(x as u32), lane_ids);
                vcgeq_u32(lane_x, vdupq_n_u32(cx as u32))
            } else {
                all_active
            };

            let d2 = vbslq_f32(active, d2, zero);
            sse_acc = vaddq_f32(sse_acc, d2);

            let absq = vabsq_f32(q);
            let nz = vcgtq_f32(absq, zero);
            let rate_mask = vandq_u32(nz, active);

            nzeros += vaddvq_u32(vshrq_n_u32::<31>(rate_mask)) as usize;

            let ratev = neon_log2p1_f32(absq);
            mag_acc = vaddq_f32(mag_acc, vbslq_f32(rate_mask, ratev, zero));
        }
    }
    (vaddvq_f32(sse_acc), nzeros, vaddvq_f32(mag_acc))
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_log2p1_f32(x: float32x4_t) -> float32x4_t {
    // y = 1 + x
    let y = vaddq_f32(x, vdupq_n_f32(1.0));

    // Decompose y = m * 2^e, m in [1, 2).
    let bits = vreinterpretq_u32_f32(y);

    let exp_u = vshrq_n_u32::<23>(bits);
    let exp_i = vsubq_s32(vreinterpretq_s32_u32(exp_u), vdupq_n_s32(127));
    let e = vcvtq_f32_s32(exp_i);

    let mant_bits = vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f80_0000),
    );
    let m = vreinterpretq_f32_u32(mant_bits);

    let t = vsubq_f32(m, vdupq_n_f32(1.0));

    // Generated by Sollya.
    let c0 = vdupq_n_f32(1.4426934719085693359375);
    let c1 = vdupq_n_f32(-0.721179187297821044921875);
    let c2 = vdupq_n_f32(0.477900087833404541015625);
    let c3 = vdupq_n_f32(-0.340080082416534423828125);
    let c4 = vdupq_n_f32(0.21719777584075927734375);
    let c5 = vdupq_n_f32(-9.749893844127655029296875e-2);
    let c6 = vdupq_n_f32(2.096841670572757720947265625e-2);

    let mut p = c6;
    p = vfmaq_f32(c5, t, p);
    p = vfmaq_f32(c4, t, p);
    p = vfmaq_f32(c3, t, p);
    p = vfmaq_f32(c2, t, p);
    p = vfmaq_f32(c1, t, p);
    p = vfmaq_f32(c0, t, p);

    let log2_m = vmulq_f32(t, p);

    vaddq_f32(e, log2_m)
}

#[cfg(test)]
mod tests {
    use crate::neon::ac_strategy::neon_log2p1_f32;
    use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

    #[test]
    fn test_log2p1() {
        unsafe {
            for i in 0..5000 {
                let q1 = ((i as f32 / 5000.) + 1.).log2();
                let q2 = vgetq_lane_f32::<0>(neon_log2p1_f32(vdupq_n_f32(i as f32 / 5000.)));
                assert!((q1 - q2).abs() < 1e-5, "q1 {} q2 {}", q1, q2);
            }
        }
    }
}
