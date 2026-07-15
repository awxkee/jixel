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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn hsum(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v); // [1,1,3,3]
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums)
}

/// SSE4.1 implementation of `sse_and_rate_scalar`.
///
/// # Safety
/// Caller must ensure `sse4.1` is available. All slice accesses are
/// bounds-validated against `width*height`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn sse_and_rate_sse(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    _rate_log2_lut: &crate::inflated_cost::RateLog2Lut,
    thr: &[f32; 4],
) -> (f32, usize, f32) {
    let n = width * height;
    assert!(coeff.len() >= n && inv_matrix.len() >= n);
    debug_assert!(width.is_multiple_of(4) && half.is_multiple_of(4));

    let qs = _mm_set1_ps(q_scaled);
    let mut sse_acc = _mm_setzero_ps();
    let mut mag_acc = _mm_setzero_ps();
    let mut nzeros = 0usize;

    let zero = _mm_setzero_ps();
    let sign_mask = _mm_set1_ps(-0.0);

    let lane_ids = _mm_setr_epi32(0, 1, 2, 3);
    let izero = _mm_setzero_si128();
    let all_active_i = _mm_cmpeq_epi32(izero, izero);

    // round-to-nearest-even via SSE4.1
    const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    for (y, (coeffs, inv_matrix)) in coeff
        .chunks_exact(width)
        .zip(inv_matrix.chunks_exact(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];

        for (x0, (coeff4, inv4)) in coeffs
            .as_chunks::<4>()
            .0
            .iter()
            .zip(inv_matrix.as_chunks::<4>().0.iter())
            .enumerate()
        {
            let x = x0 * 4;

            let threshold = if x >= half { thr_hi } else { thr_lo };
            let thrv = _mm_set1_ps(threshold);

            let cv = unsafe { _mm_loadu_ps(coeff4.as_ptr()) };
            let mv = unsafe { _mm_loadu_ps(inv4.as_ptr()) };

            // a = inv_matrix * q_scaled * coeff
            let a = _mm_mul_ps(_mm_mul_ps(mv, qs), cv);
            let absa = _mm_andnot_ps(sign_mask, a);

            // keep = |a| >= threshold
            let keep = _mm_cmpge_ps(absa, thrv);

            let rounded = _mm_round_ps::<RND>(a);
            let q = _mm_and_ps(rounded, keep);

            let d = _mm_sub_ps(a, q);
            let d2 = _mm_mul_ps(d, d);

            // Active mask: keep lanes where !(y < cy && x + lane < cx).
            let active_i = if y < cy && x < cx {
                let lane_x = _mm_add_epi32(_mm_set1_epi32(x as i32), lane_ids);
                let cxv = _mm_set1_epi32(cx as i32);

                // llf = cx > lane_x  <=> lane_x < cx
                let llf = _mm_cmpgt_epi32(cxv, lane_x);

                // active = !llf
                _mm_andnot_si128(llf, all_active_i)
            } else {
                all_active_i
            };

            let active = _mm_castsi128_ps(active_i);

            // SSE: zero LLF lanes.
            let d2 = _mm_and_ps(d2, active);
            sse_acc = _mm_add_ps(sse_acc, d2);

            // Rate mask: active && abs(q) > 0.
            let absq = _mm_andnot_ps(sign_mask, q);
            let nz = _mm_cmpgt_ps(absq, zero);
            let rate_mask = _mm_and_ps(nz, active);

            let rate_bits = _mm_movemask_ps(rate_mask);
            nzeros += rate_bits.count_ones() as usize;

            if rate_bits != 0 {
                let ratev = sse_log2p1_f32(absq);
                mag_acc = _mm_add_ps(mag_acc, _mm_and_ps(ratev, rate_mask));
            }
        }
    }

    (hsum(sse_acc), nzeros, hsum(mag_acc))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse_log2p1_f32(x: __m128) -> __m128 {
    // y = 1 + x
    let y = _mm_add_ps(x, _mm_set1_ps(1.0));

    // Decompose y = m * 2^e, m in [1, 2).
    let bits = _mm_castps_si128(y);

    let exp_u = _mm_srli_epi32(bits, 23);
    let exp_i = _mm_sub_epi32(exp_u, _mm_set1_epi32(127));
    let e = _mm_cvtepi32_ps(exp_i);

    let mant_bits = _mm_or_si128(
        _mm_and_si128(bits, _mm_set1_epi32(0x007f_ffff)),
        _mm_set1_epi32(0x3f80_0000),
    );
    let m = _mm_castsi128_ps(mant_bits);

    let t = _mm_sub_ps(m, _mm_set1_ps(1.0));

    // stats/log2p1.sollya
    let c0 = _mm_set1_ps(1.4426934719085693359375);
    let c1 = _mm_set1_ps(-0.721179187297821044921875);
    let c2 = _mm_set1_ps(0.477900087833404541015625);
    let c3 = _mm_set1_ps(-0.340080082416534423828125);
    let c4 = _mm_set1_ps(0.21719777584075927734375);
    let c5 = _mm_set1_ps(-9.749893844127655029296875e-2);
    let c6 = _mm_set1_ps(2.096841670572757720947265625e-2);

    let mut p = c6;
    p = _mm_add_ps(c5, _mm_mul_ps(t, p));
    p = _mm_add_ps(c4, _mm_mul_ps(t, p));
    p = _mm_add_ps(c3, _mm_mul_ps(t, p));
    p = _mm_add_ps(c2, _mm_mul_ps(t, p));
    p = _mm_add_ps(c1, _mm_mul_ps(t, p));
    p = _mm_add_ps(c0, _mm_mul_ps(t, p));

    let log2_m = _mm_mul_ps(t, p);

    _mm_add_ps(e, log2_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_and_rate_sse_vs_reference() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        crate::inflated_cost::assert_sse_and_rate_matches_reference(sse_and_rate_sse);
    }

    #[test]
    fn test_log2p1() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        unsafe {
            for i in 0..5000 {
                let q1 = ((i as f32 / 5000.) + 1.).log2();
                let q2 = f32::from_bits(_mm_extract_ps::<0>(sse_log2p1_f32(_mm_set1_ps(
                    i as f32 / 5000.,
                ))) as u32);
                assert!((q1 - q2).abs() < 1e-5, "q1 {} q2 {}", q1, q2);
            }
        }
    }
}
