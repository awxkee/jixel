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
/// Caller must ensure `sse4.1` is available (checked via [`supported`]). All
/// slice accesses are bounds-validated against `width*height`.
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
    thr: &[f32; 4],
) -> (f32, usize, f32) {
    let n = width * height;
    assert!(coeff.len() >= n && inv_matrix.len() >= n);
    debug_assert!(width.is_multiple_of(4) && half.is_multiple_of(4));

    let qs = _mm_set1_ps(q_scaled);
    let mut sse_acc = _mm_setzero_ps();
    let mut nzeros = 0usize;
    let mut mag_bits = 0.0f32;

    // round-to-nearest-even via SSE4.1
    const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    let mut qbuf = [0.0f32; 4];

    for y in 0..height {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];
        let row = y * width;
        let mut x = 0;
        while x < width {
            let threshold = if x >= half { thr_hi } else { thr_lo };
            let thrv = _mm_set1_ps(threshold);
            let idx = row + x;
            let cv = unsafe { _mm_loadu_ps(coeff[idx..idx + 4].as_ptr()) };
            let mv = unsafe { _mm_loadu_ps(inv_matrix[idx..idx + 4].as_ptr()) };
            // a = inv_matrix * q_scaled * coeff
            let a = _mm_mul_ps(_mm_mul_ps(mv, qs), cv);
            let absa = _mm_andnot_ps(_mm_set1_ps(-0.0), a);
            // keep = |a| >= threshold
            let keep = _mm_cmpge_ps(absa, thrv);
            let rounded = _mm_round_ps::<RND>(a);
            let q = _mm_and_ps(rounded, keep); // zero where not kept
            let d = _mm_sub_ps(a, q);
            let mut d2 = _mm_mul_ps(d, d);

            // Mask out LLF lanes (x'<cx && y<cy) so they contribute nothing.
            if y < cy && x == 0 {
                // lanes 0..cx are LLF → 0.0; all other lanes kept → 1.0
                let mut lanemask = [1.0f32; 4];

                for lane in 0..cx.min(4) {
                    lanemask[lane] = 0.0;
                }
                let lm = unsafe { _mm_loadu_ps(lanemask.as_ptr()) };
                d2 = _mm_mul_ps(d2, lm); // zero LLF squared-errors
            }
            sse_acc = _mm_add_ps(sse_acc, d2);

            // Rate: count nonzeros and accumulate log2(1+|q|) from kept lanes.
            unsafe {
                _mm_storeu_ps(qbuf.as_mut_ptr(), q);
            }
            for lane in 0..4 {
                if y < cy && x == 0 && lane < cx {
                    continue; // LLF
                }
                let qv = qbuf[lane];
                if qv != 0.0 {
                    nzeros += 1;
                    mag_bits += crate::enc_ac_strategy::rate_log2(qv.abs());
                }
            }
            x += 4;
        }
    }
    (hsum(sse_acc), nzeros, mag_bits)
}
