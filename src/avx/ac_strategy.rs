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

use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx")]
fn hsum256(v: __m256) -> f32 {
    // Fold the upper 128 into the lower, then horizontal-add the 4 lanes.
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s);
    let sums = _mm_add_ps(s, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums)
}

/// AVX implementation of `sse_and_rate_scalar`.
///
/// # Safety
/// Caller must ensure `avx` is available (checked via runtime detection in the
/// dispatcher). All slice accesses are bounds-validated against `width*height`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn sse_and_rate_avx2(
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
    // Block widths are cx*8 ∈ {8,16,32}, all multiples of 8.
    debug_assert!(width.is_multiple_of(8));

    let qs = _mm256_set1_ps(q_scaled);
    let sign = _mm256_set1_ps(-0.0);
    let mut sse_acc = _mm256_setzero_ps();
    let mut nzeros = 0usize;
    let mut mag_bits = 0.0f32;

    // round-to-nearest-even
    const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    let lut = crate::enc_ac_strategy::rate_log2_lut();
    let mut qbuf = [0.0f32; 8];

    for y in 0..height {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];
        let row = y * width;
        let mut x = 0;
        while x < width {
            // Threshold per 8-lane group. For width 16/32 (half a multiple of 8)
            // a group lies wholly on one side of `half`; for width 8 (half = 4)
            // the single group straddles it, so build a per-lane vector.
            let thr_v = if x + 8 <= half {
                _mm256_set1_ps(thr_lo)
            } else if x >= half {
                _mm256_set1_ps(thr_hi)
            } else {
                let mut t = [0.0f32; 8];
                for (i, ti) in t.iter_mut().enumerate() {
                    *ti = if x + i >= half { thr_hi } else { thr_lo };
                }
                unsafe { _mm256_loadu_ps(t.as_ptr()) }
            };

            let idx = row + x;
            let cv = unsafe { _mm256_loadu_ps(coeff[idx..idx + 8].as_ptr()) };
            let mv = unsafe { _mm256_loadu_ps(inv_matrix[idx..idx + 8].as_ptr()) };
            // a = inv_matrix * q_scaled * coeff
            let a = _mm256_mul_ps(_mm256_mul_ps(mv, qs), cv);
            let absa = _mm256_andnot_ps(sign, a);
            // keep = |a| >= threshold
            let keep = _mm256_cmp_ps::<_CMP_GE_OQ>(absa, thr_v);
            let rounded = _mm256_round_ps::<RND>(a);
            let q = _mm256_and_ps(rounded, keep); // zero where not kept
            let d = _mm256_sub_ps(a, q);
            let mut d2 = _mm256_mul_ps(d, d);

            // Mask out LLF lanes (x'<cx && y<cy) so they contribute nothing.
            if y < cy && x == 0 {
                let mut lanemask = [1.0f32; 8];
                for lm in lanemask.iter_mut().take(cx.min(8)) {
                    *lm = 0.0;
                }
                let lmv = unsafe { _mm256_loadu_ps(lanemask.as_ptr()) };
                d2 = _mm256_mul_ps(d2, lmv);
            }
            sse_acc = _mm256_add_ps(sse_acc, d2);

            // Rate: count nonzeros and accumulate log2(1+|q|) from kept lanes,
            // in increasing lane order (matches scalar/SSE → bit-identical).
            unsafe {
                _mm256_storeu_ps(qbuf.as_mut_ptr(), q);
            }
            for (lane, &qv) in qbuf.iter().enumerate() {
                if y < cy && x == 0 && lane < cx {
                    continue; // LLF
                }
                if qv != 0.0 {
                    nzeros += 1;
                    let qa = qv.abs();
                    let k = qa as usize;
                    mag_bits += if k < lut.len() {
                        lut[k]
                    } else {
                        (1.0 + qa).log2()
                    };
                }
            }
            x += 8;
        }
    }
    (hsum256(sse_acc), nzeros, mag_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round-ties-even reference matching `_mm256_round_ps::<NEAREST>` + the
    // shared LUT, summed sequentially. Verifies the AVX kernel: nzeros/mag_bits
    // must be exact; sse may differ by float reassociation (8-lane reduction).
    #[allow(clippy::too_many_arguments)]
    fn reference(
        coeff: &[f32],
        inv: &[f32],
        qs: f32,
        w: usize,
        h: usize,
        half: usize,
        cx: usize,
        cy: usize,
        thr: &[f32; 4],
    ) -> (f32, usize, f32) {
        let (mut sse, mut nz, mut mb) = (0.0f32, 0usize, 0.0f32);
        for y in 0..h {
            let yfix = if y >= h / 2 { 2 } else { 0 };
            let (lo, hi) = (thr[yfix], thr[yfix + 1]);
            for x in 0..w {
                if x < cx && y < cy {
                    continue;
                }
                let idx = y * w + x;
                let t = if x >= half { hi } else { lo };
                let a = (inv[idx] * qs) * coeff[idx];
                let q = if a.abs() >= t {
                    a.round_ties_even()
                } else {
                    0.0
                };
                let d = a - q;
                sse += d * d;
                if q != 0.0 {
                    nz += 1;
                    mb += crate::enc_ac_strategy::rate_log2(q.abs());
                }
            }
        }
        (sse, nz, mb)
    }

    #[test]
    fn test_sse_and_rate_avx2_vs_reference() {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let mut st: u64 = 0x5e5e_a11d_0f00_d00d;
        let mut rnd = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 33) as u32 as f32) / (u32::MAX as f32)
        };
        for &(w, h, half, cx, cy) in &[
            (8usize, 8usize, 4usize, 1usize, 1usize),
            (16, 16, 8, 2, 2),
            (32, 32, 16, 4, 4),
            (16, 8, 8, 2, 1),
            (8, 16, 4, 1, 2),
        ] {
            for _ in 0..200 {
                let n = w * h;
                let coeff: Vec<f32> = (0..n).map(|_| (rnd() - 0.5) * 200.0).collect();
                let inv: Vec<f32> = (0..n).map(|_| 0.001 + rnd() * 0.5).collect();
                let qs = 0.5 + rnd() * 3.0;
                let thr = [rnd() * 0.6, rnd() * 0.6, rnd() * 0.6, rnd() * 0.6];
                let r = reference(&coeff, &inv, qs, w, h, half, cx, cy, &thr);
                let a = unsafe { sse_and_rate_avx2(&coeff, &inv, qs, w, h, half, cx, cy, &thr) };
                assert_eq!(a.1, r.1, "nzeros mismatch {w}x{h}");
                assert_eq!(a.2, r.2, "mag_bits mismatch {w}x{h}");
                let rel = if r.0.abs() > 1e-3 {
                    ((a.0 - r.0) / r.0).abs()
                } else {
                    (a.0 - r.0).abs()
                };
                assert!(rel < 1e-4, "sse rel diff {rel} too large {w}x{h}");
            }
        }
    }
}
