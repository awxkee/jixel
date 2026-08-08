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
pub(crate) fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s);
    let sums = _mm_add_ps(s, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums)
}

#[inline]
#[target_feature(enable = "avx")]
fn hsum128(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[target_feature(enable = "avx2,fma")]
fn gradient_region_stats_impl<const CHROMA: bool>(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    if !w.is_multiple_of(8)
        || !h.is_multiple_of(4)
        || px0 > opsin.xsize().saturating_sub(w)
        || py0 > opsin.ysize().saturating_sub(h)
    {
        return if CHROMA {
            crate::ac_strategy::gradient_region_stats_with_chroma_scalar(opsin, px0, py0, w, h, eps)
        } else {
            crate::ac_strategy::gradient_region_stats_scalar(opsin, px0, py0, w, h, eps)
        };
    }

    let xs = opsin.xsize();
    let cw = w / 4;
    let ch = h / 4;
    let mut means = [0.0f32; 256];
    let mut within = 0.0f32;
    let mut chroma = _mm256_setzero_ps();
    const ONE_OVER_16: f32 = 1.0 / 16.0;
    let sign = _mm256_set1_ps(-0.0);

    let row_start = py0 * xs;
    let row_end = (py0 + h) * xs;
    let x_rows = &opsin.plane_data(0)[row_start..row_end];
    let y_rows = &opsin.plane_data(1)[row_start..row_end];
    let b_rows = &opsin.plane_data(2)[row_start..row_end];
    for (mean_row, ((y_rows4, x_rows4), b_rows4)) in means[..cw * ch].chunks_exact_mut(cw).zip(
        y_rows
            .chunks_exact(xs * 4)
            .zip(x_rows.chunks_exact(xs * 4))
            .zip(b_rows.chunks_exact(xs * 4)),
    ) {
        for (cx, mean_pair) in mean_row.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let mut sum = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            if CHROMA {
                for ((y_row, x_row), b_row) in y_rows4
                    .chunks_exact(xs)
                    .zip(x_rows4.chunks_exact(xs))
                    .zip(b_rows4.chunks_exact(xs))
                {
                    let y8 = &y_row[px0..px0 + w].as_chunks::<8>().0[cx];
                    let x8 = &x_row[px0..px0 + w].as_chunks::<8>().0[cx];
                    let b8 = &b_row[px0..px0 + w].as_chunks::<8>().0[cx];
                    let yv = unsafe { _mm256_loadu_ps(y8.as_ptr()) };
                    let xv = unsafe { _mm256_loadu_ps(x8.as_ptr()) };
                    let bv = unsafe { _mm256_loadu_ps(b8.as_ptr()) };
                    sum = _mm256_add_ps(sum, yv);
                    sum2 = _mm256_fmadd_ps(yv, yv, sum2);
                    let xa = _mm256_andnot_ps(sign, xv);
                    let bya = _mm256_andnot_ps(sign, _mm256_sub_ps(bv, yv));
                    chroma = _mm256_add_ps(chroma, _mm256_add_ps(xa, bya));
                }
            } else {
                for y_row in y_rows4.chunks_exact(xs) {
                    let y8 = &y_row[px0..px0 + w].as_chunks::<8>().0[cx];
                    let yv = unsafe { _mm256_loadu_ps(y8.as_ptr()) };
                    sum = _mm256_add_ps(sum, yv);
                    sum2 = _mm256_fmadd_ps(yv, yv, sum2);
                }
            }
            let sum_lo = _mm256_castps256_ps128(sum);
            let sum_hi = _mm256_extractf128_ps::<1>(sum);
            let sum2_lo = _mm256_castps256_ps128(sum2);
            let sum2_hi = _mm256_extractf128_ps::<1>(sum2);
            let m0 = hsum128(sum_lo) * ONE_OVER_16;
            let m1 = hsum128(sum_hi) * ONE_OVER_16;
            *mean_pair = [m0, m1];
            within += (hsum128(sum2_lo) * ONE_OVER_16 - m0 * m0).max(0.0);
            within += (hsum128(sum2_hi) * ONE_OVER_16 - m1 * m1).max(0.0);
        }
    }

    crate::ac_strategy::finish_gradient_region_stats(
        &means[..cw * ch],
        within,
        if CHROMA { hsum256(chroma) } else { 0.0 },
        w * h,
        eps,
    )
}

/// AVX2+FMA region statistics for the luma-only merge-risk path.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn gradient_region_stats_avx2(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    gradient_region_stats_impl::<false>(opsin, px0, py0, w, h, eps)
}

/// AVX2+FMA region statistics including XYB chroma activity.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn gradient_region_stats_with_chroma_avx2(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    gradient_region_stats_impl::<true>(opsin, px0, py0, w, h, eps)
}

/// # Safety
/// Caller must ensure `avx2` and `fma` are available.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn sse_and_rate_avx2(
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
    scan_pos: &[u32],
) -> (f32, usize, f32, u32) {
    let n = width * height;
    assert!(coeff.len() >= n && inv_matrix.len() >= n && scan_pos.len() >= n);
    debug_assert!(width.is_multiple_of(8));

    let qs = _mm256_set1_ps(q_scaled);
    let sign = _mm256_set1_ps(-0.0);
    let zero = _mm256_setzero_ps();

    let mut sse_acc = _mm256_setzero_ps();
    let mut mag_acc = _mm256_setzero_ps();
    let mut nzeros = 0usize;
    let mut scan_acc = _mm256_setzero_si256();

    let lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let izero = _mm256_setzero_si256();
    let all_active_i = _mm256_cmpeq_epi32(izero, izero);

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

        for (x0, (coeff8, inv8)) in coeffs
            .as_chunks::<8>()
            .0
            .iter()
            .zip(inv_matrix.as_chunks::<8>().0.iter())
            .enumerate()
        {
            let x = x0 * 8;

            let thr_v = if x + 8 <= half {
                _mm256_set1_ps(thr_lo)
            } else if x >= half {
                _mm256_set1_ps(thr_hi)
            } else {
                // Only needed for cases like width=8, half=4.
                let lane_x = _mm256_add_epi32(_mm256_set1_epi32(x as i32), lane_ids);
                let hi_mask_i = _mm256_cmpgt_epi32(lane_x, _mm256_set1_epi32(half as i32 - 1));
                let hi_mask = _mm256_castsi256_ps(hi_mask_i);

                _mm256_blendv_ps(_mm256_set1_ps(thr_lo), _mm256_set1_ps(thr_hi), hi_mask)
            };

            let cv = unsafe { _mm256_loadu_ps(coeff8.as_ptr()) };
            let mv = unsafe { _mm256_loadu_ps(inv8.as_ptr()) };

            let a = _mm256_mul_ps(_mm256_mul_ps(mv, qs), cv);
            let absa = _mm256_andnot_ps(sign, a);

            let keep = _mm256_cmp_ps::<_CMP_GE_OQ>(absa, thr_v);
            let rounded = _mm256_round_ps::<RND>(a);
            let q = _mm256_and_ps(rounded, keep);

            let d = _mm256_sub_ps(a, q);
            let d2 = _mm256_mul_ps(d, d);

            let active_i = if y < cy && x < cx {
                let lane_x = _mm256_add_epi32(_mm256_set1_epi32(x as i32), lane_ids);
                let cxv = _mm256_set1_epi32(cx as i32);

                // llf = x + lane < cx  <=>  cx > x + lane
                let llf = _mm256_cmpgt_epi32(cxv, lane_x);

                // active = !llf
                _mm256_andnot_si256(llf, all_active_i)
            } else {
                all_active_i
            };

            let active = _mm256_castsi256_ps(active_i);

            let d2 = _mm256_and_ps(d2, active);
            sse_acc = _mm256_add_ps(sse_acc, d2);

            let absq = _mm256_andnot_ps(sign, q);
            let nz = _mm256_cmp_ps::<_CMP_GT_OQ>(absq, zero);
            let rate_mask = _mm256_and_ps(nz, active);

            let rate_bits = _mm256_movemask_ps(rate_mask);
            nzeros += rate_bits.count_ones() as usize;

            if rate_bits != 0 {
                let ratev = avx2_log2p1_f32(absq);
                mag_acc = _mm256_add_ps(mag_acc, _mm256_and_ps(ratev, rate_mask));
                // Scan position of the nonzeros (masked lanes drop to zero,
                // which is neutral: LLF slots are never nonzero here).
                let sv = unsafe { _mm256_loadu_si256(scan_pos.as_ptr().add(y * width + x).cast()) };
                scan_acc = _mm256_max_epu32(
                    scan_acc,
                    _mm256_and_si256(sv, _mm256_castps_si256(rate_mask)),
                );
            }
        }
    }

    let v_max = _mm_max_epu32(
        _mm256_castsi256_si128(scan_acc),
        _mm256_extracti128_si256::<1>(scan_acc),
    );
    let max_scan = hmax_u32(v_max);

    (hsum256(sse_acc), nzeros, hsum256(mag_acc), max_scan)
}

#[inline]
pub(crate) const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn hmax_u32(v: __m128i) -> u32 {
    // swap the two 64-bit halves: [a,b,c,d] -> [c,d,a,b]
    let t1 = _mm_shuffle_epi32::<{ _mm_shuffle(1, 0, 3, 2) }>(v);
    let m1 = _mm_max_epu32(v, t1);

    // swap adjacent 32-bit lanes: [x,y,x,y] -> [y,x,y,x]
    let t2 = _mm_shuffle_epi32::<{ _mm_shuffle(2, 3, 0, 1) }>(m1);
    let m2 = _mm_max_epu32(m1, t2);

    _mm_cvtsi128_si32(m2) as u32
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn avx2_log2p1_f32(x: __m256) -> __m256 {
    // y = 1 + x
    let y = _mm256_add_ps(x, _mm256_set1_ps(1.0));

    // Decompose y = m * 2^e, m in [1, 2).
    let bits = _mm256_castps_si256(y);

    let exp_u = _mm256_srli_epi32::<23>(bits);
    let exp_i = _mm256_sub_epi32(exp_u, _mm256_set1_epi32(127));
    let e = _mm256_cvtepi32_ps(exp_i);

    let mant_bits = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f80_0000),
    );
    let m = _mm256_castsi256_ps(mant_bits);

    let t = _mm256_sub_ps(m, _mm256_set1_ps(1.0));

    // Generated by app/log2p1.sollya.
    let c0 = _mm256_set1_ps(1.4426934719085693359375);
    let c1 = _mm256_set1_ps(-0.721179187297821044921875);
    let c2 = _mm256_set1_ps(0.477900087833404541015625);
    let c3 = _mm256_set1_ps(-0.340080082416534423828125);
    let c4 = _mm256_set1_ps(0.21719777584075927734375);
    let c5 = _mm256_set1_ps(-9.749893844127655029296875e-2);
    let c6 = _mm256_set1_ps(2.096841670572757720947265625e-2);

    let mut p = c6;

    //   _mm256_fmadd_ps(t, p, c5); // t*p + c5
    p = _mm256_fmadd_ps(t, p, c5);
    p = _mm256_fmadd_ps(t, p, c4);
    p = _mm256_fmadd_ps(t, p, c3);
    p = _mm256_fmadd_ps(t, p, c2);
    p = _mm256_fmadd_ps(t, p, c1);
    p = _mm256_fmadd_ps(t, p, c0);

    _mm256_fmadd_ps(t, p, e)
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
        scan_pos: &[u32],
    ) -> (f32, usize, f32, u32) {
        let (mut sse, mut nz, mut mb) = (0.0f32, 0usize, 0.0f32);
        let mut max_scan = 0u32;
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
                    mb += crate::inflated_cost::rate_log2_with_lut(
                        crate::inflated_cost::rate_log2_lut(),
                        q.abs(),
                    );
                    max_scan = max_scan.max(scan_pos[idx]);
                }
            }
        }
        (sse, nz, mb, max_scan)
    }

    #[test]
    fn test_sse_and_rate_avx2_vs_reference() {
        if !std::is_x86_feature_detected!("avx") || !std::is_x86_feature_detected!("fma") {
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
            for case in 0..200 {
                let n = w * h;
                let coeff_scale = if case % 2 == 0 { 0.2 } else { 200.0 };
                let coeff: Vec<f32> = (0..n).map(|_| (rnd() - 0.5) * coeff_scale).collect();
                let inv: Vec<f32> = (0..n).map(|_| 0.001 + rnd() * 0.5).collect();
                let qs = 0.5 + rnd() * 3.0;
                let thr = [rnd() * 0.6, rnd() * 0.6, rnd() * 0.6, rnd() * 0.6];
                let scan_pos = crate::coeff_order::scan_pos_lut(w, h);
                let r = reference(&coeff, &inv, qs, w, h, half, cx, cy, &thr, scan_pos);
                let a = unsafe {
                    sse_and_rate_avx2(
                        &coeff,
                        &inv,
                        qs,
                        w,
                        h,
                        half,
                        cx,
                        cy,
                        crate::inflated_cost::rate_log2_lut(),
                        &thr,
                        scan_pos,
                    )
                };
                // `nzeros` is a discrete decision and must match exactly.
                assert_eq!(a.1, r.1, "nzeros mismatch {w}x{h}");
                assert_eq!(a.3, r.3, "max-scan mismatch {w}x{h}");
                let mag_rel = if r.2.abs() > 1e-3 {
                    ((a.2 - r.2) / r.2).abs()
                } else {
                    (a.2 - r.2).abs()
                };
                assert!(
                    mag_rel < 1e-4,
                    "mag_bits rel diff {mag_rel} too large {w}x{h}"
                );
                let rel = if r.0.abs() > 1e-3 {
                    ((a.0 - r.0) / r.0).abs()
                } else {
                    (a.0 - r.0).abs()
                };
                assert!(rel < 1e-4, "sse rel diff {rel} too large {w}x{h}");
            }
        }
    }

    #[test]
    fn test_log2p1() {
        if !std::is_x86_feature_detected!("avx") && !std::is_x86_feature_detected!("fma") {
            return;
        }
        unsafe {
            for i in 0..5000 {
                let q1 = ((i as f32 / 5000.) + 1.).log2();
                let r = avx2_log2p1_f32(_mm256_set1_ps(i as f32 / 5000.));
                let q2 = f32::from_bits(_mm_extract_ps::<0>(_mm256_castps256_ps128(r)) as u32);
                assert!((q1 - q2).abs() < 1e-5, "q1 {} q2 {}", q1, q2);
            }
        }
    }
}
