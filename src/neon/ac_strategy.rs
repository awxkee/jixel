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

#[target_feature(enable = "neon")]
fn gradient_region_stats_impl<const CHROMA: bool>(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    if !w.is_multiple_of(4)
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
    let mut chroma = vdupq_n_f32(0.0);
    const ONE_OVER_16: f32 = 1.0 / 16.0;

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
        for (cx, mean) in mean_row.iter_mut().enumerate() {
            let mut sum = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            if CHROMA {
                for ((y_row, x_row), b_row) in y_rows4
                    .chunks_exact(xs)
                    .zip(x_rows4.chunks_exact(xs))
                    .zip(b_rows4.chunks_exact(xs))
                {
                    let y4 = &y_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let x4 = &x_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let b4 = &b_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let yv = unsafe { vld1q_f32(y4.as_ptr()) };
                    let xv = unsafe { vld1q_f32(x4.as_ptr()) };
                    let bv = unsafe { vld1q_f32(b4.as_ptr()) };
                    sum = vaddq_f32(sum, yv);
                    sum2 = vfmaq_f32(sum2, yv, yv);
                    chroma = vaddq_f32(chroma, vaddq_f32(vabsq_f32(xv), vabdq_f32(bv, yv)));
                }
            } else {
                for y_row in y_rows4.chunks_exact(xs) {
                    let y4 = &y_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let yv = unsafe { vld1q_f32(y4.as_ptr()) };
                    sum = vaddq_f32(sum, yv);
                    sum2 = vfmaq_f32(sum2, yv, yv);
                }
            }
            let m = vaddvq_f32(sum) * ONE_OVER_16;
            *mean = m;
            within += (vaddvq_f32(sum2) * ONE_OVER_16 - m * m).max(0.0);
        }
    }

    crate::ac_strategy::finish_gradient_region_stats(
        &means[..cw * ch],
        within,
        if CHROMA { vaddvq_f32(chroma) } else { 0.0 },
        w * h,
        eps,
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn gradient_region_stats_neon(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    gradient_region_stats_impl::<false>(opsin, px0, py0, w, h, eps)
}

#[target_feature(enable = "neon")]
pub(crate) fn gradient_region_stats_with_chroma_neon(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    gradient_region_stats_impl::<true>(opsin, px0, py0, w, h, eps)
}

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
    _rate_log2_lut: &crate::inflated_cost::RateLog2Lut,
    thr: &[f32; 4],
    scan_pos: &[u32],
) -> (f32, usize, f32, u32) {
    let n = width * height;
    assert!(coeff.len() >= n && inv_matrix.len() >= n && scan_pos.len() >= n);
    debug_assert!(width.is_multiple_of(4) && half.is_multiple_of(4));

    let qs = vdupq_n_f32(q_scaled);
    let mut sse_acc = vdupq_n_f32(0.0);
    let mut nz_acc = vdupq_n_u32(0);
    let mut mag_acc = vdupq_n_f32(0.0);
    let mut scan_acc = vdupq_n_u32(0);
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
            // Ties-away rounding, matching the real quantizer (vcvtaq/fast_round).
            let rounded = vrndaq_f32(a);

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

            // Keep the count lane-local until the end. A horizontal reduction
            // for every four coefficients was a sizable part of this loop.
            nz_acc = vaddq_u32(nz_acc, vshrq_n_u32::<31>(rate_mask));

            // Quantized AC coefficients are commonly zero. Do not run the
            // seven-FMA logarithm when none of this vector contributes rate.
            if vmaxvq_u32(rate_mask) != 0 {
                let ratev = neon_log2p1_f32(absq);
                mag_acc = vaddq_f32(mag_acc, vbslq_f32(rate_mask, ratev, zero));
                // Scan position of the nonzeros (LLF slots are never nonzero
                // here, so masked-to-zero lanes are neutral for the max).
                let sv = unsafe { vld1q_u32(scan_pos.as_ptr().add(y * width + x)) };
                scan_acc = vmaxq_u32(scan_acc, vandq_u32(sv, rate_mask));
            }
        }
    }
    (
        vaddvq_f32(sse_acc),
        vaddvq_u32(nz_acc) as usize,
        vaddvq_f32(mag_acc),
        vmaxvq_u32(scan_acc),
    )
}

/// NEON counterpart of `enc_group::dequantized_level_f32`: the decoder's biased
/// dequant of an integer-valued float level (0 -> 0, +-1 -> +-0.9299455,
/// q -> q - 0.145/q). Dormant until the biased-distortion selection/rerank
/// variants land with their margin re-fit (see `dequantized_level_f32`).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn neon_dequantized_level_f32(q: float32x4_t) -> float32x4_t {
    let absq = vabsq_f32(q);
    // q - 0.145/q (q == 0 lanes produce non-finite values, masked out below).
    let big = vsubq_f32(
        q,
        vdivq_f32(vdupq_n_f32(crate::group::DEFAULT_QUANT_BIAS_3), q),
    );
    let sign = vandq_u32(vreinterpretq_u32_f32(q), vdupq_n_u32(0x8000_0000));
    let one = vreinterpretq_f32_u32(vorrq_u32(
        sign,
        vreinterpretq_u32_f32(vdupq_n_f32(crate::group::DEFAULT_QUANT_BIAS_1)),
    ));
    let use_big = vcgeq_f32(absq, vdupq_n_f32(1.125));
    let dq = vbslq_f32(use_big, big, one);
    let nz = vcgtq_f32(absq, vdupq_n_f32(0.0));
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dq), nz))
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn neon_log2p1_f32(x: float32x4_t) -> float32x4_t {
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

    vfmaq_f32(e, t, p)
}

#[cfg(test)]
mod tests {
    use super::{neon_log2p1_f32, sse_and_rate_neon};
    use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

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
        let (mut sse, mut nzeros, mut mag_bits) = (0.0f32, 0usize, 0.0f32);
        let mut max_scan = 0u32;
        for y in 0..h {
            let yfix = if y >= h / 2 { 2 } else { 0 };
            for x in 0..w {
                if x < cx && y < cy {
                    continue;
                }
                let i = y * w + x;
                let threshold = if x >= half { thr[yfix + 1] } else { thr[yfix] };
                let a = inv[i] * qs * coeff[i];
                let q = if a.abs() >= threshold { a.round() } else { 0.0 };
                let d = a - q;
                sse += d * d;
                if q != 0.0 {
                    nzeros += 1;
                    mag_bits += (1.0 + q.abs()).log2();
                    max_scan = max_scan.max(scan_pos[i]);
                }
            }
        }
        (sse, nzeros, mag_bits, max_scan)
    }

    #[test]
    fn test_sse_and_rate_neon_vs_reference() {
        let mut state = 0x5e5e_a11d_0f00_d00du64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32 as f32 / u32::MAX as f32
        };

        for &(w, h, half, cx, cy) in &[
            (8usize, 8usize, 4usize, 1usize, 1usize),
            (16, 8, 8, 2, 1),
            (8, 16, 4, 1, 2),
            (16, 16, 8, 2, 2),
            (32, 32, 16, 4, 4),
        ] {
            for case in 0..100 {
                let n = w * h;
                // Alternate sparse and dense inputs so the all-zero-vector
                // fast path and the logarithm path are both exercised.
                let coeff_scale = if case % 2 == 0 { 0.2 } else { 200.0 };
                let coeff: Vec<f32> = (0..n).map(|_| (random() - 0.5) * coeff_scale).collect();
                let inv: Vec<f32> = (0..n).map(|_| 0.001 + random() * 0.5).collect();
                let qs = 0.5 + random() * 3.0;
                let thr = [
                    random() * 0.6,
                    random() * 0.6,
                    random() * 0.6,
                    random() * 0.6,
                ];
                let scan_pos = crate::coeff_order::scan_pos_lut(w, h);
                let expected = reference(&coeff, &inv, qs, w, h, half, cx, cy, &thr, scan_pos);
                let actual = unsafe {
                    sse_and_rate_neon(
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

                assert_eq!(actual.1, expected.1, "nzeros mismatch for {w}x{h}");
                assert_eq!(actual.3, expected.3, "max-scan mismatch for {w}x{h}");
                let sse_rel = (actual.0 - expected.0).abs() / expected.0.abs().max(1.0);
                let mag_rel = (actual.2 - expected.2).abs() / expected.2.abs().max(1.0);
                assert!(sse_rel < 1e-4, "SSE relative error {sse_rel} for {w}x{h}");
                assert!(
                    mag_rel < 1e-5,
                    "magnitude-rate relative error {mag_rel} for {w}x{h}"
                );
            }
        }
    }

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
