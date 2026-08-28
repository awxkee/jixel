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
use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn hsum4(v: v128) -> f32 {
    // WASM SIMD128 has no native horizontal f32 add.
    // This is fine here because reduction happens once at the end.
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

#[target_feature(enable = "simd128")]
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
    let mut chroma = f32x4_splat(0.0);
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
            let mut sum = f32x4_splat(0.0);
            let mut sum2 = f32x4_splat(0.0);
            if CHROMA {
                for ((y_row, x_row), b_row) in y_rows4
                    .chunks_exact(xs)
                    .zip(x_rows4.chunks_exact(xs))
                    .zip(b_rows4.chunks_exact(xs))
                {
                    let y4 = &y_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let x4 = &x_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let b4 = &b_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let yv = unsafe { v128_load(y4.as_ptr().cast()) };
                    let xv = unsafe { v128_load(x4.as_ptr().cast()) };
                    let bv = unsafe { v128_load(b4.as_ptr().cast()) };
                    sum = f32x4_add(sum, yv);
                    sum2 = f32x4_add(sum2, f32x4_mul(yv, yv));
                    chroma = f32x4_add(
                        chroma,
                        f32x4_add(f32x4_abs(xv), f32x4_abs(f32x4_sub(bv, yv))),
                    );
                }
            } else {
                for y_row in y_rows4.chunks_exact(xs) {
                    let y4 = &y_row[px0..px0 + w].as_chunks::<4>().0[cx];
                    let yv = unsafe { v128_load(y4.as_ptr().cast()) };
                    sum = f32x4_add(sum, yv);
                    sum2 = f32x4_add(sum2, f32x4_mul(yv, yv));
                }
            }
            let m = hsum4(sum) * ONE_OVER_16;
            *mean = m;
            within += (hsum4(sum2) * ONE_OVER_16 - m * m).max(0.0);
        }
    }

    crate::ac_strategy::finish_gradient_region_stats(
        &means[..cw * ch],
        within,
        if CHROMA { hsum4(chroma) } else { 0.0 },
        w * h,
        eps,
    )
}

#[target_feature(enable = "simd128")]
pub(crate) fn gradient_region_stats_wasm(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    gradient_region_stats_impl::<false>(opsin, px0, py0, w, h, eps)
}

#[target_feature(enable = "simd128")]
pub(crate) fn gradient_region_stats_with_chroma_wasm(
    opsin: &crate::image::Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> crate::ac_strategy::GradientRegionStats {
    gradient_region_stats_impl::<true>(opsin, px0, py0, w, h, eps)
}

/// WASM SIMD128 implementation of `sse_and_rate_scalar`.
///
/// # Safety
/// Caller must ensure `simd128` is available. All slice accesses are
/// bounds-validated against `width*height`.
#[inline]
#[target_feature(enable = "simd128")]
fn wasm_dequantized_level_f32(q: v128) -> v128 {
    let absq = f32x4_abs(q);
    // q - 0.145/q (q == 0 lanes produce non-finite values, masked out below).
    let big = f32x4_sub(
        q,
        f32x4_div(f32x4_splat(crate::group::DEFAULT_QUANT_BIAS_3), q),
    );
    let sign = v128_and(q, f32x4_splat(-0.0));
    let one = v128_or(sign, f32x4_splat(crate::group::DEFAULT_QUANT_BIAS_1));
    let use_big = f32x4_ge(absq, f32x4_splat(1.125));
    let dq = v128_bitselect(big, one, use_big);
    let nz = f32x4_gt(absq, f32x4_splat(0.0));
    v128_and(dq, nz)
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "simd128")]
pub(crate) fn sse_and_rate_wasm<const BIASED: bool>(
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

    let qs = f32x4_splat(q_scaled);

    let mut sse_acc = f32x4_splat(0.0);
    let mut mag_acc = f32x4_splat(0.0);
    let mut nzeros = 0usize;
    let mut scan_acc = u32x4_splat(0);

    let zero = f32x4_splat(0.0);

    let lane_ids_arr = [0i32, 1, 2, 3];
    let lane_ids = unsafe { v128_load(lane_ids_arr.as_ptr() as *const v128) };

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
            let thrv = f32x4_splat(threshold);

            let cv = unsafe { v128_load(coeff4.as_ptr() as *const v128) };
            let mv = unsafe { v128_load(inv4.as_ptr() as *const v128) };

            // a = inv_matrix * q_scaled * coeff
            let a = f32x4_mul(f32x4_mul(mv, qs), cv);
            let absa = f32x4_abs(a);

            // keep = |a| >= threshold
            let keep = f32x4_ge(absa, thrv);

            // round-to-nearest, ties-to-even
            let rounded = f32x4_nearest(a);

            // zero rounded where not kept
            let q = v128_and(rounded, keep);

            let d = if BIASED {
                f32x4_sub(a, wasm_dequantized_level_f32(q))
            } else {
                f32x4_sub(a, q)
            };
            let d2 = f32x4_mul(d, d);

            // Active mask: keep lanes where !(y < cy && x + lane < cx).
            let active = if y < cy && x < cx {
                let lane_x = i32x4_add(i32x4_splat(x as i32), lane_ids);

                // active = x + lane >= cx
                i32x4_ge(lane_x, i32x4_splat(cx as i32))
            } else {
                i32x4_splat(-1)
            };

            // SSE: zero LLF lanes.
            let d2 = v128_bitselect(d2, zero, active);
            sse_acc = f32x4_add(sse_acc, d2);

            // Rate: active && abs(q) > 0.
            let absq = f32x4_abs(q);
            let nz = f32x4_gt(absq, zero);
            let rate_mask = v128_and(nz, active);

            let rate_bits = i32x4_bitmask(rate_mask);
            nzeros += rate_bits.count_ones() as usize;

            if rate_bits != 0 {
                let ratev = wasm_log2p1_f32(absq);
                mag_acc = f32x4_add(mag_acc, v128_bitselect(ratev, zero, rate_mask));
                // Scan position of the nonzeros (masked lanes drop to zero,
                // which is neutral: LLF slots are never nonzero here).
                let sv = unsafe { v128_load(scan_pos.as_ptr().add(y * width + x) as *const v128) };
                scan_acc = u32x4_max(scan_acc, v128_and(sv, rate_mask));
            }
        }
    }

    let max_scan = u32x4_extract_lane::<0>(scan_acc)
        .max(u32x4_extract_lane::<1>(scan_acc))
        .max(u32x4_extract_lane::<2>(scan_acc))
        .max(u32x4_extract_lane::<3>(scan_acc));

    (hsum4(sse_acc), nzeros, hsum4(mag_acc), max_scan)
}

#[inline]
#[target_feature(enable = "simd128")]
fn wasm_log2p1_f32(x: v128) -> v128 {
    // y = 1 + x
    let y = f32x4_add(x, f32x4_splat(1.0));

    // Decompose y = m * 2^e, m in [1, 2).
    let bits = y;

    let exp_u = u32x4_shr(bits, 23);
    let exp_i = i32x4_sub(exp_u, i32x4_splat(127));
    let e = f32x4_convert_i32x4(exp_i);

    let mant_bits = v128_or(
        v128_and(bits, i32x4_splat(0x007f_ffff)),
        i32x4_splat(0x3f80_0000),
    );

    let m = mant_bits;

    // t in [0, 1)
    let t = f32x4_sub(m, f32x4_splat(1.0));

    // Generated by stats/log2p1.sollya.
    let c0 = f32x4_splat(1.4426934719085693359375);
    let c1 = f32x4_splat(-0.721179187297821044921875);
    let c2 = f32x4_splat(0.477900087833404541015625);
    let c3 = f32x4_splat(-0.340080082416534423828125);
    let c4 = f32x4_splat(0.21719777584075927734375);
    let c5 = f32x4_splat(-9.749893844127655029296875e-2);
    let c6 = f32x4_splat(2.096841670572757720947265625e-2);

    let mut p = c6;
    p = f32x4_add(c5, f32x4_mul(t, p));
    p = f32x4_add(c4, f32x4_mul(t, p));
    p = f32x4_add(c3, f32x4_mul(t, p));
    p = f32x4_add(c2, f32x4_mul(t, p));
    p = f32x4_add(c1, f32x4_mul(t, p));
    p = f32x4_add(c0, f32x4_mul(t, p));

    let log2_m = f32x4_mul(t, p);

    f32x4_add(e, log2_m)
}

#[cfg(test)]
mod tests {
    use super::sse_and_rate_wasm;

    #[test]
    fn test_sse_and_rate_wasm_vs_reference() {
        crate::inflated_cost::assert_sse_and_rate_matches_reference(
            sse_and_rate_wasm::<false>,
            false,
        );
        crate::inflated_cost::assert_sse_and_rate_matches_reference(
            sse_and_rate_wasm::<true>,
            true,
        );
    }
}
