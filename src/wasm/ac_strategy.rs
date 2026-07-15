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

/// WASM SIMD128 implementation of `sse_and_rate_scalar`.
///
/// # Safety
/// Caller must ensure `simd128` is available. All slice accesses are
/// bounds-validated against `width*height`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "simd128")]
pub(crate) fn sse_and_rate_wasm(
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

    let qs = f32x4_splat(q_scaled);

    let mut sse_acc = f32x4_splat(0.0);
    let mut mag_acc = f32x4_splat(0.0);
    let mut nzeros = 0usize;

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

            let d = f32x4_sub(a, q);
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
            }
        }
    }

    (hsum4(sse_acc), nzeros, hsum4(mag_acc))
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
        crate::inflated_cost::assert_sse_and_rate_matches_reference(sse_and_rate_wasm);
    }
}
