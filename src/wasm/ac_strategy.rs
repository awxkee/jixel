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

/// WASM SIMD128 implementation of `sse_and_rate_scalar`.
///
/// # Safety
/// Caller must ensure `wasm` is available (checked via [`supported`]); on
/// AArch64 WASM SIMD128 is part of the baseline ISA. All slice accesses are
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
    thr: &[f32; 4],
) -> (f32, usize, f32) {
    let n = width * height;
    assert!(coeff.len() >= n && inv_matrix.len() >= n);
    debug_assert!(width.is_multiple_of(4) && half.is_multiple_of(4));

    let qs = f32x4_splat(q_scaled);
    let mut sse_acc = f32x4_splat(0.0);
    let mut nzeros = 0usize;
    let mut mag_bits = 0.0f32;

    let mut qbuf = [0.0f32; 4];

    for y in 0..height {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];
        let row = y * width;
        let mut x = 0;
        while x < width {
            let threshold = if x >= half { thr_hi } else { thr_lo };
            let thrv = f32x4_splat(threshold);
            let idx = row + x;
            let cv =
                unsafe { v128_load(coeff[idx..idx + 4].as_ptr() as *const f32 as *const v128) };
            let mv = unsafe {
                v128_load(inv_matrix[idx..idx + 4].as_ptr() as *const f32 as *const v128)
            };
            // a = inv_matrix * q_scaled * coeff
            let a = f32x4_mul(f32x4_mul(mv, qs), cv);
            let absa = f32x4_abs(a);
            // keep = |a| >= threshold  (all-ones / all-zeros lane mask)
            let keep = f32x4_ge(absa, thrv);
            // round-to-nearest, ties-to-even (matches _mm_round_ps NEAREST_INT)
            let rounded = f32x4_nearest(a);
            // zero the rounded value where not kept
            let q = v128_and(rounded, keep);
            let d = f32x4_sub(a, q);
            let mut d2 = f32x4_mul(d, d);

            // Mask out LLF lanes (x'<cx && y<cy) so they contribute nothing.
            if y < cy && x == 0 {
                // lanes 0..cx are LLF → 0.0; all other lanes kept → 1.0
                let mut lanemask = [1.0f32; 4];
                for lane in 0..cx.min(4) {
                    lanemask[lane] = 0.0;
                }
                let lm = unsafe { v128_load(lanemask.as_ptr() as *const f32 as *const v128) };
                d2 = f32x4_mul(d2, lm); // zero LLF squared-errors
            }
            sse_acc = f32x4_add(sse_acc, d2);

            // Rate: count nonzeros and accumulate log2(1+|q|) from kept lanes.
            unsafe {
                v128_store(qbuf.as_mut_ptr() as *mut f32 as *mut v128, q);
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
    (
        {
            let __h = sse_acc;
            f32x4_extract_lane::<0>(__h)
                + f32x4_extract_lane::<1>(__h)
                + f32x4_extract_lane::<2>(__h)
                + f32x4_extract_lane::<3>(__h)
        },
        nzeros,
        mag_bits,
    )
}
