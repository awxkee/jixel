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

use crate::xyb::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn halley_cbrt_sse41(x: __m128, a: __m128) -> __m128 {
    let tx = _mm_mul_ps(_mm_mul_ps(x, x), x);
    let two = _mm_set1_ps(2.0);
    let num = _mm_add_ps(tx, _mm_mul_ps(a, two));
    let den = _mm_add_ps(a, _mm_mul_ps(tx, two));

    _mm_mul_ps(x, _mm_div_ps(num, den))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn integer_pow_1_3_sse41(hx: __m128i) -> __m128i {
    let scale = _mm_set1_epi32(341);
    let even64 = _mm_mul_epu32(hx, scale);
    let even = _mm_srli_epi64::<10>(even64);

    let odd_src = _mm_srli_epi64::<32>(hx);
    let odd64 = _mm_mul_epu32(odd_src, scale);
    let odd = _mm_slli_epi64::<32>(_mm_srli_epi64::<10>(odd64));

    // even: [r0, 0,  r2, 0]
    // odd:  [0,  r1, 0,  r3]
    // out:  [r0, r1, r2, r3]
    _mm_or_si128(even, odd)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn cbrt_seed_positive_sse41(x: __m128) -> __m128 {
    let ui = _mm_castps_si128(x);
    let hx = _mm_and_si128(ui, _mm_set1_epi32(0x7fff_ffff));

    let hx = _mm_add_epi32(integer_pow_1_3_sse41(hx), _mm_set1_epi32(709958130));

    _mm_castsi128_ps(hx)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn vcbrt_fast3_positive_sse41(a0: __m128, a1: __m128, a2: __m128) -> (__m128, __m128, __m128) {
    let mut x0 = cbrt_seed_positive_sse41(a0);
    let mut x1 = cbrt_seed_positive_sse41(a1);
    let mut x2 = cbrt_seed_positive_sse41(a2);

    x0 = halley_cbrt_sse41(x0, a0);
    x1 = halley_cbrt_sse41(x1, a1);
    x2 = halley_cbrt_sse41(x2, a2);

    x0 = halley_cbrt_sse41(x0, a0);
    x1 = halley_cbrt_sse41(x1, a1);
    x2 = halley_cbrt_sse41(x2, a2);

    (x0, x1, x2)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn rgb_to_xyb_f32x4_sse41(r: __m128, g: __m128, b: __m128) -> (__m128, __m128, __m128) {
    let bias = _mm_set1_ps(OPSIN_BIAS);

    // mixed0 = M00*r + M01*g + M02*b + OPSIN_BIAS
    let mut mixed0 = _mm_add_ps(bias, _mm_mul_ps(b, _mm_set1_ps(M02)));
    mixed0 = _mm_add_ps(mixed0, _mm_mul_ps(g, _mm_set1_ps(M01)));
    mixed0 = _mm_add_ps(mixed0, _mm_mul_ps(r, _mm_set1_ps(M00)));

    // mixed1 = M10*r + M11*g + M12*b + OPSIN_BIAS
    let mut mixed1 = _mm_add_ps(bias, _mm_mul_ps(b, _mm_set1_ps(M12)));
    mixed1 = _mm_add_ps(mixed1, _mm_mul_ps(g, _mm_set1_ps(M11)));
    mixed1 = _mm_add_ps(mixed1, _mm_mul_ps(r, _mm_set1_ps(M10)));

    // mixed2 = M20*r + M21*g + M22*b + OPSIN_BIAS
    let mut mixed2 = _mm_add_ps(bias, _mm_mul_ps(b, _mm_set1_ps(M22)));
    mixed2 = _mm_add_ps(mixed2, _mm_mul_ps(g, _mm_set1_ps(M21)));
    mixed2 = _mm_add_ps(mixed2, _mm_mul_ps(r, _mm_set1_ps(M20)));

    let zero = _mm_setzero_ps();

    mixed0 = _mm_max_ps(mixed0, zero);
    mixed1 = _mm_max_ps(mixed1, zero);
    mixed2 = _mm_max_ps(mixed2, zero);

    let (tm0, tm1, tm2) = vcbrt_fast3_positive_sse41(mixed0, mixed1, mixed2);

    let neg_bias = _mm_set1_ps(NEG_BIAS_CBRT);

    let tm0 = _mm_add_ps(tm0, neg_bias);
    let tm1 = _mm_add_ps(tm1, neg_bias);
    let tm2 = _mm_add_ps(tm2, neg_bias);

    let half = _mm_set1_ps(0.5);

    let x = _mm_mul_ps(_mm_sub_ps(tm0, tm1), half);
    let y = _mm_mul_ps(_mm_add_ps(tm0, tm1), half);

    (x, y, tm2)
}

/// Transform one row-band into separate output planes.
#[target_feature(enable = "sse4.1")]
pub(crate) fn to_xyb_sse41_band(input: [&[f32]; 3], output: [&mut [f32]; 3], w: usize) {
    let [rp, gp, bp] = input;
    let [xp, yp, out_bp] = output;
    for (((((r_row, g_row), b_row), x_row), y_row), out_b_row) in rp
        .chunks_exact(w)
        .zip(gp.chunks_exact(w))
        .zip(bp.chunks_exact(w))
        .zip(xp.chunks_exact_mut(w))
        .zip(yp.chunks_exact_mut(w))
        .zip(out_bp.chunks_exact_mut(w))
    {
        let (r_chunks, r_tail) = r_row.as_chunks::<4>();
        let (g_chunks, g_tail) = g_row.as_chunks::<4>();
        let (b_chunks, b_tail) = b_row.as_chunks::<4>();
        let (x_chunks, x_tail) = x_row.as_chunks_mut::<4>();
        let (y_chunks, y_tail) = y_row.as_chunks_mut::<4>();
        let (out_b_chunks, out_b_tail) = out_b_row.as_chunks_mut::<4>();

        for (((((r4, g4), b4), x4), y4), out_b4) in r_chunks
            .iter()
            .zip(g_chunks.iter())
            .zip(b_chunks.iter())
            .zip(x_chunks.iter_mut())
            .zip(y_chunks.iter_mut())
            .zip(out_b_chunks.iter_mut())
        {
            let r = unsafe { _mm_loadu_ps(r4.as_ptr()) };
            let g = unsafe { _mm_loadu_ps(g4.as_ptr()) };
            let b = unsafe { _mm_loadu_ps(b4.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_sse41(r, g, b);

            unsafe {
                _mm_storeu_ps(x4.as_mut_ptr(), xv);
                _mm_storeu_ps(y4.as_mut_ptr(), yv);
                _mm_storeu_ps(out_b4.as_mut_ptr(), bv);
            }
        }

        if !r_tail.is_empty() {
            let mut r4 = [0.0f32; 4];
            let mut g4 = [0.0f32; 4];
            let mut b4 = [0.0f32; 4];

            r4[..r_tail.len()].copy_from_slice(r_tail);
            g4[..g_tail.len()].copy_from_slice(g_tail);
            b4[..b_tail.len()].copy_from_slice(b_tail);

            let r = unsafe { _mm_loadu_ps(r4.as_ptr()) };
            let g = unsafe { _mm_loadu_ps(g4.as_ptr()) };
            let b = unsafe { _mm_loadu_ps(b4.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_sse41(r, g, b);

            unsafe {
                _mm_storeu_ps(r4.as_mut_ptr(), xv);
                _mm_storeu_ps(g4.as_mut_ptr(), yv);
                _mm_storeu_ps(b4.as_mut_ptr(), bv);
            }

            x_tail.copy_from_slice(&r4[..r_tail.len()]);
            y_tail.copy_from_slice(&g4[..g_tail.len()]);
            out_b_tail.copy_from_slice(&b4[..b_tail.len()]);
        }
    }
}
