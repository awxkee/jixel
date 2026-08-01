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
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn halley_cbrt_avx2(x: __m256, a: __m256) -> __m256 {
    let tx = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    let two = _mm256_set1_ps(2.0);
    let num = _mm256_fmadd_ps(a, two, tx);
    let den = _mm256_fmadd_ps(tx, two, a);

    _mm256_mul_ps(x, _mm256_div_ps(num, den))
}

#[inline]
#[target_feature(enable = "avx2")]
fn integer_pow_1_3_avx2(hx: __m256i) -> __m256i {
    let scale = _mm256_set1_epi32(341);
    let even64 = _mm256_mul_epu32(hx, scale);
    let even = _mm256_srli_epi64::<10>(even64);
    let odd_src = _mm256_srli_epi64::<32>(hx);
    let odd64 = _mm256_mul_epu32(odd_src, scale);
    let odd = _mm256_slli_epi64::<32>(_mm256_srli_epi64::<10>(odd64));
    _mm256_or_si256(even, odd)
}

#[inline]
#[target_feature(enable = "avx2")]
fn cbrt_seed_positive_avx2(x: __m256) -> __m256 {
    let ui = _mm256_castps_si256(x);
    let hx = _mm256_and_si256(ui, _mm256_set1_epi32(0x7fff_ffff));

    let hx = _mm256_add_epi32(integer_pow_1_3_avx2(hx), _mm256_set1_epi32(709958130));

    _mm256_castsi256_ps(hx)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn vcbrt_fast3_positive_avx2(a0: __m256, a1: __m256, a2: __m256) -> (__m256, __m256, __m256) {
    let mut x0 = cbrt_seed_positive_avx2(a0);
    let mut x1 = cbrt_seed_positive_avx2(a1);
    let mut x2 = cbrt_seed_positive_avx2(a2);

    x0 = halley_cbrt_avx2(x0, a0);
    x1 = halley_cbrt_avx2(x1, a1);
    x2 = halley_cbrt_avx2(x2, a2);

    x0 = halley_cbrt_avx2(x0, a0);
    x1 = halley_cbrt_avx2(x1, a1);
    x2 = halley_cbrt_avx2(x2, a2);

    (x0, x1, x2)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn rgb_to_xyb_f32x8_avx2(
    m: &XybMatrix,
    r: __m256,
    g: __m256,
    b: __m256,
) -> (__m256, __m256, __m256) {
    let bias = _mm256_set1_ps(OPSIN_BIAS);

    // mixed rows: m.fwd[r0..r2] per channel + OPSIN_BIAS
    let mut mixed0 = _mm256_fmadd_ps(b, _mm256_set1_ps(m.fwd[2]), bias);
    mixed0 = _mm256_fmadd_ps(g, _mm256_set1_ps(m.fwd[1]), mixed0);
    mixed0 = _mm256_fmadd_ps(r, _mm256_set1_ps(m.fwd[0]), mixed0);

    // mixed rows: m.fwd[r0..r2] per channel + OPSIN_BIAS
    let mut mixed1 = _mm256_fmadd_ps(b, _mm256_set1_ps(m.fwd[5]), bias);
    mixed1 = _mm256_fmadd_ps(g, _mm256_set1_ps(m.fwd[4]), mixed1);
    mixed1 = _mm256_fmadd_ps(r, _mm256_set1_ps(m.fwd[3]), mixed1);

    // mixed2 = m20*r + m21*g + m22*b + OPSIN_BIAS (B row selected by RED)
    let mut mixed2 = _mm256_fmadd_ps(b, _mm256_set1_ps(m.fwd[8]), bias);
    mixed2 = _mm256_fmadd_ps(g, _mm256_set1_ps(m.fwd[7]), mixed2);
    mixed2 = _mm256_fmadd_ps(r, _mm256_set1_ps(m.fwd[6]), mixed2);

    let zero = _mm256_setzero_ps();
    mixed0 = _mm256_max_ps(mixed0, zero);
    mixed1 = _mm256_max_ps(mixed1, zero);
    mixed2 = _mm256_max_ps(mixed2, zero);

    let (tm0, tm1, tm2) = vcbrt_fast3_positive_avx2(mixed0, mixed1, mixed2);

    let neg_bias = _mm256_set1_ps(NEG_BIAS_CBRT);
    let tm0 = _mm256_add_ps(tm0, neg_bias);
    let tm1 = _mm256_add_ps(tm1, neg_bias);
    let tm2 = _mm256_add_ps(tm2, neg_bias);

    let half = _mm256_set1_ps(0.5);

    let x = _mm256_mul_ps(_mm256_sub_ps(tm0, tm1), half);
    let y = _mm256_mul_ps(_mm256_add_ps(tm0, tm1), half);

    (x, y, tm2)
}

/// Transform one row-band into separate output planes.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn to_xyb_avx2_band(
    m: &XybMatrix,
    input: [&[f32]; 3],
    output: [&mut [f32]; 3],
    w: usize,
) {
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
        let (r_chunks, r_tail) = r_row.as_chunks::<8>();
        let (g_chunks, g_tail) = g_row.as_chunks::<8>();
        let (b_chunks, b_tail) = b_row.as_chunks::<8>();
        let (x_chunks, x_tail) = x_row.as_chunks_mut::<8>();
        let (y_chunks, y_tail) = y_row.as_chunks_mut::<8>();
        let (out_b_chunks, out_b_tail) = out_b_row.as_chunks_mut::<8>();

        for (((((r8, g8), b8), x8), y8), out_b8) in r_chunks
            .iter()
            .zip(g_chunks.iter())
            .zip(b_chunks.iter())
            .zip(x_chunks.iter_mut())
            .zip(y_chunks.iter_mut())
            .zip(out_b_chunks.iter_mut())
        {
            let r = unsafe { _mm256_loadu_ps(r8.as_ptr()) };
            let g = unsafe { _mm256_loadu_ps(g8.as_ptr()) };
            let b = unsafe { _mm256_loadu_ps(b8.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x8_avx2(m, r, g, b);

            unsafe {
                _mm256_storeu_ps(x8.as_mut_ptr(), xv);
                _mm256_storeu_ps(y8.as_mut_ptr(), yv);
                _mm256_storeu_ps(out_b8.as_mut_ptr(), bv);
            }
        }

        if !r_tail.is_empty() {
            let mut r8: [f32; 8] = [0.; 8];
            let mut g8: [f32; 8] = [0.; 8];
            let mut b8: [f32; 8] = [0.; 8];
            r8[..r_tail.len()].copy_from_slice(r_tail);
            g8[..g_tail.len()].copy_from_slice(g_tail);
            b8[..b_tail.len()].copy_from_slice(b_tail);
            let r = unsafe { _mm256_loadu_ps(r8.as_ptr()) };
            let g = unsafe { _mm256_loadu_ps(g8.as_ptr()) };
            let b = unsafe { _mm256_loadu_ps(b8.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x8_avx2(m, r, g, b);

            unsafe {
                _mm256_storeu_ps(r8.as_mut_ptr(), xv);
                _mm256_storeu_ps(g8.as_mut_ptr(), yv);
                _mm256_storeu_ps(b8.as_mut_ptr(), bv);
            }

            x_tail.copy_from_slice(&r8[..r_tail.len()]);
            y_tail.copy_from_slice(&g8[..g_tail.len()]);
            out_b_tail.copy_from_slice(&b8[..b_tail.len()]);
        }
    }
}
