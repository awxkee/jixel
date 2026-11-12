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
#[cfg_attr(target_feature = "fma", target_feature(enable = "fma"))]
fn xyb_to_rgb_sse41(m: &XybMatrix, x: __m128, y: __m128, b: __m128) -> (__m128, __m128, __m128) {
    let cube = |value: __m128| {
        let value = _mm_sub_ps(value, _mm_set1_ps(NEG_BIAS_CBRT));
        // Keep the cube and bias subtraction separate, as in the scalar inverse.
        _mm_sub_ps(
            _mm_mul_ps(_mm_mul_ps(value, value), value),
            _mm_set1_ps(OPSIN_BIAS),
        )
    };
    let m0 = cube(_mm_add_ps(y, x));
    let m1 = cube(_mm_sub_ps(y, x));
    let m2 = cube(b);
    // Match dct::fmla, including builds that do not fuse the inverse matrix.
    let row = |offset: usize| {
        let value = _mm_mul_ps(m2, _mm_set1_ps(m.inv[offset + 2]));
        #[cfg(target_feature = "fma")]
        {
            let value = _mm_fmadd_ps(m1, _mm_set1_ps(m.inv[offset + 1]), value);
            _mm_fmadd_ps(m0, _mm_set1_ps(m.inv[offset]), value)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            _mm_add_ps(
                _mm_mul_ps(m0, _mm_set1_ps(m.inv[offset])),
                _mm_add_ps(_mm_mul_ps(m1, _mm_set1_ps(m.inv[offset + 1])), value),
            )
        }
    };
    (row(0), row(3), row(6))
}

/// # Safety
/// The CPU must support the enabled target features and all six slices must
/// have equal lengths. Only complete SIMD vectors are read and written.
#[target_feature(enable = "sse4.1")]
#[cfg_attr(target_feature = "fma", target_feature(enable = "fma"))]
pub(crate) unsafe fn to_linear_rows_sse41(
    m: &XybMatrix,
    input: [&[f32]; 3],
    output: [&mut [f32]; 3],
) -> usize {
    let [r, g, b] = output;
    let done = unsafe { to_linear_rows::<4>(m, input, [r, g, b]) };
    done + unsafe {
        to_linear_rows::<1>(
            m,
            input.map(|row| &row[done..]),
            [&mut r[done..], &mut g[done..], &mut b[done..]],
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg_attr(target_feature = "fma", target_feature(enable = "fma"))]
unsafe fn to_linear_rows<const N: usize>(
    m: &XybMatrix,
    [x, y, b]: [&[f32]; 3],
    [r, g, out_b]: [&mut [f32]; 3],
) -> usize {
    const LANES: usize = 4;
    let done = r.len() / (N * LANES) * (N * LANES);
    for start in (0..done).step_by(N * LANES) {
        for lane in 0..N {
            let i = start + lane * LANES;
            // The caller's equal-length contract and complete-vector loop
            // bound cover every unaligned load and store.
            unsafe {
                let xv = _mm_loadu_ps(x.as_ptr().add(i));
                let yv = _mm_loadu_ps(y.as_ptr().add(i));
                let bv = _mm_loadu_ps(b.as_ptr().add(i));
                let (rv, gv, bv) = xyb_to_rgb_sse41(m, xv, yv, bv);
                _mm_storeu_ps(r.as_mut_ptr().add(i), rv);
                _mm_storeu_ps(g.as_mut_ptr().add(i), gv);
                _mm_storeu_ps(out_b.as_mut_ptr().add(i), bv);
            }
        }
    }
    done
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn halley_cbrt_sse41<const SCALAR_ORDER: bool>(x: __m128, a: __m128) -> __m128 {
    let tx = _mm_mul_ps(_mm_mul_ps(x, x), x);
    let two = _mm_set1_ps(2.0);
    let num = _mm_add_ps(tx, _mm_mul_ps(a, two));
    let den = _mm_add_ps(a, _mm_mul_ps(tx, two));

    if SCALAR_ORDER {
        _mm_div_ps(_mm_mul_ps(x, num), den)
    } else {
        _mm_mul_ps(x, _mm_div_ps(num, den))
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn integer_pow_1_3_sse41<const SCALAR_ORDER: bool>(hx: __m128i) -> __m128i {
    let scale = _mm_set1_epi32(if SCALAR_ORDER {
        0xaaaa_aaabu32 as i32
    } else {
        341
    });
    let even64 = _mm_mul_epu32(hx, scale);
    let odd64 = _mm_mul_epu32(_mm_srli_epi64::<32>(hx), scale);
    let (even, odd) = if SCALAR_ORDER {
        (_mm_srli_epi64::<33>(even64), _mm_srli_epi64::<33>(odd64))
    } else {
        (_mm_srli_epi64::<10>(even64), _mm_srli_epi64::<10>(odd64))
    };
    let odd = _mm_slli_epi64::<32>(odd);

    // even: [r0, 0,  r2, 0]
    // odd:  [0,  r1, 0,  r3]
    // out:  [r0, r1, r2, r3]
    _mm_or_si128(even, odd)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn cbrt_seed_positive_sse41<const SCALAR_ORDER: bool>(x: __m128) -> __m128 {
    let ui = _mm_castps_si128(x);
    let hx = _mm_and_si128(ui, _mm_set1_epi32(0x7fff_ffff));

    let hx = _mm_add_epi32(
        integer_pow_1_3_sse41::<SCALAR_ORDER>(hx),
        _mm_set1_epi32(709958130),
    );

    _mm_castsi128_ps(hx)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn vcbrt_fast3_positive_sse41(
    a0: __m128,
    a1: __m128,
    a2: __m128,
) -> (__m128, __m128, __m128) {
    cbrt3_positive_sse41::<false>(a0, a1, a2)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn cbrt3_positive_sse41<const SCALAR_ORDER: bool>(
    a0: __m128,
    a1: __m128,
    a2: __m128,
) -> (__m128, __m128, __m128) {
    let mut x0 = cbrt_seed_positive_sse41::<SCALAR_ORDER>(a0);
    let mut x1 = cbrt_seed_positive_sse41::<SCALAR_ORDER>(a1);
    let mut x2 = cbrt_seed_positive_sse41::<SCALAR_ORDER>(a2);

    x0 = halley_cbrt_sse41::<SCALAR_ORDER>(x0, a0);
    x1 = halley_cbrt_sse41::<SCALAR_ORDER>(x1, a1);
    x2 = halley_cbrt_sse41::<SCALAR_ORDER>(x2, a2);

    x0 = halley_cbrt_sse41::<SCALAR_ORDER>(x0, a0);
    x1 = halley_cbrt_sse41::<SCALAR_ORDER>(x1, a1);
    x2 = halley_cbrt_sse41::<SCALAR_ORDER>(x2, a2);

    let zero = _mm_setzero_ps();
    x0 = _mm_blendv_ps(x0, zero, _mm_cmpeq_ps(a0, zero));
    x1 = _mm_blendv_ps(x1, zero, _mm_cmpeq_ps(a1, zero));
    x2 = _mm_blendv_ps(x2, zero, _mm_cmpeq_ps(a2, zero));

    (x0, x1, x2)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn rgb_to_xyb_f32x4_sse41<const SCALAR_ORDER: bool>(
    m: &XybMatrix,
    r: __m128,
    g: __m128,
    b: __m128,
) -> (__m128, __m128, __m128) {
    let bias = _mm_set1_ps(OPSIN_BIAS);

    // mixed rows: m.fwd[r0..r2] per channel + OPSIN_BIAS
    let mut mixed0 = _mm_add_ps(bias, _mm_mul_ps(b, _mm_set1_ps(m.fwd[2])));
    mixed0 = _mm_add_ps(mixed0, _mm_mul_ps(g, _mm_set1_ps(m.fwd[1])));
    mixed0 = _mm_add_ps(mixed0, _mm_mul_ps(r, _mm_set1_ps(m.fwd[0])));

    // mixed rows: m.fwd[r0..r2] per channel + OPSIN_BIAS
    let mut mixed1 = _mm_add_ps(bias, _mm_mul_ps(b, _mm_set1_ps(m.fwd[5])));
    mixed1 = _mm_add_ps(mixed1, _mm_mul_ps(g, _mm_set1_ps(m.fwd[4])));
    mixed1 = _mm_add_ps(mixed1, _mm_mul_ps(r, _mm_set1_ps(m.fwd[3])));

    // mixed2 = m20*r + m21*g + m22*b + OPSIN_BIAS (B row selected by RED)
    let mut mixed2 = _mm_add_ps(bias, _mm_mul_ps(b, _mm_set1_ps(m.fwd[8])));
    mixed2 = _mm_add_ps(mixed2, _mm_mul_ps(g, _mm_set1_ps(m.fwd[7])));
    mixed2 = _mm_add_ps(mixed2, _mm_mul_ps(r, _mm_set1_ps(m.fwd[6])));

    let zero = _mm_setzero_ps();

    mixed0 = _mm_max_ps(mixed0, zero);
    mixed1 = _mm_max_ps(mixed1, zero);
    mixed2 = _mm_max_ps(mixed2, zero);

    let (tm0, tm1, tm2) = cbrt3_positive_sse41::<SCALAR_ORDER>(mixed0, mixed1, mixed2);

    let neg_bias = _mm_set1_ps(NEG_BIAS_CBRT);

    let tm0 = _mm_add_ps(tm0, neg_bias);
    let tm1 = _mm_add_ps(tm1, neg_bias);
    let tm2 = _mm_add_ps(tm2, neg_bias);

    let half = _mm_set1_ps(0.5);

    let x = _mm_mul_ps(_mm_sub_ps(tm0, tm1), half);
    let y = _mm_mul_ps(_mm_add_ps(tm0, tm1), half);

    (x, y, tm2)
}

/// Convert the complete SIMD vectors of the modular XYB lattice.
#[cfg(not(target_feature = "fma"))]
#[target_feature(enable = "sse4.1")]
pub(crate) fn xyb_lattice_sse41(
    m: &XybMatrix,
    [r, g, b]: [&[f32]; 3],
    [luma, co, cg]: [&mut [i32]; 3],
) -> usize {
    use super::quantize_xyb::round_ties_away_i32;
    use crate::lossless::rct_simd::sse::forward_vec;
    use crate::quant_weights::INV_DC_QUANT;

    let [sx, sy, sb] = [
        _mm_set1_ps(INV_DC_QUANT[0]),
        _mm_set1_ps(INV_DC_QUANT[1]),
        _mm_set1_ps(INV_DC_QUANT[2]),
    ];
    let r = r.as_chunks::<4>().0;
    let g = g.as_chunks::<4>().0;
    let b = b.as_chunks::<4>().0;
    let luma = luma.as_chunks_mut::<4>().0;
    let co = co.as_chunks_mut::<4>().0;
    let cg = cg.as_chunks_mut::<4>().0;
    for (((r, g), b), ((luma, co_out), cg_out)) in
        r.iter().zip(g).zip(b).zip(luma.iter_mut().zip(co).zip(cg))
    {
        let r = unsafe { _mm_loadu_ps(r.as_ptr()) };
        let g = unsafe { _mm_loadu_ps(g.as_ptr()) };
        let b = unsafe { _mm_loadu_ps(b.as_ptr()) };
        let (x, y, b) = rgb_to_xyb_f32x4_sse41::<true>(m, r, g, b);
        let y = round_ties_away_i32(_mm_mul_ps(y, sy));
        let x = round_ties_away_i32(_mm_mul_ps(x, sx));
        let b_minus_y = _mm_sub_epi32(round_ties_away_i32(_mm_mul_ps(b, sb)), y);
        let [luma_value, co, cg] = forward_vec(y, x, b_minus_y);
        unsafe {
            _mm_storeu_si128(luma.as_mut_ptr().cast(), luma_value);
            _mm_storeu_si128(co_out.as_mut_ptr().cast(), co);
            _mm_storeu_si128(cg_out.as_mut_ptr().cast(), cg);
        }
    }
    r.len() * 4
}

/// Transform one row-band into separate output planes.
#[target_feature(enable = "sse4.1")]
pub(crate) fn to_xyb_sse41_band(
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

            let (xv, yv, bv) = rgb_to_xyb_f32x4_sse41::<false>(m, r, g, b);

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

            let (xv, yv, bv) = rgb_to_xyb_f32x4_sse41::<false>(m, r, g, b);

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
