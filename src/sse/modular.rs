/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

use crate::entropy::Token;
use crate::modular::{alpha_token_u8, alpha_token_u16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
fn store_four_tokens(contexts: __m128i, values: __m128i, out: &mut [Token]) {
    debug_assert_eq!(out.len(), 4);
    let out = out.as_mut_ptr().cast::<u32>();
    unsafe {
        _mm_storeu_si128(out.cast::<__m128i>(), _mm_unpacklo_epi32(contexts, values));
        _mm_storeu_si128(
            out.add(4).cast::<__m128i>(),
            _mm_unpackhi_epi32(contexts, values),
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tokenize_eight_u8(
    values: __m128i,
    west: __m128i,
    north: __m128i,
    northwest: __m128i,
    out: &mut [Token],
) {
    debug_assert_eq!(out.len(), 8);
    let v = _mm_cvtepu8_epi16(values);
    let w = _mm_cvtepu8_epi16(west);
    let n = _mm_cvtepu8_epi16(north);
    let nw = _mm_cvtepu8_epi16(northwest);
    let grad = _mm_sub_epi16(_mm_add_epi16(w, n), nw);
    let pred = _mm_max_epi16(
        _mm_min_epi16(w, n),
        _mm_min_epi16(_mm_max_epi16(w, n), grad),
    );
    let residual = _mm_sub_epi16(v, pred);
    let packed = _mm_xor_si128(
        _mm_slli_epi16::<1>(residual),
        _mm_srai_epi16::<15>(residual),
    );
    let contexts = _mm_andnot_si128(
        _mm_cmpgt_epi16(grad, _mm_setzero_si128()),
        _mm_set1_epi16(1),
    );
    store_four_tokens(
        _mm_cvtepu16_epi32(contexts),
        _mm_cvtepu16_epi32(packed),
        &mut out[..4],
    );
    store_four_tokens(
        _mm_cvtepu16_epi32(_mm_srli_si128::<8>(contexts)),
        _mm_cvtepu16_epi32(_mm_srli_si128::<8>(packed)),
        &mut out[4..],
    );
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn tokenize_alpha_u8_first_row_sse41(row: &[u8], out: &mut [Token]) {
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<16>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<16>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<16>();
    let zero = _mm_setzero_si128();

    for ((values, west), out) in values.iter().zip(west.iter()).zip(out_chunks.iter_mut()) {
        let values = unsafe { _mm_loadu_si128(values.as_ptr().cast()) };
        let west = unsafe { _mm_loadu_si128(west.as_ptr().cast()) };
        tokenize_eight_u8(values, west, zero, zero, &mut out[..8]);
        tokenize_eight_u8(
            _mm_srli_si128::<8>(values),
            _mm_srli_si128::<8>(west),
            zero,
            zero,
            &mut out[8..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<8>();
    let (west_halves, west_tail) = west_tail.as_chunks::<8>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<8>();
    for ((values, west), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_eight_u8(
            unsafe { _mm_loadl_epi64(values.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(west.as_ptr().cast()) },
            zero,
            zero,
            out,
        );
    }

    for ((&value, &west), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u8(value, west, 0, 0);
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn tokenize_alpha_u8_interior_sse41(row: &[u8], north: &[u8], out: &mut [Token]) {
    assert_eq!(row.len(), north.len());
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<16>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<16>();
    let (north_chunks, north_tail) = north[1..].as_chunks::<16>();
    let (northwest, northwest_tail) = north[..north.len() - 1].as_chunks::<16>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<16>();

    for ((((values, west), north), northwest), out) in values
        .iter()
        .zip(west.iter())
        .zip(north_chunks.iter())
        .zip(northwest.iter())
        .zip(out_chunks.iter_mut())
    {
        let values = unsafe { _mm_loadu_si128(values.as_ptr().cast()) };
        let west = unsafe { _mm_loadu_si128(west.as_ptr().cast()) };
        let north = unsafe { _mm_loadu_si128(north.as_ptr().cast()) };
        let northwest = unsafe { _mm_loadu_si128(northwest.as_ptr().cast()) };
        tokenize_eight_u8(values, west, north, northwest, &mut out[..8]);
        tokenize_eight_u8(
            _mm_srli_si128::<8>(values),
            _mm_srli_si128::<8>(west),
            _mm_srli_si128::<8>(north),
            _mm_srli_si128::<8>(northwest),
            &mut out[8..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<8>();
    let (west_halves, west_tail) = west_tail.as_chunks::<8>();
    let (north_halves, north_tail) = north_tail.as_chunks::<8>();
    let (northwest_halves, northwest_tail) = northwest_tail.as_chunks::<8>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<8>();
    for ((((values, west), north), northwest), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(north_halves.iter())
        .zip(northwest_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_eight_u8(
            unsafe { _mm_loadl_epi64(values.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(west.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(north.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(northwest.as_ptr().cast()) },
            out,
        );
    }

    for ((((&value, &west), &north), &northwest), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(north_tail.iter())
        .zip(northwest_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u8(value, west, north, northwest);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tokenize_four_u16(
    values: __m128i,
    west: __m128i,
    north: __m128i,
    northwest: __m128i,
    out: &mut [Token],
) {
    debug_assert_eq!(out.len(), 4);
    let v = _mm_cvtepu16_epi32(values);
    let w = _mm_cvtepu16_epi32(west);
    let n = _mm_cvtepu16_epi32(north);
    let nw = _mm_cvtepu16_epi32(northwest);
    let grad = _mm_sub_epi32(_mm_add_epi32(w, n), nw);
    let pred = _mm_max_epi32(
        _mm_min_epi32(w, n),
        _mm_min_epi32(_mm_max_epi32(w, n), grad),
    );
    let residual = _mm_sub_epi32(v, pred);
    let packed = _mm_xor_si128(
        _mm_slli_epi32::<1>(residual),
        _mm_srai_epi32::<31>(residual),
    );
    let contexts = _mm_andnot_si128(
        _mm_cmpgt_epi32(grad, _mm_setzero_si128()),
        _mm_set1_epi32(1),
    );
    store_four_tokens(contexts, packed, out);
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn tokenize_alpha_u16_first_row_sse41(row: &[u16], out: &mut [Token]) {
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<8>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<8>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<8>();
    let zero = _mm_setzero_si128();

    for ((values, west), out) in values.iter().zip(west.iter()).zip(out_chunks.iter_mut()) {
        let values = unsafe { _mm_loadu_si128(values.as_ptr().cast()) };
        let west = unsafe { _mm_loadu_si128(west.as_ptr().cast()) };
        tokenize_four_u16(values, west, zero, zero, &mut out[..4]);
        tokenize_four_u16(
            _mm_srli_si128::<8>(values),
            _mm_srli_si128::<8>(west),
            zero,
            zero,
            &mut out[4..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<4>();
    let (west_halves, west_tail) = west_tail.as_chunks::<4>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<4>();
    for ((values, west), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_four_u16(
            unsafe { _mm_loadl_epi64(values.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(west.as_ptr().cast()) },
            zero,
            zero,
            out,
        );
    }

    for ((&value, &west), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u16(value, west, 0, 0);
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn tokenize_alpha_u16_interior_sse41(row: &[u16], north: &[u16], out: &mut [Token]) {
    assert_eq!(row.len(), north.len());
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<8>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<8>();
    let (north_chunks, north_tail) = north[1..].as_chunks::<8>();
    let (northwest, northwest_tail) = north[..north.len() - 1].as_chunks::<8>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<8>();

    for ((((values, west), north), northwest), out) in values
        .iter()
        .zip(west.iter())
        .zip(north_chunks.iter())
        .zip(northwest.iter())
        .zip(out_chunks.iter_mut())
    {
        let values = unsafe { _mm_loadu_si128(values.as_ptr().cast()) };
        let west = unsafe { _mm_loadu_si128(west.as_ptr().cast()) };
        let north = unsafe { _mm_loadu_si128(north.as_ptr().cast()) };
        let northwest = unsafe { _mm_loadu_si128(northwest.as_ptr().cast()) };
        tokenize_four_u16(values, west, north, northwest, &mut out[..4]);
        tokenize_four_u16(
            _mm_srli_si128::<8>(values),
            _mm_srli_si128::<8>(west),
            _mm_srli_si128::<8>(north),
            _mm_srli_si128::<8>(northwest),
            &mut out[4..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<4>();
    let (west_halves, west_tail) = west_tail.as_chunks::<4>();
    let (north_halves, north_tail) = north_tail.as_chunks::<4>();
    let (northwest_halves, northwest_tail) = northwest_tail.as_chunks::<4>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<4>();
    for ((((values, west), north), northwest), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(north_halves.iter())
        .zip(northwest_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_four_u16(
            unsafe { _mm_loadl_epi64(values.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(west.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(north.as_ptr().cast()) },
            unsafe { _mm_loadl_epi64(northwest.as_ptr().cast()) },
            out,
        );
    }

    for ((((&value, &west), &north), &northwest), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(north_tail.iter())
        .zip(northwest_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u16(value, west, north, northwest);
    }
}
