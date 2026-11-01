/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Plane division by a loop-invariant step through [`DividerI32`]: every
//! sample becomes its nearest multiple of `q` (ties away from zero), stored
//! divided by `q`. Powers of two take a plain shift; other steps a 32x32->64
//! multiply-high, plus the numerator when the magic needs 33 bits.

use crate::fast_divide::{DividerI32, rounded_quotients_scalar};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// `m * magic >> 32`, lane-wise: even and odd lanes through `pmuldq`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mulhi_epi32(m: __m128i, magic: __m128i) -> __m128i {
    let even = _mm_srli_epi64::<32>(_mm_mul_epi32(m, magic));
    let odd = _mm_mul_epi32(_mm_srli_epi64::<32>(m), magic);
    _mm_blend_epi16::<0xCC>(even, odd)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn magic_quotients<const ADD: bool>(v4s: &mut [[i32; 4]], half: i32, divider: DividerI32) {
    let half = _mm_set1_epi32(half);
    let magic = _mm_set1_epi32(divider.magic());
    let shift = _mm_cvtsi32_si128(divider.shift() as i32);
    for v4 in v4s {
        let v = unsafe { _mm_loadu_si128(v4.as_ptr().cast()) };
        let sign = _mm_srai_epi32::<31>(v);
        let m = _mm_add_epi32(_mm_abs_epi32(v), half);
        let mut uq = mulhi_epi32(m, magic);
        if ADD {
            uq = _mm_add_epi32(uq, m);
        }
        let q = _mm_sra_epi32(uq, shift);
        let q = _mm_sub_epi32(q, _mm_srai_epi32::<31>(q));
        let out = _mm_sub_epi32(_mm_xor_si128(q, sign), sign);
        unsafe { _mm_storeu_si128(v4.as_mut_ptr().cast(), out) };
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn rounded_quotients_sse41(plane: &mut [i32], q: i32) {
    debug_assert!(q > 1);
    let divider = DividerI32::new(q);
    let (v4s, tail) = plane.as_chunks_mut::<4>();
    if divider.magic() == 0 {
        let half = _mm_set1_epi32(q / 2);
        let shift = _mm_cvtsi32_si128(divider.shift() as i32);
        for v4 in v4s.iter_mut() {
            let v = unsafe { _mm_loadu_si128(v4.as_ptr().cast()) };
            let sign = _mm_srai_epi32::<31>(v);
            let r = _mm_sra_epi32(_mm_add_epi32(_mm_abs_epi32(v), half), shift);
            let out = _mm_sub_epi32(_mm_xor_si128(r, sign), sign);
            unsafe { _mm_storeu_si128(v4.as_mut_ptr().cast(), out) };
        }
    } else if divider.adds_numerator() {
        magic_quotients::<true>(v4s, q / 2, divider);
    } else {
        magic_quotients::<false>(v4s, q / 2, divider);
    }
    rounded_quotients_scalar(tail, q);
}
