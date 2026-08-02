/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::entropy::ALPHABET_SIZE;

#[inline]
#[target_feature(enable = "sse4.1")]
fn dirty_log2f_x4(d: __m128) -> __m128 {
    let one = _mm_set1_ps(1.0);
    let mut ix = _mm_castps_si128(d);
    ix = _mm_add_epi32(ix, _mm_set1_epi32((0x3f80_0000u32 - 0x3f35_04f3u32) as i32));
    let n = _mm_sub_epi32(_mm_srli_epi32::<23>(ix), _mm_set1_epi32(0x7f));
    ix = _mm_add_epi32(
        _mm_and_si128(ix, _mm_set1_epi32(0x007f_ffff)),
        _mm_set1_epi32(0x3f35_04f3),
    );

    let a = _mm_castsi128_ps(ix);
    let numerator = _mm_sub_ps(a, one);
    let denominator = _mm_add_ps(a, one);
    let reciprocal0 = _mm_rcp_ps(denominator);
    let reciprocal = _mm_mul_ps(
        reciprocal0,
        _mm_sub_ps(_mm_set1_ps(2.0), _mm_mul_ps(denominator, reciprocal0)),
    );
    let x = _mm_mul_ps(numerator, reciprocal);
    let x2 = _mm_mul_ps(x, x);
    let mut u = _mm_set1_ps(0.412_198_57);
    u = _mm_add_ps(_mm_mul_ps(u, x2), _mm_set1_ps(0.577_078_04));
    u = _mm_add_ps(_mm_mul_ps(u, x2), _mm_set1_ps(0.961_796_7));
    let base = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(2.885_390_1)), _mm_cvtepi32_ps(n));
    _mm_add_ps(_mm_mul_ps(_mm_mul_ps(x2, x), u), base)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn cvtepu32_ps(v: __m128i) -> __m128 {
    let signed = _mm_cvtepi32_ps(v);
    let high = _mm_cmplt_epi32(v, _mm_setzero_si128());
    _mm_add_ps(
        signed,
        _mm_and_ps(_mm_castsi128_ps(high), _mm_set1_ps(4_294_967_296.0)),
    )
}

/// Shannon population cost used by entropy histogram clustering.
///
/// # Safety
/// The caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn counts_bit_cost_sse41(counts: &[u32; ALPHABET_SIZE], total_count: u32) -> f32 {
    debug_assert_ne!(total_count, 0);
    let log_total = _mm_set1_ps(crate::adaptive_quant::dirty_log2f(total_count as f32));
    let one = _mm_set1_ps(1.0);
    // One chain leaves enough room for the log polynomial and its constants in
    // the 16-register SSE file. A second accumulator makes LLVM spill
    // `log_total` on every iteration and is slower despite the extra ILP.
    let mut cost = _mm_setzero_ps();
    for counts4 in counts.as_chunks::<4>().0 {
        let count_i = unsafe { _mm_loadu_si128(counts4.as_ptr().cast()) };
        let count = cvtepu32_ps(count_i);
        let positive = _mm_max_ps(count, one);
        cost = _mm_add_ps(
            cost,
            _mm_mul_ps(count, _mm_sub_ps(log_total, dirty_log2f_x4(positive))),
        );
    }
    super::adaptive_quant::hsum(cost).max(0.0)
}
