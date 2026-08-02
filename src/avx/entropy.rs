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

use std::arch::x86_64::*;

use crate::entropy::ALPHABET_SIZE;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dirty_log2f_x8(d: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let mut ix = _mm256_castps_si256(d);
    ix = _mm256_add_epi32(
        ix,
        _mm256_set1_epi32((0x3f80_0000u32 - 0x3f35_04f3u32) as i32),
    );
    let n = _mm256_sub_epi32(_mm256_srli_epi32::<23>(ix), _mm256_set1_epi32(0x7f));
    ix = _mm256_add_epi32(
        _mm256_and_si256(ix, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f35_04f3),
    );

    let a = _mm256_castsi256_ps(ix);
    let numerator = _mm256_sub_ps(a, one);
    let denominator = _mm256_add_ps(a, one);
    let reciprocal0 = _mm256_rcp_ps(denominator);
    let reciprocal = _mm256_mul_ps(
        reciprocal0,
        _mm256_fnmadd_ps(denominator, reciprocal0, _mm256_set1_ps(2.0)),
    );
    let x = _mm256_mul_ps(numerator, reciprocal);
    let x2 = _mm256_mul_ps(x, x);
    let mut u = _mm256_set1_ps(0.412_198_57);
    u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(0.577_078_04));
    u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(0.961_796_7));
    let base = _mm256_fmadd_ps(x, _mm256_set1_ps(2.885_390_1), _mm256_cvtepi32_ps(n));
    _mm256_fmadd_ps(_mm256_mul_ps(x2, x), u, base)
}

#[inline]
#[target_feature(enable = "avx2")]
fn cvtepu32_ps(v: __m256i) -> __m256 {
    let signed = _mm256_cvtepi32_ps(v);
    let high = _mm256_cmpgt_epi32(_mm256_setzero_si256(), v);
    _mm256_add_ps(
        signed,
        _mm256_and_ps(_mm256_castsi256_ps(high), _mm256_set1_ps(4_294_967_296.0)),
    )
}

/// Shannon population cost used by entropy histogram clustering.
///
/// # Safety
/// The caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn counts_bit_cost_avx2(counts: &[u32; ALPHABET_SIZE], total_count: u32) -> f32 {
    debug_assert_ne!(total_count, 0);
    let log_total = _mm256_set1_ps(crate::adaptive_quant::dirty_log2f(total_count as f32));
    let one = _mm256_set1_ps(1.0);
    let mut cost0 = _mm256_setzero_ps();
    let mut cost1 = _mm256_setzero_ps();
    for counts16 in counts.as_chunks::<16>().0 {
        let count0_i = unsafe { _mm256_loadu_si256(counts16.as_ptr().cast()) };
        let count1_i = unsafe { _mm256_loadu_si256(counts16.as_ptr().add(8).cast()) };
        let count0 = cvtepu32_ps(count0_i);
        let count1 = cvtepu32_ps(count1_i);
        let positive0 = _mm256_max_ps(count0, one);
        let positive1 = _mm256_max_ps(count1, one);
        cost0 = _mm256_fmadd_ps(
            count0,
            _mm256_sub_ps(log_total, dirty_log2f_x8(positive0)),
            cost0,
        );
        cost1 = _mm256_fmadd_ps(
            count1,
            _mm256_sub_ps(log_total, dirty_log2f_x8(positive1)),
            cost1,
        );
    }
    super::ac_strategy::hsum256(_mm256_add_ps(cost0, cost1)).max(0.0)
}
