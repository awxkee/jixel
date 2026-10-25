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

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// The caller must ensure sse4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn polish_count_nonzero_sse41(values: &[i32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 4 <= values.len() {
        let v = unsafe { _mm_loadu_si128(values.as_ptr().add(i).cast()) };
        let eq = _mm_cmpeq_epi32(v, _mm_setzero_si128());
        count += _mm_movemask_ps(_mm_castsi128_ps(eq)).count_ones() as usize;
        i += 4;
    }
    values.len() - count - values[i..].iter().filter(|&&v| v == 0).count()
}

/// Separate multiply and add preserve the scalar polish's rounding (no FMA).
/// The caller must ensure sse4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn polish_add_scaled_sse41(row: &mut [f32], delta: &[f32], sign: f32) {
    assert_eq!(row.len(), delta.len());
    let factor = _mm_set1_ps(sign);
    let mut i = 0;
    while i + 4 <= row.len() {
        let v = unsafe { _mm_loadu_ps(row.as_ptr().add(i)) };
        let d = unsafe { _mm_loadu_ps(delta.as_ptr().add(i)) };
        let scaled = _mm_mul_ps(d, factor);
        let sum = _mm_add_ps(v, scaled);
        unsafe { _mm_storeu_ps(row.as_mut_ptr().add(i), sum) };
        i += 4;
    }
    for (v, &d) in row[i..].iter_mut().zip(&delta[i..]) {
        *v += sign * d;
    }
}
