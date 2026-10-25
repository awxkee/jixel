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

use core::arch::wasm32::*;

/// The caller must ensure simd128 is available.
#[target_feature(enable = "simd128")]
pub(crate) fn polish_count_nonzero_wasm(values: &[i32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 4 <= values.len() {
        let v = unsafe { v128_load(values.as_ptr().add(i).cast()) };
        count += i32x4_bitmask(i32x4_eq(v, i32x4_splat(0))).count_ones() as usize;
        i += 4;
    }
    values.len() - count - values[i..].iter().filter(|&&v| v == 0).count()
}

/// Separate multiply and add preserve the scalar polish's rounding (no FMA).
/// The caller must ensure simd128 is available.
#[target_feature(enable = "simd128")]
pub(crate) fn polish_add_scaled_wasm(row: &mut [f32], delta: &[f32], sign: f32) {
    assert_eq!(row.len(), delta.len());
    let factor = f32x4_splat(sign);
    let mut i = 0;
    while i + 4 <= row.len() {
        let v = unsafe { v128_load(row.as_ptr().add(i).cast()) };
        let d = unsafe { v128_load(delta.as_ptr().add(i).cast()) };
        let scaled = f32x4_mul(d, factor);
        let sum = f32x4_add(v, scaled);
        unsafe { v128_store(row.as_mut_ptr().add(i).cast(), sum) };
        i += 4;
    }
    for (v, &d) in row[i..].iter_mut().zip(&delta[i..]) {
        *v += sign * d;
    }
}
