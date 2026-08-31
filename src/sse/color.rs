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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::color::LutSample;

pub(crate) fn color_matrix_shaper_sse41<
    T: LutSample,
    const LUT_SIZE: usize,
    const CHANNELS: usize,
>(
    lut: &[f32; LUT_SIZE],
    matrix: &[f32; 9],
    src: &[[T; CHANNELS]],
    output: [&mut [f32]; 3],
) {
    assert!(CHANNELS >= 3);
    assert_eq!(src.len(), output[0].len());
    assert_eq!(src.len(), output[1].len());
    assert_eq!(src.len(), output[2].len());
    debug_assert!(is_x86_feature_detected!("sse4.1"));
    unsafe { color_matrix_shaper_sse41_impl(lut, matrix, src, output) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_rgb(value: __m128, r: &mut f32, g: &mut f32, b: &mut f32) {
    unsafe {
        _mm_store_ss(r, value);
    }
    *g = f32::from_bits(_mm_extract_ps::<1>(value) as u32);
    *b = f32::from_bits(_mm_extract_ps::<2>(value) as u32);
}

#[target_feature(enable = "sse4.1")]
fn color_matrix_shaper_sse41_impl<T: LutSample, const LUT_SIZE: usize, const CHANNELS: usize>(
    lut: &[f32; LUT_SIZE],
    matrix: &[f32; 9],
    src: &[[T; CHANNELS]],
    output: [&mut [f32]; 3],
) {
    let [r_out, g_out, b_out] = output;
    let m0 = _mm_setr_ps(matrix[0], matrix[3], matrix[6], 0.0);
    let m1 = _mm_setr_ps(matrix[1], matrix[4], matrix[7], 0.0);
    let m2 = _mm_setr_ps(matrix[2], matrix[5], matrix[8], 0.0);
    for (((&pixel, r_out), g_out), b_out) in src.iter().zip(r_out).zip(g_out).zip(b_out) {
        let r = _mm_set1_ps(lut[pixel[0].as_index()]);
        let g = _mm_set1_ps(lut[pixel[1].as_index()]);
        let b = _mm_set1_ps(lut[pixel[2].as_index()]);
        let v = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(r, m0), _mm_mul_ps(g, m1)),
            _mm_mul_ps(b, m2),
        );
        store_rgb(v, r_out, g_out, b_out);
    }
}
