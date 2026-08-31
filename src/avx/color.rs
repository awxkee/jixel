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

use std::arch::x86_64::*;

use crate::color::LutSample;

pub(crate) fn color_matrix_shaper_avx2<
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
    debug_assert!(is_x86_feature_detected!("avx2"));
    debug_assert!(is_x86_feature_detected!("fma"));
    unsafe { color_matrix_shaper_avx2_impl(lut, matrix, src, output) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn store_rgb(value: __m128, r: &mut f32, g: &mut f32, b: &mut f32) {
    unsafe {
        _mm_store_ss(r, value);
    }
    *g = f32::from_bits(_mm_extract_ps::<1>(value) as u32);
    *b = f32::from_bits(_mm_extract_ps::<2>(value) as u32);
}

#[target_feature(enable = "avx2,fma")]
fn color_matrix_shaper_avx2_impl<T: LutSample, const LUT_SIZE: usize, const CHANNELS: usize>(
    lut: &[f32; LUT_SIZE],
    matrix: &[f32; 9],
    src: &[[T; CHANNELS]],
    output: [&mut [f32]; 3],
) {
    let [r_out, g_out, b_out] = output;
    let m0 = _mm256_setr_ps(
        matrix[0], matrix[3], matrix[6], 0.0, matrix[0], matrix[3], matrix[6], 0.0,
    );
    let m1 = _mm256_setr_ps(
        matrix[1], matrix[4], matrix[7], 0.0, matrix[1], matrix[4], matrix[7], 0.0,
    );
    let m2 = _mm256_setr_ps(
        matrix[2], matrix[5], matrix[8], 0.0, matrix[2], matrix[5], matrix[8], 0.0,
    );

    let (pairs, remainder) = src.as_chunks::<2>();
    let (r_pairs, r_remainder) = r_out.as_chunks_mut::<2>();
    let (g_pairs, g_remainder) = g_out.as_chunks_mut::<2>();
    let (b_pairs, b_remainder) = b_out.as_chunks_mut::<2>();
    for (((pixels, r_out), g_out), b_out) in pairs.iter().zip(r_pairs).zip(g_pairs).zip(b_pairs) {
        let p0 = pixels[0];
        let p1 = pixels[1];
        let lr0 = lut[p0[0].as_index()];
        let lg0 = lut[p0[1].as_index()];
        let lb0 = lut[p0[2].as_index()];
        let lr1 = lut[p1[0].as_index()];
        let lg1 = lut[p1[1].as_index()];
        let lb1 = lut[p1[2].as_index()];
        let r = _mm256_setr_ps(lr0, lr0, lr0, lr0, lr1, lr1, lr1, lr1);
        let g = _mm256_setr_ps(lg0, lg0, lg0, lg0, lg1, lg1, lg1, lg1);
        let b = _mm256_setr_ps(lb0, lb0, lb0, lb0, lb1, lb1, lb1, lb1);
        let v = _mm256_fmadd_ps(b, m2, _mm256_fmadd_ps(g, m1, _mm256_mul_ps(r, m0)));
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        store_rgb(lo, &mut r_out[0], &mut g_out[0], &mut b_out[0]);
        store_rgb(hi, &mut r_out[1], &mut g_out[1], &mut b_out[1]);
    }

    for (((&pixel, r_out), g_out), b_out) in remainder
        .iter()
        .zip(r_remainder)
        .zip(g_remainder)
        .zip(b_remainder)
    {
        let lr = lut[pixel[0].as_index()];
        let lg = lut[pixel[1].as_index()];
        let lb = lut[pixel[2].as_index()];
        let r = _mm_set1_ps(lr);
        let g = _mm_set1_ps(lg);
        let b = _mm_set1_ps(lb);
        let m0 = _mm256_castps256_ps128(m0);
        let m1 = _mm256_castps256_ps128(m1);
        let m2 = _mm256_castps256_ps128(m2);
        let v = _mm_fmadd_ps(b, m2, _mm_fmadd_ps(g, m1, _mm_mul_ps(r, m0)));
        store_rgb(v, r_out, g_out, b_out);
    }
}
