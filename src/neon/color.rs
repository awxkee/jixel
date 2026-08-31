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

use std::arch::aarch64::*;

use crate::color::LutSample;

pub(crate) fn color_matrix_shaper_neon<
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
    // NEON is part of the AArch64 baseline. This module is only built for
    // AArch64 when the `neon` feature is enabled.
    unsafe { color_matrix_shaper_neon_impl(lut, matrix, src, output) }
}

#[inline]
#[target_feature(enable = "neon")]
fn transform_pixel<T: LutSample, const LUT_SIZE: usize, const CHANNELS: usize>(
    lut: &[f32; LUT_SIZE],
    pixel: [T; CHANNELS],
    m0: float32x4_t,
    m1: float32x4_t,
    m2: float32x4_t,
) -> float32x4_t {
    let r = vdupq_n_f32(lut[pixel[0].as_index()]);
    let g = vdupq_n_f32(lut[pixel[1].as_index()]);
    let b = vdupq_n_f32(lut[pixel[2].as_index()]);
    vfmaq_f32(vfmaq_f32(vmulq_f32(r, m0), g, m1), b, m2)
}

#[inline]
#[target_feature(enable = "neon")]
fn store_pixel(value: float32x4_t, r: &mut f32, g: &mut f32, b: &mut f32) {
    unsafe {
        vst1q_lane_f32::<0>(r, value);
        vst1q_lane_f32::<1>(g, value);
        vst1q_lane_f32::<2>(b, value);
    }
}

#[target_feature(enable = "neon")]
fn color_matrix_shaper_neon_impl<T: LutSample, const LUT_SIZE: usize, const CHANNELS: usize>(
    lut: &[f32; LUT_SIZE],
    matrix: &[f32; 9],
    src: &[[T; CHANNELS]],
    output: [&mut [f32]; 3],
) {
    let [r_out, g_out, b_out] = output;
    let m0_values = [matrix[0], matrix[3], matrix[6], 0.0];
    let m1_values = [matrix[1], matrix[4], matrix[7], 0.0];
    let m2_values = [matrix[2], matrix[5], matrix[8], 0.0];
    let m0 = unsafe { vld1q_f32(m0_values.as_ptr()) };
    let m1 = unsafe { vld1q_f32(m1_values.as_ptr()) };
    let m2 = unsafe { vld1q_f32(m2_values.as_ptr()) };

    let (groups, remainder) = src.as_chunks::<4>();
    let (r_groups, r_remainder) = r_out.as_chunks_mut::<4>();
    let (g_groups, g_remainder) = g_out.as_chunks_mut::<4>();
    let (b_groups, b_remainder) = b_out.as_chunks_mut::<4>();
    for (((pixels, r_out), g_out), b_out) in groups.iter().zip(r_groups).zip(g_groups).zip(b_groups)
    {
        let v0 = transform_pixel(lut, pixels[0], m0, m1, m2);
        let v1 = transform_pixel(lut, pixels[1], m0, m1, m2);
        let v2 = transform_pixel(lut, pixels[2], m0, m1, m2);
        let v3 = transform_pixel(lut, pixels[3], m0, m1, m2);
        store_pixel(v0, &mut r_out[0], &mut g_out[0], &mut b_out[0]);
        store_pixel(v1, &mut r_out[1], &mut g_out[1], &mut b_out[1]);
        store_pixel(v2, &mut r_out[2], &mut g_out[2], &mut b_out[2]);
        store_pixel(v3, &mut r_out[3], &mut g_out[3], &mut b_out[3]);
    }

    for (((&pixel, r_out), g_out), b_out) in remainder
        .iter()
        .zip(r_remainder)
        .zip(g_remainder)
        .zip(b_remainder)
    {
        let v = transform_pixel(lut, pixel, m0, m1, m2);
        store_pixel(v, r_out, g_out, b_out);
    }
}
