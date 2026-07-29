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
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn quantize_xyb_channels_neon(
    input: [&[f32]; 3],
    output: [&mut [i32]; 3],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [dst_y, dst_x, dst_b] = output;
    let [scale_x, scale_y, scale_b] = scales;
    let (x4s, x_tail) = src_x.as_chunks::<4>();
    let (y4s, y_tail) = src_y.as_chunks::<4>();
    let (b4s, b_tail) = src_b.as_chunks::<4>();
    let (yq4s, yq_tail) = dst_y.as_chunks_mut::<4>();
    let (xq4s, xq_tail) = dst_x.as_chunks_mut::<4>();
    let (bq4s, bq_tail) = dst_b.as_chunks_mut::<4>();
    let sx = vdupq_n_f32(scale_x);
    let sy = vdupq_n_f32(scale_y);
    let sb = vdupq_n_f32(scale_b);

    for (((((x4, y4), b4), yq4), xq4), bq4) in
        x4s.iter().zip(y4s).zip(b4s).zip(yq4s).zip(xq4s).zip(bq4s)
    {
        let x = unsafe { vld1q_f32(x4.as_ptr()) };
        let y = unsafe { vld1q_f32(y4.as_ptr()) };
        let b = unsafe { vld1q_f32(b4.as_ptr()) };
        let yq = vcvtaq_s32_f32(vmulq_f32(y, sy));
        let xq = vcvtaq_s32_f32(vmulq_f32(x, sx));
        let bq = vcvtaq_s32_f32(vmulq_f32(b, sb));
        unsafe {
            vst1q_s32(yq4.as_mut_ptr(), yq);
            vst1q_s32(xq4.as_mut_ptr(), xq);
            vst1q_s32(bq4.as_mut_ptr(), vsubq_s32(bq, yq));
        }
    }

    let src = x_tail.iter().zip(y_tail).zip(b_tail);
    let dst = yq_tail.iter_mut().zip(xq_tail).zip(bq_tail);
    for (((x, y), b), ((yq, xq), bq)) in src.zip(dst) {
        let quantized_y = (*y * scale_y).round() as i32;
        *yq = quantized_y;
        *xq = (*x * scale_x).round() as i32;
        *bq = (*b * scale_b).round() as i32 - quantized_y;
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn quantize_xyb_tile_colors_neon(
    input: [&[f32]; 3],
    output: &mut [[i32; 3]],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [scale_x, scale_y, scale_b] = scales;
    let (x4s, x_tail) = src_x.as_chunks::<4>();
    let (y4s, y_tail) = src_y.as_chunks::<4>();
    let (b4s, b_tail) = src_b.as_chunks::<4>();
    let (out4s, out_tail) = output.as_chunks_mut::<4>();
    let sx = vdupq_n_f32(scale_x);
    let sy = vdupq_n_f32(scale_y);
    let sb = vdupq_n_f32(scale_b);

    for (((x4, y4), b4), out4) in x4s.iter().zip(y4s).zip(b4s).zip(out4s) {
        let x = unsafe { vld1q_f32(x4.as_ptr()) };
        let y = unsafe { vld1q_f32(y4.as_ptr()) };
        let b = unsafe { vld1q_f32(b4.as_ptr()) };
        let yq = vcvtaq_s32_f32(vmulq_f32(y, sy));
        let xq = vcvtaq_s32_f32(vmulq_f32(x, sx));
        let bq = vcvtaq_s32_f32(vmulq_f32(b, sb));
        unsafe {
            vst3q_s32(out4.as_mut_ptr().cast(), int32x4x3_t(yq, xq, bq));
        }
    }

    let src = x_tail.iter().zip(y_tail).zip(b_tail);
    for (((x, y), b), out) in src.zip(out_tail) {
        *out = [
            (*y * scale_y).round() as i32,
            (*x * scale_x).round() as i32,
            (*b * scale_b).round() as i32,
        ];
    }
}
