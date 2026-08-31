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

use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) fn reduce_f32x4(v: v128) -> f32 {
    let pair = f32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    f32x4_extract_lane::<0>(f32x4_add(pair, i32x4_shuffle::<1, 0, 3, 2>(pair, pair)))
}

#[target_feature(enable = "simd128")]
pub(crate) fn chroma_gradient_sums_wasm(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32],
    width: usize,
) -> [f32; 4] {
    if width < 2 {
        return [0.0; 4];
    }

    let mut sum_x = f32x4_splat(0.0);
    let mut sum_y_x = f32x4_splat(0.0);
    let mut sum_by = f32x4_splat(0.0);
    let mut sum_y_b = f32x4_splat(0.0);
    let mut tail_x = 0.0f32;
    let mut tail_y_x = 0.0f32;
    let mut tail_by = 0.0f32;
    let mut tail_y_b = 0.0f32;
    let diff_count = width - 1;
    let rows = x_plane
        .chunks_exact(width)
        .zip(y_plane.chunks_exact(width))
        .zip(b_plane.chunks_exact(width))
        .step_by(4);

    for ((x, y), b) in rows {
        let (x0_chunks, x0_tail) = x[..diff_count].as_chunks::<4>();
        let (x1_chunks, x1_tail) = x[1..].as_chunks::<4>();
        let (y0_chunks, y0_tail) = y[..diff_count].as_chunks::<4>();
        let (y1_chunks, y1_tail) = y[1..].as_chunks::<4>();
        let (b0_chunks, b0_tail) = b[..diff_count].as_chunks::<4>();
        let (b1_chunks, b1_tail) = b[1..].as_chunks::<4>();
        for (((((x0, x1), y0), y1), b0), b1) in x0_chunks
            .iter()
            .zip(x1_chunks)
            .zip(y0_chunks)
            .zip(y1_chunks)
            .zip(b0_chunks)
            .zip(b1_chunks)
        {
            let x0 = unsafe { v128_load(x0.as_ptr().cast()) };
            let x1 = unsafe { v128_load(x1.as_ptr().cast()) };
            let y0 = unsafe { v128_load(y0.as_ptr().cast()) };
            let y1 = unsafe { v128_load(y1.as_ptr().cast()) };
            let b0 = unsafe { v128_load(b0.as_ptr().cast()) };
            let b1 = unsafe { v128_load(b1.as_ptr().cast()) };
            let dy = f32x4_abs(f32x4_sub(y1, y0));
            sum_x = f32x4_add(sum_x, f32x4_abs(f32x4_sub(x1, x0)));
            sum_y_x = f32x4_add(sum_y_x, dy);
            sum_by = f32x4_add(
                sum_by,
                f32x4_abs(f32x4_sub(f32x4_sub(b1, y1), f32x4_sub(b0, y0))),
            );
            sum_y_b = f32x4_add(sum_y_b, dy);
        }

        for (((((&x0, &x1), &y0), &y1), &b0), &b1) in x0_tail
            .iter()
            .zip(x1_tail)
            .zip(y0_tail)
            .zip(y1_tail)
            .zip(b0_tail)
            .zip(b1_tail)
        {
            let dy = (y1 - y0).abs();
            tail_x += (x1 - x0).abs();
            tail_y_x += dy;
            tail_by += ((b1 - y1) - (b0 - y0)).abs();
            tail_y_b += dy;
        }
    }

    [
        reduce_f32x4(sum_x) + tail_x,
        reduce_f32x4(sum_y_x) + tail_y_x,
        reduce_f32x4(sum_by) + tail_by,
        reduce_f32x4(sum_y_b) + tail_y_b,
    ]
}
