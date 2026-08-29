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

#[target_feature(enable = "neon")]
pub(crate) fn x_gradient_sums_neon(x_plane: &[f32], y_plane: &[f32], width: usize) -> [f32; 2] {
    if width < 2 {
        return [0.0; 2];
    }

    let mut sum_x = vdupq_n_f32(0.0);
    let mut sum_y = vdupq_n_f32(0.0);
    let mut tail_x = 0.0f32;
    let mut tail_y = 0.0f32;
    let diff_count = width - 1;
    let rows = x_plane
        .chunks_exact(width)
        .zip(y_plane.chunks_exact(width))
        .step_by(4);

    for (x, y) in rows {
        let (x0_chunks, x0_tail) = x[..diff_count].as_chunks::<4>();
        let (x1_chunks, x1_tail) = x[1..].as_chunks::<4>();
        let (y0_chunks, y0_tail) = y[..diff_count].as_chunks::<4>();
        let (y1_chunks, y1_tail) = y[1..].as_chunks::<4>();
        for (((x0, x1), y0), y1) in x0_chunks
            .iter()
            .zip(x1_chunks)
            .zip(y0_chunks)
            .zip(y1_chunks)
        {
            let x0 = unsafe { vld1q_f32(x0.as_ptr()) };
            let x1 = unsafe { vld1q_f32(x1.as_ptr()) };
            let y0 = unsafe { vld1q_f32(y0.as_ptr()) };
            let y1 = unsafe { vld1q_f32(y1.as_ptr()) };
            sum_x = vaddq_f32(sum_x, vabsq_f32(vsubq_f32(x1, x0)));
            sum_y = vaddq_f32(sum_y, vabsq_f32(vsubq_f32(y1, y0)));
        }

        for (((&x0, &x1), &y0), &y1) in x0_tail.iter().zip(x1_tail).zip(y0_tail).zip(y1_tail) {
            tail_x += (x1 - x0).abs();
            tail_y += (y1 - y0).abs();
        }
    }

    [vaddvq_f32(sum_x) + tail_x, vaddvq_f32(sum_y) + tail_y]
}
