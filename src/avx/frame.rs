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
use crate::avx::ac_strategy::hsum256;
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn chroma_gradient_sums_avx2(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32],
    width: usize,
) -> [f32; 4] {
    if width < 2 {
        return [0.0; 4];
    }

    let mut sum_x = _mm256_setzero_ps();
    let mut sum_y_x = _mm256_setzero_ps();
    let mut sum_by = _mm256_setzero_ps();
    let mut sum_y_b = _mm256_setzero_ps();
    let sign = _mm256_set1_ps(-0.0);
    let diff_count = width - 1;
    let tail_len = diff_count % 8;
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let tail_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(tail_len as i32), lanes);
    let rows = x_plane
        .chunks_exact(width)
        .zip(y_plane.chunks_exact(width))
        .zip(b_plane.chunks_exact(width))
        .step_by(4);

    for ((x, y), b) in rows {
        let (x0_chunks, x0_tail) = x[..diff_count].as_chunks::<8>();
        let (x1_chunks, x1_tail) = x[1..].as_chunks::<8>();
        let (y0_chunks, y0_tail) = y[..diff_count].as_chunks::<8>();
        let (y1_chunks, y1_tail) = y[1..].as_chunks::<8>();
        let (b0_chunks, b0_tail) = b[..diff_count].as_chunks::<8>();
        let (b1_chunks, b1_tail) = b[1..].as_chunks::<8>();
        for (((((x0, x1), y0), y1), b0), b1) in x0_chunks
            .iter()
            .zip(x1_chunks)
            .zip(y0_chunks)
            .zip(y1_chunks)
            .zip(b0_chunks)
            .zip(b1_chunks)
        {
            let x0 = unsafe { _mm256_loadu_ps(x0.as_ptr()) };
            let x1 = unsafe { _mm256_loadu_ps(x1.as_ptr()) };
            let y0 = unsafe { _mm256_loadu_ps(y0.as_ptr()) };
            let y1 = unsafe { _mm256_loadu_ps(y1.as_ptr()) };
            let b0 = unsafe { _mm256_loadu_ps(b0.as_ptr()) };
            let b1 = unsafe { _mm256_loadu_ps(b1.as_ptr()) };
            let dy = _mm256_andnot_ps(sign, _mm256_sub_ps(y1, y0));
            sum_x = _mm256_add_ps(sum_x, _mm256_andnot_ps(sign, _mm256_sub_ps(x1, x0)));
            sum_y_x = _mm256_add_ps(sum_y_x, dy);
            sum_by = _mm256_add_ps(
                sum_by,
                _mm256_andnot_ps(
                    sign,
                    _mm256_sub_ps(_mm256_sub_ps(b1, y1), _mm256_sub_ps(b0, y0)),
                ),
            );
            sum_y_b = _mm256_add_ps(sum_y_b, dy);
        }

        if tail_len != 0 {
            let x0 = unsafe { _mm256_maskload_ps(x0_tail.as_ptr(), tail_mask) };
            let x1 = unsafe { _mm256_maskload_ps(x1_tail.as_ptr(), tail_mask) };
            let y0 = unsafe { _mm256_maskload_ps(y0_tail.as_ptr(), tail_mask) };
            let y1 = unsafe { _mm256_maskload_ps(y1_tail.as_ptr(), tail_mask) };
            let b0 = unsafe { _mm256_maskload_ps(b0_tail.as_ptr(), tail_mask) };
            let b1 = unsafe { _mm256_maskload_ps(b1_tail.as_ptr(), tail_mask) };
            let dy = _mm256_andnot_ps(sign, _mm256_sub_ps(y1, y0));
            sum_x = _mm256_add_ps(sum_x, _mm256_andnot_ps(sign, _mm256_sub_ps(x1, x0)));
            sum_y_x = _mm256_add_ps(sum_y_x, dy);
            sum_by = _mm256_add_ps(
                sum_by,
                _mm256_andnot_ps(
                    sign,
                    _mm256_sub_ps(_mm256_sub_ps(b1, y1), _mm256_sub_ps(b0, y0)),
                ),
            );
            sum_y_b = _mm256_add_ps(sum_y_b, dy);
        }
    }

    [
        hsum256(sum_x),
        hsum256(sum_y_x),
        hsum256(sum_by),
        hsum256(sum_y_b),
    ]
}
