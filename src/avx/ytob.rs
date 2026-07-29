/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
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

#[inline]
#[target_feature(enable = "avx2")]
fn fill_ytob_8(b: __m128i, y: __m128i, slope: __m256) -> __m256i {
    let b = _mm256_cvtepi16_epi32(b);
    let y = _mm256_cvtepi16_epi32(y);
    let adjusted = _mm256_mul_ps(slope, _mm256_cvtepi32_ps(y));
    let adjusted = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(adjusted);
    _mm256_sub_epi32(b, _mm256_cvttps_epi32(adjusted))
}

#[inline]
#[target_feature(enable = "avx2")]
fn grad_residual_8(current: __m128i, north: __m128i, west: __m128i, northwest: __m128i) -> __m256i {
    let current = _mm256_cvtepi16_epi32(current);
    let north = _mm256_cvtepi16_epi32(north);
    let west = _mm256_cvtepi16_epi32(west);
    let northwest = _mm256_cvtepi16_epi32(northwest);
    let prediction = _mm256_sub_epi32(_mm256_add_epi32(north, west), northwest);
    let prediction = _mm256_max_epi32(
        _mm256_min_epi32(north, west),
        _mm256_min_epi32(_mm256_max_epi32(north, west), prediction),
    );
    _mm256_sub_epi32(current, prediction)
}

#[inline]
fn grad_residual_scalar(current: i16, north: i16, west: i16, northwest: i16) -> i32 {
    let (current, north, west, northwest) =
        (current as i32, north as i32, west as i32, northwest as i32);
    current - (north + west - northwest).clamp(north.min(west), north.max(west))
}

#[inline]
#[target_feature(enable = "avx2")]
fn fill_grad_residuals_plane(dst: &mut [i32], row: &[i16], up: &[i16]) {
    let Some((dst_first, dst_rest)) = dst.split_first_mut() else {
        return;
    };
    let (&row_first, current) = row.split_first().unwrap();
    let (&up_first, north) = up.split_first().unwrap();
    *dst_first = row_first as i32 - up_first as i32;
    let (_, west) = row.split_last().unwrap();
    let (_, northwest) = up.split_last().unwrap();

    let (dst_chunks, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (current_chunks, current_tail) = current.as_chunks::<16>();
    let (north_chunks, north_tail) = north.as_chunks::<16>();
    let (west_chunks, west_tail) = west.as_chunks::<16>();
    let (northwest_chunks, northwest_tail) = northwest.as_chunks::<16>();

    for ((((dst, current), north), west), northwest) in dst_chunks
        .iter_mut()
        .zip(current_chunks)
        .zip(north_chunks)
        .zip(west_chunks)
        .zip(northwest_chunks)
    {
        let current = unsafe { _mm256_loadu_si256(current.as_ptr().cast()) };
        let north = unsafe { _mm256_loadu_si256(north.as_ptr().cast()) };
        let west = unsafe { _mm256_loadu_si256(west.as_ptr().cast()) };
        let northwest = unsafe { _mm256_loadu_si256(northwest.as_ptr().cast()) };
        let result_lo = grad_residual_8(
            _mm256_castsi256_si128(current),
            _mm256_castsi256_si128(north),
            _mm256_castsi256_si128(west),
            _mm256_castsi256_si128(northwest),
        );
        let result_hi = grad_residual_8(
            _mm256_extracti128_si256::<1>(current),
            _mm256_extracti128_si256::<1>(north),
            _mm256_extracti128_si256::<1>(west),
            _mm256_extracti128_si256::<1>(northwest),
        );
        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), result_lo);
            _mm256_storeu_si256(dst[8..].as_mut_ptr().cast(), result_hi);
        }
    }

    for ((((dst, &current), &north), &west), &northwest) in dst_tail
        .iter_mut()
        .zip(current_tail)
        .zip(north_tail)
        .zip(west_tail)
        .zip(northwest_tail)
    {
        *dst = grad_residual_scalar(current, north, west, northwest);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn fill_ytob_residuals_avx2(
    rb: &mut [i32],
    ry: &mut [i32],
    b_row: &[i16],
    y_row: &[i16],
    b_up: &[i16],
    y_up: &[i16],
) {
    let len = rb
        .len()
        .min(ry.len())
        .min(b_row.len())
        .min(y_row.len())
        .min(b_up.len())
        .min(y_up.len());
    if len == 0 {
        return;
    }

    fill_grad_residuals_plane(&mut rb[..len], &b_row[..len], &b_up[..len]);
    fill_grad_residuals_plane(&mut ry[..len], &y_row[..len], &y_up[..len]);
}

#[inline]
#[target_feature(enable = "avx2")]
fn ytob_weight_indices_8(rb: __m256i, ry: __m256i, step: __m256) -> (__m256i, __m256i) {
    let ratio = _mm256_div_ps(
        _mm256_cvtepi32_ps(rb),
        _mm256_mul_ps(_mm256_cvtepi32_ps(ry), step),
    );
    let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(ratio);
    let k = _mm256_cvttps_epi32(rounded);
    let k = _mm256_max_epi32(
        _mm256_set1_epi32(-127),
        _mm256_min_epi32(_mm256_set1_epi32(127), k),
    );
    (
        _mm256_add_epi32(k, _mm256_set1_epi32(127)),
        _mm256_abs_epi32(ry),
    )
}

#[inline(always)]
fn scatter_ytob_weights(
    indices: &[i32; 8],
    magnitudes: &[i32; 8],
    lanes: usize,
    weights: &mut [u64],
) {
    for (&idx, &magnitude) in indices[..lanes].iter().zip(&magnitudes[..lanes]) {
        if magnitude != 0 {
            weights[idx as usize] += magnitude as u64;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn accumulate_ytob_weights_avx2(rb: &[i32], ry: &[i32], step: f32, weights: &mut [u64]) {
    let len = rb.len().min(ry.len());
    let (rb_chunks, rb_tail) = rb[..len].as_chunks::<8>();
    let (ry_chunks, ry_tail) = ry[..len].as_chunks::<8>();
    let step = _mm256_set1_ps(step);

    for (rb, ry) in rb_chunks.iter().zip(ry_chunks) {
        let rb = unsafe { _mm256_loadu_si256(rb.as_ptr().cast()) };
        let ry = unsafe { _mm256_loadu_si256(ry.as_ptr().cast()) };
        let (indices, magnitudes) = ytob_weight_indices_8(rb, ry, step);
        let mut indices_array = [0i32; 8];
        let mut magnitudes_array = [0i32; 8];
        unsafe {
            _mm256_storeu_si256(indices_array.as_mut_ptr().cast(), indices);
            _mm256_storeu_si256(magnitudes_array.as_mut_ptr().cast(), magnitudes);
        }
        scatter_ytob_weights(&indices_array, &magnitudes_array, 8, weights);
    }

    if !rb_tail.is_empty() {
        let mut rb_padded = [0i32; 8];
        let mut ry_padded = [0i32; 8];
        rb_padded[..rb_tail.len()].copy_from_slice(rb_tail);
        ry_padded[..ry_tail.len()].copy_from_slice(ry_tail);
        let rb = unsafe { _mm256_loadu_si256(rb_padded.as_ptr().cast()) };
        let ry = unsafe { _mm256_loadu_si256(ry_padded.as_ptr().cast()) };
        let (indices, magnitudes) = ytob_weight_indices_8(rb, ry, step);
        let mut indices_array = [0i32; 8];
        let mut magnitudes_array = [0i32; 8];
        unsafe {
            _mm256_storeu_si256(indices_array.as_mut_ptr().cast(), indices);
            _mm256_storeu_si256(magnitudes_array.as_mut_ptr().cast(), magnitudes);
        }
        scatter_ytob_weights(&indices_array, &magnitudes_array, rb_tail.len(), weights);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn fill_ytob_row_avx2(dst: &mut [i32], b: &[i16], y: &[i16], slope: f32) {
    let len = dst.len().min(b.len()).min(y.len());
    let (dst_chunks, dst_tail) = dst[..len].as_chunks_mut::<16>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<16>();
    let (y_chunks, y_tail) = y[..len].as_chunks::<16>();
    let slope_vec = _mm256_set1_ps(slope);

    for ((dst, b), y) in dst_chunks.iter_mut().zip(b_chunks).zip(y_chunks) {
        let b16 = unsafe { _mm256_loadu_si256(b.as_ptr().cast()) };
        let y16 = unsafe { _mm256_loadu_si256(y.as_ptr().cast()) };
        let result_lo = fill_ytob_8(
            _mm256_castsi256_si128(b16),
            _mm256_castsi256_si128(y16),
            slope_vec,
        );
        let result_hi = fill_ytob_8(
            _mm256_extracti128_si256::<1>(b16),
            _mm256_extracti128_si256::<1>(y16),
            slope_vec,
        );
        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), result_lo);
            _mm256_storeu_si256(dst[8..].as_mut_ptr().cast(), result_hi);
        }
    }

    if !dst_tail.is_empty() {
        let mut b_padded = [0i16; 16];
        let mut y_padded = [0i16; 16];
        b_padded[..b_tail.len()].copy_from_slice(b_tail);
        y_padded[..y_tail.len()].copy_from_slice(y_tail);

        let b16 = unsafe { _mm256_loadu_si256(b_padded.as_ptr().cast()) };
        let y16 = unsafe { _mm256_loadu_si256(y_padded.as_ptr().cast()) };
        let result_lo = fill_ytob_8(
            _mm256_castsi256_si128(b16),
            _mm256_castsi256_si128(y16),
            slope_vec,
        );
        let result_hi = fill_ytob_8(
            _mm256_extracti128_si256::<1>(b16),
            _mm256_extracti128_si256::<1>(y16),
            slope_vec,
        );

        let mut result_padded = [0i32; 16];
        unsafe {
            _mm256_storeu_si256(result_padded.as_mut_ptr().cast(), result_lo);
            _mm256_storeu_si256(result_padded[8..].as_mut_ptr().cast(), result_hi);
        }
        dst_tail.copy_from_slice(&result_padded[..dst_tail.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::{accumulate_ytob_weights_avx2, fill_ytob_residuals_avx2, fill_ytob_row_avx2};
    use crate::enc_color_correlation::{
        accumulate_ytob_weights_scalar, fill_ytob_residuals_scalar, fill_ytob_row_scalar,
    };

    #[test]
    fn fill_ytob_row_matches_scalar_ties_to_even() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let b = std::array::from_fn::<_, 33, _>(|i| i as i16 * 100 - 1600);
        let y = std::array::from_fn::<_, 33, _>(|i| i as i16 * 2 - 31);
        for len in 0..=b.len() {
            let mut expected = [0i32; 33];
            let mut actual = [0i32; 33];
            fill_ytob_row_scalar(&mut expected[..len], &b[..len], &y[..len], 0.5);
            unsafe {
                fill_ytob_row_avx2(&mut actual[..len], &b[..len], &y[..len], 0.5);
            }
            assert_eq!(&actual[..len], &expected[..len], "length {len}");
        }
    }

    #[test]
    fn accumulate_ytob_weights_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let rb = std::array::from_fn::<_, 33, _>(|i| (i as i32 - 16) * 173);
        let ry = std::array::from_fn::<_, 33, _>(|i| {
            if i.is_multiple_of(7) {
                0
            } else {
                (i as i32 - 15) * 11
            }
        });
        for &step in &[0.5f32, 1.0 / 168.0, 0.125] {
            for len in 0..=rb.len() {
                let mut expected = [0u64; 255];
                let mut actual = [0u64; 255];
                accumulate_ytob_weights_scalar(&rb[..len], &ry[..len], step, &mut expected);
                unsafe {
                    accumulate_ytob_weights_avx2(&rb[..len], &ry[..len], step, &mut actual);
                }
                assert_eq!(actual, expected, "length {len}, step {step}");
            }
        }
    }

    #[test]
    fn fill_ytob_residuals_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let b_row =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 7919).wrapping_add(16381) as i16);
        let y_row =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 3253).wrapping_sub(22003) as i16);
        let b_up =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 1297).wrapping_sub(31111) as i16);
        let y_up =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 5347).wrapping_add(27011) as i16);
        for len in 0..=b_row.len() {
            let mut expected_b = [0i32; 35];
            let mut expected_y = [0i32; 35];
            let mut actual_b = [0i32; 35];
            let mut actual_y = [0i32; 35];
            fill_ytob_residuals_scalar(
                &mut expected_b[..len],
                &mut expected_y[..len],
                &b_row[..len],
                &y_row[..len],
                &b_up[..len],
                &y_up[..len],
            );
            unsafe {
                fill_ytob_residuals_avx2(
                    &mut actual_b[..len],
                    &mut actual_y[..len],
                    &b_row[..len],
                    &y_row[..len],
                    &b_up[..len],
                    &y_up[..len],
                );
            }
            assert_eq!(&actual_b[..len], &expected_b[..len], "B length {len}");
            assert_eq!(&actual_y[..len], &expected_y[..len], "Y length {len}");
        }
    }
}
