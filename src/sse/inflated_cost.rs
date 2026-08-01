/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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

#[inline]
#[target_feature(enable = "sse4.1")]
fn accumulate_gradient_vectors_x4(left: __m128, right: __m128, sum: __m128) -> __m128 {
    let difference = _mm_sub_ps(right, left);
    _mm_add_ps(sum, _mm_mul_ps(difference, difference))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn accumulate_gradient_x4(left: &[f32; 4], right: &[f32; 4], sum: __m128) -> __m128 {
    let left = unsafe { _mm_loadu_ps(left.as_ptr()) };
    let right = unsafe { _mm_loadu_ps(right.as_ptr()) };
    accumulate_gradient_vectors_x4(left, right, sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn horizontal_sum_x4(value: __m128) -> f32 {
    let sum = _mm_hadd_ps(value, value);
    _mm_cvtss_f32(_mm_hadd_ps(sum, sum))
}

/// # Safety
/// The caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn error_gradient_energy_sse41(error: &[f32], width: usize, height: usize) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("gradient plane size overflow");
    assert!(error.len() >= n);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let rows = error[..n].chunks_exact(width);
    let mut sum = _mm_setzero_ps();

    for row in rows.clone() {
        let (row4, row_tail) = row.as_chunks::<4>();
        if row_tail.is_empty() {
            for (chunk, left) in row4.iter().enumerate() {
                let left = unsafe { _mm_loadu_ps(left.as_ptr()) };
                let right = if chunk + 1 == row4.len() {
                    _mm_shuffle_ps::<249>(left, left)
                } else {
                    unsafe { _mm_loadu_ps(row.as_ptr().add(chunk * 4 + 1)) }
                };
                sum = accumulate_gradient_vectors_x4(left, right, sum);
            }
            continue;
        }
        let (left4, tail) = row[..width - 1].as_chunks::<4>();
        for (chunk, left) in left4.iter().enumerate() {
            let right = row[chunk * 4 + 1..].first_chunk::<4>().unwrap();
            sum = accumulate_gradient_x4(left, right, sum);
        }
        if !tail.is_empty() {
            let offset = left4.len() * 4;
            let mut left = [0.0; 4];
            let mut right = [0.0; 4];
            left[..tail.len()].copy_from_slice(tail);
            right[..tail.len()].copy_from_slice(&row[offset + 1..width]);
            sum = accumulate_gradient_x4(&left, &right, sum);
        }
    }

    for (top, bottom) in rows.clone().zip(rows.skip(1)) {
        let (top4, top_tail) = top.as_chunks::<4>();
        let (bottom4, bottom_tail) = bottom.as_chunks::<4>();
        for (top, bottom) in top4.iter().zip(bottom4) {
            sum = accumulate_gradient_x4(top, bottom, sum);
        }
        if !top_tail.is_empty() {
            let mut top = [0.0; 4];
            let mut bottom = [0.0; 4];
            top[..top_tail.len()].copy_from_slice(top_tail);
            bottom[..bottom_tail.len()].copy_from_slice(bottom_tail);
            sum = accumulate_gradient_x4(&top, &bottom, sum);
        }
    }
    horizontal_sum_x4(sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn combine_error_x4(spatial: &[f32; 4], luma: &[f32; 4], factor: __m128, combined: &mut [f32; 4]) {
    let spatial = unsafe { _mm_loadu_ps(spatial.as_ptr()) };
    let luma = unsafe { _mm_loadu_ps(luma.as_ptr()) };
    let value = _mm_add_ps(_mm_mul_ps(factor, luma), spatial);
    unsafe { _mm_storeu_ps(combined.as_mut_ptr(), value) };
}

/// # Safety
/// The caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn combine_error_sse41(
    spatial: &[f32],
    luma: &[f32],
    factor: f32,
    combined: &mut [f32],
) {
    debug_assert_eq!(spatial.len(), luma.len());
    debug_assert_eq!(spatial.len(), combined.len());
    let (spatial16, spatial_tail) = spatial.as_chunks::<16>();
    let (luma16, luma_tail) = luma.as_chunks::<16>();
    let (combined16, combined_tail) = combined.as_chunks_mut::<16>();
    let factor = _mm_set1_ps(factor);

    for ((spatial, luma), combined) in spatial16.iter().zip(luma16).zip(combined16) {
        let [s0, s1, s2, s3] = spatial.as_chunks::<4>().0 else {
            unreachable!()
        };
        let [l0, l1, l2, l3] = luma.as_chunks::<4>().0 else {
            unreachable!()
        };
        let [c0, c1, c2, c3] = combined.as_chunks_mut::<4>().0 else {
            unreachable!()
        };
        combine_error_x4(s0, l0, factor, c0);
        combine_error_x4(s1, l1, factor, c1);
        combine_error_x4(s2, l2, factor, c2);
        combine_error_x4(s3, l3, factor, c3);
    }

    let (spatial4, spatial_remainder) = spatial_tail.as_chunks::<4>();
    let (luma4, luma_remainder) = luma_tail.as_chunks::<4>();
    let (combined4, combined_remainder) = combined_tail.as_chunks_mut::<4>();
    for ((spatial, luma), combined) in spatial4.iter().zip(luma4).zip(combined4) {
        combine_error_x4(spatial, luma, factor, combined);
    }
    if !spatial_remainder.is_empty() {
        let mut spatial = [0.0; 4];
        let mut luma = [0.0; 4];
        let mut combined = [0.0; 4];
        spatial[..spatial_remainder.len()].copy_from_slice(spatial_remainder);
        luma[..luma_remainder.len()].copy_from_slice(luma_remainder);
        combine_error_x4(&spatial, &luma, factor, &mut combined);
        combined_remainder.copy_from_slice(&combined[..combined_remainder.len()]);
    }
}
