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
use crate::avx::ac_strategy::hsum256;
use crate::dct::{DctFn, DctInput};
use crate::structure_aq::{Features, finish_block_features};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn first_lanes_mask(n: usize) -> __m256i {
    _mm256_cmpgt_epi32(
        _mm256_set1_epi32(n as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<8>().0;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    for group in 0..2 {
        let chunk = group * 4;
        sum0 = _mm256_add_ps(sum0, unsafe { _mm256_loadu_ps(chunks[chunk].as_ptr()) });
        sum1 = _mm256_add_ps(sum1, unsafe { _mm256_loadu_ps(chunks[chunk + 1].as_ptr()) });
        sum2 = _mm256_add_ps(sum2, unsafe { _mm256_loadu_ps(chunks[chunk + 2].as_ptr()) });
        sum3 = _mm256_add_ps(sum3, unsafe { _mm256_loadu_ps(chunks[chunk + 3].as_ptr()) });
    }
    let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let mean = hsum256(sum) * (1.0 / 64.0);
    let mean_v = _mm256_set1_ps(mean);
    let mut variance0 = _mm256_setzero_ps();
    let mut variance1 = _mm256_setzero_ps();
    let mut variance2 = _mm256_setzero_ps();
    let mut variance3 = _mm256_setzero_ps();
    for group in 0..2 {
        let chunk = group * 4;
        let d0 = _mm256_sub_ps(unsafe { _mm256_loadu_ps(chunks[chunk].as_ptr()) }, mean_v);
        let d1 = _mm256_sub_ps(
            unsafe { _mm256_loadu_ps(chunks[chunk + 1].as_ptr()) },
            mean_v,
        );
        let d2 = _mm256_sub_ps(
            unsafe { _mm256_loadu_ps(chunks[chunk + 2].as_ptr()) },
            mean_v,
        );
        let d3 = _mm256_sub_ps(
            unsafe { _mm256_loadu_ps(chunks[chunk + 3].as_ptr()) },
            mean_v,
        );
        variance0 = _mm256_fmadd_ps(d0, d0, variance0);
        variance1 = _mm256_fmadd_ps(d1, d1, variance1);
        variance2 = _mm256_fmadd_ps(d2, d2, variance2);
        variance3 = _mm256_fmadd_ps(d3, d3, variance3);
    }
    let variance = _mm256_add_ps(
        _mm256_add_ps(variance0, variance1),
        _mm256_add_ps(variance2, variance3),
    );
    (mean, hsum256(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn tensor_gradients(rows: &[[f32; 8]], y: usize, mask: __m256i) -> (__m256, __m256) {
    let left = unsafe { _mm256_maskload_ps(rows[y].as_ptr(), mask) };
    let right = unsafe { _mm256_maskload_ps(rows[y].as_ptr().add(2), mask) };
    let top = unsafe { _mm256_maskload_ps(rows[y - 1].as_ptr().add(1), mask) };
    let bottom = unsafe { _mm256_maskload_ps(rows[y + 1].as_ptr().add(1), mask) };
    (
        _mm256_mul_ps(_mm256_sub_ps(right, left), _mm256_set1_ps(0.5)),
        _mm256_mul_ps(_mm256_sub_ps(bottom, top), _mm256_set1_ps(0.5)),
    )
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mask = first_lanes_mask(6);
    let mut jxx0 = _mm256_setzero_ps();
    let mut jxy0 = _mm256_setzero_ps();
    let mut jyy0 = _mm256_setzero_ps();
    let mut jxx1 = _mm256_setzero_ps();
    let mut jxy1 = _mm256_setzero_ps();
    let mut jyy1 = _mm256_setzero_ps();
    for y in (1..7).step_by(2) {
        let (gx0, gy0) = tensor_gradients(rows, y, mask);
        jxx0 = _mm256_fmadd_ps(gx0, gx0, jxx0);
        jxy0 = _mm256_fmadd_ps(gx0, gy0, jxy0);
        jyy0 = _mm256_fmadd_ps(gy0, gy0, jyy0);
        let (gx1, gy1) = tensor_gradients(rows, y + 1, mask);
        jxx1 = _mm256_fmadd_ps(gx1, gx1, jxx1);
        jxy1 = _mm256_fmadd_ps(gx1, gy1, jxy1);
        jyy1 = _mm256_fmadd_ps(gy1, gy1, jyy1);
    }
    [
        hsum256(_mm256_add_ps(jxx0, jxx1)),
        hsum256(_mm256_add_ps(jxy0, jxy1)),
        hsum256(_mm256_add_ps(jyy0, jyy1)),
    ]
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mask = first_lanes_mask(6);
    let mut error0 = _mm256_setzero_ps();
    let mut error1 = _mm256_setzero_ps();
    let mut error2 = _mm256_setzero_ps();
    let mut error3 = _mm256_setzero_ps();
    let mut error4 = _mm256_setzero_ps();
    for y in 1..7 {
        let left = unsafe { _mm256_maskload_ps(rows[y].as_ptr(), mask) };
        let value = unsafe { _mm256_maskload_ps(rows[y].as_ptr().add(1), mask) };
        let top = unsafe { _mm256_maskload_ps(rows[y - 1].as_ptr().add(1), mask) };
        let top_left = unsafe { _mm256_maskload_ps(rows[y - 1].as_ptr(), mask) };
        let bottom_left = unsafe { _mm256_maskload_ps(rows[y + 1].as_ptr(), mask) };
        let gradient = _mm256_sub_ps(_mm256_add_ps(left, top), top_left);
        let e0 = _mm256_sub_ps(value, left);
        let e1 = _mm256_sub_ps(value, top);
        let e2 = _mm256_sub_ps(value, top_left);
        let e3 = _mm256_sub_ps(value, bottom_left);
        let e4 = _mm256_sub_ps(value, gradient);
        error0 = _mm256_fmadd_ps(e0, e0, error0);
        error1 = _mm256_fmadd_ps(e1, e1, error1);
        error2 = _mm256_fmadd_ps(e2, e2, error2);
        error3 = _mm256_fmadd_ps(e3, e3, error3);
        error4 = _mm256_fmadd_ps(e4, e4, error4);
    }
    [
        hsum256(error0),
        hsum256(error1),
        hsum256(error2),
        hsum256(error3),
        hsum256(error4),
    ]
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn gradient_energy(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let horizontal_mask = first_lanes_mask(width - 1);
    let vertical_mask = first_lanes_mask(width);
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    for base in (0..height).step_by(4) {
        let row0 = &values[base * stride..];
        let row1 = &values[(base + 1) * stride..];
        let row2 = &values[(base + 2) * stride..];
        let row3 = &values[(base + 3) * stride..];
        let d0 = _mm256_sub_ps(
            unsafe { _mm256_maskload_ps(row0.as_ptr().add(1), horizontal_mask) },
            unsafe { _mm256_maskload_ps(row0.as_ptr(), horizontal_mask) },
        );
        let d1 = _mm256_sub_ps(
            unsafe { _mm256_maskload_ps(row1.as_ptr().add(1), horizontal_mask) },
            unsafe { _mm256_maskload_ps(row1.as_ptr(), horizontal_mask) },
        );
        let d2 = _mm256_sub_ps(
            unsafe { _mm256_maskload_ps(row2.as_ptr().add(1), horizontal_mask) },
            unsafe { _mm256_maskload_ps(row2.as_ptr(), horizontal_mask) },
        );
        let d3 = _mm256_sub_ps(
            unsafe { _mm256_maskload_ps(row3.as_ptr().add(1), horizontal_mask) },
            unsafe { _mm256_maskload_ps(row3.as_ptr(), horizontal_mask) },
        );
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
    }
    let vertical_rows = height - 1;
    for base in (0..vertical_rows).step_by(4) {
        let d0 = _mm256_sub_ps(
            unsafe { _mm256_maskload_ps(values[(base + 1) * stride..].as_ptr(), vertical_mask) },
            unsafe { _mm256_maskload_ps(values[base * stride..].as_ptr(), vertical_mask) },
        );
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        if base + 1 < vertical_rows {
            let d1 = _mm256_sub_ps(
                unsafe {
                    _mm256_maskload_ps(values[(base + 2) * stride..].as_ptr(), vertical_mask)
                },
                unsafe {
                    _mm256_maskload_ps(values[(base + 1) * stride..].as_ptr(), vertical_mask)
                },
            );
            sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        }
        if base + 2 < vertical_rows {
            let d2 = _mm256_sub_ps(
                unsafe {
                    _mm256_maskload_ps(values[(base + 3) * stride..].as_ptr(), vertical_mask)
                },
                unsafe {
                    _mm256_maskload_ps(values[(base + 2) * stride..].as_ptr(), vertical_mask)
                },
            );
            sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        }
        if base + 3 < vertical_rows {
            let d3 = _mm256_sub_ps(
                unsafe {
                    _mm256_maskload_ps(values[(base + 4) * stride..].as_ptr(), vertical_mask)
                },
                unsafe {
                    _mm256_maskload_ps(values[(base + 3) * stride..].as_ptr(), vertical_mask)
                },
            );
            sum3 = _mm256_fmadd_ps(d3, d3, sum3);
        }
    }
    hsum256(_mm256_add_ps(
        _mm256_add_ps(sum0, sum1),
        _mm256_add_ps(sum2, sum3),
    ))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn downsample_8x8(block: &[f32; 64]) -> [f32; 16] {
    let rows = block.as_chunks::<8>().0;
    let mut half = [0.0; 16];
    for (y, dst) in half.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let top = unsafe { _mm256_loadu_ps(rows[2 * y].as_ptr()) };
        let bottom = unsafe { _mm256_loadu_ps(rows[2 * y + 1].as_ptr()) };
        let pairs = _mm256_hadd_ps(top, bottom);
        let top_pairs = _mm256_permutevar8x32_ps(pairs, _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0));
        let bottom_pairs =
            _mm256_permutevar8x32_ps(pairs, _mm256_setr_epi32(2, 3, 6, 7, 0, 0, 0, 0));
        let value = _mm256_mul_ps(_mm256_add_ps(top_pairs, bottom_pairs), _mm256_set1_ps(0.25));
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), _mm256_castps256_ps128(value)) };
    }
    half
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn energy_sum(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let value = unsafe { _mm256_maskload_ps(values.as_ptr(), first_lanes_mask(values.len())) };
    hsum256(_mm256_mul_ps(value, value))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn spectral_energy(coeffs: &[f32; 64]) -> (f32, f32) {
    let mut mid = 0.0;
    let mut high = 0.0;
    for (y, row) in coeffs.as_chunks::<8>().0.iter().enumerate() {
        let mid_start = 2usize.saturating_sub(y);
        let mid_end = (8 - y).min(8);
        mid += energy_sum(&row[mid_start..mid_end]);
        high += energy_sum(&row[8usize.saturating_sub(y)..]);
    }
    (mid, high)
}

/// # Safety
/// The caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn block_features_avx2(block: &[f32; 64], dct8x8: &DctFn<8, 8, 64>) -> Features {
    let (_, variance) = moments(block);
    let tensor = tensor(block);
    let errors = predictor_errors(block);
    let energy_1x = gradient_energy(block, 8, 8, 8);
    let half = downsample_8x8(block);
    let energy_2x = gradient_energy(&half, 4, 4, 4);
    let mut coeffs = [0.0; 64];
    dct8x8(DctInput::from_flat(block), &mut coeffs);
    let (mid, high) = spectral_energy(&coeffs);
    finish_block_features(variance, tensor, errors, energy_1x, energy_2x, mid, high)
}
