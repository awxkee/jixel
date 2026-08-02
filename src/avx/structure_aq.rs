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
    _mm256_setr_epi32(
        if n > 0 { -1 } else { 0 },
        if n > 1 { -1 } else { 0 },
        if n > 2 { -1 } else { 0 },
        if n > 3 { -1 } else { 0 },
        if n > 4 { -1 } else { 0 },
        if n > 5 { -1 } else { 0 },
        if n > 6 { -1 } else { 0 },
        if n > 7 { -1 } else { 0 },
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<8>().0;
    let mut sums = [_mm256_setzero_ps(); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        sums[i & 3] = _mm256_add_ps(sums[i & 3], unsafe { _mm256_loadu_ps(chunk.as_ptr()) });
    }
    let sum = _mm256_add_ps(
        _mm256_add_ps(sums[0], sums[1]),
        _mm256_add_ps(sums[2], sums[3]),
    );
    let mean = hsum256(sum) * (1.0 / 64.0);
    let mean_v = _mm256_set1_ps(mean);
    let mut variances = [_mm256_setzero_ps(); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        let value = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        let d = _mm256_sub_ps(value, mean_v);
        variances[i & 3] = _mm256_fmadd_ps(d, d, variances[i & 3]);
    }
    let variance = _mm256_add_ps(
        _mm256_add_ps(variances[0], variances[1]),
        _mm256_add_ps(variances[2], variances[3]),
    );
    (mean, hsum256(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mask = first_lanes_mask(6);
    let mut jxx = [_mm256_setzero_ps(); 2];
    let mut jxy = [_mm256_setzero_ps(); 2];
    let mut jyy = [_mm256_setzero_ps(); 2];
    for y in 1..7 {
        let lane = y & 1;
        let left = unsafe { _mm256_maskload_ps(rows[y].as_ptr(), mask) };
        let right = unsafe { _mm256_maskload_ps(rows[y].as_ptr().add(2), mask) };
        let top = unsafe { _mm256_maskload_ps(rows[y - 1].as_ptr().add(1), mask) };
        let bottom = unsafe { _mm256_maskload_ps(rows[y + 1].as_ptr().add(1), mask) };
        let gx = _mm256_mul_ps(_mm256_sub_ps(right, left), _mm256_set1_ps(0.5));
        let gy = _mm256_mul_ps(_mm256_sub_ps(bottom, top), _mm256_set1_ps(0.5));
        jxx[lane] = _mm256_fmadd_ps(gx, gx, jxx[lane]);
        jxy[lane] = _mm256_fmadd_ps(gx, gy, jxy[lane]);
        jyy[lane] = _mm256_fmadd_ps(gy, gy, jyy[lane]);
    }
    [
        hsum256(_mm256_add_ps(jxx[0], jxx[1])),
        hsum256(_mm256_add_ps(jxy[0], jxy[1])),
        hsum256(_mm256_add_ps(jyy[0], jyy[1])),
    ]
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mask = first_lanes_mask(6);
    let mut errors = [_mm256_setzero_ps(); 5];
    for y in 1..7 {
        let left = unsafe { _mm256_maskload_ps(rows[y].as_ptr(), mask) };
        let value = unsafe { _mm256_maskload_ps(rows[y].as_ptr().add(1), mask) };
        let top = unsafe { _mm256_maskload_ps(rows[y - 1].as_ptr().add(1), mask) };
        let top_left = unsafe { _mm256_maskload_ps(rows[y - 1].as_ptr(), mask) };
        let bottom_left = unsafe { _mm256_maskload_ps(rows[y + 1].as_ptr(), mask) };
        let gradient = _mm256_sub_ps(_mm256_add_ps(left, top), top_left);
        for (slot, prediction) in [left, top, top_left, bottom_left, gradient]
            .into_iter()
            .enumerate()
        {
            let e = _mm256_sub_ps(value, prediction);
            errors[slot] = _mm256_fmadd_ps(e, e, errors[slot]);
        }
    }
    errors.map(|error| hsum256(error))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn gradient_energy(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let horizontal_mask = first_lanes_mask(width - 1);
    let vertical_mask = first_lanes_mask(width);
    let mut sums = [_mm256_setzero_ps(); 4];
    for y in 0..height {
        let row = &values[y * stride..];
        let left = unsafe { _mm256_maskload_ps(row.as_ptr(), horizontal_mask) };
        let right = unsafe { _mm256_maskload_ps(row.as_ptr().add(1), horizontal_mask) };
        let d = _mm256_sub_ps(right, left);
        sums[y & 3] = _mm256_fmadd_ps(d, d, sums[y & 3]);
    }
    for y in 0..height - 1 {
        let top = unsafe { _mm256_maskload_ps(values[y * stride..].as_ptr(), vertical_mask) };
        let bottom =
            unsafe { _mm256_maskload_ps(values[(y + 1) * stride..].as_ptr(), vertical_mask) };
        let d = _mm256_sub_ps(bottom, top);
        sums[y & 3] = _mm256_fmadd_ps(d, d, sums[y & 3]);
    }
    hsum256(_mm256_add_ps(
        _mm256_add_ps(sums[0], sums[1]),
        _mm256_add_ps(sums[2], sums[3]),
    ))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn downsample_8x8(block: &[f32; 64]) -> [f32; 16] {
    let rows = block.as_chunks::<8>().0;
    let mut half = [0.0; 16];
    for y in 0..4 {
        let top = unsafe { _mm256_loadu_ps(rows[2 * y].as_ptr()) };
        let bottom = unsafe { _mm256_loadu_ps(rows[2 * y + 1].as_ptr()) };
        let pairs = _mm256_hadd_ps(top, bottom);
        let top_pairs = _mm256_permutevar8x32_ps(pairs, _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0));
        let bottom_pairs =
            _mm256_permutevar8x32_ps(pairs, _mm256_setr_epi32(2, 3, 6, 7, 0, 0, 0, 0));
        let value = _mm256_mul_ps(_mm256_add_ps(top_pairs, bottom_pairs), _mm256_set1_ps(0.25));
        unsafe { _mm_storeu_ps(half[y * 4..].as_mut_ptr(), _mm256_castps256_ps128(value)) };
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
