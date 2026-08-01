use crate::dct::{DctFn, DctInput};
use crate::sse::adaptive_quant::hsum;
use crate::structure_aq::{Features, finish_block_features};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<4>().0;
    let mut sums = [_mm_setzero_ps(); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        sums[i & 3] = _mm_add_ps(sums[i & 3], unsafe { _mm_loadu_ps(chunk.as_ptr()) });
    }
    let mean = hsum(_mm_add_ps(
        _mm_add_ps(sums[0], sums[1]),
        _mm_add_ps(sums[2], sums[3]),
    )) * (1.0 / 64.0);
    let mean_v = _mm_set1_ps(mean);
    let mut variances = [_mm_setzero_ps(); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        let d = _mm_sub_ps(unsafe { _mm_loadu_ps(chunk.as_ptr()) }, mean_v);
        variances[i & 3] = _mm_add_ps(variances[i & 3], _mm_mul_ps(d, d));
    }
    let variance = _mm_add_ps(
        _mm_add_ps(variances[0], variances[1]),
        _mm_add_ps(variances[2], variances[3]),
    );
    (mean, hsum(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tensor_chunk(left: __m128, right: __m128, top: __m128, bottom: __m128, sums: &mut [__m128; 3]) {
    let gx = _mm_mul_ps(_mm_sub_ps(right, left), _mm_set1_ps(0.5));
    let gy = _mm_mul_ps(_mm_sub_ps(bottom, top), _mm_set1_ps(0.5));
    sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(gx, gx));
    sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(gx, gy));
    sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(gy, gy));
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mut sums = [_mm_setzero_ps(); 3];
    for y in 1..7 {
        tensor_chunk(
            unsafe { _mm_loadu_ps(rows[y].as_ptr()) },
            unsafe { _mm_loadu_ps(rows[y].as_ptr().add(2)) },
            unsafe { _mm_loadu_ps(rows[y - 1].as_ptr().add(1)) },
            unsafe { _mm_loadu_ps(rows[y + 1].as_ptr().add(1)) },
            &mut sums,
        );
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        let mut top = [0.0; 4];
        let mut bottom = [0.0; 4];
        left[..2].copy_from_slice(&rows[y][4..6]);
        right[..2].copy_from_slice(&rows[y][6..8]);
        top[..2].copy_from_slice(&rows[y - 1][5..7]);
        bottom[..2].copy_from_slice(&rows[y + 1][5..7]);
        tensor_chunk(
            unsafe { _mm_loadu_ps(left.as_ptr()) },
            unsafe { _mm_loadu_ps(right.as_ptr()) },
            unsafe { _mm_loadu_ps(top.as_ptr()) },
            unsafe { _mm_loadu_ps(bottom.as_ptr()) },
            &mut sums,
        );
    }
    sums.map(|sum| hsum(sum))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn predictor_chunk(value: __m128, predictions: [__m128; 5], errors: &mut [__m128; 5]) {
    for (slot, prediction) in predictions.into_iter().enumerate() {
        let e = _mm_sub_ps(value, prediction);
        errors[slot] = _mm_add_ps(errors[slot], _mm_mul_ps(e, e));
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mut errors = [_mm_setzero_ps(); 5];
    for y in 1..7 {
        let left = unsafe { _mm_loadu_ps(rows[y].as_ptr()) };
        let value = unsafe { _mm_loadu_ps(rows[y].as_ptr().add(1)) };
        let top = unsafe { _mm_loadu_ps(rows[y - 1].as_ptr().add(1)) };
        let top_left = unsafe { _mm_loadu_ps(rows[y - 1].as_ptr()) };
        let bottom_left = unsafe { _mm_loadu_ps(rows[y + 1].as_ptr()) };
        predictor_chunk(
            value,
            [
                left,
                top,
                top_left,
                bottom_left,
                _mm_sub_ps(_mm_add_ps(left, top), top_left),
            ],
            &mut errors,
        );
        let mut left = [0.0; 4];
        let mut value = [0.0; 4];
        let mut top = [0.0; 4];
        let mut top_left = [0.0; 4];
        let mut bottom_left = [0.0; 4];
        left[..2].copy_from_slice(&rows[y][4..6]);
        value[..2].copy_from_slice(&rows[y][5..7]);
        top[..2].copy_from_slice(&rows[y - 1][5..7]);
        top_left[..2].copy_from_slice(&rows[y - 1][4..6]);
        bottom_left[..2].copy_from_slice(&rows[y + 1][4..6]);
        let left = unsafe { _mm_loadu_ps(left.as_ptr()) };
        let value = unsafe { _mm_loadu_ps(value.as_ptr()) };
        let top = unsafe { _mm_loadu_ps(top.as_ptr()) };
        let top_left = unsafe { _mm_loadu_ps(top_left.as_ptr()) };
        let bottom_left = unsafe { _mm_loadu_ps(bottom_left.as_ptr()) };
        predictor_chunk(
            value,
            [
                left,
                top,
                top_left,
                bottom_left,
                _mm_sub_ps(_mm_add_ps(left, top), top_left),
            ],
            &mut errors,
        );
    }
    errors.map(|error| hsum(error))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn squared_difference_sum(left: &[f32], right: &[f32]) -> f32 {
    let (left4, left_tail) = left.as_chunks::<4>();
    let (right4, right_tail) = right.as_chunks::<4>();
    let mut sum = _mm_setzero_ps();
    for (left, right) in left4.iter().zip(right4) {
        let d = _mm_sub_ps(unsafe { _mm_loadu_ps(right.as_ptr()) }, unsafe {
            _mm_loadu_ps(left.as_ptr())
        });
        sum = _mm_add_ps(sum, _mm_mul_ps(d, d));
    }
    if !left_tail.is_empty() {
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        left[..left_tail.len()].copy_from_slice(left_tail);
        right[..right_tail.len()].copy_from_slice(right_tail);
        let d = _mm_sub_ps(unsafe { _mm_loadu_ps(right.as_ptr()) }, unsafe {
            _mm_loadu_ps(left.as_ptr())
        });
        sum = _mm_add_ps(sum, _mm_mul_ps(d, d));
    }
    hsum(sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn gradient_energy(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let mut sums = [0.0; 4];
    for y in 0..height {
        let row = &values[y * stride..][..width];
        sums[y & 3] += squared_difference_sum(&row[..width - 1], &row[1..]);
    }
    for y in 0..height - 1 {
        sums[y & 3] += squared_difference_sum(
            &values[y * stride..][..width],
            &values[(y + 1) * stride..][..width],
        );
    }
    sums.into_iter().sum()
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn downsample_8x8(block: &[f32; 64]) -> [f32; 16] {
    let rows = block.as_chunks::<8>().0;
    let mut half = [0.0; 16];
    for y in 0..4 {
        let top0 = unsafe { _mm_loadu_ps(rows[2 * y].as_ptr()) };
        let top1 = unsafe { _mm_loadu_ps(rows[2 * y].as_ptr().add(4)) };
        let bottom0 = unsafe { _mm_loadu_ps(rows[2 * y + 1].as_ptr()) };
        let bottom1 = unsafe { _mm_loadu_ps(rows[2 * y + 1].as_ptr().add(4)) };
        let value = _mm_mul_ps(
            _mm_add_ps(_mm_hadd_ps(top0, top1), _mm_hadd_ps(bottom0, bottom1)),
            _mm_set1_ps(0.25),
        );
        unsafe { _mm_storeu_ps(half[y * 4..].as_mut_ptr(), value) };
    }
    half
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn energy_sum(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut sum = _mm_setzero_ps();
    for chunk in chunks {
        let value = unsafe { _mm_loadu_ps(chunk.as_ptr()) };
        sum = _mm_add_ps(sum, _mm_mul_ps(value, value));
    }
    if !tail.is_empty() {
        let mut padded = [0.0; 4];
        padded[..tail.len()].copy_from_slice(tail);
        let value = unsafe { _mm_loadu_ps(padded.as_ptr()) };
        sum = _mm_add_ps(sum, _mm_mul_ps(value, value));
    }
    hsum(sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
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
/// The caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn block_features_sse41(block: &[f32; 64], dct8x8: &DctFn<8, 8, 64>) -> Features {
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
