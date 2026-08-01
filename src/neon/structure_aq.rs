use crate::dct::{DctFn, DctInput};
use crate::structure_aq::{Features, finish_block_features};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<4>().0;
    let mut sums = [vdupq_n_f32(0.0); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        sums[i & 3] = vaddq_f32(sums[i & 3], unsafe { vld1q_f32(chunk.as_ptr()) });
    }
    let mean = vaddvq_f32(vaddq_f32(
        vaddq_f32(sums[0], sums[1]),
        vaddq_f32(sums[2], sums[3]),
    )) * (1.0 / 64.0);
    let mean_v = vdupq_n_f32(mean);
    let mut variances = [vdupq_n_f32(0.0); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        let d = vsubq_f32(unsafe { vld1q_f32(chunk.as_ptr()) }, mean_v);
        variances[i & 3] = vfmaq_f32(variances[i & 3], d, d);
    }
    let variance = vaddq_f32(
        vaddq_f32(variances[0], variances[1]),
        vaddq_f32(variances[2], variances[3]),
    );
    (mean, vaddvq_f32(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "neon")]
fn tensor_chunk(
    left: float32x4_t,
    right: float32x4_t,
    top: float32x4_t,
    bottom: float32x4_t,
    sums: &mut [float32x4_t; 3],
) {
    let gx = vmulq_n_f32(vsubq_f32(right, left), 0.5);
    let gy = vmulq_n_f32(vsubq_f32(bottom, top), 0.5);
    sums[0] = vfmaq_f32(sums[0], gx, gx);
    sums[1] = vfmaq_f32(sums[1], gx, gy);
    sums[2] = vfmaq_f32(sums[2], gy, gy);
}

#[inline]
#[target_feature(enable = "neon")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mut sums = [vdupq_n_f32(0.0); 3];
    for y in 1..7 {
        tensor_chunk(
            unsafe { vld1q_f32(rows[y].as_ptr()) },
            unsafe { vld1q_f32(rows[y].as_ptr().add(2)) },
            unsafe { vld1q_f32(rows[y - 1].as_ptr().add(1)) },
            unsafe { vld1q_f32(rows[y + 1].as_ptr().add(1)) },
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
            unsafe { vld1q_f32(left.as_ptr()) },
            unsafe { vld1q_f32(right.as_ptr()) },
            unsafe { vld1q_f32(top.as_ptr()) },
            unsafe { vld1q_f32(bottom.as_ptr()) },
            &mut sums,
        );
    }
    sums.map(|sum| vaddvq_f32(sum))
}

#[inline]
#[target_feature(enable = "neon")]
fn predictor_chunk(
    value: float32x4_t,
    predictions: [float32x4_t; 5],
    errors: &mut [float32x4_t; 5],
) {
    for (slot, prediction) in predictions.into_iter().enumerate() {
        let e = vsubq_f32(value, prediction);
        errors[slot] = vfmaq_f32(errors[slot], e, e);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mut errors = [vdupq_n_f32(0.0); 5];
    for y in 1..7 {
        let left = unsafe { vld1q_f32(rows[y].as_ptr()) };
        let value = unsafe { vld1q_f32(rows[y].as_ptr().add(1)) };
        let top = unsafe { vld1q_f32(rows[y - 1].as_ptr().add(1)) };
        let top_left = unsafe { vld1q_f32(rows[y - 1].as_ptr()) };
        let bottom_left = unsafe { vld1q_f32(rows[y + 1].as_ptr()) };
        predictor_chunk(
            value,
            [
                left,
                top,
                top_left,
                bottom_left,
                vsubq_f32(vaddq_f32(left, top), top_left),
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
        let left = unsafe { vld1q_f32(left.as_ptr()) };
        let value = unsafe { vld1q_f32(value.as_ptr()) };
        let top = unsafe { vld1q_f32(top.as_ptr()) };
        let top_left = unsafe { vld1q_f32(top_left.as_ptr()) };
        let bottom_left = unsafe { vld1q_f32(bottom_left.as_ptr()) };
        predictor_chunk(
            value,
            [
                left,
                top,
                top_left,
                bottom_left,
                vsubq_f32(vaddq_f32(left, top), top_left),
            ],
            &mut errors,
        );
    }
    errors.map(|error| vaddvq_f32(error))
}

#[inline]
#[target_feature(enable = "neon")]
fn squared_difference_sum(left: &[f32], right: &[f32]) -> f32 {
    let (left4, left_tail) = left.as_chunks::<4>();
    let (right4, right_tail) = right.as_chunks::<4>();
    let mut sum = vdupq_n_f32(0.0);
    for (left, right) in left4.iter().zip(right4) {
        let d = vsubq_f32(unsafe { vld1q_f32(right.as_ptr()) }, unsafe {
            vld1q_f32(left.as_ptr())
        });
        sum = vfmaq_f32(sum, d, d);
    }
    if !left_tail.is_empty() {
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        left[..left_tail.len()].copy_from_slice(left_tail);
        right[..right_tail.len()].copy_from_slice(right_tail);
        let d = vsubq_f32(unsafe { vld1q_f32(right.as_ptr()) }, unsafe {
            vld1q_f32(left.as_ptr())
        });
        sum = vfmaq_f32(sum, d, d);
    }
    vaddvq_f32(sum)
}

#[inline]
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
fn downsample_8x8(block: &[f32; 64]) -> [f32; 16] {
    let rows = block.as_chunks::<8>().0;
    let mut half = [0.0; 16];
    for y in 0..4 {
        let top0 = unsafe { vld1q_f32(rows[2 * y].as_ptr()) };
        let top1 = unsafe { vld1q_f32(rows[2 * y].as_ptr().add(4)) };
        let bottom0 = unsafe { vld1q_f32(rows[2 * y + 1].as_ptr()) };
        let bottom1 = unsafe { vld1q_f32(rows[2 * y + 1].as_ptr().add(4)) };
        let value = vmulq_n_f32(
            vaddq_f32(vpaddq_f32(top0, top1), vpaddq_f32(bottom0, bottom1)),
            0.25,
        );
        unsafe { vst1q_f32(half[y * 4..].as_mut_ptr(), value) };
    }
    half
}

#[inline]
#[target_feature(enable = "neon")]
fn energy_sum(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut sum = vdupq_n_f32(0.0);
    for chunk in chunks {
        let value = unsafe { vld1q_f32(chunk.as_ptr()) };
        sum = vfmaq_f32(sum, value, value);
    }
    if !tail.is_empty() {
        let mut padded = [0.0; 4];
        padded[..tail.len()].copy_from_slice(tail);
        let value = unsafe { vld1q_f32(padded.as_ptr()) };
        sum = vfmaq_f32(sum, value, value);
    }
    vaddvq_f32(sum)
}

#[inline]
#[target_feature(enable = "neon")]
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
/// AArch64 NEON must be available.
#[target_feature(enable = "neon")]
pub(crate) fn block_features_neon(block: &[f32; 64], dct8x8: &DctFn<8, 8, 64>) -> Features {
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
