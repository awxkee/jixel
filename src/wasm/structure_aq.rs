use crate::dct::{DctFn, DctInput};
use crate::structure_aq::{Features, finish_block_features};
use std::arch::wasm32::*;

#[inline]
fn horizontal_sum_x4(value: v128) -> f32 {
    f32x4_extract_lane::<0>(value)
        + f32x4_extract_lane::<1>(value)
        + f32x4_extract_lane::<2>(value)
        + f32x4_extract_lane::<3>(value)
}

#[inline]
#[target_feature(enable = "simd128")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<4>().0;
    let mut sums = [f32x4_splat(0.0); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        sums[i & 3] = f32x4_add(sums[i & 3], unsafe { v128_load(chunk.as_ptr().cast()) });
    }
    let mean = horizontal_sum_x4(f32x4_add(
        f32x4_add(sums[0], sums[1]),
        f32x4_add(sums[2], sums[3]),
    )) * (1.0 / 64.0);
    let mean_v = f32x4_splat(mean);
    let mut variances = [f32x4_splat(0.0); 4];
    for (i, chunk) in chunks.iter().enumerate() {
        let d = f32x4_sub(unsafe { v128_load(chunk.as_ptr().cast()) }, mean_v);
        variances[i & 3] = f32x4_add(variances[i & 3], f32x4_mul(d, d));
    }
    let variance = f32x4_add(
        f32x4_add(variances[0], variances[1]),
        f32x4_add(variances[2], variances[3]),
    );
    (mean, horizontal_sum_x4(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "simd128")]
fn tensor_chunk(left: v128, right: v128, top: v128, bottom: v128, sums: &mut [v128; 3]) {
    let gx = f32x4_mul(f32x4_sub(right, left), f32x4_splat(0.5));
    let gy = f32x4_mul(f32x4_sub(bottom, top), f32x4_splat(0.5));
    sums[0] = f32x4_add(sums[0], f32x4_mul(gx, gx));
    sums[1] = f32x4_add(sums[1], f32x4_mul(gx, gy));
    sums[2] = f32x4_add(sums[2], f32x4_mul(gy, gy));
}

#[inline]
#[target_feature(enable = "simd128")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mut sums = [f32x4_splat(0.0); 3];
    for y in 1..7 {
        tensor_chunk(
            unsafe { v128_load(rows[y].as_ptr().cast()) },
            unsafe { v128_load(rows[y].as_ptr().add(2).cast()) },
            unsafe { v128_load(rows[y - 1].as_ptr().add(1).cast()) },
            unsafe { v128_load(rows[y + 1].as_ptr().add(1).cast()) },
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
            unsafe { v128_load(left.as_ptr().cast()) },
            unsafe { v128_load(right.as_ptr().cast()) },
            unsafe { v128_load(top.as_ptr().cast()) },
            unsafe { v128_load(bottom.as_ptr().cast()) },
            &mut sums,
        );
    }
    sums.map(horizontal_sum_x4)
}

#[inline]
#[target_feature(enable = "simd128")]
fn predictor_chunk(value: v128, predictions: [v128; 5], errors: &mut [v128; 5]) {
    for (slot, prediction) in predictions.into_iter().enumerate() {
        let e = f32x4_sub(value, prediction);
        errors[slot] = f32x4_add(errors[slot], f32x4_mul(e, e));
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mut errors = [f32x4_splat(0.0); 5];
    for y in 1..7 {
        let left = unsafe { v128_load(rows[y].as_ptr().cast()) };
        let value = unsafe { v128_load(rows[y].as_ptr().add(1).cast()) };
        let top = unsafe { v128_load(rows[y - 1].as_ptr().add(1).cast()) };
        let top_left = unsafe { v128_load(rows[y - 1].as_ptr().cast()) };
        let bottom_left = unsafe { v128_load(rows[y + 1].as_ptr().cast()) };
        predictor_chunk(
            value,
            [
                left,
                top,
                top_left,
                bottom_left,
                f32x4_sub(f32x4_add(left, top), top_left),
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
        let left = unsafe { v128_load(left.as_ptr().cast()) };
        let value = unsafe { v128_load(value.as_ptr().cast()) };
        let top = unsafe { v128_load(top.as_ptr().cast()) };
        let top_left = unsafe { v128_load(top_left.as_ptr().cast()) };
        let bottom_left = unsafe { v128_load(bottom_left.as_ptr().cast()) };
        predictor_chunk(
            value,
            [
                left,
                top,
                top_left,
                bottom_left,
                f32x4_sub(f32x4_add(left, top), top_left),
            ],
            &mut errors,
        );
    }
    errors.map(horizontal_sum_x4)
}

#[inline]
#[target_feature(enable = "simd128")]
fn squared_difference_sum(left: &[f32], right: &[f32]) -> f32 {
    let (left4, left_tail) = left.as_chunks::<4>();
    let (right4, right_tail) = right.as_chunks::<4>();
    let mut sum = f32x4_splat(0.0);
    for (left, right) in left4.iter().zip(right4) {
        let d = f32x4_sub(unsafe { v128_load(right.as_ptr().cast()) }, unsafe {
            v128_load(left.as_ptr().cast())
        });
        sum = f32x4_add(sum, f32x4_mul(d, d));
    }
    if !left_tail.is_empty() {
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        left[..left_tail.len()].copy_from_slice(left_tail);
        right[..right_tail.len()].copy_from_slice(right_tail);
        let d = f32x4_sub(unsafe { v128_load(right.as_ptr().cast()) }, unsafe {
            v128_load(left.as_ptr().cast())
        });
        sum = f32x4_add(sum, f32x4_mul(d, d));
    }
    horizontal_sum_x4(sum)
}

#[inline]
#[target_feature(enable = "simd128")]
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
#[target_feature(enable = "simd128")]
fn downsample_8x8(block: &[f32; 64]) -> [f32; 16] {
    let rows = block.as_chunks::<8>().0;
    let mut half = [0.0; 16];
    for y in 0..4 {
        let top0 = unsafe { v128_load(rows[2 * y].as_ptr().cast()) };
        let top1 = unsafe { v128_load(rows[2 * y].as_ptr().add(4).cast()) };
        let bottom0 = unsafe { v128_load(rows[2 * y + 1].as_ptr().cast()) };
        let bottom1 = unsafe { v128_load(rows[2 * y + 1].as_ptr().add(4).cast()) };
        let top_pairs = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(top0, top1),
            i32x4_shuffle::<1, 3, 5, 7>(top0, top1),
        );
        let bottom_pairs = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(bottom0, bottom1),
            i32x4_shuffle::<1, 3, 5, 7>(bottom0, bottom1),
        );
        let value = f32x4_mul(f32x4_add(top_pairs, bottom_pairs), f32x4_splat(0.25));
        unsafe { v128_store(half[y * 4..].as_mut_ptr().cast(), value) };
    }
    half
}

#[inline]
#[target_feature(enable = "simd128")]
fn energy_sum(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut sum = f32x4_splat(0.0);
    for chunk in chunks {
        let value = unsafe { v128_load(chunk.as_ptr().cast()) };
        sum = f32x4_add(sum, f32x4_mul(value, value));
    }
    if !tail.is_empty() {
        let mut padded = [0.0; 4];
        padded[..tail.len()].copy_from_slice(tail);
        let value = unsafe { v128_load(padded.as_ptr().cast()) };
        sum = f32x4_add(sum, f32x4_mul(value, value));
    }
    horizontal_sum_x4(sum)
}

#[inline]
#[target_feature(enable = "simd128")]
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

#[target_feature(enable = "simd128")]
pub(crate) fn block_features_wasm(block: &[f32; 64], dct8x8: &DctFn<8, 8, 64>) -> Features {
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
