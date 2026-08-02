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

#[inline(never)]
#[target_feature(enable = "simd128")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<4>().0;
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);
    for group in 0..4 {
        let chunk = group * 4;
        sum0 = f32x4_add(sum0, unsafe { v128_load(chunks[chunk].as_ptr().cast()) });
        sum1 = f32x4_add(sum1, unsafe {
            v128_load(chunks[chunk + 1].as_ptr().cast())
        });
        sum2 = f32x4_add(sum2, unsafe {
            v128_load(chunks[chunk + 2].as_ptr().cast())
        });
        sum3 = f32x4_add(sum3, unsafe {
            v128_load(chunks[chunk + 3].as_ptr().cast())
        });
    }
    let mean =
        horizontal_sum_x4(f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3))) * (1.0 / 64.0);
    let mean_v = f32x4_splat(mean);
    let mut variance0 = f32x4_splat(0.0);
    let mut variance1 = f32x4_splat(0.0);
    let mut variance2 = f32x4_splat(0.0);
    let mut variance3 = f32x4_splat(0.0);
    for group in 0..4 {
        let chunk = group * 4;
        let d0 = f32x4_sub(unsafe { v128_load(chunks[chunk].as_ptr().cast()) }, mean_v);
        let d1 = f32x4_sub(
            unsafe { v128_load(chunks[chunk + 1].as_ptr().cast()) },
            mean_v,
        );
        let d2 = f32x4_sub(
            unsafe { v128_load(chunks[chunk + 2].as_ptr().cast()) },
            mean_v,
        );
        let d3 = f32x4_sub(
            unsafe { v128_load(chunks[chunk + 3].as_ptr().cast()) },
            mean_v,
        );
        variance0 = f32x4_add(variance0, f32x4_mul(d0, d0));
        variance1 = f32x4_add(variance1, f32x4_mul(d1, d1));
        variance2 = f32x4_add(variance2, f32x4_mul(d2, d2));
        variance3 = f32x4_add(variance3, f32x4_mul(d3, d3));
    }
    let variance = f32x4_add(
        f32x4_add(variance0, variance1),
        f32x4_add(variance2, variance3),
    );
    (mean, horizontal_sum_x4(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "simd128")]
fn load2(row: &[f32], x: usize) -> v128 {
    unsafe { v128_load64_zero(row.as_ptr().add(x).cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
fn tensor_gradients(left: v128, right: v128, top: v128, bottom: v128) -> (v128, v128) {
    (
        f32x4_mul(f32x4_sub(right, left), f32x4_splat(0.5)),
        f32x4_mul(f32x4_sub(bottom, top), f32x4_splat(0.5)),
    )
}

#[inline(never)]
#[target_feature(enable = "simd128")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mut jxx = f32x4_splat(0.0);
    let mut jxy = f32x4_splat(0.0);
    let mut jyy = f32x4_splat(0.0);
    for y in 1..7 {
        let (gx, gy) = tensor_gradients(
            unsafe { v128_load(rows[y].as_ptr().cast()) },
            unsafe { v128_load(rows[y].as_ptr().add(2).cast()) },
            unsafe { v128_load(rows[y - 1].as_ptr().add(1).cast()) },
            unsafe { v128_load(rows[y + 1].as_ptr().add(1).cast()) },
        );
        jxx = f32x4_add(jxx, f32x4_mul(gx, gx));
        jxy = f32x4_add(jxy, f32x4_mul(gx, gy));
        jyy = f32x4_add(jyy, f32x4_mul(gy, gy));
        let (gx, gy) = tensor_gradients(
            load2(&rows[y], 4),
            load2(&rows[y], 6),
            load2(&rows[y - 1], 5),
            load2(&rows[y + 1], 5),
        );
        jxx = f32x4_add(jxx, f32x4_mul(gx, gx));
        jxy = f32x4_add(jxy, f32x4_mul(gx, gy));
        jyy = f32x4_add(jyy, f32x4_mul(gy, gy));
    }
    [
        horizontal_sum_x4(jxx),
        horizontal_sum_x4(jxy),
        horizontal_sum_x4(jyy),
    ]
}

#[inline(never)]
#[target_feature(enable = "simd128")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mut error0 = f32x4_splat(0.0);
    let mut error1 = f32x4_splat(0.0);
    let mut error2 = f32x4_splat(0.0);
    let mut error3 = f32x4_splat(0.0);
    let mut error4 = f32x4_splat(0.0);
    for y in 1..7 {
        let left = unsafe { v128_load(rows[y].as_ptr().cast()) };
        let value = unsafe { v128_load(rows[y].as_ptr().add(1).cast()) };
        let top = unsafe { v128_load(rows[y - 1].as_ptr().add(1).cast()) };
        let top_left = unsafe { v128_load(rows[y - 1].as_ptr().cast()) };
        let bottom_left = unsafe { v128_load(rows[y + 1].as_ptr().cast()) };
        let e0 = f32x4_sub(value, left);
        error0 = f32x4_add(error0, f32x4_mul(e0, e0));
        let e1 = f32x4_sub(value, top);
        error1 = f32x4_add(error1, f32x4_mul(e1, e1));
        let e2 = f32x4_sub(value, top_left);
        error2 = f32x4_add(error2, f32x4_mul(e2, e2));
        let e3 = f32x4_sub(value, bottom_left);
        error3 = f32x4_add(error3, f32x4_mul(e3, e3));
        let e4 = f32x4_sub(value, f32x4_sub(f32x4_add(left, top), top_left));
        error4 = f32x4_add(error4, f32x4_mul(e4, e4));

        let left = load2(&rows[y], 4);
        let value = load2(&rows[y], 5);
        let top = load2(&rows[y - 1], 5);
        let top_left = load2(&rows[y - 1], 4);
        let bottom_left = load2(&rows[y + 1], 4);
        let e0 = f32x4_sub(value, left);
        error0 = f32x4_add(error0, f32x4_mul(e0, e0));
        let e1 = f32x4_sub(value, top);
        error1 = f32x4_add(error1, f32x4_mul(e1, e1));
        let e2 = f32x4_sub(value, top_left);
        error2 = f32x4_add(error2, f32x4_mul(e2, e2));
        let e3 = f32x4_sub(value, bottom_left);
        error3 = f32x4_add(error3, f32x4_mul(e3, e3));
        let e4 = f32x4_sub(value, f32x4_sub(f32x4_add(left, top), top_left));
        error4 = f32x4_add(error4, f32x4_mul(e4, e4));
    }
    [
        horizontal_sum_x4(error0),
        horizontal_sum_x4(error1),
        horizontal_sum_x4(error2),
        horizontal_sum_x4(error3),
        horizontal_sum_x4(error4),
    ]
}

#[inline]
#[target_feature(enable = "simd128")]
fn horizontal_energy(row: &[f32], width: usize) -> v128 {
    let tail = f32x4_replace_lane::<2>(
        f32x4_replace_lane::<1>(
            f32x4_replace_lane::<0>(f32x4_splat(0.0), row[width - 3] - row[width - 4]),
            row[width - 2] - row[width - 3],
        ),
        row[width - 1] - row[width - 2],
    );
    let mut energy = f32x4_mul(tail, tail);
    if width == 8 {
        let d = f32x4_sub(unsafe { v128_load(row.as_ptr().add(1).cast()) }, unsafe {
            v128_load(row.as_ptr().cast())
        });
        energy = f32x4_add(energy, f32x4_mul(d, d));
    }
    energy
}

#[inline]
#[target_feature(enable = "simd128")]
fn vertical_energy(top: &[f32], bottom: &[f32], width: usize) -> v128 {
    let d0 = f32x4_sub(unsafe { v128_load(bottom.as_ptr().cast()) }, unsafe {
        v128_load(top.as_ptr().cast())
    });
    let mut energy = f32x4_mul(d0, d0);
    if width == 8 {
        let d1 = f32x4_sub(
            unsafe { v128_load(bottom.as_ptr().add(4).cast()) },
            unsafe { v128_load(top.as_ptr().add(4).cast()) },
        );
        energy = f32x4_add(energy, f32x4_mul(d1, d1));
    }
    energy
}

#[inline]
#[target_feature(enable = "simd128")]
fn gradient_energy(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);
    for base in (0..height).step_by(4) {
        let row0 = &values[base * stride..][..width];
        let row1 = &values[(base + 1) * stride..][..width];
        let row2 = &values[(base + 2) * stride..][..width];
        let row3 = &values[(base + 3) * stride..][..width];
        sum0 = f32x4_add(sum0, horizontal_energy(row0, width));
        sum1 = f32x4_add(sum1, horizontal_energy(row1, width));
        sum2 = f32x4_add(sum2, horizontal_energy(row2, width));
        sum3 = f32x4_add(sum3, horizontal_energy(row3, width));
    }
    let vertical_rows = height - 1;
    for base in (0..vertical_rows).step_by(4) {
        sum0 = f32x4_add(
            sum0,
            vertical_energy(
                &values[base * stride..],
                &values[(base + 1) * stride..],
                width,
            ),
        );
        if base + 1 < vertical_rows {
            sum1 = f32x4_add(
                sum1,
                vertical_energy(
                    &values[(base + 1) * stride..],
                    &values[(base + 2) * stride..],
                    width,
                ),
            );
        }
        if base + 2 < vertical_rows {
            sum2 = f32x4_add(
                sum2,
                vertical_energy(
                    &values[(base + 2) * stride..],
                    &values[(base + 3) * stride..],
                    width,
                ),
            );
        }
        if base + 3 < vertical_rows {
            sum3 = f32x4_add(
                sum3,
                vertical_energy(
                    &values[(base + 3) * stride..],
                    &values[(base + 4) * stride..],
                    width,
                ),
            );
        }
    }
    horizontal_sum_x4(f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3)))
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
