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
use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn accumulate_gradient_vectors_x4(left: v128, right: v128, sum: v128) -> v128 {
    let difference = f32x4_sub(right, left);
    f32x4_add(sum, f32x4_mul(difference, difference))
}

#[inline]
#[target_feature(enable = "simd128")]
fn accumulate_gradient_x4(left: &[f32; 4], right: &[f32; 4], sum: v128) -> v128 {
    let left = unsafe { v128_load(left.as_ptr().cast()) };
    let right = unsafe { v128_load(right.as_ptr().cast()) };
    accumulate_gradient_vectors_x4(left, right, sum)
}

#[target_feature(enable = "simd128")]
pub(crate) fn error_gradient_energy_wasm(error: &[f32], width: usize, height: usize) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("gradient plane size overflow");
    assert!(error.len() >= n);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let rows = error[..n].chunks_exact(width);
    let mut sum = f32x4_splat(0.0);

    for row in rows.clone() {
        let (row4, row_tail) = row.as_chunks::<4>();
        if row_tail.is_empty() {
            let (right4, _) = row[1..].as_chunks::<4>();
            for (left, right) in row4.iter().zip(right4) {
                sum = accumulate_gradient_x4(left, right, sum);
            }
            let left = unsafe { v128_load(row4.last().unwrap().as_ptr().cast()) };
            sum =
                accumulate_gradient_vectors_x4(left, i32x4_shuffle::<1, 2, 3, 3>(left, left), sum);
            continue;
        }
        let (left4, tail) = row[..width - 1].as_chunks::<4>();
        let (right4, right_tail) = row[1..].as_chunks::<4>();
        for (left, right) in left4.iter().zip(right4) {
            sum = accumulate_gradient_x4(left, right, sum);
        }
        if !tail.is_empty() {
            let mut left = [0.0; 4];
            let mut right = [0.0; 4];
            left[..tail.len()].copy_from_slice(tail);
            right[..right_tail.len()].copy_from_slice(right_tail);
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

    f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum)
}

#[inline]
#[target_feature(enable = "simd128")]
fn peak_excess_x4(a: v128, b: v128, source_a: v128, source_b: v128, floor: v128) -> v128 {
    let error_gradient = f32x4_abs(f32x4_sub(b, a));
    let source_gradient = f32x4_abs(f32x4_sub(source_b, source_a));
    let excess = f32x4_max(
        f32x4_splat(0.0),
        f32x4_sub(
            f32x4_sub(error_gradient, f32x4_mul(source_gradient, f32x4_splat(0.5))),
            floor,
        ),
    );
    f32x4_mul(excess, excess)
}

#[inline]
#[target_feature(enable = "simd128")]
fn horizontal_max_x4(value: v128) -> f32 {
    f32x4_extract_lane::<0>(value)
        .max(f32x4_extract_lane::<1>(value))
        .max(f32x4_extract_lane::<2>(value))
        .max(f32x4_extract_lane::<3>(value))
}

#[target_feature(enable = "simd128")]
pub(crate) fn error_gradient_peak_energy_wasm(
    error: &[f32],
    original: &[f32],
    width: usize,
    height: usize,
    floor: f32,
) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("gradient plane size overflow");
    assert!(error.len() >= n && original.len() >= n);
    assert!(floor.is_finite() && floor >= 0.0);
    if width == 0 || height == 0 {
        return 0.0;
    }
    if !width.is_multiple_of(4) {
        return crate::inflated_cost::error_gradient_peak_energy_scalar(
            error, original, width, height, floor,
        );
    }

    let floor = f32x4_splat(floor);
    let zero = f32x4_splat(0.0);
    let mut total = 0.0f32;
    for cell_y in (0..height).step_by(4) {
        for cell_x in (0..width).step_by(4) {
            let mut max_x = zero;
            let mut max_y = zero;
            for y in cell_y..(cell_y + 4).min(height) {
                let p = y * width + cell_x;
                let current = unsafe { v128_load(error.as_ptr().add(p).cast()) };
                let source = unsafe { v128_load(original.as_ptr().add(p).cast()) };
                let (right, source_right) = if cell_x + 4 < width {
                    (
                        unsafe { v128_load(error.as_ptr().add(p + 1).cast()) },
                        unsafe { v128_load(original.as_ptr().add(p + 1).cast()) },
                    )
                } else {
                    (
                        i32x4_shuffle::<1, 2, 3, 3>(current, current),
                        i32x4_shuffle::<1, 2, 3, 3>(source, source),
                    )
                };
                max_x = f32x4_max(
                    max_x,
                    peak_excess_x4(current, right, source, source_right, floor),
                );
                if y + 1 < height {
                    let below = unsafe { v128_load(error.as_ptr().add(p + width).cast()) };
                    let source_below =
                        unsafe { v128_load(original.as_ptr().add(p + width).cast()) };
                    max_y = f32x4_max(
                        max_y,
                        peak_excess_x4(current, below, source, source_below, floor),
                    );
                }
            }
            total += horizontal_max_x4(max_x) + horizontal_max_x4(max_y);
        }
    }
    total
}

#[inline]
#[target_feature(enable = "simd128")]
fn combine_error_x4(spatial: &[f32; 4], luma: &[f32; 4], factor: v128, combined: &mut [f32; 4]) {
    let spatial = unsafe { v128_load(spatial.as_ptr().cast()) };
    let luma = unsafe { v128_load(luma.as_ptr().cast()) };
    let value = f32x4_add(f32x4_mul(factor, luma), spatial);
    unsafe { v128_store(combined.as_mut_ptr().cast(), value) };
}

#[target_feature(enable = "simd128")]
pub(crate) fn combine_error_wasm(spatial: &[f32], luma: &[f32], factor: f32, combined: &mut [f32]) {
    debug_assert_eq!(spatial.len(), luma.len());
    debug_assert_eq!(spatial.len(), combined.len());
    let (spatial16, spatial_tail) = spatial.as_chunks::<16>();
    let (luma16, luma_tail) = luma.as_chunks::<16>();
    let (combined16, combined_tail) = combined.as_chunks_mut::<16>();
    let factor = f32x4_splat(factor);

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
