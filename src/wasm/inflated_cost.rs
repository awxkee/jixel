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
use crate::image::Image3F;
use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn clamp01_f32x4(value: v128) -> v128 {
    f32x4_min(f32x4_max(value, f32x4_splat(0.0)), f32x4_splat(1.0))
}

#[inline]
#[target_feature(enable = "simd128")]
fn xyb_to_oklab_f32x4(
    matrix: &crate::xyb::XybMatrix,
    x: v128,
    y: v128,
    b: v128,
) -> (v128, v128, v128) {
    let cube = |value: v128| {
        let value = f32x4_sub(value, f32x4_splat(crate::xyb::NEG_BIAS_CBRT));
        f32x4_sub(
            f32x4_mul(f32x4_mul(value, value), value),
            f32x4_splat(crate::xyb::OPSIN_BIAS),
        )
    };
    let mixed0 = cube(f32x4_add(y, x));
    let mixed1 = cube(f32x4_sub(y, x));
    let mixed2 = cube(b);
    let inverse_row = |offset: usize| {
        f32x4_add(
            f32x4_mul(mixed0, f32x4_splat(matrix.inv[offset])),
            f32x4_add(
                f32x4_mul(mixed1, f32x4_splat(matrix.inv[offset + 1])),
                f32x4_mul(mixed2, f32x4_splat(matrix.inv[offset + 2])),
            ),
        )
    };
    let r = inverse_row(0);
    let g = inverse_row(3);
    let b = inverse_row(6);
    let zero = f32x4_splat(0.0);
    let rgb_row = |cr: f32, cg: f32, cb: f32| {
        f32x4_max(
            f32x4_add(
                f32x4_mul(r, f32x4_splat(cr)),
                f32x4_add(f32x4_mul(g, f32x4_splat(cg)), f32x4_mul(b, f32x4_splat(cb))),
            ),
            zero,
        )
    };
    let l = rgb_row(0.412_221_46, 0.536_332_55, 0.051_445_995);
    let m = rgb_row(0.211_903_5, 0.680_699_5, 0.107_396_96);
    let s = rgb_row(0.088_302_46, 0.281_718_85, 0.629_978_7);
    let (l, m, s) = super::xyb::vcbrt_fast3_positive_wasm(l, m, s);
    let lab_row = |cl: f32, cm: f32, cs: f32| {
        f32x4_add(
            f32x4_mul(l, f32x4_splat(cl)),
            f32x4_add(f32x4_mul(m, f32x4_splat(cm)), f32x4_mul(s, f32x4_splat(cs))),
        )
    };
    (
        lab_row(0.210_454_26, 0.793_617_8, -0.004_072_047),
        lab_row(1.977_998_5, -2.428_592_2, 0.450_593_7),
        lab_row(0.025_904_037, 0.782_771_77, -0.808_675_77),
    )
}

#[inline]
#[target_feature(enable = "simd128")]
fn shifted_right_f32x4(value: v128) -> v128 {
    i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15>(value, value)
}

/// # Safety
/// The input slices must cover the requested block.
#[target_feature(enable = "simd128")]
pub(crate) fn rgb_hue_chroma_edge_loss_wasm(
    opsin: &Image3F,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    spatial_error: [&[f32]; 3],
    matrix: &crate::xyb::XybMatrix,
) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let Some(end_x) = px.checked_add(width) else {
        return crate::inflated_cost::rgb_hue_chroma_edge_loss_scalar(
            opsin,
            px,
            py,
            width,
            height,
            spatial_error,
            matrix,
        );
    };
    let Some(end_y) = py.checked_add(height) else {
        return crate::inflated_cost::rgb_hue_chroma_edge_loss_scalar(
            opsin,
            px,
            py,
            width,
            height,
            spatial_error,
            matrix,
        );
    };
    if end_x > opsin.xsize() || end_y > opsin.ysize() || !width.is_multiple_of(4) {
        return crate::inflated_cost::rgb_hue_chroma_edge_loss_scalar(
            opsin,
            px,
            py,
            width,
            height,
            spatial_error,
            matrix,
        );
    }

    let zero = f32x4_splat(0.0);
    let mut sum = zero;
    for y in 0..height {
        let rows = [
            opsin.plane_row(0, py + y),
            opsin.plane_row(1, py + y),
            opsin.plane_row(2, py + y),
        ];
        let below = if y + 1 < height {
            [
                opsin.plane_row(0, py + y + 1),
                opsin.plane_row(1, py + y + 1),
                opsin.plane_row(2, py + y + 1),
            ]
        } else {
            rows
        };
        let error_offset = y * width;
        for x in (0..width).step_by(4) {
            let p = px + x;
            let sx = unsafe { v128_load(rows[0].as_ptr().add(p).cast()) };
            let sy = unsafe { v128_load(rows[1].as_ptr().add(p).cast()) };
            let sb = unsafe { v128_load(rows[2].as_ptr().add(p).cast()) };
            let (rx, ry, rb) = if x + 4 < width {
                (
                    unsafe { v128_load(rows[0].as_ptr().add(p + 1).cast()) },
                    unsafe { v128_load(rows[1].as_ptr().add(p + 1).cast()) },
                    unsafe { v128_load(rows[2].as_ptr().add(p + 1).cast()) },
                )
            } else {
                (
                    shifted_right_f32x4(sx),
                    shifted_right_f32x4(sy),
                    shifted_right_f32x4(sb),
                )
            };
            let bx = unsafe { v128_load(below[0].as_ptr().add(p).cast()) };
            let by = unsafe { v128_load(below[1].as_ptr().add(p).cast()) };
            let bb = unsafe { v128_load(below[2].as_ptr().add(p).cast()) };
            let cb = f32x4_sub(sb, sy);
            let horizontal = f32x4_add(
                f32x4_abs(f32x4_sub(rx, sx)),
                f32x4_abs(f32x4_sub(f32x4_sub(rb, ry), cb)),
            );
            let vertical = f32x4_add(
                f32x4_abs(f32x4_sub(bx, sx)),
                f32x4_abs(f32x4_sub(f32x4_sub(bb, by), cb)),
            );
            let edge_risk = clamp01_f32x4(f32x4_mul(
                f32x4_sub(f32x4_max(horizontal, vertical), f32x4_splat(0.006)),
                f32x4_splat(1.0 / 0.030),
            ));
            let e = error_offset + x;
            let ex = unsafe { v128_load(spatial_error[0].as_ptr().add(e).cast()) };
            let ey = unsafe { v128_load(spatial_error[1].as_ptr().add(e).cast()) };
            let eb = unsafe { v128_load(spatial_error[2].as_ptr().add(e).cast()) };
            let (source_l, source_a, source_b) = xyb_to_oklab_f32x4(matrix, sx, sy, sb);
            let (_, recon_a, recon_b) = xyb_to_oklab_f32x4(
                matrix,
                f32x4_sub(sx, ex),
                f32x4_sub(sy, ey),
                f32x4_sub(sb, eb),
            );
            let source_chroma = f32x4_sqrt(f32x4_add(
                f32x4_mul(source_a, source_a),
                f32x4_mul(source_b, source_b),
            ));
            let recon_chroma = f32x4_sqrt(f32x4_add(
                f32x4_mul(recon_a, recon_a),
                f32x4_mul(recon_b, recon_b),
            ));
            let brightness_risk = clamp01_f32x4(f32x4_mul(
                f32x4_sub(source_l, f32x4_splat(0.35)),
                f32x4_splat(1.0 / 0.40),
            ));
            let chroma_risk = clamp01_f32x4(f32x4_mul(
                f32x4_sub(source_chroma, f32x4_splat(0.03)),
                f32x4_splat(1.0 / 0.12),
            ));
            let risk = f32x4_mul(f32x4_mul(edge_risk, brightness_risk), chroma_risk);
            let desaturation = f32x4_max(f32x4_sub(source_chroma, recon_chroma), zero);
            let perpendicular = f32x4_div(
                f32x4_sub(f32x4_mul(source_a, recon_b), f32x4_mul(source_b, recon_a)),
                f32x4_add(source_chroma, f32x4_splat(1e-4)),
            );
            let penalty = f32x4_add(
                f32x4_mul(desaturation, desaturation),
                f32x4_mul(f32x4_splat(0.75), f32x4_mul(perpendicular, perpendicular)),
            );
            sum = f32x4_add(sum, f32x4_mul(risk, penalty));
        }
    }
    f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum)
}

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
