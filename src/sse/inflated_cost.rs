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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn clamp01_f32x4(value: __m128) -> __m128 {
    _mm_min_ps(_mm_max_ps(value, _mm_setzero_ps()), _mm_set1_ps(1.0))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn xyb_to_oklab_f32x4(
    matrix: &crate::xyb::XybMatrix,
    x: __m128,
    y: __m128,
    b: __m128,
) -> (__m128, __m128, __m128) {
    let cube = |value: __m128| {
        let value = _mm_sub_ps(value, _mm_set1_ps(crate::xyb::NEG_BIAS_CBRT));
        _mm_sub_ps(
            _mm_mul_ps(_mm_mul_ps(value, value), value),
            _mm_set1_ps(crate::xyb::OPSIN_BIAS),
        )
    };
    let mixed0 = cube(_mm_add_ps(y, x));
    let mixed1 = cube(_mm_sub_ps(y, x));
    let mixed2 = cube(b);
    let inverse_row = |offset: usize| {
        _mm_add_ps(
            _mm_mul_ps(mixed0, _mm_set1_ps(matrix.inv[offset])),
            _mm_add_ps(
                _mm_mul_ps(mixed1, _mm_set1_ps(matrix.inv[offset + 1])),
                _mm_mul_ps(mixed2, _mm_set1_ps(matrix.inv[offset + 2])),
            ),
        )
    };
    let r = inverse_row(0);
    let g = inverse_row(3);
    let b = inverse_row(6);
    let zero = _mm_setzero_ps();
    let rgb_row = |cr: f32, cg: f32, cb: f32| {
        _mm_max_ps(
            _mm_add_ps(
                _mm_mul_ps(r, _mm_set1_ps(cr)),
                _mm_add_ps(
                    _mm_mul_ps(g, _mm_set1_ps(cg)),
                    _mm_mul_ps(b, _mm_set1_ps(cb)),
                ),
            ),
            zero,
        )
    };
    let l = rgb_row(0.412_221_46, 0.536_332_55, 0.051_445_995);
    let m = rgb_row(0.211_903_5, 0.680_699_5, 0.107_396_96);
    let s = rgb_row(0.088_302_46, 0.281_718_85, 0.629_978_7);
    let (l, m, s) = super::xyb::vcbrt_fast3_positive_sse41(l, m, s);
    let lab_row = |cl: f32, cm: f32, cs: f32| {
        _mm_add_ps(
            _mm_mul_ps(l, _mm_set1_ps(cl)),
            _mm_add_ps(
                _mm_mul_ps(m, _mm_set1_ps(cm)),
                _mm_mul_ps(s, _mm_set1_ps(cs)),
            ),
        )
    };
    (
        lab_row(0.210_454_26, 0.793_617_8, -0.004_072_047),
        lab_row(1.977_998_5, -2.428_592_2, 0.450_593_7),
        lab_row(0.025_904_037, 0.782_771_77, -0.808_675_77),
    )
}

/// # Safety
/// The caller must ensure SSE4.1 is available and the input slices cover the block.
#[target_feature(enable = "sse4.1")]
pub(crate) fn rgb_hue_chroma_edge_loss_sse41(
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

    let zero = _mm_setzero_ps();
    let abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
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
            let sx = unsafe { _mm_loadu_ps(rows[0].as_ptr().add(p)) };
            let sy = unsafe { _mm_loadu_ps(rows[1].as_ptr().add(p)) };
            let sb = unsafe { _mm_loadu_ps(rows[2].as_ptr().add(p)) };
            let (rx, ry, rb) = if x + 4 < width {
                (
                    unsafe { _mm_loadu_ps(rows[0].as_ptr().add(p + 1)) },
                    unsafe { _mm_loadu_ps(rows[1].as_ptr().add(p + 1)) },
                    unsafe { _mm_loadu_ps(rows[2].as_ptr().add(p + 1)) },
                )
            } else {
                (
                    _mm_shuffle_ps::<0xf9>(sx, sx),
                    _mm_shuffle_ps::<0xf9>(sy, sy),
                    _mm_shuffle_ps::<0xf9>(sb, sb),
                )
            };
            let bx = unsafe { _mm_loadu_ps(below[0].as_ptr().add(p)) };
            let by = unsafe { _mm_loadu_ps(below[1].as_ptr().add(p)) };
            let bb = unsafe { _mm_loadu_ps(below[2].as_ptr().add(p)) };
            let cb = _mm_sub_ps(sb, sy);
            let horizontal = _mm_add_ps(
                _mm_and_ps(abs_mask, _mm_sub_ps(rx, sx)),
                _mm_and_ps(abs_mask, _mm_sub_ps(_mm_sub_ps(rb, ry), cb)),
            );
            let vertical = _mm_add_ps(
                _mm_and_ps(abs_mask, _mm_sub_ps(bx, sx)),
                _mm_and_ps(abs_mask, _mm_sub_ps(_mm_sub_ps(bb, by), cb)),
            );
            let edge_risk = clamp01_f32x4(_mm_mul_ps(
                _mm_sub_ps(_mm_max_ps(horizontal, vertical), _mm_set1_ps(0.006)),
                _mm_set1_ps(1.0 / 0.030),
            ));
            let e = error_offset + x;
            let ex = unsafe { _mm_loadu_ps(spatial_error[0].as_ptr().add(e)) };
            let ey = unsafe { _mm_loadu_ps(spatial_error[1].as_ptr().add(e)) };
            let eb = unsafe { _mm_loadu_ps(spatial_error[2].as_ptr().add(e)) };
            let (source_l, source_a, source_b) = xyb_to_oklab_f32x4(matrix, sx, sy, sb);
            let (_, recon_a, recon_b) = xyb_to_oklab_f32x4(
                matrix,
                _mm_sub_ps(sx, ex),
                _mm_sub_ps(sy, ey),
                _mm_sub_ps(sb, eb),
            );
            let source_chroma = _mm_sqrt_ps(_mm_add_ps(
                _mm_mul_ps(source_a, source_a),
                _mm_mul_ps(source_b, source_b),
            ));
            let recon_chroma = _mm_sqrt_ps(_mm_add_ps(
                _mm_mul_ps(recon_a, recon_a),
                _mm_mul_ps(recon_b, recon_b),
            ));
            let brightness_risk = clamp01_f32x4(_mm_mul_ps(
                _mm_sub_ps(source_l, _mm_set1_ps(0.35)),
                _mm_set1_ps(1.0 / 0.40),
            ));
            let chroma_risk = clamp01_f32x4(_mm_mul_ps(
                _mm_sub_ps(source_chroma, _mm_set1_ps(0.03)),
                _mm_set1_ps(1.0 / 0.12),
            ));
            let risk = _mm_mul_ps(_mm_mul_ps(edge_risk, brightness_risk), chroma_risk);
            let desaturation = _mm_max_ps(_mm_sub_ps(source_chroma, recon_chroma), zero);
            let perpendicular = _mm_div_ps(
                _mm_sub_ps(_mm_mul_ps(source_a, recon_b), _mm_mul_ps(source_b, recon_a)),
                _mm_add_ps(source_chroma, _mm_set1_ps(1e-4)),
            );
            let penalty = _mm_add_ps(
                _mm_mul_ps(desaturation, desaturation),
                _mm_mul_ps(_mm_set1_ps(0.75), _mm_mul_ps(perpendicular, perpendicular)),
            );
            sum = _mm_add_ps(sum, _mm_mul_ps(risk, penalty));
        }
    }
    horizontal_sum_x4(sum)
}

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
            let (right4, _) = row[1..].as_chunks::<4>();
            for (left, right) in row4.iter().zip(right4) {
                sum = accumulate_gradient_x4(left, right, sum);
            }
            let left = unsafe { _mm_loadu_ps(row4.last().unwrap().as_ptr()) };
            sum = accumulate_gradient_vectors_x4(left, _mm_shuffle_ps::<249>(left, left), sum);
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
    horizontal_sum_x4(sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn peak_excess_x4(
    a: __m128,
    b: __m128,
    source_a: __m128,
    source_b: __m128,
    floor: __m128,
) -> __m128 {
    let sign = _mm_set1_ps(-0.0);
    let error_gradient = _mm_andnot_ps(sign, _mm_sub_ps(b, a));
    let source_gradient = _mm_andnot_ps(sign, _mm_sub_ps(source_b, source_a));
    let excess = _mm_max_ps(
        _mm_setzero_ps(),
        _mm_sub_ps(
            _mm_sub_ps(
                error_gradient,
                _mm_mul_ps(source_gradient, _mm_set1_ps(0.5)),
            ),
            floor,
        ),
    );
    _mm_mul_ps(excess, excess)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn horizontal_max_x4(value: __m128) -> f32 {
    let pair_max = _mm_max_ps(value, _mm_shuffle_ps::<0x4e>(value, value));
    _mm_cvtss_f32(_mm_max_ps(
        pair_max,
        _mm_shuffle_ps::<0xb1>(pair_max, pair_max),
    ))
}

/// # Safety
/// The caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn error_gradient_peak_energy_sse41(
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

    let floor = _mm_set1_ps(floor);
    let zero = _mm_setzero_ps();
    let mut total = 0.0f32;
    for cell_y in (0..height).step_by(4) {
        for cell_x in (0..width).step_by(4) {
            let mut max_x = zero;
            let mut max_y = zero;
            for y in cell_y..(cell_y + 4).min(height) {
                let p = y * width + cell_x;
                let current = unsafe { _mm_loadu_ps(error.as_ptr().add(p)) };
                let source = unsafe { _mm_loadu_ps(original.as_ptr().add(p)) };
                let (right, source_right) = if cell_x + 4 < width {
                    (unsafe { _mm_loadu_ps(error.as_ptr().add(p + 1)) }, unsafe {
                        _mm_loadu_ps(original.as_ptr().add(p + 1))
                    })
                } else {
                    (
                        _mm_shuffle_ps::<249>(current, current),
                        _mm_shuffle_ps::<249>(source, source),
                    )
                };
                max_x = _mm_max_ps(
                    max_x,
                    peak_excess_x4(current, right, source, source_right, floor),
                );
                if y + 1 < height {
                    let below = unsafe { _mm_loadu_ps(error.as_ptr().add(p + width)) };
                    let source_below = unsafe { _mm_loadu_ps(original.as_ptr().add(p + width)) };
                    max_y = _mm_max_ps(
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
