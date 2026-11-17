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
            let radial_error = _mm_div_ps(
                _mm_add_ps(
                    _mm_mul_ps(source_a, _mm_sub_ps(source_a, recon_a)),
                    _mm_mul_ps(source_b, _mm_sub_ps(source_b, recon_b)),
                ),
                _mm_max_ps(source_chroma, _mm_set1_ps(1e-4)),
            );
            let brightness_risk = clamp01_f32x4(_mm_mul_ps(
                _mm_sub_ps(
                    _mm_sub_ps(source_l, _mm_set1_ps(0.35)),
                    _mm_min_ps(source_b, zero),
                ),
                _mm_set1_ps(1.0 / 0.40),
            ));
            let chroma_risk = clamp01_f32x4(_mm_mul_ps(
                _mm_sub_ps(source_chroma, _mm_set1_ps(0.03)),
                _mm_set1_ps(1.0 / 0.12),
            ));
            let risk = _mm_mul_ps(_mm_mul_ps(edge_risk, brightness_risk), chroma_risk);
            let desaturation = _mm_max_ps(radial_error, zero);
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
fn horizontal_sum_x4(value: __m128) -> f32 {
    let sum = _mm_hadd_ps(value, value);
    _mm_cvtss_f32(_mm_hadd_ps(sum, sum))
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

/// `f32::round` (ties away from zero) on every lane, signed zeros included.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(super) fn round_ties_away(x: __m128) -> __m128 {
    let sign = _mm_and_ps(x, _mm_set1_ps(-0.0));
    let truncated = _mm_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x);
    let fraction = _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(x, truncated));
    let adjust = _mm_cmpge_ps(fraction, _mm_set1_ps(0.5));
    let signed_one = _mm_or_ps(_mm_set1_ps(1.0), sign);
    _mm_or_ps(_mm_add_ps(truncated, _mm_and_ps(signed_one, adjust)), sign)
}

/// The biased, rate-free coefficient error of `recon_quantize_scalar`, bit
/// for bit: levels round ties away from zero as the scalar quantizer does.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn recon_quantize_error_sse41(
    coeff: &[f32],
    inv: &[f32],
    matrix: &[f32],
    quant_scale: f32,
    bias: crate::group::DequantBias,
    thresholds: &[f32; 4],
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    coeff_error: &mut [f32],
    _rate_log2_lut: &crate::inflated_cost::RateLog2Lut,
    _scan_pos: &[u32],
) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("coefficient size overflow");
    assert!(width.is_multiple_of(4) && half.is_multiple_of(4));
    assert!(coeff.len() >= n && inv.len() >= n && matrix.len() >= n && coeff_error.len() >= n);
    let scale = _mm_set1_ps(quant_scale);
    let inv_scale = _mm_set1_ps(1.0 / quant_scale);
    let sign = _mm_set1_ps(-0.0);
    let lane_ids = _mm_setr_epi32(0, 1, 2, 3);
    for (y, (((coeff_row, inv_row), matrix_row), error_row)) in coeff
        .chunks_exact(width)
        .zip(inv.chunks_exact(width))
        .zip(matrix.chunks_exact(width))
        .zip(coeff_error.chunks_exact_mut(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        for (chunk_x, (((coeff4, inv4), matrix4), error4)) in coeff_row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(inv_row.as_chunks::<4>().0.iter())
            .zip(matrix_row.as_chunks::<4>().0.iter())
            .zip(error_row.as_chunks_mut::<4>().0.iter_mut())
            .enumerate()
        {
            let x = chunk_x * 4;
            let threshold = _mm_set1_ps(if x >= half {
                thresholds[yfix + 1]
            } else {
                thresholds[yfix]
            });
            let coeff_v = unsafe { _mm_loadu_ps(coeff4.as_ptr()) };
            let inv_v = unsafe { _mm_loadu_ps(inv4.as_ptr()) };
            let scaled = _mm_mul_ps(_mm_mul_ps(inv_v, scale), coeff_v);
            let keep = _mm_cmpge_ps(_mm_andnot_ps(sign, scaled), threshold);
            let quantized = _mm_and_ps(round_ties_away(scaled), keep);
            let dequantized = super::ac_strategy::sse_dequantized_level_f32(quantized, bias);
            let matrix_v = unsafe { _mm_loadu_ps(matrix4.as_ptr()) };
            let error = _mm_mul_ps(
                _mm_sub_ps(scaled, dequantized),
                _mm_mul_ps(matrix_v, inv_scale),
            );
            let active = if y < cy && x < cx {
                _mm_castsi128_ps(_mm_cmpgt_epi32(
                    _mm_add_epi32(_mm_set1_epi32(x as i32), lane_ids),
                    _mm_set1_epi32(cx as i32 - 1),
                ))
            } else {
                _mm_castsi128_ps(_mm_set1_epi32(-1))
            };
            unsafe { _mm_storeu_ps(error4.as_mut_ptr(), _mm_and_ps(error, active)) };
        }
    }
    0.0
}
