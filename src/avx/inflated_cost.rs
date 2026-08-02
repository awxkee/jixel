/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use crate::avx::ac_strategy::{avx2_log2p1_f32, hmax_u32, hsum256};
use crate::image::Plane;
use crate::inflated_cost::{
    RateLog2Lut, ReconDistInput, ReconErrorKernels, ReconKernels, recon_dist_and_rate_with_kernels,
    validate_ssim_inputs,
};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn accumulate_gradient_vectors_x8(left: __m256, right: __m256, sum: __m256) -> __m256 {
    let difference = _mm256_sub_ps(right, left);
    _mm256_fmadd_ps(difference, difference, sum)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn accumulate_gradient_x8(left: &[f32; 8], right: &[f32; 8], sum: __m256) -> __m256 {
    let left = unsafe { _mm256_loadu_ps(left.as_ptr()) };
    let right = unsafe { _mm256_loadu_ps(right.as_ptr()) };
    accumulate_gradient_vectors_x8(left, right, sum)
}

/// # Safety
/// The caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn error_gradient_energy_avx2(error: &[f32], width: usize, height: usize) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("gradient plane size overflow");
    assert!(error.len() >= n);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let rows = error[..n].chunks_exact(width);
    let mut sum = _mm256_setzero_ps();

    for row in rows.clone() {
        let (row8, row_tail) = row.as_chunks::<8>();
        if row_tail.is_empty() {
            for (chunk, left) in row8.iter().enumerate() {
                let left = unsafe { _mm256_loadu_ps(left.as_ptr()) };
                let right = if chunk + 1 == row8.len() {
                    _mm256_permutevar8x32_ps(left, _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7))
                } else {
                    unsafe { _mm256_loadu_ps(row.as_ptr().add(chunk * 8 + 1)) }
                };
                sum = accumulate_gradient_vectors_x8(left, right, sum);
            }
            continue;
        }
        let (left8, tail) = row[..width - 1].as_chunks::<8>();
        for (chunk, left) in left8.iter().enumerate() {
            let right = row[chunk * 8 + 1..].first_chunk::<8>().unwrap();
            sum = accumulate_gradient_x8(left, right, sum);
        }
        if !tail.is_empty() {
            let offset = left8.len() * 8;
            let mut left = [0.0; 8];
            let mut right = [0.0; 8];
            left[..tail.len()].copy_from_slice(tail);
            right[..tail.len()].copy_from_slice(&row[offset + 1..width]);
            sum = accumulate_gradient_x8(&left, &right, sum);
        }
    }

    for (top, bottom) in rows.clone().zip(rows.skip(1)) {
        let (top8, top_tail) = top.as_chunks::<8>();
        let (bottom8, bottom_tail) = bottom.as_chunks::<8>();
        for (top, bottom) in top8.iter().zip(bottom8) {
            sum = accumulate_gradient_x8(top, bottom, sum);
        }
        if !top_tail.is_empty() {
            let mut top = [0.0; 8];
            let mut bottom = [0.0; 8];
            top[..top_tail.len()].copy_from_slice(top_tail);
            bottom[..bottom_tail.len()].copy_from_slice(bottom_tail);
            sum = accumulate_gradient_x8(&top, &bottom, sum);
        }
    }
    hsum256(sum)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn combine_error_x8(spatial: &[f32; 8], luma: &[f32; 8], factor: __m256, combined: &mut [f32; 8]) {
    let spatial = unsafe { _mm256_loadu_ps(spatial.as_ptr()) };
    let luma = unsafe { _mm256_loadu_ps(luma.as_ptr()) };
    let value = _mm256_fmadd_ps(factor, luma, spatial);
    unsafe { _mm256_storeu_ps(combined.as_mut_ptr(), value) };
}

/// # Safety
/// The caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn combine_error_avx2(spatial: &[f32], luma: &[f32], factor: f32, combined: &mut [f32]) {
    debug_assert_eq!(spatial.len(), luma.len());
    debug_assert_eq!(spatial.len(), combined.len());
    let (spatial32, spatial_tail) = spatial.as_chunks::<32>();
    let (luma32, luma_tail) = luma.as_chunks::<32>();
    let (combined32, combined_tail) = combined.as_chunks_mut::<32>();
    let factor = _mm256_set1_ps(factor);

    for ((spatial, luma), combined) in spatial32.iter().zip(luma32).zip(combined32) {
        let [s0, s1, s2, s3] = spatial.as_chunks::<8>().0 else {
            unreachable!()
        };
        let [l0, l1, l2, l3] = luma.as_chunks::<8>().0 else {
            unreachable!()
        };
        let [c0, c1, c2, c3] = combined.as_chunks_mut::<8>().0 else {
            unreachable!()
        };
        combine_error_x8(s0, l0, factor, c0);
        combine_error_x8(s1, l1, factor, c1);
        combine_error_x8(s2, l2, factor, c2);
        combine_error_x8(s3, l3, factor, c3);
    }

    let (spatial8, spatial_remainder) = spatial_tail.as_chunks::<8>();
    let (luma8, luma_remainder) = luma_tail.as_chunks::<8>();
    let (combined8, combined_remainder) = combined_tail.as_chunks_mut::<8>();
    for ((spatial, luma), combined) in spatial8.iter().zip(luma8).zip(combined8) {
        combine_error_x8(spatial, luma, factor, combined);
    }
    if !spatial_remainder.is_empty() {
        let mut spatial = [0.0; 8];
        let mut luma = [0.0; 8];
        let mut combined = [0.0; 8];
        spatial[..spatial_remainder.len()].copy_from_slice(spatial_remainder);
        luma[..luma_remainder.len()].copy_from_slice(luma_remainder);
        combine_error_x8(&spatial, &luma, factor, &mut combined);
        combined_remainder.copy_from_slice(&combined[..combined_remainder.len()]);
    }
}

/// # Safety
/// The caller must ensure AVX2 is available. Slice bounds are validated before
/// any vector load or store.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn recon_dist_and_rate_avx2(
    scratch: &mut [[f32; 1024]; 8],
    input: &ReconDistInput<'_>,
    error: &ReconErrorKernels,
) -> (f32, f32) {
    recon_dist_and_rate_with_kernels(
        scratch,
        input,
        &ReconKernels {
            quantize: recon_quantize_avx2,
            ssim: ssim_deficit_avx2,
            prepare: prepare_reconstruction_avx2,
            error,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn recon_quantize_avx2(
    coeff: &[f32],
    inv: &[f32],
    quant_scale: f32,
    thresholds: &[f32; 4],
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    coeff_error: &mut [f32],
    _rate_log2_lut: &RateLog2Lut,
    scan_pos: &[u32],
) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("coefficient size overflow");
    assert!(width.is_multiple_of(8));
    assert!(coeff.len() >= n && inv.len() >= n && coeff_error.len() >= n && scan_pos.len() >= n);
    let scale = _mm256_set1_ps(quant_scale);
    let sign = _mm256_set1_ps(-0.0);
    let lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let all = _mm256_set1_epi32(-1);
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let mut nonzero = 0usize;
    let mut magnitude_bits = _mm256_setzero_ps();
    let mut scan_acc = _mm256_setzero_si256();

    for (y, ((coeff_row, inv_row), error_row)) in coeff
        .chunks_exact(width)
        .zip(inv.chunks_exact(width))
        .zip(coeff_error.chunks_exact_mut(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        for (chunk_x, ((coeff8, inv8), error8)) in coeff_row
            .as_chunks::<8>()
            .0
            .iter()
            .zip(inv_row.as_chunks::<8>().0.iter())
            .zip(error_row.as_chunks_mut::<8>().0.iter_mut())
            .enumerate()
        {
            let x = chunk_x * 8;
            let threshold = if x + 8 <= half {
                _mm256_set1_ps(thresholds[yfix])
            } else if x >= half {
                _mm256_set1_ps(thresholds[yfix + 1])
            } else {
                let lane_x = _mm256_add_epi32(_mm256_set1_epi32(x as i32), lane_ids);
                let high = _mm256_cmpgt_epi32(lane_x, _mm256_set1_epi32(half as i32 - 1));
                _mm256_blendv_ps(
                    _mm256_set1_ps(thresholds[yfix]),
                    _mm256_set1_ps(thresholds[yfix + 1]),
                    _mm256_castsi256_ps(high),
                )
            };
            let coeff_v = unsafe { _mm256_loadu_ps(coeff8.as_ptr()) };
            let inv_v = unsafe { _mm256_loadu_ps(inv8.as_ptr()) };
            let denominator = _mm256_mul_ps(inv_v, scale);
            let scaled = _mm256_mul_ps(denominator, coeff_v);
            let absolute = _mm256_andnot_ps(sign, scaled);
            let keep = _mm256_cmp_ps::<_CMP_GE_OQ>(absolute, threshold);
            let quantized = _mm256_and_ps(_mm256_round_ps::<ROUND>(scaled), keep);
            let error = _mm256_div_ps(_mm256_sub_ps(scaled, quantized), denominator);

            let active_i = if y < cy && x < cx {
                let lane_x = _mm256_add_epi32(_mm256_set1_epi32(x as i32), lane_ids);
                _mm256_cmpgt_epi32(lane_x, _mm256_set1_epi32(cx as i32 - 1))
            } else {
                all
            };
            let active = _mm256_castsi256_ps(active_i);
            unsafe {
                _mm256_storeu_ps(error8.as_mut_ptr(), _mm256_and_ps(error, active));
            }
            let active_quantized = _mm256_and_ps(quantized, active);
            let absolute_q = _mm256_andnot_ps(sign, active_quantized);
            let nonzero_mask = _mm256_cmp_ps::<_CMP_GT_OQ>(absolute_q, _mm256_setzero_ps());
            nonzero += _mm256_movemask_ps(nonzero_mask).count_ones() as usize;
            magnitude_bits = _mm256_add_ps(
                magnitude_bits,
                _mm256_and_ps(avx2_log2p1_f32(absolute_q), nonzero_mask),
            );
            let sv = unsafe {
                _mm256_loadu_si256(scan_pos.as_ptr().add(y * width + x) as *const __m256i)
            };
            scan_acc = _mm256_max_epu32(
                scan_acc,
                _mm256_and_si256(sv, _mm256_castps_si256(nonzero_mask)),
            );
        }
    }

    let v_max = _mm_max_epu32(
        _mm256_castsi256_si128(scan_acc),
        _mm256_extracti128_si256::<1>(scan_acc),
    );
    let max_scan = hmax_u32(v_max);

    let header_input = _mm256_setr_ps(nonzero as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let header = _mm_cvtss_f32(_mm256_castps256_ps128(avx2_log2p1_f32(header_input)));
    nonzero as f32 * 1.6
        + hsum256(magnitude_bits)
        + 0.4 * header
        + crate::inflated_cost::R_ZERO
            * crate::inflated_cost::visited_zeros(nonzero, max_scan, cx, cy)
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
fn prepare_reconstruction_avx2(
    plane: &Plane<f32>,
    px: usize,
    py: usize,
    pixel_width: usize,
    pixel_height: usize,
    error: &[f32],
    original: &mut [f32],
    reconstructed: &mut [f32],
) {
    let n = pixel_width
        .checked_mul(pixel_height)
        .expect("reconstruction size overflow");
    assert!(pixel_width.is_multiple_of(8));
    assert!(error.len() >= n && original.len() >= n && reconstructed.len() >= n);
    let (image_width, image_height) = (plane.xsize(), plane.ysize());
    assert!(image_width != 0 && image_height != 0);
    let source_x = px.min(image_width - 1);
    let copied = pixel_width.min(image_width - source_x);
    for (y, ((original_row, reconstructed_row), error_row)) in original
        .chunks_exact_mut(pixel_width)
        .zip(reconstructed.chunks_exact_mut(pixel_width))
        .zip(error.chunks_exact(pixel_width))
        .take(pixel_height)
        .enumerate()
    {
        let source = plane.row(py.saturating_add(y).min(image_height - 1));
        original_row[..copied].copy_from_slice(&source[source_x..source_x + copied]);
        original_row[copied..].fill(source[image_width - 1]);
        for ((output, value), delta) in reconstructed_row
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(original_row.as_chunks::<8>().0.iter())
            .zip(error_row.as_chunks::<8>().0.iter())
        {
            let values = unsafe { _mm256_loadu_ps(value.as_ptr()) };
            let errors = unsafe { _mm256_loadu_ps(delta.as_ptr()) };
            unsafe { _mm256_storeu_ps(output.as_mut_ptr(), _mm256_sub_ps(values, errors)) };
        }
    }
}

/// # Safety
/// The caller must ensure AVX2 is available. Dimensions and slice lengths are
/// checked before vector access.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn ssim_deficit_avx2(orig: &[f32], recon: &[f32], width: usize, height: usize) -> f32 {
    validate_ssim_inputs(orig, recon, width, height);
    const C1: f32 = 1e-4;
    const C2: f32 = 9e-4;
    const INV_64: f32 = 1.0 / 64.0;
    let mut deficit = 0.0f32;
    for wy in 0..height / 8 {
        for wx in 0..width / 8 {
            let (x0, y0) = (wx * 8, wy * 8);
            let base_index = y0 * width + x0;
            let base_o = orig[base_index];
            let base_r = recon[base_index];
            let base_ov = _mm256_set1_ps(base_o);
            let base_rv = _mm256_set1_ps(base_r);
            let (mut sum_o, mut sum_r) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for (orig_row, recon_row) in orig
                .chunks_exact(width)
                .zip(recon.chunks_exact(width))
                .skip(y0)
                .take(8)
            {
                sum_o = _mm256_add_ps(
                    sum_o,
                    _mm256_sub_ps(unsafe { _mm256_loadu_ps(orig_row[x0..].as_ptr()) }, base_ov),
                );
                sum_r = _mm256_add_ps(
                    sum_r,
                    _mm256_sub_ps(
                        unsafe { _mm256_loadu_ps(recon_row[x0..].as_ptr()) },
                        base_rv,
                    ),
                );
            }
            let mean_o = base_o + hsum256(sum_o) * INV_64;
            let mean_r = base_r + hsum256(sum_r) * INV_64;
            let mean_ov = _mm256_set1_ps(mean_o);
            let mean_rv = _mm256_set1_ps(mean_r);
            let (mut var_o, mut var_r, mut cov) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            for (orig_row, recon_row) in orig
                .chunks_exact(width)
                .zip(recon.chunks_exact(width))
                .skip(y0)
                .take(8)
            {
                let o = _mm256_sub_ps(unsafe { _mm256_loadu_ps(orig_row[x0..].as_ptr()) }, mean_ov);
                let r = _mm256_sub_ps(
                    unsafe { _mm256_loadu_ps(recon_row[x0..].as_ptr()) },
                    mean_rv,
                );
                var_o = _mm256_fmadd_ps(o, o, var_o);
                var_r = _mm256_fmadd_ps(r, r, var_r);
                cov = _mm256_fmadd_ps(o, r, cov);
            }
            let vo = hsum256(var_o) * INV_64;
            let vr = hsum256(var_r) * INV_64;
            let covariance = hsum256(cov) * INV_64;
            let luminance = (2.0 * mean_o * mean_r + C1) / (mean_o * mean_o + mean_r * mean_r + C1);
            let structure = (2.0 * covariance + C2) / (vo + vr + C2);
            deficit += (1.0 - luminance * structure) * 64.0;
        }
    }
    deficit
}

#[cfg(test)]
mod tests {
    use super::{
        combine_error_avx2, error_gradient_energy_avx2, recon_dist_and_rate_avx2, ssim_deficit_avx2,
    };
    use crate::dc_group_data::{
        STRATEGY_DCT, STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32,
        STRATEGY_DCT32X16, STRATEGY_DCT32X32,
    };
    use crate::image::Image3F;
    use crate::inflated_cost::{
        ReconDistInput, ReconErrorKernels, ReconQuantization, ReconScoring, ReconSource,
        ReconTransform, rate_log2_lut, recon_dist_and_rate_scalar, ssim_deficit_scalar,
    };

    fn available() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    #[test]
    fn ssim_avx2_matches_scalar() {
        if !available() {
            return;
        }
        let mut state = 91u32;
        for (width, height) in [(8, 8), (16, 8), (8, 16), (16, 16), (32, 32)] {
            let n = width * height;
            let mut orig = vec![0.0f32; n];
            let mut recon = vec![0.0f32; n];
            for i in 0..n {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                orig[i] = (state >> 8) as f32 / (1u32 << 24) as f32;
                recon[i] = orig[i] + ((i % 13) as f32 - 6.0) * 0.0007;
            }
            let scalar = ssim_deficit_scalar(&orig, &recon, width, height);
            let simd = unsafe { ssim_deficit_avx2(&orig, &recon, width, height) };
            let tolerance = 2e-4f32.max(scalar.abs() * 2e-5);
            assert!(
                (simd - scalar).abs() <= tolerance,
                "{width}x{height}: simd={simd} scalar={scalar}"
            );
        }
    }

    #[test]
    fn recon_avx2_matches_scalar_for_all_standard_transforms() {
        if !available() {
            return;
        }
        let mut image = Image3F::new(40, 40);
        for c in 0..3 {
            for y in 0..40 {
                for x in 0..40 {
                    image.plane_mut(c).row_mut(y)[x] =
                        0.15 * c as f32 + 0.003 * x as f32 + 0.002 * y as f32;
                }
            }
        }
        let mut coeffs = [[0.0f32; 1024]; 3];
        let mut state = 17u32;
        for channel in &mut coeffs {
            for value in channel {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *value = ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0;
            }
        }
        let inv_storage = [vec![1.0f32; 1024], vec![0.9f32; 1024], vec![1.1f32; 1024]];
        for (strategy, cx, cy) in [
            (STRATEGY_DCT, 1, 1),
            (STRATEGY_DCT8X16, 2, 1),
            (STRATEGY_DCT16X8, 2, 1),
            (STRATEGY_DCT16X16, 2, 2),
            (STRATEGY_DCT16X32, 4, 2),
            (STRATEGY_DCT32X16, 4, 2),
            (STRATEGY_DCT32X32, 4, 4),
        ] {
            let inv = [
                &inv_storage[0][..],
                &inv_storage[1][..],
                &inv_storage[2][..],
            ];
            let input = ReconDistInput {
                quantization: ReconQuantization {
                    rate_log2_lut: rate_log2_lut(),
                    coeffs: &coeffs,
                    inverse_matrices: inv,
                    qac: 7.0,
                    qm_mult_x: 1.2,
                    distance: 1.5,
                },
                transform: ReconTransform {
                    blocks_x: cx,
                    blocks_y: cy,
                    strategy,
                },
                source: ReconSource {
                    opsin: &image,
                    x: 3,
                    y: 5,
                },
                scoring: ReconScoring {
                    factor_x: 0.15,
                    factor_b: -0.1,
                    banding: true,
                },
            };
            let mut scalar_scratch = [[0.0f32; 1024]; 8];
            let scalar = recon_dist_and_rate_scalar(&mut scalar_scratch, &input);
            let mut simd_scratch = [[0.0f32; 1024]; 8];
            let simd = unsafe {
                recon_dist_and_rate_avx2(
                    &mut simd_scratch,
                    &input,
                    &ReconErrorKernels {
                        gradient_energy: |error, width, height| {
                            error_gradient_energy_avx2(error, width, height)
                        },
                        combine: |spatial, luma, factor, combined| {
                            combine_error_avx2(spatial, luma, factor, combined)
                        },
                    },
                )
            };
            let rate_tolerance = 2e-4f32.max(scalar.1.abs() * 3e-6);
            assert!(
                (simd.1 - scalar.1).abs() <= rate_tolerance,
                "strategy {strategy} rate: simd={} scalar={}",
                simd.1,
                scalar.1
            );
            let tolerance = 5e-4f32.max(scalar.0.abs() * 3e-5);
            assert!(
                (simd.0 - scalar.0).abs() <= tolerance,
                "strategy {strategy}: simd={simd:?} scalar={scalar:?}"
            );
        }
    }

    #[test]
    fn ssim_avx2_rejects_short_slices_before_loading() {
        if !available() {
            return;
        }
        let result =
            std::panic::catch_unwind(|| unsafe { ssim_deficit_avx2(&[0.0; 63], &[0.0; 64], 8, 8) });
        assert!(result.is_err());
    }
}
