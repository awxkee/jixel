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

use super::ac_strategy::neon_log2p1_f32;
use crate::image::Plane;
use crate::inflated_cost::{
    RateLog2Lut, ReconDistInput, ReconErrorKernels, ReconKernels, recon_dist_and_rate_with_kernels,
    validate_ssim_inputs,
};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_gradient_vectors_x4(
    left: float32x4_t,
    right: float32x4_t,
    sum: float32x4_t,
) -> float32x4_t {
    let difference = vsubq_f32(right, left);
    vfmaq_f32(sum, difference, difference)
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_gradient_x4(left: &[f32; 4], right: &[f32; 4], sum: float32x4_t) -> float32x4_t {
    let left = unsafe { vld1q_f32(left.as_ptr()) };
    let right = unsafe { vld1q_f32(right.as_ptr()) };
    accumulate_gradient_vectors_x4(left, right, sum)
}

/// # Safety
/// The caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub(crate) fn error_gradient_energy_neon(error: &[f32], width: usize, height: usize) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("gradient plane size overflow");
    assert!(error.len() >= n);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let rows = error[..n].chunks_exact(width);
    let mut sum = vdupq_n_f32(0.0);

    for row in rows.clone() {
        let (row4, row_tail) = row.as_chunks::<4>();
        if row_tail.is_empty() {
            let (right4, _) = row[1..].as_chunks::<4>();
            for (left, right) in row4.iter().zip(right4) {
                sum = accumulate_gradient_x4(left, right, sum);
            }
            let left = unsafe { vld1q_f32(row4.last().unwrap().as_ptr()) };
            let right = vsetq_lane_f32::<3>(vgetq_lane_f32::<3>(left), vextq_f32::<1>(left, left));
            sum = accumulate_gradient_vectors_x4(left, right, sum);
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
    vaddvq_f32(sum)
}

#[inline]
#[target_feature(enable = "neon")]
fn peak_excess_x4(
    a: float32x4_t,
    b: float32x4_t,
    source_a: float32x4_t,
    source_b: float32x4_t,
    floor: float32x4_t,
) -> float32x4_t {
    let error_gradient = vabsq_f32(vsubq_f32(b, a));
    let source_gradient = vabsq_f32(vsubq_f32(source_b, source_a));
    let excess = vmaxq_f32(
        vdupq_n_f32(0.0),
        vsubq_f32(
            vsubq_f32(error_gradient, vmulq_n_f32(source_gradient, 0.5)),
            floor,
        ),
    );
    vmulq_f32(excess, excess)
}

/// # Safety
/// The caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub(crate) fn error_gradient_peak_energy_neon(
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

    let floor = vdupq_n_f32(floor);
    let zero = vdupq_n_f32(0.0);
    let mut total = 0.0f32;
    for cell_y in (0..height).step_by(4) {
        for cell_x in (0..width).step_by(4) {
            let mut max_x = zero;
            let mut max_y = zero;
            for y in cell_y..(cell_y + 4).min(height) {
                let p = y * width + cell_x;
                let current = unsafe { vld1q_f32(error.as_ptr().add(p)) };
                let source = unsafe { vld1q_f32(original.as_ptr().add(p)) };
                let (right, source_right) = if cell_x + 4 < width {
                    (unsafe { vld1q_f32(error.as_ptr().add(p + 1)) }, unsafe {
                        vld1q_f32(original.as_ptr().add(p + 1))
                    })
                } else {
                    (
                        vsetq_lane_f32::<3>(
                            vgetq_lane_f32::<3>(current),
                            vextq_f32::<1>(current, current),
                        ),
                        vsetq_lane_f32::<3>(
                            vgetq_lane_f32::<3>(source),
                            vextq_f32::<1>(source, source),
                        ),
                    )
                };
                max_x = vmaxq_f32(
                    max_x,
                    peak_excess_x4(current, right, source, source_right, floor),
                );
                if y + 1 < height {
                    let below = unsafe { vld1q_f32(error.as_ptr().add(p + width)) };
                    let source_below = unsafe { vld1q_f32(original.as_ptr().add(p + width)) };
                    max_y = vmaxq_f32(
                        max_y,
                        peak_excess_x4(current, below, source, source_below, floor),
                    );
                }
            }
            total += vmaxvq_f32(max_x) + vmaxvq_f32(max_y);
        }
    }
    total
}

#[inline]
#[target_feature(enable = "neon")]
fn combine_error_x4(spatial: &[f32; 4], luma: &[f32; 4], factor: f32, combined: &mut [f32; 4]) {
    let spatial = unsafe { vld1q_f32(spatial.as_ptr()) };
    let luma = unsafe { vld1q_f32(luma.as_ptr()) };
    let value = vfmaq_n_f32(spatial, luma, factor);
    unsafe { vst1q_f32(combined.as_mut_ptr(), value) };
}

/// # Safety
/// The caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub(crate) fn combine_error_neon(spatial: &[f32], luma: &[f32], factor: f32, combined: &mut [f32]) {
    debug_assert_eq!(spatial.len(), luma.len());
    debug_assert_eq!(spatial.len(), combined.len());
    let (spatial16, spatial_tail) = spatial.as_chunks::<16>();
    let (luma16, luma_tail) = luma.as_chunks::<16>();
    let (combined16, combined_tail) = combined.as_chunks_mut::<16>();

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

/// # Safety
/// AArch64 NEON must be available. All slice bounds are validated before loads.
#[target_feature(enable = "neon")]
pub(crate) fn recon_dist_and_rate_neon(
    scratch: &mut [[f32; 1024]; 8],
    input: &ReconDistInput<'_>,
    error: &ReconErrorKernels,
) -> (f32, f32) {
    recon_dist_and_rate_with_kernels(
        scratch,
        input,
        &ReconKernels {
            quantize: recon_quantize_neon,
            ssim: ssim_deficit_neon,
            prepare: prepare_reconstruction_neon,
            error,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn recon_quantize_neon(
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
    assert!(width.is_multiple_of(4) && half.is_multiple_of(4));
    assert!(coeff.len() >= n && inv.len() >= n && coeff_error.len() >= n && scan_pos.len() >= n);
    let scale = vdupq_n_f32(quant_scale);
    let zero = vdupq_n_f32(0.0);
    let all = vdupq_n_u32(u32::MAX);
    let lanes = unsafe { vld1q_u32([0, 1, 2, 3].as_ptr()) };
    let mut nonzero = 0usize;
    let mut magnitude_bits = vdupq_n_f32(0.0);
    let mut scan_acc = vdupq_n_u32(0);
    for (y, (((coeff_row, inv_row), error_row), scan_row)) in coeff
        .chunks_exact(width)
        .zip(inv.chunks_exact(width))
        .zip(coeff_error.chunks_exact_mut(width))
        .zip(scan_pos.chunks_exact(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        for (chunk_x, (((coeff4, inv4), error4), scan4)) in coeff_row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(inv_row.as_chunks::<4>().0.iter())
            .zip(error_row.as_chunks_mut::<4>().0.iter_mut())
            .zip(scan_row.as_chunks::<4>().0.iter())
            .enumerate()
        {
            let x = chunk_x * 4;
            let threshold = vdupq_n_f32(if x >= half {
                thresholds[yfix + 1]
            } else {
                thresholds[yfix]
            });
            let coeff_v = unsafe { vld1q_f32(coeff4.as_ptr()) };
            let inv_v = unsafe { vld1q_f32(inv4.as_ptr()) };
            let denominator = vmulq_f32(inv_v, scale);
            let scaled = vmulq_f32(denominator, coeff_v);
            let keep = vcgeq_f32(vabsq_f32(scaled), threshold);
            // Ties-away rounding, matching the real quantizer (vcvtaq/fast_round).
            let quantized =
                vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vrndaq_f32(scaled)), keep));
            let error = vdivq_f32(vsubq_f32(scaled, quantized), denominator);
            let active = if y < cy && x < cx {
                vcgeq_u32(
                    vaddq_u32(vdupq_n_u32(x as u32), lanes),
                    vdupq_n_u32(cx as u32),
                )
            } else {
                all
            };
            unsafe { vst1q_f32(error4.as_mut_ptr(), vbslq_f32(active, error, zero)) };
            let active_quantized = vbslq_f32(active, quantized, zero);
            let absolute_q = vabsq_f32(active_quantized);
            let nonzero_mask = vcgtq_f32(absolute_q, zero);
            nonzero += vaddvq_u32(vshrq_n_u32::<31>(nonzero_mask)) as usize;
            magnitude_bits = vaddq_f32(
                magnitude_bits,
                vbslq_f32(nonzero_mask, neon_log2p1_f32(absolute_q), zero),
            );
            let sv = unsafe { vld1q_u32(scan4.as_ptr()) };
            scan_acc = vmaxq_u32(scan_acc, vandq_u32(sv, nonzero_mask));
        }
    }
    let header = vgetq_lane_f32::<0>(neon_log2p1_f32(vsetq_lane_f32::<0>(
        nonzero as f32,
        vdupq_n_f32(0.0),
    )));
    nonzero as f32 * 1.6
        + vaddvq_f32(magnitude_bits)
        + 0.4 * header
        + crate::inflated_cost::R_ZERO
            * crate::inflated_cost::visited_zeros(nonzero, vmaxvq_u32(scan_acc), cx, cy)
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn prepare_reconstruction_neon(
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
    assert!(pixel_width.is_multiple_of(4));
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
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(original_row.as_chunks::<4>().0.iter())
            .zip(error_row.as_chunks::<4>().0.iter())
        {
            let values = unsafe { vld1q_f32(value.as_ptr()) };
            let errors = unsafe { vld1q_f32(delta.as_ptr()) };
            unsafe { vst1q_f32(output.as_mut_ptr(), vsubq_f32(values, errors)) };
        }
    }
}

/// # Safety
/// AArch64 NEON must be available. Dimensions and slice lengths are validated.
#[target_feature(enable = "neon")]
pub(crate) fn ssim_deficit_neon(orig: &[f32], recon: &[f32], width: usize, height: usize) -> f32 {
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
            let base_ov = vdupq_n_f32(base_o);
            let base_rv = vdupq_n_f32(base_r);
            let (mut sum_o, mut sum_r) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            for (orig_row, recon_row) in orig
                .chunks_exact(width)
                .zip(recon.chunks_exact(width))
                .skip(y0)
                .take(8)
            {
                let orig8 = &orig_row.as_chunks::<8>().0[wx];
                let recon8 = &recon_row.as_chunks::<8>().0[wx];
                for (orig4, recon4) in orig8
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(recon8.as_chunks::<4>().0)
                {
                    sum_o = vaddq_f32(
                        sum_o,
                        vsubq_f32(unsafe { vld1q_f32(orig4.as_ptr()) }, base_ov),
                    );
                    sum_r = vaddq_f32(
                        sum_r,
                        vsubq_f32(unsafe { vld1q_f32(recon4.as_ptr()) }, base_rv),
                    );
                }
            }
            let mean_o = f32::mul_add(vaddvq_f32(sum_o), INV_64, base_o);
            let mean_r = f32::mul_add(vaddvq_f32(sum_r), INV_64, base_r);
            let mean_ov = vdupq_n_f32(mean_o);
            let mean_rv = vdupq_n_f32(mean_r);
            let (mut var_o, mut var_r, mut cov) =
                (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            for (orig_row, recon_row) in orig
                .chunks_exact(width)
                .zip(recon.chunks_exact(width))
                .skip(y0)
                .take(8)
            {
                let orig8 = &orig_row.as_chunks::<8>().0[wx];
                let recon8 = &recon_row.as_chunks::<8>().0[wx];
                for (orig4, recon4) in orig8
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(recon8.as_chunks::<4>().0)
                {
                    let o = vsubq_f32(unsafe { vld1q_f32(orig4.as_ptr()) }, mean_ov);
                    let r = vsubq_f32(unsafe { vld1q_f32(recon4.as_ptr()) }, mean_rv);
                    var_o = vfmaq_f32(var_o, o, o);
                    var_r = vfmaq_f32(var_r, r, r);
                    cov = vfmaq_f32(cov, o, r);
                }
            }
            let vo = vaddvq_f32(var_o) * INV_64;
            let vr = vaddvq_f32(var_r) * INV_64;
            let covariance = vaddvq_f32(cov) * INV_64;
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
        combine_error_neon, error_gradient_energy_neon, error_gradient_peak_energy_neon,
        recon_dist_and_rate_neon, ssim_deficit_neon,
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

    #[test]
    fn ssim_neon_matches_scalar() {
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
            let simd = unsafe { ssim_deficit_neon(&orig, &recon, width, height) };
            let tolerance = 2e-4f32.max(scalar.abs() * 2e-5);
            assert!(
                (simd - scalar).abs() <= tolerance,
                "{width}x{height}: simd={simd} scalar={scalar}"
            );
        }
    }

    #[test]
    fn recon_neon_matches_scalar_for_all_standard_transforms() {
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
        let idct = crate::dct::IdctMethods::scalar();
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
                idct: &idct,
                quantization: ReconQuantization {
                    rate_log2_lut: rate_log2_lut(),
                    coeffs: [&coeffs[0], &coeffs[1], &coeffs[2]],
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
                    gradient_alpha: 0.0,
                    gradient_peak_alpha: 3.0,
                },
            };
            let mut scalar_scratch = [[0.0f32; 1024]; 8];
            let scalar = recon_dist_and_rate_scalar(&mut scalar_scratch, &input);
            let mut simd_scratch = [[0.0f32; 1024]; 8];
            let simd = unsafe {
                recon_dist_and_rate_neon(
                    &mut simd_scratch,
                    &input,
                    &ReconErrorKernels {
                        gradient_energy: |error, width, height| {
                            error_gradient_energy_neon(error, width, height)
                        },
                        gradient_peak_energy: |error, original, width, height, floor| {
                            error_gradient_peak_energy_neon(error, original, width, height, floor)
                        },
                        combine: |spatial, luma, factor, combined| {
                            combine_error_neon(spatial, luma, factor, combined)
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
    fn ssim_neon_rejects_short_slices_before_loading() {
        let result =
            std::panic::catch_unwind(|| unsafe { ssim_deficit_neon(&[0.0; 63], &[0.0; 64], 8, 8) });
        assert!(result.is_err());
    }
}
