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

use crate::adaptive_quant::fast_exp2;
use crate::dct::{DctFn, DctInput, fmla};
use crate::image::{Image3F, ImageB};

pub(crate) type ApplyCorrectionsFn = fn(&[f32], &mut ImageB, f32, f32, f32);
pub(crate) type BlockFeaturesFn = fn(&[f32; 64], &DctFn<8, 8, 64>) -> Features;

#[allow(dead_code)]
pub(crate) fn apply_corrections_scalar(
    corrections: &[f32],
    field: &mut ImageB,
    amount: f32,
    center: f32,
    inv_stddev: f32,
) {
    debug_assert_eq!(corrections.len(), field.as_slice().len());
    for (q, &correction) in field.as_mut_slice().iter_mut().zip(corrections) {
        let delta = (-amount * (correction - center) * inv_stddev).clamp(-0.18, 0.22);
        *q = (*q as f32 * fast_exp2(delta)).round().clamp(1.0, 255.0) as u8;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_apply_corrections_fn() -> ApplyCorrectionsFn {
    crate::wasm::apply_structure_aq_wasm
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_block_features_fn() -> BlockFeaturesFn {
    crate::wasm::block_features_wasm
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_block_features_fn() -> BlockFeaturesFn {
    |block, dct| unsafe { crate::neon::block_features_neon(block, dct) }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_block_features_fn() -> BlockFeaturesFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |block, dct| unsafe { crate::avx::block_features_avx2(block, dct) };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |block, dct| unsafe { crate::sse::block_features_sse41(block, dct) };
    }
    block_features_scalar
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_apply_corrections_fn() -> ApplyCorrectionsFn {
    |corrections, field, amount, center, inv_stddev| unsafe {
        crate::neon::apply_structure_aq_neon(corrections, field, amount, center, inv_stddev)
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_apply_corrections_fn() -> ApplyCorrectionsFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |corrections, field, amount, center, inv_stddev| unsafe {
            crate::avx::apply_structure_aq_avx2(corrections, field, amount, center, inv_stddev)
        };
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |corrections, field, amount, center, inv_stddev| unsafe {
            crate::sse::apply_structure_aq_sse41(corrections, field, amount, center, inv_stddev)
        };
    }

    apply_corrections_scalar
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Features {
    pub(crate) persistence: f32,
    pub(crate) coherence: f32,
    pub(crate) predictability: f32,
    pub(crate) mid_share: f32,
    pub(crate) high_share: f32,
    pub(crate) gradient_presence: f32,
}

#[inline]
fn strength(distance: f32) -> f32 {
    static POINTS: &[(f32, f32)] = &[
        (1.5, 0.0),
        (2.0, 0.08),
        (3.0, 0.13),
        (4.0, 0.12),
        (4.5, 0.06),
        (5.0, 0.0),
    ];
    if distance <= POINTS[0].0 {
        return 0.0;
    }
    for pair in POINTS.array_windows::<2>() {
        let (d0, v0) = pair[0];
        let (d1, v1) = pair[1];
        if distance <= d1 {
            return v0 + (distance - d0) * (v1 - v0) / (d1 - d0);
        }
    }
    POINTS[POINTS.len() - 1].1
}

pub(crate) const FEATURE_EPS: f32 = 1.0e-10;

#[inline]
fn gradient_sparsity(block: &[f32; 64]) -> f32 {
    let rows = block.as_chunks::<8>().0;
    let mut sum = 0.0f32;
    let mut sum_squared = 0.0f32;
    for row_window in rows.array_windows::<3>() {
        let [top, middle, bottom] = row_window;
        for ((&left, &right), (&top, &bottom)) in middle[..6]
            .iter()
            .zip(&middle[2..])
            .zip(top[1..7].iter().zip(&bottom[1..7]))
        {
            let gx = 0.5 * (right - left);
            let gy = 0.5 * (bottom - top);
            let energy = fmla(gx, gx, gy * gy);
            sum += energy;
            sum_squared = fmla(energy, energy, sum_squared);
        }
    }
    ((36.0 * sum_squared / (sum * sum + FEATURE_EPS) - 1.0) * 0.5).clamp(0.0, 1.0)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn finish_block_features(
    variance: f32,
    tensor: [f32; 3],
    errors: [f32; 5],
    energy_1x: f32,
    energy_2x: f32,
    mid: f32,
    high: f32,
) -> Features {
    let [jxx, jxy, jyy] = tensor;
    let trace = jxx + jyy;
    let coherence = (((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy).sqrt() / (trace + FEATURE_EPS))
        .clamp(0.0, 1.0);
    let gradient_presence = (trace / (36.0 * variance + FEATURE_EPS)).clamp(0.0, 1.0);
    let min_error = errors.into_iter().fold(f32::INFINITY, f32::min) * (1.0 / 36.0);
    let predictability = (1.0 - min_error / (variance + FEATURE_EPS)).clamp(0.0, 1.0);
    let energy_1x = energy_1x * (1.0 / 112.0);
    let energy_2x = energy_2x * (1.0 / 24.0);
    let persistence_ratio = energy_2x / (energy_1x + FEATURE_EPS);
    let persistence = (persistence_ratio / (1.0 + persistence_ratio)).clamp(0.0, 1.0);
    let spectral = mid + high + FEATURE_EPS;
    Features {
        persistence,
        coherence,
        predictability,
        mid_share: mid / spectral,
        high_share: high / spectral,
        gradient_presence,
    }
}

#[inline]
#[allow(dead_code)]
fn moments_scalar(block: &[f32; 64]) -> (f32, f32) {
    let mean = block.iter().sum::<f32>() * (1.0 / 64.0);
    let variance = block
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        * (1.0 / 64.0);
    (mean, variance)
}

#[allow(dead_code)]
fn tensor_features_scalar(block: &[f32; 64], variance: f32) -> (f32, f32) {
    let rows = block.as_chunks::<8>().0;
    let (mut jxx, mut jxy, mut jyy) = (0.0f32, 0.0f32, 0.0f32);
    for row_window in rows.array_windows::<3>() {
        let [top, middle, bottom] = row_window;
        for ((&left, &right), (&top, &bottom)) in middle[..6]
            .iter()
            .zip(&middle[2..])
            .zip(top[1..7].iter().zip(&bottom[1..7]))
        {
            let gx = 0.5 * (right - left);
            let gy = 0.5 * (bottom - top);
            jxx = fmla(gx, gx, jxx);
            jxy = fmla(gx, gy, jxy);
            jyy = fmla(gy, gy, jyy);
        }
    }
    let trace = jxx + jyy;
    let coherence = (((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy).sqrt() / (trace + FEATURE_EPS))
        .clamp(0.0, 1.0);
    let gradient_presence = (trace / (36.0 * variance + FEATURE_EPS)).clamp(0.0, 1.0);
    (coherence, gradient_presence)
}

#[allow(dead_code)]
fn predictability_scalar(block: &[f32; 64], variance: f32) -> f32 {
    let rows = block.as_chunks::<8>().0;
    let mut errors = [0.0f32; 5];
    for row_window in rows.array_windows::<3>() {
        let [top_row, middle, bottom_row] = row_window;
        for ((((&left, &v), &top), &top_left), &bottom_left) in middle[..6]
            .iter()
            .zip(&middle[1..7])
            .zip(&top_row[1..7])
            .zip(&top_row[..6])
            .zip(&bottom_row[..6])
        {
            for (slot, prediction) in [left, top, top_left, bottom_left, left + top - top_left]
                .into_iter()
                .enumerate()
            {
                let e = v - prediction;
                errors[slot] = fmla(e, e, errors[slot]);
            }
        }
    }
    let min_error = errors.into_iter().fold(f32::INFINITY, f32::min) * (1.0 / 36.0);
    (1.0 - min_error / (variance + FEATURE_EPS)).clamp(0.0, 1.0)
}

#[allow(dead_code)]
fn gradient_energy_scalar(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let mut energy = 0.0f32;
    for row in values.chunks_exact(stride).take(height) {
        for pair in row[..width].array_windows::<2>() {
            let d = pair[1] - pair[0];
            energy = fmla(d, d, energy);
        }
    }
    for y in 0..height - 1 {
        let top = &values[y * stride..][..width];
        let bottom = &values[(y + 1) * stride..][..width];
        for (&top, &bottom) in top.iter().zip(bottom) {
            let d = bottom - top;
            energy = fmla(d, d, energy);
        }
    }
    energy
}

#[inline]
#[allow(dead_code)]
fn downsample_8x8_scalar(block: &[f32; 64]) -> [f32; 16] {
    let mut half = [0.0; 16];
    for y in 0..4 {
        for x in 0..4 {
            let src = 2 * y * 8 + 2 * x;
            half[y * 4 + x] =
                0.25 * (block[src] + block[src + 1] + block[src + 8] + block[src + 9]);
        }
    }
    half
}

#[inline]
#[allow(dead_code)]
fn spectral_shares_scalar(coeffs: &[f32; 64]) -> (f32, f32) {
    let (mut mid, mut high) = (0.0, 0.0);
    for (y, row) in coeffs.as_chunks::<8>().0.iter().enumerate() {
        let mid_start = 2usize.saturating_sub(y);
        let mid_end = (8 - y).min(8);
        for &coeff in &row[mid_start..mid_end] {
            mid = fmla(coeff, coeff, mid);
        }
        let high_start = 8usize.saturating_sub(y);
        for &coeff in &row[high_start..] {
            high = fmla(coeff, coeff, high);
        }
    }
    let spectral = mid + high + FEATURE_EPS;
    (mid / spectral, high / spectral)
}

#[allow(dead_code)]
fn block_features_scalar(block: &[f32; 64], dct8x8: &DctFn<8, 8, 64>) -> Features {
    let (_, variance) = moments_scalar(block);
    let (coherence, gradient_presence) = tensor_features_scalar(block, variance);
    let predictability = predictability_scalar(block, variance);
    let energy_1x = gradient_energy_scalar(block, 8, 8, 8) * (1.0 / 112.0);
    let half = downsample_8x8_scalar(block);
    let energy_2x = gradient_energy_scalar(&half, 4, 4, 4) * (1.0 / 24.0);
    let persistence_ratio = energy_2x / (energy_1x + FEATURE_EPS);
    let persistence = (persistence_ratio / (1.0 + persistence_ratio)).clamp(0.0, 1.0);

    let mut coeffs = [0.0f32; 64];
    dct8x8(DctInput::from_flat(block), &mut coeffs);
    let (mid_share, high_share) = spectral_shares_scalar(&coeffs);
    Features {
        persistence,
        coherence,
        predictability,
        mid_share,
        high_share,
        gradient_presence,
    }
}

#[inline]
fn ringing_mix(distance: f32) -> f32 {
    (if distance <= 4.0 {
        1.0 - 2.0 * (4.0 - distance)
    } else {
        1.0 - 3.0 * (distance - 4.0)
    })
    .clamp(0.0, 1.0)
}

#[inline]
fn correction_score(f: Features, distance: f32, ringing_mix: f32, gradient_sparsity: f32) -> f32 {
    let structure =
        0.30 * f.persistence + 0.25 * f.coherence + 0.25 * f.predictability + 0.20 * f.mid_share;
    let noise =
        f.high_share * (1.0 - f.coherence) * (1.0 - f.predictability) * (1.0 - f.persistence);
    let weak_edge = f.coherence * f.predictability * f.gradient_presence;
    let banding = weak_edge * f.mid_share * (1.0 - f.high_share);
    let ringing = banding * gradient_sparsity;
    let low_quality_mix = ((distance - 3.5) / 2.5).clamp(0.0, 1.0);
    // A lower-than-center score receives a finer quantizer. Protect sparse
    // edges here; the centered correction funds them from more maskable blocks.
    structure + (0.20 + 0.35 * low_quality_mix) * weak_edge + 0.35 * low_quality_mix * banding
        - 0.50 * ringing_mix * ringing
        - (0.90 - 0.15 * low_quality_mix) * noise
}

pub(crate) fn apply(
    corrections: &mut Vec<f32>,
    opsin: &Image3F,
    field: &mut ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
    dct8x8: &DctFn<8, 8, 64>,
    block_features: BlockFeaturesFn,
    apply_corrections: ApplyCorrectionsFn,
) {
    let amount = strength(distance);
    if amount == 0.0 || field.xsize() == 0 || field.ysize() == 0 {
        return;
    }
    let field_width = field.xsize();
    let field_height = field.ysize();
    if corrections.len() < field_width * field_height {
        corrections.resize(field_width * field_height, 0.0);
    }
    let corrections = &mut corrections[..field_width * field_height];
    let ringing_mix = ringing_mix(distance);
    let mut weighted_sum = 0.0f32;
    let mut total_weight = 0.0f32;
    for (by, correction_row) in corrections.chunks_exact_mut(field_width).enumerate() {
        let py = y0 + by * 8;
        let h = opsin.ysize().saturating_sub(py).min(8);
        for (bx, correction_out) in correction_row.iter_mut().enumerate() {
            let px = x0 + bx * 8;
            let w = opsin.xsize().saturating_sub(px).min(8);
            let mut block = [0.0f32; 64];
            let block_rows = block.as_chunks_mut::<8>().0;
            if w == 8 && h == 8 {
                for (dst, src_y) in block_rows.iter_mut().zip(py..py + 8) {
                    let source = &opsin.plane_row(1, src_y)[px..];
                    dst.copy_from_slice(&source.as_chunks::<8>().0[0]);
                }
            } else {
                for (dy, dst) in block_rows.iter_mut().enumerate() {
                    let row = opsin.plane_row(1, (py + dy).min(opsin.ysize() - 1));
                    for (dx, out) in dst.iter_mut().enumerate() {
                        *out = row[(px + dx).min(opsin.xsize() - 1)];
                    }
                }
            }
            let sparsity = if ringing_mix == 0.0 {
                0.0
            } else {
                gradient_sparsity(&block)
            };
            let correction = correction_score(
                block_features(&block, dct8x8),
                distance,
                ringing_mix,
                sparsity,
            );
            *correction_out = correction;
            let weight = (w * h) as f32;
            weighted_sum = fmla(weight, correction, weighted_sum);
            total_weight += weight;
        }
    }
    let center = if total_weight == 0.0 {
        0.0
    } else {
        weighted_sum / total_weight
    };
    let mut weighted_variance = 0.0f32;
    for (by, correction_row) in corrections.chunks_exact(field_width).enumerate() {
        let py = y0 + by * 8;
        let h = opsin.ysize().saturating_sub(py).min(8);
        for (bx, &correction) in correction_row.iter().enumerate() {
            let px = x0 + bx * 8;
            let w = opsin.xsize().saturating_sub(px).min(8);
            let d = correction - center;
            weighted_variance = fmla((w * h) as f32, d * d, weighted_variance);
        }
    }
    let inv_stddev = if weighted_variance == 0.0 {
        0.0
    } else {
        (total_weight / weighted_variance).sqrt()
    };
    apply_corrections(corrections, field, amount, center, inv_stddev);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_block_features(method: BlockFeaturesFn) {
        let dct = crate::dct::selected_dct8x8();
        let mut blocks = Vec::new();
        blocks.push([0.0; 64]);
        blocks.push([0.37; 64]);
        let mut line = [0.0; 64];
        let mut checker = [0.0; 64];
        let mut random = [0.0; 64];
        let mut replicated_edge = [0.0; 64];
        let mut state = 0x9e37_79b9u32;
        for y in 0..8 {
            for x in 0..8 {
                line[y * 8 + x] = if x < 4 { 0.2 } else { 0.8 };
                checker[y * 8 + x] = if (x + y) & 1 == 0 { 0.2 } else { 0.8 };
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                random[y * 8 + x] = (state >> 8) as f32 / (1u32 << 24) as f32;
                let edge_x = x.min(4);
                let edge_y = y.min(2);
                replicated_edge[y * 8 + x] = (edge_y * 5 + edge_x) as f32 * (1.0 / 16.0);
            }
        }
        blocks.extend([line, checker, random, replicated_edge]);
        for (index, block) in blocks.iter().enumerate() {
            let expected = block_features_scalar(block, dct);
            let actual = method(block, dct);
            for (name, actual, expected) in [
                ("persistence", actual.persistence, expected.persistence),
                ("coherence", actual.coherence, expected.coherence),
                (
                    "predictability",
                    actual.predictability,
                    expected.predictability,
                ),
                ("mid_share", actual.mid_share, expected.mid_share),
                ("high_share", actual.high_share, expected.high_share),
                (
                    "gradient_presence",
                    actual.gradient_presence,
                    expected.gradient_presence,
                ),
            ] {
                let tolerance = 2e-6f32.max(expected.abs() * 2e-5);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "block {index} {name}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn selected_block_features_matches_scalar() {
        check_block_features(select_block_features_fn());
    }

    #[test]
    fn gradient_sparsity_separates_isolated_edges_from_distributed_activity() {
        let mut edge = [0.0f32; 64];
        let mut ramp = [0.0f32; 64];
        let mut checker = [0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                edge[y * 8 + x] = if x < 4 { 0.0 } else { 0.5 };
                ramp[y * 8 + x] = x as f32 / 7.0;
                checker[y * 8 + x] = ((x + y) & 1) as f32;
            }
        }
        assert!(gradient_sparsity(&edge) > 0.9);
        assert!(gradient_sparsity(&ramp) < 1.0e-5);
        assert!(gradient_sparsity(&checker) < 1.0e-5);
    }

    #[test]
    fn ringing_protection_is_confined_to_the_low_quality_window() {
        assert_eq!(ringing_mix(3.5), 0.0);
        assert_eq!(ringing_mix(4.0), 1.0);
        assert_eq!(ringing_mix(4.25), 0.25);
        assert_eq!(ringing_mix(4.5), 0.0);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_block_features_matches_scalar() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            check_block_features(|block, dct| unsafe {
                crate::avx::block_features_avx2(block, dct)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_block_features_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_block_features(|block, dct| unsafe {
                crate::sse::block_features_sse41(block, dct)
            });
        }
    }

    fn check_apply_corrections(method: ApplyCorrectionsFn) {
        let parameters = [
            (0.0, 0.0, 0.0),
            (0.08, 0.4, 0.5),
            (0.13, -0.2, 1.75),
            (0.12, 1.1, 8.0),
        ];
        for len in (0usize..=35).chain([257, 4099]) {
            let corrections: Vec<f32> = (0..len)
                .map(|i| {
                    (i.wrapping_mul(4051).wrapping_add(17) % 8191) as f32 * (4.0 / 8190.0) - 2.0
                })
                .collect();
            for &(amount, center, inv_stddev) in &parameters {
                let mut expected = ImageB::new(len, 1);
                for (i, value) in expected.as_mut_slice().iter_mut().enumerate() {
                    *value = (i * 73 + 1) as u8;
                }
                let mut actual = expected.clone();
                apply_corrections_scalar(&corrections, &mut expected, amount, center, inv_stddev);
                method(&corrections, &mut actual, amount, center, inv_stddev);
                assert_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                    "length {len}, amount {amount}, center {center}, inv_stddev {inv_stddev}"
                );
            }
        }
    }

    #[test]
    fn selected_apply_corrections_matches_scalar() {
        check_apply_corrections(select_apply_corrections_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_apply_corrections_matches_scalar() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            check_apply_corrections(|corrections, field, amount, center, inv_stddev| unsafe {
                crate::avx::apply_structure_aq_avx2(corrections, field, amount, center, inv_stddev)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_apply_corrections_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_apply_corrections(|corrections, field, amount, center, inv_stddev| unsafe {
                crate::sse::apply_structure_aq_sse41(corrections, field, amount, center, inv_stddev)
            });
        }
    }

    #[test]
    fn coherent_line_scores_above_checker_noise() {
        let mut line = [0.0f32; 64];
        let mut checker = [0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                line[y * 8 + x] = if x < 4 { 0.2 } else { 0.8 };
                checker[y * 8 + x] = if (x + y) & 1 == 0 { 0.2 } else { 0.8 };
            }
        }
        let dct = crate::dct::selected_dct8x8();
        let line_f = block_features_scalar(&line, dct);
        let noise_f = block_features_scalar(&checker, dct);
        assert!(line_f.coherence > noise_f.coherence);
        assert!(correction_score(line_f, 3.0, 0.0, 0.0) > correction_score(noise_f, 3.0, 0.0, 0.0));
        assert!(correction_score(line_f, 4.0, 1.0, 1.0) < correction_score(line_f, 4.0, 1.0, 0.0));
    }
}
