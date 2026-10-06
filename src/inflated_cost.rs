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

//! Coefficient-SSE and reconstructed-SSIM costs used by AC strategy selection.

use crate::dc_group_data::{
    NUM_STRATEGIES, STRATEGY_AFV0, STRATEGY_AFV1, STRATEGY_AFV2, STRATEGY_AFV3, STRATEGY_DCT,
    STRATEGY_DCT2X2, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4, STRATEGY_DCT8X16,
    STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16, STRATEGY_DCT32X32,
    STRATEGY_DCT32X64, STRATEGY_DCT64X32, STRATEGY_DCT64X64, STRATEGY_IDENTITY,
};
use crate::dct::{DctInput, IdctMethods, fmla};
use crate::image::{Image3F, Plane};
use std::sync::OnceLock;

const R_NZ_BASE: f32 = 1.6;
/// Cost of each zero token the coder emits while walking the scan order up to
/// the last nonzero (review-3 §1: the estimator previously priced only the
/// nonzeros, under-pricing large transforms with sparse late coefficients).
pub(crate) const R_ZERO: f32 = 0.5;
const R_MAG: f32 = 1.0;
const R_HEADER: f32 = 0.4;
// Per-channel distortion weights (X, Y, B).
pub(crate) static CHANNEL_WEIGHT: [f32; 3] = [0.30, 1.0, 0.28];

pub(crate) const RATE_LOG2_LUT_N: usize = 1024;
pub(crate) type RateLog2Lut = [f32; RATE_LOG2_LUT_N];

#[inline]
pub(crate) fn rate_log2_lut() -> &'static RateLog2Lut {
    static LUT: OnceLock<RateLog2Lut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut values = [0.0f32; RATE_LOG2_LUT_N];
        for (i, value) in values.iter_mut().enumerate() {
            *value = (1.0 + i as f32).log2();
        }
        values
    })
}

#[inline]
pub(crate) fn rate_log2_with_lut(lut: &RateLog2Lut, qabs: f32) -> f32 {
    let k = qabs as usize;
    if k < RATE_LOG2_LUT_N {
        lut[k]
    } else {
        (1.0 + qabs).log2()
    }
}

pub(crate) type SseAndRateFn = unsafe fn(
    &[f32],
    &[f32],
    f32,
    usize,
    usize,
    usize,
    usize,
    usize,
    &RateLog2Lut,
    &[f32; 4],
    &[u32],
) -> (f32, usize, f32, u32);

fn select_sse_and_rate_fn() -> SseAndRateFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return crate::avx::sse_and_rate_avx2::<true>;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::sse_and_rate_sse::<true>;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        crate::neon::sse_and_rate_neon::<true>
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::sse_and_rate_wasm::<true>
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        sse_and_rate_scalar::<true>
    }
}

static SSE_AND_RATE_FN: OnceLock<SseAndRateFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_sse_and_rate_fn() -> SseAndRateFn {
    *SSE_AND_RATE_FN.get_or_init(select_sse_and_rate_fn)
}

/// Coefficient distortion and pricing levels, without the discarded fixed rate.
pub(crate) type SseAndQuantizeFn = unsafe fn(
    &[f32],
    &[f32],
    f32,
    usize,
    usize,
    usize,
    usize,
    usize,
    &RateLog2Lut,
    &[f32; 4],
    &[u32],
    &mut [i32],
) -> (f32, usize, f32, u32);

fn select_sse_and_quantize_fn() -> SseAndQuantizeFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return crate::avx::sse_and_rate_avx2_impl::<true, true>;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::sse_and_rate_sse_impl::<true, true>;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        crate::neon::sse_and_rate_neon_impl::<true, true>
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::sse_and_rate_wasm_impl::<true, true>
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        sse_and_rate_scalar_impl::<true, true>
    }
}

static SSE_AND_QUANTIZE_FN: OnceLock<SseAndQuantizeFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_sse_and_quantize_fn() -> SseAndQuantizeFn {
    *SSE_AND_QUANTIZE_FN.get_or_init(select_sse_and_quantize_fn)
}

#[test]
fn fused_coefficient_scoring_preserves_distortion_and_coding_levels() {
    #[allow(unused_mut)]
    let mut kernels = vec![
        (selected_sse_and_rate_fn(), selected_sse_and_quantize_fn()),
        (
            sse_and_rate_scalar::<true> as SseAndRateFn,
            sse_and_rate_scalar_impl::<true, true> as SseAndQuantizeFn,
        ),
    ];
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        kernels.push((
            crate::sse::sse_and_rate_sse::<true>,
            crate::sse::sse_and_rate_sse_impl::<true, true>,
        ));
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        kernels.push((
            crate::avx::sse_and_rate_avx2::<true>,
            crate::avx::sse_and_rate_avx2_impl::<true, true>,
        ));
    }
    let quantize = crate::group::selected_quantize_block_ac_fn();
    for (fixed, fused) in kernels {
        for (cx, cy) in [
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (4, 2),
            (4, 4),
            (8, 4),
            (8, 8),
        ] {
            let (width, height) = (cx * 8, cy * 8);
            let size = width * height;
            for c in 0..3 {
                for distance in [0.1, 1.0, 4.0] {
                    for qac in [1.0, 17.5] {
                        let qm = if c == 0 { 1.25 } else { 1.0 };
                        let thresholds =
                            crate::group::quantize_ac_thresholds_scaled(c, cx, cy, distance, qm);
                        let inverse: Vec<_> = (0..size)
                            .map(|i| if i % 3 == 0 { 1.0 } else { 0.125 })
                            .collect();
                        // Include both signs of exact ties and their neighbouring
                        // floats, zeros, and larger levels across every quadrant.
                        let values = [
                            0.0f32,
                            0.5,
                            -0.5,
                            1.5,
                            -1.5,
                            2.5,
                            -2.5,
                            0.49999997,
                            -0.49999997,
                            0.50000006,
                            -0.50000006,
                            31.5,
                            -128.0,
                        ];
                        let coeffs: Vec<_> = (0..size)
                            .map(|i| values[i % values.len()] / (inverse[i] * qac * qm))
                            .collect();
                        let scan = crate::coeff_order::scan_pos_lut(width, height);
                        let mut expected = vec![0; size];
                        let mut actual = vec![i32::MAX; size];
                        quantize(
                            &coeffs,
                            c,
                            &inverse,
                            1,
                            qac,
                            qm,
                            distance,
                            cx,
                            cy,
                            &mut expected,
                        );
                        for row in expected.chunks_exact_mut(width).take(cy) {
                            row[..cx].fill(0);
                        }
                        let original = unsafe {
                            fixed(
                                &coeffs,
                                &inverse,
                                qac * qm,
                                width,
                                height,
                                width / 2,
                                cx,
                                cy,
                                rate_log2_lut(),
                                &thresholds,
                                scan,
                            )
                        };
                        let combined = unsafe {
                            fused(
                                &coeffs,
                                &inverse,
                                qac * qm,
                                width,
                                height,
                                width / 2,
                                cx,
                                cy,
                                rate_log2_lut(),
                                &thresholds,
                                &[],
                                &mut actual,
                            )
                        };
                        assert_eq!(
                            original.0.to_bits(),
                            combined.0.to_bits(),
                            "{cx}x{cy} c={c} d={distance} q={qac}"
                        );
                        assert_eq!(original.1, combined.1);
                        assert_eq!(expected, actual, "{cx}x{cy} c={c} d={distance} q={qac}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(dead_code)] // Used by target-specific SSE/WASM test modules.
pub(crate) fn assert_sse_and_rate_matches_reference(kernel: SseAndRateFn, biased: bool) {
    let mut state = 0x5e5e_a11d_0f00_d00du64;
    let mut random = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32 as f32 / u32::MAX as f32
    };

    for &(width, height, half, cx, cy) in &[
        (8usize, 8usize, 4usize, 1usize, 1usize),
        (16, 8, 8, 2, 1),
        (8, 16, 4, 1, 2),
        (16, 16, 8, 2, 2),
        (32, 32, 16, 4, 4),
    ] {
        for case in 0..100 {
            let n = width * height;
            let coeff_scale = if case % 2 == 0 { 0.2 } else { 200.0 };
            let coeff: Vec<f32> = (0..n).map(|_| (random() - 0.5) * coeff_scale).collect();
            let inv: Vec<f32> = (0..n).map(|_| 0.001 + random() * 0.5).collect();
            let q_scaled = 0.5 + random() * 3.0;
            let thr = [
                random() * 0.6,
                random() * 0.6,
                random() * 0.6,
                random() * 0.6,
            ];

            let mut expected_max_scan = 0u32;
            let scan_pos = crate::coeff_order::scan_pos_lut(width, height);
            let (mut expected_sse, mut expected_nzeros, mut expected_mag) =
                (0.0f32, 0usize, 0.0f32);
            for y in 0..height {
                let yfix = if y >= height / 2 { 2 } else { 0 };
                for x in 0..width {
                    if x < cx && y < cy {
                        continue;
                    }
                    let i = y * width + x;
                    let threshold = if x >= half { thr[yfix + 1] } else { thr[yfix] };
                    let a = inv[i] * q_scaled * coeff[i];
                    let q = if a.abs() >= threshold {
                        a.round_ties_even()
                    } else {
                        0.0
                    };
                    let d = if biased {
                        a - crate::group::dequantized_level_f32(q)
                    } else {
                        a - q
                    };
                    expected_sse += d * d;
                    if q != 0.0 {
                        expected_nzeros += 1;
                        expected_mag += (1.0 + q.abs()).log2();
                        expected_max_scan = expected_max_scan.max(scan_pos[i]);
                    }
                }
            }

            let actual = unsafe {
                kernel(
                    &coeff,
                    &inv,
                    q_scaled,
                    width,
                    height,
                    half,
                    cx,
                    cy,
                    rate_log2_lut(),
                    &thr,
                    scan_pos,
                )
            };
            assert_eq!(
                actual.1, expected_nzeros,
                "nzeros mismatch {width}x{height}"
            );
            assert_eq!(
                actual.3, expected_max_scan,
                "max-scan mismatch {width}x{height}"
            );
            let sse_rel = (actual.0 - expected_sse).abs() / expected_sse.abs().max(1.0);
            let mag_rel = (actual.2 - expected_mag).abs() / expected_mag.abs().max(1.0);
            assert!(
                sse_rel < 1e-4,
                "SSE relative error {sse_rel} for {width}x{height}"
            );
            assert!(
                mag_rel < 1e-5,
                "magnitude-rate relative error {mag_rel} for {width}x{height}"
            );
        }
    }
}

// Compare the new lane reduction against the original population-count kernel,
// including the coding quantizer's ties-away levels and special float inputs.
#[cfg(test)]
#[allow(dead_code)] // Called by target-specific SSE/AVX/WASM tests.
pub(crate) fn assert_coefficient_kernel_preserves_original(
    original: SseAndQuantizeFn,
    optimized: SseAndQuantizeFn,
) {
    let special = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut rng = 73u32;
    for (width, height, cx, cy) in [
        (8, 8, 1, 1),
        (16, 8, 2, 1),
        (8, 16, 1, 2),
        (16, 16, 2, 2),
        (32, 16, 4, 2),
        (32, 32, 4, 4),
        (64, 32, 8, 4),
        (64, 64, 8, 8),
    ] {
        let n = width * height;
        let scan: Vec<u32> = (0..n).map(|i| ((i * 17) % n) as u32).collect();
        for case in 0..32 {
            let coeff: Vec<f32> = (0..n)
                .map(|_| {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    match case % 4 {
                        0 => 0.0,
                        1 => ((rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2,
                        2 => ((rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 200.0,
                        _ => special[rng as usize % special.len()],
                    }
                })
                .collect();
            let inv: Vec<f32> = (0..n).map(|i| 0.01 + (i % 7) as f32 * 0.15).collect();
            let scale = [0.01, 0.5, 1.0, 7.0, 80.0][case % 5];
            let thresholds = [0.0, 0.42, 0.57, 0.91];
            let mut expected_levels = vec![-123; n + 5];
            let mut actual_levels = expected_levels.clone();
            let run = |kernel: SseAndQuantizeFn, levels: &mut [i32]| unsafe {
                kernel(
                    &coeff,
                    &inv,
                    scale,
                    width,
                    height,
                    width / 2,
                    cx,
                    cy,
                    rate_log2_lut(),
                    &thresholds,
                    &scan,
                    levels,
                )
            };
            let expected = run(original, &mut expected_levels);
            let actual = run(optimized, &mut actual_levels);
            assert_eq!(
                (actual.0.to_bits(), actual.1, actual.2.to_bits(), actual.3),
                (
                    expected.0.to_bits(),
                    expected.1,
                    expected.2.to_bits(),
                    expected.3
                ),
                "{width}x{height}, case {case}"
            );
            assert_eq!(actual_levels, expected_levels);
        }
    }
}

#[allow(unused, clippy::too_many_arguments)]
pub(crate) fn sse_and_rate_scalar<const BIASED: bool>(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    rate_log2_lut: &RateLog2Lut,
    thr: &[f32; 4],
    scan_pos: &[u32],
) -> (f32, usize, f32, u32) {
    sse_and_rate_scalar_impl::<BIASED, false>(
        coeff,
        inv_matrix,
        q_scaled,
        width,
        height,
        half,
        cx,
        cy,
        rate_log2_lut,
        thr,
        scan_pos,
        &mut [],
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_and_rate_scalar_impl<const BIASED: bool, const SAVE_LEVELS: bool>(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    rate_log2_lut: &RateLog2Lut,
    thr: &[f32; 4],
    scan_pos: &[u32],
    levels: &mut [i32],
) -> (f32, usize, f32, u32) {
    assert!(!SAVE_LEVELS || levels.len() >= width * height);
    let mut sse = 0.0f32;
    let mut nzeros = 0usize;
    let mut mag_bits = 0.0f32;
    let mut max_scan = 0u32;
    for (y, (coeff_row, inv_row)) in coeff
        .chunks_exact(width)
        .zip(inv_matrix.chunks_exact(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        for (x, (&coefficient, &inverse)) in coeff_row.iter().zip(inv_row.iter()).enumerate() {
            if x < cx && y < cy {
                if SAVE_LEVELS {
                    levels[y * width + x] = 0;
                }
                continue;
            }
            let threshold = if x >= half { thr[yfix + 1] } else { thr[yfix] };
            let a = inverse * q_scaled * coefficient;
            let q = if a.abs() >= threshold { a.round() } else { 0.0 };
            if SAVE_LEVELS {
                levels[y * width + x] = q as i32;
            }
            let d = if BIASED {
                a - crate::group::dequantized_level_f32(q)
            } else {
                a - q
            };
            sse += d * d;
            if q != 0.0 {
                nzeros += 1;
                if !SAVE_LEVELS {
                    mag_bits += rate_log2_with_lut(rate_log2_lut, q.abs());
                    max_scan = max_scan.max(scan_pos[y * width + x]);
                }
            }
        }
    }
    (sse, nzeros, mag_bits, max_scan)
}

/// Zero tokens the coder must emit before the last nonzero: scan span minus
/// the LLF prefix (never coded) minus the nonzeros themselves.
#[inline]
pub(crate) fn visited_zeros(nzeros: usize, max_scan: u32, cx: usize, cy: usize) -> f32 {
    if nzeros == 0 {
        return 0.0;
    }
    (max_scan as usize + 1).saturating_sub(cx * cy + nzeros) as f32
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn channel_rd(
    sse_and_rate_fn: SseAndRateFn,
    rate_log2_lut: &RateLog2Lut,
    coeff: &[f32],
    inv_matrix: &[f32],
    channel: usize,
    qac: f32,
    qm_mult: f32,
    distance: f32,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let width = cx * 8;
    let height = cy * 8;
    let half = width / 2;
    let thr = crate::group::quantize_ac_thresholds_scaled(channel, cx, cy, distance, qm_mult);
    let scan_pos = crate::coeff_order::scan_pos_lut(width, height);
    let (sse, nzeros, mag_bits, max_scan) = unsafe {
        sse_and_rate_fn(
            coeff,
            inv_matrix,
            qac * qm_mult,
            width,
            height,
            half,
            cx,
            cy,
            rate_log2_lut,
            &thr,
            scan_pos,
        )
    };
    (sse, model_bits(nzeros, mag_bits, max_scan, cx, cy))
}

/// The coefficient rate model's bit estimate for a channel block from its
/// nonzero count, summed magnitude log2 and last scan position.
#[inline]
pub(crate) fn model_bits(nzeros: usize, mag_bits: f32, max_scan: u32, cx: usize, cy: usize) -> f32 {
    let header = R_HEADER * rate_log2_with_lut(rate_log2_lut(), nzeros as f32);
    nzeros as f32 * R_NZ_BASE
        + R_MAG * mag_bits
        + header
        + R_ZERO * visited_zeros(nzeros, max_scan, cx, cy)
}

pub(crate) fn strategy_pixel_dims(strategy: u8) -> (usize, usize) {
    match strategy {
        STRATEGY_DCT16X8 => (8, 16),
        STRATEGY_DCT8X16 => (16, 8),
        STRATEGY_DCT16X16 => (16, 16),
        STRATEGY_DCT32X16 => (16, 32),
        STRATEGY_DCT16X32 => (32, 16),
        STRATEGY_DCT32X32 => (32, 32),
        STRATEGY_DCT64X64 => (64, 64),
        STRATEGY_DCT64X32 => (32, 64),
        STRATEGY_DCT32X64 => (64, 32),
        _ => (8, 8),
    }
}

pub(crate) fn strategy_pixel_count(strategy: u8) -> usize {
    let (w, h) = strategy_pixel_dims(strategy);
    w * h
}

/// A transform's pixel footprint and the part of it inside the image. The
/// rest is edge-replicated padding the decoder crops away, so its error must
/// not be charged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Footprint {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) visible_width: usize,
    pub(crate) visible_height: usize,
}

impl Footprint {
    #[inline]
    pub(crate) fn new(
        image_width: usize,
        image_height: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            visible_width: width.min(image_width.saturating_sub(x)),
            visible_height: height.min(image_height.saturating_sub(y)),
        }
    }

    #[inline]
    pub(crate) fn of_strategy(image: &Image3F, x: usize, y: usize, strategy: u8) -> Self {
        let (width, height) = strategy_pixel_dims(strategy);
        Self::new(image.xsize(), image.ysize(), x, y, width, height)
    }

    #[inline]
    pub(crate) fn is_full(&self) -> bool {
        self.visible_width == self.width && self.visible_height == self.height
    }

    /// The footprint's pixels in `plane` as `(data, stride)`: a view into the
    /// plane when the footprint lies inside it, otherwise an edge-replicated
    /// gather into `tmp`.
    #[inline]
    pub(crate) fn pixels<'a>(
        &self,
        plane: &'a Plane<f32>,
        tmp: &'a mut [f32],
    ) -> (&'a [f32], usize) {
        if self.width <= plane.xsize().saturating_sub(self.x)
            && self.height <= plane.ysize().saturating_sub(self.y)
        {
            let stride = plane.xsize();
            return (&plane.as_slice()[self.y * stride + self.x..], stride);
        }
        let n = self.width * self.height;
        crate::ac_strategy::gather_pixels(
            plane,
            self.x,
            self.y,
            self.width,
            self.height,
            &mut tmp[..n],
        );
        (&tmp[..n], self.width)
    }

    /// Share of the footprint the decoder shows. Coefficient-domain errors
    /// cannot be split by pixel, so they are charged by area.
    #[inline]
    pub(crate) fn visible_fraction(&self) -> f32 {
        if self.is_full() {
            return 1.0;
        }
        debug_assert!(
            self.visible_width > 0 && self.visible_height > 0,
            "footprint origin outside the image"
        );
        (self.visible_width * self.visible_height) as f32 / (self.width * self.height) as f32
    }
}

pub(crate) fn forward_for(strategy: u8, input: &[f32], out: &mut [f32]) {
    use crate::dct;
    macro_rules! fwd {
        ($f:path, $n:literal) => {{
            let i: &[f32; $n] = input.first_chunk::<$n>().unwrap();
            let o: &mut [f32; $n] = out.first_chunk_mut::<$n>().unwrap();
            $f(i, o);
        }};
    }
    macro_rules! fwd_input {
        ($f:path, $n:literal, $w:literal, $h:literal) => {{
            let i: &[f32; $n] = input.first_chunk::<$n>().unwrap();
            let o: &mut [f32; $n] = out.first_chunk_mut::<$n>().unwrap();
            $f(DctInput::<$w, $h>::from_flat(i), o);
        }};
    }
    match strategy {
        STRATEGY_IDENTITY => {
            let i: &[f32; 64] = input.first_chunk::<64>().unwrap();
            let o: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            dct::identity8x8(DctInput::from_flat(i), o);
        }
        STRATEGY_DCT2X2 => {
            let i: &[f32; 64] = input.first_chunk::<64>().unwrap();
            let o: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            dct::dct2x2_8x8(DctInput::from_flat(i), o);
        }
        STRATEGY_DCT4X4 => fwd!(dct::dct4x4, 64),
        STRATEGY_DCT4X8 => fwd!(dct::dct4x8, 64),
        STRATEGY_DCT8X4 => fwd!(dct::dct8x4, 64),
        STRATEGY_AFV0 => fwd!(crate::afv::afv0, 64),
        STRATEGY_AFV1 => fwd!(crate::afv::afv1, 64),
        STRATEGY_AFV2 => fwd!(crate::afv::afv2, 64),
        STRATEGY_AFV3 => fwd!(crate::afv::afv3, 64),
        STRATEGY_DCT16X8 => fwd!(dct::dct16x8, 128),
        STRATEGY_DCT8X16 => fwd!(dct::dct8x16, 128),
        STRATEGY_DCT16X16 => fwd!(dct::dct16x16, 256),
        STRATEGY_DCT32X32 => fwd!(dct::dct32x32, 1024),
        STRATEGY_DCT32X16 => fwd!(dct::dct32x16, 512),
        STRATEGY_DCT16X32 => fwd!(dct::dct16x32, 512),
        STRATEGY_DCT64X64 => fwd_input!(dct::dct64x64_scalar_input, 4096, 64, 64),
        STRATEGY_DCT64X32 => fwd_input!(dct::dct64x32_scalar_input, 2048, 32, 64),
        STRATEGY_DCT32X64 => fwd_input!(dct::dct32x64_scalar_input, 2048, 64, 32),
        _ => fwd!(dct::dct8x8, 64),
    }
}

pub(crate) fn forward_matrix(strategy: u8) -> &'static [f32] {
    static MATRICES: [OnceLock<Vec<f32>>; NUM_STRATEGIES] =
        [const { OnceLock::new() }; NUM_STRATEGIES];
    MATRICES[strategy as usize].get_or_init(|| {
        let n = strategy_pixel_count(strategy);
        let mut matrix = vec![0.0f32; n * n];
        let mut input = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];
        for impulse in 0..n {
            input.fill(0.0);
            input[impulse] = 1.0;
            forward_for(strategy, &input, &mut output);
            matrix[impulse * n..impulse * n + n].copy_from_slice(&output);
        }
        matrix
    })
}

pub(crate) fn reconstruct_error(
    idct: &IdctMethods,
    strategy: u8,
    coeff_err: &[f32],
    err_out: &mut [f32],
) {
    macro_rules! inverse {
        ($f:expr, $n:literal, $w:literal, $h:literal) => {
            ($f)(
                DctInput::<$w, $h>::new(coeff_err, $w),
                err_out.first_chunk_mut::<$n>().unwrap(),
            )
        };
    }
    match strategy {
        STRATEGY_DCT => inverse!(idct.idct8x8, 64, 8, 8),
        STRATEGY_IDENTITY => inverse!(idct.inv_identity8x8, 64, 8, 8),
        STRATEGY_DCT2X2 => inverse!(idct.inv_dct2x2_8x8, 64, 8, 8),
        STRATEGY_DCT8X16 => inverse!(idct.idct8x16, 128, 16, 8),
        STRATEGY_DCT16X8 => inverse!(idct.idct16x8, 128, 16, 8),
        STRATEGY_DCT16X16 => inverse!(idct.idct16x16, 256, 16, 16),
        STRATEGY_DCT16X32 => inverse!(idct.idct16x32, 512, 32, 16),
        STRATEGY_DCT32X16 => inverse!(idct.idct32x16, 512, 32, 16),
        STRATEGY_DCT32X32 => inverse!(idct.idct32x32, 1024, 32, 32),
        STRATEGY_DCT64X64 => inverse!(idct.idct64x64, 4096, 64, 64),
        STRATEGY_DCT64X32 => inverse!(idct.idct64x32, 2048, 64, 32),
        STRATEGY_DCT32X64 => inverse!(idct.idct32x64, 2048, 64, 32),
        _ => {
            let n = strategy_pixel_count(strategy);
            let matrix = inverse_matrix(strategy);
            for (row, value) in err_out[..n].iter_mut().enumerate() {
                let base = row * n;
                *value = (0..n).map(|k| matrix[base + k] * coeff_err[k]).sum::<f32>();
            }
        }
    }
}

/// Inverse of an orthogonal (not orthonormal) forward basis:
/// `x = Σ_k c_k · f_k / ‖f_k‖²`. DCT4/AFV basis vectors have unequal norms, so
/// the uniform `N·Fᵀ` shortcut that holds for DCT8 is wrong for them.
pub(crate) fn inverse_matrix(strategy: u8) -> &'static [f32] {
    static MATRICES: [OnceLock<Vec<f32>>; NUM_STRATEGIES] =
        [const { OnceLock::new() }; NUM_STRATEGIES];
    MATRICES[strategy as usize].get_or_init(|| {
        let n = strategy_pixel_count(strategy);
        let forward = forward_matrix(strategy);
        let inv_norm: Vec<f64> = (0..n)
            .map(|k| {
                let norm2: f64 = (0..n).map(|p| (forward[p * n + k] as f64).powi(2)).sum();
                1.0 / norm2
            })
            .collect();
        (0..n * n)
            .map(|i| (forward[i] as f64 * inv_norm[i % n]) as f32)
            .collect()
    })
}

pub(crate) type ReconQuantizeFn = unsafe fn(
    &[f32],
    &[f32],
    f32,
    &[f32; 4],
    usize,
    usize,
    usize,
    usize,
    usize,
    &mut [f32],
    &RateLog2Lut,
    &[u32],
) -> f32;

pub(crate) type SsimDeficitFn = unsafe fn(&[f32], &[f32], usize, usize) -> f32;
pub(crate) type PrepareReconstructionFn =
    unsafe fn(&Plane<f32>, usize, usize, usize, usize, &[f32], &mut [f32], &mut [f32]);
pub(crate) type ErrorGradientEnergyFn = fn(&[f32], usize, usize) -> f32;
pub(crate) type ErrorGradientPeakEnergyFn = fn(&[f32], &[f32], usize, usize, f32) -> f32;
pub(crate) type CombineErrorFn = fn(&[f32], &[f32], f32, &mut [f32]);

pub(crate) struct ReconQuantization<'a> {
    /// The caller replaces the fixed rate when this transform has learned prices.
    pub(crate) compute_rate: bool,
    pub(crate) rate_log2_lut: &'a RateLog2Lut,
    pub(crate) coeffs: [&'a [f32]; 3],
    pub(crate) inverse_matrices: [&'a [f32]; 3],
    pub(crate) qac: f32,
    pub(crate) qm_mult_x: f32,
    pub(crate) qm_mult_b: f32,
    pub(crate) distance: f32,
}

pub(crate) struct ReconTransform {
    pub(crate) blocks_x: usize,
    pub(crate) blocks_y: usize,
    pub(crate) strategy: u8,
}

pub(crate) struct ReconSource<'a> {
    pub(crate) opsin: &'a Image3F,
    pub(crate) x: usize,
    pub(crate) y: usize,
}

pub(crate) struct ReconScoring {
    pub(crate) factor_x: f32,
    pub(crate) factor_b: f32,
    pub(crate) channel_weights: [f32; 3],
    pub(crate) xyb_matrix: crate::xyb::XybMatrix,
    /// Linear-RGB/OKLab hue and desaturation penalty on risky chroma edges.
    pub(crate) rgb_hue_alpha: f32,
    /// Spatial-error gradient weight used by transform reranking.
    pub(crate) gradient_alpha: f32,
    /// Peak-pooled spatial-error gradient weight used by transform reranking.
    pub(crate) gradient_peak_alpha: f32,
    /// Leave each channel's spatial error plane in `scratch[0..3]` after the
    /// call. The joint mosaic rerank reads them back to measure error jumps
    /// across child-block seams without reconstructing a second time.
    pub(crate) keep_spatial_errors: bool,
}

pub(crate) struct ReconDistInput<'a> {
    pub(crate) idct: &'a IdctMethods,
    pub(crate) quantization: ReconQuantization<'a>,
    pub(crate) transform: ReconTransform,
    pub(crate) source: ReconSource<'a>,
    pub(crate) scoring: ReconScoring,
}

pub(crate) struct ReconErrorKernels {
    pub(crate) gradient_energy: ErrorGradientEnergyFn,
    pub(crate) gradient_peak_energy: ErrorGradientPeakEnergyFn,
    pub(crate) combine: CombineErrorFn,
    pub(crate) rgb_hue_chroma_edge_loss: RgbHueChromaEdgeLossFn,
}

pub(crate) struct ReconKernels<'a> {
    pub(crate) quantize: ReconQuantizeFn,
    pub(crate) ssim: SsimDeficitFn,
    pub(crate) prepare: PrepareReconstructionFn,
    pub(crate) error: &'a ReconErrorKernels,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReconCost {
    pub(crate) distortion: f32,
    pub(crate) rate: f32,
    /// Weighted hue/desaturation contribution already included in distortion.
    pub(crate) hue_distortion: f32,
}

impl ReconCost {
    #[inline]
    pub(crate) fn dist_and_rate(self) -> (f32, f32) {
        (self.distortion, self.rate)
    }
}

pub(crate) type ReconDistAndRateFn = for<'a, 'input, 'kernels> fn(
    &mut [[f32; 1024]; 8],
    &'input ReconDistInput<'a>,
    &'kernels ReconErrorKernels,
) -> ReconCost;

#[allow(dead_code)]
pub(crate) fn combine_error_scalar(
    spatial: &[f32],
    luma: &[f32],
    factor: f32,
    combined: &mut [f32],
) {
    debug_assert_eq!(spatial.len(), luma.len());
    debug_assert_eq!(spatial.len(), combined.len());
    for ((combined, &spatial), &luma) in combined.iter_mut().zip(spatial).zip(luma) {
        *combined = fmla(factor, luma, spatial);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_combine_error_fn() -> CombineErrorFn {
    crate::wasm::combine_error_wasm
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_combine_error_fn() -> CombineErrorFn {
    |spatial, luma, factor, combined| unsafe {
        crate::neon::combine_error_neon(spatial, luma, factor, combined)
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_combine_error_fn() -> CombineErrorFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |spatial, luma, factor, combined| unsafe {
            crate::avx::combine_error_avx2(spatial, luma, factor, combined)
        };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |spatial, luma, factor, combined| unsafe {
            crate::sse::combine_error_sse41(spatial, luma, factor, combined)
        };
    }
    combine_error_scalar
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_error_gradient_energy_fn() -> ErrorGradientEnergyFn {
    crate::wasm::error_gradient_energy_wasm
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_error_gradient_energy_fn() -> ErrorGradientEnergyFn {
    |error, width, height| unsafe { crate::neon::error_gradient_energy_neon(error, width, height) }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_error_gradient_energy_fn() -> ErrorGradientEnergyFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |error, width, height| unsafe {
            crate::avx::error_gradient_energy_avx2(error, width, height)
        };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |error, width, height| unsafe {
            crate::sse::error_gradient_energy_sse41(error, width, height)
        };
    }
    error_gradient_energy_scalar
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_error_gradient_peak_energy_fn() -> ErrorGradientPeakEnergyFn {
    crate::wasm::error_gradient_peak_energy_wasm
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_error_gradient_peak_energy_fn() -> ErrorGradientPeakEnergyFn {
    |error, original, width, height, floor| unsafe {
        crate::neon::error_gradient_peak_energy_neon(error, original, width, height, floor)
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_error_gradient_peak_energy_fn() -> ErrorGradientPeakEnergyFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |error, original, width, height, floor| unsafe {
            crate::avx::error_gradient_peak_energy_avx2(error, original, width, height, floor)
        };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |error, original, width, height, floor| unsafe {
            crate::sse::error_gradient_peak_energy_sse41(error, original, width, height, floor)
        };
    }
    error_gradient_peak_energy_scalar
}

pub(crate) fn select_recon_dist_and_rate_fn() -> ReconDistAndRateFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |scratch, input, error_kernels| unsafe {
            crate::avx::recon_dist_and_rate_avx2(scratch, input, error_kernels)
        };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return |scratch, input, error_kernels| unsafe {
            crate::neon::recon_dist_and_rate_neon(scratch, input, error_kernels)
        };
    }
    #[allow(unreachable_code)]
    recon_dist_and_rate_default
}

fn recon_dist_and_rate_default(
    scratch: &mut [[f32; 1024]; 8],
    input: &ReconDistInput<'_>,
    error: &ReconErrorKernels,
) -> ReconCost {
    recon_dist_and_rate_with_kernels(
        scratch,
        input,
        &ReconKernels {
            quantize: if input.quantization.compute_rate {
                recon_quantize_scalar::<true, true>
            } else {
                recon_quantize_scalar::<true, false>
            },
            ssim: ssim_deficit_dispatch_kernel,
            prepare: prepare_reconstruction_scalar,
            error,
        },
    )
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn recon_dist_and_rate_scalar<const BIASED: bool>(
    scratch: &mut [[f32; 1024]; 8],
    input: &ReconDistInput<'_>,
) -> ReconCost {
    let error = ReconErrorKernels {
        gradient_energy: error_gradient_energy_scalar,
        gradient_peak_energy: error_gradient_peak_energy_scalar,
        combine: combine_error_scalar,
        rgb_hue_chroma_edge_loss: rgb_hue_chroma_edge_loss_scalar,
    };
    recon_dist_and_rate_with_kernels(
        scratch,
        input,
        &ReconKernels {
            quantize: if input.quantization.compute_rate {
                recon_quantize_scalar::<BIASED, true>
            } else {
                recon_quantize_scalar::<BIASED, false>
            },
            ssim: ssim_deficit_scalar_kernel,
            prepare: prepare_reconstruction_scalar,
            error: &error,
        },
    )
}

/// `sum((dx err)^2 + (dy err)^2)` over one channel's spatial error plane.
#[allow(dead_code)]
pub(crate) fn error_gradient_energy_scalar(error: &[f32], width: usize, height: usize) -> f32 {
    error_gradient_energy_strided(error, width, width, height)
}

/// [`error_gradient_energy_scalar`] over the leading `width × height` window
/// of a `stride`-wide plane.
pub(crate) fn error_gradient_energy_strided(
    error: &[f32],
    stride: usize,
    width: usize,
    height: usize,
) -> f32 {
    assert!(width <= stride);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let n = (height - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(width))
        .expect("gradient plane size overflow");
    assert!(error.len() >= n);
    let rows = error.chunks(stride).take(height).map(|row| &row[..width]);
    let mut grad = 0.0f32;
    for row in rows.clone() {
        for &[left, right] in row.array_windows::<2>() {
            let d = right - left;
            grad = fmla(d, d, grad);
        }
    }
    for (top, bottom) in rows.clone().zip(rows.skip(1)) {
        for (&top, &bottom) in top.iter().zip(bottom) {
            let d = bottom - top;
            grad = fmla(d, d, grad);
        }
    }
    grad
}

pub(crate) fn error_gradient_peak_energy_scalar(
    error: &[f32],
    original: &[f32],
    width: usize,
    height: usize,
    floor: f32,
) -> f32 {
    error_gradient_peak_energy_strided(error, original, width, width, height, floor)
}

/// [`error_gradient_peak_energy_scalar`] over the leading `width × height`
/// window of `stride`-wide planes.
pub(crate) fn error_gradient_peak_energy_strided(
    error: &[f32],
    original: &[f32],
    stride: usize,
    width: usize,
    height: usize,
    floor: f32,
) -> f32 {
    assert!(width <= stride);
    assert!(floor.is_finite() && floor >= 0.0);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let n = (height - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(width))
        .expect("gradient plane size overflow");
    assert!(error.len() >= n && original.len() >= n);
    let mut total = 0.0f32;
    for cell_y in (0..height).step_by(4) {
        for cell_x in (0..width).step_by(4) {
            let mut max_x = 0.0f32;
            let mut max_y = 0.0f32;
            for y in cell_y..(cell_y + 4).min(height) {
                for x in cell_x..(cell_x + 4).min(width) {
                    let p = y * stride + x;
                    if x + 1 < width {
                        let error_gradient = (error[p + 1] - error[p]).abs();
                        let source_gradient = (original[p + 1] - original[p]).abs();
                        let excess = (error_gradient - 0.5 * source_gradient - floor).max(0.0);
                        max_x = max_x.max(excess * excess);
                    }
                    if y + 1 < height {
                        let error_gradient = (error[p + stride] - error[p]).abs();
                        let source_gradient = (original[p + stride] - original[p]).abs();
                        let excess = (error_gradient - 0.5 * source_gradient - floor).max(0.0);
                        max_y = max_y.max(excess * excess);
                    }
                }
            }
            total += max_x + max_y;
        }
    }
    total
}

#[inline]
fn gradient_peak_floor(distance: f32) -> f32 {
    // Preserve the validated high-quality metric, then reject ordinary coarse
    // quantization error more aggressively once the long tail engages.
    let coarse_mix = ((distance - 1.9) / 0.1).clamp(0.0, 1.0);
    distance * fmla(coarse_mix, 0.0045 - 0.0015, 0.0015)
}

pub(crate) fn recon_dist_and_rate_with_kernels(
    scratch: &mut [[f32; 1024]; 8],
    input: &ReconDistInput<'_>,
    kernels: &ReconKernels<'_>,
) -> ReconCost {
    let quantization = &input.quantization;
    let transform = &input.transform;
    let source = &input.source;
    let scoring = &input.scoring;
    let rate_log2_lut = quantization.rate_log2_lut;
    let coeffs = quantization.coeffs;
    let inverse_matrices = &quantization.inverse_matrices;
    let qac = quantization.qac;
    let qm_mult_x = quantization.qm_mult_x;
    let distance = quantization.distance;
    let cx = transform.blocks_x;
    let cy = transform.blocks_y;
    let strategy = transform.strategy;
    let opsin = source.opsin;
    let px = source.x;
    let py = source.y;
    let factor_x = scoring.factor_x;
    let factor_b = scoring.factor_b;
    let gradient_alpha = scoring.gradient_alpha;
    let gradient_peak_alpha = scoring.gradient_peak_alpha;
    let rgb_hue_alpha = scoring.rgb_hue_alpha;
    let gradient_peak_floor = if gradient_peak_alpha > 0.0 {
        gradient_peak_floor(distance)
    } else {
        0.0
    };
    let quantize = kernels.quantize;
    let ssim = kernels.ssim;
    let prepare_reconstruction = kernels.prepare;
    let error_gradient_energy = kernels.error.gradient_energy;
    let error_gradient_peak_energy = kernels.error.gradient_peak_energy;
    let combine_error = kernels.error.combine;
    let rgb_hue_chroma_edge_loss = kernels.error.rgb_hue_chroma_edge_loss;
    let n = strategy_pixel_count(strategy);
    let width = cx.checked_mul(8).expect("coefficient width overflow");
    let height = cy.checked_mul(8).expect("coefficient height overflow");
    assert_eq!(
        width
            .checked_mul(height)
            .expect("coefficient size overflow"),
        n,
        "strategy and coefficient dimensions disagree"
    );
    assert!(inverse_matrices.iter().all(|matrix| matrix.len() >= n));
    assert!(qac.is_finite() && qac > 0.0);
    assert!(qm_mult_x.is_finite() && qm_mult_x > 0.0);
    assert!((0..3).all(|c| opsin.plane(c).xsize() != 0 && opsin.plane(c).ysize() != 0));

    let half = width / 2;
    let (pixel_width, pixel_height) = strategy_pixel_dims(strategy);
    let thresholds = [
        crate::group::quantize_ac_thresholds_scaled(0, cx, cy, distance, qm_mult_x),
        crate::group::quantize_ac_thresholds(1, cx, cy, distance),
        crate::group::quantize_ac_thresholds_scaled(2, cx, cy, distance, quantization.qm_mult_b),
    ];
    let quant_scales = [qac * qm_mult_x, qac, qac * quantization.qm_mult_b];

    let (coeff_error, rest) = scratch.split_at_mut(3);
    let [spatial_error, _, combined_error, reconstructed, original] = rest else {
        unreachable!()
    };
    let scan_pos = if quantization.compute_rate {
        crate::coeff_order::scan_pos_lut(width, height)
    } else {
        &[]
    };
    let mut rate = 0.0f32;
    rate += unsafe {
        quantize(
            &coeffs[1][..n],
            &inverse_matrices[1][..n],
            quant_scales[1],
            &thresholds[1],
            width,
            height,
            half,
            cx,
            cy,
            &mut coeff_error[1][..n],
            rate_log2_lut,
            scan_pos,
        )
    };
    for c in [0usize, 2] {
        let factor = if c == 0 { factor_x } else { factor_b };
        combine_error(
            &coeffs[c][..n],
            &coeff_error[1][..n],
            factor,
            &mut combined_error[..n],
        );
        rate += unsafe {
            quantize(
                &combined_error[..n],
                &inverse_matrices[c][..n],
                quant_scales[c],
                &thresholds[c],
                width,
                height,
                half,
                cx,
                cy,
                &mut coeff_error[c][..n],
                rate_log2_lut,
                scan_pos,
            )
        };
    }

    let footprint = Footprint::new(
        opsin.xsize(),
        opsin.ysize(),
        px,
        py,
        pixel_width,
        pixel_height,
    );
    let (visible_width, visible_height) = (footprint.visible_width, footprint.visible_height);
    let mut distortion = 0.0f32;
    for c in 0..3 {
        reconstruct_error(
            input.idct,
            strategy,
            &coeff_error[c][..n],
            &mut spatial_error[..n],
        );
        let error: &[f32] = &spatial_error[..n];
        let plane = opsin.plane(c);
        unsafe {
            prepare_reconstruction(
                plane,
                px,
                py,
                pixel_width,
                pixel_height,
                error,
                &mut original[..n],
                &mut reconstructed[..n],
            );
        }
        let weight = input.scoring.channel_weights[c];
        if footprint.is_full() {
            distortion += weight
                * unsafe {
                    ssim(
                        &original[..n],
                        &reconstructed[..n],
                        pixel_width,
                        pixel_height,
                    )
                };
            if gradient_alpha > 0.0 {
                distortion += weight
                    * gradient_alpha
                    * error_gradient_energy(&error[..n], pixel_width, pixel_height);
            }
            if gradient_peak_alpha > 0.0 {
                distortion += weight
                    * gradient_peak_alpha
                    * error_gradient_peak_energy(
                        &error[..n],
                        &original[..n],
                        pixel_width,
                        pixel_height,
                        gradient_peak_floor,
                    );
            }
        } else {
            // Edge footprints: score only the pixels the decoder keeps,
            // reading the footprint-strided buffers in place.
            distortion += weight
                * ssim_deficit_visible(
                    &original[..n],
                    &reconstructed[..n],
                    pixel_width,
                    visible_width,
                    visible_height,
                );
            if gradient_alpha > 0.0 {
                distortion += weight
                    * gradient_alpha
                    * error_gradient_energy_strided(
                        &error[..n],
                        pixel_width,
                        visible_width,
                        visible_height,
                    );
            }
            if gradient_peak_alpha > 0.0 {
                distortion += weight
                    * gradient_peak_alpha
                    * error_gradient_peak_energy_strided(
                        &error[..n],
                        &original[..n],
                        pixel_width,
                        visible_width,
                        visible_height,
                        gradient_peak_floor,
                    );
            }
        }
        if rgb_hue_alpha > 0.0 || scoring.keep_spatial_errors {
            coeff_error[c][..n].copy_from_slice(error);
        }
    }
    let mut hue_distortion = 0.0;
    if rgb_hue_alpha > 0.0 {
        hue_distortion = rgb_hue_alpha
            * if footprint.is_full() {
                unsafe {
                    rgb_hue_chroma_edge_loss(
                        opsin,
                        px,
                        py,
                        pixel_width,
                        pixel_height,
                        [
                            &coeff_error[0][..n],
                            &coeff_error[1][..n],
                            &coeff_error[2][..n],
                        ],
                        &scoring.xyb_matrix,
                    )
                }
            } else {
                rgb_hue_chroma_edge_loss_strided(
                    opsin,
                    px,
                    py,
                    pixel_width,
                    visible_width,
                    visible_height,
                    [
                        &coeff_error[0][..n],
                        &coeff_error[1][..n],
                        &coeff_error[2][..n],
                    ],
                    &scoring.xyb_matrix,
                )
            };
        distortion += hue_distortion;
    }
    ReconCost {
        distortion,
        rate,
        hue_distortion,
    }
}

#[inline]
fn linear_rgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let l =
        crate::xyb::cbrtf(fmla(0.412_221_46, r, fmla(0.536_332_55, g, 0.051_445_995 * b)).max(0.0));
    let m =
        crate::xyb::cbrtf(fmla(0.211_903_5, r, fmla(0.680_699_5, g, 0.107_396_96 * b)).max(0.0));
    let s =
        crate::xyb::cbrtf(fmla(0.088_302_46, r, fmla(0.281_718_85, g, 0.629_978_7 * b)).max(0.0));
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

pub(crate) type RgbHueChromaEdgeLossFn =
    unsafe fn(&Image3F, usize, usize, usize, usize, [&[f32]; 3], &crate::xyb::XybMatrix) -> f32;

pub(crate) fn select_rgb_hue_chroma_edge_loss_fn() -> RgbHueChromaEdgeLossFn {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        return crate::wasm::rgb_hue_chroma_edge_loss_wasm;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return crate::avx::rgb_hue_chroma_edge_loss_avx2;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::rgb_hue_chroma_edge_loss_sse41;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return crate::neon::rgb_hue_chroma_edge_loss_neon;
    }
    #[allow(unreachable_code)]
    rgb_hue_chroma_edge_loss_scalar
}

/// Resolve color error along and across the source hue. Using reconstructed
/// chroma magnitude for the radial term would miss an equal-chroma hue reversal:
/// its magnitude is unchanged and its perpendicular projection is zero.
#[inline]
fn hue_chroma_error(source: [f32; 2], reconstructed: [f32; 2], source_chroma: f32) -> f32 {
    let radial_error = fmla(
        source[0],
        source[0] - reconstructed[0],
        source[1] * (source[1] - reconstructed[1]),
    ) / source_chroma.max(1e-4);
    let desaturation = radial_error.max(0.0);
    let perpendicular =
        (source[0] * reconstructed[1] - source[1] * reconstructed[0]) / (source_chroma + 1e-4);
    fmla(
        desaturation,
        desaturation,
        0.75 * perpendicular * perpendicular,
    )
}

#[inline]
fn rgb_hue_chroma_pixel_loss(
    source: [f32; 3],
    reconstructed: [f32; 3],
    edge: f32,
    matrix: &crate::xyb::XybMatrix,
) -> f32 {
    let edge_risk = ((edge - 0.006) * (1.0 / 0.030)).clamp(0.0, 1.0);
    if edge_risk == 0.0 {
        return 0.0;
    }
    let source_rgb = crate::xyb::xyb_to_rgb_pixel_f32(matrix, source[0], source[1], source[2]);
    let recon_rgb = crate::xyb::xyb_to_rgb_pixel_f32(
        matrix,
        reconstructed[0],
        reconstructed[1],
        reconstructed[2],
    );
    let source_lab = linear_rgb_to_oklab(source_rgb);
    let recon_lab = linear_rgb_to_oklab(recon_rgb);
    let source_chroma = fmla(source_lab[1], source_lab[1], source_lab[2] * source_lab[2]).sqrt();
    // Blue has low perceptual lightness even when visibly saturated. Its
    // negative OKLab b supplies a local lift; warm hues retain the original
    // brightness gate, avoiding a frame-wide increase in the chroma budget.
    let brightness_risk =
        ((source_lab[0] - 0.35 - source_lab[2].min(0.0)) * (1.0 / 0.40)).clamp(0.0, 1.0);
    let chroma_risk = ((source_chroma - 0.03) * (1.0 / 0.12)).clamp(0.0, 1.0);
    let risk = edge_risk * brightness_risk * chroma_risk;
    risk * hue_chroma_error(
        [source_lab[1], source_lab[2]],
        [recon_lab[1], recon_lab[2]],
        source_chroma,
    )
}

/// Penalize decoder-domain hue rotation and chroma collapse only on visible,
/// saturated source pixels that sit on an opponent-color edge. The radial
/// component penalizes loss along the original hue, including hue reversal;
/// the perpendicular component measures rotation without an angle singularity
/// near neutral colors. Same-hue oversaturation remains unpenalized.
pub(crate) fn rgb_hue_chroma_edge_loss_scalar(
    opsin: &Image3F,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    spatial_error: [&[f32]; 3],
    matrix: &crate::xyb::XybMatrix,
) -> f32 {
    rgb_hue_chroma_edge_loss_strided(opsin, px, py, width, width, height, spatial_error, matrix)
}

/// [`rgb_hue_chroma_edge_loss_scalar`] over the leading `width × height`
/// window of `stride`-wide error planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rgb_hue_chroma_edge_loss_strided(
    opsin: &Image3F,
    px: usize,
    py: usize,
    stride: usize,
    width: usize,
    height: usize,
    spatial_error: [&[f32]; 3],
    matrix: &crate::xyb::XybMatrix,
) -> f32 {
    assert!(width <= stride);
    let image_width = opsin.xsize();
    let image_height = opsin.ysize();
    if width == 0 || height == 0 {
        return 0.0;
    }
    let mut loss = 0.0f32;
    for y in 0..height {
        let sy = py.saturating_add(y).min(image_height - 1);
        let rows = [
            opsin.plane_row(0, sy),
            opsin.plane_row(1, sy),
            opsin.plane_row(2, sy),
        ];
        let below = (sy + 1 < image_height).then(|| {
            [
                opsin.plane_row(0, sy + 1),
                opsin.plane_row(1, sy + 1),
                opsin.plane_row(2, sy + 1),
            ]
        });
        let error_rows = [
            &spatial_error[0][y * stride..y * stride + width],
            &spatial_error[1][y * stride..y * stride + width],
            &spatial_error[2][y * stride..y * stride + width],
        ];
        for (x, ((&error_x, &error_y), &error_b)) in error_rows[0]
            .iter()
            .zip(error_rows[1])
            .zip(error_rows[2])
            .enumerate()
        {
            let sx = px.saturating_add(x).min(image_width - 1);
            let source = [rows[0][sx], rows[1][sx], rows[2][sx]];
            let cb = source[2] - source[1];
            let mut edge = 0.0f32;
            if x + 1 < width && sx + 1 < image_width {
                let nx = rows[0][sx + 1];
                let ny = rows[1][sx + 1];
                let nb = rows[2][sx + 1];
                edge = edge.max((nx - source[0]).abs() + ((nb - ny) - cb).abs());
            }
            if y + 1 < height
                && let Some(below) = below
            {
                let nx = below[0][sx];
                let ny = below[1][sx];
                let nb = below[2][sx];
                edge = edge.max((nx - source[0]).abs() + ((nb - ny) - cb).abs());
            }
            let reconstructed = [
                source[0] - error_x,
                source[1] - error_y,
                source[2] - error_b,
            ];
            loss += rgb_hue_chroma_pixel_loss(source, reconstructed, edge, matrix);
        }
    }
    loss
}

#[allow(clippy::too_many_arguments)]
fn prepare_reconstruction_scalar(
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
        for ((output, &value), &delta) in reconstructed_row
            .iter_mut()
            .zip(original_row.iter())
            .zip(error_row.iter())
        {
            *output = value - delta;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn recon_quantize_scalar<const BIASED: bool, const COMPUTE_RATE: bool>(
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
    rate_log2_lut: &RateLog2Lut,
    scan_pos: &[u32],
) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("coefficient size overflow");
    assert!(coeff.len() >= n && inv.len() >= n && coeff_error.len() >= n);
    let (mut nonzero, mut magnitude_bits) = (0usize, 0.0f32);
    let mut max_scan = 0u32;
    for (y, ((coeff_row, inv_row), error_row)) in coeff
        .chunks_exact(width)
        .zip(inv.chunks_exact(width))
        .zip(coeff_error.chunks_exact_mut(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        for (x, ((&coefficient, &inverse), error)) in coeff_row
            .iter()
            .zip(inv_row.iter())
            .zip(error_row.iter_mut())
            .enumerate()
        {
            if x < cx && y < cy {
                *error = 0.0;
                continue;
            }
            let threshold = if x >= half {
                thresholds[yfix + 1]
            } else {
                thresholds[yfix]
            };
            let denominator = inverse * quant_scale;
            let scaled = denominator * coefficient;
            let quantized = if scaled.abs() >= threshold {
                scaled.round()
            } else {
                0.0
            };
            *error = if BIASED {
                (scaled - crate::group::dequantized_level_f32(quantized)) / denominator
            } else {
                (scaled - quantized) / denominator
            };
            if COMPUTE_RATE && quantized != 0.0 {
                nonzero += 1;
                magnitude_bits += rate_log2_with_lut(rate_log2_lut, quantized.abs());
                max_scan = max_scan.max(scan_pos[y * width + x]);
            }
        }
    }
    if !COMPUTE_RATE {
        return 0.0;
    }
    nonzero as f32 * R_NZ_BASE
        + R_MAG * magnitude_bits
        + R_HEADER * rate_log2_with_lut(rate_log2_lut, nonzero as f32)
        + R_ZERO * visited_zeros(nonzero, max_scan, cx, cy)
}

pub(crate) fn ssim_deficit(orig: &[f32], recon: &[f32], width: usize, height: usize) -> f32 {
    validate_ssim_inputs(orig, recon, width, height);
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { crate::avx::ssim_deficit_avx2(orig, recon, width, height) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return unsafe { crate::neon::ssim_deficit_neon(orig, recon, width, height) };
    }
    #[allow(unreachable_code)]
    ssim_deficit_scalar_kernel(orig, recon, width, height)
}

unsafe fn ssim_deficit_dispatch_kernel(
    orig: &[f32],
    recon: &[f32],
    width: usize,
    height: usize,
) -> f32 {
    ssim_deficit(orig, recon, width, height)
}

#[cfg(test)]
pub(crate) fn ssim_deficit_scalar(orig: &[f32], recon: &[f32], width: usize, height: usize) -> f32 {
    validate_ssim_inputs(orig, recon, width, height);
    ssim_deficit_scalar_kernel(orig, recon, width, height)
}

pub(crate) fn validate_ssim_inputs(orig: &[f32], recon: &[f32], width: usize, height: usize) {
    let n = width.checked_mul(height).expect("SSIM dimensions overflow");
    assert!(width != 0 && height != 0);
    assert!(width.is_multiple_of(8) && height.is_multiple_of(8));
    assert!(orig.len() >= n && recon.len() >= n);
}

/// [`ssim_deficit`] over the leading `width × height` window of
/// `stride`-wide planes, any size: windows clipped by the right/bottom edge
/// take their statistics from the pixels they hold and are weighted by that
/// count, so whole windows score exactly as the 8-aligned kernel does.
pub(crate) fn ssim_deficit_visible(
    orig: &[f32],
    recon: &[f32],
    stride: usize,
    width: usize,
    height: usize,
) -> f32 {
    const C1: f32 = 1e-4;
    const C2: f32 = 9e-4;
    assert!(width <= stride);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let n = (height - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(width))
        .expect("SSIM dimensions overflow");
    assert!(orig.len() >= n && recon.len() >= n);
    let mut deficit = 0.0f32;
    for y0 in (0..height).step_by(8) {
        let window_height = 8.min(height - y0);
        for x0 in (0..width).step_by(8) {
            let window_width = 8.min(width - x0);
            let count = (window_width * window_height) as f32;
            let inv_count = 1.0 / count;
            let rows = || {
                (y0..y0 + window_height).map(move |y| {
                    let row = y * stride + x0;
                    (
                        &orig[row..row + window_width],
                        &recon[row..row + window_width],
                    )
                })
            };
            let base_index = y0 * stride + x0;
            let base_orig = orig[base_index];
            let base_recon = recon[base_index];
            let (mut sum_orig_delta, mut sum_recon_delta) = (0.0f32, 0.0f32);
            for (orig_row, recon_row) in rows() {
                for (&orig_value, &recon_value) in orig_row.iter().zip(recon_row) {
                    sum_orig_delta += orig_value - base_orig;
                    sum_recon_delta += recon_value - base_recon;
                }
            }
            let mean_orig = base_orig + sum_orig_delta * inv_count;
            let mean_recon = base_recon + sum_recon_delta * inv_count;
            let (mut var_orig, mut var_recon, mut covariance) = (0.0f32, 0.0f32, 0.0f32);
            for (orig_row, recon_row) in rows() {
                for (&orig_value, &recon_value) in orig_row.iter().zip(recon_row) {
                    let centered_orig = orig_value - mean_orig;
                    let centered_recon = recon_value - mean_recon;
                    var_orig = fmla(centered_orig, centered_orig, var_orig);
                    var_recon = fmla(centered_recon, centered_recon, var_recon);
                    covariance = fmla(centered_orig, centered_recon, covariance);
                }
            }
            var_orig *= inv_count;
            var_recon *= inv_count;
            covariance *= inv_count;
            let luminance = (2.0 * mean_orig * mean_recon + C1)
                / (mean_orig * mean_orig + mean_recon * mean_recon + C1);
            let structure = (2.0 * covariance + C2) / (var_orig + var_recon + C2);
            deficit += (1.0 - luminance * structure) * count;
        }
    }
    deficit
}

fn ssim_deficit_scalar_kernel(orig: &[f32], recon: &[f32], width: usize, height: usize) -> f32 {
    const C1: f32 = 1e-4;
    const C2: f32 = 9e-4;
    const ONE_OVER_64: f32 = 1.0 / 64.0;
    let mut deficit = 0.0f32;
    for window_y in 0..height / 8 {
        for window_x in 0..width / 8 {
            let (x0, y0) = (window_x * 8, window_y * 8);
            let base_index = y0 * width + x0;
            let base_orig = orig[base_index];
            let base_recon = recon[base_index];
            let (mut sum_orig_delta, mut sum_recon_delta) = (0.0f32, 0.0f32);
            for (orig_row, recon_row) in orig
                .chunks_exact(width)
                .zip(recon.chunks_exact(width))
                .skip(y0)
                .take(8)
            {
                for (&orig_value, &recon_value) in orig_row[x0..x0 + 8]
                    .iter()
                    .zip(recon_row[x0..x0 + 8].iter())
                {
                    sum_orig_delta += orig_value - base_orig;
                    sum_recon_delta += recon_value - base_recon;
                }
            }
            let mean_orig = base_orig + sum_orig_delta * ONE_OVER_64;
            let mean_recon = base_recon + sum_recon_delta * ONE_OVER_64;
            let (mut var_orig, mut var_recon, mut covariance) = (0.0f32, 0.0f32, 0.0f32);
            for (orig_row, recon_row) in orig
                .chunks_exact(width)
                .zip(recon.chunks_exact(width))
                .skip(y0)
                .take(8)
            {
                for (&orig_value, &recon_value) in orig_row[x0..x0 + 8]
                    .iter()
                    .zip(recon_row[x0..x0 + 8].iter())
                {
                    let centered_orig = orig_value - mean_orig;
                    let centered_recon = recon_value - mean_recon;
                    var_orig = fmla(centered_orig, centered_orig, var_orig);
                    var_recon = fmla(centered_recon, centered_recon, var_recon);
                    covariance = fmla(centered_orig, centered_recon, covariance);
                }
            }
            var_orig *= ONE_OVER_64;
            var_recon *= ONE_OVER_64;
            covariance *= ONE_OVER_64;
            let luminance = (2.0 * mean_orig * mean_recon + C1)
                / (mean_orig * mean_orig + mean_recon * mean_recon + C1);
            let structure = (2.0 * covariance + C2) / (var_orig + var_recon + C2);
            deficit += (1.0 - luminance * structure) * 64.0;
        }
    }
    deficit
}

#[cfg(test)]
mod tests {
    use super::{
        CombineErrorFn, ErrorGradientEnergyFn, ErrorGradientPeakEnergyFn, combine_error_scalar,
        error_gradient_energy_scalar, error_gradient_peak_energy_scalar, gradient_peak_floor,
        rgb_hue_chroma_edge_loss_scalar, select_combine_error_fn, select_error_gradient_energy_fn,
        select_error_gradient_peak_energy_fn, select_rgb_hue_chroma_edge_loss_fn,
        ssim_deficit_scalar, validate_ssim_inputs,
    };

    fn pseudo_random_plane(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104_729) % 1013) as f32 / 1013.0)
            .collect()
    }

    #[test]
    fn visible_ssim_matches_aligned_kernel_on_whole_windows() {
        for (width, height) in [(8, 8), (16, 8), (32, 16), (24, 32)] {
            let orig = pseudo_random_plane(width * height, 1);
            let recon = pseudo_random_plane(width * height, 2);
            assert_eq!(
                super::ssim_deficit_visible(&orig, &recon, width, width, height).to_bits(),
                ssim_deficit_scalar(&orig, &recon, width, height).to_bits(),
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn strided_visible_kernels_match_dense_copies() {
        use super::{
            error_gradient_energy_strided, error_gradient_peak_energy_strided,
            rgb_hue_chroma_edge_loss_strided, ssim_deficit_visible,
        };
        let stride = 32;
        let height = 16;
        let original = pseudo_random_plane(stride * height, 3);
        let reconstructed = pseudo_random_plane(stride * height, 4);
        let errors: [Vec<f32>; 3] =
            std::array::from_fn(|c| pseudo_random_plane(stride * height, 5 + c));
        let mut opsin = crate::image::Image3F::new(45, 40);
        for c in 0..3 {
            for (i, v) in opsin.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *v = ((i * 31 + c * 11) % 97) as f32 * 0.004;
            }
        }
        let dense = |plane: &[f32], w: usize, h: usize| -> Vec<f32> {
            plane
                .chunks_exact(stride)
                .take(h)
                .flat_map(|row| row[..w].iter().copied())
                .collect()
        };
        for (w, h) in [(1, 1), (5, 8), (13, 11), (8, 16), (31, 3)] {
            let (o, r) = (dense(&original, w, h), dense(&reconstructed, w, h));
            assert_eq!(
                ssim_deficit_visible(&original, &reconstructed, stride, w, h).to_bits(),
                ssim_deficit_visible(&o, &r, w, w, h).to_bits(),
                "ssim {w}x{h}"
            );
            let e = dense(&errors[0], w, h);
            assert_eq!(
                error_gradient_energy_strided(&errors[0], stride, w, h).to_bits(),
                error_gradient_energy_scalar(&e, w, h).to_bits(),
                "gradient {w}x{h}"
            );
            assert_eq!(
                error_gradient_peak_energy_strided(&errors[0], &original, stride, w, h, 0.01)
                    .to_bits(),
                error_gradient_peak_energy_scalar(&e, &o, w, h, 0.01).to_bits(),
                "peak {w}x{h}"
            );
            let dense_errors: [Vec<f32>; 3] = std::array::from_fn(|c| dense(&errors[c], w, h));
            assert_eq!(
                rgb_hue_chroma_edge_loss_strided(
                    &opsin,
                    40,
                    30,
                    stride,
                    w,
                    h,
                    [&errors[0], &errors[1], &errors[2]],
                    &crate::xyb::XybMatrix::SPEC,
                )
                .to_bits(),
                rgb_hue_chroma_edge_loss_scalar(
                    &opsin,
                    40,
                    30,
                    w,
                    h,
                    [&dense_errors[0], &dense_errors[1], &dense_errors[2]],
                    &crate::xyb::XybMatrix::SPEC,
                )
                .to_bits(),
                "hue {w}x{h}"
            );
        }
    }

    #[test]
    fn footprint_views_interior_and_gathers_edges() {
        use super::Footprint;
        let mut plane = crate::image::Plane::<f32>::new(21, 19);
        for (i, v) in plane.as_mut_slice().iter_mut().enumerate() {
            *v = i as f32;
        }
        let mut tmp = [0.0f32; 4096];
        let inside = Footprint::new(21, 19, 8, 8, 8, 8);
        assert!(inside.is_full());
        assert_eq!(inside.visible_fraction(), 1.0);
        let (data, stride) = inside.pixels(&plane, &mut tmp);
        assert_eq!(stride, 21);
        assert_eq!(data[0], (8 * 21 + 8) as f32);

        let edge = Footprint::new(21, 19, 16, 16, 8, 8);
        assert_eq!((edge.visible_width, edge.visible_height), (5, 3));
        assert_eq!(edge.visible_fraction(), 15.0 / 64.0);
        let (data, stride) = edge.pixels(&plane, &mut tmp);
        assert_eq!(stride, 8);
        for y in 0..8 {
            for x in 0..8 {
                let (sx, sy) = ((16 + x).min(20), (16 + y).min(18));
                assert_eq!(data[y * 8 + x], plane.row(sy)[sx], "({x}, {y})");
            }
        }
    }

    #[test]
    fn edge_reconstruction_cost_skips_padding() {
        use super::*;
        let idct = IdctMethods::scalar();
        // `narrow` ends inside the footprint; `wide` holds the same pixels
        // plus the replicated padding as real columns/rows.
        let (narrow_w, narrow_h) = (37, 35);
        let mut narrow = Image3F::new(narrow_w, narrow_h);
        let mut wide = Image3F::new(64, 64);
        for c in 0..3 {
            for (i, v) in narrow.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *v = ((i * 17 + c * 7) % 29) as f32 * 0.013;
            }
            for y in 0..64 {
                for x in 0..64 {
                    wide.plane_row_mut(c, y)[x] =
                        narrow.plane_row(c, y.min(narrow_h - 1))[x.min(narrow_w - 1)];
                }
            }
        }
        let coeffs: [[f32; 1024]; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((i * 17 + c * 7) % 29) as f32 * 0.13 - 1.8)
        });
        let inverse: [f32; 1024] = std::array::from_fn(|i| 0.1 + (i % 7) as f32 * 0.15);
        let error = ReconErrorKernels {
            gradient_energy: select_error_gradient_energy_fn(),
            gradient_peak_energy: select_error_gradient_peak_energy_fn(),
            combine: select_combine_error_fn(),
            rgb_hue_chroma_edge_loss: select_rgb_hue_chroma_edge_loss_fn(),
        };
        for strategy in [STRATEGY_DCT, STRATEGY_DCT16X16, STRATEGY_DCT32X16] {
            let (w, h) = strategy_pixel_dims(strategy);
            let (cx, cy) = (w.max(h), w.min(h));
            let cost = |opsin: &Image3F| {
                let input = ReconDistInput {
                    idct: &idct,
                    quantization: ReconQuantization {
                        compute_rate: true,
                        rate_log2_lut: rate_log2_lut(),
                        coeffs: [&coeffs[0], &coeffs[1], &coeffs[2]],
                        inverse_matrices: [&inverse; 3],
                        qac: 7.0,
                        qm_mult_x: 1.2,
                        qm_mult_b: 1.6,
                        distance: 1.5,
                    },
                    transform: ReconTransform {
                        blocks_x: cx / 8,
                        blocks_y: cy / 8,
                        strategy,
                    },
                    source: ReconSource {
                        opsin,
                        x: 32,
                        y: 32,
                    },
                    scoring: ReconScoring {
                        factor_x: 0.15,
                        factor_b: -0.1,
                        channel_weights: CHANNEL_WEIGHT,
                        xyb_matrix: crate::xyb::XybMatrix::SPEC,
                        rgb_hue_alpha: 800.0,
                        gradient_alpha: 8.0,
                        gradient_peak_alpha: 3.0,
                        keep_spatial_errors: false,
                    },
                };
                recon_dist_and_rate_default(&mut [[0.0; 1024]; 8], &input, &error)
            };
            let (visible, padded) = (cost(&narrow), cost(&wide));
            assert_eq!(visible.rate.to_bits(), padded.rate.to_bits());
            assert!(
                visible.distortion < padded.distortion,
                "strategy {strategy}: {} vs {}",
                visible.distortion,
                padded.distortion
            );
        }
    }

    #[test]
    fn omitting_fixed_rate_preserves_reconstruction_and_all_distortion_terms() {
        use super::*;
        let idct = IdctMethods::scalar();
        let mut opsin = Image3F::new(37, 35);
        for c in 0..3 {
            for (i, value) in opsin.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *value = ((i * 17 + c * 7) % 29) as f32 * 0.013;
            }
        }
        let coeffs: [[f32; 1024]; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((i * 17 + c * 7) % 29) as f32 * 0.13 - 1.8)
        });
        let inverse: [f32; 1024] = std::array::from_fn(|i| 0.1 + (i % 7) as f32 * 0.15);
        let error = ReconErrorKernels {
            gradient_energy: select_error_gradient_energy_fn(),
            gradient_peak_energy: select_error_gradient_peak_energy_fn(),
            combine: select_combine_error_fn(),
            rgb_hue_chroma_edge_loss: select_rgb_hue_chroma_edge_loss_fn(),
        };
        let kernels: [ReconDistAndRateFn; 3] = [
            select_recon_dist_and_rate_fn(),
            recon_dist_and_rate_default,
            |scratch, input, _| recon_dist_and_rate_scalar::<false>(scratch, input),
        ];
        for strategy in (0..NUM_STRATEGIES as u8)
            .filter(|&s| !matches!(s, STRATEGY_DCT64X64 | STRATEGY_DCT64X32 | STRATEGY_DCT32X64))
        {
            let (w, h) = strategy_pixel_dims(strategy);
            let (cx, cy) = (w.max(h), w.min(h));
            for (px, py) in [(0, 0), (3, 5), (34, 33)] {
                for qac in [0.5, 7.0, 80.0] {
                    let mut input = ReconDistInput {
                        idct: &idct,
                        quantization: ReconQuantization {
                            compute_rate: true,
                            rate_log2_lut: rate_log2_lut(),
                            coeffs: [&coeffs[0], &coeffs[1], &coeffs[2]],
                            inverse_matrices: [&inverse; 3],
                            qac,
                            qm_mult_x: 1.2,
                            qm_mult_b: 1.6,
                            distance: 1.5,
                        },
                        transform: ReconTransform {
                            blocks_x: cx / 8,
                            blocks_y: cy / 8,
                            strategy,
                        },
                        source: ReconSource {
                            opsin: &opsin,
                            x: px,
                            y: py,
                        },
                        scoring: ReconScoring {
                            factor_x: 0.15,
                            factor_b: -0.1,
                            channel_weights: CHANNEL_WEIGHT,
                            xyb_matrix: crate::xyb::XybMatrix::SPEC,
                            rgb_hue_alpha: 800.0,
                            gradient_alpha: 8.0,
                            gradient_peak_alpha: 3.0,
                            keep_spatial_errors: true,
                        },
                    };
                    for kernel in kernels {
                        let mut expected = [[-123.0; 1024]; 8];
                        let mut actual = expected;
                        input.quantization.compute_rate = true;
                        let full = kernel(&mut expected, &input, &error);
                        input.quantization.compute_rate = false;
                        let without_rate = kernel(&mut actual, &input, &error);
                        assert_eq!(without_rate.rate, 0.0);
                        assert_eq!(without_rate.distortion.to_bits(), full.distortion.to_bits());
                        assert_eq!(
                            without_rate.hue_distortion.to_bits(),
                            full.hue_distortion.to_bits()
                        );
                        for (expected, actual) in
                            expected.iter().flatten().zip(actual.iter().flatten())
                        {
                            assert_eq!(
                                expected.to_bits(),
                                actual.to_bits(),
                                "strategy {strategy}, qac {qac}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reported_hue_cost_matches_the_reused_reconstruction_errors() {
        use super::*;
        let matrix = crate::xyb::XybMatrix::SPEC;
        let mut opsin = Image3F::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let rgb = if (x + y) % 3 == 0 {
                    [0.9, 0.85, 0.05]
                } else {
                    [0.02, 0.03, 0.8]
                };
                let (a, b, c) = crate::xyb::rgb_to_xyb_pixel_f32(&matrix, rgb[0], rgb[1], rgb[2]);
                for (channel, value) in [a, b, c].into_iter().enumerate() {
                    opsin.plane_row_mut(channel, y)[x] = value;
                }
            }
        }
        let coeffs: [[f32; 64]; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((i * 17 + c * 7) % 29) as f32 * 0.013 - 0.18)
        });
        let inverse = [1.0; 64];
        let idct = IdctMethods::scalar();
        let mut input = ReconDistInput {
            idct: &idct,
            quantization: ReconQuantization {
                compute_rate: true,
                rate_log2_lut: rate_log2_lut(),
                coeffs: [&coeffs[0], &coeffs[1], &coeffs[2]],
                inverse_matrices: [&inverse; 3],
                qac: 7.0,
                qm_mult_x: 1.2,
                qm_mult_b: 1.6,
                distance: 4.0,
            },
            transform: ReconTransform {
                blocks_x: 1,
                blocks_y: 1,
                strategy: crate::dc_group_data::STRATEGY_DCT,
            },
            source: ReconSource {
                opsin: &opsin,
                x: 0,
                y: 0,
            },
            scoring: ReconScoring {
                factor_x: 0.15,
                factor_b: -0.1,
                channel_weights: CHANNEL_WEIGHT,
                xyb_matrix: matrix,
                rgb_hue_alpha: 0.0,
                gradient_alpha: 0.0,
                gradient_peak_alpha: 0.0,
                keep_spatial_errors: true,
            },
        };
        let error = ReconErrorKernels {
            gradient_energy: select_error_gradient_energy_fn(),
            gradient_peak_energy: select_error_gradient_peak_energy_fn(),
            combine: select_combine_error_fn(),
            rgb_hue_chroma_edge_loss: select_rgb_hue_chroma_edge_loss_fn(),
        };
        let reconstruct = select_recon_dist_and_rate_fn();
        let mut scratch = [[0.0; 1024]; 8];
        let base = reconstruct(&mut scratch, &input, &error);
        assert_eq!(base.hue_distortion, 0.0);
        input.scoring.rgb_hue_alpha = 800.0;
        let scored = reconstruct(&mut scratch, &input, &error);
        let expected = 800.0
            * unsafe {
                (error.rgb_hue_chroma_edge_loss)(
                    &opsin,
                    0,
                    0,
                    8,
                    8,
                    [&scratch[0][..64], &scratch[1][..64], &scratch[2][..64]],
                    &matrix,
                )
            };
        assert!(expected > 0.0);
        assert_eq!(scored.hue_distortion, expected);
        assert_eq!(scored.rate, base.rate);
        assert_eq!(scored.distortion, base.distortion + expected);
    }

    #[test]
    fn hue_reversal_costs_more_than_losing_chroma() {
        let source = [0.2, 0.0];
        let loss = |reconstructed| super::hue_chroma_error(source, reconstructed, 0.2);
        assert_eq!(loss(source), 0.0);
        assert_eq!(loss([0.3, 0.0]), 0.0); // Same-hue oversaturation is not the target.
        assert!(loss([0.0, 0.2]) > loss([0.0, 0.0]));
        assert!(loss([-0.2, 0.0]) > loss([0.0, 0.2]));
        assert!((loss([-0.2, 0.0]) - 4.0 * loss([0.0, 0.0])).abs() < 1e-6);
        assert_eq!(super::hue_chroma_error([0.0; 2], [0.0; 2], 0.0), 0.0);
    }

    #[test]
    fn dark_blue_edges_keep_a_hue_penalty_below_the_old_lightness_cutoff() {
        let matrix = crate::xyb::XybMatrix::SPEC;
        let blue_rgb = [0.0, 0.0, 0.2];
        assert!(super::linear_rgb_to_oklab(blue_rgb)[0] < 0.35);
        let blue = crate::xyb::rgb_to_xyb_pixel_f32(&matrix, 0.0, 0.0, 0.2);
        let gray = crate::xyb::rgb_to_xyb_pixel_f32(&matrix, 0.02, 0.02, 0.02);
        let mut opsin = crate::image::Image3F::new(8, 8);
        let mut error = [[0.0; 64]; 3];
        for y in 0..8 {
            for x in 0..4 {
                for (c, value) in [blue.0, blue.1, blue.2].into_iter().enumerate() {
                    opsin.plane_row_mut(c, y)[x] = value;
                    error[c][y * 8 + x] = value - [gray.0, gray.1, gray.2][c];
                }
            }
        }
        let errors = [&error[0][..], &error[1][..], &error[2][..]];
        let expected = rgb_hue_chroma_edge_loss_scalar(&opsin, 0, 0, 8, 8, errors, &matrix);
        assert!(
            expected > 1e-4,
            "dark blue desaturation was unpenalized: {expected}"
        );
        let actual =
            unsafe { select_rgb_hue_chroma_edge_loss_fn()(&opsin, 0, 0, 8, 8, errors, &matrix) };
        assert!((actual / expected - 1.0).abs() < 2e-4);
        let zero = [0.0; 64];
        assert_eq!(
            unsafe {
                select_rgb_hue_chroma_edge_loss_fn()(&opsin, 0, 0, 8, 8, [&zero; 3], &matrix)
            },
            0.0
        );
    }

    #[test]
    fn rgb_hue_loss_simd_matches_scalar() {
        let matrix = crate::xyb::XybMatrix::SPEC;
        let mut opsin = crate::image::Image3F::new(24, 16);
        let mut state = 0x91e1_0da5_c79e_7b1du64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32 as f32 / u32::MAX as f32
        };
        for y in 0..16 {
            for x in 0..24 {
                let rgb = [random(), random(), random()];
                let value = crate::xyb::rgb_to_xyb_pixel_f32(&matrix, rgb[0], rgb[1], rgb[2]);
                opsin.plane_row_mut(0, y)[x] = value.0;
                opsin.plane_row_mut(1, y)[x] = value.1;
                opsin.plane_row_mut(2, y)[x] = value.2;
            }
        }
        let mut errors = [[0.0f32; 128]; 3];
        for error in &mut errors {
            for value in error {
                *value = (random() - 0.5) * 0.08;
            }
        }
        let spatial_error = [&errors[0][..], &errors[1][..], &errors[2][..]];
        let expected = rgb_hue_chroma_edge_loss_scalar(&opsin, 4, 3, 16, 8, spatial_error, &matrix);
        let actual = unsafe {
            select_rgb_hue_chroma_edge_loss_fn()(&opsin, 4, 3, 16, 8, spatial_error, &matrix)
        };
        let relative_error = (actual - expected).abs() / expected.abs().max(1e-9);
        assert!(
            relative_error < 2e-4,
            "SIMD RGB hue loss {actual} differs from scalar {expected} by {relative_error}"
        );
    }

    #[test]
    fn rgb_hue_loss_is_zero_for_identity_and_positive_for_edge_desaturation() {
        let matrix = crate::xyb::XybMatrix::SPEC;
        let mut opsin = crate::image::Image3F::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let rgb = if x < 4 {
                    [1.0, 0.9, 0.03]
                } else {
                    [0.03, 0.25, 1.0]
                };
                let value = crate::xyb::rgb_to_xyb_pixel_f32(&matrix, rgb[0], rgb[1], rgb[2]);
                opsin.plane_row_mut(0, y)[x] = value.0;
                opsin.plane_row_mut(1, y)[x] = value.1;
                opsin.plane_row_mut(2, y)[x] = value.2;
            }
        }
        let zero = [0.0f32; 64];
        assert_eq!(
            unsafe {
                select_rgb_hue_chroma_edge_loss_fn()(
                    &opsin,
                    0,
                    0,
                    8,
                    8,
                    [&zero, &zero, &zero],
                    &matrix,
                )
            },
            0.0
        );

        let gray = crate::xyb::rgb_to_xyb_pixel_f32(&matrix, 0.5, 0.5, 0.5);
        let mut errors = [[0.0f32; 64]; 3];
        for y in 0..8 {
            for x in 0..8 {
                let i = y * 8 + x;
                errors[0][i] = opsin.plane_row(0, y)[x] - gray.0;
                errors[1][i] = opsin.plane_row(1, y)[x] - gray.1;
                errors[2][i] = opsin.plane_row(2, y)[x] - gray.2;
            }
        }
        let loss = unsafe {
            select_rgb_hue_chroma_edge_loss_fn()(
                &opsin,
                0,
                0,
                8,
                8,
                [&errors[0], &errors[1], &errors[2]],
                &matrix,
            )
        };
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn peak_gradient_energy_favors_localized_errors() {
        let mut diffuse = [0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                diffuse[y * 8 + x] = x as f32;
            }
        }
        let mut localized = [0.0f32; 64];
        for y in 0..8 {
            localized[y * 8 + 4..y * 8 + 8].fill(1.0);
        }
        let diffuse_raw = error_gradient_energy_scalar(&diffuse, 8, 8);
        let localized_raw = error_gradient_energy_scalar(&localized, 8, 8);
        let original = [0.0f32; 64];
        let diffuse_peak = error_gradient_peak_energy_scalar(&diffuse, &original, 8, 8, 0.0);
        let localized_peak = error_gradient_peak_energy_scalar(&localized, &original, 8, 8, 0.0);
        assert!(diffuse_raw > localized_raw);
        assert!(diffuse_raw / localized_raw > diffuse_peak / localized_peak);
    }

    #[test]
    fn peak_gradient_energy_exposes_edge_displacement() {
        let mut original = [0.0f32; 64];
        let mut aligned_error = [0.0f32; 64];
        let mut displaced_error = [0.0f32; 64];
        for y in 0..8 {
            original[y * 8 + 4..y * 8 + 8].fill(1.0);
            aligned_error[y * 8 + 4..y * 8 + 8].fill(0.5);
            displaced_error[y * 8 + 4] = 1.0;
        }
        let aligned = error_gradient_peak_energy_scalar(&aligned_error, &original, 8, 8, 0.0);
        let displaced = error_gradient_peak_energy_scalar(&displaced_error, &original, 8, 8, 0.0);
        assert_eq!(aligned, 0.0);
        assert!(displaced > 0.0);
    }

    #[test]
    fn peak_gradient_floor_rises_for_the_coarse_tail() {
        assert!((gradient_peak_floor(1.9) - 0.0015 * 1.9).abs() < 1e-7);
        assert!((gradient_peak_floor(2.0) - 0.0045 * 2.0).abs() < 1e-7);
        assert!((gradient_peak_floor(3.0) - 0.0045 * 3.0).abs() < 1e-7);
    }

    fn check_combine_error(kernel: CombineErrorFn) {
        for len in (0usize..=40).chain([65, 257, 1024]) {
            let spatial: Vec<f32> = (0..len)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03125)
                .collect();
            let luma: Vec<f32> = (0..len)
                .map(|i| ((i * 53 % 113) as f32 - 56.0) * 0.015625)
                .collect();
            for factor in [-1.25, -0.1, 0.0, 0.3, 1.75] {
                let mut expected = vec![f32::NAN; len];
                let mut actual = vec![f32::NAN; len];
                combine_error_scalar(&spatial, &luma, factor, &mut expected);
                kernel(&spatial, &luma, factor, &mut actual);
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    let tolerance = 2e-7f32.max(expected.abs() * 2e-7);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "len={len}, index={index}, factor={factor}: actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn selected_combine_error_matches_scalar() {
        check_combine_error(select_combine_error_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_combine_error_matches_scalar() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            check_combine_error(|spatial, luma, factor, combined| unsafe {
                crate::avx::combine_error_avx2(spatial, luma, factor, combined)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_combine_error_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_combine_error(|spatial, luma, factor, combined| unsafe {
                crate::sse::combine_error_sse41(spatial, luma, factor, combined)
            });
        }
    }

    fn check_error_gradient_energy(kernel: ErrorGradientEnergyFn) {
        for &(width, height) in &[
            (0usize, 0usize),
            (1, 1),
            (1, 7),
            (7, 1),
            (3, 5),
            (8, 8),
            (16, 8),
            (17, 11),
            (32, 32),
        ] {
            let mut state = 0x9e37_79b9u32;
            let values: Vec<f32> = (0..width * height)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 8.0
                })
                .collect();
            let expected = error_gradient_energy_scalar(&values, width, height);
            let actual = kernel(&values, width, height);
            let tolerance = 2e-5f32.max(expected.abs() * 3e-6);
            assert!(
                (actual - expected).abs() <= tolerance,
                "shape {width}x{height}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn selected_error_gradient_energy_matches_scalar() {
        check_error_gradient_energy(select_error_gradient_energy_fn());
    }

    fn check_error_gradient_peak_energy(kernel: ErrorGradientPeakEnergyFn) {
        for &(width, height) in &[
            (0usize, 0usize),
            (1, 1),
            (1, 7),
            (7, 1),
            (3, 5),
            (4, 4),
            (8, 8),
            (16, 8),
            (8, 16),
            (17, 11),
            (32, 32),
        ] {
            let mut state = 0x243f_6a88u32;
            let mut random = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32
            };
            let error: Vec<f32> = (0..width * height)
                .map(|_| (random() - 0.5) * 0.2)
                .collect();
            let original: Vec<f32> = (0..width * height).map(|_| random()).collect();
            for floor in [0.0, 0.0015, 0.009, 0.05] {
                let expected =
                    error_gradient_peak_energy_scalar(&error, &original, width, height, floor);
                let actual = kernel(&error, &original, width, height, floor);
                let tolerance = 2e-7f32.max(expected.abs() * 2e-6);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "shape {width}x{height}, floor={floor}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn selected_error_gradient_peak_energy_matches_scalar() {
        check_error_gradient_peak_energy(select_error_gradient_peak_energy_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_error_gradient_energy_matches_scalar() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            check_error_gradient_energy(|error, width, height| unsafe {
                crate::avx::error_gradient_energy_avx2(error, width, height)
            });
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_error_gradient_peak_energy_matches_scalar() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            check_error_gradient_peak_energy(|error, original, width, height, floor| unsafe {
                crate::avx::error_gradient_peak_energy_avx2(error, original, width, height, floor)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_error_gradient_energy_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_error_gradient_energy(|error, width, height| unsafe {
                crate::sse::error_gradient_energy_sse41(error, width, height)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_error_gradient_peak_energy_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_error_gradient_peak_energy(|error, original, width, height, floor| unsafe {
                crate::sse::error_gradient_peak_energy_sse41(error, original, width, height, floor)
            });
        }
    }

    fn reference(orig: &[f32], recon: &[f32], width: usize, height: usize) -> f64 {
        const C1: f64 = 1e-4;
        const C2: f64 = 9e-4;
        let mut deficit = 0.0;
        for wy in 0..height / 8 {
            for wx in 0..width / 8 {
                let mut o = [0.0f64; 64];
                let mut r = [0.0f64; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        let j = y * 8 + x;
                        let i = (wy * 8 + y) * width + wx * 8 + x;
                        o[j] = orig[i] as f64;
                        r[j] = recon[i] as f64;
                    }
                }
                let mo = o.iter().sum::<f64>() / 64.0;
                let mr = r.iter().sum::<f64>() / 64.0;
                let vo = o.iter().map(|v| (v - mo) * (v - mo)).sum::<f64>() / 64.0;
                let vr = r.iter().map(|v| (v - mr) * (v - mr)).sum::<f64>() / 64.0;
                let cov = o
                    .iter()
                    .zip(r.iter())
                    .map(|(a, b)| (a - mo) * (b - mr))
                    .sum::<f64>()
                    / 64.0;
                let l = (2.0 * mo * mr + C1) / (mo * mo + mr * mr + C1);
                let s = (2.0 * cov + C2) / (vo + vr + C2);
                deficit += (1.0 - l * s) * 64.0;
            }
        }
        deficit
    }

    #[test]
    fn ssim_matches_f64_reference_and_is_symmetric() {
        let mut a = [0.0f32; 256];
        let mut b = [0.0f32; 256];
        let mut state = 7u32;
        for i in 0..256 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            a[i] = (state >> 8) as f32 / (1u32 << 24) as f32;
            b[i] = a[i] + ((i % 11) as f32 - 5.0) * 0.001;
        }
        let got = ssim_deficit_scalar(&a, &b, 16, 16);
        let want = reference(&a, &b, 16, 16) as f32;
        assert!((got - want).abs() < 2e-4, "got={got} want={want}");
        let reverse = ssim_deficit_scalar(&b, &a, 16, 16);
        assert!((got - reverse).abs() < 1e-5);
    }

    #[test]
    fn ssim_identity_and_high_offset_are_stable() {
        let identical = [0.42f32; 64];
        assert_eq!(ssim_deficit_scalar(&identical, &identical, 8, 8), 0.0);
        let mut a = [10_000.0f32; 64];
        let mut b = [10_000.0f32; 64];
        for i in 0..64 {
            a[i] += (i % 3) as f32 * 0.01;
            b[i] += (i % 5) as f32 * 0.01;
        }
        let value = ssim_deficit_scalar(&a, &b, 8, 8);
        assert!(value.is_finite());
        let expected = reference(&a, &b, 8, 8) as f32;
        assert!(
            (value - expected).abs() < 5e-3,
            "value={value} expected={expected}"
        );
    }

    #[test]
    fn ssim_rejects_invalid_shapes() {
        assert!(
            std::panic::catch_unwind(|| validate_ssim_inputs(&[0.0; 63], &[0.0; 64], 8, 8))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| validate_ssim_inputs(&[0.0; 64], &[0.0; 64], 7, 8))
                .is_err()
        );
    }
}
