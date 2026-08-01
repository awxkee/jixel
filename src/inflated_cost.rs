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
    STRATEGY_DCT, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4, STRATEGY_DCT8X16,
    STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16, STRATEGY_DCT32X32,
};
use crate::dct::{DctInput, fmla};
use crate::image::{Image3F, Plane};
use std::sync::OnceLock;

const R_NZ_BASE: f32 = 1.6;
const R_MAG: f32 = 1.0;
const R_HEADER: f32 = 0.4;
// Per-channel distortion weights (X, Y, B).
pub(crate) static CHANNEL_WEIGHT: [f32; 3] = [0.10, 1.0, 0.83];

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
) -> (f32, usize, f32);

fn select_sse_and_rate_fn() -> SseAndRateFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return crate::avx::sse_and_rate_avx2;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::sse_and_rate_sse;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        crate::neon::sse_and_rate_neon
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::sse_and_rate_wasm
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        sse_and_rate_scalar
    }
}

static SSE_AND_RATE_FN: OnceLock<SseAndRateFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_sse_and_rate_fn() -> SseAndRateFn {
    *SSE_AND_RATE_FN.get_or_init(select_sse_and_rate_fn)
}

#[cfg(test)]
#[allow(dead_code)] // Used by target-specific SSE/WASM test modules.
pub(crate) fn assert_sse_and_rate_matches_reference(kernel: SseAndRateFn) {
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
                    let d = a - q;
                    expected_sse += d * d;
                    if q != 0.0 {
                        expected_nzeros += 1;
                        expected_mag += (1.0 + q.abs()).log2();
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
                )
            };
            assert_eq!(
                actual.1, expected_nzeros,
                "nzeros mismatch {width}x{height}"
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

#[allow(unused, clippy::too_many_arguments)]
pub(crate) fn sse_and_rate_scalar(
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
) -> (f32, usize, f32) {
    let mut sse = 0.0f32;
    let mut nzeros = 0usize;
    let mut mag_bits = 0.0f32;
    for (y, (coeff_row, inv_row)) in coeff
        .chunks_exact(width)
        .zip(inv_matrix.chunks_exact(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        for (x, (&coefficient, &inverse)) in coeff_row.iter().zip(inv_row.iter()).enumerate() {
            if x < cx && y < cy {
                continue;
            }
            let threshold = if x >= half { thr[yfix + 1] } else { thr[yfix] };
            let a = inverse * q_scaled * coefficient;
            let q = if a.abs() >= threshold { a.round() } else { 0.0 };
            let d = a - q;
            sse += d * d;
            if q != 0.0 {
                nzeros += 1;
                mag_bits += rate_log2_with_lut(rate_log2_lut, q.abs());
            }
        }
    }
    (sse, nzeros, mag_bits)
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
    let thr = crate::enc_group::quantize_ac_thresholds(channel, cx, cy, distance);
    let (sse, nzeros, mag_bits) = unsafe {
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
        )
    };
    let header = R_HEADER * rate_log2_with_lut(rate_log2_lut, nzeros as f32);
    let bits = nzeros as f32 * R_NZ_BASE + R_MAG * mag_bits + header;
    (sse, bits)
}

pub(crate) fn strategy_pixel_dims(strategy: u8) -> (usize, usize) {
    match strategy {
        STRATEGY_DCT16X8 => (8, 16),
        STRATEGY_DCT8X16 => (16, 8),
        STRATEGY_DCT16X16 => (16, 16),
        STRATEGY_DCT32X16 => (16, 32),
        STRATEGY_DCT16X32 => (32, 16),
        STRATEGY_DCT32X32 => (32, 32),
        _ => (8, 8),
    }
}

pub(crate) fn strategy_pixel_count(strategy: u8) -> usize {
    let (w, h) = strategy_pixel_dims(strategy);
    w * h
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
    match strategy {
        STRATEGY_DCT4X4 => fwd!(dct::dct4x4, 64),
        STRATEGY_DCT4X8 => fwd!(dct::dct4x8, 64),
        STRATEGY_DCT8X4 => fwd!(dct::dct8x4, 64),
        STRATEGY_DCT16X8 => fwd!(dct::dct16x8, 128),
        STRATEGY_DCT8X16 => fwd!(dct::dct8x16, 128),
        STRATEGY_DCT16X16 => fwd!(dct::dct16x16, 256),
        STRATEGY_DCT32X32 => fwd!(dct::dct32x32, 1024),
        STRATEGY_DCT32X16 => fwd!(dct::dct32x16, 512),
        STRATEGY_DCT16X32 => fwd!(dct::dct16x32, 512),
        _ => fwd!(dct::dct8x8, 64),
    }
}

pub(crate) fn forward_matrix(strategy: u8) -> &'static [f32] {
    static MATRICES: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    &MATRICES.get_or_init(|| {
        (0u8..10)
            .map(|s| {
                let n = strategy_pixel_count(s);
                let mut matrix = vec![0.0f32; n * n];
                let mut input = [0.0f32; 1024];
                let mut output = [0.0f32; 1024];
                for impulse in 0..n {
                    input[..n].fill(0.0);
                    input[impulse] = 1.0;
                    forward_for(s, &input, &mut output);
                    matrix[impulse * n..impulse * n + n].copy_from_slice(&output[..n]);
                }
                matrix
            })
            .collect()
    })[strategy as usize]
}

macro_rules! idct_simd_or_scalar {
    ($name:ident, $n:literal, $w:literal, $h:literal, $avx:path, $neon:path, $scalar:path) => {
        #[inline]
        fn $name(c: DctInput<'_, $w, $h>, o: &mut [f32; $n]) {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::is_x86_feature_detected!("avx2") {
                unsafe { $avx(c, o) };
                return;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                unsafe { $neon(c, o) };
                return;
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            $scalar(c, o)
        }
    };
}

idct_simd_or_scalar!(
    idct8x8,
    64,
    8,
    8,
    crate::avx::inv_dct8x8_avx2,
    crate::neon::inv_dct8x8_neon,
    crate::dct::inv_dct8x8
);
idct_simd_or_scalar!(
    idct8x16,
    128,
    16,
    8,
    crate::avx::inv_dct8x16_avx2,
    crate::neon::inv_dct8x16_neon,
    crate::dct::inv_dct8x16
);
idct_simd_or_scalar!(
    idct16x8,
    128,
    16,
    8,
    crate::avx::inv_dct16x8_avx2,
    crate::neon::inv_dct16x8_neon,
    crate::dct::inv_dct16x8
);
idct_simd_or_scalar!(
    idct16x16,
    256,
    16,
    16,
    crate::avx::inv_dct16x16_avx2,
    crate::neon::inv_dct16x16_neon,
    crate::dct::inv_dct16x16
);
idct_simd_or_scalar!(
    idct16x32,
    512,
    32,
    16,
    crate::avx::inv_dct16x32_avx2,
    crate::neon::inv_dct16x32_neon,
    crate::dct::inv_dct16x32
);
idct_simd_or_scalar!(
    idct32x16,
    512,
    32,
    16,
    crate::avx::inv_dct32x16_avx2,
    crate::neon::inv_dct32x16_neon,
    crate::dct::inv_dct32x16
);
idct_simd_or_scalar!(
    idct32x32,
    1024,
    32,
    32,
    crate::avx::inv_dct32x32_avx2,
    crate::neon::inv_dct32x32_neon,
    crate::dct::inv_dct32x32
);

pub(crate) fn reconstruct_error(strategy: u8, coeff_err: &[f32], err_out: &mut [f32]) {
    macro_rules! inverse {
        ($f:path, $n:literal, $w:literal, $h:literal) => {
            $f(
                DctInput::<$w, $h>::new(coeff_err, $w),
                err_out.first_chunk_mut::<$n>().unwrap(),
            )
        };
    }
    match strategy {
        STRATEGY_DCT => inverse!(idct8x8, 64, 8, 8),
        STRATEGY_DCT8X16 => inverse!(idct8x16, 128, 16, 8),
        STRATEGY_DCT16X8 => inverse!(idct16x8, 128, 16, 8),
        STRATEGY_DCT16X16 => inverse!(idct16x16, 256, 16, 16),
        STRATEGY_DCT16X32 => inverse!(idct16x32, 512, 32, 16),
        STRATEGY_DCT32X16 => inverse!(idct32x16, 512, 32, 16),
        STRATEGY_DCT32X32 => inverse!(idct32x32, 1024, 32, 32),
        _ => {
            let n = strategy_pixel_count(strategy);
            let matrix = forward_matrix(strategy);
            for (row, value) in err_out[..n].iter_mut().enumerate() {
                let base = row * n;
                *value = n as f32 * (0..n).map(|k| matrix[base + k] * coeff_err[k]).sum::<f32>();
            }
        }
    }
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
) -> f32;

pub(crate) type SsimDeficitFn = unsafe fn(&[f32], &[f32], usize, usize) -> f32;
pub(crate) type PrepareReconstructionFn =
    unsafe fn(&Plane<f32>, usize, usize, usize, usize, &[f32], &mut [f32], &mut [f32]);
pub(crate) type ErrorGradientEnergyFn = fn(&[f32], usize, usize) -> f32;
pub(crate) type CombineErrorFn = fn(&[f32], &[f32], f32, &mut [f32]);

pub(crate) struct ReconQuantization<'a> {
    pub(crate) rate_log2_lut: &'a RateLog2Lut,
    pub(crate) coeffs: &'a [[f32; 1024]; 3],
    pub(crate) inverse_matrices: [&'a [f32]; 3],
    pub(crate) qac: f32,
    pub(crate) qm_mult_x: f32,
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
    pub(crate) banding: bool,
}

pub(crate) struct ReconDistInput<'a> {
    pub(crate) quantization: ReconQuantization<'a>,
    pub(crate) transform: ReconTransform,
    pub(crate) source: ReconSource<'a>,
    pub(crate) scoring: ReconScoring,
}

pub(crate) struct ReconErrorKernels {
    pub(crate) gradient_energy: ErrorGradientEnergyFn,
    pub(crate) combine: CombineErrorFn,
}

pub(crate) struct ReconKernels<'a> {
    pub(crate) quantize: ReconQuantizeFn,
    pub(crate) ssim: SsimDeficitFn,
    pub(crate) prepare: PrepareReconstructionFn,
    pub(crate) error: &'a ReconErrorKernels,
}

pub(crate) type ReconDistAndRateFn = for<'a, 'input, 'kernels> fn(
    &mut [[f32; 1024]; 8],
    &'input ReconDistInput<'a>,
    &'kernels ReconErrorKernels,
) -> (f32, f32);

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
) -> (f32, f32) {
    recon_dist_and_rate_with_kernels(
        scratch,
        input,
        &ReconKernels {
            quantize: recon_quantize_scalar,
            ssim: ssim_deficit_dispatch_kernel,
            prepare: prepare_reconstruction_scalar,
            error,
        },
    )
}

#[cfg(test)]
pub(crate) fn recon_dist_and_rate_scalar(
    scratch: &mut [[f32; 1024]; 8],
    input: &ReconDistInput<'_>,
) -> (f32, f32) {
    let error = ReconErrorKernels {
        gradient_energy: error_gradient_energy_scalar,
        combine: combine_error_scalar,
    };
    recon_dist_and_rate_with_kernels(
        scratch,
        input,
        &ReconKernels {
            quantize: recon_quantize_scalar,
            ssim: ssim_deficit_scalar_kernel,
            prepare: prepare_reconstruction_scalar,
            error: &error,
        },
    )
}

/// Banding protection
const BANDING_ALPHA: f32 = 46.5;
const BANDING_MIN_D: f32 = 1.4;

/// `sum((dx err)^2 + (dy err)^2)` over one channel's spatial error plane.
#[allow(dead_code)]
pub(crate) fn error_gradient_energy_scalar(error: &[f32], width: usize, height: usize) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("gradient plane size overflow");
    assert!(error.len() >= n);
    if width == 0 || height == 0 {
        return 0.0;
    }
    let rows = error[..n].chunks_exact(width);
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

pub(crate) fn recon_dist_and_rate_with_kernels(
    scratch: &mut [[f32; 1024]; 8],
    input: &ReconDistInput<'_>,
    kernels: &ReconKernels<'_>,
) -> (f32, f32) {
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
    let banding = scoring.banding;
    let quantize = kernels.quantize;
    let ssim = kernels.ssim;
    let prepare_reconstruction = kernels.prepare;
    let error_gradient_energy = kernels.error.gradient_energy;
    let combine_error = kernels.error.combine;
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
        crate::enc_group::quantize_ac_thresholds(0, cx, cy, distance),
        crate::enc_group::quantize_ac_thresholds(1, cx, cy, distance),
        crate::enc_group::quantize_ac_thresholds(2, cx, cy, distance),
    ];
    let quant_scales = [qac * qm_mult_x, qac, qac * crate::enc_frame::b_qm_mul()];

    let (coeff_error, rest) = scratch.split_at_mut(3);
    let [
        spatial_error,
        y_error,
        combined_error,
        reconstructed,
        original,
    ] = rest
    else {
        unreachable!()
    };
    let mut rate = 0.0f32;
    for c in 0..3 {
        rate += unsafe {
            quantize(
                &coeffs[c][..n],
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
            )
        };
    }

    reconstruct_error(strategy, &coeff_error[1][..n], &mut y_error[..n]);
    let mut distortion = 0.0f32;
    for c in 0..3 {
        let factor = if c == 0 { factor_x } else { factor_b };
        let error: &[f32] = if c == 1 {
            &y_error[..n]
        } else {
            reconstruct_error(strategy, &coeff_error[c][..n], &mut spatial_error[..n]);
            combine_error(
                &spatial_error[..n],
                &y_error[..n],
                factor,
                &mut combined_error[..n],
            );
            &combined_error[..n]
        };
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
        distortion += CHANNEL_WEIGHT[c]
            * unsafe {
                ssim(
                    &original[..n],
                    &reconstructed[..n],
                    pixel_width,
                    pixel_height,
                )
            };
        if banding && distance >= BANDING_MIN_D {
            distortion += CHANNEL_WEIGHT[c]
                * BANDING_ALPHA
                * error_gradient_energy(&error[..n], pixel_width, pixel_height);
        }
    }
    (distortion, rate)
}

#[allow(clippy::too_many_arguments)]
unsafe fn prepare_reconstruction_scalar(
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
unsafe fn recon_quantize_scalar(
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
) -> f32 {
    let n = width
        .checked_mul(height)
        .expect("coefficient size overflow");
    assert!(coeff.len() >= n && inv.len() >= n && coeff_error.len() >= n);
    let (mut nonzero, mut magnitude_bits) = (0usize, 0.0f32);
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
            *error = (scaled - quantized) / denominator;
            if quantized != 0.0 {
                nonzero += 1;
                magnitude_bits += rate_log2_with_lut(rate_log2_lut, quantized.abs());
            }
        }
    }
    nonzero as f32 * R_NZ_BASE
        + R_MAG * magnitude_bits
        + R_HEADER * rate_log2_with_lut(rate_log2_lut, nonzero as f32)
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
        CombineErrorFn, ErrorGradientEnergyFn, combine_error_scalar, error_gradient_energy_scalar,
        select_combine_error_fn, select_error_gradient_energy_fn, ssim_deficit_scalar,
        validate_ssim_inputs,
    };

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

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_error_gradient_energy_matches_scalar() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            check_error_gradient_energy(|error, width, height| unsafe {
                crate::avx::error_gradient_energy_avx2(error, width, height)
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
