/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! Decoder-side grain synthesis: the JPEG XL noise feature (frame flag
//! `kNoise`).
//!
//! The decoder draws three planes of white noise, high-passes them with a
//! fixed 5x5 kernel and adds them to the XYB image scaled by an 8-point
//! strength LUT indexed by the pixel's intensity `(Y ± X) / 2`. The LUT is
//! the whole encoder-side parameter space: 8 values in `[0, 1)`, 10 bits
//! each, written in LfGlobal after the patch and spline dictionaries.

use crate::bit_writer::BitWriter;
use crate::image::Image3F;

#[derive(Clone, Copy)]
pub(crate) struct NoiseKernels {
    pub(crate) shrink: fn(&mut [f32; 64], f32),
    pub(crate) texture_score: fn(&[f32; 64]) -> f32,
    pub(crate) laplacian_stats: fn(&[f32; 64]) -> [f32; 2],
}

pub(crate) fn selected_noise_kernels() -> NoiseKernels {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        NoiseKernels {
            shrink: |coeffs, t2| unsafe { crate::neon::noise_shrink_neon(coeffs, t2) },
            texture_score: |patch| unsafe { crate::neon::noise_texture_score_neon(patch) },
            laplacian_stats: |patch| unsafe { crate::neon::noise_laplacian_stats_neon(patch) },
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return NoiseKernels {
            shrink: |coeffs, t2| unsafe { crate::avx::noise_shrink_avx2(coeffs, t2) },
            texture_score: |patch| unsafe { crate::avx::noise_texture_score_avx2(patch) },
            laplacian_stats: |patch| unsafe { crate::avx::noise_laplacian_stats_avx2(patch) },
        };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        return NoiseKernels {
            shrink: |coeffs, t2| unsafe { crate::sse::noise_shrink_sse41(coeffs, t2) },
            texture_score: |patch| unsafe { crate::sse::noise_texture_score_sse41(patch) },
            laplacian_stats: |patch| unsafe { crate::sse::noise_laplacian_stats_sse41(patch) },
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return NoiseKernels {
        shrink: crate::wasm::noise_shrink_wasm,
        texture_score: crate::wasm::noise_texture_score_wasm,
        laplacian_stats: crate::wasm::noise_laplacian_stats_wasm,
    };
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    NoiseKernels {
        shrink: noise_shrink_scalar,
        texture_score,
        laplacian_stats: noise_laplacian_stats_scalar,
    }
}

/// Keep the six coefficients with u + v < 3, including DC.
#[allow(dead_code)]
pub(crate) static NOISE_AC_MASK: [u32; 64] = {
    let mut mask = [u32::MAX; 64];
    mask[0] = 0;
    mask[1] = 0;
    mask[2] = 0;
    mask[8] = 0;
    mask[9] = 0;
    mask[16] = 0;
    mask
};

#[allow(dead_code)]
pub(crate) fn noise_shrink_scalar(coeffs: &mut [f32; 64], t2: f32) {
    if t2 <= 0.0 || t2.is_nan() {
        return;
    }
    for (coeff, &mask) in coeffs.iter_mut().zip(&NOISE_AC_MASK) {
        if mask != 0 {
            let c2 = *coeff * *coeff;
            *coeff *= ((c2 - t2) / c2).max(0.0);
        }
    }
}

/// Number of LUT points; point `i` sits at intensity `i / 6`, so the last
/// point covers intensities above 1.
pub(crate) const NUM_POINTS: usize = 8;
const PRECISION: f32 = 1024.0;
/// Largest value that still rounds to the 10-bit maximum.
const LUT_MAX: f32 = 1023.4999 / PRECISION;

/// Spec `kNoise` frame flag.
pub(crate) const FLAG_NOISE: u64 = 1;

/// Grain synthesis the decoder applies on top of a lossy VarDCT frame
/// (see [`EncodeConfig::grain`](crate::EncodeConfig::grain)).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Grain {
    /// No synthesized grain (the default).
    #[default]
    Off,
    /// Photon shot noise of a 35 mm sensor downsampled to the image size at
    /// this ISO sensitivity, libjxl's `--photon_noise_iso` model. Tiny below
    /// ISO 800 on small images; film-like from ISO 3200 up.
    PhotonIso(f32),
    /// Film-grain synthesis in AV1's manner: the fine noise the source
    /// carries at each intensity is measured on its flat areas, removed from
    /// the coded image, and re-synthesized by the decoder at this multiple of
    /// the measured level. `1.0` targets the source's fine-noise level while
    /// spending fewer bits on noise; larger values stylize. The decoder's
    /// fixed high-pass grain does not preserve the source's noise spectrum.
    /// Sources with no flat areas to measure on get none.
    Measured(f32),
    /// `Measured(1.0)` on sources that carry sensor or film grain, nothing
    /// on the rest: the grain level is measured as for `Measured`, and the
    /// frame gets synthesized grain only when the flat areas it was measured
    /// on cover enough of the image and the level is above what the
    /// quantizer would keep for free. Clean renders, screenshots and
    /// denoised photos are left untouched.
    Auto,
    /// Explicit strength LUT, one value in `0..1` per intensity point
    /// `i / 6` (the last point covers intensities above 1).
    Lut([f32; NUM_POINTS]),
}

impl Grain {
    pub(crate) fn enabled(self) -> bool {
        !matches!(self, Grain::Off)
    }
}

/// The serialized noise parameters of one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NoiseParams {
    pub(crate) lut: [f32; NUM_POINTS],
}

impl NoiseParams {
    /// Bits the LfGlobal field costs.
    pub(crate) const BITS: usize = NUM_POINTS * 10;

    /// False when every point rounds to zero: the frame then drops the flag.
    pub(crate) fn has_any(&self) -> bool {
        self.lut.iter().any(|v| v.abs() > 1e-3)
    }

    pub(crate) fn write(&self, w: &mut BitWriter) {
        for &v in &self.lut {
            let q = (v.clamp(0.0, LUT_MAX) * PRECISION).round() as u64;
            w.write(10, q.min(1023));
        }
    }

    /// Resolve the configured grain for the frame. Measured grain also
    /// removes the measured noise from the coded image, so the decoder's
    /// grain replaces it instead of stacking on whatever survives
    /// quantization.
    pub(crate) fn resolve(
        grain: Grain,
        ctx: &crate::encoding_context::EncodingContext,
        scratch: &mut crate::coder_scratch::CoderScratch,
        distance: f32,
        opsin: &mut Image3F,
    ) -> Option<Self> {
        let params = match grain {
            Grain::Off => return None,
            Grain::PhotonIso(iso) if iso > 0.0 => photon_noise(opsin.xsize(), opsin.ysize(), iso),
            Grain::PhotonIso(_) => return None,
            Grain::Measured(mul) if mul > 0.0 => {
                let (measured, _) = measured_grain_with_kernels(opsin, ctx.noise_kernels);
                Self::denoise_and_scale(ctx, scratch, distance, opsin, measured, mul, 1.0)
            }
            Grain::Measured(_) => return None,
            // Decided by `encode_frame_core` from the probe reconstruction.
            Grain::Auto => return None,
            Grain::Lut(lut) => NoiseParams { lut },
        };
        params.has_any().then_some(params)
    }

    fn denoise_and_scale(
        ctx: &crate::encoding_context::EncodingContext,
        scratch: &mut crate::coder_scratch::CoderScratch,
        distance: f32,
        opsin: &mut Image3F,
        measured: NoiseParams,
        mul: f32,
        weight: f32,
    ) -> Self {
        // From d=1.5 on the quantizer removes the grain itself;
        // denoising there costs quality for almost no bytes.
        let denoise_scale = ((3.0 - distance) / 1.5).clamp(0.0, 1.0) * weight;
        if measured.has_any() && denoise_scale > 0.0 {
            denoise(ctx, scratch, opsin, &measured, denoise_scale);
        }
        let mut lut = measured.lut;
        for v in &mut lut {
            *v = (*v * mul).clamp(0.0, LUT_MAX);
        }
        NoiseParams { lut }
    }

    /// Mean strength over the image's own intensity distribution: what the
    /// decoder would add on average, in LUT units.
    fn effective_strength(&self, block_intensities: &[f32]) -> f32 {
        if block_intensities.is_empty() {
            return 0.0;
        }
        block_intensities
            .iter()
            .map(|&i| self.strength_at(i))
            .sum::<f32>()
            / block_intensities.len() as f32
    }

    /// Scale the LUT to the gap the coded image leaves: with `survival` of
    /// the source grain level still in the reconstruction at each LUT
    /// point, the decoder adds the quadrature remainder there so the total
    /// comes out at the source level at every intensity.
    pub(crate) fn fill_gap(&mut self, survival: &Survival, weight: f32) {
        for (v, &s) in self.lut.iter_mut().zip(&survival.per_point) {
            let k = (1.0 - s.clamp(0.0, 1.0).powi(2)).max(0.0).sqrt();
            // On the ramp the look moves with the bytes: a partly denoised
            // frame gets a partly restored grain level, not the full one.
            *v = (*v * k * weight).clamp(0.0, LUT_MAX);
        }
    }

    /// Strength at an intensity, as the decoder interpolates it.
    fn strength_at(&self, intensity: f32) -> f32 {
        let scale = (NUM_POINTS - 2) as f32;
        let pos = (intensity * scale).max(0.0);
        if pos >= scale + 1.0 {
            return self.lut[NUM_POINTS - 1];
        }
        let i = pos.floor() as usize;
        let f = pos - i as f32;
        self.lut[i] + (self.lut[i + 1] - self.lut[i]) * f
    }
}

/// Standard deviation the decoder adds to Y per unit of LUT strength
/// (measured on flat grays through libjxl and jxl-oxide).
const Y_SIGMA_PER_UNIT: f32 = 0.494;

/// Remove the measured grain from the image to be coded, the way AV1's
/// film-grain encoders do: a sliding 8x8 DCT Wiener filter (stride 4, the
/// four overlapping estimates averaged) shrinks the higher AC coefficients
/// against the noise variance expected at the block's intensity, so the
/// grain the decoder re-synthesizes is not also spent bits on, while
/// structure above the noise floor and the lowest bands pass. `scale` fades
/// the filter out toward the distances where quantization removes the grain
/// anyway. Correlated (coarse) grain is handled as well as white noise,
/// which a small bilateral window cannot do.
fn denoise(
    ctx: &crate::encoding_context::EncodingContext,
    scratch: &mut crate::coder_scratch::CoderScratch,
    opsin: &mut Image3F,
    measured: &NoiseParams,
    scale: f32,
) {
    const N: usize = 8;
    const STRIDE: usize = 4;
    const THRESH_K: [f32; 3] = [1.5, 3.0, 3.0];
    let w = opsin.xsize();
    let h = opsin.ysize();
    if w < N || h < N || scale <= 0.0 {
        return;
    }
    let origins =
        |len: usize| -> Vec<usize> { (0..len - N).step_by(STRIDE).chain([len - N]).collect() };
    let xs = origins(w);
    let ys = origins(h);
    // The windows form a Cartesian product: coverage is separable, so two
    // short vectors replace the full image of per-pixel overlap counts.
    let coverage = |len: usize, origins: &[usize]| {
        let mut counts = vec![0.0f32; len];
        for &origin in origins {
            for count in &mut counts[origin..origin + N] {
                *count += 1.0;
            }
        }
        counts
    };
    let count_x = coverage(w, &xs);
    let count_y = coverage(h, &ys);
    let src = &*opsin;
    let mut acc = Image3F::new(w, h);
    crate::encode_image::for_each_linear_band(&mut acc, ctx, scratch, |start, mut band| {
        let first_y = start / w;
        let last_y = first_y + band[0].len() / w;
        // Read the halo windows but accumulate only rows owned by this band.
        // Every pixel sees the same raster-order sum regardless of workers.
        let first_window = ys.partition_point(|&y| y + N <= first_y);
        let last_window = ys.partition_point(|&y| y < last_y);
        let mut coeffs = [[0.0f32; 64]; 3];
        let mut block = [0.0f32; 64];
        for &y0 in &ys[first_window..last_window] {
            let row_start = first_y.saturating_sub(y0);
            let row_end = N.min(last_y - y0);
            for &x0 in &xs {
                let offset = y0 * w + x0;
                for (c, coeff) in coeffs.iter_mut().enumerate() {
                    (ctx.dct8x8)(
                        crate::dct::DctInput::new(&src.plane_data(c)[offset..], w),
                        coeff,
                    );
                }
                // DCT DC is the block mean. The encoder's coefficients are
                // the orthonormal DCT divided by 8, so noise variance is /64.
                let mean = 0.5 * (coeffs[0][0] + coeffs[1][0]);
                let sigma_y = measured.strength_at(mean) * Y_SIGMA_PER_UNIT * scale;
                for (c, coeff) in coeffs.iter_mut().enumerate() {
                    let thresh = THRESH_K[c] * sigma_y;
                    (ctx.noise_kernels.shrink)(coeff, thresh * thresh * (1.0 / 64.0));
                    (ctx.idct.idct8x8)(crate::dct::DctInput::from_flat(coeff), &mut block);
                    let out_start = (y0 + row_start - first_y) * w + x0;
                    for (out, pixels) in band[c][out_start..]
                        .chunks_mut(w)
                        .zip(block.as_chunks::<N>().0[row_start..row_end].iter())
                    {
                        for (out, &pixel) in out[..N].iter_mut().zip(pixels) {
                            *out += pixel;
                        }
                    }
                }
            }
        }
        for plane in &mut band {
            for (row, &cy) in plane.chunks_exact_mut(w).zip(&count_y[first_y..last_y]) {
                for (pixel, &cx) in row.iter_mut().zip(&count_x) {
                    *pixel /= cx * cy;
                }
            }
        }
    });
    // Source stays immutable until all bands finish; avoid cloning it and
    // avoid copying the accumulated image back through a second full pass.
    *opsin = acc;
}

/// Photon shot noise of a full-frame sensor at `iso`, as libjxl models it:
/// read noise, Poisson shot noise and photo-response non-uniformity summed
/// in quadrature per pixel, converted to an XYB-domain standard deviation
/// through the opsin cube root and normalized to the decoder's noise scale.
pub(crate) fn photon_noise(xsize: usize, ysize: usize, iso: f32) -> NoiseParams {
    // Daylight spectrum: photons per lx·s per µm².
    const PHOTONS_PER_LX_S_PER_UM2: f32 = 11260.0;
    // Effective quantum efficiency of a 2010s CFA sensor.
    const QUANTUM_EFFICIENCY: f32 = 0.20;
    const PHOTO_RESPONSE_NON_UNIFORMITY: f32 = 0.005;
    // Electrons rms.
    const READ_NOISE: f32 = 3.0;
    // 35 mm sensor in µm².
    const SENSOR_AREA_UM2: f32 = 36000.0 * 24000.0;

    let bias = crate::xyb::OPSIN_BIAS;
    let bias_cbrt = -crate::xyb::NEG_BIAS_CBRT;
    // Focal-plane exposure for 18% gray in lx·s (ISO = 10 lx·s / H).
    let h_18 = 10.0 / iso;
    let pixel_area_um2 = SENSOR_AREA_UM2 / (xsize.max(1) * ysize.max(1)) as f32;
    let electrons_18 = QUANTUM_EFFICIENCY * PHOTONS_PER_LX_S_PER_UM2 * h_18 * pixel_area_um2;

    let mut lut = [0f32; NUM_POINTS];
    for (i, out) in lut.iter_mut().enumerate() {
        // Point `i` is intensity i/6, i.e. XYB (0, y, y) with y = 2 * i/6.
        let y = 2.0 * i as f32 / (NUM_POINTS - 2) as f32;
        let linear = ((y - bias_cbrt).powi(3) + bias).max(0.0);
        let electrons = electrons_18 * (linear / 0.18);
        let noise = (READ_NOISE * READ_NOISE
            + electrons
            + (PHOTO_RESPONSE_NON_UNIFORMITY * electrons).powi(2))
        .sqrt();
        let linear_noise = noise * (0.18 / electrons_18);
        let opsin_derivative = (1.0 / 3.0) / (linear - bias).cbrt().powi(2);
        let opsin_noise = linear_noise * opsin_derivative;
        *out = (opsin_noise / DECODER_NOISE_SCALE).clamp(0.0, LUT_MAX);
    }
    NoiseParams { lut }
}

/// Standard deviation the decoder adds to Y per unit of LUT strength:
/// `norm_const` 0.22 × √2 for the red and green draws × 1.13, the standard
/// deviation of one high-passed noise plane.
const DECODER_NOISE_SCALE: f32 = 0.22 * std::f32::consts::SQRT_2 * 1.13;

/// AV1-style grain estimation. Flat 8x8 patches (by a robust rank-order
/// SAD texture score) give the source's fine noise level per intensity; the
/// LUT is fit to those samples.
#[cfg(test)]
fn measured_grain(opsin: &Image3F) -> NoiseParams {
    measured_grain_with_kernels(opsin, selected_noise_kernels()).0
}

/// What the estimator saw, for the automatic gate and the survival measure.
#[derive(Clone)]
pub(crate) struct GrainStats {
    /// Mean intensity of every 8x8 block, flat or not.
    block_intensities: Vec<f32>,
    /// Flat patches the LUT was fitted on, as a fraction of all blocks.
    flat_fraction: f32,
    /// The flat patches: origin, mean intensity and measured source level.
    flat_blocks: Vec<(usize, usize, f32, f32)>,
}

/// How much of the measured grain a reconstruction still carries, overall
/// and per decoder LUT point (1 = the source level, 0 = none).
pub(crate) struct Survival {
    pub(crate) overall: f32,
    pub(crate) per_point: [f32; NUM_POINTS],
}

/// How `encode_frame_core` treats grain for one encode of a frame.
pub(crate) enum GrainPass {
    /// Follow `EncodingContext::grain`.
    Inherit,
    /// No grain, whatever the configuration (rate probes, reference frames).
    Off,
    /// The first pass of `Grain::Auto`: plain coding whose reconstruction
    /// tells how much of the measured grain survives quantization.
    Probe(GrainStats),
    /// The second pass of `Grain::Auto`: denoise and set the flag, with the
    /// LUT scaled down by whatever grain this pass's reconstruction still
    /// carries so the decoded total matches the source.
    Apply(NoiseParams, GrainStats, f32),
}

/// Weight of the grain pass from the plain coding's grain survival: 1 when
/// the quantizer removes the grain, 0 when it keeps it (replacing coded
/// grain by synthesized grain only trades real detail for a look-alike),
/// and a ramp in between so neighboring distances and similar images do
/// not flip between two different encodes. The weight scales the denoiser;
/// the LUT is then set from whatever the second pass still carries.
pub(crate) fn grain_weight(survival: f32) -> f32 {
    const HIGH: f32 = 0.75;
    const LOW: f32 = 0.55;
    ((HIGH - survival) / (HIGH - LOW)).clamp(0.0, 1.0)
}

/// `Grain::Auto` first stage, on the source: the measured LUT and the
/// estimator's statistics when the source carries enough grain on enough
/// flat area for the decoder's grain to be worth considering.
pub(crate) fn auto_candidate(
    opsin: &Image3F,
    ctx: &crate::encoding_context::EncodingContext,
) -> Option<(NoiseParams, GrainStats)> {
    let (measured, stats) = measured_grain_with_kernels(opsin, ctx.noise_kernels);
    stats.accept(&measured).then_some((measured, stats))
}

/// `Grain::Auto` second stage: remove the measured grain from the image to
/// be coded and return the LUT the decoder re-synthesizes it with.
pub(crate) fn apply_measured(
    ctx: &crate::encoding_context::EncodingContext,
    scratch: &mut crate::coder_scratch::CoderScratch,
    distance: f32,
    opsin: &mut Image3F,
    measured: NoiseParams,
    weight: f32,
) -> Option<NoiseParams> {
    let params =
        NoiseParams::denoise_and_scale(ctx, scratch, distance, opsin, measured, 1.0, weight);
    params.has_any().then_some(params)
}

impl GrainStats {
    /// Smallest mean LUT strength worth synthesizing: 0.004 is about 0.12
    /// 8-bit levels of Y standard deviation at mid-gray, below which the
    /// grain is invisible and only the LUT's 80 bits and the SS2 loss remain.
    const MIN_STRENGTH: f32 = 0.006;
    /// The LUT must have been fitted on a real population of flat patches.
    const MIN_FLAT_FRACTION: f32 = 0.15;

    pub(crate) fn effective_strength(&self, params: &NoiseParams) -> f32 {
        params.effective_strength(&self.block_intensities)
    }

    /// Flat patches the LUT fit and the survival measure need at least.
    const MIN_FLAT_BLOCKS: usize = 64;

    fn accept(&self, params: &NoiseParams) -> bool {
        self.flat_blocks.len() >= Self::MIN_FLAT_BLOCKS
            && self.flat_fraction >= Self::MIN_FLAT_FRACTION
            && self.effective_strength(params) >= Self::MIN_STRENGTH
    }

    /// Fraction of the measured grain level a reconstruction keeps on the
    /// flat patches the grain was measured on (1 = all of it), overall and
    /// per LUT point with the fit's linear weights. A point no flat patch
    /// reaches reports the overall value.
    pub(crate) fn survival(&self, recon: &Image3F, kernels: NoiseKernels) -> Survival {
        let scale = (NUM_POINTS - 2) as f32;
        let mut src = [0.0f32; NUM_POINTS];
        let mut dec = [0.0f32; NUM_POINTS];
        for &(x0, y0, intensity, level) in &self.flat_blocks {
            if x0 + BLOCK > recon.xsize() || y0 + BLOCK > recon.ysize() {
                continue;
            }
            let patch = y_patch(recon, x0, y0);
            let got = (kernels.laplacian_stats)(&patch)[1];
            let pos = (intensity.max(0.0) * scale).min(scale + 1.0 - 1e-4);
            let i = (pos.floor() as usize).min(NUM_POINTS - 2);
            let f = pos - i as f32;
            for (k, w) in [(i, 1.0 - f), (i + 1, f)] {
                src[k] += w * level;
                dec[k] += w * got;
            }
        }
        let total_src: f32 = src.iter().sum();
        let total_dec: f32 = dec.iter().sum();
        let overall = if total_src <= 0.0 {
            0.0
        } else {
            total_dec / total_src
        };
        // Require a few patches' worth of level before trusting a point.
        let floor = 4.0 * 1e-3;
        let per_point = std::array::from_fn(|k| {
            if src[k] > floor {
                dec[k] / src[k]
            } else {
                overall
            }
        });
        Survival { overall, per_point }
    }
}

fn measured_grain_with_kernels(
    opsin: &Image3F,
    kernels: NoiseKernels,
) -> (NoiseParams, GrainStats) {
    let (samples, block_intensities) = noise_samples(opsin, kernels);
    let blocks = block_intensities.len().max(1) as f32;
    let stats = GrainStats {
        flat_fraction: samples.len() as f32 / blocks,
        flat_blocks: samples
            .iter()
            .map(|s| (s.x0, s.y0, s.intensity, s.level))
            .collect(),
        block_intensities,
    };
    let mut lut = fit_lut(&samples);
    for v in &mut lut {
        *v = v.clamp(0.0, LUT_MAX);
    }
    (NoiseParams { lut }, stats)
}

/// One flat patch: mean intensity `(X + Y) / 2` and the mean absolute
/// response of a 3x3 Laplacian, which for white noise of standard deviation
/// σ is about 2.2 σ.
struct NoiseSample {
    intensity: f32,
    level: f32,
    x0: usize,
    y0: usize,
}

const BLOCK: usize = 8;

// Several measurement bins per decoder LUT segment keep genuinely different
// noise levels from competing for the same flat-patch cutoff or noise floor.
const INTENSITY_BINS: usize = (NUM_POINTS - 1) * 4;

fn intensity_bin(intensity: f32) -> usize {
    ((intensity.max(0.0) * ((NUM_POINTS - 2) * 4) as f32) as usize).min(INTENSITY_BINS - 1)
}

/// Flat-patch detection and per-patch noise level, after libjxl's
/// `GetNoiseParameter`: the texture score of a patch is the mean of the
/// smallest half of its 3x4 sub-patch SADs against the center sub-patch;
/// the histogram mode of the scores is taken as the flat-patch level within
/// each intensity bin. A global cutoff would reject noisy shadows whenever
/// quieter highlights dominate the image (or vice versa).
fn noise_samples(opsin: &Image3F, kernels: NoiseKernels) -> (Vec<NoiseSample>, Vec<f32>) {
    let xsize = opsin.xsize();
    let ysize = opsin.ysize();
    if xsize < BLOCK || ysize < BLOCK {
        return (Vec::new(), Vec::new());
    }
    let bx = xsize / BLOCK;
    let by = ysize / BLOCK;
    let mut scores = Vec::with_capacity(bx * by);
    let mut intensities = Vec::with_capacity(bx * by);
    let mut histograms = [[0u32; 256]; INTENSITY_BINS];
    for gy in 0..by {
        for gx in 0..bx {
            let patch = intensity_patch(opsin, gx * BLOCK, gy * BLOCK);
            let score = (kernels.texture_score)(&patch);
            let intensity = patch.iter().sum::<f32>() / (BLOCK * BLOCK) as f32;
            let bin = intensity_bin(intensity);
            scores.push((score, bin));
            intensities.push(intensity);
            histograms[bin][((score * 256.0) as usize).min(255)] += 1;
        }
    }
    let thresholds = histograms.map(|hist| {
        let mode = hist
            .iter()
            .enumerate()
            .max_by_key(|&(i, &n)| (n, std::cmp::Reverse(i)))
            .map_or(0, |(i, _)| i);
        // A strong regular pattern dominates this intensity range; nothing
        // flat to measure on. Use the upper edge of the modal bin so faint
        // grain in bin zero is retained too.
        (mode as f32 / 256.0 <= 0.15).then_some((mode + 1) as f32 / 256.0)
    });

    let mut samples = Vec::new();
    for gy in 0..by {
        for gx in 0..bx {
            let (score, bin) = scores[gy * bx + gx];
            let Some(threshold) = thresholds[bin] else {
                continue;
            };
            if score >= threshold {
                continue;
            }
            let x0 = gx * BLOCK;
            let y0 = gy * BLOCK;
            let patch = intensity_patch(opsin, x0, y0);
            let intensity = (kernels.laplacian_stats)(&patch)[0];
            let level = (kernels.laplacian_stats)(&y_patch(opsin, x0, y0))[1];
            // Periodic texture can match the center sub-patch almost exactly
            // despite a large high-frequency response. White noise's mean
            // Laplacian is smaller than its 12-pixel SAD; leave ample margin
            // and allow rounding error on constant patches.
            if level > 2.0 * score + 1e-6 {
                continue;
            }
            samples.push(NoiseSample {
                intensity,
                level,
                x0,
                y0,
            });
        }
    }
    (samples, intensities)
}

/// Mean intensity and mean absolute 3x3 Laplacian of an 8x8 patch.
#[allow(dead_code)]
pub(crate) fn noise_laplacian_stats_scalar(patch: &[f32; 64]) -> [f32; 2] {
    const LAPLACIAN: [[f32; 3]; 3] = [
        [-0.25, -1.0, -0.25],
        [-1.0, 5.0, -1.0],
        [-0.25, -1.0, -0.25],
    ];
    let mut mean = 0.0;
    let mut level = 0.0;
    for y in 0..BLOCK {
        for x in 0..BLOCK {
            mean += patch[y * BLOCK + x];
            let mut filtered = 0.0;
            for (dy, row) in LAPLACIAN.iter().enumerate() {
                let sy = reflect(y, dy as isize - 1);
                for (dx, &k) in row.iter().enumerate() {
                    let sx = reflect(x, dx as isize - 1);
                    filtered += k * patch[sy * BLOCK + sx];
                }
            }
            level += filtered.abs();
        }
    }
    [mean / 64.0, level / 64.0]
}

/// Keep the scalar raster-order reductions so SIMD cannot perturb LUT fits.
#[inline]
#[allow(dead_code)]
pub(crate) fn noise_laplacian_stats_from_abs(patch: &[f32; 64], abs: &[f32; 64]) -> [f32; 2] {
    let mut mean = 0.0;
    let mut level = 0.0;
    for (&pixel, &filtered) in patch.iter().zip(abs) {
        mean += pixel;
        level += filtered;
    }
    [mean / 64.0, level / 64.0]
}

/// Mirror a tap that leaves the patch back inside it.
fn reflect(p: usize, d: isize) -> usize {
    let q = p as isize + d;
    if (0..BLOCK as isize).contains(&q) {
        q as usize
    } else {
        (p as isize - d) as usize
    }
}

fn intensity_patch(opsin: &Image3F, x0: usize, y0: usize) -> [f32; 64] {
    let mut patch = [0.0; 64];
    for (y, row) in patch.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        for ((out, &x), &y) in row
            .iter_mut()
            .zip(&opsin.plane_row(0, y0 + y)[x0..x0 + BLOCK])
            .zip(&opsin.plane_row(1, y0 + y)[x0..x0 + BLOCK])
        {
            *out = 0.5 * (x + y);
        }
    }
    patch
}

#[allow(dead_code)]
fn texture_score(patch: &[f32; 64]) -> f32 {
    const SW: usize = 3;
    const SH: usize = 4;
    const OFFSET: usize = 2;
    let mut sads = [0.0; (BLOCK - SW) * (BLOCK - SH)];
    for yb in 0..BLOCK - SH {
        for xb in 0..BLOCK - SW {
            let mut sad = 0.0;
            for cy in 0..SH {
                for cx in 0..SW {
                    let wnd = patch[(yb + cy) * BLOCK + xb + cx];
                    let center = patch[(OFFSET + cy) * BLOCK + OFFSET + cx];
                    sad += (center - wnd).abs();
                }
            }
            sads[yb * (BLOCK - SW) + xb] = sad;
        }
    }
    texture_score_from_sads(sads)
}

pub(crate) fn texture_score_from_sads(mut sads: [f32; 20]) -> f32 {
    sads.sort_unstable_by(f32::total_cmp);
    let keep = sads.len() / 2;
    sads[..keep].iter().sum::<f32>() / keep as f32
}

/// SIMD kernels compute the first four horizontal candidates together.
/// Complete the fifth candidate of each row without reading past the patch.
#[allow(dead_code)]
pub(crate) fn texture_score_from_simd_sads(patch: &[f32; 64], mut sads: [f32; 20]) -> f32 {
    for (yb, row) in sads.as_chunks_mut::<5>().0.iter_mut().enumerate() {
        let mut sad = 0.0;
        for cy in 0..4 {
            for cx in 0..3 {
                sad += (patch[(2 + cy) * 8 + 2 + cx] - patch[(yb + cy) * 8 + 4 + cx]).abs();
            }
        }
        row[4] = sad;
    }
    texture_score_from_sads(sads)
}

/// Mean |3x3 Laplacian| of the decoded Y plane that one unit of LUT strength
/// produces, calibrated end to end on flat grays and colours with known noise
/// (gain 0.98-1.07 at sigma 1.5-6, gray 16-240). Dividing a source patch's Y
/// Laplacian level by it gives the strength that reproduces that level. The
/// level is measured on Y alone: the decoder's grain is luma-dominant, while
/// the intensity `(X + Y) / 2` also counts chroma noise it does not reproduce.
const LAPLACIAN_LEVEL_PER_UNIT: f32 = 2.06;

/// The 8x8 Y patch at `(x0, y0)`, where the grain level is measured.
fn y_patch(opsin: &Image3F, x0: usize, y0: usize) -> [f32; 64] {
    let mut patch = [0.0; 64];
    for (y, row) in patch.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        row.copy_from_slice(&opsin.plane_row(1, y0 + y)[x0..x0 + BLOCK]);
    }
    patch
}

/// Least-squares fit of the 8-point LUT to the samples with a small
/// smoothness prior, by Gauss-Seidel on the normal equations. The result
/// is in decoder LUT units.
fn fit_lut(samples: &[NoiseSample]) -> [f32; NUM_POINTS] {
    if samples.is_empty() {
        return [0.0; NUM_POINTS];
    }
    let scale = (NUM_POINTS - 2) as f32;
    let segment = |s: &NoiseSample| {
        let pos = (s.intensity.max(0.0) * scale).min(scale + 1.0 - 1e-4);
        (pos, (pos.floor() as usize).min(NUM_POINTS - 2))
    };
    // Noise is the floor of the patch levels: texture only adds to it. Per
    // intensity bin keep the lower half of the levels, which drops the flat
    // patches that still carry some structure without discarding noisier
    // shadows that share a decoder LUT segment with quieter midtones.
    let mut floors = vec![Vec::new(); INTENSITY_BINS];
    for s in samples {
        floors[intensity_bin(s.intensity)].push(s.level);
    }
    let cutoffs: Vec<f32> = floors
        .iter_mut()
        .map(|v| {
            if v.is_empty() {
                return f32::INFINITY;
            }
            let mid = v.len() / 2;
            *v.select_nth_unstable_by(mid, f32::total_cmp).1
        })
        .collect();
    // Normal equations of a piecewise-linear fit: A w = b.
    let mut a = [[0f64; NUM_POINTS]; NUM_POINTS];
    let mut b = [0f64; NUM_POINTS];
    for s in samples {
        let (pos, i) = segment(s);
        if s.level > cutoffs[intensity_bin(s.intensity)] {
            continue;
        }
        let f = (pos - i as f32) as f64;
        let target = (s.level / LAPLACIAN_LEVEL_PER_UNIT) as f64;
        let w = [1.0 - f, f];
        for ((row, rhs), &wi) in a[i..i + w.len()]
            .iter_mut()
            .zip(&mut b[i..i + w.len()])
            .zip(&w)
        {
            *rhs += wi * target;
            for (cell, &wj) in row[i..i + w.len()].iter_mut().zip(&w) {
                *cell += wi * wj;
            }
        }
    }
    // Smoothness prior: reg × n × Σ (w[i] − w[i+1])².
    let reg = 0.005 * samples.len() as f64;
    for i in 0..NUM_POINTS - 1 {
        a[i][i] += reg;
        a[i + 1][i + 1] += reg;
        a[i][i + 1] -= reg;
        a[i + 1][i] -= reg;
    }
    // Gauss-Seidel on the (positive definite) system.
    let mut w = [0f64; NUM_POINTS];
    for _ in 0..200 {
        for i in 0..NUM_POINTS {
            let mut r = b[i];
            for j in 0..NUM_POINTS {
                if j != i {
                    r -= a[i][j] * w[j];
                }
            }
            w[i] = if a[i][i] > 1e-12 {
                (r / a[i][i]).max(0.0)
            } else {
                0.0
            };
        }
    }
    std::array::from_fn(|i| w[i] as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn denoise_reference(
        ctx: &crate::encoding_context::EncodingContext,
        scratch: &mut crate::coder_scratch::CoderScratch,
        opsin: &mut Image3F,
        measured: &NoiseParams,
        scale: f32,
    ) {
        const N: usize = 8;
        const STRIDE: usize = 4;
        // Shrinkage threshold relative to the noise standard deviation per
        // channel; chroma noise in XYB is a fraction of the luma noise.
        const THRESH_K: [f32; 3] = [1.5, 3.0, 3.0];
        let w = opsin.xsize();
        let h = opsin.ysize();
        if w < N || h < N {
            return;
        }
        let src = opsin.clone();
        // Orthonormal DCT-II basis.
        let basis: [[f32; N]; N] = std::array::from_fn(|k| {
            std::array::from_fn(|n| {
                let s = if k == 0 {
                    (1.0 / N as f32).sqrt()
                } else {
                    (2.0 / N as f32).sqrt()
                };
                s * (std::f32::consts::PI * (2 * n + 1) as f32 * k as f32 / (2 * N) as f32).cos()
            })
        });
        // Window origins cover the image, the last column/row clamped inside.
        let origins = |len: usize| -> Vec<usize> {
            let mut v: Vec<usize> = (0..len.saturating_sub(N)).step_by(STRIDE).collect();
            v.push(len - N);
            v
        };
        let xs = origins(w);
        let ys = origins(h);
        let mut acc = Image3F::new(w, h);
        let mut count = vec![0f32; w * h];
        let mut block = [[0f32; N]; N];
        let mut tmp = [[0f32; N]; N];
        for &y0 in &ys {
            for &x0 in &xs {
                // Noise level at the block's mean intensity.
                let mut mean = 0.0;
                for y in 0..N {
                    let rx = &src.plane_row(0, y0 + y)[x0..x0 + N];
                    let ry = &src.plane_row(1, y0 + y)[x0..x0 + N];
                    for x in 0..N {
                        mean += 0.5 * (rx[x] + ry[x]);
                    }
                }
                let sigma_y =
                    measured.strength_at(mean / (N * N) as f32) * Y_SIGMA_PER_UNIT * scale;
                for c in 0..3 {
                    let thresh = THRESH_K[c] * sigma_y;
                    let t2 = thresh * thresh;
                    for y in 0..N {
                        block[y].copy_from_slice(&src.plane_row(c, y0 + y)[x0..x0 + N]);
                    }
                    // Forward: tmp = B · block · Bᵀ.
                    for v in 0..N {
                        for u in 0..N {
                            let mut s = 0.0;
                            for y in 0..N {
                                let mut r = 0.0;
                                for x in 0..N {
                                    r += basis[u][x] * block[y][x];
                                }
                                s += basis[v][y] * r;
                            }
                            tmp[v][u] = s;
                        }
                    }
                    // Wiener-style shrinkage of the AC coefficients.
                    if t2 > 0.0 {
                        for v in 0..N {
                            for u in 0..N {
                                // DC and the three lowest AC bands pass: grain
                                // lives above them, structure below.
                                if u + v < 3 {
                                    continue;
                                }
                                let c2 = tmp[v][u] * tmp[v][u];
                                tmp[v][u] *= ((c2 - t2) / c2).max(0.0);
                            }
                        }
                    }
                    // Inverse: block = Bᵀ · tmp · B.
                    for y in 0..N {
                        for x in 0..N {
                            let mut s = 0.0;
                            for v in 0..N {
                                let mut r = 0.0;
                                for u in 0..N {
                                    r += basis[u][x] * tmp[v][u];
                                }
                                s += basis[v][y] * r;
                            }
                            block[y][x] = s;
                        }
                    }
                    for y in 0..N {
                        let row = &mut acc.plane_row_mut(c, y0 + y)[x0..x0 + N];
                        for x in 0..N {
                            row[x] += block[y][x];
                        }
                    }
                }
                for y in 0..N {
                    for n in &mut count[(y0 + y) * w + x0..(y0 + y) * w + x0 + N] {
                        *n += 1.0;
                    }
                }
            }
        }
        crate::encode_image::for_each_linear_band(opsin, ctx, scratch, |start, band| {
            for (c, plane) in band.into_iter().enumerate() {
                for (k, v) in plane.iter_mut().enumerate() {
                    let p = start + k;
                    *v = acc.plane_row(c, p / w)[p % w] / count[p];
                }
            }
        });
    }

    fn noisy_image(w: usize, h: usize) -> Image3F {
        let mut image = Image3F::new(w, h);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for c in 0..3 {
            for y in 0..h {
                for (x, pixel) in image.plane_row_mut(c, y).iter_mut().enumerate() {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let noise = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.16;
                    *pixel = if c == 0 { 0.02 } else { 0.5 }
                        + x as f32 * 0.0001
                        + y as f32 * 0.0002
                        + noise;
                }
            }
        }
        image
    }

    fn kernels() -> Vec<(&'static str, NoiseKernels)> {
        #[allow(unused_mut)]
        let mut kernels = vec![
            (
                "scalar",
                NoiseKernels {
                    shrink: noise_shrink_scalar,
                    texture_score,
                    laplacian_stats: noise_laplacian_stats_scalar,
                },
            ),
            ("selected", selected_noise_kernels()),
        ];
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push((
                "avx2",
                NoiseKernels {
                    shrink: |coeffs, t2| unsafe { crate::avx::noise_shrink_avx2(coeffs, t2) },
                    texture_score: |patch| unsafe { crate::avx::noise_texture_score_avx2(patch) },
                    laplacian_stats: |patch| unsafe {
                        crate::avx::noise_laplacian_stats_avx2(patch)
                    },
                },
            ));
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            kernels.push((
                "sse4.1",
                NoiseKernels {
                    shrink: |coeffs, t2| unsafe { crate::sse::noise_shrink_sse41(coeffs, t2) },
                    texture_score: |patch| unsafe { crate::sse::noise_texture_score_sse41(patch) },
                    laplacian_stats: |patch| unsafe {
                        crate::sse::noise_laplacian_stats_sse41(patch)
                    },
                },
            ));
        }
        kernels
    }

    #[test]
    fn laplacian_kernels_match_scalar_at_reflected_edges() {
        let mut patches = vec![
            [0.0; 64],
            [0.37; 64],
            std::array::from_fn(|i| i as f32 * 0.013 - 0.4),
            std::array::from_fn(|i| if (i / 8 + i % 8) % 2 == 0 { -0.7 } else { 0.3 }),
        ];
        // An impulse at every position exercises all reflected edges/corners
        // and both sides of each SIMD lane boundary.
        for i in 0..64 {
            let mut patch = [0.0; 64];
            patch[i] = 1.0;
            patches.push(patch);
        }
        for case in 0..128 {
            patches.push(std::array::from_fn(|i| {
                ((i * 37 + case * 13) % 101) as f32 * 0.013 - 0.57
            }));
        }
        for (name, kernels) in kernels() {
            for (case, patch) in patches.iter().enumerate() {
                let expected = noise_laplacian_stats_scalar(patch);
                let actual = (kernels.laplacian_stats)(patch);
                assert_eq!(
                    actual.map(f32::to_bits),
                    expected.map(f32::to_bits),
                    "{name} patch {case}"
                );
            }
            let mut patch = [0.0; 64];
            patch[0] = 1.0;
            assert_eq!((kernels.laplacian_stats)(&patch), [1.0 / 64.0, 7.25 / 64.0]);
            patch[0] = 0.0;
            patch[3 * 8 + 3] = 1.0;
            assert_eq!((kernels.laplacian_stats)(&patch), [1.0 / 64.0, 10.0 / 64.0]);
        }
    }

    #[test]
    fn laplacian_dispatch_preserves_fitted_grain() {
        let mut image = noisy_image(65, 35);
        for c in 0..3 {
            for pixel in image.plane_mut(c).as_mut_slice() {
                *pixel *= 0.125;
            }
        }
        for (name, kernels) in kernels() {
            let scalar = NoiseKernels {
                laplacian_stats: noise_laplacian_stats_scalar,
                ..kernels
            };
            let expected = measured_grain_with_kernels(&image, scalar).0;
            assert!(expected.has_any());
            assert_eq!(
                measured_grain_with_kernels(&image, kernels).0,
                expected,
                "{name}"
            );
        }
    }

    #[test]
    #[ignore = "Laplacian benchmark; run in release mode with --ignored --nocapture"]
    fn benchmark_noise_laplacian() {
        use std::hint::black_box;
        use std::time::Instant;
        let patches: Vec<[f32; 64]> = (0..256)
            .map(|case| std::array::from_fn(|i| ((i * 37 + case * 13) % 101) as f32 * 0.013 - 0.57))
            .collect();
        for (name, kernels) in kernels() {
            let mut times = Vec::new();
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..100 {
                    for patch in &patches {
                        black_box(black_box(kernels.laplacian_stats)(black_box(patch)));
                    }
                }
                times.push(start.elapsed().as_secs_f64() * 1e9 / (100 * patches.len()) as f64);
            }
            times.sort_by(f64::total_cmp);
            eprintln!("Laplacian {name}: {:.2} ns/patch", times[3]);
        }
    }

    #[test]
    fn kernels_match_scalar_and_preserve_low_frequencies() {
        for (name, kernels) in kernels() {
            for case in 0..32 {
                let patch =
                    std::array::from_fn(|i| ((i * 37 + case * 13) % 101) as f32 * 0.013 - 0.57);
                assert_eq!(
                    (kernels.texture_score)(&patch),
                    texture_score(&patch),
                    "{name} case {case}"
                );
                for t2 in [0.0, 1e-8, 0.0001, 0.01, 1.0, f32::NAN] {
                    let mut actual = patch;
                    // Zero coefficients exercise 0/0 in the Wiener gain.
                    actual[3] = 0.0;
                    actual[63] = -0.0;
                    let mut expected = actual;
                    noise_shrink_scalar(&mut expected, t2);
                    (kernels.shrink)(&mut actual, t2);
                    for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{name} case {case} t2 {t2} coefficient {i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn denoise_matches_orthonormal_reference_at_edges_and_overlaps() {
        let ctx = crate::encoding_context::EncodingContext::default();
        let mut scratch = crate::coder_scratch::CoderScratch::default();
        let measured = NoiseParams {
            lut: [0.015, 0.03, 0.07, 0.04, 0.02, 0.09, 0.08, 0.05],
        };
        for (w, h) in [
            (1, 9),
            (9, 7),
            (8, 8),
            (9, 11),
            (12, 12),
            (17, 21),
            (32, 24),
        ] {
            let source = noisy_image(w, h);
            for scale in [0.0, 0.3, 1.0] {
                let mut actual = source.clone();
                let mut expected = source.clone();
                denoise_reference(&ctx, &mut scratch, &mut expected, &measured, scale);
                denoise(&ctx, &mut scratch, &mut actual, &measured, scale);
                for c in 0..3 {
                    for (i, (&actual, &expected)) in actual
                        .plane_data(c)
                        .iter()
                        .zip(expected.plane_data(c))
                        .enumerate()
                    {
                        assert!(
                            (actual - expected).abs() < 3e-6,
                            "{w}x{h} scale {scale} plane {c} pixel {i}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn denoise_is_independent_of_worker_count() {
        let mut ctx = crate::encoding_context::EncodingContext::default();
        let mut scratch = crate::coder_scratch::CoderScratch::default();
        let measured = NoiseParams {
            lut: [0.04; NUM_POINTS],
        };
        for (w, h) in [(8, 9), (13, 17), (67, 49)] {
            let source = noisy_image(w, h);
            let mut expected = source.clone();
            ctx.thread_pool = crate::thread_pool::ThreadPool::new(1);
            denoise(&ctx, &mut scratch, &mut expected, &measured, 1.0);
            for threads in [2, 4] {
                ctx.thread_pool = crate::thread_pool::ThreadPool::new(threads);
                let mut actual = source.clone();
                denoise(&ctx, &mut scratch, &mut actual, &measured, 1.0);
                for c in 0..3 {
                    assert_eq!(
                        actual.plane_data(c),
                        expected.plane_data(c),
                        "{w}x{h} threads {threads} plane {c}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "Noise pipeline benchmark; run in release mode with --ignored --nocapture"]
    fn benchmark_noise_pipeline() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut ctx = crate::encoding_context::EncodingContext::default();
        let mut scratch = crate::coder_scratch::CoderScratch::default();
        for (w, h) in [(256, 256), (1024, 768)] {
            let source = noisy_image(w, h);
            let params = NoiseParams {
                lut: [0.04; NUM_POINTS],
            };
            let mut estimate_times = Vec::new();
            let mut denoise_times = Vec::new();
            let mut reference_times = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                black_box(measured_grain(black_box(&source)));
                estimate_times.push(start.elapsed().as_secs_f64() * 1000.0);
                let mut image = source.clone();
                let start = Instant::now();
                denoise(&ctx, &mut scratch, black_box(&mut image), &params, 1.0);
                denoise_times.push(start.elapsed().as_secs_f64() * 1000.0);
                black_box(image);
                let mut image = source.clone();
                let start = Instant::now();
                denoise_reference(&ctx, &mut scratch, black_box(&mut image), &params, 1.0);
                reference_times.push(start.elapsed().as_secs_f64() * 1000.0);
                black_box(image);
            }
            estimate_times.sort_by(f64::total_cmp);
            denoise_times.sort_by(f64::total_cmp);
            reference_times.sort_by(f64::total_cmp);
            eprintln!(
                "noise {w}x{h}: estimate {:.3} ms, denoise {:.3} ms, original {:.3} ms ({:.2}x)",
                estimate_times[2],
                denoise_times[2],
                reference_times[2],
                reference_times[2] / denoise_times[2]
            );
            ctx.thread_pool = crate::thread_pool::ThreadPool::new(4);
            let mut parallel_times = Vec::new();
            for _ in 0..5 {
                let mut image = source.clone();
                let start = Instant::now();
                denoise(&ctx, &mut scratch, black_box(&mut image), &params, 1.0);
                parallel_times.push(start.elapsed().as_secs_f64() * 1000.0);
                black_box(image);
            }
            parallel_times.sort_by(f64::total_cmp);
            eprintln!(
                "noise {w}x{h}: denoise 4 workers {:.3} ms",
                parallel_times[2]
            );
            ctx.thread_pool = crate::thread_pool::ThreadPool::new(1);

            // Lower noise reaches the Laplacian measurement and LUT fit,
            // rather than being rejected as texture by the score histogram.
            let mut grainy = source;
            for c in 0..3 {
                for pixel in grainy.plane_mut(c).as_mut_slice() {
                    *pixel *= 0.125;
                }
            }
            assert!(measured_grain(&grainy).has_any());
            let mut grainy_times = Vec::new();
            let mut scalar_laplacian_times = Vec::new();
            let scalar_laplacian = NoiseKernels {
                laplacian_stats: noise_laplacian_stats_scalar,
                ..ctx.noise_kernels
            };
            for _ in 0..5 {
                let start = Instant::now();
                black_box(measured_grain(black_box(&grainy)));
                grainy_times.push(start.elapsed().as_secs_f64() * 1000.0);
                let start = Instant::now();
                black_box(measured_grain_with_kernels(
                    black_box(&grainy),
                    scalar_laplacian,
                ));
                scalar_laplacian_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            grainy_times.sort_by(f64::total_cmp);
            scalar_laplacian_times.sort_by(f64::total_cmp);
            eprintln!(
                "grain {w}x{h}: full estimate {:.3} ms, scalar Laplacian {:.3} ms",
                grainy_times[2], scalar_laplacian_times[2]
            );
        }
    }

    /// The LUT libjxl writes for `--photon_noise_iso=6400` on a 768x512
    /// image; the decoded grain amplitude matched cjxl's within 0.3%.
    #[test]
    fn photon_noise_matches_libjxl() {
        let lut = photon_noise(768, 512, 6400.0).lut;
        let expected = [
            0.0027226498,
            0.0143037215,
            0.00695769,
            0.0064165955,
            0.0070083137,
            0.008079084,
            0.0093738455,
            0.010781403,
        ];
        for (got, want) in lut.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn lut_serializes_to_ten_bit_fields() {
        let params = NoiseParams {
            lut: [0.0, 0.5, 1.0, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        };
        let mut w = BitWriter::new();
        params.write(&mut w);
        assert_eq!(w.bits_written(), NoiseParams::BITS);
        let bytes = w.into_bytes();
        // Fields are ten bits each, LSB first: 0, then 512.
        let first = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
        assert_eq!(first & 0x3ff, 0);
        assert_eq!((first >> 10) & 0x3ff, 512);
        // 1.0 clamps to the ten-bit maximum.
        assert_eq!(
            (first >> 20) & 0xf | ((u32::from(bytes[3]) & 0x3f) << 4),
            1023
        );
    }

    #[test]
    fn flat_image_measures_no_grain() {
        let mut img = Image3F::new(64, 64);
        for c in 0..3 {
            for y in 0..64 {
                img.plane_row_mut(c, y).fill(0.3);
            }
        }
        assert!(!measured_grain(&img).has_any());
    }

    fn noisy_gray_bands(bands: &[(f32, f32, usize)]) -> Image3F {
        let w = bands.iter().map(|&(_, _, width)| width).sum();
        let mut img = Image3F::new(w, 256);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut x0 = 0;
        for &(mean, sigma, width) in bands {
            for y in 0..img.ysize() {
                for x in x0..x0 + width {
                    let mut n = -6.0;
                    for _ in 0..12 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        n += (state >> 40) as f32 / (1u64 << 24) as f32;
                    }
                    img.plane_row_mut(1, y)[x] = mean + sigma * n;
                    img.plane_row_mut(2, y)[x] = mean + sigma * n;
                }
            }
            x0 += width;
        }
        img
    }

    #[test]
    fn faint_noise_is_not_discarded_by_the_score_histogram() {
        let img = noisy_gray_bands(&[(0.5, 0.0006, 256)]);
        let measured = measured_grain(&img);
        assert!(measured.has_any(), "lut {:?}", measured.lut);
    }

    #[test]
    fn repeated_texture_is_not_measured_as_faint_noise() {
        for sigma in [0.0, 0.0006] {
            for pattern in [0, 1, 2] {
                let mut img = noisy_gray_bands(&[(0.5, sigma, 256)]);
                for y in 0..img.ysize() {
                    for x in 0..img.xsize() {
                        let phase = match pattern {
                            0 => x + y,
                            1 => x,
                            _ => y,
                        };
                        let texture = if phase % 2 == 0 { -0.05 } else { 0.05 };
                        img.plane_row_mut(1, y)[x] += texture;
                        img.plane_row_mut(2, y)[x] += texture;
                    }
                }
                let measured = measured_grain(&img);
                assert!(
                    !measured.has_any(),
                    "sigma {sigma}, pattern {pattern}, lut {:?}",
                    measured.lut
                );
            }
        }
    }

    #[test]
    fn measured_noise_preserves_shadow_highlight_strengths() {
        // A large, quiet highlight must not set the flat-patch cutoff for
        // noisier shadows. Exercise both noise profiles and area ratios.
        for widths in [[256, 256], [64, 448], [448, 64]] {
            for (means, sigmas) in [
                ([0.1, 0.7], [0.02, 0.002]),
                ([0.1, 0.7], [0.002, 0.02]),
                // Both intensities share one decoder LUT segment.
                ([0.1, 0.26], [0.02, 0.01]),
                ([0.1, 0.26], [0.01, 0.02]),
            ] {
                let img = noisy_gray_bands(&[
                    (means[0], sigmas[0], widths[0]),
                    (means[1], sigmas[1], widths[1]),
                ]);
                let measured = measured_grain(&img);
                for (mean, sigma) in means.into_iter().zip(sigmas) {
                    let level = (2.0 / std::f32::consts::PI).sqrt() * 29.25f32.sqrt() * sigma * 0.5;
                    let want = level * 2.0 / LAPLACIAN_LEVEL_PER_UNIT;
                    let got = measured.strength_at(mean * 0.5);
                    assert!(
                        got > want * 0.6 && got < want * 1.3,
                        "widths {widths:?}, sigmas {sigmas:?}, mean {mean}: {got} vs {want}, lut {:?}",
                        measured.lut
                    );
                }
            }
        }
    }

    /// White noise of a known standard deviation on a flat field comes back
    /// as a LUT near the strength that reproduces its Laplacian level.
    #[test]
    fn white_noise_measures_its_level() {
        let (w, h) = (256, 256);
        let mut img = Image3F::new(w, h);
        let sigma = 0.01f32;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut gauss = || {
            // Sum of 12 uniforms minus 6: a unit-variance Gaussian.
            let mut s = 0.0f32;
            for _ in 0..12 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                s += (state >> 40) as f32 / (1u64 << 24) as f32;
            }
            s - 6.0
        };
        for y in 0..h {
            for x in 0..w {
                let n = sigma * gauss();
                img.plane_row_mut(0, y)[x] = 0.0;
                img.plane_row_mut(1, y)[x] = 0.5 + n;
                img.plane_row_mut(2, y)[x] = 0.5 + n;
            }
        }
        let lut = measured_grain(&img).lut;
        // Intensity (X+Y)/2 = 0.25 is halfway through the second segment.
        // Mean |Laplacian| of white noise: sqrt(2/pi) * ||kernel|| * sigma/2.
        let level =
            (2.0 / std::f32::consts::PI).sqrt() * (25.0f32 + 4.0 + 0.25).sqrt() * sigma * 0.5;
        let want = level * 2.0 / LAPLACIAN_LEVEL_PER_UNIT;
        let got = lut[1] * 0.5 + lut[2] * 0.5;
        assert!(
            got > want * 0.6 && got < want * 1.2,
            "lut {lut:?}, want about {want}"
        );
    }
}
