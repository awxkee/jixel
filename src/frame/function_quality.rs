/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! Quality screen for the scalar function representations of an extra
//! channel.
//!
//! Below the slowest tier this is a proxy, not the target metric: multi-scale
//! SSIM (`msssim-lite`) of the plane itself and of the source color's lightness
//! composited over black and over white, candidate against the ordinary
//! coding. The race it gates only has to keep a function coding from looking
//! worse than the coding it replaces. The slowest tier asks SSIMULACRA2 and
//! CVVDP directly, on the plane and on its color composites.

use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

/// How much one view's deficit (`msssim_lite::Score::deficit`: mean per
/// pixel, summed over the scales) may grow when the pooled deficit does not.
const VIEW_SLACK: f64 = 1e-4;

/// What every candidate of one race is compared with: the source's views,
/// prepared, and the ordinary coding's distance from them.
pub(super) struct Incumbent {
    luma_scale: f32,
    views: Vec<msssim_lite::PreparedReference>,
    deficits: Vec<f64>,
}

/// Kept between the candidates of one race (same source, color and
/// incumbent).
#[derive(Default)]
pub(super) struct GateCache {
    errors: std::sync::OnceLock<PlaneErrors>,
    proxy: std::sync::OnceLock<Option<Incumbent>>,
    #[cfg(feature = "extraslow")]
    exact: std::sync::OnceLock<Option<ExactReference>>,
}

#[derive(Default)]
struct PlaneErrors {
    squared: u64,
    max: u8,
    endpoints: usize,
}

fn plane_errors(source: &[u8], candidate: &[u8]) -> PlaneErrors {
    let mut result = PlaneErrors::default();
    // Narrow sums vectorize without widening each pixel's square to u64.
    // Each chunk can contribute at most 32768 * 255^2.
    const CHUNK: usize = 32768;
    for (source, candidate) in source.chunks(CHUNK).zip(candidate.chunks(CHUNK)) {
        let (mut squared, mut max, mut endpoints) = (0u32, 0u8, 0u32);
        for (&s, &v) in source.iter().zip(candidate) {
            let error = s.abs_diff(v);
            squared += u32::from(error).pow(2);
            max = max.max(error);
            endpoints += u32::from(((s == 0) | (s == 255)) & (error != 0));
        }
        result.squared += u64::from(squared);
        result.max = result.max.max(max);
        result.endpoints += endpoints as usize;
    }
    result
}

/// Which measure decides. The direct metrics cost several times the proxy
/// and are what it stands in for, so only the slowest tier runs them, and
/// only on SDR sRGB input, which their composites assume.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Gate {
    Proxy,
    #[cfg(feature = "extraslow")]
    Exact,
}

impl Gate {
    fn select(ctx: &EncodingContext) -> Self {
        #[cfg(feature = "extraslow")]
        if ctx.function_fields_exact {
            return Self::Exact;
        }
        let _ = ctx;
        Self::Proxy
    }
}

/// Whether `candidate` may replace `incumbent`; see [`score`].
#[cfg(test)]
pub(super) fn acceptable(
    ctx: &EncodingContext,
    color: &Image3F,
    source: &AlphaPlane,
    incumbent: &AlphaPlane,
    candidate: &AlphaPlane,
    with_residual: bool,
    cache: &GateCache,
) -> bool {
    score(
        ctx,
        color,
        source,
        incumbent,
        candidate,
        with_residual,
        cache,
    )
    .is_some()
}

/// `None` when `candidate` may not replace `incumbent`; otherwise its pooled
/// distance from the source (lower is better, zero for an exact plane), for
/// choosing between candidates of similar size.
pub(super) fn score(
    ctx: &EncodingContext,
    color: &Image3F,
    source: &AlphaPlane,
    incumbent: &AlphaPlane,
    candidate: &AlphaPlane,
    // A function completed by a coded residual has the incumbent's kind of
    // error, so only the comparisons with the incumbent apply to it.
    with_residual: bool,
    cache: &GateCache,
) -> Option<f64> {
    let (AlphaPlane::U8(source), AlphaPlane::U8(incumbent), AlphaPlane::U8(candidate)) =
        (source, incumbent, candidate)
    else {
        return None;
    };
    let (width, height) = (color.xsize(), color.ysize());
    let pixels = width.checked_mul(height)?;
    if source.len() != pixels || incumbent.len() != pixels || candidate.len() != pixels {
        return None;
    }
    if candidate == incumbent {
        return Some(f64::INFINITY);
    }
    let errors = plane_errors(source, candidate);
    if errors.squared == 0 {
        return Some(0.0);
    }
    let incumbent_errors = cache.errors.get_or_init(|| plane_errors(source, incumbent));
    if errors.squared > incumbent_errors.squared {
        return None;
    }
    if with_residual {
        if errors.endpoints > incumbent_errors.endpoints {
            return None;
        }
    } else if errors.max > incumbent_errors.max
        || errors.max > 2
        || errors.squared * 25 > pixels as u64 * 9
        || errors.endpoints != 0
    {
        return None;
    }

    let layout = Layout {
        width,
        height,
        black: ctx.extra_black,
        color,
    };
    let gate = Gate::select(ctx);
    match gate {
        Gate::Proxy => {
            let reference = cache
                .proxy
                .get_or_init(|| {
                    // K has no color to composite: its second view is the
                    // ink amount.
                    let luma_scale = if ctx.extra_black {
                        1.0
                    } else {
                        source_luma_scale(ctx, color)
                    };
                    let views = layout.prepare(ctx, source, luma_scale)?;
                    let deficits = layout.deficits(ctx, &views, incumbent, luma_scale)?;
                    Some(Incumbent {
                        luma_scale,
                        views,
                        deficits,
                    })
                })
                .as_ref()?;
            let deficits =
                layout.deficits(ctx, &reference.views, candidate, reference.luma_scale)?;
            let reference = &reference.deficits;
            // The comparison is between two sums of the same windows: a
            // relative margin only absorbs their summation order.
            let not_worse = |new: f64, old: f64| new <= old * (1.0 + 1e-6) + 1e-9;
            let pooled: f64 = deficits.iter().sum();
            let passes = if with_residual {
                // A residual trades error against the incumbent sample by
                // sample, as any other rate-distortion choice does, so its
                // views are pooled. But a view that gets visibly worse is
                // not paid for by the others: on 209 candidates of 15 real
                // planes the pooled rule alone accepted all 20 that
                // SSIMULACRA2 + CVVDP reject (rings over a bright color lose
                // on black while the plane itself improves); with this slack
                // it accepts 8 of them and still none of the 189 they pass.
                let no_view_worse = deficits
                    .iter()
                    .zip(reference)
                    .all(|(&new, &old)| new <= old + VIEW_SLACK);
                no_view_worse && not_worse(pooled, reference.iter().sum())
            } else {
                // A single-frame function must not lose any view.
                deficits
                    .iter()
                    .zip(reference)
                    .all(|(&new, &old)| not_worse(new, old))
            };
            passes.then_some(pooled)
        }
        #[cfg(feature = "extraslow")]
        Gate::Exact => {
            let views = exact_views(ctx, color, source, incumbent, candidate, &cache.exact)?;
            let passes = if with_residual {
                let ss: f64 = views.iter().map(|v| v[1] - v[0]).sum();
                let cv: f64 = views.iter().map(|v| v[3] - v[2]).sum();
                // 0.001 of CVVDP distortion per view absorbs a nonzero score
                // beside an incumbent that is exact in that view.
                ss >= 0.0 && cv <= 3e-3
            } else {
                views.iter().all(|&[ss_old, ss_new, cv_old, cv_new]| {
                    ss_new + 1e-6 >= ss_old && cv_new <= cv_old * (1.0 + 1e-6) + 1e-12
                })
            };
            passes.then(|| -views.iter().map(|v| v[1]).sum::<f64>())
        }
    }
}

/// What the direct metrics keep between the candidates of one race: the
/// source color as 8-bit sRGB and, per view, both prepared references and
/// the incumbent's scores.
#[cfg(feature = "extraslow")]
pub(super) struct ExactReference {
    rgb: Vec<[u8; 3]>,
    views: Vec<ExactView>,
}

#[cfg(feature = "extraslow")]
struct ExactView {
    ssimulacra2: ssimu2_lite::PreparedReference,
    cvvdp: cvvdp_lite::PreparedReference,
    incumbent: (f64, f64),
}

/// SSIMULACRA2 (higher is better) and CVVDP distortion (lower is better) of
/// the incumbent and the candidate per view: `[ss_old, ss_new, cv_old,
/// cv_new]`. The views are the scalar plane, then for K its ink-amount
/// inversion (no ICC rendering is inferred from a native black plate),
/// otherwise the 8-bit sRGB color composited over black and over white, as
/// the external trial composites decoded files.
#[cfg(feature = "extraslow")]
fn exact_views(
    ctx: &EncodingContext,
    color: &Image3F,
    source: &[u8],
    incumbent: &[u8],
    candidate: &[u8],
    cache: &std::sync::OnceLock<Option<ExactReference>>,
) -> Option<Vec<[f64; 4]>> {
    let (width, height) = (color.xsize(), color.ysize());
    let views = if ctx.extra_black { 2 } else { 3 };
    let size = (width, height);
    let reference = cache
        .get_or_init(|| {
            let rgb = if ctx.extra_black {
                Vec::new()
            } else {
                source_srgb8(ctx, color)
            };
            let metric = cvvdp_lite::CvvdpLite::new(cvvdp_lite::Display::standard_4k()).ok()?;
            let prepared = ctx
                .thread_pool
                .map(views, |view| {
                    let mut samples = vec![[0; 3]; source.len()];
                    composite(view, ctx.extra_black, source, &rgb, &mut samples);
                    let mut workspace = cvvdp_lite::Workspace::new();
                    let (ss, cv) = ctx.thread_pool.join(
                        || ssimu2_lite::PreparedReference::new(ss2_image(&samples, size)?).ok(),
                        || {
                            metric
                                .prepare_reference_with(vdp_image(&samples, size)?, &mut workspace)
                                .ok()
                        },
                    );
                    let (ssimulacra2, cvvdp) = (ss?, cv?);
                    composite(view, ctx.extra_black, incumbent, &rgb, &mut samples);
                    let incumbent =
                        exact_compare(ctx, size, &samples, &ssimulacra2, &cvvdp, workspace)?;
                    Some(ExactView {
                        ssimulacra2,
                        cvvdp,
                        incumbent,
                    })
                })
                .into_iter()
                .collect::<Option<Vec<_>>>()?;
            Some(ExactReference {
                rgb,
                views: prepared,
            })
        })
        .as_ref()?;
    ctx.thread_pool
        .map(views, |view| {
            let prepared = &reference.views[view];
            let mut samples = vec![[0; 3]; candidate.len()];
            composite(
                view,
                ctx.extra_black,
                candidate,
                &reference.rgb,
                &mut samples,
            );
            let (ss_new, cv_new) = exact_compare(
                ctx,
                size,
                &samples,
                &prepared.ssimulacra2,
                &prepared.cvvdp,
                cvvdp_lite::Workspace::new(),
            )?;
            let (ss_old, cv_old) = prepared.incumbent;
            Some([ss_old, ss_new, cv_old, cv_new])
        })
        .into_iter()
        .collect()
}

#[cfg(feature = "extraslow")]
fn exact_compare(
    ctx: &EncodingContext,
    size: (usize, usize),
    samples: &[[u8; 3]],
    ssimulacra2: &ssimu2_lite::PreparedReference,
    cvvdp: &cvvdp_lite::PreparedReference,
    workspace: cvvdp_lite::Workspace,
) -> Option<(f64, f64)> {
    let (ss, cv) = ctx.thread_pool.join(
        || ssimulacra2.compare(ss2_image(samples, size)?).ok(),
        || {
            Some(
                cvvdp
                    .evaluator_with(workspace)
                    .compare(vdp_image(samples, size)?)
                    .ok()?
                    .distortion,
            )
        },
    );
    let (ss, cv) = (ss?, cv?);
    (ss.is_finite() && cv.is_finite()).then_some((ss, cv))
}

#[cfg(feature = "extraslow")]
fn ss2_image(
    samples: &[[u8; 3]],
    (width, height): (usize, usize),
) -> Option<ssimu2_lite::RgbImage<'_>> {
    ssimu2_lite::RgbImage::srgb8(samples.as_flattened(), width, height, width * 3).ok()
}

#[cfg(feature = "extraslow")]
fn vdp_image(
    samples: &[[u8; 3]],
    (width, height): (usize, usize),
) -> Option<cvvdp_lite::RgbImage<'_>> {
    cvvdp_lite::RgbImage::srgb8(samples.as_flattened(), width, height, width * 3).ok()
}

/// One view of `plane` as interleaved 8-bit RGB: the plane as gray, for K
/// its inversion, otherwise `rgb` composited over black (view 1) or white
/// (view 2) with rounding.
#[cfg(feature = "extraslow")]
fn composite(view: usize, black_plate: bool, plane: &[u8], rgb: &[[u8; 3]], out: &mut [[u8; 3]]) {
    debug_assert_eq!(plane.len(), out.len());
    if view == 0 || black_plate {
        let flip = if view == 0 { 0 } else { 255 };
        for (&alpha, dst) in plane.iter().zip(out) {
            *dst = [alpha ^ flip; 3];
        }
    } else {
        debug_assert_eq!(plane.len(), rgb.len());
        if view == 1 {
            composite_color::<false>(plane, rgb, out);
        } else {
            composite_color::<true>(plane, rgb, out);
        }
    }
}

#[cfg(all(
    feature = "extraslow",
    not(all(target_arch = "aarch64", feature = "neon"))
))]
fn composite_color<const WHITE: bool>(plane: &[u8], rgb: &[[u8; 3]], out: &mut [[u8; 3]]) {
    static OVER_BLACK: std::sync::OnceLock<Box<[[u8; 256]; 256]>> = std::sync::OnceLock::new();
    let over_black = OVER_BLACK.get_or_init(|| {
        let mut table = Box::new([[0u8; 256]; 256]);
        for (alpha, row) in table.iter_mut().enumerate() {
            for (value, cell) in row.iter_mut().enumerate() {
                *cell = ((alpha * value * 2 + 255) / 510) as u8;
            }
        }
        table
    });
    for ((&alpha, &color), dst) in plane.iter().zip(rgb).zip(out) {
        let row = &over_black[usize::from(alpha)];
        // Over white the uncovered share adds exactly `255 - alpha`.
        let lift = if WHITE { 255 - alpha } else { 0 };
        *dst = color.map(|value| row[usize::from(value)] + lift);
    }
}

#[cfg(all(feature = "extraslow", target_arch = "aarch64", feature = "neon"))]
fn composite_color<const WHITE: bool>(plane: &[u8], rgb: &[[u8; 3]], out: &mut [[u8; 3]]) {
    use std::arch::aarch64::*;

    let done = plane.len().min(rgb.len()).min(out.len()) / 16 * 16;
    // SAFETY: each complete block has 16 alpha and 16 interleaved RGB pixels.
    unsafe {
        let half = vdupq_n_u16(128);
        let multiply = |alpha, color| {
            let lo = vmlal_u8(half, vget_low_u8(alpha), vget_low_u8(color));
            let hi = vmlal_high_u8(half, alpha, color);
            // Round a byte product p divided by 255 using only u16 lanes.
            let lo = vaddq_u16(lo, vshrq_n_u16::<8>(lo));
            let hi = vaddq_u16(hi, vshrq_n_u16::<8>(hi));
            vshrn_high_n_u16::<8>(vshrn_n_u16::<8>(lo), hi)
        };
        for i in (0..done).step_by(16) {
            let a = vld1q_u8(plane.as_ptr().add(i));
            let c = vld3q_u8(rgb.as_ptr().add(i).cast());
            let [r, g, b] = [multiply(a, c.0), multiply(a, c.1), multiply(a, c.2)];
            let color = if WHITE {
                let lift = vmvnq_u8(a);
                uint8x16x3_t(vaddq_u8(r, lift), vaddq_u8(g, lift), vaddq_u8(b, lift))
            } else {
                uint8x16x3_t(r, g, b)
            };
            vst3q_u8(out.as_mut_ptr().add(i).cast(), color);
        }
    }
    for ((&alpha, &color), dst) in plane[done..].iter().zip(&rgb[done..]).zip(&mut out[done..]) {
        let lift = if WHITE { 255 - alpha } else { 0 };
        *dst = color.map(|v| ((u16::from(alpha) * u16::from(v) + 127) / 255) as u8 + lift);
    }
}

/// The source color as interleaved 8-bit sRGB.
#[cfg(feature = "extraslow")]
fn source_srgb8(ctx: &EncodingContext, color: &Image3F) -> Vec<[u8; 3]> {
    let (width, height) = (color.xsize(), color.ysize());
    let mut rgb = vec![[0; 3]; width * height];
    if rgb.is_empty() {
        return rgb;
    }
    let rows = height.div_ceil(ctx.thread_pool.num_threads());
    let mut bands: Vec<_> = rgb.chunks_mut(width * rows).collect();
    let srgb = Srgb8::table();
    ctx.thread_pool.for_each_mut(&mut bands, |band, dst| {
        let mut linear: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; width]);
        for (row, dst) in dst.chunks_exact_mut(width).enumerate() {
            let y = band * rows + row;
            let [r, g, b] = &mut linear;
            crate::xyb::xyb_to_rgb_rows(
                &ctx.xyb,
                [
                    &color.plane_row(0, y)[..width],
                    &color.plane_row(1, y)[..width],
                    &color.plane_row(2, y)[..width],
                ],
                [r, g, b],
            );
            for (((&r, &g), &b), dst) in r.iter().zip(g.iter()).zip(b.iter()).zip(dst) {
                *dst = [srgb.convert(r), srgb.convert(g), srgb.convert(b)];
            }
        }
    });
    rgb
}

#[cfg(feature = "extraslow")]
const SRGB8_STEPS: usize = 4096;

/// Rounding boundaries for `round(255 * sRGB OETF(linear))`.
#[cfg(feature = "extraslow")]
struct Srgb8 {
    /// `bounds[i]` is the linear value from which the result is `> i`.
    bounds: [f32; 256],
    guess: [u8; SRGB8_STEPS + 1],
}

#[cfg(feature = "extraslow")]
impl Srgb8 {
    fn table() -> &'static Self {
        static TABLE: std::sync::OnceLock<Srgb8> = std::sync::OnceLock::new();
        TABLE.get_or_init(|| {
            let mut bounds = [f32::INFINITY; 256];
            for (i, bound) in bounds[..255].iter_mut().enumerate() {
                let encoded = (i as f64 + 0.5) / 255.0;
                *bound = (if encoded <= 0.04045 {
                    encoded / 12.92
                } else {
                    ((encoded + 0.055) / 1.055).powf(2.4)
                }) as f32;
            }
            let guess = std::array::from_fn(|i| {
                let linear = i as f32 / SRGB8_STEPS as f32;
                bounds.partition_point(|&bound| bound <= linear) as u8
            });
            Self { bounds, guess }
        })
    }

    fn convert(&self, linear: f32) -> u8 {
        // NaN compares false everywhere and lands on 0.
        let linear = if linear > 0.0 { linear.min(1.0) } else { 0.0 };
        let value = self.guess[(linear * SRGB8_STEPS as f32) as usize];
        // The closest pair of rounding boundaries is farther apart than a
        // table step. The lower end of a step can miss at most one boundary.
        value + u8::from(linear >= self.bounds[usize::from(value)])
    }
}

/// The source XYB lightness is normalized to SDR white, or to the brightest
/// source sample when the image goes above it (HDR, wide gamut).
fn source_luma_scale(ctx: &EncodingContext, color: &Image3F) -> f32 {
    let (width, height) = (color.xsize(), color.ysize());
    let mut peak = crate::xyb::rgb_to_xyb_pixel_f32(&ctx.xyb, 1.0, 1.0, 1.0).1;
    for y in 0..height {
        peak = color.plane_row(1, y)[..width]
            .iter()
            .fold(peak, |peak, &value| peak.max(finite_luma(value)));
    }
    if peak > 0.0 { 1.0 / peak } else { 1.0 }
}

fn finite_luma(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[derive(Clone, Copy)]
struct Layout<'a> {
    width: usize,
    height: usize,
    black: bool,
    color: &'a Image3F,
}

impl Layout<'_> {
    fn views(&self) -> usize {
        if self.black { 2 } else { 3 }
    }

    /// Hands `use_image` one view of `plane`: the plane itself, then for K
    /// its ink amount, otherwise the lightness composites over black and
    /// over white. Only the view being used is materialized.
    fn with_view<T>(
        &self,
        view: usize,
        plane: &[u8],
        luma_scale: f32,
        use_image: impl FnOnce(msssim_lite::GrayImage<'_>) -> Option<T>,
    ) -> Option<T> {
        if view == 0 {
            let image =
                msssim_lite::GrayImage::u8(plane, self.width, self.height, self.width).ok()?;
            return use_image(image);
        }
        let samples: Vec<f32> = if self.black {
            plane
                .iter()
                .map(|&sample| 1.0 - f32::from(sample) * (1.0 / 255.0))
                .collect()
        } else if view == 1 {
            self.lightness_composite::<false>(plane, luma_scale)
        } else {
            self.lightness_composite::<true>(plane, luma_scale)
        };
        let image =
            msssim_lite::GrayImage::unit_f32(&samples, self.width, self.height, self.width).ok()?;
        use_image(image)
    }

    fn lightness_composite<const WHITE: bool>(&self, plane: &[u8], scale: f32) -> Vec<f32> {
        let mut samples = Vec::with_capacity(plane.len());
        for (y, row) in plane.chunks_exact(self.width).enumerate() {
            let luma = &self.color.plane_row(1, y)[..self.width];
            samples.extend(row.iter().zip(luma).map(|(&alpha, &luma)| {
                let alpha = f32::from(alpha) * (1.0 / 255.0);
                let lightness = (finite_luma(luma) * scale) * alpha;
                if WHITE {
                    lightness + 1.0 - alpha
                } else {
                    lightness
                }
            }));
        }
        samples
    }

    fn prepare(
        &self,
        ctx: &EncodingContext,
        plane: &[u8],
        luma_scale: f32,
    ) -> Option<Vec<msssim_lite::PreparedReference>> {
        ctx.thread_pool
            .map(self.views(), |view| {
                self.with_view(view, plane, luma_scale, |image| {
                    msssim_lite::PreparedReference::new(image).ok()
                })
            })
            .into_iter()
            .collect()
    }

    /// Per view, the multi-scale SSIM deficit of `plane` from `reference`.
    fn deficits(
        &self,
        ctx: &EncodingContext,
        reference: &[msssim_lite::PreparedReference],
        plane: &[u8],
        luma_scale: f32,
    ) -> Option<Vec<f64>> {
        ctx.thread_pool
            .map(reference.len(), |view| {
                self.with_view(view, plane, luma_scale, |image| {
                    let score = reference[view].compare(image).ok()?;
                    Some(score.deficit()).filter(|deficit| deficit.is_finite())
                })
            })
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> EncodingContext {
        EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            1.5,
            1,
        )
    }

    #[test]
    fn error_statistics_include_all_chunks_and_the_tail() {
        let source = vec![0u8; 32768 * 3 + 1];
        let mut candidate = vec![255u8; source.len()];
        *candidate.last_mut().unwrap() = 7;
        let errors = plane_errors(&source, &candidate);
        assert_eq!(errors.squared, (32768 * 3) * 255u64.pow(2) + 49);
        assert_eq!(errors.max, 255);
        assert_eq!(errors.endpoints, source.len());
    }

    #[cfg(feature = "extraslow")]
    #[test]
    fn exact_color_composites_match_integer_rounding() {
        // All byte products, plus a tail shorter than one SIMD vector.
        let alpha: Vec<_> = (0..=255).flat_map(|a| [a; 256]).chain([253]).collect();
        let rgb: Vec<_> = (0..alpha.len())
            .map(|i| {
                let v = i as u8;
                [v, v.wrapping_add(17), 255 - v]
            })
            .collect();
        let mut out = vec![[0; 3]; alpha.len()];
        for view in 1..=2 {
            composite(view, false, &alpha, &rgb, &mut out);
            for ((&a, &src), &dst) in alpha.iter().zip(&rgb).zip(&out) {
                let expected = src.map(|v| {
                    let covered = (u32::from(a) * u32::from(v) * 2 + 255) / 510;
                    (covered + if view == 1 { 0 } else { 255 - u32::from(a) }) as u8
                });
                assert_eq!(dst, expected);
            }
        }
    }

    #[cfg(feature = "extraslow")]
    #[test]
    fn srgb_lookup_resolves_rounding_boundaries() {
        let table = Srgb8::table();
        for (i, &boundary) in table.bounds[..255].iter().enumerate() {
            assert_eq!(table.convert(boundary.next_down()), i as u8);
            assert_eq!(table.convert(boundary), i as u8 + 1);
            assert_eq!(table.convert(boundary.next_up()), i as u8 + 1);
        }
        assert_eq!(table.convert(f32::NAN), 0);
        assert_eq!(table.convert(f32::NEG_INFINITY), 0);
        assert_eq!(table.convert(f32::INFINITY), 255);
    }

    #[test]
    fn preserves_exact_alpha_and_hard_endpoints() {
        let ctx = context();
        let color = Image3F::try_new(32, 32).unwrap();
        let source = AlphaPlane::U8(vec![0; 32 * 32]);
        let softened = AlphaPlane::U8(vec![1; 32 * 32]);
        assert!(!acceptable(
            &ctx,
            &color,
            &source,
            &source,
            &softened,
            false,
            &Default::default()
        ));
        assert!(acceptable(
            &ctx,
            &color,
            &source,
            &softened,
            &source,
            false,
            &Default::default()
        ));
    }

    #[test]
    fn residual_candidates_are_judged_by_pooled_structure() {
        let ctx = context();
        let (width, height) = (70, 45);
        let color = Image3F::try_new(width, height).unwrap();
        let ramp: Vec<u8> = (0..width * height)
            .map(|i| (40 + (i % width) * 2) as u8)
            .collect();
        // The incumbent is off by a smooth two levels; one candidate is off
        // by one level, the other breaks the ramp into bands.
        let incumbent = AlphaPlane::U8(ramp.iter().map(|&v| v + 2).collect());
        let closer = AlphaPlane::U8(ramp.iter().map(|&v| v + 1).collect());
        let banded = AlphaPlane::U8(ramp.iter().map(|&v| v / 4 * 4).collect());
        let source = AlphaPlane::U8(ramp);
        let cache = Default::default();
        assert!(acceptable(
            &ctx, &color, &source, &incumbent, &closer, true, &cache
        ));
        assert!(!acceptable(
            &ctx, &color, &source, &incumbent, &banded, true, &cache
        ));
    }
}
