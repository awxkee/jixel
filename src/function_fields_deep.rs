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
//! Function fields for 9- to 16-bit planes.
//!
//! The shape of a field does not depend on its sample depth, so the geometry
//! and both axes come from the 8-bit fits on a rounded view of the plane.
//! What depth changes is the profile, the residual and their bias: those are
//! rebuilt here from the full-depth samples, and the plane and its residual
//! keep the source's depth.
//!
//! The candidates still have to be raced and judged by the caller, with
//! [`dead_zone`], [`compose`] and [`score`] standing in for the 8-bit
//! counterparts.

use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::function_fields::{self, FittedProfile, NestedProfile, ResidualFit};
use crate::image::Image3F;
use crate::thread_pool::ThreadPool;

/// How much one view's deficit may grow when the pooled deficit does not;
/// the same slack as the 8-bit proxy.
const VIEW_SLACK: f64 = 1e-4;

fn depth(plane: &AlphaPlane) -> Option<(&[u16], u8, i32)> {
    match plane {
        AlphaPlane::U16 { data, bits } if (9..=16).contains(bits) => {
            Some((data, *bits, (1i32 << *bits) - 1))
        }
        _ => None,
    }
}

/// Half the range: the base frame carries `P - bias` and the residual plane
/// `A - P + bias`, which then fits the plane's own depth.
fn bias(max: i32) -> i32 {
    (max + 1) / 2
}

/// A tile, else (for lossy planes) a smooth function, fitted on the 8-bit
/// view and carried to the plane's depth.
pub(crate) fn fit(
    pool: &ThreadPool,
    plane: &AlphaPlane,
    width: usize,
    height: usize,
    lossy: bool,
) -> Option<ResidualFit> {
    let (data, bits, max) = depth(plane)?;
    if width.checked_mul(height)? != data.len() {
        return None;
    }
    let scale = max as u32;
    let view = AlphaPlane::U8(
        data.iter()
            .map(|&value| ((u32::from(value) * 255 + scale / 2) / scale) as u8)
            .collect(),
    );
    let (periodic, smooth) = pool.join(
        || function_fields::fit_periodic(&view, width, height),
        || {
            lossy
                .then(|| function_fields::fit_residual(pool, &view, width, height))
                .flatten()
        },
    );
    deepen(periodic.or(smooth)?, data, bits, max, width)
}

/// Rebuild the profile of `coarse` from full-depth samples: per coordinate
/// the median, moved into the range that keeps every residual at that
/// coordinate within the plane's depth; empty coordinates interpolate.
fn deepen(
    coarse: ResidualFit,
    data: &[u16],
    bits: u8,
    max: i32,
    width: usize,
) -> Option<ResidualFit> {
    let NestedProfile { profile, u, v } = coarse.base.model;
    let length = profile.len();
    let bias = bias(max);
    // Samples grouped by coordinate with a counting sort.
    let mut starts = vec![0usize; length + 1];
    for (row, &vy) in data.chunks_exact(width).zip(&v) {
        for &ux in &u[..row.len()] {
            starts[(ux + vy) as usize + 1] += 1;
        }
    }
    for i in 0..length {
        starts[i + 1] += starts[i];
    }
    let mut grouped = vec![0u16; data.len()];
    let mut next = starts.clone();
    for (row, &vy) in data.chunks_exact(width).zip(&v) {
        for (&value, &ux) in row.iter().zip(&u) {
            let slot = &mut next[(ux + vy) as usize];
            grouped[*slot] = value;
            *slot += 1;
        }
    }
    let mut deep = vec![0i32; length];
    let mut previous: Option<(usize, i32)> = None;
    for i in 0..length {
        let group = &mut grouped[starts[i]..starts[i + 1]];
        if group.is_empty() {
            continue;
        }
        let middle = group.len() / 2;
        let median = i32::from(*group.select_nth_unstable(middle).1);
        let lo = i32::from(*group.iter().min()?);
        let hi = i32::from(*group.iter().max()?);
        let value = median.clamp(hi - (max - bias), lo + bias);
        deep[i] = value;
        match previous {
            Some((start, from)) => {
                for j in start + 1..i {
                    let span = (i - start) as i64;
                    deep[j] = from + (i64::from(value - from) * (j - start) as i64 / span) as i32;
                }
            }
            None => deep[..i].fill(value),
        }
        previous = Some((i, value));
    }
    let (last, value) = previous?;
    deep[last + 1..].fill(value);
    drop(grouped);

    let mut function = Vec::with_capacity(data.len());
    let mut residual = Vec::with_capacity(data.len());
    let (mut missed, mut error, mut max_error) = (0usize, 0u64, 0i32);
    for (row, &vy) in data.chunks_exact(width).zip(&v) {
        for (&value, &ux) in row.iter().zip(&u) {
            let model = deep[(ux + vy) as usize];
            let difference = i32::from(value) - model;
            missed += usize::from(difference != 0);
            error += u64::from(difference.unsigned_abs()).pow(2);
            max_error = max_error.max(difference.abs());
            function.push(model as u16);
            residual.push((difference + bias) as u16);
        }
    }
    // A tile is only worth its exact residual while that residual is sparse;
    // the low bits of a deep plane need not repeat with its top eight.
    if coarse.periodic && missed * 8 > data.len() {
        return None;
    }
    for value in &mut deep {
        *value -= bias;
    }
    Some(ResidualFit {
        base: FittedProfile {
            model: NestedProfile {
                profile: deep,
                u,
                v,
            },
            reconstructed: AlphaPlane::U16 {
                data: function,
                bits,
            },
            squared_error: error,
            max_error: max_error.min(255) as u8,
        },
        residual: AlphaPlane::U16 {
            data: residual,
            bits,
        },
        periodic: coarse.periodic,
    })
}

/// The residual with every misfit of at most `zone` 8-bit code values
/// dropped. Samples whose source is fully transparent or opaque keep theirs.
pub(crate) fn dead_zone(fit: &ResidualFit, source: &AlphaPlane, zone: u8) -> AlphaPlane {
    let (Some((source, _, max)), Some((residual, bits, _))) = (depth(source), depth(&fit.residual))
    else {
        return fit.residual.clone();
    };
    let bias = bias(max);
    let threshold = ((i32::from(zone) * max + 127) / 255).unsigned_abs();
    AlphaPlane::U16 {
        data: residual
            .iter()
            .zip(source)
            .map(|(&r, &s)| {
                let keep = s == 0
                    || i32::from(s) == max
                    || (i32::from(r) - bias).unsigned_abs() > threshold;
                if keep { r } else { bias as u16 }
            })
            .collect(),
        bits,
    }
}

/// The decoder's sum of the base and a (possibly lossy) residual plane.
pub(crate) fn compose(fit: &ResidualFit, residual: &AlphaPlane) -> Option<AlphaPlane> {
    let (function, bits, max) = depth(&fit.base.reconstructed)?;
    let (residual, residual_bits, _) = depth(residual)?;
    if bits != residual_bits || function.len() != residual.len() {
        return None;
    }
    let bias = bias(max);
    Some(AlphaPlane::U16 {
        data: function
            .iter()
            .zip(residual)
            .map(|(&f, &r)| (i32::from(f) - bias + i32::from(r)).clamp(0, max) as u16)
            .collect(),
        bits,
    })
}

/// What every candidate of one race is compared with.
struct Incumbent {
    luma: Vec<f32>,
    views: Vec<msssim_lite::PreparedReference>,
    deficits: Vec<f64>,
    squared_error: u64,
    moved_endpoints: usize,
}

/// Kept between the candidates of one race (same source, color and
/// incumbent).
#[derive(Default)]
pub(crate) struct GateCache {
    incumbent: std::sync::OnceLock<Option<Incumbent>>,
}

/// `None` when the residual coding `candidate` may not replace `incumbent`;
/// otherwise its pooled distance from the source (lower is better, zero for
/// an exact plane). The proxy of the 8-bit gate at the plane's depth:
/// multi-scale SSIM of the plane and of the source lightness composited over
/// black and over white.
pub(crate) fn score(
    ctx: &EncodingContext,
    color: &Image3F,
    source: &AlphaPlane,
    incumbent: &AlphaPlane,
    candidate: &AlphaPlane,
    cache: &GateCache,
) -> Option<f64> {
    let (source, bits, max) = depth(source)?;
    let (incumbent, incumbent_bits, _) = depth(incumbent)?;
    let (candidate, candidate_bits, _) = depth(candidate)?;
    let (width, height) = (color.xsize(), color.ysize());
    let pixels = width.checked_mul(height)?;
    if bits != incumbent_bits
        || bits != candidate_bits
        || [source.len(), incumbent.len(), candidate.len()] != [pixels; 3]
    {
        return None;
    }
    if candidate == incumbent {
        return Some(f64::INFINITY);
    }
    let errors = |plane: &[u16]| {
        source
            .iter()
            .zip(plane)
            .fold((0u64, 0usize), |(squared, moved), (&s, &v)| {
                let endpoint = s == 0 || i32::from(s) == max;
                (
                    squared + u64::from(s.abs_diff(v)).pow(2),
                    moved + usize::from(endpoint && s != v),
                )
            })
    };
    let (squared_error, moved_endpoints) = errors(candidate);
    if squared_error == 0 {
        return Some(0.0);
    }
    let unit = 1.0 / max as f32;
    // One view at a time: the plane, then the lightness composites.
    let view = |index: usize, plane: &[u16], luma: &[f32]| -> Vec<f32> {
        let alpha = plane.iter().map(|&sample| f32::from(sample) * unit);
        match index {
            0 => alpha.collect(),
            1 => alpha.zip(luma).map(|(alpha, &luma)| luma * alpha).collect(),
            _ => alpha
                .zip(luma)
                .map(|(alpha, &luma)| luma * alpha + 1.0 - alpha)
                .collect(),
        }
    };
    let size = (width, height);
    let reference = cache
        .incumbent
        .get_or_init(|| {
            let luma = lightness(ctx, color);
            let views = ctx
                .thread_pool
                .map(3, |index| {
                    let samples = view(index, source, &luma);
                    msssim_lite::PreparedReference::new(image(&samples, size).ok()?).ok()
                })
                .into_iter()
                .collect::<Option<Vec<_>>>()?;
            let deficits = ctx
                .thread_pool
                .map(3, |index| {
                    let samples = view(index, incumbent, &luma);
                    Some(
                        views[index]
                            .compare(image(&samples, size).ok()?)
                            .ok()?
                            .deficit(),
                    )
                })
                .into_iter()
                .collect::<Option<Vec<f64>>>()?;
            let (squared_error, moved_endpoints) = errors(incumbent);
            Some(Incumbent {
                luma,
                views,
                deficits,
                squared_error,
                moved_endpoints,
            })
        })
        .as_ref()?;
    if squared_error > reference.squared_error || moved_endpoints > reference.moved_endpoints {
        return None;
    }
    let deficits = ctx
        .thread_pool
        .map(3, |index| {
            let samples = view(index, candidate, &reference.luma);
            let score = reference.views[index]
                .compare(image(&samples, size).ok()?)
                .ok()?;
            Some(score.deficit()).filter(|deficit| deficit.is_finite())
        })
        .into_iter()
        .collect::<Option<Vec<f64>>>()?;
    let no_view_worse = deficits
        .iter()
        .zip(&reference.deficits)
        .all(|(&new, &old)| new <= old + VIEW_SLACK);
    let (pooled, old): (f64, f64) = (deficits.iter().sum(), reference.deficits.iter().sum());
    (no_view_worse && pooled <= old * (1.0 + 1e-6) + 1e-9).then_some(pooled)
}

fn image(
    samples: &[f32],
    (width, height): (usize, usize),
) -> Result<msssim_lite::GrayImage<'_>, msssim_lite::Error> {
    msssim_lite::GrayImage::unit_f32(samples, width, height, width)
}

/// Lightness of the source color in `0..=1`: its XYB luma over the luma of
/// SDR white, or of the brightest sample when the image goes above that.
fn lightness(ctx: &EncodingContext, color: &Image3F) -> Vec<f32> {
    let (width, height) = (color.xsize(), color.ysize());
    let mut peak = crate::xyb::rgb_to_xyb_pixel_f32(&ctx.xyb, 1.0, 1.0, 1.0).1;
    let mut luma = Vec::with_capacity(width * height);
    for y in 0..height {
        for &value in &color.plane_row(1, y)[..width] {
            let value = if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            };
            peak = peak.max(value);
            luma.push(value);
        }
    }
    if peak > 0.0 {
        let scale = 1.0 / peak;
        for value in &mut luma {
            *value *= scale;
        }
    }
    luma
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A radial glow at `bits` with a hard-edged disc and a little noise in
    /// the low bits.
    fn glow(width: usize, height: usize, bits: u8) -> Vec<u16> {
        let max = f64::from((1u32 << bits) - 1);
        let mut state = 7u32;
        (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f64, (i / width) as f64);
                let (dx, dy) = (x - 0.5 * width as f64, y - 0.5 * height as f64);
                let r = (dx * dx + dy * dy).sqrt() / width as f64;
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if r < 0.15 {
                    0
                } else {
                    let noise = f64::from(state % 5) - 2.0;
                    let value = max * (-((r - 0.15) / 0.2).powi(2)).exp() + noise * max / 4096.0;
                    value.round().clamp(0.0, max) as u16
                }
            })
            .collect()
    }

    #[test]
    fn deep_fit_reconstructs_every_sample_at_its_depth() {
        let (width, height) = (320, 288);
        let pool = ThreadPool::new(2);
        for bits in [10u8, 12, 16] {
            let data = glow(width, height, bits);
            let plane = AlphaPlane::U16 {
                data: data.clone(),
                bits,
            };
            let fitted = fit(&pool, &plane, width, height, true).expect("deep fit");
            let max = (1i32 << bits) - 1;
            let (function, _, _) = depth(&fitted.base.reconstructed).unwrap();
            let (residual, residual_bits, _) = depth(&fitted.residual).unwrap();
            assert_eq!(residual_bits, bits);
            let model = &fitted.base.model;
            for (i, &value) in data.iter().enumerate() {
                let s = (model.u[i % width] + model.v[i / width]) as usize;
                // Base frame entry plus residual sample is the source.
                assert_eq!(model.profile[s] + i32::from(residual[i]), i32::from(value));
                assert_eq!(model.profile[s] + bias(max), i32::from(function[i]));
            }
            let composed = compose(&fitted, &fitted.residual).expect("composition");
            assert_eq!(depth(&composed).unwrap().0, &data[..]);
            // The dead zone keeps hard coverage and only removes small misfits.
            let zoned = dead_zone(&fitted, &plane, 2);
            let (zoned, _, _) = depth(&zoned).unwrap();
            let threshold = (2 * max + 127) / 255;
            for ((&z, &r), &s) in zoned.iter().zip(residual).zip(&data) {
                let misfit = (i32::from(r) - bias(max)).abs();
                if s == 0 || i32::from(s) == max || misfit > threshold {
                    assert_eq!(z, r);
                } else {
                    assert_eq!(i32::from(z), bias(max));
                }
            }
        }
    }

    #[test]
    fn deep_fit_declines_other_planes() {
        let pool = ThreadPool::new(1);
        assert!(fit(&pool, &AlphaPlane::U8(vec![0; 64 * 64]), 64, 64, true).is_none());
        let mut state = 1u32;
        let noise = AlphaPlane::U16 {
            data: (0..96 * 96)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state & 1023) as u16
                })
                .collect(),
            bits: 10,
        };
        assert!(fit(&pool, &noise, 96, 96, true).is_none());
        let short = AlphaPlane::U16 {
            data: vec![0; 10],
            bits: 16,
        };
        assert!(fit(&pool, &short, 64, 64, true).is_none());
    }
}
