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

//! Classify chroma and screen content, and choose frame-wide precision policy.

use super::quantization::{DistanceParams, X_QM_REFINE_MAX_DISTANCE};
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

/// Fraction of exactly constant 8x8 blocks above which a frame is screen content.
const SCREEN_CONTENT_CONSTANT_BLOCKS: f32 = 0.15;
/// Fraction of whole 8x8 blocks whose three XYB planes are each constant.
/// Identical input pixels map to identical XYB values, so the test is exact.
fn constant_block_fraction(xyb: &Image3F) -> f32 {
    let (bw, bh) = (xyb.xsize() / 8, xyb.ysize() / 8);
    if bw == 0 || bh == 0 {
        return 0.0;
    }
    let mut constant = 0usize;
    for by in 0..bh {
        for bx in 0..bw {
            let x0 = bx * 8;
            let is_constant = (0..3).all(|c| {
                let v = xyb.plane_row(c, by * 8)[x0];
                (0..8).all(|iy| {
                    xyb.plane_row(c, by * 8 + iy)[x0..x0 + 8]
                        .iter()
                        .all(|&p| p == v)
                })
            });
            constant += usize::from(is_constant);
        }
    }
    constant as f32 / (bw * bh) as f32
}

const SAT_QM_THRESHOLD: f32 = 0.055;
const PAIR_B_TEXTURE_THRESHOLD: f32 = 0.12;

/// `x_gradient_stat` gate for keeping the fine X quantizer through the
/// x_qm_scale=2 distance band (0.299..=1.25)
const X_QM_GRAD_THRESHOLD: f32 = 0.04;

/// `b_gradient_stat` gate for blue-axis-dominant structure (the B twin of
/// `x_heavy`)
const B_GRAD_THRESHOLD: f32 = 0.45;

const X_HEAVY_B_QM_SCALE: u32 = 7;

/// Below this the X scale is 2 and the extra precision only costs rate.
const PIXEL_CHROMACITY_MIN_DISTANCE: f32 = 1.25;

/// A pixel-step level counts once this many pixels pass it; libjxl takes the
/// single worst pixel, which lets one stray edge refine chroma frame-wide.
const PIXEL_CHROMACITY_MIN_PIXELS: u32 = 4;

/// libjxl's `PixelStatsForChromacityAdjustment`: extra X and B quant-scale
/// steps from the pixel-to-pixel steps of X and of B - Y, plus one B step for
/// strongly exposed blue. Thresholds are libjxl's; the step levels need
/// `PIXEL_CHROMACITY_MIN_PIXELS` pixels, exposed blue a single one.
fn pixel_chromacity_steps(xyb: &Image3F) -> (u32, u32) {
    const X_LEVELS: [f32; 3] = [0.015, 0.022, 0.026];
    const B_LEVELS: [f32; 3] = [0.28, 0.33, 0.38];
    let (w, h) = (xyb.xsize(), xyb.ysize());
    if w < 2 || h < 2 {
        return (0, 0);
    }
    let (mut x_counts, mut b_counts, mut exposed_blue) = ([0u32; 3], [0u32; 3], 0u32);
    let rows = xyb
        .plane_data(0)
        .chunks_exact(w)
        .zip(xyb.plane_data(1).chunks_exact(w))
        .zip(xyb.plane_data(2).chunks_exact(w));
    for (((xp, yp), bp), ((xr, yr), br)) in rows.clone().zip(rows.skip(1)) {
        // Carry the left pixel and its B - Y difference between iterations.
        let mut current = xr.iter().zip(yr).zip(br);
        let Some(((&xl, &yl), &bl)) = current.next() else {
            continue;
        };
        let (mut left_x, mut left_b, mut left_diff) = (xl, bl, bl - yl);
        let previous = xp.iter().zip(yp).zip(bp).skip(1);
        for (((&x, &y), &b), ((&px, &py), &pb)) in current.zip(previous) {
            let step_x = (x - left_x).abs().max((x - px).abs());
            let diff_b = b - y;
            let step_b = (diff_b - left_diff).abs().max((diff_b - (pb - py)).abs());
            for (count, level) in x_counts.iter_mut().zip(X_LEVELS) {
                *count += u32::from(step_x >= level);
            }
            for (count, level) in b_counts.iter_mut().zip(B_LEVELS) {
                *count += u32::from(step_b > level);
            }
            // The gradient factor is nonnegative, so negative exposure never passes.
            let step = (b - left_b).abs() + (b - pb).abs();
            exposed_blue += u32::from((b - y * 1.2) * step >= 0.13);
            left_x = x;
            left_b = b;
            left_diff = diff_b;
        }
        if x_counts[2] >= PIXEL_CHROMACITY_MIN_PIXELS
            && b_counts[2] >= PIXEL_CHROMACITY_MIN_PIXELS
            && exposed_blue > 0
        {
            return (3, 3);
        }
    }
    // Counts shrink level by level, so the passed levels are a prefix.
    let level = |counts: [u32; 3]| {
        counts
            .iter()
            .filter(|&&n| n >= PIXEL_CHROMACITY_MIN_PIXELS)
            .count() as u32
    };
    let b_steps = match level(b_counts) {
        0 => 0,
        l => l - 1 + u32::from(exposed_blue > 0),
    };
    (level(x_counts), b_steps)
}

/// Source-domain companion to the saturation detector. Sparse luminance
/// impulses can dominate its gradient statistics while diffuse chroma remains
/// visible between them. Crosses at one and four pixels estimate point prominence
/// without running the dot fitter or quantizing proxy DCTs.
#[derive(Default, Clone, Copy, Debug)]
struct ChromaContentStats {
    saturation: f32,
    dark_fraction: f32,
    point_fraction: f32,
    point_energy: f32,
    sharp_energy: f32,
}

impl ChromaContentStats {
    fn diffuse_point_chroma(self) -> bool {
        self.saturation >= 0.005
            && self.dark_fraction >= 0.5
            && (0.003..0.05).contains(&self.point_fraction)
            && self.point_energy >= 0.35
            && self.sharp_energy >= 0.12
    }
}

/// Sparse luminance points with diffuse color need extra chroma precision.
/// Keep that protection as distance decreases through the HQ table boundary.
fn apply_point_chroma_policy(
    ctx: &EncodingContext,
    distp: &DistanceParams,
    content: ChromaContentStats,
    is_achromatic: bool,
) {
    // X-heavy content already has a separate chroma precision policy.
    let point_chroma = ctx.speed.effort().chroma_policy
        && !is_achromatic
        && content.diffuse_point_chroma()
        && !ctx.x_heavy();
    ctx.set_point_chroma(point_chroma);
    if point_chroma && distp.distance < 2.25 {
        let base_x = distp.x_qm_scale.max(ctx.x_qm_scale_floor());
        // The default X scale already includes this step below the HQ cutoff.
        let extra_x = u32::from(distp.distance >= X_QM_REFINE_MAX_DISTANCE);
        ctx.raise_x_qm_scale_floor(base_x.saturating_add(extra_x));
        ctx.raise_b_qm_scale(ctx.b_qm_scale().saturating_add(1));
    }
}

fn chroma_content_stats(xyb: &Image3F) -> ChromaContentStats {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let (mut sum, mut n, mut dark, mut points, mut interior) = (0.0, 0u64, 0u64, 0u64, 0u64);
    let (mut point_energy, mut contrast_energy) = (0.0f64, 0.0f64);
    let (mut sharp_energy, mut sharp_contrast) = (0.0f64, 0.0f64);
    for py in (0..h).step_by(4) {
        let xr = xyb.plane_row(0, py);
        let yr = xyb.plane_row(1, py);
        let br = xyb.plane_row(2, py);
        for px in (0..w).step_by(4) {
            let y = yr[px];
            sum += xr[px].abs() + (br[px] - y).abs();
            n += 1;
            dark += u64::from(y < 0.3);
            if px < 4 || py < 4 || px + 4 >= w || py + 4 >= h {
                continue;
            }
            let ring = [
                yr[px - 4],
                yr[px + 4],
                xyb.plane_row(1, py - 4)[px],
                xyb.plane_row(1, py + 4)[px],
            ];
            let sharp_ring = [
                yr[px - 1],
                yr[px + 1],
                xyb.plane_row(1, py - 1)[px],
                xyb.plane_row(1, py + 1)[px],
            ];
            let sharp_background = sharp_ring.iter().sum::<f32>() * 0.25;
            let sharp_peak =
                f64::from((y - sharp_ring.into_iter().fold(f32::NEG_INFINITY, f32::max)).max(0.0));
            let sharp_delta = f64::from(y - sharp_background);
            sharp_energy += sharp_peak * sharp_peak;
            sharp_contrast += sharp_delta * sharp_delta;
            let background = ring.iter().sum::<f32>() * 0.25;
            let peak = y - ring.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let contrast = f64::from(y - background);
            contrast_energy += contrast * contrast;
            let positive = f64::from(peak.max(0.0));
            point_energy += positive * positive;
            points += u64::from(peak > 0.1 && background < 0.35);
            interior += 1;
        }
    }
    ChromaContentStats {
        saturation: if n > 0 { sum / n as f32 } else { 0.0 },
        dark_fraction: if n > 0 { dark as f32 / n as f32 } else { 0.0 },
        point_fraction: if interior > 0 {
            points as f32 / interior as f32
        } else {
            0.0
        },
        sharp_energy: if sharp_contrast > 1e-15 {
            (sharp_energy / sharp_contrast) as f32
        } else {
            0.0
        },
        point_energy: if contrast_energy > 1e-15 {
            (point_energy / contrast_energy) as f32
        } else {
            0.0
        },
    }
}

#[cfg(test)]
fn chroma_saturation_stat(xyb: &Image3F) -> f32 {
    chroma_content_stats(xyb).saturation
}

/// Horizontal opponent-gradient sums on a 1-in-4 row subsample. X and B−Y
/// share one traversal and one set of Y loads. The result is
/// `[sum_x, sum_y_for_x, sum_b_minus_y, sum_y_for_b]`; keeping the two Y
/// reductions explicit makes both classifier ratios independently testable.
/// Achromatic content reads ~0 in both opponent numerators.
pub(crate) type ChromaGradientSumsFn = fn(&[f32], &[f32], &[f32], usize) -> [f32; 4];

#[allow(dead_code)]
pub(crate) fn chroma_gradient_sums_scalar(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32],
    width: usize,
) -> [f32; 4] {
    if width < 2 {
        return [0.0; 4];
    }

    let mut sum_x = 0.0f32;
    let mut sum_y_x = 0.0f32;
    let mut sum_by = 0.0f32;
    let mut sum_y_b = 0.0f32;
    let rows = x_plane
        .chunks_exact(width)
        .zip(y_plane.chunks_exact(width))
        .zip(b_plane.chunks_exact(width))
        .step_by(4);
    for ((xr, yr), br) in rows {
        for ((x, l), b) in xr
            .array_windows::<2>()
            .zip(yr.array_windows::<2>())
            .zip(br.array_windows::<2>())
        {
            let dy = (l[1] - l[0]).abs();
            sum_x += (x[1] - x[0]).abs();
            sum_y_x += dy;
            sum_by += ((b[1] - l[1]) - (b[0] - l[0])).abs();
            sum_y_b += dy;
        }
    }
    [sum_x, sum_y_x, sum_by, sum_y_b]
}

pub(crate) fn selected_chroma_gradient_sums_fn() -> ChromaGradientSumsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |x, y, b, width| unsafe { crate::neon::chroma_gradient_sums_neon(x, y, b, width) }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |x, y, b, width| unsafe { crate::avx::chroma_gradient_sums_avx2(x, y, b, width) };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::chroma_gradient_sums_wasm;
    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
        all(target_arch = "aarch64", feature = "neon")
    )))]
    chroma_gradient_sums_scalar
}

/// Share of high-frequency signal on the blue-yellow axis: mean |∇(B−Y)|
/// over mean |∇Y| on a 1-in-4 row subsample. Content whose structure lives
/// in the B channel (saturated blue detail) is nearly invisible to the
/// luma-driven masking field, exactly like the X statistic
/// covers on the red-green axis.
/// `[x_gradient_stat, b_gradient_stat, chroma_texture_stat]` from one
/// gradient pass. The third is the relative chroma texture: mean of
/// |∇X| + |∇(B−Y)| per sampled pixel pair over the frame's mean |X| + |B−Y|
/// (`saturation_stat`). High on thin chromatic detail over low-chroma
/// ground, low on saturated smooth subjects however colorful.
fn chroma_gradient_stats(ctx: &EncodingContext, xyb: &Image3F, saturation_stat: f32) -> [f32; 4] {
    let sums = (ctx.chroma_gradient_sums)(
        xyb.plane_data(0),
        xyb.plane_data(1),
        xyb.plane_data(2),
        xyb.xsize(),
    );
    chroma_stats_from_sums(sums, xyb.xsize(), xyb.ysize(), saturation_stat)
}

fn chroma_stats_from_sums(
    [sum_x, sum_y_x, sum_by, sum_y_b]: [f32; 4],
    width: usize,
    height: usize,
    saturation_stat: f32,
) -> [f32; 4] {
    // The kernel visits rows 0, 4, 8, ... and `width - 1` pairs per row.
    let pairs = (height.div_ceil(4) * width.saturating_sub(1)) as f32;
    [
        if sum_y_x > f32::EPSILON {
            sum_x / sum_y_x
        } else {
            0.0
        },
        if sum_y_b > f32::EPSILON {
            sum_by / sum_y_b
        } else {
            0.0
        },
        if saturation_stat > 1e-9 && pairs > 0.0 {
            (sum_x + sum_by) / pairs / saturation_stat
        } else {
            0.0
        },
        // Luma gradient density: mean |∇Y| per sampled pixel pair.
        if pairs > 0.0 { sum_y_x / pairs } else { 0.0 },
    ]
}

pub(super) fn configure_chroma(
    ctx: &EncodingContext,
    distance: f32,
    distp: &DistanceParams,
    xyb: &Image3F,
    is_achromatic: bool,
) {
    let slow_chromatic = ctx.speed.effort().chroma_policy && !is_achromatic;
    let content = if slow_chromatic {
        chroma_content_stats(xyb)
    } else {
        ChromaContentStats::default()
    };
    let saturation_stat = content.saturation;
    let [
        x_grad_stat,
        b_grad_stat,
        chroma_texture_stat,
        luma_texture_stat,
    ] = if slow_chromatic {
        chroma_gradient_stats(ctx, xyb, saturation_stat)
    } else {
        [0.0; 4]
    };
    ctx.set_luma_texture(luma_texture_stat);
    let (x_steps, b_steps) = match ctx.speed {
        _ if is_achromatic => (0, 0),
        _ if slow_chromatic => pixel_chromacity_steps(xyb),
        // Without the chroma policy the B steps only cost rate.
        _ if ctx.speed.effort().pixel_chromacity_x && distance >= PIXEL_CHROMACITY_MIN_DISTANCE => {
            (pixel_chromacity_steps(xyb).0, 0)
        }
        _ => (0, 0),
    };
    if distance >= PIXEL_CHROMACITY_MIN_DISTANCE {
        ctx.raise_x_qm_scale_floor(2 + x_steps);
        ctx.raise_b_qm_scale(2 + b_steps);
    }
    // Saturated-content tables are validated under Slow only: at Fast the
    // same swap buys chroma with a d=1 rate premium instead of saving bytes.
    ctx.set_chroma_heavy(slow_chromatic && saturation_stat >= SAT_QM_THRESHOLD);
    // Chroma-texture class: thin chromatic detail whose B energy the pair
    // transforms' coarse first band zeroes. Only the mid-band pays off.
    let pair_b_fine = slow_chromatic
        && (crate::quant_weights::PAIR_B_FINE_MIN_DISTANCE
            ..crate::quant_weights::PAIR_B_FINE_MAX_DISTANCE)
            .contains(&distance)
        && chroma_texture_stat >= PAIR_B_TEXTURE_THRESHOLD;
    ctx.set_pair_b_fine(pair_b_fine);
    // This is a content class, not a distance tier. Keeping its decision
    // independent of the 1.25 table boundary prevents saturated HF images
    // from switching plain B weights, custom opsin and both chroma scales at
    // the same point. Ordinary photos measured far below this gate.
    ctx.set_x_heavy(slow_chromatic && x_grad_stat >= X_QM_GRAD_THRESHOLD);
    ctx.set_b_heavy(slow_chromatic && b_grad_stat >= B_GRAD_THRESHOLD);
    if ctx.x_heavy() && b_steps > 0 {
        ctx.raise_b_qm_scale(X_HEAVY_B_QM_SCALE);
    }

    apply_point_chroma_policy(ctx, distp, content, is_achromatic);
    ctx.set_screen_content(constant_block_fraction(xyb) >= SCREEN_CONTENT_CONSTANT_BLOCKS);
}

#[cfg(test)]
mod tests {
    use super::super::quantization::compute_distance_params;

    #[test]
    fn chroma_texture_stat_separates_thin_chroma_from_smooth_saturation() {
        fn texture(xyb: &crate::image::Image3F) -> f32 {
            let sums = super::chroma_gradient_sums_scalar(
                xyb.plane_data(0),
                xyb.plane_data(1),
                xyb.plane_data(2),
                xyb.xsize(),
            );
            let sat = super::chroma_saturation_stat(xyb);
            super::chroma_stats_from_sums(sums, xyb.xsize(), xyb.ysize(), sat)[2]
        }
        let mut xyb = crate::image::Image3F::new(32, 32);
        for y in 0..32 {
            let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.0);
            y_row.fill(0.5);
            b_row.fill(0.5);
        }
        assert_eq!(texture(&xyb), 0.0);
        // Smooth, strongly saturated: high chroma, zero texture.
        for y in 0..32 {
            xyb.all_plane_rows_mut(y)[2].fill(0.8);
        }
        assert_eq!(texture(&xyb), 0.0);
        // Thin chromatic lines on neutral ground: the pair-B class. Period 3
        // so the saturation stat's 4x4 subsample does not alias to zero.
        for y in 0..32 {
            let [_, _, b_row] = xyb.all_plane_rows_mut(y);
            for (x, b) in b_row.iter_mut().enumerate() {
                *b = if x % 3 == 0 { 0.6 } else { 0.5 };
            }
        }
        assert!(texture(&xyb) > super::PAIR_B_TEXTURE_THRESHOLD);
    }

    #[test]
    fn chroma_saturation_stat_measures_xyb_deviation() {
        let mut xyb = crate::image::Image3F::new(16, 16);
        for y in 0..16 {
            let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.0);
            y_row.fill(0.5);
            b_row.fill(0.5);
        }
        // Achromatic frames (X = 0, B = Y) read exactly zero.
        assert_eq!(super::chroma_saturation_stat(&xyb), 0.0);

        for y in 0..16 {
            let [x_row, _, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.02);
            b_row.fill(0.55);
        }
        // |X| + |B - Y| = 0.02 + 0.05 on every sampled pixel.
        let stat = super::chroma_saturation_stat(&xyb);
        assert!((stat - 0.07).abs() < 1e-5, "{stat}");
        assert!(stat >= super::SAT_QM_THRESHOLD);
    }

    #[test]
    fn x_gradient_stat_separates_red_structure_from_achromatic() {
        let ctx = crate::encoding_context::EncodingContext::default();
        // Achromatic: X flat at 0 → stat 0, far below the gate.
        let mut xyb = crate::image::Image3F::new(32, 32);
        for y in 0..32 {
            let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.0);
            for (x, v) in y_row.iter_mut().enumerate() {
                *v = (x % 7) as f32 * 0.1;
            }
            b_row.fill(0.5);
        }
        assert_eq!(super::chroma_gradient_stats(&ctx, &xyb, 0.0)[0], 0.0);

        // X carrying half of Y's gradient amplitude → stat 0.5, over the gate.
        for y in 0..32 {
            let [x_row, y_row, _] = xyb.all_plane_rows_mut(y);
            for (x, v) in x_row.iter_mut().enumerate() {
                *v = y_row[x] * 0.5;
            }
        }
        let stat = super::chroma_gradient_stats(&ctx, &xyb, 0.0)[0];
        assert!((stat - 0.5).abs() < 1e-5, "{stat}");
        assert!(stat >= super::X_QM_GRAD_THRESHOLD);
    }

    #[test]
    fn point_chroma_policy_preserves_precision_across_hq_boundaries() {
        use crate::{
            Speed, encoding_context::EncodingContext, quant_weights::DequantMatrices, xyb,
        };
        let content = super::ChromaContentStats {
            saturation: 0.03,
            dark_fraction: 0.75,
            point_fraction: 0.01,
            point_energy: 0.4,
            sharp_energy: 0.15,
        };
        for distance in [0.03, 0.15, 0.25, 0.295, 0.2989, 0.299, 0.2999, 0.3, 0.3001] {
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                xyb::XybMatrix::SPEC,
                distance,
                1,
            );
            let distp = compute_distance_params(distance);
            super::apply_point_chroma_policy(&ctx, &distp, content, false);
            assert_eq!(
                distp.x_qm_scale.max(ctx.x_qm_scale_floor()),
                3,
                "d={distance}"
            );
            assert_eq!(ctx.b_qm_scale(), 3, "d={distance}");
            assert!(std::ptr::eq(
                ctx.matrices(),
                DequantMatrices::new(0.3, crate::Arbiter::Ssimulacra2, false)
            ));
        }
        // Existing mid-band protection and the upper cutoff stay intact.
        for (distance, x_scale, b_scale) in [(1.25, 4, 3), (2.249, 4, 3), (2.25, 3, 2)] {
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                xyb::XybMatrix::SPEC,
                distance,
                1,
            );
            let distp = compute_distance_params(distance);
            let matrices = ctx.matrices();
            super::apply_point_chroma_policy(&ctx, &distp, content, false);
            assert_eq!(distp.x_qm_scale.max(ctx.x_qm_scale_floor()), x_scale);
            assert_eq!(ctx.b_qm_scale(), b_scale);
            assert!(std::ptr::eq(ctx.matrices(), matrices));
        }
        // Excluded content retains both its tables and scales.
        for (speed, achromatic, x_heavy, detected) in [
            (Speed::Fast, false, false, true),
            (Speed::Slow, true, false, true),
            (Speed::Slow, false, true, true),
            (Speed::Slow, false, false, false),
        ] {
            let ctx = EncodingContext::new(
                speed,
                crate::Arbiter::Ssimulacra2,
                xyb::XybMatrix::SPEC,
                0.295,
                1,
            );
            ctx.set_x_heavy(x_heavy);
            let matrices = ctx.matrices();
            let stats = if detected {
                content
            } else {
                Default::default()
            };
            super::apply_point_chroma_policy(
                &ctx,
                &compute_distance_params(0.295),
                stats,
                achromatic,
            );
            assert_eq!(ctx.x_qm_scale_floor(), 2);
            assert_eq!(ctx.b_qm_scale(), 2);
            assert!(std::ptr::eq(ctx.matrices(), matrices));
            assert!(ctx.point_chroma_hq_matrices().is_none());
        }
    }

    #[test]
    fn point_chroma_stats_separate_impulses_from_lines_and_hue_edges() {
        let mut stars = crate::image::Image3F::new(256, 256);
        for py in 0..256 {
            for px in 0..256 {
                let y = if px % 32 == 12 && py % 32 == 12 {
                    0.8
                } else {
                    0.12
                };
                stars.plane_row_mut(0, py)[px] = 0.001;
                stars.plane_row_mut(1, py)[px] = y;
                stars.plane_row_mut(2, py)[px] = y + 0.03;
            }
        }
        let stats = super::chroma_content_stats(&stars);
        assert!(stats.diffuse_point_chroma(), "{stats:?}");
        assert!(stats.point_energy > 0.7);

        // The same contrast and chroma in continuous lines must not qualify.
        for py in 0..256 {
            for px in 0..256 {
                let y = if px % 32 == 12 { 0.8 } else { 0.12 };
                stars.plane_row_mut(1, py)[px] = y;
                stars.plane_row_mut(2, py)[px] = y + 0.03;
            }
        }
        assert!(!super::chroma_content_stats(&stars).diffuse_point_chroma());
        for py in 0..256 {
            stars.plane_row_mut(1, py).fill(0.12);
            stars.plane_row_mut(2, py).fill(0.15);
            for px in 0..256 {
                stars.plane_row_mut(0, py)[px] = if (px + py) % 2 == 0 { 0.01 } else { -0.01 };
            }
        }
        assert!(!super::chroma_content_stats(&stars).diffuse_point_chroma());
    }

    #[test]
    fn point_chroma_stats_handle_small_and_achromatic_images() {
        for (w, h) in [(1, 1), (3, 17), (17, 3), (9, 9), (16, 16)] {
            let mut image = crate::image::Image3F::new(w, h);
            for py in 0..h {
                for px in 0..w {
                    let y = if (px + py) % 5 == 0 { 0.7 } else { 0.1 };
                    image.plane_row_mut(1, py)[px] = y;
                    image.plane_row_mut(2, py)[px] = y;
                }
            }
            let stats = super::chroma_content_stats(&image);
            assert_eq!(stats.saturation, 0.0);
            assert!(stats.point_fraction.is_finite() && stats.point_energy.is_finite());
            assert!(!stats.diffuse_point_chroma());
        }
    }

    #[test]
    fn selected_chroma_gradient_sums_match_scalar() {
        let selected = super::selected_chroma_gradient_sums_fn();
        for width in [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for height in [0, 1, 4, 5, 9] {
                let len = width * height;
                let x: Vec<f32> = (0..len)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03125)
                    .collect();
                let y: Vec<f32> = (0..len)
                    .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.0234375)
                    .collect();
                let b: Vec<f32> = (0..len)
                    .map(|i| ((i * 71 % 89) as f32 - 44.0) * 0.01953125)
                    .collect();
                let expected = super::chroma_gradient_sums_scalar(&x, &y, &b, width);
                let actual = selected(&x, &y, &b, width);
                for channel in 0..4 {
                    let tolerance = 1e-5 * expected[channel].abs().max(1.0);
                    assert!(
                        (actual[channel] - expected[channel]).abs() <= tolerance,
                        "channel {channel}, {width}x{height}: SIMD={}, scalar={}, tolerance={tolerance}",
                        actual[channel],
                        expected[channel]
                    );
                }
            }
        }
    }

    #[test]
    fn constant_block_fraction_counts_exactly_flat_blocks() {
        use super::constant_block_fraction;
        use crate::image::Image3F;
        let mut xyb = Image3F::new(32, 16);
        // Left half flat, right half a ramp: 4 of 8 blocks are constant.
        for c in 0..3 {
            for y in 0..16 {
                for x in 16..32 {
                    xyb.plane_row_mut(c, y)[x] = x as f32 * 0.01;
                }
            }
        }
        assert_eq!(constant_block_fraction(&xyb), 0.5);
        assert_eq!(constant_block_fraction(&Image3F::new(7, 7)), 0.0);
    }
}
