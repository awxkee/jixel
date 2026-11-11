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

//! Saturated pixels are free beyond the display range. The decoder clips RGB,
//! so a reconstruction that overshoots white at a white pixel, or undershoots
//! black at a black one, shows no error, while the coefficient-domain
//! distortion charges it in full. Before the forward transform, each such
//! pixel takes the value the block quantizer reconstructs there, as long as it
//! stays beyond the limit; a hard edge becomes a smooth wave whose lobes land
//! in the clipped region. B moves with Y along the gray axis, where CfL
//! carries the shift at no chroma cost.

use crate::encoding_context::EncodingContext;

const ITERATIONS: usize = 3;
/// Least share of the footprint at a limit for the pass to run.
const MIN_SATURATED_SHARE: f32 = 0.02;
/// Distance from the white or black Y within which a pixel is at that limit.
const Y_TOLERANCE: f32 = 0.002;
const MAX_COVERED_BLOCKS: usize = 16;

const NORMAL: u8 = 0;
const WHITE: u8 = 1;
const BLACK: u8 = 2;

#[derive(Default)]
pub(crate) struct ClipScratch {
    source: Vec<f32>,
    target: Vec<f32>,
    /// Target minus source per footprint pixel, valid after a `true` return.
    shift: Vec<f32>,
    coefficients: Vec<f32>,
    levels: Vec<i32>,
    pixels: Vec<f32>,
    class: Vec<u8>,
    shifted: Vec<f32>,
}

/// Moves the saturated pixels of one luma footprint beyond their limit.
/// `input` is the footprint's luma, `stride` wide, replicated past the image
/// edge. Returns whether any pixel
/// moved; the shift then applies to the Y and B inputs through
/// [`shifted_input`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn continue_saturated(
    ctx: &EncodingContext,
    raw_strategy: u8,
    cx: usize,
    cy: usize,
    inv_qm: &[f32],
    qm: &[f32],
    quant_ac: i32,
    scale: f32,
    distance: f32,
    input: &[f32],
    stride: usize,
    scratch: &mut ClipScratch,
) -> bool {
    let covered = cx * cy;
    if covered > MAX_COVERED_BLOCKS {
        return false;
    }
    let n = covered * 64;
    let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(raw_strategy);
    debug_assert_eq!(pw * ph, n);
    let (_, white, _) = crate::xyb::rgb_to_xyb_pixel_f32(&ctx.xyb, 1.0, 1.0, 1.0);
    let (_, black, _) = crate::xyb::rgb_to_xyb_pixel_f32(&ctx.xyb, 0.0, 0.0, 0.0);

    // Padding past the image edge arrives replicated from the edge pixel and
    // is classified like it: it is never shown, so where it goes does not
    // matter, and a view and a copy of the same stripe agree.
    scratch.class.clear();
    scratch.class.reserve(n);
    let mut saturated = 0;
    let (mut any_white, mut any_black, mut any_normal) = (false, false, false);
    for row in input.chunks(stride).take(ph) {
        scratch.class.extend(row[..pw].iter().map(|&v| {
            let class = if v >= white - Y_TOLERANCE {
                any_white = true;
                WHITE
            } else if v <= black + Y_TOLERANCE {
                any_black = true;
                BLACK
            } else {
                any_normal = true;
                NORMAL
            };
            saturated += usize::from(class != NORMAL);
            class
        }));
    }
    // A footprint at one limit only has nothing to constrain it; one with
    // both limits is pinned by the sign pattern and gains the most.
    let single_class = !any_normal && (any_white != any_black);
    if single_class || (saturated as f32) < MIN_SATURATED_SHARE * n as f32 {
        return false;
    }

    scratch.source.clear();
    scratch.source.reserve(n);
    for row in input.chunks(stride).take(ph) {
        scratch.source.extend_from_slice(&row[..pw]);
    }
    scratch.target.clone_from(&scratch.source);
    scratch.coefficients.resize(n, 0.0);
    scratch.levels.resize(n, 0);
    scratch.pixels.resize(n, 0.0);
    let bias = ctx.dequant_bias(1);
    let inv_qac = 1.0 / (scale * quant_ac as f32);
    let wc = cx * 8;
    for _ in 0..ITERATIONS {
        crate::inflated_cost::forward_for(raw_strategy, &scratch.target, &mut scratch.coefficients);
        (ctx.quantize_block_ac)(
            &scratch.coefficients,
            1,
            inv_qm,
            quant_ac,
            scale,
            1.0,
            distance,
            cx,
            cy,
            &mut scratch.levels,
        );
        // The cx*cy lowest frequencies ride on the finer DC path and stay exact.
        for (y, ((coefficients, levels), weights)) in scratch
            .coefficients
            .chunks_exact_mut(wc)
            .zip(scratch.levels.chunks_exact(wc))
            .zip(qm.chunks_exact(wc))
            .enumerate()
        {
            let skip = if y < cy { cx } else { 0 };
            for ((coefficient, &level), &weight) in
                coefficients.iter_mut().zip(levels).zip(weights).skip(skip)
            {
                *coefficient = bias.level(level) * weight * inv_qac;
            }
        }
        crate::inflated_cost::reconstruct_error(
            ctx.idct,
            raw_strategy,
            &scratch.coefficients,
            &mut scratch.pixels,
        );
        // Normal pixels retain the source copied above.
        for ((target, &recon), &class) in scratch
            .target
            .iter_mut()
            .zip(&scratch.pixels)
            .zip(&scratch.class)
        {
            match class {
                WHITE => *target = recon.max(white),
                BLACK => *target = recon.min(black),
                _ => {}
            }
        }
    }
    scratch.shift.clear();
    scratch.shift.extend(
        scratch
            .target
            .iter()
            .zip(&scratch.source)
            .map(|(t, s)| t - s),
    );
    scratch.shift.iter().any(|&d| d != 0.0)
}

/// `input` (strided) plus the stored shift, contiguous with width `pw`.
pub(crate) fn shifted_input<'a>(
    scratch: &'a mut ClipScratch,
    input: &[f32],
    stride: usize,
    pw: usize,
    ph: usize,
) -> &'a [f32] {
    scratch.shifted.clear();
    for y in 0..ph {
        let row = &input[y * stride..y * stride + pw];
        let shift = &scratch.shift[y * pw..y * pw + pw];
        scratch
            .shifted
            .extend(row.iter().zip(shift).map(|(a, b)| a + b));
    }
    &scratch.shifted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Speed;
    use crate::dc_group_data::STRATEGY_DCT;
    use crate::image::Image3F;
    use crate::xyb::{XybMatrix, rgb_to_xyb_pixel_f32};

    fn context() -> EncodingContext {
        EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            2.0,
            1,
        )
    }

    /// 8x8 luma footprint: `left` columns of one gray level, the rest another.
    fn image(left_value: f32, right_value: f32, left: usize) -> Image3F {
        let mut xyb = Image3F::new(8, 8);
        for y in 0..8 {
            let row = xyb.plane_row_mut(1, y);
            for (x, v) in row.iter_mut().enumerate() {
                *v = if x < left { left_value } else { right_value };
            }
        }
        xyb
    }

    fn run(ctx: &EncodingContext, xyb: &Image3F, scratch: &mut ClipScratch) -> bool {
        let matrices = ctx.matrices();
        let plane = xyb.plane(1);
        continue_saturated(
            ctx,
            STRATEGY_DCT,
            1,
            1,
            &matrices.inv_matrix(1)[..],
            &matrices.matrix(1)[..],
            8,
            1.0,
            2.0,
            plane.as_slice(),
            plane.xsize(),
            scratch,
        )
    }

    #[test]
    fn saturated_pixels_only_move_beyond_their_limit() {
        let ctx = context();
        let (_, white, _) = rgb_to_xyb_pixel_f32(&ctx.xyb, 1.0, 1.0, 1.0);
        let (_, black, _) = rgb_to_xyb_pixel_f32(&ctx.xyb, 0.0, 0.0, 0.0);
        let xyb = image(black, white, 3);
        let mut scratch = ClipScratch::default();
        assert!(run(&ctx, &xyb, &mut scratch));
        for y in 0..8 {
            for x in 0..8 {
                let shift = scratch.shift[y * 8 + x];
                if x < 3 {
                    assert!(shift <= 0.0, "black pixel moved up at {x},{y}");
                } else {
                    assert!(shift >= 0.0, "white pixel moved down at {x},{y}");
                }
            }
        }
        let shifted = shifted_input(&mut scratch, xyb.plane(1).as_slice(), 8, 8, 8).to_vec();
        assert!(
            shifted
                .iter()
                .zip(xyb.plane(1).as_slice())
                .any(|(a, b)| a != b)
        );
        // A pinned column in the middle stays where the source put it.
        let mut xyb = image(black, white, 3);
        for y in 0..8 {
            xyb.plane_row_mut(1, y)[3] = 0.5 * (black + white);
        }
        let mut scratch = ClipScratch::default();
        assert!(run(&ctx, &xyb, &mut scratch));
        for y in 0..8 {
            assert_eq!(scratch.shift[y * 8 + 3], 0.0);
        }
    }

    #[test]
    fn unsaturated_and_single_limit_footprints_are_untouched() {
        let ctx = context();
        let (_, white, _) = rgb_to_xyb_pixel_f32(&ctx.xyb, 1.0, 1.0, 1.0);
        let mut scratch = ClipScratch::default();
        assert!(!run(&ctx, &image(0.2, 0.6, 4), &mut scratch));
        assert!(!run(&ctx, &image(white, white, 4), &mut scratch));
    }
}
