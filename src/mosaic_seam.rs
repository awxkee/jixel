/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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
//! Adaptive transform (AC strategy) selection via rate-distortion optimization.

//! Boundary-error scoring for cached fine-mosaic assignments.

use crate::dct::fmla;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

pub(crate) type MosaicSeamStatsFn = fn(
    &EncodingContext,
    &Image3F,
    usize,
    usize,
    usize,
    usize,
    f32,
    &[&[[f32; 64]; 3]],
) -> (f32, f32);

pub(crate) fn select_mosaic_seam_stats_fn() -> MosaicSeamStatsFn {
    mosaic_seam_stats_scalar
}

/// Boundary-error statistics of one candidate mosaic assignment, computed
/// straight from the cached per-child spatial error planes. Only the seams
/// between adjacent children exist in a mosaic's 8-grid, so this visits the
/// same pixels, in the same order, as an assembled-region scan, without
/// constructing that larger error plane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mosaic_seam_stats_scalar(
    ctx: &EncodingContext,
    opsin: &Image3F,
    px: usize,
    py: usize,
    cxb: usize,
    cyb: usize,
    distance: f32,
    selected: &[&[[f32; 64]; 3]],
) -> (f32, f32) {
    debug_assert_eq!(selected.len(), cxb * cyb);
    let coarse_mix = ((distance - 1.9) / 0.1).clamp(0.0, 1.0);
    let floor = distance * fmla(coarse_mix, 0.0045 - 0.0015, 0.0015);
    let mut energy = 0.0f32;
    let mut peak = 0.0f32;
    for c in 0..3 {
        let plane = opsin.plane(c);
        let weight = ctx.channel_weight(c);
        let mut channel = 0.0f32;
        for sx in 1..cxb {
            for ky in 0..cyb {
                let left = &selected[ky * cxb + sx - 1][c];
                let right = &selected[ky * cxb + sx][c];
                let left_x = (px + sx * 8 - 1).min(plane.xsize() - 1);
                let right_x = (px + sx * 8).min(plane.xsize() - 1);
                for (y, (left, right)) in left
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(right.as_chunks::<8>().0)
                    .enumerate()
                {
                    let sy = (py + ky * 8 + y).min(plane.ysize() - 1);
                    let source_gradient = (plane.row(sy)[right_x] - plane.row(sy)[left_x]).abs();
                    let excess =
                        ((right[0] - left[7]).abs() - 0.5 * source_gradient - floor).max(0.0);
                    channel = fmla(excess, excess, channel);
                    peak = peak.max(weight * excess * excess);
                }
            }
        }
        for sy_b in 1..cyb {
            for kx in 0..cxb {
                let top = &selected[(sy_b - 1) * cxb + kx][c];
                let bottom = &selected[sy_b * cxb + kx][c];
                let top_y = (py + sy_b * 8 - 1).min(plane.ysize() - 1);
                let bottom_y = (py + sy_b * 8).min(plane.ysize() - 1);
                // Fixed array indexing lets the compiler prove these eight
                // accesses in bounds; a zipped pixel iterator was slower here.
                for x in 0..8 {
                    let sx_abs = (px + kx * 8 + x).min(plane.xsize() - 1);
                    let source_gradient =
                        (plane.row(bottom_y)[sx_abs] - plane.row(top_y)[sx_abs]).abs();
                    let excess =
                        ((bottom[x] - top[7 * 8 + x]).abs() - 0.5 * source_gradient - floor)
                            .max(0.0);
                    channel = fmla(excess, excess, channel);
                    peak = peak.max(weight * excess * excess);
                }
            }
        }
        energy += weight * channel;
    }
    (energy, peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_nonfinite_excess_handling() {
        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            None,
            crate::xyb::XybMatrix::SPEC,
            1.0,
            1,
        );
        let opsin = Image3F::new(16, 16);
        for value in [
            0.0,
            -0.0,
            0.0015,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let left = [[0.0; 64]; 3];
            let right = [[value; 64]; 3];
            for (cxb, cyb) in [(2, 1), (1, 2)] {
                let selected = [&left, &right];
                let expected = if value.is_infinite() {
                    (f32::INFINITY, f32::INFINITY)
                } else {
                    (0.0, 0.0)
                };
                let got = mosaic_seam_stats_scalar(&ctx, &opsin, 0, 0, cxb, cyb, 1.0, &selected);
                assert_eq!(got, expected, "{value}");
            }
        }
    }
}
