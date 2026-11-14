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

//! Native-transform costs for bounded edge-layer proposals.
//!
//! The real AC-group writer quantizes complete selected transforms, including
//! native thresholds, CfL against dequantized Y, matrix scales, quantizer bias
//! and the normal clipping-continuation policy. Its first-pass reconstruction
//! capture is completed with the scalar DC quantization error. No final Slow
//! tools are disabled: learned-price RDOQ, later DC choices and loop filters
//! remain downstream decisions, so this is a proposal model, not an exact
//! prediction of the final encoded frame.
//!
//! Distortion is weighted visible-pixel SSE. Rate uses the existing fixed
//! coefficient-rate terms on the writer's actual tokens, plus an independent
//! DC-magnitude estimate. It is not the final frame's marginal entropy cost.

use crate::ac_context::{K_NON_ZERO_BUCKETS, K_NUM_FINE_BLOCK_CTXS};
use crate::coder_scratch::CoderScratch;
use crate::coeff_order::CoeffOrders;
use crate::color_correlation::{dc_cfl_factor, dc_cfl_factor_x};
use crate::dc_group_data::{AcStrategyImage, DcGroupData, NUM_STRATEGIES, STRATEGY_DCT};
use crate::encoding_context::EncodingContext;
use crate::entropy::{Token, f_log2};
use crate::group::{SourceDc, dc_from_coefficients, write_ac_group};
use crate::image::{Image3B, Image3F, Image3S, Plane, Rect};
use crate::quant_weights::INV_DC_QUANT;
use std::sync::OnceLock;

/// The actual quantization parameters used for the current native layout.
/// Global DC slope searches have not run when edge proposals are selected;
/// their extra X/B slope coefficients are therefore zero in this model.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Quantization {
    pub(crate) scale: f32,
    pub(crate) scale_dc: f32,
    pub(crate) x_qm_scale: u32,
    pub(crate) dc_step: [f32; 3],
    pub(crate) distance: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct NativeCost {
    pub(crate) distortion: f64,
    pub(crate) rate: f64,
    pub(crate) ac_rate: f64,
    pub(crate) dc_rate: f64,
    pub(crate) transforms: usize,
    pub(crate) pixels: usize,
}

impl NativeCost {
    pub(crate) fn is_finite(self) -> bool {
        self.distortion.is_finite() && self.rate.is_finite()
    }

    fn invalid() -> Self {
        Self {
            distortion: f64::INFINITY,
            rate: f64::INFINITY,
            ..Self::default()
        }
    }
}

struct NativeBlock {
    group: usize,
    bx: usize,
    by: usize,
    x: usize,
    y: usize,
    cov_x: usize,
    cov_y: usize,
    strategy: u8,
}

/// Fixed metadata from one native setup pass. The caller scores the current
/// residual, temporarily draws a proposal into its touched tiles, scores it
/// again, then restores those tiles. Any accepted change requires a fresh
/// native setup before the final frame is encoded.
pub(crate) struct NativeModel<'a> {
    ctx: &'a EncodingContext,
    quantization: Quantization,
    groups: &'a [DcGroupData],
    width: usize,
    height: usize,
    owners: Vec<usize>,
    blocks: Vec<NativeBlock>,
    /// Per-pixel weight of the reconstruction error (1 when `None`).
    visibility: Option<&'a [f32]>,
    orders: CoeffOrders,
    qf_threshold: u32,
}

impl<'a> NativeModel<'a> {
    pub(crate) fn new(
        ctx: &'a EncodingContext,
        quantization: Quantization,
        groups: &'a [DcGroupData],
        width: usize,
        height: usize,
        visibility: Option<&'a [f32]>,
    ) -> Self {
        assert!(width > 0 && height > 0);
        assert!(visibility.is_none_or(|v| v.len() == width * height));
        assert!(quantization.scale.is_finite() && quantization.scale > 0.0);
        assert!(quantization.scale_dc.is_finite() && quantization.scale_dc > 0.0);
        assert!(
            quantization
                .dc_step
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        );
        assert!(quantization.distance.is_finite() && quantization.distance > 0.0);
        let groups_x = width.div_ceil(crate::frame::K_DC_GROUP_DIM);
        assert_eq!(
            groups.len(),
            groups_x * height.div_ceil(crate::frame::K_DC_GROUP_DIM)
        );
        let (blocks_x, blocks_y) = (width.div_ceil(8), height.div_ceil(8));
        let mut owners = vec![usize::MAX; blocks_x * blocks_y];
        let mut blocks = Vec::new();
        let mut histogram = [0u64; 256];
        for (group, data) in groups.iter().enumerate() {
            let ox = group % groups_x * (crate::frame::K_DC_GROUP_DIM / 8);
            let oy = group / groups_x * (crate::frame::K_DC_GROUP_DIM / 8);
            for &value in data.raw_quant_field.as_slice() {
                histogram[value as usize] += 1;
            }
            for (bx, by, strategy) in data.ac_strategy.iter_first_blocks() {
                let cov_x = AcStrategyImage::covered_blocks_x_of(strategy);
                let cov_y = AcStrategyImage::covered_blocks_y_of(strategy);
                assert!(
                    bx + cov_x <= data.ac_strategy.xsize()
                        && by + cov_y <= data.ac_strategy.ysize()
                );
                let index = blocks.len();
                for y in oy + by..oy + by + cov_y {
                    for x in ox + bx..ox + bx + cov_x {
                        assert!(x < blocks_x && y < blocks_y);
                        let owner = &mut owners[y * blocks_x + x];
                        assert_eq!(*owner, usize::MAX, "overlapping native transforms");
                        *owner = index;
                    }
                }
                blocks.push(NativeBlock {
                    group,
                    bx,
                    by,
                    x: (ox + bx) * 8,
                    y: (oy + by) * 8,
                    cov_x,
                    cov_y,
                    strategy,
                });
            }
        }
        assert!(
            owners.iter().all(|&owner| owner != usize::MAX),
            "native layout has holes"
        );
        let total: u64 = histogram.iter().sum();
        let mut sum = 0;
        let qf_threshold = histogram
            .iter()
            .position(|&count| {
                sum += count;
                sum * 2 >= total
            })
            .unwrap_or(1) as u32;
        Self {
            ctx,
            quantization,
            groups,
            width,
            height,
            owners,
            blocks,
            visibility,
            orders: CoeffOrders::natural(),
            qf_threshold,
        }
    }

    pub(crate) fn num_transforms(&self) -> usize {
        self.blocks.len()
    }

    /// Native transforms owning the touched 8x8 cells, ascending.
    pub(crate) fn transforms(&self, affected_cells: &[usize]) -> Vec<usize> {
        let mut selected: Vec<usize> = affected_cells
            .iter()
            .map(|&cell| {
                *self
                    .owners
                    .get(cell)
                    .expect("edge support outside the native frame")
            })
            .collect();
        selected.sort_unstable();
        selected.dedup();
        selected
    }

    /// Each transform's cost on `image`, in the given order. A transform reads
    /// only its own pixels, so they are scored in parallel.
    pub(crate) fn score_transforms(
        &self,
        scratch: &mut CoderScratch,
        image: &Image3F,
        transforms: &[usize],
    ) -> Vec<NativeCost> {
        assert_eq!((image.xsize(), image.ysize()), (self.width, self.height));
        match transforms {
            [] => Vec::new(),
            [index] => vec![self.score_block(scratch, image, &self.blocks[*index])],
            _ => self
                .ctx
                .thread_pool
                .steal_map(scratch, transforms.len(), |i, scratch| {
                    self.score_block(scratch, image, &self.blocks[transforms[i]])
                }),
        }
    }

    /// Sum of transform costs in order; any invalid one invalidates the total.
    pub(crate) fn total(costs: impl IntoIterator<Item = NativeCost>) -> NativeCost {
        let mut total = NativeCost::default();
        for cost in costs {
            if !cost.is_finite() {
                return NativeCost::invalid();
            }
            total.distortion += cost.distortion;
            total.rate += cost.rate;
            total.ac_rate += cost.ac_rate;
            total.dc_rate += cost.dc_rate;
            total.transforms += 1;
            total.pixels += cost.pixels;
        }
        if !total.is_finite() {
            NativeCost::invalid()
        } else {
            total
        }
    }

    /// Expands touched 8x8 cells to complete native transforms and scores each
    /// transform once. Buffers are bounded by one transform; no frame clone or
    /// candidate image is allocated here. Empty support has zero cost.
    #[cfg(test)]
    pub(crate) fn score(
        &self,
        scratch: &mut CoderScratch,
        image: &Image3F,
        affected_cells: &[usize],
    ) -> NativeCost {
        let transforms = self.transforms(affected_cells);
        Self::total(self.score_transforms(scratch, image, &transforms))
    }

    fn score_block(
        &self,
        scratch: &mut CoderScratch,
        image: &Image3F,
        block: &NativeBlock,
    ) -> NativeCost {
        let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(block.strategy);
        let (visible_width, visible_height) =
            (pw.min(self.width - block.x), ph.min(self.height - block.y));
        let mut reconstruction = Image3F::new(pw, ph);
        let mut quant_dc = Image3S::new(block.cov_x, block.cov_y);
        let mut source_dc = SourceDc {
            x: Plane::new(block.cov_x, block.cov_y),
            y: Plane::new(block.cov_x, block.cov_y),
            b: Plane::new(block.cov_x, block.cov_y),
        };
        let mut nzeros = [Image3B::new(32, 32)];
        let mut tokens = [Vec::new()];
        let p = self.quantization;
        write_ac_group(
            self.ctx,
            &mut scratch.ac_group,
            image,
            (block.x, block.y),
            Rect::new(block.bx, block.by, block.cov_x, block.cov_y),
            p.scale,
            p.scale_dc,
            p.dc_step,
            p.distance,
            p.x_qm_scale,
            &self.groups[block.group],
            0,
            &mut quant_dc,
            Some(&mut source_dc),
            block.bx,
            block.by,
            &mut nzeros,
            &[0],
            None,
            &self.orders,
            &[],
            None,
            false,
            self.qf_threshold,
            Some((&mut reconstruction, 0, None)),
            None,
            &mut tokens,
        );
        let ac_rate = token_rate(&tokens[0]);
        let dc_rate = (0..3)
            .map(|c| {
                quant_dc
                    .plane_data(c)
                    .iter()
                    .map(|&v| 0.5 + f_log2(1.0 + f64::from(v).abs()))
                    .sum::<f64>()
            })
            .sum();
        let dc_steps: [f32; 3] =
            std::array::from_fn(|c| p.dc_step[c] / (INV_DC_QUANT[c] * p.scale_dc));
        let cfl = self.ctx.cfl_frame();
        let factors = [
            dc_cfl_factor_x(p.dc_step, cfl.base_x),
            dc_cfl_factor(p.dc_step, 0, cfl),
        ];
        let mut decoded_dc = Image3F::new(block.cov_x, block.cov_y);
        for y in 0..block.cov_y {
            (self.ctx.dc_rows.reconstruct)(
                std::array::from_fn(|c| quant_dc.plane_row(c, y)),
                factors,
                dc_steps,
                decoded_dc.all_plane_rows_mut(y),
            );
        }
        let count = block.cov_x * block.cov_y;
        let cx = block.cov_x.max(block.cov_y);
        let source = [&source_dc.x, &source_dc.y, &source_dc.b];
        let Some(inverse) = llf_from_dc(self.ctx, block.strategy) else {
            return NativeCost::invalid();
        };
        let mut coefficient_error = vec![0.0f32; pw * ph];
        let mut pixel_error = vec![0.0f32; pw * ph];
        let mut delta = vec![0.0f32; count];
        let mut distortion = 0.0f64;
        for c in 0..3 {
            for iy in 0..block.cov_y {
                for ix in 0..block.cov_x {
                    delta[iy * block.cov_x + ix] =
                        decoded_dc.plane_row(c, iy)[ix] - source[c].row(iy)[ix];
                }
            }
            if block.strategy == STRATEGY_DCT {
                pixel_error.fill(delta[0]);
            } else {
                coefficient_error.fill(0.0);
                for j in 0..count {
                    let value: f32 = inverse[j * count..(j + 1) * count]
                        .iter()
                        .zip(&delta)
                        .map(|(&a, &b)| a * b)
                        .sum();
                    coefficient_error[j / cx * cx * 8 + j % cx] = value;
                }
                crate::inflated_cost::reconstruct_error(
                    self.ctx.idct,
                    block.strategy,
                    &coefficient_error,
                    &mut pixel_error,
                );
            }
            let weight = f64::from(self.ctx.channel_weight(c));
            for y in 0..visible_height {
                for x in 0..visible_width {
                    let decoded = reconstruction.plane_row(c, y)[x] + pixel_error[y * pw + x];
                    let error = f64::from(decoded)
                        - f64::from(image.plane_row(c, block.y + y)[block.x + x]);
                    match self.visibility {
                        None => distortion += weight * error * error,
                        Some(v) => {
                            let shown = f64::from(v[(block.y + y) * self.width + block.x + x]);
                            distortion += shown * (weight * error * error);
                        }
                    }
                }
            }
        }
        NativeCost {
            distortion,
            rate: ac_rate + dc_rate,
            ac_rate,
            dc_rate,
            transforms: 1,
            pixels: visible_width * visible_height,
        }
    }
}

/// Existing coefficient-rate constants, applied to the native writer's actual
/// first-pass tokens. Nzero headers occupy the first fine-context range;
/// remaining tokens carry signed coefficient levels, including visited zeros.
fn token_rate(tokens: &[Token]) -> f64 {
    let nzero_contexts = (K_NON_ZERO_BUCKETS * K_NUM_FINE_BLOCK_CTXS) as u32;
    tokens
        .iter()
        .map(|token| {
            if token.context < nzero_contexts {
                0.4 * f_log2(1.0 + f64::from(token.value))
            } else if token.value == 0 {
                0.5
            } else {
                let magnitude = u64::from(token.value).div_ceil(2) as f64;
                1.6 + f_log2(1.0 + magnitude)
            }
        })
        .sum()
}

/// Inverse of the real DC-from-lowest-frequencies transform. This mirrors the
/// DC reconstruction capture used by EPF, with one bounded matrix per strategy.
fn llf_from_dc(ctx: &EncodingContext, strategy: u8) -> Option<&'static [f32]> {
    static TABLES: [OnceLock<Option<Vec<f32>>>; NUM_STRATEGIES] =
        [const { OnceLock::new() }; NUM_STRATEGIES];
    TABLES[strategy as usize]
        .get_or_init(|| {
            let add_scaled = ctx.spline_kernels.add_scaled_fma_f32;
            let (cov_x, cov_y) = (
                AcStrategyImage::covered_blocks_x_of(strategy),
                AcStrategyImage::covered_blocks_y_of(strategy),
            );
            let (cx, count) = (cov_x.max(cov_y), cov_x * cov_y);
            let mut coefficients = vec![0.0; count * 64];
            let mut dc = vec![0.0; count];
            let mut matrix = vec![0.0; count * count];
            for j in 0..count {
                let index = j / cx * cx * 8 + j % cx;
                coefficients[index] = 1.0;
                dc_from_coefficients(ctx, strategy, &coefficients, &mut dc);
                coefficients[index] = 0.0;
                for k in 0..count {
                    matrix[k * count + j] = dc[k];
                }
            }
            let mut inverse = vec![0.0f32; count * count];
            for i in 0..count {
                inverse[i * count + i] = 1.0;
            }
            for col in 0..count {
                let pivot = (col..count)
                    .max_by(|&a, &b| {
                        matrix[a * count + col]
                            .abs()
                            .total_cmp(&matrix[b * count + col].abs())
                    })
                    .unwrap();
                if !matrix[pivot * count + col].is_finite()
                    || matrix[pivot * count + col].abs() < 1e-20
                {
                    return None;
                }
                for k in 0..count {
                    matrix.swap(col * count + k, pivot * count + k);
                    inverse.swap(col * count + k, pivot * count + k);
                }
                let divisor = matrix[col * count + col].recip();
                let matrix_row = matrix.chunks_exact_mut(count).nth(col).unwrap();
                let inverse_row = inverse.chunks_exact_mut(count).nth(col).unwrap();

                for (m, inv) in matrix_row.iter_mut().zip(inverse_row) {
                    *m *= divisor;
                    *inv *= divisor;
                }
                let (matrix_before, rest) = matrix.split_at_mut(col * count);
                let (pivot, matrix_after) = rest.split_at_mut(count);

                let (inverse_before, rest) = inverse.split_at_mut(col * count);
                let (pivot_inverse, inverse_after) = rest.split_at_mut(count);

                let matrix_rows = matrix_before
                    .chunks_exact_mut(count)
                    .chain(matrix_after.chunks_exact_mut(count));

                let inverse_rows = inverse_before
                    .chunks_exact_mut(count)
                    .chain(inverse_after.chunks_exact_mut(count));

                for (row, inverse_row) in matrix_rows.zip(inverse_rows) {
                    let factor = row[col];
                    if factor == 0.0 {
                        continue;
                    }

                    add_scaled(row, pivot, -factor);
                    add_scaled(inverse_row, pivot_inverse, -factor);
                }
            }
            Some(inverse)
        })
        .as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantization() -> Quantization {
        Quantization {
            scale: 0.2,
            scale_dc: 6.0,
            x_qm_scale: 3,
            dc_step: [1.0; 3],
            distance: 2.0,
        }
    }

    fn image(width: usize, height: usize) -> Image3F {
        let mut image = Image3F::new(width, height);
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    image.plane_row_mut(c, y)[x] =
                        0.05 * c as f32 + 0.002 * ((x * 7 + y * 3 + c) % 31) as f32;
                }
            }
        }
        image
    }

    #[test]
    fn touched_cells_expand_to_whole_native_transform_once() {
        let ctx = EncodingContext::default();
        let mut group = DcGroupData::new(8, 8).unwrap();
        group.raw_quant_field.as_mut_slice().fill(8);
        group
            .ac_strategy
            .set_first(0, 0, crate::dc_group_data::STRATEGY_DCT32X32);
        let groups = [group];
        let model = NativeModel::new(&ctx, quantization(), &groups, 64, 64, None);
        let image = image(64, 64);
        let mut scratch = CoderScratch::default();
        let one = model.score(&mut scratch, &image, &[0]);
        let repeated = model.score(&mut scratch, &image, &[0, 1, 9, 0]);
        assert_eq!(one, repeated);
        assert_eq!(one.transforms, 1);
        assert_eq!(one.pixels, 32 * 32);
        assert!(one.is_finite());
        assert_eq!(
            model.score(&mut scratch, &image, &[]),
            NativeCost::default()
        );
    }

    #[test]
    fn actual_group_quantizer_scores_every_native_strategy() {
        let ctx = EncodingContext::default();
        let mut scratch = CoderScratch::default();
        for strategy in 0..NUM_STRATEGIES as u8 {
            #[cfg(not(feature = "extraslow"))]
            if matches!(
                strategy,
                crate::dc_group_data::STRATEGY_DCT128X128
                    | crate::dc_group_data::STRATEGY_DCT128X64
                    | crate::dc_group_data::STRATEGY_DCT64X128
            ) {
                continue;
            }
            let (w, h) = crate::inflated_cost::strategy_pixel_dims(strategy);
            let mut group = DcGroupData::new(w / 8, h / 8).unwrap();
            group.ac_strategy.set_first(0, 0, strategy);
            group.raw_quant_field.as_mut_slice().fill(8);
            let groups = [group];
            let model = NativeModel::new(&ctx, quantization(), &groups, w, h, None);
            let cost = model.score(&mut scratch, &image(w, h), &[0]);
            assert!(cost.is_finite(), "native strategy {strategy}");
            assert_eq!(cost.transforms, 1);
            assert_eq!(cost.pixels, w * h);
        }
    }

    #[test]
    fn partial_footprints_and_dc_group_offsets_use_only_visible_pixels() {
        let ctx = EncodingContext::default();
        let mut groups = [
            DcGroupData::new(256, 3).unwrap(),
            DcGroupData::new(1, 3).unwrap(),
        ];
        for group in &mut groups {
            group.raw_quant_field.as_mut_slice().fill(8);
        }
        let model = NativeModel::new(&ctx, quantization(), &groups, 2049, 17, None);
        let mut scratch = CoderScratch::default();
        let image = Image3F::new(2049, 17);
        let cost = model.score(&mut scratch, &image, &[2 * 257 + 256]);
        assert_eq!(cost.transforms, 1);
        assert_eq!(cost.pixels, 1);
        assert_eq!(cost.distortion, 0.0);
        assert!(cost.is_finite());
    }

    #[test]
    fn dc_projection_inverts_the_real_extractor_for_every_strategy() {
        let ctx = EncodingContext::default();
        for strategy in 0..NUM_STRATEGIES as u8 {
            #[cfg(not(feature = "extraslow"))]
            if matches!(
                strategy,
                crate::dc_group_data::STRATEGY_DCT128X128
                    | crate::dc_group_data::STRATEGY_DCT128X64
                    | crate::dc_group_data::STRATEGY_DCT64X128
            ) {
                continue;
            }
            let cov_x = AcStrategyImage::covered_blocks_x_of(strategy);
            let cov_y = AcStrategyImage::covered_blocks_y_of(strategy);
            let (cx, n) = (cov_x.max(cov_y), cov_x * cov_y);
            let inverse = llf_from_dc(&ctx, strategy).unwrap();
            let expected: Vec<f32> = (0..n)
                .map(|i| ((i * 17 % 23) as f32 - 11.0) * 0.001)
                .collect();
            let mut coefficients = vec![0.0; n * 64];
            for j in 0..n {
                coefficients[j / cx * cx * 8 + j % cx] = inverse[j * n..(j + 1) * n]
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
            let mut actual = vec![0.0; n];
            dc_from_coefficients(&ctx, strategy, &coefficients, &mut actual);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 1e-5, "DC projection strategy {strategy}: {error}");
        }
    }
}
