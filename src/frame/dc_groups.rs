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

//! Set up DC-group quant fields and transform layouts, and assemble DC planes.

use super::metadata::meta_entropy_cost;
use super::quantization::DistanceParams;
use super::{ImageDim, K_BLOCK_DIM, K_DC_GROUP_DIM, K_GROUP_DIM, K_GROUP_DIM_IN_BLOCKS};
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::{DcGroupData, STRATEGY_DCT};
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, Image3S};
use crate::util::EncodeError;

/// The frame's unrounded per-block XYB DC (the DC frame's input), stitched
/// from the DC groups.
pub(super) fn assemble_source_dc(
    dc_datas: &[DcGroupData],
    group_coords: &[(usize, usize)],
    dim: &ImageDim,
) -> Image3F {
    let mut image = Image3F::new(dim.xsize_blocks, dim.ysize_blocks);
    for (dc, &(gx, gy)) in dc_datas.iter().zip(group_coords) {
        let ox = gx * (K_DC_GROUP_DIM / K_BLOCK_DIM);
        let oy = gy * (K_DC_GROUP_DIM / K_BLOCK_DIM);
        for (c, plane) in [&dc.source_dc_x, &dc.source_dc_y, &dc.source_dc_b]
            .into_iter()
            .enumerate()
        {
            let plane = plane.as_ref().expect("source DC kept for the DC frame");
            for y in 0..plane.ysize() {
                image.plane_row_mut(c, oy + y)[ox..ox + plane.xsize()]
                    .copy_from_slice(plane.row(y));
            }
        }
    }
    image
}

#[allow(clippy::too_many_arguments)]
/// Set up one DC group (quant field, AC strategy, CfL, sub-8 gate); its AC
/// groups are encoded separately. Returns the data and its group-grid dims.
pub(super) fn setup_dc_group(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    dc_gx: usize,
    dc_gy: usize,
    num_threads: usize,
) -> Result<(DcGroupData, usize, usize), EncodeError> {
    // DC group rect in pixels (clamped to image bounds).
    let dc_group_x0 = dc_gx * K_DC_GROUP_DIM;
    let dc_group_y0 = dc_gy * K_DC_GROUP_DIM;
    let dc_group_xsize = K_DC_GROUP_DIM.min(dim.xsize.saturating_sub(dc_group_x0));
    let dc_group_ysize = K_DC_GROUP_DIM.min(dim.ysize.saturating_sub(dc_group_y0));
    let dc_group_xsize_blocks = dc_group_xsize.div_ceil(K_BLOCK_DIM);
    let dc_group_ysize_blocks = dc_group_ysize.div_ceil(K_BLOCK_DIM);
    let dc_group_xsize_groups = dc_group_xsize.div_ceil(K_GROUP_DIM);
    let dc_group_ysize_groups = dc_group_ysize.div_ceil(K_GROUP_DIM);

    let mut dc_data = DcGroupData::new(dc_group_xsize_blocks, dc_group_ysize_blocks)?;

    (ctx.fill_quant_field)(
        &mut scratch.aq_map,
        ctx.aq_source.get().unwrap_or(opsin),
        &mut dc_data.raw_quant_field,
        dc_group_x0,
        dc_group_y0,
        distp.distance,
        1.0 / distp.scale,
    );

    if ctx.speed.effort().chroma_policy && (ctx.x_heavy() || ctx.b_heavy()) {
        crate::adaptive_quant::apply_chroma_hf_protection(
            opsin,
            &mut dc_data.raw_quant_field,
            dc_group_x0,
            dc_group_y0,
            distp.distance,
        );
    }

    // Compute the per-tile CfL slopes before strategy selection so candidate
    // costs use the same Y-to-X/Y-to-B subtraction as final coefficient coding.
    crate::color_correlation::fill_cmap(
        ctx,
        opsin,
        dc_group_x0 / K_BLOCK_DIM,
        dc_group_y0 / K_BLOCK_DIM,
        dc_group_xsize_blocks,
        dc_group_ysize_blocks,
        &dc_data.raw_quant_field,
        distp.scale,
        &mut scratch.cfl_rdo,
        &mut dc_data.ytox_map,
        &mut dc_data.ytob_map,
        distp.distance,
        distp.x_qm_scale,
    );
    let mut fine_rollbacks = Vec::new();
    let mut fine_fallbacks = Vec::new();
    dc_data.block_quant_field = Some(dc_data.raw_quant_field.clone());
    dc_data.sub8_benefit = crate::ac_strategy::fill_ac_strategy(
        ctx,
        scratch,
        opsin,
        dc_group_x0,
        dc_group_y0,
        distp.distance,
        distp.scale,
        distp.x_qm_scale,
        &mut dc_data.raw_quant_field,
        &dc_data.ytox_map,
        &dc_data.ytob_map,
        &mut dc_data.ac_strategy,
        &mut fine_rollbacks,
        &mut fine_fallbacks,
        &mut dc_data.dp_regions,
        num_threads,
    );
    // A fine mosaic that split an otherwise-retained merge must repay its own
    // exact metadata delta. Keep this separate from the legacy sub-8 gate so
    // unrelated fine-block gains cannot subsidize a weak mosaic decision.
    let fine_accepted = if fine_rollbacks.is_empty() {
        false
    } else {
        let fine_benefit: f32 = fine_rollbacks.iter().map(|r| r.benefit).sum();
        let cost_with = meta_entropy_cost(&dc_data, scratch, distp.distance);
        let mut fine_quants = Vec::with_capacity(fine_rollbacks.len());
        for rollback in &fine_rollbacks {
            let mut quants = [0u8; 16];
            for iy in 0..rollback.cov_y {
                let row = dc_data.raw_quant_field.row_mut(rollback.by + iy);
                let cells = &mut row[rollback.bx..rollback.bx + rollback.cov_x];
                quants[iy * 4..iy * 4 + rollback.cov_x].copy_from_slice(cells);
                cells.fill(rollback.quant);
            }
            fine_quants.push(quants);
            dc_data
                .ac_strategy
                .set_first(rollback.bx, rollback.by, rollback.strategy);
        }
        let cost_without = meta_entropy_cost(&dc_data, scratch, distp.distance);
        let meta_delta = cost_with.saturating_sub(cost_without) as f32;
        let fine_lambda = crate::ac_strategy::fine_mosaic_lambda(distp.distance);
        let accepted = fine_benefit > fine_lambda * meta_delta;
        if accepted {
            for (rollback, quants) in fine_rollbacks.iter().zip(&fine_quants) {
                for iy in 0..rollback.cov_y {
                    dc_data.raw_quant_field.row_mut(rollback.by + iy)
                        [rollback.bx..rollback.bx + rollback.cov_x]
                        .copy_from_slice(&quants[iy * 4..iy * 4 + rollback.cov_x]);
                    for ix in 0..rollback.cov_x {
                        dc_data.ac_strategy.set_first(
                            rollback.bx + ix,
                            rollback.by + iy,
                            rollback.fine_grid[iy * 4 + ix],
                        );
                    }
                }
            }
            true
        } else {
            false
        }
    };

    // Sub-8x8 activation gate. `fill_ac_strategy` greedily commits every block
    // where a fine 8x8-family strategy wins the per-block RD comparison, but a
    // sparse set can disrupt prefix-code clustering of the (otherwise nearly
    // free) AC-strategy meta stream. Measure the *real* meta-token cost with the
    // exact selected set vs with all of it reverted to DCT8, and retain the set
    // only when its accumulated RD benefit covers the metadata increase. Done
    // before `write_ac_group`, so a rejected set cannot affect coefficients.
    {
        let mut positions: Vec<(usize, usize, u8)> = Vec::new();
        for y in 0..dc_data.ac_strategy.ysize() {
            for x in 0..dc_data.ac_strategy.xsize() {
                if dc_data.ac_strategy.is_first_block(x, y) {
                    let strategy = dc_data.ac_strategy.raw_strategy(x, y);
                    let in_accepted_mosaic = fine_accepted
                        && fine_rollbacks.iter().any(|rollback| {
                            x >= rollback.bx
                                && x < rollback.bx + rollback.cov_x
                                && y >= rollback.by
                                && y < rollback.by + rollback.cov_y
                        });
                    if crate::dc_group_data::is_gated_sub8_strategy(strategy) && !in_accepted_mosaic
                    {
                        positions.push((x, y, strategy));
                    }
                }
            }
        }
        if !positions.is_empty() {
            let cost_with = meta_entropy_cost(&dc_data, scratch, distp.distance);
            let xsize = dc_data.ac_strategy.xsize();
            for &(x, y, _) in &positions {
                let fallback = fine_fallbacks
                    .get(y * xsize + x)
                    .copied()
                    .unwrap_or(STRATEGY_DCT);
                dc_data.ac_strategy.set_first(x, y, fallback);
            }
            let cost_without = meta_entropy_cost(&dc_data, scratch, distp.distance);
            let meta_delta = cost_with.saturating_sub(cost_without) as f32;
            if dc_data.sub8_benefit > crate::ac_strategy::RD_LAMBDA * meta_delta {
                // Worth it: restore each exact sub-8x8 selection.
                for &(x, y, strategy) in &positions {
                    dc_data.ac_strategy.set_first(x, y, strategy);
                }
            }
            // Else leave ordinary fine blocks on the incumbent they replaced.
        }
    }
    Ok((dc_data, dc_group_xsize_groups, dc_group_ysize_groups))
}

/// Merge an AC group's origin-relative `quant_dc` into its parent DC group.
pub(super) fn merge_quant_dc(dc: &mut DcGroupData, gx: usize, gy: usize, local: &Image3S) {
    let ox = gx * K_GROUP_DIM_IN_BLOCKS;
    let oy = gy * K_GROUP_DIM_IN_BLOCKS;
    let (gwb, ghb) = (local.xsize(), local.ysize());
    for c in 0..3 {
        for ly in 0..ghb {
            let src = local.plane_row(c, ly);
            dc.quant_dc.plane_row_mut(c, oy + ly)[ox..ox + gwb].copy_from_slice(&src[..gwb]);
        }
    }
}
