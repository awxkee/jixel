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

//! AC strategy selection, quant-field refinement, and transform reranking.

use super::*;

#[derive(Clone, Copy)]
struct AcStrategyParams<'a> {
    ctx: &'a EncodingContext,
    opsin: &'a Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    distance: f32,
    scale: f32,
    qm_mult_x: f32,
    ytox_map: &'a ImageSB,
    ytob_map: &'a ImageSB,
}

struct SuperBlockContext<'a> {
    params: AcStrategyParams<'a>,
    meta_r: f32,
    scope: SearchScope,
    dct8_cost_y0: usize,
    dct8_cost_stride: usize,
}

struct SuperBlockInput {
    bx: usize,
    by: usize,
    qac: [[f32; 2]; 2],
}

fn select_super_block(
    context: &SuperBlockContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &mut AcStrategyImage,
    input: SuperBlockInput,
    saved: &mut Vec<SavedChild>,
) -> SuperBlockCost {
    let params = context.params;
    let ctx = params.ctx;
    let (bx0, by0) = (input.bx, input.by);
    let (px0, py0) = (params.dc_group_px + bx0 * 8, params.dc_group_py + by0 * 8);
    let qac = input.qac;
    let cmap_factor = cmap_factors(params.ytox_map, params.ytob_map, bx0, by0);

    // Cost of the four individual DCT8 blocks: cost[dy][dx]. DCT8 is the
    // incumbent; every merge below must beat the corresponding tiled cost by a
    // transform-size-dependent safety margin rather than by an arbitrarily tiny
    // model delta.
    let mut c8 = [[0.0f32; 2]; 2];
    for dy in 0..2 {
        for dx in 0..2 {
            c8[dy][dx] = strategy_cost(
                ctx,
                scratch,
                STRATEGY_DCT,
                params.opsin,
                px0 + dx * 8,
                py0 + dy * 8,
                qac[dy][dx],
                params.qm_mult_x,
                context.meta_r,
                params.distance,
                cmap_factor,
            );
            scratch.dct8_costs
                [(by0 + dy - context.dct8_cost_y0) * context.dct8_cost_stride + bx0 + dx] =
                c8[dy][dx];
        }
    }

    // Vertical pairs (DCT16X8): one per column. Skipped entirely under
    // `SearchScope::Squares` — four `strategy_cost` calls per super-block.
    let merge = ctx.merge;
    let mut rect_cost = |px: usize, py: usize, strategy: u8, qac: f32| -> f32 {
        if !context.scope.rectangles() {
            return f32::INFINITY;
        }
        merge.bias_rect
            * strategy_cost(
                ctx,
                scratch,
                strategy,
                params.opsin,
                px,
                py,
                qac,
                params.qm_mult_x,
                context.meta_r,
                params.distance,
                cmap_factor,
            )
    };
    let v_left = rect_cost(px0, py0, STRATEGY_DCT16X8, qac[0][0].max(qac[1][0]));
    let v_right = rect_cost(px0 + 8, py0, STRATEGY_DCT16X8, qac[0][1].max(qac[1][1]));
    let h_top = rect_cost(px0, py0, STRATEGY_DCT8X16, qac[0][0].max(qac[0][1]));
    let h_bot = rect_cost(px0, py0 + 8, STRATEGY_DCT8X16, qac[1][0].max(qac[1][1]));

    // The single DCT16X16 over all four.
    let c16 = merge.bias_16x16
        * strategy_cost(
            ctx,
            scratch,
            STRATEGY_DCT16X16,
            params.opsin,
            px0,
            py0,
            aggregate_qac_2x2(qac, params.scale, params.distance),
            params.qm_mult_x,
            context.meta_r,
            params.distance,
            cmap_factor,
        );

    let dct8_left = c8[0][0] + c8[1][0];
    let dct8_right = c8[0][1] + c8[1][1];
    let dct8_top = c8[0][0] + c8[0][1];
    let dct8_bottom = c8[1][0] + c8[1][1];
    let total_dct8 = dct8_left + dct8_right;

    let use_v_left = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X8)
        && merge_beats_dct8(v_left, dct8_left, merge.accept_pair);
    let use_v_right = ac_strategy.can_place_strategy(bx0 + 1, by0, STRATEGY_DCT16X8)
        && merge_beats_dct8(v_right, dct8_right, merge.accept_pair);
    let use_h_top = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_top, dct8_top, merge.accept_pair);
    let use_h_bottom = ac_strategy.can_place_strategy(bx0, by0 + 1, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_bot, dct8_bottom, merge.accept_pair);

    let cost_16x8 = if use_v_left { v_left } else { dct8_left }
        + if use_v_right { v_right } else { dct8_right };
    let cost_8x16 =
        if use_h_top { h_top } else { dct8_top } + if use_h_bottom { h_bot } else { dct8_bottom };
    let best_rect = cost_16x8.min(cost_8x16);

    let (q_min, q_max) = qac
        .iter()
        .flatten()
        .fold((f32::INFINITY, 0.0f32), |(mn, mx), &q| {
            (mn.min(q), mx.max(q))
        });
    let pick_16x16 = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X16)
        && c16 < best_rect
        && merge_beats_dct8(
            c16,
            total_dct8,
            risk_gated(merge.risk_k, merge.accept_16, q_min, q_max, 1.0),
        );

    let chosen = if pick_16x16 {
        // The losing rectangular arm is this merge's true child layout; the
        // map still holds plain DCT8s (rects are only committed when 16X16
        // loses), so rebuild it from the decision flags. All-DCT8 arms are
        // skipped — the rerank's incumbent fallback covers them.
        let mut grid = [NO_CHILD_BLOCK; 16];
        let any_rect = if cost_16x8 <= cost_8x16 {
            grid[0] = if use_v_left {
                STRATEGY_DCT16X8
            } else {
                STRATEGY_DCT
            };
            grid[4] = if use_v_left {
                NO_CHILD_BLOCK
            } else {
                STRATEGY_DCT
            };
            grid[1] = if use_v_right {
                STRATEGY_DCT16X8
            } else {
                STRATEGY_DCT
            };
            grid[5] = if use_v_right {
                NO_CHILD_BLOCK
            } else {
                STRATEGY_DCT
            };
            use_v_left || use_v_right
        } else {
            grid[0] = if use_h_top {
                STRATEGY_DCT8X16
            } else {
                STRATEGY_DCT
            };
            grid[1] = if use_h_top {
                NO_CHILD_BLOCK
            } else {
                STRATEGY_DCT
            };
            grid[4] = if use_h_bottom {
                STRATEGY_DCT8X16
            } else {
                STRATEGY_DCT
            };
            grid[5] = if use_h_bottom {
                NO_CHILD_BLOCK
            } else {
                STRATEGY_DCT
            };
            use_h_top || use_h_bottom
        };
        if any_rect {
            saved.push(SavedChild {
                bx: bx0 as u16,
                by: by0 as u16,
                grid,
            });
        }
        ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X16);
        c16
    } else if cost_16x8 <= cost_8x16 {
        if use_v_left {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X8);
        }
        if use_v_right {
            ac_strategy.set_first(bx0 + 1, by0, STRATEGY_DCT16X8);
        }
        cost_16x8
    } else {
        if use_h_top {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT8X16);
        }
        if use_h_bottom {
            ac_strategy.set_first(bx0, by0 + 1, STRATEGY_DCT8X16);
        }
        cost_8x16
    };

    SuperBlockCost {
        chosen,
        dct8: total_dct8,
    }
}

/// For each multi-block transform, aggregate `raw_quant` across the covered
/// blocks so the field is consistent within a transform (libjxl-tiny
/// `AdjustQuantField`).
#[inline]
fn mean_max_mixer(butteraugli_target: f32) -> f32 {
    const K_LIMIT: f32 = 1.54138;
    const K_MUL: f32 = 0.56391;
    (1.0 - (butteraugli_target - K_LIMIT).max(0.0) * K_MUL).max(0.0)
}

#[inline]
fn aggregate_quant(max_q: u8, sum: u32, covered: usize, butteraugli_target: f32) -> u8 {
    if covered < 4 {
        return max_q;
    }
    let mixer = mean_max_mixer(butteraugli_target);
    let mean = sum as f32 / covered as f32;
    let mixed = max_q as f32 * mixer + mean * (1.0 - mixer);
    (mixed + 0.5).clamp(1.0, 255.0) as u8
}

#[inline]
fn aggregate_qac_2x2(qac: [[f32; 2]; 2], scale: f32, butteraugli_target: f32) -> f32 {
    let max_q = qac[0][0].max(qac[0][1]).max(qac[1][0]).max(qac[1][1]);
    let mixer = mean_max_mixer(butteraugli_target);
    let mean = (qac[0][0] + qac[0][1] + qac[1][0] + qac[1][1]) * 0.25;
    let mixed_q = (max_q * mixer + mean * (1.0 - mixer)) / scale;
    scale * (mixed_q + 0.5).clamp(1.0, 255.0).floor()
}

pub(crate) fn adjust_quant_field(
    ac_strategy: &AcStrategyImage,
    butteraugli_target: f32,
    quant_field: &mut ImageB,
) {
    for (x, y, raw_strategy) in ac_strategy.iter_first_blocks() {
        let cov_x = AcStrategyImage::covered_blocks_x_of(raw_strategy);
        let cov_y = AcStrategyImage::covered_blocks_y_of(raw_strategy);
        if cov_x == 1 && cov_y == 1 {
            continue;
        }
        let mut max_q: u8 = 0;
        let mut sum: u32 = 0;
        for iy in 0..cov_y {
            for &q in &quant_field.row(y + iy)[x..x + cov_x] {
                max_q = max_q.max(q);
                sum += q as u32;
            }
        }
        let val = aggregate_quant(max_q, sum, cov_x * cov_y, butteraugli_target);
        for iy in 0..cov_y {
            quant_field.row_mut(y + iy)[x..x + cov_x].fill(val);
        }
    }
}

#[inline]
fn quant_refinement_steps(distance: f32) -> usize {
    if distance < 1.5 {
        0
    } else if distance >= 2.0 {
        2
    } else {
        1
    }
}

struct QuantRefinementContext<'a> {
    params: AcStrategyParams<'a>,
    ac_strategy: &'a AcStrategyImage,
    current_costs: &'a [f32],
    bands: &'a [(usize, usize)],
    num_threads: usize,
}

struct QuantRefinementSearch<'a> {
    context: &'a QuantRefinementContext<'a>,
    quant_field: &'a ImageB,
}

fn find_quant_refinements(
    search: &QuantRefinementSearch<'_>,
    scratch: &mut CoderScratch,
    band: (usize, usize),
    refinements: &mut Vec<QuantRefinement>,
) {
    let context = search.context;
    let params = context.params;
    let quant_field = search.quant_field;
    let ac_strategy = context.ac_strategy;
    let (y0, y1) = band;
    let ctx = params.ctx;
    let distance = params.distance;
    let steps = quant_refinement_steps(distance);
    if steps == 0 {
        return;
    }
    for (bx, by, strategy) in ac_strategy.iter_first_blocks() {
        if by < y0 || by >= y1 {
            continue;
        }
        if matches!(
            strategy,
            STRATEGY_DCT64X64 | STRATEGY_DCT64X32 | STRATEGY_DCT32X64
        ) {
            continue;
        }
        let cov_x = AcStrategyImage::covered_blocks_x_of(strategy);
        let cov_y = AcStrategyImage::covered_blocks_y_of(strategy);
        let current_q = quant_field.row(by)[bx];
        if current_q <= 1 {
            continue;
        }
        let cmap = cmap_factors(params.ytox_map, params.ytob_map, bx, by);
        let px = params.dc_group_px + bx * 8;
        let py = params.dc_group_py + by * 8;
        let CoderScratch {
            strategy_coeffs: coeffs,
            transform_gather,
            recon,
            ..
        } = scratch;
        let (cx, cy, _) = prepare_strategy_coeffs(
            ctx,
            coeffs,
            transform_gather,
            strategy,
            params.opsin,
            px,
            py,
            cmap,
        );
        let mut cost = |q: u8| {
            let qac = params.scale * q as f32;
            let (distortion, rate) = reconstruction_dist_and_rate(
                ctx,
                recon,
                coeffs,
                strategy,
                params.opsin,
                px,
                py,
                qac,
                params.qm_mult_x,
                distance,
                cmap,
                cx,
                cy,
                0.0,
                0.0,
            );
            rd_cost(
                DistortionModel::Reconstruction,
                distance,
                0.0,
                distortion,
                rate,
            )
        };
        let mut best_q = current_q;
        let cached_cost = context
            .current_costs
            .get(by * ac_strategy.xsize() + bx)
            .copied()
            .unwrap_or(f32::NAN);
        let mut best_cost = if cached_cost.is_finite() {
            cached_cost
        } else {
            cost(current_q)
        };
        // Bidirectional: q-1/q-2 can only save rate on over-spent blocks; q+1/q+2
        // let the field *spend* a step where the reconstruction says it is cheap.
        let candidates = if steps == 2 {
            // Upward stays at +1: q+2 measurably over-spends (its metadata-rate
            // cost is not priced here), while the -2 rate save still pays.
            [
                current_q.saturating_sub(1),
                current_q.saturating_sub(2),
                current_q.saturating_add(1),
                current_q,
            ]
        } else {
            [
                current_q.saturating_sub(1),
                current_q.saturating_add(1),
                current_q,
                current_q,
            ]
        };
        for candidate in candidates {
            if candidate == 0 || candidate == best_q || candidate == current_q {
                continue;
            }
            let candidate_cost = cost(candidate);
            if candidate_cost < best_cost {
                best_q = candidate;
                best_cost = candidate_cost;
            }
        }
        if best_q != current_q {
            refinements.push(QuantRefinement {
                bx,
                by,
                cov_x,
                cov_y,
                q: best_q,
            });
        }
    }
}

fn refine_quant_field(
    context: &QuantRefinementContext<'_>,
    scratch: &mut CoderScratch,
    quant_field: &mut ImageB,
    band_scratch: &mut [AcStrategyBandScratch],
) {
    if quant_refinement_steps(context.params.distance) == 0 {
        return;
    }
    {
        let search = QuantRefinementSearch {
            context,
            quant_field,
        };
        context
            .params
            .ctx
            .thread_pool
            .steal_for_each_mut_with_threads(
                scratch,
                &mut band_scratch[..context.bands.len()],
                context.num_threads,
                |i, band_scratch, scratch| {
                    band_scratch.quant_refinements.clear();
                    find_quant_refinements(
                        &search,
                        scratch,
                        context.bands[i],
                        &mut band_scratch.quant_refinements,
                    );
                },
            );
    }
    for band in &band_scratch[..context.bands.len()] {
        for refinement in &band.quant_refinements {
            for iy in 0..refinement.cov_y {
                quant_field.row_mut(refinement.by + iy)
                    [refinement.bx..refinement.bx + refinement.cov_x]
                    .fill(refinement.q);
            }
        }
    }
}

/// Select transforms for every aligned 2×2 super-block in the DC group, then
/// reconcile the quant field. `(dc_group_px, dc_group_py)` is the DC group's
/// top-left in absolute image pixels (so `opsin` can be the full image).
#[allow(clippy::too_many_arguments)]
#[inline]
fn region_qac(
    quant_field: &ImageB,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    scale: f32,
    butteraugli_target: f32,
) -> f32 {
    let mut max_q = 1u8;
    let mut sum = 0u32;
    for y in by..by + h {
        for &q in &quant_field.row(y)[bx..bx + w] {
            max_q = max_q.max(q);
            sum += u32::from(q);
        }
    }
    scale * aggregate_quant(max_q, sum, w * h, butteraugli_target) as f32
}

#[inline]
fn block_qac_2x2(quant_field: &ImageB, bx: usize, by: usize, scale: f32) -> [[f32; 2]; 2] {
    [
        [
            scale * quant_field.row(by)[bx] as f32,
            scale * quant_field.row(by)[bx + 1] as f32,
        ],
        [
            scale * quant_field.row(by + 1)[bx] as f32,
            scale * quant_field.row(by + 1)[bx + 1] as f32,
        ],
    ]
}

/// Run AC-strategy selection for block rows `by` in `[y_begin, y_end)` into
/// `ac_strategy`, returning the accumulated sub-8x8 RD benefit for those rows.
/// `ysize`/`xsize` are the *full* group dimensions: edge tests (`four_row`,
/// loop bounds) use the global size, exactly as the serial loop would, so a
/// 4-aligned `[y_begin, y_end)` partition reproduces the single-threaded
/// decision sequence bit-for-bit. Reads `quant_field`/`opsin` only.
struct SelectionContext<'a> {
    params: AcStrategyParams<'a>,
    quant_field: &'a ImageB,
    meta_r: f32,
    scope: SearchScope,
}

#[derive(Clone, Copy)]
pub(crate) struct Chosen32Cost {
    pub(crate) bx: u16,
    pub(crate) by: u16,
    pub(crate) cost: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct SavedChild {
    pub(crate) bx: u16,
    pub(crate) by: u16,
    pub(crate) grid: [u8; 16],
}

pub(crate) const NO_CHILD_BLOCK: u8 = 0xFF;

fn capture_children(
    map: &AcStrategyImage,
    saved: &mut Vec<SavedChild>,
    bx: usize,
    by: usize,
    cov_x: usize,
    cov_y: usize,
) {
    let mut grid = [NO_CHILD_BLOCK; 16];
    let mut nontrivial = false;
    for iy in 0..cov_y {
        for ix in 0..cov_x {
            if map.is_first_block(bx + ix, by + iy) {
                let s = map.raw_strategy(bx + ix, by + iy);
                grid[iy * 4 + ix] = s;
                nontrivial |= s != STRATEGY_DCT;
            }
        }
    }
    if nontrivial {
        saved.push(SavedChild {
            bx: bx as u16,
            by: by as u16,
            grid,
        });
    }
}

fn select_band(
    selection: &SelectionContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &mut AcStrategyImage,
    band: (usize, usize),
    chosen32: &mut Vec<Chosen32Cost>,
    saved: &mut Vec<SavedChild>,
) -> f32 {
    let params = selection.params;
    let ctx = params.ctx;
    let opsin = params.opsin;
    let distance = params.distance;
    let scale = params.scale;
    let qm_mult_x = params.qm_mult_x;
    let quant_field = selection.quant_field;
    let ytox_map = params.ytox_map;
    let ytob_map = params.ytob_map;
    let meta_r = selection.meta_r;
    let scope = selection.scope;
    let (y_begin, y_end) = band;
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    let super_block = SuperBlockContext {
        params,
        meta_r,
        scope,
        dct8_cost_y0: y_begin,
        dct8_cost_stride: xsize,
    };
    let merge = ctx.merge;
    // First-pass DCT8 incumbents are consumed again by the sub-8 refinement.
    // Keep only this worker's row band so parallel selection stays independent.
    let costs_size = xsize * (y_end - y_begin);
    if scratch.dct8_costs.len() != xsize * (y_end - y_begin) {
        scratch.dct8_costs.clear();
        scratch.dct8_costs.resize(costs_size, f32::NAN);
    } else {
        scratch.dct8_costs.fill(f32::NAN);
    }
    let mut by = y_begin;
    while by + 1 < ysize && by < y_end {
        // A 4-block-tall band can host DCT32X32 only when 4-aligned and fitting.
        let four_row = by.is_multiple_of(4) && by + 4 <= ysize;
        let mut bx = 0;
        while bx + 1 < xsize {
            let four_col = bx % 4 == 0 && bx + 4 <= xsize;
            if four_row && four_col && ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT32X32) {
                let mut sub_total = 0.0f32;
                let mut dct8_total = 0.0f32;
                for sy in 0..2 {
                    for sx in 0..2 {
                        let sbx = bx + sx * 2;
                        let sby = by + sy * 2;
                        let qac = block_qac_2x2(quant_field, sbx, sby, scale);
                        let costs = select_super_block(
                            &super_block,
                            scratch,
                            ac_strategy,
                            SuperBlockInput {
                                bx: sbx,
                                by: sby,
                                qac,
                            },
                            saved,
                        );
                        sub_total += costs.chosen;
                        dct8_total += costs.dct8;
                    }
                }
                let qac32 = region_qac(quant_field, bx, by, 4, 4, scale, distance);
                let cmap_factor = cmap_factors(ytox_map, ytob_map, bx, by);
                let cost32 = strategy_cost(
                    ctx,
                    scratch,
                    STRATEGY_DCT32X32,
                    opsin,
                    params.dc_group_px + bx * 8,
                    params.dc_group_py + by * 8,
                    qac32,
                    qm_mult_x,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                // Two DCT32X16 (each 2 wide x 4 tall) tiling the region: left +
                // right, and two DCT16X32 (each 4 wide x 2 tall): top + bottom.
                // Skipped under `SearchScope::Squares`.
                let mut rect32 =
                    |bx: usize, by: usize, strategy: u8, cw: usize, ch: usize| -> f32 {
                        if !scope.rectangles() {
                            return f32::INFINITY;
                        }
                        strategy_cost(
                            ctx,
                            scratch,
                            strategy,
                            opsin,
                            params.dc_group_px + bx * 8,
                            params.dc_group_py + by * 8,
                            region_qac(quant_field, bx, by, cw, ch, scale, distance),
                            qm_mult_x,
                            meta_r,
                            distance,
                            cmap_factor,
                        )
                    };
                let cl = rect32(bx, by, STRATEGY_DCT32X16, 2, 4);
                let cr = rect32(bx + 2, by, STRATEGY_DCT32X16, 2, 4);
                let ct = rect32(bx, by, STRATEGY_DCT16X32, 4, 2);
                let cb = rect32(bx, by + 2, STRATEGY_DCT16X32, 4, 2);

                let can_32x32 = ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT32X32);
                let can_32x16 = ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT32X16)
                    && ac_strategy.can_place_strategy(bx + 2, by, STRATEGY_DCT32X16);
                let can_16x32 = ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT16X32)
                    && ac_strategy.can_place_strategy(bx, by + 2, STRATEGY_DCT16X32);

                let cost_32x32 = if can_32x32 {
                    merge.bias_32x32 * cost32
                } else {
                    f32::INFINITY
                };
                let cost_32x16 = if can_32x16 {
                    merge.bias_rect32 * (cl + cr)
                } else {
                    f32::INFINITY
                };
                let cost_16x32 = if can_16x32 {
                    merge.bias_rect32 * (ct + cb)
                } else {
                    f32::INFINITY
                };

                let (mut q_min, mut q_max) = (u8::MAX, 0u8);
                for iy in 0..4 {
                    for &q in &quant_field.row(by + iy)[bx..bx + 4] {
                        q_min = q_min.min(q);
                        q_max = q_max.max(q);
                    }
                }
                let gate =
                    |accept: f32| risk_gated(merge.risk_k, accept, q_min as f32, q_max as f32, 2.0);
                let (best_big, best_strategy, accept) =
                    if cost_32x32 <= cost_32x16 && cost_32x32 <= cost_16x32 {
                        (cost_32x32, STRATEGY_DCT32X32, gate(merge.accept_32))
                    } else if cost_32x16 <= cost_16x32 {
                        (cost_32x16, STRATEGY_DCT32X16, gate(merge.accept_32_rect))
                    } else {
                        (cost_16x32, STRATEGY_DCT16X32, gate(merge.accept_32_rect))
                    };

                // Compare against both the already-selected subdivision and the
                // pure DCT8 incumbent. The latter prevents a sequence of locally
                // marginal merges from making a 32×32 merge look trustworthy.
                let merged = best_big < sub_total && merge_beats_dct8(best_big, dct8_total, accept);
                if merged {
                    match best_strategy {
                        STRATEGY_DCT32X32 => {
                            capture_children(ac_strategy, saved, bx, by, 4, 4);
                            ac_strategy.set_first(bx, by, STRATEGY_DCT32X32);
                        }
                        STRATEGY_DCT32X16 => {
                            capture_children(ac_strategy, saved, bx, by, 2, 4);
                            capture_children(ac_strategy, saved, bx + 2, by, 2, 4);
                            ac_strategy.set_first(bx, by, STRATEGY_DCT32X16);
                            ac_strategy.set_first(bx + 2, by, STRATEGY_DCT32X16);
                        }
                        STRATEGY_DCT16X32 => {
                            capture_children(ac_strategy, saved, bx, by, 4, 2);
                            capture_children(ac_strategy, saved, bx, by + 2, 4, 2);
                            ac_strategy.set_first(bx, by, STRATEGY_DCT16X32);
                            ac_strategy.set_first(bx, by + 2, STRATEGY_DCT16X32);
                        }
                        _ => unreachable!(),
                    }
                }
                chosen32.push(Chosen32Cost {
                    bx: bx as u16,
                    by: by as u16,
                    cost: if merged { best_big } else { sub_total },
                });
                bx += 4;
            } else if four_row {
                for sby in [by, by + 2] {
                    let qac = block_qac_2x2(quant_field, bx, sby, scale);
                    let _ = select_super_block(
                        &super_block,
                        scratch,
                        ac_strategy,
                        SuperBlockInput { bx, by: sby, qac },
                        saved,
                    );
                }
                bx += 2;
            } else {
                let qac = block_qac_2x2(quant_field, bx, by, scale);
                let _ = select_super_block(
                    &super_block,
                    scratch,
                    ac_strategy,
                    SuperBlockInput { bx, by, qac },
                    saved,
                );
                bx += 2;
            }
        }
        by += if four_row { 4 } else { 2 };
    }

    // Sub-8x8 refinement for this row band. `SearchScope::Squares` stops here:
    // it commits no sub-8 strategies, so the caller's activation gate (which
    // weighs their metadata cost) sees an empty set and is a no-op.
    let mut benefit = 0.0f32;
    if !scope.rectangles() {
        return benefit;
    }
    let with_dct4 = distance <= SUB8_MAX_DISTANCE;
    let with_fine = distance <= FINE_TRANSFORM_MAX_DISTANCE;
    if !with_dct4 && distance > AFV_MAX_DISTANCE && !with_fine {
        return benefit;
    }
    let bias_afv = BIAS_AFV.at(distance);
    for by in y_begin..y_end {
        for bx in 0..xsize {
            if ac_strategy.raw_strategy(bx, by) != STRATEGY_DCT {
                continue;
            }
            let cached_dct8 = scratch.dct8_costs[(by - y_begin) * xsize + bx];
            if let Some((cand, gain)) = evaluate_sub8_candidate(
                params,
                scratch,
                bx,
                by,
                region_qac(quant_field, bx, by, 1, 1, scale, distance),
                meta_r,
                cached_dct8.is_finite().then_some(cached_dct8),
                with_dct4,
                with_fine,
                bias_afv,
            ) {
                ac_strategy.set_first(bx, by, cand);
                benefit += gain;
            }
        }
    }
    benefit
}

#[allow(clippy::too_many_arguments)]
fn evaluate_sub8_candidate(
    params: AcStrategyParams<'_>,
    scratch: &mut CoderScratch,
    bx: usize,
    by: usize,
    qac: f32,
    meta_r: f32,
    cached_dct8: Option<f32>,
    with_dct4: bool,
    with_fine: bool,
    bias_afv: f32,
) -> Option<(u8, f32)> {
    let ctx = params.ctx;
    let px = params.dc_group_px + bx * 8;
    let py = params.dc_group_py + by * 8;
    let cmap_factor = cmap_factors(params.ytox_map, params.ytob_map, bx, by);
    let costs = sub8_strategy_costs(
        ctx,
        scratch,
        params.opsin,
        px,
        py,
        qac,
        params.qm_mult_x,
        meta_r,
        params.distance,
        cmap_factor,
        cached_dct8,
        with_dct4,
        with_fine,
    );
    let cost8 = costs.dct8;
    let cost_identity = fine_transform_bias(1.042_754_3, params.distance) * costs.identity;
    let cost_dct2x2 = fine_transform_bias(0.95, params.distance) * costs.dct2x2;
    let cost4 = BIAS_4X4 * costs.dct4x4;
    let cost48 = BIAS_4X8 * costs.dct4x8;
    let cost84 = BIAS_4X8 * costs.dct8x4;
    // Keep the already-fitted DCT4/AFV chooser intact. IDENTITY/DCT2X2 have
    // non-orthogonal coefficient scales, so their coefficient-domain costs are
    // suitable only as a shortlist; final admission is reconstruction-domain.
    let (cand, cand_cost) = {
        let mut best = STRATEGY_DCT4X4;
        let mut bc = cost4;
        if cost48 < bc {
            best = STRATEGY_DCT4X8;
            bc = cost48;
        }
        if cost84 < bc {
            best = STRATEGY_DCT8X4;
            bc = cost84;
        }
        for (kind, &afv_cost) in costs.afv.iter().enumerate() {
            let biased = bias_afv * afv_cost;
            if biased < bc {
                best = STRATEGY_AFV0 + kind as u8;
                bc = biased;
            }
        }
        (best, bc)
    };

    let (fine, fine_coeff_cost) = if cost_identity < cost_dct2x2 {
        (STRATEGY_IDENTITY, cost_identity)
    } else {
        (STRATEGY_DCT2X2, cost_dct2x2)
    };
    if fine_coeff_cost < cost8 && fine_coeff_cost < cand_cost {
        let reconstruction_cost = |scratch: &mut CoderScratch, strategy| {
            strategy_cost_impl(
                ctx,
                scratch,
                strategy,
                params.opsin,
                px,
                py,
                qac,
                params.qm_mult_x,
                meta_r,
                params.distance,
                cmap_factor,
                DistortionModel::Reconstruction,
            )
        };
        let recon8 = reconstruction_cost(scratch, STRATEGY_DCT);
        let recon_fine = reconstruction_cost(scratch, fine);
        // A small safety margin absorbs the remaining mismatch between the
        // local reconstruction metric and the final post-filtered image.
        if recon_fine < recon8 * FINE_RECON_MARGIN {
            return Some((fine, recon8 - recon_fine));
        }
    }
    (cand_cost < cost8).then_some((cand, cost8 - cand_cost))
}

/// Partition `[0, ysize)` into at most `n` contiguous bands whose interior
/// boundaries are multiples of 4 (so DCT32X32's 4-block super-rows never span a
/// boundary). The serial loop only ever takes non-4 (`+2`) steps at the image
/// bottom, which lands wholly inside the final band — hence the partition
/// reproduces the single-threaded `by` sequence exactly.
fn fill_selection_bands(bands: &mut Vec<(usize, usize)>, ysize: usize, n: usize) {
    bands.clear();
    if n <= 1 || ysize < 8 {
        bands.push((0, ysize));
        return;
    }
    let mut previous = 0;
    for k in 1..n {
        let b = (ysize * k / n) / 4 * 4;
        if b > previous && b < ysize {
            bands.push((previous, b));
            previous = b;
        }
    }
    bands.push((previous, ysize));
}

pub(crate) fn fill_ac_strategy(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    distance: f32,
    scale: f32,
    x_qm_scale: u32,
    quant_field: &mut ImageB,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
    num_threads: usize,
) -> f32 {
    let speed = ctx.speed;
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    // DCT8 wins the high-quality RD comparison outright; Fastest skips the
    // search by contract. Either way the default strategy image (all DCT8
    // first blocks) is already the answer.
    if use_dct8_only(distance) || speed == crate::Speed::Fastest {
        return 0.0;
    }
    let scope = SearchScope::for_speed(speed);
    let qm_mult_x = 1.25f32.powf(x_qm_scale as f32 - 2.0);
    // Per-candidate-block metadata rate for the strategy chooser (bits),
    // faded in above d=1 (see strategy_cost).
    let meta_r = META_R;
    let params = AcStrategyParams {
        ctx,
        opsin,
        dc_group_px,
        dc_group_py,
        distance,
        scale,
        qm_mult_x,
        ytox_map,
        ytob_map,
    };

    // Keep all pipeline storage on the worker scratch. Taking ownership lets
    // its fields remain borrowed while `scratch` is handed to nested work.
    let mut pipeline = std::mem::take(&mut scratch.ac_strategy);
    fill_selection_bands(&mut pipeline.bands, ysize, num_threads);
    let parallel_selection = pipeline.bands.len() > 1;
    pipeline.prepare_bands(xsize, ysize, parallel_selection);

    let selection = SelectionContext {
        params,
        quant_field,
        meta_r,
        scope,
    };
    let mut benefit = if !parallel_selection {
        let band0 = &mut pipeline.band_scratch[0];
        band0.chosen32.clear();
        band0.saved_children.clear();
        select_band(
            &selection,
            scratch,
            ac_strategy,
            (0, ysize),
            &mut band0.chosen32,
            &mut band0.saved_children,
        )
    } else {
        // Each band selects into its reusable strategy image, reading the shared
        // opsin/quant field; results merge deterministically by row.
        let bands = &pipeline.bands;
        let band_scratch = &mut pipeline.band_scratch[..bands.len()];
        ctx.thread_pool.steal_for_each_mut_with_threads(
            scratch,
            band_scratch,
            num_threads,
            |i, output, scratch| {
                let (y0, y1) = bands[i];
                output.benefit = select_band(
                    &selection,
                    scratch,
                    &mut output.strategy,
                    (y0, y1),
                    &mut output.chosen32,
                    &mut output.saved_children,
                );
            },
        );
        let mut benefit = 0.0f32;
        for (&(y0, y1), output) in bands.iter().zip(band_scratch) {
            ac_strategy.copy_rows_from(&output.strategy, y0, y1);
            benefit += output.benefit;
        }
        benefit
    };

    // Assemble the full-image quadrant-cost grid for the DCT64 baseline.
    let qx = xsize.div_ceil(4);
    let qy = ysize.div_ceil(4);
    pipeline.chosen32_grid.clear();
    pipeline.chosen32_grid.resize(qx * qy, f32::NAN);
    for band in &pipeline.band_scratch[..pipeline.bands.len().max(1)] {
        for c in &band.chosen32 {
            pipeline.chosen32_grid[(c.by as usize / 4) * qx + c.bx as usize / 4] = c.cost;
        }
    }

    if use_dct64(speed, distance) {
        let qx = xsize.div_ceil(4);
        for by in (0..ysize.saturating_sub(7)).step_by(8) {
            for bx in (0..xsize.saturating_sub(7)).step_by(8) {
                if !ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT64X64) {
                    continue;
                }
                let quad = |sx: usize, sy: usize| -> f32 {
                    pipeline.chosen32_grid[((by + sy) / 4) * qx + (bx + sx) / 4]
                };
                let cost32 = [[quad(0, 0), quad(4, 0)], [quad(0, 4), quad(4, 4)]];
                if cost32.iter().flatten().any(|c| !c.is_finite()) {
                    continue;
                }
                let cmap = cmap_factors(ytox_map, ytob_map, bx, by);
                let cost64 = BIAS_64X64
                    * strategy_cost64(
                        ctx,
                        scratch,
                        STRATEGY_DCT64X64,
                        opsin,
                        dc_group_px + bx * 8,
                        dc_group_py + by * 8,
                        region_qac(quant_field, bx, by, 8, 8, scale, distance),
                        qm_mult_x,
                        meta_r,
                        distance,
                        cmap,
                    );

                let tall = [0usize, 4].map(|sx| {
                    BIAS_64_RECT
                        * strategy_cost64(
                            ctx,
                            scratch,
                            STRATEGY_DCT64X32,
                            opsin,
                            dc_group_px + (bx + sx) * 8,
                            dc_group_py + by * 8,
                            region_qac(quant_field, bx + sx, by, 4, 8, scale, distance),
                            qm_mult_x,
                            meta_r,
                            distance,
                            cmap,
                        )
                });
                let wide = [0usize, 4].map(|sy| {
                    BIAS_64_RECT
                        * strategy_cost64(
                            ctx,
                            scratch,
                            STRATEGY_DCT32X64,
                            opsin,
                            dc_group_px + bx * 8,
                            dc_group_py + (by + sy) * 8,
                            region_qac(quant_field, bx, by + sy, 8, 4, scale, distance),
                            qm_mult_x,
                            meta_r,
                            distance,
                            cmap,
                        )
                });

                let accept_rect = ctx.merge.accept_64_rect;
                let tall_use = [
                    merge_beats_dct8(tall[0], cost32[0][0] + cost32[1][0], accept_rect),
                    merge_beats_dct8(tall[1], cost32[0][1] + cost32[1][1], accept_rect),
                ];
                let wide_use = [
                    merge_beats_dct8(wide[0], cost32[0][0] + cost32[0][1], accept_rect),
                    merge_beats_dct8(wide[1], cost32[1][0] + cost32[1][1], accept_rect),
                ];
                let base_cost: f32 = cost32.iter().flatten().sum();
                let tall_score = (if tall_use[0] {
                    tall[0] / accept_rect
                } else {
                    cost32[0][0] + cost32[1][0]
                }) + if tall_use[1] {
                    tall[1] / accept_rect
                } else {
                    cost32[0][1] + cost32[1][1]
                };
                let wide_score = (if wide_use[0] {
                    wide[0] / accept_rect
                } else {
                    cost32[0][0] + cost32[0][1]
                }) + if wide_use[1] {
                    wide[1] / accept_rect
                } else {
                    cost32[1][0] + cost32[1][1]
                };
                let square_score = cost64 / ctx.merge.accept_64;

                if square_score < base_cost
                    && square_score <= tall_score
                    && square_score <= wide_score
                {
                    ac_strategy.set_first(bx, by, STRATEGY_DCT64X64);
                } else if tall_score < base_cost && tall_score <= wide_score {
                    if tall_use[0] {
                        ac_strategy.set_first(bx, by, STRATEGY_DCT64X32);
                    }
                    if tall_use[1] {
                        ac_strategy.set_first(bx + 4, by, STRATEGY_DCT64X32);
                    }
                } else if wide_score < base_cost {
                    if wide_use[0] {
                        ac_strategy.set_first(bx, by, STRATEGY_DCT32X64);
                    }
                    if wide_use[1] {
                        ac_strategy.set_first(bx, by + 4, STRATEGY_DCT32X64);
                    }
                }
            }
        }
    }

    // Second pass — reconstruction-based rerank. The fast selector over-merges at
    // high quality; here we revisit only the *selected* large transforms and
    // downgrade a merge to its tiled DCT8 when the SSIM-reconstruction RD cost
    // prefers it. Only large transforms are scored (a fraction of blocks), so the
    // expensive recon distortion runs on far fewer candidates than a full recon
    // selection while capturing the same structural win.

    // The SSIM reconstruction rerank runs in both scopes. Under
    // `SearchScope::Squares` the only merges present are DCT16X16 and
    // DCT32X32, so it scores exactly those.
    let reranked = scope.rerank(distance);
    if reranked {
        pipeline.prepare_rerank(xsize, ysize);
        let rerank = RerankContext {
            params,
            quant_field,
            bands: &pipeline.bands,
            num_threads,
        };
        let band_count = pipeline.bands.len();
        rerank_large_transforms(
            &rerank,
            scratch,
            ac_strategy,
            &mut pipeline.band_scratch[..band_count],
            &mut pipeline.current_costs,
        );

        if scope.rectangles() && distance <= AFV_MAX_DISTANCE.max(FINE_TRANSFORM_MAX_DISTANCE) {
            let with_dct4 = distance <= SUB8_MAX_DISTANCE;
            let with_fine = distance <= FINE_TRANSFORM_MAX_DISTANCE;
            let bias_afv = BIAS_AFV.at(distance);
            for band_idx in 0..band_count {
                for di in 0..pipeline.band_scratch[band_idx].rerank_downgrades.len() {
                    let downgrade = pipeline.band_scratch[band_idx].rerank_downgrades[di];
                    for iy in 0..downgrade.cov_y {
                        for ix in 0..downgrade.cov_x {
                            if downgrade.restore[iy * 4 + ix] != STRATEGY_DCT {
                                continue;
                            }
                            let (bx, by) = (downgrade.bx + ix, downgrade.by + iy);
                            if let Some((cand, gain)) = evaluate_sub8_candidate(
                                params,
                                scratch,
                                bx,
                                by,
                                region_qac(quant_field, bx, by, 1, 1, scale, distance),
                                META_R,
                                None,
                                with_dct4,
                                with_fine,
                                bias_afv,
                            ) {
                                ac_strategy.set_first(bx, by, cand);
                                benefit += gain;
                            }
                        }
                    }
                }
            }
        }
    }

    adjust_quant_field(ac_strategy, distance, quant_field);
    if quant_refinement_steps(distance) != 0 {
        pipeline.prepare_refinement(xsize);
    }
    let current_costs = if reranked {
        pipeline.current_costs.as_slice()
    } else {
        &[]
    };
    let refinement = QuantRefinementContext {
        params,
        ac_strategy,
        current_costs,
        bands: &pipeline.bands,
        num_threads,
    };
    let band_count = pipeline.bands.len();
    refine_quant_field(
        &refinement,
        scratch,
        quant_field,
        &mut pipeline.band_scratch[..band_count],
    );
    scratch.ac_strategy = pipeline;
    benefit
}

/// Reconstruction-based rerank pass: for each selected merge, compare its
/// SSIM-reconstruction cost against the tiled DCT8 and downgrade if DCT8 wins.
struct RerankContext<'a> {
    params: AcStrategyParams<'a>,
    quant_field: &'a ImageB,
    bands: &'a [(usize, usize)],
    num_threads: usize,
}

fn find_rerank_downgrades(
    rerank: &RerankContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &AcStrategyImage,
    band: (usize, usize),
    output: &mut AcStrategyBandScratch,
) {
    let params = rerank.params;
    let ctx = params.ctx;
    let (y0, y1) = band;
    let rerank_margin = ctx.merge.rerank_margin;
    let pair_gradient_alpha = rerank_pair_gradient_alpha(params.distance);
    let pair_gradient_peak_alpha = rerank_pair_gradient_peak_alpha(params.distance);
    let pair_gradient_fit = &PAIR_GRAD_FIT;
    // The rerank's own metadata charge. The selection pass keeps META_R, but
    // here the tiled-DCT8 alternative pays it PER TILE (16x for a 32x32)
    // while the merge pays once — a structural pro-merge credit this
    // constant prices independently (review-3 §4).
    let meta_r = RERANK_META_R;
    // Owned copy so pushes into output.current_costs below don't fight the borrow.
    let saved_map: std::collections::HashMap<(u16, u16), [u8; 16]> = output
        .saved_children
        .iter()
        .map(|s| ((s.bx, s.by), s.grid))
        .collect();
    output.rerank_downgrades.clear();
    output.current_costs.clear();
    for (bx, by, strat) in ac_strategy.iter_first_blocks() {
        if by < y0 || by >= y1 {
            continue;
        }
        if matches!(
            strat,
            STRATEGY_DCT64X64 | STRATEGY_DCT64X32 | STRATEGY_DCT32X64
        ) {
            continue;
        }
        let cxb = AcStrategyImage::covered_blocks_x_of(strat);
        let cyb = AcStrategyImage::covered_blocks_y_of(strat);
        if cxb * cyb <= 1 {
            continue; // only merges
        }
        let (px, py) = (params.dc_group_px + bx * 8, params.dc_group_py + by * 8);
        // The gradient term targets the measured sun-beam failure: rectangular
        // pairs whose rate advantage hides a localized directional error. Keep
        // square/larger reranks byte-identical while the pair weight is fitted.
        let (gradient_alpha, gradient_peak_alpha) = match strat {
            STRATEGY_DCT16X8 | STRATEGY_DCT8X16 => {
                let alpha = pair_gradient_alpha;
                let peak_alpha = pair_gradient_peak_alpha;
                if alpha == 0.0 && peak_alpha == 0.0 {
                    (0.0, 0.0)
                } else {
                    let (width, height) = if strat == STRATEGY_DCT16X8 {
                        (8, 16)
                    } else {
                        (16, 8)
                    };
                    let stats = if params.distance < RERANK_PAIR_GRADIENT_PEAK_COARSE_END {
                        (ctx.gradient_region_stats)(params.opsin, px, py, width, height, 1e-5)
                    } else {
                        (ctx.gradient_region_stats_with_chroma)(
                            params.opsin,
                            px,
                            py,
                            width,
                            height,
                            1e-5,
                        )
                    };
                    (
                        if stats.dominance >= RERANK_PAIR_GRADIENT_MIN_DOMINANCE {
                            alpha
                        } else {
                            0.0
                        },
                        if stats.dominance >= pair_gradient_fit.peak_min_dominance
                            && stats.mean >= pair_gradient_fit.min_luma
                            && (params.distance < RERANK_PAIR_GRADIENT_PEAK_COARSE_END
                                || stats.chroma <= pair_gradient_fit.chroma_cap)
                        {
                            peak_alpha
                        } else {
                            0.0
                        },
                    )
                }
            }
            _ => (0.0, 0.0),
        };
        let qac_big = region_qac(
            rerank.quant_field,
            bx,
            by,
            cxb,
            cyb,
            params.scale,
            params.distance,
        );
        let (j_big, big_current_cost) = reconstruction_strategy_cost_and_base(
            ctx,
            scratch,
            strat,
            params.opsin,
            px,
            py,
            qac_big,
            params.qm_mult_x,
            meta_r,
            params.distance,
            cmap_factors(params.ytox_map, params.ytob_map, bx, by),
            gradient_alpha,
            gradient_peak_alpha,
        );
        let mut j_dct8 = 0.0f32;
        let tiled_costs_start = output.current_costs.len();
        for iy in 0..cyb {
            for ix in 0..cxb {
                let q = region_qac(
                    rerank.quant_field,
                    bx + ix,
                    by + iy,
                    1,
                    1,
                    params.scale,
                    params.distance,
                );
                let (cost, current_cost) = reconstruction_strategy_cost_and_base(
                    ctx,
                    scratch,
                    STRATEGY_DCT,
                    params.opsin,
                    px + ix * 8,
                    py + iy * 8,
                    q,
                    params.qm_mult_x,
                    meta_r,
                    params.distance,
                    cmap_factors(params.ytox_map, params.ytob_map, bx + ix, by + iy),
                    gradient_alpha,
                    gradient_peak_alpha,
                );
                j_dct8 += cost;
                output.current_costs.push(CachedQuantCost {
                    bx: bx + ix,
                    by: by + iy,
                    cost: current_cost,
                });
            }
        }
        // Third arm: the child layout this merge displaced at selection time.
        let child_costs_start = output.current_costs.len();
        let mut j_child = f32::INFINITY;
        let mut child_grid = [NO_CHILD_BLOCK; 16];
        if let Some(grid) = saved_map.get(&(bx as u16, by as u16)) {
            let mut sum = 0.0f32;
            for iy in 0..cyb {
                for ix in 0..cxb {
                    let s = grid[iy * 4 + ix];
                    if s == NO_CHILD_BLOCK {
                        continue;
                    }
                    let ccx = AcStrategyImage::covered_blocks_x_of(s);
                    let ccy = AcStrategyImage::covered_blocks_y_of(s);
                    let q = region_qac(
                        rerank.quant_field,
                        bx + ix,
                        by + iy,
                        ccx,
                        ccy,
                        params.scale,
                        params.distance,
                    );
                    let (cost, current_cost) = reconstruction_strategy_cost_and_base(
                        ctx,
                        scratch,
                        s,
                        params.opsin,
                        px + ix * 8,
                        py + iy * 8,
                        q,
                        params.qm_mult_x,
                        meta_r,
                        params.distance,
                        cmap_factors(params.ytox_map, params.ytob_map, bx + ix, by + iy),
                        gradient_alpha,
                        gradient_peak_alpha,
                    );
                    sum += cost;
                    output.current_costs.push(CachedQuantCost {
                        bx: bx + ix,
                        by: by + iy,
                        cost: current_cost,
                    });
                }
            }
            j_child = sum;
            child_grid = *grid;
        }

        let child_wins = j_child < j_dct8;
        let j_alt = if child_wins { j_child } else { j_dct8 };
        if j_alt < j_big * rerank_margin {
            let restore = if child_wins {
                // Drop the DCT8 arm's cached costs, keep the child's.
                output
                    .current_costs
                    .drain(tiled_costs_start..child_costs_start);
                child_grid
            } else {
                output.current_costs.truncate(child_costs_start);
                let mut g = [NO_CHILD_BLOCK; 16];
                for iy in 0..cyb {
                    for ix in 0..cxb {
                        g[iy * 4 + ix] = STRATEGY_DCT;
                    }
                }
                g
            };
            output.rerank_downgrades.push(RerankDowngrade {
                bx,
                by,
                cov_x: cxb,
                cov_y: cyb,
                restore,
            });
        } else {
            output.current_costs.truncate(tiled_costs_start);
            output.current_costs.push(CachedQuantCost {
                bx,
                by,
                cost: big_current_cost,
            });
        }
    }
}

fn rerank_large_transforms(
    rerank: &RerankContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &mut AcStrategyImage,
    band_scratch: &mut [AcStrategyBandScratch],
    current_costs: &mut [f32],
) {
    let ysize = ac_strategy.ysize();
    let ac_strategy_ref: &AcStrategyImage = ac_strategy;
    rerank
        .params
        .ctx
        .thread_pool
        .steal_for_each_mut_with_threads(
            scratch,
            &mut band_scratch[..rerank.bands.len()],
            rerank.num_threads,
            |i, output, scratch| {
                find_rerank_downgrades(rerank, scratch, ac_strategy_ref, rerank.bands[i], output);
            },
        );
    debug_assert_eq!(current_costs.len(), ac_strategy.xsize() * ysize);
    current_costs.fill(f32::NAN);
    for result in &band_scratch[..rerank.bands.len()] {
        for cached in &result.current_costs {
            current_costs[cached.by * ac_strategy.xsize() + cached.bx] = cached.cost;
        }
        for downgrade in &result.rerank_downgrades {
            for iy in 0..downgrade.cov_y {
                for ix in 0..downgrade.cov_x {
                    let s = downgrade.restore[iy * 4 + ix];
                    if s != NO_CHILD_BLOCK {
                        ac_strategy.set_first(downgrade.bx + ix, downgrade.by + iy, s);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BIAS_4X4, BIAS_4X8, BIAS_16X16, BIAS_AFV, BIAS_RECT32, DCT8_ONLY_MAX_DISTANCE,
        FAST_RERANK_MAX_DISTANCE, MERGE_MARGIN_16, MERGE_MARGIN_32, MERGE_MARGIN_PAIR, MergeTuning,
        RERANK_DOWNGRADE_MARGIN, RERANK_PAIR_GRADIENT_ALPHA, RERANK_PAIR_GRADIENT_FADE_IN_END,
        RERANK_PAIR_GRADIENT_FADE_IN_START, RERANK_PAIR_GRADIENT_FADE_OUT_END,
        RERANK_PAIR_GRADIENT_FADE_OUT_START, RERANK_PAIR_GRADIENT_MIN_DOMINANCE,
        RERANK_PAIR_GRADIENT_PEAK_ALPHA, RERANK_PAIR_GRADIENT_PEAK_COARSE_ALPHA,
        RERANK_PAIR_GRADIENT_PEAK_COARSE_END, RERANK_PAIR_GRADIENT_PEAK_COARSE_START,
        RERANK_PAIR_GRADIENT_PEAK_FADE_IN_END, RERANK_PAIR_GRADIENT_PEAK_FADE_IN_START,
        RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_END, RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_START,
        SUB8_MAX_DISTANCE, SearchScope, aggregate_qac_2x2, aggregate_quant, cmap_factors,
        fill_ac_strategy, fill_selection_bands, gradient_region_stats_scalar,
        gradient_region_stats_with_chroma_scalar, merge_beats_dct8, merge_margin,
        quant_refinement_steps, rerank_pair_gradient_peak_alpha, rerank_pair_gradient_scale,
        select_gradient_region_stats_fn, select_gradient_region_stats_with_chroma_fn,
        strategy_cost, sub8_strategy_costs, use_dct8_only,
    };
    use crate::coder_scratch::CoderScratch;
    use crate::dc_group_data::{
        AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT2X2, STRATEGY_DCT4X4, STRATEGY_DCT4X8,
        STRATEGY_DCT8X4, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT32X32, STRATEGY_DCT32X64,
        STRATEGY_DCT64X32, STRATEGY_DCT64X64, STRATEGY_IDENTITY,
    };
    use crate::encoding_context::EncodingContext;
    use crate::image::{Image3F, ImageB, ImageSB};
    use crate::inflated_cost::{
        forward_for, forward_matrix, reconstruct_error, strategy_pixel_count,
    };

    /// Merges are searched at every usable distance; only the near-lossless
    /// tail is DCT8-only. The high-quality band is held back by the banded
    /// margins below, not by this gate.
    #[test]
    fn dct8_only_covers_the_near_lossless_tail_only() {
        assert_eq!(DCT8_ONLY_MAX_DISTANCE, 0.056_713_393);
        assert!(use_dct8_only(0.01));
        assert!(use_dct8_only(0.05));
        assert!(!use_dct8_only(0.06));
        assert!(!use_dct8_only(0.2));
        assert!(!use_dct8_only(1.0));
    }

    #[test]
    fn selection_band_storage_is_reused_for_smaller_jobs() {
        let mut bands = Vec::new();
        fill_selection_bands(&mut bands, 32, 12);
        assert_eq!(
            bands,
            [
                (0, 4),
                (4, 8),
                (8, 12),
                (12, 16),
                (16, 20),
                (20, 24),
                (24, 28),
                (28, 32),
            ]
        );
        let capacity = bands.capacity();
        fill_selection_bands(&mut bands, 4, 12);
        assert_eq!(bands, [(0, 4)]);
        assert_eq!(bands.capacity(), capacity);
    }

    /// A banded knob holds each end flat outside the band and interpolates
    /// inside it, so a value fitted in one band cannot move the other.
    #[test]
    fn banded_knobs_are_flat_outside_the_band() {
        assert_eq!(MERGE_MARGIN_16.at(0.1), MERGE_MARGIN_16.at(0.5));
        assert_eq!(MERGE_MARGIN_16.at(1.5), MERGE_MARGIN_16.at(6.0));
        let mid = MERGE_MARGIN_16.at(1.0);
        assert!((mid - 0.5 * (MERGE_MARGIN_16.hq + MERGE_MARGIN_16.base)).abs() < 1e-6);
        // High quality admits merges on much stiffer terms than mid/low.
        assert!(MERGE_MARGIN_16.at(0.3) > MERGE_MARGIN_16.at(2.0));
        assert!(BIAS_16X16.at(0.3) > BIAS_16X16.at(2.0));
        // The rerank defends merges at mid/low quality, knife-edge at HQ.
        assert_eq!(RERANK_DOWNGRADE_MARGIN.at(0.3), 1.0);
        assert!(RERANK_DOWNGRADE_MARGIN.at(3.0) < 1.0);
    }

    #[test]
    fn pair_gradient_weights_are_banded() {
        const { assert!(RERANK_PAIR_GRADIENT_ALPHA > 0.0) };
        assert!((0.0..1.0).contains(&RERANK_PAIR_GRADIENT_MIN_DOMINANCE));
        assert_eq!(rerank_pair_gradient_scale(0.35), 0.0);
        assert_eq!(
            rerank_pair_gradient_scale(RERANK_PAIR_GRADIENT_FADE_IN_START),
            0.0
        );
        assert_eq!(
            rerank_pair_gradient_scale(RERANK_PAIR_GRADIENT_FADE_IN_END),
            1.0
        );
        assert_eq!(
            rerank_pair_gradient_scale(RERANK_PAIR_GRADIENT_FADE_OUT_START),
            1.0
        );
        assert_eq!(
            rerank_pair_gradient_scale(RERANK_PAIR_GRADIENT_FADE_OUT_END),
            0.0
        );
        assert_eq!(rerank_pair_gradient_scale(3.0), 0.0);
        let fade_in_mid =
            0.5 * (RERANK_PAIR_GRADIENT_FADE_IN_START + RERANK_PAIR_GRADIENT_FADE_IN_END);
        assert!((rerank_pair_gradient_scale(fade_in_mid) - 0.5).abs() < 1e-6);
        let fade_out_mid =
            0.5 * (RERANK_PAIR_GRADIENT_FADE_OUT_START + RERANK_PAIR_GRADIENT_FADE_OUT_END);
        assert!((rerank_pair_gradient_scale(fade_out_mid) - 0.5).abs() < 1e-6);

        const { assert!(RERANK_PAIR_GRADIENT_PEAK_ALPHA > 0.0) };
        assert_eq!(
            rerank_pair_gradient_peak_alpha(RERANK_PAIR_GRADIENT_PEAK_FADE_IN_START),
            0.0
        );
        assert_eq!(
            rerank_pair_gradient_peak_alpha(RERANK_PAIR_GRADIENT_PEAK_FADE_IN_END),
            RERANK_PAIR_GRADIENT_PEAK_ALPHA
        );
        assert_eq!(
            rerank_pair_gradient_peak_alpha(RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_START),
            RERANK_PAIR_GRADIENT_PEAK_COARSE_ALPHA
        );
        assert_eq!(
            rerank_pair_gradient_peak_alpha(RERANK_PAIR_GRADIENT_PEAK_COARSE_START),
            RERANK_PAIR_GRADIENT_PEAK_ALPHA
        );
        assert_eq!(
            rerank_pair_gradient_peak_alpha(RERANK_PAIR_GRADIENT_PEAK_COARSE_END),
            RERANK_PAIR_GRADIENT_PEAK_COARSE_ALPHA
        );
        assert_eq!(
            rerank_pair_gradient_peak_alpha(RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_END),
            0.0
        );
    }

    #[test]
    fn quant_refinement_does_not_release_at_coarse_distances() {
        assert_eq!(quant_refinement_steps(1.0), 0);
        assert_eq!(quant_refinement_steps(1.5), 1);
        assert_eq!(quant_refinement_steps(1.99), 1);
        assert_eq!(quant_refinement_steps(2.0), 2);
        assert_eq!(quant_refinement_steps(3.49), 2);
        assert_eq!(quant_refinement_steps(3.5), 2);
        assert_eq!(quant_refinement_steps(5.0), 2);
        assert_eq!(quant_refinement_steps(25.0), 2);
    }

    /// The Fast tier runs the same RD model but offers only square merges.
    /// A flat 32x32 region must still merge, and must never come back as a
    /// rectangle, a sub-8x8 split, or a 64px transform.
    #[test]
    fn fast_scope_selects_squares_and_no_other_merge_shape() {
        let ctx = EncodingContext::new(
            crate::Speed::Fast,
            None,
            crate::xyb::XybMatrix::SPEC,
            1.0,
            1,
        );
        let maps = ImageSB::new_fill(1, 1, 0);
        let opsin = Image3F::new(32, 32);
        let mut qf = ImageB::new_fill(4, 4, 8);
        let mut strategies = AcStrategyImage::new(4, 4);
        let mut scratch = CoderScratch::default();
        fill_ac_strategy(
            &ctx,
            &mut scratch,
            &opsin,
            0,
            0,
            3.0,
            1.0,
            2,
            &mut qf,
            &maps,
            &maps,
            &mut strategies,
            1,
        );
        assert_eq!(strategies.raw_strategy(0, 0), STRATEGY_DCT32X32);
        for (_, _, strat) in strategies.iter_first_blocks() {
            assert!(
                matches!(strat, STRATEGY_DCT | STRATEGY_DCT16X16 | STRATEGY_DCT32X32),
                "Fast produced a non-square merge: {strat}"
            );
        }
    }

    /// The Fast tier reranks only at high quality, where the coefficient model
    /// over-merges; past the cutoff the rerank costs time to lose rate.
    #[test]
    fn fast_rerank_is_gated_by_distance_but_slow_always_reranks() {
        assert!(SearchScope::Squares.rerank(FAST_RERANK_MAX_DISTANCE));
        assert!(!SearchScope::Squares.rerank(FAST_RERANK_MAX_DISTANCE + 0.001));
        assert!(SearchScope::Full.rerank(6.0));
    }

    /// The sub-8 biases are a fitted pair, not incidental defaults: both are
    /// above 1.0 (making sub-8 harder to select than the raw RD comparison
    /// would), and the family is gated off past the mid band. Guards a silent
    /// revert to the unfitted flat-1.0/no-gate behaviour, which measured
    /// +0.238% BD-rate combined across Kodak and train0.
    #[test]
    fn sub8_selection_is_fitted_not_neutral() {
        // Read through runtime bindings so the checks are not const-folded away.
        let (b44, b48, gate) = (
            std::hint::black_box(BIAS_4X4),
            std::hint::black_box(BIAS_4X8),
            std::hint::black_box(SUB8_MAX_DISTANCE),
        );
        assert!(b44 > 1.0, "BIAS_4X4 = {b44}");
        assert!(b48 > 1.0, "BIAS_4X8 = {b48}");
        assert!(b44 > b48, "4X4 carries the stiffer bar");
        assert!(
            (0.8..=2.5).contains(&gate),
            "gate {gate} outside the band the corpora agree on"
        );
        // The gate must sit above the DCT8-only floor, or sub-8 would never run.
        assert!(gate > std::hint::black_box(DCT8_ONLY_MAX_DISTANCE));
        // AFV carries a stiffer-than-neutral bar in both bands too: at 1.0 it
        // over-selects (+0.107% BD flat; butteraugli pays below ~1.1 banded).
        let afv = std::hint::black_box(BIAS_AFV);
        assert!(
            afv.hq > 1.0 && afv.base > 1.0,
            "AFV bias reverted to neutral"
        );
    }

    /// AFV (with the rest of sub-8) is a Slow-only feature: the sub-8 pass
    /// runs only under `SearchScope::Full`, and only Slow maps to it.
    #[test]
    fn sub8_and_afv_are_slow_only() {
        assert_eq!(
            SearchScope::for_speed(crate::Speed::Fastest),
            SearchScope::Squares
        );
        assert_eq!(
            SearchScope::for_speed(crate::Speed::Fast),
            SearchScope::Squares
        );
        assert_eq!(
            SearchScope::for_speed(crate::Speed::Slow),
            SearchScope::Full
        );
        // The sub-8 refinement is gated on `scope.rectangles()`.
        assert!(!SearchScope::Squares.rectangles());
        assert!(SearchScope::Full.rectangles());
    }

    #[test]
    fn bundled_sub8_costs_match_independent_evaluation_and_cached_dct8() {
        let mut opsin = Image3F::new(11, 13);
        let mut state = 0x1234_5678u32;
        for c in 0..3 {
            for y in 0..opsin.ysize() {
                for value in opsin.plane_mut(c).row_mut(y) {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                }
            }
        }

        let ctx = EncodingContext::default();
        let (px, py) = (7usize, 9usize); // exercise right and bottom replication
        let qac = 3.75;
        let qm_mult_x = 1.25;
        let meta_r = 0.4;
        let distance = 1.5;
        let cmap = [0.25, 0.0, -0.375];
        let mut scratch = CoderScratch::default();
        let independent = [
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_IDENTITY,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT2X2,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT4X4,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT4X8,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT8X4,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
        ];

        for cached in [None, Some(independent[0])] {
            let bundled = sub8_strategy_costs(
                &ctx,
                &mut scratch,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
                cached,
                true,
                true,
            );
            let actual = [
                bundled.dct8,
                bundled.identity,
                bundled.dct2x2,
                bundled.dct4x4,
                bundled.dct4x8,
                bundled.dct8x4,
            ];
            for (a, b) in actual.into_iter().zip(independent) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn reconstruction_round_trips() {
        // x = N·Fᵀ·(F·x) must return the original block (exact inverse).
        // DCT4X4/4X8/8X4 use a sub-DC Hadamard (non-orthogonal), so the
        // `x=N·Fᵀc` inverse doesn't apply — but they are not merge candidates.
        let idct = crate::dct::IdctMethods::scalar();
        for strategy in [
            STRATEGY_DCT,
            STRATEGY_DCT16X8,
            STRATEGY_DCT16X16,
            STRATEGY_DCT32X32,
            STRATEGY_DCT64X64,
            STRATEGY_DCT64X32,
            STRATEGY_DCT32X64,
        ] {
            let n = strategy_pixel_count(strategy);
            // deterministic pseudo-random input
            let mut x = vec![0.0f32; n];
            let mut s = 12345u32;
            for v in &mut x {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            }
            let mut c = vec![0.0f32; n];
            forward_for(strategy, &x, &mut c);
            let mut recon = vec![0.0f32; n];
            reconstruct_error(&idct, strategy, &c, &mut recon);
            let max_err = (0..n).map(|i| (recon[i] - x[i]).abs()).fold(0.0, f32::max);
            assert!(
                max_err < 1e-3,
                "strategy {strategy}: max reconstruction err {max_err}"
            );
        }
    }

    #[test]
    fn forward_matrix_is_orthogonal() {
        // Check F·Fᵀ off-diagonals are ~0 for DCT8 (rows must be orthogonal for
        // x = N·Fᵀc to be a valid inverse).
        let n = 64;
        let ft = forward_matrix(STRATEGY_DCT); // ft[pixel*n + coeff] = F[coeff, pixel]
        // dot of coeff rows j,k = Σ_pixel F[j,pixel]F[k,pixel] = Σ_p ft[p*n+j]ft[p*n+k]
        let mut max_off = 0.0f32;
        for j in 0..8 {
            for k in 0..8 {
                if j == k {
                    continue;
                }
                let dot: f32 = (0..n).map(|p| ft[p * n + j] * ft[p * n + k]).sum();
                max_off = max_off.max(dot.abs());
            }
        }
        println!("max off-diagonal |<f_j,f_k>| = {max_off}");
        assert!(max_off < 1e-4, "DCT8 basis not orthogonal: {max_off}");
    }

    #[test]
    fn merge_guard_is_stricter_for_large_transforms_and_high_quality() {
        assert!(merge_margin(0.3, MERGE_MARGIN_32) > merge_margin(0.3, MERGE_MARGIN_16));
        assert!(merge_margin(0.3, MERGE_MARGIN_16) > merge_margin(0.3, MERGE_MARGIN_PAIR));
        assert!(merge_margin(0.3, MERGE_MARGIN_16) > merge_margin(4.0, MERGE_MARGIN_16));

        // At high quality the 19% 16x16 margin rejects a 10% estimated win but
        // accepts a clear 25% one. At coarse quality the guard fades.
        let hq = MergeTuning::new(0.3);
        let lowq = MergeTuning::new(4.0);
        assert!(!merge_beats_dct8(90.0, 100.0, hq.accept_16));
        assert!(merge_beats_dct8(75.0, 100.0, hq.accept_16));
        assert!(merge_beats_dct8(95.0, 100.0, lowq.accept_16));

        // The resolved table is exactly what the per-block path used to compute.
        assert_eq!(hq.accept_16, 1.0 - merge_margin(0.3, MERGE_MARGIN_16));
        assert_eq!(hq.bias_rect32, BIAS_RECT32.at(0.3));
        assert_eq!(lowq.rerank_margin, RERANK_DOWNGRADE_MARGIN.at(4.0));
    }

    #[test]
    fn quant_aggregation_uses_max_for_pairs_and_high_quality() {
        assert_eq!(aggregate_quant(40, 50, 2, 1.0), 40);
        assert_eq!(aggregate_quant(40, 100, 4, 1.0), 40);
    }

    #[test]
    fn quant_aggregation_matches_scaled_candidate_cost() {
        let qac = [[2.5, 5.0], [7.5, 10.0]];
        let raw = aggregate_quant(40, 100, 4, 2.0);
        assert_eq!(aggregate_qac_2x2(qac, 0.25, 2.0), raw as f32 * 0.25);
    }

    #[test]
    fn strategy_cfl_factors_match_signaled_tile_maps() {
        let ytox = ImageSB::new_fill(1, 1, 42);
        let ytob = ImageSB::new_fill(1, 1, -42);
        assert_eq!(cmap_factors(&ytox, &ytob, 0, 0), [0.5, 0.0, 0.5]);
    }

    #[test]
    fn luma_only_gradient_stats_match_full_stats() {
        let mut opsin = Image3F::new(16, 16);
        for c in 0..3 {
            for y in 0..16 {
                for x in 0..16 {
                    opsin.plane_mut(c).row_mut(y)[x] =
                        0.01 * x as f32 + 0.02 * y as f32 + 0.1 * c as f32;
                }
            }
        }
        let luma = gradient_region_stats_scalar(&opsin, 0, 0, 16, 8, 1e-5);
        let full = gradient_region_stats_with_chroma_scalar(&opsin, 0, 0, 16, 8, 1e-5);
        assert_eq!(luma.dominance, full.dominance);
        assert_eq!(luma.mean, full.mean);
        assert!(full.chroma > 0.0);
    }

    #[test]
    fn dispatched_gradient_stats_match_scalar() {
        let mut opsin = Image3F::new(24, 24);
        for c in 0..3 {
            for y in 0..24 {
                for x in 0..24 {
                    let hash = (x * 37 + y * 61 + c * 17 + x * y * 3) % 101;
                    opsin.plane_mut(c).row_mut(y)[x] = hash as f32 * 0.007 - 0.2 + c as f32 * 0.03;
                }
            }
        }

        let kernels = [
            (
                select_gradient_region_stats_fn(),
                gradient_region_stats_scalar as super::GradientRegionStatsFn,
            ),
            (
                select_gradient_region_stats_with_chroma_fn(),
                gradient_region_stats_with_chroma_scalar as super::GradientRegionStatsFn,
            ),
        ];
        for &(simd, scalar) in &kernels {
            for &(px, py, w, h) in &[(1, 2, 8, 16), (3, 4, 16, 8), (20, 21, 8, 16)] {
                let got = simd(&opsin, px, py, w, h, 1e-5);
                let expected = scalar(&opsin, px, py, w, h, 1e-5);
                assert!((got.dominance - expected.dominance).abs() < 2e-5);
                assert!((got.mean - expected.mean).abs() < 2e-6);
                assert!((got.chroma - expected.chroma).abs() < 2e-6);
            }
        }
    }
}
