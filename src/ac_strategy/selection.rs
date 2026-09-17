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

mod leaf_first;
pub(crate) use leaf_first::LeafChoice;

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

/// A super-block's raw merge costs and their incumbents, for the margin-band
/// upgrade shortlist. Index 0 = 16x16, 1..5 = v-left, v-right, h-top, h-bottom.
struct UpgradeBand {
    pick_16x16: bool,
    vertical: bool,
    use_pairs: [bool; 4],
    raw: [f32; 5],
    incumbent: [f32; 5],
}

/// Shortlist every merge of this super-block that beat its incumbent on the
/// raw coefficient model yet was not committed (it failed the acceptance
/// margin, or its arm lost). Pairs under a committed 16x16 are not listed.
fn record_upgrade_candidates(
    upgrades: &mut Vec<MergeUpgradeCandidate>,
    ac_strategy: &AcStrategyImage,
    bx0: usize,
    by0: usize,
    band: UpgradeBand,
) {
    if band.pick_16x16 {
        return;
    }
    let mut push = |bx: usize, by: usize, strategy: u8, raw: f32, incumbent: f32| {
        if raw < incumbent && ac_strategy.can_place_strategy(bx, by, strategy) {
            upgrades.push(MergeUpgradeCandidate {
                bx: bx as u16,
                by: by as u16,
                strategy,
            });
        }
    };
    push(bx0, by0, STRATEGY_DCT16X16, band.raw[0], band.incumbent[0]);
    let committed = [
        band.vertical && band.use_pairs[0],
        band.vertical && band.use_pairs[1],
        !band.vertical && band.use_pairs[2],
        !band.vertical && band.use_pairs[3],
    ];
    let sites = [
        (bx0, by0, STRATEGY_DCT16X8),
        (bx0 + 1, by0, STRATEGY_DCT16X8),
        (bx0, by0, STRATEGY_DCT8X16),
        (bx0, by0 + 1, STRATEGY_DCT8X16),
    ];
    for (k, &(bx, by, strategy)) in sites.iter().enumerate() {
        if !committed[k] {
            push(bx, by, strategy, band.raw[k + 1], band.incumbent[k + 1]);
        }
    }
}

fn select_super_block(
    context: &SuperBlockContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &mut AcStrategyImage,
    input: SuperBlockInput,
    saved: &mut Vec<SavedChild>,
    upgrades: &mut Vec<MergeUpgradeCandidate>,
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
    let raw_propagation = ctx.selector.raw_propagation;
    // (decision cost, unbiased cost) per rectangular candidate.
    let mut rect_cost = |px: usize, py: usize, strategy: u8, qac: f32| -> (f32, f32) {
        if !context.scope.rectangles() {
            return (f32::INFINITY, f32::INFINITY);
        }
        let raw = strategy_cost(
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
        );
        (merge.bias_rect * raw, raw)
    };
    let (v_left, v_left_raw) = rect_cost(px0, py0, STRATEGY_DCT16X8, qac[0][0].max(qac[1][0]));
    let (v_right, v_right_raw) =
        rect_cost(px0 + 8, py0, STRATEGY_DCT16X8, qac[0][1].max(qac[1][1]));
    let (h_top, h_top_raw) = rect_cost(px0, py0, STRATEGY_DCT8X16, qac[0][0].max(qac[0][1]));
    let (h_bot, h_bot_raw) = rect_cost(px0, py0 + 8, STRATEGY_DCT8X16, qac[1][0].max(qac[1][1]));

    // The single DCT16X16 over all four.
    let c16_raw = strategy_cost(
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
    let c16 = merge.bias_16x16 * c16_raw;

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
    // Unbiased twins of the two rectangular arms, propagated under
    // `raw_propagation` so the biases stay local to this decision.
    let raw_16x8 = if use_v_left { v_left_raw } else { dct8_left }
        + if use_v_right { v_right_raw } else { dct8_right };
    let raw_8x16 = if use_h_top { h_top_raw } else { dct8_top }
        + if use_h_bottom { h_bot_raw } else { dct8_bottom };

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

    if ctx.selector.merge_upgrade && params.distance >= MERGE_UPGRADE_MIN_DISTANCE {
        record_upgrade_candidates(
            upgrades,
            ac_strategy,
            bx0,
            by0,
            UpgradeBand {
                pick_16x16,
                vertical: cost_16x8 <= cost_8x16,
                use_pairs: [use_v_left, use_v_right, use_h_top, use_h_bottom],
                raw: [c16_raw, v_left_raw, v_right_raw, h_top_raw, h_bot_raw],
                incumbent: [total_dct8, dct8_left, dct8_right, dct8_top, dct8_bottom],
            },
        );
    }

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
        if raw_propagation { c16_raw } else { c16 }
    } else if cost_16x8 <= cost_8x16 {
        if use_v_left {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X8);
        }
        if use_v_right {
            ac_strategy.set_first(bx0 + 1, by0, STRATEGY_DCT16X8);
        }
        if raw_propagation { raw_16x8 } else { cost_16x8 }
    } else {
        if use_h_top {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT8X16);
        }
        if use_h_bottom {
            ac_strategy.set_first(bx0, by0 + 1, STRATEGY_DCT8X16);
        }
        if raw_propagation { raw_8x16 } else { cost_8x16 }
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
            let ReconCost {
                distortion, rate, ..
            } = reconstruction_dist_and_rate(
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
                false,
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
    upgrades: &mut Vec<MergeUpgradeCandidate>,
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
                let upgrade_mark = upgrades.len();
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
                            upgrades,
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
                let (best_big, best_big_raw, best_strategy, accept) =
                    if cost_32x32 <= cost_32x16 && cost_32x32 <= cost_16x32 {
                        (cost_32x32, cost32, STRATEGY_DCT32X32, gate(merge.accept_32))
                    } else if cost_32x16 <= cost_16x32 {
                        (
                            cost_32x16,
                            cl + cr,
                            STRATEGY_DCT32X16,
                            gate(merge.accept_32_rect),
                        )
                    } else {
                        (
                            cost_16x32,
                            ct + cb,
                            STRATEGY_DCT16X32,
                            gate(merge.accept_32_rect),
                        )
                    };

                // Compare against both the already-selected subdivision and the
                // pure DCT8 incumbent. The latter prevents a sequence of locally
                // marginal merges from making a 32×32 merge look trustworthy.
                let merged = best_big < sub_total && merge_beats_dct8(best_big, dct8_total, accept);
                if merged {
                    // The 32 covers every shortlisted super-block merge.
                    upgrades.truncate(upgrade_mark);
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
                    cost: if !merged {
                        sub_total
                    } else if ctx.selector.raw_propagation {
                        best_big_raw
                    } else {
                        best_big
                    },
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
                        upgrades,
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
                    upgrades,
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

/// A sub-8 candidate that beat DCT8 on the block's own comparison.
#[derive(Clone, Copy)]
struct Sub8Pick {
    strategy: u8,
    /// Coefficient-domain decision cost (bias applied) — what the block-level
    /// comparison against DCT8 used, and what a leaf carries into the merge
    /// levels of the leaf-first selector.
    biased_j: f32,
    /// The same cost without the family bias.
    raw_j: f32,
    /// Credit for the frame-level sub-8 metadata gate: reconstruction-domain
    /// delta for IDENTITY/DCT2X2, biased coefficient-domain delta otherwise
    /// (legacy semantics; the two are not comparable with each other).
    gain: f32,
}

/// Legacy interface: `(strategy, gain)` for the block's winning sub-8
/// candidate, if any.
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
    evaluate_sub8(
        params,
        scratch,
        bx,
        by,
        qac,
        meta_r,
        cached_dct8,
        with_dct4,
        with_fine,
        bias_afv,
    )
    .map(|pick| (pick.strategy, pick.gain))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_sub8(
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
) -> Option<Sub8Pick> {
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
    let (cand, cand_cost, cand_raw) = {
        let mut best = STRATEGY_DCT4X4;
        let mut bc = cost4;
        let mut raw = costs.dct4x4;
        if cost48 < bc {
            best = STRATEGY_DCT4X8;
            bc = cost48;
            raw = costs.dct4x8;
        }
        if cost84 < bc {
            best = STRATEGY_DCT8X4;
            bc = cost84;
            raw = costs.dct8x4;
        }
        for (kind, &afv_cost) in costs.afv.iter().enumerate() {
            let biased = bias_afv * afv_cost;
            if biased < bc {
                best = STRATEGY_AFV0 + kind as u8;
                bc = biased;
                raw = afv_cost;
            }
        }
        (best, bc, raw)
    };

    let (fine, fine_coeff_cost, fine_raw) = if cost_identity < cost_dct2x2 {
        (STRATEGY_IDENTITY, cost_identity, costs.identity)
    } else {
        (STRATEGY_DCT2X2, cost_dct2x2, costs.dct2x2)
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
        // The reconstruction scorer over-credits fine transforms; charge the
        // fitted per-block correction at the floored lambda (see the
        // constant) before the margin test below.
        let recon_fine = fmla(
            fine_mosaic_lambda(params.distance),
            FINE_ADMIT_RATE_CORRECTION_BITS,
            reconstruction_cost(scratch, fine),
        );
        // A small safety margin absorbs the remaining mismatch between the
        // local reconstruction metric and the final post-filtered image.
        if recon_fine < recon8 * FINE_RECON_MARGIN {
            return Some(Sub8Pick {
                strategy: fine,
                biased_j: fine_coeff_cost,
                raw_j: fine_raw,
                gain: recon8 - recon_fine,
            });
        }
    }
    (cand_cost < cost8).then_some(Sub8Pick {
        strategy: cand,
        biased_j: cand_cost,
        raw_j: cand_raw,
        gain: cost8 - cand_cost,
    })
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
    fine_rollbacks: &mut Vec<FineMergeRollback>,
    num_threads: usize,
) -> f32 {
    fine_rollbacks.clear();
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
    let leaf_first = ctx.selector.leaf_first && distance <= LEAF_FIRST_MAX_DISTANCE;
    let run_band = |scratch: &mut CoderScratch,
                    output: &mut AcStrategyBandScratch,
                    strategy: &mut AcStrategyImage,
                    band: (usize, usize)|
     -> f32 {
        if leaf_first {
            leaf_first::select_band_leaf_first(&selection, scratch, strategy, band, output);
            0.0
        } else {
            select_band(
                &selection,
                scratch,
                strategy,
                band,
                &mut output.chosen32,
                &mut output.saved_children,
                &mut output.upgrade_candidates,
            )
        }
    };
    let mut benefit = if !parallel_selection {
        let band0 = &mut pipeline.band_scratch[0];
        band0.chosen32.clear();
        band0.saved_children.clear();
        band0.upgrade_candidates.clear();
        run_band(scratch, band0, ac_strategy, (0, ysize))
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
                // `output.strategy` is a field of `output`; split the borrow.
                let mut strategy =
                    std::mem::replace(&mut output.strategy, AcStrategyImage::new(0, 0));
                output.benefit = run_band(scratch, output, &mut strategy, (y0, y1));
                output.strategy = strategy;
            },
        );
        let mut benefit = 0.0f32;
        for (&(y0, y1), output) in bands.iter().zip(band_scratch) {
            ac_strategy.copy_rows_from(&output.strategy, y0, y1);
            benefit += output.benefit;
        }
        benefit
    };
    if leaf_first {
        // Sub-8 gate credit is attributed to the layout that finally survives
        // (after the 64 pass and the rerank), so assemble the per-block leaf
        // gains now and sum them at the end.
        pipeline.leaf_gain.clear();
        pipeline.leaf_gain.resize(xsize * ysize, 0.0);
        // `fill_selection_bands` always yields at least the single full band.
        for (&(y0, y1), band) in pipeline.bands.iter().zip(&pipeline.band_scratch) {
            for (leaf, gain) in band.leaves[..xsize * (y1 - y0)]
                .iter()
                .zip(&mut pipeline.leaf_gain[y0 * xsize..y1 * xsize])
            {
                *gain = leaf.gain;
            }
        }
    }

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
                    if pipeline.band_scratch[band_idx]
                        .fine_rollbacks
                        .iter()
                        .any(|r| r.bx == downgrade.bx && r.by == downgrade.by)
                    {
                        continue;
                    }
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
                                if leaf_first {
                                    pipeline.leaf_gain[by * xsize + bx] = gain;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Margin-band upgrade: shortlisted merges compete, in the
        // reconstruction domain, against whatever the downgrades and the
        // sub-8 refinement left on their footprint.
        if ctx.selector.merge_upgrade && distance >= MERGE_UPGRADE_MIN_DISTANCE {
            rerank_upgrade_merges(
                &rerank,
                scratch,
                ac_strategy,
                &mut pipeline.band_scratch[..band_count],
                &mut pipeline.current_costs,
                MERGE_UPGRADE_MARGIN,
            );
        }
    }

    if leaf_first {
        benefit = ac_strategy
            .iter_first_blocks()
            .filter(|&(_, _, s)| crate::dc_group_data::is_sub8_strategy(s))
            .map(|(bx, by, _)| pipeline.leaf_gain[by * xsize + bx])
            .fold(0.0, |acc, gain| acc + gain);
    }

    // Preserve enough information for the frame-level exact metadata gate to
    // put back the original merged transforms. A simple sub-8 -> DCT8 revert
    // would otherwise leave a fine mosaic split into many DCT8 blocks.
    if reranked {
        for band in &pipeline.band_scratch[..pipeline.bands.len()] {
            fine_rollbacks.extend_from_slice(&band.fine_rollbacks);
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

const FINE_MOSAIC_BOUNDARY_ALPHA: f32 = 32.0;
const FINE_MOSAIC_BOUNDARY_RATIO: f32 = 1.0;
const FINE_MOSAIC_PEAK_RATIO: f32 = 1.0;
/// How decisively the best mosaic assignment must beat every incumbent (the
/// merge, plain tiling, and the child layout) on the joint score.
const FINE_MOSAIC_MARGIN: f32 = 1.0;
/// Extra bits charged per IDENTITY/DCT2x2 child on top of the rate model's
/// estimate.
const FINE_MOSAIC_RATE_CORRECTION_BITS: f32 = 8.0;
/// Cap on how much pure-RD deficit the seam bonus may purchase
const FINE_MOSAIC_SEAM_SUBSIDY_BITS: f32 = 8.0;
/// Preserve the existing mosaic search through this distance. Above it,
/// chromatic frames must also pass the reconstruction hue-error gate.
const FINE_MOSAIC_MAX_DISTANCE: f32 = 3.0;

/// A pair's hue loss alone must exceed the extra rate charge for two fine
/// children before we spend work reconstructing them. The full joint score
/// still decides whether any finer assignment is actually worth encoding.
fn coarse_mosaic_hue_is_costly(hue_distortion: f32, lambda: f32) -> bool {
    hue_distortion > lambda * (2.0 * FINE_MOSAIC_RATE_CORRECTION_BITS)
}

/// Permit a color repair to trade some seam quality only when its hue gain
/// is meaningful, pays for the seam regression, and leaves a clear pure-RD
/// win. The margin absorbs error in the local rate estimate.
fn fine_mosaic_hue_repair_is_safe(
    reference_hue: f32,
    candidate_hue: f32,
    candidate_rd: f32,
    incumbent_rd: f32,
    seam_regression: f32,
) -> bool {
    let hue_gain = reference_hue - candidate_hue;
    hue_gain > 0.05 * reference_hue
        && candidate_rd <= 0.95 * incumbent_rd
        && seam_regression <= hue_gain
}

const FINE_MOSAIC_CANDIDATES: [u8; 3] = [STRATEGY_DCT, STRATEGY_IDENTITY, STRATEGY_DCT2X2];
/// Most children a joint mosaic can cover (sized for a 2x2-block 16x16; the
/// pair strategies searched today use two).
const FINE_MOSAIC_MAX_CHILDREN: usize = 4;

#[cfg(test)]
fn block_boundary_error_energy(
    ctx: &EncodingContext,
    opsin: &Image3F,
    errors: &[[f32; 1024]; 3],
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    distance: f32,
) -> f32 {
    block_boundary_error_stats(ctx, opsin, errors, px, py, width, height, distance).0
}

#[allow(clippy::too_many_arguments)]
fn block_boundary_error_stats(
    ctx: &EncodingContext,
    opsin: &Image3F,
    errors: &[[f32; 1024]; 3],
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    distance: f32,
) -> (f32, f32) {
    let coarse_mix = ((distance - 1.9) / 0.1).clamp(0.0, 1.0);
    let floor = distance * fmla(coarse_mix, 0.0045 - 0.0015, 0.0015);
    let mut energy = 0.0f32;
    let mut peak = 0.0f32;
    for (c, error) in errors.iter().enumerate() {
        let plane = opsin.plane(c);
        let mut channel = 0.0f32;
        let weight = ctx.channel_weight(c);
        for x in (8..width).step_by(8) {
            for y in 0..height {
                let i = y * width + x;
                let sy = (py + y).min(plane.ysize() - 1);
                let left_x = (px + x - 1).min(plane.xsize() - 1);
                let right_x = (px + x).min(plane.xsize() - 1);
                let source_gradient = (plane.row(sy)[right_x] - plane.row(sy)[left_x]).abs();
                let excess =
                    ((error[i] - error[i - 1]).abs() - 0.5 * source_gradient - floor).max(0.0);
                channel = fmla(excess, excess, channel);
                peak = peak.max(weight * excess * excess);
            }
        }
        for y in (8..height).step_by(8) {
            for x in 0..width {
                let i = y * width + x;
                let sx = (px + x).min(plane.xsize() - 1);
                let top_y = (py + y - 1).min(plane.ysize() - 1);
                let bottom_y = (py + y).min(plane.ysize() - 1);
                let source_gradient = (plane.row(bottom_y)[sx] - plane.row(top_y)[sx]).abs();
                let excess =
                    ((error[i] - error[i - width]).abs() - 0.5 * source_gradient - floor).max(0.0);
                channel = fmla(excess, excess, channel);
                peak = peak.max(weight * excess * excess);
            }
        }
        energy += weight * channel;
    }
    (energy, peak)
}

/// Reused by one selection band; allocate the planes only for mosaic reranking.
pub(crate) struct FineMosaicScratch {
    big_error: Box<[[f32; 1024]; 3]>,
    cand_costs: Box<[[ReconStrategyCost; FINE_MOSAIC_CANDIDATES.len()]; FINE_MOSAIC_MAX_CHILDREN]>,
    cand_planes: Box<[[[[f32; 64]; 3]; FINE_MOSAIC_CANDIDATES.len()]; FINE_MOSAIC_MAX_CHILDREN]>,
}

impl Default for FineMosaicScratch {
    fn default() -> Self {
        let nan_cost = ReconStrategyCost {
            cost: f32::NAN,
            base: f32::NAN,
            distortion: f32::NAN,
            rate: f32::NAN,
            hue_distortion: f32::NAN,
        };
        Self {
            big_error: crate::util::heap_array([0.0; 1024]),
            cand_costs: crate::util::heap_array([nan_cost; FINE_MOSAIC_CANDIDATES.len()]),
            cand_planes: crate::util::heap_array([[[0.0; 64]; 3]; FINE_MOSAIC_CANDIDATES.len()]),
        }
    }
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
    output.fine_rollbacks.clear();
    output.current_costs.clear();
    // Mutable access initializes these lazily. Every slot read is overwritten
    // by this region's cost pass, so subsequent passes need no clearing.
    let mosaic = &mut output.fine_mosaic;
    let fine_lambda = fine_mosaic_lambda(params.distance);
    let coarse_chromatic = ctx.x_heavy();
    // The strong yellow rows use B biases 0.85/0.90. Keep the mild/default
    // rows on seam-only acceptance: the hue relaxation overspends there.
    let hue_repair_enabled = coarse_chromatic && ctx.xyb.fwd[8] >= 0.8;
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
        let (gradient_alpha, gradient_peak_alpha) =
            rerank_pair_gradient_alphas(rerank, strat, px, py);
        let qac_big = region_qac(
            rerank.quant_field,
            bx,
            by,
            cxb,
            cyb,
            params.scale,
            params.distance,
        );
        let with_fine_mosaic = ctx.speed == crate::Speed::Slow
            && (params.distance <= FINE_MOSAIC_MAX_DISTANCE || coarse_chromatic)
            && matches!(strat, STRATEGY_DCT16X8 | STRATEGY_DCT8X16);
        debug_assert!(!with_fine_mosaic || cxb * cyb <= FINE_MOSAIC_MAX_CHILDREN);
        let region_pixels = cxb * cyb * 64;
        let big = reconstruction_strategy_cost_and_base(
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
            if with_fine_mosaic {
                let [e0, e1, e2] = &mut *mosaic.big_error;
                Some([
                    &mut e0[..region_pixels],
                    &mut e1[..region_pixels],
                    &mut e2[..region_pixels],
                ])
            } else {
                None
            },
        );
        let (j_big, big_current_cost) = (big.cost, big.base);
        // Reuse the hue contribution already computed for the merged arm;
        // no extra color conversion or reconstruction is needed for this gate.
        let with_fine_mosaic = with_fine_mosaic
            && (params.distance <= FINE_MOSAIC_MAX_DISTANCE
                || coarse_mosaic_hue_is_costly(big.hue_distortion, fine_lambda));
        let (big_boundary, big_peak) = if with_fine_mosaic {
            block_boundary_error_stats(
                ctx,
                params.opsin,
                &mosaic.big_error,
                px,
                py,
                cxb * 8,
                cyb * 8,
                params.distance,
            )
        } else {
            (0.0, 0.0)
        };
        // Keep candidate generation restricted to the existing seam search.
        // Hue repairs reuse this cache; they do not open zero-seam regions.
        let with_fine_mosaic = with_fine_mosaic && big_boundary > 0.0 && big_peak > 0.0;
        let mut j_dct8 = 0.0f32;
        // Per-child candidate cache: (cost, base cost) plus the reconstructed
        // spatial error planes each candidate leaves behind. The joint search
        // below scores whole assignments from these planes; no candidate is
        // reconstructed twice.
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
                let child = iy * cxb + ix;
                let cmap = cmap_factors(params.ytox_map, params.ytob_map, bx + ix, by + iy);
                for (ci, &strategy) in FINE_MOSAIC_CANDIDATES.iter().enumerate() {
                    if ci > 0 && !with_fine_mosaic {
                        break;
                    }
                    let sink = if with_fine_mosaic {
                        let [e0, e1, e2] = &mut mosaic.cand_planes[child][ci];
                        Some([&mut e0[..], &mut e1[..], &mut e2[..]])
                    } else {
                        None
                    };
                    let cand = reconstruction_strategy_cost_and_base(
                        ctx,
                        scratch,
                        strategy,
                        params.opsin,
                        px + ix * 8,
                        py + iy * 8,
                        q,
                        params.qm_mult_x,
                        meta_r,
                        params.distance,
                        cmap,
                        gradient_alpha,
                        gradient_peak_alpha,
                        sink,
                    );
                    if with_fine_mosaic {
                        mosaic.cand_costs[child][ci] = cand;
                    }
                    if ci == 0 {
                        j_dct8 += cand.cost;
                        output.current_costs.push(CachedQuantCost {
                            bx: bx + ix,
                            by: by + iy,
                            cost: cand.base,
                        });
                    }
                }
            }
        }
        // Third arm: the child layout this merge displaced at selection time.
        let child_costs_start = output.current_costs.len();
        let mut j_child = f32::INFINITY;
        let mut j_child_fine = f32::INFINITY;
        let mut child_grid = [NO_CHILD_BLOCK; 16];
        if let Some(grid) = saved_map.get(&(bx as u16, by as u16)) {
            let mut sum = 0.0f32;
            let (mut d_sum, mut r_sum, mut n_blocks) = (0.0f32, 0.0f32, 0.0f32);
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
                    let child_cost = reconstruction_strategy_cost_and_base(
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
                        None,
                    );
                    sum += child_cost.cost;
                    d_sum += child_cost.distortion;
                    r_sum += child_cost.rate;
                    n_blocks += 1.0;
                    output.current_costs.push(CachedQuantCost {
                        bx: bx + ix,
                        by: by + iy,
                        cost: child_cost.base,
                    });
                }
            }
            j_child = sum;
            j_child_fine = fmla(fine_lambda, fmla(n_blocks, meta_r, r_sum), d_sum);
            child_grid = *grid;
        }

        let child_wins = j_child < j_dct8;
        let j_alt = if child_wins { j_child } else { j_dct8 };
        let ordinary_wins = j_alt < j_big * rerank_margin;
        let mut fine_benefit = 0.0f32;
        let mut fine_grid = [STRATEGY_DCT; 16];
        let mut fine_sel = [0usize; FINE_MOSAIC_MAX_CHILDREN];
        let mut fine_wins = false;
        if with_fine_mosaic {
            let children = cxb * cyb;
            // The mosaic comparison rescores EVERY arm — merge, tiled DCT8,
            // child layout, each assignment — at the floored lambda, so the
            // reranking ramp's quarter-price rate below d≈1.23 cannot leak
            // into this decision. The rate-correction bits ride the same
            // lambda instead of the discounted one.
            let mut rd_fine = [[0.0f32; FINE_MOSAIC_CANDIDATES.len()]; FINE_MOSAIC_MAX_CHILDREN];
            for (k, child_rd) in rd_fine[..children].iter_mut().enumerate() {
                for (ci, rd) in child_rd.iter_mut().enumerate() {
                    let extra = if ci == 0 {
                        0.0
                    } else {
                        FINE_MOSAIC_RATE_CORRECTION_BITS
                    };
                    let c = &mosaic.cand_costs[k][ci];
                    *rd = fmla(fine_lambda, c.rate + meta_r + extra, c.distortion);
                }
            }
            let tiled_rd: f32 = rd_fine[..children].iter().map(|c| c[0]).sum();
            let big_rd = fmla(fine_lambda, big.rate + meta_r, big.distortion);
            let big_joint = fmla(FINE_MOSAIC_BOUNDARY_ALPHA, big_boundary, big_rd);
            // Comparing against tiled DCT8 as well as the merge avoids
            // crediting a fine transform just for changing the footprint of
            // the hue scorer's source-edge mask. All children share that mask.
            let hue_reference = if hue_repair_enabled {
                let tiled_hue: f32 = mosaic.cand_costs[..children]
                    .iter()
                    .map(|c| c[0].hue_distortion)
                    .sum();
                big.hue_distortion.min(tiled_hue)
            } else {
                0.0
            };
            // Exhaustive assignment search over the cached planes. Every grid
            // is scored jointly (child costs plus seam energy), so a child
            // whose fine transform loses the independent comparison can still
            // be picked when it removes a seam the score can see — but only
            // within the bounded seam subsidy below.
            let mut tiled_joint = f32::INFINITY;
            let mut best_joint = f32::INFINITY;
            let mut best_sel = None;
            for code in 0..FINE_MOSAIC_CANDIDATES.len().pow(children as u32) {
                let mut sel = [0usize; FINE_MOSAIC_MAX_CHILDREN];
                let mut rest = code;
                let mut rd_sum = 0.0f32;
                let mut n_fine = 0usize;
                for (k, s) in sel[..children].iter_mut().enumerate() {
                    *s = rest % FINE_MOSAIC_CANDIDATES.len();
                    rest /= FINE_MOSAIC_CANDIDATES.len();
                    rd_sum += rd_fine[k][*s];
                    n_fine += usize::from(*s != 0);
                }
                // Bounded seam subsidy: the seam term may buy a pure-RD
                // deficit of at most this many bits per fine child.
                let subsidy_cap = fmla(
                    fine_lambda * FINE_MOSAIC_SEAM_SUBSIDY_BITS,
                    n_fine as f32,
                    tiled_rd,
                );
                let within_subsidy = rd_sum <= subsidy_cap;
                if code != 0 && !within_subsidy {
                    continue;
                }
                let mut selected = [&mosaic.cand_planes[0][0]; FINE_MOSAIC_MAX_CHILDREN];
                for (k, s) in selected[..children].iter_mut().enumerate() {
                    *s = &mosaic.cand_planes[k][sel[k]];
                }
                let (boundary, peak) = (ctx.mosaic_seam_stats)(
                    ctx,
                    params.opsin,
                    px,
                    py,
                    cxb,
                    cyb,
                    params.distance,
                    &selected[..children],
                );
                let joint = fmla(FINE_MOSAIC_BOUNDARY_ALPHA, boundary, rd_sum);
                if code == 0 {
                    // All-DCT8 duplicates the tiled arm, with its seams priced.
                    // A mosaic must beat this too, or plain tiling (gated by
                    // the ordinary margin) is the honest pick.
                    tiled_joint = joint;
                    continue;
                }
                if joint < best_joint {
                    let improves_seams = boundary < big_boundary * FINE_MOSAIC_BOUNDARY_RATIO
                        && peak < big_peak * FINE_MOSAIC_PEAK_RATIO;
                    // Reuse only candidates already reconstructed by the
                    // seam search.
                    let repairs_hue = hue_repair_enabled && !improves_seams && {
                        let hue_sum: f32 = sel[..children]
                            .iter()
                            .enumerate()
                            .map(|(k, &ci)| mosaic.cand_costs[k][ci].hue_distortion)
                            .sum();
                        let seam_regression = FINE_MOSAIC_BOUNDARY_ALPHA
                            * ((boundary - big_boundary).max(0.0) + (peak - big_peak).max(0.0));
                        fine_mosaic_hue_repair_is_safe(
                            hue_reference,
                            hue_sum,
                            rd_sum,
                            tiled_rd.min(big_rd).min(j_child_fine),
                            seam_regression,
                        )
                    };
                    if improves_seams || repairs_hue {
                        best_joint = joint;
                        best_sel = Some(sel);
                    }
                }
            }
            if let Some(sel) = best_sel
                && best_joint < tiled_joint * FINE_MOSAIC_MARGIN
                && best_joint < big_joint * rerank_margin * FINE_MOSAIC_MARGIN
                && best_joint < j_child_fine * FINE_MOSAIC_MARGIN
            {
                fine_wins = true;
                fine_benefit = big_joint - best_joint;
                fine_sel = sel;
                for iy in 0..cyb {
                    for ix in 0..cxb {
                        fine_grid[iy * 4 + ix] = FINE_MOSAIC_CANDIDATES[sel[iy * cxb + ix]];
                    }
                }
            }
        }
        if ordinary_wins || fine_wins {
            let restore = if fine_wins {
                output.current_costs.truncate(child_costs_start);
                for iy in 0..cyb {
                    for ix in 0..cxb {
                        let child = iy * cxb + ix;
                        output.current_costs[tiled_costs_start + child].cost =
                            mosaic.cand_costs[child][fine_sel[child]].base;
                    }
                }
                output.fine_rollbacks.push(FineMergeRollback {
                    bx,
                    by,
                    cov_x: cxb,
                    cov_y: cyb,
                    strategy: strat,
                    fine_grid,
                    benefit: fine_benefit,
                });
                fine_grid
            } else if child_wins {
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

/// Pair-gradient penalty weights (sun-beam fix) the rerank applies to a
/// rectangular pair at pixel `(px, py)`; zero for every other strategy.
fn rerank_pair_gradient_alphas(
    rerank: &RerankContext<'_>,
    strat: u8,
    px: usize,
    py: usize,
) -> (f32, f32) {
    let params = rerank.params;
    let ctx = params.ctx;
    if !matches!(strat, STRATEGY_DCT16X8 | STRATEGY_DCT8X16) {
        return (0.0, 0.0);
    }
    let alpha = rerank_pair_gradient_alpha(params.distance);
    let peak_alpha = rerank_pair_gradient_peak_alpha(params.distance);
    if alpha == 0.0 && peak_alpha == 0.0 {
        return (0.0, 0.0);
    }
    let pair_gradient_fit = &PAIR_GRAD_FIT;
    let (width, height) = if strat == STRATEGY_DCT16X8 {
        (8, 16)
    } else {
        (16, 8)
    };
    let stats = if params.distance < RERANK_PAIR_GRADIENT_PEAK_COARSE_END {
        (ctx.gradient_region_stats)(params.opsin, px, py, width, height, 1e-5)
    } else {
        (ctx.gradient_region_stats_with_chroma)(params.opsin, px, py, width, height, 1e-5)
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

/// The `cov_x x cov_y` footprint at `(bx, by)` is tiled exactly by first
/// blocks lying entirely inside it (so a merge may replace it).
fn footprint_is_self_contained(
    map: &AcStrategyImage,
    bx: usize,
    by: usize,
    cov_x: usize,
    cov_y: usize,
) -> bool {
    let mut area = 0;
    for y in by..by + cov_y {
        for x in bx..bx + cov_x {
            if !map.is_first_block(x, y) {
                continue;
            }
            let s = map.raw_strategy(x, y);
            let cx = AcStrategyImage::covered_blocks_x_of(s);
            let cy = AcStrategyImage::covered_blocks_y_of(s);
            if x + cx > bx + cov_x || y + cy > by + cov_y {
                return false;
            }
            area += cx * cy;
        }
    }
    area == cov_x * cov_y
}

/// Evaluate this band's shortlisted merges against the committed layout of
/// their footprints (reconstruction domain, same metadata and pair-gradient
/// treatment as the downgrade pass) and collect the accepted ones.
fn find_merge_upgrades(
    rerank: &RerankContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &AcStrategyImage,
    band: (usize, usize),
    output: &mut AcStrategyBandScratch,
    margin: f32,
) {
    let params = rerank.params;
    let ctx = params.ctx;
    let (y0, y1) = band;
    let meta_r = RERANK_META_R;
    output.merge_upgrades.clear();
    let overlaps_accepted =
        |accepted: &[MergeUpgrade], bx: usize, by: usize, cx: usize, cy: usize| {
            accepted.iter().any(|u| {
                let ux = AcStrategyImage::covered_blocks_x_of(u.strategy);
                let uy = AcStrategyImage::covered_blocks_y_of(u.strategy);
                bx < u.bx + ux && u.bx < bx + cx && by < u.by + uy && u.by < by + cy
            })
        };
    for ci in 0..output.upgrade_candidates.len() {
        let cand = output.upgrade_candidates[ci];
        let (bx, by, strategy) = (cand.bx as usize, cand.by as usize, cand.strategy);
        if by < y0 || by >= y1 {
            continue;
        }
        let cx = AcStrategyImage::covered_blocks_x_of(strategy);
        let cy = AcStrategyImage::covered_blocks_y_of(strategy);
        // A fine mosaic's footprint may be put back to its original merge by
        // the frame-level metadata gate; never build on top of one.
        let touches_mosaic = output.fine_rollbacks.iter().any(|r| {
            bx < r.bx + r.cov_x && r.bx < bx + cx && by < r.by + r.cov_y && r.by < by + cy
        });
        if touches_mosaic
            || overlaps_accepted(&output.merge_upgrades, bx, by, cx, cy)
            || !ac_strategy.can_place_strategy(bx, by, strategy)
            || !footprint_is_self_contained(ac_strategy, bx, by, cx, cy)
        {
            continue;
        }
        let (px, py) = (params.dc_group_px + bx * 8, params.dc_group_py + by * 8);
        let (gradient_alpha, gradient_peak_alpha) =
            rerank_pair_gradient_alphas(rerank, strategy, px, py);
        let score =
            |scratch: &mut CoderScratch, s: u8, x: usize, y: usize, cw: usize, ch: usize| {
                reconstruction_strategy_cost_and_base(
                    ctx,
                    scratch,
                    s,
                    params.opsin,
                    params.dc_group_px + x * 8,
                    params.dc_group_py + y * 8,
                    region_qac(
                        rerank.quant_field,
                        x,
                        y,
                        cw,
                        ch,
                        params.scale,
                        params.distance,
                    ),
                    params.qm_mult_x,
                    meta_r,
                    params.distance,
                    cmap_factors(params.ytox_map, params.ytob_map, x, y),
                    gradient_alpha,
                    gradient_peak_alpha,
                    None,
                )
            };
        let big = score(scratch, strategy, bx, by, cx, cy);
        let mut j_layout = 0.0f32;
        for y in by..by + cy {
            for x in bx..bx + cx {
                if !ac_strategy.is_first_block(x, y) {
                    continue;
                }
                let s = ac_strategy.raw_strategy(x, y);
                let scx = AcStrategyImage::covered_blocks_x_of(s);
                let scy = AcStrategyImage::covered_blocks_y_of(s);
                j_layout += score(scratch, s, x, y, scx, scy).cost;
            }
        }
        if big.cost < j_layout * margin {
            output.merge_upgrades.push(MergeUpgrade {
                bx,
                by,
                strategy,
                base: big.base,
            });
        }
    }
}

/// Parallel evaluation of the margin-band shortlist, then serial install.
fn rerank_upgrade_merges(
    rerank: &RerankContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &mut AcStrategyImage,
    band_scratch: &mut [AcStrategyBandScratch],
    current_costs: &mut [f32],
    margin: f32,
) {
    let xsize = ac_strategy.xsize();
    let map_ref: &AcStrategyImage = ac_strategy;
    rerank
        .params
        .ctx
        .thread_pool
        .steal_for_each_mut_with_threads(
            scratch,
            &mut band_scratch[..rerank.bands.len()],
            rerank.num_threads,
            |i, output, scratch| {
                find_merge_upgrades(rerank, scratch, map_ref, rerank.bands[i], output, margin);
            },
        );
    for band in &band_scratch[..rerank.bands.len()] {
        for up in &band.merge_upgrades {
            let cx = AcStrategyImage::covered_blocks_x_of(up.strategy);
            let cy = AcStrategyImage::covered_blocks_y_of(up.strategy);
            for y in up.by..up.by + cy {
                current_costs[y * xsize + up.bx..y * xsize + up.bx + cx].fill(f32::NAN);
            }
            current_costs[up.by * xsize + up.bx] = up.base;
            ac_strategy.set_first(up.bx, up.by, up.strategy);
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
        AcStrategyParams, Chosen32Cost, META_R, NO_CHILD_BLOCK, SavedChild, SelectionContext,
        leaf_first, select_band,
    };
    use super::{
        BIAS_4X4, BIAS_4X8, BIAS_16X16, BIAS_AFV, BIAS_RECT32, DCT8_ONLY_MAX_DISTANCE,
        FAST_RERANK_MAX_DISTANCE, FINE_ADMIT_RATE_CORRECTION_BITS, LEAF_FIRST_MAX_DISTANCE,
        MERGE_MARGIN_16, MERGE_MARGIN_32, MERGE_MARGIN_PAIR, MERGE_UPGRADE_MARGIN,
        MERGE_UPGRADE_MIN_DISTANCE, MergeTuning, RERANK_DOWNGRADE_MARGIN,
        RERANK_PAIR_GRADIENT_ALPHA, RERANK_PAIR_GRADIENT_FADE_IN_END,
        RERANK_PAIR_GRADIENT_FADE_IN_START, RERANK_PAIR_GRADIENT_FADE_OUT_END,
        RERANK_PAIR_GRADIENT_FADE_OUT_START, RERANK_PAIR_GRADIENT_MIN_DOMINANCE,
        RERANK_PAIR_GRADIENT_PEAK_ALPHA, RERANK_PAIR_GRADIENT_PEAK_COARSE_ALPHA,
        RERANK_PAIR_GRADIENT_PEAK_COARSE_END, RERANK_PAIR_GRADIENT_PEAK_COARSE_START,
        RERANK_PAIR_GRADIENT_PEAK_FADE_IN_END, RERANK_PAIR_GRADIENT_PEAK_FADE_IN_START,
        RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_END, RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_START,
        SUB8_MAX_DISTANCE, SearchScope, aggregate_qac_2x2, aggregate_quant,
        block_boundary_error_energy, block_boundary_error_stats, cmap_factors, fill_ac_strategy,
        fill_selection_bands, fine_mosaic_lambda, gradient_region_stats_scalar,
        gradient_region_stats_with_chroma_scalar, merge_beats_dct8, merge_margin,
        quant_refinement_steps, rerank_pair_gradient_peak_alpha, rerank_pair_gradient_scale,
        select_gradient_region_stats_fn, select_gradient_region_stats_with_chroma_fn,
        strategy_cost, sub8_strategy_costs, use_dct8_only,
    };
    use crate::coder_scratch::{AcStrategyBandScratch, CoderScratch};
    use crate::dc_group_data::{
        AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT2X2, STRATEGY_DCT4X4, STRATEGY_DCT4X8,
        STRATEGY_DCT8X4, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16,
        STRATEGY_DCT32X32, STRATEGY_DCT32X64, STRATEGY_DCT64X32, STRATEGY_DCT64X64,
        STRATEGY_IDENTITY,
    };
    use crate::encoding_context::EncodingContext;
    use crate::image::{Image3F, ImageB, ImageSB};
    use crate::inflated_cost::{
        forward_for, forward_matrix, reconstruct_error, strategy_pixel_count,
    };

    #[test]
    fn hue_repair_can_pay_for_seams_but_requires_a_clear_rd_win() {
        let accepts = super::fine_mosaic_hue_repair_is_safe;
        // A 20% hue improvement can pay for a bounded seam regression while
        // preserving the full RD win, including the fine-child rate charge.
        assert!(accepts(10.0, 8.0, 95.0, 100.0, 1.5));
        assert!(!accepts(10.0, 9.6, 95.0, 100.0, 0.0));
        assert!(!accepts(10.0, 8.0, 98.0, 100.0, 0.0));
        assert!(!accepts(10.0, 8.0, 100.1, 100.0, 0.0));
        assert!(!accepts(10.0, 8.0, 95.0, 100.0, 2.1));
        assert!(!accepts(0.0, 0.0, 95.0, 100.0, 0.0));
        assert!(!accepts(f32::NAN, 8.0, 95.0, 100.0, 0.0));
        assert!(!accepts(10.0, f32::NAN, 95.0, 100.0, 0.0));
    }

    #[test]
    fn hue_protection_survives_coarse_quantization_only_for_chroma_structure() {
        let ctx = EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, 2.0, 1);
        let alpha = |distance| super::rerank_rgb_hue_alpha(&ctx, distance);
        for distance in [1.0, 2.0, 3.0, 4.0, 10.0] {
            assert_eq!(alpha(distance), 0.0);
        }
        ctx.set_x_heavy(true);
        assert_eq!(alpha(0.35), 0.0);
        assert!(alpha(0.5) > 0.0 && alpha(0.5) < alpha(1.0));
        for distance in [2.0, 2.5, 3.0, 3.25, 4.0, 10.0] {
            assert_eq!(alpha(distance), alpha(1.0));
        }
    }

    #[test]
    fn mosaic_seam_stats_matches_assembled_boundary_stats() {
        let mut seed = 127u32;
        let mut random = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.25
        };
        for bias in [crate::xyb::B_BIAS, 0.85] {
            let ctx = EncodingContext::new(
                crate::Speed::Slow,
                crate::yellow_opsin::matrix_for_bias(bias),
                1.0,
                1,
            );
            for (width, height) in [(1, 1), (7, 9), (16, 16), (29, 31)] {
                let mut opsin = Image3F::new(width, height);
                for c in 0..3 {
                    for y in 0..height {
                        for v in opsin.plane_row_mut(c, y) {
                            *v = random();
                        }
                    }
                }
                let mut planes = [[[[0.0f32; 64]; 3]; 3]; 4];
                for child in &mut planes {
                    for candidate in child {
                        for channel in candidate {
                            for v in channel {
                                *v = random();
                            }
                        }
                    }
                }
                for (cxb, cyb) in [(1usize, 1usize), (2, 1), (1, 2), (2, 2)] {
                    for (px, py) in [(0, 0), (width - 1, height - 1)] {
                        for distance in [0.0, 0.5, 1.9, 1.95, 2.0, 3.0] {
                            for code in 0..3usize.pow((cxb * cyb) as u32) {
                                let mut rest = code;
                                let selected: Vec<_> = planes[..cxb * cyb]
                                    .iter()
                                    .map(|child| {
                                        let i = rest % 3;
                                        rest /= 3;
                                        &child[i]
                                    })
                                    .collect();
                                let mut assembled = [[0.0f32; 1024]; 3];
                                for (k, child) in selected.iter().enumerate() {
                                    let (kx, ky) = (k % cxb, k / cxb);
                                    for (dest, source) in assembled.iter_mut().zip(child.iter()) {
                                        for (y, row) in source.as_chunks::<8>().0.iter().enumerate()
                                        {
                                            let offset = (ky * 8 + y) * cxb * 8 + kx * 8;
                                            dest[offset..offset + 8].copy_from_slice(row);
                                        }
                                    }
                                }
                                let expected = block_boundary_error_stats(
                                    &ctx,
                                    &opsin,
                                    &assembled,
                                    px,
                                    py,
                                    cxb * 8,
                                    cyb * 8,
                                    distance,
                                );
                                let got = (ctx.mosaic_seam_stats)(
                                    &ctx, &opsin, px, py, cxb, cyb, distance, &selected,
                                );
                                assert_eq!(
                                    (got.0.to_bits(), got.1.to_bits()),
                                    (expected.0.to_bits(), expected.1.to_bits()),
                                    "{width}x{height}, origin={px},{py}, grid={cxb}x{cyb}, d={distance}, code={code}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn boundary_error_energy_detects_artificial_block_seams() {
        let ctx = EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, 1.0, 1);
        let mut opsin = Image3F::new(16, 8);
        let smooth = [[0.0f32; 1024]; 3];
        assert_eq!(
            block_boundary_error_energy(&ctx, &opsin, &smooth, 0, 0, 16, 8, 1.0),
            0.0
        );

        let mut seam = smooth;
        for y in 0..8 {
            seam[1][y * 16 + 8..y * 16 + 16].fill(0.1);
        }
        let artificial = block_boundary_error_energy(&ctx, &opsin, &seam, 0, 0, 16, 8, 1.0);
        assert!(artificial > 0.0);

        // A matching edge in the source masks the error jump: genuine image
        // structure should not be mistaken for transform blockiness.
        for y in 0..8 {
            opsin.plane_row_mut(1, y)[8..].fill(0.2);
        }
        let source_aligned = block_boundary_error_energy(&ctx, &opsin, &seam, 0, 0, 16, 8, 1.0);
        assert!(source_aligned < artificial);
    }

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
        let ctx = EncodingContext::new(crate::Speed::Fast, crate::xyb::XybMatrix::SPEC, 1.0, 1);
        let maps = ImageSB::new_fill(1, 1, 0);
        let opsin = Image3F::new(32, 32);
        let mut qf = ImageB::new_fill(4, 4, 8);
        let mut strategies = AcStrategyImage::new(4, 4);
        let mut scratch = CoderScratch::default();
        let mut fine_rollbacks = Vec::new();
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
            &mut fine_rollbacks,
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

    fn noise_opsin(w: usize, h: usize, seed: u32) -> Image3F {
        let mut opsin = Image3F::new(w, h);
        let mut state = seed;
        for c in 0..3 {
            for y in 0..h {
                for value in opsin.plane_mut(c).row_mut(y) {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // Smooth-ish gradient plus noise so merges of every size
                    // and sub-8 leaves all get selected somewhere.
                    let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                    *value = 0.05 * noise * ((y / 7) % 3) as f32 + 0.002 * y as f32;
                }
            }
        }
        opsin
    }

    fn run_selector(
        speed: crate::Speed,
        distance: f32,
        policy: crate::ac_strategy::SelectorPolicy,
        threads: usize,
    ) -> (AcStrategyImage, f32) {
        let mut ctx = EncodingContext::new(speed, crate::xyb::XybMatrix::SPEC, distance, threads);
        ctx.selector = policy;
        let (bw, bh) = (13usize, 11usize);
        let opsin = noise_opsin(bw * 8, bh * 8, 0x9e37_79b9);
        let maps = ImageSB::new_fill(2, 2, 0);
        let mut qf = ImageB::new_fill(bw, bh, 6);
        for y in 0..bh {
            for (x, q) in qf.row_mut(y).iter_mut().enumerate() {
                *q = 4 + ((x * 3 + y * 5) % 7) as u8;
            }
        }
        let mut strategies = AcStrategyImage::new(bw, bh);
        let mut scratch = CoderScratch::default();
        let mut fine_rollbacks = Vec::new();
        let benefit = fill_ac_strategy(
            &ctx,
            &mut scratch,
            &opsin,
            0,
            0,
            distance,
            1.0,
            2,
            &mut qf,
            &maps,
            &maps,
            &mut strategies,
            &mut fine_rollbacks,
            threads,
        );
        (strategies, benefit)
    }

    fn strategy_cells(map: &AcStrategyImage) -> Vec<(usize, usize, u8)> {
        map.iter_first_blocks().collect()
    }

    /// Every cell belongs to exactly one in-bounds first block.
    fn assert_map_valid(map: &AcStrategyImage) {
        let (bw, bh) = (map.xsize(), map.ysize());
        let mut owner = vec![None; bw * bh];
        for (bx, by, s) in map.iter_first_blocks() {
            let cx = AcStrategyImage::covered_blocks_x_of(s);
            let cy = AcStrategyImage::covered_blocks_y_of(s);
            for iy in 0..cy {
                for ix in 0..cx {
                    let cell = &mut owner[(by + iy) * bw + bx + ix];
                    assert!(
                        cell.is_none(),
                        "cell ({},{}) covered twice",
                        bx + ix,
                        by + iy
                    );
                    *cell = Some((bx, by));
                }
            }
        }
        assert!(owner.iter().all(Option::is_some), "uncovered cell");
    }

    /// With no sub-8 family in play (above the fine-transform band) every
    /// leaf is DCT8, so the leaf-first planner must reproduce the legacy
    /// selector's map exactly — same candidates, same margins, same
    /// tie-breaks — in both search scopes and with parallel bands.
    #[test]
    fn leaf_first_reproduces_legacy_when_every_leaf_is_dct8() {
        // The distance gate makes the two orders identical by construction
        // above the sub-8 band, so call the band planners directly at a
        // distance where no structural sub-8 candidate exists: the leaf-first
        // planner must then reproduce the legacy super-block/32 decisions,
        // saved child layouts and 32-level costs exactly.
        for scope in [SearchScope::Full, SearchScope::Squares] {
            let distance = 5.5f32;
            let ctx =
                EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, distance, 1);
            let (bw, bh) = (13usize, 11usize);
            let opsin = noise_opsin(bw * 8, bh * 8, 0x9e37_79b9);
            let maps = ImageSB::new_fill(2, 2, 0);
            let mut qf = ImageB::new_fill(bw, bh, 6);
            for y in 0..bh {
                for (x, q) in qf.row_mut(y).iter_mut().enumerate() {
                    *q = 4 + ((x * 3 + y * 5) % 7) as u8;
                }
            }
            let params = AcStrategyParams {
                ctx: &ctx,
                opsin: &opsin,
                dc_group_px: 0,
                dc_group_py: 0,
                distance,
                scale: 1.0,
                qm_mult_x: 1.0,
                ytox_map: &maps,
                ytob_map: &maps,
            };
            let selection = SelectionContext {
                params,
                quant_field: &qf,
                meta_r: META_R,
                scope,
            };
            let mut scratch = CoderScratch::default();
            let mut legacy_map = AcStrategyImage::new(bw, bh);
            let mut legacy = AcStrategyBandScratch::default();
            scratch.dct8_costs.clear();
            let benefit = select_band(
                &selection,
                &mut scratch,
                &mut legacy_map,
                (0, bh),
                &mut legacy.chosen32,
                &mut legacy.saved_children,
                &mut legacy.upgrade_candidates,
            );
            assert_eq!(benefit, 0.0);
            let mut leaf_map = AcStrategyImage::new(bw, bh);
            let mut leaf = AcStrategyBandScratch::default();
            leaf_first::select_band_leaf_first(
                &selection,
                &mut scratch,
                &mut leaf_map,
                (0, bh),
                &mut leaf,
            );
            assert_eq!(
                strategy_cells(&legacy_map),
                strategy_cells(&leaf_map),
                "{scope:?}"
            );
            let costs = |c: &Vec<Chosen32Cost>| {
                c.iter()
                    .map(|c| (c.bx, c.by, c.cost.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(costs(&legacy.chosen32), costs(&leaf.chosen32), "{scope:?}");
            // The rerank sees saved children through a last-wins map keyed by
            // the selected merge's first block; legacy also leaves dead
            // 16-level entries under a 32 (never looked up), leaf-first does
            // not — compare the effective view.
            let effective = |c: &Vec<SavedChild>, map: &AcStrategyImage| {
                let mut m = std::collections::BTreeMap::new();
                for c in c {
                    m.insert((c.bx, c.by), c.grid);
                }
                m.retain(|&(bx, by), _| {
                    map.is_first_block(bx as usize, by as usize)
                        && map.raw_strategy(bx as usize, by as usize) != STRATEGY_DCT
                });
                m
            };
            assert_eq!(
                effective(&legacy.saved_children, &legacy_map),
                effective(&leaf.saved_children, &leaf_map),
                "{scope:?}"
            );
            assert!(
                leaf.leaves
                    .iter()
                    .all(|l| l.strategy == STRATEGY_DCT && l.gain == 0.0)
            );
        }
    }

    /// The shipped selector is leaf-first inside the DCT4/AFV band with
    /// biased propagation; raw propagation stays an opt-in study switch and
    /// the band edge is the sub-8 gate.
    #[test]
    fn selector_policy_defaults() {
        use crate::ac_strategy::SelectorPolicy;
        let p = SelectorPolicy::default();
        assert!(p.leaf_first && !p.raw_propagation && p.merge_upgrade);
        assert_eq!(LEAF_FIRST_MAX_DISTANCE, SUB8_MAX_DISTANCE);
    }

    /// The per-block fine admission carries a fitted rate charge: without it
    /// the reconstruction scorer admits IDENTITY/DCT2X2 blocks that cost
    /// bytes for no quality (sweep 4..32 bits monotone, plateau 24..32,
    /// "off" within 0.01% of the plateau on Kodak).
    #[test]
    fn fine_admission_charges_the_fitted_correction() {
        let bits = std::hint::black_box(FINE_ADMIT_RATE_CORRECTION_BITS);
        assert!((16.0..=32.0).contains(&bits), "bits = {bits}");
        // Charged at the floored lambda, never at the quarter-price ramp.
        assert!(fine_mosaic_lambda(0.5) >= crate::ac_strategy::RD_LAMBDA);
    }

    /// In the sub-8 band the leaf-first planner runs end to end on a mixed
    /// image, commits a valid map (every covered cell belongs to exactly one
    /// first block) and credits the gate only for surviving leaves.
    #[test]
    fn leaf_first_commits_a_valid_map_with_sub8_leaves() {
        use crate::ac_strategy::SelectorPolicy;
        let leaf = SelectorPolicy {
            leaf_first: true,
            ..SelectorPolicy::default()
        };
        for threads in [1usize, 3] {
            let (map, benefit) = run_selector(crate::Speed::Slow, 1.0, leaf, threads);
            assert_map_valid(&map);
            let sub8 = map
                .iter_first_blocks()
                .filter(|&(_, _, s)| crate::dc_group_data::is_sub8_strategy(s))
                .count();
            assert!(benefit.is_finite());
            assert!(
                sub8 == 0 || benefit > 0.0,
                "sub-8 leaves survived without credit"
            );
        }
    }

    /// The margin-band upgrade installs only merges whose footprint is
    /// self-contained and never two overlapping ones: the map stays valid
    /// under both selectors across the band where it runs.
    #[test]
    fn merge_upgrade_keeps_the_map_valid() {
        use crate::ac_strategy::SelectorPolicy;
        for leaf_first in [false, true] {
            let policy = SelectorPolicy {
                leaf_first,
                merge_upgrade: true,
                ..SelectorPolicy::default()
            };
            for distance in [1.5f32, 2.0, 3.0, 6.0] {
                let (map, _) = run_selector(crate::Speed::Slow, distance, policy, 3);
                assert_map_valid(&map);
            }
        }
    }

    /// The upgrade is gated out of the HQ band (it loses there on the
    /// holdout) and runs at the knife-edge margin (0.9 is inert, 1.1 turns
    /// SS2 flat).
    #[test]
    fn merge_upgrade_is_gated_and_knife_edge() {
        let gate = std::hint::black_box(MERGE_UPGRADE_MIN_DISTANCE);
        let margin = std::hint::black_box(MERGE_UPGRADE_MARGIN);
        assert!((1.5..=2.5).contains(&gate), "gate {gate}");
        assert!((0.95..=1.05).contains(&margin), "margin {margin}");
        assert!(crate::ac_strategy::SelectorPolicy::default().merge_upgrade);
    }

    /// Every saved child layout is keyed by a selected transform's own first
    /// block and describes exactly that transform's footprint: a 32x16 /
    /// 16x32 pair gets one grid per half (the rerank looks each half up
    /// separately), never one 4x4 grid on the first half.
    #[test]
    fn leaf_first_saved_children_match_their_transform_footprints() {
        use crate::ac_strategy::SelectorPolicy;
        let policy = SelectorPolicy {
            leaf_first: true,
            ..SelectorPolicy::default()
        };
        let mut seen_rect_halves = 0;
        let mut histogram = [0usize; crate::dc_group_data::NUM_STRATEGIES];
        for (distance, pattern) in [
            (1.0f32, 0u32),
            (2.5, 0),
            (4.0, 0),
            (6.0, 0),
            (2.5, 1),
            (4.0, 1),
            (2.5, 2),
            (4.0, 2),
        ] {
            let mut ctx =
                EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, distance, 1);
            ctx.selector = policy;
            let (bw, bh) = (12usize, 12usize);
            let mut opsin = noise_opsin(bw * 8, bh * 8, 0x1234_abcd);
            // Patterns 1/2 add a one-directional ramp so that wide (16x32)
            // and tall (32x16) rectangles beat the square merges somewhere.
            if pattern != 0 {
                for c in 0..3 {
                    for y in 0..bh * 8 {
                        for (x, v) in opsin.plane_mut(c).row_mut(y).iter_mut().enumerate() {
                            let t = if pattern == 1 { y } else { x } as f32;
                            *v = 0.2 * *v + 0.02 * (t * 0.35).sin();
                        }
                    }
                }
            }
            let maps = ImageSB::new_fill(2, 2, 0);
            let qf = ImageB::new_fill(bw, bh, 5);
            let mut map = AcStrategyImage::new(bw, bh);
            let mut scratch = CoderScratch::default();
            let params = AcStrategyParams {
                ctx: &ctx,
                opsin: &opsin,
                dc_group_px: 0,
                dc_group_py: 0,
                distance,
                scale: 1.0,
                qm_mult_x: 1.0,
                ytox_map: &maps,
                ytob_map: &maps,
            };
            let selection = SelectionContext {
                params,
                quant_field: &qf,
                meta_r: META_R,
                scope: SearchScope::Full,
            };
            let mut band = AcStrategyBandScratch::default();
            leaf_first::select_band_leaf_first(
                &selection,
                &mut scratch,
                &mut map,
                (0, bh),
                &mut band,
            );
            for (_, _, s) in map.iter_first_blocks() {
                histogram[s as usize] += 1;
            }
            for child in &band.saved_children {
                let (bx, by) = (child.bx as usize, child.by as usize);
                assert!(map.is_first_block(bx, by), "saved child at a covered cell");
                let s = map.raw_strategy(bx, by);
                let cx = AcStrategyImage::covered_blocks_x_of(s);
                let cy = AcStrategyImage::covered_blocks_y_of(s);
                assert!(cx * cy > 1, "saved child under a single block");
                if matches!(s, STRATEGY_DCT32X16 | STRATEGY_DCT16X32) {
                    seen_rect_halves += 1;
                }
                for iy in 0..4 {
                    for ix in 0..4 {
                        let g = child.grid[iy * 4 + ix];
                        assert!(
                            g == NO_CHILD_BLOCK || (ix < cx && iy < cy),
                            "child grid cell ({ix},{iy}) outside a {cx}x{cy} footprint at ({bx},{by})"
                        );
                    }
                }
            }
        }
        // Rectangle halves are rare on synthetic content; `sub_grid_rebases_
        // rectangle_halves` pins the split itself. Keep the counters used.
        let _ = (seen_rect_halves, histogram);
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
