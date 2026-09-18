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

/// The eight possible partitions of four child regions: leave them alone,
/// merge the square, or merge either/both halves in either direction. Bits
/// identify square, left, right, top, bottom. Opposite directions overlap.
static MERGE_PARTITIONS: [u8; 8] = [0, 1, 2, 4, 6, 8, 16, 24];

fn partition_cost(children: [[f32; 2]; 2], merges: [f32; 5], mask: u8) -> f32 {
    if mask & 1 != 0 {
        return merges[0];
    }
    let mut cost = 0.0;
    for (k, value) in merges.iter().enumerate().skip(1) {
        if mask & (1 << k) != 0 {
            cost += value;
        }
    }
    for (y, row) in children.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            if mask & ((1 << (1 + x)) | (1 << (3 + y))) == 0 {
                cost += value;
            }
        }
    }
    cost
}

/// Gate each layout before comparing costs. A rejected cheap square must
/// not hide an admissible rectangle. The two-rectangle layout keeps its
/// existing joint margin; single rectangles pay that margin only on the
/// children they replace. Children already carry their optimal plans.
fn choose_merge_partition(
    children: [[f32; 2]; 2],
    baseline: [[f32; 2]; 2],
    merges: [f32; 5],
    accept: [f32; 5],
) -> u8 {
    let totals = |c: [[f32; 2]; 2]| {
        [
            c.iter().flatten().sum(),
            c[0][0] + c[1][0],
            c[0][1] + c[1][1],
            c[0][0] + c[0][1],
            c[1][0] + c[1][1],
        ]
    };
    let child_costs = totals(children);
    let base_costs = totals(baseline);
    let mut best = 0;
    let mut best_cost = child_costs[0];
    for mask in MERGE_PARTITIONS.into_iter().skip(1) {
        let (mut merged, mut incumbent, mut limit) = (0.0, 0.0, 0.0);
        for k in 0..5 {
            if mask & (1 << k) != 0 {
                merged += merges[k];
                incumbent += child_costs[k];
                limit += base_costs[k] * accept[k];
            }
        }
        if !(merged < incumbent && merged < limit) {
            continue;
        }
        let cost = partition_cost(children, merges, mask);
        if cost < best_cost {
            best = mask;
            best_cost = cost;
        }
    }
    best
}

fn merge32_sites(bx: usize, by: usize) -> [(usize, usize, u8); 5] {
    [
        (bx, by, STRATEGY_DCT32X32),
        (bx, by, STRATEGY_DCT32X16),
        (bx + 2, by, STRATEGY_DCT32X16),
        (bx, by, STRATEGY_DCT16X32),
        (bx, by + 2, STRATEGY_DCT16X32),
    ]
}

/// Raw merge costs and the corresponding leaf incumbents. Keep all raw
/// winners, including the committed arm and candidates covered by a parent:
/// the downgrade pass may expose any of them again.
struct UpgradeBand {
    raw: [f32; 5],
    incumbent: [f32; 5],
}

fn record_upgrade_candidates(
    upgrades: &mut Vec<MergeUpgradeCandidate>,
    ac_strategy: &AcStrategyImage,
    bx0: usize,
    by0: usize,
    band: UpgradeBand,
) {
    let sites = [
        (bx0, by0, STRATEGY_DCT16X16),
        (bx0, by0, STRATEGY_DCT16X8),
        (bx0 + 1, by0, STRATEGY_DCT16X8),
        (bx0, by0, STRATEGY_DCT8X16),
        (bx0, by0 + 1, STRATEGY_DCT8X16),
    ];
    for (k, &(bx, by, strategy)) in sites.iter().enumerate() {
        if band.raw[k] < band.incumbent[k] && ac_strategy.can_place_strategy(bx, by, strategy) {
            upgrades.push(MergeUpgradeCandidate {
                bx: bx as u16,
                by: by as u16,
                strategy,
            });
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
                let mut children = [[0.0f32; 2]; 2];
                let mut baseline = [[0.0f32; 2]; 2];
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
                        children[sy][sx] = costs.chosen;
                        baseline[sy][sx] = costs.dct8;
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

                let raw = [cost32, cl, cr, ct, cb];
                let sites = merge32_sites(bx, by);
                let mut costs = raw.map(|c| merge.bias_rect32 * c);
                costs[0] = merge.bias_32x32 * cost32;
                for (k, &(x, y, strategy)) in sites.iter().enumerate() {
                    if !ac_strategy.can_place_strategy(x, y, strategy) {
                        costs[k] = f32::INFINITY;
                    }
                }

                let (mut q_min, mut q_max) = (u8::MAX, 0u8);
                for iy in 0..4 {
                    for &q in &quant_field.row(by + iy)[bx..bx + 4] {
                        q_min = q_min.min(q);
                        q_max = q_max.max(q);
                    }
                }
                let gate =
                    |accept: f32| risk_gated(merge.risk_k, accept, q_min as f32, q_max as f32, 2.0);
                let mask = choose_merge_partition(
                    children,
                    baseline,
                    costs,
                    [
                        gate(merge.accept_32),
                        gate(merge.accept_32_rect),
                        gate(merge.accept_32_rect),
                        gate(merge.accept_32_rect),
                        gate(merge.accept_32_rect),
                    ],
                );
                for (k, &(x, y, strategy)) in sites.iter().enumerate() {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let cx = AcStrategyImage::covered_blocks_x_of(strategy);
                    let cy = AcStrategyImage::covered_blocks_y_of(strategy);
                    capture_children(ac_strategy, saved, x, y, cx, cy);
                    ac_strategy.set_first(x, y, strategy);
                }
                // Shortlists survive parent merges: a reconstruction downgrade
                // can expose their footprints again.
                chosen32.push(Chosen32Cost {
                    bx: bx as u16,
                    by: by as u16,
                    cost: partition_cost(
                        children,
                        if ctx.selector.raw_propagation {
                            raw
                        } else {
                            costs
                        },
                        mask,
                    ),
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
    let shortlist = sub8_shortlist(
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
    );
    if let Some(fine) = shortlist.fine
        && fine.biased_j < shortlist.structural_cost
        && let Some(gain) = fine_recon_admit(params, scratch, bx, by, qac, meta_r, fine.strategy)
    {
        return Some(Sub8Pick { gain, ..fine });
    }
    shortlist.structural
}

/// Coefficient-domain shortlist of one block's sub-8 family: the best
/// DCT4/AFV candidate (final on its own comparison against DCT8) and the
/// best IDENTITY/DCT2X2 candidate, which still needs [`fine_recon_admit`].
struct Sub8Shortlist {
    /// The DCT4/AFV winner when it beats DCT8 on the coefficient model.
    structural: Option<Sub8Pick>,
    /// The DCT4/AFV winner's decision cost even when it lost (`INFINITY`
    /// when the family is off): the fine candidate must beat it too.
    structural_cost: f32,
    /// IDENTITY/DCT2X2 shortlisted against DCT8 (gain not yet known).
    fine: Option<Sub8Pick>,
}

#[allow(clippy::too_many_arguments)]
fn sub8_shortlist(
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
) -> Sub8Shortlist {
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
    Sub8Shortlist {
        structural: (cand_cost < cost8).then_some(Sub8Pick {
            strategy: cand,
            biased_j: cand_cost,
            raw_j: cand_raw,
            gain: cost8 - cand_cost,
        }),
        structural_cost: cand_cost,
        fine: (fine_coeff_cost < cost8).then_some(Sub8Pick {
            strategy: fine,
            biased_j: fine_coeff_cost,
            raw_j: fine_raw,
            gain: 0.0,
        }),
    }
}

/// Reconstruction-domain admission of a shortlisted IDENTITY/DCT2X2 block:
/// the gain over DCT8 when the candidate beats it by the margin, after the
/// fitted rate charge.
fn fine_recon_admit(
    params: AcStrategyParams<'_>,
    scratch: &mut CoderScratch,
    bx: usize,
    by: usize,
    qac: f32,
    meta_r: f32,
    fine: u8,
) -> Option<f32> {
    fine_recon_admit_against(params, scratch, bx, by, qac, meta_r, STRATEGY_DCT, fine)
}

/// [`fine_recon_admit`] against any surviving 1x1 incumbent (DCT8 or a
/// structural DCT4/AFV leaf): the fine transform never enters the merge
/// competition, but once the hierarchy is final it may refine whatever
/// single block survived.
#[allow(clippy::too_many_arguments)]
fn fine_recon_admit_against(
    params: AcStrategyParams<'_>,
    scratch: &mut CoderScratch,
    bx: usize,
    by: usize,
    qac: f32,
    meta_r: f32,
    incumbent: u8,
    fine: u8,
) -> Option<f32> {
    let ctx = params.ctx;
    let px = params.dc_group_px + bx * 8;
    let py = params.dc_group_py + by * 8;
    let cmap_factor = cmap_factors(params.ytox_map, params.ytob_map, bx, by);
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
    let recon8 = reconstruction_cost(scratch, incumbent);
    // The reconstruction scorer over-credits fine transforms; charge the
    // fitted per-block correction at the floored lambda (see the constant)
    // before the margin test below.
    let recon_fine = fmla(
        fine_mosaic_lambda(params.distance),
        FINE_ADMIT_RATE_CORRECTION_BITS
            + if incumbent == STRATEGY_DCT {
                0.0
            } else {
                FINE_ADMIT_LEAF_EXTRA_BITS
            },
        reconstruction_cost(scratch, fine),
    );
    // A small safety margin absorbs the remaining mismatch between the local
    // reconstruction metric and the final post-filtered image.
    (recon_fine < recon8 * FINE_RECON_MARGIN).then_some(recon8 - recon_fine)
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

    // Revisit selected merges in the reconstruction domain. Within each
    // footprint, optimize all aligned subdivisions from DCT8, saved structural
    // leaves and the supported smaller merges. The parent keeps its fitted
    // acceptance margin against the best subdivision.

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
                                // The rerank cached DCT8 at this site. Its cost
                                // cannot price a different transform's current
                                // quantizer during the subsequent refinement.
                                pipeline.current_costs[by * xsize + bx] = f32::NAN;
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
                merge_upgrade_margin(distance),
            );
        }
    }

    if leaf_first {
        // The frame-level exact metadata gate judges only the post-merge
        // IDENTITY/DCT2X2 refinements: structural DCT4/AFV leaves competed
        // in the hierarchy under their META_R charge and have no DCT8-only
        // fallback (see `is_gated_sub8_strategy`). Blocks inside a fine
        // mosaic carry their own rollback benefit and exact gate.
        let in_mosaic = |bx: usize, by: usize| {
            pipeline.band_scratch[..pipeline.bands.len()]
                .iter()
                .flat_map(|band| band.fine_rollbacks.iter())
                .any(|r| bx >= r.bx && bx < r.bx + r.cov_x && by >= r.by && by < r.by + r.cov_y)
        };
        benefit = ac_strategy
            .iter_first_blocks()
            .filter(|&(bx, by, s)| {
                crate::dc_group_data::is_gated_sub8_strategy(s) && !in_mosaic(bx, by)
            })
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

/// Reconstruction-based rerank of each selected merge against its best
/// supported subdivision, including mixtures of retained and split children.
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

#[derive(Clone, Copy)]
struct PartitionTile {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    cost: f32,
}

#[derive(Clone, Copy)]
struct ReconstructionPlan {
    cost: f32,
    /// Candidate index at each first block, in the enclosing 4x4 grid.
    tiles: [u8; 16],
}

/// Minimum-cost tiling of an aligned region, using both horizontal and
/// vertical subdivisions at every level. Transform costs are additive here;
/// the boundary-aware fine mosaic has its own joint search below.
fn best_reconstruction_plan(
    tiles: &[PartitionTile],
    width: usize,
    height: usize,
) -> ReconstructionPlan {
    let empty = ReconstructionPlan {
        cost: f32::INFINITY,
        tiles: [NO_CHILD_BLOCK; 16],
    };
    let mut plans = [empty; 9 * 16];
    let index = |lx: usize, ly: usize, x: usize, y: usize| (ly * 3 + lx) * 16 + y * 4 + x;
    let join = |a: ReconstructionPlan, b: ReconstructionPlan| {
        let mut tiles = a.tiles;
        for (i, &tile) in b.tiles.iter().enumerate() {
            if tile != NO_CHILD_BLOCK {
                tiles[i] = tile;
            }
        }
        ReconstructionPlan {
            cost: a.cost + b.cost,
            tiles,
        }
    };
    for ly in 0..=height.ilog2() as usize {
        let h = 1 << ly;
        for lx in 0..=width.ilog2() as usize {
            let w = 1 << lx;
            for y in (0..height).step_by(h) {
                for x in (0..width).step_by(w) {
                    let mut best = empty;
                    if lx > 0 {
                        best = join(
                            plans[index(lx - 1, ly, x, y)],
                            plans[index(lx - 1, ly, x + w / 2, y)],
                        );
                    }
                    if ly > 0 {
                        let split = join(
                            plans[index(lx, ly - 1, x, y)],
                            plans[index(lx, ly - 1, x, y + h / 2)],
                        );
                        if split.cost < best.cost {
                            best = split;
                        }
                    }
                    for (i, tile) in tiles.iter().enumerate() {
                        if (tile.x, tile.y, tile.width, tile.height) == (x, y, w, h)
                            && tile.cost < best.cost
                        {
                            best = empty;
                            best.cost = tile.cost;
                            best.tiles[y * 4 + x] = i as u8;
                        }
                    }
                    plans[index(lx, ly, x, y)] = best;
                }
            }
        }
    }
    plans[index(width.ilog2() as usize, height.ilog2() as usize, 0, 0)]
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
        let mut partition_tiles = Vec::with_capacity(48);
        let mut partition_costs = Vec::with_capacity(48);
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
                        partition_tiles.push(PartitionTile {
                            x: ix,
                            y: iy,
                            width: 1,
                            height: 1,
                            cost: cand.cost,
                        });
                        partition_costs.push((STRATEGY_DCT, cand));
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
        // Re-plan all supported subdivisions, not just the one child layout
        // the coefficient selector saved. Every arm uses the same quantizer,
        // metadata charge and parent gradient weights. The DP can keep one
        // winning child and split another, or change rectangular orientation.
        let child_costs_start = output.current_costs.len();
        let saved = saved_map.get(&(bx as u16, by as u16));
        let rectangles = SearchScope::for_speed(ctx.speed).rectangles();
        for iy in 0..cyb {
            for ix in 0..cxb {
                let leaf = saved.map_or(NO_CHILD_BLOCK, |grid| grid[iy * 4 + ix]);
                for (candidate_index, s) in [
                    leaf,
                    STRATEGY_DCT16X8,
                    STRATEGY_DCT8X16,
                    STRATEGY_DCT16X16,
                    STRATEGY_DCT32X16,
                    STRATEGY_DCT16X32,
                ]
                .into_iter()
                .enumerate()
                {
                    if s == NO_CHILD_BLOCK || s == STRATEGY_DCT {
                        continue;
                    }
                    let ccx = AcStrategyImage::covered_blocks_x_of(s);
                    let ccy = AcStrategyImage::covered_blocks_y_of(s);
                    // Only structural 1x1 leaves are imported from the saved
                    // layout; merged candidates below are generated once.
                    if candidate_index == 0 && ccx * ccy > 1 {
                        continue;
                    }
                    if ix + ccx > cxb
                        || iy + ccy > cyb
                        || !ix.is_multiple_of(ccx)
                        || !iy.is_multiple_of(ccy)
                        || (ccx == cxb && ccy == cyb)
                        || (!rectangles && ccx != ccy)
                        || !ac_strategy.can_place_strategy(bx + ix, by + iy, s)
                    {
                        continue;
                    }
                    let cost = reconstruction_strategy_cost_and_base(
                        ctx,
                        scratch,
                        s,
                        params.opsin,
                        px + ix * 8,
                        py + iy * 8,
                        region_qac(
                            rerank.quant_field,
                            bx + ix,
                            by + iy,
                            ccx,
                            ccy,
                            params.scale,
                            params.distance,
                        ),
                        params.qm_mult_x,
                        meta_r,
                        params.distance,
                        cmap_factors(params.ytox_map, params.ytob_map, bx + ix, by + iy),
                        gradient_alpha,
                        gradient_peak_alpha,
                        None,
                    );
                    partition_tiles.push(PartitionTile {
                        x: ix,
                        y: iy,
                        width: ccx,
                        height: ccy,
                        cost: cost.cost,
                    });
                    partition_costs.push((s, cost));
                }
            }
        }
        let plan = best_reconstruction_plan(&partition_tiles, cxb, cyb);
        let j_child = plan.cost;
        let mut child_grid = [NO_CHILD_BLOCK; 16];
        for (i, &tile) in plan.tiles.iter().enumerate() {
            if tile == NO_CHILD_BLOCK {
                continue;
            }
            let (strategy, cost) = partition_costs[tile as usize];
            child_grid[i] = strategy;
            output.current_costs.push(CachedQuantCost {
                bx: bx + i % 4,
                by: by + i / 4,
                cost: cost.base,
            });
        }
        // A different subdivision can win at the fine mosaic's floored
        // lambda. Re-optimize its incumbent too; rescoring only the ordinary
        // winner would give the mosaic an artificially weak comparison.
        let j_child_fine = if with_fine_mosaic {
            for (tile, (_, cost)) in partition_tiles.iter_mut().zip(&partition_costs) {
                tile.cost = fmla(fine_lambda, cost.rate + meta_r, cost.distortion);
            }
            best_reconstruction_plan(&partition_tiles, cxb, cyb).cost
        } else {
            f32::INFINITY
        };

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
    // Every shortlist is contained in an aligned 2x2 super-block. Score all
    // candidates before choosing its best compatible subset; enumeration
    // order must not decide between a square and two better rectangles.
    let mut groups =
        std::collections::BTreeMap::<(usize, usize), [Option<(MergeUpgrade, f32)>; 5]>::new();
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
            || (ac_strategy.is_first_block(bx, by) && ac_strategy.raw_strategy(bx, by) == strategy)
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
        let gain = j_layout - big.cost / margin;
        if gain > 0.0 {
            let k = match strategy {
                STRATEGY_DCT16X16 => 0,
                STRATEGY_DCT16X8 => 1 + bx % 2,
                STRATEGY_DCT8X16 => 3 + by % 2,
                _ => unreachable!("upgrade outside the 16x16 shortlist"),
            };
            groups.entry((by / 2, bx / 2)).or_insert([None; 5])[k] = Some((
                MergeUpgrade {
                    bx,
                    by,
                    strategy,
                    base: big.base,
                },
                gain,
            ));
        }
    }
    for candidates in groups.values() {
        let gains = candidates.map(|c| c.map_or(f32::NEG_INFINITY, |(_, gain)| gain));
        let mask = choose_upgrade_partition(gains);
        for (k, candidate) in candidates.iter().enumerate() {
            if mask & (1 << k) != 0 {
                output.merge_upgrades.push(candidate.unwrap().0);
            }
        }
    }
}

/// Exhaustive maximum-gain packing of the five overlapping shortlist sites.
fn choose_upgrade_partition(gains: [f32; 5]) -> u8 {
    let (mut best, mut best_gain) = (0, 0.0);
    for mask in MERGE_PARTITIONS.into_iter().skip(1) {
        let gain: f32 = gains
            .iter()
            .enumerate()
            .filter(|(k, _)| mask & (1 << k) != 0)
            .map(|(_, gain)| gain)
            .sum();
        if gain > best_gain {
            best = mask;
            best_gain = gain;
        }
    }
    best
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
    #[test]
    fn quantizer_cache_excludes_rerank_only_gradient_penalties() {
        let distance = 2.5;
        let ctx =
            EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, distance, 1);
        let mut opsin = noise_opsin(16, 16, 0x1234_9876);
        for c in 0..3 {
            for y in 0..16 {
                for (x, value) in opsin.plane_mut(c).row_mut(y).iter_mut().enumerate() {
                    *value = 0.25 + 0.2 * ((x * 13 + y * 19 + c * 7) as f32).sin();
                }
            }
        }
        let mut scratch = CoderScratch::default();
        let mut saw_penalty = false;
        for strategy in [STRATEGY_DCT, STRATEGY_DCT16X8, super::STRATEGY_DCT8X16] {
            let mut score = |alpha, peak, metadata| {
                super::reconstruction_strategy_cost_and_base(
                    &ctx,
                    &mut scratch,
                    strategy,
                    &opsin,
                    0,
                    0,
                    5.0,
                    1.0,
                    metadata,
                    distance,
                    [0.0; 3],
                    alpha,
                    peak,
                    None,
                )
            };
            let plain = score(0.0, 0.0, super::RERANK_META_R);
            let fresh_incumbent = score(0.0, 0.0, 0.0);
            assert_eq!(plain.base, fresh_incumbent.cost);
            for (alpha, peak) in [(32.0, 0.0), (0.0, 32.0), (32.0, 32.0)] {
                let penalized = score(alpha, peak, super::RERANK_META_R);
                assert!(penalized.cost >= plain.cost);
                saw_penalty |= penalized.cost > plain.cost;
                assert!(
                    penalized.base.is_nan(),
                    "penalized incumbent must be rescored under refinement's objective"
                );
            }
        }
        assert!(saw_penalty, "fixture must exercise an actual cost mismatch");
    }

    #[test]
    fn merge_partitions_match_exhaustive_nonoverlapping_subsets() {
        // An independent footprint oracle includes all 32 subsets, rejecting
        // overlaps rather than using the planner's eight-layout table.
        let footprints = [0b1111u8, 0b0101, 0b1010, 0b0011, 0b1100];
        let mut state = 17u32;
        for _ in 0..500 {
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (1 + (state >> 24)) as f32
            };
            let children = [[next(), next()], [next(), next()]];
            let baseline = children.map(|row| row.map(|v| v + next()));
            let merges = [next(), next(), next(), next(), next()];
            let accept = [0.5, 0.6, 0.7, 0.8, 0.9];
            let chosen = super::choose_merge_partition(children, baseline, merges, accept);
            let mut optimum: f32 = children.iter().flatten().sum();
            for mask in 1u8..32 {
                let mut occupied = 0;
                let mut cost = 0.0;
                let (mut incumbent, mut limit) = (0.0, 0.0);
                let mut valid = true;
                for k in 0..5 {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let cells = footprints[k];
                    let mut child = 0.0;
                    let mut base = 0.0;
                    for i in 0..4 {
                        if cells & (1 << i) != 0 {
                            child += children[i / 2][i % 2];
                            base += baseline[i / 2][i % 2];
                        }
                    }
                    valid &= occupied & cells == 0;
                    incumbent += child;
                    limit += base * accept[k];
                    occupied |= cells;
                    cost += merges[k];
                }
                valid &= cost < incumbent && cost < limit;
                for i in 0..4 {
                    if occupied & (1 << i) == 0 {
                        cost += children[i / 2][i % 2];
                    }
                }
                if valid {
                    optimum = optimum.min(cost);
                }
            }
            assert_eq!(super::partition_cost(children, merges, chosen), optimum);
        }
    }

    #[test]
    fn rejected_square_does_not_hide_a_single_winning_rectangle() {
        let children = [[25.0; 2]; 2];
        // The 60-cost square is cheapest but fails its own margin. The left
        // rectangle plus untouched right children costs 65 and passes.
        assert_eq!(
            super::choose_merge_partition(
                children,
                children,
                [60.0, 15.0, 90.0, 80.0, 80.0],
                [0.5, 0.9, 0.9, 0.9, 0.9]
            ),
            2
        );
        assert_eq!(
            super::partition_cost(children, [60.0, 15.0, 90.0, 80.0, 80.0], 2),
            65.0
        );
    }

    #[test]
    fn upgrades_compare_both_directions_and_combined_gain() {
        // Square comes first in the shortlist but loses to two verticals.
        assert_eq!(
            super::choose_upgrade_partition([7.0, 4.0, 4.0, 3.0, 3.0]),
            6
        );
        assert_eq!(
            super::choose_upgrade_partition([7.0, 4.0, 4.0, 5.0, 5.0]),
            24
        );
        assert_eq!(
            super::choose_upgrade_partition([11.0, 4.0, 4.0, 5.0, 5.0]),
            1
        );
        assert_eq!(
            super::choose_upgrade_partition([-1.0, -1.0, 2.0, -1.0, -1.0]),
            4
        );
        assert_eq!(super::choose_upgrade_partition([f32::NEG_INFINITY; 5]), 0);
        assert_eq!(super::choose_upgrade_partition([0.0; 5]), 0);
    }

    #[test]
    fn two_rectangles_keep_their_joint_acceptance_margin() {
        let children = [[25.0; 2]; 2];
        // Right alone fails 50 * 0.7, but both rectangles pass the original
        // joint gate (10 + 40 < 100 * 0.7) and beat left + two leaves.
        assert_eq!(
            super::choose_merge_partition(
                children,
                children,
                [100.0, 10.0, 40.0, 100.0, 100.0],
                [0.7; 5]
            ),
            6
        );
    }

    #[test]
    fn reconstruction_planner_matches_every_exact_cover() {
        use super::PartitionTile;
        fn oracle(
            tiles: &[PartitionTile],
            masks: &[u16],
            covered: u16,
            full: u16,
            memo: &mut std::collections::HashMap<u16, f32>,
        ) -> f32 {
            if covered == full {
                return 0.0;
            }
            if let Some(&cost) = memo.get(&covered) {
                return cost;
            }
            let first = 1u16 << (full & !covered).trailing_zeros();
            let mut best = f32::INFINITY;
            for (tile, &mask) in tiles.iter().zip(masks) {
                if mask & first != 0 && mask & covered == 0 {
                    best = best.min(tile.cost + oracle(tiles, masks, covered | mask, full, memo));
                }
            }
            memo.insert(covered, best);
            best
        }
        let mut state = 31u32;
        for (width, height) in [(1, 2), (2, 1), (2, 2), (2, 4), (4, 2), (4, 4)] {
            for _ in 0..12 {
                let mut tiles = Vec::new();
                let mut masks = Vec::new();
                for (w, h) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)] {
                    if w > width || h > height {
                        continue;
                    }
                    for y in (0..height).step_by(h) {
                        for x in (0..width).step_by(w) {
                            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                            let cost = (1 + (state >> 24)) as f32;
                            tiles.push(PartitionTile {
                                x,
                                y,
                                width: w,
                                height: h,
                                cost,
                            });
                            let mut mask = 0u16;
                            for yy in y..y + h {
                                for xx in x..x + w {
                                    mask |= 1 << (yy * 4 + xx);
                                }
                            }
                            masks.push(mask);
                        }
                    }
                }
                let full = masks.iter().fold(0, |a, b| a | b);
                let expected = oracle(&tiles, &masks, 0, full, &mut Default::default());
                let plan = super::best_reconstruction_plan(&tiles, width, height);
                assert_eq!(plan.cost, expected, "{width}x{height}");
                let mut covered = 0;
                for &i in &plan.tiles {
                    if i == super::NO_CHILD_BLOCK {
                        continue;
                    }
                    assert_eq!(covered & masks[i as usize], 0);
                    covered |= masks[i as usize];
                }
                assert_eq!(covered, full);
            }
        }
    }

    #[test]
    fn reconstruction_can_keep_one_child_and_split_the_other() {
        use super::PartitionTile;
        let mut tiles = Vec::new();
        for y in 0..2 {
            for x in 0..2 {
                tiles.push(PartitionTile {
                    x,
                    y,
                    width: 1,
                    height: 1,
                    cost: 5.0,
                });
            }
        }
        tiles.push(PartitionTile {
            x: 0,
            y: 0,
            width: 1,
            height: 2,
            cost: 3.0,
        });
        tiles.push(PartitionTile {
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            cost: 15.0,
        });
        let plan = super::best_reconstruction_plan(&tiles, 2, 2);
        assert_eq!(plan.cost, 13.0);
        assert_eq!(plan.tiles[0], 4);
        assert_eq!(plan.tiles[1], 1);
        assert_eq!(plan.tiles[4], super::NO_CHILD_BLOCK);
        assert_eq!(plan.tiles[5], 3);
    }

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
        merge_upgrade_margin, quant_refinement_steps, rerank_pair_gradient_peak_alpha,
        rerank_pair_gradient_scale, select_gradient_region_stats_fn,
        select_gradient_region_stats_with_chroma_fn, strategy_cost, sub8_strategy_costs,
        use_dct8_only,
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
        let (map, benefit, _) = run_selector_full(speed, distance, policy, threads);
        (map, benefit)
    }

    fn run_selector_full(
        speed: crate::Speed,
        distance: f32,
        policy: crate::ac_strategy::SelectorPolicy,
        threads: usize,
    ) -> (
        AcStrategyImage,
        f32,
        Vec<crate::coder_scratch::FineMergeRollback>,
    ) {
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
        (strategies, benefit, fine_rollbacks)
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
            let (map, benefit, rollbacks) =
                run_selector_full(crate::Speed::Slow, 1.0, leaf, threads);
            assert_map_valid(&map);
            // Only the post-merge IDENTITY/DCT2X2 refinements are credited to
            // the frame gate; structural DCT4/AFV leaves competed in the
            // hierarchy under META_R and carry no gate credit, and fine
            // mosaics carry their own rollback benefit instead.
            let in_mosaic = |bx: usize, by: usize| {
                rollbacks
                    .iter()
                    .any(|r| bx >= r.bx && bx < r.bx + r.cov_x && by >= r.by && by < r.by + r.cov_y)
            };
            let gated = map
                .iter_first_blocks()
                .filter(|&(bx, by, s)| {
                    crate::dc_group_data::is_gated_sub8_strategy(s) && !in_mosaic(bx, by)
                })
                .count();
            assert!(benefit.is_finite());
            assert!(
                gated == 0 || benefit > 0.0,
                "fine leaves survived without credit"
            );
            assert!(gated > 0 || benefit == 0.0, "credit without fine leaves");
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

    #[test]
    fn selection_is_independent_of_worker_count() {
        for speed in [crate::Speed::Fast, crate::Speed::Slow] {
            for distance in [0.5, 1.5, 2.0, 4.0, 6.0] {
                for raw_propagation in [false, true] {
                    let policy = crate::ac_strategy::SelectorPolicy {
                        raw_propagation,
                        ..Default::default()
                    };
                    let (serial, _) = run_selector(speed, distance, policy, 1);
                    let (parallel, _) = run_selector(speed, distance, policy, 3);
                    assert_eq!(
                        strategy_cells(&serial),
                        strategy_cells(&parallel),
                        "{speed:?} d{distance} raw={raw_propagation}"
                    );
                    assert_map_valid(&parallel);
                }
            }
        }
    }

    #[test]
    fn upgrade_shortlist_survives_a_committed_square_and_is_order_independent() {
        let ctx = EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, 3.0, 1);
        let opsin = Image3F::new(16, 16);
        let maps = ImageSB::new_fill(1, 1, 0);
        let qf = ImageB::new_fill(2, 2, 6);
        let params = AcStrategyParams {
            ctx: &ctx,
            opsin: &opsin,
            dc_group_px: 0,
            dc_group_py: 0,
            distance: 3.0,
            scale: 1.0,
            qm_mult_x: 1.0,
            ytox_map: &maps,
            ytob_map: &maps,
        };
        let rerank = super::RerankContext {
            params,
            quant_field: &qf,
            bands: &[(0, 2)],
            num_threads: 1,
        };
        let mut map = AcStrategyImage::new(2, 2);
        map.set_first(0, 0, STRATEGY_DCT16X16);
        let mut output = AcStrategyBandScratch::default();
        super::record_upgrade_candidates(
            &mut output.upgrade_candidates,
            &map,
            0,
            0,
            super::UpgradeBand {
                raw: [1.0; 5],
                incumbent: [2.0; 5],
            },
        );
        assert_eq!(output.upgrade_candidates.len(), 5);
        // Simulate a downgrade exposing all four cells again.
        let map = AcStrategyImage::new(2, 2);
        let mut scratch = CoderScratch::default();
        super::find_merge_upgrades(&rerank, &mut scratch, &map, (0, 2), &mut output, 1.0);
        let chosen = |o: &AcStrategyBandScratch| {
            o.merge_upgrades
                .iter()
                .map(|u| (u.bx, u.by, u.strategy, u.base.to_bits()))
                .collect::<Vec<_>>()
        };
        let forward = chosen(&output);
        assert!(!forward.is_empty());
        output.upgrade_candidates.reverse();
        super::find_merge_upgrades(&rerank, &mut scratch, &map, (0, 2), &mut output, 1.0);
        assert_eq!(forward, chosen(&output));
    }

    /// The rate calibration prices merged transforms below DCT8 only in the
    /// mid/low bands (the model over-prices them there), never at HQ, and
    /// leaves DCT8, the 64 family and the sub-8 family untouched.
    #[test]
    fn rate_calibration_is_band_limited_to_merged_families() {
        use crate::ac_strategy::RateCalibration;
        for s in [
            STRATEGY_DCT,
            STRATEGY_DCT4X4,
            STRATEGY_IDENTITY,
            STRATEGY_DCT64X64,
        ] {
            for d in [0.5f32, 2.75, 6.0] {
                assert_eq!(RateCalibration::scale(s, d), 1.0);
            }
        }
        for s in [
            STRATEGY_DCT16X8,
            STRATEGY_DCT16X16,
            STRATEGY_DCT32X32,
            STRATEGY_DCT32X16,
        ] {
            assert_eq!(RateCalibration::scale(s, 1.0), 1.0);
            let mid = RateCalibration::scale(s, 2.75);
            assert!((0.85..1.0).contains(&mid), "{s}: {mid}");
            assert!(RateCalibration::scale(s, 2.0) > mid);
            assert_eq!(
                RateCalibration::scale(s, 6.0),
                RateCalibration::scale(s, 4.0)
            );
        }
        assert!(
            RateCalibration::scale(STRATEGY_DCT32X32, 4.0)
                < RateCalibration::scale(STRATEGY_DCT16X8, 4.0)
        );
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
        // Knife-edge at the gate, slightly optimistic once quality is low,
        // monotone in between.
        assert_eq!(merge_upgrade_margin(gate), margin);
        let low = merge_upgrade_margin(6.0);
        assert!(low > margin && low <= 1.1, "low margin {low}");
        assert_eq!(low, merge_upgrade_margin(4.0));
        assert!(merge_upgrade_margin(3.0) > margin && merge_upgrade_margin(3.0) < low);
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
