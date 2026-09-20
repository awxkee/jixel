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

use super::*;

/// One 8x8 block's best 8x8-family transform.
#[derive(Clone, Copy)]
pub(crate) struct LeafChoice {
    pub(crate) strategy: u8,
    /// Coefficient-domain decision cost (family bias applied) — the incumbent
    /// merges above this block compete against.
    pub(crate) j: f32,
    /// The same cost without the family bias.
    pub(crate) raw_j: f32,
    /// Sub-8 metadata-gate credit if this leaf survives to the final map.
    pub(crate) gain: f32,
    /// IDENTITY/DCT2X2 shortlisted against DCT8 on the coefficient model
    /// (`NO_CHILD_BLOCK` = none); admitted after the merges against whatever
    /// 1x1 leaf survived.
    pub(crate) fine: u8,
    /// The fine candidate's coefficient-domain decision cost.
    pub(crate) fine_j: f32,
}

impl Default for LeafChoice {
    fn default() -> Self {
        Self {
            strategy: STRATEGY_DCT,
            j: f32::NAN,
            raw_j: f32::NAN,
            gain: 0.0,
            fine: NO_CHILD_BLOCK,
            fine_j: f32::INFINITY,
        }
    }
}

/// A 2x2 super-block's decision. `grid` holds the strategy at offsets
/// `dy * 2 + dx`, `NO_CHILD_BLOCK` where a preceding first block covers it.
#[derive(Clone, Copy)]
struct SuperPlan {
    grid: [u8; 4],
    /// Cost propagated to the 32 level (decision or raw, per policy).
    j: f32,
    /// Sum of the four leaves' decision costs.
    leaf_total: f32,
    /// The losing rectangular/leaf arm when a 16x16 won, for the rerank's
    /// child-layout restore; `None` when that arm is plain DCT8 tiling.
    child: Option<[u8; 4]>,
    /// The four leaves (`dy * 2 + dx`), so a committed pair can save the
    /// leaf layout it displaced.
    leaves: [u8; 4],
}

fn expand_2x2(grid: [u8; 4], sx: usize, sy: usize, into: &mut [u8; 16]) {
    for dy in 0..2 {
        for dx in 0..2 {
            into[(sy * 2 + dy) * 4 + sx * 2 + dx] = grid[dy * 2 + dx];
        }
    }
}

/// The `cov_x x cov_y` window of a 4x4 child grid at offset `(ox, oy)`,
/// re-based so its own origin is index 0 (the rerank indexes `iy * 4 + ix`
/// from the selected transform's first block).
fn sub_grid(grid: &[u8; 16], ox: usize, oy: usize, cov_x: usize, cov_y: usize) -> [u8; 16] {
    let mut out = [NO_CHILD_BLOCK; 16];
    for iy in 0..cov_y {
        for ix in 0..cov_x {
            out[iy * 4 + ix] = grid[(oy + iy) * 4 + ox + ix];
        }
    }
    out
}

fn grid_nontrivial(grid: &[u8]) -> bool {
    grid.iter()
        .any(|&s| s != NO_CHILD_BLOCK && s != STRATEGY_DCT)
}

struct LeafBand<'a> {
    leaves: &'a [LeafChoice],
    y0: usize,
    xsize: usize,
}

impl LeafBand<'_> {
    #[inline]
    fn at(&self, bx: usize, by: usize) -> LeafChoice {
        self.leaves[(by - self.y0) * self.xsize + bx]
    }
}

/// Plan one 2x2 super-block at block `(bx0, by0)` without touching the map.
fn plan_super_block(
    selection: &SelectionContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &AcStrategyImage,
    leaves: &LeafBand<'_>,
    bx0: usize,
    by0: usize,
    upgrades: &mut Vec<MergeUpgradeCandidate>,
) -> SuperPlan {
    let params = selection.params;
    let ctx = params.ctx;
    let merge = ctx.merge;
    let raw_propagation = ctx.selector.raw_propagation;
    let (px0, py0) = (params.dc_group_px + bx0 * 8, params.dc_group_py + by0 * 8);
    let qac = block_qac_2x2(selection.quant_field, bx0, by0, params.scale);
    let cmap_factor = cmap_factors(params.ytox_map, params.ytob_map, bx0, by0);

    let l = [
        [leaves.at(bx0, by0), leaves.at(bx0 + 1, by0)],
        [leaves.at(bx0, by0 + 1), leaves.at(bx0 + 1, by0 + 1)],
    ];
    let leaf_left = l[0][0].j + l[1][0].j;
    let leaf_right = l[0][1].j + l[1][1].j;
    let leaf_top = l[0][0].j + l[0][1].j;
    let leaf_bottom = l[1][0].j + l[1][1].j;
    let leaf_total = leaf_left + leaf_right;
    let raw_left = l[0][0].raw_j + l[1][0].raw_j;
    let raw_right = l[0][1].raw_j + l[1][1].raw_j;
    let raw_top = l[0][0].raw_j + l[0][1].raw_j;
    let raw_bottom = l[1][0].raw_j + l[1][1].raw_j;

    // (decision cost, unbiased cost) per rectangular candidate.
    let mut rect_cost = |px: usize, py: usize, strategy: u8, qac: f32| -> (f32, f32) {
        if !selection.scope.rectangles() {
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
            selection.meta_r,
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

    let c16_raw = strategy_cost(
        ctx,
        scratch,
        STRATEGY_DCT16X16,
        params.opsin,
        px0,
        py0,
        aggregate_qac_2x2(qac, params.scale, params.distance),
        params.qm_mult_x,
        selection.meta_r,
        params.distance,
        cmap_factor,
    );
    let c16 = merge.bias_16x16 * c16_raw;

    // Pairs must beat the two leaves they replace by the pair margin.
    let use_v_left = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X8)
        && merge_beats_dct8(v_left, leaf_left, merge.accept_pair);
    let use_v_right = ac_strategy.can_place_strategy(bx0 + 1, by0, STRATEGY_DCT16X8)
        && merge_beats_dct8(v_right, leaf_right, merge.accept_pair);
    let use_h_top = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_top, leaf_top, merge.accept_pair);
    let use_h_bottom = ac_strategy.can_place_strategy(bx0, by0 + 1, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_bot, leaf_bottom, merge.accept_pair);

    let cost_16x8 = if use_v_left { v_left } else { leaf_left }
        + if use_v_right { v_right } else { leaf_right };
    let cost_8x16 =
        if use_h_top { h_top } else { leaf_top } + if use_h_bottom { h_bot } else { leaf_bottom };
    let raw_16x8 = if use_v_left { v_left_raw } else { raw_left }
        + if use_v_right { v_right_raw } else { raw_right };
    let raw_8x16 = if use_h_top { h_top_raw } else { raw_top }
        + if use_h_bottom { h_bot_raw } else { raw_bottom };
    let vertical = cost_16x8 <= cost_8x16;
    let best_rect = cost_16x8.min(cost_8x16);

    let (q_min, q_max) = qac
        .iter()
        .flatten()
        .fold((f32::INFINITY, 0.0f32), |(mn, mx), &q| {
            (mn.min(q), mx.max(q))
        });
    // The 16x16 must beat the best rectangular/leaf arm outright and the leaf
    // sum by the (risk-gated) 16 margin.
    let pick_16x16 = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X16)
        && c16 < best_rect
        && merge_beats_dct8(
            c16,
            leaf_total,
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
                incumbent: [leaf_total, leaf_left, leaf_right, leaf_top, leaf_bottom],
            },
        );
    }

    let leaf_s = |dy: usize, dx: usize| l[dy][dx].strategy;
    let leaves = [leaf_s(0, 0), leaf_s(0, 1), leaf_s(1, 0), leaf_s(1, 1)];
    let arm = if vertical {
        [
            if use_v_left {
                STRATEGY_DCT16X8
            } else {
                leaf_s(0, 0)
            },
            if use_v_right {
                STRATEGY_DCT16X8
            } else {
                leaf_s(0, 1)
            },
            if use_v_left {
                NO_CHILD_BLOCK
            } else {
                leaf_s(1, 0)
            },
            if use_v_right {
                NO_CHILD_BLOCK
            } else {
                leaf_s(1, 1)
            },
        ]
    } else {
        [
            if use_h_top {
                STRATEGY_DCT8X16
            } else {
                leaf_s(0, 0)
            },
            if use_h_top {
                NO_CHILD_BLOCK
            } else {
                leaf_s(0, 1)
            },
            if use_h_bottom {
                STRATEGY_DCT8X16
            } else {
                leaf_s(1, 0)
            },
            if use_h_bottom {
                NO_CHILD_BLOCK
            } else {
                leaf_s(1, 1)
            },
        ]
    };

    if pick_16x16 {
        SuperPlan {
            grid: [
                STRATEGY_DCT16X16,
                NO_CHILD_BLOCK,
                NO_CHILD_BLOCK,
                NO_CHILD_BLOCK,
            ],
            j: if raw_propagation { c16_raw } else { c16 },
            leaf_total,
            child: grid_nontrivial(&arm).then_some(arm),
            leaves,
        }
    } else {
        SuperPlan {
            grid: arm,
            j: if raw_propagation {
                if vertical { raw_16x8 } else { raw_8x16 }
            } else {
                best_rect
            },
            leaf_total,
            child: None,
            leaves,
        }
    }
}

/// Write a super-block plan into the map. Leaves were committed beforehand,
/// so only merges (which overwrite the leaf cells they cover) are placed.
fn commit_super_block(
    ac_strategy: &mut AcStrategyImage,
    saved: &mut Vec<SavedChild>,
    bx0: usize,
    by0: usize,
    plan: &SuperPlan,
) {
    if plan.grid[0] == STRATEGY_DCT16X16 {
        ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X16);
        if let Some(child) = plan.child {
            let mut grid = [NO_CHILD_BLOCK; 16];
            expand_2x2(child, 0, 0, &mut grid);
            saved.push(SavedChild {
                bx: bx0 as u16,
                by: by0 as u16,
                grid,
            });
        }
        return;
    }
    for dy in 0..2 {
        for dx in 0..2 {
            let s = plan.grid[dy * 2 + dx];
            if !matches!(s, STRATEGY_DCT16X8 | STRATEGY_DCT8X16) {
                continue;
            }
            ac_strategy.set_first(bx0 + dx, by0 + dy, s);
            // A pair that displaced structural leaves saves them, so the
            // rerank compares it against that layout and not only against
            // tiled DCT8 (rebased to the pair's own first block).
            let mut child = [NO_CHILD_BLOCK; 16];
            if s == STRATEGY_DCT16X8 {
                child[0] = plan.leaves[dx];
                child[4] = plan.leaves[2 + dx];
            } else {
                child[0] = plan.leaves[dy * 2];
                child[1] = plan.leaves[dy * 2 + 1];
            }
            if grid_nontrivial(&child) {
                saved.push(SavedChild {
                    bx: (bx0 + dx) as u16,
                    by: (by0 + dy) as u16,
                    grid: child,
                });
            }
        }
    }
}

/// Leaf-first selection for block rows `[y_begin, y_end)`. Same band
/// contract as `select_band`: full-group edge tests, reads
/// `quant_field`/`opsin` only, results independent per band. Fills
/// `output.leaves` (band-local), `output.chosen32` and `output.saved_children`.
pub(super) fn select_band_leaf_first(
    selection: &SelectionContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &mut AcStrategyImage,
    band: (usize, usize),
    output: &mut AcStrategyBandScratch,
) {
    let params = selection.params;
    let ctx = params.ctx;
    let distance = params.distance;
    let scale = params.scale;
    let quant_field = selection.quant_field;
    let meta_r = selection.meta_r;
    let scope = selection.scope;
    let merge = ctx.merge;
    let raw_propagation = ctx.selector.raw_propagation;
    let (y_begin, y_end) = band;
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    output.chosen32.clear();
    output.saved_children.clear();
    output.upgrade_candidates.clear();

    // --- 1. Leaves -------------------------------------------------------
    let with_dct4 = distance <= SUB8_MAX_DISTANCE;
    let with_fine = distance <= FINE_TRANSFORM_MAX_DISTANCE;
    // IDENTITY/DCT2X2 stay out of the structural leaves (their
    // coefficient-domain cost is not comparable with the orthogonal
    // transforms; as leaves they tie with the post-merge order on every
    // corpus): they are shortlisted here and admitted on the remaining DCT8
    // blocks after the merge hierarchy.
    let sub8_enabled =
        scope.rectangles() && (with_dct4 || distance <= AFV_MAX_DISTANCE || with_fine);
    let bias_afv = BIAS_AFV.at(distance);
    let n_leaves = xsize * (y_end - y_begin);
    output.leaves.clear();
    output.leaves.resize(n_leaves, LeafChoice::default());
    // Pass 1: the DCT8 incumbent of every block; pass 2 the sub-8 bundle.
    for by in y_begin..y_end {
        for bx in 0..xsize {
            let qac = region_qac(quant_field, bx, by, 1, 1, scale, distance);
            let cmap_factor = cmap_factors(params.ytox_map, params.ytob_map, bx, by);
            output.leaves[(by - y_begin) * xsize + bx].j = strategy_cost(
                ctx,
                scratch,
                STRATEGY_DCT,
                params.opsin,
                params.dc_group_px + bx * 8,
                params.dc_group_py + by * 8,
                qac,
                params.qm_mult_x,
                meta_r,
                distance,
                cmap_factor,
            );
        }
    }
    for by in y_begin..y_end {
        for bx in 0..xsize {
            let qac = region_qac(quant_field, bx, by, 1, 1, scale, distance);
            let dct8 = output.leaves[(by - y_begin) * xsize + bx].j;
            let leaf = &mut output.leaves[(by - y_begin) * xsize + bx];
            *leaf = LeafChoice {
                strategy: STRATEGY_DCT,
                j: dct8,
                raw_j: dct8,
                gain: 0.0,
                fine: NO_CHILD_BLOCK,
                fine_j: f32::INFINITY,
            };
            if !sub8_enabled {
                continue;
            }
            // One bundle per block covers both the structural leaf (final on
            // the coefficient model) and the fine shortlist, whose
            // reconstruction admission waits until the merges are known.
            let shortlist = sub8_shortlist(
                params,
                scratch,
                bx,
                by,
                qac,
                meta_r,
                Some(dct8),
                with_dct4,
                with_fine,
                bias_afv,
            );
            if let Some(p) = shortlist.structural {
                leaf.strategy = p.strategy;
                leaf.j = p.biased_j;
                leaf.raw_j = p.raw_j;
                leaf.gain = p.gain;
            }
            if let Some(f) = shortlist.fine {
                leaf.fine = f.strategy;
                leaf.fine_j = f.biased_j;
            }
        }
    }
    // Commit every non-DCT8 leaf; merges below overwrite the cells they cover.
    for by in y_begin..y_end {
        for bx in 0..xsize {
            let s = output.leaves[(by - y_begin) * xsize + bx].strategy;
            if s != STRATEGY_DCT {
                ac_strategy.set_first(bx, by, s);
            }
        }
    }
    let leaves = LeafBand {
        leaves: &output.leaves,
        y0: y_begin,
        xsize,
    };

    // --- 2./3. Super-blocks and 32-class regions ---------------------------
    let saved = &mut output.saved_children;
    let chosen32 = &mut output.chosen32;
    let upgrades = &mut output.upgrade_candidates;
    let mut by = y_begin;
    while by + 1 < ysize && by < y_end {
        let four_row = by.is_multiple_of(4) && by + 4 <= ysize;
        let mut bx = 0;
        while bx + 1 < xsize {
            let four_col = bx % 4 == 0 && bx + 4 <= xsize;
            if four_row && four_col && ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT32X32) {
                let mut plans = [[SuperPlan {
                    grid: [STRATEGY_DCT; 4],
                    j: 0.0,
                    leaf_total: 0.0,
                    child: None,
                    leaves: [STRATEGY_DCT; 4],
                }; 2]; 2];
                let mut children = [[0.0f32; 2]; 2];
                let mut baseline = [[0.0f32; 2]; 2];
                for sy in 0..2 {
                    for sx in 0..2 {
                        let plan = plan_super_block(
                            selection,
                            scratch,
                            ac_strategy,
                            &leaves,
                            bx + sx * 2,
                            by + sy * 2,
                            upgrades,
                        );
                        children[sy][sx] = plan.j;
                        baseline[sy][sx] = plan.leaf_total;
                        plans[sy][sx] = plan;
                    }
                }
                let qac32 = region_qac(quant_field, bx, by, 4, 4, scale, distance);
                let cmap_factor = cmap_factors(params.ytox_map, params.ytob_map, bx, by);
                let cost32 = strategy_cost(
                    ctx,
                    scratch,
                    STRATEGY_DCT32X32,
                    params.opsin,
                    params.dc_group_px + bx * 8,
                    params.dc_group_py + by * 8,
                    qac32,
                    params.qm_mult_x,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                let mut rect32 =
                    |bx: usize, by: usize, strategy: u8, cw: usize, ch: usize| -> f32 {
                        if !scope.rectangles() {
                            return f32::INFINITY;
                        }
                        strategy_cost(
                            ctx,
                            scratch,
                            strategy,
                            params.opsin,
                            params.dc_group_px + bx * 8,
                            params.dc_group_py + by * 8,
                            region_qac(quant_field, bx, by, cw, ch, scale, distance),
                            params.qm_mult_x,
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
                let mut grid = [NO_CHILD_BLOCK; 16];
                for sy in 0..2 {
                    for sx in 0..2 {
                        expand_2x2(plans[sy][sx].grid, sx, sy, &mut grid);
                        // Only surviving super-blocks publish their children.
                        // Parent merges save the planned grid directly below.
                        let covered = mask & (1 | (1 << (1 + sx)) | (1 << (3 + sy))) != 0;
                        if !covered {
                            commit_super_block(
                                ac_strategy,
                                saved,
                                bx + sx * 2,
                                by + sy * 2,
                                &plans[sy][sx],
                            );
                        }
                    }
                }
                for (k, &(x, y, strategy)) in sites.iter().enumerate() {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let cx = AcStrategyImage::covered_blocks_x_of(strategy);
                    let cy = AcStrategyImage::covered_blocks_y_of(strategy);
                    let child = sub_grid(&grid, x - bx, y - by, cx, cy);
                    if grid_nontrivial(&child) {
                        saved.push(SavedChild {
                            bx: x as u16,
                            by: y as u16,
                            grid: child,
                        });
                    }
                    ac_strategy.set_first(x, y, strategy);
                }
                chosen32.push(Chosen32Cost {
                    bx: bx as u16,
                    by: by as u16,
                    cost: partition_cost(children, if raw_propagation { raw } else { costs }, mask),
                });
                bx += 4;
            } else if four_row {
                for sby in [by, by + 2] {
                    let plan = plan_super_block(
                        selection,
                        scratch,
                        ac_strategy,
                        &leaves,
                        bx,
                        sby,
                        upgrades,
                    );
                    commit_super_block(ac_strategy, saved, bx, sby, &plan);
                }
                bx += 2;
            } else {
                let plan =
                    plan_super_block(selection, scratch, ac_strategy, &leaves, bx, by, upgrades);
                commit_super_block(ac_strategy, saved, bx, by, &plan);
                bx += 2;
            }
        }
        by += if four_row { 4 } else { 2 };
    }

    if with_fine {
        // Post-merge fine admission over every surviving 1x1 structural leaf
        // that carries a shortlisted IDENTITY/DCT2X2 candidate.
        for by in y_begin..y_end {
            for bx in 0..xsize {
                let leaf = output.leaves[(by - y_begin) * xsize + bx];
                let fine = leaf.fine;
                if fine == NO_CHILD_BLOCK || !ac_strategy.is_first_block(bx, by) {
                    continue;
                }
                let incumbent = ac_strategy.raw_strategy(bx, by);
                // Any structural 1x1 leaf may be refined; a DCT4/AFV
                // incumbent pays `FINE_ADMIT_LEAF_EXTRA_BITS` on top.
                let structural = AcStrategyImage::covered_blocks_x_of(incumbent) == 1
                    && AcStrategyImage::covered_blocks_y_of(incumbent) == 1
                    && !matches!(incumbent, STRATEGY_IDENTITY | STRATEGY_DCT2X2);
                if !structural || leaf.fine_j >= leaf.j {
                    continue;
                }
                let qac = region_qac(quant_field, bx, by, 1, 1, scale, distance);
                if let Some(gain) =
                    fine_recon_admit_against(params, scratch, bx, by, qac, meta_r, incumbent, fine)
                {
                    ac_strategy.set_first(bx, by, fine);
                    output.leaves[(by - y_begin) * xsize + bx].gain = gain;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NO_CHILD_BLOCK, sub_grid};

    /// A 32x16 pair's child layouts are looked up per half by the rerank, so
    /// the right half's grid must be re-based to its own origin (and the
    /// same for a 16x32 pair's bottom half).
    #[test]
    fn sub_grid_rebases_rectangle_halves() {
        let mut grid = [NO_CHILD_BLOCK; 16];
        for (i, cell) in grid.iter_mut().enumerate() {
            *cell = i as u8; // unique marker per cell
        }
        let left = sub_grid(&grid, 0, 0, 2, 4);
        let right = sub_grid(&grid, 2, 0, 2, 4);
        for iy in 0..4 {
            assert_eq!(left[iy * 4], (iy * 4) as u8);
            assert_eq!(left[iy * 4 + 1], (iy * 4 + 1) as u8);
            assert_eq!(right[iy * 4], (iy * 4 + 2) as u8);
            assert_eq!(right[iy * 4 + 1], (iy * 4 + 3) as u8);
            assert_eq!(left[iy * 4 + 2], NO_CHILD_BLOCK);
            assert_eq!(right[iy * 4 + 3], NO_CHILD_BLOCK);
        }
        let top = sub_grid(&grid, 0, 0, 4, 2);
        let bottom = sub_grid(&grid, 0, 2, 4, 2);
        for ix in 0..4 {
            assert_eq!(top[ix], ix as u8);
            assert_eq!(top[4 + ix], (4 + ix) as u8);
            assert_eq!(bottom[ix], (8 + ix) as u8);
            assert_eq!(bottom[4 + ix], (12 + ix) as u8);
            assert_eq!(bottom[8 + ix], NO_CHILD_BLOCK);
        }
    }
}
