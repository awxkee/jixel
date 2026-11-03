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

//! Spatial reranking for selected 64x64, 64x32 and 32x64 transforms.
//!
//! The smaller rerank retains its 4x4 plans and fine-mosaic acceptance. This
//! path compares the large incumbent with aligned subdivisions through DCT32,
//! the two 64 rectangles, and the source leaves hidden by the original merge.

use super::*;

#[derive(Clone, Copy)]
enum Choice {
    Empty,
    Tile(usize),
    SplitX,
    SplitY,
}

#[derive(Clone, Copy)]
struct Region {
    cost: f32,
    choice: Choice,
}

const EMPTY: Region = Region {
    cost: f32::INFINITY,
    choice: Choice::Empty,
};

/// All storage grows on first use and is retained by the selection band.
#[derive(Default)]
pub(crate) struct Recon64Scratch {
    regions: Vec<Region>,
    direct: Vec<[usize; 2]>,
    lower: Vec<f32>,
    failed_budget: Vec<f32>,
    tiles: Vec<PartitionTile>,
    costs: Vec<(u8, ReconStrategyCost)>,
    rate_bounds: Vec<f32>,
    walk: Vec<(usize, usize, usize, usize)>,
}

struct Plan {
    cost: f32,
    /// Tile index at each first block of the enclosing 8x8 grid. `usize`
    /// avoids the smaller planner's 255-candidate limit.
    tiles: [Option<usize>; 64],
}

fn region_index(lx: usize, ly: usize, x: usize, y: usize) -> usize {
    (ly * 4 + lx) * 64 + y * 8 + x
}

/// Store only the winning decision per region, then backtrack once. Carrying
/// a whole 8x8 layout through every DP state would multiply scratch traffic.
#[cfg(test)]
fn best_plan(storage: &mut Recon64Scratch, width: usize, height: usize) -> Plan {
    assert!(width.is_power_of_two() && height.is_power_of_two());
    assert!(width <= 8 && height <= 8);
    storage.regions.clear();
    storage.regions.resize(16 * 64, EMPTY);
    for (i, tile) in storage.tiles.iter().enumerate() {
        if !tile.width.is_power_of_two()
            || !tile.height.is_power_of_two()
            || tile.x + tile.width > width
            || tile.y + tile.height > height
            || !tile.x.is_multiple_of(tile.width)
            || !tile.y.is_multiple_of(tile.height)
        {
            continue;
        }
        let slot = region_index(
            tile.width.ilog2() as usize,
            tile.height.ilog2() as usize,
            tile.x,
            tile.y,
        );
        if tile.cost < storage.regions[slot].cost {
            storage.regions[slot] = Region {
                cost: tile.cost,
                choice: Choice::Tile(i),
            };
        }
    }
    let lx_max = width.ilog2() as usize;
    let ly_max = height.ilog2() as usize;
    for ly in 0..=ly_max {
        for lx in 0..=lx_max {
            let (w, h) = (1 << lx, 1 << ly);
            for y in (0..height).step_by(h) {
                for x in (0..width).step_by(w) {
                    let slot = region_index(lx, ly, x, y);
                    let mut best = EMPTY;
                    if lx > 0 {
                        best = Region {
                            cost: storage.regions[region_index(lx - 1, ly, x, y)].cost
                                + storage.regions[region_index(lx - 1, ly, x + w / 2, y)].cost,
                            choice: Choice::SplitX,
                        };
                    }
                    if ly > 0 {
                        let cost = storage.regions[region_index(lx, ly - 1, x, y)].cost
                            + storage.regions[region_index(lx, ly - 1, x, y + h / 2)].cost;
                        if cost < best.cost {
                            best = Region {
                                cost,
                                choice: Choice::SplitY,
                            };
                        }
                    }
                    // Same strict tie rule as the smaller reconstruction DP.
                    if storage.regions[slot].cost < best.cost {
                        best = storage.regions[slot];
                    }
                    storage.regions[slot] = best;
                }
            }
        }
    }
    finish_plan(storage, width, height)
}

fn finish_plan(storage: &mut Recon64Scratch, width: usize, height: usize) -> Plan {
    let lx_max = width.ilog2() as usize;
    let ly_max = height.ilog2() as usize;
    let mut plan = Plan {
        cost: storage.regions[region_index(lx_max, ly_max, 0, 0)].cost,
        tiles: [None; 64],
    };
    if !plan.cost.is_finite() {
        return plan;
    }
    storage.walk.clear();
    storage.walk.push((0, 0, lx_max, ly_max));
    while let Some((x, y, lx, ly)) = storage.walk.pop() {
        match storage.regions[region_index(lx, ly, x, y)].choice {
            Choice::Empty => unreachable!("finite plan has an empty region"),
            Choice::Tile(i) => plan.tiles[y * 8 + x] = Some(i),
            Choice::SplitX => {
                storage.walk.push((x + (1 << (lx - 1)), y, lx - 1, ly));
                storage.walk.push((x, y, lx - 1, ly));
            }
            Choice::SplitY => {
                storage.walk.push((x, y + (1 << (ly - 1)), lx, ly - 1));
                storage.walk.push((x, y, lx, ly - 1));
            }
        }
    }
    plan
}

const NO_TILE: usize = usize::MAX;
// SSIM deficits use 8x8 windows. Allow 0.001 weighted deficit per window
// below the mathematical zero floor to cover scalar/SIMD rounding noise.
const SSIM_ROUNDOFF_PER_WINDOW: f32 = 1e-3;
const UNMEASURED: ReconStrategyCost = ReconStrategyCost {
    cost: f32::NAN,
    base: f32::NAN,
    distortion: f32::NAN,
    rate: f32::NAN,
    hue_distortion: f32::NAN,
};

fn priority(choice: Choice) -> usize {
    match choice {
        Choice::SplitX => 0,
        Choice::SplitY => 1,
        Choice::Tile(_) => 2,
        Choice::Empty => 3,
    }
}

/// A split's operands can be rounded in opposite directions when deriving a
/// remaining budget. Admit a few ulps of slack; the original strict comparison
/// still decides the winner after the ordinary f32 addition.
fn remaining_budget(budget: f32, other: f32) -> f32 {
    (budget - other).next_up().next_up() + (budget.abs() + other.abs()) * (4.0 * f32::EPSILON)
}

struct BudgetSearch<'a, F> {
    storage: &'a mut Recon64Scratch,
    evaluate: F,
    lambda: f32,
}

impl<F: FnMut(usize, u8, usize, usize) -> ReconStrategyCost> BudgetSearch<'_, F> {
    fn search(&mut self, x: usize, y: usize, lx: usize, ly: usize, budget: f32) -> Region {
        let slot = region_index(lx, ly, x, y);
        let cached = self.storage.regions[slot];
        if !matches!(cached.choice, Choice::Empty) {
            return if cached.cost <= budget { cached } else { EMPTY };
        }
        if self.storage.lower[slot] > budget || self.storage.failed_budget[slot] >= budget {
            return EMPTY;
        }
        let mut best = EMPTY;
        // Measure the direct transform first. It often bounds out every
        // smaller subdivision, while the explicit priority preserves old ties.
        for tile in self.storage.direct[slot] {
            if tile == NO_TILE {
                continue;
            }
            if self.storage.tiles[tile].cost.is_nan() {
                let (strategy, _) = self.storage.costs[tile];
                let region = self.storage.tiles[tile];
                let cost = (self.evaluate)(tile, strategy, region.x, region.y);
                let charge = if matches!(strategy, STRATEGY_IDENTITY | STRATEGY_DCT2X2) {
                    self.lambda * learned_fine_rate_correction(cost.rate)
                } else {
                    0.0
                };
                self.storage.tiles[tile].cost = cost.cost + charge;
                debug_assert!(
                    cost.cost + charge >= self.storage.lower[slot],
                    "reconstruction cost fell below its search bound"
                );
                self.storage.costs[tile].1 = cost;
            }
            let cost = self.storage.tiles[tile].cost;
            if cost <= budget && cost < best.cost {
                best = Region {
                    cost,
                    choice: Choice::Tile(tile),
                };
            }
        }
        for split_x in [true, false] {
            if (split_x && lx == 0) || (!split_x && ly == 0) {
                continue;
            }
            let (clx, cly) = if split_x { (lx - 1, ly) } else { (lx, ly - 1) };
            let (rx, ry) = if split_x {
                (x + (1 << clx), y)
            } else {
                (x, y + (1 << cly))
            };
            let left_lower = self.storage.lower[region_index(clx, cly, x, y)];
            let right_lower = self.storage.lower[region_index(clx, cly, rx, ry)];
            let limit = budget.min(best.cost);
            if left_lower + right_lower > limit {
                continue;
            }
            let left = self.search(x, y, clx, cly, remaining_budget(limit, right_lower));
            if !left.cost.is_finite() {
                continue;
            }
            let right = self.search(rx, ry, clx, cly, remaining_budget(limit, left.cost));
            if !right.cost.is_finite() {
                continue;
            }
            let cost = left.cost + right.cost;
            let choice = if split_x {
                Choice::SplitX
            } else {
                Choice::SplitY
            };
            if cost <= budget
                && (cost < best.cost
                    || (cost == best.cost && priority(choice) < priority(best.choice)))
            {
                best = Region { cost, choice };
            }
        }
        if best.cost.is_finite() {
            self.storage.regions[slot] = best;
            self.storage.lower[slot] = best.cost;
        } else {
            self.storage.failed_budget[slot] = budget;
        }
        best
    }
}

/// Lazy version of the same aligned-partition DP. Every transform keeps its
/// original cost; a positive per-transform metadata charge bounds the number
/// of useful leaves before their forward/inverse transforms are evaluated.
fn bounded_plan(
    storage: &mut Recon64Scratch,
    size: (usize, usize),
    budget: f32,
    metadata_cost: f32,
    roundoff_per_block: f32,
    lambda: f32,
    evaluate: impl FnMut(usize, u8, usize, usize) -> ReconStrategyCost,
) -> Plan {
    let (width, height) = size;
    storage.regions.clear();
    storage.regions.resize(16 * 64, EMPTY);
    storage.direct.clear();
    storage.direct.resize(16 * 64, [NO_TILE; 2]);
    storage.lower.clear();
    storage.lower.resize(16 * 64, f32::INFINITY);
    storage.failed_budget.clear();
    storage.failed_budget.resize(16 * 64, f32::NEG_INFINITY);
    for (i, tile) in storage.tiles.iter().enumerate() {
        let slot = region_index(
            tile.width.ilog2() as usize,
            tile.height.ilog2() as usize,
            tile.x,
            tile.y,
        );
        let direct = &mut storage.direct[slot];
        if direct[0] == NO_TILE {
            direct[0] = i;
        } else {
            assert_eq!(direct[1], NO_TILE);
            direct[1] = i;
        }
        // Rates, hue error and the fine-transform charge are nonnegative.
        // SSIM is nonnegative in exact arithmetic; leave a conservative
        // allowance for the floating-point deficit of an identical window.
        let lower = if tile.cost.is_nan() {
            (metadata_cost + lambda * storage.rate_bounds.get(i).copied().unwrap_or(0.0)
                - roundoff_per_block * (tile.width * tile.height) as f32)
                .next_down()
        } else {
            tile.cost
        };
        storage.lower[slot] = storage.lower[slot].min(lower);
    }
    for ly in 0..=height.ilog2() as usize {
        for lx in 0..=width.ilog2() as usize {
            for y in (0..height).step_by(1 << ly) {
                for x in (0..width).step_by(1 << lx) {
                    let slot = region_index(lx, ly, x, y);
                    if lx > 0 {
                        let cost = storage.lower[region_index(lx - 1, ly, x, y)]
                            + storage.lower[region_index(lx - 1, ly, x + (1 << (lx - 1)), y)];
                        storage.lower[slot] = storage.lower[slot].min(cost.next_down());
                    }
                    if ly > 0 {
                        let cost = storage.lower[region_index(lx, ly - 1, x, y)]
                            + storage.lower[region_index(lx, ly - 1, x, y + (1 << (ly - 1)))];
                        storage.lower[slot] = storage.lower[slot].min(cost.next_down());
                    }
                }
            }
        }
    }
    let mut search = BudgetSearch {
        storage,
        evaluate,
        lambda,
    };
    let result = search.search(
        0,
        0,
        width.ilog2() as usize,
        height.ilog2() as usize,
        budget,
    );
    if result.cost.is_finite() {
        finish_plan(search.storage, width, height)
    } else {
        Plan {
            cost: f32::INFINITY,
            tiles: [None; 64],
        }
    }
}

pub(super) fn find_downgrade(
    rerank: &RerankContext<'_>,
    scratch: &mut CoderScratch,
    ac_strategy: &AcStrategyImage,
    bx: usize,
    by: usize,
    strategy: u8,
    output: &mut AcStrategyBandScratch,
) {
    let params = rerank.params;
    let ctx = params.ctx;
    let (width, height) = (
        AcStrategyImage::covered_blocks_x_of(strategy),
        AcStrategyImage::covered_blocks_y_of(strategy),
    );
    let score = |scratch: &mut CoderScratch, s: u8, x: usize, y: usize| {
        let w = AcStrategyImage::covered_blocks_x_of(s);
        let h = AcStrategyImage::covered_blocks_y_of(s);
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
                w,
                h,
                params.scale,
                params.distance,
            ),
            params.qm_mult_x,
            RERANK_META_R,
            params.distance,
            cmap_factors(ctx.cfl_frame(), params.ytox_map, params.ytob_map, x, y),
            None,
        )
    };
    let incumbent = score(scratch, strategy, bx, by);
    let rectangles = SearchScope::for_speed(ctx.speed).rectangles();
    let storage = &mut output.recon64;
    storage.tiles.clear();
    storage.costs.clear();
    storage.rate_bounds.clear();
    for y in 0..height {
        for x in 0..width {
            let leaf = rerank
                .leaves
                .get((by + y) * ac_strategy.xsize() + bx + x)
                .copied()
                .unwrap_or(STRATEGY_DCT);
            for (k, s) in [
                STRATEGY_DCT,
                leaf,
                STRATEGY_DCT16X8,
                STRATEGY_DCT8X16,
                STRATEGY_DCT16X16,
                STRATEGY_DCT32X16,
                STRATEGY_DCT16X32,
                STRATEGY_DCT32X32,
                STRATEGY_DCT32X8,
                STRATEGY_DCT8X32,
                STRATEGY_DCT64X32,
                STRATEGY_DCT32X64,
            ]
            .into_iter()
            .enumerate()
            {
                if k == 1 && s == STRATEGY_DCT {
                    continue;
                }
                let w = AcStrategyImage::covered_blocks_x_of(s);
                let h = AcStrategyImage::covered_blocks_y_of(s);
                if x + w > width
                    || y + h > height
                    || (w == width && h == height)
                    || !x.is_multiple_of(w)
                    || !y.is_multiple_of(h)
                    || (!rectangles && w != h)
                    || !ac_strategy.can_place_strategy(bx + x, by + y, s)
                {
                    continue;
                }
                storage.tiles.push(PartitionTile {
                    x,
                    y,
                    width: w,
                    height: h,
                    cost: f32::NAN,
                });
                storage.costs.push((s, UNMEASURED));
                let qac = region_qac(
                    rerank.quant_field,
                    bx + x,
                    by + y,
                    w,
                    h,
                    params.scale,
                    params.distance,
                );
                storage.rate_bounds.push(
                    scratch
                        .rate_prices
                        .as_deref()
                        .and_then(|prices| prices.minimum_rate(s, qac, params.distance))
                        .unwrap_or(0.0),
                );
            }
        }
    }
    let lambda = reconstruction_lambda(params.distance);
    let plan = bounded_plan(
        storage,
        (width, height),
        incumbent.cost * ctx.merge.rerank_margin,
        lambda * RERANK_META_R,
        SSIM_ROUNDOFF_PER_WINDOW * ctx.channel_weights().iter().sum::<f32>(),
        lambda,
        |_, s, x, y| score(scratch, s, bx + x, by + y),
    );
    if plan.cost < incumbent.cost * ctx.merge.rerank_margin {
        let mut restore = [NO_CHILD_BLOCK; 64];
        for (i, tile) in plan.tiles.into_iter().enumerate() {
            if let Some(tile) = tile {
                let (s, cost) = storage.costs[tile];
                restore[i] = s;
                output.current_costs.push(CachedQuantCost {
                    bx: bx + i % 8,
                    by: by + i / 8,
                    cost: cost.base,
                });
            }
        }
        output
            .rerank64_downgrades
            .push(crate::coder_scratch::Rerank64Downgrade {
                bx,
                by,
                cov_x: width,
                cov_y: height,
                restore,
            });
    } else {
        output.current_costs.push(CachedQuantCost {
            bx,
            by,
            cost: incumbent.base,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_search_preserves_exhaustive_plans_costs_and_ties() {
        let mut state = 91u32;
        for (width, height) in [(4usize, 4usize), (8, 4), (4, 8), (8, 8)] {
            for case in 0..48 {
                let mut storage = Recon64Scratch::default();
                let mut actual_costs = Vec::new();
                for y in 0..height {
                    for x in 0..width {
                        for (w, h) in [
                            (1, 1),
                            (1, 2),
                            (2, 1),
                            (2, 2),
                            (1, 4),
                            (4, 1),
                            (2, 4),
                            (4, 2),
                            (4, 4),
                            (4, 8),
                            (8, 4),
                        ] {
                            if x + w > width
                                || y + h > height
                                || (w == width && h == height)
                                || !x.is_multiple_of(w)
                                || !y.is_multiple_of(h)
                            {
                                continue;
                            }
                            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            // Exact ties, and non-integer costs near rounding boundaries.
                            let cost = if case % 2 == 0 {
                                2.0 + (state % 13) as f32
                            } else {
                                2.0 + (state % 97) as f32 * 0.0000317
                            };
                            storage.tiles.push(PartitionTile {
                                x,
                                y,
                                width: w,
                                height: h,
                                cost,
                            });
                            actual_costs.push(cost);
                            storage.costs.push((STRATEGY_DCT, UNMEASURED));
                        }
                    }
                }
                let expected = best_plan(&mut storage, width, height);
                for budget in [
                    expected.cost - 0.00001,
                    expected.cost,
                    expected.cost.next_up(),
                    expected.cost * 1.1,
                ] {
                    for tile in &mut storage.tiles {
                        tile.cost = f32::NAN;
                    }
                    let mut visits = vec![0; storage.tiles.len()];
                    let actual = bounded_plan(
                        &mut storage,
                        (width, height),
                        budget,
                        2.0,
                        0.001,
                        1.0,
                        |i, _, _, _| {
                            visits[i] += 1;
                            ReconStrategyCost {
                                cost: actual_costs[i],
                                base: actual_costs[i],
                                distortion: 0.0,
                                rate: 0.0,
                                hue_distortion: 0.0,
                            }
                        },
                    );
                    assert!(visits.iter().all(|&n| n <= 1));
                    if expected.cost <= budget {
                        assert_eq!(
                            actual.cost.to_bits(),
                            expected.cost.to_bits(),
                            "{width}x{height} case={case}"
                        );
                        assert_eq!(actual.tiles, expected.tiles, "{width}x{height} case={case}");
                    } else {
                        assert!(actual.cost.is_infinite());
                    }
                }
            }
        }
    }

    #[test]
    fn reranks_sparse_large_blocks_across_bands_and_retains_smooth_incumbents() {
        let distance = 4.8;
        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            distance,
            1,
        );
        for strategy in [STRATEGY_DCT64X64, STRATEGY_DCT64X32, STRATEGY_DCT32X64] {
            let w = AcStrategyImage::covered_blocks_x_of(strategy);
            let h = AcStrategyImage::covered_blocks_y_of(strategy);
            let (point_x, point_y) = (w * 8 - 9, h * 8 - 9);
            let qf = ImageB::new_fill(8, 8, 5);
            let maps = ImageSB::new_fill(1, 1, 0);
            let mut source = Image3F::new(64, 64);
            for c in [1, 2] {
                source.plane_mut(c).as_mut_slice().fill(0.05);
            }
            let mut map = AcStrategyImage::new(8, 8);
            map.set_first(0, 0, strategy);
            let mut leaves = vec![STRATEGY_DCT; 64];
            leaves[(point_y / 8) * 8 + point_x / 8] = STRATEGY_IDENTITY;
            let mut scratch = CoderScratch::default();
            let mut output = AcStrategyBandScratch::default();
            for with_point in [true, false] {
                for c in [1, 2] {
                    source.plane_row_mut(c, point_y)[point_x] = if with_point { 0.4 } else { 0.05 };
                }
                let rerank = RerankContext {
                    params: AcStrategyParams {
                        ctx: &ctx,
                        opsin: &source,
                        dc_group_px: 0,
                        dc_group_py: 0,
                        distance,
                        scale: 0.016662598,
                        qm_mult_x: 1.25,
                        ytox_map: &maps,
                        ytob_map: &maps,
                        prices: None,
                    },
                    quant_field: &qf,
                    bands: &[(0, 4), (4, 8)],
                    num_threads: 1,
                    leaf_first: true,
                    leaves: &leaves,
                };
                // The first band's parent may cover leaves in the second.
                find_rerank_downgrades(&rerank, &mut scratch, &map, (0, 4), &mut output);
                assert_eq!(
                    !output.rerank64_downgrades.is_empty(),
                    with_point,
                    "strategy {strategy}"
                );
                assert!(output.current_costs.iter().all(|c| c.cost.is_finite()));
                if with_point {
                    let d = &output.rerank64_downgrades[0];
                    let mut covered = [0u8; 64];
                    for y in 0..h {
                        for x in 0..w {
                            let s = d.restore[y * 8 + x];
                            if s == NO_CHILD_BLOCK {
                                continue;
                            }
                            let cw = AcStrategyImage::covered_blocks_x_of(s);
                            let ch = AcStrategyImage::covered_blocks_y_of(s);
                            assert!(x + cw <= w && y + ch <= h);
                            for yy in y..y + ch {
                                for xx in x..x + cw {
                                    covered[yy * 8 + xx] += 1;
                                }
                            }
                        }
                    }
                    for y in 0..8 {
                        for x in 0..8 {
                            assert_eq!(covered[y * 8 + x], u8::from(x < w && y < h));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compact_plan_preserves_the_small_planners_costs_and_ties() {
        let mut state = 17u32;
        for (width, height) in [(1, 2), (2, 1), (2, 2), (4, 2), (2, 4), (4, 4)] {
            for _ in 0..32 {
                let mut storage = Recon64Scratch::default();
                for h in [1, 2, 4] {
                    for w in [1, 2, 4] {
                        if w > width || h > height {
                            continue;
                        }
                        for y in (0..height).step_by(h) {
                            for x in (0..width).step_by(w) {
                                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                storage.tiles.push(PartitionTile {
                                    x,
                                    y,
                                    width: w,
                                    height: h,
                                    cost: ((state >> 20) % 13) as f32,
                                });
                            }
                        }
                    }
                }
                let expected = best_reconstruction_plan(&storage.tiles, width, height);
                let actual = best_plan(&mut storage, width, height);
                assert_eq!(actual.cost.to_bits(), expected.cost.to_bits());
                for y in 0..height {
                    for x in 0..width {
                        let tile = expected.tiles[y * 4 + x];
                        let expected = (tile != NO_CHILD_BLOCK).then_some(tile as usize);
                        assert_eq!(actual.tiles[y * 8 + x], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn plans_mixed_64_footprints_without_overlap_or_truncating_candidate_indices() {
        for (width, height) in [(8, 8), (8, 4), (4, 8)] {
            let mut storage = Recon64Scratch::default();
            // Fill beyond u8 range before adding the useful candidates.
            storage.tiles.resize(
                260,
                PartitionTile {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    cost: 100.0,
                },
            );
            for y in 0..height {
                for x in 0..width {
                    storage.tiles.push(PartitionTile {
                        x,
                        y,
                        width: 1,
                        height: 1,
                        cost: 10.0,
                    });
                }
            }
            storage.tiles.push(PartitionTile {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
                cost: 1.0,
            });
            let plan = best_plan(&mut storage, width, height);
            assert_eq!(plan.cost, 1.0 + (width * height - 16) as f32 * 10.0);
            let mut covered = [0u8; 64];
            for i in plan.tiles.into_iter().flatten() {
                assert!(i >= 260);
                let tile = storage.tiles[i];
                for y in tile.y..tile.y + tile.height {
                    for x in tile.x..tile.x + tile.width {
                        covered[y * 8 + x] += 1;
                    }
                }
            }
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(covered[y * 8 + x], u8::from(y < height && x < width));
                }
            }
        }
    }
}
