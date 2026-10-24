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

//! Frontier DP on an 8-pixel grid inside a 64x64 region.
//!
//! Every 8/16/32-family piece is a direct candidate at every legal 8-pixel
//! origin; the 64-family blocks of the incumbent layout compete only as
//! placed. Retained fine transforms compete with DCT8 at each leaf. A sparse
//! frontier memo covers slicing and nonslicing tilings.
//! Guillotine plans provide feasible upper bounds; independent row/column
//! relaxations provide admissible lower bounds. Two cuts are not admissible:
//! a larger piece is dropped once its luma-and-rate bound exceeds
//! `BOUND_SLACK` of the best tiling of its rectangle by smaller pieces, and a
//! tile that exhausts `VISIT_BUDGET` settles its remaining states for their
//! guillotine plans. Both are measured choices, not approximations to remove:
//! slack 1.0 loses quality and an unbounded budget changes almost nothing.

use super::*;
use std::collections::HashMap;

const TILE: usize = 8;
const SIDE: usize = TILE;
const CACHE_SIZE: usize = crate::dc_group_data::NUM_STRATEGIES * TILE * TILE;
const REGION_COUNT: usize = SIDE * SIDE * (SIDE + 1) * (SIDE + 1);
const LINE_COUNT: usize = SIDE * (SIDE + 1) * (SIDE + 1);
/// The 400 grid placements plus at most two incumbent 64-family blocks.
const MAX_CANDIDATES: usize = 402;
/// Frontier states a tile may expand before the remaining states settle for
/// their guillotine plan.
const VISIT_BUDGET: usize = 5_000;
/// A piece is scored in full only while its luma-and-rate lower bound stays
/// under this fraction of the best tiling of its rectangle by smaller pieces.
const BOUND_SLACK: f32 = 0.9;
const STRATEGIES: [u8; 9] = [
    STRATEGY_DCT,
    STRATEGY_DCT16X8,
    STRATEGY_DCT8X16,
    STRATEGY_DCT16X16,
    STRATEGY_DCT32X8,
    STRATEGY_DCT8X32,
    STRATEGY_DCT32X16,
    STRATEGY_DCT16X32,
    STRATEGY_DCT32X32,
];
/// The coefficient-model pass: DCT8 leaves, pairs and 16x16 squares only.
const COEFFICIENT_STRATEGIES: [u8; 4] = [
    STRATEGY_DCT,
    STRATEGY_DCT16X8,
    STRATEGY_DCT8X16,
    STRATEGY_DCT16X16,
];

#[cfg(test)]
fn for_each_piece(width: usize, height: usize, mut visit: impl FnMut(Piece)) {
    for strategy in STRATEGIES {
        let w = AcStrategyImage::covered_blocks_x_of(strategy);
        let h = AcStrategyImage::covered_blocks_y_of(strategy);
        if w > width || h > height {
            continue;
        }
        for y in 0..=height - h {
            for x in 0..=width - w {
                visit(Piece::new(x, y, strategy));
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Piece {
    x: u8,
    y: u8,
    strategy: u8,
}
impl Piece {
    fn new(x: usize, y: usize, strategy: u8) -> Self {
        Self {
            x: x as u8,
            y: y as u8,
            strategy,
        }
    }
    fn width(self) -> usize {
        AcStrategyImage::covered_blocks_x_of(self.strategy)
    }
    fn height(self) -> usize {
        AcStrategyImage::covered_blocks_y_of(self.strategy)
    }
    fn key(self) -> usize {
        self.strategy as usize * 64 + self.y as usize * 8 + self.x as usize
    }
    #[cfg(test)]
    fn shifted32(self) -> bool {
        matches!(
            self.strategy,
            STRATEGY_DCT32X32 | STRATEGY_DCT32X16 | STRATEGY_DCT16X32
        ) && (!(self.x as usize).is_multiple_of(self.width())
            || !(self.y as usize).is_multiple_of(self.height()))
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    piece: Piece,
    width: usize,
    height: usize,
    cost: f64,
}
impl Candidate {
    fn new(piece: Piece, cost: f64) -> Self {
        Self {
            piece,
            width: piece.width(),
            height: piece.height(),
            cost,
        }
    }
    fn x(self) -> usize {
        self.piece.x as usize
    }
    fn y(self) -> usize {
        self.piece.y as usize
    }
}

#[derive(Clone, Copy, Debug)]
enum Choice {
    None,
    Tile(usize),
    Vertical(usize),
    Horizontal(usize),
    Fallback,
}
#[derive(Clone, Copy, Debug)]
struct Entry {
    cost: f64,
    choice: Choice,
}
const EMPTY: Entry = Entry {
    cost: f64::INFINITY,
    choice: Choice::None,
};

#[derive(Clone, Copy)]
enum Memo {
    Solved(Entry),
    // An admissible relaxation or a failed caller budget. It need not be
    // solved completely unless a later prefix can beat this lower bound.
    AtLeast(f64),
}

#[derive(Default)]
struct PlannerScratch {
    memo: HashMap<u32, Memo>,
    regions: Vec<Entry>,
    anchors: Vec<Vec<usize>>,
    columns: [[f64; SIDE + 1]; SIDE],
    rows: Vec<f64>,
    packing: [[f64; SIDE + 1]; SIDE],
    packing_valid: bool,
    dominated: Vec<bool>,
    tableau: Vec<f64>,
    budget: usize,
}

fn region_index(x: usize, y: usize, w: usize, h: usize) -> usize {
    ((y * SIDE + x) * (SIDE + 1) + h) * (SIDE + 1) + w
}
fn line_index(y: usize, start: usize, end: usize) -> usize {
    (y * (SIDE + 1) + start) * (SIDE + 1) + end
}
fn profile_key(profile: [u8; SIDE]) -> u32 {
    profile
        .into_iter()
        .enumerate()
        .fold(0, |key, (x, y)| key | ((y as u32) << (x * 4)))
}
fn first_open(profile: [u8; SIDE], width: usize) -> (usize, usize) {
    let (x, &y) = profile[..width]
        .iter()
        .enumerate()
        .min_by_key(|&(x, &y)| (y, x))
        .unwrap();
    (x, y as usize)
}
fn advance(mut profile: [u8; SIDE], c: Candidate) -> [u8; SIDE] {
    profile[c.x()..c.x() + c.width].fill((c.y() + c.height) as u8);
    profile
}

// Error-free TwoSum identifies upward rounding. Keep exact additions exact,
// including ties, and round an inexact upward result down for admissible bounds.
fn add_lower(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if !sum.is_finite() {
        return sum;
    }
    let bv = sum - a;
    let error = (a - (sum - bv)) + (b - bv);
    if error < 0.0 { sum.next_down() } else { sum }
}

fn prepare_regions(candidates: &[Candidate], width: usize, height: usize, s: &mut PlannerScratch) {
    s.regions.clear();
    s.regions.resize(REGION_COUNT, EMPTY);
    for (i, &c) in candidates.iter().enumerate() {
        if !c.cost.is_finite() {
            continue;
        }
        let r = &mut s.regions[region_index(c.x(), c.y(), c.width, c.height)];
        if c.cost < r.cost {
            *r = Entry {
                cost: c.cost,
                choice: Choice::Tile(i),
            };
        }
    }
    for h in 1..=height {
        for w in 1..=width {
            for y in 0..=height - h {
                for x in 0..=width - w {
                    let slot = region_index(x, y, w, h);
                    let mut best = s.regions[slot];
                    for cut in 1..w {
                        let cost = s.regions[region_index(x, y, cut, h)].cost
                            + s.regions[region_index(x + cut, y, w - cut, h)].cost;
                        if cost < best.cost {
                            best = Entry {
                                cost,
                                choice: Choice::Vertical(cut),
                            };
                        }
                    }
                    for cut in 1..h {
                        let cost = s.regions[region_index(x, y, w, cut)].cost
                            + s.regions[region_index(x, y + cut, w, h - cut)].cost;
                        if cost < best.cost {
                            best = Entry {
                                cost,
                                choice: Choice::Horizontal(cut),
                            };
                        }
                    }
                    s.regions[slot] = best;
                }
            }
        }
    }
}

// A fractional packing maximizes savings over the best all-leaf tiling.
// Slack-column prices of its dual give one feasible per-cell lower bound.
// The simplex work is bounded, but the final dual is repaired against every
// candidate: early stopping only weakens pruning, never limits the exact DP.
fn prepare_packing_bound(
    candidates: &[Candidate],
    width: usize,
    height: usize,
    s: &mut PlannerScratch,
) {
    const PIVOTS: usize = 512;
    let cells = width * height;
    let mut leaves = [f64::INFINITY; SIDE * SIDE];
    for &c in candidates {
        if c.width == 1 && c.height == 1 && c.cost.is_finite() {
            let i = c.y() * width + c.x();
            leaves[i] = leaves[i].min(c.cost);
        }
    }
    s.packing_valid = false;
    s.dominated.clear();
    s.dominated.resize(candidates.len(), false);
    if leaves[..cells].iter().any(|x| !x.is_finite()) {
        return;
    }
    let mut variables = Vec::new();
    for (k, &c) in candidates.iter().enumerate() {
        if !c.cost.is_finite() {
            continue;
        }
        let mut baseline = 0.0;
        let mut baseline_upper = 0.0;
        for y in c.y()..c.y() + c.height {
            for x in c.x()..c.x() + c.width {
                baseline += leaves[y * width + x];
                baseline_upper = -add_lower(-baseline_upper, -leaves[y * width + x]);
            }
        }
        if c.width * c.height > 1 && c.cost >= baseline_upper {
            s.dominated[k] = true;
        } else if c.width * c.height > 1 {
            variables.push((k, baseline - c.cost));
        }
    }
    let n = variables.len();
    let cols = n + cells + 1;
    let rows = cells + 1;
    let objective = cells * cols;
    s.tableau.clear();
    s.tableau.resize(rows * cols, 0.0);
    let mut basis: Vec<_> = (n..n + cells).collect();
    for (v, &(k, benefit)) in variables.iter().enumerate() {
        let c = candidates[k];
        for y in c.y()..c.y() + c.height {
            for x in c.x()..c.x() + c.width {
                s.tableau[(y * width + x) * cols + v] = 1.0;
            }
        }
        s.tableau[objective + v] = -benefit;
    }
    for row in 0..cells {
        s.tableau[row * cols + n + row] = 1.0;
        s.tableau[row * cols + cols - 1] = 1.0;
    }
    for _ in 0..PIVOTS {
        let mut enter = None;
        let mut reduced = -1e-10;
        for col in 0..cols - 1 {
            let value = s.tableau[objective + col];
            if value < reduced {
                reduced = value;
                enter = Some(col);
            }
        }
        let Some(enter) = enter else {
            break;
        };
        let mut leave = None;
        let mut ratio = f64::INFINITY;
        for row in 0..cells {
            let a = s.tableau[row * cols + enter];
            if a <= 1e-12 {
                continue;
            }
            let r = s.tableau[row * cols + cols - 1] / a;
            if r < ratio - 1e-12
                || ((r - ratio).abs() <= 1e-12 && leave.is_none_or(|old| basis[row] < basis[old]))
            {
                ratio = r;
                leave = Some(row);
            }
        }
        let Some(leave) = leave else {
            break;
        };
        let pivot = s.tableau[leave * cols + enter];
        for col in 0..cols {
            s.tableau[leave * cols + col] /= pivot;
        }
        s.tableau[leave * cols + enter] = 1.0;
        for row in 0..rows {
            if row == leave {
                continue;
            }
            let factor = s.tableau[row * cols + enter];
            if factor == 0.0 {
                continue;
            }
            for col in 0..cols {
                s.tableau[row * cols + col] -= factor * s.tableau[leave * cols + col];
            }
            s.tableau[row * cols + enter] = 0.0;
        }
        basis[leave] = enter;
    }
    let mut potentials = [0.0; SIDE * SIDE];
    for i in 0..cells {
        let price = s.tableau[objective + n + i].max(0.0);
        potentials[i] = leaves[i] - price;
    }
    if potentials[..cells].iter().any(|x| !x.is_finite()) {
        return;
    }
    // For every tile, sum(cell potentials) <= tile cost is the dual constraint.
    // Only lower a potential when repairing, so earlier constraints stay valid.
    for &c in candidates {
        if !c.cost.is_finite() {
            continue;
        }
        let mut sum = 0.0;
        for y in c.y()..c.y() + c.height {
            for x in c.x()..c.x() + c.width {
                sum = -add_lower(-sum, -potentials[y * width + x]);
            }
        }
        if sum > c.cost {
            let excess = -add_lower(-sum, c.cost);
            let i = c.y() * width + c.x();
            potentials[i] = add_lower(potentials[i], -excess);
        }
    }
    for x in 0..width {
        s.packing[x][height] = 0.0;
        for y in (0..height).rev() {
            s.packing[x][y] = add_lower(potentials[y * width + x], s.packing[x][y + 1]);
        }
    }
    s.packing_valid = true;
}

fn prepare_bounds(candidates: &[Candidate], width: usize, height: usize, s: &mut PlannerScratch) {
    // Give each column an independent vertical tiling and divide a rectangle's
    // cost across its width. Ignoring agreement between columns can only lower
    // the optimum. Repeat horizontally with cost divided by height; either
    // relaxation bounds every legal completion, so their maximum is valid.
    let mut vertical = [[[f64::INFINITY; SIDE + 1]; SIDE + 1]; SIDE];
    let mut horizontal = vertical;
    for &c in candidates {
        if !c.cost.is_finite() {
            continue;
        }
        // Widths and heights are powers of two, so f32 candidate costs can be
        // divided exactly after promotion to f64.
        for column in vertical.iter_mut().skip(c.x()).take(c.width) {
            column[c.y()][c.height] = column[c.y()][c.height].min(c.cost / c.width as f64);
        }
        for row in horizontal.iter_mut().skip(c.y()).take(c.height) {
            row[c.x()][c.width] = row[c.x()][c.width].min(c.cost / c.height as f64);
        }
    }
    for (x, edges) in vertical.iter().enumerate().take(width) {
        s.columns[x].fill(f64::INFINITY);
        s.columns[x][height] = 0.0;
        for y in (0..height).rev() {
            for h in 1..=height - y {
                s.columns[x][y] = s.columns[x][y].min(add_lower(edges[y][h], s.columns[x][y + h]));
            }
        }
    }
    s.rows.clear();
    s.rows.resize(LINE_COUNT, f64::INFINITY);
    for (y, edges) in horizontal.iter().enumerate().take(height) {
        for end in 0..=width {
            s.rows[line_index(y, end, end)] = 0.0;
            for x in (0..end).rev() {
                let mut best = f64::INFINITY;
                for w in 1..=end - x {
                    best = best.min(add_lower(edges[x][w], s.rows[line_index(y, x + w, end)]));
                }
                s.rows[line_index(y, x, end)] = best;
            }
        }
    }
}

fn lower_bound(profile: [u8; SIDE], width: usize, height: usize, s: &PlannerScratch) -> f64 {
    let mut columns = 0.0;
    for (x, &y) in profile.iter().enumerate().take(width) {
        columns = add_lower(columns, s.columns[x][y as usize]);
    }
    let mut rows = 0.0;
    for y in 0..height {
        let mut x = 0;
        while x < width {
            if profile[x] as usize > y {
                x += 1;
                continue;
            }
            let start = x;
            while x < width && profile[x] as usize <= y {
                x += 1;
            }
            rows = add_lower(rows, s.rows[line_index(y, start, x)]);
        }
    }
    let mut packing = f64::NEG_INFINITY;
    if s.packing_valid {
        packing = 0.0;
        for (x, &y) in profile.iter().enumerate().take(width) {
            packing = add_lower(packing, s.packing[x][y as usize]);
        }
    }
    columns.max(rows).max(packing)
}

fn fallback_cost(profile: [u8; SIDE], width: usize, height: usize, s: &PlannerScratch) -> f64 {
    let mut cost = 0.0;
    let mut x = 0;
    while x < width {
        let start = x;
        let y = profile[x] as usize;
        while x < width && profile[x] as usize == y {
            x += 1;
        }
        if y < height {
            cost += s.regions[region_index(start, y, x - start, height - y)].cost;
        }
    }
    cost
}

fn visit(
    profile: [u8; SIDE],
    width: usize,
    height: usize,
    budget: f64,
    candidates: &[Candidate],
    s: &mut PlannerScratch,
) -> Option<Entry> {
    let key = profile_key(profile);
    match s.memo.get(&key).copied() {
        Some(Memo::Solved(best)) => return Some(best),
        Some(Memo::AtLeast(bound)) if bound >= budget => return None,
        _ => {}
    }
    let (x, y) = first_open(profile, width);
    if y == height {
        return Some(Entry {
            cost: 0.0,
            choice: Choice::None,
        });
    }
    let mut best = Entry {
        cost: fallback_cost(profile, width, height, s),
        choice: Choice::Fallback,
    };
    let anchor = y * SIDE + x;
    let expand = if s.budget == 0 {
        0
    } else {
        s.budget -= 1;
        s.anchors[anchor].len()
    };
    for i in 0..expand {
        let k = s.anchors[anchor][i];
        let c = candidates[k];
        if !c.cost.is_finite() || profile[x..x + c.width].iter().any(|&h| h as usize != y) {
            continue;
        }
        let next = advance(profile, c);
        let bound = match s.memo.get(&profile_key(next)).copied() {
            Some(Memo::Solved(entry)) => entry.cost,
            Some(Memo::AtLeast(bound)) => bound,
            None => {
                let bound = lower_bound(next, width, height, s);
                s.memo.insert(profile_key(next), Memo::AtLeast(bound));
                bound
            }
        };
        let limit = best.cost.min(budget);
        if add_lower(c.cost, bound) >= limit {
            continue;
        }
        // Round the residual budget upwards: rounding must never exclude a
        // suffix that could improve the parent's complete layout.
        let remaining = -add_lower(-limit, c.cost);
        if let Some(suffix) = visit(next, width, height, remaining, candidates, s) {
            let cost = c.cost + suffix.cost;
            if cost < best.cost {
                best = Entry {
                    cost,
                    choice: Choice::Tile(k),
                };
            }
        }
    }
    if best.cost <= budget && best.cost.is_finite() {
        s.memo.insert(key, Memo::Solved(best));
        Some(best)
    } else {
        s.memo.insert(key, Memo::AtLeast(budget));
        None
    }
}

fn unpack_region(x: usize, y: usize, w: usize, h: usize, s: &PlannerScratch, out: &mut Vec<usize>) {
    match s.regions[region_index(x, y, w, h)].choice {
        Choice::Tile(k) => out.push(k),
        Choice::Vertical(cut) => {
            unpack_region(x, y, cut, h, s, out);
            unpack_region(x + cut, y, w - cut, h, s, out);
        }
        Choice::Horizontal(cut) => {
            unpack_region(x, y, w, cut, s, out);
            unpack_region(x, y + cut, w, h - cut, s, out);
        }
        _ => unreachable!("infeasible rectangle in a finite fallback"),
    }
}

fn plan(
    candidates: &[Candidate],
    width: usize,
    height: usize,
    s: &mut PlannerScratch,
    out: &mut Vec<usize>,
) -> f64 {
    assert!((1..=SIDE).contains(&width) && (1..=SIDE).contains(&height));
    s.memo.clear();
    s.budget = VISIT_BUDGET;
    s.anchors.resize_with(SIDE * SIDE, Vec::new);
    for anchor in &mut s.anchors {
        anchor.clear();
    }
    for &c in candidates {
        assert!(c.x() + c.width <= width && c.y() + c.height <= height);
    }
    prepare_packing_bound(candidates, width, height, s);
    for (i, &c) in candidates.iter().enumerate() {
        if !s.dominated[i] {
            s.anchors[c.y() * SIDE + c.x()].push(i);
        }
    }
    prepare_regions(candidates, width, height, s);
    prepare_bounds(candidates, width, height, s);
    let root = visit([0; SIDE], width, height, f64::INFINITY, candidates, s).unwrap_or(EMPTY);
    out.clear();
    if !root.cost.is_finite() {
        return root.cost;
    }
    let mut profile = [0; SIDE];
    while first_open(profile, width).1 < height {
        let Memo::Solved(entry) = s.memo[&profile_key(profile)] else {
            unreachable!("a winning backpointer must have an exact suffix");
        };
        match entry.choice {
            Choice::Tile(k) => {
                out.push(k);
                profile = advance(profile, candidates[k]);
            }
            Choice::Fallback => {
                let mut x = 0;
                while x < width {
                    let start = x;
                    let y = profile[x] as usize;
                    while x < width && profile[x] as usize == y {
                        x += 1;
                    }
                    if y < height {
                        unpack_region(start, y, x - start, height - y, s, out);
                    }
                }
                break;
            }
            _ => unreachable!("invalid frontier backpointer"),
        }
    }
    root.cost
}

const UNMEASURED: ReconStrategyCost = ReconStrategyCost {
    cost: f32::NAN,
    base: f32::NAN,
    distortion: f32::NAN,
    rate: f32::NAN,
    hue_distortion: f32::NAN,
};
#[derive(Default)]
pub(crate) struct LayoutDpScratch {
    reconstruction: Vec<ReconStrategyCost>,
    candidates: Vec<Candidate>,
    planner: PlannerScratch,
    current: Vec<Piece>,
    layout: Vec<Piece>,
    choices: Vec<usize>,
    best: Vec<MergeUpgrade>,
    small: SmallPlanner,
    proposals: Vec<crate::dc_group_data::DpRegion>,
}
impl LayoutDpScratch {
    fn reset(&mut self) {
        self.reconstruction.resize(CACHE_SIZE, UNMEASURED);
        self.reconstruction.fill(UNMEASURED);
        self.candidates.clear();
        self.choices.clear();
        self.best.clear();
        self.proposals.clear();
    }
}
struct Scores<'a, 'b> {
    rerank: &'a RerankContext<'b>,
    bx: usize,
    by: usize,
}

impl Scores<'_, '_> {
    fn reconstruction(
        &self,
        coder: &mut CoderScratch,
        cache: &mut LayoutDpScratch,
        p: Piece,
    ) -> ReconStrategyCost {
        self.reconstruction_bounded(coder, cache, p, f32::INFINITY)
    }

    /// Scores `p`, giving up with an infinite cost once its luma channel
    /// alone exceeds `limit`.
    fn reconstruction_bounded(
        &self,
        coder: &mut CoderScratch,
        cache: &mut LayoutDpScratch,
        p: Piece,
        limit: f32,
    ) -> ReconStrategyCost {
        let key = p.key();
        if cache.reconstruction[key].cost.is_nan() {
            let params = self.rerank.params;
            let (x, y) = (self.bx + p.x as usize, self.by + p.y as usize);
            let (px, py) = (params.dc_group_px + x * 8, params.dc_group_py + y * 8);
            let (ga, gpa) = rerank_pair_gradient_alphas(self.rerank, p.strategy, px, py);
            let mut score = reconstruction_strategy_cost_bounded(
                params.ctx,
                coder,
                p.strategy,
                params.opsin,
                px,
                py,
                region_qac(
                    self.rerank.quant_field,
                    x,
                    y,
                    p.width(),
                    p.height(),
                    params.scale,
                    params.distance,
                ),
                params.qm_mult_x,
                RERANK_META_R,
                params.distance,
                cmap_factors(
                    params.ctx.cfl_frame(),
                    params.ytox_map,
                    params.ytob_map,
                    x,
                    y,
                ),
                ga,
                gpa,
                None,
                limit,
            );
            if crate::dc_group_data::is_gated_sub8_strategy(p.strategy) {
                let (lambda, bits) =
                    fine_admit_charge(coder.rate_prices.is_some(), params.distance, score.rate);
                score.cost = fmla(lambda, bits, score.cost);
            }
            cache.reconstruction[key] = score;
        }
        cache.reconstruction[key]
    }
}

fn leaf(
    scores: &Scores<'_, '_>,
    coder: &mut CoderScratch,
    cache: &mut LayoutDpScratch,
    map: &AcStrategyImage,
    x: usize,
    y: usize,
) -> Candidate {
    let mut best = Piece::new(x, y, STRATEGY_DCT);
    let mut cost = scores.reconstruction(coder, cache, best).cost;
    let (bx, by) = (scores.bx + x, scores.by + y);
    let current = map.raw_strategy(bx, by);
    let retained = scores
        .rerank
        .leaves
        .get(by * map.xsize() + bx)
        .copied()
        .unwrap_or(STRATEGY_DCT);
    for strategy in [current, retained] {
        if strategy as usize >= crate::dc_group_data::NUM_STRATEGIES
            || AcStrategyImage::covered_blocks_x_of(strategy)
                * AcStrategyImage::covered_blocks_y_of(strategy)
                != 1
        {
            continue;
        }
        let p = Piece::new(x, y, strategy);
        let trial = scores.reconstruction(coder, cache, p).cost;
        if trial < cost {
            cost = trial;
            best = p;
        }
    }
    Candidate::new(best, cost as f64)
}
/// The minimal closed rectangles (x, y, w, h) covering every block whose
/// piece differs between `current` and `layout`: connected changed blocks
/// grown until no piece of either layout crosses the border, with
/// overlapping rectangles merged.
fn changed_regions(current: &[Piece], layout: &[Piece]) -> Vec<(usize, usize, usize, usize)> {
    let footprint = |p: &Piece| (p.x as usize, p.y as usize, p.width(), p.height());
    let mut changed = [false; 64];
    for p in layout
        .iter()
        .filter(|p| !current.contains(p))
        .chain(current.iter().filter(|p| !layout.contains(p)))
    {
        let (x, y, w, h) = footprint(p);
        for yy in y..y + h {
            for xx in x..x + w {
                changed[yy * 8 + xx] = true;
            }
        }
    }
    let close = |mut r: (usize, usize, usize, usize)| {
        loop {
            let mut grew = false;
            for p in layout.iter().chain(current.iter()) {
                let (x, y, w, h) = footprint(p);
                let overlaps = x < r.0 + r.2 && r.0 < x + w && y < r.1 + r.3 && r.1 < y + h;
                let inside = x >= r.0 && x + w <= r.0 + r.2 && y >= r.1 && y + h <= r.1 + r.3;
                if overlaps && !inside {
                    let (x0, y0) = (r.0.min(x), r.1.min(y));
                    let (x1, y1) = ((r.0 + r.2).max(x + w), (r.1 + r.3).max(y + h));
                    r = (x0, y0, x1 - x0, y1 - y0);
                    grew = true;
                }
            }
            if !grew {
                return r;
            }
        }
    };
    let mut seen = [false; 64];
    let mut regions: Vec<(usize, usize, usize, usize)> = Vec::new();
    for start in 0..64 {
        if !changed[start] || seen[start] {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (8usize, 8usize, 0usize, 0usize);
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % 8, i / 8);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            let mut neighbours = Vec::with_capacity(4);
            if x > 0 {
                neighbours.push(i - 1);
            }
            if x < 7 {
                neighbours.push(i + 1);
            }
            if y > 0 {
                neighbours.push(i - 8);
            }
            if y < 7 {
                neighbours.push(i + 8);
            }
            for n in neighbours {
                if changed[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        regions.push(close((x0, y0, x1 - x0, y1 - y0)));
    }
    // Merge overlapping rectangles until they are disjoint.
    loop {
        let mut merged = false;
        'outer: for i in 0..regions.len() {
            for j in i + 1..regions.len() {
                let (a, b) = (regions[i], regions[j]);
                if a.0 < b.0 + b.2 && b.0 < a.0 + a.2 && a.1 < b.1 + b.3 && b.1 < a.1 + a.3 {
                    let (x0, y0) = (a.0.min(b.0), a.1.min(b.1));
                    let (x1, y1) = ((a.0 + a.2).max(b.0 + b.2), (a.1 + a.3).max(b.1 + b.3));
                    regions[i] = close((x0, y0, x1 - x0, y1 - y0));
                    regions.swap_remove(j);
                    merged = true;
                    break 'outer;
                }
            }
        }
        if !merged {
            break;
        }
    }
    regions
}

fn pieces_mask(pieces: &[Piece]) -> u64 {
    let mut mask = 0;
    for p in pieces {
        for y in p.y as usize..p.y as usize + p.height() {
            for x in p.x as usize..p.x as usize + p.width() {
                let bit = 1u64 << (y * 8 + x);
                debug_assert_eq!(mask & bit, 0, "layout overlaps");
                mask |= bit;
            }
        }
    }
    mask
}

fn search_tile(
    scores: &Scores<'_, '_>,
    coder: &mut CoderScratch,
    cache: &mut LayoutDpScratch,
    map: &AcStrategyImage,
    width: usize,
    height: usize,
    accepted: &mut Vec<MergeUpgrade>,
) {
    cache.reset();
    let mut current = std::mem::take(&mut cache.current);
    current.clear();
    for y in 0..height {
        for x in 0..width {
            if map.is_first_block(scores.bx + x, scores.by + y) {
                current.push(Piece::new(
                    x,
                    y,
                    map.raw_strategy(scores.bx + x, scores.by + y),
                ));
            }
        }
    }
    // Legacy fine credits are accumulated before this pass. Leaf-first credits
    // are instead summed from the final map, so removing those leaves is safe.
    if !scores.rerank.leaf_first
        && current
            .iter()
            .any(|p| crate::dc_group_data::is_gated_sub8_strategy(p.strategy))
    {
        cache.current = current;
        return;
    }
    // Larger pieces are scored after everything that can tile their
    // rectangle, so the best such tiling bounds the score: a piece whose
    // luma channel and rate alone cost more than `BOUND_SLACK` of it is
    // dropped before its chroma is scored. The incumbent's pieces are
    // always scored in full.
    let families = if scores.rerank.params.ctx.speed.effort().layout_retile_32 {
        STRATEGIES.len()
    } else {
        COEFFICIENT_STRATEGIES.len()
    };
    for (i, &strategy) in STRATEGIES[..families].iter().enumerate() {
        if matches!(i, 3 | 4 | 6 | 8) {
            prepare_regions(&cache.candidates, width, height, &mut cache.planner);
        }
        let w = AcStrategyImage::covered_blocks_x_of(strategy);
        let h = AcStrategyImage::covered_blocks_y_of(strategy);
        if w > width || h > height {
            continue;
        }
        for y in 0..=height - h {
            for x in 0..=width - w {
                let p = Piece::new(x, y, strategy);
                if !map.can_place_strategy(scores.bx + x, scores.by + y, strategy) {
                    continue;
                }
                let candidate = if strategy == STRATEGY_DCT {
                    leaf(scores, coder, cache, map, x, y)
                } else {
                    let limit = if i < 3 || current.contains(&p) {
                        f32::INFINITY
                    } else {
                        (cache.planner.regions[region_index(x, y, w, h)].cost as f32 * BOUND_SLACK)
                            .next_up()
                    };
                    Candidate::new(
                        p,
                        f64::from(scores.reconstruction_bounded(coder, cache, p, limit).cost),
                    )
                };
                cache.candidates.push(candidate);
            }
        }
    }
    // The incumbent layout stays feasible: its 64-family blocks, which only
    // the 64 pass places, compete as themselves.
    for &p in &current {
        if p.width() * p.height() > 1 && !cache.candidates.iter().any(|c| c.piece == p) {
            let cost = scores.reconstruction(coder, cache, p).cost as f64;
            cache.candidates.push(Candidate::new(p, cost));
        }
    }
    debug_assert!(cache.candidates.len() <= MAX_CANDIDATES);
    let winner = plan(
        &cache.candidates,
        width,
        height,
        &mut cache.planner,
        &mut cache.choices,
    );
    if !winner.is_finite() {
        cache.current = current;
        return;
    }
    let mut layout = std::mem::take(&mut cache.layout);
    layout.clear();
    for &i in &cache.choices {
        layout.push(cache.candidates[i].piece);
    }
    layout.sort_by_key(|p| (p.y, p.x, p.strategy));
    debug_assert_eq!(pieces_mask(&layout), pieces_mask(&current));
    // The incumbent can contain fine leaves outside this pass's candidate set.
    // Compare only changed pieces so unchanged costs cancel without rounding.
    let old = current
        .iter()
        .filter(|p| !layout.contains(p))
        .map(|&p| scores.reconstruction(coder, cache, p).cost)
        .sum::<f32>();
    let new = layout
        .iter()
        .filter(|p| !current.contains(p))
        .map(|&p| scores.reconstruction(coder, cache, p).cost)
        .sum::<f32>();
    if new < old {
        for &p in layout.iter().filter(|p| !current.contains(p)) {
            let base = scores.reconstruction(coder, cache, p).base;
            cache.best.push(MergeUpgrade {
                bx: scores.bx + p.x as usize,
                by: scores.by + p.y as usize,
                strategy: p.strategy,
                base,
            });
        }
        for (x, y, w, h) in changed_regions(&current, &layout) {
            cache.proposals.push(crate::dc_group_data::DpRegion {
                bx: scores.bx + x,
                by: scores.by + y,
                w,
                h,
                pieces: layout
                    .iter()
                    .filter(|p| {
                        (p.x as usize) >= x
                            && (p.x as usize) < x + w
                            && (p.y as usize) >= y
                            && (p.y as usize) < y + h
                    })
                    .map(|p| (p.strategy, p.x as usize - x, p.y as usize - y))
                    .collect(),
            });
        }
    }
    accepted.extend_from_slice(&cache.best);
    cache.current = current;
    cache.layout = layout;
}

/// Coefficient-model scores of a tile: the selection pass's `strategy_cost`
/// with its metadata charge, as the merge search priced its candidates.
struct CoefficientScores<'a, 'b> {
    selection: &'a SelectionContext<'b>,
    bx: usize,
    by: usize,
}

impl CoefficientScores<'_, '_> {
    fn score(&self, coder: &mut CoderScratch, cache: &mut LayoutDpScratch, p: Piece) -> f32 {
        let key = p.key();
        if cache.reconstruction[key].cost.is_nan() {
            let params = self.selection.params;
            let (x, y) = (self.bx + p.x as usize, self.by + p.y as usize);
            let (px, py) = (params.dc_group_px + x * 8, params.dc_group_py + y * 8);
            let cost = strategy_cost(
                params.ctx,
                coder,
                p.strategy,
                params.opsin,
                px,
                py,
                region_qac(
                    self.selection.quant_field,
                    x,
                    y,
                    p.width(),
                    p.height(),
                    params.scale,
                    params.distance,
                ),
                params.qm_mult_x,
                self.selection.meta_r,
                params.distance,
                cmap_factors(
                    params.ctx.cfl_frame(),
                    params.ytox_map,
                    params.ytob_map,
                    x,
                    y,
                ),
            );
            cache.reconstruction[key] = ReconStrategyCost { cost, ..UNMEASURED };
        }
        cache.reconstruction[key].cost
    }
}

/// Choice of the small planner at a cell, in the low bits of a `from` entry.
const SMALL_PREFILLED: u16 = 0;
const SMALL_LEAF: u16 = 1;
const SMALL_WIDE: u16 = 2;
const SMALL_TALL: u16 = 3;
const SMALL_SQUARE: u16 = 4;
const SMALL_STATES: usize = 1 << (SIDE + 2);

/// Exact broken-profile DP over 1x1, 2x1, 1x2 and 2x2 pieces on a `w x h`
/// grid of at most 8x8 cells. Cells are visited in row-major order; a state
/// holds the filled flags of the next `w + 2` cells, which is all a piece
/// anchored at the current cell can reach.
#[derive(Default)]
struct SmallPlanner {
    cost: Vec<f32>,
    next: Vec<f32>,
    from: Vec<u16>,
    /// States with a finite cost in `cost` / `next`.
    active: Vec<u16>,
    next_active: Vec<u16>,
    /// Per visited cell, its reached states as `(state, previous_state << 3 | choice)`.
    layers: Vec<(u16, u16)>,
    layer_start: Vec<usize>,
    plan: Vec<(usize, u16)>,
}

/// Costs of the four piece shapes anchored at each cell (`y * w + x`).
struct SmallCosts {
    leaf: [f32; SIDE * SIDE],
    wide: [f32; SIDE * SIDE],
    tall: [f32; SIDE * SIDE],
    square: [f32; SIDE * SIDE],
}

/// Returns the optimum and writes `(cell, choice)` per placed piece. `fixed`
/// marks cells that are already covered (no cost, no piece).
fn plan_small(
    costs: &SmallCosts,
    fixed: u64,
    w: usize,
    h: usize,
    s: &mut SmallPlanner,
    out: &mut Vec<(usize, u16)>,
) -> f32 {
    let cells = w * h;
    let bit = |cell: usize| -> usize {
        if cell < cells {
            ((fixed >> cell) & 1) as usize
        } else {
            0
        }
    };
    // A piece that costs at least as much as some tiling of its own cells by
    // smaller pieces is never needed: every layout using it has a cheaper
    // or equal twin. Such pieces open no states.
    let mut live = [0u8; SIDE * SIDE];
    for i in 0..cells {
        let (x, y) = (i % w, i / w);
        let right = x + 1 < w;
        let below = y + 1 < h;
        let wide = right && costs.wide[i] < costs.leaf[i] + costs.leaf[i + 1];
        let tall = below && costs.tall[i] < costs.leaf[i] + costs.leaf[i + w];
        let square = right && below && {
            let leaves =
                costs.leaf[i] + costs.leaf[i + 1] + costs.leaf[i + w] + costs.leaf[i + w + 1];
            let wides = costs.wide[i] + costs.wide[i + w];
            let talls = costs.tall[i] + costs.tall[i + 1];
            costs.square[i] < leaves.min(wides).min(talls)
        };
        live[i] = u8::from(wide) | u8::from(tall) << 1 | u8::from(square) << 2;
    }
    if s.cost.len() != SMALL_STATES {
        s.cost = vec![f32::INFINITY; SMALL_STATES];
        s.next = vec![f32::INFINITY; SMALL_STATES];
        s.from = vec![0; SMALL_STATES];
    }
    s.active.clear();
    s.layers.clear();
    s.layer_start.clear();
    let start = (fixed & ((1u64 << (w + 2)) - 1)) as usize;
    s.cost[start] = 0.0;
    s.active.push(start as u16);
    for i in 0..cells {
        let incoming = bit(i + w + 2) << (w + 1);
        let SmallPlanner {
            cost,
            next,
            from,
            active,
            next_active,
            layers,
            layer_start,
            ..
        } = &mut *s;
        next_active.clear();
        for &state in active.iter() {
            let state = state as usize;
            let here = cost[state];
            cost[state] = f32::INFINITY;
            let mut relax = |filled: usize, cost: f32, choice: u16| {
                let to = (filled >> 1) | incoming;
                if cost < next[to] {
                    if next[to] == f32::INFINITY {
                        next_active.push(to as u16);
                    }
                    next[to] = cost;
                    from[to] = (state as u16) << 3 | choice;
                }
            };
            if state & 1 != 0 {
                relax(state, here, SMALL_PREFILLED);
                continue;
            }
            relax(state, here + costs.leaf[i], SMALL_LEAF);
            let right = state & 2 == 0;
            let below = state & (1 << w) == 0;
            if right && live[i] & 1 != 0 {
                relax(state | 2, here + costs.wide[i], SMALL_WIDE);
            }
            if below && live[i] & 2 != 0 {
                relax(state | 1 << w, here + costs.tall[i], SMALL_TALL);
            }
            if right && below && live[i] & 4 != 0 && state & (1 << (w + 1)) == 0 {
                relax(
                    state | 2 | 1 << w | 1 << (w + 1),
                    here + costs.square[i],
                    SMALL_SQUARE,
                );
            }
        }
        layer_start.push(layers.len());
        layers.extend(next_active.iter().map(|&to| (to, from[to as usize])));
        std::mem::swap(cost, next);
        std::mem::swap(active, next_active);
    }
    out.clear();
    let total = s.cost[0];
    for &state in &s.active {
        s.cost[state as usize] = f32::INFINITY;
    }
    if !total.is_finite() {
        return total;
    }
    let mut state = 0u16;
    for i in (0..cells).rev() {
        let end = s.layer_start.get(i + 1).copied().unwrap_or(s.layers.len());
        let layer = &s.layers[s.layer_start[i]..end];
        let &(_, from) = layer
            .iter()
            .find(|&&(reached, _)| reached == state)
            .expect("the optimum's states were all reached");
        let choice = from & 7;
        if choice != SMALL_PREFILLED {
            out.push((i, choice));
        }
        state = from >> 3;
    }
    total
}

fn small_strategy(choice: u16) -> u8 {
    match choice {
        SMALL_LEAF => STRATEGY_DCT,
        SMALL_WIDE => STRATEGY_DCT8X16,
        SMALL_TALL => STRATEGY_DCT16X8,
        _ => STRATEGY_DCT16X16,
    }
}

/// Coefficient-model tile search: DCT8 leaves, pairs and 16x16 squares at
/// every cell, either around the incumbent's larger blocks or over them.
#[allow(clippy::too_many_arguments)]
fn search_tile_coefficient(
    scores: &CoefficientScores<'_, '_>,
    coder: &mut CoderScratch,
    cache: &mut LayoutDpScratch,
    map: &AcStrategyImage,
    width: usize,
    height: usize,
    accepted: &mut Vec<MergeUpgrade>,
) {
    cache.reset();
    let mut current = std::mem::take(&mut cache.current);
    current.clear();
    for y in 0..height {
        for x in 0..width {
            if map.is_first_block(scores.bx + x, scores.by + y) {
                current.push(Piece::new(
                    x,
                    y,
                    map.raw_strategy(scores.bx + x, scores.by + y),
                ));
            }
        }
    }
    if current
        .iter()
        .any(|p| crate::dc_group_data::is_gated_sub8_strategy(p.strategy))
    {
        cache.current = current;
        return;
    }
    let mut costs = SmallCosts {
        leaf: [f32::INFINITY; SIDE * SIDE],
        wide: [f32::INFINITY; SIDE * SIDE],
        tall: [f32::INFINITY; SIDE * SIDE],
        square: [f32::INFINITY; SIDE * SIDE],
    };
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            for (choice, slot) in [
                (SMALL_LEAF, &mut costs.leaf),
                (SMALL_WIDE, &mut costs.wide),
                (SMALL_TALL, &mut costs.tall),
                (SMALL_SQUARE, &mut costs.square),
            ] {
                let p = Piece::new(x, y, small_strategy(choice));
                if x + p.width() <= width
                    && y + p.height() <= height
                    && map.can_place_strategy(scores.bx + x, scores.by + y, p.strategy)
                {
                    slot[i] = scores.score(coder, cache, p);
                }
            }
        }
    }
    // Incumbent blocks outside the candidate shapes (32-family) either all
    // stay, fixed, or all give way.
    let mut fixed = 0u64;
    let mut fixed_cost = 0.0f32;
    for &p in &current {
        if COEFFICIENT_STRATEGIES.contains(&p.strategy) {
            continue;
        }
        for y in p.y as usize..p.y as usize + p.height() {
            for x in p.x as usize..p.x as usize + p.width() {
                fixed |= 1 << (y * width + x);
            }
        }
        fixed_cost += scores.score(coder, cache, p);
    }
    let mut layout = std::mem::take(&mut cache.layout);
    let mut small = std::mem::take(&mut cache.small);
    let mut plan_out = std::mem::take(&mut small.plan);
    let around = plan_small(&costs, fixed, width, height, &mut small, &mut plan_out) + fixed_cost;
    layout.clear();
    let mut keep_fixed = true;
    let mut winner = around;
    if fixed != 0 {
        let mut over_out = Vec::new();
        let over = plan_small(&costs, 0, width, height, &mut small, &mut over_out);
        if over < around {
            winner = over;
            keep_fixed = false;
            plan_out = over_out;
        }
    }
    if winner.is_finite() {
        for &(i, choice) in &plan_out {
            layout.push(Piece::new(i % width, i / width, small_strategy(choice)));
        }
        if keep_fixed {
            layout.extend(
                current
                    .iter()
                    .filter(|p| !COEFFICIENT_STRATEGIES.contains(&p.strategy)),
            );
        }
        layout.sort_by_key(|p| (p.y, p.x, p.strategy));
        debug_assert_eq!(pieces_mask(&layout), pieces_mask(&current));
        let mut was = [0u64; CACHE_SIZE.div_ceil(64)];
        let mut now = was;
        for p in &current {
            was[p.key() / 64] |= 1 << (p.key() % 64);
        }
        for p in &layout {
            now[p.key() / 64] |= 1 << (p.key() % 64);
        }
        let changed = |set: &[u64], p: &Piece| set[p.key() / 64] & 1 << (p.key() % 64) == 0;
        let old = current
            .iter()
            .filter(|p| changed(&now, p))
            .map(|&p| scores.score(coder, cache, p))
            .sum::<f32>();
        let new = layout
            .iter()
            .filter(|p| changed(&was, p))
            .map(|&p| scores.score(coder, cache, p))
            .sum::<f32>();
        if new < old {
            for &p in layout.iter().filter(|p| changed(&was, p)) {
                cache.best.push(MergeUpgrade {
                    bx: scores.bx + p.x as usize,
                    by: scores.by + p.y as usize,
                    strategy: p.strategy,
                    base: f32::NAN,
                });
            }
        }
    }
    accepted.extend_from_slice(&cache.best);
    small.plan = plan_out;
    cache.small = small;
    cache.current = current;
    cache.layout = layout;
}

fn find(
    rerank: &RerankContext<'_>,
    coder: &mut CoderScratch,
    map: &AcStrategyImage,
    band: (usize, usize),
    output: &mut AcStrategyBandScratch,
    rollbacks: &[(usize, usize, usize, usize)],
) {
    output.merge_upgrades.clear();
    output.dp_regions.clear();
    let mut cache = std::mem::take(&mut output.layout_dp);
    for by in (band.0.div_ceil(TILE) * TILE..band.1.min(map.ysize())).step_by(TILE) {
        for bx in (0..map.xsize()).step_by(TILE) {
            let (w, h) = ((map.xsize() - bx).min(TILE), (map.ysize() - by).min(TILE));
            if w * h < 2
                || !footprint_is_self_contained(map, bx, by, w, h)
                || rollbacks
                    .iter()
                    .any(|&(x, y, cx, cy)| bx < x + cx && x < bx + w && by < y + cy && y < by + h)
            {
                continue;
            }
            search_tile(
                &Scores { rerank, bx, by },
                coder,
                &mut cache,
                map,
                w,
                h,
                &mut output.merge_upgrades,
            );
            output.dp_regions.append(&mut cache.proposals);
        }
    }
    output.layout_dp = cache;
}

/// Installs every band's accepted layout and invalidates the cached
/// reconstruction costs under it (when the caller keeps any).
fn apply(map: &mut AcStrategyImage, bands: &[AcStrategyBandScratch], current_costs: &mut [f32]) {
    let stride = map.xsize();
    for band in bands {
        for up in &band.merge_upgrades {
            let w = AcStrategyImage::covered_blocks_x_of(up.strategy);
            let h = AcStrategyImage::covered_blocks_y_of(up.strategy);
            if !current_costs.is_empty() {
                for y in up.by..up.by + h {
                    current_costs[y * stride + up.bx..y * stride + up.bx + w].fill(f32::NAN);
                }
                current_costs[up.by * stride + up.bx] = up.base;
            }
            map.set_first(up.bx, up.by, up.strategy);
        }
    }
}

/// Re-tiles the map's closed 64x64 tiles. With `install` the winning
/// layouts replace the map's; otherwise the changed regions only remain in
/// each band's `dp_regions` as proposals.
pub(super) fn retile(
    rerank: &RerankContext<'_>,
    scratch: &mut CoderScratch,
    map: &mut AcStrategyImage,
    bands: &mut [AcStrategyBandScratch],
    current_costs: &mut [f32],
    install: bool,
) {
    let rollbacks: Vec<_> = bands
        .iter()
        .flat_map(|b| &b.fine_rollbacks)
        .map(|r| (r.bx, r.by, r.cov_x, r.cov_y))
        .collect();
    let read: &AcStrategyImage = map;
    rerank
        .params
        .ctx
        .thread_pool
        .steal_for_each_mut_with_threads(scratch, bands, rerank.num_threads, |i, out, coder| {
            let mut prices = rerank.params.price(coder);
            find(rerank, &mut prices, read, rerank.bands[i], out, &rollbacks);
        });
    if install {
        apply(map, bands, current_costs);
    }
}

fn find_coefficient(
    selection: &SelectionContext<'_>,
    coder: &mut CoderScratch,
    map: &AcStrategyImage,
    band: (usize, usize),
    output: &mut AcStrategyBandScratch,
) {
    output.merge_upgrades.clear();
    let mut cache = std::mem::take(&mut output.layout_dp);
    for by in (band.0.div_ceil(TILE) * TILE..band.1.min(map.ysize())).step_by(TILE) {
        for bx in (0..map.xsize()).step_by(TILE) {
            let (w, h) = ((map.xsize() - bx).min(TILE), (map.ysize() - by).min(TILE));
            if w * h < 2 || !footprint_is_self_contained(map, bx, by, w, h) {
                continue;
            }
            search_tile_coefficient(
                &CoefficientScores { selection, bx, by },
                coder,
                &mut cache,
                map,
                w,
                h,
                &mut output.merge_upgrades,
            );
        }
    }
    output.layout_dp = cache;
}

/// Coefficient-model re-tiling of every closed 64x64 tile after the fast
/// tiers' merge search: DCT8 leaves, pairs and 16x16 squares at any 8-pixel
/// origin, priced as the merge search priced its candidates.
pub(super) fn retile_coefficient(
    selection: &SelectionContext<'_>,
    scratch: &mut CoderScratch,
    map: &mut AcStrategyImage,
    bands: &[(usize, usize)],
    band_scratch: &mut [AcStrategyBandScratch],
    num_threads: usize,
    current_costs: &mut [f32],
) {
    let read: &AcStrategyImage = map;
    selection
        .params
        .ctx
        .thread_pool
        .steal_for_each_mut_with_threads(scratch, band_scratch, num_threads, |i, out, coder| {
            let mut prices = selection.params.price(coder);
            find_coefficient(selection, &mut prices, read, bands[i], out);
        });
    apply(map, band_scratch, current_costs);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn direct(x: usize, y: usize, strategy: u8, cost: f64) -> Candidate {
        Candidate::new(Piece::new(x, y, strategy), cost)
    }
    fn solve(candidates: &[Candidate], w: usize, h: usize) -> (f64, Vec<Piece>) {
        let mut s = PlannerScratch::default();
        let mut out = Vec::new();
        let cost = plan(candidates, w, h, &mut s, &mut out);
        if cost.is_finite() {
            assert!(lower_bound([0; SIDE], w, h, &s) <= cost);
        }
        (cost, out.into_iter().map(|k| candidates[k].piece).collect())
    }
    fn enumerate(candidates: &[Candidate], w: usize, h: usize, covered: u64) -> f64 {
        let full = if w * h == 64 {
            u64::MAX
        } else {
            (1u64 << (w * h)) - 1
        };
        if covered == full {
            return 0.0;
        }
        let first = (!covered & full).trailing_zeros() as usize;
        let (x, y) = (first % w, first / w);
        let mut best = f64::INFINITY;
        for &c in candidates {
            if c.x() != x || c.y() != y || !c.cost.is_finite() {
                continue;
            }
            let mut mask = 0;
            for cy in y..y + c.height {
                for cx in x..x + c.width {
                    mask |= 1 << (cy * w + cx);
                }
            }
            if mask & covered == 0 {
                best = best.min(c.cost + enumerate(candidates, w, h, covered | mask));
            }
        }
        best
    }
    fn assert_coverage(pieces: &[Piece], w: usize, h: usize) {
        let mut expected = 0u64;
        for y in 0..h {
            for x in 0..w {
                expected |= 1 << (y * 8 + x);
            }
        }
        assert_eq!(pieces_mask(pieces), expected);
    }
    #[test]
    fn coefficient_search_runs_where_the_reconstruction_search_does_not() {
        let gate = |speed: crate::Speed| {
            let effort = speed.effort();
            effort.layout_dp_coefficient && !effort.layout_retile
        };
        assert!(!gate(crate::Speed::UltraFast));
        assert!(gate(crate::Speed::Fast));
        assert!(!gate(crate::Speed::Medium));
        assert!(!gate(crate::Speed::MediumSlow));
        assert!(!gate(crate::Speed::Slow));
        assert!(!gate(crate::Speed::ExtraSlow));
    }
    #[test]
    fn search_runs_at_slow_and_under_the_extraslow_loop_only_once() {
        let gate = |speed: crate::Speed| {
            let effort = speed.effort();
            effort.layout_retile && !effort.layout_loop()
        };
        assert!(gate(crate::Speed::Medium));
        assert!(gate(crate::Speed::MediumSlow));
        assert!(gate(crate::Speed::Slow));
        assert_eq!(gate(crate::Speed::ExtraSlow), !cfg!(feature = "extraslow"));
        for speed in [
            crate::Speed::UltraFast,
            crate::Speed::Fastest,
            crate::Speed::Fast,
        ] {
            assert!(!gate(speed));
        }
    }
    #[test]
    fn every_family_uses_the_same_eight_pixel_grid() {
        let map = AcStrategyImage::new(24, 24);
        for w in 1..=8 {
            for h in 1..=8 {
                let mut pieces = Vec::new();
                for_each_piece(w, h, |p| pieces.push(p));
                for (i, &p) in pieces.iter().enumerate() {
                    assert!(p.x as usize + p.width() <= w && p.y as usize + p.height() <= h);
                    assert!(!pieces[..i].contains(&p));
                    assert!(map.can_place_strategy(8 + p.x as usize, 8 + p.y as usize, p.strategy));
                }
                assert!(pieces.len() + 2 <= MAX_CANDIDATES);
                if w == 8 && h == 8 {
                    assert_eq!(pieces.len(), 400);
                }
            }
        }
    }
    #[test]
    fn frontier_and_bounds_match_exhaustive_tilings() {
        for (w, h) in [(3, 2), (3, 3), (4, 3), (4, 4)] {
            for seed in 0..8 {
                let mut candidates = Vec::new();
                for_each_piece(w, h, |p| {
                    let value = ((p.key() * 17 + seed * 11) % 23) as f64 - 3.0;
                    candidates.push(Candidate::new(p, value / 8.0));
                });
                let expected = enumerate(&candidates, w, h, 0);
                let (actual, pieces) = solve(&candidates, w, h);
                assert_eq!(actual, expected, "{w}x{h} seed={seed}");
                assert_coverage(&pieces, w, h);
            }
        }
    }
    #[test]
    fn nonslicing_windmill_beats_the_guillotine_seed() {
        let mut c = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                c.push(direct(x, y, STRATEGY_DCT, 20.0));
            }
        }
        c.extend([
            direct(0, 0, STRATEGY_DCT8X16, 1.0),
            direct(2, 0, STRATEGY_DCT16X8, 1.0),
            direct(0, 1, STRATEGY_DCT16X8, 1.0),
            direct(1, 2, STRATEGY_DCT8X16, 1.0),
            direct(1, 1, STRATEGY_DCT, 1.0),
        ]);
        let mut s = PlannerScratch::default();
        let mut out = Vec::new();
        let cost = plan(&c, 3, 3, &mut s, &mut out);
        assert_eq!(cost, 5.0);
        assert!(s.regions[region_index(0, 0, 3, 3)].cost > cost);
        assert_coverage(&out.iter().map(|&k| c[k].piece).collect::<Vec<_>>(), 3, 3);
    }
    #[test]
    fn sixteen_plus_thirty_two_plus_sixteen_is_joint() {
        let mut c = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                c.push(direct(x, y, STRATEGY_DCT, 100.0));
            }
        }
        c.extend([
            direct(0, 0, STRATEGY_DCT32X32, 2.0),
            direct(4, 0, STRATEGY_DCT32X32, 2.0),
            direct(0, 4, STRATEGY_DCT32X16, 1.0),
            direct(2, 4, STRATEGY_DCT32X32, 1.0),
            direct(6, 4, STRATEGY_DCT32X16, 1.0),
        ]);
        let (cost, pieces) = solve(&c, 8, 8);
        assert_eq!(cost, 7.0);
        assert_eq!(pieces.len(), 5);
        assert_coverage(&pieces, 8, 8);
    }
    #[test]
    fn pairs_cross_old_sixteen_pixel_cell_boundaries() {
        let mut c = Vec::new();
        for y in 0..2 {
            for x in 0..4 {
                c.push(direct(
                    x,
                    y,
                    STRATEGY_DCT,
                    if x == 0 || x == 3 { 1.0 } else { 20.0 },
                ));
            }
        }
        c.extend([
            direct(1, 0, STRATEGY_DCT8X16, 1.0),
            direct(1, 1, STRATEGY_DCT8X16, 1.0),
        ]);
        let (cost, pieces) = solve(&c, 4, 2);
        assert_eq!(cost, 6.0);
        assert_eq!(pieces.len(), 6);
        assert!(pieces.contains(&Piece::new(1, 0, STRATEGY_DCT8X16)));
        assert_coverage(&pieces, 4, 2);
    }
    #[test]
    fn strips_and_pairs_cover_an_eight_pixel_sliver() {
        let mut c = Vec::new();
        for y in 0..8 {
            c.push(direct(
                0,
                y,
                STRATEGY_DCT,
                if y == 0 || y == 7 { 1.0 } else { 20.0 },
            ));
        }
        c.extend([
            direct(0, 1, STRATEGY_DCT32X8, 1.0),
            direct(0, 5, STRATEGY_DCT16X8, 1.0),
        ]);
        let (cost, pieces) = solve(&c, 1, 8);
        assert_eq!(cost, 4.0);
        assert_eq!(pieces.len(), 4);
        assert_coverage(&pieces, 1, 8);
    }
    #[test]
    fn incumbent_sixty_four_blocks_compete_only_as_placed() {
        let mut c = Vec::new();
        for_each_piece(8, 8, |p| {
            c.push(Candidate::new(
                p,
                if p.strategy == STRATEGY_DCT {
                    1.0
                } else {
                    1000.0
                },
            ))
        });
        assert!(c.iter().all(|c| c.width * c.height <= 16));
        let (cost, pieces) = solve(&c, 8, 8);
        assert_eq!(cost, 64.0);
        c.push(direct(0, 1, STRATEGY_DCT32X64, 1.0));
        let (cost, pieces64) = solve(&c, 8, 8);
        assert_eq!(cost, 33.0);
        assert!(pieces64.contains(&Piece::new(0, 1, STRATEGY_DCT32X64)));
        assert_coverage(&pieces, 8, 8);
        assert_coverage(&pieces64, 8, 8);
    }
    #[test]
    fn changed_regions_are_minimal_closed_and_disjoint() {
        let mut current = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                current.push(Piece::new(x, y, STRATEGY_DCT));
            }
        }
        // One 16x16 in the corner and one pair far away: two regions.
        let mut layout: Vec<Piece> = current
            .iter()
            .copied()
            .filter(|p| !((p.x < 2 && p.y < 2) || (p.x == 6 && p.y == 7) || (p.x == 7 && p.y == 7)))
            .collect();
        layout.push(Piece::new(0, 0, STRATEGY_DCT16X16));
        layout.push(Piece::new(6, 7, STRATEGY_DCT8X16));
        let mut regions = changed_regions(&current, &layout);
        regions.sort_unstable();
        assert_eq!(regions, vec![(0, 0, 2, 2), (6, 7, 2, 1)]);
        // A 32x32 replaced by pieces crossing its grid grows to one closed region.
        let mut big = Vec::new();
        big.push(Piece::new(0, 0, STRATEGY_DCT32X32));
        for y in 0..8 {
            for x in 0..8 {
                if x >= 4 || y >= 4 {
                    big.push(Piece::new(x, y, STRATEGY_DCT));
                }
            }
        }
        let mut shifted: Vec<Piece> = big
            .iter()
            .copied()
            .filter(|p| !(p.x < 5 && p.y < 4))
            .collect();
        shifted.push(Piece::new(1, 0, STRATEGY_DCT32X32));
        for y in 0..4 {
            shifted.push(Piece::new(0, y, STRATEGY_DCT));
        }
        assert_eq!(changed_regions(&big, &shifted), vec![(0, 0, 5, 4)]);
        assert!(changed_regions(&current, &current).is_empty());
    }
    #[test]
    fn failed_budgets_do_not_poison_the_exact_memo() {
        let mut c = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                c.push(direct(x, y, STRATEGY_DCT, 20.0));
            }
        }
        c.extend([
            direct(0, 0, STRATEGY_DCT8X16, 1.0),
            direct(2, 0, STRATEGY_DCT16X8, 1.0),
            direct(0, 1, STRATEGY_DCT16X8, 1.0),
            direct(1, 2, STRATEGY_DCT8X16, 1.0),
            direct(1, 1, STRATEGY_DCT, 1.0),
        ]);
        let mut scratch = PlannerScratch::default();
        let mut choices = Vec::new();
        assert_eq!(plan(&c, 3, 3, &mut scratch, &mut choices), 5.0);
        scratch.memo.clear();
        for budget in [4.0, 4.5] {
            assert!(visit([0; SIDE], 3, 3, budget, &c, &mut scratch).is_none());
            assert!(matches!(scratch.memo[&0],Memo::AtLeast(bound) if bound >= budget));
        }
        assert_eq!(
            visit([0; SIDE], 3, 3, 6.0, &c, &mut scratch).unwrap().cost,
            5.0
        );
        assert!(matches!(scratch.memo[&0],Memo::Solved(entry) if entry.cost == 5.0));
    }

    #[test]
    fn planner_reuse_and_infeasible_candidates_do_not_leak() {
        let mut s = PlannerScratch::default();
        let mut out = Vec::new();
        for flip in [false, true, false] {
            let c = [
                direct(0, 0, STRATEGY_DCT, 1.0),
                direct(1, 0, STRATEGY_DCT, 1.0),
                direct(0, 0, STRATEGY_DCT8X16, if flip { 0.5 } else { 3.0 }),
            ];
            assert_eq!(
                plan(&c, 2, 1, &mut s, &mut out),
                if flip { 0.5 } else { 2.0 }
            );
            assert_eq!(out.len(), if flip { 1 } else { 2 });
        }
        let c = [
            direct(0, 0, STRATEGY_DCT, f64::NAN),
            direct(0, 0, STRATEGY_DCT, f64::INFINITY),
        ];
        assert!(plan(&c, 1, 1, &mut s, &mut out).is_infinite());
        assert!(out.is_empty());
    }
    #[test]
    fn sixty_four_pixel_tiles_have_one_owner_for_every_thread_partition() {
        for height in [8, 11, 16, 19, 32, 35] {
            let expected: Vec<_> = (0..height).step_by(TILE).collect();
            for threads in 1..=12 {
                let mut bands = Vec::new();
                fill_selection_bands(&mut bands, height, threads);
                let actual: Vec<_> = bands
                    .iter()
                    .flat_map(|&(y0, y1)| (y0.div_ceil(TILE) * TILE..y1).step_by(TILE))
                    .collect();
                assert_eq!(actual, expected, "height {height}, threads {threads}");
            }
        }
    }

    fn small_enumerate(costs: &SmallCosts, fixed: u64, w: usize, h: usize, covered: u64) -> f32 {
        let full = (1u64 << (w * h)) - 1;
        if covered == full {
            return 0.0;
        }
        let first = (!covered & full).trailing_zeros() as usize;
        let (x, y) = (first % w, first / w);
        let mut best = f32::INFINITY;
        if fixed & 1 << first != 0 {
            return small_enumerate(costs, fixed, w, h, covered | 1 << first);
        }
        let mut place = |cells: &[usize], cost: f32| {
            let mask = cells.iter().fold(0u64, |m, &c| m | 1 << c);
            if mask & (covered | fixed) == 0 {
                best = best.min(cost + small_enumerate(costs, fixed, w, h, covered | mask));
            }
        };
        place(&[first], costs.leaf[first]);
        if x + 1 < w {
            place(&[first, first + 1], costs.wide[first]);
        }
        if y + 1 < h {
            place(&[first, first + w], costs.tall[first]);
        }
        if x + 1 < w && y + 1 < h {
            place(
                &[first, first + 1, first + w, first + w + 1],
                costs.square[first],
            );
        }
        best
    }
    #[test]
    fn small_planner_matches_exhaustive_tilings_around_fixed_cells() {
        let mut s = SmallPlanner::default();
        let mut out = Vec::new();
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for (w, h) in [(1, 1), (1, 5), (3, 2), (4, 4), (5, 3), (4, 5)] {
            for trial in 0..6 {
                let mut costs = SmallCosts {
                    leaf: [f32::INFINITY; SIDE * SIDE],
                    wide: [f32::INFINITY; SIDE * SIDE],
                    tall: [f32::INFINITY; SIDE * SIDE],
                    square: [f32::INFINITY; SIDE * SIDE],
                };
                for i in 0..w * h {
                    costs.leaf[i] = 1.0 + rnd();
                    costs.wide[i] = 1.0 + 2.5 * rnd();
                    costs.tall[i] = 1.0 + 2.5 * rnd();
                    costs.square[i] = 2.0 + 5.0 * rnd();
                }
                let fixed = if trial % 2 == 0 {
                    0
                } else {
                    (0..w * h)
                        .filter(|_| rnd() < 0.2)
                        .fold(0u64, |m, i| m | 1 << i)
                };
                let expected = small_enumerate(&costs, fixed, w, h, 0);
                let actual = plan_small(&costs, fixed, w, h, &mut s, &mut out);
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected,
                    "{w}x{h}: {actual} vs {expected}"
                );
                let mut covered = fixed;
                let mut total = 0.0;
                for &(i, choice) in &out {
                    let p = Piece::new(i % w, i / w, small_strategy(choice));
                    total += match choice {
                        SMALL_LEAF => costs.leaf[i],
                        SMALL_WIDE => costs.wide[i],
                        SMALL_TALL => costs.tall[i],
                        _ => costs.square[i],
                    };
                    for y in p.y as usize..p.y as usize + p.height() {
                        for x in p.x as usize..p.x as usize + p.width() {
                            let bit = 1u64 << (y * w + x);
                            assert_eq!(covered & bit, 0, "overlap at {x},{y}");
                            covered |= bit;
                        }
                    }
                }
                assert_eq!(covered, (1u64 << (w * h)) - 1);
                assert!((total - expected).abs() <= 1e-4 * expected);
            }
        }
    }
    #[test]
    fn small_planner_takes_the_eight_by_eight_optimum_without_a_state_cap() {
        // Uniform costs where every 2x2 square beats its parts: the optimum is
        // sixteen aligned squares, a nonslicing-free check of the full state space.
        let mut s = SmallPlanner::default();
        let mut out = Vec::new();
        let costs = SmallCosts {
            leaf: [4.0; SIDE * SIDE],
            wide: [7.5; SIDE * SIDE],
            tall: [7.5; SIDE * SIDE],
            square: [13.0; SIDE * SIDE],
        };
        assert_eq!(plan_small(&costs, 0, 8, 8, &mut s, &mut out), 16.0 * 13.0);
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|&(_, c)| c == SMALL_SQUARE));
        // With the top-left 4x4 fixed, the rest still tiles with squares.
        let fixed = (0..4).fold(0u64, |m, y| m | 0b1111 << (y * 8));
        assert_eq!(
            plan_small(&costs, fixed, 8, 8, &mut s, &mut out),
            12.0 * 13.0
        );
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn real_reconstruction_can_promote_a_shifted_square() {
        let maps = ImageSB::new_fill(1, 1, 0);
        let quants = ImageB::new_fill(8, 8, 6);
        let bands = [(0, 8)];
        let mut winners = 0;
        for frequency in [1, 2, 3, 5] {
            let mut opsin = Image3F::new(64, 64);
            for y in 0..64 {
                for x in 0..64 {
                    let ink = if (16..48).contains(&x) && y >= 32 {
                        0.16 * ((x as f32 - 15.5) * std::f32::consts::PI / 32.0).cos()
                            * ((y as f32 - 31.5) * std::f32::consts::PI * frequency as f32 / 32.0)
                                .cos()
                    } else {
                        0.0
                    };
                    opsin.plane_row_mut(0, y)[x] = 0.02;
                    opsin.plane_row_mut(1, y)[x] = 0.3 + ink;
                    opsin.plane_row_mut(2, y)[x] = 0.3;
                }
            }
            for scale in [0.02, 0.05, 0.1, 0.25] {
                let ctx = EncodingContext::new(
                    crate::Speed::Slow,
                    crate::Arbiter::Ssimulacra2,
                    crate::xyb::XybMatrix::SPEC,
                    2.0,
                    1,
                );
                let params = AcStrategyParams {
                    ctx: &ctx,
                    opsin: &opsin,
                    dc_group_px: 0,
                    dc_group_py: 0,
                    distance: 2.0,
                    scale,
                    qm_mult_x: 1.0,
                    ytox_map: &maps,
                    ytob_map: &maps,
                    prices: None,
                };
                let rerank = RerankContext {
                    params,
                    quant_field: &quants,
                    bands: &bands,
                    num_threads: 1,
                    leaf_first: false,
                    leaves: &[],
                };
                let mut map = AcStrategyImage::new(8, 8);
                for y in [0, 4] {
                    for x in [0, 4] {
                        map.set_first(x, y, STRATEGY_DCT32X32);
                    }
                }
                let mut coder = CoderScratch::default();
                let mut cache = LayoutDpScratch::default();
                let mut accepted = Vec::new();
                search_tile(
                    &Scores {
                        rerank: &rerank,
                        bx: 0,
                        by: 0,
                    },
                    &mut coder,
                    &mut cache,
                    &map,
                    8,
                    8,
                    &mut accepted,
                );
                assert!(cache.candidates.len() <= MAX_CANDIDATES);
                for up in &accepted {
                    map.set_first(up.bx, up.by, up.strategy);
                }
                let pieces: Vec<_> = map
                    .iter_first_blocks()
                    .map(|(x, y, s)| Piece::new(x, y, s))
                    .collect();
                assert_eq!(pieces_mask(&pieces), u64::MAX);
                assert!(pieces.iter().all(|p| map.can_place_strategy(
                    p.x as usize,
                    p.y as usize,
                    p.strategy
                )));
                winners += usize::from(pieces.iter().any(|p| p.shifted32()));
            }
        }
        assert!(
            winners > 0,
            "real forward/inverse scoring never selected a shifted 32-family transform"
        );
    }
}

#[cfg(test)]
mod protection_tests {
    use super::*;
    #[test]
    fn coefficient_model_retiles_around_or_over_the_merge_search_blocks() {
        let ctx = EncodingContext::new(
            crate::Speed::Fast,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            2.0,
            1,
        );
        let maps = ImageSB::new_fill(1, 1, 0);
        let quant = ImageB::new_fill(8, 8, 6);
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let ink = if (24..40).contains(&x) {
                    0.12 * ((y as f32) * 0.4).sin()
                } else {
                    0.0
                };
                opsin.plane_row_mut(0, y)[x] = 0.02;
                opsin.plane_row_mut(1, y)[x] = 0.3 + ink;
                opsin.plane_row_mut(2, y)[x] = 0.3;
            }
        }
        let params = AcStrategyParams {
            ctx: &ctx,
            opsin: &opsin,
            dc_group_px: 0,
            dc_group_py: 0,
            distance: 2.0,
            scale: 0.1,
            qm_mult_x: 1.0,
            ytox_map: &maps,
            ytob_map: &maps,
            prices: None,
        };
        let selection = SelectionContext {
            params,
            quant_field: &quant,
            meta_r: META_R,
            scope: SearchScope::SQUARES,
        };
        let mut coder = CoderScratch::default();
        let mut output = AcStrategyBandScratch::default();
        let mut map = AcStrategyImage::new(8, 8);
        map.set_first(0, 0, STRATEGY_DCT32X32);
        map.set_first(4, 4, STRATEGY_DCT32X32);
        map.set_first(4, 0, STRATEGY_DCT16X16);
        find_coefficient(&selection, &mut coder, &map, (0, 8), &mut output);
        for up in &output.merge_upgrades {
            assert!(COEFFICIENT_STRATEGIES.contains(&up.strategy));
            assert!(up.base.is_nan());
            map.set_first(up.bx, up.by, up.strategy);
        }
        let pieces: Vec<_> = map
            .iter_first_blocks()
            .map(|(x, y, s)| Piece::new(x, y, s))
            .collect();
        assert_eq!(pieces_mask(&pieces), u64::MAX);
        let kept32 = pieces
            .iter()
            .filter(|p| p.strategy == STRATEGY_DCT32X32)
            .count();
        assert!(
            kept32 == 0 || kept32 == 2,
            "32x32 blocks stay or go together"
        );
        // A 128-family footprint stays untouched.
        let mut large = AcStrategyImage::new(16, 16);
        large.set_first(0, 0, crate::dc_group_data::STRATEGY_DCT128X128);
        find_coefficient(&selection, &mut coder, &large, (0, 16), &mut output);
        assert!(output.merge_upgrades.is_empty());
    }

    #[test]
    fn protected_and_nonclosed_regions_do_not_get_repartitioned() {
        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            2.0,
            1,
        );
        let opsin = Image3F::new(128, 128);
        let maps = ImageSB::new_fill(2, 2, 0);
        let quant = ImageB::new_fill(16, 16, 6);
        let bands = [(0, 16)];
        let params = AcStrategyParams {
            ctx: &ctx,
            opsin: &opsin,
            dc_group_px: 0,
            dc_group_py: 0,
            distance: 2.0,
            scale: 0.1,
            qm_mult_x: 1.0,
            ytox_map: &maps,
            ytob_map: &maps,
            prices: None,
        };
        let rerank = RerankContext {
            params,
            quant_field: &quant,
            bands: &bands,
            num_threads: 1,
            leaf_first: false,
            leaves: &[],
        };
        let mut coder = CoderScratch::default();
        let mut output = AcStrategyBandScratch::default();
        let mut large = AcStrategyImage::new(16, 16);
        large.set_first(0, 0, crate::dc_group_data::STRATEGY_DCT128X128);
        find(&rerank, &mut coder, &large, (0, 16), &mut output, &[]);
        assert!(output.merge_upgrades.is_empty());
        assert!(
            output.layout_dp.reconstruction.is_empty(),
            "128-family footprints must not enter this search"
        );
        let mut small = AcStrategyImage::new(8, 8);
        find(
            &rerank,
            &mut coder,
            &small,
            (0, 8),
            &mut output,
            &[(1, 1, 2, 2)],
        );
        assert!(output.merge_upgrades.is_empty());
        assert!(
            output.layout_dp.reconstruction.is_empty(),
            "fine-mosaic rollback regions must remain intact"
        );
        small.set_first(0, 0, crate::dc_group_data::STRATEGY_IDENTITY);
        find(&rerank, &mut coder, &small, (0, 8), &mut output, &[]);
        assert!(
            output.merge_upgrades.is_empty(),
            "legacy fine credits cannot be removed"
        );
        let thin = AcStrategyImage::new(1, 8);
        find(&rerank, &mut coder, &thin, (0, 8), &mut output, &[]);
        assert!(
            !output.layout_dp.candidates.is_empty(),
            "8-pixel slivers must enter the DP"
        );
        assert!(output.layout_dp.candidates.iter().all(|c| c.width == 1));
    }
}
