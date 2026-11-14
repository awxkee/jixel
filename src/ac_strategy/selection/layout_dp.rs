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
//! placed. Retained fine transforms compete with DCT8 at each leaf. Frontier
//! memos cover slicing and nonslicing tilings: compact occupancy states for
//! small pieces, and a sparse skyline memo when larger pieces compete.
//! Guillotine plans provide feasible upper bounds; independent row/column
//! relaxations provide admissible lower bounds. Two cuts are not admissible:
//! a larger piece is dropped once its luma-and-rate bound exceeds
//! `BOUND_SLACK` of the best tiling of its rectangle by smaller pieces, and a
//! tile that exhausts `VISIT_BUDGET` settles its remaining states for their
//! guillotine plans. Both are measured choices, not approximations to remove:
//! slack 1.0 loses quality and an unbounded budget changes almost nothing.

use super::*;
use crate::util::heap_array;

mod compact;

const TILE: usize = 8;
const SIDE: usize = TILE;
const CACHE_SIZE: usize = crate::dc_group_data::NUM_STRATEGIES * TILE * TILE;
const AXIS_INTERVALS: usize = SIDE * (SIDE + 1) / 2;
const REGION_COUNT: usize = AXIS_INTERVALS * AXIS_INTERVALS;
// Each end column has one entry for each possible start, including itself.
const LINE_INTERVALS: usize = (SIDE + 1) * (SIDE + 2) / 2;
const LINE_COUNT: usize = SIDE * LINE_INTERVALS;
/// The 400 grid placements plus at most two incumbent 64-family blocks.
const MAX_CANDIDATES: usize = 402;
/// Tile-grid offsets, in blocks, of the re-tiling passes that follow the
/// aligned one: half a tile diagonally (`layout_crossing`), then along each
/// axis (`layout_crossing_axes`).
static CROSSING_GRIDS: [(usize, usize); 3] = [(4, 4), (4, 0), (0, 4)];

fn crossing_grids(effort: &crate::effort::Effort) -> &'static [(usize, usize)] {
    match (effort.layout_crossing, effort.layout_crossing_axes) {
        (false, _) => &[],
        (true, false) => &CROSSING_GRIDS[..1],
        (true, true) => &CROSSING_GRIDS,
    }
}
/// Frontier states a tile may expand before the remaining states settle for
/// their guillotine plan.
const VISIT_BUDGET: usize = 5_000;
/// A piece is scored in full only while its luma-and-rate lower bound stays
/// under this fraction of the best tiling of its rectangle by smaller pieces.
const BOUND_SLACK: f32 = 0.9;
static STRATEGIES: [u8; 9] = [
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
static COEFFICIENT_STRATEGIES: [u8; 4] = [
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
    fn mask(self, stride: usize) -> u64 {
        let row = (1u64 << self.width()) - 1;
        let offset = self.y as usize * stride + self.x as usize;
        let mut mask = 0;
        for y in 0..self.height() {
            mask |= row << (offset + y * stride);
        }
        mask
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
    width: u8,
    height: u8,
    cost: f64,
}
impl Candidate {
    fn new(piece: Piece, cost: f64) -> Self {
        Self {
            piece,
            width: piece.width() as u8,
            height: piece.height() as u8,
            cost,
        }
    }
    fn width(self) -> usize {
        usize::from(self.width)
    }
    fn height(self) -> usize {
        usize::from(self.height)
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
    Tile(u16),
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

/// Open-addressing table keyed by a `u32`: Fibonacci hashing, linear
/// probing, and a generation stamp in place of clearing.
struct OpenTable<V: Copy + Default> {
    keys: Vec<u32>,
    stamps: Vec<u32>,
    values: Vec<V>,
    generation: u32,
    len: usize,
    shift: u32,
}

impl<V: Copy + Default> Default for OpenTable<V> {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            stamps: Vec::new(),
            values: Vec::new(),
            generation: 1,
            len: 0,
            shift: 32,
        }
    }
}

impl<V: Copy + Default> OpenTable<V> {
    fn grow(&mut self) {
        let capacity = (self.keys.len() * 2).max(128);
        // Rehash directly from the old buffers, copying only this generation.
        let old_keys = std::mem::replace(&mut self.keys, vec![0; capacity]);
        let old_stamps = std::mem::replace(&mut self.stamps, vec![0; capacity]);
        let old_values = std::mem::replace(&mut self.values, vec![V::default(); capacity]);
        self.shift = 32 - capacity.trailing_zeros();
        for ((key, stamp), value) in old_keys.into_iter().zip(old_stamps).zip(old_values) {
            if stamp != self.generation {
                continue;
            }
            let (slot, _) = self.slot(key);
            self.keys[slot] = key;
            self.stamps[slot] = stamp;
            self.values[slot] = value;
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamps.fill(0);
            self.generation = 1;
        }
    }

    /// The slot holding `key`, or the empty slot it belongs in.
    #[inline]
    fn slot(&self, key: u32) -> (usize, bool) {
        let mask = self.keys.len() - 1;
        let mut slot = (key.wrapping_mul(0x9E37_79B1) >> self.shift) as usize;
        loop {
            if self.stamps[slot] != self.generation {
                return (slot, false);
            }
            if self.keys[slot] == key {
                return (slot, true);
            }
            slot = (slot + 1) & mask;
        }
    }

    #[inline]
    fn get(&self, key: u32) -> Option<V> {
        if self.len == 0 {
            return None;
        }
        let (slot, found) = self.slot(key);
        found.then(|| self.values[slot])
    }

    fn insert(&mut self, key: u32, value: V) {
        if self.keys.is_empty() {
            self.grow();
        }
        let (mut slot, found) = self.slot(key);
        if !found {
            if (self.len + 1) * 2 > self.keys.len() {
                self.grow();
                slot = self.slot(key).0;
            }
            self.keys[slot] = key;
            self.stamps[slot] = self.generation;
            self.len += 1;
        }
        self.values[slot] = value;
    }
}

impl Default for Memo {
    fn default() -> Self {
        Self::AtLeast(0.0)
    }
}

#[derive(Default)]
struct PlannerScratch {
    memo: OpenTable<Memo>,
    // Cut scans read the costs without loading the backpointers.
    region_costs: Vec<f64>,
    region_choices: Vec<u16>,
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
    // Group the nonempty intervals of each axis by length. The longer
    // intervals have fewer valid origins and need no padding slots.
    let axis = |origin, len| origin + (len - 1) * (2 * SIDE + 2 - len) / 2;
    axis(y, h) * AXIS_INTERVALS + axis(x, w)
}
fn line_index(y: usize, start: usize, end: usize) -> usize {
    y * LINE_INTERVALS + end * (end + 1) / 2 + start
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
    profile[c.x()..c.x() + c.width()].fill((c.y() + c.height()) as u8);
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

// A zero backpointer means no choice. Tile IDs are stored plus one, followed
// by the two ranges for vertical and horizontal cut positions.
const REGION_LAST_TILE: u16 = MAX_CANDIDATES as u16;
const REGION_VERTICAL_BASE: u16 = REGION_LAST_TILE;
const REGION_HORIZONTAL_BASE: u16 = REGION_VERTICAL_BASE + SIDE as u16;

fn prepare_regions(
    candidates: &[Candidate],
    width: usize,
    height: usize,
    max_side: usize,
    short_side: usize,
    s: &mut PlannerScratch,
) {
    debug_assert!(candidates.len() <= MAX_CANDIDATES);
    s.region_costs.clear();
    s.region_costs.resize(REGION_COUNT, f64::INFINITY);
    s.region_choices.clear();
    s.region_choices.resize(REGION_COUNT, 0);
    for (i, &c) in candidates.iter().enumerate() {
        if !c.cost.is_finite() {
            continue;
        }
        let index = region_index(c.x(), c.y(), c.width(), c.height());
        if c.cost < s.region_costs[index] {
            s.region_costs[index] = c.cost;
            s.region_choices[index] = 1 + i as u16;
        }
    }
    assert!(width <= SIDE && height <= SIDE);
    let regions = &mut s.region_costs[..REGION_COUNT];
    let choices = &mut s.region_choices[..REGION_COUNT];
    // SAFETY: every region read or written below lies inside the
    // width x height <= SIDE x SIDE tile, so region_index < REGION_COUNT.
    let cost_of =
        |regions: &[f64], x, y, w, h| unsafe { *regions.get_unchecked(region_index(x, y, w, h)) };
    for h in 1..=height.min(max_side) {
        let max_width = if h <= short_side {
            max_side
        } else {
            short_side
        };
        for w in 1..=width.min(max_width) {
            for y in 0..=height - h {
                for x in 0..=width - w {
                    let slot = region_index(x, y, w, h);
                    // SAFETY: as above.
                    let mut best = unsafe { *regions.get_unchecked(slot) };
                    let mut choice = unsafe { *choices.get_unchecked(slot) };
                    for cut in 1..w {
                        let cost = cost_of(regions, x, y, cut, h)
                            + cost_of(regions, x + cut, y, w - cut, h);
                        if cost < best {
                            best = cost;
                            choice = REGION_VERTICAL_BASE + cut as u16;
                        }
                    }
                    for cut in 1..h {
                        let cost = cost_of(regions, x, y, w, cut)
                            + cost_of(regions, x, y + cut, w, h - cut);
                        if cost < best {
                            best = cost;
                            choice = REGION_HORIZONTAL_BASE + cut as u16;
                        }
                    }
                    // SAFETY: as above.
                    unsafe {
                        *regions.get_unchecked_mut(slot) = best;
                        *choices.get_unchecked_mut(slot) = choice;
                    }
                }
            }
        }
    }
}

fn entering_column(objective: &[f64], column_ids: &[u16]) -> Option<usize> {
    const LIMIT: f64 = -1e-10;
    let (quads, tail) = objective.as_chunks::<4>();
    let mut minima = [LIMIT; 4];
    for quad in quads {
        for (minimum, &value) in minima.iter_mut().zip(quad) {
            *minimum = minimum.min(value);
        }
    }
    let mut reduced = minima.into_iter().fold(LIMIT, f64::min);
    for &value in tail {
        reduced = reduced.min(value);
    }
    // Only costs strictly below LIMIT can enter. Break ties by the original
    // column order even after pivots have exchanged the stored columns.
    if reduced >= LIMIT {
        return None;
    }
    let mut best = None;
    let mut best_id = u16::MAX;
    for (i, (&value, &id)) in objective.iter().zip(column_ids).enumerate() {
        if value == reduced && id < best_id {
            best = Some(i);
            best_id = id;
        }
    }
    best
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
        if c.width() == 1 && c.height() == 1 && c.cost.is_finite() {
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
    // Candidate IDs seed the tableau, then the same slots hold simplex
    // column IDs. The bounded tile sizes keep both sets of IDs in u16.
    let mut column_ids = [0u16; MAX_CANDIDATES];
    let mut benefits = [0.0f64; MAX_CANDIDATES];
    let mut n = 0;
    for (k, &c) in candidates.iter().enumerate() {
        if !c.cost.is_finite() || c.width() * c.height() == 1 {
            continue;
        }
        let mut baseline = 0.0;
        let mut baseline_upper = 0.0;
        for y in c.y()..c.y() + c.height() {
            for x in c.x()..c.x() + c.width() {
                baseline += leaves[y * width + x];
                baseline_upper = -add_lower(-baseline_upper, -leaves[y * width + x]);
            }
        }
        if c.cost >= baseline_upper {
            s.dominated[k] = true;
        } else {
            column_ids[n] = k as u16;
            benefits[n] = baseline - c.cost;
            n += 1;
        }
    }
    // Basic columns are identity vectors. Store only the nonbasic columns
    // and exchange entering/leaving column IDs after each pivot.
    let cols = n + 1;
    let rows = cells + 1;
    let objective = cells * cols;
    s.tableau.clear();
    s.tableau.resize(rows * cols, 0.0);
    let mut basis: [u16; SIDE * SIDE] = std::array::from_fn(|row| (n + row) as u16);
    let nonbasic = &mut column_ids[..n];
    for (v, (id, &benefit)) in nonbasic.iter_mut().zip(&benefits).enumerate() {
        let c = candidates[usize::from(*id)];
        *id = v as u16;
        for y in c.y()..c.y() + c.height() {
            for x in c.x()..c.x() + c.width() {
                s.tableau[(y * width + x) * cols + v] = 1.0;
            }
        }
        s.tableau[objective + v] = -benefit;
    }
    for row in 0..cells {
        s.tableau[row * cols + cols - 1] = 1.0;
    }
    for _ in 0..PIVOTS {
        let Some(enter) = entering_column(&s.tableau[objective..objective + n], nonbasic) else {
            break;
        };
        let mut leave = None;
        let mut ratio = f64::INFINITY;
        for (row, values) in s.tableau[..objective].chunks_exact(cols).enumerate() {
            let a = values[enter];
            if a <= 1e-12 {
                continue;
            }
            let r = values[cols - 1] / a;
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
        // Row slices: every other row reads only the pivot row, and the
        // elementwise updates vectorize without per-index checks.
        let (above, rest) = s.tableau[..rows * cols].split_at_mut(leave * cols);
        let (pivot_row, below) = rest.split_at_mut(cols);
        let pivot = pivot_row[enter];
        // Unit pivots leave every entry of the pivot row unchanged.
        if pivot != 1.0 {
            for v in pivot_row.iter_mut() {
                *v /= pivot;
            }
        }
        // The entering column takes the place of the leaving basis column.
        // Before this pivot that column had 1 in the pivot row and 0 elsewhere.
        pivot_row[enter] = 1.0 / pivot;
        let pivot_row = &*pivot_row;
        let mut changed = above
            .chunks_exact_mut(cols)
            .chain(below.chunks_exact_mut(cols))
            .filter_map(|row| {
                let factor = row[enter];
                row[enter] = 0.0;
                (factor != 0.0).then_some((row, factor))
            });
        while let Some((row, factor)) = changed.next() {
            if let Some((next, next_factor)) = changed.next() {
                // Both rows read the same immutable pivot. Loading it once
                // preserves each element's multiply and subtraction.
                for ((v, n), &p) in row.iter_mut().zip(next.iter_mut()).zip(pivot_row) {
                    *v -= factor * p;
                    *n -= next_factor * p;
                }
            } else {
                for (v, &p) in row.iter_mut().zip(pivot_row) {
                    *v -= factor * p;
                }
            }
        }
        std::mem::swap(&mut basis[leave], &mut nonbasic[enter]);
    }
    let mut potentials = [0.0; SIDE * SIDE];
    for i in 0..cells {
        potentials[i] = leaves[i] - 0.0;
    }
    for (i, &column) in nonbasic.iter().enumerate() {
        let column = usize::from(column);
        if column >= n {
            let cell = column - n;
            potentials[cell] = leaves[cell] - s.tableau[objective + i].max(0.0);
        }
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
        for y in c.y()..c.y() + c.height() {
            for x in c.x()..c.x() + c.width() {
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
    // Every transform side is a power of two. Store those lengths in
    // ascending order, omitting the other lengths' permanently empty edges.
    const EDGE_LENGTHS: usize = SIDE.ilog2() as usize + 1;
    let mut vertical = [[[f64::INFINITY; EDGE_LENGTHS]; SIDE]; SIDE];
    let mut horizontal = vertical;
    for &c in candidates {
        if !c.cost.is_finite() {
            continue;
        }
        // Widths and heights are powers of two, so f32 candidate costs can be
        // divided exactly after promotion to f64.
        let width_class = c.width().trailing_zeros() as usize;
        let height_class = c.height().trailing_zeros() as usize;
        let vertical_cost = c.cost / c.width() as f64;
        let horizontal_cost = c.cost / c.height() as f64;
        for column in &mut vertical[c.x()..c.x() + c.width()] {
            let edge = &mut column[c.y()][height_class];
            *edge = edge.min(vertical_cost);
        }
        for row in &mut horizontal[c.y()..c.y() + c.height()] {
            let edge = &mut row[c.x()][width_class];
            *edge = edge.min(horizontal_cost);
        }
    }
    for (x, edges) in vertical.iter().enumerate().take(width) {
        s.columns[x].fill(f64::INFINITY);
        s.columns[x][height] = 0.0;
        for y in (0..height).rev() {
            for (power, &edge) in edges[y].iter().enumerate() {
                let end = y + (1 << power);
                if end > height {
                    break;
                }
                s.columns[x][y] = s.columns[x][y].min(add_lower(edge, s.columns[x][end]));
            }
        }
    }
    s.rows.clear();
    s.rows.resize(LINE_COUNT, f64::INFINITY);
    assert!(width <= SIDE);
    let rows = &mut s.rows[..LINE_COUNT];
    for (y, edges) in horizontal.iter().enumerate().take(height) {
        for end in 0..=width {
            let base = line_index(y, 0, end);
            let intervals = &mut rows[base..=base + end];
            intervals[end] = 0.0;
            for x in (0..end).rev() {
                let mut best = f64::INFINITY;
                for (power, &edge) in edges[x].iter().enumerate() {
                    let next = x + (1 << power);
                    if next > end {
                        break;
                    }
                    best = best.min(add_lower(edge, intervals[next]));
                }
                intervals[x] = best;
            }
        }
    }
}

// Store the first run's triangular-table offset in the low six bits and
// the remaining open cells in the high bits.
const ROW_RUNS: [u16; 1 << SIDE] = {
    let mut runs = [0; 1 << SIDE];
    let mut mask = 1usize;
    while mask < 1 << SIDE {
        let start = mask.trailing_zeros() as usize;
        let end = start + (mask >> start).trailing_ones() as usize;
        let offset = end * (end + 1) / 2 + start;
        runs[mask] = ((mask & (usize::MAX << end)) << 6 | offset) as u16;
        mask += 1;
    }
    runs
};

fn lower_bound(profile: [u8; SIDE], width: usize, height: usize, s: &PlannerScratch) -> f64 {
    let mut columns = 0.0;
    for (x, &y) in profile.iter().enumerate().take(width) {
        columns = add_lower(columns, s.columns[x][y as usize]);
    }
    let lanes = u64::from_le_bytes(profile);
    // Bit 7 of each lane guards the subtraction against cross-lane borrows.
    // Bits stay set for columns where y >= profile[x].
    let guard: u64 = 0x8080_8080_8080_8080;
    let mut levels = guard;
    let mask = guard >> ((SIDE - width) * 8);
    let mut rows = 0.0;
    for row in s.rows.chunks_exact(LINE_INTERVALS).take(height) {
        let open = levels.wrapping_sub(lanes) & mask;
        // Gather each lane's flag into its column bit in the low byte.
        let mut open = ((open >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize;
        while open != 0 {
            let entry = ROW_RUNS[open];
            rows = add_lower(rows, row[usize::from(entry & 63)]);
            open = usize::from(entry >> 6);
        }
        levels += 0x0101_0101_0101_0101;
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
            cost += s.region_costs[region_index(start, y, x - start, height - y)];
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
    match s.memo.get(key) {
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
        if !c.cost.is_finite() || profile[x..x + c.width()].iter().any(|&h| h as usize != y) {
            continue;
        }
        let next = advance(profile, c);
        let bound = match s.memo.get(profile_key(next)) {
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
                    choice: Choice::Tile(k as u16),
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
    let choice = s.region_choices[region_index(x, y, w, h)];
    match choice {
        0 => unreachable!("infeasible rectangle in a finite fallback"),
        1..=REGION_LAST_TILE => out.push(usize::from(choice - 1)),
        REGION_HORIZONTAL_BASE.. => {
            let cut = usize::from(choice - REGION_HORIZONTAL_BASE);
            unpack_region(x, y, w, cut, s, out);
            unpack_region(x, y + cut, w, h - cut, s, out);
        }
        _ => {
            let cut = usize::from(choice - REGION_VERTICAL_BASE);
            unpack_region(x, y, cut, h, s, out);
            unpack_region(x + cut, y, w - cut, h, s, out);
        }
    }
}

fn plan(
    candidates: &[Candidate],
    width: usize,
    height: usize,
    s: &mut PlannerScratch,
    out: &mut Vec<usize>,
) -> f64 {
    prepare_planner(candidates, width, height, s);
    let root = visit([0; SIDE], width, height, f64::INFINITY, candidates, s).unwrap_or(EMPTY);
    out.clear();
    if !root.cost.is_finite() {
        return root.cost;
    }
    let mut profile = [0; SIDE];
    while first_open(profile, width).1 < height {
        let Some(Memo::Solved(entry)) = s.memo.get(profile_key(profile)) else {
            unreachable!("a winning backpointer must have an exact suffix");
        };
        match entry.choice {
            Choice::Tile(k) => {
                let k = usize::from(k);
                out.push(k);
                profile = advance(profile, candidates[k]);
            }
            Choice::Fallback => {
                unpack_fallback(profile, width, height, s, out);
                break;
            }
            _ => unreachable!("invalid frontier backpointer"),
        }
    }
    root.cost
}

fn prepare_planner(candidates: &[Candidate], width: usize, height: usize, s: &mut PlannerScratch) {
    assert!((1..=SIDE).contains(&width) && (1..=SIDE).contains(&height));
    debug_assert!(candidates.len() <= MAX_CANDIDATES);
    s.memo.clear();
    s.budget = VISIT_BUDGET;
    s.anchors.resize_with(SIDE * SIDE, Vec::new);
    for anchor in &mut s.anchors {
        anchor.clear();
    }
    for &c in candidates {
        assert!(c.x() + c.width() <= width && c.y() + c.height() <= height);
    }
    prepare_packing_bound(candidates, width, height, s);
    for (i, &c) in candidates.iter().enumerate() {
        if !s.dominated[i] {
            s.anchors[c.y() * SIDE + c.x()].push(i);
        }
    }
    prepare_regions(candidates, width, height, SIDE, SIDE, s);
    prepare_bounds(candidates, width, height, s);
}

fn unpack_fallback(
    profile: [u8; SIDE],
    width: usize,
    height: usize,
    s: &PlannerScratch,
    out: &mut Vec<usize>,
) {
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
}

/// Only the final cost and the accepted piece's baseline outlive scoring.
#[derive(Clone, Copy)]
struct CachedCost {
    cost: f32,
    base: f32,
}

impl Default for CachedCost {
    fn default() -> Self {
        Self {
            cost: f32::NAN,
            base: f32::NAN,
        }
    }
}

#[derive(Default)]
pub(crate) struct LayoutDpScratch {
    reconstruction: Vec<CachedCost>,
    /// Scores by absolute block position and strategy, kept across the
    /// tile grids of one re-tiling so a shifted grid rescores only the
    /// pieces the aligned grid could not place.
    global: Option<OpenTable<CachedCost>>,
    candidates: Vec<Candidate>,
    planner: PlannerScratch,
    current: Vec<Piece>,
    layout: Vec<Piece>,
    choices: Vec<usize>,
    best: Vec<MergeUpgrade>,
    small: Option<SmallPlanner>,
    compact: compact::CompactPlanner,
    proposals: Vec<crate::dc_group_data::DpRegion>,
}
impl LayoutDpScratch {
    fn start_grid(&mut self, grid: Grid) -> (bool, (usize, usize)) {
        match grid {
            Grid::Aligned { keep_scores } => {
                // Aligned tiles have disjoint anchors. Only a later shifted
                // grid can use scores outside the tile-local cache.
                if keep_scores {
                    self.global.get_or_insert_with(OpenTable::default).clear();
                } else {
                    self.global = None;
                }
                (false, (0, 0))
            }
            Grid::Shifted(offset) => (true, offset),
        }
    }

    fn reset(&mut self) {
        self.reconstruction.clear();
        self.reconstruction
            .resize(CACHE_SIZE, CachedCost::default());
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
    ) -> CachedCost {
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
    ) -> CachedCost {
        let key = p.key();
        if cache.reconstruction[key].cost.is_nan() {
            let params = self.rerank.params;
            let (x, y) = (self.bx + p.x as usize, self.by + p.y as usize);
            let global_key = ((y * 256 + x) << 5 | p.strategy as usize) as u32;
            if let Some(hit) = cache
                .global
                .as_ref()
                .and_then(|table| table.get(global_key))
            {
                cache.reconstruction[key] = hit;
                return hit;
            }
            let (px, py) = (params.dc_group_px + x * 8, params.dc_group_py + y * 8);
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
                None,
                limit,
            );
            if crate::dc_group_data::is_gated_sub8_strategy(p.strategy) {
                let (lambda, bits) =
                    fine_admit_charge(coder.rate_prices.is_some(), params.distance, score.rate);
                score.cost = fmla(lambda, bits, score.cost);
            }
            let score = CachedCost {
                cost: score.cost,
                base: score.base,
            };
            if let Some(table) = &mut cache.global
                && score.cost.is_finite()
            {
                table.insert(global_key, score);
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
/// Membership of complete pieces, indexed by strategy and top-left cell.
#[derive(Default)]
struct PieceSet([u64; crate::dc_group_data::NUM_STRATEGIES]);

impl PieceSet {
    fn from_slice(pieces: &[Piece]) -> Self {
        let mut set = Self::default();
        for &piece in pieces {
            set.insert(piece);
        }
        set
    }

    fn insert(&mut self, piece: Piece) {
        self.0[piece.strategy as usize] |= 1 << (piece.y as usize * SIDE + piece.x as usize);
    }

    fn contains(&self, piece: Piece) -> bool {
        self.0[piece.strategy as usize] & (1 << (piece.y as usize * SIDE + piece.x as usize)) != 0
    }
}

/// The minimal closed rectangles (x, y, w, h) covering every block whose
/// piece differs between `current` and `layout`: connected changed blocks
/// grown until no piece of either layout crosses the border, with
/// overlapping rectangles merged.
fn changed_regions(
    current: &[Piece],
    layout: &[Piece],
    mut changed: u64,
) -> Vec<(usize, usize, usize, usize)> {
    let footprint = |p: &Piece| (p.x as usize, p.y as usize, p.width(), p.height());
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
    let mut regions: Vec<(usize, usize, usize, usize)> = Vec::new();
    while changed != 0 {
        // Start with the first remaining cell in raster order. Flood its
        // four-connected neighbors without crossing either horizontal edge.
        let mut component = 1u64 << changed.trailing_zeros();
        loop {
            let adjacent = ((component & 0xfefe_fefe_fefe_fefe) >> 1)
                | ((component & 0x7f7f_7f7f_7f7f_7f7f) << 1)
                | (component >> SIDE)
                | (component << SIDE);
            let expanded = component | (adjacent & changed);
            if expanded == component {
                break;
            }
            component = expanded;
        }
        changed &= !component;
        // Collapse the rows to locate the two extreme columns.
        let columns = component | component >> 32;
        let columns = columns | columns >> 16;
        let columns = (columns | columns >> 8) as u8;
        let x0 = columns.trailing_zeros() as usize;
        let x1 = SIDE - columns.leading_zeros() as usize;
        let y0 = component.trailing_zeros() as usize / SIDE;
        let y1 = SIDE - component.leading_zeros() as usize / SIDE;
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
        let piece = p.mask(SIDE);
        debug_assert_eq!(mask & piece, 0, "layout overlaps");
        mask |= piece;
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
    crossing: bool,
    accepted: &mut Vec<MergeUpgrade>,
) {
    cache.reset();
    let mut current = std::mem::take(&mut cache.current);
    current.clear();
    for y in 0..height {
        for x in 0..width {
            if map.is_first_block(scores.bx + x, scores.by + y) {
                let p = Piece::new(x, y, map.raw_strategy(scores.bx + x, scores.by + y));
                if !crossing || (x + p.width() <= width && y + p.height() <= height) {
                    current.push(p);
                }
            }
        }
    }
    // On a shifted grid the cells of incumbent pieces that cross the tile
    // border keep their piece, as zero-cost placeholders in the plan.
    let full = (0..height).fold(0u64, |m, y| m | (((1u64 << width) - 1) << (y * 8)));
    let blocked = if crossing {
        full & !pieces_mask(&current)
    } else {
        0
    };
    let is_blocked = |x: usize, y: usize| blocked >> (y * 8 + x) & 1 != 0;
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
    let incumbent = PieceSet::from_slice(&current);
    let mut considered = PieceSet::default();
    for (i, &strategy) in STRATEGIES[..families].iter().enumerate() {
        // At each family boundary prepare only the shapes read by that
        // family's score caps and their guillotine subrectangles.
        match i {
            3 => {
                compact::prepare_square_bounds(&cache.candidates, width, height, &mut cache.planner)
            }
            4 | 6 | 8 => {
                let short_side = 1 << ((i - 4) / 2);
                if width.min(height) >= short_side && width.max(height) >= 4 {
                    prepare_regions(
                        &cache.candidates,
                        width,
                        height,
                        4,
                        short_side,
                        &mut cache.planner,
                    );
                }
            }
            _ => {}
        }
        let w = AcStrategyImage::covered_blocks_x_of(strategy);
        let h = AcStrategyImage::covered_blocks_y_of(strategy);
        if w > width || h > height {
            continue;
        }
        for y in 0..=height - h {
            for x in 0..=width - w {
                let p = Piece::new(x, y, strategy);
                let legal = if crossing {
                    map.can_place_strategy_crossing(scores.bx + x, scores.by + y, strategy)
                } else {
                    map.can_place_strategy(scores.bx + x, scores.by + y, strategy)
                };
                if !legal {
                    continue;
                }
                if blocked != 0 {
                    if strategy == STRATEGY_DCT && is_blocked(x, y) {
                        considered.insert(p);
                        cache.candidates.push(Candidate::new(p, 0.0));
                        continue;
                    }
                    if p.mask(SIDE) & blocked != 0 {
                        continue;
                    }
                }
                let candidate = if strategy == STRATEGY_DCT {
                    leaf(scores, coder, cache, map, x, y)
                } else {
                    let limit = if i < 3 || incumbent.contains(p) {
                        f32::INFINITY
                    } else {
                        (cache.planner.region_costs[region_index(x, y, w, h)] as f32 * BOUND_SLACK)
                            .next_up()
                    };
                    Candidate::new(
                        p,
                        f64::from(scores.reconstruction_bounded(coder, cache, p, limit).cost),
                    )
                };
                considered.insert(candidate.piece);
                cache.candidates.push(candidate);
            }
        }
    }
    // The incumbent layout stays feasible: its 64-family blocks, which only
    // the 64 pass places, compete as themselves.
    for &p in &current {
        if p.width() * p.height() > 1 && !considered.contains(p) {
            let cost = scores.reconstruction(coder, cache, p).cost as f64;
            cache.candidates.push(Candidate::new(p, cost));
        }
    }
    debug_assert!(cache.candidates.len() <= MAX_CANDIDATES);
    let winner = compact::plan(
        &cache.candidates,
        width,
        height,
        &mut cache.planner,
        &mut cache.compact,
        &mut cache.choices,
    );
    if !winner.is_finite() {
        cache.current = current;
        return;
    }
    let mut layout = std::mem::take(&mut cache.layout);
    layout.clear();
    for &i in &cache.choices {
        let p = cache.candidates[i].piece;
        if !(p.strategy == STRATEGY_DCT && is_blocked(p.x as usize, p.y as usize)) {
            layout.push(p);
        }
    }
    layout.sort_by_key(|p| (p.y, p.x, p.strategy));
    debug_assert_eq!(pieces_mask(&layout), pieces_mask(&current));
    // The incumbent can contain fine leaves outside this pass's candidate set.
    // Compare only changed pieces so unchanged costs cancel without rounding.
    let winner = PieceSet::from_slice(&layout);
    let mut changed = 0;
    let old = current
        .iter()
        .copied()
        .filter(|&p| !winner.contains(p))
        .map(|p| {
            changed |= p.mask(SIDE);
            scores.reconstruction(coder, cache, p).cost
        })
        .sum::<f32>();
    let new = layout
        .iter()
        .copied()
        .filter(|&p| !incumbent.contains(p))
        .map(|p| {
            changed |= p.mask(SIDE);
            scores.reconstruction(coder, cache, p).cost
        })
        .sum::<f32>();
    if new < old {
        for p in layout.iter().copied().filter(|&p| !incumbent.contains(p)) {
            let base = scores.reconstruction(coder, cache, p).base;
            cache.best.push(MergeUpgrade {
                bx: scores.bx + p.x as usize,
                by: scores.by + p.y as usize,
                strategy: p.strategy,
                base,
            });
        }
        for (x, y, w, h) in changed_regions(&current, &layout, changed) {
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
    /// The band's first row and the merge search's DCT8 costs of its rows:
    /// the same `strategy_cost` this pass would compute.
    dct8: (usize, &'a [f32]),
}

impl CoefficientScores<'_, '_> {
    fn score(&self, coder: &mut CoderScratch, cache: &mut LayoutDpScratch, p: Piece) -> f32 {
        let key = p.key();
        if cache.reconstruction[key].cost.is_nan() {
            let params = self.selection.params;
            let (x, y) = (self.bx + p.x as usize, self.by + p.y as usize);
            let global_key = ((y * 256 + x) << 5 | p.strategy as usize) as u32;
            if let Some(hit) = cache
                .global
                .as_ref()
                .and_then(|table| table.get(global_key))
            {
                cache.reconstruction[key] = hit;
                return hit.cost;
            }
            let (px, py) = (params.dc_group_px + x * 8, params.dc_group_py + y * 8);
            let stride = self.selection.quant_field.xsize();
            let cached = (p.strategy == STRATEGY_DCT && y >= self.dct8.0)
                .then(|| self.dct8.1.get((y - self.dct8.0) * stride + x).copied())
                .flatten()
                .filter(|cost| !cost.is_nan());
            let cost = cached.unwrap_or_else(|| {
                strategy_cost(
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
                )
            });
            let score = CachedCost {
                cost,
                ..CachedCost::default()
            };
            if let Some(table) = &mut cache.global
                && cost.is_finite()
            {
                table.insert(global_key, score);
            }
            cache.reconstruction[key] = score;
        }
        cache.reconstruction[key].cost
    }
}

/// Choice of the small planner at a cell.
const SMALL_PREFILLED: u8 = 0;
const SMALL_LEAF: u8 = 1;
const SMALL_WIDE: u8 = 2;
const SMALL_TALL: u8 = 3;
const SMALL_SQUARE: u8 = 4;
const SMALL_STATES: usize = 1 << (SIDE + 1);

/// Exact broken-profile DP over 1x1, 2x1, 1x2 and 2x2 pieces on a `w x h`
/// grid of at most 8x8 cells. Cells are visited in row-major order; a state
/// holds the filled flags of the next `w + 1` cells. Only a square anchored
/// at the current cell can reach the following cell, so its bit comes from
/// the fixed mask and is shared by every state in the layer.
struct SmallPlanner {
    cost: Box<[f32; SMALL_STATES]>,
    next: Box<[f32; SMALL_STATES]>,
    /// States with a finite cost in `cost` / `next`, the first `active_len`
    /// and `next_len` entries. Each state enters a list at most once.
    active: Box<[u16; SMALL_STATES]>,
    next_active: Box<[u16; SMALL_STATES]>,
    active_len: usize,
    /// The winning placement per cell and state. Its predecessor is recovered
    /// by undoing the shift and removing the cells filled by the placement.
    choices: Vec<u8>,
    plan: Vec<(usize, u8)>,
}

impl Default for SmallPlanner {
    fn default() -> Self {
        Self {
            cost: heap_array(f32::INFINITY),
            next: heap_array(f32::INFINITY),
            active: heap_array(0),
            next_active: heap_array(0),
            active_len: 0,
            choices: Vec::new(),
            plan: Vec::new(),
        }
    }
}

/// Costs of the four piece shapes anchored at each cell (`y * w + x`).
struct SmallCosts {
    leaf: [f32; SIDE * SIDE],
    wide: [f32; SIDE * SIDE],
    tall: [f32; SIDE * SIDE],
    square: [f32; SIDE * SIDE],
}

impl SmallCosts {
    /// Prune pieces that cost at least as much as a tiling of the same cells
    /// by smaller pieces.
    fn live_masks(&self, w: usize, h: usize) -> [u8; SIDE * SIDE] {
        let cells = w * h;
        let mut live = [0; SIDE * SIDE];
        if cells == 0 {
            return live;
        }

        for (y, masks) in live[..cells].chunks_exact_mut(w).enumerate() {
            let start = y * w;
            let end = start + w;
            let leaf = &self.leaf[start..end];
            let wide = &self.wide[start..end];
            if y + 1 == h {
                for ((mask, &[left, right]), &wide) in
                    masks.iter_mut().zip(leaf.array_windows::<2>()).zip(wide)
                {
                    *mask = u8::from(wide < left + right);
                }
                continue;
            }

            let leaf_below = &self.leaf[end..end + w];
            let wide_below = &self.wide[end..end + w];
            let tall = &self.tall[start..end];
            let square = &self.square[start..end];
            let leaves = leaf
                .array_windows::<2>()
                .zip(leaf_below.array_windows::<2>());
            let interior = masks
                .iter_mut()
                .zip(leaves)
                .zip(wide.iter().zip(wide_below))
                .zip(tall.array_windows::<2>().zip(square));
            for (((mask, leaves), (&wide, &wide_below)), (&[tall, tall_right], &square)) in interior
            {
                let (&[top_left, top_right], &[bottom_left, bottom_right]) = leaves;
                let top_leaves = top_left + top_right;
                let leaves = top_leaves + bottom_left + bottom_right;
                let wides = wide + wide_below;
                let talls = tall + tall_right;
                *mask = u8::from(wide < top_leaves)
                    | u8::from(tall < top_left + bottom_left) << 1
                    | u8::from(square < leaves.min(wides).min(talls)) << 2;
            }

            // The windows stop before the right edge, where only a tall piece fits.
            let right = w - 1;
            masks[right] = u8::from(tall[right] < leaf[right] + leaf_below[right]) << 1;
        }
        live
    }
}

/// Returns the optimum. If it beats `replace_below`, save the placed pieces
/// in the scratch plan. `fixed` marks cells that are already covered.
fn plan_small(
    costs: &SmallCosts,
    live: &[u8; SIDE * SIDE],
    fixed: u64,
    [w, h]: [usize; 2],
    s: &mut SmallPlanner,
    replace_below: f32,
) -> f32 {
    let cells = w * h;
    let states = 1 << (w + 1);
    let mask = states - 1;
    if s.choices.len() < cells * states {
        s.choices.resize(cells * states, 0);
    }
    // The index mask is a no-op that lets the compiler drop checks.
    const STATE_MASK: usize = SMALL_STATES - 1;
    let start = fixed as usize & mask;
    let below_bit = 1 << w;
    let corner_bit = 2 << w;
    let added = [0, 0, 2, below_bit, 2 | below_bit | corner_bit];
    let SmallPlanner {
        cost,
        next,
        active,
        next_active,
        active_len,
        choices,
        ..
    } = &mut *s;
    cost[start] = 0.0;
    active[0] = start as u16;
    *active_len = 1;
    let mut incoming = fixed >> (w + 1);
    for (i, from) in choices[..cells * states]
        .chunks_exact_mut(states)
        .enumerate()
    {
        let corner = (incoming as usize & 1) << w;
        incoming >>= 1;
        let shapes = live[i];
        let mut next_len = 0usize;
        for &state in &active[..*active_len] {
            let state = state as usize & STATE_MASK;
            let here = cost[state];
            cost[state] = f32::INFINITY;
            let mut relax = |filled: usize, cost: f32, choice: u8| {
                let to = ((filled >> 1) | corner) & STATE_MASK;
                if cost < next[to] {
                    if next[to] == f32::INFINITY {
                        next_active[next_len & STATE_MASK] = to as u16;
                        next_len += 1;
                    }
                    next[to] = cost;
                    from[to] = choice;
                }
            };
            if state & 1 != 0 {
                relax(state, here, SMALL_PREFILLED);
                continue;
            }
            relax(state, here + costs.leaf[i], SMALL_LEAF);
            let right = state & 2 == 0;
            let below = state & below_bit == 0;
            if right && shapes & 1 != 0 {
                relax(state | 2, here + costs.wide[i], SMALL_WIDE);
            }
            if below && shapes & 2 != 0 {
                relax(state | below_bit, here + costs.tall[i], SMALL_TALL);
            }
            if right && below && corner == 0 && shapes & 4 != 0 {
                relax(
                    state | added[SMALL_SQUARE as usize],
                    here + costs.square[i],
                    SMALL_SQUARE,
                );
            }
        }
        std::mem::swap(cost, next);
        std::mem::swap(active, next_active);
        *active_len = next_len;
    }
    let total = s.cost[0];
    for &state in &s.active[..s.active_len] {
        s.cost[state as usize & STATE_MASK] = f32::INFINITY;
    }
    if !(total.is_finite() && total < replace_below) {
        return total;
    }
    s.plan.clear();
    let mut state = 0;
    for i in (0..cells).rev() {
        // Every reached predecessor was written during this plan. Unreached
        // entries may retain old values; no clearing or layer copies are needed.
        let choice = s.choices[i * states + state];
        if choice != SMALL_PREFILLED {
            s.plan.push((i, choice));
        }
        state =
            ((state << 1) ^ added[choice as usize]) & mask | usize::from(choice == SMALL_PREFILLED);
    }
    total
}

fn small_strategy(choice: u8) -> u8 {
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
    crossing: bool,
    accepted: &mut Vec<MergeUpgrade>,
) {
    cache.reset();
    let mut current = std::mem::take(&mut cache.current);
    current.clear();
    for y in 0..height {
        for x in 0..width {
            if map.is_first_block(scores.bx + x, scores.by + y) {
                let p = Piece::new(x, y, map.raw_strategy(scores.bx + x, scores.by + y));
                if !crossing || (x + p.width() <= width && y + p.height() <= height) {
                    current.push(p);
                }
            }
        }
    }
    // On a shifted grid the cells of incumbents crossing the tile border stay
    // out of the plan (cell bits are `y * width + x` here).
    let blocked = if crossing {
        let full = (0..width * height).fold(0u64, |m, i| m | 1 << i);
        current.iter().fold(full, |m, &p| m & !p.mask(width))
    } else {
        0
    };
    if current
        .iter()
        .any(|p| crate::dc_group_data::is_gated_sub8_strategy(p.strategy))
    {
        cache.current = current;
        return;
    }
    let unaligned = scores
        .selection
        .params
        .ctx
        .speed
        .effort()
        .layout_dp_unaligned;
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
                if x + p.width() > width
                    || y + p.height() > height
                    || (blocked != 0 && p.mask(width) & blocked != 0)
                {
                    continue;
                }
                // Without unaligned placements a piece stays inside its
                // aligned 2x2 quad.
                let (gx, gy) = (scores.bx + x, scores.by + y);
                let off_grid = match choice {
                    SMALL_WIDE => gx % 2 != 0,
                    SMALL_TALL => gy % 2 != 0,
                    SMALL_SQUARE => gx % 2 != 0 || gy % 2 != 0,
                    _ => false,
                };
                if off_grid && !unaligned {
                    continue;
                }
                let legal = if crossing {
                    map.can_place_strategy_crossing(scores.bx + x, scores.by + y, p.strategy)
                } else {
                    map.can_place_strategy(scores.bx + x, scores.by + y, p.strategy)
                };
                if legal {
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
        fixed |= p.mask(width);
        fixed_cost += scores.score(coder, cache, p);
    }
    let mut layout = std::mem::take(&mut cache.layout);
    let live = costs.live_masks(width, height);
    let small = cache.small.get_or_insert_with(SmallPlanner::default);
    let around = plan_small(
        &costs,
        &live,
        fixed | blocked,
        [width, height],
        small,
        f32::INFINITY,
    ) + fixed_cost;
    layout.clear();
    let mut keep_fixed = true;
    let mut winner = around;
    if fixed != 0 {
        let over = plan_small(&costs, &live, blocked, [width, height], small, around);
        if over < around {
            winner = over;
            keep_fixed = false;
        }
    }
    if winner.is_finite() {
        layout.extend(
            small
                .plan
                .iter()
                .map(|&(i, choice)| Piece::new(i % width, i / width, small_strategy(choice))),
        );
        if keep_fixed {
            layout.extend(
                current
                    .iter()
                    .filter(|p| !COEFFICIENT_STRATEGIES.contains(&p.strategy)),
            );
        }
        layout.sort_by_key(|p| (p.y, p.x, p.strategy));
        debug_assert_eq!(pieces_mask(&layout), pieces_mask(&current));
        let was = PieceSet::from_slice(&current);
        let now = PieceSet::from_slice(&layout);
        let old = current
            .iter()
            .copied()
            .filter(|&p| !now.contains(p))
            .map(|p| scores.score(coder, cache, p))
            .sum::<f32>();
        let new = layout
            .iter()
            .copied()
            .filter(|&p| !was.contains(p))
            .map(|p| scores.score(coder, cache, p))
            .sum::<f32>();
        if new < old {
            for p in layout.iter().copied().filter(|&p| !was.contains(p)) {
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
    grid: Grid,
) {
    output.merge_upgrades.clear();
    output.dp_regions.clear();
    let mut cache = std::mem::take(&mut output.layout_dp);
    let (crossing, (ox, oy)) = cache.start_grid(grid);
    let first_y = oy + band.0.saturating_sub(oy).div_ceil(TILE) * TILE;
    for by in (first_y..band.1.min(map.ysize())).step_by(TILE) {
        for bx in (ox..map.xsize()).step_by(TILE) {
            let (w, h) = ((map.xsize() - bx).min(TILE), (map.ysize() - by).min(TILE));
            if w * h < 2
                || (!crossing && !footprint_is_self_contained(map, bx, by, w, h))
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
                crossing,
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
    let offsets = crossing_grids(rerank.params.ctx.speed.effort());
    let read: &AcStrategyImage = map;
    rerank
        .params
        .ctx
        .thread_pool
        .steal_for_each_mut_with_threads(scratch, bands, rerank.num_threads, |i, out, coder| {
            let mut prices = rerank.params.price(coder);
            find(
                rerank,
                &mut prices,
                read,
                rerank.bands[i],
                out,
                &rollbacks,
                Grid::Aligned {
                    keep_scores: install && !offsets.is_empty(),
                },
            );
        });
    if install {
        apply(map, bands, current_costs);
        // Further passes on shifted grids, where pieces may cross the 64px
        // tiles the aligned grid fences. Each pass sees the previous one's
        // layout; the score cache carries over.
        for &offset in offsets {
            let read: &AcStrategyImage = map;
            rerank
                .params
                .ctx
                .thread_pool
                .steal_for_each_mut_with_threads(
                    scratch,
                    bands,
                    rerank.num_threads,
                    |i, out, coder| {
                        let mut prices = rerank.params.price(coder);
                        find(
                            rerank,
                            &mut prices,
                            read,
                            rerank.bands[i],
                            out,
                            &rollbacks,
                            Grid::Shifted(offset),
                        );
                    },
                );
            apply(map, bands, current_costs);
        }
    }
}

/// The tile grid a re-tiling pass runs on.
#[derive(Clone, Copy)]
enum Grid {
    /// The 64px CfL tiles; pieces stay inside their tile.
    Aligned { keep_scores: bool },
    /// A grid shifted by this many blocks, whose pieces may cross the CfL
    /// tiles (the decoder takes the tile of a piece's first block).
    Shifted((usize, usize)),
}

fn find_coefficient(
    selection: &SelectionContext<'_>,
    coder: &mut CoderScratch,
    map: &AcStrategyImage,
    band: (usize, usize),
    output: &mut AcStrategyBandScratch,
    grid: Grid,
) {
    output.merge_upgrades.clear();
    let mut cache = std::mem::take(&mut output.layout_dp);
    let (crossing, (ox, oy)) = cache.start_grid(grid);
    let first_y = oy + band.0.saturating_sub(oy).div_ceil(TILE) * TILE;
    for by in (first_y..band.1.min(map.ysize())).step_by(TILE) {
        for bx in (ox..map.xsize()).step_by(TILE) {
            let (w, h) = ((map.xsize() - bx).min(TILE), (map.ysize() - by).min(TILE));
            if w * h < 2 || (!crossing && !footprint_is_self_contained(map, bx, by, w, h)) {
                continue;
            }
            search_tile_coefficient(
                &CoefficientScores {
                    selection,
                    bx,
                    by,
                    dct8: (band.0, &output.dct8_costs),
                },
                coder,
                &mut cache,
                map,
                w,
                h,
                crossing,
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
    let offsets = crossing_grids(selection.params.ctx.speed.effort());
    let read: &AcStrategyImage = map;
    selection
        .params
        .ctx
        .thread_pool
        .steal_for_each_mut_with_threads(scratch, band_scratch, num_threads, |i, out, coder| {
            let mut prices = selection.params.price(coder);
            find_coefficient(
                selection,
                &mut prices,
                read,
                bands[i],
                out,
                Grid::Aligned {
                    keep_scores: !offsets.is_empty(),
                },
            );
        });
    apply(map, band_scratch, current_costs);
    for &offset in offsets {
        let read: &AcStrategyImage = map;
        selection
            .params
            .ctx
            .thread_pool
            .steal_for_each_mut_with_threads(
                scratch,
                band_scratch,
                num_threads,
                |i, out, coder| {
                    let mut prices = selection.params.price(coder);
                    find_coefficient(
                        selection,
                        &mut prices,
                        read,
                        bands[i],
                        out,
                        Grid::Shifted(offset),
                    );
                },
            );
        apply(map, band_scratch, current_costs);
    }
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
            for cy in y..y + c.height() {
                for cx in x..x + c.width() {
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
        assert!(gate(crate::Speed::Fastest));
        assert!(gate(crate::Speed::Fast));
        assert!(gate(crate::Speed::Medium));
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
        assert!(gate(crate::Speed::MediumSlow));
        assert!(gate(crate::Speed::Slow));
        assert_eq!(gate(crate::Speed::ExtraSlow), !cfg!(feature = "extraslow"));
        for speed in [
            crate::Speed::UltraFast,
            crate::Speed::Fastest,
            crate::Speed::Fast,
            crate::Speed::Medium,
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
        assert!(s.region_costs[region_index(0, 0, 3, 3)] > cost);
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
        assert!(c.iter().all(|c| c.width() * c.height() <= 16));
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
        let changed = |current: &[Piece], layout: &[Piece]| {
            let cells = current
                .iter()
                .filter(|p| !layout.contains(p))
                .chain(layout.iter().filter(|p| !current.contains(p)))
                .fold(0, |cells, &piece| cells | piece.mask(SIDE));
            changed_regions(current, layout, cells)
        };
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
        let mut regions = changed(&current, &layout);
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
        assert_eq!(changed(&big, &shifted), vec![(0, 0, 5, 4)]);
        assert!(changed(&current, &current).is_empty());
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
            assert!(matches!(scratch.memo.get(0),Some(Memo::AtLeast(bound)) if bound >= budget));
        }
        assert_eq!(
            visit([0; SIDE], 3, 3, 6.0, &c, &mut scratch).unwrap().cost,
            5.0
        );
        assert!(matches!(scratch.memo.get(0),Some(Memo::Solved(entry)) if entry.cost == 5.0));
    }

    #[test]
    fn entering_column_uses_the_earliest_original_column_on_ties() {
        assert_eq!(
            entering_column(&[0.0, -0.0, -1e-10, f64::NAN], &[0, 1, 2, 3]),
            None
        );
        assert_eq!(
            entering_column(
                &[f64::NAN, -2.0, -1.0, -0.0, -2.0, f64::INFINITY, -2.0],
                &[0, 4, 5, 3, 1, 6, 2],
            ),
            Some(4),
        );
        assert_eq!(
            entering_column(&[-2.0, 0.0, 0.0, 0.0, f64::NEG_INFINITY], &[0, 1, 2, 3, 4]),
            Some(4)
        );
    }

    #[test]
    fn open_table_growth_preserves_current_generation() {
        let mut table = OpenTable::default();
        let key = |i: u32| i.wrapping_mul(0x9e37_79b1);
        for i in 0u32..129 {
            table.insert(key(i), i);
        }
        for i in 0u32..129 {
            assert_eq!(table.get(key(i)), Some(i));
        }
        table.clear();
        for i in 129u32..1024 {
            table.insert(key(i), i);
        }
        for i in 0u32..129 {
            assert_eq!(table.get(key(i)), None);
        }
        for i in 129u32..1024 {
            assert_eq!(table.get(key(i)), Some(i));
        }
        assert_eq!(table.len, 895);
    }

    #[test]
    fn restricted_guillotine_regions_match_full_table() {
        let mut full = PlannerScratch::default();
        let mut restricted = PlannerScratch::default();
        for width in 1..=SIDE {
            for height in (1..=SIDE).rev() {
                for (families, short_side) in [(4, 1), (6, 2), (8, 4)] {
                    let mut candidates = Vec::new();
                    for_each_piece(width, height, |p| {
                        if !STRATEGIES[..families].contains(&p.strategy) {
                            return;
                        }
                        let cost = match p.key() % 19 {
                            0 => f64::INFINITY,
                            1 => -0.0,
                            2 => 0.0,
                            n => (n % 4) as f64 - 1.0,
                        };
                        candidates.push(Candidate::new(p, cost));
                    });
                    prepare_regions(&candidates, width, height, SIDE, SIDE, &mut full);
                    prepare_regions(&candidates, width, height, 4, short_side, &mut restricted);
                    for (w, h) in [(4, short_side), (short_side, 4)] {
                        if w > width || h > height {
                            continue;
                        }
                        for y in 0..=height - h {
                            for x in 0..=width - w {
                                let i = region_index(x, y, w, h);
                                assert_eq!(
                                    full.region_costs[i].to_bits(),
                                    restricted.region_costs[i].to_bits()
                                );
                                assert_eq!(full.region_choices[i], restricted.region_choices[i]);
                            }
                        }
                    }
                }
            }
        }
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
                let live = costs.live_masks(w, h);
                let actual = plan_small(&costs, &live, fixed, [w, h], &mut s, f32::INFINITY);
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected,
                    "{w}x{h}: {actual} vs {expected}"
                );
                let mut covered = fixed;
                let mut total = 0.0;
                for &(i, choice) in &s.plan {
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
        let costs = SmallCosts {
            leaf: [4.0; SIDE * SIDE],
            wide: [7.5; SIDE * SIDE],
            tall: [7.5; SIDE * SIDE],
            square: [13.0; SIDE * SIDE],
        };
        let live = costs.live_masks(8, 8);
        assert_eq!(
            plan_small(&costs, &live, 0, [8, 8], &mut s, f32::INFINITY),
            16.0 * 13.0
        );
        assert_eq!(s.plan.len(), 16);
        assert!(s.plan.iter().all(|&(_, c)| c == SMALL_SQUARE));
        // With the top-left 4x4 fixed, the rest still tiles with squares.
        let fixed = (0..4).fold(0u64, |m, y| m | 0b1111 << (y * 8));
        assert_eq!(
            plan_small(&costs, &live, fixed, [8, 8], &mut s, f32::INFINITY),
            12.0 * 13.0
        );
        assert_eq!(s.plan.len(), 12);
        // A cost tie leaves the previous layout in place.
        assert_eq!(
            plan_small(&costs, &live, 0, [8, 8], &mut s, 16.0 * 13.0),
            16.0 * 13.0
        );
        assert_eq!(s.plan.len(), 12);
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
                    false,
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
        find_coefficient(
            &selection,
            &mut coder,
            &map,
            (0, 8),
            &mut output,
            Grid::Aligned { keep_scores: false },
        );
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
        find_coefficient(
            &selection,
            &mut coder,
            &large,
            (0, 16),
            &mut output,
            Grid::Aligned { keep_scores: false },
        );
        assert!(output.merge_upgrades.is_empty());
        // A shifted grid keeps the cells of an incumbent crossing its tile
        // border out of the plan and never re-tiles over them.
        let mut crossed = AcStrategyImage::new(8, 8);
        crossed.set_first(2, 2, STRATEGY_DCT32X32);
        find_coefficient(
            &selection,
            &mut coder,
            &crossed,
            (0, 8),
            &mut output,
            Grid::Shifted((4, 4)),
        );
        for up in &output.merge_upgrades {
            let w = AcStrategyImage::covered_blocks_x_of(up.strategy);
            let h = AcStrategyImage::covered_blocks_y_of(up.strategy);
            assert!(
                up.bx >= 6 || up.bx + w <= 4 || up.by >= 6 || up.by + h <= 4,
                "upgrade at ({}, {}) overlaps the crossing incumbent",
                up.bx,
                up.by
            );
        }
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
        find(
            &rerank,
            &mut coder,
            &large,
            (0, 16),
            &mut output,
            &[],
            Grid::Aligned { keep_scores: false },
        );
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
            Grid::Aligned { keep_scores: false },
        );
        assert!(output.merge_upgrades.is_empty());
        assert!(
            output.layout_dp.reconstruction.is_empty(),
            "fine-mosaic rollback regions must remain intact"
        );
        small.set_first(0, 0, crate::dc_group_data::STRATEGY_IDENTITY);
        find(
            &rerank,
            &mut coder,
            &small,
            (0, 8),
            &mut output,
            &[],
            Grid::Aligned { keep_scores: false },
        );
        assert!(
            output.merge_upgrades.is_empty(),
            "legacy fine credits cannot be removed"
        );
        let thin = AcStrategyImage::new(1, 8);
        find(
            &rerank,
            &mut coder,
            &thin,
            (0, 8),
            &mut output,
            &[],
            Grid::Aligned { keep_scores: false },
        );
        assert!(
            !output.layout_dp.candidates.is_empty(),
            "8-pixel slivers must enter the DP"
        );
        assert!(output.layout_dp.candidates.iter().all(|c| c.width() == 1));

        // A shifted grid keeps the cells of an incumbent crossing its tile
        // border out of the plan and never re-tiles over them.
        let mut crossed = AcStrategyImage::new(8, 8);
        crossed.set_first(2, 2, STRATEGY_DCT32X32);
        find(
            &rerank,
            &mut coder,
            &crossed,
            (0, 8),
            &mut output,
            &[],
            Grid::Shifted((4, 4)),
        );
        assert!(!output.layout_dp.candidates.is_empty());
        for up in &output.merge_upgrades {
            let w = AcStrategyImage::covered_blocks_x_of(up.strategy);
            let h = AcStrategyImage::covered_blocks_y_of(up.strategy);
            assert!(
                up.bx >= 6 || up.bx + w <= 4 || up.by >= 6 || up.by + h <= 4,
                "upgrade at ({}, {}) overlaps the crossing incumbent",
                up.bx,
                up.by
            );
        }
    }
}
