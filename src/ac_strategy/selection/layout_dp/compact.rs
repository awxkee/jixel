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

//! Reconstruction planner for leaves, pairs and 2x2 squares. Use the broken
//! profile of the coefficient solver, but retain the generic planner's f64
//! costs, bounds, traversal, guillotine ties and visit budget exactly.

use super::*;

const UNSEEN: u16 = 0;
const AT_LEAST: u16 = 1;
const FALLBACK: u16 = 2;
const FIRST_TILE: u16 = 3;

#[derive(Clone, Copy)]
struct CompactMemo {
    cost: f64,
    choice: u16,
}
const EMPTY_MEMO: CompactMemo = CompactMemo {
    cost: 0.0,
    choice: UNSEEN,
};

#[derive(Default)]
pub(super) struct CompactPlanner {
    memo: Vec<CompactMemo>,
    touched: Vec<u16>,
}

impl CompactPlanner {
    fn reset(&mut self, width: usize, height: usize) {
        // Only clear slots actually used by the previous tile. The index is
        // independent of the previous tile's dimensions once recorded here.
        for i in self.touched.drain(..) {
            self.memo[i as usize].choice = UNSEEN;
        }
        let len = (width * height + 1) << width;
        if self.memo.len() < len {
            self.memo.resize(len, EMPTY_MEMO);
        }
    }

    fn get(&self, state: State, width: usize) -> Option<Memo> {
        let entry = self.memo[state.key(width)];
        match entry.choice {
            UNSEEN => None,
            AT_LEAST => Some(Memo::AtLeast(entry.cost)),
            choice => Some(Memo::Solved(Entry {
                cost: entry.cost,
                choice: if choice == FALLBACK {
                    Choice::Fallback
                } else {
                    Choice::Tile((choice - FIRST_TILE) as usize)
                },
            })),
        }
    }

    fn insert(&mut self, state: State, width: usize, value: Memo) {
        let i = state.key(width);
        if self.memo[i].choice == UNSEEN {
            self.touched.push(i as u16);
        }
        self.memo[i] = match value {
            Memo::AtLeast(cost) => CompactMemo {
                cost,
                choice: AT_LEAST,
            },
            Memo::Solved(entry) => CompactMemo {
                cost: entry.cost,
                choice: match entry.choice {
                    Choice::Fallback => FALLBACK,
                    Choice::Tile(k) => FIRST_TILE + k as u16,
                    _ => unreachable!("only finite nonterminal suffixes are memoized"),
                },
            },
        };
    }
}

/// At the first open cell, `filled` marks the following w+1 cells. Bit zero
/// is always clear. A small piece reaches at most w+1 cells ahead; advancing
/// once shifts that last bit down to w, so only w bits need to be stored.
#[derive(Clone, Copy)]
struct State {
    cell: usize,
    filled: usize,
}
impl State {
    fn key(self, width: usize) -> usize {
        (self.cell << width) | (self.filled >> 1)
    }

    fn mask(c: Candidate, width: usize) -> usize {
        let row = if c.width == 2 { 3 } else { 1 };
        if c.height == 2 {
            row | row << width
        } else {
            row
        }
    }

    fn advance(self, mask: usize) -> Self {
        let filled = (self.filled | mask) >> 1;
        let skip = filled.trailing_ones() as usize;
        Self {
            cell: self.cell + 1 + skip,
            filled: filled >> skip,
        }
    }
}

fn visit(
    state: State,
    profile: [u8; SIDE],
    width: usize,
    height: usize,
    budget: f64,
    candidates: &[Candidate],
    s: &mut PlannerScratch,
    compact: &mut CompactPlanner,
) -> Option<Entry> {
    match compact.get(state, width) {
        Some(Memo::Solved(best)) => return Some(best),
        Some(Memo::AtLeast(bound)) if bound >= budget => return None,
        _ => {}
    }
    let (x, y) = (state.cell % width, state.cell / width);
    debug_assert_eq!(first_open(profile, width), (x, y));
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
        let mask = State::mask(c, width);
        if !c.cost.is_finite() || state.filled & mask != 0 {
            continue;
        }
        let next = state.advance(mask);
        let next_profile = advance(profile, c);
        let bound = match compact.get(next, width) {
            Some(Memo::Solved(entry)) => entry.cost,
            Some(Memo::AtLeast(bound)) => bound,
            None => {
                let bound = lower_bound(next_profile, width, height, s);
                compact.insert(next, width, Memo::AtLeast(bound));
                bound
            }
        };
        let limit = best.cost.min(budget);
        if add_lower(c.cost, bound) >= limit {
            continue;
        }
        let remaining = -add_lower(-limit, c.cost);
        if let Some(suffix) = visit(
            next,
            next_profile,
            width,
            height,
            remaining,
            candidates,
            s,
            compact,
        ) {
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
        compact.insert(state, width, Memo::Solved(best));
        Some(best)
    } else {
        compact.insert(state, width, Memo::AtLeast(budget));
        None
    }
}

pub(super) fn plan(
    candidates: &[Candidate],
    width: usize,
    height: usize,
    s: &mut PlannerScratch,
    compact: &mut CompactPlanner,
    out: &mut Vec<usize>,
) -> f64 {
    if candidates.iter().any(|c| c.width > 2 || c.height > 2) {
        return super::plan(candidates, width, height, s, out);
    }
    prepare_planner(candidates, width, height, s);
    compact.reset(width, height);
    let mut state = State { cell: 0, filled: 0 };
    let mut profile = [0; SIDE];
    let root = visit(
        state,
        profile,
        width,
        height,
        f64::INFINITY,
        candidates,
        s,
        compact,
    )
    .unwrap_or(EMPTY);
    out.clear();
    if !root.cost.is_finite() {
        return root.cost;
    }
    while state.cell < width * height {
        let Some(Memo::Solved(entry)) = compact.get(state, width) else {
            unreachable!("a winning backpointer must have an exact suffix");
        };
        match entry.choice {
            Choice::Tile(k) => {
                out.push(k);
                state = state.advance(State::mask(candidates[k], width));
                profile = advance(profile, candidates[k]);
            }
            Choice::Fallback => {
                unpack_fallback(profile, width, height, s, out);
                break;
            }
            _ => unreachable!("invalid compact frontier backpointer"),
        }
    }
    root.cost
}

/// Before squares are scored, their bounds need only the 2x2 rectangles
/// tiled by the already scored leaves and pairs. Reproduce prepare_regions'
/// additions and cut order without preparing every larger rectangle.
pub(super) fn prepare_square_bounds(
    candidates: &[Candidate],
    width: usize,
    height: usize,
    s: &mut PlannerScratch,
) {
    // Strict comparison preserves the incumbent on ties, including signed
    // zeros, just as prepare_regions does.
    let improve = |a: f64, b: f64| if b < a { b } else { a };
    let mut leaf = [f64::INFINITY; SIDE * SIDE];
    let mut wide = leaf;
    let mut tall = leaf;
    for &c in candidates {
        if !c.cost.is_finite() {
            continue;
        }
        let slots = match (c.width, c.height) {
            (1, 1) => &mut leaf,
            (2, 1) => &mut wide,
            (1, 2) => &mut tall,
            _ => unreachable!("only leaves and pairs precede squares"),
        };
        let slot = &mut slots[c.y() * SIDE + c.x()];
        if c.cost < *slot {
            *slot = c.cost;
        }
    }
    s.regions.resize(REGION_COUNT, EMPTY);
    for y in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            let i = y * SIDE + x;
            let left = improve(tall[i], leaf[i] + leaf[i + SIDE]);
            let right = improve(tall[i + 1], leaf[i + 1] + leaf[i + SIDE + 1]);
            let top = improve(wide[i], leaf[i] + leaf[i + 1]);
            let bottom = improve(wide[i + SIDE], leaf[i + SIDE] + leaf[i + SIDE + 1]);
            s.regions[region_index(x, y, 2, 2)] = Entry {
                cost: improve(left + right, top + bottom),
                choice: Choice::None,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(w: usize, h: usize, seed: usize) -> Vec<Candidate> {
        let mut out = Vec::new();
        for strategy in COEFFICIENT_STRATEGIES {
            let p = Piece::new(0, 0, strategy);
            if p.width() > w || p.height() > h {
                continue;
            }
            for y in 0..=h - p.height() {
                for x in 0..=w - p.width() {
                    let mut p = Piece::new(x, y, strategy);
                    let n = (p.key() * 1664525 + seed * 1013904223) % 257;
                    let cost = match seed % 6 {
                        // Many exact ties, including both signed zeros.
                        0 if n.is_multiple_of(7) && n % 2 == 1 => -0.0,
                        0 => (n % 7) as f64,
                        1 => (n as f32 * 0.12345) as f64,
                        2 => (n as f64 - 128.0) / 8.0,
                        3 => ((n + 1) as f64) * 2.0f64.powi(n as i32 % 80 - 40),
                        4 if n.is_multiple_of(11) => f64::INFINITY,
                        5 if n.is_multiple_of(13) => f64::NAN,
                        _ => (n + 1) as f64 / 32.0,
                    };
                    if p.width() == 1 && p.height() == 1 && n.is_multiple_of(3) {
                        p.strategy = STRATEGY_AFV0;
                    }
                    out.push(Candidate::new(p, cost));
                }
            }
        }
        out
    }

    #[test]
    fn compact_reconstruction_matches_generic_cost_choices_and_visits() {
        let mut generic = PlannerScratch::default();
        let mut specialized = PlannerScratch::default();
        let mut compact = CompactPlanner::default();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        for w in (1..=8).rev() {
            for h in 1..=8 {
                for seed in 0..12 {
                    let mut c = candidates(w, h, seed);
                    if seed % 2 == 1 {
                        c.reverse(); // candidate order and fine IDs must survive
                    }
                    let a = super::super::plan(&c, w, h, &mut generic, &mut expected);
                    let b = plan(&c, w, h, &mut specialized, &mut compact, &mut actual);
                    assert_eq!(a.to_bits(), b.to_bits(), "{w}x{h} seed {seed}");
                    assert_eq!(expected, actual, "{w}x{h} seed {seed}");
                    assert_eq!(generic.budget, specialized.budget, "{w}x{h} seed {seed}");
                }
            }
        }
    }

    #[test]
    fn compact_reconstruction_preserves_failed_and_exhausted_budgets() {
        let c = candidates(8, 8, 1);
        let mut generic = PlannerScratch::default();
        let mut specialized = PlannerScratch::default();
        let mut compact = CompactPlanner::default();
        for visits in [0, 1, 2, 5, 50, VISIT_BUDGET] {
            for limit in [0.0, 10.0, 100.0, f64::INFINITY] {
                prepare_planner(&c, 8, 8, &mut generic);
                prepare_planner(&c, 8, 8, &mut specialized);
                compact.reset(8, 8);
                generic.budget = visits;
                specialized.budget = visits;
                // Revisit a failed prefix with more budget, as another parent
                // reaching the same suffix would do.
                for budget in [limit, f64::INFINITY] {
                    let a = super::super::visit([0; SIDE], 8, 8, budget, &c, &mut generic);
                    let b = visit(
                        State { cell: 0, filled: 0 },
                        [0; SIDE],
                        8,
                        8,
                        budget,
                        &c,
                        &mut specialized,
                        &mut compact,
                    );
                    assert_eq!(format!("{a:?}"), format!("{b:?}"));
                    assert_eq!(generic.budget, specialized.budget);
                }
            }
        }
    }

    #[test]
    fn larger_incumbents_use_generic_planner() {
        let mut compact = CompactPlanner::default();
        for strategy in [STRATEGY_DCT32X8, STRATEGY_DCT32X32, STRATEGY_DCT64X64] {
            let mut c = candidates(8, 8, 1);
            c.push(Candidate::new(Piece::new(0, 0, strategy), 0.25));
            let mut s = PlannerScratch::default();
            let mut expected = Vec::new();
            let a = super::super::plan(&c, 8, 8, &mut s, &mut expected);
            let mut actual = Vec::new();
            let b = plan(&c, 8, 8, &mut s, &mut compact, &mut actual);
            assert_eq!(a.to_bits(), b.to_bits());
            assert_eq!(expected, actual);
            assert!(compact.memo.is_empty());
        }
    }

    #[test]
    fn square_bounds_match_generic_regions_bit_for_bit() {
        let mut generic = PlannerScratch::default();
        let mut specialized = PlannerScratch::default();
        for w in 1..=8 {
            for h in (1..=8).rev() {
                for seed in 0..12 {
                    let c: Vec<_> = candidates(w, h, seed)
                        .into_iter()
                        .filter(|c| c.width != 2 || c.height != 2)
                        .collect();
                    prepare_regions(&c, w, h, &mut generic);
                    prepare_square_bounds(&c, w, h, &mut specialized);
                    for y in 0..h.saturating_sub(1) {
                        for x in 0..w.saturating_sub(1) {
                            let i = region_index(x, y, 2, 2);
                            assert_eq!(
                                generic.regions[i].cost.to_bits(),
                                specialized.regions[i].cost.to_bits(),
                                "{w}x{h} seed {seed} at {x},{y}"
                            );
                        }
                    }
                }
            }
        }
    }
}
