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

//! Visiting orders for point sets coded as one absolute position followed by
//! signed deltas (spline starts, patch occurrences): an open Hamiltonian path
//! priced by the hybrid-uint tokens it emits.

use crate::entropy::{pack_signed, uint_encode};

const ALPHABET: usize = 128;
const HELD_KARP_MAX: usize = 9;
const NEIGHBORS: usize = 8;
const OR_OPT_SEGMENT: usize = 3;
const MAX_ROUNDS: usize = 8;
const MAX_POINTS: usize = 1 << 16;
const MIN_GAIN: f64 = 1e-3;
const SMALL_VALUES: usize = 4096;

pub(crate) type Pos = (i32, i32);

/// Bits of a value in one context: the token's share of a histogram plus the
/// raw bits below it.
#[derive(Clone)]
pub(crate) struct Prices {
    tokens: [f32; ALPHABET],
    small: Vec<f32>,
}

impl Prices {
    pub(crate) fn learn(values: impl IntoIterator<Item = u32>) -> Self {
        let mut counts = [0u32; ALPHABET];
        let mut total = 0u32;
        let mut top = 0usize;
        for v in values {
            let t = (uint_encode(v).0 as usize).min(ALPHABET - 1);
            counts[t] += 1;
            total += 1;
            top = top.max(t);
        }
        // Half a count on every token up to just past the largest seen; each
        // token beyond that costs a bit more than the last.
        let span = (top + 3).min(ALPHABET);
        let denom = total as f32 + 0.5 * span as f32;
        let mut bits = [0f32; ALPHABET];
        for (t, b) in bits.iter_mut().enumerate() {
            *b = if t < span {
                (denom / (counts[t] as f32 + 0.5)).log2()
            } else {
                (denom / 0.5).log2() + (t + 1 - span) as f32
            };
        }
        let small = (0..SMALL_VALUES as u32)
            .map(|v| Self::encoded_bits(&bits, v))
            .collect();
        Self {
            tokens: bits,
            small,
        }
    }

    fn encoded_bits(tokens: &[f32; ALPHABET], value: u32) -> f32 {
        let (token, nbits, _) = uint_encode(value);
        tokens[(token as usize).min(ALPHABET - 1)] + nbits as f32
    }

    #[inline]
    pub(crate) fn bits(&self, value: u32) -> f32 {
        match self.small.get(value as usize) {
            Some(&bits) => bits,
            None => Self::encoded_bits(&self.tokens, value),
        }
    }
}

/// Prices of the first (absolute) position and of every later delta.
pub(crate) struct PathCost<'a> {
    pub(crate) first: &'a Prices,
    pub(crate) delta: &'a Prices,
}

impl PathCost<'_> {
    #[inline]
    fn first(&self, p: Pos) -> f32 {
        self.first.bits(p.0 as u32) + self.first.bits(p.1 as u32)
    }

    #[inline]
    fn step(&self, a: Pos, b: Pos) -> f32 {
        self.delta.bits(pack_signed(b.0 - a.0)) + self.delta.bits(pack_signed(b.1 - a.1))
    }

    fn of(&self, points: &[Pos], order: &[usize]) -> f64 {
        let Some(&head) = order.first() else {
            return 0.0;
        };
        self.first(points[head]) as f64
            + order
                .array_windows::<2>()
                .map(|&[a, b]| self.step(points[a], points[b]) as f64)
                .sum::<f64>()
    }
}

/// Visiting orders of point sets that share one context for first positions
/// and one for deltas (the same one when `one_context`). Sets not `movable`
/// keep `given`. The start is the given orders or serpentine sweeps, each
/// judged under prices learned from itself; then `PRICE_ROUNDS` times the
/// prices are relearned and every movable set is reordered under them.
pub(crate) fn order_sets(
    sets: &[Vec<Pos>],
    movable: &[bool],
    given: Vec<Vec<usize>>,
    one_context: bool,
) -> Vec<Vec<usize>> {
    const PRICE_ROUNDS: usize = 2;
    let sweeps: Vec<Vec<Vec<usize>>> = sets
        .iter()
        .zip(movable)
        .map(|(points, &m)| {
            if m && points.len() > HELD_KARP_MAX {
                serpentine_sweeps(points)
            } else {
                Vec::new()
            }
        })
        .collect();
    let mut orders = given;
    let mut best = self_priced(sets, &orders, one_context);
    for k in 0..SWEEPS.len() {
        let candidate: Vec<Vec<usize>> = orders
            .iter()
            .zip(&sweeps)
            .map(|(order, sweeps)| sweeps.get(k).unwrap_or(order).clone())
            .collect();
        let bits = self_priced(sets, &candidate, one_context);
        if bits < best {
            (best, orders) = (bits, candidate);
        }
    }
    for _ in 0..PRICE_ROUNDS {
        let (first, delta) = learn(sets, &orders, one_context);
        let cost = PathCost {
            first: &first,
            delta: &delta,
        };
        for (r, points) in sets.iter().enumerate() {
            if movable[r] {
                orders[r] = order(points, &cost, &orders[r]);
            }
        }
    }
    orders
}

fn learn(sets: &[Vec<Pos>], orders: &[Vec<usize>], one_context: bool) -> (Prices, Prices) {
    let firsts = || {
        orders
            .iter()
            .zip(sets)
            .filter(|(o, _)| !o.is_empty())
            .flat_map(|(o, p)| {
                let (x, y) = p[o[0]];
                [x as u32, y as u32]
            })
    };
    let deltas = || {
        orders.iter().zip(sets).flat_map(|(o, p)| {
            o.array_windows::<2>().flat_map(move |&[a, b]| {
                [pack_signed(p[b].0 - p[a].0), pack_signed(p[b].1 - p[a].1)]
            })
        })
    };
    if one_context {
        let prices = Prices::learn(firsts().chain(deltas()));
        (prices.clone(), prices)
    } else {
        (Prices::learn(firsts()), Prices::learn(deltas()))
    }
}

/// Bits of `orders` under prices learned from themselves.
fn self_priced(sets: &[Vec<Pos>], orders: &[Vec<usize>], one_context: bool) -> f64 {
    let (first, delta) = learn(sets, orders, one_context);
    let cost = PathCost {
        first: &first,
        delta: &delta,
    };
    sets.iter().zip(orders).map(|(p, o)| cost.of(p, o)).sum()
}

/// Cheapest open path through `points` under `cost`: exact for small sets,
/// otherwise 2-opt and Or-opt from `initial` over spatial neighbors.
fn order(points: &[Pos], cost: &PathCost, initial: &[usize]) -> Vec<usize> {
    debug_assert_eq!(points.len(), initial.len());
    if points.len() <= 1 {
        initial.to_vec()
    } else if points.len() <= HELD_KARP_MAX {
        held_karp(points, cost)
    } else if points.len() <= MAX_POINTS {
        LocalSearch::new(points, cost, initial.to_vec()).run()
    } else {
        initial.to_vec()
    }
}

/// Band heights of the serpentine sweeps, in mean point spacings.
static SWEEPS: [f64; 3] = [1.0, 2.0, 4.0];

/// Boustrophedon orders over horizontal bands: scattered points sorted by
/// row alone zigzag across the whole width, which local moves cannot undo.
fn serpentine_sweeps(points: &[Pos]) -> Vec<Vec<usize>> {
    let (min_x, max_x) = points
        .iter()
        .fold((i32::MAX, i32::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (min_y, max_y) = points
        .iter()
        .fold((i32::MAX, i32::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let area = (max_x - min_x + 1) as f64 * (max_y - min_y + 1) as f64;
    let spacing = (area / points.len() as f64).sqrt().max(1.0);
    SWEEPS
        .iter()
        .map(|&k| {
            let band = (spacing * k).round().max(1.0) as i32;
            let mut path: Vec<usize> = (0..points.len()).collect();
            path.sort_unstable_by_key(|&i| {
                let (x, y) = points[i];
                let row = (y - min_y) / band;
                (row, if row & 1 == 0 { x } else { -x }, y)
            });
            path
        })
        .collect()
}

fn held_karp(points: &[Pos], cost: &PathCost) -> Vec<usize> {
    let n = points.len();
    let full = (1usize << n) - 1;
    let step: Vec<f32> = (0..n * n)
        .map(|k| cost.step(points[k / n], points[k % n]))
        .collect();
    let mut best = vec![f32::INFINITY; (full + 1) * n];
    let mut parent = vec![u8::MAX; (full + 1) * n];
    for i in 0..n {
        best[(1 << i) * n + i] = cost.first(points[i]);
    }
    for mask in 1..=full {
        for last in 0..n {
            let here = best[mask * n + last];
            if mask & (1 << last) == 0 || here == f32::INFINITY {
                continue;
            }
            for next in 0..n {
                if mask & (1 << next) != 0 {
                    continue;
                }
                let to = (mask | (1 << next)) * n + next;
                let c = here + step[last * n + next];
                if c < best[to] {
                    best[to] = c;
                    parent[to] = last as u8;
                }
            }
        }
    }
    let mut last = (0..n)
        .min_by(|&a, &b| best[full * n + a].total_cmp(&best[full * n + b]))
        .unwrap();
    let mut mask = full;
    let mut path = Vec::with_capacity(n);
    loop {
        path.push(last);
        let p = parent[mask * n + last];
        mask &= !(1 << last);
        if p == u8::MAX {
            break;
        }
        last = p as usize;
    }
    path.reverse();
    path
}

/// Up to `NEIGHBORS` nearest points (Chebyshev) of each point plus its
/// neighbors in row-major and column-major order, which keep long runs along
/// one row or column reachable.
fn neighbor_lists(points: &[Pos]) -> (Vec<usize>, Vec<usize>) {
    let n = points.len();
    let (min_x, max_x) = points
        .iter()
        .fold((i32::MAX, i32::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (min_y, max_y) = points
        .iter()
        .fold((i32::MAX, i32::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let (w, h) = ((max_x - min_x + 1) as i64, (max_y - min_y + 1) as i64);
    let cell = (((w * h) as f64 / n as f64).sqrt().ceil() as i64).max(1);
    let (gw, gh) = (
        ((w + cell - 1) / cell) as usize,
        ((h + cell - 1) / cell) as usize,
    );
    let cell_of = |p: Pos| {
        (
            ((p.0 - min_x) as i64 / cell) as usize,
            ((p.1 - min_y) as i64 / cell) as usize,
        )
    };
    let mut start = vec![0usize; gw * gh + 1];
    for &p in points {
        let (cx, cy) = cell_of(p);
        start[cy * gw + cx + 1] += 1;
    }
    for k in 0..gw * gh {
        start[k + 1] += start[k];
    }
    let mut fill = start.clone();
    let mut members = vec![0usize; n];
    for (i, &p) in points.iter().enumerate() {
        let (cx, cy) = cell_of(p);
        members[fill[cy * gw + cx]] = i;
        fill[cy * gw + cx] += 1;
    }

    let mut by_row: Vec<usize> = (0..n).collect();
    by_row.sort_unstable_by_key(|&i| (points[i].1, points[i].0));
    let mut by_col: Vec<usize> = (0..n).collect();
    by_col.sort_unstable_by_key(|&i| (points[i].0, points[i].1));
    let mut extra = vec![Vec::new(); n];
    for order in [&by_row, &by_col] {
        for k in 0..n {
            for d in 1..=2 {
                if k >= d {
                    extra[order[k]].push(order[k - d]);
                }
                if k + d < n {
                    extra[order[k]].push(order[k + d]);
                }
            }
        }
    }

    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0);
    let mut lists = Vec::with_capacity(n * (NEIGHBORS + 4));
    let mut near: Vec<(i32, usize)> = Vec::with_capacity(NEIGHBORS + 1);
    for (i, &p) in points.iter().enumerate() {
        near.clear();
        let (cx, cy) = cell_of(p);
        for ring in 0..gw.max(gh) {
            // Points in ring r are at least (r - 1) * cell + 1 away.
            if near.len() == NEIGHBORS && (ring as i64 - 1) * cell >= near[NEIGHBORS - 1].0 as i64 {
                break;
            }
            let (x0, x1) = (cx.saturating_sub(ring), (cx + ring).min(gw - 1));
            let (y0, y1) = (cy.saturating_sub(ring), (cy + ring).min(gh - 1));
            for gy in y0..=y1 {
                for gx in x0..=x1 {
                    if gx.abs_diff(cx).max(gy.abs_diff(cy)) != ring {
                        continue;
                    }
                    for &j in &members[start[gy * gw + gx]..start[gy * gw + gx + 1]] {
                        if j == i {
                            continue;
                        }
                        let d = (points[j].0 - p.0).abs().max((points[j].1 - p.1).abs());
                        if near.len() < NEIGHBORS || d < near[NEIGHBORS - 1].0 {
                            let at = near.partition_point(|&(e, _)| e <= d);
                            near.insert(at, (d, j));
                            near.truncate(NEIGHBORS);
                        }
                    }
                }
            }
        }
        let begin = lists.len();
        lists.extend(near.iter().map(|&(_, j)| j));
        for &j in &extra[i] {
            if !lists[begin..].contains(&j) {
                lists.push(j);
            }
        }
        offsets.push(lists.len());
    }
    (offsets, lists)
}

struct LocalSearch<'a> {
    points: &'a [Pos],
    cost: &'a PathCost<'a>,
    path: Vec<usize>,
    pos: Vec<usize>,
    /// `forward[k]`: cost of the steps path[0] -> ... -> path[k].
    forward: Vec<f64>,
    /// `backward[k]`: the same steps walked the other way.
    backward: Vec<f64>,
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
    /// Points next to a move of this round or the last; the others had no
    /// improving move and are skipped (don't-look bits).
    looking: Vec<bool>,
    touched: Vec<bool>,
}

impl<'a> LocalSearch<'a> {
    fn new(points: &'a [Pos], cost: &'a PathCost<'a>, path: Vec<usize>) -> Self {
        let n = points.len();
        let (offsets, neighbors) = neighbor_lists(points);
        let mut search = Self {
            points,
            cost,
            path,
            pos: vec![0; n],
            forward: vec![0.0; n],
            backward: vec![0.0; n],
            offsets,
            neighbors,
            looking: vec![true; n],
            touched: vec![false; n],
        };
        search.reindex(0, n);
        search
    }

    /// Refreshes `pos` and the step sums after path[lo..hi] changed: steps up
    /// to path[hi] are repriced, the later sums only shift.
    fn reindex(&mut self, lo: usize, hi: usize) {
        for k in lo..hi {
            self.pos[self.path[k]] = k;
        }
        let n = self.path.len();
        let stop = hi.min(n - 1);
        let (forward, backward) = (self.forward[stop], self.backward[stop]);
        for k in lo.max(1)..=stop {
            let (a, b) = (self.points[self.path[k - 1]], self.points[self.path[k]]);
            self.forward[k] = self.forward[k - 1] + self.cost.step(a, b) as f64;
            self.backward[k] = self.backward[k - 1] + self.cost.step(b, a) as f64;
        }
        let (df, db) = (self.forward[stop] - forward, self.backward[stop] - backward);
        for k in stop + 1..n {
            self.forward[k] += df;
            self.backward[k] += db;
        }
    }

    /// Cost of joining `a` to `b`: the absolute position when nothing comes
    /// before `b`, nothing when nothing follows `a`.
    #[inline]
    fn edge(&self, a: Option<usize>, b: Option<usize>) -> f64 {
        match (a, b) {
            (Some(a), Some(b)) => self.cost.step(self.points[a], self.points[b]) as f64,
            (None, Some(b)) => self.cost.first(self.points[b]) as f64,
            _ => 0.0,
        }
    }

    fn neighbors(&self, i: usize) -> &[usize] {
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
    }

    fn looks_at(&self, i: usize) -> bool {
        self.looking[i] || self.touched[i]
    }

    fn touch(&mut self, points: impl IntoIterator<Item = Option<usize>>) {
        for i in points.into_iter().flatten() {
            self.touched[i] = true;
        }
    }

    fn run(mut self) -> Vec<usize> {
        for _ in 0..MAX_ROUNDS {
            let reversed = self.two_opt();
            let moved = self.or_opt();
            if !reversed && !moved {
                break;
            }
            std::mem::swap(&mut self.looking, &mut self.touched);
            self.touched.fill(false);
        }
        self.path
    }

    /// Reversing [i..=j] joins i-1 with j, i with j+1 and walks the inside
    /// backwards.
    fn two_opt(&mut self) -> bool {
        let n = self.path.len();
        let mut improved = false;
        let mut candidates = Vec::new();
        for i in 0..n - 1 {
            if !self.looks_at(self.path[i]) && (i == 0 || !self.looks_at(self.path[i - 1])) {
                continue;
            }
            candidates.clear();
            candidates.push(n - 1);
            if i > 0 {
                candidates.extend(
                    self.neighbors(self.path[i - 1])
                        .iter()
                        .map(|&c| self.pos[c])
                        .filter(|&j| j > i),
                );
            }
            candidates.extend(
                self.neighbors(self.path[i])
                    .iter()
                    .map(|&c| self.pos[c])
                    .filter(|&k| k > i + 1)
                    .map(|k| k - 1),
            );
            let prev = i.checked_sub(1).map(|k| self.path[k]);
            let mut best = (MIN_GAIN, 0usize);
            for &j in &candidates {
                let next = self.path.get(j + 1).copied();
                let (head, tail) = (self.path[i], self.path[j]);
                let before = self.edge(prev, Some(head))
                    + (self.forward[j] - self.forward[i])
                    + self.edge(Some(tail), next);
                let after = self.edge(prev, Some(tail))
                    + (self.backward[j] - self.backward[i])
                    + self.edge(Some(head), next);
                if before - after > best.0 {
                    best = (before - after, j);
                }
            }
            if best.0 > MIN_GAIN {
                let j = best.1;
                self.touch([
                    prev,
                    Some(self.path[i]),
                    Some(self.path[j]),
                    self.path.get(j + 1).copied(),
                ]);
                self.path[i..=j].reverse();
                self.reindex(i, j + 1);
                improved = true;
            }
        }
        improved
    }

    /// Moves [i, i + len) between two points of the remaining path, next to a
    /// neighbor of either end or at either end of the path, keeping or
    /// reversing its direction.
    fn or_opt(&mut self) -> bool {
        let n = self.path.len();
        let mut improved = false;
        for len in 1..=OR_OPT_SEGMENT.min(n - 1) {
            let mut i = 0;
            while i + len <= n {
                let end = i + len;
                let (first, last) = (self.path[i], self.path[end - 1]);
                if !self.looks_at(first) && !self.looks_at(last) {
                    i += 1;
                    continue;
                }
                let prev = i.checked_sub(1).map(|k| self.path[k]);
                let next = self.path.get(end).copied();
                let inside = [
                    self.forward[end - 1] - self.forward[i],
                    self.backward[end - 1] - self.backward[i],
                ];
                let removed =
                    self.edge(prev, Some(first)) + inside[0] + self.edge(Some(last), next)
                        - self.edge(prev, next);
                let reduced = |k: usize| if k < i { k } else { k - len };
                let left_of = |k: usize| -> Option<usize> {
                    let k = if k == end { i } else { k };
                    k.checked_sub(1).map(|k| self.path[k])
                };
                let right_of = |k: usize| -> Option<usize> {
                    let k = if k + 1 == i { end } else { k + 1 };
                    self.path.get(k).copied()
                };
                // (gain, insertion index in the reduced path, reversed, left, right)
                let mut best = (MIN_GAIN, 0usize, false, None, None);
                let mut consider = |left: Option<usize>, right: Option<usize>, at: usize| {
                    let cut = self.edge(left, right);
                    for reversed in [false, true] {
                        if reversed && len == 1 {
                            break;
                        }
                        let (head, tail) = if reversed {
                            (last, first)
                        } else {
                            (first, last)
                        };
                        let added = self.edge(left, Some(head))
                            + inside[reversed as usize]
                            + self.edge(Some(tail), right)
                            - cut;
                        if removed - added > best.0 {
                            best = (removed - added, at, reversed, left, right);
                        }
                    }
                };
                let front = if i == 0 { next } else { Some(self.path[0]) };
                let back = if end == n {
                    prev
                } else {
                    Some(self.path[n - 1])
                };
                consider(None, front, 0);
                consider(back, None, n - len);
                let ends: &[usize] = if len == 1 {
                    &self.neighbors(first)[..0]
                } else {
                    self.neighbors(last)
                };
                for &c in self.neighbors(first).iter().chain(ends) {
                    let k = self.pos[c];
                    if (i..end).contains(&k) {
                        continue;
                    }
                    consider(Some(c), right_of(k), reduced(k) + 1);
                    consider(left_of(k), Some(c), reduced(k));
                }
                if best.0 > MIN_GAIN {
                    let (_, at, reversed, left, right) = best;
                    self.touch([prev, next, Some(first), Some(last), left, right]);
                    let mut segment: Vec<usize> = self.path.drain(i..end).collect();
                    if reversed {
                        segment.reverse();
                    }
                    self.path.splice(at..at, segment);
                    self.reindex(i.min(at), end.max(at + len));
                    improved = true;
                }
                i += 1;
            }
        }
        improved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Prices {
        Prices::learn(0..64)
    }

    fn brute_force(points: &[Pos], cost: &PathCost) -> f64 {
        fn go(points: &[Pos], cost: &PathCost, path: &mut Vec<usize>, best: &mut f64) {
            if path.len() == points.len() {
                *best = best.min(cost.of(points, path));
                return;
            }
            for i in 0..points.len() {
                if !path.contains(&i) {
                    path.push(i);
                    go(points, cost, path, best);
                    path.pop();
                }
            }
        }
        let mut best = f64::INFINITY;
        go(points, cost, &mut Vec::new(), &mut best);
        best
    }

    fn scatter(n: usize, seed: u32) -> Vec<Pos> {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        (0..n)
            .map(|_| ((next() % 400) as i32, (next() % 300) as i32))
            .collect()
    }

    #[test]
    fn held_karp_matches_brute_force() {
        let prices = flat();
        let cost = PathCost {
            first: &prices,
            delta: &prices,
        };
        for seed in 1..6 {
            let points = scatter(7, seed);
            let path = held_karp(&points, &cost);
            let mut sorted = path.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..7).collect::<Vec<_>>());
            assert!((cost.of(&points, &path) - brute_force(&points, &cost)).abs() < 1e-3);
        }
    }

    #[test]
    fn local_search_never_worsens_and_stays_a_permutation() {
        let prices = flat();
        let cost = PathCost {
            first: &prices,
            delta: &prices,
        };
        for seed in 1..4 {
            let points = scatter(300, seed);
            let initial: Vec<usize> = (0..points.len()).collect();
            let path = order(&points, &cost, &initial);
            let mut sorted = path.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, initial);
            assert!(cost.of(&points, &path) < cost.of(&points, &initial));
        }
    }

    #[test]
    fn rows_are_walked_back_and_forth() {
        // Raster order pays a long jump back at every row end.
        let points: Vec<Pos> = (0..6)
            .flat_map(|y| (0..20).map(move |x| (x * 9, y * 14)))
            .collect();
        let initial: Vec<usize> = (0..points.len()).collect();
        let deltas = initial
            .array_windows::<2>()
            .flat_map(|&[a, b]| {
                [
                    pack_signed(points[b].0 - points[a].0),
                    pack_signed(points[b].1 - points[a].1),
                ]
            })
            .collect::<Vec<_>>();
        let prices = Prices::learn(deltas);
        let cost = PathCost {
            first: &prices,
            delta: &prices,
        };
        let path = order(&points, &cost, &initial);
        assert!(cost.of(&points, &path) < cost.of(&points, &initial) - 10.0);
    }
}
