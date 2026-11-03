/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! Entropy-driven JPEG block context map (after libjxl PR 4719): DC
//! thresholds on every plane placed by an interval DP over the coded AC
//! statistics, then the (channel, cell) contexts merged down to the spec's 16.

use super::{
    BlockCtx, ChannelLayout, DCT_BLOCK_SIZE, Dim, JpegCfl, JpegData, for_each_coeff,
    visit_group_blocks,
};
use crate::ac_context::{K_NON_ZERO_BUCKETS, K_ZERO_DENSITY_CONTEXT_COUNT, non_zero_bucket};
use crate::coder_scratch::CoderScratch;
use crate::entropy::{pack_signed, uint_encode};
use crate::thread_pool::ThreadPool;

/// Default HybridUint tokens per zero-density context; larger ones saturate.
const TOKENS: u32 = 64;
/// Symbols are `zdc * TOKENS + token` for coefficients, then
/// `NZ_BASE + bucket * 64 + nzeros` for the non-zero counts.
const NZ_BASE: u32 = K_ZERO_DENSITY_CONTEXT_COUNT as u32 * TOKENS;
const NUM_SYMS: usize = NZ_BASE as usize + K_NON_ZERO_BUCKETS * 64;
/// A symbol's histogram (zero-density context or non-zero bucket) is `sym >> 6`.
const NUM_SYM_GROUPS: usize = NUM_SYMS >> 6;
const SYM_BITS: u32 = 15;
const _: () = assert!(NUM_SYMS <= 1 << SYM_BITS);

/// Fine DC buckets per plane (X, Y, B); thresholds fall between them.
const FINE: [usize; 3] = [16, 32, 16];
const MAX_INTERVALS: usize = 16;
const MAX_CELLS: usize = 64;
const MAX_CTXS: usize = 16;
/// Interval counts (Y, X, B) tried; each is optimized, clustered and priced.
/// Between them they come within 0.1 % (model bits) of the best of ten
/// factorizations on the photo originals.
const CANDIDATES: [[usize; 3]; 2] = [[8, 8, 1], [16, 1, 4]];
/// Contexts left when clustering starts pricing non-neighboring pairs.
const NEIGHBOR_PHASE_CTXS: usize = 32;

/// `n * log2(n)`, tabulated for small counts.
struct NLogN(Vec<f64>);

impl NLogN {
    fn new() -> Self {
        Self(
            (0..1u32 << 16)
                .map(|n| {
                    if n == 0 {
                        0.0
                    } else {
                        n as f64 * (n as f64).log2()
                    }
                })
                .collect(),
        )
    }

    #[inline]
    fn get(&self, n: u32) -> f64 {
        match self.0.get(n as usize) {
            Some(&v) => v,
            None => n as f64 * (n as f64).log2(),
        }
    }
}

/// Equal-population DC buckets of one plane.
struct Axis {
    /// Smallest DC value of each bucket.
    lo: Vec<i32>,
    /// DC value + 32768 to bucket.
    lut: Vec<u8>,
}

impl Axis {
    fn new(counts: &[u32], cap: usize) -> Self {
        let total: u64 = counts.iter().map(|&n| u64::from(n)).sum();
        let mut lo = Vec::new();
        let mut lut = vec![0u8; counts.len()];
        let mut cum = 0u64;
        let mut bucket = 0usize;
        for (i, &n) in counts.iter().enumerate() {
            if n != 0 {
                if lo.is_empty() {
                    lo.push(i as i32 - 32768);
                } else if bucket + 1 < cap && cum * cap as u64 >= total * (bucket as u64 + 1) {
                    bucket += 1;
                    lo.push(i as i32 - 32768);
                }
                cum += u64::from(n);
            }
            lut[i] = bucket as u8;
        }
        if lo.is_empty() {
            lo.push(0);
        }
        Self { lo, lut }
    }

    fn len(&self) -> usize {
        self.lo.len()
    }

    #[inline]
    fn bucket(&self, dc: i32) -> usize {
        self.lut[(dc + 32768).clamp(0, 65535) as usize] as usize
    }
}

/// Sparse coded-symbol counts of every (channel, fine DC cell) key.
struct Stats {
    axes: [Axis; 3],
    /// `key_start[k]..key_start[k + 1]` indexes `syms`/`counts`.
    key_start: Vec<u32>,
    syms: Vec<u16>,
    counts: Vec<u32>,
}

impl Stats {
    fn num_keys(&self) -> usize {
        self.key_start.len() - 1
    }

    #[inline]
    fn key(&self, c: usize, f: [usize; 3]) -> usize {
        ((c * self.axes[1].len() + f[1]) * self.axes[0].len() + f[0]) * self.axes[2].len() + f[2]
    }

    #[inline]
    fn unkey(&self, key: usize) -> (usize, [usize; 3]) {
        let n = [self.axes[0].len(), self.axes[1].len(), self.axes[2].len()];
        let f2 = key % n[2];
        let rest = key / n[2];
        let f0 = rest % n[0];
        let rest = rest / n[0];
        (rest / n[1], [f0, rest % n[1], f2])
    }

    #[inline]
    fn entries(&self, key: usize) -> std::ops::Range<usize> {
        self.key_start[key] as usize..self.key_start[key + 1] as usize
    }
}

fn collect_stats(
    jpg: &JpegData,
    dim: &Dim,
    ss: &ChannelLayout,
    cfl: Option<&JpegCfl>,
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Stats {
    let axes: [Axis; 3] = std::array::from_fn(|p| {
        let mut counts = vec![0u32; 1 << 16];
        if let Some(comp) = ss.component(jpg, p) {
            for by in 0..dim.ysize_blocks >> ss.vshift[p] {
                for bx in 0..dim.xsize_blocks >> ss.hshift[p] {
                    let dc = i32::from(
                        comp.coeffs[(by * comp.width_in_blocks + bx) * DCT_BLOCK_SIZE]
                            + ss.dc_offset[p],
                    );
                    counts[(dc + 32768).clamp(0, 65535) as usize] += 1;
                }
            }
        }
        Axis::new(&counts, FINE[p])
    });
    let mut stats = Stats {
        axes,
        key_start: Vec::new(),
        syms: Vec::new(),
        counts: Vec::new(),
    };
    let num_keys = 3 * stats.axes.iter().map(Axis::len).product::<usize>();
    let xsize_groups = dim.xsize_groups;
    let stats_ref = &stats;
    let per_group: Vec<Vec<(u32, u32)>> = pool.steal_map(scratch, dim.num_groups, |g, _| {
        let mut raw: Vec<u32> = Vec::new();
        let (gx, gy) = (g % xsize_groups, g / xsize_groups);
        visit_group_blocks(jpg, dim, ss, cfl, gx, gy, |b| {
            let f = std::array::from_fn(|p| stats_ref.axes[p].bucket(b.dc[p]));
            let base = (stats_ref.key(b.c, f) as u32) << SYM_BITS;
            raw.push(base | (NZ_BASE + non_zero_bucket(b.predicted) * 64 + b.nzeros));
            for_each_coeff(&scan[b.c], b.block, b.nzeros, |zdc, coef| {
                let token = uint_encode(pack_signed(coef)).0.min(TOKENS - 1);
                raw.push(base | (zdc as u32 * TOKENS + token));
            });
        });
        raw.sort_unstable();
        run_lengths(raw.into_iter().map(|v| (v, 1)))
    });
    // Counting sort by key; each key's few runs then merge by symbol.
    let mut key_start = vec![0u32; num_keys + 1];
    for &(packed, _) in per_group.iter().flatten() {
        key_start[(packed >> SYM_BITS) as usize + 1] += 1;
    }
    for k in 0..num_keys {
        key_start[k + 1] += key_start[k];
    }
    let mut fill = key_start.clone();
    let mut by_key = vec![(0u16, 0u32); key_start[num_keys] as usize];
    for &(packed, n) in per_group.iter().flatten() {
        let slot = &mut fill[(packed >> SYM_BITS) as usize];
        by_key[*slot as usize] = ((packed & ((1 << SYM_BITS) - 1)) as u16, n);
        *slot += 1;
    }
    drop(per_group);
    stats.key_start = Vec::with_capacity(num_keys + 1);
    stats.key_start.push(0);
    for k in 0..num_keys {
        let runs = &mut by_key[key_start[k] as usize..key_start[k + 1] as usize];
        runs.sort_unstable_by_key(|&(s, _)| s);
        let first = stats.syms.len();
        for &(sym, n) in runs.iter() {
            if stats.syms.len() > first && stats.syms.last() == Some(&sym) {
                *stats.counts.last_mut().unwrap() += n;
            } else {
                stats.syms.push(sym);
                stats.counts.push(n);
            }
        }
        stats.key_start.push(stats.syms.len() as u32);
    }
    stats
}

/// Sums the counts of equal adjacent values of a sorted stream.
fn run_lengths(sorted: impl Iterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (v, n) in sorted {
        match out.last_mut() {
            Some((last, count)) if *last == v => *count += n,
            _ => out.push((v, n)),
        }
    }
    out
}

/// Dense accumulator pricing a growing histogram set: the cost is the coded
/// bits `Σ_g N_g log N_g − Σ_s h_s log h_s` over symbol groups `g`.
struct Accum {
    h: Vec<u32>,
    n: Vec<u32>,
    touched_h: Vec<u16>,
    touched_n: Vec<u16>,
    cost: f64,
}

impl Accum {
    fn new() -> Self {
        Self {
            h: vec![0; NUM_SYMS],
            n: vec![0; NUM_SYM_GROUPS],
            touched_h: Vec::new(),
            touched_n: Vec::new(),
            cost: 0.0,
        }
    }

    #[inline]
    fn add(&mut self, f: &NLogN, sym: u16, count: u32) {
        let s = sym as usize;
        let h = self.h[s];
        if h == 0 {
            self.touched_h.push(sym);
        }
        self.h[s] = h + count;
        self.cost -= f.get(h + count) - f.get(h);
        let g = s >> 6;
        let n = self.n[g];
        if n == 0 {
            self.touched_n.push(g as u16);
        }
        self.n[g] = n + count;
        self.cost += f.get(n + count) - f.get(n);
    }

    fn clear(&mut self) {
        for &s in &self.touched_h {
            self.h[s as usize] = 0;
        }
        for &g in &self.touched_n {
            self.n[g as usize] = 0;
        }
        self.touched_h.clear();
        self.touched_n.clear();
        self.cost = 0.0;
    }

    /// Sparse sorted copy of the current histogram set.
    fn take_sparse(&mut self) -> Hist {
        self.touched_h.sort_unstable();
        let entries = self
            .touched_h
            .iter()
            .map(|&s| (s, self.h[s as usize]))
            .collect();
        let cost = self.cost;
        self.clear();
        Hist { entries, cost }
    }
}

/// Interval index of every fine bucket of each plane.
fn intervals(stats: &Stats, starts: &[Vec<usize>; 3]) -> [Vec<usize>; 3] {
    std::array::from_fn(|p| {
        (0..stats.axes[p].len())
            .map(|f| starts[p].partition_point(|&s| s <= f))
            .collect()
    })
}

/// Re-places the thresholds of plane `axis` exactly (interval DP) with the
/// other planes fixed, pricing every (channel, cell) as its own context.
fn sweep_axis(
    stats: &Stats,
    starts: &mut [Vec<usize>; 3],
    axis: usize,
    k: usize,
    f: &NLogN,
    acc: &mut Accum,
) {
    let b = stats.axes[axis].len();
    let k = k.min(b);
    let iv = intervals(stats, starts);
    let (o1, o2) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let n2 = starts[o2].len() + 1;
    let n_other = (starts[o1].len() + 1) * n2;
    // [channel * n_other + other cell][fine bucket on `axis`] -> keys
    let mut groups: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); b]; 3 * n_other];
    for key in 0..stats.num_keys() {
        if stats.entries(key).is_empty() {
            continue;
        }
        let (c, fb) = stats.unkey(key);
        let ci = iv[o1][fb[o1]] * n2 + iv[o2][fb[o2]];
        groups[c * n_other + ci][fb[axis]].push(key as u32);
    }
    // cost[l * b + n]: all groups' bits with buckets l..=n in one interval.
    let mut cost = vec![0f64; b * b];
    for group in &groups {
        let Some(last) = group.iter().rposition(|keys| !keys.is_empty()) else {
            continue;
        };
        // Keys sharing a bucket always travel together: merge them once.
        let buckets: Vec<Vec<(u16, u32)>> = group[..=last]
            .iter()
            .map(|keys| {
                for &key in keys {
                    for e in stats.entries(key as usize) {
                        acc.add(f, stats.syms[e], stats.counts[e]);
                    }
                }
                acc.take_sparse().entries
            })
            .collect();
        for l in 0..=last {
            for n in l..b {
                if let Some(entries) = buckets.get(n) {
                    for &(s, count) in entries {
                        acc.add(f, s, count);
                    }
                }
                cost[l * b + n] += acc.cost;
            }
            acc.clear();
        }
    }
    // dp[j][n]: best cost of buckets 0..=n in j + 1 intervals.
    let mut dp = vec![vec![f64::INFINITY; b]; k];
    let mut back = vec![vec![0usize; b]; k];
    dp[0].copy_from_slice(&cost[..b]);
    for j in 1..k {
        for n in j..b {
            for l in j..=n {
                let v = dp[j - 1][l - 1] + cost[l * b + n];
                if v < dp[j][n] {
                    dp[j][n] = v;
                    back[j][n] = l;
                }
            }
        }
    }
    let mut new_starts = Vec::with_capacity(k - 1);
    let mut n = b - 1;
    for j in (1..k).rev() {
        let l = back[j][n];
        new_starts.push(l);
        n = l - 1;
    }
    new_starts.reverse();
    starts[axis] = new_starts;
}

/// Sparse histogram set of one context, sorted by symbol.
#[derive(Clone)]
struct Hist {
    entries: Vec<(u16, u32)>,
    cost: f64,
}

/// Bits of the union of two histogram sets.
fn merged_cost(a: &Hist, b: &Hist, f: &NLogN) -> f64 {
    let (mut i, mut j) = (0, 0);
    let mut cost = 0.0;
    let mut group = usize::MAX;
    let mut n = 0u32;
    loop {
        let next = match (a.entries.get(i), b.entries.get(j)) {
            (Some(&(sa, ca)), Some(&(sb, cb))) => {
                if sa == sb {
                    i += 1;
                    j += 1;
                    Some((sa, ca + cb))
                } else if sa < sb {
                    i += 1;
                    Some((sa, ca))
                } else {
                    j += 1;
                    Some((sb, cb))
                }
            }
            (Some(&e), None) => {
                i += 1;
                Some(e)
            }
            (None, Some(&e)) => {
                j += 1;
                Some(e)
            }
            (None, None) => None,
        };
        let Some((s, h)) = next else { break };
        let g = s as usize >> 6;
        if g != group {
            cost += f.get(n);
            group = g;
            n = 0;
        }
        n += h;
        cost -= f.get(h);
    }
    cost + f.get(n)
}

fn merge(a: &Hist, b: &Hist, f: &NLogN) -> Hist {
    let mut entries = Vec::with_capacity(a.entries.len() + b.entries.len());
    let (mut i, mut j) = (0, 0);
    while i < a.entries.len() || j < b.entries.len() {
        match (a.entries.get(i), b.entries.get(j)) {
            (Some(&(sa, ca)), Some(&(sb, cb))) if sa == sb => {
                entries.push((sa, ca + cb));
                i += 1;
                j += 1;
            }
            (Some(&(sa, ca)), Some(&(sb, _))) if sa < sb => {
                entries.push((sa, ca));
                i += 1;
            }
            (Some(&e), None) => {
                entries.push(e);
                i += 1;
            }
            (_, Some(&e)) => {
                entries.push(e);
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    Hist {
        cost: merged_cost(a, b, f),
        entries,
    }
}

/// One optimized layout: interval starts per plane and the context of every
/// (channel, cell), `None` where the cell codes no block of that channel.
struct Layout {
    starts: [Vec<usize>; 3],
    assign: Vec<Option<u8>>,
    cost: f64,
}

/// Greedy agglomerative clustering of the (channel, cell) contexts.
fn cluster(stats: &Stats, starts: &[Vec<usize>; 3], f: &NLogN, acc: &mut Accum) -> Layout {
    let iv = intervals(stats, starts);
    let n = [
        starts[0].len() + 1,
        starts[1].len() + 1,
        starts[2].len() + 1,
    ];
    let num_cells = n[0] * n[1] * n[2];
    let mut ctx_keys: Vec<Vec<u32>> = vec![Vec::new(); 3 * num_cells];
    for key in 0..stats.num_keys() {
        if stats.entries(key).is_empty() {
            continue;
        }
        let (c, fb) = stats.unkey(key);
        let cell = (iv[0][fb[0]] * n[2] + iv[2][fb[2]]) * n[1] + iv[1][fb[1]];
        ctx_keys[c * num_cells + cell].push(key as u32);
    }
    let mut ctx_ids = Vec::new();
    let mut hists: Vec<Option<Hist>> = Vec::new();
    for (ctx, keys) in ctx_keys.iter().enumerate() {
        if keys.is_empty() {
            continue;
        }
        for &key in keys {
            for e in stats.entries(key as usize) {
                acc.add(f, stats.syms[e], stats.counts[e]);
            }
        }
        ctx_ids.push(ctx);
        hists.push(Some(acc.take_sparse()));
    }
    let m = hists.len();
    let mut owner: Vec<usize> = (0..m).collect();
    let pair = |hists: &[Option<Hist>], i: usize, j: usize| {
        let (a, b) = (hists[i].as_ref().unwrap(), hists[j].as_ref().unwrap());
        merged_cost(a, b, f) - a.cost - b.cost
    };
    // Until NEIGHBOR_PHASE_CTXS remain, only contexts of one channel whose
    // cells touch in DC space merge; pricing every pair of up to 192 contexts
    // dominates the search otherwise.
    let coords = |ctx: usize| {
        let (c, cell) = (ctx / num_cells, ctx % num_cells);
        let i1 = cell % n[1];
        let i2 = (cell / n[1]) % n[2];
        (c, [cell / n[1] / n[2], i1, i2])
    };
    let mut adjacent = vec![false; m * m];
    for i in 0..m {
        let (ci, a) = coords(ctx_ids[i]);
        for j in i + 1..m {
            let (cj, b) = coords(ctx_ids[j]);
            let dist: usize = (0..3).map(|p| a[p].abs_diff(b[p])).sum();
            adjacent[i * m + j] = ci == cj && dist == 1;
        }
    }
    let mut delta = vec![f64::INFINITY; m * m];
    let mut all_pairs = m <= NEIGHBOR_PHASE_CTXS;
    for i in 0..m {
        for j in i + 1..m {
            if all_pairs || adjacent[i * m + j] {
                delta[i * m + j] = pair(&hists, i, j);
            }
        }
    }
    let mut active = m;
    let mut stuck = false;
    loop {
        if !all_pairs && (active <= NEIGHBOR_PHASE_CTXS || stuck) {
            all_pairs = true;
            for i in 0..m {
                for j in i + 1..m {
                    if hists[i].is_some() && hists[j].is_some() && !adjacent[i * m + j] {
                        delta[i * m + j] = pair(&hists, i, j);
                    }
                }
            }
        }
        let mut best = (f64::INFINITY, 0, 0);
        for i in 0..m {
            if hists[i].is_none() {
                continue;
            }
            for j in i + 1..m {
                if hists[j].is_some() && delta[i * m + j] < best.0 {
                    best = (delta[i * m + j], i, j);
                }
            }
        }
        let (d, i, j) = best;
        if !d.is_finite() {
            if all_pairs {
                break;
            }
            // No neighbors left to merge: price everything.
            stuck = true;
            continue;
        }
        if active <= MAX_CTXS {
            break;
        }
        let b = hists[j].take().unwrap();
        let merged = merge(hists[i].as_ref().unwrap(), &b, f);
        hists[i] = Some(merged);
        for o in owner.iter_mut() {
            if *o == j {
                *o = i;
            }
        }
        active -= 1;
        for o in 0..m {
            if o == i || hists[o].is_none() {
                continue;
            }
            let (lo, hi) = (o.min(i), o.max(i));
            let (lj, hj) = (o.min(j), o.max(j));
            adjacent[lo * m + hi] |= adjacent[lj * m + hj];
            delta[lo * m + hi] = if all_pairs || adjacent[lo * m + hi] {
                pair(&hists, lo, hi)
            } else {
                f64::INFINITY
            };
        }
    }
    // Cluster ids by first appearance.
    let mut id = vec![u8::MAX; m];
    let mut next = 0u8;
    let mut assign = vec![None; 3 * num_cells];
    for (local, &ctx) in ctx_ids.iter().enumerate() {
        let root = owner[local];
        if id[root] == u8::MAX {
            id[root] = next;
            next += 1;
        }
        assign[ctx] = Some(id[root]);
    }
    let cost = hists.iter().flatten().map(|h| h.cost).sum::<f64>();
    Layout {
        starts: starts.clone(),
        assign,
        cost,
    }
}

/// Removes thresholds whose two sides share every channel's context.
fn prune_dead(layout: &mut Layout) {
    loop {
        let n = [
            layout.starts[0].len() + 1,
            layout.starts[1].len() + 1,
            layout.starts[2].len() + 1,
        ];
        let num_cells = n[0] * n[1] * n[2];
        let cell = |i: [usize; 3]| (i[0] * n[2] + i[2]) * n[1] + i[1];
        let mut removed = false;
        'axes: for p in [1, 0, 2] {
            for j in 1..n[p] {
                let mut compatible = true;
                'check: for c in 0..3 {
                    for a in 0..n[0] {
                        for bb in 0..n[1] {
                            for cc in 0..n[2] {
                                let mut i = [a, bb, cc];
                                if i[p] != j {
                                    continue;
                                }
                                let hi = layout.assign[c * num_cells + cell(i)];
                                i[p] = j - 1;
                                let lo = layout.assign[c * num_cells + cell(i)];
                                if let (Some(x), Some(y)) = (lo, hi)
                                    && x != y
                                {
                                    compatible = false;
                                    break 'check;
                                }
                            }
                        }
                    }
                }
                if !compatible {
                    continue;
                }
                // Fold interval j into j - 1.
                let mut m = n;
                m[p] -= 1;
                let new_cells = m[0] * m[1] * m[2];
                let new_cell = |i: [usize; 3]| (i[0] * m[2] + i[2]) * m[1] + i[1];
                let mut assign = vec![None; 3 * new_cells];
                for c in 0..3 {
                    for a in 0..n[0] {
                        for bb in 0..n[1] {
                            for cc in 0..n[2] {
                                let i = [a, bb, cc];
                                let mut t = i;
                                if t[p] >= j {
                                    t[p] -= 1;
                                }
                                let slot = &mut assign[c * new_cells + new_cell(t)];
                                if slot.is_none() {
                                    *slot = layout.assign[c * num_cells + cell(i)];
                                }
                            }
                        }
                    }
                }
                layout.assign = assign;
                layout.starts[p].remove(j - 1);
                removed = true;
                break 'axes;
            }
        }
        if !removed {
            return;
        }
    }
}

/// The block context map of the cheapest layout over [`CANDIDATES`].
pub(super) fn optimize(
    jpg: &JpegData,
    dim: &Dim,
    ss: &ChannelLayout,
    cfl: Option<&JpegCfl>,
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> BlockCtx {
    let stats = collect_stats(jpg, dim, ss, cfl, scan, pool, scratch);
    let f = NLogN::new();
    let sizes = [
        stats.axes[0].len(),
        stats.axes[1].len(),
        stats.axes[2].len(),
    ];
    let mut candidates: Vec<[usize; 3]> = Vec::new();
    for [y, x, b] in CANDIDATES {
        let k = [x.min(sizes[0]), y.min(sizes[1]), b.min(sizes[2])];
        debug_assert!(k.iter().product::<usize>() <= MAX_CELLS);
        debug_assert!(k.iter().all(|&v| v <= MAX_INTERVALS));
        if !candidates.contains(&k) {
            candidates.push(k);
        }
    }
    let layouts: Vec<Layout> = pool.steal_map(scratch, candidates.len(), |i, _| {
        let k = candidates[i];
        let mut acc = Accum::new();
        // Equal-population starting thresholds; one descent pass over the
        // planes gets within 0.01 % of three.
        let mut starts: [Vec<usize>; 3] =
            std::array::from_fn(|p| (1..k[p]).map(|j| j * sizes[p] / k[p]).collect());
        for p in [1, 0, 2] {
            if k[p] > 1 {
                sweep_axis(&stats, &mut starts, p, k[p], &f, &mut acc);
            }
        }
        cluster(&stats, &starts, &f, &mut acc)
    });
    let mut best = layouts
        .into_iter()
        .min_by(|a, b| a.cost.total_cmp(&b.cost))
        .unwrap();
    prune_dead(&mut best);
    let thresholds: [Vec<i32>; 3] = std::array::from_fn(|p| {
        best.starts[p]
            .iter()
            .map(|&s| stats.axes[p].lo[s] - 1)
            .collect()
    });
    let num_cells = thresholds.iter().map(|t| t.len() + 1).product::<usize>();
    BlockCtx::from_cells(thresholds, |row, i| {
        let c = [1, 0, 2][row];
        best.assign[c * num_cells + i].unwrap_or(0)
    })
}
