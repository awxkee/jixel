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

//! Global palette construction, index ordering, and cost estimation.

use super::super::entropy::EntropyOfHistFn;
use super::super::ma_source::{MaChannel, MaPixels, MaSource, MaTransform};
use super::super::predictor::clamped_gradient;
use super::super::rct::inverse_ycocg;
use super::{visit_gray_colors, visit_palette_colors};
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::{pack_signed, uint_encode};
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;

/// Largest global palette the learned-tree path tries (libjxl's default).
const GLOBAL_PALETTE_MAX_COLORS: usize = 4096;

const _: () = assert!(GLOBAL_PALETTE_MAX_COLORS <= u16::MAX as usize + 1);

/// The palette candidate is only deepened and written when its first-stage
/// estimate is within this factor of the RGB candidate's.
pub(in crate::lossless) const PALETTE_COARSE_MARGIN: f64 = 1.10;

/// The palette frame is only written when its final learned estimate is
/// within this factor of the RGB frame's (the estimate is ~0.5% noisy).
pub(in crate::lossless) const PALETTE_FINAL_MARGIN: f64 = 1.03;

/// A global palette: RGB(A) or gray(A) entries in the index order that scored best in
/// `build_global_palette` (luma, PCA, a color-space chain or the spatial
/// co-occurrence chain — index neighbors must be spatial neighbors for the
/// predictors and the tree to exploit them) and one index per pixel.
pub(in crate::lossless) struct GlobalPalette {
    pub(super) palette: Vec<i32>,
    // At most 1024 colors: retain two bytes per pixel instead of four.
    pub(super) indices: Vec<u16>,
    pub(super) nb_colors: usize,
    pub(super) num_c: usize,
}

impl GlobalPalette {
    pub(in crate::lossless) fn source(&self, xsize: usize, ysize: usize) -> MaSource<'_> {
        let nb_colors = self.nb_colors;
        MaSource {
            channels: vec![
                MaChannel {
                    w: nb_colors,
                    h: self.num_c,
                    meta: true,
                    pixels: MaPixels::I32(&self.palette),
                    constant: None,
                },
                MaChannel {
                    w: xsize,
                    h: ysize,
                    meta: false,
                    pixels: MaPixels::U16(&self.indices),
                    constant: None,
                },
            ],
            transform: MaTransform::Palette {
                num_c: self.num_c as u32,
                nb_colors: nb_colors as u32,
            },
            group_channels: Vec::new(),
        }
    }
}

fn luma_key(c: &[i32; 4]) -> i64 {
    299 * c[0] as i64 + 587 * c[1] as i64 + 114 * c[2] as i64
}

/// Candidate index orders for a palette, each as `order[canonical] = index`
/// over the luma-sorted canonical list: luma itself, the spatial
/// co-occurrence chain, the principal axis of the color cloud, and a greedy
/// nearest-neighbor chain from the darkest color. Gradient-style predictors on the index image only work when
/// spatially adjacent colors get adjacent indices, and which order does
/// that is content-dependent.
fn palette_order_candidates(colors: &[[i32; 4]], cooc: &[u32]) -> Vec<Vec<i32>> {
    let n = colors.len();
    let identity: Vec<i32> = (0..n as i32).collect();
    let mut orders = vec![identity];
    if n < 3 {
        return orders;
    }
    let chain = cooccurrence_chain_path(n, cooc);
    let tour = improve_chain_tour(n, cooc, chain.clone());
    orders.push(order_of_path(&chain));
    if tour != chain {
        orders.push(order_of_path(&tour));
    }
    // Single precision is ample for the axis (the color cloud spans 16
    // bits; the projection only ranks colors, ties broken by index) and the
    // chain distances are exact integers.
    let comp = |c: &[i32; 4]| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32];
    let mean = colors.iter().fold([0.0f32; 4], |mut m, c| {
        for (m, v) in m.iter_mut().zip(comp(c)) {
            *m += v / n as f32;
        }
        m
    });
    // Principal axis by power iteration on the 4x4 covariance.
    let mut cov = [[0.0f32; 4]; 4];
    for c in colors {
        let d: [f32; 4] = std::array::from_fn(|i| comp(c)[i] - mean[i]);
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }
    let mut axis = [0.5f32, 0.5, 0.5, 0.1];
    for _ in 0..32 {
        let mut next = [0.0f32; 4];
        for (i, row) in cov.iter().enumerate() {
            next[i] = row.iter().zip(axis).map(|(c, a)| c * a).sum();
        }
        let norm = next.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm < 1e-9 {
            break;
        }
        axis = next.map(|v| v / norm);
    }
    let projection: Vec<f32> = colors
        .iter()
        .map(|c| comp(c).iter().zip(axis).map(|(v, x)| v * x).sum())
        .collect();
    let mut by_axis: Vec<usize> = (0..n).collect();
    by_axis.sort_by(|&a, &b| projection[a].total_cmp(&projection[b]).then(a.cmp(&b)));
    let mut pca = vec![0i32; n];
    for (index, &canonical) in by_axis.iter().enumerate() {
        pca[canonical] = index as i32;
    }
    orders.push(pca);

    // Greedy nearest-neighbor chain (O(n^2), n <= 1024) on exact squared
    // distances.
    let mut used = vec![false; n];
    let mut chain = vec![0i32; n];
    let mut current = 0usize; // darkest color in canonical (luma) order
    used[0] = true;
    for index in 1..n {
        let cur = colors[current];
        let mut best = usize::MAX;
        let mut best_d = i64::MAX;
        for (k, (&u, color)) in used.iter().zip(colors).enumerate() {
            if u {
                continue;
            }
            let d: i64 = color
                .iter()
                .zip(cur)
                .map(|(&v, c)| {
                    let d = (v - c) as i64;
                    d * d
                })
                .sum();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        used[best] = true;
        chain[best] = index as i32;
        current = best;
    }
    orders.push(chain);
    orders
}

/// Counts of horizontally and vertically adjacent distinct color pairs, as a
/// symmetric dense `n * n` matrix over canonical indices (n <= 1024).
pub(super) fn index_cooccurrence(
    indices: &[u16],
    xsize: usize,
    ysize: usize,
    n: usize,
) -> Vec<u32> {
    let mut cooc = vec![0u32; n * n];
    let mut bump = |a: u16, b: u16| {
        if a != b {
            cooc[a as usize * n + b as usize] += 1;
            cooc[b as usize * n + a as usize] += 1;
        }
    };
    for y in 0..ysize {
        let row = &indices[y * xsize..(y + 1) * xsize];
        for pair in row.array_windows::<2>() {
            bump(pair[0], pair[1]);
        }
        if y > 0 {
            let above = &indices[(y - 1) * xsize..y * xsize];
            for (&a, &b) in above.iter().zip(row) {
                bump(a, b);
            }
        }
    }
    cooc
}

/// `order[canonical] = index` of the colors in `path` order.
fn order_of_path(path: &[usize]) -> Vec<i32> {
    let mut order = vec![0i32; path.len()];
    for (index, &canonical) in path.iter().enumerate() {
        order[canonical] = index as i32;
    }
    order
}

/// Order colors by spatial adjacency rather than by color: grow a chain from
/// the most-connected color, at each step attaching the unused color with the
/// most co-occurrences to whichever chain end it touches more.
#[cfg(test)]
pub(super) fn cooccurrence_chain(n: usize, cooc: &[u32]) -> Vec<i32> {
    order_of_path(&cooccurrence_chain_path(n, cooc))
}

/// The `cooccurrence_chain` as the list of canonical colors in chain order.
pub(super) fn cooccurrence_chain_path(n: usize, cooc: &[u32]) -> Vec<usize> {
    let total: Vec<u64> = (0..n)
        .map(|i| cooc[i * n..(i + 1) * n].iter().map(|&v| v as u64).sum())
        .collect();
    let most_connected = |used: &[bool]| {
        (0..n)
            .filter(|&i| !used[i])
            .max_by_key(|&i| (total[i], std::cmp::Reverse(i)))
            .expect("an unused color remains")
    };
    let mut used = vec![false; n];
    let start = most_connected(&used);
    used[start] = true;
    let mut chain = std::collections::VecDeque::with_capacity(n);
    chain.push_back(start);
    let (mut left, mut right) = (start, start);
    let best_from = |end: usize, used: &[bool]| {
        let row = &cooc[end * n..(end + 1) * n];
        (0..n)
            .filter(|&i| !used[i])
            .map(|i| (row[i], i))
            .max_by_key(|&(v, i)| (v, std::cmp::Reverse(i)))
            .unwrap_or((0, usize::MAX))
    };
    for _ in 1..n {
        let (lv, li) = best_from(left, &used);
        let (rv, ri) = best_from(right, &used);
        if lv >= rv && lv > 0 {
            chain.push_front(li);
            used[li] = true;
            left = li;
        } else if rv > 0 {
            chain.push_back(ri);
            used[ri] = true;
            right = ri;
        } else {
            // Disconnected component: continue from its most-connected color.
            let next = most_connected(&used);
            chain.push_back(next);
            used[next] = true;
            right = next;
        }
    }
    chain.into()
}

/// Co-occurrences along a path: the sum over its adjacent pairs.
#[cfg(test)]
pub(super) fn path_cooccurrence(n: usize, cooc: &[u32], path: &[usize]) -> u64 {
    path.array_windows::<2>()
        .map(|&[a, b]| cooc[a * n + b] as u64)
        .sum()
}

/// The chain is a greedy maximum-weight Hamiltonian path on the
/// co-occurrence graph. Reverse segments (2-opt) and move segments of up to
/// `TOUR_OR_OPT_SEGMENT` colors (Or-opt) while either raises the path's
/// co-occurrence. A move pays only where it joins two colors that co-occur
/// or frees two that do, so the candidates are the non-zero neighbors of
/// the colors at the ends of the segment moved and the path's own ends, and
/// a round costs the graph's edges rather than the square of the palette.
/// Segments of eight beat a dense search over every insertion point.
pub(super) fn improve_chain_tour(n: usize, cooc: &[u32], mut path: Vec<usize>) -> Vec<usize> {
    const TOUR_OR_OPT_SEGMENT: usize = 8;
    const TOUR_MAX_ROUNDS: usize = 16;
    if n < 3 {
        return path;
    }
    let mut offsets = vec![0usize; n + 1];
    let mut neighbors = Vec::new();
    for a in 0..n {
        neighbors.extend((0..n).filter(|&b| cooc[a * n + b] != 0));
        offsets[a + 1] = neighbors.len();
    }
    let neighbors = |c: usize| &neighbors[offsets[c]..offsets[c + 1]];
    let w = |a: usize, b: usize| cooc[a * n + b] as i64;
    let mut pos = vec![0usize; n];
    for (k, &c) in path.iter().enumerate() {
        pos[c] = k;
    }
    // Co-occurrence of the colors at two path positions, zero past an end.
    let link = |path: &[usize], a: Option<usize>, b: Option<usize>| match (a, b) {
        (Some(a), Some(b)) if a < n && b < n => w(path[a], path[b]),
        _ => 0,
    };
    let mut candidates: Vec<usize> = Vec::new();
    for _ in 0..TOUR_MAX_ROUNDS {
        let mut improved = false;
        // 2-opt: reversing [i..=j] joins i-1 with j and i with j+1.
        for i in 0..n {
            candidates.clear();
            if i > 0 {
                candidates.extend(
                    neighbors(path[i - 1])
                        .iter()
                        .map(|&c| pos[c])
                        .filter(|&j| j > i),
                );
            }
            candidates.extend(
                neighbors(path[i])
                    .iter()
                    .map(|&c| pos[c])
                    .filter(|&k| k > i + 1)
                    .map(|k| k - 1),
            );
            let mut best = (0i64, 0usize);
            for &j in &candidates {
                let before =
                    link(&path, i.checked_sub(1), Some(i)) + link(&path, Some(j), Some(j + 1));
                let after =
                    link(&path, i.checked_sub(1), Some(j)) + link(&path, Some(i), Some(j + 1));
                if after - before > best.0 {
                    best = (after - before, j);
                }
            }
            if best.0 > 0 {
                let j = best.1;
                path[i..=j].reverse();
                for (k, &c) in path.iter().enumerate().take(j + 1).skip(i) {
                    pos[c] = k;
                }
                improved = true;
            }
        }
        // Or-opt: the segment [i, end) moves next to a neighbor of either
        // of its end colors, keeping or reversing its direction.
        for len in 1..=TOUR_OR_OPT_SEGMENT.min(n - 2) {
            for i in 0..=n - len {
                let end = i + len;
                let (first, last) = (path[i], path[end - 1]);
                let removed = link(&path, i.checked_sub(1), Some(i))
                    + link(&path, Some(end - 1), Some(end))
                    - link(&path, i.checked_sub(1), Some(end));
                // Position in the path without the segment.
                let reduced = |k: usize| if k < i { k } else { k - len };
                // Neighbors in the path without the segment.
                // `left_of(0)` and `right_of(n)` are the reduced path's ends.
                let left_of = |k: usize| -> Option<usize> {
                    let k = if k == 0 { n } else { k };
                    let k = if k == end { i } else { k };
                    k.checked_sub(1).map(|k| path[k])
                };
                let right_of = |k: usize| -> Option<usize> {
                    let k = if k == n { 0 } else { k + 1 };
                    let k = if k == i { end } else { k };
                    (k < n).then(|| path[k])
                };
                let value = |a: Option<usize>, b: Option<usize>| match (a, b) {
                    (Some(a), Some(b)) => w(a, b),
                    _ => 0,
                };
                // (gain, insertion index in the reduced path, reversed)
                let mut best = (0i64, 0usize, false);
                let mut consider = |left: Option<usize>, right: Option<usize>, at: usize| {
                    for reversed in [false, true] {
                        let (head, tail) = if reversed {
                            (last, first)
                        } else {
                            (first, last)
                        };
                        let gain = value(left, Some(head)) + value(Some(tail), right)
                            - value(left, right)
                            - removed;
                        if gain > best.0 {
                            best = (gain, at, reversed);
                        }
                    }
                };
                // The path's ends cut nothing: a segment between two colors
                // that co-occur with each other more than with it leaves.
                consider(None, right_of(n), 0);
                consider(left_of(0), None, n - len);
                for &c in neighbors(first).iter().chain(neighbors(last)) {
                    let k = pos[c];
                    if (i..end).contains(&k) {
                        continue;
                    }
                    consider(Some(c), right_of(k), reduced(k) + 1);
                    consider(left_of(k), Some(c), reduced(k));
                }
                if best.0 > 0 {
                    let (_, at, reversed) = best;
                    let mut segment: Vec<usize> = path.drain(i..end).collect();
                    if reversed {
                        segment.reverse();
                    }
                    path.splice(at..at, segment);
                    for (k, &c) in path.iter().enumerate() {
                        pos[c] = k;
                    }
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    path
}

/// Contexts of `estimate_index_plane_cost`: the sign pattern of four
/// neighbor differences (W-NW, N-NW, N-NE, W-WW).
const INDEX_COST_CONTEXTS: usize = 81;

/// Token alphabet of its residual histograms.
const INDEX_COST_ALPHABET: usize = 64;

/// Residual predictors it scores per context: Left, Top, clamped gradient.
const INDEX_COST_PREDS: usize = 3;

/// Histograms of one palette order's index plane: per context, the
/// hybrid-uint token histogram and raw bits of every predictor's residual,
/// and the histogram of the indices themselves.
pub(super) struct IndexCostHist {
    /// `[context][predictor][token]`.
    pub(super) tokens: Vec<u64>,
    /// `[context][predictor]`.
    pub(super) raw_bits: Vec<u64>,
    /// `[context][index]`, `nb_colors` wide.
    pub(super) values: Vec<u64>,
    pub(super) nb_colors: usize,
}

impl IndexCostHist {
    pub(super) fn new(nb_colors: usize) -> Self {
        Self {
            tokens: vec![0; INDEX_COST_CONTEXTS * INDEX_COST_PREDS * INDEX_COST_ALPHABET],
            raw_bits: vec![0; INDEX_COST_CONTEXTS * INDEX_COST_PREDS],
            values: vec![0; INDEX_COST_CONTEXTS * nb_colors],
            nb_colors,
        }
    }

    pub(super) fn add(&mut self, other: &Self) {
        for (a, b) in self.tokens.iter_mut().zip(&other.tokens) {
            *a += b;
        }
        for (a, b) in self.raw_bits.iter_mut().zip(&other.raw_bits) {
            *a += b;
        }
        for (a, b) in self.values.iter_mut().zip(&other.values) {
            *a += b;
        }
    }

    /// Bits of the plane with, per context, the cheapest of the residual
    /// predictors and a constant prediction of the context's own index
    /// distribution (what a learned tree's leaf would code).
    pub(super) fn cost(&self, entropy_of_hist: EntropyOfHistFn) -> f32 {
        let mut cost = 0.0f32;
        for ctx in 0..INDEX_COST_CONTEXTS {
            let values = &self.values[ctx * self.nb_colors..][..self.nb_colors];
            let mut best = entropy_of_hist(values, values.iter().sum());
            for pred in 0..INDEX_COST_PREDS {
                let slot = ctx * INDEX_COST_PREDS + pred;
                let tokens = &self.tokens[slot * INDEX_COST_ALPHABET..][..INDEX_COST_ALPHABET];
                let bits =
                    entropy_of_hist(tokens, tokens.iter().sum()) + self.raw_bits[slot] as f32;
                best = best.min(bits);
            }
            cost += best;
        }
        cost
    }
}

/// Estimated bits of the index plane under `order` (`order[canonical] =
/// final index`), the measure that ranks the candidate palette orders:
/// per sign-pattern context the cheaper of Left, Top and gradient residuals
/// or the context's index distribution. A gradient-only estimate ranked the
/// co-occurrence chain first on text and UI renders, where luma order codes
/// 1-22% smaller.
pub(super) fn estimate_index_plane_cost(
    indices: &[u16],
    xsize: usize,
    ysize: usize,
    order: &[i32],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> f32 {
    let entropy_of_hist = scratch.entropy_of_hist;
    let nb_colors = order.len();
    let threads = pool.num_threads().min(ysize.max(1));
    let rows_per = ysize.div_ceil(threads).max(1);
    let parts = pool.steal_map(scratch, ysize.div_ceil(rows_per), |part, _scratch| {
        let y0 = part * rows_per;
        index_cost_rows(
            indices,
            xsize,
            y0,
            (y0 + rows_per).min(ysize),
            order,
            nb_colors,
        )
    });
    let mut hist = IndexCostHist::new(nb_colors);
    for part in &parts {
        hist.add(part);
    }
    hist.cost(entropy_of_hist)
}

/// Histograms of rows `y_begin..y_end` of the plane under `order`. Neighbors
/// outside the plane read as index 0; the row above a band is the plane's.
pub(super) fn index_cost_rows(
    indices: &[u16],
    xsize: usize,
    y_begin: usize,
    y_end: usize,
    order: &[i32],
    nb_colors: usize,
) -> IndexCostHist {
    let mut hist = IndexCostHist::new(nb_colors);
    let at = |x: usize, y: usize| order[indices[y * xsize + x] as usize];
    for y in y_begin..y_end {
        for x in 0..xsize {
            let v = at(x, y);
            let left = if x > 0 { at(x - 1, y) } else { 0 };
            let left_left = if x > 1 { at(x - 2, y) } else { 0 };
            let (top, top_left, top_right) = if y > 0 {
                (
                    at(x, y - 1),
                    if x > 0 { at(x - 1, y - 1) } else { 0 },
                    if x + 1 < xsize { at(x + 1, y - 1) } else { 0 },
                )
            } else {
                (0, 0, 0)
            };
            let sign = |d: i32| (d.signum() + 1) as usize;
            let ctx = sign(left - top_left) * 27
                + sign(top - top_left) * 9
                + sign(top - top_right) * 3
                + sign(left - left_left);
            let gradient = clamped_gradient(left as i64, top as i64, top_left as i64) as i32;
            for (pred, prediction) in [left, top, gradient].into_iter().enumerate() {
                let slot = ctx * INDEX_COST_PREDS + pred;
                let (tok, nb, _) = uint_encode(pack_signed(v - prediction));
                hist.tokens
                    [slot * INDEX_COST_ALPHABET + (tok as usize).min(INDEX_COST_ALPHABET - 1)] += 1;
                hist.raw_bits[slot] += nb as u64;
            }
            hist.values[ctx * nb_colors + v as usize] += 1;
        }
    }
    hist
}

#[derive(Default)]
pub(super) struct ColorHasher(u64);

impl std::hash::Hasher for ColorHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let bytes: &[u8; 16] = bytes.first_chunk::<16>().unwrap();
        let chunks = bytes.as_chunks::<8>().0;
        assert_eq!(chunks.len(), 2);
        let a = u64::from_le_bytes(chunks[0]);
        let b = u64::from_le_bytes(chunks[1]);
        const MUL: u64 = 0x9E37_79B9_7F4A_7C15;
        self.0 = (a.wrapping_mul(MUL).rotate_left(29) ^ b)
            .wrapping_mul(MUL)
            .rotate_left(29);
    }

    fn write_usize(&mut self, length: usize) {
        // Every ColorMap key has the same length, so omit its prefix.
        debug_assert_eq!(length, 4);
    }
}

pub(super) type ColorMap<V> =
    std::collections::HashMap<[i32; 4], V, std::hash::BuildHasherDefault<ColorHasher>>;

pub(in crate::lossless) fn build_global_palette(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    num_color: usize,
    xsize: usize,
    ysize: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<GlobalPalette> {
    let gray = num_color == 1;
    let num_c = num_color + usize::from(alpha.is_some());
    let npx = xsize * ysize;
    let (colors, canonical_of_id, mut indices) = {
        let (first_seen, indices) = {
            // One pass: first-seen ids per pixel, remapped to the final order below.
            let mut seen: ColorMap<i32> = ColorMap::default();
            seen.reserve(GLOBAL_PALETTE_MAX_COLORS + 1);
            let mut first_seen: Vec<[i32; 4]> = Vec::with_capacity(GLOBAL_PALETTE_MAX_COLORS + 1);
            let mut indices: Vec<u16> = Vec::with_capacity(npx);
            let mut previous = ([0; 4], -1);
            // YCoCg is reversible, so distinct input tuples identify exactly the same
            // colors. Reconstruct RGB only when adding a new palette entry.
            let visit = |color: [i32; 4]| {
                if previous.1 < 0 || previous.0 != color {
                    let next = first_seen.len() as i32;
                    let id = *seen.entry(color).or_insert_with(|| {
                        let (r, g, b) = if gray {
                            (color[0], color[0], color[0])
                        } else {
                            inverse_ycocg(color[0], color[1], color[2])
                        };
                        first_seen.push([r, g, b, color[3]]);
                        next
                    });
                    if first_seen.len() > GLOBAL_PALETTE_MAX_COLORS {
                        return false;
                    }
                    previous = (color, id);
                }
                indices.push(previous.1 as u16);
                true
            };
            let complete = if gray {
                visit_gray_colors(linear, alpha, xsize, ysize, visit)
            } else {
                visit_palette_colors::<false>(linear, alpha, xsize, 0, 0, xsize, ysize, visit)
            };
            if !complete {
                return None;
            }
            let nb_colors = first_seen.len();
            if nb_colors == 0 {
                return None;
            }
            // The index image must be narrower than the samples it replaces:
            // fewer indices than samples per pixel, or for one channel a
            // palette at most half as long as the span of its values.
            let narrower = if num_c == 1 {
                let (lo, hi) = first_seen.iter().fold((i32::MAX, i32::MIN), |(lo, hi), c| {
                    (lo.min(c[0]), hi.max(c[0]))
                });
                2 * nb_colors as i64 <= i64::from(hi) - i64::from(lo) + 1
            } else {
                npx + num_c * nb_colors < num_c * npx
            };
            if !narrower {
                return None;
            }
            (first_seen, indices)
        };
        let nb_colors = first_seen.len();
        // Canonical order (luma) gives one index image; every other candidate
        // order is a permutation of it, scored in place.
        let mut canonical_ids: Vec<usize> = (0..nb_colors).collect();
        canonical_ids.sort_unstable_by_key(|&id| {
            let c = &first_seen[id];
            (c[3], luma_key(c), c[0], c[1], c[2])
        });
        let colors: Vec<[i32; 4]> = canonical_ids.iter().map(|&id| first_seen[id]).collect();
        let mut canonical_of_id = vec![0i32; nb_colors];
        for (index, &id) in canonical_ids.iter().enumerate() {
            canonical_of_id[id] = index as i32;
        }
        (colors, canonical_of_id, indices)
    };
    for v in indices.iter_mut() {
        *v = canonical_of_id[*v as usize] as u16;
    }
    let nb_colors = colors.len();
    let orders = {
        let cooc = index_cooccurrence(&indices, xsize, ysize, nb_colors);
        palette_order_candidates(&colors, &cooc)
    };
    let mut best_order = 0usize;
    let mut best_cost = f32::INFINITY;
    for (k, order) in orders.iter().enumerate() {
        let cost = estimate_index_plane_cost(&indices, xsize, ysize, order, pool, scratch);
        if cost < best_cost {
            best_cost = cost;
            best_order = k;
        }
    }
    let order = &orders[best_order];
    // `order[canonical] = final index`.
    for v in indices.iter_mut() {
        *v = order[*v as usize] as u16;
    }
    let mut palette = vec![0i32; num_c * nb_colors];
    for (canonical, color) in colors.iter().enumerate() {
        let index = order[canonical] as usize;
        for c in 0..num_c {
            // A gray palette holds the value and, after it, the alpha.
            let component = if gray && c == 1 { 3 } else { c };
            palette[c * nb_colors + index] = color[component];
        }
    }
    Some(GlobalPalette {
        palette,
        indices,
        nb_colors,
        num_c,
    })
}
