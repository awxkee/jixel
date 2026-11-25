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

use super::super::GroupLayout;
use super::super::entropy::EntropyOfHistFn;
use super::super::ma_learning::{rank_bits, rank_coarse};
use super::super::ma_source::MA_LITE_SAMPLING;
use super::super::ma_source::{MaChannel, MaPixels, MaSource, MaTransform};
use super::super::rct::selected_rct_row_fn;
use super::{visit_gray_colors, visit_palette_colors};
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::{pack_signed, uint_encode};
use crate::image::Image3Si;
use crate::shared_sort::SharedSort;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Largest global palette the learned-tree path tries (libjxl's default).
const GLOBAL_PALETTE_MAX_COLORS: usize = 4096;

/// Largest one when the frame races its palette orders. Past
/// [`GLOBAL_PALETTE_MAX_COLORS`] only the ramp order makes one index plane
/// code below per-group palettes, and only on flat art.
const RACED_PALETTE_MAX_COLORS: usize = 16384;

/// Pixels in a hundred that repeat the one before them for a palette past
/// [`GLOBAL_PALETTE_MAX_COLORS`] to be tried: flat art holds 86-94, photos
/// and dithers of as many colors under 40.
const LARGE_PALETTE_MIN_REPEATS: usize = 85;

/// A frame of at most this many pixels tries its large palette whatever it
/// repeats: the try costs it no time worth the gate.
const LARGE_PALETTE_FREE_PIXELS: usize = 1 << 18;

const _: () = assert!(GLOBAL_PALETTE_MAX_COLORS <= u16::MAX as usize + 1);

/// The palette candidate is only deepened and written when its first-stage
/// estimate is within this factor of the RGB candidate's.
pub(in crate::lossless) const PALETTE_COARSE_MARGIN: f64 = 1.10;

/// The same bar for a palette of more than 4096 colors, which is learned
/// beside the RGB tree. Of 21 such frames the 13 whose palette frame came out
/// smaller ranked at 0.45-0.92 of the RGB tree, and five of the eight others
/// at 0.94-1.07: their full learns were a third of the large palettes' time.
pub(in crate::lossless) const LARGE_PALETTE_COARSE_MARGIN: f64 = 0.93;

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
    /// More colors than a frame that does not race its orders may hold.
    /// Such a palette is one more candidate: the RGB tree and the per-group
    /// palettes it would otherwise stand in for are still learned.
    pub(in crate::lossless) fn is_large(&self) -> bool {
        self.nb_colors > GLOBAL_PALETTE_MAX_COLORS
    }

    /// Two colors: a bi-level image.
    pub(in crate::lossless) fn is_bilevel(&self) -> bool {
        self.nb_colors == 2
    }

    /// [`screen_share`] of a bi-level palette's index image.
    pub(in crate::lossless) fn screen_share(&self, xsize: usize, ysize: usize) -> f64 {
        screen_share(|x, y| self.indices[y * xsize + x] != 0, xsize, ysize)
    }

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
            hshift: 0,
            vshift: 0,
            value_tree: false,
            gain_order: false,
            group_channels: Vec::new(),
        }
    }
}

/// A bi-level frame runs its Zero-predictor arms when at least this share
/// of its busy 64-pixel spans is a repeated screen ([`screen_share`]).
pub(in crate::lossless) const SCREEN_MIN_SHARE: f64 = 0.2;

/// Rows above that a repeated span is looked for in.
const SCREEN_REACH: usize = 16;

/// Share of the busy 64-pixel spans of a bi-level image (neither all set
/// nor all clear) that differ from the span right above and equal one 2 to
/// [`SCREEN_REACH`] rows up: a dot screen repeating down the page. Strokes
/// and rules repeat from the row above, which a tree predicts; a screen's
/// rows repeat a period back, which only LZ77 reaches.
pub(in crate::lossless) fn screen_share(
    bit: impl Fn(usize, usize) -> bool,
    xsize: usize,
    ysize: usize,
) -> f64 {
    let spans = xsize / 64;
    if spans == 0 || ysize <= SCREEN_REACH {
        return 0.0;
    }
    let mut words = vec![0u64; spans * ysize];
    for (y, row) in words.chunks_exact_mut(spans).enumerate() {
        for (span, word) in row.iter_mut().enumerate() {
            *word = (0..64).fold(0u64, |acc, i| acc << 1 | u64::from(bit(span * 64 + i, y)));
        }
    }
    let (mut busy, mut screen) = (0usize, 0usize);
    for y in SCREEN_REACH..ysize {
        for span in 0..spans {
            let word = words[y * spans + span];
            if word == 0 || word == u64::MAX {
                continue;
            }
            busy += 1;
            if word != words[(y - 1) * spans + span]
                && (2..=SCREEN_REACH).any(|d| word == words[(y - d) * spans + span])
            {
                screen += 1;
            }
        }
    }
    if busy == 0 {
        0.0
    } else {
        screen as f64 / busy as f64
    }
}

/// [`screen_share`] of an image of exactly two colors, `None` for any other.
pub(in crate::lossless) fn bilevel_screen_share(linear: &Image3Si) -> Option<f64> {
    let (xsize, ysize) = (linear.xsize(), linear.ysize());
    if xsize == 0 || ysize == 0 {
        return None;
    }
    let planes = [0, 1, 2].map(|c| linear.plane_data(c));
    let color_at = |i: usize| planes.map(|p| p[i]);
    let first = color_at(0);
    let mut second = None;
    for i in 0..xsize * ysize {
        let color = color_at(i);
        if color != first && *second.get_or_insert(color) != color {
            return None;
        }
    }
    second?;
    Some(screen_share(
        |x, y| color_at(y * xsize + x) != first,
        xsize,
        ysize,
    ))
}

/// Most colors a global palette may have for its index image to be packed.
pub(in crate::lossless) const PACKED_PALETTE_MAX_COLORS: usize = 4;

/// A global palette of at most [`PACKED_PALETTE_MAX_COLORS`] colors with its
/// index image packed two pixels to a sample. One Squeeze of the index
/// channel leaves an average and a residual per pixel pair, and a Palette
/// over those two channels codes one index per (average, residual) tuple.
/// On a 0 / 1 index the squeeze smoothing term is always zero, so each pair
/// is one of four tuples; with more colors it is not, and one pair can take
/// a few tuples. The half-size index image gives the tree twice the
/// neighborhood and LZ77 a wider alphabet.
pub(in crate::lossless) struct PackedPalette<'a> {
    base: &'a GlobalPalette,
    /// The second palette: its averages, then its residuals.
    pub(super) tuples: Vec<i32>,
    pub(super) indices: Vec<u16>,
    pub(super) packed_colors: usize,
    /// Pairs lie along rows; otherwise along columns.
    pub(super) horizontal: bool,
}

impl<'a> PackedPalette<'a> {
    /// `None` unless the palette has 2..=[`PACKED_PALETTE_MAX_COLORS`] colors
    /// and the samples pair up along a row or, failing that, along a column:
    /// the average channel of an odd size is one sample longer than the
    /// residual channel, and a Palette needs equal channels.
    pub(in crate::lossless) fn new(
        base: &'a GlobalPalette,
        xsize: usize,
        ysize: usize,
    ) -> Option<Self> {
        let colors = base.nb_colors;
        if !(2..=PACKED_PALETTE_MAX_COLORS).contains(&colors) || xsize == 0 || ysize == 0 {
            return None;
        }
        let horizontal = xsize.is_multiple_of(2);
        if !horizontal && !ysize.is_multiple_of(2) {
            return None;
        }
        // The decoder's own forward Squeeze, smoothing term included.
        let channel = crate::squeeze::Channel {
            data: base.indices.iter().map(|&i| i32::from(i)).collect(),
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        };
        let (avg, res) = if horizontal {
            crate::squeeze::fwd_h_squeeze(&channel)
        } else {
            crate::squeeze::fwd_v_squeeze(&channel)
        };
        debug_assert_eq!(avg.data.len(), res.data.len());
        // Averages lie in 0..colors. A pair difference is below `colors` and
        // the smoothing term below `2 * colors` in magnitude, so a residual
        // lies within `3 * colors`.
        let reach = 3 * colors as i32;
        let span = 2 * reach + 1;
        let slot = |a: i32, r: i32| {
            debug_assert!((0..colors as i32).contains(&a) && r.abs() <= reach);
            (a * span + r + reach) as usize
        };
        let mut used = vec![false; colors * span as usize];
        for (&a, &r) in avg.data.iter().zip(&res.data) {
            used[slot(a, r)] = true;
        }
        let mut tuples: Vec<(i32, i32)> = (0..colors as i32)
            .flat_map(|a| (-reach..=reach).map(move |r| (a, r)))
            .filter(|&(a, r)| used[slot(a, r)])
            .collect();
        // Index order: the pair a tuple decodes to under a zero smoothing
        // term (the decoder's inverse Squeeze), first pixel major. A bi-level
        // palette's tuples are exactly its four pairs in `2 * first + second`
        // order.
        tuples.sort_by_key(|&(a, r)| {
            let first = a + r / 2;
            (first, first - r)
        });
        let mut index_of = vec![0u16; used.len()];
        for (index, &(a, r)) in tuples.iter().enumerate() {
            index_of[slot(a, r)] = index as u16;
        }
        let packed_colors = tuples.len();
        let mut table: Vec<i32> = tuples.iter().map(|&(a, _)| a).collect();
        table.extend(tuples.iter().map(|&(_, r)| r));
        Some(Self {
            base,
            tuples: table,
            indices: avg
                .data
                .iter()
                .zip(&res.data)
                .map(|(&a, &r)| index_of[slot(a, r)])
                .collect(),
            packed_colors,
            horizontal,
        })
    }

    pub(in crate::lossless) fn source(&self, xsize: usize, ysize: usize) -> MaSource<'_> {
        let (hshift, vshift) = (u32::from(self.horizontal), u32::from(!self.horizontal));
        MaSource {
            channels: vec![
                MaChannel {
                    w: self.packed_colors,
                    h: 2,
                    meta: true,
                    pixels: MaPixels::I32(&self.tuples),
                    constant: None,
                },
                MaChannel {
                    w: self.base.nb_colors,
                    h: self.base.num_c,
                    meta: true,
                    pixels: MaPixels::I32(&self.base.palette),
                    constant: None,
                },
                MaChannel {
                    w: xsize >> hshift,
                    h: ysize >> vshift,
                    meta: false,
                    pixels: MaPixels::U16(&self.indices),
                    constant: None,
                },
            ],
            transform: MaTransform::PackedPalette {
                num_c: self.base.num_c as u32,
                nb_colors: self.base.nb_colors as u32,
                packed_colors: self.packed_colors as u32,
                horizontal: self.horizontal,
            },
            hshift,
            vshift,
            value_tree: false,
            gain_order: false,
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
///
/// With `chroma_orders` (color palettes), two sorts on YCoCg components
/// follow: (Co, Cg, Y) and (Y, Co, Cg). Anti-aliased flat art
/// holds a few chroma pairs with many lumas each; a chroma-major order keeps
/// every such ramp contiguous (a full encode under (Co, Cg, Y) was 7-16 %
/// smaller than under the best order above on three flat illustrations).
fn palette_order_candidates(
    colors: &[[i32; 4]],
    cooc: &Cooccurrence,
    chroma_orders: bool,
) -> (Vec<Vec<i32>>, usize) {
    let mut orders = estimated_order_candidates(colors, cooc);
    let estimated = orders.len();
    if chroma_orders && (3..=GLOBAL_PALETTE_MAX_COLORS).contains(&colors.len()) {
        let ycc: Vec<[i32; 3]> = colors
            .iter()
            .map(|c| {
                let co = c[0] - c[2];
                let t = c[2] + (co >> 1);
                let cg = c[1] - t;
                [t + (cg >> 1), co, cg]
            })
            .collect();
        for key in [[1, 2, 0], [0, 1, 2]] {
            let mut path: Vec<usize> = (0..colors.len()).collect();
            path.shared_sort_by_key(|&i| (colors[i][3], key.map(|k| ycc[i][k]), i));
            orders.push(order_of_path(&path));
        }
    }
    (orders, estimated)
}

fn estimated_order_candidates(colors: &[[i32; 4]], cooc: &Cooccurrence) -> Vec<Vec<i32>> {
    let n = colors.len();
    let identity: Vec<i32> = (0..n as i32).collect();
    let mut orders = vec![identity];
    if n < 3 {
        return orders;
    }
    let chain = cooccurrence_chain_path(cooc);
    // A large palette skips the tour: it takes 0.9-1.1 s at 6655-9117
    // colors, and its order coded 10 % above the chain's there.
    let tour = if n > GLOBAL_PALETTE_MAX_COLORS {
        chain.clone()
    } else {
        improve_chain_tour(cooc, chain.clone())
    };
    orders.push(order_of_path(&chain));
    if tour != chain {
        orders.push(order_of_path(&tour));
    }
    // A large palette races the chain, the nearest-neighbor chain and the
    // ramp order only: no principal-axis or chroma order won on one, and
    // each costs a coarse learn.
    if n <= GLOBAL_PALETTE_MAX_COLORS {
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
        by_axis.shared_sort_by(|&a, &b| projection[a].total_cmp(&projection[b]).then(a.cmp(&b)));
        let mut pca = vec![0i32; n];
        for (index, &canonical) in by_axis.iter().enumerate() {
            pca[canonical] = index as i32;
        }
        orders.push(pca);
    }

    // Greedy nearest-neighbor chain (O(n^2)) on exact squared distances. The
    // unvisited colors stay packed in canonical order: the distance pass is
    // one branch-free sweep, and ties keep going to the darkest.
    let mut chain = vec![0i32; n];
    let mut left: Vec<[i32; 4]> = colors[1..].to_vec();
    let mut left_ids: Vec<u32> = (1..n as u32).collect();
    let mut dist = vec![0i64; n];
    let mut cur = colors[0]; // darkest color in canonical (luma) order
    for index in 1..n {
        let dist = &mut dist[..left.len()];
        for (d, color) in dist.iter_mut().zip(&left) {
            let mut sum = 0i64;
            for c in 0..4 {
                let delta = i64::from(color[c] - cur[c]);
                sum += delta * delta;
            }
            *d = sum;
        }
        let nearest = dist.iter().copied().min().unwrap_or(0);
        let at = dist.iter().position(|&d| d == nearest).unwrap_or(0);
        cur = left.remove(at);
        chain[left_ids.remove(at) as usize] = index as i32;
    }
    orders.push(chain);
    orders
}

/// Colors holding at least one pixel in this many are ramp anchors.
const RAMP_ANCHOR_SHARE_DEN: usize = 200;

/// Most anchors, the most frequent first.
const RAMP_MAX_ANCHORS: usize = 48;

/// Anchor pairs remembered per color.
const RAMP_MAX_VOTES: usize = 16;

/// An index order for anti-aliased flat art: every blend ramp between two
/// flat colors as one monotone run of indices. The anchors are the frequent
/// colors; every other color joins the anchor pair that most often brackets
/// its pixels along rows and columns, at its projection on the segment
/// between the two. Ramps follow each other through shared anchors, the
/// heaviest first. Returns canonical colors in index order, `None` without
/// two anchors.
pub(super) fn ramp_path(colors: &[[i32; 4]], indices: &[u16], xsize: usize) -> Option<Vec<usize>> {
    const NONE: u8 = u8::MAX;
    let n = colors.len();
    let npx = indices.len();
    if xsize == 0 {
        return None;
    }
    let mut count = vec![0u32; n];
    for &i in indices {
        count[i as usize] += 1;
    }
    let mut anchors: Vec<usize> = (0..n)
        .filter(|&c| count[c] as usize * RAMP_ANCHOR_SHARE_DEN >= npx)
        .collect();
    anchors.shared_sort_by_key(|&c| (std::cmp::Reverse(count[c]), c));
    anchors.truncate(RAMP_MAX_ANCHORS);
    let k = anchors.len();
    if k < 2 || k == n {
        return None;
    }
    let mut anchor_of = vec![NONE; n];
    for (a, &c) in anchors.iter().enumerate() {
        anchor_of[c] = a as u8;
    }
    // `votes[color]`: (lo * k + hi, pixels bracketed by anchors lo and hi).
    let mut votes: Vec<Vec<(u16, u32)>> = vec![Vec::new(); n];
    let mut vote = |color: u16, a: u8, b: u8| {
        let code = (a.min(b) as usize * k + a.max(b) as usize) as u16;
        let list = &mut votes[color as usize];
        if let Some(at) = list.iter().position(|(c, _)| *c == code) {
            list[at].1 += 1;
        } else if list.len() < RAMP_MAX_VOTES {
            list.push((code, 1));
        }
    };
    for row in indices.chunks_exact(xsize) {
        let (mut last, mut start) = (NONE, 0usize);
        for (x, &c) in row.iter().enumerate() {
            let a = anchor_of[c as usize];
            if a != NONE {
                if last != NONE {
                    for &between in &row[start..x] {
                        vote(between, last, a);
                    }
                }
                (last, start) = (a, x + 1);
            }
        }
    }
    let mut last = vec![NONE; xsize];
    let mut start = vec![0usize; xsize];
    for (y, row) in indices.chunks_exact(xsize).enumerate() {
        for (x, ((&c, last), start)) in row.iter().zip(&mut last).zip(&mut start).enumerate() {
            let a = anchor_of[c as usize];
            if a != NONE {
                if *last != NONE {
                    // The column's pixels between its two anchors.
                    let between = indices[*start * xsize + x..y * xsize + x]
                        .iter()
                        .step_by(xsize);
                    for &color in between {
                        vote(color, *last, a);
                    }
                }
                (*last, *start) = (a, y + 1);
            }
        }
    }
    let point = |c: usize| colors[c].map(|v| v as f64);
    // Squared distance of `c` to the segment between anchors `i` and `j`
    // and its position along it.
    let segment = |c: usize, i: usize, j: usize| -> (f64, f64) {
        let (a, b, v) = (point(anchors[i]), point(anchors[j]), point(c));
        let (mut dot, mut len2) = (0.0, 0.0);
        for q in 0..4 {
            dot += (v[q] - a[q]) * (b[q] - a[q]);
            len2 += (b[q] - a[q]) * (b[q] - a[q]);
        }
        let t = if len2 > 0.0 {
            (dot / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let d2 = (0..4)
            .map(|q| (v[q] - (a[q] + t * (b[q] - a[q]))).powi(2))
            .sum();
        (d2, t)
    };
    // Per anchor pair, its colors with their positions. The distances that
    // pick the pair stay in `f64`: in `f32` near ties fall the other way.
    // The stored position only sorts a ramp.
    let mut ramps: std::collections::BTreeMap<(usize, usize), Vec<(f32, u32)>> = Default::default();
    for c in 0..n {
        if anchor_of[c] != NONE {
            continue;
        }
        let mut best: Option<(u32, f64, usize, usize, f64)> = None;
        for &(code, v) in &votes[c] {
            let (mut i, mut j) = (code as usize / k, code as usize % k);
            if i == j {
                // Bracketed by one color: the other end is the nearest segment's.
                let other = (0..k)
                    .filter(|&x| x != i)
                    .min_by(|&x, &y| segment(c, i, x).0.total_cmp(&segment(c, i, y).0))
                    .expect("two anchors");
                (i, j) = (i.min(other), i.max(other));
            }
            let (d2, t) = segment(c, i, j);
            if best.is_none_or(|(bv, bd, ..)| v > bv || (v == bv && d2 < bd)) {
                best = Some((v, d2, i, j, t));
            }
        }
        let (i, j, t) = match best {
            Some((_, _, i, j, t)) => (i, j, t),
            None => {
                let mut near = (f64::INFINITY, 0, 1, 0.0);
                for i in 0..k {
                    for j in i + 1..k {
                        let (d2, t) = segment(c, i, j);
                        if d2 < near.0 {
                            near = (d2, i, j, t);
                        }
                    }
                }
                (near.1, near.2, near.3)
            }
        };
        ramps.entry((i, j)).or_default().push((t as f32, c as u32));
    }
    type Ramp = ((usize, usize), u64, Vec<(f32, u32)>);
    let mut remaining: Vec<Ramp> = ramps
        .into_iter()
        .map(|(pair, mut list)| {
            list.shared_sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let weight = list.iter().map(|&(_, c)| count[c as usize] as u64).sum();
            (pair, weight, list)
        })
        .collect();
    let mut path = Vec::with_capacity(n);
    let mut placed = vec![false; k];
    let mut current: Option<usize> = None;
    while !remaining.is_empty() {
        // The heaviest ramp at the current anchor, else the heaviest left.
        let heaviest = |touching: Option<usize>| {
            remaining
                .iter()
                .enumerate()
                .filter(|(_, ((i, j), ..))| touching.is_none_or(|a| a == *i || a == *j))
                .max_by_key(|(at, (_, weight, _))| (*weight, std::cmp::Reverse(*at)))
                .map(|(at, _)| at)
        };
        let at = current
            .and_then(|a| heaviest(Some(a)))
            .or_else(|| heaviest(None))
            .expect("a ramp remains");
        let ((i, j), _, list) = remaining.remove(at);
        let (from, to) = if current == Some(j) { (j, i) } else { (i, j) };
        if !std::mem::replace(&mut placed[from], true) {
            path.push(anchors[from]);
        }
        if from == i {
            path.extend(list.iter().map(|&(_, c)| c as usize));
        } else {
            path.extend(list.iter().rev().map(|&(_, c)| c as usize));
        }
        if !std::mem::replace(&mut placed[to], true) {
            path.push(anchors[to]);
        }
        current = Some(to);
    }
    path.extend((0..k).filter(|&a| !placed[a]).map(|a| anchors[a]));
    debug_assert_eq!(path.len(), n);
    Some(path)
}

/// Counts of horizontally and vertically adjacent distinct color pairs: a
/// symmetric matrix over canonical indices, kept per color as the colors it
/// touches, ascending, with their counts. A flat image's matrix is about 1 %
/// dense, and a dense one of 16384 colors would hold a gigabyte.
pub(super) struct Cooccurrence {
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
    counts: Vec<u32>,
    /// The count of any pair in one step: the matrix itself for a small
    /// palette, the pair table for a large one.
    dense: Vec<u32>,
    pairs: PairMap,
}

impl Cooccurrence {
    pub(super) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The colors `a` touches, ascending, and how often.
    pub(super) fn row(&self, a: usize) -> (&[u32], &[u32]) {
        let range = self.offsets[a] as usize..self.offsets[a + 1] as usize;
        (&self.neighbors[range.clone()], &self.counts[range])
    }

    pub(super) fn get(&self, a: usize, b: usize) -> u32 {
        if !self.dense.is_empty() {
            return self.dense[a * self.len() + b];
        }
        let key = (a.min(b) as u32) << 16 | a.max(b) as u32;
        self.pairs.get(&key).copied().unwrap_or(0)
    }

    pub(super) fn from_dense(n: usize, dense: Vec<u32>) -> Self {
        let mut cooc = Self {
            offsets: Vec::with_capacity(n + 1),
            neighbors: Vec::new(),
            counts: Vec::new(),
            dense: Vec::new(),
            pairs: PairMap::default(),
        };
        cooc.offsets.push(0);
        for row in dense.chunks_exact(n.max(1)).take(n) {
            for (b, &count) in row.iter().enumerate() {
                if count != 0 {
                    cooc.neighbors.push(b as u32);
                    cooc.counts.push(count);
                }
            }
            cooc.offsets.push(cooc.neighbors.len() as u32);
        }
        cooc.dense = dense;
        cooc
    }

    /// From the counts of the pairs `lo << 16 | hi`, `lo < hi`.
    fn from_pairs(n: usize, pairs: PairMap) -> Self {
        let mut entries: Vec<(u32, u32, u32)> = Vec::with_capacity(2 * pairs.len());
        for (&key, &count) in &pairs {
            let (lo, hi) = (key >> 16, key & 0xFFFF);
            entries.push((lo, hi, count));
            entries.push((hi, lo, count));
        }
        entries.shared_sort_unstable_by(|a, b| a.cmp(b));
        let mut offsets = vec![0u32; n + 1];
        for &(a, ..) in &entries {
            offsets[a as usize + 1] += 1;
        }
        for a in 0..n {
            offsets[a + 1] += offsets[a];
        }
        Self {
            offsets,
            neighbors: entries.iter().map(|e| e.1).collect(),
            counts: entries.iter().map(|e| e.2).collect(),
            dense: Vec::new(),
            pairs,
        }
    }
}

/// Palettes up to this size count their pairs in a dense matrix.
const DENSE_COOCCURRENCE_MAX_COLORS: usize = 4096;

#[derive(Default)]
struct PairHasher(u64);

impl std::hash::Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("pair keys hash as one u32");
    }

    fn write_u32(&mut self, key: u32) {
        self.0 = u64::from(key)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(29);
    }
}

type PairMap = std::collections::HashMap<u32, u32, std::hash::BuildHasherDefault<PairHasher>>;

pub(super) fn index_cooccurrence(
    indices: &[u16],
    xsize: usize,
    ysize: usize,
    n: usize,
) -> Cooccurrence {
    cooccurrence(indices, xsize, ysize, n, n <= DENSE_COOCCURRENCE_MAX_COLORS)
}

pub(super) fn cooccurrence(
    indices: &[u16],
    xsize: usize,
    ysize: usize,
    n: usize,
    dense: bool,
) -> Cooccurrence {
    let mut dense = vec![0u32; if dense { n * n } else { 0 }];
    let mut pairs = PairMap::default();
    // Runs of one pair (an edge between two flat colors) count without a
    // lookup.
    let mut run = (u32::MAX, 0u32);
    let mut bump = |a: u16, b: u16| {
        if a == b {
            return;
        }
        if !dense.is_empty() {
            dense[a as usize * n + b as usize] += 1;
            dense[b as usize * n + a as usize] += 1;
            return;
        }
        let key = u32::from(a.min(b)) << 16 | u32::from(a.max(b));
        if key != run.0 {
            if run.1 != 0 {
                *pairs.entry(run.0).or_insert(0) += run.1;
            }
            run = (key, 0);
        }
        run.1 += 1;
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
    if !dense.is_empty() {
        return Cooccurrence::from_dense(n, dense);
    }
    if run.1 != 0 {
        *pairs.entry(run.0).or_insert(0) += run.1;
    }
    Cooccurrence::from_pairs(n, pairs)
}

/// `order[canonical] = index` of the colors in `path` order.
pub(super) fn order_of_path(path: &[usize]) -> Vec<i32> {
    let mut order = vec![0i32; path.len()];
    for (index, &canonical) in path.iter().enumerate() {
        order[canonical] = index as i32;
    }
    order
}

/// Order colors by spatial adjacency rather than by color: grow a chain from
/// the most-connected color, at each step attaching the unused color with the
/// most co-occurrences to whichever chain end it touches more. Returns the
/// canonical colors in chain order.
pub(super) fn cooccurrence_chain_path(cooc: &Cooccurrence) -> Vec<usize> {
    let n = cooc.len();
    let total: Vec<u64> = (0..n)
        .map(|i| cooc.row(i).1.iter().map(|&v| v as u64).sum())
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
        let (neighbors, counts) = cooc.row(end);
        neighbors
            .iter()
            .zip(counts)
            .map(|(&i, &v)| (v, i as usize))
            .filter(|&(_, i)| !used[i])
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
pub(super) fn path_cooccurrence(cooc: &Cooccurrence, path: &[usize]) -> u64 {
    path.array_windows::<2>()
        .map(|&[a, b]| cooc.get(a, b) as u64)
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
pub(super) fn improve_chain_tour(cooc: &Cooccurrence, mut path: Vec<usize>) -> Vec<usize> {
    let n = cooc.len();
    const TOUR_OR_OPT_SEGMENT: usize = 8;
    const TOUR_MAX_ROUNDS: usize = 16;
    if n < 3 {
        return path;
    }
    let mut offsets = vec![0usize; n + 1];
    let mut neighbors = Vec::new();
    for a in 0..n {
        neighbors.extend(cooc.row(a).0.iter().map(|&b| b as usize));
        offsets[a + 1] = neighbors.len();
    }
    let neighbors = |c: usize| &neighbors[offsets[c]..offsets[c + 1]];
    let w = |a: usize, b: usize| cooc.get(a, b) as i64;
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
    // Rows under `order` with two zeros before and one after: the neighbors
    // outside the plane.
    let remap = |y: usize, row: &mut [i32]| {
        for (dst, &index) in row[2..2 + xsize]
            .iter_mut()
            .zip(&indices[y * xsize..][..xsize])
        {
            *dst = order[index as usize];
        }
    };
    let mut above = vec![0i32; xsize + 3];
    let mut row = vec![0i32; xsize + 3];
    let mut samples = vec![[0; 4]; xsize];
    let price_row = super::selected_palette_row_fn();
    if y_begin > 0 {
        remap(y_begin - 1, &mut above);
    }
    for y in y_begin..y_end {
        remap(y, &mut row);
        price_row(&above, &row, &mut samples);
        for (&[ctx, left, top, gradient], &v) in samples.iter().zip(&row[2..2 + xsize]) {
            let ctx = ctx as usize;
            for (pred, residual) in [left, top, gradient].into_iter().enumerate() {
                let slot = ctx * INDEX_COST_PREDS + pred;
                let (tok, nb, _) = uint_encode(pack_signed(residual));
                hist.tokens
                    [slot * INDEX_COST_ALPHABET + (tok as usize).min(INDEX_COST_ALPHABET - 1)] += 1;
                hist.raw_bits[slot] += nb as u64;
            }
            hist.values[ctx * nb_colors + v as usize] += 1;
        }
        std::mem::swap(&mut above, &mut row);
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

#[cfg(test)]
pub(in crate::lossless) fn build_global_palette(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    num_color: usize,
    xsize: usize,
    ysize: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<GlobalPalette> {
    build_global_palettes(linear, alpha, num_color, xsize, ysize, None, pool, scratch)
        .map(|(palette, _)| palette)
}

/// The order race's second order must lead the first by this factor in the
/// coarse learn.
const ORDER_RACE_COARSE_LEAD: f64 = 0.995;

/// The coarse learn's bits of a large palette's ramp order count this much.
const RAMP_COARSE_HANDICAP: f64 = 0.9;

/// What the order race's coarse learns need of the frame.
#[derive(Clone, Copy)]
pub(in crate::lossless) struct OrderRace {
    pub(in crate::lossless) min_symbol: u32,
    pub(in crate::lossless) use_wp: bool,
    /// Try a palette of up to [`RACED_PALETTE_MAX_COLORS`]. The frame then
    /// learns it beside its other sources, not in their place.
    pub(in crate::lossless) large: bool,
    /// Race the ramp order on a palette of any size.
    pub(in crate::lossless) ramp: bool,
    /// Race the orders of a palette of at most 4096 colors too.
    pub(in crate::lossless) small: bool,
}

/// The global palette under the order the cheap estimate ranks first and,
/// with `race`, under the best other order by the coarse tree learn when
/// that learn puts it ahead. The estimate picks an order coding 5-16 %
/// larger than another candidate's on flat illustrations and line art, and
/// the coarse learn alone picks wrong too (+2 % on a UI capture), so the
/// caller learns both and compares their complete estimates. The coarse
/// learns sample at the lite budget; the full one picked no better order.
#[allow(clippy::too_many_arguments)]
pub(in crate::lossless) fn build_global_palettes(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    num_color: usize,
    xsize: usize,
    ysize: usize,
    race: Option<OrderRace>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<(GlobalPalette, Option<GlobalPalette>)> {
    let gray = num_color == 1;
    let num_c = num_color + usize::from(alpha.is_some());
    let npx = xsize * ysize;
    let (colors, canonical_of_id, mut indices) = {
        let (first_seen, indices) = {
            // One pass: first-seen ids per pixel, remapped to the final order below.
            let mut seen: ColorMap<i32> = ColorMap::default();
            let cap = if race.is_some_and(|race| race.large) && !gray {
                RACED_PALETTE_MAX_COLORS
            } else {
                GLOBAL_PALETTE_MAX_COLORS
            };
            let mut repeats = 0usize;
            seen.reserve(cap + 1);
            let mut first_seen: Vec<[i32; 4]> = Vec::with_capacity(cap + 1);
            let mut indices: Vec<u16> = Vec::with_capacity(npx);
            let mut previous = ([0; 4], -1);
            // YCoCg is reversible, so distinct input tuples identify exactly the same
            // colors. Reconstruct the accepted palette entries in batches.
            let visit = |color: [i32; 4]| {
                if previous.1 < 0 || previous.0 != color {
                    let next = first_seen.len() as i32;
                    let id = *seen.entry(color).or_insert_with(|| {
                        first_seen.push(color);
                        next
                    });
                    if first_seen.len() > cap {
                        return false;
                    }
                    previous = (color, id);
                } else {
                    repeats += 1;
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
            if nb_colors > GLOBAL_PALETTE_MAX_COLORS
                && npx > LARGE_PALETTE_FREE_PIXELS
                && repeats * 100 < LARGE_PALETTE_MIN_REPEATS * npx
            {
                return None;
            }
            if !gray {
                let inverse = selected_rct_row_fn();
                const BLOCK: usize = 64;
                let mut src = [[0; BLOCK]; 3];
                let mut rgb = [[0; BLOCK]; 3];
                for colors in first_seen.chunks_mut(BLOCK) {
                    let len = colors.len();
                    let [y, co, cg] = &mut src;
                    for (i, &[a, b, c, _]) in colors.iter().enumerate() {
                        (y[i], co[i], cg[i]) = (a, b, c);
                    }
                    inverse(
                        src.each_ref().map(|p| &p[..len]),
                        0,
                        rgb.each_mut().map(|p| &mut p[..len]),
                    );
                    let [r, g, b] = &rgb;
                    for (i, color) in colors.iter_mut().enumerate() {
                        color[..3].copy_from_slice(&[r[i], g[i], b[i]]);
                    }
                }
            }
            (first_seen, indices)
        };
        let nb_colors = first_seen.len();
        // Canonical order (luma) gives one index image; every other candidate
        // order is a permutation of it, scored in place.
        let mut canonical_ids: Vec<usize> = (0..nb_colors).collect();
        canonical_ids.shared_sort_unstable_by_key(|&id| {
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
    let race = race.filter(|race| race.small || nb_colors > GLOBAL_PALETTE_MAX_COLORS);
    let (orders, estimated, ramp) = {
        let cooc = index_cooccurrence(&indices, xsize, ysize, nb_colors);
        let (mut orders, estimated) =
            palette_order_candidates(&colors, &cooc, race.is_some() && !gray);
        // The ramp order races for a large palette and in a vector-layer
        // frame. On lossless palettes of at most 4096 colors it changed 2 of
        // 48 flat images by 0.5 % for 8 % of the encode time.
        let mut ramp = None;
        if !gray
            && (nb_colors > GLOBAL_PALETTE_MAX_COLORS || race.is_some_and(|race| race.ramp))
            && let Some(path) = ramp_path(&colors, &indices, xsize)
        {
            ramp = Some(orders.len());
            orders.push(order_of_path(&path));
        }
        (orders, estimated, ramp)
    };
    let large = nb_colors > GLOBAL_PALETTE_MAX_COLORS;
    let mut best_order = 0usize;
    let mut best_cost = f32::INFINITY;
    for (k, order) in orders.iter().enumerate().take(estimated) {
        // A large palette's order is the coarse learn's alone.
        if large || orders[..k].contains(order) {
            continue;
        }
        let cost = estimate_index_plane_cost(&indices, xsize, ysize, order, pool, scratch);
        if cost < best_cost {
            best_cost = cost;
            best_order = k;
        }
    }
    // The palette under `order` (`order[canonical] = final index`).
    let under = |order: &[i32]| {
        let mut palette = vec![0i32; num_c * nb_colors];
        for (canonical, color) in colors.iter().enumerate() {
            let index = order[canonical] as usize;
            for c in 0..num_c {
                // A gray palette holds the value and, after it, the alpha.
                let component = if gray && c == 1 { 3 } else { c };
                palette[c * nb_colors + index] = color[component];
            }
        }
        GlobalPalette {
            palette,
            indices: indices.iter().map(|&v| order[v as usize] as u16).collect(),
            nb_colors,
            num_c,
        }
    };
    let Some(race) = race else {
        return Some((under(&orders[best_order]), None));
    };
    let mut coarse = |palette: &GlobalPalette, gain_order: bool| {
        let mut source = palette.source(xsize, ysize);
        source.gain_order = gain_order;
        rank_coarse(
            &source,
            xsize,
            ysize,
            race.min_symbol,
            WpParams::DEFAULT,
            race.use_wp,
            GroupLayout::LARGE,
            MA_LITE_SAMPLING,
            pool,
            scratch,
        )
        .map_or(f64::INFINITY, |stage| rank_bits(&stage))
    };
    if large {
        // One order, the coarse learn's best past the luma order (which led
        // on none of 21 frames). The estimate's pick was luma on half of
        // them, and the order racing it then coded smaller on 9 of 10; its
        // full learn was a quarter of what the large palette cost.
        let mut best: Option<(f64, GlobalPalette)> = None;
        for (k, order) in orders.iter().enumerate().skip(1) {
            let palette = under(order);
            let handicap = if ramp == Some(k) {
                RAMP_COARSE_HANDICAP
            } else {
                1.0
            };
            let bits = coarse(&palette, false) * handicap;
            if best.as_ref().is_none_or(|(least, _)| bits < *least) {
                best = Some((bits, palette));
            }
        }
        return best.map(|(_, palette)| (palette, None));
    }
    // Whether a second order races is read from learns that split in the
    // order of the gains: index first, a fractal's two orders came within
    // the lead of each other, both were learned and written (1.6 times the
    // encode) and the second lost.
    let first = under(&orders[best_order]);
    let first_bits = coarse(&first, true);
    let mut second: Option<(f64, GlobalPalette)> = None;
    for (k, order) in orders.iter().enumerate() {
        if k == best_order || orders[..k].contains(order) {
            continue;
        }
        let palette = under(order);
        let bits = coarse(&palette, true);
        let to_beat = second
            .as_ref()
            .map_or(first_bits * ORDER_RACE_COARSE_LEAD, |(best, _)| *best);
        if bits < to_beat {
            second = Some((bits, palette));
        }
    }
    Some((first, second.map(|(_, palette)| palette)))
}
