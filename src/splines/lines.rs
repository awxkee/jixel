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

//! Long straight lines. A faint seam or wire is traced as several short
//! fragments: it breaks at every crossing, where its contrast dips, and where
//! its polarity flips against a changing background. Fragments lying on one
//! common line are grouped here, whatever their polarity and however wide the
//! gaps, and handed to the fitter as one chain; the color DCT can vanish in
//! the gaps and change sign along the way.

use super::detect::Chain;
use super::{Point, fast_hypot};

const MIN_FRAGMENT: usize = 20;
/// RMS distance of a fragment's points from its own line.
const MAX_FRAGMENT_RMS: f32 = 1.2;
/// Cosine of the largest angle between a fragment and the line it joins.
const MIN_DIRECTION_COS: f32 = 0.992_546_2;
const MAX_OFFSET: f32 = 1.6;
const MAX_GAP: f32 = 160.0;
/// Fragments overlapping the line's span by more than this run beside it.
const MAX_OVERLAP: f32 = 4.0;
/// A single straight chain at least this long is a long line on its own.
const MIN_SINGLE: usize = 60;

#[derive(Clone, Copy)]
struct Line {
    center: Point<f32>,
    dir: Point<f32>,
}

impl Line {
    /// Least-squares line through `points`: centroid and principal direction.
    fn through<'a>(points: impl Iterator<Item = &'a Point<f32>> + Clone) -> Self {
        let n = points.clone().count().max(1) as f32;
        let (mut cx, mut cy) = (0f32, 0f32);
        for p in points.clone() {
            cx += p.x;
            cy += p.y;
        }
        (cx, cy) = (cx / n, cy / n);
        let (mut sxx, mut sxy, mut syy) = (0f32, 0f32, 0f32);
        for p in points {
            let (dx, dy) = (p.x - cx, p.y - cy);
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        let angle = 0.5 * super::trig::f_atan2f(2.0 * sxy, sxx - syy);
        let (sin, cos) = super::trig::f_sincosf(angle);
        Line {
            center: Point::new(cx, cy),
            dir: Point::new(cos, sin),
        }
    }

    #[inline]
    fn along(&self, p: Point<f32>) -> f32 {
        (p.x - self.center.x) * self.dir.x + (p.y - self.center.y) * self.dir.y
    }

    #[inline]
    fn across(&self, p: Point<f32>) -> f32 {
        (p.y - self.center.y) * self.dir.x - (p.x - self.center.x) * self.dir.y
    }
}

struct Fragment {
    index: usize,
    line: Line,
}

fn straight_fragment(index: usize, chain: &Chain) -> Option<Fragment> {
    if chain.points.len() < MIN_FRAGMENT {
        return None;
    }
    let line = Line::through(chain.points.iter());
    let sum: f32 = chain.points.iter().map(|&p| line.across(p).powi(2)).sum();
    ((sum / chain.points.len() as f32).sqrt() <= MAX_FRAGMENT_RMS)
        .then_some(Fragment { index, line })
}

/// Span of a nearly straight chain along `line` and its largest offset from it.
fn placement(line: &Line, chain: &Chain) -> (f32, f32, f32) {
    let points = &chain.points;
    let probes = [
        points[0],
        points[points.len() / 2],
        points[points.len() - 1],
    ];
    let offset = probes
        .iter()
        .map(|&p| line.across(p).abs())
        .fold(0.0, f32::max);
    let (a, b) = (line.along(probes[0]), line.along(probes[2]));
    (a.min(b), a.max(b), offset)
}

/// Members ordered along the line, oriented the same way, gaps bridged at 1 px.
fn join(chains: &[Chain], members: &[usize]) -> Chain {
    let line = Line::through(members.iter().flat_map(|&m| chains[m].points.iter()));
    let mut order = members.to_vec();
    order.sort_by(|&a, &b| {
        let mid = |m: usize| line.along(chains[m].points[chains[m].points.len() / 2]);
        mid(a).total_cmp(&mid(b))
    });
    let mut points: Vec<Point<f32>> = Vec::new();
    let (mut scale, mut weight) = (0f32, 0f32);
    for &m in &order {
        let chain = &chains[m];
        let forward =
            line.along(chain.points[chain.points.len() - 1]) >= line.along(chain.points[0]);
        let first = if forward {
            chain.points[0]
        } else {
            chain.points[chain.points.len() - 1]
        };
        if let Some(&last) = points.last() {
            let gap = fast_hypot(first.x - last.x, first.y - last.y) as usize;
            for k in 1..gap {
                let t = k as f32 / gap as f32;
                points.push(Point::new(
                    last.x + t * (first.x - last.x),
                    last.y + t * (first.y - last.y),
                ));
            }
        }
        if forward {
            points.extend_from_slice(&chain.points);
        } else {
            points.extend(chain.points.iter().rev());
        }
        scale += chain.scale * chain.points.len() as f32;
        weight += chain.points.len() as f32;
    }
    Chain {
        points,
        scale: scale / weight.max(1.0),
    }
}

// Avoid index construction for small fragment sets.
const INDEX_MIN_FRAGMENTS: usize = 256;
const ENDPOINT_LEAF_SIZE: usize = 16;

struct Endpoint {
    point: Point<f32>,
    fragment: usize,
}

struct EndpointNode {
    min: Point<f32>,
    max: Point<f32>,
    range: std::ops::Range<usize>,
    /// Zero for a leaf; the left child immediately follows its parent.
    right: usize,
}

struct EndpointIndex {
    endpoints: Vec<Endpoint>,
    nodes: Vec<EndpointNode>,
}

impl EndpointIndex {
    fn new(chains: &[Chain], fragments: &[Fragment]) -> Option<Self> {
        if fragments.len() < INDEX_MIN_FRAGMENTS {
            return None;
        }
        let mut endpoints = Vec::with_capacity(2 * fragments.len());
        for (fragment, f) in fragments.iter().enumerate() {
            let points = &chains[f.index].points;
            for point in [points[0], points[points.len() - 1]] {
                if !point.x.is_finite() || !point.y.is_finite() {
                    return None;
                }
                endpoints.push(Endpoint { point, fragment });
            }
        }
        let mut index = Self {
            endpoints,
            nodes: Vec::new(),
        };
        index.build(0..index.endpoints.len());
        Some(index)
    }

    fn build(&mut self, range: std::ops::Range<usize>) -> usize {
        let mut min = Point::new(f32::INFINITY, f32::INFINITY);
        let mut max = Point::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for e in &self.endpoints[range.clone()] {
            min = Point::new(min.x.min(e.point.x), min.y.min(e.point.y));
            max = Point::new(max.x.max(e.point.x), max.y.max(e.point.y));
        }
        let node = self.nodes.len();
        self.nodes.push(EndpointNode {
            min,
            max,
            range: range.clone(),
            right: 0,
        });
        if range.len() > ENDPOINT_LEAF_SIZE {
            let split = range.start + range.len() / 2;
            let coord = |p: Point<f32>| {
                if max.x - min.x >= max.y - min.y {
                    p.x
                } else {
                    p.y
                }
            };
            self.endpoints[range.clone()].select_nth_unstable_by(range.len() / 2, |a, b| {
                coord(a.point).total_cmp(&coord(b.point))
            });
            self.build(range.start..split);
            self.nodes[node].right = self.build(split..range.end);
        }
        node
    }

    fn candidates(&self, line: &Line, lo: f32, hi: f32, out: &mut Vec<usize>) {
        out.clear();
        self.collect(0, line, lo, hi, out);
        // Original fragment rank resolves equal-gap ties, independent of tree order.
        out.sort_unstable();
        out.dedup();
    }

    fn collect(&self, node_index: usize, line: &Line, lo: f32, hi: f32, out: &mut Vec<usize>) {
        let node = &self.nodes[node_index];
        // Bound the exact f32 expressions used by along/across. Each subtraction,
        // multiplication and addition is monotone: choosing the appropriate box
        // corners gives conservative bounds without inverse projections or epsilons.
        let (x0, x1) = if line.dir.x >= 0.0 {
            (node.min.x, node.max.x)
        } else {
            (node.max.x, node.min.x)
        };
        let (y0, y1) = if line.dir.y >= 0.0 {
            (node.min.y, node.max.y)
        } else {
            (node.max.y, node.min.y)
        };
        let amin = line.along(Point::new(x0, y0));
        let amax = line.along(Point::new(x1, y1));
        let (x0, x1) = if line.dir.y >= 0.0 {
            (node.max.x, node.min.x)
        } else {
            (node.min.x, node.max.x)
        };
        let (y0, y1) = if line.dir.x >= 0.0 {
            (node.min.y, node.max.y)
        } else {
            (node.max.y, node.min.y)
        };
        let cmin = line.across(Point::new(x0, y0));
        let cmax = line.across(Point::new(x1, y1));
        // Retain uncertain bounds (for example overflow on artificial inputs).
        if ![amin, amax, cmin, cmax].iter().any(|v| v.is_nan()) {
            // gap = max(a-hi, lo-b) can pass only if at least one endpoint
            // lies in one of these intervals; placement still checks all probes.
            let near_hi = amax - hi >= -MAX_OVERLAP && amin - hi <= MAX_GAP;
            let near_lo = lo - amin >= -MAX_OVERLAP && lo - amax <= MAX_GAP;
            if cmin > MAX_OFFSET || cmax < -MAX_OFFSET || !(near_hi || near_lo) {
                return;
            }
        }
        if node.right == 0 {
            out.extend(
                self.endpoints[node.range.clone()]
                    .iter()
                    .map(|e| e.fragment),
            );
        } else {
            // Nodes are stored in preorder; the left child follows this node.
            self.collect(node_index + 1, line, lo, hi, out);
            self.collect(node.right, line, lo, hi, out);
        }
    }
}

pub(super) struct LongLines {
    pub(super) chains: Vec<Chain>,
    /// Indices of well-supported joins to try before their individual fragments.
    pub(super) joined: Vec<usize>,
}

/// Long straight lines among `chains`: groups of collinear fragments, plus long
/// straight chains that found no partner.
pub(super) fn find_long_lines(chains: &[Chain]) -> LongLines {
    let mut fragments: Vec<Fragment> = chains
        .iter()
        .enumerate()
        .filter_map(|(i, c)| straight_fragment(i, c))
        .collect();
    fragments.sort_by_key(|f| std::cmp::Reverse(chains[f.index].points.len()));
    let index = EndpointIndex::new(chains, &fragments);
    let mut candidates = Vec::new();
    let mut used = vec![false; chains.len()];
    let mut lines = Vec::new();
    let mut joined = Vec::new();
    for seed in 0..fragments.len() {
        let seed_index = fragments[seed].index;
        if used[seed_index] {
            continue;
        }
        let mut members = vec![seed_index];
        loop {
            let line = Line::through(members.iter().flat_map(|&m| chains[m].points.iter()));
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &m in &members {
                let (a, b, _) = placement(&line, &chains[m]);
                (lo, hi) = (lo.min(a), hi.max(b));
            }
            let mut nearest: Option<(f32, usize)> = None;
            let mut consider = |fragment: &Fragment| {
                let j = fragment.index;
                if used[j] || members.contains(&j) {
                    return;
                }
                let cos = line.dir.x * fragment.line.dir.x + line.dir.y * fragment.line.dir.y;
                if cos.abs() < MIN_DIRECTION_COS {
                    return;
                }
                let (a, b, offset) = placement(&line, &chains[j]);
                let gap = (a - hi).max(lo - b);
                if offset <= MAX_OFFSET
                    && (-MAX_OVERLAP..=MAX_GAP).contains(&gap)
                    && nearest.is_none_or(|(g, _)| gap < g)
                {
                    nearest = Some((gap, j));
                }
            };
            if let Some(index) = &index {
                index.candidates(&line, lo, hi, &mut candidates);
                for &rank in &candidates {
                    consider(&fragments[rank]);
                }
            } else {
                fragments.iter().for_each(consider);
            }
            match nearest {
                Some((_, j)) => members.push(j),
                None => break,
            }
        }
        if members.len() >= 2 {
            members.iter().for_each(|&m| used[m] = true);
            let line = join(chains, &members);
            let observed: usize = members.iter().map(|&m| chains[m].points.len()).sum();
            // Repeated interruptions provide evidence of a crossing line. Do
            // not promote a lone line, a pair, or a mostly invented gap ahead
            // of the ordinary curved and colored candidates.
            if members.len() >= 3 && observed >= line.points.len().div_ceil(2) {
                joined.push(lines.len());
            }
            lines.push(line);
        } else if chains[seed_index].points.len() >= MIN_SINGLE {
            used[seed_index] = true;
            lines.push(Chain {
                points: chains[seed_index].points.clone(),
                scale: chains[seed_index].scale,
            });
        }
    }
    LongLines {
        chains: lines,
        joined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_long_lines_reference(chains: &[Chain]) -> Vec<Chain> {
        let mut fragments: Vec<Fragment> = chains
            .iter()
            .enumerate()
            .filter_map(|(i, c)| straight_fragment(i, c))
            .collect();
        fragments.sort_by_key(|f| std::cmp::Reverse(chains[f.index].points.len()));
        let mut used = vec![false; chains.len()];
        let mut lines = Vec::new();
        for seed in 0..fragments.len() {
            let seed_index = fragments[seed].index;
            if used[seed_index] {
                continue;
            }
            let mut members = vec![seed_index];
            loop {
                let line = Line::through(members.iter().flat_map(|&m| chains[m].points.iter()));
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &m in &members {
                    let (a, b, _) = placement(&line, &chains[m]);
                    (lo, hi) = (lo.min(a), hi.max(b));
                }
                let mut nearest: Option<(f32, usize)> = None;
                for fragment in &fragments {
                    let j = fragment.index;
                    if used[j] || members.contains(&j) {
                        continue;
                    }
                    let cos = line.dir.x * fragment.line.dir.x + line.dir.y * fragment.line.dir.y;
                    if cos.abs() < MIN_DIRECTION_COS {
                        continue;
                    }
                    let (a, b, offset) = placement(&line, &chains[j]);
                    let gap = (a - hi).max(lo - b);
                    if offset <= MAX_OFFSET
                        && (-MAX_OVERLAP..=MAX_GAP).contains(&gap)
                        && nearest.is_none_or(|(g, _)| gap < g)
                    {
                        nearest = Some((gap, j));
                    }
                }
                match nearest {
                    Some((_, j)) => members.push(j),
                    None => break,
                }
            }
            if members.len() >= 2 {
                members.iter().for_each(|&m| used[m] = true);
                lines.push(join(chains, &members));
            } else if chains[seed_index].points.len() >= MIN_SINGLE {
                used[seed_index] = true;
                lines.push(Chain {
                    points: chains[seed_index].points.clone(),
                    scale: chains[seed_index].scale,
                });
            }
        }
        lines
    }

    fn segment(x0: f32, x1: f32, y: f32, slope: f32) -> Chain {
        let n = (x1 - x0).abs() as usize;
        let step = if x1 >= x0 { 1.0 } else { -1.0 };
        let points = (0..=n)
            .map(|k| {
                let x = x0 + step * k as f32;
                Point::new(x, y + slope * x)
            })
            .collect();
        Chain { points, scale: 1.0 }
    }

    fn assert_same_lines(actual: &[Chain], expected: &[Chain]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.scale.to_bits(), b.scale.to_bits());
            assert_eq!(a.points.len(), b.points.len());
            for (a, b) in a.points.iter().zip(&b.points) {
                assert_eq!(
                    (a.x.to_bits(), a.y.to_bits()),
                    (b.x.to_bits(), b.y.to_bits())
                );
            }
        }
    }

    #[test]
    fn indexed_joining_preserves_original_groups_order_and_geometry() {
        for shift in [0.0, 16_777_216.0] {
            let mut chains = Vec::new();
            for row in 0..40 {
                for col in 0..9 {
                    let x = (col * 135) as f32;
                    let mut chain = segment(x, x + 30.0 + (col % 3) as f32, 30.0 * row as f32, 0.0);
                    for p in &mut chain.points {
                        // Mix horizontal, vertical, and both diagonal directions.
                        let (x, y) = (p.x, p.y);
                        *p = match row % 4 {
                            0 => Point::new(x + shift, y + shift),
                            1 => Point::new(y + shift, x + shift),
                            2 => Point::new(0.8 * x - 0.6 * y + shift, 0.6 * x + 0.8 * y + shift),
                            _ => Point::new(0.6 * x + 0.8 * y + shift, -0.8 * x + 0.6 * y + shift),
                        };
                    }
                    if (row + col) % 2 == 0 {
                        chain.points.reverse();
                    }
                    chain.scale = 0.7 + (col % 4) as f32 * 0.3;
                    chains.push(chain);
                }
            }
            // Equal-gap alternatives must retain stable fragment rank.
            chains.extend([
                segment(0.0, 30.0, -50.0, 0.0),
                segment(60.0, 90.0, -49.0, 0.0),
                segment(60.0, 90.0, -51.0, 0.0),
            ]);
            assert_same_lines(
                &find_long_lines(&chains).chains,
                &find_long_lines_reference(&chains),
            );
        }
    }

    #[test]
    fn endpoint_queries_retain_boundary_candidates() {
        let mut chains = Vec::new();
        for gap in [-MAX_OVERLAP, MAX_GAP] {
            for gap in [gap.next_down(), gap, gap.next_up()] {
                for offset in [-MAX_OFFSET, MAX_OFFSET] {
                    for offset in [offset.next_down(), offset, offset.next_up()] {
                        chains.push(segment(40.0 + gap, 70.0 + gap, offset, 0.0));
                        chains.push(segment(-30.0 - gap, -gap, offset, 0.0));
                    }
                }
            }
        }
        // Keep an interior candidate when large-coordinate rounding moves all
        // of an oblique line's offset-boundary probes outside the tolerance.
        chains.push(segment(-30.0, 0.0, 0.0, 0.0));
        while chains.len() < INDEX_MIN_FRAGMENTS {
            let y = 1000.0 + chains.len() as f32 * 10.0;
            chains.push(segment(0.0, 30.0, y, 0.0));
        }
        for shift in [
            0.0,
            16_777_216.0,
            crate::encode_image::MAX_DIMENSION as f32 - 512.0,
        ] {
            for dir in [
                Point::new(1.0, 0.0),
                Point::new(-1.0, -0.0),
                Point::new(0.0, 1.0),
                Point::new(0.6, 0.8),
                Point::new(-0.8, 0.6),
            ] {
                let line = Line {
                    center: Point::new(shift, shift),
                    dir,
                };
                let rotated: Vec<_> = chains
                    .iter()
                    .map(|chain| Chain {
                        points: chain
                            .points
                            .iter()
                            .map(|p| {
                                Point::new(
                                    shift + (p.x * dir.x - p.y * dir.y),
                                    shift + (p.x * dir.y + p.y * dir.x),
                                )
                            })
                            .collect(),
                        scale: chain.scale,
                    })
                    .collect();
                // Query correctness is independent of the fragment's line fit.
                let fragments: Vec<_> = (0..rotated.len())
                    .map(|index| Fragment { index, line })
                    .collect();
                let index = EndpointIndex::new(&rotated, &fragments).unwrap();
                let mut candidates = Vec::new();
                index.candidates(&line, 0.0, 40.0, &mut candidates);
                let mut accepted = 0;
                for (rank, fragment) in fragments.iter().enumerate() {
                    let (a, b, offset) = placement(&line, &rotated[fragment.index]);
                    let gap = (a - 40.0).max(-b);
                    if offset <= MAX_OFFSET && (-MAX_OVERLAP..=MAX_GAP).contains(&gap) {
                        accepted += 1;
                        assert!(
                            candidates.binary_search(&rank).is_ok(),
                            "lost rank {rank}, gap {gap}, offset {offset}, shift {shift}"
                        );
                    }
                }
                assert!(accepted > 0);
            }
        }
    }

    #[test]
    fn collinear_fragments_join_across_gaps_in_either_direction() {
        let chains = [
            segment(10.0, 50.0, 20.0, 0.1),
            segment(200.0, 150.0, 20.0, 0.1),
            segment(80.0, 110.0, 20.0, 0.1),
        ];
        let lines = find_long_lines(&chains);
        assert_eq!(lines.chains.len(), 1);
        assert_eq!(lines.joined, [0]);
        let points = &lines.chains[0].points;
        assert!(
            (points[0].x - 10.0).abs() < 1e-3 && (points[points.len() - 1].x - 200.0).abs() < 1e-3
        );
        for pair in points.windows(2) {
            assert!(
                pair[1].x > pair[0].x,
                "points must run monotonically along the line"
            );
            assert!(
                (pair[1].x - pair[0].x) <= 1.5,
                "gaps must be bridged at about 1 px"
            );
        }
    }

    #[test]
    fn parallel_and_tilted_neighbours_stay_apart() {
        let chains = [
            segment(10.0, 60.0, 20.0, 0.0),
            segment(80.0, 130.0, 24.0, 0.0),
            segment(80.0, 130.0, 0.0, 0.3),
        ];
        assert!(find_long_lines(&chains).chains.is_empty());
    }

    #[test]
    fn a_long_straight_chain_is_a_line_on_its_own_and_a_short_one_is_not() {
        let lines = find_long_lines(&[
            segment(10.0, 100.0, 30.0, -0.2),
            segment(10.0, 45.0, 90.0, 0.0),
        ]);
        assert_eq!(lines.chains.len(), 1);
        assert_eq!(lines.chains[0].points.len(), 91);
        assert!(lines.joined.is_empty());
    }

    #[test]
    fn distant_fragments_are_not_joined() {
        let chains = [
            segment(10.0, 40.0, 20.0, 0.0),
            segment(260.0, 300.0, 20.0, 0.0),
        ];
        assert!(find_long_lines(&chains).chains.is_empty());
    }

    #[test]
    fn sparse_joins_and_pairs_keep_the_faint_line_fallback() {
        for chains in [
            vec![
                segment(10.0, 50.0, 20.0, 0.0),
                segment(60.0, 100.0, 20.0, 0.0),
            ],
            vec![
                segment(10.0, 30.0, 20.0, 0.0),
                segment(160.0, 180.0, 20.0, 0.0),
                segment(310.0, 330.0, 20.0, 0.0),
            ],
        ] {
            let lines = find_long_lines(&chains);
            assert_eq!(lines.chains.len(), 1);
            assert!(lines.joined.is_empty());
        }
    }
}
