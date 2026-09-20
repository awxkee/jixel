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
//! gaps, and handed to the fitter as one chain; the colour DCT can vanish in
//! the gaps and change sign along the way.

use super::Point;
use super::detect::Chain;

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
        let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        Line {
            center: Point::new(cx, cy),
            dir: Point::new(angle.cos(), angle.sin()),
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
            let gap = (first.x - last.x).hypot(first.y - last.y) as usize;
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

/// Long straight lines among `chains`: groups of collinear fragments, plus long
/// straight chains that found no partner.
pub(super) fn find_long_lines(chains: &[Chain]) -> Vec<Chain> {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn collinear_fragments_join_across_gaps_in_either_direction() {
        let chains = [
            segment(10.0, 50.0, 20.0, 0.1),
            segment(200.0, 150.0, 20.0, 0.1),
            segment(80.0, 110.0, 20.0, 0.1),
        ];
        let lines = find_long_lines(&chains);
        assert_eq!(lines.len(), 1);
        let points = &lines[0].points;
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
        assert!(find_long_lines(&chains).is_empty());
    }

    #[test]
    fn a_long_straight_chain_is_a_line_on_its_own_and_a_short_one_is_not() {
        let lines = find_long_lines(&[
            segment(10.0, 100.0, 30.0, -0.2),
            segment(10.0, 45.0, 90.0, 0.0),
        ]);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].points.len(), 91);
    }

    #[test]
    fn distant_fragments_are_not_joined() {
        let chains = [
            segment(10.0, 40.0, 20.0, 0.0),
            segment(260.0, 300.0, 20.0, 0.0),
        ];
        assert!(find_long_lines(&chains).is_empty());
    }
}
