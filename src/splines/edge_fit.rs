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

//! Bounded source-edge proposals using two ordinary additive JPEG XL splines.
//!
//! VarDCT keeps the broad transition. Opposite signed ribbons carry a narrow
//! normal high-pass correction. Profile coherence limits texture proposals;
//! one decoder-rendered constant-color fit supplies three amplitude choices.
//! This module proposes geometry and color; it never substitutes a proxy for
//! the caller's native transform, quantization and final-composite decision.

use super::edge_detect::{self, Contour};
use super::render::RenderPlan;
use super::{CHANNEL_WEIGHT, Point, QUANT_ADJUST, QuantizedSpline, adjusted_quant, round_i32};
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::entropy::pack_signed;
use crate::image::Image3F;
use std::collections::BTreeMap;

const MAX_CONTOURS: usize = 256;
const MAX_PAIRS: usize = 64;
const MAX_CONTROUR_POINTS: usize = 16_384;
const MAX_NORMAL_SAMPLES: usize = 2048;
const MAX_FIT_TILES: usize = 1024;
const MAX_CONTROL_POINTS: usize = 64;
const DP_SAMPLES: usize = 192;
const DP_WINDOW: usize = 96;
const DP_OPTION_REACH: f32 = 0.75;
// Bits per squared pixel of normal deviation: a pair only pays when its two
// ribbons sit on the edge, so placement wins over fewer points (1–4 lose,
// 16–32 tie, 64 loses on butteraugli).
const DP_LAMBDA: f64 = 16.0;
const DP_MAX_DEVIATION: f32 = 1.0;
const MAX_PROFILE_SECTIONS: usize = 32;
const PROFILE_REACH: i32 = 8;
const PROFILE_COHERENCE: f64 = 0.65;
const FIT_EXPLAINED: f64 = 0.05;
const UNIT: i32 = 1024;
const NO_NORMAL: u32 = u32::MAX;
static WEIGHTS: [f64; 3] = [14.0, 1.0, 0.6];
static WIDTHS: [i32; 4] = [2, 3, 4, 5];
static OFFSETS: [f32; 3] = [0.5, 1.0, 1.5];
static AMPLITUDES: [f32; 3] = [0.5, 0.75, 1.0];

#[derive(Clone, Debug)]
pub(super) struct EdgePair {
    pub(super) alts: Vec<[QuantizedSpline; 2]>,
}

struct Target {
    kernel: Vec<f32>,
}

impl Target {
    fn new() -> Self {
        let mut kernel: Vec<f32> = (-6..=6)
            .map(|j| (-0.5 * (j as f32 / 2.0).powi(2)).exp())
            .collect();
        let sum: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|value| *value /= sum);
        Self { kernel }
    }

    /// A one-dimensional normal blur preserves constant and linear ground,
    /// and avoids a three-plane full-frame high-pass allocation.
    fn at(&self, source: &Image3F, point: Point<f32>, normal: Point<f32>) -> [f32; 3] {
        let mut result = sample(source, point);
        for (j, &weight) in (-6..=6).zip(&self.kernel) {
            let value = sample(
                source,
                Point::new(point.x + j as f32 * normal.x, point.y + j as f32 * normal.y),
            );
            for c in 0..3 {
                result[c] -= weight * value[c];
            }
        }
        result
    }
}

fn sample(source: &Image3F, point: Point<f32>) -> [f32; 3] {
    let (w, h) = (source.xsize(), source.ysize());
    let (x, y) = (
        point.x.clamp(0.0, (w - 1) as f32),
        point.y.clamp(0.0, (h - 1) as f32),
    );
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    std::array::from_fn(|c| {
        let data = source.plane_data(c);
        let top = data[y0 * w + x0] * (1.0 - fx) + data[y0 * w + x1] * fx;
        let bottom = data[y1 * w + x0] * (1.0 - fx) + data[y1 * w + x1] * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

fn smooth(contour: &Contour) -> (Vec<Point<f32>>, Vec<Point<f32>>) {
    let mut points = Vec::with_capacity(contour.points.len());
    let mut normals = Vec::with_capacity(contour.points.len());
    for i in 0..contour.points.len() {
        let (mut p, mut normal, mut weight) =
            (Point::<f32>::default(), Point::<f32>::default(), 0.0);
        for j in i.saturating_sub(2)..=(i + 2).min(contour.points.len() - 1) {
            let v = (3 - i.abs_diff(j)) as f32;
            p.x += v * contour.points[j].x;
            p.y += v * contour.points[j].y;
            normal.x += v * contour.normals[j].x;
            normal.y += v * contour.normals[j].y;
            weight += v;
        }
        points.push(if i == 0 || i + 1 == contour.points.len() {
            contour.points[i]
        } else {
            Point::new(p.x / weight, p.y / weight)
        });
        let norm = normal.x.hypot(normal.y).max(1e-8);
        normals.push(Point::new(normal.x / norm, normal.y / norm));
    }
    (points, normals)
}

struct Geometry {
    points: [Vec<Point<i32>>; 2],
    sigma_n: i32,
    offset: f32,
    score: f64,
}

fn side_curve(
    points: &[Point<f32>],
    normals: &[Point<f32>],
    sign: f32,
    offset: f32,
) -> Vec<Point<f32>> {
    points
        .iter()
        .zip(normals)
        .map(|(p, n)| Point::new(p.x + sign * offset * n.x, p.y + sign * offset * n.y))
        .collect()
}

fn in_frame(curve: &[Point<f32>], w: usize, h: usize) -> bool {
    curve
        .iter()
        .all(|p| p.x >= 1.5 && p.y >= 1.5 && p.x < w as f32 - 2.5 && p.y < h as f32 - 2.5)
}

/// Bits of one double-delta coordinate under a flat adaptive-code guess.
fn delta_bits(value: i32) -> f64 {
    1.0 + 2.0 * (1.0 + pack_signed(value) as f64).log2()
}

const DELTA_TABLE_REACH: i32 = 1024;

/// `delta_bits` tabulated for the double deltas the DP prices in bulk.
static DELTA_TABLE: std::sync::LazyLock<Vec<f64>> = std::sync::LazyLock::new(|| {
    (-DELTA_TABLE_REACH..=DELTA_TABLE_REACH)
        .map(delta_bits)
        .collect()
});

fn table_delta_bits(table: &[f64], value: i32) -> f64 {
    table
        .get(value.wrapping_add(DELTA_TABLE_REACH) as usize)
        .copied()
        .unwrap_or_else(|| delta_bits(value))
}

struct DpState {
    previous: u32,
    cost: f64,
    parent: u32,
}

/// Integer control points for one dense side curve. A shortest path over
/// (previous, current) point states prices every point's double delta, as the
/// section codes it, against the squared chord distance of all skipped curve
/// samples. Straight runs become evenly spaced points with zero double deltas.
fn dp_controls(
    curve: &[Point<f32>],
    normals: &[Point<f32>],
    lambda: f64,
    max_deviation: f32,
) -> Option<Vec<Point<i32>>> {
    const NONE: u32 = u32::MAX;
    let n = curve.len();
    // Catmull-Rom leaves a control point along the secant of its neighbors,
    // so tangents are taken at the span's own scale; local tangents would
    // follow detection wiggles a long span averages out.
    let secant = |i: usize, reach: usize| {
        let (a, b) = (
            curve[i.saturating_sub(reach)],
            curve[(i + reach).min(n - 1)],
        );
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let length = dx.hypot(dy).max(1e-6);
        Point::new(dx / length, dy / length)
    };
    let normal_error = |i: usize, p: Point<f32>| {
        (curve[i].x - p.x) * normals[i].x + (curve[i].y - p.y) * normals[i].y
    };
    let stride = n.div_ceil(DP_SAMPLES).max(1);
    let mut samples: Vec<usize> = (0..n).step_by(stride).collect();
    if samples.last() != Some(&(n - 1)) {
        samples.push(n - 1);
    }
    let mut nodes: Vec<(usize, Point<i32>)> = Vec::new();
    let mut first = Vec::with_capacity(samples.len() + 1);
    for (c, &i) in samples.iter().enumerate() {
        first.push(nodes.len());
        let p = curve[i];
        let (fx, fy) = (p.x.floor() as i32, p.y.floor() as i32);
        let mut options = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| {
            let q = Point::new(fx + dx, fy + dy);
            ((q.x as f32 - p.x).powi(2) + (q.y as f32 - p.y).powi(2), q)
        });
        options.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| (a.1.y, a.1.x).cmp(&(b.1.y, b.1.x)))
        });
        nodes.push((c, options[0].1));
        if options[1].0 <= DP_OPTION_REACH * DP_OPTION_REACH {
            nodes.push((c, options[1].1));
        }
    }
    first.push(nodes.len());
    let limit = max_deviation * max_deviation;
    // Include the far endpoint's own rounding in the span error.
    let span_error = |i0: usize, i1: usize, a: Point<i32>, b: Point<i32>, tangents| {
        let (ta, tb): (Point<f32>, Point<f32>) = tangents;
        let (ax, ay, bx, by) = (a.x as f32, a.y as f32, b.x as f32, b.y as f32);
        let length = (bx - ax).hypot(by - ay);
        let mut error = 0.0f64;
        for i in i0 + 1..=i1 {
            let u = (i - i0) as f32 / (i1 - i0) as f32;
            let (u2, u3) = (u * u, u * u * u);
            let (h00, h10, h01, h11) = (
                2.0 * u3 - 3.0 * u2 + 1.0,
                u3 - 2.0 * u2 + u,
                -2.0 * u3 + 3.0 * u2,
                u3 - u2,
            );
            let p = Point::new(
                h00 * ax + h10 * length * ta.x + h01 * bx + h11 * length * tb.x,
                h00 * ay + h10 * length * ta.y + h01 * by + h11 * length * tb.y,
            );
            let d = normal_error(i, p).powi(2);
            if d > limit {
                return None;
            }
            error += d as f64;
        }
        Some(error)
    };
    let mut states: Vec<Vec<DpState>> = (0..nodes.len()).map(|_| Vec::new()).collect();
    for q in first[0]..first[1] {
        let start = nodes[q].1;
        let d = normal_error(0, Point::new(start.x as f32, start.y as f32)).powi(2);
        states[q].push(DpState {
            previous: NONE,
            cost: lambda * d as f64,
            parent: NONE,
        });
    }
    let prices: &[f64] = &DELTA_TABLE;
    let mut incoming = Vec::new();
    for q in 0..nodes.len() {
        if states[q].is_empty() {
            continue;
        }
        let (cq, qp) = nodes[q];
        incoming.clear();
        incoming.extend(states[q].iter().map(|state| {
            let (pdx, pdy) = if state.previous == NONE {
                (0, 0)
            } else {
                let pp = nodes[state.previous as usize].1;
                (qp.x - pp.x, qp.y - pp.y)
            };
            (pdx, pdy, state.cost, state.previous)
        }));
        // Generate successors only for reachable nodes and consume each span
        // as soon as its error is known.
        for cr in cq + 1..samples.len().min(cq + 1 + DP_WINDOW) {
            let (i0, i1) = (samples[cq], samples[cr]);
            let reach = ((i1 - i0) / 2).max(1);
            let tangents = (secant(i0, reach), secant(i1, reach));
            let mut any = false;
            for r in first[cr]..first[cr + 1] {
                let rp = nodes[r].1;
                let Some(error) = span_error(i0, i1, qp, rp, tangents) else {
                    continue;
                };
                any = true;
                let (dx, dy) = (rp.x - qp.x, rp.y - qp.y);
                let mut best = (f64::INFINITY, NONE);
                for &(pdx, pdy, cost, previous) in &incoming {
                    let cost = cost
                        + table_delta_bits(prices, dx - pdx)
                        + table_delta_bits(prices, dy - pdy);
                    if cost < best.0 {
                        best = (cost, previous);
                    }
                }
                states[r].push(DpState {
                    previous: q as u32,
                    cost: best.0 + lambda * error,
                    parent: best.1,
                });
            }
            if !any {
                break;
            }
        }
    }
    let last = samples.len() - 1;
    let (mut node, mut state) = (first[last]..first[last + 1])
        .flat_map(|r| {
            states[r]
                .iter()
                .enumerate()
                .map(move |(k, s)| (r, k, s.cost))
        })
        .min_by(|a, b| {
            a.2.total_cmp(&b.2)
                .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        })
        .map(|(r, k, _)| (r, k))?;
    let mut out = vec![nodes[node].1];
    loop {
        let s = &states[node][state];
        if s.previous == NONE {
            break;
        }
        let previous = s.previous as usize;
        let parent = s.parent;
        out.push(nodes[previous].1);
        state = states[previous].iter().position(|t| t.previous == parent)?;
        node = previous;
    }
    out.reverse();
    out.dedup();
    Some(out)
}

fn build_points(
    points: &[Point<f32>],
    normals: &[Point<f32>],
    offset: f32,
    w: usize,
    h: usize,
) -> Option<[Vec<Point<i32>>; 2]> {
    let mut sides: [Vec<Point<i32>>; 2] = Default::default();
    for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
        let curve = side_curve(points, normals, sign, offset);
        if !in_frame(&curve, w, h) {
            return None;
        }
        sides[side] = dp_controls(&curve, normals, DP_LAMBDA, DP_MAX_DEVIATION)?;
        if sides[side].len() > MAX_CONTROL_POINTS
            || sides[side]
                .iter()
                .any(|p| p.x < 2 || p.y < 2 || p.x >= w as i32 - 2 || p.y >= h as i32 - 2)
        {
            return None;
        }
    }
    if sides.iter().any(|side| side.len() < 2) || sides[0] == sides[1] {
        return None;
    }
    Some(sides)
}

fn profile_gain(profiles: &[[f32; 3]], sigma: f32, offset: f32) -> f64 {
    let (mut dot, mut norm) = ([0.0; 3], 0.0f64);
    for (i, target) in profiles.iter().enumerate() {
        let t = (i % (2 * PROFILE_REACH + 1) as usize) as f32 * 0.5 - PROFILE_REACH as f32 * 0.5;
        let value = ((-0.5 * ((t - offset) / sigma).powi(2)).exp()
            - (-0.5 * ((t + offset) / sigma).powi(2)).exp()) as f64;
        norm += value * value;
        for c in 0..3 {
            dot[c] += value * target[c] as f64;
        }
    }
    (0..3)
        .map(|c| WEIGHTS[c] * dot[c] * dot[c] / norm.max(1e-12))
        .sum()
}

fn profile_energy(profiles: &[[f32; 3]]) -> f64 {
    profiles
        .iter()
        .map(|v| {
            (0..3)
                .map(|c| WEIGHTS[c] * (v[c] as f64).powi(2))
                .sum::<f64>()
        })
        .sum()
}

/// Only the best coherent width/offset receives an expensive decoder fit.
fn shortlist(source: &Image3F, target: &Target, contour: &Contour) -> Option<Geometry> {
    if contour.points.len() > MAX_CONTROUR_POINTS || !contour.strength.is_finite() {
        return None;
    }
    let (points, normals) = smooth(contour);
    let mut profiles = Vec::new();
    for i in (0..points.len()).step_by(points.len().div_ceil(MAX_PROFILE_SECTIONS).max(1)) {
        profiles.extend((-PROFILE_REACH..=PROFILE_REACH).map(|j| {
            let t = j as f32 * 0.5;
            target.at(
                source,
                Point::new(
                    points[i].x + t * normals[i].x,
                    points[i].y + t * normals[i].y,
                ),
                normals[i],
            )
        }));
    }
    let energy = profile_energy(&profiles);
    if energy <= 1e-12 {
        return None;
    }
    let (w, h) = (source.xsize(), source.ysize());
    let mut best: Option<Geometry> = None;
    for width in WIDTHS {
        let sigma = width as f32 * CHANNEL_WEIGHT[3] / adjusted_quant(QUANT_ADJUST);
        for offset in OFFSETS {
            let gain = profile_gain(&profiles, sigma, offset);
            if gain / energy < PROFILE_COHERENCE
                || best.as_ref().is_some_and(|old| old.score >= gain)
                || [1.0, -1.0]
                    .into_iter()
                    .any(|sign| !in_frame(&side_curve(&points, &normals, sign, offset), w, h))
            {
                continue;
            }
            best = Some(Geometry {
                points: Default::default(),
                sigma_n: width,
                offset,
                score: gain,
            });
        }
    }
    if let Some(geometry) = &mut best {
        let sections = profiles.len() / (2 * PROFILE_REACH + 1) as usize;
        geometry.score *= contour.points.len() as f64 / sections.max(1) as f64;
    }
    best
}

struct BasisTile {
    value: [f32; 64],
    nearest: [u32; 64],
    distance: [f32; 64],
}

impl Default for BasisTile {
    fn default() -> Self {
        Self {
            value: [0.0; 64],
            nearest: [NO_NORMAL; 64],
            distance: [f32::INFINITY; 64],
        }
    }
}

/// The decoder raster includes DC normalization, quantization weights and its
/// exact Gaussian profile. Normalize a large X-only coefficient into a basis
/// shared by X/Y/B; a large unit retains the tails of real color amplitudes.
fn raster_basis(
    ctx: &EncodingContext,
    geometry: &Geometry,
    w: usize,
    h: usize,
) -> Option<BTreeMap<usize, BasisTile>> {
    let scale = UNIT as f32 * CHANNEL_WEIGHT[0] / adjusted_quant(QUANT_ADJUST);
    let mut basis = BTreeMap::<usize, BasisTile>::new();
    for side in 0..2 {
        let mut sp = QuantizedSpline {
            points: geometry.points[side].clone(),
            dct: [[0; 32]; 4],
        };
        sp.dct[0][0] = if side == 0 { UNIT } else { -UNIT };
        sp.dct[3][0] = geometry.sigma_n;
        let plan = RenderPlan::new(ctx, &sp, QUANT_ADJUST, w, h).tiled();
        if plan.tiles.len() > MAX_FIT_TILES {
            return None;
        }
        for tile in &plan.tiles {
            let mut pixels = [[0.0; 64]; 3];
            plan.draw_tile(tile, &mut pixels, 1.0);
            let output = basis.entry(tile.cell).or_default();
            for (out, &value) in output.value.iter_mut().zip(&pixels[0]) {
                *out += value / scale;
            }
        }
        if basis.len() > MAX_FIT_TILES {
            return None;
        }
    }
    Some(basis)
}

fn nearest_normals(basis: &mut BTreeMap<usize, BasisTile>, contour: &Contour, w: usize, h: usize) {
    let blocks_w = w.div_ceil(8);
    let stride = contour.points.len().div_ceil(MAX_NORMAL_SAMPLES).max(1);
    // Fixed narrow bands fit inside this radius, including unit-render tails.
    let reach = 12isize;
    for i in (0..contour.points.len()).step_by(stride) {
        let p = contour.points[i];
        let (cx, cy) = (p.x.round() as isize, p.y.round() as isize);
        let (x0, y0) = ((cx - reach).max(0) as usize, (cy - reach).max(0) as usize);
        let (x1, y1) = (
            (cx + reach).min(w as isize - 1) as usize,
            (cy + reach).min(h as isize - 1) as usize,
        );
        for by in y0 / 8..=y1 / 8 {
            for bx in x0 / 8..=x1 / 8 {
                let Some(tile) = basis.get_mut(&(by * blocks_w + bx)) else {
                    continue;
                };
                for y in y0.max(by * 8)..=y1.min(by * 8 + 7) {
                    for x in x0.max(bx * 8)..=x1.min(bx * 8 + 7) {
                        let k = (y - by * 8) * 8 + x - bx * 8;
                        let distance = (x as f32 - p.x).powi(2) + (y as f32 - p.y).powi(2);
                        if distance < tile.distance[k] {
                            tile.distance[k] = distance;
                            tile.nearest[k] = i as u32;
                        }
                    }
                }
            }
        }
    }
}

fn pair_from_color(
    geometry: &Geometry,
    color: [i32; 3],
    amplitude: f32,
) -> Option<[QuantizedSpline; 2]> {
    let color = color.map(|value| round_i32(value as f32 * amplitude));
    if color.iter().all(|&value| value == 0) {
        return None;
    }
    Some(std::array::from_fn(|side| {
        let mut dct = [[0; 32]; 4];
        for c in 0..3 {
            dct[c][0] = if side == 0 { color[c] } else { -color[c] };
        }
        dct[3][0] = geometry.sigma_n;
        QuantizedSpline {
            points: geometry.points[side].clone(),
            dct,
        }
    }))
}

fn fit(
    ctx: &EncodingContext,
    source: &Image3F,
    target: &Target,
    contour: &Contour,
    geometry: &Geometry,
) -> Option<EdgePair> {
    let (w, h) = (source.xsize(), source.ysize());
    let (points, normals) = smooth(contour);
    let geometry = &Geometry {
        points: build_points(&points, &normals, geometry.offset, w, h)?,
        sigma_n: geometry.sigma_n,
        offset: geometry.offset,
        score: geometry.score,
    };
    let mut basis = raster_basis(ctx, geometry, w, h)?;
    nearest_normals(&mut basis, contour, w, h);
    let (mut norm, mut rhs, mut energy) = (0.0f64, [0.0f64; 3], 0.0f64);
    for (cell, tile) in basis {
        let (x0, y0) = (cell % w.div_ceil(8) * 8, cell / w.div_ceil(8) * 8);
        for j in 0..64 {
            let (x, y) = (x0 + j % 8, y0 + j / 8);
            let value = tile.value[j] as f64;
            if x >= w || y >= h || tile.nearest[j] == NO_NORMAL || value.abs() < 1e-7 {
                continue;
            }
            let t = target.at(
                source,
                Point::new(x as f32, y as f32),
                contour.normals[tile.nearest[j] as usize],
            );
            norm += value * value;
            for c in 0..3 {
                rhs[c] += value * t[c] as f64;
                energy += WEIGHTS[c] * (t[c] as f64).powi(2);
            }
        }
    }
    if norm < 1e-12 {
        return None;
    }
    let solution = rhs.map(|value| value / norm);
    let step: [f64; 3] =
        std::array::from_fn(|c| (CHANNEL_WEIGHT[c] / adjusted_quant(QUANT_ADJUST)) as f64);
    let q = |value: f64| value.round().clamp(-32767.0, 32767.0) as i32;
    let x = q(solution[0] / step[0]);
    let y = q(solution[1] / step[1]);
    let b = q((solution[2] - y as f64 * step[1]) / step[2]);
    let restored = [
        x as f64 * step[0],
        y as f64 * step[1],
        b as f64 * step[2] + y as f64 * step[1],
    ];
    let gain: f64 = (0..3)
        .map(|c| WEIGHTS[c] * (2.0 * restored[c] * rhs[c] - restored[c] * restored[c] * norm))
        .sum();
    if !gain.is_finite() || gain <= (energy * FIT_EXPLAINED).max(1e-7) {
        return None;
    }
    let mut alts = Vec::new();
    for amplitude in AMPLITUDES {
        if let Some(pair) = pair_from_color(geometry, [x, y, b], amplitude)
            && !alts
                .iter()
                .any(|old: &[QuantizedSpline; 2]| old[0].dct == pair[0].dct)
        {
            alts.push(pair);
        }
    }
    (!alts.is_empty()).then_some(EdgePair { alts })
}

/// `detectable` marks the pixels whose edges the decoder shows (all if `None`).
pub(super) fn find_pairs(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    source: &Image3F,
    detectable: Option<&[bool]>,
) -> Vec<EdgePair> {
    let contours = edge_detect::detect(ctx, scratch, source, detectable, MAX_CONTOURS);
    let target = Target::new();
    let options = ctx.thread_pool.steal_map(scratch, contours.len(), |i, _| {
        shortlist(source, &target, &contours[i])
    });
    let mut selected: Vec<_> = options
        .into_iter()
        .enumerate()
        .filter_map(|(i, g)| g.map(|g| (i, g)))
        .collect();
    selected.sort_by(|(ia, a), (ib, b)| b.score.total_cmp(&a.score).then_with(|| ia.cmp(ib)));
    selected.truncate(MAX_PAIRS);
    ctx.thread_pool
        .steal_map(scratch, selected.len(), |i, _| {
            let (index, geometry) = &selected[i];
            fit(ctx, source, &target, &contours[*index], geometry)
        })
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_geometry() -> Geometry {
        Geometry {
            points: [
                vec![Point::new(20, 63), Point::new(108, 63)],
                vec![Point::new(20, 65), Point::new(108, 65)],
            ],
            sigma_n: 4,
            offset: 1.0,
            score: 1.0,
        }
    }

    #[test]
    fn normal_highpass_preserves_affine_background() {
        let mut image = Image3F::new(96, 96);
        for y in 0..96 {
            for x in 0..96 {
                for c in 0..3 {
                    image.plane_row_mut(c, y)[x] =
                        0.1 * c as f32 + 0.002 * x as f32 + 0.003 * y as f32;
                }
            }
        }
        assert!(
            Target::new()
                .at(&image, Point::new(40.3, 51.1), Point::new(0.6, 0.8))
                .iter()
                .all(|v| v.abs() < 1e-6)
        );
    }

    #[test]
    fn decoder_basis_matches_quantized_pair_and_opposite_color() {
        let ctx = EncodingContext::default();
        let geometry = straight_geometry();
        let basis = raster_basis(&ctx, &geometry, 128, 128).unwrap();
        let pair = pair_from_color(&geometry, [5, 4, -1], 1.0).unwrap();
        let mut rendered = Image3F::new(128, 128);
        for sp in &pair {
            RenderPlan::new(&ctx, sp, QUANT_ADJUST, 128, 128).draw(&mut rendered, 1.0);
        }
        let step = CHANNEL_WEIGHT.map(|weight| weight / adjusted_quant(QUANT_ADJUST));
        let color = [5.0 * step[0], 4.0 * step[1], -step[2] + 4.0 * step[1]];
        let mut max_error = 0.0f32;
        for (cell, tile) in basis {
            let (x0, y0) = (cell % 16 * 8, cell / 16 * 8);
            for j in 0..64 {
                for c in 0..3 {
                    max_error = max_error.max(
                        (tile.value[j] * color[c] - rendered.plane_row(c, y0 + j / 8)[x0 + j % 8])
                            .abs(),
                    );
                }
            }
        }
        assert!(max_error < 3e-5, "decoder basis mismatch {max_error}");
        for c in 0..3 {
            assert_eq!(pair[0].dct[c][0], -pair[1].dct[c][0]);
        }
        assert_eq!(pair[0].dct[3], pair[1].dct[3]);
    }

    #[test]
    fn amplitude_choices_keep_width_and_geometry_and_drop_zero() {
        let geometry = straight_geometry();
        assert!(pair_from_color(&geometry, [0; 3], 1.0).is_none());
        let half = pair_from_color(&geometry, [8, -4, 2], 0.5).unwrap();
        let full = pair_from_color(&geometry, [8, -4, 2], 1.0).unwrap();
        for side in 0..2 {
            assert_eq!(half[side].points, full[side].points);
            assert_eq!(half[side].dct[3], full[side].dct[3]);
            for c in 0..3 {
                assert_eq!(half[side].dct[c][0] * 2, full[side].dct[c][0]);
            }
        }
    }

    #[test]
    fn incoherent_normal_residuals_have_no_shared_profile_gain() {
        let section: Vec<_> = (-PROFILE_REACH..=PROFILE_REACH)
            .map(|j| {
                let t = j as f32 * 0.5;
                [
                    0.0,
                    (-0.5 * ((t - 1.0) / 0.8).powi(2)).exp()
                        - (-0.5 * ((t + 1.0) / 0.8).powi(2)).exp(),
                    0.0,
                ]
            })
            .collect();
        let coherent = profile_gain(&section, 0.8, 1.0) / profile_energy(&section);
        assert!(coherent > 0.99);
        let incoherent: Vec<_> = section
            .iter()
            .copied()
            .chain(section.iter().map(|v| v.map(|x| -x)))
            .collect();
        assert!(profile_gain(&incoherent, 0.8, 1.0) / profile_energy(&incoherent) < 1e-12);
    }

    #[test]
    fn constant_image_has_no_pairs() {
        let image = Image3F::new(96, 96);
        assert!(
            find_pairs(
                &EncodingContext::default(),
                &mut CoderScratch::default(),
                &image,
                None
            )
            .is_empty()
        );
    }

    #[test]
    fn coherent_step_produces_bounded_deterministic_pairs() {
        let (w, h) = (129usize, 113usize);
        let mut source = Image3F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let value = 0.4 + 0.2 * ((y as f32 - 35.3 - 0.31 * x as f32) / 0.7).tanh();
                source.plane_row_mut(1, y)[x] = value;
                source.plane_row_mut(2, y)[x] = value;
            }
        }
        let run = |threads| {
            let ctx = EncodingContext::new(
                crate::Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                crate::xyb::XybMatrix::SPEC,
                2.0,
                threads,
            );
            find_pairs(&ctx, &mut CoderScratch::default(), &source, None)
        };
        let first = run(1);
        let parallel = run(4);
        assert!(!first.is_empty() && first.len() <= MAX_PAIRS);
        assert_eq!(first.len(), parallel.len());
        for (a, b) in first.iter().zip(parallel) {
            assert!(!a.alts.is_empty() && a.alts.len() <= AMPLITUDES.len());
            assert_eq!(a.alts.len(), b.alts.len());
            for (a, b) in a.alts.iter().zip(b.alts) {
                for side in 0..2 {
                    assert_eq!(a[side].points, b[side].points);
                    assert_eq!(a[side].dct, b[side].dct);
                    assert!(a[side].points.len() <= MAX_CONTROL_POINTS);
                }
            }
        }
    }
}

#[cfg(test)]
mod dp_tests {
    use super::*;

    fn normals_of(curve: &[Point<f32>]) -> Vec<Point<f32>> {
        (0..curve.len())
            .map(|i| {
                let (a, b) = (
                    curve[i.saturating_sub(1)],
                    curve[(i + 1).min(curve.len() - 1)],
                );
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                let length = dx.hypot(dy);
                Point::new(-dy / length, dx / length)
            })
            .collect()
    }

    /// Largest distance from a dense curve sample to the decoder's curve.
    fn rendered_deviation(curve: &[Point<f32>], controls: &[Point<i32>]) -> f32 {
        let controls: Vec<_> = controls
            .iter()
            .map(|p| Point::new(p.x as f32, p.y as f32))
            .collect();
        let rendered = crate::splines::catmull_rom(&controls);
        curve
            .iter()
            .map(|p| {
                rendered
                    .windows(2)
                    .map(|w| {
                        let (dx, dy) = (w[1].x - w[0].x, w[1].y - w[0].y);
                        let l2 = (dx * dx + dy * dy).max(1e-9);
                        let t = (((p.x - w[0].x) * dx + (p.y - w[0].y) * dy) / l2).clamp(0.0, 1.0);
                        (p.x - w[0].x - t * dx).hypot(p.y - w[0].y - t * dy)
                    })
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn dp_controls_follow_lines_arcs_and_wiggles_with_few_points() {
        let line: Vec<_> = (0..120)
            .map(|i| Point::new(10.3 + i as f32, 20.6 + i as f32 / 3.0))
            .collect();
        let arc: Vec<_> = (0..200)
            .map(|i| {
                let t = i as f32 / 199.0 * 1.5;
                Point::new(100.0 + 80.0 * t.cos(), 100.0 + 80.0 * t.sin())
            })
            .collect();
        // Detection jitter must not buy control points.
        let wiggle: Vec<_> = (0..200)
            .map(|i| Point::new(10.0 + i as f32, 50.0 + 0.3 * (i as f32 * 0.9).sin()))
            .collect();
        for (curve, most) in [(&line, 5), (&arc, 7), (&wiggle, 6)] {
            let controls =
                dp_controls(curve, &normals_of(curve), DP_LAMBDA, DP_MAX_DEVIATION).unwrap();
            assert!(
                controls.len() >= 2 && controls.len() <= most,
                "{controls:?}"
            );
            assert!(rendered_deviation(curve, &controls) < 1.2);
        }
    }

    #[test]
    fn dp_controls_space_a_straight_run_for_zero_double_deltas() {
        let line: Vec<_> = (0..241)
            .map(|i| Point::new(8.0 + i as f32, 30.0 + i as f32 / 2.0))
            .collect();
        let controls = dp_controls(&line, &normals_of(&line), DP_LAMBDA, DP_MAX_DEVIATION).unwrap();
        let deltas: Vec<_> = controls
            .windows(2)
            .map(|w| (w[1].x - w[0].x, w[1].y - w[0].y))
            .collect();
        assert!(
            deltas.windows(2).filter(|d| d[0] != d[1]).count() <= 1,
            "{controls:?}"
        );
        assert_eq!(controls[0], Point::new(8, 30));
        assert_eq!(*controls.last().unwrap(), Point::new(248, 150));
    }
}
