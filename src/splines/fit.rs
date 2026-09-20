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

//! Spline fitting: the decoder's render is linear in the color DCT, so color
//! is one least-squares solve per candidate width; widths live on the format's
//! quantized lattice and are seeded from the ridge scale; integer control
//! points are refined by +-1 px coordinate descent. Every candidate leaves with
//! a few alternative parametrisations for the RD gate to choose from.

use super::detect::{Chain, gaussian_kernels};
use super::filter::FilterPlan;
use super::geometry::Polyline;
use super::select::{BlockModel, pretest};
use super::{
    CHANNEL_WEIGHT, Point, QUANT_ADJUST, QuantizedSpline, adjusted_quant, blob_weight,
    catmull_rom_spans, fast_hypot, round_i32, sample_curve,
};
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

static SIGMA_LATTICE: [i32; 8] = [1, 2, 3, 4, 5, 6, 8, 10];
const SEEDED_WIDTHS: usize = 2;
const SCALE_TO_SIGMA: (f32, f32) = (0.86, -0.21);
static FIT_CHANNEL_WEIGHT: [f64; 3] = [14.0, 1.0, 0.6];
const MIN_EXPLAINED: f64 = 0.5;
const REFINE_MIN_EXPLAINED: f64 = 0.6;
const MIN_ARC: f32 = 4.0;
const ARC_PER_COEFF: f32 = 10.0;
const LEAN_ARC_PER_COEFF: f32 = 30.0;
const RDP_EPS: f32 = 0.6;
const MAX_CONTROL_GAP: usize = 24;
/// Alternative geometries: control points inserted until the traced ridge lies
/// within this many pixels of the Catmull-Rom curve the decoder draws.
static CURVE_TOLERANCES: [f32; 2] = [1.0, 1.5];
const MAX_CURVE_POINTS: usize = 24;
const BORDER: i32 = 3;
const BACKGROUND_SIGMA: f32 = 3.0;
const MASK_RADIUS: isize = 2;
const MAX_COEFFS: usize = 32;
/// Refinement only needs the alignment signal: few color coefficients, tight window.
const REFINE_COEFFS: usize = 3;
const COMPACT_COEFFS: usize = 6;
const REFINE_NEIGHBOURS: usize = 3;
const WEIGHT_LUT_STEPS: f32 = 32.0;
/// Unrefined fits slightly below `MIN_EXPLAINED` may still get there after refinement.
const QUICK_MIN_EXPLAINED: f64 = 0.45;
/// Long faint lines get the cheapest representation there is: sparse control
/// points, luma only, one to three coefficients. Anything richer fits grain.
const LONG_LINE_MIN_SPACING: f32 = 64.0;
const LONG_LINE_SPANS: f32 = 3.0;
const LONG_LINE_MAX_COEFFS: usize = 3;
static LONG_LINE_WIDTHS: [i32; 5] = [2, 3, 4, 5, 6];
/// A long line touches many blocks, and the DCT8 proxy's optimism adds up over
/// them: on textured ground lean long splines must clear a higher bar.
const LONG_LINE_BITS_FACTOR: f32 = 4.0;

/// Alternative parametrisations of one detected line; the gate keeps at most one.
pub(super) struct Candidate {
    pub(super) alts: Vec<QuantizedSpline>,
    /// Extra factor on the spline bits in the RD gate.
    pub(super) bits_factor: f32,
}

#[inline]
fn sigma_step() -> f32 {
    CHANNEL_WEIGHT[3] / adjusted_quant(QUANT_ADJUST)
}

/// `image - background`, the background being a normalized convolution that
/// ignores the detected line pixels.
fn line_targets(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
    chains: &[Chain],
) -> [Vec<f32>; 3] {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let mut keep = vec![1f32; w * h];
    for chain in chains {
        for &Point { x: px, y: py } in &chain.points {
            let (cx, cy) = (px as isize, py as isize);
            for dy in -MASK_RADIUS..=MASK_RADIUS {
                for dx in -(MASK_RADIUS - dy.abs())..=(MASK_RADIUS - dy.abs()) {
                    let (x, y) = (cx + dx, cy + dy);
                    if x >= 0 && y >= 0 && x < w as isize && y < h as isize {
                        keep[y as usize * w + x as usize] = 0.0;
                    }
                }
            }
        }
    }
    let [kernel, _, _] = gaussian_kernels(BACKGROUND_SIGMA);
    let horizontal = FilterPlan::new(w, &kernel);
    let vertical = FilterPlan::new(h, &kernel);
    let mut rows = vec![0.0; w * h];
    let mut blur = |plane: &[f32]| {
        let mut out = vec![0.0; w * h];
        horizontal.horizontal(ctx, scratch, plane, &mut rows);
        vertical.vertical(ctx, scratch, &rows, &mut out);
        out
    };
    let weight = blur(&keep);
    [0, 1, 2].map(|c| {
        let src = xyb.plane_data(c);
        let masked: Vec<f32> = src.iter().zip(&keep).map(|(&v, &m)| v * m).collect();
        let smooth = blur(&masked);
        src.iter()
            .zip(smooth.iter().zip(&weight))
            .map(|(&v, (&s, &n))| v - s / (n + 1e-6))
            .collect()
    })
}

fn smooth_chain(points: &[Point<f32>], sigma: f32) -> Vec<Point<f32>> {
    let [kernel, _, _] = gaussian_kernels(sigma);
    let radius = kernel.len() / 2;
    let n = points.len();
    (0..n)
        .map(|i| {
            let (mut x, mut y) = (0f32, 0f32);
            for (j, &k) in kernel.iter().enumerate() {
                let p = points[(i + j).saturating_sub(radius).min(n - 1)];
                x += k * p.x;
                y += k * p.y;
            }
            Point::new(x, y)
        })
        .collect()
}

fn push_rounded(out: &mut Vec<Point<i32>>, p: Point<f32>) {
    let q = Point::new(round_i32(p.x), round_i32(p.y));
    if out.last() != Some(&q) {
        out.push(q);
    }
}

/// Douglas-Peucker control points, with a cap on the span between neighbors
/// (Catmull-Rom overshoots on long spans).
fn rdp_control_points(chain: &[Point<f32>]) -> Vec<Point<i32>> {
    let sm = smooth_chain(chain, 1.5);
    let n = sm.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (p0, p1) = (sm[a], sm[b]);
        let d = (p1.x - p0.x, p1.y - p0.y);
        let len = fast_hypot(d.0, d.1);
        let (mut worst, mut worst_dist) = (a + 1, -1f32);
        for (i, p) in sm.iter().enumerate().take(b).skip(a + 1) {
            let dist = if len < 1e-9 {
                fast_hypot(p.x - p0.x, p.y - p0.y)
            } else {
                (d.0 * (p.y - p0.y) - d.1 * (p.x - p0.x)).abs() / len
            };
            if dist > worst_dist {
                (worst, worst_dist) = (i, dist);
            }
        }
        if worst_dist > RDP_EPS {
            keep[worst] = true;
            stack.push((a, worst));
            stack.push((worst, b));
        }
    }
    let mut out = Vec::new();
    let mut last = 0usize;
    push_rounded(&mut out, sm[0]);
    for i in (1..n).filter(|&i| keep[i]) {
        while i - last > MAX_CONTROL_GAP {
            last += MAX_CONTROL_GAP;
            push_rounded(&mut out, sm[last]);
        }
        push_rounded(&mut out, sm[i]);
        last = i;
    }
    out
}

/// Control points chosen against the curve the decoder actually draws: start
/// from the end points and insert the chain point farthest from the
/// Catmull-Rom curve until the whole chain lies within `tol` px of it.
/// Douglas-Peucker measures against the polyline instead and over-samples
/// every smooth bend.
fn curve_control_points(ctx: &EncodingContext, chain: &[Point<f32>], tol: f32) -> Vec<Point<i32>> {
    let sm = smooth_chain(chain, 1.5);
    let mut idx = vec![0usize, sm.len() - 1];
    loop {
        let mut ctrl: Vec<Point<i32>> = Vec::with_capacity(idx.len());
        for &i in &idx {
            push_rounded(&mut ctrl, sm[i]);
        }
        if ctrl.len() < 2 || ctrl.len() != idx.len() || idx.len() >= MAX_CURVE_POINTS {
            return ctrl;
        }
        let control: Vec<Point<f32>> = ctrl.iter().map(|p| p.as_f32()).collect();
        let curve = super::catmull_rom(&control);
        let curve_distance = Polyline::new(ctx, &curve);
        let (mut worst, mut worst_d) = (0usize, tol * tol);
        for (k, &p) in sm.iter().enumerate() {
            let d = curve_distance.distance(p);
            if d > worst_d && !idx.contains(&k) {
                (worst, worst_d) = (k, d);
            }
        }
        if worst_d <= tol * tol {
            return ctrl;
        }
        let at = idx.partition_point(|&i| i < worst);
        idx.insert(at, worst);
    }
}

/// Remove redundant controls from an already aligned curve. Compare both
/// directions: checking only the new curve would allow shortcuts across bends.
/// Each deletion changes at most four spans, so no color fitting is needed.
fn prune_control_points(
    ctx: &EncodingContext,
    points: &[Point<i32>],
    tolerance: f32,
) -> Vec<Point<i32>> {
    let reference: Vec<_> = points.iter().map(|p| p.as_f32()).collect();
    let reference = super::catmull_rom(&reference);
    let reference_distance = Polyline::new(ctx, &reference);
    let mut indices: Vec<usize> = (0..points.len()).collect();
    let mut points = points.to_vec();
    let mut errors = vec![None; points.len()];
    loop {
        let mut best = None;
        let mut best_error = tolerance * tolerance;
        for i in 1..points.len() - 1 {
            let error = *errors[i].get_or_insert_with(|| {
                let mut control: Vec<_> = points.iter().map(|p| p.as_f32()).collect();
                control.remove(i);
                let lo = i.saturating_sub(2);
                let hi = (i + 1).min(control.len() - 1);
                let curve = catmull_rom_spans(&control, lo..hi);
                let curve_distance = Polyline::new(ctx, &curve);
                let original_hi = if hi >= i { hi + 1 } else { hi };
                let range = indices[lo] * 16..indices[original_hi] * 16;
                let original = &reference[range.start..=range.end];
                let mut error = 0.0f32;
                for distance in curve
                    .iter()
                    .map(|&p| reference_distance.distance_range(p, range.clone()))
                    .chain(original.iter().map(|&p| curve_distance.distance(p)))
                {
                    error = error.max(distance);
                    if error >= tolerance * tolerance {
                        break;
                    }
                }
                error
            });
            if error < best_error {
                best_error = error;
                best = Some(i);
            }
        }
        let Some(i) = best else { break };
        points.remove(i);
        indices.remove(i);
        errors.remove(i);
        // A deletion only changes the tangents of neighbouring spans.
        let end = (i + 4).min(errors.len());
        errors[i.saturating_sub(3)..end].fill(None);
    }
    points
}

/// Equal arc spacing keeps the double-delta coordinate tokens small.
fn uniform_control_points(chain: &[Point<f32>], spacing: f32) -> Vec<Point<i32>> {
    let sm = smooth_chain(chain, 1.5);
    let mut cum = vec![0f32; sm.len()];
    let mut length = 0.0;
    for (distance, [a, b]) in cum[1..].iter_mut().zip(sm.array_windows::<2>()) {
        length += fast_hypot(b.x - a.x, b.y - a.y);
        *distance = length;
    }
    let total = cum[sm.len() - 1];
    let n = ((total / spacing + 0.5) as usize).max(1);
    let mut out = Vec::new();
    let mut seg = 0usize;
    for k in 0..=n {
        let target = total * k as f32 / n as f32;
        while seg + 2 < sm.len() && cum[seg + 1] < target {
            seg += 1;
        }
        let span = (cum[seg + 1] - cum[seg]).max(1e-9);
        let f = ((target - cum[seg]) / span).clamp(0.0, 1.0);
        push_rounded(
            &mut out,
            Point::new(
                sm[seg].x + f * (sm[seg + 1].x - sm[seg].x),
                sm[seg].y + f * (sm[seg + 1].y - sm[seg].y),
            ),
        );
    }
    out
}

struct Fit {
    err: f64,
    e0: f64,
    sigma_n: i32,
    coeffs: usize,
    arc: f32,
    /// Dequantized-convention color DCT per channel (X, Y, B).
    sol: [[f64; MAX_COEFFS]; 3],
}

impl Fit {
    fn gain(&self) -> f64 {
        self.e0 - self.err
    }

    fn explained(&self) -> f64 {
        1.0 - self.err / self.e0.max(1e-12)
    }
}

#[derive(Default)]
pub(crate) struct FitScratch {
    slots: Vec<i32>,
    rows: Vec<f64>,
    pixels: Vec<u32>,
    weight_luts: Vec<(i32, isize, Vec<f32>)>,
    points: Vec<Point<i32>>,
    samples: Vec<super::ArcSample>,
    arc: f32,
    basis: Vec<f64>,
    basis_coeffs: usize,
    origin: (usize, usize),
    width: usize,
}

impl FitScratch {
    fn geometry(&mut self, points: &[Point<i32>]) {
        if self.points == points {
            return;
        }
        self.points.clear();
        self.points.extend_from_slice(points);
        (self.samples, self.arc) = sample_curve(points);
        self.basis_coeffs = 0;
    }

    fn prepare_basis(&mut self, coeffs: usize) {
        if self.basis_coeffs == coeffs {
            return;
        }
        self.basis.clear();
        for k in 0..self.samples.len() {
            let t = 31.0 * (k as f32 / self.arc).min(1.0);
            self.basis.extend((0..coeffs).map(|i| {
                std::f64::consts::SQRT_2
                    * (std::f64::consts::PI / 32.0 * i as f64 * (t as f64 + 0.5)).cos()
            }));
        }
        self.basis_coeffs = coeffs;
    }

    fn weight_table(&mut self, sigma_n: i32, sigma: f32, reach: isize) -> usize {
        if let Some(i) = self
            .weight_luts
            .iter()
            .position(|&(n, r, _)| n == sigma_n && r == reach)
        {
            return i;
        }
        let len = ((reach as f32 + 1.0) * std::f32::consts::SQRT_2 * WEIGHT_LUT_STEPS) as usize + 2;
        let table = (0..len)
            .map(|i| blob_weight(i as f32 / WEIGHT_LUT_STEPS, 0.0, 1.0 / sigma, 1.0))
            .collect();
        self.weight_luts.push((sigma_n, reach, table));
        self.weight_luts.len() - 1
    }

    /// Retain ordinary candidate buffers, without keeping an unusually large
    /// curve's allocations alive on every encoder worker.
    fn trim(&mut self) {
        const MAX_BYTES: usize = 32 * 1024 * 1024;
        fn trim<T>(v: &mut Vec<T>) {
            if v.capacity() > MAX_BYTES / size_of::<T>() {
                *v = Vec::new();
            }
        }
        trim(&mut self.slots);
        trim(&mut self.rows);
        trim(&mut self.pixels);
        if self.basis.capacity() > MAX_BYTES / size_of::<f64>() {
            self.basis = Vec::new();
            self.basis_coeffs = 0;
        }
    }
}

/// Fit precision: `Full` is what gets quantized, `Probe` ranks geometry moves.
#[derive(Clone, Copy, PartialEq)]
enum Precision {
    Full,
    Probe,
    Compact,
}

/// Cholesky solve of the leading `k x k` block of `ata`.
fn solve_leading(
    ata: &[[f64; MAX_COEFFS]; MAX_COEFFS],
    k: usize,
    rhs: &[f64; MAX_COEFFS],
    out: &mut [f64; MAX_COEFFS],
) {
    out.fill(0.0);
    if k == 0 {
        return;
    }
    let mut l = [[0f64; MAX_COEFFS]; MAX_COEFFS];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = ata[i][j] + if i == j { 1e-9 } else { 0.0 };
            for (&a, &b) in l[i][..j].iter().zip(&l[j][..j]) {
                sum -= a * b;
            }
            l[i][j] = if i == j {
                sum.max(1e-30).sqrt()
            } else {
                sum / l[j][j]
            };
        }
    }
    let mut y = [0f64; MAX_COEFFS];
    for i in 0..k {
        let mut sum = rhs[i];
        for (&a, &b) in l[i][..i].iter().zip(&y[..i]) {
            sum -= a * b;
        }
        y[i] = sum / l[i][i];
    }
    for i in (0..k).rev() {
        let mut sum = y[i];
        for (row, &value) in l[i + 1..k].iter().zip(&out[i + 1..k]) {
            sum -= row[i] * value;
        }
        out[i] = sum / l[i][i];
    }
}

/// Least-squares color fit of the curve through `points` at lattice width
/// `sigma_n`, with `(kx, ky, kb)` free coefficients per channel (`None` = all).
fn fit_at(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    points: &[Point<i32>],
    sigma_n: i32,
    free: Option<(usize, usize, usize)>,
    precision: Precision,
    scratch: &mut FitScratch,
) -> Option<Fit> {
    Some(fit_system(targets, w, h, points, sigma_n, precision, scratch)?.solve(free))
}

/// Geometry and pixel terms shared by all coefficient budgets of one fit.
struct FitSystem {
    ata: [[f64; MAX_COEFFS]; MAX_COEFFS],
    atb: [[f64; MAX_COEFFS]; 3],
    btb: [f64; 3],
    sigma_n: i32,
    coeffs: usize,
    arc: f32,
}

fn fit_system(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    points: &[Point<i32>],
    sigma_n: i32,
    precision: Precision,
    scratch: &mut FitScratch,
) -> Option<FitSystem> {
    scratch.geometry(points);
    let arc = scratch.arc;
    if arc < MIN_ARC {
        return None;
    }
    let mut coeffs = ((arc / ARC_PER_COEFF + 0.5) as usize).clamp(1, MAX_COEFFS);
    let sigma = sigma_n as f32 * sigma_step();
    let mut reach = (3.6 * sigma + 1.5).ceil() as isize;
    if precision == Precision::Probe {
        coeffs = coeffs.min(REFINE_COEFFS);
        reach = (2.6 * sigma + 1.0).ceil() as isize;
    }
    if precision == Precision::Compact {
        coeffs = coeffs.min(COMPACT_COEFFS);
    }
    scratch.prepare_basis(coeffs);
    let table = scratch.weight_table(sigma_n, sigma, reach);
    // Refinement uses three coefficients; compact extension and long-line
    // fits use six. Expose these common sizes to the compiler without changing
    // the raster or normal-equation accumulation order.
    match coeffs {
        REFINE_COEFFS => raster_system::<REFINE_COEFFS>(
            targets, w, h, sigma_n, sigma, reach, coeffs, table, scratch,
        ),
        COMPACT_COEFFS => raster_system::<COMPACT_COEFFS>(
            targets, w, h, sigma_n, sigma, reach, coeffs, table, scratch,
        ),
        _ => raster_system::<0>(targets, w, h, sigma_n, sigma, reach, coeffs, table, scratch),
    }
}

fn raster_system<const FIXED_COEFFS: usize>(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    sigma_n: i32,
    sigma: f32,
    reach: isize,
    coeffs: usize,
    table: usize,
    scratch: &mut FitScratch,
) -> Option<FitSystem> {
    let coeffs = if FIXED_COEFFS == 0 {
        coeffs
    } else {
        FIXED_COEFFS
    };
    let arc = scratch.arc;
    let weight_lut = &scratch.weight_luts[table].2;
    let samples = &scratch.samples;

    let (mut bx0, mut by0, mut bx1, mut by1) = (isize::MAX, isize::MAX, isize::MIN, isize::MIN);
    for sample in samples {
        let Point { x: cx, y: cy } = sample.position;
        let (rx, ry) = (round_i32(cx) as isize, round_i32(cy) as isize);
        bx0 = bx0.min(rx - reach);
        by0 = by0.min(ry - reach);
        bx1 = bx1.max(rx + reach);
        by1 = by1.max(ry + reach);
    }
    let (bx0, by0) = (bx0.max(0), by0.max(0));
    let (bx1, by1) = (bx1.min(w as isize - 1), by1.min(h as isize - 1));
    if bx1 < bx0 || by1 < by0 {
        return None;
    }
    let (bw, bh) = ((bx1 - bx0 + 1) as usize, (by1 - by0 + 1) as usize);
    scratch.origin = (bx0 as usize, by0 as usize);
    scratch.width = bw;
    scratch.slots.clear();
    scratch.slots.resize(bw * bh, -1);
    scratch.rows.clear();
    scratch.pixels.clear();

    for (sample, basis) in samples.iter().zip(scratch.basis.chunks_exact(coeffs)) {
        let Point { x: cx, y: cy } = sample.position;
        let mult = sample.multiplier;
        let amp = 0.25 * sigma * mult;
        let (rx, ry) = (round_i32(cx) as isize, round_i32(cy) as isize);
        let x0 = (rx - reach).max(bx0);
        let x1 = (rx + reach).min(bx1);
        if x0 > x1 {
            continue;
        }
        for y in (ry - reach).max(by0)..=(ry + reach).min(by1) {
            let dy = y as f32 - cy;
            let offset = (y - by0) as usize * bw;
            let slots = &mut scratch.slots[offset..offset + bw];
            for (x, slot) in (x0..=x1).zip(&mut slots[(x0 - bx0) as usize..=(x1 - bx0) as usize]) {
                let dx = x as f32 - cx;
                let pos = (dx * dx + dy * dy).sqrt() * WEIGHT_LUT_STEPS;
                let cell = pos as usize;
                let frac = pos - cell as f32;
                let weight = amp * (weight_lut[cell] * (1.0 - frac) + weight_lut[cell + 1] * frac);
                if weight.abs() <= 1e-7 {
                    continue;
                }
                if *slot < 0 {
                    *slot = scratch.pixels.len() as i32;
                    scratch.pixels.push((y as usize * w + x as usize) as u32);
                    // Initialize directly on first contact. Keep the addition
                    // to +0.0 so signed-zero behavior matches the old update.
                    scratch
                        .rows
                        .extend(basis.iter().map(|&b| 0.0 + weight as f64 * b));
                } else {
                    let row =
                        &mut scratch.rows[*slot as usize * coeffs..(*slot as usize + 1) * coeffs];
                    for (r, &b) in row.iter_mut().zip(&basis[..coeffs]) {
                        *r += weight as f64 * b;
                    }
                }
            }
        }
    }
    if scratch.pixels.is_empty() {
        return None;
    }

    let mut ata = [[0f64; MAX_COEFFS]; MAX_COEFFS];
    let mut atb = [[0f64; MAX_COEFFS]; 3];
    let mut btb = [0f64; 3];
    for (&pix, row) in scratch.pixels.iter().zip(scratch.rows.chunks_exact(coeffs)) {
        for (i, (&value, ata_row)) in row.iter().zip(&mut ata).enumerate() {
            for (a, &b) in ata_row[..=i].iter_mut().zip(&row[..=i]) {
                *a += value * b;
            }
        }
        for ((atb, btb), target) in atb.iter_mut().zip(&mut btb).zip(targets) {
            let t = target[pix as usize] as f64;
            *btb += t * t;
            for (a, &b) in atb.iter_mut().zip(row) {
                *a += b * t;
            }
        }
    }
    for i in 0..coeffs {
        for j in i + 1..coeffs {
            ata[i][j] = ata[j][i];
        }
    }

    Some(FitSystem {
        ata,
        atb,
        btb,
        sigma_n,
        coeffs,
        arc,
    })
}

impl FitSystem {
    /// Fit the color to the gradient of the original image. A smooth pedestal
    /// under a line belongs in VarDCT: removing it along with the line creates
    /// a narrow trough that costs AC coefficients. Differencing both the image
    /// and the render basis eliminates constant and linear backgrounds without
    /// trusting the masked blur. Reuse the expensive rasterized basis rows.
    fn residual_system(&self, source: &Image3F, scratch: &FitScratch) -> Self {
        let mut result = Self {
            ata: [[0.0; MAX_COEFFS]; MAX_COEFFS],
            atb: [[0.0; MAX_COEFFS]; 3],
            btb: [0.0; 3],
            sigma_n: self.sigma_n,
            coeffs: self.coeffs,
            arc: self.arc,
        };
        let k = self.coeffs;
        let (w, h) = (source.xsize(), source.ysize());
        let (x0, y0) = scratch.origin;
        let bw = scratch.width;
        let bh = scratch.slots.len() / bw;
        for (slot, &pixel) in scratch.pixels.iter().enumerate() {
            let pixel = pixel as usize;
            let (x, y) = (pixel % w, pixel / w);
            let row = &scratch.rows[slot * k..][..k];
            for (dx, dy) in [(1isize, 0isize), (0, 1), (-1, 0), (0, -1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                let neighbour = if nx >= x0 && ny >= y0 && nx < x0 + bw && ny < y0 + bh {
                    scratch.slots[(ny - y0) * bw + nx - x0]
                } else {
                    -1
                };
                // Interior edges once, and all edges of the finite support.
                if neighbour >= 0 && (neighbour as usize) < slot {
                    continue;
                }
                let mut diff = [0.0; MAX_COEFFS];
                for i in 0..k {
                    diff[i] = row[i]
                        - if neighbour >= 0 {
                            scratch.rows[neighbour as usize * k + i]
                        } else {
                            0.0
                        };
                }
                for i in 0..k {
                    for j in 0..=i {
                        result.ata[i][j] += diff[i] * diff[j];
                    }
                }
                for c in 0..3 {
                    let plane = source.plane_data(c);
                    let target = (plane[pixel] - plane[ny * w + nx]) as f64;
                    result.btb[c] += target * target;
                    for (a, &d) in result.atb[c][..k].iter_mut().zip(&diff) {
                        *a += d * target;
                    }
                }
            }
        }
        for i in 0..k {
            for j in i + 1..k {
                result.ata[i][j] = result.ata[j][i];
            }
        }
        result
    }

    fn solve(&self, free: Option<(usize, usize, usize)>) -> Fit {
        let Self { ata, atb, btb, .. } = self;
        let coeffs = self.coeffs;
        let (kx, ky, kb) = free.unwrap_or((coeffs, coeffs, coeffs));
        let mut sol = [[0f64; MAX_COEFFS]; 3];
        solve_leading(ata, ky.min(coeffs), &atb[1], &mut sol[1]);
        solve_leading(ata, kx.min(coeffs), &atb[0], &mut sol[0]);
        // B is stored as a residual of Y (base correlation 1.0): fit the residual
        let mut rhs = atb[2];
        for (rhs, row) in rhs[..coeffs].iter_mut().zip(ata) {
            for (&a, &y) in row[..coeffs].iter().zip(&sol[1]) {
                *rhs -= a * y;
            }
        }
        let mut residual = [0f64; MAX_COEFFS];
        solve_leading(ata, kb.min(coeffs), &rhs, &mut residual);
        let [_, y, b] = &mut sol;
        for ((b, &y), &residual) in b[..coeffs].iter_mut().zip(y.iter()).zip(&residual) {
            *b = y + residual;
        }

        let (mut err, mut e0) = (0f64, 0f64);
        for c in 0..3 {
            let mut quad = 0f64;
            let mut lin = 0f64;
            for ((&value, &rhs), row) in sol[c][..coeffs].iter().zip(&atb[c]).zip(ata) {
                lin += value * rhs;
                for (&a, &b) in row[..coeffs].iter().zip(&sol[c]) {
                    quad += value * a * b;
                }
            }
            let weight = FIT_CHANNEL_WEIGHT[c] * FIT_CHANNEL_WEIGHT[c];
            err += weight * (btb[c] - 2.0 * lin + quad).max(0.0);
            e0 += weight * btb[c];
        }
        Fit {
            err,
            e0,
            sigma_n: self.sigma_n,
            coeffs,
            arc: self.arc,
            sol,
        }
    }
}

/// Best fit over a set of lattice widths. Supports differ per width, so widths
/// compete on energy reduction, not on the residual sum.
fn fit_best(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    points: &[Point<i32>],
    widths: &[i32],
    free: Option<(usize, usize, usize)>,
    scratch: &mut FitScratch,
) -> Option<Fit> {
    let mut best: Option<Fit> = None;
    for &n in widths {
        if let Some(fit) = fit_at(targets, w, h, points, n, free, Precision::Full, scratch)
            && best.as_ref().is_none_or(|b| fit.gain() > b.gain())
        {
            best = Some(fit);
        }
    }
    best
}

#[inline]
fn quantized_coefficient(value: f64) -> i32 {
    if value.abs() < 0.5 {
        0
    } else {
        (value + 0.5f64.copysign(value)) as i32
    }
}

fn quantize(points: &[Point<i32>], fit: &Fit) -> QuantizedSpline {
    let quant = adjusted_quant(QUANT_ADJUST) as f64;
    let mut dct = [[0i32; 32]; 4];
    let [x, y, b, sigma] = &mut dct;
    for (i, ((x, y), b)) in x[..fit.coeffs].iter_mut().zip(y).zip(b).enumerate() {
        let f = if i == 0 {
            std::f64::consts::SQRT_2
        } else {
            1.0
        };
        *y = quantized_coefficient(fit.sol[1][i] * f * quant / CHANNEL_WEIGHT[1] as f64);
        let restored = *y as f64 / f * CHANNEL_WEIGHT[1] as f64 / quant;
        *x = quantized_coefficient(fit.sol[0][i] * f * quant / CHANNEL_WEIGHT[0] as f64);
        *b = quantized_coefficient(
            (fit.sol[2][i] - restored) * f * quant / CHANNEL_WEIGHT[2] as f64,
        );
    }
    sigma[0] = fit.sigma_n;
    QuantizedSpline {
        points: points.to_vec(),
        dct,
    }
}

/// +-1 px coordinate descent of every control point at a fixed width.
fn refine(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    points: &[Point<i32>],
    sigma_n: i32,
    sweeps: usize,
    scratch: &mut FitScratch,
) -> Vec<Point<i32>> {
    let cost = |pts: &[Point<i32>], scratch: &mut FitScratch| -> f64 {
        if pts.windows(2).any(|p| p[0] == p[1]) {
            return f64::INFINITY;
        }
        fit_at(targets, w, h, pts, sigma_n, None, Precision::Probe, scratch)
            .map_or(f64::INFINITY, |f| -f.gain())
    };
    // A control point only shapes the curve a few spans around it: rank its
    // moves on that sub-curve instead of re-fitting the whole spline.
    let mut points = points.to_vec();
    for _ in 0..sweeps {
        let mut moved = false;
        for i in 0..points.len() {
            let lo = i.saturating_sub(REFINE_NEIGHBOURS);
            let hi = (i + REFINE_NEIGHBOURS).min(points.len() - 1);
            let local = &mut points[lo..=hi];
            let center = i - lo;
            let mut best_point = local[center];
            let mut best = cost(local, scratch);
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let trial = Point::new(best_point.x + dx, best_point.y + dy);
                local[center] = trial;
                let value = cost(local, scratch);
                if value < best - 1e-12 {
                    best = value;
                    best_point = trial;
                    moved = true;
                }
            }
            local[center] = best_point;
        }
        if !moved {
            break;
        }
    }
    points
}

/// Squared distance from `p` to the polyline `chain`.
#[cfg(test)]
fn distance_sq_to_chain(p: Point<f32>, chain: &[Point<f32>]) -> f32 {
    let mut best = f32::INFINITY;
    for &[a, b] in chain.array_windows::<2>() {
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 1e-12 {
            (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (ex, ey) = (p.x - (a.x + t * dx), p.y - (a.y + t * dy));
        best = best.min(ex * ex + ey * ey);
    }
    best
}

/// Collapse two interior controls, allowing their replacement to leave either
/// old knot. Test positions on the old curve and a small integer neighborhood;
/// this is geometry only, before the selector pays for rendering one proposal.
pub(super) fn collapse_control_pair(
    ctx: &EncodingContext,
    points: &[Point<i32>],
    i: usize,
) -> Option<Vec<Point<i32>>> {
    if i == 0 || i + 2 >= points.len() {
        return None;
    }
    let control: Vec<_> = points.iter().map(|p| p.as_f32()).collect();
    // The merged knot can influence four spans in the shorter polygon.
    let lo = i.saturating_sub(2);
    let hi = (i + 3).min(points.len() - 1);
    let reference = catmull_rom_spans(&control, lo..hi);
    let reference_distance = Polyline::new(ctx, &reference);
    let samples = catmull_rom_spans(&control, i..i + 1);
    let mut reduced = points.to_vec();
    reduced.remove(i + 1);
    let cost = |points: &[Point<i32>]| {
        if points.array_windows::<2>().any(|p| p[0] == p[1]) {
            return f32::INFINITY;
        }
        let control: Vec<_> = points.iter().map(|p| p.as_f32()).collect();
        let curve = catmull_rom_spans(&control, lo..hi - 1);
        let curve_distance = Polyline::new(ctx, &curve);
        let forward: f32 = curve.iter().map(|&p| reference_distance.distance(p)).sum();
        let backward: f32 = reference.iter().map(|&p| curve_distance.distance(p)).sum();
        forward / curve.len() as f32 + backward / reference.len() as f32
    };
    let (mut best_point, mut best_cost) = (reduced[i], f32::INFINITY);
    for k in [0, 4, 8, 12, 16] {
        reduced[i] = Point::new(round_i32(samples[k].x), round_i32(samples[k].y));
        let value = cost(&reduced);
        if value < best_cost {
            best_cost = value;
            best_point = reduced[i];
        }
    }
    for step in [2, 1] {
        let origin = best_point;
        for (dx, dy) in [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ] {
            reduced[i] = Point::new(origin.x + dx * step, origin.y + dy * step);
            let value = cost(&reduced);
            if value < best_cost {
                best_cost = value;
                best_point = reduced[i];
            }
        }
    }
    reduced[i] = best_point;
    best_cost.is_finite().then_some(reduced)
}

/// +-1 px coordinate descent against the sub-pixel ridge: pulls the
/// Catmull-Rom curve onto the chain. Pure geometry, no rendering; a control
/// point only moves the four curve segments around it.
fn refine_geometric(
    ctx: &EncodingContext,
    points: &[Point<i32>],
    chain: &[Point<f32>],
    sweeps: usize,
) -> Vec<Point<i32>> {
    let sm = smooth_chain(chain, 1.0);
    let distance = Polyline::new(ctx, &sm);
    let local_cost = |pts: &[Point<i32>], i: usize| -> f32 {
        if pts.windows(2).any(|p| p[0] == p[1]) {
            return f32::INFINITY;
        }
        let control: Vec<Point<f32>> = pts.iter().map(|&p| p.as_f32()).collect();
        let spans = i.saturating_sub(2)..(i + 2).min(pts.len() - 1);
        let first = spans.start * 16;
        let last = spans.end * 16;
        let curve = catmull_rom_spans(&control, spans);
        // the curve runs along the chain: only search the matching stretch
        let per_point = sm.len() as f32 / ((pts.len() - 1) * 16).max(1) as f32;
        let slack = (3 * sm.len() / pts.len() + 8) as f32;
        let lo = ((first as f32 * per_point - slack).max(0.0) as usize).min(sm.len() - 2);
        let hi = ((last as f32 * per_point + slack) as usize + 2).min(sm.len());
        curve
            .iter()
            .map(|&p| distance.distance_range(p, lo..hi - 1))
            .sum()
    };
    let mut points = points.to_vec();
    for _ in 0..sweeps {
        let mut moved = false;
        for i in 0..points.len() {
            let mut best = local_cost(&points, i);
            for (dx, dy) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let previous = points[i];
                points[i] = Point::new(previous.x + dx, previous.y + dy);
                let value = local_cost(&points, i);
                if value < best - 1e-6 {
                    best = value;
                    moved = true;
                } else {
                    points[i] = previous;
                }
            }
        }
        if !moved {
            break;
        }
    }
    points
}

fn inside(points: &[Point<i32>], w: usize, h: usize) -> bool {
    points.iter().all(|p| {
        p.x >= BORDER && p.y >= BORDER && p.x < w as i32 - BORDER && p.y < h as i32 - BORDER
    })
}

struct Primary {
    points: Vec<Point<i32>>,
    fit: Fit,
    spline: QuantizedSpline,
}

/// Seeded-width fit of the unrefined control points.
fn quick_primary(
    ctx: &EncodingContext,
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    chain: &Chain,
    scratch: &mut FitScratch,
) -> Option<Primary> {
    let points = prune_control_points(ctx, &rdp_control_points(&chain.points), 0.5);
    if points.len() < 2 || !inside(&points, w, h) {
        return None;
    }
    let predicted = SCALE_TO_SIGMA.0 * chain.scale + SCALE_TO_SIGMA.1;
    let mut seeds = SIGMA_LATTICE;
    seeds.sort_by(|&a, &b| {
        (a as f32 * sigma_step() - predicted)
            .abs()
            .total_cmp(&(b as f32 * sigma_step() - predicted).abs())
    });
    let fit = fit_best(
        targets,
        w,
        h,
        &points,
        &seeds[..SEEDED_WIDTHS],
        None,
        scratch,
    )?;
    if fit.explained() < QUICK_MIN_EXPLAINED {
        return None;
    }
    let spline = quantize(&points, &fit);
    Some(Primary {
        points,
        fit,
        spline,
    })
}

/// Refines a primary that survived the RD pre-test and builds its alternatives.
fn expand(
    ctx: &EncodingContext,
    targets: &[Vec<f32>; 3],
    source: &Image3F,
    w: usize,
    h: usize,
    chain: &Chain,
    primary: Primary,
    scratch: &mut FitScratch,
) -> Option<Candidate> {
    let Primary {
        mut points,
        mut fit,
        ..
    } = primary;
    if fit.explained() > REFINE_MIN_EXPLAINED {
        let aligned = refine_geometric(ctx, &points, &chain.points, 2);
        let refined = refine(targets, w, h, &aligned, fit.sigma_n, 1, scratch);
        let at = SIGMA_LATTICE
            .iter()
            .position(|&n| n == fit.sigma_n)
            .unwrap_or(0);
        let nearby = &SIGMA_LATTICE[at.saturating_sub(1)..(at + 2).min(SIGMA_LATTICE.len())];
        if let Some(better) = fit_best(targets, w, h, &refined, nearby, None, scratch)
            && better.err < fit.err
        {
            (points, fit) = (refined, better);
        }
    }
    if fit.explained() < MIN_EXPLAINED || !inside(&points, w, h) {
        return None;
    }
    let primary = quantize(&points, &fit);
    if primary.dct[..3]
        .iter()
        .all(|row| row.iter().all(|&v| v == 0))
    {
        return None;
    }

    let lean = (1, ((fit.arc / LEAN_ARC_PER_COEFF + 0.5) as usize).max(1), 1);
    let mut alts = vec![primary];
    let mut geometries = vec![points];
    for tol in CURVE_TOLERANCES {
        let sparse = curve_control_points(ctx, &chain.points, tol);
        if sparse.len() >= 2 {
            let aligned = refine_geometric(ctx, &sparse, &chain.points, 2);
            let refined = if aligned.len() < geometries[0].len() {
                refine(targets, w, h, &aligned, fit.sigma_n, 1, scratch)
            } else {
                aligned
            };
            if !geometries.contains(&refined) {
                geometries.push(refined);
            }
        }
    }
    for (g, geometry) in geometries.iter().enumerate() {
        if !inside(geometry, w, h) || geometry.windows(2).any(|p| p[0] == p[1]) {
            continue;
        }
        let Some(system) = fit_system(
            targets,
            w,
            h,
            geometry,
            fit.sigma_n,
            Precision::Full,
            scratch,
        ) else {
            continue;
        };
        for free in [None, Some(lean)] {
            if g == 0 && free.is_none() {
                continue;
            }
            let alt = system.solve(free);
            let spline = quantize(geometry, &alt);
            if spline.dct[1].iter().any(|&v| v != 0) {
                alts.push(spline);
            }
        }
        let residual = system.residual_system(source, scratch);
        for free in [None, Some(lean)] {
            let spline = quantize(geometry, &residual.solve(free));
            if spline.dct[1].iter().any(|&v| v != 0)
                && !alts
                    .iter()
                    .any(|a| a.points == spline.points && a.dct == spline.dct)
            {
                alts.push(spline);
            }
        }
    }
    Some(Candidate {
        alts,
        bits_factor: 1.0,
    })
}

/// Economical representation for a long continuation: six color coefficients,
/// full rendering support, and only the two widths predicted from its seed.
fn fit_extension(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    chain: &Chain,
    scratch: &mut FitScratch,
) -> Option<QuantizedSpline> {
    let points = rdp_control_points(&chain.points);
    if points.len() < 2 || !inside(&points, w, h) {
        return None;
    }
    let predicted = SCALE_TO_SIGMA.0 * chain.scale + SCALE_TO_SIGMA.1;
    let mut widths = SIGMA_LATTICE;
    widths.sort_by(|&a, &b| {
        (a as f32 * sigma_step() - predicted)
            .abs()
            .total_cmp(&(b as f32 * sigma_step() - predicted).abs())
    });
    let mut best: Option<Fit> = None;
    for n in &widths[..SEEDED_WIDTHS] {
        if let Some(fit) = fit_at(
            targets,
            w,
            h,
            &points,
            *n,
            None,
            Precision::Compact,
            scratch,
        ) && best.as_ref().is_none_or(|b| fit.gain() > b.gain())
        {
            best = Some(fit);
        }
    }
    let fit = best?;
    // The original block pre-test remains the decision about coding value.
    // Broad residual support can depress the explained fraction of a faint line.
    let spline = quantize(&points, &fit);
    spline.dct[..3]
        .iter()
        .flatten()
        .any(|&v| v != 0)
        .then_some(spline)
}

/// Lean luma-only alternatives for a long straight line. Coding value is left
/// to the RD tests: a faint line's explained fraction is dominated by grain.
fn fit_long_line(
    ctx: &EncodingContext,
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    chain: &Chain,
    scratch: &mut FitScratch,
) -> Option<Candidate> {
    let spacing = LONG_LINE_MIN_SPACING.max(chain.points.len() as f32 / LONG_LINE_SPANS);
    let points = uniform_control_points(&chain.points, spacing);
    if points.len() < 2 || !inside(&points, w, h) {
        return None;
    }
    let points = refine_geometric(ctx, &points, &chain.points, 2);
    if !inside(&points, w, h) || points.windows(2).any(|p| p[0] == p[1]) {
        return None;
    }
    // Geometry and width determine the system; coefficient budgets only solve it.
    let systems: Vec<FitSystem> = LONG_LINE_WIDTHS
        .iter()
        .filter_map(|&n| fit_system(targets, w, h, &points, n, Precision::Compact, scratch))
        .collect();
    let mut alts: Vec<QuantizedSpline> = Vec::new();
    for k in 1..=LONG_LINE_MAX_COEFFS {
        let mut best: Option<Fit> = None;
        for system in &systems {
            let fit = system.solve(Some((0, k, 0)));
            if best.as_ref().is_none_or(|b| fit.gain() > b.gain()) {
                best = Some(fit);
            }
        }
        let Some(fit) = best else { continue };
        let mut spline = quantize(&points, &fit);
        spline.dct[0] = [0; 32];
        spline.dct[2] = [0; 32];
        if spline.dct[1].iter().any(|&v| v != 0) && !alts.iter().any(|a| a.dct == spline.dct) {
            alts.push(spline);
        }
    }
    (!alts.is_empty()).then_some(Candidate {
        alts,
        bits_factor: LONG_LINE_BITS_FACTOR,
    })
}

pub(super) fn fit_candidates(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    xyb: &Image3F,
    quant_field: &[f32],
    chains: &[Chain],
    extensions: &[Chain],
    long_lines: &[Chain],
) -> Vec<Candidate> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let targets = line_targets(ctx, scratch, xyb, chains);
    let targets = &targets;
    let model = BlockModel::new(ctx, distance);
    let model = &model;
    ctx.thread_pool
        .steal_map(
            scratch,
            chains.len() + extensions.len() + long_lines.len(),
            |i, worker| {
                let fit_scratch = &mut worker.spline_fit;
                let result = (|| {
                    // Long lines come last: the greedy selector must see the proper
                    // coloured, curved candidates before a lean straight stand-in.
                    if i >= chains.len() + extensions.len() {
                        let line = &long_lines[i - chains.len() - extensions.len()];
                        let candidate = fit_long_line(ctx, targets, w, h, line, fit_scratch)?;
                        return candidate
                            .alts
                            .iter()
                            .any(|alt| pretest(model, xyb, quant_field, alt))
                            .then_some(candidate);
                    }
                    if i >= chains.len() {
                        let spline = fit_extension(
                            targets,
                            w,
                            h,
                            &extensions[i - chains.len()],
                            fit_scratch,
                        )?;
                        return pretest(model, xyb, quant_field, &spline).then_some(Candidate {
                            alts: vec![spline],
                            bits_factor: 1.0,
                        });
                    }
                    let primary = quick_primary(ctx, targets, w, h, &chains[i], fit_scratch)?;
                    if !pretest(model, xyb, quant_field, &primary.spline) {
                        return None;
                    }
                    expand(ctx, targets, xyb, w, h, &chains[i], primary, fit_scratch)
                })();
                fit_scratch.trim();
                result
            },
        )
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_coefficient_kernels_match_dynamic_accumulation() {
        let (w, h) = (93, 67);
        let targets = std::array::from_fn(|c| {
            (0..w * h)
                .map(|i| ((i * 19 + c * 31) % 101) as f32 * 0.003 - 0.15)
                .collect::<Vec<_>>()
        });
        let points = [Point::new(2, 4), Point::new(33, 53), Point::new(88, 58)];
        let mut scratch = FitScratch::default();
        for precision in [Precision::Probe, Precision::Compact] {
            for sigma_n in SIGMA_LATTICE {
                let fixed =
                    fit_system(&targets, w, h, &points, sigma_n, precision, &mut scratch).unwrap();
                let coeffs = if precision == Precision::Probe {
                    REFINE_COEFFS
                } else {
                    COMPACT_COEFFS
                };
                assert_eq!(fixed.coeffs, coeffs);
                let sigma = sigma_n as f32 * sigma_step();
                let reach = if precision == Precision::Probe {
                    (2.6 * sigma + 1.0).ceil()
                } else {
                    (3.6 * sigma + 1.5).ceil()
                } as isize;
                let table = scratch.weight_table(sigma_n, sigma, reach);
                let dynamic = raster_system::<0>(
                    &targets,
                    w,
                    h,
                    sigma_n,
                    sigma,
                    reach,
                    coeffs,
                    table,
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(fixed.ata, dynamic.ata);
                assert_eq!(fixed.atb, dynamic.atb);
                assert_eq!(fixed.btb, dynamic.btb);
            }
        }
    }

    #[test]
    fn cached_geometry_and_widths_do_not_reuse_image_terms() {
        let (w, h) = (91, 63);
        let mut targets = std::array::from_fn(|c| {
            (0..w * h)
                .map(|i| ((i * 19 + c * 31) % 101) as f32 * 0.003)
                .collect::<Vec<_>>()
        });
        let shapes = [
            vec![Point::new(2, 4), Point::new(33, 53), Point::new(88, 58)],
            vec![Point::new(4, 60), Point::new(49, 17), Point::new(89, 3)],
        ];
        let mut reused = FitScratch::default();
        for (shape, width, precision) in [
            (0, 1, Precision::Full),
            (0, 6, Precision::Full),
            (0, 6, Precision::Compact),
            (1, 4, Precision::Probe),
            (1, 4, Precision::Full),
            (0, 1, Precision::Full),
        ] {
            let cached = fit_system(
                &targets,
                w,
                h,
                &shapes[shape],
                width,
                precision,
                &mut reused,
            )
            .unwrap();
            let fresh = fit_system(
                &targets,
                w,
                h,
                &shapes[shape],
                width,
                precision,
                &mut FitScratch::default(),
            )
            .unwrap();
            assert_eq!(cached.ata, fresh.ata);
            assert_eq!(cached.atb, fresh.atb);
            assert_eq!(cached.btb, fresh.btb);
            assert_eq!(cached.arc, fresh.arc);
            for plane in &mut targets {
                for v in plane {
                    *v *= 0.75;
                }
            }
        }
    }

    #[test]
    fn pair_collapse_relocates_controls_to_preserve_a_bend() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let points: Vec<_> = (0..=8)
            .map(|i| {
                let a = i as f32 * std::f32::consts::FRAC_PI_2 / 8.0;
                Point::new(
                    round_i32(32.0 + 100.0 * a.sin()),
                    round_i32(32.0 + 100.0 * (1.0 - a.cos())),
                )
            })
            .collect();
        let reference =
            super::super::catmull_rom(&points.iter().map(|p| p.as_f32()).collect::<Vec<_>>());
        let error = |points: &[Point<i32>]| {
            let curve =
                super::super::catmull_rom(&points.iter().map(|p| p.as_f32()).collect::<Vec<_>>());
            curve
                .iter()
                .map(|&p| distance_sq_to_chain(p, &reference))
                .sum::<f32>()
                / curve.len() as f32
                + reference
                    .iter()
                    .map(|&p| distance_sq_to_chain(p, &curve))
                    .sum::<f32>()
                    / reference.len() as f32
        };
        let mut relocated = false;
        for i in 1..points.len() - 2 {
            let reduced = collapse_control_pair(&kernels, &points, i).unwrap();
            assert_eq!(reduced.len(), points.len() - 1);
            assert_eq!(reduced.first(), points.first());
            assert_eq!(reduced.last(), points.last());
            assert!(reduced.array_windows::<2>().all(|p| p[0] != p[1]));
            let mut left = points.clone();
            left.remove(i);
            let mut right = points.clone();
            right.remove(i + 1);
            relocated |= reduced[i] != points[i]
                && reduced[i] != points[i + 1]
                && error(&reduced) < error(&left).min(error(&right));
        }
        assert!(
            relocated,
            "merging should improve on merely deleting either old knot"
        );
        assert!(collapse_control_pair(&kernels, &points, 0).is_none());
        assert!(collapse_control_pair(&kernels, &points, points.len() - 2).is_none());
        assert!(collapse_control_pair(&kernels, &points[..3], 1).is_none());
    }

    #[test]
    fn pruning_keeps_curve_shape_and_collapses_straight_spans() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let straight: Vec<_> = (0..12)
            .map(|i| Point::new(20 + i * 8, 40 + i * 4))
            .collect();
        assert_eq!(
            prune_control_points(&kernels, &straight, 0.5),
            [straight[0], straight[11]]
        );
        let chain: Vec<_> = (0..=160)
            .map(|i| {
                let t = i as f32 / 90.0;
                Point::new(30.0 + 90.0 * t.sin(), 40.0 + 90.0 * (1.0 - t.cos()))
            })
            .collect();
        let dense = rdp_control_points(&chain);
        let sparse = prune_control_points(&kernels, &dense, 0.5);
        assert!(sparse.len() < dense.len());
        assert_eq!(sparse.first(), dense.first());
        assert_eq!(sparse.last(), dense.last());
        let a = super::super::catmull_rom(&dense.iter().map(|p| p.as_f32()).collect::<Vec<_>>());
        let b = super::super::catmull_rom(&sparse.iter().map(|p| p.as_f32()).collect::<Vec<_>>());
        for (&point, other) in a
            .iter()
            .map(|p| (p, b.as_slice()))
            .chain(b.iter().map(|p| (p, a.as_slice())))
        {
            assert!(distance_sq_to_chain(point, other) <= 0.501 * 0.501);
        }
    }

    #[test]
    fn residual_fit_leaves_affine_background_in_vardct() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let (w, h) = (96, 64);
        let points = vec![Point::new(16, 30), Point::new(48, 34), Point::new(80, 30)];
        let mut line = QuantizedSpline {
            points: points.clone(),
            dct: [[0; 32]; 4],
        };
        line.dct[0][0] = 3;
        line.dct[1][0] = 5;
        line.dct[2][0] = -2;
        line.dct[3][0] = 4;
        let mut image = Image3F::new(w, h);
        super::super::render_spline(&kernels, &line, QUANT_ADJUST, &mut image, 1.0);
        let targets = std::array::from_fn(|c| image.plane_data(c).to_vec());
        let mut scratch = FitScratch::default();
        let system = fit_system(&targets, w, h, &points, 4, Precision::Full, &mut scratch).unwrap();
        let bare = system.residual_system(&image, &scratch).solve(None);
        for c in 0..3 {
            for y in 0..h {
                for (x, v) in image.plane_row_mut(c, y).iter_mut().enumerate() {
                    *v += 0.2 + c as f32 * 0.03 + 0.001 * x as f32 - 0.002 * y as f32;
                }
            }
        }
        let pedestal = system.residual_system(&image, &scratch).solve(None);
        for (a, b) in bare.sol.iter().flatten().zip(pedestal.sol.iter().flatten()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
        assert_eq!(quantize(&points, &bare).dct, line.dct);
        assert_eq!(quantize(&points, &pedestal).dct, line.dct);
    }

    #[test]
    fn curve_aware_control_points_follow_a_bend_with_fewer_points() {
        let kernels = crate::encoding_context::EncodingContext::default();
        // a quarter circle of radius 120 px, traced at about 1 px steps
        let chain: Vec<Point<f32>> = (0..=188)
            .map(|k| {
                let a = k as f32 / 120.0;
                Point::new(40.0 + 120.0 * a.sin(), 160.0 - 120.0 * a.cos())
            })
            .collect();
        let sparse = curve_control_points(&kernels, &chain, 1.0);
        let dense = rdp_control_points(&chain);
        assert!(
            sparse.len() >= 2 && sparse.len() < dense.len(),
            "{} vs {}",
            sparse.len(),
            dense.len()
        );
        let control: Vec<Point<f32>> = sparse.iter().map(|p| p.as_f32()).collect();
        let curve = crate::splines::catmull_rom(&control);
        for &p in &chain[4..chain.len() - 4] {
            assert!(distance_sq_to_chain(p, &curve) <= 1.6 * 1.6);
        }
        assert!(sparse.windows(2).all(|p| p[0] != p[1]));
    }

    #[test]
    fn compact_fit_matches_full_support_with_six_free_coefficients() {
        let (w, h) = (256, 128);
        let targets = std::array::from_fn(|c| {
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f32, (i / w) as f32);
                    let a = [0.002, 0.08, 0.07][c] * (1.0 + 0.2 * (x * 0.03).sin());
                    a * (-0.5 * ((y - 64.0) / 1.2).powi(2)).exp()
                        + 0.0001 * (x * 0.1 + y * 0.2).cos()
                })
                .collect()
        });
        let points = [Point::new(20, 64), Point::new(220, 64)];
        let mut scratch = FitScratch::default();
        let full = fit_at(
            &targets,
            w,
            h,
            &points,
            2,
            Some((6, 6, 6)),
            Precision::Full,
            &mut scratch,
        )
        .unwrap();
        let compact = fit_at(
            &targets,
            w,
            h,
            &points,
            2,
            None,
            Precision::Compact,
            &mut scratch,
        )
        .unwrap();
        assert!(full.coeffs > 6);
        assert_eq!(compact.coeffs, 6);
        assert_eq!(full.sol, compact.sol);
        assert_eq!(full.e0, compact.e0);
        assert!((full.err - compact.err).abs() < 1e-9);
    }

    #[test]
    fn cholesky_recovers_known_solution_for_each_leading_size() {
        let ata = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 40.0 } else { 1.0 }));
        let expected: [f64; MAX_COEFFS] = std::array::from_fn(|i| (i as f64 - 15.0) * 0.125);
        for k in 0..=MAX_COEFFS {
            let rhs = std::array::from_fn(|i| {
                ata[i][..k]
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &x)| a * x)
                    .sum()
            });
            let mut actual = [1.0; MAX_COEFFS];
            solve_leading(&ata, k, &rhs, &mut actual);
            for (&actual, &expected) in actual[..k].iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-9, "k={k}");
            }
            assert!(actual[k..].iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn signed_coefficients_round_away_from_zero_at_ties() {
        for value in [
            -1024.5, -2.75, -1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5, 2.75, 1024.5,
        ] {
            assert_eq!(quantized_coefficient(value), value.round() as i32);
        }
        for value in [0.5f64.next_down(), 0.5, 0.5f64.next_up(), i32::MAX as f64] {
            for value in [value, -value] {
                assert_eq!(quantized_coefficient(value), value.round() as i32);
            }
        }
    }
}
