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
//! quantized lattice and are seeded from the ridge scale. Integer control
//! points are aligned to the traced ridge; the primary curve's points are then
//! refined by +-1 px coordinate descent on the raster fit. Candidates carry
//! several alternatives for the RD gate to choose from.

use super::detect::{Chain, gaussian_kernels};
use super::filter::FilterPlan;
use super::geometry::Polyline;
use super::lines::{LineIndex, curve_samples, unit_tangents};
use super::select::Pretest;
use super::{
    BlobWeightRow, CHANNEL_WEIGHT, FitMatrix, FitTargets, Point, QUANT_ADJUST, QuantizedSpline,
    SplinesContext, adjusted_quant, catmull_rom_spans, round_i32, sample_curve,
};
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::util::fast_hypot;

static SIGMA_LATTICE: [i32; 8] = [1, 2, 3, 4, 5, 6, 8, 10];
const SEEDED_WIDTHS: usize = 2;
const SCALE_TO_SIGMA: (f32, f32) = (0.86, -0.21);
static FIT_CHANNEL_WEIGHT: [f64; 3] = [14.0, 1.0, 0.6];
const MIN_EXPLAINED: f64 = 0.3;
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
pub(super) const MAX_COEFFS: usize = 32;
const BASIS_FREQUENCIES: [f32; MAX_COEFFS] = {
    let mut frequencies = [0.0; MAX_COEFFS];
    let mut i = 1;
    while i < MAX_COEFFS {
        frequencies[i] = (std::f32::consts::PI / 32.0) * i as f32;
        i += 1;
    }
    frequencies
};
/// Refinement only needs the alignment signal: few color coefficients, tight window.
const REFINE_COEFFS: usize = 3;
const COMPACT_COEFFS: usize = 6;
const REFINE_NEIGHBORS: usize = 3;
const WEIGHT_LUT_STEPS: f32 = 32.0;
/// Unrefined fits slightly below `MIN_EXPLAINED` may still get there after refinement.
const QUICK_MIN_EXPLAINED: f64 = 0.27;
/// Long faint lines get the cheapest representation there is: sparse control
/// points, luma only, one to three coefficients. Anything richer fits grain.
const LONG_LINE_MIN_SPACING: f32 = 64.0;
const LONG_LINE_SPANS: f32 = 3.0;
const LONG_LINE_MAX_COEFFS: usize = 3;
static LONG_LINE_WIDTHS: [i32; 5] = [2, 3, 4, 5, 6];
/// Long-line widths tried, nearest to the ridge-scale prediction first.
const LONG_LINE_SEEDED_WIDTHS: usize = 2;
/// A long line touches many blocks, and the DCT8 proxy's optimism adds up over
/// them: long splines must clear a higher bar on textured ground.
const LONG_LINE_BITS_FACTOR: f32 = 4.0;
/// Bounding-box slack when looking for candidates on a group, in pixels.
const GROUP_SLACK: f32 = 8.0;
/// A joined line gets two endpoints only when its ridge stays this close.
const JOINED_LINE_TOLERANCE: f32 = 0.7;

/// A collinear group from `lines` with whole-line alternatives. Once any of
/// its pieces is accepted, the gate decides the line as a unit.
pub(super) struct LineGroup {
    pub(super) points: Vec<Point<f32>>,
    pub(super) alts: Vec<QuantizedSpline>,
}

/// Alternative parametrisations of one detected line; the gate keeps at most one.
pub(super) struct Candidate {
    pub(super) alts: Vec<QuantizedSpline>,
    /// Extra factor on the spline bits in the RD gate.
    pub(super) bits_factor: f32,
    /// Star dot (see `dots`): priced by position and template, selected after lines.
    pub(super) dot: bool,
}

/// Coefficient count the Full fit would use for this geometry.
fn full_coeffs(points: &[Point<i32>], scratch: &mut FitScratch) -> usize {
    scratch.geometry(points);
    ((scratch.arc / ARC_PER_COEFF + 0.5) as usize).clamp(1, MAX_COEFFS)
}

/// Width with the best Compact gain; None when no width fits. On long chains
/// six coefficients rank widths like the full budget at a fraction of the
/// raster cost.
fn compact_width(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    points: &[Point<i32>],
    widths: &[i32],
    scratch: &mut FitScratch,
) -> Option<(i32, f64)> {
    let mut best: Option<(i32, f64)> = None;
    for &n in widths {
        if let Some(system) = fit_system(targets, w, h, points, n, Precision::Compact, scratch) {
            let gain = system.solve(None).gain();
            if best.is_none_or(|(_, b)| gain > b) {
                best = Some((n, gain));
            }
        }
    }
    best
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
    let mut weight = vec![0.0; w * h];
    horizontal.horizontal(ctx, scratch, &keep, &mut rows);
    vertical.vertical(ctx, scratch, &rows, &mut weight);
    // The masked input can hold the vertical result after the horizontal pass.
    let mut residual = |src: &[f32], mut target: Vec<f32>| {
        horizontal.horizontal(ctx, scratch, &target, &mut rows);
        vertical.vertical(ctx, scratch, &rows, &mut target);
        for ((target, &v), &n) in target.iter_mut().zip(src).zip(&weight) {
            *target = v - *target / (n + 1e-6);
        }
        target
    };
    let [x, y] = [0, 1].map(|c| {
        let src = xyb.plane_data(c);
        let masked = src.iter().zip(&keep).map(|(&v, &m)| v * m).collect();
        residual(src, masked)
    });
    // The last channel can take ownership of the mask's storage.
    let b = xyb.plane_data(2);
    for (mask, &v) in keep.iter_mut().zip(b) {
        *mask *= v;
    }
    [x, y, residual(b, keep)]
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
/// Stop once the rounded controls exceed the caller's acceptance budget.
fn curve_control_points(
    ctx: &EncodingContext,
    chain: &[Point<f32>],
    tol: f32,
    max_points: usize,
) -> Vec<Point<i32>> {
    let [points] = curve_controls(ctx, chain, [tol], max_points);
    points
}

fn curve_controls<const N: usize>(
    ctx: &EncodingContext,
    chain: &[Point<f32>],
    tolerances: [f32; N],
    max_points: usize,
) -> [Vec<Point<i32>>; N] {
    let limits = tolerances.map(|tol| tol * tol);
    let tightest = limits.into_iter().fold(f32::INFINITY, f32::min);
    let mut results = [const { Vec::new() }; N];
    let mut remaining = N;
    let sm = smooth_chain(chain, 1.5);
    let mut idx = vec![0usize, sm.len() - 1];
    while remaining > 0 {
        let mut ctrl: Vec<Point<i32>> = Vec::with_capacity(idx.len());
        for &i in &idx {
            push_rounded(&mut ctrl, sm[i]);
        }
        // Check after rounding: an inserted point can collapse onto its
        // neighbor, leaving a two-point line that the caller still accepts.
        let exhausted = ctrl.len() < 2
            || ctrl.len() != idx.len()
            || idx.len() >= MAX_CURVE_POINTS
            || ctrl.len() > max_points;
        let (mut worst, mut worst_d) = (0usize, tightest);
        if !exhausted {
            let control: Vec<Point<f32>> = ctrl.iter().map(|p| p.as_f32()).collect();
            let curve = super::catmull_rom(&control);
            let curve_distance = Polyline::new(ctx, &curve);
            for (k, &p) in sm.iter().enumerate() {
                let d = curve_distance.distance(p);
                if d > worst_d && !idx.contains(&k) {
                    (worst, worst_d) = (k, d);
                }
            }
        }
        // Tolerances share the same sequence of inserted points. Save each
        // result at the iteration where its independent search would stop.
        for (result, limit) in results.iter_mut().zip(limits) {
            if result.is_empty() && (exhausted || worst_d <= limit) {
                remaining -= 1;
                if remaining == 0 {
                    *result = ctrl;
                    return results;
                }
                result.clone_from(&ctrl);
            }
        }
        let at = idx.partition_point(|&i| i < worst);
        idx.insert(at, worst);
    }
    results
}

fn curve_control_variants(
    ctx: &EncodingContext,
    chain: &[Point<f32>],
) -> impl Iterator<Item = Vec<Point<i32>>> {
    let [tight, loose] = curve_controls(ctx, chain, CURVE_TOLERANCES, MAX_CURVE_POINTS);
    let loose = (tight != loose).then_some(loose);
    std::iter::once(tight).chain(loose)
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
        // A deletion only changes the tangents of neighboring spans.
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

const SLOT_TILE: usize = 16;
const MISSING_TILE: usize = usize::MAX;

/// Pixel-to-basis-row lookup. Sparse fits allocate slots only for tiles
/// reached by a sample window. Slot numbers still follow first contact.
#[derive(Default)]
struct RasterSlots {
    width: usize,
    height: usize,
    tile_stride: usize,
    tiles: Vec<usize>,
    slots: Vec<i32>,
}

impl RasterSlots {
    fn reset(&mut self, width: usize, height: usize, tiled: bool) {
        self.width = width;
        self.height = height;
        self.slots.clear();
        self.tiles.clear();
        self.tile_stride = if tiled { width.div_ceil(SLOT_TILE) } else { 0 };
        if tiled {
            self.tiles
                .resize(self.tile_stride * height.div_ceil(SLOT_TILE), MISSING_TILE);
        } else {
            self.slots.resize(width * height, -1);
        }
    }

    #[inline]
    fn dense_row_chunk(&mut self, y: usize, x: usize, end: usize) -> &mut [i32] {
        let row = y * self.width;
        &mut self.slots[row + x..row + end]
    }

    /// The part of x..end in the current tile.
    #[inline]
    fn tiled_row_chunk(&mut self, y: usize, x: usize, end: usize) -> &mut [i32] {
        let tile = &mut self.tiles[y / SLOT_TILE * self.tile_stride + x / SLOT_TILE];
        if *tile == MISSING_TILE {
            *tile = self.slots.len();
            self.slots.resize(*tile + SLOT_TILE * SLOT_TILE, -1);
        }
        let column = x % SLOT_TILE;
        let start = *tile + y % SLOT_TILE * SLOT_TILE + column;
        let len = (end - x).min(SLOT_TILE - column);
        &mut self.slots[start..start + len]
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> i32 {
        if x >= self.width || y >= self.height {
            return -1;
        }
        if self.tile_stride == 0 {
            return self.slots[y * self.width + x];
        }
        let tile = self.tiles[y / SLOT_TILE * self.tile_stride + x / SLOT_TILE];
        if tile == MISSING_TILE {
            return -1;
        }
        self.slots[tile + y % SLOT_TILE * SLOT_TILE + x % SLOT_TILE]
    }
}

struct WeightTable {
    sigma_n: i32,
    reach: isize,
    weights: Vec<f32>,
    cutoff_sq: f32,
}

pub(crate) struct FitScratch {
    kernels: &'static SplinesContext,
    raster: RasterSlots,
    rows: Vec<f64>,
    pixels: Vec<u32>,
    weight_luts: Vec<WeightTable>,
    points: Vec<Point<i32>>,
    samples: Vec<super::ArcSample>,
    arc: f32,
    basis: Vec<f64>,
    basis_coeffs: usize,
    origin: (usize, usize),
}

impl Default for FitScratch {
    fn default() -> Self {
        Self {
            kernels: SplinesContext::get(),
            raster: RasterSlots::default(),
            rows: Vec::new(),
            pixels: Vec::new(),
            weight_luts: Vec::new(),
            points: Vec::new(),
            samples: Vec::new(),
            arc: 0.0,
            basis: Vec::new(),
            basis_coeffs: 0,
            origin: (0, 0),
        }
    }
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
        #[cfg(not(target_arch = "aarch64"))]
        let cos = self.kernels.cos;
        #[cfg(target_arch = "aarch64")]
        let cos = super::trig::cos_scalar;

        let frequencies = &BASIS_FREQUENCIES[1..coeffs];
        self.basis.clear();
        self.basis.reserve(self.samples.len() * coeffs);
        for k in 0..self.samples.len() {
            let phase = 31.0 * (k as f32 / self.arc).min(1.0) + 0.5;
            self.basis.push(std::f64::consts::SQRT_2);
            self.basis.extend(frequencies.iter().map(|&frequency| {
                // Evaluate the angle in f32; retain f64 accumulation and solving.
                std::f64::consts::SQRT_2 * cos(frequency * phase) as f64
            }));
        }
        self.basis_coeffs = coeffs;
    }

    fn weight_table(&mut self, sigma_n: i32, sigma: f32, reach: isize) -> usize {
        if let Some(i) = self
            .weight_luts
            .iter()
            .position(|table| table.sigma_n == sigma_n && table.reach == reach)
        {
            return i;
        }
        let len = ((reach as f32 + 1.0) * std::f32::consts::SQRT_2 * WEIGHT_LUT_STEPS) as usize + 2;
        let mut weights = vec![0.0; len];
        let row = BlobWeightRow {
            center: 0.0,
            step: 1.0 / WEIGHT_LUT_STEPS,
            dy: 0.0,
            inv_sigma: 1.0 / sigma,
            amp: 1.0,
        };
        (self.kernels.blob_weight_row)(row, 0, &mut weights);
        // Both interpolation factors are at most one. A cell whose endpoints
        // each pass this bound cannot contribute even at full sample amplitude.
        // Leave one extra cell for rounding in the squared-distance test.
        let max_amp = 0.25 * sigma;
        let cutoff = weights
            .iter()
            .rposition(|&weight| max_amp * (2.0 * weight) > 1e-7)
            .map_or(0.0, |i| (i + 2) as f32 / WEIGHT_LUT_STEPS);
        self.weight_luts.push(WeightTable {
            sigma_n,
            reach,
            weights,
            cutoff_sq: cutoff * cutoff,
        });
        self.weight_luts.len() - 1
    }

    /// Retain ordinary candidate buffers, without keeping an unusually large
    /// curve's allocations alive on every encoder worker.
    fn trim(&mut self) {
        const MAX_BYTES: usize = 32 * 1024 * 1024;
        fn trim<T>(v: &mut Vec<T>) {
            if v.capacity() > MAX_BYTES / size_of::<T>() {
                v.clear();
                v.shrink_to_fit();
            }
        }
        trim(&mut self.raster.slots);
        trim(&mut self.raster.tiles);
        trim(&mut self.rows);
        trim(&mut self.pixels);
        if self.basis.capacity() > MAX_BYTES / size_of::<f64>() {
            self.basis.clear();
            self.basis.shrink_to_fit();
            self.basis_coeffs = 0;
        }
    }
}

/// Fit support and number of basis columns.
#[derive(Clone, Copy, PartialEq)]
enum Precision {
    Full,
    Probe,
    Compact,
    Lean,
}

#[inline]
pub(super) fn add_scaled_scalar(dst: &mut [f64], src: &[f64], scale: f64) {
    for (dst, &src) in dst.iter_mut().zip(src) {
        *dst += scale * src;
    }
}

pub(super) fn accumulate_fit_scalar(
    ata: &mut FitMatrix,
    atb: &mut FitTargets,
    btb: &mut [f64; 3],
    row: &[f64],
    target: [f32; 3],
) {
    let target = target.map(f64::from);
    for (i, (&value, ata_row)) in row.iter().zip(ata).enumerate() {
        add_scaled_scalar(&mut ata_row[..=i], &row[..=i], value);
    }
    for (atb, target) in atb.iter_mut().zip(target) {
        add_scaled_scalar(&mut atb[..row.len()], row, target);
    }
    for (energy, value) in btb.iter_mut().zip(target) {
        *energy += value * value;
    }
}

#[inline]
pub(super) fn sub_dot_scalar(mut acc: f64, left: &[f64], right: &[f64]) -> f64 {
    for (&a, &b) in left.iter().zip(right) {
        acc -= a * b;
    }
    acc
}

#[inline]
pub(super) fn fit_score_scalar(ata: &FitMatrix, rhs: &[f64], sol: &[f64]) -> (f64, f64) {
    let (mut lin, mut quad) = (0.0, 0.0);
    for ((&value, &rhs), row) in sol.iter().zip(rhs).zip(ata) {
        lin += value * rhs;
        for (&a, &b) in row.iter().zip(sol) {
            quad += value * a * b;
        }
    }
    (lin, quad)
}

#[inline]
fn sub_dot(kernels: &SplinesContext, acc: f64, left: &[f64], right: &[f64]) -> f64 {
    // Most fits have three or six coefficients. A short reduction costs less
    // inline and keeps the sequential rounding for these systems.
    if left.len() < 8 {
        sub_dot_scalar(acc, left, right)
    } else {
        (kernels.fit_sub_dot)(acc, left, right)
    }
}

/// A leading block of the factor is also the factor for that block of ata.
fn factor_leading(kernels: &SplinesContext, ata: &FitMatrix, k: usize) -> FitMatrix {
    let mut l = [[0f64; MAX_COEFFS]; MAX_COEFFS];
    for i in 0..k {
        for j in 0..=i {
            let sum = sub_dot(
                kernels,
                ata[i][j] + if i == j { 1e-9 } else { 0.0 },
                &l[i][..j],
                &l[j][..j],
            );
            let value = if i == j {
                sum.max(1e-30).sqrt()
            } else {
                sum / l[j][j]
            };
            // Store the transpose in the upper triangle for backward substitution.
            l[i][j] = value;
            l[j][i] = value;
        }
    }
    l
}

fn solve_leading(
    kernels: &SplinesContext,
    l: &FitMatrix,
    k: usize,
    rhs: &[f64; MAX_COEFFS],
    out: &mut [f64; MAX_COEFFS],
) {
    out.fill(0.0);
    let mut y = [0f64; MAX_COEFFS];
    for i in 0..k {
        y[i] = sub_dot(kernels, rhs[i], &l[i][..i], &y[..i]) / l[i][i];
    }
    for i in (0..k).rev() {
        out[i] = sub_dot(kernels, y[i], &l[i][i + 1..k], &out[i + 1..k]) / l[i][i];
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
    kernels: &'static SplinesContext,
    ata: FitMatrix,
    atb: FitTargets,
    btb: [f64; 3],
    sigma_n: i32,
    coeffs: usize,
    arc: f32,
}

struct RasterParams {
    sigma_n: i32,
    sigma: f32,
    reach: isize,
    coeffs: usize,
    table: usize,
}

fn prepare_raster(
    points: &[Point<i32>],
    sigma_n: i32,
    precision: Precision,
    scratch: &mut FitScratch,
) -> Option<RasterParams> {
    scratch.geometry(points);
    let arc = scratch.arc;
    if arc < MIN_ARC {
        return None;
    }
    let mut coeffs = ((arc / ARC_PER_COEFF + 0.5) as usize).clamp(1, MAX_COEFFS);
    let sigma = sigma_n as f32 * sigma_step();
    let mut reach = (3.6 * sigma + 1.5).ceil() as isize;
    match precision {
        Precision::Full => {}
        Precision::Probe => {
            coeffs = coeffs.min(REFINE_COEFFS);
            reach = (2.6 * sigma + 1.0).ceil() as isize;
        }
        Precision::Compact => coeffs = coeffs.min(COMPACT_COEFFS),
        Precision::Lean => coeffs = coeffs.min(LONG_LINE_MAX_COEFFS),
    }
    scratch.prepare_basis(coeffs);
    let table = scratch.weight_table(sigma_n, sigma, reach);
    Some(RasterParams {
        sigma_n,
        sigma,
        reach,
        coeffs,
        table,
    })
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
    let params = prepare_raster(points, sigma_n, precision, scratch)?;
    // Expose the common three- and six-column fits to the compiler while
    // preserving the raster and normal-equation accumulation order.
    match params.coeffs {
        REFINE_COEFFS => raster_system::<REFINE_COEFFS>(targets, w, h, params, scratch),
        COMPACT_COEFFS => raster_system::<COMPACT_COEFFS>(targets, w, h, params, scratch),
        _ => raster_system::<0>(targets, w, h, params, scratch),
    }
}

fn raster_system<const FIXED_COEFFS: usize>(
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    params: RasterParams,
    scratch: &mut FitScratch,
) -> Option<FitSystem> {
    let coeffs = if FIXED_COEFFS == 0 {
        params.coeffs
    } else {
        FIXED_COEFFS
    };
    raster_rows::<FIXED_COEFFS>(w, h, &params, scratch)?;

    let mut ata = [[0f64; MAX_COEFFS]; MAX_COEFFS];
    let mut atb = [[0f64; MAX_COEFFS]; 3];
    let mut btb = [0f64; 3];
    let accumulate = scratch.kernels.fit_accumulate;
    for (&pix, row) in scratch.pixels.iter().zip(scratch.rows.chunks_exact(coeffs)) {
        let target = targets.each_ref().map(|plane| plane[pix as usize]);
        #[cfg(target_arch = "aarch64")]
        if FIXED_COEFFS != 0 {
            // SAFETY: NEON is part of the supported AArch64 target baseline.
            unsafe {
                super::fit_neon::accumulate_fit_fixed_neon(
                    &mut ata,
                    &mut atb,
                    &mut btb,
                    row.first_chunk::<FIXED_COEFFS>().unwrap(),
                    target,
                );
            }
            continue;
        }
        accumulate(&mut ata, &mut atb, &mut btb, row, target);
    }
    for i in 0..coeffs {
        for j in i + 1..coeffs {
            ata[i][j] = ata[j][i];
        }
    }

    Some(FitSystem {
        kernels: scratch.kernels,
        ata,
        atb,
        btb,
        sigma_n: params.sigma_n,
        coeffs,
        arc: scratch.arc,
    })
}

fn raster_rows<const FIXED_COEFFS: usize>(
    w: usize,
    h: usize,
    params: &RasterParams,
    scratch: &mut FitScratch,
) -> Option<()> {
    let reach = params.reach;
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
    // Samples are about one pixel apart. Allow room for tile padding before
    // switching away from the contiguous lookup used by compact fits.
    let span = (2 * reach + 1) as usize;
    let tiled = bw * bh > samples.len().saturating_mul(2 * (span + SLOT_TILE));
    scratch.raster.reset(bw, bh, tiled);
    scratch.rows.clear();
    scratch.pixels.clear();

    // The layout stays fixed across all samples. Dense rows need only one
    // slot slice, while tiled rows can cross several independently stored tiles.
    if tiled {
        splat_rows::<FIXED_COEFFS, true>(w, params, (bx0, by0, bx1, by1), scratch);
    } else {
        splat_rows::<FIXED_COEFFS, false>(w, params, (bx0, by0, bx1, by1), scratch);
    }
    (!scratch.pixels.is_empty()).then_some(())
}

fn splat_rows<const FIXED_COEFFS: usize, const TILED: bool>(
    w: usize,
    params: &RasterParams,
    (bx0, by0, bx1, by1): (isize, isize, isize, isize),
    scratch: &mut FitScratch,
) {
    let coeffs = if FIXED_COEFFS == 0 {
        params.coeffs
    } else {
        FIXED_COEFFS
    };
    let table = &scratch.weight_luts[params.table];
    let weight_lut = &table.weights;
    let sigma = params.sigma;
    let reach = params.reach;
    let samples = &scratch.samples;
    let add_scaled = scratch.kernels.fit_add_scaled;
    for (sample, basis) in samples.iter().zip(scratch.basis.chunks_exact(coeffs)) {
        let Point { x: cx, y: cy } = sample.position;
        let mult = sample.multiplier;
        let amp = 0.25 * sigma * mult;
        let cutoff_sq = if cx.is_finite() && mult.abs() <= 1.0 {
            table.cutoff_sq
        } else {
            f32::INFINITY
        };
        let (rx, ry) = (round_i32(cx) as isize, round_i32(cy) as isize);
        let x0 = (rx - reach).max(bx0);
        let x1 = (rx + reach).min(bx1);
        if x0 > x1 {
            continue;
        }
        let x_start = (x0 - bx0) as usize;
        let x_end = (x1 - bx0 + 1) as usize;
        for y in (ry - reach).max(by0)..=(ry + reach).min(by1) {
            let dy = y as f32 - cy;
            let dy_sq = dy * dy;
            if dy_sq > cutoff_sq {
                continue;
            }
            let mut x = x_start;
            while x < x_end {
                let local_y = (y - by0) as usize;
                let slots = if TILED {
                    scratch.raster.tiled_row_chunk(local_y, x, x_end)
                } else {
                    scratch.raster.dense_row_chunk(local_y, x, x_end)
                };
                let start = bx0 as usize + x;
                x = if TILED { x + slots.len() } else { x_end };
                for (x, slot) in (start..).zip(slots) {
                    let dx = x as f32 - cx;
                    let distance_sq = dx * dx + dy_sq;
                    if distance_sq > cutoff_sq {
                        continue;
                    }
                    let pos = distance_sq.sqrt() * WEIGHT_LUT_STEPS;
                    let cell = pos as usize;
                    let frac = pos - cell as f32;
                    let weight =
                        amp * (weight_lut[cell] * (1.0 - frac) + weight_lut[cell + 1] * frac);
                    if weight.abs() <= 1e-7 {
                        continue;
                    }
                    if *slot < 0 {
                        *slot = scratch.pixels.len() as i32;
                        scratch.pixels.push((y as usize * w + x) as u32);
                        // Keep the addition to +0.0 for first contact, including signed zero.
                        scratch
                            .rows
                            .extend(basis.iter().map(|&b| weight as f64 * b));
                    } else {
                        let row = &mut scratch.rows
                            [*slot as usize * coeffs..(*slot as usize + 1) * coeffs];
                        #[cfg(target_arch = "aarch64")]
                        if FIXED_COEFFS != 0 {
                            // The AArch64 kernel uses a fused multiply-add
                            // even in the tail. Keep small rows in this loop.
                            let row = row.first_chunk_mut::<FIXED_COEFFS>().unwrap();
                            let basis = basis.first_chunk::<FIXED_COEFFS>().unwrap();
                            for (dst, &src) in row.iter_mut().zip(basis) {
                                *dst = src.mul_add(weight as f64, *dst);
                            }
                            continue;
                        }
                        add_scaled(row, basis, weight as f64);
                    }
                }
            }
        }
    }
}

impl FitSystem {
    /// Fit the color to the gradient of the original image. A smooth pedestal
    /// under a line belongs in VarDCT: removing it along with the line creates
    /// a narrow trough that costs AC coefficients. Differencing both the image
    /// and the render basis eliminates constant and linear backgrounds without
    /// trusting the masked blur. Reuse the expensive rasterized basis rows.
    fn residual_system(&self, source: &Image3F, scratch: &FitScratch) -> Self {
        let mut result = Self {
            kernels: self.kernels,
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
        let planes = [0, 1, 2].map(|c| source.plane_data(c));
        let zero = [0.0; MAX_COEFFS];
        let accumulate = self.kernels.fit_accumulate;
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
                let neighbor = scratch.raster.get(nx.wrapping_sub(x0), ny.wrapping_sub(y0));
                // Interior edges once, and all edges of the finite support.
                if neighbor >= 0 && (neighbor as usize) < slot {
                    continue;
                }
                let other = if neighbor >= 0 {
                    &scratch.rows[neighbor as usize * k..][..k]
                } else {
                    &zero[..k]
                };
                let mut diff = [0.0; MAX_COEFFS];
                for ((d, &a), &b) in diff.iter_mut().zip(row).zip(other) {
                    *d = a - b;
                }
                let neighbor_pixel = ny * w + nx;
                let target = planes.map(|plane| plane[pixel] - plane[neighbor_pixel]);
                accumulate(
                    &mut result.ata,
                    &mut result.atb,
                    &mut result.btb,
                    &diff[..k],
                    target,
                );
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
        let (kx, ky, kb) = (kx.min(coeffs), ky.min(coeffs), kb.min(coeffs));
        let kernels = self.kernels;
        let factor = factor_leading(kernels, ata, kx.max(ky).max(kb));
        let mut sol = [[0f64; MAX_COEFFS]; 3];
        solve_leading(kernels, &factor, ky, &atb[1], &mut sol[1]);
        solve_leading(kernels, &factor, kx, &atb[0], &mut sol[0]);
        // B is stored as a residual of Y (base correlation 1.0): fit the residual
        let mut rhs = atb[2];
        for (rhs, row) in rhs[..coeffs].iter_mut().zip(ata) {
            *rhs = sub_dot(kernels, *rhs, &row[..ky], &sol[1][..ky]);
        }
        let mut residual = [0f64; MAX_COEFFS];
        solve_leading(kernels, &factor, kb, &rhs, &mut residual);
        let [_, y, b] = &mut sol;
        for ((b, &y), &residual) in b[..coeffs].iter_mut().zip(y.iter()).zip(&residual) {
            *b = y + residual;
        }

        let (mut err, mut e0) = (0f64, 0f64);
        for c in 0..3 {
            // Short fits retain scalar accumulation, including the common
            // three- and six-coefficient budgets.
            let (lin, quad) = if coeffs < 8 {
                fit_score_scalar(ata, &atb[c][..coeffs], &sol[c][..coeffs])
            } else {
                (kernels.fit_score)(ata, &atb[c][..coeffs], &sol[c][..coeffs])
            };
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
    scratch: &mut FitScratch,
) -> Option<(Fit, FitSystem)> {
    let mut best: Option<(Fit, FitSystem)> = None;
    for &n in widths {
        if let Some(system) = fit_system(targets, w, h, points, n, Precision::Full, scratch) {
            let fit = system.solve(None);
            if best.as_ref().is_none_or(|(b, _)| fit.gain() > b.gain()) {
                best = Some((fit, system));
            }
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
        if pts.array_windows::<2>().any(|p| p[0] == p[1]) {
            return f64::INFINITY;
        }
        fit_at(targets, w, h, pts, sigma_n, None, Precision::Probe, scratch)
            .map_or(f64::INFINITY, |f| -f.gain())
    };
    // A control point only shapes the curve a few spans around it: rank its
    // moves on that sub-curve instead of re-fitting the whole spline.
    let mut points = points.to_vec();
    let mut previous = None;
    for _ in 0..sweeps {
        let mut moved = false;
        for i in 0..points.len() {
            let lo = i.saturating_sub(REFINE_NEIGHBORS);
            let hi = (i + REFINE_NEIGHBORS).min(points.len() - 1);
            let local = &mut points[lo..=hi];
            let center = i - lo;
            let start = local[center];
            let start_cost = match previous {
                Some((a, b, value)) if (a, b) == (lo, hi) => value,
                _ => cost(local, scratch),
            };
            let mut best_point = start;
            let mut best = start_cost;
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let trial = Point::new(best_point.x + dx, best_point.y + dy);
                local[center] = trial;
                // A reverse move can revisit the starting geometry after an
                // accepted step. Neighboring knots can also share a window.
                let value = if trial == start {
                    start_cost
                } else {
                    cost(local, scratch)
                };
                if value < best - 1e-12 {
                    best = value;
                    best_point = trial;
                    moved = true;
                }
            }
            local[center] = best_point;
            previous = Some((lo, hi, best));
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
    let per_point = sm.len() as f32 / (points.len().saturating_sub(1) * 16).max(1) as f32;
    let slack = (3 * sm.len() / points.len().max(1) + 8) as f32;
    let mut points = points.to_vec();
    let mut control: Vec<_> = points.iter().map(|&p| p.as_f32()).collect();
    let mut distinct = !points.array_windows::<2>().any(|p| p[0] == p[1]);
    let mut last_cost = None;
    for _ in 0..sweeps {
        let mut moved = false;
        for i in 0..points.len() {
            let spans = i.saturating_sub(2)..(i + 2).min(points.len() - 1);
            // The curve runs along the chain. Search only the matching stretch.
            let first = spans.start * 16;
            let last = spans.end * 16;
            let lo = ((first as f32 * per_point - slack).max(0.0) as usize).min(sm.len() - 2);
            let hi = ((last as f32 * per_point + slack) as usize + 2).min(sm.len());
            let cost = |control: &[Point<f32>], cutoff| {
                let mut cost = -0.0f32;
                for p in catmull_rom_spans(control, spans.clone()) {
                    cost += distance.distance_range(p, lo..hi - 1);
                    // Squared distances cannot recover after crossing the threshold.
                    if cost >= cutoff {
                        break;
                    }
                }
                cost
            };
            // Adjacent controls can affect the same spans and use the same
            // segment range. The previous accepted cost is already exact.
            let mut best = match last_cost {
                Some((start, end, value)) if (start, end) == (spans.start, spans.end) => value,
                _ if distinct => cost(&control, f32::INFINITY),
                _ => f32::INFINITY,
            };
            let left = points[..i].last().copied();
            let right = points.get(i + 1).copied();
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
                let trial = Point::new(previous.x + dx, previous.y + dy);
                if left == Some(trial) || right == Some(trial) {
                    continue;
                }
                points[i] = trial;
                if !distinct && points.array_windows::<2>().any(|p| p[0] == p[1]) {
                    points[i] = previous;
                    continue;
                }
                control[i] = trial.as_f32();
                let cutoff = best - 1e-6;
                let value = cost(&control, cutoff);
                if value < cutoff {
                    best = value;
                    distinct = true;
                    moved = true;
                } else {
                    points[i] = previous;
                    control[i] = previous.as_f32();
                }
            }
            last_cost = Some((spans.start, spans.end, best));
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
    system: FitSystem,
    spline: QuantizedSpline,
}

/// Seeded-width fit of the unrefined control points. Long chains pick the
/// width on a Compact fit and build the full system only for the winner.
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
    let seeds = &seeds[..SEEDED_WIDTHS];
    let (fit, system) = if full_coeffs(&points, scratch) > COMPACT_COEFFS {
        let (n, _) = compact_width(targets, w, h, &points, seeds, scratch)?;
        fit_best(targets, w, h, &points, &[n], scratch)?
    } else {
        fit_best(targets, w, h, &points, seeds, scratch)?
    };
    if fit.explained() < QUICK_MIN_EXPLAINED {
        return None;
    }
    let spline = quantize(&points, &fit);
    Some(Primary {
        points,
        fit,
        system,
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
        mut system,
        ..
    } = primary;
    if fit.explained() > REFINE_MIN_EXPLAINED {
        // Align the knots against the ridge, then nudge them on the raster
        // fit. Sparse variants skip the raster pass: it costs time there
        // without changing what the RD gate picks.
        let aligned = refine_geometric(ctx, &points, &chain.points, 2);
        let refined = refine(targets, w, h, &aligned, fit.sigma_n, 1, scratch);
        let at = SIGMA_LATTICE
            .iter()
            .position(|&n| n == fit.sigma_n)
            .unwrap_or(0);
        let nearby = &SIGMA_LATTICE[at.saturating_sub(1)..(at + 2).min(SIGMA_LATTICE.len())];
        let same_geometry = refined == points;
        let mut best_gain = None;
        let mut best_fit = None;
        for &width in nearby {
            let (gain, fitted) = if same_geometry && width == fit.sigma_n {
                // Keep the incumbent's place in the width search without
                // rebuilding its equations.
                (fit.gain(), None)
            } else {
                let Some(system) =
                    fit_system(targets, w, h, &refined, width, Precision::Full, scratch)
                else {
                    continue;
                };
                let fit = system.solve(None);
                (fit.gain(), Some((fit, system)))
            };
            if best_gain.is_none_or(|best| gain > best) {
                best_gain = Some(gain);
                best_fit = fitted;
            }
        }
        // Width and geometry change pixel support, so compare energy removed.
        if let Some((better, better_system)) = best_fit
            && better.gain() > fit.gain()
        {
            (points, fit, system) = (refined, better, better_system);
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
    for sparse in curve_control_variants(ctx, &chain.points) {
        if sparse.len() >= 2 {
            let refined = refine_geometric(ctx, &sparse, &chain.points, 2);
            if !geometries.contains(&refined) {
                geometries.push(refined);
            }
        }
    }
    for (g, geometry) in geometries.iter().enumerate() {
        if !inside(geometry, w, h) || geometry.array_windows::<2>().any(|p| p[0] == p[1]) {
            continue;
        }
        let other;
        let system = if g == 0 {
            // The winning equations are already available. Rebuild only the
            // basis rows needed by the gradient fit.
            let Some(params) = prepare_raster(geometry, fit.sigma_n, Precision::Full, scratch)
            else {
                continue;
            };
            let rows = match params.coeffs {
                REFINE_COEFFS => raster_rows::<REFINE_COEFFS>(w, h, &params, scratch),
                COMPACT_COEFFS => raster_rows::<COMPACT_COEFFS>(w, h, &params, scratch),
                _ => raster_rows::<0>(w, h, &params, scratch),
            };
            if rows.is_none() {
                continue;
            }
            &system
        } else {
            let Some(fitted) = fit_system(
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
            other = fitted;
            &other
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
    // A fit explaining less than half its footprint is admitted on the gate's
    // say-so alone; weight its bits by how far below that it sits.
    let bits_factor = (0.5 / fit.explained() as f32).max(1.0).powi(2);
    Some(Candidate {
        alts,
        bits_factor,
        dot: false,
    })
}

/// Economical representations for a long continuation: six color coefficients
/// on the dense geometry, then curve-aware sparse geometries with full, lean
/// and luma-only color. A continuation is faint and nearly straight; dense
/// points and full color alone price most of them out of the gate.
fn fit_extension(
    ctx: &EncodingContext,
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    chain: &Chain,
    scratch: &mut FitScratch,
) -> Option<Candidate> {
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
    let mut alts = Vec::new();
    if spline.dct[..3].iter().flatten().any(|&v| v != 0) {
        alts.push(spline);
    }
    let at = SIGMA_LATTICE
        .iter()
        .position(|&n| n == fit.sigma_n)
        .unwrap_or(0);
    let nearby = &SIGMA_LATTICE[at.saturating_sub(1)..(at + 2).min(SIGMA_LATTICE.len())];
    let lean = ((chain.points.len() as f32 / LEAN_ARC_PER_COEFF + 0.5) as usize).max(1);
    for sparse in curve_control_variants(ctx, &chain.points) {
        if sparse.len() < 2 || sparse.len() >= points.len() {
            continue;
        }
        let sparse = refine_geometric(ctx, &sparse, &chain.points, 2);
        if !inside(&sparse, w, h) || sparse.array_windows::<2>().any(|p| p[0] == p[1]) {
            continue;
        }
        let budgets = [(1, lean, 1), (0, lean, 0)];
        let mut best: [Option<Fit>; 2] = [None, None];
        for &n in nearby {
            let Some(system) = fit_system(targets, w, h, &sparse, n, Precision::Compact, scratch)
            else {
                continue;
            };
            for (free, best) in budgets.into_iter().zip(&mut best) {
                let fit = system.solve(Some(free));
                if best.as_ref().is_none_or(|b| fit.gain() > b.gain()) {
                    *best = Some(fit);
                }
            }
        }
        for ((kx, _, _), best) in budgets.into_iter().zip(best) {
            let Some(fit) = best else { continue };
            let mut spline = quantize(&sparse, &fit);
            if kx == 0 {
                spline.dct[0] = [0; 32];
                spline.dct[2] = [0; 32];
            }
            if spline.dct[1].iter().any(|&v| v != 0)
                && !alts
                    .iter()
                    .any(|a| a.points == spline.points && a.dct == spline.dct)
            {
                alts.push(spline);
            }
        }
    }
    (!alts.is_empty()).then_some(Candidate {
        alts,
        bits_factor: 1.0,
        dot: false,
    })
}

/// Whole-line alternatives of a long line's uniform-spacing geometry. The
/// lean stand-in fit builds the same systems, so it solves these up front
/// for the group fit instead of rasterizing that geometry twice.
pub(super) struct UniformLine {
    points: Vec<Point<i32>>,
    alts: Vec<QuantizedSpline>,
}

fn long_line_spacing(chain: &Chain) -> f32 {
    LONG_LINE_MIN_SPACING.max(chain.points.len() as f32 / LONG_LINE_SPANS)
}

/// Long-line widths, nearest the ridge scale first.
fn long_line_widths(chain: &Chain) -> [i32; LONG_LINE_WIDTHS.len()] {
    let predicted = SCALE_TO_SIGMA.0 * chain.scale + SCALE_TO_SIGMA.1;
    let mut widths = LONG_LINE_WIDTHS;
    widths.sort_by(|&a, &b| {
        (a as f32 * sigma_step() - predicted)
            .abs()
            .total_cmp(&(b as f32 * sigma_step() - predicted).abs())
    });
    widths
}

/// Lean luma-only alternatives for a long straight line. Coding value is left
/// to the RD tests: a faint line's explained fraction is dominated by grain.
/// Also returns the uniform geometry's whole-line alternatives for
/// `fit_whole_line`; `None` there means that geometry is unusable.
fn fit_long_line(
    ctx: &EncodingContext,
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    chain: &Chain,
    scratch: &mut FitScratch,
) -> (Option<Candidate>, Option<UniformLine>) {
    let points = uniform_control_points(&chain.points, long_line_spacing(chain));
    if points.len() < 2 || !inside(&points, w, h) {
        return (None, None);
    }
    let points = refine_geometric(ctx, &points, &chain.points, 2);
    if !inside(&points, w, h) || points.array_windows::<2>().any(|p| p[0] == p[1]) {
        return (None, None);
    }
    let widths = long_line_widths(chain);
    // Geometry and width determine the system; coefficient budgets only solve it.
    let systems: Vec<FitSystem> = widths[..LONG_LINE_SEEDED_WIDTHS]
        .iter()
        .filter_map(|&n| fit_system(targets, w, h, &points, n, Precision::Lean, scratch))
        .collect();
    let mut uniform_alts = Vec::new();
    whole_line_alts(&points, &systems, &mut uniform_alts);
    let uniform = Some(UniformLine {
        points: points.clone(),
        alts: uniform_alts,
    });
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
    let candidate = (!alts.is_empty()).then_some(Candidate {
        alts,
        bits_factor: LONG_LINE_BITS_FACTOR,
        dot: false,
    });
    (candidate, uniform)
}

/// Fit a well-supported join as a single straight, colored spline. These
/// candidates precede fragments so the greedy gate can buy the complete line
/// once. The lean luma-only fallback still handles faint, interrupted lines.
fn fit_joined_line(
    ctx: &EncodingContext,
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    chain: &Chain,
    scratch: &mut FitScratch,
) -> Option<Candidate> {
    let points = curve_control_points(ctx, &chain.points, JOINED_LINE_TOLERANCE, 2);
    if points.len() != 2 || !inside(&points, w, h) {
        return None;
    }
    let mut alts: Vec<QuantizedSpline> = Vec::new();
    // Include the narrowest width: fine grids can be thinner than the faint
    // lines handled by LONG_LINE_WIDTHS. Share each system across budgets.
    for &n in &SIGMA_LATTICE[..6] {
        let Some(system) = fit_system(targets, w, h, &points, n, Precision::Compact, scratch)
        else {
            continue;
        };
        for k in [1, REFINE_COEFFS, COMPACT_COEFFS] {
            let fit = system.solve(Some((k, k, k)));
            if fit.explained() < MIN_EXPLAINED {
                continue;
            }
            let spline = quantize(&points, &fit);
            // Different coefficient budgets often quantize to the same
            // spline. Keep the first occurrence, preserving RD tie order.
            // All alternatives here share the same two control points.
            if spline.dct[1].iter().any(|&v| v != 0)
                && !alts.iter().any(|alt| alt.dct == spline.dct)
            {
                alts.push(spline);
            }
        }
    }
    (!alts.is_empty()).then_some(Candidate {
        alts,
        bits_factor: LONG_LINE_BITS_FACTOR,
        dot: false,
    })
}

/// Whole-line alternatives for a collinear group: the long-line spacing and
/// curve-aware sparse geometries, in luma-only and lean color.
fn fit_whole_line(
    ctx: &EncodingContext,
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    chain: &Chain,
    uniform_fit: Option<&UniformLine>,
    scratch: &mut FitScratch,
) -> Vec<QuantizedSpline> {
    let uniform = uniform_control_points(&chain.points, long_line_spacing(chain));
    let [tight, loose] = curve_controls(ctx, &chain.points, CURVE_TOLERANCES, MAX_CURVE_POINTS);
    let loose = (loose != tight && loose != uniform).then_some(loose);
    let tight = (tight != uniform).then_some(tight);
    // The uniform geometry comes first; `fit_long_line` already fitted it.
    let mut geometries: Vec<Vec<Point<i32>>> = Vec::new();
    let mut alts: Vec<QuantizedSpline> = Vec::new();
    if let Some(fitted) = uniform_fit {
        geometries.push(fitted.points.clone());
        alts.extend_from_slice(&fitted.alts);
    }
    for points in tight.into_iter().chain(loose) {
        if points.len() < 2 || !inside(&points, w, h) {
            continue;
        }
        let points = refine_geometric(ctx, &points, &chain.points, 2);
        if inside(&points, w, h)
            && !points.array_windows::<2>().any(|p| p[0] == p[1])
            && !geometries.contains(&points)
        {
            geometries.push(points);
        }
    }
    let widths = long_line_widths(chain);
    let fitted = usize::from(uniform_fit.is_some());
    for points in &geometries[fitted..] {
        let systems: Vec<FitSystem> = widths[..LONG_LINE_SEEDED_WIDTHS]
            .iter()
            .filter_map(|&n| fit_system(targets, w, h, points, n, Precision::Lean, scratch))
            .collect();
        whole_line_alts(points, &systems, &mut alts);
    }
    alts
}

/// Luma-only and lean color alternatives of one whole-line geometry, best
/// width per coefficient budget, appended unless already present.
fn whole_line_alts(points: &[Point<i32>], systems: &[FitSystem], alts: &mut Vec<QuantizedSpline>) {
    for k in 1..=LONG_LINE_MAX_COEFFS {
        for chroma in [0, 1] {
            let mut best: Option<Fit> = None;
            for system in systems {
                let fit = system.solve(Some((chroma, k, chroma)));
                if best.as_ref().is_none_or(|b| fit.gain() > b.gain()) {
                    best = Some(fit);
                }
            }
            let Some(fit) = best else { continue };
            let mut spline = quantize(points, &fit);
            if chroma == 0 {
                spline.dct[0] = [0; 32];
                spline.dct[2] = [0; 32];
            }
            if spline.dct[1].iter().any(|&v| v != 0)
                && !alts
                    .iter()
                    .any(|a| a.points == spline.points && a.dct == spline.dct)
            {
                alts.push(spline);
            }
        }
    }
}

/// Whole-line alternatives for the groups that some candidate lies on; no
/// other group can hold an accepted spline.
fn fit_groups(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    targets: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    long_lines: &super::lines::LongLines,
    uniform: &[Option<UniformLine>],
    candidates: &[Candidate],
) -> Vec<LineGroup> {
    if long_lines.chains.is_empty() {
        return Vec::new();
    }
    // Group membership depends on the control points alone. Color and
    // width alternatives of the same curve share its geometry.
    let mut seen = std::collections::HashSet::new();
    let geometries: Vec<_> = candidates
        .iter()
        .flat_map(|candidate| candidate.alts.iter())
        .map(|alt| alt.points.as_slice())
        .filter(|&points| seen.insert(points))
        .collect();
    let curves = ctx
        .thread_pool
        .steal_map(scratch, geometries.len(), |i, _| {
            let points = geometries[i];
            let (lo, hi) = points.iter().fold(
                (
                    Point::new(i32::MAX, i32::MAX),
                    Point::new(i32::MIN, i32::MIN),
                ),
                |(lo, hi), p| {
                    (
                        Point::new(lo.x.min(p.x), lo.y.min(p.y)),
                        Point::new(hi.x.max(p.x), hi.y.max(p.y)),
                    )
                },
            );
            let (positions, tangents) = curve_samples(points);
            (lo, hi, positions, tangents)
        });
    ctx.thread_pool
        .steal_map(scratch, long_lines.chains.len(), |i, worker| {
            let chain = &long_lines.chains[i];
            let (lo, hi) = chain.points.iter().fold(
                (
                    Point::new(f32::MAX, f32::MAX),
                    Point::new(f32::MIN, f32::MIN),
                ),
                |(lo, hi), p| {
                    (
                        Point::new(lo.x.min(p.x), lo.y.min(p.y)),
                        Point::new(hi.x.max(p.x), hi.y.max(p.y)),
                    )
                },
            );
            let index = LineIndex::new(&chain.points, &unit_tangents(&chain.points));
            let held = curves.iter().any(|(a, b, positions, tangents)| {
                (b.x as f32) >= lo.x - GROUP_SLACK
                    && (a.x as f32) <= hi.x + GROUP_SLACK
                    && (b.y as f32) >= lo.y - GROUP_SLACK
                    && (a.y as f32) <= hi.y + GROUP_SLACK
                    && positions.len() >= 2
                    && index.holds(positions, tangents)
            });
            if !held {
                return None;
            }
            let alts = fit_whole_line(
                ctx,
                targets,
                w,
                h,
                chain,
                uniform[i].as_ref(),
                &mut worker.spline_fit,
            );
            worker.spline_fit.trim();
            (!alts.is_empty()).then(|| LineGroup {
                points: chain.points.clone(),
                alts,
            })
        })
        .into_iter()
        .flatten()
        .collect()
}

pub(super) fn fit_candidates(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    xyb: &Image3F,
    quant_field: &[f32],
    chains: &[Chain],
    extensions: &[Chain],
    long_lines: &super::lines::LongLines,
) -> (Vec<Candidate>, Vec<LineGroup>) {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let targets = line_targets(ctx, scratch, xyb, chains);
    let targets = &targets;
    let pretest = Pretest::new(ctx, distance, xyb, quant_field);
    let lean_start = long_lines.joined.len() + chains.len() + extensions.len();
    let fitted: Vec<(Option<Candidate>, Option<UniformLine>)> = ctx.thread_pool.steal_map(
        scratch,
        lean_start + long_lines.chains.len(),
        |i, worker| {
            let fit_scratch = &mut worker.spline_fit;
            if i >= lean_start {
                // Lean stand-ins still come last, after the colored and
                // curved candidates and the well-supported straight joins.
                let line = &long_lines.chains[i - lean_start];
                let (candidate, uniform) = fit_long_line(ctx, targets, w, h, line, fit_scratch);
                fit_scratch.trim();
                let candidate = candidate
                    .filter(|candidate| candidate.alts.iter().any(|alt| pretest.accepts(alt)));
                return (candidate, uniform);
            }
            let result = (|| {
                if i < long_lines.joined.len() {
                    let line = &long_lines.chains[long_lines.joined[i]];
                    let candidate = fit_joined_line(ctx, targets, w, h, line, fit_scratch)?;
                    return candidate
                        .alts
                        .iter()
                        .any(|alt| pretest.accepts(alt))
                        .then_some(candidate);
                }
                let i = i - long_lines.joined.len();
                if i >= chains.len() {
                    let candidate = fit_extension(
                        ctx,
                        targets,
                        w,
                        h,
                        &extensions[i - chains.len()],
                        fit_scratch,
                    )?;
                    return candidate
                        .alts
                        .iter()
                        .any(|alt| pretest.accepts(alt))
                        .then_some(candidate);
                }
                let primary = quick_primary(ctx, targets, w, h, &chains[i], fit_scratch)?;
                if !pretest.accepts(&primary.spline) {
                    return None;
                }
                expand(ctx, targets, xyb, w, h, &chains[i], primary, fit_scratch)
            })();
            fit_scratch.trim();
            (result, None)
        },
    );
    let mut uniform = Vec::with_capacity(long_lines.chains.len());
    let mut candidates = Vec::with_capacity(fitted.len());
    for (i, (candidate, line)) in fitted.into_iter().enumerate() {
        candidates.extend(candidate);
        if i >= lean_start {
            uniform.push(line);
        }
    }
    let groups = fit_groups(
        ctx,
        scratch,
        targets,
        w,
        h,
        long_lines,
        &uniform,
        &candidates,
    );
    (candidates, groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_basis_matches_selected_cosine() {
        let mut scratch = FitScratch::default();
        for points in [
            &[Point::new(4, 8), Point::new(132, 36)][..],
            &[Point::new(0, 0), Point::new(48, 89), Point::new(90, 71)],
        ] {
            scratch.geometry(points);
            let cos = scratch.kernels.cos;
            for coeffs in [1, 3, 6, 12, MAX_COEFFS] {
                scratch.prepare_basis(coeffs);
                assert_eq!(scratch.basis.len(), scratch.samples.len() * coeffs);
                for (k, row) in scratch.basis.chunks_exact(coeffs).enumerate() {
                    let t = 31.0 * (k as f32 / scratch.arc).min(1.0);
                    for (i, &value) in row.iter().enumerate() {
                        let expected = std::f64::consts::SQRT_2
                            * cos(std::f32::consts::PI / 32.0 * i as f32 * (t + 0.5)) as f64;
                        assert_eq!(value.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn raster_slots_match_dense_lookup_across_tile_edges() {
        let (w, h) = (97, 65);
        let mut dense = RasterSlots::default();
        let mut tiled = RasterSlots::default();
        for offset in [0, 13] {
            dense.reset(w, h, false);
            tiled.reset(w, h, true);
            for (y, start, end) in [(0, 0, 2), (15, 14, 35), (16, 14, 35), (64, 93, 97)] {
                let mut x = start;
                while x < end {
                    let chunk = tiled.tiled_row_chunk(y, x, end);
                    let reference = dense.dense_row_chunk(y, x, x + chunk.len());
                    for (i, (slot, expected)) in chunk.iter_mut().zip(reference).enumerate() {
                        *slot = (y * w + x + i + offset) as i32;
                        *expected = *slot;
                    }
                    x += chunk.len();
                }
            }
            for y in 0..=h {
                for x in 0..=w {
                    assert_eq!(tiled.get(x, y), dense.get(x, y));
                }
            }
            assert!(tiled.slots.len() < dense.slots.len());
        }
    }

    #[test]
    fn refinement_compares_gain_over_different_pixel_supports() {
        let ctx = EncodingContext::default();
        let (w, h) = (128, 96);
        let points = vec![Point::new(16, 48), Point::new(112, 48)];
        let mut truth = QuantizedSpline {
            points: points.clone(),
            dct: [[0; 32]; 4],
        };
        truth.dct[1][0] = 10;
        truth.dct[3][0] = 6;
        let mut image = Image3F::new(w, h);
        super::super::render_spline(&ctx, &truth, QUANT_ADJUST, &mut image, 1.0);
        let mut targets = std::array::from_fn(|c| image.plane_data(c).to_vec());
        // Unrelated detail enters the wider fit's support. Its raw residual
        // error grows even though that width removes more of the actual line.
        for x in 16..113 {
            targets[1][55 * w + x] += 0.1;
        }
        let mut scratch = FitScratch::default();
        let initial_system =
            fit_system(&targets, w, h, &points, 5, Precision::Full, &mut scratch).unwrap();
        let initial = initial_system.solve(None);
        let wider = fit_at(
            &targets,
            w,
            h,
            &points,
            6,
            None,
            Precision::Full,
            &mut scratch,
        )
        .unwrap();
        assert!(wider.err > initial.err);
        assert!(wider.gain() > initial.gain());
        assert!(initial.explained() > REFINE_MIN_EXPLAINED);
        let primary = Primary {
            spline: quantize(&points, &initial),
            points,
            fit: initial,
            system: initial_system,
        };
        let chain = Chain {
            points: (16..=112).map(|x| Point::new(x as f32, 48.0)).collect(),
            scale: 1.5,
        };
        let candidate =
            expand(&ctx, &targets, &image, w, h, &chain, primary, &mut scratch).unwrap();
        assert_eq!(candidate.alts[0].dct[3][0], truth.dct[3][0]);
    }

    #[test]
    fn two_point_budget_preserves_acceptance_at_rounding_boundaries() {
        let ctx = EncodingContext::default();
        let (mut accepted, mut rejected) = (0, 0);
        for length in [2, 20, 120] {
            for origin in [0.499_999_97, 0.5, 0.500_000_06, 1_048_576.0] {
                for amplitude in [0.0, 0.1, 0.7, 1.5, 6.0] {
                    for step in [0.03, 0.65, 1.0] {
                        let chain: Vec<_> = (0..length)
                            .map(|i| {
                                let x = i as f32 * step;
                                Point::new(
                                    origin + x,
                                    origin + 0.2 * x + amplitude * (x * 0.1).sin(),
                                )
                            })
                            .collect();
                        for tol in [
                            JOINED_LINE_TOLERANCE.next_down(),
                            JOINED_LINE_TOLERANCE,
                            JOINED_LINE_TOLERANCE.next_up(),
                        ] {
                            let reference =
                                curve_control_points(&ctx, &chain, tol, MAX_CURVE_POINTS);
                            let limited = curve_control_points(&ctx, &chain, tol, 2);
                            if reference.len() <= 2 {
                                assert_eq!(limited, reference);
                                accepted += 1;
                            } else {
                                assert!(limited.len() > 2);
                                rejected += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(accepted > 0 && rejected > 0);
    }

    #[test]
    fn joined_line_fit_preserves_color_and_rejects_bends() {
        let ctx = EncodingContext::default();
        let (w, h) = (256, 96);
        let targets = [0.012f32, -0.35, -0.15].map(|color| {
            (0..w * h)
                .map(|i| {
                    if (16..240).contains(&(i % w)) {
                        let d = (i / w) as f32 - 48.0;
                        color * (-0.5 * (d / 0.6).powi(2)).exp()
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mut chain = Chain {
            points: (16..240).map(|x| Point::new(x as f32, 48.0)).collect(),
            scale: 0.7,
        };
        let mut scratch = FitScratch::default();
        let fit = fit_joined_line(&ctx, &targets, w, h, &chain, &mut scratch).unwrap();
        assert!(fit.alts.iter().all(|s| s.points.len() == 2));
        assert!(
            fit.alts
                .iter()
                .enumerate()
                .all(|(i, s)| { fit.alts[..i].iter().all(|earlier| earlier.dct != s.dct) })
        );
        assert!(
            fit.alts
                .iter()
                .any(|s| { s.dct[0].iter().any(|&c| c != 0) && s.dct[2].iter().any(|&c| c != 0) })
        );
        for p in &mut chain.points {
            p.y += 8.0 * ((p.x - 16.0) * std::f32::consts::PI / 223.0).sin();
        }
        assert!(fit_joined_line(&ctx, &targets, w, h, &chain, &mut scratch).is_none());
    }

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
        for precision in [Precision::Lean, Precision::Compact] {
            for sigma_n in SIGMA_LATTICE {
                let fixed =
                    fit_system(&targets, w, h, &points, sigma_n, precision, &mut scratch).unwrap();
                let coeffs = if precision == Precision::Lean {
                    LONG_LINE_MAX_COEFFS
                } else {
                    COMPACT_COEFFS
                };
                assert_eq!(fixed.coeffs, coeffs);
                let params = prepare_raster(&points, sigma_n, precision, &mut scratch).unwrap();
                let dynamic = raster_system::<0>(&targets, w, h, params, &mut scratch).unwrap();
                assert_eq!(fixed.ata, dynamic.ata);
                assert_eq!(fixed.atb, dynamic.atb);
                assert_eq!(fixed.btb, dynamic.btb);
            }
        }
    }

    #[test]
    fn winning_fit_equations_match_rebuilt_rows_after_other_fits() {
        let (w, h) = (165, 124);
        let mut source = Image3F::new(w, h);
        let targets = std::array::from_fn(|c| {
            let data: Vec<_> = (0..w * h)
                .map(|i| ((i * 19 + c * 31) % 101) as f32 * 0.003 - 0.15)
                .collect();
            for (y, row) in data.chunks_exact(w).enumerate() {
                source.plane_row_mut(c, y).copy_from_slice(row);
            }
            data
        });
        let shapes = [
            [Point::new(5, 9), Point::new(42, 14), Point::new(75, 20)],
            [Point::new(4, 3), Point::new(63, 79), Point::new(153, 115)],
        ];
        let mut scratch = FitScratch::default();
        for points in shapes {
            let (fit, system) = fit_best(&targets, w, h, &points, &[2, 4, 6], &mut scratch)
                .expect("candidate fits");
            fit_system(&targets, w, h, &shapes[0], 2, Precision::Lean, &mut scratch).unwrap();
            let params =
                prepare_raster(&points, fit.sigma_n, Precision::Full, &mut scratch).unwrap();
            raster_rows::<0>(w, h, &params, &mut scratch).unwrap();
            let gradient = system.residual_system(&source, &scratch);
            let rebuilt = fit_system(
                &targets,
                w,
                h,
                &points,
                fit.sigma_n,
                Precision::Full,
                &mut scratch,
            )
            .unwrap();
            let rebuilt_gradient = rebuilt.residual_system(&source, &scratch);
            for (a, b) in [(&system, &rebuilt), (&gradient, &rebuilt_gradient)] {
                assert_eq!(
                    a.ata.map(|row| row.map(f64::to_bits)),
                    b.ata.map(|row| row.map(f64::to_bits))
                );
                assert_eq!(
                    a.atb.map(|row| row.map(f64::to_bits)),
                    b.atb.map(|row| row.map(f64::to_bits))
                );
                assert_eq!(a.btb.map(f64::to_bits), b.btb.map(f64::to_bits));
            }
        }
    }

    #[test]
    fn weight_cutoff_preserves_pixel_order_and_normal_equations() {
        let (w, h) = (109, 95);
        let targets = std::array::from_fn(|c| {
            (0..w * h)
                .map(|i| ((i * 19 + c * 31) % 101) as f32 * 0.003 - 0.15)
                .collect::<Vec<_>>()
        });
        let points = [Point::new(0, 1), Point::new(52, 45), Point::new(108, 93)];
        for sigma_n in SIGMA_LATTICE {
            for precision in [Precision::Full, Precision::Lean] {
                let mut scratch = FitScratch::default();
                let bounded =
                    fit_system(&targets, w, h, &points, sigma_n, precision, &mut scratch).unwrap();
                let pixels = scratch.pixels.clone();
                let rows = scratch.rows.clone();
                scratch.weight_luts[0].cutoff_sq = f32::INFINITY;
                let complete =
                    fit_system(&targets, w, h, &points, sigma_n, precision, &mut scratch).unwrap();
                assert_eq!(pixels, scratch.pixels);
                assert_eq!(rows, scratch.rows);
                assert_eq!(bounded.ata, complete.ata);
                assert_eq!(bounded.atb, complete.atb);
                assert_eq!(bounded.btb, complete.btb);
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
            (1, 4, Precision::Lean),
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
        let sparse = curve_control_points(&kernels, &chain, 1.0, MAX_CURVE_POINTS);
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
        assert!(sparse.array_windows::<2>().all(|p| p[0] != p[1]));
    }

    #[test]
    fn shared_curve_controls_match_separate_tolerances() {
        let ctx = EncodingContext::default();
        let mut distinct = false;
        for amplitude in [0.0, 0.75, 1.25, 3.0, 12.0] {
            let chain: Vec<_> = (0..=192)
                .map(|i| {
                    let x = i as f32 * 0.75;
                    Point::new(x + 0.25, 40.0 + amplitude * (x / 9.0).sin())
                })
                .collect();
            for max_points in [2, 6, MAX_CURVE_POINTS] {
                let shared = curve_controls(&ctx, &chain, CURVE_TOLERANCES, max_points);
                let separate =
                    CURVE_TOLERANCES.map(|tol| curve_control_points(&ctx, &chain, tol, max_points));
                assert_eq!(shared, separate);
                distinct |= shared[0] != shared[1];
            }
        }
        assert!(distinct);
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
    fn lean_fit_preserves_compact_scores_and_quantization() {
        let (w, h) = (128, 96);
        let targets = std::array::from_fn(|c| {
            (0..w * h)
                .map(|i| (((i * 19 + c * 31) % 101) as f32 - 50.0) * 0.003)
                .collect()
        });
        let mut scratch = FitScratch::default();
        for points in [
            [Point::new(3, 5), Point::new(123, 89)],
            [Point::new(40, 47), Point::new(56, 49)],
        ] {
            for sigma_n in LONG_LINE_WIDTHS {
                let system = |precision, scratch: &mut FitScratch| {
                    fit_system(&targets, w, h, &points, sigma_n, precision, scratch).unwrap()
                };
                let compact = system(Precision::Compact, &mut scratch);
                let lean = system(Precision::Lean, &mut scratch);
                assert_eq!(lean.coeffs, compact.coeffs.min(LONG_LINE_MAX_COEFFS));
                for k in 1..=LONG_LINE_MAX_COEFFS {
                    for chroma in [0, 1] {
                        let budget = Some((chroma, k, chroma));
                        let full = compact.solve(budget);
                        let trimmed = lean.solve(budget);
                        assert_eq!(full.err.to_bits(), trimmed.err.to_bits());
                        assert_eq!(full.e0.to_bits(), trimmed.e0.to_bits());
                        assert_eq!(
                            quantize(&points, &full).dct,
                            quantize(&points, &trimmed).dct
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fit_kernels_cover_each_coefficient_count_and_tail() {
        let kernels = SplinesContext::get();
        let row: [f64; MAX_COEFFS + 1] = std::array::from_fn(|i| (i as f64 - 13.0) * 0.17);
        let target = [-0.91, 0.17, 0.42];
        for k in 0..=MAX_COEFFS {
            let basis = &row[1..=k];
            let mut dst = [0.53; MAX_COEFFS + 2];
            let mut expected = dst;
            (kernels.fit_add_scaled)(&mut dst[1..=k], basis, f64::from(target[0]));
            add_scaled_scalar(&mut expected[1..=k], basis, f64::from(target[0]));
            for (actual, expected) in dst.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-14, "k={k}");
            }

            let mut ata = [[0.53; MAX_COEFFS]; MAX_COEFFS];
            let mut atb = [[-0.41; MAX_COEFFS]; 3];
            let mut btb = [0.13, 0.57, 0.81];
            let (mut expected_ata, mut expected_atb, mut expected_btb) = (ata, atb, btb);
            (kernels.fit_accumulate)(&mut ata, &mut atb, &mut btb, basis, target);
            accumulate_fit_scalar(
                &mut expected_ata,
                &mut expected_atb,
                &mut expected_btb,
                basis,
                target,
            );
            assert_eq!(
                btb.map(f64::to_bits),
                expected_btb.map(f64::to_bits),
                "k={k}"
            );
            for (actual, expected) in ata
                .iter()
                .chain(&atb)
                .flatten()
                .zip(expected_ata.iter().chain(&expected_atb).flatten())
            {
                assert!((actual - expected).abs() <= 1e-14, "k={k}");
            }

            let left = &row[..k];
            let actual = (kernels.fit_sub_dot)(5.2, left, basis);
            let expected = sub_dot_scalar(5.2, left, basis);
            assert!((actual - expected).abs() <= 1e-12, "k={k}");
        }
    }

    #[test]
    fn fit_score_matches_scalar_for_all_widths() {
        let score = SplinesContext::get().fit_score;
        let ata = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let x = ((i * j + i + j) % 23) as f64 * 0.07 - 0.9;
                if i == j { x + 1.0 } else { x }
            })
        });
        let rhs: [f64; MAX_COEFFS] = std::array::from_fn(|i| (i as f64 * 0.31).sin());
        let sol: [f64; MAX_COEFFS] = std::array::from_fn(|i| (i as f64 * 0.71).cos());
        for k in 0..=MAX_COEFFS {
            let (lin, quad) = score(&ata, &rhs[..k], &sol[..k]);
            let (want_lin, want_quad) = fit_score_scalar(&ata, &rhs[..k], &sol[..k]);
            assert!((lin - want_lin).abs() < 1e-12, "linear term, k={k}");
            assert!((quad - want_quad).abs() < 1e-12, "quadratic term, k={k}");
        }
    }

    #[test]
    fn fit_energy_preserves_f32_squares_bitwise() {
        let accumulate = SplinesContext::get().fit_accumulate;
        let mut ata = [[0.0; MAX_COEFFS]; MAX_COEFFS];
        let mut atb = [[0.0; MAX_COEFFS]; 3];
        let mut btb = [-0.0f64, 0.0, 0.81];
        let mut expected = btb;
        for target in [
            [-0.0, 0.0, -0.31],
            [f32::from_bits(1), -f32::MIN_POSITIVE, 0.19],
            [-0.91, 0.17, f32::MAX],
            [f32::MAX, -f32::MAX, f32::MAX],
        ] {
            accumulate(&mut ata, &mut atb, &mut btb, &[1.0; 3], target);
            for (energy, value) in expected.iter_mut().zip(target.map(f64::from)) {
                *energy += value * value;
            }
            assert_eq!(btb.map(f64::to_bits), expected.map(f64::to_bits));
        }
    }

    #[test]
    fn cholesky_recovers_known_solution_for_each_leading_size() {
        let ata = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 40.0 } else { 1.0 }));
        let expected: [f64; MAX_COEFFS] = std::array::from_fn(|i| (i as f64 - 15.0) * 0.125);
        let kernels = SplinesContext::get();
        let factor = factor_leading(kernels, &ata, MAX_COEFFS);
        for k in 0..=MAX_COEFFS {
            let rhs = std::array::from_fn(|i| {
                ata[i][..k]
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &x)| a * x)
                    .sum()
            });
            let mut actual = [1.0; MAX_COEFFS];
            solve_leading(kernels, &factor, k, &rhs, &mut actual);
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
