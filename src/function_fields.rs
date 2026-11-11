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

//! Bounded fitting of a scalar profile over an additive spatial coordinate.
//!
//! This only proposes representations. The caller must compare their complete
//! coding cost and reconstruction with the incumbent before using one.
//!
//! [`fit`] proposes functions that are the whole plane (within two code
//! values); [`fit_residual`] proposes a function that explains most of the
//! soft samples and leaves the rest to a coded residual.

use crate::encode_image::AlphaPlane;

const MIN_DIM: usize = 32;
const MAX_DIM: usize = 16384;
/// Largest plane a function field is tried on.
pub(crate) const MAX_PIXELS: usize = 8 * 1024 * 1024;
const SAMPLE_DIM: usize = 128;
const SEARCH_PROFILE: usize = 4096;
static PROFILE_LENGTHS: [usize; 2] = [1024, 4096];
const MAX_ERROR: i32 = 2;

#[derive(Clone, Debug)]
pub(crate) struct NestedProfile {
    pub(crate) profile: Vec<i32>,
    pub(crate) u: Vec<i32>,
    pub(crate) v: Vec<i32>,
}

pub(crate) struct FittedProfile {
    pub(crate) model: NestedProfile,
    pub(crate) reconstructed: AlphaPlane,
    pub(crate) squared_error: u64,
    pub(crate) max_error: u8,
}

#[derive(Clone, Copy, Debug)]
enum Geometry {
    Linear(f64),
    Quadratic { cx: f64, cy: f64, ratio: f64 },
}

struct Samples {
    x: Vec<usize>,
    y: Vec<usize>,
    values: Vec<u8>,
}

impl Samples {
    fn new(data: &[u8], width: usize, height: usize) -> Self {
        Self::with_dim(data, width, height, SAMPLE_DIM)
    }

    fn with_dim(data: &[u8], width: usize, height: usize, dim: usize) -> Self {
        let coordinates = |length: usize| {
            let n = length.min(dim);
            (0..n)
                .map(|i| i * (length - 1) / (n - 1))
                .collect::<Vec<_>>()
        };
        let x = coordinates(width);
        let y = coordinates(height);
        let values = y
            .iter()
            .flat_map(|&y| x.iter().map(move |&x| data[y * width + x]))
            .collect();
        Self { x, y, values }
    }

    fn axes(
        &self,
        geometry: Geometry,
        width: usize,
        height: usize,
        length: usize,
    ) -> Option<(Vec<i32>, Vec<i32>)> {
        quantized_axes(
            geometry,
            width,
            height,
            length,
            self.x.iter().copied(),
            self.y.iter().copied(),
        )
    }
}

/// Fit at most two whole-field U8 candidates. A candidate keeps source zero
/// and maximum coverage exactly, has a maximum error of two code values and
/// RMSE at most 0.6. These are preliminary source bounds, not an RD decision.
///
/// Search uses at most 128x128 original samples and a fixed geometry budget.
/// Only the final two profile lengths visit the complete plane.
pub(crate) fn fit(plane: &AlphaPlane, width: usize, height: usize) -> Vec<FittedProfile> {
    let AlphaPlane::U8(data) = plane else {
        return Vec::new();
    };
    let Some(pixels) = width.checked_mul(height) else {
        return Vec::new();
    };
    if !(MIN_DIM..=MAX_DIM).contains(&width)
        || !(MIN_DIM..=MAX_DIM).contains(&height)
        || pixels > MAX_PIXELS
        || pixels != data.len()
    {
        return Vec::new();
    }
    let (mut lo, mut hi) = (255u8, 0u8);
    for &v in data {
        lo = lo.min(v);
        hi = hi.max(v);
        if hi - lo >= 8 {
            break;
        }
    }
    if hi - lo < 8 {
        return Vec::new();
    }
    let samples = Samples::new(data, width, height);
    let mut geometries = Vec::with_capacity(24);
    for slope in [0.0f64, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0] {
        geometries.push(Geometry::Linear(slope.atan()));
        if slope != 0.0 {
            geometries.push(Geometry::Linear(-slope.atan()));
        }
    }
    geometries.push(Geometry::Linear(std::f64::consts::FRAC_PI_2));
    if let Some(angle) = gradient_angle(data, width, height, &samples) {
        geometries.push(Geometry::Linear(angle));
    }
    for ratio in [0.25, 0.5, 1.0, 2.0, 4.0] {
        geometries.push(Geometry::Quadratic {
            cx: 0.5,
            cy: 0.5,
            ratio,
        });
    }
    if let Some(geometry) = moment_geometry(&samples, width, height) {
        geometries.push(geometry);
    }
    for invert in [false, true] {
        if let Some(geometry) = log_geometry(&samples, width, height, invert) {
            geometries.push(geometry);
        }
    }

    let mut bins = vec![MeanBin::EMPTY; SEARCH_PROFILE];
    let mut score = |geometry| sample_error(geometry, &samples, width, height, &mut bins);
    let mut linear = None;
    let mut quadratic = None;
    for geometry in geometries {
        let error = score(geometry);
        let target = match geometry {
            Geometry::Linear(_) => &mut linear,
            Geometry::Quadratic { .. } => &mut quadratic,
        };
        if error.is_finite() && target.as_ref().is_none_or(|&(_, best)| error < best) {
            *target = Some((geometry, error));
        }
    }
    // A poor free profile is not worth geometric refinement or a full encode.
    let mut best = None;
    for (geometry, error) in [linear, quadratic].into_iter().flatten() {
        if error > 4.0 {
            continue;
        }
        let (geometry, error) = refine(geometry, error, &mut score);
        if best.as_ref().is_none_or(|&(_, previous)| error < previous) {
            best = Some((geometry, error));
        }
    }
    let Some((geometry, _)) = best else {
        return Vec::new();
    };
    PROFILE_LENGTHS
        .into_iter()
        .filter_map(|length| fit_complete(geometry, data, width, height, length))
        .collect()
}

/// Code-value offset between the base function and its coded residual: the
/// base frame carries `P - RESIDUAL_BIAS` and the residual plane carries
/// `A - P + RESIDUAL_BIAS`, so the residual of every sample fits eight bits.
pub(crate) const RESIDUAL_BIAS: i32 = 128;
const RESIDUAL_LENGTH: usize = 4096;
const RESIDUAL_MARGIN: usize = 96;
const RESIDUAL_SEARCH: usize = 512;
const RESIDUAL_SAMPLE_DIM: usize = 256;
/// Truncation of the absolute error in the robust objective: a hard edge
/// crossing the field costs no more than a moderate misfit.
const RESIDUAL_CAP: i32 = 6;
/// Planes above this size pass a cheap parametric screen before the free
/// axes are fitted.
const SCREENED_PIXELS: usize = 512 * 512;

/// A function that explains most of a plane, plus what it leaves over.
pub(crate) struct ResidualFit {
    /// `model.profile` is biased by `-RESIDUAL_BIAS`; `reconstructed` holds
    /// the unbiased function samples.
    pub(crate) base: FittedProfile,
    /// `source - function + RESIDUAL_BIAS`, exact for every sample.
    pub(crate) residual: AlphaPlane,
    /// The function is a repeated tile: what it misses is sparse and is
    /// worth coding exactly.
    pub(crate) periodic: bool,
}

impl ResidualFit {
    /// The residual with every misfit of at most `zone` code values dropped.
    /// Samples whose source is fully transparent or opaque keep theirs, so
    /// hard coverage stays exact wherever the residual is coded exactly.
    pub(crate) fn dead_zone(&self, source: &AlphaPlane, zone: u8) -> AlphaPlane {
        let (AlphaPlane::U8(source), AlphaPlane::U8(residual)) = (source, &self.residual) else {
            return self.residual.clone();
        };
        AlphaPlane::U8(
            residual
                .iter()
                .zip(source)
                .map(|(&r, &s)| {
                    let keep = s == 0
                        || s == 255
                        || (i32::from(r) - RESIDUAL_BIAS).unsigned_abs() > u32::from(zone);
                    if keep { r } else { RESIDUAL_BIAS as u8 }
                })
                .collect(),
        )
    }

    /// The decoder's sum of the base and a (possibly lossy) residual plane.
    pub(crate) fn compose(&self, residual: &AlphaPlane) -> Option<AlphaPlane> {
        let (AlphaPlane::U8(function), AlphaPlane::U8(residual)) =
            (&self.base.reconstructed, residual)
        else {
            return None;
        };
        (function.len() == residual.len()).then(|| {
            AlphaPlane::U8(
                function
                    .iter()
                    .zip(residual)
                    .map(|(&f, &r)| {
                        (i32::from(f) - RESIDUAL_BIAS + i32::from(r)).clamp(0, 255) as u8
                    })
                    .collect(),
            )
        })
    }
}

/// Per-coordinate value histogram: robust profile and admissible range.
struct Bins {
    counts: Vec<[u32; 256]>,
    populations: Vec<Population>,
}

#[derive(Clone, Copy)]
struct Population {
    total: u32,
    lo: u8,
    hi: u8,
}

impl Population {
    const EMPTY: Self = Self {
        total: 0,
        lo: 255,
        hi: 0,
    };

    #[inline]
    fn add(&mut self, value: u8) {
        self.total += 1;
        self.lo = self.lo.min(value);
        self.hi = self.hi.max(value);
    }
}

impl Bins {
    fn new(length: usize) -> Self {
        Self {
            counts: vec![[0; 256]; length],
            populations: vec![Population::EMPTY; length],
        }
    }

    fn clear(&mut self) {
        for (row, pop) in self.counts.iter_mut().zip(&mut self.populations) {
            if pop.total != 0 {
                row[usize::from(pop.lo)..=usize::from(pop.hi)].fill(0);
                *pop = Population::EMPTY;
            }
        }
    }

    #[inline]
    fn add_row(&mut self, values: &[u8], u: &[i32], v: i32) {
        for (&value, &ux) in values.iter().zip(u) {
            let index = (ux + v) as usize;
            self.counts[index][usize::from(value)] += 1;
            self.populations[index].add(value);
        }
    }

    /// Upper median, moved into the range that keeps all residuals within eight bits.
    fn models(&self) -> impl Iterator<Item = (usize, &[u32; 256], u32, i32)> {
        self.counts
            .iter()
            .zip(&self.populations)
            .enumerate()
            .filter_map(|(i, (row, &pop))| {
                if pop.total == 0 {
                    return None;
                }
                let lo = usize::from(pop.lo);
                let hi = usize::from(pop.hi);
                let mut seen = 0;
                let median = row[lo..=hi]
                    .iter()
                    .position(|&count| {
                        seen += count;
                        seen > pop.total / 2
                    })
                    .expect("nonempty histogram");
                let model = clamp_residual_model((lo + median) as u8, pop.lo, pop.hi);
                Some((i, row, pop.total, model))
            })
    }

    fn profile_and_error(&self, profile: &mut [i32]) -> Option<u32> {
        let mut error = 0;
        let mut previous: Option<(usize, i32)> = None;
        for (i, row, total, value) in self.models() {
            error += robust_bin_error(row, total, value);
            profile[i] = value;
            if let Some((start, from)) = previous {
                let span = (i - start) as i32;
                for (j, sample) in profile[start + 1..i].iter_mut().enumerate() {
                    *sample = from + (value - from) * (j + 1) as i32 / span;
                }
            } else {
                profile[..i].fill(value);
            }
            previous = Some((i, value));
        }
        let (last, value) = previous?;
        profile[last + 1..].fill(value);
        Some(error)
    }

    fn add_samples(&mut self, u: &[i32], v: &[i32], values: &[u8]) {
        self.clear();
        for (row, &vy) in values.chunks_exact(u.len()).zip(v) {
            self.add_row(row, u, vy);
        }
    }
}

#[inline]
fn clamp_residual_model(value: u8, lo: u8, hi: u8) -> i32 {
    i32::from(value).clamp(
        i32::from(hi) - (255 - RESIDUAL_BIAS),
        i32::from(lo) + RESIDUAL_BIAS,
    )
}

#[inline]
fn robust_bin_error(row: &[u32; 256], total: u32, model: i32) -> u32 {
    // Start with every sample at the cap. Only nearby values reduce the cost.
    let radius = RESIDUAL_CAP - 1;
    let lo = (model - radius).max(0) as usize;
    let hi = (model + radius).min(255) as usize;
    let discount = row[lo..=hi]
        .iter()
        .enumerate()
        .map(|(i, &count)| count * (RESIDUAL_CAP - (model - (lo + i) as i32).abs()) as u32)
        .sum::<u32>();
    total * RESIDUAL_CAP as u32 - discount
}

fn robust_sample_error(
    geometry: Geometry,
    samples: &Samples,
    width: usize,
    height: usize,
    bins: &mut Bins,
) -> f64 {
    let Some((u, v)) = samples.axes(geometry, width, height, RESIDUAL_SEARCH) else {
        return f64::INFINITY;
    };
    bins.add_samples(&u, &v, &samples.values);
    let error: u32 = bins
        .models()
        .map(|(_, row, total, model)| robust_bin_error(row, total, model))
        .sum();
    f64::from(error) / samples.values.len() as f64
}

/// Share of the sampled soft values the parametric geometry's own profile
/// puts within two code values.
fn sample_explained(
    geometry: Geometry,
    samples: &Samples,
    width: usize,
    height: usize,
    bins: &mut Bins,
) -> f64 {
    let Some((u, v)) = samples.axes(geometry, width, height, RESIDUAL_SEARCH) else {
        return 0.0;
    };
    bins.add_samples(&u, &v, &samples.values);
    let (mut soft, mut explained) = (0u32, 0u32);
    for (_, row, total, model) in bins.models() {
        soft += total - row[0] - row[255];
        explained += row[(model - 2).max(1) as usize..=(model + 2).min(254) as usize]
            .iter()
            .sum::<u32>();
    }
    f64::from(explained) / f64::from(soft.max(1))
}

// Five shifts in eight lanes, so the robust cost can be accumulated with SIMD.
const SHIFT_LANES: usize = 8;
const SHIFTS: usize = 5;

#[inline]
fn add_shift_cost(cost: &mut [u16; SHIFT_LANES], models: &[u8; SHIFT_LANES], value: u8) {
    for (cost, &model) in cost.iter_mut().zip(models) {
        *cost += u16::from(value.abs_diff(model).min(RESIDUAL_CAP as u8));
    }
}

#[inline]
fn best_shift(cost: &[u16; SHIFT_LANES], step: i32) -> i32 {
    let k = (0..SHIFTS)
        .min_by_key(|&k| (cost[k], k.abs_diff(2)))
        .unwrap_or(2);
    (k as i32 - 2) * step
}

/// Free both axes: every column, then every row, moves to the coordinate
/// shift that best matches the current profile. This absorbs what the
/// parametric geometry cannot (off-centre or slightly non-elliptic fields,
/// separable products) while the representation stays `P[U(x) + V(y)]`.
fn free_axes(
    pool: &crate::thread_pool::ThreadPool,
    data: &[u8],
    width: usize,
    u: &mut [i32],
    v: &mut [i32],
) -> Option<Vec<i32>> {
    // Every second line across the axis being moved is enough to place it.
    const STRIDE: usize = 2;
    const BAND: usize = 64;
    let length = RESIDUAL_LENGTH as i32;
    let height = v.len();
    let jobs = height.div_ceil(BAND).min(pool.num_threads());
    let rows_per_job = height.div_ceil(jobs).next_multiple_of(STRIDE);
    let jobs = height.div_ceil(rows_per_job);
    let mut column_costs = vec![vec![[0u16; SHIFT_LANES]; width]; jobs];
    let mut models = vec![[0u8; SHIFT_LANES]; RESIDUAL_LENGTH];
    let mut bins = Bins::new(RESIDUAL_LENGTH);
    let mut p = vec![0i32; RESIDUAL_LENGTH];
    let mut candidate = vec![0i32; RESIDUAL_LENGTH];
    let mut previous_u = vec![0; width];
    let mut previous_v = vec![0; height];
    bins.add_samples(u, v, data);
    let mut best = bins.profile_and_error(&mut p)?;
    for _ in 0..6 {
        previous_u.copy_from_slice(u);
        previous_v.copy_from_slice(v);
        for step in [16i32, 4, 1] {
            for (s, models) in models.iter_mut().enumerate() {
                for (k, model) in models[..SHIFTS].iter_mut().enumerate() {
                    let index = (s as i32 + (k as i32 - 2) * step).clamp(0, length - 1);
                    *model = p[index as usize] as u8;
                }
            }
            // At most 2048 samples per row or column at a cost of six each.
            pool.for_each_mut(&mut column_costs, |index, costs| {
                costs.fill([0; SHIFT_LANES]);
                let start = index * rows_per_job;
                let lines = data[start * width..]
                    .chunks_exact(width)
                    .zip(&v[start..][..rows_per_job.min(height - start)]);
                for (row, &vy) in lines.step_by(STRIDE) {
                    for ((&value, &ux), cost) in row.iter().zip(u.iter()).zip(costs.iter_mut()) {
                        add_shift_cost(cost, &models[(ux + vy) as usize], value);
                    }
                }
            });
            let (vmin, vmax) = (*v.iter().min()?, *v.iter().max()?);
            let (totals, parts) = column_costs.split_first_mut()?;
            for costs in parts {
                for (total, cost) in totals.iter_mut().zip(costs) {
                    for (total, &part) in total.iter_mut().zip(cost.iter()) {
                        *total += part;
                    }
                }
            }
            for (ux, cost) in u.iter_mut().zip(totals) {
                *ux = (*ux + best_shift(cost, step)).clamp(-vmin, length - 1 - vmax);
            }
            // Rows: each one is independent of the others.
            let (umin, umax) = (*u.iter().min()?, *u.iter().max()?);
            let mut rows: Vec<_> = v.chunks_mut(rows_per_job).collect();
            pool.for_each_mut(&mut rows, |index, axes| {
                let start = index * rows_per_job;
                for (row, vy) in data[start * width..]
                    .chunks_exact(width)
                    .zip(axes.iter_mut())
                {
                    let mut cost = [0; SHIFT_LANES];
                    for (&value, &ux) in row.iter().zip(u.iter()).step_by(STRIDE) {
                        add_shift_cost(&mut cost, &models[(ux + *vy) as usize], value);
                    }
                    *vy = (*vy + best_shift(&cost, step)).clamp(-umin, length - 1 - umax);
                }
            });
        }
        bins.add_samples(u, v, data);
        let error = bins.profile_and_error(&mut candidate)?;
        if error >= best {
            u.copy_from_slice(&previous_u);
            v.copy_from_slice(&previous_v);
            break;
        }
        let converged = u64::from(best - error) * 200 < u64::from(best);
        std::mem::swap(&mut p, &mut candidate);
        best = error;
        if converged {
            break;
        }
    }
    Some(p)
}

/// Fit one whole-plane function meant to be completed by a coded residual.
/// Unlike [`fit`], samples the function cannot explain (hard edges, objects
/// lying over a glow) are left to the residual instead of rejecting the fit.
///
/// Returns `None` unless soft samples are a real share of the plane and the
/// function explains nearly all of them; the caller still has to race the
/// complete two-frame representation against the incumbent.
pub(crate) fn fit_residual(
    pool: &crate::thread_pool::ThreadPool,
    plane: &AlphaPlane,
    width: usize,
    height: usize,
) -> Option<ResidualFit> {
    let AlphaPlane::U8(data) = plane else {
        return None;
    };
    let pixels = width.checked_mul(height)?;
    if !(MIN_DIM..=MAX_DIM).contains(&width)
        || !(MIN_DIM..=MAX_DIM).contains(&height)
        || pixels > MAX_PIXELS
        || pixels != data.len()
    {
        return None;
    }
    let soft = data.iter().filter(|&&v| v != 0 && v != 255).count();
    if soft * 16 < pixels {
        return None;
    }
    let coarse = Samples::new(data, width, height);
    let samples = Samples::with_dim(data, width, height, RESIDUAL_SAMPLE_DIM);
    let mut geometries = Vec::with_capacity(24);
    for slope in [0.0f64, 0.25, 0.5, 1.0, 2.0, 4.0] {
        geometries.push(Geometry::Linear(slope.atan()));
        if slope != 0.0 {
            geometries.push(Geometry::Linear(-slope.atan()));
        }
    }
    geometries.push(Geometry::Linear(std::f64::consts::FRAC_PI_2));
    if let Some(angle) = gradient_angle(data, width, height, &coarse) {
        geometries.push(Geometry::Linear(angle));
    }
    // Circles in pixels, then the plane's own aspect and its neighbours.
    let aspect = ((height - 1) as f64 / (width - 1) as f64).powi(2);
    let mut centres = vec![(0.5, 0.5)];
    let moment = moment_geometry(&coarse, width, height);
    if let Some(Geometry::Quadratic { cx, cy, .. }) = moment {
        centres.push((cx, cy));
    }
    for (cx, cy) in centres {
        for ratio in [aspect, 1.0, 0.5 * aspect, 2.0 * aspect] {
            geometries.push(Geometry::Quadratic { cx, cy, ratio });
        }
    }
    geometries.extend(moment);
    for invert in [false, true] {
        geometries.extend(log_geometry(&coarse, width, height, invert));
    }
    // The candidates are scored side by side, then the best line and the
    // best ellipse are refined side by side; each job has its own histogram.
    let scorer = || {
        let mut bins = Bins::new(RESIDUAL_SEARCH);
        let samples = &samples;
        move |geometry| robust_sample_error(geometry, samples, width, height, &mut bins)
    };
    const JOBS: usize = 6;
    let errors: Vec<f64> = pool
        .map(JOBS, |job| {
            let mut score = scorer();
            geometries
                .iter()
                .skip(job)
                .step_by(JOBS)
                .map(|&geometry| score(geometry))
                .collect::<Vec<f64>>()
        })
        .into_iter()
        .enumerate()
        .fold(
            vec![f64::INFINITY; geometries.len()],
            |mut errors, (job, part)| {
                for (slot, error) in part.into_iter().enumerate() {
                    errors[job + slot * JOBS] = error;
                }
                errors
            },
        );
    let (mut linear, mut quadratic) = (None, None);
    for (&geometry, error) in geometries.iter().zip(errors) {
        let target = match geometry {
            Geometry::Linear(_) => &mut linear,
            Geometry::Quadratic { .. } => &mut quadratic,
        };
        if error.is_finite() && target.as_ref().is_none_or(|&(_, best)| error < best) {
            *target = Some((geometry, error));
        }
    }
    let refined = |seed: Option<(Geometry, f64)>| {
        seed.map(|(geometry, error)| refine(geometry, error, scorer()))
    };
    let (linear, quadratic) = pool.join(|| refined(linear), || refined(quadratic));
    let (geometry, _) = [linear, quadratic]
        .into_iter()
        .flatten()
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    // The free axes repair a geometry that is nearly right; they are the
    // expensive part and do not rescue one that is not. On 18 real planes
    // of about 1 MP the share was 0.92-0.95 where the fit went on to pass
    // and at most 0.37 elsewhere. The coarse search profile underrates a
    // steep field on a small plane, where the free axes cost little anyway.
    if pixels > SCREENED_PIXELS
        && sample_explained(
            geometry,
            &samples,
            width,
            height,
            &mut Bins::new(RESIDUAL_SEARCH),
        ) < 0.6
    {
        return None;
    }

    let (mut u, mut v) = axes(
        geometry,
        width,
        height,
        RESIDUAL_LENGTH - 2 * RESIDUAL_MARGIN,
    )?;
    for a in &mut u {
        *a += RESIDUAL_MARGIN as i32;
    }
    for a in &mut v {
        *a += RESIDUAL_MARGIN as i32;
    }
    let mut p = free_axes(pool, data, width, &mut u, &mut v)?;
    // Re-centre so the nested coder sees non-negative axes.
    let (umin, vmin) = (*u.iter().min()?, *v.iter().min()?);
    let shift = umin + vmin;
    for a in &mut u {
        *a -= umin;
    }
    for a in &mut v {
        *a -= vmin;
    }
    let shift = shift as usize;
    p.copy_within(shift.., 0);
    p.truncate(p.len() - shift);

    let mut function = Vec::with_capacity(pixels);
    let mut residual = Vec::with_capacity(pixels);
    // At least 95% of the soft samples must be within two code values.
    let missed_budget = soft / 20;
    let (mut missed, mut error, mut max_error) = (0usize, 0u64, 0i32);
    for (row, &vy) in data.chunks_exact(width).zip(&v) {
        for (&value, &ux) in row.iter().zip(&u) {
            let model = p[(ux + vy) as usize];
            let difference = i32::from(value) - model;
            missed += usize::from(value != 0 && value != 255 && difference.abs() > 2);
            error += (difference * difference) as u64;
            max_error = max_error.max(difference.abs());
            function.push(model as u8);
            residual.push((difference + RESIDUAL_BIAS) as u8);
        }
        if missed > missed_budget {
            return None;
        }
    }
    for value in &mut p {
        *value -= RESIDUAL_BIAS;
    }
    Some(ResidualFit {
        base: FittedProfile {
            model: NestedProfile { profile: p, u, v },
            reconstructed: AlphaPlane::U8(function),
            squared_error: error,
            max_error: max_error as u8,
        },
        residual: AlphaPlane::U8(residual),
        periodic: false,
    })
}

#[inline]
fn count_mismatches(a: &[u8], b: &[u8]) -> usize {
    // Summing lane counts lets the compiler vectorize long comparisons.
    a.iter()
        .zip(b)
        .map(|(&a, &b)| u32::from(a != b))
        .sum::<u32>() as usize
}

const PERIOD_MAX: usize = 256;

/// Smallest shift of at most [`PERIOD_MAX`] along one axis under which all
/// but one sample in 64 repeat, judged on every third line.
fn period(data: &[u8], width: usize, height: usize, horizontal: bool) -> Option<usize> {
    let length = if horizontal { width } else { height };
    (2..=PERIOD_MAX.min(length / 3)).find(|&shift| {
        // A shift that is not a period runs out of its budget within a few
        // lines, so the scan of all shifts stays cheap on ordinary planes.
        let mut different = 0usize;
        if horizontal {
            let budget = height.div_ceil(3) * (width - shift) / 64;
            for row in data.chunks_exact(width).step_by(3) {
                different += count_mismatches(&row[shift..], row);
                if different > budget {
                    return false;
                }
            }
        } else {
            let budget = (height - shift).div_ceil(3) * width / 64;
            for y in (shift..height).step_by(3) {
                different += count_mismatches(
                    &data[y * width..][..width],
                    &data[(y - shift) * width..][..width],
                );
                if different > budget {
                    return false;
                }
            }
        }
        true
    })
}

/// A plane that repeats one tile on an axis-aligned grid, as
/// `P[x mod p + p (y mod q)]` with the tile as the profile. The samples that
/// break the repetition (borders, a faded corner) are left in the residual.
///
/// `None` for planes without texture, without a period of at most
/// [`PERIOD_MAX`] on both axes, or whose tile leaves more than a quarter of
/// the plane's transitions unexplained.
pub(crate) fn fit_periodic(plane: &AlphaPlane, width: usize, height: usize) -> Option<ResidualFit> {
    let AlphaPlane::U8(data) = plane else {
        return None;
    };
    let pixels = width.checked_mul(height)?;
    if !(MIN_DIM..=MAX_DIM).contains(&width)
        || !(MIN_DIM..=MAX_DIM).contains(&height)
        || pixels > MAX_PIXELS
        || pixels != data.len()
    {
        return None;
    }
    let transitions: usize = data
        .chunks_exact(width)
        .map(|row| count_mismatches(&row[1..], row))
        .sum();
    if transitions * 64 < pixels {
        return None;
    }
    let p = period(data, width, height, true)?;
    let q = period(data, width, height, false)?;
    let cells = p * q;
    // Majority vote per tile cell, with the cell's range for the bias clamp.
    #[derive(Clone, Copy)]
    struct TileCell {
        count: u32,
        candidate: u8,
        lo: u8,
        hi: u8,
    }
    let mut vote = vec![
        TileCell {
            count: 0,
            candidate: 0,
            lo: 255,
            hi: 0
        };
        cells
    ];
    for (y, row) in data.chunks_exact(width).enumerate() {
        let base = (y % q) * p;
        let cells = &mut vote[base..base + p];
        for chunk in row.chunks(p) {
            for (&value, cell) in chunk.iter().zip(cells.iter_mut()) {
                if cell.count == 0 {
                    (cell.candidate, cell.count) = (value, 1);
                } else if cell.candidate == value {
                    cell.count += 1;
                } else {
                    cell.count -= 1;
                }
                cell.lo = cell.lo.min(value);
                cell.hi = cell.hi.max(value);
            }
        }
    }
    let mut tile: Vec<i32> = vote
        .into_iter()
        .map(|cell| clamp_residual_model(cell.candidate, cell.lo, cell.hi))
        .collect();
    let mut function = Vec::with_capacity(pixels);
    let mut residual = Vec::with_capacity(pixels);
    let (mut missed, mut error, mut max_error) = (0usize, 0u64, 0i32);
    for (y, row) in data.chunks_exact(width).enumerate() {
        let base = (y % q) * p;
        let tile_row = &tile[base..base + p];
        for chunk in row.chunks(p) {
            for (&value, &model) in chunk.iter().zip(tile_row) {
                let difference = i32::from(value) - model;
                missed += usize::from(difference != 0);
                error += (difference * difference) as u64;
                max_error = max_error.max(difference.abs());
                function.push(model as u8);
                residual.push((difference + RESIDUAL_BIAS) as u8);
            }
        }
        if missed > transitions / 4 {
            return None;
        }
    }
    for value in &mut tile {
        *value -= RESIDUAL_BIAS;
    }
    Some(ResidualFit {
        base: FittedProfile {
            model: NestedProfile {
                profile: tile,
                u: (0..width).map(|x| (x % p) as i32).collect(),
                v: (0..height).map(|y| (p * (y % q)) as i32).collect(),
            },
            reconstructed: AlphaPlane::U8(function),
            squared_error: error,
            max_error: max_error as u8,
        },
        residual: AlphaPlane::U8(residual),
        periodic: true,
    })
}

#[derive(Clone, Copy)]
enum Axis {
    Linear(f64),
    Quadratic { center: f64, ratio: f64 },
}

impl Axis {
    #[inline]
    fn at(self, x: usize, last: usize) -> f64 {
        match self {
            Self::Linear(slope) => slope * x as f64 / last as f64,
            Self::Quadratic { center, ratio } => ratio * (x as f64 / last as f64 - center).powi(2),
        }
    }

    fn extrema(self, last: usize) -> (f64, f64) {
        let first = self.at(0, last);
        let end = self.at(last, last);
        let mut lo = first.min(end);
        if let Self::Quadratic { center, .. } = self {
            // The minimum is at one of the two integer coordinates around the vertex.
            let center = (center * last as f64).clamp(0.0, last as f64);
            lo = lo
                .min(self.at(center.floor() as usize, last))
                .min(self.at(center.ceil() as usize, last));
        }
        (lo, first.max(end))
    }
}

fn quantized_axes(
    geometry: Geometry,
    width: usize,
    height: usize,
    length: usize,
    xs: impl ExactSizeIterator<Item = usize>,
    ys: impl ExactSizeIterator<Item = usize>,
) -> Option<(Vec<i32>, Vec<i32>)> {
    let (x_axis, y_axis) = match geometry {
        Geometry::Linear(angle) if angle.is_finite() => {
            (Axis::Linear(angle.cos()), Axis::Linear(angle.sin()))
        }
        Geometry::Quadratic { cx, cy, ratio }
            if (0.0625..=16.0).contains(&ratio)
                && (-0.25..=1.25).contains(&cx)
                && (-0.25..=1.25).contains(&cy) =>
        {
            (
                Axis::Quadratic {
                    center: cx,
                    ratio: 1.0,
                },
                Axis::Quadratic { center: cy, ratio },
            )
        }
        _ => return None,
    };
    let (u_min, u_max) = x_axis.extrema(width - 1);
    let (v_min, v_max) = y_axis.extrema(height - 1);
    let span = (u_max - u_min) + (v_max - v_min);
    if !span.is_finite() || span <= 0.0 {
        return None;
    }
    let scale = (length - 1) as f64 / span;
    let quantize = |value: f64| (value * scale).round_ties_even() as i32;
    let u_max = quantize(u_max - u_min);
    let u = xs
        .map(|x| quantize(x_axis.at(x, width - 1) - u_min))
        .collect();
    let v = ys
        .map(|y| quantize(y_axis.at(y, height - 1) - v_min).min(length as i32 - 1 - u_max))
        .collect();
    Some((u, v))
}

fn axes(
    geometry: Geometry,
    width: usize,
    height: usize,
    length: usize,
) -> Option<(Vec<i32>, Vec<i32>)> {
    quantized_axes(geometry, width, height, length, 0..width, 0..height)
}

#[derive(Clone, Copy)]
struct MeanBin {
    squares: u64,
    sum: u32,
    pop: Population,
}

impl MeanBin {
    const EMPTY: Self = Self {
        squares: 0,
        sum: 0,
        pop: Population::EMPTY,
    };

    #[inline]
    fn add(&mut self, value: u8) {
        let v = u32::from(value);
        self.sum += v;
        self.squares += u64::from(v * v);
        self.pop.add(value);
    }

    fn model(self) -> Option<i32> {
        match (self.pop.lo, self.pop.hi) {
            (0, 255) => None,
            (0, _) => Some(0),
            (_, 255) => Some(255),
            _ => {
                let quotient = self.sum / self.pop.total;
                let remainder = self.sum % self.pop.total;
                Some(
                    (quotient
                        + u32::from(
                            remainder * 2 > self.pop.total
                                || remainder * 2 == self.pop.total && !quotient.is_multiple_of(2),
                        )) as i32,
                )
            }
        }
    }

    fn squared_error(self, model: i32) -> u64 {
        let m = model as u64;
        self.squares + m * m * u64::from(self.pop.total) - 2 * m * u64::from(self.sum)
    }
}

fn mean_bins(bins: &mut [MeanBin], values: &[u8], u: &[i32], v: &[i32]) {
    for (row, &vy) in values.chunks_exact(u.len()).zip(v) {
        let bins = &mut bins[vy as usize..];
        for (&value, &ux) in row.iter().zip(u) {
            bins[ux as usize].add(value);
        }
    }
}

fn sample_error(
    geometry: Geometry,
    samples: &Samples,
    width: usize,
    height: usize,
    bins: &mut [MeanBin],
) -> f64 {
    let Some((u, v)) = samples.axes(geometry, width, height, SEARCH_PROFILE) else {
        return f64::INFINITY;
    };
    bins.fill(MeanBin::EMPTY);
    mean_bins(bins, &samples.values, &u, &v);
    let mut error = 0u64;
    for &bin in bins.iter().filter(|bin| bin.pop.total != 0) {
        let Some(model) = bin.model() else {
            return f64::INFINITY;
        };
        error += bin.squared_error(model);
    }
    error as f64 / samples.values.len() as f64
}

fn complete_profile(bins: &[MeanBin]) -> Option<(Vec<i32>, u64, u8)> {
    let mut profile = vec![0; bins.len()];
    let mut previous = None;
    let (mut error, mut maximum) = (0, 0);
    for (i, &bin) in bins
        .iter()
        .enumerate()
        .filter(|(_, bin)| bin.pop.total != 0)
    {
        let value = bin.model()?;
        let farthest = (value - i32::from(bin.pop.lo)).max(i32::from(bin.pop.hi) - value);
        if farthest > MAX_ERROR {
            return None;
        }
        maximum = maximum.max(farthest);
        error += bin.squared_error(value);
        profile[i] = value;
        if let Some((start, from)) = previous {
            for j in start + 1..i {
                profile[j] = ((from as f64
                    + (value - from) as f64 * (j - start) as f64 / (i - start) as f64)
                    .round_ties_even() as i32)
                    .clamp(0, 255);
            }
        } else {
            profile[..i].fill(value);
        }
        previous = Some((i, value));
    }
    let (last, value) = previous?;
    profile[last + 1..].fill(value);
    Some((profile, error, maximum as u8))
}

fn fit_complete(
    geometry: Geometry,
    data: &[u8],
    width: usize,
    height: usize,
    length: usize,
) -> Option<FittedProfile> {
    let (u, v) = axes(geometry, width, height, length)?;
    let mut bins = vec![MeanBin::EMPTY; length];
    mean_bins(&mut bins, data, &u, &v);
    let (p, error, maximum) = complete_profile(&bins)?;
    // 0.6 squared, expressed exactly as an integer bound.
    if error * 25 > data.len() as u64 * 9 {
        return None;
    }
    let mut reconstructed = Vec::with_capacity(data.len());
    for &vy in &v {
        let row = &p[vy as usize..];
        reconstructed.extend(u.iter().map(|&ux| row[ux as usize] as u8));
    }
    Some(FittedProfile {
        model: NestedProfile { profile: p, u, v },
        reconstructed: AlphaPlane::U8(reconstructed),
        squared_error: error,
        max_error: maximum,
    })
}

fn refine(
    mut geometry: Geometry,
    mut error: f64,
    mut score: impl FnMut(Geometry) -> f64,
) -> (Geometry, f64) {
    match geometry {
        Geometry::Linear(_) => {
            for step in [0.02, 0.005, 0.001, 0.0002, 0.00004] {
                let Geometry::Linear(angle) = geometry else {
                    unreachable!()
                };
                for sign in [-1.0, 1.0] {
                    let candidate = Geometry::Linear(angle + sign * step);
                    let current = score(candidate);
                    if current < error {
                        geometry = candidate;
                        error = current;
                    }
                }
            }
        }
        Geometry::Quadratic { .. } => {
            for step in [0.01, 0.002, 0.0004, 0.00008] {
                for axis in 0..3 {
                    let Geometry::Quadratic { cx, cy, ratio } = geometry else {
                        unreachable!()
                    };
                    for sign in [-1.0, 1.0] {
                        let candidate = Geometry::Quadratic {
                            cx: cx + if axis == 0 { sign * step } else { 0.0 },
                            cy: cy + if axis == 1 { sign * step } else { 0.0 },
                            ratio: ratio
                                * if axis == 2 {
                                    (sign * step * 4.0).exp()
                                } else {
                                    1.0
                                },
                        };
                        let current = score(candidate);
                        if current < error {
                            geometry = candidate;
                            error = current;
                        }
                    }
                }
            }
        }
    }
    (geometry, error)
}

fn gradient_angle(data: &[u8], width: usize, height: usize, samples: &Samples) -> Option<f64> {
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for &y in &samples.y {
        for &x in &samples.x {
            if x < 2 || y < 2 || x + 2 >= width || y + 2 >= height {
                continue;
            }
            let gx =
                f64::from(i32::from(data[y * width + x + 2]) - i32::from(data[y * width + x - 2]));
            let gy = f64::from(
                i32::from(data[(y + 2) * width + x]) - i32::from(data[(y - 2) * width + x]),
            );
            xx += gx * gx;
            xy += gx * gy;
            yy += gy * gy;
        }
    }
    (xx + yy > 0.0).then(|| {
        // Estimate orientation in sample coordinates, where quantization noise
        // has equal scale on both axes, then convert to normalized coordinates.
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        (angle.sin() * (height - 1) as f64).atan2(angle.cos() * (width - 1) as f64)
    })
}

fn moment_geometry(samples: &Samples, width: usize, height: usize) -> Option<Geometry> {
    let (mut weight, mut sx, mut sy) = (0.0, 0.0, 0.0);
    let lo = *samples.values.iter().min()?;
    let hi = *samples.values.iter().max()?;
    let invert = samples.values.iter().filter(|&&v| v == hi).count()
        > samples.values.iter().filter(|&&v| v == lo).count();
    for (row, &y) in samples.y.iter().enumerate() {
        for (column, &x) in samples.x.iter().enumerate() {
            let a = samples.values[row * samples.x.len() + column];
            let w = f64::from(if invert { hi - a } else { a - lo });
            weight += w;
            sx += w * x as f64 / (width - 1) as f64;
            sy += w * y as f64 / (height - 1) as f64;
        }
    }
    if weight <= 0.0 {
        return None;
    }
    let (cx, cy) = (sx / weight, sy / weight);
    let (mut vx, mut vy) = (0.0, 0.0);
    for (row, &y) in samples.y.iter().enumerate() {
        for (column, &x) in samples.x.iter().enumerate() {
            let a = samples.values[row * samples.x.len() + column];
            let w = f64::from(if invert { hi - a } else { a - lo });
            vx += w * (x as f64 / (width - 1) as f64 - cx).powi(2);
            vy += w * (y as f64 / (height - 1) as f64 - cy).powi(2);
        }
    }
    (vx > 0.0 && vy > 0.0).then_some(Geometry::Quadratic {
        cx,
        cy,
        ratio: vx / vy,
    })
}

fn log_geometry(samples: &Samples, width: usize, height: usize, invert: bool) -> Option<Geometry> {
    let mut normal = [[0.0; 6]; 5];
    for (row, &y) in samples.y.iter().enumerate() {
        for (column, &x) in samples.x.iter().enumerate() {
            let value = samples.values[row * samples.x.len() + column];
            let value = if invert { 255 - value } else { value };
            if !(8..=254).contains(&value) {
                continue;
            }
            let (x, y) = (
                x as f64 / (width - 1) as f64,
                y as f64 / (height - 1) as f64,
            );
            let basis = [1.0, x, y, x * x, y * y];
            let target = f64::from(value).ln();
            let weight = f64::from(value);
            for i in 0..5 {
                for j in 0..5 {
                    normal[i][j] += weight * basis[i] * basis[j];
                }
                normal[i][5] += weight * basis[i] * target;
            }
        }
    }
    for i in 0..5 {
        let pivot = (i..5).max_by(|&a, &b| normal[a][i].abs().total_cmp(&normal[b][i].abs()))?;
        normal.swap(i, pivot);
        let scale = normal[i][i];
        if !scale.is_finite() || scale.abs() < 1e-12 {
            return None;
        }
        for j in i..6 {
            normal[i][j] /= scale;
        }
        for k in 0..5 {
            if k != i {
                let factor = normal[k][i];
                for j in i..6 {
                    normal[k][j] -= factor * normal[i][j];
                }
            }
        }
    }
    let a = normal[3][5];
    let b = normal[4][5];
    if a >= -1e-9 || b >= -1e-9 {
        return None;
    }
    Some(Geometry::Quadratic {
        cx: -normal[1][5] / (2.0 * a),
        cy: -normal[2][5] / (2.0 * b),
        ratio: b / a,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function(width: usize, height: usize, shape: usize) -> AlphaPlane {
        let values = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let x = x as f64 / (width - 1) as f64;
                    let y = y as f64 / (height - 1) as f64;
                    let s = if shape == 0 {
                        (x + 0.63 * y) / 1.63
                    } else {
                        ((x - 0.51) / 0.49).powi(2)
                            + if shape == 1 { 0.55 } else { 1.0 } * ((y - 0.47) / 0.53).powi(2)
                    };
                    let value = if shape == 0 {
                        255.0 * (0.5 - 0.5 * (s * std::f64::consts::TAU).cos())
                    } else if shape == 1 {
                        255.0 * (-3.367 * s).exp()
                    } else {
                        255.0 * (-(((s / 2.0835 * 2.1).sqrt() - 0.55) / 0.13).powi(2)).exp()
                    };
                    value.round_ties_even().clamp(0.0, 255.0) as u8
                })
            })
            .collect();
        AlphaPlane::U8(values)
    }

    #[test]
    fn coherent_functions_fit_without_supplied_parameters() {
        for (shape, invert) in [(0, false), (1, false), (2, false), (1, true)] {
            let mut plane = function(513, 257, shape);
            if invert && let AlphaPlane::U8(values) = &mut plane {
                for value in values {
                    *value = 255 - *value;
                }
            }
            let candidates = fit(&plane, 513, 257);
            assert!(!candidates.is_empty(), "shape {shape}, invert {invert}");
            for candidate in candidates {
                let AlphaPlane::U8(source) = &plane else {
                    unreachable!()
                };
                let AlphaPlane::U8(decoded) = &candidate.reconstructed else {
                    unreachable!()
                };
                assert!(candidate.squared_error * 25 <= source.len() as u64 * 9);
                assert!(candidate.max_error <= 2);
                for (i, (&a, &b)) in source.iter().zip(decoded).enumerate() {
                    if a == 0 || a == 255 {
                        assert_eq!(a, b, "endpoint {i}");
                    }
                    assert_eq!(
                        i32::from(b),
                        candidate.model.profile
                            [(candidate.model.u[i % 513] + candidate.model.v[i / 513]) as usize]
                    );
                }
            }
        }
    }

    /// A radial glow around a hard-edged disc, with an opaque bar across it:
    /// no whole-plane function is within two code values of this plane.
    fn occluded_glow(width: usize, height: usize) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let (dx, dy) = (
                        x as f64 - 0.52 * width as f64,
                        y as f64 - 0.47 * height as f64,
                    );
                    let r = (dx * dx + 1.1 * dy * dy).sqrt() / width as f64;
                    if y > height * 3 / 4 && y < height * 3 / 4 + 12 {
                        255
                    } else if r < 0.2 {
                        0
                    } else {
                        (255.0 * (-((r - 0.2) / 0.11).powi(2)).exp()).round_ties_even() as u8
                    }
                })
            })
            .collect()
    }

    #[test]
    fn residual_fit_explains_an_occluded_glow_exactly() {
        let (width, height) = (301, 257);
        let data = occluded_glow(width, height);
        let plane = AlphaPlane::U8(data.clone());
        assert!(fit(&plane, width, height).is_empty());
        let pool = crate::thread_pool::ThreadPool::new(2);
        let fitted = fit_residual(&pool, &plane, width, height).expect("residual fit");
        let model = &fitted.base.model;
        let (AlphaPlane::U8(function), AlphaPlane::U8(residual)) =
            (&fitted.base.reconstructed, &fitted.residual)
        else {
            unreachable!()
        };
        let mut unexplained = 0;
        for (i, &value) in data.iter().enumerate() {
            let s = (model.u[i % width] + model.v[i / width]) as usize;
            // The base frame's biased profile plus the residual is the source.
            assert_eq!(model.profile[s] + i32::from(residual[i]), i32::from(value));
            assert_eq!(model.profile[s] + RESIDUAL_BIAS, i32::from(function[i]));
            unexplained += usize::from(residual[i] != RESIDUAL_BIAS as u8);
        }
        assert!(unexplained * 3 < data.len(), "{unexplained}");
        let Some(AlphaPlane::U8(composed)) = fitted.compose(&fitted.residual) else {
            panic!("composition failed")
        };
        assert_eq!(composed, data);
    }

    #[test]
    fn periodic_fit_finds_the_tile_and_leaves_defects_to_the_residual() {
        let (width, height, p, q) = (203usize, 157usize, 20usize, 12usize);
        let mut data: Vec<u8> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width % p, i / width % q);
                ((x * 37 + y * 91 + x * y * 5) % 256) as u8
            })
            .collect();
        // Defects within the residual's range of the tile; a farther one
        // would move its whole cell (see the clamp in `fit_periodic`).
        for i in (0..data.len()).step_by(997) {
            data[i] ^= 0x10;
        }
        let plane = AlphaPlane::U8(data.clone());
        let fitted = fit_periodic(&plane, width, height).expect("periodic fit");
        assert!(fitted.periodic);
        assert_eq!(fitted.base.model.profile.len(), p * q);
        let AlphaPlane::U8(residual) = &fitted.residual else {
            unreachable!()
        };
        let defects = residual
            .iter()
            .filter(|&&r| r != RESIDUAL_BIAS as u8)
            .count();
        assert_eq!(defects, data.len().div_ceil(997));
        let Some(AlphaPlane::U8(composed)) = fitted.compose(&fitted.residual) else {
            panic!("composition failed")
        };
        assert_eq!(composed, data);
        // A glow has texture but no period.
        let glow = AlphaPlane::U8(occluded_glow(301, 257));
        assert!(fit_periodic(&glow, 301, 257).is_none());
        assert!(fit_periodic(&AlphaPlane::U8(vec![7; 64 * 64]), 64, 64).is_none());
    }

    #[test]
    fn residual_fit_declines_hard_masks_and_noise() {
        let (width, height) = (257, 129);
        let mask: Vec<u8> = (0..width * height)
            .map(|i| if (i % width) / 32 % 2 == 0 { 255 } else { 0 })
            .collect();
        let pool = crate::thread_pool::ThreadPool::new(1);
        assert!(fit_residual(&pool, &AlphaPlane::U8(mask), width, height).is_none());
        let mut state = 1u32;
        let noise = (0..width * height)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        assert!(fit_residual(&pool, &AlphaPlane::U8(noise), width, height).is_none());
        assert!(fit_residual(&pool, &AlphaPlane::F32(vec![0; 32 * 32]), 32, 32).is_none());
    }

    #[test]
    fn unsupported_or_unstructured_fields_are_skipped() {
        assert!(fit(&AlphaPlane::U8(vec![42; 32 * 32]), 32, 32).is_empty());
        assert!(fit(&AlphaPlane::U8(vec![0; 32 * 32 - 1]), 32, 32).is_empty());
        assert!(fit(&AlphaPlane::U8(vec![]), usize::MAX, 2).is_empty());
        assert!(fit(&AlphaPlane::F32(vec![0; 32 * 32]), 32, 32).is_empty());
        assert!(
            fit(
                &AlphaPlane::U16 {
                    data: vec![0; 32 * 32],
                    bits: 16
                },
                32,
                32
            )
            .is_empty()
        );
        let mut state = 1u32;
        let noise = (0..257 * 129)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        assert!(fit(&AlphaPlane::U8(noise), 257, 129).is_empty());
    }

    #[test]
    fn profile_anchors_and_negative_direction_are_preserved() {
        let mut bins = [MeanBin::EMPTY; 3];
        bins[0].add(0);
        bins[2].add(255);
        assert_eq!(complete_profile(&bins), Some((vec![0, 128, 255], 0, 0)));
        bins[0].add(255);
        assert!(complete_profile(&bins).is_none());
        let (u, v) = axes(Geometry::Linear(-0.7), 65, 33, 1024).unwrap();
        assert_eq!(*u.iter().min().unwrap(), 0);
        assert_eq!(*v.iter().min().unwrap(), 0);
        assert!(u.iter().max().unwrap() + v.iter().max().unwrap() < 1024);
        assert!(v[0] > v[32]);
    }
}
