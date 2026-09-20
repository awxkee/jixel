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

use super::filter::FilterPlan;
use super::{Point, fast_hypot};
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

static SCALES: [f32; 4] = [0.7, 1.0, 1.5, 2.2];
const HYSTERESIS_HIGH: f32 = 0.02;
const HYSTERESIS_LOW: f32 = 0.008;
const MIN_CHAIN_LEN: usize = 20;
/// Line-ness: curvature across the ridge must dominate curvature along it.
pub(crate) const ALONG_PENALTY: f32 = 1.5;
pub(crate) const MAX_SUBPIXEL_OFFSET: f32 = 0.75;
const TRACE_MIN_COS: f32 = 0.35;
/// Chains are trimmed to this distance from the frame: a line leaving the image
/// keeps its interior part, a line hugging the border (scan frames) vanishes.
const FRAME_MARGIN: f32 = 4.0;

/// A traced ridge: sub-pixel center line and its strength-weighted ridge scale.
pub(super) struct Chain {
    pub(super) points: Vec<Point<f32>>,
    pub(super) scale: f32,
}

/// Gaussian kernels of derivative order 0, 1 and 2 (radius `4 * sigma`).
pub(super) fn gaussian_kernels(sigma: f32) -> [Vec<f32>; 3] {
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut k0: Vec<f32> = (-radius..=radius)
        .map(|j| (-0.5 * (j as f32 / sigma).powi(2)).exp())
        .collect();
    let sum: f32 = k0.iter().sum();
    k0.iter_mut().for_each(|v| *v /= sum);
    let s2 = sigma * sigma;
    let k1 = k0
        .iter()
        .zip(-radius..=radius)
        .map(|(&g, j)| j as f32 / s2 * g)
        .collect();
    let k2 = k0
        .iter()
        .zip(-radius..=radius)
        .map(|(&g, j)| ((j * j) as f32 / (s2 * s2) - 1.0 / s2) * g)
        .collect();
    [k0, k1, k2]
}

struct RidgeMap {
    strength: Vec<f32>,
    nx: Vec<f32>,
    ny: Vec<f32>,
    scale: Vec<f32>,
    polarity: Vec<i8>,
    offset: Vec<f32>,
}

/// Disjoint output rows for the scale that wins at each pixel.
pub(crate) struct RidgeRow<'a> {
    pub(crate) strength: &'a mut [f32],
    pub(crate) nx: &'a mut [f32],
    pub(crate) ny: &'a mut [f32],
    pub(crate) scale: &'a mut [f32],
    pub(crate) polarity: &'a mut [i8],
    pub(crate) offset: &'a mut [f32],
}

impl RidgeRow<'_> {
    /// Check every slice before architecture kernels use full-vector loads.
    pub(crate) fn validate(&self, derivatives: [&[f32]; 5]) -> usize {
        let n = self.strength.len();
        assert_eq!(self.nx.len(), n);
        assert_eq!(self.ny.len(), n);
        assert_eq!(self.scale.len(), n);
        assert_eq!(self.polarity.len(), n);
        assert_eq!(self.offset.len(), n);
        for row in derivatives {
            assert_eq!(row.len(), n);
        }
        n
    }
}

/// Derivative rows are hxx, hyy, hxy, gx and gy, in that order.
pub(crate) type RidgeRowFn = fn(f32, [&[f32]; 5], RidgeRow<'_>);

pub(crate) fn select_ridge_row_fn() -> RidgeRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |s, derivatives, output| unsafe {
        crate::neon::spline_ridge_row_neon(s, derivatives, output)
    };
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |s, derivatives, output| unsafe {
            crate::avx::spline_ridge_row_avx2(s, derivatives, output)
        };
    }
    #[allow(unreachable_code)]
    ridge_row_scalar
}

pub(crate) fn ridge_row_scalar(s: f32, derivatives: [&[f32]; 5], output: RidgeRow<'_>) {
    output.validate(derivatives);
    let [hxx, hyy, hxy, gx, gy] = derivatives;
    let RidgeRow {
        strength: strengths,
        nx,
        ny,
        scale: scales,
        polarity: polarities,
        offset: offsets,
    } = output;
    let s2 = s * s;
    let hessian = hxx.iter().zip(hyy.iter()).zip(hxy.iter());
    let gradients = gx.iter().zip(gy.iter());
    let normals = nx.iter_mut().zip(ny);
    let attributes = scales.iter_mut().zip(polarities).zip(offsets);
    let outputs = strengths.iter_mut().zip(normals).zip(attributes);
    for (derivatives, output) in hessian.zip(gradients).zip(outputs) {
        let (((&hxx, &hyy), &hxy), (&gx, &gy)) = derivatives;
        let ((best, (nx, ny)), ((scale, polarity), offset)) = output;
        let tr = 0.5 * (hxx + hyy);
        let df = ((0.5 * (hxx - hyy)).powi(2) + hxy * hxy).sqrt();
        let (la, lb) = (tr + df, tr - df);
        let (big, small) = if la.abs() > lb.abs() {
            (la, lb)
        } else {
            (lb, la)
        };
        let strength = s2 * (big.abs() - ALONG_PENALTY * small.abs()).max(0.0);
        if strength <= *best {
            continue;
        }
        // a line center has ~zero gradient, a step edge does not
        if s * fast_hypot(gx, gy) > s2 * big.abs() {
            continue;
        }
        let (mut vx, mut vy) = (hxy, big - hxx);
        let norm = fast_hypot(vx, vy);
        if norm < 1e-12 {
            (vx, vy) = (1.0, 0.0);
        } else {
            (vx, vy) = (vx / norm, vy / norm);
        }
        let t = if big.abs() < 1e-12 {
            0.0
        } else {
            (-(gx * vx + gy * vy) / big).clamp(-MAX_SUBPIXEL_OFFSET, MAX_SUBPIXEL_OFFSET)
        };
        *best = strength;
        *nx = vx;
        *ny = vy;
        *scale = s;
        *polarity = if big > 0.0 { -1 } else { 1 };
        *offset = t;
    }
}

fn ridge_map(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    luma: &[f32],
    w: usize,
    h: usize,
) -> RidgeMap {
    let n = w * h;
    let mut map = RidgeMap {
        strength: vec![0.0; n],
        nx: vec![0.0; n],
        ny: vec![0.0; n],
        scale: vec![0.0; n],
        polarity: vec![0; n],
        offset: vec![0.0; n],
    };
    let mut r0 = vec![0.0; n];
    let mut r1 = vec![0.0; n];
    let mut r2 = vec![0.0; n];
    for s in SCALES {
        let [k0, k1, k2] = gaussian_kernels(s);
        FilterPlan::new(w, &k0).horizontal(ctx, scratch, luma, &mut r0);
        FilterPlan::new(w, &k1).horizontal(ctx, scratch, luma, &mut r1);
        FilterPlan::new(w, &k2).horizontal(ctx, scratch, luma, &mut r2);
        let v0 = FilterPlan::new(h, &k0);
        let v1 = FilterPlan::new(h, &k1);
        let v2 = FilterPlan::new(h, &k2);
        let bands = (ctx.thread_pool.num_threads().max(1) * 4).min(h).max(1);
        let band_len = h.div_ceil(bands) * w;
        struct Band<'a> {
            start: usize,
            strength: &'a mut [f32],
            nx: &'a mut [f32],
            ny: &'a mut [f32],
            scale: &'a mut [f32],
            polarity: &'a mut [i8],
            offset: &'a mut [f32],
        }
        let mut items: Vec<Band> = map
            .strength
            .chunks_mut(band_len)
            .zip(map.nx.chunks_mut(band_len))
            .zip(map.ny.chunks_mut(band_len))
            .zip(map.scale.chunks_mut(band_len))
            .zip(map.polarity.chunks_mut(band_len))
            .zip(map.offset.chunks_mut(band_len))
            .enumerate()
            .map(
                |(b, (((((strength, nx), ny), scale), polarity), offset))| Band {
                    start: b * band_len,
                    strength,
                    nx,
                    ny,
                    scale,
                    polarity,
                    offset,
                },
            )
            .collect();

        ctx.thread_pool
            .steal_for_each_mut(scratch, &mut items, |_, band, _| {
                // Consume derivative rows immediately instead of materializing
                // five full-image planes. Horizontal passes remain first, so
                // floating-point accumulation matches the original detector.
                let mut derivatives = vec![0.0; 5 * w];
                let (hxx, rest) = derivatives.split_at_mut(w);
                let (hyy, rest) = rest.split_at_mut(w);
                let (hxy, rest) = rest.split_at_mut(w);
                let (gx, gy) = rest.split_at_mut(w);
                let rows = band
                    .strength
                    .chunks_exact_mut(w)
                    .zip(band.nx.chunks_exact_mut(w))
                    .zip(band.ny.chunks_exact_mut(w))
                    .zip(band.scale.chunks_exact_mut(w))
                    .zip(band.polarity.chunks_exact_mut(w))
                    .zip(band.offset.chunks_exact_mut(w));
                for (row, outputs) in rows.enumerate() {
                    let y = band.start / w + row;
                    v0.vertical_row(&r2, w, y, hxx);
                    v2.vertical_row(&r0, w, y, hyy);
                    v1.vertical_row(&r1, w, y, hxy);
                    v0.vertical_row(&r1, w, y, gx);
                    v1.vertical_row(&r0, w, y, gy);
                    let (((((strengths, nx), ny), scales), polarities), offsets) = outputs;
                    (ctx.spline_ridge_row)(
                        s,
                        [hxx, hyy, hxy, gx, gy],
                        RidgeRow {
                            strength: strengths,
                            nx,
                            ny,
                            scale: scales,
                            polarity: polarities,
                            offset: offsets,
                        },
                    );
                }
            });
    }
    map
}

#[inline]
fn bilinear(plane: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = plane[y0 * w + x0] * (1.0 - fx) + plane[y0 * w + x1] * fx;
    let bottom = plane[y1 * w + x0] * (1.0 - fx) + plane[y1 * w + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Ridge peaks above the weak threshold. Tracing starts only at strong peaks,
/// so each retained chain contains its own strong seed without a flood fill.
fn ridge_mask(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    map: &mut RidgeMap,
    w: usize,
    h: usize,
) -> (Vec<i8>, Vec<u64>) {
    // Ridge evaluation is complete: reuse its polarity allocation as the mask.
    let mut weak = std::mem::take(&mut map.polarity);
    debug_assert_eq!(weak.len(), w * h);
    let band_rows = h.div_ceil(ctx.thread_pool.num_threads() * 4).max(1);
    let mut bands: Vec<_> = weak
        .chunks_mut(band_rows * w)
        .map(|mask| (mask, Vec::new()))
        .collect();
    ctx.thread_pool
        .steal_for_each_mut(scratch, &mut bands, |band, (weak, seeds), _| {
            let offset = band * band_rows * w;
            let rows = weak
                .chunks_exact_mut(w)
                .zip(map.strength[offset..].chunks_exact(w))
                .zip(
                    map.nx[offset..]
                        .chunks_exact(w)
                        .zip(map.ny[offset..].chunks_exact(w)),
                );
            for (row, ((weak, strength), (nx, ny))) in rows.enumerate() {
                let y = band * band_rows + row;
                let pixels = weak.iter_mut().zip(strength).zip(nx.iter().zip(ny));
                for (x, ((weak, &s), (&nx, &ny))) in pixels.enumerate() {
                    if s <= HYSTERESIS_LOW {
                        *weak = 0;
                        continue;
                    }
                    let (fx, fy) = (x as f32, y as f32);
                    let a = bilinear(&map.strength, w, h, fx + nx, fy + ny);
                    let b = bilinear(&map.strength, w, h, fx - nx, fy - ny);
                    if s >= a && s >= b {
                        if s > HYSTERESIS_HIGH {
                            seeds.push(seed_key((y * w + x) as u32, s));
                        }
                    } else {
                        *weak = 0;
                    }
                }
            }
        });
    // Concatenate in raster order, independent of worker completion order.
    // This avoids a second serial scan of the full mask and strength plane.
    let mut seeds = Vec::with_capacity(bands.iter().map(|(_, seeds)| seeds.len()).sum());
    for (_, mut band_seeds) in bands {
        seeds.append(&mut band_seeds);
    }
    (weak, seeds)
}

/// Strong seeds have positive, non-NaN strengths, whose float bits sort in
/// numerical order. Pack the descending key beside the pixel index so radix
/// passes never gather strengths from the full image in permuted order.
fn seed_key(index: u32, strength: f32) -> u64 {
    ((!strength.to_bits() as u64) << 32) | index as u64
}

fn sort_seeds(order: &mut Vec<u64>) {
    if order.len() < 2048 {
        // Ignore the pixel index: equal-strength seeds retain their input order.
        order.sort_by_key(|entry| entry >> 32);
        return;
    }
    let mut tmp = vec![0; order.len()];
    for shift in [32, 40, 48, 56] {
        let key = |entry: u64| ((entry >> shift) & 255) as usize;
        let mut offsets = [0usize; 256];
        for &entry in order.iter() {
            offsets[key(entry)] += 1;
        }
        let mut sum = 0;
        for offset in &mut offsets {
            let count = *offset;
            *offset = sum;
            sum += count;
        }
        for &entry in order.iter() {
            let offset = &mut offsets[key(entry)];
            tmp[*offset] = entry;
            *offset += 1;
        }
        std::mem::swap(order, &mut tmp);
    }
}

static NEIGHBORS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

#[derive(Clone, Copy)]
struct Neighbor {
    dx: isize,
    dy: isize,
    index: isize,
    norm: f32,
    direction: (f32, f32),
}

/// Zero marks an unavailable pixel; otherwise the byte stores its polarity.
/// Suppression still removes both polarities, as in the original walk.
struct Tracer<'a> {
    avail: &'a mut [i8],
    w: usize,
    h: usize,
    neighbors: [Neighbor; 8],
}

impl<'a> Tracer<'a> {
    fn new(avail: &'a mut [i8], w: usize, h: usize) -> Self {
        Self {
            avail,
            w,
            h,
            neighbors: NEIGHBORS.map(|(dx, dy)| {
                let norm = ((dx * dx + dy * dy) as f32).sqrt();
                Neighbor {
                    dx,
                    dy,
                    index: dy * w as isize + dx,
                    norm,
                    direction: (dx as f32 / norm, dy as f32 / norm),
                }
            }),
        }
    }

    /// Greedy tangent-following walk, retaining neighbor order and score division.
    fn walk(
        &mut self,
        start: (usize, usize),
        mut tangent: (f32, f32),
        polarity: i8,
        out: &mut Vec<(usize, usize)>,
    ) {
        out.clear();
        let (mut x, mut y) = (start.0 as isize, start.1 as isize);
        loop {
            let center = y * self.w as isize + x;
            let interior = x > 0 && y > 0 && x + 1 < self.w as isize && y + 1 < self.h as isize;
            let in_bounds = |n: &Neighbor| {
                interior
                    || (x + n.dx >= 0
                        && y + n.dy >= 0
                        && x + n.dx < self.w as isize
                        && y + n.dy < self.h as isize)
            };
            let mut best: Option<Neighbor> = None;
            let mut best_score = TRACE_MIN_COS;
            for &n in &self.neighbors {
                if !in_bounds(&n) || self.avail[(center + n.index) as usize] != polarity {
                    continue;
                }
                let score = (n.dx as f32 * tangent.0 + n.dy as f32 * tangent.1) / n.norm;
                if score > best_score {
                    best_score = score;
                    best = Some(n);
                }
            }
            let Some(next) = best else { return };
            // Suppress perpendicular neighbors of the pixel we leave.
            for n in &self.neighbors {
                if in_bounds(n)
                    && n.index != next.index
                    && (n.dx as f32 * tangent.0 + n.dy as f32 * tangent.1).abs() < 0.5
                {
                    self.avail[(center + n.index) as usize] = 0;
                }
            }
            self.avail[(center + next.index) as usize] = 0;
            (x, y) = (x + next.dx, y + next.dy);
            out.push((x as usize, y as usize));
            tangent = (
                0.6 * tangent.0 + 0.4 * next.direction.0,
                0.6 * tangent.1 + 0.4 * next.direction.1,
            );
            let norm = fast_hypot(tangent.0, tangent.1);
            tangent = (tangent.0 / norm, tangent.1 / norm);
        }
    }
}

pub(super) fn detect_chains(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
) -> Vec<Chain> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let mut map = ridge_map(ctx, scratch, xyb.plane_data(1), w, h);
    let (mut avail, mut order) = ridge_mask(ctx, scratch, &mut map, w, h);
    sort_seeds(&mut order);

    let mut tracer = Tracer::new(&mut avail, w, h);
    let (mut forward, mut backward) = (Vec::new(), Vec::new());
    let mut chains = Vec::new();
    for entry in order {
        let i = entry as u32 as usize;
        let polarity = tracer.avail[i];
        if polarity == 0 {
            continue;
        }
        tracer.avail[i] = 0;
        let start = (i % w, i / w);
        let tangent = (-map.ny[i], map.nx[i]);
        tracer.walk(start, tangent, polarity, &mut forward);
        tracer.walk(start, (-tangent.0, -tangent.1), polarity, &mut backward);
        if backward.len() + 1 + forward.len() < MIN_CHAIN_LEN {
            continue;
        }
        let pixels = backward
            .iter()
            .rev()
            .chain(std::iter::once(&start))
            .chain(forward.iter());
        let mut points = Vec::with_capacity(backward.len() + 1 + forward.len());
        let (mut scale_sum, mut weight_sum) = (0f32, 0f32);
        for &(x, y) in pixels {
            let j = y * w + x;
            points.push(Point::new(
                x as f32 + map.offset[j] * map.nx[j],
                y as f32 + map.offset[j] * map.ny[j],
            ));
            scale_sum += map.scale[j] * map.strength[j];
            weight_sum += map.strength[j];
        }
        let interior = |p: &Point<f32>| {
            p.x >= FRAME_MARGIN
                && p.y >= FRAME_MARGIN
                && p.x <= w as f32 - 1.0 - FRAME_MARGIN
                && p.y <= h as f32 - 1.0 - FRAME_MARGIN
        };
        let (mut best, mut run_start) = (0..0, 0usize);
        for k in 0..=points.len() {
            if k == points.len() || !interior(&points[k]) {
                if k - run_start > best.len() {
                    best = run_start..k;
                }
                run_start = k + 1;
            }
        }
        if best.len() < MIN_CHAIN_LEN {
            continue;
        }
        chains.push(Chain {
            points: points[best].to_vec(),
            scale: scale_sum / weight_sum.max(1e-12),
        });
    }
    chains
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Greedy tangent-following walk over the available ridge pixels.
    fn walk_reference(
        map: &RidgeMap,
        avail: &mut [bool],
        w: usize,
        h: usize,
        start: (usize, usize),
        mut tangent: (f32, f32),
        polarity: i8,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let (mut x, mut y) = (start.0 as isize, start.1 as isize);
        loop {
            let mut best: Option<(isize, isize, f32, f32)> = None;
            let mut best_score = TRACE_MIN_COS;
            for (dx, dy) in NEIGHBORS {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if !avail[j] || map.polarity[j] != polarity {
                    continue;
                }
                let norm = ((dx * dx + dy * dy) as f32).sqrt();
                let score = (dx as f32 * tangent.0 + dy as f32 * tangent.1) / norm;
                if score > best_score {
                    best_score = score;
                    best = Some((xx, yy, dx as f32 / norm, dy as f32 / norm));
                }
            }
            let Some((nx_, ny_, ddx, ddy)) = best else {
                return out;
            };
            // suppress the perpendicular neighbors of the pixel we leave
            for (dx, dy) in NEIGHBORS {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0
                    || yy < 0
                    || xx >= w as isize
                    || yy >= h as isize
                    || (xx, yy) == (nx_, ny_)
                {
                    continue;
                }
                if (dx as f32 * tangent.0 + dy as f32 * tangent.1).abs() < 0.5 {
                    avail[yy as usize * w + xx as usize] = false;
                }
            }
            avail[ny_ as usize * w + nx_ as usize] = false;
            out.push((nx_ as usize, ny_ as usize));
            tangent = (0.6 * tangent.0 + 0.4 * ddx, 0.6 * tangent.1 + 0.4 * ddy);
            let norm = fast_hypot(tangent.0, tangent.1);
            tangent = (tangent.0 / norm, tangent.1 / norm);
            (x, y) = (nx_, ny_);
        }
    }

    #[test]
    fn packed_tracing_matches_reference_paths_and_suppression() {
        let directions = [
            (1.0, 0.0),
            (-0.0, -1.0),
            (0.6, 0.8),
            (-0.8, 0.6),
            (
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ),
        ];
        for (w, h) in [(1, 1), (1, 31), (31, 1), (2, 2), (17, 23), (64, 65)] {
            for pattern in 0..4 {
                let n = w * h;
                let map = RidgeMap {
                    strength: vec![1.0; n],
                    nx: vec![0.0; n],
                    ny: vec![1.0; n],
                    scale: vec![1.0; n],
                    offset: vec![0.0; n],
                    polarity: (0..n)
                        .map(|i| {
                            if (i * 19 + i / w + pattern) % 7 < 3 {
                                -1
                            } else {
                                1
                            }
                        })
                        .collect(),
                };
                let mut expected: Vec<bool> = (0..n)
                    .map(|i| pattern == 0 || (i * 37 + i / w) % 5 != pattern)
                    .collect();
                let mut actual: Vec<i8> = expected
                    .iter()
                    .zip(&map.polarity)
                    .map(|(&v, &p)| if v { p } else { 0 })
                    .collect();
                let mut tracer = Tracer::new(&mut actual, w, h);
                let mut out = Vec::new();
                for i in (0..n).rev() {
                    if !expected[i] {
                        continue;
                    }
                    expected[i] = false;
                    tracer.avail[i] = 0;
                    let start = (i % w, i / w);
                    let t = directions[i % directions.len()];
                    for tangent in [t, (-t.0, -t.1)] {
                        let reference = walk_reference(
                            &map,
                            &mut expected,
                            w,
                            h,
                            start,
                            tangent,
                            map.polarity[i],
                        );
                        tracer.walk(start, tangent, map.polarity[i], &mut out);
                        assert_eq!(out, reference, "shape {w}x{h}, pattern {pattern}, seed {i}");
                        assert!(
                            tracer
                                .avail
                                .iter()
                                .zip(&expected)
                                .all(|(&a, &e)| (a != 0) == e)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ridge_kernel_matches_scalar_thresholds_tails_and_scale_ties() {
        let ctx = EncodingContext::default();
        let mut cases = vec![
            [0.0, -0.0, 0.0, -0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0], // degenerate eigenvector
            [-1.0, 0.0, 0.0, -0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0, 0.0], // equal eigenvalue magnitudes
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0], // no ridge strength
        ];
        for scale in SCALES {
            for big in [1.0, -1.0, 1e-12f32.next_down(), 1e-12, 1e-12f32.next_up()] {
                for limit in [scale * big.abs(), MAX_SUBPIXEL_OFFSET * big.abs()] {
                    for gradient in [limit.next_down(), limit, limit.next_up()] {
                        cases.push([0.0, big, 0.0, 0.0, gradient]);
                        cases.push([big, 0.0, 0.0, -gradient, 0.0]);
                    }
                }
            }
        }
        // Miri still exercises every vector alignment and tail, but uses a
        // smaller corpus because interpreting SIMD is much slower than native.
        if cfg!(miri) {
            cases.truncate(64);
        }
        // Mixed orientations and gradient strengths, with many accepted lanes
        // as well as isolated rejections inside otherwise accepted vectors.
        let mut state = 0x71b9a535u32;
        for i in 0..if cfg!(miri) { 32 } else { 4096 } {
            let magnitude = [1e-14, 1e-7, 1.0, 1e7][i % 4];
            cases.push(std::array::from_fn(|channel| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let value = (state as i32 as f32 / i32::MAX as f32) * magnitude;
                if channel >= 3 { value * 0.25 } else { value }
            }));
        }
        let inputs: [Vec<f32>; 5] = std::array::from_fn(|c| cases.iter().map(|v| v[c]).collect());
        let make_map = || RidgeMap {
            strength: vec![0.0; cases.len()],
            nx: vec![17.0; cases.len()],
            ny: vec![-19.0; cases.len()],
            scale: vec![23.0; cases.len()],
            polarity: vec![7; cases.len()],
            offset: vec![29.0; cases.len()],
        };
        fn row(map: &mut RidgeMap, range: std::ops::Range<usize>) -> RidgeRow<'_> {
            RidgeRow {
                strength: &mut map.strength[range.clone()],
                nx: &mut map.nx[range.clone()],
                ny: &mut map.ny[range.clone()],
                scale: &mut map.scale[range.clone()],
                polarity: &mut map.polarity[range.clone()],
                offset: &mut map.offset[range],
            }
        }
        // All alignments and short tails, including zero-length rows. Values
        // outside the supplied slice are sentinels and must remain untouched.
        for start in 0..8 {
            for len in (0..=25).chain((cases.len() - 15)..=(cases.len() - 8)) {
                let range = start..start + len;
                let derivatives = inputs.each_ref().map(|v| &v[range.clone()]);
                let (mut expected, mut actual) = (make_map(), make_map());
                for s in [0.7, 0.7, 1.0, 1.5, 2.2] {
                    ridge_row_scalar(s, derivatives, row(&mut expected, range.clone()));
                    (ctx.spline_ridge_row)(s, derivatives, row(&mut actual, range.clone()));
                    for (a, b) in [
                        (&actual.strength, &expected.strength),
                        (&actual.nx, &expected.nx),
                        (&actual.ny, &expected.ny),
                        (&actual.scale, &expected.scale),
                        (&actual.offset, &expected.offset),
                    ] {
                        for (i, (a, b)) in a.iter().zip(b).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "range {range:?}, scale {s}, pixel {i}, input {:?}",
                                cases[i]
                            );
                        }
                    }
                    assert_eq!(actual.polarity, expected.polarity);
                }
            }
        }
    }
    use crate::{Speed, xyb::XybMatrix};

    #[test]
    fn radix_seeds_preserve_strength_order_and_stable_ties() {
        let strengths: Vec<f32> = (0..10000)
            .map(|i| match i % 11 {
                0 => f32::INFINITY,
                1 => HYSTERESIS_HIGH.next_up(),
                2 => 0.125,
                _ => f32::from_bits(0x3c800000 + ((i * 2654435761u64) % 0x04000000) as u32),
            })
            .collect();
        for n in [0, 1, 17, 2047, 2048, 10000] {
            let mut expected: Vec<u32> = (0..n).rev().collect();
            let mut order: Vec<u64> = expected
                .iter()
                .map(|&i| seed_key(i, strengths[i as usize]))
                .collect();
            expected.sort_by(|&a, &b| strengths[b as usize].total_cmp(&strengths[a as usize]));
            sort_seeds(&mut order);
            assert_eq!(
                order.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn parallel_peak_collection_preserves_raster_order_and_thresholds() {
        for (w, h) in [(1, 1), (3, 17), (37, 31), (256, 257)] {
            let n = w * h;
            let normals = [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8), (-0.8, 0.6)];
            let mut map = RidgeMap {
                strength: (0..n)
                    .map(|i| match i % 13 {
                        0 => HYSTERESIS_LOW,
                        1 => HYSTERESIS_HIGH,
                        2 => HYSTERESIS_HIGH.next_up(),
                        _ => (i * 37 % 19) as f32 * 0.005,
                    })
                    .collect(),
                nx: (0..n).map(|i| normals[i % 4].0).collect(),
                ny: (0..n).map(|i| normals[i % 4].1).collect(),
                scale: vec![1.0; n],
                polarity: (0..n).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect(),
                offset: vec![0.0; n],
            };
            let expected_mask: Vec<_> = (0..n)
                .map(|i| {
                    let (x, y) = ((i % w) as f32, (i / w) as f32);
                    let (nx, ny, s) = (map.nx[i], map.ny[i], map.strength[i]);
                    if s > HYSTERESIS_LOW
                        && s >= bilinear(&map.strength, w, h, x + nx, y + ny)
                        && s >= bilinear(&map.strength, w, h, x - nx, y - ny)
                    {
                        map.polarity[i]
                    } else {
                        0
                    }
                })
                .collect();
            let expected_seeds: Vec<_> = (0..n)
                .filter(|&i| expected_mask[i] != 0 && map.strength[i] > HYSTERESIS_HIGH)
                .map(|i| seed_key(i as u32, map.strength[i]))
                .collect();
            let polarity = map.polarity.clone();
            for threads in [1, 4] {
                map.polarity.clone_from(&polarity);
                let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, threads);
                let (mask, seeds) = ridge_mask(&ctx, &mut CoderScratch::default(), &mut map, w, h);
                assert_eq!(mask, expected_mask);
                assert_eq!(seeds, expected_seeds);
            }
        }
    }

    #[test]
    fn strong_seed_tracks_weak_tails_but_does_not_start_weak_only_lines() {
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let mut scratch = Box::<CoderScratch>::default();
        let mut image = Image3F::new(128, 128);
        for y in 0..128 {
            let amplitude = 0.04 + 0.12 * (-0.5 * ((y as f32 - 64.0) / 12.0).powi(2)).exp();
            for x in 0..128 {
                let seeded = amplitude * (-0.5 * ((x as f32 - 32.0) / 0.7).powi(2)).exp();
                let weak = 0.04 * (-0.5 * ((x as f32 - 96.0) / 0.7).powi(2)).exp();
                image.plane_row_mut(1, y)[x] = 0.4 + seeded + weak;
            }
        }
        let chains = detect_chains(&ctx, &mut scratch, &image);
        // A bright ridge can also yield opposite-polarity shoulder chains.
        // Check the center's continuation separately from those shoulders.
        let center = chains
            .iter()
            .find(|c| c.points.iter().all(|p| (p.x - 32.0).abs() < 1.0))
            .expect("strong center was not traced");
        assert!(center.points.len() > 100, "weak tails were lost");
        assert!(
            chains.iter().all(|c| c.points.iter().all(|p| p.x < 64.0)),
            "isolated weak line must not seed a chain"
        );
    }

    #[test]
    fn parallel_convolution_matches_clamped_reference() {
        for threads in [1, 4] {
            let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, threads);
            let mut scratch = Box::<CoderScratch>::default();
            for (w, h) in [(1, 1), (1, 9), (9, 1), (7, 3), (16, 4), (31, 19), (64, 65)] {
                let src: Vec<_> = (0..w * h).map(|i| (i % 37) as f32 * 0.013 - 0.2).collect();
                for sigma in [0.7, 1.0, 1.5, 2.2, 3.0] {
                    for kernel in gaussian_kernels(sigma) {
                        let radius = kernel.len() / 2;
                        let mut rows = vec![0f32; w * h];
                        let mut cols = vec![0f32; w * h];
                        for y in 0..h {
                            for x in 0..w {
                                for (j, &k) in kernel.iter().enumerate() {
                                    let xx = (x + j).saturating_sub(radius).min(w - 1);
                                    let yy = (y + j).saturating_sub(radius).min(h - 1);
                                    rows[y * w + x] += k * src[y * w + xx];
                                    cols[y * w + x] += k * src[yy * w + x];
                                }
                            }
                        }
                        let mut actual = vec![f32::NAN; w * h];
                        let horizontal = FilterPlan::new(w, &kernel);
                        let vertical = FilterPlan::new(h, &kernel);
                        horizontal.horizontal(&ctx, &mut scratch, &src, &mut actual);
                        assert_eq!(actual, rows);
                        vertical.vertical(&ctx, &mut scratch, &src, &mut actual);
                        assert_eq!(actual, cols);
                        horizontal.horizontal(&ctx, &mut scratch, &src, &mut actual);
                        assert_eq!(actual, rows, "reused destination must be overwritten");
                    }
                }
            }
        }
    }
}
