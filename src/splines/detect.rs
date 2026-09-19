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

use super::{Point, fast_hypot};
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

static SCALES: [f32; 4] = [0.7, 1.0, 1.5, 2.2];
const HYSTERESIS_HIGH: f32 = 0.02;
const HYSTERESIS_LOW: f32 = 0.008;
const MIN_CHAIN_LEN: usize = 20;
/// Line-ness: curvature across the ridge must dominate curvature along it.
const ALONG_PENALTY: f32 = 1.5;
const MAX_SUBPIXEL_OFFSET: f32 = 0.75;
const TRACE_MIN_COS: f32 = 0.35;
/// Chains are trimmed to this distance from the frame: a line leaving the image
/// keeps its interior part, a line hugging the border (scan frames) vanishes.
const FRAME_MARGIN: f32 = 4.0;

/// A traced ridge: sub-pixel center line and its strength-weighted ridge scale.
pub(super) struct Chain {
    pub(super) points: Vec<Point<f32>>,
    pub(super) scale: f32,
}

/// Runs `f(y, row)` over all rows on the encoder's thread pool.
pub(super) fn par_plane<F>(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    w: usize,
    h: usize,
    f: F,
) -> Vec<f32>
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let bands = (ctx.thread_pool.num_threads().max(1) * 4).min(h).max(1);
    let rows_per = h.div_ceil(bands);
    let mut plane = vec![0f32; w * h];
    let mut parts: Vec<_> = plane.chunks_mut(rows_per * w).collect();
    ctx.thread_pool
        .steal_for_each_mut(scratch, &mut parts, |b, rows, _| {
            for (y, row) in rows.chunks_exact_mut(w).enumerate() {
                f(b * rows_per + y, row);
            }
        });
    plane
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

pub(super) fn convolve_rows(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    src: &[f32],
    w: usize,
    h: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let radius = kernel.len() / 2;
    par_plane(ctx, scratch, w, h, |y, out| {
        let row = &src[y * w..(y + 1) * w];
        if w > 2 * radius {
            // tap-major over the interior keeps the inner loop a plain saxpy
            let inner = w - 2 * radius;
            for (j, &k) in kernel.iter().enumerate() {
                for (o, &v) in out[radius..radius + inner]
                    .iter_mut()
                    .zip(&row[j..j + inner])
                {
                    *o += k * v;
                }
            }
        }
        let edge = |x: usize| -> f32 {
            kernel
                .iter()
                .enumerate()
                .map(|(j, &k)| k * row[(x + j).saturating_sub(radius).min(w - 1)])
                .sum()
        };
        if w > 2 * radius {
            for x in (0..radius).chain(w - radius..w) {
                out[x] = edge(x);
            }
        } else {
            for (x, o) in out.iter_mut().enumerate() {
                *o = edge(x);
            }
        }
    })
}

pub(super) fn convolve_cols(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    src: &[f32],
    w: usize,
    h: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let radius = kernel.len() / 2;
    par_plane(ctx, scratch, w, h, |y, out| {
        out.fill(0.0);
        for (j, &k) in kernel.iter().enumerate() {
            let yi = (y + j).saturating_sub(radius).min(h - 1);
            for (o, &v) in out.iter_mut().zip(&src[yi * w..(yi + 1) * w]) {
                *o += k * v;
            }
        }
    })
}

struct RidgeMap {
    strength: Vec<f32>,
    nx: Vec<f32>,
    ny: Vec<f32>,
    scale: Vec<f32>,
    polarity: Vec<i8>,
    offset: Vec<f32>,
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
    for s in SCALES {
        let [k0, k1, k2] = gaussian_kernels(s);
        let r0 = convolve_rows(ctx, scratch, luma, w, h, &k0);
        let r1 = convolve_rows(ctx, scratch, luma, w, h, &k1);
        let r2 = convolve_rows(ctx, scratch, luma, w, h, &k2);
        let hxx = convolve_cols(ctx, scratch, &r2, w, h, &k0);
        let hyy = convolve_cols(ctx, scratch, &r0, w, h, &k2);
        let hxy = convolve_cols(ctx, scratch, &r1, w, h, &k1);
        let gx = convolve_cols(ctx, scratch, &r1, w, h, &k0);
        let gy = convolve_cols(ctx, scratch, &r0, w, h, &k1);
        let s2 = s * s;
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
        let (hxx, hyy, hxy, gx, gy) = (&hxx, &hyy, &hxy, &gx, &gy);
        ctx.thread_pool
            .steal_for_each_mut(scratch, &mut items, |_, band, _| {
                for k in 0..band.strength.len() {
                    let i = band.start + k;
                    let tr = 0.5 * (hxx[i] + hyy[i]);
                    let df = ((0.5 * (hxx[i] - hyy[i])).powi(2) + hxy[i] * hxy[i]).sqrt();
                    let (la, lb) = (tr + df, tr - df);
                    let (big, small) = if la.abs() > lb.abs() {
                        (la, lb)
                    } else {
                        (lb, la)
                    };
                    let strength = s2 * (big.abs() - ALONG_PENALTY * small.abs()).max(0.0);
                    if strength <= band.strength[k] {
                        continue;
                    }
                    // a line center has ~zero gradient, a step edge does not
                    if s * fast_hypot(gx[i], gy[i]) > s2 * big.abs() {
                        continue;
                    }
                    let (mut vx, mut vy) = (hxy[i], big - hxx[i]);
                    let norm = fast_hypot(vx, vy);
                    if norm < 1e-12 {
                        (vx, vy) = (1.0, 0.0);
                    } else {
                        (vx, vy) = (vx / norm, vy / norm);
                    }
                    let t = if big.abs() < 1e-12 {
                        0.0
                    } else {
                        (-(gx[i] * vx + gy[i] * vy) / big)
                            .clamp(-MAX_SUBPIXEL_OFFSET, MAX_SUBPIXEL_OFFSET)
                    };
                    band.strength[k] = strength;
                    band.nx[k] = vx;
                    band.ny[k] = vy;
                    band.scale[k] = s;
                    band.polarity[k] = if big > 0.0 { -1 } else { 1 };
                    band.offset[k] = t;
                }
            });
    }
    map
}

#[inline]
fn bilinear(plane: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = plane[y0 * w + x0] * (1.0 - fx) + plane[y0 * w + x1] * fx;
    let bottom = plane[y1 * w + x0] * (1.0 - fx) + plane[y1 * w + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Ridge peaks above the weak threshold. Tracing starts only at strong peaks,
/// so each retained chain contains its own strong seed without a flood fill.
fn ridge_mask(map: &RidgeMap, w: usize, h: usize) -> Vec<bool> {
    let n = w * h;
    let mut weak = vec![false; n];
    let rows = weak
        .chunks_exact_mut(w)
        .zip(map.strength.chunks_exact(w))
        .zip(map.nx.chunks_exact(w).zip(map.ny.chunks_exact(w)));
    for (y, ((weak, strength), (nx, ny))) in rows.enumerate() {
        let pixels = weak.iter_mut().zip(strength).zip(nx.iter().zip(ny));
        for (x, ((weak, &s), (&nx, &ny))) in pixels.enumerate() {
            if s <= HYSTERESIS_LOW {
                continue;
            }
            let (fx, fy) = (x as f32, y as f32);
            let a = bilinear(&map.strength, w, h, fx + nx, fy + ny);
            let b = bilinear(&map.strength, w, h, fx - nx, fy - ny);
            *weak = s >= a && s >= b;
        }
    }
    weak
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

/// Greedy tangent-following walk over the available ridge pixels.
fn walk(
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
            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize || (xx, yy) == (nx_, ny_) {
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

pub(super) fn detect_chains(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
) -> Vec<Chain> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let map = ridge_map(ctx, scratch, xyb.plane_data(1), w, h);
    let mut avail = ridge_mask(&map, w, h);
    let mut order: Vec<u32> = (0..(w * h) as u32)
        .filter(|&i| avail[i as usize] && map.strength[i as usize] > HYSTERESIS_HIGH)
        .collect();
    order.sort_by(|&a, &b| map.strength[b as usize].total_cmp(&map.strength[a as usize]));

    let mut chains = Vec::new();
    for i in order {
        let i = i as usize;
        if !avail[i] {
            continue;
        }
        avail[i] = false;
        let start = (i % w, i / w);
        let tangent = (-map.ny[i], map.nx[i]);
        let polarity = map.polarity[i];
        let forward = walk(&map, &mut avail, w, h, start, tangent, polarity);
        let backward = walk(
            &map,
            &mut avail,
            w,
            h,
            start,
            (-tangent.0, -tangent.1),
            polarity,
        );
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
    use crate::{Speed, xyb::XybMatrix};

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
            for (w, h) in [(7, 3), (31, 19)] {
                let src: Vec<_> = (0..w * h).map(|i| (i % 37) as f32 * 0.013 - 0.2).collect();
                for sigma in [0.7, 1.5] {
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
                        assert_eq!(convolve_rows(&ctx, &mut scratch, &src, w, h, &kernel), rows);
                        assert_eq!(convolve_cols(&ctx, &mut scratch, &src, w, h, &kernel), cols);
                    }
                }
            }
        }
    }
}
