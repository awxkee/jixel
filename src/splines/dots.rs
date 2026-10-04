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

//! Bright and dark isolated dots as signed Gaussians from a small template atlas.
//!
//! Candidates ride through the spline RD selection as one-point
//! `QuantizedSpline`s (a one-point spline renders nothing in the format, so
//! the shape is free as a marker): points[0] = centre pixel, dct[1][0] = Y
//! level on a geometric lattice, dct[0][0] = X/Y ratio steps,
//! dct[2][0] = (B-Y)/Y ratio steps, dct[3][0] = signed sigma lattice index
//! (negative for dark dots). Selected
//! dots become lattice templates in a modular reference frame, placed with
//! kAdd patch entries.

use super::fit::Candidate;
use super::{
    CHANNEL_WEIGHT, Point, QUANT_ADJUST, QuantizedSpline, adjusted_quant, blob_weight, round_i32,
};
use crate::adaptive_quant::{dirty_log2f, fast_exp2};
use crate::coder_scratch::CoderScratch;
use crate::image::Image3F;
use crate::patches::{DOT_PATCH_REF_ID, PatchReference};
use crate::thread_pool::ThreadPool;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

const OPEN_RADIUS: usize = 3;
const RING: isize = 4;
const BORDER: usize = 5;
const MIN_TOP_HAT: f32 = 0.025;
const MIN_DARK_TOP_HAT: f32 = 0.015;
const MIN_EXPLAINED: f32 = 0.65;
const STAR_FIELD_MIN_EXPLAINED: f32 = 0.15;
const STAR_FIELD_DENSITY: f32 = 400.0;
const MIN_ISOLATION: f32 = 5.0;
const Y0: f32 = 0.002;
const Y_LEVELS_PER_OCTAVE: f32 = 2.0;
const X_RATIO_STEP: f32 = 0.02;
const B_RATIO_STEP: f32 = 0.25;
const MAX_RATIO_STEPS: f32 = 4.0;

pub(crate) type TemplateKey = (i32, i32, i32, i32);

const BAND: usize = 64;

/// Running min (`max == false`) or max over `2 * OPEN_RADIUS + 1` samples of
/// `src`, clipped at the row ends: two-sample windows, then four-sample
/// windows, then the overlapping pair of four-sample windows.
fn row_extrema(src: &[f32], out: &mut [f32], pad: &mut Vec<f32>, max: bool) {
    const R: usize = OPEN_RADIUS;
    let fill = if max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let pick = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    let w = src.len();
    pad.clear();
    pad.resize(R, fill);
    pad.extend_from_slice(src);
    pad.resize(w + 2 * R + 1, fill);
    for j in 0..w + 2 * R {
        pad[j] = pick(pad[j], pad[j + 1]);
    }
    for j in 0..w + R + 1 {
        pad[j] = pick(pad[j], pad[j + 2]);
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = pick(pad[i], pad[i + R]);
    }
}

/// Elementwise min or max of the clipped `2 * OPEN_RADIUS + 1` rows around
/// each row of `out_rows`. `rows` holds image rows from `first` on.
fn column_extrema(
    rows: &[f32],
    first: usize,
    h: usize,
    w: usize,
    out_rows: std::ops::Range<usize>,
    out: &mut Vec<f32>,
    max: bool,
) {
    out.clear();
    for k in out_rows {
        let lo = k.saturating_sub(OPEN_RADIUS);
        let hi = (k + OPEN_RADIUS).min(h - 1);
        let start = out.len();
        out.extend_from_slice(&rows[(lo - first) * w..][..w]);
        let dst = &mut out[start..];
        for r in lo + 1..=hi {
            let src = &rows[(r - first) * w..][..w];
            if max {
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = d.max(s);
                }
            } else {
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = d.min(s);
                }
            }
        }
    }
}

/// Local maxima of the white and black Y top-hats, `Y - opening(Y)` and
/// `closing(Y) - Y`, at or above each polarity's threshold. Raster order,
/// bright before dark at a pixel; dark peaks are negated. Each band computes
/// its own top-hat rows, and closing(Y) = -opening(-Y) lets both polarities
/// share the opening, so no image-sized buffer is formed.
fn top_hat_peaks(
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
) -> Vec<(f32, usize, usize)> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let r = OPEN_RADIUS;
    pool.steal_map(scratch, (h - 2 * BORDER).div_ceil(BAND), |band, _| {
        let ya = BORDER + band * BAND;
        let yb = (BORDER + (band + 1) * BAND).min(h - BORDER);
        // Top-hat rows t0..t1 hold the 3x3 neighbourhoods of the peak rows;
        // the dilation reads eroded rows e0..e1, the erosion input rows i0..i1.
        let (t0, t1) = (ya - 1, yb + 1);
        let (e0, e1) = (t0.saturating_sub(r), (t1 + r).min(h));
        let (i0, i1) = (t0.saturating_sub(2 * r), (t1 + 2 * r).min(h));
        let mut pad = Vec::with_capacity(w + 2 * r + 1);
        let mut line = vec![0f32; w];
        let mut eroded_rows = Vec::with_capacity((i1 - i0) * w);
        let mut eroded = Vec::with_capacity((e1 - e0) * w);
        let mut dilated_rows = Vec::with_capacity((e1 - e0) * w);
        let mut th: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for (polarity, th) in th.iter_mut().enumerate() {
            let sign = if polarity == 0 { 1.0 } else { -1.0 };
            eroded_rows.clear();
            for y in i0..i1 {
                for (l, &v) in line.iter_mut().zip(xyb.plane_row(1, y)) {
                    *l = sign * v;
                }
                let start = eroded_rows.len();
                eroded_rows.resize(start + w, 0.0);
                row_extrema(&line, &mut eroded_rows[start..], &mut pad, false);
            }
            column_extrema(&eroded_rows, i0, h, w, e0..e1, &mut eroded, false);
            dilated_rows.clear();
            dilated_rows.resize((e1 - e0) * w, 0.0);
            for (src, dst) in eroded.chunks_exact(w).zip(dilated_rows.chunks_exact_mut(w)) {
                row_extrema(src, dst, &mut pad, true);
            }
            column_extrema(&dilated_rows, e0, h, w, t0..t1, th, true);
            for (y, row) in (t0..t1).zip(th.chunks_exact_mut(w)) {
                for (o, &v) in row.iter_mut().zip(xyb.plane_row(1, y)) {
                    *o = (sign * v - *o).max(0.0);
                }
            }
        }
        let mut peaks = Vec::new();
        for y in ya..yb {
            for x in BORDER..w - BORDER {
                for (polarity, th) in th.iter().enumerate() {
                    let at = |dx: isize, dy: isize| {
                        th[((y as isize + dy) as usize - t0) * w + (x as isize + dx) as usize]
                    };
                    let v = at(0, 0);
                    let threshold = if polarity == 0 {
                        MIN_TOP_HAT
                    } else {
                        MIN_DARK_TOP_HAT
                    };
                    if v < threshold {
                        continue;
                    }
                    let mut is_max = true;
                    'n: for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let o = at(dx, dy);
                            let earlier = dy < 0 || (dy == 0 && dx < 0);
                            if o > v || (earlier && o == v) {
                                is_max = false;
                                break 'n;
                            }
                        }
                    }
                    if is_max {
                        peaks.push((if polarity == 0 { v } else { -v }, x, y));
                    }
                }
            }
        }
        peaks
    })
    .concat()
}

fn median(v: &mut [f32]) -> f32 {
    let k = v.len() / 2;
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

fn sigma_step() -> f32 {
    CHANNEL_WEIGHT[3] / adjusted_quant(QUANT_ADJUST)
}

fn quantized_y(level: i32) -> f32 {
    let exponent = level as f32 / Y_LEVELS_PER_OCTAVE;
    let scale = if (-126.0..=127.0).contains(&exponent) {
        fast_exp2(exponent)
    } else {
        exponent.exp2()
    };
    Y0 * scale
}

fn template_params(key: TemplateKey) -> ([f32; 3], f32) {
    let (n, level, xc, bc) = key;
    let y = quantized_y(level) * n.signum() as f32;
    let x = y * xc as f32 * X_RATIO_STEP;
    let b = y * (1.0 + bc as f32 * B_RATIO_STEP);
    ([x, y, b], n.abs() as f32 * sigma_step())
}

pub(crate) fn is_dot(sp: &QuantizedSpline) -> bool {
    sp.points.len() == 1
}

/// Centre, [X, Y, B] color and sigma of a dot.
pub(crate) fn dot_params(sp: &QuantizedSpline) -> (Point<f32>, [f32; 3], f32) {
    let c = Point {
        x: sp.points[0].x as f32,
        y: sp.points[0].y as f32,
    };
    let (color, sigma) = template_params(template_key(sp));
    (c, color, sigma)
}

/// Render reach, the spline renderer's rule.
pub(crate) fn dot_reach(color: [f32; 3], sigma: f32) -> f32 {
    let r = 0.1f32.ln() * 5.0;
    let max_color = color.iter().fold(0.01f32, |m, c| m.max(c.abs()));
    (-2.0 * sigma * sigma * (r - dirty_log2f(max_color) * std::f32::consts::LN_2)).sqrt()
}

/// Template box: top-left pixel and side.
fn dot_box(sp: &QuantizedSpline) -> (i32, i32, usize) {
    let (_, color, sigma) = dot_params(sp);
    let r = dot_reach(color, sigma).ceil() as i32;
    (sp.points[0].x - r, sp.points[0].y - r, (2 * r + 1) as usize)
}

pub(crate) fn template_key(sp: &QuantizedSpline) -> TemplateKey {
    (sp.dct[3][0], sp.dct[1][0], sp.dct[0][0], sp.dct[2][0])
}

struct FitKernel {
    weights: Vec<f32>,
    energy: f32,
}

// Bounded by one peak chunk; integer-centered kernels are shared across colors
// and polarities without retaining image-dependent fitting scratch.
type FitKernelKey = (i32, isize, i32, i32, Option<(i32, i32)>);
type FitKernels = HashMap<FitKernelKey, FitKernel>;

/// Gray and colored dots at the sigma lattice widths around `n0` whose Y
/// projection explains at least `min_explained` of the target's Y energy,
/// with that share.
#[allow(clippy::too_many_arguments)]
fn dot_alts(
    cx: f32,
    cy: f32,
    n0: i32,
    targets: &[Vec<f32>; 3],
    x: i32,
    y: i32,
    r: isize,
    polarity: f32,
    min_explained: f32,
    kernels: &mut FitKernels,
) -> Vec<(QuantizedSpline, f32)> {
    let side = (2 * r + 1) as usize;
    let (px, py) = (cx.round() as i32, cy.round() as i32);
    let mut alts = Vec::new();
    let energy: f32 = targets[1].iter().map(|v| v * v).sum();
    for n in [n0 - 1, n0, n0 + 1] {
        if n < 1 {
            continue;
        }
        // Beyond f32's exact integer range, rounding depends on the origin.
        let origin = ((x as i64).abs() + r as i64 >= 1 << 24
            || (y as i64).abs() + r as i64 >= 1 << 24)
            .then_some((x, y));
        let kernel = kernels
            .entry((n, r, x - px, y - py, origin))
            .or_insert_with(|| {
                let sigma = n as f32 * sigma_step();
                let mut weights = vec![0f32; side * side];
                for dy in -r..=r {
                    for dx in -r..=r {
                        weights[((dy + r) as usize) * side + (dx + r) as usize] = blob_weight(
                            (x as isize + dx) as f32 - px as f32,
                            (y as isize + dy) as f32 - py as f32,
                            1.0 / sigma,
                            0.25 * sigma,
                        );
                    }
                }
                let energy = weights.iter().map(|v| v * v).sum();
                FitKernel { weights, energy }
            });
        let ff = kernel.energy;
        if ff <= 0.0 {
            continue;
        }
        let proj = |t: &[f32]| {
            t.iter()
                .zip(&kernel.weights)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / ff
        };
        let cyv = proj(&targets[1]);
        // Reject edges and texture before paying for chroma and template RD.
        let explained = cyv * cyv * ff;
        if explained < min_explained * energy {
            continue;
        }
        if cyv * polarity < Y0 {
            continue;
        }
        let (cxv, cbv) = (proj(&targets[0]), proj(&targets[2]));
        let level = (Y_LEVELS_PER_OCTAVE * dirty_log2f(cyv.abs() / Y0)).round() as i32;
        let yq = quantized_y(level) * polarity;
        let xc = (cxv / yq / X_RATIO_STEP)
            .round()
            .clamp(-MAX_RATIO_STEPS, MAX_RATIO_STEPS) as i32;
        let bc = ((cbv / yq - 1.0) / B_RATIO_STEP)
            .round()
            .clamp(-MAX_RATIO_STEPS, MAX_RATIO_STEPS) as i32;
        let mut dct = [[0i32; 32]; 4];
        dct[0][0] = xc;
        dct[1][0] = level;
        dct[2][0] = bc;
        dct[3][0] = n * polarity as i32;
        alts.push((
            QuantizedSpline {
                points: vec![Point { x: px, y: py }],
                dct,
            },
            explained / energy,
        ));
    }
    alts
}

/// Isolated extrema of the signed Y top-hat, strongest contrast first.
pub(super) fn dot_candidates(
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
) -> Vec<Candidate> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    if w <= 2 * BORDER || h <= 2 * BORDER {
        return Vec::new();
    }
    let peaks = top_hat_peaks(pool, scratch, xyb);
    // Fit in raster order, which keeps each chunk's ring reads local, then
    // put candidates strongest contrast first, ties in raster order.
    let mut fitted: Vec<(usize, Candidate, Vec<f32>)> = pool
        .steal_map(scratch, peaks.len().div_ceil(PEAK_CHUNK), |chunk, _| {
            let start = chunk * PEAK_CHUNK;
            let mut fitted = fit_peaks(xyb, &peaks[start..(start + PEAK_CHUNK).min(peaks.len())]);
            for (index, _, _) in &mut fitted {
                *index += start;
            }
            fitted
        })
        .into_iter()
        .flatten()
        .collect();
    fitted.sort_unstable_by_key(|&(index, _, _)| (!peaks[index].0.abs().to_bits(), index));
    // Star fields keep loosely fitted bright dots: neighbours and nebulae
    // share a star's fit window. Elsewhere such dots are non-Gaussian blobs
    // the block model over-prices, and only compact fits compete.
    let compact = |explained: &f32| *explained >= MIN_EXPLAINED;
    let bright = |candidate: &Candidate| candidate.alts[0].dct[3][0] > 0;
    let compact_bright = fitted
        .iter()
        .filter(|(_, candidate, explained)| bright(candidate) && explained.iter().any(compact))
        .count();
    let star_field = compact_bright as f32 * 1e6 >= STAR_FIELD_DENSITY * (w * h) as f32;
    let mut out: Vec<Candidate> = fitted
        .into_iter()
        .filter_map(|(_, mut candidate, explained)| {
            if !star_field && bright(&candidate) {
                let mut explained = explained.iter();
                candidate
                    .alts
                    .retain(|_| compact(explained.next().unwrap()));
            }
            (!candidate.alts.is_empty()).then_some(candidate)
        })
        .collect();
    finish_alts(&mut out, w, h);
    out
}

const PEAK_CHUNK: usize = 1024;

/// Fits dot alternatives to `peaks`, skipping peaks in busy surroundings.
/// Returns each candidate's peak index and its alternatives' explained Y
/// energy shares.
fn fit_peaks(xyb: &Image3F, peaks: &[(f32, usize, usize)]) -> Vec<(usize, Candidate, Vec<f32>)> {
    let mut out = Vec::new();
    let mut ring = Vec::with_capacity(32);
    let mut kernels = FitKernels::new();
    'peaks: for (index, &(peak, x, y)) in peaks.iter().enumerate() {
        let polarity = peak.signum();
        let (xi, yi) = (x as isize, y as isize);
        let px = |c: usize, dx: isize, dy: isize| {
            xyb.plane_row(c, (yi + dy) as usize)[(xi + dx) as usize]
        };
        // Per-channel background: median of the ring; a peak in busy
        // surroundings (texture) is not a dot.
        let mut bg = [0f32; 3];
        let mut slope = [[0f32; 2]; 3];
        // Test Y isolation before fitting chroma: textured images reject most
        // peaks here and need neither of the other two ring medians.
        for c in [1, 0, 2] {
            if polarity < 0.0 {
                // Opposite ring differences estimate the illumination slope
                // robustly; an isolated mark on the ring cannot set the plane.
                let mut gradients = [0f32; (2 * RING + 1) as usize];
                for (i, k) in (-RING..=RING).enumerate() {
                    gradients[i] = (px(c, RING, k) - px(c, -RING, k)) / (2 * RING) as f32;
                }
                slope[c][0] = median(&mut gradients);
                for (i, k) in (-RING..=RING).enumerate() {
                    gradients[i] = (px(c, k, RING) - px(c, k, -RING)) / (2 * RING) as f32;
                }
                slope[c][1] = median(&mut gradients);
            }
            let detrended = |dx: isize, dy: isize| {
                px(c, dx, dy) - slope[c][0] * dx as f32 - slope[c][1] * dy as f32
            };
            ring.clear();
            for k in -RING..=RING {
                ring.push(detrended(k, -RING));
                ring.push(detrended(k, RING));
                if k != -RING && k != RING {
                    ring.push(detrended(-RING, k));
                    ring.push(detrended(RING, k));
                }
            }
            bg[c] = median(&mut ring);
            if c == 1 {
                // The scaled median absolute deviation passes exactly when
                // more than half of the ring's deviations pass.
                let med = bg[c];
                let passing = ring
                    .iter()
                    .filter(|&&v| peak.abs() >= MIN_ISOLATION * ((v - med).abs() * 1.4826 + 1e-4))
                    .count();
                if passing <= ring.len() / 2 {
                    continue 'peaks;
                }
            }
        }
        let contrast = |c: usize, dx: isize, dy: isize| {
            px(c, dx, dy) - bg[c] - slope[c][0] * dx as f32 - slope[c][1] * dy as f32
        };
        // Centroid and spread of the Y contrast.
        let (mut sw, mut sx, mut sy) = (0f32, 0f32, 0f32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let v = (polarity * contrast(1, dx, dy)).max(0.0);
                sw += v;
                sx += v * dx as f32;
                sy += v * dy as f32;
            }
        }
        if sw <= 0.0 {
            continue;
        }
        let (cx, cy) = (x as f32 + sx / sw, y as f32 + sy / sw);
        let (mut vw, mut vr) = (0f32, 0f32);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let v = (polarity * contrast(1, dx, dy)).max(0.0);
                let (rx, ry) = (xi as f32 + dx as f32 - cx, yi as f32 + dy as f32 - cy);
                vw += v;
                vr += v * (rx * rx + ry * ry);
            }
        }
        let var = (vr / vw.max(1e-9) * 0.5 - 1.0 / 12.0).max(0.04);
        let n0 = (var.sqrt() / sigma_step()).round().clamp(1.0, 10.0) as i32;
        let r = ((3.0 * (n0 + 1) as f32 * sigma_step() + 1.5).ceil() as isize).min(RING);
        let side = (2 * r + 1) as usize;
        let mut targets = [
            vec![0f32; side * side],
            vec![0f32; side * side],
            vec![0f32; side * side],
        ];
        for (c, t) in targets.iter_mut().enumerate() {
            for dy in -r..=r {
                for dx in -r..=r {
                    t[((dy + r) as usize) * side + (dx + r) as usize] = contrast(c, dx, dy);
                }
            }
        }
        let alts = dot_alts(
            cx,
            cy,
            n0,
            &targets,
            x as i32,
            y as i32,
            r,
            polarity,
            if polarity > 0.0 {
                STAR_FIELD_MIN_EXPLAINED
            } else {
                MIN_EXPLAINED
            },
            &mut kernels,
        );
        if !alts.is_empty() {
            let (alts, explained) = alts.into_iter().unzip();
            out.push((
                index,
                Candidate {
                    alts,
                    bits_factor: 1.0,
                    dot: true,
                },
                explained,
            ));
        }
    }
    out
}

/// Every fitted color also competes with gray and with the most common
/// fitted color: the colors templates are most likely to share.
fn finish_alts(out: &mut Vec<Candidate>, w: usize, h: usize) {
    let mut counts: BTreeMap<(bool, i32, i32), usize> = BTreeMap::new();
    for c in out.iter() {
        let sp = &c.alts[c.alts.len() / 2];
        *counts
            .entry((sp.dct[3][0] < 0, sp.dct[0][0], sp.dct[2][0]))
            .or_insert(0) += 1;
    }
    // Chroma ratios of dark spots can differ from highlights on the same
    // image. Share each polarity's most common fitted color separately.
    let common: [_; 2] = std::array::from_fn(|dark| {
        counts
            .iter()
            .filter(|&(&(polarity, _, _), _)| polarity == (dark != 0))
            .max_by_key(|&(&(_, x, b), &n)| (n, std::cmp::Reverse((x, b))))
            .map_or((0, 0), |(&(_, x, b), _)| (x, b))
    });
    let inside = |sp: &QuantizedSpline| {
        let (x0, y0, side) = dot_box(sp);
        x0 >= 0 && y0 >= 0 && x0 as usize + side <= w && y0 as usize + side <= h
    };
    for c in out.iter_mut() {
        let common = common[usize::from(c.alts[0].dct[3][0] < 0)];
        let shared = if common == (0, 0) {
            vec![common]
        } else {
            vec![common, (0, 0)]
        };
        for sp in std::mem::take(&mut c.alts) {
            let color = (sp.dct[0][0], sp.dct[2][0]);
            if !shared.contains(&color) && inside(&sp) {
                c.alts.push(sp.clone());
            }
            for &color in &shared {
                let mut alt = sp.clone();
                (alt.dct[0][0], alt.dct[2][0]) = color;
                if inside(&alt) {
                    c.alts.push(alt);
                }
            }
        }
    }
    out.retain(|c| !c.alts.is_empty());
}

/// Dot templates on the default LF dequant lattice and their placements.
pub(crate) struct DotPatches {
    /// Y, X, B-Y lattice channels of the atlas.
    pub(crate) atlas: [Vec<i32>; 3],
    pub(crate) width: usize,
    pub(crate) height: usize,
    references: Vec<PatchReference>,
}

impl DotPatches {
    pub(crate) fn references(&self) -> Vec<PatchReference> {
        self.references.clone()
    }

    pub(super) fn sort_positions(&mut self, band: Option<usize>) {
        for reference in &mut self.references {
            if let Some(band) = band {
                reference.positions.sort_unstable_by_key(|&(x, y)| {
                    let row = y / band;
                    (row, if row & 1 == 0 { x } else { usize::MAX - x }, y)
                });
            } else {
                reference
                    .positions
                    .sort_unstable_by_key(|&(x, y)| morton(x, y));
            }
        }
    }
}

fn morton(x: usize, y: usize) -> u64 {
    let spread = |v: usize| {
        let mut v = v as u64;
        v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
        v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        (v | (v << 1)) & 0x5555_5555_5555_5555
    };
    spread(x) | (spread(y) << 1)
}

/// Shared Gaussian proposals and the decoder's lattice pixels. Only the
/// nonzero lattice support is needed when checking the retained residual.
pub(super) struct DotTemplate {
    pub(super) key: TemplateKey,
    pub(super) radius: i32,
    pub(super) atlas_radius: i32,
    pub(super) side: usize,
    /// Three contiguous planes in a single allocation for each representation.
    pub(super) lattice: Vec<i32>,
    pub(super) xyb: Vec<f32>,
    pub(super) empty: bool,
    pub(super) trial_radius: i32,
    pub(super) trial_side: usize,
    pub(super) trial_xyb: Vec<f32>,
}

impl DotTemplate {
    pub(super) fn new(key: TemplateKey) -> Self {
        let (color, sigma) = template_params(key);
        let reach = dot_reach(color, sigma);
        let trial_radius = round_i32(reach);
        let trial_side = (2 * trial_radius + 1) as usize;
        let trial_area = trial_side * trial_side;
        let mut trial_xyb = vec![0f32; 3 * trial_area];
        let (x_plane, yb_planes) = trial_xyb.split_at_mut(trial_area);
        let (y_plane, b_plane) = yb_planes.split_at_mut(trial_area);
        for (j, ((x_row, y_row), b_row)) in x_plane
            .chunks_exact_mut(trial_side)
            .zip(y_plane.chunks_exact_mut(trial_side))
            .zip(b_plane.chunks_exact_mut(trial_side))
            .enumerate()
        {
            let y = j as i32 - trial_radius;
            for (i, ((x, y_out), b)) in x_row.iter_mut().zip(y_row).zip(b_row).enumerate() {
                let dx = i as i32 - trial_radius;
                let weight = blob_weight(dx as f32, y as f32, 1.0 / sigma, 0.25 * sigma);
                *x = color[0] * weight;
                *y_out = color[1] * weight;
                *b = color[2] * weight;
            }
        }
        let atlas_radius = reach.ceil() as i32;
        let (x0, y0) = (-atlas_radius, -atlas_radius);
        let side = (2 * atlas_radius + 1) as usize;
        let area = side * side;
        const INLINE_SIDE: usize = 25;
        let mut inline = [0i32; 3 * INLINE_SIDE * INLINE_SIDE];
        let mut heap;
        let planes = if side <= INLINE_SIDE {
            &mut inline[..3 * area]
        } else {
            heap = vec![0i32; 3 * area];
            &mut heap[..]
        };
        let mut radius = 0;
        let mut empty = true;
        let (y_plane, xb_planes) = planes.split_at_mut(area);
        let (x_plane, by_plane) = xb_planes.split_at_mut(area);
        for (j, ((y_row, x_row), by_row)) in y_plane
            .chunks_exact_mut(side)
            .zip(x_plane.chunks_exact_mut(side))
            .zip(by_plane.chunks_exact_mut(side))
            .enumerate()
        {
            let dy = y0 + j as i32;
            for (i, ((y, x), by)) in y_row.iter_mut().zip(x_row).zip(by_row).enumerate() {
                let dx = x0 + i as i32;
                let wgt = blob_weight(dx as f32, dy as f32, 1.0 / sigma, 0.25 * sigma);
                let yi = round_i32(color[1] * wgt * 512.0);
                let xi = round_i32(color[0] * wgt * 4096.0);
                let byi = round_i32(color[2] * wgt * 256.0) - yi;
                *y = yi;
                *x = xi;
                *by = byi;
                if yi != 0 || xi != 0 || byi != 0 {
                    empty = false;
                    radius = radius.max(dx.abs().max(dy.abs()));
                }
            }
        }
        let trimmed_side = (2 * radius + 1) as usize;
        let trimmed_area = trimmed_side * trimmed_side;
        let offset = (-x0 - radius) as usize;
        let mut lattice = Vec::with_capacity(3 * trimmed_area);
        for p in planes.chunks_exact(area) {
            for j in offset..offset + trimmed_side {
                lattice.extend_from_slice(&p[j * side + offset..][..trimmed_side]);
            }
        }
        const INV_X_SCALE: f32 = 1.0 / 4096.0;
        const INV_Y_SCALE: f32 = 1.0 / 512.0;
        const INV_B_SCALE: f32 = 1.0 / 256.0;
        let (y_plane, xb_planes) = lattice.split_at(trimmed_area);
        let (x_plane, by_plane) = xb_planes.split_at(trimmed_area);
        let mut xyb = Vec::with_capacity(3 * trimmed_area);
        xyb.extend(x_plane.iter().map(|&v| v as f32 * INV_X_SCALE));
        xyb.extend(y_plane.iter().map(|&v| v as f32 * INV_Y_SCALE));
        xyb.extend(
            by_plane
                .iter()
                .zip(y_plane)
                .map(|(&b, &y)| (b + y) as f32 * INV_B_SCALE),
        );
        Self {
            key,
            radius,
            atlas_radius,
            side: trimmed_side,
            lattice,
            xyb,
            empty,
            trial_radius,
            trial_side,
            trial_xyb,
        }
    }
}

/// Share each fitted shape across all placements and repricing passes.
pub(super) fn share_templates(
    templates: impl IntoIterator<Item = (TemplateKey, DotTemplate)>,
) -> HashMap<TemplateKey, Arc<DotTemplate>> {
    templates
        .into_iter()
        .map(|(key, template)| (key, Arc::new(template)))
        .collect()
}

/// The final RGB atlas encoding can be carried from RD pruning to output.
#[derive(Clone)]
pub(crate) struct DotEncoding {
    pub(crate) atlas: Arc<crate::bit_writer::BitWriter>,
    pub(crate) references: Vec<PatchReference>,
}

impl std::fmt::Debug for DotEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DotEncoding")
            .field("atlas_bits", &self.atlas.bits_written())
            .field("templates", &self.references.len())
            .finish()
    }
}

/// Groups dots by template and shelf-packs the atlas.
pub(crate) fn build_dot_patches(dots: &[QuantizedSpline]) -> DotPatches {
    let keys: std::collections::BTreeSet<_> = dots.iter().map(template_key).collect();
    let templates = share_templates(keys.into_iter().map(|key| (key, DotTemplate::new(key))));
    build_dot_patches_with_templates(dots, &templates, false)
}

pub(super) fn build_dot_patches_with_templates(
    dots: &[QuantizedSpline],
    templates: &HashMap<TemplateKey, Arc<DotTemplate>>,
    trim: bool,
) -> DotPatches {
    type TemplatePlacements = (Arc<DotTemplate>, Vec<(usize, usize)>);
    let mut groups: BTreeMap<TemplateKey, TemplatePlacements> = BTreeMap::new();
    for sp in dots {
        let template = &templates[&template_key(sp)];
        groups
            .entry(template.key)
            .or_insert_with(|| (template.clone(), Vec::new()))
            .1
            .push((sp.points[0].x as usize, sp.points[0].y as usize));
    }
    let templates: Vec<_> = groups
        .into_values()
        .map(|(template, mut positions)| {
            let radius = if trim {
                template.radius
            } else {
                template.atlas_radius
            };
            let side = (2 * radius + 1) as usize;
            for (x, y) in &mut positions {
                *x -= radius as usize;
                *y -= radius as usize;
            }
            positions.sort_unstable_by_key(|&(x, y)| (y, x));
            (side, template, positions)
        })
        .collect();
    let area: usize = templates.iter().map(|t| t.0 * t.0).sum();
    let max_side = templates.iter().map(|t| t.0).max().unwrap_or(1);
    let width = (((area as f64).sqrt() * 1.5) as usize).clamp(max_side.max(16), 1024);
    let mut slots = Vec::with_capacity(templates.len());
    let (mut ax, mut ay, mut shelf) = (0usize, 0usize, 0usize);
    for t in &templates {
        if ax + t.0 > width {
            ax = 0;
            ay += shelf;
            shelf = 0;
        }
        slots.push((ax, ay));
        ax += t.0;
        shelf = shelf.max(t.0);
    }
    let height = (ay + shelf).max(1);
    let mut atlas = [
        vec![0i32; width * height],
        vec![0i32; width * height],
        vec![0i32; width * height],
    ];
    let mut references = Vec::with_capacity(templates.len());
    for ((side, template, positions), (sx, sy)) in templates.into_iter().zip(slots) {
        let padding = (side - template.side) / 2;
        for (dst, src) in atlas
            .iter_mut()
            .zip(template.lattice.chunks_exact(template.side * template.side))
        {
            for j in 0..template.side {
                dst[(sy + padding + j) * width + sx + padding..][..template.side]
                    .copy_from_slice(&src[j * template.side..][..template.side]);
            }
        }
        references.push(PatchReference {
            atlas_x: sx,
            atlas_y: sy,
            width: side,
            height: side,
            ref_frame: DOT_PATCH_REF_ID,
            add: true,
            positions,
        });
    }
    DotPatches {
        atlas,
        width,
        height,
        references,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_spots_fit_on_sloped_backgrounds_but_reject_texture() {
        for slope in [0.0, 0.003, 0.008] {
            let mut image = Image3F::new(64, 64);
            for c in 0..3 {
                for y in 0..64 {
                    for x in 0..64 {
                        let (dx, dy) = (x as f32 - 32.0, y as f32 - 32.0);
                        image.plane_row_mut(c, y)[x] = 0.6 + slope * (dx + dy * 0.5)
                            - 0.12 * (-0.5 * (dx * dx + dy * dy) / 0.64).exp();
                    }
                }
            }
            let fits = fit_peaks(&image, &[(-0.12, 32, 32)]);
            assert_eq!(fits.len(), 1, "slope={slope}");
            assert!(fits[0].1.alts.iter().all(|sp| sp.dct[3][0] < 0));
            for y in 0..64 {
                for x in 0..64 {
                    image.plane_row_mut(1, y)[x] += if (x + y) % 2 == 0 { 0.04 } else { -0.04 };
                }
            }
            assert!(fit_peaks(&image, &[(-0.12, 32, 32)]).is_empty());
        }
    }

    #[test]
    fn fitting_cache_preserves_large_coordinate_rounding() {
        let r = 4isize;
        let side = (2 * r + 1) as usize;
        let mut targets: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for dy in -r..=r {
            for dx in -r..=r {
                let value = -0.1 * (-0.5 * (dx * dx + dy * dy) as f32 / 0.81).exp();
                for t in &mut targets {
                    t.push(value);
                }
            }
        }
        assert_eq!(targets[1].len(), side * side);
        let mut kernels = FitKernels::new();
        for x in [32, 16_777_216, crate::encode_image::MAX_DIMENSION - 32, 40] {
            let fit = |cache: &mut FitKernels| {
                dot_alts(
                    x as f32,
                    32.0,
                    4,
                    &targets,
                    x as i32,
                    32,
                    r,
                    -1.0,
                    MIN_EXPLAINED,
                    cache,
                )
                .iter()
                .map(|(sp, _)| template_key(sp))
                .collect::<Vec<_>>()
            };
            assert_eq!(fit(&mut kernels), fit(&mut FitKernels::new()), "x={x}");
        }
    }

    #[test]
    fn compact_fit_screen_rejects_long_edges_in_both_polarities() {
        let r = 4isize;
        let side = (2 * r + 1) as usize;
        let mut targets = [
            vec![0.0; side * side],
            vec![0.0; side * side],
            vec![0.0; side * side],
        ];
        for dy in -r..=r {
            for dx in -1..=1 {
                for channel in [1, 2] {
                    targets[channel][(dy + r) as usize * side + (dx + r) as usize] = -0.08;
                }
            }
        }
        for polarity in [-1.0, 1.0] {
            let targets = targets
                .clone()
                .map(|values| values.into_iter().map(|v| -polarity * v).collect());
            assert!(
                dot_alts(
                    16.0,
                    16.0,
                    4,
                    &targets,
                    16,
                    16,
                    r,
                    polarity,
                    MIN_EXPLAINED,
                    &mut FitKernels::new()
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn row_extrema_match_scalar_windows_at_borders() {
        for w in [1, 2, 3, 7, 9, 67] {
            let src: Vec<f32> = (0..w).map(|i| (i * 73 % 97) as f32).collect();
            let mut out = vec![0f32; w];
            let mut pad = Vec::new();
            for max in [false, true] {
                row_extrema(&src, &mut out, &mut pad, max);
                for (x, &o) in out.iter().enumerate() {
                    let window = &src[x.saturating_sub(OPEN_RADIUS)..=(x + OPEN_RADIUS).min(w - 1)];
                    let expected = window
                        .iter()
                        .copied()
                        .reduce(|a, b| if max { a.max(b) } else { a.min(b) })
                        .unwrap();
                    assert_eq!(o, expected, "w={w} x={x} max={max}");
                }
            }
        }
    }

    #[test]
    fn banded_top_hat_peaks_match_full_image_morphology() {
        // A single band, a band boundary and clipped edges.
        for (w, h) in [(23, 17), (61, 150), (130, 71)] {
            let mut image = Image3F::new(w, h);
            let mut s = 0x9e37_79b9u32;
            for y in 0..h {
                for (x, v) in image.plane_row_mut(1, y).iter_mut().enumerate() {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    let spike = match s % 61 {
                        0 => 0.3,
                        1 => -0.2,
                        _ => 0.0,
                    };
                    *v = 0.002 * x as f32 + 0.001 * y as f32 + (s % 97) as f32 * 1e-4 + spike;
                }
            }
            let y: Vec<f32> = (0..h)
                .flat_map(|r| image.plane_row(1, r).to_vec())
                .collect();
            let pass = |src: &[f32], max: bool, horizontal: bool| {
                (0..w * h)
                    .map(|i| {
                        let (x, yy) = (i % w, i / w);
                        let (axis, size) = if horizontal { (x, w) } else { (yy, h) };
                        (axis.saturating_sub(OPEN_RADIUS)..=(axis + OPEN_RADIUS).min(size - 1))
                            .map(|k| {
                                if horizontal {
                                    src[yy * w + k]
                                } else {
                                    src[k * w + x]
                                }
                            })
                            .reduce(|a, b| if max { a.max(b) } else { a.min(b) })
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            };
            let top_hat = |src: &[f32]| {
                let eroded = pass(&pass(src, false, true), false, false);
                let opened = pass(&pass(&eroded, true, true), true, false);
                src.iter()
                    .zip(&opened)
                    .map(|(v, o)| (v - o).max(0.0))
                    .collect::<Vec<_>>()
            };
            let negated: Vec<f32> = y.iter().map(|v| -v).collect();
            let th = [top_hat(&y), top_hat(&negated)];
            let mut expected = Vec::new();
            for yy in BORDER..h - BORDER {
                for x in BORDER..w - BORDER {
                    for (polarity, th) in th.iter().enumerate() {
                        let v = th[yy * w + x];
                        if v < [MIN_TOP_HAT, MIN_DARK_TOP_HAT][polarity] {
                            continue;
                        }
                        let is_max = (-1isize..=1).all(|dy| {
                            (-1isize..=1).all(|dx| {
                                let o = th
                                    [(yy as isize + dy) as usize * w + (x as isize + dx) as usize];
                                let earlier = dy < 0 || (dy == 0 && dx < 0);
                                (dx == 0 && dy == 0) || !(o > v || (earlier && o == v))
                            })
                        });
                        if is_max {
                            expected.push((if polarity == 0 { v } else { -v }, x, yy));
                        }
                    }
                }
            }
            assert!(expected.iter().any(|p| p.0 > 0.0) && expected.iter().any(|p| p.0 < 0.0));
            for threads in [1, 3] {
                let peaks = top_hat_peaks(
                    &ThreadPool::new(threads),
                    &mut CoderScratch::default(),
                    &image,
                );
                assert_eq!(peaks, expected, "{w}x{h} threads={threads}");
            }
        }
    }

    #[test]
    fn bright_and_dark_candidates_are_symmetric_and_thread_deterministic() {
        let (w, h) = (80, 64);
        let make = |polarity: f32| {
            let mut image = Image3F::new(w, h);
            for c in 0..3 {
                for y in 0..h {
                    image.plane_row_mut(c, y).fill(0.5);
                }
            }
            for (cx, cy) in [(16, 16), (40, 24), (64, 48)] {
                for y in cy - 4..=cy + 4 {
                    for x in cx - 4..=cx + 4 {
                        let d2 = ((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2))
                            as f32;
                        let v = polarity * 0.15 * (-0.5 * d2 / 0.64).exp();
                        for c in [1, 2] {
                            image.plane_row_mut(c, y)[x] += v;
                        }
                    }
                }
            }
            image
        };
        let detect = |image: &Image3F, threads| {
            dot_candidates(
                &ThreadPool::new(threads),
                &mut CoderScratch::default(),
                image,
            )
            .into_iter()
            .flat_map(|c| c.alts)
            .map(|sp| (sp.points[0], template_key(&sp)))
            .collect::<Vec<_>>()
        };
        let bright = detect(&make(1.0), 1);
        let dark = detect(&make(-1.0), 1);
        assert!(!dark.is_empty());
        assert_eq!(dark, detect(&make(-1.0), 4));
        let normalize = |dots: Vec<(Point<i32>, TemplateKey)>| {
            dots.into_iter()
                .map(|(p, (n, y, x, b))| (p, (n.abs(), y, x, b)))
                .collect::<Vec<_>>()
        };
        assert!(bright.iter().all(|(_, key)| key.0 > 0));
        assert!(dark.iter().all(|(_, key)| key.0 < 0));
        assert_eq!(normalize(bright), normalize(dark));
    }

    #[test]
    fn signed_templates_have_opposite_lattice_pixels_and_equal_support() {
        for sigma in [1, 4, 10] {
            for level in [3, 10, 17] {
                let bright = DotTemplate::new((sigma, level, -3, 2));
                let dark = DotTemplate::new((-sigma, level, -3, 2));
                assert_eq!(bright.radius, dark.radius);
                assert_eq!(bright.atlas_radius, dark.atlas_radius);
                assert_eq!(bright.side, dark.side);
                assert_eq!(bright.empty, dark.empty);
                assert_eq!(
                    bright.lattice,
                    dark.lattice.iter().map(|&v| -v).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn quantized_y_matches_exp2_including_extreme_levels() {
        for level in -300..=300 {
            let expected = Y0 * (level as f32 / Y_LEVELS_PER_OCTAVE).exp2();
            assert_eq!(quantized_y(level).to_bits(), expected.to_bits(), "{level}");
        }
    }

    #[test]
    fn atlas_layout_and_position_orders_preserve_added_pixels() {
        let mut splines = Vec::new();
        for (i, (x, y)) in [(16, 17), (42, 22), (31, 47), (57, 53)]
            .into_iter()
            .enumerate()
        {
            let mut sp = QuantizedSpline {
                points: vec![Point::new(x, y)],
                dct: [[0; 32]; 4],
            };
            sp.dct[3][0] = 5 + (i % 2) as i32;
            sp.dct[1][0] = 12;
            sp.dct[0][0] = -2;
            sp.dct[2][0] = 1;
            splines.push(sp);
        }
        let keys: std::collections::BTreeSet<_> = splines.iter().map(template_key).collect();
        let templates = share_templates(keys.into_iter().map(|key| (key, DotTemplate::new(key))));
        let decode = |plan: &DotPatches| {
            let mut pixels = vec![[0i32; 3]; 80 * 80];
            for reference in plan.references() {
                for (x, y) in reference.positions {
                    for j in 0..reference.height {
                        for i in 0..reference.width {
                            let src = (reference.atlas_y + j) * plan.width + reference.atlas_x + i;
                            for c in 0..3 {
                                pixels[(y + j) * 80 + x + i][c] += plan.atlas[c][src];
                            }
                        }
                    }
                }
            }
            pixels
        };
        let expected = decode(&build_dot_patches_with_templates(
            &splines, &templates, false,
        ));
        for trim in [false, true] {
            let mut plan = build_dot_patches_with_templates(&splines, &templates, trim);
            assert_eq!(decode(&plan), expected);
            for band in [None, Some(16), Some(32), Some(64)] {
                plan.sort_positions(band);
                assert_eq!(decode(&plan), expected);
            }
        }
    }
}
