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

//! Star-like dots as Gaussians drawn from a small template atlas.
//!
//! Candidates ride through the spline RD selection as one-point
//! `QuantizedSpline`s (a one-point spline renders nothing in the format, so
//! the shape is free as a marker): points[0] = centre pixel, dct[1][0] = Y
//! level on a geometric lattice, dct[0][0] = X/Y ratio steps,
//! dct[2][0] = (B-Y)/Y ratio steps, dct[3][0] = sigma lattice index. Selected
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
const MIN_TOP_HAT: f32 = 0.01;
const MIN_ISOLATION: f32 = 5.0;
const Y0: f32 = 0.002;
const Y_LEVELS_PER_OCTAVE: f32 = 2.0;
const X_RATIO_STEP: f32 = 0.02;
const B_RATIO_STEP: f32 = 0.25;
const MAX_RATIO_STEPS: f32 = 4.0;

pub(crate) type TemplateKey = (i32, i32, i32, i32);

const BAND: usize = 64;

/// Running min (or max) over `2 * OPEN_RADIUS + 1` samples along rows
/// (`horizontal`) or columns.
fn running(
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    src: &[f32],
    w: usize,
    h: usize,
    horizontal: bool,
    max: bool,
) -> Vec<f32> {
    let pick = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    pool.steal_map(scratch, h.div_ceil(BAND), |band, _| {
        let rows = band * BAND..((band + 1) * BAND).min(h);
        let mut out = Vec::with_capacity(rows.len() * w);
        for y in rows {
            if horizontal {
                let row = &src[y * w..][..w];
                for i in 0..w {
                    let lo = i.saturating_sub(OPEN_RADIUS);
                    let hi = (i + OPEN_RADIUS).min(w - 1);
                    out.push(row[lo + 1..=hi].iter().fold(row[lo], |v, &s| pick(v, s)));
                }
            } else {
                let lo = y.saturating_sub(OPEN_RADIUS);
                let hi = (y + OPEN_RADIUS).min(h - 1);
                for x in 0..w {
                    out.push((lo + 1..=hi).fold(src[lo * w + x], |v, k| pick(v, src[k * w + x])));
                }
            }
        }
        out
    })
    .concat()
}

/// `Y - opening(Y)`: bright structures narrower than the opening window.
fn top_hat(pool: &ThreadPool, scratch: &mut CoderScratch, xyb: &Image3F) -> Vec<f32> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let mut y = Vec::with_capacity(w * h);
    for row in 0..h {
        y.extend_from_slice(xyb.plane_row(1, row));
    }
    let a = running(pool, scratch, &y, w, h, true, false);
    let b = running(pool, scratch, &a, w, h, false, false);
    let a = running(pool, scratch, &b, w, h, true, true);
    let mut b = running(pool, scratch, &a, w, h, false, true);
    for (o, v) in b.iter_mut().zip(&y) {
        *o = v - *o;
    }
    b
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
    let y = quantized_y(level);
    let x = y * xc as f32 * X_RATIO_STEP;
    let b = y * (1.0 + bc as f32 * B_RATIO_STEP);
    ([x, y, b], n as f32 * sigma_step())
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

/// Gray and colored dots at the sigma lattice widths around `n0`.
#[allow(clippy::too_many_arguments)]
fn dot_alts(
    cx: f32,
    cy: f32,
    n0: i32,
    targets: &[Vec<f32>; 3],
    x: i32,
    y: i32,
    r: isize,
) -> Vec<QuantizedSpline> {
    let side = (2 * r + 1) as usize;
    let (px, py) = (cx.round() as i32, cy.round() as i32);
    let mut alts: Vec<QuantizedSpline> = Vec::new();
    for n in [n0 - 1, n0, n0 + 1] {
        if n < 1 {
            continue;
        }
        let sigma = n as f32 * sigma_step();
        let mut f = vec![0f32; side * side];
        for dy in -r..=r {
            for dx in -r..=r {
                f[((dy + r) as usize) * side + (dx + r) as usize] = blob_weight(
                    (x as isize + dx) as f32 - px as f32,
                    (y as isize + dy) as f32 - py as f32,
                    1.0 / sigma,
                    0.25 * sigma,
                );
            }
        }
        let ff: f32 = f.iter().map(|v| v * v).sum();
        if ff <= 0.0 {
            continue;
        }
        let proj = |t: &[f32]| t.iter().zip(&f).map(|(a, b)| a * b).sum::<f32>() / ff;
        let (cxv, cyv, cbv) = (proj(&targets[0]), proj(&targets[1]), proj(&targets[2]));
        if cyv < Y0 {
            continue;
        }
        let level = (Y_LEVELS_PER_OCTAVE * dirty_log2f(cyv / Y0)).round() as i32;
        let yq = quantized_y(level);
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
        dct[3][0] = n;
        alts.push(QuantizedSpline {
            points: vec![Point { x: px, y: py }],
            dct,
        });
    }
    alts
}

/// Isolated bright peaks of the Y top-hat, brightest first.
pub(super) fn dot_candidates(
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
) -> Vec<Candidate> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    if w <= 2 * BORDER || h <= 2 * BORDER {
        return Vec::new();
    }
    let th = top_hat(pool, scratch, xyb);
    let inner = h - 2 * BORDER;
    let mut peaks = pool
        .steal_map(scratch, inner.div_ceil(BAND), |band, _| {
            let mut peaks = Vec::new();
            for y in BORDER + band * BAND..(BORDER + (band + 1) * BAND).min(h - BORDER) {
                for x in BORDER..w - BORDER {
                    let v = th[y * w + x];
                    if v < MIN_TOP_HAT {
                        continue;
                    }
                    let mut is_max = true;
                    'n: for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let o = th[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                            let earlier = dy < 0 || (dy == 0 && dx < 0);
                            if o > v || (earlier && o == v) {
                                is_max = false;
                                break 'n;
                            }
                        }
                    }
                    if is_max {
                        peaks.push((v, x, y));
                    }
                }
            }
            peaks
        })
        .concat();
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out: Vec<Candidate> = pool
        .steal_map(scratch, peaks.len().div_ceil(PEAK_CHUNK), |chunk, _| {
            let peaks = &peaks[chunk * PEAK_CHUNK..((chunk + 1) * PEAK_CHUNK).min(peaks.len())];
            fit_peaks(xyb, peaks)
        })
        .into_iter()
        .flatten()
        .collect();
    finish_alts(&mut out, w, h);
    out
}

const PEAK_CHUNK: usize = 1024;

/// Fits dot alternatives to `peaks`, skipping peaks in busy surroundings.
fn fit_peaks(xyb: &Image3F, peaks: &[(f32, usize, usize)]) -> Vec<Candidate> {
    let mut out = Vec::new();
    let mut ring = Vec::with_capacity(32);
    'peaks: for &(peak, x, y) in peaks {
        let (xi, yi) = (x as isize, y as isize);
        let px = |c: usize, dx: isize, dy: isize| {
            xyb.plane_row(c, (yi + dy) as usize)[(xi + dx) as usize]
        };
        // Per-channel background: median of the ring; a peak in busy
        // surroundings (texture) is not a dot.
        let mut bg = [0f32; 3];
        // Test Y isolation before fitting chroma: textured images reject most
        // peaks here and need neither of the other two ring medians.
        for c in [1, 0, 2] {
            ring.clear();
            for k in -RING..=RING {
                ring.push(px(c, k, -RING));
                ring.push(px(c, k, RING));
                if k != -RING && k != RING {
                    ring.push(px(c, -RING, k));
                    ring.push(px(c, RING, k));
                }
            }
            bg[c] = median(&mut ring);
            if c == 1 {
                let med = bg[c];
                for v in ring.iter_mut() {
                    *v = (*v - med).abs();
                }
                let mad = median(&mut ring) * 1.4826;
                if peak < MIN_ISOLATION * (mad + 1e-4) {
                    continue 'peaks;
                }
            }
        }
        // Centroid and spread of the Y excess.
        let (mut sw, mut sx, mut sy) = (0f32, 0f32, 0f32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let v = (px(1, dx, dy) - bg[1]).max(0.0);
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
                let v = (px(1, dx, dy) - bg[1]).max(0.0);
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
                    t[((dy + r) as usize) * side + (dx + r) as usize] = px(c, dx, dy) - bg[c];
                }
            }
        }
        let alts = dot_alts(cx, cy, n0, &targets, x as i32, y as i32, r);
        if !alts.is_empty() {
            out.push(Candidate {
                alts,
                bits_factor: 1.0,
                dot: true,
            });
        }
    }
    out
}

/// Every fitted color also competes with gray and with the most common
/// fitted color: the colors templates are most likely to share.
fn finish_alts(out: &mut Vec<Candidate>, w: usize, h: usize) {
    let mut counts: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    for c in out.iter() {
        let sp = &c.alts[c.alts.len() / 2];
        *counts.entry((sp.dct[0][0], sp.dct[2][0])).or_insert(0) += 1;
    }
    let common = counts
        .into_iter()
        .max_by_key(|&(color, n)| (n, std::cmp::Reverse(color)))
        .map_or((0, 0), |(color, _)| color);
    let shared = if common == (0, 0) {
        vec![common]
    } else {
        vec![common, (0, 0)]
    };
    let inside = |sp: &QuantizedSpline| {
        let (x0, y0, side) = dot_box(sp);
        x0 >= 0 && y0 >= 0 && x0 as usize + side <= w && y0 as usize + side <= h
    };
    for c in out.iter_mut() {
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
