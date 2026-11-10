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
//! Periodic-lattice patches: an image whose luma repeats with an integer
//! period along both axes is coded as one period tile in a modular atlas,
//! placed on every lattice node in raster order (constant dictionary offsets,
//! a few bits per placement), plus a VarDCT frame holding what the tile does
//! not explain. Two arms share the same nodes:
//! - kAdd: tile = zero-mean per-pixel mean over the nodes; the frame holds the
//!   image minus the tile, so it keeps DC, drift and noise.
//! - kMul: per channel the node tiles are factored rank-one into a gain per
//!   node and a unit tile; the frame holds the bilinear gain plane, the tile
//!   multiplies it, and the remainder is a reference frame added back whole.

use crate::image::Image3F;
use crate::patches::{MODULAR_PATCH_REF_ID, PATCH_REF_ID, PatchReference};
use crate::quant_weights::INV_DC_QUANT;

/// Shortest and longest period tried, in pixels.
const MIN_PERIOD: usize = 4;
const MAX_PERIOD: usize = 96;
/// Normalized luma autocorrelation a lattice peak must reach on both axes.
const MIN_CORRELATION: f32 = 0.5;
/// A later peak replaces an earlier one only when it is this much higher.
const PEAK_MARGIN: f32 = 0.02;
/// Rows (columns) sampled for the autocorrelation.
const CORRELATION_STRIDE: usize = 4;
/// Nodes sampled for the tile estimate.
const MAX_SAMPLED_NODES: usize = 4096;
/// Power iterations of the rank-one factorization.
const RANK_ONE_ITERATIONS: usize = 3;
/// Unit tiles below this RMS keep their scale.
const MIN_TILE_RMS: f32 = 1e-6;

pub(crate) struct LatticePlan {
    pub(crate) period: (usize, usize),
    /// Lattice nodes in raster order; every tile lies inside the image.
    pub(crate) nodes: Vec<(usize, usize)>,
    /// Modular atlas planes Y, X, B-Y in lattice units, `period` wide and tall.
    pub(crate) add_atlas: [Vec<i32>; 3],
    pub(crate) add_base: Image3F,
    pub(crate) mul_atlas: [Vec<i32>; 3],
    /// Bilinear gain inside the covered area, the source outside it.
    pub(crate) gain: Image3F,
    /// Source minus gain times the decoded unit tile.
    pub(crate) residual: Image3F,
}

impl LatticePlan {
    pub(crate) fn add_references(&self) -> Vec<PatchReference> {
        vec![PatchReference {
            atlas_x: 0,
            atlas_y: 0,
            width: self.period.0,
            height: self.period.1,
            ref_frame: MODULAR_PATCH_REF_ID,
            alpha_only: false,
            add: true,
            mul: false,
            positions: self.nodes.clone(),
        }]
    }

    /// The unit tile multiplies the gain frame, then the residual reference
    /// frame is added over the whole image.
    pub(crate) fn mul_references(&self) -> Vec<PatchReference> {
        vec![
            PatchReference {
                atlas_x: 0,
                atlas_y: 0,
                width: self.period.0,
                height: self.period.1,
                ref_frame: MODULAR_PATCH_REF_ID,
                alpha_only: false,
                add: true,
                mul: true,
                positions: self.nodes.clone(),
            },
            PatchReference {
                atlas_x: 0,
                atlas_y: 0,
                width: self.gain.xsize(),
                height: self.gain.ysize(),
                ref_frame: PATCH_REF_ID,
                alpha_only: false,
                add: true,
                mul: false,
                positions: vec![(0, 0)],
            },
        ]
    }
}

/// Normalized autocorrelation of the zero-mean luma at lags `0..=MAX_PERIOD`
/// along one axis, from every `CORRELATION_STRIDE`-th line.
fn axis_correlation(xyb: &Image3F, horizontal: bool) -> Vec<f32> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let (lines, len) = if horizontal { (h, w) } else { (w, h) };
    let mut line = vec![0.0f32; len];
    let mut mean = 0.0f64;
    let mut count = 0usize;
    let luma = xyb.plane_data(1);
    if horizontal {
        for row in luma.chunks_exact(w).step_by(CORRELATION_STRIDE) {
            for &v in row {
                mean += f64::from(v);
                count += 1;
            }
        }
    } else {
        for l in (0..lines).step_by(CORRELATION_STRIDE) {
            for &v in luma[l..].iter().step_by(w) {
                mean += f64::from(v);
                count += 1;
            }
        }
    }
    let mean = (mean / count.max(1) as f64) as f32;
    let max_lag = MAX_PERIOD.min(len / 2);
    // Dot product, head energy and tail energy for each lag.
    let mut sums = vec![[0.0f64; 3]; max_lag + 1];
    const LANES: usize = 8;
    let grouped_lags = sums.len() / LANES * LANES;
    for l in (0..lines).step_by(CORRELATION_STRIDE) {
        if horizontal {
            for (v, &src) in line.iter_mut().zip(xyb.plane_row(1, l)) {
                *v = src - mean;
            }
        } else {
            for (v, &src) in line.iter_mut().zip(luma[l..].iter().step_by(w)) {
                *v = src - mean;
            }
        }
        // Accumulate adjacent lags together: each lane keeps the original
        // summation order, while the contiguous tail samples can use SIMD.
        let (groups, remainder) = sums.split_at_mut(grouped_lags);
        for (k, group) in (0..).step_by(LANES).zip(groups.as_chunks_mut::<LANES>().0) {
            let common_len = len - (k + LANES - 1);
            let mut d = [0.0f32; LANES];
            let mut ts = [0.0f32; LANES];
            let mut hs = 0.0f32;
            for (&a, b) in line.iter().zip(line[k..].windows(LANES)) {
                hs += a * a;
                for ((d, ts), &b) in d.iter_mut().zip(&mut ts).zip(b) {
                    *d += a * b;
                    *ts += b * b;
                }
            }
            for (j, ((sum, mut d), mut ts)) in group.iter_mut().zip(d).zip(ts).enumerate() {
                let lag = k + j;
                let mut h = hs;
                for (&a, &b) in line[common_len..len - lag]
                    .iter()
                    .zip(&line[common_len + lag..])
                {
                    d += a * b;
                    h += a * a;
                    ts += b * b;
                }
                sum[0] += f64::from(d);
                sum[1] += f64::from(h);
                sum[2] += f64::from(ts);
            }
        }
        for (k, sum) in (grouped_lags..).zip(remainder) {
            let (mut d, mut hs, mut ts) = (0.0f32, 0.0f32, 0.0f32);
            for (&a, &b) in line.iter().zip(&line[k..]) {
                d += a * b;
                hs += a * a;
                ts += b * b;
            }
            sum[0] += f64::from(d);
            sum[1] += f64::from(hs);
            sum[2] += f64::from(ts);
        }
    }
    sums.into_iter()
        .map(|[dot, head, tail]| {
            let norm = (head * tail).sqrt();
            if norm > 0.0 { (dot / norm) as f32 } else { 0.0 }
        })
        .collect()
}

/// First local maximum of the correlation past `MIN_PERIOD` that reaches
/// `MIN_CORRELATION`; a later, clearly higher peak replaces it until the
/// correlation falls below the best one.
fn first_peak(corr: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (k, peak) in (MIN_PERIOD..).zip(corr.windows(3).skip(MIN_PERIOD - 1)) {
        let v = peak[1];
        if v >= MIN_CORRELATION && v >= peak[0] && v >= peak[2] {
            if best.is_none_or(|(_, b)| v > b + PEAK_MARGIN) {
                best = Some((k, v));
            }
            if best.is_some_and(|(_, b)| v <= b) {
                break;
            }
        }
    }
    best.map(|(k, _)| k)
}

/// Lattice units of the three image planes X, Y, B.
const LATTICE: [f32; 3] = [INV_DC_QUANT[0], INV_DC_QUANT[1], INV_DC_QUANT[2]];
const INV_LATTICE: [f32; 3] = [1.0 / LATTICE[0], 1.0 / LATTICE[1], 1.0 / LATTICE[2]];

/// Atlas planes Y, X, B-Y from a tile given in image plane order X, Y, B.
fn atlas_from_tile(tile: &[Vec<f32>; 3]) -> [Vec<i32>; 3] {
    let [x, y, b] = tile;
    let y: Vec<i32> = y.iter().map(|v| (v * LATTICE[1]).round() as i32).collect();
    let x = x.iter().map(|v| (v * LATTICE[0]).round() as i32).collect();
    let by = b
        .iter()
        .zip(&y)
        .map(|(b, y)| (b * LATTICE[2]).round() as i32 - y)
        .collect();
    [y, x, by]
}

/// What the decoder adds or multiplies: the atlas back in image plane order.
fn decoded_tile(atlas: &[Vec<i32>; 3]) -> [Vec<f32>; 3] {
    let [y, x, by] = atlas;
    [
        x.iter().map(|&v| v as f32 * INV_LATTICE[0]).collect(),
        y.iter().map(|&v| v as f32 * INV_LATTICE[1]).collect(),
        by.iter()
            .zip(y)
            .map(|(b, y)| (b + y) as f32 * INV_LATTICE[2])
            .collect(),
    ]
}

/// Evenly spaced nodes for the tile estimates.
fn sampled_nodes(nodes: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let count = nodes.len();
    let take = count.min(MAX_SAMPLED_NODES);
    (0..take).map(|i| nodes[i * count / take]).collect()
}

pub(crate) fn find(xyb: &Image3F) -> Option<LatticePlan> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    if w < 2 * MIN_PERIOD || h < 2 * MIN_PERIOD {
        return None;
    }
    let px = first_peak(&axis_correlation(xyb, true))?;
    let py = first_peak(&axis_correlation(xyb, false))?;
    let (nx, ny) = (w / px, h / py);
    let nodes: Vec<(usize, usize)> = (0..ny)
        .flat_map(|j| (0..nx).map(move |i| (i * px, j * py)))
        .collect();
    let t = px * py;
    let inv_t = (t as f32).recip();
    let inv_px = (px as f32).recip();
    let inv_py = (py as f32).recip();
    let sample = sampled_nodes(&nodes);
    let inv_samples = (sample.len() as f64).recip();
    let tile_of = |c: usize, (x0, y0): (usize, usize), out: &mut [f32]| {
        for (dy, row) in out.chunks_exact_mut(px).enumerate() {
            row.copy_from_slice(&xyb.plane_row(c, y0 + dy)[x0..x0 + px]);
        }
    };

    // Mean tile over the sampled nodes, per channel.
    let mut mean_tile = [vec![0.0f32; t], vec![0.0f32; t], vec![0.0f32; t]];
    let mut buf = vec![0.0f32; t];
    for (c, plane) in mean_tile.iter_mut().enumerate() {
        let mut acc = vec![0.0f64; t];
        for &node in &sample {
            tile_of(c, node, &mut buf);
            for (a, &v) in acc.iter_mut().zip(&buf) {
                *a += f64::from(v);
            }
        }
        for (m, a) in plane.iter_mut().zip(&acc) {
            *m = (a * inv_samples) as f32;
        }
    }

    // kAdd arm: zero-mean tile, base keeps the rest.
    let mut add_tile = mean_tile.clone();
    for plane in add_tile.iter_mut() {
        let m = plane.iter().sum::<f32>() * inv_t;
        for v in plane.iter_mut() {
            *v -= m;
        }
    }
    let add_atlas = atlas_from_tile(&add_tile);
    let add_decoded = decoded_tile(&add_atlas);
    let mut add_base = xyb.clone();
    for &(x0, y0) in &nodes {
        for (c, tile) in add_decoded.iter().enumerate() {
            for (dy, decoded) in tile.chunks_exact(px).enumerate() {
                let row = &mut add_base.plane_row_mut(c, y0 + dy)[x0..x0 + px];
                for (v, &d) in row.iter_mut().zip(decoded) {
                    *v -= d;
                }
            }
        }
    }

    // kMul arm: rank-one node gain x unit tile per channel.
    let mut unit = mean_tile;
    let mut gains = [
        vec![0.0f32; nodes.len()],
        vec![0.0f32; nodes.len()],
        vec![0.0f32; nodes.len()],
    ];
    let mut next = vec![0.0f64; t];
    for (c, (tv, gains)) in unit.iter_mut().zip(&mut gains).enumerate() {
        for _ in 0..RANK_ONE_ITERATIONS {
            let inv_tt = tv.iter().map(|v| v * v).sum::<f32>().max(1e-12).recip();
            next.fill(0.0);
            let mut gg = 0.0f64;
            for &node in &sample {
                tile_of(c, node, &mut buf);
                let g = buf.iter().zip(tv.iter()).map(|(a, b)| a * b).sum::<f32>() * inv_tt;
                gg += f64::from(g * g);
                for (nv, &b) in next.iter_mut().zip(&buf) {
                    *nv += f64::from(g * b);
                }
            }
            let inv_gg = gg.max(1e-12).recip();
            for (v, nv) in tv.iter_mut().zip(&next) {
                *v = (nv * inv_gg) as f32;
            }
        }
        // Unit RMS: a channel near zero (X on a two-color pattern) keeps a
        // well-conditioned tile and leaves its magnitude and sign to the gain.
        let rms = (tv.iter().map(|v| v * v).sum::<f32>() * inv_t).sqrt();
        if rms > MIN_TILE_RMS {
            let inv_rms = rms.recip();
            for v in tv.iter_mut() {
                *v *= inv_rms;
            }
        }
        let inv_tt = tv.iter().map(|v| v * v).sum::<f32>().max(1e-12).recip();
        for (gain, node) in gains.iter_mut().zip(&nodes) {
            tile_of(c, *node, &mut buf);
            *gain = buf.iter().zip(tv.iter()).map(|(a, b)| a * b).sum::<f32>() * inv_tt;
        }
    }
    let mul_atlas = atlas_from_tile(&unit);
    let mul_decoded = decoded_tile(&mul_atlas);
    let mut gain = xyb.clone();
    let mut residual = Image3F::new(w, h);
    let (cw, ch) = (nx * px, ny * py);
    // Horizontal interpolation is shared by every row and channel.
    let x_weights: Vec<_> = (0..cw)
        .map(|x| {
            let fx = ((x as f32 - (px as f32 - 1.0) * 0.5) * inv_px).clamp(0.0, (nx - 1) as f32);
            let i0 = fx.floor() as usize;
            (i0, (i0 + 1).min(nx - 1), fx.fract())
        })
        .collect();
    for (c, (gains, decoded)) in gains.iter().zip(&mul_decoded).enumerate() {
        for y in 0..ch {
            let fy = ((y as f32 - (py as f32 - 1.0) * 0.5) * inv_py).clamp(0.0, (ny - 1) as f32);
            let (j0, wy) = (fy.floor() as usize, fy.fract());
            let j1 = (j0 + 1).min(ny - 1);
            let row = gain.plane_row_mut(c, y);
            let top = &gains[j0 * nx..(j0 + 1) * nx];
            let bottom = &gains[j1 * nx..(j1 + 1) * nx];
            for (g, &(i0, i1, wx)) in row[..cw].iter_mut().zip(&x_weights) {
                let g00 = top[i0];
                let g01 = top[i1];
                let g10 = bottom[i0];
                let g11 = bottom[i1];
                *g =
                    (g00 * (1.0 - wx) + g01 * wx) * (1.0 - wy) + (g10 * (1.0 - wx) + g11 * wx) * wy;
            }
            let dy = y % py;
            let src = xyb.plane_row(c, y);
            let out = residual.plane_row_mut(c, y);
            let tile_row = &decoded[dy * px..(dy + 1) * px];
            for ((out, src), g) in out[..cw]
                .chunks_exact_mut(px)
                .zip(src[..cw].chunks_exact(px))
                .zip(row[..cw].chunks_exact(px))
            {
                for (((out, &src), &g), &unit) in out.iter_mut().zip(src).zip(g).zip(tile_row) {
                    *out = src - g * unit;
                }
            }
        }
    }
    Some(LatticePlan {
        period: (px, py),
        nodes,
        add_atlas,
        add_base,
        mul_atlas,
        gain,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference with independent reductions for each lag.
    fn scalar_axis_correlation(xyb: &Image3F, horizontal: bool) -> Vec<f32> {
        let (w, h) = (xyb.xsize(), xyb.ysize());
        let (lines, len) = if horizontal { (h, w) } else { (w, h) };
        let sample = |l: usize, i: usize| {
            if horizontal {
                xyb.plane_row(1, l)[i]
            } else {
                xyb.plane_row(1, i)[l]
            }
        };
        let mut mean = 0.0f64;
        let mut count = 0usize;
        for l in (0..lines).step_by(CORRELATION_STRIDE) {
            for i in 0..len {
                mean += f64::from(sample(l, i));
                count += 1;
            }
        }
        let mean = (mean / count.max(1) as f64) as f32;
        (0..=MAX_PERIOD.min(len / 2))
            .map(|k| {
                let (mut dot, mut head, mut tail) = (0.0f64, 0.0f64, 0.0f64);
                for l in (0..lines).step_by(CORRELATION_STRIDE) {
                    let (mut d, mut hs, mut ts) = (0.0f32, 0.0f32, 0.0f32);
                    for i in 0..len - k {
                        let a = sample(l, i) - mean;
                        let b = sample(l, i + k) - mean;
                        d += a * b;
                        hs += a * a;
                        ts += b * b;
                    }
                    dot += f64::from(d);
                    head += f64::from(hs);
                    tail += f64::from(ts);
                }
                let norm = (head * tail).sqrt();
                if norm > 0.0 { (dot / norm) as f32 } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn grouped_correlation_matches_scalar_exactly() {
        // Exercise partial groups of lags, the maximum lag and both image axes.
        for len in (1..=33).chain([63, 64, 65, 95, 96, 97, 191, 192, 193, 257]) {
            for (w, h) in [(len, 19), (19, len)] {
                let mut img = Image3F::new(w, h);
                let mut seed = 17u32;
                for y in 0..h {
                    for v in img.plane_row_mut(1, y) {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *v = ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 0.7;
                    }
                }
                for horizontal in [false, true] {
                    let expected = scalar_axis_correlation(&img, horizontal);
                    let actual = axis_correlation(&img, horizontal);
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{w}x{h}, horizontal={horizontal}"
                    );
                }
            }
        }
    }

    fn periodic(w: usize, h: usize, px: usize, py: usize, noise: f32) -> Image3F {
        let mut img = Image3F::new(w, h);
        let mut seed = 7u32;
        for y in 0..h {
            for x in 0..w {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let n = ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * noise;
                let on = ((x % px) < px / 2) != ((y % py) < py / 2);
                let gain = 0.6 + 0.4 * x as f32 / w as f32;
                img.plane_row_mut(1, y)[x] = gain * if on { 0.5 } else { 0.2 } + n;
                img.plane_row_mut(0, y)[x] = 0.01 * if on { 1.0 } else { -1.0 };
                img.plane_row_mut(2, y)[x] = gain * if on { 0.4 } else { 0.3 } + n;
            }
        }
        img
    }

    #[test]
    fn finds_the_period_and_explains_the_pattern() {
        let img = periodic(256, 192, 10, 12, 0.002);
        let plan = find(&img).expect("lattice");
        assert_eq!(plan.period, (10, 12));
        assert_eq!(plan.nodes.len(), 25 * 16);
        assert_eq!(plan.nodes[1], (10, 0));
        assert_eq!(plan.nodes[25], (0, 12));
        // The multiplicative model leaves only the noise.
        let energy: f32 = (0..192)
            .map(|y| {
                plan.residual.plane_row(1, y)[..250]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
            })
            .sum();
        assert!(energy / (192.0 * 250.0) < 1e-5, "residual {energy}");
        // The additive tile has no DC.
        assert!(plan.add_atlas[0].iter().sum::<i32>().abs() <= 120);
    }

    #[test]
    fn smooth_content_has_no_lattice() {
        let mut img = Image3F::new(128, 96);
        for y in 0..96 {
            for x in 0..128 {
                img.plane_row_mut(1, y)[x] = 0.2 + 0.001 * (x + y) as f32;
            }
        }
        assert!(find(&img).is_none());
    }

    #[test]
    fn both_arms_reconstruct_channels_and_uncovered_edges() {
        for (w, h, px, py) in [(67, 53, 6, 10), (128, 96, 8, 8), (257, 199, 32, 48)] {
            let img = periodic(w, h, px, py, 0.002);
            let plan = find(&img).expect("lattice");
            assert_eq!(plan.period, (px, py));
            let add = decoded_tile(&plan.add_atlas);
            let mul = decoded_tile(&plan.mul_atlas);
            let (cw, ch) = (w / px * px, h / py * py);
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        let src = img.plane_row(c, y)[x];
                        let base = plan.add_base.plane_row(c, y)[x];
                        let gain = plan.gain.plane_row(c, y)[x];
                        let residual = plan.residual.plane_row(c, y)[x];
                        if x < cw && y < ch {
                            let i = y % py * px + x % px;
                            assert!((base + add[c][i] - src).abs() <= 6e-8);
                            assert!((gain * mul[c][i] + residual - src).abs() <= 6e-8);
                        } else {
                            assert_eq!(base.to_bits(), src.to_bits());
                            assert_eq!(gain.to_bits(), src.to_bits());
                            assert_eq!(residual, 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reciprocal_normalization_preserves_zero_channels() {
        let mut img = periodic(67, 53, 6, 10, 0.002);
        for c in [0, 2] {
            img.plane_mut(c).as_mut_slice().fill(0.0);
        }
        let plan = find(&img).expect("lattice");
        for c in [0, 2] {
            assert!(decoded_tile(&plan.mul_atlas)[c].iter().all(|&v| v == 0.0));
            for output in [&plan.add_base, &plan.gain, &plan.residual] {
                assert!(output.plane_data(c).iter().all(|&v| v == 0.0));
            }
        }
    }
}
