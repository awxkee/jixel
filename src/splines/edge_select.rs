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

//! Bounded edge-amplitude selection on the encoder's native transform layout.
//!
//! Only affected pixels are saved. Each trial is restored by copying those
//! pixels, never by adding the spline back. The model runs the real first-pass
//! quantizers; final layout search, RDOQ and filters still run normally.

use super::edge_fit::{self, EdgePair};
use super::edge_model::{NativeCost, NativeModel, Quantization};
use super::render::{RenderPlan, TiledSpline};
use super::{QUANT_ADJUST, SplineSet};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

// The scalar first-pass model cannot predict the later learned-price RDOQ or
// final entropy contexts exactly. Require a rate reserve before spending any
// spline bits.
const SIDE_COST_MARGIN: f64 = 1.15;
const FIRST_SECTION_RESERVE: f64 = 16.0;
// A local distortion change is credited as this fraction of the local rate
// per relative change: the pair is drawn after EPF and restores edge
// sharpness the pre-filter model cannot see. With DP control points the fits
// are tighter, and 1.0 starts accepting pairs on textured photographs that
// lose (red-rose); requiring no distortion increase at all (the former 1.01
// ratio gate) rejects most pairs that pay on illustrations.
const DISTORTION_ELASTICITY: f64 = 0.75;

struct SavedBlock {
    cell: usize,
    pixels: [[f32; 64]; 3],
}

struct Snapshot {
    blocks: Vec<SavedBlock>,
}

fn read_block(image: &Image3F, cell: usize) -> [[f32; 64]; 3] {
    let (w, h) = (image.xsize(), image.ysize());
    let (x0, y0) = (cell % w.div_ceil(8) * 8, cell / w.div_ceil(8) * 8);
    let n = (w - x0).min(8);
    let mut pixels = [[0.0; 64]; 3];
    for (c, plane) in pixels.iter_mut().enumerate() {
        for y in 0..8 {
            let row = image.plane_row(c, (y0 + y).min(h - 1));
            plane[y * 8..y * 8 + n].copy_from_slice(&row[x0..x0 + n]);
            plane[y * 8 + n..y * 8 + 8].fill(row[x0 + n - 1]);
        }
    }
    pixels
}

fn write_block(image: &mut Image3F, cell: usize, pixels: &[[f32; 64]; 3]) {
    let w = image.xsize();
    let (x0, y0) = (cell % w.div_ceil(8) * 8, cell / w.div_ceil(8) * 8);
    let n = (w - x0).min(8);
    for (c, plane) in pixels.iter().enumerate() {
        for y in 0..(image.ysize() - y0).min(8) {
            image.plane_row_mut(c, y0 + y)[x0..x0 + n].copy_from_slice(&plane[y * 8..y * 8 + n]);
        }
    }
}

/// What the decoder shows of each frame pixel once patches, splines and later
/// frames are applied.
#[derive(Default)]
pub(crate) struct EdgeView {
    /// Squared-error weight of the frame's own colors (alpha, and the share a
    /// blended overlay leaves visible); `None` is 1 everywhere.
    pub(crate) weight: Option<Vec<f32>>,
    /// Pixels a Replace patch overwrites. The decoder draws splines on top
    /// of the patch there, so a pair adds its own color to the patch.
    pub(crate) replaced: Option<Vec<bool>>,
}

/// Below this weight a pixel contributes no edges to detect.
const DETECT_MIN_WEIGHT: f32 = 0.05;
/// Detection blurs once and takes a central gradient: an edge two pixels
/// away from a hidden pixel can be its border rather than image content.
const DETECT_HIDDEN_REACH: usize = 2;

impl EdgeView {
    fn is_empty(&self) -> bool {
        self.weight.is_none() && self.replaced.is_none()
    }

    /// Weight of the frame's reconstruction error, zero under Replace patches.
    fn model_weight(&self, len: usize) -> Option<Vec<f32>> {
        if self.is_empty() {
            return None;
        }
        Some(
            (0..len)
                .map(|i| {
                    if self.replaced.as_ref().is_some_and(|r| r[i]) {
                        0.0
                    } else {
                        self.weight.as_ref().map_or(1.0, |v| v[i])
                    }
                })
                .collect(),
        )
    }

    /// Pixels whose edges detection may follow: visible, and not within
    /// `DETECT_HIDDEN_REACH` of a hidden pixel.
    fn detectable(&self, w: usize, h: usize) -> Option<Vec<bool>> {
        if self.is_empty() {
            return None;
        }
        let hidden: Vec<bool> = (0..w * h)
            .map(|i| {
                self.replaced.as_ref().is_some_and(|r| r[i])
                    || self
                        .weight
                        .as_ref()
                        .is_some_and(|v| v[i] < DETECT_MIN_WEIGHT)
            })
            .collect();
        let reach = DETECT_HIDDEN_REACH;
        // Separable dilation of the hidden set.
        let mut rows = vec![false; w * h];
        for y in 0..h {
            let row = &hidden[y * w..(y + 1) * w];
            let mut last_hidden = None;
            for x in 0..w {
                if row[x] {
                    last_hidden = Some(x);
                }
                let ahead = row[x..(x + reach + 1).min(w)].iter().any(|&v| v);
                rows[y * w + x] = ahead || last_hidden.is_some_and(|l| x - l <= reach);
            }
        }
        Some(
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    !(y.saturating_sub(reach)..(y + reach + 1).min(h)).any(|yy| rows[yy * w + x])
                })
                .collect(),
        )
    }

    /// Weighted energy a pair adds on top of Replace patches.
    fn replaced_energy(&self, ctx: &EncodingContext, plans: &[TiledSpline; 2], w: usize) -> f64 {
        let Some(replaced) = &self.replaced else {
            return 0.0;
        };
        let blocks_w = w.div_ceil(8);
        let mut cells: std::collections::BTreeMap<usize, [[f32; 64]; 3]> = Default::default();
        for plan in plans {
            for tile in &plan.tiles {
                let (x0, y0) = (tile.cell % blocks_w * 8, tile.cell / blocks_w * 8);
                let touches = (0..64).any(|j| {
                    let (x, y) = (x0 + j % 8, y0 + j / 8);
                    x < w && replaced.get(y * w + x).copied().unwrap_or(false)
                });
                if touches {
                    let pixels = cells.entry(tile.cell).or_insert([[0.0; 64]; 3]);
                    plan.draw_tile(tile, pixels, 1.0);
                }
            }
        }
        let mut energy = 0.0f64;
        for (cell, pixels) in cells {
            let (x0, y0) = (cell % blocks_w * 8, cell / blocks_w * 8);
            for j in 0..64 {
                let (x, y) = (x0 + j % 8, y0 + j / 8);
                let i = y * w + x;
                if x >= w || !replaced.get(i).copied().unwrap_or(false) {
                    continue;
                }
                let weight = self.weight.as_ref().map_or(1.0, |v| v[i]);
                for (c, plane) in pixels.iter().enumerate() {
                    energy +=
                        f64::from(weight * ctx.channel_weight(c)) * f64::from(plane[j]).powi(2);
                }
            }
        }
        energy
    }
}

/// Subtracts the pair from the residual. Replaced pixels keep their value:
/// the residual there is never shown, so it must not cost VarDCT bits.
fn draw(image: &mut Image3F, plans: &[TiledSpline; 2], replaced: Option<&[bool]>) {
    let w = image.xsize();
    for plan in plans {
        for tile in &plan.tiles {
            let original = read_block(image, tile.cell);
            let mut pixels = original;
            plan.draw_tile(tile, &mut pixels, -1.0);
            if let Some(replaced) = replaced {
                let (x0, y0) = (tile.cell % w.div_ceil(8) * 8, tile.cell / w.div_ceil(8) * 8);
                for j in 0..64 {
                    let (x, y) = (x0 + j % 8, y0 + j / 8);
                    if x < w && y < image.ysize() && replaced[y * w + x] {
                        for c in 0..3 {
                            pixels[c][j] = original[c][j];
                        }
                    }
                }
            }
            write_block(image, tile.cell, &pixels);
        }
    }
}

impl Snapshot {
    fn new(image: &Image3F, cells: &[usize]) -> Self {
        Self {
            blocks: cells
                .iter()
                .map(|&cell| SavedBlock {
                    cell,
                    pixels: read_block(image, cell),
                })
                .collect(),
        }
    }

    fn restore(&self, image: &mut Image3F) {
        for block in &self.blocks {
            write_block(image, block.cell, &block.pixels);
        }
    }
}

/// Bits of the spline section the frame will carry. Dots leave the set for
/// a patch atlas before it is written, so they cost no section bits here.
fn section_bits(set: Option<&SplineSet>, scratch: &mut CoderScratch) -> f64 {
    let Some(set) = set else { return 0.0 };
    let mut writer = BitWriter::new();
    if set.splines.iter().any(super::dots::is_dot) {
        let lines = SplineSet {
            adjust: set.adjust,
            splines: set
                .splines
                .iter()
                .filter(|sp| !super::dots::is_dot(sp))
                .cloned()
                .collect(),
            dot_encoding: None,
        };
        if lines.splines.is_empty() {
            return 0.0;
        }
        super::write_splines(&lines, scratch, &mut writer);
    } else {
        super::write_splines(set, scratch, &mut writer);
    }
    writer.bits_written() as f64
}

/// Returns a replacement spline set only when at least one pair is accepted.
/// The caller must then rebuild native groups from the changed source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_edges(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    image: &mut Image3F,
    groups: &[DcGroupData],
    quantization: Quantization,
    existing: Option<&SplineSet>,
    view: &EdgeView,
) -> Option<SplineSet> {
    // Pairs are quantized at the section's fixed adjust, which every
    // selected set uses.
    if existing.is_some_and(|set| set.adjust != QUANT_ADJUST) {
        return None;
    }
    let (w, h) = (image.xsize(), image.ysize());
    let detectable = view.detectable(w, h);
    let pairs = edge_fit::find_pairs(ctx, scratch, image, detectable.as_deref());
    if pairs.is_empty() {
        return None;
    }
    let weight = view.model_weight(w * h);
    let replaced = view.replaced.as_deref();
    let model = NativeModel::new(ctx, quantization, groups, w, h, weight.as_deref());
    // Cost of each native transform on the current image. A trial only reads
    // its own transforms, and an accepted pair leaves the costs it measured.
    let mut costs: Vec<Option<NativeCost>> = Vec::new();
    let mut selected = existing.cloned();
    let mut current_bits = section_bits(selected.as_ref(), scratch);
    let mut changed = false;
    for EdgePair { alts } in pairs {
        let mut trials = Vec::new();
        for (index, alternative) in alts.iter().enumerate() {
            let plans = alternative
                .each_ref()
                .map(|spline| RenderPlan::new(ctx, spline, QUANT_ADJUST, w, h).tiled());
            if plans.iter().any(|plan| plan.bounds().is_none()) {
                continue;
            }
            let mut cells: Vec<_> = plans
                .iter()
                .flat_map(|plan| plan.tiles.iter().map(|tile| tile.cell))
                .collect();
            cells.sort_unstable();
            cells.dedup();
            let transforms = model.transforms(&cells);
            if costs.is_empty() {
                costs = vec![None; model.num_transforms()];
            }
            let missing: Vec<usize> = transforms
                .iter()
                .copied()
                .filter(|&t| costs[t].is_none())
                .collect();
            for (t, cost) in missing
                .iter()
                .zip(model.score_transforms(scratch, image, &missing))
            {
                costs[*t] = Some(cost);
            }
            let before = NativeModel::total(transforms.iter().map(|&t| costs[t].unwrap()));
            let saved = Snapshot::new(image, &cells);
            draw(image, &plans, replaced);
            let drawn = model.score_transforms(scratch, image, &transforms);
            saved.restore(image);
            let mut after = NativeModel::total(drawn.iter().copied());
            after.distortion += view.replaced_energy(ctx, &plans, w);
            if !after.distortion.is_finite()
                || !before.distortion.is_finite()
                || before.distortion <= 0.0
            {
                continue;
            }
            let mut proposed = selected.clone().unwrap_or(SplineSet {
                adjust: QUANT_ADJUST,
                splines: Vec::new(),
                dot_encoding: None,
            });
            proposed.splines.extend(alternative.iter().cloned());
            trials.push((index, plans, transforms, drawn, before, after, proposed));
        }
        let bits = ctx
            .thread_pool
            .steal_map(scratch, trials.len(), |i, scratch| {
                section_bits(Some(&trials[i].6), scratch)
            });
        let reserve = if selected.is_none() {
            FIRST_SECTION_RESERVE
        } else {
            0.0
        };
        let mut best = None;
        let mut best_saving = 0.0;
        for (trial, bits) in trials.into_iter().zip(bits) {
            let (before, after) = (trial.4, trial.5);
            let saving = before.rate - after.rate
                + DISTORTION_ELASTICITY
                    * before.rate
                    * (1.0 - after.distortion / before.distortion)
                - SIDE_COST_MARGIN * (bits - current_bits).max(0.0)
                - reserve;
            if saving.is_finite() && saving > best_saving {
                best_saving = saving;
                best = Some((trial, bits));
            }
        }
        if let Some(((_, plans, transforms, drawn, _, _, proposed), bits)) = best {
            draw(image, &plans, replaced);
            for (t, cost) in transforms.into_iter().zip(drawn) {
                costs[t] = Some(cost);
            }
            selected = Some(proposed);
            current_bits = bits;
            changed = true;
        }
    }
    changed.then_some(selected).flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splines::{Point, QuantizedSpline};

    fn pair_plans(ctx: &EncodingContext, w: usize, h: usize) -> [TiledSpline; 2] {
        let mut spline = QuantizedSpline {
            points: vec![Point::new(4, 20), Point::new(60, 20)],
            dct: [[0; 32]; 4],
        };
        spline.dct[1][0] = 11;
        spline.dct[3][0] = 5;
        let mut other = spline.clone();
        other.points = vec![Point::new(4, 22), Point::new(60, 22)];
        other.dct[1][0] = -11;
        [spline, other]
            .each_ref()
            .map(|spline| RenderPlan::new(ctx, spline, QUANT_ADJUST, w, h).tiled())
    }

    #[test]
    fn replaced_pixels_keep_residual_and_charge_the_pair() {
        let ctx = EncodingContext::default();
        let (w, h) = (64, 40);
        let plans = pair_plans(&ctx, w, h);
        let mut replaced = vec![false; w * h];
        for y in 16..24 {
            replaced[y * w + 24..y * w + 40].fill(true);
        }
        let view = EdgeView {
            weight: None,
            replaced: Some(replaced.clone()),
        };
        let mut image = Image3F::new(w, h);
        let original: Vec<_> = (0..3).map(|c| image.plane_data(c).to_vec()).collect();
        draw(&mut image, &plans, view.replaced.as_deref());
        let mut changed_outside = false;
        for c in 0..3 {
            for (i, (&a, &b)) in image.plane_data(c).iter().zip(&original[c]).enumerate() {
                if replaced[i] {
                    assert_eq!(a.to_bits(), b.to_bits());
                } else if a != b {
                    changed_outside = true;
                }
            }
        }
        assert!(changed_outside);
        assert!(view.replaced_energy(&ctx, &plans, w) > 0.0);
        let clear = EdgeView {
            weight: None,
            replaced: Some(vec![false; w * h]),
        };
        assert_eq!(clear.replaced_energy(&ctx, &plans, w), 0.0);
        let weight = view.model_weight(w * h).unwrap();
        assert!(replaced.iter().zip(&weight).all(|(&r, &v)| (v == 0.0) == r));
    }

    #[test]
    fn detection_skips_hidden_pixels_and_their_borders() {
        let (w, h) = (32, 24);
        let mut weight = vec![1.0; w * h];
        for y in 8..12 {
            weight[y * w + 10..y * w + 14].fill(0.0);
        }
        let view = EdgeView {
            weight: Some(weight),
            replaced: None,
        };
        let detectable = view.detectable(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                let near = (8..16).contains(&x) && (6..14).contains(&y);
                assert_eq!(!detectable[y * w + x], near, "({x}, {y})");
            }
        }
        assert!(EdgeView::default().detectable(w, h).is_none());
    }

    #[test]
    fn overlapping_signed_trials_restore_exact_pixels_at_partial_image_edges() {
        let ctx = EncodingContext::default();
        let mut image = Image3F::new(67, 73);
        for c in 0..3 {
            for y in 0..73 {
                for (x, v) in image.plane_row_mut(c, y).iter_mut().enumerate() {
                    *v = 0.01 * c as f32 + ((y * 67 + x) * 19 % 257) as f32 * 0.00037;
                }
            }
        }
        image.plane_row_mut(0, 72)[66] = -0.0;
        let original: Vec<_> = (0..3).map(|c| image.plane_data(c).to_vec()).collect();
        let mut spline = QuantizedSpline {
            points: vec![Point::new(2, 65), Point::new(65, 70)],
            dct: [[0; 32]; 4],
        };
        spline.dct[1][0] = 11;
        spline.dct[3][0] = 5;
        let mut other = spline.clone();
        other.dct[1][0] = -9;
        let plans = [spline, other]
            .each_ref()
            .map(|spline| RenderPlan::new(&ctx, spline, QUANT_ADJUST, 67, 73).tiled());
        let mut cells: Vec<_> = plans
            .iter()
            .flat_map(|plan| plan.tiles.iter().map(|tile| tile.cell))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        let saved = Snapshot::new(&image, &cells);
        for _ in 0..3 {
            draw(&mut image, &plans, None);
            assert_ne!(image.plane_data(1), &original[1]);
            saved.restore(&mut image);
            for (c, plane) in original.iter().enumerate() {
                assert!(
                    image
                        .plane_data(c)
                        .iter()
                        .zip(plane)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }
}
