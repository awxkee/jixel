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

//! Greedy RD acceptance of spline candidates with the encoder's own coefficient
//! rate model on the blocks each spline touches (DCT8 proxy, real quant field).
//! Alternative parametriяations of a candidate compete; the spline side is
//! priced with an adaptive token-cost model refined over two passes.

use super::fit::Candidate;
use super::render::{RenderPlan, TiledSpline};
use super::{
    CTX_DCT, CTX_NUM_POINTS, CTX_POINTS, NUM_SPLINE_CONTEXTS, PixelBox, Point, QUANT_ADJUST,
    QuantizedSpline, SplineSet, render_spline, spline_tokens,
};
use crate::adaptive_quant::dirty_log2f;
use crate::coder_scratch::CoderScratch;
use crate::dct::DctInput;
use crate::encoding_context::EncodingContext;
use crate::entropy::uint_encode;
use crate::image::Image3F;
use std::collections::HashMap;

const START_POSITION_BITS: f32 = 20.0;
/// Spline bits are charged `margin` times: the DCT8 proxy is optimistic where
/// the encoder merges transforms.
const MARGIN_NEAR: (f32, f32) = (4.0, 2.0);
const MARGIN_FAR: (f32, f32) = (8.0, 1.0);
/// Below this modeled saving the section's fixed cost and the encoder's
/// decision noise outweigh the gain.
const MIN_TOTAL_GAIN_BITS: f32 = 256.0;

fn margin(distance: f32) -> f32 {
    let t = ((distance - MARGIN_NEAR.0) / (MARGIN_FAR.0 - MARGIN_NEAR.0)).clamp(0.0, 1.0);
    MARGIN_NEAR.1 + t * (MARGIN_FAR.1 - MARGIN_NEAR.1)
}

/// Per-context symbol histograms of the spline token stream with additive smoothing.
struct CostModel {
    counts: [HashMap<u32, f32>; NUM_SPLINE_CONTEXTS],
    totals: [f32; NUM_SPLINE_CONTEXTS],
}

impl CostModel {
    fn prior(weight: f32) -> Self {
        let mut model = CostModel {
            counts: Default::default(),
            totals: [0.0; NUM_SPLINE_CONTEXTS],
        };
        model.add(CTX_DCT, 0, 60.0 * weight);
        for v in 1..24 {
            model.add(CTX_DCT, v, weight);
        }
        for v in 0..32 {
            model.add(CTX_POINTS, v, weight);
        }
        for v in 0..24 {
            model.add(CTX_NUM_POINTS, v, weight);
        }
        model
    }

    fn add(&mut self, ctx: u32, value: u32, weight: f32) {
        let (symbol, _, _) = uint_encode(value);
        *self.counts[ctx as usize].entry(symbol).or_insert(0.0) += weight;
        self.totals[ctx as usize] += weight;
    }

    fn bits(&self, ctx: u32, value: u32) -> f32 {
        let (symbol, nbits, _) = uint_encode(value);
        let count = self.counts[ctx as usize]
            .get(&symbol)
            .copied()
            .unwrap_or(0.0);
        let symbols = self.counts[ctx as usize].len() as f32 + 1.0;
        let p = (count + 0.3) / (self.totals[ctx as usize] + 0.3 * symbols);
        nbits as f32 - dirty_log2f(p)
    }

    fn spline_bits(&self, sp: &QuantizedSpline) -> f32 {
        START_POSITION_BITS
            + spline_tokens(sp)
                .iter()
                .map(|&(c, v)| self.bits(c, v))
                .sum::<f32>()
    }
}

pub(super) struct BlockModel<'a> {
    ctx: &'a EncodingContext,
    inv: [&'a [f32; 64]; 3],
    qm: [f32; 3],
    distance: f32,
}

impl<'a> BlockModel<'a> {
    pub(super) fn new(ctx: &'a EncodingContext, distance: f32) -> Self {
        let matrices = crate::quant_weights::DequantMatrices::new(distance);
        BlockModel {
            ctx,
            inv: [
                matrices.inv_matrix(0),
                matrices.inv_matrix(1),
                matrices.inv_matrix(2),
            ],
            qm: [1.0, 1.0, ctx.b_qm_mul()],
            distance,
        }
    }

    /// Model (distortion, bits) of one 8x8 block under the DCT8 proxy.
    fn cost(&self, img: &Image3F, bx: usize, by: usize, qac: f32) -> (f32, f32) {
        let (w, h) = (img.xsize(), img.ysize());
        let (mut d_total, mut r_total) = (0f32, 0f32);
        for c in 0..3 {
            let mut coef = [0f32; 64];
            let (x0, y0) = (bx * 8, by * 8);
            if x0 + 8 <= w && y0 + 8 <= h {
                let input = DctInput::new(&img.plane_data(c)[y0 * w + x0..], w);
                (self.ctx.dct8x8)(input, &mut coef);
            } else {
                let mut px = [0f32; 64];
                for (y, out) in px.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let row = img.plane_row(c, (y0 + y).min(h - 1));
                    let src = &row[x0..w.min(x0 + 8)];
                    out[..src.len()].copy_from_slice(src);
                    out[src.len()..].fill(*src.last().unwrap());
                }
                (self.ctx.dct8x8)(DctInput::from_flat(&px), &mut coef);
            }
            let (d, r) = crate::inflated_cost::channel_rd(
                self.ctx.sse_and_rate,
                self.ctx.rate_log2_lut,
                &coef,
                self.inv[c],
                c,
                qac,
                self.qm[c],
                self.distance,
                1,
                1,
            );
            d_total += self.ctx.channel_weight(c) * d;
            r_total += r;
        }
        (d_total, r_total)
    }

    fn pixels_cost(&self, pixels: &[[f32; 64]; 3], qac: f32) -> (f32, f32) {
        let (mut distortion, mut rate) = (0.0, 0.0);
        for (c, plane) in pixels.iter().enumerate() {
            let mut coef = [0.0; 64];
            (self.ctx.dct8x8)(DctInput::from_flat(plane), &mut coef);
            let (d, r) = crate::inflated_cost::channel_rd(
                self.ctx.sse_and_rate,
                self.ctx.rate_log2_lut,
                &coef,
                self.inv[c],
                c,
                qac,
                self.qm[c],
                self.distance,
                1,
                1,
            );
            distortion += self.ctx.channel_weight(c) * d;
            rate += r;
        }
        (distortion, rate)
    }
}

struct TrialBlock {
    cell: usize,
    pixels: [[f32; 64]; 3],
}

/// A trial owns only affected blocks. Coordinates and sample accumulation stay
/// identical to full-image rendering; different alternatives share no writes.
struct Trial {
    blocks: Vec<TrialBlock>,
}

impl Trial {
    fn new(current: &Image3F, plans: &[&TiledSpline]) -> Self {
        let mut cells: Vec<_> = plans
            .iter()
            .flat_map(|p| p.tiles.iter().map(|t| t.cell))
            .collect();
        if plans.len() > 1 {
            cells.sort_unstable();
            cells.dedup();
        }
        let (w, h) = (current.xsize(), current.ysize());
        let blocks_w = w.div_ceil(8);
        let blocks = cells
            .into_iter()
            .map(|cell| {
                let (x0, y0) = (cell % blocks_w * 8, cell / blocks_w * 8);
                let width = (w - x0).min(8);
                let mut pixels = [[0.0; 64]; 3];
                for (c, plane) in pixels.iter_mut().enumerate() {
                    for (y, row) in plane.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        row[..width].copy_from_slice(
                            &current.plane_row(c, (y0 + y).min(h - 1))[x0..x0 + width],
                        );
                        let edge = row[width - 1];
                        row[width..].fill(edge);
                    }
                }
                TrialBlock { cell, pixels }
            })
            .collect();
        Self { blocks }
    }

    fn draw(&mut self, plan: &TiledSpline, sign: f32) {
        let mut blocks = self.blocks.iter_mut();
        for tile in &plan.tiles {
            let block = blocks.find(|block| block.cell == tile.cell).unwrap();
            plan.draw_tile(tile, &mut block.pixels, sign);
        }
    }

    /// Reuse a prepared residual on a subset of this trial's sorted blocks.
    fn copy_blocks_from(&mut self, source: &Self) {
        let mut blocks = self.blocks.iter_mut();
        for source in &source.blocks {
            let block = blocks.find(|block| block.cell == source.cell).unwrap();
            block.pixels = source.pixels;
        }
    }

    fn delta(
        &mut self,
        model: &BlockModel,
        current: &Image3F,
        quant_field: &[f32],
        cache: &[Option<(f32, f32)>],
        forbidden: Option<&[bool]>,
    ) -> Option<f32> {
        let (w, h) = (current.xsize(), current.ysize());
        let blocks_w = w.div_ceil(8);
        let mut delta = 0.0;
        for block in &mut self.blocks {
            let (bx, by) = (block.cell % blocks_w, block.cell / blocks_w);
            let (x0, y0) = (bx * 8, by * 8);
            let (width, height) = ((w - x0).min(8), (h - y0).min(8));
            let touched = (0..height).any(|y| {
                (0..width).any(|x| {
                    let at = y * 8 + x;
                    (current.plane_row(1, y0 + y)[x0 + x] - block.pixels[1][at]).abs() > 1e-5
                        || (current.plane_row(0, y0 + y)[x0 + x] - block.pixels[0][at]).abs() > 1e-6
                        || (current.plane_row(2, y0 + y)[x0 + x] - block.pixels[2][at]).abs() > 1e-5
                })
            });
            if !touched {
                continue;
            }
            if forbidden.is_some_and(|mask| mask[block.cell]) {
                return None;
            }
            // Replicate the rendered edge, rather than the original edge used
            // when loading the trial. Interior blocks need no padding work.
            if width < 8 || height < 8 {
                for plane in &mut block.pixels {
                    for y in 0..height {
                        let edge = plane[y * 8 + width - 1];
                        plane[y * 8 + width..y * 8 + 8].fill(edge);
                    }
                    for y in height..8 {
                        plane.copy_within((height - 1) * 8..height * 8, y * 8);
                    }
                }
            }
            let qac = quant_field[block.cell];
            let (d0, r0) = cache[block.cell].unwrap_or_else(|| model.cost(current, bx, by, qac));
            let (d1, r1) = model.pixels_cost(&block.pixels, qac);
            delta += (d1 - d0) + crate::ac_strategy::RD_LAMBDA * (r1 - r0);
        }
        Some(delta)
    }

    fn commit(&self, image: &mut Image3F) {
        let (w, h) = (image.xsize(), image.ysize());
        let blocks_w = w.div_ceil(8);
        for block in &self.blocks {
            let (x0, y0) = (block.cell % blocks_w * 8, block.cell / blocks_w * 8);
            let (width, height) = ((w - x0).min(8), (h - y0).min(8));
            for (c, plane) in block.pixels.iter().enumerate() {
                for y in 0..height {
                    image.plane_row_mut(c, y0 + y)[x0..x0 + width]
                        .copy_from_slice(&plane[y * 8..y * 8 + width]);
                }
            }
        }
    }
}

fn cache_blocks(
    model: &BlockModel,
    current: &Image3F,
    quant_field: &[f32],
    cache: &mut [Option<(f32, f32)>],
    plan: &TiledSpline,
) {
    let blocks_w = current.xsize().div_ceil(8);
    for tile in &plan.tiles {
        cache[tile.cell].get_or_insert_with(|| {
            model.cost(
                current,
                tile.cell % blocks_w,
                tile.cell / blocks_w,
                quant_field[tile.cell],
            )
        });
    }
}

fn block_touched(a: &Image3F, b: &Image3F, bx: usize, by: usize) -> bool {
    let (w, h) = (a.xsize(), a.ysize());
    for y in by * 8..(by * 8 + 8).min(h) {
        let (ay, ty) = (a.plane_row(1, y), b.plane_row(1, y));
        let (ax, tx) = (a.plane_row(0, y), b.plane_row(0, y));
        let (ab, tb) = (a.plane_row(2, y), b.plane_row(2, y));
        let xs = bx * 8..(bx * 8 + 8).min(w);
        if ay[xs.clone()]
            .iter()
            .zip(&ty[xs.clone()])
            .zip(ax[xs.clone()].iter().zip(&tx[xs.clone()]))
            .zip(ab[xs.clone()].iter().zip(&tb[xs]))
            .any(|(((&ay, &ty), (&ax, &tx)), (&ab, &tb))| {
                (ay - ty).abs() > 1e-5 || (ax - tx).abs() > 1e-6 || (ab - tb).abs() > 1e-5
            })
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
fn trial_cost(
    model: &BlockModel,
    current: &Image3F,
    trial: &Image3F,
    quant_field: &[f32],
    block_cache: &mut [Option<(f32, f32)>],
    b: PixelBox,
    forbidden: Option<&[bool]>,
) -> Option<f32> {
    let blocks_w = current.xsize().div_ceil(8);
    let mut delta = 0.0;
    for by in b.1 / 8..=b.3 / 8 {
        for bx in b.0 / 8..=b.2 / 8 {
            if !block_touched(current, trial, bx, by) {
                continue;
            }
            let cell = by * blocks_w + bx;
            if forbidden.is_some_and(|mask| mask[cell]) {
                return None;
            }
            let qac = quant_field[cell];
            let (d0, r0) =
                *block_cache[cell].get_or_insert_with(|| model.cost(current, bx, by, qac));
            let (d1, r1) = model.cost(trial, bx, by, qac);
            delta += (d1 - d0) + crate::ac_strategy::RD_LAMBDA * (r1 - r0);
        }
    }
    Some(delta)
}

/// Simplify accepted curves with joint thinning, single deletion and pair
/// collapse. Every proposal competes on the rendered residual and token cost;
/// color and width coefficients stay fixed, so no image fit is rebuilt.
fn prune_selected(
    model: &BlockModel,
    scratch: &mut CoderScratch,
    current: &mut Image3F,
    quant_field: &[f32],
    kept: &mut [QuantizedSpline],
    prices: &CostModel,
    forbidden: Option<&[bool]>,
) {
    let (w, h) = (current.xsize(), current.ysize());
    let blocks_w = w.div_ceil(8);
    // Cap speculation at four proposals: larger batches waste too much work
    // after early acceptances. One worker retains the serial schedule.
    let batch_size = model.ctx.thread_pool.num_threads().min(4);
    let mut cache = vec![None; blocks_w * h.div_ceil(8)];
    let mut try_points = |sp: &mut QuantizedSpline,
                          incumbent: &mut TiledSpline,
                          restored: &mut Option<Trial>,
                          points: Vec<Vec<Point<i32>>>|
     -> Option<usize> {
        let old = incumbent.bounds()?;
        let proposals = model
            .ctx
            .thread_pool
            .steal_map(scratch, points.len(), |index, _| {
                let points = &points[index];
                if points.len() < 2
                    || points.len() >= sp.points.len()
                    || points.array_windows::<2>().any(|p| p[0] == p[1])
                    || points
                        .iter()
                        .any(|p| p.x < 0 || p.y < 0 || p.x >= w as i32 || p.y >= h as i32)
                {
                    return None;
                }
                let simpler = QuantizedSpline {
                    points: points.clone(),
                    dct: sp.dct,
                };
                let bits = prices.spline_bits(&simpler) - prices.spline_bits(sp);
                let plan = RenderPlan::new(model.ctx, &simpler, QUANT_ADJUST, w, h).tiled();
                plan.bounds()?;
                Some((simpler, plan, bits))
            });
        if proposals.iter().all(Option::is_none) {
            return None;
        }
        cache_blocks(model, current, quant_field, &mut cache, incumbent);
        for (_, plan, _) in proposals.iter().flatten() {
            cache_blocks(model, current, quant_field, &mut cache, plan);
        }
        // Rejected proposals leave both the image and incumbent unchanged.
        // Preserve the exact sample-by-sample addition once and share its
        // rounded pixels until an accepted proposal changes the residual.
        let restored_trial = restored.get_or_insert_with(|| {
            let mut trial = Trial::new(current, &[incumbent]);
            trial.draw(incumbent, 1.0);
            trial
        });
        let trials = model
            .ctx
            .thread_pool
            .steal_map(scratch, proposals.len(), |i, _| {
                let (_, proposal, bits) = proposals[i].as_ref()?;
                let mut trial = Trial::new(current, &[incumbent, proposal]);
                trial.copy_blocks_from(restored_trial);
                trial.draw(proposal, -1.0);
                let delta = trial.delta(model, current, quant_field, &cache, forbidden)?;
                (delta + crate::ac_strategy::RD_LAMBDA * margin(model.distance) * bits < 0.0)
                    .then_some(trial)
            });
        // Greedy pruning accepts the first improving proposal, not the best
        // of the batch. Later trials used stale geometry after this commit.
        for (index, (trial, proposal)) in trials.into_iter().zip(proposals).enumerate() {
            let Some(trial) = trial else { continue };
            let (simpler, proposal, _) = proposal.unwrap();
            let new = proposal.bounds().unwrap();
            let b = (
                old.0.min(new.0),
                old.1.min(new.1),
                old.2.max(new.2),
                old.3.max(new.3),
            );
            trial.commit(current);
            *sp = simpler;
            *incumbent = proposal;
            *restored = None;
            for by in b.1 / 8..=b.3 / 8 {
                cache[by * blocks_w..][b.0 / 8..=b.2 / 8].fill(None);
            }
            return Some(index);
        }
        None
    };
    for sp in kept {
        let mut incumbent = RenderPlan::new(model.ctx, sp, QUANT_ADJUST, w, h).tiled();
        let mut restored = None;
        // Deleting only one regularly spaced knot can make its double deltas
        // expensive. These proposals can cross that barrier in one decision.
        for phase in 0..2 {
            if sp.points.len() < 5 {
                break;
            }
            let points = sp
                .points
                .iter()
                .enumerate()
                .filter(|&(i, _)| i == 0 || i + 1 == sp.points.len() || i % 2 == phase)
                .map(|(_, &p)| p)
                .collect();
            try_points(sp, &mut incumbent, &mut restored, vec![points]);
        }
        for merge in [false, true] {
            let mut i = 1;
            while i + 1 < sp.points.len() {
                let mut proposals = Vec::with_capacity(batch_size);
                let mut stopped = false;
                for index in i..(i + batch_size).min(sp.points.len() - 1) {
                    let points = if merge {
                        let Some(points) =
                            super::fit::collapse_control_pair(model.ctx, &sp.points, index)
                        else {
                            stopped = true;
                            break;
                        };
                        points
                    } else {
                        let mut points = sp.points.clone();
                        points.remove(index);
                        points
                    };
                    proposals.push(points);
                }
                let count = proposals.len();
                if count == 0 {
                    break;
                }
                if let Some(accepted) = try_points(sp, &mut incumbent, &mut restored, proposals) {
                    // Earlier proposals failed; resume at the accepted index
                    // because deletion shifted the following control into it.
                    i += accepted;
                } else if stopped {
                    break;
                } else {
                    i += count;
                }
            }
        }
    }
}

const PRETEST_PAD: i32 = 24;
/// The pre-test sees unrefined geometry, so it lets through splines whose
/// VarDCT saving covers only this share of their (prior-priced) bits.
const PRETEST_BITS_SHARE: f32 = 0.25;

/// Independent, state-free RD pre-test of one spline against the untouched
/// image, on a block-aligned local copy (so it can run in parallel).
pub(super) fn pretest(
    model: &BlockModel,
    xyb: &Image3F,
    quant_field: &[f32],
    spline: &QuantizedSpline,
) -> bool {
    let (w, h) = (xyb.xsize() as i32, xyb.ysize() as i32);
    let xs = spline.points.iter().map(|p| p.x);
    let ys = spline.points.iter().map(|p| p.y);
    let x0 = ((xs.clone().min().unwrap() - PRETEST_PAD).max(0) / 8 * 8) as usize;
    let y0 = ((ys.clone().min().unwrap() - PRETEST_PAD).max(0) / 8 * 8) as usize;
    let x1 = (xs.max().unwrap() + PRETEST_PAD).min(w - 1) as usize;
    let y1 = (ys.max().unwrap() + PRETEST_PAD).min(h - 1) as usize;
    // extend to whole blocks, or to the image edge
    let x1 = ((x1 / 8 + 1) * 8).min(w as usize);
    let y1 = ((y1 / 8 + 1) * 8).min(h as usize);
    let mut local = Image3F::new(x1 - x0, y1 - y0);
    for c in 0..3 {
        for y in y0..y1 {
            local
                .plane_row_mut(c, y - y0)
                .copy_from_slice(&xyb.plane_row(c, y)[x0..x1]);
        }
    }
    let mut trial = local.clone();
    let shifted = QuantizedSpline {
        points: spline
            .points
            .iter()
            .map(|p| Point::new(p.x - x0 as i32, p.y - y0 as i32))
            .collect(),
        dct: spline.dct,
    };
    let Some(b) = render_spline(model.ctx, &shifted, QUANT_ADJUST, &mut trial, -1.0) else {
        return false;
    };
    let blocks_w = xyb.xsize().div_ceil(8);
    let mut dj = 0f32;
    for by in b.1 / 8..=b.3 / 8 {
        for bx in b.0 / 8..=b.2 / 8 {
            if !block_touched(&local, &trial, bx, by) {
                continue;
            }
            let qac = quant_field[(y0 / 8 + by) * blocks_w + x0 / 8 + bx];
            let (d0, r0) = model.cost(&local, bx, by, qac);
            let (d1, r1) = model.cost(&trial, bx, by, qac);
            dj += (d1 - d0) + crate::ac_strategy::RD_LAMBDA * (r1 - r0);
        }
    }
    let bits = CostModel::prior(1.0).spline_bits(spline);
    dj + crate::ac_strategy::RD_LAMBDA * PRETEST_BITS_SHARE * margin(model.distance) * bits < 0.0
}

/// Subtracts the accepted splines from `xyb` and returns them.
pub(super) fn rd_select(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    xyb: &mut Image3F,
    quant_field: &[f32],
    candidates: &[Candidate],
    forbidden: Option<&[bool]>,
) -> Option<SplineSet> {
    let lambda = crate::ac_strategy::RD_LAMBDA;
    let margin = margin(distance);
    let model = BlockModel::new(ctx, distance);
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let blocks_w = w.div_ceil(8);
    let prepared = ctx
        .thread_pool
        .steal_map(scratch, candidates.len(), |i, _| {
            candidates[i]
                .alts
                .iter()
                .map(|alt| RenderPlan::new(model.ctx, alt, QUANT_ADJUST, w, h).tiled())
                .collect::<Vec<_>>()
        });
    struct Evaluation {
        delta: f32,
        trial: Trial,
    }
    let evaluate = |plan: &TiledSpline, current: &Image3F, cache: &[Option<(f32, f32)>]| {
        plan.bounds()?;
        let mut trial = Trial::new(current, &[plan]);
        trial.draw(plan, -1.0);
        let delta = trial.delta(&model, current, quant_field, cache, forbidden)?;
        Some(Evaluation { delta, trial })
    };
    let invalidate = |cache: &mut [Option<(f32, f32)>], b: PixelBox| {
        for by in b.1 / 8..=b.3 / 8 {
            cache[by * blocks_w..][b.0 / 8..=b.2 / 8].fill(None);
        }
    };

    // Candidates still commit in their original greedy order. Only alternatives
    // reading the same residual run concurrently. Keep the winning trial rather
    // than rendering it again, and retain preparation across repricing passes.
    let mut prices = CostModel::prior(1.0);
    let mut current = xyb.clone();
    let mut block_cache = vec![None; blocks_w * h.div_ceil(8)];
    let mut deltas: Vec<Vec<Option<(f32, PixelBox)>>> = Vec::with_capacity(candidates.len());
    let mut first_pass = Vec::new();
    let mut first_choices = Vec::with_capacity(candidates.len());
    for (candidate, plans) in candidates.iter().zip(&prepared) {
        for plan in plans {
            cache_blocks(&model, &current, quant_field, &mut block_cache, plan);
        }
        let mut evaluations = ctx.thread_pool.steal_map(scratch, plans.len(), |i, _| {
            evaluate(&plans[i], &current, &block_cache)
        });
        let mut row = vec![None; candidate.alts.len()];
        let mut best: Option<(f32, usize, PixelBox)> = None;
        for (index, evaluation) in evaluations.iter().enumerate() {
            let Some(evaluation) = evaluation else {
                continue;
            };
            let b = plans[index].bounds().unwrap();
            let dj = evaluation.delta;
            row[index] = Some((dj, b));
            let j = dj
                + lambda
                    * margin
                    * candidate.bits_factor
                    * prices.spline_bits(&candidate.alts[index]);
            if best.is_none_or(|(bj, _, _)| j < bj) {
                best = Some((j, index, b));
            }
        }
        let accepted = best.filter(|&(j, _, _)| j < 0.0);
        first_choices.push(accepted.map(|(_, index, _)| index));
        if let Some((_, index, b)) = accepted {
            evaluations[index]
                .take()
                .unwrap()
                .trial
                .commit(&mut current);
            invalidate(&mut block_cache, b);
            first_pass.push(candidate.alts[index].clone());
        }
        deltas.push(row);
    }
    for c in 0..3 {
        for y in 0..h {
            current
                .plane_row_mut(c, y)
                .copy_from_slice(xyb.plane_row(c, y));
        }
    }
    block_cache.fill(None);
    let mut changed = vec![false; block_cache.len()];
    prices = CostModel::prior(0.1);
    for sp in &first_pass {
        for (c, v) in spline_tokens(sp) {
            prices.add(c, v, 1.0);
        }
    }
    let mut kept = Vec::new();
    let mut gain_bits = 0.0;
    for (((candidate, plans), row), &first) in candidates
        .iter()
        .zip(&prepared)
        .zip(&deltas)
        .zip(&first_choices)
    {
        let recompute: Vec<_> = row
            .iter()
            .map(|cached| {
                cached.is_some_and(|(_, b)| {
                    (b.1 / 8..=b.3 / 8).any(|by| {
                        changed[by * blocks_w..][b.0 / 8..=b.2 / 8]
                            .iter()
                            .any(|&v| v)
                    })
                })
            })
            .collect();
        for (plan, &needed) in plans.iter().zip(&recompute) {
            if needed {
                cache_blocks(&model, &current, quant_field, &mut block_cache, plan);
            }
        }
        let mut evaluations = if recompute.iter().any(|&needed| needed) {
            ctx.thread_pool.steal_map(scratch, plans.len(), |i, _| {
                recompute[i]
                    .then(|| evaluate(&plans[i], &current, &block_cache))
                    .flatten()
            })
        } else {
            std::iter::repeat_with(|| None).take(plans.len()).collect()
        };
        let mut best: Option<(f32, usize, PixelBox)> = None;
        for (index, (alt, cached)) in candidate.alts.iter().zip(row).enumerate() {
            let Some((mut dj, b)) = *cached else { continue };
            if recompute[index] {
                let Some(evaluation) = &evaluations[index] else {
                    continue;
                };
                dj = evaluation.delta;
            }
            let j = dj + lambda * margin * candidate.bits_factor * prices.spline_bits(alt);
            if best.is_none_or(|(bj, _, _)| j < bj) {
                best = Some((j, index, b));
            }
        }
        let accepted = best.filter(|&(j, _, _)| j < 0.0);
        let choice = accepted.map(|(_, index, _)| index);
        if first != choice {
            for index in first.into_iter().chain(choice) {
                let (_, b) = row[index].unwrap();
                for by in b.1 / 8..=b.3 / 8 {
                    changed[by * blocks_w..][b.0 / 8..=b.2 / 8].fill(true);
                }
            }
        }
        if let Some((j, index, b)) = accepted {
            let trial = if let Some(evaluation) = evaluations[index].take() {
                evaluation.trial
            } else {
                let mut trial = Trial::new(&current, &[&plans[index]]);
                trial.draw(&plans[index], -1.0);
                trial
            };
            trial.commit(&mut current);
            invalidate(&mut block_cache, b);
            gain_bits -= j / lambda;
            kept.push(candidate.alts[index].clone());
        }
    }
    if kept.is_empty() || gain_bits < MIN_TOTAL_GAIN_BITS {
        return None;
    }
    drop(prepared);
    prune_selected(
        &model,
        scratch,
        &mut current,
        quant_field,
        &mut kept,
        &prices,
        forbidden,
    );
    *xyb = current;
    Some(SplineSet {
        adjust: QUANT_ADJUST,
        splines: kept,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Speed, xyb::XybMatrix};

    #[test]
    fn sparse_trials_match_full_render_and_rd_at_image_edges() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let (w, h) = (137, 91);
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let model = BlockModel::new(&ctx, 3.0);
        let mut current = Image3F::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for (x, value) in current.plane_row_mut(c, y).iter_mut().enumerate() {
                    *value = ((x * 19 + y * 31 + c * 7) % 83) as f32 * 0.003;
                }
            }
        }
        let mut first = QuantizedSpline {
            points: vec![Point::new(0, 1), Point::new(33, 49), Point::new(136, 89)],
            dct: [[0; 32]; 4],
        };
        first.dct[1][0] = 7;
        first.dct[2][1] = -5;
        first.dct[3][0] = 4;
        first.dct[3][2] = 1;
        let mut second = first.clone();
        second.points[1] = Point::new(74, 31);
        second.dct[0][0] = -13;
        let a = RenderPlan::new(&kernels, &first, QUANT_ADJUST, w, h).tiled();
        let b = RenderPlan::new(&kernels, &second, QUANT_ADJUST, w, h).tiled();
        let mut restored = Trial::new(&current, &[&a]);
        restored.draw(&a, 1.0);
        let mut trial = Trial::new(&current, &[&a, &b]);
        trial.copy_blocks_from(&restored);
        trial.draw(&b, -1.0);
        let mut full = current.clone();
        render_spline(&kernels, &first, QUANT_ADJUST, &mut full, 1.0);
        render_spline(&kernels, &second, QUANT_ADJUST, &mut full, -1.0);
        let quant = vec![5.0; w.div_ceil(8) * h.div_ceil(8)];
        let mut cache = vec![None; quant.len()];
        let expected = trial_cost(
            &model,
            &current,
            &full,
            &quant,
            &mut cache,
            (0, 0, w - 1, h - 1),
            None,
        );
        assert_eq!(
            trial.delta(&model, &current, &quant, &cache, None),
            expected
        );
        let mut committed = current.clone();
        trial.commit(&mut committed);
        for c in 0..3 {
            assert!(
                full.plane_data(c)
                    .iter()
                    .zip(committed.plane_data(c))
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        let forbidden = vec![true; quant.len()];
        assert!(
            trial
                .delta(&model, &current, &quant, &cache, Some(&forbidden))
                .is_none()
        );
    }

    #[test]
    fn alternatives_commit_identically_with_one_or_four_workers() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let (w, h) = (263, 193);
        let mut image = Image3F::new(w, h);
        let mut candidates = Vec::new();
        for i in 0..6 {
            let mut sp = QuantizedSpline {
                points: vec![
                    Point::new(4, 15 + i * 28),
                    Point::new(128, 30 + i * 24),
                    Point::new(260, 15 + i * 28),
                ],
                dct: [[0; 32]; 4],
            };
            sp.dct[1][0] = 8;
            sp.dct[3][0] = 4;
            render_spline(&kernels, &sp, QUANT_ADJUST, &mut image, 1.0);
            let mut displaced = sp.clone();
            displaced.points[1].y += 2;
            let mut wider = sp.clone();
            wider.dct[3][0] += 1;
            candidates.push(Candidate {
                alts: vec![displaced, sp, wider],
                bits_factor: 0.01,
            });
        }
        // Duplicate candidates share every affected block: evaluating candidates
        // against the original image in parallel would subtract them twice.
        candidates.push(Candidate {
            alts: candidates[0].alts.clone(),
            bits_factor: 0.01,
        });
        let quant = vec![8.0; w.div_ceil(8) * h.div_ceil(8)];
        let mut results = Vec::new();
        for threads in [1, 4] {
            let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, threads);
            let mut residual = image.clone();
            let selected = rd_select(
                &ctx,
                &mut CoderScratch::default(),
                3.0,
                &mut residual,
                &quant,
                &candidates,
                None,
            )
            .unwrap();
            results.push((selected, residual));
        }
        let [(a, ar), (b, br)] = results.as_slice() else {
            unreachable!()
        };
        assert_eq!(a.splines.len(), b.splines.len());
        assert!(a.splines.len() < candidates.len());
        for (a, b) in a.splines.iter().zip(&b.splines) {
            assert_eq!(a.points, b.points);
            assert_eq!(a.dct, b.dct);
        }
        for c in 0..3 {
            assert!(
                ar.plane_data(c)
                    .iter()
                    .zip(br.plane_data(c))
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn batched_pruning_matches_serial_with_overlaps_edges_and_forbidden_blocks() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let (w, h) = (263, 193);
        let mut kept = Vec::new();
        for i in 0..3 {
            let points = match i {
                0 => (0..13).map(|x| Point::new(4 + 21 * x, 48)).collect(),
                1 => [0, 3, 9, 12, 9, 3, 0, -6, -9]
                    .iter()
                    .enumerate()
                    .map(|(x, &dy)| Point::new(4 + 32 * x as i32, 128 + dy))
                    .collect(),
                _ => (0..11)
                    .map(|x| Point::new(20 + 24 * x, 4 + 18 * x))
                    .collect(),
            };
            let mut sp = QuantizedSpline {
                points,
                dct: [[0; 32]; 4],
            };
            sp.dct[1][0] = 8 + i;
            sp.dct[2][0] = 5;
            sp.dct[3][0] = 4;
            kept.push(sp);
        }
        let mut residual = Image3F::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for (x, value) in residual.plane_row_mut(c, y).iter_mut().enumerate() {
                    *value = ((x * 13 + y * 7 + c * 3) % 31) as f32 * 0.0003;
                }
            }
        }
        for sp in &kept {
            render_spline(&kernels, sp, QUANT_ADJUST, &mut residual, 1.0);
        }
        for sp in &kept {
            render_spline(&kernels, sp, QUANT_ADJUST, &mut residual, -1.0);
        }
        let blocks = w.div_ceil(8) * h.div_ceil(8);
        let quant = vec![8.0; blocks];
        let mask: Vec<_> = (0..blocks)
            .map(|i| i % w.div_ceil(8) == 16 || i / w.div_ceil(8) == 15)
            .collect();
        for forbidden in [None, Some(mask.as_slice())] {
            let mut reference: Option<(Vec<QuantizedSpline>, Image3F)> = None;
            for threads in [1, 2, 4] {
                let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, threads);
                let mut actual = kept.clone();
                let mut image = residual.clone();
                prune_selected(
                    &BlockModel::new(&ctx, 3.0),
                    &mut CoderScratch::default(),
                    &mut image,
                    &quant,
                    &mut actual,
                    &CostModel::prior(1.0),
                    forbidden,
                );
                if let Some((expected, expected_image)) = &reference {
                    for (a, b) in actual.iter().zip(expected) {
                        assert_eq!(a.points, b.points);
                        assert_eq!(a.dct, b.dct);
                    }
                    for c in 0..3 {
                        assert!(
                            image
                                .plane_data(c)
                                .iter()
                                .zip(expected_image.plane_data(c))
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                        );
                    }
                } else {
                    if forbidden.is_none() {
                        assert!(
                            actual.iter().map(|s| s.points.len()).sum::<usize>()
                                < kept.iter().map(|s| s.points.len()).sum::<usize>()
                        );
                        assert!(actual.iter().any(|s| s.points.len() > 2));
                    }
                    reference = Some((actual, image));
                }
            }
        }
    }

    #[test]
    fn joint_thinning_crosses_the_single_deletion_rate_barrier() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let model = BlockModel::new(&ctx, 3.0);
        let mut sp = QuantizedSpline {
            points: (0..9).map(|i| Point::new(24 + i * 24, 48)).collect(),
            dct: [[0; 32]; 4],
        };
        sp.dct[1][0] = 5;
        sp.dct[3][0] = 4;
        let mut prices = CostModel::prior(1.0);
        prices.add(CTX_POINTS, 0, 32.0);
        for i in 1..sp.points.len() - 1 {
            let mut single = sp.clone();
            single.points.remove(i);
            assert!(
                prices.spline_bits(&single) >= prices.spline_bits(&sp),
                "deletion {i} has no rate barrier"
            );
        }
        let mut original = Image3F::new(264, 96);
        render_spline(&kernels, &sp, QUANT_ADJUST, &mut original, 1.0);
        let mut residual = Image3F::new(264, 96);
        let mut kept = vec![sp.clone()];
        prune_selected(
            &model,
            &mut CoderScratch::default(),
            &mut residual,
            &vec![8.0; 33 * 12],
            &mut kept,
            &prices,
            None,
        );
        assert!(kept[0].points.len() < sp.points.len());
        assert!(prices.spline_bits(&kept[0]) < prices.spline_bits(&sp));
        render_spline(&kernels, &kept[0], QUANT_ADJUST, &mut residual, 1.0);
        for c in 0..3 {
            for (&a, &b) in original.plane_data(c).iter().zip(residual.plane_data(c)) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn selected_straight_line_loses_controls_without_changing_reconstruction() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let model = BlockModel::new(&ctx, 3.0);
        let mut sp = QuantizedSpline {
            points: (0..7).map(|i| Point::new(24 + i * 24, 48)).collect(),
            dct: [[0; 32]; 4],
        };
        sp.dct[1][0] = 5;
        sp.dct[3][0] = 4;
        let mut original = Image3F::new(192, 96);
        render_spline(&kernels, &sp, QUANT_ADJUST, &mut original, 1.0);
        let mut residual = Image3F::new(192, 96);
        let mut kept = vec![sp];
        prune_selected(
            &model,
            &mut CoderScratch::default(),
            &mut residual,
            &vec![8.0; 24 * 12],
            &mut kept,
            &CostModel::prior(1.0),
            None,
        );
        assert_eq!(kept[0].points, [Point::new(24, 48), Point::new(168, 48)]);
        render_spline(&kernels, &kept[0], QUANT_ADJUST, &mut residual, 1.0);
        for c in 0..3 {
            for (&a, &b) in original.plane_data(c).iter().zip(residual.plane_data(c)) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn repricing_does_not_accept_duplicate_lines_against_a_stale_residual() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let model = BlockModel::new(&ctx, 3.0);
        let mut image = Image3F::new(256, 256);
        let quant = vec![8.0; 32 * 32];
        let mut candidates = Vec::new();
        let mut prices = CostModel::prior(0.1);
        for i in 0..9 {
            let mut sp = QuantizedSpline {
                points: vec![Point::new(24, 24 + i * 24), Point::new(224, 24 + i * 24)],
                dct: [[0; 32]; 4],
            };
            sp.dct[1][0] = 5;
            sp.dct[3][0] = 4;
            render_spline(&kernels, &sp, QUANT_ADJUST, &mut image, 1.0);
            if i < 8 {
                for (c, v) in spline_tokens(&sp) {
                    prices.add(c, v, 1.0);
                }
            }
            candidates.push(Candidate {
                alts: vec![sp],
                bits_factor: 0.01,
            });
        }
        let sp = candidates.last().unwrap().alts[0].clone();
        let mut trial = image.clone();
        let b = render_spline(&kernels, &sp, QUANT_ADJUST, &mut trial, -1.0).unwrap();
        let saving = -trial_cost(
            &model,
            &image,
            &trial,
            &quant,
            &mut vec![None; 32 * 32],
            b,
            None,
        )
        .unwrap()
            / (crate::ac_strategy::RD_LAMBDA * margin(3.0));
        let prior_bits = CostModel::prior(1.0).spline_bits(&sp);
        let learned_bits = prices.spline_bits(&sp);
        assert!(saving > 0.0 && learned_bits < prior_bits);
        // This line fails the first pass, but becomes affordable with the
        // learned histogram. Only the first of its two copies may be accepted.
        let factor = saving / ((prior_bits + learned_bits) * 0.5);
        candidates.last_mut().unwrap().bits_factor = factor;
        candidates.push(Candidate {
            alts: vec![sp.clone()],
            bits_factor: factor,
        });
        let selected = rd_select(
            &ctx,
            &mut CoderScratch::default(),
            3.0,
            &mut image,
            &quant,
            &candidates,
            None,
        )
        .unwrap();
        assert_eq!(
            selected
                .splines
                .iter()
                .filter(|s| s.points == sp.points)
                .count(),
            1
        );
        for c in 0..3 {
            assert!(image.plane_data(c).iter().all(|v| v.abs() < 1e-6));
        }
    }

    #[test]
    fn blue_only_changes_respect_forbidden_blocks() {
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let model = BlockModel::new(&ctx, 3.0);
        let current = Image3F::new(8, 8);
        let mut trial = current.clone();
        trial.plane_row_mut(2, 4)[4] = 0.1;
        assert!(block_touched(&current, &trial, 0, 0));
        assert!(
            trial_cost(
                &model,
                &current,
                &trial,
                &[1.0],
                &mut [None],
                (0, 0, 7, 7),
                Some(&[true])
            )
            .is_none()
        );
    }

    #[test]
    fn block_cost_matches_explicit_edge_replication() {
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let model = BlockModel::new(&ctx, 3.0);
        for (w, h) in [(1, 1), (7, 9), (17, 19), (24, 24)] {
            let mut img = Image3F::new(w, h);
            for c in 0..3 {
                for y in 0..h {
                    for (x, v) in img.plane_row_mut(c, y).iter_mut().enumerate() {
                        *v = ((x * 7 + y * 13 + c * 3) % 37) as f32 * 0.013 - 0.2;
                    }
                }
            }
            let mut padded = Image3F::new(w.next_multiple_of(8), h.next_multiple_of(8));
            for c in 0..3 {
                for y in 0..padded.ysize() {
                    for (x, v) in padded.plane_row_mut(c, y).iter_mut().enumerate() {
                        *v = img.plane_row(c, y.min(h - 1))[x.min(w - 1)];
                    }
                }
            }
            for by in 0..h.div_ceil(8) {
                for bx in 0..w.div_ceil(8) {
                    assert_eq!(
                        model.cost(&img, bx, by, 0.75),
                        model.cost(&padded, bx, by, 0.75),
                        "{w}x{h}, block ({bx}, {by})"
                    );
                }
            }
        }
    }
}
