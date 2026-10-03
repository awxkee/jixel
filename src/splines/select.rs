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

use super::dots;
use super::fit::Candidate;
use super::render::{RenderPlan, TiledSpline};
#[cfg(test)]
use super::render_spline;
use super::{
    CTX_DCT, CTX_NUM_POINTS, CTX_POINTS, NUM_SPLINE_CONTEXTS, PixelBox, Point, QUANT_ADJUST,
    QuantizedSpline, SplineSet, spline_tokens,
};
use crate::adaptive_quant::dirty_log2f;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::dct::DctInput;
use crate::encoding_context::EncodingContext;
use crate::entropy::uint_encode;
use crate::image::Image3F;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

const START_POSITION_BITS: f32 = 20.0;
/// Spline bits are charged `margin` times: the DCT8 proxy is optimistic where
/// the encoder merges transforms.
const MARGIN_NEAR: (f32, f32) = (4.0, 2.0);
const MARGIN_FAR: (f32, f32) = (8.0, 1.0);
/// Below this modeled saving the section's fixed cost and the encoder's
/// decision noise outweigh the gain.
const MIN_TOTAL_GAIN_BITS: f32 = 256.0;
/// Bound the combined trial storage of independent candidates to about 12 MiB.
/// A single larger candidate still uses the existing alternative-parallel path.
const MAX_BATCH_TILES: usize = 16_384;
const DOT_POSITION_BITS: f32 = 4.0;
const DOT_FIRST_PASS_TEMPLATE_BITS: f32 = 12.0;
const DOT_TEMPLATE_BITS: f32 = 300.0;
const DOT_BATCH: usize = 256;

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
        let matrices = ctx
            .point_chroma_hq_matrices()
            .unwrap_or_else(|| crate::quant_weights::DequantMatrices::new(distance));
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
            let (d0, r0) = cache
                .get(block.cell)
                .copied()
                .flatten()
                .unwrap_or_else(|| model.cost(current, bx, by, qac));
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

/// `cache_blocks` over many plans, computing the missing block costs on the
/// pool. Each cost depends on its own block only, so the cache matches.
fn cache_blocks_parallel<'p>(
    model: &BlockModel,
    scratch: &mut CoderScratch,
    current: &Image3F,
    quant_field: &[f32],
    cache: &mut [Option<(f32, f32)>],
    plans: impl Iterator<Item = &'p TiledSpline>,
) {
    const CHUNK: usize = 16;
    let blocks_w = current.xsize().div_ceil(8);
    let cost =
        |cell: usize| model.cost(current, cell % blocks_w, cell / blocks_w, quant_field[cell]);
    let mut cells: Vec<usize> = plans
        .flat_map(|plan| plan.tiles.iter().map(|tile| tile.cell))
        .filter(|&cell| cache[cell].is_none())
        .collect();
    cells.sort_unstable();
    cells.dedup();
    let costs =
        model
            .ctx
            .thread_pool
            .steal_map(scratch, cells.len().div_ceil(CHUNK), |chunk, _| {
                cells[chunk * CHUNK..((chunk + 1) * CHUNK).min(cells.len())]
                    .iter()
                    .map(|&cell| cost(cell))
                    .collect::<Vec<_>>()
            });
    for (&cell, cost) in cells.iter().zip(costs.into_iter().flatten()) {
        cache[cell] = Some(cost);
    }
}

#[cfg(test)]
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

fn dictionary_bits_of(
    scratch: &mut CoderScratch,
    references: &[crate::patches::PatchReference],
) -> f32 {
    let mut dictionary = BitWriter::new();
    crate::lossless::write_patch_dictionary(references, false, scratch, &mut dictionary);
    dictionary.bits_written() as f32
}

fn encode_dot_plan(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    plan: &dots::DotPatches,
) -> (BitWriter, f32) {
    let mut atlas = BitWriter::new();
    crate::lossless::encode_modular_xyb_atlas_tree_slot(
        &plan.atlas,
        plan.width,
        plan.height,
        None,
        ctx.speed,
        &ctx.thread_pool,
        scratch,
        crate::patches::DOT_PATCH_REF_ID,
        &mut atlas,
    );
    (atlas, dictionary_bits_of(scratch, &plan.references()))
}

/// Charges the selected dots their real atlas and dictionary bits, shared per
/// template. Placements on unprofitable templates can reuse a paying template
/// before being removed. Rechecks the retained residual against the original
/// image and returns its net gain and final RGB atlas encoding.
fn prune_dot_templates(
    model: &BlockModel,
    scratch: &mut CoderScratch,
    original: &Image3F,
    current: &mut Image3F,
    quant_field: &[f32],
    kept: &mut Vec<QuantizedSpline>,
    raw: &mut [f32],
    candidates: &[Candidate],
    kept_candidates: &[usize],
    forbidden: Option<&[bool]>,
    templates: &HashMap<dots::TemplateKey, Arc<dots::DotTemplate>>,
) -> (f32, Option<dots::DotEncoding>) {
    debug_assert!(kept.iter().all(dots::is_dot));
    debug_assert_eq!(raw.len(), kept.len());
    debug_assert_eq!(kept_candidates.len(), kept.len());
    let ctx = model.ctx;
    let (w, h) = (current.xsize(), current.ysize());
    let mut alive = vec![true; kept.len()];
    // Removal rounds price each template at the second pass's estimate; a
    // Slow atlas encode costs tens of milliseconds. The final set is encoded
    // for real, judged on those bits and carried to output.
    for round in 0..=8 {
        let idx: Vec<usize> = (0..kept.len())
            .filter(|&i| alive[i] && dots::is_dot(&kept[i]))
            .collect();
        if idx.is_empty() {
            kept.clear();
            return (0.0, None);
        }
        let set: Vec<QuantizedSpline> = idx.iter().map(|&i| kept[i].clone()).collect();
        let mut plan = dots::build_dot_patches_with_templates(&set, templates, false);
        let mut dictionary_bits = dictionary_bits_of(scratch, &plan.references());
        let mut atlas_bits = DOT_TEMPLATE_BITS * plan.references().len() as f32;
        let mut by_key: HashMap<dots::TemplateKey, (f32, usize)> = HashMap::new();
        for &i in &idx {
            let e = by_key
                .entry(templates[&dots::template_key(&kept[i])].key)
                .or_insert((0.0, 0));
            e.0 += raw[i];
            e.1 += 1;
        }
        let per_dot = dictionary_bits / idx.len() as f32;
        let per_template = atlas_bits / by_key.len() as f32;
        let losing: Vec<usize> = idx
            .iter()
            .copied()
            .filter(|&i| {
                let (gain, n) = by_key[&templates[&dots::template_key(&kept[i])].key];
                gain < n as f32 * per_dot + per_template
            })
            .collect();
        // The whole set is judged once every template pays its share.
        let total: f32 = idx.iter().map(|&i| raw[i]).sum();
        let dropped = if losing.is_empty() && total < atlas_bits + dictionary_bits {
            &idx
        } else {
            &losing
        };
        if dropped.is_empty() || round == 8 {
            let (padded_atlas, padded_dictionary_bits) = encode_dot_plan(ctx, scratch, &plan);
            let mut atlas = padded_atlas;
            atlas_bits = atlas.bits_written() as f32;
            dictionary_bits = padded_dictionary_bits;
            // Zero padding can help Modular prediction, so race the trimmed
            // layout only once for the final set instead of assuming it wins.
            let trimmed = dots::build_dot_patches_with_templates(&set, templates, true);
            let (trimmed_atlas, trimmed_dictionary_bits) = encode_dot_plan(ctx, scratch, &trimmed);
            let trimmed_bits = trimmed_atlas.bits_written() as f32;
            if trimmed_bits + trimmed_dictionary_bits < atlas_bits + dictionary_bits {
                plan = trimmed;
                atlas = trimmed_atlas;
                atlas_bits = trimmed_bits;
                dictionary_bits = trimmed_dictionary_bits;
            }
            // Patch entries code signed offsets within each template group.
            // Race spatial orders against the raster baseline using the real
            // entropy coder; fitting and template pixels stay unchanged.
            let mut best_references = plan.references();
            for band in [None, Some(16), Some(32), Some(64)] {
                plan.sort_positions(band);
                let bits = dictionary_bits_of(scratch, &plan.references());
                if bits < dictionary_bits {
                    dictionary_bits = bits;
                    best_references = plan.references();
                }
            }

            let blocks_w = w.div_ceil(8);
            let mut touched = vec![false; blocks_w * h.div_ceil(8)];
            for &i in &idx {
                let sp = &kept[i];
                let template = &templates[&dots::template_key(sp)];
                RenderPlan::dot(ctx, sp, template.clone(), w, h).draw(current, 1.0);
                RenderPlan::quantized_dot(ctx, sp, template.clone(), w, h).draw(current, -1.0);
                let x0 = (sp.points[0].x - template.radius) as usize;
                let y0 = (sp.points[0].y - template.radius) as usize;
                for by in y0 / 8..=(y0 + template.side - 1) / 8 {
                    touched[by * blocks_w..][x0 / 8..=(x0 + template.side - 1) / 8].fill(true);
                }
            }
            let mut gain = -(atlas_bits + dictionary_bits);
            // This final check can cover most of a star field. Price disjoint
            // block bands concurrently, then sum in raster order so worker
            // count cannot change the floating-point acceptance decision.
            const COST_BAND: usize = 64;
            let gains =
                ctx.thread_pool
                    .steal_map(scratch, touched.len().div_ceil(COST_BAND), |band, _| {
                        let start = band * COST_BAND;
                        touched[start..(start + COST_BAND).min(touched.len())]
                            .iter()
                            .enumerate()
                            .filter_map(|(offset, &touches)| {
                                if !touches {
                                    return None;
                                }
                                let cell = start + offset;
                                let (bx, by) = (cell % blocks_w, cell / blocks_w);
                                let (d0, r0) = model.cost(original, bx, by, quant_field[cell]);
                                let (d1, r1) = model.cost(current, bx, by, quant_field[cell]);
                                Some((d0 - d1) / crate::ac_strategy::RD_LAMBDA + r0 - r1)
                            })
                            .collect::<Vec<_>>()
                    });
            for delta in gains.into_iter().flatten() {
                gain += delta;
            }
            let mut alive = alive.into_iter();
            kept.retain(|_| alive.next().unwrap());
            return (
                gain,
                Some(dots::DotEncoding {
                    atlas: Arc::new(atlas),
                    references: best_references,
                }),
            );
        }
        let mut cache = vec![None; w.div_ceil(8) * h.div_ceil(8)];
        for &i in dropped {
            let sp = &kept[i];
            let incumbent =
                RenderPlan::dot(ctx, sp, templates[&dots::template_key(sp)].clone(), w, h).tiled();
            // An unprofitable template need not discard every placement. Try
            // the candidate's existing alternatives that reuse a paying
            // template: only another position must be coded for those.
            let alternatives: Vec<_> = candidates[kept_candidates[i]]
                .alts
                .iter()
                .filter(|alt| {
                    // Keep the fitted point's width and brightness. Sharing
                    // another shape can hide a local error behind the RD
                    // proxy's lower aggregate coefficient cost.
                    if alt.dct[3][0] != sp.dct[3][0] || alt.dct[1][0] != sp.dct[1][0] {
                        return false;
                    }
                    by_key
                        .get(&dots::template_key(alt))
                        .is_some_and(|&(gain, n)| gain >= n as f32 * per_dot + per_template)
                })
                .map(|alt| {
                    let plan = RenderPlan::dot(
                        ctx,
                        alt,
                        templates[&dots::template_key(alt)].clone(),
                        w,
                        h,
                    )
                    .tiled();
                    (alt, plan)
                })
                .collect();
            if !alternatives.is_empty() {
                cache_blocks(model, current, quant_field, &mut cache, &incumbent);
            }
            let mut restored = Trial::new(current, &[&incumbent]);
            restored.draw(&incumbent, 1.0);
            let mut best: Option<(f32, &QuantizedSpline, Trial)> = None;
            for (alt, plan) in &alternatives {
                cache_blocks(model, current, quant_field, &mut cache, plan);
                let mut trial = Trial::new(current, &[&incumbent, plan]);
                trial.copy_blocks_from(&restored);
                trial.draw(plan, -1.0);
                let Some(delta) = trial.delta(model, current, quant_field, &cache, forbidden)
                else {
                    continue;
                };
                let gain = raw[i] - delta / crate::ac_strategy::RD_LAMBDA;
                if gain > per_dot && best.as_ref().is_none_or(|(best, _, _)| gain > *best) {
                    best = Some((gain, alt, trial));
                }
            }
            let trial = if let Some((gain, alt, trial)) = best {
                raw[i] = gain;
                kept[i] = alt.clone();
                trial
            } else {
                alive[i] = false;
                restored
            };
            trial.commit(current);
            for block in &trial.blocks {
                cache[block.cell] = None;
            }
        }
    }
    unreachable!()
}

/// Refine the first two width harmonics against the current residual. Keep
/// the fitted mean width, color and geometry; every trial uses the decoder
/// render and must pay for its extra coefficients in the residual RD model.
fn refine_selected_widths(
    model: &BlockModel,
    current: &mut Image3F,
    quant_field: &[f32],
    kept: &mut [QuantizedSpline],
    prices: &CostModel,
    forbidden: Option<&[bool]>,
) {
    let (w, h) = (current.xsize(), current.ysize());
    let mut cache = vec![None; w.div_ceil(8) * h.div_ceil(8)];
    for sp in kept {
        if dots::is_dot(sp) {
            continue;
        }
        let mut incumbent = RenderPlan::new(model.ctx, sp, QUANT_ADJUST, w, h).tiled();
        for i in 1..=2 {
            let mut restored = Trial::new(current, &[&incumbent]);
            restored.draw(&incumbent, 1.0);
            cache_blocks(model, current, quant_field, &mut cache, &incumbent);
            let mut best: Option<(f32, QuantizedSpline, TiledSpline, Trial)> = None;
            for step in [-1, 1] {
                let mut proposal = sp.clone();
                let Some(value) = proposal.dct[3][i].checked_add(step) else {
                    continue;
                };
                proposal.dct[3][i] = value;
                let dc = proposal.dct[3][0] as f32;
                let swing = std::f32::consts::SQRT_2
                    * proposal.dct[3][1..]
                        .iter()
                        .map(|x| x.unsigned_abs() as f32)
                        .sum::<f32>();
                // Bound every arc sample, including those between DCT knots.
                // Units are the format's quantized sigma lattice.
                if dc - swing < 0.5 || dc + swing > 12.0 {
                    continue;
                }
                let plan = RenderPlan::new(model.ctx, &proposal, QUANT_ADJUST, w, h).tiled();
                if plan.bounds().is_none() {
                    continue;
                }
                cache_blocks(model, current, quant_field, &mut cache, &plan);
                let mut trial = Trial::new(current, &[&incumbent, &plan]);
                trial.copy_blocks_from(&restored);
                trial.draw(&plan, -1.0);
                let Some(delta) = trial.delta(model, current, quant_field, &cache, forbidden)
                else {
                    continue;
                };
                let bits = prices.spline_bits(&proposal) - prices.spline_bits(sp);
                let delta = delta + crate::ac_strategy::RD_LAMBDA * margin(model.distance) * bits;
                if delta < -1e-6 && best.as_ref().is_none_or(|(d, _, _, _)| delta < *d) {
                    best = Some((delta, proposal, plan, trial));
                }
            }
            if let Some((_, proposal, plan, trial)) = best {
                trial.commit(current);
                for block in &trial.blocks {
                    cache[block.cell] = None;
                }
                *sp = proposal;
                incumbent = plan;
            }
        }
    }
}

/// The pre-test sees unrefined geometry, so it lets through splines whose
/// VarDCT saving covers only this share of their (prior-priced) bits.
const PRETEST_BITS_SHARE: f32 = 0.25;

/// Independent, state-free RD pre-test of one spline against the untouched
/// image. Works on the blocks the spline covers only, so it can run in parallel.
pub(super) fn pretest(
    model: &BlockModel,
    xyb: &Image3F,
    quant_field: &[f32],
    spline: &QuantizedSpline,
) -> bool {
    let plan = RenderPlan::new(model.ctx, spline, QUANT_ADJUST, xyb.xsize(), xyb.ysize()).tiled();
    if plan.bounds().is_none() {
        return false;
    }
    let mut trial = Trial::new(xyb, &[&plan]);
    trial.draw(&plan, -1.0);
    let Some(dj) = trial.delta(model, xyb, quant_field, &[], None) else {
        return false;
    };
    let bits = CostModel::prior(1.0).spline_bits(spline);
    dj + crate::ac_strategy::RD_LAMBDA * PRETEST_BITS_SHARE * margin(model.distance) * bits < 0.0
}

// Residual distortion and rate are nonnegative in the DCT8 proxy. Even
// removing every modeled coefficient in the affected blocks cannot save more
// than their current objective. Cached block costs make this bound cheap.
fn dot_saving_bound(plan: &TiledSpline, cache: &[Option<(f32, f32)>]) -> Option<f32> {
    let mut saving = 0.0;
    for tile in &plan.tiles {
        let (distortion, rate) = cache.get(tile.cell).copied().flatten()?;
        saving += distortion + crate::ac_strategy::RD_LAMBDA * rate;
    }
    // Slack for accumulation and subtractive rounding in Trial::delta.
    Some(saving + 0.001 * (1.0 + saving))
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
    let mut n_dots = [0usize; 2];
    for candidate in candidates.iter().filter(|c| c.dot) {
        n_dots[usize::from(candidate.alts[0].dct[3][0] < 0)] += 1;
    }
    // Price each polarity's positions independently. Extra dark proposals
    // must not make existing bright dots look cheaper in the first pass.
    let dot_start = n_dots.map(|n| {
        dirty_log2f((w * h) as f32 / n.max(1) as f32).clamp(4.0, 16.0) + DOT_POSITION_BITS
    });
    let keys: Vec<_> = candidates
        .iter()
        .filter(|c| c.dot)
        .flat_map(|c| c.alts.iter().map(dots::template_key))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let dot_templates = dots::share_templates(keys.iter().copied().zip(ctx.thread_pool.steal_map(
        scratch,
        keys.len(),
        |i, _| dots::DotTemplate::new(keys[i]),
    )));
    // Second pass: a dot also pays for its template, by that template's share
    // of the first-pass dots.
    type Popularity = (HashMap<dots::TemplateKey, usize>, [usize; 2]);
    let popularity: RefCell<Option<Popularity>> = RefCell::new(None);
    let price = |prices: &CostModel, candidate: &Candidate, alt: &QuantizedSpline| {
        if dots::is_dot(alt) {
            let template = match popularity.borrow().as_ref() {
                Some((counts, total)) => {
                    let n = counts
                        .get(&dot_templates[&dots::template_key(alt)].key)
                        .copied()
                        .unwrap_or(0)
                        .max(1) as f32;
                    dirty_log2f(total[usize::from(alt.dct[3][0] < 0)].max(1) as f32 / n)
                        + DOT_TEMPLATE_BITS / n
                }
                None => DOT_FIRST_PASS_TEMPLATE_BITS,
            };
            dot_start[usize::from(alt.dct[3][0] < 0)] + template
        } else {
            margin * candidate.bits_factor * prices.spline_bits(alt)
        }
    };
    let prepared = ctx
        .thread_pool
        .steal_map(scratch, candidates.len(), |i, _| {
            candidates[i]
                .alts
                .iter()
                .map(|alt| {
                    if candidates[i].dot {
                        RenderPlan::dot(
                            ctx,
                            alt,
                            dot_templates[&dots::template_key(alt)].clone(),
                            w,
                            h,
                        )
                        .tiled()
                    } else {
                        RenderPlan::new(model.ctx, alt, QUANT_ADJUST, w, h).tiled()
                    }
                })
                .collect::<Vec<_>>()
        });
    struct Evaluation {
        delta: f32,
        trial: Trial,
    }
    let dot_floor = |i: usize| {
        candidates[i]
            .dot
            .then(|| dot_start[usize::from(candidates[i].alts[0].dct[3][0] < 0)])
    };
    let evaluate = |plan: &TiledSpline,
                    current: &Image3F,
                    cache: &[Option<(f32, f32)>],
                    floor: Option<f32>| {
        plan.bounds()?;
        if let Some(bits) = floor
            && let Some(saving) = dot_saving_bound(plan, cache)
            && saving < lambda * bits
        {
            // Retain a rejecting lower bound, rather than None: a changed
            // residual must still trigger a second-pass trial.
            return Some(Evaluation {
                delta: -saving,
                trial: Trial { blocks: Vec::new() },
            });
        }
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

    // Candidates commit in their original greedy order. Disjoint candidates can
    // share a residual snapshot while all their alternatives run concurrently.
    // Keep winning trials and retain preparation across repricing passes.
    let mut prices = CostModel::prior(1.0);
    let mut current = xyb.clone();
    let mut block_cache = vec![None; blocks_w * h.div_ceil(8)];
    let mut deltas: Vec<Vec<Option<(f32, PixelBox)>>> = Vec::with_capacity(candidates.len());
    let mut first_pass = Vec::new();
    let mut first_choices = Vec::with_capacity(candidates.len());
    let mut occupied = vec![false; block_cache.len()];
    let mut next = 0;
    let mut second_pass_batches = Vec::new();
    while next < candidates.len() {
        let start = next;
        let mut batch_tiles = 0usize;
        // The first pass has fixed prices. Only candidates whose complete
        // alternative supports are disjoint may read the same residual.
        // Dots are tiny and plentiful: wide batches keep the pool busy.
        let batch = if candidates[start].dot {
            DOT_BATCH
        } else {
            ctx.thread_pool.num_threads().max(1)
        };
        while next < candidates.len() && next - start < batch {
            let plans = &prepared[next];
            let tiles: usize = plans.iter().map(|plan| plan.tiles.len()).sum();
            if next != start
                && (batch_tiles + tiles > MAX_BATCH_TILES
                    || plans
                        .iter()
                        .any(|plan| plan.tiles.iter().any(|tile| occupied[tile.cell])))
            {
                break;
            }
            for plan in plans {
                for tile in &plan.tiles {
                    occupied[tile.cell] = true;
                }
            }
            batch_tiles += tiles;
            next += 1;
        }
        second_pass_batches.push(start..next);
        cache_blocks_parallel(
            &model,
            scratch,
            &current,
            quant_field,
            &mut block_cache,
            prepared[start..next].iter().flatten(),
        );
        let assess = |candidate: &Candidate,
                      plans: &[TiledSpline],
                      mut evaluations: Vec<Option<Evaluation>>| {
            let mut row = vec![None; candidate.alts.len()];
            let mut best: Option<(f32, usize, PixelBox)> = None;
            for (index, evaluation) in evaluations.iter().enumerate() {
                let Some(evaluation) = evaluation else {
                    continue;
                };
                let b = plans[index].bounds().unwrap();
                let dj = evaluation.delta;
                row[index] = Some((dj, b));
                let j = dj + lambda * price(&prices, candidate, &candidate.alts[index]);
                if best.is_none_or(|(bj, _, _)| j < bj) {
                    best = Some((j, index, b));
                }
            }
            let accepted = best.filter(|&(j, _, _)| j < 0.0);

            let chosen =
                accepted.map(|(_, index, b)| (index, b, evaluations[index].take().unwrap().trial));
            (row, chosen)
        };
        let results = if next - start == 1 {
            let plans = &prepared[start];
            let evaluations = ctx.thread_pool.steal_map(scratch, plans.len(), |i, _| {
                evaluate(&plans[i], &current, &block_cache, dot_floor(start))
            });
            vec![assess(&candidates[start], plans, evaluations)]
        } else {
            let jobs: Vec<_> = (start..next)
                .flat_map(|i| (0..prepared[i].len()).map(move |alt| (i, alt)))
                .collect();
            let mut evaluations = ctx
                .thread_pool
                .steal_map(scratch, jobs.len(), |i, _| {
                    let (candidate, alt) = jobs[i];
                    evaluate(
                        &prepared[candidate][alt],
                        &current,
                        &block_cache,
                        dot_floor(candidate),
                    )
                })
                .into_iter();
            (start..next)
                .map(|i| {
                    let plans = &prepared[i];
                    assess(
                        &candidates[i],
                        plans,
                        evaluations.by_ref().take(plans.len()).collect(),
                    )
                })
                .collect()
        };
        for (i, (row, chosen)) in results.into_iter().enumerate() {
            first_choices.push(chosen.as_ref().map(|(index, _, _)| *index));
            if let Some((index, b, trial)) = chosen {
                trial.commit(&mut current);
                invalidate(&mut block_cache, b);
                first_pass.push(candidates[start + i].alts[index].clone());
            }
            deltas.push(row);
        }
        for plans in &prepared[start..next] {
            for plan in plans {
                for tile in &plan.tiles {
                    occupied[tile.cell] = false;
                }
            }
        }
    }
    drop(occupied);
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
    {
        let mut counts = HashMap::new();
        let mut total = [0usize; 2];
        for sp in first_pass.iter().filter(|sp| dots::is_dot(sp)) {
            *counts
                .entry(dot_templates[&dots::template_key(sp)].key)
                .or_insert(0usize) += 1;
            total[usize::from(sp.dct[3][0] < 0)] += 1;
        }
        if total.iter().any(|&n| n > 0) {
            *popularity.borrow_mut() = Some((counts, total));
        }
    }
    for sp in first_pass.iter().filter(|sp| !dots::is_dot(sp)) {
        for (c, v) in spline_tokens(sp) {
            prices.add(c, v, 1.0);
        }
    }
    let mut kept = Vec::new();
    let mut kept_raw: Vec<f32> = Vec::new();
    let mut kept_candidates = Vec::new();
    let mut gain_bits = 0.0;
    // Prices are fixed during repricing, and the first pass already proved
    // these complete alternative supports disjoint. Recompute them together
    // against one snapshot while keeping commits in the original greedy order.
    for batch in second_pass_batches {
        let parallel_batch = batch.len() > 1;
        let recomputations: Vec<Vec<bool>> = batch
            .clone()
            .map(|i| {
                deltas[i]
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
                    .collect()
            })
            .collect();
        cache_blocks_parallel(
            &model,
            scratch,
            &current,
            quant_field,
            &mut block_cache,
            batch
                .clone()
                .zip(&recomputations)
                .flat_map(|(i, recompute)| {
                    prepared[i]
                        .iter()
                        .zip(recompute)
                        .filter_map(|(plan, &needed)| needed.then_some(plan))
                }),
        );
        let mut batched_evaluations = if parallel_batch {
            let jobs: Vec<_> = batch
                .clone()
                .zip(&recomputations)
                .flat_map(|(i, needed)| {
                    needed
                        .iter()
                        .enumerate()
                        .filter_map(move |(alt, &needed)| needed.then_some((i, alt)))
                })
                .collect();
            ctx.thread_pool.steal_map(scratch, jobs.len(), |job, _| {
                let (i, alt) = jobs[job];
                evaluate(&prepared[i][alt], &current, &block_cache, dot_floor(i))
            })
        } else {
            Vec::new()
        }
        .into_iter();
        for (i, recompute) in batch.zip(recomputations) {
            let candidate = &candidates[i];
            let floor = dot_floor(i);
            let plans = &prepared[i];
            let row = &deltas[i];
            let first = first_choices[i];
            let mut evaluations = if parallel_batch {
                recompute
                    .iter()
                    .map(|&needed| {
                        if needed {
                            batched_evaluations.next().unwrap()
                        } else {
                            None
                        }
                    })
                    .collect()
            } else if candidate.dot {
                // A dot's few one-sample trials cost less than a pool dispatch.
                (0..plans.len())
                    .map(|i| {
                        recompute[i]
                            .then(|| evaluate(&plans[i], &current, &block_cache, floor))
                            .flatten()
                    })
                    .collect()
            } else if recompute.iter().any(|&needed| needed) {
                ctx.thread_pool.steal_map(scratch, plans.len(), |i, _| {
                    recompute[i]
                        .then(|| evaluate(&plans[i], &current, &block_cache, floor))
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
                let j = dj + lambda * price(&prices, candidate, alt);
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
                kept_raw.push(-j / lambda + price(&prices, candidate, &candidate.alts[index]));
                kept_candidates.push(i);
                kept.push(candidate.alts[index].clone());
            }
        }
    }
    let dot_encoding = if kept.iter().any(dots::is_dot) {
        let (gain, encoding) = prune_dot_templates(
            &model,
            scratch,
            xyb,
            &mut current,
            quant_field,
            &mut kept,
            &mut kept_raw,
            candidates,
            &kept_candidates,
            forbidden,
            &dot_templates,
        );
        gain_bits = gain;
        encoding
    } else {
        None
    };
    if kept.is_empty() || gain_bits < MIN_TOTAL_GAIN_BITS {
        return None;
    }
    drop(prepared);
    if dot_encoding.is_none() {
        prune_selected(
            &model,
            scratch,
            &mut current,
            quant_field,
            &mut kept,
            &prices,
            forbidden,
        );
        refine_selected_widths(
            &model,
            &mut current,
            quant_field,
            &mut kept,
            &prices,
            forbidden,
        );
    }
    *xyb = current;
    Some(SplineSet {
        adjust: QUANT_ADJUST,
        splines: kept,
        dot_encoding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Speed, xyb::XybMatrix};

    #[test]
    fn dot_template_reuse_preserves_shape_protected_blocks_and_reconstruction() {
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 2.0, 1);
        let model = BlockModel::new(&ctx, 2.0);
        let (w, h) = (192usize, 128usize);
        let mut selected = Vec::new();
        for i in 0..20 {
            let mut sp = QuantizedSpline {
                points: vec![Point::new(16 + 32 * (i % 5), 16 + 24 * (i / 5))],
                dct: [[0; 32]; 4],
            };
            sp.dct[3][0] = 4;
            sp.dct[1][0] = 16;
            selected.push(sp);
        }
        for (sigma, level) in [(4, 16), (5, 16), (4, 17)] {
            let mut selected = selected.clone();
            let mut rare = selected[0].clone();
            rare.points[0] = Point::new(176, 112);
            rare.dct[0][0] = 1;
            rare.dct[3][0] = sigma;
            rare.dct[1][0] = level;
            let mut shared = selected[0].clone();
            shared.points[0] = rare.points[0];
            let mut unused = rare.clone();
            unused.dct[0][0] = 2;
            selected.push(rare.clone());
            let candidates: Vec<_> = selected
                .iter()
                .enumerate()
                .map(|(i, sp)| Candidate {
                    alts: if i == 20 {
                        vec![unused.clone(), shared.clone(), rare.clone()]
                    } else {
                        vec![sp.clone()]
                    },
                    dot: true,
                    bits_factor: 1.0,
                })
                .collect();
            let keys: BTreeSet<_> = candidates
                .iter()
                .flat_map(|c| c.alts.iter().map(dots::template_key))
                .collect();
            let templates = dots::share_templates(
                keys.into_iter()
                    .map(|key| (key, dots::DotTemplate::new(key))),
            );
            let mut original = Image3F::new(w, h);
            for sp in &selected {
                RenderPlan::dot(&ctx, sp, templates[&dots::template_key(sp)].clone(), w, h)
                    .draw(&mut original, 1.0);
            }
            let quant = vec![8.0; w.div_ceil(8) * h.div_ceil(8)];
            let mask = vec![true; quant.len()];
            for forbidden in [None, Some(mask.as_slice())] {
                let mut residual = original.clone();
                for sp in &selected {
                    RenderPlan::dot(&ctx, sp, templates[&dots::template_key(sp)].clone(), w, h)
                        .draw(&mut residual, -1.0);
                }
                let mut kept = selected.clone();
                let mut raw = vec![500.0; kept.len()];
                raw[20] = 150.0;
                let (_, encoding) = prune_dot_templates(
                    &model,
                    &mut CoderScratch::default(),
                    &original,
                    &mut residual,
                    &quant,
                    &mut kept,
                    &mut raw,
                    &candidates,
                    &(0..selected.len()).collect::<Vec<_>>(),
                    forbidden,
                    &templates,
                );
                assert_eq!(encoding.unwrap().references.len(), 1);
                if forbidden.is_none() && sigma == 4 && level == 16 {
                    assert_eq!(kept.len(), selected.len());
                    assert_eq!(
                        dots::template_key(kept.last().unwrap()),
                        dots::template_key(&shared)
                    );
                } else {
                    assert_eq!(kept.len(), selected.len() - 1);
                    for c in 0..3 {
                        assert_eq!(
                            residual.plane_row(c, 112)[176],
                            original.plane_row(c, 112)[176]
                        );
                    }
                }
                // The committed residual must match the chosen decoder templates.
                for sp in &kept {
                    RenderPlan::quantized_dot(
                        &ctx,
                        sp,
                        templates[&dots::template_key(sp)].clone(),
                        w,
                        h,
                    )
                    .draw(&mut residual, 1.0);
                }
                for c in 0..3 {
                    assert!(
                        original
                            .plane_data(c)
                            .iter()
                            .zip(residual.plane_data(c))
                            .all(|(a, b)| (a - b).abs() < 1e-6)
                    );
                }
            }
        }
    }

    #[test]
    fn width_refinement_recovers_taper_and_preserves_reconstruction() {
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 2.0, 1);
        let model = BlockModel::new(&ctx, 2.0);
        let (w, h) = (259, 97);
        let mut truth = QuantizedSpline {
            points: vec![Point::new(8, 18), Point::new(251, 79)],
            dct: [[0; 32]; 4],
        };
        truth.dct[1][0] = -12;
        truth.dct[3][0] = 5;
        truth.dct[3][1] = 1;
        let mut source = Image3F::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                source.plane_row_mut(c, y).fill(0.5);
            }
        }
        render_spline(&ctx, &truth, QUANT_ADJUST, &mut source, 1.0);
        let mut initial = truth.clone();
        initial.dct[3][1] = 0;
        let mut residual = source.clone();
        render_spline(&ctx, &initial, QUANT_ADJUST, &mut residual, -1.0);
        let before = residual.clone();
        let quant = vec![8.0; w.div_ceil(8) * h.div_ceil(8)];
        let prices = CostModel::prior(1.0);
        let mut kept = vec![initial.clone()];
        refine_selected_widths(&model, &mut residual, &quant, &mut kept, &prices, None);
        assert_eq!(kept[0].dct[3], truth.dct[3]);
        assert_eq!(kept[0].dct[..3], initial.dct[..3]);
        assert_eq!(kept[0].points, initial.points);
        render_spline(&ctx, &kept[0], QUANT_ADJUST, &mut residual, 1.0);
        for c in 0..3 {
            for (&actual, &expected) in residual.plane_data(c).iter().zip(source.plane_data(c)) {
                assert!((actual - expected).abs() < 2e-6);
            }
        }
        let mut forbidden_residual = before.clone();
        let mut forbidden_splines = vec![initial.clone()];
        refine_selected_widths(
            &model,
            &mut forbidden_residual,
            &quant,
            &mut forbidden_splines,
            &prices,
            Some(&vec![true; quant.len()]),
        );
        assert_eq!(forbidden_splines[0].dct, initial.dct);
        for c in 0..3 {
            assert_eq!(forbidden_residual.plane_data(c), before.plane_data(c));
        }
    }

    #[test]
    fn sparse_trials_match_full_render_and_rd_at_image_edges() {
        let kernels = EncodingContext::default();
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
                dot: false,
            });
        }
        // Duplicate candidates share every affected block: evaluating candidates
        // against the original image in parallel would subtract them twice.
        candidates.push(Candidate {
            alts: candidates[0].alts.clone(),
            bits_factor: 0.01,
            dot: false,
        });
        let quant = vec![8.0; w.div_ceil(8) * h.div_ceil(8)];
        let mut results = Vec::new();
        for (duplicates, threads) in [(false, 1), (false, 4), (true, 1), (true, 4)] {
            let repeated;
            let candidates = if duplicates {
                repeated = candidates
                    .iter()
                    .map(|candidate| Candidate {
                        alts: candidate
                            .alts
                            .iter()
                            .flat_map(|alt| [alt.clone(), alt.clone()])
                            .collect(),
                        bits_factor: candidate.bits_factor,
                        dot: false,
                    })
                    .collect::<Vec<_>>();
                &repeated
            } else {
                &candidates
            };
            let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, threads);
            let mut residual = image.clone();
            let selected = rd_select(
                &ctx,
                &mut CoderScratch::default(),
                3.0,
                &mut residual,
                &quant,
                candidates,
                None,
            )
            .unwrap();
            results.push((selected, residual));
        }
        let (a, ar) = &results[0];
        assert!(a.splines.len() < candidates.len());
        for (b, br) in &results[1..] {
            assert_eq!(a.splines.len(), b.splines.len());
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
                dot: false,
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
            dot: false,
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
    fn dot_bound_covers_measured_savings_and_requires_current_costs() {
        for distance in [1.0, 4.0] {
            let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, distance, 1);
            let model = BlockModel::new(&ctx, distance);
            for width in [-11, -7, 4, 11] {
                let mut sp = QuantizedSpline {
                    points: vec![Point::new(23, 19)],
                    dct: [[0; 32]; 4],
                };
                sp.dct[3][0] = width;
                sp.dct[1][0] = 12;
                let plan = RenderPlan::new(&ctx, &sp, QUANT_ADJUST, 48, 40).tiled();
                let mut image = Image3F::new(48, 40);
                RenderPlan::new(&ctx, &sp, QUANT_ADJUST, 48, 40).draw(&mut image, 1.0);
                let quant = vec![3.0; 6 * 5];
                let mut cache = vec![None; quant.len()];
                assert!(dot_saving_bound(&plan, &cache).is_none());
                cache_blocks(&model, &image, &quant, &mut cache, &plan);
                let saving = dot_saving_bound(&plan, &cache).unwrap();
                let mut trial = Trial::new(&image, &[&plan]);
                trial.draw(&plan, -1.0);
                let delta = trial.delta(&model, &image, &quant, &cache, None).unwrap();
                assert!(-delta <= saving, "distance={distance}, width={width}");
            }
        }
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
    // Original candidate-by-candidate selector: keep independent of the batched
    // first pass so the differential test detects changes to greedy decisions.
    fn rd_select_reference(
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
        refine_selected_widths(
            &model,
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
            dot_encoding: None,
        })
    }

    #[test]
    fn disjoint_candidate_batches_match_original_greedy_selection() {
        let kernels = EncodingContext::default();
        let (w, h) = (263usize, 577usize);
        let mut image = Image3F::new(w, h);
        let mut candidates = Vec::new();
        let mut occupied = std::collections::BTreeSet::new();
        for i in 0..8 {
            let y = 4 + i * 80;
            let mut sp = QuantizedSpline {
                points: vec![Point::new(0, y), Point::new(128, y + 3), Point::new(262, y)],
                dct: [[0; 32]; 4],
            };
            sp.dct[1][0] = 8;
            sp.dct[3][0] = 4;
            render_spline(&kernels, &sp, QUANT_ADJUST, &mut image, 1.0);
            let mut shifted = sp.clone();
            shifted.points[1].y += 2;
            let mut wider = sp.clone();
            wider.dct[3][0] += 1;
            let candidate = Candidate {
                alts: vec![shifted, sp, wider],
                bits_factor: 0.01,
                dot: false,
            };
            let cells: std::collections::BTreeSet<_> = candidate
                .alts
                .iter()
                .flat_map(|alt| {
                    RenderPlan::new(&kernels, alt, QUANT_ADJUST, w, h)
                        .tiled()
                        .tiles
                        .into_iter()
                        .map(|t| t.cell)
                })
                .collect();
            assert!(occupied.is_disjoint(&cells));
            occupied.extend(cells);
            candidates.push(candidate);
        }
        candidates.push(Candidate {
            alts: candidates[0].alts.clone(),
            bits_factor: 0.01,
            dot: false,
        });
        let blocks_w = w.div_ceil(8);
        let quant = vec![8.0; blocks_w * h.div_ceil(8)];
        let forbidden: Vec<_> = (0..quant.len()).map(|i| i / blocks_w == 30).collect();
        for mask in [None, Some(forbidden.as_slice())] {
            let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
            let mut expected_image = image.clone();
            let expected = rd_select_reference(
                &ctx,
                &mut CoderScratch::default(),
                3.0,
                &mut expected_image,
                &quant,
                &candidates,
                mask,
            )
            .unwrap();
            assert!(expected.splines.len() >= 4 && expected.splines.len() < candidates.len());
            for threads in [2, 4, 8] {
                let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, threads);
                let mut actual_image = image.clone();
                let actual = rd_select(
                    &ctx,
                    &mut CoderScratch::default(),
                    3.0,
                    &mut actual_image,
                    &quant,
                    &candidates,
                    mask,
                )
                .unwrap();
                assert_eq!(actual.splines.len(), expected.splines.len());
                for (a, b) in actual.splines.iter().zip(&expected.splines) {
                    assert_eq!(a.points, b.points);
                    assert_eq!(a.dct, b.dct);
                }
                for c in 0..3 {
                    assert!(
                        actual_image
                            .plane_data(c)
                            .iter()
                            .zip(expected_image.plane_data(c))
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
    }
}
