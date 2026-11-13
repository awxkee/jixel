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
use super::fit::{Candidate, LineGroup};
use super::lines::{LineIndex, curve_samples, unit_tangents};
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
    thresholds: [[f32; 4]; 3],
    bias: [crate::group::DequantBias; 3],
    weights: [f32; 3],
    scan: &'static [u32],
    distance: f32,
}

impl<'a> BlockModel<'a> {
    pub(super) fn new(ctx: &'a EncodingContext, distance: f32) -> Self {
        let matrices = ctx.point_chroma_hq_matrices().unwrap_or_else(|| {
            crate::quant_weights::DequantMatrices::new(distance, crate::Arbiter::Ssimulacra2, false)
        });
        let qm = [1.0, 1.0, ctx.b_qm_mul()];
        BlockModel {
            ctx,
            inv: [
                matrices.inv_matrix(0),
                matrices.inv_matrix(1),
                matrices.inv_matrix(2),
            ],
            qm,
            thresholds: std::array::from_fn(|c| {
                crate::group::quantize_ac_thresholds_scaled(c, 1, 1, distance, qm[c])
            }),
            bias: std::array::from_fn(|c| ctx.dequant_bias(c)),
            weights: ctx.channel_weights(),
            scan: crate::coeff_order::scan_pos_lut(8, 8),
            distance,
        }
    }

    #[inline]
    fn coeff_cost(&self, c: usize, coeff: &[f32; 64], qac: f32) -> (f32, f32) {
        // SAFETY: coefficients, quant matrix and scan order cover one 8×8 block.
        let (sse, nonzeros, magnitude, last) = unsafe {
            (self.ctx.sse_and_rate)(
                coeff,
                self.inv[c],
                qac * self.qm[c],
                self.bias[c],
                8,
                8,
                4,
                1,
                1,
                self.ctx.rate_log2_lut,
                &self.thresholds[c],
                self.scan,
            )
        };
        (
            sse,
            crate::inflated_cost::model_bits(nonzeros, magnitude, last, 1, 1),
        )
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
            let (d, r) = self.coeff_cost(c, &coef, qac);
            d_total += self.weights[c] * d;
            r_total += r;
        }
        (d_total, r_total)
    }

    fn plane_cost(&self, c: usize, pixels: &[f32; 64], qac: f32) -> (f32, f32) {
        let mut coef = [0.0; 64];
        (self.ctx.dct8x8)(DctInput::from_flat(pixels), &mut coef);
        self.coeff_cost(c, &coef, qac)
    }

    fn pixels_cost(&self, pixels: &[[f32; 64]; 3], qac: f32) -> (f32, f32) {
        let (mut distortion, mut rate) = (0.0, 0.0);
        for (c, plane) in pixels.iter().enumerate() {
            let (d, r) = self.plane_cost(c, plane, qac);
            distortion += self.weights[c] * d;
            rate += r;
        }
        (distortion, rate)
    }
}

/// Costs for adjacent color alternatives of the same dot and residual.
/// A shared shape and clipped support give identical rendered pixels for each
/// channel with matching color, including the edge padding applied in trials.
#[derive(Default)]
struct DotCosts {
    shape: Option<(Point<i32>, i32, i32, PixelBox)>,
    blocks: Vec<[Option<(i32, f32, f32)>; 3]>,
}

impl DotCosts {
    #[cfg(test)]
    fn clear(&mut self) {
        self.shape = None;
    }

    fn prepare(&mut self, alt: &QuantizedSpline, bounds: PixelBox) {
        let (width, level, _, _) = dots::template_key(alt);
        let shape = (alt.points[0], width, level, bounds);
        if self.shape != Some(shape) {
            self.shape = Some(shape);
            let (x0, y0, x1, y1) = bounds;
            self.blocks.clear();
            self.blocks
                .resize((x1 / 8 - x0 / 8 + 1) * (y1 / 8 - y0 / 8 + 1), [None; 3]);
        }
    }

    fn pixels_cost(
        &mut self,
        model: &BlockModel,
        colors: [i32; 3],
        pixels: &[[f32; 64]; 3],
        qac: f32,
        bx: usize,
        by: usize,
    ) -> (f32, f32) {
        let (_, _, _, (x0, y0, x1, _)) = self.shape.unwrap();
        let i = (by - y0 / 8) * (x1 / 8 - x0 / 8 + 1) + bx - x0 / 8;
        let (mut distortion, mut rate) = (0.0, 0.0);
        for (c, ((plane, slot), color)) in pixels
            .iter()
            .zip(&mut self.blocks[i])
            .zip(colors)
            .enumerate()
        {
            let (d, r) = match *slot {
                Some((cached, d, r)) if cached == color => (d, r),
                _ => {
                    let (d, r) = model.plane_cost(c, plane, qac);
                    *slot = Some((color, d, r));
                    (d, r)
                }
            };
            distortion += model.weights[c] * d;
            rate += r;
        }
        (distortion, rate)
    }
}

#[derive(Clone)]
struct TrialBlock {
    cell: usize,
    pixels: [[f32; 64]; 3],
}

impl TrialBlock {
    fn load(current: &Image3F, cell: usize) -> Self {
        let (w, h) = (current.xsize(), current.ysize());
        let blocks_w = w.div_ceil(8);
        let (x0, y0) = (cell % blocks_w * 8, cell / blocks_w * 8);
        let width = (w - x0).min(8);
        let mut pixels = [[0.0; 64]; 3];
        for (c, plane) in pixels.iter_mut().enumerate() {
            for (y, row) in plane.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                row[..width]
                    .copy_from_slice(&current.plane_row(c, (y0 + y).min(h - 1))[x0..x0 + width]);
                let edge = row[width - 1];
                row[width..].fill(edge);
            }
        }
        Self { cell, pixels }
    }

    #[inline]
    fn add_delta(
        &mut self,
        model: &BlockModel,
        current: &Image3F,
        quant_field: &[f32],
        cache: &[Option<(f32, f32)>],
        forbidden: Option<&[bool]>,
        pixels_cost: &mut impl FnMut(&[[f32; 64]; 3], f32, usize, usize) -> (f32, f32),
        delta: &mut f32,
    ) -> Option<()> {
        let (w, h) = (current.xsize(), current.ysize());
        let blocks_w = w.div_ceil(8);
        let (bx, by) = (self.cell % blocks_w, self.cell / blocks_w);
        let (x0, y0) = (bx * 8, by * 8);
        let (width, height) = ((w - x0).min(8), (h - y0).min(8));
        let touched = (0..height).any(|y| {
            let xs = x0..x0 + width;
            let a = &current.plane_row(0, y0 + y)[xs.clone()];
            let b = &current.plane_row(1, y0 + y)[xs.clone()];
            let c = &current.plane_row(2, y0 + y)[xs];
            let [x, yy, z] = &self.pixels;
            let at = y * 8;
            a.iter()
                .zip(b)
                .zip(c)
                .zip(x[at..].iter().zip(&yy[at..]).zip(&z[at..]))
                .any(|(((&a, &b), &c), ((&x, &y), &z))| {
                    (b - y).abs() > 1e-5 || (a - x).abs() > 1e-6 || (c - z).abs() > 1e-5
                })
        });
        if !touched {
            return Some(());
        }
        if forbidden.is_some_and(|mask| mask[self.cell]) {
            return None;
        }
        // Replicate the rendered edge, rather than the original edge used
        // when loading the trial. Interior blocks need no padding work.
        if width < 8 || height < 8 {
            for plane in &mut self.pixels {
                for y in 0..height {
                    let edge = plane[y * 8 + width - 1];
                    plane[y * 8 + width..y * 8 + 8].fill(edge);
                }
                for y in height..8 {
                    plane.copy_within((height - 1) * 8..height * 8, y * 8);
                }
            }
        }
        let qac = quant_field[self.cell];
        let (d0, r0) = cache
            .get(self.cell)
            .copied()
            .flatten()
            .unwrap_or_else(|| model.cost(current, bx, by, qac));
        let (d1, r1) = pixels_cost(&self.pixels, qac, bx, by);
        *delta += (d1 - d0) + crate::ac_strategy::RD_LAMBDA * (r1 - r0);
        Some(())
    }
}

/// A trial owns only affected blocks. Coordinates and sample accumulation stay
/// identical to full-image rendering; different alternatives share no writes.
struct Trial {
    blocks: Vec<TrialBlock>,
}

impl Trial {
    fn new(current: &Image3F, plans: &[&TiledSpline]) -> Self {
        if let [plan] = plans {
            return Self {
                blocks: plan
                    .tiles
                    .iter()
                    .map(|tile| TrialBlock::load(current, tile.cell))
                    .collect(),
            };
        }
        let mut cells: Vec<_> = plans
            .iter()
            .flat_map(|p| p.tiles.iter().map(|t| t.cell))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        let blocks = cells
            .into_iter()
            .map(|cell| TrialBlock::load(current, cell))
            .collect();
        Self { blocks }
    }

    /// Load and draw one alternative in block order.
    fn drawn(current: &Image3F, plan: &TiledSpline) -> Self {
        let blocks = plan
            .tiles
            .iter()
            .map(|tile| {
                let mut block = TrialBlock::load(current, tile.cell);
                plan.draw_tile(tile, &mut block.pixels, -1.0);
                block
            })
            .collect();
        Self { blocks }
    }

    /// Merge the restored incumbent and replacement's sorted cells. Only
    /// cells outside the incumbent need to be read from the current image.
    fn replacement(current: &Image3F, restored: &Self, plan: &TiledSpline) -> Self {
        let mut cached = restored.blocks.iter().peekable();
        let mut cells = plan.tiles.iter().map(|tile| tile.cell).peekable();
        let merged = std::iter::from_fn(move || {
            match (cached.peek(), cells.peek()) {
                (Some(block), Some(&cell)) if cell <= block.cell => {
                    cells.next();
                    if cell < block.cell {
                        return Some((cell, None));
                    }
                }
                (None, _) => return cells.next().map(|cell| (cell, None)),
                _ => {}
            }
            cached.next().map(|block| (block.cell, Some(block)))
        });
        // Reserve the union size. Most replacements overlap the incumbent.
        let mut blocks = Vec::with_capacity(merged.clone().count());
        blocks.extend(merged.map(|(cell, cached)| {
            cached
                .cloned()
                .unwrap_or_else(|| TrialBlock::load(current, cell))
        }));
        Self { blocks }
    }

    fn draw(&mut self, plan: &TiledSpline, sign: f32) {
        let mut blocks = self.blocks.iter_mut();
        for tile in &plan.tiles {
            let block = blocks.find(|block| block.cell == tile.cell).unwrap();
            plan.draw_tile(tile, &mut block.pixels, sign);
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
        self.delta_with(
            model,
            current,
            quant_field,
            cache,
            forbidden,
            |pixels, qac, _, _| model.pixels_cost(pixels, qac),
        )
    }

    fn delta_with_dot_costs(
        &mut self,
        model: &BlockModel,
        current: &Image3F,
        quant_field: &[f32],
        cache: &[Option<(f32, f32)>],
        forbidden: Option<&[bool]>,
        alt: &QuantizedSpline,
        bounds: PixelBox,
        costs: &mut DotCosts,
    ) -> Option<f32> {
        costs.prepare(alt, bounds);
        let colors = [alt.dct[0][0], 0, alt.dct[2][0]];
        self.delta_with(
            model,
            current,
            quant_field,
            cache,
            forbidden,
            |pixels, qac, bx, by| costs.pixels_cost(model, colors, pixels, qac, bx, by),
        )
    }

    fn delta_with(
        &mut self,
        model: &BlockModel,
        current: &Image3F,
        quant_field: &[f32],
        cache: &[Option<(f32, f32)>],
        forbidden: Option<&[bool]>,
        mut pixels_cost: impl FnMut(&[[f32; 64]; 3], f32, usize, usize) -> (f32, f32),
    ) -> Option<f32> {
        let mut delta = 0.0;
        for block in &mut self.blocks {
            block.add_delta(
                model,
                current,
                quant_field,
                cache,
                forbidden,
                &mut pixels_cost,
                &mut delta,
            )?;
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
    if model.ctx.thread_pool.num_threads() == 1 {
        for cell in plans.flat_map(|plan| plan.tiles.iter().map(|tile| tile.cell)) {
            cache[cell].get_or_insert_with(|| cost(cell));
        }
        return;
    }
    let mut cells = Vec::new();
    for cell in plans.flat_map(|plan| plan.tiles.iter().map(|tile| tile.cell)) {
        if cache[cell].is_none() {
            // Mark scheduled entries so overlapping alternatives only add
            // each cell once. Nothing reads the cache until the fill ends.
            cache[cell] = Some((0.0, 0.0));
            cells.push(cell);
        }
    }
    if cells.len() <= CHUNK {
        for cell in cells {
            cache[cell] = Some(cost(cell));
        }
        return;
    }
    cells.sort_unstable();
    let mut jobs = Vec::with_capacity(cells.len().div_ceil(CHUNK));
    let mut start = 0;
    let mut tail = cache;
    for cells in cells.chunks(CHUNK) {
        let end = cells.last().unwrap() + 1;
        let (entries, rest) = tail.split_at_mut(end - start);
        jobs.push((start, cells, entries));
        start = end;
        tail = rest;
    }
    // The sorted ranges give each worker a disjoint mutable cache slice.
    model.ctx.thread_pool.steal_for_each_mut(
        scratch,
        &mut jobs,
        |_, (start, cells, entries), _| {
            for &cell in *cells {
                entries[cell - *start] = Some(cost(cell));
            }
        },
    );
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
                          incumbent_bits: &mut f32,
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
                let bits = prices.spline_bits(&simpler);
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
                let mut trial = Trial::replacement(current, restored_trial, proposal);
                trial.draw(proposal, -1.0);
                let delta = trial.delta(model, current, quant_field, &cache, forbidden)?;
                let extra = bits - *incumbent_bits;
                (delta + crate::ac_strategy::RD_LAMBDA * margin(model.distance) * extra < 0.0)
                    .then_some(trial)
            });
        // Greedy pruning accepts the first improving proposal, not the best
        // of the batch. Later trials used stale geometry after this commit.
        for (index, (trial, proposal)) in trials.into_iter().zip(proposals).enumerate() {
            let Some(trial) = trial else { continue };
            let (simpler, proposal, bits) = proposal.unwrap();
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
            *incumbent_bits = bits;
            *restored = None;
            for by in b.1 / 8..=b.3 / 8 {
                cache[by * blocks_w..][b.0 / 8..=b.2 / 8].fill(None);
            }
            return Some(index);
        }
        None
    };
    for sp in kept {
        if sp.points.len() < 3 {
            continue;
        }
        let mut incumbent = RenderPlan::new(model.ctx, sp, QUANT_ADJUST, w, h).tiled();
        let mut incumbent_bits = prices.spline_bits(sp);
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
            try_points(
                sp,
                &mut incumbent,
                &mut incumbent_bits,
                &mut restored,
                vec![points],
            );
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
                if let Some(accepted) = try_points(
                    sp,
                    &mut incumbent,
                    &mut incumbent_bits,
                    &mut restored,
                    proposals,
                ) {
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

fn encode_dot_dictionary(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    references: &[crate::patches::PatchReference],
) -> BitWriter {
    let mut dictionary = BitWriter::new();
    crate::lossless::write_patch_dictionary_with_pool(
        references,
        false,
        scratch,
        &mut dictionary,
        Some(&ctx.thread_pool),
    );
    dictionary
}

fn encode_dot_plan(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    plan: &dots::DotPatches,
) -> BitWriter {
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
    atlas
}

/// Charges the selected dots their real atlas and dictionary bits, shared per
/// template. Placements on unprofitable templates can reuse a paying template
/// before being removed. Rechecks the retained residual against the original
/// image and returns its net gain and final RGB atlas and dictionary encoding.
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
        let set = || idx.iter().map(|&i| &kept[i]);
        let mut plan = dots::build_dot_patches_with_templates(set(), templates, false);
        let mut dictionary = encode_dot_dictionary(ctx, scratch, &plan.references);
        let mut dictionary_bits = dictionary.bits_written() as f32;
        let mut atlas_bits = DOT_TEMPLATE_BITS * plan.references.len() as f32;
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
            let mut atlas = encode_dot_plan(ctx, scratch, &plan);
            atlas_bits = atlas.bits_written() as f32;
            // Zero padding can help Modular prediction, so race the trimmed
            // layout only once for the final set instead of assuming it wins.
            let trimmed = dots::build_dot_patches_with_templates(set(), templates, true);
            let trimmed_atlas = encode_dot_plan(ctx, scratch, &trimmed);
            let trimmed_dictionary = encode_dot_dictionary(ctx, scratch, &trimmed.references);
            let trimmed_dictionary_bits = trimmed_dictionary.bits_written() as f32;
            let trimmed_bits = trimmed_atlas.bits_written() as f32;
            if trimmed_bits + trimmed_dictionary_bits < atlas_bits + dictionary_bits {
                plan = trimmed;
                atlas = trimmed_atlas;
                dictionary = trimmed_dictionary;
                atlas_bits = trimmed_bits;
                dictionary_bits = trimmed_dictionary_bits;
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
                    dictionary: Arc::new(dictionary),
                    references: plan.references,
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
                let mut trial = Trial::replacement(current, &restored, plan);
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
                let mut trial = Trial::replacement(current, &restored, &plan);
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

/// Groups whose accepted pieces already cover this share of the line are left
/// alone unless several pieces overlap on it.
const GROUP_FULL_COVERAGE: f32 = 0.9;
const GROUP_SLACK: i32 = 8;

/// Decoder curve samples of a kept line spline and its control-point box.
struct KeptCurve {
    positions: Vec<Point<f32>>,
    tangents: Vec<Point<f32>>,
    lo: Point<i32>,
    hi: Point<i32>,
}

impl KeptCurve {
    fn new(sp: &QuantizedSpline) -> Option<Self> {
        if dots::is_dot(sp) {
            return None;
        }
        let (positions, tangents) = curve_samples(&sp.points);
        if positions.len() < 2 {
            return None;
        }
        let (mut lo, mut hi) = (sp.points[0], sp.points[0]);
        for p in &sp.points {
            lo = Point::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Point::new(hi.x.max(p.x), hi.y.max(p.y));
        }
        Some(Self {
            positions,
            tangents,
            lo,
            hi,
        })
    }
}

fn has_chroma(sp: &QuantizedSpline) -> bool {
    sp.dct[0].iter().chain(&sp.dct[2]).any(|&v| v != 0)
}

/// Decides each collinear group as a unit once accepted splines lie on it:
/// keep the accepted pieces, replace them with one whole-line spline, or drop
/// them. A line coded in pieces is sharp where a piece is and blurred between,
/// and pieces overlapping along it subtract the line twice. The whole line
/// pays the pieces' own charge: it replaces a line already judged worth
/// coding. Color the pieces carry is never traded for a luma-only line.
/// Returns the indices removed from `kept`, ascending; replacements are
/// appended to `kept`.
#[allow(clippy::too_many_arguments)]
fn decide_groups(
    model: &BlockModel,
    scratch: &mut CoderScratch,
    current: &mut Image3F,
    quant_field: &[f32],
    groups: &[LineGroup],
    kept: &mut Vec<QuantizedSpline>,
    kept_price: &mut Vec<f32>,
    prices: &CostModel,
    margin: f32,
    forbidden: Option<&[bool]>,
    gain_bits: &mut f32,
) -> Vec<usize> {
    let lambda = crate::ac_strategy::RD_LAMBDA;
    let (w, h) = (current.xsize(), current.ysize());
    let mut curves: Vec<Option<KeptCurve>> = kept.iter().map(KeptCurve::new).collect();
    // Original index of each kept entry; appended replacements have none.
    let mut origin: Vec<Option<usize>> = (0..kept.len()).map(Some).collect();
    let mut removed = Vec::new();
    let mut cache = vec![None; w.div_ceil(8) * h.div_ceil(8)];
    let invalidate = |cache: &mut [Option<(f32, f32)>], plans: &[&TiledSpline]| {
        for plan in plans {
            for tile in &plan.tiles {
                cache[tile.cell] = None;
            }
        }
    };
    for group in groups {
        let index = LineIndex::new(&group.points, &unit_tangents(&group.points));
        let (mut lo, mut hi) = (group.points[0], group.points[0]);
        for p in &group.points {
            lo = Point::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Point::new(hi.x.max(p.x), hi.y.max(p.y));
        }
        let (lo, hi) = (
            Point::new(lo.x as i32 - GROUP_SLACK, lo.y as i32 - GROUP_SLACK),
            Point::new(hi.x as i32 + GROUP_SLACK, hi.y as i32 + GROUP_SLACK),
        );
        let mut members: Vec<usize> = curves
            .iter()
            .enumerate()
            .filter_map(|(i, curve)| {
                let curve = curve.as_ref()?;
                (curve.hi.x >= lo.x
                    && curve.lo.x <= hi.x
                    && curve.hi.y >= lo.y
                    && curve.lo.y <= hi.y
                    && index.holds(&curve.positions, &curve.tangents))
                .then_some(i)
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        // Pieces traced in a neighboring group of the same line lie on the
        // whole-line alternatives instead.
        let (mut alt_positions, mut alt_tangents) = (Vec::new(), Vec::new());
        for (i, alt) in group.alts.iter().enumerate() {
            if !group.alts[..i].iter().any(|prev| prev.points == alt.points) {
                let (positions, tangents) = curve_samples(&alt.points);
                alt_positions.extend(positions);
                alt_tangents.extend(tangents);
            }
        }
        let alt_index = LineIndex::new(&alt_positions, &alt_tangents);
        let (mut alt_lo, mut alt_hi) = (lo, hi);
        for p in group.alts.iter().flat_map(|alt| alt.points.iter()) {
            alt_lo = Point::new(
                alt_lo.x.min(p.x - GROUP_SLACK),
                alt_lo.y.min(p.y - GROUP_SLACK),
            );
            alt_hi = Point::new(
                alt_hi.x.max(p.x + GROUP_SLACK),
                alt_hi.y.max(p.y + GROUP_SLACK),
            );
        }
        for (i, curve) in curves.iter().enumerate() {
            if let Some(curve) = curve
                && !members.contains(&i)
                && curve.hi.x >= alt_lo.x
                && curve.lo.x <= alt_hi.x
                && curve.hi.y >= alt_lo.y
                && curve.lo.y <= alt_hi.y
                && alt_index.overlaps(&curve.positions, &curve.tangents)
            {
                members.push(i);
            }
        }
        members.sort_unstable();
        let member_points: Vec<Point<f32>> = members
            .iter()
            .flat_map(|&i| curves[i].as_ref().unwrap().positions.iter().copied())
            .collect();
        let member_tangents: Vec<Point<f32>> = members
            .iter()
            .flat_map(|&i| curves[i].as_ref().unwrap().tangents.iter().copied())
            .collect();
        let covered = LineIndex::new(&member_points, &member_tangents);
        let coverage =
            covered.count_near(index.points(), index.tangents()) as f32 / group.points.len() as f32;
        if coverage >= GROUP_FULL_COVERAGE && members.len() < 2 {
            continue;
        }
        let member_plans: Vec<TiledSpline> = members
            .iter()
            .map(|&i| RenderPlan::new(model.ctx, &kept[i], QUANT_ADJUST, w, h).tiled())
            .collect();
        let member_refs: Vec<&TiledSpline> = member_plans.iter().collect();
        let mut restore = Trial::new(current, &member_refs);
        for plan in &member_plans {
            restore.draw(plan, 1.0);
        }
        restore.commit(current);
        invalidate(&mut cache, &member_refs);
        let colored = members.iter().any(|&i| has_chroma(&kept[i]));
        let alt_plans = model
            .ctx
            .thread_pool
            .steal_map(scratch, group.alts.len(), |a, _| {
                let alt = &group.alts[a];
                if colored && !has_chroma(alt) {
                    return None;
                }
                let plan = RenderPlan::new(model.ctx, alt, QUANT_ADJUST, w, h).tiled();
                plan.bounds().map(|_| plan)
            });
        cache_blocks_parallel(
            model,
            scratch,
            current,
            quant_field,
            &mut cache,
            member_plans.iter().chain(alt_plans.iter().flatten()),
        );
        let mut subset = Trial::new(current, &member_refs);
        for plan in &member_plans {
            subset.draw(plan, -1.0);
        }
        let Some(d_subset) = subset.delta(model, current, quant_field, &cache, forbidden) else {
            subset.commit(current);
            invalidate(&mut cache, &member_refs);
            continue;
        };
        let j_subset = d_subset + lambda * members.iter().map(|&i| kept_price[i]).sum::<f32>();
        let (state, cached): (&Image3F, &[Option<(f32, f32)>]) = (current, &cache);
        let evaluations = model
            .ctx
            .thread_pool
            .steal_map(scratch, group.alts.len(), |a, _| {
                let plan = alt_plans[a].as_ref()?;
                let mut trial = Trial::new(state, &[plan]);
                trial.draw(plan, -1.0);
                let d = trial.delta(model, state, quant_field, cached, forbidden)?;
                Some((
                    d + lambda * margin * prices.spline_bits(&group.alts[a]),
                    trial,
                ))
            });
        let mut best: Option<(f32, usize, Trial)> = None;
        for (a, evaluation) in evaluations.into_iter().enumerate() {
            if let Some((j, trial)) = evaluation
                && best.as_ref().is_none_or(|(bj, _, _)| j < *bj)
            {
                best = Some((j, a, trial));
            }
        }
        if let Some((j, a, trial)) = best.filter(|&(j, _, _)| j < j_subset && j < 0.0) {
            trial.commit(current);
            invalidate(&mut cache, &[alt_plans[a].as_ref().unwrap()]);
            *gain_bits += (j_subset - j) / lambda;
            let alt = group.alts[a].clone();
            kept_price.push(margin * prices.spline_bits(&alt));
            curves.push(KeptCurve::new(&alt));
            kept.push(alt);
            origin.push(None);
        } else if j_subset >= 0.0 {
            *gain_bits += j_subset / lambda;
        } else {
            subset.commit(current);
            invalidate(&mut cache, &member_refs);
            continue;
        }
        for &i in members.iter().rev() {
            removed.extend(origin[i]);
            kept.remove(i);
            kept_price.remove(i);
            curves.remove(i);
            origin.remove(i);
        }
    }
    removed.sort_unstable();
    removed
}

/// Subtracts the accepted splines from `xyb` and returns them.
pub(super) fn rd_select(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    xyb: &mut Image3F,
    quant_field: &[f32],
    candidates: &[Candidate],
    groups: &[LineGroup],
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
    // All placements of a template share its price within each pass.
    let first_dot_price = dot_start.map(|bits| bits + DOT_FIRST_PASS_TEMPLATE_BITS);
    let line_price = |prices: &CostModel, candidate: &Candidate, alt: &QuantizedSpline| {
        margin * candidate.bits_factor * prices.spline_bits(alt)
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
                    floor: Option<f32>,
                    dot: Option<(&QuantizedSpline, &mut DotCosts)>| {
        let bounds = plan.bounds()?;
        if let Some(bits) = floor
            && let Some(saving) = dot_saving_bound(plan, cache)
            && saving < lambda * bits
        {
            // Preserve a rejecting score for first-pass alternatives so a
            // changed residual can trigger a second-pass trial.
            return Some(Evaluation {
                delta: -saving,
                trial: Trial { blocks: Vec::new() },
            });
        }
        let mut trial = Trial::drawn(current, plan);
        let delta = if let Some((alt, costs)) = dot {
            trial.delta_with_dot_costs(
                &model,
                current,
                quant_field,
                cache,
                forbidden,
                alt,
                bounds,
                costs,
            )?
        } else {
            trial.delta(&model, current, quant_field, cache, forbidden)?
        };
        Some(Evaluation { delta, trial })
    };
    let score_dot = |plan: &TiledSpline,
                     current: &Image3F,
                     cache: &[Option<(f32, f32)>],
                     floor: f32,
                     alt: &QuantizedSpline,
                     mut costs: Option<&mut DotCosts>| {
        let bounds = plan.bounds()?;
        let mut delta = 0.0;
        if let Some(saving) = dot_saving_bound(plan, cache)
            && saving < lambda * floor
        {
            delta = -saving;
        } else {
            if let Some(costs) = &mut costs {
                costs.prepare(alt, bounds);
            }
            let colors = [alt.dct[0][0], 0, alt.dct[2][0]];
            let mut cost = |pixels: &[[f32; 64]; 3], qac, bx, by| {
                if let Some(costs) = &mut costs {
                    costs.pixels_cost(&model, colors, pixels, qac, bx, by)
                } else {
                    model.pixels_cost(pixels, qac)
                }
            };
            for tile in &plan.tiles {
                let mut block = TrialBlock::load(current, tile.cell);
                plan.draw_tile(tile, &mut block.pixels, -1.0);
                block.add_delta(
                    &model,
                    current,
                    quant_field,
                    cache,
                    forbidden,
                    &mut cost,
                    &mut delta,
                )?;
            }
        }
        Some(delta)
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
    let mut deltas: Vec<Vec<Option<f32>>> = Vec::with_capacity(candidates.len());
    let mut first_choices = Vec::with_capacity(candidates.len());
    let mut occupied = vec![false; block_cache.len()];
    let mut changed = vec![false; block_cache.len()];
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
                row[index] = Some(dj);
                let alt = &candidate.alts[index];
                let bits = if dots::is_dot(alt) {
                    first_dot_price[usize::from(alt.dct[3][0] < 0)]
                } else {
                    line_price(&prices, candidate, alt)
                };
                let j = dj + lambda * bits;
                if best.is_none_or(|(bj, _, _)| j < bj) {
                    best = Some((j, index, b));
                }
            }
            let accepted = best.filter(|&(j, _, _)| j < 0.0);

            let chosen = accepted
                .map(|(_, index, b)| (index, b, Some(evaluations[index].take().unwrap().trial)));
            (row, chosen)
        };
        let threads = ctx.thread_pool.num_threads();
        let results = if next - start >= threads && candidates[start..next].iter().all(|c| c.dot) {
            // Color alternatives of a shape share unchanged channel costs.
            // Keep them on one worker and keep just their scores for pass two.
            ctx.thread_pool
                .steal_map(scratch, next - start, |offset, _| {
                    let i = start + offset;
                    let candidate = &candidates[i];
                    let plans = &prepared[i];
                    let floor = dot_floor(i).unwrap();
                    let mut costs = DotCosts::default();
                    let mut best: Option<(f32, usize)> = None;
                    let row = plans
                        .iter()
                        .zip(&candidate.alts)
                        .enumerate()
                        .map(|(index, (plan, alt))| {
                            let delta = score_dot(
                                plan,
                                &current,
                                &block_cache,
                                floor,
                                alt,
                                Some(&mut costs),
                            )?;
                            let bits = first_dot_price[usize::from(alt.dct[3][0] < 0)];
                            let score = delta + lambda * bits;
                            if best.is_none_or(|(previous, _)| score < previous) {
                                best = Some((score, index));
                            }
                            Some(delta)
                        })
                        .collect();
                    let chosen = best
                        .filter(|&(score, _)| score < 0.0)
                        .map(|(_, index)| (index, plans[index].bounds().unwrap(), None));
                    (row, chosen)
                })
        } else if next - start == 1 && !candidates[start].dot {
            let plans = &prepared[start];
            let evaluations = ctx.thread_pool.steal_map(scratch, plans.len(), |i, _| {
                evaluate(&plans[i], &current, &block_cache, None, None)
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
                    if let Some(floor) = dot_floor(candidate) {
                        score_dot(
                            &prepared[candidate][alt],
                            &current,
                            &block_cache,
                            floor,
                            &candidates[candidate].alts[alt],
                            None,
                        )
                        .map(|delta| Evaluation {
                            delta,
                            trial: Trial { blocks: Vec::new() },
                        })
                    } else {
                        evaluate(
                            &prepared[candidate][alt],
                            &current,
                            &block_cache,
                            None,
                            None,
                        )
                    }
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
                if candidates[start + i].dot {
                    prepared[start + i][index].draw(&mut current, -1.0);
                } else {
                    trial.unwrap().commit(&mut current);
                }
                invalidate(&mut block_cache, b);
                for by in b.1 / 8..=b.3 / 8 {
                    changed[by * blocks_w..][b.0 / 8..=b.2 / 8].fill(true);
                }
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
    // Cached costs for blocks untouched in the first pass still describe
    // the original pixels. Discard only costs that may reflect its commits.
    for (cost, modified) in block_cache.iter_mut().zip(&mut changed) {
        if *modified {
            *cost = None;
            *modified = false;
        }
    }
    prices = CostModel::prior(0.1);
    let first_pass = || {
        candidates
            .iter()
            .zip(&first_choices)
            .filter_map(|(candidate, &choice)| choice.map(|index| &candidate.alts[index]))
    };
    let dot_prices = {
        let mut counts = HashMap::new();
        let mut total = [0usize; 2];
        for sp in first_pass().filter(|sp| dots::is_dot(sp)) {
            *counts.entry(dots::template_key(sp)).or_insert(0usize) += 1;
            total[usize::from(sp.dct[3][0] < 0)] += 1;
        }
        if counts.is_empty() {
            HashMap::new()
        } else {
            keys.into_iter()
                .map(|key| {
                    let dark = usize::from(key.0 < 0);
                    let n = counts.get(&key).copied().unwrap_or(0).max(1) as f32;
                    let template =
                        dirty_log2f(total[dark].max(1) as f32 / n) + DOT_TEMPLATE_BITS / n;
                    (key, dot_start[dark] + template)
                })
                .collect()
        }
    };
    let price = |prices: &CostModel, candidate: &Candidate, alt: &QuantizedSpline| {
        if dots::is_dot(alt) {
            if dot_prices.is_empty() {
                first_dot_price[usize::from(alt.dct[3][0] < 0)]
            } else {
                dot_prices[&dots::template_key(alt)]
            }
        } else {
            line_price(prices, candidate, alt)
        }
    };
    for sp in first_pass().filter(|sp| !dots::is_dot(sp)) {
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
        let dot_batch = candidates[batch.clone()].iter().all(|c| c.dot);
        let recomputations: Vec<Vec<bool>> = batch
            .clone()
            .map(|i| {
                deltas[i]
                    .iter()
                    .zip(&prepared[i])
                    .map(|(cached, plan)| {
                        cached.is_some_and(|_| {
                            let b = plan.bounds().unwrap();
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
        let choose_dot =
            |i: usize, needed: &[bool], current: &Image3F, cache: &[Option<(f32, f32)>]| {
                let candidate = &candidates[i];
                let mut costs = DotCosts::default();
                let mut best: Option<(f32, usize, PixelBox, Option<Trial>)> = None;
                for (index, (((plan, alt), &needed), cached)) in prepared[i]
                    .iter()
                    .zip(&candidate.alts)
                    .zip(needed)
                    .zip(&deltas[i])
                    .enumerate()
                {
                    let Some(mut delta) = *cached else {
                        continue;
                    };
                    let bounds = plan.bounds().unwrap();
                    let bits = price(&prices, candidate, alt);
                    let trial = if needed {
                        let floor = dot_floor(i).unwrap();
                        // Alternatives are visited in their tie-breaking order.
                        // Once a dot pays, a bound equal to its score also loses.
                        if let Some((score, ..)) = best
                            && score < 0.0
                            && let Some(saving) = dot_saving_bound(plan, cache)
                            && -saving + lambda * bits >= score
                        {
                            continue;
                        }
                        let Some(evaluation) = evaluate(
                            plan,
                            current,
                            cache,
                            Some(bits.max(floor)),
                            Some((alt, &mut costs)),
                        ) else {
                            continue;
                        };
                        delta = evaluation.delta;
                        Some(evaluation.trial)
                    } else {
                        None
                    };
                    let score = delta + lambda * bits;
                    if best.as_ref().is_none_or(|(previous, ..)| score < *previous) {
                        best = Some((score, index, bounds, trial));
                    }
                }
                best.filter(|(score, ..)| *score < 0.0)
            };
        let mut dot_choices = if dot_batch && parallel_batch {
            ctx.thread_pool
                .steal_map(scratch, batch.len(), |offset, _| {
                    choose_dot(
                        batch.start + offset,
                        &recomputations[offset],
                        &current,
                        &block_cache,
                    )
                })
        } else {
            Vec::new()
        }
        .into_iter();
        let mut batched_evaluations = if !dot_batch && parallel_batch {
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
                evaluate(
                    &prepared[i][alt],
                    &current,
                    &block_cache,
                    dot_floor(i),
                    None,
                )
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
            let accepted = if dot_batch && parallel_batch {
                dot_choices.next().unwrap()
            } else if dot_batch {
                choose_dot(i, &recompute, &current, &block_cache)
            } else {
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
                } else if recompute.iter().any(|&needed| needed) {
                    ctx.thread_pool.steal_map(scratch, plans.len(), |i, _| {
                        recompute[i]
                            .then(|| evaluate(&plans[i], &current, &block_cache, floor, None))
                            .flatten()
                    })
                } else {
                    std::iter::repeat_with(|| None).take(plans.len()).collect()
                };
                let mut best: Option<(f32, usize, PixelBox)> = None;
                for (index, (alt, cached)) in candidate.alts.iter().zip(row).enumerate() {
                    let Some(mut dj) = *cached else { continue };
                    let b = plans[index].bounds().unwrap();
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
                best.filter(|&(j, _, _)| j < 0.0).map(|(j, index, b)| {
                    (
                        j,
                        index,
                        b,
                        evaluations[index].take().map(|eval| eval.trial),
                    )
                })
            };
            let choice = accepted.as_ref().map(|(_, index, _, _)| *index);
            if first != choice {
                for index in first.into_iter().chain(choice) {
                    let b = plans[index].bounds().unwrap();
                    for by in b.1 / 8..=b.3 / 8 {
                        changed[by * blocks_w..][b.0 / 8..=b.2 / 8].fill(true);
                    }
                }
            }
            if let Some((j, index, b, trial)) = accepted {
                if let Some(trial) = trial {
                    trial.commit(&mut current);
                } else if candidate.dot {
                    // A cached score needs only the winning pixels. The dot
                    // renderer applies the same pointwise addition in place.
                    plans[index].draw(&mut current, -1.0);
                } else {
                    Trial::drawn(&current, &plans[index]).commit(&mut current);
                }
                invalidate(&mut block_cache, b);
                gain_bits -= j / lambda;
                kept_raw.push(-j / lambda + price(&prices, candidate, &candidate.alts[index]));
                kept_candidates.push(i);
                kept.push(candidate.alts[index].clone());
            }
        }
    }
    drop(prepared);
    drop(deltas);
    drop(block_cache);
    drop(changed);
    drop(first_choices);
    if !groups.is_empty() && !kept.is_empty() {
        let mut kept_price: Vec<f32> = kept
            .iter()
            .zip(&kept_candidates)
            .map(|(sp, &i)| price(&prices, &candidates[i], sp))
            .collect();
        let removed = decide_groups(
            &model,
            scratch,
            &mut current,
            quant_field,
            groups,
            &mut kept,
            &mut kept_price,
            &prices,
            margin,
            forbidden,
            &mut gain_bits,
        );
        // Replacements are lines, never dots: only the dot path reads these.
        for i in removed.into_iter().rev() {
            kept_raw.remove(i);
            kept_candidates.remove(i);
        }
        kept_raw.resize(kept.len(), 0.0);
        kept_candidates.resize(kept.len(), usize::MAX);
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
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            2.0,
            1,
        );
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
                let encoding = encoding.unwrap();
                assert_eq!(encoding.references.len(), 1);
                let mut cached = BitWriter::new();
                cached.write(3, 0b101);
                let mut fresh = cached.clone();
                cached.append_bits(&encoding.dictionary);
                crate::lossless::write_patch_dictionary(
                    &encoding.references,
                    false,
                    &mut CoderScratch::default(),
                    &mut fresh,
                );
                assert_eq!(cached.bits_written(), fresh.bits_written());
                for writer in [&mut cached, &mut fresh] {
                    writer.write(7, 0b1011011);
                    writer.zero_pad_to_byte();
                }
                assert_eq!(cached.into_bytes(), fresh.into_bytes());
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
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            2.0,
            1,
        );
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
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            3.0,
            1,
        );
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
        let mut trial = Trial::replacement(&current, &restored, &b);
        let union = Trial::new(&current, &[&a, &b]);
        assert_eq!(trial.blocks.len(), union.blocks.len());
        for (actual, original) in trial.blocks.iter().zip(&union.blocks) {
            assert_eq!(actual.cell, original.cell);
            let expected = restored
                .blocks
                .binary_search_by_key(&actual.cell, |block| block.cell)
                .map_or(original, |index| &restored.blocks[index]);
            assert_eq!(actual.pixels, expected.pixels);
        }
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
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                XybMatrix::SPEC,
                3.0,
                threads,
            );
            let mut residual = image.clone();
            let selected = rd_select(
                &ctx,
                &mut CoderScratch::default(),
                3.0,
                &mut residual,
                &quant,
                candidates,
                &[],
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
                let ctx = EncodingContext::new(
                    Speed::Slow,
                    crate::Arbiter::Ssimulacra2,
                    XybMatrix::SPEC,
                    3.0,
                    threads,
                );
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
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            3.0,
            1,
        );
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
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            3.0,
            1,
        );
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

    fn group_fixture(
        line_dct: impl Fn(&mut [[i32; 32]; 4]),
        pieces: &[(i32, i32)],
        whole_alt: impl Fn(&mut [[i32; 32]; 4]),
    ) -> (SplineSet, QuantizedSpline) {
        let kernels = crate::encoding_context::EncodingContext::default();
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            3.0,
            1,
        );
        let line = |x0: i32, x1: i32, color: &dyn Fn(&mut [[i32; 32]; 4])| {
            let mut sp = QuantizedSpline {
                points: vec![Point::new(x0, 32), Point::new(x1, 32)],
                dct: [[0; 32]; 4],
            };
            color(&mut sp.dct);
            sp.dct[3][0] = 4;
            sp
        };
        let mut image = Image3F::new(256, 64);
        render_spline(
            &kernels,
            &line(16, 240, &line_dct),
            QUANT_ADJUST,
            &mut image,
            1.0,
        );
        let candidates: Vec<Candidate> = pieces
            .iter()
            .map(|&(x0, x1)| Candidate {
                alts: vec![line(x0, x1, &line_dct)],
                bits_factor: 1.0,
                dot: false,
            })
            .collect();
        let whole = line(16, 240, &whole_alt);
        let groups = [LineGroup {
            points: (16..=240).map(|x| Point::new(x as f32, 32.0)).collect(),
            alts: vec![whole.clone()],
        }];
        let selected = rd_select(
            &ctx,
            &mut CoderScratch::default(),
            3.0,
            &mut image,
            &vec![8.0; 32 * 8],
            &candidates,
            &groups,
            None,
        )
        .unwrap();
        (selected, whole)
    }

    #[test]
    fn a_line_accepted_in_part_is_coded_whole() {
        let luma = |dct: &mut [[i32; 32]; 4]| dct[1][0] = 5;
        let (selected, whole) = group_fixture(luma, &[(16, 96)], luma);
        assert_eq!(selected.splines.len(), 1);
        assert_eq!(selected.splines[0].points, whole.points);
    }

    #[test]
    fn overlapping_pieces_of_a_line_become_one_spline() {
        let luma = |dct: &mut [[i32; 32]; 4]| dct[1][0] = 5;
        let (selected, whole) = group_fixture(luma, &[(16, 140), (110, 240)], luma);
        assert_eq!(selected.splines.len(), 1);
        assert_eq!(selected.splines[0].points, whole.points);
    }

    #[test]
    fn colored_pieces_are_not_replaced_by_a_luma_only_line() {
        let colored = |dct: &mut [[i32; 32]; 4]| {
            dct[1][0] = 5;
            dct[2][0] = 6;
        };
        let luma = |dct: &mut [[i32; 32]; 4]| dct[1][0] = 5;
        let (selected, whole) = group_fixture(colored, &[(16, 96)], luma);
        assert!(!selected.splines.is_empty());
        assert!(selected.splines.iter().all(|sp| sp.points != whole.points));
        assert!(selected.splines.iter().all(|sp| sp.dct[2][0] != 0));
    }

    #[test]
    fn repricing_does_not_accept_duplicate_lines_against_a_stale_residual() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            3.0,
            1,
        );
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
            &[],
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
    fn dot_color_alternatives_reuse_exact_plane_costs_at_partial_blocks() {
        let ctx = EncodingContext::default();
        let model = BlockModel::new(&ctx, 4.0);
        let (w, h) = (30, 27);
        let mut image = Image3F::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for (x, pixel) in image.plane_row_mut(c, y).iter_mut().enumerate() {
                    *pixel = ((x * 43 + y * 23 + c * 37) % 131) as f32 * 0.002 - 0.1;
                }
            }
        }
        let quant = [2.0; 16];
        let mut costs = DotCosts::default();
        for center in [Point::new(27, 25), Point::new(14, 12)] {
            costs.clear();
            for (n, y, x, b) in [(4, 10, 0, 0), (4, 10, 2, 0), (4, 10, 2, 1), (3, 10, 0, 0)] {
                let mut alt = QuantizedSpline {
                    points: vec![center],
                    dct: [[0; 32]; 4],
                };
                alt.dct[0][0] = x;
                alt.dct[1][0] = y;
                alt.dct[2][0] = b;
                alt.dct[3][0] = n;
                let plan = RenderPlan::new(&ctx, &alt, QUANT_ADJUST, w, h).tiled();
                let mut exact = Trial::drawn(&image, &plan);
                let mut cached = Trial::drawn(&image, &plan);
                let expected = exact.delta(&model, &image, &quant, &[], None);
                let actual = cached.delta_with_dot_costs(
                    &model,
                    &image,
                    &quant,
                    &[],
                    None,
                    &alt,
                    plan.bounds().unwrap(),
                    &mut costs,
                );
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));

                let mut committed = image.clone();
                exact.commit(&mut committed);
                let mut drawn = image.clone();
                plan.draw(&mut drawn, -1.0);
                for c in 0..3 {
                    assert!(
                        drawn
                            .plane_data(c)
                            .iter()
                            .zip(committed.plane_data(c))
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
    }

    #[test]
    fn dot_first_pass_matches_across_candidate_and_alternative_jobs() {
        let (w, h) = (259, 131);
        let mut image = Image3F::new(w, h);
        let kernels = EncodingContext::default();
        let candidates: Vec<_> = (0..24)
            .map(|i| {
                let x = 16 + i % 8 * 32;
                let y = 16 + i / 8 * 32;
                let mut dot = QuantizedSpline {
                    points: vec![Point::new(x, y)],
                    dct: [[0; 32]; 4],
                };
                dot.dct[1][0] = 22;
                dot.dct[3][0] = 5;
                dot.dct[2][0] = 2;
                RenderPlan::new(&kernels, &dot, QUANT_ADJUST, w, h).draw(&mut image, 1.0);
                let mut gray = dot.clone();
                gray.dct[2][0] = 0;
                let mut wider = dot.clone();
                wider.dct[3][0] += 1;
                Candidate {
                    alts: vec![gray, dot, wider],
                    bits_factor: 1.0,
                    dot: true,
                }
            })
            .collect();
        // The duplicate forces a singleton batch before the disjoint dots.
        let duplicate = Candidate {
            alts: candidates[0].alts.clone(),
            bits_factor: 1.0,
            dot: true,
        };
        let candidates: Vec<_> = std::iter::once(duplicate).chain(candidates).collect();
        let quant = vec![8.0; w.div_ceil(8) * h.div_ceil(8)];
        let select = |threads| {
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                XybMatrix::SPEC,
                2.0,
                threads,
            );
            let mut residual = image.clone();
            let selected = rd_select(
                &ctx,
                &mut CoderScratch::default(),
                2.0,
                &mut residual,
                &quant,
                &candidates,
                &[],
                None,
            )
            .unwrap();
            (selected, residual)
        };
        let (serial, serial_residual) = select(1);
        let (parallel, parallel_residual) = select(4);
        assert_eq!(serial.splines.len(), 24);
        assert_eq!(serial.splines.len(), parallel.splines.len());
        for (a, b) in serial.splines.iter().zip(&parallel.splines) {
            assert_eq!(a.points, b.points);
            assert_eq!(a.dct, b.dct);
        }
        for c in 0..3 {
            assert!(
                serial_residual
                    .plane_data(c)
                    .iter()
                    .zip(parallel_residual.plane_data(c))
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    #[test]
    fn blue_only_changes_respect_forbidden_blocks() {
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            3.0,
            1,
        );
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
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                XybMatrix::SPEC,
                distance,
                1,
            );
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
        let ctx = EncodingContext::new(
            Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            XybMatrix::SPEC,
            3.0,
            1,
        );
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
                    let cell = by * w.div_ceil(8) + bx;
                    let block = TrialBlock::load(&img, cell);
                    for (c, pixels) in block.pixels.iter().enumerate() {
                        let mut coeff = [0.0; 64];
                        (ctx.dct8x8)(DctInput::from_flat(pixels), &mut coeff);
                        for qac in [0.375, 0.75, 2.0] {
                            assert_eq!(
                                model.plane_cost(c, pixels, qac),
                                crate::inflated_cost::channel_rd(
                                    ctx.sse_and_rate,
                                    ctx.rate_log2_lut,
                                    &coeff,
                                    model.inv[c],
                                    c,
                                    ctx.dequant_bias(c),
                                    qac,
                                    model.qm[c],
                                    model.distance,
                                    1,
                                    1,
                                ),
                            );
                        }
                    }
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
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                XybMatrix::SPEC,
                3.0,
                1,
            );
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
                let ctx = EncodingContext::new(
                    Speed::Slow,
                    crate::Arbiter::Ssimulacra2,
                    XybMatrix::SPEC,
                    3.0,
                    threads,
                );
                let mut actual_image = image.clone();
                let actual = rd_select(
                    &ctx,
                    &mut CoderScratch::default(),
                    3.0,
                    &mut actual_image,
                    &quant,
                    &candidates,
                    &[],
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
