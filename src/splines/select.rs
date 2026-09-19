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
//! Alternative parametrisations of a candidate compete; the spline side is
//! priced with an adaptive token-cost model refined over two passes.

use super::fit::Candidate;
use super::{
    CTX_DCT, CTX_NUM_POINTS, CTX_POINTS, NUM_SPLINE_CONTEXTS, PixelBox, Point, QUANT_ADJUST,
    QuantizedSpline, SplineSet, render_spline, spline_tokens,
};
use crate::adaptive_quant::dirty_log2f;
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
}

fn copy_box(src: &Image3F, dst: &mut Image3F, b: PixelBox) {
    for c in 0..3 {
        for y in b.1..=b.3 {
            dst.plane_row_mut(c, y)[b.0..=b.2].copy_from_slice(&src.plane_row(c, y)[b.0..=b.2]);
        }
    }
}

fn block_touched(a: &Image3F, b: &Image3F, bx: usize, by: usize) -> bool {
    let (w, h) = (a.xsize(), a.ysize());
    for y in by * 8..(by * 8 + 8).min(h) {
        let (ay, ty) = (a.plane_row(1, y), b.plane_row(1, y));
        let (ax, tx) = (a.plane_row(0, y), b.plane_row(0, y));
        let xs = bx * 8..(bx * 8 + 8).min(w);
        if ay[xs.clone()]
            .iter()
            .zip(&ty[xs.clone()])
            .zip(ax[xs.clone()].iter().zip(&tx[xs]))
            .any(|((&ay, &ty), (&ax, &tx))| (ay - ty).abs() > 1e-5 || (ax - tx).abs() > 1e-6)
        {
            return true;
        }
    }
    false
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
    let Some(b) = render_spline(&shifted, QUANT_ADJUST, &mut trial, -1.0) else {
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
    distance: f32,
    xyb: &mut Image3F,
    quant_field: &[f32],
    candidates: &[Candidate],
    forbidden: Option<&[bool]>,
) -> Option<SplineSet> {
    let lambda = crate::ac_strategy::RD_LAMBDA;
    let margin = margin(distance);
    let matrices = crate::quant_weights::DequantMatrices::new(distance);
    let model = BlockModel {
        ctx,
        inv: [
            matrices.inv_matrix(0),
            matrices.inv_matrix(1),
            matrices.inv_matrix(2),
        ],
        qm: [1.0, 1.0, ctx.b_qm_mul()],
        distance,
    };
    let blocks_w = xyb.xsize().div_ceil(8);

    // Pass one walks the candidates greedily against the running residual and
    // records every alternative's VarDCT cost change; pass two only re-prices
    // the spline side from what pass one kept and decides again.
    let mut prices = CostModel::prior(1.0);
    let mut current = xyb.clone();
    let mut trial = xyb.clone();
    let blocks_h = xyb.ysize().div_ceil(8);
    let mut block_cache: Vec<Option<(f32, f32)>> = vec![None; blocks_w * blocks_h];
    let mut deltas: Vec<Vec<Option<f32>>> = Vec::with_capacity(candidates.len());
    let mut first_pass: Vec<QuantizedSpline> = Vec::new();
    for candidate in candidates {
        let mut row = vec![None; candidate.alts.len()];
        let mut best: Option<(f32, usize, PixelBox)> = None;
        for (index, alt) in candidate.alts.iter().enumerate() {
            let Some(b) = render_spline(alt, QUANT_ADJUST, &mut trial, -1.0) else {
                continue;
            };
            let mut dj = 0f32;
            let mut blocked = false;
            'blocks: for by in b.1 / 8..=b.3 / 8 {
                for bx in b.0 / 8..=b.2 / 8 {
                    if !block_touched(&current, &trial, bx, by) {
                        continue;
                    }
                    let cell = by * blocks_w + bx;
                    if forbidden.is_some_and(|mask| mask[cell]) {
                        blocked = true;
                        break 'blocks;
                    }
                    let qac = quant_field[cell];
                    let (d0, r0) =
                        *block_cache[cell].get_or_insert_with(|| model.cost(&current, bx, by, qac));
                    let (d1, r1) = model.cost(&trial, bx, by, qac);
                    dj += (d1 - d0) + lambda * (r1 - r0);
                }
            }
            copy_box(&current, &mut trial, b);
            if blocked {
                continue;
            }
            row[index] = Some(dj);
            let j = dj + lambda * margin * candidate.bits_factor * prices.spline_bits(alt);
            if best.is_none_or(|(bj, _, _)| j < bj) {
                best = Some((j, index, b));
            }
        }
        if let Some((j, index, b)) = best
            && j < 0.0
        {
            let alt = &candidate.alts[index];
            render_spline(alt, QUANT_ADJUST, &mut current, -1.0);
            copy_box(&current, &mut trial, b);
            for by in b.1 / 8..=b.3 / 8 {
                block_cache[by * blocks_w..][b.0 / 8..=b.2 / 8].fill(None);
            }
            first_pass.push(alt.clone());
        }
        deltas.push(row);
    }
    drop((current, trial));

    prices = CostModel::prior(0.1);
    for sp in &first_pass {
        for (c, v) in spline_tokens(sp) {
            prices.add(c, v, 1.0);
        }
    }
    let mut kept: Vec<QuantizedSpline> = Vec::new();
    let mut gain_bits = 0f32;
    for (candidate, row) in candidates.iter().zip(&deltas) {
        let best = candidate
            .alts
            .iter()
            .zip(row)
            .filter_map(|(alt, dj)| {
                Some((
                    dj.as_ref()? + lambda * margin * candidate.bits_factor * prices.spline_bits(alt),
                    alt,
                ))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((j, alt)) = best
            && j < 0.0
        {
            gain_bits -= j / lambda;
            kept.push(alt.clone());
        }
    }
    if kept.is_empty() || gain_bits < MIN_TOTAL_GAIN_BITS {
        return None;
    }
    for sp in &kept {
        render_spline(sp, QUANT_ADJUST, xyb, -1.0);
    }
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
