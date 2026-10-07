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
//! Rate of a candidate transform under the frame's own statistics.
//!
//! The fixed rate model charges a block by its nonzero count and magnitudes,
//! the same for every image. The entropy coder does not: it codes each token
//! by the distribution the frame gives its context, and walks each block in
//! an order derived from the frame. Where many blocks carry the same values
//! at the same positions the two differ by an order of magnitude, and by a
//! different amount for every transform, so the selector compares
//! candidates on prices that have little to do with their coded size.
//!
//! [`RatePrices`] prices a candidate the way it will be coded. Every
//! selectable transform is applied to a sample of the blocks that a pilot
//! selection gave it, under their quant field and chroma-from-luma slopes;
//! the sample yields the transform's scan order and the price of every
//! token in every context. A candidate is then quantized and walked like
//! the coder walks it: the nonzero count, then the coefficients up to the
//! last nonzero.
//!
//! Each transform keeps its own statistics and order although the coder
//! shares contexts and orders between some of them: shared statistics
//! describe a mixture that no frame settling on one transform has.

use super::*;
use crate::ac_context::{
    K_NON_ZERO_BUCKETS, K_NUM_FINE_BLOCK_CTXS, K_ZERO_DENSITY_CONTEXT_COUNT, fine_non_zero_context,
    zero_density_context, zero_density_context_8x8,
};
use crate::adaptive_quant::dirty_log2p1f;
use crate::coeff_order::{CoeffOrders, OrderStats, derive_orders, order_slot_of};
use crate::dc_group_data::{NUM_STRATEGIES, STRATEGY_CODE_LUT};
use crate::entropy::{pack_signed, uint_encode};
use std::cell::RefCell;
use std::sync::Arc;

/// Token symbols priced individually; rarer ones share the last price.
const SYMBOLS: usize = 64;
/// Contexts of one channel and quant class of one transform.
const CONTEXTS: usize = K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT;
const QUANT_CLASSES: usize = 2;
const ROWS: usize = 3 * QUANT_CLASSES * CONTEXTS;

/// Sampled blocks per transform, by the 8x8 blocks it covers.
const fn sample_budget(covered_blocks: usize) -> usize {
    match covered_blocks {
        1 => 1024,
        2 => 512,
        4 => 256,
        8 => 128,
        16 => 64,
        _ => 32,
    }
}

/// A transform whose samples cover fewer 8x8 blocks keeps the fixed rate
/// model.
const MIN_SAMPLED_BLOCKS: usize = 16;
/// And so does one with fewer samples, which would price its own blocks.
const MIN_SAMPLES: usize = 4;

#[inline]
fn too_few(samples: usize, covered_blocks: usize) -> bool {
    samples < MIN_SAMPLES || samples * covered_blocks < MIN_SAMPLED_BLOCKS
}
/// Samples the pooled distribution of a context's kind counts for in that
/// context: a context with few samples is priced mostly by the pool, one
/// with many by itself.
const POOL_WEIGHT: f32 = 64.0;
/// Pseudo-count of every symbol in the pool, so that none is free of charge
/// or impossible.
const POOL_FLOOR: f32 = 0.5;
/// 8x8 blocks the any-block sample counts for in the table of a transform
/// learned from the blocks the pilot gave it.
const PRIOR_BLOCKS: f32 = 256.0;
/// Blocks of that sample.
const PRIOR_SAMPLES: usize = 256;

struct Table {
    bits: Vec<f32>,
    /// Context to stored row. Empty contexts of the same pool share a row.
    rows: Vec<u16>,
    orders: CoeffOrders,
    /// Bits of a channel without nonzero coefficients, by channel and
    /// quant class.
    empty: [[f32; QUANT_CLASSES]; 3],
}

/// One transform's table and, for DCT8, the bits of its sampled blocks
/// under the fixed model and under the table.
struct Learned {
    table: Table,
    fixed: f64,
    learned: f64,
}

pub(crate) struct RatePrices {
    tables: Vec<Option<Table>>,
    qf_threshold: u32,
    inv_scale: f32,
    /// Learned bits to the scale of the fixed model, which the lambda and
    /// the merge constants are fitted to: the two agree on the group's
    /// sampled DCT8 blocks.
    norm: f32,
}

thread_local! {
    static LEVELS: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn token_symbol(value: u32) -> (usize, f32) {
    let (symbol, nbits, _) = uint_encode(value);
    ((symbol as usize).min(SYMBOLS - 1), nbits as f32)
}

/// The tokens the coder emits for one channel of one block, as (context
/// within the transform's own context set, value). The nonzero count is
/// predicted by the block's own, the neighbors being unknown.
fn walk_block(
    orders: &CoeffOrders,
    strategy: u8,
    c: usize,
    block: &[i32],
    cx: usize,
    cy: usize,
    mut emit: impl FnMut(usize, u32),
) {
    let size = cx * cy * 64;
    let covered = cx * cy;
    let log2_covered = covered.trailing_zeros() as usize;
    let width = cx * 8;
    // Count one contiguous span so the reduction can vectorize across rows.
    // Subtract LLF entries separately, retaining their original exclusion even
    // for inputs whose low frequencies have not already been cleared.
    let values = &block[..size];
    let mut nzeros = values.iter().filter(|&&q| q != 0).count();
    for row in values.chunks_exact(width).take(cy) {
        nzeros -= row[..cx].iter().filter(|&&q| q != 0).count();
    }
    let predicted = ((nzeros + covered - 1) >> log2_covered).min(32) as u32;
    emit(
        fine_non_zero_context(predicted, 0) as usize / K_NUM_FINE_BLOCK_CTXS,
        nzeros as u32,
    );
    let scan = orders.scan_for(STRATEGY_CODE_LUT[strategy as usize], c);
    let mut prev = usize::from(nzeros <= size / 16);
    let mut remaining = nzeros;
    let mut k = covered;
    while k < size && remaining != 0 {
        let coef = block[scan[k] as usize];
        let context = if covered == 1 {
            zero_density_context_8x8(remaining, k, prev)
        } else {
            zero_density_context(remaining, k, covered, log2_covered, prev)
        };
        emit(K_NON_ZERO_BUCKETS + context, pack_signed(coef));
        prev = usize::from(coef != 0);
        remaining -= usize::from(coef != 0);
        k += 1;
    }
}

// Original row-wise count and token walk, retained as an exact test oracle.
#[cfg(test)]
fn walk_block_reference(
    orders: &CoeffOrders,
    strategy: u8,
    c: usize,
    block: &[i32],
    cx: usize,
    cy: usize,
    mut emit: impl FnMut(usize, u32),
) {
    let size = cx * cy * 64;
    let covered = cx * cy;
    let log2_covered = covered.trailing_zeros() as usize;
    let width = cx * 8;
    let mut nzeros = 0usize;
    for (v, row) in block[..size].chunks_exact(width).enumerate() {
        let skip = if v < cy { cx } else { 0 };
        nzeros += row[skip..].iter().filter(|&&q| q != 0).count();
    }
    let predicted = ((nzeros + covered - 1) >> log2_covered).min(32) as u32;
    emit(
        fine_non_zero_context(predicted, 0) as usize / K_NUM_FINE_BLOCK_CTXS,
        nzeros as u32,
    );
    let scan = orders.scan_for(STRATEGY_CODE_LUT[strategy as usize], c);
    let mut prev = usize::from(nzeros <= size / 16);
    let mut remaining = nzeros;
    let mut k = covered;
    while k < size && remaining != 0 {
        let coef = block[scan[k] as usize];
        let context = if covered == 1 {
            zero_density_context_8x8(remaining, k, prev)
        } else {
            zero_density_context(remaining, k, covered, log2_covered, prev)
        };
        emit(K_NON_ZERO_BUCKETS + context, pack_signed(coef));
        prev = usize::from(coef != 0);
        remaining -= usize::from(coef != 0);
        k += 1;
    }
}

fn inverse_matrix_of(ctx: &EncodingContext, strategy: u8, c: usize) -> &[f32] {
    match strategy {
        STRATEGY_DCT64X64 => &ctx.matrices().inv_matrix_64x64(c)[..],
        STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => &ctx.matrices().inv_matrix_64x32(c)[..],
        _ => inverse_matrix_for(ctx, strategy, c),
    }
}

/// The quantizer of the rate kernels (`sse_and_rate`).
#[allow(clippy::too_many_arguments)]
fn quantize_channel(
    ctx: &EncodingContext,
    strategy: u8,
    c: usize,
    coeffs: &[f32],
    qac: f32,
    qm_mult_x: f32,
    distance: f32,
    cx: usize,
    cy: usize,
    out: &mut [i32],
) {
    let qm = match c {
        0 => qm_mult_x,
        2 => ctx.b_qm_mul(),
        _ => 1.0,
    };
    (ctx.quantize_block_ac)(
        coeffs,
        c,
        inverse_matrix_of(ctx, strategy, c),
        1,
        qac,
        qm,
        distance,
        cx,
        cy,
        out,
    );
    // The lowest frequencies travel with the DC.
    for out in out.chunks_exact_mut(cx * 8).take(cy) {
        out[..cx].fill(0);
    }
}

/// Bits of the fixed rate model for one quantized channel.
fn fixed_model_bits(block: &[i32], cx: usize, cy: usize) -> f32 {
    let (width, height) = (cx * 8, cy * 8);
    let scan_pos = crate::coeff_order::scan_pos_lut(width, height);
    let (mut nzeros, mut mag_bits, mut max_scan) = (0usize, 0f32, 0u32);
    for (i, &q) in block[..width * height].iter().enumerate() {
        if q != 0 {
            nzeros += 1;
            mag_bits += dirty_log2p1f(q.unsigned_abs() as f32);
            max_scan = max_scan.max(scan_pos[i]);
        }
    }
    crate::inflated_cost::model_bits(nzeros, mag_bits, max_scan, cx, cy)
}

/// Forward transform of all three channels with chroma-from-luma applied.
fn transform_block(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    cmap: [f32; 3],
) -> (usize, usize, usize) {
    let CoderScratch {
        strategy_coeffs: coeffs,
        transform_gather,
        ..
    } = scratch;
    prepare_strategy_coeffs(ctx, coeffs, transform_gather, strategy, opsin, px, py, cmap)
}

/// The transforms the selector can place at this speed and distance.
fn selectable(speed: crate::Speed, distance: f32) -> impl Iterator<Item = u8> {
    let merges = [
        STRATEGY_DCT,
        STRATEGY_DCT16X8,
        STRATEGY_DCT8X16,
        STRATEGY_DCT16X16,
        STRATEGY_DCT32X16,
        STRATEGY_DCT16X32,
        STRATEGY_DCT32X32,
    ];
    let fine = [STRATEGY_IDENTITY, STRATEGY_DCT2X2];
    let sub8 = [
        STRATEGY_DCT4X4,
        STRATEGY_DCT4X8,
        STRATEGY_DCT8X4,
        STRATEGY_AFV0,
        STRATEGY_AFV1,
        STRATEGY_AFV2,
        STRATEGY_AFV3,
    ];
    let family64 = [STRATEGY_DCT64X64, STRATEGY_DCT64X32, STRATEGY_DCT32X64];
    merges
        .into_iter()
        .chain(
            fine.into_iter()
                .filter(move |_| distance <= FINE_TRANSFORM_MAX_DISTANCE),
        )
        .chain(
            sub8.into_iter()
                .filter(move |_| distance <= AFV_MAX_DISTANCE.max(SUB8_MAX_DISTANCE)),
        )
        .chain(
            family64
                .into_iter()
                .filter(move |_| use_dct64(speed, distance)),
        )
}

/// Prices of the contexts of one kind from their counts, each context
/// smoothed by the distribution of all of them together.
fn price_contexts(counts: &mut [f32]) {
    let mut pool = [POOL_FLOOR; SYMBOLS];
    let (counts, _) = counts.as_chunks_mut::<SYMBOLS>();
    for context in counts.iter() {
        for (pooled, &count) in pool.iter_mut().zip(context) {
            *pooled += count;
        }
    }
    let pool_total: f32 = pool.iter().sum();
    for pooled in &mut pool {
        *pooled *= POOL_WEIGHT / pool_total;
    }
    // Most contexts never occur; they share one smoothed distribution.
    let empty: [f32; SYMBOLS] = std::array::from_fn(|i| (POOL_WEIGHT / pool[i]).log2());
    // The pool and each row total are known before that row is overwritten,
    // so the count allocation can become the final price table.
    for row in counts {
        if row.iter().all(|&count| count == 0.0) {
            *row = empty;
            continue;
        }
        let total = row.iter().sum::<f32>() + POOL_WEIGHT;
        for (count, &pooled) in row.iter_mut().zip(&pool) {
            *count = (total / (*count + pooled)).log2();
        }
    }
}

/// Compact in place after pricing so pooling and floating-point arithmetic
/// are unchanged. Only empty rows within the same channel/class/kind share.
fn compact_price_rows(bits: &mut Vec<f32>, empty: &[bool]) -> Vec<u16> {
    let mut pooled_rows = [None; 3 * QUANT_CLASSES * 2];
    let mut rows = Vec::with_capacity(ROWS);
    let mut stored = 0;
    for (row, &empty) in empty.iter().enumerate() {
        let pool = (row / CONTEXTS) * 2 + usize::from(row % CONTEXTS >= K_NON_ZERO_BUCKETS);
        if empty && let Some(index) = pooled_rows[pool] {
            rows.push(index);
            continue;
        }
        let index = stored as u16;
        if empty {
            pooled_rows[pool] = Some(index);
        }
        bits.copy_within(row * SYMBOLS..(row + 1) * SYMBOLS, stored * SYMBOLS);
        rows.push(index);
        stored += 1;
    }
    bits.truncate(stored * SYMBOLS);
    bits.shrink_to_fit();
    rows
}

/// The `index`-th of `count` samples out of `total` raster positions: one
/// per stratum, at an offset that does not follow the image's period.
#[inline]
fn sample_position(index: usize, count: usize, total: usize) -> usize {
    let start = index * total / count;
    let end = (index + 1) * total / count;
    let offset = (index as u32).wrapping_mul(0x9e37_79b9) >> 16;
    start + offset as usize % (end - start).max(1)
}

/// Coded/learned bits of a large transform relative to DCT8, at d = 1.5,
/// 2.5 and 4.75 (linear between, flat outside). The learned prices overcharge
/// large transforms once quantization coarsens.
fn learned_calibration(strategy: u8, distance: f32) -> f32 {
    let row: [f32; 3] = match strategy {
        STRATEGY_DCT16X16 => [0.98, 0.99, 0.92],
        STRATEGY_DCT32X32 | STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => [1.0, 0.93, 0.83],
        STRATEGY_DCT64X64 | STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => [0.98, 0.88, 0.74],
        _ => return 1.0,
    };
    if distance <= 1.5 {
        row[0]
    } else if distance <= 2.5 {
        fmla(distance - 1.5, row[1] - row[0], row[0])
    } else if distance <= 4.75 {
        fmla((distance - 2.5) / 2.25, row[2] - row[1], row[1])
    } else {
        row[2]
    }
}

impl RatePrices {
    pub(crate) fn has_table(&self, strategy: u8) -> bool {
        self.tables[strategy as usize].is_some()
    }

    /// Score coefficient distortion and learned rate in one quantization pass.
    /// The kernel preserves the distortion scorer's rounding and reduction;
    /// the saved levels use the coder's rounding for the learned token walk.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn coefficient_dist_and_rate(
        &self,
        ctx: &EncodingContext,
        strategy: u8,
        coeffs: &[[f32; 4096]; 3],
        qac: f32,
        qm_mult_x: f32,
        distance: f32,
        cx: usize,
        cy: usize,
    ) -> Option<(f32, f32)> {
        let table = self.tables[strategy as usize].as_ref()?;
        let (width, height) = (cx * 8, cy * 8);
        let size = width * height;
        let qf_hi = (qac * self.inv_scale).round() as u32 > self.qf_threshold;
        Some(LEVELS.with_borrow_mut(|levels| {
            if levels.len() < size {
                levels.resize(4096, 0);
            }
            let mut distortion = 0.0f32;
            let mut bits = 0.0f32;
            for (c, coeffs) in coeffs.iter().enumerate() {
                let qm = match c {
                    0 => qm_mult_x,
                    2 => ctx.b_qm_mul(),
                    _ => 1.0,
                };
                let thresholds =
                    crate::group::quantize_ac_thresholds_scaled(c, cx, cy, distance, qm);
                let (d, nonzero, _, _) = unsafe {
                    (ctx.sse_and_quantize)(
                        &coeffs[..size],
                        inverse_matrix_of(ctx, strategy, c),
                        qac * qm,
                        width,
                        height,
                        width / 2,
                        cx,
                        cy,
                        ctx.rate_log2_lut,
                        &thresholds,
                        &[],
                        &mut levels[..size],
                    )
                };
                distortion += ctx.channel_weight(c) * d;
                // Keep the old scorer's empty-channel shortcut even on targets
                // where its tie rounding differs from the coding quantizer.
                bits += if nonzero == 0 {
                    table.empty[c][usize::from(qf_hi)]
                } else {
                    Self::channel_bits(table, strategy, c, &levels[..size], cx, cy, qf_hi)
                };
            }
            (
                distortion,
                bits * self.norm * learned_calibration(strategy, distance),
            )
        }))
    }

    /// Bits of one quantized channel of one block, unnormalized.
    fn channel_bits(
        table: &Table,
        strategy: u8,
        c: usize,
        block: &[i32],
        cx: usize,
        cy: usize,
        qf_hi: bool,
    ) -> f32 {
        let base = (c * QUANT_CLASSES + usize::from(qf_hi)) * CONTEXTS;
        let mut bits = 0f32;
        walk_block(
            &table.orders,
            strategy,
            c,
            block,
            cx,
            cy,
            |context, value| {
                let (symbol, extra) = token_symbol(value);
                bits +=
                    table.bits[usize::from(table.rows[base + context]) * SYMBOLS + symbol] + extra;
            },
        );
        bits
    }

    /// Rate of a candidate whose coefficients are in `coeffs`, or `None`
    /// for a transform without learned prices. A channel the caller's rate
    /// kernel found without nonzero coefficients is `empty`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rate(
        &self,
        ctx: &EncodingContext,
        strategy: u8,
        coeffs: &[[f32; 4096]; 3],
        empty: [bool; 3],
        qac: f32,
        qm_mult_x: f32,
        distance: f32,
        cx: usize,
        cy: usize,
    ) -> Option<f32> {
        let table = self.tables[strategy as usize].as_ref()?;
        let size = cx * cy * 64;
        let qf_hi = (qac * self.inv_scale).round() as u32 > self.qf_threshold;
        let bits = LEVELS.with_borrow_mut(|levels| {
            if levels.len() < size {
                levels.resize(4096, 0);
            }
            let mut bits = 0f32;
            for (c, coeffs) in coeffs.iter().enumerate() {
                if empty[c] {
                    bits += table.empty[c][usize::from(qf_hi)];
                    continue;
                }
                quantize_channel(
                    ctx,
                    strategy,
                    c,
                    &coeffs[..size],
                    qac,
                    qm_mult_x,
                    distance,
                    cx,
                    cy,
                    &mut levels[..size],
                );
                bits += Self::channel_bits(table, strategy, c, &levels[..size], cx, cy, qf_hi);
            }
            bits
        });
        Some(bits * self.norm * learned_calibration(strategy, distance))
    }

    /// One transform's table and the bits of its sampled blocks.
    #[allow(clippy::too_many_arguments)]
    fn learn_transform(
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        strategy: u8,
        opsin: &Image3F,
        origin: (usize, usize),
        (xsize, ysize): (usize, usize),
        quant_field: &ImageB,
        cfl: (&ImageSB, &ImageSB),
        scale: f32,
        qm_mult_x: f32,
        distance: f32,
        qf_threshold: u32,
        layout: Option<&AcStrategyImage>,
    ) -> Option<Learned> {
        let cx = AcStrategyImage::covered_blocks_x_of(strategy);
        let cy = AcStrategyImage::covered_blocks_y_of(strategy);
        let (columns, rows) = (xsize / cx, ysize / cy);
        let mut chosen: Vec<(usize, usize)> = Vec::new();
        if let Some(layout) = layout {
            chosen.extend(
                layout
                    .iter_first_blocks()
                    .filter(|&(bx, by, s)| s == strategy && bx + cx <= xsize && by + cy <= ysize)
                    .map(|(bx, by, _)| (bx, by)),
            );
        }
        if too_few(chosen.len(), cx * cy) {
            chosen.clear();
        }
        let candidates = if chosen.is_empty() {
            columns * rows
        } else {
            chosen.len()
        };
        let samples = candidates.min(sample_budget(cx * cy));
        if too_few(samples, cx * cy) {
            return None;
        }
        let slot = order_slot_of(STRATEGY_CODE_LUT[strategy as usize]);
        let mut stats = OrderStats::new();
        let size = cx * cy * 64;
        // Quantized levels of the block at `(bx, by)`, its coefficient
        // dimensions and its quant class.
        let quantize = |scratch: &mut CoderScratch, bx: usize, by: usize, block: &mut [i32]| {
            let quant = quant_field.row(by)[bx];
            let (pcx, pcy, _) = transform_block(
                ctx,
                scratch,
                strategy,
                opsin,
                origin.0 + bx * 8,
                origin.1 + by * 8,
                cmap_factors(ctx.cfl_frame(), cfl.0, cfl.1, bx, by),
            );
            for (c, block) in block.chunks_exact_mut(size).enumerate() {
                quantize_channel(
                    ctx,
                    strategy,
                    c,
                    &scratch.strategy_coeffs[c][..size],
                    scale * quant as f32,
                    qm_mult_x,
                    distance,
                    pcx,
                    pcy,
                    block,
                );
            }
            (pcx, pcy, quant as u32 > qf_threshold)
        };
        let any_block = |position: usize| (position % columns * cx, position / columns * cy);
        let mut levels = vec![0i32; samples * 3 * size];
        let mut classes: Vec<bool> = Vec::with_capacity(samples);
        let (mut pcx, mut pcy) = (cx, cy);
        for (index, block) in levels.chunks_exact_mut(3 * size).enumerate() {
            let position = sample_position(index, samples, candidates);
            let (bx, by) = if chosen.is_empty() {
                any_block(position)
            } else {
                chosen[position]
            };
            let qf_hi;
            (pcx, pcy, qf_hi) = quantize(scratch, bx, by, block);
            classes.push(qf_hi);
            let Some(slot) = slot else {
                continue;
            };
            stats.tally_block(slot);
            for (c, block) in block.chunks_exact(size).enumerate() {
                for (raw, _) in block.iter().enumerate().filter(|&(_, &q)| q != 0) {
                    stats.tally(slot, c, raw);
                }
            }
        }
        let mut orders = CoeffOrders::natural();
        let _ = derive_orders(&stats, &mut orders);
        let mut counts = vec![0f32; ROWS * SYMBOLS];
        let mut tally = |block: &[i32], qf_hi: bool, weight: f32| {
            for (c, block) in block.chunks_exact(size).enumerate() {
                let base = (c * QUANT_CLASSES + usize::from(qf_hi)) * CONTEXTS;
                walk_block(&orders, strategy, c, block, pcx, pcy, |context, value| {
                    counts[(base + context) * SYMBOLS + token_symbol(value).0] += weight;
                });
            }
        };
        for (block, &qf_hi) in levels.chunks_exact(3 * size).zip(&classes) {
            tally(block, qf_hi, 1.0);
        }
        if !chosen.is_empty() {
            // The blocks the pilot gave the transform say nothing of the
            // values it would code elsewhere; a sample of any block stands
            // in for them, at the weight of a fixed area.
            let total = columns * rows;
            let prior = total.min(PRIOR_SAMPLES).min(sample_budget(cx * cy));
            let weight = PRIOR_BLOCKS / (cx * cy * prior) as f32;
            let mut block = vec![0i32; 3 * size];
            for index in 0..prior {
                let (bx, by) = any_block(sample_position(index, prior, total));
                let (_, _, qf_hi) = quantize(scratch, bx, by, &mut block);
                tally(&block, qf_hi, weight);
            }
        }
        let empty_rows: Vec<bool> = counts
            .as_chunks::<SYMBOLS>()
            .0
            .iter()
            .map(|row| row.iter().all(|&count| count == 0.0))
            .collect();
        let mut bits = counts;
        let (slots, _) = bits.as_chunks_mut::<{ CONTEXTS * SYMBOLS }>();
        for prices in slots {
            // The nonzero counts and the coefficients are different
            // alphabets; each kind pools its own contexts.
            let split = K_NON_ZERO_BUCKETS * SYMBOLS;
            price_contexts(&mut prices[..split]);
            price_contexts(&mut prices[split..]);
        }
        let rows = compact_price_rows(&mut bits, &empty_rows);
        let mut table = Table {
            bits,
            rows,
            orders,
            empty: [[0.0; QUANT_CLASSES]; 3],
        };
        let zeros = vec![0i32; size];
        for (c, empty) in table.empty.clone().iter().enumerate() {
            for qf_hi in 0..empty.len() {
                table.empty[c][qf_hi] =
                    Self::channel_bits(&table, strategy, c, &zeros, pcx, pcy, qf_hi != 0);
            }
        }
        let (mut fixed, mut learned) = (0f64, 0f64);
        if strategy == STRATEGY_DCT {
            for (block, &qf_hi) in levels.chunks_exact(3 * size).zip(&classes) {
                for (c, block) in block.chunks_exact(size).enumerate() {
                    fixed += fixed_model_bits(block, pcx, pcy) as f64;
                    learned +=
                        Self::channel_bits(&table, strategy, c, block, pcx, pcy, qf_hi) as f64;
                }
            }
        }
        Some(Learned {
            table,
            fixed,
            learned,
        })
    }

    /// Learn the prices of the blocks whose top-left pixel is `(px0, py0)`.
    /// With a `layout`, a transform is sampled on the blocks that chose it
    /// there, with a lighter sample of any block behind them, and on any
    /// block alone where too few did.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn learn(
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        opsin: &Image3F,
        px0: usize,
        py0: usize,
        quant_field: &ImageB,
        ytox_map: &ImageSB,
        ytob_map: &ImageSB,
        scale: f32,
        qm_mult_x: f32,
        distance: f32,
        layout: Option<&AcStrategyImage>,
        num_threads: usize,
    ) -> Arc<Self> {
        let xsize = quant_field.xsize().min((opsin.xsize() - px0) / 8);
        let ysize = quant_field.ysize().min((opsin.ysize() - py0) / 8);
        let qf_threshold = {
            let mut histogram = [0u32; 256];
            for y in 0..ysize {
                for &q in &quant_field.row(y)[..xsize] {
                    histogram[q as usize] += 1;
                }
            }
            let half = (xsize * ysize / 2) as u32;
            let mut below = 0u32;
            histogram
                .iter()
                .position(|&count| {
                    below += count;
                    below > half
                })
                .unwrap_or(0) as u32
        };
        let strategies: Vec<u8> = selectable(ctx.speed, distance).collect();
        let learned = ctx.thread_pool.steal_map_with_threads(
            scratch,
            strategies.len(),
            num_threads,
            |index, scratch| {
                Self::learn_transform(
                    ctx,
                    scratch,
                    strategies[index],
                    opsin,
                    (px0, py0),
                    (xsize, ysize),
                    quant_field,
                    (ytox_map, ytob_map),
                    scale,
                    qm_mult_x,
                    distance,
                    qf_threshold,
                    layout,
                )
            },
        );
        let mut tables: Vec<Option<Table>> = (0..NUM_STRATEGIES).map(|_| None).collect();
        let mut norm = 1.0;
        for (&strategy, learned) in strategies.iter().zip(learned) {
            let Some(learned) = learned else {
                continue;
            };
            if strategy == STRATEGY_DCT && learned.fixed > 1.0 && learned.learned > 1.0 {
                norm = (learned.fixed / learned.learned) as f32;
            }
            tables[strategy as usize] = Some(learned.table);
        }
        Arc::new(Self {
            tables,
            qf_threshold,
            inv_scale: 1.0 / scale,
            norm,
        })
    }
}

/// Installs a DC group's prices on a worker's scratch for as long as the
/// worker selects that group's transforms, then puts back what was there:
/// the thread that owns the group also runs its bands.
pub(crate) struct RatePricesScope<'a> {
    scratch: &'a mut CoderScratch,
    previous: Option<Arc<RatePrices>>,
}

impl<'a> RatePricesScope<'a> {
    pub(crate) fn new(scratch: &'a mut CoderScratch, prices: Option<Arc<RatePrices>>) -> Self {
        let previous = std::mem::replace(&mut scratch.rate_prices, prices);
        Self { scratch, previous }
    }
}

impl std::ops::Deref for RatePricesScope<'_> {
    type Target = CoderScratch;

    fn deref(&self) -> &CoderScratch {
        self.scratch
    }
}

impl std::ops::DerefMut for RatePricesScope<'_> {
    fn deref_mut(&mut self) -> &mut CoderScratch {
        self.scratch
    }
}

impl Drop for RatePricesScope<'_> {
    fn drop(&mut self) {
        self.scratch.rate_prices = self.previous.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_nonzero_count_preserves_every_learned_rate_token() {
        let mut rng = 73u32;
        let natural = CoeffOrders::natural();
        let mut learned = CoeffOrders::natural();
        let mut stats = OrderStats::new();
        for (slot, &(_, _, size)) in crate::coeff_order::ORDER_SPECS.iter().enumerate() {
            for _ in 0..10_000 {
                stats.tally_block(slot);
                for c in 0..3 {
                    stats.tally(slot, c, size - 1);
                }
            }
        }
        derive_orders(&stats, &mut learned);
        for strategy in selectable(crate::Speed::Slow, 1.0) {
            let x = AcStrategyImage::covered_blocks_x_of(strategy);
            let y = AcStrategyImage::covered_blocks_y_of(strategy);
            let (cx, cy) = (x.max(y), x.min(y));
            let size = cx * cy * 64;
            for case in 0..32 {
                let block: Vec<i32> = (0..size + 5)
                    .map(|i| {
                        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                        match case % 4 {
                            0 => 0,
                            1 => {
                                if i % 17 == 0 {
                                    1
                                } else {
                                    0
                                }
                            }
                            2 => (rng as i32) % 7,
                            _ => rng as i32,
                        }
                    })
                    .collect();
                // Keep nonzero LF and trailing sentinels: both exclusions are
                // part of the original block walk, independent of quantization.
                for orders in [&natural, &learned] {
                    for c in 0..3 {
                        let mut expected = Vec::new();
                        let mut actual = Vec::new();
                        walk_block_reference(
                            orders,
                            strategy,
                            c,
                            &block,
                            cx,
                            cy,
                            |context, value| expected.push((context, value)),
                        );
                        walk_block(orders, strategy, c, &block, cx, cy, |context, value| {
                            actual.push((context, value))
                        });
                        assert_eq!(
                            actual, expected,
                            "strategy {strategy}, channel {c}, case {case}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dct64_views_preserve_gathered_coefficients_and_edge_replication() {
        let ctx = EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, 2.0, 1);
        let mut image = Image3F::new(131, 137);
        for c in 0..3 {
            for (i, v) in image.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *v = ((i * 7919 + c * 97) % 104729) as f32 / 104729.0;
            }
        }
        for (strategy, width, height, cx, cy) in [
            (STRATEGY_DCT64X64, 64, 64, 8, 8),
            (STRATEGY_DCT64X32, 32, 64, 8, 4),
            (STRATEGY_DCT32X64, 64, 32, 8, 4),
        ] {
            for (px, py) in [(0, 0), (3, 5), (67, 73), (120, 5), (3, 130), (129, 135)] {
                for c in 0..3 {
                    let mut gathered = [0.0; 4096];
                    gather_pixels(image.plane(c), px, py, width, height, &mut gathered);
                    let mut expected = [0.0; 4096];
                    match strategy {
                        STRATEGY_DCT64X64 => {
                            (ctx.dct64x64)(DctInput::from_flat(&gathered), &mut expected)
                        }
                        STRATEGY_DCT64X32 => (ctx.dct64x32)(
                            DctInput::from_flat(gathered.first_chunk::<2048>().unwrap()),
                            expected.first_chunk_mut::<2048>().unwrap(),
                        ),
                        _ => (ctx.dct32x64)(
                            DctInput::from_flat(gathered.first_chunk::<2048>().unwrap()),
                            expected.first_chunk_mut::<2048>().unwrap(),
                        ),
                    }
                    let mut tmp = [123.0; 4096];
                    let mut actual = [0.0; 4096];
                    assert_eq!(
                        forward_transform(
                            &ctx,
                            &mut tmp,
                            strategy,
                            image.plane(c),
                            px,
                            py,
                            &mut actual
                        ),
                        (cx, cy)
                    );
                    for (a, e) in actual.iter().zip(expected) {
                        assert_eq!(
                            a.to_bits(),
                            e.to_bits(),
                            "strategy {strategy}, ({px}, {py})"
                        );
                    }
                    if px + width <= image.xsize() && py + height <= image.ysize() {
                        assert_eq!(tmp, [123.0; 4096], "interior transforms must not gather");
                    }
                }
            }
        }
    }

    /// A 4-pixel checkerboard under a slow brightness drift.
    fn checkerboard(size: usize) -> Image3F {
        let mut opsin = Image3F::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let cell = ((x / 4 + y / 4) % 2) as f32;
                let v = (0.1 + 0.5 * cell) * (1.0 - 0.1 * x as f32 / size as f32);
                opsin.plane_row_mut(0, y)[x] = 0.02 * cell;
                opsin.plane_row_mut(1, y)[x] = v;
                opsin.plane_row_mut(2, y)[x] = v;
            }
        }
        opsin
    }

    fn noise(size: usize) -> Image3F {
        let mut opsin = Image3F::new(size, size);
        let mut state = 0x1234_5678u32;
        for c in 0..3 {
            for y in 0..size {
                for x in 0..size {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    opsin.plane_row_mut(c, y)[x] = (state >> 24) as f32 / 255.0 * 0.5;
                }
            }
        }
        opsin
    }

    fn learn(opsin: &Image3F, distance: f32) -> Arc<RatePrices> {
        let ctx =
            EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, distance, 1);
        let blocks = opsin.xsize() / 8;
        let quant_field = ImageB::new_fill(blocks, blocks, 5);
        let maps = ImageSB::new_fill(blocks.div_ceil(8), blocks.div_ceil(8), 0);
        let mut scratch = CoderScratch::default();
        RatePrices::learn(
            &ctx,
            &mut scratch,
            opsin,
            0,
            0,
            &quant_field,
            &maps,
            &maps,
            0.04,
            1.0,
            distance,
            None,
            1,
        )
    }

    #[test]
    fn repeated_blocks_cost_a_fraction_of_the_fixed_model() {
        // The norm is the fixed model's bits over the learned bits of the
        // sampled DCT8 blocks.
        let periodic = learn(&checkerboard(256), 2.0);
        assert!(periodic.norm > 2.0, "{}", periodic.norm);
        let random = learn(&noise(256), 2.0);
        assert!(random.norm < periodic.norm * 0.5, "{}", random.norm);
    }

    #[test]
    fn a_layout_restricts_a_transform_to_its_blocks() {
        // Left half: the checkerboard. Right half: noise.
        let (pattern, random) = (checkerboard(256), noise(256));
        let mut opsin = Image3F::new(256, 256);
        for c in 0..3 {
            for y in 0..256 {
                let row = opsin.plane_row_mut(c, y);
                row[..128].copy_from_slice(&pattern.plane_row(c, y)[..128]);
                row[128..256].copy_from_slice(&random.plane_row(c, y)[128..256]);
            }
        }
        // DCT8 took the noise; 16x16 took the pattern.
        let mut layout = AcStrategyImage::new(32, 32);
        for by in (0..32).step_by(2) {
            for bx in (0..16).step_by(2) {
                layout.set_first(bx, by, STRATEGY_DCT16X16);
            }
        }
        let learn = |layout: Option<&AcStrategyImage>| {
            let ctx = EncodingContext::new(crate::Speed::Slow, crate::xyb::XybMatrix::SPEC, 2.0, 1);
            let quant_field = ImageB::new_fill(32, 32, 5);
            let maps = ImageSB::new_fill(4, 4, 0);
            let mut scratch = CoderScratch::default();
            RatePrices::learn(
                &ctx,
                &mut scratch,
                &opsin,
                0,
                0,
                &quant_field,
                &maps,
                &maps,
                0.04,
                1.0,
                2.0,
                layout,
                1,
            )
        };
        // The norm compares the two models on the sampled DCT8 blocks: on
        // noise alone they agree better than with the pattern mixed in.
        let (everywhere, chosen) = (learn(None), learn(Some(&layout)));
        assert!(
            chosen.norm < everywhere.norm * 0.95,
            "{} vs {}",
            chosen.norm,
            everywhere.norm
        );
    }

    #[test]
    fn only_selectable_transforms_are_priced() {
        let prices = learn(&checkerboard(256), 3.0);
        for strategy in [
            STRATEGY_DCT,
            STRATEGY_DCT16X16,
            STRATEGY_DCT32X32,
            STRATEGY_IDENTITY,
        ] {
            assert!(prices.tables[strategy as usize].is_some(), "{strategy}");
        }
        // Sub-8 transforms stop at d = 1.5.
        for strategy in [STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_AFV0] {
            assert!(prices.tables[strategy as usize].is_none(), "{strategy}");
        }
        let prices = learn(&checkerboard(256), 1.0);
        assert!(prices.tables[STRATEGY_DCT4X4 as usize].is_some());
    }

    #[test]
    fn compact_prices_preserve_every_context_and_pool() {
        let mut dense = vec![0.0f32; ROWS * SYMBOLS];
        for (row, counts) in dense.as_chunks_mut::<SYMBOLS>().0.iter_mut().enumerate() {
            if row % 7 == 0 {
                counts[row % SYMBOLS] = row as f32 + 0.25;
            }
        }
        let empty: Vec<bool> = dense
            .as_chunks::<SYMBOLS>()
            .0
            .iter()
            .map(|row| row.iter().all(|&count| count == 0.0))
            .collect();
        for slot in dense.as_chunks_mut::<{ CONTEXTS * SYMBOLS }>().0 {
            let split = K_NON_ZERO_BUCKETS * SYMBOLS;
            price_contexts(&mut slot[..split]);
            price_contexts(&mut slot[split..]);
        }
        let mut compact = dense.clone();
        let rows = compact_price_rows(&mut compact, &empty);
        assert!(compact.len() < dense.len() / 2);
        for (row, &stored) in rows.iter().enumerate() {
            let stored = usize::from(stored) * SYMBOLS;
            for symbol in 0..SYMBOLS {
                assert_eq!(
                    dense[row * SYMBOLS + symbol].to_bits(),
                    compact[stored + symbol].to_bits()
                );
            }
        }
    }

    #[test]
    fn in_place_prices_match_separate_count_and_price_buffers() {
        let mut counts = vec![0f32; 8 * SYMBOLS];
        for (row, counts) in counts.as_chunks_mut::<SYMBOLS>().0.iter_mut().enumerate() {
            if row % 3 == 0 {
                continue;
            }
            for (symbol, count) in counts.iter_mut().enumerate() {
                *count = ((row * 7919 + symbol * 104729) % 97) as f32 * 0.25;
            }
        }
        let mut pool = [POOL_FLOOR; SYMBOLS];
        for row in counts.as_chunks::<SYMBOLS>().0 {
            for (pooled, count) in pool.iter_mut().zip(row) {
                *pooled += count;
            }
        }
        let pool_total: f32 = pool.iter().sum();
        for pooled in &mut pool {
            *pooled *= POOL_WEIGHT / pool_total;
        }
        let mut expected = Vec::new();
        for row in counts.as_chunks::<SYMBOLS>().0 {
            let total = row.iter().sum::<f32>() + POOL_WEIGHT;
            expected.extend(
                row.iter()
                    .zip(pool)
                    .map(|(count, pooled)| (total / (count + pooled)).log2().to_bits()),
            );
        }
        price_contexts(&mut counts);
        assert_eq!(
            expected,
            counts.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn contexts_are_priced_by_their_counts_and_the_pool() {
        let mut counts = vec![0f32; 3 * SYMBOLS];
        counts[0] = 900.0;
        counts[1] = 100.0;
        counts[SYMBOLS] = 3.0;
        let mut prices = counts;
        price_contexts(&mut prices);
        // A well sampled context is priced by itself.
        assert!((prices[0] - (1000.0f32 / 900.0).log2()).abs() < 0.05);
        assert!((prices[1] - (1000.0f32 / 100.0).log2()).abs() < 0.2);
        // Seen is cheaper than unseen, and nothing is free or impossible.
        assert!(prices[2] > prices[1]);
        assert!(prices.iter().all(|&p| p > 0.0 && p.is_finite()));
        // A thin context leans on the pool: a symbol it never saw but the
        // pool did is cheaper than a symbol nobody saw.
        assert!(prices[SYMBOLS + 1] < prices[SYMBOLS + 2]);
        // An empty context is the pool, a distribution.
        let pooled = &prices[2 * SYMBOLS..];
        assert!(pooled[0] < pooled[1] && pooled[1] < pooled[2]);
        let mass: f32 = pooled.iter().map(|&p| 0.5f32.powf(p)).sum();
        assert!((mass - 1.0).abs() < 1e-3, "{mass}");
    }

    #[test]
    fn samples_are_distinct_and_cover_the_range() {
        for (count, total) in [(16usize, 16usize), (64, 1000), (2048, 6144), (1, 7)] {
            let positions: Vec<usize> = (0..count)
                .map(|index| sample_position(index, count, total))
                .collect();
            assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(*positions.last().unwrap() < total);
            assert!(positions[0] < total.div_ceil(count));
        }
    }

    #[test]
    fn a_scope_puts_back_what_it_replaced() {
        let outer = learn(&checkerboard(64), 2.0);
        let inner = learn(&checkerboard(64), 3.0);
        let mut scratch = CoderScratch::default();
        {
            let mut scratch = RatePricesScope::new(&mut scratch, Some(outer.clone()));
            {
                let scratch = RatePricesScope::new(&mut scratch, Some(inner.clone()));
                assert!(Arc::ptr_eq(scratch.rate_prices.as_ref().unwrap(), &inner));
            }
            assert!(Arc::ptr_eq(scratch.rate_prices.as_ref().unwrap(), &outer));
        }
        assert!(scratch.rate_prices.is_none());
    }
}
