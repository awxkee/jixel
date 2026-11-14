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

//! Decoder-feedback coefficient polish of the ExtraSlow closed loop.
//!
//! After the loop has settled the quant field and chroma scales, every block's
//! quantized levels are revisited one coefficient at a time: restore a zeroed
//! coefficient, step a level toward or away from zero. Each proposal is priced
//! exactly under the frozen token prices and verified against the frame itself:
//! the pixel delta of the move (a Y move also moves the decoded X and B through
//! the tile's CfL factors) is applied to the frame reconstruction and scored by
//! the incremental SSIMULACRA2 state and, for the best SSIMULACRA2-accepted
//! proposals, the incremental ColorVideoVDP arbiter. A proposal is kept only
//! when both metrics improve at their own exchange rates (deficit per bit from
//! the loop's uniform quant-field shifts). Without a ColorVideoVDP exchange
//! rate (near transparency the shifts do not separate) SSIMULACRA2 decides.
//!
//! The trellis's quantizer-unit SSE does not predict these decisions (sign
//! agreement 0.65 with the metric change; no block-local feature does better),
//! which is why the pass verifies against the metrics instead of a proxy.

use crate::ac_context::{zero_density_context, zero_density_context_8x8};
use crate::cvvdp_arbiter::{CvvdpArbiter, CvvdpCheckpoint};
use crate::dc_group_data::{STRATEGY_AFV0, STRATEGY_AFV3, STRATEGY_DCT4X4, STRATEGY_DCT8X4};
use crate::encoding_context::EncodingContext;
use crate::entropy::{FrozenTokenPrices, Token, pack_signed};
use crate::group::{DequantBias, SMALL_LEVELS};
use crate::image::Image3F;
use crate::simd::scaled_add::{AddScaledFn, selected_add_scaled_fn};
use crate::ssimulacra2::{Rect, Ssimulacra2Checkpoint, Ssimulacra2State};
use std::sync::Arc;

/// Proposals verified per channel and round, best trellis proxy first.
const PER_CHANNEL: usize = 4;
/// Greedy rounds per block.
const ROUNDS: usize = 2;
/// Proposals costing more bits than this are not verified: restores that
/// expensive are accepted a few percent of the time.
const MAX_COST_BITS: f32 = 2.0;
/// SSIMULACRA2-ranked proposals the ColorVideoVDP judge checks per round.
const CV_TRIES: usize = 2;
/// A zeroed coefficient is a restore candidate from this unquantized level.
const RESTORE_MIN: f32 = 0.25;
/// After the best move is committed, this many of the round's other
/// SSIMULACRA2-accepted proposals are re-verified on the changed block.
const EXTRA_TRIES: usize = 4;
/// Scan positions examined per channel.
const MAX_POSITIONS: usize = 256;
/// Distortion-ranked proposals priced per channel.
const PRICE_CAP: usize = PER_CHANNEL * 6;

/// Pass counter: a polisher left in a worker's scratch from an earlier pass
/// is inert and gets rebuilt (see `CoefPolisher::is_current`).
pub(crate) static POLISH_EPOCH: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// ColorVideoVDP judge of the polish: the incremental arbiter following the
/// accepted frame, the DC error image it scores with, and its exchange rate.
pub(crate) struct CvJudge {
    judge: CvvdpArbiter,
    dc: Arc<Image3F>,
    lambda_bits: f32,
    cp: CvvdpCheckpoint,
    base: f32,
}

impl CvJudge {
    pub(crate) fn new(judge: CvvdpArbiter, dc: Arc<Image3F>, lambda_bits: f32) -> Self {
        let base = judge.deficit();
        Self {
            judge,
            dc,
            lambda_bits,
            cp: CvvdpCheckpoint::default(),
            base,
        }
    }
}

pub(crate) struct CoefPolisher {
    baseline: Arc<Image3F>,
    /// Mutable XYB trial pixels of the current block. Earlier commits remain
    /// in the metric states and in the output capture.
    recon: Image3F,
    state: Ssimulacra2State,
    /// Deficit per bit at the frame's exchange rate.
    lambda_bits: f32,
    cv: Option<CvJudge>,
    epoch: usize,
    base_d: f32,
    cp: Ssimulacra2Checkpoint,
    delta_coef: Vec<f32>,
    delta_pix: Vec<f32>,
    /// Forward luma updates and inactive planes add a canonical +0. Preserve
    /// their effect on initial signed zeros without repeatedly visiting them.
    may_have_negative_zero: [bool; 3],
    prepared_delta: Option<(usize, usize, i32, i32)>,
    options: Vec<PolishOption>,
    count_nonzero: CountNonzeroFn,
    add_scaled: AddScaledFn,
    area_budget: bool,
}

/// One channel of the block being polished.
pub(crate) struct PolishChannel<'a> {
    /// What the quantizer saw: luma source coefficients, chroma CfL residuals.
    pub(crate) source: &'a [f32],
    pub(crate) inv_qm: &'a [f32],
    pub(crate) q_scaled: f32,
    pub(crate) bias: DequantBias,
    /// Trellis distortion weight of the channel.
    pub(crate) weight: f32,
    pub(crate) scan: &'a [u32],
    pub(crate) histo_offset: u32,
    pub(crate) nzero_ctx: u32,
}

pub(crate) struct PolishBlock<'a> {
    pub(crate) raw_strategy: u8,
    pub(crate) cx: usize,
    pub(crate) cy: usize,
    pub(crate) size: usize,
    /// Frame pixel origin and transform pixel dims.
    pub(crate) px0: usize,
    pub(crate) py0: usize,
    pub(crate) pw: usize,
    pub(crate) ph: usize,
    /// The trellis lambda, for the proxy ranking.
    pub(crate) lambda: f32,
    pub(crate) channels: [PolishChannel<'a>; 3],
    /// Decoder-side multiplier of a Y pixel change on each plane: [ytox, 1, ytob].
    pub(crate) cfl: [f32; 3],
}

#[derive(Clone, Copy, Debug, Default)]
struct PolishMove {
    c: usize,
    idx: usize,
    order: u16,
    from: i32,
    to: i32,
    bits: f32,
    dbits: f32,
    proxy: f32,
}

#[derive(Clone, Copy, Debug)]
struct PolishOption {
    from: i32,
    to: i32,
    dd: f32,
    /// Scan position times two plus proposal direction. This preserves the
    /// original insertion order for ties and locates the coefficient. Even
    /// for 128x128 transforms the value is at most 32767.
    order: u16,
}

impl PolishOption {
    fn position(self) -> usize {
        usize::from(self.order) / 2
    }

    fn distortion_key(self) -> u64 {
        // The unsigned float key has the same ordering as f32::total_cmp,
        // including signed zeros and all NaNs. The low bits break ties
        // in the original scan order.
        let bits = self.dd.to_bits();
        let mask = 0u32.wrapping_sub(bits >> 31) | (1 << 31);
        (u64::from(bits ^ mask) << 16) | u64::from(self.order)
    }
}

/// Score the original moves in insertion order. The cached dequantized
/// magnitudes match `DequantBias::level` for the entire small-level range.
fn push_distortion_options(
    options: &mut Vec<PolishOption>,
    bias: DequantBias,
    weight: f32,
    cur: i32,
    ideal: f32,
    order: u16,
) {
    let mut push = |to, base_err: f32, to_level: f32, order| {
        let dd = weight * ((ideal - to_level).powi(2) - base_err);
        options.push(PolishOption {
            from: cur,
            to,
            dd,
            order,
        });
    };
    let abs = cur.unsigned_abs() as usize;
    if abs == 0 {
        if ideal.abs() >= RESTORE_MIN {
            let to = if ideal > 0.0 { 1 } else { -1 };
            push(to, (ideal - 0.0).powi(2), bias.one.copysign(ideal), order);
        }
        return;
    }
    let sign = cur.signum();
    let (base, toward, away) = if abs < SMALL_LEVELS - 1 {
        let sign_source = cur as f32;
        (
            bias.small_levels[abs].copysign(sign_source),
            // The zero level is +0 even when approached from below.
            0.0 + bias.small_levels[abs - 1].copysign(sign_source),
            bias.small_levels[abs + 1].copysign(sign_source),
        )
    } else {
        (
            bias.level(cur),
            bias.level(cur - sign),
            bias.level(cur + sign),
        )
    };
    let base_err = (ideal - base).powi(2);
    push(cur - sign, base_err, toward, order);
    push(cur + sign, base_err, away, order + 1);
}

#[cfg(test)]
fn rank_options(options: &mut Vec<PolishOption>) {
    rank_options_cap(options, PRICE_CAP);
}

fn rank_options_cap(options: &mut Vec<PolishOption>, cap: usize) {
    if options.len() > cap {
        options.select_nth_unstable_by_key(cap, |option| option.distortion_key());
        options.truncate(cap);
        options.sort_unstable_by_key(|option| option.distortion_key());
    }
}

/// Retain the best priced moves in stable proxy order.
fn insert_priced_move(priced: &mut [PolishMove], len: &mut usize, candidate: PolishMove) {
    if *len == priced.len() && !candidate.proxy.total_cmp(&priced[*len - 1].proxy).is_lt() {
        return;
    }
    let mut i = (*len).min(priced.len() - 1);
    while i > 0 && candidate.proxy.total_cmp(&priced[i - 1].proxy).is_lt() {
        priced[i] = priced[i - 1];
        i -= 1;
    }
    priced[i] = candidate;
    *len = (*len + 1).min(priced.len());
}

type CountNonzeroFn = fn(&[i32]) -> usize;

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    ))
))]
fn count_nonzero_scalar(values: &[i32]) -> usize {
    values.iter().filter(|&&v| v != 0).count()
}

fn select_count_nonzero_fn() -> CountNonzeroFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |v| unsafe { crate::neon::polish_count_nonzero_neon(v) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::polish_count_nonzero_wasm;
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |v| unsafe { crate::avx::polish_count_nonzero_avx2(v) };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |v| unsafe { crate::sse::polish_count_nonzero_sse41(v) };
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    count_nonzero_scalar
}

fn count_nonzero_except_llf(
    block: &[i32],
    cx: usize,
    cy: usize,
    count_nonzero: CountNonzeroFn,
) -> usize {
    let mut count = count_nonzero(block);
    for row in block.chunks_exact(cx * 8).take(cy) {
        count -= row[..cx].iter().filter(|&&v| v != 0).count();
    }
    count
}

/// Exact left-to-right accumulator and context before a scan position.
#[derive(Clone, Copy, Default)]
struct TokenPrefix {
    bits: f32,
    remaining: usize,
    prev: usize,
}

impl TokenPrefix {
    fn new(size: usize, nzeros: usize, ch: &PolishChannel, prices: &FrozenTokenPrices) -> Self {
        Self {
            bits: prices.token_bits(Token::new(ch.nzero_ctx, nzeros as u32)),
            remaining: nzeros,
            prev: usize::from(nzeros <= size / 16),
        }
    }

    #[inline]
    fn token_bits(
        ch: &PolishChannel,
        covered_blocks: usize,
        log2_covered_blocks: usize,
        k: usize,
        remaining: usize,
        prev: usize,
        coef: i32,
        prices: &FrozenTokenPrices,
    ) -> f32 {
        let ctx = ch.histo_offset as usize
            + if covered_blocks == 1 {
                zero_density_context_8x8(remaining, k, prev)
            } else {
                zero_density_context(remaining, k, covered_blocks, log2_covered_blocks, prev)
            };
        prices.token_bits(Token::new(ctx as u32, pack_signed(coef)))
    }
}

fn finish_bits(
    block: &[i32],
    ch: &PolishChannel,
    covered_blocks: usize,
    prices: &FrozenTokenPrices,
    mut k: usize,
    mut prefix: TokenPrefix,
) -> f32 {
    let log2_covered_blocks = covered_blocks.trailing_zeros() as usize;
    while k < block.len() && prefix.remaining != 0 {
        let coef = block[ch.scan[k] as usize];
        prefix.bits += TokenPrefix::token_bits(
            ch,
            covered_blocks,
            log2_covered_blocks,
            k,
            prefix.remaining,
            prefix.prev,
            coef,
            prices,
        );
        prefix.prev = usize::from(coef != 0);
        prefix.remaining -= prefix.prev;
        k += 1;
    }
    prefix.bits
}

/// Frozen-price bits of one channel's block tokens (nonzero count + walk).
fn block_bits(
    block: &[i32],
    ch: &PolishChannel,
    covered_blocks: usize,
    nzeros: usize,
    prices: &FrozenTokenPrices,
) -> f32 {
    finish_bits(
        block,
        ch,
        covered_blocks,
        prices,
        covered_blocks,
        TokenPrefix::new(block.len(), nzeros, ch, prices),
    )
}

fn reconstruct_coefficient_delta(
    idct: &crate::dct::IdctMethods,
    strategy: u8,
    idx: usize,
    delta: f32,
    coeffs: &mut [f32],
    pixels: &mut [f32],
) {
    if matches!(strategy, STRATEGY_DCT4X4..=STRATEGY_DCT8X4 | STRATEGY_AFV0..=STRATEGY_AFV3)
        && idx != 0
    {
        let matrix = crate::inflated_cost::inverse_matrix(strategy);
        // The unused DC product is +0 in every row. The other omitted
        // products are ±0, so a single rounded product gives the same sum.
        for (pixel, row) in pixels.iter_mut().zip(matrix.as_chunks::<64>().0) {
            *pixel = 0.0 + row[idx] * delta;
        }
    } else {
        coeffs[idx] = delta;
        crate::inflated_cost::reconstruct_error(idct, strategy, coeffs, pixels);
        coeffs[idx] = 0.0;
    }
}

/// Price single edits together. Before the edit, their contexts depend only
/// on the new count. After the edit and its successor, the original count
/// and previous-token flag apply again. Share both price walks while adding
/// every price to each proposal in the original f32 order.
fn price_options(
    block: &[i32],
    ch: &PolishChannel,
    covered: usize,
    nzeros: usize,
    prices: &FrozenTokenPrices,
    options: &[PolishOption],
) -> (f32, [f32; PRICE_CAP]) {
    let mut order = [0; PRICE_CAP];
    let order = &mut order[..options.len()];
    let mut last = [0; 3];
    let count_kind = |o: &PolishOption| 1 + usize::from(o.to != 0) - usize::from(o.from != 0);
    for (i, option) in options.iter().enumerate() {
        order[i] = i;
        last[count_kind(option)] = last[count_kind(option)].max(option.position());
    }
    order.sort_unstable_by_key(|&i| options[i].order);

    let log2_covered = covered.trailing_zeros() as usize;
    let token = |k, remaining, prev, coef| {
        TokenPrefix::token_bits(ch, covered, log2_covered, k, remaining, prev, coef, prices)
    };
    let mut base = TokenPrefix::new(block.len(), nzeros, ch, prices);
    if nzeros == 0 {
        // All proposals are restores. They share the zero run up to the
        // restored coefficient, which also ends the candidate's token walk.
        let mut prefix = TokenPrefix::new(block.len(), 1, ch, prices);
        let mut bits = [0.0; PRICE_CAP];
        let mut k = covered;
        for &i in order.iter() {
            let option = &options[i];
            let position = option.position();
            while k < position {
                prefix.bits += token(k, 1, prefix.prev, 0);
                prefix.prev = 0;
                k += 1;
            }
            bits[i] = prefix.bits + token(k, 1, prefix.prev, option.to);
        }
        return (base.bits, bits);
    }

    let mut prefixes = [TokenPrefix::default(); 3];
    for (kind, prefix) in prefixes.iter_mut().enumerate() {
        if kind != 1 && last[kind] != 0 {
            *prefix = TokenPrefix::new(block.len(), nzeros + kind - 1, ch, prices);
        }
    }
    let mut ready = 0;
    let mut active = 0;
    let mut sums = [0.0; PRICE_CAP];
    // Large transforms often code many tokens beyond the last edit. Every
    // trial has rejoined the base contexts two positions past that edit.
    let tail_start = if covered >= 64 {
        (last[0].max(last[1]).max(last[2]) + 2)
            .max(covered)
            .min(block.len())
    } else {
        block.len()
    };
    for k in covered..tail_start {
        if base.remaining == 0 && ready == order.len() {
            break;
        }
        let coef = block[ch.scan[k] as usize];
        let next_prev = usize::from(coef != 0);
        let next_remaining = base.remaining - next_prev;

        let mut end = ready;
        while end < order.len() {
            let option = &options[order[end]];
            if option.position() != k {
                break;
            }
            debug_assert_eq!(coef, option.from);
            let kind = count_kind(option);
            let prefix = if kind == 1 { &base } else { &prefixes[kind] };
            let mut bits = prefix.bits;
            if prefix.remaining != 0 {
                bits += token(k, prefix.remaining, prefix.prev, option.to);
            }
            if next_remaining != 0 && k + 1 < block.len() {
                bits += token(
                    k + 1,
                    next_remaining,
                    usize::from(option.to != 0),
                    block[ch.scan[k + 1] as usize],
                );
            }
            sums[end] = bits;
            end += 1;
        }

        if base.remaining != 0 {
            let price = token(k, base.remaining, base.prev, coef);
            base.bits += price;
            for bits in &mut sums[..active] {
                *bits += price;
            }
        }
        active = ready;
        ready = end;

        for kind in [0, 2] {
            let prefix = &mut prefixes[kind];
            if k < last[kind] && prefix.remaining != 0 {
                prefix.bits += token(k, prefix.remaining, prefix.prev, coef);
                prefix.remaining -= next_prev;
                prefix.prev = next_prev;
            }
        }
        base.remaining = next_remaining;
        base.prev = next_prev;
    }
    let all_sums = &mut sums[..order.len()];
    for (k, &index) in ch.scan[tail_start..block.len()].iter().enumerate() {
        if base.remaining == 0 {
            break;
        }
        let coef = block[index as usize];
        let price = token(tail_start + k, base.remaining, base.prev, coef);
        base.bits += price;
        for bits in &mut *all_sums {
            *bits += price;
        }
        base.prev = usize::from(coef != 0);
        base.remaining -= base.prev;
    }
    let mut bits = [0.0; PRICE_CAP];
    for (i, &option) in order.iter().enumerate() {
        bits[option] = sums[i];
    }
    (base.bits, bits)
}

impl CoefPolisher {
    pub(crate) fn new(
        baseline: Arc<Image3F>,
        state: Ssimulacra2State,
        lambda_bits: f32,
        cv: Option<CvJudge>,
    ) -> Self {
        let base_d = state.deficit();
        Self {
            baseline,
            recon: Image3F::new(0, 0),
            state,
            lambda_bits,
            cv,
            epoch: POLISH_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            base_d,
            cp: Ssimulacra2Checkpoint::default(),
            delta_coef: Vec::new(),
            delta_pix: Vec::new(),
            may_have_negative_zero: [false; 3],
            prepared_delta: None,
            options: Vec::new(),
            count_nonzero: select_count_nonzero_fn(),
            add_scaled: selected_add_scaled_fn(),
            area_budget: false,
        }
    }

    /// Keep refinement work proportional to covered area when content-adaptive
    /// selection chooses a denser layout. Both exact judges remain active.
    pub(crate) fn set_area_budget(&mut self, enabled: bool) {
        self.area_budget = enabled;
    }

    /// Built for the pass that is running now (stale ones stay in worker scratch).
    pub(crate) fn is_current(&self) -> bool {
        self.epoch == POLISH_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn rect(&self, b: &PolishBlock) -> Rect {
        Rect::new(
            b.px0,
            b.py0,
            (b.px0 + b.pw).min(self.baseline.xsize()),
            (b.py0 + b.ph).min(self.baseline.ysize()),
        )
    }

    /// Add `sign` times the accumulated plane deltas to the block reconstruction.
    /// The dispatched add rounds the CfL multiplication before adding it.
    fn apply_plane_deltas(&mut self, b: &PolishBlock, sign: f32) {
        let width = self.recon.xsize();
        let height = self.recon.ysize();
        let (channel, ..) = self.prepared_delta.expect("prepared pixel delta");
        for (c, plane) in self.recon.all_planes_mut().into_iter().enumerate() {
            let pixels = plane.as_mut_slice();
            let scale = if channel == 1 {
                b.cfl[c]
            } else if c == channel {
                1.0
            } else {
                0.0
            };
            // A forward luma or inactive-plane update adds +0 wherever the
            // old materialized delta was zero. Canonicalize negative zeros
            // once so applying the unmaterialized delta preserves that effect.
            if sign > 0.0 && (channel == 1 || scale == 0.0) && self.may_have_negative_zero[c] {
                for pixel in pixels.iter_mut() {
                    *pixel += 0.0;
                }
                self.may_have_negative_zero[c] = false;
            }
            if scale == 0.0 {
                continue;
            }
            let scale = sign * scale;
            if width == b.pw {
                (self.add_scaled)(pixels, &self.delta_pix[..width * height], scale);
            } else {
                for (row, delta) in pixels
                    .chunks_exact_mut(width)
                    .zip(self.delta_pix.chunks_exact(b.pw))
                {
                    (self.add_scaled)(row, &delta[..width], scale);
                }
            }
        }
    }

    /// Pixel-domain delta of `m` on the three decoded planes.
    fn plane_deltas(&mut self, ctx: &EncodingContext, b: &PolishBlock, m: &PolishMove) {
        let key = (m.c, m.idx, m.from, m.to);
        if self.prepared_delta == Some(key) {
            return;
        }
        self.prepared_delta = Some(key);
        if self.delta_pix.len() < b.size {
            self.delta_coef.resize(b.size, 0.0);
            self.delta_pix.resize(b.size, 0.0);
        }
        let ch = &b.channels[m.c];
        let dq = (ch.bias.level(m.to) - ch.bias.level(m.from)) / (ch.inv_qm[m.idx] * ch.q_scaled);
        reconstruct_coefficient_delta(
            ctx.idct,
            b.raw_strategy,
            m.idx,
            dq,
            &mut self.delta_coef[..b.size],
            &mut self.delta_pix[..b.size],
        );
    }

    /// SSIMULACRA2 objective change of `m` alone.
    fn judge_ss2(
        &mut self,
        ctx: &EncodingContext,
        b: &PolishBlock,
        rect: Rect,
        m: &PolishMove,
    ) -> f32 {
        self.plane_deltas(ctx, b, m);
        self.apply_plane_deltas(b, 1.0);
        let (base_d, lambda) = (self.base_d, self.lambda_bits);
        let CoefPolisher {
            recon, state, cp, ..
        } = self;
        state.update_region_checkpointed(recon, (rect.x0, rect.y0), rect, cp);
        let j = state.deficit() - base_d + lambda * m.dbits;
        state.rollback(cp);
        self.apply_plane_deltas(b, -1.0);
        j
    }

    /// ColorVideoVDP objective change of `m` alone (`None` without a judge).
    /// Always prepares its pixel delta so acceptance can reuse the same IDCT.
    fn judge_cv(
        &mut self,
        ctx: &EncodingContext,
        b: &PolishBlock,
        rect: Rect,
        m: &PolishMove,
    ) -> Option<f32> {
        self.plane_deltas(ctx, b, m);
        self.cv.as_ref()?;
        self.apply_plane_deltas(b, 1.0);
        let CoefPolisher { recon, cv, .. } = self;
        let cv = cv.as_mut().expect("judge");
        let d = cv.judge.update_region_checkpointed(
            recon,
            (rect.x0, rect.y0),
            &cv.dc,
            rect,
            &mut cv.cp,
        );
        cv.judge.rollback(&cv.cp);
        let j = d - cv.base + cv.lambda_bits * m.dbits;
        self.apply_plane_deltas(b, -1.0);
        Some(j)
    }

    /// Make the current frame reconstruction inside `rect` the judges' accepted frame.
    fn accept_rect(&mut self, rect: Rect) {
        let CoefPolisher {
            recon,
            state,
            cv,
            base_d,
            ..
        } = self;
        let origin = (rect.x0, rect.y0);
        state.update_region(recon, origin, rect);
        *base_d = state.deficit();
        if let Some(cv) = cv {
            cv.base = cv.judge.update_region(recon, origin, &cv.dc, rect);
        }
    }

    /// A layout visits each transform once. Its pixels start at the shared
    /// frame baseline, unless earlier moves changed its quantizer contexts.
    fn start_block(&mut self, rect: Rect, capture: (&Image3F, usize, usize)) {
        let (plane, lx0, ly0) = capture;
        let (width, height) = (rect.x1 - rect.x0, rect.y1 - rect.y0);
        self.recon.resize(width, height);
        self.may_have_negative_zero = [true; 3];
        let mut changed = false;
        for c in 0..3 {
            for y in 0..height {
                let src = &plane.plane_row(c, ly0 + y)[lx0..lx0 + width];
                let baseline = &self.baseline.plane_row(c, rect.y0 + y)[rect.x0..rect.x1];
                let row_changed = src != baseline;
                self.recon
                    .plane_row_mut(c, y)
                    .copy_from_slice(if row_changed { src } else { baseline });
                changed |= row_changed;
            }
        }
        if changed {
            self.accept_rect(rect);
        }
    }

    /// Polish one block. `capture` is the group-local reconstruction plane
    /// holding this block's pre-polish pixels at `(lx0, ly0)`.
    pub(crate) fn polish(
        &mut self,
        ctx: &EncodingContext,
        prices: &FrozenTokenPrices,
        b: &PolishBlock,
        quantized: &mut [&mut [i32]; 3],
        capture: (&mut Image3F, usize, usize),
    ) {
        let rect = self.rect(b);
        if rect.x1 <= rect.x0 || rect.y1 <= rect.y0 {
            return;
        }
        let (plane, lx0, ly0) = capture;
        self.start_block(rect, (plane, lx0, ly0));
        self.prepared_delta = None;
        let covered = b.cx * b.cy;
        // Fixed per-transform budgets multiply the work when a 32x32 turns
        // into sixteen leaves. Spend by area on the adaptive selection path.
        let positions = if self.area_budget {
            MAX_POSITIONS.min((b.size - covered).div_ceil(4))
        } else {
            MAX_POSITIONS
        };
        let per_channel = if self.area_budget {
            (PER_CHANNEL * covered).div_ceil(16).clamp(1, PER_CHANNEL)
        } else {
            PER_CHANNEL
        };
        let rounds = if self.area_budget && covered <= 2 {
            1
        } else {
            ROUNDS
        };
        let price_cap = if self.area_budget {
            6 * per_channel
        } else {
            PRICE_CAP
        };
        let cv_tries = if self.area_budget {
            CV_TRIES.min(per_channel)
        } else {
            CV_TRIES
        };
        let extra_tries = if self.area_budget {
            (EXTRA_TRIES * covered / 16).min(EXTRA_TRIES)
        } else {
            EXTRA_TRIES
        };
        // Transforms span a power-of-two number of blocks: shift and mask.
        let row_stride = b.cx * 8;
        debug_assert!(row_stride.is_power_of_two());
        let (row_shift, column_mask) = (row_stride.trailing_zeros(), row_stride - 1);
        let mut nzeros: [usize; 3] = std::array::from_fn(|c| {
            count_nonzero_except_llf(quantized[c], b.cx, b.cy, self.count_nonzero)
        });
        let mut singles = [PolishMove::default(); PER_CHANNEL * 3];
        let mut npriced = [0; 3];
        let mut base_bits = [0.0; 3];
        let mut reprice = [true; 3];
        for _round in 0..rounds {
            // ---- Proposals: proxy-ranked singles per channel ----
            let options = &mut self.options;
            for c in [1usize, 0, 2] {
                if !reprice[c] {
                    continue;
                }
                reprice[c] = false;
                npriced[c] = 0;
                let ch = &b.channels[c];
                let block = &*quantized[c];
                options.clear();
                let end = b.size.min(covered + positions);
                // Zero levels produce at most one proposal. Bound the
                // extra slots by the nonzero count and cap amortized growth
                // at the largest possible proposal list.
                let positions = end - covered;
                let needed = positions + nzeros[c].min(positions);
                let capacity = options.capacity();
                if capacity < needed {
                    options.reserve_exact(needed.max((capacity * 2).min(MAX_POSITIONS * 2)));
                }
                for k in covered..end {
                    let idx = ch.scan[k] as usize;
                    let (v, u) = (idx >> row_shift, idx & column_mask);
                    if v < b.cy && u < b.cx {
                        continue;
                    }
                    let cur = block[idx];
                    let ideal = ch.source[idx] * ch.inv_qm[idx] * ch.q_scaled;
                    push_distortion_options(
                        options,
                        ch.bias,
                        ch.weight,
                        cur,
                        ideal,
                        (k * 2) as u16,
                    );
                }
                // Price the most promising by distortion, then rank by the proxy.
                rank_options_cap(options, price_cap);
                if options.is_empty() {
                    continue;
                }
                let (base, bits) = price_options(block, ch, covered, nzeros[c], prices, options);
                base_bits[c] = base;
                let start = c * PER_CHANNEL;
                let priced = &mut singles[start..start + per_channel];
                for (i, option) in options.iter().enumerate() {
                    let PolishOption { from, to, dd, .. } = *option;
                    let dbits = bits[i] - base;
                    if dbits > MAX_COST_BITS {
                        continue;
                    }
                    insert_priced_move(
                        priced,
                        &mut npriced[c],
                        PolishMove {
                            c,
                            idx: ch.scan[option.position()] as usize,
                            order: option.order,
                            from,
                            to,
                            bits: bits[i],
                            dbits,
                            proxy: dd + b.lambda * dbits,
                        },
                    );
                }
            }

            // ---- Judge: SSIMULACRA2 ranks, ColorVideoVDP checks the best few ----
            let mut scored = [(0.0f32, 0usize); PER_CHANNEL * 3];
            let mut nscored = 0;
            for c in [1usize, 0, 2] {
                let start = c * PER_CHANNEL;
                for (i, m) in singles[start..start + npriced[c]].iter().enumerate() {
                    let j = self.judge_ss2(ctx, b, rect, m);
                    if j < -1e-6 {
                        scored[nscored] = (j, start + i);
                        nscored += 1;
                    }
                }
            }
            let scored = &mut scored[..nscored];
            scored.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut best: Option<(PolishMove, usize)> = None;
            for &(_, i) in scored.iter().take(cv_tries) {
                let m = singles[i];
                match self.judge_cv(ctx, b, rect, &m) {
                    Some(j_cv) if j_cv >= 0.0 => continue,
                    _ => {
                        best = Some((m, i));
                        break;
                    }
                }
            }

            // ---- Accept ----
            let Some((m, best_i)) = best else { break };
            // The final judge_cv call prepared this move, even without a judge.
            reprice[m.c] = true;
            let at = (&mut *plane, lx0, ly0);
            self.commit(b, rect, &m, at, quantized, &mut nzeros, &mut base_bits);

            // ---- Further accepted proposals, re-priced and re-judged on the new block ----
            let mut touched = [(m.c, m.idx); EXTRA_TRIES + 1];
            let mut ntouched = 1;
            let mut tries = 0;
            let mut cached_price = [None; PER_CHANNEL * 3];
            for (rank, &(_, i)) in scored.iter().enumerate() {
                if tries == extra_tries {
                    break;
                }
                let mut m = singles[i];
                if i == best_i
                    || touched[..ntouched].contains(&(m.c, m.idx))
                    || quantized[m.c][m.idx] != m.from
                {
                    continue;
                }
                if reprice[m.c] {
                    if covered >= 64 && tries + 1 < extra_tries && cached_price[i].is_none() {
                        let mut batch = [PolishOption {
                            from: 0,
                            to: 0,
                            dd: 0.0,
                            order: 0,
                        }; PER_CHANNEL];
                        let mut indices = [0; PER_CHANNEL];
                        let mut len = 0;
                        for &(_, j) in &scored[rank..] {
                            let other = &singles[j];
                            if other.c != m.c
                                || touched[..ntouched].contains(&(other.c, other.idx))
                                || quantized[m.c][other.idx] != other.from
                            {
                                continue;
                            }
                            batch[len] = PolishOption {
                                from: other.from,
                                to: other.to,
                                dd: 0.0,
                                order: other.order,
                            };
                            indices[len] = j;
                            len += 1;
                        }
                        if len > 1 {
                            let (_, bits) = price_options(
                                quantized[m.c],
                                &b.channels[m.c],
                                covered,
                                nzeros[m.c],
                                prices,
                                &batch[..len],
                            );
                            for (&j, &bits) in indices[..len].iter().zip(&bits) {
                                cached_price[j] = Some(bits);
                            }
                        }
                    }
                    m.bits = if let Some(bits) = cached_price[i] {
                        bits
                    } else {
                        quantized[m.c][m.idx] = m.to;
                        let nz = nzeros[m.c] + usize::from(m.to != 0) - usize::from(m.from != 0);
                        let bits =
                            block_bits(quantized[m.c], &b.channels[m.c], covered, nz, prices);
                        quantized[m.c][m.idx] = m.from;
                        bits
                    };
                    m.dbits = m.bits - base_bits[m.c];
                    if m.dbits > MAX_COST_BITS {
                        continue;
                    }
                }
                tries += 1;
                if self.judge_ss2(ctx, b, rect, &m) >= -1e-6 {
                    continue;
                }
                if let Some(j_cv) = self.judge_cv(ctx, b, rect, &m)
                    && j_cv >= 0.0
                {
                    continue;
                }
                touched[ntouched] = (m.c, m.idx);
                ntouched += 1;
                reprice[m.c] = true;
                let start = m.c * PER_CHANNEL;
                cached_price[start..start + PER_CHANNEL].fill(None);
                let at = (&mut *plane, lx0, ly0);
                self.commit(b, rect, &m, at, quantized, &mut nzeros, &mut base_bits);
            }
        }
    }

    /// Commit `m`, whose plane deltas the last judge call prepared.
    fn commit(
        &mut self,
        b: &PolishBlock,
        rect: Rect,
        m: &PolishMove,
        (plane, lx0, ly0): (&mut Image3F, usize, usize),
        quantized: &mut [&mut [i32]; 3],
        nzeros: &mut [usize; 3],
        base_bits: &mut [f32; 3],
    ) {
        self.apply_plane_deltas(b, 1.0);
        self.accept_rect(rect);
        // Keep the capture plane consistent with the frame reconstruction.
        for c in 0..3 {
            for y in 0..self.recon.ysize() {
                let src = self.recon.plane_row(c, y);
                plane.plane_row_mut(c, ly0 + y)[lx0..lx0 + src.len()].copy_from_slice(src);
            }
        }
        quantized[m.c][m.idx] = m.to;
        nzeros[m.c] = nzeros[m.c] + usize::from(m.to != 0) - usize::from(m.from != 0);
        base_bits[m.c] = m.bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_prices(prefix: bool) -> FrozenTokenPrices {
        use crate::entropy::{
            Histogram, HybridUintConfig, OwnedEntropyCode, build_ans_code_parts,
            build_huffman_codes,
        };
        let histograms: Vec<_> = (0..4)
            .map(|c| {
                let mut histogram = Histogram::new();
                for (symbol, count) in histogram.counts.iter_mut().enumerate() {
                    *count = 1 + ((symbol * 13 + c * 31) % 109) as u32;
                }
                histogram.total_count = histogram.counts.iter().sum();
                histogram
            })
            .collect();
        let (ans_histograms, ans_symbols, ans_reverse_maps) =
            build_ans_code_parts(&histograms, false);
        FrozenTokenPrices::new(&OwnedEntropyCode {
            context_map: (0..crate::ac_context::K_NUM_FINE_AC_CONTEXTS)
                .map(|i| ((i * 13 + i / 7) % 4) as u8)
                .collect(),
            prefix_codes: build_huffman_codes(&histograms, &mut Vec::new()),
            hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; 4],
            orig_context_map: None,
            orig_num_contexts: crate::ac_context::K_NUM_FINE_AC_CONTEXTS,
            use_prefix_code: prefix,
            ans_histograms,
            ans_pricing_freqs: Vec::new(),
            ans_symbols,
            ans_reverse_maps,
        })
    }

    /// Original full walk, independent of the cached accumulator implementation.
    fn reference_block_bits(
        block: &[i32],
        ch: &PolishChannel,
        covered: usize,
        nzeros: usize,
        prices: &FrozenTokenPrices,
    ) -> f32 {
        let mut bits = prices.token_bits(Token::new(ch.nzero_ctx, nzeros as u32));
        let mut remaining = nzeros;
        let mut prev = usize::from(nzeros <= block.len() / 16);
        let mut k = covered;
        while k < block.len() && remaining != 0 {
            let coef = block[ch.scan[k] as usize];
            let ctx = ch.histo_offset as usize
                + if covered == 1 {
                    zero_density_context_8x8(remaining, k, prev)
                } else {
                    zero_density_context(
                        remaining,
                        k,
                        covered,
                        covered.trailing_zeros() as usize,
                        prev,
                    )
                };
            bits += prices.token_bits(Token::new(ctx as u32, pack_signed(coef)));
            prev = usize::from(coef != 0);
            remaining -= usize::from(coef != 0);
            k += 1;
        }
        bits
    }

    fn pricing_options(block: &[i32], scan: &[u32], covered: usize) -> Vec<PolishOption> {
        let mut options = Vec::new();
        for (k, &idx) in scan.iter().enumerate().skip(covered).take(MAX_POSITIONS) {
            let idx = idx as usize;
            let from = block[idx];
            let mut push = |to, direction| {
                options.push(PolishOption {
                    from,
                    to,
                    dd: ((k * 17) % 29) as f32,
                    order: (k * 2 + direction) as u16,
                })
            };
            if from != 0 {
                push(from - from.signum(), 0);
                push(from + from.signum(), 1);
            } else {
                push(if k % 2 == 0 { 1 } else { -1 }, 0);
            }
        }
        rank_options(&mut options);
        options
    }

    #[test]
    fn sparse_matrix_inverse_preserves_dense_impulse_bits() {
        let idct = crate::dct::IdctMethods::scalar();
        for strategy in (STRATEGY_DCT4X4..=STRATEGY_DCT8X4).chain(STRATEGY_AFV0..=STRATEGY_AFV3) {
            let mut coeffs = [0.0; 64];
            let mut expected = [0.0; 64];
            let mut actual = [0.0; 64];
            for idx in 0..64 {
                for magnitude in [0.0, f32::from_bits(1), 1e-7, 0.03875, 127.5] {
                    for delta in [magnitude, -magnitude] {
                        coeffs[idx] = delta;
                        crate::inflated_cost::reconstruct_error(
                            &idct,
                            strategy,
                            &coeffs,
                            &mut expected,
                        );
                        coeffs[idx] = 0.0;
                        reconstruct_coefficient_delta(
                            &idct,
                            strategy,
                            idx,
                            delta,
                            &mut coeffs,
                            &mut actual,
                        );
                        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
                    }
                }
            }
        }
    }

    #[test]
    fn batched_prices_preserve_full_walk_bits_for_all_shapes_and_moves() {
        let orders = crate::coeff_order::CoeffOrders::natural();
        for prefix_code in [false, true] {
            let prices = test_prices(prefix_code);
            for (strategy, cx, cy) in [
                (0, 1, 1),
                (4, 2, 2),
                (5, 4, 4),
                (6, 2, 1),
                (7, 2, 1),
                (8, 4, 1),
                (9, 4, 1),
                (10, 4, 2),
                (11, 4, 2),
                (18, 8, 8),
                (19, 8, 4),
                (20, 8, 4),
                (21, 16, 16),
                (22, 16, 8),
                (23, 16, 8),
            ] {
                let covered = cx * cy;
                let size = covered * 64;
                for learned in [false, true] {
                    let mut scan = orders.scan_for(strategy, 0).to_vec();
                    assert_eq!(scan.len(), size);
                    if learned {
                        scan[covered..].reverse();
                    }
                    let ch = PolishChannel {
                        source: &[],
                        inv_qm: &[],
                        q_scaled: 1.0,
                        bias: DequantBias::DEFAULT[0],
                        weight: 1.0,
                        scan: &scan,
                        nzero_ctx: 17,
                        histo_offset: crate::ac_context::fine_zero_density_contexts_offset(3),
                    };
                    for pattern in 0..7 {
                        let mut block = vec![0; size];
                        for (k, &idx) in scan.iter().enumerate() {
                            let level = [-30000, -129, -3, -1, 1, 2, 129, 30000][k % 8];
                            block[idx as usize] = if k < covered {
                                7 // LLF levels must not enter the walk.
                            } else {
                                match pattern {
                                    0 => 0,
                                    1 | 6 => level,
                                    2 if k % 17 == 0 || k == size - 1 => level,
                                    3 if k == covered => level,
                                    4 if k == (covered + MAX_POSITIONS - 1).min(size - 1) => level,
                                    5 if k == size - 1 => level,
                                    _ => 0,
                                }
                            };
                        }
                        for _round in 0..2 {
                            let nzeros =
                                count_nonzero_except_llf(&block, cx, cy, count_nonzero_scalar);
                            let original = block.clone();
                            let base = reference_block_bits(&block, &ch, covered, nzeros, &prices);
                            assert_eq!(
                                block_bits(&block, &ch, covered, nzeros, &prices).to_bits(),
                                base.to_bits()
                            );
                            let options = pricing_options(&block, &scan, covered);
                            let (base_batched, bits) =
                                price_options(&block, &ch, covered, nzeros, &prices, &options);
                            assert_eq!(base_batched.to_bits(), base.to_bits());
                            for (i, option) in options.iter().enumerate() {
                                let idx = scan[option.position()] as usize;
                                block[idx] = option.to;
                                let trial_nzeros =
                                    count_nonzero_except_llf(&block, cx, cy, count_nonzero_scalar);
                                let expected = reference_block_bits(
                                    &block,
                                    &ch,
                                    covered,
                                    trial_nzeros,
                                    &prices,
                                );
                                block[idx] = option.from;
                                let actual = bits[i];
                                assert_eq!(
                                    actual.to_bits(),
                                    expected.to_bits(),
                                    "strategy={strategy}, learned={learned}, pattern={pattern}, option={option:?}"
                                );
                                assert_eq!((actual - base).to_bits(), (expected - base).to_bits());
                                assert_eq!(block, original, "pricing must restore the level");
                            }
                            // Rebuild against a changed accepted baseline in the next round.
                            let accepted = options[0];
                            block[scan[accepted.position()] as usize] = accepted.to;
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "isolated token pricing benchmark; run in release with --ignored --nocapture"]
    fn benchmark_token_prefix_pricing() {
        use std::hint::black_box;
        use std::time::Instant;
        let prices = test_prices(false);
        let orders = crate::coeff_order::CoeffOrders::natural();
        for (strategy, covered, repeats) in
            [(0, 1, 10000), (4, 4, 3000), (5, 16, 700), (21, 256, 50)]
        {
            let scan = orders.scan_for(strategy, 0);
            let ch = PolishChannel {
                source: &[],
                inv_qm: &[],
                q_scaled: 1.0,
                bias: DequantBias::DEFAULT[0],
                weight: 1.0,
                scan,
                nzero_ctx: 17,
                histo_offset: crate::ac_context::fine_zero_density_contexts_offset(3),
            };
            let mut block = vec![2; scan.len()];
            let nzeros = block.len() - covered;
            let options = pricing_options(&block, scan, covered);
            let mut full_times = Vec::new();
            let mut cached_times = Vec::new();
            for sample in 0..5 {
                let mut sums = [0.0f32; 2];
                for cached in if sample % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    for _ in 0..repeats {
                        if cached {
                            let (_, bits) = price_options(
                                black_box(&block),
                                &ch,
                                covered,
                                nzeros,
                                &prices,
                                &options,
                            );
                            sums[1] += black_box(bits[..options.len()].iter().sum::<f32>());
                        } else {
                            let mut bits = 0.0;
                            for option in &options {
                                let idx = scan[option.position()] as usize;
                                block[idx] = option.to;
                                bits += black_box(block_bits(
                                    black_box(&block),
                                    &ch,
                                    covered,
                                    nzeros,
                                    &prices,
                                ));
                                block[idx] = option.from;
                            }
                            sums[0] += bits;
                        }
                    }
                    let time = start.elapsed().as_secs_f64();
                    if cached {
                        cached_times.push(time);
                    } else {
                        full_times.push(time);
                    }
                }
                assert_eq!(sums[0].to_bits(), sums[1].to_bits());
            }
            full_times.sort_by(f64::total_cmp);
            cached_times.sort_by(f64::total_cmp);
            println!(
                "size={} proposals={} full_ms={:.3} cached_ms={:.3} speedup={:.3}x",
                block.len(),
                options.len(),
                full_times[2] * 1000.0,
                cached_times[2] * 1000.0,
                full_times[2] / cached_times[2]
            );
        }
    }

    #[test]
    fn verification_without_cv_prepares_the_selected_delta_for_acceptance() {
        let ctx = EncodingContext::new(
            crate::Speed::ExtraSlow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            3.0,
            1,
        );
        let mut recon = Image3F::new(23, 21);
        for y in 0..recon.ysize() {
            recon.plane_row_mut(1, y).fill(0.4);
        }
        let reference = crate::ssimulacra2::Ssimulacra2Reference::new(&recon).unwrap();
        let source = [0.0; 64];
        let inv_qm = [1.0; 64];
        let orders = crate::coeff_order::CoeffOrders::natural();
        let block = PolishBlock {
            raw_strategy: 0,
            cx: 1,
            cy: 1,
            size: 64,
            px0: 16,
            py0: 16,
            pw: 8,
            ph: 8,
            lambda: 1.0,
            channels: std::array::from_fn(|c| PolishChannel {
                source: &source,
                inv_qm: &inv_qm,
                q_scaled: 128.0,
                bias: DequantBias::DEFAULT[c],
                weight: 1.0,
                scan: orders.scan_for(0, c),
                histo_offset: 0,
                nzero_ctx: 0,
            }),
            cfl: [0.5, 1.0, -0.25],
        };
        let recon = Arc::new(recon);
        for c in 0..3 {
            let mut p = CoefPolisher::new(Arc::clone(&recon), reference.state(&recon), 1.0, None);
            let rect = p.rect(&block);
            p.start_block(rect, (&recon, rect.x0, rect.y0));
            let selected = PolishMove {
                c,
                idx: 1,
                from: 0,
                to: 1,
                ..PolishMove::default()
            };
            assert!(p.judge_cv(&ctx, &block, rect, &selected).is_none());
            for plane in 0..3 {
                for y in rect.y0..rect.y1 {
                    assert_eq!(
                        p.recon.plane_row(plane, y - rect.y0),
                        &recon.plane_row(plane, y)[rect.x0..rect.x1],
                    );
                }
            }
            p.apply_plane_deltas(&block, 1.0);
            p.accept_rect(rect);
            let mut expected_recon = recon.as_ref().clone();
            for plane in 0..3 {
                for y in rect.y0..rect.y1 {
                    for x in rect.x0..rect.x1 {
                        let pixel = p.delta_pix[(y - block.py0) * 8 + x - block.px0];
                        let delta = if c == 1 {
                            0.0 + block.cfl[plane] * pixel
                        } else if c == plane {
                            pixel
                        } else {
                            0.0
                        };
                        let expected = recon.plane_row(plane, y)[x] + delta;
                        assert_eq!(
                            p.recon.plane_row(plane, y - rect.y0)[x - rect.x0].to_bits(),
                            expected.to_bits(),
                        );
                        expected_recon.plane_row_mut(plane, y)[x] = expected;
                    }
                }
            }
            assert!(p.base_d.is_finite());
            assert!(p.delta_pix.iter().any(|&v| v != 0.0));
            assert!((p.base_d - reference.deficit(&expected_recon)).abs() < 1e-3);
        }
    }

    #[test]
    fn shared_pixel_delta_preserves_forward_and_rollback_bits() {
        let mut baseline = Image3F::new(8, 8);
        let samples = [0.0, -0.0, f32::from_bits(1), -f32::from_bits(1), -0.3, 0.6];
        for c in 0..3 {
            for (i, pixel) in baseline.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *pixel = samples[(i * 7 + c) % samples.len()];
            }
        }
        let baseline = Arc::new(baseline);
        let reference = crate::ssimulacra2::Ssimulacra2Reference::new(&baseline).unwrap();
        let mut p = CoefPolisher::new(Arc::clone(&baseline), reference.state(&baseline), 1.0, None);
        let channel = || PolishChannel {
            source: &[],
            inv_qm: &[],
            q_scaled: 1.0,
            bias: DequantBias::DEFAULT[0],
            weight: 1.0,
            scan: &[],
            histo_offset: 0,
            nzero_ctx: 0,
        };
        p.delta_pix
            .extend((0..64).map(|i| samples[i % samples.len()]));
        for (width, height) in [(8, 8), (7, 5)] {
            for cfl in [[0.5, 1.0, -0.25], [-0.0, 1.0, 0.0]] {
                let block = PolishBlock {
                    raw_strategy: 0,
                    cx: 1,
                    cy: 1,
                    size: 64,
                    px0: 0,
                    py0: 0,
                    pw: 8,
                    ph: 8,
                    lambda: 1.0,
                    channels: std::array::from_fn(|_| channel()),
                    cfl,
                };
                let rect = Rect::new(0, 0, width, height);
                p.start_block(rect, (&baseline, 0, 0));
                let mut expected = p.recon.clone();
                for channel in [0, 2, 1, 2, 0, 1] {
                    p.prepared_delta = Some((channel, 0, 0, 1));
                    for sign in [1.0, -1.0, 1.0] {
                        p.apply_plane_deltas(&block, sign);
                        for (c, plane) in expected.all_planes_mut().into_iter().enumerate() {
                            for y in 0..height {
                                let delta: Vec<_> = p.delta_pix[y * 8..y * 8 + width]
                                    .iter()
                                    .map(|&pixel| {
                                        if channel == 1 && cfl[c] != 0.0 {
                                            0.0 + cfl[c] * pixel
                                        } else if channel != 1 && c == channel {
                                            pixel
                                        } else {
                                            0.0
                                        }
                                    })
                                    .collect();
                                (p.add_scaled)(plane.row_mut(y), &delta, sign);
                            }
                            assert!(
                                p.recon
                                    .plane_data(c)
                                    .iter()
                                    .zip(plane.as_slice())
                                    .all(|(&actual, &expected)| actual.to_bits()
                                        == expected.to_bits())
                            );
                        }
                    }
                }
            }
        }
    }

    fn check_count_nonzero(count: CountNonzeroFn) {
        let values: Vec<i32> = (0..16386)
            .map(|i| [0, -1, i32::MIN, 0, i32::MAX, 7, 0][i % 7])
            .collect();
        for len in (0..129).chain([255, 256, 1023, 16384]) {
            let input = &values[1..1 + len];
            assert_eq!(count(input), count_nonzero_scalar(input), "len={len}");
        }
    }

    #[test]
    fn selected_nonzero_count_matches_scalar() {
        check_count_nonzero(select_count_nonzero_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_nonzero_count_matches_scalar() {
        if is_x86_feature_detected!("avx2") {
            check_count_nonzero(|v| unsafe { crate::avx::polish_count_nonzero_avx2(v) });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_nonzero_count_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_count_nonzero(|v| unsafe { crate::sse::polish_count_nonzero_sse41(v) });
        }
    }

    #[test]
    fn partial_ranking_matches_stable_sort_including_ties() {
        for len in 0..=MAX_POSITIONS * 2 {
            let mut options: Vec<_> = (0..len)
                .map(|i| PolishOption {
                    from: 0,
                    to: 1,
                    dd: [
                        -0.0,
                        0.0,
                        -1.0,
                        1.0,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        f32::NAN,
                        -f32::NAN,
                        f32::from_bits(0x7f80_0001),
                        f32::from_bits(0xff80_0001),
                        f32::from_bits(1),
                        f32::from_bits(0x8000_0001),
                    ][(i * 31) % 12],
                    order: i as u16,
                })
                .collect();
            let mut expected = options.clone();
            if expected.len() > PRICE_CAP {
                expected.sort_by(|a, b| a.dd.total_cmp(&b.dd));
                expected.truncate(PRICE_CAP);
            }
            rank_options(&mut options);
            assert_eq!(
                options.iter().map(|o| o.order).collect::<Vec<_>>(),
                expected.iter().map(|o| o.order).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn distortion_proposals_match_direct_dequantization() {
        let mut actual = Vec::new();
        for bias in DequantBias::DEFAULT
            .into_iter()
            .chain(DequantBias::channels(true))
        {
            for ideal in [-3.4, -0.25, -0.0, 0.0, 0.249, 0.25, 1.6, 17.2] {
                for cur in (-40i32..=40).chain([-32768, 32768]) {
                    actual.clear();
                    push_distortion_options(&mut actual, bias, 0.37, cur, ideal, 17);
                    let mut expected = Vec::new();
                    if cur != 0 {
                        expected.extend([cur - cur.signum(), cur + cur.signum()]);
                    } else if ideal.abs() >= RESTORE_MIN {
                        expected.push(if ideal > 0.0 { 1 } else { -1 });
                    }
                    assert_eq!(actual.len(), expected.len());
                    let base = (ideal - bias.level(cur)).powi(2);
                    for (i, (&to, proposal)) in expected.iter().zip(&actual).enumerate() {
                        let dd = 0.37 * ((ideal - bias.level(to)).powi(2) - base);
                        assert_eq!(proposal.order, 17 + i as u16);
                        assert_eq!((proposal.from, proposal.to), (cur, to));
                        assert_eq!(proposal.dd.to_bits(), dd.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn priced_top_moves_preserve_stable_total_order() {
        let values = [
            f32::NAN,
            -0.0,
            0.0,
            -1.0,
            1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for len in 0..=PRICE_CAP {
            let input: Vec<_> = (0..len)
                .map(|idx| PolishMove {
                    idx,
                    proxy: values[(idx * 3) % values.len()],
                    ..PolishMove::default()
                })
                .collect();
            let mut sorted = input.clone();
            sorted.sort_by(|a, b| a.proxy.total_cmp(&b.proxy));
            for cap in 1..=PER_CHANNEL {
                let mut kept = [PolishMove::default(); PER_CHANNEL];
                let mut len = 0;
                for &m in &input {
                    insert_priced_move(&mut kept[..cap], &mut len, m);
                }
                assert_eq!(len, sorted.len().min(cap));
                for (actual, expected) in kept[..len].iter().zip(&sorted) {
                    assert_eq!(actual.idx, expected.idx);
                }
            }
        }
    }

    #[test]
    fn cached_nonzero_counts_match_full_recounts_for_every_move() {
        let count = select_count_nonzero_fn();
        let orders = crate::coeff_order::CoeffOrders::natural();
        for (strategy, cx, cy) in [
            (0, 1, 1),
            (4, 2, 2),
            (5, 4, 4),
            (6, 2, 1),
            (7, 1, 2),
            (18, 8, 8),
            (19, 8, 4),
            (20, 4, 8),
            (21, 16, 16),
        ] {
            let size = cx * cy * 64;
            let scan = orders.scan_for(strategy, 0);
            for period in [1, 5, 67, size + 1] {
                let mut block: Vec<i32> = (0..size)
                    .map(|i| {
                        if i % period == 0 {
                            (i % 7) as i32 - 3
                        } else {
                            0
                        }
                    })
                    .collect();
                let recount = |block: &[i32]| {
                    block
                        .iter()
                        .enumerate()
                        .filter(|&(i, &v)| v != 0 && !(i / (cx * 8) < cy && i % (cx * 8) < cx))
                        .count()
                };
                let nzeros = recount(&block);
                assert_eq!(count_nonzero_except_llf(&block, cx, cy, count), nzeros);
                for &idx in scan.iter().skip(cx * cy).take(MAX_POSITIONS) {
                    let idx = idx as usize;
                    if idx / (cx * 8) < cy && idx % (cx * 8) < cx {
                        continue;
                    }
                    let from = block[idx];
                    for to in [0, -1, 1, from - 1, from + 1] {
                        block[idx] = to;
                        assert_eq!(
                            nzeros + usize::from(to != 0) - usize::from(from != 0),
                            recount(&block)
                        );
                    }
                    block[idx] = from;
                }
            }
        }
    }
}
