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
use crate::encoding_context::EncodingContext;
use crate::entropy::{FrozenTokenPrices, Token, pack_signed};
use crate::group::DequantBias;
use crate::image::Image3F;
use crate::ssimulacra2::{Rect, Ssimulacra2Checkpoint, Ssimulacra2State};

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
    dc: Image3F,
    lambda_bits: f32,
    cp: CvvdpCheckpoint,
    base: f32,
}

impl CvJudge {
    pub(crate) fn new(judge: CvvdpArbiter, dc: Image3F, lambda_bits: f32) -> Self {
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
    plane_delta: [Vec<f32>; 3],
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
    from: i32,
    to: i32,
    dbits: f32,
    proxy: f32,
}

#[derive(Clone, Copy, Debug)]
struct PolishOption {
    idx: usize,
    from: i32,
    to: i32,
    dd: f32,
    /// Scan position times two plus proposal direction. This preserves the
    /// original insertion order for ties and also locates the token prefix.
    order: usize,
}

#[cfg(test)]
fn rank_options(options: &mut Vec<PolishOption>) {
    rank_options_cap(options, PRICE_CAP);
}

fn rank_options_cap(options: &mut Vec<PolishOption>, cap: usize) {
    let cmp =
        |a: &PolishOption, b: &PolishOption| a.dd.total_cmp(&b.dd).then(a.order.cmp(&b.order));
    if options.len() > cap {
        options.select_nth_unstable_by(cap, cmp);
        options.truncate(cap);
        options.sort_unstable_by(cmp);
    }
}

type CountNonzeroFn = fn(&[i32]) -> usize;
type AddScaledFn = fn(&mut [f32], &[f32], f32);

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

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    ))
))]
fn add_scaled_scalar(row: &mut [f32], delta: &[f32], sign: f32) {
    assert_eq!(row.len(), delta.len());
    for (v, &d) in row.iter_mut().zip(delta) {
        *v += sign * d;
    }
}

fn select_kernels() -> (CountNonzeroFn, AddScaledFn) {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return (
        |v| unsafe { crate::neon::polish_count_nonzero_neon(v) },
        |r, d, s| unsafe { crate::neon::polish_add_scaled_neon(r, d, s) },
    );
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return (
        crate::wasm::polish_count_nonzero_wasm,
        crate::wasm::polish_add_scaled_wasm,
    );
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return (
            |v| unsafe { crate::avx::polish_count_nonzero_avx2(v) },
            |r, d, s| unsafe { crate::avx::polish_add_scaled_avx2(r, d, s) },
        );
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return (
            |v| unsafe { crate::sse::polish_count_nonzero_sse41(v) },
            |r, d, s| unsafe { crate::sse::polish_add_scaled_sse41(r, d, s) },
        );
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    (count_nonzero_scalar, add_scaled_scalar)
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
    fn advance(
        &mut self,
        ch: &PolishChannel,
        covered_blocks: usize,
        log2_covered_blocks: usize,
        k: usize,
        coef: i32,
        prices: &FrozenTokenPrices,
    ) {
        let ctx = ch.histo_offset as usize
            + if covered_blocks == 1 {
                zero_density_context_8x8(self.remaining, k, self.prev)
            } else {
                zero_density_context(
                    self.remaining,
                    k,
                    covered_blocks,
                    log2_covered_blocks,
                    self.prev,
                )
            };
        self.bits += prices.token_bits(Token::new(ctx as u32, pack_signed(coef)));
        self.prev = usize::from(coef != 0);
        self.remaining -= usize::from(coef != 0);
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
        prefix.advance(
            ch,
            covered_blocks,
            log2_covered_blocks,
            k,
            block[ch.scan[k] as usize],
            prices,
        );
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

/// Save only prefixes requested by count-preserving proposals, before any
/// trial edits. The bounded stack cache is rebuilt per channel and round.
fn token_prefixes(
    block: &[i32],
    ch: &PolishChannel,
    covered_blocks: usize,
    nzeros: usize,
    prices: &FrozenTokenPrices,
    options: &[PolishOption],
) -> [TokenPrefix; PRICE_CAP] {
    let mut requests = [(0usize, 0usize); PRICE_CAP];
    let mut len = 0;
    for (i, option) in options.iter().enumerate() {
        if option.from != 0 && option.to != 0 {
            requests[len] = (option.order / 2, i);
            len += 1;
        }
    }
    let mut prefixes = [TokenPrefix::default(); PRICE_CAP];
    if len == 0 {
        return prefixes;
    }
    requests[..len].sort_unstable();
    let log2_covered_blocks = covered_blocks.trailing_zeros() as usize;
    let mut prefix = TokenPrefix::new(block.len(), nzeros, ch, prices);
    let mut k = covered_blocks;
    for &(position, i) in &requests[..len] {
        while k < position && prefix.remaining != 0 {
            prefix.advance(
                ch,
                covered_blocks,
                log2_covered_blocks,
                k,
                block[ch.scan[k] as usize],
                prices,
            );
            k += 1;
        }
        debug_assert_eq!(
            k, position,
            "nonzero proposal must lie inside the token walk"
        );
        prefixes[i] = prefix;
    }
    prefixes
}

fn proposal_bits(
    block: &mut [i32],
    ch: &PolishChannel,
    covered_blocks: usize,
    nzeros: usize,
    prices: &FrozenTokenPrices,
    option: &PolishOption,
    prefix: TokenPrefix,
) -> f32 {
    debug_assert_eq!(block[option.idx], option.from);
    block[option.idx] = option.to;
    let bits = if option.from != 0 && option.to != 0 {
        // Both the header and earlier contexts are unchanged. Continue from
        // the exact saved f32 accumulator; do not subtract or regroup sums.
        finish_bits(block, ch, covered_blocks, prices, option.order / 2, prefix)
    } else {
        let nzeros = nzeros + usize::from(option.to != 0) - usize::from(option.from != 0);
        block_bits(block, ch, covered_blocks, nzeros, prices)
    };
    block[option.idx] = option.from;
    bits
}

impl CoefPolisher {
    pub(crate) fn new(
        recon: Image3F,
        state: Ssimulacra2State,
        lambda_bits: f32,
        cv: Option<CvJudge>,
    ) -> Self {
        let base_d = state.deficit();
        let (count_nonzero, add_scaled) = select_kernels();
        Self {
            recon,
            state,
            lambda_bits,
            cv,
            epoch: POLISH_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            base_d,
            cp: Ssimulacra2Checkpoint::default(),
            delta_coef: Vec::new(),
            delta_pix: Vec::new(),
            plane_delta: Default::default(),
            options: Vec::with_capacity(MAX_POSITIONS * 2),
            count_nonzero,
            add_scaled,
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
            (b.px0 + b.pw).min(self.recon.xsize()),
            (b.py0 + b.ph).min(self.recon.ysize()),
        )
    }

    /// Add `sign` times the accumulated plane deltas to the frame reconstruction.
    fn apply_plane_deltas(&mut self, b: &PolishBlock, sign: f32) {
        let width = b.pw.min(self.recon.xsize() - b.px0);
        let height = b.ph.min(self.recon.ysize() - b.py0);
        for c in 0..3 {
            for y in 0..height {
                let delta = &self.plane_delta[c][y * b.pw..y * b.pw + width];
                let row = &mut self.recon.plane_row_mut(c, b.py0 + y)[b.px0..b.px0 + width];
                (self.add_scaled)(row, delta, sign);
            }
        }
    }

    /// Pixel-domain delta of `m` on the three decoded planes.
    fn plane_deltas(&mut self, ctx: &EncodingContext, b: &PolishBlock, m: &PolishMove) {
        let n = b.pw * b.ph;
        for plane in &mut self.plane_delta {
            plane[..n].fill(0.0);
        }
        let ch = &b.channels[m.c];
        let dq = (ch.bias.level(m.to) - ch.bias.level(m.from)) / (ch.inv_qm[m.idx] * ch.q_scaled);
        self.delta_coef[..b.size].fill(0.0);
        self.delta_coef[m.idx] = dq;
        crate::inflated_cost::reconstruct_error(
            ctx.idct,
            b.raw_strategy,
            &self.delta_coef[..b.size],
            &mut self.delta_pix[..b.size],
        );
        if m.c == 1 {
            for c in 0..3 {
                let f = b.cfl[c];
                if f == 0.0 {
                    continue;
                }
                for (d, &p) in self.plane_delta[c][..n]
                    .iter_mut()
                    .zip(&self.delta_pix[..n])
                {
                    *d += f * p;
                }
            }
        } else {
            self.plane_delta[m.c][..n].copy_from_slice(&self.delta_pix[..n]);
        }
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
        state.update_checkpointed(recon, rect, cp);
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
        let d = cv
            .judge
            .update_checkpointed(recon, &cv.dc, rect, &mut cv.cp);
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
        state.update(recon, rect);
        *base_d = state.deficit();
        if let Some(cv) = cv {
            cv.base = cv.judge.update(recon, &cv.dc, rect);
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
        // Baseline: this block's pre-polish pixels into the frame reconstruction
        // (earlier blocks' changed contexts can have moved its quantization).
        let (plane, lx0, ly0) = capture;
        let mut changed = false;
        for c in 0..3 {
            for y in rect.y0..rect.y1 {
                let src = &plane.plane_row(c, ly0 + (y - b.py0))[lx0..lx0 + (rect.x1 - rect.x0)];
                let dst = &mut self.recon.plane_row_mut(c, y)[rect.x0..rect.x1];
                if src != dst {
                    dst.copy_from_slice(src);
                    changed = true;
                }
            }
        }
        if changed {
            self.accept_rect(rect);
        }
        self.delta_coef
            .resize(self.delta_coef.len().max(b.size), 0.0);
        self.delta_pix.resize(self.delta_pix.len().max(b.size), 0.0);
        for plane in &mut self.plane_delta {
            plane.resize(plane.len().max(b.pw * b.ph), 0.0);
        }
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
        let row_stride = b.cx * 8;
        let mut nzeros: [usize; 3] = std::array::from_fn(|c| {
            count_nonzero_except_llf(quantized[c], b.cx, b.cy, self.count_nonzero)
        });
        let mut base_bits: [f32; 3] = std::array::from_fn(|c| {
            block_bits(quantized[c], &b.channels[c], covered, nzeros[c], prices)
        });

        for _round in 0..rounds {
            // ---- Proposals: proxy-ranked singles per channel ----
            let mut singles = [PolishMove::default(); PER_CHANNEL * 3];
            let mut nsingles = 0;
            let options = &mut self.options;
            for c in [1usize, 0, 2] {
                let ch = &b.channels[c];
                let block = &mut *quantized[c];
                options.clear();
                let end = b.size.min(covered + positions);
                for k in covered..end {
                    let idx = ch.scan[k] as usize;
                    let (v, u) = (idx / row_stride, idx % row_stride);
                    if v < b.cy && u < b.cx {
                        continue;
                    }
                    let cur = block[idx];
                    let ideal = ch.source[idx] * ch.inv_qm[idx] * ch.q_scaled;
                    let base_err = (ideal - ch.bias.level(cur)).powi(2);
                    let mut order = k * 2;
                    let mut push = |to: i32| {
                        let dd = ch.weight * ((ideal - ch.bias.level(to)).powi(2) - base_err);
                        options.push(PolishOption {
                            idx,
                            from: cur,
                            to,
                            dd,
                            order,
                        });
                        order += 1;
                    };
                    if cur != 0 {
                        push(cur - cur.signum());
                        push(cur + cur.signum());
                    } else if ideal.abs() >= RESTORE_MIN {
                        push(if ideal > 0.0 { 1 } else { -1 });
                    }
                }
                // Price the most promising by distortion, then rank by the proxy.
                rank_options_cap(options, price_cap);
                if options.is_empty() {
                    continue;
                }
                let prefixes = token_prefixes(block, ch, covered, nzeros[c], prices, options);
                let mut priced = [PolishMove::default(); PRICE_CAP];
                let mut npriced = 0;
                for (i, option) in options.iter().enumerate() {
                    let PolishOption {
                        idx, from, to, dd, ..
                    } = *option;
                    let dbits =
                        proposal_bits(block, ch, covered, nzeros[c], prices, option, prefixes[i])
                            - base_bits[c];
                    if dbits > MAX_COST_BITS {
                        continue;
                    }
                    priced[npriced] = PolishMove {
                        c,
                        idx,
                        from,
                        to,
                        dbits,
                        proxy: dd + b.lambda * dbits,
                    };
                    npriced += 1;
                }
                let priced = &mut priced[..npriced];
                priced.sort_by(|a, b| a.proxy.total_cmp(&b.proxy));
                let len = npriced.min(per_channel);
                singles[nsingles..nsingles + len].copy_from_slice(&priced[..len]);
                nsingles += len;
            }
            let singles = &singles[..nsingles];

            // ---- Judge: SSIMULACRA2 ranks, ColorVideoVDP checks the best few ----
            let mut scored = [(0.0f32, 0usize); PER_CHANNEL * 3];
            let mut nscored = 0;
            for (i, m) in singles.iter().enumerate() {
                let j = self.judge_ss2(ctx, b, rect, m);
                if j < -1e-6 {
                    scored[nscored] = (j, i);
                    nscored += 1;
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
            let at = (&mut *plane, lx0, ly0);
            self.commit(
                prices,
                b,
                rect,
                &m,
                at,
                quantized,
                &mut nzeros,
                &mut base_bits,
            );

            // ---- Further accepted proposals, re-priced and re-judged on the new block ----
            let mut touched = [(m.c, m.idx); EXTRA_TRIES + 1];
            let mut ntouched = 1;
            let mut tries = 0;
            for &(_, i) in scored.iter() {
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
                quantized[m.c][m.idx] = m.to;
                let nz = nzeros[m.c] + usize::from(m.to != 0) - usize::from(m.from != 0);
                let bits = block_bits(quantized[m.c], &b.channels[m.c], covered, nz, prices);
                quantized[m.c][m.idx] = m.from;
                m.dbits = bits - base_bits[m.c];
                if m.dbits > MAX_COST_BITS {
                    continue;
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
                let at = (&mut *plane, lx0, ly0);
                self.commit(
                    prices,
                    b,
                    rect,
                    &m,
                    at,
                    quantized,
                    &mut nzeros,
                    &mut base_bits,
                );
            }
        }
    }

    /// Commit `m`, whose plane deltas the last judge call prepared.
    #[allow(clippy::too_many_arguments)]
    fn commit(
        &mut self,
        prices: &FrozenTokenPrices,
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
            for y in rect.y0..rect.y1 {
                let src = &self.recon.plane_row(c, y)[rect.x0..rect.x1];
                plane.plane_row_mut(c, ly0 + (y - b.py0))[lx0..lx0 + (rect.x1 - rect.x0)]
                    .copy_from_slice(src);
            }
        }
        quantized[m.c][m.idx] = m.to;
        nzeros[m.c] = nzeros[m.c] + usize::from(m.to != 0) - usize::from(m.from != 0);
        base_bits[m.c] = block_bits(
            quantized[m.c],
            &b.channels[m.c],
            b.cx * b.cy,
            nzeros[m.c],
            prices,
        );
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
                    idx,
                    from,
                    to,
                    dd: ((k * 17) % 29) as f32,
                    order: k * 2 + direction,
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
    fn token_prefixes_preserve_full_walk_bits_for_all_shapes_and_moves() {
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
                            let prefixes =
                                token_prefixes(&block, &ch, covered, nzeros, &prices, &options);
                            for (i, option) in options.iter().enumerate() {
                                block[option.idx] = option.to;
                                let trial_nzeros =
                                    count_nonzero_except_llf(&block, cx, cy, count_nonzero_scalar);
                                let expected = reference_block_bits(
                                    &block,
                                    &ch,
                                    covered,
                                    trial_nzeros,
                                    &prices,
                                );
                                block[option.idx] = option.from;
                                let actual = proposal_bits(
                                    &mut block,
                                    &ch,
                                    covered,
                                    nzeros,
                                    &prices,
                                    option,
                                    prefixes[i],
                                );
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
                            block[accepted.idx] = accepted.to;
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
                            let prefixes = token_prefixes(
                                black_box(&block),
                                &ch,
                                covered,
                                nzeros,
                                &prices,
                                &options,
                            );
                            for (i, option) in options.iter().enumerate() {
                                sums[1] += black_box(proposal_bits(
                                    black_box(&mut block),
                                    &ch,
                                    covered,
                                    nzeros,
                                    &prices,
                                    option,
                                    prefixes[i],
                                ));
                            }
                        } else {
                            for option in &options {
                                block[option.idx] = option.to;
                                sums[0] += black_box(block_bits(
                                    black_box(&block),
                                    &ch,
                                    covered,
                                    nzeros,
                                    &prices,
                                ));
                                block[option.idx] = option.from;
                            }
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
        for c in 0..3 {
            let mut p = CoefPolisher::new(recon.clone(), reference.state(&recon), 1.0, None);
            p.delta_coef.resize(64, 0.0);
            p.delta_pix.resize(64, 0.0);
            for plane in &mut p.plane_delta {
                plane.resize(64, 123.0);
            }
            let rect = p.rect(&block);
            let selected = PolishMove {
                c,
                idx: 1,
                from: 0,
                to: 1,
                ..PolishMove::default()
            };
            assert!(p.judge_cv(&ctx, &block, rect, &selected).is_none());
            for plane in 0..3 {
                assert_eq!(
                    p.recon.plane(plane).as_slice(),
                    recon.plane(plane).as_slice()
                );
            }
            let prepared = p.plane_delta.clone();
            p.apply_plane_deltas(&block, 1.0);
            p.accept_rect(rect);
            for plane in 0..3 {
                for y in rect.y0..rect.y1 {
                    for x in rect.x0..rect.x1 {
                        let expected = recon.plane_row(plane, y)[x]
                            + prepared[plane][(y - block.py0) * 8 + x - block.px0];
                        assert_eq!(p.recon.plane_row(plane, y)[x].to_bits(), expected.to_bits());
                    }
                }
            }
            assert!(p.base_d.is_finite());
            assert!(prepared[c].iter().any(|&v| v != 0.0));
            assert!((p.base_d - reference.deficit(&p.recon)).abs() < 1e-3);
        }
    }

    fn check_kernels(count: CountNonzeroFn, add: AddScaledFn) {
        let values: Vec<i32> = (0..16386)
            .map(|i| [0, -1, i32::MIN, 0, i32::MAX, 7, 0][i % 7])
            .collect();
        for len in (0..129).chain([255, 256, 1023, 16384]) {
            let input = &values[1..1 + len];
            assert_eq!(count(input), count_nonzero_scalar(input), "len={len}");
            let delta: Vec<f32> = (0..len)
                .map(|i| [0.0, -0.0, f32::MIN_POSITIVE, 1e-8, -0.7, 1e10, -1e-10][i % 7])
                .collect();
            for sign in [-1.0, 1.0, 0.73] {
                let mut actual: Vec<f32> = (0..len + 2)
                    .map(|i| [0.0, -0.0, 0.3, -0.2, 1e-8, 1e10][i % 6])
                    .collect();
                let mut expected = actual.clone();
                add_scaled_scalar(&mut expected[1..len + 1], &delta, sign);
                add(&mut actual[1..len + 1], &delta, sign);
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "len={len}, sign={sign}"
                );
            }
        }
    }

    #[test]
    fn selected_kernels_preserve_scalar_bits_and_handle_tails() {
        let (count, add) = select_kernels();
        check_kernels(count, add);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_kernels_preserve_scalar_bits_and_handle_tails() {
        if is_x86_feature_detected!("avx2") {
            check_kernels(
                |v| unsafe { crate::avx::polish_count_nonzero_avx2(v) },
                |r, d, s| unsafe { crate::avx::polish_add_scaled_avx2(r, d, s) },
            );
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_kernels_preserve_scalar_bits_and_handle_tails() {
        if is_x86_feature_detected!("sse4.1") {
            check_kernels(
                |v| unsafe { crate::sse::polish_count_nonzero_sse41(v) },
                |r, d, s| unsafe { crate::sse::polish_add_scaled_sse41(r, d, s) },
            );
        }
    }

    #[test]
    fn partial_ranking_matches_stable_sort_including_ties() {
        for len in 0..=MAX_POSITIONS * 2 {
            let mut options: Vec<_> = (0..len)
                .map(|i| PolishOption {
                    idx: i,
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
                    ][(i * 31) % 7],
                    order: i,
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
    fn cached_nonzero_counts_match_full_recounts_for_every_move() {
        let (count, _) = select_kernels();
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
