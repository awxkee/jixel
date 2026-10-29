/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
//! AC coefficient trellis (RDOQ): a Viterbi pass over each block's scan that
//! re-picks quantized levels against the frozen token prices.

use super::DequantBias;
use crate::Arbiter;
use crate::ac_context::{
    fine_block_context, fine_non_zero_context, fine_zero_density_contexts_offset,
    zero_density_context, zero_density_context_8x8,
};
use crate::entropy::{FrozenTokenPrices, Token, pack_signed};
use crate::image::Image3F;
use crate::util::{FastRound, HeapMatrix, heap_array};

const RDOQ_MAX_STRIDE: usize = 2 * (256 + 1);

#[inline]
pub(crate) fn rdoq_channel_weights(coding: crate::CodingTransform) -> [f32; 3] {
    // The YCbCr weights improve rendered RD for both RGB and CMYK input.
    // RGB and adaptive XYB regress slightly with their reconstruction weights,
    // so keep the trellis's existing calibration for those transforms.
    match coding {
        crate::CodingTransform::YCbCr => coding.channel_weights(),
        crate::CodingTransform::Xyb | crate::CodingTransform::Rgb => {
            crate::inflated_cost::CHANNEL_WEIGHT
        }
    }
}

const RDOQ_MAX_CHOICES: usize = 64 * RDOQ_MAX_STRIDE;

/// Levels the trellis may pick for one coefficient. Without `restore` a level
/// only keeps, steps toward zero or drops, so a coefficient the deadzone
/// removed stays removed.
#[inline]
fn rdoq_candidates(ideal: f32, current: i32, restore: bool) -> ([i32; 5], usize) {
    if !restore {
        let toward_zero = current - current.signum();
        return match (current, toward_zero) {
            (0, _) => ([0; 5], 1),
            (_, 0) => ([current, 0, 0, 0, 0], 2),
            _ => ([current, toward_zero, 0, 0, 0], 3),
        };
    }
    let rounded = ideal.fast_round() as i32;
    if ideal.abs() > 2.5 && (ideal - rounded as f32).abs() < 0.25 {
        return ([current; 5], 1);
    }
    let proposed = [
        current,
        0,
        rounded,
        rounded.saturating_sub(1),
        rounded.saturating_add(1),
    ];
    let mut candidates = [current; 5];
    let mut len = 0;
    for level in proposed {
        if !candidates[..len].contains(&level) {
            candidates[len] = level;
            len += 1;
        }
    }
    (candidates, len)
}

#[inline]
fn rdoq_distortion_weight(window_index: usize, window_len: usize, distance: f32) -> f32 {
    let scan_position = window_index as f32 / window_len.max(1) as f32;
    let strength = (distance - 2.0).clamp(0.0, 1.0);
    1.0 + strength * (1.0 - scan_position).powi(2)
}

#[inline]
pub(crate) fn rdoq_lambda(arbiter: Arbiter, qf_ratio: f32) -> f32 {
    crate::ac_strategy::RD_LAMBDA * 0.25 * arbiter.tuning().rdoq_lambda * qf_ratio.clamp(0.5, 2.0)
}

#[inline(always)]
fn update_rdoq_state_pair(
    distortion: f32,
    token_cost: [f32; 2],
    tail: f32,
    candidate: u8,
    current: &mut [f32; 2],
    choices: &mut [u8; 2],
) {
    #[cfg(all(target_arch = "aarch64", feature = "neon", target_feature = "neon"))]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is enabled for this target. Each load/store accesses
        // exactly the two f32 elements of its array; no alignment is required.
        unsafe {
            let cost = vadd_f32(
                vadd_f32(vdup_n_f32(distortion), vld1_f32(token_cost.as_ptr())),
                vdup_n_f32(tail),
            );
            let old = vld1_f32(current.as_ptr());
            let better = vclt_f32(cost, old);
            vst1_f32(current.as_mut_ptr(), vbsl_f32(better, cost, old));
            // Narrow the two all-zero/all-one comparison lanes to the two
            // choice bytes, keeping exactly the same mask as the cost update.
            let mask16 = vmovn_u32(vcombine_u32(better, better));
            let mask8 = vmovn_u16(vcombine_u16(mask16, mask16));
            let mask = vget_lane_u16::<0>(vreinterpret_u16_u8(mask8));
            let old_choice = u16::from_ne_bytes(*choices);
            let new_choice = u16::from_ne_bytes([candidate; 2]);
            *choices = ((new_choice & mask) | (old_choice & !mask)).to_ne_bytes();
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon", target_feature = "neon")))]
    for prev in 0..2 {
        let cost = distortion + token_cost[prev] + tail;
        if cost < current[prev] {
            current[prev] = cost;
            choices[prev] = candidate;
        }
    }
}

/// Trellis window over the scan, and the prefix of it whose levels may be
/// restored; past that prefix a level only keeps, steps toward zero or drops.
fn rdoq_window_len(covered_blocks: usize) -> (usize, usize) {
    if covered_blocks == 1 {
        (63, 20)
    } else {
        (
            (64 * covered_blocks).min(1024),
            (24 * covered_blocks).min(256),
        )
    }
}

/// A block whose source has at most this many distinct colors per 8x8 is
/// graphic: flat fills with hard edges, where a coefficient the deadzone
/// zeroed reads as a stray blip once restored.
pub(super) const RESTORE_FEW_COLORS_PER_BLOCK: usize = 24;
/// Rate multiplier a restore must beat in such a block.
pub(super) const RESTORE_FEW_COLORS_PENALTY: f32 = 2.0;

/// Whether the source pixels of `[x0, x1) x [y0, y1)` hold at most `limit`
/// distinct XYB colors (quantized to 1/1024).
pub(super) fn few_colors(
    opsin: &Image3F,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    limit: usize,
    seen: &mut Vec<u64>,
) -> bool {
    let key_at = |rows: &[&[f32]; 3], x: usize| {
        rows.iter().fold(0u64, |k, r| {
            (k << 21) | (((r[x] * 1024.0) + 0.5) as i64 as u64 & 0x1f_ffff)
        })
    };
    seen.clear();
    // Large footprints: the linear scan is quadratic in the color limit.
    if limit > 64 {
        for y in y0..y1 {
            let rows = [
                opsin.plane_row(0, y),
                opsin.plane_row(1, y),
                opsin.plane_row(2, y),
            ];
            seen.extend((x0..x1).map(|x| key_at(&rows, x)));
        }
        seen.sort_unstable();
        seen.dedup();
        return seen.len() <= limit;
    }
    seen.reserve(limit);
    for y in y0..y1 {
        let rows = [
            opsin.plane_row(0, y),
            opsin.plane_row(1, y),
            opsin.plane_row(2, y),
        ];
        for x in x0..x1 {
            let key = key_at(&rows, x);
            if !seen.contains(&key) {
                if seen.len() == limit {
                    return false;
                }
                seen.push(key);
            }
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rdoq_block(
    channel_weight: f32,
    prices: &FrozenTokenPrices,
    scan: &[u32],
    source: &[f32],
    inv_qm: &[f32],
    q_scaled: f32,
    block: &mut [i32],
    raw_strategy: u8,
    strategy_code: u8,
    c: usize,
    bias: DequantBias,
    predicted: u8,
    cx: usize,
    cy: usize,
    distance: f32,
    qf_hi: bool,
    lambda: f32,
    restore_penalty: f32,
    choices: &mut [u8; RDOQ_MAX_CHOICES],
    costs: &mut [[f32; RDOQ_MAX_STRIDE]; 2],
) {
    let covered_blocks = cx * cy;
    let (window_cap, restore_len) = rdoq_window_len(covered_blocks);
    // The 128 family always takes the banded trellis: the full DP grows with
    // a block's nonzero count, thousands of states over 16384 coefficients.
    let legacy_limit = window_cap <= 64 && !crate::dc_group_data::is_128_strategy(raw_strategy);
    // Keep the original 8x8 loop. Factoring amortizes level selection over
    // shared count contexts in multi-block transforms. Separate instantiations
    // remove the unused loop and its scratch from each production path.
    macro_rules! refine {
        ($factored:literal) => {
            rdoq_block_banded::<$factored>(
                channel_weight,
                window_cap,
                restore_len,
                legacy_limit,
                prices,
                scan,
                source,
                inv_qm,
                q_scaled,
                block,
                strategy_code,
                c,
                bias,
                predicted,
                cx,
                cy,
                distance,
                qf_hi,
                lambda,
                restore_penalty,
                choices,
                costs,
            )
        };
    }
    if covered_blocks == 1 {
        refine!(false);
    } else {
        refine!(true);
    }
}

#[allow(clippy::too_many_arguments)]
fn rdoq_block_banded<const FACTORED: bool>(
    channel_weight: f32,
    window_cap: usize,
    restore_len: usize,
    legacy_limit: bool,
    prices: &FrozenTokenPrices,
    scan: &[u32],
    source: &[f32],
    inv_qm: &[f32],
    q_scaled: f32,
    block: &mut [i32],
    strategy_code: u8,
    c: usize,
    bias: DequantBias,
    predicted: u8,
    cx: usize,
    cy: usize,
    distance: f32,
    qf_hi: bool,
    lambda: f32,
    restore_penalty: f32,
    choices: &mut [u8; RDOQ_MAX_CHOICES],
    costs: &mut [[f32; RDOQ_MAX_STRIDE]; 2],
) {
    const MAX_NZERO_DELTA: usize = 6;
    let covered_blocks = cx * cy;
    let search_end = block.len().min(covered_blocks + window_cap);
    let window_len = search_end - covered_blocks;
    let base_window_len = (24 * covered_blocks)
        .min(64)
        .min(block.len() - covered_blocks);
    const BAND_STRIDE: usize = 2 * (2 * MAX_NZERO_DELTA + 2);
    assert!(window_len * BAND_STRIDE <= choices.len());

    let block_ctx = fine_block_context(c, strategy_code, qf_hi);
    let histo_offset = fine_zero_density_contexts_offset(block_ctx);
    let log2_covered_blocks = covered_blocks.trailing_zeros() as usize;
    let context = |remaining: usize, k: usize, prev: usize| -> u32 {
        histo_offset
            + if covered_blocks == 1 {
                zero_density_context_8x8(remaining, k, prev) as u32
            } else {
                zero_density_context(remaining, k, covered_blocks, log2_covered_blocks, prev) as u32
            }
    };

    // The suffix is fixed. Price it twice, once for each possible nonzero state
    // immediately before the suffix.
    let suffix_nzeros = (search_end..block.len())
        .filter(|&k| block[scan[k] as usize] != 0)
        .count();
    let mut suffix_cost = [0.0f32; 2];
    for (initial_prev, target) in suffix_cost.iter_mut().enumerate() {
        let mut remaining = suffix_nzeros;
        let mut prev = initial_prev;
        let mut k = search_end;
        while k < block.len() && remaining != 0 {
            let coef = block[scan[k] as usize];
            *target += lambda
                * prices.token_bits(Token::new(context(remaining, k, prev), pack_signed(coef)));
            prev = usize::from(coef != 0);
            remaining -= usize::from(coef != 0);
            k += 1;
        }
    }

    let max_nzeros = suffix_nzeros + window_len;
    // Keep the old dense-block skip for unchanged presets/windows. Expanded
    // windows store a local count band and can also refine dense suffixes.
    let legacy_stride = (max_nzeros + 1) * 2;
    if legacy_limit
        && (legacy_stride > RDOQ_MAX_STRIDE || window_len * legacy_stride > RDOQ_MAX_CHOICES)
    {
        return;
    }
    // Only +/- MAX_NZERO_DELTA around the original count is reachable at
    // each row. Store that band rather than every suffix nonzero count.
    let stride = BAND_STRIDE;
    let [next_buf, current_buf] = costs;
    let mut next = &mut next_buf[..stride];
    let mut current_cost = &mut current_buf[..stride];
    next[0] = suffix_cost[0];
    next[1] = suffix_cost[1];
    let mut next_min = suffix_nzeros;
    let mut next_max = suffix_nzeros;
    // Only the active interval of each row can be read. A finite state always
    // writes its choice, and backtracking only follows finite states, so the
    // inactive costs and the choice table do not need a per-block clear.

    let mut original_after_nzeros = 0usize;
    for window_index in (0..window_len).rev() {
        let k = covered_blocks + window_index;
        let idx = scan[k] as usize;
        let ideal = source[idx] * inv_qm[idx] * q_scaled;
        let distortion_weight = channel_weight
            * if c == 1 {
                1.0
            } else {
                chroma_rdoq_weight(distance)
            }
            * rdoq_distortion_weight(window_index.min(base_window_len), base_window_len, distance);
        let (candidates, candidate_count) =
            rdoq_candidates(ideal, block[idx], window_index < restore_len);
        // Distortion against the decoder's dequantization of each candidate.
        let mut candidate_distortion = [0.0f32; 5];
        for (d, &level) in candidate_distortion[..candidate_count]
            .iter_mut()
            .zip(candidates.iter())
        {
            *d = distortion_weight * (ideal - bias.level(level)).powi(2);
        }
        let processed_after = window_len - 1 - window_index;
        let min_remaining = suffix_nzeros + original_after_nzeros.saturating_sub(MAX_NZERO_DELTA);
        let max_remaining =
            suffix_nzeros + (original_after_nzeros + MAX_NZERO_DELTA).min(processed_after);
        // A candidate adds either zero or one nonzero. Clear precisely the
        // destination interval, and intersect the source interval with the
        // preceding row before reading it (scratch outside it can be stale).
        let current_min = min_remaining;
        let current_max = (max_remaining + 1).min(max_nzeros);
        current_cost[..(current_max - current_min + 1) * 2].fill(f32::INFINITY);
        if !FACTORED {
            for next_remaining in min_remaining.max(next_min)..=max_remaining.min(next_max) {
                for (candidate_index, &level) in candidates[..candidate_count].iter().enumerate() {
                    let nonzero = usize::from(level != 0);
                    let remaining = next_remaining + nonzero;
                    if remaining > max_nzeros {
                        continue;
                    }
                    let tail = next[(next_remaining - next_min) * 2 + nonzero];
                    if !tail.is_finite() {
                        continue;
                    }
                    let distortion = candidate_distortion[candidate_index];
                    let token_cost = if remaining == 0 {
                        [0.0; 2]
                    } else {
                        let bits =
                            prices.token_bits_pair(context(remaining, k, 0), pack_signed(level));
                        // Restoring a deadzone-zeroed coefficient pays the block's
                        // penalty.
                        let rate = if block[idx] == 0 && level != 0 {
                            lambda * restore_penalty
                        } else {
                            lambda
                        };
                        [rate * bits[0], rate * bits[1]]
                    };
                    let state = (remaining - current_min) * 2;
                    update_rdoq_state_pair(
                        distortion,
                        token_cost,
                        tail,
                        candidate_index as u8,
                        (&mut current_cost[state..state + 2]).try_into().unwrap(),
                        (&mut choices
                            [window_index * stride + state..window_index * stride + state + 2])
                            .try_into()
                            .unwrap(),
                    );
                }
            }
        } else {
            // Nonzero levels share a tail state. Choose their local minimum
            // once per count context, then compare zero/nonzero transitions
            // for each exact count; counts in one bucket cannot be merged.
            let source_min = min_remaining.max(next_min);
            let source_max = max_remaining.min(next_max);
            let zero_candidate = candidates[..candidate_count].iter().position(|&q| q == 0);
            let mut cached_context = u32::MAX;
            let mut local_cost = [[f32::INFINITY; 2]; 5];
            let mut nz_cost = [f32::INFINITY; 2];
            let mut nz_before = [f32::INFINITY; 2];
            let mut nz_choice = [u8::MAX; 2];
            for remaining in current_min..=current_max {
                let zero_tail = if zero_candidate.is_some()
                    && remaining >= source_min
                    && remaining <= source_max
                {
                    next[(remaining - next_min) * 2]
                } else {
                    f32::INFINITY
                };
                let nz_tail = if remaining > source_min && remaining - 1 <= source_max {
                    next[(remaining - 1 - next_min) * 2 + 1]
                } else {
                    f32::INFINITY
                };
                if !zero_tail.is_finite() && !nz_tail.is_finite() {
                    continue;
                }
                let ctx = if remaining == 0 {
                    u32::MAX - 1
                } else {
                    context(remaining, k, 0)
                };
                if ctx != cached_context {
                    cached_context = ctx;
                    nz_cost = [f32::INFINITY; 2];
                    nz_before = [f32::INFINITY; 2];
                    nz_choice = [u8::MAX; 2];
                    for (candidate_index, &level) in
                        candidates[..candidate_count].iter().enumerate()
                    {
                        let token_cost = if remaining == 0 {
                            [0.0; 2]
                        } else {
                            let bits = prices.token_bits_pair(ctx, pack_signed(level));
                            let rate = if block[idx] == 0 && level != 0 {
                                lambda * restore_penalty
                            } else {
                                lambda
                            };
                            [rate * bits[0], rate * bits[1]]
                        };
                        for prev in 0..2 {
                            let cost = candidate_distortion[candidate_index] + token_cost[prev];
                            local_cost[candidate_index][prev] = cost;
                            if level != 0 && cost < nz_cost[prev] {
                                nz_before[prev] = nz_cost[prev];
                                nz_cost[prev] = cost;
                                nz_choice[prev] = candidate_index as u8;
                            }
                        }
                    }
                }
                let state = (remaining - current_min) * 2;
                let row = window_index * stride + state;
                for prev in 0..2 {
                    // The original source-ascending loop sees nonzero edges before
                    // the zero edge into this destination, regardless of level order.
                    let mut cost = if nz_tail.is_finite() {
                        nz_cost[prev] + nz_tail
                    } else {
                        f32::INFINITY
                    };
                    let mut choice = nz_choice[prev];
                    // Adding a large common tail can round distinct local costs to
                    // the same f32. Recover the earliest original candidate only
                    // when the cheapest earlier candidate can tie the winner.
                    if cost.is_finite() && nz_before[prev] + nz_tail == cost {
                        for i in 0..choice as usize {
                            if candidates[i] != 0 && local_cost[i][prev] + nz_tail == cost {
                                choice = i as u8;
                                break;
                            }
                        }
                    }
                    if zero_tail.is_finite() {
                        let zero = zero_candidate.unwrap();
                        let zero_cost = local_cost[zero][prev] + zero_tail;
                        if zero_cost < cost {
                            cost = zero_cost;
                            choice = zero as u8;
                        }
                    }
                    current_cost[state + prev] = cost;
                    if cost.is_finite() {
                        choices[row + prev] = choice;
                    }
                }
            }
        }
        std::mem::swap(&mut current_cost, &mut next);
        next_min = current_min;
        next_max = current_max;
        original_after_nzeros += usize::from(block[idx] != 0);
    }

    let nzero_ctx = fine_non_zero_context(predicted as u32, block_ctx);
    let mut best_remaining = 0;
    let mut best_cost = f32::INFINITY;
    for remaining in next_min..=next_max {
        let initial_prev = usize::from(remaining <= block.len() / 16);
        let tail = next[(remaining - next_min) * 2 + initial_prev];
        if !tail.is_finite() {
            continue;
        }
        let cost = tail + lambda * prices.token_bits(Token::new(nzero_ctx, remaining as u32));
        if cost < best_cost {
            best_cost = cost;
            best_remaining = remaining;
        }
    }

    let mut remaining = best_remaining;
    let mut prev = usize::from(remaining <= block.len() / 16);
    let mut original_remaining = original_after_nzeros;
    for window_index in 0..window_len {
        let k = covered_blocks + window_index;
        let idx = scan[k] as usize;
        let ideal = source[idx] * inv_qm[idx] * q_scaled;
        let (candidates, candidate_count) =
            rdoq_candidates(ideal, block[idx], window_index < restore_len);
        original_remaining -= usize::from(block[idx] != 0);
        let row_min = suffix_nzeros + original_remaining.saturating_sub(MAX_NZERO_DELTA);
        let choice = choices[window_index * stride + (remaining - row_min) * 2 + prev] as usize;
        debug_assert!(choice < candidate_count);
        let level = candidates[choice];
        block[idx] = level;
        let nonzero = usize::from(level != 0);
        remaining -= nonzero;
        prev = nonzero;
    }
}

const CHROMA_RDOQ_WEIGHT_HQ: f32 = 2.0;
const CHROMA_RDOQ_WEIGHT_D0: f32 = 2.25;
const CHROMA_RDOQ_WEIGHT_D1: f32 = 4.0;

/// Chroma trellis distortion-weight multiplier at `distance`.
#[inline]
pub(crate) fn chroma_rdoq_weight(distance: f32) -> f32 {
    let t = ((distance - CHROMA_RDOQ_WEIGHT_D0) / (CHROMA_RDOQ_WEIGHT_D1 - CHROMA_RDOQ_WEIGHT_D0))
        .clamp(0.0, 1.0);
    CHROMA_RDOQ_WEIGHT_HQ + t * (1.0 - CHROMA_RDOQ_WEIGHT_HQ)
}

pub(super) struct RdoqScratch {
    pub(super) choices: Box<[u8; RDOQ_MAX_CHOICES]>,
    pub(super) costs: HeapMatrix<f32, 2, RDOQ_MAX_STRIDE>,
}

impl Default for RdoqScratch {
    fn default() -> Self {
        Self {
            choices: heap_array(u8::MAX),
            costs: HeapMatrix::new(f32::INFINITY),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn few_colors_counts_distinct_source_colors() {
        use crate::image::Image3F;
        let mut xyb = Image3F::new(16, 8);
        // Left 8x8: two flat halves (2 colors). Right 8x8: a ramp.
        for c in 0..3 {
            for y in 0..8 {
                let row = xyb.plane_row_mut(c, y);
                for x in 0..8 {
                    row[x] = if x < 4 { 0.1 } else { 0.5 };
                    row[8 + x] = (y * 8 + x) as f32 * 0.01;
                }
            }
        }
        let mut seen = vec![u64::MAX; 100];
        let storage = seen.as_ptr();
        assert!(super::few_colors(&xyb, 0, 0, 8, 8, 24, &mut seen));
        assert!(super::few_colors(&xyb, 0, 0, 8, 8, 2, &mut seen));
        assert!(!super::few_colors(&xyb, 0, 0, 8, 8, 1, &mut seen));
        assert!(!super::few_colors(&xyb, 8, 0, 16, 8, 24, &mut seen));
        assert!(super::few_colors(&xyb, 8, 0, 16, 8, 64, &mut seen));
        assert!(!super::few_colors(&xyb, 0, 0, 8, 8, 0, &mut seen));
        assert!(super::few_colors(&xyb, 0, 0, 0, 0, 0, &mut seen));
        assert!(seen.is_empty());
        assert_eq!(seen.as_ptr(), storage, "color storage should be reused");
    }

    #[test]
    fn rdoq_state_pair_preserves_scalar_cost_bits_and_choices() {
        let special = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_5678),
        ];
        let mut seed = 73u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            if seed.is_multiple_of(4) {
                special[seed as usize % special.len()]
            } else {
                f32::from_bits(seed)
            }
        };
        let mut masks = [false; 4];
        for case in 0..16_384 {
            let distortion = random();
            let token_cost = [random(), random()];
            let tail = random();
            let cost = token_cost.map(|price| distortion + price + tail);
            let mut initial = [random(), random()];
            // Deliberate equal-cost candidates must preserve both the prior
            // choices and their exact cost bits, including signed zeros.
            if case % 8 == 0 {
                initial = cost;
            }
            let initial_choices = [case as u8, (case >> 3) as u8];
            let candidate = (case % 5) as u8;
            let mut expected = initial;
            let mut expected_choices = initial_choices;
            let mut mask = 0;
            for prev in 0..2 {
                if cost[prev] < expected[prev] {
                    expected[prev] = cost[prev];
                    expected_choices[prev] = candidate;
                    mask |= 1 << prev;
                }
            }
            masks[mask] = true;
            let mut actual = initial;
            let mut actual_choices = initial_choices;
            super::update_rdoq_state_pair(
                distortion,
                token_cost,
                tail,
                candidate,
                &mut actual,
                &mut actual_choices,
            );
            assert_eq!(
                actual.map(f32::to_bits),
                expected.map(f32::to_bits),
                "costs in case {case}"
            );
            assert_eq!(actual_choices, expected_choices, "choices in case {case}");
        }
        assert_eq!(masks, [true; 4]);
    }

    #[test]
    fn rdoq_reused_scratch_matches_fresh_scratch() {
        use crate::dc_group_data::STRATEGY_CODE_LUT;
        use crate::entropy::{FrozenTokenPrices, Token, build_entropy_code_no_cluster};

        let tokens: Vec<_> = (0..4096)
            .map(|i| Token::new(i % 2, if i % 3 == 0 { 0 } else { i % 17 }))
            .collect();
        let mut code = build_entropy_code_no_cluster(&tokens, 2, &mut Vec::new());
        code.context_map = (0..crate::ac_context::K_NUM_FINE_AC_CONTEXTS)
            .map(|i| (i % 2) as u8)
            .collect();
        let prices = FrozenTokenPrices::new(&code);
        let orders = crate::coeff_order::CoeffOrders::natural();
        let mut choices = super::heap_array(0xA5u8);
        let mut costs = Box::new([[f32::NAN; super::RDOQ_MAX_STRIDE]; 2]);
        let mut seed = 73u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut changed = 0;
        for case in 0..384 {
            let (raw, cx, cy) = [
                (crate::dc_group_data::STRATEGY_DCT, 1, 1),
                (crate::dc_group_data::STRATEGY_DCT16X8, 2, 1),
                (crate::dc_group_data::STRATEGY_DCT16X16, 2, 2),
                (crate::dc_group_data::STRATEGY_DCT32X16, 4, 2),
                (crate::dc_group_data::STRATEGY_DCT32X32, 4, 4),
            ][case % 5];
            let strategy = STRATEGY_CODE_LUT[raw as usize];
            let c = case % 3;
            let coding = [
                crate::CodingTransform::Xyb,
                crate::CodingTransform::YCbCr,
                crate::CodingTransform::Rgb,
            ][case / 3 % 3];
            let mut scan = orders.scan_for(strategy, c).to_vec();
            let size = cx * cy * 64;
            if case % 2 == 0 {
                // Learned AC orders need not follow spatial frequency.
                for i in cx * cy..size {
                    let j = cx * cy + random() as usize % (size - cx * cy);
                    scan.swap(i, j);
                }
            }
            let inv_qm = vec![1.; size];
            let source: Vec<f32> = (0..size)
                .map(|_| {
                    if random() % 10 < (case % 10) as u32 {
                        0.
                    } else {
                        (random() % 1024) as f32 / 128. - 4.
                    }
                })
                .collect();
            let original: Vec<i32> = source.iter().map(|v| v.round() as i32).collect();
            let mut expected = original.clone();
            let mut actual = original.clone();
            let mut fresh_choices = super::heap_array(u8::MAX);
            let mut fresh_costs = Box::new([[f32::INFINITY; super::RDOQ_MAX_STRIDE]; 2]);
            let run = |block: &mut [i32], choices: &mut _, costs: &mut _| {
                super::rdoq_block(
                    super::rdoq_channel_weights(coding)[c],
                    &prices,
                    &scan,
                    &source,
                    &inv_qm,
                    1.,
                    block,
                    raw,
                    strategy,
                    c,
                    super::DequantBias::DEFAULT[c],
                    (case % 64) as u8,
                    cx,
                    cy,
                    [1., 2., 3., 4.][case % 4],
                    case % 2 == 0,
                    super::rdoq_lambda(crate::Arbiter::Ssimulacra2, [0.5, 1.0, 1.2, 2.0][case % 4]),
                    1.0,
                    choices,
                    costs,
                );
            };
            run(&mut expected, &mut fresh_choices, &mut fresh_costs);
            run(&mut actual, &mut choices, &mut costs);
            assert_eq!(actual, expected, "scratch reuse in case {case}");
            changed += usize::from(actual != original);
        }
        assert!(
            changed > 100,
            "exercise finite trellis paths, including changed blocks"
        );
    }
    #[test]
    fn rdoq_window_policy() {
        for covered in [1, 2, 4, 8, 16, 32, 64] {
            // Past the restoring prefix a multi-block window only lowers levels.
            let expected = if covered == 1 {
                (63, 20)
            } else {
                ((64 * covered).min(1024), (24 * covered).min(256))
            };
            assert_eq!(super::rdoq_window_len(covered), expected);
        }
    }

    #[test]
    fn factored_rdoq_preserves_cost_bits_choices_and_restore_penalties() {
        use crate::dc_group_data::STRATEGY_CODE_LUT;
        use crate::entropy::{FrozenTokenPrices, Token, build_entropy_code_no_cluster};

        let tokens: Vec<_> = (0..16_384)
            .map(|i| {
                let context = i % 4;
                let value = if i % 7 < context + 1 {
                    0
                } else {
                    (i / 4) % 601
                };
                Token::new(context, value)
            })
            .collect();
        let mut code = build_entropy_code_no_cluster(&tokens, 4, &mut Vec::new());
        code.context_map = (0..crate::ac_context::K_NUM_FINE_AC_CONTEXTS)
            .map(|i| ((i / 2 + i % 2) % 4) as u8)
            .collect();
        let prices = FrozenTokenPrices::new(&code);
        let orders = crate::coeff_order::CoeffOrders::natural();
        let mut choices0 = super::heap_array(0xA5u8);
        let mut choices1 = super::heap_array(0xA5u8);
        let mut costs0 = Box::new([[f32::NAN; super::RDOQ_MAX_STRIDE]; 2]);
        let mut costs1 = costs0.clone();
        let mut seed = 173u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let shapes = [
            (crate::dc_group_data::STRATEGY_DCT, 1, 1),
            (crate::dc_group_data::STRATEGY_DCT16X8, 2, 1),
            (crate::dc_group_data::STRATEGY_DCT16X16, 2, 2),
            (crate::dc_group_data::STRATEGY_DCT32X16, 4, 2),
            (crate::dc_group_data::STRATEGY_DCT32X32, 4, 4),
            (crate::dc_group_data::STRATEGY_DCT64X32, 8, 4),
            (crate::dc_group_data::STRATEGY_DCT64X64, 8, 8),
            #[cfg(feature = "extraslow")]
            (crate::dc_group_data::STRATEGY_DCT128X64, 16, 8),
            #[cfg(feature = "extraslow")]
            (crate::dc_group_data::STRATEGY_DCT128X128, 16, 16),
        ];
        let mut changed = 0;
        for case in 0..144 {
            let (raw, cx, cy) = shapes[case % shapes.len()];
            let strategy = STRATEGY_CODE_LUT[raw as usize];
            let c = case / shapes.len() % 3;
            let size = cx * cy * 64;
            let mut scan = orders.scan_for(strategy, c).to_vec();
            if case % 2 == 0 {
                for i in cx * cy..size {
                    let j = cx * cy + random() as usize % (size - cx * cy);
                    scan.swap(i, j);
                }
            }
            let inv_qm: Vec<_> = (0..size).map(|i| 0.75 + (i % 9) as f32 / 16.).collect();
            let source: Vec<f32> = (0..size)
                .map(|_| {
                    let r = random();
                    if r % 10 < (case / shapes.len() % 10) as u32 {
                        0.
                    } else if case % 7 == 0 {
                        (r % 4001) as f32 / 8. - 250.
                    } else {
                        (r % 1024) as f32 / 128. - 4.
                    }
                })
                .collect();
            let original: Vec<i32> = source.iter().map(|v| v.round() as i32).collect();
            for cap in [64, 256, 1024] {
                let mut expected = original.clone();
                let mut actual = original.clone();
                let run = |factored, block: &mut [i32], choices: &mut _, costs: &mut _| {
                    let refine = if factored {
                        super::rdoq_block_banded::<true>
                    } else {
                        super::rdoq_block_banded::<false>
                    };
                    refine(
                        [0.3, 1., 0.6][c],
                        cap,
                        cap,
                        false,
                        &prices,
                        &scan,
                        &source,
                        &inv_qm,
                        [0.75, 1.5][case % 2],
                        block,
                        strategy,
                        c,
                        super::DequantBias::DEFAULT[c],
                        (case % 256) as u8,
                        cx,
                        cy,
                        [0.3, 2., 4., 8.][case % 4],
                        case % 2 == 0,
                        super::rdoq_lambda(crate::Arbiter::Ssimulacra2, [0.5, 1.2][case % 2])
                            / [1., 0.125][case % 2],
                        [1., 2.][case / shapes.len() % 2],
                        choices,
                        costs,
                    );
                };
                run(false, &mut expected, &mut choices0, &mut costs0);
                run(true, &mut actual, &mut choices1, &mut costs1);
                assert_eq!(actual, expected, "coefficients, case={case}, cap={cap}");
                assert_eq!(
                    choices0.as_slice(),
                    choices1.as_slice(),
                    "choices, case={case}, cap={cap}"
                );
                for (&a, &b) in costs0.iter().flatten().zip(costs1.iter().flatten()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "cost bits, case={case}, cap={cap}"
                    );
                }
                changed += usize::from(actual != original);
            }
        }
        assert!(changed > 100, "must exercise changed blocks: {changed}");
    }

    #[test]
    fn factored_rdoq_preserves_tail_rounding_ties() {
        use crate::dc_group_data::STRATEGY_CODE_LUT;
        use crate::entropy::{
            FrozenTokenPrices, Token, build_entropy_code_no_cluster, pack_signed,
        };

        let mut tokens = vec![Token::new(0, 0); 1000];
        tokens.extend((0..100).flat_map(|_| [Token::new(0, 1), Token::new(0, 2)]));
        let mut code = build_entropy_code_no_cluster(&tokens, 1, &mut Vec::new());
        code.context_map = vec![0; crate::ac_context::K_NUM_FINE_AC_CONTEXTS];
        let prices = FrozenTokenPrices::new(&code);
        let orders = crate::coeff_order::CoeffOrders::natural();
        let raw = crate::dc_group_data::STRATEGY_DCT16X8;
        let strategy = STRATEGY_CODE_LUT[raw as usize];
        let scan = orders.scan_for(strategy, 1);
        let inv_qm = [1.; 128];
        let mut collapsed = 0;
        for i in 1..100 {
            let ideal = i as f32 * 1e-8;
            let mut source = [0.; 128];
            source[scan[2] as usize] = ideal;
            let mut expected = [0; 128];
            for &idx in &scan[3..54] {
                source[idx as usize] = 1.;
                expected[idx as usize] = 1;
            }
            let mut actual = expected;
            let mut choices0 = super::heap_array(0xA5);
            let mut choices1 = super::heap_array(0xA5);
            let mut costs0 = Box::new([[f32::NAN; super::RDOQ_MAX_STRIDE]; 2]);
            let mut costs1 = costs0.clone();
            let run = |factored, block: &mut [i32], choices: &mut _, costs: &mut _| {
                let refine = if factored {
                    super::rdoq_block_banded::<true>
                } else {
                    super::rdoq_block_banded::<false>
                };
                refine(
                    1.,
                    1,
                    1,
                    false,
                    &prices,
                    scan,
                    &source,
                    &inv_qm,
                    1.,
                    block,
                    strategy,
                    1,
                    super::DequantBias::DEFAULT[1],
                    17,
                    2,
                    1,
                    1.,
                    false,
                    super::rdoq_lambda(crate::Arbiter::Ssimulacra2, 1.),
                    2.,
                    choices,
                    costs,
                );
            };
            run(false, &mut expected, &mut choices0, &mut costs0);
            run(true, &mut actual, &mut choices1, &mut costs1);
            assert_eq!(actual, expected);
            assert_eq!(choices0.as_slice(), choices1.as_slice());
            for (&a, &b) in costs0.iter().flatten().zip(costs1.iter().flatten()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            let rate = 2. * super::rdoq_lambda(crate::Arbiter::Ssimulacra2, 1.);
            let minus = (ideal - super::DequantBias::DEFAULT[1].level(-1)).powi(2)
                + rate * prices.token_bits(Token::new(0, pack_signed(-1)));
            let plus = (ideal - super::DequantBias::DEFAULT[1].level(1)).powi(2)
                + rate * prices.token_bits(Token::new(0, pack_signed(1)));
            let tail = costs0[0][1];
            if plus < minus && plus + tail == minus + tail {
                collapsed += 1;
                // Local +1 is cheaper, but the original loop sees -1 first
                // and keeps it when their totals round to the same f32.
                assert_eq!(choices1[2], 1);
                assert_eq!(choices1[3], 1);
            }
        }
        assert!(
            collapsed > 0,
            "must exercise distinct local costs with equal totals"
        );
    }
}
