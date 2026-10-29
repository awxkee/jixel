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
#![allow(clippy::excessive_precision)]

use crate::Arbiter;
use crate::ac_context::{
    fine_block_context, fine_non_zero_context, fine_zero_density_contexts_offset,
    zero_density_context, zero_density_context_8x8,
};
use crate::dc_group_data::{
    AcStrategyImage, DcGroupData, STRATEGY_AFV0, STRATEGY_AFV1, STRATEGY_AFV2, STRATEGY_AFV3,
    STRATEGY_DCT, STRATEGY_DCT2X2, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4,
    STRATEGY_DCT8X16, STRATEGY_DCT8X32, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32,
    STRATEGY_DCT32X8, STRATEGY_DCT32X16, STRATEGY_DCT32X32, STRATEGY_DCT32X64, STRATEGY_DCT64X32,
    STRATEGY_DCT64X64, STRATEGY_IDENTITY,
};
#[cfg(feature = "extraslow")]
use crate::dc_group_data::{STRATEGY_DCT64X128, STRATEGY_DCT128X64, STRATEGY_DCT128X128};
use crate::dct::{DctInput, dc_from_dct8x16, dc_from_dct16x8, dc_from_dct16x16, fmla};
use crate::encoding_context::EncodingContext;
use crate::entropy::{FrozenTokenPrices, Token, pack_signed};
use crate::image::{Image3B, Image3F, Image3S, Rect};
use crate::inflated_cost::Footprint;
use crate::quant_weights::INV_DC_QUANT;
use crate::util::{FastRound, HeapMatrix, heap_array, round_i32};
use std::sync::OnceLock;

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

const K_GROUP_DIM_IN_BLOCKS: usize = 32;

// The 128 family is placed only by ExtraSlow's closed loop. Keep ordinary
// encodes' per-worker buffers sized for the largest enabled transform.
pub(crate) const MAX_BLOCK_COEFFICIENTS: usize = if cfg!(feature = "extraslow") {
    128 * 128
} else {
    64 * 64
};

#[inline]
pub(crate) fn predict_from_top_and_left(
    row_top: Option<&[u8]>,
    row: &[u8],
    x: usize,
    default_val: u8,
) -> u8 {
    if x == 0 {
        match row_top {
            Some(t) => t[x],
            None => default_val,
        }
    } else if row_top.is_none() {
        row[x - 1]
    } else {
        (row_top.unwrap()[x] as u16 + row[x - 1] as u16).div_ceil(2) as u8
    }
}

/// Number of non-zero coefficients in an 8×8 quantized block, excluding the
/// DC position (k=0). Returns the count for use as nzeros token.
#[inline]
fn num_nonzero_except_dc(block: &[i32; 64]) -> i32 {
    let mut count: i32 = 0;
    for &z in block[1..64].iter() {
        if z != 0 {
            count += 1;
        }
    }
    count
}

/// Multi-block analog: counts nonzeros in `size = cx*cy*64` coefficients
/// excluding the `cx × cy` LLF positions. For DCT16X8/DCT8X16 (cx=2, cy=1
/// after the cx>=cy swap), the LLF positions are coeffs[0] and coeffs[1] in
/// the 8x16 stride-16 layout — contiguous. For DCT16X16 (cx=cy=2) the LLF
/// positions are {0, 1, 16, 17} in the 16x16 stride-16 layout — NOT
/// contiguous, so we mask them explicitly using the row-stride.
#[inline]
fn num_nonzero_except_llf(block: &[i32], cx: usize, cy: usize) -> i32 {
    let row_stride = cx * 8;
    let xsize_pixels = cx * 8;
    let ysize_pixels = cy * 8;
    let mut count: i32 = 0;
    for v in 0..ysize_pixels {
        for u in 0..xsize_pixels {
            if v < cy && u < cx {
                continue;
            }
            if block[v * row_stride + u] != 0 {
                count += 1;
            }
        }
    }
    count
}

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
const RESTORE_FEW_COLORS_PER_BLOCK: usize = 24;
/// Rate multiplier a restore must beat in such a block.
const RESTORE_FEW_COLORS_PENALTY: f32 = 2.0;

/// Whether the source pixels of `[x0, x1) x [y0, y1)` hold at most `limit`
/// distinct XYB colors (quantized to 1/1024).
fn few_colors(
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
fn rdoq_block(
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn rdoq_block_wide_oracle(
    channel_weight: f32,
    window_cap: usize,
    legacy_limit: bool,
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
    choices: &mut [u8],
    costs: &mut [Vec<f32>; 2],
) {
    const MAX_NZERO_DELTA: usize = 6;
    if !matches!(
        raw_strategy,
        STRATEGY_DCT
            | STRATEGY_DCT16X8
            | STRATEGY_DCT8X16
            | STRATEGY_DCT16X16
            | STRATEGY_DCT32X32
            | STRATEGY_DCT32X16
            | STRATEGY_DCT16X32
            | STRATEGY_DCT32X8
            | STRATEGY_DCT8X32
            | STRATEGY_DCT64X64
            | STRATEGY_DCT64X32
            | STRATEGY_DCT32X64
    ) {
        return;
    }
    let covered_blocks = cx * cy;
    let search_end = block.len().min(covered_blocks + window_cap);
    let window_len = search_end - covered_blocks;

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
    let stride = (max_nzeros + 1) * 2;
    // The DP buffers are sized for the <=16x16 worst case. Denser suffixes
    // (the 32-family at very fine quantization) skip RDOQ at runtime: that is
    // the regime where hard thresholding is closest to optimal anyway, while
    // the sparse 32s that dominate smooth content fit comfortably.
    if legacy_limit && (stride > RDOQ_MAX_STRIDE || window_len * stride > RDOQ_MAX_CHOICES) {
        return;
    }
    assert!(costs[0].len() >= stride);
    assert!(choices.len() >= window_len * stride);
    let [next_buf, current_buf] = costs;
    let mut next = &mut next_buf[..stride];
    let mut current_cost = &mut current_buf[..stride];
    next[suffix_nzeros * 2] = suffix_cost[0];
    next[suffix_nzeros * 2 + 1] = suffix_cost[1];
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
            * rdoq_distortion_weight(
                window_index.min((24 * covered_blocks).min(64)),
                (24 * covered_blocks).min(64),
                distance,
            );
        let (candidates, candidate_count) = rdoq_candidates(ideal, block[idx], true);
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
        current_cost[current_min * 2..(current_max + 1) * 2].fill(f32::INFINITY);
        for next_remaining in min_remaining.max(next_min)..=max_remaining.min(next_max) {
            for (candidate_index, &level) in candidates[..candidate_count].iter().enumerate() {
                let nonzero = usize::from(level != 0);
                let remaining = next_remaining + nonzero;
                if remaining > max_nzeros {
                    continue;
                }
                let tail = next[next_remaining * 2 + nonzero];
                if !tail.is_finite() {
                    continue;
                }
                let distortion = candidate_distortion[candidate_index];
                let token_cost = if remaining == 0 {
                    [0.0; 2]
                } else {
                    let bits = prices.token_bits_pair(context(remaining, k, 0), pack_signed(level));
                    [lambda * bits[0], lambda * bits[1]]
                };
                let state = remaining * 2;
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
        let tail = next[remaining * 2 + initial_prev];
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
    for window_index in 0..window_len {
        let k = covered_blocks + window_index;
        let idx = scan[k] as usize;
        let ideal = source[idx] * inv_qm[idx] * q_scaled;
        let (candidates, candidate_count) = rdoq_candidates(ideal, block[idx], true);
        let choice = choices[window_index * stride + remaining * 2 + prev] as usize;
        debug_assert!(choice < candidate_count);
        let level = candidates[choice];
        block[idx] = level;
        let nonzero = usize::from(level != 0);
        remaining -= nonzero;
        prev = nonzero;
    }
}

/// Generalized AC quantization for an `xsize × ysize` (in 8x8 blocks) region.
/// `xsize=1, ysize=1` reproduces the 8×8 path. The block buffer has
/// `xsize*ysize*64` floats in row-major order with row stride `xsize*8`.
///
/// Thresholds mirror libjxl-tiny: per-quadrant in the 8×8 case; biased lower
/// for multi-block; for the 1×N or N×1 case the second half of the columns
/// (or rows for ysize=1, xsize=1 special) uses the second threshold pair.
pub(crate) type QuantizeBlockAcFn = fn(
    block_in: &[f32],
    c: usize,
    qm: &[f32],
    quant: i32,
    scale: f32,
    qm_multiplier: f32,
    distance: f32,
    xsize: usize,
    ysize: usize,
    block_out: &mut [i32],
);

pub(crate) type QuantizeDcFn = fn(input: &[f32], scale: f32, output: &mut [i16]);
pub(crate) type QuantizeDcCflFn =
    fn(input: &[f32], y_quant: &[i16], scale: f32, cfl: f32, output: &mut [i16]);
pub(crate) type QuantizeDcI32Fn = fn(input: &[f32], scale: f32, output: &mut [i32]);
pub(crate) type QuantizeDcCflI32Fn =
    fn(input: &[f32], y_quant: &[i32], scale: f32, cfl: f32, output: &mut [i32]);

#[derive(Clone, Copy)]
pub(crate) struct QuantizeDcMethods {
    pub(crate) quantize: QuantizeDcFn,
    pub(crate) quantize_cfl: QuantizeDcCflFn,
    pub(crate) quantize_i32: QuantizeDcI32Fn,
    pub(crate) quantize_cfl_i32: QuantizeDcCflI32Fn,
}

/// The DC values of one channel's coefficient block, one per covered 8x8
/// block in `iy * cov_x + ix` order. Single-block transforms carry their DC in
/// `coeffs[0]`; larger ones derive it from the lowest frequencies.
pub(crate) fn dc_from_coefficients(
    ctx: &EncodingContext,
    raw_strategy: u8,
    coeffs: &[f32],
    dc: &mut [f32],
) {
    match raw_strategy {
        STRATEGY_DCT
        | STRATEGY_IDENTITY
        | STRATEGY_DCT2X2
        | STRATEGY_DCT4X4
        | STRATEGY_DCT4X8
        | STRATEGY_DCT8X4
        | STRATEGY_AFV0..=STRATEGY_AFV3 => dc[0] = coeffs[0],
        STRATEGY_DCT16X8 => dc_from_dct16x8(
            coeffs.first_chunk::<128>().unwrap(),
            dc.first_chunk_mut::<2>().unwrap(),
        ),
        STRATEGY_DCT8X16 => dc_from_dct8x16(
            coeffs.first_chunk::<128>().unwrap(),
            dc.first_chunk_mut::<2>().unwrap(),
        ),
        // [TL, TR, BL, BR], the [iy][ix] grid.
        STRATEGY_DCT16X16 => dc_from_dct16x16(
            coeffs.first_chunk::<256>().unwrap(),
            dc.first_chunk_mut::<4>().unwrap(),
        ),
        STRATEGY_DCT32X32 => (ctx.dc_from_dct32x32)(
            coeffs.first_chunk::<1024>().unwrap(),
            dc.first_chunk_mut::<16>().unwrap(),
        ),
        STRATEGY_DCT64X64 => (ctx.dc_from_dct64x64)(
            coeffs.first_chunk::<4096>().unwrap(),
            dc.first_chunk_mut::<64>().unwrap(),
        ),
        STRATEGY_DCT64X32 => (ctx.dc_from_dct64x32)(
            coeffs.first_chunk::<2048>().unwrap(),
            dc.first_chunk_mut::<32>().unwrap(),
        ),
        STRATEGY_DCT32X64 => (ctx.dc_from_dct32x64)(
            coeffs.first_chunk::<2048>().unwrap(),
            dc.first_chunk_mut::<32>().unwrap(),
        ),
        // 4 rows x 2 columns (cov_x = 2, cov_y = 4).
        STRATEGY_DCT32X16 => (ctx.dc_from_dct32x16)(
            coeffs.first_chunk::<512>().unwrap(),
            dc.first_chunk_mut::<8>().unwrap(),
        ),
        // 2 rows x 4 columns (cov_x = 4, cov_y = 2).
        STRATEGY_DCT16X32 => (ctx.dc_from_dct16x32)(
            coeffs.first_chunk::<512>().unwrap(),
            dc.first_chunk_mut::<8>().unwrap(),
        ),
        // Four values along the long axis, a 4x1 column or a 1x4 row.
        STRATEGY_DCT32X8 | STRATEGY_DCT8X32 => (ctx.dc_from_dct8x32)(
            coeffs.first_chunk::<256>().unwrap(),
            dc.first_chunk_mut::<4>().unwrap(),
        ),
        #[cfg(feature = "extraslow")]
        STRATEGY_DCT128X128 => (ctx.dc_from_dct128x128)(
            coeffs.first_chunk::<16384>().unwrap(),
            dc.first_chunk_mut::<256>().unwrap(),
        ),
        #[cfg(feature = "extraslow")]
        STRATEGY_DCT128X64 => (ctx.dc_from_dct128x64)(
            coeffs.first_chunk::<8192>().unwrap(),
            dc.first_chunk_mut::<128>().unwrap(),
        ),
        #[cfg(feature = "extraslow")]
        STRATEGY_DCT64X128 => (ctx.dc_from_dct64x128)(
            coeffs.first_chunk::<8192>().unwrap(),
            dc.first_chunk_mut::<128>().unwrap(),
        ),
        _ => unreachable!("invalid raw strategy {raw_strategy}"),
    }
}

/// Wide levels used by DC-step pricing and the plain modular lattice. Keep
/// Rust's ties-away rounding and saturating float-to-integer conversion.
#[allow(dead_code)]
pub(crate) fn quantize_dc_i32_scalar(input: &[f32], scale: f32, output: &mut [i32]) {
    debug_assert_eq!(input.len(), output.len());
    for (&value, target) in input.iter().zip(output) {
        *target = round_i32(value * scale);
    }
}

#[allow(dead_code)]
pub(crate) fn quantize_dc_cfl_i32_scalar(
    input: &[f32],
    y_quant: &[i32],
    scale: f32,
    cfl: f32,
    output: &mut [i32],
) {
    debug_assert_eq!(input.len(), y_quant.len());
    debug_assert_eq!(input.len(), output.len());
    for ((&value, &yq), target) in input.iter().zip(y_quant).zip(output) {
        *target = round_i32(fmla(value, scale, -(yq as f32) * cfl));
    }
}

#[allow(dead_code)]
pub(crate) fn quantize_dc_scalar(input: &[f32], scale: f32, output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len());
    for (&value, target) in input.iter().zip(output) {
        *target = (value * scale).round() as i16;
    }
}

#[allow(dead_code)]
pub(crate) fn quantize_dc_cfl_scalar(
    input: &[f32],
    y_quant: &[i16],
    scale: f32,
    cfl: f32,
    output: &mut [i16],
) {
    debug_assert_eq!(input.len(), y_quant.len());
    debug_assert_eq!(input.len(), output.len());
    for ((&value, &yq), target) in input.iter().zip(y_quant).zip(output) {
        *target = fmla(value, scale, -(yq as f32) * cfl).round() as i16;
    }
}

static QUANTIZE_DC_METHODS: OnceLock<QuantizeDcMethods> = OnceLock::new();

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
fn select_quantize_dc_methods() -> QuantizeDcMethods {
    QuantizeDcMethods {
        quantize: crate::wasm::quantize_dc_wasm,
        quantize_cfl: crate::wasm::quantize_dc_cfl_wasm,
        quantize_i32: crate::wasm::quantize_dc_i32_wasm,
        quantize_cfl_i32: crate::wasm::quantize_dc_cfl_i32_wasm,
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn select_quantize_dc_methods() -> QuantizeDcMethods {
    QuantizeDcMethods {
        quantize: |input, scale, output| unsafe {
            crate::neon::quantize_dc_neon(input, scale, output)
        },
        quantize_cfl: |input, y_quant, scale, cfl, output| unsafe {
            crate::neon::quantize_dc_cfl_neon(input, y_quant, scale, cfl, output)
        },
        quantize_i32: |input, scale, output| unsafe {
            crate::neon::quantize_dc_i32_neon(input, scale, output)
        },
        quantize_cfl_i32: |input, y_quant, scale, cfl, output| unsafe {
            crate::neon::quantize_dc_cfl_i32_neon(input, y_quant, scale, cfl, output)
        },
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
fn select_quantize_dc_methods() -> QuantizeDcMethods {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return QuantizeDcMethods {
            quantize: |input, scale, output| unsafe {
                crate::avx::quantize_dc_avx2(input, scale, output)
            },
            quantize_cfl: |input, y_quant, scale, cfl, output| unsafe {
                crate::avx::quantize_dc_cfl_avx2(input, y_quant, scale, cfl, output)
            },
            quantize_i32: |input, scale, output| unsafe {
                crate::avx::quantize_dc_i32_avx2(input, scale, output)
            },
            quantize_cfl_i32: |input, y_quant, scale, cfl, output| unsafe {
                crate::avx::quantize_dc_cfl_i32_avx2(input, y_quant, scale, cfl, output)
            },
        };
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return QuantizeDcMethods {
            quantize: |input, scale, output| unsafe {
                crate::sse::quantize_dc_sse41(input, scale, output)
            },
            quantize_cfl: |input, y_quant, scale, cfl, output| unsafe {
                crate::sse::quantize_dc_cfl_sse41(input, y_quant, scale, cfl, output)
            },
            quantize_i32: |input, scale, output| unsafe {
                crate::sse::quantize_dc_i32_sse41(input, scale, output)
            },
            quantize_cfl_i32: |input, y_quant, scale, cfl, output| unsafe {
                crate::sse::quantize_dc_cfl_i32_sse41(input, y_quant, scale, cfl, output)
            },
        };
    }

    QuantizeDcMethods {
        quantize: quantize_dc_scalar,
        quantize_cfl: quantize_dc_cfl_scalar,
        quantize_i32: quantize_dc_i32_scalar,
        quantize_cfl_i32: quantize_dc_cfl_i32_scalar,
    }
}

pub(crate) fn selected_quantize_dc_methods() -> QuantizeDcMethods {
    *QUANTIZE_DC_METHODS.get_or_init(select_quantize_dc_methods)
}

static QUANTIZE_BLOCK_AC_METHOD: OnceLock<QuantizeBlockAcFn> = OnceLock::new();

#[inline]
pub(crate) fn quantize_ac_thresholds(
    c: usize,
    xsize: usize,
    ysize: usize,
    distance: f32,
) -> [f32; 4] {
    quantize_ac_thresholds_scaled(c, xsize, ysize, distance, 1.0)
}

const X_THRESHOLD_EXTRA: f32 = 0.08;
const X_FIRST_THRESHOLD_WINDOW: f32 = 0.5;
static X_DEADZONE_WINDOW: [f32; 4] = [0.5, 0.75, 1.5, 2.0];

#[inline]
fn x_deadzone_window(distance: f32) -> f32 {
    let [a, b, c, d] = X_DEADZONE_WINDOW;
    let up = ((distance - a) / (b - a)).clamp(0.0, 1.0);
    let down = ((d - distance) / (d - c)).clamp(0.0, 1.0);
    up.min(down)
}

/// Quantization zero thresholds for the effective per-channel precision.
/// Frames which already spend bits on finer B precision preserve near-one
/// high-frequency B levels instead of applying the ordinary 0.75 deadzone.
/// This only changes the constants loaded by the existing scalar/SIMD loop.
#[inline]
pub(crate) fn quantize_ac_thresholds_scaled(
    c: usize,
    xsize: usize,
    ysize: usize,
    distance: f32,
    qm_multiplier: f32,
) -> [f32; 4] {
    let mut normal = [0.58f32, 0.635, 0.66, 0.7];
    if c == 0 {
        let w = x_deadzone_window(distance);
        normal[0] += w * (X_FIRST_THRESHOLD_WINDOW - normal[0]);
        for t in &mut normal[1..] {
            *t += (1.0 - w) * X_THRESHOLD_EXTRA;
        }
    }
    if c == 2 {
        let b_hf_threshold = if qm_multiplier > 1.0 { 0.65 } else { 0.75 };
        for t in &mut normal[1..] {
            *t = b_hf_threshold;
        }
    }
    if xsize > 1 || ysize > 1 {
        // The area clamp was never calibrated past 32px
        let delta =
            (0.003_f32 * xsize as f32 * ysize as f32).clamp(0.0, if c > 0 { 0.08 } else { 0.12 });
        for t in &mut normal {
            *t -= delta;
        }
    }
    let high_quality = match c {
        1 => [0.50, 0.51, 0.52, 0.54],
        0 | 2 => normal,
        _ => unreachable!("invalid channel {c}"),
    };
    let t = ((distance - 0.1) / 0.9).clamp(0.0, 1.0);
    std::array::from_fn(|i| high_quality[i] + t * (normal[i] - high_quality[i]))
}

#[inline]
pub(crate) fn quantize_ac_q_scaled(quant: i32, scale: f32, qm_multiplier: f32) -> f32 {
    scale * quant as f32 * qm_multiplier
}

fn select_quantize_block_ac_fn() -> QuantizeBlockAcFn {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::quantize_block_ac_wasm
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |block_in, c, qm, quant, scale, qm_multiplier, distance, xsize, ysize, block_out| unsafe {
            crate::neon::quantize_block_ac_neon(
                block_in,
                c,
                qm,
                quant,
                scale,
                qm_multiplier,
                distance,
                xsize,
                ysize,
                block_out,
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") {
            return |block_in,
                    c,
                    qm,
                    quant,
                    scale,
                    qm_multiplier,
                    distance,
                    xsize,
                    ysize,
                    block_out| unsafe {
                crate::avx::quantize_block_ac_avx2(
                    block_in,
                    c,
                    qm,
                    quant,
                    scale,
                    qm_multiplier,
                    distance,
                    xsize,
                    ysize,
                    block_out,
                );
            };
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            return |block_in,
                    c,
                    qm,
                    quant,
                    scale,
                    qm_multiplier,
                    distance,
                    xsize,
                    ysize,
                    block_out| unsafe {
                crate::sse::quantize_block_ac_sse41(
                    block_in,
                    c,
                    qm,
                    quant,
                    scale,
                    qm_multiplier,
                    distance,
                    xsize,
                    ysize,
                    block_out,
                );
            };
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        quantize_block_ac_scalar
    }
}

#[inline]
pub(crate) fn selected_quantize_block_ac_fn() -> QuantizeBlockAcFn {
    *QUANTIZE_BLOCK_AC_METHOD.get_or_init(select_quantize_block_ac_fn)
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_block_ac_scalar(
    block_in: &[f32],
    c: usize,
    qm: &[f32],
    quant: i32,
    scale: f32,
    qm_multiplier: f32,
    distance: f32,
    xsize: usize,
    ysize: usize,
    block_out: &mut [i32],
) {
    let thr = quantize_ac_thresholds_scaled(c, xsize, ysize, distance, qm_multiplier);
    let q_scaled = quantize_ac_q_scaled(quant, scale, qm_multiplier);
    let width = xsize * 8;
    let height = ysize * 8;
    // The quant matrix, input, and output must all be sized for this transform.
    // A mismatch here means the caller selected the wrong matrix for the
    // strategy (e.g. handing the 64-entry DCT8 matrix to a DCT16X8/DCT16X16
    // block). Fail with a clear message rather than an opaque index panic.
    debug_assert_eq!(
        qm.len(),
        width * height,
        "quant matrix size {} != transform size {}x{}={} (wrong matrix for strategy?)",
        qm.len(),
        width,
        height,
        width * height
    );
    debug_assert!(block_in.len() >= width * height, "block_in too small");
    debug_assert!(block_out.len() >= width * height, "block_out too small");
    let n = width * height;
    let qm = &qm[..n];
    let block_in = &block_in[..n];
    let block_out = &mut block_out[..n];
    let half = width / 2;
    for (y, ((qm_row, in_row), out_row)) in qm
        .chunks_exact(width)
        .zip(block_in.chunks_exact(width))
        .zip(block_out.chunks_exact_mut(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];
        for (x, ((&qmv, &inv), out)) in qm_row
            .iter()
            .zip(in_row.iter())
            .zip(out_row.iter_mut())
            .enumerate()
        {
            let threshold = if x >= half { thr_hi } else { thr_lo };
            let q = qmv * q_scaled;
            let val = q * inv;
            *out = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
        }
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

/// libjxl `kDefaultQuantBias` (X, Y, B, numerator): what an `all_default`
/// CustomTransformData signals.
pub(crate) static DEFAULT_QUANT_BIASES: [f32; 4] = [
    1.0 - 0.05465007330715401,
    1.0 - 0.07005449891748593,
    1.0 - 0.049935103337343655,
    0.145,
];

/// The explicit CustomTransformData biases written with a non-default opsin
/// matrix (as f16): X/Y keep the defaults, B's ±1 reconstructs a full step.
pub(crate) static EXPLICIT_QUANT_BIASES: [f32; 4] =
    [1.0 - 0.054_650_075, 1.0 - 0.070_054_5, 1.0, 0.145];

/// The decoder's dequantization of one channel's integer AC levels
/// (libjxl `AdjustQuantBias`): `±one` at |q| = 1, `q - numerator / q` above.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DequantBias {
    pub(crate) one: f32,
    pub(crate) numerator: f32,
}

impl DequantBias {
    pub(crate) const DEFAULT: [DequantBias; 3] = [
        DequantBias::new(DEFAULT_QUANT_BIASES[0]),
        DequantBias::new(DEFAULT_QUANT_BIASES[1]),
        DequantBias::new(DEFAULT_QUANT_BIASES[2]),
    ];

    const fn new(one: f32) -> Self {
        Self {
            one,
            numerator: DEFAULT_QUANT_BIASES[3],
        }
    }

    /// Per-channel biases the decoder applies: the defaults, or the explicit
    /// bundle after its f16 round trip.
    pub(crate) fn channels(explicit: bool) -> [DequantBias; 3] {
        if !explicit {
            return Self::DEFAULT;
        }
        let f16 = |v: f32| crate::util::f16_bits_to_f32(crate::util::f32_to_f16_bits(v));
        let numerator = f16(EXPLICIT_QUANT_BIASES[3]);
        std::array::from_fn(|c| DequantBias {
            one: f16(EXPLICIT_QUANT_BIASES[c]),
            numerator,
        })
    }

    /// [`Self::level`] for a float-valued (but integer) level, as produced by
    /// the encoder-side quantizers before conversion to `i32`.
    #[inline]
    pub(crate) fn level_f32(self, quant: f32) -> f32 {
        let aq = quant.abs();
        if aq < 1.125 {
            if quant == 0.0 {
                0.0
            } else {
                self.one.copysign(quant)
            }
        } else {
            quant - self.numerator / quant
        }
    }

    #[inline]
    pub(crate) fn level(self, quant: i32) -> f32 {
        let aq = quant.unsigned_abs() as f32;
        if aq < 1.125 {
            if quant == 0 {
                0.0
            } else if quant > 0 {
                self.one
            } else {
                -self.one
            }
        } else {
            let q = quant as f32;
            q - self.numerator / q
        }
    }
}

/// Y-channel quantize then dequantize-with-bias for CfL roundtrip. `inout`
/// holds size=xsize*ysize*64 floats. `quantized` holds size integers.
fn quantize_roundtrip_y_block(
    ctx: &EncodingContext,
    qm: &[f32],
    dqm: &[f32],
    scale: f32,
    quant: i32,
    distance: f32,
    xsize: usize,
    ysize: usize,
    inout: &mut [f32],
    quantized: &mut [i32],
) {
    (ctx.quantize_block_ac)(
        inout, 1, qm, quant, scale, 1.0, distance, xsize, ysize, quantized,
    );
    let inv_qac = 1.0 / (scale * quant as f32);
    let bias = ctx.dequant_bias(1);
    let size = xsize * ysize * 64;
    for (out, (&q, &dq)) in inout[..size]
        .iter_mut()
        .zip(quantized[..size].iter().zip(dqm[..size].iter()))
    {
        *out = bias.level(q) * dq * inv_qac;
    }
}

/// Luma reconstruction capture for the closed-loop quant field: the
/// group-local plane, the pixel row offset of the current stripe in it, and
/// the pass-0 token index at which each coded block starts.
pub(crate) type ReconCapture<'a> = (
    &'a mut crate::image::Image3F,
    usize,
    &'a mut Vec<(u16, u16, u32)>,
);

/// Per-block memo of one AC group for the closed-loop quant field: a block's
/// quantized coefficients, DC levels and reconstruction are a function of its
/// field value and of the nonzero-count contexts predicted from its
/// neighbors, so a re-encode after a one-tile change recomputes only the
/// blocks whose key changed and tokenizes the rest from the memo.
pub(crate) struct AcBlockCache {
    xsize_blocks: usize,
    /// Per channel and first block: the key its stored channel was computed
    /// under. Y: field value and Y nonzero context; X/B: field value, own and
    /// Y contexts (CfL reads the dequantized Y) and the channel's qm scale.
    keys: [Vec<u32>; 3],
    quantized: [Vec<i32>; 3],
    dc: Image3S,
    /// Unquantized DC per cell and channel (the chroma saturation test reads it).
    dc_f: [Vec<f32>; 3],
    recon: Image3F,
    /// Token streams per block and channel (single-pass frames): one arena per
    /// output stream, `spans[(block * 3 + c) * streams + s]` = (start, len),
    /// and the shifted nonzero count stored in the context map per channel.
    tokens: Vec<Vec<Token>>,
    spans: Vec<(u32, u32)>,
    nz_shift: Vec<[u8; 3]>,
    /// Evaluation counter and, per block cell, the evaluation that last
    /// recomputed it: an evaluation synced at `s` needs the cells with
    /// `cell_gen > s`.
    pub(crate) epoch: u32,
    cell_gen: Vec<u32>,
    /// Frozen-price bits of each first block's first stream, per channel.
    bits: Vec<[f32; 3]>,
    /// Trial evaluation: a cached channel adds its bits to `trial_bits`
    /// instead of copying its tokens (the group's token streams are partial).
    pub(crate) trial: bool,
    pub(crate) trial_bits: f64,
}

impl AcBlockCache {
    pub(crate) fn new(
        xsize_blocks: usize,
        ysize_blocks: usize,
        xsize: usize,
        ysize: usize,
    ) -> Self {
        let n = xsize_blocks * ysize_blocks;
        Self {
            xsize_blocks,
            keys: std::array::from_fn(|_| vec![u32::MAX; n]),
            quantized: std::array::from_fn(|_| vec![0; n * 64]),
            dc: Image3S::new(xsize_blocks, ysize_blocks),
            dc_f: std::array::from_fn(|_| vec![0.0; n]),
            recon: Image3F::new(xsize, ysize),
            tokens: Vec::new(),
            spans: Vec::new(),
            nz_shift: vec![[0; 3]; n],
            epoch: 0,
            cell_gen: vec![0; n],
            bits: vec![[f32::NAN; 3]; n],
            trial: false,
            trial_bits: 0.0,
        }
    }

    #[cfg(feature = "extraslow")]
    /// Forget the blocks of a group-local block region (its transform layout
    /// changed, which the keys do not carry).
    pub(crate) fn invalidate(&mut self, x: usize, y: usize, w: usize, h: usize) {
        for row in y..y + h {
            for key in &mut self.keys {
                key[row * self.xsize_blocks + x..row * self.xsize_blocks + x + w].fill(u32::MAX);
            }
        }
    }

    #[cfg(feature = "extraslow")]
    /// Reclaim superseded token versions between trials, when no snapshot
    /// holds arena offsets. Keep every referenced span, including cells hidden
    /// by a larger transform: restoring a layout can make those cells live again.
    pub(crate) fn compact_tokens(&mut self) {
        let streams = self.tokens.len();
        for (s, arena) in self.tokens.iter_mut().enumerate() {
            let live: usize = self.spans[s..]
                .iter()
                .step_by(streams)
                .map(|&(_, len)| len as usize)
                .sum();
            // Amortize the copy over at least one whole live set of discarded
            // tokens. Tiny streams do not justify repeated allocations.
            if arena.len() <= live.saturating_mul(2).max(4096) {
                continue;
            }
            let mut compact = Vec::with_capacity(live);
            for (start, len) in self.spans[s..].iter_mut().step_by(streams) {
                let old_start = *start as usize;
                *start = compact.len() as u32;
                compact.extend_from_slice(&arena[old_start..old_start + *len as usize]);
            }
            *arena = compact;
        }
    }

    #[cfg(feature = "extraslow")]
    /// Everything the memo holds for the cells of a group-local block region,
    /// so a trial layout there can be undone without re-encoding. Between the
    /// snapshot and its restore/discard, arenas must stay append-only.
    pub(crate) fn snapshot(&self, x: usize, y: usize, w: usize, h: usize) -> MemoSnapshot {
        let mut snap = MemoSnapshot {
            rect: (x, y, w, h),
            ..Default::default()
        };
        let streams = if self.keys[0].is_empty() {
            0
        } else {
            self.spans.len() / (self.keys[0].len() * 3)
        };
        let (pw, ph) = (self.recon.xsize(), self.recon.ysize());
        for row in y..y + h {
            let cells = row * self.xsize_blocks + x..row * self.xsize_blocks + x + w;
            for c in 0..3 {
                snap.keys.extend_from_slice(&self.keys[c][cells.clone()]);
                snap.quantized
                    .extend_from_slice(&self.quantized[c][cells.start * 64..cells.end * 64]);
                snap.dc
                    .extend_from_slice(&self.dc.plane_row(c, row)[x..x + w]);
                snap.dc_f.extend_from_slice(&self.dc_f[c][cells.clone()]);
            }
            if streams > 0 {
                snap.spans.extend_from_slice(
                    &self.spans[cells.start * 3 * streams..cells.end * 3 * streams],
                );
            }
            snap.nz_shift
                .extend_from_slice(&self.nz_shift[cells.clone()]);
            snap.bits.extend_from_slice(&self.bits[cells.clone()]);
            snap.cell_gen.extend_from_slice(&self.cell_gen[cells]);
        }
        for c in 0..3 {
            for py in (y * 8).min(ph)..((y + h) * 8).min(ph) {
                snap.recon.extend_from_slice(
                    &self.recon.plane_row(c, py)[(x * 8).min(pw)..((x + w) * 8).min(pw)],
                );
            }
        }
        snap.streams = streams;
        snap
    }

    #[cfg(feature = "extraslow")]
    /// Undo everything since `snapshot` for its region (the spans layout must
    /// not have changed in between).
    pub(crate) fn restore(&mut self, snap: &MemoSnapshot) {
        let (x, y, w, h) = snap.rect;
        let streams = if self.keys[0].is_empty() {
            0
        } else {
            self.spans.len() / (self.keys[0].len() * 3)
        };
        let (pw, ph) = (self.recon.xsize(), self.recon.ysize());
        let (mut k, mut q, mut sp, mut n) = (0, 0, 0, 0);
        for row in y..y + h {
            let cells = row * self.xsize_blocks + x..row * self.xsize_blocks + x + w;
            for c in 0..3 {
                self.keys[c][cells.clone()].copy_from_slice(&snap.keys[k..k + w]);
                self.quantized[c][cells.start * 64..cells.end * 64]
                    .copy_from_slice(&snap.quantized[q..q + w * 64]);
                self.dc.plane_row_mut(c, row)[x..x + w].copy_from_slice(&snap.dc[k..k + w]);
                self.dc_f[c][cells.clone()].copy_from_slice(&snap.dc_f[k..k + w]);
                k += w;
                q += w * 64;
            }
            if streams > 0 && streams == snap.streams {
                let len = w * 3 * streams;
                self.spans[cells.start * 3 * streams..cells.end * 3 * streams]
                    .copy_from_slice(&snap.spans[sp..sp + len]);
                sp += len;
            } else if streams > 0 {
                // The span table was (re)allocated after the snapshot.
                for key in &mut self.keys {
                    key[cells.clone()].fill(u32::MAX);
                }
            }
            self.nz_shift[cells.clone()].copy_from_slice(&snap.nz_shift[n..n + w]);
            self.bits[cells.clone()].copy_from_slice(&snap.bits[n..n + w]);
            // Pixels went back to the snapshot: mark them recomputed so a
            // later sync compares (and skips equal) cells.
            self.cell_gen[cells].fill(self.epoch);
            n += w;
        }
        let mut r = 0;
        for c in 0..3 {
            for py in (y * 8).min(ph)..((y + h) * 8).min(ph) {
                let row =
                    &mut self.recon.plane_row_mut(c, py)[(x * 8).min(pw)..((x + w) * 8).min(pw)];
                let len = row.len();
                row.copy_from_slice(&snap.recon[r..r + len]);
                r += len;
            }
        }
    }

    /// The memo's reconstruction of the group; the capture target of every
    /// evaluation that uses the memo.
    pub(crate) fn recon_mut(&mut self) -> &mut Image3F {
        &mut self.recon
    }

    #[cfg(feature = "extraslow")]
    /// Copy into `dst` (offset `x0`, `y0`) the cells recomputed after
    /// evaluation `synced` whose pixels differ from it; returns their pixel
    /// bounding box in `dst` coordinates. Cells recomputed to the values
    /// `dst` already holds (a rejected move re-encoded back) are skipped, so
    /// the box covers the change itself and not every cell touched since.
    pub(crate) fn sync_into(
        &self,
        synced: u32,
        dst: &mut Image3F,
        x0: usize,
        y0: usize,
    ) -> Option<(usize, usize, usize, usize)> {
        let (mut rx0, mut ry0, mut rx1, mut ry1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let (w, h) = (self.recon.xsize(), self.recon.ysize());
        for (i, &g) in self.cell_gen.iter().enumerate() {
            if g <= synced {
                continue;
            }
            let (bx, by) = (i % self.xsize_blocks, i / self.xsize_blocks);
            let (cx0, cy0) = (bx * 8, by * 8);
            let (cx1, cy1) = ((cx0 + 8).min(w), (cy0 + 8).min(h));
            if cx0 >= cx1 || cy0 >= cy1 {
                continue;
            }
            let differs = (0..3).any(|c| {
                (cy0..cy1).any(|y| {
                    self.recon.plane_row(c, y)[cx0..cx1]
                        != dst.plane_row(c, y0 + y)[x0 + cx0..x0 + cx1]
                })
            });
            if !differs {
                continue;
            }
            for c in 0..3 {
                for y in cy0..cy1 {
                    dst.plane_row_mut(c, y0 + y)[x0 + cx0..x0 + cx1]
                        .copy_from_slice(&self.recon.plane_row(c, y)[cx0..cx1]);
                }
            }
            rx0 = rx0.min(x0 + cx0);
            ry0 = ry0.min(y0 + cy0);
            rx1 = rx1.max(x0 + cx1);
            ry1 = ry1.max(y0 + cy1);
        }
        (rx1 > 0).then_some((rx0, ry0, rx1, ry1))
    }
}

#[cfg(feature = "extraslow")]
/// See [`AcBlockCache::snapshot`].
#[derive(Default)]
pub(crate) struct MemoSnapshot {
    rect: (usize, usize, usize, usize),
    streams: usize,
    keys: Vec<u32>,
    quantized: Vec<i32>,
    dc: Vec<i16>,
    dc_f: Vec<f32>,
    spans: Vec<(u32, u32)>,
    nz_shift: Vec<[u8; 3]>,
    bits: Vec<[f32; 3]>,
    cell_gen: Vec<u32>,
    recon: Vec<f32>,
}

struct RdoqScratch {
    choices: Box<[u8; RDOQ_MAX_CHOICES]>,
    costs: HeapMatrix<f32, 2, RDOQ_MAX_STRIDE>,
}

impl Default for RdoqScratch {
    fn default() -> Self {
        Self {
            choices: heap_array(u8::MAX),
            costs: HeapMatrix::new(f32::INFINITY),
        }
    }
}

struct ReconScratch {
    err: Box<[f32; MAX_BLOCK_COEFFICIENTS]>,
    pix: Box<[f32; MAX_BLOCK_COEFFICIENTS]>,
}

impl Default for ReconScratch {
    fn default() -> Self {
        Self {
            err: heap_array(0.0),
            pix: heap_array(0.0),
        }
    }
}

#[cfg(feature = "extraslow")]
pub(crate) type PolisherSlot = Option<Box<crate::coef_polish::CoefPolisher>>;
#[cfg(not(feature = "extraslow"))]
pub(crate) type PolisherSlot = Option<Box<()>>;

pub(crate) struct AcGroupScratch {
    /// Study: decoder-feedback coefficient polish, set by the closed loop's
    /// serial polish pass.
    pub(crate) polisher: PolisherSlot,
    coeffs: HeapMatrix<f32, 3, MAX_BLOCK_COEFFICIENTS>,
    quantized: HeapMatrix<i32, 3, MAX_BLOCK_COEFFICIENTS>,
    /// Saved luma coefficients for RDOQ or reconstruction capture.
    source_y: Option<Box<[f32; MAX_BLOCK_COEFFICIENTS]>>,
    block: Box<[i32; MAX_BLOCK_COEFFICIENTS]>,
    rdoq: Option<RdoqScratch>,
    /// Edge-replicated pixels of a footprint that crosses the image edge.
    gather: Box<[f32; MAX_BLOCK_COEFFICIENTS]>,
    /// Distinct source colors, reused across RDOQ blocks and groups.
    few_color_keys: Vec<u64>,
    /// Y coefficient error and its pixel-domain image for the recon capture.
    recon: Option<ReconScratch>,
    clip: Option<Box<crate::clip_continuation::ClipScratch>>,
}

impl AcGroupScratch {
    /// The coefficient planes and edge gather buffer, for forward transforms
    /// taken outside the writer.
    pub(crate) fn transform_buffers(
        &mut self,
    ) -> (
        &mut [[f32; MAX_BLOCK_COEFFICIENTS]; 3],
        &mut [f32; MAX_BLOCK_COEFFICIENTS],
    ) {
        (&mut self.coeffs, &mut self.gather)
    }
}

impl Default for AcGroupScratch {
    fn default() -> Self {
        Self {
            polisher: None,
            coeffs: HeapMatrix::new(0.0),
            quantized: HeapMatrix::new(0),
            source_y: None,
            block: heap_array(0),
            rdoq: None,
            gather: heap_array(0.0),
            few_color_keys: Vec::new(),
            clip: None,
            recon: None,
        }
    }
}

/// Unrounded Y and B DC of an AC group.
#[derive(Clone)]
pub(crate) struct SourceDc {
    pub(crate) y: crate::image::Plane<f32>,
    pub(crate) b: crate::image::Plane<f32>,
    pub(crate) x: crate::image::Plane<f32>,
}

/// The part of quantized coefficient `c` that pass `pass` transmits. Each
/// pass sends the remainder rounded toward zero (libjxl's split), so a small
/// coefficient stays zero in the earlier passes instead of landing in every
/// pass with opposite signs; the decoder sums `sent << shift` over passes.
#[inline]
fn pass_coefficient(c: i32, coeff_shifts: &[u32], pass: usize) -> i32 {
    let mut remaining = c;
    let mut sent = 0;
    for &shift in &coeff_shifts[..=pass] {
        sent = remaining.signum() * (remaining.abs() >> shift);
        remaining -= sent << shift;
    }
    sent
}

/// Process and tokenize one stripe of an AC group, pushing tokens into `out`.
/// Callers buffer tokens across all AC groups, build an adaptive entropy code
/// from the aggregate distribution, then emit them in `encode_frame`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ac_group(
    ctx: &EncodingContext,
    scratch: &mut AcGroupScratch,
    opsin: &Image3F,
    opsin_origin: (usize, usize),
    group_brect: Rect,
    scale: f32,
    scale_dc: f32,
    dc_step: [f32; 3],
    distance: f32,
    x_qm_scale: u32,
    dc_data: &DcGroupData,
    ytob_dc: i32,
    quant_dc: &mut Image3S,
    mut source_dc: Option<&mut SourceDc>,
    qorigin_x: usize,
    qorigin_y: usize,
    num_nzeros: &mut [Image3B],
    coeff_shifts: &[u32],
    rdoq_prices: Option<&FrozenTokenPrices>,
    coeff_orders: &crate::coeff_order::CoeffOrders,
    alternative_orders: &[crate::coeff_order::CoeffOrders],
    mut order_stats: Option<&mut crate::coeff_order::OrderStats>,
    measure_chroma_distortion: bool,
    qf_threshold: u32,
    mut recon_y: Option<ReconCapture<'_>>,
    mut memo: Option<&mut AcBlockCache>,
    out: &mut [Vec<Token>],
) -> f32 {
    let (out, alternative_out) = out.split_at_mut(coeff_shifts.len());
    debug_assert_eq!(alternative_out.len(), alternative_orders.len());
    debug_assert!(alternative_out.is_empty() || coeff_shifts.len() == 1);
    let matrices = ctx.matrices();
    let xsize_blocks = group_brect.xsize;
    let ysize_blocks = group_brect.ysize;

    let inv_factor = [
        INV_DC_QUANT[0] / dc_step[0] * scale_dc,
        INV_DC_QUANT[1] / dc_step[1] * scale_dc,
        INV_DC_QUANT[2] / dc_step[2] * scale_dc,
    ];
    // `base_correlation_b` (= 1) plus the frame's signaled `ytob_dc / 84`,
    // converted from dequantized XYB into stored-B-DC units. Folded into the
    // quantizer below so the slope costs no extra rounding error.
    let cfl_factor_b = crate::color_correlation::dc_cfl_factor(dc_step, ytob_dc, ctx.cfl_frame());
    let x_qm_mul = 1.25f32.powf(x_qm_scale as f32 - 2.0);

    let nzeros_by0 = group_brect.y0 % K_GROUP_DIM_IN_BLOCKS;
    let mut chroma_distortion = 0.0f32;
    let cfl_factor_x = crate::color_correlation::dc_cfl_factor_x(dc_step, ctx.cfl_base_x());

    // All the big per-block buffers live in the worker scratch: re-creating them
    // per group cost ~130 KB of zeroing, and `pblock` was being zeroed once per
    // block per pass. Every one is written over `..size` before it is read, so
    // carrying values across groups is not observable.
    let AcGroupScratch {
        polisher,
        coeffs,
        quantized,
        source_y,
        block: pblock,
        rdoq,
        gather,
        few_color_keys,
        recon: recon_scratch,
        clip: clip_scratch,
    } = scratch;
    #[cfg(not(feature = "extraslow"))]
    let _ = polisher;

    // Reuse this small vector across channels and blocks. Only memo misses
    // need the offsets; ordinary encodes never read them.
    let mut stream_starts = Vec::new();
    for by in 0..ysize_blocks {
        let nz_by = nzeros_by0 + by;
        let global_by = group_brect.y0 + by;

        for bx in 0..xsize_blocks {
            let global_bx = group_brect.x0 + bx;

            // Skip non-first blocks of multi-block transforms.
            if !dc_data.ac_strategy.is_first_block(global_bx, global_by) {
                continue;
            }
            if let Some((_, _, starts)) = recon_y.as_mut() {
                starts.push((global_bx as u16, global_by as u16, out[0].len() as u32));
            }

            let raw_strategy = dc_data.ac_strategy.raw_strategy(global_bx, global_by);
            let cov_x = AcStrategyImage::covered_blocks_x_of(raw_strategy);
            let cov_y = AcStrategyImage::covered_blocks_y_of(raw_strategy);
            // libjxl-tiny normalizes: cx >= cy. For DCT16X8 (1×2) and DCT8X16
            // (2×1), both end up as cx=2, cy=1, matching the 8×16 storage.
            let (cx, cy) = if cov_y > cov_x {
                (cov_y, cov_x)
            } else {
                (cov_x, cov_y)
            };
            let size = cx * cy * 64;
            let quant_ac = dc_data.raw_quant_field.row(global_by)[global_bx] as i32;

            // ---- Block memo ----
            // Per channel: a block's channel is a function of the field value,
            // the nonzero contexts predicted from its neighbors, Y's context
            // for the CfL residuals and the channel's qm scale.
            let lby = global_by - qorigin_y;
            let predicted_ctx: [u8; 3] = std::array::from_fn(|c| {
                let nzero_map = &num_nzeros[0];
                let row_top = (nz_by != 0).then(|| nzero_map.plane_row(c, nz_by - 1));
                predict_from_top_and_left(row_top, nzero_map.plane_row(c, nz_by), bx, 32)
            });
            let memo_keys: [u32; 3] = [
                quant_ac as u32
                    | u32::from(predicted_ctx[0]) << 8
                    | u32::from(predicted_ctx[1]) << 16
                    | x_qm_scale << 24,
                quant_ac as u32 | u32::from(predicted_ctx[1]) << 8,
                quant_ac as u32
                    | u32::from(predicted_ctx[2]) << 8
                    | u32::from(predicted_ctx[1]) << 16
                    | ctx.b_qm_scale() << 24,
            ];
            let bi = lby * memo.as_deref().map_or(0, |m| m.xsize_blocks) + bx;
            // Token streams are memoized for single-pass frames (the key
            // carries the first pass's contexts) and never while tallying
            // orders. The capture image is the memo's own reconstruction, so
            // a hit copies no pixels.
            let streams = out.len() + alternative_out.len();
            let token_memo = coeff_shifts.len() == 1 && order_stats.is_none();
            let tokens_cached = token_memo
                && memo
                    .as_deref()
                    .is_some_and(|m| m.spans.len() == m.keys[0].len() * streams * 3);
            let hit: [bool; 3] = std::array::from_fn(|c| {
                memo.as_deref()
                    .is_some_and(|m| m.keys[c][bi] == memo_keys[c])
            });
            debug_assert!(memo.is_none() || !measure_chroma_distortion);
            // A hit channel's levels (unless its tokens are cached too) and DC.
            if let Some(m) = memo.as_deref() {
                for c in 0..3 {
                    if !hit[c] {
                        continue;
                    }
                    for j in 0..cov_y {
                        if !tokens_cached {
                            for i in 0..cov_x {
                                let slot = ((lby + j) * m.xsize_blocks + bx + i) * 64;
                                let k = (j * cov_x + i) * 64;
                                quantized[c][k..k + 64]
                                    .copy_from_slice(&m.quantized[c][slot..slot + 64]);
                            }
                        }
                        let src = &m.dc.plane_row(c, lby + j)[bx..bx + cov_x];
                        quant_dc.plane_row_mut(c, lby + j)[bx..bx + cov_x].copy_from_slice(src);
                    }
                }
            }
            let memo_hit = hit.iter().all(|&h| h);
            if !memo_hit {
                // ---- Forward DCT for all 3 channels ----
                let footprint = Footprint::of_strategy(
                    opsin,
                    opsin_origin.0 + bx * 8,
                    opsin_origin.1 + by * 8,
                    raw_strategy,
                );
                debug_assert_eq!((footprint.width, footprint.height), (cov_x * 8, cov_y * 8));
                // Padding is never shown, so RDOQ charges coefficient error by
                // the footprint's visible area.
                let visible = footprint.visible_fraction();
                let restore_penalty = if rdoq_prices.is_some() {
                    let x0 = opsin_origin.0 + bx * 8;
                    let y0 = opsin_origin.1 + by * 8;
                    let x1 = (x0 + cov_x * 8).min(opsin.xsize());
                    let y1 = (y0 + cov_y * 8).min(opsin.ysize());
                    let limit = RESTORE_FEW_COLORS_PER_BLOCK * cov_x * cov_y;
                    if few_colors(opsin, x0, y0, x1, y1, limit, few_color_keys) {
                        RESTORE_FEW_COLORS_PENALTY
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                let (inv_qm_y, qm_y): (&[f32], &[f32]) = match raw_strategy {
                STRATEGY_DCT => (&matrices.inv_matrix(1)[..], &matrices.matrix(1)[..]),
                STRATEGY_IDENTITY => (
                    &matrices.inv_matrix_identity(1)[..],
                    &matrices.matrix_identity(1)[..],
                ),
                STRATEGY_DCT2X2 => (
                    &matrices.inv_matrix_dct2x2(1)[..],
                    &matrices.matrix_dct2x2(1)[..],
                ),
                STRATEGY_DCT4X4 => (&matrices.inv_matrix_4x4(1)[..], &matrices.matrix_4x4(1)[..]),
                STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => {
                    (&matrices.inv_matrix_4x8(1)[..], &matrices.matrix_4x8(1)[..])
                }
                STRATEGY_AFV0..=STRATEGY_AFV3 => {
                    (&matrices.inv_matrix_afv(1)[..], &matrices.matrix_afv(1)[..])
                }
                STRATEGY_DCT16X16 => (&matrices.inv_matrix_16x16(1)[..], &matrices.matrix_16x16(1)[..]),
                STRATEGY_DCT32X32 => (&matrices.inv_matrix_32x32(1)[..], &matrices.matrix_32x32(1)[..]),
                STRATEGY_DCT64X64 => (&matrices.inv_matrix_64x64(1)[..], &matrices.matrix_64x64(1)[..]),
                #[cfg(feature = "extraslow")]
                STRATEGY_DCT128X128 | STRATEGY_DCT128X64 | STRATEGY_DCT64X128 => (
                    matrices.inv_matrix_128(raw_strategy, 1),
                    matrices.matrix_128(raw_strategy, 1),
                ),
                STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => {
                    (&matrices.inv_matrix_64x32(1)[..], &matrices.matrix_64x32(1)[..])
                }
                STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => {
                    (&matrices.inv_matrix_32x16(1)[..], &matrices.matrix_32x16(1)[..])
                }
                STRATEGY_DCT32X8 | STRATEGY_DCT8X32 => {
                    (&matrices.inv_matrix_8x32(1)[..], &matrices.matrix_8x32(1)[..])
                }
                _ /* 16X8/8X16 */ => (&matrices.inv_matrix_16x8(1)[..], &matrices.matrix_16x8(1)[..]),
            };
                // Saturated pixels get a free target beyond the clip limit;
                // the shift rides on Y and B, so either one recomputing needs it.
                let clip_shift = ctx.speed.effort().clip_continuation && (!hit[1] || !hit[2]) && {
                    let (input, stride) = footprint.pixels(opsin.plane(1), &mut gather[..]);
                    crate::clip_continuation::continue_saturated(
                        ctx,
                        raw_strategy,
                        cx,
                        cy,
                        inv_qm_y,
                        qm_y,
                        quant_ac,
                        scale,
                        distance,
                        input,
                        stride,
                        clip_scratch.get_or_insert_with(Default::default),
                    )
                };
                for c in 0..3 {
                    if hit[c] {
                        continue;
                    }
                    let (input, stride) = footprint.pixels(opsin.plane(c), &mut gather[..]);
                    let (input, stride) = if clip_shift && c != 0 {
                        let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(raw_strategy);
                        (
                            crate::clip_continuation::shifted_input(
                                clip_scratch.as_mut().expect("shift computed"),
                                input,
                                stride,
                                pw,
                                ph,
                            ),
                            pw,
                        )
                    } else {
                        (input, stride)
                    };
                    match raw_strategy {
                        STRATEGY_DCT => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.dct8x8)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_IDENTITY => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.identity8x8)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT2X2 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.dct2x2_8x8)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT16X8 => {
                            let dst: &mut [f32; 128] = coeffs[c].first_chunk_mut::<128>().unwrap();
                            (ctx.dct16x8)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT8X16 => {
                            let dst: &mut [f32; 128] = coeffs[c].first_chunk_mut::<128>().unwrap();
                            (ctx.dct8x16)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT16X16 => {
                            let dst: &mut [f32; 256] = coeffs[c].first_chunk_mut::<256>().unwrap();
                            (ctx.dct16x16)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT32X32 => {
                            let dst: &mut [f32; 1024] =
                                coeffs[c].first_chunk_mut::<1024>().unwrap();
                            (ctx.dct32x32)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT64X64 => {
                            let dst: &mut [f32; 4096] =
                                coeffs[c].first_chunk_mut::<4096>().unwrap();
                            (ctx.dct64x64)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT64X32 => {
                            let dst: &mut [f32; 2048] =
                                coeffs[c].first_chunk_mut::<2048>().unwrap();
                            (ctx.dct64x32)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT32X64 => {
                            let dst: &mut [f32; 2048] =
                                coeffs[c].first_chunk_mut::<2048>().unwrap();
                            (ctx.dct32x64)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT4X4 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.dct4x4)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT4X8 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.dct4x8)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT8X4 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.dct8x4)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT32X16 => {
                            let dst: &mut [f32; 512] = coeffs[c].first_chunk_mut::<512>().unwrap();
                            (ctx.dct32x16)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT16X32 => {
                            let dst: &mut [f32; 512] = coeffs[c].first_chunk_mut::<512>().unwrap();
                            (ctx.dct16x32)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT32X8 => {
                            let dst: &mut [f32; 256] = coeffs[c].first_chunk_mut::<256>().unwrap();
                            (ctx.dct32x8)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_DCT8X32 => {
                            let dst: &mut [f32; 256] = coeffs[c].first_chunk_mut::<256>().unwrap();
                            (ctx.dct8x32)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_AFV0 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.afv0)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_AFV1 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.afv1)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_AFV2 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.afv2)(DctInput::new(input, stride), dst);
                        }
                        STRATEGY_AFV3 => {
                            let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                            (ctx.afv3)(DctInput::new(input, stride), dst);
                        }
                        #[cfg(feature = "extraslow")]
                        STRATEGY_DCT128X128 => {
                            let dst = coeffs[c].first_chunk_mut::<16384>().unwrap();
                            (ctx.dct128x128)(DctInput::new(input, stride), dst);
                        }
                        #[cfg(feature = "extraslow")]
                        STRATEGY_DCT128X64 => {
                            let dst = coeffs[c].first_chunk_mut::<8192>().unwrap();
                            (ctx.dct128x64)(DctInput::new(input, stride), dst);
                        }
                        #[cfg(feature = "extraslow")]
                        STRATEGY_DCT64X128 => {
                            let dst = coeffs[c].first_chunk_mut::<8192>().unwrap();
                            (ctx.dct64x128)(DctInput::new(input, stride), dst);
                        }
                        _ => unreachable!("invalid raw strategy {}", raw_strategy),
                    }
                }

                // For DCT8, DC = coeffs[0]. For multi-block, use DCFromLowestFrequencies.
                // dc_vals[c] holds up to 4 DC values (DCT16X16 = 2×2 covered blocks);
                // indexing is didx = iy * cov_x + ix.
                let mut dc_vals = [[0.0f32; 256]; 3];
                for c in 0..3 {
                    dc_from_coefficients(ctx, raw_strategy, &coeffs[c], &mut dc_vals[c]);
                }

                // Hit channels: the DC values the skipped transform would have given.
                if let Some(m) = memo.as_deref() {
                    for c in 0..3 {
                        if hit[c] {
                            for iy in 0..cov_y {
                                let row = (lby + iy) * m.xsize_blocks;
                                dc_vals[c][iy * cov_x..(iy + 1) * cov_x]
                                    .copy_from_slice(&m.dc_f[c][row + bx..row + bx + cov_x]);
                            }
                        }
                    }
                }
                // DC for storage (per covered block, using pre-swap cov_x/cov_y).
                let covered_dc = cov_x * cov_y;
                let mut y_dc_q = [0i16; 256];
                if let Some(m) = memo.as_deref().filter(|_| hit[1]) {
                    for iy in 0..cov_y {
                        y_dc_q[iy * cov_x..(iy + 1) * cov_x]
                            .copy_from_slice(&m.dc.plane_row(1, lby + iy)[bx..bx + cov_x]);
                    }
                } else {
                    (ctx.quantize_dc)(
                        &dc_vals[1][..covered_dc],
                        inv_factor[1],
                        &mut y_dc_q[..covered_dc],
                    );
                }
                if let Some(source) = source_dc.as_mut() {
                    for iy in 0..cov_y {
                        let bx = global_bx - qorigin_x;
                        source.y.row_mut(global_by - qorigin_y + iy)[bx..bx + cov_x]
                            .copy_from_slice(&dc_vals[1][iy * cov_x..(iy + 1) * cov_x]);
                    }
                }
                for iy in 0..cov_y {
                    let lbx = global_bx - qorigin_x;
                    let quant_target = &mut quant_dc.plane_row_mut(1, global_by - qorigin_y + iy)
                        [lbx..lbx + cov_x];
                    let row_start = iy * cov_x;
                    quant_target.copy_from_slice(&y_dc_q[row_start..row_start + cov_x]);
                }
                // Quantize Y AC with roundtrip (modifies coeffs[1] to dequantized).
                // Matrix selection: DCT8 uses 8×8 weights, DCT16X8/8X16 share the
                // 128-float 16×8 weights, DCT16X16 uses the 256-float 16×16 weights.
                if let Some(m) = memo.as_deref().filter(|_| hit[1]) {
                    // The levels are skipped above when the tokens are cached too.
                    for j in 0..cov_y {
                        for i in 0..cov_x {
                            let slot = ((lby + j) * m.xsize_blocks + bx + i) * 64;
                            let k = (j * cov_x + i) * 64;
                            quantized[1][k..k + 64]
                                .copy_from_slice(&m.quantized[1][slot..slot + 64]);
                        }
                    }
                    let inv_qac = 1.0 / (scale * quant_ac as f32);
                    for (out, (&q, &dq)) in coeffs[1][..size]
                        .iter_mut()
                        .zip(quantized[1][..size].iter().zip(qm_y[..size].iter()))
                    {
                        *out = ctx.dequant_bias(1).level(q) * dq * inv_qac;
                    }
                } else {
                    if rdoq_prices.is_some() || recon_y.is_some() {
                        let source_y = source_y.get_or_insert_with(|| heap_array(0.0));
                        source_y[..size].copy_from_slice(&coeffs[1][..size]);
                    }
                    quantize_roundtrip_y_block(
                        ctx,
                        inv_qm_y,
                        qm_y,
                        scale,
                        quant_ac,
                        distance,
                        cx,
                        cy,
                        &mut coeffs[1][..size],
                        &mut quantized[1][..size],
                    );
                    if let Some(prices) = rdoq_prices {
                        let RdoqScratch {
                            choices: rdoq_choices,
                            costs: rdoq_costs,
                        } = rdoq.get_or_insert_with(RdoqScratch::default);
                        let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
                        let nzero_map = &num_nzeros[0];
                        let row_top = (nz_by != 0).then(|| nzero_map.plane_row(1, nz_by - 1));
                        let predicted = predict_from_top_and_left(
                            row_top,
                            nzero_map.plane_row(1, nz_by),
                            bx,
                            32,
                        );
                        rdoq_block(
                            rdoq_channel_weights(ctx.coding)[1],
                            prices,
                            coeff_orders.scan_for(strategy_code, 1),
                            &source_y.as_ref().expect("luma saved for RDOQ")[..size],
                            inv_qm_y,
                            quantize_ac_q_scaled(quant_ac, scale, 1.0),
                            &mut quantized[1][..size],
                            raw_strategy,
                            strategy_code,
                            1,
                            ctx.dequant_bias(1),
                            predicted,
                            cx,
                            cy,
                            distance,
                            quant_ac as u32 > qf_threshold,
                            rdoq_lambda(ctx.arbiter, quant_ac as f32 / qf_threshold.max(1) as f32)
                                / visible,
                            restore_penalty,
                            rdoq_choices,
                            rdoq_costs,
                        );
                        let inv_qac = 1.0 / (scale * quant_ac as f32);
                        for (out, (&q, &dq)) in coeffs[1][..size]
                            .iter_mut()
                            .zip(quantized[1][..size].iter().zip(qm_y[..size].iter()))
                        {
                            *out = ctx.dequant_bias(1).level(q) * dq * inv_qac;
                        }
                    }
                }
                // ---- Luma reconstruction capture (closed-loop quant field) ----
                // Source plus the inverse-transformed Y quantization error, the
                // rerank's reconstruction model; the cx*cy lowest frequencies ride
                // on the DC path and are excluded.
                if !hit[1]
                    && let Some((recon, y_off, _)) = recon_y.as_mut()
                {
                    let ReconScratch {
                        err: recon_err,
                        pix: recon_pix,
                    } = recon_scratch.get_or_insert_with(ReconScratch::default);
                    let source_y = source_y.as_ref().expect("luma saved for reconstruction");
                    let y_off = *y_off;
                    for ((e, &q), &src) in recon_err[..size]
                        .iter_mut()
                        .zip(&coeffs[1][..size])
                        .zip(&source_y[..size])
                    {
                        *e = q - src;
                    }
                    let wc = cx * 8;
                    for row in recon_err[..cy * wc].chunks_exact_mut(wc) {
                        row[..cx].fill(0.0);
                    }
                    crate::inflated_cost::reconstruct_error(
                        ctx.idct,
                        raw_strategy,
                        &recon_err[..size],
                        &mut recon_pix[..size],
                    );
                    let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(raw_strategy);
                    let (px0, py0) = (opsin_origin.0 + bx * 8, opsin_origin.1 + by * 8);
                    let plane = opsin.plane(1);
                    let recon = recon.plane_mut(1);
                    for y in 0..ph {
                        let (sy, ly) = (py0 + y, y_off + by * 8 + y);
                        if sy >= plane.ysize() || ly >= recon.ysize() {
                            break;
                        }
                        let src_row = plane.row(sy);
                        let dst_row = recon.row_mut(ly);
                        for x in 0..pw {
                            let (sx, lx) = (px0 + x, bx * 8 + x);
                            if sx >= src_row.len() || lx >= dst_row.len() {
                                break;
                            }
                            dst_row[lx] = src_row[sx] + recon_pix[y * pw + x];
                        }
                    }
                }

                // ---- Per-tile CfL factors ----
                let tx = global_bx / 8;
                let ty = global_by / 8;
                let cmap_x = dc_data.ytox_map.row(ty)[tx];
                let cmap_b = dc_data.ytob_map.row(ty)[tx];
                // y_to_x = 0 + cmap_x / 84;  y_to_b = 1 + cmap_b / 84.
                let x_factor = crate::color_correlation::y_to_x_ratio(ctx.cfl_frame(), cmap_x);
                let b_factor = crate::color_correlation::y_to_b_ratio(ctx.cfl_frame(), cmap_b);

                // ---- Apply CfL: X -= x_factor·Y, B -= b_factor·Y on every coefficient ----
                // The decoder reverses CfL in coefficient space (DequantLane) using the
                // dequantized Y AC coefficients, whose cx*cy LLF positions are zero at
                // that point — they are filled from the DC plane (LowestFrequenciesFromDC)
                // only afterwards. So the encoder must subtract using a Y whose LLF
                // positions are likewise zero; otherwise the AC-quantized LLF energy of Y
                // gets folded into the B/X DC (via the DCFromLowestFrequencies extraction
                // below) with no decoder-side counterpart, corrupting chroma by up to a
                // full-scale per-block shift (worst on B, where b_factor ≈ 1).
                {
                    let wc = cx * 8;
                    for row in coeffs[1][..cy * wc].chunks_exact_mut(wc) {
                        row[..cx].fill(0.0);
                    }
                }
                {
                    let [c0, c1, c2] = &mut **coeffs;
                    (ctx.apply_cfl)(
                        &mut c0[..size],
                        &c1[..size],
                        &mut c2[..size],
                        [x_factor, 0.0, b_factor],
                    );
                }
                // ---- Extract post-CfL X and B DC ----
                let mut x_dc_post = [0.0f32; 256];
                let mut b_dc_post = [0.0f32; 256];
                match raw_strategy {
                    STRATEGY_DCT
                    | STRATEGY_IDENTITY
                    | STRATEGY_DCT2X2
                    | STRATEGY_DCT4X4
                    | STRATEGY_DCT4X8
                    | STRATEGY_DCT8X4
                    | STRATEGY_AFV0..=STRATEGY_AFV3 => {
                        x_dc_post[0] = coeffs[0][0];
                        b_dc_post[0] = coeffs[2][0];
                    }
                    STRATEGY_DCT16X8 => {
                        let xb: &[f32; 128] = coeffs[0].first_chunk::<128>().unwrap();
                        let bb: &[f32; 128] = coeffs[2].first_chunk::<128>().unwrap();
                        dc_from_dct16x8(xb, x_dc_post.first_chunk_mut::<2>().unwrap());
                        dc_from_dct16x8(bb, b_dc_post.first_chunk_mut::<2>().unwrap());
                    }
                    STRATEGY_DCT8X16 => {
                        let xb: &[f32; 128] = coeffs[0].first_chunk::<128>().unwrap();
                        let bb: &[f32; 128] = coeffs[2].first_chunk::<128>().unwrap();
                        dc_from_dct8x16(xb, x_dc_post.first_chunk_mut::<2>().unwrap());
                        dc_from_dct8x16(bb, b_dc_post.first_chunk_mut::<2>().unwrap());
                    }
                    STRATEGY_DCT16X16 => {
                        let xb: &[f32; 256] = coeffs[0].first_chunk::<256>().unwrap();
                        let bb: &[f32; 256] = coeffs[2].first_chunk::<256>().unwrap();
                        dc_from_dct16x16(xb, x_dc_post.first_chunk_mut::<4>().unwrap());
                        dc_from_dct16x16(bb, b_dc_post.first_chunk_mut::<4>().unwrap());
                    }
                    STRATEGY_DCT32X32 => {
                        let xb: &[f32; 1024] = coeffs[0].first_chunk::<1024>().unwrap();
                        let bb: &[f32; 1024] = coeffs[2].first_chunk::<1024>().unwrap();
                        (ctx.dc_from_dct32x32)(xb, x_dc_post.first_chunk_mut::<16>().unwrap());
                        (ctx.dc_from_dct32x32)(bb, b_dc_post.first_chunk_mut::<16>().unwrap());
                    }
                    STRATEGY_DCT64X64 => {
                        let xb: &[f32; 4096] = coeffs[0].first_chunk::<4096>().unwrap();
                        let bb: &[f32; 4096] = coeffs[2].first_chunk::<4096>().unwrap();
                        (ctx.dc_from_dct64x64)(xb, x_dc_post.first_chunk_mut::<64>().unwrap());
                        (ctx.dc_from_dct64x64)(bb, b_dc_post.first_chunk_mut::<64>().unwrap());
                    }
                    STRATEGY_DCT64X32 => {
                        let xb: &[f32; 2048] = coeffs[0].first_chunk::<2048>().unwrap();
                        let bb: &[f32; 2048] = coeffs[2].first_chunk::<2048>().unwrap();
                        (ctx.dc_from_dct64x32)(xb, x_dc_post.first_chunk_mut::<32>().unwrap());
                        (ctx.dc_from_dct64x32)(bb, b_dc_post.first_chunk_mut::<32>().unwrap());
                    }
                    STRATEGY_DCT32X64 => {
                        let xb: &[f32; 2048] = coeffs[0].first_chunk::<2048>().unwrap();
                        let bb: &[f32; 2048] = coeffs[2].first_chunk::<2048>().unwrap();
                        (ctx.dc_from_dct32x64)(xb, x_dc_post.first_chunk_mut::<32>().unwrap());
                        (ctx.dc_from_dct32x64)(bb, b_dc_post.first_chunk_mut::<32>().unwrap());
                    }
                    STRATEGY_DCT32X16 => {
                        let xb: &[f32; 512] = coeffs[0].first_chunk::<512>().unwrap();
                        let bb: &[f32; 512] = coeffs[2].first_chunk::<512>().unwrap();
                        (ctx.dc_from_dct32x16)(xb, x_dc_post.first_chunk_mut::<8>().unwrap());
                        (ctx.dc_from_dct32x16)(bb, b_dc_post.first_chunk_mut::<8>().unwrap());
                    }
                    STRATEGY_DCT16X32 => {
                        let xb: &[f32; 512] = coeffs[0].first_chunk::<512>().unwrap();
                        let bb: &[f32; 512] = coeffs[2].first_chunk::<512>().unwrap();
                        (ctx.dc_from_dct16x32)(xb, x_dc_post.first_chunk_mut::<8>().unwrap());
                        (ctx.dc_from_dct16x32)(bb, b_dc_post.first_chunk_mut::<8>().unwrap());
                    }
                    STRATEGY_DCT32X8 | STRATEGY_DCT8X32 => {
                        let xb: &[f32; 256] = coeffs[0].first_chunk::<256>().unwrap();
                        let bb: &[f32; 256] = coeffs[2].first_chunk::<256>().unwrap();
                        (ctx.dc_from_dct8x32)(xb, x_dc_post.first_chunk_mut::<4>().unwrap());
                        (ctx.dc_from_dct8x32)(bb, b_dc_post.first_chunk_mut::<4>().unwrap());
                    }
                    #[cfg(feature = "extraslow")]
                    STRATEGY_DCT128X128 => {
                        (ctx.dc_from_dct128x128)(
                            coeffs[0].first_chunk::<16384>().unwrap(),
                            x_dc_post.first_chunk_mut::<256>().unwrap(),
                        );
                        (ctx.dc_from_dct128x128)(
                            coeffs[2].first_chunk::<16384>().unwrap(),
                            b_dc_post.first_chunk_mut::<256>().unwrap(),
                        );
                    }
                    #[cfg(feature = "extraslow")]
                    STRATEGY_DCT128X64 => {
                        (ctx.dc_from_dct128x64)(
                            coeffs[0].first_chunk::<8192>().unwrap(),
                            x_dc_post.first_chunk_mut::<128>().unwrap(),
                        );
                        (ctx.dc_from_dct128x64)(
                            coeffs[2].first_chunk::<8192>().unwrap(),
                            b_dc_post.first_chunk_mut::<128>().unwrap(),
                        );
                    }
                    #[cfg(feature = "extraslow")]
                    STRATEGY_DCT64X128 => {
                        (ctx.dc_from_dct64x128)(
                            coeffs[0].first_chunk::<8192>().unwrap(),
                            x_dc_post.first_chunk_mut::<128>().unwrap(),
                        );
                        (ctx.dc_from_dct64x128)(
                            coeffs[2].first_chunk::<8192>().unwrap(),
                            b_dc_post.first_chunk_mut::<128>().unwrap(),
                        );
                    }
                    _ => unreachable!(),
                }

                // ---- X channel: write post-CfL DC, quantize AC ----
                let inv_qm_x: &[f32] = match raw_strategy {
                    STRATEGY_DCT => &matrices.inv_matrix(0)[..],
                    STRATEGY_IDENTITY => &matrices.inv_matrix_identity(0)[..],
                    STRATEGY_DCT2X2 => &matrices.inv_matrix_dct2x2(0)[..],
                    STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(0)[..],
                    STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(0)[..],
                    STRATEGY_AFV0..=STRATEGY_AFV3 => &matrices.inv_matrix_afv(0)[..],
                    STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(0)[..],
                    STRATEGY_DCT32X32 => &matrices.inv_matrix_32x32(0)[..],
                    STRATEGY_DCT64X64 => &matrices.inv_matrix_64x64(0)[..],
                    #[cfg(feature = "extraslow")]
                    STRATEGY_DCT128X128 | STRATEGY_DCT128X64 | STRATEGY_DCT64X128 => {
                        matrices.inv_matrix_128(raw_strategy, 0)
                    }
                    STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => &matrices.inv_matrix_64x32(0)[..],
                    STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => &matrices.inv_matrix_32x16(0)[..],
                    STRATEGY_DCT32X8 | STRATEGY_DCT8X32 => &matrices.inv_matrix_8x32(0)[..],
                    _ => &matrices.inv_matrix_16x8(0)[..],
                };
                let mut chroma_dc_q = [0i16; 256];
                if !hit[0] {
                    if cfl_factor_x != 0.0 {
                        (ctx.quantize_dc_cfl)(
                            &x_dc_post[..covered_dc],
                            &y_dc_q[..covered_dc],
                            inv_factor[0],
                            cfl_factor_x,
                            &mut chroma_dc_q[..covered_dc],
                        );
                    } else {
                        (ctx.quantize_dc)(
                            &x_dc_post[..covered_dc],
                            inv_factor[0],
                            &mut chroma_dc_q[..covered_dc],
                        );
                    }
                    for iy in 0..cov_y {
                        let lbx = global_bx - qorigin_x;
                        let quant_dc_row = &mut quant_dc
                            .plane_row_mut(0, global_by - qorigin_y + iy)
                            [lbx..lbx + cov_x];
                        let row_start = iy * cov_x;
                        quant_dc_row.copy_from_slice(&chroma_dc_q[row_start..row_start + cov_x]);
                    }
                    (ctx.quantize_block_ac)(
                        &coeffs[0][..size],
                        0,
                        inv_qm_x,
                        quant_ac,
                        scale,
                        x_qm_mul,
                        distance,
                        cx,
                        cy,
                        &mut quantized[0][..size],
                    );
                    // Chroma RDOQ (review §8): the trellis is channel-generic — the
                    // CfL residuals are the source, contexts/prices/orders are
                    // channel-specific, and unlike Y the input coefficients are not
                    // overwritten afterwards. Runs at every distance with the
                    // transform's calibrated weights and the distance-scheduled X/B
                    // multiplier (`chroma_rdoq_weight`).
                    if let Some(prices) = rdoq_prices {
                        let RdoqScratch {
                            choices: rdoq_choices,
                            costs: rdoq_costs,
                        } = rdoq.get_or_insert_with(RdoqScratch::default);
                        let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
                        let nzero_map = &num_nzeros[0];
                        let row_top = (nz_by != 0).then(|| nzero_map.plane_row(0, nz_by - 1));
                        let predicted = predict_from_top_and_left(
                            row_top,
                            nzero_map.plane_row(0, nz_by),
                            bx,
                            32,
                        );
                        rdoq_block(
                            rdoq_channel_weights(ctx.coding)[0],
                            prices,
                            coeff_orders.scan_for(strategy_code, 0),
                            &coeffs[0][..size],
                            inv_qm_x,
                            quantize_ac_q_scaled(quant_ac, scale, x_qm_mul),
                            &mut quantized[0][..size],
                            raw_strategy,
                            strategy_code,
                            0,
                            ctx.dequant_bias(0),
                            predicted,
                            cx,
                            cy,
                            distance,
                            quant_ac as u32 > qf_threshold,
                            rdoq_lambda(ctx.arbiter, quant_ac as f32 / qf_threshold.max(1) as f32)
                                / visible,
                            restore_penalty,
                            rdoq_choices,
                            rdoq_costs,
                        );
                    }
                }
                let row_stride = cx * 8;

                // Keep the fractional source for the DC predictor check. Y DC
                // and these B lowest frequencies are unchanged by AC reranking.
                if let Some(source) = source_dc.as_mut() {
                    for iy in 0..cov_y {
                        let bx = global_bx - qorigin_x;
                        source.b.row_mut(global_by - qorigin_y + iy)[bx..bx + cov_x]
                            .copy_from_slice(&b_dc_post[iy * cov_x..(iy + 1) * cov_x]);
                        source.x.row_mut(global_by - qorigin_y + iy)[bx..bx + cov_x]
                            .copy_from_slice(&x_dc_post[iy * cov_x..(iy + 1) * cov_x]);
                    }
                }
                // ---- B channel: write CfL'd DC, quantize AC ----
                let inv_qm_b: &[f32] = match raw_strategy {
                    STRATEGY_DCT => &matrices.inv_matrix(2)[..],
                    STRATEGY_IDENTITY => &matrices.inv_matrix_identity(2)[..],
                    STRATEGY_DCT2X2 => &matrices.inv_matrix_dct2x2(2)[..],
                    STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(2)[..],
                    STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(2)[..],
                    STRATEGY_AFV0..=STRATEGY_AFV3 => &matrices.inv_matrix_afv(2)[..],
                    STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(2)[..],
                    STRATEGY_DCT32X32 => &matrices.inv_matrix_32x32(2)[..],
                    STRATEGY_DCT64X64 => &matrices.inv_matrix_64x64(2)[..],
                    #[cfg(feature = "extraslow")]
                    STRATEGY_DCT128X128 | STRATEGY_DCT128X64 | STRATEGY_DCT64X128 => {
                        matrices.inv_matrix_128(raw_strategy, 2)
                    }
                    STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => &matrices.inv_matrix_64x32(2)[..],
                    STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => &matrices.inv_matrix_32x16(2)[..],
                    STRATEGY_DCT32X8 | STRATEGY_DCT8X32 => &matrices.inv_matrix_8x32(2)[..],
                    _ => &matrices.inv_matrix_16x8(2)[..],
                };
                if !hit[2] {
                    (ctx.quantize_dc_cfl)(
                        &b_dc_post[..covered_dc],
                        &y_dc_q[..covered_dc],
                        inv_factor[2],
                        cfl_factor_b,
                        &mut chroma_dc_q[..covered_dc],
                    );
                    for iy in 0..cov_y {
                        let lbx = global_bx - qorigin_x;
                        let quant_dc_row = &mut quant_dc
                            .plane_row_mut(2, global_by - qorigin_y + iy)
                            [lbx..lbx + cov_x];
                        let row_start = iy * cov_x;
                        quant_dc_row.copy_from_slice(&chroma_dc_q[row_start..row_start + cov_x]);
                    }
                    (ctx.quantize_block_ac)(
                        &coeffs[2][..size],
                        2,
                        inv_qm_b,
                        quant_ac,
                        scale,
                        ctx.b_qm_mul(),
                        distance,
                        cx,
                        cy,
                        &mut quantized[2][..size],
                    );
                    if let Some(prices) = rdoq_prices {
                        let RdoqScratch {
                            choices: rdoq_choices,
                            costs: rdoq_costs,
                        } = rdoq.get_or_insert_with(RdoqScratch::default);
                        let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
                        let nzero_map = &num_nzeros[0];
                        let row_top = (nz_by != 0).then(|| nzero_map.plane_row(2, nz_by - 1));
                        let predicted = predict_from_top_and_left(
                            row_top,
                            nzero_map.plane_row(2, nz_by),
                            bx,
                            32,
                        );
                        rdoq_block(
                            rdoq_channel_weights(ctx.coding)[2],
                            prices,
                            coeff_orders.scan_for(strategy_code, 2),
                            &coeffs[2][..size],
                            inv_qm_b,
                            quantize_ac_q_scaled(quant_ac, scale, ctx.b_qm_mul()),
                            &mut quantized[2][..size],
                            raw_strategy,
                            strategy_code,
                            2,
                            ctx.dequant_bias(2),
                            predicted,
                            cx,
                            cy,
                            distance,
                            quant_ac as u32 > qf_threshold,
                            rdoq_lambda(ctx.arbiter, quant_ac as f32 / qf_threshold.max(1) as f32)
                                / visible,
                            restore_penalty,
                            rdoq_choices,
                            rdoq_costs,
                        );
                    }
                }
                if measure_chroma_distortion && !hit[0] {
                    let q_scaled_x = quantize_ac_q_scaled(quant_ac, scale, x_qm_mul);
                    let bias_x = ctx.dequant_bias(0);
                    for i in 0..size {
                        let v = i / row_stride;
                        let u = i % row_stride;
                        if v < cy && u < cx {
                            continue;
                        }
                        let ideal = coeffs[0][i] * inv_qm_x[i] * q_scaled_x;
                        let error = ideal - bias_x.level(quantized[0][i]);
                        chroma_distortion += ctx.channel_weight(0) * error * error;
                    }
                }
                if measure_chroma_distortion && !hit[2] {
                    let q_scaled_b = quantize_ac_q_scaled(quant_ac, scale, ctx.b_qm_mul());
                    let bias_b = ctx.dequant_bias(2);
                    for i in 0..size {
                        let v = i / row_stride;
                        let u = i % row_stride;
                        if v < cy && u < cx {
                            continue;
                        }
                        let ideal = coeffs[2][i] * inv_qm_b[i] * q_scaled_b;
                        let error = ideal - bias_b.level(quantized[2][i]);
                        chroma_distortion =
                            fmla(ctx.channel_weight(2), error * error, chroma_distortion);
                    }
                }
                // ---- Chroma reconstruction capture (closed-loop quant field) ----
                // CfL was applied against the dequantized Y, so the decoder's chroma
                // error is the residual's own quantization error.
                if let Some((recon, y_off, _)) = recon_y.as_mut() {
                    let ReconScratch {
                        err: recon_err,
                        pix: recon_pix,
                    } = recon_scratch.get_or_insert_with(ReconScratch::default);
                    let y_off = *y_off;
                    let q_x = quantize_ac_q_scaled(quant_ac, scale, x_qm_mul);
                    let q_b = quantize_ac_q_scaled(quant_ac, scale, ctx.b_qm_mul());
                    for (c, inv_qm, q_scaled) in [(0usize, inv_qm_x, q_x), (2, inv_qm_b, q_b)] {
                        if hit[c] {
                            continue;
                        }
                        let bias = ctx.dequant_bias(c);
                        for i in 0..size {
                            let (v, u) = (i / row_stride, i % row_stride);
                            recon_err[i] = if v < cy && u < cx {
                                0.0
                            } else {
                                bias.level(quantized[c][i]) / (inv_qm[i] * q_scaled) - coeffs[c][i]
                            };
                        }
                        crate::inflated_cost::reconstruct_error(
                            ctx.idct,
                            raw_strategy,
                            &recon_err[..size],
                            &mut recon_pix[..size],
                        );
                        let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(raw_strategy);
                        let (px0, py0) = (opsin_origin.0 + bx * 8, opsin_origin.1 + by * 8);
                        let plane = opsin.plane(c);
                        let dst = recon.plane_mut(c);
                        for y in 0..ph {
                            let (sy, ly) = (py0 + y, y_off + by * 8 + y);
                            if sy >= plane.ysize() || ly >= dst.ysize() {
                                break;
                            }
                            let src_row = plane.row(sy);
                            let dst_row = dst.row_mut(ly);
                            for x in 0..pw {
                                let (sx, lx) = (px0 + x, bx * 8 + x);
                                if sx >= src_row.len() || lx >= dst_row.len() {
                                    break;
                                }
                                dst_row[lx] = src_row[sx] + recon_pix[y * pw + x];
                            }
                        }
                    }
                }
                if let Some(m) = memo.as_deref_mut() {
                    let epoch = m.epoch;
                    for j in 0..cov_y {
                        let row = (lby + j) * m.xsize_blocks;
                        m.cell_gen[row + bx..row + bx + cov_x].fill(epoch);
                    }
                    for c in 0..3 {
                        if hit[c] {
                            continue;
                        }
                        m.keys[c][bi] = memo_keys[c];
                        for j in 0..cov_y {
                            let row = (lby + j) * m.xsize_blocks;
                            m.dc_f[c][row + bx..row + bx + cov_x]
                                .copy_from_slice(&dc_vals[c][j * cov_x..(j + 1) * cov_x]);
                            for i in 0..cov_x {
                                let slot = (row + bx + i) * 64;
                                let k = (j * cov_x + i) * 64;
                                m.quantized[c][slot..slot + 64]
                                    .copy_from_slice(&quantized[c][k..k + 64]);
                            }
                            let src = &quant_dc.plane_row(c, lby + j)[bx..bx + cov_x];
                            m.dc.plane_row_mut(c, lby + j)[bx..bx + cov_x].copy_from_slice(src);
                        }
                    }
                }
                // ---- Study: decoder-feedback coefficient polish ----
                #[cfg(feature = "extraslow")]
                if let (Some(p), Some(prices), Some((plane, y_off, _))) = (
                    polisher.as_deref_mut().filter(|p| p.is_current()),
                    rdoq_prices,
                    recon_y.as_mut(),
                ) {
                    let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
                    let qf_hi = quant_ac as u32 > qf_threshold;
                    let weights = rdoq_channel_weights(ctx.coding);
                    let cw = chroma_rdoq_weight(distance);
                    macro_rules! channel {
                        ($c:expr, $source:expr, $inv_qm:expr, $q_scaled:expr) => {{
                            let c: usize = $c;
                            let block_ctx = fine_block_context(c, strategy_code, qf_hi);
                            crate::coef_polish::PolishChannel {
                                source: $source,
                                inv_qm: $inv_qm,
                                q_scaled: $q_scaled,
                                bias: ctx.dequant_bias(c),
                                weight: weights[c] * if c == 1 { 1.0 } else { cw },
                                scan: coeff_orders.scan_for(strategy_code, c),
                                histo_offset: fine_zero_density_contexts_offset(block_ctx),
                                nzero_ctx: fine_non_zero_context(
                                    predicted_ctx[c] as u32,
                                    block_ctx,
                                ),
                            }
                        }};
                    }
                    let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(raw_strategy);
                    let source_y = source_y.as_ref().expect("luma saved for polish");
                    let block = crate::coef_polish::PolishBlock {
                        raw_strategy,
                        cx,
                        cy,
                        size,
                        px0: opsin_origin.0 + bx * 8,
                        py0: opsin_origin.1 + by * 8,
                        pw,
                        ph,
                        lambda: rdoq_lambda(
                            ctx.arbiter,
                            quant_ac as f32 / qf_threshold.max(1) as f32,
                        ) / visible,
                        channels: [
                            channel!(
                                0,
                                &coeffs[0][..size],
                                inv_qm_x,
                                quantize_ac_q_scaled(quant_ac, scale, x_qm_mul)
                            ),
                            channel!(
                                1,
                                &source_y[..size],
                                inv_qm_y,
                                quantize_ac_q_scaled(quant_ac, scale, 1.0)
                            ),
                            channel!(
                                2,
                                &coeffs[2][..size],
                                inv_qm_b,
                                quantize_ac_q_scaled(quant_ac, scale, ctx.b_qm_mul())
                            ),
                        ],
                        cfl: [x_factor, 1.0, b_factor],
                    };
                    let [q0, q1, q2] = &mut **quantized;
                    p.polish(
                        ctx,
                        prices,
                        &block,
                        &mut [&mut q0[..size], &mut q1[..size], &mut q2[..size]],
                        (&mut **plane, bx * 8, *y_off + by * 8),
                    );
                }
            }
            // Hit channels skipped their transforms: the memo holds every
            // channel's unrounded DC (X and B equal post-CfL, Y's LLF being zero).
            if let (Some(source), Some(m)) = (source_dc.as_mut(), memo.as_deref()) {
                for iy in 0..cov_y {
                    let row = (lby + iy) * m.xsize_blocks;
                    for (c, plane) in [(0, &mut source.x), (1, &mut source.y), (2, &mut source.b)] {
                        plane.row_mut(lby + iy)[bx..bx + cov_x]
                            .copy_from_slice(&m.dc_f[c][row + bx..row + bx + cov_x]);
                    }
                }
            }
            // ---- Tokenize in order Y, X, B ----
            if let Some(m) = memo.as_deref_mut()
                && token_memo
                && m.spans.len() != m.keys[0].len() * streams * 3
            {
                m.spans = vec![(0, 0); m.keys[0].len() * streams * 3];
                m.tokens = vec![Vec::new(); streams];
            }
            let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
            let covered_blocks = cx * cy;
            // log2(covered_blocks): 0/1/2/4 for 1/2/4/16 covered blocks.
            let log2_covered_blocks = match covered_blocks {
                1 => 0,
                2 => 1,
                4 => 2,
                8 => 3,
                16 => 4,
                32 => 5,
                64 => 6,
                128 => 7,
                256 => 8,
                _ => unreachable!("invalid covered_blocks {}", covered_blocks),
            };
            for &c in &[1usize, 0, 2] {
                // A cached channel: copy its token streams and context cells.
                if tokens_cached
                    && hit[c]
                    && let Some(m) = memo.as_deref_mut()
                {
                    if m.trial && m.bits[bi][c].is_finite() {
                        m.trial_bits += f64::from(m.bits[bi][c]);
                        for iy in 0..cov_y {
                            num_nzeros[0].plane_row_mut(c, nz_by + iy)[bx..bx + cov_x]
                                .fill(m.nz_shift[bi][c]);
                        }
                        continue;
                    }
                    for s in 0..streams {
                        let (start, len) = m.spans[(bi * 3 + c) * streams + s];
                        let (start, len) = (start as usize, len as usize);
                        let dst = if s < out.len() {
                            &mut out[s]
                        } else {
                            &mut alternative_out[s - out.len()]
                        };
                        dst.extend_from_slice(&m.tokens[s][start..start + len]);
                    }
                    for iy in 0..cov_y {
                        num_nzeros[0].plane_row_mut(c, nz_by + iy)[bx..bx + cov_x]
                            .fill(m.nz_shift[bi][c]);
                    }
                    continue;
                }
                if token_memo && memo.is_some() {
                    stream_starts.clear();
                    stream_starts.extend(
                        out.iter()
                            .map(Vec::len)
                            .chain(alternative_out.iter().map(Vec::len)),
                    );
                }
                let full_block = &quantized[c][..size];

                for pass in 0..coeff_shifts.len() {
                    // Materialize the coefficients pass `pass` transmits. With
                    // decreasing per-pass shifts ending at 0, the decoder sums
                    // (sent_p << shift_p) over passes to recover `full_block`
                    // (jxl-vardct hf_coeff.rs:185,191).
                    let block = if pass == 0 && coeff_shifts[0] == 0 {
                        full_block
                    } else {
                        for k in 0..size {
                            pblock[k] = pass_coefficient(full_block[k], coeff_shifts, pass);
                        }
                        &pblock[..size]
                    };
                    let num_nzeros = &mut num_nzeros[pass];
                    let out = &mut out[pass];

                    let nzeros = if covered_blocks == 1 {
                        num_nonzero_except_dc(block.first_chunk::<64>().unwrap())
                    } else {
                        num_nonzero_except_llf(block, cx, cy)
                    };

                    // libjxl-tiny: NumNonZeroExceptLLF stores `(nzeros + covered_blocks - 1) >> log2_covered_blocks`
                    // to all covered cells in num_nzeros.
                    let shifted =
                        ((nzeros as usize + covered_blocks - 1) >> log2_covered_blocks) as u8;
                    if pass == 0
                        && let Some(m) = memo.as_deref_mut()
                    {
                        m.nz_shift[lby * m.xsize_blocks + bx][c] = shifted;
                    }
                    // Pre-swap iteration (cov_x, cov_y from raw strategy).
                    for iy in 0..cov_y {
                        let target_row =
                            &mut num_nzeros.plane_row_mut(c, nz_by + iy)[bx..bx + cov_x];
                        for target in target_row.iter_mut() {
                            *target = shifted;
                        }
                    }

                    // Predict from top and left.
                    let row_top: Option<&[u8]> = if nz_by == 0 {
                        None
                    } else {
                        Some(num_nzeros.plane_row(c, nz_by - 1))
                    };
                    let row = num_nzeros.plane_row(c, nz_by);
                    let predicted = predict_from_top_and_left(row_top, row, bx, 32);

                    let block_ctx =
                        fine_block_context(c, strategy_code, quant_ac as u32 > qf_threshold);
                    let nzero_ctx = fine_non_zero_context(predicted as u32, block_ctx);
                    let histo_offset = fine_zero_density_contexts_offset(block_ctx);

                    let token_start = out.len();
                    write_token_into(Token::new(nzero_ctx, nzeros as u32), out);

                    let mut prev: usize = if nzeros as usize > size / 16 { 0 } else { 1 };
                    let mut remaining = nzeros;
                    // Hoisted so the per-coefficient lookup stays one indexed
                    // load, as it was with the static natural-order tables.
                    let scan = coeff_orders.scan_for(strategy_code, c);
                    // First pass only: tally which raw positions actually carry
                    // nonzeros, so a shorter scan can be derived for pass two.
                    // Counted over the whole block, not just the walked prefix,
                    // so the tally does not depend on the current scan.
                    if pass == 0
                        && let Some(stats) = order_stats.as_deref_mut()
                        && let Some(slot) = crate::coeff_order::order_slot_of(strategy_code)
                    {
                        // `c` iterates [1, 0, 2] with the pass loop nested
                        // inside, so pin the block tally to one (channel, pass).
                        if c == 1 && pass == 0 {
                            stats.tally_block(slot);
                        }
                        for (raw, &coef) in block[..size].iter().enumerate() {
                            if coef != 0 {
                                stats.tally(slot, c, raw);
                            }
                        }
                    }
                    // Skip the first `covered_blocks` positions (LF).
                    let mut k = covered_blocks;
                    while k < size && remaining != 0 {
                        let raw = scan[k] as usize;
                        let coef = block[raw];
                        let ctx = histo_offset as usize
                            + if covered_blocks == 1 {
                                zero_density_context_8x8(remaining as usize, k, prev)
                            } else {
                                zero_density_context(
                                    remaining as usize,
                                    k,
                                    covered_blocks,
                                    log2_covered_blocks,
                                    prev,
                                )
                            };
                        write_token_into(Token::new(ctx as u32, pack_signed(coef)), out);
                        prev = if coef != 0 { 1 } else { 0 };
                        if coef != 0 {
                            remaining -= 1;
                        }
                        k += 1;
                    }
                    for (alternative_out, orders) in
                        alternative_out.iter_mut().zip(alternative_orders)
                    {
                        let alternative_scan = orders.scan_for(strategy_code, c);
                        if alternative_scan == scan {
                            alternative_out.extend_from_slice(&out[token_start..]);
                            continue;
                        }
                        write_token_into(Token::new(nzero_ctx, nzeros as u32), alternative_out);
                        let scan = alternative_scan;
                        let mut prev = if nzeros as usize > size / 16 { 0 } else { 1 };
                        let mut remaining = nzeros;
                        let mut k = covered_blocks;
                        while k < size && remaining != 0 {
                            let raw = scan[k] as usize;
                            let coef = block[raw];
                            let ctx = histo_offset as usize
                                + if covered_blocks == 1 {
                                    zero_density_context_8x8(remaining as usize, k, prev)
                                } else {
                                    zero_density_context(
                                        remaining as usize,
                                        k,
                                        covered_blocks,
                                        log2_covered_blocks,
                                        prev,
                                    )
                                };
                            write_token_into(
                                Token::new(ctx as u32, pack_signed(coef)),
                                alternative_out,
                            );
                            prev = if coef != 0 { 1 } else { 0 };
                            if coef != 0 {
                                remaining -= 1;
                            }
                            k += 1;
                        }
                    }
                    debug_assert_eq!(
                        remaining, 0,
                        "remaining nzeros at end: strategy={} c={} pass={}",
                        strategy_code, c, pass
                    );
                }
                if token_memo && let Some(m) = memo.as_deref_mut() {
                    for s in 0..streams {
                        let src = if s < out.len() {
                            &out[s]
                        } else {
                            &alternative_out[s - out.len()]
                        };
                        let new_tokens = &src[stream_starts[s]..];
                        if s == 0 {
                            m.bits[bi][c] = rdoq_prices.map_or(f32::NAN, |p| {
                                new_tokens.iter().map(|&k| p.token_bits(k)).sum()
                            });
                        }
                        let arena = &mut m.tokens[s];
                        m.spans[(bi * 3 + c) * streams + s] =
                            (arena.len() as u32, new_tokens.len() as u32);
                        arena.extend_from_slice(new_tokens);
                    }
                }
            }
        }
    }
    chroma_distortion
}

#[inline]
fn write_token_into(t: Token, out: &mut Vec<Token>) {
    out.push(t);
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "extraslow")]
    #[test]
    fn block_memo_compaction_preserves_all_streams_and_snapshot_rollback() {
        use super::AcBlockCache;
        use crate::entropy::Token;

        let mut memo = AcBlockCache::new(2, 2, 16, 16);
        memo.tokens = vec![Vec::new(); 3];
        memo.spans = vec![(0, 0); 4 * 3 * 3];
        let write = |memo: &mut AcBlockCache, cell: usize, version: u32| {
            for c in 0..3 {
                for s in 0..3 {
                    let arena = &mut memo.tokens[s];
                    let start = arena.len() as u32;
                    // Include empty spans and different lengths across streams.
                    let len = (cell + c + s) * 47;
                    arena.extend(
                        (0..len)
                            .map(|i| Token::new((cell * 3 + c) as u32, version * 1000 + i as u32)),
                    );
                    memo.spans[(cell * 3 + c) * 3 + s] = (start, len as u32);
                }
            }
        };
        let read = |memo: &AcBlockCache| -> Vec<Vec<(u32, u32)>> {
            memo.spans
                .iter()
                .enumerate()
                .map(|(i, &(start, len))| {
                    memo.tokens[i % 3][start as usize..(start + len) as usize]
                        .iter()
                        .map(|t| (t.context, t.value))
                        .collect()
                })
                .collect()
        };
        for version in 0..80 {
            write(&mut memo, version as usize % 4, version);
            let expected = read(&memo);
            memo.compact_tokens();
            assert_eq!(read(&memo), expected);
            for s in 0..3 {
                let live: usize = memo.spans[s..]
                    .iter()
                    .step_by(3)
                    .map(|&(_, len)| len as usize)
                    .sum();
                assert!(memo.tokens[s].len() <= (2 * live).max(4096));
            }
        }
        // Snapshot offsets must remain valid until restore, even though the
        // trial appends replacements both inside and outside the saved region.
        let before = read(&memo);
        let snapshot = memo.snapshot(1, 0, 1, 1);
        write(&mut memo, 1, 100);
        write(&mut memo, 2, 101);
        let changed = read(&memo);
        memo.restore(&snapshot);
        drop(snapshot);
        memo.compact_tokens();
        let restored = read(&memo);
        assert_eq!(&restored[9..18], &before[9..18]);
        assert_eq!(&restored[18..27], &changed[18..27]);

        // An entirely dead stream releases its old arena too.
        memo.spans.fill((0, 0));
        for arena in &mut memo.tokens {
            arena.resize(8192, Token::new(0, 0));
        }
        memo.compact_tokens();
        assert!(memo.tokens.iter().all(|arena| arena.capacity() == 0));
        AcBlockCache::new(0, 0, 0, 0).compact_tokens();
    }

    #[test]
    fn ac_scratch_allocates_optional_buffers_only_for_active_tools_and_reuses_them() {
        use super::{AcGroupScratch, write_ac_group};
        use crate::coeff_order::CoeffOrders;
        use crate::dc_group_data::DcGroupData;
        use crate::encoding_context::EncodingContext;
        use crate::entropy::{FrozenTokenPrices, Token, build_entropy_code_no_cluster};
        use crate::image::{Image3B, Image3F, Image3S, Rect};

        let (w, h) = (17, 19);
        let mut image = Image3F::new(w, h);
        for c in 0..3 {
            for (i, value) in image.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *value = ((i * 97 + c * 13) % 257) as f32 / 257.0;
            }
        }
        let mut dc = DcGroupData::new(w.div_ceil(8), h.div_ceil(8)).unwrap();
        dc.raw_quant_field.as_mut_slice().fill(8);
        let orders = CoeffOrders::natural();
        let samples: Vec<_> = (0..4096).map(|i| Token::new(i % 2, i % 17)).collect();
        let mut code = build_entropy_code_no_cluster(&samples, 2, &mut Vec::new());
        code.context_map = (0..crate::ac_context::K_NUM_FINE_AC_CONTEXTS)
            .map(|i| (i % 2) as u8)
            .collect();
        let prices = FrozenTokenPrices::new(&code);
        let run = |ctx: &EncodingContext,
                   scratch: &mut AcGroupScratch,
                   rdoq: bool,
                   capture: bool,
                   shifts: &[u32]| {
            let mut quant_dc = Image3S::new(w.div_ceil(8), h.div_ceil(8));
            let mut nzeros: Vec<_> = shifts.iter().map(|_| Image3B::new(32, 32)).collect();
            let mut tokens: Vec<_> = shifts.iter().map(|_| Vec::new()).collect();
            let mut recon = Image3F::new(w, h);
            let mut starts = Vec::new();
            write_ac_group(
                ctx,
                scratch,
                &image,
                (0, 0),
                Rect::new(0, 0, w.div_ceil(8), h.div_ceil(8)),
                1.0,
                1.0,
                [1.0; 3],
                2.0,
                2,
                &dc,
                0,
                &mut quant_dc,
                None,
                0,
                0,
                &mut nzeros,
                shifts,
                rdoq.then_some(&prices),
                &orders,
                &[],
                None,
                false,
                8,
                capture.then_some((&mut recon, 0, &mut starts)),
                None,
                &mut tokens,
            );
            (
                tokens
                    .into_iter()
                    .map(|stream| {
                        stream
                            .into_iter()
                            .map(|t| (t.context, t.value))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
                [0, 1, 2].map(|c| {
                    recon
                        .plane_data(c)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                }),
            )
        };
        let pointers = |scratch: &AcGroupScratch| {
            (
                scratch.source_y.as_ref().map(|v| v.as_ptr()),
                scratch
                    .rdoq
                    .as_ref()
                    .map(|v| (v.choices.as_ptr(), v.costs[0].as_ptr())),
                scratch
                    .recon
                    .as_ref()
                    .map(|v| (v.err.as_ptr(), v.pix.as_ptr())),
            )
        };
        for speed in [
            crate::Speed::UltraFast,
            crate::Speed::Fastest,
            crate::Speed::Fast,
            crate::Speed::Medium,
        ] {
            let ctx = EncodingContext::new(
                speed,
                crate::Arbiter::Ssimulacra2,
                crate::xyb::XybMatrix::SPEC,
                2.0,
                1,
            );
            let mut scratch = AcGroupScratch::default();
            for shifts in [&[0u32][..], &[2, 0]] {
                run(&ctx, &mut scratch, false, false, shifts);
                assert_eq!(pointers(&scratch), (None, None, None));
            }
        }
        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            2.0,
            1,
        );
        for (rdoq, capture) in [(true, false), (false, true), (true, true)] {
            let mut scratch = AcGroupScratch::default();
            for shifts in [&[0u32][..], &[2, 0]] {
                let expected = run(&ctx, &mut scratch, rdoq, capture, shifts);
                assert!(scratch.source_y.is_some());
                assert_eq!(scratch.rdoq.is_some(), rdoq);
                assert_eq!(scratch.recon.is_some(), capture);
                let storage = pointers(&scratch);
                assert_eq!(run(&ctx, &mut scratch, rdoq, capture, shifts), expected);
                assert_eq!(pointers(&scratch), storage);
                let mut fresh = AcGroupScratch::default();
                assert_eq!(run(&ctx, &mut fresh, rdoq, capture, shifts), expected);
            }
        }
    }

    /// A warm block memo must reproduce a fresh evaluation of the same
    /// candidate: tokens, quantized DC and the unrounded source DC, on full
    /// hits and when only B misses (a B matrix-scale change keeps Y cached).
    #[cfg(feature = "extraslow")]
    #[test]
    fn block_memo_matches_fresh_evaluation_on_partial_and_full_hits() {
        use super::{AcBlockCache, AcGroupScratch, SourceDc, write_ac_group};
        use crate::coeff_order::CoeffOrders;
        use crate::dc_group_data::DcGroupData;
        use crate::encoding_context::EncodingContext;
        use crate::entropy::{FrozenTokenPrices, Token, build_entropy_code_no_cluster};
        use crate::image::{Image3B, Image3F, Image3S, Plane, Rect};
        use crate::xyb::rgb_to_xyb_pixel_f32;

        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            2.0,
            1,
        );
        assert!(ctx.speed.effort().clip_continuation);
        // Black/white edges at a different column in every 8x8 block.
        let (w, h) = (56, 16);
        let black = rgb_to_xyb_pixel_f32(&ctx.xyb, 0.0, 0.0, 0.0);
        let white = rgb_to_xyb_pixel_f32(&ctx.xyb, 1.0, 1.0, 1.0);
        let mut image = Image3F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let edge = (x / 8 + y / 8) % 7 + 1;
                let (vx, vy, vb) = if x % 8 < edge { black } else { white };
                image.plane_row_mut(0, y)[x] = vx;
                image.plane_row_mut(1, y)[x] = vy;
                image.plane_row_mut(2, y)[x] = vb;
            }
        }
        let (bw, bh) = (w / 8, h / 8);
        let orders = CoeffOrders::natural();
        let samples: Vec<_> = (0..4096).map(|i| Token::new(i % 2, i % 17)).collect();
        let mut code = build_entropy_code_no_cluster(&samples, 2, &mut Vec::new());
        code.context_map = (0..crate::ac_context::K_NUM_FINE_AC_CONTEXTS)
            .map(|i| (i % 2) as u8)
            .collect();
        let prices = FrozenTokenPrices::new(&code);
        let run = |dc: &DcGroupData, b_scale: u32, memo: Option<&mut AcBlockCache>| {
            ctx.set_b_qm_scale(b_scale);
            let mut scratch = AcGroupScratch::default();
            let mut quant_dc = Image3S::new(bw, bh);
            let mut source = SourceDc {
                y: Plane::new(bw, bh),
                b: Plane::new(bw, bh),
                x: Plane::new(bw, bh),
            };
            let mut nzeros = vec![Image3B::new(32, 32)];
            let mut tokens = vec![Vec::new()];
            let mut starts = Vec::new();
            let mut memo = memo;
            let mut recon = match memo.as_deref_mut() {
                Some(m) => std::mem::replace(m.recon_mut(), Image3F::new(0, 0)),
                None => Image3F::new(w, h),
            };
            write_ac_group(
                &ctx,
                &mut scratch,
                &image,
                (0, 0),
                Rect::new(0, 0, bw, bh),
                1.0,
                1.0,
                [1.0; 3],
                2.0,
                2,
                dc,
                0,
                &mut quant_dc,
                Some(&mut source),
                0,
                0,
                &mut nzeros,
                &[0],
                Some(&prices),
                &orders,
                &[],
                None,
                false,
                8,
                Some((&mut recon, 0, &mut starts)),
                memo.as_deref_mut(),
                &mut tokens,
            );
            if let Some(m) = memo {
                *m.recon_mut() = recon;
            }
            let bits =
                |p: &Plane<f32>| p.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            (
                tokens[0]
                    .iter()
                    .map(|t| (t.context, t.value))
                    .collect::<Vec<_>>(),
                [0, 1, 2].map(|c| quant_dc.plane_data(c).to_vec()),
                [bits(&source.x), bits(&source.y), bits(&source.b)],
            )
        };
        for qf in [6u8, 10, 16] {
            let mut dc = DcGroupData::new(bw, bh).unwrap();
            dc.raw_quant_field.as_mut_slice().fill(qf);
            let mut memo = AcBlockCache::new(bw, bh, w, h);
            let warm = run(&dc, 6, Some(&mut memo));
            assert_eq!(warm, run(&dc, 6, None), "cold memo, qf {qf}");
            for b_scale in [7, 7] {
                let fresh = run(&dc, b_scale, None);
                assert_eq!(
                    run(&dc, b_scale, Some(&mut memo)),
                    fresh,
                    "qf {qf} b {b_scale}"
                );
            }
        }
    }

    #[test]
    fn pass_split_sums_back_and_rounds_toward_zero() {
        for shifts in [&[1, 0][..], &[2, 0], &[3, 0], &[2, 1, 0], &[3, 2, 1, 0]] {
            for c in -40..=40 {
                let sent: Vec<i32> = (0..shifts.len())
                    .map(|p| super::pass_coefficient(c, shifts, p))
                    .collect();
                let sum: i32 = sent.iter().zip(shifts).map(|(&v, &s)| v << s).sum();
                assert_eq!(sum, c, "{shifts:?} {c}");
                assert!(sent.iter().all(|&v| v == 0 || v.signum() == c.signum()));
                assert_eq!(sent[0], c.signum() * (c.abs() >> shifts[0]));
            }
        }
    }

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

    use super::{
        QuantizeDcMethods, quantize_ac_thresholds, quantize_ac_thresholds_scaled,
        quantize_dc_cfl_scalar, quantize_dc_scalar, selected_quantize_dc_methods,
    };

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
                (super::STRATEGY_DCT, 1, 1),
                (super::STRATEGY_DCT16X8, 2, 1),
                (super::STRATEGY_DCT16X16, 2, 2),
                (super::STRATEGY_DCT32X16, 4, 2),
                (super::STRATEGY_DCT32X32, 4, 4),
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
            (super::STRATEGY_DCT, 1, 1),
            (super::STRATEGY_DCT16X8, 2, 1),
            (super::STRATEGY_DCT16X16, 2, 2),
            (super::STRATEGY_DCT32X16, 4, 2),
            (super::STRATEGY_DCT32X32, 4, 4),
            (super::STRATEGY_DCT64X32, 8, 4),
            (super::STRATEGY_DCT64X64, 8, 8),
            #[cfg(feature = "extraslow")]
            (super::STRATEGY_DCT128X64, 16, 8),
            #[cfg(feature = "extraslow")]
            (super::STRATEGY_DCT128X128, 16, 16),
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
        let raw = super::STRATEGY_DCT16X8;
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

    fn check_dc_quantizers(methods: QuantizeDcMethods) {
        let input = [
            f32::NEG_INFINITY,
            -40_000.5,
            -32_768.5,
            -3.5,
            -2.5,
            -1.5,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.5,
            2.5,
            3.5,
            32_767.0,
            32_767.5,
            40_000.5,
            f32::INFINITY,
            f32::NAN,
            11.25,
        ];
        let y_quant = [
            -31i16, -29, -23, -19, -17, -13, -11, -7, -5, -3, 2, 4, 8, 10, 14, 16, 20, 22, 26,
        ];

        for len in 1..=input.len() {
            let mut want = [i16::MIN; 19];
            let mut got = [i16::MIN; 19];
            quantize_dc_scalar(&input[..len], 1.0, &mut want[..len]);
            (methods.quantize)(&input[..len], 1.0, &mut got[..len]);
            assert_eq!(got, want, "plain DC length {len}");

            want.fill(i16::MIN);
            got.fill(i16::MIN);
            quantize_dc_cfl_scalar(&input[..len], &y_quant[..len], 1.0, 0.5, &mut want[..len]);
            (methods.quantize_cfl)(&input[..len], &y_quant[..len], 1.0, 0.5, &mut got[..len]);
            assert_eq!(got, want, "CfL DC length {len}");
        }

        check_dc_i32_quantizers(methods);
    }

    fn check_dc_i32_quantizers(methods: QuantizeDcMethods) {
        let special = [
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            -0.0,
            0.0,
            -0.5,
            0.5,
            -1.5,
            1.5,
            -2.5,
            2.5,
            f32::from_bits(0.5f32.to_bits() - 1),
            f32::from_bits(0.5f32.to_bits() + 1),
            -f32::from_bits(0.5f32.to_bits() - 1),
            -f32::from_bits(0.5f32.to_bits() + 1),
            -65_535.5,
            65_535.5,
            f32::from_bits(2_147_483_648.0f32.to_bits() - 1),
            2_147_483_648.0,
            -2_147_483_648.0,
            -4_294_967_296.0,
        ];
        let input: Vec<f32> = (0..257)
            .map(|i| {
                if i % 3 == 0 {
                    special[i / 3 % special.len()]
                } else {
                    f32::from_bits(0x3b00_0000 + (i as u32 * 7919) % 0x1000_0000)
                        * if i % 2 == 0 { 1.0 } else { -1.0 }
                }
            })
            .collect();
        let y_quant: Vec<i32> = (0..257)
            .map(|i| match i % 7 {
                0 => i32::MIN,
                1 => i32::MAX,
                _ => (i * 7919) - 900_000,
            })
            .collect();
        for len in (0..=33).chain([63, 64, 65, 127, 128, 129, 257]) {
            for scale in [0.0, -0.0, 0.5, 1.0, -1.25, 1.137, 65535.0] {
                let mut want = vec![0x1234_5678i32; len + 2];
                let mut got = want.clone();
                super::quantize_dc_i32_scalar(&input[..len], scale, &mut want[1..=len]);
                (methods.quantize_i32)(&input[..len], scale, &mut got[1..=len]);
                assert_eq!(got, want, "wide DC length {len}, scale={scale}");
                for cfl in [0.0, -0.0, 0.5, -0.75, 0.137] {
                    super::quantize_dc_cfl_i32_scalar(
                        &input[..len],
                        &y_quant[..len],
                        scale,
                        cfl,
                        &mut want[1..=len],
                    );
                    (methods.quantize_cfl_i32)(
                        &input[..len],
                        &y_quant[..len],
                        scale,
                        cfl,
                        &mut got[1..=len],
                    );
                    assert_eq!(got, want, "wide CfL length {len}, scale={scale}, cfl={cfl}");
                }
            }
        }
    }

    #[test]
    fn selected_dc_quantizers_match_scalar() {
        check_dc_quantizers(selected_quantize_dc_methods());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_dc_quantizers_match_scalar() {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_dc_quantizers(QuantizeDcMethods {
                quantize: |input, scale, output| unsafe {
                    crate::avx::quantize_dc_avx2(input, scale, output)
                },
                quantize_cfl: |input, y_quant, scale, cfl, output| unsafe {
                    crate::avx::quantize_dc_cfl_avx2(input, y_quant, scale, cfl, output)
                },
                quantize_i32: |input, scale, output| unsafe {
                    crate::avx::quantize_dc_i32_avx2(input, scale, output)
                },
                quantize_cfl_i32: |input, y_quant, scale, cfl, output| unsafe {
                    crate::avx::quantize_dc_cfl_i32_avx2(input, y_quant, scale, cfl, output)
                },
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_dc_quantizers_match_scalar() {
        if std::is_x86_feature_detected!("sse4.1") {
            check_dc_quantizers(QuantizeDcMethods {
                quantize: |input, scale, output| unsafe {
                    crate::sse::quantize_dc_sse41(input, scale, output)
                },
                quantize_cfl: |input, y_quant, scale, cfl, output| unsafe {
                    crate::sse::quantize_dc_cfl_sse41(input, y_quant, scale, cfl, output)
                },
                quantize_i32: |input, scale, output| unsafe {
                    crate::sse::quantize_dc_i32_sse41(input, scale, output)
                },
                quantize_cfl_i32: |input, y_quant, scale, cfl, output| unsafe {
                    crate::sse::quantize_dc_cfl_i32_sse41(input, y_quant, scale, cfl, output)
                },
            });
        }
    }

    #[test]
    fn deadzones_fade_to_existing_values_by_distance_one() {
        assert_eq!(
            quantize_ac_thresholds(1, 1, 1, 0.1),
            [0.50, 0.51, 0.52, 0.54]
        );
        assert_eq!(
            quantize_ac_thresholds(2, 1, 1, 0.1),
            [0.58, 0.75, 0.75, 0.75]
        );
        assert_eq!(
            quantize_ac_thresholds(1, 1, 1, 1.0),
            [0.58, 0.635, 0.66, 0.7]
        );
        assert_eq!(
            quantize_ac_thresholds(2, 1, 1, 1.0),
            [0.58, 0.75, 0.75, 0.75]
        );
        assert_eq!(
            quantize_ac_thresholds_scaled(2, 1, 1, 1.0, 1.25),
            [0.58, 0.65, 0.65, 0.65]
        );
        assert_eq!(
            quantize_ac_thresholds_scaled(1, 1, 1, 1.0, 1.25),
            quantize_ac_thresholds(1, 1, 1, 1.0)
        );
    }
}
