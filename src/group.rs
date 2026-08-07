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

use crate::ac_context::{
    fine_block_context, fine_non_zero_context, fine_zero_density_contexts_offset,
    zero_density_context, zero_density_context_8x8,
};
use crate::dc_group_data::{
    AcStrategyImage, DcGroupData, STRATEGY_AFV0, STRATEGY_AFV1, STRATEGY_AFV2, STRATEGY_AFV3,
    STRATEGY_DCT, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4, STRATEGY_DCT8X16,
    STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16, STRATEGY_DCT32X32,
    STRATEGY_DCT32X64, STRATEGY_DCT64X32, STRATEGY_DCT64X64,
};
use crate::dct::{DctInput, dc_from_dct8x16, dc_from_dct16x8, dc_from_dct16x16, fmla};
use crate::encoding_context::EncodingContext;
use crate::entropy::{FrozenTokenPrices, Token, pack_signed};
use crate::image::{Image3B, Image3F, Image3S, Rect};
use crate::quant_weights::{DC_QUANT, INV_DC_QUANT};
use crate::util::{FastRound, HeapMatrix, heap_array};
use std::sync::OnceLock;

const RDOQ_MAX_STRIDE: usize = 2 * (256 + 1);
const RDOQ_MAX_CHOICES: usize = 64 * RDOQ_MAX_STRIDE;

const K_GROUP_DIM_IN_BLOCKS: usize = 32;

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

#[inline]
fn rdoq_candidates(ideal: f32, current: i32) -> ([i32; 5], usize) {
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

#[allow(clippy::too_many_arguments)]
fn rdoq_block(
    prices: &FrozenTokenPrices,
    scan: &[u32],
    source: &[f32],
    inv_qm: &[f32],
    q_scaled: f32,
    block: &mut [i32],
    raw_strategy: u8,
    strategy_code: u8,
    c: usize,
    predicted: u8,
    cx: usize,
    cy: usize,
    distance: f32,
    qf_hi: bool,
    choices: &mut [u8; RDOQ_MAX_CHOICES],
    costs: &mut [[f32; RDOQ_MAX_STRIDE]; 2],
) {
    const RDOQ_LAMBDA: f32 = crate::ac_strategy::RD_LAMBDA * 0.25;
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
    ) {
        return;
    }
    let covered_blocks = cx * cy;
    let search_end = block
        .len()
        .min(covered_blocks + (24 * covered_blocks).min(64));
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
            *target += RDOQ_LAMBDA
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
    if stride > RDOQ_MAX_STRIDE || window_len * stride > RDOQ_MAX_CHOICES {
        return;
    }
    let [next_buf, current_buf] = costs;
    let mut next = &mut next_buf[..stride];
    let mut current_cost = &mut current_buf[..stride];
    next.fill(f32::INFINITY);
    current_cost.fill(f32::INFINITY);
    next[suffix_nzeros * 2] = suffix_cost[0];
    next[suffix_nzeros * 2 + 1] = suffix_cost[1];
    choices[..window_len * stride].fill(u8::MAX);

    let mut original_after_nzeros = 0usize;
    for window_index in (0..window_len).rev() {
        current_cost.fill(f32::INFINITY);
        let k = covered_blocks + window_index;
        let idx = scan[k] as usize;
        let ideal = source[idx] * inv_qm[idx] * q_scaled;
        let distortion_weight = crate::inflated_cost::CHANNEL_WEIGHT[c]
            * rdoq_distortion_weight(window_index, window_len, distance);
        let (candidates, candidate_count) = rdoq_candidates(ideal, block[idx]);
        // Distortion against what the decoder actually reconstructs (the biased
        // dequant: +-1 -> +-0.9299, q -> q - 0.145/q), not the raw integer level.
        let mut candidate_distortion = [0.0f32; 5];
        for (d, &level) in candidate_distortion[..candidate_count]
            .iter_mut()
            .zip(candidates.iter())
        {
            *d = distortion_weight * (ideal - dequantized_level(level)).powi(2);
        }
        let processed_after = window_len - 1 - window_index;
        let min_remaining = suffix_nzeros + original_after_nzeros.saturating_sub(MAX_NZERO_DELTA);
        let max_remaining =
            suffix_nzeros + (original_after_nzeros + MAX_NZERO_DELTA).min(processed_after);
        for next_remaining in min_remaining..=max_remaining {
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
                    [RDOQ_LAMBDA * bits[0], RDOQ_LAMBDA * bits[1]]
                };
                let state = remaining * 2;
                let cost0 = distortion + token_cost[0] + tail;
                if cost0 < current_cost[state] {
                    current_cost[state] = cost0;
                    choices[window_index * stride + state] = candidate_index as u8;
                }
                let cost1 = distortion + token_cost[1] + tail;
                if cost1 < current_cost[state + 1] {
                    current_cost[state + 1] = cost1;
                    choices[window_index * stride + state + 1] = candidate_index as u8;
                }
            }
        }
        std::mem::swap(&mut current_cost, &mut next);
        original_after_nzeros += usize::from(block[idx] != 0);
    }

    let nzero_ctx = fine_non_zero_context(predicted as u32, block_ctx);
    let mut best_remaining = 0;
    let mut best_cost = f32::INFINITY;
    for remaining in 0..=max_nzeros {
        let initial_prev = usize::from(remaining <= block.len() / 16);
        let tail = next[remaining * 2 + initial_prev];
        if !tail.is_finite() {
            continue;
        }
        let cost = tail + RDOQ_LAMBDA * prices.token_bits(Token::new(nzero_ctx, remaining as u32));
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
        let (candidates, candidate_count) = rdoq_candidates(ideal, block[idx]);
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

#[derive(Clone, Copy)]
pub(crate) struct QuantizeDcMethods {
    pub(crate) quantize: QuantizeDcFn,
    pub(crate) quantize_cfl: QuantizeDcCflFn,
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
        };
    }

    QuantizeDcMethods {
        quantize: quantize_dc_scalar,
        quantize_cfl: quantize_dc_cfl_scalar,
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
    let mut normal = [0.58f32, 0.635, 0.66, 0.7];
    if c == 0 {
        for t in &mut normal[1..] {
            *t += 0.08;
        }
    }
    if c == 2 {
        for t in &mut normal[1..] {
            *t = 0.75;
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
    let thr = quantize_ac_thresholds(c, xsize, ysize, distance);
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

/// Chroma (X/B) RDOQ opens at the SS2 quant tier: below it the trellis
/// regresses Kodak HQ BD at every lambda tried (+0.27% unweighted, +0.39%
/// channel-weighted), above it the channel-weighted form wins −0.137%.
pub(crate) const CHROMA_RDOQ_MIN_DISTANCE: f32 = 2.25;

pub(crate) const DEFAULT_QUANT_BIAS_1: f32 = 1.0 - 0.07005449891748593;
pub(crate) const DEFAULT_QUANT_BIAS_3: f32 = 0.145;

/// [`dequantized_level`] for a float-valued (but integer) level, as produced by
/// the encoder-side quantizers before conversion to `i32`.
///
/// Currently only exercised by the (measured, deferred) biased-distortion
/// variants of `sse_and_rate_scalar` / `recon_quantize_scalar`: plugging the
/// true dequant into those paths shifts strategy selection, so it has to land
/// together with a merge-margin / rerank-lambda re-fit.
#[allow(dead_code)]
#[inline]
pub(crate) fn dequantized_level_f32(quant: f32) -> f32 {
    let aq = quant.abs();
    if aq < 1.125 {
        if quant == 0.0 {
            0.0
        } else {
            DEFAULT_QUANT_BIAS_1.copysign(quant)
        }
    } else {
        quant - DEFAULT_QUANT_BIAS_3 / quant
    }
}

#[inline]
pub(crate) fn dequantized_level(quant: i32) -> f32 {
    let aq = quant.unsigned_abs() as f32;
    if aq < 1.125 {
        if quant == 0 {
            0.0
        } else if quant > 0 {
            DEFAULT_QUANT_BIAS_1
        } else {
            -DEFAULT_QUANT_BIAS_1
        }
    } else {
        let q = quant as f32;
        q - DEFAULT_QUANT_BIAS_3 / q
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
    let size = xsize * ysize * 64;
    for (out, (&q, &dq)) in inout[..size]
        .iter_mut()
        .zip(quantized[..size].iter().zip(dqm[..size].iter()))
    {
        *out = dequantized_level(q) * dq * inv_qac;
    }
}

pub(crate) struct AcGroupScratch {
    coeffs: HeapMatrix<f32, 3, 4096>,
    quantized: HeapMatrix<i32, 3, 4096>,
    source_y: Box<[f32; 4096]>,
    block: Box<[i32; 4096]>,
    rdoq_choices: Box<[u8; RDOQ_MAX_CHOICES]>,
    rdoq_costs: HeapMatrix<f32, 2, RDOQ_MAX_STRIDE>,
}

impl Default for AcGroupScratch {
    fn default() -> Self {
        Self {
            coeffs: HeapMatrix::new(0.0),
            quantized: HeapMatrix::new(0),
            source_y: heap_array(0.0),
            block: heap_array(0),
            rdoq_choices: heap_array(u8::MAX),
            rdoq_costs: HeapMatrix::new(f32::INFINITY),
        }
    }
}

/// Process and tokenize one stripe of an AC group, pushing tokens into `out`.
/// Callers buffer tokens across all AC groups, build an adaptive entropy code
/// from the aggregate distribution, then emit them in `encode_frame`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ac_group(
    ctx: &EncodingContext,
    scratch: &mut AcGroupScratch,
    opsin: &Image3F,
    group_brect: Rect,
    scale: f32,
    scale_dc: f32,
    distance: f32,
    x_qm_scale: u32,
    dc_data: &DcGroupData,
    ytob_dc: i32,
    quant_dc: &mut Image3S,
    dc_float: &mut Image3F,
    qorigin_x: usize,
    qorigin_y: usize,
    num_nzeros: &mut [Image3B],
    coeff_shifts: &[u32],
    rdoq_prices: Option<&FrozenTokenPrices>,
    coeff_orders: &crate::coeff_order::CoeffOrders,
    mut order_stats: Option<&mut crate::coeff_order::OrderStats>,
    measure_chroma_distortion: bool,
    qf_threshold: u32,
    out: &mut [Vec<Token>],
) -> f32 {
    let matrices = &ctx.matrices;
    let xsize_blocks = group_brect.xsize;
    let ysize_blocks = group_brect.ysize;

    let inv_factor = [
        INV_DC_QUANT[0] * scale_dc,
        INV_DC_QUANT[1] * scale_dc,
        INV_DC_QUANT[2] * scale_dc,
    ];
    // `base_correlation_b` (= 1) plus the frame's signaled `ytob_dc / 84`,
    // converted from dequantized XYB into stored-B-DC units. Folded into the
    // quantizer below so the slope costs no extra rounding error.
    let cfl_factor_b = INV_DC_QUANT[2]
        * DC_QUANT[1]
        * (1.0 + ytob_dc as f32 / crate::color_correlation::K_COLOR_FACTOR);
    let x_qm_mul = 1.25f32.powf(x_qm_scale as f32 - 2.0);

    let nzeros_by0 = group_brect.y0 % K_GROUP_DIM_IN_BLOCKS;
    let mut chroma_distortion = 0.0f32;

    // All the big per-block buffers live in the worker scratch: re-creating them
    // per group cost ~130 KB of zeroing, and `pblock` was being zeroed once per
    // block per pass. Every one is written over `..size` before it is read, so
    // carrying values across groups is not observable.
    let AcGroupScratch {
        coeffs,
        quantized,
        source_y,
        block: pblock,
        rdoq_choices,
        rdoq_costs,
    } = scratch;

    for by in 0..ysize_blocks {
        let nz_by = nzeros_by0 + by;
        let global_by = group_brect.y0 + by;

        for bx in 0..xsize_blocks {
            let global_bx = group_brect.x0 + bx;

            // Skip non-first blocks of multi-block transforms.
            if !dc_data.ac_strategy.is_first_block(global_bx, global_by) {
                continue;
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

            // ---- Forward DCT for all 3 channels ----
            let opsin_bx = bx * 8;
            let opsin_by = by * 8;
            for c in 0..3 {
                let plane = opsin.plane(c);
                let stride = plane.xsize();
                let input = &opsin.plane_data(c)[opsin_by * stride + opsin_bx..];
                match raw_strategy {
                    STRATEGY_DCT => {
                        let dst: &mut [f32; 64] = coeffs[c].first_chunk_mut::<64>().unwrap();
                        (ctx.dct8x8)(DctInput::new(input, stride), dst);
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
                        let dst: &mut [f32; 1024] = coeffs[c].first_chunk_mut::<1024>().unwrap();
                        (ctx.dct32x32)(DctInput::new(input, stride), dst);
                    }
                    STRATEGY_DCT64X64 => {
                        let dst: &mut [f32; 4096] = coeffs[c].first_chunk_mut::<4096>().unwrap();
                        (ctx.dct64x64)(DctInput::new(input, stride), dst);
                    }
                    STRATEGY_DCT64X32 => {
                        let dst: &mut [f32; 2048] = coeffs[c].first_chunk_mut::<2048>().unwrap();
                        (ctx.dct64x32)(DctInput::new(input, stride), dst);
                    }
                    STRATEGY_DCT32X64 => {
                        let dst: &mut [f32; 2048] = coeffs[c].first_chunk_mut::<2048>().unwrap();
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
                    _ => unreachable!("invalid raw strategy {}", raw_strategy),
                }
            }

            // For DCT8, DC = coeffs[0]. For multi-block, use DCFromLowestFrequencies.
            // dc_vals[c] holds up to 4 DC values (DCT16X16 = 2×2 covered blocks);
            // indexing is didx = iy * cov_x + ix.
            let mut dc_vals = [[0.0f32; 64]; 3];
            match raw_strategy {
                STRATEGY_DCT
                | STRATEGY_DCT4X4
                | STRATEGY_DCT4X8
                | STRATEGY_DCT8X4
                | STRATEGY_AFV0..=STRATEGY_AFV3 => {
                    for c in 0..3 {
                        dc_vals[c][0] = coeffs[c][0];
                    }
                }
                STRATEGY_DCT16X8 => {
                    for c in 0..3 {
                        let cb: &[f32; 128] = coeffs[c].first_chunk::<128>().unwrap();
                        dc_from_dct16x8(cb, dc_vals[c].first_chunk_mut::<2>().unwrap());
                    }
                }
                STRATEGY_DCT8X16 => {
                    for c in 0..3 {
                        let cb: &[f32; 128] = coeffs[c].first_chunk::<128>().unwrap();
                        dc_from_dct8x16(cb, dc_vals[c].first_chunk_mut::<2>().unwrap());
                    }
                }
                STRATEGY_DCT16X16 => {
                    // dc_from_dct16x16 returns 4 DC values in [TL, TR, BL, BR]
                    // order, matching the [iy=0,1][ix=0,1] grid the caller uses
                    // (didx = iy * 2 + ix).
                    for c in 0..3 {
                        let cb: &[f32; 256] = coeffs[c].first_chunk::<256>().unwrap();
                        dc_from_dct16x16(cb, dc_vals[c].first_chunk_mut::<4>().unwrap());
                    }
                }
                STRATEGY_DCT32X32 => {
                    // dc_from_dct32x32 returns 16 DC values in the 4×4 grid the
                    // caller uses (didx = iy * 4 + ix).
                    for c in 0..3 {
                        let cb: &[f32; 1024] = coeffs[c].first_chunk::<1024>().unwrap();
                        (ctx.dc_from_dct32x32)(cb, dc_vals[c].first_chunk_mut::<16>().unwrap());
                    }
                }
                STRATEGY_DCT64X64 => {
                    for c in 0..3 {
                        let cb: &[f32; 4096] = coeffs[c].first_chunk::<4096>().unwrap();
                        (ctx.dc_from_dct64x64)(cb, &mut dc_vals[c]);
                    }
                }
                STRATEGY_DCT64X32 => {
                    for c in 0..3 {
                        let cb: &[f32; 2048] = coeffs[c].first_chunk::<2048>().unwrap();
                        (ctx.dc_from_dct64x32)(cb, dc_vals[c].first_chunk_mut::<32>().unwrap());
                    }
                }
                STRATEGY_DCT32X64 => {
                    for c in 0..3 {
                        let cb: &[f32; 2048] = coeffs[c].first_chunk::<2048>().unwrap();
                        (ctx.dc_from_dct32x64)(cb, dc_vals[c].first_chunk_mut::<32>().unwrap());
                    }
                }
                STRATEGY_DCT32X16 => {
                    // 8 DC values in a 4-row × 2-col grid (didx = iy*2 + ix),
                    // matching cov_x=2, cov_y=4.
                    for c in 0..3 {
                        let cb: &[f32; 512] = coeffs[c].first_chunk::<512>().unwrap();
                        (ctx.dc_from_dct32x16)(cb, dc_vals[c].first_chunk_mut::<8>().unwrap());
                    }
                }
                STRATEGY_DCT16X32 => {
                    // 8 DC values in a 2-row × 4-col grid (didx = iy*4 + ix),
                    // matching cov_x=4, cov_y=2.
                    for c in 0..3 {
                        let cb: &[f32; 512] = coeffs[c].first_chunk::<512>().unwrap();
                        (ctx.dc_from_dct16x32)(cb, dc_vals[c].first_chunk_mut::<8>().unwrap());
                    }
                }
                _ => unreachable!(),
            }

            // DC for storage (per covered block, using pre-swap cov_x/cov_y).
            let covered_dc = cov_x * cov_y;
            let mut y_dc_q = [0i16; 64];
            (ctx.quantize_dc)(
                &dc_vals[1][..covered_dc],
                inv_factor[1],
                &mut y_dc_q[..covered_dc],
            );
            for iy in 0..cov_y {
                let lbx = global_bx - qorigin_x;
                let quant_target =
                    &mut quant_dc.plane_row_mut(1, global_by - qorigin_y + iy)[lbx..lbx + cov_x];
                let row_start = iy * cov_x;
                quant_target.copy_from_slice(&y_dc_q[row_start..row_start + cov_x]);
                if dc_float.xsize() != 0 {
                    let float_target = &mut dc_float.plane_row_mut(1, global_by - qorigin_y + iy)
                        [lbx..lbx + cov_x];
                    float_target.copy_from_slice(&dc_vals[1][row_start..row_start + cov_x]);
                }
            }
            // Quantize Y AC with roundtrip (modifies coeffs[1] to dequantized).
            // Matrix selection: DCT8 uses 8×8 weights, DCT16X8/8X16 share the
            // 128-float 16×8 weights, DCT16X16 uses the 256-float 16×16 weights.
            let (inv_qm_y, qm_y): (&[f32], &[f32]) = match raw_strategy {
                    STRATEGY_DCT => (&matrices.inv_matrix(1)[..], &matrices.matrix(1)[..]),
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
                    STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => {
                        (&matrices.inv_matrix_64x32(1)[..], &matrices.matrix_64x32(1)[..])
                    }
                    STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => {
                        (&matrices.inv_matrix_32x16(1)[..], &matrices.matrix_32x16(1)[..])
                    }
                    _ /* 16X8/8X16 */ => (&matrices.inv_matrix_16x8(1)[..], &matrices.matrix_16x8(1)[..]),
                };
            source_y[..size].copy_from_slice(&coeffs[1][..size]);
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
                let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
                let nzero_map = &num_nzeros[0];
                let row_top = (nz_by != 0).then(|| nzero_map.plane_row(1, nz_by - 1));
                let predicted =
                    predict_from_top_and_left(row_top, nzero_map.plane_row(1, nz_by), bx, 32);
                rdoq_block(
                    prices,
                    coeff_orders.scan_for(strategy_code, 1),
                    &source_y[..size],
                    inv_qm_y,
                    quantize_ac_q_scaled(quant_ac, scale, 1.0),
                    &mut quantized[1][..size],
                    raw_strategy,
                    strategy_code,
                    1,
                    predicted,
                    cx,
                    cy,
                    distance,
                    quant_ac as u32 > qf_threshold,
                    rdoq_choices,
                    rdoq_costs,
                );
                let inv_qac = 1.0 / (scale * quant_ac as f32);
                for (out, (&q, &dq)) in coeffs[1][..size]
                    .iter_mut()
                    .zip(quantized[1][..size].iter().zip(qm_y[..size].iter()))
                {
                    *out = dequantized_level(q) * dq * inv_qac;
                }
            }

            // ---- Per-tile CfL factors ----
            let tx = global_bx / 8;
            let ty = global_by / 8;
            let cmap_x = dc_data.ytox_map.row(ty)[tx];
            let cmap_b = dc_data.ytob_map.row(ty)[tx];
            // y_to_x = 0 + cmap_x / 84;  y_to_b = 1 + cmap_b / 84.
            let x_factor = crate::color_correlation::y_to_x_ratio(cmap_x);
            let b_factor = crate::color_correlation::y_to_b_ratio(cmap_b);

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
            let mut x_dc_post = [0.0f32; 64];
            let mut b_dc_post = [0.0f32; 64];
            match raw_strategy {
                STRATEGY_DCT
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
                    (ctx.dc_from_dct64x64)(xb, &mut x_dc_post);
                    (ctx.dc_from_dct64x64)(bb, &mut b_dc_post);
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
                _ => unreachable!(),
            }

            // ---- X channel: write post-CfL DC, quantize AC ----
            let mut chroma_dc_q = [0i16; 64];
            (ctx.quantize_dc)(
                &x_dc_post[..covered_dc],
                inv_factor[0],
                &mut chroma_dc_q[..covered_dc],
            );
            for iy in 0..cov_y {
                let lbx = global_bx - qorigin_x;
                let quant_dc_row =
                    &mut quant_dc.plane_row_mut(0, global_by - qorigin_y + iy)[lbx..lbx + cov_x];
                let row_start = iy * cov_x;
                quant_dc_row.copy_from_slice(&chroma_dc_q[row_start..row_start + cov_x]);
                if dc_float.xsize() != 0 {
                    let float_target = &mut dc_float.plane_row_mut(0, global_by - qorigin_y + iy)
                        [lbx..lbx + cov_x];
                    float_target.copy_from_slice(&x_dc_post[row_start..row_start + cov_x]);
                }
            }
            let inv_qm_x: &[f32] = match raw_strategy {
                STRATEGY_DCT => &matrices.inv_matrix(0)[..],
                STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(0)[..],
                STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(0)[..],
                STRATEGY_AFV0..=STRATEGY_AFV3 => &matrices.inv_matrix_afv(0)[..],
                STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(0)[..],
                STRATEGY_DCT32X32 => &matrices.inv_matrix_32x32(0)[..],
                STRATEGY_DCT64X64 => &matrices.inv_matrix_64x64(0)[..],
                STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => &matrices.inv_matrix_64x32(0)[..],
                STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => &matrices.inv_matrix_32x16(0)[..],
                _ => &matrices.inv_matrix_16x8(0)[..],
            };
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
            // overwritten afterwards. Mid-band only: with CHANNEL_WEIGHT'd
            // distortion it reads −0.137% Kodak BD at d≥2.5 but +0.39% at
            // d=1-2.2 (HQ chroma coefficients are precious — same story as
            // the deadzone/flat-B studies), so it opens at the SS2 tier.
            if distance >= CHROMA_RDOQ_MIN_DISTANCE
                && let Some(prices) = rdoq_prices
            {
                let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
                let nzero_map = &num_nzeros[0];
                let row_top = (nz_by != 0).then(|| nzero_map.plane_row(0, nz_by - 1));
                let predicted =
                    predict_from_top_and_left(row_top, nzero_map.plane_row(0, nz_by), bx, 32);
                rdoq_block(
                    prices,
                    coeff_orders.scan_for(strategy_code, 0),
                    &coeffs[0][..size],
                    inv_qm_x,
                    quantize_ac_q_scaled(quant_ac, scale, x_qm_mul),
                    &mut quantized[0][..size],
                    raw_strategy,
                    strategy_code,
                    0,
                    predicted,
                    cx,
                    cy,
                    distance,
                    quant_ac as u32 > qf_threshold,
                    rdoq_choices,
                    rdoq_costs,
                );
            }
            let row_stride = cx * 8;
            if measure_chroma_distortion {
                let q_scaled_x = quantize_ac_q_scaled(quant_ac, scale, x_qm_mul);
                for i in 0..size {
                    let v = i / row_stride;
                    let u = i % row_stride;
                    if v < cy && u < cx {
                        continue;
                    }
                    let ideal = coeffs[0][i] * inv_qm_x[i] * q_scaled_x;
                    let error = ideal - dequantized_level(quantized[0][i]);
                    chroma_distortion += crate::inflated_cost::CHANNEL_WEIGHT[0] * error * error;
                }
            }

            // ---- B channel: write CfL'd DC, quantize AC ----
            (ctx.quantize_dc_cfl)(
                &b_dc_post[..covered_dc],
                &y_dc_q[..covered_dc],
                inv_factor[2],
                cfl_factor_b,
                &mut chroma_dc_q[..covered_dc],
            );
            for iy in 0..cov_y {
                let lbx = global_bx - qorigin_x;
                let quant_dc_row =
                    &mut quant_dc.plane_row_mut(2, global_by - qorigin_y + iy)[lbx..lbx + cov_x];
                let row_start = iy * cov_x;
                quant_dc_row.copy_from_slice(&chroma_dc_q[row_start..row_start + cov_x]);
                if dc_float.xsize() != 0 {
                    let float_target = &mut dc_float.plane_row_mut(2, global_by - qorigin_y + iy)
                        [lbx..lbx + cov_x];
                    float_target.copy_from_slice(&b_dc_post[row_start..row_start + cov_x]);
                }
            }
            let inv_qm_b: &[f32] = match raw_strategy {
                STRATEGY_DCT => &matrices.inv_matrix(2)[..],
                STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(2)[..],
                STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(2)[..],
                STRATEGY_AFV0..=STRATEGY_AFV3 => &matrices.inv_matrix_afv(2)[..],
                STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(2)[..],
                STRATEGY_DCT32X32 => &matrices.inv_matrix_32x32(2)[..],
                STRATEGY_DCT64X64 => &matrices.inv_matrix_64x64(2)[..],
                STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => &matrices.inv_matrix_64x32(2)[..],
                STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => &matrices.inv_matrix_32x16(2)[..],
                _ => &matrices.inv_matrix_16x8(2)[..],
            };
            (ctx.quantize_block_ac)(
                &coeffs[2][..size],
                2,
                inv_qm_b,
                quant_ac,
                scale,
                crate::frame::b_qm_mul(),
                distance,
                cx,
                cy,
                &mut quantized[2][..size],
            );
            if distance >= CHROMA_RDOQ_MIN_DISTANCE
                && let Some(prices) = rdoq_prices
            {
                let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
                let nzero_map = &num_nzeros[0];
                let row_top = (nz_by != 0).then(|| nzero_map.plane_row(2, nz_by - 1));
                let predicted =
                    predict_from_top_and_left(row_top, nzero_map.plane_row(2, nz_by), bx, 32);
                rdoq_block(
                    prices,
                    coeff_orders.scan_for(strategy_code, 2),
                    &coeffs[2][..size],
                    inv_qm_b,
                    quantize_ac_q_scaled(quant_ac, scale, crate::frame::b_qm_mul()),
                    &mut quantized[2][..size],
                    raw_strategy,
                    strategy_code,
                    2,
                    predicted,
                    cx,
                    cy,
                    distance,
                    quant_ac as u32 > qf_threshold,
                    rdoq_choices,
                    rdoq_costs,
                );
            }
            if measure_chroma_distortion {
                let q_scaled_b = quantize_ac_q_scaled(quant_ac, scale, crate::frame::b_qm_mul());
                for i in 0..size {
                    let v = i / row_stride;
                    let u = i % row_stride;
                    if v < cy && u < cx {
                        continue;
                    }
                    let ideal = coeffs[2][i] * inv_qm_b[i] * q_scaled_b;
                    let error = ideal - dequantized_level(quantized[2][i]);
                    chroma_distortion = fmla(
                        crate::inflated_cost::CHANNEL_WEIGHT[2],
                        error * error,
                        chroma_distortion,
                    );
                }
            }

            // ---- Tokenize in order Y, X, B ----
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
                _ => unreachable!("invalid covered_blocks {}", covered_blocks),
            };

            for &c in &[1usize, 0, 2] {
                let full_block = &quantized[c][..size];

                for pass in 0..coeff_shifts.len() {
                    // Materialize the coefficients pass `pass` transmits. With
                    // decreasing per-pass shifts ending at 0, the decoder sums
                    // (sent_p << shift_p) over passes to recover `full_block`
                    // (jxl-vardct hf_coeff.rs:185,191). For 2 passes/shifts
                    // [s,0]: pass0 = C>>s, pass1 = C-((C>>s)<<s).
                    for k in 0..size {
                        let mut remaining = full_block[k];
                        let mut sent = 0i32;
                        for p in 0..=pass {
                            sent = remaining >> coeff_shifts[p];
                            remaining -= sent << coeff_shifts[p];
                        }
                        pblock[k] = sent;
                    }
                    let block = &pblock[..size];
                    let num_nzeros = &mut num_nzeros[pass];
                    let out = &mut out[pass];

                    let nzeros = if covered_blocks == 1 {
                        num_nonzero_except_dc(<&[i32; 64]>::try_from(block).unwrap())
                    } else {
                        num_nonzero_except_llf(block, cx, cy)
                    };

                    // libjxl-tiny: NumNonZeroExceptLLF stores `(nzeros + covered_blocks - 1) >> log2_covered_blocks`
                    // to all covered cells in num_nzeros.
                    let shifted =
                        ((nzeros as usize + covered_blocks - 1) >> log2_covered_blocks) as u8;
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
                    debug_assert_eq!(
                        remaining, 0,
                        "remaining nzeros at end: strategy={} c={} pass={}",
                        strategy_code, c, pass
                    );
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
    use super::{
        QuantizeDcMethods, quantize_ac_thresholds, quantize_dc_cfl_scalar, quantize_dc_scalar,
        selected_quantize_dc_methods,
    };

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
    }

    #[test]
    fn selected_dc_quantizers_match_scalar() {
        check_dc_quantizers(selected_quantize_dc_methods());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_dc_quantizers_match_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            check_dc_quantizers(QuantizeDcMethods {
                quantize: |input, scale, output| unsafe {
                    crate::avx::quantize_dc_avx2(input, scale, output)
                },
                quantize_cfl: |input, y_quant, scale, cfl, output| unsafe {
                    crate::avx::quantize_dc_cfl_avx2(input, y_quant, scale, cfl, output)
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
    }
}
