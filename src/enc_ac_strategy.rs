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
//! Adaptive transform (AC strategy) selection via rate-distortion optimization.
//!
//! For each aligned 2×2 super-block we evaluate the candidate transform layouts
//! — four DCT8, two vertical DCT16X8, two horizontal DCT8X16, and one DCT16X16 —
//! and choose the one that minimizes the Lagrangian cost `J = D + λ·R`:
//!
//! * **D — distortion (SSE).** The transform is forward-DCT'd and quantized
//!   exactly as the real encoder does (`enc_group::quantize_block_ac`), then the
//!   per-coefficient quantization error is squared and summed. This error is
//!   measured in the *dequant-normalized* coefficient space: the JXL dequant
//!   matrices fold both the basis scaling and the perceptual weighting into the
//!   step size, so a uniform error in this space is (by construction) a uniform
//!   perceptual error. Summing its square is therefore a perceptual SSE that is
//!   directly comparable across transform sizes. Unlike the old libjxl-tiny
//!   `info_loss` heuristic, the error is *threshold-aware*: a coefficient that
//!   the quantizer kills (`|a| < threshold → 0`) contributes its full magnitude,
//!   not just a rounding residue.
//!
//! * **R — rate.** A bit estimate for the quantized AC coefficients: a per-
//!   nonzero context/sign overhead plus a magnitude term (`log2(1+|q|)`), plus a
//!   small `num_nonzeros` header term. This approximates the ANS-coded cost.
//!
//! * **λ — Lagrange multiplier.** Because the quantizer always rounds to integer
//!   quant-units (step Δ = 1), the high-rate-optimal multiplier for entropy-
//!   constrained scalar quantization, `λ* = Δ²·ln2/6`, is a constant independent
//!   of the target distance. That gives a principled, distance-robust trade-off
//!   without per-distance magic constants.

use crate::dc_group_data::{
    AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4,
    STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16,
    STRATEGY_DCT32X32,
};
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, ImageB, ImageSB};
use crate::quant_weights::DequantMatrices;
use crate::util::FastRound;
use std::sync::OnceLock;

/// At visually lossless / near-lossless settings, transform metadata and the
/// occasional wrong merge cost more than the large-DCT decorrelation saves.
const DCT8_ONLY_MAX_DISTANCE: f32 = 0.5;

/// Rate-term lookup: `mag_bits` accumulates `log2(1 + |q|)` where `q` is always
/// a *rounded* (hence non-negative integer-valued) quantized coefficient. Since
/// `1 + |q|` is an exact integer-valued `f32`, the log can be precomputed once
/// and looked up — **bit-identical** to calling `f32::log2` (same inputs, same
/// `log2f`), but trading a per-coefficient transcendental for an L1 load. Values
/// of `|q|` at or above the table size (rare, large coefficients) fall back to
/// the direct computation.
pub(crate) const RATE_LOG2_LUT_N: usize = 1024;
pub(crate) type RateLog2Lut = [f32; RATE_LOG2_LUT_N];

#[inline]
pub(crate) fn rate_log2_lut() -> &'static RateLog2Lut {
    static LUT: OnceLock<RateLog2Lut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut a = [0.0f32; RATE_LOG2_LUT_N];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (1.0 + i as f32).log2();
        }
        a
    })
}

#[inline]
fn use_dct8_only(distance: f32) -> bool {
    distance <= DCT8_ONLY_MAX_DISTANCE
}

/// `log2(1 + qabs)` for a non-negative integer-valued `qabs`, via a resolved LUT.
#[inline]
pub(crate) fn rate_log2_with_lut(lut: &RateLog2Lut, qabs: f32) -> f32 {
    let k = qabs as usize;
    if k < RATE_LOG2_LUT_N {
        lut[k]
    } else {
        (1.0 + qabs).log2()
    }
}

/// High-rate-optimal Lagrange multiplier for unit-step (Δ = 1) scalar
/// quantization: `λ* = Δ²·ln2 / 6`. Distortion is in quant-units², rate in
/// bits, so `λ·R` is in quant-units² and adds cleanly to D.
pub(crate) const RD_LAMBDA: f32 = 0.080_867_17;

/// Per-channel distortion weights. The dequant matrices already normalize each
/// channel perceptually, so equal weights are the principled default; X (red-
/// green) gets a touch more weight because the selection's distortion model is
/// deliberately compact.
static CHANNEL_WEIGHT: [f32; 3] = [1.0, 1.0, 1.0];

/// Fixed overhead per nonzero AC coefficient (context selection + sign).
const R_NZ_BASE: f32 = 1.6;
/// Weight on the magnitude term `log2(1+|q|)`.
const R_MAG: f32 = 1.0;
/// Weight on the `num_nonzeros` header term `log2(1+nzeros)`.
const R_HEADER: f32 = 0.4;

const BIAS_RECT: f32 = 1.0;
const BIAS_16X16: f32 = 1.0;
const BIAS_32X32: f32 = 1.0;
const BIAS_RECT32: f32 = 1.10;

const MERGE_MARGIN_PAIR_HQ: f32 = 0.04;
const MERGE_MARGIN_16_HQ: f32 = 0.08;
const MERGE_MARGIN_32_RECT_HQ: f32 = 0.11;
const MERGE_MARGIN_32_HQ: f32 = 0.14;

const MERGE_MARGIN_LOWQ_FRACTION: f32 = 0.20;
const MERGE_MARGIN_FADE_START: f32 = 1.0;
const MERGE_MARGIN_FADE_END: f32 = 4.0;
const BIAS_4X4: f32 = 1.0;
const BIAS_4X8: f32 = 1.0;

#[inline]
fn merge_margin(distance: f32, high_quality_margin: f32) -> f32 {
    let fade = ((distance - MERGE_MARGIN_FADE_START)
        / (MERGE_MARGIN_FADE_END - MERGE_MARGIN_FADE_START))
        .clamp(0.0, 1.0);
    high_quality_margin * (1.0 - fade * (1.0 - MERGE_MARGIN_LOWQ_FRACTION))
}

#[inline]
fn merge_beats_dct8(
    candidate_cost: f32,
    dct8_cost: f32,
    distance: f32,
    high_quality_margin: f32,
) -> bool {
    candidate_cost < dct8_cost * (1.0 - merge_margin(distance, high_quality_margin))
}

#[derive(Clone, Copy)]
struct SuperBlockCost {
    /// Estimated cost of the layout actually committed by `select_super_block`.
    chosen: f32,
    /// Cost of the pure four-DCT8 incumbent before any merge decisions.
    dct8: f32,
}

thread_local! {
    /// Reused gather scratch for [`forward_transform`] (avoids re-zeroing 1024
    /// floats on every call). Single-threaded encode; one buffer per thread.
    static FT_GATHER_SCRATCH: std::cell::RefCell<[f32; 1024]> =
        const { std::cell::RefCell::new([0.0; 1024]) };
    /// Reused per-channel coefficient scratch for [`strategy_cost`].
    static SC_COEFFS_SCRATCH: std::cell::RefCell<[[f32; 1024]; 3]> =
        const { std::cell::RefCell::new([[0.0; 1024]; 3]) };
}

/// Forward-transform the `strategy`'s pixel footprint at absolute pixel
/// `(px, py)` for one channel into `out` (natural coefficient storage matching
/// `write_ac_group`). Returns `(cx, cy)` covered-block counts after the
/// libjxl-tiny `cx ≥ cy` normalisation, i.e. the storage shape in 8-blocks.
fn forward_transform(
    ctx: &EncodingContext,
    strategy: u8,
    plane: &crate::image::Plane<f32>,
    px: usize,
    py: usize,
    out: &mut [f32; 1024],
) -> (usize, usize) {
    let pw = plane.xsize();
    let ph = plane.ysize();
    // Gather `w×h` pixels with edge replication, matching `build_stripe`'s
    // padding so the selection sees exactly what `write_ac_group` will transform.
    let gather = |w: usize, h: usize, dst: &mut [f32]| {
        let safe_w = w.min(pw.saturating_sub(px));
        let safe_h = h.min(ph.saturating_sub(py));
        for v in 0..h {
            let sy = if v < safe_h { py + v } else { ph - 1 };
            let row = plane.row(sy);
            let drow = &mut dst[v * w..v * w + w];
            drow[..safe_w].copy_from_slice(&row[px..px + safe_w]);
            if safe_w < w {
                let edge = row[pw - 1];
                for d in &mut drow[safe_w..] {
                    *d = edge;
                }
            }
        }
    };
    // Reused scratch: the gather fully overwrites the region each transform reads,
    // so re-zeroing a fresh `[0.0; 1024]` on every call is pure waste (this is the
    // hottest function in selection — thousands of calls per group).
    FT_GATHER_SCRATCH.with(|cell| {
        let tmp = &mut *cell.borrow_mut();
        match strategy {
            STRATEGY_DCT => {
                gather(8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct8x8)(src, dst);
                (1, 1)
            }
            STRATEGY_DCT16X8 => {
                gather(8, 16, &mut tmp[..128]);
                let src: &[f32; 128] = (&tmp[..128]).try_into().unwrap();
                let dst: &mut [f32; 128] = (&mut out[..128]).try_into().unwrap();
                (ctx.dct16x8)(src, dst);
                (2, 1)
            }
            STRATEGY_DCT8X16 => {
                gather(16, 8, &mut tmp[..128]);
                let src: &[f32; 128] = (&tmp[..128]).try_into().unwrap();
                let dst: &mut [f32; 128] = (&mut out[..128]).try_into().unwrap();
                (ctx.dct8x16)(src, dst);
                (2, 1)
            }
            STRATEGY_DCT16X16 => {
                gather(16, 16, &mut tmp[..256]);
                let src: &[f32; 256] = (&tmp[..256]).try_into().unwrap();
                let dst: &mut [f32; 256] = (&mut out[..256]).try_into().unwrap();
                (ctx.dct16x16)(src, dst);
                (2, 2)
            }
            STRATEGY_DCT32X32 => {
                gather(32, 32, &mut tmp[..1024]);
                let src: &[f32; 1024] = (&tmp[..1024]).try_into().unwrap();
                let dst: &mut [f32; 1024] = (&mut out[..1024]).try_into().unwrap();
                (ctx.dct32x32)(src, dst);
                (4, 4)
            }
            STRATEGY_DCT32X16 => {
                // 16 wide × 32 tall pixels (cov 2×4); normalized (cx,cy) = (4,2).
                gather(16, 32, &mut tmp[..512]);
                let src: &[f32; 512] = (&tmp[..512]).try_into().unwrap();
                let dst: &mut [f32; 512] = (&mut out[..512]).try_into().unwrap();
                (ctx.dct32x16)(src, dst);
                (4, 2)
            }
            STRATEGY_DCT16X32 => {
                // 32 wide × 16 tall pixels (cov 4×2); normalized (cx,cy) = (4,2).
                gather(32, 16, &mut tmp[..512]);
                let src: &[f32; 512] = (&tmp[..512]).try_into().unwrap();
                let dst: &mut [f32; 512] = (&mut out[..512]).try_into().unwrap();
                (ctx.dct16x32)(src, dst);
                (4, 2)
            }
            STRATEGY_DCT4X4 => {
                gather(8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct4x4)(src, dst);
                (1, 1)
            }
            STRATEGY_DCT4X8 => {
                gather(8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct4x8)(src, dst);
                (1, 1)
            }
            STRATEGY_DCT8X4 => {
                gather(8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct8x4)(src, dst);
                (1, 1)
            }
            _ => unreachable!("invalid strategy {strategy}"),
        }
    })
}

/// Quantize one channel exactly as the encoder will, accumulating the threshold-
/// aware squared quantization error (SSE, in quant-units²) and a rate estimate
/// (bits). LLF positions (`x < cx && y < cy`, coded via the DC plane) are
/// excluded from both, since DC coding is transform-choice-independent here.
fn channel_rd(
    sse_and_rate_fn: SseAndRateFn,
    rate_log2_lut: &RateLog2Lut,
    coeff: &[f32],
    inv_matrix: &[f32],
    channel: usize,
    qac: f32,
    qm_mult: f32,
    distance: f32,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let width = cx * 8;
    let height = cy * 8;
    let half = width / 2;
    let thr = crate::enc_group::quantize_ac_thresholds(channel, cx, cy, distance);
    let q_scaled = qac * qm_mult;

    let (sse, nzeros, mag_bits) = unsafe {
        sse_and_rate_fn(
            coeff,
            inv_matrix,
            q_scaled,
            width,
            height,
            half,
            cx,
            cy,
            rate_log2_lut,
            &thr,
        )
    };

    let header = R_HEADER * rate_log2_with_lut(rate_log2_lut, nzeros as f32);
    let bits = nzeros as f32 * R_NZ_BASE + R_MAG * mag_bits + header;
    (sse, bits)
}

pub(crate) type SseAndRateFn = unsafe fn(
    &[f32],
    &[f32],
    f32,
    usize,
    usize,
    usize,
    usize,
    usize,
    &RateLog2Lut,
    &[f32; 4],
) -> (f32, usize, f32);

fn select_sse_and_rate_fn() -> SseAndRateFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return crate::avx::sse_and_rate_avx2;
        }
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            return crate::sse::sse_and_rate_sse;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        crate::neon::sse_and_rate_neon
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::sse_and_rate_wasm
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        sse_and_rate_scalar
    }
}

static SSE_AND_RATE_FN: OnceLock<SseAndRateFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_sse_and_rate_fn() -> SseAndRateFn {
    *SSE_AND_RATE_FN.get_or_init(select_sse_and_rate_fn)
}

#[allow(unused)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_and_rate_scalar(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    rate_log2_lut: &RateLog2Lut,
    thr: &[f32; 4],
) -> (f32, usize, f32) {
    let mut sse = 0.0f32;
    let mut nzeros = 0usize;
    let mut mag_bits = 0.0f32;
    for y in 0..height {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];
        let row = y * width;
        for x in 0..width {
            if x < cx && y < cy {
                continue; // LLF → DC plane
            }
            let idx = row + x;
            let threshold = if x >= half { thr_hi } else { thr_lo };
            let a = inv_matrix[idx] * q_scaled * coeff[idx];
            let q = if a.abs() >= threshold {
                a.fast_round()
            } else {
                0.0
            };
            let d = a - q;
            sse += d * d;
            if q != 0.0 {
                nzeros += 1;
                mag_bits += rate_log2_with_lut(rate_log2_lut, q.abs());
            }
        }
    }
    (sse, nzeros, mag_bits)
}

/// Full RD cost `J = D + λR` of coding `strategy` at absolute pixel `(px, py)`.
/// Combines the three channels with the selection-time CfL approximation.
fn strategy_cost(
    ctx: &EncodingContext,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
) -> f32 {
    strategy_cost_impl(
        ctx,
        strategy,
        opsin,
        px,
        py,
        qac,
        qm_mult_x,
        matrices,
        meta_r,
        distance,
        cmap_factor,
        DistortionModel::Coefficient,
    )
}

/// Reconstruction-based RD cost used by the second-pass transform rerank.
#[allow(clippy::too_many_arguments)]
fn reconstruction_strategy_cost(
    ctx: &EncodingContext,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
) -> f32 {
    strategy_cost_impl(
        ctx,
        strategy,
        opsin,
        px,
        py,
        qac,
        qm_mult_x,
        matrices,
        meta_r,
        distance,
        cmap_factor,
        DistortionModel::Reconstruction,
    )
}

#[derive(Clone, Copy)]
enum DistortionModel {
    Coefficient,
    Reconstruction,
}

#[allow(clippy::too_many_arguments)]
fn strategy_cost_impl(
    ctx: &EncodingContext,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
    distortion_model: DistortionModel,
) -> f32 {
    let mut cxy = (1usize, 1usize);
    SC_COEFFS_SCRATCH.with(|cell| {
        let coeffs = &mut *cell.borrow_mut();
        for c in 0..3 {
            cxy = forward_transform(ctx, strategy, opsin.plane(c), px, py, &mut coeffs[c]);
        }
        let (cx, cy) = cxy;
        let size = cx * cy * 64;

        // Apply the same per-tile CfL slopes used by final coefficient coding.
        {
            let [c0, c1, c2] = coeffs;
            let y = &c1[..size];
            for ((x, b), &yi) in c0[..size]
                .iter_mut()
                .zip(c2[..size].iter_mut())
                .zip(y.iter())
            {
                *x -= cmap_factor[0] * yi;
                *b -= cmap_factor[2] * yi;
            }
        }

        let inv = |c: usize| -> &[f32] {
            match strategy {
                STRATEGY_DCT => &matrices.inv_matrix(c)[..],
                STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(c)[..],
                STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(c)[..],
                STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(c)[..],
                STRATEGY_DCT32X32 => &matrices.inv_matrix_32x32(c)[..],
                STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => &matrices.inv_matrix_32x16(c)[..],
                _ => &matrices.inv_matrix_16x8(c)[..],
            }
        };

        let (d_total, r_total) = match distortion_model {
            DistortionModel::Reconstruction => recon_dist_and_rate(
                ctx.rate_log2_lut,
                coeffs,
                [inv(0), inv(1), inv(2)],
                qac,
                qm_mult_x,
                cmap_factor[0],
                cmap_factor[2],
                distance,
                cx,
                cy,
                strategy,
                opsin,
                px,
                py,
            ),
            DistortionModel::Coefficient => {
                let mut d_total = 0.0f32;
                let mut r_total = 0.0f32;
                for c in 0..3 {
                    let qm_mult = if c == 0 { qm_mult_x } else { 1.0 };
                    let (d, r) = channel_rd(
                        ctx.sse_and_rate,
                        ctx.rate_log2_lut,
                        &coeffs[c][..size],
                        inv(c),
                        c,
                        qac,
                        qm_mult,
                        distance,
                        cx,
                        cy,
                    );
                    d_total += CHANNEL_WEIGHT[c] * d;
                    r_total += r;
                }
                (d_total, r_total)
            }
        };
        // `meta_r` prices the per-first-block AC-metadata rate (ACS + QF
        // tokens, ~2-4 bits each) the per-coefficient model can't see; charged
        // once per candidate block so merged tilings are credited for the
        // blocks they remove. Faded in above d=1 (detail-dense content prefers
        // the unbiased model at low distance); measured never below the RD
        // curve on photo/fractal/abstract sets, up to -7% bytes at d>=3.
        let lam = match distortion_model {
            DistortionModel::Coefficient => RD_LAMBDA,
            DistortionModel::Reconstruction => {
                let multiplier = 0.1 + (distance - 1.0).clamp(0.0, 2.0) / 2.0 * (3.0 - 0.1);
                RD_LAMBDA * multiplier
            }
        };
        d_total + lam * (r_total + meta_r)
    })
}

fn strategy_pixel_count(strategy: u8) -> usize {
    let (w, h) = strategy_pixel_dims(strategy);
    w * h
}

/// Feed one channel's pixel block through the strategy's forward transform.
fn forward_for(strategy: u8, input: &[f32], out: &mut [f32]) {
    use crate::dct;
    macro_rules! fwd {
        ($f:path, $n:literal) => {{
            let i: &[f32; $n] = input[..$n].try_into().unwrap();
            let o: &mut [f32; $n] = (&mut out[..$n]).try_into().unwrap();
            $f(i, o);
        }};
    }
    match strategy {
        STRATEGY_DCT4X4 => fwd!(dct::dct4x4, 64),
        STRATEGY_DCT4X8 => fwd!(dct::dct4x8, 64),
        STRATEGY_DCT8X4 => fwd!(dct::dct8x4, 64),
        STRATEGY_DCT16X8 => fwd!(dct::dct16x8, 128),
        STRATEGY_DCT8X16 => fwd!(dct::dct8x16, 128),
        STRATEGY_DCT16X16 => fwd!(dct::dct16x16, 256),
        STRATEGY_DCT32X32 => fwd!(dct::dct32x32, 1024),
        STRATEGY_DCT32X16 => fwd!(dct::dct32x16, 512),
        STRATEGY_DCT16X32 => fwd!(dct::dct16x32, 512),
        _ => fwd!(dct::dct8x8, 64),
    }
}

/// Transposed forward matrix `FT[n*N + k] = coeff k from pixel-impulse n`, so a
/// spatial reconstruction is `x[n] = N · Σ_k FT[n][k] · c[k]` (exact inverse of
/// the uniform-energy scaled DCT, `basis_energy = N`). Cached per strategy.
fn forward_matrix(strategy: u8) -> &'static [f32] {
    static M: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    &M.get_or_init(|| {
        (0u8..10)
            .map(|s| {
                let n = strategy_pixel_count(s);
                let mut ft = vec![0.0f32; n * n];
                let mut inp = [0.0f32; 1024];
                let mut out = [0.0f32; 1024];
                for imp in 0..n {
                    inp[..n].fill(0.0);
                    inp[imp] = 1.0;
                    forward_for(s, &inp, &mut out);
                    ft[imp * n..imp * n + n].copy_from_slice(&out[..n]);
                }
                ft
            })
            .collect()
    })[strategy as usize]
}

// SIMD IDCTs on AVX2/NEON, scalar butterfly fallback elsewhere.
macro_rules! idct_simd_or_scalar {
    ($name:ident, $n:literal, $avx:path, $neon:path, $scalar:path) => {
        #[inline]
        fn $name(c: &[f32; $n], o: &mut [f32; $n]) {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::is_x86_feature_detected!("avx2") {
                unsafe { $avx(c, o) };
                return;
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                unsafe { $neon(c, o) };
                return;
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            $scalar(c, o)
        }
    };
}
idct_simd_or_scalar!(
    idct8x8,
    64,
    crate::avx::inv_dct8x8_avx2,
    crate::neon::inv_dct8x8_neon,
    crate::dct::inv_dct8x8
);
idct_simd_or_scalar!(
    idct16x16,
    256,
    crate::avx::inv_dct16x16_avx2,
    crate::neon::inv_dct16x16_neon,
    crate::dct::inv_dct16x16
);
idct_simd_or_scalar!(
    idct32x32,
    1024,
    crate::avx::inv_dct32x32_avx2,
    crate::neon::inv_dct32x32_neon,
    crate::dct::inv_dct32x32
);
idct_simd_or_scalar!(
    idct8x16,
    128,
    crate::avx::inv_dct8x16_avx2,
    crate::neon::inv_dct8x16_neon,
    crate::dct::inv_dct8x16
);
idct_simd_or_scalar!(
    idct16x8,
    128,
    crate::avx::inv_dct16x8_avx2,
    crate::neon::inv_dct16x8_neon,
    crate::dct::inv_dct16x8
);
idct_simd_or_scalar!(
    idct16x32,
    512,
    crate::avx::inv_dct16x32_avx2,
    crate::neon::inv_dct16x32_neon,
    crate::dct::inv_dct16x32
);
idct_simd_or_scalar!(
    idct32x16,
    512,
    crate::avx::inv_dct32x16_avx2,
    crate::neon::inv_dct32x16_neon,
    crate::dct::inv_dct32x16
);

/// Reconstruct the spatial error (IDCT of the coefficient error). All standard
/// square and rectangular transforms use separable butterfly IDCTs, fully
/// vectorized on NEON. Only the special sub-8 layouts fall back to the dense
/// `N·FTᵀ·c` product.
fn reconstruct_error(strategy: u8, coeff_err: &[f32], err_out: &mut [f32]) {
    match strategy {
        STRATEGY_DCT => idct8x8(
            coeff_err[..64].try_into().unwrap(),
            (&mut err_out[..64]).try_into().unwrap(),
        ),
        STRATEGY_DCT16X16 => idct16x16(
            coeff_err[..256].try_into().unwrap(),
            (&mut err_out[..256]).try_into().unwrap(),
        ),
        STRATEGY_DCT32X32 => idct32x32(
            coeff_err[..1024].try_into().unwrap(),
            (&mut err_out[..1024]).try_into().unwrap(),
        ),
        STRATEGY_DCT8X16 => idct8x16(
            coeff_err[..128].try_into().unwrap(),
            (&mut err_out[..128]).try_into().unwrap(),
        ),
        STRATEGY_DCT16X8 => idct16x8(
            coeff_err[..128].try_into().unwrap(),
            (&mut err_out[..128]).try_into().unwrap(),
        ),
        STRATEGY_DCT16X32 => idct16x32(
            coeff_err[..512].try_into().unwrap(),
            (&mut err_out[..512]).try_into().unwrap(),
        ),
        STRATEGY_DCT32X16 => idct32x16(
            coeff_err[..512].try_into().unwrap(),
            (&mut err_out[..512]).try_into().unwrap(),
        ),
        _ => {
            let n = strategy_pixel_count(strategy);
            let ft = forward_matrix(strategy);
            let nf = n as f32;
            for (row, o) in err_out[..n].iter_mut().enumerate() {
                let base = row * n;
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += ft[base + k] * coeff_err[k];
                }
                *o = nf * acc;
            }
        }
    }
}

thread_local! {
    static RECON_SCRATCH: std::cell::RefCell<[[f32; 1024]; 4]> =
        const { std::cell::RefCell::new([[0.0; 1024]; 4]) };
}

#[allow(clippy::too_many_arguments)]
fn recon_dist_and_rate(
    rate_log2_lut: &RateLog2Lut,
    coeffs: &[[f32; 1024]; 3],
    inv: [&[f32]; 3],
    qac: f32,
    qm_mult_x: f32,
    factor_x: f32,
    factor_b: f32,
    distance: f32,
    cx: usize,
    cy: usize,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
) -> (f32, f32) {
    let n = strategy_pixel_count(strategy);
    let width = cx * 8;
    let height = cy * 8;
    let half = width / 2;
    let (pw, ph) = strategy_pixel_dims(strategy);
    let thr = [
        crate::enc_group::quantize_ac_thresholds(0, cx, cy, distance),
        crate::enc_group::quantize_ac_thresholds(1, cx, cy, distance),
        crate::enc_group::quantize_ac_thresholds(2, cx, cy, distance),
    ];
    let qs = [qac * qm_mult_x, qac, qac];

    RECON_SCRATCH.with_borrow_mut(|buf| {
        let (cerr_all, rest) = buf.split_at_mut(3);
        let serr = &mut rest[0];
        let mut r_total = 0.0f32;
        // Per channel: coefficient error field (LLF/DC excluded → 0) + rate.
        for c in 0..3 {
            let cerr = &mut cerr_all[c];
            let (mut nz, mut mag) = (0usize, 0.0f32);
            for y in 0..height {
                let yfix = if y >= height / 2 { 2 } else { 0 };
                for x in 0..width {
                    let idx = y * width + x;
                    if x < cx && y < cy {
                        cerr[idx] = 0.0; // LLF → DC plane, no AC error here
                        continue;
                    }
                    let t = if x >= half {
                        thr[c][yfix + 1]
                    } else {
                        thr[c][yfix]
                    };
                    let im = inv[c][idx];
                    let a = im * qs[c] * coeffs[c][idx];
                    let q = if a.abs() >= t { a.fast_round() } else { 0.0 };
                    cerr[idx] = (a - q) / (im * qs[c]); // dequant coeff error
                    if q != 0.0 {
                        nz += 1;
                        mag += rate_log2_with_lut(rate_log2_lut, q.abs());
                    }
                }
            }
            let header = R_HEADER * rate_log2_with_lut(rate_log2_lut, nz as f32);
            r_total += nz as f32 * R_NZ_BASE + R_MAG * mag + header;
        }

        // Spatial error per channel (reuse serr for each; accumulate D inline).
        // First reconstruct Y (needed for CfL), keep it.
        let mut yerr = [0.0f32; 1024];
        reconstruct_error(strategy, &cerr_all[1][..n], &mut yerr[..n]);

        let mut d_total = 0.0f32;
        let mut errbuf = [0.0f32; 1024];
        let mut recon = [0.0f32; 1024];
        let mut origbuf = [0.0f32; 1024];
        for c in 0..3 {
            // Per-channel decoder error (pixel space), decoder-order CfL folded in.
            let f = if c == 0 { factor_x } else { factor_b };
            let err: &[f32] = if c == 1 {
                &yerr[..n]
            } else {
                reconstruct_error(strategy, &cerr_all[c][..n], &mut serr[..n]);
                for idx in 0..n {
                    errbuf[idx] = serr[idx] + f * yerr[idx];
                }
                &errbuf[..n]
            };
            // recon = original − error, then windowed structural deficit.
            let plane = opsin.plane(c);
            let (cw, chh) = (plane.xsize(), plane.ysize());
            for iy in 0..ph {
                let row = plane.row((py + iy).min(chh - 1));
                for ix in 0..pw {
                    let i = iy * pw + ix;
                    let o = row[(px + ix).min(cw - 1)];
                    origbuf[i] = o;
                    recon[i] = o - err[i];
                }
            }
            d_total += CHANNEL_WEIGHT[c] * ssim_deficit(&origbuf[..n], &recon[..n], pw, ph);
        }
        (d_total, r_total)
    })
}

/// Sum of `(1 − SSIM)·window_pixels` over non-overlapping 8×8 windows — a
/// structural distortion comparable across transform sizes (a 32×32 candidate
/// has 16 windows, a tiled 8×8 has one each).
fn ssim_deficit(orig: &[f32], recon: &[f32], pw: usize, ph: usize) -> f32 {
    const C1: f32 = 1e-4;
    const C2: f32 = 9e-4;
    let (wx, wy) = (pw / 8, ph / 8);
    let mut d = 0.0f32;
    for wyi in 0..wy {
        for wxi in 0..wx {
            let (x0, y0) = (wxi * 8, wyi * 8);
            let (mut so, mut sr, mut soo, mut srr, mut sor) = (0.0f32, 0.0, 0.0, 0.0, 0.0);
            for yy in 0..8 {
                for xx in 0..8 {
                    let i = (y0 + yy) * pw + (x0 + xx);
                    let (o, r) = (orig[i], recon[i]);
                    so += o;
                    sr += r;
                    soo += o * o;
                    srr += r * r;
                    sor += o * r;
                }
            }
            let (mo, mr) = (so / 64.0, sr / 64.0);
            let vo = soo / 64.0 - mo * mo;
            let vr = srr / 64.0 - mr * mr;
            let cov = sor / 64.0 - mo * mr;
            let ssim = ((2.0 * mo * mr + C1) * (2.0 * cov + C2))
                / ((mo * mo + mr * mr + C1) * (vo + vr + C2));
            d += (1.0 - ssim) * 64.0;
        }
    }
    d
}

/// Pixel footprint (width, height) each strategy gathers/transforms.
fn strategy_pixel_dims(strategy: u8) -> (usize, usize) {
    match strategy {
        STRATEGY_DCT16X8 => (8, 16),
        STRATEGY_DCT8X16 => (16, 8),
        STRATEGY_DCT16X16 => (16, 16),
        STRATEGY_DCT32X16 => (16, 32),
        STRATEGY_DCT16X32 => (32, 16),
        STRATEGY_DCT32X32 => (32, 32),
        _ => (8, 8),
    }
}

#[inline]
fn cmap_factors(ytox_map: &ImageSB, ytob_map: &ImageSB, bx: usize, by: usize) -> [f32; 3] {
    let tx = (bx / 8).min(ytox_map.xsize() - 1);
    let ty = (by / 8).min(ytox_map.ysize() - 1);
    [
        crate::enc_color_correlation::y_to_x_ratio(ytox_map.row(ty)[tx]),
        0.0,
        crate::enc_color_correlation::y_to_b_ratio(ytob_map.row(ty)[tx]),
    ]
}

#[allow(clippy::too_many_arguments)]
fn select_super_block(
    ctx: &EncodingContext,
    meta_r: f32,
    distance: f32,
    opsin: &Image3F,
    bx0: usize,
    by0: usize,
    px0: usize,
    py0: usize,
    qac: [[f32; 2]; 2],
    qac_scale: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
) -> SuperBlockCost {
    let cmap_factor = cmap_factors(ytox_map, ytob_map, bx0, by0);

    // Cost of the four individual DCT8 blocks: cost[dy][dx]. DCT8 is the
    // incumbent; every merge below must beat the corresponding tiled cost by a
    // transform-size-dependent safety margin rather than by an arbitrarily tiny
    // model delta.
    let mut c8 = [[0.0f32; 2]; 2];
    for dy in 0..2 {
        for dx in 0..2 {
            c8[dy][dx] = strategy_cost(
                ctx,
                STRATEGY_DCT,
                opsin,
                px0 + dx * 8,
                py0 + dy * 8,
                qac[dy][dx],
                qm_mult_x,
                matrices,
                meta_r,
                distance,
                cmap_factor,
            );
        }
    }

    // Vertical pairs (DCT16X8): one per column.
    let v_left = BIAS_RECT
        * strategy_cost(
            ctx,
            STRATEGY_DCT16X8,
            opsin,
            px0,
            py0,
            qac[0][0].max(qac[1][0]),
            qm_mult_x,
            matrices,
            meta_r,
            distance,
            cmap_factor,
        );
    let v_right = BIAS_RECT
        * strategy_cost(
            ctx,
            STRATEGY_DCT16X8,
            opsin,
            px0 + 8,
            py0,
            qac[0][1].max(qac[1][1]),
            qm_mult_x,
            matrices,
            meta_r,
            distance,
            cmap_factor,
        );

    // Horizontal pairs (DCT8X16): one per row.
    let h_top = BIAS_RECT
        * strategy_cost(
            ctx,
            STRATEGY_DCT8X16,
            opsin,
            px0,
            py0,
            qac[0][0].max(qac[0][1]),
            qm_mult_x,
            matrices,
            meta_r,
            distance,
            cmap_factor,
        );
    let h_bot = BIAS_RECT
        * strategy_cost(
            ctx,
            STRATEGY_DCT8X16,
            opsin,
            px0,
            py0 + 8,
            qac[1][0].max(qac[1][1]),
            qm_mult_x,
            matrices,
            meta_r,
            distance,
            cmap_factor,
        );

    // The single DCT16X16 over all four.
    let c16 = BIAS_16X16
        * strategy_cost(
            ctx,
            STRATEGY_DCT16X16,
            opsin,
            px0,
            py0,
            aggregate_qac_2x2(qac, qac_scale, distance),
            qm_mult_x,
            matrices,
            meta_r,
            distance,
            cmap_factor,
        );

    let dct8_left = c8[0][0] + c8[1][0];
    let dct8_right = c8[0][1] + c8[1][1];
    let dct8_top = c8[0][0] + c8[0][1];
    let dct8_bottom = c8[1][0] + c8[1][1];
    let total_dct8 = dct8_left + dct8_right;

    let use_v_left = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X8)
        && merge_beats_dct8(v_left, dct8_left, distance, MERGE_MARGIN_PAIR_HQ);
    let use_v_right = ac_strategy.can_place_strategy(bx0 + 1, by0, STRATEGY_DCT16X8)
        && merge_beats_dct8(v_right, dct8_right, distance, MERGE_MARGIN_PAIR_HQ);
    let use_h_top = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_top, dct8_top, distance, MERGE_MARGIN_PAIR_HQ);
    let use_h_bottom = ac_strategy.can_place_strategy(bx0, by0 + 1, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_bot, dct8_bottom, distance, MERGE_MARGIN_PAIR_HQ);

    let cost_16x8 = if use_v_left { v_left } else { dct8_left }
        + if use_v_right { v_right } else { dct8_right };
    let cost_8x16 =
        if use_h_top { h_top } else { dct8_top } + if use_h_bottom { h_bot } else { dct8_bottom };
    let best_rect = cost_16x8.min(cost_8x16);

    let pick_16x16 = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X16)
        && c16 < best_rect
        && merge_beats_dct8(c16, total_dct8, distance, MERGE_MARGIN_16_HQ);

    let chosen = if pick_16x16 {
        ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X16);
        c16
    } else if cost_16x8 <= cost_8x16 {
        if use_v_left {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X8);
        }
        if use_v_right {
            ac_strategy.set_first(bx0 + 1, by0, STRATEGY_DCT16X8);
        }
        cost_16x8
    } else {
        if use_h_top {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT8X16);
        }
        if use_h_bottom {
            ac_strategy.set_first(bx0, by0 + 1, STRATEGY_DCT8X16);
        }
        cost_8x16
    };

    SuperBlockCost {
        chosen,
        dct8: total_dct8,
    }
}

/// For each multi-block transform, aggregate `raw_quant` across the covered
/// blocks so the field is consistent within a transform (libjxl-tiny
/// `AdjustQuantField`).
#[inline]
fn mean_max_mixer(butteraugli_target: f32) -> f32 {
    const K_LIMIT: f32 = 1.54138;
    const K_MUL: f32 = 0.56391;
    (1.0 - (butteraugli_target - K_LIMIT).max(0.0) * K_MUL).max(0.0)
}

#[inline]
fn aggregate_quant(max_q: u8, sum: u32, covered: usize, butteraugli_target: f32) -> u8 {
    if covered < 4 {
        return max_q;
    }
    let mixer = mean_max_mixer(butteraugli_target);
    let mean = sum as f32 / covered as f32;
    let mixed = max_q as f32 * mixer + mean * (1.0 - mixer);
    (mixed + 0.5).clamp(1.0, 255.0) as u8
}

#[inline]
fn aggregate_qac_2x2(qac: [[f32; 2]; 2], scale: f32, butteraugli_target: f32) -> f32 {
    let max_q = qac[0][0].max(qac[0][1]).max(qac[1][0]).max(qac[1][1]);
    let mixer = mean_max_mixer(butteraugli_target);
    let mean = (qac[0][0] + qac[0][1] + qac[1][0] + qac[1][1]) * 0.25;
    let mixed_q = (max_q * mixer + mean * (1.0 - mixer)) / scale;
    scale * (mixed_q + 0.5).clamp(1.0, 255.0).floor()
}

pub(crate) fn adjust_quant_field(
    ac_strategy: &AcStrategyImage,
    butteraugli_target: f32,
    quant_field: &mut ImageB,
) {
    for (x, y, raw_strategy) in ac_strategy.iter_first_blocks() {
        let cov_x = AcStrategyImage::covered_blocks_x_of(raw_strategy);
        let cov_y = AcStrategyImage::covered_blocks_y_of(raw_strategy);
        if cov_x == 1 && cov_y == 1 {
            continue;
        }
        let mut max_q: u8 = 0;
        let mut sum: u32 = 0;
        for iy in 0..cov_y {
            for &q in &quant_field.row(y + iy)[x..x + cov_x] {
                max_q = max_q.max(q);
                sum += q as u32;
            }
        }
        let val = aggregate_quant(max_q, sum, cov_x * cov_y, butteraugli_target);
        for iy in 0..cov_y {
            for q in &mut quant_field.row_mut(y + iy)[x..x + cov_x] {
                *q = val;
            }
        }
    }
}

/// Select transforms for every aligned 2×2 super-block in the DC group, then
/// reconcile the quant field. `(dc_group_px, dc_group_py)` is the DC group's
/// top-left in absolute image pixels (so `opsin` can be the full image).
#[allow(clippy::too_many_arguments)]
/// Aggregate quant over a transform footprint exactly as `adjust_quant_field`
/// will after strategy selection, then scale it for the RD model.
#[inline]
fn region_qac(
    quant_field: &ImageB,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    scale: f32,
    butteraugli_target: f32,
) -> f32 {
    let mut max_q: u8 = 1;
    let mut sum = 0u32;
    for iy in 0..h {
        for ix in 0..w {
            let q = quant_field.row(by + iy)[bx + ix];
            max_q = max_q.max(q);
            sum += q as u32;
        }
    }
    scale * aggregate_quant(max_q, sum, w * h, butteraugli_target) as f32
}

#[inline]
fn block_qac_2x2(quant_field: &ImageB, bx: usize, by: usize, scale: f32) -> [[f32; 2]; 2] {
    [
        [
            scale * quant_field.row(by)[bx] as f32,
            scale * quant_field.row(by)[bx + 1] as f32,
        ],
        [
            scale * quant_field.row(by + 1)[bx] as f32,
            scale * quant_field.row(by + 1)[bx + 1] as f32,
        ],
    ]
}

/// Run AC-strategy selection for block rows `by` in `[y_begin, y_end)` into
/// `ac_strategy`, returning the accumulated sub-8x8 RD benefit for those rows.
/// `ysize`/`xsize` are the *full* group dimensions: edge tests (`four_row`,
/// loop bounds) use the global size, exactly as the serial loop would, so a
/// 4-aligned `[y_begin, y_end)` partition reproduces the single-threaded
/// decision sequence bit-for-bit. Reads `quant_field`/`opsin` only.
#[allow(clippy::too_many_arguments)]
fn select_band(
    ctx: &EncodingContext,
    meta_r: f32,
    distance: f32,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    scale: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    quant_field: &ImageB,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
    xsize: usize,
    ysize: usize,
    y_begin: usize,
    y_end: usize,
) -> f32 {
    let mut by = y_begin;
    while by + 1 < ysize && by < y_end {
        // A 4-block-tall band can host DCT32X32 only when 4-aligned and fitting.
        let four_row = by.is_multiple_of(4) && by + 4 <= ysize;
        let mut bx = 0;
        while bx + 1 < xsize {
            let four_col = bx % 4 == 0 && bx + 4 <= xsize;
            if four_row && four_col && ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT32X32) {
                let mut sub_total = 0.0f32;
                let mut dct8_total = 0.0f32;
                for sy in 0..2 {
                    for sx in 0..2 {
                        let sbx = bx + sx * 2;
                        let sby = by + sy * 2;
                        let qac = block_qac_2x2(quant_field, sbx, sby, scale);
                        let costs = select_super_block(
                            ctx,
                            meta_r,
                            distance,
                            opsin,
                            sbx,
                            sby,
                            dc_group_px + sbx * 8,
                            dc_group_py + sby * 8,
                            qac,
                            scale,
                            qm_mult_x,
                            matrices,
                            ytox_map,
                            ytob_map,
                            ac_strategy,
                        );
                        sub_total += costs.chosen;
                        dct8_total += costs.dct8;
                    }
                }
                let qac32 = region_qac(quant_field, bx, by, 4, 4, scale, distance);
                let cmap_factor = cmap_factors(ytox_map, ytob_map, bx, by);
                let cost32 = strategy_cost(
                    ctx,
                    STRATEGY_DCT32X32,
                    opsin,
                    dc_group_px + bx * 8,
                    dc_group_py + by * 8,
                    qac32,
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                // Two DCT32X16 (each 2 wide × 4 tall) tiling the region: left + right.
                let cl = strategy_cost(
                    ctx,
                    STRATEGY_DCT32X16,
                    opsin,
                    dc_group_px + bx * 8,
                    dc_group_py + by * 8,
                    region_qac(quant_field, bx, by, 2, 4, scale, distance),
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                let cr = strategy_cost(
                    ctx,
                    STRATEGY_DCT32X16,
                    opsin,
                    dc_group_px + (bx + 2) * 8,
                    dc_group_py + by * 8,
                    region_qac(quant_field, bx + 2, by, 2, 4, scale, distance),
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                // Two DCT16X32 (each 4 wide × 2 tall) tiling the region: top + bottom.
                let ct = strategy_cost(
                    ctx,
                    STRATEGY_DCT16X32,
                    opsin,
                    dc_group_px + bx * 8,
                    dc_group_py + by * 8,
                    region_qac(quant_field, bx, by, 4, 2, scale, distance),
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                let cb = strategy_cost(
                    ctx,
                    STRATEGY_DCT16X32,
                    opsin,
                    dc_group_px + bx * 8,
                    dc_group_py + (by + 2) * 8,
                    region_qac(quant_field, bx, by + 2, 4, 2, scale, distance),
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                let can_32x32 = ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT32X32);
                let can_32x16 = ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT32X16)
                    && ac_strategy.can_place_strategy(bx + 2, by, STRATEGY_DCT32X16);
                let can_16x32 = ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT16X32)
                    && ac_strategy.can_place_strategy(bx, by + 2, STRATEGY_DCT16X32);

                let cost_32x32 = if can_32x32 {
                    BIAS_32X32 * cost32
                } else {
                    f32::INFINITY
                };
                let cost_32x16 = if can_32x16 {
                    BIAS_RECT32 * (cl + cr)
                } else {
                    f32::INFINITY
                };
                let cost_16x32 = if can_16x32 {
                    BIAS_RECT32 * (ct + cb)
                } else {
                    f32::INFINITY
                };

                let (best_big, best_strategy, margin) =
                    if cost_32x32 <= cost_32x16 && cost_32x32 <= cost_16x32 {
                        (cost_32x32, STRATEGY_DCT32X32, MERGE_MARGIN_32_HQ)
                    } else if cost_32x16 <= cost_16x32 {
                        (cost_32x16, STRATEGY_DCT32X16, MERGE_MARGIN_32_RECT_HQ)
                    } else {
                        (cost_16x32, STRATEGY_DCT16X32, MERGE_MARGIN_32_RECT_HQ)
                    };

                // Compare against both the already-selected subdivision and the
                // pure DCT8 incumbent. The latter prevents a sequence of locally
                // marginal merges from making a 32×32 merge look trustworthy.
                if best_big < sub_total && merge_beats_dct8(best_big, dct8_total, distance, margin)
                {
                    match best_strategy {
                        STRATEGY_DCT32X32 => {
                            ac_strategy.set_first(bx, by, STRATEGY_DCT32X32);
                        }
                        STRATEGY_DCT32X16 => {
                            ac_strategy.set_first(bx, by, STRATEGY_DCT32X16);
                            ac_strategy.set_first(bx + 2, by, STRATEGY_DCT32X16);
                        }
                        STRATEGY_DCT16X32 => {
                            ac_strategy.set_first(bx, by, STRATEGY_DCT16X32);
                            ac_strategy.set_first(bx, by + 2, STRATEGY_DCT16X32);
                        }
                        _ => unreachable!(),
                    }
                }
                bx += 4;
            } else if four_row {
                for sby in [by, by + 2] {
                    let qac = block_qac_2x2(quant_field, bx, sby, scale);
                    let _ = select_super_block(
                        ctx,
                        meta_r,
                        distance,
                        opsin,
                        bx,
                        sby,
                        dc_group_px + bx * 8,
                        dc_group_py + sby * 8,
                        qac,
                        scale,
                        qm_mult_x,
                        matrices,
                        ytox_map,
                        ytob_map,
                        ac_strategy,
                    );
                }
                bx += 2;
            } else {
                let qac = block_qac_2x2(quant_field, bx, by, scale);
                let _ = select_super_block(
                    ctx,
                    meta_r,
                    distance,
                    opsin,
                    bx,
                    by,
                    dc_group_px + bx * 8,
                    dc_group_py + by * 8,
                    qac,
                    scale,
                    qm_mult_x,
                    matrices,
                    ytox_map,
                    ytob_map,
                    ac_strategy,
                );
                bx += 2;
            }
        }
        by += if four_row { 4 } else { 2 };
    }

    // Sub-8x8 refinement for this row band.
    let mut benefit = 0.0f32;
    for by in y_begin..y_end {
        for bx in 0..xsize {
            if ac_strategy.raw_strategy(bx, by) != STRATEGY_DCT {
                continue;
            }
            let qac = region_qac(quant_field, bx, by, 1, 1, scale, distance);
            let px = dc_group_px + bx * 8;
            let py = dc_group_py + by * 8;
            let cmap_factor = cmap_factors(ytox_map, ytob_map, bx, by);
            let cost8 = strategy_cost(
                ctx,
                STRATEGY_DCT,
                opsin,
                px,
                py,
                qac,
                qm_mult_x,
                matrices,
                meta_r,
                distance,
                cmap_factor,
            );
            let cost4 = BIAS_4X4
                * strategy_cost(
                    ctx,
                    STRATEGY_DCT4X4,
                    opsin,
                    px,
                    py,
                    qac,
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
            let cost48 = BIAS_4X8
                * strategy_cost(
                    ctx,
                    STRATEGY_DCT4X8,
                    opsin,
                    px,
                    py,
                    qac,
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
            let cost84 = BIAS_4X8
                * strategy_cost(
                    ctx,
                    STRATEGY_DCT8X4,
                    opsin,
                    px,
                    py,
                    qac,
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factor,
                );
            // Choose the cheapest sub-8×8 candidate, and take it only if it beats
            // the 8×8 incumbent. DCT4X8 (fine vertical res) and DCT8X4 (fine
            // horizontal res) are transposes that suit opposite edge orientations.
            let (cand, cand_cost) = {
                let mut best = STRATEGY_DCT4X4;
                let mut bc = cost4;
                if cost48 < bc {
                    best = STRATEGY_DCT4X8;
                    bc = cost48;
                }
                if cost84 < bc {
                    best = STRATEGY_DCT8X4;
                    bc = cost84;
                }
                (best, bc)
            };
            if cand_cost < cost8 {
                ac_strategy.set_first(bx, by, cand);
                benefit += cost8 - cand_cost;
            }
        }
    }
    benefit
}

/// Partition `[0, ysize)` into at most `n` contiguous bands whose interior
/// boundaries are multiples of 4 (so DCT32X32's 4-block super-rows never span a
/// boundary). The serial loop only ever takes non-4 (`+2`) steps at the image
/// bottom, which lands wholly inside the final band — hence the partition
/// reproduces the single-threaded `by` sequence exactly.
fn selection_bands(ysize: usize, n: usize) -> Vec<(usize, usize)> {
    let mut bounds = vec![0usize];
    for k in 1..n {
        let b = (ysize * k / n) / 4 * 4;
        if b > *bounds.last().unwrap() && b < ysize {
            bounds.push(b);
        }
    }
    bounds.push(ysize);
    bounds.windows(2).map(|w| (w[0], w[1])).collect()
}

pub(crate) fn fill_ac_strategy(
    ctx: &EncodingContext,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    distance: f32,
    scale: f32,
    x_qm_scale: u32,
    matrices: &DequantMatrices,
    quant_field: &mut ImageB,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
    num_threads: usize,
) -> f32 {
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    // DCT8 wins the high-quality RD comparison.
    if use_dct8_only(distance) {
        return 0.0;
    }
    let qm_mult_x = 1.25f32.powf(x_qm_scale as f32 - 2.0);
    // Per-candidate-block metadata rate for the strategy chooser (bits),
    // faded in above d=1 (see strategy_cost).
    let meta_r = 2.0f32 * (distance - 1.0).clamp(0.0, 1.0);

    let bands = if num_threads > 1 && ysize >= 8 {
        selection_bands(ysize, num_threads)
    } else {
        vec![(0, ysize)]
    };

    let benefit = if bands.len() <= 1 {
        select_band(
            ctx,
            meta_r,
            distance,
            opsin,
            dc_group_px,
            dc_group_py,
            scale,
            qm_mult_x,
            matrices,
            quant_field,
            ytox_map,
            ytob_map,
            ac_strategy,
            xsize,
            ysize,
            0,
            ysize,
        )
    } else {
        // Each band selects into its own fresh (default) strategy image, reading
        // the shared opsin/quant_field; results merge deterministically by row.
        let qf: &ImageB = quant_field;
        let bands_ref = &bands;
        let results = crate::thread_pool::steal_map(bands.len(), num_threads, |i| {
            let (y0, y1) = bands_ref[i];
            let mut local = AcStrategyImage::new(xsize, ysize);
            let b = select_band(
                ctx,
                meta_r,
                distance,
                opsin,
                dc_group_px,
                dc_group_py,
                scale,
                qm_mult_x,
                matrices,
                qf,
                ytox_map,
                ytob_map,
                &mut local,
                xsize,
                ysize,
                y0,
                y1,
            );
            (local, b)
        });
        let mut benefit = 0.0f32;
        for (&(y0, y1), (local, b)) in bands.iter().zip(results.iter()) {
            ac_strategy.copy_rows_from(local, y0, y1);
            benefit += b;
        }
        benefit
    };

    // Second pass — reconstruction-based rerank. The fast selector over-merges at
    // high quality; here we revisit only the *selected* large transforms and
    // downgrade a merge to its tiled DCT8 when the SSIM-reconstruction RD cost
    // prefers it. Only large transforms are scored (a fraction of blocks), so the
    // expensive recon distortion runs on far fewer candidates than a full recon
    // selection while capturing the same structural win.
    rerank_large_transforms(
        ctx,
        opsin,
        dc_group_px,
        dc_group_py,
        distance,
        scale,
        qm_mult_x,
        meta_r,
        matrices,
        quant_field,
        ytox_map,
        ytob_map,
        ac_strategy,
    );

    adjust_quant_field(ac_strategy, distance, quant_field);
    benefit
}

/// Reconstruction-based rerank pass: for each selected merge, compare its
/// SSIM-reconstruction cost against the tiled DCT8 and downgrade if DCT8 wins.
#[allow(clippy::too_many_arguments)]
fn rerank_large_transforms(
    ctx: &EncodingContext,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    distance: f32,
    scale: f32,
    qm_mult_x: f32,
    meta_r: f32,
    matrices: &DequantMatrices,
    quant_field: &ImageB,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
) {
    let mut downgrades: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (bx, by, strat) in ac_strategy.iter_first_blocks() {
        let cxb = AcStrategyImage::covered_blocks_x_of(strat);
        let cyb = AcStrategyImage::covered_blocks_y_of(strat);
        if cxb * cyb <= 1 {
            continue; // only merges
        }
        let (px, py) = (dc_group_px + bx * 8, dc_group_py + by * 8);
        let qac_big = region_qac(quant_field, bx, by, cxb, cyb, scale, distance);
        let j_big = reconstruction_strategy_cost(
            ctx,
            strat,
            opsin,
            px,
            py,
            qac_big,
            qm_mult_x,
            matrices,
            meta_r,
            distance,
            cmap_factors(ytox_map, ytob_map, bx, by),
        );
        let mut j_dct8 = 0.0f32;
        for iy in 0..cyb {
            for ix in 0..cxb {
                let q = region_qac(quant_field, bx + ix, by + iy, 1, 1, scale, distance);
                j_dct8 += reconstruction_strategy_cost(
                    ctx,
                    STRATEGY_DCT,
                    opsin,
                    px + ix * 8,
                    py + iy * 8,
                    q,
                    qm_mult_x,
                    matrices,
                    meta_r,
                    distance,
                    cmap_factors(ytox_map, ytob_map, bx + ix, by + iy),
                );
            }
        }
        if j_dct8 < j_big {
            downgrades.push((bx, by, cxb, cyb));
        }
    }
    for (bx, by, cxb, cyb) in downgrades {
        for iy in 0..cyb {
            for ix in 0..cxb {
                ac_strategy.set_first(bx + ix, by + iy, STRATEGY_DCT);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DCT8_ONLY_MAX_DISTANCE, MERGE_MARGIN_16_HQ, MERGE_MARGIN_32_HQ, MERGE_MARGIN_PAIR_HQ,
        aggregate_qac_2x2, aggregate_quant, cmap_factors, forward_for, forward_matrix,
        merge_beats_dct8, merge_margin, reconstruct_error, strategy_pixel_count, use_dct8_only,
    };
    use crate::dc_group_data::{
        STRATEGY_DCT, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT32X32,
    };
    use crate::image::ImageSB;

    #[test]
    fn high_quality_dct8_cutoff_is_half_distance() {
        assert_eq!(DCT8_ONLY_MAX_DISTANCE, 0.5);
        assert!(use_dct8_only(0.2));
        assert!(use_dct8_only(0.3));
        assert!(use_dct8_only(0.5));
        assert!(!use_dct8_only(0.500_001));
        assert!(!use_dct8_only(1.0));
    }

    #[test]
    fn reconstruction_round_trips() {
        // x = N·Fᵀ·(F·x) must return the original block (exact inverse).
        // DCT4X4/4X8/8X4 use a sub-DC Hadamard (non-orthogonal), so the
        // `x=N·Fᵀc` inverse doesn't apply — but they are not merge candidates.
        for strategy in [
            STRATEGY_DCT,
            STRATEGY_DCT16X8,
            STRATEGY_DCT16X16,
            STRATEGY_DCT32X32,
        ] {
            let n = strategy_pixel_count(strategy);
            // deterministic pseudo-random input
            let mut x = [0.0f32; 1024];
            let mut s = 12345u32;
            for v in x[..n].iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            }
            let mut c = [0.0f32; 1024];
            forward_for(strategy, &x, &mut c);
            let mut recon = [0.0f32; 1024];
            reconstruct_error(strategy, &c[..n], &mut recon[..n]);
            let max_err = (0..n).map(|i| (recon[i] - x[i]).abs()).fold(0.0, f32::max);
            assert!(
                max_err < 1e-3,
                "strategy {strategy}: max reconstruction err {max_err}"
            );
        }
    }

    #[test]
    fn forward_matrix_is_orthogonal() {
        // Check F·Fᵀ off-diagonals are ~0 for DCT8 (rows must be orthogonal for
        // x = N·Fᵀc to be a valid inverse).
        let n = 64;
        let ft = forward_matrix(STRATEGY_DCT); // ft[pixel*n + coeff] = F[coeff, pixel]
        // dot of coeff rows j,k = Σ_pixel F[j,pixel]F[k,pixel] = Σ_p ft[p*n+j]ft[p*n+k]
        let mut max_off = 0.0f32;
        for j in 0..8 {
            for k in 0..8 {
                if j == k {
                    continue;
                }
                let dot: f32 = (0..n).map(|p| ft[p * n + j] * ft[p * n + k]).sum();
                max_off = max_off.max(dot.abs());
            }
        }
        println!("max off-diagonal |<f_j,f_k>| = {max_off}");
        assert!(max_off < 1e-4, "DCT8 basis not orthogonal: {max_off}");
    }

    #[test]
    fn merge_guard_is_stricter_for_large_transforms_and_high_quality() {
        assert!(merge_margin(0.3, MERGE_MARGIN_32_HQ) > merge_margin(0.3, MERGE_MARGIN_16_HQ));
        assert!(merge_margin(0.3, MERGE_MARGIN_16_HQ) > merge_margin(0.3, MERGE_MARGIN_PAIR_HQ));
        assert!(merge_margin(0.3, MERGE_MARGIN_16_HQ) > merge_margin(4.0, MERGE_MARGIN_16_HQ));

        // At high quality an 8% 16x16 margin rejects a 5% estimated win, but
        // accepts a clear 10% win. At coarse quality the guard fades.
        assert!(!merge_beats_dct8(95.0, 100.0, 0.3, MERGE_MARGIN_16_HQ));
        assert!(merge_beats_dct8(89.0, 100.0, 0.3, MERGE_MARGIN_16_HQ));
        assert!(merge_beats_dct8(95.0, 100.0, 4.0, MERGE_MARGIN_16_HQ));
    }

    #[test]
    fn quant_aggregation_uses_max_for_pairs_and_high_quality() {
        assert_eq!(aggregate_quant(40, 50, 2, 1.0), 40);
        assert_eq!(aggregate_quant(40, 100, 4, 1.0), 40);
    }

    #[test]
    fn quant_aggregation_matches_scaled_candidate_cost() {
        let qac = [[2.5, 5.0], [7.5, 10.0]];
        let raw = aggregate_quant(40, 100, 4, 2.0);
        assert_eq!(aggregate_qac_2x2(qac, 0.25, 2.0), raw as f32 * 0.25);
    }

    #[test]
    fn strategy_cfl_factors_match_signalled_tile_maps() {
        let ytox = ImageSB::new_fill(1, 1, 42);
        let ytob = ImageSB::new_fill(1, 1, -42);
        assert_eq!(cmap_factors(&ytox, &ytob, 0, 0), [0.5, 0.0, 0.5]);
    }
}
