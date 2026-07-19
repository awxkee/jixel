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

use crate::dc_group_data::{
    AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4,
    STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16,
    STRATEGY_DCT32X32, STRATEGY_DCT32X64, STRATEGY_DCT64X32, STRATEGY_DCT64X64,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
use crate::dct::fmla;
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, ImageB, ImageSB};
use crate::inflated_cost::{CHANNEL_WEIGHT, channel_rd, recon_dist_and_rate};
use crate::quant_weights::DequantMatrices;
use std::cell::RefCell;

/// At visually lossless / near-lossless settings, transform metadata and the
/// occasional wrong merge cost more than the large-DCT decorrelation saves.
const DCT8_ONLY_MAX_DISTANCE: f32 = 0.5;

/// Above this distance the [`SearchScope::Squares`] tier stops reranking; see
/// [`SearchScope::rerank`].
const FAST_RERANK_MAX_DISTANCE: f32 = 2.0;

#[inline]
fn use_dct8_only(distance: f32) -> bool {
    distance <= DCT8_ONLY_MAX_DISTANCE
}

#[inline]
fn use_dct64(speed: crate::Speed, distance: f32) -> bool {
    speed == crate::Speed::Slow && distance >= DCT64_MIN_DISTANCE
}

#[inline]
fn use_dct64_rect(speed: crate::Speed, distance: f32) -> bool {
    speed == crate::Speed::Slow && distance >= DCT64_RECT_MIN_DISTANCE
}

/// High-rate-optimal Lagrange multiplier for unit-step (Δ = 1) scalar
/// quantization: `λ* = Δ²·ln2 / 6`. Distortion is in quant-units², rate in
/// bits, so `λ·R` is in quant-units² and adds cleanly to D.
pub(crate) const RD_LAMBDA: f32 = 0.080_867_17;

const BIAS_RECT: f32 = 1.0;
const BIAS_16X16: f32 = 1.0;
const BIAS_32X32: f32 = 1.0;
const BIAS_RECT32: f32 = 1.10;

const DCT64_MIN_DISTANCE: f32 = 3.0;
const DCT64_RECT_MIN_DISTANCE: f32 = 2.5;

const BIAS_64X64: f32 = 1.10;
const BIAS_64_RECT: f32 = 1.10;

const MERGE_MARGIN_PAIR_HQ: f32 = 0.04;
const MERGE_MARGIN_16_HQ: f32 = 0.08;
const MERGE_MARGIN_32_RECT_HQ: f32 = 0.11;
const MERGE_MARGIN_32_HQ: f32 = 0.14;
const MERGE_MARGIN_64_HQ: f32 = 0.16;

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

/// How much of the transform space the chooser explores.
///
/// Both scopes run the same coefficient-domain RD model; they differ only in
/// which candidates are offered. `Squares` is the [`crate::Speed::Fast`] tier:
/// DCT16X16 and DCT32X32 compete against their tiled DCT8 incumbent, and the
/// rectangular merges, the sub-8x8 refinement and the 64px pass are skipped.
/// The SSIM reconstruction rerank still runs — it is what keeps the chooser
/// from over-merging at high quality, where the coefficient-domain model
/// systematically prefers a merge that SSIMULACRA2 does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SearchScope {
    Squares,
    Full,
}

impl SearchScope {
    #[inline]
    fn for_speed(speed: crate::Speed) -> Self {
        match speed {
            crate::Speed::Fast => SearchScope::Squares,
            crate::Speed::Slow => SearchScope::Full,
        }
    }

    #[inline]
    fn rectangles(self) -> bool {
        self == SearchScope::Full
    }

    /// Whether the SSIM reconstruction rerank runs.
    ///
    /// `Full` always reranks. `Squares` only reranks at high quality: measured
    /// on 42 images, the rerank's own marginal value inside the Fast tier is
    /// -0.69% at d=1.0, -0.30% at d=1.5 and -0.06% at d=2.0, then flips
    /// positive (+0.1..+0.17% at d=3.5-4.5) with the win rate falling to
    /// ~18/42. It is also the tier's most expensive stage, so paying for it
    /// past the crossover would cost time to lose rate.
    #[inline]
    fn rerank(self, distance: f32) -> bool {
        self == SearchScope::Full || distance <= FAST_RERANK_MAX_DISTANCE
    }
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
    static FT_GATHER_SCRATCH: RefCell<[f32; 1024]> =
        const { RefCell::new([0.0; 1024]) };
    /// Reused per-channel coefficient scratch for [`strategy_cost`].
    static SC_COEFFS_SCRATCH: RefCell<[[f32; 1024]; 3]> =
        const { RefCell::new([[0.0; 1024]; 3]) };
}

/// Gather a transform footprint with edge replication, matching
/// `build_stripe`'s padding.
#[inline]
fn gather_pixels(
    plane: &crate::image::Plane<f32>,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    dst: &mut [f32],
) {
    let pw = plane.xsize();
    let ph = plane.ysize();
    if width <= pw.saturating_sub(px) && height <= ph.saturating_sub(py) {
        for (v, dst) in dst.chunks_exact_mut(width).take(height).enumerate() {
            dst.copy_from_slice(&plane.row(py + v)[px..px + width]);
        }
        return;
    }
    let safe_w = width.min(pw.saturating_sub(px));
    let safe_h = height.min(ph.saturating_sub(py));
    for v in 0..height {
        let sy = if v < safe_h { py + v } else { ph - 1 };
        let row = plane.row(sy);
        let drow = &mut dst[v * width..v * width + width];
        drow[..safe_w].copy_from_slice(&row[px..px + safe_w]);
        if safe_w < width {
            drow[safe_w..].fill(row[pw - 1]);
        }
    }
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
    // Reused scratch: the gather fully overwrites the region each transform reads,
    // so re-zeroing a fresh `[0.0; 1024]` on every call is pure waste (this is the
    // hottest function in selection — thousands of calls per group).
    FT_GATHER_SCRATCH.with(|cell| {
        let tmp = &mut *cell.borrow_mut();
        match strategy {
            STRATEGY_DCT => {
                gather_pixels(plane, px, py, 8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct8x8)(src, dst);
                (1, 1)
            }
            STRATEGY_DCT16X8 => {
                gather_pixels(plane, px, py, 8, 16, &mut tmp[..128]);
                let src: &[f32; 128] = (&tmp[..128]).try_into().unwrap();
                let dst: &mut [f32; 128] = (&mut out[..128]).try_into().unwrap();
                (ctx.dct16x8)(src, dst);
                (2, 1)
            }
            STRATEGY_DCT8X16 => {
                gather_pixels(plane, px, py, 16, 8, &mut tmp[..128]);
                let src: &[f32; 128] = (&tmp[..128]).try_into().unwrap();
                let dst: &mut [f32; 128] = (&mut out[..128]).try_into().unwrap();
                (ctx.dct8x16)(src, dst);
                (2, 1)
            }
            STRATEGY_DCT16X16 => {
                gather_pixels(plane, px, py, 16, 16, &mut tmp[..256]);
                let src: &[f32; 256] = (&tmp[..256]).try_into().unwrap();
                let dst: &mut [f32; 256] = (&mut out[..256]).try_into().unwrap();
                (ctx.dct16x16)(src, dst);
                (2, 2)
            }
            STRATEGY_DCT32X32 => {
                gather_pixels(plane, px, py, 32, 32, &mut tmp[..1024]);
                let src: &[f32; 1024] = (&tmp[..1024]).try_into().unwrap();
                let dst: &mut [f32; 1024] = (&mut out[..1024]).try_into().unwrap();
                (ctx.dct32x32)(src, dst);
                (4, 4)
            }
            STRATEGY_DCT32X16 => {
                // 16 wide × 32 tall pixels (cov 2×4); normalized (cx,cy) = (4,2).
                gather_pixels(plane, px, py, 16, 32, &mut tmp[..512]);
                let src: &[f32; 512] = (&tmp[..512]).try_into().unwrap();
                let dst: &mut [f32; 512] = (&mut out[..512]).try_into().unwrap();
                (ctx.dct32x16)(src, dst);
                (4, 2)
            }
            STRATEGY_DCT16X32 => {
                // 32 wide × 16 tall pixels (cov 4×2); normalized (cx,cy) = (4,2).
                gather_pixels(plane, px, py, 32, 16, &mut tmp[..512]);
                let src: &[f32; 512] = (&tmp[..512]).try_into().unwrap();
                let dst: &mut [f32; 512] = (&mut out[..512]).try_into().unwrap();
                (ctx.dct16x32)(src, dst);
                (4, 2)
            }
            STRATEGY_DCT4X4 => {
                gather_pixels(plane, px, py, 8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct4x4)(src, dst);
                (1, 1)
            }
            STRATEGY_DCT4X8 => {
                gather_pixels(plane, px, py, 8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct4x8)(src, dst);
                (1, 1)
            }
            STRATEGY_DCT8X4 => {
                gather_pixels(plane, px, py, 8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                (ctx.dct8x4)(src, dst);
                (1, 1)
            }
            _ => unreachable!("invalid strategy {strategy}"),
        }
    })
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

struct Dct64Scratch {
    input: [f32; 4096],
    coeffs: [[f32; 4096]; 3],
}

thread_local! {
    static DCT64_SCRATCH: RefCell<Dct64Scratch> = const { RefCell::new(Dct64Scratch {
        input: [0.0; 4096],
        coeffs: [[0.0; 4096]; 3],
    }) };
}

/// Coefficient-domain cost for DCT64. Kept separate from the standard chooser
/// so the hot-path scratch and reconstruction reranker remain sized for 32x32.
#[allow(clippy::too_many_arguments)]
fn strategy_cost_large(
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
    DCT64_SCRATCH.with_borrow_mut(|cell| {
        let (width, height, size, cx, cy) = match strategy {
            STRATEGY_DCT64X64 => (64, 64, 4096, 8, 8),
            STRATEGY_DCT64X32 => (32, 64, 2048, 8, 4),
            STRATEGY_DCT32X64 => (64, 32, 2048, 8, 4),
            _ => unreachable!("invalid large strategy {strategy}"),
        };
        for (c, coeff) in cell.coeffs.iter_mut().enumerate() {
            gather_pixels(
                opsin.plane(c),
                px,
                py,
                width,
                height,
                &mut cell.input[..size],
            );
            match strategy {
                STRATEGY_DCT64X64 => (ctx.dct64x64)(&cell.input, coeff),
                STRATEGY_DCT64X32 => (ctx.dct64x32)(
                    (&cell.input[..2048]).try_into().unwrap(),
                    (&mut coeff[..2048]).try_into().unwrap(),
                ),
                STRATEGY_DCT32X64 => (ctx.dct32x64)(
                    (&cell.input[..2048]).try_into().unwrap(),
                    (&mut coeff[..2048]).try_into().unwrap(),
                ),
                _ => unreachable!(),
            }
        }
        let [x, y, b] = &mut cell.coeffs;
        apply_cfl(ctx, CflXyb { x, y, b }, size, cmap_factor);

        let mut distortion = 0.0f32;
        let mut rate = 0.0f32;
        for (c, cell) in cell.coeffs.iter().enumerate() {
            let inv_matrix: &[f32] = match strategy {
                STRATEGY_DCT64X64 => &matrices.inv_matrix_64x64(c)[..],
                STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => &matrices.inv_matrix_64x32(c)[..],
                _ => unreachable!(),
            };
            let (d, r) = channel_rd(
                ctx.sse_and_rate,
                ctx.rate_log2_lut,
                &cell[..size],
                inv_matrix,
                c,
                qac,
                if c == 0 { qm_mult_x } else { 1.0 },
                distance,
                cx,
                cy,
            );
            distortion += CHANNEL_WEIGHT[c] * d;
            rate += r;
        }
        distortion + RD_LAMBDA * (rate + meta_r)
    })
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

struct CflXyb<'a> {
    x: &'a mut [f32],
    y: &'a mut [f32],
    b: &'a mut [f32],
}

pub(crate) type ApplyCflFn = fn(&mut [f32], &[f32], &mut [f32], [f32; 3]);

#[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
fn apply_cfl_scalar(x: &mut [f32], y: &[f32], b: &mut [f32], cmap_factor: [f32; 3]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), b.len());
    let neg_cx = -cmap_factor[0];
    let neg_cb = -cmap_factor[2];
    for ((x, b), &y) in x.iter_mut().zip(b).zip(y) {
        *x = fmla(neg_cx, y, *x);
        *b = fmla(neg_cb, y, *b);
    }
}

pub(crate) fn selected_apply_cfl_fn() -> ApplyCflFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return |x, y, b, cmap_factor| unsafe { crate::neon::apply_cfl_neon(x, y, b, cmap_factor) };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |x, y, b, cmap_factor| unsafe { crate::avx::apply_cfl_avx2(x, y, b, cmap_factor) };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return |x, y, b, cmap_factor| crate::wasm::apply_cfl_wasm(x, y, b, cmap_factor);
    #[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
    apply_cfl_scalar
}

fn apply_cfl(ctx: &EncodingContext, coeffs: CflXyb<'_>, size: usize, cmap_factor: [f32; 3]) {
    let CflXyb { x, y, b } = coeffs;
    assert!(size <= x.len() && size <= y.len() && size <= b.len());
    (ctx.apply_cfl)(&mut x[..size], &y[..size], &mut b[..size], cmap_factor);
}

#[inline]
fn inverse_matrix_for(matrices: &DequantMatrices, strategy: u8, channel: usize) -> &[f32] {
    match strategy {
        STRATEGY_DCT => &matrices.inv_matrix(channel)[..],
        STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(channel)[..],
        STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(channel)[..],
        STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(channel)[..],
        STRATEGY_DCT32X32 => &matrices.inv_matrix_32x32(channel)[..],
        STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => &matrices.inv_matrix_32x16(channel)[..],
        _ => &matrices.inv_matrix_16x8(channel)[..],
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn coefficient_dist_and_rate(
    ctx: &EncodingContext,
    strategy: u8,
    coeffs: &[[f32; 1024]; 3],
    size: usize,
    qac: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    distance: f32,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let mut d_total = 0.0f32;
    let mut r_total = 0.0f32;
    for c in 0..3 {
        let qm_mult = if c == 0 { qm_mult_x } else { 1.0 };
        let (d, r) = channel_rd(
            ctx.sse_and_rate,
            ctx.rate_log2_lut,
            &coeffs[c][..size],
            inverse_matrix_for(matrices, strategy, c),
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
        let [x, y, b] = coeffs;
        apply_cfl(ctx, CflXyb { x, y, b }, size, cmap_factor);

        let (d_total, r_total) = match distortion_model {
            DistortionModel::Reconstruction => recon_dist_and_rate(
                ctx.rate_log2_lut,
                coeffs,
                [
                    inverse_matrix_for(matrices, strategy, 0),
                    inverse_matrix_for(matrices, strategy, 1),
                    inverse_matrix_for(matrices, strategy, 2),
                ],
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
            DistortionModel::Coefficient => coefficient_dist_and_rate(
                ctx, strategy, coeffs, size, qac, qm_mult_x, matrices, distance, cx, cy,
            ),
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

#[derive(Clone, Copy)]
struct Sub8Costs {
    dct8: f32,
    dct4x4: f32,
    dct4x8: f32,
    dct8x4: f32,
}

#[inline]
fn forward_sub8_transform(
    ctx: &EncodingContext,
    strategy: u8,
    input: &[f32; 64],
    output: &mut [f32; 1024],
) {
    let dst: &mut [f32; 64] = (&mut output[..64]).try_into().unwrap();
    match strategy {
        STRATEGY_DCT => (ctx.dct8x8)(input, dst),
        STRATEGY_DCT4X4 => (ctx.dct4x4)(input, dst),
        STRATEGY_DCT4X8 => (ctx.dct4x8)(input, dst),
        STRATEGY_DCT8X4 => (ctx.dct8x4)(input, dst),
        _ => unreachable!("non-sub8 strategy {strategy}"),
    }
}

/// Evaluate all sub-8 candidates after gathering each channel's shared 8x8
/// footprint once. `cached_dct8` is the identical first-pass incumbent cost;
/// it is absent only for an odd edge block that was not part of a 2x2 superblock.
#[allow(clippy::too_many_arguments)]
fn sub8_strategy_costs(
    ctx: &EncodingContext,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
    cached_dct8: Option<f32>,
) -> Sub8Costs {
    let mut pixels = [[0.0f32; 64]; 3];
    for (c, input) in pixels.iter_mut().enumerate() {
        gather_pixels(opsin.plane(c), px, py, 8, 8, input);
    }

    SC_COEFFS_SCRATCH.with(|cell| {
        let coeffs = &mut *cell.borrow_mut();
        let mut evaluate = |strategy| {
            for c in 0..3 {
                forward_sub8_transform(ctx, strategy, &pixels[c], &mut coeffs[c]);
            }
            let [x, y, b] = &mut *coeffs;
            apply_cfl(ctx, CflXyb { x, y, b }, 64, cmap_factor);
            let (distortion, rate) = coefficient_dist_and_rate(
                ctx, strategy, coeffs, 64, qac, qm_mult_x, matrices, distance, 1, 1,
            );
            distortion + RD_LAMBDA * (rate + meta_r)
        };

        let dct8 = if let Some(cost) = cached_dct8 {
            cost
        } else {
            evaluate(STRATEGY_DCT)
        };
        Sub8Costs {
            dct8,
            dct4x4: evaluate(STRATEGY_DCT4X4),
            dct4x8: evaluate(STRATEGY_DCT4X8),
            dct8x4: evaluate(STRATEGY_DCT8X4),
        }
    })
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
    dct8_costs: &mut [f32],
    dct8_cost_y0: usize,
    dct8_cost_stride: usize,
    scope: SearchScope,
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
            dct8_costs[(by0 + dy - dct8_cost_y0) * dct8_cost_stride + bx0 + dx] = c8[dy][dx];
        }
    }

    // Vertical pairs (DCT16X8): one per column. Skipped entirely under
    // `SearchScope::Squares` — four `strategy_cost` calls per super-block.
    let rect_cost = |px: usize, py: usize, strategy: u8, qac: f32| -> f32 {
        if !scope.rectangles() {
            return f32::INFINITY;
        }
        BIAS_RECT
            * strategy_cost(
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
            )
    };
    let v_left = rect_cost(px0, py0, STRATEGY_DCT16X8, qac[0][0].max(qac[1][0]));
    let v_right = rect_cost(px0 + 8, py0, STRATEGY_DCT16X8, qac[0][1].max(qac[1][1]));
    let h_top = rect_cost(px0, py0, STRATEGY_DCT8X16, qac[0][0].max(qac[0][1]));
    let h_bot = rect_cost(px0, py0 + 8, STRATEGY_DCT8X16, qac[1][0].max(qac[1][1]));

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
    let mut max_q = 1u8;
    let mut sum = 0u32;
    for y in by..by + h {
        for &q in &quant_field.row(y)[bx..bx + w] {
            max_q = max_q.max(q);
            sum += u32::from(q);
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
    scope: SearchScope,
) -> f32 {
    // First-pass DCT8 incumbents are consumed again by the sub-8 refinement.
    // Keep only this worker's row band so parallel selection stays independent.
    let mut dct8_costs = vec![f32::NAN; xsize * (y_end - y_begin)];
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
                            &mut dct8_costs,
                            y_begin,
                            xsize,
                            scope,
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
                // Two DCT32X16 (each 2 wide x 4 tall) tiling the region: left +
                // right, and two DCT16X32 (each 4 wide x 2 tall): top + bottom.
                // Skipped under `SearchScope::Squares`.
                let rect32 = |bx: usize, by: usize, strategy: u8, cw: usize, ch: usize| -> f32 {
                    if !scope.rectangles() {
                        return f32::INFINITY;
                    }
                    strategy_cost(
                        ctx,
                        strategy,
                        opsin,
                        dc_group_px + bx * 8,
                        dc_group_py + by * 8,
                        region_qac(quant_field, bx, by, cw, ch, scale, distance),
                        qm_mult_x,
                        matrices,
                        meta_r,
                        distance,
                        cmap_factor,
                    )
                };
                let cl = rect32(bx, by, STRATEGY_DCT32X16, 2, 4);
                let cr = rect32(bx + 2, by, STRATEGY_DCT32X16, 2, 4);
                let ct = rect32(bx, by, STRATEGY_DCT16X32, 4, 2);
                let cb = rect32(bx, by + 2, STRATEGY_DCT16X32, 4, 2);

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
                        &mut dct8_costs,
                        y_begin,
                        xsize,
                        scope,
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
                    &mut dct8_costs,
                    y_begin,
                    xsize,
                    scope,
                );
                bx += 2;
            }
        }
        by += if four_row { 4 } else { 2 };
    }

    // Sub-8x8 refinement for this row band. `SearchScope::Squares` stops here:
    // it commits no sub-8 strategies, so the caller's activation gate (which
    // weighs their metadata cost) sees an empty set and is a no-op.
    let mut benefit = 0.0f32;
    if !scope.rectangles() {
        return benefit;
    }
    for by in y_begin..y_end {
        for bx in 0..xsize {
            if ac_strategy.raw_strategy(bx, by) != STRATEGY_DCT {
                continue;
            }
            let qac = region_qac(quant_field, bx, by, 1, 1, scale, distance);
            let px = dc_group_px + bx * 8;
            let py = dc_group_py + by * 8;
            let cmap_factor = cmap_factors(ytox_map, ytob_map, bx, by);
            let cached_dct8 = dct8_costs[(by - y_begin) * xsize + bx];
            let costs = sub8_strategy_costs(
                ctx,
                opsin,
                px,
                py,
                qac,
                qm_mult_x,
                matrices,
                meta_r,
                distance,
                cmap_factor,
                cached_dct8.is_finite().then_some(cached_dct8),
            );
            let cost8 = costs.dct8;
            let cost4 = BIAS_4X4 * costs.dct4x4;
            let cost48 = BIAS_4X8 * costs.dct4x8;
            let cost84 = BIAS_4X8 * costs.dct8x4;
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
    speed: crate::Speed,
) -> f32 {
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    // DCT8 wins the high-quality RD comparison outright.
    if use_dct8_only(distance) {
        return 0.0;
    }
    let scope = SearchScope::for_speed(speed);
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
            scope,
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
                scope,
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

    // The 64x32/32x64 transforms are slow-tier choices from distance 2.5;
    // DCT64 joins the same full-tile competition at distance 3.0.
    if use_dct64(speed, distance) || use_dct64_rect(speed, distance) {
        let rect_live = use_dct64_rect(speed, distance);
        for by in (0..ysize).step_by(8) {
            for bx in (0..xsize).step_by(8) {
                if !ac_strategy.can_place_strategy(bx, by, STRATEGY_DCT64X64) {
                    if !rect_live {
                        continue;
                    }
                    let cmap = cmap_factors(ytox_map, ytob_map, bx, by);
                    for sx in [0usize, 4] {
                        let x = bx + sx;
                        if !ac_strategy.can_place_strategy(x, by, STRATEGY_DCT64X32) {
                            continue;
                        }
                        let rect = strategy_cost_large(
                            ctx,
                            STRATEGY_DCT64X32,
                            opsin,
                            dc_group_px + x * 8,
                            dc_group_py + by * 8,
                            region_qac(quant_field, x, by, 4, 8, scale, distance),
                            qm_mult_x,
                            matrices,
                            meta_r,
                            distance,
                            cmap,
                        );
                        let mut tiled = 0.0;
                        for sy in [0usize, 4] {
                            tiled += strategy_cost(
                                ctx,
                                STRATEGY_DCT32X32,
                                opsin,
                                dc_group_px + x * 8,
                                dc_group_py + (by + sy) * 8,
                                region_qac(quant_field, x, by + sy, 4, 4, scale, distance),
                                qm_mult_x,
                                matrices,
                                meta_r,
                                distance,
                                cmap,
                            );
                        }
                        if merge_beats_dct8(
                            BIAS_64_RECT * rect,
                            tiled,
                            distance,
                            MERGE_MARGIN_64_HQ,
                        ) {
                            ac_strategy.set_first(x, by, STRATEGY_DCT64X32);
                        }
                    }
                    for sy in [0usize, 4] {
                        let y = by + sy;
                        if !ac_strategy.can_place_strategy(bx, y, STRATEGY_DCT32X64) {
                            continue;
                        }
                        let rect = strategy_cost_large(
                            ctx,
                            STRATEGY_DCT32X64,
                            opsin,
                            dc_group_px + bx * 8,
                            dc_group_py + y * 8,
                            region_qac(quant_field, bx, y, 8, 4, scale, distance),
                            qm_mult_x,
                            matrices,
                            meta_r,
                            distance,
                            cmap,
                        );
                        let mut tiled = 0.0;
                        for sx in [0usize, 4] {
                            tiled += strategy_cost(
                                ctx,
                                STRATEGY_DCT32X32,
                                opsin,
                                dc_group_px + (bx + sx) * 8,
                                dc_group_py + y * 8,
                                region_qac(quant_field, bx + sx, y, 4, 4, scale, distance),
                                qm_mult_x,
                                matrices,
                                meta_r,
                                distance,
                                cmap,
                            );
                        }
                        if merge_beats_dct8(
                            BIAS_64_RECT * rect,
                            tiled,
                            distance,
                            MERGE_MARGIN_64_HQ,
                        ) {
                            ac_strategy.set_first(bx, y, STRATEGY_DCT32X64);
                        }
                    }
                    continue;
                }
                let cmap = cmap_factors(ytox_map, ytob_map, bx, by);
                let mut cost32 = 0.0f32;
                for sy in [0usize, 4] {
                    for sx in [0usize, 4] {
                        cost32 += strategy_cost(
                            ctx,
                            STRATEGY_DCT32X32,
                            opsin,
                            dc_group_px + (bx + sx) * 8,
                            dc_group_py + (by + sy) * 8,
                            region_qac(quant_field, bx + sx, by + sy, 4, 4, scale, distance),
                            qm_mult_x,
                            matrices,
                            meta_r,
                            distance,
                            cmap,
                        );
                    }
                }

                let mut best_cost = cost32;
                let mut best_strategy = None;

                let mut cost64x32 = if rect_live { 0.0f32 } else { f32::INFINITY };
                for sx in [0usize, 4] {
                    if !rect_live {
                        break;
                    }
                    cost64x32 += strategy_cost_large(
                        ctx,
                        STRATEGY_DCT64X32,
                        opsin,
                        dc_group_px + (bx + sx) * 8,
                        dc_group_py + by * 8,
                        region_qac(quant_field, bx + sx, by, 4, 8, scale, distance),
                        qm_mult_x,
                        matrices,
                        meta_r,
                        distance,
                        cmap,
                    );
                }
                let cost64x32 = BIAS_64_RECT * cost64x32;
                if cost64x32 < best_cost {
                    best_cost = cost64x32;
                    best_strategy = Some(STRATEGY_DCT64X32);
                }

                let mut cost32x64 = if rect_live { 0.0f32 } else { f32::INFINITY };
                for sy in [0usize, 4] {
                    if !rect_live {
                        break;
                    }
                    cost32x64 += strategy_cost_large(
                        ctx,
                        STRATEGY_DCT32X64,
                        opsin,
                        dc_group_px + bx * 8,
                        dc_group_py + (by + sy) * 8,
                        region_qac(quant_field, bx, by + sy, 8, 4, scale, distance),
                        qm_mult_x,
                        matrices,
                        meta_r,
                        distance,
                        cmap,
                    );
                }
                let cost32x64 = BIAS_64_RECT * cost32x64;
                if cost32x64 < best_cost {
                    best_cost = cost32x64;
                    best_strategy = Some(STRATEGY_DCT32X64);
                }

                if use_dct64(speed, distance) {
                    let cost64 = BIAS_64X64
                        * strategy_cost_large(
                            ctx,
                            STRATEGY_DCT64X64,
                            opsin,
                            dc_group_px + bx * 8,
                            dc_group_py + by * 8,
                            region_qac(quant_field, bx, by, 8, 8, scale, distance),
                            qm_mult_x,
                            matrices,
                            meta_r,
                            distance,
                            cmap,
                        );
                    if cost64 < best_cost {
                        best_cost = cost64;
                        best_strategy = Some(STRATEGY_DCT64X64);
                    }
                }

                if merge_beats_dct8(best_cost, cost32, distance, MERGE_MARGIN_64_HQ) {
                    match best_strategy {
                        Some(STRATEGY_DCT64X32) => {
                            ac_strategy.set_first(bx, by, STRATEGY_DCT64X32);
                            ac_strategy.set_first(bx + 4, by, STRATEGY_DCT64X32);
                        }
                        Some(STRATEGY_DCT32X64) => {
                            ac_strategy.set_first(bx, by, STRATEGY_DCT32X64);
                            ac_strategy.set_first(bx, by + 4, STRATEGY_DCT32X64);
                        }
                        Some(STRATEGY_DCT64X64) => {
                            ac_strategy.set_first(bx, by, STRATEGY_DCT64X64);
                        }
                        None => {}
                        _ => unreachable!(),
                    }
                }
            }
        }
    }

    // Second pass — reconstruction-based rerank. The fast selector over-merges at
    // high quality; here we revisit only the *selected* large transforms and
    // downgrade a merge to its tiled DCT8 when the SSIM-reconstruction RD cost
    // prefers it. Only large transforms are scored (a fraction of blocks), so the
    // expensive recon distortion runs on far fewer candidates than a full recon
    // selection while capturing the same structural win.
    // The SSIM reconstruction rerank runs in both scopes. Under
    // `SearchScope::Squares` the only merges present are DCT16X16 and
    // DCT32X32, so it scores exactly those.
    if scope.rerank(distance) {
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
    }

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
        // DCT64 is an explicitly gated slow-tier choice. The reconstruction
        // selector uses 32x32 scratch and intentionally does not rerank it.
        if matches!(
            strat,
            STRATEGY_DCT64X64 | STRATEGY_DCT64X32 | STRATEGY_DCT32X64
        ) {
            continue;
        }
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
        DCT8_ONLY_MAX_DISTANCE, FAST_RERANK_MAX_DISTANCE, MERGE_MARGIN_16_HQ, MERGE_MARGIN_32_HQ,
        MERGE_MARGIN_PAIR_HQ, SearchScope, aggregate_qac_2x2, aggregate_quant, cmap_factors,
        fill_ac_strategy, merge_beats_dct8, merge_margin, strategy_cost, sub8_strategy_costs,
        use_dct8_only, use_dct64, use_dct64_rect,
    };
    use crate::dc_group_data::{
        AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4,
        STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT32X32, STRATEGY_DCT32X64,
        STRATEGY_DCT64X32, STRATEGY_DCT64X64,
    };
    use crate::encoding_context::EncodingContext;
    use crate::image::{Image3F, ImageB, ImageSB};
    use crate::inflated_cost::{
        forward_for, forward_matrix, reconstruct_error, strategy_pixel_count,
    };
    use crate::quant_weights::DequantMatrices;

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
    fn dct64_is_gated_to_slow_at_distance_three() {
        assert!(!use_dct64(crate::Speed::Fast, 3.0));
        assert!(!use_dct64(crate::Speed::Slow, 2.999_999));
        assert!(use_dct64(crate::Speed::Slow, 3.0));
        assert!(use_dct64(crate::Speed::Slow, 6.0));
    }

    #[test]
    fn dct64_rectangles_are_gated_to_slow_at_distance_two_point_five() {
        assert!(!use_dct64_rect(crate::Speed::Fast, 2.5));
        assert!(!use_dct64_rect(crate::Speed::Slow, 2.499_999));
        assert!(use_dct64_rect(crate::Speed::Slow, 2.5));
        assert!(use_dct64_rect(crate::Speed::Slow, 6.0));
    }

    #[test]
    fn dct64_selection_obeys_gate() {
        let ctx = EncodingContext::new();
        let opsin = Image3F::new(64, 64);
        let matrices = DequantMatrices::new();
        let maps = ImageSB::new_fill(1, 1, 0);

        let select = |speed, distance| {
            let mut qf = ImageB::new_fill(8, 8, 8);
            let mut strategies = AcStrategyImage::new(8, 8);
            fill_ac_strategy(
                &ctx,
                &opsin,
                0,
                0,
                distance,
                1.0,
                2,
                &matrices,
                &mut qf,
                &maps,
                &maps,
                &mut strategies,
                1,
                speed,
            );
            strategies.raw_strategy(0, 0)
        };

        assert_ne!(select(crate::Speed::Fast, 3.0), STRATEGY_DCT64X64);
        assert_ne!(select(crate::Speed::Slow, 2.999), STRATEGY_DCT64X64);
        assert_eq!(select(crate::Speed::Slow, 3.0), STRATEGY_DCT64X64);
        assert!(!matches!(
            select(crate::Speed::Fast, 2.5),
            STRATEGY_DCT64X32 | STRATEGY_DCT32X64
        ));
        assert!(!matches!(
            select(crate::Speed::Slow, 2.499),
            STRATEGY_DCT64X32 | STRATEGY_DCT32X64
        ));
        assert!(matches!(
            select(crate::Speed::Slow, 2.5),
            STRATEGY_DCT64X32 | STRATEGY_DCT32X64
        ));
    }

    #[test]
    fn standalone_dct64_rectangles_are_selected() {
        let ctx = EncodingContext::new();
        let matrices = DequantMatrices::new();
        let maps = ImageSB::new_fill(1, 1, 0);
        let select = |blocks_x, blocks_y| {
            let opsin = Image3F::new(blocks_x * 8, blocks_y * 8);
            let mut qf = ImageB::new_fill(blocks_x, blocks_y, 8);
            let mut strategies = AcStrategyImage::new(blocks_x, blocks_y);
            fill_ac_strategy(
                &ctx,
                &opsin,
                0,
                0,
                2.5,
                1.0,
                2,
                &matrices,
                &mut qf,
                &maps,
                &maps,
                &mut strategies,
                1,
                crate::Speed::Slow,
            );
            strategies.raw_strategy(0, 0)
        };
        assert_eq!(select(4, 8), STRATEGY_DCT64X32);
        assert_eq!(select(8, 4), STRATEGY_DCT32X64);
    }

    /// The Fast tier runs the same RD model but offers only square merges.
    /// A flat 32x32 region must still merge, and must never come back as a
    /// rectangle, a sub-8x8 split, or a 64px transform.
    #[test]
    fn fast_scope_selects_squares_and_no_other_merge_shape() {
        let ctx = EncodingContext::new();
        let matrices = DequantMatrices::new();
        let maps = ImageSB::new_fill(1, 1, 0);
        let opsin = Image3F::new(32, 32);
        let mut qf = ImageB::new_fill(4, 4, 8);
        let mut strategies = AcStrategyImage::new(4, 4);
        fill_ac_strategy(
            &ctx,
            &opsin,
            0,
            0,
            3.0,
            1.0,
            2,
            &matrices,
            &mut qf,
            &maps,
            &maps,
            &mut strategies,
            1,
            crate::Speed::Fast,
        );
        assert_eq!(strategies.raw_strategy(0, 0), STRATEGY_DCT32X32);
        for (_, _, strat) in strategies.iter_first_blocks() {
            assert!(
                matches!(strat, STRATEGY_DCT | STRATEGY_DCT16X16 | STRATEGY_DCT32X32),
                "Fast produced a non-square merge: {strat}"
            );
        }
    }

    /// The Fast tier reranks only at high quality, where the coefficient model
    /// over-merges; past the cutoff the rerank costs time to lose rate.
    #[test]
    fn fast_rerank_is_gated_by_distance_but_slow_always_reranks() {
        assert!(SearchScope::Squares.rerank(FAST_RERANK_MAX_DISTANCE));
        assert!(!SearchScope::Squares.rerank(FAST_RERANK_MAX_DISTANCE + 0.001));
        assert!(SearchScope::Full.rerank(6.0));
    }

    /// The DCT64 pass used to be gated entirely on `use_dct64_rect`, so the
    /// square transform was only ever considered when the rectangles were also
    /// live. The two gates are independent now; raising the rectangle threshold
    /// above the square's must not silently disable DCT64X64.
    #[test]
    fn square_dct64_is_considered_independently_of_the_rectangle_gate() {
        let ctx = EncodingContext::new();
        let matrices = DequantMatrices::new();
        let maps = ImageSB::new_fill(1, 1, 0);
        let opsin = Image3F::new(64, 64);
        let mut qf = ImageB::new_fill(8, 8, 8);
        let mut strategies = AcStrategyImage::new(8, 8);
        // distance 3.0: use_dct64 is live, and so is use_dct64_rect.
        fill_ac_strategy(
            &ctx,
            &opsin,
            0,
            0,
            3.0,
            1.0,
            2,
            &matrices,
            &mut qf,
            &maps,
            &maps,
            &mut strategies,
            1,
            crate::Speed::Slow,
        );
        // A flat tile merges to the largest available transform; whichever of
        // the three wins, the pass must have run and produced a 64px merge.
        assert!(matches!(
            strategies.raw_strategy(0, 0),
            STRATEGY_DCT64X64 | STRATEGY_DCT64X32 | STRATEGY_DCT32X64
        ));
    }

    #[test]
    fn bundled_sub8_costs_match_independent_evaluation_and_cached_dct8() {
        let mut opsin = Image3F::new(11, 13);
        let mut state = 0x1234_5678u32;
        for c in 0..3 {
            for y in 0..opsin.ysize() {
                for value in opsin.plane_mut(c).row_mut(y) {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                }
            }
        }

        let ctx = EncodingContext::new();
        let matrices = DequantMatrices::new();
        let (px, py) = (7usize, 9usize); // exercise right and bottom replication
        let qac = 3.75;
        let qm_mult_x = 1.25;
        let meta_r = 0.4;
        let distance = 1.5;
        let cmap = [0.25, 0.0, -0.375];
        let independent = [
            strategy_cost(
                &ctx,
                STRATEGY_DCT,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                &matrices,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                STRATEGY_DCT4X4,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                &matrices,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                STRATEGY_DCT4X8,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                &matrices,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                STRATEGY_DCT8X4,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                &matrices,
                meta_r,
                distance,
                cmap,
            ),
        ];

        for cached in [None, Some(independent[0])] {
            let bundled = sub8_strategy_costs(
                &ctx, &opsin, px, py, qac, qm_mult_x, &matrices, meta_r, distance, cmap, cached,
            );
            let actual = [bundled.dct8, bundled.dct4x4, bundled.dct4x8, bundled.dct8x4];
            for (a, b) in actual.into_iter().zip(independent) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
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
