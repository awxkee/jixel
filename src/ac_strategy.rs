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

use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::{
    AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4,
    STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16,
    STRATEGY_DCT32X32,
};
use crate::dct::{DctInput, fmla};
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, ImageB, ImageSB};
use crate::inflated_cost::{
    CHANNEL_WEIGHT, ReconDistInput, ReconQuantization, ReconScoring, ReconSource, ReconTransform,
    channel_rd,
};

const DCT8_ONLY_MAX_DISTANCE: f32 = 0.056_713_393;

/// Above this distance the [`SearchScope::Squares`] tier stops reranking; see
/// [`SearchScope::rerank`].
const FAST_RERANK_MAX_DISTANCE: f32 = 2.0;

#[inline]
fn use_dct8_only(distance: f32) -> bool {
    distance <= DCT8_ONLY_MAX_DISTANCE
}

/// Distances bounding the high-quality band: a [`Banded`] knob holds its
/// high-quality value at or below `MERGE_BAND_D0`, its base value from
/// `MERGE_BAND_D1` up, and interpolates in between.
const MERGE_BAND_D0: f32 = 0.5;
const MERGE_BAND_D1: f32 = 1.5;

/// A merge-selection constant whose value differs between the high-quality and
/// the mid/low band. Quantization error at high quality is small enough that the
/// coefficient-domain model's merge estimate is mostly noise, so merges have to
/// clear a far higher bar there than they do once quantization bites.
#[derive(Clone, Copy)]
struct Banded {
    hq: f32,
    base: f32,
}

impl Banded {
    const fn new(hq: f32, base: f32) -> Self {
        Self { hq, base }
    }

    #[inline]
    fn at(self, distance: f32) -> f32 {
        let t = ((distance - MERGE_BAND_D0) / (MERGE_BAND_D1 - MERGE_BAND_D0)).clamp(0.0, 1.0);
        fmla(t, self.base - self.hq, self.hq)
    }
}

/// Bits charged once per placed transform for the AC metadata the
/// coefficient model cannot see: the ACS symbol, the quant-field symbol and the
/// per-block bookkeeping around them. Flat, because a candidate pays for those
/// tokens whatever the distance.
const META_R: f32 = 4.0;

/// High-rate-optimal Lagrange multiplier for unit-step (Δ = 1) scalar
/// quantization: `λ* = Δ²·ln2 / 6`. Distortion is in quant-units², rate in
/// bits, so `λ·R` is in quant-units² and adds cleanly to D.
pub(crate) const RD_LAMBDA: f32 = 0.080_867_17;

/// How much better the tiled DCT8 must look to [`rerank_large_transforms`]
/// before a committed merge is downgraded: the test is `j_dct8 < j_big * m`.
/// At high quality the reconstruction costs sit on the noise floor and a
/// knife-edge (m = 1) comparison is as good as any; from d=1.5 up the merge is
/// worth defending, and the tiled DCT8 has to be ~13% cheaper to win.
const RERANK_DOWNGRADE_MARGIN: Banded = Banded::new(1.0, 0.87);

const RERANK_LAMBDA_LO: f32 = 0.25;
const RERANK_LAMBDA_HI: f32 = 10.0;
const RERANK_LAMBDA_D0: f32 = 1.0;
const RERANK_LAMBDA_D1: f32 = 4.0;

const BIAS_RECT: Banded = Banded::new(1.026_200_7, 1.0);
const BIAS_16X16: Banded = Banded::new(1.205_927_5, 1.0);
const BIAS_32X32: Banded = Banded::new(1.205_927_5, 1.0);
const BIAS_RECT32: Banded = Banded::new(1.326_520_2, 1.10);

const MERGE_MARGIN_PAIR: Banded = Banded::new(0.036_252_015, 0.04);
const MERGE_MARGIN_16: Banded = Banded::new(0.192_438_88, 0.08);
const MERGE_MARGIN_32_RECT: Banded = Banded::new(0.161_916_72, 0.11);
const MERGE_MARGIN_32: Banded = Banded::new(0.329_207_82, 0.14);

const MERGE_MARGIN_LOWQ_FRACTION: f32 = 0.20;
const MERGE_MARGIN_FADE_START: f32 = 1.0;
const MERGE_MARGIN_FADE_END: f32 = 4.0;
/// Sub-8x8 selection biases. Each candidate's RD cost is scaled by its bias
/// before being compared with the 8x8 incumbent, so >1 makes that strategy
/// harder to select.
const BIAS_4X4: f32 = 1.33;
const BIAS_4X8: f32 = 1.09;

/// Sub-8 is skipped entirely above this distance.
const SUB8_MAX_DISTANCE: f32 = 1.5;
#[inline]
fn merge_margin(distance: f32, margin: Banded) -> f32 {
    let fade = ((distance - MERGE_MARGIN_FADE_START)
        / (MERGE_MARGIN_FADE_END - MERGE_MARGIN_FADE_START))
        .clamp(0.0, 1.0);
    margin.at(distance) * (1.0 - fade * (1.0 - MERGE_MARGIN_LOWQ_FRACTION))
}

#[inline]
fn merge_beats_dct8(candidate_cost: f32, dct8_cost: f32, accept: f32) -> bool {
    candidate_cost < dct8_cost * accept
}

/// The merge-selection knobs resolved at the encode's distance, so the band
/// interpolation and the low-quality margin fade are paid once per encode
/// instead of per candidate block. Lives in [`EncodingContext`].
#[derive(Clone, Copy)]
pub(crate) struct MergeTuning {
    /// Per-transform-size cost biases.
    pub(crate) bias_rect: f32,
    pub(crate) bias_16x16: f32,
    pub(crate) bias_32x32: f32,
    pub(crate) bias_rect32: f32,
    /// Fraction of the tiled-DCT8 cost a merge must come in under, i.e.
    /// `1 - margin` with the fade already applied.
    pub(crate) accept_pair: f32,
    pub(crate) accept_16: f32,
    pub(crate) accept_32_rect: f32,
    pub(crate) accept_32: f32,
    /// See [`RERANK_DOWNGRADE_MARGIN`].
    pub(crate) rerank_margin: f32,
}

impl MergeTuning {
    pub(crate) fn new(distance: f32) -> Self {
        let accept = |margin: Banded| 1.0 - merge_margin(distance, margin);
        Self {
            bias_rect: BIAS_RECT.at(distance),
            bias_16x16: BIAS_16X16.at(distance),
            bias_32x32: BIAS_32X32.at(distance),
            bias_rect32: BIAS_RECT32.at(distance),
            accept_pair: accept(MERGE_MARGIN_PAIR),
            accept_16: accept(MERGE_MARGIN_16),
            accept_32_rect: accept(MERGE_MARGIN_32_RECT),
            accept_32: accept(MERGE_MARGIN_32),
            rerank_margin: RERANK_DOWNGRADE_MARGIN.at(distance),
        }
    }
}

/// How much of the transform space the chooser explores.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SearchScope {
    Squares,
    Full,
}

impl SearchScope {
    #[inline]
    fn for_speed(speed: crate::Speed) -> Self {
        match speed {
            // Fastest never reaches the chooser (`fill_ac_strategy` returns
            // before scoping); Squares is the defensive mapping.
            crate::Speed::Fastest | crate::Speed::Fast => SearchScope::Squares,
            crate::Speed::Slow => SearchScope::Full,
        }
    }

    #[inline]
    fn rectangles(self) -> bool {
        self == SearchScope::Full
    }

    /// Whether the SSIM reconstruction rerank runs.
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

#[inline]
fn dct_input<'a, const W: usize, const H: usize>(
    plane: &'a crate::image::Plane<f32>,
    tmp: &'a mut [f32; 1024],
    px: usize,
    py: usize,
) -> DctInput<'a, W, H> {
    if W <= plane.xsize().saturating_sub(px) && H <= plane.ysize().saturating_sub(py) {
        let stride = plane.xsize();
        DctInput::new(&plane.as_slice()[py * stride + px..], stride)
    } else {
        gather_pixels(plane, px, py, W, H, &mut tmp[..W * H]);
        DctInput::new(&tmp[..W * H], W)
    }
}

/// Forward-transform the `strategy`'s pixel footprint at absolute pixel
/// `(px, py)` for one channel into `out` (natural coefficient storage matching
/// `write_ac_group`). Returns `(cx, cy)` covered-block counts after the
/// libjxl-tiny `cx ≥ cy` normalisation, i.e. the storage shape in 8-blocks.
fn forward_transform(
    ctx: &EncodingContext,
    tmp: &mut [f32; 1024],
    strategy: u8,
    plane: &crate::image::Plane<f32>,
    px: usize,
    py: usize,
    out: &mut [f32; 1024],
) -> (usize, usize) {
    // Interior blocks point directly into the image plane. Edge blocks retain
    // the replicated-border gather because their logical footprint extends past
    // the allocated plane.
    match strategy {
        STRATEGY_DCT => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.dct8x8)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        STRATEGY_DCT16X8 => {
            let dst: &mut [f32; 128] = out.first_chunk_mut::<128>().unwrap();
            (ctx.dct16x8)(dct_input(plane, tmp, px, py), dst);
            (2, 1)
        }
        STRATEGY_DCT8X16 => {
            let dst: &mut [f32; 128] = out.first_chunk_mut::<128>().unwrap();
            (ctx.dct8x16)(dct_input(plane, tmp, px, py), dst);
            (2, 1)
        }
        STRATEGY_DCT16X16 => {
            let dst: &mut [f32; 256] = out.first_chunk_mut::<256>().unwrap();
            (ctx.dct16x16)(dct_input(plane, tmp, px, py), dst);
            (2, 2)
        }
        STRATEGY_DCT32X32 => {
            let dst: &mut [f32; 1024] = out.first_chunk_mut::<1024>().unwrap();
            (ctx.dct32x32)(dct_input(plane, tmp, px, py), dst);
            (4, 4)
        }
        STRATEGY_DCT32X16 => {
            // 16 wide × 32 tall pixels (cov 2×4); normalized (cx,cy) = (4,2).
            let dst: &mut [f32; 512] = out.first_chunk_mut::<512>().unwrap();
            (ctx.dct32x16)(dct_input(plane, tmp, px, py), dst);
            (4, 2)
        }
        STRATEGY_DCT16X32 => {
            // 32 wide × 16 tall pixels (cov 4×2); normalized (cx,cy) = (4,2).
            let dst: &mut [f32; 512] = out.first_chunk_mut::<512>().unwrap();
            (ctx.dct16x32)(dct_input(plane, tmp, px, py), dst);
            (4, 2)
        }
        STRATEGY_DCT4X4 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.dct4x4)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        STRATEGY_DCT4X8 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.dct4x8)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        STRATEGY_DCT8X4 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.dct8x4)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        _ => unreachable!("invalid strategy {strategy}"),
    }
}

/// Full RD cost `J = D + λR` of coding `strategy` at absolute pixel `(px, py)`.
/// Combines the three channels with the selection-time CfL approximation.
fn strategy_cost(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
) -> f32 {
    strategy_cost_impl(
        ctx,
        scratch,
        strategy,
        opsin,
        px,
        py,
        qac,
        qm_mult_x,
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
    scratch: &mut CoderScratch,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
) -> f32 {
    strategy_cost_impl(
        ctx,
        scratch,
        strategy,
        opsin,
        px,
        py,
        qac,
        qm_mult_x,
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
fn inverse_matrix_for(ctx: &EncodingContext, strategy: u8, channel: usize) -> &[f32] {
    let matrices = &ctx.matrices;
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
    distance: f32,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let mut d_total = 0.0f32;
    let mut r_total = 0.0f32;
    for c in 0..3 {
        let qm_mult = if c == 0 {
            qm_mult_x
        } else if c == 2 {
            crate::frame::b_qm_mul()
        } else {
            1.0
        };
        let (d, r) = channel_rd(
            ctx.sse_and_rate,
            ctx.rate_log2_lut,
            &coeffs[c][..size],
            inverse_matrix_for(ctx, strategy, c),
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
    scratch: &mut CoderScratch,
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
    distortion_model: DistortionModel,
) -> f32 {
    let mut cxy = (1usize, 1usize);
    let CoderScratch {
        strategy_coeffs: coeffs,
        transform_gather,
        recon,
        ..
    } = scratch;
    for c in 0..3 {
        cxy = forward_transform(
            ctx,
            transform_gather,
            strategy,
            opsin.plane(c),
            px,
            py,
            &mut coeffs[c],
        );
    }
    let (cx, cy) = cxy;
    let size = cx * cy * 64;

    // Apply the same per-tile CfL slopes used by final coefficient coding.
    let [x, y, b] = &mut **coeffs;
    apply_cfl(ctx, CflXyb { x, y, b }, size, cmap_factor);

    let (d_total, r_total) = match distortion_model {
        DistortionModel::Reconstruction => (ctx.recon_dist_and_rate)(
            recon,
            &ReconDistInput {
                quantization: ReconQuantization {
                    rate_log2_lut: ctx.rate_log2_lut,
                    coeffs,
                    inverse_matrices: [
                        inverse_matrix_for(ctx, strategy, 0),
                        inverse_matrix_for(ctx, strategy, 1),
                        inverse_matrix_for(ctx, strategy, 2),
                    ],
                    qac,
                    qm_mult_x,
                    distance,
                },
                transform: ReconTransform {
                    blocks_x: cx,
                    blocks_y: cy,
                    strategy,
                },
                source: ReconSource {
                    opsin,
                    x: px,
                    y: py,
                },
                scoring: ReconScoring {
                    factor_x: cmap_factor[0],
                    factor_b: cmap_factor[2],
                    banding: ctx.banding_protection,
                },
            },
            &ctx.recon_error_kernels,
        ),
        DistortionModel::Coefficient => coefficient_dist_and_rate(
            ctx, strategy, coeffs, size, qac, qm_mult_x, distance, cx, cy,
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
            const RECIP_RERANKING: f32 = 1. / (RERANK_LAMBDA_D1 - RERANK_LAMBDA_D0);
            let ramp = ((distance - RERANK_LAMBDA_D0) * RECIP_RERANKING).clamp(0.0, 1.0);
            let multiplier = fmla(ramp, RERANK_LAMBDA_HI - RERANK_LAMBDA_LO, RERANK_LAMBDA_LO);
            RD_LAMBDA * multiplier
        }
    };
    // fmla, matching `sub8_strategy_costs::evaluate` bit-for-bit — the sub8
    // differential test compares the two paths' costs exactly.
    fmla(lam, r_total + meta_r, d_total)
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
    let dst: &mut [f32; 64] = output.first_chunk_mut::<64>().unwrap();
    let input = DctInput::from_flat(input);
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
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    meta_r: f32,
    distance: f32,
    cmap_factor: [f32; 3],
    cached_dct8: Option<f32>,
) -> Sub8Costs {
    let mut pixels = [[0.0f32; 64]; 3];
    for (c, input) in pixels.iter_mut().enumerate() {
        gather_pixels(opsin.plane(c), px, py, 8, 8, input);
    }

    let coeffs = &mut scratch.strategy_coeffs;
    let mut evaluate = |strategy| {
        for c in 0..3 {
            forward_sub8_transform(ctx, strategy, &pixels[c], &mut coeffs[c]);
        }
        let [x, y, b] = &mut **coeffs;
        apply_cfl(ctx, CflXyb { x, y, b }, 64, cmap_factor);
        let (distortion, rate) =
            coefficient_dist_and_rate(ctx, strategy, coeffs, 64, qac, qm_mult_x, distance, 1, 1);
        fmla(RD_LAMBDA, rate + meta_r, distortion)
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
}

#[inline]
fn cmap_factors(ytox_map: &ImageSB, ytob_map: &ImageSB, bx: usize, by: usize) -> [f32; 3] {
    let tx = (bx / 8).min(ytox_map.xsize() - 1);
    let ty = (by / 8).min(ytox_map.ysize() - 1);
    [
        crate::color_correlation::y_to_x_ratio(ytox_map.row(ty)[tx]),
        0.0,
        crate::color_correlation::y_to_b_ratio(ytob_map.row(ty)[tx]),
    ]
}

#[allow(clippy::too_many_arguments)]
fn select_super_block(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
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
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
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
                scratch,
                STRATEGY_DCT,
                opsin,
                px0 + dx * 8,
                py0 + dy * 8,
                qac[dy][dx],
                qm_mult_x,
                meta_r,
                distance,
                cmap_factor,
            );
            scratch.dct8_costs[(by0 + dy - dct8_cost_y0) * dct8_cost_stride + bx0 + dx] =
                c8[dy][dx];
        }
    }

    // Vertical pairs (DCT16X8): one per column. Skipped entirely under
    // `SearchScope::Squares` — four `strategy_cost` calls per super-block.
    let merge = ctx.merge;
    let mut rect_cost = |px: usize, py: usize, strategy: u8, qac: f32| -> f32 {
        if !scope.rectangles() {
            return f32::INFINITY;
        }
        merge.bias_rect
            * strategy_cost(
                ctx,
                scratch,
                strategy,
                opsin,
                px,
                py,
                qac,
                qm_mult_x,
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
    let c16 = merge.bias_16x16
        * strategy_cost(
            ctx,
            scratch,
            STRATEGY_DCT16X16,
            opsin,
            px0,
            py0,
            aggregate_qac_2x2(qac, qac_scale, distance),
            qm_mult_x,
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
        && merge_beats_dct8(v_left, dct8_left, merge.accept_pair);
    let use_v_right = ac_strategy.can_place_strategy(bx0 + 1, by0, STRATEGY_DCT16X8)
        && merge_beats_dct8(v_right, dct8_right, merge.accept_pair);
    let use_h_top = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_top, dct8_top, merge.accept_pair);
    let use_h_bottom = ac_strategy.can_place_strategy(bx0, by0 + 1, STRATEGY_DCT8X16)
        && merge_beats_dct8(h_bot, dct8_bottom, merge.accept_pair);

    let cost_16x8 = if use_v_left { v_left } else { dct8_left }
        + if use_v_right { v_right } else { dct8_right };
    let cost_8x16 =
        if use_h_top { h_top } else { dct8_top } + if use_h_bottom { h_bot } else { dct8_bottom };
    let best_rect = cost_16x8.min(cost_8x16);

    let pick_16x16 = ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X16)
        && c16 < best_rect
        && merge_beats_dct8(c16, total_dct8, merge.accept_16);

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
            quant_field.row_mut(y + iy)[x..x + cov_x].fill(val);
        }
    }
}

#[inline]
fn quant_refinement_steps(distance: f32) -> usize {
    if !(1.5..5.0).contains(&distance) {
        0
    } else if (2.0..3.5).contains(&distance) {
        2
    } else {
        1
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_quant_field(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    distance: f32,
    scale: f32,
    qm_mult_x: f32,
    quant_field: &mut ImageB,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &AcStrategyImage,
) {
    let steps = quant_refinement_steps(distance);
    if steps == 0 {
        return;
    }
    for (bx, by, strategy) in ac_strategy.iter_first_blocks() {
        let cov_x = AcStrategyImage::covered_blocks_x_of(strategy);
        let cov_y = AcStrategyImage::covered_blocks_y_of(strategy);
        let current_q = quant_field.row(by)[bx];
        if current_q <= 1 {
            continue;
        }
        let cmap = cmap_factors(ytox_map, ytob_map, bx, by);
        let px = dc_group_px + bx * 8;
        let py = dc_group_py + by * 8;
        let mut cost = |q: u8| {
            let qac = scale * q as f32;
            reconstruction_strategy_cost(
                ctx, scratch, strategy, opsin, px, py, qac, qm_mult_x, 0.0, distance, cmap,
            )
        };
        let mut best_q = current_q;
        let mut best_cost = cost(current_q);
        // Bidirectional: q-1/q-2 can only save rate on over-spent blocks; q+1/q+2
        // let the field *spend* a step where the reconstruction says it is cheap.
        let candidates = if steps == 2 {
            // Upward stays at +1: q+2 measurably over-spends (its metadata-rate
            // cost is not priced here), while the -2 rate save still pays.
            [
                current_q.saturating_sub(1),
                current_q.saturating_sub(2),
                current_q.saturating_add(1),
                current_q,
            ]
        } else {
            [
                current_q.saturating_sub(1),
                current_q.saturating_add(1),
                current_q,
                current_q,
            ]
        };
        for candidate in candidates {
            if candidate == 0 || candidate == best_q || candidate == current_q {
                continue;
            }
            let candidate_cost = cost(candidate);
            if candidate_cost < best_cost {
                best_q = candidate;
                best_cost = candidate_cost;
            }
        }
        if best_q != current_q {
            for iy in 0..cov_y {
                quant_field.row_mut(by + iy)[bx..bx + cov_x].fill(best_q);
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
    scratch: &mut CoderScratch,
    meta_r: f32,
    distance: f32,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    scale: f32,
    qm_mult_x: f32,
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
    let merge = ctx.merge;
    // First-pass DCT8 incumbents are consumed again by the sub-8 refinement.
    // Keep only this worker's row band so parallel selection stays independent.
    let costs_size = xsize * (y_end - y_begin);
    if scratch.dct8_costs.len() != xsize * (y_end - y_begin) {
        scratch.dct8_costs.clear();
        scratch.dct8_costs.resize(costs_size, f32::NAN);
    } else {
        scratch.dct8_costs.fill(f32::NAN);
    }
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
                            scratch,
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
                            ytox_map,
                            ytob_map,
                            ac_strategy,
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
                    scratch,
                    STRATEGY_DCT32X32,
                    opsin,
                    dc_group_px + bx * 8,
                    dc_group_py + by * 8,
                    qac32,
                    qm_mult_x,
                    meta_r,
                    distance,
                    cmap_factor,
                );
                // Two DCT32X16 (each 2 wide x 4 tall) tiling the region: left +
                // right, and two DCT16X32 (each 4 wide x 2 tall): top + bottom.
                // Skipped under `SearchScope::Squares`.
                let mut rect32 =
                    |bx: usize, by: usize, strategy: u8, cw: usize, ch: usize| -> f32 {
                        if !scope.rectangles() {
                            return f32::INFINITY;
                        }
                        strategy_cost(
                            ctx,
                            scratch,
                            strategy,
                            opsin,
                            dc_group_px + bx * 8,
                            dc_group_py + by * 8,
                            region_qac(quant_field, bx, by, cw, ch, scale, distance),
                            qm_mult_x,
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
                    merge.bias_32x32 * cost32
                } else {
                    f32::INFINITY
                };
                let cost_32x16 = if can_32x16 {
                    merge.bias_rect32 * (cl + cr)
                } else {
                    f32::INFINITY
                };
                let cost_16x32 = if can_16x32 {
                    merge.bias_rect32 * (ct + cb)
                } else {
                    f32::INFINITY
                };

                let (best_big, best_strategy, accept) =
                    if cost_32x32 <= cost_32x16 && cost_32x32 <= cost_16x32 {
                        (cost_32x32, STRATEGY_DCT32X32, merge.accept_32)
                    } else if cost_32x16 <= cost_16x32 {
                        (cost_32x16, STRATEGY_DCT32X16, merge.accept_32_rect)
                    } else {
                        (cost_16x32, STRATEGY_DCT16X32, merge.accept_32_rect)
                    };

                // Compare against both the already-selected subdivision and the
                // pure DCT8 incumbent. The latter prevents a sequence of locally
                // marginal merges from making a 32×32 merge look trustworthy.
                if best_big < sub_total && merge_beats_dct8(best_big, dct8_total, accept) {
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
                        scratch,
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
                        ytox_map,
                        ytob_map,
                        ac_strategy,
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
                    scratch,
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
                    ytox_map,
                    ytob_map,
                    ac_strategy,
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
    if distance > SUB8_MAX_DISTANCE {
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
            let cached_dct8 = scratch.dct8_costs[(by - y_begin) * xsize + bx];
            let costs = sub8_strategy_costs(
                ctx,
                scratch,
                opsin,
                px,
                py,
                qac,
                qm_mult_x,
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
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    distance: f32,
    scale: f32,
    x_qm_scale: u32,
    quant_field: &mut ImageB,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
    num_threads: usize,
) -> f32 {
    let speed = ctx.speed;
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    // DCT8 wins the high-quality RD comparison outright; Fastest skips the
    // search by contract. Either way the default strategy image (all DCT8
    // first blocks) is already the answer.
    if use_dct8_only(distance) || speed == crate::Speed::Fastest {
        return 0.0;
    }
    let scope = SearchScope::for_speed(speed);
    let qm_mult_x = 1.25f32.powf(x_qm_scale as f32 - 2.0);
    // Per-candidate-block metadata rate for the strategy chooser (bits),
    // faded in above d=1 (see strategy_cost).
    let meta_r = META_R;

    let bands = if num_threads > 1 && ysize >= 8 {
        selection_bands(ysize, num_threads)
    } else {
        vec![(0, ysize)]
    };

    let benefit = if bands.len() <= 1 {
        select_band(
            ctx,
            scratch,
            meta_r,
            distance,
            opsin,
            dc_group_px,
            dc_group_py,
            scale,
            qm_mult_x,
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
        let results = ctx.thread_pool.steal_map_with_threads(
            scratch,
            bands.len(),
            num_threads,
            |i, scratch| {
                let (y0, y1) = bands_ref[i];
                let mut local = AcStrategyImage::new(xsize, ysize);
                let b = select_band(
                    ctx,
                    scratch,
                    meta_r,
                    distance,
                    opsin,
                    dc_group_px,
                    dc_group_py,
                    scale,
                    qm_mult_x,
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
            },
        );
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

    // The SSIM reconstruction rerank runs in both scopes. Under
    // `SearchScope::Squares` the only merges present are DCT16X16 and
    // DCT32X32, so it scores exactly those.
    if scope.rerank(distance) {
        rerank_large_transforms(
            ctx,
            scratch,
            opsin,
            dc_group_px,
            dc_group_py,
            distance,
            scale,
            qm_mult_x,
            meta_r,
            quant_field,
            ytox_map,
            ytob_map,
            ac_strategy,
        );
    }

    adjust_quant_field(ac_strategy, distance, quant_field);
    refine_quant_field(
        ctx,
        scratch,
        opsin,
        dc_group_px,
        dc_group_py,
        distance,
        scale,
        qm_mult_x,
        quant_field,
        ytox_map,
        ytob_map,
        ac_strategy,
    );
    benefit
}

/// Reconstruction-based rerank pass: for each selected merge, compare its
/// SSIM-reconstruction cost against the tiled DCT8 and downgrade if DCT8 wins.
#[allow(clippy::too_many_arguments)]
fn rerank_large_transforms(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    distance: f32,
    scale: f32,
    qm_mult_x: f32,
    meta_r: f32,
    quant_field: &ImageB,
    ytox_map: &ImageSB,
    ytob_map: &ImageSB,
    ac_strategy: &mut AcStrategyImage,
) {
    let rerank_margin = ctx.merge.rerank_margin;
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
            scratch,
            strat,
            opsin,
            px,
            py,
            qac_big,
            qm_mult_x,
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
                    scratch,
                    STRATEGY_DCT,
                    opsin,
                    px + ix * 8,
                    py + iy * 8,
                    q,
                    qm_mult_x,
                    meta_r,
                    distance,
                    cmap_factors(ytox_map, ytob_map, bx + ix, by + iy),
                );
            }
        }
        if j_dct8 < j_big * rerank_margin {
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
        BIAS_4X4, BIAS_4X8, BIAS_16X16, BIAS_RECT32, DCT8_ONLY_MAX_DISTANCE,
        FAST_RERANK_MAX_DISTANCE, MERGE_MARGIN_16, MERGE_MARGIN_32, MERGE_MARGIN_PAIR, MergeTuning,
        RERANK_DOWNGRADE_MARGIN, SUB8_MAX_DISTANCE, SearchScope, aggregate_qac_2x2,
        aggregate_quant, cmap_factors, fill_ac_strategy, merge_beats_dct8, merge_margin,
        quant_refinement_steps, strategy_cost, sub8_strategy_costs, use_dct8_only,
    };
    use crate::coder_scratch::CoderScratch;
    use crate::dc_group_data::{
        AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4,
        STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT32X32,
    };
    use crate::encoding_context::EncodingContext;
    use crate::image::{Image3F, ImageB, ImageSB};
    use crate::inflated_cost::{
        forward_for, forward_matrix, reconstruct_error, strategy_pixel_count,
    };

    /// Merges are searched at every usable distance; only the near-lossless
    /// tail is DCT8-only. The high-quality band is held back by the banded
    /// margins below, not by this gate.
    #[test]
    fn dct8_only_covers_the_near_lossless_tail_only() {
        assert_eq!(DCT8_ONLY_MAX_DISTANCE, 0.056_713_393);
        assert!(use_dct8_only(0.01));
        assert!(use_dct8_only(0.05));
        assert!(!use_dct8_only(0.06));
        assert!(!use_dct8_only(0.2));
        assert!(!use_dct8_only(1.0));
    }

    /// A banded knob holds each end flat outside the band and interpolates
    /// inside it, so a value fitted in one band cannot move the other.
    #[test]
    fn banded_knobs_are_flat_outside_the_band() {
        assert_eq!(MERGE_MARGIN_16.at(0.1), MERGE_MARGIN_16.at(0.5));
        assert_eq!(MERGE_MARGIN_16.at(1.5), MERGE_MARGIN_16.at(6.0));
        let mid = MERGE_MARGIN_16.at(1.0);
        assert!((mid - 0.5 * (0.192_438_88 + 0.08)).abs() < 1e-6);
        // High quality admits merges on much stiffer terms than mid/low.
        assert!(MERGE_MARGIN_16.at(0.3) > MERGE_MARGIN_16.at(2.0));
        assert!(BIAS_16X16.at(0.3) > BIAS_16X16.at(2.0));
        // The rerank defends merges at mid/low quality, knife-edge at HQ.
        assert_eq!(RERANK_DOWNGRADE_MARGIN.at(0.3), 1.0);
        assert!(RERANK_DOWNGRADE_MARGIN.at(3.0) < 1.0);
    }

    #[test]
    fn quant_refinement_is_gated_and_conservative_at_boundaries() {
        assert_eq!(quant_refinement_steps(1.0), 0);
        assert_eq!(quant_refinement_steps(1.5), 1);
        assert_eq!(quant_refinement_steps(2.0), 2);
        assert_eq!(quant_refinement_steps(3.49), 2);
        assert_eq!(quant_refinement_steps(3.5), 1);
        assert_eq!(quant_refinement_steps(4.99), 1);
        assert_eq!(quant_refinement_steps(5.0), 0);
    }

    /// The Fast tier runs the same RD model but offers only square merges.
    /// A flat 32x32 region must still merge, and must never come back as a
    /// rectangle, a sub-8x8 split, or a 64px transform.
    #[test]
    fn fast_scope_selects_squares_and_no_other_merge_shape() {
        let ctx = EncodingContext::new(
            crate::Speed::Fast,
            None,
            false,
            crate::xyb::XybMatrix::SPEC,
            1.0,
            1,
        );
        let maps = ImageSB::new_fill(1, 1, 0);
        let opsin = Image3F::new(32, 32);
        let mut qf = ImageB::new_fill(4, 4, 8);
        let mut strategies = AcStrategyImage::new(4, 4);
        let mut scratch = CoderScratch::default();
        fill_ac_strategy(
            &ctx,
            &mut scratch,
            &opsin,
            0,
            0,
            3.0,
            1.0,
            2,
            &mut qf,
            &maps,
            &maps,
            &mut strategies,
            1,
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

    /// The sub-8 biases are a fitted pair, not incidental defaults: both are
    /// above 1.0 (making sub-8 harder to select than the raw RD comparison
    /// would), and the family is gated off past the mid band. Guards a silent
    /// revert to the unfitted flat-1.0/no-gate behaviour, which measured
    /// +0.238% BD-rate combined across Kodak and train0.
    #[test]
    fn sub8_selection_is_fitted_not_neutral() {
        // Read through runtime bindings so the checks are not const-folded away.
        let (b44, b48, gate) = (
            std::hint::black_box(BIAS_4X4),
            std::hint::black_box(BIAS_4X8),
            std::hint::black_box(SUB8_MAX_DISTANCE),
        );
        assert!(b44 > 1.0, "BIAS_4X4 = {b44}");
        assert!(b48 > 1.0, "BIAS_4X8 = {b48}");
        assert!(b44 > b48, "4X4 carries the stiffer bar");
        assert!(
            (0.8..=2.5).contains(&gate),
            "gate {gate} outside the band the corpora agree on"
        );
        // The gate must sit above the DCT8-only floor, or sub-8 would never run.
        assert!(gate > std::hint::black_box(DCT8_ONLY_MAX_DISTANCE));
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

        let ctx = EncodingContext::default();
        let (px, py) = (7usize, 9usize); // exercise right and bottom replication
        let qac = 3.75;
        let qm_mult_x = 1.25;
        let meta_r = 0.4;
        let distance = 1.5;
        let cmap = [0.25, 0.0, -0.375];
        let mut scratch = CoderScratch::default();
        let independent = [
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT4X4,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT4X8,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
            strategy_cost(
                &ctx,
                &mut scratch,
                STRATEGY_DCT8X4,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
            ),
        ];

        for cached in [None, Some(independent[0])] {
            let bundled = sub8_strategy_costs(
                &ctx,
                &mut scratch,
                &opsin,
                px,
                py,
                qac,
                qm_mult_x,
                meta_r,
                distance,
                cmap,
                cached,
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
        assert!(merge_margin(0.3, MERGE_MARGIN_32) > merge_margin(0.3, MERGE_MARGIN_16));
        assert!(merge_margin(0.3, MERGE_MARGIN_16) > merge_margin(0.3, MERGE_MARGIN_PAIR));
        assert!(merge_margin(0.3, MERGE_MARGIN_16) > merge_margin(4.0, MERGE_MARGIN_16));

        // At high quality the 19% 16x16 margin rejects a 10% estimated win but
        // accepts a clear 25% one. At coarse quality the guard fades.
        let hq = MergeTuning::new(0.3);
        let lowq = MergeTuning::new(4.0);
        assert!(!merge_beats_dct8(90.0, 100.0, hq.accept_16));
        assert!(merge_beats_dct8(75.0, 100.0, hq.accept_16));
        assert!(merge_beats_dct8(95.0, 100.0, lowq.accept_16));

        // The resolved table is exactly what the per-block path used to compute.
        assert_eq!(hq.accept_16, 1.0 - merge_margin(0.3, MERGE_MARGIN_16));
        assert_eq!(hq.bias_rect32, BIAS_RECT32.at(0.3));
        assert_eq!(lowq.rerank_margin, RERANK_DOWNGRADE_MARGIN.at(4.0));
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
    fn strategy_cfl_factors_match_signaled_tile_maps() {
        let ytox = ImageSB::new_fill(1, 1, 42);
        let ytob = ImageSB::new_fill(1, 1, -42);
        assert_eq!(cmap_factors(&ytox, &ytob, 0, 0), [0.5, 0.0, 0.5]);
    }
}
