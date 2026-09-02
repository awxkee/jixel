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

use crate::adaptive_quant::fast_exp2;
use crate::coder_scratch::{
    AcStrategyBandScratch, CachedQuantCost, CoderScratch, QuantRefinement, RerankDowngrade,
};
use crate::dc_group_data::{
    AcStrategyImage, STRATEGY_AFV0, STRATEGY_AFV1, STRATEGY_AFV2, STRATEGY_AFV3, STRATEGY_DCT,
    STRATEGY_DCT2X2, STRATEGY_DCT4X4, STRATEGY_DCT4X8, STRATEGY_DCT8X4, STRATEGY_DCT8X16,
    STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16, STRATEGY_DCT32X32,
    STRATEGY_DCT32X64, STRATEGY_DCT64X32, STRATEGY_DCT64X64, STRATEGY_IDENTITY,
};
use crate::dct::{DctInput, fmla};
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, ImageB, ImageSB};
use crate::inflated_cost::{
    ReconDistInput, ReconQuantization, ReconScoring, ReconSource, ReconTransform, channel_rd,
};

mod selection;

pub(crate) use selection::{Chosen32Cost, SavedChild, fill_ac_strategy};

const DCT8_ONLY_MAX_DISTANCE: f32 = 0.056_713_393;

/// Above this distance the [`SearchScope::Squares`] tier stops reranking; see
/// [`SearchScope::rerank`].
const FAST_RERANK_MAX_DISTANCE: f32 = 1.7090991223128462;

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

pub(crate) const RD_LAMBDA: f32 = 0.080_867_17;

const RERANK_META_R: f32 = 2.6465739748323758;
const RERANK_DOWNGRADE_MARGIN: Banded = Banded::new(1.0, 0.9098454411056834);

const RERANK_LAMBDA_LO: f32 = 0.25;
const RERANK_LAMBDA_HI: f32 = 10.0;
const RERANK_LAMBDA_D0: f32 = 1.0;
const RERANK_LAMBDA_D1: f32 = 4.0;

const BIAS_RECT: Banded = Banded::new(1.026_200_7, 0.915);
const BIAS_16X16: Banded = Banded::new(1.205_927_5, 1.05);
const BIAS_32X32: Banded = Banded::new(1.205_927_5, 1.06);
const BIAS_RECT32: Banded = Banded::new(1.326_520_2, 1.02);

const MERGE_MARGIN_PAIR: Banded = Banded::new(0.036_252_015, 0.072);
const MERGE_MARGIN_16: Banded = Banded::new(0.192_438_88, 0.048);
const MERGE_MARGIN_32_RECT: Banded = Banded::new(0.161_916_72, 0.29);
const MERGE_MARGIN_32: Banded = Banded::new(0.329_207_82, 0.36);

const MERGE_MARGIN_LOWQ_FRACTION: f32 = 0.82;
const MERGE_MARGIN_LOWQ3_D: f32 = 3.85;
const MERGE_MARGIN_LOWQ3_SCALE: f32 = 1.10;
const MERGE_MARGIN_FADE_START: f32 = 1.0;
const MERGE_MARGIN_FADE_END: f32 = 4.0;
const BIAS_4X4: f32 = 1.33;
const BIAS_4X8: f32 = 1.09;
const BIAS_AFV: Banded = Banded::new(1.15, 1.25);

/// Sub-8 is skipped entirely above this distance.
const SUB8_MAX_DISTANCE: f32 = 1.5;
/// AFV keeps competing (alone — the DCT4 family stays gated at
/// [`SUB8_MAX_DISTANCE`]) up to this distance. At the default the extension
/// band is empty and AFV shares the sub-8 gate exactly.
const AFV_MAX_DISTANCE: f32 = SUB8_MAX_DISTANCE;
/// IDENTITY and DCT2X2 remain useful much farther into the lossy range: unlike
/// the DCT4/AFV family they avoid spreading a thin one-pixel chroma feature
/// over an entire basis function. This mirrors libjxl's search range.
const FINE_TRANSFORM_MAX_DISTANCE: f32 = 5.0;
/// Reconstruction-domain admission margin: a fine candidate must beat the
/// DCT8 incumbent's reconstruction cost by this factor.
const FINE_RECON_MARGIN: f32 = 0.98;

#[inline]
fn fine_transform_bias(base: f32, distance: f32) -> f32 {
    // libjxl normalizes the small-transform entropy multipliers by DCT8's 0.8
    // and explicitly favors IDENTITY/DCT2X2 below distance 5.
    let favor = if distance < 5.0 {
        let t = (5.0 - distance) * 0.2;
        0.4 * t * t
    } else {
        0.0
    };
    base / 0.8 - favor
}
#[inline]
fn merge_margin(distance: f32, margin: Banded) -> f32 {
    let fade = ((distance - MERGE_MARGIN_FADE_START)
        / (MERGE_MARGIN_FADE_END - MERGE_MARGIN_FADE_START))
        .clamp(0.0, 1.0);
    let lowq3 = if distance >= MERGE_MARGIN_LOWQ3_D {
        MERGE_MARGIN_LOWQ3_SCALE
    } else {
        1.0
    };
    margin.at(distance) * (1.0 - fade * (1.0 - MERGE_MARGIN_LOWQ_FRACTION)) * lowq3
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
    /// Heterogeneity merge-risk gate strength (see [`MERGE_RISK_K`]).
    pub(crate) risk_k: f32,
    /// DCT64-family acceptance factors (see [`ACCEPT_64`]/[`ACCEPT_64_RECT`]).
    pub(crate) accept_64: f32,
    pub(crate) accept_64_rect: f32,
}

/// Very-low-quality band
const VLQ_D0: f32 = 4.0;
const VLQ_D1: f32 = 7.5;
const VLQ_MARGIN_PAIR: f32 = -0.005;
const VLQ_MARGIN_16: f32 = 0.067;
const VLQ_MARGIN_32_RECT: f32 = 0.363;
const VLQ_MARGIN_32: f32 = 0.045;
const VLQ_BIAS_RECT: f32 = 1.073;
const VLQ_BIAS_16X16: f32 = 0.880;
const VLQ_BIAS_32X32: f32 = 0.867;
const VLQ_BIAS_RECT32: f32 = 1.180;
const VLQ_RERANK_MARGIN: f32 = 0.782;
const VLQ_ACCEPT_64: f32 = 0.973;
const VLQ_ACCEPT_64_RECT: f32 = 0.736;

impl MergeTuning {
    pub(crate) fn new(distance: f32) -> Self {
        let accept = |margin: Banded| 1.0 - merge_margin(distance, margin);
        let mut tuning = Self {
            bias_rect: BIAS_RECT.at(distance),
            bias_16x16: BIAS_16X16.at(distance),
            bias_32x32: BIAS_32X32.at(distance),
            bias_rect32: BIAS_RECT32.at(distance),
            accept_pair: accept(MERGE_MARGIN_PAIR),
            accept_16: accept(MERGE_MARGIN_16),
            accept_32_rect: accept(MERGE_MARGIN_32_RECT),
            accept_32: accept(MERGE_MARGIN_32),
            rerank_margin: RERANK_DOWNGRADE_MARGIN.at(distance),
            risk_k: if crate::adaptive_quant::aq_dampen(distance) > 0.0 {
                MERGE_RISK_K
            } else {
                0.0
            },
            accept_64: dct64_accept(),
            accept_64_rect: dct64_rect_accept(),
        };
        let vlq_t = ((distance - VLQ_D0) / (VLQ_D1 - VLQ_D0)).clamp(0.0, 1.0);
        if vlq_t > 0.0 {
            let lerp = |cur: f32, vlq: f32| fmla(vlq_t, vlq - cur, cur);
            tuning.bias_rect = lerp(tuning.bias_rect, VLQ_BIAS_RECT);
            tuning.bias_16x16 = lerp(tuning.bias_16x16, VLQ_BIAS_16X16);
            tuning.bias_32x32 = lerp(tuning.bias_32x32, VLQ_BIAS_32X32);
            tuning.bias_rect32 = lerp(tuning.bias_rect32, VLQ_BIAS_RECT32);
            tuning.accept_pair = lerp(tuning.accept_pair, 1.0 - VLQ_MARGIN_PAIR);
            tuning.accept_16 = lerp(tuning.accept_16, 1.0 - VLQ_MARGIN_16);
            tuning.accept_32_rect = lerp(tuning.accept_32_rect, 1.0 - VLQ_MARGIN_32_RECT);
            tuning.accept_32 = lerp(tuning.accept_32, 1.0 - VLQ_MARGIN_32);
            tuning.rerank_margin = lerp(tuning.rerank_margin, VLQ_RERANK_MARGIN);
            tuning.accept_64 = lerp(tuning.accept_64, VLQ_ACCEPT_64);
            tuning.accept_64_rect = lerp(tuning.accept_64_rect, VLQ_ACCEPT_64_RECT);
        }
        tuning
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
    tmp: &'a mut [f32],
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
    tmp: &mut [f32],
    strategy: u8,
    plane: &crate::image::Plane<f32>,
    px: usize,
    py: usize,
    out: &mut [f32],
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
        STRATEGY_IDENTITY => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.identity8x8)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        STRATEGY_DCT2X2 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.dct2x2_8x8)(dct_input(plane, tmp, px, py), dst);
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
        STRATEGY_AFV0 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.afv0)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        STRATEGY_AFV1 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.afv1)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        STRATEGY_AFV2 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.afv2)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        STRATEGY_AFV3 => {
            let dst: &mut [f32; 64] = out.first_chunk_mut::<64>().unwrap();
            (ctx.afv3)(dct_input(plane, tmp, px, py), dst);
            (1, 1)
        }
        _ => unreachable!("invalid strategy {strategy}"),
    }
}

const DCT64_MIN_DISTANCE: f32 = 3.0;
const DCT64_WINDOW_LO: f32 = 1.0;
const DCT64_WINDOW_HI: f32 = 2.25;
const BIAS_64X64: f32 = 1.0;
const BIAS_64_RECT: f32 = 1.0;
const ACCEPT_64: f32 = 0.945;
const ACCEPT_64_RECT: f32 = 0.642;

const fn dct64_accept() -> f32 {
    ACCEPT_64
}

const fn dct64_rect_accept() -> f32 {
    ACCEPT_64_RECT
}

#[inline]
fn use_dct64(speed: crate::Speed, distance: f32) -> bool {
    speed == crate::Speed::Slow
        && (distance >= DCT64_MIN_DISTANCE
            || (DCT64_WINDOW_LO..DCT64_WINDOW_HI).contains(&distance))
}

/// DCT64 is evaluated outside the standard chooser, reusing the chooser's
/// lazily allocated coefficient and gather buffers at their maximum footprint.
#[allow(clippy::too_many_arguments)]
fn strategy_cost64(
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
    let CoderScratch {
        strategy_coeffs: coeffs,
        transform_gather: input,
        ..
    } = scratch;
    let coeffs: &mut [[f32; 4096]; 3] = coeffs;
    let input: &mut [f32; 4096] = input;
    let (width, height, size, cx, cy) = match strategy {
        STRATEGY_DCT64X64 => (64, 64, 4096, 8, 8),
        STRATEGY_DCT64X32 => (32, 64, 2048, 8, 4),
        STRATEGY_DCT32X64 => (64, 32, 2048, 8, 4),
        _ => unreachable!("not a DCT64-family strategy: {strategy}"),
    };
    for (c, coeff) in coeffs.iter_mut().enumerate() {
        gather_pixels(opsin.plane(c), px, py, width, height, &mut input[..size]);
        match strategy {
            STRATEGY_DCT64X64 => (ctx.dct64x64)(DctInput::from_flat(input), coeff),
            STRATEGY_DCT64X32 => (ctx.dct64x32)(
                DctInput::from_flat(input.first_chunk::<2048>().unwrap()),
                coeff.first_chunk_mut::<2048>().unwrap(),
            ),
            STRATEGY_DCT32X64 => (ctx.dct32x64)(
                DctInput::from_flat(input.first_chunk::<2048>().unwrap()),
                coeff.first_chunk_mut::<2048>().unwrap(),
            ),
            _ => unreachable!(),
        }
    }
    let [x, y, b] = coeffs;
    apply_cfl(ctx, CflXyb { x, y, b }, size, cmap_factor);

    let mut distortion = 0.0f32;
    let mut rate = 0.0f32;
    for (c, coeff) in coeffs.iter().enumerate() {
        let qm_mult = if c == 0 {
            qm_mult_x
        } else if c == 2 {
            ctx.b_qm_mul()
        } else {
            1.0
        };
        let matrix: &[f32] = match strategy {
            STRATEGY_DCT64X64 => &ctx.matrices().inv_matrix_64x64(c)[..],
            STRATEGY_DCT64X32 | STRATEGY_DCT32X64 => &ctx.matrices().inv_matrix_64x32(c)[..],
            _ => unreachable!(),
        };
        let (d, r) = channel_rd(
            ctx.sse_and_rate,
            ctx.rate_log2_lut,
            &coeff[..size],
            matrix,
            c,
            qac,
            qm_mult,
            distance,
            cx,
            cy,
        );
        distortion += ctx.channel_weight(c) * d;
        rate += r;
    }
    distortion + RD_LAMBDA * (rate + meta_r)
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

/// Returns the reconstruction RD cost both with the caller's metadata charge
/// and without it. The latter can be reused as quant refinement's incumbent.
#[allow(clippy::too_many_arguments)]
fn reconstruction_strategy_cost_and_base(
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
    gradient_alpha: f32,
    gradient_peak_alpha: f32,
) -> (f32, f32) {
    let CoderScratch {
        strategy_coeffs: coeffs,
        transform_gather,
        recon,
        ..
    } = scratch;
    let (cx, cy, _) = prepare_strategy_coeffs(
        ctx,
        coeffs,
        transform_gather,
        strategy,
        opsin,
        px,
        py,
        cmap_factor,
    );
    let (distortion, rate) = reconstruction_dist_and_rate(
        ctx,
        recon,
        coeffs,
        strategy,
        opsin,
        px,
        py,
        qac,
        qm_mult_x,
        distance,
        cmap_factor,
        cx,
        cy,
        gradient_alpha,
        gradient_peak_alpha,
    );
    (
        rd_cost(
            DistortionModel::Reconstruction,
            distance,
            meta_r,
            distortion,
            rate,
        ),
        rd_cost(
            DistortionModel::Reconstruction,
            distance,
            0.0,
            distortion,
            rate,
        ),
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

#[cfg(not(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
)))]
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
    {
        |x, y, b, cmap_factor| unsafe { crate::neon::apply_cfl_neon(x, y, b, cmap_factor) }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |x, y, b, cmap_factor| unsafe { crate::avx::apply_cfl_avx2(x, y, b, cmap_factor) };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return |x, y, b, cmap_factor| crate::wasm::apply_cfl_wasm(x, y, b, cmap_factor);
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    apply_cfl_scalar
}

fn apply_cfl(ctx: &EncodingContext, coeffs: CflXyb<'_>, size: usize, cmap_factor: [f32; 3]) {
    let CflXyb { x, y, b } = coeffs;
    assert!(size <= x.len() && size <= y.len() && size <= b.len());
    (ctx.apply_cfl)(&mut x[..size], &y[..size], &mut b[..size], cmap_factor);
}

#[inline]
fn inverse_matrix_for(ctx: &EncodingContext, strategy: u8, channel: usize) -> &[f32] {
    let matrices = ctx.matrices();
    match strategy {
        STRATEGY_DCT => &matrices.inv_matrix(channel)[..],
        STRATEGY_IDENTITY => &matrices.inv_matrix_identity(channel)[..],
        STRATEGY_DCT2X2 => &matrices.inv_matrix_dct2x2(channel)[..],
        STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(channel)[..],
        STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(channel)[..],
        STRATEGY_AFV0..=STRATEGY_AFV3 => &matrices.inv_matrix_afv(channel)[..],
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
    coeffs: &[[f32; 4096]; 3],
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
            ctx.b_qm_mul()
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
        d_total += ctx.channel_weight(c) * d;
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
    let CoderScratch {
        strategy_coeffs: coeffs,
        transform_gather,
        recon,
        ..
    } = scratch;
    let (cx, cy, size) = prepare_strategy_coeffs(
        ctx,
        coeffs,
        transform_gather,
        strategy,
        opsin,
        px,
        py,
        cmap_factor,
    );

    let (d_total, r_total) = match distortion_model {
        DistortionModel::Reconstruction => reconstruction_dist_and_rate(
            ctx,
            recon,
            coeffs,
            strategy,
            opsin,
            px,
            py,
            qac,
            qm_mult_x,
            distance,
            cmap_factor,
            cx,
            cy,
            0.0,
            0.0,
        ),
        DistortionModel::Coefficient => coefficient_dist_and_rate(
            ctx, strategy, coeffs, size, qac, qm_mult_x, distance, cx, cy,
        ),
    };
    rd_cost(distortion_model, distance, meta_r, d_total, r_total)
}

#[allow(clippy::too_many_arguments)]
fn prepare_strategy_coeffs(
    ctx: &EncodingContext,
    coeffs: &mut [[f32; 4096]; 3],
    transform_gather: &mut [f32; 4096],
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    cmap_factor: [f32; 3],
) -> (usize, usize, usize) {
    let mut cxy = (1usize, 1usize);
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
    let [x, y, b] = coeffs;
    apply_cfl(ctx, CflXyb { x, y, b }, size, cmap_factor);
    (cx, cy, size)
}

#[allow(clippy::too_many_arguments)]
fn reconstruction_dist_and_rate(
    ctx: &EncodingContext,
    recon: &mut [[f32; 1024]; 8],
    coeffs: &[[f32; 4096]; 3],
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    distance: f32,
    cmap_factor: [f32; 3],
    cx: usize,
    cy: usize,
    gradient_alpha: f32,
    gradient_peak_alpha: f32,
) -> (f32, f32) {
    (ctx.recon_dist_and_rate)(
        recon,
        &ReconDistInput {
            idct: ctx.idct,
            quantization: ReconQuantization {
                rate_log2_lut: ctx.rate_log2_lut,
                coeffs: [&coeffs[0], &coeffs[1], &coeffs[2]],
                inverse_matrices: [
                    inverse_matrix_for(ctx, strategy, 0),
                    inverse_matrix_for(ctx, strategy, 1),
                    inverse_matrix_for(ctx, strategy, 2),
                ],
                qac,
                qm_mult_x,
                qm_mult_b: ctx.b_qm_mul(),
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
                channel_weights: ctx.channel_weights(),
                xyb_matrix: ctx.xyb,
                rgb_hue_alpha: rerank_rgb_hue_alpha(ctx, distance),
                gradient_alpha,
                gradient_peak_alpha,
            },
        },
        &ctx.recon_error_kernels,
    )
}

/// Pair-transform gradient protection used only by the reconstruction rerank.
/// The raw energy term handles small high-quality errors; the source-masked,
/// peak-pooled term takes over as quantization grows. Its coarse tail is limited
/// to bright, nearly neutral gradients, where blocking remains visible without
/// texture or chroma masking.
// Fitted on the sun-beam set, then checked on the general 768px corpus and Kodak.
const RERANK_PAIR_GRADIENT_FADE_IN_START: f32 = 0.5;
const RERANK_PAIR_GRADIENT_FADE_IN_END: f32 = 0.8;
const RERANK_PAIR_GRADIENT_FADE_OUT_START: f32 = 1.0;
const RERANK_PAIR_GRADIENT_FADE_OUT_END: f32 = 1.5;
const RERANK_PAIR_GRADIENT_ALPHA: f32 = 8.0;
const RERANK_PAIR_GRADIENT_MIN_DOMINANCE: f32 = 0.4;
const RERANK_PAIR_GRADIENT_PEAK_FADE_IN_START: f32 = 1.2;
const RERANK_PAIR_GRADIENT_PEAK_FADE_IN_END: f32 = 1.8;
const RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_START: f32 = 3.0;
const RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_END: f32 = 3.5;
const RERANK_PAIR_GRADIENT_PEAK_COARSE_START: f32 = 1.9;
const RERANK_PAIR_GRADIENT_PEAK_COARSE_END: f32 = 2.0;
const RERANK_PAIR_GRADIENT_PEAK_ALPHA: f32 = 192.0;
const RERANK_PAIR_GRADIENT_PEAK_COARSE_ALPHA: f32 = 96.0;
const RERANK_PAIR_GRADIENT_PEAK_MIN_DOMINANCE: f32 = 0.4;
const RERANK_PAIR_GRADIENT_PEAK_MIN_LUMA: f32 = 0.45;
const RERANK_PAIR_GRADIENT_PEAK_MAX_COARSE_CHROMA: f32 = 0.01;
// Fitted on Burning_Ship at d=0.5/1/1.25/1.5. The frame-level X-gradient
// gate keeps this entirely off the yellow-photo and Kodak guards.
const RERANK_RGB_HUE_ALPHA: f32 = 800.0;

fn rerank_rgb_hue_alpha(ctx: &EncodingContext, distance: f32) -> f32 {
    if !ctx.x_heavy() {
        return 0.0;
    }
    let fade_in = ((distance - 0.35) / 0.40).clamp(0.0, 1.0);
    let fade_out = 1.0 - ((distance - 1.25) / 0.50).clamp(0.0, 1.0);
    RERANK_RGB_HUE_ALPHA * fade_in * fade_out
}

#[inline]
fn rerank_pair_gradient_scale(distance: f32) -> f32 {
    let fade_in = ((distance - RERANK_PAIR_GRADIENT_FADE_IN_START)
        / (RERANK_PAIR_GRADIENT_FADE_IN_END - RERANK_PAIR_GRADIENT_FADE_IN_START))
        .clamp(0.0, 1.0);
    let fade_out = 1.0
        - ((distance - RERANK_PAIR_GRADIENT_FADE_OUT_START)
            / (RERANK_PAIR_GRADIENT_FADE_OUT_END - RERANK_PAIR_GRADIENT_FADE_OUT_START))
            .clamp(0.0, 1.0);
    fade_in * fade_out
}

fn rerank_pair_gradient_alpha(distance: f32) -> f32 {
    RERANK_PAIR_GRADIENT_ALPHA * rerank_pair_gradient_scale(distance)
}

#[inline]
fn rerank_pair_gradient_peak_alpha(distance: f32) -> f32 {
    let fade_in = ((distance - RERANK_PAIR_GRADIENT_PEAK_FADE_IN_START)
        / (RERANK_PAIR_GRADIENT_PEAK_FADE_IN_END - RERANK_PAIR_GRADIENT_PEAK_FADE_IN_START))
        .clamp(0.0, 1.0);
    let fade_out = 1.0
        - ((distance - RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_START)
            / (RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_END - RERANK_PAIR_GRADIENT_PEAK_FADE_OUT_START))
            .clamp(0.0, 1.0);
    let coarse_mix = ((distance - RERANK_PAIR_GRADIENT_PEAK_COARSE_START)
        / (RERANK_PAIR_GRADIENT_PEAK_COARSE_END - RERANK_PAIR_GRADIENT_PEAK_COARSE_START))
        .clamp(0.0, 1.0);
    let fit = &PAIR_GRAD_FIT;
    let alpha = fmla(
        coarse_mix,
        fit.peak_coarse_alpha - fit.peak_alpha,
        fit.peak_alpha,
    );
    alpha * fade_in * fade_out
}

struct PairGradFit {
    peak_alpha: f32,
    peak_coarse_alpha: f32,
    min_luma: f32,
    chroma_cap: f32,
    peak_min_dominance: f32,
}

static PAIR_GRAD_FIT: PairGradFit = PairGradFit {
    peak_alpha: RERANK_PAIR_GRADIENT_PEAK_ALPHA,
    peak_coarse_alpha: RERANK_PAIR_GRADIENT_PEAK_COARSE_ALPHA,
    min_luma: RERANK_PAIR_GRADIENT_PEAK_MIN_LUMA,
    chroma_cap: RERANK_PAIR_GRADIENT_PEAK_MAX_COARSE_CHROMA,
    peak_min_dominance: RERANK_PAIR_GRADIENT_PEAK_MIN_DOMINANCE,
};

#[inline]
fn rd_cost(
    distortion_model: DistortionModel,
    distance: f32,
    meta_r: f32,
    d_total: f32,
    r_total: f32,
) -> f32 {
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
    identity: f32,
    dct2x2: f32,
    dct4x4: f32,
    dct4x8: f32,
    dct8x4: f32,
    afv: [f32; 4],
}

#[inline]
fn forward_sub8_transform(
    ctx: &EncodingContext,
    strategy: u8,
    input: &[f32; 64],
    output: &mut [f32],
) {
    let dst: &mut [f32; 64] = output.first_chunk_mut::<64>().unwrap();
    let input = DctInput::from_flat(input);
    match strategy {
        STRATEGY_DCT => (ctx.dct8x8)(input, dst),
        STRATEGY_IDENTITY => (ctx.identity8x8)(input, dst),
        STRATEGY_DCT2X2 => (ctx.dct2x2_8x8)(input, dst),
        STRATEGY_DCT4X4 => (ctx.dct4x4)(input, dst),
        STRATEGY_DCT4X8 => (ctx.dct4x8)(input, dst),
        STRATEGY_DCT8X4 => (ctx.dct8x4)(input, dst),
        STRATEGY_AFV0 => (ctx.afv0)(input, dst),
        STRATEGY_AFV1 => (ctx.afv1)(input, dst),
        STRATEGY_AFV2 => (ctx.afv2)(input, dst),
        STRATEGY_AFV3 => (ctx.afv3)(input, dst),
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
    with_dct4: bool,
    with_fine: bool,
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
        let [x, y, b] = &mut ***coeffs;
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
    let identity = if with_fine {
        evaluate(STRATEGY_IDENTITY)
    } else {
        f32::INFINITY
    };
    let dct2x2 = if with_fine {
        evaluate(STRATEGY_DCT2X2)
    } else {
        f32::INFINITY
    };
    // In the AFV-only extension band the DCT4 family is out of contention;
    // infinite costs flow through the biased comparison and never win.
    let mut evaluate_dct4 = |strategy| {
        if with_dct4 {
            evaluate(strategy)
        } else {
            f32::INFINITY
        }
    };
    Sub8Costs {
        dct8,
        identity,
        dct2x2,
        dct4x4: evaluate_dct4(STRATEGY_DCT4X4),
        dct4x8: evaluate_dct4(STRATEGY_DCT4X8),
        dct8x4: evaluate_dct4(STRATEGY_DCT8X4),
        afv: std::array::from_fn(|kind| evaluate(STRATEGY_AFV0 + kind as u8)),
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

const MERGE_RISK_K: f32 = std::f32::consts::LOG2_E;

/// Fraction of the region's Y variance carried by the smooth >=4px-scale
/// component. Luminance ramps (sun beams, vignettes) score high; flat sky and
/// texture-dominated regions score low.
/// `w`/`h` must be multiples of 4 (transform sizes always are).
#[derive(Clone, Copy)]
pub(crate) struct GradientRegionStats {
    pub(crate) dominance: f32,
    pub(crate) mean: f32,
    pub(crate) chroma: f32,
}

pub(crate) type GradientRegionStatsFn =
    fn(&Image3F, usize, usize, usize, usize, f32) -> GradientRegionStats;

#[inline]
pub(crate) fn finish_gradient_region_stats(
    means: &[f32],
    within_sum: f32,
    chroma_sum: f32,
    pixel_count: usize,
    eps: f32,
) -> GradientRegionStats {
    let n_cells = means.len() as f32;
    let within = within_sum / n_cells;
    let grand = means.iter().sum::<f32>() / n_cells;
    let lf = means.iter().map(|m| (m - grand) * (m - grand)).sum::<f32>() / n_cells;
    GradientRegionStats {
        dominance: lf / (lf + within + eps),
        mean: grand,
        chroma: chroma_sum / pixel_count as f32,
    }
}

pub(crate) fn gradient_region_stats_scalar(
    opsin: &Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> GradientRegionStats {
    let xs = opsin.xsize();
    let ys = opsin.ysize();
    let cw = w / 4;
    let ch = h / 4;
    let mut means = [0.0f32; 256]; // 64x64 region -> 16x16 cells max
    let mut within = 0.0f32;
    for cy in 0..ch {
        for cx in 0..cw {
            let mut s = 0.0f32;
            let mut s2 = 0.0f32;
            for dy in 0..4 {
                let row = opsin.plane_row(1, (py0 + cy * 4 + dy).min(ys - 1));
                for dx in 0..4 {
                    let sx = (px0 + cx * 4 + dx).min(xs - 1);
                    let v = row[sx];
                    s += v;
                    s2 = fmla(v, v, s2);
                }
            }
            let m = s * (1.0 / 16.0);
            means[cy * cw + cx] = m;
            within += (s2 * (1.0 / 16.0) - m * m).max(0.0);
        }
    }
    finish_gradient_region_stats(&means[..cw * ch], within, 0.0, w * h, eps)
}

pub(crate) fn gradient_region_stats_with_chroma_scalar(
    opsin: &Image3F,
    px0: usize,
    py0: usize,
    w: usize,
    h: usize,
    eps: f32,
) -> GradientRegionStats {
    let xs = opsin.xsize();
    let ys = opsin.ysize();
    let cw = w / 4;
    let ch = h / 4;
    let mut means = [0.0f32; 256];
    let mut within = 0.0f32;
    let mut chroma = 0.0f32;
    for cy in 0..ch {
        for cx in 0..cw {
            let mut s = 0.0f32;
            let mut s2 = 0.0f32;
            for dy in 0..4 {
                let y = (py0 + cy * 4 + dy).min(ys - 1);
                let row = opsin.plane_row(1, y);
                let xrow = opsin.plane_row(0, y);
                let brow = opsin.plane_row(2, y);
                for dx in 0..4 {
                    let sx = (px0 + cx * 4 + dx).min(xs - 1);
                    let v = row[sx];
                    s += v;
                    s2 = fmla(v, v, s2);
                    chroma += xrow[sx].abs() + (brow[sx] - v).abs();
                }
            }
            let m = s * (1.0 / 16.0);
            means[cy * cw + cx] = m;
            within += (s2 * (1.0 / 16.0) - m * m).max(0.0);
        }
    }
    finish_gradient_region_stats(&means[..cw * ch], within, chroma, w * h, eps)
}

pub(crate) fn select_gradient_region_stats_fn() -> GradientRegionStatsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return |opsin, px, py, w, h, eps| unsafe {
            crate::neon::gradient_region_stats_neon(opsin, px, py, w, h, eps)
        };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |opsin, px, py, w, h, eps| unsafe {
            crate::avx::gradient_region_stats_avx2(opsin, px, py, w, h, eps)
        };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |opsin, px, py, w, h, eps| unsafe {
            crate::sse::gradient_region_stats_sse41(opsin, px, py, w, h, eps)
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::gradient_region_stats_wasm;
    #[allow(unreachable_code)]
    gradient_region_stats_scalar
}

pub(crate) fn select_gradient_region_stats_with_chroma_fn() -> GradientRegionStatsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return |opsin, px, py, w, h, eps| unsafe {
            crate::neon::gradient_region_stats_with_chroma_neon(opsin, px, py, w, h, eps)
        };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |opsin, px, py, w, h, eps| unsafe {
            crate::avx::gradient_region_stats_with_chroma_avx2(opsin, px, py, w, h, eps)
        };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |opsin, px, py, w, h, eps| unsafe {
            crate::sse::gradient_region_stats_with_chroma_sse41(opsin, px, py, w, h, eps)
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::gradient_region_stats_with_chroma_wasm;
    #[allow(unreachable_code)]
    gradient_region_stats_with_chroma_scalar
}

#[inline]
fn risk_gated(k: f32, accept: f32, q_min: f32, q_max: f32, area_scale: f32) -> f32 {
    if q_min <= 0.0 || k <= 0.0 {
        return accept;
    }
    let spread = q_max / q_min - 1.0;
    accept * fast_exp2(-k * area_scale * spread)
}
