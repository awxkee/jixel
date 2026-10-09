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
use crate::adaptive_quant::{dirty_log2f, dirty_log2p1f};
use crate::dc_group_data::DcGroupData;
use crate::dct::{DctInput, fmla};
use crate::encoding_context::EncodingContext;
use crate::entropy::f_log2;
use crate::image::{Image3F, ImageSB};
use crate::lossless::{
    GradPackInteriorFn, GradientScratch, pack_gradient_row, selected_grad_pack_interior_fn,
};
use crate::quant_weights::{DC_QUANT, INV_DC_QUANT};

const K_BLOCK_DIM: usize = 8;
const K_TILE_DIM_IN_BLOCKS: usize = 8;

/// libjxl default color factor: stored as 1/x in the encoder hot path.
const K_INV_COLOR_FACTOR: f32 = 1.0 / K_COLOR_FACTOR;
/// Regularisation toward base correlation. Matches libjxl-tiny.
const K_DISTANCE_MULTIPLIER_AC: f32 = 1e-3;

pub(crate) type CflRegressionFn =
    fn(&[f32; 64], &[f32; 64], &[f32; 64], &[f32; 64], &[f32; 64]) -> [f32; 4];

pub(crate) type CflRdoBlockFn = fn(
    &mut [f32; 63],
    &mut [f32; 63],
    &mut [f32; 63],
    &mut [f32; 63],
    &[f32; 64],
    &[f32; 64],
    &[f32; 64],
    &[f32; 64],
    &[f32; 64],
    f32,
);

pub(crate) type CflRdoStatsFn = fn(&[f32], &[f32]) -> [f32; 4];
pub(crate) type CflClosedLoopCostFn = fn(&[f32], &[f32], f32, &[f32; 63]) -> [f32; 2];

#[allow(dead_code)]
pub(crate) fn cfl_closed_loop_cost_scalar(
    m: &[f32],
    s: &[f32],
    factor: f32,
    thresholds: &[f32; 63],
) -> [f32; 2] {
    let mut distortion = 0.0f32;
    let mut coeff_bits = 0.0f32;
    let mut coeff = 1usize;
    for (&mv, &sv) in m.iter().zip(s) {
        let residual = sv - factor * mv;
        let level = if residual.abs() >= thresholds[coeff - 1] {
            residual.round()
        } else {
            0.0
        };
        let reconstructed = crate::group::dequantized_level_f32(level);
        let quant_error = residual - reconstructed;
        distortion += quant_error.abs() + 0.15 * residual.abs();
        if level != 0.0 {
            coeff_bits += 1.0 + dirty_log2p1f(level.abs());
        }
        coeff += 1;
        if coeff == 64 {
            coeff = 1;
        }
    }
    [distortion, coeff_bits]
}

pub(crate) fn selected_cfl_closed_loop_cost_fn() -> CflClosedLoopCostFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |m, s, factor, thresholds| unsafe {
            crate::neon::cfl_closed_loop_cost_neon(m, s, factor, thresholds)
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return |m, s, factor, thresholds| unsafe {
            crate::avx::cfl_closed_loop_cost_avx2(m, s, factor, thresholds)
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::cfl_closed_loop_cost_wasm;
    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
        all(target_arch = "aarch64", feature = "neon")
    )))]
    cfl_closed_loop_cost_scalar
}

#[allow(dead_code)]
pub(crate) fn cfl_rdo_stats_scalar(m: &[f32], s: &[f32]) -> [f32; 4] {
    let mut stats = [0.0f32; 4];
    for (&mv, &sv) in m.iter().zip(s) {
        stats[0] = fmla(mv, sv, stats[0]);
        stats[1] = fmla(mv, mv, stats[1]);
        stats[2] = fmla(sv, sv, stats[2]);
        stats[3] += sv.abs();
    }
    stats
}

pub(crate) fn selected_cfl_rdo_stats_fn() -> CflRdoStatsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |m, s| unsafe { crate::neon::cfl_rdo_stats_neon(m, s) }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return |m, s| unsafe { crate::avx::cfl_rdo_stats_avx2(m, s) };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::cfl_rdo_stats_wasm;
    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
        all(target_arch = "aarch64", feature = "neon")
    )))]
    cfl_rdo_stats_scalar
}

#[allow(dead_code)]
pub(crate) fn cfl_rdo_block_scalar(
    m_x: &mut [f32; 63],
    s_x: &mut [f32; 63],
    m_b: &mut [f32; 63],
    s_b: &mut [f32; 63],
    block_y: &[f32; 64],
    block_x: &[f32; 64],
    block_b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
    q_block: f32,
) {
    for i in 0..63 {
        let coeff = i + 1;
        let qx = qm_x[coeff] * q_block;
        let qb = qm_b[coeff] * q_block;
        m_x[i] = block_y[coeff] * qx;
        s_x[i] = block_x[coeff] * qx;
        m_b[i] = block_y[coeff] * qb;
        s_b[i] = block_b[coeff] * qb;
    }
}

pub(crate) fn selected_cfl_rdo_block_fn() -> CflRdoBlockFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |m_x, s_x, m_b, s_b, y, x, b, qm_x, qm_b, q_block| unsafe {
            crate::neon::cfl_rdo_block_neon(m_x, s_x, m_b, s_b, y, x, b, qm_x, qm_b, q_block);
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |m_x, s_x, m_b, s_b, y, x, b, qm_x, qm_b, q_block| unsafe {
            crate::avx::cfl_rdo_block_avx2(m_x, s_x, m_b, s_b, y, x, b, qm_x, qm_b, q_block);
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::cfl_rdo_block_wasm;
    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
        all(target_arch = "aarch64", feature = "neon")
    )))]
    cfl_rdo_block_scalar
}

#[allow(dead_code)]
pub(crate) fn cfl_regression_scalar(
    block_y: &[f32; 64],
    block_x: &[f32; 64],
    block_b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
) -> [f32; 4] {
    let mut sums = [0.0f32; 4];
    for i in 0..64 {
        let m_x = block_y[i] * qm_x[i];
        let s_x = block_x[i] * qm_x[i];
        let a_x = K_INV_COLOR_FACTOR * m_x;
        let b_x = -s_x;
        sums[0] = fmla(a_x, a_x, sums[0]);
        sums[1] = fmla(a_x, b_x, sums[1]);

        let m_b = block_y[i] * qm_b[i];
        let s_b = block_b[i] * qm_b[i];
        let a_b = K_INV_COLOR_FACTOR * m_b;
        let b_b = fmla(1.0, m_b, -s_b);
        sums[2] = fmla(a_b, a_b, sums[2]);
        sums[3] = fmla(a_b, b_b, sums[3]);
    }
    sums
}

pub(crate) fn selected_cfl_regression_fn() -> CflRegressionFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |y, x, b, qm_x, qm_b| unsafe { crate::neon::cfl_regression_neon(y, x, b, qm_x, qm_b) }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return |y, x, b, qm_x, qm_b| unsafe {
            crate::avx::cfl_regression_avx2(y, x, b, qm_x, qm_b)
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::cfl_regression_wasm;
    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
        all(target_arch = "aarch64", feature = "neon")
    )))]
    cfl_regression_scalar
}

const CFL_DEADZONE_AMOUNT: f32 = 1.5;
const CFL_DEADZONE_LO: f32 = 1.0;
const CFL_DEADZONE_HI: f32 = 2.0;

struct CflRdoFit {
    /// Rate weight for the (cand - pred) token estimate, scaled by tile_q.
    lambda_bits: f32,
    /// Per-unit multiplier magnitude cost, scaled by tile_q.
    mag_cost: f32,
    /// Magnitude-cost amplification on strongly/mildly neutral tiles.
    dz_strong: f32,
    dz_mild: f32,
    corr_strong: f32,
    corr_mild: f32,
    energy_strong: f32,
    energy_mild: f32,
    /// Mean |chroma DC| gates, raw opsin scale (X is tiny; B tracks luma).
    dc_thr_x: f32,
    dc_thr_b: f32,
    /// Fall back to the L2 path at and above this distance: the L1
    /// objective drifts off-metric at high d (Kodak: win < 3, loss at 3 —
    /// the stress-corpus fit wanted 3.5, the holdout overrules).
    max_d: f32,
}

static CFL_RDO: CflRdoFit = CflRdoFit {
    lambda_bits: 0.045,
    mag_cost: 0.09,
    dz_strong: 3.4,
    dz_mild: 2.45,
    corr_strong: 0.20,
    corr_mild: 0.35,
    energy_strong: 0.09,
    energy_mild: 0.073,
    dc_thr_x: 0.10,
    dc_thr_b: 0.55,
    max_d: 3.0,
};

/// From this luma correlation up, an X tile is one hue modulated by
/// luminance and takes the least-squares multiplier.
const CFL_X_LS_MIN_CORRELATION: f32 = 0.98;

#[inline]
fn cfl_deadzone(distance: f32) -> f32 {
    if distance <= CFL_DEADZONE_LO {
        return CFL_DEADZONE_AMOUNT;
    }
    if distance >= CFL_DEADZONE_HI {
        return 0.0;
    }
    const CFL_DEADZONE_SCALE: f32 = CFL_DEADZONE_AMOUNT / (CFL_DEADZONE_HI - CFL_DEADZONE_LO);
    CFL_DEADZONE_SCALE * (CFL_DEADZONE_HI - distance)
}

/// `units` converts the kernels' 1/84 slope steps to the frame's.
fn solve_multiplier(ca: f32, cb: f32, num: usize, distance_mul: f32, dz: f32, units: f32) -> i32 {
    if num == 0 {
        return 0;
    }
    let mut x = -cb / fmla(num as f32 * distance_mul, 0.5, ca);
    // Slopes inside the deadzone are noise fits and snap to the base
    // correlation. A slope outside it is signal and is kept whole: shrinking
    // it leaves a residual that is a scaled copy of luma, which the chroma
    // quantizer drops, desaturating the tile. No-op when `dz` is 0.
    if x.abs() < dz {
        x = 0.0;
    }
    (x * units).round().clamp(-128.0, 127.0) as i32
}

/// The regressions measure X slopes around a base correlation of 0 and B
/// slopes around 1. Under other bases, shift the planes by the difference so
/// the same slope lands on the same signaled multiplier.
#[inline]
fn shift_to_xyb_bases(
    ctx: &EncodingContext,
    block_y: &[f32; 64],
    block_x: &mut [f32; 64],
    block_b: &mut [f32; 64],
) {
    for (block, shift) in [
        (block_x, -ctx.cfl_base_x()),
        (block_b, 1.0 - ctx.cfl_base_b()),
    ] {
        if shift != 0.0 {
            for (v, &y) in block.iter_mut().zip(block_y) {
                *v = fmla(shift, y, *v);
            }
        }
    }
}

struct CflScratch {
    block_y: [f32; 64],
    block_x: [f32; 64],
    block_b: [f32; 64],
}

/// Compute (ytox, ytob) for one tile. `tile_brect_*` are block coordinates
/// (top-left inclusive) into `opsin`, sizes capped at K_TILE_DIM_IN_BLOCKS.
fn compute_cmap_tile(
    ctx: &EncodingContext,
    opsin: &Image3F,
    bx0: usize,
    by0: usize,
    bx_count: usize,
    by_count: usize,
    distance: f32,
    scratch: &mut CflScratch,
) -> (i32, i32) {
    let matrices = ctx.matrices();
    let qm_x: &[f32; 64] = matrices.inv_matrix(0).first_chunk::<64>().unwrap();
    let qm_b: &[f32; 64] = matrices.inv_matrix(2).first_chunk::<64>().unwrap();

    let mut block_y = scratch.block_y;
    let mut block_x = scratch.block_x;
    let mut block_b = scratch.block_b;

    let mut ca_x = 0.0f32;
    let mut cb_x = 0.0f32;
    let mut ca_b = 0.0f32;
    let mut cb_b = 0.0f32;
    let mut num = 0usize;
    for by in 0..by_count {
        for bx in 0..bx_count {
            let px = (bx0 + bx) * K_BLOCK_DIM;
            let py = (by0 + by) * K_BLOCK_DIM;
            // Bounds check: skip if outside opsin (last edge tile may be small).
            if px + K_BLOCK_DIM > opsin.xsize() || py + K_BLOCK_DIM > opsin.ysize() {
                continue;
            }
            let stride = opsin.xsize();
            let offset = py * stride + px;
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(1)[offset..], stride),
                &mut block_y,
            );
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(0)[offset..], stride),
                &mut block_x,
            );
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(2)[offset..], stride),
                &mut block_b,
            );

            // Zero DC (LF position) — libjxl-tiny zeros it so it doesn't affect
            // the regression; the per-tile AC factor controls AC only.
            block_y[0] = 0.0;
            block_x[0] = 0.0;
            block_b[0] = 0.0;
            shift_to_xyb_bases(ctx, &block_y, &mut block_x, &mut block_b);

            let sums = (ctx.cfl_regression)(&block_y, &block_x, &block_b, qm_x, qm_b);
            ca_x += sums[0];
            cb_x += sums[1];
            ca_b += sums[2];
            cb_b += sums[3];
            num += 64;
        }
    }

    let dz = cfl_deadzone(distance);
    let units = ctx.cfl_frame().color_factor / K_COLOR_FACTOR;
    let ytox = solve_multiplier(ca_x, cb_x, num, K_DISTANCE_MULTIPLIER_AC, dz, units);
    let ytob = solve_multiplier(ca_b, cb_b, num, K_DISTANCE_MULTIPLIER_AC, dz, units);
    (ytox, ytob)
}

/// Worst-case per-tile coefficient count: a full 8×8-block tile keeps 63 AC
/// coefficients per block. The L1 candidate cost needs every coefficient (it
/// has no closed-form sufficient statistics, unlike the default L2 path), so
/// the tile's worth of them is staged here.
const CFL_RDO_TILE_AC: usize = K_TILE_DIM_IN_BLOCKS * K_TILE_DIM_IN_BLOCKS * 63;

pub(crate) struct CflRdoScratch {
    m_x: Box<[f32; CFL_RDO_TILE_AC]>,
    s_x: Box<[f32; CFL_RDO_TILE_AC]>,
    m_b: Box<[f32; CFL_RDO_TILE_AC]>,
    s_b: Box<[f32; CFL_RDO_TILE_AC]>,
    len: usize,
}

impl CflRdoScratch {
    fn new() -> Self {
        let alloc = || {
            vec![0.0f32; CFL_RDO_TILE_AC]
                .into_boxed_slice()
                .try_into()
                .unwrap()
        };
        CflRdoScratch {
            m_x: alloc(),
            s_x: alloc(),
            m_b: alloc(),
            s_b: alloc(),
            len: 0,
        }
    }
}

impl Default for CflRdoScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Clamped-gradient prediction of a tile's multiplier from its causal
/// neighbors — the same predictor `tokenize_dc_global` charges the map with,
/// so rate-to-pred here is rate-to-pred in the bitstream. Safe because
/// `fill_cmap` walks tiles serially in raster order (the upstream fork races
/// here; jixel does not).
fn cmap_tile_pred(map: &ImageSB, tx: usize, ty: usize) -> i32 {
    if tx == 0 && ty == 0 {
        return 0;
    }
    if tx == 0 {
        return map.row(ty - 1)[tx] as i32;
    }
    if ty == 0 {
        return map.row(ty)[tx - 1] as i32;
    }
    grad_predict(
        map.row(ty - 1)[tx] as i32,
        map.row(ty)[tx - 1] as i32,
        map.row(ty - 1)[tx - 1] as i32,
    )
}

/// Approximate hybrid-uint token cost in bits for a map residual.
fn cmap_residual_bits(res: i32) -> f32 {
    let res = res.unsigned_abs();
    match res {
        0 => 0.8,
        1 => 2.0,
        _ => 2.5 + dirty_log2f(res as f32),
    }
}

/// Pick one channel's multiplier by candidate search: L1 distortion + rate
/// to the spatial predictor + a neutral-gated magnitude cost. `m`/`s` are
/// luma/chroma coefficients in quant-step units; `base` is the base
/// correlation the signaled value offsets.
#[allow(clippy::too_many_arguments)]
fn optimize_channel_rdo(
    stats_fn: CflRdoStatsFn,
    closed_loop_cost_fn: CflClosedLoopCostFn,
    m: &[f32],
    s: &[f32],
    channel: usize,
    distance: f32,
    closed_loop: bool,
    qm_multiplier: f32,
    pred: i32,
    base: f32,
    dc_avg: f32,
    dc_thr: f32,
    tile_q: f32,
    color_factor: f32,
) -> i32 {
    if m.is_empty() {
        return if pred.abs() <= 4 { pred } else { 0 };
    }

    let [dot_ms, dot_mm, dot_ss, sum_abs_s] = stats_fn(m, s);
    let energy = sum_abs_s / m.len() as f32;
    let corr = if dot_mm > 1e-6 && dot_ss > 1e-6 {
        dot_ms.abs() / (dot_mm * dot_ss).sqrt()
    } else {
        0.0
    };

    // Analytical least-squares seed: argmin_f sum (s - f*m)^2.
    let target_factor = if dot_mm > 1e-6 { dot_ms / dot_mm } else { base };
    let target_cand = ((target_factor - base) * color_factor)
        .round()
        .clamp(-128.0, 127.0) as i32;

    // Neutral-tile gate: only amplify the magnitude cost where chroma is
    // uncorrelated, low-energy, and near-neutral, so the deadzone suppresses
    // color noise without desaturating regions with a real slope.
    let dz_penalty =
        if corr < CFL_RDO.corr_strong && energy < CFL_RDO.energy_strong && dc_avg < dc_thr {
            CFL_RDO.dz_strong
        } else if corr < CFL_RDO.corr_mild && energy < CFL_RDO.energy_mild && dc_avg < dc_thr * 1.5
        {
            CFL_RDO.dz_mild
        } else {
            1.0
        };

    // The candidate costs count error per coefficient. A multiplier off the
    // least-squares one leaves a residual that copies luma, and what the
    // quantizer drops of it shifts the whole tile's saturation one way.
    if channel == 0 && corr >= CFL_X_LS_MIN_CORRELATION {
        return target_cand;
    }

    let mut cands = [0i32; 19];
    let mut num_cands = 0usize;
    let mut add_cand = |cand: i32| {
        let cand = cand.clamp(-128, 127);
        if !cands[..num_cands].contains(&cand) {
            cands[num_cands] = cand;
            num_cands += 1;
        }
    };
    add_cand(0);
    add_cand(pred);
    add_cand(target_cand);
    for d in 1..=4 {
        add_cand(target_cand - d);
        add_cand(target_cand + d);
        add_cand(pred - d);
        add_cand(pred + d);
    }

    let lambda = CFL_RDO.lambda_bits * tile_q;
    let mag_cost = CFL_RDO.mag_cost * tile_q * dz_penalty;
    let quadrant_thresholds =
        crate::group::quantize_ac_thresholds_scaled(channel, 1, 1, distance, qm_multiplier);
    let thresholds: [f32; 63] = std::array::from_fn(|i| {
        let coeff = i + 1;
        quadrant_thresholds[usize::from(coeff >= 32) * 2 + usize::from(coeff & 7 >= 4)]
    });
    let mut best = (f32::MAX, 0i32);
    for &cand in &cands[..num_cands] {
        let factor = fmla(cand as f32, 1.0 / color_factor, base);
        let (distortion, coeff_bits) = if closed_loop {
            // Closed-loop error is the primary term. A small residual-energy
            // term and coefficient-rate proxy retain CfL's compression goal
            // instead of choosing an expensive phase-aligned residual.
            let [distortion, coeff_bits] = closed_loop_cost_fn(m, s, factor, &thresholds);
            (distortion, coeff_bits)
        } else {
            let mut distortion = 0.0f32;
            for (&mv, &sv) in m.iter().zip(s) {
                let residual = sv - factor * mv;
                distortion += residual.abs();
            }
            (distortion, 0.0)
        };
        let cost = distortion
            + 0.15 * lambda * coeff_bits
            + lambda * cmap_residual_bits(cand - pred)
            + cand.abs() as f32 * mag_cost;
        if cost < best.0 {
            best = (cost, cand);
        }
    }
    best.1
}

struct CflBlockRegion {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

struct CflRdoQuantization<'a> {
    field: &'a crate::image::ImageB,
    scale: f32,
    distance: f32,
    closed_loop: bool,
    block_x: usize,
    block_y: usize,
}

struct CflTilePrediction {
    ytox: i32,
    ytob: i32,
}

struct CflRdoTile<'a> {
    opsin: &'a Image3F,
    blocks: CflBlockRegion,
    quant: CflRdoQuantization<'a>,
    prediction: CflTilePrediction,
}

struct CflRdoTileScratch<'a> {
    dct: &'a mut CflScratch,
    rdo: &'a mut CflRdoScratch,
}

/// RDO variant of `compute_cmap_tile`. `tile.quant.block_x/block_y` index the
/// tile into the DC-group-local quant field; its scale converts entries to
/// quantizer step scale (`quantize_ac_q_scaled` without the qm multiplier).
fn compute_cmap_tile_rdo(
    ctx: &EncodingContext,
    tile: CflRdoTile<'_>,
    scratch: CflRdoTileScratch<'_>,
) -> (i32, i32) {
    let CflRdoTile {
        opsin,
        blocks,
        quant,
        prediction,
    } = tile;
    let CflRdoTileScratch { dct: scratch, rdo } = scratch;
    let matrices = ctx.matrices();
    let qm_x: &[f32; 64] = matrices.inv_matrix(0).first_chunk::<64>().unwrap();
    let qm_b: &[f32; 64] = matrices.inv_matrix(2).first_chunk::<64>().unwrap();

    let mut block_y = scratch.block_y;
    let mut block_x = scratch.block_x;
    let mut block_b = scratch.block_b;

    rdo.len = 0;

    let mut dc_abs_x = 0.0f32;
    let mut dc_abs_b = 0.0f32;
    let mut tile_q_sum = 0.0f32;
    let mut num_blocks = 0usize;
    let x_mul = if quant.closed_loop && (ctx.x_heavy() || quant.distance >= 1.25) {
        1.25
    } else {
        1.0
    };
    let b_mul = if quant.closed_loop {
        ctx.b_qm_mul()
    } else {
        1.0
    };
    for by in 0..blocks.height {
        for bx in 0..blocks.width {
            let px = (blocks.x + bx) * K_BLOCK_DIM;
            let py = (blocks.y + by) * K_BLOCK_DIM;
            if px + K_BLOCK_DIM > opsin.xsize() || py + K_BLOCK_DIM > opsin.ysize() {
                continue;
            }
            let stride = opsin.xsize();
            let offset = py * stride + px;
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(1)[offset..], stride),
                &mut block_y,
            );
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(0)[offset..], stride),
                &mut block_x,
            );
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(2)[offset..], stride),
                &mut block_b,
            );

            shift_to_xyb_bases(ctx, &block_y, &mut block_x, &mut block_b);
            dc_abs_x += block_x[0].abs();
            dc_abs_b += block_b[0].abs();

            let q_block =
                quant.scale * quant.field.row(quant.block_y + by)[quant.block_x + bx] as f32;
            tile_q_sum += q_block;
            num_blocks += 1;

            // Skip the DC (index 0): the per-tile factor controls AC only.
            let n = rdo.len;

            let mx = rdo.m_x[n..].first_chunk_mut::<63>().unwrap();
            let sx = rdo.s_x[n..].first_chunk_mut::<63>().unwrap();
            let mb = rdo.m_b[n..].first_chunk_mut::<63>().unwrap();
            let sb = rdo.s_b[n..].first_chunk_mut::<63>().unwrap();
            (ctx.cfl_rdo_block)(
                mx, sx, mb, sb, &block_y, &block_x, &block_b, qm_x, qm_b, q_block,
            );
            // The SIMD staging kernel produces base quant-step units. CfL is
            // selected before coefficient coding but must see the same frame
            // X/B precision multipliers that coding will use.
            if quant.closed_loop {
                for value in mx.iter_mut().chain(sx.iter_mut()) {
                    *value *= x_mul;
                }
                for value in mb.iter_mut().chain(sb.iter_mut()) {
                    *value *= b_mul;
                }
            }
            rdo.len = n + 63;
        }
    }

    let (dc_avg_x, dc_avg_b, tile_q) = if num_blocks > 0 {
        let inv = 1.0 / num_blocks as f32;
        (dc_abs_x * inv, dc_abs_b * inv, tile_q_sum * inv)
    } else {
        (0.0, 0.0, 1.0)
    };

    let ytox = optimize_channel_rdo(
        ctx.cfl_rdo_stats,
        ctx.cfl_closed_loop_cost,
        &rdo.m_x[..rdo.len],
        &rdo.s_x[..rdo.len],
        0,
        quant.distance,
        quant.closed_loop,
        x_mul,
        prediction.ytox,
        0.0,
        dc_avg_x,
        CFL_RDO.dc_thr_x,
        tile_q,
        ctx.cfl_frame().color_factor,
    );
    let ytob = optimize_channel_rdo(
        ctx.cfl_rdo_stats,
        ctx.cfl_closed_loop_cost,
        &rdo.m_b[..rdo.len],
        &rdo.s_b[..rdo.len],
        2,
        quant.distance,
        quant.closed_loop,
        b_mul,
        prediction.ytob,
        1.0,
        dc_avg_b,
        CFL_RDO.dc_thr_b,
        tile_q,
        ctx.cfl_frame().color_factor,
    );
    (ytox, ytob)
}

/// Fill `ytox_map` / `ytob_map` (sized `(xtiles, ytiles)`) by running the
/// per-tile regression on `opsin`. `(dc_group_x0_blocks, dc_group_y0_blocks)`
/// is the block offset of this DC group's (0, 0) tile into `opsin`.
pub(crate) fn fill_cmap(
    ctx: &EncodingContext,
    opsin: &Image3F,
    dc_group_x0_blocks: usize,
    dc_group_y0_blocks: usize,
    dc_group_xsize_blocks: usize,
    dc_group_ysize_blocks: usize,
    raw_quant_field: &crate::image::ImageB,
    scale: f32,
    rdo_scratch: &mut crate::coder_scratch::LazyScratch<CflRdoScratch>,
    ytox_map: &mut ImageSB,
    ytob_map: &mut ImageSB,
    distance: f32,
) {
    let xtiles = ytox_map.xsize();
    let ytiles = ytox_map.ysize();

    let mut scratch = CflScratch {
        block_b: [0.; 64],
        block_x: [0.; 64],
        block_y: [0.; 64],
    };

    // Coarse chroma structure still benefits from searching the
    // quantized residual cost. Dropping to regression at d=3 erases yellow detail
    // on the very frames for which we retain extra X/B precision.
    // YCbCr chroma follows luma with large, content-dependent slopes, which
    // the regression's deadzone misses; it searches them at Fast too.
    let ycbcr =
        ctx.coding == crate::coding::CodingTransform::YCbCr && ctx.speed.effort().cfl_ycbcr_rdo;
    let use_rdo =
        (distance < CFL_RDO.max_d || ctx.x_heavy()) && (ctx.speed.effort().cfl_rdo || ycbcr);

    for ty in 0..ytiles {
        for tx in 0..xtiles {
            let bx0 = dc_group_x0_blocks + tx * K_TILE_DIM_IN_BLOCKS;
            let by0 = dc_group_y0_blocks + ty * K_TILE_DIM_IN_BLOCKS;
            let bx_count = K_TILE_DIM_IN_BLOCKS
                .min(dc_group_xsize_blocks.saturating_sub(tx * K_TILE_DIM_IN_BLOCKS));
            let by_count = K_TILE_DIM_IN_BLOCKS
                .min(dc_group_ysize_blocks.saturating_sub(ty * K_TILE_DIM_IN_BLOCKS));
            if bx_count == 0 || by_count == 0 {
                continue;
            }
            let (ytox, ytob) = if use_rdo {
                // The predictor reads only already-written tiles: this loop
                // is serial raster order, matching the token predictor.
                let pred_x = cmap_tile_pred(ytox_map, tx, ty);
                let pred_b = cmap_tile_pred(ytob_map, tx, ty);
                compute_cmap_tile_rdo(
                    ctx,
                    CflRdoTile {
                        opsin,
                        blocks: CflBlockRegion {
                            x: bx0,
                            y: by0,
                            width: bx_count,
                            height: by_count,
                        },
                        quant: CflRdoQuantization {
                            field: raw_quant_field,
                            scale,
                            distance,
                            closed_loop: ctx.x_heavy(),
                            block_x: tx * K_TILE_DIM_IN_BLOCKS,
                            block_y: ty * K_TILE_DIM_IN_BLOCKS,
                        },
                        prediction: CflTilePrediction {
                            ytox: pred_x,
                            ytob: pred_b,
                        },
                    },
                    CflRdoTileScratch {
                        dct: &mut scratch,
                        rdo: rdo_scratch,
                    },
                )
            } else {
                compute_cmap_tile(
                    ctx,
                    opsin,
                    bx0,
                    by0,
                    bx_count,
                    by_count,
                    distance,
                    &mut scratch,
                )
            };
            ytox_map.row_mut(ty)[tx] = ytox as i8;
            ytob_map.row_mut(ty)[tx] = ytob as i8;
        }
    }
}

/// Returns the per-tile factor as a slope (= base_correlation + cmap/84).
#[inline]
pub(crate) fn y_to_x_ratio(cfl: CflFrame, cmap_x: i8) -> f32 {
    fmla(cmap_x as f32, cfl.inv_factor(), cfl.base_x)
}

#[inline]
pub(crate) fn y_to_b_ratio(cfl: CflFrame, cmap_b: i8) -> f32 {
    fmla(cmap_b as f32, cfl.inv_factor(), cfl.base_b)
}

pub(crate) const K_COLOR_FACTOR: f32 = 84.0;

/// A frame's chroma-from-luma parameters: the base correlations every tile
/// and the DC offset, and the `color_factor` whose reciprocal is one signaled
/// step of the per-tile maps and `ytob_dc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CflFrame {
    pub(crate) base_x: f32,
    pub(crate) base_b: f32,
    pub(crate) color_factor: f32,
}

impl CflFrame {
    /// The XYB defaults, which the all-default header bundle implies.
    pub(crate) const XYB: Self = Self {
        base_x: 0.0,
        base_b: 1.0,
        color_factor: K_COLOR_FACTOR,
    };

    #[inline]
    pub(crate) fn inv_factor(self) -> f32 {
        1.0 / self.color_factor
    }
}
const YTOB_DC_LIMIT: i32 = 127;
const COLOR_CORRELATION_HEADER_BITS: f64 = 2.0 + 16.0 + 16.0 + 8.0 + 8.0;

#[inline]
fn grad_predict(n: i32, w: i32, nw: i32) -> i32 {
    (n + w - nw).clamp(n.min(w), n.max(w))
}

struct DcGradientPrice {
    hist: [u64; 64],
    extra_bits: u64,
    total: u64,
    scratch: GradientScratch,
    grad_pack: GradPackInteriorFn,
}

impl Default for DcGradientPrice {
    fn default() -> Self {
        Self {
            hist: [0; 64],
            extra_bits: 0,
            total: 0,
            scratch: GradientScratch::default(),
            grad_pack: selected_grad_pack_interior_fn(),
        }
    }
}

impl DcGradientPrice {
    fn clear(&mut self) {
        self.hist.fill(0);
        self.extra_bits = 0;
        self.total = 0;
    }

    /// Reset prediction on the first row of each DC group while pooling its
    /// symbols with the other groups. Keep all row buffers across candidates.
    fn add_row<T: Copy>(&mut self, row: &[T], first_row: bool)
    where
        i32: From<T>,
    {
        let scratch = &mut self.scratch;
        scratch.cur.resize(row.len(), 0);
        scratch.prev.resize(row.len(), 0);
        scratch.buf.resize(row.len(), 0);
        for (dest, &value) in scratch.cur.iter_mut().zip(row) {
            *dest = i32::from(value);
        }
        pack_gradient_row(
            &scratch.cur,
            (!first_row).then_some(scratch.prev.as_slice()),
            &mut scratch.buf,
            self.grad_pack,
        );
        for &value in &scratch.buf {
            let (tok, nbits, _) = crate::entropy::uint_encode(value);
            self.hist[(tok as usize).min(self.hist.len() - 1)] += 1;
            self.extra_bits += u64::from(nbits);
        }
        self.total += row.len() as u64;
        std::mem::swap(&mut scratch.cur, &mut scratch.prev);
    }

    fn bits(&self) -> f64 {
        dc_histogram_bits(&self.hist, self.extra_bits, self.total)
    }
}

/// Order-0 entropy of the DC symbols plus their raw extra bits.
fn dc_histogram_bits(hist: &[u64], extra_bits: u64, total: u64) -> f64 {
    let inv = 1.0 / total.max(1) as f64;
    hist.iter()
        .filter(|&&n| n != 0)
        .fold(extra_bits as f64, |sum, &n| {
            sum - n as f64 * f_log2(n as f64 * inv)
        })
}

pub(crate) type FillYtobRowFn = fn(&mut [i32], &[i16], &[i16], f32);

#[inline(always)]
#[allow(dead_code)]
pub(crate) fn fill_ytob_row_scalar(dst: &mut [i32], b: &[i16], y: &[i16], slope: f32) {
    for ((dst, &b), &y) in dst.iter_mut().zip(b).zip(y) {
        let adj = (slope * y as f32).round_ties_even() as i32;
        *dst = b as i32 - adj;
    }
}

pub(crate) fn selected_fill_ytob_row_fn() -> FillYtobRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |dst, b, y, slope| unsafe {
            crate::neon::fill_ytob_row_neon(dst, b, y, slope);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |dst, b, y, slope| unsafe {
            crate::avx::fill_ytob_row_avx2(dst, b, y, slope);
        };
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::fill_ytob_row_wasm;

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    fill_ytob_row_scalar
}

fn ytob_dc_cost(
    dc_datas: &[DcGroupData],
    k: i32,
    step: f32,
    fill_ytob_row: FillYtobRowFn,
    cur: &mut Vec<i32>,
    price: &mut DcGradientPrice,
) -> f64 {
    price.clear();
    let slope = k as f32 * step;

    for dc in dc_datas {
        let b = dc.quant_dc.plane(2);
        let y = dc.quant_dc.plane(1);
        let (xsize, ysize) = (b.xsize(), b.ysize());
        debug_assert_eq!((xsize, ysize), (y.xsize(), y.ysize()));
        if xsize == 0 || ysize == 0 {
            continue;
        }

        if cur.len() != xsize {
            cur.resize(xsize, 0i32);
        }

        let rows = b
            .as_slice()
            .chunks_exact(xsize)
            .zip(y.as_slice().chunks_exact(xsize));

        for (row, (b_row, y_row)) in rows.enumerate() {
            fill_ytob_row(cur, b_row, y_row, slope);
            price.add_row(cur, row == 0);
        }
    }
    price.bits()
}

#[inline]
fn add_ytob_weight(rb: i32, ry: i32, step: f32, weights: &mut [u64]) {
    if ry == 0 {
        return;
    }

    let k = (rb as f32 / (ry as f32 * step)).round_ties_even() as i32;
    let idx = k.clamp(-YTOB_DC_LIMIT, YTOB_DC_LIMIT) + YTOB_DC_LIMIT;
    weights[idx as usize] += ry.unsigned_abs() as u64;
}

pub(crate) type AccumulateYtobWeightsFn = fn(&[i32], &[i32], f32, &mut [u64]);

#[allow(dead_code)]
pub(crate) fn accumulate_ytob_weights_scalar(
    rb: &[i32],
    ry: &[i32],
    step: f32,
    weights: &mut [u64],
) {
    for (&rb, &ry) in rb.iter().zip(ry) {
        add_ytob_weight(rb, ry, step, weights);
    }
}

pub(crate) fn selected_accumulate_ytob_weights_fn() -> AccumulateYtobWeightsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |rb, ry, step, weights| unsafe {
            crate::neon::accumulate_ytob_weights_neon(rb, ry, step, weights);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |rb, ry, step, weights| unsafe {
            crate::avx::accumulate_ytob_weights_avx2(rb, ry, step, weights);
        };
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    accumulate_ytob_weights_scalar
}

pub(crate) type FillYtobResidualsFn = fn(&mut [i32], &mut [i32], &[i16], &[i16], &[i16], &[i16]);

fn fill_ytob_residuals_plane_scalar(dst: &mut [i32], row: &[i16], up: &[i16]) {
    let Some((dst_first, dst_rest)) = dst.split_first_mut() else {
        return;
    };
    let (&row_first, _) = row.split_first().unwrap();
    let (&up_first, _) = up.split_first().unwrap();
    *dst_first = row_first as i32 - up_first as i32;

    for ((dst, row), above) in dst_rest
        .iter_mut()
        .zip(row.array_windows::<2>())
        .zip(up.array_windows::<2>())
    {
        *dst = row[1] as i32 - grad_predict(above[1] as i32, row[0] as i32, above[0] as i32);
    }
}

#[allow(dead_code)]
pub(crate) fn fill_ytob_residuals_scalar(
    rb: &mut [i32],
    ry: &mut [i32],
    b_row: &[i16],
    y_row: &[i16],
    b_up: &[i16],
    y_up: &[i16],
) {
    let len = rb
        .len()
        .min(ry.len())
        .min(b_row.len())
        .min(y_row.len())
        .min(b_up.len())
        .min(y_up.len());
    if len == 0 {
        return;
    }

    fill_ytob_residuals_plane_scalar(&mut rb[..len], &b_row[..len], &b_up[..len]);
    fill_ytob_residuals_plane_scalar(&mut ry[..len], &y_row[..len], &y_up[..len]);
}

pub(crate) fn selected_fill_ytob_residuals_fn() -> FillYtobResidualsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |rb, ry, b_row, y_row, b_up, y_up| unsafe {
            crate::neon::fill_ytob_residuals_neon(rb, ry, b_row, y_row, b_up, y_up);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |rb, ry, b_row, y_row, b_up, y_up| unsafe {
            crate::avx::fill_ytob_residuals_avx2(rb, ry, b_row, y_row, b_up, y_up);
        };
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::fill_ytob_residuals_wasm;

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    fill_ytob_residuals_scalar
}

fn ytob_dc_seed(
    dc_datas: &[DcGroupData],
    accumulate_weights: AccumulateYtobWeightsFn,
    fill_residuals: FillYtobResidualsFn,
    rb_scratch: &mut Vec<i32>,
    ry_scratch: &mut Vec<i32>,
    step: f32,
) -> i32 {
    // L1 fit: weighted median of the per-sample ratio, bucketed straight into
    // signaled-slope units so no sort is needed.
    let mut weights = [0u64; (2 * YTOB_DC_LIMIT + 1) as usize];
    for dc in dc_datas {
        let (b, y) = (dc.quant_dc.plane(2), dc.quant_dc.plane(1));
        let (xsize, ysize) = (b.xsize(), b.ysize());
        debug_assert_eq!((xsize, ysize), (y.xsize(), y.ysize()));
        if xsize == 0 || ysize == 0 {
            continue;
        }
        if rb_scratch.len() != xsize {
            rb_scratch.resize(xsize, 0);
        }
        if ry_scratch.len() != xsize {
            ry_scratch.resize(xsize, 0);
        }
        let rb = &mut rb_scratch[..xsize];
        let ry = &mut ry_scratch[..xsize];

        let mut rows = b
            .as_slice()
            .chunks_exact(xsize)
            .zip(y.as_slice().chunks_exact(xsize));
        let (mut b_up, mut y_up) = rows.next().unwrap();

        // On the top row the clamped-gradient predictor is simply the value
        // to the left. Track it directly instead of branching on every x.
        rb[0] = b_up[0] as i32;
        ry[0] = y_up[0] as i32;
        for (((rb, ry), b), y) in rb[1..]
            .iter_mut()
            .zip(&mut ry[1..])
            .zip(b_up.array_windows::<2>())
            .zip(y_up.array_windows::<2>())
        {
            *rb = b[1] as i32 - b[0] as i32;
            *ry = y[1] as i32 - y[0] as i32;
        }
        accumulate_weights(rb, ry, step, &mut weights);

        for (b_row, y_row) in rows {
            fill_residuals(rb, ry, b_row, y_row, b_up, y_up);
            accumulate_weights(rb, ry, step, &mut weights);

            (b_up, y_up) = (b_row, y_row);
        }
    }

    let half = weights.iter().sum::<u64>() / 2;
    let mut acc = 0u64;
    weights
        .iter()
        .position(|&weight| {
            acc += weight;
            acc > half
        })
        .map_or(0, |idx| idx as i32 - YTOB_DC_LIMIT)
}

pub(crate) fn choose_ytob_dc(
    dc_datas: &[DcGroupData],
    fill_ytob_row: FillYtobRowFn,
    accumulate_weights: AccumulateYtobWeightsFn,
    fill_residuals: FillYtobResidualsFn,
    rb_scratch: &mut Vec<i32>,
    ry_scratch: &mut Vec<i32>,
    dc_step: [f32; 3],
    color_factor: f32,
) -> i32 {
    // One signaled step moves the stored B DC by this much per stored Y unit:
    // the slope is `ytob_dc / 84` in dequantized XYB, and the two planes are
    // stored in different units, hence the same `DC_QUANT[1] / DC_QUANT[2]`
    // ratio `enc_group` applies for the `base_correlation_b` term.
    let step = (INV_DC_QUANT[2] / dc_step[2] * (DC_QUANT[1] * dc_step[1])) / color_factor;
    let seed = ytob_dc_seed(
        dc_datas,
        accumulate_weights,
        fill_residuals,
        rb_scratch,
        ry_scratch,
        step,
    );

    // Refine over a window around the fit; 0 is always a candidate so the
    // search can decline.
    const WINDOW: i32 = 6;
    let mut price = DcGradientPrice::default();
    let base_cost = ytob_dc_cost(dc_datas, 0, step, fill_ytob_row, rb_scratch, &mut price);
    let mut best = (base_cost - COLOR_CORRELATION_HEADER_BITS, 0i32);
    let candidates = (seed - WINDOW).max(-YTOB_DC_LIMIT)..=(seed + WINDOW).min(YTOB_DC_LIMIT);
    for k in candidates.filter(|&k| k != 0) {
        let cost = ytob_dc_cost(dc_datas, k, step, fill_ytob_row, rb_scratch, &mut price);
        if cost < best.0 {
            best = (cost, k);
        }
    }
    best.1
}

/// ExtraSlow B search: retain the fitted candidate window, but quantize every
/// candidate from the source and rank it in pooled weighted-predictor contexts.
pub(crate) fn choose_ytob_dc_weighted(
    dc_datas: &[DcGroupData],
    ctx: &EncodingContext,
    scale_dc: f32,
    dc_step: [f32; 3],
    rb_scratch: &mut Vec<i32>,
    ry_scratch: &mut Vec<i32>,
) -> i32 {
    if dc_datas.is_empty() || dc_datas.iter().any(|dc| dc.source_dc_b.is_none()) {
        return 0;
    }
    let cfl = ctx.cfl_frame();
    let step = (INV_DC_QUANT[2] / dc_step[2] * (DC_QUANT[1] * dc_step[1])) / cfl.color_factor;
    let seed = ytob_dc_seed(
        dc_datas,
        ctx.accumulate_ytob_weights,
        ctx.fill_ytob_residuals,
        rb_scratch,
        ry_scratch,
        step,
    );
    let mut price = crate::frame::DcPlanePrice::default();
    let mut levels = Vec::new();
    let mut cost = |k| {
        ytob_dc_weighted_bits(
            dc_datas,
            k,
            INV_DC_QUANT[2] / dc_step[2] * scale_dc,
            dc_step,
            cfl,
            ctx.quantize_dc_cfl,
            &mut price,
            &mut levels,
        )
    };
    let header = if cfl == CflFrame::XYB {
        COLOR_CORRELATION_HEADER_BITS
    } else {
        0.0
    };
    let mut best = (cost(0), 0);
    const WINDOW: i32 = 6;
    for k in ((seed - WINDOW).max(-YTOB_DC_LIMIT)..=(seed + WINDOW).min(YTOB_DC_LIMIT))
        .filter(|&k| k != 0)
    {
        let bits = cost(k) + header;
        if bits < best.0 {
            best = (bits, k);
        }
    }
    best.1
}

/// Check the chosen DC predictor on the unrounded source using the same
/// quantizer as coefficient coding. The initial search operates on integers.
/// The B DC's share of the stored Y DC: `base_correlation_b` (= 1) plus the
/// signaled `ytob_dc / 84`, converted from dequantized XYB into stored-B-DC
/// units under the frame's DC steps.
#[inline]
pub(crate) fn dc_cfl_factor(dc_step: [f32; 3], ytob_dc: i32, cfl: CflFrame) -> f32 {
    INV_DC_QUANT[2] / dc_step[2]
        * (DC_QUANT[1] * dc_step[1])
        * (cfl.base_b + ytob_dc as f32 / cfl.color_factor)
}

/// The X DC's share of the stored Y DC, in stored-X-DC units under the
/// frame's DC steps. DC callers fold `ytox_dc` into `base_x` first; AC keeps
/// the original frame correlations.
#[inline]
pub(crate) fn dc_cfl_factor_x(dc_step: [f32; 3], base_x: f32) -> f32 {
    INV_DC_QUANT[0] / dc_step[0] * (DC_QUANT[1] * dc_step[1]) * base_x
}

#[derive(Clone, Copy, Debug)]
struct XdcCost {
    bits: f64,
    distortion: f64,
}

/// Re-quantize B directly from its fractional source, including the zero
/// candidate, as `validate_ytob_dc` does. Pool weighted-predictor contexts
/// across groups using the DC coder's shared walk.
#[allow(clippy::too_many_arguments)]
fn ytob_dc_weighted_bits(
    dc_datas: &[DcGroupData],
    k: i32,
    scale: f32,
    dc_step: [f32; 3],
    cfl: CflFrame,
    quantize: crate::group::QuantizeDcCflFn,
    price: &mut crate::frame::DcPlanePrice,
    levels: &mut Vec<i16>,
) -> f64 {
    let factor = dc_cfl_factor(dc_step, k, cfl);
    price.clear();
    for dc in dc_datas {
        let source = dc.source_dc_b.as_ref().unwrap();
        let (w, h) = (source.xsize(), source.ysize());
        levels.resize(w * h, 0);
        for y in 0..h {
            let luma = dc.quant_dc.plane_row(1, y);
            let row = &mut levels[y * w..][..w];
            quantize(source.row(y), luma, scale, factor, row);
        }
        price.add(levels, w, h, w);
    }
    price.bits()
}

/// Confirm the gradient proposal under the coder's default weighted
/// predictor and context layout. Small gradient-only gains can otherwise
/// disappear when the DC entropy search chooses weighted prediction.
fn ytox_dc_weighted_bits(
    dc_datas: &[DcGroupData],
    k: i32,
    scale: f32,
    dc_step: [f32; 3],
    cfl: CflFrame,
    quantize: crate::group::QuantizeDcCflFn,
) -> f64 {
    let factor = dc_cfl_factor_x(dc_step, y_to_x_ratio(cfl, k as i8));
    let mut total = 0.0;
    for dc in dc_datas {
        let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
        let mut levels = Vec::with_capacity(w * h);
        if k == 0 {
            levels.extend(
                dc.quant_dc
                    .plane(0)
                    .as_slice()
                    .iter()
                    .map(|&v| i32::from(v)),
            );
        } else {
            let source = dc.source_dc_x.as_ref().unwrap();
            let mut row = vec![0i16; w];
            for y in 0..h {
                quantize(
                    source.row(y),
                    dc.quant_dc.plane_row(1, y),
                    scale,
                    factor,
                    &mut row,
                );
                levels.extend(row.iter().map(|&v| i32::from(v)));
            }
        }
        total += crate::frame::price_dc_plane(&levels, w, h, w);
    }
    total
}

/// Price X after quantizing the original fractional source against the final
/// stored luma. The zero arm uses the actual incumbent integers. Predictors
/// restart at DC-group boundaries, as they do in the Modular stream.
fn ytox_dc_cost(
    dc_datas: &[DcGroupData],
    k: i32,
    scale: f32,
    dc_step: [f32; 3],
    cfl: CflFrame,
    quantize: crate::group::QuantizeDcCflFn,
    price: &mut DcGradientPrice,
    current: &mut [i16],
) -> XdcCost {
    let ratio = y_to_x_ratio(cfl, k as i8);
    let factor = dc_cfl_factor_x(dc_step, ratio);
    price.clear();
    let mut distortion = 0.0;
    for dc in dc_datas {
        let source = dc.source_dc_x.as_ref().unwrap();
        let w = source.xsize();
        let current = &mut current[..w];
        for y in 0..source.ysize() {
            let luma = dc.quant_dc.plane_row(1, y);
            if k == 0 {
                current.copy_from_slice(dc.quant_dc.plane_row(0, y));
            } else {
                quantize(source.row(y), luma, scale, factor, current);
            }
            price.add_row(current, y == 0);
            for ((&src, &luma), &level) in source.row(y).iter().zip(luma).zip(current.iter()) {
                let target = fmla(src, scale, -(luma as f32) * factor);
                distortion += f64::from(target - f32::from(level)).powi(2);
            }
        }
    }
    let bits = price.bits();
    XdcCost { bits, distortion }
}

/// Search every signaled X DC slope. A candidate must save its incremental
/// header under gradient and weighted-predictor prices, and must not increase
/// squared DC reconstruction error against the
/// unrounded source. Apply the winner with the coding quantizer, never by
/// subtracting from previously rounded X integers.
pub(crate) fn choose_ytox_dc(
    dc_datas: &mut [DcGroupData],
    scale_dc: f32,
    dc_step: [f32; 3],
    cfl: CflFrame,
    header_already_paid: bool,
    quantize: crate::group::QuantizeDcCflFn,
) -> i32 {
    if dc_datas.is_empty() || dc_datas.iter().any(|dc| dc.source_dc_x.is_none()) {
        return 0;
    }
    let width = dc_datas
        .iter()
        .map(|dc| dc.quant_dc.xsize())
        .max()
        .unwrap_or(0);
    let mut gradient_price = DcGradientPrice::default();
    let mut current = vec![0i16; width];
    let scale = INV_DC_QUANT[0] / dc_step[0] * scale_dc;
    let base = ytox_dc_cost(
        dc_datas,
        0,
        scale,
        dc_step,
        cfl,
        quantize,
        &mut gradient_price,
        &mut current,
    );
    let header = if header_already_paid || cfl != CflFrame::XYB {
        0.0
    } else {
        COLOR_CORRELATION_HEADER_BITS
    };
    let (mut best_bits, mut best_k) = (base.bits, 0);
    for k in (-128..=127).filter(|&k| k != 0) {
        let cost = ytox_dc_cost(
            dc_datas,
            k,
            scale,
            dc_step,
            cfl,
            quantize,
            &mut gradient_price,
            &mut current,
        );
        if cost.distortion <= base.distortion && cost.bits + header < best_bits {
            best_bits = cost.bits + header;
            best_k = k;
        }
    }
    if best_k != 0 {
        let base_bits = ytox_dc_weighted_bits(dc_datas, 0, scale, dc_step, cfl, quantize);
        let proposed_bits = ytox_dc_weighted_bits(dc_datas, best_k, scale, dc_step, cfl, quantize);
        if proposed_bits + header >= base_bits {
            best_k = 0;
        }
    }
    if best_k != 0 {
        let factor = dc_cfl_factor_x(dc_step, y_to_x_ratio(cfl, best_k as i8));
        for dc in dc_datas {
            let source = dc.source_dc_x.as_ref().unwrap();
            for row in 0..source.ysize() {
                let [x, y, _] = dc.quant_dc.all_plane_rows_mut(row);
                quantize(source.row(row), y, scale, factor, x);
            }
        }
    }
    best_k
}

/// DC step candidates per channel: the default and finer ones a percent
/// apart.
const DC_STEP_CANDIDATES: usize = 9;
/// Bits of the three signaled steps.
const DC_STEP_HEADER_BITS: f64 = 48.0;
/// Share of the DC rate a custom step must save on top of its header.
const DC_STEP_MIN_GAIN: f64 = 0.02;

/// A DC step multiplier as the decoder reads it back from its 16-bit wire
/// form.
pub(crate) fn signaled_dc_step(channel: usize, multiplier: f32) -> f32 {
    let wire = DC_QUANT[channel] * 128.0 * multiplier;
    crate::util::f16_bits_to_f32(crate::util::f32_to_f16_bits(wire)) / (DC_QUANT[channel] * 128.0)
}

/// The wire form of a frame's DC steps, or `None` for the defaults.
pub(crate) fn dc_step_wire(dc_step: [f32; 3]) -> Option<[u16; 3]> {
    (dc_step != [1.0; 3]).then(|| {
        std::array::from_fn(|c| crate::util::f32_to_f16_bits(DC_QUANT[c] * 128.0 * dc_step[c]))
    })
}

/// Choose the frame's Y and B DC steps.
///
/// A smooth region holds one value per channel. Where that value falls
/// between two levels of the DC quantizer its blocks round either way, and
/// the DC plane carries the flicker: more bits, and visible noise on flat
/// ground. A slightly finer step moves the levels under the value. Textured
/// images have no such value and keep the default, which costs no header.
pub(crate) fn choose_dc_steps(
    dc_datas: &[DcGroupData],
    ctx: &EncodingContext,
    scale_dc: f32,
    ytob_dc: i32,
    base_step: [f32; 3],
    cfl: CflFrame,
) -> [f32; 3] {
    let default = base_step;
    if dc_datas
        .iter()
        .any(|dc| dc.source_dc_y.is_none() || dc.source_dc_b.is_none())
    {
        return default;
    }
    let candidates: [f32; DC_STEP_CANDIDATES] = std::array::from_fn(|k| 1.0 - 0.01 * k as f32);
    // Y levels under `step_y`, and their rate.
    let mut y_price = DcGradientPrice::default();
    let mut quantize_y = |step_y: f32, y_levels: &mut [Vec<i32>]| {
        let factor_y = INV_DC_QUANT[1] / (base_step[1] * step_y) * scale_dc;
        y_price.clear();
        for (dc, levels) in dc_datas.iter().zip(y_levels) {
            let source = dc.source_dc_y.as_ref().unwrap();
            let w = source.xsize();
            for y in 0..source.ysize() {
                let row = &mut levels[y * w..][..w];
                (ctx.quantize_dc_i32)(source.row(y), factor_y, row);
                y_price.add_row(row, y == 0);
            }
        }
        y_price.bits()
    };
    // Rate of B under both steps, against the Y levels of `step_y`.
    let max_width = dc_datas
        .iter()
        .map(|dc| dc.quant_dc.xsize())
        .max()
        .unwrap_or(0);
    let mut current = vec![0i32; max_width];
    let mut b_price = DcGradientPrice::default();
    let mut rate_b = |step_y: f32, step_b: f32, y_levels: &[Vec<i32>]| {
        let factor_b = INV_DC_QUANT[2] / (base_step[2] * step_b) * scale_dc;
        let cfl = dc_cfl_factor(
            [base_step[0], base_step[1] * step_y, base_step[2] * step_b],
            ytob_dc,
            cfl,
        );
        b_price.clear();
        for (dc, levels) in dc_datas.iter().zip(y_levels) {
            let source = dc.source_dc_b.as_ref().unwrap();
            let w = source.xsize();
            let current = &mut current[..w];
            for y in 0..source.ysize() {
                let luma = &levels[y * w..][..w];
                (ctx.quantize_dc_cfl_i32)(source.row(y), luma, factor_b, cfl, current);
                b_price.add_row(current, y == 0);
            }
        }
        b_price.bits()
    };
    let mut y_levels: Vec<Vec<i32>> = dc_datas
        .iter()
        .map(|dc| vec![0i32; dc.quant_dc.xsize() * dc.quant_dc.ysize()])
        .collect();
    // Multipliers of the base step, rounded through the wire like the
    // signaled steps they become.
    let steps_of = |channel: usize| {
        candidates
            .map(|step| signaled_dc_step(channel, base_step[channel] * step) / base_step[channel])
    };
    let (steps_y, steps_b) = (steps_of(1), steps_of(2));
    let mut costs = [[None; DC_STEP_CANDIDATES]; DC_STEP_CANDIDATES];
    let mut current_y = None;
    let mut rate_y = 0.0;
    let mut cost = |y: usize, b: usize| {
        if let Some(cost) = costs[y][b] {
            return cost;
        }
        if current_y != Some(y) {
            rate_y = quantize_y(steps_y[y], &mut y_levels);
            current_y = Some(y);
        }
        let cost = rate_y + rate_b(steps_y[y], steps_b[b], &y_levels);
        costs[y][b] = Some(cost);
        cost
    };
    // B depends on the Y levels, so the two steps are searched in turn: B,
    // then Y under that B, then B again. Revisited pairs retain their exact
    // costs and tie order; when Y stays put the last B search needs no scans.
    let base = cost(0, 0);
    let (mut best, mut step_y, mut step_b) = (base, 0, 0);
    for round in 0..3 {
        if round == 1 {
            for candidate in 0..DC_STEP_CANDIDATES {
                let candidate_cost = cost(candidate, step_b);
                if candidate_cost < best {
                    (best, step_y) = (candidate_cost, candidate);
                }
            }
        } else {
            for candidate in 0..DC_STEP_CANDIDATES {
                let candidate_cost = cost(step_y, candidate);
                if candidate_cost < best {
                    (best, step_b) = (candidate_cost, candidate);
                }
            }
        }
    }
    if best + DC_STEP_HEADER_BITS + DC_STEP_MIN_GAIN * base < base {
        [
            base_step[0],
            base_step[1] * steps_y[step_y],
            base_step[2] * steps_b[step_b],
        ]
    } else {
        default
    }
}

/// Subtracting two rounded values can invent a coding gain that disappears
/// when the source is re-quantized. Recheck zero and the proposed slope using
/// the existing gradient-token rate proxy, including its header charge.
pub(crate) fn validate_ytob_dc(
    dc_datas: &[DcGroupData],
    candidate: i32,
    scale_dc: f32,
    dc_step: [f32; 3],
    cfl_frame: CflFrame,
    quantize: crate::group::QuantizeDcCflFn,
) -> i32 {
    if candidate == 0 {
        return 0;
    }
    let mut price = DcGradientPrice::default();
    let mut current = Vec::new();
    let mut cost = |k: i32| {
        price.clear();
        let cfl = dc_cfl_factor(dc_step, k, cfl_frame);
        for dc in dc_datas {
            let Some(source) = &dc.source_dc_b else {
                return f64::INFINITY;
            };
            let w = source.xsize();
            let h = source.ysize();
            current.resize(w, 0i16);
            for y in 0..h {
                let yr = dc.quant_dc.plane_row(1, y);
                quantize(
                    source.row(y),
                    yr,
                    INV_DC_QUANT[2] / dc_step[2] * scale_dc,
                    cfl,
                    &mut current,
                );
                price.add_row(&current, y == 0);
            }
        }
        price.bits()
    };
    let base = cost(0);
    let proposed = cost(candidate);
    if proposed + COLOR_CORRELATION_HEADER_BITS < base {
        candidate
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_dc_price_matches_group_local_reference() {
        fn check<T: Copy>(sample: impl Fn(usize) -> T)
        where
            i32: From<T>,
        {
            let mut price = DcGradientPrice::default();
            for candidate in 0..3 {
                price.clear();
                let (mut hist, mut extra, mut total) = ([0u64; 64], 0u64, 0u64);
                // Reuse scratch across changing group sizes, including empty
                // planes, one-column/one-row planes, SIMD widths and tails.
                for w in [33, 1, 0, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 257] {
                    for h in [0, 1, 2, 5] {
                        let data: Vec<T> = (0..w * h).map(|i| sample(i + candidate * 97)).collect();
                        for y in 0..h {
                            let row = &data[y * w..][..w];
                            price.add_row(row, y == 0);
                            for x in 0..w {
                                let left = if x > 0 {
                                    i32::from(row[x - 1])
                                } else if y > 0 {
                                    i32::from(data[(y - 1) * w + x])
                                } else {
                                    0
                                };
                                let north = if y > 0 {
                                    i32::from(data[(y - 1) * w + x])
                                } else {
                                    left
                                };
                                let northwest = if x > 0 && y > 0 {
                                    i32::from(data[(y - 1) * w + x - 1])
                                } else {
                                    left
                                };
                                let prediction = (i64::from(north) + i64::from(left)
                                    - i64::from(northwest))
                                .clamp(i64::from(north.min(left)), i64::from(north.max(left)));
                                let residual = (i64::from(i32::from(row[x])) - prediction) as i32;
                                let (token, nbits, _) = crate::entropy::uint_encode(
                                    crate::entropy::pack_signed(residual),
                                );
                                hist[(token as usize).min(hist.len() - 1)] += 1;
                                extra += u64::from(nbits);
                            }
                        }
                        total += (w * h) as u64;
                    }
                }
                assert_eq!(price.hist, hist);
                assert_eq!(price.extra_bits, extra);
                assert_eq!(price.total, total);
                assert_eq!(price.bits(), dc_histogram_bits(&hist, extra, total));
            }
        }

        let sample = |i: usize| match i % 5 {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => 0,
            _ => (i.wrapping_mul(7919) ^ i.wrapping_mul(104729)) as i16,
        };
        check(sample);
        check(|i| i32::from(sample(i)) * 31);
    }

    fn x_dc_fixture(k: i32, fraction: f32, w: usize, h: usize) -> DcGroupData {
        let ctx = EncodingContext::default();
        let mut dc = DcGroupData::new(w, h).unwrap();
        let mut source = crate::image::Plane::new(w, h);
        let factor = dc_cfl_factor_x([1.0; 3], y_to_x_ratio(CflFrame::XYB, k as i8));
        let mut state = 19u32;
        for row in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let y = ((state >> 16) % 2001) as i16 - 1000;
                dc.quant_dc.plane_row_mut(1, row)[x] = y;
                dc.quant_dc.plane_row_mut(2, row)[x] = 23;
                source.row_mut(row)[x] = (factor * f32::from(y) + fraction) / INV_DC_QUANT[0];
            }
            (ctx.quantize_dc)(
                source.row(row),
                INV_DC_QUANT[0],
                dc.quant_dc.plane_row_mut(0, row),
            );
        }
        dc.source_dc_x = Some(source);
        dc
    }

    #[test]
    fn x_dc_search_recovers_signed_correlations_and_preserves_y_b() {
        let ctx = EncodingContext::default();
        for k in [-128, -42, -9, 5, 42, 127] {
            let dc = x_dc_fixture(k, 0.0, 48, 48);
            let y = dc.quant_dc.plane(1).as_slice().to_vec();
            let b = dc.quant_dc.plane(2).as_slice().to_vec();
            let mut groups = [dc];
            assert_eq!(
                choose_ytox_dc(
                    &mut groups,
                    1.0,
                    [1.0; 3],
                    CflFrame::XYB,
                    false,
                    ctx.quantize_dc_cfl
                ),
                k
            );
            assert!(
                groups[0]
                    .quant_dc
                    .plane(0)
                    .as_slice()
                    .iter()
                    .all(|&v| v == 0)
            );
            assert_eq!(groups[0].quant_dc.plane(1).as_slice(), y);
            assert_eq!(groups[0].quant_dc.plane(2).as_slice(), b);
        }
    }

    #[test]
    fn x_dc_search_quantizes_the_fractional_source_once() {
        let ctx = EncodingContext::default();
        let k = 5;
        let dc = x_dc_fixture(k, 0.15, 48, 48);
        let source = dc.source_dc_x.as_ref().unwrap();
        let factor = dc_cfl_factor_x([1.0; 3], y_to_x_ratio(CflFrame::XYB, k as i8));
        // Double rounding invents ones; direct source quantization yields zero.
        assert!(
            dc.quant_dc
                .plane(0)
                .as_slice()
                .iter()
                .zip(dc.quant_dc.plane(1).as_slice())
                .any(|(&x, &y)| i32::from(x) - (factor * f32::from(y)).round() as i32 != 0)
        );
        assert!(source.as_slice().iter().all(|v| v.is_finite()));
        let mut groups = [dc];
        assert_eq!(
            choose_ytox_dc(
                &mut groups,
                1.0,
                [1.0; 3],
                CflFrame::XYB,
                false,
                ctx.quantize_dc_cfl
            ),
            k
        );
        assert!(
            groups[0]
                .quant_dc
                .plane(0)
                .as_slice()
                .iter()
                .all(|&v| v == 0)
        );
    }

    #[test]
    fn x_dc_search_obeys_source_error_and_incremental_header() {
        let ctx = EncodingContext::default();
        // It would be cheap to store a zero residual, but a 0.49 phase has
        // much worse error than the baseline's nearest source rounding.
        let mut groups = [x_dc_fixture(5, 0.49, 48, 48)];
        let mut gradient_price = DcGradientPrice::default();
        let mut current = vec![0; 48];
        let base = ytox_dc_cost(
            &groups,
            0,
            INV_DC_QUANT[0],
            [1.0; 3],
            CflFrame::XYB,
            ctx.quantize_dc_cfl,
            &mut gradient_price,
            &mut current,
        );
        let cheap = ytox_dc_cost(
            &groups,
            5,
            INV_DC_QUANT[0],
            [1.0; 3],
            CflFrame::XYB,
            ctx.quantize_dc_cfl,
            &mut gradient_price,
            &mut current,
        );
        assert!(cheap.bits < base.bits);
        assert!(cheap.distortion > base.distortion);
        let chosen = choose_ytox_dc(
            &mut groups,
            1.0,
            [1.0; 3],
            CflFrame::XYB,
            false,
            ctx.quantize_dc_cfl,
        );
        assert_ne!(chosen, 5);
        let actual = ytox_dc_cost(
            &groups,
            chosen,
            INV_DC_QUANT[0],
            [1.0; 3],
            CflFrame::XYB,
            ctx.quantize_dc_cfl,
            &mut gradient_price,
            &mut current,
        );
        assert!(actual.distortion <= base.distortion);
        let mut tiny = [x_dc_fixture(42, 0.0, 1, 1)];
        assert_eq!(
            choose_ytox_dc(
                &mut tiny,
                1.0,
                [1.0; 3],
                CflFrame::XYB,
                false,
                ctx.quantize_dc_cfl
            ),
            0
        );
        assert_eq!(
            choose_ytox_dc(
                &mut tiny,
                1.0,
                [1.0; 3],
                CflFrame::XYB,
                true,
                ctx.quantize_dc_cfl
            ),
            42
        );
        let mut missing = [DcGroupData::new(48, 48).unwrap()];
        assert_eq!(
            choose_ytox_dc(
                &mut missing,
                1.0,
                [1.0; 3],
                CflFrame::XYB,
                false,
                ctx.quantize_dc_cfl
            ),
            0
        );
    }

    #[test]
    fn constant_image_returns_zero() {
        let mut opsin = Image3F::new(64, 64);
        for c in 0..3 {
            for y in 0..64 {
                let row = opsin.plane_row_mut(c, y);
                for x in 0..64 {
                    row[x] = 0.5;
                }
            }
        }
        let ctx = EncodingContext::default();
        let mut scratch = CflScratch {
            block_b: [0.; 64],
            block_x: [0.; 64],
            block_y: [0.; 64],
        };
        let (ytox, ytob) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0, &mut scratch);
        // No variation → no useful correlation → slope == 0 (the regression
        // collapses to numerator=0, denominator>0 → 0).
        assert_eq!(ytox, 0);
        assert_eq!(ytob, 0);
    }

    #[test]
    fn perfectly_correlated_chroma_finds_nonzero_slope() {
        // X = 0.3 * Y per pixel (within float roundoff). The regression
        // should land on roughly ytox = 0.3 * 84 ≈ 25.
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x ^ y) as f32) / 64.0; // varied, non-flat Y
                opsin.plane_row_mut(1, y)[x] = v; // Y
                opsin.plane_row_mut(0, y)[x] = 0.3 * v; // X = 0.3 Y
                opsin.plane_row_mut(2, y)[x] = v; // B = Y (no extra slope)
            }
        }
        let ctx = EncodingContext::default();
        let mut scratch = CflScratch {
            block_b: [0.; 64],
            block_x: [0.; 64],
            block_y: [0.; 64],
        };
        let (ytox, ytob) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0, &mut scratch);
        assert!((ytox - 25).abs() < 3, "ytox = {}, expected ~25", ytox);
        // B = Y → slope 1 = base; cmap should be near 0.
        assert!(ytob.abs() < 3, "ytob = {}, expected ~0", ytob);
    }

    #[test]
    fn selected_regression_matches_scalar() {
        let mut y = [0.0f32; 64];
        let mut x = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        let mut qm_x = [0.0f32; 64];
        let mut qm_b = [0.0f32; 64];
        for i in 0..64 {
            y[i] = ((i * 37 % 97) as f32 - 48.0) * 0.03125;
            x[i] = ((i * 19 % 89) as f32 - 44.0) * 0.046875;
            b[i] = ((i * 53 % 101) as f32 - 50.0) * 0.0234375;
            qm_x[i] = 0.5 + (i % 11) as f32 * 0.0625;
            qm_b[i] = 0.75 + (i % 7) as f32 * 0.09375;
        }

        let expected = cfl_regression_scalar(&y, &x, &b, &qm_x, &qm_b);
        let actual = selected_cfl_regression_fn()(&y, &x, &b, &qm_x, &qm_b);
        for i in 0..4 {
            let tolerance = 1e-5 * expected[i].abs().max(1.0);
            assert!(
                (actual[i] - expected[i]).abs() <= tolerance,
                "sum {i}: SIMD={}, scalar={}, tolerance={tolerance}",
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn selected_rdo_block_matches_scalar() {
        let mut y = [0.0f32; 64];
        let mut x = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        let mut qm_x = [0.0f32; 64];
        let mut qm_b = [0.0f32; 64];
        for i in 0..64 {
            y[i] = ((i * 37 % 97) as f32 - 48.0) * 0.03125;
            x[i] = ((i * 19 % 89) as f32 - 44.0) * 0.046875;
            b[i] = ((i * 53 % 101) as f32 - 50.0) * 0.0234375;
            qm_x[i] = 0.5 + (i % 11) as f32 * 0.0625;
            qm_b[i] = 0.75 + (i % 7) as f32 * 0.09375;
        }

        // DC is intentionally invalid: the RDO kernel must consume only the
        // 63 AC coefficients.
        y[0] = f32::NAN;
        x[0] = f32::NAN;
        b[0] = f32::NAN;
        qm_x[0] = f32::NAN;
        qm_b[0] = f32::NAN;

        let mut expected = [[0.0f32; 63]; 4];
        let [m_x, s_x, m_b, s_b] = &mut expected;
        cfl_rdo_block_scalar(m_x, s_x, m_b, s_b, &y, &x, &b, &qm_x, &qm_b, 1.375);

        let mut actual = [[0.0f32; 63]; 4];
        let [m_x, s_x, m_b, s_b] = &mut actual;
        selected_cfl_rdo_block_fn()(m_x, s_x, m_b, s_b, &y, &x, &b, &qm_x, &qm_b, 1.375);

        assert_eq!(actual, expected);
        assert!(actual.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn selected_rdo_stats_match_scalar() {
        let stats_fn = selected_cfl_rdo_stats_fn();
        for len in [0, 1, 3, 4, 7, 8, 15, 63, 64, CFL_RDO_TILE_AC] {
            let m: Vec<f32> = (0..len)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03125)
                .collect();
            let s: Vec<f32> = (0..len)
                .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.0234375)
                .collect();
            let expected = cfl_rdo_stats_scalar(&m, &s);
            let actual = stats_fn(&m, &s);
            for i in 0..4 {
                let tolerance = 1e-4 * expected[i].abs().max(1.0);
                assert!(
                    (actual[i] - expected[i]).abs() <= tolerance,
                    "stat {i} at len {len}: SIMD={}, scalar={}, tolerance={tolerance}",
                    actual[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn selected_closed_loop_cost_matches_scalar() {
        let cost_fn = selected_cfl_closed_loop_cost_fn();
        let quadrant_thresholds = [0.42, 0.57, 0.73, 0.91];
        let thresholds: [f32; 63] = std::array::from_fn(|i| {
            let coeff = i + 1;
            quadrant_thresholds[usize::from(coeff >= 32) * 2 + usize::from(coeff & 7 >= 4)]
        });
        for len in [0, 1, 3, 4, 7, 8, 15, 62, 63, 64, 127, CFL_RDO_TILE_AC] {
            let m: Vec<f32> = (0..len)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.07125)
                .collect();
            let s: Vec<f32> = (0..len)
                .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.05375)
                .collect();
            for factor in [-1.25, -0.1875, 0.0, 0.3125, 1.75] {
                let expected = cfl_closed_loop_cost_scalar(&m, &s, factor, &thresholds);
                let actual = cost_fn(&m, &s, factor, &thresholds);
                for i in 0..2 {
                    let tolerance = 2e-5 * expected[i].abs().max(1.0);
                    assert!(
                        (actual[i] - expected[i]).abs() <= tolerance,
                        "cost {i} at len {len}, factor {factor}: SIMD={}, scalar={}, tolerance={tolerance}",
                        actual[i],
                        expected[i]
                    );
                }
            }
        }
    }

    #[test]
    fn deadzone_schedule_ramps_from_full_to_zero() {
        // Full amount at/below LO, zero at/above HI, linear in between, and off
        // (byte-identical) at low quality.
        assert_eq!(cfl_deadzone(0.5), CFL_DEADZONE_AMOUNT);
        assert_eq!(cfl_deadzone(CFL_DEADZONE_LO), CFL_DEADZONE_AMOUNT);
        assert_eq!(cfl_deadzone(1.5), CFL_DEADZONE_AMOUNT * 0.5); // midpoint of 1.0..2.0
        assert_eq!(cfl_deadzone(CFL_DEADZONE_HI), 0.0);
        assert_eq!(cfl_deadzone(3.0), 0.0);
    }

    /// X = k*Y on every pixel, so the tile's slope is k*84.
    fn correlated_tile(k: f32) -> Image3F {
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x ^ y) as f32) / 64.0;
                opsin.plane_row_mut(1, y)[x] = v;
                opsin.plane_row_mut(0, y)[x] = k * v;
                opsin.plane_row_mut(2, y)[x] = v;
            }
        }
        opsin
    }

    #[test]
    fn deadzone_keeps_a_real_slope_whole() {
        let ctx = EncodingContext::default();
        let mut scratch = CflScratch {
            block_b: [0.; 64],
            block_x: [0.; 64],
            block_y: [0.; 64],
        };
        // ytox ≈ 25 and ≈ 4: both outside the deadzone, identical at every
        // distance.
        for k in [0.3f32, 0.045] {
            let opsin = correlated_tile(k);
            let (ytox_hq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 0.5, &mut scratch);
            let (ytox_lq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0, &mut scratch);
            assert_eq!(ytox_hq, ytox_lq, "k={k}");
            assert!((ytox_hq as f32 - k * 84.0).abs() <= 1.0, "k={k}: {ytox_hq}");
        }
    }

    #[test]
    fn deadzone_snaps_a_sub_threshold_slope_at_high_quality_only() {
        let ctx = EncodingContext::default();
        let mut scratch = CflScratch {
            block_b: [0.; 64],
            block_x: [0.; 64],
            block_y: [0.; 64],
        };
        // ytox ≈ 1, inside the 1.5 deadzone.
        let opsin = correlated_tile(1.0 / 84.0);
        let (ytox_hq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 0.5, &mut scratch);
        let (ytox_lq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0, &mut scratch);
        assert_eq!(ytox_hq, 0);
        assert_eq!(ytox_lq, 1);
    }

    fn rdo_tile(opsin: &Image3F, pred_x: i32, pred_b: i32) -> (i32, i32) {
        let ctx = EncodingContext::default();
        let mut scratch = CflScratch {
            block_b: [0.; 64],
            block_x: [0.; 64],
            block_y: [0.; 64],
        };
        let mut rdo = CflRdoScratch::new();
        let qf = crate::image::ImageB::try_new_fill(8, 8, 32).unwrap();
        compute_cmap_tile_rdo(
            &ctx,
            CflRdoTile {
                opsin,
                blocks: CflBlockRegion {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                quant: CflRdoQuantization {
                    field: &qf,
                    scale: 1.0 / 32.0,
                    distance: 1.0,
                    closed_loop: true,
                    block_x: 0,
                    block_y: 0,
                },
                prediction: CflTilePrediction {
                    ytox: pred_x,
                    ytob: pred_b,
                },
            },
            CflRdoTileScratch {
                dct: &mut scratch,
                rdo: &mut rdo,
            },
        )
    }

    #[test]
    fn rdo_keeps_full_slope_on_correlated_tiles() {
        // X = 0.3*Y: the RDO path must land on the least-squares slope (~25)
        // with no high-quality shrinkage — the anti-desaturation property the
        // default path's distance-scheduled deadzone gives up below d=2.
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x ^ y) as f32) / 64.0;
                opsin.plane_row_mut(1, y)[x] = v;
                opsin.plane_row_mut(0, y)[x] = 0.3 * v;
                opsin.plane_row_mut(2, y)[x] = v;
            }
        }
        let (ytox, ytob) = rdo_tile(&opsin, 0, 0);
        assert!((ytox - 25).abs() < 3, "ytox = {ytox}, expected ~25");
        assert!(ytob.abs() < 3, "ytob = {ytob}, expected ~0");
    }

    #[test]
    fn rdo_returns_zero_on_neutral_tiles() {
        // Achromatic content (X = 0, B = Y): both multipliers must stay 0.
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x * 7 + y * 13) % 31) as f32 / 62.0;
                opsin.plane_row_mut(1, y)[x] = v;
                opsin.plane_row_mut(0, y)[x] = 0.0;
                opsin.plane_row_mut(2, y)[x] = v;
            }
        }
        let (ytox, ytob) = rdo_tile(&opsin, 0, 0);
        assert_eq!(ytox, 0);
        assert_eq!(ytob, 0);
    }

    #[test]
    fn rdo_empty_tile_falls_back_to_small_predictions() {
        assert_eq!(
            optimize_channel_rdo(
                cfl_rdo_stats_scalar,
                cfl_closed_loop_cost_scalar,
                &[],
                &[],
                0,
                1.0,
                true,
                1.0,
                3,
                0.0,
                0.0,
                0.1,
                1.0,
                K_COLOR_FACTOR,
            ),
            3
        );
        assert_eq!(
            optimize_channel_rdo(
                cfl_rdo_stats_scalar,
                cfl_closed_loop_cost_scalar,
                &[],
                &[],
                0,
                1.0,
                true,
                1.0,
                100,
                0.0,
                0.0,
                0.1,
                1.0,
                K_COLOR_FACTOR,
            ),
            0
        );
    }

    fn dc_group_with_residual_correlation(slope: f32) -> DcGroupData {
        let (w, h) = (48usize, 48usize);
        let mut dc = DcGroupData::new(w, h).unwrap();
        let mut state = 12345u32;
        for y in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let yv = ((state >> 16) as i32 % 2000) - 1000;
                dc.quant_dc.plane_row_mut(1, y)[x] = yv as i16;
                dc.quant_dc.plane_row_mut(2, y)[x] = (slope * yv as f32).round_ties_even() as i16;
            }
        }
        dc
    }

    /// A DC group whose Y is `luma` and whose B source is `blue`, both
    /// unrounded.
    fn dc_group_with_sources(
        luma: impl Fn(usize, usize) -> f32,
        blue: impl Fn(usize, usize) -> f32,
    ) -> DcGroupData {
        let (w, h) = (48usize, 48usize);
        let mut dc = DcGroupData::new(w, h).unwrap();
        let mut source_y = crate::image::Plane::new(w, h);
        let mut source_b = crate::image::Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                source_y.row_mut(y)[x] = luma(x, y);
                source_b.row_mut(y)[x] = blue(x, y);
            }
        }
        dc.source_dc_y = Some(source_y);
        dc.source_dc_b = Some(source_b);
        dc
    }

    #[test]
    fn dc_steps_move_the_levels_under_a_flat_value() {
        // A flat B that sits half a level from its neighbours under the
        // default step, with a flicker that rounds it either way.
        let flicker = |x: usize, y: usize| {
            let mut state = (x as u32 * 7919 + y as u32 * 104729) | 1;
            for _ in 0..3 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
            }
            (state >> 16) as f32 / 65536.0 - 0.5
        };
        let flat = dc_group_with_sources(
            |_, _| 0.0,
            |x, y| (40.5 + 0.2 * flicker(x, y)) / INV_DC_QUANT[2],
        );
        let steps = choose_dc_steps(
            &[flat],
            &EncodingContext::default(),
            1.0,
            0,
            [1.0; 3],
            CflFrame::XYB,
        );
        assert_eq!(steps[0], 1.0);
        assert!(steps[2] < 1.0, "{steps:?}");
        // The chosen step is what the decoder reads back.
        let wire = dc_step_wire(steps).unwrap();
        assert_eq!(
            crate::util::f16_bits_to_f32(wire[2]),
            DC_QUANT[2] * 128.0 * steps[2]
        );
    }

    #[test]
    fn dc_steps_stay_default_on_textured_dc() {
        let noise = |seed: u32| {
            move |x: usize, y: usize| {
                let mut state = (x as u32 * 7919 + y as u32 * 104729 + seed) | 1;
                for _ in 0..3 {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                }
                (state >> 12) as f32 / 1_048_576.0 * 300.0 - 150.0
            }
        };
        let (luma, blue) = (noise(1), noise(99));
        let textured = dc_group_with_sources(
            |x, y| luma(x, y) / INV_DC_QUANT[1],
            |x, y| blue(x, y) / INV_DC_QUANT[2],
        );
        assert_eq!(
            choose_dc_steps(
                &[textured],
                &EncodingContext::default(),
                1.0,
                0,
                [1.0; 3],
                CflFrame::XYB,
            ),
            [1.0; 3]
        );
        assert!(dc_step_wire([1.0; 3]).is_none());
        // Without the first pass's sources there is nothing to choose from.
        let bare = DcGroupData::new(48, 48).unwrap();
        assert_eq!(
            choose_dc_steps(
                &[bare],
                &EncodingContext::default(),
                1.0,
                0,
                [1.0; 3],
                CflFrame::XYB,
            ),
            [1.0; 3]
        );
    }

    #[test]
    fn source_dc_guard_rejects_double_rounding_gain() {
        let ctx = EncodingContext::default();
        let (w, h) = (48, 48);
        let mut dc = DcGroupData::new(w, h).unwrap();
        let mut source = crate::image::Plane::new(w, h);
        let mut rng = 12345u32;
        for y in 0..h {
            for x in 0..w {
                let yv = (80.0 + 64.0 * (x as f32 * 0.15).sin() + 32.0 * (y as f32 * 0.13).sin())
                    .round() as i16;
                let bv = (yv as f32 / 168.0).round_ties_even() as i16;
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                let fraction = if rng & 0x80000000 == 0 { -0.49 } else { 0.49 };
                dc.quant_dc.plane_row_mut(1, y)[x] = yv;
                dc.quant_dc.plane_row_mut(2, y)[x] = bv;
                source.row_mut(y)[x] = (0.5 * yv as f32 + bv as f32 + fraction) / 256.0;
            }
        }
        // Source and stored baseline DC agree; the lost fraction is the only
        // difference between the old proxy and re-quantizing the real source.
        let mut row = vec![0i16; w];
        for y in 0..h {
            (ctx.quantize_dc_cfl)(
                source.row(y),
                dc.quant_dc.plane_row(1, y),
                256.0,
                0.5,
                &mut row,
            );
            assert_eq!(row, dc.quant_dc.plane_row(2, y));
        }
        dc.source_dc_b = Some(source);
        let groups = [dc];
        let chosen = choose_ytob_dc(
            &groups,
            ctx.fill_ytob_row,
            ctx.accumulate_ytob_weights,
            ctx.fill_ytob_residuals,
            &mut Vec::new(),
            &mut Vec::new(),
            [1.0; 3],
            K_COLOR_FACTOR,
        );
        assert_ne!(chosen, 0, "the integer proxy must propose a change");
        assert_eq!(
            validate_ytob_dc(
                &groups,
                chosen,
                1.0,
                [1.0; 3],
                CflFrame::XYB,
                ctx.quantize_dc_cfl
            ),
            0
        );
        assert_eq!(
            choose_ytob_dc_weighted(
                &groups,
                &ctx,
                1.0,
                [1.0; 3],
                &mut Vec::new(),
                &mut Vec::new(),
            ),
            0,
            "weighted search must also reject the invented rounding gain"
        );
    }

    #[test]
    fn weighted_b_dc_search_handles_group_borders_steps_and_missing_source() {
        let ctx = EncodingContext::default();
        let cfl = CflFrame {
            base_b: 0.75,
            color_factor: 128.0,
            ..CflFrame::XYB
        };
        ctx.set_cfl_frame(cfl);
        let steps = [0.9, 1.1, 0.8];
        let scale_dc = 0.71;
        let k = 17;
        let scale = INV_DC_QUANT[2] / steps[2] * scale_dc;
        let factor = dc_cfl_factor(steps, k, cfl);
        let mut groups = Vec::new();
        let mut state = 19u32;
        for (w, h) in [(1, 65), (65, 1), (7, 5)] {
            let mut dc = DcGroupData::new(w, h).unwrap();
            let mut source = crate::image::Plane::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let luma = ((state >> 16) % 4001) as i16 - 2000;
                    dc.quant_dc.plane_row_mut(1, y)[x] = luma;
                    source.row_mut(y)[x] = factor * f32::from(luma) / scale;
                }
                let [_, luma, b] = dc.quant_dc.all_plane_rows_mut(y);
                (ctx.quantize_dc_cfl)(source.row(y), luma, scale, dc_cfl_factor(steps, 0, cfl), b);
            }
            dc.source_dc_b = Some(source);
            groups.push(dc);
        }
        let original: Vec<_> = groups
            .iter()
            .map(|dc| dc.quant_dc.plane(2).as_slice().to_vec())
            .collect();
        assert_eq!(
            choose_ytob_dc_weighted(
                &groups,
                &ctx,
                scale_dc,
                steps,
                &mut Vec::new(),
                &mut Vec::new(),
            ),
            k
        );
        for (dc, original) in groups.iter().zip(original) {
            assert_eq!(dc.quant_dc.plane(2).as_slice(), original);
        }
        groups[1].source_dc_b = None;
        assert_eq!(
            choose_ytob_dc_weighted(
                &groups,
                &ctx,
                scale_dc,
                steps,
                &mut Vec::new(),
                &mut Vec::new(),
            ),
            0
        );
    }

    #[test]
    fn source_dc_guard_preserves_real_correlation() {
        let ctx = EncodingContext::default();
        for k in [-9, 5, 11] {
            let mut dc = dc_group_with_residual_correlation(k as f32 / 168.0);
            let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
            let mut source = crate::image::Plane::new(w, h);
            let factor = 0.5 * (1.0 + k as f32 / 84.0);
            for y in 0..h {
                for x in 0..w {
                    source.row_mut(y)[x] = factor * dc.quant_dc.plane_row(1, y)[x] as f32 / 256.0;
                }
            }
            dc.source_dc_b = Some(source);
            let groups = [dc];
            assert_eq!(
                validate_ytob_dc(
                    &groups,
                    k,
                    1.0,
                    [1.0; 3],
                    CflFrame::XYB,
                    ctx.quantize_dc_cfl
                ),
                k
            );
            assert_eq!(
                choose_ytob_dc_weighted(
                    &groups,
                    &ctx,
                    1.0,
                    [1.0; 3],
                    &mut Vec::new(),
                    &mut Vec::new(),
                ),
                k
            );
        }
    }

    #[test]
    fn ytob_dc_recovers_a_planted_dc_correlation() {
        let step = (INV_DC_QUANT[2] * DC_QUANT[1]) / K_COLOR_FACTOR;
        for k in [-9i32, -4, 5, 11] {
            let groups = [dc_group_with_residual_correlation(k as f32 * step)];
            // The planted slope is exactly representable, so the search must
            // land on it rather than merely near it.
            let mut s0 = Vec::new();
            let mut s1 = Vec::new();
            assert_eq!(
                choose_ytob_dc(
                    &groups,
                    selected_fill_ytob_row_fn(),
                    selected_accumulate_ytob_weights_fn(),
                    selected_fill_ytob_residuals_fn(),
                    &mut s0,
                    &mut s1,
                    [1.0; 3],
                    K_COLOR_FACTOR
                ),
                k,
                "planted ytob_dc {k}"
            );
        }
    }

    #[test]
    fn ytob_dc_declines_when_there_is_nothing_to_gain() {
        // No residual correlation: the explicit ColorCorrelationParams bundle
        // would cost header bits for nothing, so the search must return 0 and
        // leave the frame on the all-default bit.
        let groups = [dc_group_with_residual_correlation(0.0)];
        let mut s0 = Vec::new();
        let mut s1 = Vec::new();
        assert_eq!(
            choose_ytob_dc(
                &groups,
                selected_fill_ytob_row_fn(),
                selected_accumulate_ytob_weights_fn(),
                selected_fill_ytob_residuals_fn(),
                &mut s0,
                &mut s1,
                [1.0; 3],
                K_COLOR_FACTOR
            ),
            0
        );
    }

    #[test]
    fn ytob_dc_stays_in_the_signalled_u8_range() {
        // A correlation far steeper than the grid can express must still
        // produce a value `write_dc_global` can bias into a u8.
        for slope in [-4.0f32, 4.0] {
            let groups = [dc_group_with_residual_correlation(slope)];
            let mut s0 = Vec::new();
            let mut s1 = Vec::new();
            let k = choose_ytob_dc(
                &groups,
                selected_fill_ytob_row_fn(),
                selected_accumulate_ytob_weights_fn(),
                selected_fill_ytob_residuals_fn(),
                &mut s0,
                &mut s1,
                [1.0; 3],
                K_COLOR_FACTOR,
            );
            assert!((-YTOB_DC_LIMIT..=YTOB_DC_LIMIT).contains(&k), "ytob_dc {k}");
            assert!((0..=255).contains(&(k + 128)));
        }
    }
}
