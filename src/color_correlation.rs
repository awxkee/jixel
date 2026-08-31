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
use crate::adaptive_quant::dirty_log2f;
use crate::dc_group_data::DcGroupData;
use crate::dct::{DctInput, fmla};
use crate::encoding_context::EncodingContext;
use crate::entropy::{f_log2, pack_signed};
use crate::image::{Image3F, ImageSB};
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

fn solve_multiplier(ca: f32, cb: f32, num: usize, distance_mul: f32, dz: f32) -> i32 {
    if num == 0 {
        return 0;
    }
    let mut x = -cb / fmla(num as f32 * distance_mul, 0.5, ca);
    // libjxl `towards_zero` deadzone: shrink toward the base correlation and
    // snap sub-threshold slopes to it. No-op when `dz` is 0.
    if x >= dz {
        x -= dz;
    } else if x <= -dz {
        x += dz;
    } else {
        x = 0.0;
    }
    x.round().clamp(-128.0, 127.0) as i32
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

            let sums = (ctx.cfl_regression)(&block_y, &block_x, &block_b, qm_x, qm_b);
            ca_x += sums[0];
            cb_x += sums[1];
            ca_b += sums[2];
            cb_b += sums[3];
            num += 64;
        }
    }

    let dz = cfl_deadzone(distance);
    let ytox = solve_multiplier(ca_x, cb_x, num, K_DISTANCE_MULTIPLIER_AC, dz);
    let ytob = solve_multiplier(ca_b, cb_b, num, K_DISTANCE_MULTIPLIER_AC, dz);
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
    let target_cand = ((target_factor - base) * K_COLOR_FACTOR)
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
    let thresholds =
        crate::group::quantize_ac_thresholds_scaled(channel, 1, 1, distance, qm_multiplier);
    let mut best = (f32::MAX, 0i32);
    for &cand in &cands[..num_cands] {
        let factor = fmla(cand as f32, K_INV_COLOR_FACTOR, base);
        let mut distortion = 0.0f32;
        let mut coeff_bits = 0.0f32;
        if closed_loop {
            // Coefficients are staged as consecutive 63-AC blocks. Advance
            // that phase explicitly: `% 63` was otherwise paid for every
            // coefficient of every RDO candidate.
            let mut coeff = 1usize;
            for (&mv, &sv) in m.iter().zip(s) {
                let residual = sv - factor * mv;
                let quadrant = usize::from(coeff >= 32) * 2 + usize::from(coeff & 7 >= 4);
                let level = if residual.abs() >= thresholds[quadrant] {
                    residual.round()
                } else {
                    0.0
                };
                let reconstructed = crate::group::dequantized_level_f32(level);
                let quant_error = residual - reconstructed;
                // Closed-loop error is the primary term. A small residual-energy
                // term and coefficient-rate proxy retain CfL's compression goal
                // instead of choosing an expensive phase-aligned residual.
                distortion += quant_error.abs() + 0.15 * residual.abs();
                if level != 0.0 {
                    coeff_bits += 1.0 + dirty_log2f(1.0 + level.abs());
                }
                coeff += 1;
                if coeff == 64 {
                    coeff = 1;
                }
            }
        } else {
            for (&mv, &sv) in m.iter().zip(s) {
                let residual = sv - factor * mv;
                distortion += residual.abs();
            }
        }
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
    );
    let ytob = optimize_channel_rdo(
        ctx.cfl_rdo_stats,
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

    let use_rdo = distance < CFL_RDO.max_d && ctx.speed == crate::encode_image::Speed::Slow;

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
                            closed_loop: ctx.x_heavy() && distance < 1.75,
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
pub(crate) fn y_to_x_ratio(cmap_x: i8) -> f32 {
    // base_correlation_x = 0
    cmap_x as f32 * K_INV_COLOR_FACTOR
}

#[inline]
pub(crate) fn y_to_b_ratio(cmap_b: i8) -> f32 {
    // base_correlation_b = 1
    fmla(cmap_b as f32, K_INV_COLOR_FACTOR, 1.0)
}

pub(crate) const K_COLOR_FACTOR: f32 = 84.0;
const YTOB_DC_LIMIT: i32 = 127;
const COLOR_CORRELATION_HEADER_BITS: f64 = 2.0 + 16.0 + 16.0 + 8.0 + 8.0;

#[inline]
fn grad_predict(n: i32, w: i32, nw: i32) -> i32 {
    (n + w - nw).clamp(n.min(w), n.max(w))
}

#[inline]
fn add_ytob_token(residual: i32, hist: &mut [u64; 64], extra_bits: &mut u64) {
    let (tok, nbits, _) = crate::entropy::uint_encode(pack_signed(residual));
    hist[(tok as usize).min(hist.len() - 1)] += 1;
    *extra_bits += nbits as u64;
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
    prev: &mut Vec<i32>,
    cur: &mut Vec<i32>,
) -> f64 {
    let mut hist = [0u64; 64];
    let mut total = 0u64;
    let mut extra_bits = 0u64;
    let slope = k as f32 * step;

    for dc in dc_datas {
        let b = dc.quant_dc.plane(2);
        let y = dc.quant_dc.plane(1);
        let (xsize, ysize) = (b.xsize(), b.ysize());
        debug_assert_eq!((xsize, ysize), (y.xsize(), y.ysize()));
        if xsize == 0 || ysize == 0 {
            continue;
        }

        if prev.len() != xsize {
            prev.resize(xsize, 0i32);
        }
        if cur.len() != xsize {
            cur.resize(xsize, 0i32);
        }

        let mut rows = b
            .as_slice()
            .chunks_exact(xsize)
            .zip(y.as_slice().chunks_exact(xsize));

        // The top row predicts from its left neighbor only. Keeping it out of
        // the main loop removes the row-boundary branches from every sample.
        let (b_row, y_row) = rows.next().unwrap();
        fill_ytob_row(cur, b_row, y_row, slope);
        add_ytob_token(cur[0], &mut hist, &mut extra_bits);
        for pair in cur.array_windows::<2>() {
            add_ytob_token(pair[1] - pair[0], &mut hist, &mut extra_bits);
        }
        std::mem::swap(prev, cur);

        for (b_row, y_row) in rows {
            fill_ytob_row(cur, b_row, y_row, slope);
            add_ytob_token(cur[0] - prev[0], &mut hist, &mut extra_bits);
            for (current, above) in cur.array_windows::<2>().zip(prev.array_windows::<2>()) {
                let prediction = grad_predict(above[1], current[0], above[0]);
                add_ytob_token(current[1] - prediction, &mut hist, &mut extra_bits);
            }
            std::mem::swap(prev, cur);
        }
        total += (xsize * ysize) as u64;
    }

    if total == 0 {
        return 0.0;
    }

    let inv = 1.0 / total as f64;
    hist.iter()
        .filter(|&&count| count != 0)
        .fold(extra_bits as f64, |bits, &count| {
            bits - count as f64 * f_log2(count as f64 * inv)
        })
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

pub(crate) fn choose_ytob_dc(
    dc_datas: &[DcGroupData],
    fill_ytob_row: FillYtobRowFn,
    accumulate_weights: AccumulateYtobWeightsFn,
    fill_residuals: FillYtobResidualsFn,
    rb_scratch: &mut Vec<i32>,
    ry_scratch: &mut Vec<i32>,
) -> i32 {
    // One signaled step moves the stored B DC by this much per stored Y unit:
    // the slope is `ytob_dc / 84` in dequantized XYB, and the two planes are
    // stored in different units, hence the same `DC_QUANT[1] / DC_QUANT[2]`
    // ratio `enc_group` applies for the `base_correlation_b` term.
    let step = (INV_DC_QUANT[2] * DC_QUANT[1]) / K_COLOR_FACTOR;

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
    let seed = weights
        .iter()
        .position(|&weight| {
            acc += weight;
            acc > half
        })
        .map_or(0, |idx| idx as i32 - YTOB_DC_LIMIT);

    // Refine over a window around the fit; 0 is always a candidate so the
    // search can decline.
    const WINDOW: i32 = 6;
    let base_cost = ytob_dc_cost(dc_datas, 0, step, fill_ytob_row, ry_scratch, rb_scratch);
    let mut best = (base_cost - COLOR_CORRELATION_HEADER_BITS, 0i32);
    let candidates = (seed - WINDOW).max(-YTOB_DC_LIMIT)..=(seed + WINDOW).min(YTOB_DC_LIMIT);
    for k in candidates.filter(|&k| k != 0) {
        let cost = ytob_dc_cost(dc_datas, k, step, fill_ytob_row, ry_scratch, rb_scratch);
        if cost < best.0 {
            best = (cost, k);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn deadzone_schedule_ramps_from_full_to_zero() {
        // Full amount at/below LO, zero at/above HI, linear in between, and off
        // (byte-identical) at low quality.
        assert_eq!(cfl_deadzone(0.5), CFL_DEADZONE_AMOUNT);
        assert_eq!(cfl_deadzone(CFL_DEADZONE_LO), CFL_DEADZONE_AMOUNT);
        assert_eq!(cfl_deadzone(1.5), CFL_DEADZONE_AMOUNT * 0.5); // midpoint of 1.0..2.0
        assert_eq!(cfl_deadzone(CFL_DEADZONE_HI), 0.0);
        assert_eq!(cfl_deadzone(3.0), 0.0);
    }

    #[test]
    fn deadzone_shrinks_slope_at_high_quality_only() {
        // Perfectly correlated X = 0.3*Y (ytox ≈ 25). At HQ the deadzone shrinks
        // it toward base; at d ≥ HI it is left untouched.
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x ^ y) as f32) / 64.0;
                opsin.plane_row_mut(1, y)[x] = v;
                opsin.plane_row_mut(0, y)[x] = 0.3 * v;
                opsin.plane_row_mut(2, y)[x] = v;
            }
        }
        let ctx = EncodingContext::default();
        let mut scratch = CflScratch {
            block_b: [0.; 64],
            block_x: [0.; 64],
            block_y: [0.; 64],
        };
        let (ytox_hq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 0.5, &mut scratch);
        let (ytox_lq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0, &mut scratch);
        assert!(
            ytox_hq < ytox_lq,
            "deadzone should shrink the HQ slope: hq={ytox_hq} lq={ytox_lq}"
        );
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
            ),
            3
        );
        assert_eq!(
            optimize_channel_rdo(
                cfl_rdo_stats_scalar,
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
            );
            assert!((-YTOB_DC_LIMIT..=YTOB_DC_LIMIT).contains(&k), "ytob_dc {k}");
            assert!((0..=255).contains(&(k + 128)));
        }
    }
}
