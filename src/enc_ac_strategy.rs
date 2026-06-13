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
    STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT32X32,
};
use crate::dct::{dct4x4, dct4x8, dct8x4, dct8x8, dct8x16, dct16x8, dct16x16, dct32x32};
use crate::image::{Image3F, ImageB};
use crate::quant_weights::DequantMatrices;
use crate::util::FastRound;
use std::sync::OnceLock;

/// Rate-term lookup: `mag_bits` accumulates `log2(1 + |q|)` where `q` is always
/// a *rounded* (hence non-negative integer-valued) quantized coefficient. Since
/// `1 + |q|` is an exact integer-valued `f32`, the log can be precomputed once
/// and looked up — **bit-identical** to calling `f32::log2` (same inputs, same
/// `log2f`), but trading a per-coefficient transcendental for an L1 load. Values
/// of `|q|` at or above the table size (rare, large coefficients) fall back to
/// the direct computation.
const RATE_LOG2_LUT_N: usize = 1024;

#[inline]
pub(crate) fn rate_log2_lut() -> &'static [f32; RATE_LOG2_LUT_N] {
    static LUT: OnceLock<[f32; RATE_LOG2_LUT_N]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut a = [0.0f32; RATE_LOG2_LUT_N];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (1.0 + i as f32).log2();
        }
        a
    })
}

/// `log2(1 + qabs)` for a non-negative integer-valued `qabs`, via [`rate_log2_lut`].
#[inline]
pub(crate) fn rate_log2(qabs: f32) -> f32 {
    let k = qabs as usize;
    let lut = rate_log2_lut();
    if k < RATE_LOG2_LUT_N {
        lut[k]
    } else {
        (1.0 + qabs).log2()
    }
}

/// High-rate-optimal Lagrange multiplier for unit-step (Δ = 1) scalar
/// quantization: `λ* = Δ²·ln2 / 6`. Distortion is in quant-units², rate in
/// bits, so `λ·R` is in quant-units² and adds cleanly to D.
pub(crate) const RD_LAMBDA: f32 = 0.115_524_53;

/// Per-channel distortion weights. The dequant matrices already normalize each
/// channel perceptually, so equal weights are the principled default; X (red-
/// green) gets a touch more weight because the selection's CfL model omits the
/// Y→X subtraction (see `CMAP_FACTOR`), slightly under-counting its error.
static CHANNEL_WEIGHT: [f32; 3] = [1.0, 1.0, 1.0];

static CMAP_FACTOR: [f32; 3] = [0.0, 0.0, 1.0];

/// Fixed overhead per nonzero AC coefficient (context selection + sign).
const R_NZ_BASE: f32 = 1.6;
/// Weight on the magnitude term `log2(1+|q|)`.
const R_MAG: f32 = 1.0;
/// Weight on the `num_nonzeros` header term `log2(1+nzeros)`.
const R_HEADER: f32 = 0.4;

const BIAS_RECT: f32 = 0.92;
const BIAS_16X16: f32 = 0.86;
const BIAS_32X32: f32 = 1.0;

const BIAS_4X4: f32 = 1.0;
/// Merge bias for DCT4X8 vs the 8×8 it replaces (neutral, like the others).
/// DCT4X8 splits the block into two 4-tall halves, giving finer vertical
/// resolution — it tends to win on horizontal edges / horizontally-structured
/// detail where a full 8×8 would ring.
const BIAS_4X8: f32 = 1.0;

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
        for v in 0..h {
            let sy = (py + v).min(ph - 1);
            let row = plane.row(sy);
            for u in 0..w {
                let sx = (px + u).min(pw - 1);
                dst[v * w + u] = row[sx];
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
                dct8x8(src, dst);
                (1, 1)
            }
            STRATEGY_DCT16X8 => {
                gather(8, 16, &mut tmp[..128]);
                let src: &[f32; 128] = (&tmp[..128]).try_into().unwrap();
                let dst: &mut [f32; 128] = (&mut out[..128]).try_into().unwrap();
                dct16x8(src, dst);
                (2, 1)
            }
            STRATEGY_DCT8X16 => {
                gather(16, 8, &mut tmp[..128]);
                let src: &[f32; 128] = (&tmp[..128]).try_into().unwrap();
                let dst: &mut [f32; 128] = (&mut out[..128]).try_into().unwrap();
                dct8x16(src, dst);
                (2, 1)
            }
            STRATEGY_DCT16X16 => {
                gather(16, 16, &mut tmp[..256]);
                let src: &[f32; 256] = (&tmp[..256]).try_into().unwrap();
                let dst: &mut [f32; 256] = (&mut out[..256]).try_into().unwrap();
                dct16x16(src, dst);
                (2, 2)
            }
            STRATEGY_DCT32X32 => {
                gather(32, 32, &mut tmp[..1024]);
                let src: &[f32; 1024] = (&tmp[..1024]).try_into().unwrap();
                let dst: &mut [f32; 1024] = (&mut out[..1024]).try_into().unwrap();
                dct32x32(src, dst);
                (4, 4)
            }
            STRATEGY_DCT4X4 => {
                gather(8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                dct4x4(src, dst);
                (1, 1)
            }
            STRATEGY_DCT4X8 => {
                gather(8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                dct4x8(src, dst);
                (1, 1)
            }
            STRATEGY_DCT8X4 => {
                gather(8, 8, &mut tmp[..64]);
                let src: &[f32; 64] = (&tmp[..64]).try_into().unwrap();
                let dst: &mut [f32; 64] = (&mut out[..64]).try_into().unwrap();
                dct8x4(src, dst);
                (1, 1)
            }
            _ => unreachable!("invalid strategy {strategy}"),
        }
    })
}

/// Replicate `quantize_block_ac`'s per-quadrant thresholds for this channel and
/// transform shape (in 8-blocks).
#[inline]
fn thresholds(channel: usize, cx: usize, cy: usize) -> [f32; 4] {
    let mut thr = [0.58f32, 0.635, 0.66, 0.7];
    if channel == 0 {
        for t in thr.iter_mut().skip(1) {
            *t += 0.08;
        }
    }
    if channel == 2 {
        for t in thr.iter_mut().skip(1) {
            *t = 0.75;
        }
    }
    if cx > 1 || cy > 1 {
        let delta =
            (0.003_f32 * cx as f32 * cy as f32).clamp(0.0, if channel > 0 { 0.08 } else { 0.12 });
        for t in thr.iter_mut() {
            *t -= delta;
        }
    }
    thr
}

/// Quantize one channel exactly as the encoder will, accumulating the threshold-
/// aware squared quantization error (SSE, in quant-units²) and a rate estimate
/// (bits). LLF positions (`x < cx && y < cy`, coded via the DC plane) are
/// excluded from both, since DC coding is transform-choice-independent here.
fn channel_rd(
    coeff: &[f32],
    inv_matrix: &[f32],
    channel: usize,
    qac: f32,
    qm_mult: f32,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let width = cx * 8;
    let height = cy * 8;
    let half = width / 2;
    let thr = thresholds(channel, cx, cy);
    let q_scaled = qac * qm_mult;

    let (sse, nzeros, mag_bits) = sse_and_rate(
        coeff, inv_matrix, q_scaled, width, height, half, cx, cy, &thr,
    );

    let header = R_HEADER * rate_log2(nzeros as f32);
    let bits = nzeros as f32 * R_NZ_BASE + R_MAG * mag_bits + header;
    (sse, bits)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn sse_and_rate(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    thr: &[f32; 4],
) -> (f32, usize, f32) {
    type SseFunction = unsafe fn(
        &[f32],
        &[f32],
        f32,
        usize,
        usize,
        usize,
        usize,
        usize,
        &[f32; 4],
    ) -> (f32, usize, f32);
    static SSE_FUNCTION: OnceLock<SseFunction> = OnceLock::new();
    let f = SSE_FUNCTION.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
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
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            crate::enc_ac_strategy::sse_and_rate_scalar
        }
    });

    unsafe {
        f(
            coeff, inv_matrix, q_scaled, width, height, half, cx, cy, thr,
        )
    }
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
                mag_bits += rate_log2(q.abs());
            }
        }
    }
    (sse, nzeros, mag_bits)
}

/// Full RD cost `J = D + λR` of coding `strategy` at absolute pixel `(px, py)`.
/// Combines the three channels with the selection-time CfL approximation.
fn strategy_cost(
    strategy: u8,
    opsin: &Image3F,
    px: usize,
    py: usize,
    qac: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
) -> f32 {
    let mut cxy = (1usize, 1usize);
    SC_COEFFS_SCRATCH.with(|cell| {
        let coeffs = &mut *cell.borrow_mut();
        for c in 0..3 {
            cxy = forward_transform(strategy, opsin.plane(c), px, py, &mut coeffs[c]);
        }
        let (cx, cy) = cxy;
        let size = cx * cy * 64;

        // Apply the selection-time CfL model: B -= 1.0·Y (X unchanged).
        {
            let [_c0, c1, c2] = coeffs;
            let y = &c1[..size];
            for (b, &yi) in c2[..size].iter_mut().zip(y.iter()) {
                *b -= CMAP_FACTOR[2] * yi;
            }
        }

        let inv = |c: usize| -> &[f32] {
            match strategy {
                STRATEGY_DCT => &matrices.inv_matrix(c)[..],
                STRATEGY_DCT4X4 => &matrices.inv_matrix_4x4(c)[..],
                STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => &matrices.inv_matrix_4x8(c)[..],
                STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(c)[..],
                STRATEGY_DCT32X32 => &matrices.inv_matrix_32x32(c)[..],
                _ => &matrices.inv_matrix_16x8(c)[..],
            }
        };

        let mut d_total = 0.0f32;
        let mut r_total = 0.0f32;
        for c in 0..3 {
            let qm_mult = if c == 0 { qm_mult_x } else { 1.0 };
            let (d, r) = channel_rd(&coeffs[c][..size], inv(c), c, qac, qm_mult, cx, cy);
            d_total += CHANNEL_WEIGHT[c] * d;
            r_total += r;
        }
        d_total + RD_LAMBDA * r_total
    })
}

#[allow(clippy::too_many_arguments)]
fn select_super_block(
    opsin: &Image3F,
    bx0: usize,
    by0: usize,
    px0: usize,
    py0: usize,
    qac: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    ac_strategy: &mut AcStrategyImage,
) -> f32 {
    // Cost of the four individual DCT8 blocks: cost[dy][dx].
    let mut c8 = [[0.0f32; 2]; 2];
    for dy in 0..2 {
        for dx in 0..2 {
            c8[dy][dx] = strategy_cost(
                STRATEGY_DCT,
                opsin,
                px0 + dx * 8,
                py0 + dy * 8,
                qac,
                qm_mult_x,
                matrices,
            );
        }
    }

    // Vertical pairs (DCT16X8): one per column.
    let v_left = strategy_cost(STRATEGY_DCT16X8, opsin, px0, py0, qac, qm_mult_x, matrices);
    let v_right = strategy_cost(
        STRATEGY_DCT16X8,
        opsin,
        px0 + 8,
        py0,
        qac,
        qm_mult_x,
        matrices,
    );
    // Horizontal pairs (DCT8X16): one per row.
    let h_top = strategy_cost(STRATEGY_DCT8X16, opsin, px0, py0, qac, qm_mult_x, matrices);
    let h_bot = strategy_cost(
        STRATEGY_DCT8X16,
        opsin,
        px0,
        py0 + 8,
        qac,
        qm_mult_x,
        matrices,
    );
    // The single DCT16X16 over all four.
    let c16 = strategy_cost(STRATEGY_DCT16X16, opsin, px0, py0, qac, qm_mult_x, matrices);

    // Best column-wise DCT16X8 layout vs the two DCT8s it would replace.
    let cost_16x8 = (BIAS_RECT * v_left).min(c8[0][0] + c8[1][0])
        + (BIAS_RECT * v_right).min(c8[0][1] + c8[1][1]);
    // Best row-wise DCT8X16 layout.
    let cost_8x16 =
        (BIAS_RECT * h_top).min(c8[0][0] + c8[0][1]) + (BIAS_RECT * h_bot).min(c8[1][0] + c8[1][1]);
    let cost_16x16 = BIAS_16X16 * c16;
    let total_dct8 = c8[0][0] + c8[0][1] + c8[1][0] + c8[1][1];

    let best_rect = cost_16x8.min(cost_8x16);
    let pick_16x16 = cost_16x16 < best_rect
        && cost_16x16 < total_dct8
        && ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X16);
    if pick_16x16 {
        ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X16);
    } else if cost_16x8 <= cost_8x16 {
        if BIAS_RECT * v_left < c8[0][0] + c8[1][0]
            && ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X8)
        {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X8);
        }
        if BIAS_RECT * v_right < c8[0][1] + c8[1][1]
            && ac_strategy.can_place_strategy(bx0 + 1, by0, STRATEGY_DCT16X8)
        {
            ac_strategy.set_first(bx0 + 1, by0, STRATEGY_DCT16X8);
        }
    } else {
        if BIAS_RECT * h_top < c8[0][0] + c8[0][1]
            && ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT8X16)
        {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT8X16);
        }
        if BIAS_RECT * h_bot < c8[1][0] + c8[1][1]
            && ac_strategy.can_place_strategy(bx0, by0 + 1, STRATEGY_DCT8X16)
        {
            ac_strategy.set_first(bx0, by0 + 1, STRATEGY_DCT8X16);
        }
    }

    // Achieved RD cost of this 2×2 super-block's decision (used by the coarser
    // 4×4 / DCT32X32 level to decide whether to merge).
    if pick_16x16 { cost_16x16 } else { best_rect }
}

/// For each multi-block transform, propagate the maximum `raw_quant` across the
/// covered blocks so the per-block quant field is consistent within a transform
/// (libjxl-tiny `AdjustQuantField`).
pub(crate) fn adjust_quant_field(ac_strategy: &AcStrategyImage, quant_field: &mut ImageB) {
    for (x, y, raw_strategy) in ac_strategy.iter_first_blocks() {
        let cov_x = AcStrategyImage::covered_blocks_x_of(raw_strategy);
        let cov_y = AcStrategyImage::covered_blocks_y_of(raw_strategy);
        if cov_x == 1 && cov_y == 1 {
            continue;
        }
        let mut max_q: u8 = 0;
        for iy in 0..cov_y {
            for &q in &quant_field.row(y + iy)[x..x + cov_x] {
                max_q = max_q.max(q);
            }
        }
        for iy in 0..cov_y {
            for q in &mut quant_field.row_mut(y + iy)[x..x + cov_x] {
                *q = max_q;
            }
        }
    }
}

/// Select transforms for every aligned 2×2 super-block in the DC group, then
/// reconcile the quant field. `(dc_group_px, dc_group_py)` is the DC group's
/// top-left in absolute image pixels (so `opsin` can be the full image).
#[allow(clippy::too_many_arguments)]
/// Max quant over a w×h block region (mirrors `adjust_quant_field`), scaled.
#[inline]
fn region_qac(quant_field: &ImageB, bx: usize, by: usize, w: usize, h: usize, scale: f32) -> f32 {
    let mut q: u8 = 1;
    for iy in 0..h {
        for ix in 0..w {
            q = q.max(quant_field.row(by + iy)[bx + ix]);
        }
    }
    scale * q as f32
}

/// Run AC-strategy selection for block rows `by` in `[y_begin, y_end)` into
/// `ac_strategy`, returning the accumulated DCT4X4 RD benefit for those rows.
/// `ysize`/`xsize` are the *full* group dimensions: edge tests (`four_row`,
/// loop bounds) use the global size, exactly as the serial loop would, so a
/// 4-aligned `[y_begin, y_end)` partition reproduces the single-threaded
/// decision sequence bit-for-bit. Reads `quant_field`/`opsin` only.
#[allow(clippy::too_many_arguments)]
fn select_band(
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    scale: f32,
    qm_mult_x: f32,
    matrices: &DequantMatrices,
    quant_field: &ImageB,
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
                for sy in 0..2 {
                    for sx in 0..2 {
                        let sbx = bx + sx * 2;
                        let sby = by + sy * 2;
                        let qac = region_qac(quant_field, sbx, sby, 2, 2, scale);
                        sub_total += select_super_block(
                            opsin,
                            sbx,
                            sby,
                            dc_group_px + sbx * 8,
                            dc_group_py + sby * 8,
                            qac,
                            qm_mult_x,
                            matrices,
                            ac_strategy,
                        );
                    }
                }
                let qac32 = region_qac(quant_field, bx, by, 4, 4, scale);
                let cost32 = strategy_cost(
                    STRATEGY_DCT32X32,
                    opsin,
                    dc_group_px + bx * 8,
                    dc_group_py + by * 8,
                    qac32,
                    qm_mult_x,
                    matrices,
                );
                if BIAS_32X32 * cost32 < sub_total {
                    ac_strategy.set_first(bx, by, STRATEGY_DCT32X32);
                }
                bx += 4;
            } else if four_row {
                for sby in [by, by + 2] {
                    let qac = region_qac(quant_field, bx, sby, 2, 2, scale);
                    select_super_block(
                        opsin,
                        bx,
                        sby,
                        dc_group_px + bx * 8,
                        dc_group_py + sby * 8,
                        qac,
                        qm_mult_x,
                        matrices,
                        ac_strategy,
                    );
                }
                bx += 2;
            } else {
                let qac = region_qac(quant_field, bx, by, 2, 2, scale);
                select_super_block(
                    opsin,
                    bx,
                    by,
                    dc_group_px + bx * 8,
                    dc_group_py + by * 8,
                    qac,
                    qm_mult_x,
                    matrices,
                    ac_strategy,
                );
                bx += 2;
            }
        }
        by += if four_row { 4 } else { 2 };
    }

    // DCT4X4 refinement for this row band.
    let mut benefit = 0.0f32;
    for by in y_begin..y_end {
        for bx in 0..xsize {
            if ac_strategy.raw_strategy(bx, by) != STRATEGY_DCT {
                continue;
            }
            let qac = region_qac(quant_field, bx, by, 1, 1, scale);
            let px = dc_group_px + bx * 8;
            let py = dc_group_py + by * 8;
            let cost8 = strategy_cost(STRATEGY_DCT, opsin, px, py, qac, qm_mult_x, matrices);
            let cost4 =
                BIAS_4X4 * strategy_cost(STRATEGY_DCT4X4, opsin, px, py, qac, qm_mult_x, matrices);
            let cost48 =
                BIAS_4X8 * strategy_cost(STRATEGY_DCT4X8, opsin, px, py, qac, qm_mult_x, matrices);
            let cost84 =
                BIAS_4X8 * strategy_cost(STRATEGY_DCT8X4, opsin, px, py, qac, qm_mult_x, matrices);
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
    opsin: &Image3F,
    dc_group_px: usize,
    dc_group_py: usize,
    _distance: f32,
    scale: f32,
    x_qm_scale: u32,
    matrices: &DequantMatrices,
    quant_field: &mut ImageB,
    ac_strategy: &mut AcStrategyImage,
    num_threads: usize,
) -> f32 {
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    let qm_mult_x = 1.25f32.powf(x_qm_scale as f32 - 2.0);

    let bands = if num_threads > 1 && ysize >= 8 {
        selection_bands(ysize, num_threads)
    } else {
        vec![(0, ysize)]
    };

    let benefit = if bands.len() <= 1 {
        select_band(
            opsin,
            dc_group_px,
            dc_group_py,
            scale,
            qm_mult_x,
            matrices,
            quant_field,
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
        let results: Vec<(AcStrategyImage, f32)> = std::thread::scope(|s| {
            let handles: Vec<_> = bands
                .iter()
                .map(|&(y0, y1)| {
                    s.spawn(move || {
                        let mut local = AcStrategyImage::new(xsize, ysize);
                        let b = select_band(
                            opsin,
                            dc_group_px,
                            dc_group_py,
                            scale,
                            qm_mult_x,
                            matrices,
                            qf,
                            &mut local,
                            xsize,
                            ysize,
                            y0,
                            y1,
                        );
                        (local, b)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut benefit = 0.0f32;
        for (&(y0, y1), (local, b)) in bands.iter().zip(results.iter()) {
            ac_strategy.copy_rows_from(local, y0, y1);
            benefit += b;
        }
        benefit
    };

    adjust_quant_field(ac_strategy, quant_field);
    benefit
}
