/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

//! Adaptive B-bias opsin selection for bright-yellow content.

use crate::image::Image3F;
use crate::quant_weights::DequantMatrices;
use crate::xyb::{XybMatrix, rgb_to_xyb_pixel_f32, xyb_to_rgb_pixel_f32};

/// Spec forward-matrix row constants (duplicated from `xyb.rs` consts so a
/// patched-const experiment build changes them in one place only).
const SPEC_BIAS: f32 = crate::xyb::B_BIAS;
const B_R_RATIO: f32 = 0.243_422_69 / (0.243_422_69 + 0.204_767_45);

/// Candidate biases tried by the proxy search. The spec value must be first:
/// it is the baseline the regression penalty is measured against.
static CANDIDATE_BIASES: [f32; 3] = [SPEC_BIAS, 0.70, 0.85];

/// Detector sampling stride: 1/64 of pixels. Validated on the pathological
/// fractal case: every 8x8 phase still hits the risky region.
const SAMPLE_STRIDE: usize = 8;

/// Early-exit gate thresholds.
const STRONG_SCORE: f32 = 0.10;
const VERY_STRONG_SCORE: f32 = 0.40;
const REQUIRED_STRONG: usize = 4;

/// Weight of the general-content regression term against yellow-chroma loss.
const REGRESSION_WEIGHT: f32 = 4.0;

/// The proxy must still rank a biased row below spec
const REL_COST_RATIO: f32 = 0.90;

/// Tiny floor so a numerically-zero spec cost (no damage at all) never
/// switches. The old 0.02 bar was the photo-corpus kill switch; the
/// separating work moved to `YELLOW_EDGE_MIN`. Proxy cost scales roughly
/// linearly with distance, so this floor also sets where tier-1 releases at
/// ultra-high quality — keep it low enough that the release happens where
/// the biased-vs-spec delta is negligible (0.004 would have released near
/// d≈0.25, recreating the boundary step this replaced).
const MIN_SPEC_COST: f32 = 0.0005;

/// Mean immediate horizontal/vertical linear-luma gradient over strong
/// yellow samples must exceed this. Thin, high-frequency yellow structure is
/// what the AC deadzone + CfL residual-death annihilate (per-pixel proxy
/// quantization cannot see this, which is why proxy cost alone never
/// separated).
const YELLOW_EDGE_MIN: f32 = 0.24;

/// Tier-1 used to switch the matrix and b_qm_scale=5 off together at 1.25.
/// On Burning_Ship that boundary dropped 6.35 SS2 for only 3.4% rate. Keep
/// full strength through the validated band, then fade the matrix while the
/// integer B scale is released in smaller stages.
const STRONG_FADE_START: f32 = 1.25;
const MAX_DISTANCE: f32 = 1.55;

/// Tier-2 (smooth yellow, below the edge gate): the mild bias fires when
/// yellow is a *subject*, not an accent — at least this fraction of sampled
/// pixels must be strong yellow-risk. The edge statistic below then scales
/// the strength so a smooth yellow subject does not pay the same rate as
/// fine yellow structure.
const TIER2_AREA_MIN: f32 = 0.10;

/// Smooth yellow fields do not need the same matrix displacement as fine
/// yellow structure. Start the tier-2 bias only once neighboring luma starts
/// to vary, and reach the normal tier-2 target at the edge level that
/// separates the visibly desaturating photo cases in the yellow corpus.
const TIER2_EDGE_START: f32 = 0.010;
const TIER2_EDGE_FULL: f32 = 0.038;

/// Tier-2 targets are separate from the tier-1 proxy candidates. Moderate
/// yellow texture uses the lower target; only dense, highly textured yellow
/// approaches the old 0.70 target.
const TIER2_BIAS: f32 = 0.65;
const TIER2_PATHOLOGICAL_BIAS: f32 = 0.70;
const TIER2_PATHOLOGICAL_EDGE_START: f32 = 0.055;
const TIER2_PATHOLOGICAL_EDGE_FULL: f32 = 0.080;
const TIER2_PATHOLOGICAL_RISK_START: f32 = 0.10;
const TIER2_PATHOLOGICAL_RISK_FULL: f32 = 0.22;
const TIER2_BIAS_GRID_SCALE: f32 = 40.0;

/// Tier-2 distance cap: measured chroma rescue at d=1..2; beyond 2.5
/// untested. (No explicit low-d cutoff: at high quality the bias is cheap
/// and the tail damage small — measured +0.4..2.6% bytes at d=0.5.)
const TIER2_MAX_DISTANCE: f32 = 2.5;

/// Number of worst yellow samples in the tail term. Tiny regions (a banana,
/// a highlight rim) can be visually catastrophic while contributing nothing
/// to a mean.
const TAIL_SAMPLES: usize = 16;

/// Tier-2 bias ramp: zero at/below MIN, full strength across FULL..FADE_OUT
/// (where the tail rescue was measured, d=1..2), zero again at/above MAX.
/// Continuous by design — hard distance gates on content-adaptive features
/// keep turning into RD kinks on exactly the content they serve (measured
/// here: a hard 0.3 floor cost +3..5% bytes in one step, stacked on the
/// FLAT/SAT table switch). Tier-1 has no explicit floor at all — its proxy
/// guard (`MIN_SPEC_COST`) fades it out where predicted damage vanishes.
const TIER2_MIN_DISTANCE: f32 = 0.3;
const TIER2_FULL_DISTANCE: f32 = 0.8;
const TIER2_FADE_OUT_START: f32 = 2.0;

/// CfL base correlation: the decoder's B prediction is ytob·Y with this base
/// slope (libjxl kYToBRatio); B quantization error is measured around it.
const YTOB_BASE: f32 = 0.935_669;

/// Weight of the top-16 tail term inside the yellow cost (mean gets 1 − this).
const TAIL_WEIGHT: f32 = 0.4;

/// Mild (tier-2) and strong (tier-1) bias values; `CANDIDATE_BIASES[0]` is
/// spec and anchors the regression baseline in the tier-1 search.
const BIAS_MID: f32 = CANDIDATE_BIASES[1];
const BIAS_HI: f32 = CANDIDATE_BIASES[2];

#[inline(always)]
fn ramp(x: f32, lo: f32, hi: f32) -> f32 {
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Per-pixel score for the bright-yellow risk region, on linear RGB in [0,1].
/// 0.0 = uninteresting, 1.0 = extremely strong bright-yellow candidate.
/// Requires brightness, yellow chroma relative to signal, and R≈G balance
/// (rejecting orange/red and green content).
#[inline(always)]
pub(crate) fn yellow_pixel_risk(r: f32, g: f32, b: f32) -> f32 {
    let rg_hi = r.max(g);
    let rg_lo = r.min(g);

    let yellow = (rg_lo - b).max(0.0);
    let inv_hi = 1.0 / rg_hi.max(1e-5);
    let rel_yellow = yellow * inv_hi;
    let rg_balance = 1.0 - (r - g).abs() * inv_hi;

    let bright_w = ramp(rg_hi, 0.55, 0.85);
    let yellow_w = ramp(rel_yellow, 0.15, 0.45);
    let balance_w = ramp(rg_balance, 0.20, 0.80);

    bright_w * yellow_w * balance_w
}

/// Cheap stage-1 gate: does the image contain enough bright, reasonably pure
/// yellow for the B-quantization failure to matter at all? Examines 1/64 of
/// pixels with early exit.
pub(crate) fn has_yellow_risk(linear: &Image3F) -> bool {
    let (w, h) = (linear.xsize(), linear.ysize());
    let rp = linear.plane_data(0);
    let gp = linear.plane_data(1);
    let bp = linear.plane_data(2);

    let mut strong = 0usize;
    for y in (0..h).step_by(SAMPLE_STRIDE) {
        let row = y * w;
        for x in (0..w).step_by(SAMPLE_STRIDE) {
            let i = row + x;
            let score = yellow_pixel_risk(rp[i], gp[i], bp[i]);
            if score >= VERY_STRONG_SCORE {
                return true;
            }
            if score >= STRONG_SCORE {
                strong += 1;
                if strong >= REQUIRED_STRONG {
                    return true;
                }
            }
        }
    }
    false
}

fn invert3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let m = m.map(|v| v as f64);
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let adj = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    Some(adj.map(|v| (v * inv_det) as f32))
}

/// Build the forward/inverse matrix pair for a given B bias. The B row keeps
/// the spec r:g ratio and rows still sum to 1, so grays map identically; the
/// inverse is derived numerically so bias and matrix stay coupled.
pub(crate) fn matrix_for_bias(bias: f32) -> XybMatrix {
    if (bias - SPEC_BIAS).abs() < 1e-6 {
        return XybMatrix::SPEC;
    }
    let m20 = B_R_RATIO * (1.0 - bias);
    let m21 = (1.0 - B_R_RATIO) * (1.0 - bias);
    let m22 = 1.0 - m20 - m21;
    let mut fwd = XybMatrix::SPEC.fwd;
    fwd[6] = m20;
    fwd[7] = m21;
    fwd[8] = m22;
    // The forward mixing rows are well-conditioned for every bias in range;
    // fall back to spec if inversion ever fails rather than emitting a
    // stream whose signaled inverse does not match.
    match invert3x3(&fwd) {
        Some(inv) => XybMatrix { fwd, inv },
        None => XybMatrix::SPEC,
    }
}

#[inline(always)]
fn yellow_chroma(rgb: [f32; 3]) -> f32 {
    (rgb[0].min(rgb[1]) - rgb[2]).max(0.0)
}

/// Relative yellow-chroma error of a reconstructed pixel vs its source.
/// Symmetric: B quantization both desaturates yellows (B pulled up toward
/// the ytob·Y prediction) and oversaturates them (ringing overshoot below
/// it); both read as wrong color.
#[inline]
fn yellow_loss(src: [f32; 3], rec: [f32; 3]) -> f32 {
    let cs = yellow_chroma(src);
    if cs < 0.02 {
        return 0.0;
    }
    let cr = yellow_chroma(rec);
    (cs - cr).abs() / cs.max(0.02)
}

/// Luma-weighted squared RGB error for the regression term.
#[inline]
fn rgb_error(src: [f32; 3], rec: [f32; 3]) -> f32 {
    let dr = src[0] - rec[0];
    let dg = src[1] - rec[1];
    let db = src[2] - rec[2];
    0.25 * dr * dr + 0.50 * dg * dg + 0.25 * db * db
}

/// Effective per-channel pixel-domain quantization steps at `distance`,
/// derived from the real DCT8 dequant tables (mean AC step over the low/mid
/// band where saturated texture lives) at the effective AC quantizer
/// `K_AC_QUANT / distance`. Order: [X, Y, B].
fn proxy_steps(distance: f32) -> [f32; 3] {
    // Matches frame::compute_distance_params: the quant-field rebalancing
    // cancels, leaving q_eff ≈ K_AC_QUANT / distance.
    const K_AC_QUANT: f32 = 0.8;
    const RECIP_K_AC_QUANT: f32 = 1.0 / K_AC_QUANT;
    let matrices = DequantMatrices::new(distance);
    let mut steps = [0.0f32; 3];
    for (c, step) in steps.iter_mut().enumerate() {
        let table = matrices.matrix(c);
        let mut sum = 0.0f32;
        let mut n = 0u32;
        for ky in 0..8usize {
            for kx in 0..8usize {
                let band = kx + ky;
                if (1..=5).contains(&band) {
                    sum += table[ky * 8 + kx];
                    n += 1;
                }
            }
        }
        *step = (sum / n as f32) * distance * RECIP_K_AC_QUANT;
    }
    steps
}

#[inline(always)]
fn quant(v: f32, step: f32) -> f32 {
    (v / step).round() * step
}

struct SamplePixel {
    rgb: [f32; 3],
    risk: f32,
}

struct BiasScore {
    yellow_mean: f32,
    yellow_tail: f32,
    general_error: f32,
}

/// Simulate `linear RGB → XYB(bias) → quantize(distance) → RGB` on the
/// sampled pixels. B is quantized as the CfL residual around ytob·Y — the
/// actual mechanism that pulls yellows toward the luma prediction.
fn evaluate_bias(samples: &[SamplePixel], m: &XybMatrix, steps: [f32; 3]) -> BiasScore {
    let [step_x, step_y, step_b] = steps;

    let mut yellow_sum = 0.0f32;
    let mut yellow_weight = 0.0f32;
    let mut worst: Vec<f32> = Vec::with_capacity(TAIL_SAMPLES + 1);
    let mut general_sum = 0.0f32;

    for s in samples {
        let [r, g, b] = s.rgb;
        let (x, y, bx) = rgb_to_xyb_pixel_f32(m, r, g, b);
        let yq = quant(y, step_y);
        let xq = quant(x, step_x);
        let bq = quant(bx - YTOB_BASE * y, step_b) + YTOB_BASE * yq;
        let rec = xyb_to_rgb_pixel_f32(m, xq, yq, bq);

        general_sum += rgb_error(s.rgb, rec);

        if s.risk > 0.0 {
            let loss = yellow_loss(s.rgb, rec) * s.risk;
            yellow_sum += loss;
            yellow_weight += s.risk;
            let pos = worst.partition_point(|&w| w > loss);
            if pos < TAIL_SAMPLES {
                worst.insert(pos, loss);
                worst.truncate(TAIL_SAMPLES);
            }
        }
    }

    let yellow_mean = if yellow_weight > 0.0 {
        yellow_sum / yellow_weight
    } else {
        0.0
    };
    let yellow_tail = if worst.is_empty() {
        0.0
    } else {
        worst.iter().sum::<f32>() / worst.len() as f32
    };
    BiasScore {
        yellow_mean,
        yellow_tail,
        general_error: general_sum / samples.len().max(1) as f32,
    }
}

/// Sampled pixels plus the mean immediate horizontal/vertical linear-luma
/// gradient over the strong yellow samples (the `YELLOW_EDGE_MIN` gate
/// statistic). Both directions avoid making the selector orientation
/// dependent.
fn collect_samples(linear: &Image3F) -> (Vec<SamplePixel>, f32) {
    let (w, h) = (linear.xsize(), linear.ysize());
    let rp = linear.plane_data(0);
    let gp = linear.plane_data(1);
    let bp = linear.plane_data(2);
    let mut samples = Vec::with_capacity(h.div_ceil(SAMPLE_STRIDE) * w.div_ceil(SAMPLE_STRIDE));
    let mut edge_sum = 0.0f32;
    let mut edge_n = 0usize;
    let luma = |i: usize| 0.3 * rp[i] + 0.6 * gp[i] + 0.1 * bp[i];
    for y in (0..h).step_by(SAMPLE_STRIDE) {
        let row = y * w;
        for x in (0..w).step_by(SAMPLE_STRIDE) {
            let i = row + x;
            let rgb = [rp[i], gp[i], bp[i]];
            let risk = yellow_pixel_risk(rgb[0], rgb[1], rgb[2]);
            if risk > STRONG_SCORE {
                if x + 1 < w {
                    edge_sum += (luma(i + 1) - luma(i)).abs();
                    edge_n += 1;
                }
                if y + 1 < h {
                    edge_sum += (luma(i + w) - luma(i)).abs();
                    edge_n += 1;
                }
            }
            samples.push(SamplePixel { rgb, risk });
        }
    }
    let yellow_edge = if edge_n > 0 {
        edge_sum / edge_n as f32
    } else {
        0.0
    };
    (samples, yellow_edge)
}

/// Tier-2: smooth-yellow content (below the edge gate). Fires the mild bias
/// on yellow-subject images (see `TIER2_AREA_MIN`).
fn choose_tier2_bias(samples: &[SamplePixel], yellow_edge: f32, distance: f32) -> f32 {
    if samples.is_empty() {
        return SPEC_BIAS;
    }
    // Continuous ramp instead of a hard band: a hard floor stacked a +3..5%
    // rate step on top of the FLAT/SAT table switch at d=0.3 (measured on
    // david-underland: 0.29→0.30 was +9% bytes combined), and a hard cap
    // would put the mirror step at 2.5. Full strength covers d=0.8..2.0,
    // where the tail rescue was measured.
    let fade_in = ramp(distance, TIER2_MIN_DISTANCE, TIER2_FULL_DISTANCE);
    let fade_out = 1.0 - ramp(distance, TIER2_FADE_OUT_START, TIER2_MAX_DISTANCE);
    let strength = fade_in * fade_out;
    if strength <= 0.0 {
        return SPEC_BIAS;
    }
    let strong = samples.iter().filter(|s| s.risk > STRONG_SCORE).count();
    let strong_frac = strong as f32 / samples.len() as f32;
    if strong_frac < TIER2_AREA_MIN {
        return SPEC_BIAS;
    }

    let edge_strength = ramp(yellow_edge, TIER2_EDGE_START, TIER2_EDGE_FULL);
    if edge_strength <= 0.0 {
        return SPEC_BIAS;
    }
    let mean_risk = samples.iter().map(|sample| sample.risk).sum::<f32>() / samples.len() as f32;
    let pathological_strength = ramp(
        yellow_edge,
        TIER2_PATHOLOGICAL_EDGE_START,
        TIER2_PATHOLOGICAL_EDGE_FULL,
    ) * ramp(
        mean_risk,
        TIER2_PATHOLOGICAL_RISK_START,
        TIER2_PATHOLOGICAL_RISK_FULL,
    );
    let target_bias = SPEC_BIAS
        + (TIER2_BIAS - SPEC_BIAS) * edge_strength
        + (TIER2_PATHOLOGICAL_BIAS - TIER2_BIAS) * pathological_strength;
    // A tiny bias change can cross many B-residual quantization thresholds at
    // once. Snap the content target to the measured 0.025 grid so images near
    // a selector boundary get one of the validated operating points; the
    // distance strength below remains continuous and prevents RD steps.
    let target_bias =
        ((target_bias * TIER2_BIAS_GRID_SCALE).round() / TIER2_BIAS_GRID_SCALE).max(SPEC_BIAS);
    SPEC_BIAS + (target_bias - SPEC_BIAS) * strength
}

/// Pick the B bias for this image at this distance. Returns the spec value
/// unless the biased matrix demonstrably reduces yellow-chroma damage more
/// than it costs on ordinary content.
fn choose_b_bias_and_tier(linear: &Image3F, distance: f32) -> (f32, bool) {
    // Each tier enforces its own distance band; the shared upper bound here
    // only skips work past both.
    if distance >= MAX_DISTANCE.max(TIER2_MAX_DISTANCE) || !has_yellow_risk(linear) {
        return (SPEC_BIAS, false);
    }

    let (samples, yellow_edge) = collect_samples(linear);
    // Only thin/high-frequency yellow structure profits from the strong bias;
    // smooth-yellow content goes to the mild tier-2 path instead (a paid
    // perceptual trade: rate for visible chroma survival that SS2 barely
    // credits — validated per-pixel on assets/yellow).
    if yellow_edge < YELLOW_EDGE_MIN {
        return (choose_tier2_bias(&samples, yellow_edge, distance), false);
    }
    // Tier-1 (thin-HF yellow, strong bias) has its own, tighter cap.
    if distance >= MAX_DISTANCE {
        return (SPEC_BIAS, false);
    }
    let steps = proxy_steps(distance);
    let candidates = [SPEC_BIAS, BIAS_MID, BIAS_HI];

    let mut best_bias = SPEC_BIAS;
    let mut best_cost = f32::INFINITY;
    let mut spec_cost = 0.0f32;
    let mut spec_general = 0.0f32;

    for (i, &bias) in candidates.iter().enumerate() {
        let m = matrix_for_bias(bias);
        let score = evaluate_bias(&samples, &m, steps);
        if i == 0 {
            spec_general = score.general_error;
        }
        let regression = ((score.general_error - spec_general) / spec_general.max(1e-9)).max(0.0);
        let cost = (1.0 - TAIL_WEIGHT) * score.yellow_mean
            + TAIL_WEIGHT * score.yellow_tail
            + REGRESSION_WEIGHT * regression;
        if i == 0 {
            spec_cost = cost;
        }
        if cost < best_cost {
            best_cost = cost;
            best_bias = bias;
        }
    }
    if spec_cost < MIN_SPEC_COST || best_cost > REL_COST_RATIO * spec_cost {
        return (SPEC_BIAS, false);
    }
    let strength = 1.0 - ramp(distance, STRONG_FADE_START, MAX_DISTANCE);
    (SPEC_BIAS + (best_bias - SPEC_BIAS) * strength, true)
}

#[cfg(test)]
fn choose_b_bias(linear: &Image3F, distance: f32) -> f32 {
    choose_b_bias_and_tier(linear, distance).0
}

pub(crate) struct YellowSelection {
    /// `None` = keep the spec matrix (nothing signaled), `Some` = use and
    /// signal the biased matrix.
    pub(crate) matrix: Option<XybMatrix>,
    /// Staged frame B precision for tier-1. The JXL header stores an integer,
    /// so 5→4→3→2 avoids the former all-at-once 5→2 cliff.
    pub(crate) b_qm_scale: u32,
}

#[inline]
fn selected_b_qm_scale(custom: bool, strong: bool, distance: f32) -> u32 {
    if !custom || !strong {
        2
    } else if distance < 1.30 {
        5
    } else if distance < 1.40 {
        4
    } else if distance < 1.50 {
        3
    } else {
        2
    }
}

/// Entry point used by the encoder: one detector + proxy pass yielding both
/// the opsin decision and the band-class signal.
pub(crate) fn select_yellow(linear: &Image3F, distance: f32) -> YellowSelection {
    let (bias, tier1) = choose_b_bias_and_tier(linear, distance);
    let custom = (bias - SPEC_BIAS).abs() >= 1e-6;
    // A custom row can come from either tier. Only thin/high-frequency tier-1
    // content validated the fine B multiplier; smooth tier-2 yellow uses the
    // matrix alone. Treating every custom row as tier-1 spent 6–17% on several
    // yellow photos without improving their chroma error.
    let strong = custom && tier1;
    let b_qm_scale = selected_b_qm_scale(custom, strong, distance);
    YellowSelection {
        b_qm_scale,
        matrix: custom.then(|| matrix_for_bias(bias)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
        let mut out = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0f64;
                for k in 0..3 {
                    s += a[i * 3 + k] as f64 * b[k * 3 + j] as f64;
                }
                out[i * 3 + j] = s as f32;
            }
        }
        out
    }

    #[test]
    fn spec_bias_returns_spec_matrix() {
        let m = matrix_for_bias(SPEC_BIAS);
        assert_eq!(m, XybMatrix::SPEC);
        assert!(m.is_decoder_default());
    }

    #[test]
    fn derived_inverse_matches_forward() {
        for bias in [0.60, 0.70, 0.85] {
            let m = matrix_for_bias(bias);
            let prod = matmul3(&m.fwd, &m.inv);
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (prod[i * 3 + j] - expected).abs() < 1e-4,
                        "bias {bias}: fwd·inv[{i}][{j}] = {}",
                        prod[i * 3 + j]
                    );
                }
            }
            assert!(!m.is_decoder_default());
            // Rows still sum to 1 so grays map identically.
            let row_sum = m.fwd[6] + m.fwd[7] + m.fwd[8];
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn bias_085_matches_validated_red_inverse() {
        // The 0.85 inverse was validated end-to-end with djxl during the
        // red-opsin experiment; the derived inverse must reproduce it.
        let expected: [f32; 9] = [
            10.785_613,
            -9.684_577,
            -0.101_036_27,
            -3.500_100_9,
            4.601_137,
            -0.101_036_27,
            -0.751_554_5,
            0.557_254,
            1.194_300_5,
        ];
        let m = matrix_for_bias(0.85);
        for (a, b) in m.inv.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 2e-4 * b.abs().max(1.0), "{a} vs {b}");
        }
    }

    #[test]
    fn spec_numeric_inverse_matches_decoder_default() {
        let inv = invert3x3(&XybMatrix::SPEC.fwd).unwrap();
        for (a, b) in inv.iter().zip(XybMatrix::SPEC.inv.iter()) {
            assert!((a - b).abs() < 2e-4 * b.abs().max(1.0), "{a} vs {b}");
        }
    }

    #[test]
    fn xyb_round_trip_through_derived_matrices() {
        for bias in [SPEC_BIAS, 0.70, 0.85] {
            let m = matrix_for_bias(bias);
            for rgb in [
                [0.9, 0.85, 0.1],
                [0.5, 0.5, 0.5],
                [0.1, 0.2, 0.8],
                [1.0, 0.0, 0.0],
                [0.02, 0.9, 0.3],
            ] {
                let (x, y, b) = rgb_to_xyb_pixel_f32(&m, rgb[0], rgb[1], rgb[2]);
                let rec = xyb_to_rgb_pixel_f32(&m, x, y, b);
                for c in 0..3 {
                    assert!(
                        (rec[c] - rgb[c]).abs() < 1e-4,
                        "bias {bias} rgb {rgb:?} → {rec:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn detector_scores_bright_yellow_only() {
        // Bright saturated yellow: strong.
        assert!(yellow_pixel_risk(0.9, 0.85, 0.05) > 0.5);
        // Bright pure red / green / blue / white / gray: zero-ish.
        assert!(yellow_pixel_risk(0.9, 0.05, 0.05) < 1e-3);
        assert!(yellow_pixel_risk(0.05, 0.9, 0.05) < 1e-3);
        assert!(yellow_pixel_risk(0.05, 0.05, 0.9) == 0.0);
        assert!(yellow_pixel_risk(0.9, 0.9, 0.9) == 0.0);
        assert!(yellow_pixel_risk(0.3, 0.3, 0.05) == 0.0); // too dark
    }

    fn filled(w: usize, h: usize, rgb: [f32; 3]) -> Image3F {
        let mut img = Image3F::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for v in img.plane_row_mut(c, y).iter_mut() {
                    *v = rgb[c];
                }
            }
        }
        img
    }

    #[test]
    fn gate_fires_on_yellow_image_only() {
        assert!(has_yellow_risk(&filled(64, 64, [0.9, 0.85, 0.05])));
        assert!(!has_yellow_risk(&filled(64, 64, [0.5, 0.5, 0.5])));
        assert!(!has_yellow_risk(&filled(64, 64, [0.9, 0.1, 0.05])));
    }

    #[test]
    fn choose_bias_spec_below_min_distance() {
        let img = filled(64, 64, [0.9, 0.85, 0.05]);
        assert_eq!(choose_b_bias(&img, 0.1), SPEC_BIAS);
    }

    #[test]
    fn choose_bias_prefers_spec_on_neutral_content() {
        let img = filled(64, 64, [0.4, 0.5, 0.6]);
        assert_eq!(choose_b_bias(&img, 2.0), SPEC_BIAS);
    }

    #[test]
    fn fine_b_is_reserved_for_the_high_frequency_tier() {
        let smooth = filled(64, 64, [0.9, 0.85, 0.05]);
        let smooth_selection = select_yellow(&smooth, 1.0);
        assert!(smooth_selection.matrix.is_none());
        assert_eq!(smooth_selection.b_qm_scale, 2);

        let mut tier2 = filled(64, 64, [0.9, 0.85, 0.05]);
        for y in (0..64).step_by(SAMPLE_STRIDE) {
            for x in (0..64).step_by(SAMPLE_STRIDE) {
                if x + 1 < 64 {
                    for c in 0..3 {
                        tier2.plane_row_mut(c, y)[x + 1] = 0.6;
                    }
                }
            }
        }
        assert!(collect_samples(&tier2).1 >= TIER2_EDGE_FULL);
        assert!(collect_samples(&tier2).1 < YELLOW_EDGE_MIN);
        let tier2_selection = select_yellow(&tier2, 1.0);
        assert!(tier2_selection.matrix.is_some());
        assert_eq!(tier2_selection.b_qm_scale, 2);

        let mut edged = filled(64, 64, [0.9, 0.85, 0.05]);
        for y in (0..64).step_by(SAMPLE_STRIDE) {
            for x in (0..64).step_by(SAMPLE_STRIDE) {
                if x + 1 < 64 {
                    for c in 0..3 {
                        edged.plane_row_mut(c, y)[x + 1] = 0.0;
                    }
                }
            }
        }
        assert!(collect_samples(&edged).1 >= YELLOW_EDGE_MIN);
        assert_eq!(selected_b_qm_scale(true, true, 1.0), 5);
        assert_eq!(selected_b_qm_scale(true, false, 1.0), 2);
    }
}
