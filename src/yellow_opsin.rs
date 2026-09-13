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

/// Ordinary content releases the strong matrix in this band. Only red-green
/// chromatic structure retains it, together with the coarse mosaic search.
const STRONG_FADE_START: f32 = 2.5;
const MAX_DISTANCE: f32 = 3.0;
/// Same opponent-gradient ratio as the frame's X-heavy classifier. This
/// pre-conversion estimate uses immediate neighbors on a 1/64 pixel grid.
const COARSE_X_GRADIENT_MIN: f32 = 0.04;
const STRONG_REFERENCE_DISTANCE: f32 = 1.25;
const STRONG_COARSE_DISTANCE: f32 = 2.0;
const STRONG_COARSE_BIAS: f32 = 0.90;

/// Ordinary tier-2 coverage for the matrix plus reconstruction-weight boost.
/// Smaller bright-yellow detections need the texture fallback below. The
/// edge statistic scales the matrix strength for both paths.
const TIER2_AREA_MIN: f32 = 0.10;
// Textured yellow can occupy a much larger visible area than this bright-only
// detector counts. Below the ordinary area gate, require opponent structure
// and use the matrix without a frame-wide reconstruction-weight boost.
const TIER2_TEXTURE_AREA_MIN: f32 = 0.01;
const TIER2_TEXTURE_B_GRAD_MIN: f32 = 0.18;
// B-heavy content has its own quantization path; the mild fallback regressed it.
const TIER2_TEXTURE_B_GRAD_MAX: f32 = 0.45;

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

fn coarse_distance_strength(distance: f32) -> f32 {
    ramp(distance, STRONG_REFERENCE_DISTANCE, STRONG_COARSE_DISTANCE)
}

/// The matrix and the coarse fine-transform search need to serve the same
/// content class. Keeping the matrix alone costs rate on blue-axis structure
/// and ordinary photos. X/Y are independent of the candidate B row, so this
/// source estimate does not depend on which matrix the proxy selects.
fn has_coarse_x_structure(linear: &Image3F) -> bool {
    let (w, h) = (linear.xsize(), linear.ysize());
    let [rp, gp, bp] = std::array::from_fn(|c| linear.plane_data(c));
    let xy = |i| {
        let (x, y, _) = rgb_to_xyb_pixel_f32(&XybMatrix::SPEC, rp[i], gp[i], bp[i]);
        (x, y)
    };
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for y in (0..h).step_by(SAMPLE_STRIDE) {
        for x in (0..w).step_by(SAMPLE_STRIDE) {
            let i = y * w + x;
            let (cx, cy) = xy(i);
            for neighbor in [(x + 1 < w).then_some(i + 1), (y + 1 < h).then_some(i + w)]
                .into_iter()
                .flatten()
            {
                let (nx, ny) = xy(neighbor);
                sum_x += (nx - cx).abs();
                sum_y += (ny - cy).abs();
            }
        }
    }
    sum_y > f32::EPSILON && sum_x >= COARSE_X_GRADIENT_MIN * sum_y
}

/// Source-only estimate of blue-yellow structure, independent of the selected
/// B row. Both directions are sampled on the existing 1/64 pixel grid.
fn sampled_b_gradient_ratio(linear: &Image3F) -> f32 {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { crate::avx::sampled_b_gradient_ratio_avx2(linear, SAMPLE_STRIDE) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        unsafe { crate::neon::sampled_b_gradient_ratio_neon(linear, SAMPLE_STRIDE) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    sampled_b_gradient_ratio_scalar(linear)
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
fn sampled_b_gradient_ratio_scalar(linear: &Image3F) -> f32 {
    let (w, h) = (linear.xsize(), linear.ysize());
    let [rp, gp, bp] = std::array::from_fn(|c| linear.plane_data(c));
    let yb = |i| {
        let (_, y, b) = rgb_to_xyb_pixel_f32(&XybMatrix::SPEC, rp[i], gp[i], bp[i]);
        (y, b - y)
    };
    let (mut sum_y, mut sum_b) = (0.0, 0.0);
    for y in (0..h).step_by(SAMPLE_STRIDE) {
        for x in (0..w).step_by(SAMPLE_STRIDE) {
            let i = y * w + x;
            let (cy, cb) = yb(i);
            for neighbor in [(x + 1 < w).then_some(i + 1), (y + 1 < h).then_some(i + w)]
                .into_iter()
                .flatten()
            {
                let (ny, nb) = yb(neighbor);
                sum_y += (ny - cy).abs();
                sum_b += (nb - cb).abs();
            }
        }
    }
    if sum_y > f32::EPSILON {
        sum_b / sum_y
    } else {
        0.0
    }
}

/// Release the fine-quality error boost before the matrix reaches its coarse
/// target. Keeping that boost until d=2 overspends in the d=1.5 transition.
pub(crate) fn coarse_weight_strength(distance: f32) -> f32 {
    ramp(distance, STRONG_REFERENCE_DISTANCE, 1.5)
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
/// derived from the calibrated DCT8 dequant tables (mean AC step over the low/mid
/// band where saturated texture lives) at the effective AC quantizer
/// `K_AC_QUANT / distance`. Order: [X, Y, B].
fn proxy_steps(distance: f32) -> [f32; 3] {
    // Matches frame::compute_distance_params: the quant-field rebalancing
    // cancels, leaving q_eff ≈ K_AC_QUANT / distance.
    const K_AC_QUANT: f32 = 0.8;
    const RECIP_K_AC_QUANT: f32 = 1.0 / K_AC_QUANT;
    let mut steps = [0.0f32; 3];
    for (c, step) in steps.iter_mut().enumerate() {
        let table = DequantMatrices::color_proxy_matrix(distance, c);
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

#[derive(Clone, Copy, Default)]
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
    if w == 0 || h == 0 {
        return (Vec::new(), 0.0);
    }

    let rp = linear.plane_data(0);
    let gp = linear.plane_data(1);
    let bp = linear.plane_data(2);
    let sample_count = h.div_ceil(SAMPLE_STRIDE) * w.div_ceil(SAMPLE_STRIDE);
    let mut samples = vec![SamplePixel::default(); sample_count];
    let mut samples_iter = samples.iter_mut();
    let mut edge_sum = 0.0f32;
    let mut edge_n = 0usize;
    let luma = |r: f32, g: f32, b: f32| 0.3 * r + 0.6 * g + 0.1 * b;
    for (sample_y, r_row) in rp.chunks_exact(w).step_by(SAMPLE_STRIDE).enumerate() {
        let y = sample_y * SAMPLE_STRIDE;
        let row = y * w;
        let g_row = &gp[row..row + w];
        let b_row = &bp[row..row + w];
        for ((sample_x, &r), sample) in r_row
            .iter()
            .step_by(SAMPLE_STRIDE)
            .enumerate()
            .zip(samples_iter.by_ref())
        {
            let x = sample_x * SAMPLE_STRIDE;
            let i = row + x;
            let rgb = [r, g_row[x], b_row[x]];
            let risk = yellow_pixel_risk(rgb[0], rgb[1], rgb[2]);
            if risk > STRONG_SCORE {
                let center_luma = luma(rgb[0], rgb[1], rgb[2]);
                if x + 1 < w {
                    edge_sum +=
                        (luma(r_row[x + 1], g_row[x + 1], b_row[x + 1]) - center_luma).abs();
                    edge_n += 1;
                }
                if y + 1 < h {
                    edge_sum += (luma(rp[i + w], gp[i + w], bp[i + w]) - center_luma).abs();
                    edge_n += 1;
                }
            }
            *sample = SamplePixel { rgb, risk };
        }
    }
    let yellow_edge = if edge_n > 0 {
        edge_sum / edge_n as f32
    } else {
        0.0
    };
    (samples, yellow_edge)
}

/// Tier-2: mild yellow matrix. Low-area detections additionally need the
/// source-structure check in `choose_b_bias_and_tier`.
fn choose_tier2_bias(
    samples: &[SamplePixel],
    yellow_edge: f32,
    distance: f32,
    strong_frac: f32,
) -> f32 {
    if samples.is_empty() {
        return SPEC_BIAS;
    }
    let fade_in = ramp(distance, TIER2_MIN_DISTANCE, TIER2_FULL_DISTANCE);
    let fade_out = 1.0 - ramp(distance, TIER2_FADE_OUT_START, TIER2_MAX_DISTANCE);
    let strength = fade_in * fade_out;
    if strength <= 0.0 {
        return SPEC_BIAS;
    }
    if strong_frac < TIER2_TEXTURE_AREA_MIN {
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
/// than it costs on ordinary content. The last flag keeps default reconstruction
/// weights for the low-area, textured-yellow fallback.
fn choose_b_bias_and_tier(linear: &Image3F, distance: f32) -> (f32, bool, bool) {
    if !has_yellow_risk(linear) {
        return (SPEC_BIAS, false, false);
    }

    let (samples, yellow_edge) = collect_samples(linear);
    let strong = samples.iter().filter(|s| s.risk > STRONG_SCORE).count();
    let strong_frac = strong as f32 / samples.len().max(1) as f32;
    let mut tier2_bias = choose_tier2_bias(&samples, yellow_edge, distance, strong_frac);
    let default_weights = strong_frac < TIER2_AREA_MIN && tier2_bias > SPEC_BIAS;
    if default_weights
        && !(TIER2_TEXTURE_B_GRAD_MIN..TIER2_TEXTURE_B_GRAD_MAX)
            .contains(&sampled_b_gradient_ratio(linear))
    {
        tier2_bias = SPEC_BIAS;
    }
    let tier2 = (tier2_bias, false, default_weights && tier2_bias > SPEC_BIAS);
    // Only thin/high-frequency yellow structure profits from the strong bias;
    // smooth-yellow content goes to the mild tier-2 path instead (a paid
    // perceptual trade: rate for visible chroma survival that SS2 barely
    // credits — validated per-pixel on assets/yellow).
    if yellow_edge < YELLOW_EDGE_MIN {
        return tier2;
    }
    let strength = if distance > STRONG_FADE_START && !has_coarse_x_structure(linear) {
        1.0 - ramp(distance, STRONG_FADE_START, MAX_DISTANCE)
    } else {
        1.0
    };
    if strength == 0.0 {
        return tier2;
    }
    // Past the validated fine-quality band, keep the proxy's selection
    // stable. Its per-pixel quantization otherwise alternates between the
    // mild and strong rows as distance crosses individual sample thresholds;
    // that is not a measurement of the high-frequency AC loss we are fixing.
    let steps = proxy_steps(distance.min(STRONG_REFERENCE_DISTANCE));
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
        return tier2;
    }
    if best_bias == BIAS_HI {
        best_bias += (STRONG_COARSE_BIAS - BIAS_HI) * coarse_distance_strength(distance);
    }
    (
        tier2_bias + (best_bias - tier2_bias) * strength,
        true,
        false,
    )
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
    /// Sparse bright-yellow detections receive the matrix alone, without
    /// increasing the reconstruction weight across the rest of the frame.
    pub(crate) default_channel_weights: bool,
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
    let (bias, tier1, default_channel_weights) = choose_b_bias_and_tier(linear, distance);
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
        default_channel_weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ))]
    #[test]
    fn sampled_b_simd_matches_scalar_at_sample_boundaries() {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let mut state = 0x716e_23b9u32;
        // Include every tail length, absent neighbors, and non-sRGB values
        // from color conversion that exercise the nonnegative opsin clamp.
        for h in [1, 2, 7, 8, 9, 15, 16, 17, 65] {
            for w in 1..=35 {
                for (offset, scale) in [(0.0, 1.0), (-0.5, 8.0)] {
                    let mut image = Image3F::new(w, h);
                    for c in 0..3 {
                        for y in 0..h {
                            for v in image.plane_row_mut(c, y) {
                                state ^= state << 13;
                                state ^= state >> 17;
                                state ^= state << 5;
                                *v = offset + scale * ((state >> 8) as f32 / 16777216.0);
                            }
                        }
                    }
                    let actual = sampled_b_gradient_ratio(&image);
                    let expected = sampled_b_gradient_ratio_scalar(&image);
                    if cfg!(all(target_arch = "x86_64", not(target_feature = "fma"))) {
                        // AVX always fuses; generic x86 scalar builds do not.
                        assert!(
                            (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                            "{w}x{h}: {actual} vs {expected}"
                        );
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{w}x{h}, offset={offset}, scale={scale}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sampled_b_structure_tracks_both_directions_and_rejects_flat_fields() {
        assert_eq!(sampled_b_gradient_ratio(&filled(16, 16, [0.2; 3])), 0.0);
        for vertical in [false, true] {
            for ratio in [0.0, 0.25, 0.65] {
                let mut image = filled(16, 16, [0.0; 3]);
                for y in 0..16 {
                    for x in 0..16 {
                        let step = ((if vertical { y } else { x }) % 2) as f32 * 0.02;
                        let rgb = xyb_to_rgb_pixel_f32(
                            &XybMatrix::SPEC,
                            0.0,
                            0.3 + step,
                            0.3 + step + ratio * step,
                        );
                        for (c, value) in rgb.into_iter().enumerate() {
                            image.plane_row_mut(c, y)[x] = value;
                        }
                    }
                }
                assert!((sampled_b_gradient_ratio(&image) - ratio).abs() < 1e-4);
            }
        }
    }

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
        for bias in [0.60, 0.70, 0.85, STRONG_COARSE_BIAS] {
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
        for bias in [SPEC_BIAS, 0.70, 0.85, STRONG_COARSE_BIAS] {
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

    #[test]
    fn coarse_structure_gate_distinguishes_opponent_axes_in_both_directions() {
        for vertical in [false, true] {
            for (other, expected) in [([1.0, 0.0, 0.0], true), ([0.0, 0.0, 1.0], false)] {
                let mut edged = filled(32, 32, [1.0, 1.0, 0.0]);
                for y in 0..32 {
                    for x in 0..32 {
                        if (if vertical { y } else { x }) % 2 == 1 {
                            for (c, value) in other.into_iter().enumerate() {
                                edged.plane_row_mut(c, y)[x] = value;
                            }
                        }
                    }
                }
                assert_eq!(has_coarse_x_structure(&edged), expected);
                if !expected {
                    assert_eq!(choose_b_bias(&edged, 6.0), SPEC_BIAS);
                }
            }
        }
        assert!(!has_coarse_x_structure(&filled(1, 1, [1.0, 1.0, 0.0])));
    }

    #[test]
    fn strong_tier_keeps_protection_at_coarse_distances() {
        // Sparse yellow edges pass the strong proxy; nearby red/green edges
        // also pass the coarse structure gate. Both protections must survive
        // the old d3 and d5 search boundaries.
        let mut edged = filled(64, 64, [0.0, 0.0, 0.0]);
        for y in (0..64).step_by(SAMPLE_STRIDE) {
            for x in (0..64).step_by(SAMPLE_STRIDE) {
                let rgb = if (x / SAMPLE_STRIDE + y / SAMPLE_STRIDE) % 4 == 0 {
                    [0.9, 0.85, 0.05]
                } else {
                    [0.5, 0.0, 0.0]
                };
                for c in 0..3 {
                    edged.plane_row_mut(c, y)[x] = rgb[c];
                }
                if rgb[1] == 0.0 {
                    let green = [0.0, 0.25, 0.0];
                    for c in 0..3 {
                        edged.plane_row_mut(c, y)[x + 1] = green[c];
                        edged.plane_row_mut(c, y + 1)[x] = green[c];
                    }
                }
            }
        }
        assert!(collect_samples(&edged).1 >= YELLOW_EDGE_MIN);
        assert!(has_coarse_x_structure(&edged));
        let bias = choose_b_bias(&edged, 2.0);
        assert!(bias > SPEC_BIAS);
        for distance in [2.5, 2.999, 3.0, 3.001, 4.0, 5.0, 6.0, 25.0] {
            assert_eq!(choose_b_bias(&edged, distance), bias);
            assert_eq!(select_yellow(&edged, distance).b_qm_scale, 2);
        }
    }
}
