/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! Predictor-aware chroma DC rounding on the existing quantization lattice.

#![forbid(unsafe_code)]

use super::{
    DC_PREDICTOR_WEIGHTED, DcPlanePrice, DcPredictorChoice, DistanceParams, K_GRAD_RANGE_MAX,
    K_GRAD_RANGE_MID, K_GRAD_RANGE_MIN, choose_dc_predictors_from_counts,
    collect_dc_tokens_with_gradient,
};
use crate::coder_scratch::DcPredictorScratch;
use crate::color_correlation::{CflFrame, dc_cfl_factor, dc_cfl_factor_x};
use crate::dc_group_data::DcGroupData;
use crate::dc_ops::BdcCompensation;
use crate::dct::fmla;
use crate::entropy::{ALPHABET_SIZE, Token, f_log2, pack_signed, uint_encode};
use crate::quant_weights::INV_DC_QUANT;
use crate::static_entropy_codes::{K_GRADIENT_CONTEXT_LUT, K_NUM_DC_CONTEXTS};
use crate::util::{EncodeError, try_vec};
use crate::weighted_predictor::{WpParams, WpPrediction, WpState, WpUpdate};

/// Rate penalty in squared DC-step units per bit, ramped linearly between
/// the two distances.
static DC_RDO_LAMBDA: [f32; 2] = [0.04, 0.15];
static DC_RDO_LAMBDA_DISTANCE: [f32; 2] = [1.0, 2.0];
/// X rounds toward its prediction more readily than B.
const DC_RDO_X_LAMBDA_SCALE: f32 = 1.5;
/// Bound the error of a newly chosen level, retaining larger incumbent errors.
const DC_RDO_MAX_ERROR: f32 = 1.0;
/// Require enough estimated payload gain to absorb small coding-model changes.
const DC_RDO_MIN_GAIN_BITS: f64 = 8.0;

struct Prices {
    bits: Vec<[f32; ALPHABET_SIZE]>,
}

impl Prices {
    fn new(
        groups: &[DcGroupData],
        scratch: &mut DcPredictorScratch,
    ) -> Result<(Self, DcPredictorChoice, [DcPlanePrice; 2]), EncodeError> {
        // Keep per-channel counts for both predictors while choosing the
        // frame's predictor from their pooled counts. Token streams can then
        // be dropped after each group instead of retaining the entire frame.
        let mut counts = try_vec![[0u32; ALPHABET_SIZE]; 6 * K_NUM_DC_CONTEXTS]?;
        scratch.counts.as_flattened_mut().fill(0);
        scratch.extra.fill(0);
        let mut raw_bits = [0u64; 3];
        for dc in groups {
            let (wp, grad) = collect_dc_tokens_with_gradient(
                dc,
                &DC_PREDICTOR_WEIGHTED,
                &mut Vec::new(),
                false,
                true,
            );
            let mut offset = 0;
            for (mchan, c) in [1, 0, 2].into_iter().enumerate() {
                let end = offset + dc.quant_dc.plane(c).as_slice().len();
                for (candidate, tokens) in [&wp[offset..end], &grad[offset..end]]
                    .into_iter()
                    .enumerate()
                {
                    for token in tokens {
                        let context = token.context as usize;
                        let (symbol, extra, _) = uint_encode(token.value);
                        let slot = candidate * K_NUM_DC_CONTEXTS + context;
                        scratch.counts[slot][symbol as usize] += 1;
                        scratch.extra[slot] += u64::from(extra);
                        if candidate == 0 {
                            raw_bits[mchan] += u64::from(extra);
                        }
                        counts[(candidate * 3 + mchan) * K_NUM_DC_CONTEXTS + context]
                            [symbol as usize] += 1;
                    }
                }
                offset = end;
            }
        }
        let predictors = choose_dc_predictors_from_counts(scratch);
        let mut bits = try_vec![[0.0; ALPHABET_SIZE]; 3 * K_NUM_DC_CONTEXTS]?;
        for (slot, bits) in bits.iter_mut().enumerate() {
            let context = slot % K_NUM_DC_CONTEXTS;
            let candidate = usize::from(predictors[context]);
            let counts = &counts[candidate * 3 * K_NUM_DC_CONTEXTS + slot];
            // A half-count keeps newly visited contexts and symbols finite.
            let total =
                counts.iter().map(|&n| f64::from(n)).sum::<f64>() + 0.5 * ALPHABET_SIZE as f64;
            for (&count, bits) in counts.iter().zip(bits) {
                *bits = f_log2(total / (f64::from(count) + 0.5)) as f32;
            }
        }
        // The pooled baseline prices use the same weighted tokens. Retain
        // these small histograms instead of walking both chroma planes again.
        let base_prices = std::array::from_fn(|i| {
            let mchan = i + 1;
            DcPlanePrice {
                tokens: Vec::new(),
                counts: counts[mchan * K_NUM_DC_CONTEXTS..(mchan + 1) * K_NUM_DC_CONTEXTS].to_vec(),
                raw_bits: raw_bits[mchan],
            }
        });
        Ok((Self { bits }, predictors, base_prices))
    }

    #[cfg(debug_assertions)]
    #[inline]
    fn residual_bits(&self, mchan: usize, context: usize, residual: i32) -> f32 {
        let (symbol, extra, _) = uint_encode(pack_signed(residual));
        self.bits[mchan * K_NUM_DC_CONTEXTS + context][symbol as usize] + extra as f32
    }
}

/// Prices are frozen before changing the planes. Final tree learning and
/// entropy coding still run on the chosen levels, including any predictor
/// flips. The frozen static contexts are only a quantization rate estimate.
pub(super) fn refine_dc_rounding(
    groups: &mut [DcGroupData],
    distp: &DistanceParams,
    ytob_dc: i32,
    cfl: CflFrame,
    compensate_b: bool,
    scratch: &mut DcPredictorScratch,
) -> Result<(), EncodeError> {
    if groups
        .iter()
        .any(|dc| dc.source_dc_y.is_none() || dc.source_dc_x.is_none() || dc.source_dc_b.is_none())
    {
        return Ok(());
    }
    let (prices, predictors, base_prices) = Prices::new(groups, scratch)?;
    let inv: [f32; 3] =
        std::array::from_fn(|c| INV_DC_QUANT[c] * distp.scale_dc / distp.dc_step[c]);
    let factors = [
        dc_cfl_factor_x(distp.dc_step, cfl.base_x),
        0.0,
        dc_cfl_factor(distp.dc_step, ytob_dc, cfl),
    ];
    let compensation = BdcCompensation::new(distp);
    let [d0, d1] = DC_RDO_LAMBDA_DISTANCE;
    let t = ((distp.distance - d0) / (d1 - d0)).clamp(0.0, 1.0);
    let lambda = DC_RDO_LAMBDA[0] + (DC_RDO_LAMBDA[1] - DC_RDO_LAMBDA[0]) * t;
    // Preserve luma: changing it also changes both chroma reconstruction
    // targets. Chroma channels can be raced independently under the final Y.
    let mut rounding = RoundingScratch::new();
    let group_sizes: Vec<usize> = groups
        .iter()
        .map(|dc| dc.quant_dc.plane(0).as_slice().len())
        .collect();
    let mut previous = try_vec![0i16; group_sizes.iter().sum()]?;
    let mut targets = try_vec![0.0f32; group_sizes.iter().copied().max().unwrap_or(0)]?;
    let mut candidate_price = DcPlanePrice::default();
    for ((mchan, c, lambda), base_price) in [(1, 0, lambda * DC_RDO_X_LAMBDA_SCALE), (2, 2, lambda)]
        .into_iter()
        .zip(base_prices)
    {
        let mut offset = 0;
        for (dc, &len) in groups.iter().zip(&group_sizes) {
            previous[offset..offset + len].copy_from_slice(dc.quant_dc.plane(c).as_slice());
            offset += len;
        }
        candidate_price.clear();
        let (mut base_error, mut candidate_error) = (0.0f64, 0.0f64);
        let mut offset = 0;
        for (dc, &len) in groups.iter_mut().zip(&group_sizes) {
            let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
            let previous = &previous[offset..offset + len];
            let targets = &mut targets[..len];
            offset += len;
            let source = match c {
                0 => dc.source_dc_x.as_ref().unwrap(),
                1 => dc.source_dc_y.as_ref().unwrap(),
                _ => dc.source_dc_b.as_ref().unwrap(),
            };
            for (((target, &source), &yq), &sy) in targets
                .iter_mut()
                .zip(source.as_slice())
                .zip(dc.quant_dc.plane(1).as_slice())
                .zip(dc.source_dc_y.as_ref().unwrap().as_slice())
            {
                *target = fmla(source, inv[c], -f32::from(yq) * factors[c]);
                if c == 2 && compensate_b {
                    *target = compensation.adjust(*target, sy, yq);
                }
            }
            base_error += targets
                .iter()
                .zip(previous)
                .map(|(&t, &q)| f64::from(t - f32::from(q)).powi(2))
                .sum::<f64>();
            refine_channel_with_scratch(
                dc.quant_dc.plane_mut(c).as_mut_slice(),
                targets,
                w,
                h,
                mchan,
                &predictors,
                &prices,
                lambda,
                &mut rounding,
                |_, weighted| candidate_price.add_token(weighted),
            );
            let levels = dc.quant_dc.plane(c).as_slice();
            candidate_error += targets
                .iter()
                .zip(levels)
                .map(|(&t, &q)| f64::from(t - f32::from(q)).powi(2))
                .sum::<f64>();
        }
        // Local choices change later predictions. Reprice the complete plane
        // with pooled histograms before keeping any of its changes, including
        // its distortion under the same source targets and quantization step.
        let (base_bits, candidate_bits) = (base_price.bits(), candidate_price.bits());
        let lambda = f64::from(lambda);
        if candidate_bits + DC_RDO_MIN_GAIN_BITS >= base_bits
            || candidate_error + lambda * candidate_bits >= base_error + lambda * base_bits
        {
            let mut offset = 0;
            for (dc, &len) in groups.iter_mut().zip(&group_sizes) {
                dc.quant_dc
                    .plane_mut(c)
                    .as_mut_slice()
                    .copy_from_slice(&previous[offset..offset + len]);
                offset += len;
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn dc_rounding_bounds<const NATIVE_ROUND: bool>(target: f32) -> [i64; 2] {
    if NATIVE_ROUND {
        [target.floor() as i64, target.ceil() as i64]
    } else {
        dc_bounds_from_truncation(target)
    }
}

#[inline]
fn dc_bounds_from_truncation(target: f32) -> [i64; 2] {
    // Every candidate is clamped to i16 below. Clamp before converting so a
    // single i32 truncation suffices, including targets outside the DC range.
    // Baseline x86 can do this without floorf/ceilf or SSE4.1.
    let bounded = target.clamp(f32::from(i16::MIN), f32::from(i16::MAX));
    let truncated = bounded as i32;
    let back = truncated as f32;
    [
        i64::from(truncated - i32::from(bounded < back)),
        i64::from(truncated + i32::from(bounded > back)),
    ]
}

/// Full predictor state is needed only for the last two trellis columns.
#[derive(Clone, Copy)]
struct TrellisNode {
    level: i16,
    back: u16,
    cost: f32,
    update: WpUpdate,
}

/// Older columns only need the chosen level and predecessor for traceback.
#[derive(Clone, Copy)]
struct TrellisStep {
    level: i16,
    back: u16,
}

type TrellisPrediction = (i64, usize, WpPrediction, f32);

struct RoundingScratch {
    wp: WpState,
    base: Vec<[u32; 4]>,
    older: Vec<TrellisNode>,
    previous: Vec<TrellisNode>,
    column: Vec<TrellisNode>,
    history: Vec<TrellisStep>,
    starts: Vec<usize>,
    predictions: Vec<TrellisPrediction>,
}

impl RoundingScratch {
    fn new() -> Self {
        Self {
            wp: WpState::new(0),
            base: Vec::new(),
            older: Vec::new(),
            previous: Vec::new(),
            column: Vec::new(),
            history: Vec::new(),
            starts: Vec::new(),
            predictions: Vec::new(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn refine_channel(
    levels: &mut [i16],
    targets: &[f32],
    w: usize,
    h: usize,
    mchan: usize,
    predictors: &DcPredictorChoice,
    prices: &Prices,
    lambda: f32,
    mut emit: impl FnMut(Token),
) {
    refine_channel_with_scratch(
        levels,
        targets,
        w,
        h,
        mchan,
        predictors,
        prices,
        lambda,
        &mut RoundingScratch::new(),
        |selected, _| emit(selected),
    );
}

/// Row-wise Viterbi over each block's candidate levels: its level, the
/// target's floor and ceiling, its neighbors and the prediction, within
/// `DC_RDO_MAX_ERROR` of the target. A block's choice is the west neighbour
/// and weighted-predictor history of the next block's prediction, so every
/// survivor carries the predictor record its path wrote and each transition
/// is priced under its own path. Rows below see the finished row.
#[allow(clippy::too_many_arguments)]
fn refine_channel_with_scratch(
    levels: &mut [i16],
    targets: &[f32],
    w: usize,
    h: usize,
    mchan: usize,
    predictors: &DcPredictorChoice,
    prices: &Prices,
    lambda: f32,
    scratch: &mut RoundingScratch,
    mut emit: impl FnMut(Token, Token),
) {
    const NATIVE_ROUND: bool =
        !cfg!(any(target_arch = "x86", target_arch = "x86_64")) || cfg!(target_feature = "sse4.1");
    if w == 0 || h == 0 {
        return;
    }
    let RoundingScratch {
        wp,
        base,
        older,
        previous,
        column,
        history,
        starts,
        predictions,
    } = scratch;
    wp.reset(w, WpParams::DEFAULT);
    let channel_bits = &prices.bits[mchan * K_NUM_DC_CONTEXTS..(mchan + 1) * K_NUM_DC_CONTEXTS];
    for y in 0..h {
        wp.save_row_base(y, base);
        history.clear();
        starts.clear();
        older.clear();
        previous.clear();
        for x in 0..w {
            let i = y * w + x;
            let get = |x: usize, y: usize| i64::from(levels[y * w + x]);
            let target = targets[i];
            let current = i64::from(levels[i]);
            let max_error = (target - current as f32).abs().max(DC_RDO_MAX_ERROR);
            // A target off the level range keeps its level at no cost, so
            // its distortion cannot overflow and freeze the rest of the row.
            let forced = !target.is_finite() || max_error > 2.0 * f32::from(i16::MAX);
            starts.push(history.len());
            column.clear();
            let push = |nodes: &mut Vec<TrellisNode>, level: i64| {
                let level = level.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
                if (forced && !nodes.is_empty())
                    || (target - f32::from(level)).abs() > max_error
                    || nodes.iter().any(|n| n.level == level)
                {
                    return;
                }
                nodes.push(TrellisNode {
                    level,
                    cost: f32::INFINITY,
                    back: 0,
                    update: WpUpdate::default(),
                });
            };
            push(column, current);
            if target.is_finite() {
                let [floor, ceil] = dc_rounding_bounds::<NATIVE_ROUND>(target);
                for candidate in [floor, ceil, current - 1, current + 1] {
                    push(column, candidate);
                }
            }
            // Predict once per surviving path into this block. Keep the
            // original candidate and predecessor order, including exact ties.
            predictions.clear();
            for s in 0..previous.len().max(1) {
                let (west_value, path_cost) = if x > 0 {
                    let node = previous[s];
                    let west2 = (x > 1).then(|| older[usize::from(node.back)].update);
                    wp.stage_path(x, y, base, Some(node.update), west2);
                    (i64::from(node.level), node.cost)
                } else {
                    (0, 0.0)
                };
                let north = if y > 0 { get(x, y - 1) } else { west_value };
                let west = if x == 0 { north } else { west_value };
                let ne = if y > 0 {
                    get((x + 1).min(w - 1), y - 1)
                } else {
                    west
                };
                let nw = if y > 0 && x > 0 {
                    get(x - 1, y - 1)
                } else {
                    north
                };
                let nn = if y > 1 { get(x, y - 2) } else { north };
                let prediction = wp.predict(x, y, north, west, ne, nw, nn);
                let prop = (K_GRAD_RANGE_MID + wp.wp_prop).clamp(K_GRAD_RANGE_MIN, K_GRAD_RANGE_MAX)
                    as usize;
                let context = K_GRADIENT_CONTEXT_LUT[prop] as usize;
                let prediction = if predictors[context] {
                    (north + west - nw).clamp(north.min(west), north.max(west))
                } else {
                    prediction
                };
                if target.is_finite() {
                    push(column, prediction);
                }
                predictions.push((prediction, context, wp.last_prediction(), path_cost));
            }
            for node in column.iter_mut() {
                let error = if forced {
                    0.0
                } else {
                    target - f32::from(node.level)
                };
                let distortion = error * error;
                let mut winning_prediction = None;
                for (back, &(prediction, context, _, path_cost)) in predictions.iter().enumerate() {
                    let (symbol, extra, _) =
                        uint_encode(pack_signed((i64::from(node.level) - prediction) as i32));
                    let cost = path_cost
                        + distortion
                        + lambda * (channel_bits[context][symbol as usize] + extra as f32);
                    if cost < node.cost {
                        node.cost = cost;
                        node.back = back as u16;
                        winning_prediction = Some(back);
                    }
                }
                // Only the winner needs an update; losing transitions never
                // contribute predictor state to a later column.
                if let Some(back) = winning_prediction {
                    node.update = WpState::update_for(predictions[back].2, i64::from(node.level));
                }
                history.push(TrellisStep {
                    level: node.level,
                    back: node.back,
                });
            }
            std::mem::swap(older, previous);
            std::mem::swap(previous, column);
        }
        let mut best = previous
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cost.total_cmp(&b.cost))
            .unwrap()
            .0;
        #[cfg(debug_assertions)]
        let (path_cost, mut replay_cost) = (previous[best].cost, 0.0f32);
        for x in (0..w).rev() {
            let step = history[starts[x] + best];
            levels[y * w + x] = step.level;
            best = usize::from(step.back);
        }
        // Commit the row through the real predictor so the rows below, and
        // the emitted tokens, see exactly what the DC coder will.
        wp.restore_row_base(y, base);
        for x in 0..w {
            let get = |x: usize, y: usize| i64::from(levels[y * w + x]);
            let west = if x > 0 { get(x - 1, y) } else { 0 };
            let north = if y > 0 { get(x, y - 1) } else { west };
            let west = if x == 0 { north } else { west };
            let ne = if y > 0 {
                get((x + 1).min(w - 1), y - 1)
            } else {
                west
            };
            let nw = if y > 0 && x > 0 {
                get(x - 1, y - 1)
            } else {
                north
            };
            let nn = if y > 1 { get(x, y - 2) } else { north };
            let weighted = wp.predict(x, y, north, west, ne, nw, nn);
            let prop =
                (K_GRAD_RANGE_MID + wp.wp_prop).clamp(K_GRAD_RANGE_MIN, K_GRAD_RANGE_MAX) as usize;
            let context = K_GRADIENT_CONTEXT_LUT[prop] as usize;
            let prediction = if predictors[context] {
                (north + west - nw).clamp(north.min(west), north.max(west))
            } else {
                weighted
            };
            let level = get(x, y);
            #[cfg(debug_assertions)]
            {
                let target = targets[y * w + x];
                let error = target - level as f32;
                let error = if !target.is_finite() || error.abs() > 2.0 * f32::from(i16::MAX) {
                    0.0
                } else {
                    error
                };
                replay_cost += error * error
                    + lambda * prices.residual_bits(mchan, context, (level - prediction) as i32);
            }
            emit(
                Token::new(context as u32, pack_signed((level - prediction) as i32)),
                Token::new(context as u32, pack_signed((level - weighted) as i32)),
            );
            wp.update(level, x, y);
        }
        // The survivors' staged predictor state must reproduce the real one.
        #[cfg(debug_assertions)]
        assert!(
            (replay_cost - path_cost).abs() <= 1e-3 * path_cost.abs().max(1.0),
            "row {y}: trellis {path_cost} vs replay {replay_cost}"
        );
    }
}

/// Rate price of the X DC step choice, in squared low-passed default-step
/// error per bit.
const X_DC_STEP_LAMBDA: f64 = 0.02;
/// Spread, in DC blocks, of the low-pass the X DC step choice measures error
/// through.
const X_DC_STEP_SIGMA: f32 = 3.0;
/// Finest X DC step multiplier the choice considers.
const X_DC_STEP_FINEST: f32 = 0.5;
/// X DC step multipliers are searched this far apart.
const X_DC_STEP_SPACING: f32 = 0.02;

/// Choose a finer X DC step for the frame when its rate-distortion cost
/// beats the default's, and re-quantize X from its source under it.
///
/// A smooth, nearly neutral region holds one X value. Where that value sits
/// between two levels every block rounds the same way: the plane costs no
/// bits and the whole region shifts hue. Squared error per block cannot tell
/// that shift from rounding noise in texture, which the eye masks, so the
/// error plane is low-passed first, as chroma vision is: noise averages out,
/// a shared offset does not. Returns the frame's DC steps.
pub(super) fn choose_x_dc_step(
    groups: &mut [DcGroupData],
    distp: &DistanceParams,
    cfl: CflFrame,
    quantize: crate::group::QuantizeDcCflFn,
) -> [f32; 3] {
    let mut dc_step = distp.dc_step;
    if groups.is_empty() || groups.iter().any(|dc| dc.source_dc_x.is_none()) {
        return dc_step;
    }
    let base = dc_step[0];
    let kernel = gaussian_kernel(X_DC_STEP_SIGMA);
    let mut price = DcPlanePrice::default();
    let (mut levels, mut errors, mut blurred) = (Vec::new(), Vec::new(), Vec::new());
    let mut cost = |m: f32, apply: bool, groups: &mut [DcGroupData]| -> f64 {
        let mut step = dc_step;
        step[0] = base * m;
        let scale = INV_DC_QUANT[0] / step[0] * distp.scale_dc;
        let factor = dc_cfl_factor_x(step, cfl.base_x);
        price.clear();
        let mut distortion = 0.0f64;
        for dc in groups.iter_mut() {
            let source = dc.source_dc_x.as_ref().unwrap();
            let (w, h) = (source.xsize(), source.ysize());
            levels.resize(w * h, 0i16);
            errors.resize(w * h, 0.0f32);
            for y in 0..h {
                let luma = dc.quant_dc.plane_row(1, y);
                let row = &mut levels[y * w..][..w];
                quantize(source.row(y), luma, scale, factor, row);
                for (((&src, &luma), &level), error) in source
                    .row(y)
                    .iter()
                    .zip(luma)
                    .zip(row.iter())
                    .zip(&mut errors[y * w..][..w])
                {
                    *error = (fmla(src, scale, -(luma as f32) * factor) - f32::from(level)) * m;
                }
            }
            distortion += lowpass_energy(&errors, w, h, &kernel, &mut blurred);
            price.add(&levels, w, h, w);
            if apply {
                for y in 0..h {
                    dc.quant_dc
                        .plane_row_mut(0, y)
                        .copy_from_slice(&levels[y * w..][..w]);
                }
            }
        }
        distortion + X_DC_STEP_LAMBDA * price.bits()
    };
    let candidates = ((1.0 - X_DC_STEP_FINEST) / X_DC_STEP_SPACING).round() as usize;
    let mut best = (cost(1.0, false, groups), 1.0f32);
    for k in 1..=candidates {
        let multiplier = 1.0 - X_DC_STEP_SPACING * k as f32;
        let m = crate::color_correlation::signaled_dc_step(0, base * multiplier) / base;
        let candidate = cost(m, false, groups);
        if candidate < best.0 {
            best = (candidate, m);
        }
    }
    if best.1 != 1.0 {
        cost(best.1, true, groups);
        dc_step[0] = base * best.1;
    }
    dc_step
}

/// Normalized Gaussian taps out to three spreads.
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as i32;
    let taps: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = taps.iter().sum();
    taps.into_iter().map(|v| v / sum).collect()
}

/// Index `i` reflected into `0..n`, edge samples repeated.
#[inline]
fn mirror(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - i - 1;
        } else {
            return i as usize;
        }
        if n == 1 {
            return 0;
        }
    }
}

/// Energy of a `w`×`h` plane after a separable blur by `kernel`.
fn lowpass_energy(plane: &[f32], w: usize, h: usize, kernel: &[f32], tmp: &mut Vec<f32>) -> f64 {
    let n = w * h;
    if n == 0 || kernel.is_empty() {
        tmp.resize(n, 0.0);
        tmp.fill(0.0);
        return 0.0;
    }
    let r = kernel.len() / 2;
    let padded_len = w + kernel.len() - 1;
    // Reuse the plane's scratch allocation for the padded source row and
    // vertical accumulator too. Restore its original length before returning.
    tmp.resize(n + padded_len + w, 0.0);
    let (horizontal, scratch) = tmp.split_at_mut(n);
    let (padded, accum) = scratch.split_at_mut(padded_len);
    for (src, dst) in plane[..n]
        .chunks_exact(w)
        .zip(horizontal.chunks_exact_mut(w))
    {
        let (left, rest) = padded.split_at_mut(r);
        let (middle, right) = rest.split_at_mut(w);
        middle.copy_from_slice(src);
        for (i, value) in left.iter_mut().enumerate() {
            *value = src[mirror(i as isize - r as isize, w)];
        }
        for (i, value) in right.iter_mut().enumerate() {
            *value = src[mirror(w as isize + i as isize, w)];
        }
        dst.fill(0.0);
        for (j, &tap) in kernel.iter().enumerate() {
            for (value, &sample) in dst.iter_mut().zip(&padded[j..j + w]) {
                *value += tap * sample;
            }
        }
    }
    let mut energy = 0.0f64;
    for y in 0..h {
        accum.fill(0.0);
        for (j, &tap) in kernel.iter().enumerate() {
            let row = mirror(y as isize + j as isize - r as isize, h);
            for (value, &sample) in accum.iter_mut().zip(&horizontal[row * w..][..w]) {
                *value += tap * sample;
            }
        }
        energy = accum.iter().fold(energy, |sum, &value| {
            let value = f64::from(value);
            sum + value * value
        });
    }
    tmp.truncate(n);
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prices() -> Prices {
        let mut bits = vec![[5.0; ALPHABET_SIZE]; 3 * K_NUM_DC_CONTEXTS];
        for row in &mut bits {
            row[0] = 0.0;
        }
        Prices { bits }
    }

    #[test]
    fn streamed_prices_match_full_token_streams() {
        let mut groups = Vec::new();
        for (w, h) in [(64, 64), (17, 9), (1, 5), (3, 2)] {
            let mut dc = DcGroupData::new(w, h).unwrap();
            for c in 0..3 {
                for (i, level) in dc
                    .quant_dc
                    .plane_mut(c)
                    .as_mut_slice()
                    .iter_mut()
                    .enumerate()
                {
                    *level = if c == 0 {
                        ((i / w + i % w) % 31) as i16
                    } else {
                        ((i * 7919 + c * 2971) % 1001) as i16 - 500
                    };
                }
            }
            groups.push(dc);
        }
        let (wp, grad): (Vec<_>, Vec<_>) = groups
            .iter()
            .map(|dc| {
                collect_dc_tokens_with_gradient(
                    dc,
                    &DC_PREDICTOR_WEIGHTED,
                    &mut Vec::new(),
                    false,
                    true,
                )
            })
            .unzip();
        let mut scratch = DcPredictorScratch::default();
        let expected = super::super::choose_dc_predictors(&wp, &grad, &mut scratch);
        let (prices, predictors, base_prices) = Prices::new(&groups, &mut scratch).unwrap();
        assert_eq!(predictors, expected);
        assert!(expected.iter().any(|&gradient| gradient));
        assert!(expected.iter().any(|&gradient| !gradient));
        let mut counts = vec![[0u32; ALPHABET_SIZE]; 3 * K_NUM_DC_CONTEXTS];
        for ((dc, wp), grad) in groups.iter().zip(wp).zip(grad) {
            let mut offset = 0;
            for (mchan, c) in [1, 0, 2].into_iter().enumerate() {
                let end = offset + dc.quant_dc.plane(c).as_slice().len();
                for (w, g) in wp[offset..end].iter().zip(&grad[offset..end]) {
                    let token = if expected[w.context as usize] { g } else { w };
                    let (symbol, _, _) = uint_encode(token.value);
                    counts[mchan * K_NUM_DC_CONTEXTS + token.context as usize][symbol as usize] +=
                        1;
                }
                offset = end;
            }
        }
        for (counts, bits) in counts.iter().zip(&prices.bits) {
            let total =
                counts.iter().map(|&n| f64::from(n)).sum::<f64>() + 0.5 * ALPHABET_SIZE as f64;
            for (&count, &bits) in counts.iter().zip(bits) {
                let expected = f_log2(total / (f64::from(count) + 0.5)) as f32;
                assert_eq!(bits.to_bits(), expected.to_bits());
            }
        }
        for (c, actual) in [0, 2].into_iter().zip(base_prices) {
            let mut expected = DcPlanePrice::default();
            for dc in &groups {
                let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
                expected.add(dc.quant_dc.plane(c).as_slice(), w, h, w);
            }
            assert_eq!(actual.counts, expected.counts);
            assert_eq!(actual.raw_bits, expected.raw_bits);
            assert_eq!(actual.bits().to_bits(), expected.bits().to_bits());
        }
        let (empty_prices, empty_predictors, _) = Prices::new(&[], &mut scratch).unwrap();
        assert_eq!(empty_predictors, DC_PREDICTOR_WEIGHTED);
        assert!(empty_prices.bits.iter().flatten().all(|&b| b == 7.0));
    }

    #[test]
    fn rounding_scratch_handles_width_changes() {
        let mut scratch = RoundingScratch::new();
        let prices = prices();
        for (w, h) in [(17, 4), (1, 7), (64, 5), (2, 3), (0, 3), (3, 0), (17, 4)] {
            let original: Vec<i16> = (0..w * h)
                .map(|i| ((i * 19 + i / w * 7) % 101) as i16 - 50)
                .collect();
            let targets: Vec<f32> = original.iter().map(|&q| f32::from(q) - 0.49).collect();
            let mut expected = original.clone();
            let mut actual = original;
            let mut expected_tokens = Vec::new();
            let mut actual_tokens = Vec::new();
            refine_channel(
                &mut expected,
                &targets,
                w,
                h,
                1,
                &DC_PREDICTOR_WEIGHTED,
                &prices,
                0.06,
                |t| expected_tokens.push((t.context, t.value)),
            );
            refine_channel_with_scratch(
                &mut actual,
                &targets,
                w,
                h,
                1,
                &DC_PREDICTOR_WEIGHTED,
                &prices,
                0.06,
                &mut scratch,
                |t, _| actual_tokens.push((t.context, t.value)),
            );
            assert_eq!(actual, expected);
            assert_eq!(actual_tokens, expected_tokens);
        }
    }

    #[test]
    fn replay_prices_match_a_separate_weighted_pass() {
        let (w, h) = (17, 9);
        let mut levels: Vec<i16> = (0..w * h)
            .map(|i| ((i * 19 + i / w * 7) % 101) as i16 - 50)
            .collect();
        let targets: Vec<f32> = levels.iter().map(|&q| f32::from(q) - 0.49).collect();
        let mut actual = DcPlanePrice::default();
        refine_channel_with_scratch(
            &mut levels,
            &targets,
            w,
            h,
            1,
            &std::array::from_fn(|i| i % 2 == 0),
            &prices(),
            0.06,
            &mut RoundingScratch::new(),
            |_, weighted| actual.add_token(weighted),
        );
        let mut expected = DcPlanePrice::default();
        expected.add(&levels, w, h, w);
        assert_eq!(actual.counts, expected.counts);
        assert_eq!(actual.raw_bits, expected.raw_bits);
        assert_eq!(actual.bits().to_bits(), expected.bits().to_bits());
    }

    // Frozen before changing trellis storage or scheduling. Include exact
    // levels and tokens so equal-cost tie order and WP history are covered.
    #[test]
    fn rounding_matches_frozen_trellis_decisions() {
        let mut prices = prices();
        for (context, row) in prices.bits.iter_mut().enumerate() {
            for (symbol, bits) in row.iter_mut().enumerate() {
                *bits = ((context * 17 + symbol * 13) % 53) as f32 * 0.125;
            }
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut record = |bytes: &[u8]| {
            for &byte in bytes {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
        };
        let mut random = 0x85eb_ca6bu32;
        for (w, h) in [
            (0, 3),
            (3, 0),
            (1, 1),
            (1, 7),
            (2, 7),
            (3, 4),
            (17, 9),
            (64, 5),
        ] {
            for pattern in 0..5 {
                let original: Vec<i16> = (0..w * h)
                    .map(|i| {
                        random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        match pattern {
                            0 => 20,
                            1 => (i % 37) as i16 - 18,
                            2 => (random % 101) as i16 - 50,
                            _ => random as i16,
                        }
                    })
                    .collect();
                let targets: Vec<f32> = original
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| {
                        if pattern == 4 {
                            match i % 13 {
                                0 => return f32::NAN,
                                1 => return f32::INFINITY,
                                2 => return f32::NEG_INFINITY,
                                3 => return 1e20,
                                4 => return -1e20,
                                5 => return f32::from(i16::MIN) - 0.75,
                                6 => return f32::from(i16::MAX) + 0.75,
                                _ => {}
                            }
                        }
                        let offset = [-0.5, 0.5, -0.49, 0.49, -0.6, 0.6, 0.0, 0.25][i % 8];
                        f32::from(q) + if pattern == 3 { offset * 12.0 } else { offset }
                    })
                    .collect();
                for predictors in [
                    DC_PREDICTOR_WEIGHTED,
                    [true; K_NUM_DC_CONTEXTS],
                    std::array::from_fn(|i| i % 2 == 0),
                ] {
                    for mchan in [1, 2] {
                        for lambda in [0.0, 0.04, 0.225] {
                            let mut levels = original.clone();
                            refine_channel(
                                &mut levels,
                                &targets,
                                w,
                                h,
                                mchan,
                                &predictors,
                                &prices,
                                lambda,
                                |token| {
                                    record(&token.context.to_le_bytes());
                                    record(&token.value.to_le_bytes());
                                },
                            );
                            for level in levels {
                                record(&level.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            hash, 0x3978_cea8_1d8d_f792,
            "trellis decisions: {hash:016x}"
        );
    }

    #[test]
    #[ignore = "isolated DC rounding timing; run in release with --nocapture"]
    fn benchmark_dc_rounding() {
        let prices = prices();
        for (name, w, h, texture) in [
            ("smooth", 256, 256, false),
            ("texture", 256, 256, true),
            ("narrow", 17, 256, true),
        ] {
            let original: Vec<i16> = (0..w * h)
                .map(|i| {
                    if texture {
                        ((i * 7919 + i / w * 2971) % 1001) as i16 - 500
                    } else {
                        ((i / w + i % w) / 13) as i16
                    }
                })
                .collect();
            let targets: Vec<f32> = original
                .iter()
                .enumerate()
                .map(|(i, &q)| f32::from(q) + [-0.49, 0.45, -0.6, 0.2, 0.55][i % 5])
                .collect();
            let mut levels = original.clone();
            let start = std::time::Instant::now();
            for _ in 0..30 {
                levels.copy_from_slice(&original);
                refine_channel(
                    &mut levels,
                    &targets,
                    w,
                    h,
                    1,
                    &DC_PREDICTOR_WEIGHTED,
                    &prices,
                    0.06,
                    |_| {},
                );
                std::hint::black_box(&levels);
            }
            eprintln!(
                "{name}: {:.3} ms/pass",
                start.elapsed().as_secs_f64() * 1e3 / 30.0
            );
        }
    }

    #[test]
    fn truncated_bounds_match_clamped_floor_and_ceil() {
        let check = |target: f32| {
            let clamp = |v: i64| v.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
            assert_eq!(
                dc_bounds_from_truncation(target),
                [clamp(target.floor() as i64), clamp(target.ceil() as i64)],
                "target={target:?}, bits={:08x}",
                target.to_bits(),
            );
        };
        for integer in -32770..=32770 {
            for fraction in [-0.75, -0.25, 0.0, 0.25, 0.75] {
                check(integer as f32 + fraction);
            }
        }
        for value in [
            f32::MIN,
            f32::MAX,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(1),
            -f32::from_bits(1),
            -0.0,
            i64::MIN as f32,
            i64::MAX as f32,
        ] {
            check(value);
        }
        let mut state = 0x85eb_ca6bu32;
        for _ in 0..100_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            check(f32::from_bits(state));
        }
    }

    #[test]
    fn rounding_favors_prediction_near_a_quantization_boundary() {
        let mut levels = [20, 21, 20];
        refine_channel(
            &mut levels,
            &[20.0, 20.51, 20.3],
            3,
            1,
            0,
            &DC_PREDICTOR_WEIGHTED,
            &prices(),
            0.02,
            |_| {},
        );
        assert_eq!(levels, [20, 20, 20]);
    }

    #[test]
    fn chosen_residuals_match_the_dc_tokenizer_at_every_edge() {
        for w in [1, 2, 3, 17] {
            for h in [1, 2, 4] {
                for predictors in [DC_PREDICTOR_WEIGHTED, std::array::from_fn(|i| i % 2 == 0)] {
                    let mut levels: Vec<i16> = (0..w * h)
                        .map(|i| ((i * 19 + i / w * 7) % 101) as i16 - 50)
                        .collect();
                    let targets: Vec<f32> = levels
                        .iter()
                        .enumerate()
                        .map(|(i, &q)| f32::from(q) + [-0.49, 0.45, -0.6, 0.2, 0.55][i % 5])
                        .collect();
                    let mut tokens = Vec::new();
                    refine_channel(
                        &mut levels,
                        &targets,
                        w,
                        h,
                        0,
                        &predictors,
                        &prices(),
                        0.04,
                        |t| tokens.push(t),
                    );
                    let mut expected = Vec::new();
                    super::super::walk_dc_channel(
                        &levels,
                        w,
                        h,
                        w,
                        0,
                        &predictors,
                        &mut Vec::new(),
                        false,
                        false,
                        &mut expected,
                        &mut Vec::new(),
                    );
                    assert_eq!(tokens.len(), expected.len());
                    for (a, b) in tokens.iter().zip(&expected) {
                        assert_eq!((a.context, a.value), (b.context, b.value), "w={w}, h={h}");
                    }
                }
            }
        }
    }

    /// Rounding the second block to its nearest level costs less there, but
    /// every block after it then pays a residual.
    #[test]
    fn a_run_is_rounded_as_a_whole() {
        let targets = [20.0, 20.6, 20.4, 20.4, 20.4, 20.4];
        let original = [20i16, 21, 20, 20, 20, 20];
        let row_cost = |levels: &[i16]| {
            let mut tokens = Vec::new();
            super::super::walk_dc_channel(
                levels,
                levels.len(),
                1,
                levels.len(),
                0,
                &DC_PREDICTOR_WEIGHTED,
                &mut Vec::new(),
                false,
                false,
                &mut tokens,
                &mut Vec::new(),
            );
            let prices = prices();
            tokens
                .iter()
                .zip(levels)
                .zip(targets)
                .map(|((t, &q), target)| {
                    let (symbol, extra, _) = uint_encode(t.value);
                    (target - f32::from(q)).powi(2)
                        + 0.03 * (prices.bits[t.context as usize][symbol as usize] + extra as f32)
                })
                .sum::<f32>()
        };
        let mut levels = original;
        refine_channel(
            &mut levels,
            &targets,
            6,
            1,
            0,
            &DC_PREDICTOR_WEIGHTED,
            &prices(),
            0.03,
            |_| {},
        );
        assert_eq!(levels, [20; 6]);
        assert!(row_cost(&levels) < row_cost(&original));
    }

    #[test]
    fn rounding_handles_saturated_and_nonfinite_targets() {
        let mut levels = [i16::MAX, i16::MIN, 42, -7];
        refine_channel(
            &mut levels,
            &[1e20, -1e20, f32::NAN, f32::INFINITY],
            2,
            2,
            0,
            &DC_PREDICTOR_WEIGHTED,
            &prices(),
            0.04,
            |_| {},
        );
        assert_eq!(levels, [i16::MAX, i16::MIN, 42, -7]);
    }

    #[test]
    fn missing_sources_leave_every_group_unchanged() {
        let mut groups = [DcGroupData::new(3, 2).unwrap()];
        groups[0].quant_dc.plane_mut(1).as_mut_slice().fill(123);
        refine_dc_rounding(
            &mut groups,
            &super::super::compute_distance_params(4.0),
            0,
            CflFrame::XYB,
            false,
            &mut DcPredictorScratch::default(),
        )
        .unwrap();
        assert!(
            groups[0]
                .quant_dc
                .plane(1)
                .as_slice()
                .iter()
                .all(|&q| q == 123)
        );
    }

    #[test]
    fn constant_chroma_keeps_its_levels_without_a_rate_gain() {
        use crate::image::Plane;
        let params = super::super::compute_distance_params(4.0);
        let mut dc = DcGroupData::new(16, 8).unwrap();
        dc.quant_dc.plane_mut(0).as_mut_slice().fill(10);
        dc.source_dc_x = Some(Plane::new_fill(
            16,
            8,
            10.75 * params.dc_step[0] / (INV_DC_QUANT[0] * params.scale_dc),
        ));
        dc.source_dc_y = Some(Plane::new(16, 8));
        dc.source_dc_b = Some(Plane::new(16, 8));
        let mut groups = [dc];
        refine_dc_rounding(
            &mut groups,
            &params,
            0,
            CflFrame::XYB,
            false,
            &mut DcPredictorScratch::default(),
        )
        .unwrap();
        assert!(
            groups[0]
                .quant_dc
                .plane(0)
                .as_slice()
                .iter()
                .all(|&q| q == 10)
        );
        assert!(
            groups[0]
                .quant_dc
                .plane(1)
                .as_slice()
                .iter()
                .all(|&q| q == 0)
        );
    }

    fn x_step_fixture(params: &DistanceParams, x_of: impl Fn(usize, usize) -> f32) -> DcGroupData {
        use crate::image::Plane;
        let (w, h) = (32, 32);
        let mut dc = DcGroupData::new(w, h).unwrap();
        let unit = params.dc_step[0] / (INV_DC_QUANT[0] * params.scale_dc);
        let mut source = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                source.row_mut(y)[x] = x_of(x, y) * unit;
            }
        }
        dc.source_dc_x = Some(source);
        dc
    }

    #[test]
    fn x_step_moves_a_level_under_a_flat_offset() {
        let params = super::super::compute_distance_params(7.0);
        // Every block sits 0.4 levels above level 3: all round down alike.
        let mut groups = [x_step_fixture(&params, |_, _| 3.4)];
        let step = choose_x_dc_step(
            &mut groups,
            &params,
            CflFrame::XYB,
            crate::group::quantize_dc_cfl_scalar,
        );
        assert!(step[0] < params.dc_step[0], "kept the default step");
        assert_eq!([step[1], step[2]], [params.dc_step[1], params.dc_step[2]]);
        // X is re-quantized under the chosen step, and lands closer.
        let m = step[0] / params.dc_step[0];
        let level = groups[0].quant_dc.plane(0).as_slice()[0];
        assert!(
            groups[0]
                .quant_dc
                .plane(0)
                .as_slice()
                .iter()
                .all(|&q| q == level)
        );
        assert!((f32::from(level) * m - 3.4).abs() < 0.4 * 0.5);
    }

    #[test]
    fn x_step_keeps_the_default_for_unbiased_texture() {
        let params = super::super::compute_distance_params(7.0);
        let mut state = 0x9e37_79b9u32;
        let mut noise = vec![0.0f32; 32 * 32];
        for v in &mut noise {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (state >> 8) as f32 / (1u32 << 24) as f32 * 40.0 - 20.0;
        }
        let mut groups = [x_step_fixture(&params, |x, y| noise[y * 32 + x])];
        let step = choose_x_dc_step(
            &mut groups,
            &params,
            CflFrame::XYB,
            crate::group::quantize_dc_cfl_scalar,
        );
        assert_eq!(step, params.dc_step);
        assert!(
            groups[0]
                .quant_dc
                .plane(0)
                .as_slice()
                .iter()
                .all(|&q| q == 0)
        );
    }

    #[test]
    fn x_step_needs_every_source() {
        let params = super::super::compute_distance_params(7.0);
        let mut groups = [
            x_step_fixture(&params, |_, _| 3.4),
            DcGroupData::new(4, 4).unwrap(),
        ];
        let step = choose_x_dc_step(
            &mut groups,
            &params,
            CflFrame::XYB,
            crate::group::quantize_dc_cfl_scalar,
        );
        assert_eq!(step, params.dc_step);
        assert!(
            groups[0]
                .quant_dc
                .plane(0)
                .as_slice()
                .iter()
                .all(|&q| q == 0)
        );
    }

    #[test]
    fn lowpass_keeps_offsets_and_averages_out_alternation() {
        let kernel = gaussian_kernel(X_DC_STEP_SIGMA);
        assert!((kernel.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let (w, h) = (24, 20);
        let mut tmp = Vec::new();
        let flat = vec![0.5f32; w * h];
        let energy = lowpass_energy(&flat, w, h, &kernel, &mut tmp);
        assert!((energy - 0.25 * (w * h) as f64).abs() < 1e-3);
        let checker: Vec<f32> = (0..w * h)
            .map(|i| if (i % w + i / w) % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        assert!(lowpass_energy(&checker, w, h, &kernel, &mut tmp) < 1e-3 * energy);
        // A single row or column mirrors onto itself.
        assert!((lowpass_energy(&[0.5; 7], 7, 1, &kernel, &mut tmp) - 1.75).abs() < 1e-4);
        assert!((lowpass_energy(&[0.5; 7], 1, 7, &kernel, &mut tmp) - 1.75).abs() < 1e-4);
    }

    #[test]
    fn lowpass_matches_reference_with_reused_scratch() {
        fn reference(plane: &[f32], w: usize, h: usize, kernel: &[f32], tmp: &mut Vec<f32>) -> f64 {
            let r = (kernel.len() / 2) as isize;
            tmp.resize(w * h, 0.0);
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0;
                    for (j, &tap) in kernel.iter().enumerate() {
                        acc += tap * plane[y * w + mirror(x as isize + j as isize - r, w)];
                    }
                    tmp[y * w + x] = acc;
                }
            }
            let mut energy = 0.0f64;
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0;
                    for (j, &tap) in kernel.iter().enumerate() {
                        acc += tap * tmp[mirror(y as isize + j as isize - r, h) * w + x];
                    }
                    energy += f64::from(acc).powi(2);
                }
            }
            energy
        }

        let mut tmp = vec![f32::NAN; 512];
        let mut expected_tmp = Vec::new();
        let kernels = [
            Vec::new(),
            vec![0.7],
            vec![0.3, -0.2],
            vec![-0.1, 0.2, 0.7, -0.4],
            gaussian_kernel(X_DC_STEP_SIGMA),
        ];
        for (w, h) in (0..12).flat_map(|w| (0..12).map(move |h| (w, h))).chain([
            (64, 32),
            (32, 64),
            (256, 3),
            (3, 256),
        ]) {
            let plane: Vec<f32> = (0..w * h)
                .map(|i| ((i * 73 % 251) as f32 - 125.0) / 19.0)
                .collect();
            for kernel in &kernels {
                let expected = reference(&plane, w, h, kernel, &mut expected_tmp);
                let actual = lowpass_energy(&plane, w, h, kernel, &mut tmp);
                assert_eq!(actual.to_bits(), expected.to_bits(), "{w}x{h} {kernel:?}");
                assert_eq!(tmp, expected_tmp, "{w}x{h} {kernel:?}");
            }
        }
    }
}
