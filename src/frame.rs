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

mod dc_rdo;
mod dc_smoothing;
mod extra_tree;
#[cfg(feature = "extraslow")]
mod loop_arbiter;
#[cfg(feature = "extraslow")]
mod loop_group_map;
use dc_smoothing::{
    choose_chroma_dc_coding, compensate_b_dc, flatten_chroma_dc, skip_dc_smoothing,
};

#[cfg(feature = "extraslow")]
use crate::Arbiter;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, DcPredictorScratch};
use crate::color_correlation::choose_ytob_dc;
use crate::dc_group_data::{DcGroupData, STRATEGY_DCT};
use crate::dct::fmla;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::encoding_context::FrameRole;
#[cfg(feature = "extraslow")]
use crate::entropy::TokenSlice;
use crate::entropy::{
    AcTokens, EntropyCode, Token, f_log2, optimize_entropy_code, pack_signed, write_entropy_code,
    write_token,
};
use crate::group::write_ac_group;
use crate::image::{Image3B, Image3F, Image3S, Rect};
use crate::patches::{MODULAR_PATCH_REF_ID, PATCH_REF_ID, VarDctFrameKind};
use crate::quant_weights::quant_table_slot_of;
use crate::static_entropy_codes::{
    K_CONTEXT_TREE_TOKENS, K_GRADIENT_CONTEXT_LUT, K_NUM_DC_CONTEXTS,
};
use crate::util::EncodeError;
#[cfg(feature = "extraslow")]
use loop_arbiter::LoopArbiter;
#[cfg(feature = "extraslow")]
use loop_group_map::LoopGroupMap;

const K_BLOCK_DIM: usize = 8;
const K_TILE_DIM: usize = 64;
pub(crate) const K_GROUP_DIM: usize = 256;
pub(crate) const K_DC_GROUP_DIM: usize = 2048;
const K_GROUP_DIM_IN_BLOCKS: usize = 32; // = K_GROUP_DIM / K_BLOCK_DIM
const K_TILE_DIM_IN_BLOCKS: usize = 8; // = K_TILE_DIM / K_BLOCK_DIM
const K_NUM_TREE_CONTEXTS: usize = 6;

const K_GRAD_RANGE_MID: i64 = 512;
const K_GRAD_RANGE_MIN: i64 = 0;
const K_GRAD_RANGE_MAX: i64 = 1023;

#[allow(dead_code)]
struct ImageDim {
    xsize: usize,
    ysize: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_groups: usize,
    ysize_groups: usize,
    xsize_dc_groups: usize,
    ysize_dc_groups: usize,
    num_groups: usize,
    num_dc_groups: usize,
}

impl ImageDim {
    fn new(xsize: usize, ysize: usize) -> Self {
        let xsize_blocks = xsize.div_ceil(K_BLOCK_DIM);
        let ysize_blocks = ysize.div_ceil(K_BLOCK_DIM);
        let xsize_groups = xsize.div_ceil(K_GROUP_DIM);
        let ysize_groups = ysize.div_ceil(K_GROUP_DIM);
        let xsize_dc_groups = xsize.div_ceil(K_DC_GROUP_DIM);
        let ysize_dc_groups = ysize.div_ceil(K_DC_GROUP_DIM);
        Self {
            xsize,
            ysize,
            xsize_blocks,
            ysize_blocks,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_groups: xsize_groups * ysize_groups,
            num_dc_groups: xsize_dc_groups * ysize_dc_groups,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DistanceParams {
    distance: f32,
    global_scale: i32,
    quant_dc: i32,
    scale: f32,
    scale_dc: f32,
    x_qm_scale: u32,
    epf_iters: u32,
    epf_pass0_scale: Option<f32>,
    epf_quant_mul: Option<f32>,
    gab_enabled: bool,
    /// Multipliers of the default DC steps of X, Y and B.
    dc_step: [f32; 3],
}

const X_QM_REFINE_MAX_DISTANCE: f32 = 0.299;
const DC_REFINE_PEAK: f32 = 1.35;
const DC_REFINE_HOLD: f32 = 3.0;
const DC_REFINE_RELEASE: f32 = 5.0;

#[inline]
fn dc_refinement(distance: f32) -> f32 {
    if distance <= DC_REFINE_HOLD {
        DC_REFINE_PEAK
    } else if distance >= DC_REFINE_RELEASE {
        1.0
    } else {
        const RECIP_REFINE: f32 = 1.0 / (DC_REFINE_RELEASE - DC_REFINE_HOLD);
        let t = (distance - DC_REFINE_HOLD) * RECIP_REFINE;
        fmla(1.0 - DC_REFINE_PEAK, t, DC_REFINE_PEAK)
    }
}

const DCEPF_VLQ_D0: f32 = 8.0;
const DCEPF_VLQ_D1: f32 = 15.0;
const VLQ_DC_MUL: f32 = 0.7;

#[inline]
fn dcepf_vlq_t(distance: f32) -> f32 {
    ((distance - DCEPF_VLQ_D0) / (DCEPF_VLQ_D1 - DCEPF_VLQ_D0)).clamp(0.0, 1.0)
}

fn quant_dc(distance: f32) -> f32 {
    let refine = dc_refinement(distance);
    let vlq = fmla(dcepf_vlq_t(distance), VLQ_DC_MUL - 1.0, 1.0);
    let k_dc_quant_pow = 0.57f32;
    let k_dc_quant = 1.12f32;
    let k_dc_mul = 2.9f32;
    let effective = k_dc_mul * (distance / k_dc_mul).powf(k_dc_quant_pow);
    let effective = f32::clamp(effective, 0.5 * distance, distance);
    (k_dc_quant / effective).min(50.0) * refine * vlq
}

fn compute_distance_params(distance: f32) -> DistanceParams {
    const K_GLOBAL_SCALE_DENOM: i32 = 1 << 16;
    const K_AC_QUANT: f32 = 0.8;
    // Keep the integer quant field away from the very coarse 2/3/4 range. The
    // corresponding reduction in global scale preserves the effective AC
    // quantizer while giving AQ enough integer resolution to vary smoothly.
    const K_QUANT_FIELD_TARGET: f32 = 10.0;

    let qdc = quant_dc(distance);
    let scale = K_GLOBAL_SCALE_DENOM as f32 * K_AC_QUANT / (distance * K_QUANT_FIELD_TARGET);
    // AC and DC are signaled independently. Capping the AC scale by qdc made
    // it freeze once quant_dc() reached its high-distance cap, producing a
    // rate/quality plateau followed by a cliff when the cap finally released.
    let global_scale = i32::clamp(scale.round() as i32, 1, 1 << 15);
    let scale_f = global_scale as f32 / K_GLOBAL_SCALE_DENOM as f32;
    let qd = ((qdc / scale_f) + 0.5) as i32;
    let qd = i32::clamp(qd, 1, 1 << 16);
    let scale_dc = qd as f32 * scale_f;

    let mut x_qm_scale: u32 = 2;
    if distance >= 1.25 {
        x_qm_scale += 1;
    }
    if distance > 9.0 {
        x_qm_scale += 1;
    }
    if distance < X_QM_REFINE_MAX_DISTANCE {
        x_qm_scale += 1;
    }
    static EPF_THRESHOLDS: [f32; 2] = [0.7, 1.5];
    let mut epf_iters: u32 = 0;
    for t in EPF_THRESHOLDS {
        if distance >= t {
            epf_iters += 1;
        }
    }
    let epf_pass0_scale = epf_pass0_scale(distance);
    if epf_pass0_scale.is_some() || dcepf_vlq_t(distance) >= 0.5 {
        epf_iters = 3;
    }
    let gab_enabled = false; // measured net-negative for rate-matched SSIMU2

    DistanceParams {
        distance,
        global_scale,
        quant_dc: qd,
        scale: scale_f,
        scale_dc,
        x_qm_scale,
        epf_iters,
        epf_pass0_scale,
        epf_quant_mul: None,
        gab_enabled,
        dc_step: chroma_dc_steps(qdc),
    }
}

/// Multiplier of the default X DC step. The spec's X DC quantizer is far
/// finer than the X AC band it sits under; twice the step buys back 1-2% of
/// the file at better SSIMULACRA2, butteraugli and cvvdp alike.
const X_DC_STEP: f32 = 2.0;

/// Distance from which the chroma DC steps grow slower than the luma DC step.
const CHROMA_DC_GROWTH_DISTANCE: f32 = 3.5;
/// Exponents of the X and B DC step growth relative to the luma DC step's.
static CHROMA_DC_GROWTH: [f32; 2] = [0.75, 0.5];

/// DC step multipliers for X, Y and B at the luma DC quantizer `qdc`.
///
/// Past `CHROMA_DC_GROWTH_DISTANCE` the chroma steps follow only a power of
/// the luma step's growth: a flat, nearly neutral region otherwise sits
/// between two chroma DC levels, and every block rounds the same way into a
/// visible hue shift.
fn chroma_dc_steps(qdc: f32) -> [f32; 3] {
    let growth = (quant_dc(CHROMA_DC_GROWTH_DISTANCE) / qdc).max(1.0);
    [
        X_DC_STEP * growth.powf(CHROMA_DC_GROWTH[0] - 1.0),
        1.0,
        growth.powf(CHROMA_DC_GROWTH[1] - 1.0),
    ]
}

const EPF_PASS0_START_D: f32 = 2.5;
const EPF_PASS0_SCALE: f32 = 3.6;
const EPF_PASS0_SPEC_SCALE: f32 = 0.9;

fn epf_pass0_scale(distance: f32) -> Option<f32> {
    if distance < EPF_PASS0_START_D {
        return None;
    }
    let t = dcepf_vlq_t(distance);
    if t >= 0.5 {
        return None;
    }
    Some(fmla(
        2.0 * t,
        EPF_PASS0_SPEC_SCALE - EPF_PASS0_SCALE,
        EPF_PASS0_SCALE,
    ))
}

#[inline]
fn clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let mn = n.min(w);
    let mx = n.max(w);
    let g = (n as i64 + w as i64 - l as i64) as i32;
    g.clamp(mn, mx)
}

#[inline(always)]
fn push_dc_wp_token(
    tokens: &mut Vec<Token>,
    gradient_tokens: &mut Vec<Token>,
    collect_gradient: bool,
    wp: &mut crate::weighted_predictor::WpState,
    x: usize,
    y: usize,
    value: i64,
    n: i64,
    w: i64,
    ne: i64,
    nw: i64,
    nn: i64,
    dc_gradient: &DcPredictorChoice,
    mchan: usize,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
) {
    let wp_prediction = wp.predict(x, y, n, w, ne, nw, nn);
    let prop = (K_GRAD_RANGE_MID + wp.wp_prop).clamp(K_GRAD_RANGE_MIN, K_GRAD_RANGE_MAX) as usize;
    if collect_props {
        props.push(crate::dc_tree::dc_prop(mchan, prop));
    }
    wp.update(value, x, y);
    // The context comes from the weighted predictor's error whichever predictor
    // the leaf ends up using, so the tree is navigated identically either way
    // and the per-leaf choice below costs nothing to signal.
    let context = K_GRADIENT_CONTEXT_LUT[prop] as usize;
    let prediction = if dc_gradient[context] {
        (n + w - nw).clamp(n.min(w), n.max(w))
    } else {
        wp_prediction
    };
    tokens.push(Token::new(
        context as u32,
        pack_signed((value - prediction) as i32),
    ));
    if collect_gradient {
        let gradient = (n + w - nw).clamp(n.min(w), n.max(w));
        gradient_tokens.push(Token::new(
            context as u32,
            pack_signed((value - gradient) as i32),
        ));
    }
}

/// Per-leaf predictor selection for the DC plane, indexed by DC context.
pub(crate) type DcPredictorChoice = [bool; K_NUM_DC_CONTEXTS];

pub(crate) const DC_PREDICTOR_WEIGHTED: DcPredictorChoice = [false; K_NUM_DC_CONTEXTS];
const DC_PREDICTOR_GRADIENT: DcPredictorChoice = [true; K_NUM_DC_CONTEXTS];

/// Pick, for every DC leaf independently, whichever predictor codes its own
/// tokens in fewer bits. Both candidate streams share a context assignment, so
/// `wp[i]` and `grad[i]` differ only in value and the merged stream is a
/// per-token pick — no third tokenization pass.
fn choose_dc_predictors(
    wp: &[Vec<Token>],
    grad: &[Vec<Token>],
    scratch: &mut DcPredictorScratch,
) -> DcPredictorChoice {
    // [candidate][context][symbol] counts, plus the raw extra bits per context.
    let counts = &mut scratch.counts;
    let extra = &mut scratch.extra;
    // A flat integer fill avoids expanding the reset into row-sized stores.
    counts.as_flattened_mut().fill(0);
    extra.fill(0);
    for (cand, groups) in [wp, grad].into_iter().enumerate() {
        for token in groups.iter().flatten() {
            let (sym, nbits, _) = crate::entropy::uint_encode(token.value);
            let slot = cand * K_NUM_DC_CONTEXTS + token.context as usize;
            counts[slot][sym as usize] += 1;
            extra[slot] += nbits as u64;
        }
    }
    choose_dc_predictors_from_counts(scratch)
}

fn choose_dc_predictors_from_counts(scratch: &DcPredictorScratch) -> DcPredictorChoice {
    let counts = &scratch.counts;
    let extra = &scratch.extra;
    let cost = |slot: usize| -> f64 {
        let total: u32 = counts[slot].iter().sum();
        if total == 0 {
            return 0.0;
        }
        let total = f64::from(total);
        let entropy: f64 = counts[slot]
            .iter()
            .filter(|&&n| n != 0)
            .map(|&n| f64::from(n) * f_log2(total / f64::from(n)))
            .sum();
        entropy + extra[slot] as f64
    };
    let mut choice = DC_PREDICTOR_WEIGHTED;
    for (ctx, out) in choice.iter_mut().enumerate() {
        // The per-context Shannon cost ignores the clustering the real code
        // applies afterward, and leaves with few tokens are exactly the ones
        // clustering folds into a neighbor. Require a population and a margin
        // so a leaf only flips when the saving survives that noise.
        let (wp_cost, grad_cost) = (cost(ctx), cost(K_NUM_DC_CONTEXTS + ctx));
        let populated = counts[ctx].iter().sum::<u32>() >= MIN_TOKENS_PER_DC_LEAF;
        *out = populated && grad_cost < wp_cost * DC_PREDICTOR_MARGIN;
    }
    choice
}

/// A DC leaf must own at least this many tokens before its predictor may flip.
const MIN_TOKENS_PER_DC_LEAF: u32 = 256;

/// Required saving before flipping a leaf away from the weighted predictor.
const DC_PREDICTOR_MARGIN: f64 = 0.995;

pub(crate) fn collect_dc_tokens(
    dc_data: &DcGroupData,
    dc_gradient: &DcPredictorChoice,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
) -> Vec<Token> {
    collect_dc_tokens_with_gradient(dc_data, dc_gradient, props, collect_props, false).0
}

fn collect_dc_tokens_with_gradient(
    dc_data: &DcGroupData,
    dc_gradient: &DcPredictorChoice,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
    collect_gradient: bool,
) -> (Vec<Token>, Vec<Token>) {
    let token_count = [1usize, 0, 2]
        .into_iter()
        .map(|c| dc_data.quant_dc.plane(c).as_slice().len())
        .sum();
    let mut tokens = Vec::with_capacity(token_count);
    let mut gradient_tokens = Vec::with_capacity(if collect_gradient { token_count } else { 0 });
    for (mchan, c) in [1usize, 0, 2].into_iter().enumerate() {
        let plane = dc_data.quant_dc.plane(c);
        walk_dc_channel(
            plane.as_slice(),
            plane.xsize(),
            plane.ysize(),
            plane.xsize(),
            mchan,
            dc_gradient,
            props,
            collect_props,
            collect_gradient,
            &mut tokens,
            &mut gradient_tokens,
        );
    }
    (tokens, gradient_tokens)
}

/// Tokenize one DC channel: a `xsize`×`ysize` window of rows `stride` apart.
///
/// Weighted-predictor DC, mirroring libjxl's kWPFixedDC path (enc_modular.cc
/// AddVarDCTDC at speed tiers falcon..squirrel): the guess is the
/// self-correcting WP and the context is the WP-error property (modular
/// property 15) bucketed by the same +-500 cutoffs the gradient tree used,
/// so K_GRADIENT_CONTEXT_LUT applies unchanged (write_context_tree emits the
/// matching tree: identical structure, property 15, Weighted leaves). The WP
/// state machine and border conventions are the bit-faithful lossless-path
/// ones, so encoder residuals match the reference decoder exactly.
///
/// NOTE: this walk is shared by the DC coder and by [`price_dc_plane`], which
/// prices the candidate planes of `choose_chroma_dc_coding`. Keep it the
/// single source of the DC tokenization: a change here reaches both, but a
/// change to how DC tokens are entropy-coded elsewhere (contexts, learned
/// trees, token packing) must be mirrored in `price_dc_plane`, or the
/// squeeze decision prices a coder that no longer exists.
#[allow(clippy::too_many_arguments)]
fn walk_dc_channel<T: Copy + Into<i64>>(
    data: &[T],
    xsize: usize,
    ysize: usize,
    stride: usize,
    mchan: usize,
    dc_gradient: &DcPredictorChoice,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
    collect_gradient: bool,
    tokens: &mut Vec<Token>,
    gradient_tokens: &mut Vec<Token>,
) {
    if xsize == 0 || ysize == 0 {
        return;
    }
    let row = |y: usize| &data[y * stride..y * stride + xsize];
    let mut wp = crate::weighted_predictor::WpState::new(xsize);

    // On the top row every unavailable north-side neighbor collapses to
    // the value on the left.
    let first_row = row(0);
    let mut left = 0i64;
    for (x, &value) in first_row.iter().enumerate() {
        let value: i64 = value.into();
        push_dc_wp_token(
            tokens,
            gradient_tokens,
            collect_gradient,
            &mut wp,
            x,
            0,
            value,
            left,
            left,
            left,
            left,
            left,
            dc_gradient,
            mchan,
            props,
            collect_props,
        );
        left = value;
    }

    let mut row_above = first_row;
    let mut row_above2 = first_row;
    for y in 1..ysize {
        let current_row = row(y);

        // First column: W and NW replicate N.
        let n: i64 = row_above[0].into();
        let ne: i64 = row_above.get(1).copied().unwrap_or(row_above[0]).into();
        push_dc_wp_token(
            tokens,
            gradient_tokens,
            collect_gradient,
            &mut wp,
            0,
            y,
            current_row[0].into(),
            n,
            n,
            ne,
            n,
            row_above2[0].into(),
            dc_gradient,
            mchan,
            props,
            collect_props,
        );

        // Interior columns have every neighbor available.
        let current_pairs = current_row.array_windows::<2>();
        let north_triplets = row_above.array_windows::<3>();
        for (offset, ((current, north), &nn)) in current_pairs
            .zip(north_triplets)
            .zip(row_above2.iter().skip(1))
            .enumerate()
        {
            push_dc_wp_token(
                tokens,
                gradient_tokens,
                collect_gradient,
                &mut wp,
                offset + 1,
                y,
                current[1].into(),
                north[1].into(),
                current[0].into(),
                north[2].into(),
                north[0].into(),
                nn.into(),
                dc_gradient,
                mchan,
                props,
                collect_props,
            );
        }

        // Last column: NE replicates N. Width one was handled above.
        if xsize > 1 {
            let x = xsize - 1;
            let n: i64 = row_above[x].into();
            push_dc_wp_token(
                tokens,
                gradient_tokens,
                collect_gradient,
                &mut wp,
                x,
                y,
                current_row[x].into(),
                n,
                current_row[x - 1].into(),
                n,
                row_above[x - 1].into(),
                row_above2[x].into(),
                dc_gradient,
                mchan,
                props,
                collect_props,
            );
        }

        row_above2 = row_above;
        row_above = current_row;
    }
}

/// Bits to code `levels` — a `width`×`height` window of a DC plane whose rows
/// are `stride` apart — the way the DC coder codes it: the weighted
/// predictor's residuals in their WP-error contexts, priced as per-context
/// order-0 entropy of the hybrid-uint symbols plus their raw bits. This is
/// the price `choose_chroma_dc_coding` compares its candidate planes with.
///
/// NOTE: keep in step with the DC coder (see [`walk_dc_channel`]).
pub(crate) fn price_dc_plane(levels: &[i32], width: usize, height: usize, stride: usize) -> f64 {
    let mut price = DcPlanePrice::default();
    price.add(levels, width, height, stride);
    price.bits()
}

/// Reusable weighted-predictor price for one channel across DC groups. The
/// predictor restarts for every group, while all groups share the histograms,
/// just as they share the final frame's entropy code.
pub(crate) struct DcPlanePrice {
    tokens: Vec<Token>,
    counts: Vec<[u32; crate::entropy::ALPHABET_SIZE]>,
    raw_bits: u64,
}

impl Default for DcPlanePrice {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            counts: vec![[0; crate::entropy::ALPHABET_SIZE]; K_NUM_DC_CONTEXTS],
            raw_bits: 0,
        }
    }
}

impl DcPlanePrice {
    pub(crate) fn clear(&mut self) {
        self.counts.as_flattened_mut().fill(0);
        self.raw_bits = 0;
    }

    pub(crate) fn add<T: Copy + Into<i64>>(
        &mut self,
        levels: &[T],
        width: usize,
        height: usize,
        stride: usize,
    ) {
        self.tokens.clear();
        walk_dc_channel(
            levels,
            width,
            height,
            stride,
            0,
            &DC_PREDICTOR_WEIGHTED,
            &mut Vec::new(),
            false,
            false,
            &mut self.tokens,
            &mut Vec::new(),
        );
        for token in &self.tokens {
            let (sym, nbits, _) = crate::entropy::uint_encode(token.value);
            self.counts[token.context as usize][sym as usize] += 1;
            self.raw_bits += u64::from(nbits);
        }
    }

    /// Count a token from an existing weighted-predictor walk.
    fn add_token(&mut self, token: Token) {
        let (sym, nbits, _) = crate::entropy::uint_encode(token.value);
        self.counts[token.context as usize][sym as usize] += 1;
        self.raw_bits += u64::from(nbits);
    }

    pub(crate) fn bits(&self) -> f64 {
        let mut bits = self.raw_bits as f64;
        for ctx in &self.counts {
            let total: u32 = ctx.iter().sum();
            if total == 0 {
                continue;
            }
            let total = f64::from(total);
            bits += ctx
                .iter()
                .filter(|&&n| n != 0)
                .map(|&n| f64::from(n) * f_log2(total / f64::from(n)))
                .sum::<f64>();
        }
        bits
    }
}

/// AC metadata: ytox/ytob CfL maps (all 0), AC strategy (all 0 = DCT-8x8),
/// quant field residuals (all 0), and EPF (token (0, PackSigned(4)) per block).
///
/// In libjxl-tiny ALL four sub-streams use the same shared dc_code.
/// Same as write_ac_metadata_tokens, but returns the tokens. Mirror of
/// collect_dc_tokens for the AC metadata (YtoX/B, ACS, QF, EPF).
#[inline]
pub(crate) fn ac_metadata_context(left: i32, base: u32) -> u32 {
    base + if left > 11 {
        0
    } else if left > 5 {
        1
    } else if left > 3 {
        2
    } else {
        3
    }
}

pub(crate) fn collect_ac_metadata_tokens(
    dc_data: &DcGroupData,
    props: &mut Vec<crate::dc_tree::DcProp>,
    epf_sharpness: i32,
    collect_props: bool,
) -> Vec<Token> {
    #[inline]
    fn wbin(w: i32) -> crate::dc_tree::DcProp {
        (512 + w).clamp(0, 1023) as crate::dc_tree::DcProp
    }
    let xsize_blocks = dc_data.ac_strategy.xsize();
    let ysize_blocks = dc_data.ac_strategy.ysize();
    let xtiles = dc_data.ytox_map.xsize();
    let ytiles = dc_data.ytox_map.ysize();
    let nblocks = xsize_blocks * ysize_blocks;
    let num_first_blocks = dc_data.ac_strategy.count_first_blocks();
    let cfl_tokens = dc_data.ytox_map.as_slice().len() + dc_data.ytob_map.as_slice().len();
    let mut tokens = Vec::with_capacity(cfl_tokens + 2 * num_first_blocks + nblocks);

    // (a) YtoX and YtoB tokens with gradient prediction.
    for (c, cfl_map) in [&dc_data.ytox_map, &dc_data.ytob_map]
        .into_iter()
        .enumerate()
    {
        debug_assert_eq!((cfl_map.xsize(), cfl_map.ysize()), (xtiles, ytiles));
        if xtiles == 0 || ytiles == 0 {
            continue;
        }

        let ctx_id = 2u32 - c as u32;
        let mut rows = cfl_map.as_slice().chunks_exact(xtiles);

        // The top row predicts from the value on the left.
        let first_row = rows.next().unwrap();
        let mut left = 0i32;
        for &here in first_row {
            let here = here as i32;
            if collect_props {
                props.push(wbin(left));
            }
            tokens.push(Token::new(ctx_id, pack_signed(here - left)));
            left = here;
        }

        let mut row_above = first_row;
        for row in rows {
            // First column replicates the value above for W and NW.
            if collect_props {
                props.push(wbin(row_above[0] as i32));
            }
            tokens.push(Token::new(
                ctx_id,
                pack_signed(row[0] as i32 - row_above[0] as i32),
            ));

            for (current, above) in row.array_windows::<2>().zip(row_above.array_windows::<2>()) {
                let prediction =
                    clamped_gradient(above[1] as i32, current[0] as i32, above[0] as i32);
                if collect_props {
                    props.push(wbin(current[0] as i32));
                }
                tokens.push(Token::new(
                    ctx_id,
                    pack_signed(current[1] as i32 - prediction),
                ));
            }
            row_above = row;
        }
    }

    // (b) AC strategy and (c) QF residual tokens. Their substreams are
    // contiguous, so reserve both ranges and fill them in one block scan.
    let strategy_base = tokens.len();
    let qf_base = strategy_base + num_first_blocks;
    tokens.resize(strategy_base + 2 * num_first_blocks, Token::new(0, 0));
    if collect_props {
        props.resize(strategy_base + 2 * num_first_blocks, 0);
    }
    let mut strategy_left = 0i32;
    let mut qf_left = if nblocks == 0 {
        0
    } else {
        dc_data.ac_strategy.strategy_code(0, 0) as i32
    };
    let mut first_idx = 0usize;
    for (y, row_qf) in (0..ysize_blocks)
        .map(|y| dc_data.raw_quant_field.row(y))
        .enumerate()
    {
        for (x, &qf) in row_qf[..xsize_blocks].iter().enumerate() {
            if !dc_data.ac_strategy.is_first_block(x, y) {
                continue;
            }

            let strategy = dc_data.ac_strategy.strategy_code(x, y) as i32;
            if collect_props {
                props[strategy_base + first_idx] = wbin(strategy_left);
            }
            tokens[strategy_base + first_idx] =
                Token::new(ac_metadata_context(strategy_left, 7), pack_signed(strategy));
            strategy_left = strategy;

            let qf = qf as i32 - 1;
            if collect_props {
                props[qf_base + first_idx] = wbin(qf_left);
            }
            tokens[qf_base + first_idx] =
                Token::new(ac_metadata_context(qf_left, 3), pack_signed(qf - qf_left));
            qf_left = qf;
            first_idx += 1;
        }
    }
    debug_assert_eq!(first_idx, num_first_blocks);

    // (d) EPF tokens (constant stream; refinement can never split it).
    if collect_props {
        props.resize(props.len() + nblocks, wbin(epf_sharpness));
    }
    tokens.resize(
        tokens.len() + nblocks,
        Token::new(0, pack_signed(epf_sharpness)),
    );
    tokens
}

/// Fraction of exactly constant 8x8 blocks above which a frame is screen content.
const SCREEN_CONTENT_CONSTANT_BLOCKS: f32 = 0.15;
const SCREEN_CONTENT_EPF_MAX_DISTANCE: f32 = 5.0;

/// Fraction of whole 8x8 blocks whose three XYB planes are each constant.
/// Identical input pixels map to identical XYB values, so the test is exact.
fn constant_block_fraction(xyb: &Image3F) -> f32 {
    let (bw, bh) = (xyb.xsize() / 8, xyb.ysize() / 8);
    if bw == 0 || bh == 0 {
        return 0.0;
    }
    let mut constant = 0usize;
    for by in 0..bh {
        for bx in 0..bw {
            let x0 = bx * 8;
            let is_constant = (0..3).all(|c| {
                let v = xyb.plane_row(c, by * 8)[x0];
                (0..8).all(|iy| {
                    xyb.plane_row(c, by * 8 + iy)[x0..x0 + 8]
                        .iter()
                        .all(|&p| p == v)
                })
            });
            constant += usize::from(is_constant);
        }
    }
    constant as f32 / (bw * bh) as f32
}

pub(crate) fn epf_sharpness_id(distance: f32, screen_content: bool) -> i32 {
    let t = dcepf_vlq_t(distance);
    if t >= 0.75 {
        return 1;
    } else if t >= 0.5 {
        return 2;
    } else if t >= 0.25 {
        return 3;
    }
    // Hard edges beside flat fills ring visibly; up to d=5 the strongest
    // filter wins both SSIMULACRA2 and butteraugli there at identical bytes.
    if distance < 1.4 || (screen_content && distance < SCREEN_CONTENT_EPF_MAX_DISTANCE) {
        7
    } else if distance < 1.75 {
        6
    } else if distance < 3.5 {
        5
    } else {
        4
    }
}

/// Real prefix-code bit cost of a DC group's AC-metadata token stream under a
/// freshly optimized entropy code. Used by the sub-8x8 activation gate to weigh
/// the exact selected set's meta-stream cost against its RD benefit.
fn meta_entropy_cost(dc_data: &DcGroupData, scratch: &mut CoderScratch, distance: f32) -> u64 {
    let toks = collect_ac_metadata_tokens(
        dc_data,
        &mut Vec::new(),
        epf_sharpness_id(distance, false),
        false,
    );
    let code_owned = optimize_entropy_code(&toks, K_NUM_DC_CONTEXTS, &mut scratch.huffman_pool);
    let code = code_owned.as_ref();
    let mut bits = 0u64;
    for t in &toks {
        let (tok, nbits, _b) = crate::entropy::uint_encode(t.value);
        let pc = &code.prefix_codes[code.context_map[t.context as usize] as usize];
        bits += if pc.single_symbol {
            nbits as u64
        } else {
            pc.depths[tok as usize] as u64 + nbits as u64
        };
    }
    bits
}

/// Build and emit the context tree.
pub(crate) fn write_context_tree(
    num_dc_groups: usize,
    dc_gradient: &DcPredictorChoice,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    writer: &mut BitWriter,
) {
    let tokens = context_tree_tokens(num_dc_groups, dc_gradient);
    write_tree_tokens(&tokens, huffman_pool, writer);
}

/// The static context tree's tokens.
fn context_tree_tokens(num_dc_groups: usize, dc_gradient: &DcPredictorChoice) -> Vec<Token> {
    // Build tokens with the patched value at index 1.
    let mut tokens: Vec<Token> = Vec::with_capacity(K_CONTEXT_TREE_TOKENS.len());
    for (i, &(ctx, val)) in K_CONTEXT_TREE_TOKENS.iter().enumerate() {
        let v = if i == 1 {
            pack_signed(1 + num_dc_groups as i32)
        } else {
            val
        };
        tokens.push(Token::new(ctx, v));
    }
    // Retarget the DC-image region (leaves 11..=44) at the Weighted predictor,
    // mirroring libjxl's kWPFixedDC tree (enc_modular.cc / enc_ma PredefinedTree):
    // identical 33 cutoffs, but splitting on the WP-error property (15) instead
    // of the gradient property (9), with Weighted(6) leaves instead of
    // Gradient(5). Parse-aware transform of the static blob: prop-9 splits only
    // occur in the DC region; Gradient leaves also exist at meta contexts 9/10
    // (CfL), so predictor tokens are rewritten only for leaf indices >= 11.
    {
        let mut i = 0usize;
        let mut leaf_idx = 0usize;
        while i < tokens.len() {
            debug_assert_eq!(tokens[i].context, 1);
            if tokens[i].value == 0 {
                if leaf_idx >= 11 && tokens[i + 1].value == 5 && !dc_gradient[leaf_idx] {
                    tokens[i + 1] = Token::new(2, 6); // Gradient -> Weighted
                }
                leaf_idx += 1;
                i += 5;
            } else {
                if tokens[i].value == 9 + 1 {
                    tokens[i] = Token::new(1, 15 + 1); // gradient prop -> WP prop
                }
                i += 2; // split: PROPERTY, SPLITVAL
            }
        }
    }
    tokens
}

/// Serialize a context tree's token stream (its own entropy code included).
pub(crate) fn write_tree_tokens(
    tokens: &[Token],
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    writer: &mut BitWriter,
) {
    // OptimizeEntropyCode clusters the K_NUM_TREE_CONTEXTS=6 contexts.
    let code = optimize_entropy_code(tokens, K_NUM_TREE_CONTEXTS, huffman_pool);
    let code_ref = code.as_ref();

    writer.write(1, 1); // not an empty tree
    writer.write(1, 0); // no lz77
    write_entropy_code(&code_ref, huffman_pool, writer);
    for t in tokens {
        write_token(*t, &code_ref, writer);
    }
}

/// Which context tree the DC bundle is written with.
pub(crate) enum DcTreeChoice {
    /// The static blob with per-leaf predictor flips.
    Static(DcPredictorChoice),
    /// A per-image learned tree, already serialized to tokens.
    Learned(Vec<Token>),
}

impl DcTreeChoice {
    fn tokens(&self, num_dc_groups: usize) -> Vec<Token> {
        match self {
            Self::Static(grad) => context_tree_tokens(num_dc_groups, grad),
            Self::Learned(tokens) => tokens.clone(),
        }
    }
}

fn write_frame_dimension(value: usize, w: &mut BitWriter) {
    let value = value as u64;
    if value < 256 {
        w.write(2, 0);
        w.write(8, value);
    } else if value < 2304 {
        w.write(2, 1);
        w.write(11, value - 256);
    } else if value < 18688 {
        w.write(2, 2);
        w.write(14, value - 2304);
    } else {
        w.write(2, 3);
        w.write(30, value - 18688);
    }
}

fn write_frame_header_kind(
    x_qm_scale: u32,
    b_qm_scale: u32,
    epf_iters: u32,
    epf_pass0_scale: Option<f32>,
    epf_quant_mul: Option<f32>,
    gab_enabled: bool,
    has_alpha: bool,
    coeff_shifts: &[u32],
    kind: VarDctFrameKind<'_>,
    has_splines: bool,
    has_noise: bool,
    skip_dc_smoothing: bool,
    use_dc_frame: bool,
    coding: crate::coding::CodingTransform,
    epf_channel_scale: Option<[f32; 3]>,
    role: FrameRole,
    w: &mut BitWriter,
) {
    match kind {
        VarDctFrameKind::Regular => write_frame_header(
            x_qm_scale,
            b_qm_scale,
            epf_iters,
            epf_pass0_scale,
            epf_quant_mul,
            gab_enabled,
            has_alpha,
            coeff_shifts,
            false,
            has_splines,
            has_noise,
            skip_dc_smoothing,
            use_dc_frame,
            coding,
            epf_channel_scale,
            role,
            w,
        ),
        VarDctFrameKind::Patched(_) => write_frame_header(
            x_qm_scale,
            b_qm_scale,
            epf_iters,
            epf_pass0_scale,
            epf_quant_mul,
            gab_enabled,
            has_alpha,
            coeff_shifts,
            true,
            has_splines,
            has_noise,
            skip_dc_smoothing,
            use_dc_frame,
            coding,
            epf_channel_scale,
            role,
            w,
        ),
        VarDctFrameKind::ReferenceOnly { width, height } => {
            debug_assert!(coding.is_xyb(), "patch atlases are XYB-only");
            w.write(1, 0); // not all default
            w.write(2, 0b10); // reference-only frame
            w.write(1, 0); // VarDCT
            w.write(2, 0); // flags = 0
            w.write(2, 0); // upsampling = 1
            if has_alpha {
                w.write(2, 0); // extra-channel upsampling = 1
            }
            w.write(3, x_qm_scale as u64);
            w.write(3, b_qm_scale as u64); // b_qm_scale
            // Reference-only frames omit Passes and use the implicit one pass.
            w.write(1, 1); // custom size
            write_frame_dimension(width, w);
            write_frame_dimension(height, w);
            w.write(2, PATCH_REF_ID as u64);
            w.write(1, 1); // save_before_color_transform
            w.write(2, 0); // empty name
            write_loop_filter(
                epf_iters,
                epf_pass0_scale,
                epf_quant_mul,
                gab_enabled,
                None,
                w,
            );
            w.write(2, 0); // no frame-header extensions
        }
    }
}

fn write_loop_filter(
    epf_iters: u32,
    epf_pass0_scale: Option<f32>,
    epf_quant_mul: Option<f32>,
    gab_enabled: bool,
    epf_channel_scale: Option<[f32; 3]>,
    w: &mut BitWriter,
) {
    if epf_iters == 2 && gab_enabled && epf_channel_scale.is_none() && epf_quant_mul.is_none() {
        w.write(1, 1); // default loop filter (gab=1, epf=2)
    } else {
        w.write(1, 0); // not default
        if gab_enabled {
            w.write(1, 1); // gaborish enabled
            w.write(1, 0); // gab_custom = false (use defaults)
        } else {
            w.write(1, 0); // no gaborish
        }
        w.write(2, epf_iters as u64);
        if epf_iters > 0 {
            w.write(1, 0); // default epf sharpness
            match epf_channel_scale {
                None => w.write(1, 0), // default epf weights
                Some(scale) => {
                    w.write(1, 1); // custom epf weights
                    // Channel scales, then the spec pass-1/2 zero-flush levels.
                    for v in [scale[0], scale[1], scale[2], 0.45, 0.6] {
                        w.write(16, u64::from(crate::util::f32_to_f16_bits(v)));
                    }
                }
            }
            // epf_sigma_custom: [quant_mul, pass0_scale, pass2_scale,
            // border_sad_mul], spec defaults 0.46, 0.9, 6.5, 2/3. The fitter
            // varies quant_mul and pass0; the other two stay at spec.
            if let Some(s) = epf_pass0_scale.or(epf_quant_mul.map(|_| 0.9)) {
                let c = [epf_quant_mul.unwrap_or(0.46), s, 6.5, 2.0 / 3.0];
                w.write(1, 1); // custom epf sigma
                for x in c {
                    w.write(16, crate::util::f32_to_f16_bits(x) as u64);
                }
            } else {
                w.write(1, 0); // default epf sigma
            }
        }
        w.write(2, 0); // no loop filter extensions
    }
}

fn write_frame_header(
    x_qm_scale: u32,
    b_qm_scale: u32,
    epf_iters: u32,
    epf_pass0_scale: Option<f32>,
    epf_quant_mul: Option<f32>,
    gab_enabled: bool,
    has_alpha: bool,
    coeff_shifts: &[u32],
    has_patches: bool,
    has_splines: bool,
    has_noise: bool,
    skip_dc_smoothing: bool,
    use_dc_frame: bool,
    coding: crate::coding::CodingTransform,
    epf_channel_scale: Option<[f32; 3]>,
    role: FrameRole,
    w: &mut BitWriter,
) {
    w.write(1, 0); // not all default
    w.write(2, 0); // regular frame
    w.write(1, 0); // vardct
    // Optional flags: kNoise (1), kPatches (2), kSplines (16), kUseDcFrame
    // (32) and kSkipAdaptiveDCSmoothing (128), chosen per frame by
    // `skip_dc_smoothing`. A DC frame implies no smoothing.
    let flags = if has_noise {
        crate::noise::FLAG_NOISE
    } else {
        0
    } + if has_patches { 2u64 } else { 0 }
        + if has_splines { 16 } else { 0 }
        + if use_dc_frame { 32 } else { 0 }
        + if skip_dc_smoothing && !use_dc_frame {
            128
        } else {
            0
        };
    match flags {
        0 => w.write(2, 0),
        1..=16 => {
            w.write(2, 1); // U64 selector for values 1..=16
            w.write(4, flags - 1);
        }
        _ => {
            w.write(2, 2); // U64 selector for values 17..=272
            w.write(8, flags - 17);
        }
    }
    if !coding.is_xyb() {
        let ycbcr = coding == crate::coding::CodingTransform::YCbCr;
        w.write(1, u64::from(ycbcr)); // do_YCbCr
        if ycbcr {
            for _ in 0..3 {
                w.write(2, 0); // chroma subsampling 4:4:4
            }
        }
    }
    // Absent with kUseDcFrame.
    if !use_dc_frame {
        w.write(2, 0); // no upsampling

        // Per-extra-channel upsampling. Same u2S(1,2,4,8) code, default 1:
        // selector "00" gives 1, written once per extra channel.
        if has_alpha {
            w.write(2, 0); // ec_upsampling[0] = 1
        }
    }

    // The quant scales are only serialized for XYB frames.
    if coding.is_xyb() {
        w.write(3, x_qm_scale as u64);
        w.write(3, b_qm_scale as u64); // b_qm_scale
    }
    // Passes bundle (jxl-frame header.rs:127-132):
    //   num_passes: U32(1,2,3,4+u(3))   default 1
    //   if num_passes != 1:
    //     num_ds:    U32(0,1,2,3+u(1))  -> 0 (no downsampling)
    //     shift:     Vec[u(2)] len (num_passes-1)
    //     downsample/last_pass: Vec len num_ds (empty here)
    // For VarDCT AC the per-pass coeff_shift used by the decoder is
    // passes.shift[pass]; the last pass implicitly has shift 0, so we only emit
    // shifts for passes 0..num_passes-1 and require coeff_shifts.last()==0.
    let num_passes = coeff_shifts.len();
    debug_assert!((1..=11).contains(&num_passes), "num_passes out of range");
    debug_assert!(num_passes == 1 || coeff_shifts[num_passes - 1] == 0);
    if num_passes == 1 {
        w.write(2, 0); // num_passes = 1 (U32 selector 0)
    } else {
        // U32(1,2,3,4+u(3)): selectors 0->1, 1->2, 2->3, 3->4+u(3).
        match num_passes {
            2 => w.write(2, 1),
            3 => w.write(2, 2),
            n => {
                w.write(2, 3);
                w.write(3, (n - 4) as u64);
            }
        }
        w.write(2, 0); // num_ds = 0 (U32 selector 0)
        for &s in &coeff_shifts[..num_passes - 1] {
            w.write(2, s as u64); // shift[p] = coeff_shift of pass p (u(2))
        }
    }
    w.write(1, 0); // no custom frame size or origin

    // Color-channel BlendingInfo: mode=Replace, full_frame=true means
    // source/alpha_channel/clamp are all omitted. The two zero bits select
    // BlendingMode::Replace.
    w.write(2, 0); // color blend mode = Replace

    // Per-extra-channel BlendingInfo. With num_extra_channels=1, BlendingInfo
    // for the alpha channel has:
    //   mode = Replace (2 bits = 00)
    //   alpha_channel: NOT written (mode != Blend/AWA)
    //   clamp: NOT written (mode != Blend/AWA/Mul)
    //   source: NOT written (full_frame && Replace)
    if has_alpha {
        w.write(2, 0); // ec_blending_info[0].mode = Replace
    }

    match role {
        FrameRole::Last => w.write(1, 1), // last frame
        FrameRole::SavedFirst => {
            w.write(1, 0); // not the last frame
            w.write(2, 0); // save_as_reference = 0
            w.write(1, 0); // save_before_color_transform = false
        }
    }
    w.write(2, 0); // no name
    write_loop_filter(
        epf_iters,
        epf_pass0_scale,
        epf_quant_mul,
        gab_enabled,
        epf_channel_scale,
        w,
    );
    w.write(2, 0); // no frame header extensions
}

pub(crate) fn write_quant_scales(global_scale: i32, quant_dc: i32, w: &mut BitWriter) {
    if global_scale < 2049 {
        w.write(2, 0);
        w.write(11, (global_scale - 1) as u64);
    } else if global_scale < 4097 {
        w.write(2, 1);
        w.write(11, (global_scale - 2049) as u64);
    } else if global_scale < 8193 {
        w.write(2, 2);
        w.write(12, (global_scale - 4097) as u64);
    } else {
        w.write(2, 3);
        w.write(16, (global_scale - 8193) as u64);
    }
    if quant_dc == 16 {
        w.write(2, 0);
    } else if quant_dc < 33 {
        w.write(2, 1);
        w.write(5, (quant_dc - 1) as u64);
    } else if quant_dc < 257 {
        w.write(2, 2);
        w.write(8, (quant_dc - 1) as u64);
    } else {
        w.write(2, 3);
        w.write(16, (quant_dc - 1) as u64);
    }
}

/// Serialize the per-image BlockCtxMap: no dc thresholds, the optional
/// quant-field threshold, and the ctx_map (qf inner dimension when present).
fn write_block_ctx_map(
    plan: &crate::ac_context::AcCtxPlan,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    w.write(1, 0); // non-default BlockCtxMap
    w.write(4, 0); // dc thresholds, channel 0
    w.write(4, 0); // dc thresholds, channel 1
    w.write(4, 0); // dc thresholds, channel 2
    match plan.qf_threshold {
        None => w.write(4, 0),
        Some(t) => {
            w.write(4, 1);
            // kQFThresholdDist: U32(Bits(2), BitsOffset(3,4), BitsOffset(5,12),
            // BitsOffset(8,44)) over t - 1.
            let v = t - 1;
            if v < 4 {
                w.write(2, 0);
                w.write(2, u64::from(v));
            } else if v < 12 {
                w.write(2, 1);
                w.write(3, u64::from(v - 4));
            } else if v < 44 {
                w.write(2, 2);
                w.write(5, u64::from(v - 12));
            } else {
                w.write(2, 3);
                w.write(8, u64::from(v - 44));
            }
        }
    }
    let entries = plan.ctx_map_entries();
    let empty_codes: [crate::entropy::PrefixCode; 0] = [];
    let empty_configs: [crate::entropy::HybridUintConfig; 0] = [];
    let empty_histograms = [];
    let empty_syms: [Vec<crate::entropy::AnsEncSymbolInfo>; 0] = [];
    let empty_reverse_maps: [u16; 0] = [];
    let cm_entropy = EntropyCode {
        context_map: &entries,
        num_contexts: entries.len(),
        prefix_codes: &empty_codes,
        hybrid_uint_configs: &empty_configs,
        num_prefix_codes: 0,
        orig_context_map: None,
        orig_num_contexts: 0,
        use_prefix_code: true,
        ans_histograms: &empty_histograms,
        ans_symbols: &empty_syms,
        ans_reverse_maps: &empty_reverse_maps,
    };
    crate::entropy::write_context_map(&cm_entropy, &mut scratch.huffman_pool, w);
}

/// Modular data of the extra channel, coded ahead of the frame sections:
/// the LfGlobal part, one part per DC group and one per AC group.
struct ExtraSections {
    global: BitWriter,
    lf: Vec<BitWriter>,
    ac: Vec<BitWriter>,
}

impl ExtraSections {
    fn bits(&self) -> usize {
        self.global.bits_written()
            + self.lf.iter().map(BitWriter::bits_written).sum::<usize>()
            + self.ac.iter().map(BitWriter::bits_written).sum::<usize>()
    }
}

/// A lossless extra channel of few distinct values, coded through a Palette
/// transform: the values in index order and one index per sample.
struct ExtraPalette {
    entries: Vec<i32>,
    indices: AlphaPlane,
}

/// Largest palette an extra channel is coded through.
const EXTRA_PALETTE_MAX_COLORS: usize = 256;
/// The global tree has not cut a plane's fixed lossless coding below a
/// quarter; past that the squeezed coding is out of its reach.
const GLOBAL_TREE_MAX_SAVING: usize = 4;

/// The palette of an integer extra channel whose distinct values number at
/// most `EXTRA_PALETTE_MAX_COLORS` and at most half its span, so that the
/// index image is narrower than the values.
fn extra_palette(plane: &AlphaPlane) -> Option<ExtraPalette> {
    let range = match plane {
        AlphaPlane::U8(_) => 1usize << 8,
        AlphaPlane::U16 { bits, .. } => 1usize << bits,
        AlphaPlane::F32(_) => return None,
    };
    let mut seen = vec![false; range];
    for i in 0..plane.len() {
        let v = plane.get_i32(i);
        if v < 0 || v as usize >= range {
            return None;
        }
        seen[v as usize] = true;
    }
    let entries: Vec<i32> = (0..range as i32).filter(|&v| seen[v as usize]).collect();
    let (Some(&lo), Some(&hi)) = (entries.first(), entries.last()) else {
        return None;
    };
    if entries.len() < 2
        || entries.len() > EXTRA_PALETTE_MAX_COLORS
        || 2 * entries.len() as i64 > i64::from(hi - lo) + 1
    {
        return None;
    }
    let mut index_of = vec![0u8; range];
    for (index, &v) in entries.iter().enumerate() {
        index_of[v as usize] = index as u8;
    }
    let indices = (0..plane.len())
        .map(|i| index_of[plane.get_i32(i) as usize])
        .collect();
    Some(ExtraPalette {
        entries,
        indices: AlphaPlane::U8(indices),
    })
}

/// Code the extra channel losslessly — through a Palette transform when it
/// has few distinct values, and through the frame's global tree where that
/// beats the fixed local trees — and, when the context asks for a lossy one,
/// also squeezed and quantized; keep the smaller. The lossless coding wins
/// wherever it is no larger, so solid and hard-edged planes stay exact. A
/// winning global-tree coding moves the frame's DC streams onto the merged
/// tree through `streams`.
fn prepare_extra_sections(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    last_pass: usize,
    mut streams: extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
) -> ExtraSections {
    let single = dim.xsize <= K_GROUP_DIM && dim.ysize <= K_GROUP_DIM;
    let lossless_plane = ctx.extra_step == 1 && !plane.is_constant();
    let palette = lossless_plane.then(|| extra_palette(plane)).flatten();
    let mut lossless = ExtraSections {
        global: BitWriter::new(),
        lf: (0..dc_group_coords.len())
            .map(|_| BitWriter::new())
            .collect(),
        ac: Vec::new(),
    };
    // A multi-group palette-coded channel codes its palette locally in the
    // global stream and its indices per AC group. A single-group one keeps
    // the values' section as the local alternative of its global-tree
    // coding, which carries the palette.
    let coded = palette.as_ref().filter(|_| !single);
    match coded {
        Some(p) => crate::modular::write_global_palette_alpha_modular(
            &p.entries,
            plane.bits(),
            scratch,
            &mut lossless.global,
        ),
        None => crate::modular::write_global_alpha_modular(
            plane,
            dim.xsize,
            dim.ysize,
            ctx.extra_step,
            !streams.code.use_prefix_code,
            scratch,
            &mut lossless.global,
        ),
    }
    let ac_plane = coded.map_or(plane, |p| &p.indices);
    lossless.ac = ctx
        .thread_pool
        .steal_map(scratch, dim.num_groups, |group_id, scratch| {
            let x0 = (group_id % dim.xsize_groups) * K_GROUP_DIM;
            let y0 = (group_id / dim.xsize_groups) * K_GROUP_DIM;
            let mut w = BitWriter::new();
            crate::modular::write_ac_group_alpha(
                ac_plane,
                dim.xsize,
                dim.ysize,
                x0,
                y0,
                K_GROUP_DIM.min(dim.xsize - x0),
                K_GROUP_DIM.min(dim.ysize - y0),
                ctx.extra_step,
                scratch,
                &mut w,
            );
            w
        });
    let lossy = (ctx.extra_squeeze_distance > 0.0 && !plane.is_constant())
        .then(|| {
            crate::lossless::ExtraSqueeze::new(
                plane,
                dim.xsize,
                dim.ysize,
                ctx.extra_squeeze_distance,
                ctx.extra_learned_trees,
            )
        })
        .flatten()
        .map(|squeezed| {
            let mut lossy = ExtraSections {
                global: BitWriter::new(),
                lf: Vec::new(),
                ac: Vec::new(),
            };
            squeezed.write_global(scratch, &mut lossy.global);
            lossy.lf = ctx
                .thread_pool
                .steal_map(scratch, dc_group_coords.len(), |i, scratch| {
                    let (gx, gy) = dc_group_coords[i];
                    let mut w = BitWriter::new();
                    squeezed.write_lf_group(gx, gy, scratch, &mut w);
                    w
                });
            lossy.ac = ctx
                .thread_pool
                .steal_map(scratch, dim.num_groups, |group_id, scratch| {
                    let mut w = BitWriter::new();
                    squeezed.write_ac_group(
                        group_id % dim.xsize_groups,
                        group_id / dim.xsize_groups,
                        scratch,
                        &mut w,
                    );
                    w
                });
            lossy
        });
    let global_tree = (lossless_plane
        && ctx.extra_learned_trees
        && lossy
            .as_ref()
            .is_none_or(|lossy| lossless.bits() < GLOBAL_TREE_MAX_SAVING * lossy.bits()))
    .then(|| {
        extra_tree::plan_global_extra_tree(
            ctx,
            plane,
            palette.as_ref(),
            dim,
            last_pass,
            &streams,
            &lossless,
            scratch,
        )
    })
    .flatten();
    let lossless_bits = lossless.bits() - global_tree.as_ref().map_or(0, |t| t.saving());
    match lossy {
        Some(lossy) if lossy.bits() < lossless_bits => lossy,
        _ => {
            if let Some(tree) = global_tree {
                tree.apply(&mut streams, &mut lossless);
            }
            lossless
        }
    }
}

fn write_dc_global(
    distp: &DistanceParams,
    num_dc_groups: usize,
    ac_plan: &crate::ac_context::AcCtxPlan,
    dc_tree: &DcTreeChoice,
    dc_code: &EntropyCode,
    ytox_dc: i32,
    ytob_dc: i32,
    cfl: crate::color_correlation::CflFrame,
    extra: Option<&ExtraSections>,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    if let Some(steps) = crate::color_correlation::dc_step_wire(distp.dc_step) {
        w.write(1, 0);
        for step in steps {
            w.write(16, u64::from(step));
        }
    } else {
        w.write(1, 1); // default dequant DC
    }
    write_quant_scales(distp.global_scale, distp.quant_dc, w);
    write_block_ctx_map(ac_plan, scratch, w);

    // ColorCorrelationParams. The all-default bundle pins the DC plane to the
    // XYB base correlations (X: 0, B: 1); either searched DC slope needs the
    // explicit form, which costs COLOR_CORRELATION_HEADER_BITS more.
    {
        let factor = cfl.color_factor as u32;
        if ytox_dc == 0 && ytob_dc == 0 && cfl == crate::color_correlation::CflFrame::XYB {
            w.write(1, 1); // all_default
        } else {
            w.write(1, 0); // not all-default
            // color_factor: U32(Val(84), Val(256), BitsOffset(8, 2), BitsOffset(16, 258)).
            if factor == 84 {
                w.write(2, 0);
            } else {
                w.write(2, 2);
                w.write(8, (factor - 2) as u64);
            }
            for base in [cfl.base_x, cfl.base_b] {
                w.write(16, u64::from(crate::util::f32_to_f16_bits(base)));
            }
            w.write(8, (ytox_dc + 128) as u64); // ytox_dc, offset by 128
            w.write(8, (ytob_dc + 128) as u64); // ytob_dc, offset by 128
        }
    }

    // Global tree.
    // write_context_tree emits "have_tree=1 + Histograms::decode (tree's own entropy code)
    // + tree tokens". The TREE'S PIXEL HISTOGRAMS are then written as the next two
    // bits + entropy code (DC entropy code, since it's the global tree used for DC).
    match dc_tree {
        DcTreeChoice::Static(grad) => {
            write_context_tree(num_dc_groups, grad, &mut scratch.huffman_pool, w);
        }
        DcTreeChoice::Learned(tokens) => {
            write_tree_tokens(tokens, &mut scratch.huffman_pool, w);
        }
    }
    w.write(1, 0); // no lz77 (for the global tree's pixel histograms = dc_code)

    // Then the static DC entropy code: this is the global tree's pixel histograms.
    write_entropy_code(dc_code, &mut scratch.huffman_pool, w);

    // FullModularImage::read happens HERE in the decoder. If we declared an alpha
    // extra channel, write its GroupHeader + local tree + pixel data now.
    if let Some(extra) = extra {
        w.append(&extra.global);
    }
}

/// Slots whose transform actually appears in the frame: the nine
/// `quant_table_slot_of` slots, then DCT2X2, DCT4X4 and AFV, then DCT32X8/8X32.
fn used_quant_table_slots(dc_datas: &[DcGroupData]) -> [bool; 13] {
    use crate::dc_group_data::{
        STRATEGY_AFV0, STRATEGY_AFV1, STRATEGY_AFV2, STRATEGY_AFV3, STRATEGY_DCT2X2,
        STRATEGY_DCT4X4,
    };
    let mut used = [false; 13];
    for dc in dc_datas {
        for (_, _, strategy) in dc.ac_strategy.iter_first_blocks() {
            if let Some(slot) = quant_table_slot_of(strategy) {
                used[slot] = true;
            }
            match strategy {
                STRATEGY_DCT2X2 => used[9] = true,
                STRATEGY_DCT4X4 => used[10] = true,
                STRATEGY_AFV0 | STRATEGY_AFV1 | STRATEGY_AFV2 | STRATEGY_AFV3 => used[11] = true,
                _ => {}
            }
        }
    }
    used
}

fn write_dequant_matrices(
    matrices: &crate::quant_weights::DequantMatrices,
    used: &[bool; 13],
    w: &mut BitWriter,
) {
    use crate::util::f32_to_f16_bits;
    // A table only earns its header if the frame actually uses the transform.
    let table = |slot: usize| -> Option<&crate::quant_weights::BandOverride> {
        used[slot]
            .then(|| matrices.custom_tables[slot].as_ref())
            .flatten()
    };
    let identity = used[8].then_some(matrices.identity_weights).flatten();
    let fine = matrices.fine_tables.as_deref();
    let dct2 = fine.filter(|_| used[9]).map(|f| f.dct2);
    let dct4x4 = fine.filter(|_| used[10]).map(|f| f.dct4x4);
    let afv = fine.filter(|_| used[11]).map(|f| f.afv);
    // DCT32X8 / DCT8X32 always carry their pair-parity table.
    let t8x32 = used[12].then(|| matrices.band_8x32());
    if (0..8).all(|slot| table(slot).is_none())
        && t8x32.is_none()
        && identity.is_none()
        && dct2.is_none()
        && dct4x4.is_none()
        && afv.is_none()
    {
        w.write(1, 1); // all_default
        return;
    }
    let write_dct_params = |bands: &[[f32; 16]; 3], num_bands: usize, w: &mut BitWriter| {
        w.write(4, num_bands as u64 - 1);
        for row in bands {
            for (i, &v) in row[..num_bands].iter().enumerate() {
                let wire = if i == 0 { v / 64.0 } else { v };
                w.write(16, u64::from(f32_to_f16_bits(wire)));
            }
        }
    };
    let pad4 = |rows: &[[f32; 4]; 3]| -> [[f32; 16]; 3] {
        std::array::from_fn(|c| std::array::from_fn(|i| if i < 4 { rows[c][i] } else { 0.0 }))
    };
    const K_QUANT_MODE_ID: u64 = 1;
    const K_QUANT_MODE_DCT2: u64 = 2;
    const K_QUANT_MODE_DCT4: u64 = 3;
    const K_QUANT_MODE_AFV: u64 = 5;
    const K_NUM_QUANT_TABLES: usize = 17;
    const K_QUANT_MODE_LIBRARY: u64 = 0;
    // `kQuantModeDCT4X8` carries three F16 `dct4x8multipliers` ahead of the
    // shared DctQuantWeightParams payload; the library table uses 1.0 for all
    // three, and jixel's 4x8 matrix is built without them, so identity is
    // preserved by writing 1.0.
    const K_QUANT_MODE_DCT4X8: u64 = 4;
    const K_QUANT_MODE_DCT: u64 = 6;
    const TABLE_DCT4X8: usize = 9;
    w.write(1, 0); // all_default = false
    for idx in 0..K_NUM_QUANT_TABLES {
        let bands = match idx {
            0 => table(0),            // DCT8
            4 => table(1),            // DCT16X16
            5 => table(2),            // DCT32X32
            6 => table(3),            // DCT8X16 (= DCT16X8)
            8 => table(4),            // DCT16X32 (= DCT32X16)
            TABLE_DCT4X8 => table(5), // DCT4X8 (= DCT8X4)
            11 => table(7),           // DCT64X64
            12 => table(6),           // DCT32X64 (= DCT64X32)
            _ => None,
        };
        if idx == 1
            && let Some(id) = identity
        {
            w.write(3, K_QUANT_MODE_ID);
            for row in &id {
                for &v in row {
                    w.write(16, u64::from(f32_to_f16_bits(v / 64.0)));
                }
            }
            continue;
        }
        if idx == 2
            && let Some(dw) = dct2
        {
            w.write(3, K_QUANT_MODE_DCT2);
            for row in &dw {
                for &v in row {
                    w.write(16, u64::from(f32_to_f16_bits(v / 64.0)));
                }
            }
            continue;
        }
        if idx == 3
            && let Some(b4) = dct4x4
        {
            w.write(3, K_QUANT_MODE_DCT4);
            for _ in 0..3 {
                w.write(16, 0x3C00); // dct4multipliers[c][0] = 1.0
                w.write(16, 0x3C00); // dct4multipliers[c][1] = 1.0
            }
            write_dct_params(&pad4(&b4), 4, w);
            continue;
        }
        if idx == 10
            && let Some(aw) = afv
        {
            // AFV carries its own 4x8 and 4x4 parameters; the 4x8 ones are
            // the frame's DCT4X8 table, the 4x4 ones its DCT4X4 table.
            let o4x8 = matrices.custom_tables[5]
                .as_ref()
                .expect("custom AFV tables come with a custom DCT4X8 table");
            let b4 = fine
                .expect("custom AFV tables come with the fine tables")
                .dct4x4;
            w.write(3, K_QUANT_MODE_AFV);
            for row in &aw {
                for (i, &v) in row.iter().enumerate() {
                    let wire = if i < 6 { v / 64.0 } else { v };
                    w.write(16, u64::from(f32_to_f16_bits(wire)));
                }
            }
            write_dct_params(&o4x8.bands, 4, w);
            write_dct_params(&pad4(&b4), 4, w);
            continue;
        }
        let bands = if idx == 7 { t8x32 } else { bands };
        match bands {
            None => w.write(3, K_QUANT_MODE_LIBRARY),
            Some(o) => {
                if idx == TABLE_DCT4X8 {
                    w.write(3, K_QUANT_MODE_DCT4X8);
                    for _ in 0..3 {
                        w.write(16, 0x3C00); // dct4x8multipliers[c] = 1.0
                    }
                } else {
                    w.write(3, K_QUANT_MODE_DCT);
                }
                w.write(4, o.num_bands as u64 - 1);
                for row in &o.bands {
                    for (i, &v) in row[..o.num_bands].iter().enumerate() {
                        let wire = if i == 0 { v / 64.0 } else { v };
                        w.write(16, u64::from(f32_to_f16_bits(wire)));
                    }
                }
            }
        }
    }
}

fn write_ac_global(
    matrices: &crate::quant_weights::DequantMatrices,
    used_quant_tables: &[bool; 13],
    coeff_orders: &crate::coeff_order::CoeffOrders,
    num_groups: usize,
    ac_codes: &[crate::entropy::OwnedEntropyCode],
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    write_dequant_matrices(matrices, used_quant_tables, w);
    if num_groups > 1 {
        let bits = 32
            - (num_groups as u32).leading_zeros()
            - if num_groups.is_power_of_two() { 1 } else { 0 };
        if bits != 0 {
            w.write(bits as usize, 0);
        }
    }
    // HfGlobal parses `num_passes` HfPass blocks (jxl-frame hf_global.rs:57-59),
    // each = used_orders(U32 sel 3 + u(13)=0 -> natural order) + hf_dist entropy
    // code. Each pass gets its own code (ac_codes[p]).
    for code in ac_codes {
        crate::coeff_order::write_coeff_orders(coeff_orders, &mut scratch.huffman_pool, w);
        w.write(1, 0);
        write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, w);
    }
}

/// Section sizes in codestream order, after the optional permutation
/// (`permutation[i]` = codestream slot of canonical section `i`).
fn write_toc(
    sizes: &[usize],
    permutation: Option<(&[u32], &mut Vec<crate::entropy::HuffmanNode>)>,
    w: &mut BitWriter,
) {
    match permutation {
        Some((permutation, huffman_pool)) => {
            w.write(1, 1);
            crate::coeff_order::write_permutation(permutation, huffman_pool, w);
        }
        None => w.write(1, 0),
    }
    w.zero_pad_to_byte();
    let k_bits = [10usize, 14, 22, 30];
    for &s in sizes {
        let mut offset: usize = 0;
        let mut ok = false;
        for (i, &b) in k_bits.iter().enumerate() {
            if s < offset + (1usize << b) {
                w.write(2, i as u64);
                w.write(b, (s - offset) as u64);
                ok = true;
                break;
            }
            offset += 1usize << b;
        }
        assert!(ok, "section size {} too large for TOC", s);
    }
    w.zero_pad_to_byte();
}

pub(crate) fn combine_sections(sections: &mut Vec<BitWriter>, writer: &mut BitWriter) {
    combine_sections_ordered(sections, None, writer);
}

/// [`combine_sections`] storing the sections in `order` (canonical section
/// indices in codestream order), signaled through the TOC permutation.
fn combine_sections_ordered(
    sections: &mut Vec<BitWriter>,
    order: Option<(&[usize], &mut Vec<crate::entropy::HuffmanNode>)>,
    writer: &mut BitWriter,
) {
    if sections.len() == 4 {
        // Single AC group case: concat sections 1..4 (bitwise) into section 0.
        let tail: Vec<BitWriter> = sections.drain(1..).collect();
        for s in &tail {
            sections[0].append(s);
        }
    }
    let order = order.filter(|(order, _)| {
        sections.len() > 1 && order.iter().enumerate().any(|(slot, &i)| slot != i)
    });
    let Some((order, huffman_pool)) = order else {
        let sizes: Vec<usize> = sections
            .iter()
            .map(|s| s.bits_written().div_ceil(8))
            .collect();
        write_toc(&sizes, None, writer);
        // After write_toc, writer is byte-aligned.
        writer.append_byte_aligned(sections);
        return;
    };
    debug_assert_eq!(order.len(), sections.len());
    let mut permutation = vec![0u32; order.len()];
    for (slot, &i) in order.iter().enumerate() {
        permutation[i] = slot as u32;
    }
    let mut taken: Vec<Option<BitWriter>> = sections.drain(..).map(Some).collect();
    sections.extend(order.iter().map(|&i| taken[i].take().expect("permutation")));
    let sizes: Vec<usize> = sections
        .iter()
        .map(|s| s.bits_written().div_ceil(8))
        .collect();
    write_toc(&sizes, Some((&permutation, huffman_pool)), writer);
    writer.append_byte_aligned(sections);
}

/// Canonical section indices of a VarDCT frame in `group_order`: the
/// LfGlobal and HfGlobal sections stay put, the DC groups and each pass's AC
/// groups are sorted by [`group_order_key`].
fn group_section_order(
    dim: &ImageDim,
    num_passes: usize,
    group_order: crate::GroupOrder,
) -> Vec<usize> {
    let sorted = |xgroups: usize, ygroups: usize, group_dim: usize| -> Vec<usize> {
        let mut keys: Vec<(i64, f64, f64, usize)> = (0..xgroups * ygroups)
            .map(|g| {
                let (x0, y0) = ((g % xgroups) * group_dim, (g / xgroups) * group_dim);
                let w = group_dim.min(dim.xsize - x0);
                let h = group_dim.min(dim.ysize - y0);
                let center = (x0 as f64 + w as f64 / 2.0, y0 as f64 + h as f64 / 2.0);
                let (band, radius, angle) = group_order_key(dim, group_dim, center, group_order);
                (band, radius, angle, g)
            })
            .collect();
        keys.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.total_cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        keys.into_iter().map(|(.., g)| g).collect()
    };
    let dc = sorted(dim.xsize_dc_groups, dim.ysize_dc_groups, K_DC_GROUP_DIM);
    let ac = sorted(dim.xsize_groups, dim.ysize_groups, K_GROUP_DIM);
    let mut order = vec![0];
    order.extend(dc.iter().map(|&g| 1 + g));
    order.push(1 + dim.num_dc_groups);
    for pass in 0..num_passes {
        let base = 2 + dim.num_dc_groups + pass * dim.num_groups;
        order.extend(ac.iter().map(|&g| base + g));
    }
    order
}

/// Sort key of a group centered at `center`: (band, distance from the image
/// center, clockwise angle from the top). Center-first bands are rings one
/// group wide, each swept by angle alone; diagonal bands are half a group
/// wide around the nearer corner-to-corner diagonal and grow outward, so the
/// image draws an X that then thickens.
fn group_order_key(
    dim: &ImageDim,
    group_dim: usize,
    center: (f64, f64),
    group_order: crate::GroupOrder,
) -> (i64, f64, f64) {
    let (w, h) = (dim.xsize as f64, dim.ysize as f64);
    let (dx, dy) = (center.0 - w / 2.0, center.1 - h / 2.0);
    let radius = dx.hypot(dy);
    let angle = dx.atan2(-dy).rem_euclid(std::f64::consts::TAU);
    let g = group_dim as f64;
    match group_order {
        crate::GroupOrder::Diagonals => {
            let diagonal = w.hypot(h);
            let main = (w * center.1 - h * center.0).abs() / diagonal;
            let anti = (w * (h - center.1) - h * center.0).abs() / diagonal;
            ((main.min(anti) / (0.5 * g)).floor() as i64, radius, angle)
        }
        _ => ((radius / g).floor() as i64, 0.0, angle),
    }
}

fn zero_alpha_for_lossy(alpha: Option<&AlphaPlane>, pixels: usize) -> Option<AlphaPlane> {
    match alpha {
        Some(AlphaPlane::U8(_)) => Some(AlphaPlane::U8(vec![0; pixels])),
        Some(AlphaPlane::U16 { bits, .. }) => Some(AlphaPlane::U16 {
            data: vec![0; pixels],
            bits: *bits,
        }),
        Some(AlphaPlane::F32(_)) => Some(AlphaPlane::F32(vec![0; pixels])),
        None => None,
    }
}

/// Auto gate of the lossy-modular arm: byte cushion for the calibration's
/// per-image quality noise.
const LM_GATE_MARGIN: f64 = 1.08;

/// Texture scaling of the lossy-modular arm's calibrated distance: the
/// VarDCT-matching ratio tracks luma gradient energy (textured images want
/// the arm coarser, smooth ones finer). Reference gradient, exponents below
/// and above it.
const LM_GRAD_REF: f32 = 0.0231;
const LM_GRAD_EXP_SMOOTH: f32 = 0.16;
const LM_GRAD_EXP_TEXTURED: f32 = 0.08;

/// `(gradient / LM_GRAD_REF)^exp` from the mean absolute horizontal and
/// vertical luminance step of the linear image, clamped to the fitted range.
fn lm_texture_factor(linear: &Image3F) -> f32 {
    let (w, h) = (linear.xsize(), linear.ysize());
    let n = w * h;
    let (r, g, b) = (
        &linear.plane_data(0)[..n],
        &linear.plane_data(1)[..n],
        &linear.plane_data(2)[..n],
    );
    let luma = |i: usize| 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
    let mut sum_h = 0.0f64;
    let mut sum_v = 0.0f64;
    for y in 0..h {
        let row = y * w;
        let mut prev = luma(row);
        for x in 1..w {
            let cur = luma(row + x);
            sum_h += (cur - prev).abs() as f64;
            prev = cur;
        }
        if y + 1 < h {
            for x in 0..w {
                sum_v += (luma(row + w + x) - luma(row + x)).abs() as f64;
            }
        }
    }
    let mean_h = if w > 1 {
        sum_h / ((w - 1) * h) as f64
    } else {
        0.0
    };
    let mean_v = if h > 1 {
        sum_v / (w * (h - 1)) as f64
    } else {
        0.0
    };
    let ratio = (0.5 * (mean_h + mean_v)) as f32 / LM_GRAD_REF;
    let exp = if ratio < 1.0 {
        LM_GRAD_EXP_SMOOTH
    } else {
        LM_GRAD_EXP_TEXTURED
    };
    ratio.clamp(0.13, 4.3).powf(exp)
}

/// Integer lattice of the plain lossy-modular arm. Squeeze rounds its
/// averages to the lattice at every level
const LM_PLAIN_MAX: f32 = 65535.0;

/// The non-XYB frame's integer planes for the lossy-modular arm on the
/// `LM_PLAIN_MAX` lattice: the coded color planes (gray: its one channel),
/// then K at its squeeze distance (rescaled to the lattice) or alpha as
/// stored (coded losslessly). None for float alpha.
fn lm_plain_input(
    ctx: &EncodingContext,
    coded: &Image3F,
    gray: bool,
    alpha: Option<&AlphaPlane>,
) -> Option<crate::lossless::LmPlainInput> {
    let n = coded.xsize() * coded.ysize();
    let max = LM_PLAIN_MAX;
    let quantize = |input: &[f32]| {
        let mut levels = vec![0i32; input.len()];
        (ctx.quantize_dc_i32)(input, max, &mut levels);
        levels
    };
    let color: Vec<Vec<i32>> = if gray {
        // Gray is coded as three equal channels; the modular frame carries
        // the sample itself (YCbCr's luma plane sits 128/255 below it).
        let offset = match ctx.coding {
            crate::coding::CodingTransform::YCbCr => 128.0 / 255.0,
            _ => 0.0,
        };
        if offset == 0.0 {
            vec![quantize(&coded.plane_data(1)[..n])]
        } else {
            let width = coded.xsize().max(1);
            let mut source = vec![0.0f32; width];
            let mut levels = vec![0i32; n];
            for (row, target) in coded.plane_data(1)[..n]
                .chunks(width)
                .zip(levels.chunks_mut(width))
            {
                for (value, &input) in source.iter_mut().zip(row) {
                    *value = input + offset;
                }
                (ctx.quantize_dc_i32)(&source[..row.len()], max, target);
            }
            vec![levels]
        }
    } else {
        (0..3)
            .map(|c| quantize(&coded.plane_data(c)[..n]))
            .collect()
    };
    let extra: Vec<Vec<i32>> = match alpha {
        Some(AlphaPlane::F32(_)) => return None,
        Some(plane) if ctx.extra_squeeze_distance > 0.0 => {
            let scale = max / ((1u32 << plane.bits()) - 1) as f32;
            let width = coded.xsize().max(1);
            let mut source = vec![0.0f32; width];
            let mut levels = vec![0i32; n];
            for (y, target) in levels.chunks_mut(width).enumerate() {
                for (x, value) in source[..target.len()].iter_mut().enumerate() {
                    *value = plane.get_i32(y * width + x) as f32;
                }
                (ctx.quantize_dc_i32)(&source[..target.len()], scale, target);
            }
            vec![levels]
        }
        Some(plane) => vec![(0..n).map(|i| plane.get_i32(i)).collect()],
        None => Vec::new(),
    };
    Some(crate::lossless::LmPlainInput {
        color,
        extra: extra
            .into_iter()
            .map(|v| (v, ctx.extra_squeeze_distance))
            .collect(),
        max,
        coding: ctx.coding,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    linear: Image3F,
    coded: Option<Image3F>,
    is_achromatic: bool,
    alpha: Option<&AlphaPlane>,
    fill_invisible: bool,
    coeff_shifts: &[u32],
    patches: bool,
    integer_samples: bool,
    writer: &mut BitWriter,
) -> Result<bool, EncodeError> {
    let distp = compute_distance_params(distance);
    if !ctx.coding.is_xyb() {
        // Non-XYB planes skip every XYB content heuristic: they read as
        // achromatic to the classifiers, and patches quantize on the XYB
        // lattice.
        let mut coded = coded.expect("non-XYB frames carry their coded planes");
        if fill_invisible && let Some(alpha) = alpha {
            crate::invisible::fill_invisible(&mut coded, alpha);
        }
        // The lossy-modular arm codes the integer samples themselves (gray
        // as its one channel; alpha or K as an extra channel).
        let modular = |scratch: &mut CoderScratch| -> Option<BitWriter> {
            if !integer_samples {
                return None;
            }
            let mut w = BitWriter::new();
            crate::lossless::encode_frame_lossy_modular_plain(
                &lm_plain_input(ctx, &coded, is_achromatic, alpha)?,
                coded.xsize(),
                coded.ysize(),
                crate::lossless::lm_plain_distance(ctx.coding, distance)
                    * lm_texture_factor(&linear),
                ctx.speed,
                &ctx.thread_pool,
                scratch,
                &mut w,
            )
            .then_some(w)
        };
        let auto =
            ctx.lossy_modular == crate::LossyModular::Auto && ctx.speed.effort().lossy_modular_auto;
        let modular_writer = match ctx.lossy_modular {
            crate::LossyModular::Force => match modular(scratch) {
                Some(w) => {
                    writer.append(&w);
                    return Ok(true);
                }
                None => None,
            },
            crate::LossyModular::Auto if auto => modular(scratch),
            _ => None,
        };
        let mut aq_source = to_xyb_image(ctx, scratch, linear);
        if fill_invisible && let Some(alpha) = alpha {
            crate::invisible::fill_invisible(&mut aq_source, alpha);
        }
        let _ = ctx.aq_source.set(aq_source);
        ctx.set_plain_matrices(&coded);
        ctx.set_cfl_frame(crate::coding::fit_cfl_frame(ctx.coding, &coded));
        let mut vardct_writer = BitWriter::new();
        encode_frame_vardct(
            ctx,
            scratch,
            distance,
            &distp,
            coded,
            true,
            alpha,
            coeff_shifts,
            false,
            &mut vardct_writer,
        )?;
        return Ok(match modular_writer {
            Some(m)
                if (m.bits_written() as f64) * LM_GATE_MARGIN
                    < vardct_writer.bits_written() as f64 =>
            {
                writer.append(&m);
                true
            }
            _ => {
                writer.append(&vardct_writer);
                false
            }
        });
    }
    let modular_distance = if alpha.is_none()
        && (ctx.lossy_modular == crate::LossyModular::Force
            || (ctx.lossy_modular == crate::LossyModular::Auto
                && ctx.speed.effort().lossy_modular_auto))
    {
        crate::lossless::lm_calibrated_distance(distance) * lm_texture_factor(&linear)
    } else {
        0.0
    };
    let mut xyb = to_xyb_image(ctx, scratch, linear);
    if fill_invisible && let Some(alpha) = alpha {
        crate::invisible::fill_invisible(&mut xyb, alpha);
    }
    if is_achromatic {
        snap_achromatic_xyb(&mut xyb);
    }
    // Lossy-modular arm (quantized Squeeze): whole-frame modular encoding of
    // the XYB image at a distance calibrated to match the VarDCT arm's
    // quality. Force always takes the arm when the frame supports it; Auto
    // (Slow only) dual-encodes and keeps the smaller frame.
    if alpha.is_none() && ctx.lossy_modular == crate::LossyModular::Force {
        let mut modular_writer = BitWriter::new();
        if crate::lossless::encode_frame_lossy_modular_squeeze(
            &xyb,
            modular_distance,
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut modular_writer,
        ) {
            writer.append(&modular_writer);
            return Ok(false);
        }
    }
    if alpha.is_none()
        && ctx.lossy_modular == crate::LossyModular::Auto
        && ctx.speed.effort().lossy_modular_auto
    {
        let mut modular_writer = BitWriter::new();
        let have_modular = crate::lossless::encode_frame_lossy_modular_squeeze(
            &xyb,
            modular_distance,
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut modular_writer,
        );
        let mut vardct_writer = BitWriter::new();
        encode_frame_vardct(
            ctx,
            scratch,
            distance,
            &distp,
            xyb,
            is_achromatic,
            alpha,
            coeff_shifts,
            patches,
            &mut vardct_writer,
        )?;
        if have_modular
            && (modular_writer.bits_written() as f64) * LM_GATE_MARGIN
                < vardct_writer.bits_written() as f64
        {
            writer.append(&modular_writer);
        } else {
            writer.append(&vardct_writer);
        }
        return Ok(false);
    }
    encode_frame_vardct(
        ctx,
        scratch,
        distance,
        &distp,
        xyb,
        is_achromatic,
        alpha,
        coeff_shifts,
        patches,
        writer,
    )?;
    Ok(false)
}

/// The VarDCT arm of lossy encoding (regular or patched), from a prepared XYB
/// image.
#[allow(clippy::too_many_arguments)]
fn encode_frame_vardct(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    distp: &DistanceParams,
    xyb: Image3F,
    is_achromatic: bool,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    patches: bool,
    writer: &mut BitWriter,
) -> Result<(), EncodeError> {
    let slow_chromatic = ctx.speed.effort().chroma_policy && !is_achromatic;
    let content = if slow_chromatic {
        chroma_content_stats(&xyb)
    } else {
        ChromaContentStats::default()
    };
    let saturation_stat = content.saturation;
    let [x_grad_stat, b_grad_stat, chroma_texture_stat] = if slow_chromatic {
        chroma_gradient_stats(ctx, &xyb, saturation_stat)
    } else {
        [0.0; 3]
    };
    let (x_steps, b_steps) = match ctx.speed {
        _ if is_achromatic => (0, 0),
        _ if slow_chromatic => pixel_chromacity_steps(&xyb),
        // Without the chroma policy the B steps only cost rate.
        _ if ctx.speed.effort().pixel_chromacity_x && distance >= PIXEL_CHROMACITY_MIN_DISTANCE => {
            (pixel_chromacity_steps(&xyb).0, 0)
        }
        _ => (0, 0),
    };
    if distance >= PIXEL_CHROMACITY_MIN_DISTANCE {
        ctx.raise_x_qm_scale_floor(2 + x_steps);
        ctx.raise_b_qm_scale(2 + b_steps);
    }
    // Saturated-content tables are validated under Slow only: at Fast the
    // same swap buys chroma with a d=1 rate premium instead of saving bytes.
    ctx.set_chroma_heavy(slow_chromatic && saturation_stat >= SAT_QM_THRESHOLD);
    // Chroma-texture class: thin chromatic detail whose B energy the pair
    // transforms' coarse first band zeroes. Only the mid-band pays off.
    let pair_b_fine = slow_chromatic
        && (crate::quant_weights::PAIR_B_FINE_MIN_DISTANCE
            ..crate::quant_weights::PAIR_B_FINE_MAX_DISTANCE)
            .contains(&distance)
        && chroma_texture_stat >= PAIR_B_TEXTURE_THRESHOLD;
    ctx.set_pair_b_fine(pair_b_fine);
    // This is a content class, not a distance tier. Keeping its decision
    // independent of the 1.25 table boundary prevents saturated HF images
    // from switching plain B weights, custom opsin and both chroma scales at
    // the same point. Ordinary photos measured far below this gate.
    ctx.set_x_heavy(slow_chromatic && x_grad_stat >= X_QM_GRAD_THRESHOLD);
    ctx.set_b_heavy(slow_chromatic && b_grad_stat >= B_GRAD_THRESHOLD);
    if ctx.x_heavy() && b_steps > 0 {
        ctx.raise_b_qm_scale(X_HEAVY_B_QM_SCALE);
    }

    apply_point_chroma_policy(ctx, distp, content, is_achromatic);
    ctx.set_screen_content(constant_block_fraction(&xyb) >= SCREEN_CONTENT_CONSTANT_BLOCKS);
    #[cfg(feature = "splines")]
    let quant_field = if ctx.splines || ctx.dots {
        spline_quant_field(ctx, scratch, &xyb, distp)?
    } else {
        Vec::new()
    };
    #[cfg(feature = "splines")]
    let spline_candidates = if ctx.splines || ctx.dots {
        crate::splines::find_candidates(ctx, scratch, distance, &xyb, &quant_field)
    } else {
        None
    };
    #[cfg(feature = "splines")]
    let stage = spline_candidates.as_ref().map(|candidates| SplineStage {
        candidates,
        quant_field: &quant_field,
        before_tiles: false,
    });
    if !patches {
        let prepared = prepare_vardct_variant(
            ctx,
            scratch,
            distance,
            distp,
            xyb,
            alpha,
            coeff_shifts,
            false,
            None,
            true,
            #[cfg(feature = "splines")]
            stage.as_ref(),
        )?;
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        return prepared.encode(ctx, scratch, distance, alpha, coeff_shifts, writer);
    }

    // Compare bounded residual samples plus the actual atlas, dictionary and
    // spline costs. Only the selected plan receives a full-frame encode.
    let lattice = crate::lattice_patches::find(&xyb);
    let glyphs = GlyphStage::find(ctx, scratch, distance, &xyb, alpha);
    let mut best = prepare_vardct_variant(
        ctx,
        scratch,
        distance,
        distp,
        xyb.clone(),
        alpha,
        coeff_shifts,
        true,
        glyphs.as_ref(),
        true,
        #[cfg(feature = "splines")]
        stage.as_ref(),
    )?;
    if !best.used_tiles && lattice.is_none() {
        drop(xyb);
        drop(glyphs);
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        return best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer);
    }
    #[allow(unused_mut)]
    let mut best_bits = best.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
    if let Some(plan) = lattice {
        drop(xyb);
        drop(glyphs);
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        // A lattice arm codes what the tile leaves at the frame's own distance,
        // where an exact-tile plan repeats it at the finer atlas distance, so
        // near-equal bits buy only a coarser remainder.
        best_bits *= LATTICE_BITS_MARGIN;
        for mut arm in lattice_variants(ctx, scratch, distance, distp, plan, alpha)? {
            let bits = arm.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
            if bits < best_bits {
                best = arm;
                best_bits = bits;
            }
        }
        return best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer);
    }
    #[cfg(feature = "splines")]
    if best.tiles_touched {
        let stage = SplineStage {
            before_tiles: true,
            ..stage.expect("candidates touched a tile")
        };
        let mut splines_first = prepare_vardct_variant(
            ctx,
            scratch,
            distance,
            distp,
            xyb.clone(),
            alpha,
            coeff_shifts,
            true,
            glyphs.as_ref(),
            true,
            Some(&stage),
        )?;
        let bits = splines_first.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
        if bits < best_bits {
            best = splines_first;
            best_bits = bits;
        }
    }
    let mut no_tiles = prepare_vardct_variant(
        ctx,
        scratch,
        distance,
        distp,
        xyb,
        alpha,
        coeff_shifts,
        true,
        glyphs.as_ref(),
        false,
        #[cfg(feature = "splines")]
        stage.as_ref(),
    )?;
    let no_tile_bits = no_tiles.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
    // Tile atlases are finer than the residual, so dropping them must repay
    // the same quality margin as the previous full-bitstream comparison.
    if no_tile_bits * GLYPH_BITS_MARGIN < best_bits {
        best = no_tiles;
    } else {
        drop(no_tiles);
    }
    drop(glyphs);
    #[cfg(feature = "splines")]
    drop(spline_candidates);
    best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer)
}

/// The kAdd and kMul lattice arms of a frame whose luma repeats on an integer
/// lattice; both race the tile/glyph plans on estimated bits.
#[allow(clippy::too_many_arguments)]
fn lattice_variants(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    distp: &DistanceParams,
    plan: crate::lattice_patches::LatticePlan,
    alpha: Option<&AlphaPlane>,
) -> Result<Vec<PreparedVarDct>, EncodeError> {
    let (px, py) = plan.period;
    let add_references = plan.add_references();
    let mul_references = plan.mul_references();
    let atlas_alpha = zero_alpha_for_lossy(alpha, px * py);
    let modular_atlas = |atlas: &[Vec<i32>; 3], scratch: &mut CoderScratch| {
        let mut bits = BitWriter::new();
        crate::lossless::encode_modular_xyb_atlas_tree(
            atlas,
            px,
            py,
            atlas_alpha.as_ref(),
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut bits,
        );
        bits
    };
    let prepared =
        |prefix: BitWriter, mut base: Image3F, references: Vec<crate::patches::PatchReference>| {
            gaborize(&mut base, distp);
            PreparedVarDct {
                prefix,
                base,
                references,
                #[cfg(feature = "splines")]
                splines: None,
                tiles_touched: false,
                used_tiles: false,
                estimated_bits: None,
            }
        };
    let add = prepared(
        modular_atlas(&plan.add_atlas, scratch),
        plan.add_base,
        add_references,
    );
    let mut mul_prefix = modular_atlas(&plan.mul_atlas, scratch);
    let (w, h) = (plan.residual.xsize(), plan.residual.ysize());
    let mut residual = plan.residual;
    gaborize(&mut residual, distp);
    let residual_alpha = zero_alpha_for_lossy(alpha, w * h);
    encode_frame_core(
        ctx,
        scratch,
        distance,
        residual,
        residual_alpha.as_ref(),
        &[0],
        VarDctFrameKind::ReferenceOnly {
            width: w,
            height: h,
        },
        #[cfg(feature = "splines")]
        None,
        &mut mul_prefix,
    )?;
    let mul = prepared(mul_prefix, plan.gain, mul_references);
    Ok(vec![add, mul])
}

/// Spline selection of one frame variant.
#[cfg(feature = "splines")]
#[derive(Clone, Copy)]
struct SplineStage<'a> {
    candidates: &'a crate::splines::SplineCandidates,
    quant_field: &'a [f32],
    /// Select before the Replace tiles are detected: splines may then cross
    /// tiles, which are cut from the spline-subtracted image.
    before_tiles: bool,
}

/// A frame's glyph patches and their coded atlas. Every variant of the
/// patch race starts from the same glyphs, so they are found and coded once.
struct GlyphStage {
    atlas: BitWriter,
    base: Image3F,
    references: Vec<crate::patches::PatchReference>,
}

impl GlyphStage {
    fn find(
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        distance: f32,
        xyb: &Image3F,
        alpha: Option<&AlphaPlane>,
    ) -> Option<Self> {
        let free_singles = distance <= GLYPH_SINGLETON_FREE_MAX_DISTANCE;
        let coarse = if distance < 1.5 {
            1
        } else if distance < 2.0 {
            2
        } else if distance < 6.0 {
            4
        } else {
            8
        };
        let plan = crate::patches::find_lossy_glyph_patches(
            ctx,
            xyb,
            &crate::patches::GlyphParams {
                min_occurrences: 1,
                min_repeat_pixels: if free_singles {
                    0
                } else {
                    GLYPH_MIN_REPEAT_PIXELS
                },
                coarse,
                min_contrast: GLYPH_MIN_CONTRAST_STEPS * coarse,
                singles_min_cover: if free_singles {
                    0.0
                } else {
                    GLYPH_SINGLETON_MIN_COVER
                },
            },
        )?;
        let mut atlas = BitWriter::new();
        let atlas_alpha = zero_alpha_for_lossy(alpha, plan.atlas_width * plan.atlas_height);
        crate::lossless::encode_modular_xyb_atlas_tree(
            &plan.atlas,
            plan.atlas_width,
            plan.atlas_height,
            atlas_alpha.as_ref(),
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut atlas,
        );
        Some(Self {
            atlas,
            base: plan.base,
            references: plan.references,
        })
    }
}

/// One patched or regular VarDCT frame, with or without Replace `tiles`.
/// Atlas frames are small and encoded immediately; the full residual is deferred.
#[allow(clippy::too_many_arguments)]
fn prepare_vardct_variant(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    distp: &DistanceParams,
    xyb: Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    patches: bool,
    glyphs: Option<&GlyphStage>,
    tiles: bool,
    #[cfg(feature = "splines")] stage: Option<&SplineStage>,
) -> Result<PreparedVarDct, EncodeError> {
    #[allow(unused_mut)]
    let mut xyb = xyb;
    #[allow(unused_mut)]
    let mut tiles_touched = false;
    #[allow(unused_mut)]
    let mut used_tiles = false;
    #[cfg(feature = "splines")]
    let before_tiles = stage.is_some_and(|s| s.before_tiles);
    #[cfg(feature = "splines")]
    let select_splines =
        |scratch: &mut CoderScratch, image: &mut Image3F, forbidden: Option<&[bool]>| {
            let stage = stage?;
            crate::splines::select_splines(
                ctx,
                scratch,
                distance,
                image,
                stage.quant_field,
                stage.candidates,
                forbidden,
            )
        };

    // Near-duplicate tiles beyond two 8-bit levels of RMS difference match
    // noise on smooth ground; the frame without them then wins the race and
    // the plan only adds atlas and sampling work.
    let near_tiles = if ctx.speed.effort().near_patch_tiles {
        distance.clamp(0.5, NEAR_TILES_MAX)
    } else {
        0.0
    };
    if let Some(glyphs) = glyphs {
        let mut patched_writer = glyphs.atlas.clone();

        // Repeated tiles of what is left (icons, rules, UI chrome). They are
        // Replace entries and must precede the additive glyph entries, which
        // then land on top of the replaced rectangles.
        let mut base = glyphs.base.clone();
        #[cfg(feature = "splines")]
        let early_splines = if before_tiles {
            select_splines(scratch, &mut base, None)
        } else {
            None
        };
        let mut references: Vec<crate::patches::PatchReference> = Vec::new();
        if tiles {
            let tiles16 = crate::patches::find_lossy_patches_sized(
                &base,
                crate::patches::PATCH_TILE,
                near_tiles,
                &ctx.thread_pool,
                scratch,
            );
            let after16 = tiles16.as_ref().map_or(&base, |t| &t.base);
            let tiles8 = crate::patches::find_lossy_patch_groups_sized(
                after16,
                8,
                near_tiles,
                &ctx.thread_pool,
                scratch,
            );
            let covered = tiles16
                .as_ref()
                .map_or(0, |t| t.groups.iter().map(|g| g.len()).sum::<usize>() * 256)
                + tiles8.as_ref().map_or(0, |groups| {
                    groups.iter().map(|g| g.len()).sum::<usize>() * 64
                });
            if covered as f64 >= PATCH_TILE_MIN_COVER * (base.xsize() * base.ysize()) as f64 {
                let mut tile_refs = Vec::new();
                let mut tile_atlas: Option<Image3F> = None;
                let mut next_base: Option<Image3F> = None;
                if let Some(t) = tiles16 {
                    let (atlas, refs) =
                        crate::patches::pack_lossy_atlas(&base, t.groups, PATCH_REF_ID);
                    tile_atlas = Some(atlas);
                    tile_refs = refs;
                    next_base = Some(t.base);
                }
                if let Some(groups8) = tiles8 {
                    let residual = next_base.get_or_insert_with(|| base.clone());
                    crate::patches::clear_patch_tiles(residual, 8, &groups8);
                    let (reused, groups8) =
                        crate::patches::reuse_tile_quadrants(&base, &tile_refs, groups8);
                    tile_refs.extend(reused);
                    if !groups8.is_empty() {
                        let (atlas8, mut refs8) =
                            crate::patches::pack_lossy_atlas_sized(&base, groups8, 8, PATCH_REF_ID);
                        tile_atlas = Some(match tile_atlas {
                            Some(top) => crate::patches::stack_atlases(top, atlas8, &mut refs8),
                            None => atlas8,
                        });
                        tile_refs.extend(refs8);
                    }
                }
                let tile_atlas = tile_atlas.expect("covered > 0");
                let atlas_distance = distance * ATLAS_DISTANCE_SCALE;
                let atlas_distp = compute_distance_params(atlas_distance);
                let tile_alpha = zero_alpha_for_lossy(
                    alpha,
                    tile_atlas.xsize().saturating_mul(tile_atlas.ysize()),
                );
                let (atlas_w, atlas_h) = (tile_atlas.xsize(), tile_atlas.ysize());
                let mut tile_atlas = tile_atlas;
                gaborize(&mut tile_atlas, &atlas_distp);
                encode_frame_core(
                    ctx,
                    scratch,
                    atlas_distance,
                    tile_atlas,
                    tile_alpha.as_ref(),
                    &[0],
                    VarDctFrameKind::ReferenceOnly {
                        width: atlas_w,
                        height: atlas_h,
                    },
                    #[cfg(feature = "splines")]
                    None,
                    &mut patched_writer,
                )?;
                references.extend(tile_refs);
                base = next_base.expect("covered > 0");
                used_tiles = true;
            }
        }
        let atlas_bits = patched_writer.bits_written();
        references.extend(glyphs.references.iter().cloned());

        // Additive glyph entries commute with splines; Replace tiles cut from
        // the unsubtracted image must stay clear of them.
        #[cfg(feature = "splines")]
        let base_splines = if before_tiles {
            early_splines
        } else if let Some(stage) = stage {
            let blocks_w = base.xsize().div_ceil(K_BLOCK_DIM);
            let mut in_patch = vec![false; blocks_w * base.ysize().div_ceil(K_BLOCK_DIM)];
            for reference in references.iter().filter(|r| !r.add) {
                for &(px, py) in &reference.positions {
                    for by in py / K_BLOCK_DIM..(py + reference.height).div_ceil(K_BLOCK_DIM) {
                        for bx in px / K_BLOCK_DIM..(px + reference.width).div_ceil(K_BLOCK_DIM) {
                            in_patch[by * blocks_w + bx] = true;
                        }
                    }
                }
            }
            tiles_touched = stage.candidates.touches(&in_patch, blocks_w);
            select_splines(scratch, &mut base, Some(&in_patch))
        } else {
            None
        };
        gaborize(&mut base, distp);
        let mut prepared = PreparedVarDct {
            prefix: patched_writer,
            base,
            references,
            #[cfg(feature = "splines")]
            splines: base_splines,
            tiles_touched,
            used_tiles,
            estimated_bits: None,
        };
        if distance >= GLYPH_CHECK_MIN_DISTANCE {
            let bits = prepared.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
            if atlas_bits as f64 > bits * GLYPH_CHECK_ATLAS_SHARE {
                let mut regular = xyb;
                #[cfg(feature = "splines")]
                let regular_splines = select_splines(scratch, &mut regular, None);
                gaborize(&mut regular, distp);
                let mut regular = PreparedVarDct {
                    prefix: BitWriter::new(),
                    base: regular,
                    references: Vec::new(),
                    #[cfg(feature = "splines")]
                    splines: regular_splines,
                    tiles_touched,
                    used_tiles,
                    estimated_bits: None,
                };
                if bits
                    >= regular.estimate(ctx, scratch, distance, alpha, coeff_shifts)?
                        * GLYPH_BITS_MARGIN
                {
                    return Ok(regular);
                }
            }
        }
        return Ok(prepared);
    }

    #[cfg(feature = "splines")]
    let early_splines = if before_tiles {
        select_splines(scratch, &mut xyb, None)
    } else {
        None
    };
    // Coverage admits a tile plan; the frame-level rate model compares it
    // with a plan without tiles before either residual is encoded.
    let tile_plan = if patches && tiles {
        let plan16 = crate::patches::find_lossy_patches_sized(
            &xyb,
            crate::patches::PATCH_TILE,
            near_tiles,
            &ctx.thread_pool,
            scratch,
        );
        let after16 = plan16.as_ref().map_or(&xyb, |p| &p.base);
        let tiles8 = crate::patches::find_lossy_patch_groups_sized(
            after16,
            8,
            near_tiles,
            &ctx.thread_pool,
            scratch,
        );
        let covered = plan16
            .as_ref()
            .map_or(0, |t| t.groups.iter().map(|g| g.len()).sum::<usize>() * 256)
            + tiles8.as_ref().map_or(0, |groups| {
                groups.iter().map(|g| g.len()).sum::<usize>() * 64
            });
        (covered as f64 >= PATCH_TILE_MIN_COVER * (xyb.xsize() * xyb.ysize()) as f64).then(|| {
            let plan = plan16.unwrap_or_else(|| crate::patches::LossyPatches {
                base: xyb.clone(),
                groups: Vec::new(),
            });
            (plan, tiles8)
        })
    } else {
        None
    };
    if let Some((plan, tiles8)) = tile_plan {
        used_tiles = true;
        // Route each patch group to the atlas that codes it best: groups whose
        // quantized tiles fit the running 256-color palette budget go to the
        // modular atlas (measured strictly dominant when the palette hits),
        // the rest to a VarDCT atlas at a finer distance. Groups arrive most
        // frequent first, so high-value groups claim the palette budget first.
        // Either atlas is emitted only when it has content, and every
        // dictionary entry names its reference frame.
        let mut palette: std::collections::HashSet<[i32; 3]> = std::collections::HashSet::new();
        let groups = plan.groups;
        let (mut modular_idx, mut vardct_idx): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        {
            let tile_colors = scratch.patch_tile_colors.as_mut();
            for (i, group) in groups.iter().enumerate() {
                let (sx, sy) = group[0];
                crate::xyb::quantize_xyb_tile_colors(
                    &xyb,
                    sx,
                    sy,
                    MODULAR_ATLAS_LATTICE_SCALE,
                    tile_colors,
                );
                tile_colors.sort_unstable();
                let unique_colors = || {
                    tile_colors.iter().enumerate().filter_map(|(i, &color)| {
                        (i == 0 || tile_colors[i - 1] != color).then_some(color)
                    })
                };
                let new_colors = unique_colors().filter(|c| !palette.contains(c)).count();
                if palette.len() + new_colors <= 256 {
                    palette.extend(unique_colors());
                    modular_idx.push(i);
                } else {
                    vardct_idx.push(i);
                }
            }
        }
        let clone_groups = |idx: &[usize]| -> Vec<Vec<(usize, usize)>> {
            idx.iter().map(|&i| groups[i].clone()).collect()
        };

        let atlas_distance = distance * ATLAS_DISTANCE_SCALE;
        let atlas_distp = compute_distance_params(atlas_distance);
        let encode_vardct_atlas =
            |atlas: Image3F, scratch: &mut CoderScratch, out: &mut BitWriter| {
                let atlas_alpha =
                    zero_alpha_for_lossy(alpha, atlas.xsize().saturating_mul(atlas.ysize()));
                let (atlas_w, atlas_h) = (atlas.xsize(), atlas.ysize());
                let mut atlas = atlas;
                gaborize(&mut atlas, &atlas_distp);
                encode_frame_core(
                    ctx,
                    scratch,
                    atlas_distance,
                    atlas,
                    atlas_alpha.as_ref(),
                    &[0],
                    VarDctFrameKind::ReferenceOnly {
                        width: atlas_w,
                        height: atlas_h,
                    },
                    #[cfg(feature = "splines")]
                    None,
                    out,
                )
            };

        let mut patched_writer = BitWriter::new();
        let mut references: Vec<crate::patches::PatchReference> = Vec::new();

        // Palette viability routed the modular set, but viable is not the same
        // as compressible: a 256-pixel noise tile fits the palette budget and
        // still codes terribly (256 incompressible indices). Price the subset
        // both ways and keep the routing only when the modular frame is
        // actually smaller; otherwise everything folds back into one VarDCT
        // atlas in the original order.
        if !modular_idx.is_empty() {
            let (modular_atlas, modular_refs) = crate::patches::pack_lossy_atlas(
                &xyb,
                clone_groups(&modular_idx),
                MODULAR_PATCH_REF_ID,
            );
            let mut modular_bits = BitWriter::new();
            let modular_ok = crate::lossless::encode_modular_xyb_atlas(
                &modular_atlas,
                alpha.is_some(),
                MODULAR_ATLAS_LATTICE_SCALE,
                ctx.speed,
                true,
                scratch,
                &mut modular_bits,
            );
            let mut vardct_bits = BitWriter::new();
            encode_vardct_atlas(modular_atlas, scratch, &mut vardct_bits)?;
            if modular_ok && modular_bits.bits_written() < vardct_bits.bits_written() {
                patched_writer.append(&modular_bits);
                references.extend(modular_refs);
            } else {
                vardct_idx.append(&mut modular_idx);
                vardct_idx.sort_unstable();
            }
        }

        let mut base = plan.base;
        let mut vardct_atlas: Option<Image3F> = None;
        if !vardct_idx.is_empty() {
            let (atlas, vardct_refs) =
                crate::patches::pack_lossy_atlas(&xyb, clone_groups(&vardct_idx), PATCH_REF_ID);
            vardct_atlas = Some(atlas);
            references.extend(vardct_refs);
        }
        if let Some(groups8) = tiles8 {
            crate::patches::clear_patch_tiles(&mut base, 8, &groups8);
            let (reused, groups8) =
                crate::patches::reuse_tile_quadrants(&xyb, &references, groups8);
            references.extend(reused);
            if !groups8.is_empty() {
                let (atlas8, mut refs8) =
                    crate::patches::pack_lossy_atlas_sized(&xyb, groups8, 8, PATCH_REF_ID);
                vardct_atlas = Some(match vardct_atlas {
                    Some(top) => crate::patches::stack_atlases(top, atlas8, &mut refs8),
                    None => atlas8,
                });
                references.extend(refs8);
            }
        }
        if let Some(atlas) = vardct_atlas {
            encode_vardct_atlas(atlas, scratch, &mut patched_writer)?;
        }

        // Patches replace their rectangles and the decoder draws splines after
        // them, so no spline may reach into a patch cut from the unsubtracted
        // image.
        #[cfg(feature = "splines")]
        let base_splines = if before_tiles {
            early_splines
        } else if let Some(stage) = stage {
            let blocks_w = base.xsize().div_ceil(K_BLOCK_DIM);
            let mut in_patch = vec![false; blocks_w * base.ysize().div_ceil(K_BLOCK_DIM)];
            for reference in &references {
                for &(px, py) in &reference.positions {
                    for by in py / K_BLOCK_DIM..(py + reference.height).div_ceil(K_BLOCK_DIM) {
                        for bx in px / K_BLOCK_DIM..(px + reference.width).div_ceil(K_BLOCK_DIM) {
                            in_patch[by * blocks_w + bx] = true;
                        }
                    }
                }
            }
            tiles_touched = stage.candidates.touches(&in_patch, blocks_w);
            select_splines(scratch, &mut base, Some(&in_patch))
        } else {
            None
        };
        gaborize(&mut base, distp);
        return Ok(PreparedVarDct {
            prefix: patched_writer,
            base,
            references,
            #[cfg(feature = "splines")]
            splines: base_splines,
            tiles_touched,
            used_tiles,
            estimated_bits: None,
        });
    }

    #[cfg(feature = "splines")]
    let splines = if before_tiles {
        early_splines
    } else {
        select_splines(scratch, &mut xyb, None)
    };
    gaborize(&mut xyb, distp);
    Ok(PreparedVarDct {
        prefix: BitWriter::new(),
        base: xyb,
        references: Vec::new(),
        #[cfg(feature = "splines")]
        splines,
        tiles_touched,
        used_tiles,
        estimated_bits: None,
    })
}

struct PreparedVarDct {
    prefix: BitWriter,
    /// Selected splines/patches removed and inverse Gaborish already applied.
    base: Image3F,
    references: Vec<crate::patches::PatchReference>,
    #[cfg(feature = "splines")]
    splines: Option<crate::splines::SplineSet>,
    #[cfg_attr(not(feature = "splines"), allow(dead_code))]
    tiles_touched: bool,
    used_tiles: bool,
    estimated_bits: Option<f64>,
}

impl PreparedVarDct {
    /// Moves the selected dots out of the spline set into a template atlas
    /// (a modular reference frame) and kAdd dictionary entries.
    #[cfg(feature = "splines")]
    fn materialize_dots(
        &mut self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        alpha: Option<&AlphaPlane>,
    ) {
        use crate::splines::dots;
        let Some(set) = self.splines.as_mut() else {
            return;
        };
        let encoding = set.dot_encoding.take();
        let selected: Vec<_> = set.splines.extract_if(.., |sp| dots::is_dot(sp)).collect();
        if set.splines.is_empty() {
            self.splines = None;
        }
        if selected.is_empty() {
            return;
        }
        if alpha.is_none()
            && let Some(encoding) = encoding
        {
            self.prefix.append_bits(&encoding.atlas);
            self.references.extend(encoding.references);
            return;
        }
        let plan = dots::build_dot_patches(&selected);
        let atlas_alpha = zero_alpha_for_lossy(alpha, plan.width * plan.height);
        crate::lossless::encode_modular_xyb_atlas_tree_slot(
            &plan.atlas,
            plan.width,
            plan.height,
            atlas_alpha.as_ref(),
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            crate::patches::DOT_PATCH_REF_ID,
            &mut self.prefix,
        );
        self.references.extend(plan.references());
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "splines"), allow(unused_mut))]
    fn encode(
        mut self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        distance: f32,
        alpha: Option<&AlphaPlane>,
        coeff_shifts: &[u32],
        writer: &mut BitWriter,
    ) -> Result<(), EncodeError> {
        #[cfg(feature = "splines")]
        self.materialize_dots(ctx, scratch, alpha);
        writer.append(&self.prefix);
        encode_frame_core(
            ctx,
            scratch,
            distance,
            self.base,
            alpha,
            coeff_shifts,
            if self.references.is_empty() {
                VarDctFrameKind::Regular
            } else {
                VarDctFrameKind::Patched(&self.references)
            },
            #[cfg(feature = "splines")]
            self.splines.as_ref(),
            writer,
        )?;
        Ok(())
    }

    fn estimate(
        &mut self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        distance: f32,
        alpha: Option<&AlphaPlane>,
        coeff_shifts: &[u32],
    ) -> Result<f64, EncodeError> {
        if let Some(bits) = self.estimated_bits {
            return Ok(bits);
        }
        #[cfg(feature = "splines")]
        self.materialize_dots(ctx, scratch, alpha);
        let mut extras = BitWriter::new();
        if !self.references.is_empty() {
            crate::lossless::write_patch_dictionary(
                &self.references,
                alpha.is_some(),
                scratch,
                &mut extras,
            );
        }
        #[cfg(feature = "splines")]
        if let Some(set) = &self.splines {
            crate::splines::write_splines(set, scratch, &mut extras);
        }
        let noise_bits = if ctx.grain.enabled() {
            crate::noise::NoiseParams::BITS as f64
        } else {
            0.0
        };
        let bits = self.prefix.bits_written() as f64
            + extras.bits_written() as f64
            + noise_bits
            + estimate_residual_bits(ctx, scratch, distance, &self.base, alpha, coeff_shifts)?;
        self.estimated_bits = Some(bits);
        Ok(bits)
    }
}

/// Encode a bounded, spatially stratified sample with the real transform,
/// quantization and entropy choices. Scale only its data sections; global
/// headers and the actual patch/spline syntax are charged once.
/// Sampled statistics and boundaries can change RDO decisions, so this is a
/// rate estimate, not an exact predictor of the full bitstream.
fn estimate_residual_bits(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    source: &Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
) -> Result<f64, EncodeError> {
    let (w, h) = (source.xsize(), source.ysize());
    let (sample, sample_alpha) = sample_vardct_residual(source, alpha);
    let (sw, sh) = (sample.xsize(), sample.ysize());
    let mut writer = BitWriter::new();
    let payload = encode_frame_core(
        ctx,
        scratch,
        distance,
        sample,
        sample_alpha.as_ref(),
        coeff_shifts,
        VarDctFrameKind::Regular,
        #[cfg(feature = "splines")]
        None,
        &mut writer,
    )?;
    Ok((writer.bits_written() - payload) as f64
        + payload as f64 * (w * h) as f64 / (sw * sh) as f64)
}

/// Keep sample windows on the original pixel lattice and copy alpha through
/// the same positions. At most sixteen 128x128 windows enter the rate model.
fn sample_vardct_residual(
    source: &Image3F,
    alpha: Option<&AlphaPlane>,
) -> (Image3F, Option<AlphaPlane>) {
    let (w, h) = (source.xsize(), source.ysize());
    let side = (w.min(h) / 2).clamp(8, 128).next_power_of_two().min(128);
    let (tw, th) = (side.min(w), side.min(h));
    let (nx, ny) = (w.div_ceil(tw), h.div_ceil(th));
    let cells = nx * ny;
    let count = (cells / 4).clamp(1, 16);
    let columns = count.min(4);
    let rows = count.div_ceil(columns);
    let count = columns * rows;
    let (sw, sh) = (tw * columns, th * rows);
    let mut sample = Image3F::new(sw, sh);
    let mut indices = alpha.map(|_| vec![0usize; sw * sh]);
    for i in 0..count {
        let cell = ((2 * i + 1) * cells / (2 * count)).min(cells - 1);
        let sx = ((cell % nx) * tw).min(w - tw);
        let sy = ((cell / nx) * th).min(h - th);
        let (dx, dy) = ((i % columns) * tw, (i / columns) * th);
        for y in 0..th {
            for c in 0..3 {
                sample.plane_row_mut(c, dy + y)[dx..dx + tw]
                    .copy_from_slice(&source.plane_row(c, sy + y)[sx..sx + tw]);
            }
            if let Some(indices) = &mut indices {
                for x in 0..tw {
                    indices[(dy + y) * sw + dx + x] = (sy + y) * w + sx + x;
                }
            }
        }
    }
    let sample_alpha = alpha.zip(indices).map(|(alpha, indices)| match alpha {
        AlphaPlane::U8(data) => AlphaPlane::U8(indices.iter().map(|&i| data[i]).collect()),
        AlphaPlane::U16 { data, bits } => AlphaPlane::U16 {
            data: indices.iter().map(|&i| data[i]).collect(),
            bits: *bits,
        },
        AlphaPlane::F32(data) => AlphaPlane::F32(indices.iter().map(|&i| data[i]).collect()),
    });
    (sample, sample_alpha)
}

/// Effective AC quant (`scale * q`) per 8x8 block, measured on the image before
/// any spline is removed; the spline RD gate prices blocks with it.
#[cfg(feature = "splines")]
fn spline_quant_field(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
    distp: &DistanceParams,
) -> Result<Vec<f32>, EncodeError> {
    let blocks_w = xyb.xsize().div_ceil(K_BLOCK_DIM);
    let blocks_h = xyb.ysize().div_ceil(K_BLOCK_DIM);
    let mut field = vec![0f32; blocks_w * blocks_h];
    for y0 in (0..xyb.ysize()).step_by(K_DC_GROUP_DIM) {
        for x0 in (0..xyb.xsize()).step_by(K_DC_GROUP_DIM) {
            let gw = K_DC_GROUP_DIM.min(xyb.xsize() - x0).div_ceil(K_BLOCK_DIM);
            let gh = K_DC_GROUP_DIM.min(xyb.ysize() - y0).div_ceil(K_BLOCK_DIM);
            let mut raw = crate::image::ImageB::try_new_fill(gw, gh, 1)?;
            (ctx.fill_quant_field)(
                &mut scratch.aq_map,
                xyb,
                &mut raw,
                x0,
                y0,
                distp.distance,
                1.0 / distp.scale,
            );
            for y in 0..gh {
                let row = &mut field[(y0 / K_BLOCK_DIM + y) * blocks_w + x0 / K_BLOCK_DIM..];
                for (dst, &q) in row.iter_mut().zip(&raw.row(y)[..gw]) {
                    *dst = distp.scale * q as f32;
                }
            }
        }
    }
    Ok(field)
}

/// Unrepeated glyphs join the atlas freely up to this distance; above it
/// only when together they cover this share of the frame.
const GLYPH_SINGLETON_FREE_MAX_DISTANCE: f32 = 1.0;
const GLYPH_SINGLETON_MIN_COVER: f64 = 0.5;
/// A glyph must differ from its background by this many lattice rounding
/// steps somewhere; fainter shapes are shading the base codes for less.
const GLYPH_MIN_CONTRAST_STEPS: i32 = 2;
/// Box pixels a repeated glyph must save beyond its first occurrence.
const GLYPH_MIN_REPEAT_PIXELS: usize = 48;
/// A glyph plan is checked against the regular frame only from this distance
/// and when the atlas frames take more than this share of the patched bits.
/// Share of the scan walk the custom coefficient orders must remove before
/// the refine pass's prices are rebuilt on them.
const RDOQ_FRESH_PRICES_MIN_SCAN_MOVE: f64 = 0.35;
const GLYPH_CHECK_MIN_DISTANCE: f32 = 6.0;
const GLYPH_CHECK_ATLAS_SHARE: f64 = 0.35;
/// Patched bits may exceed the regular frame's by this factor.
const GLYPH_BITS_MARGIN: f64 = 1.10;
/// A lattice arm must stay under this share of the best tile/glyph plan's bits.
const LATTICE_BITS_MARGIN: f64 = 0.5;
/// Tile patches are used only when they cover this share of the frame.
const PATCH_TILE_MIN_COVER: f64 = 0.03;
const NEAR_TILES_MAX: f32 = 2.0;

/// Power-of-two refinement of the modular atlas quantization lattice.
const MODULAR_ATLAS_LATTICE_SCALE: u32 = 8;

/// The VarDCT atlas is coded this much finer than the frame it serves.
const ATLAS_DISTANCE_SCALE: f32 = 0.45;

/// Every opsin row sums to 1, so achromatic input gives L = M = S and hence
/// X = 0, B = Y — but only up to f32 rounding, since the three mixes are
/// evaluated with different constants. Snap the planes exact so chroma
/// quantizes to pure zeros and the CfL fit sees slope 0/1 instead of noise.
const SAT_QM_THRESHOLD: f32 = 0.055;
const PAIR_B_TEXTURE_THRESHOLD: f32 = 0.12;

/// `x_gradient_stat` gate for keeping the fine X quantizer through the
/// x_qm_scale=2 distance band (0.299..=1.25)
const X_QM_GRAD_THRESHOLD: f32 = 0.04;

/// `b_gradient_stat` gate for blue-axis-dominant structure (the B twin of
/// `x_heavy`)
const B_GRAD_THRESHOLD: f32 = 0.45;

const X_HEAVY_B_QM_SCALE: u32 = 7;

/// Below this the X scale is 2 and the extra precision only costs rate.
const PIXEL_CHROMACITY_MIN_DISTANCE: f32 = 1.25;

/// A pixel-step level counts once this many pixels pass it; libjxl takes the
/// single worst pixel, which lets one stray edge refine chroma frame-wide.
const PIXEL_CHROMACITY_MIN_PIXELS: u32 = 4;

/// libjxl's `PixelStatsForChromacityAdjustment`: extra X and B quant-scale
/// steps from the pixel-to-pixel steps of X and of B - Y, plus one B step for
/// strongly exposed blue. Thresholds are libjxl's; the step levels need
/// `PIXEL_CHROMACITY_MIN_PIXELS` pixels, exposed blue a single one.
fn pixel_chromacity_steps(xyb: &Image3F) -> (u32, u32) {
    const X_LEVELS: [f32; 3] = [0.015, 0.022, 0.026];
    const B_LEVELS: [f32; 3] = [0.28, 0.33, 0.38];
    let (w, h) = (xyb.xsize(), xyb.ysize());
    if w < 2 || h < 2 {
        return (0, 0);
    }
    let (mut x_counts, mut b_counts, mut exposed_blue) = ([0u32; 3], [0u32; 3], 0u32);
    let rows = xyb
        .plane_data(0)
        .chunks_exact(w)
        .zip(xyb.plane_data(1).chunks_exact(w))
        .zip(xyb.plane_data(2).chunks_exact(w));
    for (((xp, yp), bp), ((xr, yr), br)) in rows.clone().zip(rows.skip(1)) {
        // Carry the left pixel and its B - Y difference between iterations.
        let mut current = xr.iter().zip(yr).zip(br);
        let Some(((&xl, &yl), &bl)) = current.next() else {
            continue;
        };
        let (mut left_x, mut left_b, mut left_diff) = (xl, bl, bl - yl);
        let previous = xp.iter().zip(yp).zip(bp).skip(1);
        for (((&x, &y), &b), ((&px, &py), &pb)) in current.zip(previous) {
            let step_x = (x - left_x).abs().max((x - px).abs());
            let diff_b = b - y;
            let step_b = (diff_b - left_diff).abs().max((diff_b - (pb - py)).abs());
            for (count, level) in x_counts.iter_mut().zip(X_LEVELS) {
                *count += u32::from(step_x >= level);
            }
            for (count, level) in b_counts.iter_mut().zip(B_LEVELS) {
                *count += u32::from(step_b > level);
            }
            // The gradient factor is nonnegative, so negative exposure never passes.
            let step = (b - left_b).abs() + (b - pb).abs();
            exposed_blue += u32::from((b - y * 1.2) * step >= 0.13);
            left_x = x;
            left_b = b;
            left_diff = diff_b;
        }
        if x_counts[2] >= PIXEL_CHROMACITY_MIN_PIXELS
            && b_counts[2] >= PIXEL_CHROMACITY_MIN_PIXELS
            && exposed_blue > 0
        {
            return (3, 3);
        }
    }
    // Counts shrink level by level, so the passed levels are a prefix.
    let level = |counts: [u32; 3]| {
        counts
            .iter()
            .filter(|&&n| n >= PIXEL_CHROMACITY_MIN_PIXELS)
            .count() as u32
    };
    let b_steps = match level(b_counts) {
        0 => 0,
        l => l - 1 + u32::from(exposed_blue > 0),
    };
    (level(x_counts), b_steps)
}

/// Source-domain companion to the saturation detector. Sparse luminance
/// impulses can dominate its gradient statistics while diffuse chroma remains
/// visible between them. Crosses at one and four pixels estimate point prominence
/// without running the dot fitter or quantizing proxy DCTs.
#[derive(Default, Clone, Copy, Debug)]
struct ChromaContentStats {
    saturation: f32,
    dark_fraction: f32,
    point_fraction: f32,
    point_energy: f32,
    sharp_energy: f32,
}

impl ChromaContentStats {
    fn diffuse_point_chroma(self) -> bool {
        self.saturation >= 0.005
            && self.dark_fraction >= 0.5
            && (0.003..0.05).contains(&self.point_fraction)
            && self.point_energy >= 0.35
            && self.sharp_energy >= 0.12
    }
}

/// Sparse luminance points with diffuse color need extra chroma precision.
/// Keep that protection as distance decreases through the HQ table boundary.
fn apply_point_chroma_policy(
    ctx: &EncodingContext,
    distp: &DistanceParams,
    content: ChromaContentStats,
    is_achromatic: bool,
) {
    // X-heavy content already has a separate chroma precision policy.
    let point_chroma = ctx.speed.effort().chroma_policy
        && !is_achromatic
        && content.diffuse_point_chroma()
        && !ctx.x_heavy();
    ctx.set_point_chroma(point_chroma);
    if point_chroma && distp.distance < 2.25 {
        let base_x = distp.x_qm_scale.max(ctx.x_qm_scale_floor());
        // The default X scale already includes this step below the HQ cutoff.
        let extra_x = u32::from(distp.distance >= X_QM_REFINE_MAX_DISTANCE);
        ctx.raise_x_qm_scale_floor(base_x.saturating_add(extra_x));
        ctx.raise_b_qm_scale(ctx.b_qm_scale().saturating_add(1));
    }
}

fn chroma_content_stats(xyb: &Image3F) -> ChromaContentStats {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let (mut sum, mut n, mut dark, mut points, mut interior) = (0.0, 0u64, 0u64, 0u64, 0u64);
    let (mut point_energy, mut contrast_energy) = (0.0f64, 0.0f64);
    let (mut sharp_energy, mut sharp_contrast) = (0.0f64, 0.0f64);
    for py in (0..h).step_by(4) {
        let xr = xyb.plane_row(0, py);
        let yr = xyb.plane_row(1, py);
        let br = xyb.plane_row(2, py);
        for px in (0..w).step_by(4) {
            let y = yr[px];
            sum += xr[px].abs() + (br[px] - y).abs();
            n += 1;
            dark += u64::from(y < 0.3);
            if px < 4 || py < 4 || px + 4 >= w || py + 4 >= h {
                continue;
            }
            let ring = [
                yr[px - 4],
                yr[px + 4],
                xyb.plane_row(1, py - 4)[px],
                xyb.plane_row(1, py + 4)[px],
            ];
            let sharp_ring = [
                yr[px - 1],
                yr[px + 1],
                xyb.plane_row(1, py - 1)[px],
                xyb.plane_row(1, py + 1)[px],
            ];
            let sharp_background = sharp_ring.iter().sum::<f32>() * 0.25;
            let sharp_peak =
                f64::from((y - sharp_ring.into_iter().fold(f32::NEG_INFINITY, f32::max)).max(0.0));
            let sharp_delta = f64::from(y - sharp_background);
            sharp_energy += sharp_peak * sharp_peak;
            sharp_contrast += sharp_delta * sharp_delta;
            let background = ring.iter().sum::<f32>() * 0.25;
            let peak = y - ring.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let contrast = f64::from(y - background);
            contrast_energy += contrast * contrast;
            let positive = f64::from(peak.max(0.0));
            point_energy += positive * positive;
            points += u64::from(peak > 0.1 && background < 0.35);
            interior += 1;
        }
    }
    ChromaContentStats {
        saturation: if n > 0 { sum / n as f32 } else { 0.0 },
        dark_fraction: if n > 0 { dark as f32 / n as f32 } else { 0.0 },
        point_fraction: if interior > 0 {
            points as f32 / interior as f32
        } else {
            0.0
        },
        sharp_energy: if sharp_contrast > 1e-15 {
            (sharp_energy / sharp_contrast) as f32
        } else {
            0.0
        },
        point_energy: if contrast_energy > 1e-15 {
            (point_energy / contrast_energy) as f32
        } else {
            0.0
        },
    }
}

#[cfg(test)]
fn chroma_saturation_stat(xyb: &Image3F) -> f32 {
    chroma_content_stats(xyb).saturation
}

/// Horizontal opponent-gradient sums on a 1-in-4 row subsample. X and B−Y
/// share one traversal and one set of Y loads. The result is
/// `[sum_x, sum_y_for_x, sum_b_minus_y, sum_y_for_b]`; keeping the two Y
/// reductions explicit makes both classifier ratios independently testable.
/// Achromatic content reads ~0 in both opponent numerators.
pub(crate) type ChromaGradientSumsFn = fn(&[f32], &[f32], &[f32], usize) -> [f32; 4];

#[allow(dead_code)]
pub(crate) fn chroma_gradient_sums_scalar(
    x_plane: &[f32],
    y_plane: &[f32],
    b_plane: &[f32],
    width: usize,
) -> [f32; 4] {
    if width < 2 {
        return [0.0; 4];
    }

    let mut sum_x = 0.0f32;
    let mut sum_y_x = 0.0f32;
    let mut sum_by = 0.0f32;
    let mut sum_y_b = 0.0f32;
    let rows = x_plane
        .chunks_exact(width)
        .zip(y_plane.chunks_exact(width))
        .zip(b_plane.chunks_exact(width))
        .step_by(4);
    for ((xr, yr), br) in rows {
        for ((x, l), b) in xr
            .array_windows::<2>()
            .zip(yr.array_windows::<2>())
            .zip(br.array_windows::<2>())
        {
            let dy = (l[1] - l[0]).abs();
            sum_x += (x[1] - x[0]).abs();
            sum_y_x += dy;
            sum_by += ((b[1] - l[1]) - (b[0] - l[0])).abs();
            sum_y_b += dy;
        }
    }
    [sum_x, sum_y_x, sum_by, sum_y_b]
}

pub(crate) fn selected_chroma_gradient_sums_fn() -> ChromaGradientSumsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |x, y, b, width| unsafe { crate::neon::chroma_gradient_sums_neon(x, y, b, width) }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |x, y, b, width| unsafe { crate::avx::chroma_gradient_sums_avx2(x, y, b, width) };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::chroma_gradient_sums_wasm;
    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
        all(target_arch = "aarch64", feature = "neon")
    )))]
    chroma_gradient_sums_scalar
}

/// Share of high-frequency signal on the blue-yellow axis: mean |∇(B−Y)|
/// over mean |∇Y| on a 1-in-4 row subsample. Content whose structure lives
/// in the B channel (saturated blue detail) is nearly invisible to the
/// luma-driven masking field, exactly like the X statistic
/// covers on the red-green axis.
/// `[x_gradient_stat, b_gradient_stat, chroma_texture_stat]` from one
/// gradient pass. The third is the relative chroma texture: mean of
/// |∇X| + |∇(B−Y)| per sampled pixel pair over the frame's mean |X| + |B−Y|
/// (`saturation_stat`). High on thin chromatic detail over low-chroma
/// ground, low on saturated smooth subjects however colorful.
fn chroma_gradient_stats(ctx: &EncodingContext, xyb: &Image3F, saturation_stat: f32) -> [f32; 3] {
    let sums = (ctx.chroma_gradient_sums)(
        xyb.plane_data(0),
        xyb.plane_data(1),
        xyb.plane_data(2),
        xyb.xsize(),
    );
    chroma_stats_from_sums(sums, xyb.xsize(), xyb.ysize(), saturation_stat)
}

fn chroma_stats_from_sums(
    [sum_x, sum_y_x, sum_by, sum_y_b]: [f32; 4],
    width: usize,
    height: usize,
    saturation_stat: f32,
) -> [f32; 3] {
    // The kernel visits rows 0, 4, 8, ... and `width - 1` pairs per row.
    let pairs = (height.div_ceil(4) * width.saturating_sub(1)) as f32;
    [
        if sum_y_x > f32::EPSILON {
            sum_x / sum_y_x
        } else {
            0.0
        },
        if sum_y_b > f32::EPSILON {
            sum_by / sum_y_b
        } else {
            0.0
        },
        if saturation_stat > 1e-9 && pairs > 0.0 {
            (sum_x + sum_by) / pairs / saturation_stat
        } else {
            0.0
        },
    ]
}

fn snap_achromatic_xyb(xyb: &mut Image3F) {
    for y in 0..xyb.ysize() {
        let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
        x_row.fill(0.0);
        b_row.copy_from_slice(y_row);
    }
}

fn to_xyb_image(ctx: &EncodingContext, scratch: &mut CoderScratch, mut linear: Image3F) -> Image3F {
    crate::xyb::to_xyb_in_place_with_fn(
        ctx.to_xyb_band,
        &ctx.xyb,
        &mut linear,
        &ctx.thread_pool,
        scratch,
    );
    linear
}

/// Pre-invert Gaborish so the decoder's forward pass reproduces `xyb`.
fn gaborize(xyb: &mut Image3F, distp: &DistanceParams) {
    if distp.gab_enabled {
        crate::gaborish::gaborish_inverse(xyb, 0.990_851_1);
    }
}

/// Largest stored DC level kept clear of the `i16` limit, with room for the
/// chroma-from-luma terms and the finer DC-step candidates.
const DC_LEVEL_LIMIT: f32 = 16_000.0;

/// Non-XYB planes hold the image's own samples, which float input may take far
/// past [0, 1]. Coarsen a plane's DC step only as far as its stored levels
/// need to stay representable.
fn dc_steps_in_range(base: [f32; 3], scale_dc: f32, coded: &Image3F) -> [f32; 3] {
    let peak = (0..3)
        .flat_map(|c| coded.plane_data(c).iter())
        .fold(0.0f32, |m, &v| m.max(v.abs()));
    std::array::from_fn(|c| {
        let level = peak * crate::quant_weights::INV_DC_QUANT[c] * scale_dc / base[c];
        if level > DC_LEVEL_LIMIT {
            base[c] * level / DC_LEVEL_LIMIT
        } else {
            base[c]
        }
    })
}

/// Returns DC-group and AC-section bits, excluding the global headers, so
/// sampled payload can be scaled separately from the frame's fixed costs.
#[allow(clippy::too_many_arguments)]
fn encode_frame_core(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    opsin: Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    #[cfg(feature = "splines")] splines: Option<&crate::splines::SplineSet>,
    writer: &mut BitWriter,
) -> Result<usize, EncodeError> {
    let num_threads = ctx.thread_pool.num_threads();
    let dim = ImageDim::new(opsin.xsize(), opsin.ysize());
    // Grain is measured on the source XYB before any frame processing; only
    // displayed frames carry it (an atlas frame's header has no flags).
    let mut opsin = opsin;
    let noise = if matches!(
        frame_kind,
        VarDctFrameKind::Regular | VarDctFrameKind::Patched(_)
    ) {
        crate::noise::NoiseParams::resolve(ctx.grain, ctx, scratch, distance, &mut opsin)
    } else {
        None
    };
    let mut distp = compute_distance_params(distance);
    if ctx.coding.is_xyb() {
        if ctx.x_heavy() && distp.x_qm_scale == 2 {
            distp.x_qm_scale = 3;
        }
        distp.x_qm_scale = distp.x_qm_scale.max(ctx.x_qm_scale_floor());
    } else {
        // Not serialized: the decoder's X multiplier is 1.
        distp.x_qm_scale = 2;
        distp.dc_step = dc_steps_in_range(ctx.coding.dc_step_base(), distp.scale_dc, &opsin);
    }

    // Progressive lossy splits each quantized AC coeff across `num_passes`
    // passes by a decreasing per-pass shift (last = 0). The decoder reconstructs
    // C = sum_p (sent_p << shift_p) (jxl-vardct hf_coeff.rs:185,191).
    let num_passes = coeff_shifts.len();
    // Progressive DC: the DC image goes out as its own squeezed DC frame,
    // which this frame references instead of coding a DC stream.
    let use_dc_frame = ctx.progressive_dc
        && matches!(
            frame_kind,
            VarDctFrameKind::Regular | VarDctFrameKind::Patched(_)
        )
        && alpha.is_none()
        && ctx.coding.is_xyb()
        && crate::lossless::dc_frame_supported(dim.xsize_blocks, dim.ysize_blocks);

    let num_sections = 2 + dim.num_dc_groups + num_passes * dim.num_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // Phase 1: set up every DC group, then encode every AC group. Keeping the
    // two apart lets AC groups from all DC groups share one steal_map instead of
    // one serialized burst per DC group. Merges back in (dc, gix) order, so the
    // output stays bit-identical to single-threaded.
    let group_coords: Vec<(usize, usize)> = (0..dim.ysize_dc_groups)
        .flat_map(|gy| (0..dim.xsize_dc_groups).map(move |gx| (gx, gy)))
        .collect();
    let opsin_image = opsin;
    let opsin = &opsin_image;

    // Split the thread budget across nesting levels: `outer` lanes steal DC
    // groups, each parallelizing its AC-strategy bands with the remainder, so
    // the setup phase saturates all cores even with few (large) DC groups.
    let outer = group_coords.len().min(num_threads.max(1));
    let setup_budget = num_threads.max(1).div_ceil(outer);
    let setups: Vec<Result<(DcGroupData, usize, usize), EncodeError>> =
        ctx.thread_pool
            .steal_map(scratch, group_coords.len(), |i, scratch| {
                let (dc_gx, dc_gy) = group_coords[i];
                setup_dc_group(
                    ctx,
                    scratch,
                    opsin,
                    &dim,
                    &distp,
                    dc_gx,
                    dc_gy,
                    setup_budget,
                )
            });
    let setups: Vec<(DcGroupData, usize, usize)> = setups.into_iter().collect::<Result<_, _>>()?;

    let mut dc_datas: Vec<DcGroupData> = Vec::with_capacity(setups.len());
    let mut ac_tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (dc_idx, (dc_data, gxs, gys)) in setups.into_iter().enumerate() {
        ac_tasks.extend((0..gxs * gys).map(|g| (dc_idx, g % gxs, g / gxs)));
        dc_datas.push(dc_data);
    }

    crate::ac_strategy::account_matrix_headers(
        ctx,
        scratch,
        opsin,
        distance,
        distp.scale,
        distp.x_qm_scale,
        &group_coords,
        &mut dc_datas,
    );
    for dc in dc_datas.iter_mut() {
        dc.block_quant_field = None;
    }

    // Layer residuals and progressive/DC frames retain the EPF schedule.
    let fit_epf = ctx.speed.effort().epf_fit
        && ctx.coding.is_xyb()
        && matches!(frame_kind, VarDctFrameKind::Regular)
        && ctx.frame_role == FrameRole::Last
        && noise.is_none()
        && !use_dc_frame
        && num_passes == 1
        && (0.5..=24.0).contains(&distance)
        && dim.xsize >= 16
        && dim.ysize >= 16;
    #[cfg(feature = "splines")]
    let fit_epf = fit_epf && splines.is_none();
    let mut epf_recon = fit_epf.then(|| Image3F::new(dim.xsize, dim.ysize));

    // Per-image quant-field threshold for the fine AC block-context layout.
    // The median splits the field into halves with genuinely different
    // coefficient statistics; whether any split is *kept* is decided later
    // from real token stats, so a useless threshold costs nothing.
    let qf_threshold = ac_qf_threshold(&dc_datas);

    // The CfL slopes were estimated on fixed 8x8 DCTs so the transform search
    // could price candidates under them; the final layout and quant field
    // are known now, so re-estimate them on the coefficients that get coded.
    if crate::color_correlation::cfl_slope_search(ctx) {
        let dc_ref = &dc_datas;
        let maps = ctx
            .thread_pool
            .steal_map(scratch, group_coords.len(), |i, scratch| {
                let (dc_gx, dc_gy) = group_coords[i];
                let (coeffs, gather) = scratch.ac_group.transform_buffers();
                crate::color_correlation::refit_cmap_on_layout(
                    ctx,
                    opsin,
                    &dc_ref[i],
                    dc_gx * K_DC_GROUP_DIM / K_BLOCK_DIM,
                    dc_gy * K_DC_GROUP_DIM / K_BLOCK_DIM,
                    distp.scale,
                    distp.distance,
                    distp.x_qm_scale,
                    coeffs,
                    gather,
                    &mut scratch.cfl_rdo,
                )
            });
        for (dc, (ytox, ytob)) in dc_datas.iter_mut().zip(maps) {
            dc.ytox_map = ytox;
            dc.ytob_map = ytob;
        }
    }

    let dc_ref = &dc_datas;
    let natural_orders = crate::coeff_order::CoeffOrders::natural();
    // Tally coefficient positions only when the refine pass will run to consume
    // the derived orders: pass one alone can never adopt them (its tokens are
    // already written on the natural scan), so the work would be wasted and its
    // mere presence perturbs codegen enough to shift a few borderline
    // quantizer roundings on paths that should be untouched.
    let want_order_stats = ctx.speed.effort().coeff_orders;
    // Passes before the refined one only feed statistics and token prices,
    // which describe whole coefficients: they tokenize as one pass, and only
    // the refined pass splits the coefficients across the frame's passes.
    let (analysis_passes, analysis_shifts): (usize, &[u32]) = if want_order_stats {
        (1, &[0])
    } else {
        (num_passes, coeff_shifts)
    };
    let results = ctx
        .thread_pool
        .steal_map(scratch, ac_tasks.len(), |t, scratch| {
            let (dc_idx, gx, gy) = ac_tasks[t];
            let (dc_gx, dc_gy) = group_coords[dc_idx];
            let (p, local, source_b, stats, capture) = process_ac_group(
                ctx,
                scratch,
                opsin,
                &dim,
                &distp,
                &dc_ref[dc_idx],
                analysis_passes,
                analysis_shifts,
                dc_gx,
                dc_gy,
                gx,
                gy,
                0,
                None,
                &natural_orders,
                &[],
                want_order_stats || use_dc_frame || fit_epf,
                qf_threshold,
                // Without custom orders this pass is the final quantization.
                fit_epf && !want_order_stats,
                None,
            );
            (
                (dc_idx, gx, gy, p, local, source_b, stats),
                capture.map(|(image, _)| image),
            )
        });

    let mut all_pending: Vec<PendingAcGroup> = Vec::with_capacity(results.len());
    // Adopt the first group's buffers instead of allocating an empty tally.
    // Fast and progressive frames never allocate aggregate order statistics.
    let mut order_stats: Option<crate::coeff_order::OrderStats> = None;
    for ((dc_idx, gx, gy, p, local, source, stats), capture) in results {
        if let (Some(target), Some(image)) = (epf_recon.as_mut(), capture) {
            paste_group(target, &dim, p.group_idx, &image);
        }
        merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
        if let Some(source) = source {
            let dc = &mut dc_datas[dc_idx];
            let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
            let ox = gx * K_GROUP_DIM_IN_BLOCKS;
            let oy = gy * K_GROUP_DIM_IN_BLOCKS;
            for (target, source) in [
                (&mut dc.source_dc_y, &source.y),
                (&mut dc.source_dc_b, &source.b),
                (&mut dc.source_dc_x, &source.x),
            ] {
                let target = target.get_or_insert_with(|| crate::image::Plane::new(w, h));
                for y in 0..source.ysize() {
                    target.row_mut(oy + y)[ox..ox + source.xsize()].copy_from_slice(source.row(y));
                }
            }
        }
        all_pending.push(p);
        if let Some(s) = stats {
            if let Some(total) = &mut order_stats {
                total.merge(&s);
            } else {
                order_stats = Some(s);
            }
        }
    }

    // The EPF fit adds the decoded-minus-source DC to its reconstruction.
    // Other DC tools key on the source planes, so only a copy outlives the
    // first pass when nothing else asked for them.
    let mut epf_source_dc = None;
    if fit_epf && !want_order_stats && !use_dc_frame {
        epf_source_dc = Some(assemble_source_dc(&dc_datas, &group_coords, &dim));
        for dc in &mut dc_datas {
            dc.source_dc_b = None;
            dc.source_dc_y = None;
            dc.source_dc_x = None;
        }
    }

    // DC-level chroma-from-luma. The slope has to be folded into the DC
    // quantizer's rounding rather than subtracted from the stored integers
    // afterwards: `round(v) - round(s*y)` carries up to a full step of B DC
    // error against `round(v - s*y)`'s half step, and that extra chroma noise
    // costs far more quality than the slope saves in rate. So the search rides
    // on the rerank pass, which re-quantizes every DC group regardless -- and
    // is skipped entirely when that pass does not run.
    // Custom coefficient orders, derived from the first pass's nonzero tallies.
    let mut coeff_orders = natural_orders;
    let mut order_alternatives = Vec::new();

    let mut ytob_dc = 0i32;
    if want_order_stats {
        let scan_moved = order_stats.as_ref().map_or(0.0, |stats| {
            crate::coeff_order::derive_orders(stats, &mut coeff_orders)
        });
        // Keep the incumbent RDOQ scan: the final race changes coding only,
        // so all arms transmit exactly the same quantized coefficients.
        if ctx.speed.effort().coeff_order_entropy_race() && num_passes == 1 {
            if let Some(stats) = &order_stats {
                let mut proposed = crate::coeff_order::CoeffOrders::natural();
                crate::coeff_order::propose_orders(stats, &mut proposed);
                if proposed.used_mask != 0 && proposed.orders != coeff_orders.orders {
                    order_alternatives.push(proposed);
                }
            }
            if coeff_orders.used_mask != 0 {
                order_alternatives.push(crate::coeff_order::CoeffOrders::natural());
            }
        }
        drop(order_stats);
        ytob_dc = if use_dc_frame {
            0
        } else if ctx.speed.effort().dc_cfl_weighted() {
            crate::color_correlation::choose_ytob_dc_weighted(
                &dc_datas,
                ctx,
                distp.scale_dc,
                distp.dc_step,
                &mut scratch.dc_cfl_cur,
                &mut scratch.dc_cfl_prev,
            )
        } else {
            let candidate = choose_ytob_dc(
                &dc_datas,
                ctx.fill_ytob_row,
                ctx.accumulate_ytob_weights,
                ctx.fill_ytob_residuals,
                &mut scratch.dc_cfl_cur,
                &mut scratch.dc_cfl_prev,
                distp.dc_step,
                ctx.cfl_frame().color_factor,
            );
            crate::color_correlation::validate_ytob_dc(
                &dc_datas,
                candidate,
                distp.scale_dc,
                distp.dc_step,
                ctx.cfl_frame(),
                ctx.quantize_dc_cfl,
            )
        };
        if !use_dc_frame {
            distp.dc_step = crate::color_correlation::choose_dc_steps(
                &dc_datas,
                ctx,
                distp.scale_dc,
                ytob_dc,
                distp.dc_step,
                ctx.cfl_frame(),
            );
        }
        if !ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
            if fit_epf && epf_source_dc.is_none() {
                epf_source_dc = Some(assemble_source_dc(&dc_datas, &group_coords, &dim));
            }
            for dc in &mut dc_datas {
                dc.source_dc_b = None;
                dc.source_dc_y = None;
                dc.source_dc_x = None;
            }
        }
        // The first pass's tokens follow the natural scans. Once the custom
        // orders move most of the walk, they no longer describe what the
        // contexts of the refine pass will see, and the quantizer would price
        // its choices on them: tokenize again on the orders it codes with.
        if ctx.speed.effort().rdoq_fresh_prices() || scan_moved >= RDOQ_FRESH_PRICES_MIN_SCAN_MOVE {
            // These streams will be replaced; release their token buffers
            // before allocating the reordered streams for the whole frame.
            all_pending.clear();
            let dc_ref = &dc_datas;
            let reordered = ctx
                .thread_pool
                .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    let (dc_gx, dc_gy) = group_coords[dc_idx];
                    process_ac_group(
                        ctx,
                        scratch,
                        opsin,
                        &dim,
                        &distp,
                        &dc_ref[dc_idx],
                        analysis_passes,
                        analysis_shifts,
                        dc_gx,
                        dc_gy,
                        gx,
                        gy,
                        ytob_dc,
                        None,
                        &coeff_orders,
                        &[],
                        false,
                        qf_threshold,
                        false,
                        None,
                    )
                    .0
                });
            all_pending.extend(reordered);
        }
        let prices = {
            let provisional_code = crate::entropy::optimize_entropy_code_ac_streams(
                all_pending.iter().map(|pg| pg.tokens[0].as_slice()),
                crate::ac_context::K_NUM_FINE_AC_CONTEXTS,
                &mut scratch.huffman_pool,
                // No config selection here: these prices feed RDOQ, and even a
                // one-token nudge can push the clustering off a knife-edge merge.
                // Selection applies to the final codes only.
                false,
                ctx.speed,
                Some(&ctx.thread_pool),
            );
            crate::entropy::FrozenTokenPrices::new(&provisional_code)
        };
        // The closed loop's reference and verification encodes are pass 2
        // itself (same RDOQ, prices and orders), so when it runs its final
        // groups replace the refined pass.
        all_pending.clear();
        #[cfg(feature = "extraslow")]
        let loop_groups =
            if ctx.speed.effort().aq_closed_loop() && ctx.coding.is_xyb() && num_passes == 1 {
                run_quant_field_loop(
                    ctx,
                    scratch,
                    opsin,
                    &dim,
                    &distp,
                    &mut dc_datas,
                    &ac_tasks,
                    &group_coords,
                    num_passes,
                    coeff_shifts,
                    &coeff_orders,
                    &order_alternatives,
                    ytob_dc,
                    qf_threshold,
                    &prices,
                )
            } else {
                None
            };
        #[cfg(not(feature = "extraslow"))]
        let loop_groups: Option<(Vec<RefinedAcGroup>, DistanceParams)> = None;
        let loop_groups = loop_groups.map(|(groups, chosen)| {
            distp = chosen;
            groups
        });
        let refined = if let Some(groups) = loop_groups {
            // The closed loop's groups carry no reconstruction capture.
            epf_recon = None;
            groups.into_iter().map(|g| (g, None)).collect::<Vec<_>>()
        } else {
            let dc_ref = &dc_datas;
            ctx.thread_pool
                .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    let (dc_gx, dc_gy) = group_coords[dc_idx];
                    let (p, local, _, _, capture) = process_ac_group(
                        ctx,
                        scratch,
                        opsin,
                        &dim,
                        &distp,
                        &dc_ref[dc_idx],
                        num_passes,
                        coeff_shifts,
                        dc_gx,
                        dc_gy,
                        gx,
                        gy,
                        ytob_dc,
                        Some(&prices),
                        &coeff_orders,
                        &order_alternatives,
                        false,
                        qf_threshold,
                        fit_epf,
                        None,
                    );
                    ((dc_idx, gx, gy, p, local), capture.map(|(image, _)| image))
                })
        };
        for ((dc_idx, gx, gy, p, local), capture) in refined {
            if let (Some(target), Some(image)) = (epf_recon.as_mut(), capture) {
                paste_group(target, &dim, p.group_idx, &image);
            }
            merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
            all_pending.push(p);
        }
    }

    // The retained source DC is invariant under AC reranking. Search against
    // the final Y levels and steps, then quantize X once from that source.
    let ytox_dc = if ctx.coding.is_xyb() && ctx.speed.effort().dc_x_cfl && !use_dc_frame {
        crate::color_correlation::choose_ytox_dc(
            &mut dc_datas,
            distp.scale_dc,
            distp.dc_step,
            ctx.cfl_frame(),
            ytob_dc != 0,
            ctx.quantize_dc_cfl,
        )
    } else {
        0
    };
    if ctx.coding.is_xyb() && ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
        let cfl = crate::color_correlation::CflFrame {
            base_x: crate::color_correlation::y_to_x_ratio(ctx.cfl_frame(), ytox_dc as i8),
            ..ctx.cfl_frame()
        };
        distp.dc_step = dc_rdo::choose_x_dc_step(&mut dc_datas, &distp, cfl, ctx.quantize_dc_cfl);
    }
    // Only DC reconstruction uses the extra X slope. The AC map and the
    // header's base correlations retain the frame's original parameters.
    let dc_cfl = crate::color_correlation::CflFrame {
        base_x: crate::color_correlation::y_to_x_ratio(ctx.cfl_frame(), ytox_dc as i8),
        ..ctx.cfl_frame()
    };
    // Couple B rounding to the final luma error after AC decisions are fixed.
    // Flattening and Haar selection must use that same target as well.
    let compensate_b = ctx.coding.is_xyb()
        && ctx.speed.effort().chroma_dc_squeeze
        && !use_dc_frame
        && compensate_b_dc(
            &mut dc_datas,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.compensate_b,
        );
    if ctx.coding.is_xyb() && !use_dc_frame {
        flatten_chroma_dc(
            opsin,
            &dim,
            &mut dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
        )?;
    }
    if ctx.coding.is_xyb() && ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
        choose_chroma_dc_coding(
            &dim,
            &mut dc_datas,
            &mut distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
            ctx.dc_rows.adjust_b,
        )?;
    }
    if ctx.coding.is_xyb() && ctx.speed.effort().dc_rdoq && !use_dc_frame {
        dc_rdo::refine_dc_rounding(
            &mut dc_datas,
            &distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
            &mut scratch.dc_predictor,
        )?;
    }
    let skip_dc_smoothing = !use_dc_frame
        && skip_dc_smoothing(
            &ctx.thread_pool,
            scratch,
            opsin,
            &dim,
            &dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.reconstruct,
        )?;
    let mut epf_sharpness = epf_sharpness_id(distance, ctx.screen_content());
    if let Some(mut recon) = epf_recon {
        let decoded_dc = dc_smoothing::decoded_dc(
            &dim,
            &dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.reconstruct,
            skip_dc_smoothing,
        )?;
        let source_dc =
            epf_source_dc.unwrap_or_else(|| assemble_source_dc(&dc_datas, &group_coords, &dim));
        crate::epf_filter::add_dc_error(
            ctx,
            scratch,
            &mut recon,
            &decoded_dc,
            &source_dc,
            &dc_datas,
            &group_coords,
        );
        let source = distp
            .gab_enabled
            .then(|| crate::epf_filter::gaborish(opsin));
        let source = source.as_ref().unwrap_or(opsin);
        if distp.gab_enabled {
            crate::epf_filter::gaborish_in_place(&mut recon);
        }
        let blocks = crate::epf_filter::FrameBlocks::new(
            &dc_datas,
            &group_coords,
            dim.xsize_blocks,
            dim.ysize_blocks,
            distp.global_scale,
        );
        let chosen = crate::epf_filter::fit(
            ctx,
            scratch,
            source,
            &recon,
            &blocks,
            distance,
            crate::epf_filter::Choice {
                iters: distp.epf_iters,
                quant_mul: distp.epf_quant_mul,
                pass0: distp.epf_pass0_scale,
                sharpness: epf_sharpness,
            },
        );
        distp.epf_iters = chosen.iters;
        distp.epf_pass0_scale = chosen.pass0;
        distp.epf_quant_mul = chosen.quant_mul;
        epf_sharpness = chosen.sharpness;
    }
    // Pixel-domain decisions are finished; entropy coding only reads DC data
    // and tokens. Release the full-resolution planes before its working sets.
    drop(opsin_image);

    // Phase 2: build adaptive DC entropy code from all DC + AC-metadata tokens.
    // Per-leaf DC predictor selection.
    let effort = ctx.speed.effort();
    // The learned tree splits on weighted-predictor errors.
    let dc_predictor_search = effort.dc_predictor_search || effort.dc_learned_tree;
    let token_groups = ctx
        .thread_pool
        .steal_map(scratch, dc_datas.len(), |i, _scratch| {
            let mut props = Vec::new();
            let predictor = if dc_predictor_search {
                &DC_PREDICTOR_WEIGHTED
            } else {
                &DC_PREDICTOR_GRADIENT
            };
            // Both residual candidates use the same WP state and context.
            let (wp, grad) = if use_dc_frame {
                (Vec::new(), Vec::new())
            } else {
                collect_dc_tokens_with_gradient(
                    &dc_datas[i],
                    predictor,
                    &mut props,
                    effort.dc_learned_tree,
                    dc_predictor_search,
                )
            };
            let mut meta_props = Vec::new();
            let meta = collect_ac_metadata_tokens(
                &dc_datas[i],
                &mut meta_props,
                epf_sharpness,
                effort.dc_learned_tree,
            );
            (wp, props, grad, meta, meta_props)
        });
    let mut wp_tokens_per_group = Vec::with_capacity(token_groups.len());
    let mut props_per_group = Vec::with_capacity(token_groups.len());
    let mut grad_tokens_per_group = Vec::with_capacity(token_groups.len());
    let mut meta_tokens_per_group = Vec::with_capacity(token_groups.len());
    let mut meta_props_per_group = Vec::with_capacity(token_groups.len());
    for (wp, props, grad, meta, meta_props) in token_groups {
        wp_tokens_per_group.push(wp);
        props_per_group.push(props);
        grad_tokens_per_group.push(grad);
        meta_tokens_per_group.push(meta);
        meta_props_per_group.push(meta_props);
    }

    let dc_gradient = if !dc_predictor_search {
        DC_PREDICTOR_GRADIENT
    } else {
        choose_dc_predictors(
            &wp_tokens_per_group,
            &grad_tokens_per_group,
            &mut scratch.dc_predictor,
        )
    };
    // Arm A: the static tree with per-leaf predictor flips.
    let dc_tokens_static: Vec<Vec<Token>> = if !dc_predictor_search {
        std::mem::take(&mut wp_tokens_per_group)
    } else {
        wp_tokens_per_group
            .iter()
            .zip(&grad_tokens_per_group)
            .map(|(wp, grad)| {
                wp.iter()
                    .zip(grad)
                    .map(|(w, g)| {
                        if dc_gradient[w.context as usize] {
                            *g
                        } else {
                            *w
                        }
                    })
                    .collect()
            })
            .collect()
    };
    let code_static = crate::entropy::optimize_entropy_code_ac_streams(
        dc_tokens_static
            .iter()
            .map(Vec::as_slice)
            .chain(meta_tokens_per_group.iter().map(Vec::as_slice)),
        K_NUM_DC_CONTEXTS,
        &mut scratch.huffman_pool,
        effort.huc_select,
        ctx.speed,
        Some(&ctx.thread_pool),
    );
    let tree_static = DcTreeChoice::Static(dc_gradient);

    // `dc_code_refined`: the Slow arm battle already priced both finalists on
    // ANS-refined codes, so the winner needs no second refinement.
    let (dc_tokens_per_group, meta_tokens_per_group, mut dc_code_owned, dc_tree, dc_code_refined) =
        if !effort.dc_learned_tree || use_dc_frame {
            (
                dc_tokens_static,
                meta_tokens_per_group,
                code_static,
                tree_static,
                false,
            )
        } else {
            // Arm B: a per-image learned tree over WP error and channel, with its own
            // per-leaf predictor choice. Leaf renumbering shifts the metadata contexts,
            // so those tokens are remapped alongside.
            let learned = crate::dc_tree::learn_dc_tree(
                dim.num_dc_groups,
                &wp_tokens_per_group,
                &grad_tokens_per_group,
                &props_per_group,
                &meta_tokens_per_group,
                &meta_props_per_group,
            );
            let dc_tokens_learned: Vec<Vec<Token>> = wp_tokens_per_group
                .iter()
                .zip(&grad_tokens_per_group)
                .zip(&props_per_group)
                .map(|((wp, grad), props)| {
                    wp.iter()
                        .zip(grad)
                        .zip(props)
                        .map(|((w, g), &p)| {
                            let ctx = learned.dc_context[p as usize];
                            let value = if learned.leaf_gradient[ctx as usize] {
                                g.value
                            } else {
                                w.value
                            };
                            Token::new(u32::from(ctx), value)
                        })
                        .collect()
                })
                .collect();
            let meta_tokens_learned: Vec<Vec<Token>> = meta_tokens_per_group
                .iter()
                .zip(&meta_props_per_group)
                .map(|(group, props)| {
                    group
                        .iter()
                        .zip(props)
                        .map(|(t, &p)| {
                            let slot = ((t.context as usize) << 10) | (p & 1023) as usize;
                            Token::new(u32::from(learned.meta_context[slot]), t.value)
                        })
                        .collect()
                })
                .collect();
            let code_learned = crate::entropy::optimize_entropy_code_ac_streams(
                dc_tokens_learned
                    .iter()
                    .map(Vec::as_slice)
                    .chain(meta_tokens_learned.iter().map(Vec::as_slice)),
                learned.num_contexts,
                &mut scratch.huffman_pool,
                effort.huc_select,
                ctx.speed,
                Some(&ctx.thread_pool),
            );

            // Price both arms end to end: serialized tree + entropy-code header
            // (measured by writing them) plus the payload estimate, and require the
            // learned arm to win by a margin. The margin absorbs the small estimator
            // noise so near-ties keep the battle-tested static tree.
            let payload = |dc: &[Vec<Token>],
                           meta: &[Vec<Token>],
                           code: &crate::entropy::OwnedEntropyCode| {
                crate::entropy::estimate_ac_plain_bits(
                    dc.iter()
                        .map(Vec::as_slice)
                        .chain(meta.iter().map(Vec::as_slice)),
                    code,
                )
            };
            let header = |tree: &DcTreeChoice, code: &EntropyCode, scratch: &mut CoderScratch| {
                let mut w = BitWriter::new();
                match tree {
                    DcTreeChoice::Static(grad) => {
                        write_context_tree(
                            dim.num_dc_groups,
                            grad,
                            &mut scratch.huffman_pool,
                            &mut w,
                        );
                    }
                    DcTreeChoice::Learned(tokens) => {
                        write_tree_tokens(tokens, &mut scratch.huffman_pool, &mut w);
                    }
                }
                write_entropy_code(code, &mut scratch.huffman_pool, &mut w);
                w.bits_written() as u64
            };
            const LEARNED_TREE_MARGIN_BITS: u64 = 64;
            let tree_learned = DcTreeChoice::Learned(learned.tokens);
            let cost_static = header(&tree_static, &code_static.as_ref(), scratch)
                + payload(&dc_tokens_static, &meta_tokens_per_group, &code_static);
            let cost_learned = header(&tree_learned, &code_learned.as_ref(), scratch)
                + payload(&dc_tokens_learned, &meta_tokens_learned, &code_learned);

            // Arm C (Slow): the DC side learned over the full modular property set
            // and every predictor, as the lossless path does.
            let ma_arm = ctx
                .speed
                .effort()
                .ma_dc_tree
                .then(|| {
                    learn_ma_dc_arm(
                        ctx,
                        scratch,
                        &dc_datas,
                        &meta_tokens_per_group,
                        &meta_props_per_group,
                    )
                })
                .flatten()
                .and_then(|(variants, meta_tokens, num_contexts)| {
                    variants
                        .into_iter()
                        .map(|(dc_tokens, tree_tokens)| {
                            let streams = || {
                                dc_tokens
                                    .iter()
                                    .map(Vec::as_slice)
                                    .chain(meta_tokens.iter().map(Vec::as_slice))
                            };
                            let mut code = crate::entropy::optimize_entropy_code_ac_streams(
                                streams(),
                                num_contexts,
                                &mut scratch.huffman_pool,
                                true,
                                ctx.speed,
                                Some(&ctx.thread_pool),
                            );
                            // The prefix-cost clustering collapses a context-rich
                            // tree (59 contexts -> 3 histograms on kodim20); price
                            // the arm on the code the final ANS refinement would
                            // produce.
                            crate::entropy::refine_ans_clusters(
                                &mut code,
                                streams(),
                                crate::entropy::AnsRefinement::Slow,
                                &ctx.thread_pool,
                                scratch,
                            );
                            let tree = DcTreeChoice::Learned(tree_tokens);
                            let cost = header(&tree, &code.as_ref(), scratch)
                                + payload(&dc_tokens, &meta_tokens, &code);
                            (cost, dc_tokens, code, tree)
                        })
                        .min_by_key(|v| v.0)
                        .map(|(cost, dc_tokens, code, tree)| {
                            (cost, dc_tokens, meta_tokens, code, tree)
                        })
                });

            let use_learned = cost_learned + LEARNED_TREE_MARGIN_BITS < cost_static;
            let best = if use_learned {
                (
                    dc_tokens_learned,
                    meta_tokens_learned,
                    code_learned,
                    tree_learned,
                )
            } else {
                (
                    dc_tokens_static,
                    meta_tokens_per_group,
                    code_static,
                    tree_static,
                )
            };
            match ma_arm {
                Some((cost, dc_tokens, meta_tokens, code, tree)) => {
                    // Refine the incumbent too, so both arms are priced on the
                    // codes the final refinement would write.
                    let (best_dc, best_meta, mut best_code, best_tree) = best;
                    crate::entropy::refine_ans_clusters(
                        &mut best_code,
                        best_dc
                            .iter()
                            .map(Vec::as_slice)
                            .chain(best_meta.iter().map(Vec::as_slice)),
                        crate::entropy::AnsRefinement::Slow,
                        &ctx.thread_pool,
                        scratch,
                    );
                    let best_cost = header(&best_tree, &best_code.as_ref(), scratch)
                        + payload(&best_dc, &best_meta, &best_code);
                    if cost + LEARNED_TREE_MARGIN_BITS < best_cost {
                        (dc_tokens, meta_tokens, code, tree, true)
                    } else {
                        (best_dc, best_meta, best_code, best_tree, true)
                    }
                }
                None => (best.0, best.1, best.2, best.3, false),
            }
        };
    drop(wp_tokens_per_group);
    drop(grad_tokens_per_group);
    drop(props_per_group);
    drop(meta_props_per_group);
    let ans_refinement = if effort.ans_refine {
        Some(crate::entropy::AnsRefinement::slow_for_speed(ctx.speed))
    } else if effort.ans_fast_refine {
        Some(crate::entropy::AnsRefinement::Fast {
            recluster: (distance >= 3.0 && effort.ans_recluster_coarse) || effort.ans_recluster,
        })
    } else {
        None
    };
    if let Some(refinement) = ans_refinement.filter(|_| !dc_code_refined) {
        crate::entropy::refine_ans_clusters(
            &mut dc_code_owned,
            dc_tokens_per_group
                .iter()
                .map(Vec::as_slice)
                .chain(meta_tokens_per_group.iter().map(Vec::as_slice)),
            refinement,
            &ctx.thread_pool,
            scratch,
        );
    } else if dc_code_refined && ctx.speed.effort().ans_exhaustive() {
        crate::entropy::refine_ans_entropy_exhaustive(
            &mut dc_code_owned,
            dc_tokens_per_group
                .iter()
                .map(Vec::as_slice)
                .chain(meta_tokens_per_group.iter().map(Vec::as_slice)),
            &ctx.thread_pool,
            scratch,
        );
    }
    let (mut dc_tree, mut dc_tokens_per_group, mut meta_tokens_per_group) =
        (dc_tree, dc_tokens_per_group, meta_tokens_per_group);
    let extra = alpha.map(|plane| {
        prepare_extra_sections(
            ctx,
            plane,
            &dim,
            &group_coords,
            num_passes - 1,
            extra_tree::GlobalStreams {
                tree: &mut dc_tree,
                code: &mut dc_code_owned,
                dc: &mut dc_tokens_per_group,
                meta: &mut meta_tokens_per_group,
                dc_written: !use_dc_frame,
            },
            scratch,
        )
    });
    let dc_code = dc_code_owned.as_ref();

    // Separate scan arms before context planning: each gets its own final
    // context map, hybrid-uint configuration, clustering and ANS refinement.
    let mut alternative_pending: Vec<Vec<PendingAcGroup>> = (0..order_alternatives.len())
        .map(|_| Vec::with_capacity(all_pending.len()))
        .collect();
    for p in &mut all_pending {
        for (tokens, arm) in p.tokens.drain(num_passes..).zip(&mut alternative_pending) {
            arm.push(PendingAcGroup {
                group_idx: p.group_idx,
                tokens: vec![tokens],
            });
        }
    }
    let mut raced_ac_sections = None;
    let (ac_plan, ac_code_per_pass) = if order_alternatives.is_empty() {
        finalize_ac_entropy(
            ctx,
            scratch,
            &mut all_pending,
            num_passes,
            qf_threshold,
            ans_refinement,
        )
    } else {
        let used_tables = used_quant_table_slots(&dc_datas);
        let mut arms: Vec<OrderArm> = Vec::with_capacity(1 + order_alternatives.len());
        // Stage 1: every arm gets its context plan, hybrid-uint configurations,
        // clustering and the Slow refinement, then serializes.
        for (orders, mut pending) in std::iter::once((coeff_orders, all_pending))
            .chain(order_alternatives.into_iter().zip(alternative_pending))
        {
            let (plan, codes) = finalize_ac_entropy(
                ctx,
                scratch,
                &mut pending,
                num_passes,
                qf_threshold,
                ans_refinement.map(|_| crate::entropy::AnsRefinement::Slow),
            );
            let (bits, global, streams) = encode_order_arm(
                ctx,
                scratch,
                &pending,
                &plan,
                &codes,
                &orders,
                &used_tables,
                dim.num_groups,
                dim.num_dc_groups,
            );
            arms.push(OrderArm {
                orders,
                pending,
                plan,
                codes,
                bits,
                global,
                streams,
            });
        }
        // Stage 2: the exhaustive table search moves every arm by nearly the
        // same amount, so only arms within reach of the leader buy it.
        let leader = arms.iter().map(|arm| arm.bits).min().unwrap_or(0);
        let reach = leader + (leader as f64 * ORDER_RACE_REFINE_MARGIN) as usize;
        if matches!(
            ans_refinement,
            Some(crate::entropy::AnsRefinement::ExtraSlow)
        ) {
            for arm in arms.iter_mut().filter(|arm| arm.bits <= reach) {
                for (pass, code) in arm.codes.iter_mut().enumerate() {
                    crate::entropy::refine_ans_entropy_exhaustive(
                        code,
                        arm.pending.iter().map(|p| p.tokens[pass].as_slice()),
                        &ctx.thread_pool,
                        scratch,
                    );
                }
                (arm.bits, arm.global, arm.streams) = encode_order_arm(
                    ctx,
                    scratch,
                    &arm.pending,
                    &arm.plan,
                    &arm.codes,
                    &arm.orders,
                    &used_tables,
                    dim.num_groups,
                    dim.num_dc_groups,
                );
            }
        }
        // Prefer natural on ties, otherwise retain the incumbent.
        let mut best = 0;
        for (i, arm) in arms.iter().enumerate().skip(1) {
            if arm.bits < arms[best].bits
                || (arm.bits == arms[best].bits && arm.orders.used_mask == 0)
            {
                best = i;
            }
        }
        let winner = arms.swap_remove(best);
        coeff_orders = winner.orders;
        all_pending = winner.pending;
        // Retain the winning serialized sections; do not encode them again.
        raced_ac_sections = Some((winner.global, winner.streams));
        (winner.plan, winner.codes)
    };

    // Phase 4: write DC global with adaptive DC code.
    if let VarDctFrameKind::Patched(references) = frame_kind {
        crate::lossless::write_patch_dictionary(
            references,
            alpha.is_some(),
            scratch,
            &mut sections[0],
        );
    }
    #[cfg(feature = "splines")]
    if let Some(set) = splines {
        crate::splines::write_splines(set, scratch, &mut sections[0]);
    }
    if let Some(noise) = &noise {
        noise.write(&mut sections[0]);
    }
    write_dc_global(
        &distp,
        dim.num_dc_groups,
        &ac_plan,
        &dc_tree,
        &dc_code,
        ytox_dc,
        ytob_dc,
        ctx.cfl_frame(),
        extra.as_ref(),
        scratch,
        &mut sections[0],
    );

    // Phase 5: each DC group is an independent entropy stream. Encode the
    // streams in parallel, then place their writers back in raster order.
    let dc_sections = ctx
        .thread_pool
        .steal_map(scratch, dc_datas.len(), |i, _scratch| {
            let dc_data = &dc_datas[i];
            let mut w = BitWriter::new();
            // With a DC frame, the group carries no DC stream.
            if !use_dc_frame {
                w.write(2, 0); // extra_dc_precision
                w.write(4, 3); // use global tree, default wp, no transforms
                if dc_code.use_prefix_code {
                    for t in &dc_tokens_per_group[i] {
                        write_token(*t, &dc_code, &mut w);
                    }
                } else {
                    crate::entropy::write_ans_tokens(
                        &dc_tokens_per_group[i],
                        dc_code.context_map,
                        dc_code.ans_symbols,
                        dc_code.ans_reverse_maps,
                        dc_code.hybrid_uint_configs,
                        &mut w,
                    );
                }
            }
            // ModularLfGroup: the extra channel's coarse squeeze levels.
            if let Some(extra) = &extra {
                w.append(&extra.lf[i]);
            }
            let num_blocks = dc_data.ac_strategy.xsize() * dc_data.ac_strategy.ysize();
            let num_ac_blocks = dc_data.ac_strategy.count_first_blocks();
            let nb_bits = if num_blocks <= 1 {
                0
            } else {
                32 - (num_blocks as u32).leading_zeros() as usize
                    - if num_blocks.is_power_of_two() { 1 } else { 0 }
            };
            if nb_bits != 0 {
                w.write(nb_bits, (num_ac_blocks - 1) as u64);
            }
            w.write(4, 3);
            if dc_code.use_prefix_code {
                for t in &meta_tokens_per_group[i] {
                    write_token(*t, &dc_code, &mut w);
                }
            } else {
                crate::entropy::write_ans_tokens(
                    &meta_tokens_per_group[i],
                    dc_code.context_map,
                    dc_code.ans_symbols,
                    dc_code.ans_reverse_maps,
                    dc_code.hybrid_uint_configs,
                    &mut w,
                );
            }
            w
        });
    for (i, section) in dc_sections.into_iter().enumerate() {
        sections[1 + i] = section;
    }
    drop(dc_tokens_per_group);
    drop(meta_tokens_per_group);

    let (ac_global, ac_sections) = if let Some(raced) = raced_ac_sections {
        raced
    } else {
        let mut global = BitWriter::new();
        write_ac_global(
            ctx.matrices(),
            &used_quant_table_slots(&dc_datas),
            &coeff_orders,
            dim.num_groups,
            &ac_code_per_pass,
            scratch,
            &mut global,
        );
        let ac = encode_ac_sections(
            ctx,
            scratch,
            &all_pending,
            &ac_code_per_pass,
            dim.num_dc_groups,
        );
        (global, ac)
    };
    sections[1 + dim.num_dc_groups] = ac_global;
    for (section_idx, section) in ac_sections {
        sections[section_idx] = section;
    }
    drop(all_pending);
    // Modular alpha: extra-channel modular data is decoded in the last pass
    // (the only pass whose modular sub-image shift range is set when num_ds=0,
    // jxl-frame lib.rs:101-108 / pass_group.rs:93). Write it into that section.
    if let Some(extra) = &extra {
        let last_pass = num_passes - 1;
        for (abs_group_id, section) in extra.ac.iter().enumerate() {
            sections[2 + dim.num_dc_groups + last_pass * dim.num_groups + abs_group_id]
                .append(section);
        }
    }

    // The DC frame precedes the frame that uses it.
    let mut dc_frame_bits = 0;
    if use_dc_frame {
        let dc_image = assemble_source_dc(&dc_datas, &group_coords, &dim);
        let mut dc_writer = BitWriter::new();
        let coded = crate::lossless::encode_dc_frame(
            &dc_image,
            std::array::from_fn(|c| {
                distp.dc_step[c] / (crate::quant_weights::INV_DC_QUANT[c] * distp.scale_dc)
            }),
            ctx.progressive_dc_levels,
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut dc_writer,
        );
        debug_assert!(coded, "DC frame size checked up front");
        dc_frame_bits = dc_writer.bits_written();
        writer.append(&dc_writer);
    }

    #[cfg(feature = "splines")]
    let has_splines = splines.is_some();
    #[cfg(not(feature = "splines"))]
    let has_splines = false;
    write_frame_header_kind(
        distp.x_qm_scale,
        ctx.b_qm_scale(),
        distp.epf_iters,
        distp.epf_pass0_scale,
        distp.epf_quant_mul,
        distp.gab_enabled,
        alpha.is_some(),
        coeff_shifts,
        frame_kind,
        has_splines,
        noise.is_some(),
        skip_dc_smoothing,
        use_dc_frame,
        ctx.coding,
        ctx.epf_channel_scale,
        ctx.frame_role,
        writer,
    );
    let payload_bits = dc_frame_bits
        + sections
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 0 && *i != 1 + dim.num_dc_groups)
            .map(|(_, section)| section.bits_written())
            .sum::<usize>();
    let order = (ctx.group_order != crate::GroupOrder::Scanline
        && matches!(
            frame_kind,
            VarDctFrameKind::Regular | VarDctFrameKind::Patched(_)
        ))
    .then(|| group_section_order(&dim, num_passes, ctx.group_order));
    combine_sections_ordered(
        &mut sections,
        order.as_deref().map(|o| (o, &mut scratch.huffman_pool)),
        writer,
    );
    Ok(payload_bits)
}

/// Copies one AC group's reconstruction capture into the frame image.
fn paste_group(target: &mut Image3F, dim: &ImageDim, group_idx: usize, image: &Image3F) {
    let x0 = (group_idx % dim.xsize_groups) * K_GROUP_DIM;
    let y0 = (group_idx / dim.xsize_groups) * K_GROUP_DIM;
    for c in 0..3 {
        for y in 0..image.ysize() {
            target.plane_row_mut(c, y0 + y)[x0..x0 + image.xsize()]
                .copy_from_slice(image.plane_row(c, y));
        }
    }
}

/// The frame's unrounded per-block XYB DC (the DC frame's input), stitched
/// from the DC groups.
fn assemble_source_dc(
    dc_datas: &[DcGroupData],
    group_coords: &[(usize, usize)],
    dim: &ImageDim,
) -> Image3F {
    let mut image = Image3F::new(dim.xsize_blocks, dim.ysize_blocks);
    for (dc, &(gx, gy)) in dc_datas.iter().zip(group_coords) {
        let ox = gx * (K_DC_GROUP_DIM / K_BLOCK_DIM);
        let oy = gy * (K_DC_GROUP_DIM / K_BLOCK_DIM);
        for (c, plane) in [&dc.source_dc_x, &dc.source_dc_y, &dc.source_dc_b]
            .into_iter()
            .enumerate()
        {
            let plane = plane.as_ref().expect("source DC kept for the DC frame");
            for y in 0..plane.ysize() {
                image.plane_row_mut(c, oy + y)[ox..ox + plane.xsize()]
                    .copy_from_slice(plane.row(y));
            }
        }
    }
    image
}

fn finalize_ac_entropy(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &mut [PendingAcGroup],
    num_passes: usize,
    qf_threshold: u32,
    ans_refinement: Option<crate::entropy::AnsRefinement>,
) -> (
    crate::ac_context::AcCtxPlan,
    Vec<crate::entropy::OwnedEntropyCode>,
) {
    let effort = ctx.speed.effort();
    // Per-image AC block-context plan: keep quant-field splits only where the
    // real token statistics pay for them within the spec's 16-context budget.
    // The greedy proposes; the arm gate below disposes, comparing real
    // entropy-code headers plus payload estimates because the Shannon proxy
    // reliably overestimates what clustering-aware coding realizes.
    let baseline = if effort.square_merges {
        crate::ac_context::AcCtxPlan::baseline()
    } else {
        crate::ac_context::AcCtxPlan::dct8_only()
    };

    let build_codes = |pending: &[PendingAcGroup],
                       num_contexts: usize,
                       scratch: &mut CoderScratch|
     -> Vec<crate::entropy::OwnedEntropyCode> {
        (0..num_passes)
            .map(|pass| {
                if effort.ac_prefix_clustering {
                    crate::entropy::optimize_entropy_code_ac_streams(
                        pending.iter().map(|p| p.tokens[pass].as_slice()),
                        num_contexts,
                        &mut scratch.huffman_pool,
                        effort.huc_select,
                        ctx.speed,
                        Some(&ctx.thread_pool),
                    )
                } else {
                    crate::entropy::optimize_entropy_code_ac_streams_fast(
                        pending.iter().map(|p| p.tokens[pass].as_slice()),
                        num_contexts,
                        &mut scratch.huffman_pool,
                        effort.huc_select,
                        Some(&ctx.thread_pool),
                    )
                }
            })
            .collect()
    };
    let arm_bits = |pending: &[PendingAcGroup],
                    codes: &[crate::entropy::OwnedEntropyCode],
                    plan: &crate::ac_context::AcCtxPlan,
                    scratch: &mut CoderScratch|
     -> u64 {
        let mut header = BitWriter::new();
        write_block_ctx_map(plan, scratch, &mut header);
        for code in codes {
            write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, &mut header);
        }
        let mut bits = header.bits_written() as u64;
        for (pass, code) in codes.iter().enumerate() {
            bits += crate::entropy::estimate_ac_plain_bits(
                pending.iter().map(|p| p.tokens[pass].as_slice()),
                code,
            );
        }
        bits
    };
    let remap_tokens = |pending: &mut [PendingAcGroup],
                        plan: &crate::ac_context::AcCtxPlan,
                        scratch: &mut CoderScratch| {
        ctx.thread_pool
            .steal_for_each_mut(scratch, pending, |_i, p, _s| {
                for pass_tokens in &mut p.tokens {
                    pass_tokens.remap_contexts(|context| plan.remap(context));
                }
            });
    };

    let (ac_plan, mut ac_code_per_pass) = if !ctx.speed.effort().ac_ctx_plan {
        remap_tokens(pending, &baseline, scratch);
        let codes = build_codes(pending, baseline.num_ac_contexts(), scratch);
        (baseline, codes)
    } else {
        let proposed = crate::ac_context::plan_block_ctx_map(
            pending
                .iter()
                .flat_map(|pending| pending.tokens.iter().map(AcTokens::as_slice)),
            qf_threshold,
        );
        if proposed == baseline {
            remap_tokens(pending, &baseline, scratch);
            let codes = build_codes(pending, baseline.num_ac_contexts(), scratch);
            (baseline, codes)
        } else {
            // Both arms keep the coefficient values. Save only the original
            // contexts so the same token buffers can be remapped for each arm.
            const { assert!(crate::ac_context::K_NUM_FINE_AC_CONTEXTS <= 1 << 16) };
            let contexts: Vec<Vec<Vec<u16>>> = pending
                .iter()
                .map(|p| {
                    p.tokens
                        .iter()
                        .map(|tokens| tokens.iter().map(|t| t.context as u16).collect())
                        .collect()
                })
                .collect();
            let remap_saved = |pending: &mut [PendingAcGroup],
                               plan: &crate::ac_context::AcCtxPlan,
                               scratch: &mut CoderScratch| {
                ctx.thread_pool
                    .steal_for_each_mut(scratch, pending, |i, p, _s| {
                        for (tokens, contexts) in p.tokens.iter_mut().zip(&contexts[i]) {
                            let mut contexts = contexts.iter();
                            tokens.remap_contexts(|_| {
                                plan.remap(u32::from(*contexts.next().unwrap()))
                            });
                        }
                    });
            };
            remap_tokens(pending, &baseline, scratch);

            let base_codes = build_codes(pending, baseline.num_ac_contexts(), scratch);
            let base_bits = arm_bits(pending, &base_codes, &baseline, scratch);
            remap_saved(pending, &proposed, scratch);
            let plan_codes = build_codes(pending, proposed.num_ac_contexts(), scratch);
            let plan_bits = arm_bits(pending, &plan_codes, &proposed, scratch);

            const AC_PLAN_MARGIN_BITS: u64 = 256;
            if plan_bits + AC_PLAN_MARGIN_BITS < base_bits {
                (proposed, plan_codes)
            } else {
                remap_saved(pending, &baseline, scratch);
                (baseline, base_codes)
            }
        }
    };

    // Slow always considers splitting clusters collapsed by the prefix-cost
    // search. Fast uses the cheaper batch of moves at high quality.
    if let Some(refinement) = ans_refinement {
        refine_ac_codes(ctx, scratch, pending, &mut ac_code_per_pass, refinement);
    }

    (ac_plan, ac_code_per_pass)
}

fn refine_ac_codes(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &[PendingAcGroup],
    codes: &mut [crate::entropy::OwnedEntropyCode],
    refinement: crate::entropy::AnsRefinement,
) {
    for (pass, code) in codes.iter_mut().enumerate() {
        crate::entropy::refine_ans_clusters(
            code,
            pending
                .iter()
                .map(|pending| pending.tokens[pass].as_slice()),
            refinement,
            &ctx.thread_pool,
            scratch,
        );
    }
}

fn encode_ac_sections(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &[PendingAcGroup],
    codes: &[crate::entropy::OwnedEntropyCode],
    num_dc_groups: usize,
) -> Vec<(usize, BitWriter)> {
    let num_passes = codes.len();
    let num_groups = pending.len();
    // Phase 7: write each (pass, group) AC section. Section index for
    // (pass, group) = 2 + num_dc_groups + pass*num_groups + group_idx
    // (jxl-frame toc.rs:196-200): raw tokens via the shared plain code.
    let num_ac_sections = pending.len() * num_passes;
    ctx.thread_pool
        .steal_map(scratch, num_ac_sections, |task, _scratch| {
            let i = task / num_passes;
            let pass = task % num_passes;
            let pg = &pending[i];
            let pass_tokens = pg.tokens[pass].as_slice();
            let mut w = BitWriter::new();
            let section_idx = 2 + num_dc_groups + pass * num_groups + pg.group_idx;
            let code_ref = codes[pass].as_ref();
            if code_ref.use_prefix_code {
                pass_tokens
                    .iter()
                    .for_each(|t| write_token(t, &code_ref, &mut w));
            } else {
                // rANS: the whole group's tokens are encoded as one LIFO unit.
                crate::entropy::write_ans_tokens(
                    pass_tokens,
                    code_ref.context_map,
                    code_ref.ans_symbols,
                    code_ref.ans_reverse_maps,
                    code_ref.hybrid_uint_configs,
                    &mut w,
                );
            }
            (section_idx, w)
        })
}

/// One coefficient-scan arm of ExtraSlow's final race.
struct OrderArm {
    orders: crate::coeff_order::CoeffOrders,
    pending: Vec<PendingAcGroup>,
    plan: crate::ac_context::AcCtxPlan,
    codes: Vec<crate::entropy::OwnedEntropyCode>,
    bits: usize,
    global: BitWriter,
    streams: Vec<(usize, BitWriter)>,
}

/// Arms whose serialized size after the Slow refinement exceeds the leader's
/// by more than this fraction skip the exhaustive table search.
const ORDER_RACE_REFINE_MARGIN: f64 = 0.001;

/// Serialize the varying AC headers and each independent entropy stream.
/// The block-context map lives in DC global; all its other data is common to
/// both scans. Multi-group AC sections are byte-aligned, while single-group
/// frames concatenate these sections without intervening padding.
#[allow(clippy::too_many_arguments)]
fn encode_order_arm(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &[PendingAcGroup],
    plan: &crate::ac_context::AcCtxPlan,
    codes: &[crate::entropy::OwnedEntropyCode],
    orders: &crate::coeff_order::CoeffOrders,
    used_tables: &[bool; 13],
    num_groups: usize,
    num_dc_groups: usize,
) -> (usize, BitWriter, Vec<(usize, BitWriter)>) {
    let mut context_header = BitWriter::new();
    write_block_ctx_map(plan, scratch, &mut context_header);
    let mut global = BitWriter::new();
    write_ac_global(
        ctx.matrices(),
        used_tables,
        orders,
        num_groups,
        codes,
        scratch,
        &mut global,
    );
    let streams = encode_ac_sections(ctx, scratch, pending, codes, num_dc_groups);
    let padded_bits = |bits: usize| {
        if num_groups > 1 {
            bits.div_ceil(8) * 8
        } else {
            bits
        }
    };
    let bits = context_header.bits_written()
        + padded_bits(global.bits_written())
        + streams
            .iter()
            .map(|(_, w)| padded_bits(w.bits_written()))
            .sum::<usize>();
    (bits, global, streams)
}

/// Per-AC-group buffered tokens. For progressive (multi-pass) encoding the
/// quantized AC coefficients of each block are split across passes; `tokens`
/// holds one token stream per pass, plus temporary alternative scan streams
/// in ExtraSlow's final refine pass. `group_idx` is the raster group index
/// (0..num_groups); the section for (pass, group) is
/// `2 + num_dc_groups + pass*num_groups + group_idx`.
/// `process_ac_group` output: tokens, the group's quantized DC, the unrounded
/// source DC and order statistics when collected, and the luma reconstruction
/// capture (plane + per-block token starts) when requested.
type ProcessedAcGroup = (
    PendingAcGroup,
    Image3S,
    Option<crate::group::SourceDc>,
    Option<crate::coeff_order::OrderStats>,
    Option<(crate::image::Image3F, Vec<(u16, u16, u32)>)>,
);

/// One coded AC group of the refined pass: DC-group index, group position,
/// tokens and quantized DC.
type RefinedAcGroup = (usize, usize, usize, PendingAcGroup, Image3S);

#[derive(Clone)]
pub(crate) struct PendingAcGroup {
    pub(crate) group_idx: usize,
    pub(crate) tokens: Vec<AcTokens>,
}

/// Closed-loop quant-field refinement (ExtraSlow). Every group is re-quantized
/// the way pass 2 codes it (custom scan orders, RDOQ at the frozen token
/// prices), priced with those prices and reconstructed in luma (source plus
/// the inverse-transformed Y quantization error, the rerank's model). The
/// whole field moved one step coarser and one step finer gives the global
/// exchange rate (MS-SSIM deficit per percent of bits) and, per 32px tile,
/// the marginal deficit and bit change of each direction. Tiles whose
/// marginal return beats the global rate move; the joint move is verified and
/// kept only if `-(1 - MS-SSIM) - slope * log rate` improves.
#[cfg(feature = "extraslow")]
#[allow(clippy::too_many_arguments)]
fn run_quant_field_loop(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    dc_datas: &mut [DcGroupData],
    ac_tasks: &[(usize, usize, usize)],
    group_coords: &[(usize, usize)],
    num_passes: usize,
    coeff_shifts: &[u32],
    coeff_orders: &crate::coeff_order::CoeffOrders,
    order_alternatives: &[crate::coeff_order::CoeffOrders],
    ytob_dc: i32,
    qf_threshold: u32,
    prices: &crate::entropy::FrozenTokenPrices,
) -> Option<(Vec<RefinedAcGroup>, DistanceParams)> {
    /// Marginals and slope come from plain quantization; the ratio of deficit
    /// to bits is what the trellis preserves, at a third of the cost.
    const SHIFT_RDOQ: bool = false;
    /// The mean of the finer and coarser chords overstates the tangent of a
    /// convex rate-distortion curve.
    const SLOPE_SCALE: f32 = 0.85;
    const TILE_BLOCKS: usize = 4;
    const MAX_ITERS: usize = 3;
    // Candidates verified exactly per iteration; with the block memo each
    // costs ~0.4 ms at 1 MP, and 300 (vs 40) buys SS2 −0.3 / cvvdp −0.15 for +0.27×.
    const GREEDY: usize = 300;
    // At most this many candidates per AC group per iteration: each extra
    // candidate in a group costs another serial round (one group encode).
    const PER_GROUP: usize = 12;
    const MAX_MOVE_FRACTION: f32 = 0.3;
    const MAX_ABS_MOVE: i8 = 2;
    const MIN_FIELD: u8 = 1;

    let (w, h) = (dim.xsize, dim.ysize);
    let (xb, yb) = (w.div_ceil(K_BLOCK_DIM), h.div_ceil(K_BLOCK_DIM));
    if xb < TILE_BLOCKS || yb < TILE_BLOCKS {
        return None;
    }
    let reference = crate::ssimulacra2::Ssimulacra2Reference::new(opsin)?;
    let blocks_per_dc = K_DC_GROUP_DIM / K_BLOCK_DIM;

    let group_map = LoopGroupMap::new(w, h, group_coords, ac_tasks);
    let locate = |bx, by| group_map.locate(bx, by);
    let task_of_block = &group_map.task_of_block;

    #[derive(Clone)]
    struct Eval {
        recon: Image3F,
        pending: Vec<Option<PendingAcGroup>>,
        quant_dc: Vec<Option<Image3S>>,
        starts: Vec<Vec<(u16, u16, u32)>>,
        /// Per task, the memo evaluation `recon` was last synced with.
        synced: Vec<u32>,
        /// Trial evaluations take cached blocks' bits instead of their tokens.
        trial: bool,
        hit_bits: Vec<f32>,
    }
    // Re-quantize, price and reconstruct the given AC groups under the fields.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        opsin: &Image3F,
        dim: &ImageDim,
        distp: &DistanceParams,
        dc_datas: &[DcGroupData],
        ac_tasks: &[(usize, usize, usize)],
        group_coords: &[(usize, usize)],
        num_passes: usize,
        coeff_shifts: &[u32],
        coeff_orders: &crate::coeff_order::CoeffOrders,
        order_alternatives: &[crate::coeff_order::CoeffOrders],
        ytob_dc: i32,
        qf_threshold: u32,
        prices: Option<&crate::entropy::FrozenTokenPrices>,
        tasks: &[usize],
        memo: Option<&[std::sync::Mutex<Option<crate::group::AcBlockCache>>]>,
        eval: &mut Eval,
    ) -> Vec<crate::ssimulacra2::Rect> {
        let eval_trial = eval.trial;
        let results = ctx
            .thread_pool
            .steal_map(scratch, tasks.len(), |i, scratch| {
                let t = tasks[i];
                let mut memo_slot = memo.map(|m| m[t].lock().expect("block memo"));
                if let Some(Some(m)) = memo_slot.as_deref_mut() {
                    m.trial = eval_trial;
                    m.trial_bits = 0.0;
                }
                let (dc_idx, gx, gy) = ac_tasks[t];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (p, local, _, _, capture) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    &dc_datas[dc_idx],
                    num_passes,
                    coeff_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    ytob_dc,
                    prices,
                    coeff_orders,
                    order_alternatives,
                    false,
                    qf_threshold,
                    true,
                    memo_slot.as_deref_mut(),
                );
                let (plane, starts) = capture.expect("recon capture");
                let hit_bits = match memo_slot.as_deref_mut() {
                    Some(Some(m)) => {
                        m.trial = false;
                        m.trial_bits as f32
                    }
                    _ => 0.0,
                };
                (t, p, local, starts, plane, (dc_gx, dc_gy, gx, gy), hit_bits)
            });
        let mut dirty = Vec::with_capacity(results.len());
        for (t, p, local, starts, plane, (dc_gx, dc_gy, gx, gy), hit_bits) in results {
            eval.hit_bits[t] = hit_bits;
            eval.pending[t] = Some(p);
            eval.quant_dc[t] = Some(local);
            eval.starts[t] = starts;
            let x0 = dc_gx * K_DC_GROUP_DIM + gx * K_GROUP_DIM;
            let y0 = dc_gy * K_DC_GROUP_DIM + gy * K_GROUP_DIM;
            let empty = crate::ssimulacra2::Rect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            };
            let rect = match memo {
                Some(memo) => {
                    // Only the cells recomputed since this evaluation last
                    // synced with the memo differ from `eval.recon`.
                    let guard = memo[t].lock().expect("block memo");
                    let m = guard.as_ref().expect("memo filled by the evaluation");
                    let rect = m.sync_into(eval.synced[t], &mut eval.recon, x0, y0).map_or(
                        empty,
                        |(rx0, ry0, rx1, ry1)| crate::ssimulacra2::Rect {
                            x0: rx0,
                            y0: ry0,
                            x1: rx1,
                            y1: ry1,
                        },
                    );
                    eval.synced[t] = m.epoch;
                    rect
                }
                None => {
                    for c in 0..3 {
                        for y in 0..plane.ysize() {
                            let src = &plane.plane_row(c, y)[..plane.xsize()];
                            eval.recon.plane_row_mut(c, y0 + y)[x0..x0 + src.len()]
                                .copy_from_slice(src);
                        }
                    }
                    crate::ssimulacra2::Rect {
                        x0,
                        y0,
                        x1: x0 + plane.xsize(),
                        y1: y0 + plane.ysize(),
                    }
                }
            };
            dirty.push(rect);
        }
        dirty
    }
    let new_eval = || Eval {
        recon: Image3F::new(w, h),
        pending: vec![None; ac_tasks.len()],
        quant_dc: vec![None; ac_tasks.len()],
        starts: vec![Vec::new(); ac_tasks.len()],
        synced: vec![0; ac_tasks.len()],
        trial: false,
        hit_bits: vec![0.0; ac_tasks.len()],
    };
    // Trials overwrite a task's tokens and starts before reading them. Only
    // pixels, DC and memo sync points are needed for untouched tasks.
    let trial_eval = |e: &Eval| Eval {
        recon: e.recon.clone(),
        pending: vec![None; ac_tasks.len()],
        quant_dc: e.quant_dc.clone(),
        starts: vec![Vec::new(); ac_tasks.len()],
        synced: e.synced.clone(),
        trial: false,
        hit_bits: vec![0.0; ac_tasks.len()],
    };
    let fill_decoded_dc =
        |e: &Eval, dp: &DistanceParams, rect: crate::ssimulacra2::Rect, out: &mut Image3F| {
            use crate::quant_weights::INV_DC_QUANT;
            let steps: [f32; 3] =
                std::array::from_fn(|c| dp.dc_step[c] / (INV_DC_QUANT[c] * dp.scale_dc));
            let cfl = ctx.cfl_frame();
            let cfl_b = crate::color_correlation::dc_cfl_factor(dp.dc_step, ytob_dc, cfl);
            let cfl_x = crate::color_correlation::dc_cfl_factor_x(dp.dc_step, cfl.base_x);
            group_map.reconstruct_dc(
                &e.quant_dc,
                [cfl_x, cfl_b],
                steps,
                ctx.dc_rows.reconstruct,
                rect,
                out,
            );
        };
    let decoded_dc = |e: &Eval, dp: &DistanceParams| -> Image3F {
        let mut out = Image3F::new(xb, yb);
        fill_decoded_dc(e, dp, crate::ssimulacra2::Rect::new(0, 0, w, h), &mut out);
        out
    };
    // Small trials reuse one plane and reconstruct only cells the judge reads.
    let mut trial_dc = None;
    let frame_bits = |e: &Eval| {
        prices.adaptive_bits(e.pending.iter().map(|p| {
            p.as_ref()
                .map_or(TokenSlice::Wide(&[]), |p| p.tokens[0].as_slice())
        }))
    };
    // Frozen-price bits of every first block, indexed by frame block.
    let block_bits = |e: &Eval| -> Vec<f32> {
        let mut out = vec![0f32; xb * yb];
        for (t, starts) in e.starts.iter().enumerate() {
            let (dc_gx, dc_gy) = group_coords[ac_tasks[t].0];
            let toks = e.pending[t]
                .as_ref()
                .map_or(TokenSlice::Wide(&[]), |p| p.tokens[0].as_slice());
            for (i, &(lbx, lby, start)) in starts.iter().enumerate() {
                let end = starts.get(i + 1).map_or(toks.len(), |s| s.2 as usize);
                let bits: f32 = toks
                    .slice(start as usize..end)
                    .iter()
                    .map(|k| prices.token_bits(k))
                    .sum();
                let bx = dc_gx * blocks_per_dc + lbx as usize;
                let by = dc_gy * blocks_per_dc + lby as usize;
                out[by * xb + bx] += bits;
            }
        }
        out
    };

    let (tiles_x, tiles_y) = (xb.div_ceil(TILE_BLOCKS), yb.div_ceil(TILE_BLOCKS));
    let n_tiles = tiles_x * tiles_y;
    let tile_of_block = |bx: usize, by: usize| (by / TILE_BLOCKS) * tiles_x + bx / TILE_BLOCKS;
    let tile_blocks = |t: usize| -> Vec<(usize, usize)> {
        let (tx, ty) = (t % tiles_x, t / tiles_x);
        let mut v = Vec::new();
        for by in ty * TILE_BLOCKS..((ty + 1) * TILE_BLOCKS).min(yb) {
            for bx in tx * TILE_BLOCKS..((tx + 1) * TILE_BLOCKS).min(xb) {
                v.push((bx, by));
            }
        }
        v
    };
    // Sum a per-pixel map over each tile.
    let tile_sums = |map: &[f32]| -> Vec<f64> {
        let mut out = vec![0f64; n_tiles];
        let tile_px = TILE_BLOCKS * K_BLOCK_DIM;
        for y in 0..h {
            let ty = y / tile_px;
            let row = &map[y * w..(y + 1) * w];
            for (x, &v) in row.iter().enumerate() {
                out[ty * tiles_x + x / tile_px] += v as f64;
            }
        }
        out
    };
    let tile_bits = |blk: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; n_tiles];
        for by in 0..yb {
            for bx in 0..xb {
                out[tile_of_block(bx, by)] += blk[by * xb + bx];
            }
        }
        out
    };
    /// Fields a move overwrote, for an exact restore (the clamp at the bounds
    /// makes the inverse step inexact).
    struct Saved(Vec<(usize, usize, usize, u8)>);
    // Apply `moves` (tile, step) to the fields; returns the touched AC tasks
    // and the previous values.
    let apply = |dc_datas: &mut [DcGroupData], moves: &[(usize, i8)]| -> (Vec<usize>, Saved) {
        let mut tasks = Vec::new();
        let mut saved = Vec::new();
        for &(t, step) in moves {
            for (bx, by) in tile_blocks(t) {
                let (dc_idx, lbx, lby) = locate(bx, by);
                let v = &mut dc_datas[dc_idx].raw_quant_field.row_mut(lby)[lbx];
                saved.push((dc_idx, lbx, lby, *v));
                *v = (*v as i16 + step as i16).clamp(MIN_FIELD as i16, 255) as u8;
                let task = task_of_block[by * xb + bx];
                if !tasks.contains(&task) {
                    tasks.push(task);
                }
            }
        }
        (tasks, Saved(saved))
    };
    let restore = |dc_datas: &mut [DcGroupData], saved: Saved| {
        for (dc_idx, lbx, lby, v) in saved.0 {
            dc_datas[dc_idx].raw_quant_field.row_mut(lby)[lbx] = v;
        }
    };
    let all_tasks: Vec<usize> = (0..ac_tasks.len()).collect();
    let all_moves = |step: i8| -> Vec<(usize, i8)> { (0..n_tiles).map(|t| (t, step)).collect() };
    // Per-group block memo shared by every priced evaluation of the loop.
    let memo: Vec<std::sync::Mutex<Option<crate::group::AcBlockCache>>> = (0..ac_tasks.len())
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    let memo: Option<&[_]> = Some(&memo[..]);
    // Only between trials: layout snapshots retain offsets into these arenas.
    let compact_memos = |tasks: &[usize]| {
        if let Some(memo) = memo {
            for &t in tasks {
                if let Some(m) = memo[t].lock().expect("block memo").as_mut() {
                    m.compact_tokens();
                }
            }
        }
    };
    let mut cur = new_eval();
    evaluate(
        ctx,
        scratch,
        opsin,
        dim,
        distp,
        dc_datas,
        ac_tasks,
        group_coords,
        num_passes,
        coeff_shifts,
        coeff_orders,
        order_alternatives,
        ytob_dc,
        qf_threshold,
        Some(prices),
        &all_tasks,
        memo,
        &mut cur,
    );
    let ref_bits = frame_bits(&cur);
    let finish =
        |e: Eval, distp: DistanceParams| -> Option<(Vec<RefinedAcGroup>, DistanceParams)> {
            Some((
                e.pending
                    .into_iter()
                    .zip(e.quant_dc)
                    .enumerate()
                    .map(|(t, (p, q))| {
                        let (dc_idx, gx, gy) = ac_tasks[t];
                        (
                            dc_idx,
                            gx,
                            gy,
                            p.expect("pending group"),
                            q.expect("group dc"),
                        )
                    })
                    .collect(),
                distp,
            ))
        };
    if ref_bits <= 0.0 {
        return finish(cur, *distp);
    }
    let rate_term = |bits: f32| 100.0 * (bits / ref_bits).ln();
    let pixels = (w * h) as f64;
    let mut cur_bits = ref_bits;
    // The incremental state follows every accepted move exactly. Defer the
    // SSIM state until after the full-frame marginal scores: their temporary
    // images would otherwise overlap its retained reconstruction pyramids.
    // Under the ColorVideoVDP arbiter it is also the exact metric; the
    // marginals that rank candidates still come from SSIMULACRA2.
    let cv_judge = (ctx.arbiter == Arbiter::Cvvdp && ctx.coding.is_xyb())
        .then(|| crate::cvvdp_arbiter::CvvdpArbiter::new(opsin))
        .flatten();
    let mut loop_state = cv_judge.map(|mut judge| {
        judge.reset(&cur.recon, &decoded_dc(&cur, distp));
        LoopArbiter::Cvvdp(Box::new(judge))
    });
    if SHIFT_RDOQ && loop_state.is_none() {
        loop_state = Some(LoopArbiter::Ssimulacra2(Box::new(
            reference.state(&cur.recon),
        )));
    }
    let mut cur_d = loop_state.as_ref().map_or(f32::NAN, LoopArbiter::deficit);
    let mut cur_map = if SHIFT_RDOQ {
        loop_state.as_ref().expect("shift judge").map()
    } else {
        Vec::new()
    };
    let mut moved = vec![0i8; n_tiles];

    let mut cached_slope = 0f32;
    // Under the ColorVideoVDP arbiter its own exchange rate, from the same
    // shift evaluations, prices every decision it arbitrates.
    let mut cv_slope = 0f32;
    let mut cached_moves: Vec<(f32, usize, i8)> = Vec::new();
    let mut accepted_moves: Vec<(usize, i8)> = Vec::new();
    // Moves the exact arbiter already rejected are not tried again: later
    // iterations rejected them again at the cost of half the evaluations.
    let mut rejected_moves: std::collections::HashSet<(usize, i8)> = Default::default();
    // Allocate the greedy scratch after the full-frame marginal evaluations.
    let mut greedy_eval = None;
    for iter in 0..MAX_ITERS {
        // Later iterations reuse the first iteration's marginals: the greedy
        // pass arbitrates every candidate exactly, so only the ranking is stale.
        let (slope, moves): (f32, Vec<(f32, usize, i8)>) = if iter > 0 {
            let moves: Vec<(f32, usize, i8)> = cached_moves
                .iter()
                .filter(|&&(_, t, step)| {
                    if step > 0 {
                        moved[t] < MAX_ABS_MOVE
                    } else {
                        moved[t] > -MAX_ABS_MOVE
                    }
                })
                .filter(|&&(_, t, step)| !accepted_moves.contains(&(t, step)))
                .filter(|&&(_, t, step)| !rejected_moves.contains(&(t, step)))
                .copied()
                .collect();
            (cached_slope, moves)
        } else {
            // Marginals and exchange rate from a plain-quantization triple: the
            // current field, the whole field one step finer, and every other
            // group one step coarser (half an evaluation; the ratio of two
            // half-frame deltas is unbiased). The one-sided finer chord
            // undervalues the tangent of a convex curve, so the two chords are
            // averaged.
            let plain = if SHIFT_RDOQ {
                cur.clone()
            } else {
                let mut e = new_eval();
                evaluate(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    dc_datas,
                    ac_tasks,
                    group_coords,
                    num_passes,
                    coeff_shifts,
                    coeff_orders,
                    &[],
                    ytob_dc,
                    qf_threshold,
                    None,
                    &all_tasks,
                    None,
                    &mut e,
                );
                e
            };
            let r_p = rate_term(frame_bits(&plain));
            let shift_prices = SHIFT_RDOQ.then_some(prices);
            let (_, saved) = apply(dc_datas, &all_moves(1));
            let mut fine = new_eval();
            evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                &[],
                ytob_dc,
                qf_threshold,
                shift_prices,
                &all_tasks,
                None,
                &mut fine,
            );
            restore(dc_datas, saved);
            let r_f = rate_term(frame_bits(&fine));
            if r_f - r_p < 1e-3 {
                break;
            }
            // Score one full-frame workspace at a time. Retain only tile
            // summaries, then reuse the plain evaluation for the coarse arm.
            // Each score and token reduction is identical to the independent
            // full-frame comparisons; only their storage lifetimes change.
            let (d_p, map_p) = if SHIFT_RDOQ {
                (cur_d, cur_map.clone())
            } else {
                reference.deficit_and_map(&plain.recon)
            };
            let map_0 = tile_sums(&map_p);
            drop(map_p);
            let (d_f, fine_map) = reference.deficit_and_map(&fine.recon);
            let map_f = tile_sums(&fine_map);
            drop(fine_map);
            let bits_f = tile_bits(&block_bits(&fine));
            let bits_0 = tile_bits(&block_bits(&plain));
            let cv_pf = loop_state
                .as_mut()
                .and_then(LoopArbiter::cvvdp)
                .map(|judge| {
                    [&plain, &fine].map(|e| judge.deficit_of(&e.recon, &decoded_dc(e, distp)))
                });
            drop(fine);
            // The coarser shift covers half the groups (slope estimate only).
            let half_tasks: Vec<usize> = (0..ac_tasks.len()).step_by(2).collect();
            let half_moves: Vec<(usize, i8)> = (0..n_tiles)
                .filter(|&t| {
                    let (bx, by) = tile_blocks(t)[0];
                    half_tasks.contains(&task_of_block[by * xb + bx])
                })
                .map(|t| (t, -1))
                .collect();
            let (_, saved) = apply(dc_datas, &half_moves);
            let mut coarse = plain;
            evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                &[],
                ytob_dc,
                qf_threshold,
                shift_prices,
                &half_tasks,
                None,
                &mut coarse,
            );
            restore(dc_datas, saved);
            let r_c = rate_term(frame_bits(&coarse));
            let d_c = reference.deficit(&coarse.recon);
            let s_fine = (d_p - d_f) / (r_f - r_p);
            let s_coarse = if r_p - r_c > 1e-3 {
                (d_c - d_p) / (r_p - r_c)
            } else {
                s_fine
            };
            let slope = SLOPE_SCALE * 0.5 * (s_fine + s_coarse);
            if !(slope.is_finite() && slope > 0.0) {
                break;
            }
            let mut cv_flat = false;
            if let Some([cv_p, cv_f]) = cv_pf {
                let judge = loop_state
                    .as_mut()
                    .and_then(LoopArbiter::cvvdp)
                    .expect("CV judge");
                let cv_c = judge.deficit_of(&coarse.recon, &decoded_dc(&coarse, distp));
                let s_fine = (cv_p - cv_f) / (r_f - r_p);
                let s_coarse = if r_p - r_c > 1e-3 {
                    (cv_c - cv_p) / (r_p - r_c)
                } else {
                    s_fine
                };
                cv_slope = SLOPE_SCALE * 0.5 * (s_fine + s_coarse);
                cv_flat = !(cv_slope.is_finite() && cv_slope > 0.0);
            }
            drop(coarse);
            // Near transparency ColorVideoVDP no longer separates the shifts;
            // SSIMULACRA2 arbitrates the frame instead of the loop stopping.
            if cv_flat {
                cv_slope = 0.0;
                let state = LoopArbiter::Ssimulacra2(Box::new(reference.state(&cur.recon)));
                cur_d = state.deficit();
                loop_state = Some(state);
            }
            // Per tile: deficit gain (frame units) minus slope times rate cost (%).
            // Only finer moves: coarsening smooth tiles is a blind spot of mean
            // MS-SSIM that SSIMULACRA2 punishes, and edge-difference terms did not
            // close it.
            let mut moves: Vec<(f32, usize, i8)> = Vec::new();
            // The linearized map attributes the frame's deficit change to tiles;
            // rescale each direction so the tile gains sum to the measured change.
            let calib = |map_x: &[f64], d_x: f32| -> f32 {
                let predicted: f64 = (0..n_tiles).map(|t| (map_0[t] - map_x[t]) / pixels).sum();
                let measured = (d_p - d_x) as f64;
                if predicted.abs() > 1e-9 && measured / predicted > 0.0 {
                    (measured / predicted) as f32
                } else {
                    1.0
                }
            };
            let cal_f = calib(&map_f, d_f);
            for t in 0..n_tiles {
                let gain = cal_f * ((map_0[t] - map_f[t]) / pixels) as f32;
                let rate = 100.0 * (bits_f[t] - bits_0[t]) / ref_bits;
                let net = gain - slope * rate;
                if net > 1e-7 && moved[t] < MAX_ABS_MOVE {
                    moves.push((net, t, 1));
                }
            }
            moves.sort_by(|a, b| b.0.total_cmp(&a.0));
            moves.truncate(((MAX_MOVE_FRACTION * n_tiles as f32) as usize).max(1));
            cached_slope = slope;
            cached_moves = moves.clone();
            (slope, moves)
        };
        if loop_state.is_none() {
            let state = LoopArbiter::Ssimulacra2(Box::new(reference.state(&cur.recon)));
            cur_d = state.deficit();
            loop_state = Some(state);
        }
        if moves.is_empty() {
            break;
        }
        // Pixels and DC follow `cur`; evaluated tasks replace their tokens.
        let cand = greedy_eval.get_or_insert_with(|| trial_eval(&cur));
        // Moves are ranked on SSIMULACRA2 marginals and accepted on the judge's.
        let slope = if loop_state.as_ref().is_some_and(LoopArbiter::needs_dc) {
            cv_slope
        } else {
            slope
        };
        let mut j_cur = -cur_d - slope * rate_term(cur_bits);
        let mut accepted = false;
        // Evaluate the top candidates one at a time with the real metric.
        // The state follows every accepted move exactly, so it is built once.
        let state = loop_state.as_mut().expect("loop judge");
        // Tokens and starts are overwritten by the next evaluation. Restore
        // only DC and pixels on rejection; move the token buffers on acceptance.
        let sync_tasks =
            |from: &Eval, to: &mut Eval, tasks: &[usize], r: crate::ssimulacra2::Rect| {
                for &task in tasks {
                    to.quant_dc[task] = from.quant_dc[task].clone();
                    to.synced[task] = from.synced[task];
                }
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                            .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                    }
                }
            };
        let group_bits = |e: &Eval, task: usize| -> f32 {
            e.pending[task].as_ref().map_or(0.0, |p| {
                p.tokens[0].iter().map(|k| prices.token_bits(k)).sum()
            })
        };
        // The incumbent only changes on acceptance. Preserve each reduction's
        // exact f32 order, but do not repeat it for every rejected trial.
        let mut cur_gbits: Vec<f32> = (0..ac_tasks.len()).map(|t| group_bits(&cur, t)).collect();
        let mut cand_gbits = vec![0.0; ac_tasks.len()];
        // Rounds: candidates in distinct AC groups are re-encoded together in
        // one parallel pass, then arbitrated one by one with the exact metric.
        // At most `per_group` candidates per AC group per iteration: each extra
        // candidate in a group costs another serial round (one group encode).
        let mut per_group_count = vec![0usize; ac_tasks.len()];
        let mut aq_checkpoint = state.checkpoint();
        let mut remaining: Vec<(usize, i8)> = moves
            .iter()
            .take(GREEDY)
            .map(|m| (m.1, m.2))
            .filter(|&(t, _)| {
                let (bx, by) = tile_blocks(t)[0];
                let task = task_of_block[by * xb + bx];
                per_group_count[task] += 1;
                per_group_count[task] <= PER_GROUP
            })
            .collect();
        while !remaining.is_empty() {
            let mut round: Vec<(usize, i8, Vec<usize>, Saved)> = Vec::new();
            let mut used_tasks: Vec<usize> = Vec::new();
            let mut deferred = Vec::new();
            for &(t, step) in &remaining {
                let (bx, by) = tile_blocks(t)[0];
                let task = task_of_block[by * xb + bx];
                if used_tasks.contains(&task) {
                    deferred.push((t, step));
                    continue;
                }
                let (tasks, saved) = apply(dc_datas, &[(t, step)]);
                used_tasks.extend(tasks.iter().copied());
                round.push((t, step, tasks, saved));
            }
            remaining = deferred;
            let round_tasks: Vec<usize> = round.iter().flat_map(|r| r.2.iter().copied()).collect();
            compact_memos(&round_tasks);
            let dirty = evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &round_tasks,
                memo,
                cand,
            );
            for (t, step, tasks, saved) in round {
                let mut region = crate::ssimulacra2::Rect {
                    x0: usize::MAX,
                    y0: usize::MAX,
                    x1: 0,
                    y1: 0,
                };
                let mut delta_bits = 0f32;
                for &task in &tasks {
                    // pixels the re-encode changed: the cells it recomputed
                    let r = dirty[round_tasks
                        .iter()
                        .position(|&x| x == task)
                        .expect("round task")];
                    if r.x1 > r.x0 {
                        region.x0 = region.x0.min(r.x0);
                        region.y0 = region.y0.min(r.y0);
                        region.x1 = region.x1.max(r.x1);
                        region.y1 = region.y1.max(r.y1);
                    }
                    cand_gbits[task] = group_bits(cand, task);
                    delta_bits += cand_gbits[task] - cur_gbits[task];
                }
                let d = if region.x1 > 0 {
                    let dc = state.needs_dc().then(|| {
                        let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                        fill_decoded_dc(cand, distp, region, out);
                        &*out
                    });
                    state.update_checkpointed(&cand.recon, dc, region, &mut aq_checkpoint)
                } else {
                    cur_d
                };
                let bits = cur_bits + delta_bits;
                let j = -d - slope * rate_term(bits);
                let region = if region.x1 > 0 {
                    region
                } else {
                    crate::ssimulacra2::Rect {
                        x0: 0,
                        y0: 0,
                        x1: 0,
                        y1: 0,
                    }
                };
                if j > j_cur + 1e-7 {
                    moved[t] += step;
                    accepted_moves.push((t, step));
                    for &task in &tasks {
                        std::mem::swap(&mut cand.pending[task], &mut cur.pending[task]);
                        std::mem::swap(&mut cand.starts[task], &mut cur.starts[task]);
                        cur_gbits[task] = cand_gbits[task];
                    }
                    sync_tasks(cand, &mut cur, &tasks, region);
                    cur_bits = bits;
                    cur_d = d;
                    j_cur = j;
                    accepted = true;
                } else {
                    if region.x1 > 0 {
                        state.rollback(&aq_checkpoint);
                    }
                    rejected_moves.insert((t, step));
                    sync_tasks(&cur, cand, &tasks, region);
                    restore(dc_datas, saved);
                }
            }
        }
        // exact frame bits after the round's approximate per-group deltas
        cur_bits = frame_bits(&cur);
        if SHIFT_RDOQ {
            cur_map = state.map();
        }
        if !accepted {
            break;
        }
    }
    drop(greedy_eval);
    drop(cur_map);
    // ---- 128-family placement arbitrated by the loop's target ----
    // Each 128x128 site tries the 128 layouts (rect halves, rect pairs, the
    // square) on top of the current frame, re-encoded exactly (block memo,
    // frozen prices) and arbitrated by the incremental SSIMULACRA2 state at the
    // loop's exchange rate. One site per AC group is encoded per parallel round.
    if ctx.speed.effort().dct128_loop() && cached_slope > 0.0 {
        use crate::dc_group_data::{
            AcStrategyImage, STRATEGY_DCT64X128, STRATEGY_DCT128X64, STRATEGY_DCT128X128,
        };
        /// Incumbent bits per pixel of a variant's region above which no 128
        /// layout paid.
        const MAX_INCUMBENT_BPP: f32 = 1.0;
        // (strategy, dx, dy) parts of each variant, site-relative in blocks,
        // and the region (x, y, w, h) it covers. Halves come first: a pair is
        // tried only if both its halves gain, the square only if any half does.
        type Variant = (&'static [(u8, usize, usize)], (usize, usize, usize, usize));
        const HALVES: [Variant; 4] = [
            (&[(STRATEGY_DCT128X64, 0, 0)], (0, 0, 8, 16)),
            (&[(STRATEGY_DCT128X64, 8, 0)], (8, 0, 8, 16)),
            (&[(STRATEGY_DCT64X128, 0, 0)], (0, 0, 16, 8)),
            (&[(STRATEGY_DCT64X128, 0, 8)], (0, 8, 16, 8)),
        ];
        const WHOLE: (usize, usize, usize, usize) = (0, 0, 16, 16);
        const PAIRS: [(Variant, [usize; 2]); 2] = [
            (
                (
                    &[(STRATEGY_DCT128X64, 0, 0), (STRATEGY_DCT128X64, 8, 0)],
                    WHOLE,
                ),
                [0, 1],
            ),
            (
                (
                    &[(STRATEGY_DCT64X128, 0, 0), (STRATEGY_DCT64X128, 0, 8)],
                    WHOLE,
                ),
                [2, 3],
            ),
        ];
        const SQUARE: Variant = (&[(STRATEGY_DCT128X128, 0, 0)], WHOLE);

        let slope = if loop_state.as_ref().is_some_and(LoopArbiter::needs_dc) {
            cv_slope
        } else {
            cached_slope
        };
        let state = loop_state.as_mut().expect("loop judge");
        let mut checkpoint = state.checkpoint();
        let mut cand = trial_eval(&cur);
        let mut j_cur = -cur_d - slope * rate_term(cur_bits);
        // Sites of each AC task, DC-group-local block coordinates.
        let sites: Vec<Vec<(usize, usize)>> = ac_tasks
            .iter()
            .map(|&(dc_idx, gx, gy)| {
                let strategy = &dc_datas[dc_idx].ac_strategy;
                let mut v = Vec::new();
                for sy in [0, 16] {
                    for sx in [0, 16] {
                        let x = gx * K_GROUP_DIM_IN_BLOCKS + sx;
                        let y = gy * K_GROUP_DIM_IN_BLOCKS + sy;
                        if strategy.can_place_strategy(x, y, STRATEGY_DCT128X128) {
                            v.push((x, y));
                        }
                    }
                }
                v
            })
            .collect();
        /// What placing a variant overwrote: layout, field and memo of the site.
        struct Placed {
            task: usize,
            site: (usize, usize),
            cells: Vec<u8>,
            field: Vec<u8>,
            memo: Option<crate::group::MemoSnapshot>,
        }
        let distance = distp.distance;
        let place = |dc_datas: &mut [DcGroupData],
                     task: usize,
                     (x, y): (usize, usize),
                     parts: &[(u8, usize, usize)]|
         -> Placed {
            let (dc_idx, gx, gy) = ac_tasks[task];
            let data = &mut dc_datas[dc_idx];
            let cells = data.ac_strategy.save_region(x, y, 16, 16);
            let field: Vec<u8> = (y..y + 16)
                .flat_map(|r| data.raw_quant_field.row(r)[x..x + 16].to_vec())
                .collect();
            for &(s, dx, dy) in parts {
                let (px, py) = (x + dx, y + dy);
                data.ac_strategy.set_first(px, py, s);
                let w = AcStrategyImage::covered_blocks_x_of(s);
                let h = AcStrategyImage::covered_blocks_y_of(s);
                let (mut max_q, mut sum) = (0u8, 0u32);
                for r in py..py + h {
                    for &q in &data.raw_quant_field.row(r)[px..px + w] {
                        max_q = max_q.max(q);
                        sum += u32::from(q);
                    }
                }
                let q = crate::ac_strategy::aggregate_quant(max_q, sum, w * h, distance);
                for r in py..py + h {
                    data.raw_quant_field.row_mut(r)[px..px + w].fill(q);
                }
            }
            // The memo keys do not carry the layout: forget the new
            // transforms' cells, and keep the rest for an exact undo.
            let memo_snapshot = memo.and_then(|memo| {
                let mut guard = memo[task].lock().expect("block memo");
                let m = guard.as_mut()?;
                m.compact_tokens();
                let (lx, ly) = (
                    x - gx * K_GROUP_DIM_IN_BLOCKS,
                    y - gy * K_GROUP_DIM_IN_BLOCKS,
                );
                let snapshot = m.snapshot(lx, ly, 16, 16);
                for &(s, dx, dy) in parts {
                    m.invalidate(
                        lx + dx,
                        ly + dy,
                        AcStrategyImage::covered_blocks_x_of(s),
                        AcStrategyImage::covered_blocks_y_of(s),
                    );
                }
                Some(snapshot)
            });
            Placed {
                task,
                site: (x, y),
                cells,
                field,
                memo: memo_snapshot,
            }
        };
        let unplace = |dc_datas: &mut [DcGroupData], p: Placed| {
            let (dc_idx, _, _) = ac_tasks[p.task];
            let data = &mut dc_datas[dc_idx];
            let (x, y) = p.site;
            data.ac_strategy.restore_region(x, y, 16, &p.cells);
            for (i, row) in p.field.as_chunks::<16>().0.iter().enumerate() {
                data.raw_quant_field.row_mut(y + i)[x..x + 16].copy_from_slice(row);
            }
            if let (Some(memo), Some(snapshot)) = (memo, &p.memo)
                && let Some(m) = memo[p.task].lock().expect("block memo").as_mut()
            {
                m.restore(snapshot);
            }
        };
        let group_bits = |e: &Eval, task: usize| -> f32 {
            e.pending[task].as_ref().map_or(0.0, |p| {
                p.tokens[0].iter().map(|k| prices.token_bits(k)).sum()
            })
        };
        let mut cur_gbits: Vec<f32> = (0..ac_tasks.len()).map(|t| group_bits(&cur, t)).collect();
        let sync_task = |from: &Eval, to: &mut Eval, task: usize, r: crate::ssimulacra2::Rect| {
            to.quant_dc[task] = from.quant_dc[task].clone();
            to.synced[task] = from.synced[task];
            for c in 0..3 {
                for y in r.y0..r.y1 {
                    to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                        .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                }
            }
        };
        let mut accepted_sites: Vec<(usize, (usize, usize))> = Vec::new();
        let max_sites = sites.iter().map(Vec::len).max().unwrap_or(0);
        for si in 0..max_sites {
            let active: Vec<usize> = (0..ac_tasks.len())
                .filter(|&t| si < sites[t].len())
                .collect();
            let incumbent_bits = block_bits(&cur);
            let region_bpp = |t: usize, (rx, ry, rw, rh): (usize, usize, usize, usize)| {
                let (dc_gx, dc_gy) = group_coords[ac_tasks[t].0];
                let (x, y) = sites[t][si];
                let (x0, y0) = (
                    dc_gx * blocks_per_dc + x + rx,
                    dc_gy * blocks_per_dc + y + ry,
                );
                let bits: f32 = (y0..y0 + rh)
                    .map(|by| {
                        incumbent_bits[by * xb + x0..by * xb + x0 + rw]
                            .iter()
                            .sum::<f32>()
                    })
                    .sum();
                bits / (rw * rh * 64) as f32
            };
            // J of each half per task; the best variant and its J per task.
            let mut half_j = vec![[f32::NEG_INFINITY; 4]; ac_tasks.len()];
            let mut best: Vec<(f32, Option<Variant>)> = vec![(j_cur, None); ac_tasks.len()];
            // Evaluate `variant` at the tasks `keep` admits; returns each J.
            let mut try_variant = |dc_datas: &mut [DcGroupData],
                                   scratch: &mut CoderScratch,
                                   variant: Variant,
                                   keep: &dyn Fn(usize) -> bool|
             -> Vec<(usize, f32)> {
                let tasks: Vec<usize> = active
                    .iter()
                    .copied()
                    .filter(|&t| keep(t) && region_bpp(t, variant.1) <= MAX_INCUMBENT_BPP)
                    .collect();
                if tasks.is_empty() {
                    return Vec::new();
                }
                let placed: Vec<Placed> = tasks
                    .iter()
                    .map(|&t| place(dc_datas, t, sites[t][si], variant.0))
                    .collect();
                let dirty = evaluate(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    dc_datas,
                    ac_tasks,
                    group_coords,
                    num_passes,
                    coeff_shifts,
                    coeff_orders,
                    order_alternatives,
                    ytob_dc,
                    qf_threshold,
                    Some(prices),
                    &tasks,
                    memo,
                    &mut cand,
                );
                let mut out = Vec::with_capacity(tasks.len());
                for (k, &t) in tasks.iter().enumerate() {
                    let r = dirty[k];
                    let d = if r.x1 > r.x0 {
                        let dc = state.needs_dc().then(|| {
                            let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                            fill_decoded_dc(&cand, distp, r, out);
                            &*out
                        });
                        let d = state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint);
                        state.rollback(&checkpoint);
                        d
                    } else {
                        cur_d
                    };
                    let bits = cur_bits + group_bits(&cand, t) - cur_gbits[t];
                    out.push((t, -d - slope * rate_term(bits)));
                    sync_task(&cur, &mut cand, t, r);
                }
                for p in placed {
                    unplace(dc_datas, p);
                }
                out
            };
            for (h, half) in HALVES.iter().enumerate() {
                for (t, j) in try_variant(dc_datas, scratch, *half, &|_| true) {
                    half_j[t][h] = j;
                    if j > best[t].0 + 1e-7 {
                        best[t] = (j, Some(*half));
                    }
                }
            }
            for (pair, [a, b]) in PAIRS {
                let keep = |t: usize| half_j[t][a] > j_cur && half_j[t][b] > j_cur;
                for (t, j) in try_variant(dc_datas, scratch, pair, &keep) {
                    if j > best[t].0 + 1e-7 {
                        best[t] = (j, Some(pair));
                    }
                }
            }
            let keep = |t: usize| half_j[t].iter().any(|&j| j > j_cur);
            for (t, j) in try_variant(dc_datas, scratch, SQUARE, &keep) {
                if j > best[t].0 + 1e-7 {
                    best[t] = (j, Some(SQUARE));
                }
            }
            // Commit each task's winner, re-encode them together and accept
            // them one at a time on the running state.
            let winners: Vec<(usize, Variant)> = active
                .iter()
                .filter_map(|&t| best[t].1.map(|v| (t, v)))
                .collect();
            if winners.is_empty() {
                continue;
            }
            let placed: Vec<Placed> = winners
                .iter()
                .map(|&(t, v)| place(dc_datas, t, sites[t][si], v.0))
                .collect();
            let tasks: Vec<usize> = winners.iter().map(|w| w.0).collect();
            let dirty = evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &tasks,
                memo,
                &mut cand,
            );
            for ((k, &t), p) in tasks.iter().enumerate().zip(placed) {
                let r = dirty[k];
                let d = if r.x1 > r.x0 {
                    let dc = state.needs_dc().then(|| {
                        let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                        fill_decoded_dc(&cand, distp, r, out);
                        &*out
                    });
                    state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint)
                } else {
                    cur_d
                };
                let cand_gbits = group_bits(&cand, t);
                let bits = cur_bits + cand_gbits - cur_gbits[t];
                let j = -d - slope * rate_term(bits);
                if j > j_cur + 1e-7 {
                    std::mem::swap(&mut cand.pending[t], &mut cur.pending[t]);
                    std::mem::swap(&mut cand.starts[t], &mut cur.starts[t]);
                    sync_task(&cand, &mut cur, t, r);
                    cur_gbits[t] = cand_gbits;
                    cur_bits = bits;
                    cur_d = d;
                    j_cur = j;
                    accepted_sites.push((t, p.site));
                } else {
                    if r.x1 > r.x0 {
                        state.rollback(&checkpoint);
                    }
                    sync_task(&cur, &mut cand, t, r);
                    unplace(dc_datas, p);
                }
            }
        }
        // The chroma DC passes after the loop read the retained source DC,
        // which depends on the layout (a large transform's DC comes from its
        // lowest frequencies): recapture it under the accepted sites.
        let mut refresh: Vec<usize> = accepted_sites.iter().map(|a| a.0).collect();
        refresh.sort_unstable();
        refresh.dedup();
        let sources = ctx
            .thread_pool
            .steal_map(scratch, refresh.len(), |i, scratch| {
                let (dc_idx, gx, gy) = ac_tasks[refresh[i]];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (_, _, source, _, _) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    &dc_datas[dc_idx],
                    num_passes,
                    coeff_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    ytob_dc,
                    None,
                    coeff_orders,
                    &[],
                    true,
                    qf_threshold,
                    false,
                    None,
                );
                source.expect("source dc")
            });
        for (&task, source) in refresh.iter().zip(sources) {
            let (dc_idx, gx, gy) = ac_tasks[task];
            let (ox, oy) = (gx * K_GROUP_DIM_IN_BLOCKS, gy * K_GROUP_DIM_IN_BLOCKS);
            let dc = &mut dc_datas[dc_idx];
            for &(_, (x, y)) in accepted_sites.iter().filter(|a| a.0 == task) {
                for (target, src) in [
                    (&mut dc.source_dc_y, &source.y),
                    (&mut dc.source_dc_b, &source.b),
                    (&mut dc.source_dc_x, &source.x),
                ] {
                    if let Some(target) = target.as_mut() {
                        for r in y..y + 16 {
                            target.row_mut(r)[x..x + 16]
                                .copy_from_slice(&src.row(r - oy)[x - ox..x - ox + 16]);
                        }
                    }
                }
            }
        }
        cur_bits = frame_bits(&cur);
    }
    // ---- 16x16 layouts arbitrated by the loop's target ----
    // Every closed 2x2-block site (no transform crosses it) tries the other
    // layouts of its family: four DCT8, DCT16X16 and the two rectangle pairs.
    // Each is re-encoded exactly (block memo, frozen prices; cached blocks
    // contribute their stored bits instead of their tokens) and arbitrated by the
    // incremental SSIMULACRA2 state at the loop's exchange rate. One site per
    // AC group per parallel round; the best variant is re-encoded in full and
    // kept if it still improves.
    if ctx.speed.effort().layout_loop() && cached_slope > 0.0 {
        use crate::dc_group_data::{
            AcStrategyImage, NUM_STRATEGIES, STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16,
        };
        type Parts = Vec<(u8, usize, usize)>;
        let cv_judged = loop_state.as_ref().is_some_and(LoopArbiter::needs_dc);
        let slope = if cv_judged { cv_slope } else { cached_slope };
        let state = loop_state.as_mut().expect("loop judge");
        let mut checkpoint = state.checkpoint();
        let mut cand = trial_eval(&cur);
        let mut j_cur = -cur_d - slope * rate_term(cur_bits);
        // Only layouts whose classes the frame already uses (matrices, orders).
        let class = |st: usize| {
            (
                crate::quant_weights::quant_table_slot_of(st as u8),
                crate::ac_context::K_STRATEGY_ORDER
                    [crate::dc_group_data::STRATEGY_CODE_LUT[st] as usize],
            )
        };
        let mut used = [false; NUM_STRATEGIES];
        for d in dc_datas.iter() {
            for (_, _, st) in d.ac_strategy.iter_first_blocks() {
                used[st as usize] = true;
            }
        }
        // A transpose shares both classes with the transform the frame uses.
        let present: [bool; NUM_STRATEGIES] = std::array::from_fn(|st| {
            used[st]
                || class(st).0.is_some()
                    && (0..NUM_STRATEGIES).any(|o| used[o] && class(o) == class(st))
        });
        let mut variants: Vec<Parts> = vec![vec![
            (STRATEGY_DCT, 0, 0),
            (STRATEGY_DCT, 1, 0),
            (STRATEGY_DCT, 0, 1),
            (STRATEGY_DCT, 1, 1),
        ]];
        if present[STRATEGY_DCT16X16 as usize] {
            variants.push(vec![(STRATEGY_DCT16X16, 0, 0)]);
        }
        if present[STRATEGY_DCT16X8 as usize] {
            variants.push(vec![(STRATEGY_DCT16X8, 0, 0), (STRATEGY_DCT16X8, 1, 0)]);
        }
        if present[STRATEGY_DCT8X16 as usize] {
            variants.push(vec![(STRATEGY_DCT8X16, 0, 0), (STRATEGY_DCT8X16, 0, 1)]);
        }
        /// A region the loop re-chooses: a 2x2 site with the 16x16-family
        /// variants, or a region the exact re-tiling proposed, with that
        /// proposal as its only variant.
        struct Site {
            x: usize,
            y: usize,
            w: usize,
            h: usize,
            proposal: Option<Parts>,
        }
        // Sites of each AC task, DC-group-local block coordinates: the
        // aligned 2x2 grid, the three odd-offset grids (the format allows
        // unaligned blocks; none may cross a 64px CfL tile), then the
        // proposed regions, judged against the refined layout.
        let sites: Vec<Vec<Site>> = ac_tasks
            .iter()
            .map(|&(dc_idx, gx, gy)| {
                let st = &dc_datas[dc_idx].ac_strategy;
                let mut v = Vec::new();
                for (px, py) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    for sy in (py..K_GROUP_DIM_IN_BLOCKS - 1).step_by(2) {
                        for sx in (px..K_GROUP_DIM_IN_BLOCKS - 1).step_by(2) {
                            if sx % K_TILE_DIM_IN_BLOCKS == K_TILE_DIM_IN_BLOCKS - 1
                                || sy % K_TILE_DIM_IN_BLOCKS == K_TILE_DIM_IN_BLOCKS - 1
                            {
                                continue;
                            }
                            let x = gx * K_GROUP_DIM_IN_BLOCKS + sx;
                            let y = gy * K_GROUP_DIM_IN_BLOCKS + sy;
                            if x + 2 <= st.xsize() && y + 2 <= st.ysize() {
                                v.push(Site {
                                    x,
                                    y,
                                    w: 2,
                                    h: 2,
                                    proposal: None,
                                });
                            }
                        }
                    }
                }
                for region in &dc_datas[dc_idx].dp_regions {
                    let inside = region.bx / K_GROUP_DIM_IN_BLOCKS == gx
                        && region.by / K_GROUP_DIM_IN_BLOCKS == gy;
                    if inside && region.pieces.iter().all(|&(st, _, _)| present[st as usize]) {
                        v.push(Site {
                            x: region.bx,
                            y: region.by,
                            w: region.w,
                            h: region.h,
                            proposal: Some(region.pieces.clone()),
                        });
                    }
                }
                v
            })
            .collect();
        // The current layout of a closed site, sorted like `variants`.
        let site_parts = |st: &AcStrategyImage, site: &Site| -> Option<Parts> {
            let mut parts = Vec::new();
            let mut covered = 0;
            for dy in 0..site.h {
                for dx in 0..site.w {
                    if st.is_first_block(site.x + dx, site.y + dy) {
                        let raw = st.raw_strategy(site.x + dx, site.y + dy);
                        let cw = AcStrategyImage::covered_blocks_x_of(raw);
                        let ch = AcStrategyImage::covered_blocks_y_of(raw);
                        if dx + cw > site.w || dy + ch > site.h {
                            return None;
                        }
                        covered += cw * ch;
                        parts.push((raw, dx, dy));
                    }
                }
            }
            (covered == site.w * site.h).then_some(parts)
        };
        /// What placing a variant overwrote: layout, field and memo of the site.
        struct Placed {
            task: usize,
            site: (usize, usize),
            size: (usize, usize),
            cells: Vec<u8>,
            field: Vec<u8>,
            memo: Option<crate::group::MemoSnapshot>,
        }
        let distance = distp.distance;
        let place =
            |dc_datas: &mut [DcGroupData], task: usize, site: &Site, parts: &Parts| -> Placed {
                let (dc_idx, gx, gy) = ac_tasks[task];
                let data = &mut dc_datas[dc_idx];
                let (x, y, w, h) = (site.x, site.y, site.w, site.h);
                let cells = data.ac_strategy.save_region(x, y, w, h);
                let field: Vec<u8> = (y..y + h)
                    .flat_map(|r| data.raw_quant_field.row(r)[x..x + w].to_vec())
                    .collect();
                for &(st, dx, dy) in parts {
                    let (px, py) = (x + dx, y + dy);
                    data.ac_strategy.set_first(px, py, st);
                    let w = AcStrategyImage::covered_blocks_x_of(st);
                    let h = AcStrategyImage::covered_blocks_y_of(st);
                    if w * h > 1 {
                        let (mut max_q, mut sum) = (0u8, 0u32);
                        for r in py..py + h {
                            for &q in &data.raw_quant_field.row(r)[px..px + w] {
                                max_q = max_q.max(q);
                                sum += u32::from(q);
                            }
                        }
                        let q = crate::ac_strategy::aggregate_quant(max_q, sum, w * h, distance);
                        for r in py..py + h {
                            data.raw_quant_field.row_mut(r)[px..px + w].fill(q);
                        }
                    }
                }
                let memo_snapshot = memo.and_then(|memo| {
                    let mut guard = memo[task].lock().expect("block memo");
                    let m = guard.as_mut()?;
                    m.compact_tokens();
                    let (lx, ly) = (
                        x - gx * K_GROUP_DIM_IN_BLOCKS,
                        y - gy * K_GROUP_DIM_IN_BLOCKS,
                    );
                    let snapshot = m.snapshot(lx, ly, w, h);
                    m.invalidate(lx, ly, w, h);
                    Some(snapshot)
                });
                Placed {
                    task,
                    site: (x, y),
                    size: (w, h),
                    cells,
                    field,
                    memo: memo_snapshot,
                }
            };
        let unplace = |dc_datas: &mut [DcGroupData], p: Placed| {
            let (dc_idx, _, _) = ac_tasks[p.task];
            let data = &mut dc_datas[dc_idx];
            let (x, y) = p.site;
            let (w, _) = p.size;
            data.ac_strategy.restore_region(x, y, w, &p.cells);
            for (i, row) in p.field.chunks_exact(w).enumerate() {
                data.raw_quant_field.row_mut(y + i)[x..x + w].copy_from_slice(row);
            }
            if let (Some(memo), Some(snapshot)) = (memo, &p.memo)
                && let Some(m) = memo[p.task].lock().expect("block memo").as_mut()
            {
                m.restore(snapshot);
            }
        };
        let group_bits = |e: &Eval, task: usize| -> f32 {
            e.hit_bits[task]
                + e.pending[task].as_ref().map_or(0.0, |p| {
                    p.tokens[0]
                        .iter()
                        .map(|k| prices.token_bits(k))
                        .sum::<f32>()
                })
        };
        // A rejected trial restores only the pixels and the sync point: the
        // task's next evaluation overwrites its tokens.
        let restore_task =
            |from: &Eval, to: &mut Eval, task: usize, r: crate::ssimulacra2::Rect| {
                to.synced[task] = from.synced[task];
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                            .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                    }
                }
            };
        let accept_task =
            |from: &mut Eval, to: &mut Eval, task: usize, r: crate::ssimulacra2::Rect| {
                std::mem::swap(&mut to.pending[task], &mut from.pending[task]);
                std::mem::swap(&mut to.quant_dc[task], &mut from.quant_dc[task]);
                std::mem::swap(&mut to.starts[task], &mut from.starts[task]);
                to.synced[task] = from.synced[task];
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                            .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                    }
                }
            };
        let mut cur_gbits: Vec<f32> = (0..ac_tasks.len()).map(|t| group_bits(&cur, t)).collect();
        /// An accepted site: its task, origin and size in blocks.
        type Accepted = (usize, (usize, usize), (usize, usize));
        let mut accepted_sites: Vec<Accepted> = Vec::new();
        let max_sites = sites.iter().map(Vec::len).max().unwrap_or(0);
        for si in 0..max_sites {
            // The variants of each task's site that differ from its layout.
            let vars: Vec<Vec<&Parts>> = (0..ac_tasks.len())
                .map(|t| {
                    let Some(site) = sites[t].get(si) else {
                        return Vec::new();
                    };
                    let st = &dc_datas[ac_tasks[t].0].ac_strategy;
                    let Some(mut current) = site_parts(st, site) else {
                        return Vec::new();
                    };
                    current.sort_unstable();
                    let differs = |v: &Parts| {
                        let mut k = v.clone();
                        k.sort_unstable();
                        k != current
                    };
                    match &site.proposal {
                        Some(proposal) => {
                            if differs(proposal) {
                                vec![proposal]
                            } else {
                                Vec::new()
                            }
                        }
                        None => variants.iter().filter(|v| differs(v)).collect(),
                    }
                })
                .collect();
            let max_v = vars.iter().map(Vec::len).max().unwrap_or(0);
            let mut best: Vec<(f32, Option<usize>)> = vec![(j_cur, None); ac_tasks.len()];
            for vi in 0..max_v {
                let tasks: Vec<usize> = (0..ac_tasks.len())
                    .filter(|&t| vi < vars[t].len())
                    .collect();
                let placed: Vec<Placed> = tasks
                    .iter()
                    .map(|&t| place(dc_datas, t, &sites[t][si], vars[t][vi]))
                    .collect();
                cand.trial = true;
                let dirty = evaluate(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    dc_datas,
                    ac_tasks,
                    group_coords,
                    num_passes,
                    coeff_shifts,
                    coeff_orders,
                    order_alternatives,
                    ytob_dc,
                    qf_threshold,
                    Some(prices),
                    &tasks,
                    memo,
                    &mut cand,
                );
                cand.trial = false;
                for (k, &t) in tasks.iter().enumerate() {
                    let r = dirty[k];
                    let d = if r.x1 > r.x0 {
                        let dc = state.needs_dc().then(|| {
                            let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                            fill_decoded_dc(&cand, distp, r, out);
                            &*out
                        });
                        let d = state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint);
                        state.rollback(&checkpoint);
                        d
                    } else {
                        cur_d
                    };
                    let bits = cur_bits + group_bits(&cand, t) - cur_gbits[t];
                    let j = -d - slope * rate_term(bits);
                    if j > best[t].0 + 1e-7 {
                        best[t] = (j, Some(vi));
                    }
                    restore_task(&cur, &mut cand, t, r);
                }
                for p in placed {
                    unplace(dc_datas, p);
                }
            }
            // Commit each task's winner, re-encode them in full and accept
            // them one at a time on the running state.
            let winners: Vec<(usize, usize)> = (0..ac_tasks.len())
                .filter_map(|t| best[t].1.map(|v| (t, v)))
                .collect();
            if winners.is_empty() {
                continue;
            }
            let placed: Vec<Placed> = winners
                .iter()
                .map(|&(t, v)| place(dc_datas, t, &sites[t][si], vars[t][v]))
                .collect();
            let tasks: Vec<usize> = winners.iter().map(|w| w.0).collect();
            let dirty = evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &tasks,
                memo,
                &mut cand,
            );
            for ((k, &t), p) in tasks.iter().enumerate().zip(placed) {
                let r = dirty[k];
                let d = if r.x1 > r.x0 {
                    let dc = state.needs_dc().then(|| {
                        let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                        fill_decoded_dc(&cand, distp, r, out);
                        &*out
                    });
                    state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint)
                } else {
                    cur_d
                };
                let cand_gbits = group_bits(&cand, t);
                let bits = cur_bits + cand_gbits - cur_gbits[t];
                let j = -d - slope * rate_term(bits);
                if j > j_cur + 1e-7 {
                    accept_task(&mut cand, &mut cur, t, r);
                    cur_gbits[t] = cand_gbits;
                    cur_bits = bits;
                    cur_d = d;
                    j_cur = j;
                    accepted_sites.push((t, p.site, p.size));
                } else {
                    if r.x1 > r.x0 {
                        state.rollback(&checkpoint);
                    }
                    restore_task(&cur, &mut cand, t, r);
                    unplace(dc_datas, p);
                }
            }
        }
        // The chroma DC passes after the loop read the retained source DC,
        // which depends on the layout: recapture it under the accepted sites.
        let mut refresh: Vec<usize> = accepted_sites.iter().map(|a| a.0).collect();
        refresh.sort_unstable();
        refresh.dedup();
        let sources = ctx
            .thread_pool
            .steal_map(scratch, refresh.len(), |i, scratch| {
                let (dc_idx, gx, gy) = ac_tasks[refresh[i]];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (_, _, source, _, _) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    &dc_datas[dc_idx],
                    num_passes,
                    coeff_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    ytob_dc,
                    None,
                    coeff_orders,
                    &[],
                    true,
                    qf_threshold,
                    false,
                    None,
                );
                source.expect("source dc")
            });
        for (&task, source) in refresh.iter().zip(sources) {
            let (dc_idx, gx, gy) = ac_tasks[task];
            let (ox, oy) = (gx * K_GROUP_DIM_IN_BLOCKS, gy * K_GROUP_DIM_IN_BLOCKS);
            let dc = &mut dc_datas[dc_idx];
            for &(_, (x, y), (w, h)) in accepted_sites.iter().filter(|a| a.0 == task) {
                for (target, src) in [
                    (&mut dc.source_dc_y, &source.y),
                    (&mut dc.source_dc_b, &source.b),
                    (&mut dc.source_dc_x, &source.x),
                ] {
                    if let Some(target) = target.as_mut() {
                        for r in y..y + h {
                            target.row_mut(r)[x..x + w]
                                .copy_from_slice(&src.row(r - oy)[x - ox..x - ox + w]);
                        }
                    }
                }
            }
        }
        cur_bits = frame_bits(&cur);
    }
    // ---- Frame-level chroma precision race ----
    // The X and B quant-matrix scales are 3-bit header fields set by distance
    // and content gates (higher = finer). A coordinate search, X then B, arbitrates
    // each neighbor of the current pair on the loop's target with a full
    // priced encode; the block memo keys the chroma channels by their scale, so
    // an arm recomputes only the channel it changes.
    let mut distp_out = *distp;
    // Each arbiter races on its own metric and exchange rate.
    let race_slope = if loop_state.as_ref().is_some_and(LoopArbiter::needs_dc) {
        cv_slope
    } else {
        cached_slope
    };
    if race_slope > 0.0 && ctx.coding.is_xyb() {
        let slope = race_slope;
        let b_start = ctx.b_qm_scale();
        // No more regional updates: full-frame comparisons do not read the
        // incremental pyramid, or any accepted evaluation's reconstruction.
        let mut judge = match loop_state.take() {
            Some(LoopArbiter::Cvvdp(mut judge)) => {
                judge.finish_updates();
                Some(judge)
            }
            _ => None,
        };
        cur.recon = Image3F::new(0, 0);
        let mut score = |e: &Eval| -> f32 {
            match judge.as_deref_mut() {
                Some(judge) => judge.deficit_of(&e.recon, &decoded_dc(e, distp)),
                None => reference.deficit(&e.recon),
            }
        };
        let arm_eval = |x: u32, b: u32, scratch: &mut CoderScratch| -> (Eval, f32) {
            let mut dp = *distp;
            dp.x_qm_scale = x;
            ctx.set_b_qm_scale(b);
            let mut e = new_eval();
            compact_memos(&all_tasks);
            evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                &dp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &all_tasks,
                memo,
                &mut e,
            );
            ctx.set_b_qm_scale(b_start);
            let bits = frame_bits(&e);
            (e, bits)
        };
        let cur_score = cur_d;
        let mut best = (
            -cur_score - slope * rate_term(cur_bits),
            distp.x_qm_scale,
            b_start,
        );
        let mut tried: Vec<(u32, u32)> = vec![(best.1, best.2)];
        let mut best_eval: Option<Eval> = None;
        for coord in 0..2usize {
            for dir in [1i32, -1] {
                loop {
                    let (mut x, mut b) = (best.1 as i32, best.2 as i32);
                    if coord == 0 {
                        x += dir;
                    } else {
                        b += dir;
                    }
                    if !(1..=7).contains(&x) || !(2..=7).contains(&b) {
                        break;
                    }
                    let (x, b) = (x as u32, b as u32);
                    // The reverse direction after a walk re-visits a point already scored.
                    if tried.contains(&(x, b)) {
                        break;
                    }
                    let (mut e, bits) = arm_eval(x, b, scratch);
                    let d = score(&e);
                    let j = -d - slope * rate_term(bits);
                    tried.push((x, b));
                    if j > best.0 + 1e-6 {
                        best = (j, x, b);
                        e.recon = Image3F::new(0, 0);
                        best_eval = Some(e);
                    } else {
                        break;
                    }
                }
            }
        }
        if let Some(e) = best_eval {
            cur = e;
            distp_out.x_qm_scale = best.1;
            ctx.set_b_qm_scale(best.2);
        }
    }
    finish(cur, distp_out)
}

#[allow(clippy::too_many_arguments)]
/// Set up one DC group (quant field, AC strategy, CfL, sub-8 gate); its AC
/// groups are encoded separately. Returns the data and its group-grid dims.
fn setup_dc_group(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    dc_gx: usize,
    dc_gy: usize,
    num_threads: usize,
) -> Result<(DcGroupData, usize, usize), EncodeError> {
    // DC group rect in pixels (clamped to image bounds).
    let dc_group_x0 = dc_gx * K_DC_GROUP_DIM;
    let dc_group_y0 = dc_gy * K_DC_GROUP_DIM;
    let dc_group_xsize = K_DC_GROUP_DIM.min(dim.xsize.saturating_sub(dc_group_x0));
    let dc_group_ysize = K_DC_GROUP_DIM.min(dim.ysize.saturating_sub(dc_group_y0));
    let dc_group_xsize_blocks = dc_group_xsize.div_ceil(K_BLOCK_DIM);
    let dc_group_ysize_blocks = dc_group_ysize.div_ceil(K_BLOCK_DIM);
    let dc_group_xsize_groups = dc_group_xsize.div_ceil(K_GROUP_DIM);
    let dc_group_ysize_groups = dc_group_ysize.div_ceil(K_GROUP_DIM);

    let mut dc_data = DcGroupData::new(dc_group_xsize_blocks, dc_group_ysize_blocks)?;

    (ctx.fill_quant_field)(
        &mut scratch.aq_map,
        ctx.aq_source.get().unwrap_or(opsin),
        &mut dc_data.raw_quant_field,
        dc_group_x0,
        dc_group_y0,
        distp.distance,
        1.0 / distp.scale,
    );

    if ctx.speed.effort().chroma_policy && (ctx.x_heavy() || ctx.b_heavy()) {
        crate::adaptive_quant::apply_chroma_hf_protection(
            opsin,
            &mut dc_data.raw_quant_field,
            dc_group_x0,
            dc_group_y0,
            distp.distance,
        );
    }

    // Compute the per-tile CfL slopes before strategy selection so candidate
    // costs use the same Y-to-X/Y-to-B subtraction as final coefficient coding.
    crate::color_correlation::fill_cmap(
        ctx,
        opsin,
        dc_group_x0 / K_BLOCK_DIM,
        dc_group_y0 / K_BLOCK_DIM,
        dc_group_xsize_blocks,
        dc_group_ysize_blocks,
        &dc_data.raw_quant_field,
        distp.scale,
        &mut scratch.cfl_rdo,
        &mut dc_data.ytox_map,
        &mut dc_data.ytob_map,
        distp.distance,
        distp.x_qm_scale,
    );
    let mut fine_rollbacks = Vec::new();
    let mut fine_fallbacks = Vec::new();
    dc_data.block_quant_field = Some(dc_data.raw_quant_field.clone());
    dc_data.sub8_benefit = crate::ac_strategy::fill_ac_strategy(
        ctx,
        scratch,
        opsin,
        dc_group_x0,
        dc_group_y0,
        distp.distance,
        distp.scale,
        distp.x_qm_scale,
        &mut dc_data.raw_quant_field,
        &dc_data.ytox_map,
        &dc_data.ytob_map,
        &mut dc_data.ac_strategy,
        &mut fine_rollbacks,
        &mut fine_fallbacks,
        &mut dc_data.dp_regions,
        num_threads,
    );
    // A fine mosaic that split an otherwise-retained merge must repay its own
    // exact metadata delta. Keep this separate from the legacy sub-8 gate so
    // unrelated fine-block gains cannot subsidize a weak mosaic decision.
    let fine_accepted = if fine_rollbacks.is_empty() {
        false
    } else {
        let fine_benefit: f32 = fine_rollbacks.iter().map(|r| r.benefit).sum();
        let cost_with = meta_entropy_cost(&dc_data, scratch, distp.distance);
        let mut fine_quants = Vec::with_capacity(fine_rollbacks.len());
        for rollback in &fine_rollbacks {
            let mut quants = [0u8; 16];
            for iy in 0..rollback.cov_y {
                let row = dc_data.raw_quant_field.row_mut(rollback.by + iy);
                let cells = &mut row[rollback.bx..rollback.bx + rollback.cov_x];
                quants[iy * 4..iy * 4 + rollback.cov_x].copy_from_slice(cells);
                cells.fill(rollback.quant);
            }
            fine_quants.push(quants);
            dc_data
                .ac_strategy
                .set_first(rollback.bx, rollback.by, rollback.strategy);
        }
        let cost_without = meta_entropy_cost(&dc_data, scratch, distp.distance);
        let meta_delta = cost_with.saturating_sub(cost_without) as f32;
        let fine_lambda = crate::ac_strategy::fine_mosaic_lambda(distp.distance);
        let accepted = fine_benefit > fine_lambda * meta_delta;
        if accepted {
            for (rollback, quants) in fine_rollbacks.iter().zip(&fine_quants) {
                for iy in 0..rollback.cov_y {
                    dc_data.raw_quant_field.row_mut(rollback.by + iy)
                        [rollback.bx..rollback.bx + rollback.cov_x]
                        .copy_from_slice(&quants[iy * 4..iy * 4 + rollback.cov_x]);
                    for ix in 0..rollback.cov_x {
                        dc_data.ac_strategy.set_first(
                            rollback.bx + ix,
                            rollback.by + iy,
                            rollback.fine_grid[iy * 4 + ix],
                        );
                    }
                }
            }
            true
        } else {
            false
        }
    };

    // Sub-8x8 activation gate. `fill_ac_strategy` greedily commits every block
    // where a fine 8x8-family strategy wins the per-block RD comparison, but a
    // sparse set can disrupt prefix-code clustering of the (otherwise nearly
    // free) AC-strategy meta stream. Measure the *real* meta-token cost with the
    // exact selected set vs with all of it reverted to DCT8, and retain the set
    // only when its accumulated RD benefit covers the metadata increase. Done
    // before `write_ac_group`, so a rejected set cannot affect coefficients.
    {
        let mut positions: Vec<(usize, usize, u8)> = Vec::new();
        for y in 0..dc_data.ac_strategy.ysize() {
            for x in 0..dc_data.ac_strategy.xsize() {
                if dc_data.ac_strategy.is_first_block(x, y) {
                    let strategy = dc_data.ac_strategy.raw_strategy(x, y);
                    let in_accepted_mosaic = fine_accepted
                        && fine_rollbacks.iter().any(|rollback| {
                            x >= rollback.bx
                                && x < rollback.bx + rollback.cov_x
                                && y >= rollback.by
                                && y < rollback.by + rollback.cov_y
                        });
                    if crate::dc_group_data::is_gated_sub8_strategy(strategy) && !in_accepted_mosaic
                    {
                        positions.push((x, y, strategy));
                    }
                }
            }
        }
        if !positions.is_empty() {
            let cost_with = meta_entropy_cost(&dc_data, scratch, distp.distance);
            let xsize = dc_data.ac_strategy.xsize();
            for &(x, y, _) in &positions {
                let fallback = fine_fallbacks
                    .get(y * xsize + x)
                    .copied()
                    .unwrap_or(STRATEGY_DCT);
                dc_data.ac_strategy.set_first(x, y, fallback);
            }
            let cost_without = meta_entropy_cost(&dc_data, scratch, distp.distance);
            let meta_delta = cost_with.saturating_sub(cost_without) as f32;
            if dc_data.sub8_benefit > crate::ac_strategy::RD_LAMBDA * meta_delta {
                // Worth it: restore each exact sub-8x8 selection.
                for &(x, y, strategy) in &positions {
                    dc_data.ac_strategy.set_first(x, y, strategy);
                }
            }
            // Else leave ordinary fine blocks on the incumbent they replaced.
        }
    }
    Ok((dc_data, dc_group_xsize_groups, dc_group_ysize_groups))
}

/// Merge an AC group's origin-relative `quant_dc` into its parent DC group.
fn merge_quant_dc(dc: &mut DcGroupData, gx: usize, gy: usize, local: &Image3S) {
    let ox = gx * K_GROUP_DIM_IN_BLOCKS;
    let oy = gy * K_GROUP_DIM_IN_BLOCKS;
    let (gwb, ghb) = (local.xsize(), local.ysize());
    for c in 0..3 {
        for ly in 0..ghb {
            let src = local.plane_row(c, ly);
            dc.quant_dc.plane_row_mut(c, oy + ly)[ox..ox + gwb].copy_from_slice(&src[..gwb]);
        }
    }
}

/// DC-side sample budget of the MA-learned DC arm; larger DC images are
/// sampled with an odd stride.
const MA_DC_TARGET_SAMPLES: usize = 1 << 19;
/// Leaf budget of the MA-learned DC side (the metadata side adds at most
/// ~25 more; context ids stay below 256).
const MA_DC_MAX_LEAVES: usize = 160;
/// Estimated header bits of one more DC leaf: its tree tokens, context-map
/// entry and a share of a histogram.
const MA_DC_SPLIT_COST_BITS: f32 = 50.0;
const MA_DC_MIN_NODE_SAMPLES: usize = 64;

/// Learn the DC side of the global tree with the lossless MA learner (full
/// property set, every predictor) and tokenize the DC groups and the
/// metadata streams through the grafted tree. Returns the candidate
/// (DC tokens, serialized tree) pairs — without and, when any leaf has one,
/// with median leaf offsets — the metadata tokens and the context count.
#[allow(clippy::type_complexity)]
fn learn_ma_dc_arm(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    meta_tokens: &[Vec<Token>],
    meta_props: &[Vec<crate::dc_tree::DcProp>],
) -> Option<(Vec<(Vec<Vec<Token>>, Vec<Token>)>, Vec<Vec<Token>>, usize)> {
    let groups: Vec<crate::lossless::DcPlanes> = dc_datas
        .iter()
        .map(|dc| {
            let q = &dc.quant_dc;
            let plane = |c: usize| {
                q.plane(c)
                    .as_slice()
                    .iter()
                    .map(|&v| i32::from(v))
                    .collect()
            };
            crate::lossless::DcPlanes {
                planes: [plane(1), plane(0), plane(2)],
                w: q.xsize(),
                h: q.ysize(),
            }
        })
        .collect();
    let total: usize = groups.iter().map(|g| 3 * g.w * g.h).sum();
    let mut stride = total.div_ceil(MA_DC_TARGET_SAMPLES).max(1);
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    let samples = crate::lossless::sample_dc_planes(&groups, stride);
    if samples.len() < 4 * MA_DC_MIN_NODE_SAMPLES {
        return None;
    }
    let alphabet = samples
        .tok
        .iter()
        .flat_map(|t| t.iter())
        .copied()
        .max()
        .unwrap_or(0) as usize
        + 1;
    let params = crate::ma_tree::MaLearnParams {
        alphabet,
        max_leaves: MA_DC_MAX_LEAVES,
        split_cost_bits: MA_DC_SPLIT_COST_BITS / stride as f32,
        min_node: MA_DC_MIN_NODE_SAMPLES,
        allow_wp: true,
        allowed_preds: u16::MAX,
        side_preds: 4,
        max_candidates: 32,
        modal_offsets: false,
        prune_leaf_bits: 0.0,
    };
    let tree = crate::ma_tree::learn_ma_tree(&samples, params, &ctx.thread_pool, scratch);
    drop(samples);
    let mut node_offsets = vec![0i32; tree.nodes.len()];
    let mut grafted = crate::dc_tree::graft_ma_dc_tree(
        dc_datas.len(),
        &tree,
        &node_offsets,
        meta_tokens,
        meta_props,
    );
    let mut dc_tokens = ctx
        .thread_pool
        .steal_map(scratch, groups.len(), |i, _scratch| {
            let g = &groups[i];
            let mut out = Vec::with_capacity(3 * g.w * g.h);
            crate::lossless::tokenize_dc_planes(g, i, &tree, &grafted.leaf_ctx, &mut out);
            out
        });
    // Residuals do not depend on the offset: center each leaf on its median
    // and re-serialize the tree with the offsets (context numbering is fixed
    // by the tree's shape). Offsets only pay through clustering, so both
    // variants are priced.
    let num_contexts = grafted.num_contexts;
    let meta_context = std::mem::take(&mut grafted.meta_context);
    let mut variants = vec![(dc_tokens.clone(), grafted.tokens.clone())];
    let ctx_offsets = crate::dc_tree::median_leaf_offsets(&dc_tokens, num_contexts);
    if ctx_offsets.iter().any(|&o| o != 0) {
        for (node, &c) in grafted.leaf_ctx.iter().enumerate() {
            if c != u32::MAX {
                node_offsets[node] = ctx_offsets[c as usize];
            }
        }
        crate::dc_tree::apply_leaf_offsets(&mut dc_tokens, &ctx_offsets);
        grafted = crate::dc_tree::graft_ma_dc_tree(
            dc_datas.len(),
            &tree,
            &node_offsets,
            meta_tokens,
            meta_props,
        );
        variants.push((dc_tokens, grafted.tokens));
    }
    let meta_tokens = meta_tokens
        .iter()
        .zip(meta_props)
        .map(|(group, props)| {
            group
                .iter()
                .zip(props)
                .map(|(t, &p)| {
                    let slot = ((t.context as usize) << 10) | (p & 1023) as usize;
                    Token::new(u32::from(meta_context[slot]), t.value)
                })
                .collect()
        })
        .collect();
    Some((variants, meta_tokens, num_contexts))
}

/// Encode a single AC group: build its tile stripes, quantize and tokenize,
/// and place its DC coefficients into a returned group-local `quant_dc`
/// (origin-relative, merged by the caller). Reads `dc_data` read-only.
#[allow(clippy::too_many_arguments)]
/// Median raw quant-field value across the image (blocks weighted equally).
fn ac_qf_threshold(dc_datas: &[DcGroupData]) -> u32 {
    let mut histogram = [0u64; 257];
    for dc_data in dc_datas {
        for y in 0..dc_data.raw_quant_field.ysize() {
            for &qf in dc_data.raw_quant_field.row(y) {
                histogram[(qf as usize).min(256)] += 1;
            }
        }
    }
    let total: u64 = histogram.iter().sum();
    let mut acc = 0u64;
    for (value, &count) in histogram.iter().enumerate() {
        acc += count;
        if acc * 2 >= total {
            return value as u32;
        }
    }
    1
}

fn process_ac_group(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    dc_data: &DcGroupData,
    num_passes: usize,
    coeff_shifts: &[u32],
    dc_gx: usize,
    dc_gy: usize,
    gx: usize,
    gy: usize,
    ytob_dc: i32,
    rdoq_prices: Option<&crate::entropy::FrozenTokenPrices>,
    coeff_orders: &crate::coeff_order::CoeffOrders,
    alternative_orders: &[crate::coeff_order::CoeffOrders],
    collect_order_stats: bool,
    qf_threshold: u32,
    capture_recon: bool,
    block_memo: Option<&mut Option<crate::group::AcBlockCache>>,
) -> ProcessedAcGroup {
    scratch.finish_strategy_search();
    let image_gx = dc_gx * (K_DC_GROUP_DIM / K_GROUP_DIM) + gx;
    let image_gy = dc_gy * (K_DC_GROUP_DIM / K_GROUP_DIM) + gy;
    let group_x0 = image_gx * K_GROUP_DIM;
    let group_y0 = image_gy * K_GROUP_DIM;
    let group_xsize = K_GROUP_DIM.min(dim.xsize.saturating_sub(group_x0));
    let group_ysize = K_GROUP_DIM.min(dim.ysize.saturating_sub(group_y0));
    let group_ysize_tiles = group_ysize.div_ceil(K_TILE_DIM);
    let gwb = group_xsize.div_ceil(K_BLOCK_DIM);
    let ghb = group_ysize.div_ceil(K_BLOCK_DIM);
    let qorigin_x = gx * K_GROUP_DIM_IN_BLOCKS;
    let qorigin_y = gy * K_GROUP_DIM_IN_BLOCKS;

    let mut local_quant_dc = Image3S::new(gwb, ghb);
    let mut source_dc = collect_order_stats.then(|| crate::group::SourceDc {
        y: crate::image::Plane::new(gwb, ghb),
        b: crate::image::Plane::new(gwb, ghb),
        x: crate::image::Plane::new(gwb, ghb),
    });
    let mut num_nzeros: Vec<Image3B> = (0..num_passes)
        .map(|_| Image3B::new(K_GROUP_DIM_IN_BLOCKS, K_GROUP_DIM_IN_BLOCKS))
        .collect();
    // Alternative streams walk the final coefficients without quantizing again.
    debug_assert!(alternative_orders.is_empty() || num_passes == 1);
    let mut tokens: Vec<Vec<Token>> = (0..num_passes + alternative_orders.len())
        .map(|_| Vec::with_capacity(K_GROUP_DIM_IN_BLOCKS * K_GROUP_DIM_IN_BLOCKS * 4))
        .collect();
    let mut order_stats = collect_order_stats.then(crate::coeff_order::OrderStats::new);
    // In memo mode the memo's reconstruction is the capture target: hits copy
    // no pixels and the caller syncs only the recomputed cells.
    debug_assert!(block_memo.is_none() || capture_recon);
    let mut memo = block_memo.map(|slot| {
        slot.get_or_insert_with(|| {
            crate::group::AcBlockCache::new(gwb, ghb, group_xsize, group_ysize)
        })
    });
    if let Some(m) = memo.as_deref_mut() {
        m.epoch += 1;
    }
    let mut recon = capture_recon.then(|| {
        let plane = match memo.as_deref_mut() {
            Some(m) => std::mem::replace(m.recon_mut(), Image3F::new(0, 0)),
            None => Image3F::new(group_xsize, group_ysize),
        };
        (plane, Vec::new())
    });

    for ty in 0..group_ysize_tiles {
        let stripe_x0 = group_x0;
        let stripe_y0 = group_y0 + ty * K_TILE_DIM;
        let stripe_xsize = group_xsize;
        let stripe_ysize = K_TILE_DIM.min(dim.ysize.saturating_sub(stripe_y0));
        let stripe_brect = Rect::new(
            qorigin_x,
            qorigin_y + ty * K_TILE_DIM_IN_BLOCKS,
            stripe_xsize.div_ceil(K_BLOCK_DIM),
            stripe_ysize.div_ceil(K_BLOCK_DIM),
        );

        // `write_ac_group` reads whole blocks straight from `opsin` and gathers
        // only footprints that cross the image edge.
        write_ac_group(
            ctx,
            &mut scratch.ac_group,
            opsin,
            (stripe_x0, stripe_y0),
            stripe_brect,
            distp.scale,
            distp.scale_dc,
            distp.dc_step,
            distp.distance,
            distp.x_qm_scale,
            dc_data,
            ytob_dc,
            &mut local_quant_dc,
            source_dc.as_mut(),
            qorigin_x,
            qorigin_y,
            &mut num_nzeros,
            coeff_shifts,
            rdoq_prices,
            coeff_orders,
            alternative_orders,
            order_stats.as_mut(),
            false,
            qf_threshold,
            recon
                .as_mut()
                .map(|(plane, starts)| (plane, ty * K_TILE_DIM, starts)),
            memo.as_deref_mut(),
            &mut tokens,
        );
    }
    if let (Some(m), Some((plane, _))) = (memo, recon.as_mut()) {
        *m.recon_mut() = std::mem::replace(plane, crate::image::Image3F::new(0, 0));
    }

    (
        PendingAcGroup {
            group_idx: image_gy * dim.xsize_groups + image_gx,
            tokens: tokens.into_iter().map(AcTokens::from).collect(),
        },
        local_quant_dc,
        source_dc,
        order_stats,
        recon,
    )
}

/// Carve a stripe out of the (already-XYB-converted, gaborized) opsin image,
/// padding to whole blocks by edge-replication.
#[cfg(test)]
fn build_stripe(
    stripe: &mut Image3F,
    opsin: &Image3F,
    x0: usize,
    y0: usize,
    xsize: usize,
    ysize: usize,
    xsize_padded: usize,
    ysize_padded: usize,
) {
    stripe.resize(xsize_padded, ysize_padded);
    for c in 0..3 {
        // Copy actual content.
        for y in 0..ysize {
            let src_row = opsin.plane_row(c, y0 + y);
            let dst_row = stripe.plane_row_mut(c, y);
            let (data, padding) = dst_row.split_at_mut(xsize);
            data.copy_from_slice(&src_row[x0..x0 + xsize]);
            let last = *data.last().unwrap();
            padding[..xsize_padded - xsize].fill(last);
        }
        // Replicate bottom row.
        for y in ysize..ysize_padded {
            let (src, dst) = stripe.plane_mut(c).two_rows_mut_safe(ysize - 1, y);
            dst.copy_from_slice(src);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn center_first_order_is_rings_around_the_center() {
        // 2048x1365: 8x6 AC groups in one DC group, two passes.
        let dim = super::ImageDim::new(2048, 1365);
        let order = super::group_section_order(&dim, 2, crate::GroupOrder::CenterFirst);
        let n = dim.num_groups;
        assert_eq!(order.len(), 3 + 2 * n);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &v)| i == v));
        assert_eq!(&order[..3], &[0, 1, 2]);
        // The four groups around the center lead each pass, starting top right.
        let first: Vec<usize> = order[3..7].iter().map(|&s| s - 3).collect();
        assert_eq!(first, [20, 28, 27, 19]);
        let pass1: Vec<usize> = order[3 + n..].iter().map(|&s| s - 3 - n).collect();
        let pass0: Vec<usize> = order[3..3 + n].iter().map(|&s| s - 3).collect();
        assert_eq!(pass0, pass1);
    }

    #[test]
    fn diagonal_order_draws_an_x_first() {
        // 2048x1365: 8x6 AC groups. The first twelve form the X: the center
        // four, then both diagonals outward.
        let dim = super::ImageDim::new(2048, 1365);
        let order = super::group_section_order(&dim, 1, crate::GroupOrder::Diagonals);
        let groups: Vec<usize> = order[3..].iter().map(|&s| s - 3).collect();
        let mut x: Vec<usize> = groups[..12].to_vec();
        x.sort_unstable();
        assert_eq!(x, [9, 10, 13, 14, 19, 20, 26, 27, 28, 29, 33, 38]);
        let mut sorted = groups.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &v)| i == v));
    }

    #[test]
    fn coefficient_order_race_prices_serialized_permutations_and_ac_sections() {
        use super::{PendingAcGroup, encode_order_arm};
        use crate::bit_writer::BitWriter;
        use crate::coeff_order::{CoeffOrders, write_coeff_orders};
        use crate::encoding_context::EncodingContext;
        use std::borrow::Cow;

        let ctx = EncodingContext::default();
        let mut scratch = crate::coder_scratch::CoderScratch::default();
        let natural = CoeffOrders::natural();
        let mut custom = CoeffOrders::natural();
        let mut scan = custom.orders[0][0].to_vec();
        scan.swap(1, 2);
        custom.orders[0][0] = Cow::Owned(scan);
        custom.used_mask = 1;
        let plan = crate::ac_context::AcCtxPlan::dct8_only();
        for groups in [1, 2] {
            let pending: Vec<_> = (0..groups)
                .map(|group_idx| PendingAcGroup {
                    group_idx,
                    tokens: vec![
                        (0..4096)
                            .map(|i| crate::entropy::Token::new(0, (i % 13) as u32))
                            .collect::<Vec<_>>()
                            .into(),
                    ],
                })
                .collect();
            let code = crate::entropy::optimize_entropy_code_ac_streams(
                pending.iter().map(|p| p.tokens[0].as_slice()),
                plan.num_ac_contexts(),
                &mut scratch.huffman_pool,
                true,
                ctx.speed,
                Some(&ctx.thread_pool),
            );
            let codes = [code];
            let natural_arm = encode_order_arm(
                &ctx,
                &mut scratch,
                &pending,
                &plan,
                &codes,
                &natural,
                &[false; 13],
                groups,
                1,
            );
            let custom_arm = encode_order_arm(
                &ctx,
                &mut scratch,
                &pending,
                &plan,
                &codes,
                &custom,
                &[false; 13],
                groups,
                1,
            );
            // Identical coefficient streams cannot pay for a permutation.
            assert!(natural_arm.0 < custom_arm.0);
            assert_eq!(natural_arm.2.len(), groups);
            for (i, (section, _)) in natural_arm.2.iter().enumerate() {
                assert_eq!(*section, 3 + i);
            }
            if groups == 1 {
                let mut a = BitWriter::new();
                let mut b = BitWriter::new();
                write_coeff_orders(&natural, &mut scratch.huffman_pool, &mut a);
                write_coeff_orders(&custom, &mut scratch.huffman_pool, &mut b);
                assert_eq!(
                    custom_arm.0 - natural_arm.0,
                    b.bits_written() - a.bits_written()
                );
            }
        }
    }

    #[test]
    fn plain_modular_lattice_matches_scalar_rounding() {
        use crate::coding::CodingTransform;
        use crate::encode_image::AlphaPlane;
        use crate::encoding_context::EncodingContext;
        use crate::image::Image3F;

        for width in [1, 3, 4, 7, 8, 9, 17] {
            let mut coded = Image3F::new(width, 3);
            for c in 0..3 {
                for (i, value) in coded.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                    *value = (i as f32 - 8.5 + c as f32 * 0.25) / super::LM_PLAIN_MAX;
                }
            }
            for coding in [CodingTransform::Rgb, CodingTransform::YCbCr] {
                let mut ctx = EncodingContext::default();
                ctx.coding = coding;
                ctx.extra_squeeze_distance = 1.0;
                for alpha in [
                    AlphaPlane::U8((0..width * 3).map(|i| (i * 37) as u8).collect()),
                    AlphaPlane::U16 {
                        data: (0..width * 3).map(|i| (i * 977) as u16).collect(),
                        bits: 16,
                    },
                ] {
                    for gray in [false, true] {
                        let got = super::lm_plain_input(&ctx, &coded, gray, Some(&alpha)).unwrap();
                        let channels: &[usize] = if gray { &[1] } else { &[0, 1, 2] };
                        for (levels, &c) in got.color.iter().zip(channels) {
                            let offset = if gray && coding == CodingTransform::YCbCr {
                                128.0 / 255.0
                            } else {
                                0.0
                            };
                            let want: Vec<i32> = coded.plane_data(c)[..width * 3]
                                .iter()
                                .map(|&value| {
                                    ((value + offset) * super::LM_PLAIN_MAX).round() as i32
                                })
                                .collect();
                            assert_eq!(*levels, want);
                        }
                        let scale = super::LM_PLAIN_MAX / ((1u32 << alpha.bits()) - 1) as f32;
                        let want: Vec<i32> = (0..width * 3)
                            .map(|i| (alpha.get_i32(i) as f32 * scale).round() as i32)
                            .collect();
                        assert_eq!(got.extra[0].0, want);
                    }
                }
            }
        }
    }

    #[test]
    fn ac_group_source_views_match_copied_stripes() {
        use crate::coeff_order::CoeffOrders;
        use crate::dc_group_data::{DcGroupData, NUM_STRATEGIES};
        use crate::encoding_context::EncodingContext;
        use crate::entropy::{FrozenTokenPrices, Token, build_entropy_code_no_cluster};
        use crate::group::{AcGroupScratch, SourceDc};
        use crate::image::{Image3B, Image3F, Image3S, Plane, Rect};

        let distance = 2.0;
        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            distance,
            1,
        );
        let distp = super::compute_distance_params(distance);
        let mut image = Image3F::new(131, 137);
        for c in 0..3 {
            for (i, value) in image.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *value = ((i * 7919 + c * 97) % 104729) as f32 / 104729.0;
            }
        }
        let tokens: Vec<_> = (0..4096)
            .map(|i| Token::new(i % 2, if i % 3 == 0 { 0 } else { i % 17 }))
            .collect();
        let mut code = build_entropy_code_no_cluster(&tokens, 2, &mut Vec::new());
        code.context_map = (0..crate::ac_context::K_NUM_FINE_AC_CONTEXTS)
            .map(|i| (i % 2) as u8)
            .collect();
        let prices = FrozenTokenPrices::new(&code);
        let orders = CoeffOrders::natural();
        let mut dc = DcGroupData::new(16, 16).unwrap();
        for (i, q) in dc.raw_quant_field.as_mut_slice().iter_mut().enumerate() {
            *q = 3 + (i % 17) as u8;
        }
        dc.ytox_map.as_mut_slice().fill(7);
        dc.ytob_map.as_mut_slice().fill(-13);
        let mut stripe = Image3F::new(64, 64);
        let mut gathered_scratch = AcGroupScratch::default();
        let mut view_scratch = AcGroupScratch::default();
        // The 128 family does not fit this 16x16-block image.
        for strategy in
            (0..NUM_STRATEGIES as u8).filter(|&s| !crate::dc_group_data::is_128_strategy(s))
        {
            dc.ac_strategy.reset();
            dc.ac_strategy.set_first(8, 8, strategy);
            // (70, 76) crosses the image edge: the view gathers the same
            // replicated pixels a padded stripe holds.
            for origin in [(0, 0), (59, 61), (70, 76)] {
                let visible_x = (image.xsize() - origin.0).min(64);
                let visible_y = (image.ysize() - origin.1).min(64);
                let at_edge = visible_x < 64 || visible_y < 64;
                super::build_stripe(
                    &mut stripe,
                    &image,
                    origin.0,
                    origin.1,
                    visible_x,
                    visible_y,
                    64,
                    64,
                );
                for (shifts, rdoq) in [
                    (&[0u32][..], None),
                    (&[0u32][..], Some(&prices)),
                    (&[2u32, 0][..], None),
                ] {
                    // RDOQ discounts padding error only where it can see the
                    // image edge, so the padded stripe legitimately differs.
                    if at_edge && rdoq.is_some() {
                        continue;
                    }
                    let run = |input: &Image3F, origin, scratch: &mut AcGroupScratch| {
                        let mut quant_dc = Image3S::new(8, 8);
                        let mut source_dc = SourceDc {
                            y: Plane::new(8, 8),
                            b: Plane::new(8, 8),
                            x: Plane::new(8, 8),
                        };
                        let mut nzeros: Vec<_> =
                            shifts.iter().map(|_| Image3B::new(32, 32)).collect();
                        let mut tokens: Vec<_> = shifts.iter().map(|_| Vec::new()).collect();
                        let distortion = super::write_ac_group(
                            &ctx,
                            scratch,
                            input,
                            origin,
                            Rect::new(8, 8, 8, 8),
                            distp.scale,
                            distp.scale_dc,
                            distp.dc_step,
                            distance,
                            distp.x_qm_scale,
                            &dc,
                            11,
                            &mut quant_dc,
                            Some(&mut source_dc),
                            8,
                            8,
                            &mut nzeros,
                            shifts,
                            rdoq,
                            &orders,
                            &[],
                            None,
                            true,
                            10,
                            None,
                            None,
                            &mut tokens,
                        );
                        (distortion, quant_dc, source_dc, nzeros, tokens)
                    };
                    let (expected_dist, expected_dc, expected_source, expected_nz, expected) =
                        run(&stripe, (0, 0), &mut gathered_scratch);
                    let (actual_dist, actual_dc, actual_source, actual_nz, actual) =
                        run(&image, origin, &mut view_scratch);
                    assert_eq!(actual_dist.to_bits(), expected_dist.to_bits());
                    for c in 0..3 {
                        assert_eq!(actual_dc.plane_data(c), expected_dc.plane_data(c));
                        for (a, e) in actual_nz.iter().zip(&expected_nz) {
                            assert_eq!(a.plane_data(c), e.plane_data(c));
                        }
                    }
                    for (a, e) in [
                        (&actual_source.y, &expected_source.y),
                        (&actual_source.b, &expected_source.b),
                    ] {
                        assert!(
                            a.as_slice()
                                .iter()
                                .zip(e.as_slice())
                                .all(|(a, e)| a.to_bits() == e.to_bits())
                        );
                    }
                    for (a, e) in actual.iter().zip(&expected) {
                        assert_eq!(a.len(), e.len());
                        assert!(
                            a.iter()
                                .zip(e)
                                .all(|(a, e)| a.context == e.context && a.value == e.value)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn residual_rate_sample_preserves_pixels_and_alpha_at_odd_edges() {
        use crate::encode_image::AlphaPlane;
        use crate::image::Image3F;
        for (w, h) in [(1, 1), (7, 19), (257, 385), (1025, 769)] {
            let mut source = Image3F::new(w, h);
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        source.plane_row_mut(c, y)[x] = (y * w + x) as f32 + c as f32 * 0.25;
                    }
                }
            }
            let alpha = AlphaPlane::U16 {
                data: (0..w * h).map(|i| (i % 4096) as u16).collect(),
                bits: 12,
            };
            let (sample, sampled_alpha) = super::sample_vardct_residual(&source, Some(&alpha));
            let Some(AlphaPlane::U16 { data, bits }) = sampled_alpha else {
                panic!("sample changed the alpha representation");
            };
            assert_eq!(bits, 12);
            assert!(sample.xsize() * sample.ysize() <= 16 * 128 * 128);
            assert_eq!(data.len(), sample.xsize() * sample.ysize());
            for y in 0..sample.ysize() {
                for x in 0..sample.xsize() {
                    let index = sample.plane_row(0, y)[x] as usize;
                    assert!(index < w * h);
                    assert_eq!(data[y * sample.xsize() + x], (index % 4096) as u16);
                    for c in 0..3 {
                        assert_eq!(
                            sample.plane_row(c, y)[x],
                            source.plane_row(c, index / w)[index % w]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sampled_tile_cost_keeps_texture_and_rejects_smooth_repetition() {
        use super::{BitWriter, GlyphStage, prepare_vardct_variant};
        use crate::coder_scratch::CoderScratch;
        use crate::encoding_context::EncodingContext;
        use crate::image::Image3F;

        for textured in [false, true] {
            let distance = 6.0;
            let ctx = EncodingContext::new(
                crate::Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                crate::xyb::XybMatrix::SPEC,
                distance,
                1,
            );
            let mut scratch = CoderScratch::default();
            let distp = super::compute_distance_params(distance);
            let mut source = Image3F::new(512, 256);
            let mut rng = 12345u32;
            let mut tile = [0.0; 256];
            for v in &mut tile {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = 0.25 + (rng >> 24) as f32 / 512.0;
            }
            for y in 0..source.ysize() {
                for x in 0..source.xsize() {
                    let value = if !textured {
                        0.5 + if (x % 16 < 8) ^ (y % 16 < 8) {
                            0.015
                        } else {
                            -0.015
                        }
                    } else if x < 320 {
                        tile[(y % 16) * 16 + x % 16]
                    } else {
                        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                        0.25 + (rng >> 24) as f32 / 512.0
                    };
                    source.plane_row_mut(1, y)[x] = value;
                    source.plane_row_mut(2, y)[x] = value;
                }
            }
            let mut estimates = Vec::new();
            let mut actual = Vec::new();
            let glyphs = GlyphStage::find(&ctx, &mut scratch, distance, &source, None);
            for tiles in [false, true] {
                let mut plan = prepare_vardct_variant(
                    &ctx,
                    &mut scratch,
                    distance,
                    &distp,
                    source.clone(),
                    None,
                    &[0],
                    true,
                    glyphs.as_ref(),
                    tiles,
                    #[cfg(feature = "splines")]
                    None,
                )
                .unwrap();
                assert_eq!(plan.used_tiles, tiles);
                let estimate = plan
                    .estimate(&ctx, &mut scratch, distance, None, &[0])
                    .unwrap();
                assert!(estimate.is_finite() && estimate > 0.0);
                assert_eq!(
                    estimate,
                    plan.estimate(&ctx, &mut scratch, distance, None, &[0])
                        .unwrap()
                );
                estimates.push(estimate);
                let mut writer = BitWriter::new();
                plan.encode(&ctx, &mut scratch, distance, None, &[0], &mut writer)
                    .unwrap();
                actual.push(writer.bits_written() as f64);
            }
            let estimated_drop = estimates[0] * super::GLYPH_BITS_MARGIN < estimates[1];
            let actual_drop = actual[0] * super::GLYPH_BITS_MARGIN < actual[1];
            assert_eq!(actual_drop, !textured);
            assert_eq!(
                estimated_drop, actual_drop,
                "estimates={estimates:?}, actual={actual:?}"
            );
        }
    }

    #[test]
    fn arithmetic_matrix_header_cost_matches_writer_for_every_subset() {
        use crate::quant_weights::{DequantMatrices, MatrixHeaderCost};
        for distance in [0.5, 1.5, 2.5, 4.0, 8.0] {
            for matrices in [
                DequantMatrices::new(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_pair_b(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_saturated(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_x_heavy(distance, crate::Arbiter::Ssimulacra2),
                DequantMatrices::new_saturated_pair_b(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_fast(distance, crate::Arbiter::Ssimulacra2),
            ] {
                let header = MatrixHeaderCost::new(matrices);
                for mask in (0..512u16).flat_map(|m| [m, m | 1 << 12]) {
                    let used = std::array::from_fn(|slot| mask & (1 << slot) != 0);
                    let mut writer = crate::bit_writer::BitWriter::new();
                    super::write_dequant_matrices(matrices, &used, &mut writer);
                    assert_eq!(
                        header.bits(mask),
                        writer.bits_written(),
                        "d={distance} mask={mask}"
                    );
                }
            }
        }
    }

    use super::{
        CHROMA_DC_GROWTH_DISTANCE, DC_REFINE_HOLD, DC_REFINE_PEAK, DC_REFINE_RELEASE,
        EPF_PASS0_SCALE, EPF_PASS0_SPEC_SCALE, MIN_TOKENS_PER_DC_LEAF, X_DC_STEP,
        choose_dc_predictors, compute_distance_params, dc_refinement, epf_sharpness_id, quant_dc,
    };
    use crate::coder_scratch::DcPredictorScratch;
    use crate::entropy::Token;

    /// Build one group of `n` tokens in `context`, all carrying `value`.
    fn leaf_tokens(context: u32, value: u32, n: usize) -> Vec<Vec<Token>> {
        vec![(0..n).map(|_| Token::new(context, value)).collect()]
    }

    #[test]
    fn chroma_texture_stat_separates_thin_chroma_from_smooth_saturation() {
        fn texture(xyb: &crate::image::Image3F) -> f32 {
            let sums = super::chroma_gradient_sums_scalar(
                xyb.plane_data(0),
                xyb.plane_data(1),
                xyb.plane_data(2),
                xyb.xsize(),
            );
            let sat = super::chroma_saturation_stat(xyb);
            super::chroma_stats_from_sums(sums, xyb.xsize(), xyb.ysize(), sat)[2]
        }
        let mut xyb = crate::image::Image3F::new(32, 32);
        for y in 0..32 {
            let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.0);
            y_row.fill(0.5);
            b_row.fill(0.5);
        }
        assert_eq!(texture(&xyb), 0.0);
        // Smooth, strongly saturated: high chroma, zero texture.
        for y in 0..32 {
            xyb.all_plane_rows_mut(y)[2].fill(0.8);
        }
        assert_eq!(texture(&xyb), 0.0);
        // Thin chromatic lines on neutral ground: the pair-B class. Period 3
        // so the saturation stat's 4x4 subsample does not alias to zero.
        for y in 0..32 {
            let [_, _, b_row] = xyb.all_plane_rows_mut(y);
            for (x, b) in b_row.iter_mut().enumerate() {
                *b = if x % 3 == 0 { 0.6 } else { 0.5 };
            }
        }
        assert!(texture(&xyb) > super::PAIR_B_TEXTURE_THRESHOLD);
    }

    #[test]
    fn chroma_saturation_stat_measures_xyb_deviation() {
        let mut xyb = crate::image::Image3F::new(16, 16);
        for y in 0..16 {
            let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.0);
            y_row.fill(0.5);
            b_row.fill(0.5);
        }
        // Achromatic frames (X = 0, B = Y) read exactly zero.
        assert_eq!(super::chroma_saturation_stat(&xyb), 0.0);

        for y in 0..16 {
            let [x_row, _, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.02);
            b_row.fill(0.55);
        }
        // |X| + |B - Y| = 0.02 + 0.05 on every sampled pixel.
        let stat = super::chroma_saturation_stat(&xyb);
        assert!((stat - 0.07).abs() < 1e-5, "{stat}");
        assert!(stat >= super::SAT_QM_THRESHOLD);
    }

    #[test]
    fn x_gradient_stat_separates_red_structure_from_achromatic() {
        let ctx = crate::encoding_context::EncodingContext::default();
        // Achromatic: X flat at 0 → stat 0, far below the gate.
        let mut xyb = crate::image::Image3F::new(32, 32);
        for y in 0..32 {
            let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
            x_row.fill(0.0);
            for (x, v) in y_row.iter_mut().enumerate() {
                *v = (x % 7) as f32 * 0.1;
            }
            b_row.fill(0.5);
        }
        assert_eq!(super::chroma_gradient_stats(&ctx, &xyb, 0.0)[0], 0.0);

        // X carrying half of Y's gradient amplitude → stat 0.5, over the gate.
        for y in 0..32 {
            let [x_row, y_row, _] = xyb.all_plane_rows_mut(y);
            for (x, v) in x_row.iter_mut().enumerate() {
                *v = y_row[x] * 0.5;
            }
        }
        let stat = super::chroma_gradient_stats(&ctx, &xyb, 0.0)[0];
        assert!((stat - 0.5).abs() < 1e-5, "{stat}");
        assert!(stat >= super::X_QM_GRAD_THRESHOLD);
    }

    #[test]
    fn point_chroma_policy_preserves_precision_across_hq_boundaries() {
        use crate::{
            Speed, encoding_context::EncodingContext, quant_weights::DequantMatrices, xyb,
        };
        let content = super::ChromaContentStats {
            saturation: 0.03,
            dark_fraction: 0.75,
            point_fraction: 0.01,
            point_energy: 0.4,
            sharp_energy: 0.15,
        };
        for distance in [0.03, 0.15, 0.25, 0.295, 0.2989, 0.299, 0.2999, 0.3, 0.3001] {
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                xyb::XybMatrix::SPEC,
                distance,
                1,
            );
            let distp = compute_distance_params(distance);
            super::apply_point_chroma_policy(&ctx, &distp, content, false);
            assert_eq!(
                distp.x_qm_scale.max(ctx.x_qm_scale_floor()),
                3,
                "d={distance}"
            );
            assert_eq!(ctx.b_qm_scale(), 3, "d={distance}");
            assert!(std::ptr::eq(
                ctx.matrices(),
                DequantMatrices::new(0.3, crate::Arbiter::Ssimulacra2, false)
            ));
        }
        // Existing mid-band protection and the upper cutoff stay intact.
        for (distance, x_scale, b_scale) in [(1.25, 4, 3), (2.249, 4, 3), (2.25, 3, 2)] {
            let ctx = EncodingContext::new(
                Speed::Slow,
                crate::Arbiter::Ssimulacra2,
                xyb::XybMatrix::SPEC,
                distance,
                1,
            );
            let distp = compute_distance_params(distance);
            let matrices = ctx.matrices();
            super::apply_point_chroma_policy(&ctx, &distp, content, false);
            assert_eq!(distp.x_qm_scale.max(ctx.x_qm_scale_floor()), x_scale);
            assert_eq!(ctx.b_qm_scale(), b_scale);
            assert!(std::ptr::eq(ctx.matrices(), matrices));
        }
        // Excluded content retains both its tables and scales.
        for (speed, achromatic, x_heavy, detected) in [
            (Speed::Fast, false, false, true),
            (Speed::Slow, true, false, true),
            (Speed::Slow, false, true, true),
            (Speed::Slow, false, false, false),
        ] {
            let ctx = EncodingContext::new(
                speed,
                crate::Arbiter::Ssimulacra2,
                xyb::XybMatrix::SPEC,
                0.295,
                1,
            );
            ctx.set_x_heavy(x_heavy);
            let matrices = ctx.matrices();
            let stats = if detected {
                content
            } else {
                Default::default()
            };
            super::apply_point_chroma_policy(
                &ctx,
                &compute_distance_params(0.295),
                stats,
                achromatic,
            );
            assert_eq!(ctx.x_qm_scale_floor(), 2);
            assert_eq!(ctx.b_qm_scale(), 2);
            assert!(std::ptr::eq(ctx.matrices(), matrices));
            assert!(ctx.point_chroma_hq_matrices().is_none());
        }
    }

    #[test]
    fn point_chroma_stats_separate_impulses_from_lines_and_hue_edges() {
        let mut stars = crate::image::Image3F::new(256, 256);
        for py in 0..256 {
            for px in 0..256 {
                let y = if px % 32 == 12 && py % 32 == 12 {
                    0.8
                } else {
                    0.12
                };
                stars.plane_row_mut(0, py)[px] = 0.001;
                stars.plane_row_mut(1, py)[px] = y;
                stars.plane_row_mut(2, py)[px] = y + 0.03;
            }
        }
        let stats = super::chroma_content_stats(&stars);
        assert!(stats.diffuse_point_chroma(), "{stats:?}");
        assert!(stats.point_energy > 0.7);

        // The same contrast and chroma in continuous lines must not qualify.
        for py in 0..256 {
            for px in 0..256 {
                let y = if px % 32 == 12 { 0.8 } else { 0.12 };
                stars.plane_row_mut(1, py)[px] = y;
                stars.plane_row_mut(2, py)[px] = y + 0.03;
            }
        }
        assert!(!super::chroma_content_stats(&stars).diffuse_point_chroma());
        for py in 0..256 {
            stars.plane_row_mut(1, py).fill(0.12);
            stars.plane_row_mut(2, py).fill(0.15);
            for px in 0..256 {
                stars.plane_row_mut(0, py)[px] = if (px + py) % 2 == 0 { 0.01 } else { -0.01 };
            }
        }
        assert!(!super::chroma_content_stats(&stars).diffuse_point_chroma());
    }

    #[test]
    fn point_chroma_stats_handle_small_and_achromatic_images() {
        for (w, h) in [(1, 1), (3, 17), (17, 3), (9, 9), (16, 16)] {
            let mut image = crate::image::Image3F::new(w, h);
            for py in 0..h {
                for px in 0..w {
                    let y = if (px + py) % 5 == 0 { 0.7 } else { 0.1 };
                    image.plane_row_mut(1, py)[px] = y;
                    image.plane_row_mut(2, py)[px] = y;
                }
            }
            let stats = super::chroma_content_stats(&image);
            assert_eq!(stats.saturation, 0.0);
            assert!(stats.point_fraction.is_finite() && stats.point_energy.is_finite());
            assert!(!stats.diffuse_point_chroma());
        }
    }

    #[test]
    fn selected_chroma_gradient_sums_match_scalar() {
        let selected = super::selected_chroma_gradient_sums_fn();
        for width in [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for height in [0, 1, 4, 5, 9] {
                let len = width * height;
                let x: Vec<f32> = (0..len)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03125)
                    .collect();
                let y: Vec<f32> = (0..len)
                    .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.0234375)
                    .collect();
                let b: Vec<f32> = (0..len)
                    .map(|i| ((i * 71 % 89) as f32 - 44.0) * 0.01953125)
                    .collect();
                let expected = super::chroma_gradient_sums_scalar(&x, &y, &b, width);
                let actual = selected(&x, &y, &b, width);
                for channel in 0..4 {
                    let tolerance = 1e-5 * expected[channel].abs().max(1.0);
                    assert!(
                        (actual[channel] - expected[channel]).abs() <= tolerance,
                        "channel {channel}, {width}x{height}: SIMD={}, scalar={}, tolerance={tolerance}",
                        actual[channel],
                        expected[channel]
                    );
                }
            }
        }
    }

    #[test]
    fn dc_plane_price_pools_histograms_and_restarts_prediction() {
        let mut price = super::DcPlanePrice::default();
        price.add(&[0i16], 1, 1, 1);
        price.add(&[1i16], 1, 1, 1);
        // Two equally frequent symbols in one shared context cost two bits;
        // pricing the one-symbol groups separately would incorrectly cost zero.
        assert_eq!(price.bits(), 2.0);

        price.clear();
        assert_eq!(price.bits(), 0.0);
        price.add(&[100i16], 1, 1, 1);
        price.add(&[100i16], 1, 1, 1);
        // Each group's first sample predicts zero, so both tokens are identical.
        // Treating them as one row would instead code residuals 100 and zero.
        assert_eq!(price.bits(), 2.0 * super::price_dc_plane(&[100], 1, 1, 1));
        assert!(super::price_dc_plane(&[100, 100], 2, 1, 2) < price.bits());
    }

    #[test]
    fn dc_predictor_flips_only_on_a_decisive_populated_win() {
        let n = MIN_TOKENS_PER_DC_LEAF as usize;
        let ctx = 11;
        let mut scratch = DcPredictorScratch::default();

        // Gradient codes every residual as zero; the weighted predictor misses
        // by a wide, multi-extra-bit margin.
        let decisive = choose_dc_predictors(
            &leaf_tokens(ctx, 4000, n),
            &leaf_tokens(ctx, 0, n),
            &mut scratch,
        );
        assert!(decisive[ctx as usize], "a decisive gradient win must flip");

        // Identical streams: the margin must keep the weighted default.
        let tied = choose_dc_predictors(
            &leaf_tokens(ctx, 7, n),
            &leaf_tokens(ctx, 7, n),
            &mut scratch,
        );
        assert!(!tied[ctx as usize], "a tie must not flip");

        // Same decisive win, but below the population floor.
        let sparse = choose_dc_predictors(
            &leaf_tokens(ctx, 4000, n - 1),
            &leaf_tokens(ctx, 0, n - 1),
            &mut scratch,
        );
        assert!(
            !sparse[ctx as usize],
            "an underpopulated leaf must not flip"
        );
    }

    #[test]
    fn chroma_dc_steps_grow_slower_than_luma_past_the_knee() {
        for d in [0.1, 1.0, 2.0, 3.0, CHROMA_DC_GROWTH_DISTANCE] {
            assert_eq!(
                compute_distance_params(d).dc_step,
                [X_DC_STEP, 1.0, 1.0],
                "d={d}"
            );
        }
        // Absolute steps per channel: luma coarsens fastest, B slowest, and
        // no chroma step ever gets finer as the distance grows.
        let absolute = |d: f32| {
            let p = compute_distance_params(d);
            std::array::from_fn::<f32, 3, _>(|c| p.dc_step[c] / p.scale_dc)
        };
        let mut prev = absolute(CHROMA_DC_GROWTH_DISTANCE);
        for d in [4.0, 5.0, 6.0, 8.0, 11.0, 15.0, 25.0] {
            let p = compute_distance_params(d);
            assert!(p.dc_step[0] < X_DC_STEP && p.dc_step[2] < 1.0, "d={d}");
            assert!(p.dc_step[0] / X_DC_STEP > p.dc_step[2], "d={d}");
            let cur = absolute(d);
            for c in [0, 2] {
                assert!(cur[c] >= prev[c] * 0.999, "chroma {c} step shrank at d={d}");
            }
            prev = cur;
        }
    }

    #[test]
    fn dc_refinement_holds_then_ramps_out_monotonically() {
        // Full refinement through the hold point, none past the release point.
        for d in [0.1, 0.5, 1.0, 2.0, DC_REFINE_HOLD] {
            assert_eq!(dc_refinement(d), DC_REFINE_PEAK);
        }
        for d in [DC_REFINE_RELEASE, 5.5, 8.0, 25.0] {
            assert_eq!(dc_refinement(d), 1.0);
        }
        // Strictly decreasing across the ramp, and continuous at both ends.
        let mid = 0.5 * (DC_REFINE_HOLD + DC_REFINE_RELEASE);
        assert!((dc_refinement(mid) - 0.5 * (DC_REFINE_PEAK + 1.0)).abs() < 1e-6);
        let mut prev = f32::INFINITY;
        for i in 0..=40 {
            let d = DC_REFINE_HOLD + (DC_REFINE_RELEASE - DC_REFINE_HOLD) * (i as f32 / 40.0);
            let v = dc_refinement(d);
            assert!(v <= prev, "refinement rose at d={d}");
            assert!((1.0..=DC_REFINE_PEAK).contains(&v));
            prev = v;
        }
    }

    #[test]
    fn dc_refinement_makes_dc_finer_where_it_applies() {
        // Refining means a *larger* DC scale (finer quantization step), and it
        // must not disturb the AC quantizer, which is signaled independently.
        for d in [0.5, 1.0, 3.0] {
            let p = compute_distance_params(d);
            assert!(p.scale_dc > 0.0);
            assert!((p.scale_dc - quant_dc(d)).abs() <= 0.5 * p.scale + f32::EPSILON);
        }
        assert!(quant_dc(1.0) > quant_dc(1.0) / DC_REFINE_PEAK);
    }

    #[test]
    fn epf_third_pass_schedule() {
        // The third EPF pass at spec strength erases texture in the normal
        // bands, so it only enters weakened (custom pass-0 sigma scale) from
        // EPF_PASS0_START_D, eases toward spec strength across the first half
        // of the VLQ ramp, and hands over to the fitted spec-strength band
        // (default sigma header) at half-ramp. Guards every edge.
        for d in [0.0, 0.1, 0.5, 0.69] {
            assert_eq!(compute_distance_params(d).epf_iters, 0, "d={d}");
        }
        for d in [0.7, 1.0, 1.49] {
            assert_eq!(compute_distance_params(d).epf_iters, 1, "d={d}");
        }
        for d in [1.5, 2.0, 2.49] {
            let p = compute_distance_params(d);
            assert_eq!(p.epf_iters, 2, "d={d}");
            assert_eq!(p.epf_pass0_scale, None, "d={d}");
        }
        for d in [2.5, 4.0, 6.0, 8.0] {
            let p = compute_distance_params(d);
            assert_eq!(p.epf_iters, 3, "d={d}");
            assert_eq!(p.epf_pass0_scale, Some(EPF_PASS0_SCALE), "d={d}");
        }
        let mid = compute_distance_params(9.75).epf_pass0_scale.unwrap();
        assert!((mid - 0.5 * (EPF_PASS0_SCALE + EPF_PASS0_SPEC_SCALE)).abs() < 1e-5);
        for d in [11.5, 15.0, 25.0] {
            let p = compute_distance_params(d);
            assert_eq!(p.epf_iters, 3, "d={d}");
            assert_eq!(p.epf_pass0_scale, None, "d={d}");
        }
    }

    #[test]
    fn epf_sharpness_strengthens_in_the_coarse_band() {
        assert_eq!(epf_sharpness_id(1.39, false), 7);
        assert_eq!(epf_sharpness_id(1.4, false), 6);
        assert_eq!(epf_sharpness_id(1.74, false), 6);
        assert_eq!(epf_sharpness_id(1.75, false), 5);
        assert_eq!(epf_sharpness_id(2.75, false), 5);
        assert_eq!(epf_sharpness_id(3.49, false), 5);
        assert_eq!(epf_sharpness_id(3.5, false), 4);
        // VLQ ramp steps (quarter-points of DCEPF_VLQ_D0..D1 = 8..15).
        assert_eq!(epf_sharpness_id(9.74, false), 4);
        assert_eq!(epf_sharpness_id(9.75, false), 3);
        assert_eq!(epf_sharpness_id(11.5, false), 2);
        assert_eq!(epf_sharpness_id(13.25, false), 1);
        assert_eq!(epf_sharpness_id(25.0, false), 1);
    }

    #[test]
    fn screen_content_keeps_the_strongest_epf_below_d5() {
        use super::epf_sharpness_id;
        assert_eq!(epf_sharpness_id(1.0, true), 7);
        assert_eq!(epf_sharpness_id(2.5, true), 7);
        assert_eq!(epf_sharpness_id(4.99, true), 7);
        assert_eq!(epf_sharpness_id(5.0, true), epf_sharpness_id(5.0, false));
        assert_eq!(epf_sharpness_id(12.0, true), epf_sharpness_id(12.0, false));
    }

    #[test]
    fn constant_block_fraction_counts_exactly_flat_blocks() {
        use super::constant_block_fraction;
        use crate::image::Image3F;
        let mut xyb = Image3F::new(32, 16);
        // Left half flat, right half a ramp: 4 of 8 blocks are constant.
        for c in 0..3 {
            for y in 0..16 {
                for x in 16..32 {
                    xyb.plane_row_mut(c, y)[x] = x as f32 * 0.01;
                }
            }
        }
        assert_eq!(constant_block_fraction(&xyb), 0.5);
        assert_eq!(constant_block_fraction(&Image3F::new(7, 7)), 0.0);
    }

    #[test]
    fn quantization_keeps_coarsening_at_low_quality() {
        let d4 = compute_distance_params(4.0);
        let d45 = compute_distance_params(4.5);
        let d5 = compute_distance_params(5.0);
        assert!(d4.global_scale > d45.global_scale);
        assert!(d45.global_scale > d5.global_scale);
        for distances in [3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0, 16.0].array_windows::<2>() {
            let fine = compute_distance_params(distances[0]);
            let coarse = compute_distance_params(distances[1]);
            assert!(fine.scale_dc > coarse.scale_dc, "distances={distances:?}");
        }
    }

    #[test]
    fn dc_quantization_stays_independent_of_ac_scale() {
        for distance in [0.5, 1.0, 3.5, 4.0, 4.5, 6.0] {
            let params = compute_distance_params(distance);
            let rounding_error = 0.5 * params.scale;
            assert!((params.scale_dc - quant_dc(distance)).abs() <= rounding_error + f32::EPSILON);
        }
    }
}

#[cfg(test)]
mod fused_dc_tests {
    use super::*;
    #[test]
    fn fused_dc_candidates_match_separate_predictor_traversals() {
        for w in [1, 2, 3, 9, 32] {
            for h in [1, 2, 3, 17] {
                let mut dc = DcGroupData::new(w, h).unwrap();
                for c in 0..3 {
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x + 1) * (c + 7);
                            dc.quant_dc.plane_row_mut(c, y)[x] =
                                (((i * 7919) ^ (i / 7 * 1237)) % 65536) as i16;
                        }
                    }
                }
                for collect_props in [false, true] {
                    for choice in [
                        DC_PREDICTOR_WEIGHTED,
                        DC_PREDICTOR_GRADIENT,
                        std::array::from_fn(|i| i % 2 == 0),
                    ] {
                        let mut expected_props = Vec::new();
                        let expected =
                            collect_dc_tokens(&dc, &choice, &mut expected_props, collect_props);
                        let grad =
                            collect_dc_tokens(&dc, &DC_PREDICTOR_GRADIENT, &mut Vec::new(), false);
                        let mut props = Vec::new();
                        let (fused, fused_grad) = collect_dc_tokens_with_gradient(
                            &dc,
                            &choice,
                            &mut props,
                            collect_props,
                            true,
                        );
                        let values = |ts: Vec<Token>| {
                            ts.into_iter()
                                .map(|t| (t.context, t.value))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(values(fused), values(expected));
                        assert_eq!(values(fused_grad), values(grad));
                        assert_eq!(props, expected_props);
                    }
                }
            }
        }
    }
}
