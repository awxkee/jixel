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

use crate::Speed;

use crate::ac_context::compact_block_context_map;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, DcPredictorScratch};
use crate::color_correlation::choose_ytob_dc;
use crate::dc_group_data::{
    DcGroupData, STRATEGY_DCT, STRATEGY_DCT4X8, STRATEGY_DCT8X4, STRATEGY_DCT8X16,
    STRATEGY_DCT16X8, STRATEGY_DCT16X16, STRATEGY_DCT16X32, STRATEGY_DCT32X16, STRATEGY_DCT32X32,
    is_sub8_strategy,
};
use crate::dct::fmla;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::entropy::{
    EntropyCode, Token, f_log2, optimize_entropy_code, pack_signed, write_entropy_code, write_token,
};
use crate::group::write_ac_group;
use crate::image::{Image3B, Image3F, Image3S, Rect};
use crate::patches::{MODULAR_PATCH_REF_ID, PATCH_REF_ID, VarDctFrameKind, find_lossy_patches};
use crate::static_entropy_codes::{
    K_CONTEXT_TREE_TOKENS, K_GRADIENT_CONTEXT_LUT, K_NUM_DC_CONTEXTS,
};

const K_BLOCK_DIM: usize = 8;
const K_TILE_DIM: usize = 64;
const K_GROUP_DIM: usize = 256;
const K_DC_GROUP_DIM: usize = 2048;
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

struct DistanceParams {
    distance: f32,
    global_scale: i32,
    quant_dc: i32,
    scale: f32,
    scale_dc: f32,
    x_qm_scale: u32,
    epf_iters: u32,
    gab_enabled: bool,
}

const DC_REFINE_PEAK: f32 = 1.35;
const DC_REFINE_HOLD: f32 = 3.0;
const DC_REFINE_RELEASE: f32 = 5.0;

/// Whether the closed-loop DC rounding pass (`enc_dc_smooth`) runs. Shared by
/// the pass itself and the `dc_float` capture-plane allocations so no float
/// DC is stored when the pass is off. Below d≈0.8 the DC steps are fine
/// enough that the smoothing gate almost never opens and the pass only adds
/// noise; it is a quality refinement, not worth Fast's time budget.
#[inline]
fn dc_smooth_enabled(distance: f32, speed: Speed) -> bool {
    distance >= 0.8 && speed == Speed::Slow
}

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

fn quant_dc(distance: f32) -> f32 {
    // Cap the DC distance at 3.5: beyond that the DC plane holds so few bits
    // (WP + ANS + decoder smoothing make fine DC cheap) that further DC
    // coarsening buys almost no rate while banding dominates the perceptual
    // loss on smooth content.
    let refine = dc_refinement(distance);
    let distance = distance.min(3.5);
    let k_dc_quant_pow = 0.57f32;
    let k_dc_quant = 1.12f32;
    let k_dc_mul = 2.9f32;
    let effective = k_dc_mul * (distance / k_dc_mul).powf(k_dc_quant_pow);
    let effective = f32::clamp(effective, 0.5 * distance, distance);
    (k_dc_quant / effective).min(50.0) * refine
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
    if distance > 1.25 {
        x_qm_scale += 1;
    }
    if distance > 9.0 {
        x_qm_scale += 1;
    }
    if distance < 0.299 {
        x_qm_scale += 1;
    }
    static EPF_THRESHOLDS: [f32; 2] = [0.7, 1.5];
    let mut epf_iters: u32 = 0;
    for t in EPF_THRESHOLDS {
        if distance >= t {
            epf_iters += 1;
        }
    }

    DistanceParams {
        distance,
        global_scale,
        quant_dc: qd,
        scale: scale_f,
        scale_dc,
        x_qm_scale,
        epf_iters,
        gab_enabled: false, // measured net-negative for rate-matched SSIMU2
    }
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
    wp: &mut crate::lossless::WpState,
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
) {
    let wp_prediction = wp.predict(x, y, n, w, ne, nw, nn);
    let prop = (K_GRAD_RANGE_MID + wp.wp_prop).clamp(K_GRAD_RANGE_MIN, K_GRAD_RANGE_MAX) as usize;
    props.push(crate::dc_tree::dc_prop(mchan, prop));
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
}

/// Per-leaf predictor selection for the DC plane, indexed by DC context.
pub(crate) type DcPredictorChoice = [bool; K_NUM_DC_CONTEXTS];

pub(crate) const DC_PREDICTOR_WEIGHTED: DcPredictorChoice = [false; K_NUM_DC_CONTEXTS];

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
    counts.fill([0; crate::entropy::ALPHABET_SIZE]);
    extra.fill(0);
    for (cand, groups) in [wp, grad].into_iter().enumerate() {
        for token in groups.iter().flatten() {
            let (sym, nbits, _) = crate::entropy::uint_encode(token.value);
            let slot = cand * K_NUM_DC_CONTEXTS + token.context as usize;
            counts[slot][sym as usize] += 1;
            extra[slot] += nbits as u64;
        }
    }
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
) -> Vec<Token> {
    let token_count = [1usize, 0, 2]
        .into_iter()
        .map(|c| dc_data.quant_dc.plane(c).as_slice().len())
        .sum();
    let mut tokens = Vec::with_capacity(token_count);

    // Weighted-predictor DC, mirroring libjxl's kWPFixedDC path (enc_modular.cc
    // AddVarDCTDC at speed tiers falcon..squirrel): the guess is the
    // self-correcting WP and the context is the WP-error property (modular
    // property 15) bucketed by the same +-500 cutoffs the gradient tree used,
    // so K_GRADIENT_CONTEXT_LUT applies unchanged (write_context_tree emits the
    // matching tree: identical structure, property 15, Weighted leaves). The WP
    // state machine and border conventions are the bit-faithful lossless-path
    // ones, so encoder residuals match the reference decoder exactly.
    for (mchan, c) in [1usize, 0, 2].into_iter().enumerate() {
        let plane = dc_data.quant_dc.plane(c);
        let ysize = plane.ysize();
        let xsize = plane.xsize();
        if xsize == 0 || ysize == 0 {
            continue;
        }

        let mut wp = crate::lossless::WpState::new(xsize);
        let mut rows = plane.as_slice().chunks_exact(xsize);

        // On the top row every unavailable north-side neighbor collapses to
        // the value on the left.
        let first_row = rows.next().unwrap();
        let mut left = 0i64;
        for (x, &value) in first_row.iter().enumerate() {
            let value = value as i64;
            push_dc_wp_token(
                &mut tokens,
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
            );
            left = value;
        }

        let mut row_above = first_row;
        let mut row_above2 = first_row;
        for (y, row) in rows.enumerate() {
            let y = y + 1;

            // First column: W and NW replicate N.
            let n = row_above[0] as i64;
            let ne = row_above.get(1).copied().unwrap_or(row_above[0]) as i64;
            push_dc_wp_token(
                &mut tokens,
                &mut wp,
                0,
                y,
                row[0] as i64,
                n,
                n,
                ne,
                n,
                row_above2[0] as i64,
                dc_gradient,
                mchan,
                props,
            );

            // Interior columns have every neighbor available.
            let current_pairs = row.array_windows::<2>();
            let north_triplets = row_above.array_windows::<3>();
            for (offset, ((current, north), &nn)) in current_pairs
                .zip(north_triplets)
                .zip(row_above2.iter().skip(1))
                .enumerate()
            {
                push_dc_wp_token(
                    &mut tokens,
                    &mut wp,
                    offset + 1,
                    y,
                    current[1] as i64,
                    north[1] as i64,
                    current[0] as i64,
                    north[2] as i64,
                    north[0] as i64,
                    nn as i64,
                    dc_gradient,
                    mchan,
                    props,
                );
            }

            // Last column: NE replicates N. Width one was handled above.
            if xsize > 1 {
                let x = xsize - 1;
                let n = row_above[x] as i64;
                push_dc_wp_token(
                    &mut tokens,
                    &mut wp,
                    x,
                    y,
                    row[x] as i64,
                    n,
                    row[x - 1] as i64,
                    n,
                    row_above[x - 1] as i64,
                    row_above2[x] as i64,
                    dc_gradient,
                    mchan,
                    props,
                );
            }

            row_above2 = row_above;
            row_above = row;
        }
    }
    tokens
}

/// AC metadata: ytox/ytob CfL maps (all 0), AC strategy (all 0 = DCT-8x8),
/// quant field residuals (all 0), and EPF (token (0, PackSigned(4)) per block).
///
/// In libjxl-tiny ALL four sub-streams use the same shared dc_code.
/// Same as write_ac_metadata_tokens, but returns the tokens. Mirror of
/// collect_dc_tokens for the AC metadata (YtoX/B, ACS, QF, EPF).
#[inline]
fn ac_metadata_context(left: i32, base: u32) -> u32 {
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
    distance: f32,
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
            props.push(wbin(left));
            tokens.push(Token::new(ctx_id, pack_signed(here - left)));
            left = here;
        }

        let mut row_above = first_row;
        for row in rows {
            // First column replicates the value above for W and NW.
            props.push(wbin(row_above[0] as i32));
            tokens.push(Token::new(
                ctx_id,
                pack_signed(row[0] as i32 - row_above[0] as i32),
            ));

            for (current, above) in row.array_windows::<2>().zip(row_above.array_windows::<2>()) {
                let prediction =
                    clamped_gradient(above[1] as i32, current[0] as i32, above[0] as i32);
                props.push(wbin(current[0] as i32));
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
    props.resize(strategy_base + 2 * num_first_blocks, 0);
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
            props[strategy_base + first_idx] = wbin(strategy_left);
            tokens[strategy_base + first_idx] =
                Token::new(ac_metadata_context(strategy_left, 7), pack_signed(strategy));
            strategy_left = strategy;

            let qf = qf as i32 - 1;
            props[qf_base + first_idx] = wbin(qf_left);
            tokens[qf_base + first_idx] =
                Token::new(ac_metadata_context(qf_left, 3), pack_signed(qf - qf_left));
            qf_left = qf;
            first_idx += 1;
        }
    }
    debug_assert_eq!(first_idx, num_first_blocks);

    // (d) EPF tokens (constant stream; refinement can never split it).
    let sharp = epf_sharpness_id(distance);
    props.resize(props.len() + nblocks, wbin(sharp));
    tokens.resize(tokens.len() + nblocks, Token::new(0, pack_signed(sharp)));
    tokens
}

/// Distance-scheduled constant per-block EPF sharpness id
pub(crate) const fn b_qm_scale() -> u32 {
    2
}

/// Encoder-side B quantizer-scale multiplier matching [`b_qm_scale`]
/// (mirrors `x_qm_mul = 1.25^(x_qm_scale - 2)`).
pub(crate) fn b_qm_mul() -> f32 {
    1.25f32.powf(b_qm_scale() as f32 - 2.0)
}

fn epf_sharpness_id(distance: f32) -> i32 {
    if distance < 1.75 {
        7
    } else if distance < 2.75 {
        6
    } else if distance < 5.0 {
        5
    } else {
        4
    }
}

/// Real prefix-code bit cost of a DC group's AC-metadata token stream under a
/// freshly optimized entropy code. Used by the sub-8x8 activation gate to weigh
/// the exact selected set's meta-stream cost against its RD benefit.
fn meta_entropy_cost(dc_data: &DcGroupData, scratch: &mut CoderScratch, distance: f32) -> u64 {
    let toks = collect_ac_metadata_tokens(dc_data, &mut Vec::new(), distance);
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
    write_tree_tokens(&tokens, huffman_pool, writer);
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
    epf_iters: u32,
    gab_enabled: bool,
    has_alpha: bool,
    coeff_shifts: &[u32],
    kind: VarDctFrameKind<'_>,
    w: &mut BitWriter,
) {
    match kind {
        VarDctFrameKind::Regular => write_frame_header(
            x_qm_scale,
            epf_iters,
            gab_enabled,
            has_alpha,
            coeff_shifts,
            false,
            w,
        ),
        VarDctFrameKind::Patched(_) => write_frame_header(
            x_qm_scale,
            epf_iters,
            gab_enabled,
            has_alpha,
            coeff_shifts,
            true,
            w,
        ),
        VarDctFrameKind::ReferenceOnly { width, height } => {
            w.write(1, 0); // not all default
            w.write(2, 0b10); // reference-only frame
            w.write(1, 0); // VarDCT
            w.write(2, 0); // flags = 0
            w.write(2, 0); // upsampling = 1
            if has_alpha {
                w.write(2, 0); // extra-channel upsampling = 1
            }
            w.write(3, x_qm_scale as u64);
            w.write(3, b_qm_scale() as u64); // b_qm_scale
            // Reference-only frames omit Passes and use the implicit one pass.
            w.write(1, 1); // custom size
            write_frame_dimension(width, w);
            write_frame_dimension(height, w);
            w.write(2, PATCH_REF_ID as u64);
            w.write(1, 1); // save_before_color_transform
            w.write(2, 0); // empty name
            write_loop_filter(epf_iters, gab_enabled, w);
            w.write(2, 0); // no frame-header extensions
        }
    }
}

fn write_loop_filter(epf_iters: u32, gab_enabled: bool, w: &mut BitWriter) {
    if epf_iters == 2 && gab_enabled {
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
            w.write(1, 0); // default epf weights
            w.write(1, 0); // default epf sigma
        }
        w.write(2, 0); // no loop filter extensions
    }
}

fn write_frame_header(
    x_qm_scale: u32,
    epf_iters: u32,
    gab_enabled: bool,
    has_alpha: bool,
    coeff_shifts: &[u32],
    has_patches: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // not all default
    w.write(2, 0); // regular frame
    w.write(1, 0); // vardct
    // Keep decoder-side adaptive DC smoothing enabled. The only optional flag
    // here is kPatches; the skip-smoothing flag remains clear.
    if has_patches {
        w.write(2, 1); // U64 selector for values 1..=16
        w.write(4, 1); // flags = kPatches (2), encoded as value - 1
    } else {
        w.write(2, 0);
    }
    w.write(2, 0); // no upsampling

    // Per-extra-channel upsampling. Same u2S(1,2,4,8) code, default 1:
    // selector "00" gives 1, written once per extra channel.
    if has_alpha {
        w.write(2, 0); // ec_upsampling[0] = 1
    }

    w.write(3, x_qm_scale as u64);
    w.write(3, b_qm_scale() as u64); // b_qm_scale
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

    w.write(1, 1); // last frame
    w.write(2, 0); // no name
    write_loop_filter(epf_iters, gab_enabled, w);
    w.write(2, 0); // no frame header extensions
}

/// Writes the compact block-context map used whenever the default shortcut is
/// not taken.
pub(crate) fn write_compact_block_context_map(
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut BitWriter,
) {
    let empty_codes: [crate::entropy::PrefixCode; 0] = [];
    let empty_configs: [crate::entropy::HybridUintConfig; 0] = [];
    let empty_freqs: [Vec<u16>; 0] = [];
    let empty_syms: [Vec<crate::entropy::AnsEncSymbolInfo>; 0] = [];
    let cm_entropy = EntropyCode {
        context_map: compact_block_context_map(),
        num_contexts: compact_block_context_map().len(),
        prefix_codes: &empty_codes,
        hybrid_uint_configs: &empty_configs,
        num_prefix_codes: 0,
        orig_context_map: None,
        orig_num_contexts: 0,
        use_prefix_code: true,
        ans_freqs: &empty_freqs,
        ans_symbols: &empty_syms,
    };
    crate::entropy::write_context_map(&cm_entropy, huffman_pool, w);
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
    let empty_freqs: [Vec<u16>; 0] = [];
    let empty_syms: [Vec<crate::entropy::AnsEncSymbolInfo>; 0] = [];
    let cm_entropy = EntropyCode {
        context_map: &entries,
        num_contexts: entries.len(),
        prefix_codes: &empty_codes,
        hybrid_uint_configs: &empty_configs,
        num_prefix_codes: 0,
        orig_context_map: None,
        orig_num_contexts: 0,
        use_prefix_code: true,
        ans_freqs: &empty_freqs,
        ans_symbols: &empty_syms,
    };
    crate::entropy::write_context_map(&cm_entropy, &mut scratch.huffman_pool, w);
}

fn write_dc_global(
    distp: &DistanceParams,
    num_dc_groups: usize,
    ac_plan: &crate::ac_context::AcCtxPlan,
    dc_tree: &DcTreeChoice,
    dc_code: &EntropyCode,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    ytob_dc: i32,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    w.write(1, 1); // default dequant DC
    write_quant_scales(distp.global_scale, distp.quant_dc, w);
    write_block_ctx_map(ac_plan, scratch, w);

    // ColorCorrelationParams. The all-default bundle pins the DC plane to the
    // XYB base correlations (X: 0, B: 1); a searched `ytob_dc` needs the
    // explicit form, which costs COLOR_CORRELATION_HEADER_BITS more.
    {
        let factor = crate::color_correlation::K_COLOR_FACTOR as u32;
        if factor == 84 && ytob_dc == 0 {
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
            w.write(16, 0); // base_correlation_x = 0.0
            w.write(16, 0x3C00); // base_correlation_b = 1.0 (kYToBRatio)
            w.write(8, 128); // ytox_dc = 0, offset by 128
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
    if let Some(alpha_plane) = alpha {
        crate::modular::write_global_alpha_modular(alpha_plane, xsize, ysize, scratch, w);
    }
}

/// Which `custom_tables` slot a strategy's quant table lives in, or `None` when
/// its table is not one jixel can override (DCT4X4 uses table 3).
#[inline]
fn quant_table_slot_of(raw_strategy: u8) -> Option<usize> {
    Some(match raw_strategy {
        STRATEGY_DCT => 0,
        STRATEGY_DCT16X16 => 1,
        STRATEGY_DCT32X32 => 2,
        STRATEGY_DCT16X8 | STRATEGY_DCT8X16 => 3,
        STRATEGY_DCT32X16 | STRATEGY_DCT16X32 => 4,
        STRATEGY_DCT4X8 | STRATEGY_DCT8X4 => 5,
        _ => return None,
    })
}

/// Slots whose transform actually appears in the frame.
fn used_quant_table_slots(dc_datas: &[DcGroupData]) -> [bool; 6] {
    let mut used = [false; 6];
    for dc in dc_datas {
        for (_, _, strategy) in dc.ac_strategy.iter_first_blocks() {
            if let Some(slot) = quant_table_slot_of(strategy) {
                used[slot] = true;
            }
        }
    }
    used
}

fn write_dequant_matrices(
    matrices: &crate::quant_weights::DequantMatrices,
    used: &[bool; 6],
    w: &mut BitWriter,
) {
    use crate::quant_weights::f32_to_f16_bits;
    // A table only earns its header if the frame actually uses the transform.
    let table = |slot: usize| -> Option<&crate::quant_weights::BandOverride> {
        used[slot]
            .then(|| matrices.custom_tables[slot].as_ref())
            .flatten()
    };
    if (0..6).all(|slot| table(slot).is_none()) {
        w.write(1, 1); // all_default
        return;
    }
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
            _ => None,
        };
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
    used_quant_tables: &[bool; 6],
    coeff_orders: &crate::coeff_order::CoeffOrders,
    num_groups: usize,
    ac_codes: &[crate::entropy::OwnedEntropyCode],
    lz_code: &crate::entropy::OwnedEntropyCode,
    use_lz77: bool,
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
    // code. Each pass gets its own code (ac_codes[p]); the single-pass LZ77 path
    // instead writes the LZ code in its one HfPass.
    for code in ac_codes {
        crate::coeff_order::write_coeff_orders(coeff_orders, &mut scratch.huffman_pool, w);
        if use_lz77 {
            crate::lz77_ac::write_ac_lz_header_and_code(lz_code, &mut scratch.huffman_pool, w);
        } else {
            w.write(1, 0);
            write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, w);
        }
    }
}

fn write_toc(sizes: &[usize], w: &mut BitWriter) {
    w.write(1, 0); // no permutation
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
    if sections.len() == 4 {
        // Single AC group case: concat sections 1..4 (bitwise) into section 0.
        let tail: Vec<BitWriter> = sections.drain(1..).collect();
        for s in &tail {
            sections[0].append(s);
        }
    }

    let sizes: Vec<usize> = sections
        .iter()
        .map(|s| s.bits_written().div_ceil(8))
        .collect();
    write_toc(&sizes, writer);
    // After write_toc, writer is byte-aligned.
    writer.append_byte_aligned(sections);
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    linear: &Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    patches: bool,
    writer: &mut BitWriter,
) {
    let distp = compute_distance_params(distance);
    let mut xyb = to_xyb_image(ctx, scratch, linear);

    if patches && let Some(plan) = find_lossy_patches(&xyb, &ctx.thread_pool, scratch) {
        let mut regular = xyb.clone();
        gaborize(&mut regular, &distp);
        let mut regular_writer = BitWriter::new();
        encode_frame_core(
            ctx,
            scratch,
            distance,
            regular,
            alpha,
            coeff_shifts,
            VarDctFrameKind::Regular,
            &mut regular_writer,
        );

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
                    out,
                );
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
                scratch,
                &mut modular_bits,
            );
            let mut vardct_bits = BitWriter::new();
            encode_vardct_atlas(modular_atlas, scratch, &mut vardct_bits);
            if modular_ok && modular_bits.bits_written() < vardct_bits.bits_written() {
                patched_writer.append(&modular_bits);
                references.extend(modular_refs);
            } else {
                vardct_idx.append(&mut modular_idx);
                vardct_idx.sort_unstable();
            }
        }

        if !vardct_idx.is_empty() {
            let (vardct_atlas, vardct_refs) =
                crate::patches::pack_lossy_atlas(&xyb, clone_groups(&vardct_idx), PATCH_REF_ID);
            encode_vardct_atlas(vardct_atlas, scratch, &mut patched_writer);
            references.extend(vardct_refs);
        }

        let mut base = plan.base;
        gaborize(&mut base, &distp);
        encode_frame_core(
            ctx,
            scratch,
            distance,
            base,
            alpha,
            coeff_shifts,
            VarDctFrameKind::Patched(&references),
            &mut patched_writer,
        );
        if patched_writer.bits_written() < regular_writer.bits_written() {
            writer.append(&patched_writer);
        } else {
            writer.append(&regular_writer);
        }
        return;
    }

    gaborize(&mut xyb, &distp);
    encode_frame_core(
        ctx,
        scratch,
        distance,
        xyb,
        alpha,
        coeff_shifts,
        VarDctFrameKind::Regular,
        writer,
    );
}

/// Power-of-two refinement of the modular atlas quantization lattice.
const MODULAR_ATLAS_LATTICE_SCALE: u32 = 8;

/// The VarDCT atlas is coded this much finer than the frame it serves.
///
/// Two Optuna studies agree: the SS2 cliff starts at ~0.5 on large screenshot
/// content (every occurrence inherits the atlas error) and the plateau is
/// [0.3, 0.5), so 0.45 sits at the rate-optimal edge with measured margin.
const ATLAS_DISTANCE_SCALE: f32 = 0.45;

fn to_xyb_image(ctx: &EncodingContext, scratch: &mut CoderScratch, linear: &Image3F) -> Image3F {
    let mut xyb = Image3F::new(linear.xsize(), linear.ysize());
    crate::xyb::to_xyb_with_fn(
        ctx.to_xyb_band,
        &ctx.xyb,
        linear,
        &mut xyb,
        &ctx.thread_pool,
        scratch,
    );
    xyb
}

/// Pre-invert Gaborish so the decoder's forward pass reproduces `xyb`.
fn gaborize(xyb: &mut Image3F, distp: &DistanceParams) {
    if distp.gab_enabled {
        crate::gaborish::gaborish_inverse(xyb, 0.990_851_1);
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_frame_core(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    opsin: Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    writer: &mut BitWriter,
) {
    let num_threads = ctx.thread_pool.num_threads();
    let dim = ImageDim::new(opsin.xsize(), opsin.ysize());
    let distp = compute_distance_params(distance);

    // Progressive lossy splits each quantized AC coeff across `num_passes`
    // passes by a decreasing per-pass shift (last = 0). The decoder reconstructs
    // C = sum_p (sent_p << shift_p) (jxl-vardct hf_coeff.rs:185,191).
    let num_passes = coeff_shifts.len();

    let num_sections = 2 + dim.num_dc_groups + num_passes * dim.num_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // Phase 1: set up every DC group, then encode every AC group. Keeping the
    // two apart lets AC groups from all DC groups share one steal_map instead of
    // one serialized burst per DC group. Merges back in (dc, gix) order, so the
    // output stays bit-identical to single-threaded.
    let group_coords: Vec<(usize, usize)> = (0..dim.ysize_dc_groups)
        .flat_map(|gy| (0..dim.xsize_dc_groups).map(move |gx| (gx, gy)))
        .collect();
    let opsin = &opsin;

    // Split the thread budget across nesting levels: `outer` lanes steal DC
    // groups, each parallelizing its AC-strategy bands with the remainder, so
    // the setup phase saturates all cores even with few (large) DC groups.
    let outer = group_coords.len().min(num_threads.max(1));
    let setup_budget = num_threads.max(1).div_ceil(outer);
    let setups = ctx
        .thread_pool
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

    let mut dc_datas: Vec<DcGroupData> = Vec::with_capacity(setups.len());
    let mut ac_tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (dc_idx, (dc_data, gxs, gys)) in setups.into_iter().enumerate() {
        ac_tasks.extend((0..gxs * gys).map(|g| (dc_idx, g % gxs, g / gxs)));
        dc_datas.push(dc_data);
    }

    // Per-image quant-field threshold for the fine AC block-context layout.
    // The median splits the field into halves with genuinely different
    // coefficient statistics; whether any split is *kept* is decided later
    // from real token stats, so a useless threshold costs nothing.
    let qf_threshold = ac_qf_threshold(&dc_datas);

    let dc_ref = &dc_datas;
    let natural_orders = crate::coeff_order::CoeffOrders::natural();
    // Tally coefficient positions only when the refine pass will run to consume
    // the derived orders: pass one alone can never adopt them (its tokens are
    // already written on the natural scan), so the work would be wasted and its
    // mere presence perturbs codegen enough to shift a few borderline
    // quantizer roundings on paths that should be untouched.
    let want_order_stats =
        num_passes == 1 && (0.03..=24.0).contains(&distance) && ctx.speed == Speed::Slow;
    let results = ctx
        .thread_pool
        .steal_map(scratch, ac_tasks.len(), |t, scratch| {
            let (dc_idx, gx, gy) = ac_tasks[t];
            let (dc_gx, dc_gy) = group_coords[dc_idx];
            let (p, local, local_float, stats) = process_ac_group(
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
                0,
                None,
                &natural_orders,
                want_order_stats,
                qf_threshold,
            );
            (dc_idx, gx, gy, p, local, local_float, stats)
        });

    let mut all_pending: Vec<PendingAcGroup> = Vec::with_capacity(results.len());
    let mut order_stats = crate::coeff_order::OrderStats::new();
    for (dc_idx, gx, gy, p, local, local_float, stats) in results {
        merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
        merge_dc_float(&mut dc_datas[dc_idx], gx, gy, &local_float);
        all_pending.push(p);
        if let Some(s) = stats {
            order_stats.merge(&s);
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
    let mut coeff_orders = crate::coeff_order::CoeffOrders::natural();

    let mut ytob_dc = 0i32;
    if num_passes == 1 && (0.03..=24.0).contains(&distance) && ctx.speed == Speed::Slow {
        coeff_orders = crate::coeff_order::derive_orders(&order_stats);
        ytob_dc = choose_ytob_dc(
            &dc_datas,
            ctx.fill_ytob_row,
            ctx.accumulate_ytob_weights,
            ctx.fill_ytob_residuals,
            &mut scratch.dc_cfl_cur,
            &mut scratch.dc_cfl_prev,
        );
        let provisional_code = crate::entropy::optimize_entropy_code_ac_streams(
            all_pending.iter().map(|pg| pg.tokens[0].as_slice()),
            crate::ac_context::K_NUM_FINE_AC_CONTEXTS,
            &mut scratch.huffman_pool,
            // No config selection here: these prices feed RDOQ, and even a
            // one-token nudge can push the clustering off a knife-edge merge.
            // Selection applies to the final codes only.
            false,
        );
        let prices = crate::entropy::FrozenTokenPrices::new(&provisional_code);
        let dc_ref = &dc_datas;
        let refined = ctx
            .thread_pool
            .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                let (dc_idx, gx, gy) = ac_tasks[t];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (p, local, _local_float, _) = process_ac_group(
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
                    false,
                    qf_threshold,
                );
                (dc_idx, gx, gy, p, local)
            });
        all_pending.clear();
        for (dc_idx, gx, gy, p, local) in refined {
            merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
            all_pending.push(p);
        }

        // Closed-loop DC rounding against the decoder's adaptive DC
        // smoothing: sub-quantstep DC precision on smooth content for
        // near-zero rate. Above the curve across d≈1..6 on both corpora.
        if dc_smooth_enabled(distance, ctx.speed) {
            let steps = [
                1.0 / (crate::quant_weights::INV_DC_QUANT[0] * distp.scale_dc),
                1.0 / (crate::quant_weights::INV_DC_QUANT[1] * distp.scale_dc),
                1.0 / (crate::quant_weights::INV_DC_QUANT[2] * distp.scale_dc),
            ];
            let r_b = 1.0 + ytob_dc as f32 / crate::color_correlation::K_COLOR_FACTOR;
            if dc_datas.len() == 1 {
                let dc = &mut dc_datas[0];
                crate::dc_smooth::optimize_dc_rounding(
                    &mut dc.quant_dc,
                    &dc.dc_float,
                    steps,
                    r_b,
                    Some((&ctx.thread_pool, scratch)),
                );
            } else {
                // The smoothing filter spans the whole DC plane, so stitch
                // the per-DC-group planes together, optimize once, and
                // scatter the result back.
                let (wb, hb) = (dim.xsize_blocks, dim.ysize_blocks);
                let mut full_q = Image3S::new(wb, hb);
                let mut full_f = Image3F::new(wb, hb);
                const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
                for (i, &(gx, gy)) in group_coords.iter().enumerate() {
                    let (ox, oy) = (gx * DC_GROUP_BLOCKS, gy * DC_GROUP_BLOCKS);
                    let src_q = &dc_datas[i].quant_dc;
                    let src_f = &dc_datas[i].dc_float;
                    for c in 0..3 {
                        for ly in 0..src_q.ysize() {
                            let n = src_q.xsize();
                            full_q.plane_row_mut(c, oy + ly)[ox..ox + n]
                                .copy_from_slice(&src_q.plane_row(c, ly)[..n]);
                            full_f.plane_row_mut(c, oy + ly)[ox..ox + n]
                                .copy_from_slice(&src_f.plane_row(c, ly)[..n]);
                        }
                    }
                }
                crate::dc_smooth::optimize_dc_rounding(
                    &mut full_q,
                    &full_f,
                    steps,
                    r_b,
                    Some((&ctx.thread_pool, scratch)),
                );
                for (i, &(gx, gy)) in group_coords.iter().enumerate() {
                    let (ox, oy) = (gx * DC_GROUP_BLOCKS, gy * DC_GROUP_BLOCKS);
                    let dst = &mut dc_datas[i].quant_dc;
                    for c in 0..3 {
                        for ly in 0..dst.ysize() {
                            let n = dst.xsize();
                            dst.plane_row_mut(c, ly)[..n]
                                .copy_from_slice(&full_q.plane_row(c, oy + ly)[ox..ox + n]);
                        }
                    }
                }
            }
        }
    }

    // Phase 2: build adaptive DC entropy code from all DC + AC-metadata tokens.
    // Per-leaf DC predictor selection.
    let token_groups = ctx
        .thread_pool
        .steal_map(scratch, dc_datas.len(), |i, _scratch| {
            let mut props = Vec::new();
            let wp = collect_dc_tokens(&dc_datas[i], &DC_PREDICTOR_WEIGHTED, &mut props);
            // Both arms run identical WP state; properties are shared.
            let mut discard = Vec::new();
            let grad = collect_dc_tokens(&dc_datas[i], &[true; K_NUM_DC_CONTEXTS], &mut discard);
            let mut meta_props = Vec::new();
            let meta = collect_ac_metadata_tokens(&dc_datas[i], &mut meta_props, distance);
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

    let dc_gradient = choose_dc_predictors(
        &wp_tokens_per_group,
        &grad_tokens_per_group,
        &mut scratch.dc_predictor,
    );
    // Arm A: the static tree with per-leaf predictor flips.
    let dc_tokens_static: Vec<Vec<Token>> = wp_tokens_per_group
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
        .collect();
    let code_static = crate::entropy::optimize_entropy_code_ac_streams(
        dc_tokens_static
            .iter()
            .map(Vec::as_slice)
            .chain(meta_tokens_per_group.iter().map(Vec::as_slice)),
        K_NUM_DC_CONTEXTS,
        &mut scratch.huffman_pool,
        ctx.speed != Speed::Fastest,
    );

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
        ctx.speed != Speed::Fastest,
    );

    // Price both arms end to end: serialized tree + entropy-code header
    // (measured by writing them) plus the payload estimate, and require the
    // learned arm to win by a margin. The margin absorbs the small estimator
    // noise so near-ties keep the battle-tested static tree.
    let payload =
        |dc: &[Vec<Token>], meta: &[Vec<Token>], code: &crate::entropy::OwnedEntropyCode| {
            crate::lz77_ac::estimate_ac_plain_bits(
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
                write_context_tree(dim.num_dc_groups, grad, &mut scratch.huffman_pool, &mut w);
            }
            DcTreeChoice::Learned(tokens) => {
                write_tree_tokens(tokens, &mut scratch.huffman_pool, &mut w);
            }
        }
        write_entropy_code(code, &mut scratch.huffman_pool, &mut w);
        w.bits_written() as u64
    };
    const LEARNED_TREE_MARGIN_BITS: u64 = 64;
    let tree_static = DcTreeChoice::Static(dc_gradient);
    let tree_learned = DcTreeChoice::Learned(learned.tokens);
    let cost_static = header(&tree_static, &code_static.as_ref(), scratch)
        + payload(&dc_tokens_static, &meta_tokens_per_group, &code_static);
    let cost_learned = header(&tree_learned, &code_learned.as_ref(), scratch)
        + payload(&dc_tokens_learned, &meta_tokens_learned, &code_learned);

    let use_learned = cost_learned + LEARNED_TREE_MARGIN_BITS < cost_static;
    let (dc_tokens_per_group, meta_tokens_per_group, dc_code_owned, dc_tree) = if use_learned {
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
    let dc_code = dc_code_owned.as_ref();

    // Per-image AC block-context plan: keep quant-field splits only where the
    // real token statistics pay for them within the spec's 16-context budget.
    // The greedy proposes; the arm gate below disposes, comparing real
    // entropy-code headers plus payload estimates because the Shannon proxy
    // reliably overestimates what clustering-aware coding realizes.
    let proposed = crate::ac_context::plan_block_ctx_map(
        all_pending
            .iter()
            .flat_map(|pending| pending.tokens.iter().map(Vec::as_slice)),
        qf_threshold,
    );
    let baseline = crate::ac_context::AcCtxPlan::baseline();

    let build_codes = |pending: &[PendingAcGroup],
                       num_contexts: usize,
                       scratch: &mut CoderScratch|
     -> Vec<crate::entropy::OwnedEntropyCode> {
        (0..num_passes)
            .map(|pass| {
                crate::entropy::optimize_entropy_code_ac_streams(
                    pending.iter().map(|p| p.tokens[pass].as_slice()),
                    num_contexts,
                    &mut scratch.huffman_pool,
                    ctx.speed != Speed::Fastest,
                )
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
            bits += crate::lz77_ac::estimate_ac_plain_bits(
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
                    for t in pass_tokens.iter_mut() {
                        *t = Token::new(plan.remap(t.context), t.value);
                    }
                }
            });
    };

    let (ac_plan, ac_code_per_pass) = if proposed == baseline {
        remap_tokens(&mut all_pending, &baseline, scratch);
        let codes = build_codes(&all_pending, baseline.num_ac_contexts(), scratch);
        (baseline, codes)
    } else {
        // Materialize the proposed arm, remap the originals to baseline in
        // place, and keep whichever arm's real bits win.
        let plan_ref = &proposed;
        let mut planned: Vec<Vec<Vec<Token>>> = all_pending
            .iter()
            .map(|p| {
                p.tokens
                    .iter()
                    .map(|pass_tokens| {
                        pass_tokens
                            .iter()
                            .map(|t| Token::new(plan_ref.remap(t.context), t.value))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        remap_tokens(&mut all_pending, &baseline, scratch);

        let base_codes = build_codes(&all_pending, baseline.num_ac_contexts(), scratch);
        let base_bits = arm_bits(&all_pending, &base_codes, &baseline, scratch);
        // Swap in the proposed tokens to price them with the same helpers.
        for (p, planned_tokens) in all_pending.iter_mut().zip(&mut planned) {
            std::mem::swap(&mut p.tokens, planned_tokens);
        }
        let plan_codes = build_codes(&all_pending, proposed.num_ac_contexts(), scratch);
        let plan_bits = arm_bits(&all_pending, &plan_codes, &proposed, scratch);

        const AC_PLAN_MARGIN_BITS: u64 = 256;
        if plan_bits + AC_PLAN_MARGIN_BITS < base_bits {
            (proposed, plan_codes)
        } else {
            // Swap the baseline tokens back.
            for (p, planned_tokens) in all_pending.iter_mut().zip(&mut planned) {
                std::mem::swap(&mut p.tokens, planned_tokens);
            }
            (baseline, base_codes)
        }
    };

    let ac_num_contexts = ac_plan.num_ac_contexts() + 1;

    // LZ77 path is single-pass only for now: it compresses one token stream per
    // group. Multi-pass uses the per-pass plain codes.
    let mut ac_lz_per_group: Vec<Vec<crate::lz77_ac::AcLz>> = Vec::new();
    let ac_lz_code_owned;
    let use_lz77;
    if num_passes == 1 {
        ac_lz_per_group = ctx
            .thread_pool
            .steal_map(scratch, all_pending.len(), |i, _scratch| {
                crate::lz77_ac::lz77_compress_ac(&all_pending[i].tokens[0])
            });
        ac_lz_code_owned = crate::lz77_ac::build_ac_lz_code(
            &ac_lz_per_group,
            ac_num_contexts,
            &mut scratch.huffman_pool,
        );
        let lz_bits = crate::lz77_ac::estimate_ac_lz_bits(
            &ac_lz_per_group,
            &ac_lz_code_owned,
            ac_num_contexts,
        );
        let plain_bits = crate::lz77_ac::estimate_ac_plain_bits(
            all_pending
                .iter()
                .map(|pending| pending.tokens[0].as_slice()),
            &ac_code_per_pass[0],
        );
        // Require a real margin to cover the LZ77 header + distance-context cost.
        use_lz77 = lz_bits + 512 < plain_bits;
    } else {
        ac_lz_code_owned = crate::lz77_ac::build_ac_lz_code(
            &ac_lz_per_group,
            ac_num_contexts,
            &mut scratch.huffman_pool,
        );
        use_lz77 = false;
    }

    // Phase 4: write DC global with adaptive DC code.
    if let VarDctFrameKind::Patched(references) = frame_kind {
        crate::lossless::write_patch_dictionary(
            references,
            alpha.is_some(),
            scratch,
            &mut sections[0],
        );
    }
    write_dc_global(
        &distp,
        dim.num_dc_groups,
        &ac_plan,
        &dc_tree,
        &dc_code,
        alpha,
        dim.xsize,
        dim.ysize,
        ytob_dc,
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
                    dc_code.hybrid_uint_configs,
                    &mut w,
                );
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
                    dc_code.hybrid_uint_configs,
                    &mut w,
                );
            }
            w
        });
    for (i, section) in dc_sections.into_iter().enumerate() {
        sections[1 + i] = section;
    }

    write_ac_global(
        ctx.matrices,
        &used_quant_table_slots(&dc_datas),
        &coeff_orders,
        dim.num_groups,
        &ac_code_per_pass,
        &ac_lz_code_owned,
        use_lz77,
        scratch,
        &mut sections[1 + dim.num_dc_groups],
    );

    // Phase 7: write each (pass, group) AC section. Section index for
    // (pass, group) = 2 + num_dc_groups + pass*num_groups + group_idx
    // (jxl-frame toc.rs:196-200). With LZ77 (single-pass only) we emit the
    // compressed stream; otherwise raw tokens via the shared plain code.
    let num_ac_sections = all_pending.len() * num_passes;
    let ac_sections = ctx
        .thread_pool
        .steal_map(scratch, num_ac_sections, |task, _scratch| {
            let i = task / num_passes;
            let pass = task % num_passes;
            let pg = &all_pending[i];
            let pass_tokens = &pg.tokens[pass];
            let mut w = BitWriter::new();
            let section_idx = 2 + dim.num_dc_groups + pass * dim.num_groups + pg.group_idx;
            if use_lz77 {
                for t in &ac_lz_per_group[i] {
                    crate::lz77_ac::write_ac_lz(*t, &ac_lz_code_owned, ac_num_contexts, &mut w);
                }
            } else {
                let code_ref = ac_code_per_pass[pass].as_ref();
                if code_ref.use_prefix_code {
                    for t in pass_tokens {
                        write_token(*t, &code_ref, &mut w);
                    }
                } else {
                    // rANS: the whole group's tokens are encoded as one LIFO unit.
                    crate::entropy::write_ans_tokens(
                        pass_tokens,
                        code_ref.context_map,
                        code_ref.ans_symbols,
                        code_ref.hybrid_uint_configs,
                        &mut w,
                    );
                }
            }
            (section_idx, w)
        });
    for (section_idx, section) in ac_sections {
        sections[section_idx] = section;
    }
    // Modular alpha: extra-channel modular data is decoded in the last pass
    // (the only pass whose modular sub-image shift range is set when num_ds=0,
    // jxl-frame lib.rs:101-108 / pass_group.rs:93). Write it into that section.
    if let Some(alpha_plane) = alpha {
        let last_pass = num_passes - 1;
        let alpha_sections =
            ctx.thread_pool
                .steal_map(scratch, dim.num_groups, |abs_group_id, scratch| {
                    let image_gx = abs_group_id % dim.xsize_groups;
                    let image_gy = abs_group_id / dim.xsize_groups;
                    let group_x0 = image_gx * K_GROUP_DIM;
                    let group_y0 = image_gy * K_GROUP_DIM;
                    let group_xsize = K_GROUP_DIM.min(dim.xsize.saturating_sub(group_x0));
                    let group_ysize = K_GROUP_DIM.min(dim.ysize.saturating_sub(group_y0));
                    let ac_group_idx =
                        2 + dim.num_dc_groups + last_pass * dim.num_groups + abs_group_id;
                    let mut w = BitWriter::new();
                    crate::modular::write_ac_group_alpha(
                        alpha_plane,
                        dim.xsize,
                        dim.ysize,
                        group_x0,
                        group_y0,
                        group_xsize,
                        group_ysize,
                        scratch,
                        &mut w,
                    );
                    (ac_group_idx, w)
                });
        for (section_idx, alpha_section) in alpha_sections {
            sections[section_idx].append(&alpha_section);
        }
    }

    write_frame_header_kind(
        distp.x_qm_scale,
        distp.epf_iters,
        distp.gab_enabled,
        alpha.is_some(),
        coeff_shifts,
        frame_kind,
        writer,
    );
    combine_sections(&mut sections, writer);
}

/// Per-AC-group buffered tokens. For progressive (multi-pass) encoding the
/// quantized AC coefficients of each block are split across passes; `tokens`
/// holds one token stream per pass. `group_idx` is the raster group index
/// (0..num_groups); the section for (pass, group) is
/// `2 + num_dc_groups + pass*num_groups + group_idx`.
pub(crate) struct PendingAcGroup {
    pub(crate) group_idx: usize,
    pub(crate) tokens: Vec<Vec<Token>>,
}

#[allow(clippy::too_many_arguments)]
/// Set up one DC group (quant field, AC strategy, CfL, DCT4X4 gate); its AC
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
) -> (DcGroupData, usize, usize) {
    // DC group rect in pixels (clamped to image bounds).
    let dc_group_x0 = dc_gx * K_DC_GROUP_DIM;
    let dc_group_y0 = dc_gy * K_DC_GROUP_DIM;
    let dc_group_xsize = K_DC_GROUP_DIM.min(dim.xsize.saturating_sub(dc_group_x0));
    let dc_group_ysize = K_DC_GROUP_DIM.min(dim.ysize.saturating_sub(dc_group_y0));
    let dc_group_xsize_blocks = dc_group_xsize.div_ceil(K_BLOCK_DIM);
    let dc_group_ysize_blocks = dc_group_ysize.div_ceil(K_BLOCK_DIM);
    let dc_group_xsize_groups = dc_group_xsize.div_ceil(K_GROUP_DIM);
    let dc_group_ysize_groups = dc_group_ysize.div_ceil(K_GROUP_DIM);

    let mut dc_data = DcGroupData::new(dc_group_xsize_blocks, dc_group_ysize_blocks);

    (ctx.fill_quant_field)(
        &mut scratch.aq_map,
        opsin,
        &mut dc_data.raw_quant_field,
        dc_group_x0,
        dc_group_y0,
        distp.distance,
        1.0 / distp.scale,
    );

    // Apply perceptual AQ before transform selection. Candidate costs and the
    // reconstruction rerank must see the same content-adaptive quant field that
    // will ultimately be used for coefficient coding. `fill_ac_strategy` applies
    // the transform-size adjustment to this field after selection.
    if let Some(boost) = ctx.boost.as_ref() {
        crate::dark_aq::apply_boost(
            &mut scratch.dark_octile,
            boost,
            opsin,
            &mut dc_data.raw_quant_field,
            dc_group_x0,
            dc_group_y0,
            distp.distance,
            ctx.apply_quant_field_gain,
            ctx.dark_structure_stats,
        );
    }

    if ctx.speed == Speed::Slow {
        crate::structure_aq::apply(
            &mut scratch.structure_corrections,
            opsin,
            &mut dc_data.raw_quant_field,
            dc_group_x0,
            dc_group_y0,
            distp.distance,
            ctx.dct8x8,
            ctx.block_features,
            ctx.apply_structure_corrections,
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
        &mut dc_data.ytox_map,
        &mut dc_data.ytob_map,
        distp.distance,
    );
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
        num_threads,
    );
    // Sub-8x8 activation gate. `fill_ac_strategy` greedily commits every block
    // where DCT4X4, DCT4X8, or DCT8X4 wins the per-block RD comparison, but a
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
                    if is_sub8_strategy(strategy) {
                        positions.push((x, y, strategy));
                    }
                }
            }
        }
        if !positions.is_empty() {
            let cost_with = meta_entropy_cost(&dc_data, scratch, distp.distance);
            for &(x, y, _) in &positions {
                dc_data.ac_strategy.set_first(x, y, STRATEGY_DCT);
            }
            let cost_without = meta_entropy_cost(&dc_data, scratch, distp.distance);
            let meta_delta = cost_with.saturating_sub(cost_without) as f32;
            if dc_data.sub8_benefit > crate::ac_strategy::RD_LAMBDA * meta_delta {
                // Worth it: restore each exact sub-8x8 selection.
                for &(x, y, strategy) in &positions {
                    dc_data.ac_strategy.set_first(x, y, strategy);
                }
            }
            // else: leave reverted to DCT8.
        }
    }

    (dc_data, dc_group_xsize_groups, dc_group_ysize_groups)
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

/// Merge a group-local unquantized DC plane (same geometry as
/// [`merge_quant_dc`]) into the DC group's `dc_float`, sizing the
/// destination on first use. A no-op when the capture is disabled (empty
/// local image).
fn merge_dc_float(dc: &mut DcGroupData, gx: usize, gy: usize, local: &Image3F) {
    if local.xsize() == 0 {
        return;
    }
    if dc.dc_float.xsize() == 0 {
        dc.dc_float = Image3F::new(dc.quant_dc.xsize(), dc.quant_dc.ysize());
    }
    let ox = gx * K_GROUP_DIM_IN_BLOCKS;
    let oy = gy * K_GROUP_DIM_IN_BLOCKS;
    let (gwb, ghb) = (local.xsize(), local.ysize());
    for c in 0..3 {
        for ly in 0..ghb {
            let src = local.plane_row(c, ly);
            dc.dc_float.plane_row_mut(c, oy + ly)[ox..ox + gwb].copy_from_slice(&src[..gwb]);
        }
    }
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
    collect_order_stats: bool,
    qf_threshold: u32,
) -> (
    PendingAcGroup,
    Image3S,
    Image3F,
    Option<crate::coeff_order::OrderStats>,
) {
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
    // Float DC targets are only captured when the DC-smoothing rounding pass
    // is going to consume them (`write_ac_group` skips the empty image).
    let mut local_dc_float = if num_passes == 1 && dc_smooth_enabled(distp.distance, ctx.speed) {
        Image3F::new(gwb, ghb)
    } else {
        Image3F::new(0, 0)
    };
    let mut num_nzeros: Vec<Image3B> = (0..num_passes)
        .map(|_| Image3B::new(K_GROUP_DIM_IN_BLOCKS, K_GROUP_DIM_IN_BLOCKS))
        .collect();
    let mut tokens: Vec<Vec<Token>> = (0..num_passes)
        .map(|_| Vec::with_capacity(K_GROUP_DIM_IN_BLOCKS * K_GROUP_DIM_IN_BLOCKS * 4))
        .collect();
    let mut order_stats = collect_order_stats.then(crate::coeff_order::OrderStats::new);

    for ty in 0..group_ysize_tiles {
        let stripe_x0 = group_x0;
        let stripe_y0 = group_y0 + ty * K_TILE_DIM;
        let stripe_xsize = group_xsize;
        let stripe_ysize = K_TILE_DIM.min(dim.ysize.saturating_sub(stripe_y0));
        let stripe_xsize_padded = stripe_xsize.div_ceil(K_BLOCK_DIM) * K_BLOCK_DIM;
        let stripe_ysize_padded = stripe_ysize.div_ceil(K_BLOCK_DIM) * K_BLOCK_DIM;

        let stripe = build_stripe(
            opsin,
            stripe_x0,
            stripe_y0,
            stripe_xsize,
            stripe_ysize,
            stripe_xsize_padded,
            stripe_ysize_padded,
        );

        let stripe_brect = Rect::new(
            qorigin_x,
            qorigin_y + ty * K_TILE_DIM_IN_BLOCKS,
            stripe_xsize_padded / K_BLOCK_DIM,
            stripe_ysize_padded / K_BLOCK_DIM,
        );

        write_ac_group(
            ctx,
            &mut scratch.ac_group,
            &stripe,
            stripe_brect,
            distp.scale,
            distp.scale_dc,
            distp.distance,
            distp.x_qm_scale,
            dc_data,
            ytob_dc,
            &mut local_quant_dc,
            &mut local_dc_float,
            qorigin_x,
            qorigin_y,
            &mut num_nzeros,
            coeff_shifts,
            rdoq_prices,
            coeff_orders,
            order_stats.as_mut(),
            false,
            qf_threshold,
            &mut tokens,
        );
    }

    (
        PendingAcGroup {
            group_idx: image_gy * dim.xsize_groups + image_gx,
            tokens,
        },
        local_quant_dc,
        local_dc_float,
        order_stats,
    )
}

/// Carve a stripe out of the (already-XYB-converted, gaborized) opsin image,
/// padding to whole blocks by edge-replication.
fn build_stripe(
    opsin: &Image3F,
    x0: usize,
    y0: usize,
    xsize: usize,
    ysize: usize,
    xsize_padded: usize,
    ysize_padded: usize,
) -> Image3F {
    let mut stripe = Image3F::new(xsize_padded, ysize_padded);
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
    stripe
}

#[cfg(test)]
mod tests {
    use super::{
        DC_REFINE_HOLD, DC_REFINE_PEAK, DC_REFINE_RELEASE, MIN_TOKENS_PER_DC_LEAF,
        choose_dc_predictors, compute_distance_params, dc_refinement, quant_dc,
    };
    use crate::coder_scratch::DcPredictorScratch;
    use crate::entropy::Token;

    /// Build one group of `n` tokens in `context`, all carrying `value`.
    fn leaf_tokens(context: u32, value: u32, n: usize) -> Vec<Vec<Token>> {
        vec![(0..n).map(|_| Token::new(context, value)).collect()]
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
    fn epf_never_requests_the_third_iteration() {
        // The third EPF pass erases texture rather than ringing, so the
        // schedule tops out at two iterations no matter how coarse the
        // distance gets. Guards against a threshold creeping back in.
        for d in [0.0, 0.1, 0.5, 0.69] {
            assert_eq!(compute_distance_params(d).epf_iters, 0, "d={d}");
        }
        for d in [0.7, 1.0, 1.49] {
            assert_eq!(compute_distance_params(d).epf_iters, 1, "d={d}");
        }
        for d in [1.5, 2.0, 4.0, 6.0, 12.0, 25.0] {
            assert_eq!(compute_distance_params(d).epf_iters, 2, "d={d}");
        }
    }

    #[test]
    fn ac_scale_keeps_changing_after_dc_distance_caps() {
        let d4 = compute_distance_params(4.0);
        let d45 = compute_distance_params(4.5);
        let d5 = compute_distance_params(5.0);
        assert!(d4.global_scale > d45.global_scale);
        assert!(d45.global_scale > d5.global_scale);
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
