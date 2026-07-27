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
use crate::ac_context::{K_COMPACT_BLOCK_CONTEXT_MAP, K_NUM_AC_CONTEXTS};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::{DcGroupData, STRATEGY_DCT, is_sub8_strategy};
use crate::dct::fmla;
use crate::enc_group::write_ac_group;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::entropy::{
    EntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code, write_token,
};
use crate::image::{Image3B, Image3F, Image3S, Rect};
use crate::patches::{PATCH_REF_ID, VarDctFrameKind, find_lossy_patches};
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
    // loss on smooth content. Rebalances the d>=4 tail toward AC; measured
    // on photo/fractal/glow/portrait sets: on-or-above the RD curve, up to
    // +1.4 SSIMULACRA2 rate-equivalent, and the quality-vs-d cliff softens
    // by 4-6 SSIMULACRA2 at d=6. No effect at d <= 3.5.
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

/// Emit DC tokens for one DC group (in channel order Y, X, B).
/// Same as write_dc_tokens, but returns the tokens instead of writing them.
/// Use this to build adaptive entropy codes from the actual token distribution
/// before committing the bit pattern.
pub(crate) fn collect_dc_tokens(dc_data: &DcGroupData) -> Vec<Token> {
    let mut tokens = Vec::new();

    // Weighted-predictor DC, mirroring libjxl's kWPFixedDC path (enc_modular.cc
    // AddVarDCTDC at speed tiers falcon..squirrel): the guess is the
    // self-correcting WP and the context is the WP-error property (modular
    // property 15) bucketed by the same +-500 cutoffs the gradient tree used,
    // so K_GRADIENT_CONTEXT_LUT applies unchanged (write_context_tree emits the
    // matching tree: identical structure, property 15, Weighted leaves). The WP
    // state machine and border conventions are the bit-faithful lossless-path
    // ones, so encoder residuals match the reference decoder exactly.
    for c in [1usize, 0, 2] {
        let plane = dc_data.quant_dc.plane(c);
        let ysize = plane.ysize();
        let xsize = plane.xsize();
        let mut wp = crate::enc_lossless::WpState::new(xsize);
        for y in 0..ysize {
            let row_cur = plane.row(y);
            let row_above = if y > 0 { Some(plane.row(y - 1)) } else { None };
            let row_above2 = if y > 1 { Some(plane.row(y - 2)) } else { None };
            for (x, &here) in row_cur[..xsize].iter().enumerate() {
                let v = here as i64;
                let w_ = if x > 0 {
                    row_cur[x - 1] as i64
                } else if let Some(a) = row_above {
                    a[x] as i64
                } else {
                    0
                };
                let n_ = match row_above {
                    Some(a) => a[x] as i64,
                    None => w_,
                };
                let nw_ = if x > 0
                    && let Some(row_above) = row_above
                {
                    row_above[x - 1] as i64
                } else {
                    w_
                };
                let ne_ = match row_above {
                    Some(a) if x + 1 < xsize => a[x + 1] as i64,
                    _ => n_,
                };
                let nn_ = match row_above2 {
                    Some(a) => a[x] as i64,
                    None => n_,
                };
                let p = wp.predict(x, y, n_, w_, ne_, nw_, nn_);
                let prop = i64::clamp(
                    K_GRAD_RANGE_MID + wp.wp_prop,
                    K_GRAD_RANGE_MIN,
                    K_GRAD_RANGE_MAX,
                );
                wp.update(v, x, y);
                let ctx_id = K_GRADIENT_CONTEXT_LUT[prop as usize] as u32;
                tokens.push(Token::new(ctx_id, pack_signed((v - p) as i32)));
            }
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
pub(crate) fn collect_ac_metadata_tokens(dc_data: &DcGroupData) -> Vec<Token> {
    let mut tokens = Vec::new();
    let xsize_blocks = dc_data.ac_strategy.xsize();
    let ysize_blocks = dc_data.ac_strategy.ysize();
    let xtiles = dc_data.ytox_map.xsize();
    let ytiles = dc_data.ytox_map.ysize();

    // (a) YtoX and YtoB tokens with gradient prediction.
    for c in 0..2usize {
        let cfl_map = if c == 0 {
            &dc_data.ytox_map
        } else {
            &dc_data.ytob_map
        };
        for y in 0..ytiles {
            let cfl_row = cfl_map.row(y);
            for (x, &here) in cfl_row[..xtiles].iter().enumerate() {
                let row_above = if y > 0 {
                    Some(cfl_map.row(y - 1))
                } else {
                    None
                };
                let left: i64 = if x > 0 {
                    cfl_map.row(y)[x - 1] as i64
                } else if let Some(rt) = row_above {
                    rt[x] as i64
                } else {
                    0
                };
                let top: i64 = match row_above {
                    Some(rt) => rt[x] as i64,
                    None => left,
                };
                let topleft: i64 = if x > 0 && y > 0 {
                    row_above.unwrap()[x - 1] as i64
                } else {
                    left
                };
                let guess = clamped_gradient(top as i32, left as i32, topleft as i32);
                let residual = here as i32 - guess;
                let ctx_id = 2u32 - c as u32;
                tokens.push(Token::new(ctx_id, pack_signed(residual)));
            }
        }
    }

    // (b) AC strategy tokens.
    let mut left: i32 = 0;
    for y in 0..ysize_blocks {
        for x in 0..xsize_blocks {
            if !dc_data.ac_strategy.is_first_block(x, y) {
                continue;
            }
            let cur = dc_data.ac_strategy.strategy_code(x, y) as i32;
            let ctx_id = if left > 11 {
                7
            } else if left > 5 {
                8
            } else if left > 3 {
                9
            } else {
                10
            } as u32;
            tokens.push(Token::new(ctx_id, pack_signed(cur)));
            left = cur;
        }
    }
    // (c) QF residuals.
    let mut left: i32 = dc_data.ac_strategy.strategy_code(0, 0) as i32;
    for y in 0..ysize_blocks {
        let row_qf = dc_data.raw_quant_field.row(y);
        for x in 0..xsize_blocks {
            if !dc_data.ac_strategy.is_first_block(x, y) {
                continue;
            }
            let cur: i32 = row_qf[x] as i32 - 1;
            let residual: i32 = cur - left;
            let ctx_id = if left > 11 {
                3
            } else if left > 5 {
                4
            } else if left > 3 {
                5
            } else {
                6
            } as u32;
            tokens.push(Token::new(ctx_id, pack_signed(residual)));
            left = cur;
        }
    }
    // (d) EPF tokens.
    let nblocks = xsize_blocks * ysize_blocks;
    for _ in 0..nblocks {
        tokens.push(Token::new(0, pack_signed(4)));
    }
    tokens
}

/// Real prefix-code bit cost of a DC group's AC-metadata token stream under a
/// freshly optimized entropy code. Used by the sub-8x8 activation gate to weigh
/// the exact selected set's meta-stream cost against its RD benefit.
fn meta_entropy_cost(dc_data: &DcGroupData, scratch: &mut CoderScratch) -> u64 {
    let toks = collect_ac_metadata_tokens(dc_data);
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
                // Leaf: PROPERTY(0), PREDICTOR, OFFSET, MUL_LOG, MUL_BITS.
                if leaf_idx >= 11 && tokens[i + 1].value == 5 {
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
    // OptimizeEntropyCode clusters the K_NUM_TREE_CONTEXTS=6 contexts.
    let code = optimize_entropy_code(&tokens, K_NUM_TREE_CONTEXTS, huffman_pool);
    let code_ref = code.as_ref();

    writer.write(1, 1); // not an empty tree
    writer.write(1, 0); // no lz77
    write_entropy_code(&code_ref, huffman_pool, writer);
    for t in &tokens {
        write_token(*t, &code_ref, writer);
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
            w.write(3, 2); // b_qm_scale
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
    w.write(3, 2); // b_qm_scale
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
        context_map: &K_COMPACT_BLOCK_CONTEXT_MAP,
        num_contexts: K_COMPACT_BLOCK_CONTEXT_MAP.len(),
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

fn write_dc_global(
    distp: &DistanceParams,
    num_dc_groups: usize,
    dc_code: &EntropyCode,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    w.write(1, 1); // default dequant DC
    write_quant_scales(distp.global_scale, distp.quant_dc, w);
    w.write(1, 0); // non-default BlockCtxMap
    w.write(16, 0); // no dc ctx, no qft

    // WriteContextMap with kCompactBlockContextMap (only context map, no prefix codes).
    {
        // Empty prefix-codes slice; WriteContextMap builds its own.
        let empty_codes: [crate::entropy::PrefixCode; 0] = [];
        let empty_configs: [crate::entropy::HybridUintConfig; 0] = [];
        let empty_freqs: [Vec<u16>; 0] = [];
        let empty_syms: [Vec<crate::entropy::AnsEncSymbolInfo>; 0] = [];
        let block_context_map = &K_COMPACT_BLOCK_CONTEXT_MAP;
        let cm_entropy = EntropyCode {
            context_map: block_context_map,
            num_contexts: block_context_map.len(),
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

    w.write(1, 1); // default DC clamp (= ColorCorrelationParams.all_default = true)

    // Global tree.
    // write_context_tree emits "have_tree=1 + Histograms::decode (tree's own entropy code)
    // + tree tokens". The TREE'S PIXEL HISTOGRAMS are then written as the next two
    // bits + entropy code (DC entropy code, since it's the global tree used for DC).
    write_context_tree(num_dc_groups, &mut scratch.huffman_pool, w);
    w.write(1, 0); // no lz77 (for the global tree's pixel histograms = dc_code)

    // Then the static DC entropy code: this is the global tree's pixel histograms.
    write_entropy_code(dc_code, &mut scratch.huffman_pool, w);

    // FullModularImage::read happens HERE in the decoder. If we declared an alpha
    // extra channel, write its GroupHeader + local tree + pixel data now.
    if let Some(alpha_plane) = alpha {
        crate::modular::write_global_alpha_modular(alpha_plane, xsize, ysize, scratch, w);
    }
}

fn write_dequant_matrices(matrices: &crate::quant_weights::DequantMatrices, w: &mut BitWriter) {
    use crate::quant_weights::f32_to_f16_bits;
    if matrices.custom_tables.iter().all(Option::is_none) {
        w.write(1, 1); // all_default
        return;
    }
    const K_NUM_QUANT_TABLES: usize = 17;
    const K_QUANT_MODE_LIBRARY: u64 = 0;
    const K_QUANT_MODE_DCT: u64 = 6;
    w.write(1, 0); // all_default = false
    for idx in 0..K_NUM_QUANT_TABLES {
        let bands = match idx {
            0 => matrices.custom_tables[0].as_ref(), // DCT8
            4 => matrices.custom_tables[1].as_ref(), // DCT16X16
            5 => matrices.custom_tables[2].as_ref(), // DCT32X32
            _ => None,
        };
        match bands {
            None => w.write(3, K_QUANT_MODE_LIBRARY),
            Some(o) => {
                w.write(3, K_QUANT_MODE_DCT);
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
    num_groups: usize,
    ac_codes: &[crate::entropy::OwnedEntropyCode],
    lz_code: &crate::entropy::OwnedEntropyCode,
    use_lz77: bool,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    write_dequant_matrices(matrices, w);
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
        w.write(2, 3);
        w.write(13, 0);
        if use_lz77 {
            crate::enc_lz77_ac::write_ac_lz_header_and_code(lz_code, &mut scratch.huffman_pool, w);
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
    if patches && let Some(plan) = find_lossy_patches(linear) {
        let mut regular_writer = BitWriter::new();
        encode_frame_core(
            ctx,
            scratch,
            distance,
            linear,
            alpha,
            coeff_shifts,
            VarDctFrameKind::Regular,
            &mut regular_writer,
        );

        let atlas_alpha =
            zero_alpha_for_lossy(alpha, plan.atlas.xsize().saturating_mul(plan.atlas.ysize()));
        let mut patched_writer = BitWriter::new();
        encode_frame_core(
            ctx,
            scratch,
            distance,
            &plan.atlas,
            atlas_alpha.as_ref(),
            &[0],
            VarDctFrameKind::ReferenceOnly {
                width: plan.atlas.xsize(),
                height: plan.atlas.ysize(),
            },
            &mut patched_writer,
        );
        encode_frame_core(
            ctx,
            scratch,
            distance,
            &plan.base,
            alpha,
            coeff_shifts,
            VarDctFrameKind::Patched(&plan.references),
            &mut patched_writer,
        );
        if patched_writer.bits_written() < regular_writer.bits_written() {
            writer.append(&patched_writer);
        } else {
            writer.append(&regular_writer);
        }
        return;
    }
    encode_frame_core(
        ctx,
        scratch,
        distance,
        linear,
        alpha,
        coeff_shifts,
        VarDctFrameKind::Regular,
        writer,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_frame_core(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    linear: &Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    writer: &mut BitWriter,
) {
    let num_threads = ctx.thread_pool.num_threads();
    let dim = ImageDim::new(linear.xsize(), linear.ysize());
    let distp = compute_distance_params(distance);

    // Progressive lossy splits each quantized AC coeff across `num_passes`
    // passes by a decreasing per-pass shift (last = 0). The decoder reconstructs
    // C = sum_p (sent_p << shift_p) (jxl-vardct hf_coeff.rs:185,191).
    let num_passes = coeff_shifts.len();

    let mut opsin = linear.clone();
    crate::enc_xyb::to_xyb_with_fn(ctx.to_xyb_band, &mut opsin, &ctx.thread_pool, scratch);
    if distp.gab_enabled {
        crate::gaborish::gaborish_inverse(&mut opsin, 0.990_851_1);
    }

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

    let dc_ref = &dc_datas;
    let results = ctx
        .thread_pool
        .steal_map(scratch, ac_tasks.len(), |t, scratch| {
            let (dc_idx, gx, gy) = ac_tasks[t];
            let (dc_gx, dc_gy) = group_coords[dc_idx];
            let (p, local) = process_ac_group(
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
                None,
            );
            (dc_idx, gx, gy, p, local)
        });

    let mut all_pending: Vec<PendingAcGroup> = Vec::with_capacity(results.len());
    for (dc_idx, gx, gy, p, local) in results {
        merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
        all_pending.push(p);
    }

    if num_passes == 1 && (0.03..=24.0).contains(&distance) && ctx.speed != Speed::Fast {
        let provisional_code = crate::entropy::optimize_entropy_code_ac_streams(
            all_pending.iter().map(|pg| pg.tokens[0].as_slice()),
            K_NUM_AC_CONTEXTS,
            &mut scratch.huffman_pool,
        );
        let prices = crate::entropy::FrozenTokenPrices::new(&provisional_code);
        let dc_ref = &dc_datas;
        let refined = ctx
            .thread_pool
            .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                let (dc_idx, gx, gy) = ac_tasks[t];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (p, local) = process_ac_group(
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
                    Some(&prices),
                );
                (dc_idx, gx, gy, p, local)
            });
        all_pending.clear();
        for (dc_idx, gx, gy, p, local) in refined {
            merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
            all_pending.push(p);
        }
    }

    // Phase 2: build adaptive DC entropy code from all DC + AC-metadata tokens.
    let mut dc_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(dim.num_dc_groups);
    let mut meta_tokens_per_group: Vec<Vec<Token>> = Vec::with_capacity(dim.num_dc_groups);
    for dc_data in &dc_datas {
        let dc_t = collect_dc_tokens(dc_data);
        let mt_t = collect_ac_metadata_tokens(dc_data);
        dc_tokens_per_group.push(dc_t);
        meta_tokens_per_group.push(mt_t);
    }
    // ANS-capable code for the DC+meta bundle (same gate as the plain-AC
    // bundle); write sites below branch on use_prefix_code.
    let dc_code_owned = crate::entropy::optimize_entropy_code_ac_streams(
        dc_tokens_per_group
            .iter()
            .map(Vec::as_slice)
            .chain(meta_tokens_per_group.iter().map(Vec::as_slice)),
        K_NUM_DC_CONTEXTS,
        &mut scratch.huffman_pool,
    );
    let dc_code = dc_code_owned.as_ref();

    let ac_num_contexts = K_NUM_AC_CONTEXTS + 1;

    // Per-pass aggregated tokens -> per-pass entropy code. Pass 0 (coarse) and
    // the residual pass(es) have very different token distributions, so a single
    // shared code is wasteful; each HfPass gets a code built from its own tokens.
    let ac_code_per_pass: Vec<crate::entropy::OwnedEntropyCode> = (0..num_passes)
        .map(|pass| {
            crate::entropy::optimize_entropy_code_ac_streams(
                all_pending
                    .iter()
                    .map(|pending| pending.tokens[pass].as_slice()),
                K_NUM_AC_CONTEXTS,
                &mut scratch.huffman_pool,
            )
        })
        .collect();

    // LZ77 path is single-pass only for now: it compresses one token stream per
    // group. Multi-pass uses the per-pass plain codes.
    let mut ac_lz_per_group: Vec<Vec<crate::enc_lz77_ac::AcLz>> = Vec::new();
    let ac_lz_code_owned;
    let use_lz77;
    if num_passes == 1 {
        ac_lz_per_group = Vec::with_capacity(all_pending.len());
        for pg in &all_pending {
            ac_lz_per_group.push(crate::enc_lz77_ac::lz77_compress_ac(&pg.tokens[0]));
        }
        ac_lz_code_owned = crate::enc_lz77_ac::build_ac_lz_code(
            &ac_lz_per_group,
            ac_num_contexts,
            &mut scratch.huffman_pool,
        );
        let lz_bits = crate::enc_lz77_ac::estimate_ac_lz_bits(
            &ac_lz_per_group,
            &ac_lz_code_owned,
            ac_num_contexts,
        );
        let plain_bits = crate::enc_lz77_ac::estimate_ac_plain_bits(
            all_pending
                .iter()
                .map(|pending| pending.tokens[0].as_slice()),
            &ac_code_per_pass[0],
        );
        // Require a real margin to cover the LZ77 header + distance-context cost.
        use_lz77 = lz_bits + 512 < plain_bits;
    } else {
        ac_lz_code_owned = crate::enc_lz77_ac::build_ac_lz_code(
            &ac_lz_per_group,
            ac_num_contexts,
            &mut scratch.huffman_pool,
        );
        use_lz77 = false;
    }

    // Phase 4: write DC global with adaptive DC code.
    if let VarDctFrameKind::Patched(references) = frame_kind {
        crate::enc_lossless::write_patch_dictionary(
            references,
            alpha.is_some(),
            scratch,
            &mut sections[0],
        );
    }
    write_dc_global(
        &distp,
        dim.num_dc_groups,
        &dc_code,
        alpha,
        dim.xsize,
        dim.ysize,
        scratch,
        &mut sections[0],
    );

    // Phase 5: write each DC group section with adaptive DC code.
    for (i, dc_data) in dc_datas.iter().enumerate() {
        let dc_group_idx = 1 + i;
        let w = &mut sections[dc_group_idx];
        w.write(2, 0); // extra_dc_precision
        w.write(4, 3); // use global tree, default wp, no transforms
        if dc_code.use_prefix_code {
            for t in &dc_tokens_per_group[i] {
                write_token(*t, &dc_code, w);
            }
        } else {
            crate::entropy::write_ans_tokens(
                &dc_tokens_per_group[i],
                dc_code.context_map,
                dc_code.ans_symbols,
                w,
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
                write_token(*t, &dc_code, w);
            }
        } else {
            crate::entropy::write_ans_tokens(
                &meta_tokens_per_group[i],
                dc_code.context_map,
                dc_code.ans_symbols,
                w,
            );
        }
    }

    // Phase 6: AC global. One HfPass per pass (each with its own code), or a
    // single HfPass carrying the LZ77 code in the single-pass case.
    write_ac_global(
        ctx.matrices,
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
    for (i, pg) in all_pending.iter().enumerate() {
        for (pass, pass_tokens) in pg.tokens.iter().enumerate() {
            let section_idx = 2 + dim.num_dc_groups + pass * dim.num_groups + pg.group_idx;
            let w = &mut sections[section_idx];
            if use_lz77 {
                for t in &ac_lz_per_group[i] {
                    crate::enc_lz77_ac::write_ac_lz(*t, &ac_lz_code_owned, ac_num_contexts, w);
                }
            } else {
                let code_ref = ac_code_per_pass[pass].as_ref();
                if code_ref.use_prefix_code {
                    for t in pass_tokens {
                        write_token(*t, &code_ref, w);
                    }
                } else {
                    // rANS: the whole group's tokens are encoded as one LIFO unit.
                    crate::entropy::write_ans_tokens(
                        pass_tokens,
                        code_ref.context_map,
                        code_ref.ans_symbols,
                        w,
                    );
                }
            }
        }
    }
    // Modular alpha: extra-channel modular data is decoded in the last pass
    // (the only pass whose modular sub-image shift range is set when num_ds=0,
    // jxl-frame lib.rs:101-108 / pass_group.rs:93). Write it into that section.
    if let Some(alpha_plane) = alpha {
        let last_pass = num_passes - 1;
        for image_gy in 0..dim.ysize_groups {
            for image_gx in 0..dim.xsize_groups {
                let group_x0 = image_gx * K_GROUP_DIM;
                let group_y0 = image_gy * K_GROUP_DIM;
                let group_xsize = K_GROUP_DIM.min(dim.xsize.saturating_sub(group_x0));
                let group_ysize = K_GROUP_DIM.min(dim.ysize.saturating_sub(group_y0));
                let abs_group_id = image_gy * dim.xsize_groups + image_gx;
                let ac_group_idx =
                    2 + dim.num_dc_groups + last_pass * dim.num_groups + abs_group_id;
                crate::modular::write_ac_group_alpha(
                    alpha_plane,
                    dim.xsize,
                    dim.ysize,
                    group_x0,
                    group_y0,
                    group_xsize,
                    group_ysize,
                    scratch,
                    &mut sections[ac_group_idx],
                );
            }
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
        );
    }

    if ctx.speed != Speed::Fast {
        crate::structure_aq::apply(
            &mut scratch.structure_corrections,
            opsin,
            &mut dc_data.raw_quant_field,
            dc_group_x0,
            dc_group_y0,
            distp.distance,
            ctx.dct8x8,
        );
    }

    // Compute the per-tile CfL slopes before strategy selection so candidate
    // costs use the same Y-to-X/Y-to-B subtraction as final coefficient coding.
    crate::enc_color_correlation::fill_cmap(
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
    dc_data.sub8_benefit = crate::enc_ac_strategy::fill_ac_strategy(
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
            let cost_with = meta_entropy_cost(&dc_data, scratch);
            for &(x, y, _) in &positions {
                dc_data.ac_strategy.set_first(x, y, STRATEGY_DCT);
            }
            let cost_without = meta_entropy_cost(&dc_data, scratch);
            let meta_delta = cost_with.saturating_sub(cost_without) as f32;
            if dc_data.sub8_benefit > crate::enc_ac_strategy::RD_LAMBDA * meta_delta {
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

/// Encode a single AC group: build its tile stripes, quantize and tokenize,
/// and place its DC coefficients into a returned group-local `quant_dc`
/// (origin-relative, merged by the caller). Reads `dc_data` read-only.
#[allow(clippy::too_many_arguments)]
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
    rdoq_prices: Option<&crate::entropy::FrozenTokenPrices>,
) -> (PendingAcGroup, Image3S) {
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
    let mut num_nzeros: Vec<Image3B> = (0..num_passes)
        .map(|_| Image3B::new(K_GROUP_DIM_IN_BLOCKS, K_GROUP_DIM_IN_BLOCKS))
        .collect();
    let mut tokens: Vec<Vec<Token>> = (0..num_passes)
        .map(|_| Vec::with_capacity(K_GROUP_DIM_IN_BLOCKS * K_GROUP_DIM_IN_BLOCKS * 4))
        .collect();

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
            &mut local_quant_dc,
            qorigin_x,
            qorigin_y,
            &mut num_nzeros,
            coeff_shifts,
            rdoq_prices,
            false,
            &mut tokens,
        );
    }

    (
        PendingAcGroup {
            group_idx: image_gy * dim.xsize_groups + image_gx,
            tokens,
        },
        local_quant_dc,
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
        DC_REFINE_HOLD, DC_REFINE_PEAK, DC_REFINE_RELEASE, compute_distance_params, dc_refinement,
        quant_dc,
    };

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
