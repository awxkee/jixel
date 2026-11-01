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

//! Global learned trees for modular planes.

use super::super::lz77::LZ77_MIN_SYMBOL;
use super::super::ma_multiplier::{
    SAFE_PREDICTORS, emit_forest_bfs, scale_ma_value_props, tokenize_channel_ma_mult,
};
use super::super::ma_sampling::sample_channel_ma;
use super::super::ma_source::{MA_MIN_NODE_SAMPLES, MA_PHOTO_CANDIDATES};
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;
use crate::ma_tree::{MaLearnParams, MaSamples, NUM_MA_PREDS, PRED_WEIGHTED, learn_ma_tree};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// One plane's subtree for the frame's global tree, learned over every crop
/// of the plane (their stream ids feed property 1).
pub(crate) struct GlobalPlaneTree {
    /// BFS tree tokens, leaves stamped with the plane's common factor.
    pub(crate) tree_tokens: Vec<Token>,
    /// Per crop, its pixel tokens (contexts are the subtree's leaves).
    pub(crate) crop_tokens: Vec<Vec<Token>>,
}

/// One rectangle of a plane in its own stream: x0, y0, width, height and the
/// decoder's stream id.
pub(crate) type StreamCrop = (usize, usize, usize, usize, i32);

const GLOBAL_PLANE_SAMPLE_TARGET: usize = 1 << 20;

const GLOBAL_PLANE_MAX_LEAVES: usize = 1024;

const GLOBAL_PLANE_SPLIT_COST_BITS: f32 = 25.0;

const GLOBAL_PLANE_SIDE_PREDS: usize = 4;

/// Every predictor but the Weighted one, for planes coded at full scale.
const PLANE_PRED_MASK: u16 = ((1 << NUM_MA_PREDS) - 1) & !(1 << PRED_WEIGHTED);

/// Learn one tree over every crop of `data` (`stride` wide, channel `chan`
/// of its streams; divided in place by its common factor) and tokenize each
/// crop through it. `palette`, a row coded as channel 0 of the first crop's
/// stream ahead of the plane, is tokenized through the same tree.
pub(crate) fn learn_global_plane_tree(
    data: &mut [i32],
    stride: usize,
    chan: u32,
    palette: Option<&[i32]>,
    crops: &[StreamCrop],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> GlobalPlaneTree {
    let (q, min_symbol) = divide_common_factor(data);
    debug_assert!(
        q == 1 || palette.is_none(),
        "palette indices share no factor"
    );
    let data = &*data;
    let px: usize = crops.iter().map(|&(_, _, w, h, _)| w * h).sum();
    let mut step = px.div_ceil(GLOBAL_PLANE_SAMPLE_TARGET).max(1);
    if step > 1 && step.is_multiple_of(2) {
        step += 1;
    }
    let mut samples = MaSamples::with_capacity(px / step + crops.len());
    for &(x0, y0, w, h, stream) in crops {
        sample_channel_ma(
            |y| &data[(y0 + y) * stride + x0..][..w],
            w,
            h,
            chan,
            stream,
            &[],
            WpParams::DEFAULT,
            false,
            step,
            |props, tok, _| samples.push(props, tok),
        );
    }
    scale_ma_value_props(&mut samples, q as i32);
    let params = MaLearnParams {
        alphabet: min_symbol as usize,
        max_leaves: GLOBAL_PLANE_MAX_LEAVES,
        split_cost_bits: GLOBAL_PLANE_SPLIT_COST_BITS / step as f32,
        min_node: MA_MIN_NODE_SAMPLES,
        allow_wp: false,
        allowed_preds: if q > 1 {
            SAFE_PREDICTORS
        } else {
            PLANE_PRED_MASK
        },
        side_preds: GLOBAL_PLANE_SIDE_PREDS,
        max_candidates: MA_PHOTO_CANDIDATES,
        modal_offsets: false,
        prune_leaf_bits: 0.0,
    };
    let tree = learn_ma_tree(&samples, params, pool, scratch);
    drop(samples);
    let mults = vec![q; tree.nodes.len()];
    let (tree_tokens, leaf_ctx, ..) = emit_forest_bfs(&tree.nodes, &mults, 0, Vec::new());
    let crop_tokens = pool.steal_map(scratch, crops.len(), |i, _scratch| {
        let (x0, y0, w, h, stream) = crops[i];
        let mut tokens = Vec::with_capacity(w * h);
        if i == 0
            && let Some(row) = palette
        {
            tokenize_channel_ma_mult(
                |_| row,
                row.len(),
                1,
                0,
                stream,
                1,
                &tree,
                &leaf_ctx,
                0,
                &[],
                &mut tokens,
            );
        }
        tokenize_channel_ma_mult(
            |y| &data[(y0 + y) * stride + x0..][..w],
            w,
            h,
            chan,
            stream,
            q as i32,
            &tree,
            &leaf_ctx,
            0,
            &[],
            &mut tokens,
        );
        tokens
    });
    GlobalPlaneTree {
        tree_tokens,
        crop_tokens,
    }
}

/// Divide every value by their common factor; returns the factor (the leaf
/// multiplier) and the coded alphabet's minimum symbol count.
pub(crate) fn divide_common_factor(data: &mut [i32]) -> (u32, u32) {
    let q = data.iter().fold(0, |g, &v| gcd(g, v.unsigned_abs())).max(1);
    if q > 1 {
        data.iter_mut().for_each(|v| *v /= q as i32);
    }
    let max_abs = data.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    let value_bits = 33 - (2 * max_abs).max(1).leading_zeros();
    let min_symbol = if value_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * value_bits + 24
    };
    (q, min_symbol)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
