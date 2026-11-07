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

//! Modular tree token serialization and flat-tree entropy writing.

use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{OwnedEntropyCode, Token, pack_signed, write_entropy_code};
use crate::ma_tree::{LearnedTree, MaNode};

pub(super) const TREE_CTX_SPLIT_VAL: u32 = 0;

pub(super) const TREE_CTX_PROPERTY: u32 = 1;

pub(super) const TREE_CTX_PREDICTOR: u32 = 2;

pub(super) const TREE_CTX_OFFSET: u32 = 3;

pub(super) const TREE_CTX_MULTIPLIER_LOG: u32 = 4;

pub(super) const TREE_CTX_MULTIPLIER_BITS: u32 = 5;

pub(super) const NUM_TREE_CONTEXTS: usize = 6;

/// BFS-emit a learned tree (matches libjxl's FIFO tree decode). Returns
/// (tree tokens, context id per node index — leaves only, context count).
pub(super) fn emit_learned_tree(tree: &LearnedTree) -> (Vec<Token>, Vec<u32>, u32) {
    use std::collections::VecDeque;
    let mut tokens = Vec::new();
    let mut leaf_ctx = vec![u32::MAX; tree.nodes.len()];
    let mut q: VecDeque<u32> = VecDeque::new();
    q.push_back(0);
    let mut ctx = 0u32;
    while let Some(i) = q.pop_front() {
        match tree.nodes[i as usize] {
            MaNode::Split { prop, val, gt, le } => {
                push_split(&mut tokens, prop as u32, val);
                q.push_back(gt);
                q.push_back(le);
            }
            MaNode::Leaf { pred, offset } => {
                push_leaf_offset(&mut tokens, pred, offset);
                leaf_ctx[i as usize] = ctx;
                ctx += 1;
            }
        }
    }
    (tokens, leaf_ctx, ctx)
}

// ---------------------------------------------------------------------------
// Tree writing (balanced N-leaf, Gradient predictor).
//
// libjxl reads modular trees in BFS order: each node is one PROPERTY token
// (=0 for leaf, =prop+1 for split) optionally followed by a SPLIT_VAL token
// or 4 leaf-data tokens (predictor, offset, multiplier-log, multiplier-bits).
// All channels share the Gradient predictor with offset 0 and multiplier 1.
// ---------------------------------------------------------------------------

pub(crate) fn push_split(out: &mut Vec<Token>, property: u32, split_val: i32) {
    out.push(Token::new(TREE_CTX_PROPERTY, property + 1));
    out.push(Token::new(TREE_CTX_SPLIT_VAL, pack_signed(split_val)));
}

pub(super) fn push_leaf(out: &mut Vec<Token>, predictor: u32) {
    push_leaf_mul(out, predictor, 1);
}

/// Leaf with a predictor offset: the decoder reconstructs
/// `pred + offset + residual`.
pub(super) fn push_leaf_offset(out: &mut Vec<Token>, predictor: u32, offset: i32) {
    out.push(Token::new(TREE_CTX_PROPERTY, 0));
    out.push(Token::new(TREE_CTX_PREDICTOR, predictor));
    out.push(Token::new(TREE_CTX_OFFSET, pack_signed(offset)));
    out.push(Token::new(TREE_CTX_MULTIPLIER_LOG, 0));
    out.push(Token::new(TREE_CTX_MULTIPLIER_BITS, 0));
}

/// `build_balanced_tree_tokens` with a predictor offset per channel.
pub(super) fn build_balanced_tree_tokens_offsets(
    predictors: &[u32],
    offsets: &[i32],
) -> Vec<Token> {
    debug_assert_eq!(predictors.len(), offsets.len());
    let mut tokens = build_balanced_tree_tokens(predictors);
    // Leaves are emitted in BFS order chan(N-1), ..., chan0.
    for (token, &offset) in tokens
        .iter_mut()
        .filter(|token| token.context == TREE_CTX_OFFSET)
        .zip(offsets.iter().rev())
    {
        token.value = pack_signed(offset);
    }
    tokens
}

/// Leaf with a residual multiplier: the decoder reconstructs
/// `pred + multiplier * residual` (multiplier = (mul_bits+1) << mul_log).
pub(crate) fn push_leaf_mul(out: &mut Vec<Token>, predictor: u32, multiplier: u32) {
    debug_assert!(multiplier >= 1);
    let mul_log = multiplier.trailing_zeros();
    let mul_bits = (multiplier >> mul_log) - 1;
    out.push(Token::new(TREE_CTX_PROPERTY, 0));
    out.push(Token::new(TREE_CTX_PREDICTOR, predictor));
    out.push(Token::new(TREE_CTX_OFFSET, pack_signed(0)));
    out.push(Token::new(TREE_CTX_MULTIPLIER_LOG, mul_log));
    out.push(Token::new(TREE_CTX_MULTIPLIER_BITS, mul_bits));
}

/// Balanced channel-split tree whose leaves carry per-channel residual
/// multipliers (the lossy-modular quantization steps).
pub(crate) fn build_balanced_tree_tokens_mul(leaves: &[(u32, u32)]) -> Vec<Token> {
    let n_leaves = leaves.len();
    let mut t = Vec::new();
    match n_leaves {
        1 => push_leaf_mul(&mut t, leaves[0].0, leaves[0].1),
        2 => {
            push_split(&mut t, 0, 0);
            push_leaf_mul(&mut t, leaves[1].0, leaves[1].1);
            push_leaf_mul(&mut t, leaves[0].0, leaves[0].1);
        }
        _ => {
            // Right-leaning chain, BFS leaves chan(N-1)..chan0 (same shape as
            // `build_balanced_tree_tokens`).
            for k in (1..=(n_leaves - 2)).rev() {
                push_split(&mut t, 0, k as i32);
                push_leaf_mul(&mut t, leaves[k + 1].0, leaves[k + 1].1);
            }
            push_split(&mut t, 0, 0);
            push_leaf_mul(&mut t, leaves[1].0, leaves[1].1);
            push_leaf_mul(&mut t, leaves[0].0, leaves[0].1);
        }
    }
    t
}

/// Build a balanced binary tree over `n_leaves` leaves splitting on property 0
/// (the channel index after RCT).  BFS leaf order is chan N-1, ..., chan 0.
pub(super) fn build_balanced_tree_tokens(predictors: &[u32]) -> Vec<Token> {
    let n_leaves = predictors.len();
    let mut t = Vec::new();
    // Leaves are emitted in BFS order chan(N-1), ..., chan0 (matches fjxl), so
    // index them with predictors[chan].
    match n_leaves {
        1 => push_leaf(&mut t, predictors[0]),
        2 => {
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        3 => {
            push_split(&mut t, 0, 1);
            push_leaf(&mut t, predictors[2]); // chan 2
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        4 => {
            push_split(&mut t, 0, 1);
            push_split(&mut t, 0, 2);
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[3]); // chan 3
            push_leaf(&mut t, predictors[2]); // chan 2
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        _ => {
            // N>=5: right-leaning chain. BFS emission yields leaves in order
            // chan(N-1)..chan0 -> contexts 0..N-1, matching channel_to_context.
            // Peels off the highest channel at each split: split(0,k) sends
            // channel>k (== k+1) to a leaf, else continue.
            for k in (1..=(n_leaves - 2)).rev() {
                push_split(&mut t, 0, k as i32);
                push_leaf(&mut t, predictors[k + 1]);
            }
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]);
            push_leaf(&mut t, predictors[0]);
        }
    }
    t
}

// ---------------------------------------------------------------------------
// f32 lossless (v1): non-negative float, RGB, no alpha. The 32-bit float bits
// are reinterpreted as int32 channels (matching libjxl float_to_int for
// bits==32: a raw memcpy), then coded as a modular image with NO RCT and NO
// LZ77. LZ77 must be off because float residual tokens can reach 127, leaving
// no room below the 128-symbol alphabet for LZ77 length symbols. Per-channel
// gradient/WP prediction and the channel-split tree are reused unchanged.
// ---------------------------------------------------------------------------

/// Write a channel-split MA tree (no LZ77 in the tree) followed by the pixel
/// entropy code header with LZ77 DISABLED. Mirrors the tree-writing convention
/// but flips the pixel stream's LZ77 flag off.
pub(super) fn write_tree_and_pixel_code_nolz(
    tree_tokens: &[Token],
    pixel_code: &OwnedEntropyCode,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    // The tree's own entropy code (no LZ77 in the tree), then its tokens.
    crate::entropy::write_raced_token_stream(
        tree_tokens,
        NUM_TREE_CONTEXTS,
        &mut scratch.huffman_pool,
        w,
    );
    // Pixel entropy code: LZ77 DISABLED (1 bit = 0), then the code itself.
    w.write(1, 0);
    write_entropy_code(&pixel_code.as_ref(), &mut scratch.huffman_pool, w);
}
