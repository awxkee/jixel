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

//! MA trees over channels stored divided by their step, with the step as
//! the leaves' residual multiplier: the predictors that stay exact under it,
//! full-scale property rescaling, the tokenizer and BFS forest emission.

use super::ma_sampling::{MaRefPlane, walk_channel_ma};
use super::predictor::predictor_value;
use super::tree::{push_leaf_mul, push_split};
use crate::entropy::{Token, pack_signed};
use crate::ma_tree::{LearnedTree, MaNode, MaSamples};
use crate::weighted_predictor::WpParams;

/// Decoder predictors that are exactly scale-equivariant over an exact
/// division (safe under a leaf multiplier): Zero, Left, Top, Select, Gradient
/// and the plain neighbor predictors. Weighted and the averages are not.
pub(crate) const SAFE_PREDICTORS: u16 =
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 9);

/// Rescale the value-derived MA properties (libjxl ids 4..=14) by `q`:
/// sampling ran on the divided channel, but the decoder computes properties on
/// the full-scale reconstruction.
pub(crate) fn scale_ma_value_props(samples: &mut MaSamples, q: i32) {
    if q <= 1 {
        return;
    }
    samples.props.map_rows(|p| {
        for v in &mut p[4..=14] {
            *v *= q;
        }
    });
}

/// Tokenize a (divided) channel rectangle through a learned tree whose splits
/// live in the full-scale property domain. `base` is the tag's node offset in
/// the merged forest arena; `leaf_ctx` is arena-indexed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tokenize_channel_ma_mult<'a>(
    get_row: impl Fn(usize) -> &'a [i32],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    q: i32,
    tree: &LearnedTree,
    leaf_ctx: &[u32],
    base: u32,
    refs: &[MaRefPlane<'a>],
    out: &mut Vec<Token>,
) {
    walk_channel_ma(
        get_row,
        gw,
        gh,
        chan,
        stream,
        refs,
        WpParams::DEFAULT,
        false,
        |_x, _y, v, p, n, _wp_pred| {
            let (node, pred) = if q > 1 {
                let mut sp = *p;
                for s in &mut sp[4..=14] {
                    *s *= q;
                }
                tree.lookup(&sp)
            } else {
                tree.lookup(p)
            };
            let pv = predictor_value(pred, n, 0);
            out.push(Token::new(
                leaf_ctx[(base + node) as usize],
                pack_signed((v - pv) as i32),
            ));
        },
    );
}

/// BFS emission of a merged forest from `root` (matches libjxl's FIFO tree
/// decode), stamping each leaf with its multiplier.
pub(crate) fn emit_forest_bfs(
    nodes: &[MaNode],
    mult_of: &[u32],
    root: u32,
    tag_base: Vec<u32>,
) -> (Vec<Token>, Vec<u32>, Vec<u32>, u32) {
    use std::collections::VecDeque;
    let mut tokens = Vec::new();
    let mut leaf_ctx = vec![u32::MAX; nodes.len()];
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(root);
    let mut ctx = 0u32;
    while let Some(i) = queue.pop_front() {
        match nodes[i as usize] {
            MaNode::Split { prop, val, gt, le } => {
                push_split(&mut tokens, prop as u32, val);
                queue.push_back(gt);
                queue.push_back(le);
            }
            MaNode::Leaf { pred, .. } => {
                push_leaf_mul(&mut tokens, pred, mult_of[i as usize]);
                leaf_ctx[i as usize] = ctx;
                ctx += 1;
            }
        }
    }
    (tokens, leaf_ctx, tag_base, ctx)
}
