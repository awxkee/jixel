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

//! Lossy modular learned-tree forests and residual multipliers.

use super::super::ma_sampling::{MaRefPlane, walk_channel_ma};
use super::super::predictor::predictor_value;
use super::super::tree::{push_leaf_mul, push_split};
use crate::entropy::{Token, pack_signed};
use crate::ma_tree::{LearnedTree, MaNode, MaSamples};
use crate::weighted_predictor::WpParams;

/// Per-tag MA sample budget; more samples plateaued (<=0.6%) in the 2026-09
/// knob sweeps, as did larger leaf budgets and split-candidate counts.
pub(super) const LM_SAMPLE_TARGET: usize = 1 << 16;

/// Learned-subtree leaf budget: pixels/div, clamped.
pub(super) const LM_BUDGET_DIV: usize = 1024;

pub(super) const LM_BUDGET_CAP: usize = 256;

pub(super) const LM_SAFE_PRED_MASK: u16 =
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 9);

/// Rescale the value-derived MA properties (libjxl ids 4..=14) by `q`:
/// sampling ran on the divided channel, but the decoder computes properties on
/// the full-scale reconstruction.
pub(super) fn scale_ma_value_props(samples: &mut MaSamples, q: i32) {
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
pub(super) fn tokenize_channel_ma_mult<'a>(
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

/// Merge the per-tag learned subtrees under the stream/channel routing
/// skeleton and BFS-emit the whole forest, stamping every leaf with its tag's
/// residual multiplier. The skeleton splits on the static stream-id property:
/// 0 = LfGlobal stream, [1+ndc, 2ndc] = DC-group streams, and AC-group
/// streams above both; each side hangs a channel / slot chain. Tags are
/// ordered [global channels][DC slots][AC slots]. Returns the tree tokens,
/// the arena-indexed leaf context table, each tag's arena base, and the
/// context count.
pub(super) fn emit_lm_forest(
    trees: &[LearnedTree],
    tag_mults: &[u32],
    split: usize,
    n_dc_slots: usize,
    num_dc_groups: usize,
) -> (Vec<Token>, Vec<u32>, Vec<u32>, u32) {
    let n_tags = trees.len();
    let n_ac_slots = n_tags - split - n_dc_slots;
    debug_assert!(split > 0 && (n_ac_slots > 0 || n_dc_slots == 0));
    let mut nodes: Vec<MaNode> = Vec::new();
    let mut mult_of: Vec<u32> = Vec::new();
    let mut tag_base = vec![0u32; n_tags];

    let append_tree = |nodes: &mut Vec<MaNode>, mult_of: &mut Vec<u32>, t: usize| -> u32 {
        let base = nodes.len() as u32;
        for n in &trees[t].nodes {
            nodes.push(match *n {
                MaNode::Split { prop, val, gt, le } => MaNode::Split {
                    prop,
                    val,
                    gt: gt + base,
                    le: le + base,
                },
                leaf @ MaNode::Leaf { .. } => leaf,
            });
            mult_of.push(tag_mults[t]);
        }
        base
    };
    // Per-side chains over property 0 (the decoder-visible channel value:
    // channel index in the global stream, within-group slot in group streams).
    let build_chain = |nodes: &mut Vec<MaNode>,
                       mult_of: &mut Vec<u32>,
                       tag_base: &mut [u32],
                       lo: usize,
                       n: usize|
     -> u32 {
        let base0 = append_tree(nodes, mult_of, lo);
        tag_base[lo] = base0;
        let mut acc = base0;
        for k in 1..n {
            let sub = append_tree(nodes, mult_of, lo + k);
            tag_base[lo + k] = sub;
            nodes.push(MaNode::Split {
                prop: 0,
                val: (k - 1) as i32,
                gt: sub,
                le: acc,
            });
            mult_of.push(1);
            acc = (nodes.len() - 1) as u32;
        }
        acc
    };
    let global_root = build_chain(&mut nodes, &mut mult_of, &mut tag_base, 0, split);
    if n_ac_slots == 0 {
        // Single-group frame: everything lives in the global stream.
        return emit_forest_bfs(&nodes, &mult_of, global_root, tag_base);
    }
    let ac_root = build_chain(
        &mut nodes,
        &mut mult_of,
        &mut tag_base,
        split + n_dc_slots,
        n_ac_slots,
    );
    let low_root = if n_dc_slots > 0 {
        // DC streams exist: separate them from the global stream below the
        // AC threshold.
        let dc_root = build_chain(&mut nodes, &mut mult_of, &mut tag_base, split, n_dc_slots);
        nodes.push(MaNode::Split {
            prop: 1,
            val: 0,
            gt: dc_root,
            le: global_root,
        });
        mult_of.push(1);
        (nodes.len() - 1) as u32
    } else {
        global_root
    };
    // AC stream ids start at 1 + 3*ndc + kNumQuantTables, always above
    // 2*ndc; global (0) and DC ([1+ndc, 2ndc]) stay at or below it.
    nodes.push(MaNode::Split {
        prop: 1,
        val: (2 * num_dc_groups) as i32,
        gt: ac_root,
        le: low_root,
    });
    mult_of.push(1);
    let root = (nodes.len() - 1) as u32;
    emit_forest_bfs(&nodes, &mult_of, root, tag_base)
}

/// BFS emission of a merged forest from `root` (matches libjxl's FIFO tree
/// decode), stamping each leaf with its multiplier.
pub(super) fn emit_forest_bfs(
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
