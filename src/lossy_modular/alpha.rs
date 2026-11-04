/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! A lossy extra channel (alpha, CMYK black) inside a VarDCT frame.
//!
//! Extra channels are always modular, so a lossy one is one squeezed,
//! quantized channel coded in the VarDCT frame's modular streams. It shares
//! the arm's luma step table and the forest's scale-safe tokenizer; unlike
//! the arm, it quantizes residuals closed-loop against the decoded averages.

use super::closed_loop::{Rounding, closed_h_squeeze, closed_v_squeeze};
use super::forest::{BUDGET_CAP, BUDGET_DIV, SAMPLE_TARGET};
use super::layout::XYB_SCALE;
use super::quantize::{SQUEEZE_QUALITY_FACTOR_XYB, SQUEEZE_XYB_QTABLE, quantize_channel_divide};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::Token;
use crate::lossless::kit::{
    GlobalPlaneTree, LZ77_MIN_SYMBOL, MA_MIN_NODE_SAMPLES, MA_PHOTO_CANDIDATES, MA_SPLIT_COST_BITS,
    PREDICTOR_GRADIENT, SAFE_PREDICTORS, build_balanced_tree_tokens_mul, build_lz_pixel_code_ans,
    channel_to_context, for_each_squeeze_group_crop, lz77_literals, push_leaf_mul, push_split,
    sample_channel_ma, scale_ma_value_props, selected_grad_pack_interior_fn,
    tokenize_channel_ma_mult, tokenize_plane_rows, write_lz_section,
    write_modular_transforms_squeeze_only, write_tree_lz77,
};
use crate::ma_tree::{LearnedTree, MaLearnParams, MaNode, MaSamples, learn_ma_tree};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Group sizes of the VarDCT frames this channel rides in.
const VARDCT_GROUP_DIM: usize = 256;

const VARDCT_LF_GROUP_DIM: usize = 2048;

/// One extra channel squeezed and quantized like the lossy-modular arm's luma
/// row, coded in the modular streams of a VarDCT frame. The frame's global
/// tree belongs to the VarDCT DC, so nonempty extra-channel streams carry
/// local trees, with quantization steps as residual multipliers.
pub(crate) struct ExtraSqueeze {
    pub(super) channels: Vec<crate::squeeze::Channel>,
    pub(super) quants: Vec<u32>,
    pub(super) steps: Vec<crate::squeeze::SqueezeStep>,
    /// Channels before `split` live in the global stream.
    pub(super) split: usize,
    pub(super) min_symbol: u32,
    /// Learn a context tree per stream (decoder MA-tree walk) instead of the
    /// fixed one-Gradient-leaf-per-channel tree.
    pub(super) learned_trees: bool,
    /// Code the fixed alternative through [`context_tree`]s rather than one
    /// Gradient leaf per channel.
    pub(super) context_trees: bool,
}

impl ExtraSqueeze {
    /// `distance` is in luma-quantizer units; `max` is the channel's largest
    /// sample. None when the plane is too small to squeeze.
    pub(crate) fn new(
        plane: &AlphaPlane,
        xsize: usize,
        ysize: usize,
        distance: f32,
        learned_trees: bool,
        context_trees: bool,
    ) -> Option<Self> {
        use crate::squeeze::{Channel, apply_step_forward, lossy_shared_steps};
        let steps = lossy_shared_steps(xsize, ysize, 1);
        if steps.is_empty() {
            return None;
        }
        let max = match plane {
            AlphaPlane::U8(_) => 255.0f32,
            AlphaPlane::U16 { bits, .. } => ((1u32 << *bits) - 1) as f32,
            AlphaPlane::F32(_) => return None,
        };
        let data = plane.to_i32_vec(xsize * ysize);
        let mut channels = vec![Channel {
            data,
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        }];
        // Keep each step's input: the residuals are quantized coarse to fine
        // against what the decoder reconstructs.
        let mut inputs = Vec::with_capacity(steps.len());
        for s in &steps {
            inputs.push(channels[0].data.clone());
            apply_step_forward(&mut channels, s);
        }
        // The arm's luma schedule is in units of 1/4096 of the sample range.
        let unit = max / XYB_SCALE[1];
        let quants: Vec<u32> = channels
            .iter()
            .map(|ch| {
                let mut shift = (ch.hshift + ch.vshift).min(16);
                if shift > 0 {
                    shift -= 1;
                }
                let q = 0.25
                    * distance
                    * SQUEEZE_QUALITY_FACTOR_XYB
                    * SQUEEZE_XYB_QTABLE[0][shift as usize]
                    * unit;
                (q as i32).max(1) as u32
            })
            .collect();
        quantize_channel_divide(&mut channels[0], quants[0] as i32);
        let mut decoded: Vec<i32> = channels[0]
            .data
            .iter()
            .map(|&v| v * quants[0] as i32)
            .collect();
        // In place, step k's residual ends up right after the average, behind
        // the residuals of the coarser steps.
        for (k, step) in steps.iter().enumerate().rev() {
            let c = steps.len() - k;
            let (w, h) = input_dims(&steps, k, xsize, ysize);
            let quantize = if step.horizontal {
                closed_h_squeeze
            } else {
                closed_v_squeeze
            };
            let (residuals, out) = quantize(
                &inputs[k],
                w,
                h,
                &decoded,
                quants[c] as i32,
                RESIDUAL_ROUNDING,
            );
            debug_assert_eq!(residuals.len(), channels[c].data.len());
            channels[c].data = residuals;
            decoded = out;
        }
        let (split, min_symbol) = extra_channel_layout(&channels);
        Some(Self {
            channels,
            quants,
            steps,
            split,
            min_symbol,
            learned_trees,
            context_trees,
        })
    }

    /// Earliest retained prefix whose remaining average and residuals are
    /// exact. Folding this tail changes only the entropy representation.
    pub(crate) fn exact_tail_start(&self) -> Option<usize> {
        if self.quants[0] != 1 {
            return None;
        }
        let exact = self.quants[1..].iter().take_while(|&&q| q == 1).count();
        (exact > 0).then_some(self.steps.len() - exact)
    }

    /// Invert an exact coarse tail, leaving every retained quantized band
    /// unchanged. Invert the coded coefficients themselves, so even an
    /// unclamped reconstruction agrees with the original pyramid.
    pub(crate) fn without_exact_tail(&self, retained_steps: usize) -> Option<Self> {
        if retained_steps >= self.steps.len() || retained_steps < self.exact_tail_start()? {
            return None;
        }
        let removed = self.steps.len() - retained_steps;
        let mut channels = self.channels[..=removed].to_vec();
        for step in self.steps[retained_steps..].iter().rev() {
            crate::squeeze::apply_step_inverse(&mut channels, step);
        }
        debug_assert_eq!(channels.len(), 1);
        channels.extend_from_slice(&self.channels[removed + 1..]);
        let quants = std::iter::once(1)
            .chain(self.quants[removed + 1..].iter().copied())
            .collect();
        let (split, min_symbol) = extra_channel_layout(&channels);
        Some(Self {
            channels,
            quants,
            steps: self.steps[..retained_steps].to_vec(),
            split,
            min_symbol,
            learned_trees: self.learned_trees,
            context_trees: self.context_trees,
        })
    }

    /// For a candidate derived by `without_exact_tail`, all changed channels
    /// stay global when its new average fits there. The retained group crops
    /// then have identical data, quantizers and within-stream slot indices.
    pub(crate) fn exact_tail_groups_unchanged(&self, candidate: &Self) -> bool {
        candidate.split > 0 && self.min_symbol == candidate.min_symbol
    }

    pub(crate) fn exact_tail_ac_unchanged(&self, candidate: &Self) -> bool {
        // Once the new average has minimum shift three, it and every removed
        // coarse band live outside the AC streams. Those crops stay identical.
        self.min_symbol == candidate.min_symbol
            && (candidate.split > 0
                || candidate.channels[0]
                    .hshift
                    .min(candidate.channels[0].vshift)
                    >= 3)
    }

    /// The plane a decoder reconstructs, before clamping to the sample range.
    pub(crate) fn reconstruct(&self) -> Vec<i32> {
        use crate::squeeze::{Channel, apply_step_inverse};
        let mut channels: Vec<Channel> = self
            .channels
            .iter()
            .zip(&self.quants)
            .map(|(ch, &q)| Channel {
                data: ch.data.iter().map(|&v| v * q as i32).collect(),
                w: ch.w,
                h: ch.h,
                hshift: ch.hshift,
                vshift: ch.vshift,
            })
            .collect();
        for s in self.steps.iter().rev() {
            apply_step_inverse(&mut channels, s);
        }
        debug_assert_eq!(channels.len(), 1);
        channels.swap_remove(0).data
    }

    /// The GlobalModular image: its header (with the squeeze transform) and
    /// the channels that fit one group.
    pub(crate) fn write_global(
        &self,
        global_code_ans: bool,
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        let crops: Vec<(usize, usize, usize, usize, usize)> = (0..self.split)
            .map(|c| (c, 0, 0, self.channels[c].w, self.channels[c].h))
            .collect();
        // Even an empty global subimage needs a valid entropy header. Reuse
        // the frame's global tree when all samples live in LF/AC groups.
        w.write(1, u64::from(self.split == 0));
        w.write(1, 1); // wp_header all_default
        if self.steps.is_empty() {
            // A Squeeze entry with zero steps requests the decoder's default
            // pyramid. A completely folded tail needs no transform instead.
            w.write(2, 0);
        } else {
            write_modular_transforms_squeeze_only(&self.steps, w);
        }
        if self.split == 0 {
            if global_code_ans {
                w.write(32, u64::from(crate::entropy::ANS_SIGNATURE << 16));
            }
        } else {
            self.write_crops(&crops, scratch, w);
        }
    }

    /// The channels of LF group (`gx`, `gy`): minimum shift 3 and up.
    pub(crate) fn write_lf_group(
        &self,
        gx: usize,
        gy: usize,
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        self.write_group(VARDCT_LF_GROUP_DIM, gx, gy, 3, 1000, scratch, w);
    }

    /// The channels of AC group (`gx`, `gy`): minimum shift 0..=2.
    pub(crate) fn write_ac_group(
        &self,
        gx: usize,
        gy: usize,
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        self.write_group(VARDCT_GROUP_DIM, gx, gy, 0, 2, scratch, w);
    }

    /// The AC-group channels coded through the frame's global tree: the fixed
    /// context forest (one subtree per within-group slot, leaves carrying the
    /// slot's step) and each AC group's pixel tokens under it. None when the
    /// groups' slots do not hold the same channels, so one forest cannot
    /// carry every group's steps. A forest learned over every group saves
    /// up to 3 % more of the file but its deeper walk decodes up to 65 %
    /// slower, so the fixed contexts serve every speed.
    pub(crate) fn global_ac_forest(
        &self,
        num_groups: usize,
        xsize_groups: usize,
        pool: &ThreadPool,
    ) -> Option<GlobalPlaneTree> {
        let crops: Vec<Vec<_>> = (0..num_groups)
            .map(|g| {
                let mut crops = Vec::new();
                for_each_squeeze_group_crop(
                    &self.channels,
                    self.split,
                    VARDCT_GROUP_DIM,
                    g % xsize_groups,
                    g / xsize_groups,
                    0,
                    2,
                    |_within, c, rx0, ry0, rw, rh| crops.push((c, rx0, ry0, rw, rh)),
                );
                crops
            })
            .collect();
        let slots: Vec<usize> = crops.first()?.iter().map(|&(c, ..)| c).collect();
        if slots.is_empty()
            || crops
                .iter()
                .any(|g| !g.iter().map(|&(c, ..)| c).eq(slots.iter().copied()))
        {
            return None;
        }
        let trees: Vec<LearnedTree> = slots.iter().map(|&c| context_tree(c)).collect();
        let mults: Vec<u32> = slots.iter().map(|&c| self.quants[c]).collect();
        let (tree_tokens, leaf_ctx, bases, _) = emit_slot_forest(&trees, &mults);
        let crop_tokens = pool.map(crops.len(), |g| {
            let mut tokens = Vec::new();
            for (slot, &(c, rx0, ry0, rw, rh)) in crops[g].iter().enumerate() {
                let ch = &self.channels[c];
                let (data, stride) = (&ch.data, ch.w);
                tokenize_channel_ma_mult(
                    |y| &data[(ry0 + y) * stride + rx0..][..rw],
                    rw,
                    rh,
                    slot as u32,
                    0,
                    self.quants[c] as i32,
                    &trees[slot],
                    &leaf_ctx,
                    bases[slot],
                    &[],
                    &mut tokens,
                );
            }
            tokens
        });
        Some(GlobalPlaneTree {
            tree_tokens,
            crop_tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_group(
        &self,
        gdim: usize,
        gx: usize,
        gy: usize,
        minsh: i32,
        maxsh: i32,
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        let mut crops = Vec::new();
        for_each_squeeze_group_crop(
            &self.channels,
            self.split,
            gdim,
            gx,
            gy,
            minsh,
            maxsh,
            |_within, c, rx0, ry0, rw, rh| crops.push((c, rx0, ry0, rw, rh)),
        );
        // A group with no channel in its shift range reads nothing.
        if crops.is_empty() {
            return;
        }
        w.write(1, 0); // use_global_tree = 0
        w.write(1, 1); // wp_header all_default
        w.write(2, 0); // no transforms
        self.write_crops(&crops, scratch, w);
    }

    /// Local tree + entropy code + tokens of one stream's channel crops: the
    /// fixed tree, or with learned trees whichever of the two is smaller.
    pub(super) fn write_crops(
        &self,
        crops: &[(usize, usize, usize, usize, usize)],
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        if crops.is_empty() {
            return;
        }
        let mut fixed = BitWriter::new();
        self.write_crops_fixed(crops, scratch, &mut fixed);
        if self.learned_trees {
            let mut learned = BitWriter::new();
            self.write_crops_learned(crops, scratch, &mut learned);
            if learned.bits_written() < fixed.bits_written() {
                w.append(&learned);
                return;
            }
        }
        w.append(&fixed);
    }

    /// One learned subtree per channel slot, chained on property 0 with the
    /// slot's step stamped on its leaves (same scheme as the lossy-modular
    /// arm, restricted to scale-equivariant predictors).
    pub(super) fn write_crops_learned(
        &self,
        crops: &[(usize, usize, usize, usize, usize)],
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        let pool = ThreadPool::new(1);
        let n = crops.len();
        let mut trees: Vec<LearnedTree> = Vec::with_capacity(n);
        for (slot, &(c, rx0, ry0, rw, rh)) in crops.iter().enumerate() {
            let ch = &self.channels[c];
            let (data, stride) = (&ch.data, ch.w);
            let px = rw * rh;
            let mut step = px.div_ceil(SAMPLE_TARGET).max(1);
            if step > 1 && step.is_multiple_of(2) {
                step += 1;
            }
            let mut samples = MaSamples::with_capacity(px / step + 1);
            sample_channel_ma(
                |y| &data[(ry0 + y) * stride + rx0..][..rw],
                rw,
                rh,
                slot as u32,
                0,
                &[],
                WpParams::DEFAULT,
                false,
                step,
                |props, tok, _| samples.push(props, tok),
            );
            scale_ma_value_props(&mut samples, self.quants[c] as i32);
            let params = |max_leaves: usize| MaLearnParams {
                alphabet: self.min_symbol as usize,
                max_leaves,
                split_cost_bits: MA_SPLIT_COST_BITS / step as f32,
                min_node: MA_MIN_NODE_SAMPLES,
                allow_wp: false,
                allowed_preds: SAFE_PREDICTORS,
                side_preds: 1,
                max_candidates: MA_PHOTO_CANDIDATES,
                modal_offsets: false,
                prune_leaf_bits: 0.0,
            };
            let budget = (px / BUDGET_DIV).clamp(1, BUDGET_CAP);
            let tree = if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
                learn_ma_tree(&samples, params(1), &pool, scratch)
            } else {
                let tree = learn_ma_tree(&samples, params(budget), &pool, scratch);
                let leaves = tree.nodes.len().div_ceil(2);
                let overhead = tree.nodes.len() as f64 * 10.0 + leaves as f64 * 200.0;
                if tree.est_bits * step as f64 + overhead < tree.flat_bits * step as f64 {
                    tree
                } else {
                    learn_ma_tree(&samples, params(1), &pool, scratch)
                }
            };
            trees.push(tree);
        }
        self.write_crops_with_trees(crops, &trees, scratch, w);
    }

    /// Code the crops through one tree per slot (chained by
    /// `emit_slot_forest`), each leaf carrying its slot's step.
    fn write_crops_with_trees(
        &self,
        crops: &[(usize, usize, usize, usize, usize)],
        trees: &[LearnedTree],
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        let mults: Vec<u32> = crops.iter().map(|&(c, ..)| self.quants[c]).collect();
        let (tree_tokens, leaf_ctx, bases, num_ctx) = emit_slot_forest(trees, &mults);
        let mut tokens: Vec<Token> = Vec::new();
        for (slot, &(c, rx0, ry0, rw, rh)) in crops.iter().enumerate() {
            let ch = &self.channels[c];
            let (data, stride) = (&ch.data, ch.w);
            tokenize_channel_ma_mult(
                |y| &data[(ry0 + y) * stride + rx0..][..rw],
                rw,
                rh,
                slot as u32,
                0,
                self.quants[c] as i32,
                &trees[slot],
                &leaf_ctx,
                bases[slot],
                &[],
                &mut tokens,
            );
        }
        let lz = lz77_literals(&tokens);
        let code = build_lz_pixel_code_ans(
            false,
            std::iter::once(lz.as_slice()),
            num_ctx as usize,
            self.min_symbol,
            true,
            true,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        write_tree_lz77(
            &tree_tokens,
            &code,
            self.min_symbol,
            &mut scratch.huffman_pool,
            w,
        );
        write_lz_section(&lz, num_ctx, &code, self.min_symbol, w);
    }

    pub(super) fn write_crops_fixed(
        &self,
        crops: &[(usize, usize, usize, usize, usize)],
        scratch: &mut CoderScratch,
        w: &mut BitWriter,
    ) {
        if self.context_trees {
            let trees: Vec<LearnedTree> = crops.iter().map(|&(c, ..)| context_tree(c)).collect();
            self.write_crops_with_trees(crops, &trees, scratch, w);
            return;
        }
        let n = crops.len();
        let grad_pack_fn = selected_grad_pack_interior_fn();
        let mut tokens: Vec<Token> = Vec::new();
        for (slot, &(c, rx0, ry0, rw, rh)) in crops.iter().enumerate() {
            let ch = &self.channels[c];
            let data = &ch.data;
            let stride = ch.w;
            tokenize_plane_rows(
                channel_to_context(slot, n),
                |y| &data[(ry0 + y) * stride + rx0..][..rw],
                rw,
                rh,
                PREDICTOR_GRADIENT,
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
        }
        let lz = lz77_literals(&tokens);
        let code = build_lz_pixel_code_ans(
            false,
            std::iter::once(lz.as_slice()),
            n,
            self.min_symbol,
            true,
            false,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        let leaves: Vec<(u32, u32)> = crops
            .iter()
            .map(|&(c, ..)| (PREDICTOR_GRADIENT, self.quants[c]))
            .collect();
        write_tree_lz77(
            &build_balanced_tree_tokens_mul(&leaves),
            &code,
            self.min_symbol,
            &mut scratch.huffman_pool,
            w,
        );
        write_lz_section(&lz, n as u32, &code, self.min_symbol, w);
    }
}

/// Group placement and alphabet bound of this representation. Folding the
/// average can move it into LF/AC groups or widen high-bit-depth symbols.
fn extra_channel_layout(channels: &[crate::squeeze::Channel]) -> (usize, u32) {
    let split = channels
        .iter()
        .position(|c| c.w > VARDCT_GROUP_DIM || c.h > VARDCT_GROUP_DIM)
        .unwrap_or(channels.len());
    let max_abs = channels
        .iter()
        .flat_map(|ch| ch.data.iter())
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    let value_bits = 33 - (2 * max_abs).max(1).leading_zeros();
    let min_symbol = if value_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * value_bits + 24
    };
    (split, min_symbol)
}

/// Dimensions of the channel that squeeze step `k` splits.
fn input_dims(
    steps: &[crate::squeeze::SqueezeStep],
    k: usize,
    mut w: usize,
    mut h: usize,
) -> (usize, usize) {
    for s in &steps[..k] {
        if s.horizontal {
            w = w.div_ceil(2);
        } else {
            h = h.div_ceil(2);
        }
    }
    (w, h)
}

/// How the squeezed residuals round: rate-aware at lambda 0.08 (8 alphas,
/// SS2 better on all 8: Slow -0.30 % SS2 / -0.07 % BA BD, Fast -0.25 /
/// -0.11, over the earlier flat 0.15 deadzone; per-level deadzones trade).
const RESIDUAL_ROUNDING: Rounding = Rounding::RateAware(0.08);

/// The fixed context tree of squeezed channel `c`: Zero prediction of the
/// residual channels (Gradient for the coarsest average) in four contexts,
/// by whether the north and the west neighbor are nonzero. Most squeezed
/// residuals are zero away from edges, so this captures most of what a
/// learned tree finds at the cost of the fixed tree.
fn context_tree(c: usize) -> LearnedTree {
    let pred = if c == 0 { PREDICTOR_GRADIENT } else { 0 };
    let leaf = MaNode::Leaf { pred, offset: 0 };
    let split = |prop, gt, le| MaNode::Split {
        prop,
        val: 0,
        gt,
        le,
    };
    LearnedTree {
        nodes: vec![
            split(4, 1, 2), // |N| > 0
            split(5, 3, 4), // |W| > 0
            split(5, 5, 6),
            leaf,
            leaf,
            leaf,
            leaf,
        ],
        est_bits: 0.0,
        flat_bits: 0.0,
    }
}

/// Chain per-slot learned subtrees on property 0 (the within-stream channel
/// index) and BFS-emit them, each leaf carrying its slot's multiplier.
/// Returns the tree tokens, the arena-indexed leaf contexts, each slot's
/// arena base and the context count.
fn emit_slot_forest(trees: &[LearnedTree], mults: &[u32]) -> (Vec<Token>, Vec<u32>, Vec<u32>, u32) {
    use std::collections::VecDeque;
    let mut nodes: Vec<MaNode> = Vec::new();
    let mut mult_of: Vec<u32> = Vec::new();
    let mut bases = vec![0u32; trees.len()];
    let mut root = 0u32;
    for (slot, tree) in trees.iter().enumerate() {
        let base = nodes.len() as u32;
        bases[slot] = base;
        for n in &tree.nodes {
            nodes.push(match *n {
                MaNode::Split { prop, val, gt, le } => MaNode::Split {
                    prop,
                    val,
                    gt: gt + base,
                    le: le + base,
                },
                leaf @ MaNode::Leaf { .. } => leaf,
            });
            mult_of.push(mults[slot]);
        }
        if slot == 0 {
            root = base;
        } else {
            nodes.push(MaNode::Split {
                prop: 0,
                val: (slot - 1) as i32,
                gt: base,
                le: root,
            });
            mult_of.push(1);
            root = (nodes.len() - 1) as u32;
        }
    }
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
    (tokens, leaf_ctx, bases, ctx)
}
