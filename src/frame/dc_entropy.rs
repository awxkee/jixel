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

//! Select and refine the static, learned or MA-learned global DC entropy code.

use super::dc_tokens::{
    DC_PREDICTOR_GRADIENT, DC_PREDICTOR_WEIGHTED, DcTreeChoice, choose_dc_predictors,
    collect_dc_tokens_with_gradient, write_context_tree, write_tree_tokens,
};
use super::metadata::collect_ac_metadata_tokens;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::encoding_context::EncodingContext;
use crate::entropy::{EntropyCode, OwnedEntropyCode, Token, write_entropy_code};
use crate::static_entropy_codes::K_NUM_DC_CONTEXTS;

pub(super) struct DcEntropy {
    pub(super) tree: DcTreeChoice,
    pub(super) dc: Vec<Vec<Token>>,
    pub(super) meta: Vec<Vec<Token>>,
    pub(super) code: OwnedEntropyCode,
    pub(super) refinement: Option<crate::entropy::AnsRefinement>,
}

pub(super) fn prepare_dc_entropy(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    num_dc_groups: usize,
    epf_sharpness: i32,
    use_dc_frame: bool,
    distance: f32,
) -> DcEntropy {
    // Build adaptive DC entropy code from all DC + AC-metadata tokens.
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
                num_dc_groups,
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
                        write_context_tree(num_dc_groups, grad, &mut scratch.huffman_pool, &mut w);
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
                        dc_datas,
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
                            crate::entropy::refine_entropy_bundle(
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
                    crate::entropy::refine_entropy_bundle(
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
        crate::entropy::refine_entropy_bundle(
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

    DcEntropy {
        tree: dc_tree,
        dc: dc_tokens_per_group,
        meta: meta_tokens_per_group,
        code: dc_code_owned,
        refinement: ans_refinement,
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
