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
use crate::entropy::{EntropyCode, OwnedEntropyCode, Token, f_log2, write_entropy_code};
use crate::static_entropy_codes::K_NUM_DC_CONTEXTS;
use crate::weighted_predictor::WpParams;

pub(crate) struct DcEntropy {
    pub(crate) tree: DcTreeChoice,
    pub(crate) dc: Vec<Vec<Token>>,
    pub(crate) meta: Vec<Vec<Token>>,
    pub(crate) code: OwnedEntropyCode,
    pub(crate) refinement: Option<crate::entropy::AnsRefinement>,
    /// Weighted-predictor parameters of the DC streams (their GroupHeaders).
    pub(crate) wp_params: WpParams,
    /// The MA-learned DC tree, for a re-run under other predictor parameters.
    pub(crate) ma_tree: Option<crate::ma_tree::LearnedTree>,
}

impl DcEntropy {
    /// Bits of the DC-global tree and code headers plus the DC and metadata
    /// payload estimate: what one choice of predictor parameters costs.
    pub(crate) fn bits(&self, num_dc_groups: usize, scratch: &mut CoderScratch) -> u64 {
        let mut w = BitWriter::new();
        crate::weighted_predictor::write_wp_header(self.wp_params, &mut w);
        match &self.tree {
            DcTreeChoice::Static(grad) => {
                write_context_tree(num_dc_groups, grad, &mut scratch.huffman_pool, &mut w);
            }
            DcTreeChoice::Learned(tokens) => {
                write_tree_tokens(tokens, &mut scratch.huffman_pool, &mut w);
            }
        }
        write_entropy_code(&self.code.as_ref(), &mut scratch.huffman_pool, &mut w);
        w.bits_written() as u64
            + crate::entropy::estimate_ac_plain_bits(
                self.dc
                    .iter()
                    .map(Vec::as_slice)
                    .chain(self.meta.iter().map(Vec::as_slice)),
                &self.code,
            )
    }
}

/// Proxy bits of the static DC arm under `params`: the DC tokens with the
/// per-leaf predictor flips, priced as order-0 Shannon bits per context
/// plus the raw bits. The metadata streams do not depend on the parameters
/// and are left out. Ranks candidates against each other; the exact
/// pipeline prices the finalist. (Pricing through the frame's MA-learned
/// tree instead was tried: the tree learned under the default parameters
/// favors them and the fit stalls.)
pub(crate) fn dc_wp_proxy_bits(
    pool: &crate::thread_pool::ThreadPool,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    params: WpParams,
) -> f64 {
    let groups: Vec<(Vec<Token>, Vec<Token>)> =
        pool.steal_map(scratch, dc_datas.len(), |i, _scratch| {
            collect_dc_tokens_with_gradient(
                &dc_datas[i],
                &DC_PREDICTOR_WEIGHTED,
                &mut Vec::new(),
                false,
                true,
                params,
            )
        });
    let (wp, grad): (Vec<_>, Vec<_>) = groups.into_iter().unzip();
    let flips = choose_dc_predictors(&wp, &grad, &mut scratch.dc_predictor);
    const SYMBOLS: usize = 64;
    let mut counts = vec![0u64; K_NUM_DC_CONTEXTS * SYMBOLS];
    let mut raw_bits = 0u64;
    for (wp, grad) in wp.iter().zip(&grad) {
        for (w, g) in wp.iter().zip(grad) {
            let t = if flips[w.context as usize] { g } else { w };
            let (sym, nbits, _) = crate::entropy::uint_encode(t.value);
            counts[t.context as usize * SYMBOLS + (sym as usize).min(SYMBOLS - 1)] += 1;
            raw_bits += u64::from(nbits);
        }
    }
    let mut bits = raw_bits as f64;
    for hist in counts.chunks(SYMBOLS) {
        let total: u64 = hist.iter().sum();
        if total == 0 {
            continue;
        }
        let total = total as f64;
        bits += hist
            .iter()
            .filter(|&&c| c != 0)
            .map(|&c| c as f64 * f_log2(total / c as f64))
            .sum::<f64>();
    }
    bits
}

/// Coordinate descent over the eleven weighted-predictor parameters from
/// `start`, one unit per step, priced by [`dc_wp_proxy_bits`]. The start
/// matters: the default parameters are a proxy minimum on most frames, so
/// the caller passes its exact preset winner.
pub(crate) fn fit_dc_wp_params(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    start: WpParams,
) -> WpParams {
    const ROUNDS: usize = 3;
    const P_MAX: i64 = 31;
    const W_MAX: u32 = 15;
    let mut best = start;
    let mut best_bits = dc_wp_proxy_bits(&ctx.thread_pool, scratch, dc_datas, best);
    for _ in 0..ROUNDS {
        let mut improved = false;
        for coord in 0..11 {
            for delta in [-1i64, 1] {
                let mut candidate = best;
                if coord < 7 {
                    let v = candidate.p[coord] + delta;
                    if !(0..=P_MAX).contains(&v) {
                        continue;
                    }
                    candidate.p[coord] = v;
                } else {
                    let v = candidate.w[coord - 7] as i64 + delta;
                    if !(0..=i64::from(W_MAX)).contains(&v) {
                        continue;
                    }
                    candidate.w[coord - 7] = v as u32;
                }
                let bits = dc_wp_proxy_bits(&ctx.thread_pool, scratch, dc_datas, candidate);
                if bits < best_bits {
                    best = candidate;
                    best_bits = bits;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    best
}

/// `wp_params` are the DC streams' weighted-predictor parameters; `ma_tree`
/// reuses an MA-learned DC tree from an earlier call (its splits stay valid
/// under other parameters, only the tokens change).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_dc_entropy(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    num_dc_groups: usize,
    epf_sharpness: i32,
    use_dc_frame: bool,
    distance: f32,
    wp_params: WpParams,
    ma_tree: Option<&crate::ma_tree::LearnedTree>,
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
                    wp_params,
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
    let mut learned_ma_tree: Option<crate::ma_tree::LearnedTree> = None;
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
            let ma_budget = if ctx.speed.effort().ma_dc_tree {
                Some(MaDcBudget::FULL)
            } else if ctx.speed.effort().ma_dc_tree_lite {
                Some(MaDcBudget::LITE)
            } else {
                None
            };
            let ma_arm = ma_budget
                .and_then(|budget| {
                    learn_ma_dc_arm(
                        &ctx.thread_pool,
                        scratch,
                        dc_datas,
                        &meta_tokens_per_group,
                        &meta_props_per_group,
                        wp_params,
                        ma_tree,
                        budget,
                    )
                })
                .and_then(|(variants, meta_tokens, num_contexts, tree)| {
                    learned_ma_tree = Some(tree);
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
    } else if effort.ans_fast_refine && !effort.ans_prefix_race {
        Some(crate::entropy::AnsRefinement::FastNoRace)
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
        wp_params,
        ma_tree: learned_ma_tree,
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

/// The quantized DC planes of every group in the modular channel order
/// (Y, X, B) the MA-learned DC arm codes them in.
fn dc_planes(dc_datas: &[DcGroupData]) -> Vec<crate::lossless::DcPlanes> {
    dc_datas
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
            let dims = |c: usize| (q.plane(c).xsize(), q.plane(c).ysize());
            crate::lossless::DcPlanes {
                planes: [plane(1), plane(0), plane(2)],
                dims: [dims(1), dims(0), dims(2)],
            }
        })
        .collect()
}

/// Learn the DC side of the global tree with the lossless MA learner (full
/// property set, every predictor) and tokenize the DC groups and the
/// metadata streams through the grafted tree. Returns the candidate
/// (DC tokens, serialized tree) pairs — without and, when any leaf has one,
/// with median leaf offsets — the metadata tokens and the context count.
/// Learning budget of the MA-learned DC arm.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaDcBudget {
    /// DC-side sample budget; larger DC images are sampled with an odd stride.
    pub(crate) target_samples: usize,
    pub(crate) max_leaves: usize,
    pub(crate) max_candidates: usize,
    pub(crate) side_preds: usize,
    /// Also price the tree with median leaf offsets.
    pub(crate) offsets_variant: bool,
}

impl MaDcBudget {
    pub(crate) const FULL: Self = Self {
        target_samples: MA_DC_TARGET_SAMPLES,
        max_leaves: MA_DC_MAX_LEAVES,
        max_candidates: 32,
        side_preds: 4,
        offsets_variant: true,
    };
    /// A quarter of the samples, 64 leaves, 8 split candidates, one side
    /// predictor, no offsets variant: three quarters of the full budget's
    /// bytes on JPEG transcodes at a third of its time.
    pub(crate) const LITE: Self = Self {
        target_samples: 1 << 17,
        max_leaves: 64,
        max_candidates: 8,
        side_preds: 1,
        offsets_variant: false,
    };
}

#[allow(clippy::type_complexity)]
pub(crate) fn learn_ma_dc_arm(
    pool: &crate::thread_pool::ThreadPool,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    meta_tokens: &[Vec<Token>],
    meta_props: &[Vec<crate::dc_tree::DcProp>],
    wp_params: WpParams,
    reuse_tree: Option<&crate::ma_tree::LearnedTree>,
    budget: MaDcBudget,
) -> Option<(
    Vec<(Vec<Vec<Token>>, Vec<Token>)>,
    Vec<Vec<Token>>,
    usize,
    crate::ma_tree::LearnedTree,
)> {
    let groups = dc_planes(dc_datas);
    let tree = match reuse_tree {
        Some(tree) => tree.clone(),
        None => {
            let total: usize = groups.iter().map(|g| g.len()).sum();
            let mut stride = total.div_ceil(budget.target_samples).max(1);
            if stride > 1 && stride.is_multiple_of(2) {
                stride += 1;
            }
            let samples = crate::lossless::sample_dc_planes(&groups, stride, wp_params);
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
                max_leaves: budget.max_leaves,
                split_cost_bits: MA_DC_SPLIT_COST_BITS / stride as f32,
                min_node: MA_DC_MIN_NODE_SAMPLES,
                allow_wp: true,
                allowed_preds: u16::MAX,
                side_preds: budget.side_preds,
                max_candidates: budget.max_candidates,
                modal_offsets: false,
                prune_leaf_bits: 0.0,
            };
            crate::ma_tree::learn_ma_tree(&samples, params, pool, scratch)
        }
    };
    let mut node_offsets = vec![0i32; tree.nodes.len()];
    let mut grafted = crate::dc_tree::graft_ma_dc_tree(
        dc_datas.len(),
        &tree,
        &node_offsets,
        meta_tokens,
        meta_props,
    );
    let mut dc_tokens = pool.steal_map(scratch, groups.len(), |i, _scratch| {
        let g = &groups[i];
        let mut out = Vec::with_capacity(g.len());
        crate::lossless::tokenize_dc_planes(g, i, &tree, &grafted.leaf_ctx, wp_params, &mut out);
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
    if budget.offsets_variant && ctx_offsets.iter().any(|&o| o != 0) {
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
    Some((variants, meta_tokens, num_contexts, tree))
}
