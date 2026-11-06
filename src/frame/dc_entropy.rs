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
    pub(crate) ma_tree: Option<MaDcTrees>,
    /// Context id per node of the written DC tree and the predictor offset
    /// per node (empty without the MA arm).
    pub(crate) ma_leaf_ctx: Vec<u32>,
    pub(crate) ma_offsets: Vec<i32>,
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
    ma_tree: Option<&MaDcTrees>,
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

    // Price an arm end to end: serialized tree + entropy-code header
    // (measured by writing them) plus the payload estimate.
    let payload =
        |dc: &[Vec<Token>], meta: &[Vec<Token>], code: &crate::entropy::OwnedEntropyCode| {
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
    // The margin absorbs the small estimator noise so near-ties keep the
    // battle-tested static tree.
    const LEARNED_TREE_MARGIN_BITS: u64 = 64;

    // Arm C: the DC side learned over the full modular property set and
    // every predictor, as the lossless path does. Above a few DC samples it
    // has won every image measured, so the static and learned arms are then
    // not built at all; small images still race all three.
    let mut learned_ma_tree: Option<MaDcTrees> = None;
    let mut ma_leaf_map: (Vec<u32>, Vec<i32>) = (Vec::new(), Vec::new());
    let ma_budget = if effort.ma_dc_tree {
        Some(MaDcBudget::FULL)
    } else if effort.ma_dc_tree_lite {
        Some(MaDcBudget::LITE)
    } else {
        None
    };
    let ma_arm = if use_dc_frame {
        None
    } else {
        ma_budget
            .and_then(|budget| {
                learn_ma_dc_arm(
                    &ctx.thread_pool,
                    scratch,
                    dc_datas,
                    &meta_tokens_per_group,
                    &meta_props_per_group,
                    epf_sharpness,
                    wp_params,
                    ma_tree,
                    budget,
                )
            })
            .and_then(|(arms, trees)| {
                let price = |arm: MaDcArm, scratch: &mut CoderScratch| {
                    let MaDcArm {
                        dc: dc_tokens,
                        meta: meta_tokens,
                        tree: tree_tokens,
                        num_contexts,
                        dc_leaf_ctx,
                        offsets,
                        meta_learned,
                    } = arm;
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
                    (
                        cost,
                        dc_tokens,
                        meta_tokens,
                        code,
                        tree,
                        dc_leaf_ctx,
                        offsets,
                        meta_learned,
                    )
                };
                // Arms price independently, one worker each, kept in arm
                // order so ties resolve as in a serial pass.
                let price_arms = |arms: Vec<MaDcArm>, scratch: &mut CoderScratch| {
                    let mut slots: Vec<_> = arms.into_iter().map(|arm| (Some(arm), None)).collect();
                    ctx.thread_pool
                        .steal_for_each_mut(scratch, &mut slots, |_, slot, scratch| {
                            slot.1 = slot.0.take().map(|arm| price(arm, scratch));
                        });
                    slots
                        .into_iter()
                        .filter_map(|slot| slot.1)
                        .collect::<Vec<_>>()
                };
                let mut best = price_arms(arms, scratch).into_iter().min_by_key(|v| v.0)?;
                let mut trees = trees;
                // One EM step: leaves re-pick their predictor under the
                // clustered histograms, and the winning arm's configuration
                // (offsets, metadata side) is assembled and priced again.
                if effort.ma_dc_repick()
                    && let Some(repicked) = repick_leaf_predictors(
                        dc_datas, &trees, &best.5, &best.6, &best.3, wp_params,
                    )
                {
                    let with_offsets = best.6.iter().any(|&o| o != 0);
                    let arms = assemble_ma_arms(
                        &ctx.thread_pool,
                        scratch,
                        dc_datas,
                        &meta_tokens_per_group,
                        &meta_props_per_group,
                        epf_sharpness,
                        wp_params,
                        &repicked,
                        with_offsets,
                    );
                    let arms = arms
                        .into_iter()
                        .filter(|arm| {
                            arm.meta_learned == best.7
                                && arm.offsets.iter().any(|&o| o != 0) == with_offsets
                        })
                        .collect();
                    if let Some(better) = price_arms(arms, scratch).into_iter().min_by_key(|v| v.0)
                        && better.0 < best.0
                    {
                        best = better;
                        trees = repicked;
                    }
                }
                learned_ma_tree = Some(trees);
                ma_leaf_map = (best.5, best.6);
                Some((best.0, best.1, best.2, best.3, best.4))
            })
    };

    // `dc_code_refined`: the arm pricing already used ANS-refined codes, so
    // the winner needs no second refinement.
    let dc_samples: usize = dc_datas
        .iter()
        .map(|d| {
            (0..3)
                .map(|c| d.quant_dc.plane(c).xsize() * d.quant_dc.plane(c).ysize())
                .sum::<usize>()
        })
        .sum();
    let (dc_tokens_per_group, meta_tokens_per_group, mut dc_code_owned, dc_tree, dc_code_refined) =
        match ma_arm {
            Some((_, dc_tokens, meta_tokens, code, tree)) if dc_samples >= MA_ARM_ALONE_SAMPLES => {
                (dc_tokens, meta_tokens, code, tree, true)
            }
            ma_arm => {
                if ma_arm.is_none() {
                    ma_leaf_map = (Vec::new(), Vec::new());
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
                let best = if !effort.dc_learned_tree || use_dc_frame {
                    (
                        dc_tokens_static,
                        meta_tokens_per_group,
                        code_static,
                        tree_static,
                    )
                } else {
                    // Arm B: a per-image learned tree over WP error and channel, with
                    // its own per-leaf predictor choice. Leaf renumbering shifts the
                    // metadata contexts, so those tokens are remapped alongside.
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
                    let tree_learned = DcTreeChoice::Learned(learned.tokens);
                    let cost_static = header(&tree_static, &code_static.as_ref(), scratch)
                        + payload(&dc_tokens_static, &meta_tokens_per_group, &code_static);
                    let cost_learned = header(&tree_learned, &code_learned.as_ref(), scratch)
                        + payload(&dc_tokens_learned, &meta_tokens_learned, &code_learned);
                    if cost_learned + LEARNED_TREE_MARGIN_BITS < cost_static {
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
                    }
                };
                match ma_arm {
                    Some((cost, dc_tokens, meta_tokens, code, tree)) => {
                        // Refine the incumbent too, so both arms are priced on
                        // the codes the final refinement would write.
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
                            ma_leaf_map = (Vec::new(), Vec::new());
                            (best_dc, best_meta, best_code, best_tree, true)
                        }
                    }
                    None => (best.0, best.1, best.2, best.3, false),
                }
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
        ma_leaf_ctx: ma_leaf_map.0,
        ma_offsets: ma_leaf_map.1,
    }
}

/// DC-side sample budget of the MA-learned DC arm; larger DC images are
/// sampled with an odd stride.
const MA_DC_TARGET_SAMPLES: usize = 1 << 19;
/// Leaf budget of the MA-learned DC side (the metadata side adds at most
/// ~25 more; context ids stay below 256).
/// Binds on large images only: small DC images stop splitting well before
/// it, and the coder clusters the contexts to 128 histograms regardless
/// (a 256-cluster coder was tried: no gain).
const MA_DC_MAX_LEAVES: usize = 400;
/// Estimated header bits of one more DC leaf: its tree tokens, context-map
/// entry and a share of a histogram.
const MA_DC_SPLIT_COST_BITS: f32 = 50.0;
const MA_DC_MIN_NODE_SAMPLES: usize = 64;

/// The quantized DC planes of every group in the modular channel order
/// (Y, X, B) the MA-learned DC arm codes them in.
fn dc_planes(dc_datas: &[DcGroupData]) -> Vec<crate::lossless::DcPlanes> {
    dc_datas
        .iter()
        .enumerate()
        .map(|(group, dc)| {
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
                planes: vec![plane(1), plane(0), plane(2)],
                dims: vec![dims(1), dims(0), dims(2)],
                stream: 1 + group as i32,
            }
        })
        .collect()
}

/// The AC-metadata modular image of every DC group as planes: the CfL tile
/// maps, the block-info channel (first blocks wide; transform type over
/// QF − 1) and the EPF sharpness plane, with the decoder's stream id.
fn meta_planes(dc_datas: &[DcGroupData], epf_sharpness: i32) -> Vec<crate::lossless::DcPlanes> {
    let num_dc_groups = dc_datas.len();
    dc_datas
        .iter()
        .enumerate()
        .map(|(group, dc)| {
            let tile = |m: &crate::image::ImageSB| -> Vec<i32> {
                m.as_slice().iter().map(|&v| i32::from(v)).collect()
            };
            let (xt, yt) = (dc.ytox_map.xsize(), dc.ytox_map.ysize());
            let (xb, yb) = (dc.ac_strategy.xsize(), dc.ac_strategy.ysize());
            let first = dc.ac_strategy.count_first_blocks();
            let mut info = Vec::with_capacity(2 * first);
            let mut qfs = Vec::with_capacity(first);
            for y in 0..yb {
                let row = dc.raw_quant_field.row(y);
                for x in 0..xb {
                    if dc.ac_strategy.is_first_block(x, y) {
                        info.push(dc.ac_strategy.strategy_code(x, y) as i32);
                        qfs.push(i32::from(row[x]) - 1);
                    }
                }
            }
            info.extend(qfs);
            crate::lossless::DcPlanes {
                planes: vec![
                    tile(&dc.ytox_map),
                    tile(&dc.ytob_map),
                    info,
                    vec![epf_sharpness; xb * yb],
                ],
                dims: vec![(xt, yt), (xt, yt), (first, 2), (xb, yb)],
                stream: (1 + 2 * num_dc_groups + group) as i32,
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
    /// Also learn the metadata side over the modular property set and race
    /// it against the static refined side.
    pub(crate) learn_meta: bool,
}

impl MaDcBudget {
    /// Slow and up. Over the 160-leaf / 4-side-predictor / static-metadata
    /// budget on 14 references at d1/2/4: every side predictor scored per
    /// split side −0.09 % (large images −0.5 %, Kodak ±0.1 %), the 400-leaf
    /// cap −0.2 % (binds on large images only), the learned metadata side
    /// −0.06 % (city/screenshots −0.1..−0.26 %); together −0.34 % at ×1.13
    /// single-thread Slow. 64 split candidates were +8 % time for −0.03 %.
    pub(crate) const FULL: Self = Self {
        target_samples: MA_DC_TARGET_SAMPLES,
        max_leaves: MA_DC_MAX_LEAVES,
        max_candidates: 32,
        side_preds: 14,
        offsets_variant: true,
        learn_meta: true,
    };
    /// Fast and Medium: a quarter of the samples, 128 leaves, 8 split
    /// candidates, four side predictors, no offsets variant, learned
    /// metadata side. Over the 64-leaf / 1-side / static-metadata budget:
    /// Fast −0.63 % at ×1.15, Medium −0.46 % at ×1.08 (33/42 and 31/42 wins).
    pub(crate) const LITE: Self = Self {
        target_samples: 1 << 17,
        max_leaves: 128,
        max_candidates: 8,
        side_preds: 4,
        offsets_variant: false,
        learn_meta: true,
    };
}

/// The MA-learned trees of a DC arm, for a re-run under other predictor
/// parameters: the DC side and, when learned, the metadata side.
#[derive(Clone)]
pub(crate) struct MaDcTrees {
    pub(crate) dc: crate::ma_tree::LearnedTree,
    pub(crate) meta: Option<crate::ma_tree::LearnedTree>,
}

/// One priced candidate of the MA-learned DC arm: DC and metadata tokens
/// numbered by the serialized tree's contexts.
pub(crate) struct MaDcArm {
    pub(crate) dc: Vec<Vec<Token>>,
    pub(crate) meta: Vec<Vec<Token>>,
    pub(crate) tree: Vec<Token>,
    pub(crate) num_contexts: usize,
    /// Context id per node of the DC tree (meaningful for its leaves).
    pub(crate) dc_leaf_ctx: Vec<u32>,
    /// Predictor offset per node of the DC tree.
    pub(crate) offsets: Vec<i32>,
    /// Whether the metadata side is the learned one.
    pub(crate) meta_learned: bool,
}

const MA_META_MAX_LEAVES: usize = 64;
/// DC samples from which the MA arm alone is coded: below it the static and
/// learned arms still win now and then (abstract 365x547 +0.5 % without
/// them at d4), above it they never have.
const MA_ARM_ALONE_SAMPLES: usize = 32_768;

/// Contexts the DC and metadata sides may take together: the coder
/// clusters them anyway, and the context map grows with each.
const MA_MAX_TREE_CONTEXTS: usize = 512;

/// Learn one MA tree over the planes of every group, sampling large images
/// with an odd stride down to the budget.
fn learn_planes(
    pool: &crate::thread_pool::ThreadPool,
    scratch: &mut CoderScratch,
    planes: &[crate::lossless::DcPlanes],
    budget: MaDcBudget,
    max_leaves: usize,
    wp_params: WpParams,
) -> Option<crate::ma_tree::LearnedTree> {
    let total: usize = planes.iter().map(|g| g.len()).sum();
    let mut stride = total.div_ceil(budget.target_samples).max(1);
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    let samples = crate::lossless::sample_dc_planes(pool, planes, stride, wp_params);
    if samples.len() < 4 * MA_DC_MIN_NODE_SAMPLES {
        return None;
    }
    let params = ma_dc_params(&samples, budget, max_leaves, stride);
    Some(crate::ma_tree::learn_ma_tree(
        &samples, params, pool, scratch,
    ))
}

/// Learner parameters of the DC arm.
fn ma_dc_params(
    samples: &crate::ma_tree::MaSamples,
    budget: MaDcBudget,
    max_leaves: usize,
    stride: usize,
) -> crate::ma_tree::MaLearnParams {
    let alphabet = samples
        .tok
        .iter()
        .flat_map(|t| t.iter())
        .copied()
        .max()
        .unwrap_or(0) as usize
        + 1;
    crate::ma_tree::MaLearnParams {
        alphabet,
        max_leaves,
        split_cost_bits: MA_DC_SPLIT_COST_BITS / stride as f32,
        min_node: MA_DC_MIN_NODE_SAMPLES,
        allow_wp: true,
        allowed_preds: u16::MAX,
        side_preds: budget.side_preds,
        max_candidates: budget.max_candidates,
        modal_offsets: false,
        prune_leaf_bits: 0.0,
    }
}

/// Learn the metadata side as one subtree per part — the CfL maps, the
/// strategy row, the QF row and the EPF plane — each on its own samples,
/// routed by fixed channel/row splits so no leaf mixes streams whose
/// statistics differ by construction. `cap` bounds the leaves in all.
fn learn_meta_parts(
    pool: &crate::thread_pool::ThreadPool,
    scratch: &mut CoderScratch,
    planes: &[crate::lossless::DcPlanes],
    budget: MaDcBudget,
    cap: usize,
) -> Option<crate::ma_tree::LearnedTree> {
    use crate::ma_tree::{LearnedTree, MaNode};
    let total: usize = planes.iter().map(|g| g.len()).sum();
    let mut stride = total.div_ceil(budget.target_samples).max(1);
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    let samples = crate::lossless::sample_dc_planes(pool, planes, stride, WpParams::DEFAULT);
    // Part of each sample: 0 CfL, 1 strategy row, 2 QF row, 3 EPF.
    let mut idx: [Vec<u32>; 4] = Default::default();
    for (i, row) in samples.props.iter().enumerate() {
        let part = match (row[0], row[2]) {
            (0 | 1, _) => 0,
            (2, 0) => 1,
            (2, _) => 2,
            _ => 3,
        };
        idx[part].push(i as u32);
    }
    // Leaves per part: the block-info rows carry the bits.
    let caps = [
        (cap / 8).max(2),
        (cap * 2 / 5).max(4),
        (cap * 2 / 5).max(4),
        2,
    ];
    let mut parts: Vec<Vec<MaNode>> = Vec::with_capacity(4);
    let mut est_bits = 0.0;
    let mut flat_bits = 0.0;
    for (part, part_idx) in idx.into_iter().enumerate() {
        if part_idx.len() < MA_DC_MIN_NODE_SAMPLES {
            parts.push(vec![MaNode::Leaf { pred: 0, offset: 0 }]);
            continue;
        }
        let params = ma_dc_params(&samples, budget, caps[part], stride);
        let tree = crate::ma_tree::learn_ma_tree_indexed(&samples, part_idx, params, pool, scratch);
        est_bits += tree.est_bits;
        flat_bits += tree.flat_bits;
        parts.push(tree.nodes);
    }
    // Routing prefix: channel > 2 → EPF; channel > 1 → (y > 0 → QF, else
    // strategy); else CfL.
    let mut nodes = vec![
        MaNode::Split {
            prop: 0,
            val: 2,
            gt: 0,
            le: 1,
        },
        MaNode::Split {
            prop: 0,
            val: 1,
            gt: 2,
            le: 0,
        },
        MaNode::Split {
            prop: 2,
            val: 0,
            gt: 0,
            le: 0,
        },
    ];
    let mut roots = [0u32; 4];
    for (part, part_nodes) in parts.into_iter().enumerate() {
        let base = nodes.len() as u32;
        roots[part] = base;
        nodes.extend(part_nodes.into_iter().map(|n| match n {
            MaNode::Split { prop, val, gt, le } => MaNode::Split {
                prop,
                val,
                gt: gt + base,
                le: le + base,
            },
            leaf => leaf,
        }));
    }
    nodes[0] = MaNode::Split {
        prop: 0,
        val: 2,
        gt: roots[3],
        le: 1,
    };
    nodes[1] = MaNode::Split {
        prop: 0,
        val: 1,
        gt: 2,
        le: roots[0],
    };
    nodes[2] = MaNode::Split {
        prop: 2,
        val: 0,
        gt: roots[2],
        le: roots[1],
    };
    Some(LearnedTree {
        nodes,
        est_bits,
        flat_bits,
    })
}

/// Learn the DC side of the global tree with the lossless MA learner (full
/// property set, every predictor) and, on request, the metadata side too;
/// tokenize the DC groups and the metadata streams through every grafted
/// tree. Returns the candidate arms — without and, when any leaf has one,
/// with median DC leaf offsets, each with the static refined metadata side
/// and with the learned one — and the trees for a re-run.
#[allow(clippy::too_many_arguments)]
pub(crate) fn learn_ma_dc_arm(
    pool: &crate::thread_pool::ThreadPool,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    meta_tokens: &[Vec<Token>],
    meta_props: &[Vec<crate::dc_tree::DcProp>],
    epf_sharpness: i32,
    wp_params: WpParams,
    reuse: Option<&MaDcTrees>,
    budget: MaDcBudget,
) -> Option<(Vec<MaDcArm>, MaDcTrees)> {
    let groups = dc_planes(dc_datas);

    let trees = match reuse {
        Some(trees) => trees.clone(),
        None => {
            let dc = learn_planes(pool, scratch, &groups, budget, budget.max_leaves, wp_params)?;
            let meta = if budget.learn_meta {
                let dc_leaves = dc
                    .nodes
                    .iter()
                    .filter(|n| matches!(n, crate::ma_tree::MaNode::Leaf { .. }))
                    .count();
                let cap = MA_META_MAX_LEAVES.min(MA_MAX_TREE_CONTEXTS.saturating_sub(dc_leaves));
                if cap >= 8 {
                    let planes = meta_planes(dc_datas, epf_sharpness);
                    learn_meta_parts(pool, scratch, &planes, budget, cap)
                } else {
                    None
                }
            } else {
                None
            };
            MaDcTrees { dc, meta }
        }
    };
    let arms = assemble_ma_arms(
        pool,
        scratch,
        dc_datas,
        meta_tokens,
        meta_props,
        epf_sharpness,
        wp_params,
        &trees,
        budget.offsets_variant,
    );
    Some((arms, trees))
}

/// Tokenize every stream through `tree`, one pool task per channel, joined
/// per stream in channel order.
fn tokenize_streams(
    pool: &crate::thread_pool::ThreadPool,
    scratch: &mut CoderScratch,
    streams: &[crate::lossless::DcPlanes],
    tree: &crate::ma_tree::LearnedTree,
    leaf_ctx: &[u32],
    wp_params: WpParams,
) -> Vec<Vec<Token>> {
    let channels = crate::lossless::dc_channels(streams);
    let parts = pool.steal_map(scratch, channels.len(), |i, _scratch| {
        let (g, c) = channels[i];
        let (w, h) = streams[g].dims[c];
        let mut out = Vec::with_capacity(w * h);
        crate::lossless::tokenize_dc_channel(&streams[g], c, tree, leaf_ctx, wp_params, &mut out);
        out
    });
    let mut tokens: Vec<Vec<Token>> = streams
        .iter()
        .map(|g| Vec::with_capacity(g.len()))
        .collect();
    for (&(g, _), part) in channels.iter().zip(parts) {
        tokens[g].extend(part);
    }
    tokens
}

/// Tokenize the DC groups and the metadata streams through every graft of
/// `trees`: without and, when any leaf has one, with median DC leaf
/// offsets, each with the static refined metadata side and, when learned,
/// the learned one.
#[allow(clippy::too_many_arguments)]
fn assemble_ma_arms(
    pool: &crate::thread_pool::ThreadPool,
    scratch: &mut CoderScratch,
    dc_datas: &[DcGroupData],
    meta_tokens: &[Vec<Token>],
    meta_props: &[Vec<crate::dc_tree::DcProp>],
    epf_sharpness: i32,
    wp_params: WpParams,
    trees: &MaDcTrees,
    offsets_variant: bool,
) -> Vec<MaDcArm> {
    let groups = dc_planes(dc_datas);
    let num_dc_groups = dc_datas.len();
    // Tokens carry their leaf's node index as context; each graft maps them
    // to its BFS context numbering.
    let dc_identity: Vec<u32> = (0..trees.dc.nodes.len() as u32).collect();
    let dc_nodes = tokenize_streams(pool, scratch, &groups, &trees.dc, &dc_identity, wp_params);
    let meta_nodes: Option<Vec<Vec<Token>>> = trees.meta.as_ref().map(|meta_tree| {
        let planes = meta_planes(dc_datas, epf_sharpness);
        let identity: Vec<u32> = (0..meta_tree.nodes.len() as u32).collect();
        tokenize_streams(
            pool,
            scratch,
            &planes,
            meta_tree,
            &identity,
            WpParams::DEFAULT,
        )
    });
    let map_ctx = |tokens: &[Vec<Token>], leaf_ctx: &[u32]| -> Vec<Vec<Token>> {
        tokens
            .iter()
            .map(|g| {
                g.iter()
                    .map(|t| Token::new(leaf_ctx[t.context as usize], t.value))
                    .collect()
            })
            .collect()
    };
    let static_meta = |meta_context: &[u16]| -> Vec<Vec<Token>> {
        meta_tokens
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
            .collect()
    };
    // Residuals do not depend on the offset: center each DC leaf on its
    // median. Offsets only pay through clustering, so both are priced.
    let node_offsets = crate::dc_tree::median_leaf_offsets(&dc_nodes, trees.dc.nodes.len());
    let with_offsets = offsets_variant && node_offsets.iter().any(|&o| o != 0);
    let zero_offsets = vec![0i32; trees.dc.nodes.len()];
    let mut arms = Vec::new();
    for offsets in [false, true] {
        if offsets && !with_offsets {
            continue;
        }
        let offs = if offsets {
            &node_offsets
        } else {
            &zero_offsets
        };
        let mut dc_tokens = dc_nodes.clone();
        if offsets {
            crate::dc_tree::apply_leaf_offsets(&mut dc_tokens, offs);
        }
        let grafted = crate::dc_tree::graft_ma_dc_tree(
            num_dc_groups,
            &trees.dc,
            offs,
            meta_tokens,
            meta_props,
        );
        arms.push(MaDcArm {
            dc: map_ctx(&dc_tokens, &grafted.leaf_ctx),
            meta: static_meta(&grafted.meta_context),
            tree: grafted.tokens,
            num_contexts: grafted.num_contexts,
            dc_leaf_ctx: grafted.leaf_ctx,
            offsets: offs.clone(),
            meta_learned: false,
        });
        if let (Some(meta_tree), Some(meta_nodes)) = (&trees.meta, &meta_nodes) {
            let joint =
                crate::dc_tree::graft_ma_dc_meta_tree(num_dc_groups, &trees.dc, offs, meta_tree);
            arms.push(MaDcArm {
                dc: map_ctx(&dc_tokens, &joint.dc_leaf_ctx),
                meta: map_ctx(meta_nodes, &joint.meta_leaf_ctx),
                tree: joint.tokens,
                num_contexts: joint.num_contexts,
                dc_leaf_ctx: joint.dc_leaf_ctx,
                offsets: offs.clone(),
                meta_learned: true,
            });
        }
    }
    arms
}

/// Re-pick every DC leaf's predictor by the cost of its residuals under the
/// leaf's clustered histogram in `code` (the learner picked by the leaf's
/// own order-0 entropy, before clustering merged it with others). Returns
/// the trees with the new leaves when any changed.
fn repick_leaf_predictors(
    dc_datas: &[DcGroupData],
    trees: &MaDcTrees,
    arm_leaf_ctx: &[u32],
    arm_offsets: &[i32],
    code: &OwnedEntropyCode,
    wp_params: WpParams,
) -> Option<MaDcTrees> {
    use crate::ma_tree::{MaNode, NUM_MA_PREDS};
    if code.use_prefix_code {
        return None;
    }
    let cost_tables: Vec<Vec<f32>> = code
        .ans_histograms
        .iter()
        .map(|h| {
            h.freqs
                .iter()
                .map(|&f| {
                    if f == 0 {
                        crate::entropy::ANS_LOG_TAB_SIZE as f32
                    } else {
                        (f64::from(crate::entropy::ANS_LOG_TAB_SIZE) - f_log2(f64::from(f))) as f32
                    }
                })
                .collect()
        })
        .collect();
    let nodes = trees.dc.nodes.len();
    let mut cost = vec![0f32; nodes * NUM_MA_PREDS];
    for g in dc_planes(dc_datas) {
        crate::lossless::walk_dc_planes_predictors(&g, &trees.dc, wp_params, |node, residuals| {
            let ctx = arm_leaf_ctx[node as usize] as usize;
            let cl = code.context_map[ctx] as usize;
            let config = code.hybrid_uint_configs[cl];
            let table = &cost_tables[cl];
            let row = &mut cost[node as usize * NUM_MA_PREDS..(node as usize + 1) * NUM_MA_PREDS];
            for (c, &r) in row.iter_mut().zip(residuals) {
                let (sym, nbits, _) = crate::entropy::uint_encode_with_config(
                    crate::entropy::pack_signed((r - i64::from(arm_offsets[node as usize])) as i32),
                    config,
                );
                *c += table[(sym as usize).min(table.len() - 1)] + nbits as f32;
            }
        });
    }
    let mut changed = false;
    let mut dc = trees.dc.clone();
    for (node, n) in dc.nodes.iter_mut().enumerate() {
        if let MaNode::Leaf { pred, .. } = n {
            let row = &cost[node * NUM_MA_PREDS..(node + 1) * NUM_MA_PREDS];
            let (best, _) = row.iter().enumerate().fold(
                (*pred as usize, row[*pred as usize]),
                |acc, (p, &c)| {
                    if c < acc.1 { (p, c) } else { acc }
                },
            );
            if best != *pred as usize {
                *pred = best as u32;
                changed = true;
            }
        }
    }
    changed.then(|| MaDcTrees {
        dc,
        meta: trees.meta.clone(),
    })
}
