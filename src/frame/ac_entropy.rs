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

//! AC context planning, entropy refinement and coefficient order serialization.

use super::ac_groups::PendingAcGroup;
use super::header::{write_ac_global, write_block_ctx_map};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::entropy::{AcTokens, write_entropy_code, write_token};

pub(super) fn finalize_ac_entropy(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &mut [PendingAcGroup],
    num_passes: usize,
    qf_threshold: u32,
    ans_refinement: Option<crate::entropy::AnsRefinement>,
) -> (
    crate::ac_context::AcCtxPlan,
    Vec<crate::entropy::OwnedEntropyCode>,
) {
    let effort = ctx.speed.effort();
    // Per-image AC block-context plan: keep quant-field splits only where the
    // real token statistics pay for them within the spec's 16-context budget.
    // The greedy proposes; the arm gate below disposes, comparing real
    // entropy-code headers plus payload estimates because the Shannon proxy
    // reliably overestimates what clustering-aware coding realizes.
    let baseline = if effort.square_merges {
        crate::ac_context::AcCtxPlan::baseline()
    } else {
        crate::ac_context::AcCtxPlan::dct8_only()
    };

    let build_codes = |pending: &[PendingAcGroup],
                       num_contexts: usize,
                       scratch: &mut CoderScratch|
     -> Vec<crate::entropy::OwnedEntropyCode> {
        (0..num_passes)
            .map(|pass| {
                if effort.ac_prefix_clustering {
                    crate::entropy::optimize_entropy_code_ac_streams(
                        pending.iter().map(|p| p.tokens[pass].as_slice()),
                        num_contexts,
                        &mut scratch.huffman_pool,
                        effort.huc_select,
                        ctx.speed,
                        Some(&ctx.thread_pool),
                    )
                } else {
                    crate::entropy::optimize_entropy_code_ac_streams_fast(
                        pending.iter().map(|p| p.tokens[pass].as_slice()),
                        num_contexts,
                        &mut scratch.huffman_pool,
                        effort.huc_select,
                        Some(&ctx.thread_pool),
                    )
                }
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
            bits += crate::entropy::estimate_ac_plain_bits(
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
                    pass_tokens.remap_contexts(|context| plan.remap(context));
                }
            });
    };

    let (ac_plan, mut ac_code_per_pass) = if !ctx.speed.effort().ac_ctx_plan {
        remap_tokens(pending, &baseline, scratch);
        let codes = build_codes(pending, baseline.num_ac_contexts(), scratch);
        (baseline, codes)
    } else {
        let proposed = crate::ac_context::plan_block_ctx_map(
            pending
                .iter()
                .flat_map(|pending| pending.tokens.iter().map(AcTokens::as_slice)),
            qf_threshold,
        );
        if proposed == baseline {
            remap_tokens(pending, &baseline, scratch);
            let codes = build_codes(pending, baseline.num_ac_contexts(), scratch);
            (baseline, codes)
        } else {
            // Both arms keep the coefficient values. Save only the original
            // contexts so the same token buffers can be remapped for each arm.
            const { assert!(crate::ac_context::K_NUM_FINE_AC_CONTEXTS <= 1 << 16) };
            let contexts: Vec<Vec<Vec<u16>>> = pending
                .iter()
                .map(|p| {
                    p.tokens
                        .iter()
                        .map(|tokens| tokens.iter().map(|t| t.context as u16).collect())
                        .collect()
                })
                .collect();
            let remap_saved = |pending: &mut [PendingAcGroup],
                               plan: &crate::ac_context::AcCtxPlan,
                               scratch: &mut CoderScratch| {
                ctx.thread_pool
                    .steal_for_each_mut(scratch, pending, |i, p, _s| {
                        for (tokens, contexts) in p.tokens.iter_mut().zip(&contexts[i]) {
                            let mut contexts = contexts.iter();
                            tokens.remap_contexts(|_| {
                                plan.remap(u32::from(*contexts.next().unwrap()))
                            });
                        }
                    });
            };
            remap_tokens(pending, &baseline, scratch);

            let base_codes = build_codes(pending, baseline.num_ac_contexts(), scratch);
            let base_bits = arm_bits(pending, &base_codes, &baseline, scratch);
            remap_saved(pending, &proposed, scratch);
            let plan_codes = build_codes(pending, proposed.num_ac_contexts(), scratch);
            let plan_bits = arm_bits(pending, &plan_codes, &proposed, scratch);

            const AC_PLAN_MARGIN_BITS: u64 = 256;
            if plan_bits + AC_PLAN_MARGIN_BITS < base_bits {
                (proposed, plan_codes)
            } else {
                remap_saved(pending, &baseline, scratch);
                (baseline, base_codes)
            }
        }
    };

    // Slow always considers splitting clusters collapsed by the prefix-cost
    // search. Fast uses the cheaper batch of moves at high quality.
    if let Some(refinement) = ans_refinement {
        refine_ac_codes(ctx, scratch, pending, &mut ac_code_per_pass, refinement);
    }

    (ac_plan, ac_code_per_pass)
}

fn refine_ac_codes(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &[PendingAcGroup],
    codes: &mut [crate::entropy::OwnedEntropyCode],
    refinement: crate::entropy::AnsRefinement,
) {
    for (pass, code) in codes.iter_mut().enumerate() {
        crate::entropy::refine_ans_clusters(
            code,
            pending
                .iter()
                .map(|pending| pending.tokens[pass].as_slice()),
            refinement,
            &ctx.thread_pool,
            scratch,
        );
    }
}

pub(super) fn encode_ac_sections(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &[PendingAcGroup],
    codes: &[crate::entropy::OwnedEntropyCode],
    num_dc_groups: usize,
) -> Vec<(usize, BitWriter)> {
    let num_passes = codes.len();
    let num_groups = pending.len();
    // Write each (pass, group) AC section. Section index for
    // (pass, group) = 2 + num_dc_groups + pass*num_groups + group_idx
    // (jxl-frame toc.rs:196-200): raw tokens via the shared plain code.
    let num_ac_sections = pending.len() * num_passes;
    ctx.thread_pool
        .steal_map(scratch, num_ac_sections, |task, _scratch| {
            let i = task / num_passes;
            let pass = task % num_passes;
            let pg = &pending[i];
            let pass_tokens = pg.tokens[pass].as_slice();
            let mut w = BitWriter::new();
            let section_idx = 2 + num_dc_groups + pass * num_groups + pg.group_idx;
            let code_ref = codes[pass].as_ref();
            if code_ref.use_prefix_code {
                pass_tokens
                    .iter()
                    .for_each(|t| write_token(t, &code_ref, &mut w));
            } else {
                // rANS: the whole group's tokens are encoded as one LIFO unit.
                crate::entropy::write_ans_tokens(
                    pass_tokens,
                    code_ref.context_map,
                    code_ref.ans_symbols,
                    code_ref.ans_reverse_maps,
                    code_ref.hybrid_uint_configs,
                    &mut w,
                );
            }
            (section_idx, w)
        })
}

/// One coefficient-scan arm of ExtraSlow's final race.
pub(super) struct OrderArm {
    pub(super) orders: crate::coeff_order::CoeffOrders,
    pub(super) pending: Vec<PendingAcGroup>,
    pub(super) plan: crate::ac_context::AcCtxPlan,
    pub(super) codes: Vec<crate::entropy::OwnedEntropyCode>,
    pub(super) bits: usize,
    pub(super) global: BitWriter,
    pub(super) streams: Vec<(usize, BitWriter)>,
}

/// Arms whose serialized size after the Slow refinement exceeds the leader's
/// by more than this fraction skip the exhaustive table search.
pub(super) const ORDER_RACE_REFINE_MARGIN: f64 = 0.001;

/// Serialize the varying AC headers and each independent entropy stream.
/// The block-context map lives in DC global; all its other data is common to
/// both scans. Multi-group AC sections are byte-aligned, while single-group
/// frames concatenate these sections without intervening padding.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_order_arm(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    pending: &[PendingAcGroup],
    plan: &crate::ac_context::AcCtxPlan,
    codes: &[crate::entropy::OwnedEntropyCode],
    orders: &crate::coeff_order::CoeffOrders,
    used_tables: &[bool; 13],
    num_groups: usize,
    num_dc_groups: usize,
) -> (usize, BitWriter, Vec<(usize, BitWriter)>) {
    let mut context_header = BitWriter::new();
    write_block_ctx_map(plan, scratch, &mut context_header);
    let mut global = BitWriter::new();
    write_ac_global(
        ctx.matrices(),
        used_tables,
        orders,
        num_groups,
        codes,
        scratch,
        &mut global,
    );
    let streams = encode_ac_sections(ctx, scratch, pending, codes, num_dc_groups);
    let padded_bits = |bits: usize| {
        if num_groups > 1 {
            bits.div_ceil(8) * 8
        } else {
            bits
        }
    };
    let bits = context_header.bits_written()
        + padded_bits(global.bits_written())
        + streams
            .iter()
            .map(|(_, w)| padded_bits(w.bits_written()))
            .sum::<usize>();
    (bits, global, streams)
}

#[cfg(test)]
mod tests {

    #[test]
    fn coefficient_order_race_prices_serialized_permutations_and_ac_sections() {
        use super::{PendingAcGroup, encode_order_arm};
        use crate::bit_writer::BitWriter;
        use crate::coeff_order::{CoeffOrders, write_coeff_orders};
        use crate::encoding_context::EncodingContext;
        use std::borrow::Cow;

        let ctx = EncodingContext::default();
        let mut scratch = crate::coder_scratch::CoderScratch::default();
        let natural = CoeffOrders::natural();
        let mut custom = CoeffOrders::natural();
        let mut scan = custom.orders[0][0].to_vec();
        scan.swap(1, 2);
        custom.orders[0][0] = Cow::Owned(scan);
        custom.used_mask = 1;
        let plan = crate::ac_context::AcCtxPlan::dct8_only();
        for groups in [1, 2] {
            let pending: Vec<_> = (0..groups)
                .map(|group_idx| PendingAcGroup {
                    group_idx,
                    tokens: vec![
                        (0..4096)
                            .map(|i| crate::entropy::Token::new(0, (i % 13) as u32))
                            .collect::<Vec<_>>()
                            .into(),
                    ],
                })
                .collect();
            let code = crate::entropy::optimize_entropy_code_ac_streams(
                pending.iter().map(|p| p.tokens[0].as_slice()),
                plan.num_ac_contexts(),
                &mut scratch.huffman_pool,
                true,
                ctx.speed,
                Some(&ctx.thread_pool),
            );
            let codes = [code];
            let natural_arm = encode_order_arm(
                &ctx,
                &mut scratch,
                &pending,
                &plan,
                &codes,
                &natural,
                &[false; 13],
                groups,
                1,
            );
            let custom_arm = encode_order_arm(
                &ctx,
                &mut scratch,
                &pending,
                &plan,
                &codes,
                &custom,
                &[false; 13],
                groups,
                1,
            );
            // Identical coefficient streams cannot pay for a permutation.
            assert!(natural_arm.0 < custom_arm.0);
            assert_eq!(natural_arm.2.len(), groups);
            for (i, (section, _)) in natural_arm.2.iter().enumerate() {
                assert_eq!(*section, 3 + i);
            }
            if groups == 1 {
                let mut a = BitWriter::new();
                let mut b = BitWriter::new();
                write_coeff_orders(&natural, &mut scratch.huffman_pool, &mut a);
                write_coeff_orders(&custom, &mut scratch.huffman_pool, &mut b);
                assert_eq!(
                    custom_arm.0 - natural_arm.0,
                    b.bits_written() - a.bits_written()
                );
            }
        }
    }
}
