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

//! Transform, quantize and serialize the selected VarDCT frame.

use super::ac_entropy::{
    ORDER_RACE_REFINE_MARGIN, OrderArm, encode_ac_sections, encode_order_arm, finalize_ac_entropy,
};
#[cfg(not(feature = "extraslow"))]
use super::ac_groups::RefinedAcGroup;
use super::ac_groups::{PendingAcGroup, ac_qf_threshold, paste_group, process_ac_group};
use super::dc_groups::{assemble_source_dc, merge_quant_dc, setup_dc_group};
use super::dc_smoothing::{
    choose_chroma_dc_coding, compensate_b_dc, flatten_chroma_dc, skip_dc_smoothing,
};
use super::extra_channels::prepare_extra_sections;
use super::header::{
    used_quant_table_slots, write_ac_global, write_dc_global, write_frame_header_kind,
};
#[cfg(feature = "extraslow")]
use super::quant_field_loop::run_quant_field_loop;
#[cfg(not(feature = "extraslow"))]
use super::quantization::DistanceParams;
use super::quantization::{compute_distance_params, dc_steps_in_range, epf_sharpness_id};
use super::sections::{combine_sections_ordered, group_section_order};
use super::{
    ImageDim, K_BLOCK_DIM, K_DC_GROUP_DIM, K_GROUP_DIM_IN_BLOCKS, dc_rdo, dc_smoothing, extra_tree,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::color_correlation::choose_ytob_dc;
use crate::dc_group_data::DcGroupData;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::{EncodingContext, FrameRole};
use crate::entropy::write_token;
use crate::image::Image3F;
use crate::patches::VarDctFrameKind;
use crate::util::EncodeError;

/// Share of the scan walk the custom coefficient orders must remove before
/// the refine pass's prices are rebuilt on them.
const RDOQ_FRESH_PRICES_MIN_SCAN_MOVE: f64 = 0.35;

/// The closed loop settles a frame's B quant scale in the shared context;
/// the next frame, or the next sampled estimate, starts from the configured one.
#[cfg(feature = "extraslow")]
struct RestoreBQmScale<'a>(&'a EncodingContext, u32);

#[cfg(feature = "extraslow")]
impl Drop for RestoreBQmScale<'_> {
    fn drop(&mut self) {
        self.0.set_b_qm_scale(self.1);
    }
}

/// Returns DC-group and AC-section bits, excluding the global headers, so
/// sampled payload can be scaled separately from the frame's fixed costs.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_frame_core(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    opsin: Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    #[cfg(feature = "splines")] splines: Option<&crate::splines::SplineSet>,
    writer: &mut BitWriter,
) -> Result<usize, EncodeError> {
    #[cfg(feature = "extraslow")]
    let _b_qm_scale = RestoreBQmScale(ctx, ctx.b_qm_scale());
    let num_threads = ctx.thread_pool.num_threads();
    let dim = ImageDim::new(opsin.xsize(), opsin.ysize());
    // Grain is measured on the source XYB before any frame processing; only
    // displayed frames carry it (an atlas frame's header has no flags).
    let mut opsin = opsin;
    let noise = if matches!(
        frame_kind,
        VarDctFrameKind::Regular | VarDctFrameKind::Patched(_)
    ) {
        crate::noise::NoiseParams::resolve(ctx.grain, ctx, scratch, distance, &mut opsin)
    } else {
        None
    };
    let mut distp = compute_distance_params(distance);
    if ctx.coding.is_xyb() {
        if ctx.x_heavy() && distp.x_qm_scale == 2 {
            distp.x_qm_scale = 3;
        }
        distp.x_qm_scale = distp.x_qm_scale.max(ctx.x_qm_scale_floor());
    } else {
        // Not serialized: the decoder's X multiplier is 1.
        distp.x_qm_scale = 2;
        distp.dc_step = dc_steps_in_range(ctx.coding.dc_step_base(), distp.scale_dc, &opsin);
    }

    // Progressive lossy splits each quantized AC coeff across `num_passes`
    // passes by a decreasing per-pass shift (last = 0). The decoder reconstructs
    // C = sum_p (sent_p << shift_p) (jxl-vardct hf_coeff.rs:185,191).
    let num_passes = coeff_shifts.len();
    // Progressive DC: the DC image goes out as its own squeezed DC frame,
    // which this frame references instead of coding a DC stream.
    let use_dc_frame = ctx.progressive_dc
        && matches!(
            frame_kind,
            VarDctFrameKind::Regular | VarDctFrameKind::Patched(_)
        )
        && alpha.is_none()
        && ctx.coding.is_xyb()
        && crate::lossless::dc_frame_supported(dim.xsize_blocks, dim.ysize_blocks);

    let num_sections = 2 + dim.num_dc_groups + num_passes * dim.num_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // Set up every DC group, then encode every AC group. Keeping the
    // two apart lets AC groups from all DC groups share one steal_map instead of
    // one serialized burst per DC group. Merges back in (dc, gix) order, so the
    // output stays bit-identical to single-threaded.
    let group_coords: Vec<(usize, usize)> = (0..dim.ysize_dc_groups)
        .flat_map(|gy| (0..dim.xsize_dc_groups).map(move |gx| (gx, gy)))
        .collect();
    let opsin_image = opsin;
    let opsin = &opsin_image;

    // Split the thread budget across nesting levels: `outer` lanes steal DC
    // groups, each parallelizing its AC-strategy bands with the remainder, so
    // the setup phase saturates all cores even with few (large) DC groups.
    let outer = group_coords.len().min(num_threads.max(1));
    let setup_budget = num_threads.max(1).div_ceil(outer);
    let setups: Vec<Result<(DcGroupData, usize, usize), EncodeError>> =
        ctx.thread_pool
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
    let setups: Vec<(DcGroupData, usize, usize)> = setups.into_iter().collect::<Result<_, _>>()?;

    let mut dc_datas: Vec<DcGroupData> = Vec::with_capacity(setups.len());
    let mut ac_tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (dc_idx, (dc_data, gxs, gys)) in setups.into_iter().enumerate() {
        ac_tasks.extend((0..gxs * gys).map(|g| (dc_idx, g % gxs, g / gxs)));
        dc_datas.push(dc_data);
    }

    crate::ac_strategy::account_matrix_headers(
        ctx,
        scratch,
        opsin,
        distance,
        distp.scale,
        distp.x_qm_scale,
        &group_coords,
        &mut dc_datas,
    );
    for dc in dc_datas.iter_mut() {
        dc.block_quant_field = None;
    }

    // Layer residuals and progressive/DC frames retain the EPF schedule.
    let fit_epf = ctx.speed.effort().epf_fit
        && ctx.coding.is_xyb()
        && matches!(frame_kind, VarDctFrameKind::Regular)
        && ctx.frame_role == FrameRole::Last
        && noise.is_none()
        && !use_dc_frame
        && num_passes == 1
        && (0.5..=24.0).contains(&distance)
        && dim.xsize >= 16
        && dim.ysize >= 16;
    #[cfg(feature = "splines")]
    let fit_epf = fit_epf && splines.is_none();
    let mut epf_recon = fit_epf.then(|| Image3F::new(dim.xsize, dim.ysize));

    // Per-image quant-field threshold for the fine AC block-context layout.
    // The median splits the field into halves with genuinely different
    // coefficient statistics; whether any split is *kept* is decided later
    // from real token stats, so a useless threshold costs nothing.
    let qf_threshold = ac_qf_threshold(&dc_datas);

    // The CfL slopes were estimated on fixed 8x8 DCTs so the transform search
    // could price candidates under them; the final layout and quant field
    // are known now, so re-estimate them on the coefficients that get coded.
    if crate::color_correlation::cfl_slope_search(ctx) {
        let dc_ref = &dc_datas;
        let maps = ctx
            .thread_pool
            .steal_map(scratch, group_coords.len(), |i, scratch| {
                let (dc_gx, dc_gy) = group_coords[i];
                let (coeffs, gather) = scratch.ac_group.transform_buffers();
                crate::color_correlation::refit_cmap_on_layout(
                    ctx,
                    opsin,
                    &dc_ref[i],
                    dc_gx * K_DC_GROUP_DIM / K_BLOCK_DIM,
                    dc_gy * K_DC_GROUP_DIM / K_BLOCK_DIM,
                    distp.scale,
                    distp.distance,
                    distp.x_qm_scale,
                    coeffs,
                    gather,
                    &mut scratch.cfl_rdo,
                )
            });
        for (dc, (ytox, ytob)) in dc_datas.iter_mut().zip(maps) {
            dc.ytox_map = ytox;
            dc.ytob_map = ytob;
        }
    }

    let dc_ref = &dc_datas;
    let natural_orders = crate::coeff_order::CoeffOrders::natural();
    // Tally coefficient positions only when the refine pass will run to consume
    // the derived orders: pass one alone can never adopt them (its tokens are
    // already written on the natural scan), so the work would be wasted and its
    // mere presence perturbs codegen enough to shift a few borderline
    // quantizer roundings on paths that should be untouched.
    let want_order_stats = ctx.speed.effort().coeff_orders;
    // Passes before the refined one only feed statistics and token prices,
    // which describe whole coefficients: they tokenize as one pass, and only
    // the refined pass splits the coefficients across the frame's passes.
    let (analysis_passes, analysis_shifts): (usize, &[u32]) = if want_order_stats {
        (1, &[0])
    } else {
        (num_passes, coeff_shifts)
    };
    let results = ctx
        .thread_pool
        .steal_map(scratch, ac_tasks.len(), |t, scratch| {
            let (dc_idx, gx, gy) = ac_tasks[t];
            let (dc_gx, dc_gy) = group_coords[dc_idx];
            let (p, local, source_b, stats, capture) = process_ac_group(
                ctx,
                scratch,
                opsin,
                &dim,
                &distp,
                &dc_ref[dc_idx],
                analysis_passes,
                analysis_shifts,
                dc_gx,
                dc_gy,
                gx,
                gy,
                0,
                None,
                &natural_orders,
                &[],
                want_order_stats || use_dc_frame || fit_epf,
                qf_threshold,
                // Without custom orders this pass is the final quantization.
                fit_epf && !want_order_stats,
                None,
            );
            (
                (dc_idx, gx, gy, p, local, source_b, stats),
                capture.map(|(image, _)| image),
            )
        });

    let mut all_pending: Vec<PendingAcGroup> = Vec::with_capacity(results.len());
    // Adopt the first group's buffers instead of allocating an empty tally.
    // Fast and progressive frames never allocate aggregate order statistics.
    let mut order_stats: Option<crate::coeff_order::OrderStats> = None;
    for ((dc_idx, gx, gy, p, local, source, stats), capture) in results {
        if let (Some(target), Some(image)) = (epf_recon.as_mut(), capture) {
            paste_group(target, &dim, p.group_idx, &image);
        }
        merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
        if let Some(source) = source {
            let dc = &mut dc_datas[dc_idx];
            let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
            let ox = gx * K_GROUP_DIM_IN_BLOCKS;
            let oy = gy * K_GROUP_DIM_IN_BLOCKS;
            for (target, source) in [
                (&mut dc.source_dc_y, &source.y),
                (&mut dc.source_dc_b, &source.b),
                (&mut dc.source_dc_x, &source.x),
            ] {
                let target = target.get_or_insert_with(|| crate::image::Plane::new(w, h));
                for y in 0..source.ysize() {
                    target.row_mut(oy + y)[ox..ox + source.xsize()].copy_from_slice(source.row(y));
                }
            }
        }
        all_pending.push(p);
        if let Some(s) = stats {
            if let Some(total) = &mut order_stats {
                total.merge(&s);
            } else {
                order_stats = Some(s);
            }
        }
    }

    // The EPF fit adds the decoded-minus-source DC to its reconstruction.
    // Other DC tools key on the source planes, so only a copy outlives the
    // first pass when nothing else asked for them.
    let mut epf_source_dc = None;
    if fit_epf && !want_order_stats && !use_dc_frame {
        epf_source_dc = Some(assemble_source_dc(&dc_datas, &group_coords, &dim));
        for dc in &mut dc_datas {
            dc.source_dc_b = None;
            dc.source_dc_y = None;
            dc.source_dc_x = None;
        }
    }

    // DC-level chroma-from-luma. The slope has to be folded into the DC
    // quantizer's rounding rather than subtracted from the stored integers
    // afterwards: `round(v) - round(s*y)` carries up to a full step of B DC
    // error against `round(v - s*y)`'s half step, and that extra chroma noise
    // costs far more quality than the slope saves in rate. So the search rides
    // on the rerank pass, which re-quantizes every DC group regardless -- and
    // is skipped entirely when that pass does not run.
    // Custom coefficient orders, derived from the first pass's nonzero tallies.
    let mut coeff_orders = natural_orders;
    let mut order_alternatives = Vec::new();

    let mut ytob_dc = 0i32;
    if want_order_stats {
        let scan_moved = order_stats.as_ref().map_or(0.0, |stats| {
            crate::coeff_order::derive_orders(stats, &mut coeff_orders)
        });
        // Keep the incumbent RDOQ scan: the final race changes coding only,
        // so all arms transmit exactly the same quantized coefficients.
        if ctx.speed.effort().coeff_order_entropy_race() && num_passes == 1 {
            if let Some(stats) = &order_stats {
                let mut proposed = crate::coeff_order::CoeffOrders::natural();
                crate::coeff_order::propose_orders(stats, &mut proposed);
                if proposed.used_mask != 0 && proposed.orders != coeff_orders.orders {
                    order_alternatives.push(proposed);
                }
            }
            if coeff_orders.used_mask != 0 {
                order_alternatives.push(crate::coeff_order::CoeffOrders::natural());
            }
        }
        drop(order_stats);
        ytob_dc = if use_dc_frame {
            0
        } else if ctx.speed.effort().dc_cfl_weighted() {
            crate::color_correlation::choose_ytob_dc_weighted(
                &dc_datas,
                ctx,
                distp.scale_dc,
                distp.dc_step,
                &mut scratch.dc_cfl_cur,
                &mut scratch.dc_cfl_prev,
            )
        } else {
            let candidate = choose_ytob_dc(
                &dc_datas,
                ctx.fill_ytob_row,
                ctx.accumulate_ytob_weights,
                ctx.fill_ytob_residuals,
                &mut scratch.dc_cfl_cur,
                &mut scratch.dc_cfl_prev,
                distp.dc_step,
                ctx.cfl_frame().color_factor,
            );
            crate::color_correlation::validate_ytob_dc(
                &dc_datas,
                candidate,
                distp.scale_dc,
                distp.dc_step,
                ctx.cfl_frame(),
                ctx.quantize_dc_cfl,
            )
        };
        if !use_dc_frame {
            distp.dc_step = crate::color_correlation::choose_dc_steps(
                &dc_datas,
                ctx,
                distp.scale_dc,
                ytob_dc,
                distp.dc_step,
                ctx.cfl_frame(),
            );
        }
        if !ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
            if fit_epf && epf_source_dc.is_none() {
                epf_source_dc = Some(assemble_source_dc(&dc_datas, &group_coords, &dim));
            }
            for dc in &mut dc_datas {
                dc.source_dc_b = None;
                dc.source_dc_y = None;
                dc.source_dc_x = None;
            }
        }
        // The first pass's tokens follow the natural scans. Once the custom
        // orders move most of the walk, they no longer describe what the
        // contexts of the refine pass will see, and the quantizer would price
        // its choices on them: tokenize again on the orders it codes with.
        if ctx.speed.effort().rdoq_fresh_prices() || scan_moved >= RDOQ_FRESH_PRICES_MIN_SCAN_MOVE {
            // These streams will be replaced; release their token buffers
            // before allocating the reordered streams for the whole frame.
            all_pending.clear();
            let dc_ref = &dc_datas;
            let reordered = ctx
                .thread_pool
                .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    let (dc_gx, dc_gy) = group_coords[dc_idx];
                    process_ac_group(
                        ctx,
                        scratch,
                        opsin,
                        &dim,
                        &distp,
                        &dc_ref[dc_idx],
                        analysis_passes,
                        analysis_shifts,
                        dc_gx,
                        dc_gy,
                        gx,
                        gy,
                        ytob_dc,
                        None,
                        &coeff_orders,
                        &[],
                        false,
                        qf_threshold,
                        false,
                        None,
                    )
                    .0
                });
            all_pending.extend(reordered);
        }
        let prices = {
            let provisional_code = crate::entropy::optimize_entropy_code_ac_streams(
                all_pending.iter().map(|pg| pg.tokens[0].as_slice()),
                crate::ac_context::K_NUM_FINE_AC_CONTEXTS,
                &mut scratch.huffman_pool,
                // No config selection here: these prices feed RDOQ, and even a
                // one-token nudge can push the clustering off a knife-edge merge.
                // Selection applies to the final codes only.
                false,
                ctx.speed,
                Some(&ctx.thread_pool),
            );
            crate::entropy::FrozenTokenPrices::new(&provisional_code)
        };
        // The closed loop's reference and verification encodes are pass 2
        // itself (same RDOQ, prices and orders), so when it runs its final
        // groups replace the refined pass.
        all_pending.clear();
        #[cfg(feature = "extraslow")]
        let loop_groups =
            if ctx.speed.effort().aq_closed_loop() && ctx.coding.is_xyb() && num_passes == 1 {
                run_quant_field_loop(
                    ctx,
                    scratch,
                    opsin,
                    &dim,
                    &distp,
                    &mut dc_datas,
                    &ac_tasks,
                    &group_coords,
                    num_passes,
                    coeff_shifts,
                    &coeff_orders,
                    &order_alternatives,
                    ytob_dc,
                    qf_threshold,
                    &prices,
                    epf_recon.is_some(),
                )
            } else {
                None
            };
        #[cfg(not(feature = "extraslow"))]
        let loop_groups: Option<(Vec<RefinedAcGroup>, DistanceParams, Option<Image3F>)> = None;
        let loop_groups = loop_groups.map(|(groups, chosen, recon)| {
            distp = chosen;
            (groups, recon)
        });
        let refined = if let Some((groups, recon)) = loop_groups {
            // The loop's final reconstruction replaces the pass-2 captures.
            epf_recon = recon;
            groups.into_iter().map(|g| (g, None)).collect::<Vec<_>>()
        } else {
            let dc_ref = &dc_datas;
            ctx.thread_pool
                .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    let (dc_gx, dc_gy) = group_coords[dc_idx];
                    let (p, local, _, _, capture) = process_ac_group(
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
                        ytob_dc,
                        Some(&prices),
                        &coeff_orders,
                        &order_alternatives,
                        false,
                        qf_threshold,
                        fit_epf,
                        None,
                    );
                    ((dc_idx, gx, gy, p, local), capture.map(|(image, _)| image))
                })
        };
        for ((dc_idx, gx, gy, p, local), capture) in refined {
            if let (Some(target), Some(image)) = (epf_recon.as_mut(), capture) {
                paste_group(target, &dim, p.group_idx, &image);
            }
            merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
            all_pending.push(p);
        }
    }

    // The retained source DC is invariant under AC reranking. Search against
    // the final Y levels and steps, then quantize X once from that source.
    let ytox_dc = if ctx.coding.is_xyb() && ctx.speed.effort().dc_x_cfl && !use_dc_frame {
        crate::color_correlation::choose_ytox_dc(
            &mut dc_datas,
            distp.scale_dc,
            distp.dc_step,
            ctx.cfl_frame(),
            ytob_dc != 0,
            ctx.quantize_dc_cfl,
        )
    } else {
        0
    };
    if ctx.coding.is_xyb() && ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
        let cfl = crate::color_correlation::CflFrame {
            base_x: crate::color_correlation::y_to_x_ratio(ctx.cfl_frame(), ytox_dc as i8),
            ..ctx.cfl_frame()
        };
        distp.dc_step = dc_rdo::choose_x_dc_step(&mut dc_datas, &distp, cfl, ctx.quantize_dc_cfl);
    }
    // Only DC reconstruction uses the extra X slope. The AC map and the
    // header's base correlations retain the frame's original parameters.
    let dc_cfl = crate::color_correlation::CflFrame {
        base_x: crate::color_correlation::y_to_x_ratio(ctx.cfl_frame(), ytox_dc as i8),
        ..ctx.cfl_frame()
    };
    // Couple B rounding to the final luma error after AC decisions are fixed.
    // Flattening and Haar selection must use that same target as well.
    let compensate_b = ctx.coding.is_xyb()
        && ctx.speed.effort().chroma_dc_squeeze
        && !use_dc_frame
        && compensate_b_dc(
            &mut dc_datas,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.compensate_b,
        );
    if ctx.coding.is_xyb() && !use_dc_frame {
        flatten_chroma_dc(
            opsin,
            &dim,
            &mut dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
        )?;
    }
    if ctx.coding.is_xyb() && ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
        choose_chroma_dc_coding(
            &dim,
            &mut dc_datas,
            &mut distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
            ctx.dc_rows.adjust_b,
        )?;
    }
    if ctx.coding.is_xyb() && ctx.speed.effort().dc_rdoq && !use_dc_frame {
        dc_rdo::refine_dc_rounding(
            &mut dc_datas,
            &distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
            &mut scratch.dc_predictor,
        )?;
    }
    let skip_dc_smoothing = !use_dc_frame
        && skip_dc_smoothing(
            &ctx.thread_pool,
            scratch,
            opsin,
            &dim,
            &dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.reconstruct,
        )?;
    let mut epf_sharpness = epf_sharpness_id(distance, ctx.screen_content());
    if let Some(mut recon) = epf_recon {
        let decoded_dc = dc_smoothing::decoded_dc(
            &dim,
            &dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.reconstruct,
            skip_dc_smoothing,
        )?;
        let source_dc =
            epf_source_dc.unwrap_or_else(|| assemble_source_dc(&dc_datas, &group_coords, &dim));
        crate::epf_filter::add_dc_error(
            ctx,
            scratch,
            &mut recon,
            &decoded_dc,
            &source_dc,
            &dc_datas,
            &group_coords,
        );
        let source = distp
            .gab_enabled
            .then(|| crate::epf_filter::gaborish(opsin));
        let source = source.as_ref().unwrap_or(opsin);
        if distp.gab_enabled {
            crate::epf_filter::gaborish_in_place(&mut recon);
        }
        let blocks = crate::epf_filter::FrameBlocks::new(
            &dc_datas,
            &group_coords,
            dim.xsize_blocks,
            dim.ysize_blocks,
            distp.global_scale,
        );
        let chosen = crate::epf_filter::fit(
            ctx,
            scratch,
            source,
            &recon,
            &blocks,
            distance,
            crate::epf_filter::Choice {
                iters: distp.epf_iters,
                quant_mul: distp.epf_quant_mul,
                pass0: distp.epf_pass0_scale,
                sharpness: epf_sharpness,
            },
        );
        distp.epf_iters = chosen.iters;
        distp.epf_pass0_scale = chosen.pass0;
        distp.epf_quant_mul = chosen.quant_mul;
        epf_sharpness = chosen.sharpness;
    }
    // Pixel-domain decisions are finished; entropy coding only reads DC data
    // and tokens. Release the full-resolution planes before its working sets.
    drop(opsin_image);

    let super::dc_entropy::DcEntropy {
        tree: dc_tree,
        dc: dc_tokens_per_group,
        meta: meta_tokens_per_group,
        code: mut dc_code_owned,
        refinement: ans_refinement,
    } = super::dc_entropy::prepare_dc_entropy(
        ctx,
        scratch,
        &dc_datas,
        dim.num_dc_groups,
        epf_sharpness,
        use_dc_frame,
        distance,
    );
    let (mut dc_tree, mut dc_tokens_per_group, mut meta_tokens_per_group) =
        (dc_tree, dc_tokens_per_group, meta_tokens_per_group);
    let extra = alpha.map(|plane| {
        prepare_extra_sections(
            ctx,
            plane,
            &dim,
            &group_coords,
            num_passes - 1,
            extra_tree::GlobalStreams {
                tree: &mut dc_tree,
                code: &mut dc_code_owned,
                dc: &mut dc_tokens_per_group,
                meta: &mut meta_tokens_per_group,
                dc_written: !use_dc_frame,
            },
            scratch,
        )
    });
    let dc_code = dc_code_owned.as_ref();

    // Separate scan arms before context planning: each gets its own final
    // context map, hybrid-uint configuration, clustering and ANS refinement.
    let mut alternative_pending: Vec<Vec<PendingAcGroup>> = (0..order_alternatives.len())
        .map(|_| Vec::with_capacity(all_pending.len()))
        .collect();
    for p in &mut all_pending {
        for (tokens, arm) in p.tokens.drain(num_passes..).zip(&mut alternative_pending) {
            arm.push(PendingAcGroup {
                group_idx: p.group_idx,
                tokens: vec![tokens],
            });
        }
    }
    let mut raced_ac_sections = None;
    let (ac_plan, ac_code_per_pass) = if order_alternatives.is_empty() {
        finalize_ac_entropy(
            ctx,
            scratch,
            &mut all_pending,
            num_passes,
            qf_threshold,
            ans_refinement,
        )
    } else {
        let used_tables = used_quant_table_slots(&dc_datas);
        let mut arms: Vec<OrderArm> = Vec::with_capacity(1 + order_alternatives.len());
        // Stage 1: every arm gets its context plan, hybrid-uint configurations,
        // clustering and the Slow refinement, then serializes.
        for (orders, mut pending) in std::iter::once((coeff_orders, all_pending))
            .chain(order_alternatives.into_iter().zip(alternative_pending))
        {
            let (plan, codes) = finalize_ac_entropy(
                ctx,
                scratch,
                &mut pending,
                num_passes,
                qf_threshold,
                ans_refinement.map(|_| crate::entropy::AnsRefinement::Slow),
            );
            let (bits, global, streams) = encode_order_arm(
                ctx,
                scratch,
                &pending,
                &plan,
                &codes,
                &orders,
                &used_tables,
                dim.num_groups,
                dim.num_dc_groups,
            );
            arms.push(OrderArm {
                orders,
                pending,
                plan,
                codes,
                bits,
                global,
                streams,
            });
        }
        // Stage 2: the exhaustive table search moves every arm by nearly the
        // same amount, so only arms within reach of the leader buy it.
        let leader = arms.iter().map(|arm| arm.bits).min().unwrap_or(0);
        let reach = leader + (leader as f64 * ORDER_RACE_REFINE_MARGIN) as usize;
        if matches!(
            ans_refinement,
            Some(crate::entropy::AnsRefinement::ExtraSlow)
        ) {
            for arm in arms.iter_mut().filter(|arm| arm.bits <= reach) {
                for (pass, code) in arm.codes.iter_mut().enumerate() {
                    crate::entropy::refine_ans_entropy_exhaustive(
                        code,
                        arm.pending.iter().map(|p| p.tokens[pass].as_slice()),
                        &ctx.thread_pool,
                        scratch,
                    );
                }
                (arm.bits, arm.global, arm.streams) = encode_order_arm(
                    ctx,
                    scratch,
                    &arm.pending,
                    &arm.plan,
                    &arm.codes,
                    &arm.orders,
                    &used_tables,
                    dim.num_groups,
                    dim.num_dc_groups,
                );
            }
        }
        // Prefer natural on ties, otherwise retain the incumbent.
        let mut best = 0;
        for (i, arm) in arms.iter().enumerate().skip(1) {
            if arm.bits < arms[best].bits
                || (arm.bits == arms[best].bits && arm.orders.used_mask == 0)
            {
                best = i;
            }
        }
        let winner = arms.swap_remove(best);
        coeff_orders = winner.orders;
        all_pending = winner.pending;
        // Retain the winning serialized sections; do not encode them again.
        raced_ac_sections = Some((winner.global, winner.streams));
        (winner.plan, winner.codes)
    };

    // Write DC global with adaptive DC code.
    if let VarDctFrameKind::Patched(references) = frame_kind {
        crate::lossless::write_patch_dictionary(
            references,
            alpha.is_some(),
            scratch,
            &mut sections[0],
        );
    }
    #[cfg(feature = "splines")]
    if let Some(set) = splines {
        crate::splines::write_splines(set, scratch, &mut sections[0]);
    }
    if let Some(noise) = &noise {
        noise.write(&mut sections[0]);
    }
    write_dc_global(
        &distp,
        dim.num_dc_groups,
        &ac_plan,
        &dc_tree,
        &dc_code,
        ytox_dc,
        ytob_dc,
        ctx.cfl_frame(),
        extra.as_ref(),
        scratch,
        &mut sections[0],
    );

    // Each DC group is an independent entropy stream. Encode the
    // streams in parallel, then place their writers back in raster order.
    let dc_sections = ctx
        .thread_pool
        .steal_map(scratch, dc_datas.len(), |i, _scratch| {
            let dc_data = &dc_datas[i];
            let mut w = BitWriter::new();
            // With a DC frame, the group carries no DC stream.
            if !use_dc_frame {
                w.write(2, 0); // extra_dc_precision
                w.write(4, 3); // use global tree, default wp, no transforms
                if dc_code.use_prefix_code {
                    for t in &dc_tokens_per_group[i] {
                        write_token(*t, &dc_code, &mut w);
                    }
                } else {
                    crate::entropy::write_ans_tokens(
                        &dc_tokens_per_group[i],
                        dc_code.context_map,
                        dc_code.ans_symbols,
                        dc_code.ans_reverse_maps,
                        dc_code.hybrid_uint_configs,
                        &mut w,
                    );
                }
            }
            // ModularLfGroup: the extra channel's coarse squeeze levels.
            if let Some(extra) = &extra {
                w.append(&extra.lf[i]);
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
                    write_token(*t, &dc_code, &mut w);
                }
            } else {
                crate::entropy::write_ans_tokens(
                    &meta_tokens_per_group[i],
                    dc_code.context_map,
                    dc_code.ans_symbols,
                    dc_code.ans_reverse_maps,
                    dc_code.hybrid_uint_configs,
                    &mut w,
                );
            }
            w
        });
    for (i, section) in dc_sections.into_iter().enumerate() {
        sections[1 + i] = section;
    }
    drop(dc_tokens_per_group);
    drop(meta_tokens_per_group);

    let (ac_global, ac_sections) = if let Some(raced) = raced_ac_sections {
        raced
    } else {
        let mut global = BitWriter::new();
        write_ac_global(
            ctx.matrices(),
            &used_quant_table_slots(&dc_datas),
            &coeff_orders,
            dim.num_groups,
            &ac_code_per_pass,
            scratch,
            &mut global,
        );
        let ac = encode_ac_sections(
            ctx,
            scratch,
            &all_pending,
            &ac_code_per_pass,
            dim.num_dc_groups,
        );
        (global, ac)
    };
    sections[1 + dim.num_dc_groups] = ac_global;
    for (section_idx, section) in ac_sections {
        sections[section_idx] = section;
    }
    drop(all_pending);
    // Modular alpha: extra-channel modular data is decoded in the last pass
    // (the only pass whose modular sub-image shift range is set when num_ds=0,
    // jxl-frame lib.rs:101-108 / pass_group.rs:93). Write it into that section.
    if let Some(extra) = &extra {
        let last_pass = num_passes - 1;
        for (abs_group_id, section) in extra.ac.iter().enumerate() {
            sections[2 + dim.num_dc_groups + last_pass * dim.num_groups + abs_group_id]
                .append(section);
        }
    }

    // The DC frame precedes the frame that uses it.
    let mut dc_frame_bits = 0;
    if use_dc_frame {
        let dc_image = assemble_source_dc(&dc_datas, &group_coords, &dim);
        let mut dc_writer = BitWriter::new();
        let coded = crate::lossless::encode_dc_frame(
            &dc_image,
            std::array::from_fn(|c| {
                distp.dc_step[c] / (crate::quant_weights::INV_DC_QUANT[c] * distp.scale_dc)
            }),
            ctx.progressive_dc_levels,
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut dc_writer,
        );
        debug_assert!(coded, "DC frame size checked up front");
        dc_frame_bits = dc_writer.bits_written();
        writer.append(&dc_writer);
    }

    #[cfg(feature = "splines")]
    let has_splines = splines.is_some();
    #[cfg(not(feature = "splines"))]
    let has_splines = false;
    write_frame_header_kind(
        distp.x_qm_scale,
        ctx.b_qm_scale(),
        distp.epf_iters,
        distp.epf_pass0_scale,
        distp.epf_quant_mul,
        distp.gab_enabled,
        alpha.is_some(),
        coeff_shifts,
        frame_kind,
        has_splines,
        noise.is_some(),
        skip_dc_smoothing,
        use_dc_frame,
        ctx.coding,
        ctx.epf_channel_scale,
        ctx.frame_role,
        writer,
    );
    let payload_bits = dc_frame_bits
        + sections
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 0 && *i != 1 + dim.num_dc_groups)
            .map(|(_, section)| section.bits_written())
            .sum::<usize>();
    let order = (ctx.group_order != crate::GroupOrder::Scanline
        && matches!(
            frame_kind,
            VarDctFrameKind::Regular | VarDctFrameKind::Patched(_)
        ))
    .then(|| group_section_order(&dim, num_passes, ctx.group_order));
    combine_sections_ordered(
        &mut sections,
        order.as_deref().map(|o| (o, &mut scratch.huffman_pool)),
        writer,
    );
    Ok(payload_bits)
}
