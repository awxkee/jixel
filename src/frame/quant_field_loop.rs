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

//! ExtraSlow quant-field, transform-layout and chroma refinement against the decoder metric.

use super::ac_groups::{PendingAcGroup, RefinedAcGroup, process_ac_group};
use super::loop_arbiter::LoopArbiter;
use super::loop_group_map::LoopGroupMap;
use super::quantization::DistanceParams;
use super::{
    ImageDim, K_BLOCK_DIM, K_DC_GROUP_DIM, K_GROUP_DIM, K_GROUP_DIM_IN_BLOCKS, K_TILE_DIM_IN_BLOCKS,
};
use crate::Arbiter;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::{DcGroupData, STRATEGY_DCT};
use crate::encoding_context::EncodingContext;
use crate::entropy::TokenSlice;
use crate::image::{Image3F, Image3S};

/// Closed-loop quant-field refinement (ExtraSlow). Every group is re-quantized
/// the way pass 2 codes it (custom scan orders, RDOQ at the frozen token
/// prices), priced with those prices and reconstructed in luma (source plus
/// the inverse-transformed Y quantization error, the rerank's model). The
/// whole field moved one step coarser and one step finer gives the global
/// exchange rate (SSIMULACRA2 deficit per percent of bits); the finer shift's
/// deficit change is attributed to 32px tiles in proportion to their share of
/// the current linearized map, against each tile's bit change. Tiles whose
/// return beats the global rate are candidates, verified one by one on the
/// arbiter's exact metric and kept only if `-deficit - slope * log rate`
/// improves.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_quant_field_loop(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    dc_datas: &mut [DcGroupData],
    ac_tasks: &[(usize, usize, usize)],
    group_coords: &[(usize, usize)],
    num_passes: usize,
    coeff_shifts: &[u32],
    coeff_orders: &crate::coeff_order::CoeffOrders,
    order_alternatives: &[crate::coeff_order::CoeffOrders],
    ytob_dc: i32,
    qf_threshold: u32,
    prices: &crate::entropy::FrozenTokenPrices,
    want_recon: bool,
) -> Option<(Vec<RefinedAcGroup>, DistanceParams, Option<Image3F>)> {
    /// Marginals and slope come from plain quantization; the ratio of deficit
    /// to bits is what the trellis preserves, at a third of the cost.
    const SHIFT_RDOQ: bool = false;
    /// The mean of the finer and coarser chords overstates the tangent of a
    /// convex rate-distortion curve.
    const SLOPE_SCALE: f32 = 0.85;
    const TILE_BLOCKS: usize = 4;
    const MAX_ITERS: usize = 3;
    // Candidates verified exactly per iteration; with the block memo each
    // costs ~0.4 ms at 1 MP, and 300 (vs 40) buys SS2 −0.3 / cvvdp −0.15 for +0.27×.
    const GREEDY: usize = 300;
    // At most this many candidates per AC group per iteration: each extra
    // candidate in a group costs another serial round (one group encode).
    const PER_GROUP: usize = 12;
    const MAX_MOVE_FRACTION: f32 = 0.3;
    const MAX_ABS_MOVE: i8 = 2;
    const MIN_FIELD: u8 = 1;

    let (w, h) = (dim.xsize, dim.ysize);
    let (xb, yb) = (w.div_ceil(K_BLOCK_DIM), h.div_ceil(K_BLOCK_DIM));
    if xb < TILE_BLOCKS || yb < TILE_BLOCKS {
        return None;
    }
    let reference = crate::ssimulacra2::Ssimulacra2Reference::new(opsin)?;
    let blocks_per_dc = K_DC_GROUP_DIM / K_BLOCK_DIM;

    let group_map = LoopGroupMap::new(w, h, group_coords, ac_tasks);
    let locate = |bx, by| group_map.locate(bx, by);
    let task_of_block = &group_map.task_of_block;

    #[derive(Clone)]
    struct Eval {
        recon: Image3F,
        pending: Vec<Option<PendingAcGroup>>,
        quant_dc: Vec<Option<Image3S>>,
        /// Unrounded DC image values of each task's blocks.
        source_dc: Vec<Option<crate::group::SourceDc>>,
        starts: Vec<Vec<(u16, u16, u32)>>,
        /// Per task, the memo evaluation `recon` was last synced with.
        synced: Vec<u32>,
        /// Trial evaluations take cached blocks' bits instead of their tokens.
        trial: bool,
        hit_bits: Vec<f32>,
    }
    // Re-quantize, price and reconstruct the given AC groups under the fields.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        opsin: &Image3F,
        dim: &ImageDim,
        distp: &DistanceParams,
        dc_datas: &[DcGroupData],
        ac_tasks: &[(usize, usize, usize)],
        group_coords: &[(usize, usize)],
        num_passes: usize,
        coeff_shifts: &[u32],
        coeff_orders: &crate::coeff_order::CoeffOrders,
        order_alternatives: &[crate::coeff_order::CoeffOrders],
        ytob_dc: i32,
        qf_threshold: u32,
        prices: Option<&crate::entropy::FrozenTokenPrices>,
        tasks: &[usize],
        memo: Option<&[std::sync::Mutex<Option<crate::group::AcBlockCache>>]>,
        eval: &mut Eval,
    ) -> Vec<crate::ssimulacra2::Rect> {
        let eval_trial = eval.trial;
        let results = ctx
            .thread_pool
            .steal_map(scratch, tasks.len(), |i, scratch| {
                let t = tasks[i];
                let mut memo_slot = memo.map(|m| m[t].lock().expect("block memo"));
                if let Some(Some(m)) = memo_slot.as_deref_mut() {
                    m.trial = eval_trial;
                    m.trial_bits = 0.0;
                }
                let (dc_idx, gx, gy) = ac_tasks[t];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (p, local, source, _, capture) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    &dc_datas[dc_idx],
                    num_passes,
                    coeff_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    ytob_dc,
                    prices,
                    coeff_orders,
                    order_alternatives,
                    false,
                    qf_threshold,
                    true,
                    memo_slot.as_deref_mut(),
                );
                let (plane, starts) = capture.expect("recon capture");
                let hit_bits = match memo_slot.as_deref_mut() {
                    Some(Some(m)) => {
                        m.trial = false;
                        m.trial_bits as f32
                    }
                    _ => 0.0,
                };
                (
                    t,
                    p,
                    local,
                    source,
                    starts,
                    plane,
                    (dc_gx, dc_gy, gx, gy),
                    hit_bits,
                )
            });
        let mut dirty = Vec::with_capacity(results.len());
        for (t, p, local, source, starts, plane, (dc_gx, dc_gy, gx, gy), hit_bits) in results {
            eval.hit_bits[t] = hit_bits;
            eval.pending[t] = Some(p);
            eval.quant_dc[t] = Some(local);
            eval.source_dc[t] = source;
            eval.starts[t] = starts;
            let x0 = dc_gx * K_DC_GROUP_DIM + gx * K_GROUP_DIM;
            let y0 = dc_gy * K_DC_GROUP_DIM + gy * K_GROUP_DIM;
            let empty = crate::ssimulacra2::Rect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            };
            let rect = match memo {
                Some(memo) => {
                    // Only the cells recomputed since this evaluation last
                    // synced with the memo differ from `eval.recon`.
                    let guard = memo[t].lock().expect("block memo");
                    let m = guard.as_ref().expect("memo filled by the evaluation");
                    let rect = m.sync_into(eval.synced[t], &mut eval.recon, x0, y0).map_or(
                        empty,
                        |(rx0, ry0, rx1, ry1)| crate::ssimulacra2::Rect {
                            x0: rx0,
                            y0: ry0,
                            x1: rx1,
                            y1: ry1,
                        },
                    );
                    eval.synced[t] = m.epoch;
                    rect
                }
                None => {
                    for c in 0..3 {
                        for y in 0..plane.ysize() {
                            let src = &plane.plane_row(c, y)[..plane.xsize()];
                            eval.recon.plane_row_mut(c, y0 + y)[x0..x0 + src.len()]
                                .copy_from_slice(src);
                        }
                    }
                    crate::ssimulacra2::Rect {
                        x0,
                        y0,
                        x1: x0 + plane.xsize(),
                        y1: y0 + plane.ysize(),
                    }
                }
            };
            dirty.push(rect);
        }
        dirty
    }
    let new_eval = || Eval {
        recon: Image3F::new(w, h),
        pending: vec![None; ac_tasks.len()],
        quant_dc: vec![None; ac_tasks.len()],
        source_dc: vec![None; ac_tasks.len()],
        starts: vec![Vec::new(); ac_tasks.len()],
        synced: vec![0; ac_tasks.len()],
        trial: false,
        hit_bits: vec![0.0; ac_tasks.len()],
    };
    // Trials overwrite a task's tokens and starts before reading them. Only
    // pixels, DC and memo sync points are needed for untouched tasks.
    let trial_eval = |e: &Eval| Eval {
        recon: e.recon.clone(),
        pending: vec![None; ac_tasks.len()],
        quant_dc: e.quant_dc.clone(),
        source_dc: e.source_dc.clone(),
        starts: vec![Vec::new(); ac_tasks.len()],
        synced: e.synced.clone(),
        trial: false,
        hit_bits: vec![0.0; ac_tasks.len()],
    };
    let fill_decoded_dc =
        |e: &Eval, dp: &DistanceParams, rect: crate::ssimulacra2::Rect, out: &mut Image3F| {
            use crate::quant_weights::INV_DC_QUANT;
            let steps: [f32; 3] =
                std::array::from_fn(|c| dp.dc_step[c] / (INV_DC_QUANT[c] * dp.scale_dc));
            let cfl = ctx.cfl_frame();
            let cfl_b = crate::color_correlation::dc_cfl_factor(dp.dc_step, ytob_dc, cfl);
            let cfl_x = crate::color_correlation::dc_cfl_factor_x(dp.dc_step, cfl.base_x);
            group_map.reconstruct_dc(
                &e.quant_dc,
                [cfl_x, cfl_b],
                steps,
                ctx.dc_rows.reconstruct,
                rect,
                out,
            );
            // The judge adds each cell's DC error to the AC-only reconstruction:
            // decoded minus the unrounded DC image value of the same transform.
            let (bx0, by0) = (rect.x0 / K_BLOCK_DIM, rect.y0 / K_BLOCK_DIM);
            let (bx1, by1) = (
                rect.x1.div_ceil(K_BLOCK_DIM).min(xb),
                rect.y1.div_ceil(K_BLOCK_DIM).min(yb),
            );
            for by in by0..by1 {
                for bx in bx0..bx1 {
                    let t = task_of_block[by * xb + bx];
                    let Some(source) = e.source_dc[t].as_ref() else {
                        continue;
                    };
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    let (dc_gx, dc_gy) = group_coords[dc_idx];
                    let lx = bx - (dc_gx * blocks_per_dc + gx * K_GROUP_DIM_IN_BLOCKS);
                    let ly = by - (dc_gy * blocks_per_dc + gy * K_GROUP_DIM_IN_BLOCKS);
                    for (c, plane) in [&source.x, &source.y, &source.b].into_iter().enumerate() {
                        out.plane_row_mut(c, by)[bx] -= plane.row(ly)[lx];
                    }
                }
            }
        };
    let decoded_dc = |e: &Eval, dp: &DistanceParams| -> Image3F {
        let mut out = Image3F::new(xb, yb);
        fill_decoded_dc(e, dp, crate::ssimulacra2::Rect::new(0, 0, w, h), &mut out);
        out
    };
    // The coefficient polish's ColorVideoVDP judge keeps the decoded DC minus
    // block means it was tuned with. Judging coefficient moves on the exact
    // DC error loses 2-3 % cvvdp at matched rate (8/8), unlike the quant-field
    // loop, which gains 1-3 % on every metric from it.
    let legacy_dc = |e: &Eval, dp: &DistanceParams| -> Image3F {
        use crate::quant_weights::INV_DC_QUANT;
        let mut out = Image3F::new(xb, yb);
        let steps: [f32; 3] =
            std::array::from_fn(|c| dp.dc_step[c] / (INV_DC_QUANT[c] * dp.scale_dc));
        let cfl = ctx.cfl_frame();
        let cfl_b = crate::color_correlation::dc_cfl_factor(dp.dc_step, ytob_dc, cfl);
        let cfl_x = crate::color_correlation::dc_cfl_factor_x(dp.dc_step, cfl.base_x);
        group_map.reconstruct_dc(
            &e.quant_dc,
            [cfl_x, cfl_b],
            steps,
            ctx.dc_rows.reconstruct,
            crate::ssimulacra2::Rect::new(0, 0, w, h),
            &mut out,
        );
        for c in 0..3 {
            for by in 0..yb {
                let rows = by * K_BLOCK_DIM..(by * K_BLOCK_DIM + K_BLOCK_DIM).min(h);
                let n_rows = rows.len() as f32;
                let row = out.plane_row_mut(c, by);
                for (bx, v) in row[..xb].iter_mut().enumerate() {
                    let cols = bx * K_BLOCK_DIM..(bx * K_BLOCK_DIM + K_BLOCK_DIM).min(w);
                    let n = n_rows * cols.len() as f32;
                    let sum: f32 = rows
                        .clone()
                        .map(|y| opsin.plane_row(c, y)[cols.clone()].iter().sum::<f32>())
                        .sum();
                    *v -= sum / n;
                }
            }
        }
        out
    };
    // Small trials reuse one plane and reconstruct only cells the judge reads.
    let mut trial_dc = None;
    let frame_bits = |e: &Eval| {
        prices.adaptive_bits(e.pending.iter().map(|p| {
            p.as_ref()
                .map_or(TokenSlice::Wide(&[]), |p| p.tokens[0].as_slice())
        }))
    };
    // Frozen-price bits of every first block, indexed by frame block.
    let block_bits = |e: &Eval| -> Vec<f32> {
        let mut out = vec![0f32; xb * yb];
        for (t, starts) in e.starts.iter().enumerate() {
            let (dc_gx, dc_gy) = group_coords[ac_tasks[t].0];
            let toks = e.pending[t]
                .as_ref()
                .map_or(TokenSlice::Wide(&[]), |p| p.tokens[0].as_slice());
            for (i, &(lbx, lby, start)) in starts.iter().enumerate() {
                let end = starts.get(i + 1).map_or(toks.len(), |s| s.2 as usize);
                let bits: f32 = toks
                    .slice(start as usize..end)
                    .iter()
                    .map(|k| prices.token_bits(k))
                    .sum();
                let bx = dc_gx * blocks_per_dc + lbx as usize;
                let by = dc_gy * blocks_per_dc + lby as usize;
                out[by * xb + bx] += bits;
            }
        }
        out
    };

    let (tiles_x, tiles_y) = (xb.div_ceil(TILE_BLOCKS), yb.div_ceil(TILE_BLOCKS));
    let n_tiles = tiles_x * tiles_y;
    let tile_of_block = |bx: usize, by: usize| (by / TILE_BLOCKS) * tiles_x + bx / TILE_BLOCKS;
    let tile_blocks = |t: usize| -> Vec<(usize, usize)> {
        let (tx, ty) = (t % tiles_x, t / tiles_x);
        let mut v = Vec::new();
        for by in ty * TILE_BLOCKS..((ty + 1) * TILE_BLOCKS).min(yb) {
            for bx in tx * TILE_BLOCKS..((tx + 1) * TILE_BLOCKS).min(xb) {
                v.push((bx, by));
            }
        }
        v
    };
    // Sum a per-pixel map over each tile.
    let tile_sums = |map: &[f32]| -> Vec<f64> {
        let mut out = vec![0f64; n_tiles];
        let tile_px = TILE_BLOCKS * K_BLOCK_DIM;
        for y in 0..h {
            let ty = y / tile_px;
            let row = &map[y * w..(y + 1) * w];
            for (x, &v) in row.iter().enumerate() {
                out[ty * tiles_x + x / tile_px] += v as f64;
            }
        }
        out
    };
    let tile_bits = |blk: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; n_tiles];
        for by in 0..yb {
            for bx in 0..xb {
                out[tile_of_block(bx, by)] += blk[by * xb + bx];
            }
        }
        out
    };
    /// Fields a move overwrote, for an exact restore (the clamp at the bounds
    /// makes the inverse step inexact).
    struct Saved(Vec<(usize, usize, usize, u8)>);
    // Apply `moves` (tile, step) to the fields; returns the touched AC tasks
    // and the previous values.
    let apply = |dc_datas: &mut [DcGroupData], moves: &[(usize, i8)]| -> (Vec<usize>, Saved) {
        let mut tasks = Vec::new();
        let mut saved = Vec::new();
        for &(t, step) in moves {
            for (bx, by) in tile_blocks(t) {
                let (dc_idx, lbx, lby) = locate(bx, by);
                let v = &mut dc_datas[dc_idx].raw_quant_field.row_mut(lby)[lbx];
                saved.push((dc_idx, lbx, lby, *v));
                *v = (*v as i16 + step as i16).clamp(MIN_FIELD as i16, 255) as u8;
                let task = task_of_block[by * xb + bx];
                if !tasks.contains(&task) {
                    tasks.push(task);
                }
            }
        }
        (tasks, Saved(saved))
    };
    let restore = |dc_datas: &mut [DcGroupData], saved: Saved| {
        for (dc_idx, lbx, lby, v) in saved.0 {
            dc_datas[dc_idx].raw_quant_field.row_mut(lby)[lbx] = v;
        }
    };
    let all_tasks: Vec<usize> = (0..ac_tasks.len()).collect();
    let all_moves = |step: i8| -> Vec<(usize, i8)> { (0..n_tiles).map(|t| (t, step)).collect() };
    // Per-group block memo shared by every priced evaluation of the loop.
    let memo: Vec<std::sync::Mutex<Option<crate::group::AcBlockCache>>> = (0..ac_tasks.len())
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    let memo: Option<&[_]> = Some(&memo[..]);
    // Only between trials: layout snapshots retain offsets into these arenas.
    let compact_memos = |tasks: &[usize]| {
        if let Some(memo) = memo {
            for &t in tasks {
                if let Some(m) = memo[t].lock().expect("block memo").as_mut() {
                    m.compact_tokens();
                }
            }
        }
    };
    let mut cur = new_eval();
    evaluate(
        ctx,
        scratch,
        opsin,
        dim,
        distp,
        dc_datas,
        ac_tasks,
        group_coords,
        num_passes,
        coeff_shifts,
        coeff_orders,
        order_alternatives,
        ytob_dc,
        qf_threshold,
        Some(prices),
        &all_tasks,
        memo,
        &mut cur,
    );
    let ref_bits = frame_bits(&cur);
    let finish = |e: Eval,
                  distp: DistanceParams,
                  recon: Option<Image3F>|
     -> Option<(Vec<RefinedAcGroup>, DistanceParams, Option<Image3F>)> {
        Some((
            e.pending
                .into_iter()
                .zip(e.quant_dc)
                .enumerate()
                .map(|(t, (p, q))| {
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    (
                        dc_idx,
                        gx,
                        gy,
                        p.expect("pending group"),
                        q.expect("group dc"),
                    )
                })
                .collect(),
            distp,
            recon,
        ))
    };
    if ref_bits <= 0.0 {
        let recon = want_recon.then(|| std::mem::replace(&mut cur.recon, Image3F::new(0, 0)));
        return finish(cur, *distp, recon);
    }
    let rate_term = |bits: f32| 100.0 * (bits / ref_bits).ln();
    let mut cur_bits = ref_bits;
    // The incremental state follows every accepted move exactly. Defer the
    // SSIM state until after the full-frame marginal scores: their temporary
    // images would otherwise overlap its retained reconstruction pyramids.
    // Under the ColorVideoVDP arbiter it is also the exact metric; the
    // marginals that rank candidates still come from SSIMULACRA2.
    let cv_judge = (ctx.arbiter == Arbiter::Cvvdp && ctx.coding.is_xyb())
        .then(|| crate::cvvdp_arbiter::CvvdpArbiter::new(opsin))
        .flatten();
    let mut loop_state = cv_judge.map(|mut judge| {
        judge.reset(&cur.recon, &decoded_dc(&cur, distp));
        LoopArbiter::Cvvdp(Box::new(judge))
    });
    if SHIFT_RDOQ && loop_state.is_none() {
        loop_state = Some(LoopArbiter::Ssimulacra2(Box::new(
            reference.state(&cur.recon),
        )));
    }
    let mut cur_d = loop_state.as_ref().map_or(f32::NAN, LoopArbiter::deficit);
    let mut cur_map = if SHIFT_RDOQ {
        loop_state.as_ref().expect("shift judge").map()
    } else {
        Vec::new()
    };
    let mut moved = vec![0i8; n_tiles];

    let mut cached_slope = 0f32;
    // Under the ColorVideoVDP arbiter its own exchange rate, from the same
    // shift evaluations, prices every decision it arbitrates.
    let mut cv_slope = 0f32;
    // The coefficient polish also judges on ColorVideoVDP; its exchange rate
    // comes from the same uniform shifts as the loop's.
    let polish_wanted = ctx.speed.effort().coef_polish()
        && ctx.arbiter == Arbiter::Ssimulacra2
        && ctx.coding.is_xyb();
    let mut polish_cv = polish_wanted
        .then(|| crate::cvvdp_arbiter::CvvdpArbiter::new(opsin))
        .flatten();
    let mut polish_cv_slope = 0f32;
    let mut cached_moves: Vec<(f32, usize, i8)> = Vec::new();
    let mut accepted_moves: Vec<(usize, i8)> = Vec::new();
    // Moves the exact arbiter already rejected are not tried again: later
    // iterations rejected them again at the cost of half the evaluations.
    let mut rejected_moves: std::collections::HashSet<(usize, i8)> = Default::default();
    // Allocate the greedy scratch after the full-frame marginal evaluations.
    let mut greedy_eval = None;
    for iter in 0..MAX_ITERS {
        // Later iterations reuse the first iteration's marginals: the greedy
        // pass arbitrates every candidate exactly, so only the ranking is stale.
        let (slope, moves): (f32, Vec<(f32, usize, i8)>) = if iter > 0 {
            let moves: Vec<(f32, usize, i8)> = cached_moves
                .iter()
                .filter(|&&(_, t, step)| {
                    if step > 0 {
                        moved[t] < MAX_ABS_MOVE
                    } else {
                        moved[t] > -MAX_ABS_MOVE
                    }
                })
                .filter(|&&(_, t, step)| !accepted_moves.contains(&(t, step)))
                .filter(|&&(_, t, step)| !rejected_moves.contains(&(t, step)))
                .copied()
                .collect();
            (cached_slope, moves)
        } else {
            // Marginals and exchange rate from a plain-quantization triple: the
            // current field, the whole field one step finer, and every other
            // group one step coarser (half an evaluation; the ratio of two
            // half-frame deltas is unbiased). The one-sided finer chord
            // undervalues the tangent of a convex curve, so the two chords are
            // averaged.
            let plain = if SHIFT_RDOQ {
                cur.clone()
            } else {
                let mut e = new_eval();
                evaluate(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    dc_datas,
                    ac_tasks,
                    group_coords,
                    num_passes,
                    coeff_shifts,
                    coeff_orders,
                    &[],
                    ytob_dc,
                    qf_threshold,
                    None,
                    &all_tasks,
                    None,
                    &mut e,
                );
                e
            };
            let r_p = rate_term(frame_bits(&plain));
            let shift_prices = SHIFT_RDOQ.then_some(prices);
            let (_, saved) = apply(dc_datas, &all_moves(1));
            let mut fine = new_eval();
            evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                &[],
                ytob_dc,
                qf_threshold,
                shift_prices,
                &all_tasks,
                None,
                &mut fine,
            );
            restore(dc_datas, saved);
            let r_f = rate_term(frame_bits(&fine));
            if r_f - r_p < 1e-3 {
                break;
            }
            // Score one full-frame workspace at a time. Retain only tile
            // summaries, then reuse the plain evaluation for the coarse arm.
            // Each score and token reduction is identical to the independent
            // full-frame comparisons; only their storage lifetimes change.
            let (d_p, map_p) = if SHIFT_RDOQ {
                (cur_d, cur_map.clone())
            } else {
                reference.deficit_and_map(&plain.recon)
            };
            let map_0 = tile_sums(&map_p);
            drop(map_p);
            let d_f = reference.deficit(&fine.recon);
            let bits_f = tile_bits(&block_bits(&fine));
            let bits_0 = tile_bits(&block_bits(&plain));
            let cv_pf = loop_state
                .as_mut()
                .and_then(LoopArbiter::cvvdp)
                .map(|judge| {
                    [&plain, &fine].map(|e| judge.deficit_of(&e.recon, &decoded_dc(e, distp)))
                });
            let pcv_pf = polish_cv.as_mut().map(|judge| {
                [&plain, &fine].map(|e| judge.deficit_of(&e.recon, &legacy_dc(e, distp)))
            });
            drop(fine);
            // The coarser shift covers half the groups (slope estimate only).
            let half_tasks: Vec<usize> = (0..ac_tasks.len()).step_by(2).collect();
            let half_moves: Vec<(usize, i8)> = (0..n_tiles)
                .filter(|&t| {
                    let (bx, by) = tile_blocks(t)[0];
                    half_tasks.contains(&task_of_block[by * xb + bx])
                })
                .map(|t| (t, -1))
                .collect();
            let (_, saved) = apply(dc_datas, &half_moves);
            let mut coarse = plain;
            evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                &[],
                ytob_dc,
                qf_threshold,
                shift_prices,
                &half_tasks,
                None,
                &mut coarse,
            );
            restore(dc_datas, saved);
            let r_c = rate_term(frame_bits(&coarse));
            let d_c = reference.deficit(&coarse.recon);
            let s_fine = (d_p - d_f) / (r_f - r_p);
            let s_coarse = if r_p - r_c > 1e-3 {
                (d_c - d_p) / (r_p - r_c)
            } else {
                s_fine
            };
            let slope = SLOPE_SCALE * 0.5 * (s_fine + s_coarse);
            if !(slope.is_finite() && slope > 0.0) {
                break;
            }
            let mut cv_flat = false;
            if let Some([cv_p, cv_f]) = cv_pf {
                let judge = loop_state
                    .as_mut()
                    .and_then(LoopArbiter::cvvdp)
                    .expect("CV judge");
                let cv_c = judge.deficit_of(&coarse.recon, &decoded_dc(&coarse, distp));
                let s_fine = (cv_p - cv_f) / (r_f - r_p);
                let s_coarse = if r_p - r_c > 1e-3 {
                    (cv_c - cv_p) / (r_p - r_c)
                } else {
                    s_fine
                };
                cv_slope = SLOPE_SCALE * 0.5 * (s_fine + s_coarse);
                cv_flat = !(cv_slope.is_finite() && cv_slope > 0.0);
            }
            if let (Some([cv_p, cv_f]), Some(judge)) = (pcv_pf, polish_cv.as_mut()) {
                let cv_c = judge.deficit_of(&coarse.recon, &legacy_dc(&coarse, distp));
                let s_fine = (cv_p - cv_f) / (r_f - r_p);
                let s_coarse = if r_p - r_c > 1e-3 {
                    (cv_c - cv_p) / (r_p - r_c)
                } else {
                    s_fine
                };
                polish_cv_slope = SLOPE_SCALE * 0.5 * (s_fine + s_coarse);
            }
            drop(coarse);
            // Near transparency ColorVideoVDP no longer separates the shifts;
            // SSIMULACRA2 arbitrates the frame instead of the loop stopping.
            if cv_flat {
                cv_slope = 0.0;
                let state = LoopArbiter::Ssimulacra2(Box::new(reference.state(&cur.recon)));
                cur_d = state.deficit();
                loop_state = Some(state);
            }
            // Per tile: deficit gain (frame units) minus slope times rate cost (%).
            // Only finer moves: coarsening smooth tiles is a blind spot of mean
            // MS-SSIM that SSIMULACRA2 punishes, and edge-difference terms did not
            // close it.
            let mut moves: Vec<(f32, usize, i8)> = Vec::new();
            // The frame's measured deficit change is attributed to tiles in
            // proportion to their share of the current linearized map. The
            // per-tile differences of the finer map carry pooling cross-talk
            // from neighbouring tiles and rank worse.
            let sum_map_0: f64 = map_0.iter().sum();
            if sum_map_0 <= 0.0 {
                break;
            }
            for t in 0..n_tiles {
                let gain = ((map_0[t] / sum_map_0) * f64::from(d_p - d_f)) as f32;
                let rate = 100.0 * (bits_f[t] - bits_0[t]) / ref_bits;
                let net = gain - slope * rate;
                if net > 1e-7 && moved[t] < MAX_ABS_MOVE {
                    moves.push((net, t, 1));
                }
            }
            moves.sort_by(|a, b| b.0.total_cmp(&a.0));
            moves.truncate(((MAX_MOVE_FRACTION * n_tiles as f32) as usize).max(1));
            cached_slope = slope;
            cached_moves = moves.clone();
            (slope, moves)
        };
        if loop_state.is_none() {
            let state = LoopArbiter::Ssimulacra2(Box::new(reference.state(&cur.recon)));
            cur_d = state.deficit();
            loop_state = Some(state);
        }
        if moves.is_empty() {
            break;
        }
        // Pixels and DC follow `cur`; evaluated tasks replace their tokens.
        let cand = greedy_eval.get_or_insert_with(|| trial_eval(&cur));
        // Moves are ranked on SSIMULACRA2 marginals and accepted on the judge's.
        let slope = if loop_state.as_ref().is_some_and(LoopArbiter::needs_dc) {
            cv_slope
        } else {
            slope
        };
        let mut j_cur = -cur_d - slope * rate_term(cur_bits);
        let mut accepted = false;
        // Evaluate the top candidates one at a time with the real metric.
        // The state follows every accepted move exactly, so it is built once.
        let state = loop_state.as_mut().expect("loop judge");
        // Tokens and starts are overwritten by the next evaluation. Restore
        // only DC and pixels on rejection; move the token buffers on acceptance.
        let sync_tasks =
            |from: &Eval, to: &mut Eval, tasks: &[usize], r: crate::ssimulacra2::Rect| {
                for &task in tasks {
                    to.quant_dc[task] = from.quant_dc[task].clone();
                    to.source_dc[task] = from.source_dc[task].clone();
                    to.synced[task] = from.synced[task];
                }
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                            .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                    }
                }
            };
        let group_bits = |e: &Eval, task: usize| -> f32 {
            e.pending[task].as_ref().map_or(0.0, |p| {
                p.tokens[0].iter().map(|k| prices.token_bits(k)).sum()
            })
        };
        // The incumbent only changes on acceptance. Preserve each reduction's
        // exact f32 order, but do not repeat it for every rejected trial.
        let mut cur_gbits: Vec<f32> = (0..ac_tasks.len()).map(|t| group_bits(&cur, t)).collect();
        let mut cand_gbits = vec![0.0; ac_tasks.len()];
        // Rounds: candidates in distinct AC groups are re-encoded together in
        // one parallel pass, then arbitrated one by one with the exact metric.
        // At most `per_group` candidates per AC group per iteration: each extra
        // candidate in a group costs another serial round (one group encode).
        let mut per_group_count = vec![0usize; ac_tasks.len()];
        let mut aq_checkpoint = state.checkpoint();
        let mut remaining: Vec<(usize, i8)> = moves
            .iter()
            .take(GREEDY)
            .map(|m| (m.1, m.2))
            .filter(|&(t, _)| {
                let (bx, by) = tile_blocks(t)[0];
                let task = task_of_block[by * xb + bx];
                per_group_count[task] += 1;
                per_group_count[task] <= PER_GROUP
            })
            .collect();
        while !remaining.is_empty() {
            let mut round: Vec<(usize, i8, Vec<usize>, Saved)> = Vec::new();
            let mut used_tasks: Vec<usize> = Vec::new();
            let mut deferred = Vec::new();
            for &(t, step) in &remaining {
                let (bx, by) = tile_blocks(t)[0];
                let task = task_of_block[by * xb + bx];
                if used_tasks.contains(&task) {
                    deferred.push((t, step));
                    continue;
                }
                let (tasks, saved) = apply(dc_datas, &[(t, step)]);
                used_tasks.extend(tasks.iter().copied());
                round.push((t, step, tasks, saved));
            }
            remaining = deferred;
            let round_tasks: Vec<usize> = round.iter().flat_map(|r| r.2.iter().copied()).collect();
            compact_memos(&round_tasks);
            let dirty = evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &round_tasks,
                memo,
                cand,
            );
            for (t, step, tasks, saved) in round {
                let mut region = crate::ssimulacra2::Rect {
                    x0: usize::MAX,
                    y0: usize::MAX,
                    x1: 0,
                    y1: 0,
                };
                let mut delta_bits = 0f32;
                for &task in &tasks {
                    // pixels the re-encode changed: the cells it recomputed
                    let r = dirty[round_tasks
                        .iter()
                        .position(|&x| x == task)
                        .expect("round task")];
                    if r.x1 > r.x0 {
                        region.x0 = region.x0.min(r.x0);
                        region.y0 = region.y0.min(r.y0);
                        region.x1 = region.x1.max(r.x1);
                        region.y1 = region.y1.max(r.y1);
                    }
                    cand_gbits[task] = group_bits(cand, task);
                    delta_bits += cand_gbits[task] - cur_gbits[task];
                }
                let d = if region.x1 > 0 {
                    let dc = state.needs_dc().then(|| {
                        let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                        fill_decoded_dc(cand, distp, region, out);
                        &*out
                    });
                    state.update_checkpointed(&cand.recon, dc, region, &mut aq_checkpoint)
                } else {
                    cur_d
                };
                let bits = cur_bits + delta_bits;
                let j = -d - slope * rate_term(bits);
                let region = if region.x1 > 0 {
                    region
                } else {
                    crate::ssimulacra2::Rect {
                        x0: 0,
                        y0: 0,
                        x1: 0,
                        y1: 0,
                    }
                };
                if j > j_cur + 1e-7 {
                    moved[t] += step;
                    accepted_moves.push((t, step));
                    for &task in &tasks {
                        std::mem::swap(&mut cand.pending[task], &mut cur.pending[task]);
                        std::mem::swap(&mut cand.starts[task], &mut cur.starts[task]);
                        cur_gbits[task] = cand_gbits[task];
                    }
                    sync_tasks(cand, &mut cur, &tasks, region);
                    cur_bits = bits;
                    cur_d = d;
                    j_cur = j;
                    accepted = true;
                } else {
                    if region.x1 > 0 {
                        state.rollback(&aq_checkpoint);
                    }
                    rejected_moves.insert((t, step));
                    sync_tasks(&cur, cand, &tasks, region);
                    restore(dc_datas, saved);
                }
            }
        }
        // exact frame bits after the round's approximate per-group deltas
        cur_bits = frame_bits(&cur);
        if SHIFT_RDOQ {
            cur_map = state.map();
        }
        if !accepted {
            break;
        }
    }
    drop(greedy_eval);
    drop(cur_map);
    // ---- 128-family placement arbitrated by the loop's target ----
    // Each 128x128 site tries the 128 layouts (rect halves, rect pairs, the
    // square) on top of the current frame, re-encoded exactly (block memo,
    // frozen prices) and arbitrated by the incremental SSIMULACRA2 state at the
    // loop's exchange rate. One site per AC group is encoded per parallel round.
    if ctx.speed.effort().dct128_loop() && cached_slope > 0.0 {
        use crate::dc_group_data::{
            AcStrategyImage, STRATEGY_DCT64X128, STRATEGY_DCT128X64, STRATEGY_DCT128X128,
        };
        /// Incumbent bits per pixel of a variant's region above which no 128
        /// layout paid.
        const MAX_INCUMBENT_BPP: f32 = 1.0;
        // (strategy, dx, dy) parts of each variant, site-relative in blocks,
        // and the region (x, y, w, h) it covers. Halves come first: a pair is
        // tried only if both its halves gain, the square only if any half does.
        type Variant = (&'static [(u8, usize, usize)], (usize, usize, usize, usize));
        const HALVES: [Variant; 4] = [
            (&[(STRATEGY_DCT128X64, 0, 0)], (0, 0, 8, 16)),
            (&[(STRATEGY_DCT128X64, 8, 0)], (8, 0, 8, 16)),
            (&[(STRATEGY_DCT64X128, 0, 0)], (0, 0, 16, 8)),
            (&[(STRATEGY_DCT64X128, 0, 8)], (0, 8, 16, 8)),
        ];
        const WHOLE: (usize, usize, usize, usize) = (0, 0, 16, 16);
        const PAIRS: [(Variant, [usize; 2]); 2] = [
            (
                (
                    &[(STRATEGY_DCT128X64, 0, 0), (STRATEGY_DCT128X64, 8, 0)],
                    WHOLE,
                ),
                [0, 1],
            ),
            (
                (
                    &[(STRATEGY_DCT64X128, 0, 0), (STRATEGY_DCT64X128, 0, 8)],
                    WHOLE,
                ),
                [2, 3],
            ),
        ];
        const SQUARE: Variant = (&[(STRATEGY_DCT128X128, 0, 0)], WHOLE);

        let slope = if loop_state.as_ref().is_some_and(LoopArbiter::needs_dc) {
            cv_slope
        } else {
            cached_slope
        };
        let state = loop_state.as_mut().expect("loop judge");
        let mut checkpoint = state.checkpoint();
        let mut cand = trial_eval(&cur);
        let mut j_cur = -cur_d - slope * rate_term(cur_bits);
        // Sites of each AC task, DC-group-local block coordinates.
        let sites: Vec<Vec<(usize, usize)>> = ac_tasks
            .iter()
            .map(|&(dc_idx, gx, gy)| {
                let strategy = &dc_datas[dc_idx].ac_strategy;
                let mut v = Vec::new();
                for sy in [0, 16] {
                    for sx in [0, 16] {
                        let x = gx * K_GROUP_DIM_IN_BLOCKS + sx;
                        let y = gy * K_GROUP_DIM_IN_BLOCKS + sy;
                        if strategy.can_place_strategy(x, y, STRATEGY_DCT128X128) {
                            v.push((x, y));
                        }
                    }
                }
                v
            })
            .collect();
        /// What placing a variant overwrote: layout, field and memo of the site.
        struct Placed {
            task: usize,
            site: (usize, usize),
            cells: Vec<u8>,
            field: Vec<u8>,
            memo: Option<crate::group::MemoSnapshot>,
        }
        let distance = distp.distance;
        let place = |dc_datas: &mut [DcGroupData],
                     task: usize,
                     (x, y): (usize, usize),
                     parts: &[(u8, usize, usize)]|
         -> Placed {
            let (dc_idx, gx, gy) = ac_tasks[task];
            let data = &mut dc_datas[dc_idx];
            let cells = data.ac_strategy.save_region(x, y, 16, 16);
            let field: Vec<u8> = (y..y + 16)
                .flat_map(|r| data.raw_quant_field.row(r)[x..x + 16].to_vec())
                .collect();
            for &(s, dx, dy) in parts {
                let (px, py) = (x + dx, y + dy);
                data.ac_strategy.set_first(px, py, s);
                let w = AcStrategyImage::covered_blocks_x_of(s);
                let h = AcStrategyImage::covered_blocks_y_of(s);
                let (mut max_q, mut sum) = (0u8, 0u32);
                for r in py..py + h {
                    for &q in &data.raw_quant_field.row(r)[px..px + w] {
                        max_q = max_q.max(q);
                        sum += u32::from(q);
                    }
                }
                let q = crate::ac_strategy::aggregate_quant(max_q, sum, w * h, distance);
                for r in py..py + h {
                    data.raw_quant_field.row_mut(r)[px..px + w].fill(q);
                }
            }
            // The memo keys do not carry the layout: forget the new
            // transforms' cells, and keep the rest for an exact undo.
            let memo_snapshot = memo.and_then(|memo| {
                let mut guard = memo[task].lock().expect("block memo");
                let m = guard.as_mut()?;
                m.compact_tokens();
                let (lx, ly) = (
                    x - gx * K_GROUP_DIM_IN_BLOCKS,
                    y - gy * K_GROUP_DIM_IN_BLOCKS,
                );
                let snapshot = m.snapshot(lx, ly, 16, 16);
                for &(s, dx, dy) in parts {
                    m.invalidate(
                        lx + dx,
                        ly + dy,
                        AcStrategyImage::covered_blocks_x_of(s),
                        AcStrategyImage::covered_blocks_y_of(s),
                    );
                }
                Some(snapshot)
            });
            Placed {
                task,
                site: (x, y),
                cells,
                field,
                memo: memo_snapshot,
            }
        };
        let unplace = |dc_datas: &mut [DcGroupData], p: Placed| {
            let (dc_idx, _, _) = ac_tasks[p.task];
            let data = &mut dc_datas[dc_idx];
            let (x, y) = p.site;
            data.ac_strategy.restore_region(x, y, 16, &p.cells);
            for (i, row) in p.field.as_chunks::<16>().0.iter().enumerate() {
                data.raw_quant_field.row_mut(y + i)[x..x + 16].copy_from_slice(row);
            }
            if let (Some(memo), Some(snapshot)) = (memo, &p.memo)
                && let Some(m) = memo[p.task].lock().expect("block memo").as_mut()
            {
                m.restore(snapshot);
            }
        };
        let group_bits = |e: &Eval, task: usize| -> f32 {
            e.pending[task].as_ref().map_or(0.0, |p| {
                p.tokens[0].iter().map(|k| prices.token_bits(k)).sum()
            })
        };
        let mut cur_gbits: Vec<f32> = (0..ac_tasks.len()).map(|t| group_bits(&cur, t)).collect();
        let sync_task = |from: &Eval, to: &mut Eval, task: usize, r: crate::ssimulacra2::Rect| {
            to.quant_dc[task] = from.quant_dc[task].clone();
            to.source_dc[task] = from.source_dc[task].clone();
            to.synced[task] = from.synced[task];
            for c in 0..3 {
                for y in r.y0..r.y1 {
                    to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                        .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                }
            }
        };
        let mut accepted_sites: Vec<(usize, (usize, usize))> = Vec::new();
        let max_sites = sites.iter().map(Vec::len).max().unwrap_or(0);
        for si in 0..max_sites {
            let active: Vec<usize> = (0..ac_tasks.len())
                .filter(|&t| si < sites[t].len())
                .collect();
            let incumbent_bits = block_bits(&cur);
            let region_bpp = |t: usize, (rx, ry, rw, rh): (usize, usize, usize, usize)| {
                let (dc_gx, dc_gy) = group_coords[ac_tasks[t].0];
                let (x, y) = sites[t][si];
                let (x0, y0) = (
                    dc_gx * blocks_per_dc + x + rx,
                    dc_gy * blocks_per_dc + y + ry,
                );
                let bits: f32 = (y0..y0 + rh)
                    .map(|by| {
                        incumbent_bits[by * xb + x0..by * xb + x0 + rw]
                            .iter()
                            .sum::<f32>()
                    })
                    .sum();
                bits / (rw * rh * 64) as f32
            };
            // J of each half per task; the best variant and its J per task.
            let mut half_j = vec![[f32::NEG_INFINITY; 4]; ac_tasks.len()];
            let mut best: Vec<(f32, Option<Variant>)> = vec![(j_cur, None); ac_tasks.len()];
            // Evaluate `variant` at the tasks `keep` admits; returns each J.
            let mut try_variant = |dc_datas: &mut [DcGroupData],
                                   scratch: &mut CoderScratch,
                                   variant: Variant,
                                   keep: &dyn Fn(usize) -> bool|
             -> Vec<(usize, f32)> {
                let tasks: Vec<usize> = active
                    .iter()
                    .copied()
                    .filter(|&t| keep(t) && region_bpp(t, variant.1) <= MAX_INCUMBENT_BPP)
                    .collect();
                if tasks.is_empty() {
                    return Vec::new();
                }
                let placed: Vec<Placed> = tasks
                    .iter()
                    .map(|&t| place(dc_datas, t, sites[t][si], variant.0))
                    .collect();
                let dirty = evaluate(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    dc_datas,
                    ac_tasks,
                    group_coords,
                    num_passes,
                    coeff_shifts,
                    coeff_orders,
                    order_alternatives,
                    ytob_dc,
                    qf_threshold,
                    Some(prices),
                    &tasks,
                    memo,
                    &mut cand,
                );
                let mut out = Vec::with_capacity(tasks.len());
                for (k, &t) in tasks.iter().enumerate() {
                    let r = dirty[k];
                    let d = if r.x1 > r.x0 {
                        let dc = state.needs_dc().then(|| {
                            let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                            fill_decoded_dc(&cand, distp, r, out);
                            &*out
                        });
                        let d = state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint);
                        state.rollback(&checkpoint);
                        d
                    } else {
                        cur_d
                    };
                    let bits = cur_bits + group_bits(&cand, t) - cur_gbits[t];
                    out.push((t, -d - slope * rate_term(bits)));
                    sync_task(&cur, &mut cand, t, r);
                }
                for p in placed {
                    unplace(dc_datas, p);
                }
                out
            };
            for (h, half) in HALVES.iter().enumerate() {
                for (t, j) in try_variant(dc_datas, scratch, *half, &|_| true) {
                    half_j[t][h] = j;
                    if j > best[t].0 + 1e-7 {
                        best[t] = (j, Some(*half));
                    }
                }
            }
            for (pair, [a, b]) in PAIRS {
                let keep = |t: usize| half_j[t][a] > j_cur && half_j[t][b] > j_cur;
                for (t, j) in try_variant(dc_datas, scratch, pair, &keep) {
                    if j > best[t].0 + 1e-7 {
                        best[t] = (j, Some(pair));
                    }
                }
            }
            let keep = |t: usize| half_j[t].iter().any(|&j| j > j_cur);
            for (t, j) in try_variant(dc_datas, scratch, SQUARE, &keep) {
                if j > best[t].0 + 1e-7 {
                    best[t] = (j, Some(SQUARE));
                }
            }
            // Commit each task's winner, re-encode them together and accept
            // them one at a time on the running state.
            let winners: Vec<(usize, Variant)> = active
                .iter()
                .filter_map(|&t| best[t].1.map(|v| (t, v)))
                .collect();
            if winners.is_empty() {
                continue;
            }
            let placed: Vec<Placed> = winners
                .iter()
                .map(|&(t, v)| place(dc_datas, t, sites[t][si], v.0))
                .collect();
            let tasks: Vec<usize> = winners.iter().map(|w| w.0).collect();
            let dirty = evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &tasks,
                memo,
                &mut cand,
            );
            for ((k, &t), p) in tasks.iter().enumerate().zip(placed) {
                let r = dirty[k];
                let d = if r.x1 > r.x0 {
                    let dc = state.needs_dc().then(|| {
                        let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                        fill_decoded_dc(&cand, distp, r, out);
                        &*out
                    });
                    state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint)
                } else {
                    cur_d
                };
                let cand_gbits = group_bits(&cand, t);
                let bits = cur_bits + cand_gbits - cur_gbits[t];
                let j = -d - slope * rate_term(bits);
                if j > j_cur + 1e-7 {
                    std::mem::swap(&mut cand.pending[t], &mut cur.pending[t]);
                    std::mem::swap(&mut cand.starts[t], &mut cur.starts[t]);
                    sync_task(&cand, &mut cur, t, r);
                    cur_gbits[t] = cand_gbits;
                    cur_bits = bits;
                    cur_d = d;
                    j_cur = j;
                    accepted_sites.push((t, p.site));
                } else {
                    if r.x1 > r.x0 {
                        state.rollback(&checkpoint);
                    }
                    sync_task(&cur, &mut cand, t, r);
                    unplace(dc_datas, p);
                }
            }
        }
        // The chroma DC passes after the loop read the retained source DC,
        // which depends on the layout (a large transform's DC comes from its
        // lowest frequencies): recapture it under the accepted sites.
        let mut refresh: Vec<usize> = accepted_sites.iter().map(|a| a.0).collect();
        refresh.sort_unstable();
        refresh.dedup();
        let sources = ctx
            .thread_pool
            .steal_map(scratch, refresh.len(), |i, scratch| {
                let (dc_idx, gx, gy) = ac_tasks[refresh[i]];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (_, _, source, _, _) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    &dc_datas[dc_idx],
                    num_passes,
                    coeff_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    ytob_dc,
                    None,
                    coeff_orders,
                    &[],
                    true,
                    qf_threshold,
                    false,
                    None,
                );
                source.expect("source dc")
            });
        for (&task, source) in refresh.iter().zip(sources) {
            let (dc_idx, gx, gy) = ac_tasks[task];
            let (ox, oy) = (gx * K_GROUP_DIM_IN_BLOCKS, gy * K_GROUP_DIM_IN_BLOCKS);
            let dc = &mut dc_datas[dc_idx];
            for &(_, (x, y)) in accepted_sites.iter().filter(|a| a.0 == task) {
                for (target, src) in [
                    (&mut dc.source_dc_y, &source.y),
                    (&mut dc.source_dc_b, &source.b),
                    (&mut dc.source_dc_x, &source.x),
                ] {
                    if let Some(target) = target.as_mut() {
                        for r in y..y + 16 {
                            target.row_mut(r)[x..x + 16]
                                .copy_from_slice(&src.row(r - oy)[x - ox..x - ox + 16]);
                        }
                    }
                }
            }
        }
        cur_bits = frame_bits(&cur);
    }
    // ---- 16x16 layouts arbitrated by the loop's target ----
    // Every closed 2x2-block site (no transform crosses it) tries the other
    // layouts of its family: four DCT8, DCT16X16 and the two rectangle pairs.
    // Each is re-encoded exactly (block memo, frozen prices; cached blocks
    // contribute their stored bits instead of their tokens) and arbitrated by the
    // incremental SSIMULACRA2 state at the loop's exchange rate. One site per
    // AC group per parallel round; the best variant is re-encoded in full and
    // kept if it still improves.
    if ctx.speed.effort().layout_loop() && cached_slope > 0.0 {
        use crate::dc_group_data::{
            AcStrategyImage, NUM_STRATEGIES, STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16,
        };
        type Parts = Vec<(u8, usize, usize)>;
        let cv_judged = loop_state.as_ref().is_some_and(LoopArbiter::needs_dc);
        let slope = if cv_judged { cv_slope } else { cached_slope };
        let state = loop_state.as_mut().expect("loop judge");
        let mut checkpoint = state.checkpoint();
        let mut cand = trial_eval(&cur);
        let mut j_cur = -cur_d - slope * rate_term(cur_bits);
        // Only layouts whose classes the frame already uses (matrices, orders).
        let class = |st: usize| {
            (
                crate::quant_weights::quant_table_slot_of(st as u8),
                crate::ac_context::K_STRATEGY_ORDER
                    [crate::dc_group_data::STRATEGY_CODE_LUT[st] as usize],
            )
        };
        let mut used = [false; NUM_STRATEGIES];
        for d in dc_datas.iter() {
            for (_, _, st) in d.ac_strategy.iter_first_blocks() {
                used[st as usize] = true;
            }
        }
        // A transpose shares both classes with the transform the frame uses.
        let present: [bool; NUM_STRATEGIES] = std::array::from_fn(|st| {
            used[st]
                || class(st).0.is_some()
                    && (0..NUM_STRATEGIES).any(|o| used[o] && class(o) == class(st))
        });
        let mut variants: Vec<Parts> = vec![vec![
            (STRATEGY_DCT, 0, 0),
            (STRATEGY_DCT, 1, 0),
            (STRATEGY_DCT, 0, 1),
            (STRATEGY_DCT, 1, 1),
        ]];
        if present[STRATEGY_DCT16X16 as usize] {
            variants.push(vec![(STRATEGY_DCT16X16, 0, 0)]);
        }
        if present[STRATEGY_DCT16X8 as usize] {
            variants.push(vec![(STRATEGY_DCT16X8, 0, 0), (STRATEGY_DCT16X8, 1, 0)]);
        }
        if present[STRATEGY_DCT8X16 as usize] {
            variants.push(vec![(STRATEGY_DCT8X16, 0, 0), (STRATEGY_DCT8X16, 0, 1)]);
        }
        /// A region the loop re-chooses: a 2x2 site with the 16x16-family
        /// variants, or a region the exact re-tiling proposed, with that
        /// proposal as its only variant.
        struct Site {
            x: usize,
            y: usize,
            w: usize,
            h: usize,
            proposal: Option<Parts>,
        }
        // Sites of each AC task, DC-group-local block coordinates: the
        // aligned 2x2 grid, the three odd-offset grids (the format allows
        // unaligned blocks; none may cross a 64px CfL tile), then the
        // proposed regions, judged against the refined layout.
        let sites: Vec<Vec<Site>> = ac_tasks
            .iter()
            .map(|&(dc_idx, gx, gy)| {
                let st = &dc_datas[dc_idx].ac_strategy;
                let mut v = Vec::new();
                for (px, py) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    for sy in (py..K_GROUP_DIM_IN_BLOCKS - 1).step_by(2) {
                        for sx in (px..K_GROUP_DIM_IN_BLOCKS - 1).step_by(2) {
                            if sx % K_TILE_DIM_IN_BLOCKS == K_TILE_DIM_IN_BLOCKS - 1
                                || sy % K_TILE_DIM_IN_BLOCKS == K_TILE_DIM_IN_BLOCKS - 1
                            {
                                continue;
                            }
                            let x = gx * K_GROUP_DIM_IN_BLOCKS + sx;
                            let y = gy * K_GROUP_DIM_IN_BLOCKS + sy;
                            if x + 2 <= st.xsize() && y + 2 <= st.ysize() {
                                v.push(Site {
                                    x,
                                    y,
                                    w: 2,
                                    h: 2,
                                    proposal: None,
                                });
                            }
                        }
                    }
                }
                for region in &dc_datas[dc_idx].dp_regions {
                    let inside = region.bx / K_GROUP_DIM_IN_BLOCKS == gx
                        && region.by / K_GROUP_DIM_IN_BLOCKS == gy;
                    if inside && region.pieces.iter().all(|&(st, _, _)| present[st as usize]) {
                        v.push(Site {
                            x: region.bx,
                            y: region.by,
                            w: region.w,
                            h: region.h,
                            proposal: Some(region.pieces.clone()),
                        });
                    }
                }
                v
            })
            .collect();
        // The current layout of a closed site, sorted like `variants`.
        let site_parts = |st: &AcStrategyImage, site: &Site| -> Option<Parts> {
            let mut parts = Vec::new();
            let mut covered = 0;
            for dy in 0..site.h {
                for dx in 0..site.w {
                    if st.is_first_block(site.x + dx, site.y + dy) {
                        let raw = st.raw_strategy(site.x + dx, site.y + dy);
                        let cw = AcStrategyImage::covered_blocks_x_of(raw);
                        let ch = AcStrategyImage::covered_blocks_y_of(raw);
                        if dx + cw > site.w || dy + ch > site.h {
                            return None;
                        }
                        covered += cw * ch;
                        parts.push((raw, dx, dy));
                    }
                }
            }
            (covered == site.w * site.h).then_some(parts)
        };
        /// What placing a variant overwrote: layout, field and memo of the site.
        struct Placed {
            task: usize,
            site: (usize, usize),
            size: (usize, usize),
            cells: Vec<u8>,
            field: Vec<u8>,
            memo: Option<crate::group::MemoSnapshot>,
        }
        let distance = distp.distance;
        let place =
            |dc_datas: &mut [DcGroupData], task: usize, site: &Site, parts: &Parts| -> Placed {
                let (dc_idx, gx, gy) = ac_tasks[task];
                let data = &mut dc_datas[dc_idx];
                let (x, y, w, h) = (site.x, site.y, site.w, site.h);
                let cells = data.ac_strategy.save_region(x, y, w, h);
                let field: Vec<u8> = (y..y + h)
                    .flat_map(|r| data.raw_quant_field.row(r)[x..x + w].to_vec())
                    .collect();
                for &(st, dx, dy) in parts {
                    let (px, py) = (x + dx, y + dy);
                    data.ac_strategy.set_first(px, py, st);
                    let w = AcStrategyImage::covered_blocks_x_of(st);
                    let h = AcStrategyImage::covered_blocks_y_of(st);
                    if w * h > 1 {
                        let (mut max_q, mut sum) = (0u8, 0u32);
                        for r in py..py + h {
                            for &q in &data.raw_quant_field.row(r)[px..px + w] {
                                max_q = max_q.max(q);
                                sum += u32::from(q);
                            }
                        }
                        let q = crate::ac_strategy::aggregate_quant(max_q, sum, w * h, distance);
                        for r in py..py + h {
                            data.raw_quant_field.row_mut(r)[px..px + w].fill(q);
                        }
                    }
                }
                let memo_snapshot = memo.and_then(|memo| {
                    let mut guard = memo[task].lock().expect("block memo");
                    let m = guard.as_mut()?;
                    m.compact_tokens();
                    let (lx, ly) = (
                        x - gx * K_GROUP_DIM_IN_BLOCKS,
                        y - gy * K_GROUP_DIM_IN_BLOCKS,
                    );
                    let snapshot = m.snapshot(lx, ly, w, h);
                    m.invalidate(lx, ly, w, h);
                    Some(snapshot)
                });
                Placed {
                    task,
                    site: (x, y),
                    size: (w, h),
                    cells,
                    field,
                    memo: memo_snapshot,
                }
            };
        let unplace = |dc_datas: &mut [DcGroupData], p: Placed| {
            let (dc_idx, _, _) = ac_tasks[p.task];
            let data = &mut dc_datas[dc_idx];
            let (x, y) = p.site;
            let (w, _) = p.size;
            data.ac_strategy.restore_region(x, y, w, &p.cells);
            for (i, row) in p.field.chunks_exact(w).enumerate() {
                data.raw_quant_field.row_mut(y + i)[x..x + w].copy_from_slice(row);
            }
            if let (Some(memo), Some(snapshot)) = (memo, &p.memo)
                && let Some(m) = memo[p.task].lock().expect("block memo").as_mut()
            {
                m.restore(snapshot);
            }
        };
        let group_bits = |e: &Eval, task: usize| -> f32 {
            e.hit_bits[task]
                + e.pending[task].as_ref().map_or(0.0, |p| {
                    p.tokens[0]
                        .iter()
                        .map(|k| prices.token_bits(k))
                        .sum::<f32>()
                })
        };
        // A rejected trial restores only the pixels and the sync point: the
        // task's next evaluation overwrites its tokens.
        let restore_task =
            |from: &Eval, to: &mut Eval, task: usize, r: crate::ssimulacra2::Rect| {
                to.synced[task] = from.synced[task];
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                            .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                    }
                }
            };
        let accept_task =
            |from: &mut Eval, to: &mut Eval, task: usize, r: crate::ssimulacra2::Rect| {
                std::mem::swap(&mut to.pending[task], &mut from.pending[task]);
                std::mem::swap(&mut to.quant_dc[task], &mut from.quant_dc[task]);
                std::mem::swap(&mut to.source_dc[task], &mut from.source_dc[task]);
                std::mem::swap(&mut to.starts[task], &mut from.starts[task]);
                to.synced[task] = from.synced[task];
                for c in 0..3 {
                    for y in r.y0..r.y1 {
                        to.recon.plane_row_mut(c, y)[r.x0..r.x1]
                            .copy_from_slice(&from.recon.plane_row(c, y)[r.x0..r.x1]);
                    }
                }
            };
        let mut cur_gbits: Vec<f32> = (0..ac_tasks.len()).map(|t| group_bits(&cur, t)).collect();
        /// An accepted site: its task, origin and size in blocks.
        type Accepted = (usize, (usize, usize), (usize, usize));
        let mut accepted_sites: Vec<Accepted> = Vec::new();
        let max_sites = sites.iter().map(Vec::len).max().unwrap_or(0);
        for si in 0..max_sites {
            // The variants of each task's site that differ from its layout.
            let vars: Vec<Vec<&Parts>> = (0..ac_tasks.len())
                .map(|t| {
                    let Some(site) = sites[t].get(si) else {
                        return Vec::new();
                    };
                    let st = &dc_datas[ac_tasks[t].0].ac_strategy;
                    let Some(mut current) = site_parts(st, site) else {
                        return Vec::new();
                    };
                    current.sort_unstable();
                    let differs = |v: &Parts| {
                        let mut k = v.clone();
                        k.sort_unstable();
                        k != current
                    };
                    match &site.proposal {
                        Some(proposal) => {
                            if differs(proposal) {
                                vec![proposal]
                            } else {
                                Vec::new()
                            }
                        }
                        None => variants.iter().filter(|v| differs(v)).collect(),
                    }
                })
                .collect();
            let max_v = vars.iter().map(Vec::len).max().unwrap_or(0);
            let mut best: Vec<(f32, Option<usize>)> = vec![(j_cur, None); ac_tasks.len()];
            for vi in 0..max_v {
                let tasks: Vec<usize> = (0..ac_tasks.len())
                    .filter(|&t| vi < vars[t].len())
                    .collect();
                let placed: Vec<Placed> = tasks
                    .iter()
                    .map(|&t| place(dc_datas, t, &sites[t][si], vars[t][vi]))
                    .collect();
                cand.trial = true;
                let dirty = evaluate(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    dc_datas,
                    ac_tasks,
                    group_coords,
                    num_passes,
                    coeff_shifts,
                    coeff_orders,
                    order_alternatives,
                    ytob_dc,
                    qf_threshold,
                    Some(prices),
                    &tasks,
                    memo,
                    &mut cand,
                );
                cand.trial = false;
                for (k, &t) in tasks.iter().enumerate() {
                    let r = dirty[k];
                    let d = if r.x1 > r.x0 {
                        let dc = state.needs_dc().then(|| {
                            let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                            fill_decoded_dc(&cand, distp, r, out);
                            &*out
                        });
                        let d = state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint);
                        state.rollback(&checkpoint);
                        d
                    } else {
                        cur_d
                    };
                    let bits = cur_bits + group_bits(&cand, t) - cur_gbits[t];
                    let j = -d - slope * rate_term(bits);
                    if j > best[t].0 + 1e-7 {
                        best[t] = (j, Some(vi));
                    }
                    restore_task(&cur, &mut cand, t, r);
                }
                for p in placed {
                    unplace(dc_datas, p);
                }
            }
            // Commit each task's winner, re-encode them in full and accept
            // them one at a time on the running state.
            let winners: Vec<(usize, usize)> = (0..ac_tasks.len())
                .filter_map(|t| best[t].1.map(|v| (t, v)))
                .collect();
            if winners.is_empty() {
                continue;
            }
            let placed: Vec<Placed> = winners
                .iter()
                .map(|&(t, v)| place(dc_datas, t, &sites[t][si], vars[t][v]))
                .collect();
            let tasks: Vec<usize> = winners.iter().map(|w| w.0).collect();
            let dirty = evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                distp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &tasks,
                memo,
                &mut cand,
            );
            for ((k, &t), p) in tasks.iter().enumerate().zip(placed) {
                let r = dirty[k];
                let d = if r.x1 > r.x0 {
                    let dc = state.needs_dc().then(|| {
                        let out = trial_dc.get_or_insert_with(|| Image3F::new(xb, yb));
                        fill_decoded_dc(&cand, distp, r, out);
                        &*out
                    });
                    state.update_checkpointed(&cand.recon, dc, r, &mut checkpoint)
                } else {
                    cur_d
                };
                let cand_gbits = group_bits(&cand, t);
                let bits = cur_bits + cand_gbits - cur_gbits[t];
                let j = -d - slope * rate_term(bits);
                if j > j_cur + 1e-7 {
                    accept_task(&mut cand, &mut cur, t, r);
                    cur_gbits[t] = cand_gbits;
                    cur_bits = bits;
                    cur_d = d;
                    j_cur = j;
                    accepted_sites.push((t, p.site, p.size));
                } else {
                    if r.x1 > r.x0 {
                        state.rollback(&checkpoint);
                    }
                    restore_task(&cur, &mut cand, t, r);
                    unplace(dc_datas, p);
                }
            }
        }
        // The chroma DC passes after the loop read the retained source DC,
        // which depends on the layout: recapture it under the accepted sites.
        let mut refresh: Vec<usize> = accepted_sites.iter().map(|a| a.0).collect();
        refresh.sort_unstable();
        refresh.dedup();
        let sources = ctx
            .thread_pool
            .steal_map(scratch, refresh.len(), |i, scratch| {
                let (dc_idx, gx, gy) = ac_tasks[refresh[i]];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (_, _, source, _, _) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    distp,
                    &dc_datas[dc_idx],
                    num_passes,
                    coeff_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    ytob_dc,
                    None,
                    coeff_orders,
                    &[],
                    true,
                    qf_threshold,
                    false,
                    None,
                );
                source.expect("source dc")
            });
        for (&task, source) in refresh.iter().zip(sources) {
            let (dc_idx, gx, gy) = ac_tasks[task];
            let (ox, oy) = (gx * K_GROUP_DIM_IN_BLOCKS, gy * K_GROUP_DIM_IN_BLOCKS);
            let dc = &mut dc_datas[dc_idx];
            for &(_, (x, y), (w, h)) in accepted_sites.iter().filter(|a| a.0 == task) {
                for (target, src) in [
                    (&mut dc.source_dc_y, &source.y),
                    (&mut dc.source_dc_b, &source.b),
                    (&mut dc.source_dc_x, &source.x),
                ] {
                    if let Some(target) = target.as_mut() {
                        for r in y..y + h {
                            target.row_mut(r)[x..x + w]
                                .copy_from_slice(&src.row(r - oy)[x - ox..x - ox + w]);
                        }
                    }
                }
            }
        }
        cur_bits = frame_bits(&cur);
    }
    // ---- Frame-level chroma precision race ----
    // The X and B quant-matrix scales are 3-bit header fields set by distance
    // and content gates (higher = finer). A coordinate search, X then B, arbitrates
    // each neighbor of the current pair on the loop's target with a full
    // priced encode; the block memo keys the chroma channels by their scale, so
    // an arm recomputes only the channel it changes.
    let mut distp_out = *distp;
    // Each arbiter races on its own metric and exchange rate.
    let race_slope = if loop_state.as_ref().is_some_and(LoopArbiter::needs_dc) {
        cv_slope
    } else {
        cached_slope
    };
    if race_slope > 0.0 && ctx.coding.is_xyb() {
        let slope = race_slope;
        let b_start = ctx.b_qm_scale();
        // No more regional updates: full-frame comparisons do not read the
        // incremental pyramid, or any accepted evaluation's reconstruction.
        let mut judge = match loop_state.take() {
            Some(LoopArbiter::Cvvdp(mut judge)) => {
                judge.finish_updates();
                Some(judge)
            }
            _ => None,
        };
        if !want_recon {
            cur.recon = Image3F::new(0, 0);
        }
        let mut score = |e: &Eval| -> f32 {
            match judge.as_deref_mut() {
                Some(judge) => judge.deficit_of(&e.recon, &decoded_dc(e, distp)),
                None => reference.deficit(&e.recon),
            }
        };
        let arm_eval = |x: u32, b: u32, scratch: &mut CoderScratch| -> (Eval, f32) {
            let mut dp = *distp;
            dp.x_qm_scale = x;
            ctx.set_b_qm_scale(b);
            let mut e = new_eval();
            compact_memos(&all_tasks);
            evaluate(
                ctx,
                scratch,
                opsin,
                dim,
                &dp,
                dc_datas,
                ac_tasks,
                group_coords,
                num_passes,
                coeff_shifts,
                coeff_orders,
                order_alternatives,
                ytob_dc,
                qf_threshold,
                Some(prices),
                &all_tasks,
                memo,
                &mut e,
            );
            ctx.set_b_qm_scale(b_start);
            let bits = frame_bits(&e);
            (e, bits)
        };
        let cur_score = cur_d;
        let mut best = (
            -cur_score - slope * rate_term(cur_bits),
            distp.x_qm_scale,
            b_start,
        );
        let mut tried: Vec<(u32, u32)> = vec![(best.1, best.2)];
        let mut best_eval: Option<Eval> = None;
        for coord in 0..2usize {
            for dir in [1i32, -1] {
                loop {
                    let (mut x, mut b) = (best.1 as i32, best.2 as i32);
                    if coord == 0 {
                        x += dir;
                    } else {
                        b += dir;
                    }
                    if !(1..=7).contains(&x) || !(2..=7).contains(&b) {
                        break;
                    }
                    let (x, b) = (x as u32, b as u32);
                    // The reverse direction after a walk re-visits a point already scored.
                    if tried.contains(&(x, b)) {
                        break;
                    }
                    let (mut e, bits) = arm_eval(x, b, scratch);
                    let d = score(&e);
                    let j = -d - slope * rate_term(bits);
                    tried.push((x, b));
                    if j > best.0 + 1e-6 {
                        best = (j, x, b);
                        if !want_recon {
                            e.recon = Image3F::new(0, 0);
                        }
                        best_eval = Some(e);
                    } else {
                        break;
                    }
                }
            }
        }
        if let Some(e) = best_eval {
            cur = e;
            distp_out.x_qm_scale = best.1;
            ctx.set_b_qm_scale(best.2);
        }
    }
    // ---- Decoder-feedback coefficient polish (`coef_polish`) ----
    // The settled frame's levels are revisited coefficient by coefficient
    // against the exact metric states; see `crate::coef_polish`.
    if polish_wanted && race_slope > 0.0 {
        drop(polish_cv);
        drop(loop_state);
        drop(cur);
        let mut e = new_eval();
        compact_memos(&all_tasks);
        evaluate(
            ctx,
            scratch,
            opsin,
            dim,
            &distp_out,
            dc_datas,
            ac_tasks,
            group_coords,
            num_passes,
            coeff_shifts,
            coeff_orders,
            order_alternatives,
            ytob_dc,
            qf_threshold,
            Some(prices),
            &all_tasks,
            memo,
            &mut e,
        );
        let bits0 = frame_bits(&e);
        let recon = std::mem::replace(&mut e.recon, Image3F::new(0, 0));
        // Near transparency ColorVideoVDP does not separate the shifts (no
        // exchange rate): SSIMULACRA2 alone decides.
        let cv_dc = (polish_cv_slope.is_finite() && polish_cv_slope > 0.0)
            .then(|| legacy_dc(&e, &distp_out));
        let cv_lambda = polish_cv_slope * 100.0 / bits0;
        let lambda_bits = race_slope * 100.0 / bits0;
        drop(e);
        // Polishing re-encodes without the loop memo. Release its coefficients,
        // reconstruction and token arenas before building any metric states.
        if let Some(memo) = memo {
            for slot in memo {
                *slot.lock().expect("block memo") = None;
            }
        }
        // The groups are polished in a fixed number of chunks, each against its
        // own copy of the frame and of the metric states built from the same
        // baseline, so the result depends on neither the thread count nor the
        // scheduling. The chunk count is bounded by the states' memory.
        const POLISH_STATE_BYTES_PER_PIXEL: usize = 256;
        const POLISH_MEMORY_BUDGET: usize = 1 << 30;
        const POLISH_MAX_CHUNKS: usize = 8;
        let chunks = (POLISH_MEMORY_BUDGET / (POLISH_STATE_BYTES_PER_PIXEL * w * h).max(1))
            .clamp(1, POLISH_MAX_CHUNKS)
            .min(all_tasks.len())
            .max(1);
        let per_chunk = all_tasks.len().div_ceil(chunks);
        use std::sync::atomic::Ordering;
        crate::coef_polish::POLISH_EPOCH.fetch_add(1, Ordering::Relaxed);
        let recon_ref = &recon;
        let reference_ref = &reference;
        let cv_dc_ref = cv_dc.as_ref();
        // The analysed source is shared; each chunk builds only its own
        // incremental state over it.
        let cv_prepared =
            cv_dc_ref.and_then(|_| crate::cvvdp_arbiter::CvvdpArbiter::prepare(opsin));
        let cv_prepared_ref = cv_prepared.as_ref();
        let build = move || -> Box<crate::coef_polish::CoefPolisher> {
            let state = reference_ref.state(recon_ref);
            let cv = cv_dc_ref.and_then(|dc| {
                let mut judge = crate::cvvdp_arbiter::CvvdpArbiter::new_shared(
                    opsin,
                    cv_prepared_ref?.clone(),
                )?;
                judge.reset(recon_ref, dc);
                Some(crate::coef_polish::CvJudge::new(
                    judge,
                    dc.clone(),
                    cv_lambda,
                ))
            });
            Box::new(crate::coef_polish::CoefPolisher::new(
                recon_ref.clone(),
                state,
                lambda_bits,
                cv,
            ))
        };
        let results = ctx.thread_pool.steal_map(scratch, chunks, |ci, scratch| {
            scratch.ac_group.polisher = Some(build());
            let mut polished = Vec::with_capacity(per_chunk);
            for &t in all_tasks.iter().skip(ci * per_chunk).take(per_chunk) {
                let (dc_idx, gx, gy) = ac_tasks[t];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (p, local, _, _, capture) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    dim,
                    &distp_out,
                    &dc_datas[dc_idx],
                    num_passes,
                    coeff_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    ytob_dc,
                    Some(prices),
                    coeff_orders,
                    order_alternatives,
                    false,
                    qf_threshold,
                    true,
                    None,
                );
                polished.push((t, p, local, capture.map(|(image, _)| image)));
            }
            scratch.ac_group.polisher = None;
            polished
        });
        // Stale from now on: nothing else in this encode may polish.
        crate::coef_polish::POLISH_EPOCH.fetch_add(1, Ordering::Relaxed);
        // Only tokens and DC survive the pass; no final frame reconstruction.
        let mut out = Eval {
            recon: Image3F::new(0, 0),
            pending: vec![None; ac_tasks.len()],
            quant_dc: vec![None; ac_tasks.len()],
            source_dc: vec![None; ac_tasks.len()],
            starts: Vec::new(),
            synced: Vec::new(),
            trial: false,
            hit_bits: Vec::new(),
        };
        // The polished groups' captures make up the frame reconstruction.
        if want_recon {
            out.recon = Image3F::new(w, h);
        }
        for (t, p, local, capture) in results.into_iter().flatten() {
            out.pending[t] = Some(p);
            out.quant_dc[t] = Some(local);
            if let (true, Some(plane)) = (want_recon, capture) {
                let (dc_idx, gx, gy) = ac_tasks[t];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let x0 = dc_gx * K_DC_GROUP_DIM + gx * K_GROUP_DIM;
                let y0 = dc_gy * K_DC_GROUP_DIM + gy * K_GROUP_DIM;
                for c in 0..3 {
                    for y in 0..plane.ysize() {
                        let src = &plane.plane_row(c, y)[..plane.xsize()];
                        out.recon.plane_row_mut(c, y0 + y)[x0..x0 + src.len()].copy_from_slice(src);
                    }
                }
            }
        }
        cur = out;
    }
    let recon = want_recon.then(|| std::mem::replace(&mut cur.recon, Image3F::new(0, 0)));
    finish(cur, distp_out, recon)
}
