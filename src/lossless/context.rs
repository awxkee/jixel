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

//! Weighted-predictor activity context trees and threshold selection.

use super::GroupLayout;
use super::entropy::EntropyOfHistFn;
use super::groups::{DeepLzScratchPool, group_lz_threads, write_lz_groups};
use super::header::{
    write_frame_header_modular_kind, write_modular_transforms, write_patch_dictionary,
    write_toc_entry,
};
use super::lz77::{
    LzToken, build_lz_pixel_code, lz77_compress_for_speed, lz77_compress_for_speed_with_depth,
    write_local_tree_lz77_offsets, write_lz_section, write_tree_lz77,
};
use super::predictor::{
    WP_CONTEXT_BUCKETS, WP_CONTEXT_CUTOFFS, WpChannelContexts, predictor_neighbors, predictor_value,
};
use super::tree::{push_leaf, push_leaf_offset, push_split};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::{Token, pack_signed};
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpParams, WpState, write_wp_header};

// ---------------------------------------------------------------------------
// Context tree (MA tree): split each channel's entropy context on the WP error
// property kWPProp (p[15]) into 3 activity buckets. The decoder must run the WP
// state for every pixel of a channel whose subtree references p[15], so we run
// WP for all channels here regardless of the selected leaf predictor.
// ---------------------------------------------------------------------------

pub(super) const PROP_WP: u32 = 15; // kNumStaticProperties(2) + 13

/// Encoder-side tree: Split(property, splitval, gt-branch, le-branch) routes
/// `prop > splitval` to the gt-branch. Leaf carries (predictor, tag) where
/// tag = channel*3 + bucket.
pub(super) enum CtTree {
    Split(u32, i32, Box<CtTree>, Box<CtTree>),
    Leaf(u32, u32),
    /// A constant channel: Zero predictor with the value as offset, no
    /// activity split (the WP property would cost the decoder WP state).
    /// Its three activity tags share the one context.
    ConstLeaf(i32, u32),
}

/// BFS-emit the tree (matches libjxl's FIFO tree decode) and return the context
/// id assigned to each leaf tag. Context id == leaf's order of appearance.
pub(super) fn emit_ct_tree(
    root: &CtTree,
    out: &mut Vec<Token>,
) -> std::collections::HashMap<u32, u32> {
    use std::collections::{HashMap, VecDeque};
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut q: VecDeque<&CtTree> = VecDeque::new();
    q.push_back(root);
    let mut ctx = 0u32;
    while let Some(node) = q.pop_front() {
        match node {
            CtTree::Split(prop, val, gt, le) => {
                push_split(out, *prop, *val);
                q.push_back(gt);
                q.push_back(le);
            }
            CtTree::Leaf(pred, tag) => {
                push_leaf(out, *pred);
                map.insert(*tag, ctx);
                ctx += 1;
            }
            CtTree::ConstLeaf(offset, tag) => {
                push_leaf_offset(out, 0, *offset);
                for bucket in 0..3 {
                    map.insert(*tag + bucket, ctx);
                }
                ctx += 1;
            }
        }
    }
    map
}

#[inline]
fn bucket_of(prop: i64, t: i64) -> u32 {
    if prop > t {
        2
    } else if prop > -t - 1 {
        1
    } else {
        0
    }
}

/// 3-leaf activity subtree for channel `c` with predictor `pred` and threshold `t`.
fn act_sub(c: u32, pred: u32, t: i32) -> CtTree {
    CtTree::Split(
        PROP_WP,
        t,
        Box::new(CtTree::Leaf(pred, c * 3 + 2)),
        Box::new(CtTree::Split(
            PROP_WP,
            -t - 1,
            Box::new(CtTree::Leaf(pred, c * 3 + 1)),
            Box::new(CtTree::Leaf(pred, c * 3)),
        )),
    )
}

/// Channel-split tree (same shape as build_balanced_tree_tokens) with each
/// channel-leaf replaced by its activity subtree.
/// `constant`: the value of a constant last channel, coded as `ConstLeaf`.
pub(super) fn build_context_tree(
    nb_chans: usize,
    preds: &[u32],
    t: &[i32],
    constant: Option<i32>,
) -> CtTree {
    let a = |c: usize| match constant {
        Some(value) if c + 1 == nb_chans => CtTree::ConstLeaf(value, c as u32 * 3),
        _ => act_sub(c as u32, preds[c], t[c]),
    };
    match nb_chans {
        1 => a(0),
        2 => CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0))),
        3 => CtTree::Split(
            0,
            1,
            Box::new(a(2)),
            Box::new(CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0)))),
        ),
        4 => CtTree::Split(
            0,
            1,
            Box::new(CtTree::Split(0, 2, Box::new(a(3)), Box::new(a(2)))),
            Box::new(CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0)))),
        ),
        _ => unreachable!("context tree supports 1..=4 channels"),
    }
}

fn order0_entropy(vals: &[u32], cell: &mut Vec<u64>, entropy_of_hist: EntropyOfHistFn) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    // Direct-indexed frequency histogram (residual symbols are small-range), in
    // place of a HashMap: no hashing, and a deterministic accumulation order.
    let max = vals.iter().copied().max().unwrap_or(0) as usize;
    if cell.len() < max + 1 {
        cell.resize(max + 1, 0);
    }
    let hist = &mut cell[..max + 1];
    hist.fill(0);
    for &v in vals {
        hist[v as usize] += 1;
    }
    entropy_of_hist(hist, vals.len() as u64)
}

/// Run WP over one channel's group rectangle, returning per-pixel
/// (packed residual under `pred_id`, WP property p[15]) in row-major order.
fn collect_channel(
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    wp_params: WpParams,
) -> (Vec<u32>, Vec<i64>) {
    let mut wp = WpState::with_params(gw, wp_params);
    let mut res = vec![0; gw * gh];
    let mut prp = vec![0; gw * gh];
    let (mut gx, mut gy) = (0, 0);
    for (res, prp) in res.iter_mut().zip(&mut prp) {
        let v = get(gx, gy) as i64;
        let neighbors = predictor_neighbors(&get, gx, gy, gw);
        let wp_pred = wp.predict(
            gx,
            gy,
            neighbors.top,
            neighbors.left,
            neighbors.top_right,
            neighbors.top_left,
            neighbors.top_top,
        );
        *prp = wp.wp_prop;
        let pred = predictor_value(pred_id, neighbors, wp_pred);
        *res = pack_signed((v - pred) as i32);
        wp.update(v, gx, gy);
        // Advance row coordinates without dividing the linear index.
        gx += 1;
        if gx == gw {
            gx = 0;
            gy += 1;
        }
    }
    (res, prp)
}

#[derive(Default)]
pub(crate) struct PickThresholdScratch {
    pub(crate) hist_scratch: Vec<u64>,
}

impl PickThresholdScratch {
    fn make_scratches(&mut self, size: usize) -> (&mut [u64], &mut [u64], &mut [u64]) {
        let bucket = size + 1;
        if self.hist_scratch.len() < bucket * 3 {
            self.hist_scratch.resize(bucket * 3, 0);
        }
        let (b0, r0) = self.hist_scratch.split_at_mut(bucket);
        let (b1, b2) = r0.split_at_mut(bucket);
        (b0, b1, &mut b2[..bucket])
    }
}

pub(super) fn pick_threshold(
    res: &[u32],
    prp: &[i64],
    scratch: &mut CoderScratch,
) -> (i32, f32, f32) {
    let entropy_of_hist = scratch.entropy_of_hist;
    let flat = order0_entropy(res, &mut scratch.order0_entropy, entropy_of_hist);
    let max = res.iter().copied().max().unwrap_or(0) as usize;
    let (h0, h1, h2) = scratch.threshold.make_scratches(max + 1);
    let mut best_t = 0i32;
    let mut best_bits = f32::INFINITY;
    for &t in &[8i64, 16, 24, 32, 48, 64, 96] {
        h0.fill(0);
        h1.fill(0);
        h2.fill(0);
        let (mut n0, mut n1, mut n2) = (0u64, 0u64, 0u64);
        for (&r, &p) in res.iter().zip(prp.iter()) {
            match bucket_of(p, t) {
                0 => {
                    h0[r as usize] += 1;
                    n0 += 1;
                }
                1 => {
                    h1[r as usize] += 1;
                    n1 += 1;
                }
                _ => {
                    h2[r as usize] += 1;
                    n2 += 1;
                }
            }
        }
        let bits = entropy_of_hist(h0, n0) + entropy_of_hist(h1, n1) + entropy_of_hist(h2, n2);
        if bits < best_bits {
            best_bits = bits;
            best_t = t as i32;
        }
    }
    (best_t, best_bits, flat)
}

/// Multi-group form of `pick_threshold` that scans the already-collected
/// group slices directly. This avoids rebuilding full-image residual and
/// property vectors for every channel solely to calculate histograms.
pub(super) fn pick_threshold_grouped(
    groups: &[Vec<(Vec<u32>, Vec<i64>)>],
    channel: usize,
    scratch: &mut CoderScratch,
) -> (i32, f32, f32) {
    let entropy_of_hist = scratch.entropy_of_hist;
    let total: usize = groups.iter().map(|group| group[channel].0.len()).sum();
    if total == 0 {
        return (0, 0.0, 0.0);
    }
    let max = groups
        .iter()
        .flat_map(|group| group[channel].0.iter().copied())
        .max()
        .unwrap_or(0) as usize;

    if scratch.order0_entropy.len() < max + 1 {
        scratch.order0_entropy.resize(max + 1, 0);
    }
    let flat_hist = &mut scratch.order0_entropy[..max + 1];
    flat_hist.fill(0);
    for group in groups {
        for &residual in &group[channel].0 {
            flat_hist[residual as usize] += 1;
        }
    }
    let flat = entropy_of_hist(flat_hist, total as u64);

    let (h0, h1, h2) = scratch.threshold.make_scratches(max + 1);
    let mut best_t = 0i32;
    let mut best_bits = f32::INFINITY;
    for &t in &[8i64, 16, 24, 32, 48, 64, 96] {
        h0.fill(0);
        h1.fill(0);
        h2.fill(0);
        let (mut n0, mut n1, mut n2) = (0u64, 0u64, 0u64);
        for group in groups {
            let (residuals, properties) = &group[channel];
            for (&residual, &property) in residuals.iter().zip(properties.iter()) {
                match bucket_of(property, t) {
                    0 => {
                        h0[residual as usize] += 1;
                        n0 += 1;
                    }
                    1 => {
                        h1[residual as usize] += 1;
                        n1 += 1;
                    }
                    _ => {
                        h2[residual as usize] += 1;
                        n2 += 1;
                    }
                }
            }
        }
        let bits = entropy_of_hist(h0, n0) + entropy_of_hist(h1, n1) + entropy_of_hist(h2, n2);
        if bits < best_bits {
            best_bits = bits;
            best_t = t as i32;
        }
    }
    (best_t, best_bits, flat)
}

/// v1 context tree: single-group only. Returns true (and writes the full frame)
/// if the context tree is estimated to help; false otherwise (caller falls
/// through to the flat path, having written nothing).
pub(super) fn try_encode_context_tree_single_group(
    speed: crate::Speed,
    num_color: usize,
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    alpha_constant: Option<i32>,
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
    frame_kind: ModularFrameKind<'_>,
    predictors: &[u32],
    min_symbol: u32,
    rct_type: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = num_color + usize::from(alpha.is_some());
    // The modular channels' predictors: the color planes, then alpha.
    let chan_preds: Vec<u32> = (0..num_color)
        .map(|c| predictors[c])
        .chain(alpha.map(|_| predictors[3]))
        .collect();

    // Collect residuals + WP property per channel (WP runs for every channel).
    let collected = pool.steal_map(scratch, nb_chans, |chan, _scratch| {
        if chan < num_color {
            let pd = linear.plane_data(chan);
            collect_channel(
                |x, y| pd[y * xsize + x],
                xsize,
                ysize,
                chan_preds[chan],
                wp_params,
            )
        } else if alpha_constant.is_some() {
            // Zero predictor + offset: every residual is 0, no WP property.
            (vec![0u32; xsize * ysize], vec![0i64; xsize * ysize])
        } else {
            let a = alpha.expect("alpha channel must exist");
            collect_channel(
                |x, y| a.get_i32(y * xsize + x),
                xsize,
                ysize,
                chan_preds[chan],
                wp_params,
            )
        }
    });
    let (chan_res, chan_prp): (Vec<Vec<u32>>, Vec<Vec<i64>>) = collected.into_iter().unzip();

    // Per-channel threshold + cost comparison.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    let threshold_costs = pool.steal_map(scratch, nb_chans, |chan, scratch| {
        pick_threshold(&chan_res[chan], &chan_prp[chan], scratch)
    });
    for (chan, (t, cb, fb)) in threshold_costs.into_iter().enumerate() {
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    // Guard: require the context tree to beat the flat path by more than the
    // extra-context header overhead (~64 bytes per added context, conservative).
    let overhead_bits = (2 * nb_chans) as f32 * 64.0 * 8.0;
    if ctx_bits + overhead_bits >= flat_bits {
        return false;
    }

    // Build tree + context map.
    let tree = build_context_tree(nb_chans, &chan_preds, &ts, alpha_constant);
    let mut tree_tokens: Vec<Token> = Vec::new();
    let ctx_map = emit_ct_tree(&tree, &mut tree_tokens);
    // Flat lookup over the dense (chan*3+bucket) property space, replacing a
    // per-pixel HashMap probe in the tokenize loop.
    let ctx_lut: Vec<u32> = (0..(nb_chans as u32) * 3).map(|k| ctx_map[&k]).collect();
    // One context per leaf (a constant channel's three tags share one).
    let num_pixel_ctx = ctx_lut.iter().max().map_or(0, |&m| m as usize + 1);

    // Tokenize: each pixel routed to context (channel,bucket).
    let channel_tokens = pool.steal_map(scratch, nb_chans, |chan, _scratch| {
        let res = &chan_res[chan];
        let prp = &chan_prp[chan];
        let t = ts[chan] as i64;
        prp[..res.len()]
            .iter()
            .zip(res)
            .map(|(&prp, &res)| {
                let bucket = bucket_of(prp, t);
                let ctx = ctx_lut[chan * 3 + bucket as usize];
                Token::new(ctx, res)
            })
            .collect::<Vec<_>>()
    });
    let mut tokens: Vec<Token> = Vec::with_capacity(xsize * ysize * nb_chans);
    for channel in channel_tokens {
        tokens.extend(channel);
    }

    // Frame header + single section (a patched frame's dictionary first).
    write_frame_header_modular_kind(alpha.is_some(), frame_kind, layout, writer);
    let mut section = BitWriter::new();
    if let ModularFrameKind::Patched { references, .. } = frame_kind {
        write_patch_dictionary(references, alpha.is_some(), scratch, &mut section);
    }
    super::header::write_dc_quant(frame_kind, &mut section);
    section.write(1, 0); // has_tree = 0
    section.write(1, 0); // use_global_tree = 0
    write_wp_header(wp_params, &mut section);
    write_modular_transforms(nb_chans, rct_type, &mut section);

    let distance_ctx = num_pixel_ctx as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy(),
        std::iter::once(lz_tokens.as_slice()),
        num_pixel_ctx,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_tree_lz77(
        &tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(std::slice::from_mut(&mut section));
    writer.zero_pad_to_byte();
    true
}

/// v1 context tree, multi-group. Global context tree in LfGlobal; each AC group
/// routes its group-local pixels through it (fresh WP per group, matching the
/// decoder). Returns true (and writes the full frame) if it helps; else false.
pub(super) fn try_encode_context_tree_multi_group(
    speed: crate::Speed,
    num_color: usize,
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    alpha_constant: Option<i32>,
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
    frame_kind: ModularFrameKind<'_>,
    predictors: &[u32],
    xsize_groups: usize,
    ysize_groups: usize,
    num_dc_groups: usize,
    min_symbol: u32,
    rct_type: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = num_color + usize::from(alpha.is_some());
    // The modular channels' predictors: the color planes, then alpha.
    let chan_preds: Vec<u32> = (0..num_color)
        .map(|c| predictors[c])
        .chain(alpha.map(|_| predictors[3]))
        .collect();
    let num_ac_groups = xsize_groups * ysize_groups;

    // 1) Collect (residual, WP property) per group per channel (group-local WP).
    let groups: Vec<Vec<(Vec<u32>, Vec<i64>)>> =
        pool.steal_map(scratch, num_ac_groups, |group_index, _scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * layout.dim();
            let y0 = gy * layout.dim();
            let gw = layout.dim().min(xsize - x0);
            let gh = layout.dim().min(ysize - y0);
            let mut chans: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(nb_chans);
            for chan in 0..num_color {
                let pd = linear.plane_data(chan);
                let get = |lx: usize, ly: usize| pd[(y0 + ly) * xsize + (x0 + lx)];
                chans.push(collect_channel(get, gw, gh, chan_preds[chan], wp_params));
            }
            if let Some(a) = alpha {
                if alpha_constant.is_some() {
                    chans.push((vec![0u32; gw * gh], vec![0i64; gw * gh]));
                } else {
                    let get = |lx: usize, ly: usize| a.get_i32((y0 + ly) * xsize + (x0 + lx));
                    chans.push(collect_channel(
                        get,
                        gw,
                        gh,
                        chan_preds[num_color],
                        wp_params,
                    ));
                }
            }
            chans
        });

    // 2) Global per-channel threshold from aggregated stats.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    let threshold_costs = pool.steal_map(scratch, nb_chans, |chan, scratch| {
        pick_threshold_grouped(&groups, chan, scratch)
    });
    for (chan, (t, cb, fb)) in threshold_costs.into_iter().enumerate() {
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    let overhead_bits = (2 * nb_chans) as f32 * 64.0 * 8.0;
    if ctx_bits + overhead_bits >= flat_bits {
        return false;
    }

    // 3) Global context tree + map.
    let tree = build_context_tree(nb_chans, &chan_preds, &ts, alpha_constant);
    let mut tree_tokens: Vec<Token> = Vec::new();
    let ctx_map = emit_ct_tree(&tree, &mut tree_tokens);
    // Flat lookup over the dense (chan*3+bucket) property space, replacing a
    // per-pixel HashMap probe in the tokenize loop.
    let ctx_lut: Vec<u32> = (0..(nb_chans as u32) * 3).map(|k| ctx_map[&k]).collect();
    // One context per leaf (a constant channel's three tags share one).
    let num_pixel_ctx = ctx_lut.iter().max().map_or(0, |&m| m as usize + 1);
    let distance_ctx = num_pixel_ctx as u32;

    // 4) Per-group tokens (reusing collected res/prop) + per-group LZ77.
    let group_lz_tokens: Vec<Vec<LzToken>> = {
        let deep_lz = DeepLzScratchPool::new(group_lz_threads(speed, pool));
        pool.steal_map_with_threads(
            scratch,
            num_ac_groups,
            group_lz_threads(speed, pool),
            |group_index, scratch| {
                let g = &groups[group_index];
                let token_count = g.iter().map(|(res, _)| res.len()).sum();
                let mut toks: Vec<Token> = Vec::with_capacity(token_count);
                for (((res, prp), &threshold), contexts) in g[..nb_chans]
                    .iter()
                    .zip(&ts[..nb_chans])
                    .zip(ctx_lut.as_chunks::<3>().0)
                {
                    let t = threshold as i64;
                    toks.extend(prp[..res.len()].iter().zip(res).map(|(&prp, &res)| {
                        let bucket = bucket_of(prp, t);
                        let ctx = contexts[bucket as usize];
                        Token::new(ctx, res)
                    }));
                }
                deep_lz.with_depth(|depth| {
                    lz77_compress_for_speed_with_depth(
                        &toks,
                        distance_ctx,
                        speed,
                        depth,
                        None,
                        scratch,
                    )
                })
            },
        )
    };
    // A patched frame's dictionary opens the first section (written before
    // the code borrows the entropy scratch).
    let dictionary: Option<BitWriter> = match frame_kind {
        ModularFrameKind::Patched { references, .. } => {
            let mut w = BitWriter::new();
            write_patch_dictionary(references, alpha.is_some(), scratch, &mut w);
            Some(w)
        }
        _ => None,
    };
    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy(),
        group_lz_tokens.iter().map(Vec::as_slice),
        num_pixel_ctx,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    // 5) Sections (same layout as the flat multi-group path).
    write_frame_header_modular_kind(alpha.is_some(), frame_kind, layout, writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    if let Some(dictionary) = &dictionary {
        sections[0].append(dictionary);
    }
    super::header::write_dc_quant(frame_kind, &mut sections[0]);
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(
        &tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    write_wp_header(wp_params, &mut sections[0]);
    write_modular_transforms(nb_chans, rct_type, &mut sections[0]);
    // An empty global stream still carries its ANS state: jxl-oxide reads it.
    write_lz_section::<LzToken>(&[], distance_ctx, &code, min_symbol, &mut sections[0]);
    sections[0].zero_pad_to_byte();

    for section in sections[1..num_dc_groups + 1].iter_mut() {
        section.write(1, 1);
        write_wp_header(wp_params, section);
        section.write(2, 0);
        section.zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    write_wp_header(wp_params, &mut sections[ac_global_idx]);
    sections[ac_global_idx].zero_pad_to_byte();

    write_lz_groups(
        &group_lz_tokens,
        &[],
        &code,
        distance_ctx,
        min_symbol,
        wp_params,
        pool,
        &mut sections[2 + num_dc_groups..],
    );

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}

/// Fixed flat context tree: a channel split, then each `bucketed` channel
/// split on the WP error property at `WP_CONTEXT_CUTOFFS` scaled by
/// `1 << shift` (the sample depth above 8 bits).
pub(super) struct WpContextTree {
    pub(super) tokens: Vec<Token>,
    /// Context of every WP bucket, per channel (one context for a channel
    /// that is not bucketed).
    pub(super) buckets: Vec<WpChannelContexts>,
    pub(super) num_ctx: usize,
}

pub(super) fn wp_context_tree(
    predictors: &[u32],
    offsets: &[i32],
    bucketed: &[bool],
    shift: u32,
) -> WpContextTree {
    enum Node {
        Split {
            prop: u32,
            val: i32,
            gt: usize,
            le: usize,
        },
        Leaf {
            chan: usize,
            lo: usize,
            hi: usize,
        },
    }
    fn buckets(nodes: &mut Vec<Node>, shift: u32, chan: usize, lo: usize, hi: usize) -> usize {
        if lo == hi {
            nodes.push(Node::Leaf { chan, lo, hi });
            return nodes.len() - 1;
        }
        let mid = (lo + hi) / 2;
        let gt = buckets(nodes, shift, chan, mid + 1, hi);
        let le = buckets(nodes, shift, chan, lo, mid);
        // `p >> shift > c` exactly when `p > ((c + 1) << shift) - 1`.
        nodes.push(Node::Split {
            prop: PROP_WP,
            val: ((WP_CONTEXT_CUTOFFS[mid] + 1) << shift) - 1,
            gt,
            le,
        });
        nodes.len() - 1
    }
    fn channels(
        nodes: &mut Vec<Node>,
        shift: u32,
        bucketed: &[bool],
        lo: usize,
        hi: usize,
    ) -> usize {
        if lo == hi {
            if bucketed[lo] {
                return buckets(nodes, shift, lo, 0, WP_CONTEXT_BUCKETS - 1);
            }
            nodes.push(Node::Leaf {
                chan: lo,
                lo: 0,
                hi: WP_CONTEXT_BUCKETS - 1,
            });
            return nodes.len() - 1;
        }
        let mid = (lo + hi) / 2;
        let gt = channels(nodes, shift, bucketed, mid + 1, hi);
        let le = channels(nodes, shift, bucketed, lo, mid);
        nodes.push(Node::Split {
            prop: 0,
            val: mid as i32,
            gt,
            le,
        });
        nodes.len() - 1
    }
    let mut nodes = Vec::new();
    let root = channels(&mut nodes, shift, bucketed, 0, bucketed.len() - 1);
    let mut tokens = Vec::new();
    let mut table: Vec<WpChannelContexts> = (0..bucketed.len())
        .map(|_| WpChannelContexts {
            contexts: [0; WP_CONTEXT_BUCKETS],
            shift,
        })
        .collect();
    let mut num_ctx = 0u32;
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(index) = queue.pop_front() {
        match nodes[index] {
            Node::Split { prop, val, gt, le } => {
                push_split(&mut tokens, prop, val);
                queue.push_back(gt);
                queue.push_back(le);
            }
            Node::Leaf { chan, lo, hi } => {
                push_leaf_offset(&mut tokens, predictors[chan], offsets[chan]);
                table[chan].contexts[lo..=hi].fill(num_ctx);
                num_ctx += 1;
            }
        }
    }
    WpContextTree {
        tokens,
        buckets: table,
        num_ctx: num_ctx as usize,
    }
}

/// The flat path's tree (`wp_tree`, else one leaf per channel) and pixel code.
pub(super) fn write_flat_tree(
    wp_tree: Option<&WpContextTree>,
    predictors: &[u32],
    offsets: &[i32],
    pixel_code: &crate::entropy::EntropyCode<'_>,
    min_symbol: u32,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut BitWriter,
) {
    match wp_tree {
        Some(tree) => write_tree_lz77(&tree.tokens, pixel_code, min_symbol, huffman_pool, w),
        None => write_local_tree_lz77_offsets(
            predictors,
            offsets,
            pixel_code,
            min_symbol,
            huffman_pool,
            w,
        ),
    }
}
