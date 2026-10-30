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

//! Learned-tree training, pruning, offsets, and layout ranking.

use super::GroupLayout;
use super::context::PROP_WP;
use super::ma_sampling::{MaRefPlane, ma_channel_sample_count, sample_channel_ma};
use super::ma_source::{
    CoarseLearn, LearnedCandidate, MA_COARSE_MAX_LEAVES, MA_COARSE_TARGET_SAMPLES,
    MA_DEEPEN_MIN_SAVINGS, MA_FINAL_SPLIT_COST_BITS, MA_LITE_CANDIDATES, MA_LITE_MAX_LEAVES,
    MA_MIN_NODE_SAMPLES, MA_PRUNE_LEAF_BITS, MA_PRUNE_MIN_NODE_SAMPLES, MA_PRUNE_SPLIT_COST_BITS,
    MA_SIDE_PREDS, MA_SPLIT_COST_BITS, MA_WIDE_SIDE_PREDS, MaSource, MaTransform, ac_stream_id,
    with_ma_pixels,
};
use super::tree::emit_learned_tree;
use crate::coder_scratch::CoderScratch;
use crate::ma_tree::{
    LearnedTree, MaLearnParams, MaNode, MaSamples, NUM_MA_PREDS, NUM_MA_PROPS, deepen_ma_tree,
    learn_ma_tree, learn_ma_tree_indexed,
};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Largest |offset| a leaf may carry. The learner keeps hybrid-uint symbols,
/// which are exact packed residuals only below the split (symbols < 16 =
/// residuals −8..=7); larger symbols carry their sign in raw bits the sample
/// does not keep, so they only count towards the zero fraction.
const MA_LEAF_OFFSET_LIMIT: i32 = 8;

const MA_LEAF_EXACT_SYMBOLS: u32 = 16;

/// Leaves with fewer routed samples keep offset 0.
const MA_LEAF_OFFSET_MIN_SAMPLES: u32 = 16;

/// No offsets at all when more than this share of residuals is exactly zero:
/// such content (screenshots, fractals, flat graphics) has no predictor bias
/// to cancel.
const MA_LEAF_OFFSET_MAX_ZERO_FRACTION: f64 = 0.9;

fn unpack_signed(token: u32) -> i32 {
    if token & 1 == 1 {
        -(((token + 1) >> 1) as i32)
    } else {
        (token >> 1) as i32
    }
}

/// Per-context predictor offsets: the median residual of each leaf's chosen
/// predictor over the samples the finished tree routes to it. Shifting a
/// leaf's residuals does not change its own entropy, but it centers the
/// distribution, which lets ANS clustering merge leaves by shape rather than
/// by bias and keeps small residuals inside the hybrid-uint's exact symbols.
pub(super) fn leaf_offsets<'a, T: crate::ma_tree::MaProperty + 'a>(
    tree: &LearnedTree,
    leaf_ctx: &[u32],
    num_ctx: u32,
    samples: impl Iterator<Item = (&'a [T; NUM_MA_PROPS], &'a [u8; NUM_MA_PREDS])>,
) -> Vec<i32> {
    const BINS: usize = 2 * MA_LEAF_OFFSET_LIMIT as usize + 1;
    let mut hist = vec![[0u32; BINS]; num_ctx as usize];
    let mut routed = 0u64;
    for (props, tok) in samples {
        let (node, pred) = tree.lookup(props);
        routed += 1;
        let symbol = tok[pred as usize] as u32;
        if symbol < MA_LEAF_EXACT_SYMBOLS {
            let r = unpack_signed(symbol);
            hist[leaf_ctx[node as usize] as usize][(r + MA_LEAF_OFFSET_LIMIT) as usize] += 1;
        }
    }
    let zeros: u64 = hist
        .iter()
        .map(|h| h[MA_LEAF_OFFSET_LIMIT as usize] as u64)
        .sum();
    if zeros as f64 > MA_LEAF_OFFSET_MAX_ZERO_FRACTION * routed as f64 {
        return vec![0; num_ctx as usize];
    }
    hist.iter()
        .map(|h| {
            let total: u32 = h.iter().sum();
            if total < MA_LEAF_OFFSET_MIN_SAMPLES {
                return 0;
            }
            let half = total.div_ceil(2);
            let mut acc = 0u32;
            for (bin, &c) in h.iter().enumerate() {
                acc += c;
                if acc >= half {
                    return bin as i32 - MA_LEAF_OFFSET_LIMIT;
                }
            }
            0
        })
        .collect()
}

/// Sampling stride for tree learning; odd to avoid column aliasing.
fn ma_sample_stride(total_px: usize, target_samples: usize, full_sample_limit: usize) -> usize {
    let mut stride = total_px.div_ceil(target_samples).max(1);
    if stride == 1 && total_px > full_sample_limit {
        stride = 3;
    }
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    stride
}

/// Header overhead model for a learned tree: tree tokens plus the extra
/// per-context histogram / context-map cost (conservative; clustering merges
/// most of it). Returns the candidate only when it beats the flat estimate.
/// `samples` is `Some` only for candidates that may be written: routing every
/// sample through the tree for the offsets is not worth doing for the
/// ranking stage's throwaway candidates.
pub(super) fn gate_ma_tree(
    tree: LearnedTree,
    sample_scale: f64,
    samples: Option<&MaSamples>,
    leaf_price: Option<f64>,
) -> Option<LearnedCandidate> {
    let mut cand = finish_ma_candidate(tree, sample_scale, leaf_price);
    if cand.est_real.partial_cmp(&cand.flat_real) != Some(std::cmp::Ordering::Less) {
        return None;
    }
    // Rejected trees do not need the sample-routing pass.
    if let Some(samples) = samples {
        cand.set_leaf_offsets(samples, 4);
    }
    Some(cand)
}

/// The writable candidate of a learned tree, whatever its estimate: tree
/// tokens, context ids and leaf offsets. `leaf_price` is the all-in bits
/// per context of a grow-then-prune tree (`ma_leaf_price`), `None` for a
/// best-first tree.
pub(super) fn finish_ma_candidate(
    tree: LearnedTree,
    sample_scale: f64,
    leaf_price: Option<f64>,
) -> LearnedCandidate {
    let (tree_tokens, leaf_ctx, num_ctx) = emit_learned_tree(&tree);
    // Contexts are clustered into at most 128 histograms before coding, so
    // the header grows with the context map (a few bits per context) and
    // the cluster count, not with every context. Charging 200 bits per
    // context had a 4095-leaf tree fail its gate on a grainy photo and hand
    // the frame to the 32-leaf coarse tree (+6%).
    let clusters_bits = (num_ctx.min(128) as f64) * 200.0;
    let overhead_bits = match leaf_price {
        // A pruned tree kept every leaf against this price (its tree tokens,
        // context-map entry and clustering loss together); the best-first
        // formula's ~78 bits per leaf had 1100-leaf trees whose frames coded
        // 2-3% smaller fail the palette gate.
        Some(price) => num_ctx as f64 * price + clusters_bits,
        None => tree_tokens.len() as f64 * 10.0 + clusters_bits + num_ctx as f64 * 8.0,
    };
    let est_real = tree.est_bits * sample_scale + overhead_bits;
    let flat_real = tree.flat_bits * sample_scale;
    let mut leaf_offset = vec![0; num_ctx as usize];
    for (node, &ctx) in tree.nodes.iter().zip(&leaf_ctx) {
        if let MaNode::Leaf { offset, .. } = *node
            && ctx != u32::MAX
        {
            leaf_offset[ctx as usize] = offset;
        }
    }
    let estimated_savings = 1.0 - tree.est_bits / tree.flat_bits.max(f64::MIN_POSITIVE);
    LearnedCandidate {
        tree,
        tree_tokens,
        leaf_ctx,
        leaf_offset,
        num_ctx,
        est_real,
        flat_real,
        estimated_savings,
        constant_ctx: None,
    }
}

/// Parameters of a final learn: `MA_FINAL_SPLIT_COST_BITS`.
fn ma_learn_params_final(
    min_symbol: u32,
    max_leaves: usize,
    max_candidates: usize,
    split_scale: f64,
    use_wp: bool,
    modal_offsets: bool,
) -> MaLearnParams {
    let mut params = ma_learn_params(
        min_symbol,
        max_leaves,
        max_candidates,
        split_scale,
        use_wp,
        modal_offsets,
    );
    params.split_cost_bits = MA_FINAL_SPLIT_COST_BITS / split_scale as f32;
    params
}

pub(super) fn ma_learn_params(
    min_symbol: u32,
    max_leaves: usize,
    max_candidates: usize,
    split_scale: f64,
    use_wp: bool,
    modal_offsets: bool,
) -> MaLearnParams {
    MaLearnParams {
        alphabet: min_symbol as usize,
        max_leaves,
        split_cost_bits: MA_SPLIT_COST_BITS / split_scale as f32,
        min_node: MA_MIN_NODE_SAMPLES,
        allow_wp: use_wp,
        allowed_preds: u16::MAX,
        side_preds: MA_SIDE_PREDS,
        max_candidates,
        modal_offsets,
        prune_leaf_bits: 0.0,
    }
}

/// A final learn of a palette index source (`modal_offsets`) grows then
/// prunes (`MaLearnParams::prune_leaf_bits`); RCT sources keep best-first
/// growth.
pub(super) fn with_ma_prune(mut params: MaLearnParams, split_scale: f64) -> MaLearnParams {
    if params.modal_offsets {
        params.split_cost_bits = MA_PRUNE_SPLIT_COST_BITS / split_scale as f32;
        params.min_node = MA_PRUNE_MIN_NODE_SAMPLES;
        params.prune_leaf_bits = MA_PRUNE_LEAF_BITS / split_scale as f32;
    }
    params
}

/// `finish_ma_candidate`'s per-context price of a tree learned with
/// `params`: a pruned tree's own leaf cost (image domain), else `None`.
pub(super) fn ma_leaf_price(params: &MaLearnParams, split_scale: f64) -> Option<f64> {
    (params.prune_leaf_bits > 0.0).then_some(params.prune_leaf_bits as f64 * split_scale)
}

fn learn_ma_coarse(
    samples: MaSamples,
    stride: usize,
    min_symbol: u32,
    use_wp: bool,
    max_leaves: usize,
    max_candidates: usize,
    leaf_offsets: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<CoarseLearn> {
    if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
        return None;
    }
    if samples.len() <= MA_COARSE_TARGET_SAMPLES {
        let best_first = || {
            ma_learn_params(
                min_symbol,
                max_leaves,
                max_candidates,
                stride as f64,
                use_wp,
                !leaf_offsets,
            )
        };
        let params = with_ma_prune(best_first(), stride as f64);
        let pruned = params.prune_leaf_bits > 0.0;
        let leaf_price = ma_leaf_price(&params, stride as f64);
        let tree = learn_ma_tree(&samples, params, pool, scratch);
        let candidate = match gate_ma_tree(
            tree,
            stride as f64,
            leaf_offsets.then_some(&samples),
            leaf_price,
        ) {
            Some(candidate) => candidate,
            // A pruned tree can keep more contexts than the gate's overhead
            // allows a small source (a glyph atlas: 158 contexts priced at
            // 58.5 K bits against 57.7 K flat), which would hand the source
            // to the fixed trees at three times the size; the best-first
            // tree stands in.
            None if pruned => gate_ma_tree(
                learn_ma_tree(&samples, best_first(), pool, scratch),
                stride as f64,
                leaf_offsets.then_some(&samples),
                None,
            )?,
            None => return None,
        };
        return Some(CoarseLearn {
            samples,
            stride,
            max_candidates,
            leaf_offsets,
            candidate,
            deepen: false,
        });
    }
    let coarse_indices = samples.evenly_sampled_indices(MA_COARSE_TARGET_SAMPLES);
    let coarse_scale = stride as f64 * samples.len() as f64 / coarse_indices.len() as f64;
    let coarse_tree = learn_ma_tree_indexed(
        &samples,
        coarse_indices,
        ma_learn_params(
            min_symbol,
            MA_COARSE_MAX_LEAVES,
            max_candidates,
            coarse_scale,
            use_wp,
            !leaf_offsets,
        ),
        pool,
        scratch,
    );
    let candidate = gate_ma_tree(coarse_tree, coarse_scale, None, None)?;
    let coarse_leaves = candidate.tree.nodes.len().div_ceil(2);
    let coarse_savings = 1.0 - candidate.est_real / candidate.flat_real;
    let deepen = coarse_leaves == MA_COARSE_MAX_LEAVES && coarse_savings >= MA_DEEPEN_MIN_SAVINGS;
    Some(CoarseLearn {
        samples,
        stride,
        max_candidates,
        leaf_offsets,
        candidate,
        deepen,
    })
}

/// Second stage: re-score the coarse tree on the full probe and keep growing
/// it, falling back to the coarse tree when the deep one fails its gate.
pub(super) fn finish_ma_learn(
    speed: crate::Speed,
    coarse: CoarseLearn,
    min_symbol: u32,
    use_wp: bool,
    max_leaves: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedCandidate {
    if !coarse.deepen {
        return coarse.candidate;
    }
    let mut params = with_ma_prune(
        ma_learn_params_final(
            min_symbol,
            max_leaves,
            coarse.max_candidates,
            coarse.stride as f64,
            use_wp,
            !coarse.leaf_offsets,
        ),
        coarse.stride as f64,
    );
    if !speed.effort().tree_deep_final && coarse.stride > 1 {
        params.max_leaves = params.max_leaves.min(MA_LITE_MAX_LEAVES);
        params.max_candidates = params.max_candidates.min(MA_LITE_CANDIDATES);
    }
    // Palette index sources keep the narrow side search: like extra split
    // candidates, it overfits their near-deterministic contexts.
    if coarse.leaf_offsets && speed.effort().tree_wide_side_preds() {
        params.side_preds = MA_WIDE_SIDE_PREDS;
    }
    let leaf_price = ma_leaf_price(&params, coarse.stride as f64);
    let (deep_tree, samples) = deepen_ma_tree(
        coarse.samples,
        params,
        coarse.candidate.tree.clone(),
        pool,
        scratch,
    );
    gate_ma_tree(
        deep_tree,
        coarse.stride as f64,
        coarse.leaf_offsets.then_some(&samples),
        leaf_price,
    )
    .unwrap_or(coarse.candidate)
}

/// Presets are ranked only when the weighted predictor matters to the
/// first-stage tree: at least this share of its rows reach a WP leaf, or at
/// least `WP_RANK_MIN_SPLIT_SHARE` of its splits test the WP-error property
/// (the preset shapes that property too: a line-art image splitting on it in
/// 20% of nodes lost 0.4% without the ranking). Text and UI trees do neither
/// (0-4% of rows, no such splits; photos route 27-75% of rows), and there the
/// other presets never changed the frame beyond a few bytes while their
/// sampling walks were a quarter to a third of the encode.
const WP_RANK_MIN_ROW_SHARE: f64 = 0.02;

const WP_RANK_MIN_SPLIT_SHARE: f64 = 0.15;

/// Whether the WP preset can matter to `stage`'s tree (see the gate above;
/// every 16th row is routed, so the row share is a gate, not a rate).
fn wp_matters(stage: &CoarseLearn) -> bool {
    let tree = &stage.candidate.tree;
    let (mut splits, mut wp_splits) = (0usize, 0usize);
    for node in &tree.nodes {
        if let crate::ma_tree::MaNode::Split { prop, .. } = node {
            splits += 1;
            wp_splits += usize::from(*prop as usize == PROP_WP as usize);
        }
    }
    if wp_splits as f64 >= WP_RANK_MIN_SPLIT_SHARE * splits as f64 && wp_splits > 0 {
        return true;
    }
    let mut rows = 0usize;
    let mut wp_rows = 0usize;
    for props in stage.samples.props.iter().step_by(16) {
        rows += 1;
        if tree.lookup(&props).1 as usize == crate::ma_tree::PRED_WEIGHTED {
            wp_rows += 1;
        }
    }
    wp_rows as f64 >= WP_RANK_MIN_ROW_SHARE * rows as f64
}

/// Rank the WP presets of `source` by their first-stage learns, starting
/// from `first` (already learned under `first_params`), the rest in order
/// of how often they win; a preset trailing the best first stage seen by
/// more than `RANK_COARSE_PRUNE` is dropped. Without `all_presets` only
/// `first` is scored. Returns (best bits, best coarse bits, preset) and, with
/// `keep_stage`, the preset's stage.
#[allow(clippy::too_many_arguments)]
pub(super) fn rank_presets(
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    use_wp: bool,
    first: Option<CoarseLearn>,
    first_params: WpParams,
    header_bits: &dyn Fn(WpParams) -> f64,
    all_presets: bool,
    layout: GroupLayout,
    sampling: (usize, usize),
    keep_stage: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (f64, f64, WpParams, Option<CoarseLearn>) {
    let mut best_stage = None;
    let mut best_bits = f64::INFINITY;
    let mut best_coarse = f64::INFINITY;
    let mut best_params = first_params;
    let mut order: Vec<WpParams> = vec![first_params];
    if use_wp && all_presets && first.as_ref().is_none_or(wp_matters) {
        for &preset in [1usize, 2, 3, 0].iter().map(|&i| &WpParams::PRESETS[i]) {
            if preset != first_params {
                order.push(preset);
            }
        }
    }
    let mut first = first;
    for preset in order {
        let stage = match first.take() {
            Some(stage) => Some(stage),
            None => rank_coarse(
                source, xsize, ysize, min_symbol, preset, use_wp, layout, sampling, pool, scratch,
            ),
        };
        let Some(stage) = stage else {
            continue;
        };
        let coarse = stage.candidate.est_real + header_bits(preset);
        if coarse > best_coarse * RANK_COARSE_PRUNE {
            continue;
        }
        best_coarse = best_coarse.min(coarse);
        let bits = rank_bits(&stage) + header_bits(preset);
        if bits < best_bits {
            best_bits = bits;
            best_params = preset;
            best_stage = keep_stage.then_some(stage);
        }
    }
    (best_bits, best_coarse, best_params, best_stage)
}

/// Sample every stream of `source` under `layout` (group-local coordinates
/// and a fresh WP state per group, exactly as the decoder will see them) and
/// run the first learning stage. `None` when no tree is estimated to beat
/// the flat path.
#[allow(clippy::too_many_arguments)]
pub(super) fn learn_ma_candidate(
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
    min_symbol: u32,
    wp_params: WpParams,
    use_wp: bool,
    max_leaves: usize,
    (target_samples, full_sample_limit): (usize, usize),
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<CoarseLearn> {
    let gdim = layout.dim();
    let xsize_groups = xsize.div_ceil(gdim);
    let num_groups = xsize_groups * ysize.div_ceil(gdim);
    let placement = source.placement(layout, num_groups == 1);
    let max_candidates = source.max_candidates();
    let stride = ma_sample_stride(source.total_values(), target_samples, full_sample_limit);

    // Sampling jobs: global streams first, then group-major / channel-minor.
    struct Job<'a> {
        channel: usize,
        chan_id: u32,
        stream: i32,
        refs: Vec<MaRefPlane<'a>>,
        group_override: Option<usize>,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
    }
    let num_dc_groups = xsize.div_ceil(layout.lf_dim()) * ysize.div_ceil(layout.lf_dim());
    let mut jobs: Vec<Job> = Vec::new();
    for p in placement.iter().filter(|p| p.global) {
        let ch = &source.channels[p.channel];
        if ch.constant.is_some() {
            continue;
        }
        jobs.push(Job {
            channel: p.channel,
            chan_id: p.chan_id,
            stream: 0,
            refs: source.ref_planes(&placement, p, 0, 0),
            group_override: None,
            x0: 0,
            y0: 0,
            w: ch.w,
            h: ch.h,
        });
    }
    if num_groups > 1 {
        for group_index in 0..num_groups {
            let x0 = (group_index % xsize_groups) * gdim;
            let y0 = (group_index / xsize_groups) * gdim;
            if let Some(over) = source.group_override(group_index) {
                // A palettised group codes its own channels whole; reference
                // properties would compare a color table with an index
                // plane, so it has none.
                for (channel, ch) in over.channels.iter().enumerate() {
                    jobs.push(Job {
                        channel,
                        chan_id: channel as u32,
                        stream: ac_stream_id(num_dc_groups, group_index),
                        refs: Vec::new(),
                        group_override: Some(group_index),
                        x0: 0,
                        y0: 0,
                        w: ch.w,
                        h: ch.h,
                    });
                }
                continue;
            }
            for p in placement.iter().filter(|p| !p.global) {
                let ch = &source.channels[p.channel];
                if ch.constant.is_some() {
                    continue;
                }
                jobs.push(Job {
                    channel: p.channel,
                    chan_id: p.chan_id,
                    stream: ac_stream_id(num_dc_groups, group_index),
                    refs: source.ref_planes(&placement, p, x0, y0),
                    group_override: None,
                    x0,
                    y0,
                    w: gdim.min(ch.w - x0),
                    h: gdim.min(ch.h - y0),
                });
            }
        }
    }

    // Each job owns a disjoint range in the final sample arrays. The rotating
    // row phase gives this exact count, including narrow and partial groups.
    let counts: Vec<usize> = jobs
        .iter()
        .map(|job| ma_channel_sample_count(job.w, job.h, stride))
        .collect();
    let total_samples = counts.iter().sum();
    let mut samples = MaSamples {
        props: crate::ma_tree::MaProperties::zeroed(total_samples),
        tok: vec![[0; NUM_MA_PREDS]; total_samples],
        val: vec![0; total_samples],
    };
    loop {
        let overflow = crate::ma_tree::with_ma_properties!(&mut samples.props, |props| {
            let mut props = props.as_mut_slice();
            let mut tok = samples.tok.as_mut_slice();
            let mut val = samples.val.as_mut_slice();
            let mut ranges = Vec::with_capacity(jobs.len());
            for &count in &counts {
                let (p, rest_p) = props.split_at_mut(count);
                let (t, rest_t) = tok.split_at_mut(count);
                let (v, rest_v) = val.split_at_mut(count);
                ranges.push((p, t, v, false));
                (props, tok, val) = (rest_p, rest_t, rest_v);
            }
            pool.steal_for_each_mut(scratch, &mut ranges, |i, range, _scratch| {
                let job = &jobs[i];
                let ch = source.channel_at(job.group_override, job.channel);
                let mut rows = range
                    .0
                    .iter_mut()
                    .zip(range.1.iter_mut())
                    .zip(range.2.iter_mut());
                with_ma_pixels!(ch, |pixels| sample_channel_ma(
                    |y| &pixels[(job.y0 + y) * ch.w + job.x0..][..job.w],
                    job.w,
                    job.h,
                    job.chan_id,
                    job.stream,
                    &job.refs,
                    wp_params,
                    use_wp,
                    stride,
                    |p, t, v| {
                        let ((props, tok), val) = rows.next().expect("sample count");
                        range.3 |= !crate::ma_tree::write_property_row(props, p);
                        *tok = t;
                        *val = v;
                    },
                ));
                debug_assert!(rows.next().is_none());
            });
            ranges.iter().any(|range| range.3)
        });
        if !overflow {
            break;
        }
        // Retrying in wide storage preserves the exact sampling phases and
        // order. Release the narrow buffer before allocating its replacement;
        // the wide fallback does not keep both full property arrays alive.
        samples.props = crate::ma_tree::MaProperties::Wide(Vec::new());
        samples.props = crate::ma_tree::MaProperties::Wide(vec![[0; NUM_MA_PROPS]; total_samples]);
    }
    learn_ma_coarse(
        samples,
        stride,
        min_symbol,
        use_wp,
        max_leaves,
        max_candidates,
        matches!(source.transform, MaTransform::Rct(_)),
        pool,
        scratch,
    )
}

/// Leaf budget of the ranking learns: the first-stage budget, also for
/// sample sets small enough to be learned directly (at most
/// `MA_COARSE_TARGET_SAMPLES` rows, i.e. images under ~20k pixels), where a
/// 96-leaf learn per preset and candidate was most of a small icon's encode
/// time (a 128x128 icon: 0.21 s single-threaded, 0.14 s of it ranking).
const MA_RANK_LEAVES: usize = MA_COARSE_MAX_LEAVES;

/// The estimator's RCT runner-up is scored by the ranking learner when its
/// estimated cost is within this factor of the best.
pub(super) const RCT_RANK_MARGIN: f32 = 1.02;

/// A ranking candidate whose 32-leaf coarse estimate trails the best coarse
/// estimate seen so far by more than this is not scored further.
pub(super) const RANK_COARSE_PRUNE: f64 = 1.005;

/// First ranking stage of `source`: its coarse learn under `layout` at
/// `sampling`, the final learn's budget. The 32-leaf first stage
/// only sees `MA_COARSE_TARGET_SAMPLES` evenly spaced rows of the set, yet
/// sampling just twice that many rows at a wider stride mis-ranked presets
/// on a third of Kodak (+0.22% overall, k01 +1.3%): the preset margins are
/// 0.3-0.6% and need the denser set's row phases.
#[allow(clippy::too_many_arguments)]
pub(super) fn rank_coarse(
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    wp_params: WpParams,
    use_wp: bool,
    layout: GroupLayout,
    sampling: (usize, usize),
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<CoarseLearn> {
    learn_ma_candidate(
        source,
        xsize,
        ysize,
        layout,
        min_symbol,
        wp_params,
        use_wp,
        MA_RANK_LEAVES,
        sampling,
        pool,
        scratch,
    )
}

/// Estimated bits of a ranking learn: its first stage, or the flat
/// alternative when the tree does not qualify. Deepening the ranking learns
/// to 96 leaves first ranked no better than the 32-leaf stage (Kodak
/// +0.003%, a UI screenshot −2.0%) and cost a quarter of the encode.
pub(super) fn rank_bits(stage: &CoarseLearn) -> f64 {
    stage.candidate.est_real.min(stage.candidate.flat_real)
}

/// First learning stage under every layout, second stage for the preferred
/// (first) layout and for any later layout whose first stage already
/// estimates smaller; the full estimates pick the winner. Layouts are
/// processed one at a time so a single sample set is alive. `coarse_limit`
/// prunes a source by its first stage, `final_limit` by its first complete
/// layout. `first_stage`, when given, is the first layout's stage.
#[allow(clippy::too_many_arguments)]
pub(super) fn learn_best_layout(
    speed: crate::Speed,
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    layouts: &[GroupLayout],
    min_symbol: u32,
    wp_params: WpParams,
    use_wp: bool,
    coarse_limit: f64,
    final_limit: f64,
    deepen_all: bool,
    sampling: (usize, usize),
    mut first_stage: Option<CoarseLearn>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<(GroupLayout, LearnedCandidate, f64)> {
    let mut best_coarse_est = f64::INFINITY;
    let mut learned: Option<(GroupLayout, LearnedCandidate)> = None;
    for (index, &layout) in layouts.iter().enumerate() {
        // A source whose complete first layout already trails `final_limit`
        // cannot be written (the palette candidate against the RGB frame's
        // final estimate); its other layouts are not learned.
        if index > 0
            && let Some((_, best)) = &learned
            && best.est_real > final_limit
        {
            break;
        }
        let stage = first_stage.take().or_else(|| {
            learn_ma_candidate(
                source,
                xsize,
                ysize,
                layout,
                min_symbol,
                wp_params,
                use_wp,
                source.max_leaves(),
                sampling,
                pool,
                scratch,
            )
        });
        let Some(stage) = stage else {
            continue;
        };
        let coarse_est = stage.candidate.est_real;
        // A source whose first stage already trails `coarse_limit` is not
        // deepened (the palette candidate against the RGB one). Between this
        // source's own layouts the coarse estimate mis-ranks palette index
        // images (it prefers 1024 where the accurate estimate prefers 256, a
        // ~4% loss), so `deepen_all` sources deepen every layout and pick by
        // the final estimate; photos keep the cheap coarse prune.
        let layout_pruned = index > 0 && !deepen_all && coarse_est > best_coarse_est;
        if coarse_est > coarse_limit || layout_pruned {
            best_coarse_est = best_coarse_est.min(coarse_est);
            continue;
        }
        best_coarse_est = best_coarse_est.min(coarse_est);
        let cand = finish_ma_learn(
            speed,
            stage,
            min_symbol,
            use_wp,
            source.max_leaves(),
            pool,
            scratch,
        );
        if learned
            .as_ref()
            .is_none_or(|(_, best)| cand.est_real < best.est_real)
        {
            learned = Some((layout, cand));
        }
    }
    learned.map(|(layout, cand)| (layout, cand, best_coarse_est))
}
