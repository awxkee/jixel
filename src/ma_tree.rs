/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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

//! Learned MA (meta-adaptive) context-tree for the Slow lossless path.

use crate::adaptive_quant::dirty_log2f;
use crate::coder_scratch::CoderScratch;
use crate::thread_pool::ThreadPool;

/// Property vector length (libjxl ids 0..=15). Index 1 (stream id) is always 0.
pub(crate) const NUM_MA_PROPS: usize = 16;
/// Decoder predictors 0..=13 (libjxl `Predictor` enum order).
pub(crate) const NUM_MA_PREDS: usize = 14;
/// Properties the learner may split on (skips the constant stream id).
const SPLIT_PROPS: [u8; 15] = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Split candidates, minus the WP-error property (last) when WP is excluded.
fn split_props(allow_wp: bool) -> &'static [u8] {
    if !allow_wp {
        &SPLIT_PROPS[..SPLIT_PROPS.len() - 1]
    } else {
        &SPLIT_PROPS
    }
}

/// Weighted predictor id in the decoder predictor enumeration.
const PRED_WEIGHTED: usize = 6;

/// Split candidates examined per property per node. Sixteen quantiles retain
/// nearly all the useful threshold resolution while bounding the repeated
/// histogram work performed for every property of every grown leaf.
const MAX_CANDIDATES: usize = 16;
/// Strided value subset used to derive candidate thresholds.
const MAX_QUANTILE_PROBE: usize = 1024;
/// Hard depth guard (decoder-side property lookups stay cheap).
const MAX_DEPTH: usize = 26;
/// Below this size, distributing fifteen short property scans costs more than
/// it saves. Large nodes dominate learning time and have enough work to keep
/// the existing encoder workers busy.
const PARALLEL_PROPERTY_MIN_SAMPLES: usize = 4 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MaNode {
    /// `prop > val` routes to `gt`, else `le` (matches libjxl tree decode).
    Split {
        prop: u8,
        val: i32,
        gt: u32,
        le: u32,
    },
    Leaf {
        pred: u32,
    },
}

#[derive(Clone)]
pub(crate) struct LearnedTree {
    pub(crate) nodes: Vec<MaNode>,
    /// Estimated coded bits of all samples routed through the tree
    /// (sample domain; multiply by the sampling stride for image-domain bits).
    pub(crate) est_bits: f64,
    /// Estimated bits of the flat alternative: split by channel only, best
    /// single predictor per channel (the existing non-tree path's shape).
    pub(crate) flat_bits: f64,
}

impl LearnedTree {
    /// Route a property vector to (leaf node index, predictor id).
    #[inline]
    pub(crate) fn lookup(&self, p: &[i32; NUM_MA_PROPS]) -> (u32, u32) {
        let mut i = 0usize;
        loop {
            match self.nodes[i] {
                MaNode::Split { prop, val, gt, le } => {
                    i = if p[prop as usize] > val { gt } else { le } as usize;
                }
                MaNode::Leaf { pred } => return (i as u32, pred),
            }
        }
    }
}

pub(crate) struct MaSamples {
    pub(crate) props: Vec<[i32; NUM_MA_PROPS]>,
    pub(crate) tok: Vec<[u8; NUM_MA_PREDS]>,
    pub(crate) nbits: Vec<[u8; NUM_MA_PREDS]>,
}

impl MaSamples {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_capacity(0)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            props: Vec::with_capacity(capacity),
            tok: Vec::with_capacity(capacity),
            nbits: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub(crate) fn push(
        &mut self,
        props: [i32; NUM_MA_PROPS],
        tok: [u8; NUM_MA_PREDS],
        nbits: [u8; NUM_MA_PREDS],
    ) {
        self.props.push(props);
        self.tok.push(tok);
        self.nbits.push(nbits);
    }

    pub(crate) fn len(&self) -> usize {
        self.props.len()
    }

    pub(crate) fn append(&mut self, other: &mut MaSamples) {
        self.props.append(&mut other.props);
        self.tok.append(&mut other.tok);
        self.nbits.append(&mut other.nbits);
    }

    /// Deterministic evenly spaced view used by the coarse learning stage.
    /// The learner already partitions sample indices, so retaining only this
    /// mapping avoids copying the much larger property and predictor records.
    pub(crate) fn evenly_sampled_indices(&self, target: usize) -> Vec<u32> {
        let selected = self.len().min(target);
        let mut indices = Vec::with_capacity(selected);
        for sample in 0..selected {
            indices.push((sample * self.len() / selected) as u32);
        }
        indices
    }
}

pub(crate) struct MaLearnParams {
    pub(crate) alphabet: usize,
    pub(crate) max_leaves: usize,
    pub(crate) split_cost_bits: f32,
    pub(crate) min_node: usize,
    /// Whether leaves may use the Weighted Predictor and splits the WP-error
    /// property. Both force per-pixel WP state in the decoder.
    pub(crate) allow_wp: bool,
}

/// Reusable scratch for scoring one property of one MA-tree node. One lives in
/// each encoder worker's `CoderScratch`, allowing independent properties to be
/// evaluated concurrently without per-node allocation.
pub(crate) struct MaPropertyScratch {
    bin_hist: Vec<u32>,
    bin_nbits: [u64; MAX_CANDIDATES + 1],
    left_hist: Vec<u32>,
    probe: Vec<i32>,
    cands: Vec<i32>,
}

impl Default for MaPropertyScratch {
    fn default() -> Self {
        Self {
            bin_hist: Vec::new(),
            bin_nbits: [0; MAX_CANDIDATES + 1],
            left_hist: Vec::new(),
            probe: Vec::with_capacity(MAX_QUANTILE_PROBE + 1),
            cands: Vec::with_capacity(MAX_CANDIDATES),
        }
    }
}

impl MaPropertyScratch {
    fn prepare(&mut self, alphabet: usize) {
        let bins = (MAX_CANDIDATES + 1) * alphabet;
        if self.bin_hist.len() < bins {
            self.bin_hist.resize(bins, 0);
        }
        if self.left_hist.len() < alphabet {
            self.left_hist.resize(alphabet, 0);
        }
    }
}

struct Learner<'a> {
    s: &'a MaSamples,
    p: MaLearnParams,
    nodes: Vec<MaNode>,
    leaves: u32,
    est_bits: f64,
    // Reused per-node scratch.
    all_hist: Vec<u32>, // NUM_MA_PREDS x alphabet
    all_nbits: [u64; NUM_MA_PREDS],
}

#[inline]
fn hist_entropy_bits(hist: &[u32], total: u32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let log_total = dirty_log2f(total as f32);
    let mut bits = 0.0f32;
    for &c in hist {
        if c != 0 {
            bits += c as f32 * (log_total - dirty_log2f(c as f32));
        }
    }
    bits
}

fn pick_candidates(
    samples: &MaSamples,
    idx: &[u32],
    prop: usize,
    scratch: &mut MaPropertyScratch,
) -> usize {
    scratch.probe.clear();
    let mut stride = (idx.len() / MAX_QUANTILE_PROBE).max(1);
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    let mut i = 0;
    while i < idx.len() {
        scratch.probe.push(samples.props[idx[i] as usize][prop]);
        i += stride;
    }
    scratch.probe.sort_unstable();
    scratch.probe.dedup();
    if scratch.probe.len() < 2 {
        return 0;
    }
    // Thresholds at even quantiles of the distinct probe values; the max value
    // is excluded so the `gt` side is never structurally empty.
    scratch.cands.clear();
    let distinct = scratch.probe.len() - 1;
    let n = distinct.min(MAX_CANDIDATES);
    for k in 0..n {
        let pos = k * distinct / n;
        let v = scratch.probe[pos];
        if scratch.cands.last() != Some(&v) {
            scratch.cands.push(v);
        }
    }
    scratch.cands.len()
}

/// Best split of `idx` on `prop`, scoring with `search_pred` fixed on both
/// sides. The caller supplies the node's total histogram for that predictor.
fn best_split_on_prop(
    samples: &MaSamples,
    idx: &[u32],
    prop: usize,
    search_pred: usize,
    alphabet: usize,
    total_hist: &[u32],
    scratch: &mut MaPropertyScratch,
) -> Option<(i32, f32)> {
    scratch.prepare(alphabet);
    let ncand = pick_candidates(samples, idx, prop, scratch);
    if ncand == 0 {
        return None;
    }
    scratch.bin_hist[..(ncand + 1) * alphabet].fill(0);
    scratch.bin_nbits[..ncand + 1].fill(0);
    let mut total_nbits = 0u64;
    let mut bin_count = [0u32; MAX_CANDIDATES + 1];
    for &i in idx {
        let v = samples.props[i as usize][prop];
        // bin = number of candidates < v; v <= cands[j] iff bin <= j.
        let bin = scratch.cands.partition_point(|&c| c < v);
        let tok = samples.tok[i as usize][search_pred] as usize;
        let nb = samples.nbits[i as usize][search_pred] as u64;
        scratch.bin_hist[bin * alphabet + tok] += 1;
        scratch.bin_nbits[bin] += nb;
        bin_count[bin] += 1;
        total_nbits += nb;
    }
    let total_count = idx.len() as u32;

    scratch.left_hist[..alphabet].fill(0);
    let mut left_nbits = 0u64;
    let mut left_count = 0u32;
    let mut best: Option<(i32, f32)> = None;
    for j in 0..ncand {
        for (left, &bin) in scratch.left_hist[..alphabet]
            .iter_mut()
            .zip(&scratch.bin_hist[j * alphabet..(j + 1) * alphabet])
        {
            *left += bin;
        }
        left_nbits += scratch.bin_nbits[j];
        left_count += bin_count[j];
        if left_count == 0 || left_count == total_count {
            continue;
        }
        let cost_l =
            hist_entropy_bits(&scratch.left_hist[..alphabet], left_count) + left_nbits as f32;
        let mut cost_r_hist_bits = 0.0f32;
        let right_count = total_count - left_count;
        let log_total = dirty_log2f(right_count as f32);
        for (&total, &left) in total_hist.iter().zip(&scratch.left_hist[..alphabet]) {
            let c = total - left;
            if c != 0 {
                cost_r_hist_bits += c as f32 * (log_total - dirty_log2f(c as f32));
            }
        }
        let cost_r = cost_r_hist_bits + (total_nbits - left_nbits) as f32;
        let cost = cost_l + cost_r;
        if best.is_none() || cost < best.unwrap().1 {
            best = Some((scratch.cands[j], cost));
        }
    }
    best
}

impl<'a> Learner<'a> {
    /// Token-histogram cost of `idx` under every predictor; returns
    /// (best bits incl. raw bits, best predictor). Fills `all_hist`/`all_nbits`.
    fn node_cost_all(&mut self, idx: &[u32]) -> (f32, usize) {
        let alpha = self.p.alphabet;
        self.all_hist[..NUM_MA_PREDS * alpha].fill(0);
        self.all_nbits = [0; NUM_MA_PREDS];
        for &i in idx {
            let toks = &self.s.tok[i as usize];
            let nbs = &self.s.nbits[i as usize];
            for p in 0..NUM_MA_PREDS {
                self.all_hist[p * alpha + toks[p] as usize] += 1;
                self.all_nbits[p] += nbs[p] as u64;
            }
        }
        let total = idx.len() as u32;
        let skip_wp = !self.p.allow_wp;
        let mut best = f32::INFINITY;
        let mut best_pred = 0usize;
        for p in 0..NUM_MA_PREDS {
            if skip_wp && p == PRED_WEIGHTED {
                continue;
            }
            let bits = hist_entropy_bits(&self.all_hist[p * alpha..(p + 1) * alpha], total)
                + self.all_nbits[p] as f32;
            if bits < best {
                best = bits;
                best_pred = p;
            }
        }
        (best, best_pred)
    }

    fn evaluate(
        &mut self,
        node_id: u32,
        start: usize,
        end: usize,
        depth: u16,
        idx: &[u32],
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Pending {
        let range = &idx[start..end];
        let (base_bits, base_pred) = self.node_cost_all(range);
        if let MaNode::Leaf { pred } = &mut self.nodes[node_id as usize] {
            *pred = base_pred as u32;
        }
        let mut split = None;
        let mut gain = 0.0f32;
        if range.len() >= self.p.min_node && (depth as usize) < MAX_DEPTH {
            let mut best: Option<(usize, i32, f32)> = None; // (prop, threshold, bits)
            let alphabet = self.p.alphabet;
            let total_hist =
                self.all_hist[base_pred * alphabet..(base_pred + 1) * alphabet].to_vec();
            let samples = self.s;
            let props = split_props(self.p.allow_wp);
            let scores = if range.len() >= PARALLEL_PROPERTY_MIN_SAMPLES && pool.num_threads() > 1 {
                pool.steal_map(scratch, props.len(), |job, worker_scratch| {
                    let prop = props[job] as usize;
                    best_split_on_prop(
                        samples,
                        range,
                        prop,
                        base_pred,
                        alphabet,
                        &total_hist,
                        &mut worker_scratch.ma_property,
                    )
                })
            } else {
                props
                    .iter()
                    .map(|&prop| {
                        best_split_on_prop(
                            samples,
                            range,
                            prop as usize,
                            base_pred,
                            alphabet,
                            &total_hist,
                            &mut scratch.ma_property,
                        )
                    })
                    .collect()
            };
            for (&prop, score) in props.iter().zip(scores) {
                if let Some((t, bits)) = score
                    && (best.is_none() || bits < best.unwrap().2)
                {
                    best = Some((prop as usize, t, bits));
                }
            }
            if let Some((prop, threshold, bits)) = best {
                let g = base_bits - bits - self.p.split_cost_bits;
                if g > 0.0 {
                    split = Some((prop as u8, threshold));
                    gain = g;
                }
            }
        }
        Pending {
            node_id,
            start,
            end,
            depth,
            base_bits,
            split,
            gain,
        }
    }
}

/// A grown-but-unsplit leaf with its best available split, ordered by gain.
struct Pending {
    node_id: u32,
    start: usize,
    end: usize,
    depth: u16,
    base_bits: f32,
    split: Option<(u8, i32)>,
    gain: f32,
}

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pending {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gain.total_cmp(&other.gain)
    }
}

fn new_learner(samples: &MaSamples, params: MaLearnParams, nodes: Vec<MaNode>) -> Learner<'_> {
    let alpha = params.alphabet;
    let leaves = nodes
        .iter()
        .filter(|node| matches!(node, MaNode::Leaf { .. }))
        .count() as u32;
    Learner {
        s: samples,
        nodes,
        leaves,
        est_bits: 0.0,
        all_hist: vec![0u32; NUM_MA_PREDS * alpha],
        all_nbits: [0; NUM_MA_PREDS],
        p: params,
    }
}

fn flat_cost(learner: &mut Learner<'_>, sample_indices: &[u32]) -> f64 {
    let samples = learner.s;
    let mut flat_bits = 0.0f64;
    let per_channel = sample_indices.len().div_ceil(4);
    let mut chan_idx: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::with_capacity(per_channel));
    for &sample in sample_indices {
        let c = samples.props[sample as usize][0].clamp(0, 3) as usize;
        chan_idx[c].push(sample);
    }
    for channel in &chan_idx {
        if !channel.is_empty() {
            let (bits, _) = learner.node_cost_all(channel);
            flat_bits += bits as f64;
        }
    }
    flat_bits
}

fn grow_tree(
    mut learner: Learner<'_>,
    mut idx: Vec<u32>,
    mut heap: std::collections::BinaryHeap<Pending>,
    flat_bits: f64,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedTree {
    while learner.leaves < learner.p.max_leaves as u32 {
        let Some(p) = heap.pop() else { break };
        let (prop, threshold) = p.split.expect("only splittable nodes are queued");
        // Partition idx[p.start..p.end]: gt side first, then le.
        let mut lo = p.start;
        let mut hi = p.end;
        while lo < hi {
            if learner.s.props[idx[lo] as usize][prop as usize] > threshold {
                lo += 1;
            } else {
                hi -= 1;
                idx.swap(lo, hi);
            }
        }
        if lo == p.start || lo == p.end {
            // Quantile candidate straddled no actual sample boundary.
            continue;
        }

        let gt_id = learner.nodes.len() as u32;
        learner.nodes.push(MaNode::Leaf { pred: 0 });
        let le_id = learner.nodes.len() as u32;
        learner.nodes.push(MaNode::Leaf { pred: 0 });
        learner.nodes[p.node_id as usize] = MaNode::Split {
            prop,
            val: threshold,
            gt: gt_id,
            le: le_id,
        };
        learner.leaves += 1;

        let gt = learner.evaluate(gt_id, p.start, lo, p.depth + 1, &idx, pool, scratch);
        let le = learner.evaluate(le_id, lo, p.end, p.depth + 1, &idx, pool, scratch);
        learner.est_bits += (gt.base_bits + le.base_bits - p.base_bits) as f64;
        if gt.split.is_some() {
            heap.push(gt);
        }
        if le.split.is_some() {
            heap.push(le);
        }
    }

    LearnedTree {
        est_bits: learner.est_bits,
        flat_bits,
        nodes: learner.nodes,
    }
}

/// Learn a context tree from `samples`. Never fails; a degenerate sample set
/// yields a single-leaf tree.
pub(crate) fn learn_ma_tree(
    samples: &MaSamples,
    params: MaLearnParams,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedTree {
    let idx: Vec<u32> = (0..samples.len() as u32).collect();
    learn_ma_tree_indexed(samples, idx, params, pool, scratch)
}

/// Learns from an index-backed selection of `samples`. The index vector is
/// consumed as the learner's mutable partition, so the coarse stage needs no
/// copied property/token records and no second index allocation.
pub(crate) fn learn_ma_tree_indexed(
    samples: &MaSamples,
    idx: Vec<u32>,
    params: MaLearnParams,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedTree {
    let n = idx.len();
    let mut learner = new_learner(samples, params, vec![MaNode::Leaf { pred: 0 }]);
    let flat_bits = flat_cost(&mut learner, &idx);
    let mut heap = std::collections::BinaryHeap::new();
    let root = learner.evaluate(0, 0, n, 0, &idx, pool, scratch);
    learner.est_bits = root.base_bits as f64;
    if root.split.is_some() {
        heap.push(root);
    }
    grow_tree(learner, idx, heap, flat_bits, pool, scratch)
}

/// Re-score a coarse tree on the full sample set and continue best-first
/// growth from its leaves. The expensive upper topology is retained instead
/// of being rediscovered by the deep stage.
pub(crate) fn deepen_ma_tree(
    samples: &MaSamples,
    params: MaLearnParams,
    seed: LearnedTree,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedTree {
    let depths = {
        let mut depths = vec![0u16; seed.nodes.len()];
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            if let MaNode::Split { gt, le, .. } = seed.nodes[node] {
                depths[gt as usize] = depths[node] + 1;
                depths[le as usize] = depths[node] + 1;
                stack.push(gt as usize);
                stack.push(le as usize);
            }
        }
        depths
    };

    // Stable counting partition by coarse leaf. Each leaf becomes one
    // contiguous range that can participate in the existing in-place growth.
    let mut leaf_for_sample = Vec::with_capacity(samples.len());
    let mut counts = vec![0usize; seed.nodes.len()];
    for props in &samples.props {
        let (leaf, _) = seed.lookup(props);
        leaf_for_sample.push(leaf);
        counts[leaf as usize] += 1;
    }
    let mut offsets = vec![0usize; seed.nodes.len() + 1];
    for node in 0..seed.nodes.len() {
        offsets[node + 1] = offsets[node] + counts[node];
    }
    let mut cursors = offsets[..seed.nodes.len()].to_vec();
    let mut idx = vec![0u32; samples.len()];
    for (sample, &leaf) in leaf_for_sample.iter().enumerate() {
        let cursor = &mut cursors[leaf as usize];
        idx[*cursor] = sample as u32;
        *cursor += 1;
    }

    let mut learner = new_learner(samples, params, seed.nodes);
    let flat_bits = flat_cost(&mut learner, &idx);
    let mut heap = std::collections::BinaryHeap::new();
    for node in 0..learner.nodes.len() {
        if !matches!(learner.nodes[node], MaNode::Leaf { .. }) || counts[node] == 0 {
            continue;
        }
        let pending = learner.evaluate(
            node as u32,
            offsets[node],
            offsets[node + 1],
            depths[node],
            &idx,
            pool,
            scratch,
        );
        learner.est_bits += pending.base_bits as f64;
        if pending.split.is_some() {
            heap.push(pending);
        }
    }
    grow_tree(learner, idx, heap, flat_bits, pool, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::{pack_signed, uint_encode};

    /// Synthetic set: property 4 (|N|) cleanly separates two populations that
    /// prefer different predictors. The learner must find that split, and
    /// lookup must route samples to leaves whose predictor matches.
    #[test]
    fn learns_a_discriminating_split() {
        let mut s = MaSamples::new();
        let mut push = |prop4: i32, good_pred: usize| {
            let mut props = [0i32; NUM_MA_PROPS];
            props[4] = prop4;
            let mut tok = [0u8; NUM_MA_PREDS];
            let mut nbits = [0u8; NUM_MA_PREDS];
            for p in 0..NUM_MA_PREDS {
                // The good predictor has residual 0; all others residual 200.
                let res = if p == good_pred { 0 } else { 200 };
                let (t, nb, _) = uint_encode(pack_signed(res));
                tok[p] = t as u8;
                nbits[p] = nb as u8;
            }
            s.push(props, tok, nbits);
        };
        for i in 0..8192 {
            push(i % 8, 1); // low-activity population: predictor 1
            push(64 + i % 8, 5); // high-activity population: predictor 5
        }
        let mut serial_scratch = CoderScratch::default();
        let serial_pool = ThreadPool::new(1);
        let serial = learn_ma_tree(
            &s,
            MaLearnParams {
                alphabet: 64,
                max_leaves: 8,
                split_cost_bits: 10.0,
                min_node: 64,
                allow_wp: true,
            },
            &serial_pool,
            &mut serial_scratch,
        );
        let mut parallel_scratch = CoderScratch::default();
        let parallel_pool = ThreadPool::new(4);
        let tree = learn_ma_tree(
            &s,
            MaLearnParams {
                alphabet: 64,
                max_leaves: 8,
                split_cost_bits: 10.0,
                min_node: 64,
                allow_wp: true,
            },
            &parallel_pool,
            &mut parallel_scratch,
        );
        assert_eq!(serial.nodes, tree.nodes);
        assert_eq!(serial.est_bits, tree.est_bits);
        assert_eq!(serial.flat_bits, tree.flat_bits);
        assert!(tree.est_bits < tree.flat_bits);
        for i in 0..s.len() {
            let (_, pred) = tree.lookup(&s.props[i]);
            let expected = if s.props[i][4] < 32 { 1 } else { 5 };
            assert_eq!(pred, expected, "sample {i} routed to wrong predictor");
        }
    }

    /// Uniform samples must not grow a tree beyond a single leaf.
    #[test]
    fn uniform_samples_stay_single_leaf() {
        let mut s = MaSamples::new();
        for _ in 0..2048 {
            let props = [0i32; NUM_MA_PROPS];
            let mut tok = [0u8; NUM_MA_PREDS];
            let mut nbits = [0u8; NUM_MA_PREDS];
            for p in 0..NUM_MA_PREDS {
                let (t, nb, _) = uint_encode(pack_signed(3));
                tok[p] = t as u8;
                nbits[p] = nb as u8;
            }
            s.push(props, tok, nbits);
        }
        let tree = learn_ma_tree(
            &s,
            MaLearnParams {
                alphabet: 64,
                max_leaves: 8,
                split_cost_bits: 10.0,
                min_node: 64,
                allow_wp: true,
            },
            &ThreadPool::new(1),
            &mut CoderScratch::default(),
        );
        assert_eq!(tree.nodes.len(), 1);
        assert!(matches!(tree.nodes[0], MaNode::Leaf { .. }));
    }

    #[test]
    fn indexed_learning_matches_materialized_selection() {
        let mut samples = MaSamples::new();
        for i in 0..4096 {
            let mut props = [0i32; NUM_MA_PROPS];
            props[0] = i % 4;
            props[4] = i % 97;
            props[7] = (i * 13) % 211 - 105;
            let mut tok = [0u8; NUM_MA_PREDS];
            let mut nbits = [0u8; NUM_MA_PREDS];
            for pred in 0..NUM_MA_PREDS {
                let residual = props[4] - pred as i32 * 3;
                let (token, extra, _) = uint_encode(pack_signed(residual));
                tok[pred] = token as u8;
                nbits[pred] = extra as u8;
            }
            samples.push(props, tok, nbits);
        }

        let indices = samples.evenly_sampled_indices(1024);
        let mut copied = MaSamples::with_capacity(indices.len());
        for &index in &indices {
            let index = index as usize;
            copied.push(
                samples.props[index],
                samples.tok[index],
                samples.nbits[index],
            );
        }
        let params = || MaLearnParams {
            alphabet: 64,
            max_leaves: 16,
            split_cost_bits: 10.0,
            min_node: 64,
            allow_wp: true,
        };
        let pool = ThreadPool::new(1);
        let indexed = learn_ma_tree_indexed(
            &samples,
            indices,
            params(),
            &pool,
            &mut CoderScratch::default(),
        );
        let materialized = learn_ma_tree(&copied, params(), &pool, &mut CoderScratch::default());

        assert_eq!(indexed.nodes, materialized.nodes);
        assert_eq!(indexed.est_bits, materialized.est_bits);
        assert_eq!(indexed.flat_bits, materialized.flat_bits);
    }
}
