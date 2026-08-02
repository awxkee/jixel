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

/// Property vector length (libjxl ids 0..=15). Index 1 (stream id) is always 0.
pub(crate) const NUM_MA_PROPS: usize = 16;
/// Decoder predictors 0..=13 (libjxl `Predictor` enum order).
pub(crate) const NUM_MA_PREDS: usize = 14;
/// Properties the learner may split on (skips the constant stream id).
const SPLIT_PROPS: [u8; 15] = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Split candidates examined per property per node.
const MAX_CANDIDATES: usize = 48;
/// Strided value subset used to derive candidate thresholds.
const MAX_QUANTILE_PROBE: usize = 1024;
/// Hard depth guard (decoder-side property lookups stay cheap).
const MAX_DEPTH: usize = 26;

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
    pub(crate) fn new() -> Self {
        Self {
            props: Vec::new(),
            tok: Vec::new(),
            nbits: Vec::new(),
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
}

pub(crate) struct MaLearnParams {
    pub(crate) alphabet: usize,
    pub(crate) max_leaves: usize,
    pub(crate) split_cost_bits: f32,
    pub(crate) min_node: usize,
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
    bin_hist: Vec<u32>, // MAX_CANDIDATES+1 bins x alphabet (search predictor)
    bin_nbits: [u64; MAX_CANDIDATES + 1],
    left_hist: Vec<u32>, // alphabet
    probe: Vec<i32>,
    cands: Vec<i32>,
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
        let mut best = f32::INFINITY;
        let mut best_pred = 0usize;
        for p in 0..NUM_MA_PREDS {
            let bits = hist_entropy_bits(&self.all_hist[p * alpha..(p + 1) * alpha], total)
                + self.all_nbits[p] as f32;
            if bits < best {
                best = bits;
                best_pred = p;
            }
        }
        (best, best_pred)
    }

    fn pick_candidates(&mut self, idx: &[u32], prop: usize) -> usize {
        self.probe.clear();
        let stride = (idx.len() / MAX_QUANTILE_PROBE).max(1);
        let mut i = 0;
        while i < idx.len() {
            self.probe.push(self.s.props[idx[i] as usize][prop]);
            i += stride;
        }
        self.probe.sort_unstable();
        self.probe.dedup();
        if self.probe.len() < 2 {
            return 0;
        }
        // Thresholds at even quantiles of the distinct probe values; the max
        // value is excluded so the `gt` side is never structurally empty.
        self.cands.clear();
        let distinct = self.probe.len() - 1;
        let n = distinct.min(MAX_CANDIDATES);
        for k in 0..n {
            let pos = k * distinct / n;
            let v = self.probe[pos];
            if self.cands.last() != Some(&v) {
                self.cands.push(v);
            }
        }
        self.cands.len()
    }

    /// Best split of `idx` on `prop`, scoring with `search_pred` fixed on both
    /// sides. Returns (threshold, combined bits) or None.
    fn best_split_on_prop(
        &mut self,
        idx: &[u32],
        prop: usize,
        search_pred: usize,
    ) -> Option<(i32, f32)> {
        let ncand = self.pick_candidates(idx, prop);
        if ncand == 0 {
            return None;
        }
        let alpha = self.p.alphabet;
        self.bin_hist[..(ncand + 1) * alpha].fill(0);
        self.bin_nbits[..ncand + 1].fill(0);
        let mut total_nbits = 0u64;
        let mut bin_count = [0u32; MAX_CANDIDATES + 1];
        for &i in idx {
            let v = self.s.props[i as usize][prop];
            // bin = number of candidates < v; v <= cands[j] iff bin <= j.
            let bin = self.cands.partition_point(|&c| c < v);
            let tok = self.s.tok[i as usize][search_pred] as usize;
            let nb = self.s.nbits[i as usize][search_pred] as u64;
            self.bin_hist[bin * alpha + tok] += 1;
            self.bin_nbits[bin] += nb;
            bin_count[bin] += 1;
            total_nbits += nb;
        }
        let total_hist = self.all_hist[search_pred * alpha..(search_pred + 1) * alpha].to_vec();
        let total_count = idx.len() as u32;

        self.left_hist[..alpha].fill(0);
        let mut left_nbits = 0u64;
        let mut left_count = 0u32;
        let mut best: Option<(i32, f32)> = None;
        for j in 0..ncand {
            for t in 0..alpha {
                self.left_hist[t] += self.bin_hist[j * alpha + t];
            }
            left_nbits += self.bin_nbits[j];
            left_count += bin_count[j];
            if left_count == 0 || left_count == total_count {
                continue;
            }
            let cost_l =
                hist_entropy_bits(&self.left_hist[..alpha], left_count) + left_nbits as f32;
            let mut cost_r_hist_bits = 0.0f32;
            let right_count = total_count - left_count;
            let log_total = dirty_log2f(right_count as f32);
            for t in 0..alpha {
                let c = total_hist[t] - self.left_hist[t];
                if c != 0 {
                    cost_r_hist_bits += c as f32 * (log_total - dirty_log2f(c as f32));
                }
            }
            let cost_r = cost_r_hist_bits + (total_nbits - left_nbits) as f32;
            let cost = cost_l + cost_r;
            if best.is_none() || cost < best.unwrap().1 {
                best = Some((self.cands[j], cost));
            }
        }
        best
    }

    fn evaluate(
        &mut self,
        node_id: u32,
        start: usize,
        end: usize,
        depth: u16,
        idx: &[u32],
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
            for &prop in &SPLIT_PROPS {
                if let Some((t, bits)) = self.best_split_on_prop(range, prop as usize, base_pred)
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

/// Learn a context tree from `samples`. Never fails; a degenerate sample set
/// yields a single-leaf tree.
pub(crate) fn learn_ma_tree(samples: &MaSamples, params: MaLearnParams) -> LearnedTree {
    let alpha = params.alphabet;
    let n = samples.len();
    let mut learner = Learner {
        s: samples,
        nodes: Vec::new(),
        leaves: 0,
        est_bits: 0.0,
        all_hist: vec![0u32; NUM_MA_PREDS * alpha],
        all_nbits: [0; NUM_MA_PREDS],
        bin_hist: vec![0u32; (MAX_CANDIDATES + 1) * alpha],
        bin_nbits: [0; MAX_CANDIDATES + 1],
        left_hist: vec![0u32; alpha],
        probe: Vec::with_capacity(MAX_QUANTILE_PROBE + 1),
        cands: Vec::with_capacity(MAX_CANDIDATES),
        p: params,
    };

    // Flat baseline: channel-only split, best single predictor per channel.
    let mut flat_bits = 0.0f64;
    {
        let mut chan_idx: [Vec<u32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for i in 0..n {
            let c = samples.props[i][0].clamp(0, 3) as usize;
            chan_idx[c].push(i as u32);
        }
        for c in chan_idx.iter() {
            if !c.is_empty() {
                let (bits, _) = learner.node_cost_all(c);
                flat_bits += bits as f64;
            }
        }
    }

    // Best-first growth: always apply the highest-gain split available
    // anywhere in the tree, so the leaf budget goes where it pays most.
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let mut heap: std::collections::BinaryHeap<Pending> = std::collections::BinaryHeap::new();
    learner.nodes.push(MaNode::Leaf { pred: 0 });
    learner.leaves = 1;
    let root = learner.evaluate(0, 0, n, 0, &idx);
    learner.est_bits = root.base_bits as f64;
    if root.split.is_some() {
        heap.push(root);
    }

    while learner.leaves < learner.p.max_leaves as u32 {
        let Some(p) = heap.pop() else { break };
        let (prop, threshold) = p.split.expect("only splittable nodes are queued");
        // Partition idx[p.start..p.end]: gt side first, then le.
        let mut lo = p.start;
        let mut hi = p.end;
        while lo < hi {
            if samples.props[idx[lo] as usize][prop as usize] > threshold {
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

        let gt = learner.evaluate(gt_id, p.start, lo, p.depth + 1, &idx);
        let le = learner.evaluate(le_id, lo, p.end, p.depth + 1, &idx);
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
        for i in 0..4096 {
            push(i % 8, 1); // low-activity population: predictor 1
            push(64 + i % 8, 5); // high-activity population: predictor 5
        }
        let tree = learn_ma_tree(
            &s,
            MaLearnParams {
                alphabet: 64,
                max_leaves: 8,
                split_cost_bits: 10.0,
                min_node: 64,
            },
        );
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
            },
        );
        assert_eq!(tree.nodes.len(), 1);
        assert!(matches!(tree.nodes[0], MaNode::Leaf { .. }));
    }
}
