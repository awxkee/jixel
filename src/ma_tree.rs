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
use crate::util::heap_array;
use std::sync::OnceLock;

/// Property vector length (libjxl ids 0..=15). Index 1 is the decoder's
/// stream id (0 for the global stream, a fixed per-group id for AC groups),
/// so a global tree can specialize per group the way libjxl's does.
/// Reference (previously coded, same-size) channels whose values feed the
/// decoder's extra properties: four per channel from id 16, most recent
/// channel first (|v|, v, |v − g|, v − g with g the clamped gradient).
pub(crate) const MA_REF_CHANNELS: usize = 2;
pub(crate) const NUM_MA_PROPS: usize = 16 + 4 * MA_REF_CHANNELS;
/// Decoder predictors 0..=13 (libjxl `Predictor` enum order).
pub(crate) const NUM_MA_PREDS: usize = 14;
/// Properties the learner may split on (the WP-error property last).
static SPLIT_PROPS: [u8; NUM_MA_PROPS] = {
    // 0..=14, then the reference properties, with the WP property last so
    // `split_props(false)` can drop it.
    let mut props = [0u8; NUM_MA_PROPS];
    let mut i = 0;
    while i < 15 {
        props[i] = i as u8;
        i += 1;
    }
    let mut r = 16;
    while r < NUM_MA_PROPS {
        props[r - 1] = r as u8;
        r += 1;
    }
    props[NUM_MA_PROPS - 1] = 15;
    props
};

/// Split candidates, minus the WP-error property (last) when WP is excluded.
fn split_props(allow_wp: bool) -> &'static [u8] {
    if !allow_wp {
        &SPLIT_PROPS[..SPLIT_PROPS.len() - 1]
    } else {
        &SPLIT_PROPS
    }
}

/// Weighted predictor id in the decoder predictor enumeration.
pub(crate) const PRED_WEIGHTED: usize = 6;

/// Split candidates examined per property per node. Sixteen quantiles retain
/// nearly all the useful threshold resolution while bounding the repeated
/// histogram work performed for every property of every grown leaf.
const MAX_CANDIDATES: usize = 32;
/// Strided value subset used to derive candidate thresholds.
const MAX_QUANTILE_PROBE: usize = 1024;
/// Narrow probe ranges use one presence bit per value instead of sorting.
const PROBE_BITMAP_BITS: usize = 4096;
/// Hard depth guard (decoder-side property lookups stay cheap).
const MAX_DEPTH: usize = 26;
/// Nodes with at least this many rows no longer sit in cache: they bin every
/// property in one pass over the rows, in concurrent row chunks of at least
/// `PARALLEL_SCAN_CHUNK_SAMPLES` (one multi-property histogram per chunk,
/// summed afterwards), instead of streaming the rows once per property.
const PARALLEL_SCAN_MIN_SAMPLES: usize = 1 << 17;
const PARALLEL_SCAN_CHUNK_SAMPLES: usize = 8192;
/// Smaller nodes score their properties concurrently, one task per
/// property, from this many rows on; below it the tasks cost more than the
/// scans.
const PARALLEL_PLAN_MIN_SAMPLES: usize = 1024;
/// Both children of a split are scored concurrently when the smaller one has
/// at least this many samples (deep, small nodes are the serial tail).
const PARALLEL_CHILD_MIN_SAMPLES: usize = 64;

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
}

fn select_rows<T: Copy>(rows: &[T], indices: &[u32]) -> Vec<T> {
    indices.iter().map(|&i| rows[i as usize]).collect()
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
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, props: [i32; NUM_MA_PROPS], tok: [u8; NUM_MA_PREDS]) {
        self.props.push(props);
        self.tok.push(tok);
    }

    pub(crate) fn len(&self) -> usize {
        self.props.len()
    }

    /// Materialize a learner's current order once. Copy each field in one
    /// pass so allocation checks stay outside the sample loop.
    fn select(&self, indices: &[u32]) -> Self {
        Self {
            props: select_rows(&self.props, indices),
            tok: select_rows(&self.tok, indices),
        }
    }

    /// Apply a full permutation in place. The deep learner's stable grouping
    /// supplies each source row exactly once; cycle rotation preserves that
    /// exact order without allocating another full property array.
    fn into_permuted(mut self, mut indices: Vec<u32>) -> Self {
        assert_eq!(indices.len(), self.len());
        for start in 0..indices.len() {
            if indices[start] as usize == start {
                continue;
            }
            let props = self.props[start];
            let tok = self.tok[start];
            let mut dst = start;
            loop {
                let src = indices[dst] as usize;
                indices[dst] = dst as u32;
                if src == start {
                    self.props[dst] = props;
                    self.tok[dst] = tok;
                    break;
                }
                self.props[dst] = self.props[src];
                self.tok[dst] = self.tok[src];
                dst = src;
            }
        }
        self
    }

    /// Deterministic evenly spaced selection for the coarse learning stage.
    /// Retain ordinals here so the learner materializes only its selected rows.
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
    /// Bitmask of decoder predictor ids leaves may use (bit p = predictor p).
    /// The lossy-modular path restricts leaves to scale-equivariant
    /// predictors; everywhere else this is `u16::MAX`.
    pub(crate) allowed_preds: u16,
    /// How many of a node's cheapest predictors each side of a candidate
    /// split may choose from independently (libjxl's FindBestSplit scores
    /// every predictor per side). `1` scores both sides with the node's best
    /// predictor only.
    pub(crate) side_preds: usize,
    /// Split candidates evaluated per property per node (<= MAX_CANDIDATES).
    /// Photos want the full 32; near-deterministic palette index images
    /// overfit at more than 16.
    pub(crate) max_candidates: usize,
}

/// Reusable scratch for scoring one property of one MA-tree node. One lives in
/// each encoder worker's `CoderScratch`, allowing independent properties to be
/// evaluated concurrently without per-node allocation.
pub(crate) struct MaPropertyScratch {
    probe: Vec<i32>,
    probe_bitmap: Box<[u64; PROBE_BITMAP_BITS / 64]>,
    cands: Vec<i32>,
    /// Candidate thresholds of every property of the node being scored.
    plans: Vec<SplitPlan>,
    /// Value -> bin tables of the plans with a narrow candidate span.
    luts: Vec<u8>,
    /// Every plan's `[side pred][bin][symbol]` histogram, back to back.
    multi_hist: Vec<u32>,
    /// Every plan's rows per bin.
    multi_bin_count: Vec<u32>,
    /// Per side predictor: raw bits of every candidate bin.
    bin_nbits: Vec<u64>,
    /// Per predictor: running histogram and raw bits of the `le` side.
    left_hist: Vec<u32>,
    left_nbits: Vec<u64>,
    /// Per predictor: token histogram and raw bits of the whole node.
    node_hist: Vec<u32>,
    node_nbits: Vec<u64>,
    /// Costs already computed when choosing the node's best predictor.
    node_costs: [f32; NUM_MA_PREDS],
}

impl Default for MaPropertyScratch {
    fn default() -> Self {
        Self {
            probe: Vec::with_capacity(MAX_QUANTILE_PROBE + 1),
            probe_bitmap: heap_array(0),
            cands: Vec::with_capacity(MAX_CANDIDATES),
            plans: Vec::with_capacity(NUM_MA_PROPS),
            luts: Vec::new(),
            multi_hist: Vec::new(),
            multi_bin_count: Vec::new(),
            bin_nbits: vec![0; NUM_MA_PREDS * (MAX_CANDIDATES + 1)],
            left_hist: Vec::new(),
            left_nbits: vec![0; NUM_MA_PREDS],
            node_hist: Vec::new(),
            node_nbits: vec![0; NUM_MA_PREDS],
            node_costs: [f32::INFINITY; NUM_MA_PREDS],
        }
    }
}

impl MaPropertyScratch {
    fn prepare(&mut self, alphabet: usize) {
        if self.left_hist.len() < NUM_MA_PREDS * alphabet {
            self.left_hist.resize(NUM_MA_PREDS * alphabet, 0);
        }
        if self.node_hist.len() < NUM_MA_PREDS * alphabet {
            self.node_hist.resize(NUM_MA_PREDS * alphabet, 0);
        }
    }
}

struct Learner<'a> {
    s: &'a mut MaSamples,
    p: MaLearnParams,
    nodes: Vec<MaNode>,
    leaves: u32,
    est_bits: f64,
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

// Split scoring repeatedly takes logarithms of small integer populations.
// Cache the same approximation in 16 KiB; larger counts retain the original
// calculation, and each histogram retains its original summation order.
fn count_log2_table() -> &'static [f32; 4096] {
    static LOGS: OnceLock<Box<[f32; 4096]>> = OnceLock::new();
    LOGS.get_or_init(|| Box::new(core::array::from_fn(|i| dirty_log2f(i as f32))))
}

#[inline]
fn count_log2(count: u32, logs: &[f32; 4096]) -> f32 {
    logs.get(count as usize)
        .copied()
        .unwrap_or_else(|| dirty_log2f(count as f32))
}

fn sort_unique_probe(probe: &mut Vec<i32>, bitmap: &mut [u64; PROBE_BITMAP_BITS / 64]) {
    if probe.len() < 32 {
        probe.sort_unstable();
        probe.dedup();
        return;
    }
    let (min, max) = probe.iter().fold((i32::MAX, i32::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });
    if min == max {
        probe.truncate(1);
        return;
    }
    // Widen before subtracting: properties can span the entire i32 range.
    let span = max as i64 - min as i64;
    if span >= PROBE_BITMAP_BITS as i64 {
        probe.sort_unstable();
        probe.dedup();
        return;
    }
    let words = span as usize / 64 + 1;
    bitmap[..words].fill(0);
    for &v in probe.iter() {
        let bit = (v - min) as usize;
        bitmap[bit / 64] |= 1 << (bit % 64);
    }
    // Enumerating presence bits preserves quantiles over distinct values.
    // There are at most probe.len() set bits, so the existing buffer suffices.
    let mut output = probe.iter_mut();
    let mut len = 0;
    for (word, &bits) in bitmap[..words].iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            let offset = word * 64 + bits.trailing_zeros() as usize;
            *output.next().unwrap() = min + offset as i32;
            len += 1;
            bits &= bits - 1;
        }
    }
    probe.truncate(len);
}

fn pick_candidates(
    properties: &[[i32; NUM_MA_PROPS]],
    prop: usize,
    max_candidates: usize,
    scratch: &mut MaPropertyScratch,
) -> usize {
    scratch.probe.clear();
    let mut stride = (properties.len() / MAX_QUANTILE_PROBE).max(1);
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    for props in properties.iter().step_by(stride) {
        scratch.probe.push(props[prop]);
    }
    sort_unique_probe(&mut scratch.probe, &mut scratch.probe_bitmap);
    if scratch.probe.len() < 2 {
        return 0;
    }
    // Thresholds at even quantiles of the distinct probe values; the max value
    // is excluded so the `gt` side is never structurally empty.
    scratch.cands.clear();
    let distinct = scratch.probe.len() - 1;
    let n = distinct.min(max_candidates.min(MAX_CANDIDATES));
    for k in 0..n {
        let pos = k * distinct / n;
        let v = scratch.probe[pos];
        if scratch.cands.last() != Some(&v) {
            scratch.cands.push(v);
        }
    }
    scratch.cands.len()
}

/// Best split of contiguous rows on one property: candidate thresholds at
/// value quantiles, every planned property binned in one pass over the rows,
/// then each side independently picks its cheapest predictor among `preds`
/// (libjxl FindBestSplit semantics). The children are re-scored with every
/// predictor afterwards, so this only decides *where* to split, but that is
/// exactly what a node-locked predictor gets wrong on mixed content.
/// Candidate spans up to this wide get a direct value -> bin table instead
/// of a per-sample binary search.
const LUT_MAX_SPAN: usize = 4096;

/// Candidate thresholds of one property for one node, with the location of
/// its bins in the node's shared multi-property histogram.
#[derive(Clone, Copy)]
struct SplitPlan {
    prop: u8,
    ncand: u8,
    use_lut: bool,
    lo_v: i32,
    hi_v: i32,
    /// Start of this property's value -> bin table in the shared LUT buffer.
    lut_start: u32,
    /// Start of this property's `[side pred][bin][symbol - lo]` block.
    hist_start: u32,
    /// `bins * width`: one side predictor's span inside the block.
    pred_stride: u32,
    cands: [i32; MAX_CANDIDATES],
}

impl SplitPlan {
    /// bin = number of candidates < v; v <= cands[j] iff bin <= j.
    #[inline]
    fn bin_for(&self, v: i32, luts: &[u8]) -> usize {
        if self.use_lut {
            if v <= self.lo_v {
                0
            } else if v > self.hi_v {
                self.ncand as usize
            } else {
                luts[self.lut_start as usize + (v - self.lo_v) as usize] as usize
            }
        } else {
            self.cands[..self.ncand as usize].partition_point(|&c| c < v)
        }
    }
}

/// Symbol window of a node: the side predictors' tokens all fall in
/// `lo..lo + width`; symbols outside it are zero on both sides of every
/// split and contribute nothing to either cost, so the binned histograms
/// only store the window. `syms[i]` lists the symbols predictor `preds[i]`
/// actually uses in the node, ascending.
struct SymbolWindow {
    lo: usize,
    width: usize,
    syms: Vec<Vec<u16>>,
}

impl SymbolWindow {
    fn new(preds: &[usize], alphabet: usize, all_hist: &[u32]) -> Self {
        let mut lo = alphabet;
        let mut hi = 0usize;
        let syms: Vec<Vec<u16>> = preds
            .iter()
            .map(|&p| {
                let h = &all_hist[p * alphabet..(p + 1) * alphabet];
                let used: Vec<u16> = (0..alphabet)
                    .filter(|&s| h[s] != 0)
                    .map(|s| s as u16)
                    .collect();
                if let (Some(&first), Some(&last)) = (used.first(), used.last()) {
                    lo = lo.min(first as usize);
                    hi = hi.max(last as usize + 1);
                }
                used
            })
            .collect();
        if hi <= lo {
            (lo, hi) = (0, 1);
        }
        Self {
            lo,
            width: hi - lo,
            syms,
        }
    }
}

/// Candidate thresholds of `prop` over the node, or `None` when the property
/// has fewer than two distinct probed values. A narrow span gets its
/// value -> bin table appended to `luts`; `hist_start` is where the
/// property's block will live and `width` the node's symbol window.
#[allow(clippy::too_many_arguments)]
fn plan_split(
    properties: &[[i32; NUM_MA_PROPS]],
    prop: usize,
    max_candidates: usize,
    width: usize,
    hist_start: u32,
    luts: &mut Vec<u8>,
    scratch: &mut MaPropertyScratch,
) -> Option<SplitPlan> {
    let ncand = pick_candidates(properties, prop, max_candidates, scratch);
    if ncand == 0 {
        return None;
    }
    let cands_src = &scratch.cands;
    let lo_v = cands_src[0];
    let hi_v = cands_src[ncand - 1];
    let span = (hi_v as i64 - lo_v as i64) as usize;
    let use_lut = span <= LUT_MAX_SPAN && span <= 4 * properties.len() + 64;
    let lut_start = luts.len() as u32;
    if use_lut {
        let mut k = 0usize;
        luts.extend((0..=span).map(|off| {
            let v = lo_v + off as i32;
            while k < ncand && cands_src[k] < v {
                k += 1;
            }
            k as u8
        }));
    }
    let mut cands = [0i32; MAX_CANDIDATES];
    cands[..ncand].copy_from_slice(cands_src);
    Some(SplitPlan {
        prop: prop as u8,
        ncand: ncand as u8,
        use_lut,
        lo_v,
        hi_v,
        lut_start,
        hist_start,
        pred_stride: ((ncand + 1) * width) as u32,
        cands,
    })
}

/// One pass over the node's rows, binning every planned property at once.
/// Per plan the block is `[side pred i][bin][symbol - lo]` (i indexing
/// `preds`) and `bin_count[k * (MAX_CANDIDATES + 1) + bin]` counts the rows
/// of plan `k`. Pure integer counting, so chunks of rows may be scanned
/// separately and summed.
#[allow(clippy::too_many_arguments)]
fn scan_split_plans(
    properties: &[[i32; NUM_MA_PROPS]],
    tokens: &[[u8; NUM_MA_PREDS]],
    plans: &[SplitPlan],
    luts: &[u8],
    preds: &[usize],
    window: (usize, usize),
    hist: &mut [u32],
    bin_count: &mut [u32],
) {
    debug_assert_eq!(properties.len(), tokens.len());
    let (lo, width) = window;
    if let &[p0, p1, p2, p3] = preds {
        // The encoder normally scores four side predictors: bind their
        // tokens once per row and keep the predictor loop out of the plan
        // loop.
        for (props, toks) in properties.iter().zip(tokens) {
            let t0 = toks[p0] as usize - lo;
            let t1 = toks[p1] as usize - lo;
            let t2 = toks[p2] as usize - lo;
            let t3 = toks[p3] as usize - lo;
            for (counts, plan) in bin_count
                .as_chunks_mut::<{ MAX_CANDIDATES + 1 }>()
                .0
                .iter_mut()
                .zip(plans)
            {
                let bin = plan.bin_for(props[plan.prop as usize], luts);
                counts[bin] += 1;
                let stride = plan.pred_stride as usize;
                let base = plan.hist_start as usize + bin * width;
                hist[base + t0] += 1;
                hist[base + stride + t1] += 1;
                hist[base + 2 * stride + t2] += 1;
                hist[base + 3 * stride + t3] += 1;
            }
        }
    } else {
        for (props, toks) in properties.iter().zip(tokens) {
            for (counts, plan) in bin_count
                .as_chunks_mut::<{ MAX_CANDIDATES + 1 }>()
                .0
                .iter_mut()
                .zip(plans)
            {
                let bin = plan.bin_for(props[plan.prop as usize], luts);
                counts[bin] += 1;
                let stride = plan.pred_stride as usize;
                let base = plan.hist_start as usize + bin * width;
                for (i, &p) in preds.iter().enumerate() {
                    hist[base + i * stride + toks[p] as usize - lo] += 1;
                }
            }
        }
    }
}

/// Samples use uint_encode's fixed (4, 2, 0) config, so a symbol determines
/// its raw-bit count: zero below 16, otherwise (symbol / 4) - 2. Summing
/// over the symbols a predictor uses in the node avoids storing and
/// accumulating it for every sample.
#[inline]
fn raw_bits_windowed(bin: &[u32], lo: usize, syms: &[u16]) -> u64 {
    syms.iter()
        .map(|&s| s as usize)
        .filter(|&s| s >= 16)
        .map(|s| bin[s - lo] as u64 * ((s >> 2) - 2) as u64)
        .sum()
}

/// Best threshold of one planned property from its binned block. `hist` is
/// the plan's `[side pred][bin][symbol - lo]` block and `bin_count` its row
/// counts per bin; `all_hist` / `all_nbits` are the node's per-predictor
/// token histograms and raw-bit totals.
#[allow(clippy::too_many_arguments)]
fn score_split_plan(
    plan: &SplitPlan,
    preds: &[usize],
    alphabet: usize,
    window: &SymbolWindow,
    hist: &[u32],
    bin_count: &[u32],
    all_hist: &[u32],
    all_nbits: &[u64; NUM_MA_PREDS],
    total_count: u32,
    scratch: &mut MaPropertyScratch,
) -> Option<(i32, f32)> {
    let ncand = plan.ncand as usize;
    let bins = ncand + 1;
    let (lo, width) = (window.lo, window.width);
    let stride = plan.pred_stride as usize;
    for (i, syms) in window.syms.iter().enumerate() {
        let pred_hist = &hist[i * stride..(i + 1) * stride];
        for (nbits, bin) in scratch.bin_nbits[i * bins..(i + 1) * bins]
            .iter_mut()
            .zip(pred_hist.chunks_exact(width))
        {
            *nbits = raw_bits_windowed(bin, lo, syms);
        }
    }
    let logs = count_log2_table();
    for i in 0..preds.len() {
        scratch.left_hist[i * alphabet..(i + 1) * alphabet].fill(0);
    }
    scratch.left_nbits[..preds.len()].fill(0);
    let mut left_count = 0u32;
    let mut best: Option<(i32, f32)> = None;
    for (j, (&candidate, &count)) in plan.cands[..ncand]
        .iter()
        .zip(&bin_count[..ncand])
        .enumerate()
    {
        left_count += count;
        let mut best_l = f32::INFINITY;
        let mut best_r = f32::INFINITY;
        for (i, (&p, syms)) in preds.iter().zip(&window.syms).enumerate() {
            let left = &mut scratch.left_hist[i * alphabet..(i + 1) * alphabet];
            let bin = &hist[i * stride + j * width..i * stride + (j + 1) * width];
            for &s in syms {
                left[s as usize] += bin[s as usize - lo];
            }
            scratch.left_nbits[i] += scratch.bin_nbits[i * bins + j];
            if left_count == 0 || left_count == total_count {
                continue;
            }
            let right_count = total_count - left_count;
            let log_left = count_log2(left_count, logs);
            let log_right = count_log2(right_count, logs);
            let mut cost_l = 0.0f32;
            let mut cost_r = 0.0f32;
            let node = &all_hist[p * alphabet..(p + 1) * alphabet];
            for &s in syms {
                let s = s as usize;
                let l = left[s];
                if l != 0 {
                    cost_l += l as f32 * (log_left - count_log2(l, logs));
                }
                let c = node[s] - l;
                if c != 0 {
                    cost_r += c as f32 * (log_right - count_log2(c, logs));
                }
            }
            cost_l += scratch.left_nbits[i] as f32;
            cost_r += (all_nbits[p] - scratch.left_nbits[i]) as f32;
            best_l = best_l.min(cost_l);
            best_r = best_r.min(cost_r);
        }
        if left_count == 0 || left_count == total_count {
            continue;
        }
        let cost = best_l + best_r;
        if best.is_none() || cost < best.unwrap().1 {
            best = Some((candidate, cost));
        }
    }
    best
}

/// Best split of contiguous rows on a single property (the multi-property
/// scan restricted to one plan); `all_hist` is the node's per-predictor
/// token histogram (NUM_MA_PREDS x alphabet) and `all_nbits` the
/// per-predictor raw-bit totals.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn best_split_on_prop(
    properties: &[[i32; NUM_MA_PROPS]],
    tokens: &[[u8; NUM_MA_PREDS]],
    prop: usize,
    preds: &[usize],
    alphabet: usize,
    max_candidates: usize,
    all_hist: &[u32],
    all_nbits: &[u64; NUM_MA_PREDS],
    scratch: &mut MaPropertyScratch,
) -> Option<(i32, f32)> {
    assert_eq!(properties.len(), tokens.len());
    scratch.prepare(alphabet);
    let window = SymbolWindow::new(preds, alphabet, all_hist);
    let mut luts = Vec::new();
    let plan = plan_split(
        properties,
        prop,
        max_candidates,
        window.width,
        0,
        &mut luts,
        scratch,
    )?;
    let mut hist = vec![0u32; preds.len() * plan.pred_stride as usize];
    let mut bin_count = vec![0u32; MAX_CANDIDATES + 1];
    scan_split_plans(
        properties,
        tokens,
        std::slice::from_ref(&plan),
        &luts,
        preds,
        (window.lo, window.width),
        &mut hist,
        &mut bin_count,
    );
    score_split_plan(
        &plan,
        preds,
        alphabet,
        &window,
        &hist,
        &bin_count,
        all_hist,
        all_nbits,
        properties.len() as u32,
        scratch,
    )
}

/// Samples use uint_encode's fixed (4, 2, 0) config, so a symbol determines
/// its raw-bit count: zero below 16, otherwise (symbol / 4) - 2. Summing
/// the integer histogram avoids storing and accumulating it for every sample.
#[inline]
fn raw_bits_from_hist(hist: &[u32]) -> u64 {
    hist.iter()
        .enumerate()
        .skip(16)
        .map(|(token, &count)| count as u64 * ((token >> 2) - 2) as u64)
        .sum()
}

/// Token-histogram cost of sample rows under every predictor; returns (best bits
/// incl. raw bits, best predictor) and leaves the per-predictor histograms
/// and costs in `scratch.node_hist` / `scratch.node_nbits` / `scratch.node_costs`.
fn node_cost_all<'a>(
    tokens: impl ExactSizeIterator<Item = &'a [u8; NUM_MA_PREDS]>,
    params: &MaLearnParams,
    scratch: &mut MaPropertyScratch,
) -> (f32, usize) {
    let alpha = params.alphabet;
    scratch.prepare(alpha);
    let MaPropertyScratch {
        node_hist,
        node_nbits,
        node_costs,
        ..
    } = scratch;
    let hist = &mut node_hist[..NUM_MA_PREDS * alpha];
    hist.fill(0);
    node_nbits.fill(0);
    node_costs.fill(f32::INFINITY);
    let total = tokens.len() as u32;
    for toks in tokens {
        for (pred_hist, &token) in hist.chunks_exact_mut(alpha).zip(toks) {
            pred_hist[token as usize] += 1;
        }
    }
    for (nbits, pred_hist) in node_nbits.iter_mut().zip(hist.chunks_exact(alpha)) {
        *nbits = raw_bits_from_hist(pred_hist);
    }
    let skip_wp = !params.allow_wp;
    let mut best = f32::INFINITY;
    let mut best_pred = 0usize;
    for (p, ((cost, &nbits), pred_hist)) in node_costs
        .iter_mut()
        .zip(node_nbits.iter())
        .zip(hist.chunks_exact(alpha))
        .enumerate()
    {
        if skip_wp && p == PRED_WEIGHTED {
            continue;
        }
        if params.allowed_preds & (1 << p) == 0 {
            continue;
        }
        let bits = hist_entropy_bits(pred_hist, total) + nbits as f32;
        *cost = bits;
        if bits < best {
            best = bits;
            best_pred = p;
        }
    }
    (best, best_pred)
}

impl Learner<'_> {
    /// Score one leaf: its best predictor and best split. Deterministic per
    /// node, so any number of leaves may be evaluated concurrently.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        node_id: u32,
        start: usize,
        end: usize,
        depth: u16,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Pending {
        self.evaluate_rows(
            node_id,
            start,
            end,
            &self.s.props[start..end],
            &self.s.tok[start..end],
            depth,
            pool,
            scratch,
        )
    }

    /// `evaluate` over rows supplied by the caller: the same rows (in the same
    /// order) that `start..end` will hold, possibly a copy made before the
    /// learner's own array was partitioned.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_rows(
        &self,
        node_id: u32,
        start: usize,
        end: usize,
        properties: &[[i32; NUM_MA_PROPS]],
        tokens: &[[u8; NUM_MA_PREDS]],
        depth: u16,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Pending {
        debug_assert_eq!(properties.len(), end - start);
        let (base_bits, base_pred) =
            node_cost_all(tokens.iter(), &self.p, &mut scratch.ma_property);
        let mut split = None;
        let mut gain = 0.0f32;
        // A split gains at most `base_bits - split_cost_bits`: a node that is
        // already (near-)deterministic cannot be worth splitting, so its
        // properties are not scored at all. Exact, and the bulk of the work
        // on flat graphics and text once the tree has isolated their zeros.
        if properties.len() >= self.p.min_node
            && (depth as usize) < MAX_DEPTH
            && base_bits > self.p.split_cost_bits
        {
            let mut best: Option<(usize, i32, f32)> = None; // (prop, threshold, bits)
            let alphabet = self.p.alphabet;
            let all_hist = scratch.ma_property.node_hist[..NUM_MA_PREDS * alphabet].to_vec();
            let mut all_nbits = [0u64; NUM_MA_PREDS];
            all_nbits.copy_from_slice(&scratch.ma_property.node_nbits[..NUM_MA_PREDS]);
            // Side candidates: the node's `side_preds` cheapest predictors.
            let mut preds: Vec<usize> = (0..NUM_MA_PREDS)
                .filter(|&p| {
                    (self.p.allow_wp || p != PRED_WEIGHTED) && self.p.allowed_preds & (1 << p) != 0
                })
                .collect();
            // Keep the stable sort's predictor-id tie order without evaluating
            // the same histogram entropy again for every comparison.
            crate::util::sort_small_by(&mut preds, |&a, &b| {
                scratch.ma_property.node_costs[a].total_cmp(&scratch.ma_property.node_costs[b])
            });
            preds.truncate(self.p.side_preds.max(1));
            let window = SymbolWindow::new(&preds, alphabet, &all_hist);
            let props = split_props(self.p.allow_wp);
            let max_candidates = self.p.max_candidates;
            let n = properties.len();

            // Candidate thresholds of every property, each with its own block
            // of the node's multi-property histogram.
            let MaPropertyScratch {
                plans,
                luts,
                multi_hist: hist,
                multi_bin_count: bin_count,
                ..
            } = &mut scratch.ma_property;
            let mut plans = std::mem::take(plans);
            let mut luts = std::mem::take(luts);
            let mut hist = std::mem::take(hist);
            let mut bin_count = std::mem::take(bin_count);
            plans.clear();
            luts.clear();
            let mut hist_len = 0usize;
            for &prop in props {
                if let Some(plan) = plan_split(
                    properties,
                    prop as usize,
                    max_candidates,
                    window.width,
                    hist_len as u32,
                    &mut luts,
                    &mut scratch.ma_property,
                ) {
                    hist_len += preds.len() * plan.pred_stride as usize;
                    plans.push(plan);
                }
            }
            let bc_len = plans.len() * (MAX_CANDIDATES + 1);

            let win = (window.lo, window.width);
            if n >= PARALLEL_SCAN_MIN_SAMPLES && pool.num_threads() > 1 {
                // Large node: its rows no longer sit in cache, so one pass
                // bins every property at once, in concurrent row chunks
                // whose histograms are summed afterwards.
                hist.clear();
                hist.resize(hist_len, 0);
                bin_count.clear();
                bin_count.resize(bc_len, 0);
                let chunks = (n / PARALLEL_SCAN_CHUNK_SAMPLES).clamp(2, pool.num_threads());
                let (plans_ref, luts_ref, preds_ref) = (&plans, &luts, &preds);
                let parts = pool.steal_map(scratch, chunks, |c, _worker_scratch| {
                    let start = c * n / chunks;
                    let end = (c + 1) * n / chunks;
                    let mut hist = vec![0u32; hist_len];
                    let mut bin_count = vec![0u32; bc_len];
                    scan_split_plans(
                        &properties[start..end],
                        &tokens[start..end],
                        plans_ref,
                        luts_ref,
                        preds_ref,
                        win,
                        &mut hist,
                        &mut bin_count,
                    );
                    (hist, bin_count)
                });
                for (part_hist, part_counts) in &parts {
                    for (a, &b) in hist.iter_mut().zip(part_hist) {
                        *a += b;
                    }
                    for (a, &b) in bin_count.iter_mut().zip(part_counts) {
                        *a += b;
                    }
                }
                for (k, plan) in plans.iter().enumerate() {
                    let start = plan.hist_start as usize;
                    let end = start + preds.len() * plan.pred_stride as usize;
                    let score = score_split_plan(
                        plan,
                        &preds,
                        alphabet,
                        &window,
                        &hist[start..end],
                        &bin_count[k * (MAX_CANDIDATES + 1)..(k + 1) * (MAX_CANDIDATES + 1)],
                        &all_hist,
                        &all_nbits,
                        n as u32,
                        &mut scratch.ma_property,
                    );
                    if let Some((t, bits)) = score
                        && (best.is_none() || bits < best.unwrap().2)
                    {
                        best = Some((plan.prop as usize, t, bits));
                    }
                }
            } else {
                // Cache-resident node: each property is binned and scored on
                // its own (concurrently for medium nodes); a small
                // single-property histogram stays in L1 and beats summing
                // per-chunk multi-property ones.
                let one_plan = |plan: &SplitPlan, sc: &mut MaPropertyScratch| {
                    sc.prepare(alphabet);
                    let mut local = *plan;
                    local.hist_start = 0;
                    let mut hist = std::mem::take(&mut sc.multi_hist);
                    let mut bin_count = std::mem::take(&mut sc.multi_bin_count);
                    hist.clear();
                    hist.resize(preds.len() * plan.pred_stride as usize, 0);
                    bin_count.clear();
                    bin_count.resize(MAX_CANDIDATES + 1, 0);
                    scan_split_plans(
                        properties,
                        tokens,
                        std::slice::from_ref(&local),
                        &luts,
                        &preds,
                        win,
                        &mut hist,
                        &mut bin_count,
                    );
                    let score = score_split_plan(
                        &local, &preds, alphabet, &window, &hist, &bin_count, &all_hist,
                        &all_nbits, n as u32, sc,
                    );
                    sc.multi_hist = hist;
                    sc.multi_bin_count = bin_count;
                    score
                };
                let scores: Vec<Option<(i32, f32)>> =
                    if n >= PARALLEL_PLAN_MIN_SAMPLES && pool.num_threads() > 1 {
                        let plans_ref = &plans;
                        pool.steal_map(scratch, plans.len(), |k, worker_scratch| {
                            one_plan(&plans_ref[k], &mut worker_scratch.ma_property)
                        })
                    } else {
                        plans
                            .iter()
                            .map(|plan| one_plan(plan, &mut scratch.ma_property))
                            .collect()
                    };
                for (plan, score) in plans.iter().zip(scores) {
                    if let Some((t, bits)) = score
                        && (best.is_none() || bits < best.unwrap().2)
                    {
                        best = Some((plan.prop as usize, t, bits));
                    }
                }
            }
            scratch.ma_property.plans = plans;
            scratch.ma_property.luts = luts;
            scratch.ma_property.multi_hist = hist;
            scratch.ma_property.multi_bin_count = bin_count;
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
            pred: base_pred as u32,
        }
    }
}

/// A grown-but-unsplit leaf with its best available split, ordered by gain.
struct Pending {
    node_id: u32,
    /// Best single predictor of the node (becomes the leaf predictor).
    pred: u32,
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

fn new_learner(samples: &mut MaSamples, params: MaLearnParams, nodes: Vec<MaNode>) -> Learner<'_> {
    let leaves = nodes
        .iter()
        .filter(|node| matches!(node, MaNode::Leaf { .. }))
        .count() as u32;
    Learner {
        s: samples,
        nodes,
        leaves,
        est_bits: 0.0,
        p: params,
    }
}

fn flat_cost(learner: &Learner<'_>, scratch: &mut CoderScratch) -> f64 {
    let samples = &*learner.s;
    let mut flat_bits = 0.0f64;
    let per_channel = samples.len().div_ceil(4);
    let mut chan_idx: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::with_capacity(per_channel));
    for (sample, props) in samples.props.iter().enumerate() {
        let c = props[0].clamp(0, 3) as usize;
        chan_idx[c].push(sample as u32);
    }
    for channel in &chan_idx {
        if !channel.is_empty() {
            let tokens = channel.iter().map(|&i| &samples.tok[i as usize]);
            let (bits, _) = node_cost_all(tokens, &learner.p, &mut scratch.ma_property);
            flat_bits += bits as f64;
        }
    }
    flat_bits
}

/// A pending split whose children are already scored, ahead of its turn.
struct Ahead {
    /// First row of the `le` child once the node's rows are partitioned
    /// (`start..lo` is `gt`), with both children's scores; `None` when the
    /// quantile candidate straddles no actual sample boundary.
    children: Option<(usize, Pending, Pending)>,
}

/// Pending splits scored ahead of their turn, on top of the one being
/// committed: the heap's best-first commit order is untouched (so the tree,
/// leaf cap included, is identical), only the child evaluations run in
/// batches of up to `2 * splits` concurrent tasks instead of two at a time.
/// This is what keeps the deep tail of small nodes, where a node's own scan
/// is too small to parallelize, off a single thread.
fn speculative_splits(pool: &ThreadPool) -> usize {
    (pool.num_threads() / 2).clamp(1, 8)
}

/// Rows of a node scored ahead of its turn are partitioned on a copy, so the
/// learner's own row order (which leaf-offset sampling later reads) changes
/// only when a split is committed; the copy bounds the speculated node size.
const SPECULATE_MAX_ROWS: usize = 1 << 16;

/// The learner's two-ended partition: `gt` rows move to the front in their
/// original relative order, the rest to the back. Returns the first `le` row.
fn partition_rows(
    props: &mut [[i32; NUM_MA_PROPS]],
    tok: &mut [[u8; NUM_MA_PREDS]],
    prop: u8,
    threshold: i32,
) -> usize {
    let mut lo = 0;
    let mut hi = props.len();
    while lo < hi {
        if props[lo][prop as usize] > threshold {
            lo += 1;
        } else {
            hi -= 1;
            props.swap(lo, hi);
            tok.swap(lo, hi);
        }
    }
    lo
}

fn grow_tree(
    mut learner: Learner<'_>,
    mut heap: std::collections::BinaryHeap<Pending>,
    flat_bits: f64,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedTree {
    let window = speculative_splits(pool);
    let mut ahead: std::collections::HashMap<u32, Ahead> = std::collections::HashMap::new();
    // Rows to score: the committed node's children in place, plus the
    // speculated nodes' children on their partitioned copies.
    struct Job {
        node_id: u32,
        start: usize,
        lo: usize,
        end: usize,
        depth: u16,
        copy: Option<MaSamples>,
    }
    while learner.leaves < learner.p.max_leaves as u32 {
        let Some(p) = heap.pop() else { break };
        let (prop, threshold) = p.split.expect("only splittable nodes are queued");
        if !ahead.contains_key(&p.node_id) {
            let mut jobs: Vec<Job> = Vec::with_capacity(window);
            // Use the original two-ended partition order, but move complete
            // rows instead of indices. Each child is a contiguous view of the
            // local rows; every child and quantile probe sees exactly the
            // original sequence.
            let lo = p.start
                + partition_rows(
                    &mut learner.s.props[p.start..p.end],
                    &mut learner.s.tok[p.start..p.end],
                    prop,
                    threshold,
                );
            jobs.push(Job {
                node_id: p.node_id,
                start: p.start,
                lo,
                end: p.end,
                depth: p.depth + 1,
                copy: None,
            });
            if window > 1 {
                // The next best pending splits by gain (ties may pick either,
                // which only decides what is computed early, never what is
                // committed). Their rows are untouched until they commit, so
                // a copy partitioned the same way holds exactly the rows the
                // children will then own.
                let mut next: Vec<&Pending> = heap
                    .iter()
                    .filter(|q| {
                        !ahead.contains_key(&q.node_id) && q.end - q.start <= SPECULATE_MAX_ROWS
                    })
                    .collect();
                next.sort_by(|a, b| b.gain.total_cmp(&a.gain));
                for q in next.into_iter().take(window - 1) {
                    let (prop, threshold) = q.split.expect("only splittable nodes are queued");
                    let mut copy = MaSamples {
                        props: learner.s.props[q.start..q.end].to_vec(),
                        tok: learner.s.tok[q.start..q.end].to_vec(),
                    };
                    let lo =
                        q.start + partition_rows(&mut copy.props, &mut copy.tok, prop, threshold);
                    jobs.push(Job {
                        node_id: q.node_id,
                        start: q.start,
                        lo,
                        end: q.end,
                        depth: q.depth + 1,
                        copy: Some(copy),
                    });
                }
            }
            let live: Vec<&Job> = jobs
                .iter()
                .filter(|job| job.lo != job.start && job.lo != job.end)
                .collect();
            let learner_ref = &learner;
            let score = |k: usize, worker_scratch: &mut CoderScratch| {
                let job = live[k / 2];
                let (start, end) = if k.is_multiple_of(2) {
                    (job.start, job.lo)
                } else {
                    (job.lo, job.end)
                };
                match &job.copy {
                    None => learner_ref.evaluate(0, start, end, job.depth, pool, worker_scratch),
                    Some(copy) => {
                        let rows = start - job.start..end - job.start;
                        learner_ref.evaluate_rows(
                            0,
                            start,
                            end,
                            &copy.props[rows.clone()],
                            &copy.tok[rows],
                            job.depth,
                            pool,
                            worker_scratch,
                        )
                    }
                }
            };
            let smallest = live
                .iter()
                .map(|job| (job.lo - job.start).min(job.end - job.lo))
                .min()
                .unwrap_or(0);
            let scored: Vec<Pending> = if pool.num_threads() > 1
                && (live.len() > 1 || smallest >= PARALLEL_CHILD_MIN_SAMPLES)
            {
                pool.steal_map(scratch, live.len() * 2, score)
            } else {
                (0..live.len() * 2).map(|k| score(k, scratch)).collect()
            };
            let mut scored = scored.into_iter();
            for job in &jobs {
                let children = (job.lo != job.start && job.lo != job.end).then(|| {
                    let gt = scored.next().expect("gt child");
                    let le = scored.next().expect("le child");
                    (job.lo, gt, le)
                });
                ahead.insert(job.node_id, Ahead { children });
            }
        } else {
            // Scored ahead on a copy: partition the learner's rows now.
            partition_rows(
                &mut learner.s.props[p.start..p.end],
                &mut learner.s.tok[p.start..p.end],
                prop,
                threshold,
            );
        }
        let Some((_, mut gt, mut le)) = ahead.remove(&p.node_id).expect("scored ahead").children
        else {
            // Quantile candidate straddled no actual sample boundary.
            continue;
        };

        let gt_id = learner.nodes.len() as u32;
        learner.nodes.push(MaNode::Leaf { pred: gt.pred });
        let le_id = learner.nodes.len() as u32;
        learner.nodes.push(MaNode::Leaf { pred: le.pred });
        learner.nodes[p.node_id as usize] = MaNode::Split {
            prop,
            val: threshold,
            gt: gt_id,
            le: le_id,
        };
        learner.leaves += 1;
        gt.node_id = gt_id;
        le.node_id = le_id;
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

/// Learns from an ordered selection of `samples`. Materialize just the selected
/// rows, retaining the input for any later deep stage. The learner partitions
/// its local rows in place and no longer needs the source index vector.
pub(crate) fn learn_ma_tree_indexed(
    samples: &MaSamples,
    idx: Vec<u32>,
    params: MaLearnParams,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedTree {
    let mut compact = {
        let selection = idx;
        samples.select(&selection)
    };
    learn_ma_tree_in_place(&mut compact, params, pool, scratch)
}

/// Learn without copying samples whose original order is no longer needed.
/// Partitions rows in place, using the same initial order as indexed learning.
pub(crate) fn learn_ma_tree_in_place(
    samples: &mut MaSamples,
    params: MaLearnParams,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedTree {
    let n = samples.len();
    let mut learner = new_learner(samples, params, vec![MaNode::Leaf { pred: 0 }]);
    let flat_bits = flat_cost(&learner, scratch);
    let mut heap = std::collections::BinaryHeap::new();
    let root = learner.evaluate(0, 0, n, 0, pool, scratch);
    learner.nodes[0] = MaNode::Leaf { pred: root.pred };
    learner.est_bits = root.base_bits as f64;
    if root.split.is_some() {
        heap.push(root);
    }
    grow_tree(learner, heap, flat_bits, pool, scratch)
}

/// Re-score a coarse tree on the full sample set and continue best-first
/// growth from its leaves. The expensive upper topology is retained instead
/// of being rediscovered by the deep stage.
pub(crate) fn deepen_ma_tree(
    samples: MaSamples,
    params: MaLearnParams,
    seed: LearnedTree,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (LearnedTree, MaSamples) {
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
    let (counts, offsets, idx) = {
        let mut leaf_for_sample = Vec::with_capacity(samples.len());
        let mut counts = vec![0usize; seed.nodes.len()];
        for props in &samples.props {
            let (leaf, _) = seed.lookup(props);
            leaf_for_sample.push(leaf);
            counts[leaf as usize] += 1;
        }
        let mut offsets = vec![0usize; seed.nodes.len() + 1];
        let mut total = 0;
        for (offset, &count) in offsets.iter_mut().skip(1).zip(&counts) {
            total += count;
            *offset = total;
        }
        let mut cursors = offsets[..seed.nodes.len()].to_vec();
        let mut idx = vec![0u32; samples.len()];
        for (sample, &leaf) in leaf_for_sample.iter().enumerate() {
            let cursor = &mut cursors[leaf as usize];
            idx[*cursor] = sample as u32;
            *cursor += 1;
        }

        (counts, offsets, idx)
    };
    let mut compact = samples.into_permuted(idx);
    let mut learner = new_learner(&mut compact, params, seed.nodes);
    let flat_bits = flat_cost(&learner, scratch);
    let mut heap = std::collections::BinaryHeap::new();
    let seed_leaves: Vec<usize> = (0..learner.nodes.len())
        .filter(|&node| matches!(learner.nodes[node], MaNode::Leaf { .. }) && counts[node] != 0)
        .collect();
    let pendings = {
        let learner_ref = &learner;
        pool.steal_map(scratch, seed_leaves.len(), |k, worker_scratch| {
            let node = seed_leaves[k];
            learner_ref.evaluate(
                node as u32,
                offsets[node],
                offsets[node + 1],
                depths[node],
                pool,
                worker_scratch,
            )
        })
    };
    for pending in pendings {
        learner.nodes[pending.node_id as usize] = MaNode::Leaf { pred: pending.pred };
        learner.est_bits += pending.base_bits as f64;
        if pending.split.is_some() {
            heap.push(pending);
        }
    }
    let tree = grow_tree(learner, heap, flat_bits, pool, scratch);
    (tree, compact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_permutation_matches_gather() {
        let mut state = 0x739feb12u32;
        for len in [0, 1, 2, 3, 31, 64, 257, 4096] {
            for kind in 0..4 {
                let mut samples = MaSamples::new();
                for i in 0..len {
                    samples.push(
                        core::array::from_fn(|p| (i * 101 + p) as i32),
                        core::array::from_fn(|p| (i * 13 + p) as u8),
                    );
                }
                let mut indices: Vec<u32> = (0..len as u32).collect();
                match kind {
                    1 => indices.reverse(),
                    2 if len > 0 => indices.rotate_left(1),
                    3 => {
                        for i in (1..len).rev() {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            indices.swap(i, state as usize % (i + 1));
                        }
                    }
                    _ => {}
                }
                let expected = samples.select(&indices);
                let props_ptr = samples.props.as_ptr();
                let tok_ptr = samples.tok.as_ptr();
                let actual = samples.into_permuted(indices);
                assert_eq!(actual.props, expected.props);
                assert_eq!(actual.tok, expected.tok);
                assert_eq!(actual.props.as_ptr(), props_ptr);
                assert_eq!(actual.tok.as_ptr(), tok_ptr);
            }
        }
    }

    #[test]
    fn probe_bitmap_matches_sort_dedup() {
        let mut state = 0x83d2eaf1u32;
        let mut bitmap = [u64::MAX; PROBE_BITMAP_BITS / 64];
        for len in [0, 1, 2, 31, 32, 33, 63, 64, 65, 1024, 2047] {
            for range in [1i64, 2, 63, 64, 65, 256, 4095, 4096, 4097, 65536, 1 << 32] {
                for min in [i32::MIN as i64, -(range / 2), i32::MAX as i64 - range + 1] {
                    let mut input = Vec::with_capacity(len);
                    for i in 0..len {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        let offset = match i {
                            0 => 0,
                            1 => range - 1,
                            _ => state as i64 % range,
                        };
                        input.push((min + offset) as i32);
                    }
                    let mut expected = input.clone();
                    expected.sort_unstable();
                    expected.dedup();
                    // Reuse the bitmap across different ranges, orders and
                    // fallback paths to catch stale presence bits as well.
                    for order in 0..3 {
                        if order == 1 {
                            input.sort_unstable();
                        } else if order == 2 {
                            input.reverse();
                        }
                        let mut actual = input.clone();
                        sort_unique_probe(&mut actual, &mut bitmap);
                        assert_eq!(actual, expected, "len={len}, range={range}, min={min}");
                    }
                }
            }
        }
    }

    #[test]
    fn cached_count_logs_match_direct_approximation() {
        let logs = count_log2_table();
        for count in (1..8192).chain([16_777_215, 16_777_216, 16_777_217, u32::MAX]) {
            assert_eq!(
                count_log2(count, logs).to_bits(),
                dirty_log2f(count as f32).to_bits(),
                "count={count}",
            );
        }
    }

    #[test]
    fn split_scoring_matches_direct_threshold_histograms() {
        let mut state = 0x83d2eaf1u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut scratch = MaPropertyScratch::default();
        // Exercise short and long nodes, LUT and binary-search bins, both candidate
        // budgets, and different side-predictor counts and orders.
        for n in [
            1, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 16385,
        ] {
            let mut samples = MaSamples::new();
            for _ in 0..n {
                let mut props = [0; NUM_MA_PROPS];
                props[2] = (next() % 83) as i32 - 41;
                props[3] = next() as i32;
                // Large nodes also exercise populations beyond the log cache.
                let symbols = if n > 1025 { 2 } else { 56 };
                samples.push(props, core::array::from_fn(|_| (next() % symbols) as u8));
            }
            // Include shuffled and repeated sample indices.
            let idx: Vec<u32> = (0..n).map(|_| next() % n as u32).collect();
            let selected = samples.select(&idx);
            let alphabet = 64;
            let mut all_hist = vec![0u32; NUM_MA_PREDS * alphabet];
            for &i in &idx {
                for (p, &token) in samples.tok[i as usize].iter().enumerate() {
                    all_hist[p * alphabet + token as usize] += 1;
                }
            }
            let all_nbits = core::array::from_fn(|p| {
                raw_bits_from_hist(&all_hist[p * alphabet..(p + 1) * alphabet])
            });
            for prop in [2, 3] {
                for max_candidates in [16, 32] {
                    for preds in [&[6][..], &[13, 0][..], &[9, 6, 0, 13][..]] {
                        let actual = best_split_on_prop(
                            &selected.props,
                            &selected.tok,
                            prop,
                            preds,
                            alphabet,
                            max_candidates,
                            &all_hist,
                            &all_nbits,
                            &mut scratch,
                        );
                        let mut expected = None;
                        for &threshold in &scratch.cands {
                            let mut left_hist = vec![0u32; all_hist.len()];
                            let mut left_count = 0;
                            for &i in &idx {
                                if samples.props[i as usize][prop] <= threshold {
                                    left_count += 1;
                                    for &p in preds {
                                        left_hist
                                            [p * alphabet + samples.tok[i as usize][p] as usize] +=
                                            1;
                                    }
                                }
                            }
                            if left_count == 0 || left_count == n as u32 {
                                continue;
                            }
                            let mut best_l = f32::INFINITY;
                            let mut best_r = f32::INFINITY;
                            for &p in preds {
                                let left = &left_hist[p * alphabet..(p + 1) * alphabet];
                                let right: Vec<u32> = all_hist[p * alphabet..(p + 1) * alphabet]
                                    .iter()
                                    .zip(left)
                                    .map(|(&a, &l)| a - l)
                                    .collect();
                                best_l = best_l.min(
                                    hist_entropy_bits(left, left_count)
                                        + raw_bits_from_hist(left) as f32,
                                );
                                best_r = best_r.min(
                                    hist_entropy_bits(&right, n as u32 - left_count)
                                        + raw_bits_from_hist(&right) as f32,
                                );
                            }
                            let cost = best_l + best_r;
                            if expected.is_none_or(|(_, best)| cost < best) {
                                expected = Some((threshold, cost));
                            }
                        }
                        assert_eq!(
                            actual, expected,
                            "n={n}, prop={prop}, candidates={max_candidates}, preds={preds:?}"
                        );
                    }
                }
            }
        }
    }

    use crate::entropy::{pack_signed, uint_encode};

    #[test]
    fn selected_rows_keep_properties_and_predictors_together() {
        let mut samples = MaSamples::new();
        for i in 0..32u8 {
            samples.push(
                [i as i32 - 16; NUM_MA_PROPS],
                std::array::from_fn(|p| i + p as u8),
            );
        }
        let mut selected = samples.select(&[31, 4, 17, 4]);
        selected.props.swap(0, 2);
        selected.tok.swap(0, 2);
        for (row, source) in [17, 4, 31, 4].into_iter().enumerate() {
            assert_eq!(selected.props[row], samples.props[source]);
            assert_eq!(selected.tok[row], samples.tok[source]);
        }
    }

    #[test]
    fn histogram_raw_bits_match_encoded_values() {
        let mut hist = [0u32; 128];
        let mut expected = 0u64;
        let mut add = |value| {
            let (token, nbits, _) = uint_encode(value);
            hist[token as usize] += 1;
            expected += nbits as u64;
        };
        for value in 0..65_536 {
            add(value);
        }
        for bit in 0..32 {
            let value = 1u32 << bit;
            for delta in 0..8 {
                add(value.saturating_sub(delta));
                add(value.saturating_add(delta));
            }
        }
        add(u32::MAX);
        assert_eq!(raw_bits_from_hist(&hist), expected);
    }

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
            for p in 0..NUM_MA_PREDS {
                // The good predictor has residual 0; all others residual 200.
                let res = if p == good_pred { 0 } else { 200 };
                let (t, _, _) = uint_encode(pack_signed(res));
                tok[p] = t as u8;
            }
            s.push(props, tok);
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
                allowed_preds: u16::MAX,
                side_preds: NUM_MA_PREDS,
                max_candidates: MAX_CANDIDATES,
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
                allowed_preds: u16::MAX,
                side_preds: NUM_MA_PREDS,
                max_candidates: MAX_CANDIDATES,
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
            for p in 0..NUM_MA_PREDS {
                let (t, _, _) = uint_encode(pack_signed(3));
                tok[p] = t as u8;
            }
            s.push(props, tok);
        }
        let tree = learn_ma_tree(
            &s,
            MaLearnParams {
                alphabet: 64,
                max_leaves: 8,
                split_cost_bits: 10.0,
                min_node: 64,
                allow_wp: true,
                allowed_preds: u16::MAX,
                side_preds: NUM_MA_PREDS,
                max_candidates: MAX_CANDIDATES,
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
            for pred in 0..NUM_MA_PREDS {
                let residual = props[4] - pred as i32 * 3;
                let (token, _, _) = uint_encode(pack_signed(residual));
                tok[pred] = token as u8;
            }
            samples.push(props, tok);
        }

        let indices = samples.evenly_sampled_indices(1024);
        let mut copied = MaSamples::with_capacity(indices.len());
        for &index in &indices {
            let index = index as usize;
            copied.push(samples.props[index], samples.tok[index]);
        }
        let params = || MaLearnParams {
            alphabet: 64,
            max_leaves: 16,
            split_cost_bits: 10.0,
            min_node: 64,
            allow_wp: true,
            allowed_preds: u16::MAX,
            side_preds: NUM_MA_PREDS,
            max_candidates: MAX_CANDIDATES,
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
        let in_place =
            learn_ma_tree_in_place(&mut copied, params(), &pool, &mut CoderScratch::default());
        assert_eq!(in_place.nodes, materialized.nodes);
        assert_eq!(in_place.est_bits.to_bits(), materialized.est_bits.to_bits());
        assert_eq!(
            in_place.flat_bits.to_bits(),
            materialized.flat_bits.to_bits()
        );
    }
}
