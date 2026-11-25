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
use crate::entropy::{pack_signed, uint_encode};
use crate::thread_pool::ThreadPool;
use crate::util::heap_array;
use std::sync::OnceLock;

mod properties;
pub(crate) use properties::{
    MaProperties, MaProperty, MaPropertySlice, with_ma_properties, write_property_row,
};

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

/// Zero and Weighted predictor ids in the decoder predictor enumeration.
pub(crate) const PRED_ZERO: usize = 0;
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
/// Single-threaded, nodes with at least this many rows bin every property
/// in one pass too.
const SERIAL_MULTI_SCAN_MIN_SAMPLES: usize = 1 << 14;
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
        /// Added to the prediction (libjxl `predictor_offset`): the leaf's
        /// most frequent residual under `modal_offsets` (for a Zero leaf, its
        /// most frequent value).
        offset: i32,
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
    pub(crate) fn lookup<T: MaProperty>(&self, p: &[T; NUM_MA_PROPS]) -> (u32, u32) {
        let mut i = 0usize;
        loop {
            match self.nodes[i] {
                MaNode::Split { prop, val, gt, le } => {
                    i = if p[prop as usize].into() > val {
                        gt
                    } else {
                        le
                    } as usize;
                }
                MaNode::Leaf { pred, .. } => return (i as u32, pred),
            }
        }
    }
}

pub(crate) struct MaSamples {
    pub(crate) props: MaProperties,
    pub(crate) tok: Vec<[u8; NUM_MA_PREDS]>,
    /// The sampled value itself, for modal leaves (`MaLearnParams::
    /// modal_offsets`); empty when the learn does not use them.
    pub(crate) val: Vec<i32>,
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
            props: MaProperties::with_capacity(capacity),
            tok: Vec::with_capacity(capacity),
            val: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, props: [i32; NUM_MA_PROPS], tok: [u8; NUM_MA_PREDS]) {
        self.props.push(props);
        self.tok.push(tok);
    }

    /// `push` with the sample's value (every row of a set has one or none).
    #[inline]
    pub(crate) fn push_val(
        &mut self,
        props: [i32; NUM_MA_PROPS],
        tok: [u8; NUM_MA_PREDS],
        val: i32,
    ) {
        self.props.push(props);
        self.tok.push(tok);
        self.val.push(val);
    }

    pub(crate) fn len(&self) -> usize {
        self.props.len()
    }

    /// Append `other`'s rows (neither carries values).
    pub(crate) fn append(&mut self, other: &Self) {
        debug_assert!(self.val.is_empty() && other.val.is_empty());
        self.props.append(&other.props);
        self.tok.extend_from_slice(&other.tok);
    }

    /// Materialize a learner's current order once. Copy each field in one
    /// pass so allocation checks stay outside the sample loop.
    fn select(&self, indices: &[u32]) -> Self {
        Self {
            props: self.props.select(indices),
            tok: select_rows(&self.tok, indices),
            val: if self.val.is_empty() {
                Vec::new()
            } else {
                select_rows(&self.val, indices)
            },
        }
    }

    /// Group the rows by `keys` (one per row; `counts[key]` rows each),
    /// keys ascending and each key's rows in their original order, in place.
    /// The property rows and the token/value rows are grouped side by side
    /// on two threads.
    fn into_grouped(mut self, keys: &[u16], counts: &[usize], pool: &ThreadPool) -> Self {
        assert_eq!(keys.len(), self.len());
        let Self { props, tok, val } = &mut self;
        pool.join(
            || {
                with_ma_properties!(props, |rows| group_rows_in_place(rows, keys, counts));
            },
            || {
                group_rows_in_place(tok, keys, counts);
                if !val.is_empty() {
                    group_rows_in_place(val, keys, counts);
                }
            },
        );
        self
    }

    // `i32::from` widens the i16 `Narrow` rows; it is a no-op on `Wide` rows.
    #[allow(clippy::useless_conversion)]
    fn partition(&mut self, start: usize, end: usize, prop: u8, threshold: i32) -> usize {
        with_ma_properties!(&mut self.props, |rows| {
            let tokens = &mut self.tok[start..end];
            let values = val_range(&mut self.val, start, end);
            partition_by(
                &mut rows[start..end],
                |row| i32::from(row[prop as usize]) > threshold,
                |a, b| {
                    tokens.swap(a, b);
                    if !values.is_empty() {
                        values.swap(a, b);
                    }
                },
            )
        })
    }

    /// Compute the partition order without copying or moving any sample rows.
    // `i32::from` widens the i16 `Narrow` rows; it is a no-op on `Wide` rows.
    #[allow(clippy::useless_conversion)]
    fn partition_indices(
        &self,
        start: usize,
        end: usize,
        prop: u8,
        threshold: i32,
    ) -> (Vec<u32>, usize) {
        let mut indices: Vec<_> = (start as u32..end as u32).collect();
        let split = with_ma_properties!(&self.props, |rows| {
            partition_by(
                &mut indices,
                |&i| i32::from(rows[i as usize][prop as usize]) > threshold,
                |_, _| {},
            )
        });
        (indices, split)
    }

    /// Deterministic evenly spaced selection for the coarse learning stage.
    /// Retain ordinals here so the learner materializes only its selected rows.
    pub(crate) fn evenly_sampled_indices(&self, target: usize) -> Vec<u32> {
        let selected = self.len().min(target);
        (0..selected)
            .map(|sample| (sample * self.len() / selected) as u32)
            .collect()
    }
}

/// Rows of one block of [`group_rows_in_place`]: a block of property rows is
/// 48-96 KiB, so the per-key buffers of a 32-leaf seed stay within a few
/// megabytes and every copy is a sequential run.
const GROUP_BLOCK_ROWS: usize = 1024;

/// Stable grouping of `rows` by `keys` (`counts[key]` rows per key) without
/// a second array. A cycle-following permutation moves every row in
/// cache-missing order and was a fifth of a large photo's wall time; here
/// every pass is sequential:
///
/// 1. Distribute: rows are read front to back into one block buffer per key;
///    a full buffer is written to the next block slot at the front, which the
///    read position has always passed.
/// 2. Order the written blocks by (key, sequence) with block swaps.
/// 3. From the last key down, shift each key's blocks right to its final
///    offset and append its partly filled buffer.
fn group_rows_in_place<T: Copy>(rows: &mut [T], keys: &[u16], counts: &[usize]) {
    const B: usize = GROUP_BLOCK_ROWS;
    debug_assert_eq!(rows.len(), keys.len());
    let num_keys = counts.len();
    let mut buffers: Vec<Vec<T>> = counts
        .iter()
        .map(|&count| Vec::with_capacity(count.min(B)))
        .collect();
    // Blocks written so far, per key, and each written slot's (key, sequence).
    let mut blocks = vec![0usize; num_keys];
    let mut slots: Vec<(u16, usize)> = Vec::with_capacity(rows.len() / B);
    for (i, &key) in keys.iter().enumerate() {
        let buffer = &mut buffers[key as usize];
        buffer.push(rows[i]);
        if buffer.len() == B {
            // Slots written, this one included, hold at most the i + 1 rows
            // read so far: the slot lies entirely behind the read position.
            let slot = slots.len();
            debug_assert!((slot + 1) * B <= i + 1);
            rows[slot * B..(slot + 1) * B].copy_from_slice(buffer);
            buffer.clear();
            slots.push((key, blocks[key as usize]));
            blocks[key as usize] += 1;
        }
    }

    // First block slot of each key once the blocks are in (key, sequence) order.
    let mut first_slot = vec![0usize; num_keys + 1];
    for key in 0..num_keys {
        first_slot[key + 1] = first_slot[key] + blocks[key];
    }
    let mut target: Vec<usize> = slots
        .iter()
        .map(|&(key, sequence)| first_slot[key as usize] + sequence)
        .collect();
    for slot in 0..target.len() {
        while target[slot] != slot {
            let other = target[slot];
            let (low, high) = rows.split_at_mut(slot.max(other) * B);
            low[slot.min(other) * B..][..B].swap_with_slice(&mut high[..B]);
            target.swap(slot, other);
        }
    }

    // A key's final offset is never left of its blocks (earlier keys only
    // add their buffered rows in between), so placing the last key first
    // overwrites nothing that is still to be moved.
    let mut end = rows.len();
    for key in (0..num_keys).rev() {
        let start = end - counts[key];
        let full = blocks[key] * B;
        let source = first_slot[key] * B;
        debug_assert!(source <= start);
        rows.copy_within(source..source + full, start);
        rows[start + full..end].copy_from_slice(&buffers[key]);
        end = start;
    }
    debug_assert_eq!(end, 0);
}

#[derive(Clone, Copy)]
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
    /// Modal leaves: the Zero predictor of every node carries the node's
    /// most frequent sample value as its offset, so a leaf predicts the
    /// constant its context implies (an AA level of a palette index image)
    /// instead of 0; Left and Top carry the node's most frequent step from
    /// that neighbour (the next palette entry of a ramp). Needs
    /// `MaSamples::val`.
    pub(crate) modal_offsets: bool,
    /// Grow-then-prune: bits one kept leaf is charged. 0 keeps best-first
    /// growth as it is. Otherwise the tree grows to its leaf cap at the
    /// caller's (small) `split_cost_bits` and `min_node`, then sibling
    /// leaves are merged bottom-up while the parent codes no more than the
    /// two of them plus this; greedy growth stops where no single split
    /// pays, pruning keeps a split whose descendants do.
    pub(crate) prune_leaf_bits: f32,
    /// A node holding more than one channel splits on the channel property
    /// only. The decoder resolves those splits once per channel, so every
    /// channel gets a tree of its own for free; left to the gains, the
    /// learner mixes channels in its early splits and codes larger.
    pub(crate) channel_first: bool,
    /// Bitmask of channel ids whose nodes may not use the Weighted Predictor
    /// or split on its error property (needs `channel_first`): the decoder
    /// then skips the per-pixel WP state for those channels.
    pub(crate) no_wp_channels: u32,
    /// Index first: a node above this depth holding `INDEX_FIRST_MIN_ROWS`
    /// rows splits on a neighbour's value (`INDEX_PROPS`) when that split
    /// pays and gains no less than 1 / `INDEX_FIRST_MAX_GAIN_RATIO` of the
    /// best one. On a palette index
    /// image ordered by colour those values name the colour family, and a
    /// tree that separates the families at its root shares the edge-shape
    /// splits below instead of repeating the partition inside every branch.
    /// 0 leaves the order to the gains.
    pub(crate) index_first_depth: u16,
}

/// The properties `MaLearnParams::index_first_depth` prefers: |N| and |W|.
const INDEX_PROPS: [u8; 2] = [4, 5];

/// Smaller nodes leave the split order to the gains: forced there, the
/// partition overfits small images.
const INDEX_FIRST_MIN_ROWS: usize = 4096;

/// Where another property gains more than this many times the best
/// neighbour-value split, the values name no family worth separating first
/// (scanned grey pages, whose trees grew 1.5% index first).
const INDEX_FIRST_MAX_GAIN_RATIO: f32 = 8.0;

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
    /// Every plan's `[bin][side pred][symbol]` histogram, back to back.
    multi_hist: Vec<u32>,
    /// One row's four histogram offsets, reused across single-property scans.
    split_slots: Vec<[u8; 4]>,
    /// Residual histogram scratch of `rebase_modal` (modal offsets).
    multi_bin_count: Vec<u32>,
    /// Per predictor: running histogram of the `le` side.
    left_hist: Vec<u32>,
    /// Per predictor: token histogram and raw bits of the whole node.
    node_hist: Vec<u32>,
    node_nbits: Vec<u64>,
    /// Costs already computed when choosing the node's best predictor.
    node_costs: [f32; NUM_MA_PREDS],
    /// Per channel and predictor: the whole sample set's flat histograms.
    flat_hist: Vec<u32>,
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
            split_slots: Vec::new(),
            multi_bin_count: Vec::new(),
            left_hist: Vec::new(),
            node_hist: Vec::new(),
            node_nbits: vec![0; NUM_MA_PREDS],
            node_costs: [f32::INFINITY; NUM_MA_PREDS],
            flat_hist: Vec::new(),
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
    /// Per node: what it codes as a leaf, kept once it is split.
    info: Vec<NodeInfo>,
    leaves: u32,
    est_bits: f64,
}

/// A node's cost and leaf predictor, so pruning can turn a split node back
/// into the leaf it was scored as.
#[derive(Clone, Copy)]
struct NodeInfo {
    base_bits: f32,
    pred: u32,
    offset: i32,
}

impl NodeInfo {
    fn of(pending: &Pending) -> Self {
        Self {
            base_bits: pending.base_bits,
            pred: pending.pred,
            offset: pending.offset,
        }
    }
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

fn pick_candidates<T: MaProperty>(
    properties: &[[T; NUM_MA_PROPS]],
    prop: usize,
    max_candidates: usize,
    scratch: &mut MaPropertyScratch,
) -> usize {
    scratch.probe.clear();
    let mut stride = (properties.len() / MAX_QUANTILE_PROBE).max(1);
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    scratch.probe.extend(
        properties
            .iter()
            .step_by(stride)
            .map(|props| Into::<i32>::into(props[prop])),
    );
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
    /// Start of this property's bins, each with packed side-predictor spans.
    hist_start: u32,
    /// One bin's span inside the block.
    bin_stride: u32,
    cands: [i32; MAX_CANDIDATES],
}

impl SplitPlan {
    /// bin = number of candidates < v; v <= cands[j] iff bin <= j.
    #[inline]
    fn bin_for(&self, v: i32, luts: &[u8]) -> usize {
        if self.use_lut {
            let offset = (v.max(self.lo_v).min(self.hi_v) - self.lo_v) as usize;
            luts[self.lut_start as usize + offset] as usize
        } else {
            self.cands[..self.ncand as usize].partition_point(|&c| c < v)
        }
    }

    fn hist_len(&self) -> usize {
        (self.ncand as usize + 1) * self.bin_stride as usize
    }
}

/// Symbol windows of a node. Each bin packs only the active symbol range of
/// each predictor, since every symbol outside that range contributes zero
/// to both sides of any split.
struct SymbolWindow {
    bin_stride: usize,
    spans: Vec<PredictorSpan>,
}

struct PredictorSpan {
    first: usize,
    width: usize,
    offset: usize,
    raw_syms: Vec<(u16, u8)>,
}

impl PredictorSpan {
    #[inline]
    fn slot(&self, token: u8) -> usize {
        self.offset + (token as usize - self.first)
    }
}

impl SymbolWindow {
    fn new(preds: &[usize], alphabet: usize, all_hist: &[u32]) -> Self {
        let mut bin_stride = 0;
        let mut spans = Vec::with_capacity(preds.len());
        for &p in preds {
            let h = &all_hist[p * alphabet..(p + 1) * alphabet];
            let first = h.iter().position(|&count| count != 0).unwrap_or(0);
            let end = h.iter().rposition(|&count| count != 0).map_or(0, |s| s + 1);
            let width = end - first;
            let raw_syms = (first.max(16)..end)
                .filter(|&s| h[s] != 0)
                .map(|s| ((s - first) as u16, ((s >> 2) - 2) as u8))
                .collect();
            spans.push(PredictorSpan {
                first,
                width,
                offset: bin_stride,
                raw_syms,
            });
            bin_stride += width;
        }
        Self {
            bin_stride: bin_stride.max(1),
            spans,
        }
    }
}

/// Candidate thresholds of `prop` over the node, or `None` when the property
/// has fewer than two distinct probed values. A narrow span gets its
/// value -> bin table appended to `luts`; `hist_start` is where the
/// property's block will live and `bin_stride` is its number of cells per bin.
#[allow(clippy::too_many_arguments)]
fn plan_split<T: MaProperty>(
    properties: &[[T; NUM_MA_PROPS]],
    prop: usize,
    max_candidates: usize,
    bin_stride: usize,
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
    // The maximum probed value is excluded, so the last threshold has room
    // for the overflow bin's sentinel.
    let hi_v = cands_src[ncand - 1] + 1;
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
        bin_stride: bin_stride as u32,
        cands,
    })
}

/// One pass over the node's rows, binning every planned property at once.
/// Every side predictor's token lies in its span, so a bin's row count is
/// the sum of any predictor's span. Pure integer counting, so chunks of rows
/// may be scanned separately and summed.
fn scan_split_plans<T: MaProperty>(
    properties: &[[T; NUM_MA_PROPS]],
    tokens: &[[u8; NUM_MA_PREDS]],
    plans: &[SplitPlan],
    luts: &[u8],
    preds: &[usize],
    window: &SymbolWindow,
    hist: &mut [u32],
) {
    debug_assert_eq!(properties.len(), tokens.len());
    if let &[p0, p1, p2, p3] = preds {
        // The encoder normally scores four side predictors: bind their
        // tokens once per row and keep the predictor loop out of the plan
        // loop.
        let [s0, s1, s2, s3] = &window.spans[..] else {
            unreachable!("one span per side predictor");
        };
        for (props, toks) in properties.iter().zip(tokens) {
            let t0 = s0.slot(toks[p0]);
            let t1 = s1.slot(toks[p1]);
            let t2 = s2.slot(toks[p2]);
            let t3 = s3.slot(toks[p3]);
            for plan in plans {
                let bin = plan.bin_for(props[plan.prop as usize].into(), luts);
                let base = plan.hist_start as usize + bin * plan.bin_stride as usize;
                hist[base + t0] += 1;
                hist[base + t1] += 1;
                hist[base + t2] += 1;
                hist[base + t3] += 1;
            }
        }
    } else {
        // A row's side-predictor slots are the same in every plan's bins:
        // resolve them once per row, then count them into each plan's bin.
        let n = preds.len().min(window.spans.len());
        let mut slots = [0u32; NUM_MA_PREDS];
        for (props, toks) in properties.iter().zip(tokens) {
            for (slot, (&p, span)) in slots.iter_mut().zip(preds.iter().zip(&window.spans)) {
                *slot = span.slot(toks[p]) as u32;
            }
            for plan in plans {
                let bin = plan.bin_for(props[plan.prop as usize].into(), luts);
                let stride = plan.bin_stride as usize;
                let base = plan.hist_start as usize + bin * stride;
                let cells = &mut hist[base..base + stride];
                for &slot in &slots[..n] {
                    cells[slot as usize] += 1;
                }
            }
        }
    }
}

/// A cache-resident node scans one property at a time. Bind its LUT or
/// thresholds once, so classification does not reload the plan in each row.
#[allow(clippy::too_many_arguments)]
fn scan_split_plan<T: MaProperty>(
    properties: &[[T; NUM_MA_PROPS]],
    tokens: &[[u8; NUM_MA_PREDS]],
    plan: &SplitPlan,
    luts: &[u8],
    preds: &[usize],
    window: &SymbolWindow,
    packed_slots: Option<&[[u8; 4]]>,
    hist: &mut [u32],
) {
    let &[p0, p1, p2, p3] = preds else {
        let mut local = *plan;
        local.hist_start = 0;
        scan_split_plans(
            properties,
            tokens,
            std::slice::from_ref(&local),
            luts,
            preds,
            window,
            hist,
        );
        return;
    };
    if let Some(slots) = packed_slots {
        debug_assert_eq!(properties.len(), slots.len());
        scan_split_plan_four(
            properties,
            plan,
            luts,
            slots.iter().map(|row| row.map(usize::from)),
            hist,
        );
    } else {
        scan_split_plan_four(
            properties,
            plan,
            luts,
            split_slots(tokens, [p0, p1, p2, p3], &window.spans),
            hist,
        );
    }
}

fn split_slots<'a>(
    tokens: &'a [[u8; NUM_MA_PREDS]],
    [p0, p1, p2, p3]: [usize; 4],
    spans: &'a [PredictorSpan],
) -> impl Iterator<Item = [usize; 4]> + 'a {
    let [s0, s1, s2, s3] = spans else {
        unreachable!("one span per side predictor");
    };
    assert!(p0 < NUM_MA_PREDS && p1 < NUM_MA_PREDS && p2 < NUM_MA_PREDS && p3 < NUM_MA_PREDS);
    tokens.iter().map(move |toks| {
        [
            s0.slot(toks[p0]),
            s1.slot(toks[p1]),
            s2.slot(toks[p2]),
            s3.slot(toks[p3]),
        ]
    })
}

fn scan_split_plan_four<T: MaProperty>(
    properties: &[[T; NUM_MA_PROPS]],
    plan: &SplitPlan,
    luts: &[u8],
    slots: impl Iterator<Item = [usize; 4]>,
    hist: &mut [u32],
) {
    let prop = plan.prop as usize;
    assert!(prop < NUM_MA_PROPS);
    let values = properties.iter().map(|row| row[prop].into());
    let stride = plan.bin_stride as usize;
    if plan.use_lut {
        let lo = plan.lo_v;
        let hi = plan.hi_v;
        let start = plan.lut_start as usize;
        let lut = &luts[start..=start + (hi - lo) as usize];
        scan_split_plan_bins(
            values.map(|v| lut[(v.max(lo).min(hi) - lo) as usize] as usize),
            slots,
            stride,
            hist,
        );
    } else {
        let candidates = &plan.cands[..plan.ncand as usize];
        scan_split_plan_bins(
            values.map(|v| candidates.partition_point(|&c| c < v)),
            slots,
            stride,
            hist,
        );
    }
}

#[inline]
fn scan_split_plan_bins(
    bins: impl Iterator<Item = usize>,
    slots: impl Iterator<Item = [usize; 4]>,
    stride: usize,
    hist: &mut [u32],
) {
    for (bin, [t0, t1, t2, t3]) in bins.zip(slots) {
        let base = bin * stride;
        hist[base + t0] += 1;
        hist[base + t1] += 1;
        hist[base + t2] += 1;
        hist[base + t3] += 1;
    }
}

/// Samples use uint_encode's fixed (4, 2, 0) config, so a symbol determines
/// its raw-bit count: zero below 16, otherwise (symbol / 4) - 2. Summing
/// over the symbols a predictor uses in the node avoids storing and
/// accumulating it for every sample.
#[inline]
fn raw_bits_windowed(bin: &[u32], syms: &[(u16, u8)]) -> u64 {
    syms.iter()
        .map(|&(s, bits)| bin[s as usize] as u64 * u64::from(bits))
        .sum()
}

/// Best threshold of one planned property from its binned block. `hist` is
/// the plan's packed bins; `all_hist` /
/// `all_nbits` are the node's per-predictor token histograms and raw-bit
/// totals.
#[allow(clippy::too_many_arguments)]
fn score_split_plan(
    plan: &SplitPlan,
    preds: &[usize],
    alphabet: usize,
    window: &SymbolWindow,
    hist: &[u32],
    all_hist: &[u32],
    all_nbits: &[u64; NUM_MA_PREDS],
    total_count: u32,
    scratch: &mut MaPropertyScratch,
) -> Option<(i32, f32)> {
    let ncand = plan.ncand as usize;
    let stride = plan.bin_stride as usize;
    let left_hist = &mut scratch.left_hist[..stride];
    left_hist.fill(0);
    let mut left_nbits = [0; NUM_MA_PREDS];
    let logs = count_log2_table();
    let mut left_count = 0u32;
    let mut best: Option<(i32, f32)> = None;
    for (&candidate, bin) in plan.cands[..ncand].iter().zip(hist.chunks_exact(stride)) {
        // Rows of bin j: every row's first side-predictor token lies in the
        // window, so its row of the bin sums to the bin's row count.
        left_count += bin[..window.spans[0].width].iter().sum::<u32>();
        if left_count == 0 {
            continue;
        }
        if left_count == total_count {
            break;
        }
        let log_left = count_log2(left_count, logs);
        let log_right = count_log2(total_count - left_count, logs);
        let mut best_l = f32::INFINITY;
        let mut best_r = f32::INFINITY;
        for (i, (&p, span)) in preds.iter().zip(&window.spans).enumerate() {
            let range = span.offset..span.offset + span.width;
            let bin = &bin[range.clone()];
            left_nbits[i] += raw_bits_windowed(bin, &span.raw_syms);
            let left = &mut left_hist[range];
            let node = &all_hist[p * alphabet + span.first..][..span.width];
            // Update and score in ascending symbol order. Outside this
            // predictor's active range both populations are zero.
            let mut cost_l = 0.0f32;
            let mut cost_r = 0.0f32;
            for ((left, &b), &n) in left.iter_mut().zip(bin).zip(node) {
                let l = *left + b;
                *left = l;
                if l != 0 {
                    cost_l += l as f32 * (log_left - count_log2(l, logs));
                }
                let c = n - l;
                if c != 0 {
                    cost_r += c as f32 * (log_right - count_log2(c, logs));
                }
            }
            cost_l += left_nbits[i] as f32;
            cost_r += (all_nbits[p] - left_nbits[i]) as f32;
            best_l = best_l.min(cost_l);
            best_r = best_r.min(cost_r);
        }
        let cost = best_l + best_r;
        if best.is_none() || cost < best.unwrap().1 {
            best = Some((candidate, cost));
        }
    }
    best
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
    let total = tokens.len() as u32;
    for toks in tokens {
        count_predictor_tokens(hist, alpha, toks);
    }
    predictor_costs(hist, total, params, node_nbits, node_costs)
}

#[inline]
fn count_predictor_tokens(hist: &mut [u32], alphabet: usize, tokens: &[u8; NUM_MA_PREDS]) {
    for (pred_hist, &token) in hist.chunks_exact_mut(alphabet).zip(tokens) {
        pred_hist[token as usize] += 1;
    }
}

fn predictor_costs(
    hist: &[u32],
    total: u32,
    params: &MaLearnParams,
    node_nbits: &mut [u64],
    node_costs: &mut [f32; NUM_MA_PREDS],
) -> (f32, usize) {
    let alpha = params.alphabet;
    node_costs.fill(f32::INFINITY);
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
        modal: ModalOffsets,
        score_split: bool,
        index_only: bool,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Pending {
        self.evaluate_rows(
            node_id,
            start,
            end,
            self.s.props.slice(start..end),
            &self.s.tok[start..end],
            depth,
            modal,
            score_split,
            index_only,
            pool,
            scratch,
        )
    }

    /// `rebase_modal` over the learner's rows `start..end` (zeros without
    /// `modal_offsets`).
    fn rebase(
        &mut self,
        start: usize,
        end: usize,
        base: Option<ModalOffsets>,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> ModalOffsets {
        if !self.p.modal_offsets || self.s.val.is_empty() {
            return [0; MODAL_PREDS.len()];
        }
        let tok = &mut self.s.tok[start..end];
        let val = &self.s.val[start..end];
        with_ma_properties!(&self.s.props, |rows| rebase_modal(
            tok,
            &rows[start..end],
            val,
            base,
            Some(pool),
            scratch,
        ))
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
        properties: MaPropertySlice<'_>,
        tokens: &[[u8; NUM_MA_PREDS]],
        depth: u16,
        modal: ModalOffsets,
        score_split: bool,
        index_only: bool,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Pending {
        match properties {
            MaPropertySlice::Narrow(rows) => self.evaluate_rows_typed(
                node_id,
                start,
                end,
                rows,
                tokens,
                depth,
                modal,
                score_split,
                index_only,
                pool,
                scratch,
            ),
            MaPropertySlice::Wide(rows) => self.evaluate_rows_typed(
                node_id,
                start,
                end,
                rows,
                tokens,
                depth,
                modal,
                score_split,
                index_only,
                pool,
                scratch,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_rows_typed<T: MaProperty>(
        &self,
        node_id: u32,
        start: usize,
        end: usize,
        properties: &[[T; NUM_MA_PROPS]],
        tokens: &[[u8; NUM_MA_PREDS]],
        depth: u16,
        modal: ModalOffsets,
        score_split: bool,
        index_only: bool,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Pending {
        debug_assert_eq!(properties.len(), end - start);
        // `channel_first`: the node's channel, or `None` when it holds several.
        let channel = (self.p.channel_first && !properties.is_empty()).then(|| {
            let first: i32 = properties[0][0].into();
            properties
                .iter()
                .all(|row| Into::<i32>::into(row[0]) == first)
                .then_some(first)
        });
        let no_wp = matches!(channel, Some(Some(c)) if (0..32).contains(&c) && self.p.no_wp_channels >> c & 1 != 0);
        let node_p;
        let params = if no_wp && self.p.allow_wp {
            node_p = MaLearnParams {
                allow_wp: false,
                ..self.p
            };
            &node_p
        } else {
            &self.p
        };
        let (base_bits, base_pred) = node_cost_all(tokens.iter(), params, &mut scratch.ma_property);
        let mut split = None;
        let mut gain = 0.0f32;
        // A split gains at most `base_bits - split_cost_bits`: a node that is
        // already (near-)deterministic cannot be worth splitting, so its
        // properties are not scored at all. Exact, and the bulk of the work
        // on flat graphics and text once the tree has isolated their zeros.
        if score_split
            && properties.len() >= self.p.min_node
            && (depth as usize) < MAX_DEPTH
            && base_bits > self.p.split_cost_bits
        {
            let mut best: Option<(usize, i32, f32)> = None; // (prop, threshold, bits)
            // Best split on a neighbour's value, tracked apart while the node
            // is within `index_first_depth`.
            let mut best_index: Option<(usize, i32, f32)> = None;
            let index_first =
                depth < self.p.index_first_depth && properties.len() >= INDEX_FIRST_MIN_ROWS;
            let consider = |best: &mut Option<(usize, i32, f32)>,
                            best_index: &mut Option<(usize, i32, f32)>,
                            prop: u8,
                            t: i32,
                            bits: f32| {
                if best.is_none_or(|b| bits < b.2) {
                    *best = Some((prop as usize, t, bits));
                }
                if index_first
                    && INDEX_PROPS.contains(&prop)
                    && best_index.is_none_or(|b| bits < b.2)
                {
                    *best_index = Some((prop as usize, t, bits));
                }
            };
            let alphabet = self.p.alphabet;
            let all_hist = scratch.ma_property.node_hist[..NUM_MA_PREDS * alphabet].to_vec();
            let mut all_nbits = [0u64; NUM_MA_PREDS];
            all_nbits.copy_from_slice(&scratch.ma_property.node_nbits[..NUM_MA_PREDS]);
            // Side candidates: the node's `side_preds` cheapest predictors.
            let mut preds: Vec<usize> = (0..NUM_MA_PREDS)
                .filter(|&p| {
                    (params.allow_wp || p != PRED_WEIGHTED) && params.allowed_preds & (1 << p) != 0
                })
                .collect();
            // Keep the stable sort's predictor-id tie order without evaluating
            // the same histogram entropy again for every comparison.
            crate::util::sort_small_by(&mut preds, |&a, &b| {
                scratch.ma_property.node_costs[a].total_cmp(&scratch.ma_property.node_costs[b])
            });
            preds.truncate(self.p.side_preds.max(1));
            let window = SymbolWindow::new(&preds, alphabet, &all_hist);
            let all_props: &[u8] = if channel == Some(None) {
                &[0]
            } else {
                split_props(params.allow_wp)
            };
            // Below an index split only the index properties are scored
            // first; every property follows where none of them pays.
            let index_only = index_only && index_first && channel != Some(None);
            let max_candidates = self.p.max_candidates;
            let n = properties.len();

            // Candidate thresholds of every property, each with its own block
            // of the node's multi-property histogram.
            let MaPropertyScratch {
                plans,
                luts,
                multi_hist: hist,
                ..
            } = &mut scratch.ma_property;
            let mut plans = std::mem::take(plans);
            let mut luts = std::mem::take(luts);
            let mut hist = std::mem::take(hist);
            for index_pass in [index_only, false] {
                let props = if index_pass {
                    &INDEX_PROPS[..]
                } else {
                    all_props
                };
                plans.clear();
                luts.clear();
                let mut hist_len = 0usize;
                let mut planned = 0u32;
                for &prop in props {
                    // An absolute-value property and its signed counterpart can
                    // be identical throughout this node. Their quantiles, bins,
                    // and costs then match, and the earlier property wins ties.
                    let absolute = match prop {
                        6 | 7 => Some(prop - 2),
                        17 | 19 | 21 | 23 => Some(prop - 1),
                        _ => None,
                    };
                    if let Some(absolute) = absolute
                        && planned & (1 << absolute) != 0
                        && properties.iter().all(|row| {
                            let a: i32 = row[absolute as usize].into();
                            let b: i32 = row[prop as usize].into();
                            a == b
                        })
                    {
                        continue;
                    }
                    if let Some(plan) = plan_split(
                        properties,
                        prop as usize,
                        max_candidates,
                        window.bin_stride,
                        hist_len as u32,
                        &mut luts,
                        &mut scratch.ma_property,
                    ) {
                        hist_len += plan.hist_len();
                        plans.push(plan);
                        planned |= 1 << prop;
                    }
                }

                let multi_scan = if pool.num_threads() > 1 {
                    n >= PARALLEL_SCAN_MIN_SAMPLES
                } else {
                    n >= SERIAL_MULTI_SCAN_MIN_SAMPLES
                };
                if multi_scan {
                    // Large node: its rows no longer sit in cache, so one pass
                    // bins every property at once, in concurrent row chunks
                    // whose histograms are summed afterwards (one chunk on a
                    // single thread).
                    hist.clear();
                    hist.resize(hist_len, 0);
                    let (plans_ref, luts_ref, preds_ref) = (&plans, &luts, &preds);
                    if pool.num_threads() > 1 {
                        let chunks = (n / PARALLEL_SCAN_CHUNK_SAMPLES).clamp(2, pool.num_threads());
                        let parts = pool.steal_map(scratch, chunks, |c, _worker_scratch| {
                            let start = c * n / chunks;
                            let end = (c + 1) * n / chunks;
                            let mut hist = vec![0u32; hist_len];
                            scan_split_plans(
                                &properties[start..end],
                                &tokens[start..end],
                                plans_ref,
                                luts_ref,
                                preds_ref,
                                &window,
                                &mut hist,
                            );
                            hist
                        });
                        for part_hist in &parts {
                            for (a, &b) in hist.iter_mut().zip(part_hist) {
                                *a += b;
                            }
                        }
                    } else {
                        scan_split_plans(
                            properties, tokens, plans_ref, luts_ref, preds_ref, &window, &mut hist,
                        );
                    }
                    for plan in plans.iter() {
                        let start = plan.hist_start as usize;
                        let end = start + plan.hist_len();
                        let score = score_split_plan(
                            plan,
                            &preds,
                            alphabet,
                            &window,
                            &hist[start..end],
                            &all_hist,
                            &all_nbits,
                            n as u32,
                            &mut scratch.ma_property,
                        );
                        if let Some((t, bits)) = score {
                            consider(&mut best, &mut best_index, plan.prop, t, bits);
                        }
                    }
                } else {
                    // Cache-resident node: each property is binned and scored on
                    // its own (concurrently for medium nodes); a small
                    // single-property histogram stays in L1 and beats summing
                    // per-chunk multi-property ones.
                    let mut slots = std::mem::take(&mut scratch.ma_property.split_slots);
                    slots.clear();
                    if plans.len() >= 4
                        && window.bin_stride <= 256
                        && let &[p0, p1, p2, p3] = &preds[..]
                    {
                        // When the active spans occupy at most 256 cells, one
                        // byte represents each row's offset.
                        slots.extend(
                            split_slots(tokens, [p0, p1, p2, p3], &window.spans)
                                .map(|row| row.map(|slot| slot as u8)),
                        );
                    }
                    let packed_slots = (!slots.is_empty()).then_some(slots.as_slice());
                    let one_plan = |plan: &SplitPlan, sc: &mut MaPropertyScratch| {
                        sc.prepare(alphabet);
                        let mut hist = std::mem::take(&mut sc.multi_hist);
                        hist.clear();
                        hist.resize(plan.hist_len(), 0);
                        scan_split_plan(
                            properties,
                            tokens,
                            plan,
                            &luts,
                            &preds,
                            &window,
                            packed_slots,
                            &mut hist,
                        );
                        let score = score_split_plan(
                            plan, &preds, alphabet, &window, &hist, &all_hist, &all_nbits,
                            n as u32, sc,
                        );
                        sc.multi_hist = hist;
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
                        if let Some((t, bits)) = score {
                            consider(&mut best, &mut best_index, plan.prop, t, bits);
                        }
                    }
                    scratch.ma_property.split_slots = slots;
                }
                if !index_pass || best.is_some_and(|b| base_bits - b.2 > self.p.split_cost_bits) {
                    break;
                }
                best = None;
                best_index = None;
            }
            scratch.ma_property.plans = plans;
            scratch.ma_property.luts = luts;
            scratch.ma_property.multi_hist = hist;
            let best = best_index
                .filter(|index| {
                    let gain = base_bits - index.2;
                    gain > self.p.split_cost_bits
                        && best.is_none_or(|free| {
                            gain * INDEX_FIRST_MAX_GAIN_RATIO >= base_bits - free.2
                        })
                })
                .or(best);
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
            offset: MODAL_PREDS
                .iter()
                .position(|&pred| pred == base_pred)
                .map_or(0, |k| modal[k]),
            modal,
        }
    }
}

/// A grown-but-unsplit leaf with its best available split, ordered by gain.
struct Pending {
    node_id: u32,
    /// Best single predictor of the node (becomes the leaf predictor).
    pred: u32,
    /// Leaf offset: the node's modal value when `pred` is Zero under
    /// `modal_offsets`, else 0.
    offset: i32,
    /// The offsets the node's `MODAL_PREDS` tokens are coded against.
    modal: ModalOffsets,
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
    // Seed splits have no leaf score: never pruned.
    let info = nodes
        .iter()
        .map(|node| match *node {
            MaNode::Leaf { pred, offset } => NodeInfo {
                base_bits: 0.0,
                pred,
                offset,
            },
            MaNode::Split { .. } => NodeInfo {
                base_bits: f32::INFINITY,
                pred: 0,
                offset: 0,
            },
        })
        .collect();
    Learner {
        s: samples,
        nodes,
        info,
        leaves,
        est_bits: 0.0,
        p: params,
    }
}

fn flat_cost(learner: &Learner<'_>, scratch: &mut CoderScratch) -> f64 {
    let samples = &*learner.s;
    let alphabet = learner.p.alphabet;
    let channel_stride = NUM_MA_PREDS * alphabet;
    let MaPropertyScratch {
        flat_hist,
        node_nbits,
        node_costs,
        ..
    } = &mut scratch.ma_property;
    flat_hist.clear();
    flat_hist.resize(4 * channel_stride, 0);
    let mut counts = [0u32; 4];
    with_ma_properties!(&samples.props, |rows| {
        for (props, tokens) in rows.iter().zip(&samples.tok) {
            let c = props[0].clamp(0, 3) as usize;
            counts[c] += 1;
            let hist = &mut flat_hist[c * channel_stride..][..channel_stride];
            count_predictor_tokens(hist, alphabet, tokens);
        }
    });
    let mut flat_bits = 0.0;
    for (hist, count) in flat_hist.chunks_exact(channel_stride).zip(counts) {
        if count != 0 {
            let (bits, _) = predictor_costs(hist, count, &learner.p, node_nbits, node_costs);
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
    match pool.num_threads() {
        1 => 1,
        threads => (2 * threads).min(SPECULATE_MAX_WINDOW),
    }
}

/// Upper bound of the speculation window (nodes scored per round).
const SPECULATE_MAX_WINDOW: usize = 8;

/// Bound the size of the child copies scored ahead of their turn. The
/// learner's own row order changes only when a split is committed.
const SPECULATE_MAX_ROWS: usize = 1 << 16;

/// The learner's two-ended partition. Returns the first `le` row. Use the
/// same swap order for rows and indices so every quantile probe sees the
/// same sequence whether it is scored before or after committing a split.
fn partition_by<T>(
    rows: &mut [T],
    is_gt: impl Fn(&T) -> bool,
    mut swap_parallel_rows: impl FnMut(usize, usize),
) -> usize {
    let mut lo = 0;
    let mut hi = rows.len();
    while lo < hi {
        if is_gt(&rows[lo]) {
            lo += 1;
        } else {
            hi -= 1;
            rows.swap(lo, hi);
            swap_parallel_rows(lo, hi);
        }
    }
    lo
}

/// `val[start..end]`, or an empty slice for a sample set without values.
fn val_range(val: &mut [i32], start: usize, end: usize) -> &mut [i32] {
    let n = val.len();
    &mut val[start.min(n)..end.min(n)]
}

/// Predictors whose tokens follow the node's modal residual: Zero, Left, Top.
const MODAL_PREDS: [usize; 3] = [PRED_ZERO, 1, 2];
/// The property holding each of `MODAL_PREDS`' predictions (none for Zero).
const MODAL_PRED_PROPS: [Option<usize>; 3] = [None, Some(7), Some(6)];
/// One offset per `MODAL_PREDS` entry.
type ModalOffsets = [i32; MODAL_PREDS.len()];

/// Residual spans up to this, or up to four bins per row, are binned to find
/// the mode; wider ones are sorted.
const MODAL_MAX_SPAN: i64 = 4096;

/// Nodes of more rows keep Left and Top at offset 0: a step worth an
/// offset shows once the neighbour's own value is pinned down, deep in the
/// tree, and scanning every large node for one is most of the cost.
const MODAL_STEP_MAX_ROWS: usize = 4096;

/// Tokens below this are the packed residual itself.
const EXACT_TOKENS: usize = 16;

/// The most frequent of `rows` residuals, the smallest on a tie.
fn residual_mode(residual: impl Fn(usize) -> i64, rows: usize, hist: &mut Vec<u32>) -> i64 {
    let (lo, hi) = (0..rows).fold((i64::MAX, i64::MIN), |(lo, hi), i| {
        let r = residual(i);
        (lo.min(r), hi.max(r))
    });
    let span = hi - lo + 1;
    hist.clear();
    if span > u32::MAX as i64 {
        0
    } else if span > MODAL_MAX_SPAN.max(4 * rows as i64) {
        // Sparse over a wide range: sort instead of binning.
        hist.extend((0..rows).map(|i| (residual(i) - lo) as u32));
        hist.sort_unstable();
        let (mut best, mut count, mut run) = (0u32, 0usize, 0usize);
        for (i, &r) in hist.iter().enumerate() {
            run = if i > 0 && hist[i - 1] == r {
                run + 1
            } else {
                1
            };
            if run > count {
                (best, count) = (r, run);
            }
        }
        lo + i64::from(best)
    } else {
        hist.resize(span as usize, 0);
        for i in 0..rows {
            hist[(residual(i) - lo) as usize] += 1;
        }
        let (best, _) =
            hist.iter().enumerate().fold(
                (0usize, 0u32),
                |acc, (i, &c)| if c > acc.1 { (i, c) } else { acc },
            );
        lo + best as i64
    }
}

/// Counts of the exact tokens of the Left and Top columns (by `MODAL_PREDS`
/// index).
fn near_counts(tok: &[[u8; NUM_MA_PREDS]]) -> [[u32; EXACT_TOKENS]; MODAL_PREDS.len()] {
    let mut near = [[0u32; EXACT_TOKENS]; MODAL_PREDS.len()];
    for t in tok {
        // Zero's mode is searched in the values: its column is not counted.
        for (counts, &pred) in near.iter_mut().zip(&MODAL_PREDS).skip(1) {
            if let Some(count) = counts.get_mut(t[pred] as usize) {
                *count += 1;
            }
        }
    }
    near
}

/// Row `i`'s residual of `MODAL_PREDS[k]`.
#[inline]
fn modal_residual<T: MaProperty>(
    props: &[[T; NUM_MA_PROPS]],
    val: &[i32],
    k: usize,
    i: usize,
) -> i64 {
    i64::from(val[i])
        - MODAL_PRED_PROPS[k].map_or(0, |prop| Into::<i32>::into(props[i][prop]) as i64)
}

/// Code the `MODAL_PREDS[k]` tokens of the rows against `offset`.
fn retoken<T: MaProperty>(
    tok: &mut [[u8; NUM_MA_PREDS]],
    props: &[[T; NUM_MA_PROPS]],
    val: &[i32],
    k: usize,
    offset: i64,
) {
    for (i, t) in tok.iter_mut().enumerate() {
        let (token, _, _) = uint_encode(pack_signed(
            (modal_residual(props, val, k, i) - offset) as i32,
        ));
        t[MODAL_PREDS[k]] = token.min(u8::MAX as u32) as u8;
    }
}

/// Nodes from this many rows on rewrite their tokens in concurrent row
/// chunks (given a pool).
const PARALLEL_REBASE_MIN_ROWS: usize = 1 << 15;

/// Rebase a node's `MODAL_PREDS` tokens on each one's modal residual: a leaf
/// of the node using that predictor adds the mode as its offset (a Zero leaf
/// predicts the node's most frequent value, a Left leaf its most frequent
/// step from the west neighbour), and the split scan prices both children
/// against it. Returns the offsets. Left and Top keep offset 0 in nodes of
/// more than `MODAL_STEP_MAX_ROWS` rows. `base` holds the offsets the tokens
/// are coded against on entry (`None` when unknown): a predictor keeping its
/// offset keeps its tokens, and those tokens name the steps near it exactly,
/// which settles its mode unless the far ones could outnumber it.
fn rebase_modal<T: MaProperty + Sync>(
    tok: &mut [[u8; NUM_MA_PREDS]],
    props: &[[T; NUM_MA_PROPS]],
    val: &[i32],
    base: Option<ModalOffsets>,
    pool: Option<&ThreadPool>,
    scratch: &mut CoderScratch,
) -> ModalOffsets {
    const N: usize = MODAL_PREDS.len();
    let rows = val.len();
    if rows == 0 {
        return [0; N];
    }
    let steps = rows <= MODAL_STEP_MAX_ROWS;
    let near = match base {
        Some(_) if steps => near_counts(tok),
        _ => [[0; EXACT_TOKENS]; N],
    };
    let hist = &mut scratch.ma_property.multi_bin_count;
    let modes: [i64; N] = core::array::from_fn(|k| {
        if k != 0 && !steps {
            return 0;
        }
        let near_mode = base.filter(|_| k != 0).and_then(|base| {
            let reach = (EXACT_TOKENS / 2) as i32;
            let (mut best, mut count) = (0i32, 0u32);
            for r in -reach..reach {
                let c = near[k][pack_signed(r) as usize];
                if c > count {
                    (best, count) = (r, c);
                }
            }
            let far = rows as u32 - near[k].iter().sum::<u32>();
            (count > far).then(|| i64::from(base[k]) + i64::from(best))
        });
        near_mode.unwrap_or_else(|| residual_mode(|i| modal_residual(props, val, k, i), rows, hist))
    });
    let pool = pool.filter(|pool| pool.num_threads() > 1 && rows >= PARALLEL_REBASE_MIN_ROWS);
    for k in 0..N {
        if base.is_some_and(|base| i64::from(base[k]) == modes[k]) {
            continue;
        }
        match pool {
            None => retoken(tok, props, val, k, modes[k]),
            Some(pool) => {
                let chunk_rows = rows
                    .div_ceil((rows / PARALLEL_SCAN_CHUNK_SAMPLES).clamp(2, pool.num_threads()));
                let mut chunks: Vec<&mut [[u8; NUM_MA_PREDS]]> =
                    tok.chunks_mut(chunk_rows).collect();
                pool.steal_for_each_mut(scratch, &mut chunks, |chunk, tok, _| {
                    let start = chunk * chunk_rows;
                    let end = start + tok.len();
                    retoken(tok, &props[start..end], &val[start..end], k, modes[k]);
                });
            }
        }
    }
    modes.map(|mode| mode as i32)
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
    let modal = learner.p.modal_offsets && !learner.s.val.is_empty();
    while learner.leaves < learner.p.max_leaves as u32 {
        let Some(p) = heap.pop() else { break };
        let remaining = (learner.p.max_leaves as u32 - learner.leaves) as usize;
        let (prop, threshold) = p.split.expect("only splittable nodes are queued");
        if !ahead.contains_key(&p.node_id) {
            // The committed node: use the original two-ended partition order,
            // but move complete rows instead of indices. Each child is a
            // contiguous view of the local rows; every child and quantile
            // probe sees exactly the original sequence.
            let lo = p.start + learner.s.partition(p.start, p.end, prop, threshold);
            let offsets = (
                learner.rebase(p.start, lo, Some(p.modal), pool, scratch),
                learner.rebase(lo, p.end, Some(p.modal), pool, scratch),
            );
            let committed_live = lo != p.start && lo != p.end;
            // Speculated nodes: the next best pending splits by gain (ties
            // may pick either, which only decides what is computed early,
            // never what is committed). Their rows are untouched until they
            // commit, so partition their indices to find the rows each child
            // will own in the same order.
            let mut speculated: Vec<(u32, usize, usize, u8, i32, u16, ModalOffsets)> = Vec::new();
            let speculate = window.min(remaining) - 1;
            if speculate > 0 {
                let mut next: Vec<&Pending> = heap
                    .iter()
                    .filter(|q| {
                        !ahead.contains_key(&q.node_id) && q.end - q.start <= SPECULATE_MAX_ROWS
                    })
                    .collect();
                next.sort_by(|a, b| b.gain.total_cmp(&a.gain));
                for q in next.into_iter().take(speculate) {
                    let (prop, threshold) = q.split.expect("only splittable nodes are queued");
                    speculated.push((
                        q.node_id,
                        q.start,
                        q.end,
                        prop,
                        threshold,
                        q.depth + 1,
                        q.modal,
                    ));
                }
            }
            // Two tasks per node, `gt` then `le`. A committed child scores
            // its rows in place; a speculated child gathers only its own rows
            // after partitioning the indices.
            let committed_tasks = if committed_live { 2 } else { 0 };
            let learner_ref = &learner;
            let speculated_ref = &speculated;
            let score = |k: usize, worker_scratch: &mut CoderScratch| -> Option<(usize, Pending)> {
                if k < committed_tasks {
                    let (start, end, offset) = if k == 0 {
                        (p.start, lo, offsets.0)
                    } else {
                        (lo, p.end, offsets.1)
                    };
                    let pending = learner_ref.evaluate(
                        0,
                        start,
                        end,
                        p.depth + 1,
                        offset,
                        // On the last split, the children only need their leaf scores.
                        remaining > 1,
                        INDEX_PROPS.contains(&prop),
                        pool,
                        worker_scratch,
                    );
                    return Some((lo, pending));
                }
                let j = (k - committed_tasks) / 2;
                let (_, start, end, prop, threshold, depth, base) = speculated_ref[j];
                let (indices, split_at) =
                    learner_ref.s.partition_indices(start, end, prop, threshold);
                if split_at == 0 || split_at == indices.len() {
                    // Quantile candidate straddled no actual sample boundary.
                    return None;
                }
                let rows = if (k - committed_tasks).is_multiple_of(2) {
                    0..split_at
                } else {
                    split_at..indices.len()
                };
                let mut copy = learner_ref.s.select(&indices[rows.clone()]);
                let offset = if modal {
                    with_ma_properties!(&copy.props, |rows| rebase_modal(
                        &mut copy.tok,
                        rows,
                        &copy.val,
                        Some(base),
                        None,
                        worker_scratch,
                    ))
                } else {
                    [0; MODAL_PREDS.len()]
                };
                let pending = learner_ref.evaluate_rows(
                    0,
                    start + rows.start,
                    start + rows.end,
                    copy.props.slice(0..copy.len()),
                    &copy.tok,
                    depth,
                    offset,
                    true,
                    INDEX_PROPS.contains(&prop),
                    pool,
                    worker_scratch,
                );
                Some((start + split_at, pending))
            };
            let tasks = committed_tasks + 2 * speculated.len();
            let smallest = if committed_live {
                (lo - p.start).min(p.end - lo)
            } else {
                0
            };
            let scored: Vec<Option<(usize, Pending)>> = if pool.num_threads() > 1
                && (tasks > 2 || smallest >= PARALLEL_CHILD_MIN_SAMPLES)
            {
                pool.steal_map(scratch, tasks, score)
            } else {
                (0..tasks).map(|k| score(k, scratch)).collect()
            };
            let mut scored = scored.into_iter();
            let children = committed_live.then(|| {
                let (_, gt) = scored.next().expect("gt child").expect("committed gt");
                let (_, le) = scored.next().expect("le child").expect("committed le");
                (lo, gt, le)
            });
            ahead.insert(p.node_id, Ahead { children });
            for &(node_id, ..) in &speculated {
                let gt = scored.next().expect("gt child");
                let le = scored.next().expect("le child");
                let children = match (gt, le) {
                    (Some((lo, gt)), Some((_, le))) => Some((lo, gt, le)),
                    _ => None,
                };
                ahead.insert(node_id, Ahead { children });
            }
        } else {
            // Scored ahead on a copy: partition the learner's rows now, and
            // rebase the children's Zero tokens as the copy was.
            let lo = p.start + learner.s.partition(p.start, p.end, prop, threshold);
            learner.rebase(p.start, lo, Some(p.modal), pool, scratch);
            learner.rebase(lo, p.end, Some(p.modal), pool, scratch);
        }
        let Some((_, mut gt, mut le)) = ahead.remove(&p.node_id).expect("scored ahead").children
        else {
            // Quantile candidate straddled no actual sample boundary.
            continue;
        };

        let gt_id = learner.nodes.len() as u32;
        learner.nodes.push(MaNode::Leaf {
            pred: gt.pred,
            offset: gt.offset,
        });
        let le_id = learner.nodes.len() as u32;
        learner.nodes.push(MaNode::Leaf {
            pred: le.pred,
            offset: le.offset,
        });
        learner.nodes[p.node_id as usize] = MaNode::Split {
            prop,
            val: threshold,
            gt: gt_id,
            le: le_id,
        };
        learner.info[p.node_id as usize] = NodeInfo::of(&p);
        learner.info.push(NodeInfo::of(&gt));
        learner.info.push(NodeInfo::of(&le));
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

    if learner.p.prune_leaf_bits > 0.0 {
        learner.est_bits = prune_tree(&mut learner.nodes, &learner.info, learner.p.prune_leaf_bits);
    }

    LearnedTree {
        est_bits: learner.est_bits,
        flat_bits,
        nodes: learner.nodes,
    }
}

/// Grow-then-prune (`MaLearnParams::prune_leaf_bits`): bottom-up over the
/// grown tree, a split becomes the leaf it was scored as where that leaf
/// codes no more than its pruned subtree plus one leaf's bits per split
/// kept, whatever the subtree's depth. Children are numbered after their
/// parent, so one descending pass prices every subtree before its parent.
/// The reachable nodes are then renumbered in preorder. Returns the pruned
/// tree's estimated bits.
fn prune_tree(nodes: &mut Vec<MaNode>, info: &[NodeInfo], leaf_bits: f32) -> f64 {
    let n = nodes.len();
    // Leaf bits of each pruned subtree, and the same plus the penalty of
    // every split it keeps: the first is the estimate, the second decides.
    let mut subtree_bits = vec![0.0f64; n];
    let mut priced_bits = vec![0.0f64; n];
    for i in (0..n).rev() {
        match nodes[i] {
            MaNode::Leaf { .. } => {
                subtree_bits[i] = info[i].base_bits as f64;
                priced_bits[i] = subtree_bits[i];
            }
            MaNode::Split { gt, le, .. } => {
                let kept = priced_bits[gt as usize] + priced_bits[le as usize] + leaf_bits as f64;
                if info[i].base_bits as f64 <= kept {
                    nodes[i] = MaNode::Leaf {
                        pred: info[i].pred,
                        offset: info[i].offset,
                    };
                    subtree_bits[i] = info[i].base_bits as f64;
                    priced_bits[i] = subtree_bits[i];
                } else {
                    subtree_bits[i] = subtree_bits[gt as usize] + subtree_bits[le as usize];
                    priced_bits[i] = kept;
                }
            }
        }
    }
    let mut new_index = vec![u32::MAX; n];
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![0u32];
    while let Some(i) = stack.pop() {
        new_index[i as usize] = order.len() as u32;
        order.push(i);
        if let MaNode::Split { gt, le, .. } = nodes[i as usize] {
            stack.push(le);
            stack.push(gt);
        }
    }
    let compact = order
        .iter()
        .map(|&i| match nodes[i as usize] {
            MaNode::Split { prop, val, gt, le } => MaNode::Split {
                prop,
                val,
                gt: new_index[gt as usize],
                le: new_index[le as usize],
            },
            leaf => leaf,
        })
        .collect();
    *nodes = compact;
    subtree_bits[0]
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
    let mut learner = new_learner(samples, params, vec![MaNode::Leaf { pred: 0, offset: 0 }]);
    let flat_bits = flat_cost(&learner, scratch);
    let mut heap = std::collections::BinaryHeap::new();
    let offset = learner.rebase(0, n, None, pool, scratch);
    let root = learner.evaluate(
        0,
        0,
        n,
        0,
        offset,
        learner.p.max_leaves as u32 > 1,
        false,
        pool,
        scratch,
    );
    learner.nodes[0] = MaNode::Leaf {
        pred: root.pred,
        offset: root.offset,
    };
    learner.info[0] = NodeInfo::of(&root);
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

    // Stable grouping by coarse leaf. Each leaf becomes one contiguous range
    // that can participate in the existing in-place growth. Rows are routed
    // in chunks on the pool's threads; the chunks come back in row order.
    let (counts, offsets, keys) = {
        const ROUTE_CHUNK_ROWS: usize = 1 << 15;
        let num_rows = samples.len();
        let num_nodes = seed.nodes.len();
        assert!(num_nodes <= usize::from(u16::MAX) + 1);
        let routed: Vec<(Vec<u16>, Vec<usize>)> = with_ma_properties!(&samples.props, |rows| {
            let seed = &seed;
            pool.steal_map(scratch, num_rows.div_ceil(ROUTE_CHUNK_ROWS), |chunk, _| {
                let start = chunk * ROUTE_CHUNK_ROWS;
                let part = &rows[start..(start + ROUTE_CHUNK_ROWS).min(num_rows)];
                let mut counts = vec![0usize; num_nodes];
                let leaves = part
                    .iter()
                    .map(|props| {
                        let (leaf, _) = seed.lookup(props);
                        counts[leaf as usize] += 1;
                        leaf as u16
                    })
                    .collect();
                (leaves, counts)
            })
        });
        let mut counts = vec![0usize; num_nodes];
        let mut keys = Vec::with_capacity(num_rows);
        for (leaves, part) in routed {
            for (count, more) in counts.iter_mut().zip(part) {
                *count += more;
            }
            keys.extend_from_slice(&leaves);
        }
        let mut offsets = vec![0usize; num_nodes + 1];
        let mut total = 0;
        for (offset, &count) in offsets.iter_mut().skip(1).zip(&counts) {
            total += count;
            *offset = total;
        }
        (counts, offsets, keys)
    };
    let mut compact = samples.into_grouped(&keys, &counts, pool);
    drop(keys);
    let mut learner = new_learner(&mut compact, params, seed.nodes);
    let flat_bits = flat_cost(&learner, scratch);
    let mut heap = std::collections::BinaryHeap::new();
    let seed_leaves: Vec<usize> = (0..learner.nodes.len())
        .filter(|&node| matches!(learner.nodes[node], MaNode::Leaf { .. }) && counts[node] != 0)
        .collect();
    let modal_offsets: Vec<ModalOffsets> = seed_leaves
        .iter()
        .map(|&node| learner.rebase(offsets[node], offsets[node + 1], None, pool, scratch))
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
                modal_offsets[k],
                learner_ref.leaves < learner_ref.p.max_leaves as u32,
                false,
                pool,
                worker_scratch,
            )
        })
    };
    for pending in pendings {
        learner.nodes[pending.node_id as usize] = MaNode::Leaf {
            pred: pending.pred,
            offset: pending.offset,
        };
        learner.info[pending.node_id as usize] = NodeInfo::of(&pending);
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
    fn in_place_grouping_matches_a_stable_sort() {
        let mut state = 0x739feb12u32;
        let pool = ThreadPool::new(3);
        for len in [0, 1, 2, 3, 31, 1023, 1024, 1025, 4096, 5000, 70_000] {
            for num_keys in [1usize, 2, 5, 63] {
                for kind in 0..4 {
                    let mut samples = MaSamples::new();
                    let mut keys = Vec::with_capacity(len);
                    for i in 0..len {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        keys.push(match kind {
                            // One key, runs, skewed and uniform keys.
                            0 => (num_keys - 1) as u16,
                            1 => ((i / 700) % num_keys) as u16,
                            2 => ((state % 97) as usize * num_keys / 4000).min(num_keys - 1) as u16,
                            _ => (state as usize % num_keys) as u16,
                        });
                        samples.push_val(
                            core::array::from_fn(|p| ((i * 101 + p) % 30_000) as i32),
                            core::array::from_fn(|p| (i * 13 + p) as u8),
                            i as i32,
                        );
                    }
                    let mut counts = vec![0usize; num_keys];
                    for &key in &keys {
                        counts[key as usize] += 1;
                    }
                    let mut indices: Vec<u32> = (0..len as u32).collect();
                    indices.sort_by_key(|&i| keys[i as usize]);
                    let expected = samples.select(&indices);
                    let props_ptr = samples.props.as_ptr();
                    let tok_ptr = samples.tok.as_ptr();
                    let actual = samples.into_grouped(&keys, &counts, &pool);
                    assert_eq!(actual.props, expected.props);
                    assert_eq!(actual.tok, expected.tok);
                    assert_eq!(actual.val, expected.val);
                    assert_eq!(actual.props.as_ptr(), props_ptr);
                    assert_eq!(actual.tok.as_ptr(), tok_ptr);
                }
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

    use crate::entropy::{pack_signed, uint_encode};

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
            props[6] = prop4;
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
                modal_offsets: false,
                index_first_depth: 0,
                prune_leaf_bits: 0.0,
                channel_first: false,
                no_wp_channels: 0,
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
                modal_offsets: false,
                index_first_depth: 0,
                prune_leaf_bits: 0.0,
                channel_first: false,
                no_wp_channels: 0,
            },
            &parallel_pool,
            &mut parallel_scratch,
        );
        assert_eq!(serial.nodes, tree.nodes);
        assert_eq!(serial.est_bits, tree.est_bits);
        assert_eq!(serial.flat_bits, tree.flat_bits);
        assert!(tree.est_bits < tree.flat_bits);
        assert!(matches!(tree.nodes[0], MaNode::Split { prop: 4, .. }));
        for i in 0..s.len() {
            let (_, pred) = tree.lookup(&s.props.get(i));
            let expected = if s.props.get(i)[4] < 32 { 1 } else { 5 };
            assert_eq!(pred, expected, "sample {i} routed to wrong predictor");
        }
    }

    #[test]
    fn prune_merges_sibling_leaves_bottom_up_and_compacts() {
        let leaf = |pred: u32| MaNode::Leaf { pred, offset: 0 };
        let split = |gt: u32, le: u32| MaNode::Split {
            prop: 2,
            val: 0,
            gt,
            le,
        };
        let info = |base_bits: f32, pred: u32, offset: i32| NodeInfo {
            base_bits,
            pred,
            offset,
        };
        // Root (200 bits as a leaf) over an inner split (100 bits) with leaf
        // children of 40 + 45 bits, and a 50-bit leaf.
        let grown = vec![split(1, 2), split(3, 4), leaf(1), leaf(2), leaf(3)];
        let infos = [
            info(200.0, 5, 0),
            info(100.0, 0, 7),
            info(50.0, 1, 0),
            info(40.0, 2, 0),
            info(45.0, 3, 0),
        ];

        // 20 bits per leaf: the inner split (40 + 45 + 20 >= 100) merges
        // into its Zero+7 leaf; the root (100 + 50 + 20 < 200) stays.
        let mut nodes = grown.clone();
        let est = prune_tree(&mut nodes, &infos, 20.0);
        assert_eq!(
            nodes,
            vec![split(1, 2), MaNode::Leaf { pred: 0, offset: 7 }, leaf(1)]
        );
        assert_eq!(est, 150.0);

        // 60 bits per leaf: the merge cascades to the root.
        let mut nodes = grown.clone();
        let est = prune_tree(&mut nodes, &infos, 60.0);
        assert_eq!(nodes, vec![leaf(5)]);
        assert_eq!(est, 200.0);

        // 10 bits per leaf: the inner split (40 + 45 + 10 < 100) stays, so
        // a root leaf of 150 bits beats the whole subtree only with both
        // splits priced (40 + 45 + 50 + 2 * 10 = 155); merging sibling
        // leaves alone would keep every node.
        let mut nodes = grown;
        let mut infos = infos;
        infos[0].base_bits = 150.0;
        let est = prune_tree(&mut nodes, &infos, 10.0);
        assert_eq!(nodes, vec![leaf(5)]);
        assert_eq!(est, 150.0);
    }
}
