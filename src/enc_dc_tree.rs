/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

//! Learned per-image DC context tree.

use crate::entropy::{Token, pack_signed, uint_encode};
use crate::static_entropy_codes::K_CONTEXT_TREE_TOKENS;

const PRED_WEIGHTED: u32 = 6;
const PRED_GRADIENT: u32 = 5;

pub(crate) const NUM_ERR_BINS: usize = 1024;
const NUM_CHANNELS: usize = 3;

/// Encoder-side property record for one DC token: `bin | (channel << 10)`.
pub(crate) type DcProp = u16;

#[inline]
pub(crate) fn dc_prop(channel: usize, err_bin: usize) -> DcProp {
    debug_assert!(channel < NUM_CHANNELS && err_bin < NUM_ERR_BINS);
    (err_bin as u16) | ((channel as u16) << 10)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LeafTag {
    /// A static-tree leaf carrying its original BFS id.
    Static(u8),
    /// A learned DC leaf, numbered in DC-subtree BFS order.
    Dc,
    /// A refinement leaf under static metadata context `ctx`, numbered in
    /// refinement-subtree walk order via `seq`.
    Refined { ctx: u8, seq: u16 },
}

#[derive(Clone)]
enum Node {
    Leaf {
        tag: LeafTag,
        pred: u32,
        offset: u32,
        mul_log: u32,
        mul_bits: u32,
    },
    Split {
        prop: u32,
        splitval: i32,
        /// "> splitval" branch (queued first in BFS).
        gt: Box<Node>,
        le: Box<Node>,
    },
}

fn unpack_signed(v: u32) -> i32 {
    if v & 1 == 0 {
        (v >> 1) as i32
    } else {
        -(((v + 1) >> 1) as i32)
    }
}

/// Parse the static blob (BFS layout, exactly like the decoder).
fn parse_static() -> Node {
    enum Flat {
        Leaf(u8, u32, u32, u32, u32),
        Split(u32, i32, usize, usize),
    }
    let mut flat: Vec<Flat> = Vec::new();
    let mut i = 0usize;
    let mut to_decode = 1usize;
    let mut leaf_id = 0u8;
    while to_decode > 0 {
        to_decode -= 1;
        let (ctx, val) = K_CONTEXT_TREE_TOKENS[i];
        debug_assert_eq!(ctx, 1);
        if val == 0 {
            flat.push(Flat::Leaf(
                leaf_id,
                K_CONTEXT_TREE_TOKENS[i + 1].1,
                K_CONTEXT_TREE_TOKENS[i + 2].1,
                K_CONTEXT_TREE_TOKENS[i + 3].1,
                K_CONTEXT_TREE_TOKENS[i + 4].1,
            ));
            leaf_id += 1;
            i += 5;
        } else {
            let gt = flat.len() + to_decode + 1;
            let le = flat.len() + to_decode + 2;
            flat.push(Flat::Split(
                val - 1,
                unpack_signed(K_CONTEXT_TREE_TOKENS[i + 1].1),
                gt,
                le,
            ));
            to_decode += 2;
            i += 2;
        }
    }
    debug_assert_eq!(i, K_CONTEXT_TREE_TOKENS.len());

    fn build(flat: &[Flat], n: usize) -> Node {
        match &flat[n] {
            Flat::Leaf(id, pred, offset, mul_log, mul_bits) => Node::Leaf {
                tag: LeafTag::Static(*id),
                pred: *pred,
                offset: *offset,
                mul_log: *mul_log,
                mul_bits: *mul_bits,
            },
            Flat::Split(prop, sv, gt, le) => Node::Split {
                prop: *prop,
                splitval: *sv,
                gt: Box::new(build(flat, *gt)),
                le: Box::new(build(flat, *le)),
            },
        }
    }
    build(&flat, 0)
}

fn serialize(root: &Node) -> (Vec<Token>, Vec<(LeafTag, u8)>, Vec<u32>) {
    let mut tokens = Vec::new();
    let mut leaves: Vec<(LeafTag, u8)> = Vec::new();
    let mut leaf_preds: Vec<u32> = Vec::new();
    let mut queue: std::collections::VecDeque<&Node> = std::collections::VecDeque::new();
    queue.push_back(root);
    let mut next_ctx = 0u8;
    while let Some(node) = queue.pop_front() {
        match node {
            Node::Leaf {
                tag,
                pred,
                offset,
                mul_log,
                mul_bits,
            } => {
                tokens.push(Token::new(1, 0));
                tokens.push(Token::new(2, *pred));
                tokens.push(Token::new(3, *offset));
                tokens.push(Token::new(4, *mul_log));
                tokens.push(Token::new(5, *mul_bits));
                leaves.push((*tag, next_ctx));
                leaf_preds.push(*pred);
                next_ctx += 1;
            }
            Node::Split {
                prop,
                splitval,
                gt,
                le,
            } => {
                tokens.push(Token::new(1, prop + 1));
                tokens.push(Token::new(0, pack_signed(*splitval)));
                queue.push_back(gt);
                queue.push_back(le);
            }
        }
    }
    (tokens, leaves, leaf_preds)
}

/// The static root splits streams: metadata on the ">" side, DC on the "<=".
fn static_sides() -> (Node, Node) {
    match parse_static() {
        Node::Split { prop, gt, le, .. } => {
            debug_assert_eq!(prop, 1, "root must split on the stream property");
            (*gt, *le)
        }
        Node::Leaf { .. } => unreachable!("static tree root is a split"),
    }
}

/// Symbol histogram width for DC residual tokens. The hybrid-uint token index
/// of any u32 stays below the 128-symbol prefix alphabet, and fine quantizers
/// (d < 0.3) genuinely reach past 48, so the full width is required.
const NUM_SYMBOLS: usize = crate::entropy::ALPHABET_SIZE;

struct CellStats {
    /// Indexed [arm][channel][bin]: symbol counts plus raw extra bits.
    counts: Vec<[u32; NUM_SYMBOLS]>,
    extra: Vec<u64>,
}

impl CellStats {
    fn new() -> Self {
        CellStats {
            counts: vec![[0u32; NUM_SYMBOLS]; 2 * NUM_CHANNELS * NUM_ERR_BINS],
            extra: vec![0u64; 2 * NUM_CHANNELS * NUM_ERR_BINS],
        }
    }
    #[inline]
    fn slot(arm: usize, ch: usize, bin: usize) -> usize {
        (arm * NUM_CHANNELS + ch) * NUM_ERR_BINS + bin
    }
    #[inline]
    fn add(&mut self, arm: usize, prop: DcProp, value: u32) {
        let (sym, nbits, _) = uint_encode(value);
        debug_assert!((sym as usize) < NUM_SYMBOLS);
        let slot = Self::slot(arm, (prop >> 10) as usize, (prop & 1023) as usize);
        self.counts[slot][sym as usize] += 1;
        self.extra[slot] += u64::from(nbits);
    }
}

/// Aggregated histogram over a (channel range × bin range) box for one arm.
struct BoxHist {
    counts: [u64; NUM_SYMBOLS],
    extra: u64,
    total: u64,
}

impl BoxHist {
    fn gather(stats: &CellStats, arm: usize, b: &LearnBox) -> Self {
        let mut counts = [0u64; NUM_SYMBOLS];
        let mut extra = 0u64;
        for ch in b.ch.clone() {
            for bin in b.bins.clone() {
                let slot = CellStats::slot(arm, ch, bin);
                for (dst, src) in counts.iter_mut().zip(&stats.counts[slot]) {
                    *dst += u64::from(*src);
                }
                extra += stats.extra[slot];
            }
        }
        let total = counts.iter().sum();
        BoxHist {
            counts,
            extra,
            total,
        }
    }
    /// Shannon cost in bits: token entropy plus raw extra bits.
    fn cost(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let total = self.total as f64;
        let entropy: f64 = self
            .counts
            .iter()
            .filter(|&&n| n != 0)
            .map(|&n| n as f64 * (total / n as f64).log2())
            .sum();
        entropy + self.extra as f64
    }
}

/// Same guard rails as the per-leaf predictor selection: a leaf only serves
/// the gradient predictor on a populated, decisive win.
const MIN_LEAF_POPULATION: u64 = 256;
const PREDICTOR_MARGIN: f64 = 0.995;

/// A split must buy at least this many bits (with margin) beyond the
/// structural cost of the extra tree node before it is taken. The margin
/// hedges against the Shannon proxy overestimating what clustering-aware
/// coding realizes — the recurring lesson of this codebase.
const SPLIT_COST_BITS: f64 = 96.0;
const SPLIT_GAIN_MARGIN: f64 = 1.3;
const MAX_DC_LEAVES: usize = 40;
const MIN_SPLIT_POPULATION: u64 = 512;

#[derive(Clone)]
struct LearnBox {
    ch: std::ops::Range<usize>,
    bins: std::ops::Range<usize>,
}

/// Leaf cost under the better predictor, honoring the margin rules.
fn leaf_cost(stats: &CellStats, b: &LearnBox) -> (f64, bool) {
    let wp = BoxHist::gather(stats, 0, b);
    let grad = BoxHist::gather(stats, 1, b);
    let (wp_cost, grad_cost) = (wp.cost(), grad.cost());
    let use_grad = wp.total >= MIN_LEAF_POPULATION && grad_cost < wp_cost * PREDICTOR_MARGIN;
    (if use_grad { grad_cost } else { wp_cost }, use_grad)
}

fn learn_subtree(stats: &CellStats, b: LearnBox, leaves_left: &mut usize) -> Node {
    let (base_cost, use_grad) = leaf_cost(stats, &b);
    let leaf = |use_grad: bool| Node::Leaf {
        tag: LeafTag::Dc,
        pred: if use_grad {
            PRED_GRADIENT
        } else {
            PRED_WEIGHTED
        },
        offset: 0,
        mul_log: 0,
        mul_bits: 0,
    };
    let population = BoxHist::gather(stats, 0, &b).total;
    if *leaves_left <= 1 || population < MIN_SPLIT_POPULATION {
        return leaf(use_grad);
    }

    let mut best: Option<(f64, u32, i32, LearnBox, LearnBox)> = None;
    let mut consider = |gain: f64, prop: u32, sv: i32, gt: LearnBox, le: LearnBox| {
        if gain > 0.0 && best.as_ref().is_none_or(|b| gain > b.0) {
            best = Some((gain, prop, sv, gt, le));
        }
    };

    // Channel splits (property 0): "> sv" keeps channels sv+1.. .
    for cut in b.ch.start..b.ch.end.saturating_sub(1) {
        let gt = LearnBox {
            ch: cut + 1..b.ch.end,
            bins: b.bins.clone(),
        };
        let le = LearnBox {
            ch: b.ch.start..cut + 1,
            bins: b.bins.clone(),
        };
        let gain = base_cost
            - leaf_cost(stats, &gt).0
            - leaf_cost(stats, &le).0
            - SPLIT_COST_BITS * SPLIT_GAIN_MARGIN;
        consider(gain, 0, cut as i32, gt, le);
    }

    // Error splits (property 15) on a population-quantile candidate grid.
    // Boundary B means "> splitval" side = bins >= B, so splitval = B - 513.
    let mut cum = vec![0u64; b.bins.len() + 1];
    for (k, bin) in b.bins.clone().enumerate() {
        let n: u64 =
            b.ch.clone()
                .map(|ch| {
                    stats.counts[CellStats::slot(0, ch, bin)]
                        .iter()
                        .map(|&c| u64::from(c))
                        .sum::<u64>()
                })
                .sum();
        cum[k + 1] = cum[k] + n;
    }
    let total = *cum.last().unwrap();
    let mut candidates: Vec<usize> = (1..=64u64)
        .map(|q| {
            let target = total * q / 65;
            b.bins.start + cum.partition_point(|&c| c <= target).saturating_sub(1)
        })
        .filter(|&boundary| boundary > b.bins.start && boundary < b.bins.end)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    for boundary in candidates {
        let gt = LearnBox {
            ch: b.ch.clone(),
            bins: boundary..b.bins.end,
        };
        let le = LearnBox {
            ch: b.ch.clone(),
            bins: b.bins.start..boundary,
        };
        let gain = base_cost
            - leaf_cost(stats, &gt).0
            - leaf_cost(stats, &le).0
            - SPLIT_COST_BITS * SPLIT_GAIN_MARGIN;
        consider(gain, 15, boundary as i32 - 513, gt, le);
    }

    match best {
        Some((_, prop, sv, gt_box, le_box)) => {
            *leaves_left -= 1;
            Node::Split {
                prop,
                splitval: sv,
                gt: Box::new(learn_subtree(stats, gt_box, leaves_left)),
                le: Box::new(learn_subtree(stats, le_box, leaves_left)),
            }
        }
        None => leaf(use_grad),
    }
}

/// DC leaves of a subtree in BFS order (their relative order in the full
/// tree's BFS numbering).
fn bfs_dc_leaves(root: &Node) -> Vec<*const Node> {
    let mut leaves = Vec::new();
    let mut queue: std::collections::VecDeque<&Node> = std::collections::VecDeque::new();
    queue.push_back(root);
    while let Some(n) = queue.pop_front() {
        match n {
            Node::Leaf {
                tag: LeafTag::Dc, ..
            } => leaves.push(n as *const Node),
            Node::Leaf { .. } => {}
            Node::Split { gt, le, .. } => {
                queue.push_back(gt);
                queue.push_back(le);
            }
        }
    }
    leaves
}

/// Everything the DC write path needs to use a learned tree.
pub(crate) struct LearnedDcTree {
    pub(crate) tokens: Vec<Token>,
    /// `(channel << 10) | bin` -> context id.
    pub(crate) dc_context: Vec<u8>,
    /// `(static metadata context << 10) | West bin` -> context id.
    pub(crate) meta_context: Vec<u8>,
    /// Context id -> serve the gradient predictor (meaningful for DC leaves).
    pub(crate) leaf_gradient: Vec<bool>,
    pub(crate) num_contexts: usize,
}

/// Learn a per-image tree from both predictor arms of the DC tokens.
/// `props[g][i]` describes token `i` of group `g`; the two arms share
/// property values because both run the same WP state.
pub(crate) fn learn_dc_tree(
    num_dc_groups: usize,
    wp_tokens: &[Vec<Token>],
    grad_tokens: &[Vec<Token>],
    props: &[Vec<DcProp>],
    meta_tokens: &[Vec<Token>],
    meta_props: &[Vec<DcProp>],
) -> LearnedDcTree {
    let mut stats = CellStats::new();
    for ((wp, grad), pr) in wp_tokens.iter().zip(grad_tokens).zip(props) {
        debug_assert_eq!(wp.len(), pr.len());
        for ((w, g), &p) in wp.iter().zip(grad).zip(pr) {
            stats.add(0, p, w.value);
            stats.add(1, p, g.value);
        }
    }

    let mut leaves_left = MAX_DC_LEAVES;
    let dc_root = learn_subtree(
        &stats,
        LearnBox {
            ch: 0..NUM_CHANNELS,
            bins: 0..NUM_ERR_BINS,
        },
        &mut leaves_left,
    );

    // Metadata refinement: per-static-context stats over the West property.
    // The pseudo-"channel" axis holds the static context, so the same cell
    // machinery serves; splits emit property 7 and never cross contexts,
    // which keeps every token's residual valid under its inherited predictor.
    let mut meta_stats = MetaStats::new();
    for (group, pr) in meta_tokens.iter().zip(meta_props) {
        debug_assert_eq!(group.len(), pr.len());
        for (t, &p) in group.iter().zip(pr) {
            meta_stats.add(t.context as usize, p, t.value);
        }
    }
    let (meta_side, _static_dc) = static_sides();
    let meta_side = refine_meta(meta_side, &meta_stats);

    let root = Node::Split {
        prop: 1,
        splitval: 1 + num_dc_groups as i32,
        gt: Box::new(meta_side),
        le: Box::new(dc_root),
    };
    let (tokens, leaves, leaf_preds) = serialize(&root);

    let (meta_root, dc_root) = match &root {
        Node::Split { gt, le, .. } => (gt.as_ref(), le.as_ref()),
        Node::Leaf { .. } => unreachable!(),
    };

    // DC context lookup: decoder-faithful walk, leaves matched through their
    // relative BFS order (identical in the full tree and the subtree alone).
    let dc_leaf_ctx: Vec<u8> = leaves
        .iter()
        .filter(|(tag, _)| *tag == LeafTag::Dc)
        .map(|&(_, ctx)| ctx)
        .collect();
    let dc_leaves = bfs_dc_leaves(dc_root);
    debug_assert_eq!(dc_leaves.len(), dc_leaf_ctx.len());
    let mut dc_context = vec![0u8; NUM_CHANNELS * NUM_ERR_BINS];
    for ch in 0..NUM_CHANNELS {
        for bin in 0..NUM_ERR_BINS {
            let err = bin as i32 - 512;
            let mut node = dc_root;
            loop {
                match node {
                    Node::Leaf { .. } => break,
                    Node::Split {
                        prop,
                        splitval,
                        gt,
                        le,
                    } => {
                        let v = if *prop == 0 { ch as i32 } else { err };
                        node = if v > *splitval { gt } else { le };
                    }
                }
            }
            let idx = dc_leaves
                .iter()
                .position(|&p| std::ptr::eq(p, node))
                .expect("resolved leaf must be a DC leaf");
            dc_context[(ch << 10) | bin] = dc_leaf_ctx[idx];
        }
    }

    // Metadata context lookup: static context -> (possibly refined) context
    // per West bin. Unrefined contexts map to a single id for every bin.
    let mut refined_ctx: std::collections::HashMap<(u8, u16), u8> =
        std::collections::HashMap::new();
    let mut static_ctx: Vec<Option<u8>> = vec![None; NUM_META_CONTEXTS];
    for &(tag, ctx) in &leaves {
        match tag {
            LeafTag::Refined { ctx: sc, seq } => {
                refined_ctx.insert((sc, seq), ctx);
            }
            LeafTag::Static(id) if (id as usize) < NUM_META_CONTEXTS => {
                static_ctx[id as usize] = Some(ctx);
            }
            _ => {}
        }
    }
    let mut meta_context = vec![0u8; NUM_META_CONTEXTS * NUM_ERR_BINS];
    let refinements = collect_refinements(meta_root);
    for sc in 0..NUM_META_CONTEXTS {
        for bin in 0..NUM_ERR_BINS {
            let ctx = match refinements.get(&(sc as u8)) {
                Some(sub) => {
                    let w = bin as i32 - 512;
                    let mut node = *sub;
                    loop {
                        match node {
                            Node::Leaf {
                                tag: LeafTag::Refined { ctx: c, seq },
                                ..
                            } => break refined_ctx[&(*c, *seq)],
                            Node::Leaf { .. } => unreachable!("refinement leaves are tagged"),
                            Node::Split {
                                splitval, gt, le, ..
                            } => {
                                node = if w > *splitval { gt } else { le };
                            }
                        }
                    }
                }
                None => static_ctx[sc].expect("unrefined static context must exist"),
            };
            meta_context[(sc << 10) | bin] = ctx;
        }
    }

    let num_contexts = leaf_preds.len();
    let mut leaf_gradient = vec![false; num_contexts];
    for (ctx, pred) in leaf_preds.iter().enumerate() {
        leaf_gradient[ctx] = *pred == PRED_GRADIENT;
    }

    LearnedDcTree {
        tokens,
        dc_context,
        meta_context,
        leaf_gradient,
        num_contexts,
    }
}

/// Map from static metadata context to its refinement subtree, if any.
fn collect_refinements(meta_root: &Node) -> std::collections::HashMap<u8, &Node> {
    // A refinement subtree replaced a static leaf, so it is the maximal
    // subtree whose leaves are all Refined with one shared context.
    fn visit<'a>(n: &'a Node, out: &mut std::collections::HashMap<u8, &'a Node>) {
        match n {
            Node::Leaf { .. } => {}
            Node::Split { gt, le, .. } => {
                for child in [gt.as_ref(), le.as_ref()] {
                    match child {
                        Node::Leaf {
                            tag: LeafTag::Refined { ctx, .. },
                            ..
                        } => {
                            // Single-leaf refinement cannot occur (refine_meta
                            // keeps the plain static leaf then), but guard.
                            out.insert(*ctx, child);
                        }
                        Node::Split { .. } if subtree_refined_ctx(child).is_some() => {
                            out.insert(subtree_refined_ctx(child).unwrap(), child);
                        }
                        _ => visit(child, out),
                    }
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    // The meta root itself could be a refinement only if the whole meta side
    // were one leaf, which the static tree rules out.
    visit(meta_root, &mut out);
    out
}

/// If every leaf under `n` is Refined with the same static context, return it.
fn subtree_refined_ctx(n: &Node) -> Option<u8> {
    match n {
        Node::Leaf {
            tag: LeafTag::Refined { ctx, .. },
            ..
        } => Some(*ctx),
        Node::Leaf { .. } => None,
        Node::Split { gt, le, .. } => {
            let a = subtree_refined_ctx(gt)?;
            let b = subtree_refined_ctx(le)?;
            (a == b).then_some(a)
        }
    }
}

/// Number of static metadata contexts (leaves 0..=10).
pub(crate) const NUM_META_CONTEXTS: usize = 11;

/// Single-arm stats for metadata refinement, keyed by (static context, bin).
struct MetaStats {
    counts: Vec<[u32; NUM_SYMBOLS]>,
    extra: Vec<u64>,
}

impl MetaStats {
    fn new() -> Self {
        MetaStats {
            counts: vec![[0u32; NUM_SYMBOLS]; NUM_META_CONTEXTS * NUM_ERR_BINS],
            extra: vec![0u64; NUM_META_CONTEXTS * NUM_ERR_BINS],
        }
    }
    #[inline]
    fn add(&mut self, static_ctx: usize, prop: DcProp, value: u32) {
        debug_assert!(static_ctx < NUM_META_CONTEXTS);
        let (sym, nbits, _) = uint_encode(value);
        let slot = static_ctx * NUM_ERR_BINS + (prop & 1023) as usize;
        self.counts[slot][sym as usize] += 1;
        self.extra[slot] += u64::from(nbits);
    }
    fn range_hist(&self, static_ctx: usize, bins: std::ops::Range<usize>) -> BoxHist {
        let mut counts = [0u64; NUM_SYMBOLS];
        let mut extra = 0u64;
        for bin in bins {
            let slot = static_ctx * NUM_ERR_BINS + bin;
            for (dst, src) in counts.iter_mut().zip(&self.counts[slot]) {
                *dst += u64::from(*src);
            }
            extra += self.extra[slot];
        }
        let total = counts.iter().sum();
        BoxHist {
            counts,
            extra,
            total,
        }
    }
}

/// Cap on extra contexts the metadata refinement may add.
const MAX_META_EXTRA_LEAVES: usize = 12;

/// Replace static metadata leaves with learned West-property refinements
/// where the split gain clears the same bar as the DC learner.
fn refine_meta(meta_side: Node, stats: &MetaStats) -> Node {
    let mut extra_leaves = MAX_META_EXTRA_LEAVES;
    refine_meta_node(meta_side, stats, &mut extra_leaves)
}

fn refine_meta_node(node: Node, stats: &MetaStats, extra: &mut usize) -> Node {
    match node {
        Node::Split {
            prop,
            splitval,
            gt,
            le,
        } => Node::Split {
            prop,
            splitval,
            gt: Box::new(refine_meta_node(*gt, stats, extra)),
            le: Box::new(refine_meta_node(*le, stats, extra)),
        },
        Node::Leaf {
            tag: LeafTag::Static(id),
            pred,
            offset,
            mul_log,
            mul_bits,
        } if (id as usize) < NUM_META_CONTEXTS => {
            let mut seq = 0u16;
            let sub = refine_leaf(
                stats,
                id,
                (pred, offset, mul_log, mul_bits),
                0..NUM_ERR_BINS,
                extra,
                &mut seq,
            );
            match sub {
                // No split found: keep the plain static leaf so unrefined
                // contexts stay recognizable.
                Node::Leaf { .. } => Node::Leaf {
                    tag: LeafTag::Static(id),
                    pred,
                    offset,
                    mul_log,
                    mul_bits,
                },
                refined => refined,
            }
        }
        leaf => leaf,
    }
}

fn refine_leaf(
    stats: &MetaStats,
    static_ctx: u8,
    template: (u32, u32, u32, u32),
    bins: std::ops::Range<usize>,
    extra: &mut usize,
    seq: &mut u16,
) -> Node {
    let (pred, offset, mul_log, mul_bits) = template;
    let make_leaf = |seq: &mut u16| {
        let leaf = Node::Leaf {
            tag: LeafTag::Refined {
                ctx: static_ctx,
                seq: *seq,
            },
            pred,
            offset,
            mul_log,
            mul_bits,
        };
        *seq += 1;
        leaf
    };
    let base = stats.range_hist(static_ctx as usize, bins.clone());
    if *extra == 0 || base.total < MIN_SPLIT_POPULATION {
        return make_leaf(seq);
    }

    // Population-quantile candidate boundaries over the West bins.
    let mut cum = vec![0u64; bins.len() + 1];
    for (k, bin) in bins.clone().enumerate() {
        let slot = static_ctx as usize * NUM_ERR_BINS + bin;
        let n: u64 = stats.counts[slot].iter().map(|&c| u64::from(c)).sum();
        cum[k + 1] = cum[k] + n;
    }
    let total = *cum.last().unwrap();
    let mut candidates: Vec<usize> = (1..=32u64)
        .map(|q| {
            let target = total * q / 33;
            bins.start + cum.partition_point(|&c| c <= target).saturating_sub(1)
        })
        .filter(|&b| b > bins.start && b < bins.end)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();

    let base_cost = base.cost();
    let mut best: Option<(f64, usize)> = None;
    for boundary in candidates {
        let gain = base_cost
            - stats
                .range_hist(static_ctx as usize, bins.start..boundary)
                .cost()
            - stats
                .range_hist(static_ctx as usize, boundary..bins.end)
                .cost()
            - SPLIT_COST_BITS * SPLIT_GAIN_MARGIN;
        if gain > 0.0 && best.is_none_or(|(g, _)| gain > g) {
            best = Some((gain, boundary));
        }
    }
    match best {
        Some((_, boundary)) => {
            *extra -= 1;
            Node::Split {
                prop: 7,
                splitval: boundary as i32 - 513,
                gt: Box::new(refine_leaf(
                    stats,
                    static_ctx,
                    template,
                    boundary..bins.end,
                    extra,
                    seq,
                )),
                le: Box::new(refine_leaf(
                    stats,
                    static_ctx,
                    template,
                    bins.start..boundary,
                    extra,
                    seq,
                )),
            }
        }
        None => make_leaf(seq),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_entropy_codes::K_GRADIENT_CONTEXT_LUT;

    /// Parser and serializer must reproduce the static blob token-exactly;
    /// this pins every convention (BFS order, child order, token grammar).
    #[test]
    fn static_tree_round_trips_exactly() {
        let root = parse_static();
        let (tokens, leaves, _preds) = serialize(&root);
        assert_eq!(tokens.len(), K_CONTEXT_TREE_TOKENS.len());
        for (i, (t, &(ctx, val))) in tokens.iter().zip(K_CONTEXT_TREE_TOKENS.iter()).enumerate() {
            assert_eq!((t.context, t.value), (ctx, val), "token {i} diverged");
        }
        // Static leaves keep their identity numbering in a pure round trip.
        for (tag, ctx) in &leaves {
            assert_eq!(*tag, LeafTag::Static(*ctx));
        }
    }

    /// Walking the static DC side must reproduce K_GRADIENT_CONTEXT_LUT for
    /// every channel — the invariant shipped bitstreams rely on.
    #[test]
    fn static_dc_side_walk_matches_lut() {
        let (_meta, dc) = static_sides();
        for bin in 0..NUM_ERR_BINS {
            let err = bin as i32 - 512;
            for ch in 0..3i32 {
                let mut node = &dc;
                loop {
                    match node {
                        Node::Leaf { tag, .. } => {
                            assert_eq!(
                                *tag,
                                LeafTag::Static(K_GRADIENT_CONTEXT_LUT[bin]),
                                "bin {bin} channel {ch}"
                            );
                            break;
                        }
                        Node::Split {
                            prop,
                            splitval,
                            gt,
                            le,
                        } => {
                            // Property 9 is rewritten to 15 at write time;
                            // the walk semantics are identical.
                            let v = match *prop {
                                9 | 15 => err,
                                0 => ch,
                                _ => unreachable!("unexpected property {prop} in DC side"),
                            };
                            node = if v > *splitval { gt } else { le };
                        }
                    }
                }
            }
        }
    }

    /// A learned tree's context lookup must agree with a decoder-faithful
    /// walk of its serialized tokens, including the metadata renumbering.
    #[test]
    fn learned_tree_lookup_matches_decoded_walk() {
        // Synthetic stats: channel 1 (X) benefits from its own context, and
        // large errors prefer the gradient arm.
        let mut wp = vec![Vec::new()];
        let mut grad = vec![Vec::new()];
        let mut props = vec![Vec::new()];
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for i in 0..60_000u32 {
            let ch = (i % 3) as usize;
            let bin = 200 + (rand() % 600) as usize;
            props[0].push(dc_prop(ch, bin));
            let magnitude = if ch == 1 { 3 } else { 200 + rand() % 64 };
            wp[0].push(Token::new(0, magnitude));
            grad[0].push(Token::new(0, magnitude / 2));
        }
        // Metadata: QF-like tokens in static context 3 whose residual scale
        // depends strongly on West, so refinement must split it.
        let mut meta = vec![Vec::new()];
        let mut meta_props = vec![Vec::new()];
        for i in 0..40_000u32 {
            let w = if i % 2 == 0 { 2 } else { 40 };
            meta_props[0].push(dc_prop(0, (512 + w) as usize));
            let value = if w == 2 { 1 } else { 300 + (i % 32) };
            meta[0].push(Token::new(3, value));
        }
        let learned = learn_dc_tree(4, &wp, &grad, &props, &meta, &meta_props);

        // Decode the serialized tokens with an independent BFS parser and
        // walk it for a sample of (channel, err) points.
        let toks = &learned.tokens;
        let mut flat: Vec<(i64, i64, usize, usize, u32)> = Vec::new(); // (prop, sv, gt, le, pred); prop -1 = leaf
        let mut i = 0usize;
        let mut to_decode = 1usize;
        while to_decode > 0 {
            to_decode -= 1;
            assert_eq!(toks[i].context, 1);
            if toks[i].value == 0 {
                flat.push((-1, 0, 0, 0, toks[i + 1].value));
                i += 5;
            } else {
                let gt = flat.len() + to_decode + 1;
                let le = flat.len() + to_decode + 2;
                flat.push((
                    i64::from(toks[i].value) - 1,
                    i64::from(unpack_signed(toks[i + 1].value)),
                    gt,
                    le,
                    0,
                ));
                to_decode += 2;
                i += 2;
            }
        }
        assert_eq!(i, toks.len());

        let walk = |props: &dyn Fn(i64) -> i64| -> usize {
            let mut n = 0usize;
            loop {
                let (prop, sv, gt, le, _) = flat[n];
                if prop == -1 {
                    return flat[..n].iter().filter(|f| f.0 == -1).count();
                }
                n = if props(prop) > sv { gt } else { le };
            }
        };

        for ch in 0..3i64 {
            for bin in (0..NUM_ERR_BINS).step_by(17) {
                let err = bin as i64 - 512;
                // A DC-group stream id (<= 1 + num_dc_groups).
                let ctx = walk(&|p| match p {
                    0 => ch,
                    1 => 1,
                    15 => err,
                    _ => 0,
                });
                assert_eq!(
                    ctx,
                    learned.dc_context[((ch as usize) << 10) | bin] as usize,
                    "channel {ch} bin {bin}"
                );
            }
        }
        // Metadata streams (id > 1 + num_dc_groups) must land on remapped
        // metadata contexts.
        let meta_ctx = walk(&|p| match p {
            0 => 5,
            1 => 100,
            9 | 15 => 1,
            _ => 1,
        });
        let meta_ctxs: std::collections::HashSet<u8> =
            learned.meta_context.iter().copied().collect();
        assert!(
            meta_ctxs.contains(&(meta_ctx as u8)),
            "metadata walk landed outside the remapped metadata contexts"
        );
        // The synthetic QF stream must have been split on West: the two West
        // populations land in different contexts.
        let ctx_low = learned.meta_context[(3usize << 10) | (512 + 2)];
        let ctx_high = learned.meta_context[(3usize << 10) | (512 + 40)];
        assert_ne!(ctx_low, ctx_high, "refinement should split context 3");
        // The synthetic X channel must have earned its own context somewhere.
        let x_ctx = learned.dc_context[(1usize << 10) | 512];
        let y_ctx = learned.dc_context[512];
        assert_ne!(x_ctx, y_ctx, "channel split should separate X from Y");
    }
}
