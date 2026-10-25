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

//! DC predictors, token collection, context trees and rate estimates.

use crate::bit_writer::BitWriter;
use crate::coder_scratch::DcPredictorScratch;
use crate::dc_group_data::DcGroupData;
use crate::entropy::{
    Token, f_log2, optimize_entropy_code, pack_signed, write_entropy_code, write_token,
};
use crate::static_entropy_codes::{
    K_CONTEXT_TREE_TOKENS, K_GRADIENT_CONTEXT_LUT, K_NUM_DC_CONTEXTS,
};

const K_NUM_TREE_CONTEXTS: usize = 6;

pub(super) const K_GRAD_RANGE_MID: i64 = 512;
pub(super) const K_GRAD_RANGE_MIN: i64 = 0;
pub(super) const K_GRAD_RANGE_MAX: i64 = 1023;

#[inline]
pub(super) fn clamped_gradient(n: i32, w: i32, l: i32) -> i32 {
    let mn = n.min(w);
    let mx = n.max(w);
    let g = (n as i64 + w as i64 - l as i64) as i32;
    g.clamp(mn, mx)
}

#[inline(always)]
fn push_dc_wp_token(
    tokens: &mut Vec<Token>,
    gradient_tokens: &mut Vec<Token>,
    collect_gradient: bool,
    wp: &mut crate::weighted_predictor::WpState,
    x: usize,
    y: usize,
    value: i64,
    n: i64,
    w: i64,
    ne: i64,
    nw: i64,
    nn: i64,
    dc_gradient: &DcPredictorChoice,
    mchan: usize,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
) {
    let wp_prediction = wp.predict(x, y, n, w, ne, nw, nn);
    let prop = (K_GRAD_RANGE_MID + wp.wp_prop).clamp(K_GRAD_RANGE_MIN, K_GRAD_RANGE_MAX) as usize;
    if collect_props {
        props.push(crate::dc_tree::dc_prop(mchan, prop));
    }
    wp.update(value, x, y);
    // The context comes from the weighted predictor's error whichever predictor
    // the leaf ends up using, so the tree is navigated identically either way
    // and the per-leaf choice below costs nothing to signal.
    let context = K_GRADIENT_CONTEXT_LUT[prop] as usize;
    let prediction = if dc_gradient[context] {
        (n + w - nw).clamp(n.min(w), n.max(w))
    } else {
        wp_prediction
    };
    tokens.push(Token::new(
        context as u32,
        pack_signed((value - prediction) as i32),
    ));
    if collect_gradient {
        let gradient = (n + w - nw).clamp(n.min(w), n.max(w));
        gradient_tokens.push(Token::new(
            context as u32,
            pack_signed((value - gradient) as i32),
        ));
    }
}

/// Per-leaf predictor selection for the DC plane, indexed by DC context.
pub(super) type DcPredictorChoice = [bool; K_NUM_DC_CONTEXTS];

pub(crate) const DC_PREDICTOR_WEIGHTED: DcPredictorChoice = [false; K_NUM_DC_CONTEXTS];
pub(super) const DC_PREDICTOR_GRADIENT: DcPredictorChoice = [true; K_NUM_DC_CONTEXTS];

/// Pick, for every DC leaf independently, whichever predictor codes its own
/// tokens in fewer bits. Both candidate streams share a context assignment, so
/// `wp[i]` and `grad[i]` differ only in value and the merged stream is a
/// per-token pick — no third tokenization pass.
pub(super) fn choose_dc_predictors(
    wp: &[Vec<Token>],
    grad: &[Vec<Token>],
    scratch: &mut DcPredictorScratch,
) -> DcPredictorChoice {
    // [candidate][context][symbol] counts, plus the raw extra bits per context.
    let counts = &mut scratch.counts;
    let extra = &mut scratch.extra;
    // A flat integer fill avoids expanding the reset into row-sized stores.
    counts.as_flattened_mut().fill(0);
    extra.fill(0);
    for (cand, groups) in [wp, grad].into_iter().enumerate() {
        for token in groups.iter().flatten() {
            let (sym, nbits, _) = crate::entropy::uint_encode(token.value);
            let slot = cand * K_NUM_DC_CONTEXTS + token.context as usize;
            counts[slot][sym as usize] += 1;
            extra[slot] += nbits as u64;
        }
    }
    choose_dc_predictors_from_counts(scratch)
}

pub(super) fn choose_dc_predictors_from_counts(scratch: &DcPredictorScratch) -> DcPredictorChoice {
    let counts = &scratch.counts;
    let extra = &scratch.extra;
    let cost = |slot: usize| -> f64 {
        let total: u32 = counts[slot].iter().sum();
        if total == 0 {
            return 0.0;
        }
        let total = f64::from(total);
        let entropy: f64 = counts[slot]
            .iter()
            .filter(|&&n| n != 0)
            .map(|&n| f64::from(n) * f_log2(total / f64::from(n)))
            .sum();
        entropy + extra[slot] as f64
    };
    let mut choice = DC_PREDICTOR_WEIGHTED;
    for (ctx, out) in choice.iter_mut().enumerate() {
        // The per-context Shannon cost ignores the clustering the real code
        // applies afterward, and leaves with few tokens are exactly the ones
        // clustering folds into a neighbor. Require a population and a margin
        // so a leaf only flips when the saving survives that noise.
        let (wp_cost, grad_cost) = (cost(ctx), cost(K_NUM_DC_CONTEXTS + ctx));
        let populated = counts[ctx].iter().sum::<u32>() >= MIN_TOKENS_PER_DC_LEAF;
        *out = populated && grad_cost < wp_cost * DC_PREDICTOR_MARGIN;
    }
    choice
}

/// A DC leaf must own at least this many tokens before its predictor may flip.
const MIN_TOKENS_PER_DC_LEAF: u32 = 256;

/// Required saving before flipping a leaf away from the weighted predictor.
const DC_PREDICTOR_MARGIN: f64 = 0.995;

#[cfg(test)]
fn collect_dc_tokens(
    dc_data: &DcGroupData,
    dc_gradient: &DcPredictorChoice,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
) -> Vec<Token> {
    collect_dc_tokens_with_gradient(dc_data, dc_gradient, props, collect_props, false).0
}

pub(crate) fn collect_dc_tokens_with_gradient(
    dc_data: &DcGroupData,
    dc_gradient: &DcPredictorChoice,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
    collect_gradient: bool,
) -> (Vec<Token>, Vec<Token>) {
    let token_count = [1usize, 0, 2]
        .into_iter()
        .map(|c| dc_data.quant_dc.plane(c).as_slice().len())
        .sum();
    let mut tokens = Vec::with_capacity(token_count);
    let mut gradient_tokens = Vec::with_capacity(if collect_gradient { token_count } else { 0 });
    for (mchan, c) in [1usize, 0, 2].into_iter().enumerate() {
        let plane = dc_data.quant_dc.plane(c);
        walk_dc_channel(
            plane.as_slice(),
            plane.xsize(),
            plane.ysize(),
            plane.xsize(),
            mchan,
            dc_gradient,
            props,
            collect_props,
            collect_gradient,
            &mut tokens,
            &mut gradient_tokens,
        );
    }
    (tokens, gradient_tokens)
}

/// Tokenize one DC channel: a `xsize`×`ysize` window of rows `stride` apart.
///
/// Weighted-predictor DC, mirroring libjxl's kWPFixedDC path (enc_modular.cc
/// AddVarDCTDC at speed tiers falcon..squirrel): the guess is the
/// self-correcting WP and the context is the WP-error property (modular
/// property 15) bucketed by the same +-500 cutoffs the gradient tree used,
/// so K_GRADIENT_CONTEXT_LUT applies unchanged (write_context_tree emits the
/// matching tree: identical structure, property 15, Weighted leaves). The WP
/// state machine and border conventions are the bit-faithful lossless-path
/// ones, so encoder residuals match the reference decoder exactly.
///
/// NOTE: this walk is shared by the DC coder and by [`price_dc_plane`], which
/// prices the candidate planes of `choose_chroma_dc_coding`. Keep it the
/// single source of the DC tokenization: a change here reaches both, but a
/// change to how DC tokens are entropy-coded elsewhere (contexts, learned
/// trees, token packing) must be mirrored in `price_dc_plane`, or the
/// squeeze decision prices a coder that no longer exists.
#[allow(clippy::too_many_arguments)]
pub(super) fn walk_dc_channel<T: Copy + Into<i64>>(
    data: &[T],
    xsize: usize,
    ysize: usize,
    stride: usize,
    mchan: usize,
    dc_gradient: &DcPredictorChoice,
    props: &mut Vec<crate::dc_tree::DcProp>,
    collect_props: bool,
    collect_gradient: bool,
    tokens: &mut Vec<Token>,
    gradient_tokens: &mut Vec<Token>,
) {
    if xsize == 0 || ysize == 0 {
        return;
    }
    let row = |y: usize| &data[y * stride..y * stride + xsize];
    let mut wp = crate::weighted_predictor::WpState::new(xsize);

    // On the top row every unavailable north-side neighbor collapses to
    // the value on the left.
    let first_row = row(0);
    let mut left = 0i64;
    for (x, &value) in first_row.iter().enumerate() {
        let value: i64 = value.into();
        push_dc_wp_token(
            tokens,
            gradient_tokens,
            collect_gradient,
            &mut wp,
            x,
            0,
            value,
            left,
            left,
            left,
            left,
            left,
            dc_gradient,
            mchan,
            props,
            collect_props,
        );
        left = value;
    }

    let mut row_above = first_row;
    let mut row_above2 = first_row;
    for y in 1..ysize {
        let current_row = row(y);

        // First column: W and NW replicate N.
        let n: i64 = row_above[0].into();
        let ne: i64 = row_above.get(1).copied().unwrap_or(row_above[0]).into();
        push_dc_wp_token(
            tokens,
            gradient_tokens,
            collect_gradient,
            &mut wp,
            0,
            y,
            current_row[0].into(),
            n,
            n,
            ne,
            n,
            row_above2[0].into(),
            dc_gradient,
            mchan,
            props,
            collect_props,
        );

        // Interior columns have every neighbor available.
        let current_pairs = current_row.array_windows::<2>();
        let north_triplets = row_above.array_windows::<3>();
        for (offset, ((current, north), &nn)) in current_pairs
            .zip(north_triplets)
            .zip(row_above2.iter().skip(1))
            .enumerate()
        {
            push_dc_wp_token(
                tokens,
                gradient_tokens,
                collect_gradient,
                &mut wp,
                offset + 1,
                y,
                current[1].into(),
                north[1].into(),
                current[0].into(),
                north[2].into(),
                north[0].into(),
                nn.into(),
                dc_gradient,
                mchan,
                props,
                collect_props,
            );
        }

        // Last column: NE replicates N. Width one was handled above.
        if xsize > 1 {
            let x = xsize - 1;
            let n: i64 = row_above[x].into();
            push_dc_wp_token(
                tokens,
                gradient_tokens,
                collect_gradient,
                &mut wp,
                x,
                y,
                current_row[x].into(),
                n,
                current_row[x - 1].into(),
                n,
                row_above[x - 1].into(),
                row_above2[x].into(),
                dc_gradient,
                mchan,
                props,
                collect_props,
            );
        }

        row_above2 = row_above;
        row_above = current_row;
    }
}

/// Bits to code `levels` — a `width`×`height` window of a DC plane whose rows
/// are `stride` apart — the way the DC coder codes it: the weighted
/// predictor's residuals in their WP-error contexts, priced as per-context
/// order-0 entropy of the hybrid-uint symbols plus their raw bits. This is
/// the price `choose_chroma_dc_coding` compares its candidate planes with.
///
/// NOTE: keep in step with the DC coder (see [`walk_dc_channel`]).
pub(crate) fn price_dc_plane(levels: &[i32], width: usize, height: usize, stride: usize) -> f64 {
    let mut price = DcPlanePrice::default();
    price.add(levels, width, height, stride);
    price.bits()
}

/// Reusable weighted-predictor price for one channel across DC groups. The
/// predictor restarts for every group, while all groups share the histograms,
/// just as they share the final frame's entropy code.
pub(crate) struct DcPlanePrice {
    pub(super) tokens: Vec<Token>,
    pub(super) counts: Vec<[u32; crate::entropy::ALPHABET_SIZE]>,
    pub(super) raw_bits: u64,
}

impl Default for DcPlanePrice {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            counts: vec![[0; crate::entropy::ALPHABET_SIZE]; K_NUM_DC_CONTEXTS],
            raw_bits: 0,
        }
    }
}

impl DcPlanePrice {
    pub(crate) fn clear(&mut self) {
        self.counts.as_flattened_mut().fill(0);
        self.raw_bits = 0;
    }

    pub(crate) fn add<T: Copy + Into<i64>>(
        &mut self,
        levels: &[T],
        width: usize,
        height: usize,
        stride: usize,
    ) {
        self.tokens.clear();
        walk_dc_channel(
            levels,
            width,
            height,
            stride,
            0,
            &DC_PREDICTOR_WEIGHTED,
            &mut Vec::new(),
            false,
            false,
            &mut self.tokens,
            &mut Vec::new(),
        );
        for token in &self.tokens {
            let (sym, nbits, _) = crate::entropy::uint_encode(token.value);
            self.counts[token.context as usize][sym as usize] += 1;
            self.raw_bits += u64::from(nbits);
        }
    }

    /// Count a token from an existing weighted-predictor walk.
    pub(super) fn add_token(&mut self, token: Token) {
        let (sym, nbits, _) = crate::entropy::uint_encode(token.value);
        self.counts[token.context as usize][sym as usize] += 1;
        self.raw_bits += u64::from(nbits);
    }

    pub(crate) fn bits(&self) -> f64 {
        let mut bits = self.raw_bits as f64;
        for ctx in &self.counts {
            let total: u32 = ctx.iter().sum();
            if total == 0 {
                continue;
            }
            let total = f64::from(total);
            bits += ctx
                .iter()
                .filter(|&&n| n != 0)
                .map(|&n| f64::from(n) * f_log2(total / f64::from(n)))
                .sum::<f64>();
        }
        bits
    }
}

/// Build and emit the context tree.
pub(crate) fn write_context_tree(
    num_dc_groups: usize,
    dc_gradient: &DcPredictorChoice,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    writer: &mut BitWriter,
) {
    let tokens = context_tree_tokens(num_dc_groups, dc_gradient);
    write_tree_tokens(&tokens, huffman_pool, writer);
}

/// The static context tree's tokens.
fn context_tree_tokens(num_dc_groups: usize, dc_gradient: &DcPredictorChoice) -> Vec<Token> {
    // Build tokens with the patched value at index 1.
    let mut tokens: Vec<Token> = Vec::with_capacity(K_CONTEXT_TREE_TOKENS.len());
    for (i, &(ctx, val)) in K_CONTEXT_TREE_TOKENS.iter().enumerate() {
        let v = if i == 1 {
            pack_signed(1 + num_dc_groups as i32)
        } else {
            val
        };
        tokens.push(Token::new(ctx, v));
    }
    // Retarget the DC-image region (leaves 11..=44) at the Weighted predictor,
    // mirroring libjxl's kWPFixedDC tree (enc_modular.cc / enc_ma PredefinedTree):
    // identical 33 cutoffs, but splitting on the WP-error property (15) instead
    // of the gradient property (9), with Weighted(6) leaves instead of
    // Gradient(5). Parse-aware transform of the static blob: prop-9 splits only
    // occur in the DC region; Gradient leaves also exist at meta contexts 9/10
    // (CfL), so predictor tokens are rewritten only for leaf indices >= 11.
    {
        let mut i = 0usize;
        let mut leaf_idx = 0usize;
        while i < tokens.len() {
            debug_assert_eq!(tokens[i].context, 1);
            if tokens[i].value == 0 {
                if leaf_idx >= 11 && tokens[i + 1].value == 5 && !dc_gradient[leaf_idx] {
                    tokens[i + 1] = Token::new(2, 6); // Gradient -> Weighted
                }
                leaf_idx += 1;
                i += 5;
            } else {
                if tokens[i].value == 9 + 1 {
                    tokens[i] = Token::new(1, 15 + 1); // gradient prop -> WP prop
                }
                i += 2; // split: PROPERTY, SPLITVAL
            }
        }
    }
    tokens
}

/// Serialize a context tree's token stream (its own entropy code included).
pub(crate) fn write_tree_tokens(
    tokens: &[Token],
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    writer: &mut BitWriter,
) {
    // OptimizeEntropyCode clusters the K_NUM_TREE_CONTEXTS=6 contexts.
    let code = optimize_entropy_code(tokens, K_NUM_TREE_CONTEXTS, huffman_pool);
    let code_ref = code.as_ref();

    writer.write(1, 1); // not an empty tree
    writer.write(1, 0); // no lz77
    write_entropy_code(&code_ref, huffman_pool, writer);
    for t in tokens {
        write_token(*t, &code_ref, writer);
    }
}

/// Which context tree the DC bundle is written with.
pub(super) enum DcTreeChoice {
    /// The static blob with per-leaf predictor flips.
    Static(DcPredictorChoice),
    /// A per-image learned tree, already serialized to tokens.
    Learned(Vec<Token>),
}

impl DcTreeChoice {
    pub(super) fn tokens(&self, num_dc_groups: usize) -> Vec<Token> {
        match self {
            Self::Static(grad) => context_tree_tokens(num_dc_groups, grad),
            Self::Learned(tokens) => tokens.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one group of `n` tokens in `context`, all carrying `value`.
    fn leaf_tokens(context: u32, value: u32, n: usize) -> Vec<Vec<Token>> {
        vec![(0..n).map(|_| Token::new(context, value)).collect()]
    }

    #[test]
    fn dc_plane_price_pools_histograms_and_restarts_prediction() {
        let mut price = super::DcPlanePrice::default();
        price.add(&[0i16], 1, 1, 1);
        price.add(&[1i16], 1, 1, 1);
        // Two equally frequent symbols in one shared context cost two bits;
        // pricing the one-symbol groups separately would incorrectly cost zero.
        assert_eq!(price.bits(), 2.0);

        price.clear();
        assert_eq!(price.bits(), 0.0);
        price.add(&[100i16], 1, 1, 1);
        price.add(&[100i16], 1, 1, 1);
        // Each group's first sample predicts zero, so both tokens are identical.
        // Treating them as one row would instead code residuals 100 and zero.
        assert_eq!(price.bits(), 2.0 * super::price_dc_plane(&[100], 1, 1, 1));
        assert!(super::price_dc_plane(&[100, 100], 2, 1, 2) < price.bits());
    }

    #[test]
    fn dc_predictor_flips_only_on_a_decisive_populated_win() {
        let n = MIN_TOKENS_PER_DC_LEAF as usize;
        let ctx = 11;
        let mut scratch = DcPredictorScratch::default();

        // Gradient codes every residual as zero; the weighted predictor misses
        // by a wide, multi-extra-bit margin.
        let decisive = choose_dc_predictors(
            &leaf_tokens(ctx, 4000, n),
            &leaf_tokens(ctx, 0, n),
            &mut scratch,
        );
        assert!(decisive[ctx as usize], "a decisive gradient win must flip");

        // Identical streams: the margin must keep the weighted default.
        let tied = choose_dc_predictors(
            &leaf_tokens(ctx, 7, n),
            &leaf_tokens(ctx, 7, n),
            &mut scratch,
        );
        assert!(!tied[ctx as usize], "a tie must not flip");

        // Same decisive win, but below the population floor.
        let sparse = choose_dc_predictors(
            &leaf_tokens(ctx, 4000, n - 1),
            &leaf_tokens(ctx, 0, n - 1),
            &mut scratch,
        );
        assert!(
            !sparse[ctx as usize],
            "an underpopulated leaf must not flip"
        );
    }
}

#[cfg(test)]
mod fused_dc_tests {
    use super::*;
    #[test]
    fn fused_dc_candidates_match_separate_predictor_traversals() {
        for w in [1, 2, 3, 9, 32] {
            for h in [1, 2, 3, 17] {
                let mut dc = DcGroupData::new(w, h).unwrap();
                for c in 0..3 {
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x + 1) * (c + 7);
                            dc.quant_dc.plane_row_mut(c, y)[x] =
                                (((i * 7919) ^ (i / 7 * 1237)) % 65536) as i16;
                        }
                    }
                }
                for collect_props in [false, true] {
                    for choice in [
                        DC_PREDICTOR_WEIGHTED,
                        DC_PREDICTOR_GRADIENT,
                        std::array::from_fn(|i| i % 2 == 0),
                    ] {
                        let mut expected_props = Vec::new();
                        let expected =
                            collect_dc_tokens(&dc, &choice, &mut expected_props, collect_props);
                        let grad =
                            collect_dc_tokens(&dc, &DC_PREDICTOR_GRADIENT, &mut Vec::new(), false);
                        let mut props = Vec::new();
                        let (fused, fused_grad) = collect_dc_tokens_with_gradient(
                            &dc,
                            &choice,
                            &mut props,
                            collect_props,
                            true,
                        );
                        let values = |ts: Vec<Token>| {
                            ts.into_iter()
                                .map(|t| (t.context, t.value))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(values(fused), values(expected));
                        assert_eq!(values(fused_grad), values(grad));
                        assert_eq!(props, expected_props);
                    }
                }
            }
        }
    }
}
