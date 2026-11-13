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

//! Global learned trees for modular planes.

use super::super::lz77::LZ77_MIN_SYMBOL;
use super::super::ma_multiplier::{
    SAFE_PREDICTORS, emit_forest_bfs, scale_ma_value_props, tokenize_channel_ma_mult,
};
use super::super::ma_sampling::sample_channel_ma;
use super::super::ma_source::{MA_MIN_NODE_SAMPLES, MA_PHOTO_CANDIDATES};
use super::super::predictor::{PREDICTOR_GRADIENT, clamped_gradient};
use crate::coder_scratch::CoderScratch;
use crate::entropy::{Token, pack_signed};
use crate::ma_tree::{
    LearnedTree, MaLearnParams, MaNode, MaSamples, NUM_MA_PREDS, NUM_MA_PROPS, PRED_WEIGHTED,
    learn_ma_tree_in_place,
};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// One plane's subtree for the frame's global tree, learned over every crop
/// of the plane (their stream ids feed property 1).
pub(crate) struct GlobalPlaneTree {
    /// BFS tree tokens, leaves stamped with the plane's common factor.
    pub(crate) tree_tokens: Vec<Token>,
    /// Per crop, its pixel tokens (contexts are the subtree's leaves).
    pub(crate) crop_tokens: Vec<Vec<Token>>,
}

/// One rectangle of a plane in its own stream: x0, y0, width, height and the
/// decoder's stream id.
pub(crate) type StreamCrop = (usize, usize, usize, usize, i32);

/// Sampling and size budget of a learned plane tree: about `samples`
/// sampled pixels, at most `leaves` leaves. `gradient_only` keeps every
/// leaf on the Gradient predictor and every split on the gradient property
/// (W + N - NW) or the static ones, which libjxl decodes through its
/// lookup-table fast path instead of the generic tree walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneTreeBudget {
    pub(crate) samples: usize,
    pub(crate) leaves: usize,
    pub(crate) gradient_only: bool,
}

impl PlaneTreeBudget {
    /// The slow tiers' full tree.
    pub(crate) const FULL: Self = Self {
        samples: 1 << 20,
        leaves: 1024,
        gradient_only: false,
    };
}

/// libjxl's gradient property (W + N - NW) and its static properties
/// (channel, stream id), the only ones its lookup-table path reads.
const GRADIENT_PROPERTY: usize = 9;
const NUM_STATIC_PROPERTIES: usize = 2;

const GLOBAL_PLANE_SPLIT_COST_BITS: f32 = 25.0;

const GLOBAL_PLANE_SIDE_PREDS: usize = 4;

/// Every predictor but the Weighted one, for planes coded at full scale.
const PLANE_PRED_MASK: u16 = ((1 << NUM_MA_PREDS) - 1) & !(1 << PRED_WEIGHTED);

/// Learn one tree over every crop of `data` (`stride` wide, channel `chan`
/// of its streams; divided in place by its common factor) and tokenize each
/// crop through it. `palette`, a row coded as channel 0 of the first crop's
/// stream ahead of the plane, is tokenized through the same tree.
#[allow(clippy::too_many_arguments)]
pub(crate) fn learn_global_plane_tree(
    data: &mut [i32],
    stride: usize,
    chan: u32,
    palette: Option<&[i32]>,
    crops: &[StreamCrop],
    budget: PlaneTreeBudget,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> GlobalPlaneTree {
    let (q, min_symbol) = divide_common_factor(data);
    debug_assert!(
        q == 1 || palette.is_none(),
        "palette indices share no factor"
    );
    let data = &*data;
    let px: usize = crops.iter().map(|&(_, _, w, h, _)| w * h).sum();
    let mut step = px.div_ceil(budget.samples).max(1);
    if step > 1 && step.is_multiple_of(2) {
        step += 1;
    }
    let mut samples = MaSamples::with_capacity(px / step + crops.len());
    for &(x0, y0, w, h, stream) in crops {
        sample_channel_ma(
            |y| &data[(y0 + y) * stride + x0..][..w],
            w,
            h,
            chan,
            stream,
            &[],
            WpParams::DEFAULT,
            false,
            step,
            |props, tok, _| samples.push(props, tok),
        );
    }
    scale_ma_value_props(&mut samples, q as i32);
    if budget.gradient_only {
        // A constant property never splits.
        samples.props.map_rows(|p| {
            for (i, v) in p.iter_mut().enumerate() {
                if i >= NUM_STATIC_PROPERTIES && i != GRADIENT_PROPERTY {
                    *v = 0;
                }
            }
        });
    }
    let params = MaLearnParams {
        alphabet: min_symbol as usize,
        max_leaves: budget.leaves,
        split_cost_bits: GLOBAL_PLANE_SPLIT_COST_BITS / step as f32,
        min_node: MA_MIN_NODE_SAMPLES,
        allow_wp: false,
        allowed_preds: if budget.gradient_only {
            1 << PREDICTOR_GRADIENT
        } else if q > 1 {
            SAFE_PREDICTORS
        } else {
            PLANE_PRED_MASK
        },
        side_preds: GLOBAL_PLANE_SIDE_PREDS,
        max_candidates: MA_PHOTO_CANDIDATES,
        modal_offsets: false,
        prune_leaf_bits: 0.0,
    };
    let tree = learn_ma_tree_in_place(&mut samples, params, pool, scratch);
    drop(samples);
    let mults = vec![q; tree.nodes.len()];
    let (tree_tokens, leaf_ctx, ..) = emit_forest_bfs(&tree.nodes, &mults, 0, Vec::new());
    let crop_tokens = pool.steal_map(scratch, crops.len(), |i, _scratch| {
        let (x0, y0, w, h, stream) = crops[i];
        let mut tokens = Vec::with_capacity(w * h);
        if i == 0
            && let Some(row) = palette
        {
            tokenize_channel_ma_mult(
                |_| row,
                row.len(),
                1,
                0,
                stream,
                1,
                &tree,
                &leaf_ctx,
                0,
                &[],
                &mut tokens,
            );
        }
        let rows = |y: usize| &data[(y0 + y) * stride + x0..][..w];
        match gradient_lut(&tree, &leaf_ctx, chan, stream) {
            Some(lut) => tokenize_gradient_lut(rows, w, h, q as i64, &lut, &mut tokens),
            None => tokenize_channel_ma_mult(
                rows,
                w,
                h,
                chan,
                stream,
                q as i32,
                &tree,
                &leaf_ctx,
                0,
                &[],
                &mut tokens,
            ),
        }
        tokens
    });
    GlobalPlaneTree {
        tree_tokens,
        crop_tokens,
    }
}

/// Half-width of libjxl's gradient lookup table (`kPropRangeFast`): it reads
/// the gradient property clamped to [-8192, 8191].
const GRADIENT_LUT_RANGE: i32 = 512 << 4;

/// A Gradient-only tree's contexts by gradient property in one stream, as
/// libjxl's lookup table holds them: None unless every leaf predicts Gradient
/// and every non-static split is on the gradient inside the table's range.
fn gradient_lut(tree: &LearnedTree, leaf_ctx: &[u32], chan: u32, stream: i32) -> Option<Vec<u32>> {
    let fits = tree.nodes.iter().all(|node| match *node {
        MaNode::Leaf { pred, offset } => pred == PREDICTOR_GRADIENT && offset == 0,
        MaNode::Split { prop, val, .. } => {
            (prop as usize) < NUM_STATIC_PROPERTIES
                || prop as usize == GRADIENT_PROPERTY
                    && (-GRADIENT_LUT_RANGE..GRADIENT_LUT_RANGE - 1).contains(&val)
        }
    });
    if !fits {
        return None;
    }
    let mut p = [0i32; NUM_MA_PROPS];
    p[0] = chan as i32;
    p[1] = stream;
    Some(
        (-GRADIENT_LUT_RANGE..GRADIENT_LUT_RANGE)
            .map(|g| {
                p[GRADIENT_PROPERTY] = g;
                leaf_ctx[tree.lookup(&p).0 as usize]
            })
            .collect(),
    )
}

/// [`tokenize_channel_ma_mult`] through a [`gradient_lut`]: the decoder's
/// neighbors, the clamped-gradient prediction and the full-scale (`q`
/// times the divided) gradient property, without the other properties.
fn tokenize_gradient_lut<'a>(
    get_row: impl Fn(usize) -> &'a [i32],
    gw: usize,
    gh: usize,
    q: i64,
    lut: &[u32],
    out: &mut Vec<Token>,
) {
    let range = i64::from(GRADIENT_LUT_RANGE);
    let mut push = |v: i64, left: i64, top: i64, top_left: i64| {
        let g = (q * (left + top - top_left)).clamp(-range, range - 1);
        let pred = clamped_gradient(left, top, top_left);
        out.push(Token::new(
            lut[(g + range) as usize],
            pack_signed((v - pred) as i32),
        ));
    };
    let mut north: &[i32] = &[];
    for y in 0..gh {
        let row = get_row(y);
        if y == 0 {
            // The first pixel's neighbors are all zero; the rest of the
            // row sees only its left one.
            let mut left = 0;
            for &v in row {
                let v = i64::from(v);
                push(v, left, left, left);
                left = v;
            }
        } else if gw > 0 {
            let top = i64::from(north[0]);
            push(i64::from(row[0]), top, top, top);
            for x in 1..gw {
                push(
                    i64::from(row[x]),
                    i64::from(row[x - 1]),
                    i64::from(north[x]),
                    i64::from(north[x - 1]),
                );
            }
        }
        north = row;
    }
}

/// Divide every value by their common factor; returns the factor (the leaf
/// multiplier) and the coded alphabet's minimum symbol count.
pub(crate) fn divide_common_factor(data: &mut [i32]) -> (u32, u32) {
    let q = data.iter().fold(0, |g, &v| gcd(g, v.unsigned_abs())).max(1);
    if q > 1 {
        let divider = crate::fast_divide::DividerI32::new(q as i32);
        data.iter_mut().for_each(|v| *v = *v / divider);
    }
    let max_abs = data.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    let value_bits = 33 - (2 * max_abs).max(1).leading_zeros();
    let min_symbol = if value_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * value_bits + 24
    };
    (q, min_symbol)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lossless::tree::{TREE_CTX_PREDICTOR, TREE_CTX_PROPERTY, TREE_CTX_SPLIT_VAL};
    use crate::util::fast_hypot;

    #[test]
    fn common_factor_division_preserves_signed_quotients() {
        for factor in [2i32, 3, 257, 65535, 1 << 30] {
            let mut values: Vec<_> = (-3i32..=3).filter_map(|v| v.checked_mul(factor)).collect();
            let expected: Vec<_> = values.iter().map(|&v| v / factor).collect();
            assert_eq!(divide_common_factor(&mut values).0, factor as u32);
            assert_eq!(values, expected);
        }
        let mut values = [i32::MIN, 0];
        assert_eq!(divide_common_factor(&mut values).0, 1 << 31);
        assert_eq!(values, [1, 0]); // Preserve the existing q-as-i32 divisor.
    }

    fn unpack(v: u32) -> i32 {
        if v & 1 == 0 {
            (v >> 1) as i32
        } else {
            -((v >> 1) as i32) - 1
        }
    }

    /// Anti-aliased discs over a 600x300 plane, coded as two 300-wide crops.
    fn discs() -> Vec<i32> {
        let (w, h) = (600usize, 300usize);
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                [
                    (150.0, 150.0, 90.0),
                    (420.0, 120.0, 70.0),
                    (460.0, 230.0, 50.0),
                ]
                .iter()
                .map(|&(cx, cy, r): &(f32, f32, f32)| {
                    (0.5 - (fast_hypot(x - cx, y - cy) - r) / 1.5).clamp(0.0, 1.0)
                })
                .fold(0.0f32, f32::max)
            })
            .map(|a| (a * 255.0 + 0.5) as i32)
            .collect()
    }

    #[test]
    fn gradient_lookup_tokens_match_the_tree_walk() {
        use crate::ma_tree::MaNode;
        let leaf = MaNode::Leaf {
            pred: PREDICTOR_GRADIENT,
            offset: 0,
        };
        // Stream 4 splits the gradient at 0 and 200, other streams at -3.
        let tree = LearnedTree {
            nodes: vec![
                MaNode::Split {
                    prop: 1,
                    val: 4,
                    gt: 2,
                    le: 1,
                },
                MaNode::Split {
                    prop: 9,
                    val: 0,
                    gt: 3,
                    le: 4,
                },
                MaNode::Split {
                    prop: 9,
                    val: -3,
                    gt: 5,
                    le: 6,
                },
                MaNode::Split {
                    prop: 9,
                    val: 200,
                    gt: 7,
                    le: 8,
                },
                leaf,
                leaf,
                leaf,
                leaf,
                leaf,
            ],
            est_bits: 0.0,
            flat_bits: 0.0,
        };
        let mults = vec![1; tree.nodes.len()];
        let (_, leaf_ctx, ..) = emit_forest_bfs(&tree.nodes, &mults, 0, Vec::new());
        let mut seed = 0x9e37_79b9u32;
        for (w, h) in [(1, 1), (1, 9), (9, 1), (2, 2), (7, 5), (300, 20)] {
            for (stream, q) in [(4, 1i64), (5, 1), (4, 3)] {
                let data: Vec<i32> = (0..w * h)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        [0, 0, 255, 255, (seed % 256) as i32][(seed % 5) as usize]
                    })
                    .collect();
                let rows = |y: usize| &data[y * w..][..w];
                let lut = gradient_lut(&tree, &leaf_ctx, 0, stream).expect("in range");
                let mut fast = Vec::new();
                tokenize_gradient_lut(rows, w, h, q, &lut, &mut fast);
                let mut walk = Vec::new();
                tokenize_channel_ma_mult(
                    rows,
                    w,
                    h,
                    0,
                    stream,
                    q as i32,
                    &tree,
                    &leaf_ctx,
                    0,
                    &[],
                    &mut walk,
                );
                let pairs =
                    |t: &[Token]| t.iter().map(|t| (t.context, t.value)).collect::<Vec<_>>();
                assert_eq!(pairs(&fast), pairs(&walk), "{w}x{h} stream {stream} q {q}");
            }
        }
        // Splits outside the decoder's table, or other properties, walk.
        let mut wide = tree.clone();
        wide.nodes[3] = MaNode::Split {
            prop: 9,
            val: 9000,
            gt: 7,
            le: 8,
        };
        assert!(gradient_lut(&wide, &leaf_ctx, 0, 4).is_none());
        let mut other = tree.clone();
        other.nodes[3] = MaNode::Split {
            prop: 4,
            val: 9,
            gt: 7,
            le: 8,
        };
        assert!(gradient_lut(&other, &leaf_ctx, 0, 4).is_none());
    }

    #[test]
    fn gradient_only_trees_stay_on_the_lookup_table_path() {
        let pool = ThreadPool::new(1);
        let mut scratch = CoderScratch::default();
        let crops = [(0, 0, 300, 300, 4), (300, 0, 300, 300, 5)];
        for budget in [
            PlaneTreeBudget {
                samples: 1 << 16,
                leaves: 16,
                gradient_only: true,
            },
            PlaneTreeBudget::FULL,
        ] {
            let mut data = discs();
            let tree = learn_global_plane_tree(
                &mut data,
                600,
                0,
                None,
                &crops,
                budget,
                &pool,
                &mut scratch,
            );
            let mut splits = Vec::new();
            let mut predictors = Vec::new();
            let mut splitvals = Vec::new();
            for t in &tree.tree_tokens {
                match t.context {
                    TREE_CTX_PROPERTY if t.value > 0 => splits.push(t.value - 1),
                    TREE_CTX_PREDICTOR => predictors.push(t.value),
                    TREE_CTX_SPLIT_VAL => splitvals.push(unpack(t.value)),
                    _ => {}
                }
            }
            assert!(!splits.is_empty(), "{budget:?} learned no splits");
            assert_eq!(tree.crop_tokens.len(), 2);
            if budget.gradient_only {
                assert!(predictors.len() <= 16);
                assert!(
                    predictors.iter().all(|&p| p == PREDICTOR_GRADIENT),
                    "{predictors:?}"
                );
                assert!(splits.iter().all(|&p| p < 2 || p == 9), "{splits:?}");
                // libjxl's lookup table spans property values in [-8192, 8191].
                assert!(splitvals.iter().all(|v| (-8192..8191).contains(v)));
            } else {
                assert!(splits.iter().any(|&p| p >= 2 && p != 9));
            }
        }
    }
}
