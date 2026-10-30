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

use super::context::{pick_threshold, pick_threshold_grouped};
use super::ma_learning::{finish_ma_candidate, gate_ma_tree};
use super::ma_lookup::MaLookup;
use super::ma_sampling::{tokenize_channel_ma, walk_channel_ma};
use super::predictor::predictor_value;
use super::tokens::{RawTokenStreams, RawTokens};
use super::tree::{
    TREE_CTX_MULTIPLIER_BITS, TREE_CTX_MULTIPLIER_LOG, TREE_CTX_OFFSET, TREE_CTX_PREDICTOR,
    TREE_CTX_PROPERTY, TREE_CTX_SPLIT_VAL, emit_learned_tree,
};
use crate::coder_scratch::CoderScratch;
use crate::entropy::pack_signed;
use crate::ma_tree::{LearnedTree, MaNode, MaSamples, NUM_MA_PREDS, NUM_MA_PROPS};
use crate::weighted_predictor::WpParams;

#[test]
fn gated_tree_keeps_bfs_tokens_with_leaf_offsets() {
    let tree = LearnedTree {
        nodes: vec![
            MaNode::Split {
                prop: 0,
                val: 0,
                gt: 1,
                le: 2,
            },
            MaNode::Leaf { pred: 1, offset: 0 },
            MaNode::Split {
                prop: 3,
                val: 3,
                gt: 3,
                le: 4,
            },
            MaNode::Leaf { pred: 6, offset: 0 },
            MaNode::Leaf { pred: 0, offset: 0 },
        ],
        est_bits: 100.,
        flat_bits: 10000.,
    };
    let mut samples = MaSamples::new();
    for (channel, x, residual) in [(1, 0, -3), (0, 4, 5), (0, 3, 0)] {
        let mut props = [0; NUM_MA_PROPS];
        props[0] = channel;
        props[3] = x;
        for _ in 0..64 {
            samples.push(props, [pack_signed(residual) as u8; NUM_MA_PREDS]);
        }
    }
    for with_offsets in [false, true] {
        let candidate =
            gate_ma_tree(tree.clone(), 1., with_offsets.then_some(&samples), None).unwrap();
        let offsets = if with_offsets { [-3, 5, 0] } else { [0; 3] };
        assert_eq!(candidate.leaf_offset, offsets);
        assert_eq!(candidate.leaf_ctx, [u32::MAX, 0, u32::MAX, 1, 2]);
        assert_eq!(candidate.num_ctx, 3);
        let leaf = |pred, offset| {
            [
                (TREE_CTX_PROPERTY, 0),
                (TREE_CTX_PREDICTOR, pred),
                (TREE_CTX_OFFSET, pack_signed(offset)),
                (TREE_CTX_MULTIPLIER_LOG, 0),
                (TREE_CTX_MULTIPLIER_BITS, 0),
            ]
        };
        let mut expected = vec![(TREE_CTX_PROPERTY, 1), (TREE_CTX_SPLIT_VAL, 0)];
        expected.extend(leaf(1, offsets[0]));
        expected.extend([(TREE_CTX_PROPERTY, 4), (TREE_CTX_SPLIT_VAL, 6)]);
        expected.extend(leaf(6, offsets[1]));
        expected.extend(leaf(0, offsets[2]));
        let actual: Vec<_> = candidate
            .tree_tokens
            .iter()
            .map(|token| (token.context, token.value))
            .collect();
        assert_eq!(actual, expected);
        // 100 sample bits + 19 tree tokens x 10 + 3 clusters x 200 + 3 contexts x 8.
        assert_eq!(candidate.est_real, 914.);
    }
    // Keeping only the original fourth rows must preserve the offsets,
    // even if the full sample set is subsequently partitioned in place.
    let retained = MaSamples {
        props: samples.props.sampled(4),
        tok: samples.tok.iter().step_by(4).copied().collect(),
        val: Vec::new(),
    };
    samples.props.reverse();
    samples.tok.reverse();
    let mut retained_candidate = finish_ma_candidate(tree.clone(), 1., None);
    retained_candidate.set_leaf_offsets(&retained, 1);
    assert_eq!(retained_candidate.leaf_offset, [-3, 5, 0]);

    // Admission uses only the unchanged token-count estimate.
    for est_bits in [10000., f64::NAN] {
        assert!(
            gate_ma_tree(
                LearnedTree {
                    est_bits,
                    ..tree.clone()
                },
                1.,
                Some(&samples),
                None,
            )
            .is_none()
        );
    }
}

#[test]
fn grouped_threshold_scoring_matches_concatenated_reference() {
    let groups = vec![
        vec![(vec![0, 3, 7, 3], vec![-20, -4, 12, 80])],
        vec![(vec![7, 1, 1, 9, 2], vec![-100, -9, 0, 17, 110])],
    ];
    let residuals: Vec<u32> = groups
        .iter()
        .flat_map(|group| group[0].0.iter().copied())
        .collect();
    let properties: Vec<i64> = groups
        .iter()
        .flat_map(|group| group[0].1.iter().copied())
        .collect();
    let expected = pick_threshold(&residuals, &properties, &mut CoderScratch::default());
    let actual = pick_threshold_grouped(&groups, 0, &mut CoderScratch::default());
    assert_eq!(actual, expected);
}
#[test]
fn compiled_ma_tokenization_matches_full_tree_with_and_without_wp() {
    let tree = LearnedTree {
        nodes: vec![
            MaNode::Split {
                prop: 0,
                val: 0,
                gt: 1,
                le: 4,
            },
            MaNode::Split {
                prop: 15,
                val: 0,
                gt: 2,
                le: 3,
            },
            MaNode::Leaf { pred: 6, offset: 0 },
            MaNode::Leaf { pred: 1, offset: 0 },
            MaNode::Split {
                prop: 3,
                val: 3,
                gt: 5,
                le: 6,
            },
            MaNode::Leaf { pred: 5, offset: 0 },
            MaNode::Leaf { pred: 1, offset: 0 },
        ],
        est_bits: 0.,
        flat_bits: 0.,
    };
    let (_, contexts, _) = emit_learned_tree(&tree);
    for channel in [0, 1, 7, u32::MAX] {
        let lookup = MaLookup::new(&tree, &contexts, channel, 0);
        assert_eq!(lookup.needs_wp(), (channel as i32) > 0);
        for width in [0, 1, 2, 3, 4, 5, 9, 257] {
            for height in [0, 1, 2, 3, 17] {
                let values: Vec<i32> = (0..width * height)
                    .map(|i| (((i * 7919) ^ (i / 7 * 1237)) % 65536) as i32 - 32768)
                    .collect();
                let get_row = |y| &values[y * width..][..width];
                for use_wp in [false, true] {
                    let mut expected = Vec::new();
                    walk_channel_ma(
                        get_row,
                        width,
                        height,
                        channel,
                        0,
                        &[],
                        WpParams::DEFAULT,
                        use_wp,
                        |_x, _y, value, props, n, wp| {
                            let (node, pred) = tree.lookup(props);
                            expected.push((
                                contexts[node as usize],
                                pack_signed((value - predictor_value(pred, n, wp)) as i32),
                            ));
                        },
                    );
                    let mut actual = RawTokens::with_capacity(values.len());
                    tokenize_channel_ma(
                        get_row,
                        width,
                        height,
                        channel,
                        0,
                        &[],
                        WpParams::DEFAULT,
                        use_wp,
                        &lookup,
                        &vec![0; contexts.len()],
                        &mut actual,
                    );
                    let actual: Vec<_> = match RawTokenStreams::new(vec![actual]) {
                        RawTokenStreams::Compact(streams) => streams
                            .into_iter()
                            .flatten()
                            .map(|t| {
                                let t = t.unpack();
                                (t.context, t.value)
                            })
                            .collect(),
                        RawTokenStreams::Wide(streams) => streams
                            .into_iter()
                            .flatten()
                            .map(|t| (t.context, t.value))
                            .collect(),
                    };
                    assert_eq!(
                        actual, expected,
                        "channel={channel}, width={width}, height={height}, wp={use_wp}"
                    );
                }
            }
        }
    }
}
