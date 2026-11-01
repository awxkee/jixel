/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! One learned MA subtree per tag.
//!
//! Each tag's samples are drawn from its whole channel (every crop of a slot
//! shares the same channel). A tree's leaf budget follows the tag's pixel
//! share, and a learned tree must beat its own flat estimate plus its
//! serialization and context overhead, or it collapses to one leaf.

use super::cache::{CacheKey, TreeCache};
use super::forest::{BUDGET_CAP, BUDGET_DIV, SAMPLE_TARGET};
use super::pyramid::Pyramid;
use super::tags::{References, Tags};
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS};
use crate::lossless::kit::{
    MA_MIN_NODE_SAMPLES, MA_PHOTO_CANDIDATES, MA_SPLIT_COST_BITS, SAFE_PREDICTORS,
    sample_channel_ma, scale_ma_value_props,
};
use crate::ma_tree::{LearnedTree, MaLearnParams, MaSamples, learn_ma_tree};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Context budget of the whole forest: the pre-4096 scratch bound, kept so
/// the lossless leaf cap can grow without perturbing this path's trees.
const MAX_CONTEXTS: usize = 1024;

const _: () = assert!(MAX_CONTEXTS <= LZ77_MAX_CONTEXTS);

/// Odd sampling stride that keeps a channel of `px` samples near
/// `SAMPLE_TARGET` samples.
fn sample_stride(px: usize) -> usize {
    let mut stride = px.div_ceil(SAMPLE_TARGET).max(1);
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    stride
}

fn learn_params(min_symbol: u32, stride: usize, max_leaves: usize) -> MaLearnParams {
    MaLearnParams {
        alphabet: min_symbol as usize,
        max_leaves,
        split_cost_bits: MA_SPLIT_COST_BITS / stride as f32,
        min_node: MA_MIN_NODE_SAMPLES,
        allow_wp: false,
        allowed_preds: SAFE_PREDICTORS,
        side_preds: 1,
        max_candidates: MA_PHOTO_CANDIDATES,
        modal_offsets: false,
        prune_leaf_bits: 0.0,
    }
}

/// MA samples of tag `t`'s channel, value properties at full scale.
fn sample(p: &Pyramid, tags: &Tags, refs: &References, t: usize) -> MaSamples {
    let ch = &p.channels[tags.channel[t]];
    let px = ch.w * ch.h;
    let stride = sample_stride(px);
    let mut samples = MaSamples::with_capacity(px / stride + 1);
    let data = &ch.data;
    let w = ch.w;
    let refs = if t < tags.split {
        refs.of(&p.channels, t)
    } else {
        refs.of_group(&p.channels, tags.split, tags.channel[t], 0, 0)
    };
    sample_channel_ma(
        |y| &data[y * w..][..w],
        ch.w,
        ch.h,
        tags.chan(t),
        0,
        &refs,
        WpParams::DEFAULT,
        false,
        stride,
        |props, tok, _| samples.push(props, tok),
    );
    scale_ma_value_props(&mut samples, tags.quants[t] as i32);
    samples
}

/// Learn every tag's subtree; tags the cache still holds keep their tree.
#[allow(clippy::too_many_arguments)]
pub(super) fn learn(
    p: &Pyramid,
    tags: &Tags,
    refs: &References,
    keys: &[Option<CacheKey>],
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    cache: Option<&mut TreeCache>,
) -> Vec<LearnedTree> {
    let n_tags = tags.len();
    let mut cached: Vec<Option<LearnedTree>> = keys
        .iter()
        .map(|key| key.and_then(|(c, stream)| cache.as_deref()?.trees[c][stream].clone()))
        .collect();
    // Unchanged fine tags retain their trees. Generate their samples only if
    // the joint context budget later requires re-learning one of them.
    let mut tag_samples: Vec<Option<MaSamples>> = pool.steal_map(scratch, n_tags, |t, _scratch| {
        if cached[t].is_some() {
            None
        } else {
            Some(sample(p, tags, refs, t))
        }
    });
    let pixels = |t: usize| {
        let ch = &p.channels[tags.channel[t]];
        ch.w * ch.h
    };

    // Learn sequentially: the learner parallelizes internally.
    let mut learned: Vec<LearnedTree> = Vec::with_capacity(n_tags);
    for t in 0..n_tags {
        if let Some(tree) = cached[t].take() {
            learned.push(tree);
            continue;
        }
        let stride = sample_stride(pixels(t));
        let samples = tag_samples[t].as_ref().expect("uncached tag was sampled");
        if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
            learned.push(learn_ma_tree(
                samples,
                learn_params(min_symbol, stride, 1),
                pool,
                scratch,
            ));
            continue;
        }
        let budget = (pixels(t) / BUDGET_DIV).clamp(1, BUDGET_CAP);
        let tree = learn_ma_tree(
            samples,
            learn_params(min_symbol, stride, budget),
            pool,
            scratch,
        );
        let n_leaves = tree.nodes.len().div_ceil(2);
        let overhead_bits = tree.nodes.len() as f64 * 10.0 + n_leaves as f64 * 200.0;
        let est_real = tree.est_bits * stride as f64 + overhead_bits;
        let flat_real = tree.flat_bits * stride as f64;
        if est_real < flat_real {
            learned.push(tree);
        } else {
            learned.push(learn_ma_tree(
                samples,
                learn_params(min_symbol, stride, 1),
                pool,
                scratch,
            ));
        }
    }

    // Large images at low distance can push the SUM of actual leaves past the
    // LZ77 context bound; only then halve the largest tree's budget and
    // re-learn it (budgets are upper bounds and most trees collapse well
    // below them, so rescaling up front would perturb every ordinary encode).
    let budget_cap = MAX_CONTEXTS - 64;
    let leaves_of = |t: &LearnedTree| t.nodes.len().div_ceil(2);
    let mut total_leaves: usize = learned.iter().map(leaves_of).sum();
    while total_leaves > budget_cap {
        let (worst, _) = learned
            .iter()
            .enumerate()
            .max_by_key(|(_, t)| leaves_of(t))
            .unwrap();
        let target = (leaves_of(&learned[worst]) / 2).max(1);
        let params = learn_params(min_symbol, sample_stride(pixels(worst)), target);
        let samples = tag_samples[worst].get_or_insert_with(|| sample(p, tags, refs, worst));
        learned[worst] = learn_ma_tree(samples, params, pool, scratch);
        total_leaves = learned.iter().map(leaves_of).sum();
    }

    if let Some(cache) = cache {
        for (tree, key) in learned.iter().zip(keys) {
            if let Some(key) = *key {
                cache.update_tree(key, tree);
            }
        }
    }
    learned
}
