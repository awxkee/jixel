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

//! Every stream's residual tokens under the merged forest.
//!
//! Each crop is walked through its tag's subtree with local leaf-node ids,
//! which the forest's leaf table then maps to contexts. The cache keeps the
//! local-id tokens, so a cached block only needs that remapping. Tokens are
//! literal-only: the shared per-pixel contexts already make zero residuals
//! nearly free.

use super::cache::{CacheKey, PendingTokens, TreeCache};
use super::pyramid::Pyramid;
use super::tags::{GroupStreams, References, Tags};
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;
use crate::lossless::kit::{
    LzToken, LzTokenSource, MaRefPlane, group_lz_threads, tokenize_channel_ma_mult,
};
use crate::ma_tree::LearnedTree;
use crate::thread_pool::ThreadPool;

/// The frame's tokens, stream by stream.
pub(super) struct Streams {
    pub(super) global: Vec<LzToken>,
    pub(super) dc_groups: Vec<Vec<LzToken>>,
    pub(super) ac_groups: Vec<Vec<LzToken>>,
}

/// The merged forest's view of each tag: `leaf_ctx` maps arena node ids to
/// contexts, `tag_base` is each tag's arena offset.
pub(super) struct ForestContexts<'a> {
    pub(super) leaf_ctx: &'a [u32],
    pub(super) tag_base: &'a [u32],
}

/// Tokenize every stream; also returns the freshly coded blocks the cache
/// admitted.
#[allow(clippy::too_many_arguments)]
pub(super) fn tokenize(
    p: &Pyramid,
    tags: &Tags,
    refs: &References,
    trees: &[LearnedTree],
    forest: ForestContexts<'_>,
    keys: &[Option<CacheKey>],
    cache: Option<&TreeCache>,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (Streams, Vec<PendingTokens>) {
    let channels = &p.channels;
    // Local leaf-node ids survive forest routing/context renumbering. The
    // stored value is the signed residual, independent of its entropy code.
    let local_leaf_ctx: Vec<Vec<u32>> = trees
        .iter()
        .map(|tree| (0..tree.nodes.len() as u32).collect())
        .collect();
    let tokenize_block = |tag: usize,
                          channel: usize,
                          chan: u32,
                          x0: usize,
                          y0: usize,
                          width: usize,
                          height: usize,
                          group: usize,
                          refs: &[MaRefPlane<'_>],
                          out: &mut Vec<LzToken>,
                          pending: &mut Vec<PendingTokens>| {
        let key = keys[tag];
        let append = |tokens: &[Token], out: &mut Vec<LzToken>| {
            let base = forest.tag_base[tag] as usize;
            out.extend(tokens.iter().map(|token| {
                Token::new(forest.leaf_ctx[base + token.context as usize], token.value).as_lz()
            }));
        };
        if let Some(tokens) = key.and_then(|key| cache?.block(key, group)) {
            append(tokens, out);
            return;
        }
        let ch = &channels[channel];
        let mut tokens = Vec::with_capacity(width * height);
        tokenize_channel_ma_mult(
            |y| &ch.data[(y0 + y) * ch.w + x0..][..width],
            width,
            height,
            chan,
            0,
            tags.quants[tag] as i32,
            &trees[tag],
            &local_leaf_ctx[tag],
            0,
            refs,
            &mut tokens,
        );
        append(&tokens, out);
        if let Some(key) = key.filter(|&key| cache.is_some_and(|cache| cache.admitted(key))) {
            pending.push(PendingTokens { key, group, tokens });
        }
    };

    // The global stream: whole channels, the DC frame's with references.
    let global_channels = pool.steal_map(scratch, tags.split, |c, _scratch| {
        let ch = &channels[c];
        let mut out = Vec::with_capacity(ch.w * ch.h);
        let mut pending = Vec::new();
        tokenize_block(
            c,
            c,
            c as u32,
            0,
            0,
            ch.w,
            ch.h,
            0,
            &refs.of(channels, c),
            &mut out,
            &mut pending,
        );
        (out, pending)
    });
    let mut global = Vec::new();
    let mut pending = Vec::new();
    for (tokens, blocks) in global_channels {
        global.extend(tokens);
        pending.extend(blocks);
    }

    // Group streams: every slot's crop, in slot order.
    let tokenize_groups = |streams: &GroupStreams, scratch: &mut CoderScratch| {
        pool.steal_map_with_threads(
            scratch,
            streams.count,
            group_lz_threads(speed, pool),
            |group, _scratch| {
                let mut out = Vec::new();
                let mut pending = Vec::new();
                streams.for_each_crop(
                    p,
                    tags.split,
                    group,
                    |slot, channel, x0, y0, width, height| {
                        tokenize_block(
                            streams.first_tag + slot,
                            channel,
                            slot as u32,
                            x0,
                            y0,
                            width,
                            height,
                            group,
                            &[],
                            &mut out,
                            &mut pending,
                        );
                    },
                );
                (out, pending)
            },
        )
    };
    let dc = tokenize_groups(&tags.dc, scratch);
    let ac = tokenize_groups(&tags.ac, scratch);
    let mut dc_groups = Vec::with_capacity(tags.dc.count);
    for (tokens, blocks) in dc {
        dc_groups.push(tokens);
        pending.extend(blocks);
    }
    let mut ac_groups = Vec::with_capacity(tags.ac.count);
    for (tokens, blocks) in ac {
        ac_groups.push(tokens);
        pending.extend(blocks);
    }
    let streams = Streams {
        global,
        dc_groups,
        ac_groups,
    };
    (streams, pending)
}
