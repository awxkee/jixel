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

//! Trees and tokens carried from one race candidate to the next.
//!
//! A fold changes only the coarse averages, so the fine channels' trees and
//! their tokens usually survive. Tokens are stored with local leaf-node ids,
//! which stay valid under any forest renumbering; a changed tree drops them.

use super::pyramid::Pyramid;
use super::tags::Tags;
use crate::entropy::Token;
use crate::ma_tree::LearnedTree;

/// A cached tag: its original channel index and stream kind (0 = global,
/// 1 = DC group, 2 = AC group).
pub(super) type CacheKey = (usize, usize);

/// Reuse fine-channel trees and local-node-id tokens while the coarse
/// representation changes. Keys retain original channel and stream identity.
pub(super) struct TreeCache {
    pub(super) original_len: usize,
    pub(super) averages: usize,
    pub(super) trees: Vec<[Option<LearnedTree>; 3]>,
    pub(super) tokens: Vec<[CachedTokens; 3]>,
    pub(super) token_bytes: usize,
}

#[derive(Default)]
pub(super) struct CachedTokens {
    blocks: Vec<Option<Vec<Token>>>,
    reserved_bytes: usize,
}

pub(super) struct PendingTokens {
    pub(super) key: CacheKey,
    pub(super) group: usize,
    pub(super) tokens: Vec<Token>,
}

pub(super) const TOKEN_CACHE_BYTES: usize = 64 * 1024 * 1024;

impl TreeCache {
    pub(super) fn new(q: &Pyramid) -> Self {
        Self {
            original_len: q.channels.len(),
            averages: q.steps.last().map_or(0, |step| step.num_c),
            trees: (0..q.channels.len()).map(|_| [None, None, None]).collect(),
            tokens: (0..q.channels.len())
                .map(|_| std::array::from_fn(|_| CachedTokens::default()))
                .collect(),
            token_bytes: 0,
        }
    }

    pub(super) fn update_tree(&mut self, key: CacheKey, tree: &LearnedTree) {
        let (channel, stream) = key;
        if self.trees[channel][stream]
            .as_ref()
            .is_none_or(|old| old.nodes != tree.nodes)
        {
            let old = std::mem::take(&mut self.tokens[channel][stream]);
            self.token_bytes -= old.reserved_bytes;
        }
        self.trees[channel][stream] = Some(tree.clone());
    }

    pub(super) fn reserve_tokens(&mut self, key: CacheKey, samples: usize, groups: usize) {
        let (channel, stream) = key;
        if !self.tokens[channel][stream].blocks.is_empty() {
            return;
        }
        let Some(bytes) = samples.checked_mul(std::mem::size_of::<Token>()) else {
            return;
        };
        if bytes > TOKEN_CACHE_BYTES - self.token_bytes {
            return;
        }
        self.tokens[channel][stream] = CachedTokens {
            blocks: (0..groups).map(|_| None).collect(),
            reserved_bytes: bytes,
        };
        self.token_bytes += bytes;
    }

    pub(super) fn block(&self, key: CacheKey, group: usize) -> Option<&[Token]> {
        self.tokens[key.0][key.1].blocks.get(group)?.as_deref()
    }

    pub(super) fn admitted(&self, key: CacheKey) -> bool {
        !self.tokens[key.0][key.1].blocks.is_empty()
    }

    pub(super) fn retain(&mut self, pending: PendingTokens) {
        let entry = &mut self.tokens[pending.key.0][pending.key.1];
        if let Some(block) = entry.blocks.get_mut(pending.group) {
            debug_assert!(block.is_none());
            *block = Some(pending.tokens);
        }
    }

    pub(super) fn key(&self, q: &Pyramid, channel: usize, stream: usize) -> Option<CacheKey> {
        // The merged averages change on every fold, and global trees see
        // previous same-shaped channels, whose reference set a fold changes
        // (a merged average takes the shape of the residuals of its level).
        if channel < self.averages || stream == 0 {
            return None;
        }
        let original = channel + self.original_len - q.channels.len();
        debug_assert!(original < self.original_len);
        Some((original, stream))
    }
}

impl TreeCache {
    /// Reserve token blocks for every cacheable tag of this frame.
    pub(super) fn reserve_frame(&mut self, p: &Pyramid, tags: &Tags, keys: &[Option<CacheKey>]) {
        let mut tag_pixels = vec![0usize; tags.len()];
        for c in 0..tags.split {
            tag_pixels[c] = p.channels[c].data.len();
        }
        for streams in [&tags.dc, &tags.ac] {
            for group in 0..streams.count {
                streams.for_each_crop(p, tags.split, group, |slot, _c, _x, _y, width, height| {
                    tag_pixels[streams.first_tag + slot] += width * height
                });
            }
        }
        for (t, key) in keys.iter().enumerate() {
            if let Some(key) = *key {
                let groups = [1, tags.dc.count, tags.ac.count][key.1];
                self.reserve_tokens(key, tag_pixels[t], groups);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_blocks_are_invalidated_when_their_tree_changes() {
        use crate::ma_tree::MaNode;
        let mut cache = TreeCache {
            original_len: 1,
            averages: 0,
            trees: vec![[None, None, None]],
            tokens: vec![std::array::from_fn(|_| CachedTokens::default())],
            token_bytes: 0,
        };
        let mut tree = LearnedTree {
            nodes: vec![MaNode::Leaf { pred: 0, offset: 0 }],
            est_bits: 0.0,
            flat_bits: 0.0,
        };
        let key = (0, 2);
        cache.update_tree(key, &tree);
        cache.reserve_tokens(key, 2, 1);
        cache.retain(PendingTokens {
            key,
            group: 0,
            tokens: vec![Token::new(0, 17), Token::new(0, 23)],
        });
        tree.est_bits = 123.0;
        cache.update_tree(key, &tree);
        assert_eq!(cache.block(key, 0).unwrap()[1].value, 23);
        tree.nodes[0] = MaNode::Leaf { pred: 7, offset: 0 };
        cache.update_tree(key, &tree);
        assert!(cache.block(key, 0).is_none());
        assert_eq!(cache.token_bytes, 0);
        cache.reserve_tokens(key, TOKEN_CACHE_BYTES / std::mem::size_of::<Token>() + 1, 1);
        assert!(!cache.admitted(key));
        assert_eq!(cache.token_bytes, 0);
    }
}
