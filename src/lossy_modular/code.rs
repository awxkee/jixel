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

//! Stage 5: one quantized pyramid as a complete modular frame.
//!
//! A frame that fits one group is a single section ([`super::single`]).
//! Otherwise the channels spread over the global, DC-group and AC-group
//! streams, and one global tree serves them all. Group streams key the tree
//! on the within-group slot while the global stream keys on the real channel
//! index, and with per-channel multipliers the two sides cannot share leaves
//! the way the lossless path does. The global tree therefore splits first on
//! the static stream-id property and hangs a separate channel chain off each
//! side ([`super::forest::emit_forest`]).

use super::cache::{CacheKey, TreeCache};
use super::forest::emit_forest;
use super::pyramid::Pyramid;
use super::tags::{Grid, References, Tags};
use super::tokens::ForestContexts;
use super::{single, tokens, trees, write};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::lossless::kit::{LZ77_MIN_SYMBOL, build_lz_pixel_code_ans};
use crate::thread_pool::ThreadPool;

/// Code `p` as a complete frame (header, TOC and sections). With a cache,
/// fine-channel trees and tokens are reused across race candidates.
pub(super) fn write_frame(
    p: &Pyramid,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
    mut cache: Option<&mut TreeCache>,
) {
    let min_symbol = min_symbol(p);
    let grid = Grid::new(p.xsize, p.ysize);
    if grid.num_ac_groups == 1 && !p.dc_frame {
        single::write_frame(p, min_symbol, speed, pool, scratch, writer);
        return;
    }

    // 1. Tags: which stream and slot codes every channel crop.
    let tags = Tags::new(p, &grid);
    let refs = References::new(p, tags.split);
    let keys: Vec<Option<CacheKey>> = (0..tags.len())
        .map(|t| {
            cache
                .as_deref()
                .and_then(|cache| cache.key(p, tags.channel[t], tags.stream(t)))
        })
        .collect();

    // 2. One learned subtree per tag, merged into the global forest.
    let trees = trees::learn(
        p,
        &tags,
        &refs,
        &keys,
        min_symbol,
        pool,
        scratch,
        cache.as_deref_mut(),
    );
    let (tree_tokens, leaf_ctx, tag_base, num_ctx) = emit_forest(
        &trees,
        &tags.quants,
        tags.split,
        tags.n_dc_slots,
        grid.num_dc_groups,
    );
    let num_ctx = num_ctx as usize;
    let distance_ctx = num_ctx as u32;

    // 3. Every stream's tokens under the forest.
    if let Some(cache) = cache.as_deref_mut() {
        cache.reserve_frame(p, &tags, &keys);
    }
    let (streams, pending) = tokens::tokenize(
        p,
        &tags,
        &refs,
        &trees,
        ForestContexts {
            leaf_ctx: &leaf_ctx,
            tag_base: &tag_base,
        },
        &keys,
        cache.as_deref(),
        speed,
        pool,
        scratch,
    );
    if let Some(cache) = cache {
        for block in pending {
            cache.retain(block);
        }
    }

    // 4. One entropy code over all streams, then the sections.
    let dictionary = p.patch_dictionary(scratch);
    let code = build_lz_pixel_code_ans(
        speed.effort().lz_exhaustive_entropy(),
        std::iter::once(streams.global.as_slice())
            .chain(streams.dc_groups.iter().map(Vec::as_slice))
            .chain(streams.ac_groups.iter().map(Vec::as_slice)),
        num_ctx,
        min_symbol,
        speed.effort().lz_refined_entropy,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write::write_frame(
        p,
        &tree_tokens,
        &code,
        min_symbol,
        distance_ctx,
        &streams,
        dictionary,
        pool,
        &mut scratch.huffman_pool,
        writer,
    );
}

/// The alphabet bound of the frame's symbols: the literal-vs-LZ77 guard
/// from the actual quantized channel range.
fn min_symbol(p: &Pyramid) -> u32 {
    let max_abs = p
        .channels
        .iter()
        .flat_map(|ch| ch.data.iter())
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    let value_bits = 33 - max_abs.saturating_mul(2).max(1).leading_zeros();
    if value_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * value_bits + 24
    }
}
