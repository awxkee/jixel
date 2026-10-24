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

//! Literal, run, deep, and priced LZ stream selection for learned trees.

use super::groups::{DeepLzScratchPool, group_lz_threads};
use super::lz77;
use super::lz77::{
    LzToken, estimate_literal_and_run_bits, estimate_streams_bits,
    lz77_compress_for_speed_with_depth, lz77_run_count,
};
use crate::coder_scratch::CoderScratch;
use crate::thread_pool::ThreadPool;

pub(super) enum TreeStreams<T> {
    /// Retain the original tokens without expanding them to LzToken records.
    Literals(Vec<Vec<T>>),
    Lz(Vec<Vec<LzToken>>),
}

/// Runs and matches collapse a token stream, but a learned tree routes flat
/// content into near-deterministic contexts where a literal costs almost
/// nothing under ANS while every run token still pays for its length and
/// distance — yet a palette index image can prefer long matches. Streams that barely
/// collapse keep the LZ77 layer outright; otherwise the order-0 estimates
/// order the literal and LZ77 variants (the deep matcher is skipped when
/// runs alone trail literals by `LZ_DEEP_MATCHER_MAX_RATIO`), and the
/// writer only codes a later variant when its code-based estimate beats
/// the one already written.
pub(super) const LZ_LITERAL_ALTERNATIVE_MAX_RATIO: f64 = 0.98;

pub(super) const LZ_DEEP_MATCHER_MAX_RATIO_PALETTE: f64 = 1.6;

pub(super) const LZ_DEEP_MATCHER_MAX_RATIO_RCT: f64 = 1.01;

/// A later variant is written when its code-based estimate is within this
/// factor of the best written one (the estimate itself is ~0.5% accurate).
pub(super) const VARIANT_WRITE_TOLERANCE: f64 = 1.005;

/// A priced parse is kept when its estimate is below this share of the
/// literal one.
const PRICED_LZ_MAX_RATIO: f64 = 0.95;

/// Priced LZ77 streams of a learned-tree frame (`lz77_compress_priced`),
/// parsed twice: the second parse prices its matches by the first one's
/// tokens. Where the second parse is a shortest path
/// (`lz77_compress_optimal`), the greedy first one only prices the
/// matches: under the default prices the path takes too few. `None`
/// unless the estimate clears `PRICED_LZ_MAX_RATIO`.
pub(super) fn priced_tree_streams<T: lz77::LiteralToken>(
    speed: crate::Speed,
    tokens: &[&[T]],
    e_lit: f64,
    num_contexts: usize,
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<Vec<Vec<LzToken>>> {
    type Parser<T> = fn(
        &[T],
        &lz77::LiteralPrices,
        &lz77::MatchPrices,
        usize,
        &mut Vec<u32>,
    ) -> Option<Vec<LzToken>>;
    let prices = lz77::LiteralPrices::new(tokens, num_contexts);
    let threads = group_lz_threads(speed, pool);
    let tables = DeepLzScratchPool::new(threads);
    let max_probes = if speed.effort().lz_priced_search() {
        64
    } else {
        lz77::PRICED_MAX_PROBES
    };
    let parse = |scratch: &mut CoderScratch,
                 match_prices: &lz77::MatchPrices,
                 parser: Parser<T>| {
        pool.steal_map_with_threads(scratch, tokens.len(), threads, |k, _scratch| {
            tables.with_depth(|table| parser(tokens[k], &prices, match_prices, max_probes, table))
        })
    };
    // A stream without a parse stays literal.
    let complete = |streams: Vec<Option<Vec<LzToken>>>| -> Vec<Vec<LzToken>> {
        streams
            .into_iter()
            .zip(tokens)
            .map(|(stream, literal)| {
                stream.unwrap_or_else(|| literal.iter().map(|t| t.as_lz()).collect())
            })
            .collect()
    };
    let estimate = |streams: &[Vec<LzToken>], scratch: &mut CoderScratch| {
        let slices: Vec<&[LzToken]> = streams.iter().map(Vec::as_slice).collect();
        estimate_streams_bits(&slices, num_contexts, min_symbol, pool, scratch)
    };
    let first = parse(
        scratch,
        &lz77::MatchPrices::initial(),
        lz77::lz77_compress_priced::<T>,
    );
    if first.iter().all(Option::is_none) {
        return None;
    }
    let learned = {
        let kept: Vec<&[LzToken]> = first.iter().flatten().map(Vec::as_slice).collect();
        lz77::MatchPrices::learned(&kept, num_contexts)
    };
    let first = complete(first);
    let first_bits = estimate(&first, scratch);
    let max_ratio = if speed.effort().lz_priced_search() {
        0.99
    } else {
        PRICED_LZ_MAX_RATIO
    };
    if first_bits >= e_lit * max_ratio {
        return None;
    }
    let second_parser: Parser<T> = if speed.effort().lz_optimal_parse() {
        lz77::lz77_compress_optimal::<T>
    } else {
        lz77::lz77_compress_priced::<T>
    };
    let second = parse(scratch, &learned, second_parser);
    if second.iter().all(Option::is_none) {
        return Some(first);
    }
    let second = complete(second);
    if estimate(&second, scratch) < first_bits {
        Some(second)
    } else {
        Some(first)
    }
}

/// Stream variants to try for one learned-tree frame, most promising first:
/// literal variants reuse the input, and LZ variants own their match streams.
pub(super) fn choose_tree_streams<T: lz77::LiteralToken>(
    speed: crate::Speed,
    tokens: Vec<Vec<T>>,
    distance_ctx: u32,
    num_contexts: usize,
    min_symbol: u32,
    deep_lz_max_ratio: f64,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<TreeStreams<T>> {
    // Preserve the standard candidates. The broader parse competes using the
    // final entropy tables and serialized section size in the variant writer.
    let extra = if speed.effort().lz_priced_search() {
        let slices: Vec<&[T]> = tokens.iter().map(Vec::as_slice).collect();
        let (e_lit, _) =
            estimate_literal_and_run_bits(&slices, num_contexts, min_symbol, pool, scratch);
        priced_tree_streams(
            speed,
            &slices,
            e_lit,
            num_contexts,
            min_symbol,
            pool,
            scratch,
        )
    } else {
        None
    };
    let mut variants = choose_tree_streams_standard(
        speed,
        tokens,
        distance_ctx,
        num_contexts,
        min_symbol,
        deep_lz_max_ratio,
        pool,
        scratch,
    );
    if let Some(extra) = extra {
        variants.push(TreeStreams::Lz(extra));
    }
    variants
}

fn choose_tree_streams_standard<T: lz77::LiteralToken>(
    speed: crate::Speed,
    tokens: Vec<Vec<T>>,
    distance_ctx: u32,
    num_contexts: usize,
    min_symbol: u32,
    deep_lz_max_ratio: f64,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<TreeStreams<T>> {
    let raw: usize = tokens.iter().map(Vec::len).sum();
    // Keep the per-stream counts from the existing gate pass so the run
    // candidates can allocate exactly once, without retaining growth slack.
    let run_counts = pool.steal_map(scratch, tokens.len(), |k, _scratch| {
        lz77_run_count(&tokens[k])
    });
    let run_len: usize = run_counts.iter().sum();
    let deep = |scratch: &mut CoderScratch| -> Vec<Vec<LzToken>> {
        let deep_lz = DeepLzScratchPool::new(group_lz_threads(speed, pool));
        pool.steal_map_with_threads(
            scratch,
            tokens.len(),
            group_lz_threads(speed, pool),
            |k, scratch| {
                deep_lz.with_depth(|depth| {
                    lz77_compress_for_speed_with_depth(
                        &tokens[k],
                        distance_ctx,
                        speed,
                        depth,
                        Some(run_counts[k]),
                        scratch,
                    )
                })
            },
        )
    };
    if (run_len as f64) >= LZ_LITERAL_ALTERNATIVE_MAX_RATIO * raw as f64 {
        return vec![TreeStreams::Lz(deep(scratch))];
    }
    let raw_slices: Vec<&[T]> = tokens.iter().map(Vec::as_slice).collect();
    let (e_lit, e_run) =
        estimate_literal_and_run_bits(&raw_slices, num_contexts, min_symbol, pool, scratch);
    if e_run > e_lit * deep_lz_max_ratio {
        // Runs lose to the literals; a structure repeated further back may
        // still pay where a match is priced against what it replaces.
        if let Some(priced) = priced_tree_streams(
            crate::Speed::Slow,
            &raw_slices,
            e_lit,
            num_contexts,
            min_symbol,
            pool,
            scratch,
        ) {
            return vec![TreeStreams::Literals(tokens), TreeStreams::Lz(priced)];
        }
        return vec![TreeStreams::Literals(tokens)];
    }
    let lz = deep(scratch);
    let slices: Vec<&[LzToken]> = lz.iter().map(Vec::as_slice).collect();
    let e_lz = estimate_streams_bits(&slices, num_contexts, min_symbol, pool, scratch);
    if e_lz < e_lit {
        vec![TreeStreams::Lz(lz), TreeStreams::Literals(tokens)]
    } else {
        vec![TreeStreams::Literals(tokens), TreeStreams::Lz(lz)]
    }
}
