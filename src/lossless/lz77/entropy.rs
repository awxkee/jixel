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

//! LZ77 histogram estimates and entropy-code construction.

use super::{
    ConstantTail, LZ77_DIST_VALUE, LZ77_MIN_LENGTH, LZ77_MIN_SYMBOL, LiteralToken, LzToken,
    LzTokenSource, TokenRuns, lz_tokens, lz77_length_encode,
};
use crate::adaptive_quant::dirty_log2f;
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS, LzEntropyScratch};
use crate::entropy::{
    EntropyCode, Histogram, HybridUintConfig, PrefixCode, Token, build_huffman_codes_into,
    cluster_histograms_fixed, f_log2, uint_encode, uint_encode_with_config,
};
use crate::thread_pool::ThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};

// Retain a worker's high-water capacity without geometric over-allocation
// when a later tree needs more contexts.
fn grow_entropy_scratch<T: Clone>(values: &mut Vec<T>, len: usize, value: T) {
    if values.len() < len {
        values.reserve_exact(len - values.len());
        values.resize(len, value);
    }
}

pub(super) fn estimate_payload_bits<I>(
    tokens: I,
    distance_context: u32,
    scratch: &mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> u64
where
    I: Iterator<Item = LzToken>,
{
    let num_contexts = distance_context as usize + 1;
    assert!(num_contexts <= LZ77_MAX_CONTEXTS);
    grow_entropy_scratch(&mut scratch.histograms, num_contexts, Histogram::new());
    grow_entropy_scratch(&mut scratch.prefix_codes, num_contexts, PrefixCode::zero());
    let histograms = &mut scratch.histograms[..num_contexts];
    histograms.fill(Histogram::new());
    let mut bits = lz_add_histograms(tokens, histograms, LZ77_MIN_SYMBOL, distance_context);
    let codes = &mut scratch.prefix_codes[..num_contexts];
    build_huffman_codes_into(histograms, codes, huffman_pool);
    // Every occurrence of a symbol has the same depth. Price each histogram
    // directly instead of symbolizing the whole stream again.
    for (histogram, code) in histograms.iter().zip(codes) {
        if histogram.total_count != 0 {
            bits += histogram
                .counts
                .iter()
                .zip(&code.depths)
                .map(|(&count, &depth)| u64::from(count) * u64::from(depth))
                .sum::<u64>();
        }
    }
    bits
}

/// First LZ77 symbol a refined code may move to: the largest that keeps
/// every length symbol (at most 43 above it) inside the 128-symbol alphabet.
/// Twenty more literal symbols let a cluster of large residuals keep three or
/// four token bits beyond the exponent: an 800x600 game capture codes 2.1 %
/// smaller than at 64, a 3.9K-colour illustration 0.6 %, the other 16 images
/// of the check within 0.1 %.
const LZ77_RAISED_MIN_SYMBOL: u32 = 84;

/// Accumulate per raw context before clustering and return the extra bits.
#[inline]
fn lz_add_histograms<I>(
    toks: I,
    histograms: &mut [Histogram],
    min_symbol: u32,
    distance_context: u32,
) -> u64
where
    I: IntoIterator<Item = LzToken>,
{
    let mut bits = 0;
    for t in toks {
        if t.is_lz77() {
            let (len_tok, length_bits, _) = lz77_length_encode(t.value);
            histograms[t.context as usize].add(min_symbol + len_tok);

            let (symbol, distance_bits, _) = uint_encode(t.distance);
            histograms[distance_context as usize].add(symbol);
            bits += u64::from(length_bits + distance_bits);
        } else {
            let (sym, literal_bits, _) = uint_encode(t.value);
            histograms[t.context as usize].add(sym);
            bits += u64::from(literal_bits);
        }
    }
    bits
}

/// Materialize literals for paths that own LZ tokens. Learned-tree candidates
/// can instead use the original Token storage through LzTokenSource.
pub(crate) fn lz77_literals(tokens: &[Token]) -> Vec<LzToken> {
    tokens
        .iter()
        .map(|t| LzToken::pixel(t.context, t.value))
        .collect()
}

pub(crate) fn build_lz_pixel_code<'tokens, 'scratch, I>(
    thorough: bool,
    streams: I,
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
    scratch: &'scratch mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> EntropyCode<'scratch>
where
    I: Iterator<Item = &'tokens [LzToken]> + Clone,
{
    build_lz_pixel_code_opts(
        thorough,
        streams,
        nb_chans,
        min_symbol,
        refined,
        false,
        scratch,
        huffman_pool,
    )
}

/// `ans_cluster` switches context clustering to the exact-ANS cost model —
/// required when contexts are finely split per-pixel (MA-tree contexts), where
/// the prefix-depth cost model cannot tell sub-bit histograms apart and
/// collapses the clustering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_lz_pixel_code_opts<'tokens, 'scratch, I>(
    thorough: bool,
    streams: I,
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
    ans_cluster: bool,
    scratch: &'scratch mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> EntropyCode<'scratch>
where
    I: Iterator<Item = &'tokens [LzToken]> + Clone,
{
    build_lz_pixel_code_threads(
        thorough,
        streams,
        nb_chans,
        min_symbol,
        refined,
        ans_cluster,
        None,
        scratch,
        huffman_pool,
    )
}

/// [`build_lz_pixel_code_opts`] that always codes with ANS, refined or not.
/// Streams of mostly-zero residuals under per-pixel contexts need it: a
/// prefix code spends at least one bit on every token. `refined` still adds
/// the hybrid-uint selection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_lz_pixel_code_ans<'tokens, 'scratch, I>(
    thorough: bool,
    streams: I,
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
    ans_cluster: bool,
    scratch: &'scratch mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> EntropyCode<'scratch>
where
    I: Iterator<Item = &'tokens [LzToken]> + Clone,
{
    let streams: Vec<&'tokens [LzToken]> = streams.collect();
    build_lz_pixel_code_slices(
        thorough,
        &streams,
        &[],
        nb_chans,
        min_symbol,
        refined,
        ans_cluster,
        true,
        None,
        scratch,
        huffman_pool,
    )
}

/// Split `streams` into contiguous pool-sized chunks and reduce a per-chunk
/// accumulator. Keep chunk boundaries and merge order stable: coded-bit
/// estimates use floating-point sums, while histogram/count passes use integers.
fn parallel_streams<R: Send, T: LzTokenSource>(
    streams: &[&[T]],
    pool: Option<&ThreadPool>,
    init: impl Fn() -> R + Sync,
    accumulate: impl Fn(&mut R, &[T]) + Sync,
    merge: impl FnMut(&mut R, R),
) -> R {
    let threads = pool
        .map_or(1, ThreadPool::num_threads)
        .min(streams.len().max(1));
    if threads <= 1 {
        let mut acc = init();
        for toks in streams {
            accumulate(&mut acc, toks);
        }
        return acc;
    }
    let chunk = streams.len().div_ceil(threads);
    let parts = pool
        .expect("parallel stream pool")
        .map(streams.len().div_ceil(chunk), |ci| {
            let begin = ci * chunk;
            let part = &streams[begin..(begin + chunk).min(streams.len())];
            let mut acc = init();
            for toks in part {
                accumulate(&mut acc, toks);
            }
            acc
        });
    let mut parts = parts.into_iter();
    let mut acc = parts.next().expect("at least one chunk");
    let mut merge = merge;
    for part in parts {
        merge(&mut acc, part);
    }
    acc
}

/// Integer histogram reductions use at most one accumulator per pool lane.
/// Claim streams dynamically: equal-size groups can have very different run
/// counts. Only integer counts are merged, so scheduling cannot affect costs.
fn pooled_streams<R: Send, T: Sync>(
    streams: &[&[T]],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    init: impl Fn() -> R + Sync,
    accumulate: impl Fn(&mut R, &[T]) + Sync,
    mut merge: impl FnMut(&mut R, R),
) -> R {
    let lanes = pool.num_threads().min(streams.len()).max(1);
    if lanes == 1 {
        let mut acc = init();
        for tokens in streams {
            accumulate(&mut acc, tokens);
        }
        return acc;
    }
    let next_stream = AtomicUsize::new(0);
    let parts = pool.steal_map(scratch, lanes, |_, _scratch| {
        let mut acc = init();
        while let Some(tokens) = streams.get(next_stream.fetch_add(1, Ordering::Relaxed)) {
            accumulate(&mut acc, tokens);
        }
        acc
    });
    let mut parts = parts.into_iter();
    let mut acc = parts.next().expect("at least one stream lane");
    for part in parts {
        merge(&mut acc, part);
    }
    acc
}

/// `estimate_literal_and_run_bits` for one stream on the calling thread.
pub(in crate::lossless) fn estimate_literal_and_run_bits_single<T: LiteralToken>(
    tokens: &[T],
    num_contexts: usize,
    min_symbol: u32,
) -> (f64, f64) {
    let distance_context = num_contexts - 1;
    let mut literal = vec![Histogram::new(); num_contexts];
    let mut runs = vec![Histogram::new(); num_contexts];
    let (mut literal_bits, mut run_bits) = (0u64, 0u64);
    accumulate_literal_and_run_bits(
        tokens,
        distance_context,
        min_symbol,
        &mut literal,
        &mut runs,
        &mut literal_bits,
        &mut run_bits,
    );
    (
        histogram_bits(&literal, literal_bits),
        histogram_bits(&runs, run_bits),
    )
}

fn accumulate_literal_and_run_bits<T: LiteralToken>(
    tokens: &[T],
    distance_context: usize,
    min_symbol: u32,
    literal: &mut [Histogram],
    runs: &mut [Histogram],
    literal_bits: &mut u64,
    run_bits: &mut u64,
) {
    let mut start = 0usize;
    for (token, count) in TokenRuns(tokens) {
        // A run is equal values; its tokens may sit in different contexts.
        let run = &tokens[start..start + count];
        start += count;
        let (symbol, nbits, _) = crate::entropy::uint_encode(token.value);
        let context = token.context as usize;
        for member in run {
            literal[member.as_token().context as usize].add(symbol);
        }
        *literal_bits += nbits as u64 * count as u64;
        if count > LZ77_MIN_LENGTH as usize {
            runs[context].add(symbol);
            let (length, length_bits, _) = lz77_length_encode((count - 1) as u32 - LZ77_MIN_LENGTH);
            runs[context].add(min_symbol + length);
            let (distance, distance_bits, _) = crate::entropy::uint_encode(LZ77_DIST_VALUE);
            runs[distance_context].add(distance);
            *run_bits += nbits as u64 + length_bits as u64 + distance_bits as u64;
        } else {
            for member in run {
                runs[member.as_token().context as usize].add(symbol);
            }
            *run_bits += nbits as u64 * count as u64;
        }
    }
}

/// Exact order-0 estimates for both variants without storing the run stream.
/// Repeated literal symbols contribute their counts and extra bits in bulk;
/// floating-point entropy is evaluated only after integer histogram merging.
pub(in crate::lossless) fn estimate_literal_and_run_bits<T: LiteralToken>(
    streams: &[&[T]],
    num_contexts: usize,
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (f64, f64) {
    let distance_context = num_contexts - 1;
    let (literal, runs, literal_bits, run_bits) = pooled_streams(
        streams,
        pool,
        scratch,
        || {
            (
                vec![Histogram::new(); num_contexts],
                vec![Histogram::new(); num_contexts],
                0u64,
                0u64,
            )
        },
        |(literal, runs, literal_bits, run_bits), tokens| {
            accumulate_literal_and_run_bits(
                tokens,
                distance_context,
                min_symbol,
                literal,
                runs,
                literal_bits,
                run_bits,
            );
        },
        |acc, part| {
            merge_histograms(&mut acc.0, &part.0);
            merge_histograms(&mut acc.1, &part.1);
            acc.2 += part.2;
            acc.3 += part.3;
        },
    );
    (
        histogram_bits(&literal, literal_bits),
        histogram_bits(&runs, run_bits),
    )
}

/// Order-0 estimate of `streams` under their raw contexts (no clustering):
/// per-context token entropy plus raw bits. Consistent across stream
/// variants of the same frame, which is all the literal-vs-LZ77 choice needs.
pub(in crate::lossless) fn estimate_streams_bits<T: LzTokenSource>(
    streams: &[&[T]],
    num_contexts: usize,
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> f64 {
    let distance_context = (num_contexts - 1) as u32;
    let (hists, raw_bits) = pooled_streams(
        streams,
        pool,
        scratch,
        || (vec![Histogram::new(); num_contexts], 0u64),
        |acc, toks| {
            for t in lz_tokens(toks) {
                if t.is_lz77() {
                    let (len_tok, nb, _) = lz77_length_encode(t.value);
                    acc.0[t.context as usize].add(min_symbol + len_tok);
                    let (sym, dnb, _) = crate::entropy::uint_encode(t.distance);
                    acc.0[distance_context as usize].add(sym);
                    acc.1 += nb as u64 + dnb as u64;
                } else {
                    let (sym, nb, _) = crate::entropy::uint_encode(t.value);
                    acc.0[t.context as usize].add(sym);
                    acc.1 += nb as u64;
                }
            }
        },
        |acc, part| {
            merge_histograms(&mut acc.0, &part.0);
            acc.1 += part.1;
        },
    );
    histogram_bits(&hists, raw_bits)
}

fn histogram_bits(hists: &[Histogram], raw_bits: u64) -> f64 {
    let mut bits = raw_bits as f64;
    for h in hists {
        let total = h.total_count as f64;
        if total == 0.0 {
            continue;
        }
        for &c in &h.counts {
            if c != 0 {
                bits += c as f64 * f_log2(total / c as f64);
            }
        }
    }
    bits
}

/// Bits `streams` would take under `code`, from its ANS probabilities
/// (prefix depths when the bundle is a prefix code) plus raw bits. Within
/// ~0.5% of the written size, and cheap enough to decide which of several
/// stream variants to write.
pub(in crate::lossless) fn estimate_coded_bits<T: LzTokenSource>(
    streams: &[&[T]],
    distance_context: u32,
    code: &EntropyCode<'_>,
    min_symbol: u32,
    pool: Option<&ThreadPool>,
) -> f64 {
    let min_symbol = code.lz_min_symbol.max(min_symbol);
    let num_clusters = code.num_prefix_codes;
    let mut sym_cost: Vec<Vec<f32>> = Vec::with_capacity(num_clusters);
    for cluster in 0..num_clusters {
        if code.use_prefix_code {
            sym_cost.push(
                code.prefix_codes[cluster]
                    .depths
                    .iter()
                    .map(|&d| d as f32)
                    .collect(),
            );
        } else {
            let freqs = &code.ans_histograms[cluster].freqs;
            sym_cost.push(
                freqs
                    .iter()
                    .map(|&f| {
                        if f == 0 {
                            crate::entropy::ANS_LOG_TAB_SIZE as f32
                        } else {
                            crate::entropy::ANS_LOG_TAB_SIZE as f32 - dirty_log2f(f as f32)
                        }
                    })
                    .collect(),
            );
        }
    }
    let sym_cost = &sym_cost;
    let context_map = code.context_map;
    let configs = code.hybrid_uint_configs;
    let dist_cluster = context_map[distance_context as usize] as usize;
    parallel_streams(
        streams,
        pool,
        || 0.0f64,
        |acc, toks| {
            let mut bits = 0.0f64;
            for t in lz_tokens(toks) {
                let cluster = context_map[t.context as usize] as usize;
                if t.is_lz77() {
                    let (len_tok, nb, _) = lz77_length_encode(t.value);
                    bits += sym_cost[cluster][(min_symbol + len_tok) as usize] as f64 + nb as f64;
                    let (sym, dnb, _) =
                        crate::entropy::uint_encode_with_config(t.distance, configs[dist_cluster]);
                    bits += sym_cost[dist_cluster][sym as usize] as f64 + dnb as f64;
                } else {
                    let (sym, nb, _) =
                        crate::entropy::uint_encode_with_config(t.value, configs[cluster]);
                    bits += sym_cost[cluster][sym as usize] as f64 + nb as f64;
                }
            }
            *acc += bits;
        },
        |acc, part| *acc += part,
    )
}

/// Map streams on the shared pool, preserving stream order.
fn map_streams<R: Send, T: LzTokenSource>(
    streams: &[&[T]],
    pool: Option<&ThreadPool>,
    f: impl Fn(&[T]) -> R + Sync,
) -> Vec<R> {
    map_streams_indexed(streams, pool, |_, toks| f(toks))
}

fn map_streams_indexed<R: Send, T: LzTokenSource>(
    streams: &[&[T]],
    pool: Option<&ThreadPool>,
    f: impl Fn(usize, &[T]) -> R + Sync,
) -> Vec<R> {
    let threads = pool
        .map_or(1, ThreadPool::num_threads)
        .min(streams.len().max(1));
    if threads <= 1 {
        return streams.iter().enumerate().map(|(i, t)| f(i, t)).collect();
    }
    let chunk = streams.len().div_ceil(threads);
    pool.expect("parallel stream pool")
        .map(streams.len().div_ceil(chunk), |ci| {
            let begin = ci * chunk;
            let part = &streams[begin..(begin + chunk).min(streams.len())];
            part.iter()
                .enumerate()
                .map(|(k, t)| f(begin + k, t))
                .collect::<Vec<R>>()
        })
        .into_iter()
        .flatten()
        .collect()
}

fn merge_histograms(into: &mut [Histogram], from: &[Histogram]) {
    for (a, b) in into.iter_mut().zip(from) {
        for (x, y) in a.counts.iter_mut().zip(&b.counts) {
            *x += y;
        }
        a.total_count += b.total_count;
    }
}

/// `build_lz_pixel_code_opts` with parallel token passes and shared encoder
/// workers for ANS clustering. Chunk boundaries and merge order remain fixed;
/// the order-dependent hybrid-uint sampling stays sequential.
#[allow(clippy::too_many_arguments)]
pub(in crate::lossless) fn build_lz_pixel_code_threads<
    'tokens,
    'scratch,
    I,
    T: LzTokenSource + 'tokens,
>(
    thorough: bool,
    streams: I,
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
    ans_cluster: bool,
    pool: Option<&ThreadPool>,
    scratch: &'scratch mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> EntropyCode<'scratch>
where
    I: Iterator<Item = &'tokens [T]> + Clone,
{
    let streams: Vec<&'tokens [T]> = streams.collect();
    build_lz_pixel_code_slices(
        thorough,
        &streams,
        &[],
        nb_chans,
        min_symbol,
        refined,
        ans_cluster,
        false,
        pool,
        scratch,
        huffman_pool,
    )
}

/// `build_lz_pixel_code_threads` for streams that each end in a constant
/// tail (`tails[k]` follows `streams[k]`; `ConstantTail::NONE` for none).
#[allow(clippy::too_many_arguments)]
pub(in crate::lossless) fn build_lz_pixel_code_tails<
    'tokens,
    'scratch,
    T: LzTokenSource + 'tokens,
>(
    thorough: bool,
    streams: &[&'tokens [T]],
    tails: &[ConstantTail],
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
    ans_cluster: bool,
    pool: Option<&ThreadPool>,
    scratch: &'scratch mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> EntropyCode<'scratch> {
    debug_assert_eq!(streams.len(), tails.len());
    build_lz_pixel_code_slices(
        thorough,
        streams,
        tails,
        nb_chans,
        min_symbol,
        refined,
        ans_cluster,
        false,
        pool,
        scratch,
        huffman_pool,
    )
}

/// The tails' contexts and symbol-0 counts, one entry per distinct context.
fn tail_counts(tails: &[ConstantTail]) -> Vec<(u32, u32)> {
    let mut counts: Vec<(u32, u32)> = Vec::new();
    for tail in tails.iter().filter(|tail| tail.count > 0) {
        let count = u32::try_from(tail.count).expect("constant tail count");
        match counts
            .iter_mut()
            .find(|(context, _)| *context == tail.context)
        {
            Some((_, total)) => *total += count,
            None => counts.push((tail.context, count)),
        }
    }
    counts
}

/// Renumber clusters in order of first use over the context ids (the
/// clusterer's convention) after a cluster was appended out of order.
fn compact_clusters_first_use(
    context_map: &mut [u8],
    histograms: &mut [Histogram],
    num_clusters: usize,
) -> usize {
    let mut remap = vec![u8::MAX; num_clusters];
    let mut next = 0u8;
    for entry in context_map.iter_mut() {
        let old = *entry as usize;
        if remap[old] == u8::MAX {
            remap[old] = next;
            next += 1;
        }
        *entry = remap[old];
    }
    let old_hists: Vec<Histogram> = histograms[..num_clusters].to_vec();
    for (old, hist) in old_hists.into_iter().enumerate() {
        if remap[old] != u8::MAX {
            histograms[remap[old] as usize] = hist;
        }
    }
    next as usize
}

fn rebuild_lz_histograms<T: LzTokenSource>(
    streams: &[&[T]],
    context_map: &[u8],
    min_symbol: u32,
    tail_counts: &[(u32, u32)],
    pool: Option<&ThreadPool>,
    configs: &mut [HybridUintConfig],
    histograms: &mut [Histogram],
    all: bool,
) {
    // `all`: `min_symbol` is not the one the histograms were counted with,
    // so every cluster's length symbols move.
    let rebuild: Vec<_> = configs
        .iter()
        .map(|&config| all || config != HybridUintConfig::DEFAULT)
        .collect();
    if !rebuild.contains(&true) {
        return;
    }

    // Keep the clustered default populations until a replacement is both
    // complete and valid. Validate literals while counting their new symbols.
    let distance_cluster = context_map[context_map.len() - 1] as usize;
    let configs_ref: &[HybridUintConfig] = configs;
    let (new_histograms, valid) = parallel_streams(
        streams,
        pool,
        || (vec![Histogram::new(); configs_ref.len()], rebuild.clone()),
        |(acc, valid), toks| {
            for tok in lz_tokens(toks) {
                let cluster = context_map[tok.context as usize] as usize;
                if tok.is_lz77() {
                    if valid[cluster] {
                        let (length, _, _) = lz77_length_encode(tok.value);
                        acc[cluster].add(min_symbol + length);
                    }
                    if valid[distance_cluster] {
                        let (symbol, _, _) =
                            uint_encode_with_config(tok.distance, configs_ref[distance_cluster]);
                        acc[distance_cluster].add(symbol);
                    }
                } else if valid[cluster] {
                    let (symbol, _, _) = uint_encode_with_config(tok.value, configs_ref[cluster]);
                    if symbol < min_symbol {
                        acc[cluster].add(symbol);
                    } else {
                        valid[cluster] = false;
                    }
                }
            }
        },
        |(acc, valid), (part, part_valid)| {
            merge_histograms(acc, &part);
            for (a, b) in valid.iter_mut().zip(part_valid) {
                *a &= b;
            }
        },
    );

    for (cluster, hist) in new_histograms.into_iter().enumerate() {
        if valid[cluster] {
            histograms[cluster] = hist;
        } else {
            configs[cluster] = HybridUintConfig::DEFAULT;
        }
    }
    for &(context, count) in tail_counts {
        let cluster = context_map[context as usize] as usize;
        if valid[cluster] {
            histograms[cluster].counts[0] += count;
            histograms[cluster].total_count += count;
        }
    }
    // A cluster sent back to the default configuration still holds the
    // histogram of the old `min_symbol`: count it again. The default is
    // valid under any `min_symbol` the callers pass.
    if all
        && rebuild
            .iter()
            .zip(&valid)
            .any(|(&wanted, &kept)| wanted && !kept)
    {
        rebuild_lz_histograms(
            streams,
            context_map,
            min_symbol,
            tail_counts,
            pool,
            configs,
            histograms,
            true,
        );
    }
}

// Share entropy setup across iterator adapters while retaining specialized
// token passes for each token representation.
#[allow(clippy::too_many_arguments)]
fn build_lz_pixel_code_slices<'scratch, T: LzTokenSource>(
    thorough: bool,
    streams: &[&[T]],
    tails: &[ConstantTail],
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
    ans_cluster: bool,
    ans: bool,
    pool: Option<&ThreadPool>,
    scratch: &'scratch mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> EntropyCode<'scratch> {
    let thorough = cfg!(any(feature = "extraslow", test)) && thorough;
    let distance_context = nb_chans as u32;
    let num_contexts = nb_chans + 1;
    assert!(num_contexts <= LZ77_MAX_CONTEXTS);
    let use_ans = refined || ans;
    let LzEntropyScratch {
        histograms,
        prefix_codes,
        context_map,
        configs,
        clustering,
        ans,
    } = scratch;
    grow_entropy_scratch(histograms, num_contexts, Histogram::new());
    let histograms = &mut histograms[..num_contexts];
    histograms.fill(Histogram::new());
    {
        let summed = parallel_streams(
            streams,
            pool,
            || vec![Histogram::new(); num_contexts],
            |acc, toks| {
                lz_add_histograms(lz_tokens(toks), acc, min_symbol, distance_context);
            },
            |acc, part| merge_histograms(acc, &part),
        );
        histograms.clone_from_slice(&summed);
    }

    // A constant tail's context is clustered alone: its stored tokens (if
    // any) plus the tail form a histogram the clusterer never sees, appended
    // as its own cluster afterwards. Sharing a cluster would cost the tail
    // bits in the other contexts' symbols and the decoder its constant fill.
    let tail_counts = tail_counts(tails);
    let pinned: Vec<(u32, Histogram)> = tail_counts
        .iter()
        .map(|&(context, count)| {
            let mut hist = std::mem::take(&mut histograms[context as usize]);
            hist.counts[0] += count;
            hist.total_count += count;
            (context, hist)
        })
        .collect();

    // The clusterer leaves the pinned contexts' clusters free.
    let cluster_limit = if ans_cluster {
        crate::entropy::ANS_CLUSTERS_LIMIT
    } else {
        crate::entropy::CLUSTERS_LIMIT
    };
    let max_clusters = cluster_limit - pinned.len();
    let mut num_clusters = if ans_cluster {
        crate::entropy::cluster_histograms_ans(
            histograms,
            &mut context_map[..num_contexts],
            pool,
            false,
            max_clusters,
            if thorough { 6 } else { 2 },
        )
    } else {
        cluster_histograms_fixed(
            histograms,
            &mut context_map[..num_contexts],
            refined,
            max_clusters,
            clustering,
            huffman_pool,
        )
    };
    if !pinned.is_empty() {
        for (context, hist) in pinned {
            assert!(
                num_clusters < histograms.len(),
                "a cluster slot for the constant tail"
            );
            histograms[num_clusters] = hist;
            context_map[context as usize] = num_clusters as u8;
            num_clusters += 1;
        }
        num_clusters =
            compact_clusters_first_use(&mut context_map[..num_contexts], histograms, num_clusters);
    }
    let histograms = &mut histograms[..num_clusters];
    let configs = &mut configs[..num_clusters];
    let mut lz_min_symbol = min_symbol;

    if refined {
        let distance_cluster = context_map[distance_context as usize] as usize;
        let context_map_ref: &[u8] = &context_map[..num_contexts];
        {
            let per_stream_counts: Vec<Vec<usize>> = map_streams(streams, pool, |toks| {
                let mut c = vec![0usize; num_clusters];
                for tok in lz_tokens(toks) {
                    let cluster = if tok.is_lz77() {
                        distance_cluster
                    } else {
                        context_map_ref[tok.context as usize] as usize
                    };
                    c[cluster] += 1;
                }
                c
            });
            // Counts provide both stream-local ordinals and cluster totals; no
            // separate full-token pass is needed to select the sampling strides.
            let mut offsets: Vec<Vec<usize>> = Vec::with_capacity(streams.len());
            let mut running = vec![0usize; num_clusters];
            for c in &per_stream_counts {
                offsets.push(running.clone());
                for (r, &n) in running.iter_mut().zip(c) {
                    *r += n;
                }
            }
            let strides = crate::entropy::HybridUintSamples::strides_for(&running);
            let offsets = &offsets;
            let strides_ref = &strides;
            let per_stream_values: Vec<(Vec<Vec<u32>>, Vec<u32>)> =
                map_streams_indexed(streams, pool, |s, toks| {
                    let mut remaining: Vec<_> = offsets[s]
                        .iter()
                        .zip(strides_ref)
                        .map(|(&offset, &stride)| {
                            let phase = offset % stride;
                            if phase == 0 { 0 } else { stride - phase }
                        })
                        .collect();
                    let mut local: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
                    let mut maxima = vec![0u32; num_clusters];
                    for tok in lz_tokens(toks) {
                        let (cluster, value) = if tok.is_lz77() {
                            (distance_cluster, tok.distance)
                        } else {
                            (context_map_ref[tok.context as usize] as usize, tok.value)
                        };
                        maxima[cluster] = maxima[cluster].max(value);
                        if remaining[cluster] == 0 {
                            local[cluster].push(value);
                            remaining[cluster] = strides_ref[cluster] - 1;
                        } else {
                            remaining[cluster] -= 1;
                        }
                    }
                    (local, maxima)
                });
            let mut values: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
            let mut maxima = vec![0u32; num_clusters];
            for (local, local_maxima) in per_stream_values {
                for (dst, src) in values.iter_mut().zip(local) {
                    dst.extend(src);
                }
                for (dst, src) in maxima.iter_mut().zip(local_maxima) {
                    *dst = (*dst).max(src);
                }
            }
            let samples = crate::entropy::HybridUintSamples::from_parts(values, strides);
            configs.copy_from_slice(&if thorough {
                match pool {
                    Some(pool) => samples.select_exhaustive_with_pool(
                        min_symbol,
                        pool,
                        &mut CoderScratch::lossless(),
                    ),
                    None => samples.select_exhaustive(min_symbol),
                }
            } else {
                match pool {
                    Some(pool) => {
                        samples.select_with_pool(pool, &mut CoderScratch::lossless(), 0.995)
                    }
                    None => samples.select(),
                }
            });
            // Any configuration whose literals stay below the first LZ77
            // symbol, beyond the fixed list searched above; then the same
            // with that symbol raised, kept when it codes smaller.
            let (free, free_bits) = samples.select_free(
                configs,
                &maxima,
                min_symbol,
                pool,
                &mut CoderScratch::lossless(),
            );
            if LZ77_RAISED_MIN_SYMBOL > min_symbol {
                let (raised, raised_bits) = samples.select_free(
                    configs,
                    &maxima,
                    LZ77_RAISED_MIN_SYMBOL,
                    pool,
                    &mut CoderScratch::lossless(),
                );
                if raised_bits < free_bits {
                    configs.copy_from_slice(&raised);
                    lz_min_symbol = LZ77_RAISED_MIN_SYMBOL;
                }
            }
            if lz_min_symbol == min_symbol {
                configs.copy_from_slice(&free);
            }
        }
        rebuild_lz_histograms(
            streams,
            context_map_ref,
            lz_min_symbol,
            &tail_counts,
            pool,
            configs,
            histograms,
            lz_min_symbol != min_symbol,
        );
    } else {
        configs.fill(crate::entropy::HybridUintConfig::DEFAULT);
    }

    finish_lz_entropy(
        histograms,
        prefix_codes,
        &context_map[..num_contexts],
        configs,
        ans,
        huffman_pool,
        use_ans,
        lz_min_symbol,
    )
}

// Code construction no longer depends on the input token representation.
fn finish_lz_entropy<'a>(
    histograms: &[Histogram],
    prefix_codes: &'a mut Vec<PrefixCode>,
    context_map: &'a [u8],
    configs: &'a [crate::entropy::HybridUintConfig],
    ans: &'a mut crate::coder_scratch::LzAnsScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    use_ans: bool,
    lz_min_symbol: u32,
) -> EntropyCode<'a> {
    let num_contexts = context_map.len();
    let num_clusters = histograms.len();
    grow_entropy_scratch(prefix_codes, num_clusters, PrefixCode::zero());
    let prefix_codes = &mut prefix_codes[..num_clusters];
    build_huffman_codes_into(histograms, prefix_codes, huffman_pool);

    // Apply the single-symbol patch (mirrors build_pixel_code) per cluster so
    // that contexts with one unique symbol still emit a parseable code.
    for pc in prefix_codes.iter_mut() {
        let mut nonzero = 0;
        let mut idx = 0;
        for (i, &d) in pc.depths.iter().enumerate() {
            if d != 0 {
                nonzero += 1;
                idx = i;
                if nonzero > 1 {
                    break;
                }
            }
        }
        if nonzero == 1 {
            if idx == 0 {
                pc.depths[idx] = 0;
                pc.bits[idx] = 0;
            } else {
                pc.depths[0] = 1;
                pc.bits[0] = 0;
                pc.depths[idx] = 1;
                pc.bits[idx] = 1;
            }
        }
    }
    if use_ans {
        let (hists, symbols, reverse_maps) = crate::entropy::build_ans_code_parts(histograms, true);
        ans.histograms = hists;
        ans.symbols = symbols;
        ans.reverse_maps = reverse_maps;
    } else {
        ans.histograms.clear();
        ans.symbols.clear();
        ans.reverse_maps.clear();
    }
    EntropyCode {
        context_map: &context_map[..num_contexts],
        num_contexts,
        prefix_codes,
        hybrid_uint_configs: configs,
        num_prefix_codes: num_clusters,
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: !use_ans,
        ans_histograms: &ans.histograms,
        ans_symbols: &ans.symbols,
        ans_reverse_maps: &ans.reverse_maps,
        lz_min_symbol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refined_histograms_keep_default_clusters_when_a_literal_is_invalid() {
        let first = [
            LzToken::pixel(0, 31),
            LzToken::lz77(0, 20, 150),
            LzToken::pixel(1, 12),
            LzToken::pixel(2, 3),
        ];
        let second = [LzToken::pixel(1, 64), LzToken::pixel(0, 60)];
        let streams = [first.as_slice(), second.as_slice()];
        let mut defaults = vec![Histogram::new(); 4];
        lz_add_histograms(first.into_iter().chain(second), &mut defaults, 64, 3);
        let distance = defaults.pop().unwrap();
        merge_histograms(&mut defaults[1..2], &[distance]);
        let tails = [(0, 5), (2, 7)];
        for (context, count) in tails {
            defaults[context as usize].counts[0] += count;
            defaults[context as usize].total_count += count;
        }
        let accepted = HybridUintConfig {
            split_exponent: 5,
            msb_in_token: 0,
            lsb_in_token: 0,
        };
        let rejected = HybridUintConfig {
            split_exponent: 6,
            msb_in_token: 1,
            lsb_in_token: 0,
        };
        let pool = ThreadPool::new_lossless(2);
        for pool in [None, Some(&pool)] {
            let mut histograms = defaults.clone();
            let mut configs = [accepted, rejected, HybridUintConfig::DEFAULT];
            rebuild_lz_histograms(
                &streams,
                &[0, 1, 2, 1],
                64,
                &tails,
                pool,
                &mut configs,
                &mut histograms,
                false,
            );
            assert_eq!(
                configs,
                [
                    accepted,
                    HybridUintConfig::DEFAULT,
                    HybridUintConfig::DEFAULT
                ],
            );
            let mut expected = Histogram::new();
            expected.counts[0] = 5;
            expected.total_count = 5;
            for symbol in [31, 32, 80] {
                expected.add(symbol);
            }
            assert_eq!(histograms[0].counts, expected.counts);
            assert_eq!(histograms[0].total_count, expected.total_count);
            for cluster in [1, 2] {
                assert_eq!(histograms[cluster].counts, defaults[cluster].counts);
                assert_eq!(
                    histograms[cluster].total_count,
                    defaults[cluster].total_count
                );
            }
        }
    }
    #[test]
    fn a_raised_min_symbol_recounts_every_cluster() {
        // Three token bits beyond the exponent put 1200 at symbol 65: over
        // the default first LZ77 symbol, under the raised one.
        let wide = HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 3,
            lsb_in_token: 0,
        };
        assert_eq!(crate::entropy::uint_encode_with_config(1200, wide).0, 65);
        let tokens = [
            LzToken::pixel(0, 1200),
            LzToken::lz77(0, 20, 150),
            LzToken::pixel(1, 5),
            LzToken::lz77(1, 3, 1),
        ];
        let streams = [tokens.as_slice()];
        let mut defaults = vec![Histogram::new(); 3];
        lz_add_histograms(tokens, &mut defaults, 64, 2);
        let length_20 = lz77_length_encode(20).0;
        let length_3 = lz77_length_encode(3).0;
        assert_eq!(defaults[1].counts[(64 + length_3) as usize], 1);

        // At the default symbol the wide cluster falls back and nothing moves.
        let mut histograms = defaults.clone();
        let mut configs = [wide, HybridUintConfig::DEFAULT, HybridUintConfig::DEFAULT];
        rebuild_lz_histograms(
            &streams,
            &[0, 1, 2],
            64,
            &[],
            None,
            &mut configs,
            &mut histograms,
            false,
        );
        assert_eq!(configs[0], HybridUintConfig::DEFAULT);
        assert_eq!(histograms[0].counts, defaults[0].counts);

        // Raised: the wide literal is kept, and the lengths of both clusters,
        // the default-configured one included, sit above the new symbol.
        let mut histograms = defaults.clone();
        let mut configs = [wide, HybridUintConfig::DEFAULT, HybridUintConfig::DEFAULT];
        rebuild_lz_histograms(
            &streams,
            &[0, 1, 2],
            LZ77_RAISED_MIN_SYMBOL,
            &[],
            None,
            &mut configs,
            &mut histograms,
            true,
        );
        assert_eq!(configs[0], wide);
        let mut first = Histogram::new();
        first.add(65);
        first.add(LZ77_RAISED_MIN_SYMBOL + length_20);
        assert_eq!(histograms[0].counts, first.counts);
        let mut second = Histogram::new();
        second.add(5);
        second.add(LZ77_RAISED_MIN_SYMBOL + length_3);
        assert_eq!(histograms[1].counts, second.counts);
        assert_eq!(histograms[2].counts, defaults[2].counts);
    }

    #[test]
    fn every_length_symbol_fits_above_the_raised_min_symbol() {
        let longest = lz77_length_encode(u32::MAX).0;
        assert!(LZ77_RAISED_MIN_SYMBOL + longest < crate::entropy::ALPHABET_SIZE as u32);
    }
}
