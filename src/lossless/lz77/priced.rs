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

//! Entropy-priced LZ77 match search.

use super::{
    LZ77_DIST_VALUE, LZ77_MIN_LENGTH, LZ77_MIN_SYMBOL, LZ77_NUM_SPECIAL_DISTANCES, LZ77_WINDOW,
    LiteralToken, LzToken, lz77_length_encode,
};
use crate::adaptive_quant::{dirty_log2f, dirty_log2p1f};
use crate::entropy::{Token, f_log2};

const PRICE_SYMBOLS: usize = crate::entropy::ALPHABET_SIZE;

/// Literal price in bits of every (context, symbol) pair under the order-0
/// statistics of the streams themselves.
pub(in crate::lossless) struct LiteralPrices {
    pub(super) bits: Vec<f32>,
}

impl LiteralPrices {
    pub(in crate::lossless) fn new<T: LiteralToken>(streams: &[&[T]], num_contexts: usize) -> Self {
        let mut counts = vec![0u32; num_contexts * PRICE_SYMBOLS];
        let mut totals = vec![0u32; num_contexts];
        for stream in streams {
            for token in stream.iter() {
                let token = token.as_token();
                let (symbol, _, _) = crate::entropy::uint_encode(token.value);
                counts[token.context as usize * PRICE_SYMBOLS + symbol as usize] += 1;
                totals[token.context as usize] += 1;
            }
        }
        let bits = counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                if count == 0 {
                    0.0
                } else {
                    (totals[i / PRICE_SYMBOLS] as f32 / count as f32).log2()
                }
            })
            .collect();
        Self { bits }
    }

    #[inline]
    pub(super) fn of(&self, token: Token) -> f32 {
        let (symbol, nbits, _) = crate::entropy::uint_encode(token.value);
        self.bits[token.context as usize * PRICE_SYMBOLS + symbol as usize] + nbits as f32
    }
}

/// Entropy-coded part of a match's length and distance tokens before any
/// match has been seen.
const PRICED_LENGTH_BITS: f32 = 6.0;

const PRICED_DISTANCE_BITS: f32 = 6.0;

/// A match must beat the literals it replaces by this many bits.
pub(super) const PRICED_MIN_GAIN: f32 = 2.0;

const PRICED_RECENT_DISTANCES: usize = 4;

/// At this many tokens, and at every doubling, the matches must have saved
/// this share of the literal bits or the stream stays literal.
const PRICED_PROBE_TOKENS: usize = 1 << 17;

const PRICED_PROBE_MIN_SAVING: f64 = 0.1;

/// Longest match of the priced parse; bounds the work per candidate.
const PRICED_MAX_LENGTH: usize = 4096;

/// A rejected match of at least this length stops the search over its
/// first half.
const PRICED_SKIP_MIN_LENGTH: usize = 16;

pub(in crate::lossless) const PRICED_MAX_PROBES: usize = 32;

const PRICED_GRAM: usize = 6;

const PRICED_HASH_BITS: u32 = 16;

const PRICED_NONE: u32 = u32::MAX;

/// Prices of a match's two tokens. The first parse has only the defaults;
/// a second one prices them by what the first one emitted.
pub(in crate::lossless) struct MatchPrices {
    pub(super) length: f32,
    pub(super) distance: [f32; PRICE_SYMBOLS],
}

impl MatchPrices {
    pub(in crate::lossless) fn initial() -> Self {
        Self {
            length: PRICED_LENGTH_BITS,
            distance: [PRICED_DISTANCE_BITS; PRICE_SYMBOLS],
        }
    }

    /// Order-0 prices of the match tokens in `streams`: the mean price of a
    /// length symbol inside its literal context, and each distance symbol's
    /// price in the distance context (an unseen one costs a first occurrence).
    pub(in crate::lossless) fn learned(streams: &[&[LzToken]], num_contexts: usize) -> Self {
        let mut counts = vec![0u32; num_contexts * PRICE_SYMBOLS];
        let mut totals = vec![0u32; num_contexts];
        let mut distances = [0u32; PRICE_SYMBOLS];
        let mut matches = 0u32;
        for stream in streams {
            for token in stream.iter() {
                let context = token.context as usize;
                totals[context] += 1;
                if token.is_lz77() {
                    let (symbol, _, _) = lz77_length_encode(token.value);
                    counts[context * PRICE_SYMBOLS + (LZ77_MIN_SYMBOL + symbol) as usize] += 1;
                    let (symbol, _, _) = crate::entropy::uint_encode(token.distance);
                    distances[symbol as usize] += 1;
                    matches += 1;
                }
            }
        }
        if matches == 0 {
            return Self::initial();
        }
        let mut length = 0f64;
        for context in 0..num_contexts {
            for symbol in LZ77_MIN_SYMBOL as usize..PRICE_SYMBOLS {
                let count = counts[context * PRICE_SYMBOLS + symbol];
                if count != 0 {
                    length += count as f64 * f_log2(totals[context] as f64 / count as f64);
                }
            }
        }
        let unseen = dirty_log2p1f(matches as f32);
        Self {
            length: (length / matches as f64) as f32,
            distance: distances.map(|count| {
                if count == 0 {
                    unseen
                } else {
                    dirty_log2f(matches as f32 / count as f32)
                }
            }),
        }
    }

    #[inline]
    pub(super) fn of(&self, len: usize, distance: usize) -> f32 {
        self.of_distance(distance) + Self::length_bits(len)
    }

    /// Price of a match at `distance` before its length's raw bits.
    #[inline]
    fn of_distance(&self, distance: usize) -> f32 {
        let (symbol, distance_bits, _) = crate::entropy::uint_encode(lz77_distance_value(distance));
        self.length + self.distance[symbol as usize] + distance_bits as f32
    }

    /// Raw bits of a length token.
    #[inline]
    fn length_bits(len: usize) -> f32 {
        let (_, length_bits, _) = lz77_length_encode(len as u32 - LZ77_MIN_LENGTH);
        length_bits as f32
    }
}

#[inline]
fn lz77_distance_value(distance: usize) -> u32 {
    if distance == 1 {
        LZ77_DIST_VALUE
    } else {
        LZ77_NUM_SPECIAL_DISTANCES + distance as u32 - 1
    }
}

#[inline]
fn priced_hash<T: LiteralToken>(tokens: &[T], pos: usize) -> usize {
    let mut hash = 0x9e37_79b9u32;
    for token in &tokens[pos..pos + PRICED_GRAM] {
        hash = (hash ^ token.as_token().value).wrapping_mul(0x85eb_ca6b);
        hash ^= hash >> 15;
    }
    (hash.wrapping_mul(0xc2b2_ae35) >> (32 - PRICED_HASH_BITS)) as usize
}

/// LZ77 parse that takes a match only where it is cheaper than the literals
/// it replaces: under a learned tree most literals cost a fraction of a bit,
/// and a match of fixed minimum length loses to them. Candidates come from a
/// hash chain over `PRICED_GRAM` values and from the distances of the latest
/// matches, which follow a period through its interruptions. `None` when the
/// stream shows no saving.
pub(in crate::lossless) fn lz77_compress_priced<T: LiteralToken>(
    tokens: &[T],
    prices: &LiteralPrices,
    match_prices: &MatchPrices,
    max_probes: usize,
    scratch: &mut Vec<u32>,
) -> Option<Vec<LzToken>> {
    let n = tokens.len();
    let mut out: Vec<LzToken> = Vec::with_capacity(n.min(16 * 1024));
    let heads_len = 1usize << PRICED_HASH_BITS;
    scratch.clear();
    scratch.resize(heads_len + n, PRICED_NONE);
    let (heads, chain) = scratch.split_at_mut(heads_len);
    let match_kernel = T::match_kernel();
    // Literal bits of tokens[..k]: a span's price is one subtraction, and a
    // position whose longest possible match cannot pay is never searched.
    let mut prefix: Vec<f32> = Vec::with_capacity(n + 1);
    let mut sum = 0f64;
    prefix.push(0.0);
    for token in tokens {
        sum += prices.of(token.as_token()) as f64;
        prefix.push(sum as f32);
    }
    let span = |from: usize, len: usize| -> f32 { prefix[from + len] - prefix[from] };
    let cheapest_match = match_prices.of(LZ77_MIN_LENGTH as usize, 1);
    let insert = |heads: &mut [u32], chain: &mut [u32], pos: usize| {
        if pos + PRICED_GRAM <= n {
            let h = priced_hash(tokens, pos);
            chain[pos] = heads[h];
            heads[h] = pos as u32;
        }
    };
    let mut recent = [0usize; PRICED_RECENT_DISTANCES];
    // A long match that does not pay leaves its tail, a shorter match over
    // fewer literals, unpaid as well.
    let mut no_search_until = 0usize;
    let (mut literal_bits, mut saved_bits) = (0f64, 0f64);
    let mut probe_at = PRICED_PROBE_TOKENS.min(n / 2);
    let mut i = 0usize;
    while i < n {
        if i >= probe_at {
            probe_at = probe_at.saturating_mul(2);
            if saved_bits < PRICED_PROBE_MIN_SAVING * (literal_bits + saved_bits) {
                return None;
            }
        }
        let limit = (n - i).min(PRICED_MAX_LENGTH);
        let mut best = (0f32, 0usize, 0usize);
        let mut longest = 0usize;
        if i >= no_search_until && span(i, limit) - cheapest_match > PRICED_MIN_GAIN {
            let mut consider = |len: usize, distance: usize| {
                longest = longest.max(len);
                if len >= LZ77_MIN_LENGTH as usize {
                    let gain = span(i, len) - match_prices.of(len, distance);
                    if gain > best.0 {
                        best = (gain, len, distance);
                    }
                }
            };
            let measure = |from: usize| -> usize {
                if !tokens[from].same(tokens[i]) {
                    return 0;
                }
                1 + match_kernel(&tokens[from + 1..from + limit], &tokens[i + 1..i + limit])
            };
            for &distance in &recent {
                if distance != 0 && distance <= i {
                    consider(measure(i - distance), distance);
                }
            }
            if i + PRICED_GRAM <= n {
                let mut candidate = heads[priced_hash(tokens, i)];
                for _ in 0..max_probes {
                    if candidate == PRICED_NONE {
                        break;
                    }
                    let from = candidate as usize;
                    if i - from > LZ77_WINDOW {
                        break;
                    }
                    let len = measure(from);
                    consider(len, i - from);
                    if len == limit {
                        break;
                    }
                    candidate = chain[from];
                }
            }
        }
        let (gain, len, distance) = best;
        if gain > PRICED_MIN_GAIN {
            out.push(LzToken::lz77(
                tokens[i].as_token().context,
                len as u32 - LZ77_MIN_LENGTH,
                lz77_distance_value(distance),
            ));
            if let Some(at) = recent.iter().position(|&d| d == distance) {
                recent[..=at].rotate_right(1);
            } else {
                recent.rotate_right(1);
                recent[0] = distance;
            }
            for pos in i..i + len {
                insert(heads, chain, pos);
            }
            saved_bits += gain as f64;
            i += len;
        } else {
            if longest >= PRICED_SKIP_MIN_LENGTH {
                no_search_until = i + longest / 2;
            }
            let token = tokens[i].as_token();
            literal_bits += prices.of(token) as f64;
            out.push(LzToken::pixel(token.context, token.value));
            insert(heads, chain, i);
            i += 1;
        }
    }
    Some(out)
}

/// A match of at least this length is taken whole: every position it
/// covers would measure the same copy again.
const OPTIMAL_NICE_LENGTH: usize = 1024;

/// Last lengths of the length-token classes (`lz77_length_encode`), below
/// `PRICED_MAX_LENGTH`. The length price is flat inside a class, and from
/// any position a copy covers, the same copy continues at the same
/// distance, so coding past a class end is never cheaper than stopping
/// there: a match relaxes the parse at each class end it reaches and at
/// its own length only.
const OPTIMAL_CLASS_ENDS: [usize; 8] = [18, 34, 66, 130, 258, 514, 1026, 2050];

/// Edge into a position of the optimal parse: a literal, or a match of
/// `len` tokens at `distance` packed as `((len - 3) << 20 | (distance - 1)) + 1`.
const OPTIMAL_LITERAL: u32 = 0;

const OPTIMAL_DISTANCE_BITS: u32 = 20;

#[inline]
fn optimal_edge(len: usize, distance: usize) -> u32 {
    debug_assert!(len >= LZ77_MIN_LENGTH as usize && len <= PRICED_MAX_LENGTH);
    debug_assert!((1..=LZ77_WINDOW).contains(&distance));
    (((len as u32 - LZ77_MIN_LENGTH) << OPTIMAL_DISTANCE_BITS) | (distance as u32 - 1)) + 1
}

#[inline]
fn optimal_edge_match(edge: u32) -> (usize, usize) {
    let packed = edge - 1;
    (
        (packed >> OPTIMAL_DISTANCE_BITS) as usize + LZ77_MIN_LENGTH as usize,
        (packed & ((1 << OPTIMAL_DISTANCE_BITS) - 1)) as usize + 1,
    )
}

/// `lz77_compress_priced` as a shortest path over token positions: every
/// position relaxes its literal and each candidate match at the lengths
/// `OPTIMAL_CLASS_ENDS` admits, and the cheapest path is read back from
/// the end. Candidates, prices and the saving gates are the greedy
/// parse's; a match of `OPTIMAL_NICE_LENGTH` is taken whole as there.
pub(in crate::lossless) fn lz77_compress_optimal<T: LiteralToken>(
    tokens: &[T],
    prices: &LiteralPrices,
    match_prices: &MatchPrices,
    max_probes: usize,
    scratch: &mut Vec<u32>,
) -> Option<Vec<LzToken>> {
    let n = tokens.len();
    let heads_len = 1usize << PRICED_HASH_BITS;
    scratch.clear();
    scratch.resize(heads_len + n + n + 1, PRICED_NONE);
    let (heads, rest) = scratch.split_at_mut(heads_len);
    let (chain, edges) = rest.split_at_mut(n);
    let match_kernel = T::match_kernel();
    let mut prefix: Vec<f32> = Vec::with_capacity(n + 1);
    let mut sum = 0f64;
    prefix.push(0.0);
    for token in tokens {
        sum += prices.of(token.as_token()) as f64;
        prefix.push(sum as f32);
    }
    let span = |from: usize, len: usize| -> f32 { prefix[from + len] - prefix[from] };
    // Path costs in f64: a stream runs to millions of bits and the edges
    // differ by fractions of one.
    let mut cost = vec![f64::INFINITY; n + 1];
    cost[0] = 0.0;
    let cheapest_match = match_prices.of(LZ77_MIN_LENGTH as usize, 1);
    let insert = |heads: &mut [u32], chain: &mut [u32], pos: usize| {
        if pos + PRICED_GRAM <= n {
            let h = priced_hash(tokens, pos);
            chain[pos] = heads[h];
            heads[h] = pos as u32;
        }
    };
    let mut recent = [0usize; PRICED_RECENT_DISTANCES];
    let mut no_search_until = 0usize;
    let mut probe_at = PRICED_PROBE_TOKENS.min(n / 2);
    let mut i = 0usize;
    while i < n {
        let here = cost[i];
        // Inside a match taken whole: no path continues from here.
        if here.is_infinite() {
            insert(heads, chain, i);
            i += 1;
            continue;
        }
        if i >= probe_at {
            probe_at = probe_at.saturating_mul(2);
            // A copy over this position has reached only its ends: the
            // saving so far is the best one within a copy's length.
            let window = i.saturating_sub(PRICED_MAX_LENGTH)..=(i + PRICED_MAX_LENGTH).min(n);
            let saved = window
                .map(|pos| prefix[pos] as f64 - cost[pos])
                .fold(0.0, f64::max);
            if saved < PRICED_PROBE_MIN_SAVING * prefix[i] as f64 {
                return None;
            }
        }
        let token = tokens[i].as_token();
        let literal = here + prices.of(token) as f64;
        if literal < cost[i + 1] {
            cost[i + 1] = literal;
            edges[i + 1] = OPTIMAL_LITERAL;
        }
        let limit = (n - i).min(PRICED_MAX_LENGTH);
        let mut longest = 0usize;
        let mut best = (0f32, 0usize, 0usize);
        if i >= no_search_until && span(i, limit) - cheapest_match > PRICED_MIN_GAIN {
            let mut relax = |len: usize, distance: usize| {
                longest = longest.max(len);
                if len < LZ77_MIN_LENGTH as usize {
                    return;
                }
                let gain = span(i, len) - match_prices.of(len, distance) - PRICED_MIN_GAIN;
                if gain > best.0 {
                    best = (gain, len, distance);
                }
                let base = here + (match_prices.of_distance(distance) + PRICED_MIN_GAIN) as f64;
                let mut relax_at = |len: usize| {
                    let price = base + MatchPrices::length_bits(len) as f64;
                    if price < cost[i + len] {
                        cost[i + len] = price;
                        edges[i + len] = optimal_edge(len, distance);
                    }
                };
                for &end in OPTIMAL_CLASS_ENDS.iter().take_while(|&&end| end < len) {
                    relax_at(end);
                }
                relax_at(len);
            };
            let measure = |from: usize| -> usize {
                if !tokens[from].same(tokens[i]) {
                    return 0;
                }
                1 + match_kernel(&tokens[from + 1..from + limit], &tokens[i + 1..i + limit])
            };
            for &distance in &recent {
                if distance != 0 && distance <= i {
                    relax(measure(i - distance), distance);
                }
            }
            if i + PRICED_GRAM <= n {
                let mut candidate = heads[priced_hash(tokens, i)];
                // Nearest first: a farther candidate pays more for its
                // distance, so it matters only where it is longer.
                let mut chain_longest = 0usize;
                for _ in 0..max_probes {
                    if candidate == PRICED_NONE {
                        break;
                    }
                    let from = candidate as usize;
                    if i - from > LZ77_WINDOW {
                        break;
                    }
                    if chain_longest == 0
                        || tokens[from + chain_longest].same(tokens[i + chain_longest])
                    {
                        let len = measure(from);
                        if len > chain_longest {
                            chain_longest = len;
                            relax(len, i - from);
                            if len == limit {
                                break;
                            }
                        }
                    }
                    candidate = chain[from];
                }
            }
        }
        let (gain, len, distance) = best;
        if gain > 0.0 {
            if let Some(at) = recent.iter().position(|&d| d == distance) {
                recent[..=at].rotate_right(1);
            } else {
                recent.rotate_right(1);
                recent[0] = distance;
            }
            if len >= OPTIMAL_NICE_LENGTH {
                // Every position the copy covers continues to its end by
                // the same copy, so each one that a path reaches hands its
                // cost to the end and the positions between are not parsed.
                let end = i + len;
                let base = (match_prices.of_distance(distance) + PRICED_MIN_GAIN) as f64;
                for pos in i + 1..end {
                    let reached = cost[pos];
                    cost[pos] = f64::INFINITY;
                    let shifted = end - pos;
                    if reached.is_finite() && shifted >= LZ77_MIN_LENGTH as usize {
                        let price = reached + base + MatchPrices::length_bits(shifted) as f64;
                        if price < cost[end] {
                            cost[end] = price;
                            edges[end] = optimal_edge(shifted, distance);
                        }
                    }
                }
                for pos in i..end {
                    insert(heads, chain, pos);
                }
                i = end;
                continue;
            }
        } else if longest >= PRICED_SKIP_MIN_LENGTH {
            no_search_until = i + longest / 2;
        }
        insert(heads, chain, i);
        i += 1;
    }
    if cost[n] >= prefix[n] as f64 {
        return None;
    }
    // Read the path back: edges end at their positions.
    let mut count = 0usize;
    let mut pos = n;
    while pos > 0 {
        let edge = edges[pos];
        pos -= if edge == OPTIMAL_LITERAL {
            1
        } else {
            optimal_edge_match(edge).0
        };
        count += 1;
    }
    let mut out: Vec<LzToken> = vec![LzToken::pixel(0, 0); count];
    let mut pos = n;
    for slot in out.iter_mut().rev() {
        let edge = edges[pos];
        if edge == OPTIMAL_LITERAL {
            pos -= 1;
            let token = tokens[pos].as_token();
            *slot = LzToken::pixel(token.context, token.value);
        } else {
            let (len, distance) = optimal_edge_match(edge);
            pos -= len;
            *slot = LzToken::lz77(
                tokens[pos].as_token().context,
                len as u32 - LZ77_MIN_LENGTH,
                lz77_distance_value(distance),
            );
        }
    }
    Some(out)
}
