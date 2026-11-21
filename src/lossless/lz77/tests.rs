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

//! Tests for lz77.

use super::entropy::estimate_literal_and_run_bits_single;
use super::priced::{
    LiteralPrices, MatchPrices, PRICED_MAX_PROBES, PRICED_MIN_GAIN, lz77_compress_optimal,
    lz77_compress_priced,
};
use super::{
    LZ77_DIST_VALUE, LZ77_MIN_LENGTH, LZ77_MIN_SYMBOL, LZ77_NUM_SPECIAL_DISTANCES, LzToken,
    RING_BUCKET, RING_ENTRIES_LEN, RING_HASH_SIZE, RING_WRAPPED, find_match_ring, lz77_compress,
    lz77_compress_channels_for_speed, lz77_compress_runs, lz77_compress_with_depth_into, ring_hash,
    ring_insert,
};
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;

#[test]
fn pruned_ring_search_matches_exhaustive_candidates() {
    let mut entries = vec![0u32; RING_ENTRIES_LEN];
    let mut cursors = vec![0u32; RING_HASH_SIZE];
    let mut epoch = 0;
    for period in [1, 2, 7, 31, 127] {
        for variant in 0..4 {
            let mut seed = 0x13ab_57f9u32;
            let tokens: Vec<_> = (0..1027)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    let mut token = Token::new((i % period % 3) as u32, (i % period) as u32);
                    if variant == 1 && i % 47 == 0 {
                        token.context ^= 1;
                    } else if variant == 2 && i % 53 == 0 {
                        token.value ^= 1;
                    } else if variant == 3 {
                        token = Token::new(seed % 3, seed % 17);
                    }
                    token
                })
                .collect();
            for max_probes in [0, 1, 3, 8, 12] {
                epoch += 1;
                for pos in 0..tokens.len() {
                    let mut expected = (0, 0);
                    if pos + LZ77_MIN_LENGTH as usize <= tokens.len() {
                        let hash = ring_hash(&tokens, pos);
                        let word = cursors[hash];
                        if word >> 16 == epoch {
                            let next = word as usize & (RING_BUCKET - 1);
                            let count = if word & RING_WRAPPED != 0 {
                                RING_BUCKET
                            } else {
                                next
                            };
                            for k in 1..=count.min(max_probes) {
                                let slot = (next + RING_BUCKET - k) % RING_BUCKET;
                                let candidate = entries[hash * RING_BUCKET + slot] as usize;
                                // Scalar oracle scans every candidate from the start,
                                // including ties; no pruning or SIMD match kernel.
                                let length = tokens[candidate..]
                                    .iter()
                                    .zip(&tokens[pos..])
                                    .take(1 << 20)
                                    .take_while(|(a, b)| a.value == b.value)
                                    .count();
                                if length > expected.0 {
                                    expected = (length, pos - candidate);
                                }
                            }
                        }
                    }
                    assert_eq!(
                        find_match_ring(
                            &tokens,
                            pos,
                            &entries,
                            &mut cursors,
                            max_probes,
                            epoch,
                            crate::lz_match::selected(),
                        ),
                        expected,
                        "period={period}, variant={variant}, probes={max_probes}, pos={pos}"
                    );
                    ring_insert(
                        &mut entries,
                        &mut cursors,
                        ring_hash(&tokens, pos),
                        pos,
                        epoch,
                    );
                }
            }
        }
    }
}

#[test]
fn overlapping_matches_keep_the_maximum_length_limit() {
    let tokens = vec![Token::new(2, 17); (1 << 20) + 257];
    let compressed = lz77_compress(&tokens);
    assert_eq!(compressed[1].value + LZ77_MIN_LENGTH, 1 << 20);
    assert!(same_tokens(&expand(&compressed), &tokens));
}

fn expand(stream: &[LzToken]) -> Vec<Token> {
    let mut out = Vec::new();
    for &token in stream {
        if token.is_lz77() {
            let distance = if token.distance == LZ77_DIST_VALUE {
                1
            } else {
                (token.distance - LZ77_NUM_SPECIAL_DISTANCES + 1) as usize
            };
            for _ in 0..token.value + LZ77_MIN_LENGTH {
                let source = out.len() - distance;
                out.push(out[source]);
            }
        } else {
            out.push(Token::new(token.context, token.value));
        }
    }
    out
}

/// A 40-token pattern with a rare value, repeated with one token changed
/// every 7th period. `context` 0 holds the common value at a fraction of
/// a bit, so only long matches can pay.
fn periodic_tokens(periods: usize) -> Vec<Token> {
    let mut tokens = Vec::new();
    for period in 0..periods {
        for k in 0..40usize {
            let value = if k % 8 == 3 || (k == 17 && period % 7 == 0) {
                1 + (k as u32 % 3)
            } else {
                0
            };
            tokens.push(Token::new(u32::from(value != 0), value));
        }
    }
    tokens
}

fn same_values(a: &[Token], b: &[Token]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.value == b.value)
}

#[test]
fn priced_parse_follows_a_period_and_round_trips() {
    let tokens = periodic_tokens(200);
    let prices = LiteralPrices::new(&[tokens.as_slice()], 3);
    let mut scratch = Vec::new();
    for probes in [PRICED_MAX_PROBES, 64] {
        let first = lz77_compress_priced(
            &tokens,
            &prices,
            &MatchPrices::initial(),
            probes,
            &mut scratch,
        )
        .expect("a periodic stream pays");
        assert!(same_values(&expand(&first), &tokens));
        assert!(first.len() * 10 < tokens.len());
        let learned = MatchPrices::learned(&[first.as_slice()], 3);
        let second = lz77_compress_priced(&tokens, &prices, &learned, probes, &mut scratch)
            .expect("still pays under learned prices");
        assert!(same_values(&expand(&second), &tokens));
    }
}

#[test]
fn learned_match_prices_agree_with_independent_log_probabilities() {
    // Three matches among twelve context-0 tokens; context 1 has two
    // matches of one length and three of another, across a second stream.
    let mut first = vec![LzToken::pixel(0, 7); 9];
    first.extend([LzToken::lz77(0, 0, 1); 3]);
    let mut second = vec![LzToken::lz77(1, 0, 1); 2];
    second.extend([LzToken::lz77(1, 1, 2); 3]);
    let prices = MatchPrices::learned(&[&first, &second], 3);
    let length = (3.0 * 4.0f64.log2() + 2.0 * 2.5f64.log2() + 3.0 * (5.0f64 / 3.0).log2()) / 8.0;
    assert!((f64::from(prices.length) - length).abs() < 2.0e-6);
    for (symbol, reference) in [
        (1, (8.0f64 / 5.0).log2()),
        (2, (8.0f64 / 3.0).log2()),
        (3, 9.0f64.log2()), // An unseen distance costs log2(matches + 1).
    ] {
        assert!((f64::from(prices.distance[symbol]) - reference).abs() < 2.0e-6);
    }
    assert!(
        prices
            .distance
            .iter()
            .all(|&price| price.is_finite() && price >= 0.0)
    );

    let only_match = [LzToken::lz77(0, 0, 1)];
    let prices = MatchPrices::learned(&[&only_match], 3);
    assert_eq!(prices.length, 0.0); // log2(1) stays exactly zero.
    assert_eq!(prices.distance[1], 0.0);
    assert_eq!(prices.distance[2], 1.0);
    for stream in [&[][..], &first[..9]] {
        let prices = MatchPrices::learned(&[stream], 3);
        let defaults = MatchPrices::initial();
        assert_eq!(prices.length, defaults.length);
        assert_eq!(prices.distance, defaults.distance);
    }
}

#[test]
fn literal_estimate_counts_every_token_in_its_own_context() {
    // Equal values alternating between two contexts: each context holds
    // one symbol, so the literals are free. Crediting the run to its
    // first context would give the same answer here, so add a second
    // value that only context 1 ever sees.
    let mut tokens = Vec::new();
    for k in 0..4096u32 {
        tokens.push(Token::new(k % 2, 0));
    }
    for _ in 0..4096 {
        tokens.push(Token::new(1, 5));
    }
    let (literal, _) = estimate_literal_and_run_bits_single(&tokens, 3, LZ77_MIN_SYMBOL);
    // Context 0: 2048 zeros. Context 1: 2048 zeros and 4096 fives.
    let expected = 2048.0 * (6144.0f64 / 2048.0).log2() + 4096.0 * (6144.0f64 / 4096.0).log2();
    assert!((literal - expected).abs() < 1.0, "{literal} vs {expected}");
}

fn same_tokens(a: &[Token], b: &[Token]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.context == b.context && a.value == b.value)
}

#[test]
fn hash_chain_finds_non_run_matches_and_round_trips() {
    let pattern: Vec<Token> = (0..37)
        .map(|i| Token::new((i % 3) as u32, ((i * 29 + i * i) % 251) as u32))
        .collect();
    let mut input = pattern.clone();
    input.extend((0..11).map(|i| Token::new(1, 900 + i)));
    input.extend_from_slice(&pattern);
    input.extend_from_slice(&pattern);
    let compressed = lz77_compress(&input);
    assert!(
        compressed
            .iter()
            .any(|token| token.is_lz77() && token.distance >= LZ77_NUM_SPECIAL_DISTANCES)
    );
    assert!(same_tokens(&expand(&compressed), &input));
}

#[test]
fn channel_run_compression_matches_concatenated_stream() {
    let channels = vec![
        vec![Token::new(2, 7); 19],
        (0..37).map(|i| Token::new(1, i % 5)).collect(),
        vec![Token::new(0, 4); 11],
    ];
    let concatenated: Vec<Token> = channels.iter().flatten().copied().collect();
    let expected = lz77_compress_runs(&concatenated);
    let actual = lz77_compress_channels_for_speed(
        channels,
        3,
        crate::Speed::Fast,
        &mut CoderScratch::default(),
    );
    assert_eq!(expected.len(), actual.len());
    assert!(expected.iter().zip(&actual).all(|(a, b)| {
        a.context == b.context && a.value == b.value && a.distance == b.distance
    }));
    assert!(same_tokens(&expand(&actual), &concatenated));
}

fn tokens(n: usize, period: usize) -> Vec<Token> {
    (0..n)
        .map(|i| Token::new((i % 3) as u32, (i % period) as u32))
        .collect()
}

#[test]
fn hash_chain_scratch_is_reset_between_calls() {
    let big = tokens(200_000, 7);
    let small = tokens(5_000, 7);
    let clean = {
        let mut scratch = Vec::new();
        let mut out = Vec::with_capacity(small.len());
        lz77_compress_with_depth_into(&small, 8, &mut scratch, &mut out);
        out.len()
    };
    let mut scratch = Vec::new();
    let mut out = Vec::with_capacity(big.len());
    lz77_compress_with_depth_into(&big, 8, &mut scratch, &mut out);
    lz77_compress_with_depth_into(&small, 8, &mut scratch, &mut out);
    assert_eq!(clean, out.len());
}

/// Priced bits of a parse under the parse's own price model, each match
/// carrying the margin it must clear.
fn priced_bits(stream: &[LzToken], prices: &LiteralPrices, match_prices: &MatchPrices) -> f64 {
    stream
        .iter()
        .map(|token| {
            if token.is_lz77() {
                let distance = if token.distance == LZ77_DIST_VALUE {
                    1
                } else {
                    (token.distance - LZ77_NUM_SPECIAL_DISTANCES + 1) as usize
                };
                f64::from(
                    match_prices.of((token.value + LZ77_MIN_LENGTH) as usize, distance)
                        + PRICED_MIN_GAIN,
                )
            } else {
                f64::from(prices.of(Token::new(token.context, token.value)))
            }
        })
        .sum()
}

#[test]
fn optimal_parse_round_trips_and_never_prices_above_greedy() {
    let mut scratch = Vec::new();
    for tokens in [periodic_tokens(200), periodic_tokens(3000)] {
        let prices = LiteralPrices::new(&[tokens.as_slice()], 3);
        let mut match_prices = MatchPrices::initial();
        for _ in 0..2 {
            let greedy = lz77_compress_priced(
                &tokens,
                &prices,
                &match_prices,
                PRICED_MAX_PROBES,
                &mut scratch,
            )
            .expect("a periodic stream pays");
            let optimal = lz77_compress_optimal(
                &tokens,
                &prices,
                &match_prices,
                PRICED_MAX_PROBES,
                &mut scratch,
            )
            .expect("a periodic stream pays");
            assert!(same_values(&expand(&optimal), &tokens));
            let (greedy_bits, optimal_bits) = (
                priced_bits(&greedy, &prices, &match_prices),
                priced_bits(&optimal, &prices, &match_prices),
            );
            assert!(
                optimal_bits <= greedy_bits + 1e-3,
                "{optimal_bits} > {greedy_bits}"
            );
            match_prices = MatchPrices::learned(&[optimal.as_slice()], 3);
        }
    }
}

#[test]
fn optimal_parse_stops_a_match_early_for_a_longer_one() {
    // "1..6", then "2..11", then "1..11", then noise. The greedy parse
    // copies "1..6" from the first span and spells out 7..11, since no
    // 6-gram of the second span continues them; the optimal parse codes
    // the literal 1 and copies "2..11" from the second span whole.
    let mut tokens: Vec<Token> = (1..=6).map(|v| Token::new(0, v)).collect();
    tokens.extend((2..=11).map(|v| Token::new(0, v)));
    tokens.extend((1..=11).map(|v| Token::new(0, v)));
    let mut state = 0x9e37_79b9u32;
    let mut noise = |floor: u32, shift: u32| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        Token::new(0, floor + (state >> shift))
    };
    tokens.extend((0..17).map(|_| noise(12, 27)));
    // Six bits a literal: priced on a stream where every value is as rare.
    let pricing: Vec<Token> = (0..4096).map(|_| noise(0, 26)).collect();
    let prices = LiteralPrices::new(&[pricing.as_slice()], 1);
    let match_prices = MatchPrices::initial();
    let mut scratch = Vec::new();
    let greedy = lz77_compress_priced(&tokens, &prices, &match_prices, 64, &mut scratch)
        .expect("the copies pay");
    let optimal = lz77_compress_optimal(&tokens, &prices, &match_prices, 64, &mut scratch)
        .expect("the copies pay");
    assert!(same_values(&expand(&optimal), &tokens));
    // Tokens of the third span: those after the first 16 literals and
    // before the noise.
    let mut covered = 0usize;
    let third: Vec<(bool, u32)> = optimal
        .iter()
        .filter_map(|t| {
            let len = if t.is_lz77() {
                t.value + LZ77_MIN_LENGTH
            } else {
                1
            } as usize;
            let at = covered;
            covered += len;
            (16..27).contains(&at).then_some((t.is_lz77(), t.value))
        })
        .collect();
    assert_eq!(third, vec![(false, 1), (true, 10 - LZ77_MIN_LENGTH)]);
    let (greedy_bits, optimal_bits) = (
        priced_bits(&greedy, &prices, &match_prices),
        priced_bits(&optimal, &prices, &match_prices),
    );
    assert!(
        optimal_bits < greedy_bits,
        "{optimal_bits} vs {greedy_bits}"
    );
}
