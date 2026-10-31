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

use super::entropy::{
    build_lz_pixel_code_opts, build_lz_pixel_code_threads, estimate_coded_bits,
    estimate_literal_and_run_bits, estimate_literal_and_run_bits_single, estimate_streams_bits,
    lz77_literals,
};
use super::priced::{
    LiteralPrices, MatchPrices, PRICED_MAX_PROBES, PRICED_MIN_GAIN, lz77_compress_optimal,
    lz77_compress_priced,
};
use super::write::{write_local_tree_lz77, write_lz_section};
use super::{
    DEEP_LZ_SCRATCH_WORDS, LZ77_DIST_VALUE, LZ77_MIN_LENGTH, LZ77_MIN_SYMBOL,
    LZ77_NUM_SPECIAL_DISTANCES, LzToken, LzTokenSource, RING_BUCKET, RING_ENTRIES_LEN,
    RING_HASH_SIZE, RING_WRAPPED, RunLzTokens, RunLzWriter, find_match_ring, fingerprint,
    lz77_compress, lz77_compress_channels_for_speed, lz77_compress_for_speed,
    lz77_compress_for_speed_with_depth, lz77_compress_runs, lz77_compress_runs_channels,
    lz77_compress_with_depth_into, lz77_length_encode, lz77_run_count, ring_hash, ring_insert,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS};
use crate::entropy::{CompactToken, EntropyCode, Histogram, Token, f_log2};
use crate::thread_pool::ThreadPool;

#[test]
fn sparse_ans_words_match_expanded_symbol_stream() {
    fn reference(tokens: &[LzToken], code: &EntropyCode<'_>, w: &mut BitWriter) {
        let mut expanded = Vec::new();
        for &t in tokens {
            let hist = code.context_map[t.context as usize] as usize;
            if t.is_lz77() {
                let (symbol, nbits, bits) = lz77_length_encode(t.value);
                expanded.push((hist, 64 + symbol, nbits, bits));
                let hist = code.context_map[3] as usize;
                let (symbol, nbits, bits) = crate::entropy::uint_encode_with_config(
                    t.distance,
                    code.hybrid_uint_configs[hist],
                );
                expanded.push((hist, symbol, nbits, bits));
            } else {
                let (symbol, nbits, bits) = crate::entropy::uint_encode_with_config(
                    t.value,
                    code.hybrid_uint_configs[hist],
                );
                expanded.push((hist, symbol, nbits, bits));
            }
        }
        let mut coder = crate::entropy::AnsCoder::new();
        let words: Vec<_> = expanded
            .iter()
            .rev()
            .map(|&(hist, symbol, _, _)| {
                let start = hist * crate::entropy::ANS_TAB_SIZE as usize;
                coder.put_symbol(
                    &code.ans_symbols[hist][symbol as usize],
                    &code.ans_reverse_maps[start..start + crate::entropy::ANS_TAB_SIZE as usize],
                )
            })
            .collect();
        w.write(32, coder.state() as u64);
        for ((_, _, nbits, bits), word) in expanded.into_iter().zip(words.into_iter().rev()) {
            if let Some(word) = word {
                w.write(16, word as u64);
            }
            w.write(nbits as usize, bits as u64);
        }
    }

    for n in [0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 1025, 65_537] {
        for pattern in 0..3 {
            let stream: Vec<_> = (0..n)
                .map(|i| match pattern {
                    0 => LzToken::pixel(0, 0),
                    1 => LzToken::pixel(i % 3, ((i * 137) ^ (i >> 3)) & 65_535),
                    _ if i % 3 == 0 => LzToken::lz77(i % 2, i * 53, 1 + i * 97),
                    _ => LzToken::pixel(i % 3, i & 255),
                })
                .collect();
            let mut scratch = CoderScratch::default();
            let code = build_lz_pixel_code_opts(
                false,
                std::iter::once(stream.as_slice()),
                3,
                64,
                true,
                true,
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            assert!(!code.use_prefix_code);
            let mut actual = BitWriter::new();
            let mut expected = BitWriter::new();
            // Exercise a section starting at an unaligned bit position.
            actual.write(3, 5);
            expected.write(3, 5);
            write_lz_section(&stream, 3, &code, 64, &mut actual);
            reference(&stream, &code, &mut expected);
            assert_eq!(actual.bits_written(), expected.bits_written());
            // Byte comparison needs byte-aligned writers (debug-checked).
            actual.zero_pad_to_byte();
            expected.zero_pad_to_byte();
            assert_eq!(
                actual.into_bytes(),
                expected.into_bytes(),
                "n={n}, pattern={pattern}"
            );
        }
    }
}

#[test]
fn streaming_estimates_match_materialized_runs_exactly() {
    fn reference(streams: &[Vec<LzToken>], contexts: usize, min_symbol: u32) -> f64 {
        let mut histograms = vec![Histogram::new(); contexts];
        let mut extra = 0u64;
        for token in streams.iter().flatten() {
            if token.is_lz77() {
                let (symbol, bits, _) = lz77_length_encode(token.value);
                histograms[token.context as usize].add(min_symbol + symbol);
                let (symbol, distance_bits, _) = crate::entropy::uint_encode(token.distance);
                histograms[contexts - 1].add(symbol);
                extra += bits as u64 + distance_bits as u64;
            } else {
                let (symbol, bits, _) = crate::entropy::uint_encode(token.value);
                histograms[token.context as usize].add(symbol);
                extra += bits as u64;
            }
        }
        let mut bits = extra as f64;
        for h in histograms {
            let total = h.total_count as f64;
            if total != 0.0 {
                for count in h.counts {
                    if count != 0 {
                        bits += count as f64 * f_log2(total / count as f64);
                    }
                }
            }
        }
        bits
    }

    for threads in [1, 2, 8] {
        let pool = ThreadPool::new_lossless(threads);
        let mut scratch = CoderScratch::lossless();
        for contexts in [2, 7, 129] {
            let mut mixed = Vec::new();
            for (i, count) in [1, 2, 3, 4, 5, 6, 17, 31, 257, 65537]
                .into_iter()
                .enumerate()
            {
                let context = (i % (contexts - 1)) as u32;
                let value = [0, 1, 31, 32, 255, 65535, u32::MAX][i % 7];
                mixed.extend(std::iter::repeat_n(Token::new(context, value), count));
            }
            let dense: Vec<_> = (0..1031)
                .map(|i| Token::new((i % (contexts - 1)) as u32, (i * 977) as u32))
                .collect();
            for streams in [
                Vec::new(),
                vec![Vec::new()],
                vec![vec![Token::new(0, 9); 3], vec![Token::new(0, 9); 3]],
                vec![
                    mixed.clone(),
                    Vec::new(),
                    dense,
                    vec![Token::new(0, 0); 70000],
                ],
            ] {
                let runs: Vec<_> = streams.iter().map(|s| lz77_compress_runs(s)).collect();
                for (tokens, runs) in streams.iter().zip(&runs) {
                    assert_eq!(lz77_run_count(tokens), runs.len());
                }
                let literals: Vec<_> = streams.iter().map(|s| lz77_literals(s)).collect();
                let slices: Vec<_> = streams.iter().map(Vec::as_slice).collect();
                for min_symbol in [16, 32, 64] {
                    let (literal_bits, run_bits) = estimate_literal_and_run_bits(
                        &slices,
                        contexts,
                        min_symbol,
                        &pool,
                        &mut scratch,
                    );
                    assert_eq!(
                        literal_bits.to_bits(),
                        reference(&literals, contexts, min_symbol).to_bits()
                    );
                    assert_eq!(
                        run_bits.to_bits(),
                        reference(&runs, contexts, min_symbol).to_bits()
                    );
                    let run_slices: Vec<_> = runs.iter().map(Vec::as_slice).collect();
                    assert_eq!(
                        estimate_streams_bits(
                            &run_slices,
                            contexts,
                            min_symbol,
                            &pool,
                            &mut scratch
                        )
                        .to_bits(),
                        run_bits.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn direct_literals_match_materialized_codes_estimates_and_sections() {
    fn encode<T: LzTokenSource>(
        streams: &[Vec<T>],
        threads: usize,
        refined: bool,
        ans_cluster: bool,
    ) -> (Vec<u8>, u64, u64) {
        let mut scratch = CoderScratch::default();
        let slices: Vec<&[T]> = streams.iter().map(Vec::as_slice).collect();
        let pool = ThreadPool::new_lossless(threads);
        let raw = estimate_streams_bits(&slices, 4, 64, &pool, &mut scratch);
        let code = build_lz_pixel_code_threads(
            false,
            slices.iter().copied(),
            3,
            64,
            refined,
            ans_cluster,
            Some(&pool),
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        let coded = estimate_coded_bits(&slices, 3, &code, 64, Some(&pool));
        let mut writer = BitWriter::new();
        write_local_tree_lz77(
            &[0, 1, 6],
            &code,
            64,
            &mut scratch.huffman_pool,
            &mut writer,
        );
        for stream in streams {
            write_lz_section(stream, 3, &code, 64, &mut writer);
            writer.zero_pad_to_byte();
        }
        (writer.into_bytes(), raw.to_bits(), coded.to_bits())
    }

    let streams = vec![
        Vec::new(),
        vec![Token::new(0, 0); 70_003],
        (0..140_011u32)
            .map(|i| Token::new(1 + i % 2, (i.wrapping_mul(137) ^ (i >> 3)) & 65_535))
            .collect(),
        vec![Token::new(2, 17); 7],
    ];
    let materialized: Vec<_> = streams.iter().map(|s| lz77_literals(s)).collect();
    for threads in [1, 4] {
        for (refined, ans_cluster) in [(false, false), (true, false), (true, true)] {
            assert_eq!(
                encode(&streams, threads, refined, ans_cluster),
                encode(&materialized, threads, refined, ans_cluster),
            );
        }
    }
}

#[test]
fn compact_literals_preserve_hashes_runs_matches_and_estimates() {
    let pool = ThreadPool::new(2);
    let mut wide_scratch = CoderScratch::default();
    let mut compact_scratch = CoderScratch::default();
    let mut wide_depth = Vec::new();
    let mut compact_depth = Vec::new();
    for n in [0, 1, 2, 3, 4, 257, 4097] {
        for pattern in 0..4 {
            let wide: Vec<_> = (0..n)
                .map(|i| match pattern {
                    0 => Token::new(2, CompactToken::MAX_VALUE),
                    1 => Token::new((i % 31 % 3) as u32, (i % 31 * 137) as u32),
                    2 => Token::new((i / 47 % 3) as u32, (i / 47) as u32),
                    _ => Token::new(
                        (i % 3) as u32,
                        (((i * 137) ^ (i / 7)) as u32) & CompactToken::MAX_VALUE,
                    ),
                })
                .collect();
            let compact: Vec<_> = wide
                .iter()
                .map(|t| CompactToken::try_new(t.context, t.value).unwrap())
                .collect();
            for pos in 0..n {
                assert_eq!(fingerprint(&wide, pos), fingerprint(&compact, pos));
            }
            let run_count = lz77_run_count(&wide);
            assert_eq!(run_count, lz77_run_count(&compact));
            assert_eq!(
                RunLzTokens::new(&wide).collect::<Vec<_>>(),
                RunLzTokens::new(&compact).collect::<Vec<_>>()
            );
            let old = lz77_compress_for_speed_with_depth(
                &wide,
                3,
                crate::Speed::Slow,
                &mut wide_depth,
                Some(run_count),
                &mut wide_scratch,
            );
            let new = lz77_compress_for_speed_with_depth(
                &compact,
                3,
                crate::Speed::Slow,
                &mut compact_depth,
                Some(run_count),
                &mut compact_scratch,
            );
            assert_eq!(old, new, "n={n}, pattern={pattern}");
            let old = estimate_literal_and_run_bits(&[&wide], 4, 96, &pool, &mut wide_scratch);
            let new =
                estimate_literal_and_run_bits(&[&compact], 4, 96, &pool, &mut compact_scratch);
            assert_eq!(
                (old.0.to_bits(), old.1.to_bits()),
                (new.0.to_bits(), new.1.to_bits())
            );
        }
    }
}

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
fn priced_parse_leaves_cheap_and_random_streams_literal() {
    // One value in one context: every literal is free.
    let flat = vec![Token::new(0, 0); 1 << 16];
    let prices = LiteralPrices::new(&[flat.as_slice()], 2);
    let mut scratch = Vec::new();
    let parsed = lz77_compress_priced(
        &flat,
        &prices,
        &MatchPrices::initial(),
        PRICED_MAX_PROBES,
        &mut scratch,
    );
    assert!(parsed.is_none_or(|stream| stream.iter().all(|t| !t.is_lz77())));

    let mut state = 0x1234_5678u32;
    let noise: Vec<Token> = (0..1 << 16)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            Token::new(0, state >> 24)
        })
        .collect();
    let prices = LiteralPrices::new(&[noise.as_slice()], 2);
    assert!(
        lz77_compress_priced(
            &noise,
            &prices,
            &MatchPrices::initial(),
            PRICED_MAX_PROBES,
            &mut scratch,
        )
        .is_none()
    );
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
fn hash_chain_uses_compact_distance_for_runs() {
    let input = vec![Token::new(0, 7); 128];
    let compressed = lz77_compress(&input);
    assert!(
        compressed
            .iter()
            .any(|token| token.is_lz77() && token.distance == LZ77_DIST_VALUE)
    );
    assert!(same_tokens(&expand(&compressed), &input));
}

#[test]
fn lz_token_is_compact() {
    assert_eq!(std::mem::size_of::<LzToken>(), 12);
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

#[test]
fn streaming_run_writer_matches_buffered_channels() {
    let channels = vec![
        vec![Token::new(2, 7); 19],
        vec![Token::new(1, 5); 3],
        (0..37).map(|i| Token::new(0, i % 5)).collect(),
    ];
    let expected = lz77_compress_runs_channels(channels.clone());
    let mut writer = RunLzWriter::with_capacity(channels.iter().map(Vec::len).sum());
    for channel in channels {
        for token in channel {
            writer.push(token);
        }
        writer.finish_channel();
    }
    let actual = writer.finish();

    assert_eq!(expected.len(), actual.len());
    assert!(expected.iter().zip(&actual).all(|(a, b)| {
        a.context == b.context && a.value == b.value && a.distance == b.distance
    }));
}

#[test]
fn fixed_context_storage_covers_the_largest_squeeze_tree() {
    let steps = crate::squeeze::default_squeeze_steps(
        crate::encode_image::MAX_DIMENSION,
        crate::encode_image::MAX_DIMENSION,
        4,
    );
    let contexts = 4 * (steps.len() + 1) + 1;
    assert_eq!(steps.len(), 54);
    assert!(contexts <= LZ77_MAX_CONTEXTS);
}

#[test]
fn speed_policy_keeps_fast_run_only_and_slow_structured_search() {
    let pattern: Vec<Token> = (0..64)
        .map(|i| Token::new((i % 3) as u32, ((i * 37 + 11) % 257) as u32))
        .collect();
    let input: Vec<Token> = pattern.iter().copied().cycle().take(512).collect();
    let mut scratch = CoderScratch::default();
    assert_eq!(scratch.lz_repetitions.capacity(), 0);
    assert_eq!(scratch.lz_depth.capacity(), 0);
    assert_eq!(scratch.lz_candidate.capacity(), 0);
    let fast = lz77_compress_for_speed(&input, 3, crate::Speed::Fast, &mut scratch);
    assert!(
        !fast
            .iter()
            .any(|token| token.is_lz77() && token.distance != LZ77_DIST_VALUE)
    );
    assert_eq!(scratch.lz_repetitions.capacity(), 0);
    assert_eq!(scratch.lz_depth.capacity(), 0);
    assert_eq!(scratch.lz_candidate.capacity(), 0);
    let slow = lz77_compress_for_speed(&input, 3, crate::Speed::Slow, &mut scratch);
    assert!(
        slow.iter()
            .any(|token| token.is_lz77() && token.distance != LZ77_DIST_VALUE)
    );
    assert!(same_tokens(&expand(&slow), &input));
    assert_eq!(scratch.lz_depth.len(), DEEP_LZ_SCRATCH_WORDS);
    assert!(scratch.lz_repetitions.capacity() >= 1 << 14);
    assert!(scratch.lz_candidate.capacity() >= scratch.lz_candidate.len());
    assert!(scratch.lz_candidate.capacity() <= input.len());

    let mut pooled_depth = Vec::new();
    let pooled = lz77_compress_for_speed_with_depth(
        &input,
        3,
        crate::Speed::Slow,
        &mut pooled_depth,
        None,
        &mut CoderScratch::default(),
    );
    assert_eq!(pooled_depth.len(), DEEP_LZ_SCRATCH_WORDS);
    assert_eq!(slow.len(), pooled.len());
    assert!(slow.iter().zip(&pooled).all(|(a, b)| {
        a.context == b.context && a.value == b.value && a.distance == b.distance
    }));

    let allocations = (
        scratch.lz_repetitions.as_ptr(),
        scratch.lz_depth.as_ptr(),
        scratch.lz_candidate.as_ptr(),
    );
    let capacities = (
        scratch.lz_repetitions.capacity(),
        scratch.lz_depth.capacity(),
        scratch.lz_candidate.capacity(),
    );
    let slow_again = lz77_compress_for_speed(&input, 3, crate::Speed::Slow, &mut scratch);
    assert!(same_tokens(&expand(&slow), &expand(&slow_again)));
    assert_eq!(
        allocations,
        (
            scratch.lz_repetitions.as_ptr(),
            scratch.lz_depth.as_ptr(),
            scratch.lz_candidate.as_ptr(),
        )
    );
    assert_eq!(
        capacities,
        (
            scratch.lz_repetitions.capacity(),
            scratch.lz_depth.capacity(),
            scratch.lz_candidate.capacity(),
        )
    );
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

#[test]
fn deep_search_handles_varied_stream_lengths() {
    let mut scratch = Vec::new();
    let mut out = Vec::with_capacity(50_000);
    for len in [1_000usize, 50_000, 3_000] {
        lz77_compress_with_depth_into(&tokens(len, 5), 8, &mut scratch, &mut out);
    }
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
