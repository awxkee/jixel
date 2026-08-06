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
use super::{NUM_TREE_CONTEXTS, build_balanced_tree_tokens};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS, LzEntropyScratch};
use crate::entropy::{
    ALPHABET_SIZE, EntropyCode, Histogram, Token, build_huffman_codes_into,
    cluster_histograms_fixed, optimize_entropy_code, write_entropy_code, write_token,
};

pub(super) const LZ77_MIN_SYMBOL: u32 = 64;
pub(super) const LZ77_MIN_LENGTH: u32 = 3;
// special_distance[1] = (dx=1, dy=0) -> one token back.
pub(super) const LZ77_DIST_VALUE: u32 = 1;
pub(super) const LZ77_NUM_SPECIAL_DISTANCES: u32 = 120;

/// Hybrid-encode `length_value` (`run_length - LZ77_MIN_LENGTH`).
/// Returns `(alphabet_token, nbits, payload)`.
#[inline]
pub(super) fn lz77_length_encode(length_value: u32) -> (u32, u32, u32) {
    // split_exponent = 4, msb_in_token = 0, lsb_in_token = 0
    if length_value < 16 {
        (length_value, 0, 0)
    } else {
        let n = 31 - length_value.leading_zeros();
        let token = 16 + n - 4;
        let nbits = n;
        let bits = length_value - (1 << n);
        (token, nbits, bits)
    }
}

/// One emission unit in an LZ77-compressed token stream.
#[derive(Clone, Copy)]
pub(crate) struct LzToken {
    context: u32,
    value: u32,
    /// Zero denotes a literal. Encoded LZ77 distances are always at least one.
    distance: u32,
}

impl LzToken {
    #[inline]
    fn pixel(context: u32, value: u32) -> Self {
        Self {
            context,
            value,
            distance: 0,
        }
    }

    #[inline]
    fn lz77(context: u32, length_value: u32, distance_value: u32) -> Self {
        debug_assert_ne!(distance_value, 0);
        Self {
            context,
            value: length_value,
            distance: distance_value,
        }
    }

    #[inline]
    fn is_lz77(self) -> bool {
        self.distance != 0
    }
}

#[inline]
fn fingerprint(tokens: &[Token], pos: usize) -> u32 {
    let mut hash = 0x9e37_79b9u32;
    for token in &tokens[pos..tokens.len().min(pos + 3)] {
        hash ^= token.value.wrapping_mul(0x85eb_ca6b).rotate_left(13);
        hash = hash.wrapping_mul(0xc2b2_ae35) ^ token.context;
    }
    hash
}

fn has_repetition(tokens: &[Token], scratch: &mut Vec<u32>) -> bool {
    const MAX_SAMPLES: usize = 8_192;
    const SAMPLE_TABLE_SIZE: usize = 1 << 14;
    if tokens.len() < 256 {
        return true;
    }
    let stride = tokens.len().div_ceil(MAX_SAMPLES).max(1);
    if scratch.len() < SAMPLE_TABLE_SIZE {
        scratch.resize(SAMPLE_TABLE_SIZE, 0);
    }
    scratch.fill(0);
    let table = scratch;
    let mut samples = 0usize;
    let mut repeats = 0usize;
    for pos in (0..tokens.len().saturating_sub(2)).step_by(stride) {
        let fingerprint = fingerprint(tokens, pos) | 1;
        let slot = fingerprint as usize & (SAMPLE_TABLE_SIZE - 1);
        repeats += usize::from(table[slot] == fingerprint);
        table[slot] = fingerprint;
        samples += 1;
    }
    repeats * 20 >= samples
}

fn match_len(tokens: &[Token], a: usize, b: usize) -> usize {
    const MAX_MATCH: usize = 1 << 20;
    let max_len = (tokens.len() - b).min(MAX_MATCH);
    let mut len = 0usize;
    while len < max_len {
        let x = tokens[a + len];
        let y = tokens[b + len];
        if x.context != y.context || x.value != y.value {
            break;
        }
        len += 1;
        if a + len == b {
            let period = b - a;
            while len < max_len {
                let x = tokens[b + len - period];
                let y = tokens[b + len];
                if x.context != y.context || x.value != y.value {
                    break;
                }
                len += 1;
            }
            break;
        }
    }
    len
}

const RING_HASH_BITS: usize = 17;
const RING_HASH_SIZE: usize = 1 << RING_HASH_BITS;
const RING_BUCKET: usize = 8;
const RING_ENTRIES_LEN: usize = RING_HASH_SIZE * RING_BUCKET;
const DEEP_LZ_SCRATCH_WORDS: usize = RING_ENTRIES_LEN + RING_HASH_SIZE + 1;

#[inline]
fn ring_hash(tokens: &[Token], pos: usize) -> usize {
    fingerprint(tokens, pos) as usize & (RING_HASH_SIZE - 1)
}

/// Cursor word layout: bits 0..3 = ring write index, bit 3 = "ring wrapped"
/// (all slots live), bits 16.. = epoch stamp. A bucket whose stamp differs
/// from the current call's epoch is logically empty — no per-call memset of
/// the (multi-MB) entry table.
const RING_WRAPPED: u32 = 1 << 3;

#[inline]
fn ring_bucket_state(cursors: &mut [u32], h: usize, epoch: u32) -> u32 {
    let word = cursors[h];
    if word >> 16 != epoch {
        let fresh = epoch << 16;
        cursors[h] = fresh;
        fresh
    } else {
        word
    }
}

#[inline]
fn ring_insert(entries: &mut [u32], cursors: &mut [u32], h: usize, pos: usize, epoch: u32) {
    let word = ring_bucket_state(cursors, h, epoch);
    let idx = (word as usize) & (RING_BUCKET - 1);
    entries[h * RING_BUCKET + idx] = pos as u32;
    let next = (idx + 1) & (RING_BUCKET - 1);
    let wrapped = (word & RING_WRAPPED) | if next == 0 { RING_WRAPPED } else { 0 };
    cursors[h] = (epoch << 16) | wrapped | next as u32;
}

fn find_match_ring(
    tokens: &[Token],
    pos: usize,
    entries: &[u32],
    cursors: &mut [u32],
    max_probes: usize,
    epoch: u32,
) -> (usize, usize) {
    if pos + LZ77_MIN_LENGTH as usize > tokens.len() {
        return (0, 0);
    }
    let h = ring_hash(tokens, pos);
    let base = h * RING_BUCKET;
    let word = ring_bucket_state(cursors, h, epoch);
    let idx = (word as usize) & (RING_BUCKET - 1);
    let live = if word & RING_WRAPPED != 0 {
        RING_BUCKET
    } else {
        idx
    };
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    // Newest-first scan: on equal lengths the nearest (cheapest) distance wins
    // for free.
    for k in 1..=live.min(max_probes) {
        let slot = (idx + RING_BUCKET - k) % RING_BUCKET;
        let candidate_pos = entries[base + slot] as usize;
        let len = match_len(tokens, candidate_pos, pos);
        if len > best_len {
            best_len = len;
            best_dist = pos - candidate_pos;
        }
    }
    (best_len, best_dist)
}

#[cfg(test)]
fn lz77_compress(tokens: &[Token]) -> Vec<LzToken> {
    let mut scratch = CoderScratch::default();
    lz77_compress_with_depth_into(tokens, 8, &mut scratch.lz_depth, &mut scratch.lz_candidate);
    scratch.lz_candidate.clone()
}

#[derive(Clone)]
struct RunLzTokens<'a> {
    tokens: &'a [Token],
    pos: usize,
    pending: Option<LzToken>,
}

impl<'a> RunLzTokens<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            pos: 0,
            pending: None,
        }
    }
}

impl Iterator for RunLzTokens<'_> {
    type Item = LzToken;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.pending.take() {
            return Some(token);
        }
        let token = *self.tokens.get(self.pos)?;
        let mut end = self.pos + 1;
        while end < self.tokens.len()
            && self.tokens[end].context == token.context
            && self.tokens[end].value == token.value
        {
            end += 1;
        }
        let copied = end - self.pos - 1;
        self.pos += 1;
        if copied >= LZ77_MIN_LENGTH as usize {
            self.pending = Some(LzToken::lz77(
                token.context,
                copied as u32 - LZ77_MIN_LENGTH,
                LZ77_DIST_VALUE,
            ));
            self.pos = end;
        }
        Some(LzToken::pixel(token.context, token.value))
    }
}

/// Streaming equivalent of `RunLzTokens`. It retains only the current equal
/// token run, allowing prediction/tokenization to emit directly into the final
/// LZ stream without staging a raw-token plane.
pub(super) struct RunLzWriter {
    out: Vec<LzToken>,
    token: Option<Token>,
    count: usize,
}

impl RunLzWriter {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            out: Vec::with_capacity(capacity),
            token: None,
            count: 0,
        }
    }

    #[inline]
    pub(super) fn push(&mut self, token: Token) {
        if self
            .token
            .is_some_and(|current| current.context == token.context && current.value == token.value)
        {
            self.count += 1;
        } else {
            self.flush_run();
            self.token = Some(token);
            self.count = 1;
        }
    }

    /// Explicitly ends a channel so streaming behavior stays identical to
    /// independently applying run compression to each channel.
    pub(super) fn finish_channel(&mut self) {
        self.flush_run();
    }

    pub(super) fn finish(mut self) -> Vec<LzToken> {
        self.flush_run();
        self.out
    }

    fn flush_run(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        let copied = self.count - 1;
        self.out.push(LzToken::pixel(token.context, token.value));
        if copied >= LZ77_MIN_LENGTH as usize {
            self.out.push(LzToken::lz77(
                token.context,
                copied as u32 - LZ77_MIN_LENGTH,
                LZ77_DIST_VALUE,
            ));
        } else {
            for _ in 0..copied {
                self.out.push(LzToken::pixel(token.context, token.value));
            }
        }
        self.count = 0;
    }
}

#[inline]
pub(super) fn lz77_compress_for_speed(
    tokens: &[Token],
    distance_context: u32,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> Vec<LzToken> {
    let CoderScratch {
        lz_repetitions,
        lz_depth,
        lz_candidate,
        lz_entropy,
        huffman_pool,
        ..
    } = scratch;
    lz77_compress_for_speed_with_parts(
        tokens,
        distance_context,
        speed,
        lz_repetitions,
        lz_depth,
        lz_candidate,
        lz_entropy,
        huffman_pool,
    )
}

pub(super) fn lz77_compress_for_speed_with_depth(
    tokens: &[Token],
    distance_context: u32,
    speed: crate::Speed,
    depth: &mut Vec<u32>,
    scratch: &mut CoderScratch,
) -> Vec<LzToken> {
    let CoderScratch {
        lz_repetitions,
        lz_candidate,
        lz_entropy,
        huffman_pool,
        ..
    } = scratch;
    lz77_compress_for_speed_with_parts(
        tokens,
        distance_context,
        speed,
        lz_repetitions,
        depth,
        lz_candidate,
        lz_entropy,
        huffman_pool,
    )
}

#[allow(clippy::too_many_arguments)]
fn lz77_compress_for_speed_with_parts(
    tokens: &[Token],
    distance_context: u32,
    speed: crate::Speed,
    lz_repetitions: &mut Vec<u32>,
    lz_depth: &mut Vec<u32>,
    lz_candidate: &mut Vec<LzToken>,
    lz_entropy: &mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> Vec<LzToken> {
    // Fast stays runs-only
    if speed != crate::Speed::Slow || !has_repetition(tokens, lz_repetitions) {
        return lz77_compress_runs(tokens);
    }
    let run_tokens = lz77_compress_runs(tokens);
    // Slow mode's broader candidate selection protects the major Modular
    // alternatives, so do not discard useful 2..10% LZ wins at this local gate.
    let max_candidate_len = run_tokens.len().saturating_mul(98) / 100;
    if !lz77_compress_with_depth_into_limit(tokens, 8, lz_depth, lz_candidate, max_candidate_len) {
        return run_tokens;
    }
    let deep_bits = estimate_payload_bits(
        lz_candidate.iter().copied(),
        distance_context,
        lz_entropy,
        huffman_pool,
    );
    let run_bits = estimate_payload_bits(
        run_tokens.iter().copied(),
        distance_context,
        lz_entropy,
        huffman_pool,
    );
    if deep_bits * 100 <= run_bits * 98 {
        lz_candidate.clone()
    } else {
        run_tokens
    }
}

fn estimate_payload_bits<I>(
    tokens: I,
    distance_context: u32,
    scratch: &mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> u64
where
    I: Iterator<Item = LzToken> + Clone,
{
    let num_contexts = distance_context as usize + 1;
    assert!(num_contexts <= LZ77_MAX_CONTEXTS);
    let histograms = &mut scratch.histograms[..num_contexts];
    histograms.fill(Histogram::new());
    lz_add_histograms(
        tokens.clone(),
        None,
        histograms,
        LZ77_MIN_SYMBOL,
        distance_context,
    );
    let codes = &mut scratch.prefix_codes[..num_contexts];
    build_huffman_codes_into(histograms, codes, huffman_pool);
    let mut bits = 0u64;
    for token in tokens {
        if token.is_lz77() {
            let (symbol, nbits, _) = lz77_length_encode(token.value);
            bits += codes[token.context as usize].depths[(LZ77_MIN_SYMBOL + symbol) as usize]
                as u64
                + nbits as u64;
            let (symbol, nbits, _) = crate::entropy::uint_encode(token.distance);
            bits += codes[distance_context as usize].depths[symbol as usize] as u64 + nbits as u64;
        } else {
            let (symbol, nbits, _) = crate::entropy::uint_encode(token.value);
            bits += codes[token.context as usize].depths[symbol as usize] as u64 + nbits as u64;
        }
    }
    bits
}

pub(super) fn lz77_compress_runs(tokens: &[Token]) -> Vec<LzToken> {
    let mut out = Vec::with_capacity(tokens.len());
    lz77_extend_runs(tokens, &mut out);
    out
}

pub(super) fn lz77_extend_runs(tokens: &[Token], out: &mut Vec<LzToken>) {
    out.extend(RunLzTokens::new(tokens));
}

pub(super) fn lz77_compress_runs_channels(channels: Vec<Vec<Token>>) -> Vec<LzToken> {
    let total_len: usize = channels.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total_len);
    for channel in channels {
        lz77_extend_runs(&channel, &mut out);
    }
    out
}

pub(super) fn lz77_compress_channels_for_speed(
    channels: Vec<Vec<Token>>,
    distance_context: u32,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> Vec<LzToken> {
    let total_len: usize = channels.iter().map(Vec::len).sum();
    if speed != crate::Speed::Slow {
        return lz77_compress_runs_channels(channels);
    }

    // Deep matching needs random access to one contiguous stream. Reuse the
    // first channel's allocation and append the others instead of allocating
    // and copying a second full-sized token vector from scratch.
    let mut channels = channels.into_iter();
    let mut tokens = channels.next().unwrap_or_default();
    tokens.reserve(total_len.saturating_sub(tokens.len()));
    for mut channel in channels {
        tokens.append(&mut channel);
    }
    lz77_compress_for_speed(&tokens, distance_context, speed, scratch)
}

pub(super) fn lz77_compress_channels_for_speed_with_depth(
    channels: Vec<Vec<Token>>,
    distance_context: u32,
    speed: crate::Speed,
    depth: &mut Vec<u32>,
    scratch: &mut CoderScratch,
) -> Vec<LzToken> {
    let total_len: usize = channels.iter().map(Vec::len).sum();
    if speed != crate::Speed::Slow {
        return lz77_compress_runs_channels(channels);
    }

    let mut channels = channels.into_iter();
    let mut tokens = channels.next().unwrap_or_default();
    tokens.reserve(total_len.saturating_sub(tokens.len()));
    for mut channel in channels {
        tokens.append(&mut channel);
    }
    lz77_compress_for_speed_with_depth(&tokens, distance_context, speed, depth, scratch)
}

#[cfg(test)]
fn lz77_compress_with_depth_into(
    tokens: &[Token],
    max_probes: usize,
    scratch: &mut Vec<u32>,
    out: &mut Vec<LzToken>,
) {
    assert!(lz77_compress_with_depth_into_limit(
        tokens,
        max_probes,
        scratch,
        out,
        usize::MAX,
    ));
}

/// Returns false once the monotonically growing output exceeds `max_output`.
/// The caller can then select an already-built alternative without completing
/// a candidate that can no longer pass its size gate.
fn lz77_compress_with_depth_into_limit(
    tokens: &[Token],
    max_probes: usize,
    scratch: &mut Vec<u32>,
    out: &mut Vec<LzToken>,
    max_output: usize,
) -> bool {
    out.clear();
    // Highly compressible groups need only a tiny fraction of the raw token
    // bound. Start modestly; persistent worker scratch retains any growth that
    // less-compressible groups actually require.
    let initial_capacity = tokens
        .len()
        .min(max_output.saturating_add(1))
        .min(16 * 1024);
    if out.capacity() < initial_capacity {
        out.reserve(initial_capacity);
    }
    if scratch.len() < DEEP_LZ_SCRATCH_WORDS {
        scratch.clear();
        scratch.resize(DEEP_LZ_SCRATCH_WORDS, 0);
    }
    let (entries, rest) = scratch.split_at_mut(RING_ENTRIES_LEN);
    let (cursors, epoch_word) = rest.split_at_mut(RING_HASH_SIZE);
    let mut epoch = (epoch_word[0] + 1) & 0xffff;
    if epoch == 0 {
        cursors.fill(0);
        epoch = 1;
    }
    epoch_word[0] = epoch;
    let mut i = 0usize;
    while i < tokens.len() {
        let (match_len, distance) = find_match_ring(tokens, i, entries, cursors, max_probes, epoch);
        let threshold = if distance <= 16 { 4 } else { 5 };
        if match_len >= threshold {
            let distance_value = if distance == 1 {
                LZ77_DIST_VALUE
            } else {
                LZ77_NUM_SPECIAL_DISTANCES + distance as u32 - 1
            };
            out.push(LzToken::lz77(
                tokens[i].context,
                match_len as u32 - LZ77_MIN_LENGTH,
                distance_value,
            ));
            if out.len() > max_output {
                return false;
            }
            for pos in i..i + match_len {
                ring_insert(entries, cursors, ring_hash(tokens, pos), pos, epoch);
            }
            i += match_len;
        } else {
            let token = tokens[i];
            out.push(LzToken::pixel(token.context, token.value));
            if out.len() > max_output {
                return false;
            }
            ring_insert(entries, cursors, ring_hash(tokens, i), i, epoch);
            i += 1;
        }
    }
    true
}

/// `context_map: None` accumulates per raw context (identity), which supports
/// context counts beyond the u8 map range used after clustering.
fn lz_add_histograms<I>(
    toks: I,
    context_map: Option<&[u8]>,
    histograms: &mut [Histogram],
    min_symbol: u32,
    distance_context: u32,
) where
    I: IntoIterator<Item = LzToken>,
{
    let slot = |context: u32| -> usize {
        match context_map {
            Some(map) => map[context as usize] as usize,
            None => context as usize,
        }
    };
    for t in toks {
        if t.is_lz77() {
            let (len_tok, _, _) = lz77_length_encode(t.value);
            histograms[slot(t.context)].add(min_symbol + len_tok);

            let (symbol, _, _) = crate::entropy::uint_encode(t.distance);
            histograms[slot(distance_context)].add(symbol);
        } else {
            let (sym, _, _) = crate::entropy::uint_encode(t.value);
            histograms[slot(t.context)].add(sym);
        }
    }
}

/// Build per-cluster prefix codes from an `LzToken` stream.
/// `nb_chans + 1` contexts: `nb_chans` channel leaves + 1 distance context.
pub(super) fn build_lz_pixel_code<'tokens, 'scratch, I>(
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
    let distance_context = nb_chans as u32;
    let num_contexts = nb_chans + 1;
    assert!(num_contexts <= LZ77_MAX_CONTEXTS);
    let use_ans = refined;
    let LzEntropyScratch {
        histograms,
        prefix_codes,
        context_map,
        configs,
        clustering,
        ans,
    } = scratch;
    let histograms = &mut histograms[..num_contexts];
    histograms.fill(Histogram::new());
    for toks in streams.clone() {
        lz_add_histograms(
            toks.iter().copied(),
            None,
            histograms,
            min_symbol,
            distance_context,
        );
    }

    let num_clusters = cluster_histograms_fixed(
        histograms,
        &mut context_map[..num_contexts],
        refined,
        clustering,
        huffman_pool,
    );
    let histograms = &mut histograms[..num_clusters];
    let configs = &mut configs[..num_clusters];

    if refined {
        let mut raw_values = vec![Vec::<u32>::new(); num_clusters];
        let mut literal_values = vec![Vec::<u32>::new(); num_clusters];
        for toks in streams.clone() {
            for &tok in toks {
                if tok.is_lz77() {
                    raw_values[context_map[distance_context as usize] as usize].push(tok.distance);
                } else {
                    let cluster = context_map[tok.context as usize] as usize;
                    raw_values[cluster].push(tok.value);
                    literal_values[cluster].push(tok.value);
                }
            }
        }
        let selected_configs = crate::entropy::select_hybrid_configs_ans(&raw_values);
        for (cluster, selected) in selected_configs.into_iter().enumerate() {
            configs[cluster] = if literal_values[cluster].iter().all(|&value| {
                crate::entropy::uint_encode_with_config(value, selected).0 < min_symbol
            }) {
                selected
            } else {
                crate::entropy::HybridUintConfig::DEFAULT
            };
        }
        histograms.fill(Histogram::new());
        for toks in streams {
            for &tok in toks {
                if tok.is_lz77() {
                    let (len_tok, _, _) = lz77_length_encode(tok.value);
                    histograms[context_map[tok.context as usize] as usize]
                        .add(min_symbol + len_tok);
                    let cluster = context_map[distance_context as usize] as usize;
                    let (symbol, _, _) =
                        crate::entropy::uint_encode_with_config(tok.distance, configs[cluster]);
                    histograms[cluster].add(symbol);
                } else {
                    let cluster = context_map[tok.context as usize] as usize;
                    let (symbol, _, _) =
                        crate::entropy::uint_encode_with_config(tok.value, configs[cluster]);
                    histograms[cluster].add(symbol);
                }
            }
        }
    } else {
        configs.fill(crate::entropy::HybridUintConfig::DEFAULT);
    }

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
        let (hists, symbols, reverse_maps) = crate::entropy::build_ans_code_parts(histograms);
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
    }
}

/// Emit a full section's `LzToken` stream: prefix codes stream token-by-token;
/// rANS buffers the section (one ANS state per section, symbols pushed in
/// reverse) exactly like `write_ans_tokens`, with the LZ77 symbol mapping of
/// `write_lz_token` (length = direct symbol `min_symbol + len_tok`; literals
/// and distances honor the per-cluster hybrid-uint config).
pub(super) fn write_lz_section(
    tokens: &[LzToken],
    distance_context: u32,
    code: &EntropyCode<'_>,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    if code.use_prefix_code {
        for t in tokens {
            write_lz_token(*t, distance_context, code, min_symbol, w);
        }
        return;
    }

    const NO_EMIT: u32 = u32::MAX;
    struct Slot {
        sym: u8,
        nbits: u8,
        hist: u8,
        bits: u32,
        emitted: u32,
    }
    let expanded_len = tokens.len() + tokens.iter().filter(|token| token.is_lz77()).count();
    let mut prepared: Vec<Slot> = Vec::with_capacity(expanded_len);
    let push = |prepared: &mut Vec<Slot>, hist: u8, sym: u32, nbits: u32, bits: u32| {
        debug_assert!(sym < ALPHABET_SIZE as u32);
        debug_assert!(nbits <= u8::MAX as u32);
        prepared.push(Slot {
            sym: sym as u8,
            nbits: nbits as u8,
            hist,
            bits,
            emitted: NO_EMIT,
        });
    };
    for t in tokens {
        if t.is_lz77() {
            let (len_tok, len_nbits, len_bits) = lz77_length_encode(t.value);
            let hist = code.context_map[t.context as usize];
            push(
                &mut prepared,
                hist,
                min_symbol + len_tok,
                len_nbits,
                len_bits,
            );
            let dhist = code.context_map[distance_context as usize];
            let (sym, nbits, bits) = crate::entropy::uint_encode_with_config(
                t.distance,
                code.hybrid_uint_configs[dhist as usize],
            );
            push(&mut prepared, dhist, sym, nbits, bits);
        } else {
            let hist = code.context_map[t.context as usize];
            let (sym, nbits, bits) = crate::entropy::uint_encode_with_config(
                t.value,
                code.hybrid_uint_configs[hist as usize],
            );
            push(&mut prepared, hist, sym, nbits, bits);
        }
    }

    let mut coder = crate::entropy::AnsCoder::new();
    for slot in prepared.iter_mut().rev() {
        let hist = slot.hist as usize;
        let reverse_start = hist * crate::entropy::ANS_TAB_SIZE as usize;
        let reverse_map = &code.ans_reverse_maps
            [reverse_start..reverse_start + crate::entropy::ANS_TAB_SIZE as usize];
        let info = &code.ans_symbols[hist][slot.sym as usize];
        if let Some(word) = coder.put_symbol(info, reverse_map) {
            slot.emitted = word as u32;
        }
    }
    w.write(32, coder.state() as u64);
    for slot in prepared {
        if slot.emitted != NO_EMIT {
            w.write(16, slot.emitted as u64);
        }
        w.write(slot.nbits as usize, slot.bits as u64);
    }
}

/// Emit one `LzToken` into the bitstream.
#[inline]
pub(super) fn write_lz_token(
    t: LzToken,
    distance_context: u32,
    code: &EntropyCode<'_>,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    if t.is_lz77() {
        let (len_tok, len_nbits, len_bits) = lz77_length_encode(t.value);
        let sym = min_symbol + len_tok;
        let pcluster = code.context_map[t.context as usize] as usize;
        let pc = &code.prefix_codes[pcluster];
        let d = pc.depths[sym as usize] as usize;
        debug_assert!(
            d > 0,
            "LZ77 length symbol {} unrepresented in histogram",
            sym
        );
        let data = (pc.bits[sym as usize] as u64) | ((len_bits as u64) << d);
        w.write(d + len_nbits as usize, data);

        let dcluster = code.context_map[distance_context as usize] as usize;
        let dc = &code.prefix_codes[dcluster];
        let (dist_symbol, dist_nbits, dist_bits) =
            crate::entropy::uint_encode_with_config(t.distance, code.hybrid_uint_configs[dcluster]);
        let dd = dc.depths[dist_symbol as usize] as usize;
        if dd > 0 {
            let data = dc.bits[dist_symbol as usize] as u64 | ((dist_bits as u64) << dd);
            w.write(dd + dist_nbits as usize, data);
        } else if dist_nbits != 0 {
            w.write(dist_nbits as usize, dist_bits as u64);
        }
    } else {
        let cluster = code.context_map[t.context as usize] as usize;
        let (sym, nbits, bits) =
            crate::entropy::uint_encode_with_config(t.value, code.hybrid_uint_configs[cluster]);
        let pc = &code.prefix_codes[cluster];
        let d = pc.depths[sym as usize] as usize;
        let data = (pc.bits[sym as usize] as u64) | ((bits as u64) << d);
        w.write(d + nbits as usize, data);
    }
}
/// Write the LZ77 sub-bundle (matches `LZ77Params::VisitFields` and `DecodeUintConfig`):
///   1 bit:  enabled = 1
///   U32 min_symbol:  U32(Val(224), Val(512), Val(4096), BitsOffset(15, 8))
///                    For LZ77_MIN_SYMBOL = 64: selector 3 → "11" + 15 bits (64 - 8) = 56
///   U32 min_length:  U32(Val(3), Val(4), BitsOffset(2, 5), BitsOffset(8, 9))
///                    For LZ77_MIN_LENGTH = 3: selector 0 → "00"
///   length_uint_config: DecodeUintConfig(log_alpha_size = 8).
///                       split_exp = 4 (in CeilLog2(8+1) = 4 bits = "0010" LSB-first),
///                       msb_in_token = 0 (in CeilLog2(4+1) = 3 bits = "000"),
///                       lsb_in_token = 0 (in CeilLog2(4-0+1) = 3 bits = "000").
fn write_lz77_header(min_symbol: u32, w: &mut BitWriter) {
    w.write(1, 1); // enabled
    // min_symbol: selector 3 (Bits(15) + 8), value 64 → payload = 56
    w.write(2, 0b11);
    w.write(15, (min_symbol - 8) as u64);
    // min_length: selector 0 (Val(3))
    w.write(2, 0b00);
    // length_uint_config (split=4, msb=0, lsb=0):
    w.write(4, 4);
    w.write(3, 0);
    w.write(3, 0);
}

/// Write the local tree + LZ77-enabled pixel histograms, then return.
/// The pixel `code` must have `nb_chans + 1` contexts (last = distance).
pub(super) fn write_local_tree_lz77(
    predictors: &[u32],
    pixel_code: &EntropyCode<'_>,
    min_symbol: u32,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut BitWriter,
) {
    let tree_tokens = build_balanced_tree_tokens(predictors);
    write_tree_lz77(&tree_tokens, pixel_code, min_symbol, huffman_pool, w);
}

/// Write a pre-built MA tree (token stream) + the LZ77 pixel code header.
pub(super) fn write_tree_lz77(
    tree_tokens: &[Token],
    pixel_code: &EntropyCode<'_>,
    min_symbol: u32,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut BitWriter,
) {
    let tree_code = optimize_entropy_code(tree_tokens, NUM_TREE_CONTEXTS, huffman_pool);
    let tree_code_ref = tree_code.as_ref();

    // Tree's entropy code: no LZ77 in the tree itself.
    w.write(1, 0);
    write_entropy_code(&tree_code_ref, huffman_pool, w);
    for tok in tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }

    // Pixel entropy code: LZ77 ENABLED for the main bitstream.
    write_lz77_header(min_symbol, w);
    // The decoder appends an extra context (distance) when LZ77 is on, so the
    // context map we write must already include it as its last entry.
    write_entropy_code(pixel_code, huffman_pool, w);
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let clean = std::thread::spawn({
            let small = small.clone();
            move || {
                let mut scratch = Vec::new();
                let mut out = Vec::with_capacity(small.len());
                lz77_compress_with_depth_into(&small, 8, &mut scratch, &mut out);
                out.len()
            }
        })
        .join()
        .unwrap();
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
}
