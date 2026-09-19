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
use super::{NUM_TREE_CONTEXTS, build_balanced_tree_tokens, build_balanced_tree_tokens_offsets};
use crate::adaptive_quant::dirty_log2f;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS, LzEntropyScratch};
use crate::entropy::{
    CompactToken, EntropyCode, Histogram, PrefixCode, Token, build_huffman_codes_into,
    cluster_histograms_fixed, f_log2, optimize_entropy_code, write_entropy_code, write_token,
};
use crate::thread_pool::ThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) const LZ77_MIN_SYMBOL: u32 = 64;
pub(super) const LZ77_MIN_LENGTH: u32 = 3;
// special_distance[1] = (dx=1, dy=0) -> one token back.
pub(super) const LZ77_DIST_VALUE: u32 = 1;
pub(super) const LZ77_NUM_SPECIAL_DISTANCES: u32 = 120;
/// The decoder's LZ77 window (libjxl `kWindowSize`): a copy may reach at
/// most this many tokens back, and the decoder clamps longer distances to
/// it, so a match further back must not be emitted. A group stream (three
/// 1024x1024 channels) is three times longer than the window.
pub(super) const LZ77_WINDOW: usize = 1 << 20;

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
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
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

/// Token storage used by the LZ entropy passes. Literal streams retain their
/// wide or compact representation instead of expanding them to LzToken.
pub(super) trait LzTokenSource: Copy + Sync {
    fn as_lz(self) -> LzToken;
}
impl LzTokenSource for LzToken {
    #[inline]
    fn as_lz(self) -> LzToken {
        self
    }
}
impl LzTokenSource for Token {
    #[inline]
    fn as_lz(self) -> LzToken {
        LzToken::pixel(self.context, self.value)
    }
}
impl LzTokenSource for CompactToken {
    #[inline]
    fn as_lz(self) -> LzToken {
        self.unpack().as_lz()
    }
}

/// A run of value-0 literal tokens in one context that follows a stream's
/// stored tokens: a constant channel coded by a Zero-predictor leaf whose
/// offset is the constant. The run is never materialized or matched by
/// LZ77, and its context is clustered alone, so its histogram has one
/// symbol and the decoder fills the channel without reading a symbol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ConstantTail {
    pub(super) context: u32,
    pub(super) count: usize,
}

impl ConstantTail {
    pub(super) const NONE: Self = Self {
        context: 0,
        count: 0,
    };
}

/// Literal storage preserves the original token hash and comparisons. Only
/// equality scanning uses the representation-specific SIMD kernel.
pub(super) trait LiteralToken: LzTokenSource {
    fn as_token(self) -> Token;
    fn same(self, other: Self) -> bool;
    fn match_kernel() -> crate::lz_match::MatchKernel<Self>;
}
impl LiteralToken for Token {
    #[inline]
    fn as_token(self) -> Token {
        self
    }
    #[inline]
    fn same(self, other: Self) -> bool {
        // An LZ77 copy reproduces past VALUES; the copied pixels' contexts
        // play no part (the decoder reads no symbols for them), so matches
        // may span leaf changes. libjxl's hash chain matches values only.
        self.value == other.value
    }
    fn match_kernel() -> crate::lz_match::MatchKernel<Self> {
        crate::lz_match::selected()
    }
}
impl LiteralToken for CompactToken {
    #[inline]
    fn as_token(self) -> Token {
        self.unpack()
    }
    #[inline]
    fn same(self, other: Self) -> bool {
        self.same_value(other)
    }
    fn match_kernel() -> crate::lz_match::MatchKernel<Self> {
        crate::lz_match::selected_compact()
    }
}

#[inline]
fn lz_tokens<T: LzTokenSource>(
    tokens: &[T],
) -> impl ExactSizeIterator<Item = LzToken> + DoubleEndedIterator + Clone + '_ {
    tokens.iter().copied().map(T::as_lz)
}

#[inline]
fn fingerprint<T: LiteralToken>(tokens: &[T], pos: usize) -> u32 {
    let mut hash = 0x9e37_79b9u32;
    for token in &tokens[pos..tokens.len().min(pos + 3)] {
        let token = token.as_token();
        hash ^= token.value.wrapping_mul(0x85eb_ca6b).rotate_left(13);
        hash = hash.wrapping_mul(0xc2b2_ae35);
    }
    hash
}

fn has_repetition<T: LiteralToken>(tokens: &[T], scratch: &mut Vec<u32>) -> bool {
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

const RING_HASH_BITS: usize = 17;
const RING_HASH_SIZE: usize = 1 << RING_HASH_BITS;
const RING_BUCKET: usize = 8;
const RING_ENTRIES_LEN: usize = RING_HASH_SIZE * RING_BUCKET;
const DEEP_LZ_SCRATCH_WORDS: usize = RING_ENTRIES_LEN + RING_HASH_SIZE + 1;

#[inline]
fn ring_hash<T: LiteralToken>(tokens: &[T], pos: usize) -> usize {
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

fn find_match_ring<T: LiteralToken>(
    tokens: &[T],
    pos: usize,
    entries: &[u32],
    cursors: &mut [u32],
    max_probes: usize,
    epoch: u32,
    match_kernel: crate::lz_match::MatchKernel<T>,
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
    let limit = (tokens.len() - pos).min(LZ77_WINDOW);
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    // Newest-first scan: on equal lengths the nearest (cheapest) distance wins
    // for free.
    for k in 1..=live.min(max_probes) {
        let slot = (idx + RING_BUCKET - k) % RING_BUCKET;
        let candidate_pos = entries[base + slot] as usize;
        // Older candidates only lie further back than the decoder's window.
        if pos - candidate_pos > LZ77_WINDOW {
            break;
        }
        // Most hash collisions fail immediately; avoid dispatch and vector
        // setup until the first complete token matches.
        if !tokens[candidate_pos].same(tokens[pos]) {
            continue;
        }
        // Only a strictly longer match can replace the newest candidate.
        // A mismatch at the current best length proves this candidate cannot
        // win, without scanning its matching prefix again.
        if best_len != 0 && !tokens[candidate_pos + best_len].same(tokens[pos + best_len]) {
            continue;
        }
        let len = 1 + match_kernel(
            &tokens[candidate_pos + 1..candidate_pos + limit],
            &tokens[pos + 1..pos + limit],
        );
        if len > best_len {
            best_len = len;
            best_dist = pos - candidate_pos;
            if best_len == limit {
                // No older candidate can beat the length cap or remaining
                // input, and equal lengths keep the nearer distance.
                break;
            }
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
struct RunLzTokens<'a, T: LiteralToken> {
    tokens: &'a [T],
    pos: usize,
    pending: Option<LzToken>,
}

impl<'a, T: LiteralToken> RunLzTokens<'a, T> {
    fn new(tokens: &'a [T]) -> Self {
        Self {
            tokens,
            pos: 0,
            pending: None,
        }
    }
}

impl<T: LiteralToken> Iterator for RunLzTokens<'_, T> {
    type Item = LzToken;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.pending.take() {
            return Some(token);
        }
        let raw_token = *self.tokens.get(self.pos)?;
        let token = raw_token.as_token();
        let mut end = self.pos + 1;
        while end < self.tokens.len() && self.tokens[end].same(raw_token) {
            end += 1;
        }
        let copied = end - self.pos - 1;
        self.pos += 1;
        if copied >= LZ77_MIN_LENGTH as usize {
            // Runs are value-equal, not context-equal: the copy's length
            // symbol is coded in the context of its first copied position.
            let copy_context = self.tokens[self.pos].as_token().context;
            self.pending = Some(LzToken::lz77(
                copy_context,
                copied as u32 - LZ77_MIN_LENGTH,
                LZ77_DIST_VALUE,
            ));
            self.pos = end;
        }
        Some(LzToken::pixel(token.context, token.value))
    }
}

/// Equal-token spans, kept separate for each group/stream. Short spans still
/// emit individual literals under the existing run-compression policy.
struct TokenRuns<'a, T: LiteralToken>(&'a [T]);

impl<T: LiteralToken> Iterator for TokenRuns<'_, T> {
    type Item = (Token, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let (&token, tail) = self.0.split_first()?;
        let count = 1 + tail
            .iter()
            .position(|&next| !next.same(token))
            .unwrap_or(tail.len());
        self.0 = &self.0[count..];
        Some((token.as_token(), count))
    }
}

pub(super) fn lz77_run_count<T: LiteralToken>(tokens: &[T]) -> usize {
    TokenRuns(tokens)
        .map(|(_, count)| {
            if count > LZ77_MIN_LENGTH as usize {
                2
            } else {
                count
            }
        })
        .sum()
}

/// Streaming equivalent of `RunLzTokens`. It retains only the current equal
/// token run, allowing prediction/tokenization to emit directly into the final
/// LZ stream without staging a raw-token plane.
pub(super) struct RunLzWriter {
    out: Vec<LzToken>,
    token: Option<Token>,
    count: usize,
    /// Contexts of the run's second and third tokens: a copy's length symbol
    /// is coded in the first copied position's context, and a run too short
    /// to copy emits its tokens as literals in their own contexts.
    next_contexts: [u32; LZ77_MIN_LENGTH as usize],
}

impl RunLzWriter {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            out: Vec::with_capacity(capacity),
            token: None,
            count: 0,
            next_contexts: [0; LZ77_MIN_LENGTH as usize],
        }
    }

    #[inline]
    pub(super) fn push(&mut self, token: Token) {
        // Runs are value-equal (an LZ77 copy reproduces values whatever the
        // contexts), matching `RunLzTokens`.
        if self
            .token
            .is_some_and(|current| current.value == token.value)
        {
            if self.count <= LZ77_MIN_LENGTH as usize {
                self.next_contexts[self.count - 1] = token.context;
            }
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
                self.next_contexts[0],
                copied as u32 - LZ77_MIN_LENGTH,
                LZ77_DIST_VALUE,
            ));
        } else {
            for &context in &self.next_contexts[..copied] {
                self.out.push(LzToken::pixel(context, token.value));
            }
        }
        self.count = 0;
    }
}

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
        None,
    )
}

pub(super) fn lz77_compress_for_speed_with_depth<T: LiteralToken>(
    tokens: &[T],
    distance_context: u32,
    speed: crate::Speed,
    depth: &mut Vec<u32>,
    run_count: Option<usize>,
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
        run_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn lz77_compress_for_speed_with_parts<T: LiteralToken>(
    tokens: &[T],
    distance_context: u32,
    speed: crate::Speed,
    lz_repetitions: &mut Vec<u32>,
    lz_depth: &mut Vec<u32>,
    lz_candidate: &mut Vec<LzToken>,
    lz_entropy: &mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    run_count: Option<usize>,
) -> Vec<LzToken> {
    let run_capacity = run_count.unwrap_or_else(|| tokens.len().min(16 * 1024));
    // Fast stays runs-only
    if speed != crate::Speed::Slow || !has_repetition(tokens, lz_repetitions) {
        return lz77_compress_runs_with_capacity(tokens, run_capacity);
    }
    let run_tokens = lz77_compress_runs_with_capacity(tokens, run_capacity);
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

// Retain a worker's high-water capacity without geometric over-allocation
// when a later tree needs more contexts.
fn grow_entropy_scratch<T: Clone>(values: &mut Vec<T>, len: usize, value: T) {
    if values.len() < len {
        values.reserve_exact(len - values.len());
        values.resize(len, value);
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
    grow_entropy_scratch(&mut scratch.histograms, num_contexts, Histogram::new());
    grow_entropy_scratch(&mut scratch.prefix_codes, num_contexts, PrefixCode::zero());
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

#[cfg(test)]
pub(super) fn lz77_compress_runs(tokens: &[Token]) -> Vec<LzToken> {
    // Flat regions can collapse millions of input tokens to a handful of
    // runs. Grow with the emitted stream instead of reserving the raw bound.
    lz77_compress_runs_with_capacity(tokens, tokens.len().min(16 * 1024))
}

fn lz77_compress_runs_with_capacity<T: LiteralToken>(
    tokens: &[T],
    capacity: usize,
) -> Vec<LzToken> {
    let mut out = Vec::with_capacity(capacity);
    lz77_extend_runs(tokens, &mut out);
    out
}

pub(super) fn lz77_extend_runs<T: LiteralToken>(tokens: &[T], out: &mut Vec<LzToken>) {
    let max_len = out.len() + tokens.len();
    for token in RunLzTokens::new(tokens) {
        if out.len() == out.capacity() {
            // Geometric growth would otherwise exceed even the raw bound
            // on dense streams, undoing the savings on photographic inputs.
            let additional = out.capacity().max(4).min(max_len - out.len());
            out.reserve_exact(additional);
        }
        out.push(token);
    }
}

pub(super) fn lz77_compress_runs_channels(channels: Vec<Vec<Token>>) -> Vec<LzToken> {
    let total_len: usize = channels.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total_len.min(16 * 1024));
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
    lz77_compress_for_speed_with_depth(&tokens, distance_context, speed, depth, None, scratch)
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
fn lz77_compress_with_depth_into_limit<T: LiteralToken>(
    tokens: &[T],
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
    let match_kernel = T::match_kernel();
    let mut i = 0usize;
    while i < tokens.len() {
        let (match_len, distance) =
            find_match_ring(tokens, i, entries, cursors, max_probes, epoch, match_kernel);
        let threshold = if distance <= 16 { 4 } else { 5 };
        if match_len >= threshold {
            let distance_value = if distance == 1 {
                LZ77_DIST_VALUE
            } else {
                LZ77_NUM_SPECIAL_DISTANCES + distance as u32 - 1
            };
            out.push(LzToken::lz77(
                tokens[i].as_token().context,
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
            let token = tokens[i].as_token();
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

/// Materialize literals for paths that own LZ tokens. Learned-tree candidates
/// can instead use the original Token storage through LzTokenSource.
pub(super) fn lz77_literals(tokens: &[Token]) -> Vec<LzToken> {
    tokens
        .iter()
        .map(|t| LzToken::pixel(t.context, t.value))
        .collect()
}

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
    build_lz_pixel_code_opts(
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
pub(super) fn build_lz_pixel_code_opts<'tokens, 'scratch, I>(
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

/// Split `streams` into `threads` contiguous chunks and reduce a per-chunk
/// accumulator. Keep chunk boundaries and merge order stable: coded-bit
/// estimates use floating-point sums, while histogram/count passes use integers.
fn parallel_streams<R: Send, T: LzTokenSource>(
    streams: &[&[T]],
    threads: usize,
    init: impl Fn() -> R + Sync,
    accumulate: impl Fn(&mut R, &[T]) + Sync,
    merge: impl FnMut(&mut R, R),
) -> R {
    let threads = threads.max(1).min(streams.len().max(1));
    if threads <= 1 {
        let mut acc = init();
        for toks in streams {
            accumulate(&mut acc, toks);
        }
        return acc;
    }
    let chunk = streams.len().div_ceil(threads);
    let parts = scoped_chunks(streams.len().div_ceil(chunk), "stream worker", &|ci| {
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
pub(super) fn estimate_literal_and_run_bits_single<T: LiteralToken>(
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
    for (token, count) in TokenRuns(tokens) {
        let (symbol, nbits, _) = crate::entropy::uint_encode(token.value);
        let context = token.context as usize;
        literal[context].counts[symbol as usize] += count as u32;
        literal[context].total_count += count as u32;
        *literal_bits += nbits as u64 * count as u64;
        if count > LZ77_MIN_LENGTH as usize {
            runs[context].add(symbol);
            let (length, length_bits, _) = lz77_length_encode((count - 1) as u32 - LZ77_MIN_LENGTH);
            runs[context].add(min_symbol + length);
            let (distance, distance_bits, _) = crate::entropy::uint_encode(LZ77_DIST_VALUE);
            runs[distance_context].add(distance);
            *run_bits += nbits as u64 + length_bits as u64 + distance_bits as u64;
        } else {
            runs[context].counts[symbol as usize] += count as u32;
            runs[context].total_count += count as u32;
            *run_bits += nbits as u64 * count as u64;
        }
    }
}

/// Exact order-0 estimates for both variants without storing the run stream.
/// Repeated literal symbols contribute their counts and extra bits in bulk;
/// floating-point entropy is evaluated only after integer histogram merging.
pub(super) fn estimate_literal_and_run_bits<T: LiteralToken>(
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
pub(super) fn estimate_streams_bits<T: LzTokenSource>(
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
pub(super) fn estimate_coded_bits<T: LzTokenSource>(
    streams: &[&[T]],
    distance_context: u32,
    code: &EntropyCode<'_>,
    min_symbol: u32,
    threads: usize,
) -> f64 {
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
        threads,
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

/// Map every stream independently over `threads` scoped threads, results in
/// stream order.
fn map_streams<R: Send, T: LzTokenSource>(
    streams: &[&[T]],
    threads: usize,
    f: impl Fn(&[T]) -> R + Sync,
) -> Vec<R> {
    map_streams_indexed(streams, threads, |_, toks| f(toks))
}

fn map_streams_indexed<R: Send, T: LzTokenSource>(
    streams: &[&[T]],
    threads: usize,
    f: impl Fn(usize, &[T]) -> R + Sync,
) -> Vec<R> {
    let threads = threads.max(1).min(streams.len().max(1));
    if threads <= 1 {
        return streams.iter().enumerate().map(|(i, t)| f(i, t)).collect();
    }
    let chunk = streams.len().div_ceil(threads);
    scoped_chunks(streams.len().div_ceil(chunk), "stream mapper", &|ci| {
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

// Callback dispatch is once per thread, outside token and stream loops.
fn scoped_chunks<R: Send>(
    count: usize,
    panic_message: &'static str,
    run: &(dyn Fn(usize) -> R + Sync),
) -> Vec<R> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..count).map(|i| scope.spawn(move || run(i))).collect();
        handles
            .into_iter()
            .map(|h| h.join().expect(panic_message))
            .collect()
    })
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
pub(super) fn build_lz_pixel_code_threads<'tokens, 'scratch, I, T: LzTokenSource + 'tokens>(
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
        &streams,
        &[],
        nb_chans,
        min_symbol,
        refined,
        ans_cluster,
        pool,
        scratch,
        huffman_pool,
    )
}

/// `build_lz_pixel_code_threads` for streams that each end in a constant
/// tail (`tails[k]` follows `streams[k]`; `ConstantTail::NONE` for none).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_lz_pixel_code_tails<'tokens, 'scratch, T: LzTokenSource + 'tokens>(
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
        streams,
        tails,
        nb_chans,
        min_symbol,
        refined,
        ans_cluster,
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

// Share entropy setup across iterator adapters while retaining specialized
// token passes for each token representation.
#[allow(clippy::too_many_arguments)]
fn build_lz_pixel_code_slices<'scratch, T: LzTokenSource>(
    streams: &[&[T]],
    tails: &[ConstantTail],
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
    ans_cluster: bool,
    pool: Option<&ThreadPool>,
    scratch: &'scratch mut LzEntropyScratch,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> EntropyCode<'scratch> {
    let threads = pool.map_or(1, ThreadPool::num_threads);
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
    grow_entropy_scratch(histograms, num_contexts, Histogram::new());
    let histograms = &mut histograms[..num_contexts];
    histograms.fill(Histogram::new());
    {
        let summed = parallel_streams(
            streams,
            threads,
            || vec![Histogram::new(); num_contexts],
            |acc, toks| lz_add_histograms(lz_tokens(toks), None, acc, min_symbol, distance_context),
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
    let max_clusters = crate::entropy::CLUSTERS_LIMIT - pinned.len();
    let mut num_clusters = if ans_cluster {
        crate::entropy::cluster_histograms_ans(
            histograms,
            &mut context_map[..num_contexts],
            pool,
            false,
            max_clusters,
            2,
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

    if refined {
        let distance_cluster = context_map[distance_context as usize] as usize;
        let context_map_ref: &[u8] = &context_map[..num_contexts];
        {
            let per_stream_counts: Vec<Vec<usize>> = map_streams(streams, threads, |toks| {
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
            let per_stream_values: Vec<Vec<Vec<u32>>> =
                map_streams_indexed(streams, threads, |s, toks| {
                    let mut ordinal = offsets[s].clone();
                    let mut local: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
                    for tok in lz_tokens(toks) {
                        let (cluster, value) = if tok.is_lz77() {
                            (distance_cluster, tok.distance)
                        } else {
                            (context_map_ref[tok.context as usize] as usize, tok.value)
                        };
                        if ordinal[cluster].is_multiple_of(strides_ref[cluster]) {
                            local[cluster].push(value);
                        }
                        ordinal[cluster] += 1;
                    }
                    local
                });
            let mut values: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
            for local in per_stream_values {
                for (dst, src) in values.iter_mut().zip(local) {
                    dst.extend(src);
                }
            }
            let samples = crate::entropy::HybridUintSamples::from_parts(values, strides);
            configs.copy_from_slice(&samples.select());
        }
        // Validate every literal against the chosen config, including values
        // outside the sampled ordinals. No duplicate literal vector is needed.
        let configs_ref: &[crate::entropy::HybridUintConfig] = configs;
        let valid = parallel_streams(
            streams,
            threads,
            || vec![true; num_clusters],
            |acc, toks| {
                for tok in lz_tokens(toks) {
                    if !tok.is_lz77() {
                        let cluster = context_map_ref[tok.context as usize] as usize;
                        if acc[cluster]
                            && crate::entropy::uint_encode_with_config(
                                tok.value,
                                configs_ref[cluster],
                            )
                            .0 >= min_symbol
                        {
                            acc[cluster] = false;
                        }
                    }
                }
            },
            |acc, part| {
                for (a, b) in acc.iter_mut().zip(part) {
                    *a &= b;
                }
            },
        );
        for (config, valid) in configs.iter_mut().zip(valid) {
            if !valid {
                *config = crate::entropy::HybridUintConfig::DEFAULT;
            }
        }
        let configs_ref: &[crate::entropy::HybridUintConfig] = configs;
        let summed = parallel_streams(
            streams,
            threads,
            || vec![Histogram::new(); num_clusters],
            |acc, toks| {
                for tok in lz_tokens(toks) {
                    if tok.is_lz77() {
                        let (len_tok, _, _) = lz77_length_encode(tok.value);
                        acc[context_map_ref[tok.context as usize] as usize]
                            .add(min_symbol + len_tok);
                        let cluster = context_map_ref[distance_context as usize] as usize;
                        let (symbol, _, _) = crate::entropy::uint_encode_with_config(
                            tok.distance,
                            configs_ref[cluster],
                        );
                        acc[cluster].add(symbol);
                    } else {
                        let cluster = context_map_ref[tok.context as usize] as usize;
                        let (symbol, _, _) = crate::entropy::uint_encode_with_config(
                            tok.value,
                            configs_ref[cluster],
                        );
                        acc[cluster].add(symbol);
                    }
                }
            },
            |acc, part| merge_histograms(acc, &part),
        );
        histograms.clone_from_slice(&summed);
        for &(context, count) in &tail_counts {
            let cluster = context_map_ref[context as usize] as usize;
            histograms[cluster].counts[0] += count;
            histograms[cluster].total_count += count;
        }
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
    }
}

/// Emit a full section's `LzToken` stream: prefix codes stream token-by-token;
/// rANS buffers the section (one ANS state per section, symbols pushed in
/// reverse) exactly like `write_ans_tokens`, with the LZ77 symbol mapping of
/// `write_lz_token` (length = direct symbol `min_symbol + len_tok`; literals
/// and distances honor the per-cluster hybrid-uint config).
pub(super) fn write_lz_section<T: LzTokenSource>(
    tokens: &[T],
    distance_context: u32,
    code: &EntropyCode<'_>,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    write_lz_section_tail(
        tokens,
        ConstantTail::NONE,
        distance_context,
        code,
        min_symbol,
        w,
    );
}

/// `write_lz_section` with a constant tail after the stored tokens.
pub(super) fn write_lz_section_tail<T: LzTokenSource>(
    tokens: &[T],
    tail: ConstantTail,
    distance_context: u32,
    code: &EntropyCode<'_>,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    if code.use_prefix_code {
        for t in lz_tokens(tokens) {
            write_lz_token(t, distance_context, code, min_symbol, w);
        }
        if tail.count > 0 {
            let hist = code.context_map[tail.context as usize] as usize;
            let (sym, nbits, _) =
                crate::entropy::uint_encode_with_config(0, code.hybrid_uint_configs[hist]);
            // A one-symbol prefix code has a zero-length codeword.
            if nbits > 0 || code.prefix_codes[hist].depths[sym as usize] > 0 {
                for _ in 0..tail.count {
                    write_lz_token(
                        LzToken::pixel(tail.context, 0),
                        distance_context,
                        code,
                        min_symbol,
                        w,
                    );
                }
            }
        }
        return;
    }

    // The tail's symbols: none at all when its cluster has the one symbol
    // (a single-symbol table moves neither the ANS state nor the stream).
    let tail_hist = code.context_map[tail.context as usize] as usize;
    let (tail_sym, tail_nbits, tail_bits) =
        crate::entropy::uint_encode_with_config(0, code.hybrid_uint_configs[tail_hist]);
    let tail_count = if tail.count > 0
        && (tail_nbits > 0
            || code.ans_symbols[tail_hist][tail_sym as usize].freq as u32
                != crate::entropy::ANS_TAB_SIZE)
    {
        tail.count
    } else {
        0
    };

    let expanded_len =
        tokens.len() + lz_tokens(tokens).filter(|token| token.is_lz77()).count() + tail_count;
    // Most symbols in a nearly deterministic context emit no ANS word.
    // Keep one presence bit per symbol and only the actual 16-bit words.
    let mut present = vec![0u64; expanded_len.div_ceil(64)];
    let mut emitted = Vec::with_capacity(expanded_len.min(16 * 1024));
    let mut ordinal = 0usize;
    let mut coder = crate::entropy::AnsCoder::new();
    let mut put = |coder: &mut crate::entropy::AnsCoder, hist: usize, sym: u32| {
        let start = hist * crate::entropy::ANS_TAB_SIZE as usize;
        if let Some(word) = coder.put_symbol(
            &code.ans_symbols[hist][sym as usize],
            &code.ans_reverse_maps[start..start + crate::entropy::ANS_TAB_SIZE as usize],
        ) {
            present[ordinal / 64] |= 1u64 << (ordinal % 64);
            emitted.push(word);
        }
        ordinal += 1;
    };
    // A copy's forward order is length, distance; advance the same ANS state
    // in the reverse order. Presence bits and words use that same order.
    // The tail follows every stored token, so it enters the state first.
    for _ in 0..tail_count {
        put(&mut coder, tail_hist, tail_sym);
    }
    for t in lz_tokens(tokens).rev() {
        let hist = code.context_map[t.context as usize] as usize;
        if t.is_lz77() {
            let dhist = code.context_map[distance_context as usize] as usize;
            let (sym, _, _) = crate::entropy::uint_encode_with_config(
                t.distance,
                code.hybrid_uint_configs[dhist],
            );
            put(&mut coder, dhist, sym);
            let (sym, _, _) = lz77_length_encode(t.value);
            put(&mut coder, hist, min_symbol + sym);
        } else {
            let (sym, _, _) =
                crate::entropy::uint_encode_with_config(t.value, code.hybrid_uint_configs[hist]);
            put(&mut coder, hist, sym);
        }
    }
    w.write(32, coder.state() as u64);
    let mut emitted = emitted.into_iter().rev();
    let mut write = |nbits: u32, bits: u32| {
        ordinal -= 1;
        if present[ordinal / 64] & (1u64 << (ordinal % 64)) != 0 {
            let word = emitted.next().expect("one word per presence bit");
            w.write(16, word as u64);
        }
        w.write(nbits as usize, bits as u64);
    };
    for t in lz_tokens(tokens) {
        if t.is_lz77() {
            let (_, nbits, bits) = lz77_length_encode(t.value);
            write(nbits, bits);
            let dhist = code.context_map[distance_context as usize] as usize;
            let (_, nbits, bits) = crate::entropy::uint_encode_with_config(
                t.distance,
                code.hybrid_uint_configs[dhist],
            );
            write(nbits, bits);
        } else {
            let hist = code.context_map[t.context as usize] as usize;
            let (_, nbits, bits) =
                crate::entropy::uint_encode_with_config(t.value, code.hybrid_uint_configs[hist]);
            write(nbits, bits);
        }
    }
    for _ in 0..tail_count {
        write(tail_nbits, tail_bits);
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

/// `write_local_tree_lz77` with a predictor offset per channel.
pub(super) fn write_local_tree_lz77_offsets(
    predictors: &[u32],
    offsets: &[i32],
    pixel_code: &EntropyCode<'_>,
    min_symbol: u32,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut BitWriter,
) {
    let tree_tokens = build_balanced_tree_tokens_offsets(predictors, offsets);
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
                        &code.ans_reverse_maps
                            [start..start + crate::entropy::ANS_TAB_SIZE as usize],
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
                        1 => LzToken::pixel(i % 3, (i * 137 ^ (i >> 3)) & 65_535),
                        _ if i % 3 == 0 => LzToken::lz77(i % 2, i * 53, 1 + i * 97),
                        _ => LzToken::pixel(i % 3, i & 255),
                    })
                    .collect();
                let mut scratch = CoderScratch::default();
                let code = build_lz_pixel_code_opts(
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
                slices.iter().copied(),
                3,
                64,
                refined,
                ans_cluster,
                Some(&pool),
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            let coded = estimate_coded_bits(&slices, 3, &code, 64, threads);
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
                            ((i * 137 ^ (i / 7)) as u32) & CompactToken::MAX_VALUE,
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

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::coder_scratch::CoderScratch;

    #[test]
    fn deep_matcher_never_reaches_past_the_decoder_window() {
        // A repeated 64-token motif separated by more than a window of
        // unique filler: the second copy may not reference the first.
        let motif: Vec<Token> = (0..64).map(|i| Token::new(0, 1000 + i)).collect();
        let mut tokens = motif.clone();
        tokens.extend((0..LZ77_WINDOW as u32 + 100).map(|i| Token::new(0, 2000 + (i % 50_000))));
        tokens.extend(motif.iter().copied());
        let mut scratch = CoderScratch::default();
        let mut out = Vec::new();
        assert!(lz77_compress_with_depth_into_limit(
            &tokens,
            8,
            &mut scratch.lz_depth,
            &mut out,
            usize::MAX
        ));
        for token in &out {
            if token.is_lz77() && token.distance != LZ77_DIST_VALUE {
                let distance = (token.distance - LZ77_NUM_SPECIAL_DISTANCES + 1) as usize;
                assert!(
                    distance <= LZ77_WINDOW,
                    "distance {distance} exceeds the window"
                );
            }
        }
    }
}

#[cfg(test)]
mod constant_tail_tests {
    use super::*;
    use crate::bit_writer::BitWriter;
    use crate::coder_scratch::CoderScratch;

    /// A pseudo-random literal stream over contexts `0..contexts`.
    fn stream(seed: u32, len: usize, contexts: u32) -> Vec<Token> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                Token::new(state % contexts, (state >> 8) % 40)
            })
            .collect()
    }

    /// (refined, ans_cluster): the learned-tree, Slow flat and Fast flat codes.
    const CODE_KINDS: [(bool, bool); 3] = [(true, true), (true, false), (false, false)];

    #[test]
    fn tail_context_is_pinned_to_a_single_symbol_cluster() {
        for (refined, ans_cluster) in CODE_KINDS {
            let tokens = stream(7, 4096, 3);
            let tail = ConstantTail {
                context: 3,
                count: 100_000,
            };
            let mut scratch = CoderScratch::default();
            let code = build_lz_pixel_code_tails(
                &[tokens.as_slice()],
                &[tail],
                4,
                64,
                refined,
                ans_cluster,
                None,
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            let cluster = code.context_map[3] as usize;
            assert!(
                code.context_map
                    .iter()
                    .enumerate()
                    .all(|(context, &c)| context == 3 || c as usize != cluster),
                "the tail shares no cluster ({refined}, {ans_cluster})"
            );
            let (symbol, nbits, _) =
                crate::entropy::uint_encode_with_config(0, code.hybrid_uint_configs[cluster]);
            assert_eq!(nbits, 0);
            if code.use_prefix_code {
                assert_eq!(code.prefix_codes[cluster].depths[symbol as usize], 0);
            } else {
                assert_eq!(
                    code.ans_symbols[cluster][symbol as usize].freq as u32,
                    crate::entropy::ANS_TAB_SIZE
                );
            }
        }
    }

    #[test]
    fn tail_writes_the_same_bits_as_materialized_zeros() {
        for (refined, ans_cluster) in CODE_KINDS {
            for with_runs in [false, true] {
                let literals = stream(11, 3000, 3);
                let tail = ConstantTail {
                    context: 3,
                    count: 5000,
                };
                let tokens: Vec<LzToken> = if with_runs {
                    lz77_compress_runs_channels(vec![literals.clone()])
                } else {
                    lz77_literals(&literals)
                };
                let mut scratch = CoderScratch::default();
                let code = build_lz_pixel_code_tails(
                    &[tokens.as_slice()],
                    &[tail],
                    4,
                    64,
                    refined,
                    ans_cluster,
                    None,
                    &mut scratch.lz_entropy,
                    &mut scratch.huffman_pool,
                );
                let mut with_tail = BitWriter::new();
                write_lz_section_tail(&tokens, tail, 4, &code, 64, &mut with_tail);
                let mut materialized = tokens.clone();
                materialized.extend((0..tail.count).map(|_| LzToken::pixel(3, 0)));
                let mut plain = BitWriter::new();
                write_lz_section(&materialized, 4, &code, 64, &mut plain);
                assert_eq!(with_tail.bits_written(), plain.bits_written());
                with_tail.zero_pad_to_byte();
                plain.zero_pad_to_byte();
                assert_eq!(with_tail.into_bytes(), plain.into_bytes());
            }
        }
    }

    #[test]
    fn no_tail_keeps_the_code_and_section_unchanged() {
        let literals = stream(3, 2000, 3);
        let mut scratch = CoderScratch::default();
        let code = build_lz_pixel_code_tails(
            &[literals.as_slice()],
            &[ConstantTail::NONE],
            3,
            64,
            true,
            true,
            None,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        let mut a = BitWriter::new();
        write_lz_section_tail(&literals, ConstantTail::NONE, 3, &code, 64, &mut a);
        let mut b = BitWriter::new();
        write_lz_section(&literals, 3, &code, 64, &mut b);
        assert_eq!(a.bits_written(), b.bits_written());
        a.zero_pad_to_byte();
        b.zero_pad_to_byte();
        assert_eq!(a.into_bytes(), b.into_bytes());
    }
}
