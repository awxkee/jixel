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

//! Lossless LZ77 tokens, run collapse, and ring matchers.

#[cfg(test)]
mod constant_tail_tests;
mod entropy;
mod priced;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod window_tests;
mod write;

pub(super) use entropy::{
    build_lz_pixel_code, build_lz_pixel_code_opts, build_lz_pixel_code_tails, estimate_coded_bits,
    estimate_literal_and_run_bits, estimate_literal_and_run_bits_single, estimate_streams_bits,
    lz77_literals,
};
pub(super) use priced::{
    LiteralPrices, MatchPrices, PRICED_MAX_PROBES, lz77_compress_optimal, lz77_compress_priced,
};
pub(super) use write::{
    write_local_tree_lz77, write_local_tree_lz77_offsets, write_lz_section, write_lz_section_tail,
    write_tree_lz77,
};

use crate::coder_scratch::{CoderScratch, LzEntropyScratch};
use crate::entropy::{CompactToken, Token};
use entropy::estimate_payload_bits;

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
    fingerprint_gram::<T, 3>(tokens, pos)
}

#[inline]
fn fingerprint_gram<T: LiteralToken, const GRAM: usize>(tokens: &[T], pos: usize) -> u32 {
    let mut hash = 0x9e37_79b9u32;
    for token in &tokens[pos..tokens.len().min(pos + GRAM)] {
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
    find_match_ring_hashed(
        tokens,
        pos,
        ring_hash(tokens, pos),
        entries,
        cursors,
        max_probes,
        epoch,
        match_kernel,
    )
}

#[allow(clippy::too_many_arguments)]
fn find_match_ring_hashed<T: LiteralToken>(
    tokens: &[T],
    pos: usize,
    h: usize,
    entries: &[u32],
    cursors: &mut [u32],
    max_probes: usize,
    epoch: u32,
    match_kernel: crate::lz_match::MatchKernel<T>,
) -> (usize, usize) {
    if pos + LZ77_MIN_LENGTH as usize > tokens.len() {
        return (0, 0);
    }
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
    lz77_compress_for_speed_with_parts::<_, 3>(
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
    lz77_compress_for_speed_with_parts::<_, 3>(
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
fn lz77_compress_for_speed_with_parts<T: LiteralToken, const GRAM: usize>(
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
    if !speed.effort().lz_deep || !has_repetition(tokens, lz_repetitions) {
        return lz77_compress_runs_with_capacity(tokens, run_capacity);
    }
    let run_tokens = lz77_compress_runs_with_capacity(tokens, run_capacity);
    // Slow mode's broader candidate selection protects the major Modular
    // alternatives, so do not discard useful 2..10% LZ wins at this local gate.
    let max_candidate_len = run_tokens.len().saturating_mul(98) / 100;
    if !lz77_compress_with_depth_into_limit_gram::<_, GRAM>(
        tokens,
        8,
        lz_depth,
        lz_candidate,
        max_candidate_len,
    ) {
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
    if !speed.effort().lz_deep {
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
    if !speed.effort().lz_deep {
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
#[cfg(test)]
fn lz77_compress_with_depth_into_limit<T: LiteralToken>(
    tokens: &[T],
    max_probes: usize,
    scratch: &mut Vec<u32>,
    out: &mut Vec<LzToken>,
    max_output: usize,
) -> bool {
    lz77_compress_with_depth_into_limit_gram::<T, 3>(tokens, max_probes, scratch, out, max_output)
}

#[inline]
fn ring_hash_gram<T: LiteralToken, const GRAM: usize>(tokens: &[T], pos: usize) -> usize {
    fingerprint_gram::<T, GRAM>(tokens, pos) as usize & (RING_HASH_SIZE - 1)
}

fn lz77_compress_with_depth_into_limit_gram<T: LiteralToken, const GRAM: usize>(
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
        let (match_len, distance) = if GRAM == 3 {
            find_match_ring(tokens, i, entries, cursors, max_probes, epoch, match_kernel)
        } else {
            find_match_ring_hashed(
                tokens,
                i,
                ring_hash_gram::<T, GRAM>(tokens, i),
                entries,
                cursors,
                max_probes,
                epoch,
                match_kernel,
            )
        };
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
            let mut insert_start = i;
            if GRAM > 3 {
                // All grams inside a constant prefix share one bucket. Only
                // its newest RING_BUCKET entries can survive insertion, so
                // skip those that would immediately be overwritten. Preserve
                // every gram crossing the end of the constant prefix.
                let constant =
                    1 + match_kernel(&tokens[i..i + match_len - 1], &tokens[i + 1..i + match_len]);
                insert_start += constant.saturating_sub(GRAM + RING_BUCKET - 1);
            }
            for pos in insert_start..i + match_len {
                ring_insert(
                    entries,
                    cursors,
                    ring_hash_gram::<T, GRAM>(tokens, pos),
                    pos,
                    epoch,
                );
            }
            i += match_len;
        } else {
            let token = tokens[i].as_token();
            out.push(LzToken::pixel(token.context, token.value));
            if out.len() > max_output {
                return false;
            }
            ring_insert(
                entries,
                cursors,
                ring_hash_gram::<T, GRAM>(tokens, i),
                i,
                epoch,
            );
            i += 1;
        }
    }
    true
}
