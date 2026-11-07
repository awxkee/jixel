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

//! LZ77 section and tree bitstream serialization.

use super::super::tree::{
    NUM_TREE_CONTEXTS, build_balanced_tree_tokens, build_balanced_tree_tokens_offsets,
};
use super::{ConstantTail, LzToken, LzTokenSource, lz_tokens, lz77_length_encode};
use crate::bit_writer::BitWriter;
use crate::entropy::{EntropyCode, Token, write_entropy_code};

/// Emit a full section's `LzToken` stream: prefix codes stream token-by-token;
/// rANS buffers the section (one ANS state per section, symbols pushed in
/// reverse) exactly like `write_ans_tokens`, with the LZ77 symbol mapping of
/// `write_lz_token` (length = direct symbol `min_symbol + len_tok`; literals
/// and distances honor the per-cluster hybrid-uint config).
pub(crate) fn write_lz_section<T: LzTokenSource>(
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
pub(in crate::lossless) fn write_lz_section_tail<T: LzTokenSource>(
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
pub(in crate::lossless) fn write_lz_token(
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
pub(in crate::lossless) fn write_local_tree_lz77(
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
pub(in crate::lossless) fn write_local_tree_lz77_offsets(
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
pub(crate) fn write_tree_lz77(
    tree_tokens: &[Token],
    pixel_code: &EntropyCode<'_>,
    min_symbol: u32,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut BitWriter,
) {
    // Tree's entropy code (no LZ77 in the tree itself), then its tokens.
    crate::entropy::write_raced_token_stream(tree_tokens, NUM_TREE_CONTEXTS, huffman_pool, w);

    // Pixel entropy code: LZ77 ENABLED for the main bitstream.
    write_lz77_header(min_symbol, w);
    // The decoder appends an extra context (distance) when LZ77 is on, so the
    // context map we write must already include it as its last entry.
    write_entropy_code(pixel_code, huffman_pool, w);
}
