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
use crate::entropy::{
    Histogram, OwnedEntropyCode, Token, optimize_entropy_code, write_entropy_code, write_token,
};
use std::cell::RefCell;

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
pub(super) enum LzToken {
    Pixel {
        context: u32,
        value: u32,
    },
    Lz77 {
        pixel_context: u32,
        distance_context: u32,
        length_value: u32,
        distance_value: u32,
    },
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

#[inline]
fn hash(tokens: &[Token], pos: usize) -> usize {
    const HASH_BITS: usize = 18;
    fingerprint(tokens, pos) as usize & ((1 << HASH_BITS) - 1)
}

thread_local! {
    static REPETITIONS_SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

fn has_repetition(tokens: &[Token]) -> bool {
    const MAX_SAMPLES: usize = 8_192;
    const SAMPLE_TABLE_SIZE: usize = 1 << 14;
    if tokens.len() < 256 {
        return true;
    }
    REPETITIONS_SCRATCH.with_borrow_mut(|scratch| {
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
        repeats * 5 >= samples
    })
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

fn find_match(
    tokens: &[Token],
    pos: usize,
    head: &[u32],
    prev: &[u32],
    max_probes: usize,
) -> (usize, usize) {
    if pos + LZ77_MIN_LENGTH as usize > tokens.len() {
        return (0, 0);
    }
    let mut candidate = head[hash(tokens, pos)];
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    let mut probes = 0usize;
    while candidate != u32::MAX && probes < max_probes {
        let candidate_pos = candidate as usize;
        let distance = pos - candidate_pos;
        if distance <= u32::MAX as usize {
            let len = match_len(tokens, candidate_pos, pos);
            if len > best_len || (len == best_len && distance < best_dist) {
                best_len = len;
                best_dist = distance;
            }
        }
        candidate = prev[candidate_pos];
        probes += 1;
    }
    (best_len, best_dist)
}

#[cfg(test)]
fn lz77_compress(tokens: &[Token], distance_context: u32) -> Vec<LzToken> {
    lz77_compress_with_depth(tokens, distance_context, 8)
}

#[inline]
pub(super) fn lz77_compress_for_speed(
    tokens: &[Token],
    distance_context: u32,
    speed: crate::Speed,
) -> Vec<LzToken> {
    if speed == crate::Speed::Fast || !has_repetition(tokens) {
        return lz77_compress_runs(tokens, distance_context);
    }
    let deep = lz77_compress_with_depth(tokens, distance_context, 8);
    let run_token_count = run_token_count(tokens);
    if deep.len() * 100 > run_token_count * 90 {
        return lz77_compress_runs(tokens, distance_context);
    }
    let runs = lz77_compress_runs(tokens, distance_context);
    if estimate_payload_bits(&deep, distance_context) * 100
        <= estimate_payload_bits(&runs, distance_context) * 90
    {
        deep
    } else {
        runs
    }
}

fn estimate_payload_bits(tokens: &[LzToken], distance_context: u32) -> u64 {
    let num_contexts = distance_context as usize + 1;
    let context_map: Vec<u8> = (0..num_contexts as u8).collect();
    let histograms = lz_build_histograms(tokens, &context_map, num_contexts, LZ77_MIN_SYMBOL);
    let codes = crate::entropy::build_huffman_codes(&histograms);
    let mut bits = 0u64;
    for &token in tokens {
        match token {
            LzToken::Pixel { context, value } => {
                let (symbol, nbits, _) = crate::entropy::uint_encode(value);
                bits += codes[context as usize].depths[symbol as usize] as u64 + nbits as u64;
            }
            LzToken::Lz77 {
                pixel_context,
                length_value,
                distance_value,
                ..
            } => {
                let (symbol, nbits, _) = lz77_length_encode(length_value);
                bits += codes[pixel_context as usize].depths[(LZ77_MIN_SYMBOL + symbol) as usize]
                    as u64
                    + nbits as u64;
                let (symbol, nbits, _) = crate::entropy::uint_encode(distance_value);
                bits +=
                    codes[distance_context as usize].depths[symbol as usize] as u64 + nbits as u64;
            }
        }
    }
    bits
}

fn run_token_count(tokens: &[Token]) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        count += 1;
        let token = tokens[i];
        let mut end = i + 1;
        while end < tokens.len()
            && tokens[end].context == token.context
            && tokens[end].value == token.value
        {
            end += 1;
        }
        if end - i > LZ77_MIN_LENGTH as usize {
            count += 1;
            i = end;
        } else {
            i += 1;
        }
    }
    count
}

pub(super) fn lz77_compress_runs(tokens: &[Token], distance_context: u32) -> Vec<LzToken> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0usize;
    while i < tokens.len() {
        let token = tokens[i];
        out.push(LzToken::Pixel {
            context: token.context,
            value: token.value,
        });
        let mut end = i + 1;
        while end < tokens.len()
            && tokens[end].context == token.context
            && tokens[end].value == token.value
        {
            end += 1;
        }
        let copied = end - i - 1;
        if copied >= LZ77_MIN_LENGTH as usize {
            out.push(LzToken::Lz77 {
                pixel_context: token.context,
                distance_context,
                length_value: copied as u32 - LZ77_MIN_LENGTH,
                distance_value: LZ77_DIST_VALUE,
            });
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

thread_local! {
    static DEPTH_SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn release_tls_scratch() {
    REPETITIONS_SCRATCH.with_borrow_mut(|scratch| *scratch = Vec::new());
    DEPTH_SCRATCH.with_borrow_mut(|scratch| *scratch = Vec::new());
}

#[cfg(test)]
pub(super) fn tls_scratch_capacity() -> usize {
    REPETITIONS_SCRATCH.with_borrow(|scratch| scratch.capacity())
        + DEPTH_SCRATCH.with_borrow(|scratch| scratch.capacity())
}

fn lz77_compress_with_depth(
    tokens: &[Token],
    distance_context: u32,
    max_probes: usize,
) -> Vec<LzToken> {
    let mut out = Vec::with_capacity(tokens.len());
    DEPTH_SCRATCH.with_borrow_mut(|scratch| {
        const HEAD_LEN: usize = 1 << 18;
        let total = HEAD_LEN + tokens.len();
        if scratch.len() < total {
            scratch.resize(total, u32::MAX);
        }
        scratch[..HEAD_LEN].fill(u32::MAX);
        let (head, tail) = scratch.split_at_mut(HEAD_LEN);
        let (prev, _) = tail.split_at_mut(tokens.len());
        let mut i = 0usize;
        while i < tokens.len() {
            let (match_len, distance) = find_match(tokens, i, head, prev, max_probes);
            let threshold = if distance <= 16 { 4 } else { 5 };
            if match_len >= threshold {
                let token_hash = hash(tokens, i);
                let old_head = head[token_hash];
                prev[i] = old_head;
                head[token_hash] = i as u32;
                let (next_len, _) = if i + 1 < tokens.len() {
                    find_match(tokens, i + 1, head, prev, max_probes)
                } else {
                    (0, 0)
                };
                if next_len > match_len + 1 {
                    let token = tokens[i];
                    out.push(LzToken::Pixel {
                        context: token.context,
                        value: token.value,
                    });
                    i += 1;
                    continue;
                }
                head[token_hash] = old_head;
                prev[i] = u32::MAX;
                let distance_value = if distance == 1 {
                    LZ77_DIST_VALUE
                } else {
                    LZ77_NUM_SPECIAL_DISTANCES + distance as u32 - 1
                };
                out.push(LzToken::Lz77 {
                    pixel_context: tokens[i].context,
                    distance_context,
                    length_value: match_len as u32 - LZ77_MIN_LENGTH,
                    distance_value,
                });
                for pos in i..i + match_len {
                    let token_hash = hash(tokens, pos);
                    prev[pos] = head[token_hash];
                    head[token_hash] = pos as u32;
                }
                i += match_len;
            } else {
                let token = tokens[i];
                out.push(LzToken::Pixel {
                    context: token.context,
                    value: token.value,
                });
                let token_hash = hash(tokens, i);
                prev[i] = head[token_hash];
                head[token_hash] = i as u32;
                i += 1;
            }
        }
    });
    out
}

fn lz_build_histograms(
    toks: &[LzToken],
    context_map: &[u8],
    num_clusters: usize,
    min_symbol: u32,
) -> Vec<Histogram> {
    let mut hs = vec![Histogram::new(); num_clusters];
    for t in toks {
        match *t {
            LzToken::Pixel { context, value } => {
                let (sym, _, _) = crate::entropy::uint_encode(value);
                let cluster = context_map[context as usize] as usize;
                hs[cluster].add(sym);
            }
            LzToken::Lz77 {
                pixel_context,
                distance_context,
                length_value,
                distance_value,
            } => {
                let (len_tok, _, _) = lz77_length_encode(length_value);
                let pixel_cluster = context_map[pixel_context as usize] as usize;
                hs[pixel_cluster].add(min_symbol + len_tok);

                let dist_cluster = context_map[distance_context as usize] as usize;
                let (symbol, _, _) = crate::entropy::uint_encode(distance_value);
                hs[dist_cluster].add(symbol);
            }
        }
    }
    hs
}

/// Build per-cluster prefix codes from an `LzToken` stream.
/// `nb_chans + 1` contexts: `nb_chans` channel leaves + 1 distance context.
pub(super) fn build_lz_pixel_code(
    toks: &[LzToken],
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
) -> OwnedEntropyCode {
    let refined = refined
        && toks.iter().any(|token| {
            matches!(
                token,
                LzToken::Lz77 { distance_value, .. }
                    if *distance_value != LZ77_DIST_VALUE
            )
        });
    use crate::entropy::build_huffman_codes;
    use crate::entropy::cluster_histograms;

    let num_contexts = nb_chans + 1;
    let context_map_initial: Vec<u8> = (0..num_contexts).map(|i| i as u8).collect();
    let mut histograms = lz_build_histograms(toks, &context_map_initial, num_contexts, min_symbol);

    let mut context_map: Vec<u8> = Vec::new();
    if refined {
        crate::entropy::cluster_histograms_refined(&mut histograms, &mut context_map);
    } else {
        cluster_histograms(&mut histograms, &mut context_map);
    }

    let hybrid_uint_configs = if refined {
        let mut raw_values = vec![Vec::<u32>::new(); histograms.len()];
        let mut literal_values = vec![Vec::<u32>::new(); histograms.len()];
        for &tok in toks {
            match tok {
                LzToken::Pixel { context, value } => {
                    let cluster = context_map[context as usize] as usize;
                    raw_values[cluster].push(value);
                    literal_values[cluster].push(value);
                }
                LzToken::Lz77 {
                    distance_context,
                    distance_value,
                    ..
                } => {
                    raw_values[context_map[distance_context as usize] as usize]
                        .push(distance_value);
                }
            }
        }
        let configs: Vec<_> = raw_values
            .iter()
            .enumerate()
            .map(|(cluster, values)| {
                let selected = crate::entropy::select_hybrid_config(values);
                if literal_values[cluster].iter().all(|&value| {
                    crate::entropy::uint_encode_with_config(value, selected).0 < min_symbol
                }) {
                    selected
                } else {
                    crate::entropy::HybridUintConfig::DEFAULT
                }
            })
            .collect();
        histograms = vec![Histogram::new(); configs.len()];
        for &tok in toks {
            match tok {
                LzToken::Pixel { context, value } => {
                    let cluster = context_map[context as usize] as usize;
                    let (symbol, _, _) =
                        crate::entropy::uint_encode_with_config(value, configs[cluster]);
                    histograms[cluster].add(symbol);
                }
                LzToken::Lz77 {
                    pixel_context,
                    distance_context,
                    length_value,
                    distance_value,
                } => {
                    let (len_tok, _, _) = lz77_length_encode(length_value);
                    histograms[context_map[pixel_context as usize] as usize]
                        .add(min_symbol + len_tok);
                    let cluster = context_map[distance_context as usize] as usize;
                    let (symbol, _, _) =
                        crate::entropy::uint_encode_with_config(distance_value, configs[cluster]);
                    histograms[cluster].add(symbol);
                }
            }
        }
        configs
    } else {
        vec![crate::entropy::HybridUintConfig::DEFAULT; histograms.len()]
    };

    let mut code = OwnedEntropyCode {
        context_map,
        prefix_codes: build_huffman_codes(&histograms),
        hybrid_uint_configs,
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: true,
        ans_freqs: Vec::new(),
        ans_symbols: Vec::new(),
    };

    // Apply the single-symbol patch (mirrors build_pixel_code) per cluster so
    // that contexts with one unique symbol still emit a parseable code.
    for pc in &mut code.prefix_codes {
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
    code
}

/// Emit one `LzToken` into the bitstream.
#[inline]
pub(super) fn write_lz_token(
    t: LzToken,
    code: &OwnedEntropyCode,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    match t {
        LzToken::Pixel { context, value } => {
            let cluster = code.context_map[context as usize] as usize;
            let (sym, nbits, bits) =
                crate::entropy::uint_encode_with_config(value, code.hybrid_uint_configs[cluster]);
            let pc = &code.prefix_codes[cluster];
            let d = pc.depths[sym as usize] as usize;
            let data = (pc.bits[sym as usize] as u64) | ((bits as u64) << d);
            w.write(d + nbits as usize, data);
        }
        LzToken::Lz77 {
            pixel_context,
            distance_context,
            length_value,
            distance_value,
        } => {
            let (len_tok, len_nbits, len_bits) = lz77_length_encode(length_value);
            let sym = min_symbol + len_tok;
            let pcluster = code.context_map[pixel_context as usize] as usize;
            let pc = &code.prefix_codes[pcluster];
            let d = pc.depths[sym as usize] as usize;
            debug_assert!(
                d > 0,
                "LZ77 length symbol {} unrepresented in histogram",
                sym
            );
            let data = (pc.bits[sym as usize] as u64) | ((len_bits as u64) << d);
            w.write(d + len_nbits as usize, data);

            // Distance symbol: value LZ77_DIST_VALUE = 0, no extra bits.
            let dcluster = code.context_map[distance_context as usize] as usize;
            let dc = &code.prefix_codes[dcluster];
            let (dist_symbol, dist_nbits, dist_bits) = crate::entropy::uint_encode_with_config(
                distance_value,
                code.hybrid_uint_configs[dcluster],
            );
            let dd = dc.depths[dist_symbol as usize] as usize;
            // (Could be 0 if it's the only symbol in a single-symbol histogram.)
            if dd > 0 {
                let data = dc.bits[dist_symbol as usize] as u64 | ((dist_bits as u64) << dd);
                w.write(dd + dist_nbits as usize, data);
            } else if dist_nbits != 0 {
                w.write(dist_nbits as usize, dist_bits as u64);
            }
        }
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
    pixel_code: &OwnedEntropyCode,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    let tree_tokens = build_balanced_tree_tokens(predictors);
    write_tree_lz77(&tree_tokens, pixel_code, min_symbol, w);
}

/// Write a pre-built MA tree (token stream) + the LZ77 pixel code header.
pub(super) fn write_tree_lz77(
    tree_tokens: &[Token],
    pixel_code: &OwnedEntropyCode,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    let tree_code = optimize_entropy_code(tree_tokens, NUM_TREE_CONTEXTS);
    let tree_code_ref = tree_code.as_ref();

    // Tree's entropy code: no LZ77 in the tree itself.
    w.write(1, 0);
    write_entropy_code(&tree_code_ref, w);
    for tok in tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }

    // Pixel entropy code: LZ77 ENABLED for the main bitstream.
    write_lz77_header(min_symbol, w);
    // The decoder appends an extra context (distance) when LZ77 is on, so the
    // context map we write must already include it as its last entry.
    write_entropy_code(&pixel_code.as_ref(), w);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(stream: &[LzToken]) -> Vec<Token> {
        let mut out = Vec::new();
        for &token in stream {
            match token {
                LzToken::Pixel { context, value } => out.push(Token::new(context, value)),
                LzToken::Lz77 {
                    length_value,
                    distance_value,
                    ..
                } => {
                    let distance = if distance_value == LZ77_DIST_VALUE {
                        1
                    } else {
                        (distance_value - LZ77_NUM_SPECIAL_DISTANCES + 1) as usize
                    };
                    for _ in 0..length_value + LZ77_MIN_LENGTH {
                        let source = out.len() - distance;
                        out.push(out[source]);
                    }
                }
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
        let compressed = lz77_compress(&input, 3);
        assert!(compressed.iter().any(|token| matches!(
            token,
            LzToken::Lz77 { distance_value, .. }
                if *distance_value >= LZ77_NUM_SPECIAL_DISTANCES
        )));
        assert!(same_tokens(&expand(&compressed), &input));
    }

    #[test]
    fn hash_chain_uses_compact_distance_for_runs() {
        let input = vec![Token::new(0, 7); 128];
        let compressed = lz77_compress(&input, 1);
        assert!(compressed.iter().any(|token| matches!(
            token,
            LzToken::Lz77 {
                distance_value: LZ77_DIST_VALUE,
                ..
            }
        )));
        assert!(same_tokens(&expand(&compressed), &input));
    }

    #[test]
    fn speed_policy_keeps_fast_run_only_and_slow_structured_search() {
        let pattern: Vec<Token> = (0..64)
            .map(|i| Token::new((i % 3) as u32, ((i * 37 + 11) % 257) as u32))
            .collect();
        let input: Vec<Token> = pattern.iter().copied().cycle().take(512).collect();
        let fast = lz77_compress_for_speed(&input, 3, crate::Speed::Fast);
        assert!(!fast.iter().any(|token| matches!(
            token,
            LzToken::Lz77 { distance_value, .. }
                if *distance_value != LZ77_DIST_VALUE
        )));
        let slow = lz77_compress_for_speed(&input, 3, crate::Speed::Slow);
        assert!(slow.iter().any(|token| matches!(
            token,
            LzToken::Lz77 { distance_value, .. }
                if *distance_value != LZ77_DIST_VALUE
        )));
        assert!(same_tokens(&expand(&slow), &input));
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
            move || lz77_compress_with_depth(&small, 9, 8).len()
        })
        .join()
        .unwrap();
        let _ = lz77_compress_with_depth(&big, 9, 8);
        assert_eq!(clean, lz77_compress_with_depth(&small, 9, 8).len());
    }

    #[test]
    fn deep_search_handles_varied_stream_lengths() {
        for len in [1_000usize, 50_000, 3_000] {
            let _ = lz77_compress_with_depth(&tokens(len, 5), 9, 8);
        }
    }
}
