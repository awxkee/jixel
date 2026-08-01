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

use super::ans::{
    AnsEncSymbolInfo, build_symbol_info, choose_use_prefix_code, encode_histogram, normalize_counts,
};
use super::cluster::cluster_histograms;
use super::entropy_code::{EntropyCode, OwnedEntropyCode};
use super::histogram::Histogram;
use super::huffman_tree::{HuffmanNode, create_huffman_tree};
use super::prefix_code::{ALPHABET_SIZE, PrefixCode, convert_bit_depths_to_symbols};
use super::token::{HybridUintConfig, Token, uint_encode, uint_encode_with_config};
use crate::bit_writer::BitWriter;
use crate::enc_lz77_ac::{LZ77_MIN_LENGTH, LZ77_MIN_SYMBOL, lz77_length_encode};
use crate::entropy::f_log2;

pub(crate) const ANS_ENABLED: bool = true;

#[allow(unused)]
pub(crate) const ANS_LOG_ALPHA_SIZE: u32 = 7;

/// Default (prefix-only) ANS fields for a freshly built code.
fn no_ans() -> (bool, Vec<Vec<u16>>, Vec<Vec<AnsEncSymbolInfo>>) {
    (true, Vec::new(), Vec::new())
}

/// Hot-path token writer. Matches libjxl-tiny's inline WriteToken.
#[inline]
pub(crate) fn write_token(t: Token, code: &EntropyCode, w: &mut BitWriter) {
    let cluster = code.context_map[t.context as usize] as usize;
    let (tok, nbits, bits) = uint_encode_with_config(t.value, code.hybrid_uint_configs[cluster]);
    let pc = &code.prefix_codes[cluster];
    if pc.single_symbol {
        // Single-symbol prefix code: the codeword is zero-length (JXL encodes
        // such a context with no bits, and the decoder reads the symbol without
        // consuming any). Emit only the extra-bits payload.
        w.write(nbits as usize, bits as u64);
        return;
    }
    let d = pc.depths[tok as usize] as usize;
    let data = (pc.bits[tok as usize] as u64) | ((bits as u64) << d);
    w.write(d + nbits as usize, data);
}

const HYBRID_CANDIDATES: [HybridUintConfig; 12] = [
    HybridUintConfig {
        split_exponent: 0,
        msb_in_token: 0,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 1,
        msb_in_token: 0,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 2,
        msb_in_token: 0,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 3,
        msb_in_token: 0,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 4,
        msb_in_token: 0,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 4,
        msb_in_token: 1,
        lsb_in_token: 0,
    },
    HybridUintConfig::DEFAULT,
    HybridUintConfig {
        split_exponent: 4,
        msb_in_token: 1,
        lsb_in_token: 1,
    },
    HybridUintConfig {
        split_exponent: 5,
        msb_in_token: 0,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 5,
        msb_in_token: 1,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 5,
        msb_in_token: 2,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 6,
        msb_in_token: 1,
        lsb_in_token: 0,
    },
];

pub(crate) fn select_hybrid_config(
    values: &[u32],
    huffman_pool: &mut Vec<HuffmanNode>,
) -> HybridUintConfig {
    const MAX_SEARCH_SAMPLES: usize = 65_536;
    let stride = values.len().div_ceil(MAX_SEARCH_SAMPLES).max(1);
    let mut best = HybridUintConfig::DEFAULT;
    let mut best_cost = u64::MAX;
    for config in HYBRID_CANDIDATES {
        let mut histogram = Histogram::new();
        let mut extra = 0u64;
        let mut valid = true;
        for &value in values.iter().step_by(stride) {
            let (symbol, nbits, _) = uint_encode_with_config(value, config);
            if symbol as usize >= ALPHABET_SIZE {
                valid = false;
                break;
            }
            histogram.add(symbol);
            extra += nbits as u64;
        }
        if !valid {
            continue;
        }
        let used = histogram.counts.iter().filter(|&&count| count != 0).count();
        let prefix = if used <= 1 {
            0
        } else {
            let mut depths = [0u8; ALPHABET_SIZE];
            create_huffman_tree(&histogram.counts, 15, &mut depths, huffman_pool);
            histogram
                .counts
                .iter()
                .zip(depths.iter())
                .map(|(&count, &depth)| count as u64 * depth as u64)
                .sum::<u64>()
        };
        // UintConfig storage is variable-width after split_exponent, so small
        // histograms should not pick a theoretically cheaper configuration
        // whose header costs more than it saves.
        let split = config.split_exponent as u32;
        let msb = config.msb_in_token as u32;
        let msb_width = if split == 0 {
            0
        } else {
            32 - split.leading_zeros()
        };
        let remaining = split - msb;
        let lsb_width = if remaining == 0 {
            0
        } else {
            32 - remaining.leading_zeros()
        };
        let cost = prefix + extra + 4 + msb_width as u64 + lsb_width as u64;
        if cost < best_cost {
            best_cost = cost;
            best = config;
        }
    }
    best
}

fn select_hybrid_config_ans(values: &[u32]) -> HybridUintConfig {
    if values.is_empty() {
        return HybridUintConfig::DEFAULT;
    }
    let stride = values.len().div_ceil(65_536).max(1);
    let mut best = HybridUintConfig::DEFAULT;
    let mut best_cost = f64::INFINITY;
    let mut default_cost = f64::INFINITY;
    for config in HYBRID_CANDIDATES {
        let mut counts = [0u32; ALPHABET_SIZE];
        let mut extra = 0u64;
        let mut total = 0u64;
        let mut valid = true;
        for &value in values.iter().step_by(stride) {
            let (sym, nbits, _) = uint_encode_with_config(value, config);
            if sym as usize >= ALPHABET_SIZE {
                valid = false;
                break;
            }
            counts[sym as usize] += 1;
            extra += nbits as u64;
            total += 1;
        }
        if !valid || total == 0 {
            continue;
        }
        let mut data = extra as f64;
        let mut used = 0u32;
        for &c in &counts {
            if c > 0 {
                used += 1;
                data += c as f64 * f_log2(total as f64 / c as f64);
            }
        }
        let cost = data * stride as f64 + f64::from(8 * used + 15);
        if config == HybridUintConfig::DEFAULT {
            default_cost = cost;
        }
        if cost < best_cost {
            best_cost = cost;
            best = config;
        }
    }
    if best_cost >= default_cost * 0.995 {
        HybridUintConfig::DEFAULT
    } else {
        best
    }
}

fn build_histograms(tokens: &[Token], context_map: Option<&[u8]>, histograms: &mut [Histogram]) {
    build_histograms_with(tokens, context_map, HybridUintConfig::DEFAULT, histograms);
}

fn build_histograms_with(
    tokens: &[Token],
    context_map: Option<&[u8]>,
    config: HybridUintConfig,
    histograms: &mut [Histogram],
) {
    for t in tokens {
        let (tok, _, _) = uint_encode_with_config(t.value, config);
        let context = match context_map {
            Some(m) => m[t.context as usize] as usize,
            None => t.context as usize,
        };
        histograms[context].add(tok);
    }
}

pub(crate) fn build_huffman_codes(
    histograms: &[Histogram],
    huffman_pool: &mut Vec<HuffmanNode>,
) -> Vec<PrefixCode> {
    let mut out = Vec::with_capacity(histograms.len());
    for histogram in histograms {
        out.push(build_huffman_code(histogram, huffman_pool));
    }
    out
}

pub(crate) fn build_huffman_codes_into(
    histograms: &[Histogram],
    out: &mut [PrefixCode],
    huffman_pool: &mut Vec<HuffmanNode>,
) {
    assert!(out.len() >= histograms.len());
    for (histogram, output) in histograms.iter().zip(out.iter_mut()) {
        *output = build_huffman_code(histogram, huffman_pool);
    }
}

fn build_huffman_code(histogram: &Histogram, huffman_pool: &mut Vec<HuffmanNode>) -> PrefixCode {
    let counts: [u32; ALPHABET_SIZE] = histogram.counts;
    let mut length = ALPHABET_SIZE;
    while length > 0 && counts[length - 1] == 0 {
        length -= 1;
    }
    let mut depths = [0u8; ALPHABET_SIZE];
    if length > 0 {
        create_huffman_tree(&counts[..length], 15, &mut depths[..length], huffman_pool);
    }
    let mut bits = [0u16; ALPHABET_SIZE];
    convert_bit_depths_to_symbols(&depths, &mut bits);
    let mut prefix = PrefixCode {
        depths,
        bits,
        single_symbol: false,
    };
    prefix.update_single_symbol();
    prefix
}

/// Build a prefix-codes-only EntropyCode given a fixed context_map and the
/// number of contexts (= number of prefix codes). Used when the context map
/// is known up front (e.g. the static AC code).
pub(crate) fn optimize_prefix_codes(
    tokens: &[Token],
    context_map: Vec<u8>,
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, Some(&context_map), &mut histograms);
    let prefix_codes = build_huffman_codes(&histograms, huffman_pool);
    let (use_prefix_code, ans_freqs, ans_symbols) = no_ans();
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; num_contexts],
        orig_context_map: None,
        orig_num_contexts: 0,
        use_prefix_code,
        ans_freqs,
        ans_symbols,
    }
}

pub(crate) fn optimize_entropy_code(
    tokens: &[Token],
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, None, &mut histograms);
    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map, huffman_pool);
    let prefix_codes = build_huffman_codes(&histograms, huffman_pool);
    let (use_prefix_code, ans_freqs, ans_symbols) = no_ans();
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; histograms.len()],
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code,
        ans_freqs,
        ans_symbols,
    }
}

/// AC-plain entropy code: identical to optimize_entropy_code, but when
/// ANS_ENABLED it may select rANS for the (clustered) histograms. Used ONLY for
/// the plain AC token bundle, whose header (write_ac_global) and token site
/// (enc_frame) both branch on use_prefix_code. No other bundle calls this, so
/// the gate cannot desynchronize a header from its token stream elsewhere.
pub(crate) fn optimize_entropy_code_ac(
    tokens: &[Token],
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> OwnedEntropyCode {
    optimize_entropy_code_ac_streams(std::iter::once(tokens), num_contexts, huffman_pool, true)
}

/// `select_configs = false` keeps every cluster on `HybridUintConfig::DEFAULT`.
/// The provisional price build MUST pass false: config-aware prices nudge RDOQ
/// by single tokens, which can push the (prefix-model) clustering off a
/// knife-edge merge worth hundreds of bytes (kodim20 d=7.5: +9.9%). With
/// default prices the coefficient stream is identical to the pre-selection
/// encoder, and selection then only recodes the final streams on the same
/// clusters — which cannot lose.
pub(crate) fn optimize_entropy_code_ac_streams<'a, I>(
    streams: I,
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
    select_configs: bool,
) -> OwnedEntropyCode
where
    I: IntoIterator<Item = &'a [Token]>,
{
    // Collected so the tokens can be walked twice: once to cluster under the
    // default config, once to rebuild histograms under the selected per-cluster
    // configs. Only the slice headers are copied.
    let streams: Vec<&[Token]> = streams.into_iter().collect();

    let mut histograms = vec![Histogram::new(); num_contexts];
    for tokens in &streams {
        build_histograms(tokens, None, &mut histograms);
    }
    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map, huffman_pool);

    // Second walk: pick each final cluster's HybridUint config from its actual
    // token values (DEFAULT is among the candidates, so this can only move
    // where the selector's cost model says it pays), then rebuild the symbol
    // histograms under the selected configs so the prefix/rANS tables match
    // what the writer will emit.
    let num_clusters = histograms.len();
    let mut cluster_values: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
    for tokens in &streams {
        for t in *tokens {
            cluster_values[context_map[t.context as usize] as usize].push(t.value);
        }
    }
    let hybrid_uint_configs: Vec<HybridUintConfig> = if select_configs {
        cluster_values
            .iter()
            .map(|values| select_hybrid_config_ans(values))
            .collect()
    } else {
        vec![HybridUintConfig::DEFAULT; num_clusters]
    };
    if hybrid_uint_configs
        .iter()
        .any(|&c| c != HybridUintConfig::DEFAULT)
    {
        for h in &mut histograms {
            *h = Histogram::new();
        }
        for (values, (config, histogram)) in cluster_values
            .iter()
            .zip(hybrid_uint_configs.iter().zip(histograms.iter_mut()))
        {
            for &value in values {
                let (tok, _, _) = uint_encode_with_config(value, *config);
                histogram.add(tok);
            }
        }
    }

    let prefix_codes = build_huffman_codes(&histograms, huffman_pool);

    let (mut use_prefix_code, mut ans_freqs, mut ans_symbols) = no_ans();
    if ANS_ENABLED {
        let depths: Vec<[u8; ALPHABET_SIZE]> = prefix_codes.iter().map(|c| c.depths).collect();
        use_prefix_code = choose_use_prefix_code(&histograms, &depths);
        if !use_prefix_code {
            let mut freqs = vec![0u16; 0];
            for h in &histograms {
                normalize_counts(&h.counts, &mut freqs);
                ans_symbols.push(build_symbol_info(&freqs));
                ans_freqs.push(freqs.clone());
            }
        }
    }

    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs,
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code,
        ans_freqs,
        ans_symbols,
    }
}

pub(crate) fn build_entropy_code_no_cluster(
    tokens: &[Token],
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, None, &mut histograms);
    let context_map: Vec<u8> = (0..num_contexts as u8).collect();
    let prefix_codes = build_huffman_codes(&histograms, huffman_pool);
    let (use_prefix_code, ans_freqs, ans_symbols) = no_ans();
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; num_contexts],
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code,
        ans_freqs,
        ans_symbols,
    }
}

// ---------------------------------------------------------------------------
// Brotli-style Huffman-tree serialization.
// ---------------------------------------------------------------------------

const NUM_CODE_LENGTH_CODES: usize = 18;

const STORAGE_ORDER: [u8; NUM_CODE_LENGTH_CODES] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

const HUFFMAN_BIT_LENGTH_HUFFMAN_CODE_SYMBOLS: [u8; 6] = [0, 7, 3, 2, 1, 15];
const HUFFMAN_BIT_LENGTH_HUFFMAN_CODE_BITLENS: [u8; 6] = [2, 4, 3, 2, 2, 4];

fn store_huffman_tree_of_huffman_tree_to_bitmask(
    num_codes: i32,
    code_length_bitdepth: &[u8; NUM_CODE_LENGTH_CODES],
    w: &mut BitWriter,
) {
    let mut codes_to_store = NUM_CODE_LENGTH_CODES;
    if num_codes > 1 {
        while codes_to_store > 0 {
            if code_length_bitdepth[STORAGE_ORDER[codes_to_store - 1] as usize] != 0 {
                break;
            }
            codes_to_store -= 1;
        }
    }
    let mut skip_some = 0usize;
    if code_length_bitdepth[STORAGE_ORDER[0] as usize] == 0
        && code_length_bitdepth[STORAGE_ORDER[1] as usize] == 0
    {
        skip_some = 2;
        if code_length_bitdepth[STORAGE_ORDER[2] as usize] == 0 {
            skip_some = 3;
        }
    }
    w.write(2, skip_some as u64);
    for i in skip_some..codes_to_store {
        let l = code_length_bitdepth[STORAGE_ORDER[i] as usize] as usize;
        w.write(
            HUFFMAN_BIT_LENGTH_HUFFMAN_CODE_BITLENS[l] as usize,
            HUFFMAN_BIT_LENGTH_HUFFMAN_CODE_SYMBOLS[l] as u64,
        );
    }
}

fn store_huffman_tree_to_bitmask(
    huffman_tree: &[u8],
    huffman_tree_extra: &[u8],
    code_length_bitdepth: &[u8; NUM_CODE_LENGTH_CODES],
    code_length_bitdepth_symbols: &[u16; NUM_CODE_LENGTH_CODES],
    w: &mut BitWriter,
) {
    for i in 0..huffman_tree.len() {
        let ix = huffman_tree[i] as usize;
        w.write(
            code_length_bitdepth[ix] as usize,
            code_length_bitdepth_symbols[ix] as u64,
        );
        match ix {
            16 => w.write(2, huffman_tree_extra[i] as u64),
            17 => w.write(3, huffman_tree_extra[i] as u64),
            _ => {}
        }
    }
}

fn store_simple_huffman_tree(
    depths: &[u8],
    symbols: &mut [usize; 4],
    num_symbols: usize,
    max_bits: usize,
    w: &mut BitWriter,
) {
    w.write(2, 1);
    w.write(2, (num_symbols - 1) as u64);
    for i in 0..num_symbols {
        for j in (i + 1)..num_symbols {
            if depths[symbols[j]] < depths[symbols[i]] {
                symbols.swap(i, j);
            }
        }
    }
    match num_symbols {
        2 => {
            w.write(max_bits, symbols[0] as u64);
            w.write(max_bits, symbols[1] as u64);
        }
        3 => {
            w.write(max_bits, symbols[0] as u64);
            w.write(max_bits, symbols[1] as u64);
            w.write(max_bits, symbols[2] as u64);
        }
        4 => {
            w.write(max_bits, symbols[0] as u64);
            w.write(max_bits, symbols[1] as u64);
            w.write(max_bits, symbols[2] as u64);
            w.write(max_bits, symbols[3] as u64);
            w.write(1, if depths[symbols[0]] == 1 { 1 } else { 0 });
        }
        _ => unreachable!(),
    }
}

fn reverse(v: &mut [u8], start: usize, end: usize) {
    if end == 0 {
        return;
    }
    let mut s = start;
    let mut e = end - 1;
    while s < e {
        v.swap(s, e);
        s += 1;
        e -= 1;
    }
}

fn write_huffman_tree_repetitions(
    previous_value: u8,
    value: u8,
    mut repetitions: usize,
    tree_size: &mut usize,
    tree: &mut Vec<u8>,
    extra: &mut Vec<u8>,
) {
    debug_assert!(repetitions > 0);
    if previous_value != value {
        tree.push(value);
        extra.push(0);
        *tree_size += 1;
        repetitions -= 1;
    }
    if repetitions == 7 {
        tree.push(value);
        extra.push(0);
        *tree_size += 1;
        repetitions -= 1;
    }
    if repetitions < 3 {
        for _ in 0..repetitions {
            tree.push(value);
            extra.push(0);
            *tree_size += 1;
        }
    } else {
        repetitions -= 3;
        let start = *tree_size;
        loop {
            tree.push(16);
            extra.push((repetitions & 0x3) as u8);
            *tree_size += 1;
            repetitions >>= 2;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        let end = *tree_size;
        reverse(tree, start, end);
        reverse(extra, start, end);
    }
}

fn write_huffman_tree_repetitions_zeros(
    mut repetitions: usize,
    tree_size: &mut usize,
    tree: &mut Vec<u8>,
    extra: &mut Vec<u8>,
) {
    if repetitions == 11 {
        tree.push(0);
        extra.push(0);
        *tree_size += 1;
        repetitions -= 1;
    }
    if repetitions < 3 {
        for _ in 0..repetitions {
            tree.push(0);
            extra.push(0);
            *tree_size += 1;
        }
    } else {
        repetitions -= 3;
        let start = *tree_size;
        loop {
            tree.push(17);
            extra.push((repetitions & 0x7) as u8);
            *tree_size += 1;
            repetitions >>= 3;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        let end = *tree_size;
        reverse(tree, start, end);
        reverse(extra, start, end);
    }
}

fn decide_over_rle_use(depth: &[u8]) -> (bool, bool) {
    let length = depth.len();
    let mut total_reps_zero = 0usize;
    let mut total_reps_nz = 0usize;
    let mut count_reps_zero = 1usize;
    let mut count_reps_nz = 1usize;
    let mut i = 0;
    while i < length {
        let value = depth[i];
        let mut reps = 1;
        let mut k = i + 1;
        while k < length && depth[k] == value {
            reps += 1;
            k += 1;
        }
        if reps >= 3 && value == 0 {
            total_reps_zero += reps;
            count_reps_zero += 1;
        }
        if reps >= 4 && value != 0 {
            total_reps_nz += reps;
            count_reps_nz += 1;
        }
        i += reps;
    }
    (
        total_reps_nz > count_reps_nz * 2,
        total_reps_zero > count_reps_zero * 2,
    )
}

fn write_huffman_tree(depth: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut new_length = depth.len();
    for i in 0..depth.len() {
        if depth[depth.len() - i - 1] == 0 {
            new_length -= 1;
        } else {
            break;
        }
    }
    let mut tree: Vec<u8> = Vec::new();
    let mut extra: Vec<u8> = Vec::new();
    let mut tree_size = 0usize;
    let (use_rle_nz, use_rle_zero) = if depth.len() > 50 {
        let (_nz, zero) = decide_over_rle_use(&depth[..new_length]);
        // NOTE: the non-zero run-length path (code 16) is disabled pending a
        // round-trip fix; it desynchronizes djxl on dense float histograms even
        // though the emitter matches libjxl. The zero path (code 17) is proven.
        (false, zero)
    } else {
        (false, false)
    };
    let mut previous_value: u8 = 8;
    let mut i = 0;
    while i < new_length {
        let value = depth[i];
        let mut reps = 1;
        if (value != 0 && use_rle_nz) || (value == 0 && use_rle_zero) {
            let mut k = i + 1;
            while k < new_length && depth[k] == value {
                reps += 1;
                k += 1;
            }
        }
        if value == 0 {
            write_huffman_tree_repetitions_zeros(reps, &mut tree_size, &mut tree, &mut extra);
        } else {
            write_huffman_tree_repetitions(
                previous_value,
                value,
                reps,
                &mut tree_size,
                &mut tree,
                &mut extra,
            );
            previous_value = value;
        }
        i += reps;
    }
    (tree, extra)
}

fn store_huffman_tree(depths: &[u8], huffman_pool: &mut Vec<HuffmanNode>, w: &mut BitWriter) {
    let (huffman_tree, huffman_tree_extra) = write_huffman_tree(depths);
    let mut histo = [0u32; NUM_CODE_LENGTH_CODES];
    for &t in &huffman_tree {
        histo[t as usize] += 1;
    }

    let mut num_codes = 0;
    let mut code_one: i32 = 0;
    for (i, &hist) in histo.iter().enumerate() {
        if hist != 0 {
            if num_codes == 0 {
                code_one = i as i32;
                num_codes = 1;
            } else if num_codes == 1 {
                num_codes = 2;
                break;
            }
        }
    }

    let mut code_length_bitdepth = [0u8; NUM_CODE_LENGTH_CODES];
    let mut code_length_bitdepth_symbols = [0u16; NUM_CODE_LENGTH_CODES];
    create_huffman_tree(&histo, 5, &mut code_length_bitdepth, huffman_pool);
    convert_bit_depths_to_symbols(&code_length_bitdepth, &mut code_length_bitdepth_symbols);

    store_huffman_tree_of_huffman_tree_to_bitmask(num_codes, &code_length_bitdepth, w);

    if num_codes == 1 {
        code_length_bitdepth[code_one as usize] = 0;
    }
    store_huffman_tree_to_bitmask(
        &huffman_tree,
        &huffman_tree_extra,
        &code_length_bitdepth,
        &code_length_bitdepth_symbols,
        w,
    );
}

fn store_var_len_u16(n: u32, w: &mut BitWriter) {
    debug_assert!(n <= 65535);
    if n == 0 {
        w.write(1, 0);
    } else {
        w.write(1, 1);
        let nbits = 31 - n.leading_zeros();
        w.write(4, nbits as u64);
        w.write(nbits as usize, (n - (1u32 << nbits)) as u64);
    }
}

fn write_prefix_code_single(
    code: &PrefixCode,
    huffman_pool: &mut Vec<HuffmanNode>,
    w: &mut BitWriter,
) {
    let mut count = 0usize;
    let mut s4: [usize; 4] = [0; 4];
    let mut length = 0usize;
    for i in 0..ALPHABET_SIZE {
        if code.depths[i] != 0 {
            if count < 4 {
                s4[count] = i;
            }
            count += 1;
            length = i + 1;
        }
    }
    let mut max_bits = 0usize;
    let mut t = length.saturating_sub(1);
    while t != 0 {
        t >>= 1;
        max_bits += 1;
    }

    if count <= 1 {
        w.write(4, 1);
        w.write(max_bits, s4[0] as u64);
        return;
    }
    if count <= 4 {
        store_simple_huffman_tree(&code.depths, &mut s4, count, max_bits, w);
    } else {
        store_huffman_tree(&code.depths[..length], huffman_pool, w);
    }
}

/// Write a vector of prefix codes (per WritePrefixCodes in libjxl-tiny).
/// Serialize one `HybridUintConfig`.
///
/// `split_width` is `ceil_log2(log_alpha_size + 1)`: 4 bits for the prefix path
/// (`log_alpha_size = 15`) and 3 for the ANS path (`ANS_LOG_ALPHA_SIZE = 7`).
/// The `msb`/`lsb` field widths then depend on the split exponent itself, which
/// is why a config cannot simply be written as a fixed 8-bit blob.
fn write_uint_config(config: HybridUintConfig, split_width: usize, w: &mut BitWriter) {
    let split = config.split_exponent as u32;
    let msb = config.msb_in_token as u32;
    let lsb = config.lsb_in_token as u32;
    w.write(split_width, split as u64);
    let msb_width = if split == 0 {
        0
    } else {
        32 - split.leading_zeros()
    };
    w.write(msb_width as usize, msb as u64);
    let remaining = split - msb;
    let lsb_width = if remaining == 0 {
        0
    } else {
        32 - remaining.leading_zeros()
    };
    w.write(lsb_width as usize, lsb as u64);
}

pub(crate) fn write_prefix_codes(
    codes: &[PrefixCode],
    configs: &[HybridUintConfig],
    huffman_pool: &mut Vec<HuffmanNode>,
    w: &mut BitWriter,
) {
    w.write(1, 1); // use_prefix_code
    debug_assert_eq!(codes.len(), configs.len());
    for &config in configs {
        write_uint_config(config, 4, w);
    }
    // num_symbol per code.
    for code in codes.iter() {
        let mut num_symbol = 1usize;
        for i in 0..ALPHABET_SIZE {
            if code.depths[i] != 0 {
                num_symbol = i + 1;
            }
        }
        store_var_len_u16((num_symbol - 1) as u32, w);
    }
    // The actual prefix codes.
    for code in codes.iter() {
        let mut num_symbol = 1usize;
        for i in 0..ALPHABET_SIZE {
            if code.depths[i] != 0 {
                num_symbol = i + 1;
            }
        }
        if num_symbol > 1 {
            write_prefix_code_single(code, huffman_pool, w);
        }
    }
}

/// Move-to-front transform of a context map, libjxl `MoveToFrontTransform`.
/// Long runs of one cluster become runs of symbol 0, which the prefix code then
/// codes in a fraction of a bit — the difference between ~0.66 bits and ~0.15
/// bits per entry on a map with thousands of contexts.
fn move_to_front_transform(map: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let Some(&max_value) = map.iter().max() else {
        return;
    };
    let mut mtf: Vec<u8> = (0..=max_value).collect();
    for &value in map {
        let index = mtf.iter().position(|&v| v == value).unwrap_or(0);
        out.push(index as u8);
        // Move the used symbol to the front.
        for i in (1..=index).rev() {
            mtf[i] = mtf[i - 1];
        }
        mtf[0] = value;
    }
}

/// Order-0 cost in bits of coding `symbols` with an optimal prefix code,
/// including a rough allowance for describing the code itself. Only used to
/// choose between the raw and MTF orderings, so relative accuracy is enough.
fn context_map_cost(symbols: &[u8]) -> f64 {
    let mut counts = [0u32; 256];
    for &s in symbols {
        counts[s as usize] += 1;
    }
    let total = symbols.len() as f64;
    let mut bits = 0.0;
    let mut used = 0.0;
    for &c in counts.iter() {
        if c != 0 {
            let p = c as f64 / total;
            bits += -(c as f64) * f_log2(p);
            used += 1.0;
        }
    }
    // ~8 bits to describe each used symbol's code length.
    bits + used * 8.0
}

/// Emit a context map, following libjxl `EncodeContextMap`: an all-zero map is
/// signaled as "simple, 0 bits per entry"; otherwise the entries are coded with
/// a prefix code, optionally after a move-to-front transform.
pub(crate) fn write_context_map(
    code: &EntropyCode,
    huffman_pool: &mut Vec<HuffmanNode>,
    w: &mut BitWriter,
) {
    let num_contexts = if code.orig_context_map.is_some() {
        code.orig_num_contexts
    } else {
        code.num_contexts
    };
    if num_contexts == 0 {
        return;
    }

    let max = *code.context_map.iter().max().unwrap_or(&0);
    if max == 0 {
        w.write(3, 1);
        return;
    }

    // The entries in signaled order.
    let mut entries: Vec<u8> = Vec::with_capacity(num_contexts);
    match code.orig_context_map {
        Some(orig) => {
            for i in 0..code.orig_num_contexts {
                entries.push(code.context_map[orig[i] as usize]);
            }
        }
        None => entries.extend_from_slice(code.context_map),
    }

    let mut mtf = Vec::with_capacity(entries.len());
    move_to_front_transform(&entries, &mut mtf);

    // Four candidate encodings: {raw, MTF} x {plain, run-length}. MTF turns runs
    // of one cluster into runs of symbol 0, but it also *breaks* the periodic
    // repeats that run-length coding feeds on, so the two are not additive and
    // the pair has to be chosen jointly. Measured on real AC maps: plain 546 B,
    // MTF 546 B, MTF+RLE 463 B, raw+RLE 437 B.
    let raw_runs = run_length_symbols(&entries);
    let mtf_runs = run_length_symbols(&mtf);
    let costs = [
        context_map_cost(&entries),
        context_map_cost(&mtf),
        run_length_cost(&raw_runs),
        run_length_cost(&mtf_runs),
    ];
    let best = costs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let use_mtf = best == 1 || best == 3;
    let use_lz77 = best >= 2;
    let symbols: &[u8] = if use_mtf { &mtf } else { &entries };

    // is_simple = 0, use_mtf, then the histogram bundle's lz77_enabled bit.
    w.write(1, 0);
    w.write(1, u64::from(use_mtf));
    if !use_lz77 {
        w.write(1, 0);
        let tokens: Vec<Token> = symbols.iter().map(|&v| Token::new(0, v as u32)).collect();
        let ctxmap_code = optimize_prefix_codes(&tokens, vec![0u8], 1, huffman_pool);
        let ctxmap_ref = ctxmap_code.as_ref();
        write_prefix_codes(
            &ctxmap_code.prefix_codes,
            &ctxmap_code.hybrid_uint_configs,
            huffman_pool,
            w,
        );
        for t in &tokens {
            write_token(*t, &ctxmap_ref, w);
        }
        return;
    }

    // Run-length coded: literals on context 0, back-reference lengths on
    // context 0 as `LZ77_MIN_SYMBOL + length_token`, one distance symbol per
    // reference on context 1. Distance value 0 decodes as distance 1 (the
    // decoder computes `distance + 1 - num_special_distances`, and this stream
    // has no special distances), i.e. exactly a run.
    let runs = if use_mtf { &mtf_runs } else { &raw_runs };
    // The histograms are built by hand: an LZ77 length symbol enters the
    // alphabet directly, whereas `build_histograms` would push it through
    // `uint_encode` and leave the prefix code without a codeword for it.
    let mut histograms = vec![Histogram::new(); 2];
    for r in runs {
        match *r {
            RunSymbol::Literal(v) => {
                let (sym, _, _) = uint_encode(u32::from(v));
                histograms[0].add(sym);
            }
            RunSymbol::Copy { length_value } => {
                let (len_tok, _, _) = lz77_length_encode(length_value);
                histograms[0].add(LZ77_MIN_SYMBOL + len_tok);
                let (dsym, _, _) = uint_encode(CTXMAP_LZ77_DISTANCE);
                histograms[1].add(dsym);
            }
        }
    }
    let mut lz_context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut lz_context_map, huffman_pool);
    let prefix_codes = build_huffman_codes(&histograms, huffman_pool);
    let code = OwnedEntropyCode {
        context_map: lz_context_map,
        prefix_codes,
        hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; histograms.len()],
        orig_context_map: None,
        orig_num_contexts: 2,
        use_prefix_code: true,
        ans_freqs: Vec::new(),
        ans_symbols: Vec::new(),
    };

    w.write(1, 1); // lz77 enabled
    // min_symbol = 64: U32 selector 3 + 15 bits of (64 - 8).
    w.write(2, 0b11);
    w.write(15, u64::from(LZ77_MIN_SYMBOL - 8));
    w.write(2, 0b00); // min_length = 3
    // length_uint_config under log_alpha_size 8: split_exp = 4, msb = 0, lsb = 0.
    w.write(4, 4);
    w.write(3, 0);
    w.write(3, 0);
    // Two contexts, so the bundle carries its own (nested) context map.
    write_entropy_code(&code.as_ref(), huffman_pool, w);

    let code_ref = code.as_ref();
    for r in runs {
        match *r {
            RunSymbol::Literal(v) => write_token(Token::new(0, u32::from(v)), &code_ref, w),
            RunSymbol::Copy { length_value } => {
                let (len_tok, len_nbits, len_bits) = lz77_length_encode(length_value);
                let sym = LZ77_MIN_SYMBOL + len_tok;
                let cluster = code_ref.context_map[0] as usize;
                let pc = &code_ref.prefix_codes[cluster];
                if pc.single_symbol {
                    w.write(len_nbits as usize, u64::from(len_bits));
                } else {
                    let d = pc.depths[sym as usize] as usize;
                    debug_assert!(d > 0, "context-map LZ77 length symbol {sym} unrepresented");
                    let data = u64::from(pc.bits[sym as usize]) | (u64::from(len_bits) << d);
                    w.write(d + len_nbits as usize, data);
                }
                write_token(Token::new(1, CTXMAP_LZ77_DISTANCE), &code_ref, w);
            }
        }
    }
}

/// Distance value for a run back-reference. The decoder reconstructs
/// `distance + 1` when the stream has no special-distance table, so 0 means
/// "copy from the immediately preceding symbol".
const CTXMAP_LZ77_DISTANCE: u32 = 0;

#[derive(Clone, Copy)]
enum RunSymbol {
    Literal(u8),
    Copy { length_value: u32 },
}

/// Split a symbol stream into literals and run back-references, mirroring the
/// AC path's distance-1 LZ77. A run of `n` equal symbols becomes one literal
/// plus a copy of `n - 1` when that clears `LZ77_MIN_LENGTH`.
fn run_length_symbols(symbols: &[u8]) -> Vec<RunSymbol> {
    let mut out: Vec<RunSymbol> = Vec::with_capacity(symbols.len());
    let mut i = 0usize;
    while i < symbols.len() {
        let mut j = i;
        while j + 1 < symbols.len() && symbols[j + 1] == symbols[i] {
            j += 1;
        }
        out.push(RunSymbol::Literal(symbols[i]));
        let extra = (j - i) as u32;
        if extra >= LZ77_MIN_LENGTH {
            out.push(RunSymbol::Copy {
                length_value: extra - LZ77_MIN_LENGTH,
            });
        } else {
            for _ in 0..extra {
                out.push(RunSymbol::Literal(symbols[i]));
            }
        }
        i = j + 1;
    }
    out
}

/// Order-0 cost of a run-length symbol stream, on the same footing as
/// [`context_map_cost`] so the four candidates are comparable.
fn run_length_cost(runs: &[RunSymbol]) -> f64 {
    let mut counts = [0u32; 512];
    let mut extra_bits = 0.0;
    let mut distances = 0u32;
    for r in runs {
        match *r {
            RunSymbol::Literal(v) => counts[v as usize] += 1,
            RunSymbol::Copy { length_value } => {
                let (len_tok, len_nbits, _) = lz77_length_encode(length_value);
                counts[(LZ77_MIN_SYMBOL + len_tok) as usize] += 1;
                extra_bits += f64::from(len_nbits);
                distances += 1;
            }
        }
    }
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut bits = 0.0;
    let mut used = 0.0;
    for &c in counts.iter() {
        if c != 0 {
            let p = f64::from(c) / f64::from(total);
            bits += -f64::from(c) * f_log2(p);
            used += 1.0;
        }
    }
    // The distance context costs one symbol per reference plus its own code, and
    // a second context makes the bundle carry a nested context map.
    bits + extra_bits + used * 8.0 + f64::from(distances) + 24.0
}

/// WriteContextMap + the per-bundle code parameters (prefix codes or ANS).
pub(crate) fn write_entropy_code(
    code: &EntropyCode,
    huffman_pool: &mut Vec<HuffmanNode>,
    w: &mut BitWriter,
) {
    write_context_map(code, huffman_pool, w);
    if code.use_prefix_code {
        write_prefix_codes(code.prefix_codes, code.hybrid_uint_configs, huffman_pool, w);
    } else {
        write_ans_params(code, w);
    }
}

fn write_ans_params(code: &EntropyCode, w: &mut BitWriter) {
    w.write(1, 0); // use_prefix_code = 0
    w.write(2, (ANS_LOG_ALPHA_SIZE - 5) as u64); // log_alpha_size = 7
    // Per-histogram hybrid-uint config. This used to hardcode (4, 2, 0), which
    // silently contradicted `hybrid_uint_configs` the moment anything but the
    // default was selected — the header would advertise one configuration while
    // the tokens used another.
    debug_assert_eq!(code.hybrid_uint_configs.len(), code.ans_freqs.len());
    for &config in code.hybrid_uint_configs.iter().take(code.ans_freqs.len()) {
        write_uint_config(config, 3, w);
    }
    // The normalized distributions, in clustered-histogram order.
    for freqs in code.ans_freqs.iter() {
        encode_histogram(freqs, ANS_LOG_ALPHA_SIZE, w);
    }
}

#[cfg(test)]
mod context_map_tests {
    use super::{RunSymbol, move_to_front_transform, run_length_symbols};
    use crate::enc_lz77_ac::{LZ77_MIN_LENGTH, lz77_length_encode};

    /// Decode a run-length stream the way the JXL reader does: a copy of
    /// `length` symbols from distance 1, i.e. repeat the previous symbol.
    fn decode(runs: &[RunSymbol]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for r in runs {
            match *r {
                RunSymbol::Literal(v) => out.push(v),
                RunSymbol::Copy { length_value } => {
                    let length = length_value + LZ77_MIN_LENGTH;
                    let last = *out.last().expect("copy with empty history");
                    for _ in 0..length {
                        out.push(last);
                    }
                }
            }
        }
        out
    }

    /// The run splitter and the decoder's copy semantics have to agree exactly;
    /// an off-by-one in the length offset silently corrupts the context map,
    /// which then mis-assigns every histogram in the frame.
    #[test]
    fn run_length_round_trips() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![7],
            vec![1, 1, 1],
            vec![1, 1, 1, 1],
            vec![0; 40],
            vec![3, 3, 3, 3, 9, 9, 1, 2, 3, 4, 4, 4, 4, 4, 4],
            (0..200u32).map(|i| (i % 5) as u8).collect(),
            (0..200u32).map(|i| if i < 100 { 2 } else { 0 }).collect(),
        ];
        for case in cases {
            let runs = run_length_symbols(&case);
            assert_eq!(decode(&runs), case, "round trip failed for {case:?}");
        }
    }

    /// Every length symbol the encoder can emit must survive the hybrid-uint
    /// split it is written with.
    #[test]
    fn run_lengths_encode_within_the_alphabet() {
        let long: Vec<u8> = vec![4; 5000];
        for r in run_length_symbols(&long) {
            if let RunSymbol::Copy { length_value } = r {
                let (tok, nbits, bits) = lz77_length_encode(length_value);
                assert!(tok < 64, "length token {tok} escapes the alphabet");
                assert!(nbits <= 32);
                if nbits < 32 {
                    assert!(bits >> nbits == 0, "extra bits overflow the field");
                }
            }
        }
    }

    #[test]
    fn move_to_front_is_a_permutation_preserving_transform() {
        let src: Vec<u8> = vec![3, 3, 1, 0, 0, 2, 3, 1, 1, 1, 4];
        let mut mtf = Vec::new();
        move_to_front_transform(&src, &mut mtf);
        assert_eq!(mtf.len(), src.len());
        // Invert it: the same table walk, reading indices back to values.
        let max = *src.iter().max().unwrap();
        let mut table: Vec<u8> = (0..=max).collect();
        let back: Vec<u8> = mtf
            .iter()
            .map(|&i| {
                let v = table[i as usize];
                table.remove(i as usize);
                table.insert(0, v);
                v
            })
            .collect();
        assert_eq!(back, src);
    }
}
