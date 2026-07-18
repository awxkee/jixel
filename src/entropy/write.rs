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
use super::huffman_tree::create_huffman_tree;
use super::prefix_code::{ALPHABET_SIZE, PrefixCode, convert_bit_depths_to_symbols};
use super::token::{HybridUintConfig, Token, uint_encode, uint_encode_with_config};
use crate::bit_writer::BitWriter;

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

pub(crate) fn select_hybrid_config(values: &[u32]) -> HybridUintConfig {
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
            create_huffman_tree(&histogram.counts, 15, &mut depths);
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

fn build_histograms(tokens: &[Token], context_map: Option<&[u8]>, histograms: &mut [Histogram]) {
    for t in tokens {
        let (tok, _, _) = uint_encode(t.value);
        let context = match context_map {
            Some(m) => m[t.context as usize] as usize,
            None => t.context as usize,
        };
        histograms[context].add(tok);
    }
}

pub(crate) fn build_huffman_codes(histograms: &[Histogram]) -> Vec<PrefixCode> {
    let mut out: Vec<PrefixCode> = Vec::with_capacity(histograms.len());
    for h in histograms {
        let counts: [u32; ALPHABET_SIZE] = h.counts;
        let mut length = ALPHABET_SIZE;
        while length > 0 && counts[length - 1] == 0 {
            length -= 1;
        }
        let mut depths = [0u8; ALPHABET_SIZE];
        if length > 0 {
            create_huffman_tree(&counts[..length], 15, &mut depths[..length]);
        }
        let mut bits = [0u16; ALPHABET_SIZE];
        convert_bit_depths_to_symbols(&depths, &mut bits);
        let mut pc = PrefixCode {
            depths,
            bits,
            single_symbol: false,
        };
        pc.update_single_symbol();
        out.push(pc);
    }
    out
}

/// Build a prefix-codes-only EntropyCode given a fixed context_map and the
/// number of contexts (= number of prefix codes). Used when the context map
/// is known up front (e.g. the static AC code).
pub(crate) fn optimize_prefix_codes(
    tokens: &[Token],
    context_map: Vec<u8>,
    num_contexts: usize,
) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, Some(&context_map), &mut histograms);
    let prefix_codes = build_huffman_codes(&histograms);
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

pub(crate) fn optimize_entropy_code(tokens: &[Token], num_contexts: usize) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, None, &mut histograms);
    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map);
    let prefix_codes = build_huffman_codes(&histograms);
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
pub(crate) fn optimize_entropy_code_ac(tokens: &[Token], num_contexts: usize) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, None, &mut histograms);
    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map);
    let prefix_codes = build_huffman_codes(&histograms);

    let (mut use_prefix_code, mut ans_freqs, mut ans_symbols) = no_ans();
    if ANS_ENABLED {
        let depths: Vec<[u8; ALPHABET_SIZE]> = prefix_codes.iter().map(|c| c.depths).collect();
        use_prefix_code = choose_use_prefix_code(&histograms, &depths);
        if !use_prefix_code {
            for h in &histograms {
                let f = normalize_counts(&h.counts);
                ans_symbols.push(build_symbol_info(&f));
                ans_freqs.push(f);
            }
        }
    }

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

pub(crate) fn build_entropy_code_no_cluster(
    tokens: &[Token],
    num_contexts: usize,
) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, None, &mut histograms);
    let context_map: Vec<u8> = (0..num_contexts as u8).collect();
    let prefix_codes = build_huffman_codes(&histograms);
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

fn store_huffman_tree(depths: &[u8], w: &mut BitWriter) {
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
    create_huffman_tree(&histo, 5, &mut code_length_bitdepth);
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

fn write_prefix_code_single(code: &PrefixCode, w: &mut BitWriter) {
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
        store_huffman_tree(&code.depths[..length], w);
    }
}

/// Write a vector of prefix codes (per WritePrefixCodes in libjxl-tiny).
pub(crate) fn write_prefix_codes(
    codes: &[PrefixCode],
    configs: &[HybridUintConfig],
    w: &mut BitWriter,
) {
    w.write(1, 1); // use_prefix_code
    debug_assert_eq!(codes.len(), configs.len());
    for &config in configs {
        let split = config.split_exponent as u32;
        let msb = config.msb_in_token as u32;
        let lsb = config.lsb_in_token as u32;
        w.write(4, split as u64);
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
            write_prefix_code_single(code, w);
        }
    }
}

/// Emit a context map. Mirrors libjxl-tiny's WriteContextMap.
pub(crate) fn write_context_map(code: &EntropyCode, w: &mut BitWriter) {
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
    w.write(3, 0);

    let mut tokens: Vec<Token> = Vec::with_capacity(num_contexts);
    match code.orig_context_map {
        Some(orig) => {
            for i in 0..code.orig_num_contexts {
                let v = code.context_map[orig[i] as usize] as u32;
                tokens.push(Token::new(0, v));
            }
        }
        None => {
            for &code in code.context_map.iter() {
                tokens.push(Token::new(0, code as u32));
            }
        }
    };

    let ctxmap_code = optimize_prefix_codes(&tokens, vec![0u8], 1);
    let ctxmap_ref = ctxmap_code.as_ref();
    write_prefix_codes(
        &ctxmap_code.prefix_codes,
        &ctxmap_code.hybrid_uint_configs,
        w,
    );
    for t in &tokens {
        write_token(*t, &ctxmap_ref, w);
    }
}

/// WriteContextMap + the per-bundle code parameters (prefix codes or ANS).
pub(crate) fn write_entropy_code(code: &EntropyCode, w: &mut BitWriter) {
    write_context_map(code, w);
    if code.use_prefix_code {
        write_prefix_codes(code.prefix_codes, code.hybrid_uint_configs, w);
    } else {
        write_ans_params(code, w);
    }
}

fn write_ans_params(code: &EntropyCode, w: &mut BitWriter) {
    w.write(1, 0); // use_prefix_code = 0
    w.write(2, (ANS_LOG_ALPHA_SIZE - 5) as u64); // log_alpha_size = 7
    // Per-histogram hybrid-uint config (4, 2, 0) under log_alpha_size = 7.
    for _ in 0..code.ans_freqs.len() {
        w.write(3, 4); // split_exponent  (VERIFY width = CeilLog2(log_alpha+1))
        w.write(3, 2); // msb_in_token
        w.write(2, 0); // lsb_in_token
    }
    // The normalized distributions, in clustered-histogram order.
    for freqs in code.ans_freqs.iter() {
        encode_histogram(freqs, ANS_LOG_ALPHA_SIZE, w);
    }
}
