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

use super::cluster::cluster_histograms;
use super::entropy_code::{EntropyCode, OwnedEntropyCode};
use super::histogram::Histogram;
use super::huffman_tree::create_huffman_tree;
use super::prefix_code::{ALPHABET_SIZE, PrefixCode, convert_bit_depths_to_symbols};
use super::token::{Token, uint_encode};
use crate::bit_writer::BitWriter;

/// Hot-path token writer. Matches libjxl-tiny's inline WriteToken.
#[inline]
pub fn write_token(t: Token, code: &EntropyCode, w: &mut BitWriter) {
    let (tok, nbits, bits) = uint_encode(t.value);
    let pc = &code.prefix_codes[code.context_map[t.context as usize] as usize];
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
pub fn optimize_prefix_codes(
    tokens: &[Token],
    context_map: Vec<u8>,
    num_contexts: usize,
) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, Some(&context_map), &mut histograms);
    let prefix_codes = build_huffman_codes(&histograms);
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        orig_context_map: None,
        orig_num_contexts: 0,
    }
}

/// Build an entropy code from scratch given tokens and the number of
/// (original, pre-cluster) contexts: builds per-context histograms, clusters
/// them into ≤ 8 groups, then builds Huffman codes for the clusters.
pub fn optimize_entropy_code(tokens: &[Token], num_contexts: usize) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    build_histograms(tokens, None, &mut histograms);
    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map);
    let prefix_codes = build_huffman_codes(&histograms);
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        orig_context_map: None,
        orig_num_contexts: num_contexts,
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
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        orig_context_map: None,
        orig_num_contexts: num_contexts,
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
        decide_over_rle_use(&depth[..new_length])
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
    for i in 0..NUM_CODE_LENGTH_CODES {
        if histo[i] != 0 {
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
pub fn write_prefix_codes(codes: &[PrefixCode], w: &mut BitWriter) {
    w.write(1, 1); // use_prefix_code
    for _ in 0..codes.len() {
        w.write(4, 4);
        w.write(3, 2);
        w.write(2, 0);
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
pub fn write_context_map(code: &EntropyCode, w: &mut BitWriter) {
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
    write_prefix_codes(&ctxmap_code.prefix_codes, w);
    for t in &tokens {
        write_token(*t, &ctxmap_ref, w);
    }
}

/// WriteContextMap + WritePrefixCodes.
pub fn write_entropy_code(code: &EntropyCode, w: &mut BitWriter) {
    write_context_map(code, w);
    write_prefix_codes(code.prefix_codes, w);
}
