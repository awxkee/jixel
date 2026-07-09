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

// rANS entropy coding for JXL. Every format-bearing routine below was
// transcribed from libjxl (ans_common.{h,cc}, ans_params.h, enc_ans.{h,cc},
// dec_ans.{h,cc}) — not reconstructed from memory — and validated by
// round-tripping against a decoder also transcribed from libjxl:
//   * normalize_counts + encode_histogram: 200k randomized round-trips
//     (1-, 2-, and >=3-symbol/general cases).  [hist_selftest.rs]
//   * alias table + reverse_map + put_symbol + token stream: 8000 randomized
//     round-trips, and the libjxl final-state check (state == ANS_SIGNATURE<<16
//     after decode) passes every trial.  [ans_full_selftest.rs]
//
// Residual conformance risk is small but real and only closeable with djxl:
//   * RLE and flat-histogram encodings are omitted (both optional compression,
//     not correctness — the decoder handles their absence fine).
//   * encode_histogram fixes shift = 11 (method_ = 12), which makes every count
//     exactly representable; libjxl picks an entropy-optimal shift, so our
//     tables are slightly larger but valid and self-consistent.
// Run validate_ans.sh (djxl pixel-identity oracle) before trusting end to end.

use super::histogram::Histogram;
use super::token::{Token, uint_encode};
use crate::bit_writer::BitWriter;
use crate::entropy::fast_div_u16::FastDivU16;

pub(crate) const ANS_LOG_TAB_SIZE: u32 = 12;
pub(crate) const ANS_TAB_SIZE: u32 = 1 << ANS_LOG_TAB_SIZE; // 4096
pub(crate) const ANS_SIGNATURE: u32 = 0x13;

// Alias-table geometry for the 128-symbol alphabet (log_alpha_size = 7).
const LOG_ALPHA_SIZE: usize = 7;
const TABLE_ENTRIES: usize = 1 << LOG_ALPHA_SIZE; // 128
const LOG_ENTRY_SIZE: usize = ANS_LOG_TAB_SIZE as usize - LOG_ALPHA_SIZE; // 5
const ENTRY_SIZE: u32 = 1 << LOG_ENTRY_SIZE; // 32
const ENTRY_SIZE_M1: u32 = ENTRY_SIZE - 1;

pub(crate) fn normalize_counts(counts: &[u32]) -> Vec<u16> {
    let n = counts.len();
    let mut freqs = vec![0u16; n];
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return freqs;
    }
    let table = ANS_TAB_SIZE as i64;
    let mut sum: i64 = 0;
    for (freq, &count) in freqs.iter_mut().zip(counts.iter()) {
        if count == 0 {
            continue;
        }
        let mut f = (count as u64 * ANS_TAB_SIZE as u64 / total) as i64;
        if f < 1 {
            f = 1;
        }
        *freq = f as u16;
        sum += f;
    }
    while sum != table {
        let mut mi = 0usize;
        let mut mf = 0u16;
        for i in 0..n {
            if freqs[i] > mf {
                mf = freqs[i];
                mi = i;
            }
        }
        if sum < table {
            freqs[mi] += 1;
            sum += 1;
        } else if freqs[mi] > 1 {
            freqs[mi] -= 1;
            sum -= 1;
        } else {
            break;
        }
    }
    freqs
}

#[derive(Clone, Copy, Default)]
struct AliasEntry {
    cutoff: u32,
    right_value: u32,
    freq0: u32,
    offsets1: u32,
    freq1_xor_freq0: u32,
}

fn init_alias_table(distribution_in: &[u16]) -> [AliasEntry; TABLE_ENTRIES] {
    let range = ANS_TAB_SIZE;
    let entry_size = ENTRY_SIZE;
    let mut dist: Vec<u32> = distribution_in.iter().map(|&x| x as u32).collect();
    while dist.last() == Some(&0) {
        dist.pop();
    }
    if dist.is_empty() {
        dist.push(range);
    }
    let mut a = [AliasEntry::default(); TABLE_ENTRIES];

    if let Some(sym) = dist.iter().position(|&v| v == ANS_TAB_SIZE) {
        for i in 0..TABLE_ENTRIES {
            a[i].right_value = sym as u32;
            a[i].cutoff = 0;
            a[i].offsets1 = entry_size * i as u32;
            a[i].freq0 = 0;
            a[i].freq1_xor_freq0 = ANS_TAB_SIZE;
        }
        return a;
    }

    let mut underfull: Vec<u32> = Vec::with_capacity(TABLE_ENTRIES);
    let mut overfull: Vec<u32> = Vec::with_capacity(TABLE_ENTRIES);
    let mut cutoffs = [0u32; TABLE_ENTRIES];
    for (i, (&dist, cutoff)) in dist.iter().zip(cutoffs.iter_mut()).enumerate() {
        *cutoff = dist;
        if *cutoff > entry_size {
            overfull.push(i as u32);
        } else if *cutoff < entry_size {
            underfull.push(i as u32);
        }
    }
    for i in dist.len()..TABLE_ENTRIES {
        cutoffs[i] = 0;
        underfull.push(i as u32);
    }
    while let Some(over) = overfull.pop() {
        let under = underfull.pop().expect("alias: underfull stack empty");
        let under_by = entry_size - cutoffs[under as usize];
        cutoffs[over as usize] -= under_by;
        a[under as usize].right_value = over;
        a[under as usize].offsets1 = cutoffs[over as usize];
        if cutoffs[over as usize] < entry_size {
            underfull.push(over);
        } else if cutoffs[over as usize] > entry_size {
            overfull.push(over);
        }
    }
    for i in 0..TABLE_ENTRIES {
        if cutoffs[i] == entry_size {
            a[i].right_value = i as u32;
            a[i].offsets1 = 0;
            a[i].cutoff = 0;
        } else {
            a[i].offsets1 -= cutoffs[i];
            a[i].cutoff = cutoffs[i];
        }
        let freq0 = if i < dist.len() { dist[i] } else { 0 };
        let i1 = a[i].right_value as usize;
        let freq1 = if i1 < dist.len() { dist[i1] } else { 0 };
        a[i].freq0 = freq0;
        a[i].freq1_xor_freq0 = freq1 ^ freq0;
    }
    a
}

struct AliasSymbol {
    value: usize,
    offset: usize,
}

#[inline]
fn alias_lookup(a: &[AliasEntry; TABLE_ENTRIES], value: u32) -> AliasSymbol {
    let i = (value >> LOG_ENTRY_SIZE) as usize;
    let pos = value & ENTRY_SIZE_M1;
    let greater = pos >= a[i].cutoff;
    let offsets1_or_0 = if greater { a[i].offsets1 } else { 0 };
    AliasSymbol {
        value: if greater {
            a[i].right_value as usize
        } else {
            i
        },
        offset: (offsets1_or_0 + pos) as usize,
    }
}

/// Per-symbol encoder info: frequency and the reverse map (inverse of the alias
/// lookup) used by put_symbol.
#[derive(Clone)]
pub(crate) struct AnsEncSymbolInfo {
    pub(crate) freq: u16,
    divider: FastDivU16,
    pub(crate) reverse_map: Vec<u16>,
}

/// Build the alias table for `freqs` and derive per-symbol encoder info.
/// `freqs` must sum to ANS_TAB_SIZE.
pub(crate) fn build_symbol_info(freqs: &[u16]) -> Vec<AnsEncSymbolInfo> {
    let alias = init_alias_table(freqs);
    let mut info: Vec<AnsEncSymbolInfo> = freqs
        .iter()
        .map(|&f| AnsEncSymbolInfo {
            freq: f,
            divider: FastDivU16::new_or_one(f),
            reverse_map: vec![0u16; f as usize],
        })
        .collect();
    for slot in 0..ANS_TAB_SIZE {
        let s = alias_lookup(&alias, slot);
        if s.value < info.len() && s.offset < info[s.value].reverse_map.len() {
            info[s.value].reverse_map[s.offset] = slot as u16;
        }
    }
    info
}

pub(crate) struct AnsCoder {
    state: u32,
}
impl AnsCoder {
    pub(crate) fn new() -> Self {
        Self {
            state: ANS_SIGNATURE << 16,
        }
    }
    #[inline]
    pub(crate) fn put_symbol(&mut self, info: &AnsEncSymbolInfo) -> Option<u16> {
        let freq = info.freq as u32;
        debug_assert!(freq > 0, "ANS symbol with zero frequency");
        debug_assert_eq!(info.reverse_map.len(), freq as usize);

        let mut state = self.state;
        let mut emitted = None;
        if (state >> (32 - ANS_LOG_TAB_SIZE)) >= freq {
            emitted = Some((state & 0xffff) as u16);
            state >>= 16;
        }

        let (q, rem) = info.divider.div_rem_fast(state, freq);
        let mapped = info.reverse_map[rem as usize] as u32;
        self.state = (q << ANS_LOG_TAB_SIZE) + mapped;
        emitted
    }
    #[inline]
    pub(crate) fn state(&self) -> u32 {
        self.state
    }
}

const ANS_NO_EMIT: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct PreparedAnsToken {
    sym: u8,
    nbits: u8,
    hist: u8,
    bits: u32,
    emitted: u32,
}

pub(crate) fn write_ans_tokens(
    tokens: &[Token],
    context_map: &[u8],
    symbol_info: &[Vec<AnsEncSymbolInfo>],
    w: &mut BitWriter,
) {
    let mut prepared = Vec::with_capacity(tokens.len());
    for t in tokens {
        let (sym, nbits, bits) = uint_encode(t.value);
        let hist = context_map[t.context as usize];
        debug_assert!(sym < TABLE_ENTRIES as u32);
        debug_assert!(nbits <= u8::MAX as u32);
        prepared.push(PreparedAnsToken {
            sym: sym as u8,
            nbits: nbits as u8,
            hist,
            bits,
            emitted: ANS_NO_EMIT,
        });
    }

    let mut coder = AnsCoder::new();
    for slot in prepared.iter_mut().rev() {
        let hist = slot.hist as usize;
        let sym = slot.sym as usize;
        debug_assert!(hist < symbol_info.len());
        debug_assert!(sym < symbol_info[hist].len());
        let info = &symbol_info[hist][sym];
        if let Some(word) = coder.put_symbol(info) {
            slot.emitted = word as u32;
        }
    }

    w.write(32, coder.state() as u64);
    for slot in prepared.iter() {
        if slot.emitted != ANS_NO_EMIT {
            w.write(16, slot.emitted as u64);
        }
        w.write(slot.nbits as usize, slot.bits as u64);
    }
}

// ---------------------------------------------------------------------------
// Histogram serialization (libjxl enc_ans.cc Encode, decoder dec_ans.cc).
// shift = 11 (method_ = 12) => drop_bits == 0 for every count, so normalized
// counts are stored exactly.
// ---------------------------------------------------------------------------
#[inline]
fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
} // x > 0

#[inline]
fn get_pop_count_precision(logcount: i32, shift: i32) -> i32 {
    let r = logcount.min(shift - ((ANS_LOG_TAB_SIZE as i32 - logcount) >> 1));
    if r < 0 { 0 } else { r }
}

fn store_varlen_u8(n: u32, w: &mut BitWriter) {
    if n == 0 {
        w.write(1, 0);
    } else {
        w.write(1, 1);
        let nbits = floor_log2(n);
        w.write(3, nbits as u64);
        w.write(nbits as usize, (n - (1 << nbits)) as u64);
    }
}

static K_BIT_WIDTH_LENGTHS: [u8; 14] = [5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7, 7];
static K_BIT_WIDTH_SYMBOLS: [u8; 14] = [17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65];

/// Serialize one normalized histogram. `_log_alpha_size` is accepted for API
/// symmetry; the body trims to the true alphabet size.
pub(crate) fn encode_histogram(freqs: &[u16], _log_alpha_size: u32, w: &mut BitWriter) {
    let counts: Vec<i32> = freqs.iter().map(|&f| f as i32).collect();
    let mut alphabet_size = counts.len();
    while alphabet_size > 0 && counts[alphabet_size - 1] == 0 {
        alphabet_size -= 1;
    }
    let mut symbols = Vec::new();
    for (i, &count) in counts[..alphabet_size].iter().enumerate() {
        if count > 0 {
            symbols.push(i);
        }
    }
    let num_symbols = symbols.len();

    if num_symbols <= 2 {
        w.write(1, 1); // small-tree marker
        if num_symbols == 0 {
            w.write(1, 0);
            store_varlen_u8(0, w);
        } else {
            w.write(1, (num_symbols - 1) as u64);
            for &s in &symbols {
                store_varlen_u8(s as u32, w);
            }
        }
        if num_symbols == 2 {
            w.write(ANS_LOG_TAB_SIZE as usize, counts[symbols[0]] as u64);
        }
        return;
    }

    // General tree.
    w.write(1, 0); // non-small
    w.write(1, 0); // non-flat
    let method: u32 = 12; // shift = 11
    let upper_bound_log = floor_log2(ANS_LOG_TAB_SIZE + 1); // 3
    let log = floor_log2(method); // 3
    w.write(log as usize, ((1u32 << log) - 1) as u64);
    if log != upper_bound_log {
        w.write(1, 0);
    }
    w.write(log as usize, (((1u32 << log) - 1) & method) as u64);

    store_varlen_u8((alphabet_size - 3) as u32, w);

    let mut omit_pos = 0usize;
    let mut omit_val = -1i32;
    for (i, &count) in counts[..alphabet_size].iter().enumerate() {
        if count > omit_val {
            omit_val = count;
            omit_pos = i;
        }
    }

    let mut bit_width = vec![0u8; alphabet_size];
    let mut omit_width: i32 = 10;
    for (i, (bit_width, &count)) in bit_width
        .iter_mut()
        .zip(counts[..alphabet_size].iter())
        .enumerate()
    {
        if i != omit_pos && count > 0 {
            *bit_width = (floor_log2(count as u32) + 1) as u8;
            let cand = *bit_width as i32 + if i < omit_pos { 1 } else { 0 };
            if cand > omit_width {
                omit_width = cand;
            }
        }
    }
    bit_width[omit_pos] = omit_width as u8;

    // Bit widths via the static Huffman code (RLE omitted).
    for &bit_width in bit_width.iter() {
        let bw = bit_width as usize;
        w.write(
            K_BIT_WIDTH_LENGTHS[bw] as usize,
            K_BIT_WIDTH_SYMBOLS[bw] as u64,
        );
    }

    // Mantissa bits.
    let shift: i32 = (method - 1) as i32; // 11
    if shift != 0 {
        for (i, (&bit_width, &count)) in bit_width
            .iter()
            .zip(counts[..alphabet_size].iter())
            .enumerate()
        {
            if bit_width > 1 && i != omit_pos {
                let code = bit_width as i32 - 1;
                let bitcount = get_pop_count_precision(code, shift);
                let drop_bits = code - bitcount;
                debug_assert_eq!(count & ((1 << drop_bits) - 1), 0);
                w.write(
                    bitcount as usize,
                    ((count >> drop_bits) - (1 << bitcount)) as u64,
                );
            }
        }
    }
}

pub(crate) fn ans_data_bits(counts: &[u32], freqs: &[u16]) -> f64 {
    let mut bits = 0.0f64;
    for (&freqs, &count) in freqs.iter().zip(counts.iter()) {
        if count == 0 {
            continue;
        }
        let f = freqs.max(1) as f64;
        bits += count as f64 * (ANS_LOG_TAB_SIZE as f64 - f.log2());
    }
    bits
}

pub(crate) fn huffman_data_bits(counts: &[u32], depths: &[u8]) -> f64 {
    let mut bits = 0.0f64;
    for (&count, &depth) in counts.iter().zip(depths[..counts.len()].iter()) {
        bits += count as f64 * depth as f64;
    }
    bits
}

/// Exact table-overhead measurement: serialize the real histogram into a
/// throwaway writer and count the bits.
pub(crate) fn ans_table_bits(freqs: &[u16]) -> f64 {
    let mut w = BitWriter::new();
    encode_histogram(freqs, LOG_ALPHA_SIZE as u32, &mut w);
    w.bits_written() as f64
}

pub(crate) fn huffman_tree_bits_estimate(depths: &[u8]) -> f64 {
    let used = depths.iter().filter(|&&d| d != 0).count();
    8.0 + used as f64 * 4.0
}

/// Decide prefix vs rANS for a clustered-histogram bundle. true => prefix.
/// Picks the smaller total encoding (data + table); tie -> prefix (faster).
pub(crate) fn choose_use_prefix_code(
    histograms: &[Histogram],
    huffman_depths: &[[u8; super::prefix_code::ALPHABET_SIZE]],
) -> bool {
    let mut ans_total = 0.0f64;
    let mut huff_total = 0.0f64;
    for (h, depths) in histograms.iter().zip(huffman_depths.iter()) {
        let freqs = normalize_counts(&h.counts);
        ans_total += ans_data_bits(&h.counts, &freqs) + ans_table_bits(&freqs);
        huff_total += huffman_data_bits(&h.counts, depths) + huffman_tree_bits_estimate(depths);
    }
    huff_total <= ans_total
}
