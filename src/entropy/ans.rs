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
//   * histogram normalization + encoding: randomized round-trips
//     (1-, 2-, and >=3-symbol/general cases).  [hist_selftest.rs]
//   * alias table + reverse_map + put_symbol + token stream: 8000 randomized
//     round-trips, and the libjxl final-state check (state == ANS_SIGNATURE<<16
//     after decode) passes every trial.  [ans_full_selftest.rs]
//
// Run the djxl pixel-identity oracle before trusting format-bearing changes end
// to end.

use super::histogram::Histogram;
use super::token::Token;
use crate::bit_writer::BitWriter;
use crate::entropy::dlog2::{f_fmla, f_log2};
use crate::entropy::fast_div_u16::FastDivU16;
use std::sync::OnceLock;

pub(crate) const ANS_LOG_TAB_SIZE: u32 = 12;
pub(crate) const ANS_TAB_SIZE: u32 = 1 << ANS_LOG_TAB_SIZE; // 4096
pub(crate) const ANS_SIGNATURE: u32 = 0x13;

/// A normalized ANS distribution together with the representation selected for
/// its bitstream header. Method 0 is flat, method 1 is the lowest precision,
/// and methods 2..=12 use progressively more count precision. Small trees use
/// their dedicated representation regardless of `method`.
#[derive(Clone)]
pub(crate) struct AnsHistogram {
    pub(crate) freqs: Vec<u16>,
    method: u8,
    omit_pos: u8,
    cost: f64,
}

// Alias-table geometry for the 128-symbol alphabet (log_alpha_size = 7).
const LOG_ALPHA_SIZE: usize = 7;
const TABLE_ENTRIES: usize = 1 << LOG_ALPHA_SIZE; // 128
const LOG_ENTRY_SIZE: usize = ANS_LOG_TAB_SIZE as usize - LOG_ALPHA_SIZE; // 5
const ENTRY_SIZE: u32 = 1 << LOG_ENTRY_SIZE; // 32
const ENTRY_SIZE_M1: u32 = ENTRY_SIZE - 1;

pub(crate) fn normalize_counts(counts: &[u32], freqs: &mut Vec<u16>) {
    let n = counts.len();
    if freqs.len() != n {
        freqs.resize(n, 0);
    }
    normalize_counts_into(counts, freqs);
}

pub(crate) fn normalize_counts_into(counts: &[u32], freqs: &mut [u16]) {
    debug_assert_eq!(counts.len(), freqs.len());
    let n = counts.len();
    freqs.fill(0);
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return;
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
}

#[derive(Clone, Copy)]
struct AllowedCount {
    count: u16,
    step_log: u8,
    delta_log: f64,
}

struct AllowedCounts {
    descending: Vec<AllowedCount>,
    index: Box<[u16; ANS_TAB_SIZE as usize]>,
}

static ALLOWED_COUNTS: OnceLock<[AllowedCounts; ANS_LOG_TAB_SIZE as usize]> = OnceLock::new();
static COUNT_LOG2: OnceLock<Box<[f64]>> = OnceLock::new();

fn count_log2() -> &'static [f64] {
    COUNT_LOG2.get_or_init(|| {
        (0..=ANS_TAB_SIZE)
            .map(|v| if v == 0 { 0.0 } else { f_log2(v as f64) })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

fn allowed_counts() -> &'static [AllowedCounts; ANS_LOG_TAB_SIZE as usize] {
    ALLOWED_COUNTS.get_or_init(|| {
        let logs = count_log2();
        core::array::from_fn(|shift| {
            let mut descending = Vec::with_capacity(ANS_TAB_SIZE as usize);
            let mut index = Box::new([0u16; ANS_TAB_SIZE as usize]);
            let mut last = u16::MAX;
            for i in (0..ANS_TAB_SIZE as u16).rev() {
                let step_log = if i == 0 {
                    0
                } else {
                    let log = floor_log2(i as u32) as i32;
                    (log - get_pop_count_precision(log, shift as i32)) as u8
                };
                let current = i & !((1u16 << step_log) - 1);
                if current == last {
                    continue;
                }
                last = current;
                let slot = descending.len();
                index[current as usize] = slot as u16;
                if current == 0 {
                    descending.push(AllowedCount {
                        count: 0,
                        step_log: 0,
                        delta_log: f64::INFINITY,
                    });
                } else {
                    let previous = descending.last().map_or(current, |v| v.count);
                    descending.push(AllowedCount {
                        count: current,
                        step_log: if slot == 0 {
                            0
                        } else {
                            floor_log2((previous - current) as u32) as u8
                        },
                        delta_log: if slot == 0 {
                            0.0
                        } else {
                            logs[previous as usize] - logs[current as usize]
                        },
                    });
                }
            }
            AllowedCounts { descending, index }
        })
    })
}

#[derive(Clone, Copy)]
struct AdjustableBin {
    source_count: u32,
    count_index: usize,
    bin_index: usize,
}

/// Normalize to a distribution representable with `shift` precision. This is
/// the same greedy entropy maximization used by libjxl: one largest bin absorbs
/// the remainder while the other bins move among representable counts.
fn rebalance_counts(counts: &[u32], alphabet_size: usize, shift: usize) -> Option<AnsHistogram> {
    let total: u64 = counts[..alphabet_size].iter().map(|&v| v as u64).sum();
    if total == 0 {
        return Some(AnsHistogram {
            freqs: vec![0; counts.len()],
            method: (shift + 1) as u8,
            omit_pos: 0,
            cost: 0.0,
        });
    }
    let allowed = &allowed_counts()[shift];
    let logs = count_log2();
    let mut freqs = vec![0u16; counts.len()];
    let norm = ANS_TAB_SIZE as f64 / total as f64;
    let mut omit_pos = 0usize;
    let mut max_source = 0u32;
    let mut rest = ANS_TAB_SIZE as i32;
    let mut bins = Vec::with_capacity(alphabet_size);

    for i in 0..alphabet_size {
        let source = counts[i];
        if source > max_source {
            max_source = source;
            omit_pos = i;
        }
        let target = source as f64 * norm;
        let mut count = target.round().clamp((source != 0) as u8 as f64, 4095.0) as u16;
        let log = if count == 0 {
            0
        } else {
            floor_log2(count as u32) as i32
        };
        let step_log = if count == 0 {
            0
        } else {
            log - get_pop_count_precision(log, shift as i32)
        };
        count &= !((1u16 << step_log) - 1);
        freqs[i] = count;
        rest -= count as i32;
        if target > 1.0 {
            bins.push(AdjustableBin {
                source_count: source,
                count_index: allowed.index[count as usize] as usize,
                bin_index: i,
            });
        }
    }
    bins.retain(|b| b.bin_index != omit_pos);
    rest += freqs[omit_pos] as i32;

    if !bins.is_empty() {
        let max_step_log = allowed.descending[1].step_log as usize;
        loop {
            let mut balance_inc = [0.0f64; ANS_LOG_TAB_SIZE as usize - 1];
            let mut balance_dec = [0.0f64; ANS_LOG_TAB_SIZE as usize - 1];
            for log in 0..=max_step_log {
                let delta = 1i32 << log;
                if rest >= ANS_TAB_SIZE as i32 {
                    balance_inc[log] = 0.0;
                    balance_dec[log] = 0.0;
                } else if rest > 1 {
                    balance_inc[log] = if rest > delta {
                        max_source as f64 * (logs[rest as usize] - logs[(rest - delta) as usize])
                    } else {
                        f64::INFINITY
                    };
                    balance_dec[log] = if rest + delta < ANS_TAB_SIZE as i32 {
                        max_source as f64 * (logs[(rest + delta) as usize] - logs[rest as usize])
                    } else {
                        f64::INFINITY
                    };
                } else {
                    balance_inc[log] = f64::INFINITY;
                    balance_dec[log] = f64::INFINITY;
                }
            }

            let inc_delta = |b: &AdjustableBin| {
                let a = allowed.descending[b.count_index];
                (b.source_count as f64 * a.delta_log - balance_inc[a.step_log as usize])
                    / (1u32 << a.step_log) as f64
            };
            let dec_delta = |b: &AdjustableBin| {
                let a = allowed.descending[b.count_index + 1];
                (b.source_count as f64 * a.delta_log - balance_dec[a.step_log as usize])
                    / (1u32 << a.step_log) as f64
            };

            let best_inc = bins
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| inc_delta(a).total_cmp(&inc_delta(b)))
                .map(|(i, _)| i)
                .unwrap();
            if inc_delta(&bins[best_inc]) > 0.0 {
                let step = allowed.descending[bins[best_inc].count_index].step_log;
                bins[best_inc].count_index -= 1;
                rest -= 1i32 << step;
                continue;
            }
            let best_dec = bins
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| dec_delta(a).total_cmp(&dec_delta(b)))
                .map(|(i, _)| i)
                .unwrap();
            if dec_delta(&bins[best_dec]) >= 0.0 {
                break;
            }
            bins[best_dec].count_index += 1;
            let step = allowed.descending[bins[best_dec].count_index].step_log;
            rest += 1i32 << step;
        }
        for bin in bins {
            freqs[bin.bin_index] = allowed.descending[bin.count_index].count;
        }
        for i in 0..omit_pos {
            if freqs[i] >= 2048 {
                freqs[omit_pos] = freqs[i];
                omit_pos = i;
                break;
            }
        }
    }
    if !(1..ANS_TAB_SIZE as i32).contains(&rest) {
        return None;
    }
    freqs[omit_pos] = rest as u16;
    Some(AnsHistogram {
        freqs,
        method: (shift + 1) as u8,
        omit_pos: omit_pos as u8,
        cost: 0.0,
    })
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
    for (i, cutoff) in
        (dist.len()..TABLE_ENTRIES).zip(cutoffs[dist.len()..TABLE_ENTRIES].iter_mut())
    {
        *cutoff = 0;
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
    for (i, (&cutoff, a)) in cutoffs[..TABLE_ENTRIES]
        .iter()
        .zip(a[..TABLE_ENTRIES].iter_mut())
        .enumerate()
    {
        if cutoff == entry_size {
            a.right_value = i as u32;
            a.offsets1 = 0;
            a.cutoff = 0;
        } else {
            a.offsets1 -= cutoff;
            a.cutoff = cutoff;
        }
        let freq0 = if i < dist.len() { dist[i] } else { 0 };
        let i1 = a.right_value as usize;
        let freq1 = if i1 < dist.len() { dist[i1] } else { 0 };
        a.freq0 = freq0;
        a.freq1_xor_freq0 = freq1 ^ freq0;
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

/// Per-symbol encoder info. `reverse_offset` selects this symbol's range in the
/// histogram's flat reverse map (the inverse of the alias lookup).
#[derive(Clone, Copy)]
pub(crate) struct AnsEncSymbolInfo {
    pub(crate) freq: u16,
    divider: FastDivU16,
    reverse_offset: u16,
}

/// Build the alias table for `freqs` and derive per-symbol encoder info.
/// `freqs` must sum to ANS_TAB_SIZE and `reverse_map` must have exactly that
/// many entries. The caller owns the storage so tables for every histogram can
/// share one allocation without placing an 8 KiB temporary on the stack.
pub(crate) fn build_symbol_info(freqs: &[u16], reverse_map: &mut [u16]) -> Vec<AnsEncSymbolInfo> {
    assert_eq!(reverse_map.len(), ANS_TAB_SIZE as usize);
    reverse_map.fill(0);
    let alias = init_alias_table(freqs);
    let mut reverse_offset = 0u16;
    let symbols: Vec<AnsEncSymbolInfo> = freqs
        .iter()
        .map(|&freq| {
            let info = AnsEncSymbolInfo {
                freq,
                divider: FastDivU16::new_or_one(freq),
                reverse_offset,
            };
            reverse_offset += freq;
            info
        })
        .collect();
    debug_assert_eq!(u32::from(reverse_offset), ANS_TAB_SIZE);

    for slot in 0..ANS_TAB_SIZE {
        let s = alias_lookup(&alias, slot);
        if s.value < symbols.len() && s.offset < symbols[s.value].freq as usize {
            let offset = symbols[s.value].reverse_offset as usize + s.offset;
            reverse_map[offset] = slot as u16;
        }
    }
    symbols
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
    pub(crate) fn put_symbol(
        &mut self,
        info: &AnsEncSymbolInfo,
        reverse_map: &[u16],
    ) -> Option<u16> {
        let freq = info.freq as u32;
        debug_assert!(freq > 0, "ANS symbol with zero frequency");
        let reverse_offset = info.reverse_offset as usize;
        debug_assert!(reverse_offset + freq as usize <= reverse_map.len());

        let mut state = self.state;
        let mut emitted = None;
        if (state >> (32 - ANS_LOG_TAB_SIZE)) >= freq {
            emitted = Some((state & 0xffff) as u16);
            state >>= 16;
        }

        let (q, rem) = info.divider.div_rem_fast(state, freq);
        let mapped = reverse_map[reverse_offset + rem as usize] as u32;
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
    reverse_maps: &[u16],
    hybrid_uint_configs: &[super::token::HybridUintConfig],
    w: &mut BitWriter,
) {
    let mut prepared = Vec::with_capacity(tokens.len());
    for t in tokens {
        let hist = context_map[t.context as usize];
        // Must honor the per-cluster config, exactly as the prefix writer
        // does. Using the default here silently desynchronizes the decoder for
        // any code that selects a non-default configuration.
        let (sym, nbits, bits) =
            super::token::uint_encode_with_config(t.value, hybrid_uint_configs[hist as usize]);
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
        let reverse_start = hist * ANS_TAB_SIZE as usize;
        let reverse_map = &reverse_maps[reverse_start..reverse_start + ANS_TAB_SIZE as usize];
        let info = &symbol_info[hist][sym];
        if let Some(word) = coder.put_symbol(info, reverse_map) {
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
// Histogram selection and serialization (libjxl enc_ans.cc Encode).
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

pub(crate) trait HistogramWriter {
    fn write(&mut self, n_bits: usize, bits: u64);
}

impl HistogramWriter for BitWriter {
    #[inline]
    fn write(&mut self, n_bits: usize, bits: u64) {
        BitWriter::write(self, n_bits, bits);
    }
}

#[derive(Default)]
struct HistogramBitCounter(usize);

impl HistogramWriter for HistogramBitCounter {
    #[inline]
    fn write(&mut self, n_bits: usize, _bits: u64) {
        self.0 += n_bits;
    }
}

fn store_varlen_u8<W: HistogramWriter>(n: u32, w: &mut W) {
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

/// Serialize one selected normalized histogram. `_log_alpha_size` is accepted
/// for API symmetry; the body trims to the true alphabet size.
pub(crate) fn encode_histogram<W: HistogramWriter>(
    histogram: &AnsHistogram,
    _log_alpha_size: u32,
    w: &mut W,
) {
    let freqs = &histogram.freqs;
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

    if histogram.method == 0 {
        w.write(1, 0); // non-small
        w.write(1, 1); // uniform
        debug_assert!(alphabet_size > 0);
        store_varlen_u8((alphabet_size - 1) as u32, w);
        return;
    }

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
    let method = histogram.method as u32;
    let upper_bound_log = floor_log2(ANS_LOG_TAB_SIZE + 1); // 3
    let log = floor_log2(method); // 3
    w.write(log as usize, ((1u32 << log) - 1) as u64);
    if log != upper_bound_log {
        w.write(1, 0);
    }
    w.write(log as usize, (((1u32 << log) - 1) & method) as u64);

    store_varlen_u8((alphabet_size - 3) as u32, w);

    let omit_pos = histogram.omit_pos as usize;
    debug_assert!(omit_pos < alphabet_size);

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

    // Runs are split around the omitted count because it may use an artificial
    // bit width. Only runs of five or more save space with the static code.
    const MIN_REPS: usize = 5;
    const RLE_SYMBOL: usize = ANS_LOG_TAB_SIZE as usize + 1;
    let mut same = vec![0u8; alphabet_size];
    let mut last = 0usize;
    for i in 1..=alphabet_size {
        if i == alphabet_size || i == omit_pos || i == omit_pos + 1 || counts[i] != counts[last] {
            same[last] = (i - last) as u8;
            last = i;
        }
    }

    let mut i = 0usize;
    while i < alphabet_size {
        let bw = bit_width[i] as usize;
        w.write(
            K_BIT_WIDTH_LENGTHS[bw] as usize,
            K_BIT_WIDTH_SYMBOLS[bw] as u64,
        );
        let run = same[i] as usize;
        if run >= MIN_REPS {
            w.write(
                K_BIT_WIDTH_LENGTHS[RLE_SYMBOL] as usize,
                K_BIT_WIDTH_SYMBOLS[RLE_SYMBOL] as u64,
            );
            store_varlen_u8((run - MIN_REPS) as u32, w);
            i += run;
        } else {
            i += 1;
        }
    }

    // Mantissa bits.
    let shift: i32 = (method - 1) as i32;
    if shift != 0 {
        let mut i = 0usize;
        while i < alphabet_size {
            let bit_width = bit_width[i];
            let count = counts[i];
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
            let run = same[i] as usize;
            i += if run >= MIN_REPS { run } else { 1 };
        }
    }
}

fn flat_histogram(len: usize, storage_len: usize) -> AnsHistogram {
    let base = ANS_TAB_SIZE as usize / len;
    let remainder = ANS_TAB_SIZE as usize % len;
    let mut freqs = vec![0u16; storage_len];
    for (i, freq) in freqs[..len].iter_mut().enumerate() {
        *freq = (base + usize::from(i < remainder)) as u16;
    }
    AnsHistogram {
        freqs,
        method: 0,
        omit_pos: 0,
        cost: 0.0,
    }
}

fn histogram_cost(counts: &[u32], histogram: &AnsHistogram) -> f64 {
    ans_data_bits(counts, &histogram.freqs) + ans_table_bits(histogram)
}

/// Select the flat, small, or count-precision representation minimizing the
/// exact serialized header bits plus the modeled ANS payload bits.
pub(crate) fn optimize_ans_histogram(counts: &[u32]) -> AnsHistogram {
    let mut alphabet_size = counts.len();
    while alphabet_size > 0 && counts[alphabet_size - 1] == 0 {
        alphabet_size -= 1;
    }
    if alphabet_size == 0 {
        return AnsHistogram {
            freqs: vec![0; counts.len()],
            method: 1,
            omit_pos: 0,
            cost: 3.0,
        };
    }

    let mut best = flat_histogram(alphabet_size, counts.len());
    let mut best_cost = histogram_cost(counts, &best);
    best.cost = best_cost;
    let num_symbols = counts[..alphabet_size].iter().filter(|&&v| v != 0).count();
    if num_symbols == 1 {
        let symbol = counts[..alphabet_size]
            .iter()
            .position(|&v| v != 0)
            .unwrap();
        let mut freqs = vec![0; counts.len()];
        freqs[symbol] = ANS_TAB_SIZE as u16;
        let mut histogram = AnsHistogram {
            freqs,
            method: 1,
            omit_pos: symbol as u8,
            cost: 0.0,
        };
        histogram.cost = histogram_cost(counts, &histogram);
        return histogram;
    }

    // Small trees always store their one free population count at full
    // precision, so trying lower shifts would only change payload cost.
    if num_symbols == 2 {
        let mut freqs = Vec::new();
        normalize_counts(counts, &mut freqs);
        let omit_pos = freqs
            .iter()
            .enumerate()
            .max_by_key(|&(_, f)| f)
            .map_or(0, |(i, _)| i);
        let precise = AnsHistogram {
            freqs,
            method: 12,
            omit_pos: omit_pos as u8,
            cost: 0.0,
        };
        let precise_cost = histogram_cost(counts, &precise);
        if precise_cost < best_cost {
            let mut precise = precise;
            precise.cost = precise_cost;
            return precise;
        }
        return best;
    }

    // Full precision needs no constrained rebalance; retain the existing fast
    // normalization and additionally test libjxl's minimum/midpoint probes.
    let mut precise_freqs = Vec::new();
    normalize_counts(counts, &mut precise_freqs);
    let mut precise_omit = 0usize;
    for i in 1..alphabet_size {
        if precise_freqs[i] > precise_freqs[precise_omit] {
            precise_omit = i;
        }
    }
    let precise = AnsHistogram {
        freqs: precise_freqs,
        method: 12,
        omit_pos: precise_omit as u8,
        cost: 0.0,
    };
    let precise_cost = histogram_cost(counts, &precise);
    if precise_cost < best_cost {
        best = precise;
        best_cost = precise_cost;
        best.cost = best_cost;
    }

    // These are the remaining libjxl fast-strategy probes. Together with the
    // exact full-precision table above they capture the useful header/data
    // trade-off without making tiny images pay for twelve greedy rebalances.
    for shift in [0usize, ANS_LOG_TAB_SIZE as usize / 2] {
        if let Some(candidate) = rebalance_counts(counts, alphabet_size, shift) {
            let cost = histogram_cost(counts, &candidate);
            if cost < best_cost {
                best = candidate;
                best_cost = cost;
                best.cost = best_cost;
            }
        }
    }
    best
}

pub(crate) fn ans_data_bits(counts: &[u32], freqs: &[u16]) -> f64 {
    let mut bits = 0.0f64;
    let logs = count_log2();
    for (&freqs, &count) in freqs.iter().zip(counts.iter()) {
        if count == 0 {
            continue;
        }
        let f = freqs.max(1) as usize;
        bits = f_fmla(count as f64, ANS_LOG_TAB_SIZE as f64 - logs[f], bits);
    }
    bits
}

pub(crate) fn huffman_data_bits(counts: &[u32], depths: &[u8]) -> f64 {
    let mut bits = 0.0f64;
    for (&count, &depth) in counts.iter().zip(depths[..counts.len()].iter()) {
        bits = f_fmla(count as f64, depth as f64, bits);
    }
    bits
}

/// Exact table-overhead measurement: serialize the real histogram into a
/// throwaway writer and count the bits.
pub(crate) fn ans_table_bits(histogram: &AnsHistogram) -> f64 {
    let mut w = HistogramBitCounter::default();
    encode_histogram(histogram, LOG_ALPHA_SIZE as u32, &mut w);
    w.0 as f64
}

pub(crate) fn huffman_tree_bits_estimate(depths: &[u8]) -> f64 {
    let used = depths.iter().filter(|&&d| d != 0).count();
    f_fmla(used as f64, 4.0, 8.0)
}

/// Decide prefix vs rANS for a clustered-histogram bundle. true => prefix.
/// Picks the smaller total encoding (data + table); tie -> prefix (faster).
pub(crate) fn choose_use_prefix_code(
    histograms: &[Histogram],
    ans_histograms: &[AnsHistogram],
    huffman_depths: &[[u8; super::prefix_code::ALPHABET_SIZE]],
) -> bool {
    let mut ans_total = 0.0f64;
    let mut huff_total = 0.0f64;
    for ((h, histogram), depths) in histograms
        .iter()
        .zip(ans_histograms)
        .zip(huffman_depths.iter())
    {
        ans_total += histogram.cost;
        huff_total += huffman_data_bits(&h.counts, depths) + huffman_tree_bits_estimate(depths);
    }
    huff_total <= ans_total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_flat_reverse_map(freqs: &[u16]) {
        let alias = init_alias_table(freqs);
        let mut reverse_map = vec![0u16; ANS_TAB_SIZE as usize];
        let info = build_symbol_info(freqs, &mut reverse_map);
        let mut legacy: Vec<Vec<u16>> = freqs
            .iter()
            .map(|&freq| vec![0u16; freq as usize])
            .collect();
        for slot in 0..ANS_TAB_SIZE {
            let decoded = alias_lookup(&alias, slot);
            legacy[decoded.value][decoded.offset] = slot as u16;
        }
        assert_eq!(reverse_map.len(), ANS_TAB_SIZE as usize);
        assert_eq!(info.len(), freqs.len());

        let mut expected_offset = 0usize;
        for (symbol, (&freq, symbol_info)) in freqs.iter().zip(&info).enumerate() {
            assert_eq!(symbol_info.freq, freq);
            assert_eq!(symbol_info.reverse_offset as usize, expected_offset);
            assert_eq!(
                &reverse_map[expected_offset..expected_offset + freq as usize],
                legacy[symbol]
            );
            for remainder in 0..freq as usize {
                let slot = reverse_map[expected_offset + remainder] as u32;
                let decoded = alias_lookup(&alias, slot);
                assert_eq!(decoded.value, symbol);
                assert_eq!(decoded.offset, remainder);
            }
            expected_offset += freq as usize;
        }
        assert_eq!(expected_offset, ANS_TAB_SIZE as usize);
    }

    #[test]
    fn flat_reverse_map_inverts_alias_lookup() {
        // Keep symbol metadata small: the 4096-entry reverse table belongs to
        // the shared flat storage, never in a per-symbol or stack-local object.
        assert!(core::mem::size_of::<AnsEncSymbolInfo>() <= 16);

        let mut single = [0u16; TABLE_ENTRIES];
        single[37] = ANS_TAB_SIZE as u16;
        assert_flat_reverse_map(&single);

        let counts: [u32; TABLE_ENTRIES] =
            core::array::from_fn(|symbol| ((symbol * 29 + symbol * symbol * 7 + 11) % 251) as u32);
        let mut normalized = Vec::new();
        normalize_counts(&counts, &mut normalized);
        assert_flat_reverse_map(&normalized);
    }

    #[test]
    fn symbol_info_builds_on_a_small_stack() {
        std::thread::Builder::new()
            .name("ans-small-stack".into())
            .stack_size(64 * 1024)
            .spawn(|| {
                let counts: [u32; TABLE_ENTRIES] =
                    core::array::from_fn(|symbol| (symbol * 37 % 257 + 1) as u32);
                let mut freqs = Vec::new();
                normalize_counts(&counts, &mut freqs);
                let mut reverse_map = vec![0u16; ANS_TAB_SIZE as usize];
                let symbols = build_symbol_info(&freqs, &mut reverse_map);
                assert_eq!(symbols.len(), TABLE_ENTRIES);
            })
            .expect("failed to spawn ANS stack test")
            .join()
            .expect("ANS table construction overflowed its test stack");
    }

    #[test]
    fn optimized_histograms_are_valid_and_representable() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for case in 0..2_000 {
            let len = case % TABLE_ENTRIES + 1;
            let mut counts = vec![0u32; TABLE_ENTRIES];
            for count in &mut counts[..len] {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *count = ((state >> 33) as u32) % 10_000;
            }
            // Keep the ANS builder's non-empty precondition while still
            // exercising sparse and long equal-count runs.
            counts[case % len] = counts[case % len].max(1);
            let histogram = optimize_ans_histogram(&counts);
            assert_eq!(
                histogram.freqs.iter().map(|&v| v as u32).sum::<u32>(),
                ANS_TAB_SIZE
            );
            assert!(histogram.method <= 12);
            if histogram.method == 0 {
                let expected = flat_histogram(len, counts.len());
                assert_eq!(histogram.freqs, expected.freqs);
                continue;
            }
            for (i, (&source, &freq)) in counts.iter().zip(&histogram.freqs).enumerate() {
                assert_eq!(source == 0, freq == 0);
                if freq != 0 && i != histogram.omit_pos as usize {
                    let log = floor_log2(freq as u32) as i32;
                    let precision = get_pop_count_precision(log, histogram.method as i32 - 1);
                    let dropped = log - precision;
                    assert_eq!(freq & ((1u16 << dropped) - 1), 0);
                }
            }
            let mut writer = BitWriter::new();
            encode_histogram(&histogram, LOG_ALPHA_SIZE as u32, &mut writer);
            assert!(writer.bits_written() != 0);
        }
    }

    #[test]
    fn flat_and_rle_forms_reduce_header_bits() {
        let flat = flat_histogram(128, 128);
        assert_eq!(ans_table_bits(&flat), 12.0);

        let mut freqs = vec![16u16; 128];
        freqs[0] += 2048;
        let histogram = AnsHistogram {
            freqs,
            method: 12,
            omit_pos: 0,
            cost: 0.0,
        };
        // Without RLE, 127 identical width-5 counts alone need 508 bits.
        assert!(ans_table_bits(&histogram) < 100.0);
    }
}
