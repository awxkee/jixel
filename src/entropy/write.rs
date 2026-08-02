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
    ANS_TAB_SIZE, AnsEncSymbolInfo, AnsHistogram, AnsTokenCodeRef, ans_tokens_bits_pair,
    build_symbol_info, choose_use_prefix_code, encode_histogram, fast_ans_population_cost,
    normalize_counts_into, optimize_ans_histogram,
};
use super::cluster::cluster_histograms;
use super::entropy_code::{EntropyCode, OwnedEntropyCode};
use super::histogram::Histogram;
use super::huffman_tree::{HuffmanNode, create_huffman_tree};
use super::prefix_code::{ALPHABET_SIZE, PrefixCode, convert_bit_depths_to_symbols};
use super::token::{HybridUintConfig, Token, uint_encode, uint_encode_with_config};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::f_log2;
use crate::lz77_ac::{LZ77_MIN_LENGTH, LZ77_MIN_SYMBOL, lz77_length_encode};
use crate::thread_pool::ThreadPool;

pub(crate) const ANS_ENABLED: bool = true;

#[allow(unused)]
pub(crate) const ANS_LOG_ALPHA_SIZE: u32 = 7;

/// ANS fields accumulated while building an entropy code. The prefix-only
/// default owns no table storage.
struct AnsCodeStorage {
    use_prefix_code: bool,
    histograms: Vec<AnsHistogram>,
    pricing_freqs: Vec<u16>,
    symbols: Vec<Vec<AnsEncSymbolInfo>>,
    reverse_maps: Vec<u16>,
}

impl Default for AnsCodeStorage {
    fn default() -> Self {
        Self {
            use_prefix_code: true,
            histograms: Vec::new(),
            pricing_freqs: Vec::new(),
            symbols: Vec::new(),
            reverse_maps: Vec::new(),
        }
    }
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

const NUM_HYBRID_CANDIDATES: usize = HYBRID_CANDIDATES.len();
const DEFAULT_HYBRID_INDEX: usize = 6;

struct HybridAnsSelectorScratch {
    counts: Box<[[u32; ALPHABET_SIZE]]>,
    extra_bits: [u64; NUM_HYBRID_CANDIDATES],
    totals: [u64; NUM_HYBRID_CANDIDATES],
    valid: [bool; NUM_HYBRID_CANDIDATES],
    proxy_costs: [f64; NUM_HYBRID_CANDIDATES],
}

impl Default for HybridAnsSelectorScratch {
    fn default() -> Self {
        Self {
            counts: vec![[0; ALPHABET_SIZE]; NUM_HYBRID_CANDIDATES].into_boxed_slice(),
            extra_bits: [0; NUM_HYBRID_CANDIDATES],
            totals: [0; NUM_HYBRID_CANDIDATES],
            valid: [true; NUM_HYBRID_CANDIDATES],
            proxy_costs: [f64::INFINITY; NUM_HYBRID_CANDIDATES],
        }
    }
}

impl HybridAnsSelectorScratch {
    fn reset(&mut self) {
        for counts in &mut self.counts {
            counts.fill(0);
        }
        self.extra_bits.fill(0);
        self.totals.fill(0);
        self.valid.fill(true);
        self.proxy_costs.fill(f64::INFINITY);
    }
}

fn select_hybrid_config_ans(
    values: &[u32],
    scratch: &mut HybridAnsSelectorScratch,
) -> HybridUintConfig {
    if values.is_empty() {
        return HybridUintConfig::DEFAULT;
    }
    let stride = values.len().div_ceil(65_536).max(1);
    scratch.reset();
    // Reuse one contiguous histogram allocation across every cluster. Keeping
    // the configuration outside the value loop also lets uint encoding remain
    // a small, predictable hot loop for each candidate.
    for (candidate_index, &config) in HYBRID_CANDIDATES.iter().enumerate() {
        for &value in values.iter().step_by(stride) {
            let (symbol, nbits, _) = uint_encode_with_config(value, config);
            if symbol as usize >= ALPHABET_SIZE {
                scratch.valid[candidate_index] = false;
                break;
            }
            scratch.counts[candidate_index][symbol as usize] += 1;
            scratch.extra_bits[candidate_index] += nbits as u64;
            scratch.totals[candidate_index] += 1;
        }
    }

    for (candidate_index, &config) in HYBRID_CANDIDATES.iter().enumerate() {
        if !scratch.valid[candidate_index] || scratch.totals[candidate_index] == 0 {
            continue;
        }
        let total = scratch.totals[candidate_index];
        let mut proxy_cost = scratch.extra_bits[candidate_index] as f64;
        let mut used = 0usize;
        for &count in &scratch.counts[candidate_index] {
            if count != 0 {
                used += 1;
                proxy_cost += count as f64 * f_log2(total as f64 / count as f64);
            }
        }
        scratch.proxy_costs[candidate_index] = proxy_cost * stride as f64
            + (8 * used + 15) as f64
            + hybrid_uint_config_bits(config, 3) as f64;
    }

    // Shannon cost only nominates the strongest non-default finalist. The
    // winner and the existing 0.5% stability gate below use actual normalized
    // ANS data and exact table bits for both finalist and default.
    let proxy_best = (0..NUM_HYBRID_CANDIDATES)
        .filter(|&i| i != DEFAULT_HYBRID_INDEX && scratch.valid[i])
        .min_by(|&a, &b| {
            scratch.proxy_costs[a]
                .total_cmp(&scratch.proxy_costs[b])
                .then_with(|| a.cmp(&b))
        })
        .unwrap_or(DEFAULT_HYBRID_INDEX);
    if proxy_best == DEFAULT_HYBRID_INDEX {
        return HybridUintConfig::DEFAULT;
    }

    let mut best = HybridUintConfig::DEFAULT;
    let mut best_cost = f64::INFINITY;
    let mut default_cost = f64::INFINITY;
    for candidate_index in [DEFAULT_HYBRID_INDEX, proxy_best] {
        if !scratch.valid[candidate_index] {
            continue;
        }
        let config = HYBRID_CANDIDATES[candidate_index];
        let counts = &scratch.counts[candidate_index];
        let cost =
            hybrid_ans_candidate_cost(counts, scratch.extra_bits[candidate_index], stride, config);
        if candidate_index == DEFAULT_HYBRID_INDEX {
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

fn hybrid_ans_candidate_cost(
    counts: &[u32],
    extra_bits: u64,
    stride: usize,
    config: HybridUintConfig,
) -> f64 {
    extra_bits as f64 * stride as f64
        + fast_ans_population_cost(counts, stride)
        + hybrid_uint_config_bits(config, 3) as f64
}

#[inline]
fn hybrid_uint_config_bits(config: HybridUintConfig, split_width: usize) -> usize {
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
    split_width + msb_width as usize + lsb_width as usize
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
    let ans = AnsCodeStorage::default();
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; num_contexts],
        orig_context_map: None,
        orig_num_contexts: 0,
        use_prefix_code: ans.use_prefix_code,
        ans_histograms: ans.histograms,
        ans_pricing_freqs: ans.pricing_freqs,
        ans_symbols: ans.symbols,
        ans_reverse_maps: ans.reverse_maps,
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
    let ans = AnsCodeStorage::default();
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; histograms.len()],
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: ans.use_prefix_code,
        ans_histograms: ans.histograms,
        ans_pricing_freqs: ans.pricing_freqs,
        ans_symbols: ans.symbols,
        ans_reverse_maps: ans.reverse_maps,
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
    optimize_entropy_code_ac_streams_impl(
        streams,
        num_contexts,
        huffman_pool,
        select_configs,
        false,
    )
}

pub(crate) fn optimize_entropy_code_ac_streams_fast<'a, I>(
    streams: I,
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> OwnedEntropyCode
where
    I: IntoIterator<Item = &'a [Token]>,
{
    optimize_entropy_code_ac_streams_impl(streams, num_contexts, huffman_pool, false, true)
}

const ANS_CLUSTER_PROXY_SYMBOL_BITS: f64 = 6.0;
const ANS_CLUSTER_PROXY_BASE_BITS: f64 = 12.0;
const MAX_ANS_RELOCATION_CONTEXTS: usize = 4096;

#[inline]
fn xlog2x(value: u32) -> f64 {
    if value <= 1 {
        0.0
    } else {
        value as f64 * f_log2(value as f64)
    }
}

/// Cheap Shannon-domain delta used only to nominate an ANS cluster move. The
/// final decision is made by serializing both complete candidates below.
fn moved_population_proxy_delta(source: &Histogram, target: &Histogram, moved: &Histogram) -> f64 {
    debug_assert!(moved.total_count <= source.total_count);
    let source_after = source.total_count - moved.total_count;
    let target_after = target.total_count + moved.total_count;
    let mut delta = xlog2x(source_after) + xlog2x(target_after)
        - xlog2x(source.total_count)
        - xlog2x(target.total_count);
    if source_after == 0 {
        delta -= ANS_CLUSTER_PROXY_BASE_BITS;
    }
    if target.total_count == 0 && target_after != 0 {
        delta += ANS_CLUSTER_PROXY_BASE_BITS;
    }
    for ((&source_count, &target_count), &moved_count) in source
        .counts
        .iter()
        .zip(target.counts.iter())
        .zip(moved.counts.iter())
    {
        if moved_count == 0 {
            continue;
        }
        debug_assert!(moved_count <= source_count);
        let source_count_after = source_count - moved_count;
        let target_count_after = target_count + moved_count;
        delta -= xlog2x(source_count_after) + xlog2x(target_count_after)
            - xlog2x(source_count)
            - xlog2x(target_count);
        if source_count_after == 0 {
            delta -= ANS_CLUSTER_PROXY_SYMBOL_BITS;
        }
        if target_count == 0 {
            delta += ANS_CLUSTER_PROXY_SYMBOL_BITS;
        }
    }
    delta
}

fn move_histogram(source: &mut Histogram, target: &mut Histogram, moved: &Histogram) {
    for ((source_count, target_count), &moved_count) in source
        .counts
        .iter_mut()
        .zip(target.counts.iter_mut())
        .zip(moved.counts.iter())
    {
        debug_assert!(moved_count <= *source_count);
        *source_count -= moved_count;
        *target_count += moved_count;
    }
    source.total_count -= moved.total_count;
    target.total_count += moved.total_count;
}

fn two_histograms_mut(
    histograms: &mut [Histogram],
    first: usize,
    second: usize,
) -> (&mut Histogram, &mut Histogram) {
    debug_assert_ne!(first, second);
    if first < second {
        let (left, right) = histograms.split_at_mut(second);
        (&mut left[first], &mut right[0])
    } else {
        let (left, right) = histograms.split_at_mut(first);
        (&mut right[0], &mut left[second])
    }
}

fn compact_ans_clusters(
    histograms: &mut Vec<Histogram>,
    context_map: &mut [u8],
    configs: &mut Vec<HybridUintConfig>,
) {
    let mut remap = vec![u8::MAX; histograms.len()];
    let mut compact_histograms = Vec::with_capacity(histograms.len());
    let mut compact_configs = Vec::with_capacity(configs.len());
    for (old, histogram) in histograms.iter().enumerate() {
        if histogram.total_count != 0 {
            remap[old] = compact_histograms.len() as u8;
            compact_histograms.push(histogram.clone());
            compact_configs.push(configs[old]);
        }
    }
    debug_assert!(!compact_histograms.is_empty());
    for cluster in context_map {
        let mapped = remap[*cluster as usize];
        debug_assert_ne!(mapped, u8::MAX);
        *cluster = mapped;
    }
    *histograms = compact_histograms;
    *configs = compact_configs;
}

/// Build one bounded ANS-aware candidate: at most one original-context
/// relocation and one whole-cluster merge. Moves are restricted to equal
/// HybridUint configurations, so histogram addition is exact and no candidate
/// requires retokenizing all values under a different representation.
fn propose_ans_cluster_refinement(
    histograms: &[Histogram],
    context_histograms: &[Histogram],
    context_map: &[u8],
    configs: &[HybridUintConfig],
) -> Option<(Vec<Histogram>, Vec<u8>, Vec<HybridUintConfig>)> {
    if histograms.len() <= 1 {
        return None;
    }

    let mut populated_contexts: Vec<usize> = context_histograms
        .iter()
        .enumerate()
        .filter_map(|(context, histogram)| (histogram.total_count != 0).then_some(context))
        .collect();
    if populated_contexts.len() > MAX_ANS_RELOCATION_CONTEXTS {
        populated_contexts.sort_unstable_by_key(|&context| {
            std::cmp::Reverse(context_histograms[context].total_count)
        });
        populated_contexts.truncate(MAX_ANS_RELOCATION_CONTEXTS);
    }

    let mut best_relocation = None;
    let mut best_delta = -0.25f64;
    for context in populated_contexts {
        let source = context_map[context] as usize;
        let moved = &context_histograms[context];
        // Emptying a cluster also requires deciding where its tokenless
        // contexts go. Whole-cluster merging handles that case cleanly.
        if moved.total_count == histograms[source].total_count {
            continue;
        }
        for target in 0..histograms.len() {
            if target == source || configs[target] != configs[source] {
                continue;
            }
            let delta =
                moved_population_proxy_delta(&histograms[source], &histograms[target], moved);
            if delta < best_delta {
                best_delta = delta;
                best_relocation = Some((context, source, target));
            }
        }
    }

    let mut candidate_histograms = histograms.to_vec();
    let mut candidate_map = context_map.to_vec();
    let mut candidate_configs = configs.to_vec();
    let mut changed = false;
    if let Some((context, source, target)) = best_relocation {
        let moved = &context_histograms[context];
        let (source_histogram, target_histogram) =
            two_histograms_mut(&mut candidate_histograms, source, target);
        move_histogram(source_histogram, target_histogram, moved);
        candidate_map[context] = target as u8;
        changed = true;
    }

    let mut best_merge = None;
    let mut best_delta = -0.25f64;
    for source in 0..candidate_histograms.len() {
        for target in 0..source {
            if candidate_configs[target] != candidate_configs[source] {
                continue;
            }
            let delta = moved_population_proxy_delta(
                &candidate_histograms[source],
                &candidate_histograms[target],
                &candidate_histograms[source],
            );
            if delta < best_delta {
                best_delta = delta;
                best_merge = Some((source, target));
            }
        }
    }
    if let Some((source, target)) = best_merge {
        let moved = candidate_histograms[source].clone();
        let (source_histogram, target_histogram) =
            two_histograms_mut(&mut candidate_histograms, source, target);
        move_histogram(source_histogram, target_histogram, &moved);
        for cluster in &mut candidate_map {
            if *cluster as usize == source {
                *cluster = target as u8;
            }
        }
        changed = true;
    }

    if !changed {
        return None;
    }
    compact_ans_clusters(
        &mut candidate_histograms,
        &mut candidate_map,
        &mut candidate_configs,
    );
    Some((candidate_histograms, candidate_map, candidate_configs))
}

fn build_ans_storage_from_selected(
    histograms: &[Histogram],
    selected_histograms: Vec<AnsHistogram>,
) -> AnsCodeStorage {
    let mut ans = AnsCodeStorage {
        use_prefix_code: false,
        histograms: selected_histograms,
        pricing_freqs: vec![0; histograms.len() * ALPHABET_SIZE],
        symbols: Vec::with_capacity(histograms.len()),
        reverse_maps: vec![0; histograms.len() * ANS_TAB_SIZE as usize],
    };
    let (pricing_tables, remainder) = ans.pricing_freqs.as_chunks_mut::<ALPHABET_SIZE>();
    debug_assert!(remainder.is_empty());
    for (source, pricing) in histograms.iter().zip(pricing_tables) {
        normalize_counts_into(&source.counts, pricing);
    }
    for (histogram_index, histogram) in ans.histograms.iter().enumerate() {
        let reverse_start = histogram_index * ANS_TAB_SIZE as usize;
        ans.symbols.push(build_symbol_info(
            &histogram.freqs,
            &mut ans.reverse_maps[reverse_start..reverse_start + ANS_TAB_SIZE as usize],
        ));
    }
    ans
}

fn build_ans_storage(histograms: &[Histogram]) -> AnsCodeStorage {
    let selected_histograms = histograms
        .iter()
        .map(|histogram| optimize_ans_histogram(&histogram.counts))
        .collect();
    build_ans_storage_from_selected(histograms, selected_histograms)
}

struct AnsBundleRef<'a> {
    context_map: &'a [u8],
    configs: &'a [HybridUintConfig],
    histograms: &'a [AnsHistogram],
    symbols: &'a [Vec<AnsEncSymbolInfo>],
    reverse_maps: &'a [u16],
}

impl AnsBundleRef<'_> {
    fn entropy_code(&self, num_contexts: usize) -> EntropyCode<'_> {
        EntropyCode {
            context_map: self.context_map,
            num_contexts,
            prefix_codes: &[],
            hybrid_uint_configs: self.configs,
            num_prefix_codes: 0,
            orig_context_map: None,
            orig_num_contexts: num_contexts,
            use_prefix_code: false,
            ans_histograms: self.histograms,
            ans_symbols: self.symbols,
            ans_reverse_maps: self.reverse_maps,
        }
    }

    fn token_code(&self) -> AnsTokenCodeRef<'_> {
        AnsTokenCodeRef {
            context_map: self.context_map,
            symbol_info: self.symbols,
            reverse_maps: self.reverse_maps,
            hybrid_uint_configs: self.configs,
        }
    }
}

fn exact_ans_bundle_bits_pair(
    streams: &[&[Token]],
    num_contexts: usize,
    first: &AnsBundleRef<'_>,
    second: &AnsBundleRef<'_>,
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (usize, usize) {
    let mut first_header = BitWriter::new();
    write_entropy_code(
        &first.entropy_code(num_contexts),
        &mut scratch.huffman_pool,
        &mut first_header,
    );
    let mut second_header = BitWriter::new();
    write_entropy_code(
        &second.entropy_code(num_contexts),
        &mut scratch.huffman_pool,
        &mut second_header,
    );
    let first_tokens = first.token_code();
    let second_tokens = second.token_code();
    let (first_data, second_data) =
        exact_ans_stream_bits_pair(streams, first_tokens, second_tokens, thread_pool, scratch);
    (
        first_header.bits_written() + first_data,
        second_header.bits_written() + second_data,
    )
}

fn exact_ans_stream_bits_pair(
    streams: &[&[Token]],
    first: AnsTokenCodeRef<'_>,
    second: AnsTokenCodeRef<'_>,
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (usize, usize) {
    let sequential = |streams: &[&[Token]]| {
        streams.iter().fold((0usize, 0usize), |acc, tokens| {
            let bits = ans_tokens_bits_pair(tokens, &first, &second);
            (acc.0 + bits.0, acc.1 + bits.1)
        })
    };

    const MIN_PARALLEL_TOKENS: usize = 16_384;
    const MAX_EXACT_LANES: usize = 8;
    let total_tokens: usize = streams.iter().map(|stream| stream.len()).sum();
    let num_lanes = thread_pool
        .num_threads()
        .min(MAX_EXACT_LANES)
        .min(streams.len());
    if num_lanes <= 1 || total_tokens < MIN_PARALLEL_TOKENS {
        return sequential(streams);
    }

    let chunk_len = streams.len().div_ceil(num_lanes);
    let num_chunks = streams.len().div_ceil(chunk_len);
    thread_pool
        .steal_map_with_threads(scratch, num_chunks, num_lanes, |lane, _scratch| {
            let start = lane * chunk_len;
            let end = (start + chunk_len).min(streams.len());
            sequential(&streams[start..end])
        })
        .into_iter()
        .fold((0usize, 0usize), |acc, bits| {
            (acc.0 + bits.0, acc.1 + bits.1)
        })
}

/// Final ANS-aware clustering pass for an already-selected bundle. Keeping it
/// separate from initial code construction lets callers run it only after
/// higher-level coding-arm decisions, rather than paying to refine candidates
/// that are immediately discarded.
pub(crate) fn refine_ans_clusters<'a, I>(
    code: &mut OwnedEntropyCode,
    streams: I,
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> bool
where
    I: IntoIterator<Item = &'a [Token]>,
{
    if code.use_prefix_code || code.hybrid_uint_configs.len() <= 1 {
        return false;
    }
    let streams: Vec<&[Token]> = streams.into_iter().collect();
    let mut histograms = vec![Histogram::new(); code.hybrid_uint_configs.len()];
    let mut context_histograms = vec![Histogram::new(); code.context_map.len()];
    for tokens in &streams {
        for token in *tokens {
            let context = token.context as usize;
            let cluster = code.context_map[context] as usize;
            let (symbol, _, _) =
                uint_encode_with_config(token.value, code.hybrid_uint_configs[cluster]);
            histograms[cluster].add(symbol);
            context_histograms[context].add(symbol);
        }
    }
    let Some((candidate_histograms, candidate_map, candidate_configs)) =
        propose_ans_cluster_refinement(
            &histograms,
            &context_histograms,
            &code.context_map,
            &code.hybrid_uint_configs,
        )
    else {
        return false;
    };
    let candidate_ans = build_ans_storage(&candidate_histograms);
    let original = AnsBundleRef {
        context_map: &code.context_map,
        configs: &code.hybrid_uint_configs,
        histograms: &code.ans_histograms,
        symbols: &code.ans_symbols,
        reverse_maps: &code.ans_reverse_maps,
    };
    let candidate = AnsBundleRef {
        context_map: &candidate_map,
        configs: &candidate_configs,
        histograms: &candidate_ans.histograms,
        symbols: &candidate_ans.symbols,
        reverse_maps: &candidate_ans.reverse_maps,
    };
    let (original_bits, candidate_bits) = exact_ans_bundle_bits_pair(
        &streams,
        code.orig_num_contexts,
        &original,
        &candidate,
        thread_pool,
        scratch,
    );
    if candidate_bits >= original_bits {
        return false;
    }

    code.context_map = candidate_map;
    code.prefix_codes = build_huffman_codes(&candidate_histograms, &mut scratch.huffman_pool);
    code.hybrid_uint_configs = candidate_configs;
    code.ans_histograms = candidate_ans.histograms;
    code.ans_pricing_freqs = candidate_ans.pricing_freqs;
    code.ans_symbols = candidate_ans.symbols;
    code.ans_reverse_maps = candidate_ans.reverse_maps;
    true
}

fn optimize_entropy_code_ac_streams_impl<'a, I>(
    streams: I,
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
    select_configs: bool,
    fast_cluster: bool,
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
    if fast_cluster {
        const COARSE_CLUSTERS: usize = 64;
        const NZ_CONTEXTS: usize = crate::ac_context::K_NON_ZERO_BUCKETS;
        let mut coarse = vec![Histogram::new(); COARSE_CLUSTERS];
        let mut coarse_map = vec![0usize; num_contexts];
        for (context, histogram) in histograms.iter().enumerate() {
            let bucket = if context < NZ_CONTEXTS {
                context * (COARSE_CLUSTERS / 2) / NZ_CONTEXTS
            } else {
                let z_contexts = num_contexts - NZ_CONTEXTS;
                COARSE_CLUSTERS / 2
                    + (context - NZ_CONTEXTS) * (COARSE_CLUSTERS / 2) / z_contexts.max(1)
            };
            coarse_map[context] = bucket.min(COARSE_CLUSTERS - 1);
            let dst = &mut coarse[coarse_map[context]];
            for (out, &count) in dst.counts.iter_mut().zip(histogram.counts.iter()) {
                *out += count;
            }
            dst.total_count += histogram.total_count;
        }
        let mut dense = [u8::MAX; COARSE_CLUSTERS];
        histograms.clear();
        for (bucket, histogram) in coarse.into_iter().enumerate() {
            if histogram.total_count != 0 {
                dense[bucket] = histograms.len() as u8;
                histograms.push(histogram);
            }
        }
        if histograms.is_empty() {
            histograms.push(Histogram::new());
        }
        context_map.extend(coarse_map.into_iter().map(|bucket| {
            if dense[bucket] != u8::MAX {
                dense[bucket]
            } else {
                0
            }
        }));
    } else {
        cluster_histograms(&mut histograms, &mut context_map, huffman_pool);
    }

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
        let mut hybrid_selector_scratch = HybridAnsSelectorScratch::default();
        cluster_values
            .iter()
            .map(|values| select_hybrid_config_ans(values, &mut hybrid_selector_scratch))
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

    let mut ans = AnsCodeStorage::default();
    if ANS_ENABLED {
        let depths: Vec<[u8; ALPHABET_SIZE]> = prefix_codes.iter().map(|c| c.depths).collect();
        let selected_histograms: Vec<AnsHistogram> = histograms
            .iter()
            .map(|h| optimize_ans_histogram(&h.counts))
            .collect();
        ans.use_prefix_code = choose_use_prefix_code(&histograms, &selected_histograms, &depths);
        if !ans.use_prefix_code {
            ans = build_ans_storage_from_selected(&histograms, selected_histograms);
        }
    }

    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs,
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: ans.use_prefix_code,
        ans_histograms: ans.histograms,
        ans_pricing_freqs: ans.pricing_freqs,
        ans_symbols: ans.symbols,
        ans_reverse_maps: ans.reverse_maps,
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
    let ans = AnsCodeStorage::default();
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![HybridUintConfig::DEFAULT; num_contexts],
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: ans.use_prefix_code,
        ans_histograms: ans.histograms,
        ans_pricing_freqs: ans.pricing_freqs,
        ans_symbols: ans.symbols,
        ans_reverse_maps: ans.reverse_maps,
    }
}

// ---------------------------------------------------------------------------
// Brotli-style Huffman-tree serialization.
// ---------------------------------------------------------------------------

const NUM_CODE_LENGTH_CODES: usize = 18;

static STORAGE_ORDER: [u8; NUM_CODE_LENGTH_CODES] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

static HUFFMAN_BIT_LENGTH_HUFFMAN_CODE_SYMBOLS: [u8; 6] = [0, 7, 3, 2, 1, 15];
static HUFFMAN_BIT_LENGTH_HUFFMAN_CODE_BITLENS: [u8; 6] = [2, 4, 3, 2, 2, 4];

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
        ans_histograms: Vec::new(),
        ans_pricing_freqs: Vec::new(),
        ans_symbols: Vec::new(),
        ans_reverse_maps: Vec::new(),
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
    debug_assert_eq!(code.hybrid_uint_configs.len(), code.ans_histograms.len());
    for &config in code
        .hybrid_uint_configs
        .iter()
        .take(code.ans_histograms.len())
    {
        write_uint_config(config, 3, w);
    }
    // The normalized distributions, in clustered-histogram order.
    for histogram in code.ans_histograms.iter() {
        encode_histogram(histogram, ANS_LOG_ALPHA_SIZE, w);
    }
}

#[cfg(test)]
mod context_map_tests {
    use super::*;
    use crate::lz77_ac::{LZ77_MIN_LENGTH, lz77_length_encode};

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

    fn exhaustive_selector(values: &[u32]) -> HybridUintConfig {
        if values.is_empty() {
            return HybridUintConfig::DEFAULT;
        }
        let stride = values.len().div_ceil(65_536).max(1);
        let mut best = HybridUintConfig::DEFAULT;
        let mut best_cost = f64::INFINITY;
        let mut default_cost = f64::INFINITY;
        for config in HYBRID_CANDIDATES {
            let mut counts = [0u32; ALPHABET_SIZE];
            let mut extra_bits = 0u64;
            let mut valid = true;
            for &value in values.iter().step_by(stride) {
                let (symbol, nbits, _) = uint_encode_with_config(value, config);
                if symbol as usize >= ALPHABET_SIZE {
                    valid = false;
                    break;
                }
                counts[symbol as usize] += 1;
                extra_bits += nbits as u64;
            }
            if !valid {
                continue;
            }
            let cost = hybrid_ans_candidate_cost(&counts, extra_bits, stride, config);
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

    #[test]
    fn config_bit_cost_matches_serializer() {
        for config in HYBRID_CANDIDATES {
            for split_width in [3, 4] {
                let mut writer = BitWriter::new();
                write_uint_config(config, split_width, &mut writer);
                assert_eq!(
                    writer.bits_written(),
                    hybrid_uint_config_bits(config, split_width)
                );
            }
        }
    }

    #[test]
    fn shortlist_matches_exhaustive_exact_selection() {
        let mut scratch = HybridAnsSelectorScratch::default();
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for case in 0..48usize {
            let len = 97 + case * 83;
            let mut values = Vec::with_capacity(len);
            for i in 0..len {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let random = (state >> 32) as u32;
                let value = match case % 6 {
                    0 => random & 15,
                    1 => random & 255,
                    2 => random % 4096,
                    3 => (random & 63) * (1 + (i % 11) as u32),
                    4 => u32::from(i % 19 != 0) * (random & 7),
                    _ => random % 65_536,
                };
                values.push(value);
            }
            assert_eq!(
                select_hybrid_config_ans(&values, &mut scratch),
                exhaustive_selector(&values),
                "hybrid shortlist diverged on generated case {case}"
            );
        }
    }
}
