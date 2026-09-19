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
    normalize_counts_into, optimize_ans_histogram, refine_ans_histogram_precision,
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

static HYBRID_CANDIDATES: [HybridUintConfig; 12] = [
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

fn select_hybrid_config_ans_sampled(
    values: &[u32],
    sample_stride: usize,
    stride: usize,
    acceptance_ratio: f64,
    scratch: &mut HybridAnsSelectorScratch,
) -> HybridUintConfig {
    if values.is_empty() {
        return HybridUintConfig::DEFAULT;
    }
    scratch.reset();
    // Reuse one contiguous histogram allocation across every cluster. Keeping
    // the configuration outside the value loop also lets uint encoding remain
    // a small, predictable hot loop for each candidate.
    for (candidate_index, &config) in HYBRID_CANDIDATES.iter().enumerate() {
        for &value in values.iter().step_by(sample_stride) {
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
    // winner and the preset's acceptance margin below use actual normalized
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
    if best_cost >= default_cost * acceptance_ratio {
        HybridUintConfig::DEFAULT
    } else {
        best
    }
}

/// Retains exactly the old per-cluster sampling ordinals, bounded to 65536
/// values per cluster. The original stride still scales the estimated cost.
pub(crate) struct HybridUintSamples {
    values: Vec<Vec<u32>>,
    strides: Vec<usize>,
    remaining: Vec<usize>,
}

impl HybridUintSamples {
    pub(crate) fn new(counts: &[usize]) -> Self {
        let strides: Vec<_> = counts.iter().map(|&n| n.div_ceil(65_536).max(1)).collect();
        let values = counts
            .iter()
            .zip(&strides)
            .map(|(&n, &stride)| Vec::with_capacity(n.div_ceil(stride)))
            .collect();
        Self {
            values,
            strides,
            remaining: vec![0; counts.len()],
        }
    }

    /// Samples gathered elsewhere with the same per-cluster ordinal rule
    /// (`ordinal % stride == 0`), e.g. by several streams in parallel.
    pub(crate) fn from_parts(values: Vec<Vec<u32>>, strides: Vec<usize>) -> Self {
        let remaining = vec![0; values.len()];
        Self {
            values,
            strides,
            remaining,
        }
    }

    pub(crate) fn strides_for(counts: &[usize]) -> Vec<usize> {
        counts.iter().map(|&n| n.div_ceil(65_536).max(1)).collect()
    }

    #[inline]
    pub(crate) fn push(&mut self, cluster: usize, value: u32) {
        if self.remaining[cluster] == 0 {
            self.values[cluster].push(value);
            self.remaining[cluster] = self.strides[cluster] - 1;
        } else {
            self.remaining[cluster] -= 1;
        }
    }

    pub(crate) fn select(&self) -> Vec<HybridUintConfig> {
        self.select_with_acceptance(0.995)
    }

    fn select_with_acceptance(&self, acceptance_ratio: f64) -> Vec<HybridUintConfig> {
        // Every cluster is selected independently; spread them over scoped
        // threads with one selector scratch each (identical results).
        let n = self.values.len();
        let threads = std::thread::available_parallelism()
            .map_or(1, |t| t.get())
            .min(n.max(1));
        if threads <= 1 || n < 4 {
            let mut scratch = HybridAnsSelectorScratch::default();
            return self
                .values
                .iter()
                .zip(&self.strides)
                .map(|(values, &stride)| {
                    select_hybrid_config_ans_sampled(
                        values,
                        1,
                        stride,
                        acceptance_ratio,
                        &mut scratch,
                    )
                })
                .collect();
        }
        let chunk = n.div_ceil(threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n)
                .step_by(chunk)
                .map(|start| {
                    let end = (start + chunk).min(n);
                    let values = &self.values[start..end];
                    let strides = &self.strides[start..end];
                    scope.spawn(move || {
                        let mut scratch = HybridAnsSelectorScratch::default();
                        values
                            .iter()
                            .zip(strides)
                            .map(|(values, &stride)| {
                                select_hybrid_config_ans_sampled(
                                    values,
                                    1,
                                    stride,
                                    acceptance_ratio,
                                    &mut scratch,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("hybrid config selector"))
                .collect()
        })
    }

    fn select_with_pool(
        &self,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
        acceptance_ratio: f64,
    ) -> Vec<HybridUintConfig> {
        let n = self.values.len();
        if n == 0 {
            return Vec::new();
        }
        let chunk = n.div_ceil(pool.num_threads() * 4);
        pool.steal_map(scratch, n.div_ceil(chunk), |part, _| {
            let begin = part * chunk;
            let end = (begin + chunk).min(n);
            let mut selector = HybridAnsSelectorScratch::default();
            (begin..end)
                .map(|i| {
                    select_hybrid_config_ans_sampled(
                        &self.values[i],
                        1,
                        self.strides[i],
                        acceptance_ratio,
                        &mut selector,
                    )
                })
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect()
    }
}

#[cfg(test)]
fn select_hybrid_config_ans(
    values: &[u32],
    scratch: &mut HybridAnsSelectorScratch,
) -> HybridUintConfig {
    let stride = values.len().div_ceil(65_536).max(1);
    select_hybrid_config_ans_sampled(values, stride, stride, 0.995, scratch)
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

/// Contiguous, balanced token ranges for order-independent counting. Only
/// slice headers are copied; ANS coding still uses the original stream bounds.
fn parallel_token_parts<'a>(
    streams: &[&'a [Token]],
    pool: Option<&ThreadPool>,
) -> Option<Vec<Vec<&'a [Token]>>> {
    let threads = pool?.num_threads();
    let total: usize = streams.iter().map(|tokens| tokens.len()).sum();
    let parts = (total / 65_536).min(threads * 2);
    if threads <= 1 || parts <= 1 {
        return None;
    }
    let target = total.div_ceil(parts);
    let mut out = Vec::with_capacity(parts);
    let mut current = Vec::new();
    let mut remaining = target;
    for &tokens in streams {
        let mut tokens = tokens;
        while !tokens.is_empty() {
            let take = remaining.min(tokens.len());
            let (head, tail) = tokens.split_at(take);
            current.push(head);
            tokens = tail;
            remaining -= take;
            if remaining == 0 {
                out.push(std::mem::take(&mut current));
                remaining = target;
            }
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    Some(out)
}

fn clustered_histograms(
    streams: &[&[Token]],
    context_map: &[u8],
    configs: &[HybridUintConfig],
    pool: Option<&ThreadPool>,
    scratch: &mut CoderScratch,
) -> Vec<Histogram> {
    let count = |streams: &[&[Token]]| {
        let mut histograms = vec![Histogram::new(); configs.len()];
        for tokens in streams {
            for token in *tokens {
                let cluster = context_map[token.context as usize] as usize;
                histograms[cluster].add(uint_encode_with_config(token.value, configs[cluster]).0);
            }
        }
        histograms
    };
    let Some(parts) = parallel_token_parts(streams, pool) else {
        return count(streams);
    };
    let partial = pool
        .unwrap()
        .steal_map(scratch, parts.len(), |i, _| count(&parts[i]));
    let mut histograms = vec![Histogram::new(); configs.len()];
    for local in partial {
        for (dst, src) in histograms.iter_mut().zip(local) {
            for (count, added) in dst.counts.iter_mut().zip(src.counts) {
                *count += added;
            }
            dst.total_count += src.total_count;
        }
    }
    histograms
}

fn gather_hybrid_samples<const TRACK_MAX: bool>(
    streams: &[&[Token]],
    context_map: &[u8],
    counts: &[usize],
    pool: Option<&ThreadPool>,
    scratch: &mut CoderScratch,
) -> (HybridUintSamples, Vec<u32>) {
    let mut samples = HybridUintSamples::new(counts);
    let mut max_values = vec![0u32; counts.len()];
    let gather = |streams: &[&[Token]], samples: &mut HybridUintSamples, max_values: &mut [u32]| {
        for tokens in streams {
            for token in *tokens {
                let cluster = context_map[token.context as usize] as usize;
                samples.push(cluster, token.value);
                if TRACK_MAX {
                    max_values[cluster] = max_values[cluster].max(token.value);
                }
            }
        }
    };
    let Some(parts) = parallel_token_parts(streams, pool) else {
        gather(streams, &mut samples, &mut max_values);
        return (samples, max_values);
    };
    let pool = pool.unwrap();
    let part_counts = pool.steal_map(scratch, parts.len(), |i, _| {
        let mut counts = vec![0usize; counts.len()];
        for tokens in &parts[i] {
            for token in *tokens {
                counts[context_map[token.context as usize] as usize] += 1;
            }
        }
        counts
    });
    // Prefix counts recover exactly the serial per-cluster sample ordinals,
    // including when a partition starts halfway through a sampling stride.
    let mut running = vec![0usize; counts.len()];
    let starts: Vec<Vec<usize>> = part_counts
        .iter()
        .map(|local| {
            running
                .iter_mut()
                .zip(local)
                .zip(&samples.strides)
                .map(|((total, &count), &stride)| {
                    let remaining = (stride - *total % stride) % stride;
                    *total += count;
                    remaining
                })
                .collect()
        })
        .collect();
    debug_assert_eq!(running, counts);
    let partial = pool.steal_map(scratch, parts.len(), |i, _| {
        let mut local = HybridUintSamples {
            values: part_counts[i]
                .iter()
                .zip(&samples.strides)
                .map(|(&n, &stride)| Vec::with_capacity(n.div_ceil(stride)))
                .collect(),
            strides: samples.strides.clone(),
            remaining: starts[i].clone(),
        };
        let mut maxima = vec![0u32; counts.len()];
        gather(&parts[i], &mut local, &mut maxima);
        (local.values, maxima)
    });
    for (local, maxima) in partial {
        for (dst, values) in samples.values.iter_mut().zip(local) {
            dst.extend(values);
        }
        if TRACK_MAX {
            for (dst, value) in max_values.iter_mut().zip(maxima) {
                *dst = (*dst).max(value);
            }
        }
    }
    (samples, max_values)
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
    optimize_entropy_code_ac_streams(
        std::iter::once(tokens),
        num_contexts,
        huffman_pool,
        true,
        crate::Speed::Fast,
        None,
    )
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
    speed: crate::Speed,
    pool: Option<&ThreadPool>,
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
        if speed == crate::Speed::Slow {
            1.0
        } else {
            0.995
        },
        pool,
    )
}

pub(crate) fn optimize_entropy_code_ac_streams_fast<'a, I>(
    streams: I,
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
    pool: Option<&ThreadPool>,
) -> OwnedEntropyCode
where
    I: IntoIterator<Item = &'a [Token]>,
{
    optimize_entropy_code_ac_streams_impl(
        streams,
        num_contexts,
        huffman_pool,
        false,
        true,
        0.995,
        pool,
    )
}

const ANS_CLUSTER_PROXY_SYMBOL_BITS: f64 = 6.0;
const ANS_CLUSTER_PROXY_BASE_BITS: f64 = 12.0;
const MAX_ANS_RELOCATION_CONTEXTS: usize = 4096;
const XLOG2X_TABLE_SIZE: u32 = 1 << 16;

#[inline]
fn xlog2x(value: u32) -> f64 {
    value as f64 * f_log2(value as f64)
}

fn xlog2x_table() -> &'static [f64] {
    static TABLE: std::sync::OnceLock<Box<[f64]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        [0.0, 0.0]
            .into_iter()
            .chain((2..XLOG2X_TABLE_SIZE).map(xlog2x))
            .collect()
    })
}

/// Cheap Shannon-domain delta used only to nominate an ANS cluster move. The
/// final decision is made by serializing both complete candidates below.
fn moved_population_proxy_delta(source: &Histogram, target: &Histogram, moved: &Histogram) -> f64 {
    let table = xlog2x_table();
    let cost = |value: u32| {
        table
            .get(value as usize)
            .copied()
            .unwrap_or_else(|| xlog2x(value))
    };
    debug_assert!(moved.total_count <= source.total_count);
    let source_after = source.total_count - moved.total_count;
    let target_after = target.total_count + moved.total_count;
    let mut delta = cost(source_after) + cost(target_after)
        - cost(source.total_count)
        - cost(target.total_count);
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
        delta -= cost(source_count_after) + cost(target_count_after)
            - cost(source_count)
            - cost(target_count);
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
) -> Option<AnsClusterProposal> {
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

/// ANS tables for externally managed (already clustered) histograms:
/// (normalized histograms, per-cluster symbol info, packed reverse maps).
/// Final tables for a bundle whose histograms are already settled. With
/// `refine_precision` every table also gets the broader shift search that the
/// lossy path runs in `refine_ans_precisions`; the lossless path has no later
/// refinement stage and no RDO prices that the table choice could feed back
/// into, so it takes the searched tables directly.
pub(crate) fn build_ans_code_parts(
    histograms: &[Histogram],
    refine_precision: bool,
) -> (Vec<AnsHistogram>, Vec<Vec<AnsEncSymbolInfo>>, Vec<u16>) {
    let selected_histograms = histograms
        .iter()
        .map(|histogram| {
            let base = optimize_ans_histogram(&histogram.counts);
            if refine_precision {
                refine_ans_histogram_precision(&histogram.counts, &base).unwrap_or(base)
            } else {
                base
            }
        })
        .collect();
    let storage = build_ans_storage_from_selected(histograms, selected_histograms);
    (storage.histograms, storage.symbols, storage.reverse_maps)
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
    let total_tokens: usize = streams.iter().map(|stream| stream.len()).sum();
    let num_lanes = thread_pool.num_threads().min(streams.len());
    if num_lanes <= 1 || total_tokens < MIN_PARALLEL_TOKENS {
        return sequential(streams);
    }

    // Each stream owns an ANS state. Schedule those independently so groups
    // with many tokens do not strand workers assigned a lighter fixed chunk.
    thread_pool
        .steal_map(scratch, streams.len(), |i, _scratch| {
            ans_tokens_bits_pair(streams[i], &first, &second)
        })
        .into_iter()
        .fold((0usize, 0usize), |acc, bits| {
            (acc.0 + bits.0, acc.1 + bits.1)
        })
}

fn refine_ans_clusters_once(
    code: &mut OwnedEntropyCode,
    streams: &[&[Token]],
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> bool {
    if code.use_prefix_code || code.hybrid_uint_configs.len() <= 1 {
        return false;
    }
    let mut histograms = vec![Histogram::new(); code.hybrid_uint_configs.len()];
    let mut context_histograms = vec![Histogram::new(); code.context_map.len()];
    for tokens in streams {
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
    accept_ans_cluster_proposal(
        code,
        streams,
        (candidate_histograms, candidate_map, candidate_configs),
        thread_pool,
        scratch,
    )
}

type AnsClusterProposal = (Vec<Histogram>, Vec<u8>, Vec<HybridUintConfig>);

fn accept_ans_cluster_proposal(
    code: &mut OwnedEntropyCode,
    streams: &[&[Token]],
    (candidate_histograms, candidate_map, candidate_configs): AnsClusterProposal,
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> bool {
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
        streams,
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

#[derive(Clone, Copy)]
pub(crate) enum AnsRefinement {
    Slow,
    Fast { recluster: bool },
}

/// Refine only the final entropy bundle, after the caller has frozen RDO
/// prices and tokens. Higher-level frame selection can still compare the
/// resulting sizes. Retain the existing Slow refinement as an
/// incumbent, then test either independent ANS clustering (which can split
/// collapsed clusters) or a bounded batch of same-configuration moves.
pub(crate) fn refine_ans_clusters<'a, I>(
    code: &mut OwnedEntropyCode,
    streams: I,
    refinement: AnsRefinement,
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> bool
where
    I: IntoIterator<Item = &'a [Token]>,
{
    if code.use_prefix_code {
        return false;
    }
    let streams: Vec<&[Token]> = streams.into_iter().collect();
    // Slow already had this refinement. Preserve it as the incumbent; Fast
    // did not, so it can go straight to the stronger candidate.
    let mut changed = matches!(refinement, AnsRefinement::Slow)
        && refine_ans_clusters_once(code, &streams, thread_pool, scratch);
    let recluster = !matches!(refinement, AnsRefinement::Fast { recluster: false });
    let proposal = if recluster {
        propose_ans_reclustering(
            &streams,
            code.context_map.len(),
            thread_pool,
            scratch,
            refinement,
        )
    } else {
        propose_ans_cluster_batch(code, &streams)
    };
    if let Some(proposal) = proposal {
        changed |= accept_ans_cluster_proposal(code, &streams, proposal, thread_pool, scratch);
    }
    // Precision changes do not feed back into clustering or provisional RDO
    // prices. Fast retains its existing table search at every distance.
    if matches!(refinement, AnsRefinement::Slow) {
        changed |= refine_ans_precisions(code, &streams, thread_pool, scratch);
    }
    changed |= fill_unused_contexts(code, &streams, scratch);
    changed
}

/// Contexts without tokens still occupy signaled context-map entries. The
/// clusterers leave them at cluster 0, which interrupts the runs the map's
/// run-length coder feeds on; repeating the preceding entry instead costs no
/// payload bits. Adopted only when the serialized map actually shrinks.
fn fill_unused_contexts(
    code: &mut OwnedEntropyCode,
    streams: &[&[Token]],
    scratch: &mut CoderScratch,
) -> bool {
    if code.orig_context_map.is_some() || code.context_map.len() <= 1 {
        return false;
    }
    let mut used = vec![false; code.context_map.len()];
    for tokens in streams {
        for token in *tokens {
            used[token.context as usize] = true;
        }
    }
    if used.iter().all(|&u| u) {
        return false;
    }
    let mut filled = code.context_map.clone();
    let mut previous = 0u8;
    for (entry, &used) in filled.iter_mut().zip(&used) {
        if used {
            previous = *entry;
        } else {
            *entry = previous;
        }
    }
    let map_bits = |map: &[u8], scratch: &mut CoderScratch| {
        let mut probe = code.as_ref();
        probe.context_map = map;
        let mut w = BitWriter::new();
        write_context_map(&probe, &mut scratch.huffman_pool, &mut w);
        w.bits_written()
    };
    if map_bits(&filled, scratch) >= map_bits(&code.context_map, scratch) {
        return false;
    }
    code.context_map = filled;
    true
}

fn refine_ans_precisions(
    code: &mut OwnedEntropyCode,
    streams: &[&[Token]],
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> bool {
    if code.use_prefix_code || code.ans_histograms.is_empty() {
        return false;
    }
    let histograms = clustered_histograms(
        streams,
        &code.context_map,
        &code.hybrid_uint_configs,
        Some(thread_pool),
        scratch,
    );
    let proposals = thread_pool.steal_map(scratch, histograms.len(), |i, _| {
        refine_ans_histogram_precision(&histograms[i].counts, &code.ans_histograms[i])
    });
    if proposals.iter().all(Option::is_none) {
        return false;
    }
    let selected = proposals
        .into_iter()
        .zip(&code.ans_histograms)
        .map(|(proposal, incumbent)| proposal.unwrap_or_else(|| incumbent.clone()))
        .collect();
    let candidate_ans = build_ans_storage_from_selected(&histograms, selected);
    let original = AnsBundleRef {
        context_map: &code.context_map,
        configs: &code.hybrid_uint_configs,
        histograms: &code.ans_histograms,
        symbols: &code.ans_symbols,
        reverse_maps: &code.ans_reverse_maps,
    };
    let candidate = AnsBundleRef {
        context_map: &code.context_map,
        configs: &code.hybrid_uint_configs,
        histograms: &candidate_ans.histograms,
        symbols: &candidate_ans.symbols,
        reverse_maps: &candidate_ans.reverse_maps,
    };
    // The histogram model nominates tables; actual ANS states decide whether
    // the complete entropy bundle improves on the already reclustered code.
    let (original_bits, candidate_bits) = exact_ans_bundle_bits_pair(
        streams,
        code.orig_num_contexts,
        &original,
        &candidate,
        thread_pool,
        scratch,
    );
    if candidate_bits >= original_bits {
        return false;
    }
    code.ans_histograms = candidate_ans.histograms;
    code.ans_symbols = candidate_ans.symbols;
    code.ans_reverse_maps = candidate_ans.reverse_maps;
    true
}

/// Empty contexts must remain in the signaled map, but need not participate
/// in the cluster search. Build dense populations without allocating a full
/// 128-bin histogram for every unused AC context.
fn propose_ans_reclustering(
    streams: &[&[Token]],
    num_contexts: usize,
    thread_pool: &ThreadPool,
    scratch: &mut CoderScratch,
    refinement: AnsRefinement,
) -> Option<AnsClusterProposal> {
    let mut dense = vec![usize::MAX; num_contexts];
    for tokens in streams {
        for token in *tokens {
            dense[token.context as usize] = 0;
        }
    }
    let mut histograms = Vec::new();
    for index in &mut dense {
        if *index != usize::MAX {
            *index = histograms.len();
            histograms.push(Histogram::new());
        }
    }
    if histograms.len() <= 1 {
        return None;
    }
    for tokens in streams {
        for token in *tokens {
            histograms[dense[token.context as usize]].add(uint_encode(token.value).0);
        }
    }
    let mut assignment = vec![0u8; histograms.len()];
    let n = super::cluster::cluster_histograms_ans(
        &mut histograms,
        &mut assignment,
        Some(thread_pool),
        true,
        super::cluster::CLUSTERS_LIMIT,
        if matches!(refinement, AnsRefinement::Slow) {
            6
        } else {
            2
        },
    );
    histograms.truncate(n);
    let context_map: Vec<u8> = dense
        .iter()
        .map(|&index| {
            if index == usize::MAX {
                0
            } else {
                assignment[index]
            }
        })
        .collect();
    let counts: Vec<usize> = histograms.iter().map(|h| h.total_count as usize).collect();
    let (samples, max_values) =
        gather_hybrid_samples::<true>(streams, &context_map, &counts, Some(thread_pool), scratch);
    let mut configs = samples.select_with_pool(
        thread_pool,
        scratch,
        if matches!(refinement, AnsRefinement::Slow) {
            1.0
        } else {
            0.995
        },
    );
    // Sampling may miss a rare large value. Check a conservative upper bound
    // for every selected configuration before rebuilding fixed-size tables.
    for (config, &max_value) in configs.iter_mut().zip(&max_values) {
        let max_symbol =
            uint_encode_with_config(max_value, *config).0 | ((1u32 << config.lsb_in_token) - 1);
        if max_symbol as usize >= ALPHABET_SIZE {
            *config = HybridUintConfig::DEFAULT;
        }
    }
    if configs.iter().any(|&c| c != HybridUintConfig::DEFAULT) {
        histograms =
            clustered_histograms(streams, &context_map, &configs, Some(thread_pool), scratch);
    }
    Some((histograms, context_map, configs))
}

fn propose_ans_cluster_batch(
    code: &OwnedEntropyCode,
    streams: &[&[Token]],
) -> Option<AnsClusterProposal> {
    if code.hybrid_uint_configs.len() <= 1 {
        return None;
    }
    let mut histograms = vec![Histogram::new(); code.hybrid_uint_configs.len()];
    let mut contexts = vec![Histogram::new(); code.context_map.len()];
    let mut map = code.context_map.clone();
    let mut configs = code.hybrid_uint_configs.clone();
    for tokens in streams {
        for token in *tokens {
            let context = token.context as usize;
            let cluster = map[context] as usize;
            let symbol = uint_encode_with_config(token.value, configs[cluster]).0;
            contexts[context].add(symbol);
            histograms[cluster].add(symbol);
        }
    }
    // Pack descending frequency and ascending context ID so integer sorting
    // preserves the existing order without a separate histogram comparator.
    // Only u32 token context IDs can populate these nonempty histograms.
    let mut order: Vec<u64> = contexts
        .iter()
        .enumerate()
        .filter(|(_, h)| h.total_count != 0)
        .map(|(i, h)| (u64::from(u32::MAX - h.total_count) << 32) | i as u64)
        .collect();
    order.sort_unstable();
    order.truncate(MAX_ANS_RELOCATION_CONTEXTS);
    let mut changed = false;
    for key in order {
        let context = (key as u32) as usize;
        let source = map[context] as usize;
        let moved = &contexts[context];
        // Whole-cluster merging below also remaps tokenless contexts.
        if moved.total_count == histograms[source].total_count {
            continue;
        }
        let mut best = source;
        let mut best_delta = -0.25;
        for target in 0..histograms.len() {
            if target == source || configs[target] != configs[source] {
                continue;
            }
            let delta =
                moved_population_proxy_delta(&histograms[source], &histograms[target], moved);
            if delta < best_delta {
                best_delta = delta;
                best = target;
            }
        }
        if best != source {
            let (a, b) = two_histograms_mut(&mut histograms, source, best);
            move_histogram(a, b, moved);
            map[context] = best as u8;
            changed = true;
        }
    }
    for _ in 0..4 {
        let mut best = None;
        let mut best_delta = -0.25;
        for source in 0..histograms.len() {
            for target in 0..source {
                if configs[source] != configs[target]
                    || histograms[source].total_count == 0
                    || histograms[target].total_count == 0
                {
                    continue;
                }
                let delta = moved_population_proxy_delta(
                    &histograms[source],
                    &histograms[target],
                    &histograms[source],
                );
                if delta < best_delta {
                    best_delta = delta;
                    best = Some((source, target));
                }
            }
        }
        let Some((source, target)) = best else { break };
        let moved = histograms[source].clone();
        let (a, b) = two_histograms_mut(&mut histograms, source, target);
        move_histogram(a, b, &moved);
        for cluster in &mut map {
            if *cluster as usize == source {
                *cluster = target as u8;
            }
        }
        changed = true;
    }
    if !changed {
        return None;
    }
    compact_ans_clusters(&mut histograms, &mut map, &mut configs);
    Some((histograms, map, configs))
}

fn optimize_entropy_code_ac_streams_impl<'a, I>(
    streams: I,
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
    select_configs: bool,
    fast_cluster: bool,
    hybrid_acceptance: f64,
    pool: Option<&ThreadPool>,
) -> OwnedEntropyCode
where
    I: IntoIterator<Item = &'a [Token]>,
{
    // Collected so the tokens can be walked twice: once to cluster under the
    // default config, once to rebuild histograms under the selected per-cluster
    // configs. Only the slice headers are copied.
    let streams: Vec<&[Token]> = streams.into_iter().collect();
    optimize_entropy_code_ac_slices(
        &streams,
        num_contexts,
        huffman_pool,
        select_configs,
        fast_cluster,
        hybrid_acceptance,
        pool,
    )
}

// Iterator adapters only collect slice references; share the entropy setup.
fn optimize_entropy_code_ac_slices(
    streams: &[&[Token]],
    num_contexts: usize,
    huffman_pool: &mut Vec<HuffmanNode>,
    select_configs: bool,
    fast_cluster: bool,
    hybrid_acceptance: f64,
    pool: Option<&ThreadPool>,
) -> OwnedEntropyCode {
    let mut histograms = vec![Histogram::new(); num_contexts];
    for tokens in streams {
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
        super::cluster::cluster_histograms_with_pool(
            &mut histograms,
            &mut context_map,
            huffman_pool,
            pool,
        );
    }

    // Second walk: pick each final cluster's HybridUint config from its actual
    // token values (DEFAULT is among the candidates, so this can only move
    // where the selector's cost model says it pays), then rebuild the symbol
    // histograms under the selected configs so the prefix/rANS tables match
    // what the writer will emit.
    let num_clusters = histograms.len();
    let hybrid_uint_configs = if select_configs {
        // Every AC token contributes one symbol to these clustered counts.
        let counts: Vec<_> = histograms.iter().map(|h| h.total_count as usize).collect();
        let (samples, _) = gather_hybrid_samples::<false>(
            streams,
            &context_map,
            &counts,
            pool,
            &mut CoderScratch::lossless(),
        );
        if let Some(pool) = pool {
            samples.select_with_pool(pool, &mut CoderScratch::lossless(), hybrid_acceptance)
        } else {
            samples.select_with_acceptance(hybrid_acceptance)
        }
    } else {
        vec![HybridUintConfig::DEFAULT; num_clusters]
    };
    if hybrid_uint_configs
        .iter()
        .any(|&c| c != HybridUintConfig::DEFAULT)
    {
        histograms = clustered_histograms(
            streams,
            &context_map,
            &hybrid_uint_configs,
            pool,
            &mut CoderScratch::lossless(),
        );
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
    // A prefix code spends at least one bit per symbol; ~8 bits describe each
    // used symbol's code length.
    bits.max(total) + used * 8.0
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

    // The order-0 estimates only rank the candidates; the pick is by exact
    // serialized size. Both estimates carry the prefix-code floor of one bit
    // per coded symbol, which a Shannon sum ignores: a 7425-entry map with two
    // clusters estimated ~145 bits for the plain coding and actually cost
    // 7449, where run-length coding takes ~240.
    static VARIANTS: [(bool, bool); 4] =
        [(false, false), (true, false), (false, true), (true, true)];
    let estimates = [
        context_map_cost(&entries),
        context_map_cost(&mtf),
        run_length_cost(&raw_runs),
        run_length_cost(&mtf_runs),
    ];
    // Without a single run the run-length codings only add the LZ77 header
    // (and would recurse into this function for their two-context map).
    let has_runs = raw_runs.len() < entries.len() || mtf_runs.len() < entries.len();
    let mut order: Vec<usize> = if has_runs {
        vec![0, 1, 2, 3]
    } else {
        vec![0, 1]
    };
    order.sort_by(|&a, &b| estimates[a].total_cmp(&estimates[b]).then(a.cmp(&b)));
    // Small maps are cheap enough to serialize every way; large ones try the
    // runner-up only when the estimates are close.
    let small = entries.len() <= 256;
    let mut best: Option<BitWriter> = None;
    for (rank, &i) in order.iter().enumerate() {
        if !small && (rank >= 2 || (rank == 1 && estimates[i] > 1.2 * estimates[order[0]])) {
            break;
        }
        let (use_mtf, use_lz77) = VARIANTS[i];
        let mut probe = BitWriter::new();
        write_context_map_body(
            &entries,
            &mtf,
            &raw_runs,
            &mtf_runs,
            use_mtf,
            use_lz77,
            huffman_pool,
            &mut probe,
        );
        if best
            .as_ref()
            .is_none_or(|b| probe.bits_written() < b.bits_written())
        {
            best = Some(probe);
        }
    }
    let best = best.expect("at least one context-map candidate");
    // Fixed-width "simple" map: is_simple=1, bits_per_entry (2 bits), entries.
    if max <= 7 {
        let bits_per_entry = (32 - u32::from(max).leading_zeros()) as usize;
        if 3 + entries.len() * bits_per_entry < best.bits_written() {
            w.write(1, 1);
            w.write(2, bits_per_entry as u64);
            for &e in &entries {
                w.write(bits_per_entry, u64::from(e));
            }
            return;
        }
    }
    w.append_bits(&best);
}

#[allow(clippy::too_many_arguments)]
fn write_context_map_body(
    entries: &[u8],
    mtf: &[u8],
    raw_runs: &[RunSymbol],
    mtf_runs: &[RunSymbol],
    use_mtf: bool,
    use_lz77: bool,
    huffman_pool: &mut Vec<HuffmanNode>,
    w: &mut BitWriter,
) {
    let symbols: &[u8] = if use_mtf { mtf } else { entries };

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
    let runs = if use_mtf { mtf_runs } else { raw_runs };
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
    // a second context makes the bundle carry a nested context map. The literal
    // and length symbols share one prefix code: at least one bit each.
    bits.max(f64::from(total)) + extra_bits + used * 8.0 + f64::from(distances) + 24.0
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
mod ans_refinement_tests {
    use super::*;

    fn code_for(
        tokens: &[Token],
        map: Vec<u8>,
        configs: Vec<HybridUintConfig>,
    ) -> OwnedEntropyCode {
        let mut histograms = vec![Histogram::new(); configs.len()];
        for token in tokens {
            let h = map[token.context as usize] as usize;
            histograms[h].add(uint_encode_with_config(token.value, configs[h]).0);
        }
        let ans = build_ans_storage(&histograms);
        OwnedEntropyCode {
            orig_num_contexts: map.len(),
            context_map: map,
            prefix_codes: build_huffman_codes(&histograms, &mut Vec::new()),
            hybrid_uint_configs: configs,
            orig_context_map: None,
            use_prefix_code: false,
            ans_histograms: ans.histograms,
            ans_pricing_freqs: ans.pricing_freqs,
            ans_symbols: ans.symbols,
            ans_reverse_maps: ans.reverse_maps,
        }
    }

    fn bundle(code: &OwnedEntropyCode) -> AnsBundleRef<'_> {
        AnsBundleRef {
            context_map: &code.context_map,
            configs: &code.hybrid_uint_configs,
            histograms: &code.ans_histograms,
            symbols: &code.ans_symbols,
            reverse_maps: &code.ans_reverse_maps,
        }
    }

    fn serialized(code: &OwnedEntropyCode, streams: &[&[Token]]) -> (usize, Vec<u8>) {
        let mut writer = BitWriter::new();
        write_entropy_code(&code.as_ref(), &mut Vec::new(), &mut writer);
        for tokens in streams {
            super::super::ans::write_ans_tokens(
                tokens,
                &code.context_map,
                &code.ans_symbols,
                &code.ans_reverse_maps,
                &code.hybrid_uint_configs,
                &mut writer,
            );
        }
        let bits = writer.bits_written();
        writer.zero_pad_to_byte();
        (bits, writer.into_bytes())
    }

    #[test]
    fn exact_comparison_handles_different_integer_configs_and_maps() {
        let tokens: Vec<_> = (0..20_000)
            .map(|i| Token::new(i % 3, (i * 73 + i / 11) % 4096))
            .collect();
        let first = code_for(&tokens, vec![0, 1, 0], vec![HybridUintConfig::DEFAULT; 2]);
        let second = code_for(
            &tokens,
            vec![1, 0, 2],
            vec![
                HYBRID_CANDIDATES[0],
                HYBRID_CANDIDATES[7],
                HYBRID_CANDIDATES[11],
            ],
        );
        // An uneven large first stream followed by small streams exercises
        // work stealing; every boundary still starts a distinct ANS state.
        let split = tokens.len() / 2;
        let mut streams: Vec<&[Token]> = std::iter::once(&tokens[..split])
            .chain(tokens[split..].chunks(997))
            .collect();
        streams.push(&[]);
        let expected = (
            serialized(&first, &streams).0,
            serialized(&second, &streams).0,
        );
        for threads in [1, 4, 12] {
            assert_eq!(
                exact_ans_bundle_bits_pair(
                    &streams,
                    3,
                    &bundle(&first),
                    &bundle(&second),
                    &ThreadPool::new_lossless(threads),
                    &mut CoderScratch::lossless(),
                ),
                expected
            );
        }
    }

    #[test]
    fn unused_contexts_repeat_the_preceding_map_entry_only_when_the_map_shrinks() {
        // Populate every third context of the first half: the first quarter
        // at cluster 1, the second at cluster 0. The unused entries at cluster
        // 0 chop the cluster-1 stretch into 1,0,0,1,0,0,... which has no run
        // long enough for the map's run-length coder; repeating the previous
        // entry turns it into one run.
        const CONTEXTS: usize = 600;
        let mut map = vec![0u8; CONTEXTS];
        let mut tokens = Vec::new();
        for context in (0..CONTEXTS / 2).step_by(3) {
            map[context] = u8::from(context < CONTEXTS / 4);
            for i in 0..40u32 {
                tokens.push(Token::new(context as u32, (i * 7 + context as u32) % 9));
            }
        }
        let mut code = code_for(&tokens, map.clone(), vec![HybridUintConfig::DEFAULT; 2]);
        let streams: Vec<&[Token]> = tokens.chunks(1000).collect();
        let (before_bits, before_bytes) = serialized(&code, &streams);
        let header_bits = |code: &OwnedEntropyCode| {
            let mut writer = BitWriter::new();
            write_entropy_code(&code.as_ref(), &mut Vec::new(), &mut writer);
            writer.bits_written()
        };
        let before_header = header_bits(&code);

        assert!(fill_unused_contexts(
            &mut code,
            &streams,
            &mut CoderScratch::lossless()
        ));
        let mut previous = 0u8;
        for context in 0..CONTEXTS {
            let used = context < CONTEXTS / 2 && context % 3 == 0;
            if used {
                assert_eq!(code.context_map[context], map[context]);
                previous = map[context];
            } else {
                assert_eq!(code.context_map[context], previous);
            }
        }
        let after_header = header_bits(&code);
        assert!(
            after_header < before_header,
            "{after_header} >= {before_header}"
        );
        let (after_bits, after_bytes) = serialized(&code, &streams);
        assert_eq!(before_bits - before_header, after_bits - after_header);
        assert!(after_bytes.len() <= before_bytes.len());
        // A second pass has nothing left to improve.
        assert!(!fill_unused_contexts(
            &mut code,
            &streams,
            &mut CoderScratch::lossless()
        ));
    }

    #[test]
    fn reclustering_splits_a_collapsed_histogram_and_is_deterministic() {
        let mut tokens = Vec::new();
        for context in 0..96 {
            for i in 0..1028 {
                tokens.push(Token::new(
                    context * 73 + 2,
                    u32::from((i * 37) % 257 < context * 2 + 3),
                ));
            }
        }
        let streams: Vec<&[Token]> = tokens.chunks(1001).collect();
        let mut expected = None;
        for threads in [1, 4] {
            let mut code = code_for(&tokens, vec![0; 7425], vec![HybridUintConfig::DEFAULT]);
            let before = serialized(&code, &streams).0;
            assert!(refine_ans_clusters(
                &mut code,
                streams.iter().copied(),
                AnsRefinement::Slow,
                &ThreadPool::new_lossless(threads),
                &mut CoderScratch::lossless(),
            ));
            assert_eq!(code.context_map.len(), 7425);
            assert!(code.ans_histograms.len() > 1);
            let encoded = serialized(&code, &streams);
            assert!(encoded.0 < before);
            if let Some(expected) = &expected {
                assert_eq!(&encoded, expected);
            } else {
                expected = Some(encoded);
            }
        }
    }

    #[test]
    fn rejected_reclustering_keeps_the_incumbent_code() {
        // The incumbent already shares the same distribution and uses the
        // shorter config header. The sampled selector's stability margin
        // nominates DEFAULT, which cannot beat it on the actual wire cost.
        let tokens: Vec<_> = (0..20_000)
            .map(|i| Token::new((i / 2) % 2, i % 2))
            .collect();
        let streams = [tokens.as_slice()];
        let mut code = code_for(&tokens, vec![0; 2], vec![HYBRID_CANDIDATES[0]]);
        let before = serialized(&code, &streams);
        assert!(!refine_ans_clusters(
            &mut code,
            streams,
            AnsRefinement::Slow,
            &ThreadPool::new_lossless(1),
            &mut CoderScratch::lossless(),
        ));
        assert_eq!(serialized(&code, &streams), before);
    }

    #[test]
    fn empty_and_single_context_bundles_remain_unchanged() {
        let pool = ThreadPool::new_lossless(1);
        let mut scratch = CoderScratch::lossless();
        for tokens in [vec![], vec![Token::new(2, 3); 1000]] {
            let streams = [tokens.as_slice()];
            let mut code = code_for(&tokens, vec![0; 7], vec![HybridUintConfig::DEFAULT]);
            let before = serialized(&code, &streams);
            for refinement in [
                AnsRefinement::Fast { recluster: false },
                AnsRefinement::Slow,
            ] {
                assert!(!refine_ans_clusters(
                    &mut code,
                    streams,
                    refinement,
                    &pool,
                    &mut scratch
                ));
                assert_eq!(serialized(&code, &streams), before);
            }
        }
    }

    #[test]
    fn batched_refinement_preserves_population_and_accepts_only_smaller_bundles() {
        let tokens: Vec<_> = (0..30_000)
            .map(|i| Token::new(i % 12, (i / 12) % (2 + i % 12)))
            .collect();
        let streams: Vec<&[Token]> = tokens.chunks(3001).collect();
        let map: Vec<_> = (0..12).map(|i| (i % 3) as u8).collect();
        let mut code = code_for(&tokens, map, vec![HybridUintConfig::DEFAULT; 3]);
        let before = serialized(&code, &streams).0;
        assert!(refine_ans_clusters(
            &mut code,
            streams.iter().copied(),
            AnsRefinement::Fast { recluster: false },
            &ThreadPool::new_lossless(4),
            &mut CoderScratch::lossless(),
        ));
        assert!(serialized(&code, &streams).0 < before);
        let proposal = propose_ans_cluster_batch(&code, &streams);
        if let Some((histograms, map, configs)) = proposal {
            let mut actual = vec![Histogram::new(); histograms.len()];
            for token in tokens {
                let cluster = map[token.context as usize] as usize;
                actual[cluster].add(uint_encode_with_config(token.value, configs[cluster]).0);
            }
            for (a, b) in histograms.iter().zip(&actual) {
                assert_eq!(a.counts, b.counts);
                assert_eq!(a.total_count, b.total_count);
            }
        }
    }

    #[test]
    fn precision_refinement_reduces_wire_cost_without_changing_token_mapping() {
        let mut accepted = 0;
        for scale in [100, 1000, 10_000] {
            let mut tokens = Vec::new();
            for value in 0..64 {
                for _ in 0..(scale / (value + 1)) {
                    tokens.push(Token::new(value % 3, value));
                }
            }
            let streams: Vec<&[Token]> = tokens.chunks(997).collect();
            let initial = code_for(&tokens, vec![0, 1, 2], vec![HybridUintConfig::DEFAULT; 3]);
            let before = serialized(&initial, &streams);
            let mut expected = None;
            for threads in [1, 4] {
                let mut code = code_for(&tokens, vec![0, 1, 2], vec![HybridUintConfig::DEFAULT; 3]);
                let changed = refine_ans_precisions(
                    &mut code,
                    &streams,
                    &ThreadPool::new_lossless(threads),
                    &mut CoderScratch::lossless(),
                );
                let after = serialized(&code, &streams);
                if changed {
                    assert!(after.0 < before.0);
                    accepted += 1;
                } else {
                    assert_eq!(after, before);
                }
                assert_eq!(code.context_map, initial.context_map);
                assert_eq!(code.hybrid_uint_configs, initial.hybrid_uint_configs);
                assert_eq!(code.ans_pricing_freqs, initial.ans_pricing_freqs);
                if let Some(expected) = &expected {
                    assert_eq!(&after, expected);
                } else {
                    expected = Some(after);
                }
            }
        }
        assert!(accepted > 0);
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
            let fast = select_hybrid_config_ans(&values, &mut scratch);
            assert_eq!(
                fast,
                exhaustive_selector(&values),
                "hybrid shortlist diverged on generated case {case}"
            );
            let slow = select_hybrid_config_ans_sampled(&values, 1, 1, 1.0, &mut scratch);
            // Removing the margin can admit a small win from the existing
            // shortlist, which need not be the exhaustive global optimum.
            // Independently price the selected configurations: Slow must
            // never raise the exact cost on the same samples.
            let exact_cost = |config| {
                let mut counts = [0u32; ALPHABET_SIZE];
                let mut extra_bits = 0u64;
                for &value in &values {
                    let (symbol, nbits, _) = uint_encode_with_config(value, config);
                    counts[symbol as usize] += 1;
                    extra_bits += nbits as u64;
                }
                hybrid_ans_candidate_cost(&counts, extra_bits, 1, config)
            };
            assert!(
                exact_cost(slow) <= exact_cost(fast),
                "Slow raised the exact HybridUint cost on generated case {case}"
            );
        }
    }
}

#[cfg(test)]
mod sampled_hybrid_tests {
    use super::*;

    #[test]
    fn parallel_token_passes_preserve_histograms_samples_and_maxima() {
        let context_map = [2, 0, 1, 2, 1, 0, 3];
        let configs = [
            HybridUintConfig::DEFAULT,
            HYBRID_CANDIDATES[0],
            HYBRID_CANDIDATES[8],
            HYBRID_CANDIDATES[11],
            HybridUintConfig::DEFAULT,
        ];
        for len in [
            0, 1, 65_535, 65_536, 65_537, 131_071, 131_072, 131_073, 524_309,
        ] {
            let mut tokens: Vec<_> = (0..len)
                .map(|i| {
                    Token::new(
                        ((i * 17 + i / 101) % context_map.len()) as u32,
                        (i as u32).wrapping_mul(731) % 4097,
                    )
                })
                .collect();
            if let Some(last) = tokens.last_mut() {
                // The maximum must survive even when its ordinal is unsampled.
                last.value = u32::MAX;
            }
            let a = len.min(19);
            let b = len.min(65_537);
            let layouts = [
                vec![tokens.as_slice()],
                vec![&tokens[..a], &[], &tokens[a..b], &tokens[b..], &[]],
            ];
            let mut counts = vec![0usize; configs.len()];
            let mut expected_histograms = vec![Histogram::new(); configs.len()];
            let mut expected_maxima = vec![0u32; configs.len()];
            for token in &tokens {
                let cluster = context_map[token.context as usize] as usize;
                counts[cluster] += 1;
                expected_maxima[cluster] = expected_maxima[cluster].max(token.value);
                expected_histograms[cluster]
                    .add(uint_encode_with_config(token.value, configs[cluster]).0);
            }
            let mut expected_samples = HybridUintSamples::new(&counts);
            for token in &tokens {
                expected_samples.push(context_map[token.context as usize] as usize, token.value);
            }
            for streams in layouts {
                for threads in [1, 4, 12] {
                    let pool = ThreadPool::new_lossless(threads);
                    let mut scratch = CoderScratch::lossless();
                    let (samples, maxima) = gather_hybrid_samples::<true>(
                        &streams,
                        &context_map,
                        &counts,
                        Some(&pool),
                        &mut scratch,
                    );
                    assert_eq!(
                        samples.values, expected_samples.values,
                        "len={len}, threads={threads}"
                    );
                    assert_eq!(samples.strides, expected_samples.strides);
                    assert_eq!(maxima, expected_maxima);
                    let (samples, _) = gather_hybrid_samples::<false>(
                        &streams,
                        &context_map,
                        &counts,
                        Some(&pool),
                        &mut scratch,
                    );
                    assert_eq!(samples.values, expected_samples.values);
                    let actual = clustered_histograms(
                        &streams,
                        &context_map,
                        &configs,
                        Some(&pool),
                        &mut scratch,
                    );
                    for (actual, expected) in actual.iter().zip(&expected_histograms) {
                        assert_eq!(actual.counts, expected.counts);
                        assert_eq!(actual.total_count, expected.total_count);
                    }
                }
            }
        }
    }

    #[test]
    fn pooled_selection_preserves_configs_and_acceptance() {
        let values: Vec<Vec<u32>> = (0..19)
            .map(|i| {
                (0..i * i * 37)
                    .map(|v| ((v * 731) % (17 + i * 193)) as u32)
                    .collect()
            })
            .collect();
        let strides = (0..values.len()).map(|i| i % 5 + 1).collect();
        let samples = HybridUintSamples::from_parts(values, strides);
        for acceptance in [0.995, 1.0] {
            let expected = samples.select_with_acceptance(acceptance);
            for threads in [1, 4, 12] {
                let actual = samples.select_with_pool(
                    &ThreadPool::new_lossless(threads),
                    &mut CoderScratch::lossless(),
                    acceptance,
                );
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn bounded_samples_keep_original_cluster_ordinals_and_configs() {
        for n in [0, 1, 65535, 65536, 65537, 131072, 131073] {
            let a: Vec<u32> = (0..n).map(|i| (i as u32 * 731) % 4097).collect();
            let b: Vec<u32> = (0..n / 3).map(|i| (i % 19) as u32).collect();
            let mut sampled = HybridUintSamples::new(&[a.len(), b.len()]);
            for i in 0..n {
                sampled.push(0, a[i]);
                if i < b.len() {
                    sampled.push(1, b[i]);
                }
            }
            let expected: Vec<_> = [&a, &b]
                .iter()
                .map(|values| {
                    select_hybrid_config_ans(values, &mut HybridAnsSelectorScratch::default())
                })
                .collect();
            assert_eq!(sampled.select(), expected);
            for (cluster, values) in [&a, &b].iter().enumerate() {
                assert_eq!(
                    sampled.values[cluster],
                    values
                        .iter()
                        .step_by(values.len().div_ceil(65536).max(1))
                        .copied()
                        .collect::<Vec<_>>()
                );
                assert!(sampled.values[cluster].len() <= 65536);
            }
        }
    }
}
