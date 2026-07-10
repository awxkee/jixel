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

use super::histogram::Histogram;
use super::huffman_tree::create_huffman_tree;
use super::prefix_code::ALPHABET_SIZE;

#[inline]
fn counts_bit_cost(counts: &[u32; ALPHABET_SIZE], total_count: u32) -> f32 {
    if total_count == 0 {
        return 0.0;
    }

    let mut used_symbols = 0usize;
    for &count in counts.iter() {
        if count != 0 {
            used_symbols += 1;
            if used_symbols > 2 {
                break;
            }
        }
    }
    // A single-symbol histogram is a depth-0 simple prefix code: 0 bits per
    // token. Costing it at 1 bit/token (as brotli does) makes the clusterer
    // merge constant-token contexts (e.g. the EPF sharpness field) into large
    // histograms, silently paying >=1 bit per token for a stream that should
    // be free.
    if used_symbols <= 1 {
        return 0.0;
    }
    if used_symbols == 2 {
        return total_count as f32;
    }

    let mut depths = [0u8; ALPHABET_SIZE];
    create_huffman_tree(counts, 15, &mut depths);

    let mut cost = 0.0f32;
    for (&count, &depth) in counts.iter().zip(depths.iter()) {
        cost += count as f32 * depth as f32;
    }
    cost
}

#[inline]
fn histogram_bit_cost(h: &Histogram) -> f32 {
    counts_bit_cost(&h.counts, h.total_count)
}

#[inline]
fn add_counts(dst: &mut [u32; ALPHABET_SIZE], src: &[u32; ALPHABET_SIZE]) {
    for (dst, &src) in dst.iter_mut().zip(src.iter()) {
        *dst += src;
    }
}

#[inline]
fn histogram_add(a: &mut Histogram, b: &Histogram) {
    add_counts(&mut a.counts, &b.counts);
    a.total_count += b.total_count;
}

#[inline]
fn histogram_distance(
    a: &Histogram,
    b: &Histogram,
    a_cost: f32,
    b_cost: f32,
    scratch: &mut [u32; ALPHABET_SIZE],
) -> f32 {
    if a.total_count == 0 || b.total_count == 0 {
        return 0.0;
    }

    *scratch = a.counts;
    add_counts(scratch, &b.counts);
    counts_bit_cost(scratch, a.total_count + b.total_count) - a_cost - b_cost
}

/// Cluster `histograms` in place down to at most 8 distinct ones; produce
/// `context_map[i] = cluster index for histogram i`.
pub(crate) fn cluster_histograms(histograms: &mut Vec<Histogram>, context_map: &mut Vec<u8>) {
    if histograms.len() <= 1 {
        context_map.clear();
        context_map.resize(histograms.len(), 0);
        return;
    }

    const CLUSTERS_LIMIT: usize = 64;
    const UNMAPPED: u8 = u8::MAX;

    let max_histograms = CLUSTERS_LIMIT.min(histograms.len());
    let unassigned = max_histograms as u8;

    let inp = core::mem::take(histograms);
    let n = inp.len();
    let mut symbols: Vec<u8> = vec![unassigned; n];

    // Pre-compute bit costs for inputs.
    let mut in_costs = vec![0.0f32; n];
    let mut dists = vec![f32::MAX; n];
    let mut largest_idx = 0usize;
    let mut largest_count = 0u32;

    for (i, (((hist, symbol), dist), cost)) in inp
        .iter()
        .zip(symbols.iter_mut())
        .zip(dists.iter_mut())
        .zip(in_costs.iter_mut())
        .enumerate()
    {
        let total_count = hist.total_count;
        if total_count == 0 {
            *symbol = 0;
            *dist = 0.0;
            continue;
        }

        *cost = histogram_bit_cost(hist);
        if total_count > largest_count {
            largest_count = total_count;
            largest_idx = i;
        }
    }

    let mut out: Vec<Histogram> = Vec::with_capacity(max_histograms);
    let mut out_costs: Vec<f32> = Vec::with_capacity(max_histograms);
    let mut scratch = [0u32; ALPHABET_SIZE];

    const MIN_DISTANCE_FOR_DISTINCT: f32 = 64.0;
    while out.len() < max_histograms {
        let symbol = out.len() as u8;
        symbols[largest_idx] = symbol;
        out.push(inp[largest_idx].clone());
        out_costs.push(in_costs[largest_idx]);
        dists[largest_idx] = 0.0;

        let last_idx = out.len() - 1;
        let last_hist = &out[last_idx];
        let last_cost = out_costs[last_idx];
        let mut next_largest_idx = 0usize;
        let mut next_largest_dist = dists[0];

        for (i, ((hist, &in_cost), dist)) in inp
            .iter()
            .zip(in_costs.iter())
            .zip(dists.iter_mut())
            .enumerate()
        {
            if *dist == 0.0 {
                continue;
            }

            let d = histogram_distance(hist, last_hist, in_cost, last_cost, &mut scratch);
            if d < *dist {
                *dist = d;
            }
            if *dist > next_largest_dist {
                next_largest_dist = *dist;
                next_largest_idx = i;
            }
        }

        largest_idx = next_largest_idx;
        if next_largest_dist < MIN_DISTANCE_FOR_DISTINCT {
            break;
        }
    }

    // Assign remaining inputs to the closest cluster, merging.
    for ((hist, &in_cost), symbol) in inp.iter().zip(in_costs.iter()).zip(symbols.iter_mut()) {
        if *symbol != unassigned {
            continue;
        }

        let mut best = 0usize;
        let mut best_dist = histogram_distance(hist, &out[0], in_cost, out_costs[0], &mut scratch);
        for (j, (candidate, &candidate_cost)) in
            out.iter().zip(out_costs.iter()).enumerate().skip(1)
        {
            let d = histogram_distance(hist, candidate, in_cost, candidate_cost, &mut scratch);
            if d < best_dist {
                best = j;
                best_dist = d;
            }
        }

        let best_hist = &mut out[best];
        histogram_add(best_hist, hist);
        out_costs[best] = histogram_bit_cost(best_hist);
        *symbol = best as u8;
    }

    // Reindex so new symbols come in increasing order, matching HistogramReindex.
    let mut remap = [UNMAPPED; CLUSTERS_LIMIT];
    let mut tmp: Vec<Option<Histogram>> = out.into_iter().map(Some).collect();
    let mut reordered: Vec<Histogram> = Vec::with_capacity(tmp.len());
    let mut next_index = 0u8;

    for &symbol in &symbols {
        debug_assert_ne!(symbol, unassigned);
        let old_index = symbol as usize;
        debug_assert!(old_index < tmp.len());

        let mapped = &mut remap[old_index];
        if *mapped == UNMAPPED {
            *mapped = next_index;
            reordered.push(
                tmp[old_index]
                    .take()
                    .expect("cluster must be reindexed once"),
            );
            next_index += 1;
        }
    }

    *histograms = reordered;
    context_map.clear();
    context_map.reserve(symbols.len());
    context_map.extend(symbols.iter().map(|&symbol| remap[symbol as usize]));
}
