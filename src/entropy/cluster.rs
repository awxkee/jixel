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
use super::huffman_tree::{HuffmanNode, create_huffman_tree};
use super::prefix_code::ALPHABET_SIZE;
use crate::util::heap_array;

pub(crate) const CLUSTERS_LIMIT: usize = 64;

pub(crate) struct FixedClusterScratch<const MAX_CONTEXTS: usize> {
    symbols: Box<[u8; MAX_CONTEXTS]>,
    in_costs: Box<[f32; MAX_CONTEXTS]>,
    dists: Box<[f32; MAX_CONTEXTS]>,
    clusters: Box<[Histogram; CLUSTERS_LIMIT]>,
    cluster_costs: Box<[f32; CLUSTERS_LIMIT]>,
    reordered: Box<[Histogram; CLUSTERS_LIMIT]>,
}

impl<const MAX_CONTEXTS: usize> Default for FixedClusterScratch<MAX_CONTEXTS> {
    fn default() -> Self {
        Self {
            symbols: heap_array(0),
            in_costs: heap_array(0.0),
            dists: heap_array(0.0),
            clusters: heap_array(Histogram::new()),
            cluster_costs: heap_array(0.0),
            reordered: heap_array(Histogram::new()),
        }
    }
}

#[inline]
fn counts_bit_cost(
    counts: &[u32; ALPHABET_SIZE],
    total_count: u32,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> f32 {
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
    create_huffman_tree(counts, 15, &mut depths, huffman_pool);

    let mut cost = 0.0f32;
    for (&count, &depth) in counts.iter().zip(depths.iter()) {
        cost += count as f32 * depth as f32;
    }
    cost
}

#[inline]
fn histogram_bit_cost(h: &Histogram, huffman_pool: &mut Vec<HuffmanNode>) -> f32 {
    counts_bit_cost(&h.counts, h.total_count, huffman_pool)
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
fn histogram_sub(a: &mut Histogram, b: &Histogram) {
    for (dst, &src) in a.counts.iter_mut().zip(b.counts.iter()) {
        *dst -= src;
    }
    a.total_count -= b.total_count;
}

#[inline]
fn histogram_distance(
    a: &Histogram,
    b: &Histogram,
    a_cost: f32,
    b_cost: f32,
    scratch: &mut [u32; ALPHABET_SIZE],
    huffman_pool: &mut Vec<HuffmanNode>,
) -> f32 {
    if a.total_count == 0 || b.total_count == 0 {
        return 0.0;
    }

    *scratch = a.counts;
    add_counts(scratch, &b.counts);
    counts_bit_cost(scratch, a.total_count + b.total_count, huffman_pool) - a_cost - b_cost
}

#[inline]
fn histogram_merge_increment(
    input: &Histogram,
    cluster: &Histogram,
    cluster_cost: f32,
    scratch: &mut [u32; ALPHABET_SIZE],
    huffman_pool: &mut Vec<HuffmanNode>,
) -> f32 {
    *scratch = input.counts;
    add_counts(scratch, &cluster.counts);
    counts_bit_cost(
        scratch,
        input.total_count + cluster.total_count,
        huffman_pool,
    ) - cluster_cost
}

/// Cluster `histograms` in place down to at most 64 distinct ones; produce
/// `context_map[i] = cluster index for histogram i`.
pub(crate) fn cluster_histograms(
    histograms: &mut Vec<Histogram>,
    context_map: &mut Vec<u8>,
    huffman_pool: &mut Vec<HuffmanNode>,
) {
    cluster_histograms_inner(histograms, context_map, false, huffman_pool);
}

pub(crate) fn cluster_histograms_fixed<const MAX_CONTEXTS: usize>(
    histograms: &mut [Histogram],
    context_map: &mut [u8],
    refined: bool,
    fixed: &mut FixedClusterScratch<MAX_CONTEXTS>,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> usize {
    let n = histograms.len();
    assert!(n <= MAX_CONTEXTS);
    assert!(context_map.len() >= n);
    if n <= 1 {
        context_map[..n].fill(0);
        return n;
    }

    const UNMAPPED: u8 = u8::MAX;
    let max_histograms = CLUSTERS_LIMIT.min(n);
    let unassigned = max_histograms as u8;
    let symbols = &mut fixed.symbols[..n];
    symbols.fill(unassigned);
    let in_costs = &mut fixed.in_costs[..n];
    let dists = &mut fixed.dists[..n];
    dists.fill(f32::MAX);

    let mut largest_idx = 0usize;
    let mut largest_count = 0u32;
    for (i, (((hist, symbol), dist), cost)) in histograms
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
        *cost = histogram_bit_cost(hist, huffman_pool);
        if total_count > largest_count {
            largest_count = total_count;
            largest_idx = i;
        }
    }
    if largest_count == 0 {
        histograms[0] = Histogram::new();
        context_map[..n].fill(0);
        return 1;
    }

    let clusters = &mut fixed.clusters;
    let cluster_costs = &mut fixed.cluster_costs;
    let mut num_clusters = 0usize;
    let mut counts_scratch = [0u32; ALPHABET_SIZE];

    const MIN_DISTANCE_FOR_DISTINCT: f32 = 64.0;
    while num_clusters < max_histograms {
        let symbol = num_clusters as u8;
        symbols[largest_idx] = symbol;
        clusters[num_clusters] = histograms[largest_idx].clone();
        cluster_costs[num_clusters] = in_costs[largest_idx];
        dists[largest_idx] = 0.0;
        num_clusters += 1;

        let last_idx = num_clusters - 1;
        let last_hist = &clusters[last_idx];
        let last_cost = cluster_costs[last_idx];
        let mut next_largest_idx = 0usize;
        let mut next_largest_dist = dists[0];
        for (i, ((hist, &in_cost), dist)) in histograms
            .iter()
            .zip(in_costs.iter())
            .zip(dists.iter_mut())
            .enumerate()
        {
            if *dist == 0.0 {
                continue;
            }
            let distance = histogram_distance(
                hist,
                last_hist,
                in_cost,
                last_cost,
                &mut counts_scratch,
                huffman_pool,
            );
            if distance < *dist {
                *dist = distance;
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

    if !refined {
        for ((hist, &in_cost), symbol) in histograms
            .iter()
            .zip(in_costs.iter())
            .zip(symbols.iter_mut())
        {
            if *symbol != unassigned {
                continue;
            }
            let mut best = 0usize;
            let mut best_dist = histogram_distance(
                hist,
                &clusters[0],
                in_cost,
                cluster_costs[0],
                &mut counts_scratch,
                huffman_pool,
            );
            for candidate in 1..num_clusters {
                let distance = histogram_distance(
                    hist,
                    &clusters[candidate],
                    in_cost,
                    cluster_costs[candidate],
                    &mut counts_scratch,
                    huffman_pool,
                );
                if distance < best_dist {
                    best = candidate;
                    best_dist = distance;
                }
            }
            histogram_add(&mut clusters[best], hist);
            cluster_costs[best] = histogram_bit_cost(&clusters[best], huffman_pool);
            *symbol = best as u8;
        }
    } else {
        for ((hist, &in_cost), symbol) in histograms
            .iter()
            .zip(in_costs.iter())
            .zip(symbols.iter_mut())
        {
            if *symbol != unassigned {
                continue;
            }
            let mut best = 0usize;
            let mut best_dist = histogram_distance(
                hist,
                &clusters[0],
                in_cost,
                cluster_costs[0],
                &mut counts_scratch,
                huffman_pool,
            );
            for candidate in 1..num_clusters {
                let distance = histogram_distance(
                    hist,
                    &clusters[candidate],
                    in_cost,
                    cluster_costs[candidate],
                    &mut counts_scratch,
                    huffman_pool,
                );
                if distance < best_dist {
                    best = candidate;
                    best_dist = distance;
                }
            }
            *symbol = best as u8;
        }

        clusters[..num_clusters].fill(Histogram::new());
        for (hist, &symbol) in histograms.iter().zip(symbols.iter()) {
            histogram_add(&mut clusters[symbol as usize], hist);
        }
        for cluster in 0..num_clusters {
            cluster_costs[cluster] = histogram_bit_cost(&clusters[cluster], huffman_pool);
        }

        for _ in 0..2 {
            let mut changed = false;
            for (hist, symbol) in histograms.iter().zip(symbols.iter_mut()) {
                if hist.total_count == 0 {
                    continue;
                }
                let old = *symbol as usize;
                let mut old_without = clusters[old].clone();
                histogram_sub(&mut old_without, hist);
                let remove_delta =
                    histogram_bit_cost(&old_without, huffman_pool) - cluster_costs[old];

                let mut best = old;
                let mut best_delta = 0.0f32;
                for candidate in 0..num_clusters {
                    if candidate == old || clusters[candidate].total_count == 0 {
                        continue;
                    }
                    let add_delta = histogram_merge_increment(
                        hist,
                        &clusters[candidate],
                        cluster_costs[candidate],
                        &mut counts_scratch,
                        huffman_pool,
                    );
                    let delta = remove_delta + add_delta;
                    if delta < best_delta - 0.01 {
                        best_delta = delta;
                        best = candidate;
                    }
                }
                if best != old {
                    histogram_sub(&mut clusters[old], hist);
                    histogram_add(&mut clusters[best], hist);
                    cluster_costs[old] = histogram_bit_cost(&clusters[old], huffman_pool);
                    cluster_costs[best] = histogram_bit_cost(&clusters[best], huffman_pool);
                    *symbol = best as u8;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut compact = [UNMAPPED; CLUSTERS_LIMIT];
        let mut compact_len = 0usize;
        for (old, histogram) in clusters[..num_clusters].iter().enumerate() {
            if histogram.total_count != 0 {
                compact[old] = compact_len as u8;
                fixed.reordered[compact_len] = histogram.clone();
                compact_len += 1;
            }
        }
        for symbol in symbols.iter_mut() {
            *symbol = compact[*symbol as usize];
        }
        clusters[..compact_len].clone_from_slice(&fixed.reordered[..compact_len]);
        num_clusters = compact_len;
    }

    let mut remap = [UNMAPPED; CLUSTERS_LIMIT];
    let mut reordered_len = 0usize;
    for (i, &symbol) in symbols.iter().enumerate() {
        debug_assert_ne!(symbol, unassigned);
        let old_index = symbol as usize;
        debug_assert!(old_index < num_clusters);
        if remap[old_index] == UNMAPPED {
            remap[old_index] = reordered_len as u8;
            fixed.reordered[reordered_len] = clusters[old_index].clone();
            reordered_len += 1;
        }
        context_map[i] = remap[old_index];
    }
    histograms[..reordered_len].clone_from_slice(&fixed.reordered[..reordered_len]);
    reordered_len
}

fn cluster_histograms_inner(
    histograms: &mut Vec<Histogram>,
    context_map: &mut Vec<u8>,
    refined: bool,
    huffman_pool: &mut Vec<HuffmanNode>,
) {
    if histograms.len() <= 1 {
        context_map.clear();
        context_map.resize(histograms.len(), 0);
        return;
    }

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

        *cost = histogram_bit_cost(hist, huffman_pool);
        if total_count > largest_count {
            largest_count = total_count;
            largest_idx = i;
        }
    }
    if largest_count == 0 {
        *histograms = vec![Histogram::new()];
        context_map.clear();
        context_map.resize(n, 0);
        return;
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

            let d = histogram_distance(
                hist,
                last_hist,
                in_cost,
                last_cost,
                &mut scratch,
                huffman_pool,
            );
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

    if !refined {
        // Low-effort path: preserve the original single-pass assignment. It is
        // intentionally cheap and is used by Fast and the many small lossy
        // entropy bundles.
        for ((hist, &in_cost), symbol) in inp.iter().zip(in_costs.iter()).zip(symbols.iter_mut()) {
            if *symbol != unassigned {
                continue;
            }
            let mut best = 0usize;
            let mut best_dist = histogram_distance(
                hist,
                &out[0],
                in_cost,
                out_costs[0],
                &mut scratch,
                huffman_pool,
            );
            for (j, (candidate, &candidate_cost)) in
                out.iter().zip(out_costs.iter()).enumerate().skip(1)
            {
                let d = histogram_distance(
                    hist,
                    candidate,
                    in_cost,
                    candidate_cost,
                    &mut scratch,
                    huffman_pool,
                );
                if d < best_dist {
                    best = j;
                    best_dist = d;
                }
            }
            histogram_add(&mut out[best], hist);
            out_costs[best] = histogram_bit_cost(&out[best], huffman_pool);
            *symbol = best as u8;
        }
    } else {
        // Assign remaining inputs against the immutable seed distributions. The
        // old implementation updated a cluster after every assignment, making the
        // result depend strongly on input order and allowing an early, broad
        // histogram to attract unrelated later contexts.
        for ((hist, &in_cost), symbol) in inp.iter().zip(in_costs.iter()).zip(symbols.iter_mut()) {
            if *symbol != unassigned {
                continue;
            }

            let mut best = 0usize;
            let mut best_dist = histogram_distance(
                hist,
                &out[0],
                in_cost,
                out_costs[0],
                &mut scratch,
                huffman_pool,
            );
            for (j, (candidate, &candidate_cost)) in
                out.iter().zip(out_costs.iter()).enumerate().skip(1)
            {
                let d = histogram_distance(
                    hist,
                    candidate,
                    in_cost,
                    candidate_cost,
                    &mut scratch,
                    huffman_pool,
                );
                if d < best_dist {
                    best = j;
                    best_dist = d;
                }
            }

            *symbol = best as u8;
        }

        // Pool the complete assigned distributions, then perform a few exact
        // cost-decreasing relocations. For a move A -> B we account for both
        // cost(A - input) and cost(B + input), which avoids the self-inclusion bias
        // of nearest-centroid clustering.
        out.fill(Histogram::new());
        for (hist, &symbol) in inp.iter().zip(symbols.iter()) {
            histogram_add(&mut out[symbol as usize], hist);
        }
        for (cost, hist) in out_costs.iter_mut().zip(out.iter()) {
            *cost = histogram_bit_cost(hist, huffman_pool);
        }

        for _ in 0..2 {
            let mut changed = false;
            for (hist, symbol) in inp.iter().zip(symbols.iter_mut()) {
                if hist.total_count == 0 {
                    continue;
                }
                let old = *symbol as usize;
                let mut old_without = out[old].clone();
                histogram_sub(&mut old_without, hist);
                let remove_delta = histogram_bit_cost(&old_without, huffman_pool) - out_costs[old];

                let mut best = old;
                let mut best_delta = 0.0f32;
                for candidate in 0..out.len() {
                    if candidate == old || out[candidate].total_count == 0 {
                        continue;
                    }
                    let add_delta = histogram_merge_increment(
                        hist,
                        &out[candidate],
                        out_costs[candidate],
                        &mut scratch,
                        huffman_pool,
                    );
                    let delta = remove_delta + add_delta;
                    if delta < best_delta - 0.01 {
                        best_delta = delta;
                        best = candidate;
                    }
                }
                if best != old {
                    histogram_sub(&mut out[old], hist);
                    histogram_add(&mut out[best], hist);
                    out_costs[old] = histogram_bit_cost(&out[old], huffman_pool);
                    out_costs[best] = histogram_bit_cost(&out[best], huffman_pool);
                    *symbol = best as u8;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Drop clusters emptied by relocation and update assignments before the
        // stable first-occurrence reindex below.
        let mut compact = vec![UNMAPPED; out.len()];
        let mut compact_out = Vec::with_capacity(out.len());
        for (old, hist) in out.into_iter().enumerate() {
            if hist.total_count != 0 {
                compact[old] = compact_out.len() as u8;
                compact_out.push(hist);
            }
        }
        for symbol in &mut symbols {
            *symbol = compact[*symbol as usize];
        }
        out = compact_out;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(n: usize) -> Vec<Histogram> {
        (0..n)
            .map(|context| {
                let mut histogram = Histogram::new();
                if context % 11 != 0 {
                    for symbol in 0..ALPHABET_SIZE {
                        let count = ((context * 17 + symbol * 13 + context * symbol) % 19) as u32;
                        histogram.counts[symbol] = count;
                        histogram.total_count += count;
                    }
                }
                histogram
            })
            .collect()
    }

    fn assert_fixed_matches_allocating(n: usize, refined: bool) {
        let mut expected = inputs(n);
        let mut expected_map = Vec::new();
        let mut expected_pool = Vec::with_capacity(1024);
        cluster_histograms_inner(
            &mut expected,
            &mut expected_map,
            refined,
            &mut expected_pool,
        );

        let mut actual = inputs(n);
        let mut actual_map = vec![0; n];
        let mut actual_pool = Vec::with_capacity(1024);
        let mut scratch = FixedClusterScratch::<221>::default();
        let actual_len = cluster_histograms_fixed(
            &mut actual,
            &mut actual_map,
            refined,
            &mut scratch,
            &mut actual_pool,
        );

        assert_eq!(actual_len, expected.len());
        assert_eq!(actual_map, expected_map);
        for (actual, expected) in actual[..actual_len].iter().zip(expected.iter()) {
            assert_eq!(actual.total_count, expected.total_count);
            assert_eq!(actual.counts, expected.counts);
        }
    }

    #[test]
    fn fixed_clustering_matches_allocating_path() {
        for n in [0, 1, 2, 7, 64, 65, 128, 221] {
            assert_fixed_matches_allocating(n, false);
            assert_fixed_matches_allocating(n, true);
        }
    }
}
