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
) -> f32 {
    if a.total_count == 0 || b.total_count == 0 {
        return 0.0;
    }

    *scratch = a.counts;
    add_counts(scratch, &b.counts);
    counts_bit_cost(scratch, a.total_count + b.total_count) - a_cost - b_cost
}

#[inline]
fn histogram_merge_increment(
    input: &Histogram,
    cluster: &Histogram,
    cluster_cost: f32,
    scratch: &mut [u32; ALPHABET_SIZE],
) -> f32 {
    *scratch = input.counts;
    add_counts(scratch, &cluster.counts);
    counts_bit_cost(scratch, input.total_count + cluster.total_count) - cluster_cost
}

/// Cluster `histograms` in place down to at most 64 distinct ones; produce
/// `context_map[i] = cluster index for histogram i`.
pub(crate) fn cluster_histograms(histograms: &mut Vec<Histogram>, context_map: &mut Vec<u8>) {
    cluster_histograms_inner(histograms, context_map, false);
}

pub(crate) fn cluster_histograms_refined(
    histograms: &mut Vec<Histogram>,
    context_map: &mut Vec<u8>,
) {
    cluster_histograms_inner(histograms, context_map, true);
}

fn cluster_histograms_inner(
    histograms: &mut Vec<Histogram>,
    context_map: &mut Vec<u8>,
    refined: bool,
) {
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

    if !refined {
        // Low-effort path: preserve the original single-pass assignment. It is
        // intentionally cheap and is used by Fast and the many small lossy
        // entropy bundles.
        for ((hist, &in_cost), symbol) in inp.iter().zip(in_costs.iter()).zip(symbols.iter_mut()) {
            if *symbol != unassigned {
                continue;
            }
            let mut best = 0usize;
            let mut best_dist =
                histogram_distance(hist, &out[0], in_cost, out_costs[0], &mut scratch);
            for (j, (candidate, &candidate_cost)) in
                out.iter().zip(out_costs.iter()).enumerate().skip(1)
            {
                let d = histogram_distance(hist, candidate, in_cost, candidate_cost, &mut scratch);
                if d < best_dist {
                    best = j;
                    best_dist = d;
                }
            }
            histogram_add(&mut out[best], hist);
            out_costs[best] = histogram_bit_cost(&out[best]);
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
            let mut best_dist =
                histogram_distance(hist, &out[0], in_cost, out_costs[0], &mut scratch);
            for (j, (candidate, &candidate_cost)) in
                out.iter().zip(out_costs.iter()).enumerate().skip(1)
            {
                let d = histogram_distance(hist, candidate, in_cost, candidate_cost, &mut scratch);
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
            *cost = histogram_bit_cost(hist);
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
                let remove_delta = histogram_bit_cost(&old_without) - out_costs[old];

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
                    out_costs[old] = histogram_bit_cost(&out[old]);
                    out_costs[best] = histogram_bit_cost(&out[best]);
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
