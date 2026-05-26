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

fn histogram_bit_cost(h: &Histogram) -> f32 {
    if h.total_count == 0 {
        return 0.0;
    }
    let mut depths = [0u8; ALPHABET_SIZE];
    let counts32: [u32; ALPHABET_SIZE] = h.counts;
    create_huffman_tree(&counts32, 15, &mut depths);
    let mut cost: f32 = 0.0;
    for i in 0..ALPHABET_SIZE {
        cost += h.counts[i] as f32 * depths[i] as f32;
    }
    cost
}

fn histogram_add(a: &mut Histogram, b: &Histogram) {
    for i in 0..ALPHABET_SIZE {
        a.counts[i] += b.counts[i];
    }
    a.total_count += b.total_count;
}

fn histogram_distance(a: &Histogram, b: &Histogram, a_cost: f32, b_cost: f32) -> f32 {
    if a.total_count == 0 || b.total_count == 0 {
        return 0.0;
    }
    let mut combined = a.clone();
    histogram_add(&mut combined, b);
    histogram_bit_cost(&combined) - a_cost - b_cost
}

/// Cluster `histograms` in place down to at most 8 distinct ones; produce
/// `context_map[i] = cluster index for histogram i`.
pub(crate) fn cluster_histograms(histograms: &mut Vec<Histogram>, context_map: &mut Vec<u8>) {
    if histograms.len() <= 1 {
        context_map.clear();
        context_map.resize(histograms.len(), 0);
        return;
    }
    const CLUSTERS_LIMIT: usize = 8;
    let max_histograms = CLUSTERS_LIMIT.min(histograms.len());

    let inp: Vec<Histogram> = histograms.clone();
    let n = inp.len();
    let mut symbols: Vec<u32> = vec![max_histograms as u32; n];

    // Pre-compute bit costs for inputs.
    let mut in_costs = vec![0.0f32; n];
    let mut dists = vec![f32::MAX; n];
    let mut largest_idx = 0usize;
    for i in 0..n {
        if inp[i].total_count == 0 {
            symbols[i] = 0;
            dists[i] = 0.0;
            continue;
        }
        in_costs[i] = histogram_bit_cost(&inp[i]);
        if inp[i].total_count > inp[largest_idx].total_count {
            largest_idx = i;
        }
    }

    let mut out: Vec<Histogram> = Vec::new();
    let mut out_costs: Vec<f32> = Vec::new();

    const MIN_DISTANCE_FOR_DISTINCT: f32 = 64.0;
    while out.len() < max_histograms {
        symbols[largest_idx] = out.len() as u32;
        out.push(inp[largest_idx].clone());
        out_costs.push(in_costs[largest_idx]);
        dists[largest_idx] = 0.0;
        largest_idx = 0;
        let last_idx = out.len() - 1;
        for i in 0..n {
            if dists[i] == 0.0 {
                continue;
            }
            let d = histogram_distance(&inp[i], &out[last_idx], in_costs[i], out_costs[last_idx]);
            if d < dists[i] {
                dists[i] = d;
            }
            if dists[i] > dists[largest_idx] {
                largest_idx = i;
            }
        }
        if dists[largest_idx] < MIN_DISTANCE_FOR_DISTINCT {
            break;
        }
    }

    // Assign remaining inputs to the closest cluster, merging.
    for i in 0..n {
        if symbols[i] != max_histograms as u32 {
            continue;
        }
        let mut best = 0usize;
        let mut best_dist = histogram_distance(&inp[i], &out[0], in_costs[i], out_costs[0]);
        for j in 1..out.len() {
            let d = histogram_distance(&inp[i], &out[j], in_costs[i], out_costs[j]);
            if d < best_dist {
                best = j;
                best_dist = d;
            }
        }
        let mut merged = out[best].clone();
        histogram_add(&mut merged, &inp[i]);
        let new_cost = histogram_bit_cost(&merged);
        out[best] = merged;
        out_costs[best] = new_cost;
        symbols[i] = best as u32;
    }

    // Reindex so new symbols come in increasing order, matching HistogramReindex.
    use std::collections::HashMap;
    let mut new_index: HashMap<u32, u32> = HashMap::new();
    let tmp = out.clone();
    let mut next_index = 0u32;
    let mut reordered: Vec<Histogram> = Vec::new();
    for &s in &symbols {
        if !new_index.contains_key(&s) {
            new_index.insert(s, next_index);
            reordered.push(tmp[s as usize].clone());
            next_index += 1;
        }
    }
    *histograms = reordered;
    context_map.clear();
    for s in &symbols {
        context_map.push(new_index[s] as u8);
    }
}
