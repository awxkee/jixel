/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! Lossy palette of an extra channel: the fewest sample levels whose squared
//! error stays within a budget, coded losslessly through the Palette path.
//!
//! The levels are an exact 1-D optimal quantization: contiguous cells over
//! the sorted distinct values, each reconstructed at its rounded weighted
//! mean, with a min-error dynamic program per level count. The error is
//! weighted per sample ([`error_weights`]): levels show as contours on smooth
//! partial ramps, where a squeeze barely errs, so error there counts more.
//! Zero is never merged and no other value reconstructs to zero, so the
//! invisible-pixel set (which the color coding reads) is unchanged; a black
//! plate (`keep_full`) keeps its no-ink value the same way.

use crate::encode_image::AlphaPlane;

/// Most distinct nonzero values the exhaustive program runs over
/// (O(values² · levels)); deeper planes take the divide-and-conquer one
/// (O(values · log values · levels)).
const MAX_VALUES: usize = 256;

/// Most nonzero levels of a deep plane: its palette holds 256 entries.
const MAX_DEEP_LEVELS: usize = 255;

/// Share of the squeeze's weighted squared error the lossy palette may
/// spend. Equal error is not equal quality: at 1.0 levels still band text
/// and UI alpha at high distances (SS2 BD up to +9.5 on one image in 32),
/// at 0.5 none of the 32 loses.
pub(crate) const LOSSY_PALETTE_ERROR: f64 = 0.5;

/// Extra weight of the squared error on smooth partial samples.
const SMOOTH_WEIGHT: f32 = 4.0;
/// Largest neighbor difference of a smooth sample, in 8-bit units.
const SMOOTH_STEP: i32 = 16;

/// Per-sample weight of the squared error: `1 + SMOOTH_WEIGHT` on partial
/// samples (neither zero nor full) whose vertical and horizontal central
/// differences are both at most `SMOOTH_STEP`, 1 elsewhere.
pub(crate) fn error_weights(plane: &AlphaPlane, xsize: usize, ysize: usize) -> Vec<f32> {
    let max = (1i32 << plane.bits()) - 1;
    let step = SMOOTH_STEP * max / 255;
    let at = |x: usize, y: usize| plane.get_i32(y * xsize + x);
    let mut weights = vec![1.0f32; xsize * ysize];
    for y in 0..ysize {
        let (up, down) = (y.saturating_sub(1), (y + 1).min(ysize - 1));
        for x in 0..xsize {
            let v = at(x, y);
            if v == 0 || v == max {
                continue;
            }
            let (left, right) = (x.saturating_sub(1), (x + 1).min(xsize - 1));
            let dv = (at(x, up) - at(x, down)).abs();
            let dh = (at(left, y) - at(right, y)).abs();
            if dv.max(dh) <= step {
                weights[y * xsize + x] += SMOOTH_WEIGHT;
            }
        }
    }
    weights
}

/// Running sums over the sorted distinct values: weight, Σwv, Σwv².
struct Sums {
    n: Vec<f64>,
    s1: Vec<f64>,
    s2: Vec<f64>,
    max: f64,
    /// The levels' grid: the greatest common divisor of the values.
    step: f64,
}

impl Sums {
    fn new(vals: &[u32], counts: &[f64], max: u32) -> Self {
        let mut n = vec![0.0; vals.len() + 1];
        let mut s1 = n.clone();
        let mut s2 = n.clone();

        let (mut count, mut sum, mut sum_sq) = (0.0, 0.0, 0.0);

        let outputs = n.iter_mut().zip(&mut s1).zip(&mut s2).skip(1);

        for (((n, s1), s2), (&v, &c)) in outputs.zip(vals.iter().zip(counts)) {
            let v = f64::from(v);

            count += c;
            sum += c * v;
            sum_sq += c * v * v;

            *n = count;
            *s1 = sum;
            *s2 = sum_sq;
        }
        let step = vals.iter().fold(0, |g, &v| gcd(g, v)).max(1);
        Self {
            n,
            s1,
            s2,
            max: f64::from(max - max % step),
            step: f64::from(step),
        }
    }

    /// The level of cell `i..=j` and its squared error.
    #[inline]
    fn cell(&self, i: usize, j: usize) -> (f64, f64) {
        let n = self.n[j + 1] - self.n[i];
        let s1 = self.s1[j + 1] - self.s1[i];
        let s2 = self.s2[j + 1] - self.s2[i];
        let r = ((s1 / n / self.step).round() * self.step).clamp(self.step, self.max);
        (r, (s2 - 2.0 * r * s1 + r * r * n).max(0.0))
    }
}

/// One more cell for ends `lo..=hi` of a deep plane: the least error of
/// `vals[..=j]` with the last cell starting in `first..=last` (its earliest
/// start is `k`, one value per earlier cell), recursing on the middle end's
/// best start.
#[allow(clippy::too_many_arguments)]
fn split_points(
    sums: &Sums,
    cost: &[f64],
    k: usize,
    lo: usize,
    hi: usize,
    first: usize,
    last: usize,
    next: &mut [f64],
    start: &mut [u16],
) {
    let j = (lo + hi) / 2;
    let mut best = (f64::INFINITY, first);
    for i in first.max(k)..=last.min(j) {
        let c = cost[i - 1] + sums.cell(i, j).1;
        if c < best.0 {
            best = (c, i);
        }
    }
    (next[j], start[j]) = (best.0, best.1 as u16);
    if lo < j {
        split_points(sums, cost, k, lo, j - 1, first, best.1, next, start);
    }
    if j < hi {
        split_points(sums, cost, k, j + 1, hi, best.1, last, next, start);
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The plane with the fewest levels whose squared error, weighted by
/// `weights`, is at most `budget`, or None when the plane is not integer,
/// needs every one of its values, or (deep planes) more than
/// `MAX_DEEP_LEVELS` levels. The levels stay on the values' own grid (their
/// greatest common divisor), so a plane scaled onto a wider lattice keeps a
/// divisible palette.
pub(crate) fn fewest_levels(
    plane: &AlphaPlane,
    weights: &[f32],
    budget: f64,
    keep_full: bool,
) -> Option<AlphaPlane> {
    let max = match plane {
        AlphaPlane::U8(_) => 255u32,
        AlphaPlane::U16 { bits, .. } => (1u32 << bits) - 1,
        AlphaPlane::F32(_) => return None,
    };
    let mut hist = vec![0f64; max as usize + 1];
    for (i, &w) in weights.iter().enumerate().take(plane.len()) {
        hist[plane.get_i32(i) as usize] += f64::from(w);
    }
    // Zero never merges (alpha: the invisible set); with `keep_full`
    // neither does the full value (K: no ink). A cell of the other values
    // reconstructs strictly inside them, so no level lands on either.
    let top = if keep_full { max - 1 } else { max };
    let (vals, counts): (Vec<u32>, Vec<f64>) = (1..=top)
        .filter(|&v| hist[v as usize] > 0.0)
        .map(|v| (v, hist[v as usize]))
        .unzip();
    let m = vals.len();
    let deep = m > MAX_VALUES;
    if m < 2 || m > usize::from(u16::MAX) {
        return None;
    }
    let sums = Sums::new(&vals, &counts, max);
    // cost[j]: least error of vals[..=j] in the current number of cells;
    // starts[k][j]: first value of the last cell in that partition.
    let mut cost: Vec<f64> = (0..m).map(|j| sums.cell(0, j).1).collect();
    let mut starts: Vec<Vec<u16>> = vec![vec![0; m]];
    while cost[m - 1] > budget {
        if starts.len() + 1 >= m || (deep && starts.len() >= MAX_DEEP_LEVELS) {
            return None;
        }
        let k = starts.len();
        let mut next = vec![f64::INFINITY; m];
        let mut start = vec![0u16; m];
        if deep {
            split_points(&sums, &cost, k, k, m - 1, k, m - 1, &mut next, &mut start);
        } else {
            for j in k..m {
                for i in k..=j {
                    let c = cost[i - 1] + sums.cell(i, j).1;
                    if c < next[j] {
                        next[j] = c;
                        start[j] = i as u16;
                    }
                }
            }
        }
        cost = next;
        starts.push(start);
    }
    let mut lut: Vec<u32> = (0..=max).collect();
    let mut j = m - 1;
    for k in (0..starts.len()).rev() {
        let i = starts[k][j] as usize;
        let level = sums.cell(i, j).0 as u32;
        for v in &mut lut[vals[i] as usize..=vals[j] as usize] {
            *v = level;
        }
        if i == 0 {
            break;
        }
        j = i - 1;
    }
    Some(match plane {
        AlphaPlane::U8(data) => {
            AlphaPlane::U8(data.iter().map(|&v| lut[v as usize] as u8).collect())
        }
        AlphaPlane::U16 { data, bits } => AlphaPlane::U16 {
            data: data.iter().map(|&v| lut[v as usize] as u16).collect(),
            bits: *bits,
        },
        AlphaPlane::F32(_) => unreachable!(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(a: &AlphaPlane, b: &AlphaPlane, weights: &[f32]) -> f64 {
        (0..a.len())
            .map(|i| f64::from(weights[i]) * f64::from(a.get_i32(i) - b.get_i32(i)).powi(2))
            .sum()
    }

    fn levels(p: &AlphaPlane) -> usize {
        let mut v: Vec<i32> = (0..p.len()).map(|i| p.get_i32(i)).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Exhaustive fewest-cells search over contiguous partitions.
    fn brute(vals: &[u32], counts: &[f64], max: u32, budget: f64) -> usize {
        let sums = Sums::new(vals, counts, max);
        let m = vals.len();
        for k in 1..=m {
            // Cut masks with k - 1 cuts among m - 1 gaps.
            for mask in 0u32..(1 << (m - 1)) {
                if mask.count_ones() as usize != k - 1 {
                    continue;
                }
                let (mut i, mut err) = (0, 0.0);
                for j in 0..m {
                    if j == m - 1 || mask & (1 << j) != 0 {
                        err += sums.cell(i, j).1;
                        i = j + 1;
                    }
                }
                if err <= budget {
                    return k;
                }
            }
        }
        m
    }

    #[test]
    fn fewest_levels_matches_exhaustive_partitions_and_keeps_zero() {
        let mut seed = 0x2545f491u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for _ in 0..60 {
            let n = 400;
            let data: Vec<u8> = (0..n)
                .map(|_| match rnd() % 4 {
                    0 => 0,
                    1 => 255,
                    _ => (rnd() % 12 * 21 + 1) as u8,
                })
                .collect();
            let plane = AlphaPlane::U8(data.clone());
            let weights: Vec<f32> = (0..n).map(|_| 1.0 + (rnd() % 3) as f32).collect();
            let mut hist = [0f64; 256];
            for (&v, &w) in data.iter().zip(&weights) {
                hist[v as usize] += f64::from(w);
            }
            let (vals, counts): (Vec<u32>, Vec<f64>) = (1..256u32)
                .filter(|&v| hist[v as usize] > 0.0)
                .map(|v| (v, hist[v as usize]))
                .unzip();
            let budget = f64::from(rnd() % 40_000);
            let want = brute(&vals, &counts, 255, budget);
            match fewest_levels(&plane, &weights, budget, false) {
                Some(q) => {
                    assert!(sse(&plane, &q, &weights) <= budget + 1e-6);
                    let zeros = data.iter().filter(|&&v| v == 0).count();
                    assert_eq!(zeros, (0..q.len()).filter(|&i| q.get_i32(i) == 0).count());
                    let zero_level = usize::from(zeros > 0);
                    assert_eq!(levels(&q), want + zero_level);
                }
                None => assert_eq!(want, vals.len()),
            }
        }
    }

    /// Fewest cells by the exhaustive program, at any number of values.
    fn exhaustive(vals: &[u32], counts: &[f64], max: u32, budget: f64) -> usize {
        let sums = Sums::new(vals, counts, max);
        let m = vals.len();
        let mut cost: Vec<f64> = (0..m).map(|j| sums.cell(0, j).1).collect();
        let mut k = 1;
        while cost[m - 1] > budget {
            let mut next = vec![f64::INFINITY; m];
            for j in k..m {
                for i in k..=j {
                    next[j] = next[j].min(cost[i - 1] + sums.cell(i, j).1);
                }
            }
            cost = next;
            k += 1;
        }
        k
    }

    #[test]
    fn deep_levels_match_the_exhaustive_program() {
        let mut seed = 0x9e3779b9u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for bits in [10, 12, 16] {
            let max = (1u32 << bits) - 1;
            let n = 6000;
            // Mostly empty or full, antialiased edges in between.
            let data: Vec<u16> = (0..n)
                .map(|_| match rnd() % 4 {
                    0 => 0,
                    1 => max as u16,
                    _ => (rnd() % max + 1) as u16,
                })
                .collect();
            let plane = AlphaPlane::U16 {
                data: data.clone(),
                bits,
            };
            let weights: Vec<f32> = (0..n).map(|_| 1.0 + (rnd() % 3) as f32).collect();
            let mut hist = vec![0f64; max as usize + 1];
            for (&v, &w) in data.iter().zip(&weights) {
                hist[v as usize] += f64::from(w);
            }
            let (vals, counts): (Vec<u32>, Vec<f64>) = (1..=max)
                .filter(|&v| hist[v as usize] > 0.0)
                .map(|v| (v, hist[v as usize]))
                .unzip();
            assert!(vals.len() > MAX_VALUES, "a deep plane");
            for share in [1e-2, 1e-3, 1e-4] {
                let budget = sums_error(&vals, &counts, max) * share;
                let q = fewest_levels(&plane, &weights, budget, false).expect("quantized");
                assert!(sse(&plane, &q, &weights) <= budget * (1.0 + 1e-9));
                let zeros = data.iter().filter(|&&v| v == 0).count();
                assert_eq!(zeros, (0..q.len()).filter(|&i| q.get_i32(i) == 0).count());
                assert_eq!(levels(&q), exhaustive(&vals, &counts, max, budget) + 1);
            }
        }
    }

    /// Squared error of one cell over every value.
    fn sums_error(vals: &[u32], counts: &[f64], max: u32) -> f64 {
        Sums::new(vals, counts, max).cell(0, vals.len() - 1).1
    }

    #[test]
    fn keep_full_preserves_the_no_ink_value_and_zero() {
        // A plate of near-full values around 255 and near-empty ones around 0.
        let data: Vec<u8> = (0..4096u32)
            .map(|i| match i % 8 {
                0 => 255,
                1 => 254,
                2 => 253,
                3 => 0,
                4 => 1,
                5 => 2,
                _ => 128 + (i % 3) as u8,
            })
            .collect();
        let plane = AlphaPlane::U8(data.clone());
        let weights = vec![1.0f32; data.len()];
        let loose = fewest_levels(&plane, &weights, 4096.0 * 4e6, false).expect("quantized");
        assert!((0..loose.len()).any(|i| data[i] == 255 && loose.get_i32(i) != 255));
        let kept = fewest_levels(&plane, &weights, 4096.0 * 4e6, true).expect("quantized");
        for i in 0..kept.len() {
            assert_eq!(kept.get_i32(i) == 255, data[i] == 255);
            assert_eq!(kept.get_i32(i) == 0, data[i] == 0);
        }
        assert!(levels(&kept) <= levels(&loose) + 1);
    }

    #[test]
    fn fewest_levels_stay_on_the_values_grid() {
        // 8-bit samples on the 16-bit lattice: every level a multiple of 257.
        let data: Vec<u16> = (0..4096).map(|i| (i % 200 + 1) as u16 * 257).collect();
        let plane = AlphaPlane::U16 { data, bits: 16 };
        let weights = vec![1.0; plane.len()];
        let q = fewest_levels(&plane, &weights, 4096.0 * 4e6, false).expect("quantized");
        assert!((0..q.len()).all(|i| q.get_i32(i) % 257 == 0 && q.get_i32(i) > 0));
        assert!(levels(&q) < 200);
    }

    #[test]
    fn fewest_levels_keeps_the_bit_depth_of_wide_planes() {
        let data: Vec<u16> = (0..4096).map(|i| (i % 64 * 16) as u16).collect();
        let plane = AlphaPlane::U16 { data, bits: 10 };
        let weights = vec![1.0; plane.len()];
        let q = fewest_levels(&plane, &weights, 4096.0 * 400.0, false).expect("quantized");
        assert!(matches!(q, AlphaPlane::U16 { bits: 10, .. }));
        assert!(levels(&q) < 64);
        assert!(sse(&plane, &q, &weights) <= 4096.0 * 400.0);
    }

    #[test]
    fn error_weights_mark_smooth_partial_ramps_only() {
        // Columns: a hard 0 → 255 edge, then a slow ramp.
        let (w, h) = (12, 3);
        let row = [0u8, 0, 255, 255, 100, 104, 108, 112, 116, 120, 255, 255];
        let plane = AlphaPlane::U8((0..h).flat_map(|_| row).collect());
        let weights = error_weights(&plane, w, h);
        for y in 0..h {
            for x in 0..w {
                let smooth = (5..9).contains(&x);
                assert_eq!(weights[y * w + x] > 1.0, smooth, "x {x} y {y}");
            }
        }
    }
}
