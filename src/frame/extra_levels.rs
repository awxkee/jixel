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
use std::ops::{Add, Div, Mul, Sub};

/// Most distinct nonzero values the exhaustive program runs over
/// (O(values² · levels)); deeper planes take the divide-and-conquer one
/// (O(values · log values · levels)).
const MAX_VALUES: usize = 256;

/// Most nonzero levels of a deep plane: its palette holds 256 entries.
const MAX_DEEP_LEVELS: usize = 255;

/// Extra weight of the squared error on smooth partial samples.
const SMOOTH_WEIGHT: u8 = 4;
/// Largest neighbor difference of a smooth sample, in 8-bit units.
const SMOOTH_STEP: i32 = 16;

/// Per-sample weight of the squared error: `1 + SMOOTH_WEIGHT` on partial
/// samples (neither zero nor full) whose vertical and horizontal central
/// differences are both at most `SMOOTH_STEP`, 1 elsewhere.
pub(crate) fn error_weights(plane: &AlphaPlane, xsize: usize, ysize: usize) -> Vec<u8> {
    let max = (1i32 << plane.bits()) - 1;
    let step = SMOOTH_STEP * max / 255;
    let at = |x: usize, y: usize| plane.get_i32(y * xsize + x);
    let mut weights = vec![1; xsize * ysize];
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

/// Allow half the squeeze's weighted squared error on an integer plane
/// after the decoder's clamp.
pub(crate) fn squeeze_budget(plane: &AlphaPlane, samples: &[i32], weights: &[u8]) -> u128 {
    fn squared_error<T: Copy + Into<i32>>(
        source: &[T],
        samples: &[i32],
        weights: &[u8],
        max: i32,
    ) -> u128 {
        let len = samples.len().min(weights.len());
        source[..len]
            .iter()
            .zip(samples)
            .zip(weights)
            .map(|((&s, &v), &w)| {
                let e = i64::from(v.clamp(0, max) - s.into());
                u128::from((e * e) as u64 * u64::from(w))
            })
            .sum()
    }

    let error = match plane {
        AlphaPlane::U8(data) => squared_error(data, samples, weights, 255),
        AlphaPlane::U16 { data, bits } => squared_error(data, samples, weights, (1i32 << bits) - 1),
        AlphaPlane::F32(_) => unreachable!("float extra channels are not squeezed"),
    };
    // Palette errors are integers, so rounding the budget down keeps the same cutoff.
    error / 2
}

trait Accumulator:
    Copy
    + From<u64>
    + Ord
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    const MAX: Self;
    fn as_level(self) -> u32;
}

impl Accumulator for u64 {
    const MAX: Self = u64::MAX;

    fn as_level(self) -> u32 {
        self as u32
    }
}

impl Accumulator for u128 {
    const MAX: Self = u128::MAX;

    fn as_level(self) -> u32 {
        self as u32
    }
}

/// Running sums over the sorted distinct values: weight, Σwv, Σwv².
struct Sums<T> {
    n: Vec<T>,
    s1: Vec<T>,
    s2: Vec<T>,
    /// The levels' grid: the greatest common divisor of the values.
    step: T,
}

impl<T: Accumulator> Sums<T> {
    fn new(vals: &[u32], counts: &[u64]) -> Self {
        let zero = T::from(0);
        let mut n = vec![zero; vals.len() + 1];
        let mut s1 = n.clone();
        let mut s2 = n.clone();

        let (mut count, mut sum, mut sum_sq) = (zero, zero, zero);

        let outputs = n.iter_mut().zip(&mut s1).zip(&mut s2).skip(1);

        for (((n, s1), s2), (&v, &c)) in outputs.zip(vals.iter().zip(counts)) {
            let v = T::from(u64::from(v));
            let c = T::from(c);
            let weighted = c * v;

            count = count + c;
            sum = sum + weighted;
            sum_sq = sum_sq + weighted * v;

            *n = count;
            *s1 = sum;
            *s2 = sum_sq;
        }
        let step = T::from(u64::from(vals.iter().fold(0, |g, &v| gcd(g, v))));
        Self { n, s1, s2, step }
    }

    /// The level of cell `i..=j` and its squared error.
    #[inline]
    fn cell(&self, i: usize, j: usize) -> (u32, T) {
        let n = self.n[j + 1] - self.n[i];
        let s1 = self.s1[j + 1] - self.s1[i];
        let s2 = self.s2[j + 1] - self.s2[i];
        let two = T::from(2);
        let divisor = n * self.step;
        let r = (s1 + divisor / two) / divisor * self.step;
        let error = s2 + r * r * n - two * r * s1;
        (r.as_level(), error)
    }
}

/// One more cell for ends `lo..=hi` of a deep plane: the least error of
/// `vals[..=j]` with the last cell starting in `first..=last` (its earliest
/// start is `k`, one value per earlier cell), recursing on the middle end's
/// best start.
#[allow(clippy::too_many_arguments)]
fn split_points<T: Accumulator>(
    sums: &Sums<T>,
    cost: &[T],
    k: usize,
    lo: usize,
    hi: usize,
    first: usize,
    last: usize,
    next: &mut [T],
    start: &mut [u16],
) {
    let j = (lo + hi) / 2;
    let mut best = (T::MAX, first);
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
    weights: &[u8],
    budget: u128,
    keep_full: bool,
) -> Option<AlphaPlane> {
    let max = match plane {
        AlphaPlane::U8(_) => 255u32,
        AlphaPlane::U16 { bits, .. } => (1u32 << bits) - 1,
        AlphaPlane::F32(_) => return None,
    };
    let mut hist = vec![0u64; max as usize + 1];
    for (i, &w) in weights.iter().enumerate().take(plane.len()) {
        hist[plane.get_i32(i) as usize] += u64::from(w);
    }
    // Zero never merges (alpha: the invisible set); with `keep_full`
    // neither does the full value (K: no ink). A cell of the other values
    // reconstructs strictly inside them, so no level lands on either.
    let top = if keep_full { max - 1 } else { max };
    let (vals, counts): (Vec<u32>, Vec<u64>) = (1..=top)
        .filter(|&v| hist[v as usize] > 0)
        .map(|v| (v, hist[v as usize]))
        .unzip();
    if vals.len() < 2 || vals.len() > usize::from(u16::MAX) {
        return None;
    }
    // At weight five, the image dimension limit keeps the total below 2^63.
    let total_weight: u64 = counts.iter().sum();
    // Bound the cell-error formula's terms and intermediate sum by
    // 2 * total_weight * top² before selecting the accumulator width.
    let lut = if total_weight <= u64::MAX / (2 * u64::from(top).pow(2)) {
        let budget = budget.min(u128::from(u64::MAX)) as u64;
        level_lut(&vals, &counts, budget, max)?
    } else {
        level_lut(&vals, &counts, budget, max)?
    };
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

fn level_lut<T: Accumulator>(
    vals: &[u32],
    counts: &[u64],
    budget: T,
    max: u32,
) -> Option<Vec<u32>> {
    let m = vals.len();
    let deep = m > MAX_VALUES;
    let sums = Sums::new(vals, counts);
    // cost[j]: least error of vals[..=j] in the current number of cells;
    // starts[k][j]: first value of the last cell in that partition.
    let mut cost: Vec<T> = (0..m).map(|j| sums.cell(0, j).1).collect();
    let mut starts: Vec<Vec<u16>> = vec![vec![0; m]];
    while cost[m - 1] > budget {
        if starts.len() + 1 >= m || (deep && starts.len() >= MAX_DEEP_LEVELS) {
            return None;
        }
        let k = starts.len();
        let mut next = vec![T::MAX; m];
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
        let level = sums.cell(i, j).0;
        for v in &mut lut[vals[i] as usize..=vals[j] as usize] {
            *v = level;
        }
        if i == 0 {
            break;
        }
        j = i - 1;
    }
    Some(lut)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(a: &AlphaPlane, b: &AlphaPlane, weights: &[u8]) -> u128 {
        (0..a.len())
            .map(|i| {
                let difference = a.get_i32(i).abs_diff(b.get_i32(i));
                u128::from(weights[i]) * u128::from(difference).pow(2)
            })
            .sum()
    }

    fn levels(p: &AlphaPlane) -> usize {
        let mut v: Vec<i32> = (0..p.len()).map(|i| p.get_i32(i)).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Exhaustive fewest-cells search over contiguous partitions.
    fn brute(vals: &[u32], counts: &[u64], budget: u128) -> usize {
        let sums = Sums::<u128>::new(vals, counts);
        let m = vals.len();
        for k in 1..=m {
            // Cut masks with k - 1 cuts among m - 1 gaps.
            for mask in 0u32..(1 << (m - 1)) {
                if mask.count_ones() as usize != k - 1 {
                    continue;
                }
                let (mut i, mut err) = (0, 0);
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
            let weights: Vec<u8> = (0..n).map(|_| 1 + (rnd() % 3) as u8).collect();
            let mut hist = [0u64; 256];
            for (&v, &w) in data.iter().zip(&weights) {
                hist[v as usize] += u64::from(w);
            }
            let (vals, counts): (Vec<u32>, Vec<u64>) = (1..256u32)
                .filter(|&v| hist[v as usize] > 0)
                .map(|v| (v, hist[v as usize]))
                .unzip();
            let budget = u128::from(rnd() % 40_000);
            let want = brute(&vals, &counts, budget);
            match fewest_levels(&plane, &weights, budget, false) {
                Some(q) => {
                    assert!(sse(&plane, &q, &weights) <= budget);
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
    fn exhaustive(vals: &[u32], counts: &[u64], budget: u128) -> usize {
        let sums = Sums::<u64>::new(vals, counts);
        let m = vals.len();
        let mut cost: Vec<u64> = (0..m).map(|j| sums.cell(0, j).1).collect();
        let mut k = 1;
        while u128::from(cost[m - 1]) > budget {
            let mut next = vec![u64::MAX; m];
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
            let weights: Vec<u8> = (0..n).map(|_| 1 + (rnd() % 3) as u8).collect();
            let mut hist = vec![0u64; max as usize + 1];
            for (&v, &w) in data.iter().zip(&weights) {
                hist[v as usize] += u64::from(w);
            }
            let (vals, counts): (Vec<u32>, Vec<u64>) = (1..=max)
                .filter(|&v| hist[v as usize] > 0)
                .map(|v| (v, hist[v as usize]))
                .unzip();
            assert!(vals.len() > MAX_VALUES, "a deep plane");
            let one_cell_error = Sums::<u128>::new(&vals, &counts).cell(0, vals.len() - 1).1;
            for divisor in [100, 1_000, 10_000] {
                let budget = one_cell_error / divisor;
                let q = fewest_levels(&plane, &weights, budget, false).expect("quantized");
                assert!(sse(&plane, &q, &weights) <= budget);
                let zeros = data.iter().filter(|&&v| v == 0).count();
                assert_eq!(zeros, (0..q.len()).filter(|&i| q.get_i32(i) == 0).count());
                assert_eq!(levels(&q), exhaustive(&vals, &counts, budget) + 1);
            }
        }
    }

    #[test]
    fn cell_keeps_small_errors_with_large_moments() {
        let vals = [65532, 65533, 65534, 65535];
        let counts = [1 << 49, 1, 1, 1];
        let sums = Sums::<u128>::new(&vals, &counts);
        assert_eq!(sums.cell(0, 3), (65532, 14));
        assert_eq!(sums.cell(1, 3), (65534, 2));
        let lut = level_lut::<u128>(&vals, &counts, 2, 65535).expect("quantized");
        assert_eq!(&lut[65532..], &[65532, 65534, 65534, 65534]);
    }

    #[test]
    fn squeeze_budget_clamps_samples_and_rounds_down() {
        let plane = AlphaPlane::U8(vec![0, 1, 255, 254]);
        assert_eq!(
            squeeze_budget(&plane, &[i32::MIN, 0, i32::MAX, 255], &[1, 5, 1, 2]),
            3
        );
        let plane = AlphaPlane::U16 {
            data: vec![0, 1, 65535, 65534],
            bits: 16,
        };
        assert_eq!(
            squeeze_budget(&plane, &[i32::MIN, 0, i32::MAX, 65535], &[1, 5, 1, 2]),
            3
        );
        let plane = AlphaPlane::U16 {
            data: vec![0, 65535],
            bits: 16,
        };
        assert_eq!(
            squeeze_budget(&plane, &[65535, 0], &[5, 5]),
            5 * 65535u128.pow(2)
        );
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
        let weights = vec![1; data.len()];
        let loose = fewest_levels(&plane, &weights, 4096 * 4_000_000, false).expect("quantized");
        assert!((0..loose.len()).any(|i| data[i] == 255 && loose.get_i32(i) != 255));
        let kept = fewest_levels(&plane, &weights, 4096 * 4_000_000, true).expect("quantized");
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
        let weights = vec![1; plane.len()];
        let q = fewest_levels(&plane, &weights, 4096 * 4_000_000, false).expect("quantized");
        assert!((0..q.len()).all(|i| q.get_i32(i) % 257 == 0 && q.get_i32(i) > 0));
        assert!(levels(&q) < 200);
    }

    #[test]
    fn fewest_levels_keeps_the_bit_depth_of_wide_planes() {
        let data: Vec<u16> = (0..4096).map(|i| (i % 64 * 16) as u16).collect();
        let plane = AlphaPlane::U16 { data, bits: 10 };
        let weights = vec![1; plane.len()];
        let q = fewest_levels(&plane, &weights, 4096 * 400, false).expect("quantized");
        assert!(matches!(q, AlphaPlane::U16 { bits: 10, .. }));
        assert!(levels(&q) < 64);
        assert!(sse(&plane, &q, &weights) <= 4096 * 400);
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
                assert_eq!(weights[y * w + x] > 1, smooth, "x {x} y {y}");
            }
        }
    }
}
