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

//! Lossy coding of few-level gray line art: manga pages and ink drawings
//! whose pixels are black, white and the anti-aliased edges between them.
//!
//! Such a page is quantized to a handful of gray levels and handed to the
//! lossless coder.

/// Gray values this close to black or white count as the page's ink and
/// paper; the rest is "mid" gray.
const END_MARGIN: u8 = 24;
/// Ink and paper proper: within this of black or white.
const PURE_MARGIN: u8 = 8;

/// A page takes the few-level path when at most this share of its pixels is
/// smooth tone: mid gray whose 5x5 neighborhood spans less than
/// [`SMOOTH_RANGE`] levels. Line art measures 0.000-0.039, shaded pages and
/// scans 0.043-0.38.
const MAX_SMOOTH_SHARE: f64 = 0.04;
const SMOOTH_RANGE: u8 = 48;
/// ...and at most this share lies between pure and mid gray, where JPEG
/// ringing and scanner noise sit; a quantizer would keep that noise as
/// texture (a JPEG-sourced page: 0.19, twice the VarDCT bytes).
const MAX_DIRTY_SHARE: f64 = 0.1;
/// ...and at least this share is pure ink or paper.
const MIN_PURE_SHARE: f64 = 0.4;

/// The smooth-tone share is measured on every `SAMPLE_STEP`-th pixel of
/// every `SAMPLE_STEP`-th row.
const SAMPLE_STEP: usize = 3;

fn level_count(distance: f32) -> usize {
    ((20.0 / distance).round() as usize).clamp(MIN_LEVELS, 64)
}

const MIN_LEVELS: usize = 6;

/// A page with at most this many gray levels is already a few-level page
/// and is coded exactly: it is distribution-format manga, whose levels
/// carry its screens.
const EXACT_MAX_LEVELS: usize = 16;

/// Whether `gray` is few-level line art (see the module doc).
fn is_line_art(gray: &[u8], width: usize, height: usize, histogram: &[usize; 256]) -> bool {
    if width < 8 || height < 8 {
        return false;
    }
    let total = gray.len() as f64;
    let share = |levels: std::ops::RangeInclusive<usize>| {
        histogram[levels].iter().sum::<usize>() as f64 / total
    };
    let (pure, end) = (usize::from(PURE_MARGIN), usize::from(END_MARGIN));
    let pure_share = share(0..=pure) + share(255 - pure..=255);
    let dirty_share = share(pure + 1..=end) + share(255 - end..=255 - pure - 1);
    if pure_share < MIN_PURE_SHARE || dirty_share > MAX_DIRTY_SHARE {
        return false;
    }
    let is_mid = |value: u8| value > END_MARGIN && value < 255 - END_MARGIN;
    let (mut samples, mut smooth) = (0usize, 0usize);
    for y in (2..height - 2).step_by(SAMPLE_STEP) {
        for x in (2..width - 2).step_by(SAMPLE_STEP) {
            samples += 1;
            if !is_mid(gray[y * width + x]) {
                continue;
            }
            let (mut low, mut high) = (255u8, 0u8);
            for row in gray[(y - 2) * width..(y + 3) * width].chunks_exact(width) {
                for &value in &row[x - 2..x + 3] {
                    low = low.min(value);
                    high = high.max(value);
                }
            }
            smooth += usize::from(high - low < SMOOTH_RANGE);
        }
    }
    samples > 0 && (smooth as f64) <= MAX_SMOOTH_SHARE * samples as f64
}

/// The `count` levels that minimize the squared error of the histogram
/// (Lloyd-Max), with the darkest and lightest present values kept; as a
/// value-to-level table.
fn level_table(histogram: &[usize; 256], count: usize) -> [u8; 256] {
    let low = histogram.iter().position(|&n| n != 0).unwrap_or(0);
    let high = histogram.iter().rposition(|&n| n != 0).unwrap_or(255);
    let mut levels: Vec<f64> = (0..count)
        .map(|i| low as f64 + (high - low) as f64 * i as f64 / (count - 1) as f64)
        .collect();
    for _ in 0..30 {
        // Each inner level moves to the mean of the values nearest to it.
        for i in 1..count - 1 {
            let from = ((levels[i - 1] + levels[i]) / 2.0).floor() as usize + 1;
            let to = ((levels[i] + levels[i + 1]) / 2.0).floor() as usize;
            let (mut weight, mut sum) = (0f64, 0f64);
            for value in from..=to.min(255) {
                weight += histogram[value] as f64;
                sum += histogram[value] as f64 * value as f64;
            }
            if weight > 0.0 {
                levels[i] = sum / weight;
            }
        }
    }
    let levels: Vec<i32> = levels.iter().map(|&level| level.round() as i32).collect();
    std::array::from_fn(|value| {
        let nearest = levels
            .iter()
            .min_by_key(|&&level| (level - value as i32).abs())
            .copied()
            .unwrap_or(value as i32);
        nearest as u8
    })
}

/// `gray` quantized for the few-level path at `distance`: `None` when the
/// page is not line art. A page that already has no more levels than the
/// distance keeps comes back unchanged, to be coded exactly.
pub(crate) fn quantize(gray: &[u8], width: usize, height: usize, distance: f32) -> Option<Vec<u8>> {
    debug_assert_eq!(gray.len(), width * height);
    let mut histogram = [0usize; 256];
    for &value in gray {
        histogram[usize::from(value)] += 1;
    }
    if !is_line_art(gray, width, height, &histogram) {
        return None;
    }
    let count = level_count(distance);
    if histogram.iter().filter(|&&n| n != 0).count() <= count.max(EXACT_MAX_LEVELS) {
        return Some(gray.to_vec());
    }
    let table = level_table(&histogram, count);
    Some(
        gray.iter()
            .map(|&value| table[usize::from(value)])
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_sort::SharedSort;

    /// Black strokes on white whose edges ramp through `steps` gray levels.
    fn line_art(width: usize, height: usize, steps: usize) -> Vec<u8> {
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                // The ramp level varies along the stroke, as sub-pixel
                // positions do.
                let ramp = |phase: usize| (255 * (1 + (y + phase) % steps) / (steps + 1)) as u8;
                match (x + y / 3) % 24 {
                    0..=5 => 0,
                    6 => ramp(0),
                    23 => ramp(steps / 2),
                    _ => 255,
                }
            })
            .collect()
    }

    fn levels_of(page: &[u8]) -> Vec<u8> {
        let mut levels = page.to_vec();
        levels.shared_sort();
        levels.dedup();
        levels
    }

    #[test]
    fn line_art_is_quantized_and_tone_is_left_alone() {
        let (w, h) = (96, 96);
        // A page of few levels is kept exactly at any distance.
        let few = line_art(w, h, 5);
        assert_eq!(levels_of(&few).len(), 7);
        assert_eq!(quantize(&few, w, h, 5.0).unwrap(), few);
        // Forty edge levels fold onto the distance's count, ends kept.
        let page = line_art(w, h, 40);
        assert_eq!(levels_of(&page).len(), 42);
        for (distance, count) in [(1.2, 17), (2.0, 10), (5.0, 6)] {
            let levels = levels_of(&quantize(&page, w, h, distance).unwrap());
            assert!(
                levels.len() <= count && levels.len() >= count - 1,
                "{distance}"
            );
            assert_eq!((levels[0], levels[levels.len() - 1]), (0, 255));
        }
        assert_eq!(quantize(&page, w, h, 0.3).unwrap(), page);

        // A smooth ramp is tone, and a page of faint noise around white is
        // a dirty scan.
        let ramp: Vec<u8> = (0..w * h).map(|i| (i % w * 255 / (w - 1)) as u8).collect();
        assert!(quantize(&ramp, w, h, 2.0).is_none());
        let dirty: Vec<u8> = (0..w * h)
            .map(|i| {
                if i % 7 == 0 {
                    0
                } else {
                    236 + (i % 3) as u8 * 4
                }
            })
            .collect();
        assert!(quantize(&dirty, w, h, 2.0).is_none());
        assert_eq!(
            (level_count(0.6), level_count(2.0), level_count(100.0)),
            (33, 10, 6)
        );
    }
}
