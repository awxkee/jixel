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

//! Flat-color art as a small palette for the lossless coder.
//!
//! Outlined cartoons and flat illustrations are a few fill colors, faintly
//! noisy, with antialiased edges between them. The DCT spends a block of
//! coefficients on every such edge. Here every pixel close to a fill takes
//! the fill, and every other pixel the nearest point of a lattice as wide
//! as the tolerance. The frame is then a few hundred colors that the
//! lossless coder's palette and trees code in a fraction of the lossy
//! file, with the edges intact.
//!
//! The edge pixels are a hundredth of such a frame and most of its colors.
//! Naming them as mixes of the two fills around them keeps more colors
//! than the lattice does, for the same picture.

use crate::Speed;
use crate::entropy::f_log2;
use std::collections::HashMap;

/// The tolerance of a palette per unit of distance: the largest distance
/// (per channel, in 8-bit levels) between a pixel and the color that
/// replaces it. At this slope the palette file of flat art is the smaller
/// and the better picture of the two next to the lossy file of the same
/// distance; below a twelfth of a unit the frame is coded as it is. Past
/// [`MAX_TOLERANCE`] a palette loses more of the picture than of the file,
/// and soon whole fills.
const TOLERANCE_PER_DISTANCE: f32 = 12.0;
const MAX_TOLERANCE: f32 = 24.0;

/// Fills kept, and the share of the frame a color and its neighbors within
/// the tolerance must hold to be one.
const MAX_FILLS: usize = 64;
const MIN_FILL_SHARE: f64 = 0.0005;
/// Colors examined as fills, most frequent first.
const FILL_CANDIDATES: usize = 4096;
/// A frame with more colors than this is not flat art: outlined cartoons
/// and flat illustrations have 700 to 18 000, mixed pages and soft-shaded
/// art 34 000 and up.
const MAX_COLORS: usize = 1 << 15;
/// Share of the frame within [`DETECT_TOLERANCE`] of a fill for the frame
/// to be flat art: 0.87 to 0.99 on such art, under 0.6 on photos and
/// gradients, 0.7 to 0.85 on soft-shaded art. A wider tolerance tells
/// nothing: 64 fills eight levels wide cover most of a photo.
const DETECT_TOLERANCE: u8 = 2;
const MIN_FLAT_SHARE: f64 = 0.86;

#[inline]
fn key(p: [u8; 3]) -> u32 {
    u32::from(p[0]) << 16 | u32::from(p[1]) << 8 | u32::from(p[2])
}

#[inline]
fn apart(a: [u8; 3], b: [u8; 3]) -> u8 {
    a[0].abs_diff(b[0])
        .max(a[1].abs_diff(b[1]))
        .max(a[2].abs_diff(b[2]))
}

/// The colors of a frame with their pixel counts, most frequent first.
struct Colors {
    counted: Vec<([u8; 3], u32)>,
    pixels: usize,
}

impl Colors {
    /// `None` when the frame has more than [`MAX_COLORS`] colors.
    fn new(pixels: &[[u8; 3]]) -> Option<Self> {
        let mut counts: HashMap<u32, u32> = HashMap::new();
        // Flat art is long runs of one color: one lookup a run.
        for run in pixels.chunk_by(|a, b| a == b) {
            *counts.entry(key(run[0])).or_insert(0) += run.len() as u32;
            if counts.len() > MAX_COLORS {
                return None;
            }
        }
        let mut counted: Vec<([u8; 3], u32)> = counts
            .into_iter()
            .map(|(k, n)| ([(k >> 16) as u8, (k >> 8) as u8, k as u8], n))
            .collect();
        counted.sort_unstable_by_key(|&(color, n)| (std::cmp::Reverse(n), color));
        Some(Self {
            counted,
            pixels: pixels.len(),
        })
    }

    /// The fills at `tolerance`: frequent colors no closer than it to each
    /// other, each holding [`MIN_FILL_SHARE`] of the frame with the
    /// candidates around it.
    fn fills(&self, tolerance: u8) -> Vec<[u8; 3]> {
        let candidates = &self.counted[..self.counted.len().min(FILL_CANDIDATES)];
        let floor = MIN_FILL_SHARE * self.pixels as f64;
        // The candidates by cell of a grid one tolerance wide: those
        // around a color are in its cell and the 26 next to it.
        let side = i32::from(tolerance) + 1;
        let cell_of = |color: [u8; 3]| color.map(|value| i32::from(value) / side);
        let mut cells: HashMap<[i32; 3], Vec<u16>> = HashMap::new();
        for (index, &(color, _)) in candidates.iter().enumerate() {
            cells.entry(cell_of(color)).or_default().push(index as u16);
        }
        let mut fills: Vec<[u8; 3]> = Vec::new();
        for &(color, count) in candidates {
            if fills.len() == MAX_FILLS {
                break;
            }
            if fills.iter().any(|&fill| apart(fill, color) <= tolerance) {
                continue;
            }
            let holds = f64::from(count) >= floor || {
                let [r, g, b] = cell_of(color);
                let mut held = 0u64;
                for cell in (r - 1..=r + 1).flat_map(|r| {
                    (g - 1..=g + 1).flat_map(move |g| (b - 1..=b + 1).map(move |b| [r, g, b]))
                }) {
                    for &other in cells.get(&cell).map_or(&[][..], Vec::as_slice) {
                        let (other, n) = candidates[usize::from(other)];
                        if apart(other, color) <= tolerance {
                            held += u64::from(n);
                        }
                    }
                }
                held as f64 >= floor
            };
            if holds {
                fills.push(color);
            }
        }
        fills
    }

    /// The fill of every color within `tolerance` of one (the nearest).
    fn fill_of(&self, fills: &[[u8; 3]], tolerance: u8) -> HashMap<u32, u8> {
        let mut fill_of = HashMap::with_capacity(self.counted.len());
        for &(color, _) in &self.counted {
            let nearest = fills
                .iter()
                .enumerate()
                .map(|(index, &fill)| (apart(fill, color), index))
                .min();
            if let Some((distance, index)) = nearest
                && distance <= tolerance
            {
                fill_of.insert(key(color), index as u8);
            }
        }
        fill_of
    }

    /// Whether most of the frame is within a few levels of a few colors.
    fn is_flat_art(&self) -> bool {
        let fills = self.fills(DETECT_TOLERANCE);
        if fills.len() < 2 {
            return false;
        }
        let fill_of = self.fill_of(&fills, DETECT_TOLERANCE);
        let flat: u64 = self
            .counted
            .iter()
            .filter(|&&(color, _)| fill_of.contains_key(&key(color)))
            .map(|&(_, n)| u64::from(n))
            .sum();
        flat as f64 >= MIN_FLAT_SHARE * self.pixels as f64
    }

    /// The frame with every pixel within `tolerance` of a fill replaced by
    /// it, and every other pixel by the nearest point of a lattice of that
    /// spacing.
    fn quantize(&self, pixels: &[[u8; 3]], tolerance: u8) -> Vec<[u8; 3]> {
        let fills = self.fills(tolerance);
        let fill_of = self.fill_of(&fills, tolerance);
        let step = u16::from(tolerance) + 1;
        let mut out: Vec<[u8; 3]> = Vec::with_capacity(pixels.len());
        for run in pixels.chunk_by(|a, b| a == b) {
            let color = match fill_of.get(&key(run[0])) {
                Some(&fill) => fills[usize::from(fill)],
                None => {
                    run[0].map(|value| ((u16::from(value) + step / 2) / step * step).min(255) as u8)
                }
            };
            out.resize(out.len() + run.len(), color);
        }
        out
    }
}

/// How far the quantized frame moved whole areas of the original: the 99th
/// percentile of the mean error of 8x8 blocks and the largest mean error
/// of a 16x16 block (the larger channel of each, in 8-bit levels).
///
/// Snapping the noise of a fill leaves a block's mean where it was. Two
/// fills merged, or shading cut into bands, move it by many levels, and
/// that is what shows: over 84 quantized frames of flat art the first
/// number ranks them as SSIMULACRA2 does (rank correlation 0.91), and the
/// second catches the merge of a small fill the first is blind to (1.6
/// before, 17 after, on under 1 % of the blocks).
fn area_error(
    original: &[[u8; 3]],
    quantized: &[[u8; 3]],
    width: usize,
    height: usize,
) -> (f32, f32) {
    let means = |side: usize| -> Vec<f32> {
        let (columns, rows) = (width / side, height / side);
        let mut sums = vec![[0i32; 3]; columns];
        let mut out = Vec::with_capacity(columns * rows);
        for block_row in 0..rows {
            sums.fill([0; 3]);
            for y in block_row * side..(block_row + 1) * side {
                let row = y * width..y * width + columns * side;
                for (sum, (a, b)) in sums.iter_mut().zip(
                    original[row.clone()]
                        .chunks_exact(side)
                        .zip(quantized[row].chunks_exact(side)),
                ) {
                    for (p, q) in a.iter().zip(b) {
                        for c in 0..3 {
                            sum[c] += i32::from(q[c]) - i32::from(p[c]);
                        }
                    }
                }
            }
            out.extend(sums.iter().map(|sum| {
                sum.iter().map(|v| v.abs()).max().unwrap_or(0) as f32 / (side * side) as f32
            }));
        }
        out
    };
    let mut fine = means(8);
    let frequent = if fine.is_empty() {
        0.0
    } else {
        let at = (fine.len() * 99 / 100).min(fine.len() - 1);
        *fine.select_nth_unstable_by(at, f32::total_cmp).1
    };
    let largest = means(16).into_iter().fold(0.0f32, f32::max);
    (frequent, largest)
}

/// How far a palette may move whole areas at `distance` (see
/// [`area_error`]): the frequent and the largest block error. The byte
/// budget picks the palette; these keep it from the tolerances where fills
/// merge. They follow what SSIMULACRA2 allowed as `9 + 4 * distance` on the
/// frames the measure was fitted on.
fn area_limits(distance: f32) -> (f32, f32) {
    ((1.5 + 4.0 * distance) / 3.0, 3.0 + 3.0 * distance)
}

/// What the lossless coder at `Medium` makes of a palette frame, in bytes,
/// from one pass over it: the entropy of where each pixel's color comes
/// from (the pixel to the left, above, above left, above right, or none of
/// them) under which of those four agree, and the entropy of the colors
/// that come from none. The weights are fitted to the Medium files of 13
/// frames at every tolerance; those came out 0.62 to 1.36 times the
/// estimate, and within one frame they follow it more closely than they
/// follow the frame's `Fast` files, which cost fifty times as much.
fn estimated_bytes(frame: &[[u8; 3]], width: usize) -> f64 {
    const SOURCES: usize = 5;
    let mut counts = [[0u32; SOURCES]; 16];
    let mut fresh: HashMap<u32, u32> = HashMap::new();
    for (above, row) in frame
        .chunks_exact(width)
        .zip(frame.chunks_exact(width).skip(1))
    {
        for x in 1..width.saturating_sub(1) {
            let (pixel, left, up) = (row[x], row[x - 1], above[x]);
            let (up_left, up_right) = (above[x - 1], above[x + 1]);
            let context = usize::from(left == up)
                | usize::from(up == up_left) << 1
                | usize::from(up == up_right) << 2
                | usize::from(x >= 2 && row[x - 2] == left) << 3;
            let source = if pixel == left {
                0
            } else if pixel == up {
                1
            } else if pixel == up_left {
                2
            } else if pixel == up_right {
                3
            } else {
                *fresh.entry(key(pixel)).or_insert(0) += 1;
                4
            };
            counts[context][source] += 1;
        }
    }
    let bits = |counts: &mut dyn Iterator<Item = u32>| -> f64 {
        let (mut total, mut sum) = (0.0f64, 0.0f64);
        for count in counts.filter(|&count| count > 0) {
            let count = f64::from(count);
            total += count;
            sum += count * f_log2(count);
        }
        if total > 0.0 {
            total * f_log2(total) - sum
        } else {
            0.0
        }
    };
    let sources: f64 = counts
        .iter()
        .map(|context| bits(&mut context.iter().copied()))
        .sum();
    (SOURCE_WEIGHT * sources + FRESH_WEIGHT * bits(&mut fresh.values().copied())) / 8.0
}

const SOURCE_WEIGHT: f64 = 0.5;
const FRESH_WEIGHT: f64 = 1.0;

/// An estimate this many times over the lossy file gives the frame up
/// without coding it. The estimate runs high on art of few colors, where
/// files fit up to a third over it; at `Medium`, where a file to check
/// costs as much as the lossy encode did, the bound is closer to that.
const ESTIMATE_MAX: f64 = 1.65;
const ESTIMATE_MAX_MEDIUM: f64 = 1.4;

/// The middle crop that stands for a large frame in the early exit, and
/// how far the whole frame's predicted Medium file may exceed the lossy
/// one before the frame is given up.
const PROBE_SIDE: usize = 768;
const PROBE_SLACK: f64 = 1.15;

/// A slower tier's file is at least this share of the `Medium` one.
const SLOW_SHARE: f64 = 0.85;

/// The palette file of a flat-art frame, when it is no larger than the
/// frame's `lossy` file of that many bytes; `None` for any other frame.
///
/// The distance sets the palette: its tolerance is
/// [`TOLERANCE_PER_DISTANCE`] times `distance`, or the widest below that
/// within the [`area_limits`], so that the file and the picture follow the
/// distance as the lossy ones do, down to the frame itself at the smallest
/// distances. On such art that palette is the better picture of the two
/// at fewer bytes, until its fills would merge; past that the palette stays
/// as it is and the lossy file takes over once it is the smaller.
///
/// `code` is the lossless coder at a given speed. [`estimated_bytes`]
/// gives up the frames that are far from fitting before any is coded.
pub(crate) fn search<E>(
    pixels: &[[u8; 3]],
    width: usize,
    height: usize,
    distance: f32,
    lossy: usize,
    speed: Speed,
    mut code: impl FnMut(&[[u8; 3]], usize, usize, Speed) -> Result<Vec<u8>, E>,
) -> Result<Option<Vec<u8>>, E> {
    let Some(colors) = Colors::new(pixels) else {
        return Ok(None);
    };
    // Two colors and gray line art take exact paths of their own.
    if colors.counted.len() <= 2
        || colors
            .counted
            .iter()
            .all(|&([r, g, b], _)| r == g && g == b)
        || !colors.is_flat_art()
    {
        return Ok(None);
    }
    // The widest tolerance up to the distance's own within the limits: the
    // error of whole areas grows with the tolerance, steeply once two
    // fills merge. No tolerance at all is the frame itself.
    let (frequent_limit, largest_limit) = area_limits(distance);
    let within = |tolerance: u8| {
        let quantized = colors.quantize(pixels, tolerance);
        let (frequent, largest) = area_error(pixels, &quantized, width, height);
        (frequent <= frequent_limit && largest <= largest_limit).then_some(quantized)
    };
    let target = (TOLERANCE_PER_DISTANCE * distance).min(MAX_TOLERANCE) as u8;
    let mut quantized = (target > 0).then(|| within(target)).flatten();
    if quantized.is_none() {
        let (mut low, mut high) = (0, target);
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            match within(middle) {
                Some(frame) => (low, quantized) = (middle, Some(frame)),
                None => high = middle,
            }
        }
    }
    let frame = quantized.as_deref().unwrap_or(pixels);
    // A screenshot or a page of text is flat too, and its lossless frame
    // larger than the lossy one at any tolerance.
    let estimate = estimated_bytes(frame, width);
    let give_up = if speed == Speed::Medium {
        ESTIMATE_MAX_MEDIUM
    } else {
        ESTIMATE_MAX
    };
    if estimate > lossy as f64 * give_up {
        return Ok(None);
    }
    let (crop_w, crop_h) = (width.min(PROBE_SIDE), height.min(PROBE_SIDE));
    if estimate > lossy as f64 && 2 * crop_w * crop_h <= width * height {
        // What Medium makes of the estimate, measured on the middle of a
        // large frame: a Medium file of the whole of a screenshot costs as
        // much as its lossy encode.
        let (x0, y0) = ((width - crop_w) / 2, (height - crop_h) / 2);
        let crop: Vec<[u8; 3]> = (y0..y0 + crop_h)
            .flat_map(|y| frame[y * width + x0..][..crop_w].iter().copied())
            .collect();
        let medium = code(&crop, crop_w, crop_h, Speed::Medium)?.len();
        let ratio = medium as f64 / estimated_bytes(&crop, crop_w).max(1.0);
        if estimate * ratio > lossy as f64 * PROBE_SLACK {
            return Ok(None);
        }
    }
    // A slower tier's file costs several Medium ones: where the estimate
    // leaves the outcome open, a Medium file decides first.
    let mut coded = None;
    if speed > Speed::Medium && estimate > lossy as f64 * SLOW_SHARE {
        let medium = code(frame, width, height, Speed::Medium)?;
        if medium.len() as f64 * SLOW_SHARE > lossy as f64 {
            return Ok(None);
        }
        coded = Some(medium);
    }
    let last = code(frame, width, height, speed)?;
    let coded = match coded {
        Some(medium) if medium.len() < last.len() => medium,
        _ => last,
    };
    Ok((coded.len() <= lossy).then_some(coded))
}
