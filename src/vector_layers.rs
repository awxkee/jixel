/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Vector layers: flat-color art with antialiased edges split into a hard
//! palette region map (the base, lossless) and a coverage band that carries
//! the antialiasing (one kBlend frame, lossless, on coverage levels that
//! shrink with the distance). One deterministic pass: exact-color histogram
//! on uniform neighborhoods for the palette, then each non-palette pixel is
//! unblended between two neighboring region colors.

use crate::util::{heap_array, round_i32};

/// A palette color must fill at least this many uniform pixels.
const MIN_PALETTE_PIXELS: usize = 32;
/// Palette entries, excluding the "no label" value.
const MAX_PALETTE: usize = 254;
const NO_LABEL: u8 = 255;
/// Share of pixels that must sit exactly on a palette color.
const MIN_EXACT_SHARE: f32 = 0.85;
/// Share of sampled pixels with two equal 4-neighbors below which the image
/// cannot reach `MIN_EXACT_SHARE`.
const MIN_FLAT_SAMPLE_SHARE: f32 = 0.5;
/// Share of pixels the coverage band may cover.
const MAX_BAND_SHARE: f32 = 0.08;
/// Share of pixels that are neither a palette color nor a two-color mix
/// (grain, photos): such content stays with the ordinary encoder.
const MAX_TEXTURED_SHARE: f32 = 0.005;
/// Largest distance (squared) of a pixel from the segment between its two
/// region colors for the pixel to count as their mix.
const MAX_MIX_RESIDUAL_SQ: f32 = 36.0;
/// Pixels around a non-palette pixel searched for its two region colors.
const MIX_REACH: i32 = 3;
/// Largest per-channel distance of a pixel, or of a smaller palette color,
/// from the palette color the snapped split moves it onto.
const SNAP_REACH: u8 = 4;
/// The snap's reach at a distance: the full `SNAP_REACH` from distance 1,
/// narrower below it down to two levels. A fixed reach capped the vector
/// arm's quality (a cat illustration held butteraugli 0.26 from distance 1
/// to 0.25 while its file grew by a third; now 0.26, 0.16, 0.16); at one
/// level resized art stops splitting.
pub(crate) fn snap_reach(distance: f32) -> u8 {
    (SNAP_REACH as f32 * distance)
        .round()
        .clamp(2.0, SNAP_REACH as f32) as u8
}

/// Half side of the window the snap error is averaged over.
const SNAP_WINDOW_RADIUS: usize = 4;
/// Coverage steps above zero at distance 1 and below: alpha = step * 17.
const COVERAGE_STEPS: u32 = 15;

/// Coverage levels of the band for a distance: 16 at distance 1, more below
/// it up to the exact 8-bit mix, fewer above it down to the bare palette.
pub(crate) fn coverage_steps(distance: f32) -> u32 {
    (COVERAGE_STEPS as f32 / distance.max(0.01))
        .round()
        .clamp(1.0, 255.0) as u32
}

/// Size of the band frame relative to its size at 16 levels: measured
/// 0.39 at 2 levels, 0.60 at 5, 0.72 at 8, 0.82 at 10.
pub(crate) fn band_bytes_ratio(steps: u32) -> f64 {
    ((steps.max(1) as f64).log2() + 1.0) / 5.0
}

/// Bytes the VarDCT encode of the same image would take at `distance`,
/// from the cost of its edges.
pub(crate) const VARDCT_BYTES_MARGIN: f64 = 1.39;

pub(crate) fn vardct_bytes(rgb: &[u8], width: usize, height: usize, distance: f32) -> f64 {
    let d = distance.max(0.1);
    let cost: [f32; 256] = std::array::from_fn(|v| (1.0 + v as f32 / (4.0 * d)).log2());
    let mut bits = 0.0;
    visit_edges(rgb, width, height, |v| bits += f64::from(cost[v]));
    1.16 * bits / 8.0 * d.powf(-0.41) as f64
}

/// Evaluate the edge budget at several distances from one image traversal.
pub(crate) fn vardct_edge_histogram(rgb: &[u8], width: usize, height: usize) -> [usize; 256] {
    let mut counts = [0; 256];
    visit_edges(rgb, width, height, |v| {
        if v != 0 {
            counts[v] += 1;
        }
    });
    counts
}

pub(crate) fn vardct_bytes_from_edges(counts: &[usize; 256], distance: f32) -> f64 {
    let d = distance.max(0.1);
    let bits: f64 = counts
        .iter()
        .enumerate()
        .skip(1)
        .map(|(v, &count)| count as f64 * f64::from((1.0 + v as f32 / (4.0 * d)).log2()))
        .sum();
    1.16 * bits / 8.0 * d.powf(-0.41) as f64
}

fn visit_edges(rgb: &[u8], width: usize, height: usize, mut visit: impl FnMut(usize)) {
    let px = rgb.as_chunks::<3>().0;
    let delta = |a: [u8; 3], b: [u8; 3]| {
        usize::from(
            a[0].abs_diff(b[0])
                .max(a[1].abs_diff(b[1]))
                .max(a[2].abs_diff(b[2])),
        )
    };
    for y in 0..height {
        let row = &px[y * width..(y + 1) * width];
        for x in 1..width {
            visit(delta(row[x - 1], row[x]));
        }
        if y > 0 {
            let above = &px[(y - 1) * width..y * width];
            for x in 0..width {
                visit(delta(above[x], row[x]));
            }
        }
    }
}

/// Open-addressing histogram for the exact colors of flat art.
struct ColorTable {
    keys: Box<[u32; ColorTable::CAPACITY]>,
    values: Box<[u32; ColorTable::CAPACITY]>,
    len: usize,
}

impl ColorTable {
    const BITS: u32 = 16;
    const CAPACITY: usize = 1 << Self::BITS;
    const EMPTY: u32 = u32::MAX;

    fn new() -> Self {
        Self {
            keys: heap_array(Self::EMPTY),
            values: heap_array(0),
            len: 0,
        }
    }

    /// Count pixels of `color`, returning false if a new color would exceed capacity.
    fn add(&mut self, color: [u8; 3], count: u32) -> bool {
        let key = color_key(color);
        let mut i = color_slot::<{ Self::BITS }>(key);
        loop {
            let k = self.keys[i];
            if k == key {
                self.values[i] += count;
                return true;
            }
            if k == Self::EMPTY {
                if self.len >= Self::CAPACITY / 2 {
                    return false;
                }
                self.keys[i] = key;
                self.values[i] = count;
                self.len += 1;
                return true;
            }
            i = (i + 1) & (Self::CAPACITY - 1);
        }
    }

    fn entries(&self) -> impl Iterator<Item = ([u8; 3], usize)> + '_ {
        self.keys
            .iter()
            .zip(self.values.iter())
            .filter(|(k, _)| **k != Self::EMPTY)
            .map(|(&k, &v)| (k.to_be_bytes()[1..].try_into().unwrap(), v as usize))
    }
}

fn color_key(color: [u8; 3]) -> u32 {
    u32::from_be_bytes([0, color[0], color[1], color[2]])
}

fn color_slot<const BITS: u32>(key: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B1) >> (32 - BITS)) as usize
}

pub(crate) struct VectorSplit {
    /// Interleaved RGB of the base: palette colors with hard edges, mix
    /// pixels given to their lower region, textured pixels verbatim.
    pub(crate) base: Vec<u8>,
    /// Interleaved RGB of the coverage frame: the upper region's color on the
    /// band, the dominant band color elsewhere.
    pub(crate) layer: Vec<u8>,
    /// Coverage alpha on `steps + 1` levels, non-zero only on the band; all
    /// zero when the art is a bare palette or the band was dropped. Plans
    /// omit both coverage and layer buffers entirely when there is no band.
    pub(crate) coverage: Vec<u8>,
    /// Alpha of the base for art with transparency: 0 or 255 on the regions,
    /// the source alpha on textured pixels. `None` for opaque art.
    pub(crate) base_alpha: Option<Vec<u8>>,
}

pub(crate) fn split(rgb: &[u8], width: usize, height: usize, steps: u32) -> Option<VectorSplit> {
    let source = SplitSource::rgb(rgb, width, height)?;
    let mut split = source.prepare(None)?.split(steps)?;
    // Crop frames consume dense coverage, even when it is all zero.
    if split.coverage.is_empty() {
        let n = width * height;
        split.coverage = vec![0; n];
        split.layer = vec![*source.palette.last().unwrap(); n].into_flattened();
    }
    Some(split)
}

/// Pixels the snap moves, each with its new palette index; `labels` is
/// updated to match.
fn snap_fringe(
    pixel: impl Fn(usize) -> [u8; 3],
    width: usize,
    height: usize,
    palette: &[[u8; 3]],
    ground: Option<u8>,
    labels: &mut [u8],
    alpha: Option<&[u8]>,
    max_reach: u8,
) -> Vec<(usize, u8)> {
    let reach = |a: [u8; 3], b: [u8; 3]| {
        a[0].abs_diff(b[0])
            .max(a[1].abs_diff(b[1]))
            .max(a[2].abs_diff(b[2]))
    };
    // The palette is sorted by area: each color folds into the first larger
    // one in reach that stands itself.
    let mut root: Vec<u8> = (0..palette.len() as u8).collect();
    for k in 1..palette.len() {
        if Some(k as u8) == ground {
            continue;
        }
        if let Some(j) = (0..k).find(|&j| {
            Some(j as u8) != ground
                && root[j] == j as u8
                && reach(palette[j], palette[k]) <= max_reach
        }) {
            root[k] = j as u8;
        }
    }
    // Each red bucket lists the roots whose snap range reaches it. Keep
    // palette order so the first root wins ties in the full RGB distance.
    const RED_BUCKET_SHIFT: usize = 3;
    let red_bucket = |red: u8| usize::from(red) >> RED_BUCKET_SHIFT;
    let mut roots_by_red = None;
    // A bounded cache avoids allocating one entry per distinct stray color.
    const CACHE_BITS: u32 = 12;
    let mut nearest = None;
    let mut moved = Vec::new();
    for (i, &l) in labels.iter().enumerate() {
        // Only opaque pixels have a color to snap.
        if alpha.is_some_and(|alpha| alpha[i] != 255) {
            continue;
        }
        let target = if l != NO_LABEL {
            root[l as usize]
        } else {
            let p = pixel(i);
            let key = color_key(p);
            let cache =
                nearest.get_or_insert_with(|| heap_array::<_, { 1 << CACHE_BITS }>(u32::MAX - 1));
            let entry = &mut cache[color_slot::<CACHE_BITS>(key)];
            if *entry >> 8 != key || *entry as u8 == NO_LABEL - 1 {
                let buckets = roots_by_red.get_or_insert_with(|| {
                    let mut buckets: [Vec<_>; 256 >> RED_BUCKET_SHIFT] =
                        std::array::from_fn(|_| Vec::new());
                    for (k, &color) in palette.iter().enumerate() {
                        if Some(k as u8) != ground && root[k] == k as u8 {
                            let first = red_bucket(color[0].saturating_sub(max_reach));
                            let last = red_bucket(color[0].saturating_add(max_reach));
                            for bucket in &mut buckets[first..=last] {
                                bucket.push((color, k as u8));
                            }
                        }
                    }
                    buckets
                });
                let mut best = (max_reach + 1, NO_LABEL);
                for &(c, k) in &buckets[red_bucket(p[0])] {
                    let d = reach(c, p);
                    if d < best.0 {
                        best = (d, k);
                    }
                }
                *entry = (key << 8) | u32::from(best.1);
            }
            *entry as u8
        };
        if target != l {
            moved.push((i, target));
        }
    }
    if moved.is_empty() {
        return moved;
    }
    // Keep signed error sums for the rows in the window. All proposals
    // contribute to the sums, even when their own snap is rejected.
    // The final window sum per channel is bounded by 81 times SNAP_REACH.
    let mut columns = vec![[0i16; 3]; width];
    let mut active = 0..0;
    let error = |i: usize, k: u8| -> [i16; 3] {
        let color = palette[k as usize];
        let p = pixel(i);
        std::array::from_fn(|c| i16::from(color[c]) - i16::from(p[c]))
    };
    for &(i, k) in &moved {
        let (x, y) = (i % width, i / width);
        let y0 = y.saturating_sub(SNAP_WINDOW_RADIUS);
        let y1 = (y + SNAP_WINDOW_RADIUS + 1).min(height);
        let start = y0 * width;
        let end = y1 * width;
        while let Some(&(j, target)) = moved.get(active.end).filter(|&&(j, _)| j < end) {
            for (sum, error) in columns[j % width].iter_mut().zip(error(j, target)) {
                *sum += error;
            }
            active.end += 1;
        }
        while active.start < active.end && moved[active.start].0 < start {
            let (j, target) = moved[active.start];
            for (sum, error) in columns[j % width].iter_mut().zip(error(j, target)) {
                *sum -= error;
            }
            active.start += 1;
        }
        let x0 = x.saturating_sub(SNAP_WINDOW_RADIUS);
        let x1 = (x + SNAP_WINDOW_RADIUS + 1).min(width);
        let mut sum = [0i16; 3];
        for column in &columns[x0..x1] {
            for (total, value) in sum.iter_mut().zip(column) {
                *total += value;
            }
        }
        let area = ((y1 - y0) * (x1 - x0)) as i16;
        // The magnitude of the mean error must be at most half a level.
        if sum.into_iter().all(|s| 2 * s.abs() <= area) {
            labels[i] = k;
        }
    }
    moved.retain(|&(i, k)| labels[i] == k);
    moved
}

// Store each square perimeter in row order, with the innermost first.
const MIX_OFFSET_COUNT: usize = (4 * MIX_REACH * (MIX_REACH + 1)) as usize;

static MIX_OFFSETS: [(i32, i32); MIX_OFFSET_COUNT] = {
    let mut offsets = [(0, 0); MIX_OFFSET_COUNT];
    let mut i = 0;
    let mut ring = 1;
    while ring <= MIX_REACH {
        let mut dy = -ring;
        while dy <= ring {
            let mut dx = -ring;
            while dx <= ring {
                if dx == -ring || dx == ring || dy == -ring || dy == ring {
                    offsets[i] = (dx, dy);
                    i += 1;
                }
                dx += 1;
            }
            dy += 1;
        }
        ring += 1;
    }
    offsets
};

fn rgb_palette(px: &[[u8; 3]], width: usize, height: usize) -> Option<Vec<[u8; 3]>> {
    let n = width * height;
    if n < 64 * 64 {
        return None;
    }
    // Flat art has such pixels nearly everywhere (0.70 and up), photos and
    // textured screens below 0.25: one pixel in 64 settles it.
    let xs = (1..width - 1).step_by(8);
    let ys = (1..height - 1).step_by(8);
    let sampled = xs.len() * ys.len();
    let flat = ys
        .flat_map(|y| xs.clone().map(move |x| y * width + x))
        .filter(|&i| {
            let p = px[i];
            [i - 1, i + 1, i - width, i + width]
                .into_iter()
                .filter(|&j| px[j] == p)
                .count()
                >= 2
        })
        .count();
    if (flat as f32) < MIN_FLAT_SAMPLE_SHARE * sampled as f32 {
        return None;
    }
    // Palette: exact colors of pixels with at least two equal 4-neighbors
    // (flat regions and strokes down to one pixel; antialiasing mixes rarely
    // repeat along a line). A horizontal run qualifies throughout its
    // interior. Only its endpoints need the vertical neighbors checked.
    let mut counts = ColorTable::new();
    for y in 1..height - 1 {
        let pos = y * width;
        let row = &px[pos..pos + width];
        let above = &px[pos - width..pos];
        let below = &px[pos + width..pos + 2 * width];
        let mut start = 0;
        for run in row.chunk_by(|a, b| a == b) {
            let color = run[0];
            let len = run.len();
            let last = start + len - 1;
            let mut flat = len.saturating_sub(2);
            if start > 0 && start < width - 1 {
                let same = usize::from(len > 1)
                    + usize::from(above[start] == color)
                    + usize::from(below[start] == color);
                flat += usize::from(same >= 2);
            }
            if len > 1 && last < width - 1 {
                flat += usize::from(above[last] == color || below[last] == color);
            }
            if flat > 0 && !counts.add(color, flat as u32) {
                return None; // more flat colors than any palette art has
            }
            start += len;
        }
    }
    let min_pixels = MIN_PALETTE_PIXELS.max(n / 10_000);
    let mut palette: Vec<([u8; 3], usize)> =
        counts.entries().filter(|&(_, c)| c >= min_pixels).collect();
    if palette.is_empty() {
        return None;
    }
    palette.sort_unstable_by_key(|&(c, count)| (std::cmp::Reverse(count), c));
    palette.truncate(MAX_PALETTE);
    drop(counts);
    Some(palette.into_iter().map(|(color, _)| color).collect())
}

/// Code of a fully transparent pixel, whatever its color.
const TRANSPARENT: u32 = 0x0100_0000;

fn rgba_palette(
    px: &[[u8; 4]],
    alpha: &[u8],
    width: usize,
    height: usize,
) -> Option<(Vec<[u8; 3]>, Option<u8>)> {
    let n = width * height;
    if n < 64 * 64 {
        return None;
    }
    // The corrected alpha plane defines regions. Partial pixels have no
    // equal neighbors, even when their RGB and alpha values agree.
    const PARTIAL: u32 = TRANSPARENT + 1;
    let code_at = |i: usize| match alpha[i] {
        0 => TRANSPARENT,
        255 => {
            let [r, g, b, _] = px[i];
            color_key([r, g, b])
        }
        _ => PARTIAL,
    };
    let mut flat = 0usize;
    let mut sampled = 0usize;
    for y in (1..height - 1).step_by(8) {
        for x in (1..width - 1).step_by(8) {
            let i = y * width + x;
            let code = code_at(i);
            if code != PARTIAL {
                let same = usize::from(code_at(i - 1) == code)
                    + usize::from(code_at(i + 1) == code)
                    + usize::from(code_at(i - width) == code)
                    + usize::from(code_at(i + width) == code);
                flat += usize::from(same >= 2);
            }
            sampled += 1;
        }
    }
    if (flat as f32) < MIN_FLAT_SAMPLE_SHARE * sampled as f32 {
        return None;
    }

    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for y in 1..height - 1 {
        let row = y * width;
        let mut start = 0;
        while start < width {
            let code = code_at(row + start);
            let mut end = start + 1;
            while end < width && code_at(row + end) == code {
                end += 1;
            }
            if code != PARTIAL {
                let len = end - start;
                let last = end - 1;
                let mut flat = len.saturating_sub(2);
                if start > 0 && start < width - 1 {
                    let i = row + start;
                    let same = usize::from(len > 1)
                        + usize::from(code_at(i - width) == code)
                        + usize::from(code_at(i + width) == code);
                    flat += usize::from(same >= 2);
                }
                if len > 1 && last < width - 1 {
                    let i = row + last;
                    flat += usize::from(code_at(i - width) == code || code_at(i + width) == code);
                }
                if flat > 0 {
                    *counts.entry(code).or_insert(0) += flat;
                    if counts.len() > ColorTable::CAPACITY / 2 {
                        return None;
                    }
                }
            }
            start = end;
        }
    }
    let min_pixels = MIN_PALETTE_PIXELS.max(n / 10_000);
    let mut palette: Vec<_> = counts
        .into_iter()
        .filter(|&(_, c)| c >= min_pixels)
        .collect();
    if palette.is_empty() {
        return None;
    }
    palette.sort_unstable_by_key(|&(c, count)| (std::cmp::Reverse(count), c));
    palette.truncate(MAX_PALETTE);
    let ground = palette
        .iter()
        .position(|&(c, _)| c == TRANSPARENT)
        .map(|k| k as u8);
    let colors = palette
        .into_iter()
        .map(|(c, _)| {
            let [_, r, g, b] = c.to_be_bytes();
            [r, g, b]
        })
        .collect();
    Some((colors, ground))
}

/// Palette analysis shared by the plain and snapped candidates.
pub(crate) struct SplitSource<'a, const CHANNELS: usize> {
    pixels: &'a [[u8; CHANNELS]],
    alpha: Option<&'a [u8]>,
    width: usize,
    height: usize,
    palette: Vec<[u8; 3]>,
    ground: Option<u8>,
}

impl<'a> SplitSource<'a, 3> {
    pub(crate) fn rgb(rgb: &'a [u8], width: usize, height: usize) -> Option<Self> {
        let pixels = rgb.as_chunks::<3>().0;
        Some(Self {
            pixels,
            alpha: None,
            width,
            height,
            palette: rgb_palette(pixels, width, height)?,
            ground: None,
        })
    }
}

impl<'a> SplitSource<'a, 4> {
    /// Opaque colors and transparent ground form the palette. The coverage
    /// band carries edges between colors and edges against the ground.
    pub(crate) fn rgba(
        pixels: &'a [[u8; 4]],
        alpha: &'a [u8],
        width: usize,
        height: usize,
    ) -> Option<Self> {
        let (palette, ground) = rgba_palette(pixels, alpha, width, height)?;
        Some(Self {
            pixels,
            alpha: Some(alpha),
            width,
            height,
            palette,
            ground,
        })
    }
}

impl<'pixels, const CHANNELS: usize> SplitSource<'pixels, CHANNELS> {
    #[inline]
    fn pixel(&self, i: usize) -> [u8; 3] {
        let p = self.pixels[i];
        [p[0], p[1], p[2]]
    }

    fn labels(&self) -> Vec<u8> {
        // The empty slot's low byte is NO_LABEL, which no palette entry uses.
        const LABEL_BITS: u32 = 9;
        const LABEL_SLOTS: usize = 1 << LABEL_BITS;
        let mut index = [u32::MAX; LABEL_SLOTS];
        for (k, &c) in self.palette.iter().enumerate() {
            if Some(k as u8) == self.ground {
                continue;
            }
            let key = color_key(c);
            let mut slot = color_slot::<LABEL_BITS>(key);
            while index[slot] != u32::MAX {
                slot = (slot + 1) & (LABEL_SLOTS - 1);
            }
            index[slot] = (key << 8) | k as u32;
        }
        let label = |color: [u8; 3]| {
            let key = color_key(color);
            let mut slot = color_slot::<LABEL_BITS>(key);
            loop {
                let entry = index[slot];
                if entry == u32::MAX || entry >> 8 == key {
                    break entry as u8;
                }
                slot = (slot + 1) & (LABEL_SLOTS - 1);
            }
        };
        if let Some(alpha) = self.alpha {
            self.pixels
                .iter()
                .zip(alpha)
                .map(|(p, &a)| match a {
                    255 => label([p[0], p[1], p[2]]),
                    0 => self.ground.unwrap_or(NO_LABEL),
                    _ => NO_LABEL,
                })
                .collect()
        } else {
            let mut labels = Vec::with_capacity(self.pixels.len());
            for run in self.pixels.chunk_by(|a, b| a == b) {
                let p = run[0];
                labels.resize(labels.len() + run.len(), label([p[0], p[1], p[2]]));
            }
            labels
        }
    }

    /// The split's plan; `snap` folds resampling fringe within that many
    /// levels onto the palette (see [`snap_reach`]).
    pub(crate) fn prepare<'source>(
        &'source self,
        snap: Option<u8>,
    ) -> Option<SplitPlan<'source, 'pixels, CHANNELS>> {
        let n = self.width * self.height;
        let mut labels = self.labels();
        let snapped = if let Some(max_reach) = snap {
            let moved = snap_fringe(
                |i| self.pixel(i),
                self.width,
                self.height,
                &self.palette,
                self.ground,
                &mut labels,
                self.alpha,
                max_reach,
            );
            if moved.is_empty() {
                return None;
            }
            moved
        } else {
            Vec::new()
        };

        let mut area = [0usize; MAX_PALETTE];
        let mut unlabeled = 0;
        let min_exact = MIN_EXACT_SHARE * n as f32;
        for run in labels.chunk_by(|a, b| a == b) {
            let label = run[0];
            if label == NO_LABEL {
                unlabeled += run.len();
                if ((n - unlabeled) as f32) < min_exact {
                    return None;
                }
            } else {
                area[label as usize] += run.len();
            }
        }
        let colors: Vec<_> = self.palette.iter().map(|&c| c.map(f32::from)).collect();
        let dominant = (0..colors.len())
            .filter(|&k| Some(k as u8) != self.ground)
            .max_by_key(|&k| area[k])
            .map(|k| k as u8);
        let mut mixes = Vec::new();
        let mut textured = 0usize;
        let max_textured = MAX_TEXTURED_SHARE * n as f32;
        let stride = self.width as isize;
        let offsets =
            MIX_OFFSETS.map(|(dx, dy)| stride.wrapping_mul(dy as isize).wrapping_add(dx as isize));
        let mut start = 0;
        for run in labels.chunk_by(|a, b| a == b) {
            let end = start + run.len();
            if run[0] == NO_LABEL {
                for i in start..end {
                    let (mut neighbors, mut count) =
                        mix_neighbors(i, &labels, self.width, self.height, &offsets);
                    let p = self.pixel(i).map(f32::from);
                    let alpha = self.alpha.map_or(255, |alpha| alpha[i]);
                    let mix = if alpha != 255 {
                        self.ground
                            .filter(|g| neighbors[..count].contains(g))
                            .and_then(|ground| {
                                alpha_mix(p, alpha, &neighbors[..count], &colors, ground)
                            })
                    } else {
                        if count == 1
                            && let Some(dominant) = dominant
                            && neighbors[0] != dominant
                        {
                            neighbors[1] = dominant;
                            count = 2;
                        }
                        color_mix(p, &neighbors[..count], &colors, &area, self.ground)
                    };
                    if let Some(mix) = mix {
                        mixes.push((i, mix));
                    } else {
                        textured += 1;
                        if textured as f32 > max_textured {
                            return None;
                        }
                    }
                }
            }
            start = end;
        }
        Some(SplitPlan {
            source: self,
            snapped,
            mixes,
            dominant,
        })
    }
}

fn mix_neighbors(
    i: usize,
    labels: &[u8],
    width: usize,
    height: usize,
    offsets: &[isize; MIX_OFFSET_COUNT],
) -> ([u8; 8], usize) {
    let (x, y) = ((i % width) as i32, (i / width) as i32);
    let edge = x
        .min(width as i32 - 1 - x)
        .min(y.min(height as i32 - 1 - y));
    let mut neighbors = [NO_LABEL; 8];
    let mut seen = [0u64; 4];
    let mut count = 0;
    let mut start = 0;
    for ring in 1..=MIX_REACH {
        let end = start + ring as usize * 8;
        for j in start..end {
            if ring > edge {
                let (dx, dy) = MIX_OFFSETS[j];
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx >= width as i32 || yy >= height as i32 {
                    continue;
                }
            }
            let label = labels[i.wrapping_add_signed(offsets[j])];
            if label == NO_LABEL {
                continue;
            }
            let mask = &mut seen[usize::from(label >> 6)];
            let bit = 1 << (label & 63);
            if *mask & bit == 0 {
                *mask |= bit;
                neighbors[count] = label;
                count += 1;
                if count == neighbors.len() {
                    break;
                }
            }
        }
        if count >= 2 {
            break;
        }
        start = end;
    }
    (neighbors, count)
}

#[derive(Clone, Copy)]
struct Mix {
    /// NO_LABEL for edges against the transparent ground.
    bottom: u8,
    top: u8,
    coverage: f32,
}

impl Mix {
    #[inline]
    fn step(self, steps: u32) -> u32 {
        round_i32(self.coverage * steps as f32) as u32
    }
}

#[inline]
fn coverage_byte(step: u32, steps: u32) -> u8 {
    ((step * 255 + steps / 2) / steps) as u8
}

fn alpha_mix(
    p: [f32; 3],
    alpha: u8,
    neighbors: &[u8],
    colors: &[[f32; 3]],
    ground: u8,
) -> Option<Mix> {
    let coverage = alpha as f32 / 255.0;
    let (error, top) = neighbors
        .iter()
        .filter(|&&l| l != ground)
        .map(|&l| {
            let t = colors[l as usize];
            let r: f32 = (0..3).map(|k| (p[k] - t[k]).powi(2)).sum();
            (r, l)
        })
        .min_by(|x, y| x.0.total_cmp(&y.0))?;
    (error * coverage * coverage <= MAX_MIX_RESIDUAL_SQ).then_some(Mix {
        bottom: NO_LABEL,
        top,
        coverage,
    })
}

fn color_mix(
    p: [f32; 3],
    neighbors: &[u8],
    colors: &[[f32; 3]],
    area: &[usize],
    ground: Option<u8>,
) -> Option<Mix> {
    let mut best: Option<(f32, u8, u8, f32)> = None;
    for (s, &a) in neighbors.iter().enumerate() {
        if Some(a) == ground {
            continue;
        }
        let ca = colors[a as usize];
        let dp: [f32; 3] = std::array::from_fn(|k| p[k] - ca[k]);
        for &b in &neighbors[s + 1..] {
            if Some(b) == ground {
                continue;
            }
            let cb = colors[b as usize];
            let d: [f32; 3] = std::array::from_fn(|k| cb[k] - ca[k]);
            let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if dd < 100.0 {
                continue;
            }
            let c = ((0..3).map(|k| dp[k] * d[k]).sum::<f32>() / dd).clamp(0.0, 1.0);
            let r: f32 = (0..3).map(|k| (dp[k] - c * d[k]).powi(2)).sum();
            if best.is_none_or(|(br, ..)| r < br) {
                best = Some((r, a, b, c));
            }
        }
    }
    let (_, a, b, c) = best.filter(|&(r, ..)| r <= MAX_MIX_RESIDUAL_SQ)?;
    // The smaller region is drawn on top of the larger one.
    let (bottom, top, coverage) = if area[a as usize] >= area[b as usize] {
        (a, b, c)
    } else {
        (b, a, 1.0 - c)
    };
    Some(Mix {
        bottom,
        top,
        coverage,
    })
}

/// Unquantized edges for one palette variant. Precision retries only
/// quantize these edges and render the resulting frames.
pub(crate) struct SplitPlan<'source, 'pixels, const CHANNELS: usize> {
    source: &'source SplitSource<'pixels, CHANNELS>,
    snapped: Vec<(usize, u8)>,
    mixes: Vec<(usize, Mix)>,
    dominant: Option<u8>,
}

impl<const CHANNELS: usize> SplitPlan<'_, '_, CHANNELS> {
    pub(crate) fn split(&self, steps: u32) -> Option<VectorSplit> {
        let source = self.source;
        let n = source.width * source.height;
        let palette = &source.palette;
        let mut top_counts = [0usize; MAX_PALETTE];
        let mut band = 0;
        let max_band = MAX_BAND_SHARE * n as f32;
        for &(_, mix) in &self.mixes {
            let step = mix.step(steps);
            if step > 0 && step < steps {
                top_counts[mix.top as usize] += 1;
                band += 1;
                if band as f32 > max_band {
                    return None;
                }
            }
        }
        let backing = self.dominant.map_or([0; 3], |k| palette[k as usize]);
        let fill = (0..palette.len())
            .filter(|&k| Some(k as u8) != source.ground)
            .max_by_key(|&k| top_counts[k])
            .map_or(backing, |k| palette[k]);

        let mut base: Vec<_> = source.pixels.iter().map(|p| [p[0], p[1], p[2]]).collect();
        let mut base_alpha = source.alpha.map(<[u8]>::to_vec);
        let mut coverage = if band == 0 { Vec::new() } else { vec![0; n] };
        let mut layer = if band == 0 { Vec::new() } else { vec![fill; n] };
        for &(i, k) in &self.snapped {
            base[i] = palette[k as usize];
        }
        for &(i, mix) in &self.mixes {
            let step = mix.step(steps);
            let color = if mix.bottom == NO_LABEL {
                base_alpha.as_mut().unwrap()[i] = if step == steps { 255 } else { 0 };
                mix.top
            } else if step == steps {
                mix.top
            } else {
                mix.bottom
            };
            base[i] = palette[color as usize];
            if step > 0 && step < steps {
                coverage[i] = coverage_byte(step, steps);
                layer[i] = palette[mix.top as usize];
            }
        }
        if let Some(alpha) = &base_alpha {
            // Each transparent run continues the color to its left so that
            // the color planes have no edge where only alpha has one.
            for (row, (base, alpha)) in base
                .chunks_exact_mut(source.width)
                .zip(alpha.chunks_exact(source.width))
                .enumerate()
            {
                let mut last = alpha
                    .iter()
                    .position(|&a| a != 0)
                    .map_or(backing, |x| base[x]);
                let offset = row * source.width;
                for (x, (color, &a)) in base.iter_mut().zip(alpha).enumerate() {
                    if a == 0 && (band == 0 || coverage[offset + x] == 0) {
                        *color = last;
                    } else {
                        last = *color;
                    }
                }
            }
        }
        Some(VectorSplit {
            base: base.into_flattened(),
            layer: layer.into_flattened(),
            coverage,
            base_alpha,
        })
    }
}

impl SplitPlan<'_, '_, 3> {
    /// Render a one-frame RGB retry directly from the sparse edge plan.
    /// The coverage and upper-color planes are only needed for layered frames.
    pub(crate) fn split_for_retry(&self, steps: u32, flattened: bool) -> Option<VectorSplit> {
        if !flattened {
            return self.split(steps);
        }
        let palette = &self.source.palette;
        let mut base = self.source.pixels.to_vec();
        for &(i, label) in &self.snapped {
            base[i] = palette[label as usize];
        }
        let mut band = 0;
        let max_band = MAX_BAND_SHARE * base.len() as f32;
        for &(i, mix) in &self.mixes {
            let step = mix.step(steps);
            let bottom = palette[mix.bottom as usize];
            let top = palette[mix.top as usize];
            base[i] = if step == steps {
                top
            } else if step == 0 {
                bottom
            } else {
                band += 1;
                if band as f32 > max_band {
                    return None;
                }
                let alpha = u16::from(coverage_byte(step, steps));
                std::array::from_fn(|c| {
                    ((u16::from(bottom[c]) * (255 - alpha) + u16::from(top[c]) * alpha + 127) / 255)
                        as u8
                })
            };
        }
        Some(VectorSplit {
            base: base.into_flattened(),
            layer: Vec::new(),
            coverage: Vec::new(),
            base_alpha: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_source<'a>(rgba: &'a [[u8; 4]], alpha: &'a [u8], size: usize) -> SplitSource<'a, 4> {
        SplitSource::rgba(rgba, alpha, size, size).expect("has a palette")
    }

    fn interleave(rgb: &[u8], alpha: &[u8]) -> Vec<[u8; 4]> {
        rgb.as_chunks::<3>()
            .0
            .iter()
            .zip(alpha)
            .map(|(&[r, g, b], &a)| [r, g, b, a])
            .collect()
    }

    fn split_snapped(rgb: &[u8], width: usize, height: usize, steps: u32) -> Option<VectorSplit> {
        SplitSource::rgb(rgb, width, height)?
            .prepare(Some(SNAP_REACH))?
            .split(steps)
    }

    /// The opaque disc of the cut-out with stray fringe pixels inside its rim:
    /// the snapped split folds the ring away and leaves the edge to the band.
    #[test]
    fn rgba_snap_folds_fringe_of_opaque_regions_only() {
        let size = 128usize;
        let mut rgb = vec![0u8; size * size * 3];
        let mut alpha = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let d = ((x as f32 - 64.0).powi(2) + (y as f32 - 64.0).powi(2)).sqrt();
                let a = ((40.5 - d).clamp(0.0, 1.0) * 255.0).round() as u8;
                let i = y * size + x;
                alpha[i] = a;
                // Sparse fringe, three levels either side of the fill.
                let v = if (30.0..33.0).contains(&d) && x % 4 == 0 && y % 4 == 0 {
                    if (x / 4 + y / 4) % 2 == 0 { 203 } else { 197 }
                } else {
                    200
                };
                rgb[i * 3..i * 3 + 3].copy_from_slice(&[v, 40, 30]);
            }
        }
        let rgba = interleave(&rgb, &alpha);
        let source = rgba_source(&rgba, &alpha, size);
        let plain = source
            .prepare(None)
            .and_then(|plan| plan.split(15))
            .expect("splits");
        let snapped = source
            .prepare(Some(SNAP_REACH))
            .and_then(|plan| plan.split(15))
            .expect("fringe snaps");
        let colors = |split: &VectorSplit| -> std::collections::HashSet<[u8; 3]> {
            split.base.as_chunks::<3>().0.iter().copied().collect()
        };
        assert!(colors(&plain).len() > 1);
        assert_eq!(colors(&snapped), [[200, 40, 30]].into_iter().collect());
        assert_eq!(snapped.base_alpha, plain.base_alpha);
        assert_eq!(snapped.coverage, plain.coverage);
    }

    /// An opaque disc on a transparent ground with an antialiased rim: the
    /// base is the hard disc, the band carries the rim at the disc's color.
    #[test]
    fn rgba_split_puts_edges_against_transparency_on_the_band() {
        let size = 128usize;
        let mut rgb = vec![0u8; size * size * 3];
        let mut alpha = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let d = ((x as f32 - 64.0).powi(2) + (y as f32 - 64.0).powi(2)).sqrt();
                let a = ((40.5 - d).clamp(0.0, 1.0) * 255.0).round() as u8;
                let i = y * size + x;
                alpha[i] = a;
                // Transparent pixels hold arbitrary color.
                let color = if a == 0 {
                    [x as u8, y as u8, 7]
                } else {
                    [200, 40, 30]
                };
                rgb[i * 3..i * 3 + 3].copy_from_slice(&color);
            }
        }
        let rgba = interleave(&rgb, &alpha);
        let split = rgba_source(&rgba, &alpha, size)
            .prepare(None)
            .and_then(|plan| plan.split(15))
            .expect("flat cut-out splits");
        let base_alpha = split.base_alpha.expect("base alpha");
        assert!(base_alpha.iter().all(|&a| a == 0 || a == 255));
        let band: Vec<usize> = (0..size * size)
            .filter(|&i| split.coverage[i] != 0)
            .collect();
        assert!(!band.is_empty());
        for i in band {
            assert_eq!(base_alpha[i], 0);
            assert_eq!(&split.layer[i * 3..i * 3 + 3], &[200, 40, 30]);
            assert!(split.coverage[i].abs_diff(alpha[i]) <= 9);
        }
        // The ground's arbitrary color is gone from the base.
        assert!(
            split
                .base
                .as_chunks::<3>()
                .0
                .iter()
                .all(|&c| c == [200, 40, 30])
        );
    }

    fn colors(rgb: &[u8]) -> std::collections::HashSet<[u8; 3]> {
        rgb.as_chunks::<3>().0.iter().copied().collect()
    }

    /// Two flat fields with an alternating halo on each side of the edge, as
    /// a resampling filter leaves it: the halo colors become palette entries of
    /// the plain split and fold away in the snapped one.
    #[test]
    fn snap_folds_resampling_fringe() {
        let size = 128;
        let mut rgb = Vec::with_capacity(size * size * 3);
        for _ in 0..size {
            for x in 0..size {
                let v: u8 = match x {
                    0..=61 => 40,
                    62 => 37,
                    63 => 43,
                    64 => 203,
                    65 => 197,
                    _ => 200,
                };
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        let plain = split(&rgb, size, size, 15).expect("flat art splits");
        assert_eq!(colors(&plain.base).len(), 6);
        let snapped = split_snapped(&rgb, size, size, 15).expect("fringe snaps");
        assert_eq!(
            colors(&snapped.base),
            [[40; 3], [200; 3]].into_iter().collect()
        );
    }

    /// Wide bands three levels apart are shading, not fringe: flattening them
    /// shifts the local mean, so nothing snaps.
    #[test]
    fn snap_keeps_shade_bands() {
        let size = 128;
        let mut rgb = Vec::with_capacity(size * size * 3);
        for _ in 0..size {
            for x in 0..size {
                let v = 100 + 3 * (x / 16) as u8;
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        assert!(split(&rgb, size, size, 15).is_some());
        assert!(split_snapped(&rgb, size, size, 15).is_none());
    }

    #[test]
    fn snap_uses_all_proposed_errors_at_clipped_edges() {
        let palette = [[100; 3]];
        let mut px = [[100; 3]; 12];
        let mut labels = [0; 12];
        for (i, red) in [(5, 104), (9, 104), (11, 96)] {
            px[i][0] = red;
            labels[i] = NO_LABEL;
        }
        // The rejected proposal at 9 still cancels the error at 11.
        assert_eq!(
            snap_fringe(
                |i| px[i],
                12,
                1,
                &palette,
                None,
                &mut labels,
                None,
                SNAP_REACH
            ),
            [(11, 0)]
        );
        assert_eq!([labels[5], labels[9], labels[11]], [NO_LABEL, NO_LABEL, 0]);

        let mut px = [[100; 3]; 8];
        let mut labels = [0; 8];
        px[4][2] = 104;
        labels[4] = NO_LABEL;
        // A total error of 4 across 8 pixels is exactly the allowed mean.
        assert_eq!(
            snap_fringe(
                |i| px[i],
                8,
                1,
                &palette,
                None,
                &mut labels,
                None,
                SNAP_REACH
            ),
            [(4, 0)]
        );
    }

    #[test]
    fn snap_keeps_palette_order_across_red_buckets() {
        let palette = [[252, 100, 100], [244, 100, 100]];
        let mut px = [palette[0]; 81];
        let mut labels = [0; 81];
        px[40][0] = 248;
        px[41][0] = 255;
        labels[40..42].fill(NO_LABEL);
        assert_eq!(
            snap_fringe(
                |i| px[i],
                9,
                9,
                &palette,
                None,
                &mut labels,
                None,
                SNAP_REACH
            ),
            [(40, 0), (41, 0)]
        );
    }
}
