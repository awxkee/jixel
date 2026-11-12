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
    let px = rgb.as_chunks::<3>().0;
    let d = distance.max(0.1);
    let cost: [f32; 256] = std::array::from_fn(|v| (1.0 + v as f32 / (4.0 * d)).log2());
    let step = |a: [u8; 3], b: [u8; 3]| {
        cost[(0..3).map(|k| a[k].abs_diff(b[k])).max().unwrap() as usize] as f64
    };
    let mut bits = 0.0;
    for y in 0..height {
        let row = &px[y * width..(y + 1) * width];
        for x in 1..width {
            bits += step(row[x - 1], row[x]);
        }
        if y > 0 {
            let above = &px[(y - 1) * width..y * width];
            for x in 0..width {
                bits += step(above[x], row[x]);
            }
        }
    }
    1.16 * bits / 8.0 * d.powf(-0.41) as f64
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
    /// zero when the art is a bare palette or the band was dropped.
    pub(crate) coverage: Vec<u8>,
}

pub(crate) fn split(rgb: &[u8], width: usize, height: usize, steps: u32) -> Option<VectorSplit> {
    split_impl(rgb, width, height, steps, false)
}

/// [`split`] with resampling fringe folded onto the palette: palette colors
/// within `SNAP_REACH` of a larger one merge into it and stray pixels that
/// close to a palette color take it, wherever the change averages out
/// locally.
pub(crate) fn split_snapped(
    rgb: &[u8],
    width: usize,
    height: usize,
    steps: u32,
) -> Option<VectorSplit> {
    split_impl(rgb, width, height, steps, true)
}

/// Pixels the snap moves, each with its new palette index; `labels` is
/// updated to match.
fn snap_fringe(
    px: &[[u8; 3]],
    width: usize,
    height: usize,
    palette: &[([u8; 3], usize)],
    labels: &mut [u8],
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
        if let Some(j) =
            (0..k).find(|&j| root[j] == j as u8 && reach(palette[j].0, palette[k].0) <= SNAP_REACH)
        {
            root[k] = j as u8;
        }
    }
    // Each red bucket lists the roots whose snap range reaches it. Keep
    // palette order so the first root wins ties in the full RGB distance.
    const RED_BUCKET_SHIFT: usize = 3;
    let red_bucket = |red: u8| usize::from(red) >> RED_BUCKET_SHIFT;
    let mut roots_by_red = None;
    let mut nearest = std::collections::HashMap::<[u8; 3], u8>::new();
    let mut moved = Vec::new();
    for (i, (&l, &p)) in labels.iter().zip(px).enumerate() {
        let target = if l != NO_LABEL {
            root[l as usize]
        } else {
            *nearest.entry(p).or_insert_with(|| {
                let buckets = roots_by_red.get_or_insert_with(|| {
                    let mut buckets: [Vec<_>; 256 >> RED_BUCKET_SHIFT] =
                        std::array::from_fn(|_| Vec::new());
                    for (k, &(color, _)) in palette.iter().enumerate() {
                        if root[k] == k as u8 {
                            let first = red_bucket(color[0].saturating_sub(SNAP_REACH));
                            let last = red_bucket(color[0].saturating_add(SNAP_REACH));
                            for bucket in &mut buckets[first..=last] {
                                bucket.push((color, k as u8));
                            }
                        }
                    }
                    buckets
                });
                let mut best = (SNAP_REACH + 1, NO_LABEL);
                for &(c, k) in &buckets[red_bucket(p[0])] {
                    let d = reach(c, p);
                    if d < best.0 {
                        best = (d, k);
                    }
                }
                best.1
            })
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
        let color = palette[k as usize].0;
        std::array::from_fn(|c| i16::from(color[c]) - i16::from(px[i][c]))
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
const MIX_OFFSETS: [(i32, i32); MIX_OFFSET_COUNT] = {
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

fn split_impl(
    rgb: &[u8],
    width: usize,
    height: usize,
    steps: u32,
    snap: bool,
) -> Option<VectorSplit> {
    let n = width * height;
    if n < 64 * 64 {
        return None;
    }
    let px = rgb.as_chunks::<3>().0;
    // Flat art has such pixels nearly everywhere (0.70 and up), photos and
    // textured screens below 0.25: one pixel in 64 settles it.
    let mut flat = 0usize;
    let mut sampled = 0usize;
    for y in (1..height - 1).step_by(8) {
        for x in (1..width - 1).step_by(8) {
            let i = y * width + x;
            let p = px[i];
            let same = usize::from(px[i - 1] == p)
                + usize::from(px[i + 1] == p)
                + usize::from(px[i - width] == p)
                + usize::from(px[i + width] == p);
            flat += usize::from(same >= 2);
            sampled += 1;
        }
    }
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
    // Index only palette colors. Store the RGB key and label in one word.
    // The empty slot's low byte is NO_LABEL, which no palette entry uses.
    const LABEL_BITS: u32 = 9;
    const LABEL_SLOTS: usize = 1 << LABEL_BITS;
    let mut palette_index = [u32::MAX; LABEL_SLOTS];
    for (k, &(c, _)) in palette.iter().enumerate() {
        let key = color_key(c);
        let mut slot = color_slot::<LABEL_BITS>(key);
        while palette_index[slot] != u32::MAX {
            slot = (slot + 1) & (LABEL_SLOTS - 1);
        }
        palette_index[slot] = (key << 8) | k as u32;
    }
    let mut labels = Vec::with_capacity(n);
    for run in px.chunk_by(|a, b| a == b) {
        let key = color_key(run[0]);
        let mut slot = color_slot::<LABEL_BITS>(key);
        let label = loop {
            let entry = palette_index[slot];
            if entry == u32::MAX || entry >> 8 == key {
                break entry as u8;
            }
            slot = (slot + 1) & (LABEL_SLOTS - 1);
        };
        labels.resize(labels.len() + run.len(), label);
    }
    let snapped = if snap {
        let moved = snap_fringe(px, width, height, &palette, &mut labels);
        if moved.is_empty() {
            return None;
        }
        moved
    } else {
        Vec::new()
    };
    let mut area = vec![0usize; palette.len()];
    let mut unlabeled = Vec::new();
    let mut start = 0;
    let min_exact = MIN_EXACT_SHARE * n as f32;
    for run in labels.chunk_by(|a, b| a == b) {
        let end = start + run.len();
        let label = run[0];
        if label == NO_LABEL {
            if ((n - unlabeled.len() - run.len()) as f32) < min_exact {
                return None;
            }
            unlabeled.extend(start..end);
        } else {
            area[label as usize] += run.len();
        }
        start = end;
    }
    let colors: Vec<[f32; 3]> = palette.iter().map(|&(c, _)| c.map(f32::from)).collect();
    let dominant = (0..colors.len()).max_by_key(|&k| area[k]).unwrap() as u8;
    // Unblend every non-palette pixel between two neighboring region colors.
    let mut base = rgb.to_vec();
    for (i, k) in snapped {
        base[i * 3..i * 3 + 3].copy_from_slice(&palette[k as usize].0);
    }
    let mut band_pixels = Vec::new();
    let mut textured = 0usize;
    let mut top_counts = vec![0usize; colors.len()];
    let max_textured = MAX_TEXTURED_SHARE * n as f32;
    let max_band = MAX_BAND_SHARE * n as f32;
    let stride = width as isize;
    let offsets =
        MIX_OFFSETS.map(|(dx, dy)| stride.wrapping_mul(dy as isize).wrapping_add(dx as isize));
    let mut neighbors = [NO_LABEL; 8];
    for i in unlabeled {
        let (x, y) = ((i % width) as i32, (i / width) as i32);
        let edge = x
            .min(width as i32 - 1 - x)
            .min(y.min(height as i32 - 1 - y));
        let mut seen = [0u64; 4];
        let mut m = 0;
        let mut start = 0;
        // An edge softer than one pixel can need a wider neighborhood.
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
                    neighbors[m] = label;
                    m += 1;
                    if m == neighbors.len() {
                        break;
                    }
                }
            }
            if m >= 2 {
                break;
            }
            start = end;
        }
        if m == 1 && neighbors[0] != dominant {
            neighbors[1] = dominant;
            m = 2;
        }
        let p = px[i].map(f32::from);
        let mut best: Option<(f32, u8, u8, f32)> = None; // residual, a, b, c
        for s in 0..m {
            let a = neighbors[s];
            let ca = colors[a as usize];
            let dp: [f32; 3] = std::array::from_fn(|k| p[k] - ca[k]);
            for &b in &neighbors[s + 1..m] {
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
        let Some((_, a, b, c)) = best.filter(|&(r, ..)| r <= MAX_MIX_RESIDUAL_SQ) else {
            textured += 1;
            if textured as f32 > max_textured {
                return None;
            }
            continue; // kept verbatim in the base
        };
        // The smaller region is drawn on top of the larger one.
        let (bottom, top, c_top) = if area[a as usize] >= area[b as usize] {
            (a, b, c)
        } else {
            (b, a, 1.0 - c)
        };
        let step = round_i32(c_top * steps as f32) as u32;
        let base_color = if step == steps { top } else { bottom };
        base[i * 3..i * 3 + 3].copy_from_slice(&palette[base_color as usize].0);
        if step > 0 && step < steps {
            let alpha = ((step * 255 + steps / 2) / steps) as u8;
            band_pixels.push((i, top, alpha));
            top_counts[top as usize] += 1;
            if band_pixels.len() as f32 > max_band {
                return None;
            }
        }
    }
    drop(labels);
    // The coverage frame's color: the upper region on the band, the most
    // common band color everywhere else (cheapest for the palette coder).
    let fill = (0..colors.len()).max_by_key(|&k| top_counts[k]).unwrap();
    let mut coverage = vec![0; n];
    let mut layer = vec![palette[fill].0; n];
    for (i, top, alpha) in band_pixels {
        layer[i] = palette[top as usize].0;
        coverage[i] = alpha;
    }
    Some(VectorSplit {
        base,
        layer: layer.into_flattened(),
        coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let palette = [([100; 3], 12)];
        let mut px = [[100; 3]; 12];
        let mut labels = [0; 12];
        for (i, red) in [(5, 104), (9, 104), (11, 96)] {
            px[i][0] = red;
            labels[i] = NO_LABEL;
        }
        // The rejected proposal at 9 still cancels the error at 11.
        assert_eq!(snap_fringe(&px, 12, 1, &palette, &mut labels), [(11, 0)]);
        assert_eq!([labels[5], labels[9], labels[11]], [NO_LABEL, NO_LABEL, 0]);

        let mut px = [[100; 3]; 8];
        let mut labels = [0; 8];
        px[4][2] = 104;
        labels[4] = NO_LABEL;
        // A total error of 4 across 8 pixels is exactly the allowed mean.
        assert_eq!(snap_fringe(&px, 8, 1, &palette, &mut labels), [(4, 0)]);
    }

    #[test]
    fn snap_keeps_palette_order_across_red_buckets() {
        let palette = [([252, 100, 100], 40), ([244, 100, 100], 32)];
        let mut px = [palette[0].0; 81];
        let mut labels = [0; 81];
        px[40][0] = 248;
        px[41][0] = 255;
        labels[40..42].fill(NO_LABEL);
        assert_eq!(
            snap_fringe(&px, 9, 9, &palette, &mut labels),
            [(40, 0), (41, 0)]
        );
    }
}
