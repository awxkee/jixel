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

/// Open-addressing table from an exact color to a count or an index; far
/// cheaper than a hash map for the few thousand colors of flat art.
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

    fn slot(&self, key: u32) -> Option<usize> {
        let mut i = (key.wrapping_mul(0x9E37_79B1) >> (32 - Self::BITS)) as usize;
        loop {
            let k = self.keys[i];
            if k == key {
                return Some(i);
            }
            if k == Self::EMPTY {
                return None;
            }
            i = (i + 1) & (Self::CAPACITY - 1);
        }
    }

    fn key(color: [u8; 3]) -> u32 {
        u32::from_be_bytes([0, color[0], color[1], color[2]])
    }

    /// Count one more pixel of `color`; false once the table is half full.
    fn add(&mut self, color: [u8; 3]) -> bool {
        let key = Self::key(color);
        let mut i = (key.wrapping_mul(0x9E37_79B1) >> (32 - Self::BITS)) as usize;
        loop {
            let k = self.keys[i];
            if k == key {
                self.values[i] += 1;
                return true;
            }
            if k == Self::EMPTY {
                if self.len >= Self::CAPACITY / 2 {
                    return false;
                }
                self.keys[i] = key;
                self.values[i] = 1;
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

    fn clear_values(&mut self) {
        self.values.fill(Self::EMPTY);
    }

    fn set(&mut self, color: [u8; 3], value: u32) {
        let i = self.slot(Self::key(color)).expect("palette color");
        self.values[i] = value;
    }

    fn get(&self, color: [u8; 3]) -> Option<u32> {
        self.slot(Self::key(color))
            .map(|i| self.values[i])
            .filter(|&v| v != Self::EMPTY)
    }
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
    // repeat along a line).
    let mut counts = ColorTable::new();
    for y in 1..height - 1 {
        let row = y * width;
        for x in 1..width - 1 {
            let i = row + x;
            let p = px[i];
            let same = usize::from(px[i - 1] == p)
                + usize::from(px[i + 1] == p)
                + usize::from(px[i - width] == p)
                + usize::from(px[i + width] == p);
            if same >= 2 && !counts.add(p) {
                return None; // more flat colors than any palette art has
            }
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
    let colors: Vec<[f32; 3]> = palette.iter().map(|&(c, _)| c.map(f32::from)).collect();
    // The table now maps each palette color to its index.
    counts.clear_values();
    for (k, &(c, _)) in palette.iter().enumerate() {
        counts.set(c, k as u32);
    }
    // Labels and region areas.
    let labels: Vec<u8> = px
        .iter()
        .map(|&p| counts.get(p).map_or(NO_LABEL, |k| k as u8))
        .collect();
    let mut area = vec![0usize; colors.len()];
    let mut exact = 0usize;
    for &l in &labels {
        if l != NO_LABEL {
            area[l as usize] += 1;
            exact += 1;
        }
    }
    if (exact as f32) < MIN_EXACT_SHARE * n as f32 {
        return None;
    }
    let dominant = (0..colors.len()).max_by_key(|&k| area[k]).unwrap() as u8;
    // Unblend every non-palette pixel between two neighboring region colors.
    let mut base = rgb.to_vec();
    let mut coverage = vec![0u8; n];
    let mut top_of = vec![NO_LABEL; n];
    let mut band = 0usize;
    let mut textured = 0usize;
    let mut top_counts = vec![0usize; colors.len()];
    let mut neighbors = [NO_LABEL; 8];
    for i in 0..n {
        if labels[i] != NO_LABEL {
            continue;
        }
        let (x, y) = (i % width, i / width);
        let mut m = 0;
        // Ring by ring until two region colors are in reach: an edge softer
        // than one pixel has mixes whose 3x3 holds no region pixel at all.
        for ring in 1..=MIX_REACH {
            for dy in -ring..=ring {
                for dx in -ring..=ring {
                    if dx.abs().max(dy.abs()) != ring {
                        continue;
                    }
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx < 0 || yy < 0 || xx >= width as i32 || yy >= height as i32 {
                        continue;
                    }
                    let l = labels[yy as usize * width + xx as usize];
                    if l != NO_LABEL && m < neighbors.len() && !neighbors[..m].contains(&l) {
                        neighbors[m] = l;
                        m += 1;
                    }
                }
            }
            if m >= 2 {
                break;
            }
        }
        if m == 0 {
            textured += 1;
            continue; // kept verbatim in the base
        }
        if m == 1 && neighbors[0] != dominant {
            neighbors[1] = dominant;
            m = 2;
        }
        let p = px[i].map(f32::from);
        let mut best: Option<(f32, u8, u8, f32)> = None; // residual, a, b, c
        for s in 0..m {
            for t in s + 1..m {
                let (a, b) = (neighbors[s], neighbors[t]);
                let ca = colors[a as usize];
                let cb = colors[b as usize];
                let d: [f32; 3] = std::array::from_fn(|k| cb[k] - ca[k]);
                let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if dd < 100.0 {
                    continue;
                }
                let c = ((0..3).map(|k| (p[k] - ca[k]) * d[k]).sum::<f32>() / dd).clamp(0.0, 1.0);
                let r: f32 = (0..3).map(|k| (p[k] - ca[k] - c * d[k]).powi(2)).sum();
                if best.is_none_or(|(br, ..)| r < br) {
                    best = Some((r, a, b, c));
                }
            }
        }
        let Some((r, a, b, c)) = best else {
            textured += 1;
            continue;
        };
        if r > MAX_MIX_RESIDUAL_SQ {
            textured += 1;
            continue; // not a two-color mix: verbatim
        }
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
            coverage[i] = ((step * 255 + steps / 2) / steps) as u8;
            top_of[i] = top;
            top_counts[top as usize] += 1;
            band += 1;
        }
    }
    if (band as f32) > MAX_BAND_SHARE * n as f32
        || (textured as f32) > MAX_TEXTURED_SHARE * n as f32
    {
        return None;
    }
    // The coverage frame's color: the upper region on the band, the most
    // common band color everywhere else (cheapest for the palette coder).
    let fill = (0..colors.len()).max_by_key(|&k| top_counts[k]).unwrap();
    let mut layer = Vec::with_capacity(n * 3);
    for &t in &top_of {
        let k = if t == NO_LABEL { fill } else { t as usize };
        layer.extend_from_slice(&palette[k].0);
    }
    Some(VectorSplit {
        base,
        layer,
        coverage,
    })
}
