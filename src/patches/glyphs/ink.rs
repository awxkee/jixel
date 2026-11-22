/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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
use super::{GLYPH_BG_TILE, GlyphParams, Lattice};
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::patches::PATCH_TILE;
use crate::patches::color_key::{color_key, most_frequent_color, round_to_multiple};

/// Share of the frame that may be ink before it stops being a page.
pub(super) const GLYPH_MAX_INK_SHARE: f64 = 0.6;
/// Distance between the tiles the screen samples.
const GLYPH_SCREEN_STRIDE: usize = 64;
/// Share of the sampled pixels that may be ink.
const GLYPH_SCREEN_MAX_INK: f64 = 0.6;

/// Cheap screen: on a sparse sample of 16-pixel tiles, how many pixels
/// differ from their tile's most frequent color. Photos are nearly all
/// "ink" and leave before the full-frame quantization. A frame whose ink is
/// smooth ground instead (each such pixel a small step from the one before
/// it, as [`absorb_smooth_ground`] will find them) stays: a ramp steeper
/// than its tiles is a page for the labels on it.
pub(super) fn sample_is_mostly_ink(xyb: &Image3F, params: &GlyphParams) -> bool {
    let (width, height) = (xyb.xsize(), xyb.ysize());
    let mut colors = [[0i32; 3]; PATCH_TILE * PATCH_TILE];
    let mut sample_keys = [0u64; PATCH_TILE * PATCH_TILE];
    let (mut ink, mut smooth, mut total) = (0usize, 0usize, 0usize);
    for y0 in (0..height.saturating_sub(PATCH_TILE)).step_by(GLYPH_SCREEN_STRIDE) {
        for x0 in (0..width.saturating_sub(PATCH_TILE)).step_by(GLYPH_SCREEN_STRIDE) {
            crate::xyb::quantize_xyb_tile_colors(xyb, x0, y0, 1, &mut colors);
            let k = params.coarse;
            for (key, c) in sample_keys.iter_mut().zip(&colors) {
                // Same granularity as the full pass; tile colors are Y, X, B-Y.
                let r = round_to_multiple;
                *key = color_key(r(c[0], k), r(c[1], k), r(c[2] + c[0], k));
            }
            let mut sorted = sample_keys;
            let mode = most_frequent_color(&mut sorted);
            for (i, &key) in sample_keys.iter().enumerate() {
                if key == mode {
                    continue;
                }
                ink += 1;
                // The pixel before it in the row, or above at the row start.
                let before = if i % PATCH_TILE > 0 {
                    i - 1
                } else if i >= PATCH_TILE {
                    i - PATCH_TILE
                } else {
                    continue;
                };
                let (a, b) = (colors[i], colors[before]);
                let step = [a[0] - b[0], a[1] - b[1], a[2] + a[0] - b[2] - b[0]];
                smooth += usize::from((0..3).all(|c| step[c].abs() <= GROUND_STEP[c]));
            }
            total += sample_keys.len();
        }
    }
    ink as f64 > GLYPH_SCREEN_MAX_INK * total as f64
        && (smooth as f64) < GROUND_MIN_INK_SHARE * ink as f64
}

/// Packed coarse color of every pixel. The full frame uses the rounding
/// kernel on bounded chunks; matching later reads coarse samples only
/// within glyph boxes.
pub(super) fn color_keys(ctx: &EncodingContext, lattice: &Lattice) -> Vec<u64> {
    let [qy, qx, qb] = &lattice.planes;
    let shift = lattice.shift;
    let mut keys = Vec::with_capacity(lattice.width * lattice.height);
    if shift == 0 {
        keys.extend(
            qy.iter()
                .zip(qx)
                .zip(qb)
                .map(|((&y, &x), &b)| color_key(y, x, b)),
        );
        return keys;
    }
    const BLOCK: usize = 512;
    let mut rounded = [[0; BLOCK]; 3];
    for ((y, x), b) in qy.chunks(BLOCK).zip(qx.chunks(BLOCK)).zip(qb.chunks(BLOCK)) {
        let len = y.len();
        for (dest, src) in rounded.each_mut().into_iter().zip([y, x, b]) {
            let dest = &mut dest[..len];
            dest.copy_from_slice(src);
            // SAFETY: The context selects a supported rounding kernel.
            unsafe { (ctx.round_lattice_pow2)(dest, shift) };
        }
        let [y, x, b] = &rounded;
        keys.extend(
            y[..len]
                .iter()
                .zip(&x[..len])
                .zip(&b[..len])
                .map(|((&y, &x), &b)| color_key(y, x, b)),
        );
    }
    keys
}

/// The most frequent color of every background tile, row-major.
fn tile_backgrounds(keys: &[u64], width: usize, tiles_x: usize, tiles_y: usize) -> Vec<u64> {
    let bg_tile = GLYPH_BG_TILE;
    let mut modes = vec![0u64; tiles_x * tiles_y];
    let mut tile_keys: Vec<u64> = Vec::with_capacity(bg_tile * bg_tile);
    // A band is one row of tiles: `bg_tile` image rows (fewer at the bottom).
    let band_len = width * bg_tile;
    for (band, mode_row) in keys.chunks(band_len).zip(modes.chunks_exact_mut(tiles_x)) {
        for (tx, mode) in mode_row.iter_mut().enumerate() {
            let x0 = tx * bg_tile;
            let x1 = (x0 + bg_tile).min(width);
            tile_keys.clear();
            tile_keys.extend(band.chunks_exact(width).flat_map(|row| &row[x0..x1]));
            *mode = most_frequent_color(&mut tile_keys);
        }
    }
    modes
}

/// Marks the pixels that are none of the background colors of their tile
/// and its eight neighbors.
pub(super) fn ink_mask(keys: &[u64], width: usize, height: usize) -> Vec<bool> {
    let bg_tile = GLYPH_BG_TILE;
    let tiles_x = width.div_ceil(bg_tile);
    let tiles_y = height.div_ceil(bg_tile);
    let modes = tile_backgrounds(keys, width, tiles_x, tiles_y);
    let band_len = width * bg_tile;
    let mut ink = vec![false; width * height];
    for (ty, (ink_band, key_band)) in ink
        .chunks_mut(band_len)
        .zip(keys.chunks(band_len))
        .enumerate()
    {
        let neighbor_rows = &modes[ty.saturating_sub(1) * tiles_x..(ty + 2).min(tiles_y) * tiles_x];
        for tx in 0..tiles_x {
            let (nx0, nx1) = (tx.saturating_sub(1), (tx + 2).min(tiles_x));
            let mut backgrounds = [0u64; 9];
            let mut n = 0;
            for &key in neighbor_rows
                .chunks_exact(tiles_x)
                .flat_map(|row| &row[nx0..nx1])
            {
                if !backgrounds[..n].contains(&key) {
                    backgrounds[n] = key;
                    n += 1;
                }
            }
            let backgrounds = &backgrounds[..n];
            let x0 = tx * bg_tile;
            let x1 = (x0 + bg_tile).min(width);
            for (ink_row, key_row) in ink_band
                .chunks_exact_mut(width)
                .zip(key_band.chunks_exact(width))
            {
                for (pixel, key) in ink_row[x0..x1].iter_mut().zip(&key_row[x0..x1]) {
                    *pixel = !backgrounds.contains(key);
                }
            }
        }
    }
    ink
}

/// Share of the frame that is ink.
pub(super) fn ink_share(ink: &[bool]) -> f64 {
    ink.iter().filter(|&&p| p).count() as f64 / ink.len().max(1) as f64
}

/// Share of the frame, and of its flat-color ink, the smooth ground must
/// take to be used. Ramps and wallpapers under labels gave 0.74..1 of their
/// ink (0.18..0.5 of the frame), text screenshots with photos and shaded
/// boxes 0..0.56 and one 0.69.
pub(super) const GROUND_MIN_FRAME_SHARE: f64 = 0.1;
pub(super) const GROUND_MIN_INK_SHARE: f64 = 0.7;

/// Largest lattice step (Y, X, B) between neighbors of one smooth ground:
/// about two 8-bit levels, the step of a banded gradient.
static GROUND_STEP: [i32; 3] = [5, 12, 5];
/// A smooth ink region reaching the background joins it from this size;
/// smaller ones are the faint fringes of glyphs.
const GROUND_MIN_PIXELS: usize = 24;
/// ... and one that reaches no background from this size.
const GROUND_MIN_ISLAND_PIXELS: usize = 4096;

/// Lattice range (Y, X, B) the ground of one tile spans when it is a ramp.
static GROUND_TILE_SPREAD: [i32; 3] = [48, 128, 48];
/// Largest region taken for a pocket of nearby ground, and how far away
/// that ground may be.
const GROUND_MAX_POCKET_PIXELS: usize = 256;
const GROUND_POCKET_REACH: usize = 6;

/// Gradients are background too. The tile vote only knows flat colors, so
/// the bands of a ramp between them come out as ink and every label over
/// the ramp drowns in one frame-sized component. Ink regions whose pixels
/// step smoothly into one another (4-connected, each step within
/// [`GROUND_STEP`]) go back to the background when they are large and
/// continue a background pixel as smoothly, or are larger than any glyph.
/// Returns the pixels cleared.
pub(super) fn absorb_smooth_ground(lattice: &Lattice, ink: &mut [bool]) -> Option<Ground> {
    let (width, height) = (lattice.width, lattice.height);
    let [py, px, pb] = &lattice.planes;
    let smooth = |a: usize, b: usize| {
        (py[a] - py[b]).abs() <= GROUND_STEP[0]
            && (px[a] - px[b]).abs() <= GROUND_STEP[1]
            && (pb[a] - pb[b]).abs() <= GROUND_STEP[2]
    };
    let mut visited = vec![false; ink.len()];
    let mut ground: Option<Vec<bool>> = None;
    let mut region: Vec<u32> = Vec::new();
    // Regions left as ink, end to end, for the pocket pass.
    let mut left: Vec<u32> = Vec::new();
    let mut left_ends: Vec<usize> = Vec::new();
    let mut next = 0;
    while let Some(offset) = ink[next..]
        .iter()
        .zip(&visited[next..])
        .position(|(&pixel, &seen)| pixel && !seen)
    {
        let start = next + offset;
        next = start + 1;
        region.clear();
        region.push(start as u32);
        visited[start] = true;
        let mut touches_background = false;
        let mut head = 0;
        while head < region.len() {
            let at = region[head] as usize;
            head += 1;
            let (x, y) = (at % width, at / width);
            let mut visit = |to: usize| {
                if !smooth(at, to) {
                    return;
                }
                if !ink[to] {
                    touches_background = true;
                } else if !visited[to] {
                    visited[to] = true;
                    region.push(to as u32);
                }
            };
            if x > 0 {
                visit(at - 1);
            }
            if x + 1 < width {
                visit(at + 1);
            }
            if y > 0 {
                visit(at - width);
            }
            if y + 1 < height {
                visit(at + width);
            }
        }
        let is_ground = region.len() >= GROUND_MIN_ISLAND_PIXELS
            || (touches_background && region.len() >= GROUND_MIN_PIXELS);
        if is_ground {
            let ground = ground.get_or_insert_with(|| vec![false; ink.len()]);
            for &at in &region {
                ground[at as usize] = true;
            }
        } else if region.len() <= GROUND_MAX_POCKET_PIXELS {
            left.extend_from_slice(&region);
            left_ends.push(left.len());
        }
    }
    drop(visited);
    // Pockets: the ground a label closes off against the rules of its cell
    // (inside a ring, between a glyph and a tick) is too small to pass on
    // its own, but continues ground a few pixels away.
    if let Some(ground) = &mut ground {
        let mut pockets: Vec<u32> = Vec::new();
        let mut first = 0;
        for &end in &left_ends {
            let pocket = &left[first..end];
            first = end;
            let seed = pocket[0] as usize;
            let (mut x0, mut x1, mut y0, mut y1) = (width, 0, height, 0);
            for &at in pocket {
                let (x, y) = (at as usize % width, at as usize / width);
                (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            }
            let (x0, x1) = (
                x0.saturating_sub(GROUND_POCKET_REACH),
                (x1 + GROUND_POCKET_REACH).min(width - 1),
            );
            let (y0, y1) = (
                y0.saturating_sub(GROUND_POCKET_REACH),
                (y1 + GROUND_POCKET_REACH).min(height - 1),
            );
            let continues = (y0..=y1).any(|y| {
                let row = y * width;
                (x0..=x1).any(|x| ground[row + x] && smooth(seed, row + x))
            });
            if continues {
                pockets.extend_from_slice(pocket);
            }
        }
        for at in pockets {
            ground[at as usize] = true;
        }
    }
    // Cleared at the end: a region must not count ground absorbed before it
    // as the background it reaches, or the result would depend on scan order.
    let mask = ground?;
    for (pixel, &cleared) in ink.iter_mut().zip(&mask) {
        *pixel &= !cleared;
    }
    Some(Ground::new(lattice, mask))
}

/// The smooth ground of a frame: the pixels [`absorb_smooth_ground`] took
/// back from the ink, and the range of their colors per background tile.
pub(super) struct Ground {
    mask: Vec<bool>,
    width: usize,
    tiles_x: usize,
    tiles_y: usize,
    /// Lowest and highest lattice value (Y, X, B) of the ground in every
    /// tile; an empty range where the tile has none.
    ranges: Vec<[(i32, i32); 3]>,
}

impl Ground {
    fn new(lattice: &Lattice, mask: Vec<bool>) -> Self {
        let (width, height) = (lattice.width, lattice.height);
        let tiles_x = width.div_ceil(GLYPH_BG_TILE);
        let tiles_y = height.div_ceil(GLYPH_BG_TILE);
        let mut ranges = vec![[(i32::MAX, i32::MIN); 3]; tiles_x * tiles_y];
        for (y, row) in mask.chunks_exact(width).enumerate() {
            let tile_row = &mut ranges[(y / GLYPH_BG_TILE) * tiles_x..][..tiles_x];
            for (x, _) in row.iter().enumerate().filter(|&(_, &ground)| ground) {
                let range = &mut tile_row[x / GLYPH_BG_TILE];
                for (c, (low, high)) in range.iter_mut().enumerate() {
                    let value = lattice.planes[c][y * width + x];
                    *low = (*low).min(value);
                    *high = (*high).max(value);
                }
            }
        }
        Self {
            mask,
            width,
            tiles_x,
            tiles_y,
            ranges,
        }
    }

    /// Makes the ground ink again.
    pub(super) fn give_back(&self, ink: &mut [bool]) {
        for (pixel, &taken) in ink.iter_mut().zip(&self.mask) {
            *pixel |= taken;
        }
    }

    #[inline]
    pub(super) fn holds(&self, x: usize, y: usize) -> bool {
        self.mask[y * self.width + x]
    }

    /// Whether `color` continues the ground around (`x`, `y`): a step of a
    /// ramp that the tile vote kept as a flat color is ground all the same,
    /// and what stands on it changes with it like its neighbors do. The
    /// ground of a tile counts when it is one ramp (a narrow range of
    /// colors), not the patches of a chart that happen to span the color.
    pub(super) fn continues(&self, x: usize, y: usize, color: [i32; 3]) -> bool {
        let (tx, ty) = (x / GLYPH_BG_TILE, y / GLYPH_BG_TILE);
        (ty.saturating_sub(1)..(ty + 2).min(self.tiles_y)).any(|ty| {
            (tx.saturating_sub(1)..(tx + 2).min(self.tiles_x)).any(|tx| {
                let range = &self.ranges[ty * self.tiles_x + tx];
                (0..3).all(|c| {
                    let (low, high) = range[c];
                    low <= high
                        && high - low <= GROUND_TILE_SPREAD[c]
                        && color[c] >= low - GROUND_STEP[c]
                        && color[c] <= high + GROUND_STEP[c]
                })
            })
        })
    }
}
