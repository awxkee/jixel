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
const GLYPH_MAX_INK_SHARE: f64 = 0.6;
/// Distance between the tiles the screen samples.
const GLYPH_SCREEN_STRIDE: usize = 64;
/// Share of the sampled pixels that may be ink.
const GLYPH_SCREEN_MAX_INK: f64 = 0.6;

/// Cheap screen: on a sparse sample of 16-pixel tiles, how many pixels
/// differ from their tile's most frequent color. Photos are nearly all
/// "ink" and leave before the full-frame quantization.
pub(super) fn sample_is_mostly_ink(xyb: &Image3F, params: &GlyphParams) -> bool {
    let (width, height) = (xyb.xsize(), xyb.ysize());
    let mut colors = [[0i32; 3]; PATCH_TILE * PATCH_TILE];
    let mut sample_keys = [0u64; PATCH_TILE * PATCH_TILE];
    let (mut ink, mut total) = (0usize, 0usize);
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
            ink += sample_keys.iter().filter(|&&key| key != mode).count();
            total += sample_keys.len();
        }
    }
    ink as f64 > GLYPH_SCREEN_MAX_INK * total as f64
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
/// and its eight neighbors. `None` when most of the frame is ink: text and
/// UI sit on flat backgrounds, and without a page to subtract the component
/// scan is wasted time.
pub(super) fn ink_mask(keys: &[u64], width: usize, height: usize) -> Option<Vec<bool>> {
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
    let ink_pixels = ink.iter().filter(|&&p| p).count();
    (ink_pixels as f64 <= GLYPH_MAX_INK_SHARE * (width * height) as f64).then_some(ink)
}
