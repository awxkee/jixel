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
//! Additive glyph patches for the lossless path: glyphs over smooth ground
//! (labels on a ramp, text on a wallpaper gradient) stored as their integer
//! difference from the ground and blended with kAdd.
//!
//! A glyph over another background is another set of samples, so exact
//! (Replace) patches find nothing to repeat there. One template blended
//! over any background (kMul, kBlendAbove) leaves fractional samples, which
//! libjxl and jxl-rs dither on 8-bit output, each with its own pattern: only
//! integers come out the same everywhere. An integer difference added to an
//! integer base is exact, and what an occurrence differs from the tile it
//! shares stays in the base under it.
use super::components::label_components;
use super::ink::{
    GLYPH_MAX_INK_SHARE, GROUND_MIN_FRAME_SHARE, GROUND_MIN_INK_SHARE, InkScreen,
    absorb_smooth_ground, ink_mask_over, ink_share, screen_ink,
};
use super::{GLYPH_ATLAS_MAX_WIDTH, GLYPH_BG_TILE, GLYPH_MIN_REPEAT_PIXELS, GlyphShape, Lattice};
use crate::patches::color_key::color_key;
use crate::patches::{PATCH_REF_ID, PatchReference};
use crate::shared_sort::SharedSort;
use std::collections::HashMap;

/// Lattice units per sample level of the planes handed to the detector
/// (green, red, blue): its thresholds are in XYB lattice units, where an
/// 8-bit level is about two luma units and four of the red-green axis.
static DETECTOR_SCALE: [i32; 3] = [2, 4, 2];
/// A shape must stand out of its ground by this many levels somewhere.
const MIN_CONTRAST_LEVELS: i32 = 4;
/// Largest difference, per 8-bit level, between an occurrence and the tile
/// it shares; the base keeps it. Tiles are costly (about 5 bits per pixel)
/// and a faint remainder under a glyph is not: at 3 the chart of hexadecimal
/// labels over ramps came out no smaller than without patches, at 16 to 48
/// 12-17 % smaller.
const SHARE_TOLERANCE: i32 = 16;
/// Tiles of its size an occurrence is compared with, oldest first, before
/// it starts its own.
const MAX_LEADERS: usize = 4096;
/// A shape stands out when it leaves its ground by more than this many
/// share tolerances somewhere: below that a tile takes away no more than
/// the remainder it may leave.
const MIN_PEAK_TOLERANCES: i32 = 2;
/// Share of the shapes' box area that must stand out for the frame to be a
/// page of glyphs. The grain of a photo's smooth sky is shapes too (8 to 16
/// pixels each, up to 175 000 of them, all within a tolerance of each other
/// and so all repeats): 0.001 to 0.005 of their area stands out, 0.96 to 1
/// of the labels' on ramps.
const MIN_STANDING_SHARE: f64 = 0.5;
/// Share of the frame the repeats must cover (occurrences beyond the first
/// of each tile): a plan is raced against the plain frame with a whole
/// second encode. Label pages gave 0.006 to 0.05, a page of contour bands
/// with 4-pixel specks 0.0014.
const MIN_REPEAT_FRAME_SHARE: f64 = 1.0 / 256.0;

/// The plan: RGB planes of the atlas (signed differences) and of the base,
/// and the kAdd entries.
pub(crate) struct AdditiveGlyphs {
    pub(crate) atlas: [Vec<i32>; 3],
    pub(crate) atlas_width: usize,
    pub(crate) atlas_height: usize,
    pub(crate) base: [Vec<i32>; 3],
    pub(crate) references: Vec<PatchReference>,
}

/// The first two tests of [`plan_additive_glyphs`] (flat-color ink on a
/// tenth of the frame, most of that ink stepping smoothly) from bands of
/// its RGB rows: `fill_band(y0, rows, rgb)` writes `rows` rows from `y0`.
/// `None` where they fail: a photo leaves here before any plane of its
/// size is copied or built.
pub(crate) fn screen_additive_glyphs(
    width: usize,
    height: usize,
    bits: u32,
    mut fill_band: impl FnMut(usize, usize, [&mut [i32]; 3]),
) -> Option<InkScreen> {
    if width < GLYPH_BG_TILE || height < GLYPH_BG_TILE {
        return None;
    }
    let shift = bits.saturating_sub(8);
    let scale = DETECTOR_SCALE;
    let screen = screen_ink(width, height, |y0, rows, planes| {
        fill_band(y0, rows, planes.each_mut().map(Vec::as_mut_slice));
        // Red, green, blue to the detector's green, red, blue.
        planes.swap(0, 1);
        for (plane, scale) in planes.iter_mut().zip(scale) {
            for v in plane.iter_mut() {
                *v = (*v * scale) >> shift;
            }
        }
    });
    // The ground must take back `GROUND_MIN_INK_SHARE` of the ink; ink
    // that does not step smoothly is not going back.
    (screen.ink_share >= GROUND_MIN_FRAME_SHARE && screen.smooth_share >= GROUND_MIN_INK_SHARE)
        .then_some(screen)
}

/// Additive glyph patches of an integer RGB frame with samples in
/// `0..=max` that passed [`screen_additive_glyphs`]; `None` unless smooth
/// ground is what most of its ink was (the gate of the lossy search) and
/// enough glyphs stand on it.
pub(crate) fn plan_additive_glyphs(
    rgb: [Vec<i32>; 3],
    width: usize,
    height: usize,
    bits: u32,
    screen: InkScreen,
) -> Option<AdditiveGlyphs> {
    // The detector works on 8-bit levels: deeper samples drop their low
    // bits for it, and the differences are taken on the samples themselves.
    let shift = bits.saturating_sub(8);
    let max = (1i32 << bits) - 1;
    if width < GLYPH_BG_TILE || height < GLYPH_BG_TILE {
        return None;
    }
    let [r, g, b] = &rgb;
    let scale = DETECTOR_SCALE;
    let lattice = Lattice {
        planes: [
            g.iter().map(|&v| (v * scale[0]) >> shift).collect(),
            r.iter().map(|&v| (v * scale[1]) >> shift).collect(),
            b.iter().map(|&v| (v * scale[2]) >> shift).collect(),
        ],
        width,
        height,
        shift: 0,
    };
    let [ly, lx, lb] = &lattice.planes;
    let keys: Vec<u64> = ly
        .iter()
        .zip(lx)
        .zip(lb)
        .map(|((&y, &x), &b)| color_key(y, x, b))
        .collect();
    let mut ink = ink_mask_over(&keys, &screen.modes, width, height);
    let flat_ink = screen.ink_share;
    drop(screen);
    let ground = absorb_smooth_ground(&lattice, &mut ink)?;
    let left = ink_share(&ink);
    let taken = flat_ink - left;
    if taken < GROUND_MIN_FRAME_SHARE
        || taken < GROUND_MIN_INK_SHARE * flat_ink
        || left > GLYPH_MAX_INK_SHARE
    {
        return None;
    }
    let components = label_components(
        &lattice,
        &ink,
        &keys,
        Some(&ground),
        MIN_CONTRAST_LEVELS * scale[0],
    );
    drop((ink, keys, ground));
    let (shapes, label) = components.into_shapes(Vec::new());
    drop(lattice);
    if shapes.len() < 4 {
        return None;
    }

    // The background of a shape in sample levels (the detector's planes are
    // green, red, blue, scaled).
    let ground_of = |s: &GlyphShape| {
        [
            (s.bg[1] << shift) / scale[1],
            (s.bg[0] << shift) / scale[0],
            (s.bg[2] << shift) / scale[2],
        ]
    };
    // Difference of a shape's pixel from its ground; zero off the shape.
    let diff = |s: &GlyphShape, bg: &[i32; 3], c: usize, dx: usize, dy: usize| -> i32 {
        let i = (s.y0 + dy) * width + s.x0 + dx;
        if label[i] == s.label {
            rgb[c][i] - bg[c]
        } else {
            0
        }
    };
    let grounds: Vec<[i32; 3]> = shapes.iter().map(ground_of).collect();
    // Occurrences meet by box size: the faint edge of a glyph comes and
    // goes with the ground, so their masks differ. One joins the first tile
    // it is within the tolerance of everywhere, provided the base it leaves
    // stays a legal sample.
    let tolerance = SHARE_TOLERANCE << shift;
    let stands_out = |s: &GlyphShape, bg: &[i32; 3]| {
        (0..3).any(|c| {
            (0..s.h).any(|dy| {
                (0..s.w).any(|dx| diff(s, bg, c, dx, dy).abs() > MIN_PEAK_TOLERANCES * tolerance)
            })
        })
    };
    let (mut standing, mut area) = (0usize, 0usize);
    for (s, bg) in shapes.iter().zip(&grounds) {
        area += s.w * s.h;
        if stands_out(s, bg) {
            standing += s.w * s.h;
        }
    }
    if (standing as f64) < MIN_STANDING_SHARE * area as f64 {
        return None;
    }
    let mut buckets: HashMap<(usize, usize), Vec<Vec<usize>>> = HashMap::new();
    for (i, s) in shapes.iter().enumerate() {
        let groups = buckets.entry((s.w, s.h)).or_default();
        let fits = |leader: usize| {
            let l = &shapes[leader];
            (0..3).all(|c| {
                (0..s.h).all(|dy| {
                    (0..s.w).all(|dx| {
                        let shared = diff(l, &grounds[leader], c, dx, dy);
                        let own = diff(s, &grounds[i], c, dx, dy);
                        let at = (s.y0 + dy) * width + s.x0 + dx;
                        (own - shared).abs() <= tolerance
                            && (shared == 0 || (0..=max).contains(&(rgb[c][at] - shared)))
                    })
                })
            })
        };
        match groups.iter_mut().take(MAX_LEADERS).find(|g| fits(g[0])) {
            Some(group) => group.push(i),
            None => groups.push(vec![i]),
        }
    }
    // Only repeats: a tile of its own costs a glyph about what it costs in
    // place, and for large text far more (every unrepeated glyph in the
    // atlas made the chart 35 % larger).
    let mut groups: Vec<Vec<usize>> = buckets
        .into_values()
        .flatten()
        .filter(|g| {
            let s = &shapes[g[0]];
            g.len() >= 2 && (g.len() - 1) * s.w * s.h >= GLYPH_MIN_REPEAT_PIXELS
        })
        .collect();
    let repeated: usize = groups
        .iter()
        .map(|g| (g.len() - 1) * shapes[g[0]].w * shapes[g[0]].h)
        .sum();
    if (repeated as f64) < MIN_REPEAT_FRAME_SHARE * (width * height) as f64 {
        return None;
    }
    // Shelf-pack the leaders, tallest first; the first occurrence breaks
    // ties so the layout does not depend on the hash order.
    groups.shared_sort_by_key(|g| {
        let s = &shapes[g[0]];
        (std::cmp::Reverse(s.h), s.w, s.y0, s.x0)
    });
    let (area, widest) = groups.iter().fold((0, 0), |(area, widest), g| {
        let s = &shapes[g[0]];
        (area + s.w * s.h, widest.max(s.w))
    });
    let atlas_width = ((area as f64).sqrt() as usize * 3 / 2)
        .max(widest)
        .clamp(16, GLYPH_ATLAS_MAX_WIDTH);
    let mut slots = Vec::with_capacity(groups.len());
    let (mut cx, mut cy, mut shelf) = (0usize, 0usize, 0usize);
    for g in &groups {
        let s = &shapes[g[0]];
        if cx + s.w > atlas_width {
            cx = 0;
            cy += shelf;
            shelf = 0;
        }
        slots.push((cx, cy));
        cx += s.w;
        shelf = shelf.max(s.h);
    }
    let atlas_height = cy + shelf;
    let mut atlas: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0; atlas_width * atlas_height]);
    // Build all tiles from the original samples before reusing those planes
    // as the base. A placement can overlap the box of another tile's leader.
    for (g, &(ax, ay)) in groups.iter().zip(&slots) {
        let leader = &shapes[g[0]];
        let bg = &grounds[g[0]];
        for (c, plane) in atlas.iter_mut().enumerate() {
            let rows = &mut plane[ay * atlas_width..(ay + leader.h) * atlas_width];
            for (dy, row) in rows.chunks_exact_mut(atlas_width).enumerate() {
                for (dx, value) in row[ax..ax + leader.w].iter_mut().enumerate() {
                    *value = diff(leader, bg, c, dx, dy);
                }
            }
        }
    }
    drop((label, grounds));
    let mut base = rgb;
    let mut references = Vec::with_capacity(groups.len());
    for (g, (ax, ay)) in groups.into_iter().zip(slots) {
        let leader = &shapes[g[0]];
        let (w, h) = (leader.w, leader.h);
        let mut positions: Vec<(usize, usize)> =
            g.iter().map(|&i| (shapes[i].x0, shapes[i].y0)).collect();
        for (atlas_plane, base_plane) in atlas.iter().zip(&mut base) {
            let tile_rows = &atlas_plane[ay * atlas_width..(ay + h) * atlas_width];
            for &(x, y) in &positions {
                let base_rows = &mut base_plane[y * width..(y + h) * width];
                for (base_row, tile_row) in base_rows
                    .chunks_exact_mut(width)
                    .zip(tile_rows.chunks_exact(atlas_width))
                {
                    for (pixel, &shared) in base_row[x..x + w].iter_mut().zip(&tile_row[ax..ax + w])
                    {
                        *pixel -= shared;
                    }
                }
            }
        }
        positions.shared_sort_unstable_by_key(|&(x, y)| (y, x));
        references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: w,
            height: h,
            ref_frame: PATCH_REF_ID,
            alpha_only: false,
            add: true,
            mul: false,
            positions,
        });
    }
    Some(AdditiveGlyphs {
        atlas,
        atlas_width,
        atlas_height,
        base,
        references,
    })
}
