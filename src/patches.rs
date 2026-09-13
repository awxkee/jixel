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
use crate::coder_scratch::CoderScratch;
use crate::image::{Image3F, Image3Si};
use crate::thread_pool::ThreadPool;
use std::collections::HashMap;

pub(crate) struct LossyPatches {
    pub(crate) base: Image3F,
    /// Occurrence positions per group, first entry = the source tile. Packing
    /// into atlases is the caller's job so groups can be routed to more than
    /// one reference frame.
    pub(crate) groups: Vec<Vec<(usize, usize)>>,
}

#[derive(Clone, Copy)]
pub(crate) enum VarDctFrameKind<'a> {
    Regular,
    ReferenceOnly { width: usize, height: usize },
    Patched(&'a [PatchReference]),
}

/// Order-sensitive bulk hash of one tile.
fn hash_tile(linear: &Image3F, x0: usize, y0: usize) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in 0..3 {
        for y in y0..y0 + PATCH_TILE {
            let row = &linear.plane_row(c, y)[x0..x0 + PATCH_TILE];
            for pair in row.as_chunks::<2>().0 {
                let v = u64::from(pair[0].to_bits()) | (u64::from(pair[1].to_bits()) << 32);
                h = (h ^ v).wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 29;
            }
        }
    }
    h
}

/// Mean absolute deviation of a tile, summed over channels: a cheap stand-in
/// for what the tile costs to code as ordinary blocks.
fn tile_energy(img: &Image3F, x0: usize, y0: usize) -> f32 {
    let mut energy = 0.0;
    for c in 0..3 {
        let mut sum = 0.0;
        for y in y0..y0 + PATCH_TILE {
            sum += img.plane_row(c, y)[x0..x0 + PATCH_TILE].iter().sum::<f32>();
        }
        let mean = sum / (PATCH_TILE * PATCH_TILE) as f32;
        for y in y0..y0 + PATCH_TILE {
            energy += img.plane_row(c, y)[x0..x0 + PATCH_TILE]
                .iter()
                .map(|v| (v - mean).abs())
                .sum::<f32>();
        }
    }
    energy / (PATCH_TILE * PATCH_TILE) as f32
}

/// Minimum per-tile energy worth spending a patch on.
const MIN_PATCH_ENERGY: f32 = 0.017;

/// Minimum occurrences before a tile group becomes a patch. The study
/// preferred 5 to the historical 3 at every good configuration: rare groups
/// pay dictionary positions without amortizing their atlas tile.
const MIN_PATCH_OCCURRENCES: usize = 5;

pub(crate) fn find_lossy_patches(
    linear: &Image3F,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<LossyPatches> {
    let tile = PATCH_TILE;
    let width = linear.xsize();
    let height = linear.ysize();
    if width < tile || height < tile {
        return None;
    }

    let (tiles_x, tiles_y) = (width / tile, height / tile);
    // Tile rows are independent; the merge below is over tile counts, not
    // pixels, so it stays negligible.
    let rows = pool.steal_map(scratch, tiles_y, |ty, _scratch| {
        (0..tiles_x)
            .map(|tx| hash_tile(linear, tx * tile, ty * tile))
            .collect::<Vec<u64>>()
    });
    let mut buckets: HashMap<u64, Vec<(usize, usize)>> =
        HashMap::with_capacity(tiles_x.saturating_mul(tiles_y));
    for (ty, row) in rows.into_iter().enumerate() {
        for (tx, hash) in row.into_iter().enumerate() {
            buckets
                .entry(hash)
                .or_default()
                .push((tx * tile, ty * tile));
        }
    }

    let mut groups = Vec::new();
    for candidates in buckets.into_values() {
        if candidates.len() < MIN_PATCH_OCCURRENCES {
            continue;
        }
        let mut exact_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        for pos in candidates {
            let matching = exact_groups.iter().position(|group| {
                let first = group[0];
                (0..3).all(|c| {
                    (0..tile).all(|dy| {
                        linear.plane_row(c, first.1 + dy)[first.0..first.0 + tile]
                            == linear.plane_row(c, pos.1 + dy)[pos.0..pos.0 + tile]
                    })
                })
            });
            if let Some(i) = matching {
                exact_groups[i].push(pos);
            } else {
                exact_groups.push(vec![pos]);
            }
        }
        groups.extend(
            exact_groups
                .into_iter()
                .filter(|g| g.len() >= MIN_PATCH_OCCURRENCES),
        );
    }

    groups.retain(|g| tile_energy(linear, g[0].0, g[0].1) >= MIN_PATCH_ENERGY);
    sort_patch_groups(&mut groups);
    groups.truncate(256);
    if groups.is_empty() {
        return None;
    }

    let mut base = linear.clone();
    for positions in &groups {
        for &(x, y) in positions {
            for c in 0..3 {
                for dy in 0..tile {
                    base.plane_row_mut(c, y + dy)[x..x + tile].fill(0.0);
                }
            }
        }
    }
    Some(LossyPatches { base, groups })
}

/// Pack a set of groups into a fresh atlas image whose dictionary entries all
/// name `ref_frame`. With at most 256 groups the atlas never exceeds 256x256,
/// i.e. a single 256-pixel modular group.
pub(crate) fn pack_lossy_atlas(
    linear: &Image3F,
    groups: Vec<Vec<(usize, usize)>>,
    ref_frame: u32,
) -> (Image3F, Vec<PatchReference>) {
    let tile = PATCH_TILE;
    let atlas_cols = groups.len().min(16);
    let atlas_rows = groups.len().div_ceil(atlas_cols);
    let mut atlas = Image3F::new(atlas_cols * tile, atlas_rows * tile);
    let mut references = Vec::with_capacity(groups.len());
    for (i, positions) in groups.into_iter().enumerate() {
        let atlas_x = (i % atlas_cols) * tile;
        let atlas_y = (i / atlas_cols) * tile;
        let (sx, sy) = positions[0];
        for c in 0..3 {
            for dy in 0..tile {
                atlas.plane_row_mut(c, atlas_y + dy)[atlas_x..atlas_x + tile]
                    .copy_from_slice(&linear.plane_row(c, sy + dy)[sx..sx + tile]);
            }
        }
        references.push(PatchReference {
            width: PATCH_TILE,
            height: PATCH_TILE,
            atlas_x,
            atlas_y,
            ref_frame,
            positions,
        });
    }
    (atlas, references)
}

pub(crate) const PATCH_TILE: usize = 16;
pub(crate) const PATCH_REF_ID: u32 = 3;
/// Reference slot for the modular atlas; the VarDCT atlas keeps slot 3, so a
/// hybrid plan can emit both and route each dictionary entry to either.
pub(crate) const MODULAR_PATCH_REF_ID: u32 = 2;
pub(crate) const NUM_PATCH_CONTEXTS: usize = 10;

#[derive(Clone)]
pub(crate) struct PatchReference {
    pub(crate) atlas_x: usize,
    pub(crate) atlas_y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Which saved reference frame this entry copies from.
    pub(crate) ref_frame: u32,
    pub(crate) positions: Vec<(usize, usize)>,
}

pub(crate) struct LosslessPatches {
    pub(crate) atlas: Image3Si,
    pub(crate) base: Image3Si,
    pub(crate) references: Vec<PatchReference>,
}

#[derive(Clone, Copy)]
pub(crate) enum ModularFrameKind<'a> {
    Regular,
    ReferenceOnly { width: usize, height: usize },
    Patched(&'a [PatchReference]),
}

impl ModularFrameKind<'_> {
    pub(crate) fn is_regular(self) -> bool {
        matches!(self, Self::Regular)
    }

    /// Every frame kind may be coded by the learned-tree path: patched
    /// frames open their first section with the dictionary, and reference-
    /// only atlas frames only differ in their header (the flat coder spent
    /// 10 bits per pixel on a glyph atlas).
    pub(crate) fn allows_learned_tree(self) -> bool {
        true
    }
}

/// Order-sensitive bulk hash of one lossless tile. Packing pairs cuts the
/// number of mixing rounds in half compared with hashing individual `i32`s.
fn hash_lossless_tile(linear: &Image3Si, x0: usize, y0: usize) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in 0..3 {
        for y in y0..y0 + PATCH_TILE {
            let row = &linear.plane_row(c, y)[x0..x0 + PATCH_TILE];
            for pair in row.as_chunks::<2>().0 {
                let v = u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32);
                h = (h ^ v).wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 29;
            }
        }
    }
    h
}

pub(crate) fn find_lossless_patches(
    linear: &Image3Si,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<LosslessPatches> {
    let width = linear.xsize();
    let height = linear.ysize();
    if width < PATCH_TILE || height < PATCH_TILE {
        return None;
    }

    let tiles_x = width / PATCH_TILE;
    let tiles_y = height / PATCH_TILE;
    let rows = pool.steal_map(scratch, tiles_y, |ty, _scratch| {
        (0..tiles_x)
            .map(|tx| hash_lossless_tile(linear, tx * PATCH_TILE, ty * PATCH_TILE))
            .collect::<Vec<u64>>()
    });
    let mut buckets: HashMap<u64, Vec<(usize, usize)>> =
        HashMap::with_capacity(tiles_x.saturating_mul(tiles_y));
    for (ty, row) in rows.into_iter().enumerate() {
        for (tx, hash) in row.into_iter().enumerate() {
            buckets
                .entry(hash)
                .or_default()
                .push((tx * PATCH_TILE, ty * PATCH_TILE));
        }
    }

    let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
    for candidates in buckets.into_values() {
        if candidates.len() < 3 {
            continue;
        }
        // Hash collisions are split into exact-equality groups. The normal case
        // takes the first branch and performs one comparison per occurrence.
        let mut exact_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        for pos in candidates {
            let mut matching = None;
            for (i, group) in exact_groups.iter().enumerate() {
                let a = group[0];
                let equal = (0..3).all(|c| {
                    (0..PATCH_TILE).all(|dy| {
                        linear.plane_row(c, a.1 + dy)[a.0..a.0 + PATCH_TILE]
                            == linear.plane_row(c, pos.1 + dy)[pos.0..pos.0 + PATCH_TILE]
                    })
                });
                if equal {
                    matching = Some(i);
                    break;
                }
            }
            if let Some(i) = matching {
                exact_groups[i].push(pos);
            } else {
                exact_groups.push(vec![pos]);
            }
        }
        groups.extend(exact_groups.into_iter().filter(|g| g.len() >= 3));
    }
    sort_patch_groups(&mut groups);
    groups.truncate(256);
    if groups.is_empty() {
        return None;
    }

    let atlas_cols = groups.len().min(16);
    let atlas_rows = groups.len().div_ceil(atlas_cols);
    let atlas_width = atlas_cols * PATCH_TILE;
    let atlas_height = atlas_rows * PATCH_TILE;

    let mut atlas = Image3Si::new(atlas_width, atlas_height);
    let mut base = linear.clone();
    let mut references = Vec::with_capacity(groups.len());
    for (i, positions) in groups.into_iter().enumerate() {
        let atlas_x = (i % atlas_cols) * PATCH_TILE;
        let atlas_y = (i / atlas_cols) * PATCH_TILE;
        let src = positions[0];
        for c in 0..3 {
            for dy in 0..PATCH_TILE {
                atlas.plane_row_mut(c, atlas_y + dy)[atlas_x..atlas_x + PATCH_TILE]
                    .copy_from_slice(&linear.plane_row(c, src.1 + dy)[src.0..src.0 + PATCH_TILE]);
                for &(x, y) in &positions {
                    base.plane_row_mut(c, y + dy)[x..x + PATCH_TILE].fill(0);
                }
            }
        }
        references.push(PatchReference {
            width: PATCH_TILE,
            height: PATCH_TILE,
            atlas_x,
            atlas_y,
            ref_frame: PATCH_REF_ID,
            positions,
        });
    }
    Some(LosslessPatches {
        atlas,
        base,
        references,
    })
}

/// Largest glyph box side the glyph detector keeps.
const GLYPH_MAX_SIDE: usize = 64;
/// Background is decided per tile of this size (the tile's most frequent
/// color, or that of its eight neighbors, so boxes and headers whose
/// edges cross a tile still count as background on both sides).
const GLYPH_BG_TILE: usize = 32;
/// A shape must repeat: (occurrences - 1) * box pixels at or above this pays
/// for its atlas entry and per-occurrence position tokens.
const GLYPH_MIN_REPEAT_PIXELS: usize = 48;
/// Widest atlas the shapes are shelf-packed into.
const GLYPH_ATLAS_MAX_WIDTH: usize = 1024;

/// Three signed samples packed into 21-bit biased fields (planes may hold
/// negative values, e.g. after a color transform); values outside ±2^20
/// alias, which only weakens the background vote.
const GLYPH_KEY_BIAS: i64 = 1 << 20;
const GLYPH_KEY_MASK: u64 = (1 << 21) - 1;

#[inline]
fn color_key(c0: i32, c1: i32, c2: i32) -> u64 {
    let f = |v: i32| ((v as i64 + GLYPH_KEY_BIAS) as u64) & GLYPH_KEY_MASK;
    f(c0) << 42 | f(c1) << 21 | f(c2)
}

#[inline]
fn key_color(key: u64) -> [i32; 3] {
    let f = |shift: u32| ((key >> shift) & GLYPH_KEY_MASK) as i64 - GLYPH_KEY_BIAS;
    [f(42) as i32, f(21) as i32, f(0) as i32]
}

/// Ties go to the smallest key, as in the per-pixel vote.
fn most_frequent_color(keys: &mut [u64]) -> u64 {
    keys.sort_unstable();
    let (mut best, mut best_run) = (keys[0], 0);
    for run in keys.chunk_by(|a, b| a == b) {
        if run.len() > best_run {
            best = run[0];
            best_run = run.len();
        }
    }
    best
}

/// Glyph-level patches for text and UI: connected components of
/// non-background pixels (libjxl's `FindTextLikePatches` approach), keyed by
/// their exact bounding-box content and kept when the same shape occurs at
/// least twice. Unlike the tile detector, occurrences sit at arbitrary
/// positions, which is what repeated glyphs need. The base image gets the
/// local background under every patch (Replace blending ignores it), so the
/// remaining page is nearly blank.
pub(crate) fn find_lossless_glyph_patches(linear: &Image3Si) -> Option<LosslessPatches> {
    let width = linear.xsize();
    let height = linear.ysize();
    if width < GLYPH_BG_TILE || height < GLYPH_BG_TILE {
        return None;
    }
    // Per-tile background color: the most frequent color of the tile.
    let tiles_x = width.div_ceil(GLYPH_BG_TILE);
    let tiles_y = height.div_ceil(GLYPH_BG_TILE);
    let mut modes = vec![0u64; tiles_x * tiles_y];
    let mut keys: Vec<u64> = Vec::with_capacity(GLYPH_BG_TILE * GLYPH_BG_TILE);
    for (ty, mode_row) in modes.chunks_exact_mut(tiles_x).enumerate() {
        for (tx, mode) in mode_row.iter_mut().enumerate() {
            let x0 = tx * GLYPH_BG_TILE;
            let x1 = (x0 + GLYPH_BG_TILE).min(width);
            keys.clear();
            for y in ty * GLYPH_BG_TILE..((ty + 1) * GLYPH_BG_TILE).min(height) {
                let c0 = &linear.plane_row(0, y)[x0..x1];
                let c1 = &linear.plane_row(1, y)[x0..x1];
                let c2 = &linear.plane_row(2, y)[x0..x1];
                keys.extend(
                    c0.iter()
                        .zip(c1)
                        .zip(c2)
                        .map(|((&a, &b), &c)| color_key(a, b, c)),
                );
            }
            *mode = most_frequent_color(&mut keys);
        }
    }

    // Ink map, consumed by the flood fill. Neighboring background colors
    // are shared by the whole tile; collect them once, removing duplicates.
    let mut ink = vec![false; width * height];
    for (ty, band) in ink.chunks_mut(width * GLYPH_BG_TILE).enumerate() {
        let ny0 = ty.saturating_sub(1);
        let ny1 = (ty + 2).min(tiles_y);
        let neighbors = &modes[ny0 * tiles_x..ny1 * tiles_x];
        for tx in 0..tiles_x {
            let nx0 = tx.saturating_sub(1);
            let nx1 = (tx + 2).min(tiles_x);
            let mut backgrounds = [0; 9];
            let mut num_backgrounds = 0;
            for row in neighbors.chunks_exact(tiles_x) {
                for &key in &row[nx0..nx1] {
                    if !backgrounds[..num_backgrounds].contains(&key) {
                        backgrounds[num_backgrounds] = key;
                        num_backgrounds += 1;
                    }
                }
            }
            let backgrounds = &backgrounds[..num_backgrounds];
            let x0 = tx * GLYPH_BG_TILE;
            let x1 = (x0 + GLYPH_BG_TILE).min(width);
            for (dy, row) in band.chunks_exact_mut(width).enumerate() {
                let y = ty * GLYPH_BG_TILE + dy;
                let c0 = &linear.plane_row(0, y)[x0..x1];
                let c1 = &linear.plane_row(1, y)[x0..x1];
                let c2 = &linear.plane_row(2, y)[x0..x1];
                for (((pixel, &a), &b), &c) in row[x0..x1].iter_mut().zip(c0).zip(c1).zip(c2) {
                    *pixel = !backgrounds.contains(&color_key(a, b, c));
                }
            }
        }
    }
    // Connected components (8-neighbor) with bounding boxes.
    struct Shape {
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        hash: u64,
    }
    let mut shapes: Vec<Shape> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut next_pixel = 0;
    while let Some(offset) = ink[next_pixel..].iter().position(|&pixel| pixel) {
        let index = next_pixel + offset;
        next_pixel = index + 1;
        let (x, y) = (index % width, index / width);
        let (mut x0, mut x1, mut y0, mut y1) = (x, x, y, y);
        let mut count = 0usize;
        stack.push((x, y));
        ink[index] = false;
        while let Some((cx, cy)) = stack.pop() {
            count += 1;
            x0 = x0.min(cx);
            x1 = x1.max(cx);
            y0 = y0.min(cy);
            y1 = y1.max(cy);
            let nx0 = cx.saturating_sub(1);
            let nx1 = (cx + 2).min(width);
            let ny0 = cy.saturating_sub(1);
            let ny1 = (cy + 2).min(height);
            for (dy, row) in ink[ny0 * width..ny1 * width]
                .chunks_exact_mut(width)
                .enumerate()
            {
                for (dx, pixel) in row[nx0..nx1].iter_mut().enumerate() {
                    if *pixel {
                        *pixel = false;
                        stack.push((nx0 + dx, ny0 + dy));
                    }
                }
            }
        }
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        if count < 2 || w > GLYPH_MAX_SIDE || h > GLYPH_MAX_SIDE {
            continue;
        }
        let mut hash: u64 = 0x9e37_79b9_7f4a_7c15 ^ ((w as u64) << 32 | h as u64);
        for c in 0..3 {
            for row in linear.plane_data(c)[y0 * width..(y1 + 1) * width].chunks_exact(width) {
                for &v in &row[x0..=x1] {
                    hash = (hash ^ (v as u32 as u64)).wrapping_mul(0xff51_afd7_ed55_8ccd);
                    hash ^= hash >> 29;
                }
            }
        }
        shapes.push(Shape { x0, y0, w, h, hash });
    }
    if shapes.len() < 4 {
        return None;
    }
    // Group by exact content.
    let same_box = |a: &Shape, b: &Shape| -> bool {
        a.w == b.w
            && a.h == b.h
            && (0..3).all(|c| {
                let plane = linear.plane_data(c);
                let a_rows = plane[a.y0 * width..(a.y0 + a.h) * width].chunks_exact(width);
                let b_rows = plane[b.y0 * width..(b.y0 + b.h) * width].chunks_exact(width);
                a_rows
                    .zip(b_rows)
                    .all(|(ar, br)| ar[a.x0..a.x0 + a.w] == br[b.x0..b.x0 + b.w])
            })
    };
    struct ShapeGroup<'a> {
        shape: &'a Shape,
        positions: Vec<(usize, usize)>,
    }
    let mut buckets: HashMap<u64, Vec<ShapeGroup<'_>>> = HashMap::new();
    for s in &shapes {
        let groups = buckets.entry(s.hash).or_default();
        match groups.iter_mut().find(|g| same_box(g.shape, s)) {
            Some(g) => g.positions.push((s.x0, s.y0)),
            None => groups.push(ShapeGroup {
                shape: s,
                positions: vec![(s.x0, s.y0)],
            }),
        }
    }
    let mut groups: Vec<_> = buckets
        .into_values()
        .flatten()
        .filter(|g| {
            let s = g.shape;
            g.positions.len() >= 2 && (g.positions.len() - 1) * s.w * s.h >= GLYPH_MIN_REPEAT_PIXELS
        })
        .collect();
    if groups.len() < 2 {
        return None;
    }
    let (covered, total_area, max_width) =
        groups.iter().fold((0, 0, 0), |(covered, area, w), g| {
            let s = g.shape;
            (
                covered + g.positions.len() * s.w * s.h,
                area + s.w * s.h,
                w.max(s.w),
            )
        });
    if covered * 200 < width * height {
        return None;
    }
    // Shelf-pack tallest first, then by value within each height. Boxes may
    // overlap (kerned glyphs): every box equals the image, so Replace
    // blending reproduces the image in any order. The groups arrive in hash
    // order, which differs between processes, so ties are broken by the
    // group's first occurrence (unique per group) to keep the atlas layout
    // and the dictionary deterministic.
    groups.sort_by_key(|g| {
        let s = g.shape;
        (
            std::cmp::Reverse(s.h),
            std::cmp::Reverse((g.positions.len() - 1) * s.w * s.h),
            s.y0,
            s.x0,
        )
    });
    let atlas_width = ((total_area as f64).sqrt() as usize * 3 / 2)
        .max(max_width)
        .clamp(16, GLYPH_ATLAS_MAX_WIDTH);
    let mut slots: Vec<(usize, usize)> = Vec::with_capacity(groups.len());
    let (mut cx, mut cy, mut shelf_h) = (0usize, 0usize, 0usize);
    for g in &groups {
        let s = g.shape;
        if cx + s.w > atlas_width {
            cx = 0;
            cy += shelf_h;
            shelf_h = 0;
        }
        slots.push((cx, cy));
        cx += s.w;
        shelf_h = shelf_h.max(s.h);
    }
    let atlas_height = cy + shelf_h;
    // Padding between shelf-packed shapes takes the page's dominant
    // background, so glyph edges in the atlas meet the color they meet on
    // the page rather than zero.
    let mut atlas = Image3Si::try_new(atlas_width, atlas_height).ok()?;
    keys.clear();
    keys.extend_from_slice(&modes);
    let fill = key_color(most_frequent_color(&mut keys));
    for (c, value) in fill.into_iter().enumerate() {
        atlas.plane_mut(c).as_mut_slice().fill(value);
    }
    let mut base = linear.clone();
    let mut references = Vec::with_capacity(groups.len());
    for (g, (ax, ay)) in groups.into_iter().zip(slots) {
        let s = g.shape;
        for c in 0..3 {
            let src = &linear.plane_data(c)[s.y0 * width..(s.y0 + s.h) * width];
            let dst =
                &mut atlas.plane_mut(c).as_mut_slice()[ay * atlas_width..(ay + s.h) * atlas_width];
            for (dst_row, src_row) in dst
                .chunks_exact_mut(atlas_width)
                .zip(src.chunks_exact(width))
            {
                dst_row[ax..ax + s.w].copy_from_slice(&src_row[s.x0..s.x0 + s.w]);
            }
        }
        for &(px, py) in &g.positions {
            let fill = key_color(modes[(py / GLYPH_BG_TILE) * tiles_x + px / GLYPH_BG_TILE]);
            for (c, value) in fill.into_iter().enumerate() {
                let rows = &mut base.plane_mut(c).as_mut_slice()[py * width..(py + s.h) * width];
                for row in rows.chunks_exact_mut(width) {
                    row[px..px + s.w].fill(value);
                }
            }
        }
        // Row-major occurrence order keeps the dictionary's position deltas
        // small (dy mostly 0, dx the glyph pitch).
        let mut positions = g.positions;
        positions.sort_unstable_by_key(|&(x, y)| (y, x));
        references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: s.w,
            height: s.h,
            ref_frame: PATCH_REF_ID,
            positions,
        });
    }
    Some(LosslessPatches {
        atlas,
        base,
        references,
    })
}

// Both patch searches use this ordering; share the comparator's sort code.
fn sort_patch_groups(groups: &mut [Vec<(usize, usize)>]) {
    groups.sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_glyph_reconstruction(img: &Image3Si, plan: &LosslessPatches) {
        let mut restored = plan.base.clone();
        for r in &plan.references {
            for &(x, y) in &r.positions {
                for c in 0..3 {
                    for dy in 0..r.height {
                        restored.plane_row_mut(c, y + dy)[x..x + r.width].copy_from_slice(
                            &plan.atlas.plane_row(c, r.atlas_y + dy)
                                [r.atlas_x..r.atlas_x + r.width],
                        );
                    }
                }
            }
        }
        for c in 0..3 {
            assert_eq!(restored.plane_data(c), img.plane_data(c));
        }
    }

    #[test]
    fn glyph_patches_preserve_signed_pixels_at_image_and_tile_edges() {
        let mut img = Image3Si::new(131, 99);
        let background = [-17, 255, -(1 << 20)];
        for (c, value) in background.into_iter().enumerate() {
            img.plane_mut(c).as_mut_slice().fill(value);
        }
        let diagonals = [(0, 0), (29, 25), (124, 92)];
        let corners = [(126, 0), (61, 59), (0, 90)];
        for (c, value) in [-(1 << 20), (1 << 20) - 1, 0].into_iter().enumerate() {
            for &(x, y) in &diagonals {
                // Only diagonal neighbors connect these seven pixels.
                for d in 0..7 {
                    img.plane_row_mut(c, y + d)[x + d] = value;
                }
            }
            for &(x, y) in &corners {
                for dy in 0..9 {
                    img.plane_row_mut(c, y + dy)[x] = value;
                }
                img.plane_row_mut(c, y + 8)[x..x + 5].fill(value);
            }
        }
        let plan = find_lossless_glyph_patches(&img).expect("two repeated glyphs");
        assert_eq!(plan.references.len(), 2);
        for r in &plan.references {
            match (r.width, r.height) {
                (7, 7) => assert_eq!(r.positions, diagonals),
                (5, 9) => assert_eq!(r.positions, corners),
                dimensions => panic!("unexpected glyph box: {dimensions:?}"),
            }
        }
        for (c, value) in background.into_iter().enumerate() {
            assert!(plan.base.plane_data(c).iter().all(|&v| v == value));
        }
        assert_glyph_reconstruction(&img, &plan);
    }

    #[test]
    fn glyph_patches_preserve_neighboring_background_colors() {
        let mut img = Image3Si::new(99, 99);
        // Nine different tile modes exercise the complete neighborhood.
        for c in 0..3 {
            for y in 0..img.ysize() {
                for (x, pixel) in img.plane_row_mut(c, y).iter_mut().enumerate() {
                    *pixel = ((y / 32).min(2) * 3 + (x / 32).min(2) + c * 10) as i32;
                }
            }
        }
        for (w, h, positions) in [
            (5, 7, [(2, 3), (35, 4), (69, 5)]),
            (3, 9, [(3, 68), (36, 69), (70, 70)]),
        ] {
            for (x, y) in positions {
                for c in 0..3 {
                    for dy in 0..h {
                        img.plane_row_mut(c, y + dy)[x..x + w].fill(-100 - c as i32);
                    }
                }
            }
        }
        // This color belongs to a diagonal neighbor. It is background,
        // even though these repeated boxes are surrounded by another mode.
        for (x, y) in [(35, 35), (49, 49)] {
            for c in 0..3 {
                for dy in 0..7 {
                    img.plane_row_mut(c, y + dy)[x..x + 5].fill((c * 10) as i32);
                }
            }
        }
        let plan = find_lossless_glyph_patches(&img).expect("two repeated glyphs");
        assert_eq!(plan.references.len(), 2);
        assert!(plan.references.iter().all(|r| r.positions.len() == 3));
        assert_glyph_reconstruction(&img, &plan);
    }

    /// Six distinct 16x16 glyphs, each repeated exactly six times (comfortably
    /// past MIN_PATCH_OCCURRENCES), so every group has the same count and only
    /// the tie-break can order them.
    fn equal_length_groups() -> Image3F {
        let tile = PATCH_TILE;
        let (cols, rows) = (6usize, 6usize);
        let mut img = Image3F::new(cols * tile, rows * tile);
        for ty in 0..rows {
            for tx in 0..cols {
                for c in 0..3 {
                    for dy in 0..tile {
                        let row = img.plane_row_mut(c, ty * tile + dy);
                        for dx in 0..tile {
                            row[tx * tile + dx] = ((tx * 31 + dy * 3 + dx + c) % 17) as f32 / 17.0;
                        }
                    }
                }
            }
        }
        img
    }

    fn equal_length_lossless_groups() -> Image3Si {
        let tile = PATCH_TILE;
        let (cols, rows) = (6usize, 4usize);
        let mut img = Image3Si::new(cols * tile, rows * tile);
        for ty in 0..rows {
            for tx in 0..cols {
                for c in 0..3 {
                    for dy in 0..tile {
                        let row = img.plane_row_mut(c, ty * tile + dy);
                        for dx in 0..tile {
                            row[tx * tile + dx] = ((tx * 31 + dy * 3 + dx + c) % 17) as i32 - 8;
                        }
                    }
                }
            }
        }
        img
    }

    #[test]
    fn equal_length_groups_get_a_stable_atlas_order() {
        let pool = ThreadPool::new(4);
        let mut scratch = CoderScratch::default();
        let img = equal_length_groups();

        let signature =
            |p: &LossyPatches| -> Vec<(usize, usize)> { p.groups.iter().map(|g| g[0]).collect() };

        let plan = find_lossy_patches(&img, &pool, &mut scratch).expect("groups");
        let first = signature(&plan);
        assert_eq!(first.len(), 6);
        assert!(
            first.windows(2).all(|w| w[0] < w[1]),
            "groups must follow the first occurrence in raster order: {first:?}"
        );
        // Packing preserves that order in atlas slots and stamps the frame id.
        let (atlas, refs) = pack_lossy_atlas(&img, plan.groups, PATCH_REF_ID);
        assert_eq!((atlas.xsize(), atlas.ysize()), (6 * PATCH_TILE, PATCH_TILE));
        for (i, r) in refs.iter().enumerate() {
            assert_eq!((r.atlas_x, r.atlas_y), (i * PATCH_TILE, 0));
            assert_eq!(r.ref_frame, PATCH_REF_ID);
            assert_eq!(r.positions[0], first[i]);
        }
        // Within one process the map is seeded once, so repeats here only guard
        // the sort itself; the raster-order assertion above is what pins the
        // layout across processes.
        for _ in 0..4 {
            let again = find_lossy_patches(&img, &pool, &mut scratch).expect("groups");
            assert_eq!(first, signature(&again));
        }
    }

    #[test]
    fn lossless_patch_discovery_is_thread_deterministic() {
        let img = equal_length_lossless_groups();
        let run = |threads| {
            let pool = ThreadPool::new(threads);
            let mut scratch = CoderScratch::default();
            find_lossless_patches(&img, &pool, &mut scratch).expect("groups")
        };

        let single = run(1);
        let parallel = run(4);
        assert_eq!(single.references.len(), 6);
        assert_eq!(single.references.len(), parallel.references.len());
        for (a, b) in single.references.iter().zip(&parallel.references) {
            assert_eq!((a.atlas_x, a.atlas_y), (b.atlas_x, b.atlas_y));
            assert_eq!(a.positions, b.positions);
        }
        for c in 0..3 {
            assert_eq!(single.atlas.plane_data(c), parallel.atlas.plane_data(c));
            assert_eq!(single.base.plane_data(c), parallel.base.plane_data(c));
        }
    }
}
