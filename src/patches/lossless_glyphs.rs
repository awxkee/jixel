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
use super::color_key::{color_key, key_color, most_frequent_color};
use super::glyphs::{
    GLYPH_ATLAS_MAX_WIDTH, GLYPH_BG_TILE, GLYPH_MAX_SIDE, GLYPH_MIN_REPEAT_PIXELS,
};
use super::{ContentHashMap, LosslessPatches, PATCH_REF_ID, PatchReference};
use crate::image::Image3Si;
use crate::shared_sort::SharedSort;
use crate::thread_pool::ThreadPool;

/// A run of ink pixels `x0..=x1` on row `y` whose neighbors are still to
/// be visited.
type InkSpan = (usize, usize, usize);

/// Inclusive bounding box and pixel count of one ink component.
#[derive(Debug, PartialEq, Eq)]
struct InkComponent {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    count: usize,
}

/// Clears the 8-connected ink component holding pixel `index` and returns
/// its extent. The component is taken run by run: a photo is one component
/// of nearly every pixel, and visiting it pixel by pixel through a stack of
/// coordinates (nine neighbor tests each) was 6% of its encode.
fn take_ink_component(
    ink: &mut [bool],
    width: usize,
    height: usize,
    index: usize,
    spans: &mut Vec<InkSpan>,
) -> InkComponent {
    // The whole run around `x` on `row`, cleared.
    fn take_run(row: &mut [bool], x: usize) -> (usize, usize) {
        let left = x - row[..x].iter().rev().take_while(|&&pixel| pixel).count();
        let right = x + row[x..].iter().take_while(|&&pixel| pixel).count() - 1;
        row[left..=right].fill(false);
        (left, right)
    }
    let (x, y) = (index % width, index / width);
    let mut component = InkComponent {
        x0: x,
        x1: x,
        y0: y,
        y1: y,
        count: 0,
    };
    let mut take = |ink: &mut [bool], spans: &mut Vec<InkSpan>, x: usize, y: usize| {
        let (left, right) = take_run(&mut ink[y * width..(y + 1) * width], x);
        component.x0 = component.x0.min(left);
        component.x1 = component.x1.max(right);
        component.y0 = component.y0.min(y);
        component.y1 = component.y1.max(y);
        component.count += right - left + 1;
        spans.push((y, left, right));
        right
    };
    spans.clear();
    take(ink, spans, x, y);
    while let Some((y, left, right)) = spans.pop() {
        // Every pixel touching the run lies on the two adjacent rows, one
        // column wider on each side.
        let (from, to) = (left.saturating_sub(1), (right + 1).min(width - 1));
        for ny in [y.wrapping_sub(1), y + 1] {
            if ny >= height {
                continue;
            }
            let mut nx = from;
            while nx <= to {
                if ink[ny * width + nx] {
                    nx = take(ink, spans, nx, ny);
                }
                nx += 1;
            }
        }
    }
    component
}

/// Glyph-level patches for text and UI: connected components of
/// non-background pixels (libjxl's `FindTextLikePatches` approach), keyed by
/// their exact bounding-box content and kept when the same shape occurs at
/// least twice. Unlike the tile detector, occurrences sit at arbitrary
/// positions, which is what repeated glyphs need. The base image gets the
/// local background under every patch (Replace blending ignores it), so the
/// remaining page is nearly blank.
pub(crate) fn find_lossless_glyph_patches(
    linear: &Image3Si,
    pool: &ThreadPool,
) -> Option<LosslessPatches> {
    let width = linear.xsize();
    let height = linear.ysize();
    if width < GLYPH_BG_TILE || height < GLYPH_BG_TILE {
        return None;
    }
    // Per-tile background color: the most frequent color of the tile.
    let tiles_x = width.div_ceil(GLYPH_BG_TILE);
    let tiles_y = height.div_ceil(GLYPH_BG_TILE);
    // Tile rows and ink bands are independent: both run on the pool.
    let mut modes = vec![0u64; tiles_x * tiles_y];
    let mut mode_rows: Vec<&mut [u64]> = modes.chunks_exact_mut(tiles_x).collect();
    pool.for_each_mut(&mut mode_rows, |ty, mode_row| {
        let mut keys: Vec<u64> = Vec::with_capacity(GLYPH_BG_TILE * GLYPH_BG_TILE);
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
    });
    drop(mode_rows);

    // Ink map, consumed by the flood fill. Neighboring background colors
    // are shared by the whole tile; collect them once, removing duplicates.
    let mut ink = vec![false; width * height];
    let mut bands: Vec<&mut [bool]> = ink.chunks_mut(width * GLYPH_BG_TILE).collect();
    pool.for_each_mut(&mut bands, |ty, band| {
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
    });
    drop(bands);
    // Connected components (8-neighbor) with bounding boxes.
    struct Shape {
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        hash: u64,
    }
    let mut shapes: Vec<Shape> = Vec::new();
    let mut spans: Vec<InkSpan> = Vec::new();
    let mut next_pixel = 0;
    while let Some(offset) = ink[next_pixel..].iter().position(|&pixel| pixel) {
        let index = next_pixel + offset;
        next_pixel = index + 1;
        let InkComponent {
            x0,
            x1,
            y0,
            y1,
            count,
        } = take_ink_component(&mut ink, width, height, index, &mut spans);
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
    let mut buckets: ContentHashMap<Vec<ShapeGroup<'_>>> = ContentHashMap::default();
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
    groups.shared_sort_by_key(|g| {
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
    let fill = key_color(most_frequent_color(&mut modes.clone()));
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
        positions.shared_sort_unstable_by_key(|&(x, y)| (y, x));
        references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: s.w,
            height: s.h,
            ref_frame: PATCH_REF_ID,
            alpha_only: false,
            add: false,
            mul: false,
            positions,
        });
    }
    Some(LosslessPatches {
        atlas,
        base,
        references,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pixel-by-pixel flood fill `take_ink_component` replaced.
    fn take_ink_component_by_pixel(
        ink: &mut [bool],
        width: usize,
        height: usize,
        index: usize,
    ) -> InkComponent {
        let (x, y) = (index % width, index / width);
        let (mut x0, mut x1, mut y0, mut y1) = (x, x, y, y);
        let mut count = 0usize;
        let mut stack = vec![(x, y)];
        ink[index] = false;
        while let Some((cx, cy)) = stack.pop() {
            count += 1;
            x0 = x0.min(cx);
            x1 = x1.max(cx);
            y0 = y0.min(cy);
            y1 = y1.max(cy);
            for ny in cy.saturating_sub(1)..(cy + 2).min(height) {
                for nx in cx.saturating_sub(1)..(cx + 2).min(width) {
                    if ink[ny * width + nx] {
                        ink[ny * width + nx] = false;
                        stack.push((nx, ny));
                    }
                }
            }
        }
        InkComponent {
            x0,
            x1,
            y0,
            y1,
            count,
        }
    }

    #[test]
    fn span_fill_takes_the_same_components_as_the_pixel_fill() {
        let mut state = 0x2545_f491u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for (width, height) in [(1, 1), (1, 9), (9, 1), (7, 5), (33, 17), (64, 64), (97, 41)] {
            // Sparse specks, diagonal-heavy noise, dense blobs, all ink.
            for density in [5u32, 30, 50, 70, 95, 100] {
                let source: Vec<bool> = (0..width * height)
                    .map(|_| next() % 100 < density)
                    .collect();
                let (mut by_span, mut by_pixel) = (source.clone(), source.clone());
                let mut spans = Vec::new();
                let mut from = 0;
                while let Some(offset) = by_span[from..].iter().position(|&pixel| pixel) {
                    let index = from + offset;
                    from = index + 1;
                    assert!(by_pixel[index]);
                    let got = take_ink_component(&mut by_span, width, height, index, &mut spans);
                    let want = take_ink_component_by_pixel(&mut by_pixel, width, height, index);
                    assert_eq!(got, want);
                    assert_eq!(by_span, by_pixel);
                }
                assert!(by_pixel.iter().all(|&pixel| !pixel));
            }
        }
    }

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
        let plan =
            find_lossless_glyph_patches(&img, &ThreadPool::new(2)).expect("two repeated glyphs");
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
        let plan =
            find_lossless_glyph_patches(&img, &ThreadPool::new(2)).expect("two repeated glyphs");
        assert_eq!(plan.references.len(), 2);
        assert!(plan.references.iter().all(|r| r.positions.len() == 3));
        assert_glyph_reconstruction(&img, &plan);
    }
}
