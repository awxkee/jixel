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

//! Region frames for lossless screenshots: a smooth-gradient or photo-like
//! rectangle inside flat UI and text is coded as its own cropped kReplace
//! frame with its own tree, palette and RCT, and the base frame shows the
//! rectangle filled with its border color, so the base keeps the palette and
//! tree of the UI around it.

use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use std::collections::HashMap;

/// Classification tile.
const TILE: usize = 32;
/// A color used fewer times than this image-wide is rare: gradients and
/// photos are made of rare colors, UI fills and antialiased text repeat
/// theirs.
const RARE_COUNT: u32 = 32;
/// Rare-pixel share that makes a tile rich.
const RICH_RARE_SHARE: f32 = 0.3;
/// Distinct colors per tile pixel that make a tile rich (antialiased text
/// sits at 0.1-0.2).
const RICH_DISTINCT_SHARE: f32 = 0.25;
/// Smallest rectangle worth its own frame.
const MIN_REGION_PIXELS: usize = 4096;
const MAX_REGIONS: usize = 8;
/// Rectangles may cover at most this share of the image: beyond it the
/// image is a photo or a painting with no UI base to keep.
const MAX_REGION_SHARE: f32 = 0.5;
/// How far a rectangle grows past its tiles to take in the rest of the
/// shape beside them (its lines holding pixels other than the fill).
const MAX_GROW: usize = 4 * TILE;
/// Growth crosses this many fill-only lines to reach the next part of the
/// shape: the gaps between the letters of gradient text (5 px in a bold
/// caps face at 30 px).
const MAX_GAP: usize = 6;

/// A rectangle coded as its own cropped frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// The color (three planes and alpha) the base frame shows in its place:
    /// the most common color on the surrounding ring.
    fill: [i32; 4],
}

type Rect = [usize; 4]; // x0, y0, x1, y1 (exclusive)

fn pack(p: [i32; 3], alpha: i32) -> u64 {
    ((p[0] + 1024) as u64) << 48
        | ((p[1] + 1024) as u64) << 32
        | ((p[2] + 1024) as u64) << 16
        | alpha as u64
}

fn unpack(key: u64) -> [i32; 4] {
    [
        ((key >> 48) & 0xFFFF) as i32 - 1024,
        ((key >> 32) & 0xFFFF) as i32 - 1024,
        ((key >> 16) & 0xFFFF) as i32 - 1024,
        (key & 0xFFFF) as i32,
    ]
}

fn pixel_keys(linear: &Image3Si, alpha: Option<&[u8]>) -> Vec<u64> {
    let (w, h) = (linear.xsize(), linear.ysize());
    let mut keys = Vec::with_capacity(w * h);
    for y in 0..h {
        let (p0, p1, p2) = (
            linear.plane_row(0, y),
            linear.plane_row(1, y),
            linear.plane_row(2, y),
        );
        let a = alpha.map(|a| &a[y * w..][..w]);
        keys.extend((0..w).map(|x| pack([p0[x], p1[x], p2[x]], a.map_or(0, |a| i32::from(a[x])))));
    }
    keys
}

/// Rectangles of the rich (gradient or photo-like) tiles, in frame order,
/// or `None` when the image has none worth a frame or is rich throughout.
pub(crate) fn plan(linear: &Image3Si, alpha: Option<&AlphaPlane>) -> Option<Vec<Region>> {
    let alpha = match alpha {
        None => None,
        Some(AlphaPlane::U8(a)) => Some(a.as_slice()),
        Some(_) => return None,
    };
    let (w, h) = (linear.xsize(), linear.ysize());
    if w < 2 * TILE || h < 2 * TILE {
        return None;
    }
    let keys = pixel_keys(linear, alpha);
    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(4096);
    for &k in &keys {
        *counts.entry(k).or_insert(0) += 1;
    }
    let rare: Vec<bool> = keys.iter().map(|k| counts[k] < RARE_COUNT).collect();
    let (tiles_x, tiles_y) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let mut rich = vec![false; tiles_x * tiles_y];
    let mut tile_keys = Vec::with_capacity(TILE * TILE);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            tile_keys.clear();
            let mut rare_count = 0usize;
            for y in ty * TILE..((ty + 1) * TILE).min(h) {
                let row = y * w + tx * TILE..y * w + ((tx + 1) * TILE).min(w);
                rare_count += rare[row.clone()].iter().filter(|&&pixel| pixel).count();
                tile_keys.extend_from_slice(&keys[row]);
            }
            let n = tile_keys.len() as f32;
            tile_keys.sort_unstable();
            tile_keys.dedup();
            rich[ty * tiles_x + tx] = rare_count as f32 >= RICH_RARE_SHARE * n
                || tile_keys.len() as f32 >= RICH_DISTINCT_SHARE * n;
        }
    }
    if !rich.contains(&true) {
        return None;
    }
    // Rectangles: the bounding boxes of the 8-connected rich tiles.
    let mut rects: Vec<Rect> = Vec::new();
    let mut seen = vec![false; rich.len()];
    let mut stack = Vec::new();
    for start in 0..rich.len() {
        if !rich[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut rect = [usize::MAX, usize::MAX, 0, 0];
        while let Some(t) = stack.pop() {
            let (tx, ty) = (t % tiles_x, t / tiles_x);
            rect = [
                rect[0].min(tx),
                rect[1].min(ty),
                rect[2].max(tx + 1),
                rect[3].max(ty + 1),
            ];
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (tx as i32 + dx, ty as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= tiles_x as i32 || ny >= tiles_y as i32 {
                        continue;
                    }
                    let n = ny as usize * tiles_x + nx as usize;
                    if rich[n] && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        rects.push([
            rect[0] * TILE,
            rect[1] * TILE,
            (rect[2] * TILE).min(w),
            (rect[3] * TILE).min(h),
        ]);
    }
    merge_overlapping(&mut rects);
    for rect in &mut rects {
        // The tile box also holds UI beside the shape: start from the rare
        // pixels and grow back over the shape's own flat parts.
        if let Some(core) = rare_bounds(*rect, &rare, w) {
            *rect = core;
        }
        let fill = ring_mode(&keys, *rect, w, h);
        grow(rect, &keys, fill, w, h);
    }
    merge_overlapping(&mut rects);
    let mut regions: Vec<Region> = rects
        .into_iter()
        .filter_map(|mut rect| {
            let fill = ring_mode(&keys, rect, w, h);
            shrink(&mut rect, &keys, fill, w);
            let (width, height) = (rect[2] - rect[0], rect[3] - rect[1]);
            (width * height >= MIN_REGION_PIXELS && width * height < w * h).then_some(Region {
                x0: rect[0],
                y0: rect[1],
                width,
                height,
                fill: unpack(fill),
            })
        })
        .collect();
    regions.sort_by_key(|r| std::cmp::Reverse(r.width * r.height));
    regions.truncate(MAX_REGIONS);
    regions.sort_by_key(|r| (r.y0, r.x0));
    let covered: usize = regions.iter().map(|r| r.width * r.height).sum();
    (!regions.is_empty() && covered as f32 <= MAX_REGION_SHARE * (w * h) as f32).then_some(regions)
}

/// Merge rectangles that overlap or come within `MAX_GAP` of each other.
fn merge_overlapping(rects: &mut Vec<Rect>) {
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let (a, b) = (rects[i], rects[j]);
                if a[0] < b[2] + MAX_GAP
                    && b[0] < a[2] + MAX_GAP
                    && a[1] < b[3] + MAX_GAP
                    && b[1] < a[3] + MAX_GAP
                {
                    rects[i] = [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                    ];
                    rects.swap_remove(j);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    rects.sort_unstable();
}

/// The bounding box of the rare pixels inside `rect`.
fn rare_bounds(rect: Rect, rare: &[bool], w: usize) -> Option<Rect> {
    let mut out = [usize::MAX, usize::MAX, 0, 0];
    for y in rect[1]..rect[3] {
        for x in rect[0]..rect[2] {
            if rare[y * w + x] {
                out = [
                    out[0].min(x),
                    out[1].min(y),
                    out[2].max(x + 1),
                    out[3].max(y + 1),
                ];
            }
        }
    }
    (out[2] > 0).then_some(out)
}

/// Extend each edge outward to every line within `MAX_GAP` that holds a
/// pixel other than the fill: the rest of the shape the rich tiles belong
/// to, across the gaps between its parts.
fn grow(rect: &mut Rect, keys: &[u64], fill: u64, w: usize, h: usize) {
    let row_shape =
        |y: usize, x0: usize, x1: usize| keys[y * w + x0..y * w + x1].iter().any(|&k| k != fill);
    let col_shape = |x: usize, y0: usize, y1: usize| (y0..y1).any(|y| keys[y * w + x] != fill);
    let mut grown = 0;
    while grown < MAX_GROW {
        let r = *rect;
        let up = (1..=MAX_GAP + 1)
            .filter(|&d| d <= r[1])
            .find(|&d| row_shape(r[1] - d, r[0], r[2]));
        let down = (0..=MAX_GAP)
            .filter(|&d| r[3] + d < h)
            .find(|&d| row_shape(r[3] + d, r[0], r[2]));
        let left = (1..=MAX_GAP + 1)
            .filter(|&d| d <= r[0])
            .find(|&d| col_shape(r[0] - d, r[1], r[3]));
        let right = (0..=MAX_GAP)
            .filter(|&d| r[2] + d < w)
            .find(|&d| col_shape(r[2] + d, r[1], r[3]));
        if up.is_none() && down.is_none() && left.is_none() && right.is_none() {
            break;
        }
        rect[1] -= up.unwrap_or(0);
        rect[3] += down.map_or(0, |d| d + 1);
        rect[0] -= left.unwrap_or(0);
        rect[2] += right.map_or(0, |d| d + 1);
        grown += 1;
    }
}

/// The most common color on the one-pixel ring around the rectangle (ties
/// to the smallest key); the rectangle's own top-left pixel when the ring
/// is empty.
fn ring_mode(keys: &[u64], rect: Rect, w: usize, h: usize) -> u64 {
    let mut counts: HashMap<u64, u32> = HashMap::new();
    let mut add = |x: usize, y: usize| *counts.entry(keys[y * w + x]).or_insert(0) += 1;
    let (x0, x1) = (rect[0].saturating_sub(1), (rect[2] + 1).min(w));
    for x in x0..x1 {
        if rect[1] > 0 {
            add(x, rect[1] - 1);
        }
        if rect[3] < h {
            add(x, rect[3]);
        }
    }
    for y in rect[1]..rect[3] {
        if rect[0] > 0 {
            add(rect[0] - 1, y);
        }
        if rect[2] < w {
            add(rect[2], y);
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(k, c)| (c, std::cmp::Reverse(k)))
        .map_or(keys[rect[1] * w + rect[0]], |(k, _)| k)
}

/// Pull each edge inward while its line is the fill color throughout.
fn shrink(rect: &mut Rect, keys: &[u64], fill: u64, w: usize) {
    let row_fill =
        |y: usize, x0: usize, x1: usize| keys[y * w + x0..y * w + x1].iter().all(|&k| k == fill);
    let col_fill = |x: usize, y0: usize, y1: usize| (y0..y1).all(|y| keys[y * w + x] == fill);
    while rect[3] - rect[1] > 1 && row_fill(rect[1], rect[0], rect[2]) {
        rect[1] += 1;
    }
    while rect[3] - rect[1] > 1 && row_fill(rect[3] - 1, rect[0], rect[2]) {
        rect[3] -= 1;
    }
    while rect[2] - rect[0] > 1 && col_fill(rect[0], rect[1], rect[3]) {
        rect[0] += 1;
    }
    while rect[2] - rect[0] > 1 && col_fill(rect[2] - 1, rect[1], rect[3]) {
        rect[2] -= 1;
    }
}

fn copy_rect(
    src: &Image3Si,
    src_alpha: Option<&AlphaPlane>,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> (Image3Si, Option<AlphaPlane>) {
    let mut out = Image3Si::new(width, height);
    for y in 0..height {
        for c in 0..3 {
            out.plane_row_mut(c, y)
                .copy_from_slice(&src.plane_row(c, y0 + y)[x0..x0 + width]);
        }
    }
    let alpha = src_alpha.map(|a| {
        let AlphaPlane::U8(a) = a else {
            unreachable!("region frames plan 8-bit alpha only")
        };
        let w = src.xsize();
        AlphaPlane::U8(
            (0..height)
                .flat_map(|y| a[(y0 + y) * w + x0..][..width].iter().copied())
                .collect(),
        )
    });
    (out, alpha)
}

/// The image with every region replaced by its fill color.
pub(crate) fn cut(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    regions: &[Region],
) -> (Image3Si, Option<AlphaPlane>) {
    let (w, h) = (linear.xsize(), linear.ysize());
    let (mut base, mut base_alpha) = copy_rect(linear, alpha, 0, 0, w, h);
    for r in regions {
        for y in r.y0..r.y0 + r.height {
            for c in 0..3 {
                base.plane_row_mut(c, y)[r.x0..r.x0 + r.width].fill(r.fill[c]);
            }
            if let Some(AlphaPlane::U8(a)) = &mut base_alpha {
                a[y * w + r.x0..][..r.width].fill(r.fill[3] as u8);
            }
        }
    }
    (base, base_alpha)
}

/// The region's own pixels.
pub(crate) fn crop(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    region: &Region,
) -> (Image3Si, Option<AlphaPlane>) {
    copy_rect(
        linear,
        alpha,
        region.x0,
        region.y0,
        region.width,
        region.height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat UI with text-like strokes and one smooth gradient rectangle.
    fn screenshot(w: usize, h: usize, rect: (usize, usize, usize, usize)) -> Image3Si {
        let mut img = Image3Si::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let band = !(40..150).contains(&y);
                let text = band && (y / 6) % 4 == 0 && (x % 5 < 2 || (x + y) % 11 == 0);
                let mut p = if text { [20, -10, 5] } else { [200, 30, -20] };
                if x >= rect.0 && x < rect.0 + rect.2 && y >= rect.1 && y < rect.1 + rect.3 {
                    let (dx, dy) = ((x - rect.0) as i32, (y - rect.1) as i32);
                    p = [dx + dy / 3, dx / 2 - dy, (dx * dy) % 97];
                }
                for c in 0..3 {
                    img.plane_row_mut(c, y)[x] = p[c];
                }
            }
        }
        img
    }

    #[test]
    fn finds_the_gradient_rectangle_and_fills_it_with_the_border_color() {
        let img = screenshot(300, 200, (70, 50, 120, 90));
        let regions = plan(&img, None).expect("one region");
        assert_eq!(regions.len(), 1);
        let r = regions[0];
        assert_eq!((r.x0, r.y0, r.width, r.height), (70, 50, 120, 90));
        assert_eq!(&r.fill[..3], &[200, 30, -20]);
        let (base, _) = cut(&img, None, &regions);
        assert_eq!(base.plane_row(0, 60)[80], 200);
        assert_eq!(base.plane_row(0, 10)[80], img.plane_row(0, 10)[80]);
        let (c, _) = crop(&img, None, &r);
        assert_eq!((c.xsize(), c.ysize()), (120, 90));
        assert_eq!(c.plane_row(2, 7)[11], img.plane_row(2, 57)[81]);
    }

    #[test]
    fn flat_images_and_rich_images_have_no_regions() {
        assert!(plan(&screenshot(300, 200, (0, 0, 0, 0)), None).is_none());
        assert!(plan(&screenshot(300, 200, (0, 0, 300, 200)), None).is_none());
        assert!(plan(&screenshot(300, 200, (10, 10, 250, 170)), None).is_none());
    }

    #[test]
    fn alpha_joins_the_color_and_follows_the_crop() {
        let img = screenshot(300, 200, (70, 50, 120, 90));
        let alpha = AlphaPlane::U8(vec![255; 300 * 200]);
        let regions = plan(&img, Some(&alpha)).expect("one region");
        let (_, base_alpha) = cut(&img, Some(&alpha), &regions);
        let (_, crop_alpha) = crop(&img, Some(&alpha), &regions[0]);
        assert_eq!(base_alpha.map(|a| a.len()), Some(300 * 200));
        assert_eq!(crop_alpha.map(|a| a.len()), Some(120 * 90));
    }

    /// The whole frame and the region split are both encoded and the
    /// smaller kept; the choice and the frames must not depend on the
    /// thread count.
    #[test]
    fn region_frames_are_deterministic_across_workers() {
        let img = screenshot(200, 136, (60, 44, 72, 64));
        assert_eq!(plan(&img, None).map(|r| r.len()), Some(1));
        let encode = |threads| {
            let mut w = crate::bit_writer::BitWriter::new();
            crate::lossless::encode_frame_lossless(
                &img,
                None,
                crate::lossless::LosslessSettings::rgb8(
                    crate::Speed::Slow,
                    crate::DecodingSpeed::Slow,
                ),
                true,
                threads,
                &mut w,
            );
            w.into_bytes()
        };
        let one = encode(1);
        assert!(!one.is_empty());
        assert_eq!(one, encode(4));
    }
}
