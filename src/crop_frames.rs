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

//! Crop frames for mixed composites: a slide, poster or screen capture that
//! is mostly flat art with a photo in it fails the whole-image layer arms
//! (vector, ink) and goes to VarDCT whole. Here the flat rectangles are cut
//! out as cropped kReplace frames on the XYB lattice of a VarDCT base whose
//! own copy of each rectangle is filled with the rectangle's most common
//! color, so the base spends almost nothing there. Each crop carries the
//! vector split's composite at the distance's coverage precision when the
//! split takes the crop, else the exact pixels.

/// Side of the planner's tiles; also the alignment of every crop, so the
/// filled base never splits a VarDCT block across a crop edge.
pub(crate) const TILE: usize = 32;
/// A tile is flat when this share of its pixels has two equal 4-neighbors
/// (the vector gate's own test, per tile).
const FLAT_TILE_SHARE: f32 = 0.8;
/// Crops per image.
const MAX_CROPS: usize = 6;
/// Smallest crop, in tiles.
const MIN_CROP_TILES: usize = 6;
/// Share of tiles that must be non-flat: an image flat nearly everywhere is
/// the whole-image arms' to take.
const MIN_RICH_TILES: f32 = 0.03;
/// Bytes a crop frame costs beyond its sections: header, TOC, tree.
pub(crate) const FRAME_OVERHEAD: f64 = 150.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CropBox {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Flat tiles of the image (row-major, `tw` x `th` whole tiles).
fn flat_tiles(rgb: &[u8], width: usize, height: usize) -> (Vec<bool>, usize, usize) {
    let (tw, th) = (width / TILE, height / TILE);
    let px = rgb.as_chunks::<3>().0;
    let at = |x: usize, y: usize| px[y * width + x];
    let mut flat = vec![false; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let mut same = 0usize;
            for y in ty * TILE..(ty + 1) * TILE {
                for x in tx * TILE..(tx + 1) * TILE {
                    let p = at(x, y);
                    // Edge pixels compare against themselves (edge padding).
                    let n = usize::from(at(x.saturating_sub(1), y) == p)
                        + usize::from(at((x + 1).min(width - 1), y) == p)
                        + usize::from(at(x, y.saturating_sub(1)) == p)
                        + usize::from(at(x, (y + 1).min(height - 1)) == p);
                    same += usize::from(n >= 2);
                }
            }
            flat[ty * tw + tx] = same as f32 >= FLAT_TILE_SHARE * (TILE * TILE) as f32;
        }
    }
    (flat, tw, th)
}

/// Largest all-free rectangle of the tile grid: (area, x0, y0, x1, y1).
fn max_rect(free: &[bool], tw: usize, th: usize) -> (usize, usize, usize, usize, usize) {
    let mut heights = vec![0usize; tw];
    let mut best = (0, 0, 0, 0, 0);
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(tw + 1);
    for (y, row) in free[..tw * th].chunks_exact(tw).enumerate() {
        for (height, &is_free) in heights.iter_mut().zip(row) {
            *height = if is_free { *height + 1 } else { 0 };
        }
        stack.clear();
        for (x, &cur) in heights.iter().chain(std::iter::once(&0)).enumerate() {
            let mut start = x;
            while let Some(&(s, h)) = stack.last() {
                if h < cur {
                    break;
                }
                stack.pop();
                let area = h * (x - s);
                if area > best.0 {
                    best = (area, s, y + 1 - h, x, y + 1);
                }
                start = s;
            }
            stack.push((start, cur));
        }
    }
    best
}

/// Tile-aligned rectangles of flat art in a mixed image, largest first;
/// empty when the image is flat (or rich) nearly everywhere.
pub(crate) fn plan(rgb: &[u8], width: usize, height: usize) -> Vec<CropBox> {
    if width < 4 * TILE || height < 4 * TILE {
        return Vec::new();
    }
    let (mut free, tw, th) = flat_tiles(rgb, width, height);
    let rich = free.iter().filter(|&&f| !f).count();
    if (rich as f32) < MIN_RICH_TILES * free.len() as f32 || rich == free.len() {
        return Vec::new();
    }
    let mut boxes = Vec::new();
    while boxes.len() < MAX_CROPS {
        let (area, x0, y0, x1, y1) = max_rect(&free, tw, th);
        if area < MIN_CROP_TILES {
            break;
        }
        for y in y0..y1 {
            free[y * tw + x0..y * tw + x1].fill(false);
        }
        boxes.push(CropBox {
            x0: x0 * TILE,
            y0: y0 * TILE,
            width: (x1 - x0) * TILE,
            height: (y1 - y0) * TILE,
        });
    }
    boxes
}

/// The image with each box filled with its own most common color: the
/// crop's background, which the surrounding flat art usually continues.
/// (A Coons patch of the border pixels streaks a neighbouring photo's grain
/// across the box: base +4..+8 %, and the corpus wins shrink by a third.)
pub(crate) fn fill(rgb: &[u8], width: usize, boxes: &[CropBox]) -> Vec<u8> {
    let mut out = rgb.to_vec();
    for b in boxes {
        let mut counts = std::collections::HashMap::<[u8; 3], usize>::new();
        for y in b.y0..b.y0 + b.height {
            let row = &rgb[(y * width + b.x0) * 3..(y * width + b.x0 + b.width) * 3];
            for &p in row.as_chunks::<3>().0 {
                *counts.entry(p).or_default() += 1;
            }
        }
        let Some(mode) = counts
            .into_iter()
            .max_by_key(|&(c, n)| (n, c))
            .map(|(c, _)| c)
        else {
            continue;
        };
        for y in b.y0..b.y0 + b.height {
            let row = &mut out[(y * width + b.x0) * 3..(y * width + b.x0 + b.width) * 3];
            for p in row.as_chunks_mut::<3>().0 {
                *p = mode;
            }
        }
    }
    out
}

/// The box's pixels, interleaved RGB.
pub(crate) fn cut(rgb: &[u8], width: usize, b: &CropBox) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.width * b.height * 3);
    for y in b.y0..b.y0 + b.height {
        let start = (y * width + b.x0) * 3;
        out.extend_from_slice(&rgb[start..start + b.width * 3]);
    }
    out
}

/// What the crop frame carries: the vector split's composite (palette
/// regions and antialiasing on the distance's coverage levels) when the
/// split takes the crop, else the exact pixels.
pub(crate) fn content(crop: Vec<u8>, b: &CropBox, distance: f32) -> Vec<u8> {
    let steps = crate::vector_layers::coverage_steps(distance);
    match crate::vector_layers::split(&crop, b.width, b.height, steps) {
        Some(mut split) => {
            crate::simd::rgb8_blend::blend_in_place(
                split.base.as_chunks_mut::<3>().0,
                split.layer.as_chunks::<3>().0,
                &split.coverage,
            );
            split.base
        }
        None => crop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat canvas with a noisy "photo" in one corner.
    fn mixed(width: usize, height: usize) -> Vec<u8> {
        let mut rgb = vec![240u8; width * height * 3];
        let mut seed = 1u32;
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                for k in 0..3 {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    rgb[(y * width + x) * 3 + k] = (seed >> 24) as u8;
                }
            }
        }
        rgb
    }

    #[test]
    fn plan_boxes_cover_only_flat_tiles() {
        let (w, h) = (512, 384);
        let rgb = mixed(w, h);
        let boxes = plan(&rgb, w, h);
        assert!(!boxes.is_empty());
        for b in &boxes {
            assert_eq!(
                b.x0 % TILE + b.y0 % TILE + b.width % TILE + b.height % TILE,
                0
            );
            // No box reaches into the noisy quadrant.
            assert!(b.x0 >= w / 2 || b.y0 >= h / 2);
        }
        // Boxes never overlap.
        for (i, a) in boxes.iter().enumerate() {
            for b in &boxes[i + 1..] {
                let ox = a.x0 < b.x0 + b.width && b.x0 < a.x0 + a.width;
                let oy = a.y0 < b.y0 + b.height && b.y0 < a.y0 + a.height;
                assert!(!(ox && oy));
            }
        }
    }

    #[test]
    fn plan_skips_flat_and_rich_images() {
        let (w, h) = (256, 256);
        assert!(plan(&vec![200u8; w * h * 3], w, h).is_empty());
        let mut seed = 7u32;
        let noise: Vec<u8> = (0..w * h * 3)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            })
            .collect();
        assert!(plan(&noise, w, h).is_empty());
    }

    #[test]
    fn fill_paints_each_box_with_its_most_common_color() {
        let (w, h) = (96, 64);
        let mut rgb = vec![10u8; w * h * 3];
        let b = CropBox {
            x0: 32,
            y0: 0,
            width: 32,
            height: 32,
        };
        // Mostly (200, 100, 50) with one stray pixel inside the box.
        for y in 0..32 {
            for x in 32..64 {
                rgb[(y * w + x) * 3..][..3].copy_from_slice(&[200, 100, 50]);
            }
        }
        rgb[(5 * w + 40) * 3..][..3].copy_from_slice(&[1, 2, 3]);
        let filled = fill(&rgb, w, &[b]);
        for y in 0..h {
            for x in 0..w {
                let p = &filled[(y * w + x) * 3..][..3];
                let inside = (32..64).contains(&x) && y < 32;
                assert_eq!(
                    p,
                    if inside {
                        &[200, 100, 50][..]
                    } else {
                        &[10, 10, 10][..]
                    }
                );
            }
        }
    }
}
