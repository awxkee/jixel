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
//!
//! Detail crops are the second kind: rectangles of small text and line work
//! whose edges move luma almost alone, coded as lossy modular crop frames
//! (the quantized squeeze beats VarDCT there, and loses once the edges
//! carry chroma, where VarDCT has chroma-from-luma).

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

/// Detail crops stop here: coarser, the squeeze loses its edge over VarDCT
/// on text (one screen at d 2.7: -1.3 % bytes for -1.1 SS2).
pub(crate) const DETAIL_MAX_DISTANCE: f32 = 2.0;
/// Smallest detail crop, in tiles, and its shorter side.
const MIN_DETAIL_TILES: usize = 16;
const MIN_DETAIL_SIDE: usize = 3;
/// Detail crops per image.
const MAX_DETAIL_CROPS: usize = 4;
/// A detail tile's mean luma step between neighbors, in 8-bit levels.
const DETAIL_MIN_STEP: f32 = 3.0;
/// Its chroma steps stay under this share of its luma steps: the edges move
/// mostly luma, whatever the two tones are (0.2 misses a scanned page with
/// blue headings, 1.0 lets saturated text in and loses SS2 there).
const DETAIL_MAX_CHROMA: f32 = 0.5;
/// A detail crop is taken when its lossy modular frame costs under this
/// share of the Fast VarDCT encode of the same crop.
pub(crate) const DETAIL_MARGIN: f64 = 0.9;
/// This share of its luma steps comes from hard steps (at least
/// `DETAIL_HARD_STEP` levels): text and line work, not photo texture.
const DETAIL_HARD_SHARE: f32 = 0.5;
const DETAIL_HARD_STEP: i32 = 24;
/// A blank tile (paper between lines) steps less than this on average.
const BLANK_MAX_STEP: f32 = 1.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DetailTile {
    Other,
    Blank,
    Detail,
}

/// Classify every whole tile for [`plan_detail`].
fn detail_tiles(rgb: &[u8], width: usize, height: usize) -> (Vec<DetailTile>, usize, usize) {
    let (tw, th) = (width / TILE, height / TILE);
    let px = rgb.as_chunks::<3>().0;
    // Luma and the two YCoCg differences of a pixel.
    let ycc = |x: usize, y: usize| {
        let [r, g, b] = px[y * width + x].map(i32::from);
        [(r + 2 * g + b) >> 2, r - b, g - ((r + b) >> 1)]
    };
    let mut tiles = vec![DetailTile::Other; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let (mut luma, mut chroma, mut hard) = (0i64, 0i64, 0i64);
            for y in ty * TILE..(ty + 1) * TILE {
                for x in tx * TILE..(tx + 1) * TILE {
                    let p = ycc(x, y);
                    for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                        if nx >= (tx + 1) * TILE || ny >= (ty + 1) * TILE {
                            continue;
                        }
                        let q = ycc(nx, ny);
                        let step = (p[0] - q[0]).abs();
                        luma += i64::from(step);
                        chroma += i64::from((p[1] - q[1]).abs() + (p[2] - q[2]).abs());
                        if step >= DETAIL_HARD_STEP {
                            hard += i64::from(step);
                        }
                    }
                }
            }
            let pairs = (2 * TILE * (TILE - 1)) as f32;
            let (luma, chroma, hard) = (luma as f32, chroma as f32, hard as f32);
            tiles[ty * tw + tx] =
                if luma < BLANK_MAX_STEP * pairs && chroma < BLANK_MAX_STEP * pairs {
                    DetailTile::Blank
                } else if luma >= DETAIL_MIN_STEP * pairs
                    && chroma <= DETAIL_MAX_CHROMA * luma
                    && hard >= DETAIL_HARD_SHARE * luma
                {
                    DetailTile::Detail
                } else {
                    DetailTile::Other
                };
        }
    }
    (tiles, tw, th)
}

/// Tile-aligned rectangles of achromatic sharp detail (small text and line
/// work in any two tones whose edges move luma alone), largest first, clear
/// of the `taken` boxes: the lossy modular arm's content. A rectangle holds
/// detail tiles and the blank tiles next to one.
pub(crate) fn plan_detail(
    rgb: &[u8],
    width: usize,
    height: usize,
    taken: &[CropBox],
) -> Vec<CropBox> {
    if width < 4 * TILE || height < 4 * TILE {
        return Vec::new();
    }
    let (tiles, tw, th) = detail_tiles(rgb, width, height);
    let mut free: Vec<bool> = (0..tw * th)
        .map(|i| {
            let (tx, ty) = (i % tw, i / tw);
            match tiles[i] {
                DetailTile::Detail => true,
                DetailTile::Other => false,
                DetailTile::Blank => (ty.saturating_sub(1)..(ty + 2).min(th)).any(|y| {
                    (tx.saturating_sub(1)..(tx + 2).min(tw))
                        .any(|x| tiles[y * tw + x] == DetailTile::Detail)
                }),
            }
        })
        .collect();
    for b in taken {
        for y in b.y0 / TILE..(b.y0 + b.height) / TILE {
            free[y * tw + b.x0 / TILE..y * tw + (b.x0 + b.width) / TILE].fill(false);
        }
    }
    let mut boxes = Vec::new();
    while boxes.len() < MAX_DETAIL_CROPS {
        let (area, x0, y0, x1, y1) = max_rect(&free, tw, th);
        if area < MIN_DETAIL_TILES || (x1 - x0).min(y1 - y0) < MIN_DETAIL_SIDE {
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

    /// Noise everywhere but a 256x192 panel of hard two-tone strokes
    /// (`ink` on `paper`) at (64, 64).
    fn strokes(width: usize, height: usize, paper: [u8; 3], ink: [u8; 3]) -> Vec<u8> {
        let mut seed = 5u32;
        let mut rgb: Vec<u8> = (0..width * height * 3)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            })
            .collect();
        for y in 64..256 {
            for x in 64..320 {
                // Glyph-like marks: 2 px strokes every 5 px on lines 7 px apart.
                let on = y % 7 < 4 && (x + y / 7) % 5 < 2;
                rgb[(y * width + x) * 3..][..3].copy_from_slice(if on { &ink } else { &paper });
            }
        }
        rgb
    }

    #[test]
    fn plan_detail_takes_two_tone_strokes_in_either_polarity() {
        let (w, h) = (448, 320);
        for (paper, ink) in [
            ([250, 250, 250], [20, 20, 20]),
            ([10, 10, 10], [230, 230, 230]),
        ] {
            let boxes = plan_detail(&strokes(w, h, paper, ink), w, h, &[]);
            assert_eq!(
                boxes,
                vec![CropBox {
                    x0: 64,
                    y0: 64,
                    width: 256,
                    height: 192
                }]
            );
        }
    }

    #[test]
    fn plan_detail_leaves_saturated_strokes_noise_and_taken_boxes() {
        let (w, h) = (448, 320);
        // Red on white: the edges move chroma as much as luma.
        let red = strokes(w, h, [255, 255, 255], [255, 0, 0]);
        assert!(plan_detail(&red, w, h, &[]).is_empty());
        let gray = strokes(w, h, [250, 250, 250], [20, 20, 20]);
        let taken = CropBox {
            x0: 64,
            y0: 64,
            width: 256,
            height: 192,
        };
        assert!(plan_detail(&gray, w, h, &[taken]).is_empty());
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
