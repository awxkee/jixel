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

//! Cell layer: the integer-upscaled part of a mixed image (pixel art, a
//! zoomed screenshot, a 2x asset inside ordinary content) coded as one
//! frame at the cell grid's resolution, scaled back by the decoder's
//! nearest-neighbor kernel and alpha-blended over the rest.
//!
//! Segmentation is two bitmaps, "pixel equals its right neighbor" and
//! "pixel equals the one below", summed so that any N x N cell at any grid
//! offset is tested in constant time. A constant cell counts as upscaled
//! content when it differs from both the cell on its left and the one
//! above (edges between two flat colors only differ on one side), and the
//! textured cells must be dense: pixel art is textured everywhere, UI
//! corners are isolated. Flat cells next to textured ones join the layer
//! so a sprite's background rides along.

use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;

/// Smallest number of image pixels a layer must cover: below it the
/// layer's frame header, table of contents and histograms cost more than
/// the quarter of the samples it saves.
const MIN_COVERED_PIXELS: usize = 16 * 1024;
/// Textured cells among the 5x5 cells around one that keep it.
const DENSITY_MIN: u32 = 6;

/// The cell layer of an image: a factor, a grid offset and, inside the
/// covered cells' bounding box, which cells the layer holds.
#[derive(Debug, Clone)]
pub(crate) struct CellPlan {
    pub(crate) factor: usize,
    /// Bounding box of the covered cells in image pixels, on the cell grid.
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Covered cells inside the box, row-major, `cells_x` per row.
    pub(crate) covered: Vec<bool>,
    pub(crate) cells_x: usize,
    pub(crate) cells_y: usize,
}

impl CellPlan {
    pub(crate) fn covered_pixels(&self) -> usize {
        self.covered.iter().filter(|c| **c).count() * self.factor * self.factor
    }
}

/// Summed-area table of a bitmap, `(w + 1) x (h + 1)`.
struct Sat {
    w: usize,
    sum: Vec<u32>,
}

impl Sat {
    fn build(bits: &[bool], w: usize, h: usize) -> Self {
        let stride = w + 1;
        let mut sum = vec![0u32; stride * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                row += u32::from(bits[y * w + x]);
                sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row;
            }
        }
        Self { w, sum }
    }

    /// Set bits in `[x0, x1) x [y0, y1)`.
    fn count(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> u32 {
        let s = self.w + 1;
        self.sum[y1 * s + x1] + self.sum[y0 * s + x0]
            - self.sum[y0 * s + x1]
            - self.sum[y1 * s + x0]
    }
}

/// `eq_h[y * w + x]`: every plane equal at (x, y) and (x + 1, y); the last
/// column is false. `eq_v` likewise for (x, y + 1).
fn equality_bitmaps(image: &Image3Si) -> (Vec<bool>, Vec<bool>) {
    let (w, h) = (image.xsize(), image.ysize());
    let mut eq_h = vec![false; w * h];
    let mut eq_v = vec![false; w * h];
    for y in 0..h {
        let rows = [
            image.plane_row(0, y),
            image.plane_row(1, y),
            image.plane_row(2, y),
        ];
        let out = &mut eq_h[y * w..(y + 1) * w];
        for x in 0..w.saturating_sub(1) {
            out[x] = rows.iter().all(|r| r[x] == r[x + 1]);
        }
        if y + 1 < h {
            let below = [
                image.plane_row(0, y + 1),
                image.plane_row(1, y + 1),
                image.plane_row(2, y + 1),
            ];
            let out = &mut eq_v[y * w..(y + 1) * w];
            for x in 0..w {
                out[x] = (0..3).all(|c| rows[c][x] == below[c][x]);
            }
        }
    }
    (eq_h, eq_v)
}

struct Grid<'a> {
    image: &'a Image3Si,
    eq_h: Sat,
    eq_v: Sat,
}

impl Grid<'_> {
    /// Whether the cell of `n` pixels at (`x`, `y`), clipped to the image,
    /// is one color.
    fn constant(&self, x: usize, y: usize, n: usize) -> bool {
        let (w, h) = (self.image.xsize(), self.image.ysize());
        let x1 = (x + n).min(w);
        let y1 = (y + n).min(h);
        let (cw, ch) = (x1 - x, y1 - y);
        self.eq_h.count(x, y, x1 - 1, y1) == ((cw - 1) * ch) as u32
            && self.eq_v.count(x, y, x1, y1 - 1) == (cw * (ch - 1)) as u32
    }

    fn color(&self, x: usize, y: usize) -> [i32; 3] {
        [
            self.image.plane_row(0, y)[x],
            self.image.plane_row(1, y)[x],
            self.image.plane_row(2, y)[x],
        ]
    }
}

/// The best cell layer of an 8-bit RGB image, if any: the factor and grid
/// offset covering the most upscaled pixels, weighted by the samples the
/// factor saves.
pub(crate) fn plan(image: &Image3Si) -> Option<CellPlan> {
    let (w, h) = (image.xsize(), image.ysize());
    if w < 16 || h < 16 {
        return None;
    }
    let (eq_h, eq_v) = equality_bitmaps(image);
    let grid = Grid {
        image,
        eq_h: Sat::build(&eq_h, w, h),
        eq_v: Sat::build(&eq_v, w, h),
    };
    let mut best: Option<(f64, CellPlan)> = None;
    for n in [8usize, 4, 2] {
        for oy in 0..n {
            for ox in 0..n {
                let Some(candidate) = plan_at(&grid, n, ox, oy) else {
                    continue;
                };
                let saved = candidate.covered_pixels() as f64 * (1.0 - 1.0 / (n * n) as f64);
                if best.as_ref().is_none_or(|(s, _)| saved > *s) {
                    best = Some((saved, candidate));
                }
            }
        }
    }
    best.map(|(_, plan)| plan)
}

/// The layer of factor `n` on the grid whose cells start at (`ox`, `oy`):
/// the strip before the offset stays in the base, cells cut by the right
/// or bottom edge are judged on their visible part.
fn plan_at(grid: &Grid<'_>, n: usize, ox: usize, oy: usize) -> Option<CellPlan> {
    let (w, h) = (grid.image.xsize(), grid.image.ysize());
    let cells_x = (w - ox).div_ceil(n);
    let cells_y = (h - oy).div_ceil(n);
    let origin = |cx: usize, cy: usize| (ox + cx * n, oy + cy * n);
    let mut constant = vec![false; cells_x * cells_y];
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let (x, y) = origin(cx, cy);
            constant[cy * cells_x + cx] = grid.constant(x, y, n);
        }
    }
    let mut textured = vec![false; cells_x * cells_y];
    let mut any = false;
    for cy in 1..cells_y {
        for cx in 1..cells_x {
            let i = cy * cells_x + cx;
            if !constant[i] || !constant[i - 1] || !constant[i - cells_x] {
                continue;
            }
            let (x, y) = origin(cx, cy);
            let (lx, ly) = origin(cx - 1, cy);
            let (tx, ty) = origin(cx, cy - 1);
            let c = grid.color(x, y);
            if c != grid.color(lx, ly) && c != grid.color(tx, ty) {
                textured[i] = true;
                any = true;
            }
        }
    }
    if !any {
        return None;
    }
    let density = Sat::build(&textured, cells_x, cells_y);
    let mut dense = vec![false; cells_x * cells_y];
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let i = cy * cells_x + cx;
            if !textured[i] {
                continue;
            }
            let x0 = cx.saturating_sub(2);
            let y0 = cy.saturating_sub(2);
            let x1 = (cx + 3).min(cells_x);
            let y1 = (cy + 3).min(cells_y);
            dense[i] = density.count(x0, y0, x1, y1) >= DENSITY_MIN;
        }
    }
    // Flat constant cells next to dense ones join.
    let mut covered = vec![false; cells_x * cells_y];
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let i = cy * cells_x + cx;
            if !constant[i] {
                continue;
            }
            let x0 = cx.saturating_sub(1);
            let y0 = cy.saturating_sub(1);
            let x1 = (cx + 2).min(cells_x);
            let y1 = (cy + 2).min(cells_y);
            covered[i] = dense[i] || (y0..y1).any(|y| (x0..x1).any(|x| dense[y * cells_x + x]));
        }
    }
    // Runs of one color inside the art are constant cells that differ from
    // neither neighbor: every constant cell enclosed by the layer joins it
    // (those reaching the image border through uncovered cells stay), so
    // the layer has no holes and the base's fill runs unbroken.
    let mut outside = vec![false; cells_x * cells_y];
    let mut stack: Vec<usize> = (0..cells_x * cells_y)
        .filter(|&i| {
            let (cx, cy) = (i % cells_x, i / cells_x);
            !covered[i] && (cx == 0 || cy == 0 || cx + 1 == cells_x || cy + 1 == cells_y)
        })
        .collect();
    for &i in &stack {
        outside[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (cx, cy) = (i % cells_x, i / cells_x);
        let neighbors = [
            (cx > 0).then(|| i - 1),
            (cx + 1 < cells_x).then(|| i + 1),
            (cy > 0).then(|| i - cells_x),
            (cy + 1 < cells_y).then(|| i + cells_x),
        ];
        for j in neighbors.into_iter().flatten() {
            if !covered[j] && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        }
    }
    for i in 0..cells_x * cells_y {
        if !covered[i] && !outside[i] && constant[i] {
            covered[i] = true;
        }
    }
    let mut bx0 = usize::MAX;
    let mut by0 = usize::MAX;
    let mut bx1 = 0;
    let mut by1 = 0;
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            if covered[cy * cells_x + cx] {
                bx0 = bx0.min(cx);
                by0 = by0.min(cy);
                bx1 = bx1.max(cx + 1);
                by1 = by1.max(cy + 1);
            }
        }
    }
    if bx1 == 0 {
        return None;
    }
    // Inside each textured cluster's own box every constant cell is the
    // layer's: an art's flat run reaching its edge would otherwise stay in
    // the base as an island in the fill, which a lossy base rings around.
    // The UI between two clusters stays in the base.
    let mut seen = vec![false; cells_x * cells_y];
    let mut stack = Vec::new();
    for start in 0..cells_x * cells_y {
        if !dense[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut cx0, mut cy0, mut cx1, mut cy1) = (usize::MAX, usize::MAX, 0, 0);
        while let Some(i) = stack.pop() {
            let (cx, cy) = (i % cells_x, i / cells_x);
            cx0 = cx0.min(cx);
            cy0 = cy0.min(cy);
            cx1 = cx1.max(cx + 1);
            cy1 = cy1.max(cy + 1);
            for ny in cy.saturating_sub(1)..(cy + 2).min(cells_y) {
                for nx in cx.saturating_sub(1)..(cx + 2).min(cells_x) {
                    let j = ny * cells_x + nx;
                    if dense[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        for cy in cy0.saturating_sub(1)..(cy1 + 1).min(cells_y) {
            for cx in cx0.saturating_sub(1)..(cx1 + 1).min(cells_x) {
                let i = cy * cells_x + cx;
                covered[i] |= constant[i];
            }
        }
    }
    let (px0, py0) = origin(bx0, by0);
    let cells_bw = bx1 - bx0;
    let cells_bh = by1 - by0;
    let width = (cells_bw * n).min(w - px0);
    let height = (cells_bh * n).min(h - py0);
    let mut boxed = vec![false; cells_bw * cells_bh];
    for cy in 0..cells_bh {
        for cx in 0..cells_bw {
            boxed[cy * cells_bw + cx] = covered[(by0 + cy) * cells_x + bx0 + cx];
        }
    }
    let plan = CellPlan {
        factor: n,
        x0: px0,
        y0: py0,
        width,
        height,
        covered: boxed,
        cells_x: cells_bw,
        cells_y: cells_bh,
    };
    let covered_pixels = plan.covered_pixels();
    if covered_pixels < MIN_COVERED_PIXELS {
        return None;
    }
    Some(plan)
}

/// The base frame: the image with every covered pixel replaced by the
/// pixel to its left (or above on a row's first covered pixel), so the
/// covered area codes as runs, plus the opaque alpha the layer blends over.
pub(crate) fn base(image: &Image3Si, plan: &CellPlan) -> (Image3Si, AlphaPlane) {
    let (w, h) = (image.xsize(), image.ysize());
    let mut base = Image3Si::new(w, h);
    for c in 0..3 {
        for y in 0..h {
            let src = image.plane_row(c, y);
            base.plane_row_mut(c, y).copy_from_slice(src);
        }
    }
    let n = plan.factor;
    for cy in 0..plan.cells_y {
        for cx in 0..plan.cells_x {
            if !plan.covered[cy * plan.cells_x + cx] {
                continue;
            }
            let x0 = plan.x0 + cx * n;
            let y0 = plan.y0 + cy * n;
            for y in y0..(y0 + n).min(h) {
                for c in 0..3 {
                    let fill = if x0 > 0 {
                        base.plane_row(c, y)[x0 - 1]
                    } else if y > 0 {
                        base.plane_row(c, y - 1)[0]
                    } else {
                        0
                    };
                    let row = base.plane_row_mut(c, y);
                    row[x0..(x0 + n).min(w)].fill(fill);
                }
            }
        }
    }
    (base, AlphaPlane::from_u8(vec![255; w * h]))
}

/// The layer frame at the cell grid: one sample per cell of the box, the
/// covered cells' colors with the coverage as alpha; uncovered cells
/// repeat the cell to their left.
pub(crate) fn layer(image: &Image3Si, plan: &CellPlan) -> (Image3Si, AlphaPlane) {
    let n = plan.factor;
    let (cw, ch) = (plan.cells_x, plan.cells_y);
    let mut layer = Image3Si::new(cw, ch);
    let mut alpha = vec![0u8; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let covered = plan.covered[cy * cw + cx];
            if covered {
                alpha[cy * cw + cx] = 255;
            }
            for c in 0..3 {
                let v = if covered {
                    image.plane_row(c, plan.y0 + cy * n)[plan.x0 + cx * n]
                } else if cx > 0 {
                    layer.plane_row(c, cy)[cx - 1]
                } else if cy > 0 {
                    layer.plane_row(c, cy - 1)[0]
                } else {
                    0
                };
                layer.plane_row_mut(c, cy)[cx] = v;
            }
        }
    }
    (layer, AlphaPlane::from_u8(alpha))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A striped UI with specks and, at (`x`, `y`), a `cw` x `ch`-cell
    /// block of `n`-pixel cells of pseudo-random colors.
    fn composite(
        w: usize,
        h: usize,
        x: usize,
        y: usize,
        cw: usize,
        ch: usize,
        n: usize,
    ) -> Image3Si {
        let mut img = Image3Si::new(w, h);
        for py in 0..h {
            for px in 0..w {
                let inside = (x..x + cw * n).contains(&px) && (y..y + ch * n).contains(&py);
                let p = if inside {
                    let (cx, cy) = ((px - x) / n, (py - y) / n);
                    let v = (cx * 7919 + cy * 104_729) % 4093;
                    [
                        (v & 255) as i32,
                        ((v >> 4) & 255) as i32,
                        (((v * 13) >> 3) & 255) as i32,
                    ]
                } else if (px * 31 + py * 17) % 23 == 0 {
                    [0, 0, 0]
                } else {
                    let s = ((px / 13 + py / 9) % 3) as i32;
                    [40 + 60 * s, 200 - 50 * s, 90 + 20 * s]
                };
                for c in 0..3 {
                    img.plane_row_mut(c, py)[px] = p[c];
                }
            }
        }
        img
    }

    #[test]
    fn finds_the_block_at_an_odd_offset() {
        let img = composite(400, 300, 37, 51, 90, 60, 2);
        let plan = plan(&img).expect("a 2x block");
        assert_eq!(plan.factor, 2);
        assert_eq!((plan.x0 % 2, plan.y0 % 2), (1, 1));
        assert!(plan.x0 <= 37 && plan.y0 <= 51);
        assert!(plan.x0 + plan.width >= 37 + 180 && plan.y0 + plan.height >= 51 + 120);
        let covered = plan.covered_pixels();
        assert!((21_000..30_000).contains(&covered), "{covered}");
    }

    #[test]
    fn finds_a_4x_block_and_ignores_plain_ui() {
        let img = composite(400, 300, 64, 32, 40, 30, 4);
        let plan4 = plan(&img).expect("a 4x block");
        assert_eq!(plan4.factor, 4);
        assert_eq!((plan4.x0 % 4, plan4.y0 % 4), (0, 0));
        let ui = composite(400, 300, 0, 0, 0, 0, 2);
        assert!(plan(&ui).is_none());
    }

    #[test]
    fn base_and_layer_reassemble_the_image() {
        let img = composite(400, 300, 37, 51, 90, 60, 2);
        let plan = plan(&img).unwrap();
        let (base, _) = base(&img, &plan);
        let (layer, alpha) = layer(&img, &plan);
        let AlphaPlane::U8(alpha) = alpha else {
            unreachable!()
        };
        let n = plan.factor;
        for y in 0..300 {
            for x in 0..400 {
                let in_box = (plan.x0..plan.x0 + plan.width).contains(&x)
                    && (plan.y0..plan.y0 + plan.height).contains(&y);
                let cell = in_box.then(|| ((x - plan.x0) / n, (y - plan.y0) / n));
                for c in 0..3 {
                    let expect = img.plane_row(c, y)[x];
                    let got = match cell {
                        Some((cx, cy)) if alpha[cy * plan.cells_x + cx] == 255 => {
                            layer.plane_row(c, cy)[cx]
                        }
                        _ => base.plane_row(c, y)[x],
                    };
                    assert_eq!(got, expect, "({x}, {y}) plane {c}");
                }
            }
        }
    }
}
