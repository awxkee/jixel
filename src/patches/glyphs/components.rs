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
use super::grid::CellSplit;
use super::{GLYPH_MAX_SIDE, GlyphShape, Lattice, PixelBox};
use crate::patches::color_key::key_color;

/// One connected ink component over a single background color.
pub(super) struct Component {
    /// The whole component as a glyph, when it fits an atlas entry.
    pub(super) shape: Option<GlyphShape>,
    /// Where its pixels sit in [`Components::order`].
    pub(super) pixels: std::ops::Range<usize>,
    pub(super) bounds: PixelBox,
    pub(super) label: u32,
    pub(super) bg: [i32; 3],
}

/// The kept components of a frame.
pub(super) struct Components {
    pub(super) list: Vec<Component>,
    /// Pixel indices of the kept components, in fill order.
    pub(super) order: Vec<u32>,
    /// Owner of every pixel; zero is background.
    pub(super) label: Vec<u32>,
    /// Largest label handed out so far.
    pub(super) next_label: u32,
}

impl Components {
    pub(super) fn whole_shapes(&self) -> Vec<&GlyphShape> {
        self.list.iter().filter_map(|c| c.shape.as_ref()).collect()
    }

    /// The glyph shapes in component order, a split component giving its
    /// cells, and the final label plane.
    pub(super) fn into_shapes(self, splits: Vec<Option<CellSplit>>) -> (Vec<GlyphShape>, Vec<u32>) {
        let mut shapes = Vec::with_capacity(self.list.len());
        let mut splits = splits.into_iter();
        for component in self.list {
            match splits.next().flatten() {
                Some(split) => shapes.extend(split.shapes),
                None => shapes.extend(component.shape),
            }
        }
        (shapes, self.label)
    }
}

/// What one flood fill found.
struct Fill {
    bounds: PixelBox,
    count: usize,
    /// The one color around the component; `None` when it touches several.
    background: Option<u64>,
}

/// Labels the 8-connected ink component holding pixel `start`, appending
/// its pixels to `order`.
fn fill_component(
    ink: &[bool],
    keys: &[u64],
    (width, height): (usize, usize),
    start: usize,
    owner_label: u32,
    label: &mut [u32],
    order: &mut Vec<u32>,
    stack: &mut Vec<(usize, usize)>,
) -> Fill {
    let (x, y) = (start % width, start / width);
    let (mut x0, mut x1, mut y0, mut y1) = (x, x, y, y);
    let mut count = 0usize;
    let mut bg_key: Option<u64> = None;
    let mut one_bg = true;
    stack.push((x, y));
    label[start] = owner_label;
    while let Some((cx, cy)) = stack.pop() {
        count += 1;
        order.push((cy * width + cx) as u32);
        x0 = x0.min(cx);
        x1 = x1.max(cx);
        y0 = y0.min(cy);
        y1 = y1.max(cy);
        let (nx0, nx1) = (cx.saturating_sub(1), (cx + 2).min(width));
        let (ny0, ny1) = (cy.saturating_sub(1), (cy + 2).min(height));
        let window = ny0 * width..ny1 * width;
        let rows = ink[window.clone()]
            .chunks_exact(width)
            .zip(label[window.clone()].chunks_exact_mut(width))
            .zip(keys[window].chunks_exact(width));
        for (dy, ((ink_row, label_row), key_row)) in rows.enumerate() {
            let pixels = ink_row[nx0..nx1]
                .iter()
                .zip(&mut label_row[nx0..nx1])
                .zip(&key_row[nx0..nx1]);
            for (dx, ((&is_ink, owner), &key)) in pixels.enumerate() {
                if is_ink {
                    if *owner == 0 {
                        *owner = owner_label;
                        stack.push((nx0 + dx, ny0 + dy));
                    }
                } else {
                    match bg_key {
                        None => bg_key = Some(key),
                        Some(k) => one_bg &= k == key,
                    }
                }
            }
        }
    }
    Fill {
        bounds: (x0, x1, y0, y1),
        count,
        background: bg_key.filter(|_| one_bg),
    }
}

/// Connected ink components of two or more pixels that sit on one
/// background color.
pub(super) fn label_components(
    lattice: &Lattice,
    ink: &[bool],
    keys: &[u64],
    min_contrast: i32,
) -> Components {
    let size = (lattice.width, lattice.height);
    let mut label = vec![0u32; ink.len()];
    let mut list: Vec<Component> = Vec::new();
    let mut order: Vec<u32> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut next_label = 0u32;
    let mut next_pixel = 0;
    while let Some(offset) = ink[next_pixel..]
        .iter()
        .zip(&label[next_pixel..])
        .position(|(&pixel, &owner)| pixel && owner == 0)
    {
        let start = next_pixel + offset;
        next_pixel = start + 1;
        next_label += 1;
        let first = order.len();
        let fill = fill_component(
            ink, keys, size, start, next_label, &mut label, &mut order, &mut stack,
        );
        let Some(bg_key) = fill.background.filter(|_| fill.count >= 2) else {
            order.truncate(first);
            continue;
        };
        let bg = key_color(bg_key);
        let (x0, x1, y0, y1) = fill.bounds;
        let fits = x1 - x0 < GLYPH_MAX_SIDE && y1 - y0 < GLYPH_MAX_SIDE;
        let shape = fits
            .then(|| lattice.shape_of(&label, fill.bounds, next_label, bg, min_contrast))
            .flatten();
        list.push(Component {
            shape,
            pixels: first..order.len(),
            bounds: fill.bounds,
            label: next_label,
            bg,
        });
    }
    Components {
        list,
        order,
        label,
        next_label,
    }
}
