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
use super::ink::Ground;
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
    /// The color around the component; `None` when it touches several
    /// that are not one ground.
    background: Option<u64>,
    /// It sits on smooth ground (see `absorb_smooth_ground`).
    soft: bool,
}

/// Distinct colors tallied around one component; past these it is no glyph.
const GROUND_MAX_COLORS: usize = 24;
/// Lattice range (Y, X, B) the colors around a glyph on a gradient may span.
static GROUND_RANGE: [i32; 3] = [24, 64, 24];
/// Share of the surrounding pixels that may be something else (the rules
/// of the cell a label fills), as a fraction.
const GROUND_MAX_FOREIGN_SHARE: (usize, usize) = (9, 20);

/// The background of a component that touches several colors: their most
/// frequent one, when most of the pixels around the component are within a
/// gradient's reach of that color.
fn gradient_background(colors: &[(u64, u32)]) -> Option<u64> {
    let &(dominant, _) = colors
        .iter()
        .max_by_key(|&&(key, count)| (count, std::cmp::Reverse(key)))?;
    let center = key_color(dominant);
    let (mut total, mut foreign) = (0usize, 0usize);
    for &(key, count) in colors {
        let color = key_color(key);
        let near = (0..3).all(|c| (color[c] - center[c]).abs() <= GROUND_RANGE[c]);
        total += count as usize;
        if !near {
            foreign += count as usize;
        }
    }
    let (most, of) = GROUND_MAX_FOREIGN_SHARE;
    (foreign * of <= total * most).then_some(dominant)
}

/// Labels the 8-connected ink component holding pixel `start`, appending
/// its pixels to `order`.
fn fill_component(
    ink: &[bool],
    keys: &[u64],
    ground: Option<&Ground>,
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
    // Colors around the component, with how many of its pixels' neighbors
    // show each.
    let mut colors: Vec<(u64, u32)> = Vec::new();
    let mut on_ground = false;
    let mut too_many = false;
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
                } else if !too_many {
                    on_ground |= ground.is_some_and(|ground| ground.holds(nx0 + dx, ny0 + dy));
                    if let Some((_, seen)) = colors.iter_mut().find(|(k, _)| *k == key) {
                        *seen += 1;
                    } else if colors.len() < GROUND_MAX_COLORS {
                        colors.push((key, 1));
                    } else {
                        too_many = true;
                    }
                }
            }
        }
    }
    // One color around the component is its background, as on any flat
    // page. Several are one only on smooth ground: elsewhere a shape between
    // colors is a cell border or a piece of a picture (the numbers of a
    // matrix plot touching their cells' edges cost more as glyphs than they
    // saved).
    let one_color = !too_many && colors.len() == 1;
    let candidate = if one_color {
        Some(colors[0].0)
    } else if too_many {
        None
    } else {
        gradient_background(&colors)
    };
    if let (false, Some(ground), Some(key)) = (on_ground, ground, candidate) {
        on_ground = ground.continues(x, y, key_color(key));
    }
    Fill {
        bounds: (x0, x1, y0, y1),
        count,
        background: candidate.filter(|_| one_color || on_ground),
        soft: on_ground,
    }
}

/// Adds one component, with its glyph shape when it fits an atlas entry.
#[allow(clippy::too_many_arguments)]
fn push_component(
    lattice: &Lattice,
    label: &[u32],
    list: &mut Vec<Component>,
    pixels: std::ops::Range<usize>,
    bounds: PixelBox,
    owner: u32,
    bg: [i32; 3],
    soft: bool,
    min_contrast: i32,
) {
    let (x0, x1, y0, y1) = bounds;
    let fits = x1 - x0 < GLYPH_MAX_SIDE && y1 - y0 < GLYPH_MAX_SIDE;
    let shape = fits
        .then(|| lattice.shape_of(label, bounds, owner, bg, min_contrast))
        .flatten()
        .map(|shape| GlyphShape { soft, ..shape });
    list.push(Component {
        shape,
        pixels,
        bounds,
        label: owner,
        bg,
    });
}

/// Largest box a component may have for its faint links to be cut.
const WEAK_LINK_MAX_BOX: usize = 1 << 16;
/// A pixel links glyphs when it differs from the ground by at least this
/// share of the component's strongest pixel, as a divisor.
const WEAK_LINK_SHARE: i32 = 6;

/// How far a pixel stands out of the ground, comparable across channels.
#[inline]
fn strength(lattice: &Lattice, at: usize, bg: [i32; 3]) -> i32 {
    let [y, x, b] = &lattice.planes;
    (y[at] - bg[0])
        .abs()
        .max((x[at] - bg[1]).abs() / 2)
        .max((b[at] - bg[2]).abs() / 2)
}

/// Cuts a component on smooth ground at its faint links. Small antialiased
/// labels (the two digits of a scale mark) touch through pixels a few
/// percent off the ground, and the pair repeats far less than its glyphs.
/// The strong pixels are relabeled by their own 8-connected parts and every
/// faint one joins a part it touches. `pixels` is reordered part by part;
/// returns (label, bounds, pixel count) per part, empty when the component
/// stays whole.
fn split_weak_links(
    lattice: &Lattice,
    bg: [i32; 3],
    (x0, x1, y0, y1): PixelBox,
    pixels: &mut [u32],
    label: &mut [u32],
    next_label: &mut u32,
) -> Vec<(u32, PixelBox, usize)> {
    let width = lattice.width;
    let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
    if w * h > WEAK_LINK_MAX_BOX || pixels.len() < 4 {
        return Vec::new();
    }
    let peak = pixels
        .iter()
        .map(|&at| strength(lattice, at as usize, bg))
        .max()
        .unwrap_or(0);
    let floor = peak / WEAK_LINK_SHARE;
    if floor == 0 {
        return Vec::new();
    }
    // Part of every box pixel, plus a one-pixel frame: 0 none, 1 strong
    // and unvisited, 2.. part index + 2.
    let (bw, bh) = (w + 2, h + 2);
    let mut part = vec![0u32; bw * bh];
    let local = |at: usize| (at / width - y0 + 1) * bw + at % width - x0 + 1;
    let mut any_weak = false;
    for &at in pixels.iter() {
        let at = at as usize;
        if strength(lattice, at, bg) >= floor {
            part[local(at)] = 1;
        } else {
            any_weak = true;
        }
    }
    if !any_weak {
        return Vec::new();
    }
    let neighbors = |i: usize| {
        [
            i - bw - 1,
            i - bw,
            i - bw + 1,
            i - 1,
            i + 1,
            i + bw - 1,
            i + bw,
            i + bw + 1,
        ]
    };
    let mut parts = 0u32;
    let mut stack = Vec::new();
    for &at in pixels.iter() {
        let start = local(at as usize);
        if part[start] != 1 {
            continue;
        }
        parts += 1;
        part[start] = parts + 1;
        stack.push(start);
        while let Some(i) = stack.pop() {
            for n in neighbors(i) {
                if part[n] == 1 {
                    part[n] = parts + 1;
                    stack.push(n);
                }
            }
        }
    }
    if parts < 2 {
        return Vec::new();
    }
    // Faint pixels join a part they touch, the lowest one on a tie so the
    // cut does not depend on the fill order.
    let mut joined = Vec::new();
    for &at in pixels.iter() {
        let i = local(at as usize);
        if part[i] != 0 {
            continue;
        }
        if let Some(p) = neighbors(i)
            .into_iter()
            .map(|n| part[n])
            .filter(|&p| p >= 2)
            .min()
        {
            joined.push((i, p));
        }
    }
    for (i, p) in joined {
        part[i] = p;
    }
    let first_label = *next_label + 1;
    // The last label is the faint pixels no part took.
    *next_label += parts + 1;
    let stray = first_label + parts;
    let mut found: Vec<(u32, PixelBox, usize)> = (0..parts)
        .map(|p| (first_label + p, (usize::MAX, 0, usize::MAX, 0), 0))
        .collect();
    for &at in pixels.iter() {
        let at = at as usize;
        let p = part[local(at)];
        if p < 2 {
            label[at] = stray;
            continue;
        }
        let (part_label, bounds, count) = &mut found[p as usize - 2];
        label[at] = *part_label;
        let (x, y) = (at % width, at / width);
        *bounds = (
            bounds.0.min(x),
            bounds.1.max(x),
            bounds.2.min(y),
            bounds.3.max(y),
        );
        *count += 1;
    }
    // A part of one pixel is no glyph.
    for &at in pixels.iter() {
        let l = label[at as usize];
        if l != stray && found[(l - first_label) as usize].2 < 2 {
            label[at as usize] = stray;
        }
    }
    // Group the pixel list by part, strays last.
    pixels.sort_unstable_by_key(|&at| {
        let l = label[at as usize];
        (l == stray, l, at)
    });
    found.retain(|&(_, _, count)| count >= 2);
    found
}

/// Connected ink components of two or more pixels that sit on one
/// background color or on one smooth ground, the latter cut at their faint
/// links.
pub(super) fn label_components(
    lattice: &Lattice,
    ink: &[bool],
    keys: &[u64],
    ground: Option<&Ground>,
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
            ink, keys, ground, size, start, next_label, &mut label, &mut order, &mut stack,
        );
        let Some(bg_key) = fill.background.filter(|_| fill.count >= 2) else {
            order.truncate(first);
            continue;
        };
        let bg = key_color(bg_key);
        let parts = if fill.soft {
            split_weak_links(
                lattice,
                bg,
                fill.bounds,
                &mut order[first..],
                &mut label,
                &mut next_label,
            )
        } else {
            Vec::new()
        };
        if parts.is_empty() {
            push_component(
                lattice,
                &label,
                &mut list,
                first..order.len(),
                fill.bounds,
                next_label,
                bg,
                fill.soft,
                min_contrast,
            );
            continue;
        }
        let mut at = first;
        for (part_label, bounds, count) in parts {
            push_component(
                lattice,
                &label,
                &mut list,
                at..at + count,
                bounds,
                part_label,
                bg,
                true,
                min_contrast,
            );
            at += count;
        }
        // Faint pixels no part took stay with the base.
        order.truncate(at);
    }
    Components {
        list,
        order,
        label,
        next_label,
    }
}
