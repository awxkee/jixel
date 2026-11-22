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

//! Lossy coding of a bi-level image: rate-distortion pixel flipping in
//! front of the lossless coder.
//!
//! A scanned or rendered bi-level page spends most of its bits on outline
//! pixels its neighbors do not predict. Flipping such a pixel saves its own
//! cost and often the cost of the pixels that had it in their context. Here
//! the page's own context model picks the flips, and a flip is taken only
//! where it moves an outline without changing any shape's topology and
//! without shifting the low-pass (viewed) image past a distance-dependent
//! budget.

use crate::adaptive_quant::dirty_log2f;
use crate::shared_sort::SharedSort;
use crate::util::fast_exp;

/// The causal 16-pixel neighborhood the flipping model conditions on
/// (`(dy, dx)`; the JBIG2 generic-region template with its nominal adaptive
/// pixels). Bit `k` of a context is the pixel at offset `k`.
static TEMPLATE: [(isize, isize); 16] = [
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-1, -2),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (-1, 2),
    (0, -4),
    (0, -3),
    (0, -2),
    (0, -1),
    (-1, 3),
    (-2, -3),
    (-2, -2),
    (-2, 2),
];

/// Half side of the low-pass kernel a flip is judged through, and its
/// standard deviation: about what the eye averages at reading distance.
const BLUR_RADIUS: usize = 4;
const BLUR_SIGMA: f32 = 1.5;

/// Flips accepted in one pass keep this far apart, so that each one's gain,
/// computed for it alone, still holds: no pixel's context holds two of them.
const FLIP_GAP_Y: usize = 3;
const FLIP_GAP_X: usize = 5;

/// A pixel's coding cost above which it, and the pixels whose context holds
/// it, are worth scoring as flips.
const HOT_BITS: f32 = 0.5;

/// How much a page may be changed at a distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FlipBudget {
    /// Largest change of the low-pass image anywhere, in full-scale units.
    /// One flipped pixel changes it by 0.071 at its center.
    blur: f32,
    /// Bits a flip must save.
    min_gain: f32,
    passes: u32,
}

/// The budget at a few distances: `(distance, blur, min_gain, passes)`,
/// interpolated between rows (the first two geometrically). Fitted on text
/// scans, drawings, sheet music and manga pages so that the bytes saved
/// against the exact page grow steadily with the distance: about 1 % at
/// 0.25, 6 % at 0.5, 10 % at 1, 15 % at 2, 19 % at 3, 21 % at 4 and 23 % from
/// 8 on, where the topology-keeping flips run out. A single flip moves the
/// low-pass image by 0.071, so the first rows admit lone flips only and
/// differ in how many bits one must save.
static FLIP_SCHEDULE: [(f32, f32, f32, f32); 9] = [
    (0.25, 0.075, 8.0, 1.0),
    (0.5, 0.075, 4.0, 1.0),
    (1.0, 0.075, 1.5, 1.0),
    (1.5, 0.075, 1.0, 3.0),
    (2.0, 0.1, 0.7, 6.0),
    (3.0, 0.15, 0.4, 10.0),
    (4.0, 0.25, 0.25, 16.0),
    (6.0, 0.4, 0.1, 16.0),
    (8.0, 0.8, 0.05, 24.0),
];

impl FlipBudget {
    /// `None` below the distance at which a single pixel may flip.
    pub(crate) fn at(distance: f32) -> Option<Self> {
        let (first, last) = (FLIP_SCHEDULE[0], FLIP_SCHEDULE[FLIP_SCHEDULE.len() - 1]);
        if distance.is_nan() || distance < first.0 {
            return None;
        }
        let row = |(_, blur, min_gain, passes): (f32, f32, f32, f32)| Self {
            blur,
            min_gain,
            passes: passes as u32,
        };
        if distance >= last.0 {
            return Some(row(last));
        }
        let at = FLIP_SCHEDULE
            .iter()
            .rposition(|knot| knot.0 <= distance)
            .unwrap_or(0);
        let (lo, hi) = (FLIP_SCHEDULE[at], FLIP_SCHEDULE[at + 1]);
        let t = (distance - lo.0) / (hi.0 - lo.0);
        let geometric = |a: f32, b: f32| a * (b / a).powf(t);
        Some(Self {
            blur: geometric(lo.1, hi.1),
            min_gain: geometric(lo.2, hi.2),
            passes: (lo.3 + t * (hi.3 - lo.3)).round() as u32,
        })
    }
}

/// Share of horizontally adjacent pixel pairs that differ, from which a page
/// counts as a dither. Error-diffusion and ordered dithers measure 0.53-0.70;
/// text, drawings, manga pages and screentone scans 0.004-0.16.
const DITHER_MIN_TRANSITIONS: f64 = 0.3;

/// Whether the page is a dither. Its pixels carry tone, not outlines: the
/// topology-keeping flips saved under 2 % there for a large quality loss,
/// and one page grew.
pub(crate) fn is_dither(ink: &[bool], width: usize, height: usize) -> bool {
    if width < 2 || height == 0 {
        return false;
    }
    let transitions: usize = ink
        .chunks_exact(width)
        .map(|row| row.windows(2).filter(|pair| pair[0] != pair[1]).count())
        .sum();
    transitions as f64 >= DITHER_MIN_TRANSITIONS * ((width - 1) * height) as f64
}

/// Whether flipping a pixel keeps the shapes around it as they are: `code`
/// holds its eight neighbors (row-major, bit set = ink). True when the
/// neighbors' ink is one 8-connected piece and exactly one 4-connected piece
/// of their paper touches the pixel. Such a flip moves an outline; it never
/// fills a hole, opens one, breaks a stroke, joins two shapes, or creates or
/// removes a shape.
fn keeps_topology(code: u8) -> bool {
    static TABLE: std::sync::OnceLock<Box<[bool; 256]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Box::new([false; 256]);
        for (code, simple) in table.iter_mut().enumerate() {
            // The 3x3 cell with the center left out of both colors.
            let mut ink = [false; 9];
            let mut bit = 0;
            for (cell, slot) in ink.iter_mut().enumerate() {
                if cell != 4 {
                    *slot = code >> bit & 1 != 0;
                    bit += 1;
                }
            }
            let pieces = |member: &dyn Fn(usize) -> bool, diagonal: bool, seeds: &[usize]| {
                let mut seen = [false; 9];
                let mut count = 0;
                for &seed in seeds {
                    if !member(seed) || seen[seed] {
                        continue;
                    }
                    count += 1;
                    let mut stack = vec![seed];
                    seen[seed] = true;
                    while let Some(at) = stack.pop() {
                        let (x, y) = (at % 3, at / 3);
                        for next in 0..9 {
                            let (nx, ny) = (next % 3, next / 3);
                            let (dx, dy) = (x.abs_diff(nx), y.abs_diff(ny));
                            let adjacent = if diagonal {
                                dx.max(dy) == 1
                            } else {
                                dx + dy == 1
                            };
                            if adjacent && next != 4 && member(next) && !seen[next] {
                                seen[next] = true;
                                stack.push(next);
                            }
                        }
                    }
                }
                count
            };
            let ink_pieces = pieces(&|cell| ink[cell], true, &[0, 1, 2, 3, 5, 6, 7, 8]);
            // Paper pieces that touch the center through an edge.
            let paper_pieces = pieces(&|cell| cell != 4 && !ink[cell], false, &[1, 3, 5, 7]);
            *simple = ink_pieces == 1 && paper_pieces == 1;
        }
        table
    })[usize::from(code)]
}

/// Bit of a context that holds the pixel at template offset `(dy, dx)`.
/// The bits follow three shift registers, one per template row, so that a
/// row's contexts roll forward one pixel at a time: bits 0-5 are row -2
/// (`dx` 2 down to -3), bits 6-11 row -1 (`dx` 3 down to -2), bits 12-15 the
/// pixel's own row (`dx` -1 down to -4).
const fn context_bit(dy: isize, dx: isize) -> u32 {
    (match dy {
        -2 => 2 - dx,
        -1 => 6 + 3 - dx,
        _ => 12 - 1 - dx,
    }) as u32
}

/// For each template pixel, in template order: where the pixel that has it
/// in its context lies relative to it (`row` rows down, column `column - 3`),
/// and which context bit it is there.
static DEPENDENTS: [(usize, usize, u32); 16] = {
    let mut out = [(0, 0, 0); 16];
    let mut k = 0;
    while k < 16 {
        let (dy, dx) = TEMPLATE[k];
        out[k] = ((-dy) as usize, (3 - dx) as usize, context_bit(dy, dx));
        k += 1;
    }
    out
};

/// Columns a dependent may lie left and right of its template pixel.
const DEPENDENT_LEFT: usize = 3;
const DEPENDENT_RIGHT: usize = 4;
const DEPENDENT_SPAN: usize = DEPENDENT_LEFT + DEPENDENT_RIGHT + 1;

/// Contexts of every pixel of `ink` into `contexts` and their value counts
/// into `counts`; pixels outside the page read as paper.
fn learn_contexts(ink: &[bool], width: usize, contexts: &mut [u16], counts: &mut [[u32; 2]]) {
    counts.fill([0; 2]);
    let paper = vec![false; width];
    for (y, (row, row_contexts)) in ink
        .chunks_exact(width)
        .zip(contexts.chunks_exact_mut(width))
        .enumerate()
    {
        let above = if y >= 1 {
            &ink[(y - 1) * width..][..width]
        } else {
            &paper
        };
        let above2 = if y >= 2 {
            &ink[(y - 2) * width..][..width]
        } else {
            &paper
        };
        // Each register holds its row's window for the pixel before `x`; a
        // step shifts in the window's newest pixel.
        let bit = |row: &[bool], x: usize| row.get(x).is_some_and(|&ink| ink) as u16;
        let mut near = bit(above, 0) << 2 | bit(above, 1) << 1 | bit(above, 2);
        let mut far = bit(above2, 0) << 1 | bit(above2, 1);
        let mut left = 0u16;
        let entering = above2[2.min(width)..]
            .iter()
            .map(|&ink| ink as u16)
            .chain(std::iter::repeat(0))
            .zip(
                above[3.min(width)..]
                    .iter()
                    .map(|&ink| ink as u16)
                    .chain(std::iter::repeat(0)),
            );
        for ((&value, context), (far_in, near_in)) in
            row.iter().zip(row_contexts.iter_mut()).zip(entering)
        {
            far = (far << 1 | far_in) & 0x3f;
            near = (near << 1 | near_in) & 0x3f;
            *context = far | near << 6 | left << 12;
            counts[usize::from(*context)][usize::from(value)] += 1;
            left = (left << 1 | value as u16) & 0xf;
        }
    }
}

/// Flips pixels of `ink` (row-major, `true` = ink) where the flip saves bits
/// under the page's own context model and `budget` allows it; returns how
/// many were flipped.
///
/// The model is a static 16-pixel template: a flip's gain is the change of
/// its own cost and of the costs of the sixteen later pixels whose context
/// holds it. Flips are taken best first while (a) the flipped pixel keeps
/// the topology of the shapes around it, and (b) the low-pass image stays
/// within the budget of the original everywhere, so the changes cannot
/// gather into a visible blob. Each pass re-learns the model.
pub(crate) fn flip_pixels(
    ink: &mut [bool],
    width: usize,
    height: usize,
    budget: FlipBudget,
) -> usize {
    debug_assert_eq!(ink.len(), width * height);
    if width < 8 || height < 4 {
        return 0;
    }
    // The low-pass kernel.
    let side = 2 * BLUR_RADIUS + 1;
    let mut kernel = vec![0f32; side * side];
    for (i, k) in kernel.iter_mut().enumerate() {
        let (dx, dy) = (
            (i % side) as f32 - BLUR_RADIUS as f32,
            (i / side) as f32 - BLUR_RADIUS as f32,
        );
        *k = fast_exp(-(dx * dx + dy * dy) / (2.0 * BLUR_SIGMA * BLUR_SIGMA));
    }
    let total: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= total);
    // Low-pass(flipped) - low-pass(original), with a margin of the radius.
    let blur_width = width + 2 * BLUR_RADIUS;
    let mut blur_error = vec![0f32; blur_width * (height + 2 * BLUR_RADIUS)];
    // Scaling by one of +-1 is exact, so the fused and the plain kernels
    // add the same footprint.
    let add_scaled = crate::simd::scaled_add::selected_add_scaled_fn();

    let pixel = |ink: &[bool], x: isize, y: isize| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < width
            && (y as usize) < height
            && ink[y as usize * width + x as usize]
    };
    let mut contexts = vec![0u16; ink.len()];
    let mut own = vec![0f32; ink.len()];
    let mut marked = vec![false; ink.len()];
    let mut counts = vec![[0u32; 2]; 1 << 16];
    let mut cost = vec![[0f32; 2]; 1 << 16];
    let mut candidates: Vec<(f32, u32)> = Vec::new();
    // An unseen context: either value costs one bit.
    let unseen = -dirty_log2f(0.4 / 0.8);
    let mut flipped = 0usize;
    for _ in 0..budget.passes {
        // The model: context of every pixel, then cost of each value in
        // each context.
        learn_contexts(ink, width, &mut contexts, &mut counts);
        for (cost, &[zeros, ones]) in cost.iter_mut().zip(&counts) {
            if zeros + ones == 0 {
                *cost = [unseen; 2];
                continue;
            }
            let total = (zeros + ones) as f32 + 0.8;
            *cost = [
                -dirty_log2f((zeros as f32 + 0.4) / total),
                -dirty_log2f((ones as f32 + 0.4) / total),
            ];
        }
        for (own, (&context, &value)) in own.iter_mut().zip(contexts.iter().zip(ink.iter())) {
            *own = cost[usize::from(context)][usize::from(value)];
        }

        // Mark the flips that can gain: a costly pixel, or a pixel in the
        // context of one.
        marked.fill(false);
        for (y, own_row) in own.chunks_exact(width).enumerate() {
            for (x, &bits) in own_row.iter().enumerate() {
                if bits <= HOT_BITS {
                    continue;
                }
                marked[y * width + x] = true;
                for &(dy, dx) in &TEMPLATE {
                    let (px, py) = (x as isize + dx, y as isize + dy);
                    if px >= 0 && py >= 0 && (px as usize) < width {
                        marked[py as usize * width + px as usize] = true;
                    }
                }
            }
        }

        // Score them. Away from the page edges the sixteen dependents of a
        // pixel sit in three fixed windows, one per row.
        candidates.clear();
        let inner_rows = height.saturating_sub(2);
        let inner_columns = DEPENDENT_LEFT..width.saturating_sub(DEPENDENT_RIGHT);
        for (y, marked_row) in marked.chunks_exact(width).enumerate() {
            for (x, _) in marked_row.iter().enumerate().filter(|&(_, &marked)| marked) {
                let at = y * width + x;
                let value = usize::from(ink[at]);
                let mut gain = own[at] - cost[usize::from(contexts[at])][1 - value];
                if y < inner_rows && inner_columns.contains(&x) {
                    let start = at - DEPENDENT_LEFT;
                    let window = |row: usize| {
                        let from = start + row * width;
                        let own: &[f32; DEPENDENT_SPAN] =
                            own[from..from + DEPENDENT_SPAN].try_into().unwrap();
                        let contexts: &[u16; DEPENDENT_SPAN] =
                            contexts[from..from + DEPENDENT_SPAN].try_into().unwrap();
                        let ink: &[bool; DEPENDENT_SPAN] =
                            ink[from..from + DEPENDENT_SPAN].try_into().unwrap();
                        (own, contexts, ink)
                    };
                    let windows = [window(0), window(1), window(2)];
                    for &(row, column, bit) in &DEPENDENTS {
                        let (own, contexts, ink) = windows[row];
                        gain += own[column]
                            - cost[usize::from(contexts[column] ^ (1 << bit))]
                                [usize::from(ink[column])];
                    }
                } else {
                    for &(row, column, bit) in &DEPENDENTS {
                        let (qx, qy) = ((x + column).wrapping_sub(DEPENDENT_LEFT), y + row);
                        if qx >= width || qy >= height {
                            continue;
                        }
                        let q = qy * width + qx;
                        gain += own[q]
                            - cost[usize::from(contexts[q] ^ (1 << bit))][usize::from(ink[q])];
                    }
                }
                if gain > budget.min_gain {
                    candidates.push((gain, at as u32));
                }
            }
        }
        candidates.shared_sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        marked.fill(false);
        let mut pass_flips = 0usize;
        for &(_, at) in &candidates {
            let at = at as usize;
            let (x, y) = (at % width, at / width);
            let crowded =
                (y.saturating_sub(FLIP_GAP_Y)..(y + FLIP_GAP_Y + 1).min(height)).any(|ny| {
                    let row = ny * width;
                    marked
                        [row + x.saturating_sub(FLIP_GAP_X)..row + (x + FLIP_GAP_X + 1).min(width)]
                        .iter()
                        .any(|&t| t)
                });
            if crowded {
                continue;
            }
            let mut code = 0u8;
            let mut bit = 0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx, dy) != (0, 0) {
                        code |= u8::from(pixel(ink, x as isize + dx, y as isize + dy)) << bit;
                        bit += 1;
                    }
                }
            }
            if !keeps_topology(code) {
                continue;
            }
            // The flip's footprint in the low-pass error.
            let sign = if ink[at] { -1.0 } else { 1.0 };
            let footprint = |ky: usize| (y + ky) * blur_width + x..(y + ky) * blur_width + x + side;
            let within = kernel
                .chunks_exact(side)
                .enumerate()
                .all(|(ky, kernel_row)| {
                    blur_error[footprint(ky)]
                        .iter()
                        .zip(kernel_row)
                        .all(|(&error, &k)| (error + sign * k).abs() <= budget.blur)
                });
            if !within {
                continue;
            }
            for (ky, kernel_row) in kernel.chunks_exact(side).enumerate() {
                add_scaled(&mut blur_error[footprint(ky)], kernel_row, sign);
            }
            ink[at] = !ink[at];
            marked[at] = true;
            pass_flips += 1;
        }
        flipped += pass_flips;
        if pass_flips == 0 {
            break;
        }
    }
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_table_knows_outline_pixels_from_bridges() {
        // Neighbors in row-major order without the center; bit set = ink.
        let code = |cells: [u8; 8]| {
            cells
                .iter()
                .enumerate()
                .fold(0u8, |acc, (i, &c)| acc | c << i)
        };
        // On a straight edge: ink below, paper above.
        assert!(keeps_topology(code([0, 0, 0, 0, 0, 1, 1, 1])));
        assert!(keeps_topology(code([0, 0, 0, 1, 1, 1, 1, 1])));
        // An isolated pixel and the inside of a hole: a shape would appear
        // or vanish.
        assert!(!keeps_topology(0));
        assert!(!keeps_topology(0xff));
        // A bridge between ink above and ink below.
        assert!(!keeps_topology(code([0, 1, 0, 0, 0, 0, 1, 0])));
        // Two diagonal neighbors that only the center connects.
        assert!(!keeps_topology(code([1, 0, 0, 0, 0, 0, 0, 1])));
    }

    #[test]
    fn flips_remove_outline_noise_and_keep_every_shape() {
        // Rows of bars with single-pixel bumps and notches on their edges.
        let (w, h) = (192, 96);
        let bar = |x: usize, y: usize| y % 16 >= 4 && y % 16 < 12 && x % 48 >= 6 && x % 48 < 42;
        let clean: Vec<bool> = (0..w * h).map(|i| bar(i % w, i / w)).collect();
        let mut ink = clean.clone();
        let mut noise = 0;
        for y in (3..h).step_by(16) {
            for x in (13..w).step_by(48) {
                ink[y * w + x] = true; // a bump above the bar
                ink[(y + 8) * w + x + 9] = false; // a notch in its lower edge
                noise += 2;
            }
        }
        let shapes = |ink: &[bool]| {
            let mut seen = vec![false; ink.len()];
            let mut count = 0;
            for start in 0..ink.len() {
                if !ink[start] || seen[start] {
                    continue;
                }
                count += 1;
                let mut stack = vec![start];
                seen[start] = true;
                while let Some(at) = stack.pop() {
                    let (x, y) = (at % w, at / w);
                    for ny in y.saturating_sub(1)..(y + 2).min(h) {
                        for nx in x.saturating_sub(1)..(x + 2).min(w) {
                            if ink[ny * w + nx] && !seen[ny * w + nx] {
                                seen[ny * w + nx] = true;
                                stack.push(ny * w + nx);
                            }
                        }
                    }
                }
            }
            count
        };
        let noisy = ink.clone();
        let flipped = flip_pixels(&mut ink, w, h, FlipBudget::at(2.0).unwrap());
        assert!(flipped > 0);
        let differing = |a: &[bool], b: &[bool]| a.iter().zip(b).filter(|(a, b)| a != b).count();
        // Closer to the clean page than the noisy one was, and no bar lost,
        // split or merged.
        assert!(differing(&ink, &clean) < noise);
        assert_eq!(differing(&ink, &noisy), flipped);
        assert_eq!(shapes(&ink), shapes(&clean));
        assert!(FlipBudget::at(0.1).is_none());
        // The budget only loosens with the distance.
        let budgets: Vec<FlipBudget> =
            [0.25, 0.4, 0.5, 0.8, 1.0, 1.7, 2.0, 2.5, 3.5, 5.0, 8.0, 20.0]
                .iter()
                .map(|&d| FlipBudget::at(d).unwrap())
                .collect();
        assert!(budgets.windows(2).all(|w| w[1].blur >= w[0].blur
            && w[1].min_gain <= w[0].min_gain
            && w[1].passes >= w[0].passes));
        assert_eq!(
            budgets[4],
            FlipBudget {
                blur: 0.075,
                min_gain: 1.5,
                passes: 1
            }
        );
        // Bars are not a dither; a checkerboard is.
        assert!(!is_dither(&clean, w, h));
        let checker: Vec<bool> = (0..w * h).map(|i| (i % w + i / w) % 2 == 0).collect();
        assert!(is_dither(&checker, w, h));
        // A clean page has nothing to gain.
        let mut page = clean.clone();
        assert_eq!(
            flip_pixels(&mut page, w, h, FlipBudget::at(2.0).unwrap()),
            0
        );
    }

    #[test]
    fn rolling_contexts_match_the_template() {
        let (w, h) = (97, 61);
        let mut seed = 777u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 16
        };
        let ink: Vec<bool> = (0..w * h).map(|_| next() % 3 == 0).collect();
        let mut contexts = vec![0u16; w * h];
        let mut counts = vec![[0u32; 2]; 1 << 16];
        learn_contexts(&ink, w, &mut contexts, &mut counts);
        let pixel = |x: isize, y: isize| {
            x >= 0
                && y >= 0
                && (x as usize) < w
                && (y as usize) < h
                && ink[y as usize * w + x as usize]
        };
        for y in 0..h {
            for x in 0..w {
                let mut want = 0u16;
                for &(dy, dx) in &TEMPLATE {
                    want |=
                        u16::from(pixel(x as isize + dx, y as isize + dy)) << context_bit(dy, dx);
                }
                assert_eq!(contexts[y * w + x], want, "at ({x}, {y})");
            }
        }
    }
}
