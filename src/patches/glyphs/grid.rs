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
use super::components::{Component, Components};
use super::{GLYPH_MAX_SIDE, GlyphParams, GlyphShape, Lattice, PixelBox};
use std::collections::{BTreeMap, HashMap};

/// Character grid of monospace text: cells of `px` x `py` pixels whose
/// boundaries fall on `x % px == ox` and `y % py == oy`.
pub(super) struct GlyphGrid {
    px: usize,
    ox: usize,
    py: usize,
    oy: usize,
}

/// Shortest and longest cell side searched for a character grid.
const GLYPH_GRID_MIN_PITCH: usize = 4;
const GLYPH_GRID_MAX_PITCH: usize = GLYPH_MAX_SIDE;
/// Normalized autocorrelation the ink profile must reach at the pitch.
const GLYPH_GRID_MIN_CORRELATION: f32 = 0.5;
/// ... and how far it must fall below that between zero lag and the pitch:
/// a smooth profile correlates at every short lag without repeating.
const GLYPH_GRID_MIN_TROUGH_DROP: f32 = 0.6;
/// Share of a component's cells that must repeat within it for the grid
/// split to stand; other wide shapes (rules, frames, icons) stay whole.
const GLYPH_GRID_MIN_REPEAT_SHARE: f32 = 0.5;

/// A shorter lag dividing the best one is the pitch when its peak reaches
/// this share of the best: a row of byte pairs repeats every three cells
/// more strongly than every cell.
const GLYPH_GRID_DIVISOR_SHARE: f32 = 0.6;
/// Share of the isolated glyphs that must fit in a divisor taken from the
/// glyph votes, and the share of the winner's votes it needs there.
const GLYPH_GRID_DIVISOR_FIT: f32 = 0.9;
/// Share of the glyphs that must fit in the voted pitch itself.
const GLYPH_GRID_PITCH_FIT: f32 = 0.5;
const GLYPH_GRID_VOTE_DIVISOR_SHARE: f32 = 0.1;

/// The lag at which an ink profile repeats, if it does so strongly: the
/// first local peak of the normalized autocorrelation within a tenth of the
/// best one (or a divisor of it that also peaks), with a clear trough
/// before it. The profile is high-passed first: left-aligned words, table
/// columns and bars give it an envelope that is not the character pitch.
fn glyph_grid_pitch(profile: &[u32]) -> Option<usize> {
    let max_lag = GLYPH_GRID_MAX_PITCH.min(profile.len() / 4);
    if max_lag < GLYPH_GRID_MIN_PITCH {
        return None;
    }
    // Moving average over one longest pitch either side, by prefix sums
    // (integer: the ink of a large frame does not fit f32 exactly).
    let mut prefix: Vec<u64> = Vec::with_capacity(profile.len() + 1);
    prefix.push(0);
    for &v in profile {
        prefix.push(prefix[prefix.len() - 1] + v as u64);
    }
    let detail: Vec<f32> = (0..profile.len())
        .map(|i| {
            let (lo, hi) = (
                i.saturating_sub(max_lag),
                (i + max_lag + 1).min(profile.len()),
            );
            profile[i] as f32 - (prefix[hi] - prefix[lo]) as f32 / (hi - lo) as f32
        })
        .collect();
    let variance: f32 = detail.iter().map(|v| v * v).sum();
    if variance <= 0.0 {
        return None;
    }
    let correlation = |lag: usize| -> f32 {
        let sum: f32 = detail[..detail.len() - lag]
            .iter()
            .zip(&detail[lag..])
            .map(|(a, b)| a * b)
            .sum();
        sum / variance
    };
    // scores[lag], with one lag past the range for the peak test.
    let scores: Vec<f32> = (0..=max_lag + 1).map(correlation).collect();
    let peak = |lag: usize| scores[lag] > scores[lag - 1] && scores[lag] >= scores[lag + 1];
    let best = (GLYPH_GRID_MIN_PITCH..=max_lag)
        .filter(|&lag| peak(lag))
        .map(|lag| scores[lag])
        .fold(f32::MIN, f32::max);
    if best < GLYPH_GRID_MIN_CORRELATION {
        return None;
    }
    let mut pitch =
        (GLYPH_GRID_MIN_PITCH..=max_lag).find(|&lag| peak(lag) && scores[lag] >= 0.9 * best)?;
    for divisor in (2..=pitch / GLYPH_GRID_MIN_PITCH).rev() {
        let lag = pitch / divisor;
        if pitch % divisor == 0 && peak(lag) && scores[lag] >= GLYPH_GRID_DIVISOR_SHARE * best {
            pitch = lag;
            break;
        }
    }
    let trough = scores[1..pitch].iter().copied().fold(f32::MAX, f32::min);
    (trough <= scores[pitch] - GLYPH_GRID_MIN_TROUGH_DROP).then_some(pitch)
}

/// Occurrences of one glyph, consecutive in scan order, that cast one vote.
const GLYPH_GRID_VOTE_GROUP: usize = 8;
/// Votes needed before the glyphs decide whether there is a grid at all.
const GLYPH_GRID_MIN_VOTES: usize = 6;
/// Share of the votes the winning pitch needs. Text in a proportional font
/// next to the monospace text votes for no pitch at all, so the bar is low;
/// by chance such text agrees on a pitch far less often than this.
const GLYPH_GRID_VOTE_SHARE: f32 = 0.25;
/// A grid cell's ink must be wider and taller than this to count as a glyph.
const GLYPH_GRID_MIN_CELL_INK: usize = 2;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The pitch of monospace text from where its glyphs repeat. Every
/// occurrence of one glyph sits at the same offset in its cell, so their
/// coordinates differ by whole cells and the greatest common divisor of the
/// differences in a small group is the pitch (seldom a multiple). Each
/// group of `GLYPH_GRID_VOTE_GROUP` occurrences votes; a glyph that also
/// appears off the grid spoils only its own groups. `positions` holds
/// (glyph, coordinate) in scan order. `Err` when there are too few votes to
/// tell, `Ok(None)` when the glyphs agree on no pitch.
///
/// A layout that repeats every few lines (a bar on every third row) votes
/// for that multiple. A divisor of the winner takes over when it has votes
/// of its own and nearly all glyphs fit in it; `sizes` are the glyphs'
/// extents along the axis. The fit keeps out the short lags that the two
/// equal parts of a colon or an equals sign vote for.
fn glyph_grid_vote(positions: &[(u64, usize)], sizes: &[usize]) -> Result<Option<usize>, ()> {
    let mut by_glyph: HashMap<u64, Vec<usize>> = HashMap::new();
    for &(glyph, at) in positions {
        by_glyph.entry(glyph).or_default().push(at);
    }
    let mut votes = [0usize; GLYPH_GRID_MAX_PITCH + 1];
    let mut cast = 0usize;
    for coordinates in by_glyph.values() {
        for group in coordinates.as_chunks::<GLYPH_GRID_VOTE_GROUP>().0 {
            let step = group.iter().fold(0, |g, &at| gcd(g, at.abs_diff(group[0])));
            // A group stacked at one coordinate says nothing along this axis.
            if step == 0 {
                continue;
            }
            cast += 1;
            if (GLYPH_GRID_MIN_PITCH..=GLYPH_GRID_MAX_PITCH).contains(&step) {
                votes[step] += 1;
            }
        }
    }
    if cast < GLYPH_GRID_MIN_VOTES {
        return Err(());
    }
    let fits = |lag: usize, share: f32| {
        let within = sizes.iter().filter(|&&size| size <= lag).count();
        within as f32 >= share * sizes.len() as f32
    };
    // The most voted pitch that holds the typical glyph, the longer one on
    // a tie. The parts of a colon, or a dot shared by several characters,
    // repeat at lags shorter than a cell.
    let pitch = (GLYPH_GRID_MIN_PITCH..=GLYPH_GRID_MAX_PITCH)
        .filter(|&pitch| fits(pitch, GLYPH_GRID_PITCH_FIT))
        .max_by_key(|&pitch| (votes[pitch], pitch))
        .unwrap_or(0);
    if (votes[pitch] as f32) < GLYPH_GRID_VOTE_SHARE * cast as f32 {
        return Ok(None);
    }
    let divisor = (2..=pitch / GLYPH_GRID_MIN_PITCH)
        .rev()
        .filter(|k| pitch % k == 0)
        .map(|k| pitch / k)
        .find(|&lag| {
            votes[lag] as f32 >= GLYPH_GRID_VOTE_DIVISOR_SHARE * votes[pitch] as f32
                && fits(lag, GLYPH_GRID_DIVISOR_FIT)
        });
    Ok(Some(divisor.unwrap_or(pitch)))
}

/// The character grid of monospace text. The pitch comes from where
/// isolated glyphs repeat; a frame with too few of those (every character
/// touching its neighbors) falls back to the period of its ink profile.
/// Cell boundaries sit where they cut the fewest isolated glyphs, then
/// where the fewest ink runs cross them.
pub(super) fn detect_glyph_grid(
    ink: &[bool],
    width: usize,
    height: usize,
    shapes: &[&GlyphShape],
) -> Option<GlyphGrid> {
    let along = |coordinate: fn(&GlyphShape) -> usize| -> Vec<(u64, usize)> {
        shapes.iter().map(|s| (s.hash, coordinate(s))).collect()
    };
    let widths: Vec<usize> = shapes.iter().map(|s| s.w).collect();
    let heights: Vec<usize> = shapes.iter().map(|s| s.h).collect();
    let vote_x = glyph_grid_vote(&along(|s| s.x0), &widths);
    let vote_y = glyph_grid_vote(&along(|s| s.y0), &heights);
    if vote_x == Ok(None) || vote_y == Ok(None) {
        return None;
    }
    let (mut cols, mut rows) = (Vec::new(), Vec::new());
    if vote_x.is_err() || vote_y.is_err() {
        cols = vec![0u32; width];
        rows = vec![0u32; height];
        for (row, total) in ink.chunks_exact(width).zip(&mut rows) {
            for (&pixel, col) in row.iter().zip(&mut cols) {
                if pixel {
                    *col += 1;
                    *total += 1;
                }
            }
        }
    }
    let px = match vote_x {
        Ok(pitch) => pitch?,
        Err(()) => glyph_grid_pitch(&cols)?,
    };
    let py = match vote_y {
        Ok(pitch) => pitch?,
        Err(()) => glyph_grid_pitch(&rows)?,
    };
    // (isolated glyphs cut, ink runs crossed) per boundary phase. Positions
    // advance one at a time, so the phase is a wrapping counter and the
    // per-pixel pass counts by column, folded onto the pitch once.
    let mut cross_x = vec![(0usize, 0usize); px];
    let mut cross_y = vec![(0usize, 0usize); py];
    let cut = |crossings: &mut [(usize, usize)], start: usize, len: usize| {
        let pitch = crossings.len();
        let mut phase = (start + 1) % pitch;
        for _ in 1..len {
            crossings[phase].0 += 1;
            phase += 1;
            if phase == pitch {
                phase = 0;
            }
        }
    };
    for s in shapes {
        if s.w <= px {
            cut(&mut cross_x, s.x0, s.w);
        }
        if s.h <= py {
            cut(&mut cross_y, s.y0, s.h);
        }
    }
    let mut runs_x = vec![0u32; width];
    let mut phase_y = 0;
    for (y, row) in ink.chunks_exact(width).enumerate() {
        for (runs, pair) in runs_x[1..].iter_mut().zip(row.windows(2)) {
            *runs += u32::from(pair[0] & pair[1]);
        }
        if y > 0 {
            let above = &ink[(y - 1) * width..y * width];
            cross_y[phase_y].1 += row.iter().zip(above).filter(|&(&a, &b)| a & b).count();
        }
        phase_y += 1;
        if phase_y == py {
            phase_y = 0;
        }
    }
    for (column, &runs) in runs_x.chunks(px).flat_map(|chunk| chunk.iter().enumerate()) {
        cross_x[column].1 += runs as usize;
    }
    let phase = |crossings: &[(usize, usize)]| {
        crossings
            .iter()
            .enumerate()
            .min_by_key(|&(_, &count)| count)
            .map_or(0, |(i, _)| i)
    };
    Some(GlyphGrid {
        px,
        ox: phase(&cross_x),
        py,
        oy: phase(&cross_y),
    })
}

/// The cells a component was cut into.
pub(super) struct CellSplit {
    /// One shape per occupied cell, glyph cells first.
    pub(super) shapes: Vec<GlyphShape>,
    /// How many of the shapes are glyph cells.
    glyphs: usize,
}

impl CellSplit {
    fn glyph_cells(&self) -> &[GlyphShape] {
        &self.shapes[..self.glyphs]
    }
}

impl GlyphGrid {
    #[inline]
    fn cell_x(&self, x: usize) -> usize {
        (x + self.px - self.ox) / self.px
    }

    #[inline]
    fn cell_y(&self, y: usize) -> usize {
        (y + self.py - self.oy) / self.py
    }

    /// Relabels a component wider or taller than a cell and a half by grid
    /// cell and returns one shape per occupied cell.
    fn cut(
        &self,
        lattice: &Lattice,
        component: &Component,
        order: &[u32],
        label: &mut [u32],
        next_label: &mut u32,
        min_contrast: i32,
    ) -> Option<CellSplit> {
        let width = lattice.width;
        let (x0, x1, y0, y1) = component.bounds;
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        if w * 2 <= self.px * 3 && h * 2 <= self.py * 3 {
            return None;
        }
        let (cx0, cy0) = (self.cell_x(x0), self.cell_y(y0));
        let cells_x = self.cell_x(x1) - cx0 + 1;
        let cells_y = self.cell_y(y1) - cy0 + 1;
        let first_label = *next_label + 1;
        // The work follows the component's pixels, not its box: a long thin
        // shape spans many cells and fills few.
        let mut cells: BTreeMap<usize, PixelBox> = BTreeMap::new();
        for &at in &order[component.pixels.clone()] {
            let at = at as usize;
            let (x, y) = (at % width, at / width);
            let cell = (self.cell_y(y) - cy0) * cells_x + self.cell_x(x) - cx0;
            label[at] = first_label + cell as u32;
            let c = cells.entry(cell).or_insert((x, x, y, y));
            *c = (c.0.min(x), c.1.max(x), c.2.min(y), c.3.max(y));
        }
        *next_label += (cells_x * cells_y) as u32;
        // Straight strokes (rules, frame edges) are no glyphs and do not
        // vouch for the split: any long line cuts into equal cells. Once
        // the glyph cells carry it they are placed too, since a line left
        // in the base with its junctions cut out costs more than the
        // whole line did.
        let is_glyph = |&(bx0, bx1, by0, by1): &PixelBox| {
            bx1 - bx0 >= GLYPH_GRID_MIN_CELL_INK && by1 - by0 >= GLYPH_GRID_MIN_CELL_INK
        };
        let label = &*label;
        let cell_shapes = |glyphs: bool| {
            cells
                .iter()
                .filter(move |&(_, bounds)| is_glyph(bounds) == glyphs)
                .filter_map(|(&cell, &bounds)| {
                    let owner = first_label + cell as u32;
                    lattice.shape_of(label, bounds, owner, component.bg, min_contrast)
                })
        };
        let mut shapes: Vec<GlyphShape> = cell_shapes(true).collect();
        let glyphs = shapes.len();
        shapes.extend(cell_shapes(false));
        Some(CellSplit { shapes, glyphs })
    }
}

/// Touching monospace characters: a component over several cells of the
/// character grid goes back to one shape per cell, where enough of those
/// cells are glyphs seen elsewhere in the frame. Returns the split of every
/// component, `None` for the ones left whole.
pub(super) fn split_touching_glyphs(
    grid: &GlyphGrid,
    lattice: &Lattice,
    components: &mut Components,
    params: &GlyphParams,
) -> Vec<Option<CellSplit>> {
    let Components {
        list,
        order,
        label,
        next_label,
    } = components;
    let mut splits: Vec<Option<CellSplit>> = list
        .iter()
        .map(|component| {
            grid.cut(
                lattice,
                component,
                order,
                label,
                next_label,
                params.min_contrast,
            )
        })
        .collect();
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let whole = list.iter().filter_map(|c| c.shape.as_ref());
    let cells = splits.iter().flatten().flat_map(CellSplit::glyph_cells);
    for shape in whole.chain(cells) {
        *seen.entry(shape.hash).or_default() += 1;
    }
    for (component, split) in list.iter().zip(&mut splits) {
        let Some(cells) = split else {
            continue;
        };
        let glyphs = cells.glyph_cells();
        let repeated = glyphs.iter().filter(|s| seen[&s.hash] >= 2).count();
        if glyphs.len() >= 2 && repeated as f32 >= GLYPH_GRID_MIN_REPEAT_SHARE * glyphs.len() as f32
        {
            continue;
        }
        // Not text on this grid: the component is whole again.
        *split = None;
        for &at in &order[component.pixels.clone()] {
            label[at as usize] = component.label;
        }
    }
    splits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Touching plus signs on a 7 x 13 character grid, `cols` x `rows` cells.
    fn plus_grid_ink(cols: usize, rows: usize) -> (Vec<bool>, usize, usize) {
        let (width, height) = (cols * 7 + 5, rows * 13 + 6);
        let mut ink = vec![false; width * height];
        for r in 0..rows {
            for c in 0..cols {
                let (x0, y0) = (3 + c * 7, 2 + r * 13);
                // Horizontal arm over the whole cell, so neighbours touch.
                for dx in 0..7 {
                    ink[(y0 + 6) * width + x0 + dx] = true;
                }
                for dy in 3..10 {
                    ink[(y0 + dy) * width + x0 + 3] = true;
                }
            }
        }
        (ink, width, height)
    }

    #[test]
    fn glyph_grid_finds_touching_monospace_cells() {
        let (ink, width, height) = plus_grid_ink(40, 20);
        let grid = detect_glyph_grid(&ink, width, height, &[]).expect("periodic ink");
        assert_eq!((grid.px, grid.py), (7, 13));
        // Rows are separated by blank lines; the boundary lies in one.
        let boundary_ink = (0..height)
            .filter(|y| y % grid.py == grid.oy)
            .flat_map(|y| ink[y * width..(y + 1) * width].iter())
            .filter(|&&p| p)
            .count();
        assert_eq!(boundary_ink, 0);
    }

    #[test]
    fn glyph_grid_rejects_aperiodic_ink() {
        // A filled disc: smooth profiles that correlate at every short lag.
        let (width, height) = (200usize, 160usize);
        let mut ink = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let (dx, dy) = (x as i64 - 100, y as i64 - 80);
                ink[y * width + x] = dx * dx + dy * dy < 60 * 60;
            }
        }
        assert!(detect_glyph_grid(&ink, width, height, &[]).is_none());
        // Pseudo-random speckle has no pitch either.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for pixel in &mut ink {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *pixel = state % 5 == 0;
        }
        assert!(detect_glyph_grid(&ink, width, height, &[]).is_none());
    }

    #[test]
    fn glyph_grid_vote_finds_cell_pitch() {
        // Three glyphs on a 9-pixel pitch, each at its own offset in the
        // cell, at scattered columns; a fourth made of two equal parts.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n) as usize
        };
        let mut positions = Vec::new();
        for glyph in 0..3u64 {
            for _ in 0..40 {
                positions.push((glyph, next(60) * 9 + glyph as usize + 1));
            }
        }
        for _ in 0..16 {
            let cell = next(60) * 9;
            positions.push((7, cell + 2));
            positions.push((7, cell + 5));
        }
        let sizes = vec![6usize; positions.len()];
        assert_eq!(glyph_grid_vote(&positions, &sizes), Ok(Some(9)));
        // Too few repeats decide nothing.
        assert_eq!(glyph_grid_vote(&positions[..5], &sizes[..5]), Err(()));
        // A proportional font: the same glyphs at arbitrary columns.
        let loose: Vec<(u64, usize)> = (0..160).map(|i| (i % 4, next(900))).collect();
        assert_eq!(glyph_grid_vote(&loose, &sizes), Ok(None));
    }

    #[test]
    fn glyph_grid_vote_ignores_lags_shorter_than_the_glyphs() {
        // A dot repeating 5 apart (colons) outvotes the 20-pixel rows, but
        // no 17-pixel glyph fits a 5-pixel cell.
        let mut positions = Vec::new();
        for i in 0..80 {
            positions.push((1u64, i * 5));
        }
        for i in 0..40 {
            positions.push((2u64, i * 20));
        }
        let sizes: Vec<usize> = (0..120).map(|i| if i < 40 { 2 } else { 17 }).collect();
        assert_eq!(glyph_grid_vote(&positions, &sizes), Ok(Some(20)));
    }

    #[test]
    fn glyph_grid_vote_prefers_cell_over_layout_period() {
        // One glyph on every third row, another on every row: the row
        // pitch is 15 although most votes are for 45.
        let mut positions = Vec::new();
        for row in 0..80usize {
            positions.push((1u64, row * 45 + 3));
        }
        for row in 0..24usize {
            positions.push((2u64, row * 15 + 4));
        }
        let sizes = vec![11usize; positions.len()];
        assert_eq!(glyph_grid_vote(&positions, &sizes), Ok(Some(15)));
        // Glyphs taller than the divisor rule it out.
        let tall = vec![20usize; positions.len()];
        assert_eq!(glyph_grid_vote(&positions, &tall), Ok(Some(45)));
    }
}
