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

//! Near-transparent fringe of a lossy alpha plane: the faint samples
//! resampling leaves around a cut-out are cleared before coding, which makes
//! their pixels invisible and frees their color as well.
//!
//! The same pass at the other end fills a faint deficit below full opacity
//! (a dithered 254 / 255 backdrop), which otherwise costs a textured plane
//! for an invisible change.

/// Largest alpha the fringe pass clears, in 8-bit levels.
const REACH: u32 = 4;
/// Half side of the window the cleared alpha is averaged over.
const WINDOW_RADIUS: usize = 4;
/// The mean cleared alpha over that window stays within one 8-bit level in
/// this many.
const MEAN_DIVISOR: u64 = 2;
/// The near-opaque fill runs only when it would change at least one sample
/// in this many: on fewer it removes no texture worth its bytes, and sparse
/// fills cost more than they save on patterned planes.
const FILL_SHARE: usize = 12;

/// An alpha sample of any integer depth.
pub(crate) trait AlphaSample: Copy {
    const ZERO: Self;

    fn level(self) -> u32;

    fn from_level(level: u32) -> Self;
}

impl AlphaSample for u8 {
    const ZERO: Self = 0;

    #[inline]
    fn level(self) -> u32 {
        u32::from(self)
    }

    #[inline]
    fn from_level(level: u32) -> Self {
        level as u8
    }
}

impl AlphaSample for u16 {
    const ZERO: Self = 0;

    #[inline]
    fn level(self) -> u32 {
        u32::from(self)
    }

    #[inline]
    fn from_level(level: u32) -> Self {
        level as u16
    }
}

/// Clears the near-transparent fringe of a lossy alpha plane, as resampling
/// leaves around a cut-out, and fills a faint deficit below full opacity;
/// `max` is the full-opacity sample value.
pub(crate) fn clear<T: AlphaSample>(alpha: &mut [T], width: usize, height: usize, max: u32) {
    if width == 0 || height == 0 {
        return;
    }
    let alpha = &mut alpha[..width * height];
    sweep(alpha, width, height, max, |level| level, T::ZERO);

    // Most planes have no broad near-opaque area and leave here.
    let reach = REACH * max / 255;
    let near_opaque = alpha
        .iter()
        .filter(|a| (1..=reach).contains(&(max - a.level().min(max))))
        .count();
    if near_opaque * FILL_SHARE < alpha.len() {
        return;
    }
    let mut filled = alpha.to_vec();
    let opaque = T::from_level(max);
    let changed = sweep(
        &mut filled,
        width,
        height,
        max,
        |level| max - level.min(max),
        opaque,
    );
    if changed * FILL_SHARE >= alpha.len() {
        alpha.copy_from_slice(&filled);
    }
}

/// Replaces the samples whose `distance` from one end of the range is at
/// most [`REACH`] levels, where the mean of that distance over the window
/// stays within one level in [`MEAN_DIVISOR`]; returns how many changed.
fn sweep<T: AlphaSample>(
    alpha: &mut [T],
    width: usize,
    height: usize,
    max: u32,
    distance: impl Fn(u32) -> u32,
    replacement: T,
) -> usize {
    let reach = REACH * max / 255;
    let fringe_level = |a: T| {
        let level = distance(a.level());
        if level <= reach { level } else { 0 }
    };
    let mut changed = 0usize;
    let Some(first) = alpha.iter().position(|&a| fringe_level(a) != 0) else {
        return 0;
    };
    let last = alpha[first..]
        .iter()
        .rposition(|&a| fringe_level(a) != 0)
        .unwrap_or(0)
        + first;
    let first_row = first / width;
    let last_row = last / width;

    // Keep the original samples in the vertical window until they slide out.
    // One bit per pending row marks the samples to clear as they leave.
    const PENDING_ROWS: usize = WINDOW_RADIUS + 1;
    let pending_bit = |y| 1u8 << (y % PENDING_ROWS);
    let mut columns = vec![0u32; width];
    let mut pending = vec![0u8; width];
    let mut top = first_row.saturating_sub(WINDOW_RADIUS);
    let mut bottom = (first_row + WINDOW_RADIUS + 1).min(height);
    for row in alpha[top * width..bottom * width].chunks_exact(width) {
        for (column, &a) in columns.iter_mut().zip(row) {
            *column += fringe_level(a);
        }
    }

    for y in first_row..=last_row {
        let row = &alpha[y * width..(y + 1) * width];
        let bit = pending_bit(y);
        let mut next_x = usize::MAX;
        let mut sum = 0u32;
        for (x, (&a, pending)) in row.iter().zip(&mut pending).enumerate() {
            if fringe_level(a) == 0 {
                continue;
            }
            let x0 = x.saturating_sub(WINDOW_RADIUS);
            let x1 = (x + WINDOW_RADIUS + 1).min(width);
            // Reuse the sum along runs of fringe pixels, and skip empty spans.
            if x == next_x {
                if x > WINDOW_RADIUS {
                    sum -= columns[x0 - 1];
                }
                if x + WINDOW_RADIUS < width {
                    sum += columns[x1 - 1];
                }
            } else {
                sum = columns[x0..x1].iter().sum();
            }
            next_x = x + 1;
            if u64::from(sum) * MEAN_DIVISOR * 255
                <= ((bottom - top) * (x1 - x0)) as u64 * u64::from(max)
            {
                *pending |= bit;
            }
        }

        if y == last_row {
            break;
        }
        if y >= WINDOW_RADIUS {
            let bit = pending_bit(top);
            let row = &mut alpha[top * width..(top + 1) * width];
            for ((column, pending), a) in columns.iter_mut().zip(&mut pending).zip(row) {
                *column -= fringe_level(*a);
                if *pending & bit != 0 {
                    *pending &= !bit;
                    *a = replacement;
                    changed += 1;
                }
            }
            top += 1;
        }
        if bottom < height {
            let row = &alpha[bottom * width..(bottom + 1) * width];
            for (column, &a) in columns.iter_mut().zip(row) {
                *column += fringe_level(a);
            }
            bottom += 1;
        }
    }

    for (y, row) in (top..=last_row).zip(alpha[top * width..].chunks_exact_mut(width)) {
        let bit = pending_bit(y);
        for (a, &pending) in row.iter_mut().zip(&pending) {
            if pending & bit != 0 {
                *a = replacement;
                changed += 1;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clears_thin_edges_and_keeps_shadow_tails() {
        let (w, h) = (64, 64);
        // A one-pixel fringe column beside an opaque block, and a wide field
        // of faint alpha (a shadow tail).
        let mut alpha = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                alpha[y * w + x] = match (x, y) {
                    (0..=19, _) => 255,
                    (20, _) => 3,
                    (40..=63, 20..=50) => 2,
                    _ => 0,
                };
            }
        }
        let mut deep: Vec<u16> = alpha.iter().map(|&a| u16::from(a) * 257).collect();
        clear(&mut alpha, w, h, 255);
        assert!((0..h).all(|y| alpha[y * w + 20] == 0));
        assert_eq!(alpha[35 * w + 52], 2);
        assert_eq!(alpha[35 * w + 10], 255);
        // The 16-bit plane of the same picture clears the same samples.
        clear(&mut deep, w, h, 65535);
        assert!(
            deep.iter()
                .zip(&alpha)
                .all(|(&d, &a)| d == u16::from(a) * 257)
        );
    }

    #[test]
    fn fills_dithered_near_opaque_backdrop_only_when_broad() {
        let (w, h) = (64, 64);
        // A backdrop that is opaque but for a one-level dither on every
        // fourth sample, over a soft band that must stay.
        let backdrop = |x: usize, y: usize| match y {
            0..=7 => 128,
            _ if (x + 2 * y) % 4 == 0 => 254,
            _ => 255,
        };
        let mut alpha: Vec<u8> = (0..w * h).map(|i| backdrop(i % w, i / w)).collect();
        let mut deep: Vec<u16> = alpha.iter().map(|&a| u16::from(a) * 257).collect();
        clear(&mut alpha, w, h, 255);
        assert!(alpha[8 * w..].iter().all(|&a| a == 255));
        assert!(alpha[..8 * w].iter().all(|&a| a == 128));
        clear(&mut deep, w, h, 65535);
        assert!(
            deep.iter()
                .zip(&alpha)
                .all(|(&d, &a)| d == u16::from(a) * 257)
        );

        // A few near-opaque samples on a patterned plane are left alone.
        let mut sparse = vec![255u8; w * h];
        for i in (0..w * h).step_by(40) {
            sparse[i] = 254;
        }
        let before = sparse.clone();
        clear(&mut sparse, w, h, 255);
        assert_eq!(sparse, before);

        // A broad deficit of more than half a level on average is content.
        let mut shade = vec![253u8; w * h];
        clear(&mut shade, w, h, 255);
        assert!(shade.iter().all(|&a| a == 253));
    }
}
