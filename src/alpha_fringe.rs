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

/// Largest alpha the fringe pass clears.
const REACH: u8 = 4;
/// Half side of the window the cleared alpha is averaged over.
const WINDOW_RADIUS: usize = 4;
/// The mean cleared alpha over that window stays within one level in this
/// many.
const MEAN_DIVISOR: u32 = 2;

/// Clears the near-transparent fringe of a lossy alpha plane, as resampling
/// leaves around a cut-out.
pub(crate) fn clear(alpha: &mut [u8], width: usize, height: usize) {
    let fringe = |a: u8| (1..=REACH).contains(&a);
    let stride = width + 1;
    let mut integral = vec![0u32; stride * (height + 1)];
    let mut any = false;
    for y in 0..height {
        let mut row = 0u32;
        for x in 0..width {
            let a = alpha[y * width + x];
            if fringe(a) {
                row += a as u32;
                any = true;
            }
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row;
        }
    }
    if !any {
        return;
    }
    for y in 0..height {
        let y0 = y.saturating_sub(WINDOW_RADIUS);
        let y1 = (y + WINDOW_RADIUS + 1).min(height);
        for x in 0..width {
            let a = &mut alpha[y * width + x];
            if !fringe(*a) {
                continue;
            }
            let x0 = x.saturating_sub(WINDOW_RADIUS);
            let x1 = (x + WINDOW_RADIUS + 1).min(width);
            let sum = integral[y1 * stride + x1] + integral[y0 * stride + x0]
                - integral[y0 * stride + x1]
                - integral[y1 * stride + x0];
            if sum * MEAN_DIVISOR <= ((y1 - y0) * (x1 - x0)) as u32 {
                *a = 0;
            }
        }
    }
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
        clear(&mut alpha, w, h);
        assert!((0..h).all(|y| alpha[y * w + 20] == 0));
        assert_eq!(alpha[35 * w + 52], 2);
        assert_eq!(alpha[35 * w + 10], 255);
    }
}
