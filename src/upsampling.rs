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

//! Decoder-side upsampling with a nearest-neighbor kernel.
//!
//! A frame may be coded at the image size over 2, 4 or 8 and scaled back by
//! the decoder with a 5x5 kernel per output phase, signaled as the upper
//! triangle of a symmetric weight matrix in the image metadata
//! (`CustomTransformData`). A kernel that is 1 at the center tap and 0
//! elsewhere copies the coded sample into every pixel of its cell: the
//! decoder's sum is then one exact product, and its min/max clamp over the
//! window is a no-op, so an image whose NxN cells are constant round-trips
//! bit-exactly from a frame N times smaller on each side (djxl and jxl-oxide
//! verified at 2, 4 and 8, lossless, including sizes that are not multiples
//! of N: the decoder crops the upsampled frame to the image).

use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;

/// Upper-triangle weight indices that are 1 in the nearest-neighbor
/// kernel of each factor; every other weight is 0. The weight matrix of
/// factor N is 5N/2 on a side, and the phase (ky, kx) kernel is its 5x5
/// block at (5ky, 5kx) for ky, kx below N/2 (mirrored for the rest), so the
/// center taps are the diagonal and off-diagonal (5i + 2, 5j + 2) entries.
static NEAREST_ONES_2: [usize; 1] = [9];
static NEAREST_ONES_4: [usize; 3] = [19, 24, 49];
static NEAREST_ONES_8: [usize; 10] = [39, 44, 49, 54, 119, 124, 129, 174, 179, 204];

/// Number of signaled weights of the kernel of `factor`.
pub(crate) const fn weight_count(factor: u32) -> usize {
    match factor {
        2 => 15,
        4 => 55,
        _ => 210,
    }
}

/// The signaled weights of the nearest-neighbor kernel of `factor`.
pub(crate) fn nearest_weights(factor: u32) -> Vec<f32> {
    let mut weights = vec![0.0f32; weight_count(factor)];
    let ones: &[usize] = match factor {
        2 => &NEAREST_ONES_2,
        4 => &NEAREST_ONES_4,
        _ => &NEAREST_ONES_8,
    };
    for &i in ones {
        weights[i] = 1.0;
    }
    weights
}

/// The largest factor in 8, 4, 2 whose cells are constant in every plane
/// (cells cut by the right or bottom edge are judged on their visible
/// part), or 1.
pub(crate) fn constant_cell_factor(image: &Image3Si, alpha: Option<&AlphaPlane>) -> u32 {
    let xsize = image.xsize();
    let ysize = image.ysize();
    if xsize < 2 && ysize < 2 {
        return 1;
    }
    for factor in [8u32, 4, 2] {
        let n = factor as usize;
        let planes_constant = (0..3).all(|c| cells_constant(|y| image.plane_row(c, y), ysize, n));
        let alpha_constant = match alpha {
            None => true,
            Some(AlphaPlane::U8(data)) => {
                cells_constant(|y| &data[y * xsize..(y + 1) * xsize], ysize, n)
            }
            Some(AlphaPlane::U16 { data, .. }) => {
                cells_constant(|y| &data[y * xsize..(y + 1) * xsize], ysize, n)
            }
            Some(AlphaPlane::F32(data)) => {
                cells_constant(|y| &data[y * xsize..(y + 1) * xsize], ysize, n)
            }
        };
        if planes_constant && alpha_constant {
            return factor;
        }
    }
    1
}

fn cells_constant<'a, T: PartialEq + 'a>(
    row: impl Fn(usize) -> &'a [T],
    ysize: usize,
    n: usize,
) -> bool {
    for y in 0..ysize {
        let current = row(y);
        if y % n != 0 {
            if current != row(y - y % n) {
                return false;
            }
            continue;
        }
        if current
            .chunks(n)
            .any(|cell| cell[1..].iter().any(|v| *v != cell[0]))
        {
            return false;
        }
    }
    true
}

/// The image at one sample per `factor`-cell: the frame the decoder scales
/// back with the nearest-neighbor kernel.
pub(crate) fn downsample(
    image: &Image3Si,
    alpha: Option<&AlphaPlane>,
    factor: u32,
) -> (Image3Si, Option<AlphaPlane>) {
    let n = factor as usize;
    let xsize = image.xsize();
    let ysize = image.ysize();
    let bx = xsize.div_ceil(n);
    let by = ysize.div_ceil(n);
    let mut base = Image3Si::new(bx, by);
    for c in 0..3 {
        for y in 0..by {
            let src = image.plane_row(c, y * n);
            let dst = base.plane_row_mut(c, y);
            for (x, d) in dst.iter_mut().enumerate() {
                *d = src[x * n];
            }
        }
    }
    fn pick<T: Copy>(data: &[T], xsize: usize, bx: usize, by: usize, n: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(bx * by);
        for y in 0..by {
            let src = &data[y * n * xsize..];
            out.extend((0..bx).map(|x| src[x * n]));
        }
        out
    }
    let base_alpha = alpha.map(|alpha| match alpha {
        AlphaPlane::U8(data) => AlphaPlane::U8(pick(data, xsize, bx, by, n)),
        AlphaPlane::U16 { data, bits } => AlphaPlane::U16 {
            data: pick(data, xsize, bx, by, n),
            bits: *bits,
        },
        AlphaPlane::F32(data) => AlphaPlane::F32(pick(data, xsize, bx, by, n)),
    });
    (base, base_alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(w: usize, h: usize, n: usize) -> Image3Si {
        let mut image = Image3Si::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    image.plane_row_mut(c, y)[x] = ((y / n) * 131 + (x / n) * 17 + c) as i32;
                }
            }
        }
        image
    }

    #[test]
    fn detects_the_largest_constant_cell() {
        assert_eq!(constant_cell_factor(&cells(64, 48, 8), None), 8);
        assert_eq!(constant_cell_factor(&cells(64, 48, 4), None), 4);
        assert_eq!(constant_cell_factor(&cells(64, 48, 2), None), 2);
        assert_eq!(constant_cell_factor(&cells(64, 48, 1), None), 1);
        // Edge cells cut by the image border count on their visible part.
        assert_eq!(constant_cell_factor(&cells(61, 47, 4), None), 4);
        // A non-constant alpha plane vetoes the factor.
        let alpha = AlphaPlane::U8((0..64 * 48).map(|i| (i % 3) as u8).collect());
        assert_eq!(constant_cell_factor(&cells(64, 48, 4), Some(&alpha)), 1);
        let mut broken = cells(64, 48, 4);
        broken.plane_row_mut(1, 45)[62] += 1;
        assert_eq!(constant_cell_factor(&broken, None), 1);
    }

    #[test]
    fn downsample_keeps_one_sample_per_cell() {
        let image = cells(61, 47, 4);
        let (base, _) = downsample(&image, None, 4);
        assert_eq!((base.xsize(), base.ysize()), (16, 12));
        for c in 0..3 {
            for y in 0..12 {
                for x in 0..16 {
                    assert_eq!(base.plane_row(c, y)[x], image.plane_row(c, y * 4)[x * 4]);
                }
            }
        }
    }

    #[test]
    fn nearest_weights_match_libjxl() {
        let w2 = nearest_weights(2);
        assert_eq!(w2.iter().filter(|w| **w == 1.0).count(), 1);
        assert_eq!(w2[9], 1.0);
        assert_eq!(nearest_weights(4).len(), 55);
        assert_eq!(nearest_weights(8).len(), 210);
    }
}
