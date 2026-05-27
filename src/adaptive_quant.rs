/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use crate::image::{Image3F, ImageB};

const LUMA_R: f32 = 0.299;
const LUMA_G: f32 = 0.587;
const LUMA_B: f32 = 0.114;

#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    LUMA_R * r + LUMA_G * g + LUMA_B * b
}

/// Fill `raw_quant_field` with per-block quantization multipliers derived from
/// local luma gradient magnitude in `linear`.
///
/// `(x0, y0)` is the pixel offset within `linear` corresponding to the (0, 0)
/// block of `raw_quant_field`. Blocks that straddle the image edge are
/// computed from whatever pixels exist.
pub fn fill_quant_field(linear: &Image3F, raw_quant_field: &mut ImageB, x0: usize, y0: usize) {
    let xsize_blocks = raw_quant_field.xsize();
    let ysize_blocks = raw_quant_field.ysize();
    let img_xsize = linear.xsize();
    let img_ysize = linear.ysize();

    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            let px_x0 = x0 + bx * 8;
            let px_y0 = y0 + by * 8;
            let q = block_quant(linear, px_x0, px_y0, img_xsize, img_ysize);
            raw_quant_field.row_mut(by)[bx] = q;
        }
    }
}

/// Compute the quantization multiplier for a single 8×8 block whose top-left
/// pixel is at (px_x0, px_y0). Returns a value in 1..=16.
fn block_quant(
    linear: &Image3F,
    px_x0: usize,
    px_y0: usize,
    img_xsize: usize,
    img_ysize: usize,
) -> u8 {
    let mut grad_sum = 0.0f32;
    let mut n = 0u32;

    let py_end = (px_y0 + 8).min(img_ysize);
    let px_end = (px_x0 + 8).min(img_xsize);

    for py in px_y0..py_end {
        let row_r = linear.plane_row(0, py);
        let row_g = linear.plane_row(1, py);
        let row_b = linear.plane_row(2, py);

        // Precompute below-row pointers once per py (replicate at last row).
        let py_below = if py + 1 < img_ysize { py + 1 } else { py };
        let below_r = linear.plane_row(0, py_below);
        let below_g = linear.plane_row(1, py_below);
        let below_b = linear.plane_row(2, py_below);

        for px in px_x0..px_end {
            let l_here = luma(row_r[px], row_g[px], row_b[px]);
            // Horizontal: forward difference, clamp to edge.
            let px_right = if px + 1 < img_xsize { px + 1 } else { px };
            let l_right = luma(row_r[px_right], row_g[px_right], row_b[px_right]);
            // Vertical: forward difference.
            let l_below = luma(below_r[px], below_g[px], below_b[px]);

            grad_sum += (l_right - l_here).abs() + (l_below - l_here).abs();
            n += 1;
        }
    }

    if n == 0 {
        return 1;
    }
    let avg_grad = grad_sum / n as f32;

    // Empirical mapping. Tuned so most blocks stay near q=1 and only sharp
    // edges/foliage push higher; the cap at q=6 keeps the bit overhead per
    // boosted block bounded (~6× baseline) rather than 16× baseline:
    //   avg_grad <= 0.03   -> q = 1
    //   avg_grad = 0.05    -> q = 2
    //   avg_grad = 0.1     -> q = 3
    //   avg_grad = 0.2     -> q = 4
    //   avg_grad >= 0.45   -> q = 6   (saturate)
    let shifted = (avg_grad - 0.03).max(0.0);
    let q = 1.0 + 6.0 * shifted.sqrt();
    q.round().clamp(1.0, 6.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image3F;

    #[test]
    fn flat_image_gives_quant_one() {
        let mut img = Image3F::new(16, 16);
        for y in 0..16 {
            let [r, g, b] = img.all_plane_rows_mut(y);
            for x in 0..16 {
                r[x] = 0.5;
                g[x] = 0.5;
                b[x] = 0.5;
            }
        }
        let mut qf = ImageB::new_fill(2, 2, 99);
        fill_quant_field(&img, &mut qf, 0, 0);
        for by in 0..2 {
            for bx in 0..2 {
                assert_eq!(qf.row(by)[bx], 1, "flat block should be q=1");
            }
        }
    }

    #[test]
    fn high_contrast_pattern_gives_higher_quant() {
        let mut img = Image3F::new(16, 16);
        for y in 0..16 {
            let [r, g, b] = img.all_plane_rows_mut(y);
            for x in 0..16 {
                let v = if (x + y) & 1 == 0 { 0.0 } else { 1.0 };
                r[x] = v;
                g[x] = v;
                b[x] = v;
            }
        }
        let mut qf = ImageB::new_fill(2, 2, 0);
        fill_quant_field(&img, &mut qf, 0, 0);
        for by in 0..2 {
            for bx in 0..2 {
                assert!(
                    qf.row(by)[bx] > 1,
                    "checkerboard block should be q>1, got {}",
                    qf.row(by)[bx]
                );
            }
        }
    }

    #[test]
    fn quant_field_is_clamped() {
        // Strong stripe pattern: max-out the gradient.
        let mut img = Image3F::new(8, 8);
        for y in 0..8 {
            let [r, g, b] = img.all_plane_rows_mut(y);
            for x in 0..8 {
                let v = if x & 1 == 0 { 0.0 } else { 1.0 };
                r[x] = v;
                g[x] = v;
                b[x] = v;
            }
        }
        let mut qf = ImageB::new_fill(1, 1, 0);
        fill_quant_field(&img, &mut qf, 0, 0);
        let q = qf.row(0)[0];
        assert!(q >= 1 && q <= 16, "q out of range: {q}");
    }

    #[test]
    fn handles_edge_blocks() {
        // 12x12 image with 2x2 blocks-worth of qf -> the last block in each
        // direction only has 4 valid pixels.
        let mut img = Image3F::new(12, 12);
        for y in 0..12 {
            let [r, g, b] = img.all_plane_rows_mut(y);
            for x in 0..12 {
                r[x] = 0.5;
                g[x] = 0.5;
                b[x] = 0.5;
            }
        }
        let mut qf = ImageB::new_fill(2, 2, 99);
        fill_quant_field(&img, &mut qf, 0, 0);
        // Edge blocks shouldn't crash and should also be q=1.
        for by in 0..2 {
            for bx in 0..2 {
                assert_eq!(qf.row(by)[bx], 1);
            }
        }
    }
}
