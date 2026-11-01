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

//! Near-lossless pre-pass: bounded-error value quantization.
//!
//! Every color sample moves to the nearest multiple of `2k + 1`, which is
//! never farther than `k` code values away. The grid is the same for every
//! pixel, so equal colors stay equal: palettes shrink, exact repeats survive
//! for LZ77 and patches, and the subtract-type RCTs keep the residuals on the
//! grid, where they cost `log2(2k + 1)` bits less each. The pass is applied
//! to the RGB planes before the YCoCg conversion; alpha stays exact.

use crate::image::Image3Si;

/// Quantize the color planes of `image` (values in `0..=max_value`) in place
/// so that every sample stays within `±max_error` of the input. A gray image
/// carries its samples in the first plane and empty planes elsewhere.
pub(crate) fn quantize_planes(image: &mut Image3Si, max_value: i32, max_error: i32) {
    if max_error <= 0 {
        return;
    }
    // Samples are non-negative and the step is odd, so the nearest multiple
    // is the rounded quotient (ties cannot occur) scaled back.
    let step = 2 * max_error + 1;
    let quotients = crate::fast_divide::selected_rounded_quotients_fn();
    for c in 0..3 {
        let plane = image.plane_mut(c).as_mut_slice();
        unsafe { quotients(plane, step) };
        for v in plane {
            *v = (*v * step).min(max_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_the_floor_formula_for_every_16_bit_value() {
        let (w, h) = (256, 256);
        for k in 1..=9 {
            let step = 2 * k + 1;
            let mut image = Image3Si::new(w, h);
            for c in 0..3 {
                for (i, v) in image.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                    *v = (i as i32).min(65535);
                }
            }
            let source = image.plane_mut(0).as_mut_slice().to_vec();
            quantize_planes(&mut image, 65535, k);
            for c in 0..3 {
                for (&got, &v) in image.plane_mut(c).as_mut_slice().iter().zip(&source) {
                    assert_eq!(got, ((v + k) / step * step).min(65535), "k={k} v={v}");
                }
            }
        }
    }

    #[test]
    fn quantization_stays_within_the_bound_on_a_shared_grid() {
        let (w, h) = (37, 11);
        let mut rgb = Image3Si::new(w, h);
        let mut seed = 12345u32;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    rgb.plane_row_mut(c, y)[x] =
                        ((x * 7 + y * 3) as i32 + (seed >> 24) as i32).min(255);
                }
            }
        }
        rgb.plane_row_mut(0, 0)[0] = 255;
        rgb.plane_row_mut(1, 0)[0] = 0;
        for k in [1, 2, 4, 7] {
            let mut quantized = rgb.clone();
            quantize_planes(&mut quantized, 255, k);
            let step = 2 * k + 1;
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        let a = rgb.plane_row(c, y)[x];
                        let b = quantized.plane_row(c, y)[x];
                        assert!((a - b).abs() <= k, "k={k} c={c} ({x},{y}): {a} -> {b}");
                        assert!((0..=255).contains(&b));
                        assert!(b % step == 0 || b == 255, "k={k}: {b} is off the grid");
                    }
                }
            }
        }
        // Equal inputs map to equal outputs: the pass never splits a color.
        let mut gray = rgb.clone();
        for y in 0..h {
            let [r, g, b] = gray.all_plane_rows_mut(y);
            g.copy_from_slice(r);
            b.copy_from_slice(r);
        }
        quantize_planes(&mut gray, 255, 2);
        for y in 0..h {
            assert_eq!(gray.plane_row(0, y), gray.plane_row(1, y));
            assert_eq!(gray.plane_row(0, y), gray.plane_row(2, y));
        }
        let mut exact = rgb.clone();
        quantize_planes(&mut exact, 255, 0);
        for c in 0..3 {
            assert_eq!(exact.plane_data(c), rgb.plane_data(c));
        }
    }

    #[test]
    fn wide_samples_use_the_declared_range() {
        let mut rgb = Image3Si::new(4, 1);
        rgb.plane_row_mut(0, 0)
            .copy_from_slice(&[65535, 65534, 1023, 1022]);
        let mut wide = rgb.clone();
        quantize_planes(&mut wide, 65535, 2);
        assert_eq!(wide.plane_row(0, 0), &[65535, 65535, 1025, 1020]);
        let mut ten_bit = rgb.clone();
        ten_bit
            .plane_row_mut(0, 0)
            .copy_from_slice(&[1023, 1022, 1021, 1020]);
        quantize_planes(&mut ten_bit, 1023, 2);
        assert_eq!(ten_bit.plane_row(0, 0), &[1023, 1020, 1020, 1020]);
    }
}
