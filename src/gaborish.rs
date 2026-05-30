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
use crate::dct::fmla;
use crate::image::Image3F;

const K_GABORISH: [f32; 5] = [
    -0.09495815671340026,
    -0.041031725066768575,
    0.013710004822696948,
    0.006510206083837737,
    -0.0014789063378272242,
];

/// Apply the forward (encoder-side) gaborish filter in place. The kernel
/// sharpens slightly; the decoder will apply its complementary inverse.
///
/// Mirror-reflects pixels at image boundaries.
pub(crate) fn gaborish_inverse(image: &mut Image3F, mul: f32) {
    debug_assert!(mul >= 0.0);
    // Build weights: c at center, r/R/d/D/L at the symmetric positions.
    // We compute the un-normalized weights, then normalize so the kernel sums
    // to 1 (this matches what libjxl does — center=1, then divide everything
    // by the total).
    let w_r = mul * K_GABORISH[0];
    let w_d = mul * K_GABORISH[1];
    let w_r2 = mul * K_GABORISH[2];
    let w_l = mul * K_GABORISH[3];
    let w_d2 = mul * K_GABORISH[4];
    let mut sum = 1.0
        + mul
            * 4.
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2. * K_GABORISH[3]);
    if sum < 1e-5 {
        sum = 1e-5;
    }
    let norm = 1.0 / sum;
    let n_c = norm;
    let n_r = norm * w_r;
    let n_d = norm * w_d;
    let n_r2 = norm * w_r2;
    let n_l = norm * w_l;
    let n_d2 = norm * w_d2;

    for c in 0..3 {
        symmetric5_plane(image, c, n_c, n_r, n_d, n_r2, n_l, n_d2);
    }
}

#[inline]
fn reflect(i: isize, len: isize) -> usize {
    let mut j = i;
    // Wrap into [-len+1, 2*len-2] then reflect.
    while j < 0 || j >= len {
        if j < 0 {
            j = -j - 1;
        } else if j >= len {
            j = 2 * len - j - 1;
        }
    }
    j as usize
}

/// Symmetric 5×5 convolution on one plane, in place (via temp buffer).
/// Reads neighbors using mirror-reflected indices.
fn symmetric5_plane(
    image: &mut Image3F,
    c: usize,
    n_c: f32,
    n_r: f32,
    n_d: f32,
    n_r2: f32,
    n_l: f32,
    n_d2: f32,
) {
    let xsize = image.xsize();
    let ysize = image.ysize();

    // Allocate full output buffer. We read from `image` and write into `out`,
    // then swap.
    let mut out: Vec<f32> = vec![0.0; xsize * ysize];

    let plane = image.plane(c);

    for y in 0..ysize {
        let iy = y as isize;
        // Precompute row indices for ±1, ±2.
        let y_m1 = reflect(iy - 1, ysize as isize);
        let y_p1 = reflect(iy + 1, ysize as isize);
        let y_m2 = reflect(iy - 2, ysize as isize);
        let y_p2 = reflect(iy + 2, ysize as isize);

        let row_0 = plane.row(y);
        let row_m1 = plane.row(y_m1);
        let row_p1 = plane.row(y_p1);
        let row_m2 = plane.row(y_m2);
        let row_p2 = plane.row(y_p2);

        let dst_row = &mut out[y * xsize..(y + 1) * xsize];
        assert!(dst_row.len() >= xsize);

        let border = 2usize;
        let center_start = border.min(xsize);
        let center_end = xsize.saturating_sub(border);

        for x in 0..center_start {
            let ix = x as isize;
            let x_m1 = reflect(ix - 1, xsize as isize);
            let x_p1 = reflect(ix + 1, xsize as isize);
            let x_m2 = reflect(ix - 2, xsize as isize);
            let x_p2 = reflect(ix + 2, xsize as isize);

            // Cardinal neighbors at distance 1 (4 of them): r
            let s_r = row_m1[x] + row_p1[x] + row_0[x_m1] + row_0[x_p1];
            // Diagonal neighbors at distance √2 (4 of them): d
            let s_d = row_m1[x_m1] + row_m1[x_p1] + row_p1[x_m1] + row_p1[x_p1];
            // Cardinal neighbors at distance 2 (4 of them): R
            let s_r2 = row_m2[x] + row_p2[x] + row_0[x_m2] + row_0[x_p2];
            // Diagonal neighbors at distance 2√2 (4 of them): D
            let s_d2 = row_m2[x_m2] + row_m2[x_p2] + row_p2[x_m2] + row_p2[x_p2];
            // Knight neighbors at distance √5 (8 of them): L
            let s_l = row_m2[x_m1]
                + row_m2[x_p1]
                + row_m1[x_m2]
                + row_m1[x_p2]
                + row_p1[x_m2]
                + row_p1[x_p2]
                + row_p2[x_m1]
                + row_p2[x_p1];

            let mut acc = n_c * row_0[x];
            acc = fmla(n_r, s_r, acc);
            acc = fmla(n_d, s_d, acc);
            acc = fmla(n_r2, s_r2, acc);
            acc = fmla(n_d2, s_d2, acc);
            acc = fmla(n_l, s_l, acc);
            dst_row[x] = acc;
        }

        if center_start < center_end {
            assert!(center_start >= 2);
            assert!(row_0.len() >= center_end + 2);
            assert!(row_m1.len() >= center_end + 2);
            assert!(row_p1.len() >= center_end + 2);
            assert!(row_m2.len() >= center_end + 2);
            assert!(row_p2.len() >= center_end + 2);
            // dst_row covers at least [0, center_end)
            assert!(dst_row.len() >= center_end);

            for x in center_start..center_end {
                let ix = x as isize;
                let x_m1 = (ix - 1) as usize;
                let x_p1 = (ix + 1) as usize;
                let x_m2 = (ix - 2) as usize;
                let x_p2 = (ix + 2) as usize;

                // Cardinal neighbors at distance 1 (4 of them): r
                let s_r = row_m1[x] + row_p1[x] + row_0[x_m1] + row_0[x_p1];
                // Diagonal neighbors at distance √2 (4 of them): d
                let s_d = row_m1[x_m1] + row_m1[x_p1] + row_p1[x_m1] + row_p1[x_p1];
                // Cardinal neighbors at distance 2 (4 of them): R
                let s_r2 = row_m2[x] + row_p2[x] + row_0[x_m2] + row_0[x_p2];
                // Diagonal neighbors at distance 2√2 (4 of them): D
                let s_d2 = row_m2[x_m2] + row_m2[x_p2] + row_p2[x_m2] + row_p2[x_p2];
                // Knight neighbors at distance √5 (8 of them): L
                let s_l = row_m2[x_m1]
                    + row_m2[x_p1]
                    + row_m1[x_m2]
                    + row_m1[x_p2]
                    + row_p1[x_m2]
                    + row_p1[x_p2]
                    + row_p2[x_m1]
                    + row_p2[x_p1];

                let mut acc = n_c * row_0[x];
                acc = fmla(n_r, s_r, acc);
                acc = fmla(n_d, s_d, acc);
                acc = fmla(n_r2, s_r2, acc);
                acc = fmla(n_d2, s_d2, acc);
                acc = fmla(n_l, s_l, acc);
                dst_row[x] = acc;
            }
        }

        for x in center_end..xsize {
            let ix = x as isize;
            let x_m1 = reflect(ix - 1, xsize as isize);
            let x_p1 = reflect(ix + 1, xsize as isize);
            let x_m2 = reflect(ix - 2, xsize as isize);
            let x_p2 = reflect(ix + 2, xsize as isize);

            // Cardinal neighbors at distance 1 (4 of them): r
            let s_r = row_m1[x] + row_p1[x] + row_0[x_m1] + row_0[x_p1];
            // Diagonal neighbors at distance √2 (4 of them): d
            let s_d = row_m1[x_m1] + row_m1[x_p1] + row_p1[x_m1] + row_p1[x_p1];
            // Cardinal neighbors at distance 2 (4 of them): R
            let s_r2 = row_m2[x] + row_p2[x] + row_0[x_m2] + row_0[x_p2];
            // Diagonal neighbors at distance 2√2 (4 of them): D
            let s_d2 = row_m2[x_m2] + row_m2[x_p2] + row_p2[x_m2] + row_p2[x_p2];
            // Knight neighbors at distance √5 (8 of them): L
            let s_l = row_m2[x_m1]
                + row_m2[x_p1]
                + row_m1[x_m2]
                + row_m1[x_p2]
                + row_p1[x_m2]
                + row_p1[x_p2]
                + row_p2[x_m1]
                + row_p2[x_p1];

            let mut acc = n_c * row_0[x];
            acc = fmla(n_r, s_r, acc);
            acc = fmla(n_d, s_d, acc);
            acc = fmla(n_r2, s_r2, acc);
            acc = fmla(n_d2, s_d2, acc);
            acc = fmla(n_l, s_l, acc);
            dst_row[x] = acc;
        }
    }

    // Copy back into image.
    for y in 0..ysize {
        let dst_row = image.plane_row_mut(c, y);
        dst_row.copy_from_slice(&out[y * xsize..(y + 1) * xsize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image3F;

    #[test]
    fn constant_input_unchanged() {
        let mut img = Image3F::new(16, 16);
        for c in 0..3 {
            for y in 0..16 {
                let row = img.plane_row_mut(c, y);
                for x in 0..16 {
                    row[x] = 0.5;
                }
            }
        }
        gaborish_inverse(&mut img, 0.9908511);
        for c in 0..3 {
            for y in 0..16 {
                let row = img.plane_row(c, y);
                for x in 0..16 {
                    assert!(
                        (row[x] - 0.5).abs() < 1e-5,
                        "c={} y={} x={}: got {} want 0.5",
                        c,
                        y,
                        x,
                        row[x]
                    );
                }
            }
        }
    }

    #[test]
    fn sharpens_step_edge() {
        // the sharpening kernel.
        let mut img = Image3F::new(16, 16);
        for c in 0..3 {
            for y in 0..16 {
                let row = img.plane_row_mut(c, y);
                for x in 0..16 {
                    row[x] = if x < 8 { 0.0 } else { 1.0 };
                }
            }
        }
        let orig_at_8 = 1.0_f32;
        let orig_at_7 = 0.0_f32;
        gaborish_inverse(&mut img, 0.9908511);
        // Sharpening means: value just *after* the edge (at x=8) is slightly
        // larger than 1.0, value just *before* the edge (at x=7) slightly
        // negative.
        let new_at_8 = img.plane_row(0, 8)[8];
        let new_at_7 = img.plane_row(0, 8)[7];
        assert!(
            new_at_8 > orig_at_8 - 0.001,
            "value at edge+ should not collapse: {}",
            new_at_8
        );
        assert!(
            new_at_7 < orig_at_7 + 0.001,
            "value at edge- should not bleed: {}",
            new_at_7
        );
    }
}
