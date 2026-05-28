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

use crate::dct::dct8x8;
use crate::image::{Image3F, ImageSB};
use crate::quant_weights::DequantMatrices;

const K_BLOCK_DIM: usize = 8;
const K_TILE_DIM_IN_BLOCKS: usize = 8;

/// libjxl default color factor: stored as 1/x in the encoder hot path.
const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;
/// Regularisation toward base correlation. Matches libjxl-tiny.
const K_DISTANCE_MULTIPLIER_AC: f32 = 1e-3;

/// Solve the regularised 1D least squares:
///     minimise sum_i ((a_i · x + b_i)^2) + distance_mul · x^2 · num
/// where a_i = K_INV_COLOR_FACTOR · values_m[i],
///       b_i = base · values_m[i] − values_s[i].
///
/// Returns the optimal slope quantised to int32 in [-128, 127], suitable for
/// storage as i8 in the YtoX/YtoB map.
fn find_best_multiplier(values_m: &[f32], values_s: &[f32], base: f32, distance_mul: f32) -> i32 {
    let num = values_m.len();
    if num == 0 {
        return 0;
    }
    let mut ca = 0.0f32;
    let mut cb = 0.0f32;
    for i in 0..num {
        let a = K_INV_COLOR_FACTOR * values_m[i];
        let b = base * values_m[i] - values_s[i];
        ca += a * a;
        cb += a * b;
    }
    let x = -cb / (ca + num as f32 * distance_mul * 0.5);
    x.round().clamp(-128.0, 127.0) as i32
}

/// Compute (ytox, ytob) for one tile. `tile_brect_*` are block coordinates
/// (top-left inclusive) into `opsin`, sizes capped at K_TILE_DIM_IN_BLOCKS.
fn compute_cmap_tile(
    opsin: &Image3F,
    bx0: usize,
    by0: usize,
    bx_count: usize,
    by_count: usize,
    matrices: &DequantMatrices,
) -> (i32, i32) {
    let mut coeffs_yx: Vec<f32> = Vec::with_capacity(bx_count * by_count * 64);
    let mut coeffs_x: Vec<f32> = Vec::with_capacity(bx_count * by_count * 64);
    let mut coeffs_yb: Vec<f32> = Vec::with_capacity(bx_count * by_count * 64);
    let mut coeffs_b: Vec<f32> = Vec::with_capacity(bx_count * by_count * 64);

    let qm_x = matrices.inv_matrix(0);
    let qm_b = matrices.inv_matrix(2);

    let mut block_y = [0.0f32; 64];
    let mut block_x = [0.0f32; 64];
    let mut block_b = [0.0f32; 64];
    let mut tmp = [0.0f32; 64];

    for by in 0..by_count {
        for bx in 0..bx_count {
            let px = (bx0 + bx) * K_BLOCK_DIM;
            let py = (by0 + by) * K_BLOCK_DIM;
            // Bounds check: skip if outside opsin (last edge tile may be small).
            if px + K_BLOCK_DIM > opsin.xsize() || py + K_BLOCK_DIM > opsin.ysize() {
                continue;
            }
            // DCT Y.
            for yy in 0..8 {
                let row = opsin.plane_row(1, py + yy);
                tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[px..px + 8]);
            }
            dct8x8(&tmp, &mut block_y);
            // DCT X.
            for yy in 0..8 {
                let row = opsin.plane_row(0, py + yy);
                tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[px..px + 8]);
            }
            dct8x8(&tmp, &mut block_x);
            // DCT B.
            for yy in 0..8 {
                let row = opsin.plane_row(2, py + yy);
                tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[px..px + 8]);
            }
            dct8x8(&tmp, &mut block_b);

            // Zero DC (LF position) — libjxl-tiny zeros it so it doesn't affect
            // the regression; the per-tile AC factor controls AC only.
            block_y[0] = 0.0;
            block_x[0] = 0.0;
            block_b[0] = 0.0;

            // Weight by inverse quant matrix, store as sample sequence.
            for i in 0..64 {
                coeffs_yx.push(block_y[i] * qm_x[i]);
                coeffs_x.push(block_x[i] * qm_x[i]);
                coeffs_yb.push(block_y[i] * qm_b[i]);
                coeffs_b.push(block_b[i] * qm_b[i]);
            }
        }
    }

    let ytox = find_best_multiplier(&coeffs_yx, &coeffs_x, 0.0, K_DISTANCE_MULTIPLIER_AC);
    let ytob = find_best_multiplier(&coeffs_yb, &coeffs_b, 1.0, K_DISTANCE_MULTIPLIER_AC);
    (ytox, ytob)
}

/// Fill `ytox_map` / `ytob_map` (sized `(xtiles, ytiles)`) by running the
/// per-tile regression on `opsin`. `(dc_group_x0_blocks, dc_group_y0_blocks)`
/// is the block offset of this DC group's (0, 0) tile into `opsin`.
pub(crate) fn fill_cmap(
    opsin: &Image3F,
    matrices: &DequantMatrices,
    dc_group_x0_blocks: usize,
    dc_group_y0_blocks: usize,
    dc_group_xsize_blocks: usize,
    dc_group_ysize_blocks: usize,
    ytox_map: &mut ImageSB,
    ytob_map: &mut ImageSB,
) {
    let xtiles = ytox_map.xsize();
    let ytiles = ytox_map.ysize();
    for ty in 0..ytiles {
        for tx in 0..xtiles {
            let bx0 = dc_group_x0_blocks + tx * K_TILE_DIM_IN_BLOCKS;
            let by0 = dc_group_y0_blocks + ty * K_TILE_DIM_IN_BLOCKS;
            let bx_count = K_TILE_DIM_IN_BLOCKS
                .min(dc_group_xsize_blocks.saturating_sub(tx * K_TILE_DIM_IN_BLOCKS));
            let by_count = K_TILE_DIM_IN_BLOCKS
                .min(dc_group_ysize_blocks.saturating_sub(ty * K_TILE_DIM_IN_BLOCKS));
            if bx_count == 0 || by_count == 0 {
                continue;
            }
            let (ytox, ytob) = compute_cmap_tile(opsin, bx0, by0, bx_count, by_count, matrices);
            ytox_map.row_mut(ty)[tx] = ytox as i8;
            ytob_map.row_mut(ty)[tx] = ytob as i8;
        }
    }
}

/// Returns the per-tile factor as a slope (= base_correlation + cmap/84).
#[inline]
pub(crate) fn y_to_x_ratio(cmap_x: i8) -> f32 {
    // base_correlation_x = 0
    cmap_x as f32 * K_INV_COLOR_FACTOR
}

#[inline]
pub(crate) fn y_to_b_ratio(cmap_b: i8) -> f32 {
    // base_correlation_b = 1
    1.0 + cmap_b as f32 * K_INV_COLOR_FACTOR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_returns_zero() {
        let mut opsin = Image3F::new(64, 64);
        for c in 0..3 {
            for y in 0..64 {
                let row = opsin.plane_row_mut(c, y);
                for x in 0..64 {
                    row[x] = 0.5;
                }
            }
        }
        let matrices = DequantMatrices::new();
        let (ytox, ytob) = compute_cmap_tile(&opsin, 0, 0, 8, 8, &matrices);
        // No variation → no useful correlation → slope == 0 (the regression
        // collapses to numerator=0, denominator>0 → 0).
        assert_eq!(ytox, 0);
        assert_eq!(ytob, 0);
    }

    #[test]
    fn perfectly_correlated_chroma_finds_nonzero_slope() {
        // X = 0.3 * Y per pixel (within float roundoff). The regression
        // should land on roughly ytox = 0.3 * 84 ≈ 25.
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x ^ y) as f32) / 64.0; // varied, non-flat Y
                opsin.plane_row_mut(1, y)[x] = v; // Y
                opsin.plane_row_mut(0, y)[x] = 0.3 * v; // X = 0.3 Y
                opsin.plane_row_mut(2, y)[x] = v; // B = Y (no extra slope)
            }
        }
        let matrices = DequantMatrices::new();
        let (ytox, ytob) = compute_cmap_tile(&opsin, 0, 0, 8, 8, &matrices);
        assert!((ytox - 25).abs() < 3, "ytox = {}, expected ~25", ytox);
        // B = Y → slope 1 = base; cmap should be near 0.
        assert!(ytob.abs() < 3, "ytob = {}, expected ~0", ytob);
    }
}
