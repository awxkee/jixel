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
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, ImageSB};
use crate::quant_weights::DequantMatrices;
use crate::util::FastRound;

const K_BLOCK_DIM: usize = 8;
const K_TILE_DIM_IN_BLOCKS: usize = 8;

/// libjxl default color factor: stored as 1/x in the encoder hot path.
const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;
/// Regularisation toward base correlation. Matches libjxl-tiny.
const K_DISTANCE_MULTIPLIER_AC: f32 = 1e-3;

fn solve_multiplier(ca: f32, cb: f32, num: usize, distance_mul: f32) -> i32 {
    if num == 0 {
        return 0;
    }
    let x = -cb / fmla(num as f32 * distance_mul, 0.5, ca);
    x.fast_round().clamp(-128.0, 127.0) as i32
}

/// Compute (ytox, ytob) for one tile. `tile_brect_*` are block coordinates
/// (top-left inclusive) into `opsin`, sizes capped at K_TILE_DIM_IN_BLOCKS.
fn compute_cmap_tile(
    ctx: &EncodingContext,
    opsin: &Image3F,
    bx0: usize,
    by0: usize,
    bx_count: usize,
    by_count: usize,
    matrices: &DequantMatrices,
) -> (i32, i32) {
    let qm_x = matrices.inv_matrix(0);
    let qm_b = matrices.inv_matrix(2);

    let mut block_y = [0.0f32; 64];
    let mut block_x = [0.0f32; 64];
    let mut block_b = [0.0f32; 64];
    let mut tmp = [0.0f32; 64];

    // Accumulate the two regressions' normal-equation sums directly rather than
    // materialising four per-sample coefficient vectors and re-reading them.
    // The accumulation order (block-major, coefficient-minor, ca-before-cb)
    // matches the previous push-then-iterate path exactly, and each per-sample
    // expression is unchanged, so the resulting (ytox, ytob) is bit-identical.
    let mut ca_x = 0.0f32;
    let mut cb_x = 0.0f32;
    let mut ca_b = 0.0f32;
    let mut cb_b = 0.0f32;
    let mut num = 0usize;

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
            (ctx.dct8x8)(&tmp, &mut block_y);
            // DCT X.
            for yy in 0..8 {
                let row = opsin.plane_row(0, py + yy);
                tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[px..px + 8]);
            }
            (ctx.dct8x8)(&tmp, &mut block_x);
            // DCT B.
            for yy in 0..8 {
                let row = opsin.plane_row(2, py + yy);
                tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[px..px + 8]);
            }
            (ctx.dct8x8)(&tmp, &mut block_b);

            // Zero DC (LF position) — libjxl-tiny zeros it so it doesn't affect
            // the regression; the per-tile AC factor controls AC only.
            block_y[0] = 0.0;
            block_x[0] = 0.0;
            block_b[0] = 0.0;

            for i in 0..64 {
                // YtoX regression: m = Y·qm_x, s = X·qm_x, base = 0.0.
                let m_x = block_y[i] * qm_x[i];
                let s_x = block_x[i] * qm_x[i];
                let a_x = K_INV_COLOR_FACTOR * m_x;
                let b_x = 0.0 * m_x - s_x;
                ca_x += a_x * a_x;
                cb_x += a_x * b_x;
                // YtoB regression: m = Y·qm_b, s = B·qm_b, base = 1.0.
                let m_b = block_y[i] * qm_b[i];
                let s_b = block_b[i] * qm_b[i];
                let a_b = K_INV_COLOR_FACTOR * m_b;
                let b_b = 1.0 * m_b - s_b;
                ca_b += a_b * a_b;
                cb_b += a_b * b_b;
            }
            num += 64;
        }
    }

    let ytox = solve_multiplier(ca_x, cb_x, num, K_DISTANCE_MULTIPLIER_AC);
    let ytob = solve_multiplier(ca_b, cb_b, num, K_DISTANCE_MULTIPLIER_AC);
    (ytox, ytob)
}

/// Fill `ytox_map` / `ytob_map` (sized `(xtiles, ytiles)`) by running the
/// per-tile regression on `opsin`. `(dc_group_x0_blocks, dc_group_y0_blocks)`
/// is the block offset of this DC group's (0, 0) tile into `opsin`.
pub(crate) fn fill_cmap(
    ctx: &EncodingContext,
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
        let ytox_lane = ytox_map.row_mut(ty);
        let ytob_lane = ytob_map.row_mut(ty);
        for (tx, (v_ytox, v_ytob)) in ytox_lane[..xtiles]
            .iter_mut()
            .zip(ytob_lane[..xtiles].iter_mut())
            .enumerate()
        {
            let bx0 = dc_group_x0_blocks + tx * K_TILE_DIM_IN_BLOCKS;
            let by0 = dc_group_y0_blocks + ty * K_TILE_DIM_IN_BLOCKS;
            let bx_count = K_TILE_DIM_IN_BLOCKS
                .min(dc_group_xsize_blocks.saturating_sub(tx * K_TILE_DIM_IN_BLOCKS));
            let by_count = K_TILE_DIM_IN_BLOCKS
                .min(dc_group_ysize_blocks.saturating_sub(ty * K_TILE_DIM_IN_BLOCKS));
            if bx_count == 0 || by_count == 0 {
                continue;
            }
            let (ytox, ytob) =
                compute_cmap_tile(ctx, opsin, bx0, by0, bx_count, by_count, matrices);
            *v_ytox = ytox as i8;
            *v_ytob = ytob as i8;
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
        let ctx = EncodingContext::new();
        let (ytox, ytob) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, &matrices);
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
        let ctx = EncodingContext::new();
        let (ytox, ytob) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, &matrices);
        assert!((ytox - 25).abs() < 3, "ytox = {}, expected ~25", ytox);
        // B = Y → slope 1 = base; cmap should be near 0.
        assert!(ytob.abs() < 3, "ytob = {}, expected ~0", ytob);
    }
}
