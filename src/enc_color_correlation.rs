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

use crate::dct::{DctInput, fmla};
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, ImageSB};
use crate::util::FastRound;

const K_BLOCK_DIM: usize = 8;
const K_TILE_DIM_IN_BLOCKS: usize = 8;

/// libjxl default color factor: stored as 1/x in the encoder hot path.
const K_INV_COLOR_FACTOR: f32 = 1.0 / 84.0;
/// Regularisation toward base correlation. Matches libjxl-tiny.
const K_DISTANCE_MULTIPLIER_AC: f32 = 1e-3;

pub(crate) type CflRegressionFn =
    fn(&[f32; 64], &[f32; 64], &[f32; 64], &[f32; 64], &[f32; 64]) -> [f32; 4];

#[allow(dead_code)]
pub(crate) fn cfl_regression_scalar(
    block_y: &[f32; 64],
    block_x: &[f32; 64],
    block_b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
) -> [f32; 4] {
    let mut sums = [0.0f32; 4];
    for i in 0..64 {
        let m_x = block_y[i] * qm_x[i];
        let s_x = block_x[i] * qm_x[i];
        let a_x = K_INV_COLOR_FACTOR * m_x;
        let b_x = -s_x;
        sums[0] = fmla(a_x, a_x, sums[0]);
        sums[1] = fmla(a_x, b_x, sums[1]);

        let m_b = block_y[i] * qm_b[i];
        let s_b = block_b[i] * qm_b[i];
        let a_b = K_INV_COLOR_FACTOR * m_b;
        let b_b = fmla(1.0, m_b, -s_b);
        sums[2] = fmla(a_b, a_b, sums[2]);
        sums[3] = fmla(a_b, b_b, sums[3]);
    }
    sums
}

pub(crate) fn selected_cfl_regression_fn() -> CflRegressionFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |y, x, b, qm_x, qm_b| unsafe { crate::neon::cfl_regression_neon(y, x, b, qm_x, qm_b) }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return |y, x, b, qm_x, qm_b| unsafe {
            crate::avx::cfl_regression_avx2(y, x, b, qm_x, qm_b)
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::cfl_regression_wasm;
    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
        all(target_arch = "aarch64", feature = "neon")
    )))]
    cfl_regression_scalar
}

const CFL_DEADZONE_AMOUNT: f32 = 1.5;
const CFL_DEADZONE_LO: f32 = 1.0;
const CFL_DEADZONE_HI: f32 = 2.0;

#[inline]
fn cfl_deadzone(distance: f32) -> f32 {
    if distance <= CFL_DEADZONE_LO {
        return CFL_DEADZONE_AMOUNT;
    }
    if distance >= CFL_DEADZONE_HI {
        return 0.0;
    }
    const CFL_DEADZONE_SCALE: f32 = CFL_DEADZONE_AMOUNT / (CFL_DEADZONE_HI - CFL_DEADZONE_LO);
    CFL_DEADZONE_SCALE * (CFL_DEADZONE_HI - distance)
}

fn solve_multiplier(ca: f32, cb: f32, num: usize, distance_mul: f32, dz: f32) -> i32 {
    if num == 0 {
        return 0;
    }
    let mut x = -cb / fmla(num as f32 * distance_mul, 0.5, ca);
    // libjxl `towards_zero` deadzone: shrink toward the base correlation and
    // snap sub-threshold slopes to it. No-op when `dz` is 0.
    if x >= dz {
        x -= dz;
    } else if x <= -dz {
        x += dz;
    } else {
        x = 0.0;
    }
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
    distance: f32,
) -> (i32, i32) {
    let matrices = &ctx.matrices;
    let qm_x: &[f32; 64] = matrices.inv_matrix(0)[..64].try_into().unwrap();
    let qm_b: &[f32; 64] = matrices.inv_matrix(2)[..64].try_into().unwrap();

    let mut block_y = [0.0f32; 64];
    let mut block_x = [0.0f32; 64];
    let mut block_b = [0.0f32; 64];

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
            let stride = opsin.xsize();
            let offset = py * stride + px;
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(1)[offset..], stride),
                &mut block_y,
            );
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(0)[offset..], stride),
                &mut block_x,
            );
            (ctx.dct8x8)(
                DctInput::new(&opsin.plane_data(2)[offset..], stride),
                &mut block_b,
            );

            // Zero DC (LF position) — libjxl-tiny zeros it so it doesn't affect
            // the regression; the per-tile AC factor controls AC only.
            block_y[0] = 0.0;
            block_x[0] = 0.0;
            block_b[0] = 0.0;

            let sums = (ctx.cfl_regression)(&block_y, &block_x, &block_b, qm_x, qm_b);
            ca_x += sums[0];
            cb_x += sums[1];
            ca_b += sums[2];
            cb_b += sums[3];
            num += 64;
        }
    }

    let dz = cfl_deadzone(distance);
    let ytox = solve_multiplier(ca_x, cb_x, num, K_DISTANCE_MULTIPLIER_AC, dz);
    let ytob = solve_multiplier(ca_b, cb_b, num, K_DISTANCE_MULTIPLIER_AC, dz);
    (ytox, ytob)
}

/// Fill `ytox_map` / `ytob_map` (sized `(xtiles, ytiles)`) by running the
/// per-tile regression on `opsin`. `(dc_group_x0_blocks, dc_group_y0_blocks)`
/// is the block offset of this DC group's (0, 0) tile into `opsin`.
pub(crate) fn fill_cmap(
    ctx: &EncodingContext,
    opsin: &Image3F,
    dc_group_x0_blocks: usize,
    dc_group_y0_blocks: usize,
    dc_group_xsize_blocks: usize,
    dc_group_ysize_blocks: usize,
    ytox_map: &mut ImageSB,
    ytob_map: &mut ImageSB,
    distance: f32,
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
                compute_cmap_tile(ctx, opsin, bx0, by0, bx_count, by_count, distance);
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
    fmla(cmap_b as f32, K_INV_COLOR_FACTOR, 1.0)
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
        let ctx = EncodingContext::default();
        let (ytox, ytob) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0);
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
        let ctx = EncodingContext::default();
        let (ytox, ytob) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0);
        assert!((ytox - 25).abs() < 3, "ytox = {}, expected ~25", ytox);
        // B = Y → slope 1 = base; cmap should be near 0.
        assert!(ytob.abs() < 3, "ytob = {}, expected ~0", ytob);
    }

    #[test]
    fn selected_regression_matches_scalar() {
        let mut y = [0.0f32; 64];
        let mut x = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        let mut qm_x = [0.0f32; 64];
        let mut qm_b = [0.0f32; 64];
        for i in 0..64 {
            y[i] = ((i * 37 % 97) as f32 - 48.0) * 0.03125;
            x[i] = ((i * 19 % 89) as f32 - 44.0) * 0.046875;
            b[i] = ((i * 53 % 101) as f32 - 50.0) * 0.0234375;
            qm_x[i] = 0.5 + (i % 11) as f32 * 0.0625;
            qm_b[i] = 0.75 + (i % 7) as f32 * 0.09375;
        }

        let expected = cfl_regression_scalar(&y, &x, &b, &qm_x, &qm_b);
        let actual = selected_cfl_regression_fn()(&y, &x, &b, &qm_x, &qm_b);
        for i in 0..4 {
            let tolerance = 1e-5 * expected[i].abs().max(1.0);
            assert!(
                (actual[i] - expected[i]).abs() <= tolerance,
                "sum {i}: SIMD={}, scalar={}, tolerance={tolerance}",
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn deadzone_schedule_ramps_from_full_to_zero() {
        // Full amount at/below LO, zero at/above HI, linear in between, and off
        // (byte-identical) at low quality.
        assert_eq!(cfl_deadzone(0.5), CFL_DEADZONE_AMOUNT);
        assert_eq!(cfl_deadzone(CFL_DEADZONE_LO), CFL_DEADZONE_AMOUNT);
        assert_eq!(cfl_deadzone(1.5), CFL_DEADZONE_AMOUNT * 0.5); // midpoint of 1.0..2.0
        assert_eq!(cfl_deadzone(CFL_DEADZONE_HI), 0.0);
        assert_eq!(cfl_deadzone(3.0), 0.0);
    }

    #[test]
    fn deadzone_shrinks_slope_at_high_quality_only() {
        // Perfectly correlated X = 0.3*Y (ytox ≈ 25). At HQ the deadzone shrinks
        // it toward base; at d ≥ HI it is left untouched.
        let mut opsin = Image3F::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x ^ y) as f32) / 64.0;
                opsin.plane_row_mut(1, y)[x] = v;
                opsin.plane_row_mut(0, y)[x] = 0.3 * v;
                opsin.plane_row_mut(2, y)[x] = v;
            }
        }
        let ctx = EncodingContext::default();
        let (ytox_hq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 0.5);
        let (ytox_lq, _) = compute_cmap_tile(&ctx, &opsin, 0, 0, 8, 8, 2.0);
        assert!(
            ytox_hq < ytox_lq,
            "deadzone should shrink the HQ slope: hq={ytox_hq} lq={ytox_lq}"
        );
    }
}
