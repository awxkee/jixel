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
use crate::dc_group_data::{
    AcStrategyImage, STRATEGY_DCT, STRATEGY_DCT8X16, STRATEGY_DCT16X8, STRATEGY_DCT16X16,
};
use crate::dct::{dct8x8, dct8x16, dct16x8, dct16x16};
use crate::image::Image3F;
use crate::quant_weights::DequantMatrices;

/// Estimate per-block coding cost for a candidate strategy at super-block
/// position (bx, by) with offset (cx, cy) inside the super-block. Mirrors
/// libjxl-tiny's `EstimateEntropy`.
///
/// `opsin` is the full XYB plane; `bx`, `by` are block coordinates of the
/// super-block top-left in the full image; `cx`, `cy` are offsets in 0..2.
/// Returns the entropy estimate plus an information-loss penalty.
fn estimate_entropy(
    raw_strategy: u8,
    opsin: &Image3F,
    bx: usize,
    by: usize,
    cx: usize,
    cy: usize,
    distance: f32,
    matrices: &DequantMatrices,
    quant_per_block: &[u8], // raw_quant_field over the super-block (2x2)
    qf_stride: usize,
) -> f32 {
    let cov_x = AcStrategyImage::covered_blocks_x_of(raw_strategy);
    let cov_y = AcStrategyImage::covered_blocks_y_of(raw_strategy);
    let num_blocks = cov_x * cov_y;
    let size = num_blocks * 64;
    let bx_pix = (bx + cx) * 8;
    let by_pix = (by + cy) * 8;

    // Bounds check — return huge cost if candidate would go off the image.
    if by_pix + 8 * cov_y > opsin.ysize() || bx_pix + 8 * cov_x > opsin.xsize() {
        return f32::INFINITY;
    }

    // Forward DCT each channel into block[c * size .. (c+1) * size].
    // Buffer sized for the largest candidate (DCT16X16 = 256 floats per channel).
    let mut block = [0.0f32; 3 * 256];
    for c in 0..3 {
        let plane = opsin.plane(c);
        match raw_strategy {
            STRATEGY_DCT => {
                let mut tmp = [0.0f32; 64];
                for yy in 0..8 {
                    let row = plane.row(by_pix + yy);
                    tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[bx_pix..bx_pix + 8]);
                }
                let dst: &mut [f32; 64] = (&mut block[c * size..c * size + 64]).try_into().unwrap();
                dct8x8(&tmp, dst);
            }
            STRATEGY_DCT16X8 => {
                let mut tmp = [0.0f32; 128];
                for yy in 0..16 {
                    let row = plane.row(by_pix + yy);
                    tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[bx_pix..bx_pix + 8]);
                }
                let dst: &mut [f32; 128] =
                    (&mut block[c * size..c * size + 128]).try_into().unwrap();
                dct16x8(&tmp, dst);
            }
            STRATEGY_DCT8X16 => {
                let mut tmp = [0.0f32; 128];
                for yy in 0..8 {
                    let row = plane.row(by_pix + yy);
                    tmp[yy * 16..yy * 16 + 16].copy_from_slice(&row[bx_pix..bx_pix + 16]);
                }
                let dst: &mut [f32; 128] =
                    (&mut block[c * size..c * size + 128]).try_into().unwrap();
                dct8x16(&tmp, dst);
            }
            STRATEGY_DCT16X16 => {
                let mut tmp = [0.0f32; 256];
                for yy in 0..16 {
                    let row = plane.row(by_pix + yy);
                    tmp[yy * 16..yy * 16 + 16].copy_from_slice(&row[bx_pix..bx_pix + 16]);
                }
                let dst: &mut [f32; 256] =
                    (&mut block[c * size..c * size + 256]).try_into().unwrap();
                dct16x16(&tmp, dst);
            }
            _ => unreachable!(),
        }
    }

    // Find max quant within candidate footprint (libjxl-tiny uses qf field).
    // We approximate using the raw quant field (the post-AQ "quant" field).
    let mut max_quant: u8 = 1;
    for iy in 0..cov_y {
        for ix in 0..cov_x {
            let q = quant_per_block[(cy + iy) * qf_stride + (cx + ix)];
            if q > max_quant {
                max_quant = q;
            }
        }
    }
    let quant = max_quant as f32;
    // libjxl-tiny normally derives masking from a per-block mask field. jixel
    // doesn't compute one; the K_MUL16X8_TUNING constant below compensates by
    // making multi-block transforms uniformly more expensive in cost space.
    let masking = 1.0f32;

    // libjxl-tiny constants for entropy estimation
    const K_INFO_LOSS_MULTIPLIER: f32 = 138.0;
    const K_INFO_LOSS_MULTIPLIER2: f32 = 50.468_4;
    const K_COST_DELTA: f32 = 5.335_918_5;
    const K_COST2: f32 = 4.462_815;
    const K_ZEROS_MUL: f32 = 7.565_053;

    let slope = (distance * (1.0 / 3.0)).min(1.0);
    let cost1 = 1.0 + slope * 8.870_325;

    // jixel: chroma-from-luma factors. X has no CfL (factor=0), B uses 1.0.
    let cmap_factors = [0.0f32, 0.0, 1.0];

    let mut entropy = 0.0f32;
    let mut info_loss = 0.0f32;
    let mut info_loss2 = 0.0f32;

    for c in 0..3 {
        let inv_matrix: &[f32] = match raw_strategy {
            STRATEGY_DCT => &matrices.inv_matrix(c)[..],
            STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(c)[..],
            _ => &matrices.inv_matrix_16x8(c)[..],
        };
        let cmap_factor = cmap_factors[c];
        let mut entropy_acc = 0.0f32;
        let mut nzeros = 0usize;
        for i in 0..size {
            let in_x = block[c * size + i];
            let in_y = block[size + i]; // channel 1 (Y)
            let im = inv_matrix[i];
            let val = (in_x - cmap_factor * in_y) * im * quant;
            let rval = val.round();
            let diff = (val - rval).abs();
            info_loss += diff;
            info_loss2 += diff * diff;
            let q = rval.abs();
            if q >= 1.5 {
                entropy_acc += K_COST2;
            }
            entropy_acc += q.sqrt() * K_COST_DELTA;
            if q > 0.0 {
                nzeros += 1;
            }
        }
        entropy_acc += nzeros as f32 * cost1;
        entropy += entropy_acc;
        // Bits-to-encode num_nonzeros estimate.
        let nbits = if nzeros + 1 > 1 {
            (32 - (nzeros as u32 + 1).leading_zeros()) as usize
        } else {
            1
        };
        let nbits = nbits.max(1);
        let log_nb = if nbits + 17 > 1 {
            (32 - (nbits as u32 + 17).leading_zeros()) as usize
        } else {
            1
        };
        let log_nb = log_nb.max(1);
        entropy += K_ZEROS_MUL * (log_nb as f32 + nbits as f32);
    }

    let infoloss_score = K_INFO_LOSS_MULTIPLIER * info_loss
        + K_INFO_LOSS_MULTIPLIER2 * (num_blocks as f32 * info_loss2).sqrt();
    entropy + masking * infoloss_score
}

/// Find the best transform layout for each aligned 2×2 super-block in the
/// full image. Decides between (4 separate DCT8) vs (two DCT16X8 / two
/// DCT8X16 — pick best orientation), and sets `ac_strategy` accordingly.
///
/// `opsin` is the full image; `bx0`, `by0` is the super-block top-left in
/// the full image (must be even-aligned). `quant_per_block` is the
/// raw_quant_field over the full image; `qf_stride` is its row stride.
pub fn find_best_16x16_transform(
    opsin: &Image3F,
    bx0: usize,
    by0: usize,
    distance: f32,
    matrices: &DequantMatrices,
    quant_per_block: &[u8],
    qf_stride: usize,
    ac_strategy: &mut AcStrategyImage,
) {
    // Per libjxl-tiny: per-distance multipliers.
    let k8x8_base = 1.4;
    let k8x8_mul1 = -0.55 * 0.75;
    let k8x8_mul2 = 1.073_575_8 * 0.75;
    let mul8x8 = k8x8_mul2 + k8x8_mul1 / (distance + k8x8_base);

    let acs_bias = 1.0 + 0.5 * ((0.7 - distance) / 0.7).clamp(0.0, 1.0);
    let k_mul16x8_tuning: f32 = 1.5 * acs_bias;
    let k8x16_base = 1.6;
    let k8x16_mul1 = -0.55 * k_mul16x8_tuning;
    let k8x16_mul2 = 0.901_958_8 * k_mul16x8_tuning;
    let mul16x8 = k8x16_mul2 + k8x16_mul1 / (distance + k8x16_base);

    let k_mul16x16_tuning: f32 = 1.8 * acs_bias;
    let k16x16_base = 1.6;
    let k16x16_mul1 = -0.55 * k_mul16x16_tuning;
    let k16x16_mul2 = 0.901_958_8 * k_mul16x16_tuning;
    let mul16x16 = k16x16_mul2 + k16x16_mul1 / (distance + k16x16_base);

    // Cache the QF rect over the 2x2 super-block: 2 rows × 2 cols.
    // quant_per_block is indexed via [by0*qf_stride .. (by0+2)*qf_stride].
    // Build a local 2x2 view in qf_local stored row-major (stride 2):
    let mut qf_local = [0u8; 4];
    for iy in 0..2 {
        for ix in 0..2 {
            let by = by0 + iy;
            let bx = bx0 + ix;
            qf_local[iy * 2 + ix] = quant_per_block[by * qf_stride + bx];
        }
    }

    // Per-(dy,dx) entropy estimates of 4 separate DCT8 blocks.
    let mut entropy = [[0.0f32; 2]; 2];
    for dy in 0..2 {
        for dx in 0..2 {
            let e = estimate_entropy(
                STRATEGY_DCT,
                opsin,
                bx0,
                by0,
                dx,
                dy,
                distance,
                matrices,
                &qf_local,
                2,
            );
            entropy[dy][dx] = mul8x8 * (3.0 + e);
        }
    }

    // Estimate the 2 candidate DCT16X8 (vertical pair) and 2 candidate DCT8X16
    // (horizontal pair) options, plus the single DCT16X16 covering all 4 blocks.
    let entropy_16x8_left = mul16x8
        * estimate_entropy(
            STRATEGY_DCT16X8,
            opsin,
            bx0,
            by0,
            0,
            0,
            distance,
            matrices,
            &qf_local,
            2,
        );
    let entropy_16x8_right = mul16x8
        * estimate_entropy(
            STRATEGY_DCT16X8,
            opsin,
            bx0,
            by0,
            1,
            0,
            distance,
            matrices,
            &qf_local,
            2,
        );
    let entropy_8x16_top = mul16x8
        * estimate_entropy(
            STRATEGY_DCT8X16,
            opsin,
            bx0,
            by0,
            0,
            0,
            distance,
            matrices,
            &qf_local,
            2,
        );
    let entropy_8x16_bottom = mul16x8
        * estimate_entropy(
            STRATEGY_DCT8X16,
            opsin,
            bx0,
            by0,
            0,
            1,
            distance,
            matrices,
            &qf_local,
            2,
        );
    let entropy_16x16 = mul16x16
        * estimate_entropy(
            STRATEGY_DCT16X16,
            opsin,
            bx0,
            by0,
            0,
            0,
            distance,
            matrices,
            &qf_local,
            2,
        );

    // Cost of choosing per-column DCT16X8 vs the 2 DCT8s in that column.
    let cost16x8 = entropy_16x8_left.min(entropy[0][0] + entropy[1][0])
        + entropy_16x8_right.min(entropy[0][1] + entropy[1][1]);
    let cost8x16 = entropy_8x16_top.min(entropy[0][0] + entropy[0][1])
        + entropy_8x16_bottom.min(entropy[1][0] + entropy[1][1]);
    // Cost of choosing the single DCT16X16 over all four blocks.
    let cost16x16 = entropy_16x16;
    let total_dct8 = entropy[0][0] + entropy[0][1] + entropy[1][0] + entropy[1][1];

    // Pick the cheapest option overall. DCT16X16 wins if it's both the best
    // rectangular choice *and* cheaper than 4 separate DCT8s.
    let best_rect = cost16x8.min(cost8x16);
    if cost16x16 < best_rect
        && cost16x16 < total_dct8
        && ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X16)
    {
        ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X16);
    } else if cost16x8 < cost8x16 {
        // Try DCT16X8 in each column independently.
        if entropy_16x8_left < entropy[0][0] + entropy[1][0]
            && ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT16X8)
        {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT16X8);
        }
        if entropy_16x8_right < entropy[0][1] + entropy[1][1]
            && ac_strategy.can_place_strategy(bx0 + 1, by0, STRATEGY_DCT16X8)
        {
            ac_strategy.set_first(bx0 + 1, by0, STRATEGY_DCT16X8);
        }
    } else {
        // Try DCT8X16 in each row independently.
        if entropy_8x16_top < entropy[0][0] + entropy[0][1]
            && ac_strategy.can_place_strategy(bx0, by0, STRATEGY_DCT8X16)
        {
            ac_strategy.set_first(bx0, by0, STRATEGY_DCT8X16);
        }
        if entropy_8x16_bottom < entropy[1][0] + entropy[1][1]
            && ac_strategy.can_place_strategy(bx0, by0 + 1, STRATEGY_DCT8X16)
        {
            ac_strategy.set_first(bx0, by0 + 1, STRATEGY_DCT8X16);
        }
    }
}

/// AdjustQuantField: for each multi-block transform, propagate the maximum
/// raw_quant across covered blocks (libjxl-tiny: AdjustQuantField). This
/// keeps the per-block QF consistent within a multi-block transform.
pub fn adjust_quant_field(ac_strategy: &AcStrategyImage, quant_field: &mut crate::image::ImageB) {
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    for (x, y, raw_strategy) in ac_strategy.iter_first_blocks() {
        let cov_x = AcStrategyImage::covered_blocks_x_of(raw_strategy);
        let cov_y = AcStrategyImage::covered_blocks_y_of(raw_strategy);
        if cov_x == 1 && cov_y == 1 {
            continue;
        }
        // Find max quant across covered blocks.
        let mut max_q: u8 = 0;
        for iy in 0..cov_y {
            for ix in 0..cov_x {
                let q = quant_field.row(y + iy)[x + ix];
                if q > max_q {
                    max_q = q;
                }
            }
        }
        for iy in 0..cov_y {
            for ix in 0..cov_x {
                quant_field.row_mut(y + iy)[x + ix] = max_q;
            }
        }
    }
    let _ = (xsize, ysize);
}

/// Run block selection across the whole image (raster order, 2×2 super-blocks).
/// Blocks not covered by a multi-block transform stay as DCT8.
pub(crate) fn fill_ac_strategy(
    opsin: &Image3F,
    distance: f32,
    matrices: &DequantMatrices,
    quant_field: &mut crate::image::ImageB,
    ac_strategy: &mut AcStrategyImage,
) {
    let xsize = ac_strategy.xsize();
    let ysize = ac_strategy.ysize();
    let qf_stride = xsize;
    let mut qf_flat = vec![0u8; xsize * ysize];
    for y in 0..ysize {
        let r = quant_field.row(y);
        qf_flat[y * xsize..(y + 1) * xsize].copy_from_slice(&r[..xsize]);
    }
    let mut by = 0;
    while by + 1 < ysize {
        let mut bx = 0;
        while bx + 1 < xsize {
            find_best_16x16_transform(
                opsin,
                bx,
                by,
                distance,
                matrices,
                &qf_flat,
                qf_stride,
                ac_strategy,
            );
            bx += 2;
        }
        by += 2;
    }
    adjust_quant_field(ac_strategy, quant_field);
}
