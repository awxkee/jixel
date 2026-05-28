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
#![allow(clippy::excessive_precision)]

use crate::ac_context::{
    K_COEFF_ORDER_8X8, K_COEFF_ORDER_16X8, K_COEFF_ORDER_16X16, block_context, non_zero_context,
    zero_density_context, zero_density_context_8x8, zero_density_contexts_offset,
};
use crate::dc_group_data::{
    AcStrategyImage, DcGroupData, STRATEGY_DCT, STRATEGY_DCT8X16, STRATEGY_DCT16X8,
    STRATEGY_DCT16X16,
};
use crate::dct::{
    dc_from_dct8x16, dc_from_dct16x8, dc_from_dct16x16, dct8x8, dct8x16, dct16x8, dct16x16,
};
use crate::entropy::{Token, pack_signed};
use crate::image::{Image3B, Image3F, Rect};
use crate::quant_weights::{DC_QUANT, DequantMatrices, INV_DC_QUANT};

const K_GROUP_DIM_IN_BLOCKS: usize = 32;

#[inline]
fn predict_from_top_and_left(row_top: Option<&[u8]>, row: &[u8], x: usize, default_val: u8) -> u8 {
    if x == 0 {
        match row_top {
            Some(t) => t[x],
            None => default_val,
        }
    } else if row_top.is_none() {
        row[x - 1]
    } else {
        (row_top.unwrap()[x] as u16 + row[x - 1] as u16).div_ceil(2) as u8
    }
}

/// Number of non-zero coefficients in an 8×8 quantized block, excluding the
/// DC position (k=0). Returns the count for use as nzeros token.
#[inline]
fn num_nonzero_except_dc(block: &[i32; 64]) -> i32 {
    let mut count: i32 = 0;
    for k in 1..64 {
        if block[k] != 0 {
            count += 1;
        }
    }
    count
}

/// Multi-block analog: counts nonzeros in `size = cx*cy*64` coefficients
/// excluding the `cx × cy` LLF positions. For DCT16X8/DCT8X16 (cx=2, cy=1
/// after the cx>=cy swap), the LLF positions are coeffs[0] and coeffs[1] in
/// the 8x16 stride-16 layout — contiguous. For DCT16X16 (cx=cy=2) the LLF
/// positions are {0, 1, 16, 17} in the 16x16 stride-16 layout — NOT
/// contiguous, so we mask them explicitly using the row-stride.
#[inline]
fn num_nonzero_except_llf(block: &[i32], cx: usize, cy: usize) -> i32 {
    let row_stride = cx * 8;
    let xsize_pixels = cx * 8;
    let ysize_pixels = cy * 8;
    let mut count: i32 = 0;
    for v in 0..ysize_pixels {
        for u in 0..xsize_pixels {
            if v < cy && u < cx {
                continue;
            }
            if block[v * row_stride + u] != 0 {
                count += 1;
            }
        }
    }
    count
}

/// Generalized AC quantization for an `xsize × ysize` (in 8x8 blocks) region.
/// `xsize=1, ysize=1` reproduces the 8×8 path. The block buffer has
/// `xsize*ysize*64` floats in row-major order with row stride `xsize*8`.
///
/// Thresholds mirror libjxl-tiny: per-quadrant in the 8×8 case; biased lower
/// for multi-block; for the 1×N or N×1 case the second half of the columns
/// (or rows for ysize=1, xsize=1 special) uses the second threshold pair.
fn quantize_block_ac(
    block_in: &[f32],
    c: usize,
    qm: &[f32],
    quant: i32,
    scale: f32,
    qm_multiplier: f32,
    xsize: usize,
    ysize: usize,
    block_out: &mut [i32],
) {
    let qac = scale * quant as f32;
    let mut thr = [0.58f32, 0.635, 0.66, 0.7];
    if c == 0 {
        for i in 1..4 {
            thr[i] += 0.08;
        }
    }
    if c == 2 {
        for i in 1..4 {
            thr[i] = 0.75;
        }
    }
    if xsize > 1 || ysize > 1 {
        let delta =
            (0.003_f32 * xsize as f32 * ysize as f32).clamp(0.0, if c > 0 { 0.08 } else { 0.12 });
        for i in 0..4 {
            thr[i] -= delta;
        }
    }
    let q_scaled = qac * qm_multiplier;
    let width = xsize * 8;
    let height = ysize * 8;
    // The quant matrix, input, and output must all be sized for this transform.
    // A mismatch here means the caller selected the wrong matrix for the
    // strategy (e.g. handing the 64-entry DCT8 matrix to a DCT16X8/DCT16X16
    // block). Fail with a clear message rather than an opaque index panic.
    debug_assert_eq!(
        qm.len(),
        width * height,
        "quant matrix size {} != transform size {}x{}={} (wrong matrix for strategy?)",
        qm.len(),
        width,
        height,
        width * height
    );
    debug_assert!(block_in.len() >= width * height, "block_in too small");
    debug_assert!(block_out.len() >= width * height, "block_out too small");
    for y in 0..height {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        for x in 0..width {
            let threshold = if xsize == 1 {
                // For DCT8 (xsize=1, ysize=1), threshold splits at x>=4.
                thr[yfix + if x >= width / 2 { 1 } else { 0 }]
            } else {
                // Multi-block: split at x >= width/2.
                thr[yfix + if x >= width / 2 { 1 } else { 0 }]
            };
            let idx = y * width + x;
            let q = qm[idx] * q_scaled;
            let val = q * block_in[idx];
            block_out[idx] = if val.abs() >= threshold {
                val.round() as i32
            } else {
                0
            };
        }
    }
}

const DEFAULT_QUANT_BIAS_1: f32 = 1.0 - 0.07005449891748593;
const DEFAULT_QUANT_BIAS_3: f32 = 0.145;

#[inline]
fn adjust_quant_bias_y(quant: i32) -> f32 {
    let aq = quant.unsigned_abs() as f32;
    if aq < 1.125 {
        if quant == 0 {
            0.0
        } else if quant > 0 {
            DEFAULT_QUANT_BIAS_1
        } else {
            -DEFAULT_QUANT_BIAS_1
        }
    } else {
        let q = quant as f32;
        q - DEFAULT_QUANT_BIAS_3 / q
    }
}

/// Y-channel quantize then dequantize-with-bias for CfL roundtrip. `inout`
/// holds size=xsize*ysize*64 floats. `quantized` holds size integers.
fn quantize_roundtrip_y_block(
    qm: &[f32],
    dqm: &[f32],
    scale: f32,
    quant: i32,
    xsize: usize,
    ysize: usize,
    inout: &mut [f32],
    quantized: &mut [i32],
) {
    quantize_block_ac(inout, 1, qm, quant, scale, 1.0, xsize, ysize, quantized);
    let inv_qac = 1.0 / (scale * quant as f32);
    let size = xsize * ysize * 64;
    for k in 0..size {
        inout[k] = adjust_quant_bias_y(quantized[k]) * dqm[k] * inv_qac;
    }
}

/// Process and tokenize one stripe of an AC group, pushing tokens into `out`.
/// Callers buffer tokens across all AC groups, build an adaptive entropy code
/// from the aggregate distribution, then emit them in `encode_frame`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ac_group(
    opsin: &Image3F,
    group_brect: Rect,
    matrices: &DequantMatrices,
    scale: f32,
    scale_dc: f32,
    x_qm_scale: u32,
    dc_data: &mut DcGroupData,
    num_nzeros: &mut Image3B,
    out: &mut Vec<Token>,
) {
    let xsize_blocks = group_brect.xsize;
    let ysize_blocks = group_brect.ysize;

    let inv_factor = [
        INV_DC_QUANT[0] * scale_dc,
        INV_DC_QUANT[1] * scale_dc,
        INV_DC_QUANT[2] * scale_dc,
    ];
    let cfl_factor_b = INV_DC_QUANT[2] * DC_QUANT[1];
    let x_qm_mul = 1.25f32.powf(x_qm_scale as f32 - 2.0);

    let nzeros_by0 = group_brect.y0 % K_GROUP_DIM_IN_BLOCKS;

    // Per-channel scratch sized for the largest transform (DCT16X16 = 2×2 blocks = 256 floats).
    let mut coeffs = [[0.0f32; 256]; 3];
    let mut quantized = [[0i32; 256]; 3];

    for by in 0..ysize_blocks {
        let nz_by = nzeros_by0 + by;
        let global_by = group_brect.y0 + by;

        for bx in 0..xsize_blocks {
            let global_bx = group_brect.x0 + bx;

            // Skip non-first blocks of multi-block transforms.
            if !dc_data.ac_strategy.is_first_block(global_bx, global_by) {
                continue;
            }

            let raw_strategy = dc_data.ac_strategy.raw_strategy(global_bx, global_by);
            let cov_x = AcStrategyImage::covered_blocks_x_of(raw_strategy);
            let cov_y = AcStrategyImage::covered_blocks_y_of(raw_strategy);
            // libjxl-tiny normalizes: cx >= cy. For DCT16X8 (1×2) and DCT8X16
            // (2×1), both end up as cx=2, cy=1, matching the 8×16 storage.
            let (cx, cy) = if cov_y > cov_x {
                (cov_y, cov_x)
            } else {
                (cov_x, cov_y)
            };
            let size = cx * cy * 64;
            let quant_ac = dc_data.raw_quant_field.row(global_by)[global_bx] as i32;

            // ---- Forward DCT for all 3 channels ----
            let opsin_bx = bx * 8;
            let opsin_by = by * 8;
            for c in 0..3 {
                let plane = opsin.plane(c);
                match raw_strategy {
                    STRATEGY_DCT => {
                        let mut tmp = [0.0f32; 64];
                        for yy in 0..8 {
                            let row = plane.row(opsin_by + yy);
                            tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[opsin_bx..opsin_bx + 8]);
                        }
                        let dst: &mut [f32; 64] = (&mut coeffs[c][..64]).try_into().unwrap();
                        dct8x8(&tmp, dst);
                    }
                    STRATEGY_DCT16X8 => {
                        let mut tmp = [0.0f32; 128]; // 16 rows × 8 cols, stride 8
                        for yy in 0..16 {
                            let row = plane.row(opsin_by + yy);
                            tmp[yy * 8..yy * 8 + 8].copy_from_slice(&row[opsin_bx..opsin_bx + 8]);
                        }
                        let dst: &mut [f32; 128] = (&mut coeffs[c][..128]).try_into().unwrap();
                        dct16x8(&tmp, dst);
                    }
                    STRATEGY_DCT8X16 => {
                        let mut tmp = [0.0f32; 128]; // 8 rows × 16 cols, stride 16
                        for yy in 0..8 {
                            let row = plane.row(opsin_by + yy);
                            tmp[yy * 16..yy * 16 + 16]
                                .copy_from_slice(&row[opsin_bx..opsin_bx + 16]);
                        }
                        let dst: &mut [f32; 128] = (&mut coeffs[c][..128]).try_into().unwrap();
                        dct8x16(&tmp, dst);
                    }
                    STRATEGY_DCT16X16 => {
                        let mut tmp = [0.0f32; 256]; // 16 rows × 16 cols, stride 16
                        for yy in 0..16 {
                            let row = plane.row(opsin_by + yy);
                            tmp[yy * 16..yy * 16 + 16]
                                .copy_from_slice(&row[opsin_bx..opsin_bx + 16]);
                        }
                        let dst: &mut [f32; 256] = (&mut coeffs[c][..256]).try_into().unwrap();
                        dct16x16(&tmp, dst);
                    }
                    _ => unreachable!("invalid raw strategy {}", raw_strategy),
                }
            }

            // ---- Extract DC values and write to DC plane ----
            // For DCT8, DC = coeffs[0]. For multi-block, use DCFromLowestFrequencies.
            // dc_vals[c] holds up to 4 DC values (DCT16X16 = 2×2 covered blocks);
            // indexing is didx = iy * cov_x + ix.
            let mut dc_vals = [[0.0f32; 4]; 3];
            match raw_strategy {
                STRATEGY_DCT => {
                    for c in 0..3 {
                        dc_vals[c][0] = coeffs[c][0];
                    }
                }
                STRATEGY_DCT16X8 => {
                    for c in 0..3 {
                        let cb: &[f32; 128] = (&coeffs[c][..128]).try_into().unwrap();
                        let mut dc2 = [0.0f32; 2];
                        dc_from_dct16x8(cb, &mut dc2);
                        dc_vals[c][0] = dc2[0]; // top covered block
                        dc_vals[c][1] = dc2[1]; // bottom covered block
                    }
                }
                STRATEGY_DCT8X16 => {
                    for c in 0..3 {
                        let cb: &[f32; 128] = (&coeffs[c][..128]).try_into().unwrap();
                        let mut dc2 = [0.0f32; 2];
                        dc_from_dct8x16(cb, &mut dc2);
                        dc_vals[c][0] = dc2[0]; // left covered block
                        dc_vals[c][1] = dc2[1]; // right covered block
                    }
                }
                STRATEGY_DCT16X16 => {
                    // dc_from_dct16x16 returns 4 DC values in [TL, TR, BL, BR]
                    // order, matching the [iy=0,1][ix=0,1] grid the caller uses
                    // (didx = iy * 2 + ix).
                    for c in 0..3 {
                        let cb: &[f32; 256] = (&coeffs[c][..256]).try_into().unwrap();
                        let mut dc4 = [0.0f32; 4];
                        dc_from_dct16x16(cb, &mut dc4);
                        dc_vals[c][0] = dc4[0];
                        dc_vals[c][1] = dc4[1];
                        dc_vals[c][2] = dc4[2];
                        dc_vals[c][3] = dc4[3];
                    }
                }
                _ => unreachable!(),
            }

            // ---- Y channel: roundtrip-quantize, then place DC ----
            // DC for storage (per covered block, using pre-swap cov_x/cov_y).
            let mut y_dc_q_arr = [[0i16; 2]; 2]; // [iy][ix] for max 2x2 — only iy*ix entries used
            for iy in 0..cov_y {
                for ix in 0..cov_x {
                    let didx = iy * cov_x + ix;
                    let y_dc_q = (inv_factor[1] * dc_vals[1][didx]).round() as i16;
                    dc_data.quant_dc.plane_row_mut(1, global_by + iy)[global_bx + ix] = y_dc_q;
                    y_dc_q_arr[iy][ix] = y_dc_q;
                }
            }
            // Quantize Y AC with roundtrip (modifies coeffs[1] to dequantized).
            // Matrix selection: DCT8 uses 8×8 weights, DCT16X8/8X16 share the
            // 128-float 16×8 weights, DCT16X16 uses the 256-float 16×16 weights.
            let (inv_qm_y, qm_y): (&[f32], &[f32]) = match raw_strategy {
                STRATEGY_DCT      => (&matrices.inv_matrix(1)[..],      &matrices.matrix(1)[..]),
                STRATEGY_DCT16X16 => (&matrices.inv_matrix_16x16(1)[..], &matrices.matrix_16x16(1)[..]),
                _ /* 16X8/8X16 */ => (&matrices.inv_matrix_16x8(1)[..], &matrices.matrix_16x8(1)[..]),
            };
            quantize_roundtrip_y_block(
                inv_qm_y,
                qm_y,
                scale,
                quant_ac,
                cx,
                cy,
                &mut coeffs[1][..size],
                &mut quantized[1][..size],
            );

            // ---- Per-tile CfL factors ----
            let tx = global_bx / 8;
            let ty = global_by / 8;
            let cmap_x = dc_data.ytox_map.row(ty)[tx];
            let cmap_b = dc_data.ytob_map.row(ty)[tx];
            // y_to_x = 0 + cmap_x / 84;  y_to_b = 1 + cmap_b / 84.
            let x_factor = crate::enc_color_correlation::y_to_x_ratio(cmap_x);
            let b_factor = crate::enc_color_correlation::y_to_b_ratio(cmap_b);

            // ---- Apply CfL: X -= x_factor·Y, B -= b_factor·Y on every coefficient ----
            // The decoder reverses CfL in coefficient space (DequantLane) using the
            // dequantized Y AC coefficients, whose cx*cy LLF positions are zero at
            // that point — they are filled from the DC plane (LowestFrequenciesFromDC)
            // only afterwards. So the encoder must subtract using a Y whose LLF
            // positions are likewise zero; otherwise the AC-quantized LLF energy of Y
            // gets folded into the B/X DC (via the DCFromLowestFrequencies extraction
            // below) with no decoder-side counterpart, corrupting chroma by up to a
            // full-scale per-block shift (worst on B, where b_factor ≈ 1).
            {
                let wc = cx * 8;
                for iy in 0..cy {
                    for ix in 0..cx {
                        coeffs[1][iy * wc + ix] = 0.0;
                    }
                }
            }
            for k in 0..size {
                coeffs[0][k] -= x_factor * coeffs[1][k];
                coeffs[2][k] -= b_factor * coeffs[1][k];
            }
            // ---- Extract post-CfL X and B DC ----
            let mut x_dc_post = [0.0f32; 4];
            let mut b_dc_post = [0.0f32; 4];
            match raw_strategy {
                STRATEGY_DCT => {
                    x_dc_post[0] = coeffs[0][0];
                    b_dc_post[0] = coeffs[2][0];
                }
                STRATEGY_DCT16X8 => {
                    let xb: &[f32; 128] = (&coeffs[0][..128]).try_into().unwrap();
                    let bb: &[f32; 128] = (&coeffs[2][..128]).try_into().unwrap();
                    let mut xd = [0.0f32; 2];
                    let mut bd = [0.0f32; 2];
                    dc_from_dct16x8(xb, &mut xd);
                    dc_from_dct16x8(bb, &mut bd);
                    x_dc_post[..2].copy_from_slice(&xd);
                    b_dc_post[..2].copy_from_slice(&bd);
                }
                STRATEGY_DCT8X16 => {
                    let xb: &[f32; 128] = (&coeffs[0][..128]).try_into().unwrap();
                    let bb: &[f32; 128] = (&coeffs[2][..128]).try_into().unwrap();
                    let mut xd = [0.0f32; 2];
                    let mut bd = [0.0f32; 2];
                    dc_from_dct8x16(xb, &mut xd);
                    dc_from_dct8x16(bb, &mut bd);
                    x_dc_post[..2].copy_from_slice(&xd);
                    b_dc_post[..2].copy_from_slice(&bd);
                }
                STRATEGY_DCT16X16 => {
                    let xb: &[f32; 256] = (&coeffs[0][..256]).try_into().unwrap();
                    let bb: &[f32; 256] = (&coeffs[2][..256]).try_into().unwrap();
                    dc_from_dct16x16(xb, &mut x_dc_post);
                    dc_from_dct16x16(bb, &mut b_dc_post);
                }
                _ => unreachable!(),
            }

            // ---- X channel: write post-CfL DC, quantize AC ----
            for iy in 0..cov_y {
                for ix in 0..cov_x {
                    let didx = iy * cov_x + ix;
                    // base_correlation_x = 0 so no Y contribution to X DC store.
                    let x_dc_q = (inv_factor[0] * x_dc_post[didx]).round() as i16;
                    dc_data.quant_dc.plane_row_mut(0, global_by + iy)[global_bx + ix] = x_dc_q;
                }
            }
            let inv_qm_x: &[f32] = match raw_strategy {
                STRATEGY_DCT => &matrices.inv_matrix(0)[..],
                STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(0)[..],
                _ => &matrices.inv_matrix_16x8(0)[..],
            };
            quantize_block_ac(
                &coeffs[0][..size],
                0,
                inv_qm_x,
                quant_ac,
                scale,
                x_qm_mul,
                cx,
                cy,
                &mut quantized[0][..size],
            );

            // ---- B channel: write CfL'd DC, quantize AC ----
            for iy in 0..cov_y {
                for ix in 0..cov_x {
                    let didx = iy * cov_x + ix;
                    let b_dc_q = (b_dc_post[didx] * inv_factor[2]
                        - y_dc_q_arr[iy][ix] as f32 * cfl_factor_b)
                        .round() as i16;
                    dc_data.quant_dc.plane_row_mut(2, global_by + iy)[global_bx + ix] = b_dc_q;
                }
            }
            let inv_qm_b: &[f32] = match raw_strategy {
                STRATEGY_DCT => &matrices.inv_matrix(2)[..],
                STRATEGY_DCT16X16 => &matrices.inv_matrix_16x16(2)[..],
                _ => &matrices.inv_matrix_16x8(2)[..],
            };
            quantize_block_ac(
                &coeffs[2][..size],
                2,
                inv_qm_b,
                quant_ac,
                scale,
                1.0,
                cx,
                cy,
                &mut quantized[2][..size],
            );

            // ---- Tokenize in order Y, X, B ----
            let strategy_code = dc_data.ac_strategy.strategy_code(global_bx, global_by);
            let covered_blocks = cx * cy;
            // log2(covered_blocks): 0 for 1, 1 for 2 (16X8/8X16), 2 for 4 (16X16).
            let log2_covered_blocks = match covered_blocks {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => unreachable!("invalid covered_blocks {}", covered_blocks),
            };

            for &c in &[1usize, 0, 2] {
                let block = &quantized[c][..size];

                let nzeros = if covered_blocks == 1 {
                    num_nonzero_except_dc(<&[i32; 64]>::try_from(block).unwrap())
                } else {
                    num_nonzero_except_llf(block, cx, cy)
                };

                // libjxl-tiny: NumNonZeroExceptLLF stores `(nzeros + covered_blocks - 1) >> log2_covered_blocks`
                // to all covered cells in num_nzeros.
                let shifted = ((nzeros as usize + covered_blocks - 1) >> log2_covered_blocks) as u8;
                // Pre-swap iteration (cov_x, cov_y from raw strategy).
                for iy in 0..cov_y {
                    for ix in 0..cov_x {
                        num_nzeros.plane_row_mut(c, nz_by + iy)[bx + ix] = shifted;
                    }
                }

                // Predict from top and left.
                let row_top: Option<&[u8]> = if nz_by == 0 {
                    None
                } else {
                    Some(num_nzeros.plane_row(c, nz_by - 1))
                };
                let row = num_nzeros.plane_row(c, nz_by);
                let predicted = predict_from_top_and_left(row_top, row, bx, 32);

                let block_ctx = block_context(c, strategy_code);
                let nzero_ctx = non_zero_context(predicted as u32, block_ctx);
                let histo_offset = zero_density_contexts_offset(block_ctx);

                write_token_into(Token::new(nzero_ctx, nzeros as u32), out);

                // Choose the natural coefficient order. For DCT16X16 we use
                // the 256-entry natural radial zigzag; for 16X8/8X16 the
                // 128-entry shared order; otherwise the 8X8 zigzag.
                let order: &[u8] = match raw_strategy {
                    STRATEGY_DCT => &K_COEFF_ORDER_8X8[..],
                    STRATEGY_DCT16X16 => &K_COEFF_ORDER_16X16[..],
                    _ => &K_COEFF_ORDER_16X8[..],
                };

                let mut prev: usize = if nzeros as usize > size / 16 { 0 } else { 1 };
                let mut remaining = nzeros;
                // Skip the first `covered_blocks` positions (LF).
                let mut k = covered_blocks;
                while k < size && remaining != 0 {
                    let coef = block[order[k] as usize];
                    let ctx = histo_offset as usize
                        + if covered_blocks == 1 {
                            zero_density_context_8x8(remaining as usize, k, prev)
                        } else {
                            zero_density_context(
                                remaining as usize,
                                k,
                                covered_blocks,
                                log2_covered_blocks,
                                prev,
                            )
                        };
                    write_token_into(Token::new(ctx as u32, pack_signed(coef)), out);
                    prev = if coef != 0 { 1 } else { 0 };
                    if coef != 0 {
                        remaining -= 1;
                    }
                    k += 1;
                }
                debug_assert_eq!(
                    remaining, 0,
                    "remaining nzeros at end: strategy={} c={}",
                    strategy_code, c
                );
            }
        }
    }
}

#[inline]
fn write_token_into(t: Token, out: &mut Vec<Token>) {
    out.push(t);
}
