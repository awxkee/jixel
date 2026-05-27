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
use crate::ac_context::{
    K_COEFF_ORDER_8X8, block_context, non_zero_context, zero_density_context_8x8,
    zero_density_contexts_offset,
};
use crate::bit_writer::BitWriter;
use crate::dc_group_data::DcGroupData;
use crate::dct::dct8x8;
use crate::entropy::{EntropyCode, Token, pack_signed, write_token};
use crate::image::{Image3B, Image3F, Rect};
use crate::quant_weights::{DC_QUANT, DequantMatrices, INV_DC_QUANT};

const K_GROUP_DIM_IN_BLOCKS: usize = 32;

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

fn quantize_block_8x8_ac(
    block_in: &[f32; 64],
    c: usize,
    qm: &[f32; 64],
    quant: i32,
    scale: f32,
    qm_multiplier: f32,
    block_out: &mut [i32; 64],
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
    let q_scaled = qac * qm_multiplier;
    for y in 0..8 {
        let yfix = if y >= 4 { 2 } else { 0 };
        for x in 0..8 {
            let q_x = if x >= 4 { 1 } else { 0 };
            let threshold = thr[yfix + q_x];
            let idx = y * 8 + x;
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

const DEFAULT_QUANT_BIAS_1: f32 = 1.0 - 0.070_054_5;
const DEFAULT_QUANT_BIAS_3: f32 = 0.145;

#[inline]
fn adjust_quant_bias_y(quant: i32) -> f32 {
    // Mirrors libjxl-tiny's AdjustQuantBias for c=1.
    // |q| < 1.125 → q == 0 ? 0 : (sign(q) * bias[1])
    // else        → q - bias[3]/q
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

fn quantize_roundtrip_y_block(
    qm: &[f32; 64],
    dqm: &[f32; 64],
    scale: f32,
    quant: i32,
    inout: &mut [f32; 64],
    quantized: &mut [i32; 64],
) {
    quantize_block_8x8_ac(inout, 1, qm, quant, scale, 1.0, quantized);
    let inv_qac = 1.0 / (scale * quant as f32);
    for k in 0..64 {
        inout[k] = adjust_quant_bias_y(quantized[k]) * dqm[k] * inv_qac;
    }
}

/// Process and tokenize one stripe of an AC group.
///
/// * `opsin`        — XYB pixels of the stripe; row-major in each plane.
/// * `group_brect`  — Stripe position WITHIN THE DC GROUP, in blocks.
/// * `matrices`     — Dequant matrices.
/// * `scale`        — AC scale (= global_scale / 65536).
/// * `scale_dc`     — DC scale (= quant_dc * scale).
/// * `x_qm_scale`   — Header-signaled X matrix multiplier knob.
/// * `dc_data`      — DC group accumulator (sized to the DC group).
/// * `ac_code`      — Static AC entropy code.
/// * `num_nzeros`   — Per-AC-group block nonzero counts (sized 32x32), addressed at `(group_brect.y0 % 32 + by, bx)`.
/// * `writer`       — Section bit writer for this AC group.
#[allow(clippy::too_many_arguments)]
pub fn write_ac_group(
    opsin: &Image3F,
    group_brect: Rect,
    matrices: &DequantMatrices,
    scale: f32,
    scale_dc: f32,
    x_qm_scale: u32,
    dc_data: &mut DcGroupData,
    ac_code: &EntropyCode,
    num_nzeros: &mut Image3B,
    writer: &mut BitWriter,
) {
    let xsize_blocks = group_brect.xsize;
    let ysize_blocks = group_brect.ysize;

    let inv_factor = [
        INV_DC_QUANT[0] * scale_dc,
        INV_DC_QUANT[1] * scale_dc,
        INV_DC_QUANT[2] * scale_dc,
    ];
    // cfl_factor for X = 0.0; for B = kInvDCQuant[2] * kDCQuant[1].
    let cfl_factor_b = INV_DC_QUANT[2] * DC_QUANT[1];
    let x_qm_mul = 1.25f32.powf(x_qm_scale as f32 - 2.0);

    let nzeros_by0 = group_brect.y0 % K_GROUP_DIM_IN_BLOCKS;

    for by in 0..ysize_blocks {
        let nz_by = nzeros_by0 + by;
        let opsin_y0 = by * 8; // stripe-local

        // dc_data raw_quant_field row at this stripe row.
        let global_by = group_brect.y0 + by;

        for bx in 0..xsize_blocks {
            let global_bx = group_brect.x0 + bx;
            let quant_ac = dc_data.raw_quant_field.row(global_by)[global_bx] as i32;

            // --- DCT each channel from pixels ---
            let mut coeffs_x = [0.0f32; 64];
            let mut coeffs_y = [0.0f32; 64];
            let mut coeffs_b = [0.0f32; 64];

            let mut tmp_in = [0.0f32; 64];

            for c in 0..3 {
                let dst: &mut [f32; 64] = match c {
                    0 => &mut coeffs_x,
                    1 => &mut coeffs_y,
                    _ => &mut coeffs_b,
                };
                let plane = opsin.plane(c);
                let src_x = bx * 8; // stripe-local
                for (yy, dst_tmp) in tmp_in.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let row = plane.row(opsin_y0 + yy);
                    dst_tmp.copy_from_slice(&row[src_x..src_x + 8]);
                }
                dct8x8(&tmp_in, dst);
            }

            // --- Y DC and AC quantize, with roundtrip for CfL ---
            let y_dc_raw = coeffs_y[0];
            let y_dc_q = (inv_factor[1] * y_dc_raw).round() as i16;
            dc_data.quant_dc.plane_row_mut(1, global_by)[global_bx] = y_dc_q;

            let mut quant_y = [0i32; 64];
            quantize_roundtrip_y_block(
                matrices.inv_matrix(1),
                matrices.matrix(1),
                scale,
                quant_ac,
                &mut coeffs_y,
                &mut quant_y,
            );

            // --- B minus Y CfL (base correlation = 1.0); X has no CfL ---
            // Save raw X DC first (X has no CfL so we use raw coeffs_x[0]).
            let x_dc_raw = coeffs_x[0];
            // For B, libjxl-tiny applies CfL *first*, then computes DC from the
            // post-CfL coefficient block (so b_dc_for_storage = b_dc_raw - y_dc_dq).
            for k in 0..64 {
                coeffs_b[k] -= coeffs_y[k];
            }
            // Now coeffs_b[0] = b_dc_raw - y_dc_dq (the post-CfL DC).

            // X DC + AC
            let x_dc_q = (inv_factor[0] * x_dc_raw).round() as i16;
            dc_data.quant_dc.plane_row_mut(0, global_by)[global_bx] = x_dc_q;
            let mut quant_x = [0i32; 64];
            quantize_block_8x8_ac(
                &coeffs_x,
                0,
                matrices.inv_matrix(0),
                quant_ac,
                scale,
                x_qm_mul,
                &mut quant_x,
            );

            // B DC + AC
            // dc_b_for_storage = round((post-CfL b_dc) * inv_factor[2] - y_dc_q * cfl_factor_b)
            let b_dc_post_cfl = coeffs_b[0];
            let b_dc_q =
                (b_dc_post_cfl * inv_factor[2] - y_dc_q as f32 * cfl_factor_b).round() as i16;
            dc_data.quant_dc.plane_row_mut(2, global_by)[global_bx] = b_dc_q;
            let mut quant_b = [0i32; 64];
            quantize_block_8x8_ac(
                &coeffs_b,
                2,
                matrices.inv_matrix(2),
                quant_ac,
                scale,
                1.0,
                &mut quant_b,
            );

            // --- Tokenize in order Y, X, B ---
            for &c in &[1usize, 0, 2] {
                let block: &[i32; 64] = match c {
                    0 => &quant_x,
                    1 => &quant_y,
                    _ => &quant_b,
                };

                let nzeros = num_nonzero_except_dc(block);
                num_nzeros.plane_row_mut(c, nz_by)[bx] = nzeros as u8;

                // Predict from top row (if any) and left.
                let row_top: Option<&[u8]> = if nz_by == 0 {
                    None
                } else {
                    Some(num_nzeros.plane_row(c, nz_by - 1))
                };
                let row = num_nzeros.plane_row(c, nz_by);
                let predicted = predict_from_top_and_left(row_top, row, bx, 32);

                let block_ctx = block_context(c, /*ac_strategy_code=*/ 0);
                let nzero_ctx = non_zero_context(predicted as u32, block_ctx);
                let histo_offset = zero_density_contexts_offset(block_ctx);

                write_token(Token::new(nzero_ctx, nzeros as u32), ac_code, writer);

                // size/16 = 4 for 8x8 blocks
                let mut prev: usize = if nzeros > 4 { 0 } else { 1 };
                let mut remaining = nzeros;
                let mut k = 1usize;
                while k < 64 && remaining != 0 {
                    let coef = block[K_COEFF_ORDER_8X8[k] as usize];
                    let ctx = histo_offset as usize
                        + zero_density_context_8x8(remaining as usize, k, prev);
                    let u = pack_signed(coef);
                    write_token(Token::new(ctx as u32, u), ac_code, writer);
                    prev = if coef != 0 { 1 } else { 0 };
                    if coef != 0 {
                        remaining -= 1;
                    }
                    k += 1;
                }
                debug_assert_eq!(remaining, 0);
            }
        }
    }
    let _ = (xsize_blocks, ysize_blocks);
}
