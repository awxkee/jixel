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
mod ac_strategy;
mod adaptive_quant;
mod afv;
mod cfl;
mod dark_aq;
mod dct;
mod entropy;
mod inflated_cost;
mod lossless_grad;
mod quant;
mod quantize_xyb;
mod structure_aq;
mod xyb;
mod ytob;

pub(crate) use ac_strategy::sse_and_rate_neon;
pub(crate) use adaptive_quant::fill_quant_field;
pub(crate) use afv::{afv0_neon, afv1_neon, afv2_neon, afv3_neon};
pub(crate) use cfl::{apply_cfl_neon, cfl_regression_neon};
pub(crate) use dark_aq::dark_structure_stats_neon;
pub(crate) use dct::{
    dc_from_dct16x32_neon, dc_from_dct32x16_neon, dc_from_dct32x32_neon, dct4x4_neon, dct4x8_neon,
    dct8x4_neon, dct8x8_neon, dct8x16_neon, dct16x8_neon, dct16x16_neon, dct16x32_neon,
    dct32x16_neon, dct32x32_neon, dct32x64_neon, dct64x32_neon, dct64x64_neon, inv_dct8x8_neon,
    inv_dct8x16_neon, inv_dct16x8_neon, inv_dct16x16_neon, inv_dct16x32_neon, inv_dct32x16_neon,
    inv_dct32x32_neon,
};
pub(crate) use entropy::counts_bit_cost_neon;
pub(crate) use inflated_cost::{
    combine_error_neon, error_gradient_energy_neon, recon_dist_and_rate_neon, ssim_deficit_neon,
};
pub(crate) use lossless_grad::grad_pack_interior;
pub(crate) use quant::{
    apply_quant_field_gain_neon, apply_structure_aq_neon, quantize_block_ac_neon,
    quantize_dc_cfl_neon, quantize_dc_neon,
};
pub(crate) use quantize_xyb::{quantize_xyb_channels_neon, quantize_xyb_tile_colors_neon};
pub(crate) use structure_aq::block_features_neon;
pub(crate) use xyb::to_xyb_neon_band;
pub(crate) use ytob::{accumulate_ytob_weights_neon, fill_ytob_residuals_neon, fill_ytob_row_neon};
