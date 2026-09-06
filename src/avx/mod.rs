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
mod color;
mod dark_aq;
mod dct;
mod entropy;
mod frame;
mod inflated_cost;
mod lossless_grad;
mod modular;
mod mosaic_seam;
mod quant;
mod quantize_xyb;
mod structure_aq;
mod xyb;
mod ytob;

pub(crate) use ac_strategy::{
    gradient_region_stats_avx2, gradient_region_stats_with_chroma_avx2, sse_and_rate_avx2,
};
pub(crate) use adaptive_quant::{chroma_hf_stats_avx2, fill_quant_field};
pub(crate) use afv::{afv0_avx2, afv1_avx2, afv2_avx2, afv3_avx2};
pub(crate) use cfl::{
    apply_cfl_avx2, cfl_closed_loop_cost_avx2, cfl_rdo_block_avx2, cfl_rdo_stats_avx2,
    cfl_regression_avx2,
};
pub(crate) use color::color_matrix_shaper_avx2;
pub(crate) use dark_aq::{dark_structure_stats_avx2, fill_blue_tile_avx2};
pub(crate) use dct::{
    dc_from_dct16x32_avx2, dc_from_dct32x16_avx2, dc_from_dct32x32_avx2, dc_from_dct32x64_avx2,
    dc_from_dct64x32_avx2, dc_from_dct64x64_avx2, dct2x2_8x8_avx2, dct4x4_avx2, dct4x8_avx2,
    dct8x4_avx2, dct8x8_avx2, dct8x16_avx2, dct16x8_avx2, dct16x16_avx2, dct16x32_avx2,
    dct32x16_avx2, dct32x32_avx2, dct32x64_avx2, dct64x32_avx2, dct64x64_avx2, identity8x8_avx2,
    inv_dct2x2_8x8_avx2, inv_dct8x8_avx2, inv_dct8x16_avx2, inv_dct16x8_avx2, inv_dct16x16_avx2,
    inv_dct16x32_avx2, inv_dct32x16_avx2, inv_dct32x32_avx2, inv_dct32x64_avx2, inv_dct64x32_avx2,
    inv_dct64x64_avx2, inv_identity8x8_avx2,
};
pub(crate) use entropy::counts_bit_cost_avx2;
pub(crate) use frame::chroma_gradient_sums_avx2;
pub(crate) use inflated_cost::{
    combine_error_avx2, error_gradient_energy_avx2, error_gradient_peak_energy_avx2,
    recon_dist_and_rate_avx2, rgb_hue_chroma_edge_loss_avx2, ssim_deficit_avx2,
};
pub(crate) use lossless_grad::grad_pack_interior;
pub(crate) use modular::{
    tokenize_alpha_u8_first_row_avx2, tokenize_alpha_u8_interior_avx2,
    tokenize_alpha_u16_first_row_avx2, tokenize_alpha_u16_interior_avx2,
};
pub(crate) use mosaic_seam::mosaic_seam_stats_avx2;
pub(crate) use quant::{
    apply_quant_field_gain_avx2, apply_structure_aq_avx2, quantize_block_ac_avx2, quantize_dc_avx2,
    quantize_dc_cfl_avx2,
};
pub(crate) use quantize_xyb::{quantize_xyb_channels_avx2, quantize_xyb_tile_colors_avx2};
pub(crate) use structure_aq::block_features_avx2;
pub(crate) use xyb::to_xyb_avx2_band;
pub(crate) use ytob::{accumulate_ytob_weights_avx2, fill_ytob_residuals_avx2, fill_ytob_row_avx2};
