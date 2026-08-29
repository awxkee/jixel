/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

pub(crate) use ac_strategy::{
    gradient_region_stats_wasm, gradient_region_stats_with_chroma_wasm, sse_and_rate_wasm,
};
pub(crate) use adaptive_quant::fill_quant_field;
pub(crate) use afv::{afv0_wasm, afv1_wasm, afv2_wasm, afv3_wasm};
pub(crate) use cfl::{apply_cfl_wasm, cfl_rdo_block_wasm, cfl_rdo_stats_wasm, cfl_regression_wasm};
pub(crate) use dark_aq::dark_structure_stats_wasm;
pub(crate) use dct::*;
pub(crate) use entropy::counts_bit_cost_wasm;
pub(crate) use inflated_cost::{
    combine_error_wasm, error_gradient_energy_wasm, error_gradient_peak_energy_wasm,
};
pub(crate) use lossless_grad::grad_pack_interior;
pub(crate) use quant::{
    apply_quant_field_gain_wasm, apply_structure_aq_wasm, quantize_block_ac_wasm,
    quantize_dc_cfl_wasm, quantize_dc_wasm,
};
pub(crate) use quantize_xyb::{quantize_xyb_channels_wasm, quantize_xyb_tile_colors_wasm};
pub(crate) use structure_aq::block_features_wasm;
pub(crate) use xyb::to_xyb_wasm_band;
pub(crate) use ytob::{fill_ytob_residuals_wasm, fill_ytob_row_wasm};
