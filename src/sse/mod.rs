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
mod dark_aq;
mod dct;
mod entropy;
mod inflated_cost;
mod lossless_grad;
mod quant;
mod quantize_xyb;
mod structure_aq;
mod xyb;

pub(crate) use ac_strategy::sse_and_rate_sse;
pub(crate) use adaptive_quant::fill_quant_field;
pub(crate) use afv::{afv0_sse41, afv1_sse41, afv2_sse41, afv3_sse41};
pub(crate) use dark_aq::dark_structure_stats_sse41;
pub(crate) use entropy::counts_bit_cost_sse41;
pub(crate) use inflated_cost::{combine_error_sse41, error_gradient_energy_sse41};
pub(crate) use lossless_grad::grad_pack_interior;
pub(crate) use quant::{
    apply_quant_field_gain_sse41, apply_structure_aq_sse41, quantize_block_ac_sse41,
    quantize_dc_cfl_sse41, quantize_dc_sse41,
};
pub(crate) use quantize_xyb::{quantize_xyb_channels_sse41, quantize_xyb_tile_colors_sse41};
pub(crate) use structure_aq::block_features_sse41;
pub(crate) use xyb::to_xyb_sse41_band;
