/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::{adaptive_quant, dct, enc_group, enc_xyb, inflated_cost};

/// Per-encode dispatch table.  The individual modules still own their `OnceLock`
/// selectors, but hot inner loops receive these already-resolved function
/// references instead of touching a static guard for every block / band / token.
#[derive(Clone, Copy)]
pub(crate) struct EncodingContext {
    pub(crate) to_xyb_band: enc_xyb::ToXybBandFn,
    pub(crate) fill_quant_field: adaptive_quant::FillQuantFieldFn,
    pub(crate) sse_and_rate: inflated_cost::SseAndRateFn,
    pub(crate) rate_log2_lut: &'static inflated_cost::RateLog2Lut,
    pub(crate) quantize_block_ac: enc_group::QuantizeBlockAcFn,

    pub(crate) dct8x8: &'static dct::DctFn<64>,
    pub(crate) dct8x16: &'static dct::DctFn<128>,
    pub(crate) dct16x8: &'static dct::DctFn<128>,
    pub(crate) dct16x16: &'static dct::DctFn<256>,
    pub(crate) dct4x4: &'static dct::DctFn<64>,
    pub(crate) dct4x8: &'static dct::DctFn<64>,
    pub(crate) dct8x4: &'static dct::DctFn<64>,
    pub(crate) dct32x32: &'static dct::DctFn<1024>,
    pub(crate) dct32x16: &'static dct::DctFn<512>,
    pub(crate) dct16x32: &'static dct::DctFn<512>,
    pub(crate) dct64x64: &'static dct::DctFn<4096>,
}

impl EncodingContext {
    pub(crate) fn new() -> Self {
        Self {
            to_xyb_band: enc_xyb::selected_to_xyb_band_fn(),
            fill_quant_field: adaptive_quant::selected_fill_quant_field_fn(),
            sse_and_rate: inflated_cost::selected_sse_and_rate_fn(),
            rate_log2_lut: inflated_cost::rate_log2_lut(),
            quantize_block_ac: enc_group::selected_quantize_block_ac_fn(),

            dct8x8: dct::selected_dct8x8(),
            dct8x16: dct::selected_dct8x16(),
            dct16x8: dct::selected_dct16x8(),
            dct16x16: dct::selected_dct16x16(),
            dct4x4: dct::selected_dct4x4(),
            dct4x8: dct::selected_dct4x8(),
            dct8x4: dct::selected_dct8x4(),
            dct32x32: dct::selected_dct32x32(),
            dct32x16: dct::selected_dct32x16(),
            dct16x32: dct::selected_dct16x32(),
            dct64x64: dct::selected_dct64x64(),
        }
    }
}

impl Default for EncodingContext {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
