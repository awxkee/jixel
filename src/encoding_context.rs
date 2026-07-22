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

use crate::ac_context::BlockContextModel;
use crate::dark_aq::DarkAqConfig;
use crate::quant_weights::DequantMatrices;
use crate::{
    Speed, adaptive_quant, dct, enc_ac_strategy, enc_color_correlation, enc_group, enc_xyb,
    inflated_cost,
};

/// Per-encode dispatch table.  The individual modules still own their `OnceLock`
/// selectors, but hot inner loops receive these already-resolved function
/// references instead of touching a static guard for every block / band / token.
pub(crate) struct EncodingContext {
    pub(crate) speed: Speed,
    pub(crate) boost: Option<DarkAqConfig>,
    pub(crate) matrices: DequantMatrices,
    pub(crate) block_context_model: BlockContextModel,

    pub(crate) to_xyb_band: enc_xyb::ToXybBandFn,
    pub(crate) fill_quant_field: adaptive_quant::FillQuantFieldFn,
    pub(crate) sse_and_rate: inflated_cost::SseAndRateFn,
    pub(crate) rate_log2_lut: &'static inflated_cost::RateLog2Lut,
    pub(crate) quantize_block_ac: enc_group::QuantizeBlockAcFn,
    pub(crate) apply_cfl: enc_ac_strategy::ApplyCflFn,
    pub(crate) cfl_regression: enc_color_correlation::CflRegressionFn,

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
    pub(crate) dct64x32: &'static dct::DctFn<2048>,
    pub(crate) dct32x64: &'static dct::DctFn<2048>,
}

impl EncodingContext {
    pub(crate) fn new(speed: Speed, boost: Option<DarkAqConfig>) -> Self {
        Self {
            speed,
            boost,
            matrices: DequantMatrices::new(),
            block_context_model: BlockContextModel::Compact,

            to_xyb_band: enc_xyb::selected_to_xyb_band_fn(),
            fill_quant_field: adaptive_quant::selected_fill_quant_field_fn(),
            sse_and_rate: inflated_cost::selected_sse_and_rate_fn(),
            rate_log2_lut: inflated_cost::rate_log2_lut(),
            quantize_block_ac: enc_group::selected_quantize_block_ac_fn(),
            apply_cfl: enc_ac_strategy::selected_apply_cfl_fn(),
            cfl_regression: enc_color_correlation::selected_cfl_regression_fn(),

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
            dct64x32: dct::selected_dct64x32(),
            dct32x64: dct::selected_dct32x64(),
        }
    }

    pub(crate) fn new_for_image(
        speed: Speed,
        boost: Option<DarkAqConfig>,
        width: usize,
        height: usize,
        distance: f32,
    ) -> Self {
        let mut ctx = Self::new(speed, boost);
        let num_blocks = width.div_ceil(8).saturating_mul(height.div_ceil(8));
        let split_threshold = ((4096.0 * distance.max(0.03)).ceil() as usize).max(8192);
        if distance >= 1.5 && num_blocks >= split_threshold {
            ctx.block_context_model = BlockContextModel::LargeTransform;
        }
        ctx
    }
}

impl Default for EncodingContext {
    #[inline]
    fn default() -> Self {
        Self::new(Speed::Fast, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_context_model_is_gated_by_size_and_distance() {
        assert_eq!(
            EncodingContext::new_for_image(Speed::Slow, None, 768, 512, 3.0).block_context_model,
            BlockContextModel::Compact
        );
        assert_eq!(
            EncodingContext::new_for_image(Speed::Slow, None, 2000, 1400, 3.0).block_context_model,
            BlockContextModel::LargeTransform
        );
        assert_eq!(
            EncodingContext::new_for_image(Speed::Slow, None, 2000, 1400, 1.0).block_context_model,
            BlockContextModel::Compact
        );
    }
}
