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

use crate::afv;
use crate::dark_aq::{self, DarkAqConfig};
use crate::quant_weights::DequantMatrices;
use crate::thread_pool::ThreadPool;
use crate::{
    Speed, ac_strategy, adaptive_quant, color_correlation, dct, frame, group, inflated_cost,
    structure_aq, xyb,
};

#[inline]
fn channel_weights_for_bias(bias: f32) -> [f32; 3] {
    let matrix_mix = ((bias - xyb::B_BIAS) / (0.85 - xyb::B_BIAS)).clamp(0.0, 1.0);
    [
        0.30 + (0.10 - 0.30) * matrix_mix,
        1.0,
        0.28 + (0.83 - 0.28) * matrix_mix,
    ]
}

/// Per-encode dispatch table.  The individual modules still own their `OnceLock`
/// selectors, but hot inner loops receive these already-resolved function
/// references instead of touching a static guard for every block / band / token.
pub(crate) struct EncodingContext {
    pub(crate) thread_pool: ThreadPool,
    pub(crate) speed: Speed,
    pub(crate) boost: Option<DarkAqConfig>,
    pub(crate) xyb: xyb::XybMatrix,
    /// Transform-merge knobs resolved at this encodes distance.
    pub(crate) merge: ac_strategy::MergeTuning,
    base_matrices: &'static DequantMatrices,
    sat_matrices: &'static DequantMatrices,
    plain_hq_matrices: &'static DequantMatrices,
    /// Set once per frame (post-XYB, before any table use) by the
    /// chroma-saturation gate in `frame::encode_frame`.
    chroma_heavy: std::sync::atomic::AtomicBool,
    x_heavy: std::sync::atomic::AtomicBool,
    /// Blue-axis twin of `x_heavy`: structure lives in B−Y where luma-driven
    /// masking cannot see it (saturated blue detail on low-luma ground).
    b_heavy: std::sync::atomic::AtomicBool,
    b_qm_scale: std::sync::atomic::AtomicU32,
    pub(crate) to_xyb_band: xyb::ToXybBandFn,
    pub(crate) fill_quant_field: adaptive_quant::FillQuantFieldFn,
    pub(crate) sse_and_rate: inflated_cost::SseAndRateFn,
    pub(crate) recon_dist_and_rate: inflated_cost::ReconDistAndRateFn,
    pub(crate) recon_error_kernels: inflated_cost::ReconErrorKernels,
    pub(crate) rate_log2_lut: &'static inflated_cost::RateLog2Lut,
    pub(crate) quantize_block_ac: group::QuantizeBlockAcFn,
    pub(crate) quantize_dc: group::QuantizeDcFn,
    pub(crate) quantize_dc_cfl: group::QuantizeDcCflFn,
    pub(crate) apply_quant_field_gain: dark_aq::ApplyQuantFieldGainFn,
    pub(crate) dark_structure_stats: dark_aq::DarkStructureStatsFn,
    pub(crate) fill_blue_tile: dark_aq::FillBlueTileFn,
    pub(crate) block_features: structure_aq::BlockFeaturesFn,
    pub(crate) apply_structure_corrections: structure_aq::ApplyCorrectionsFn,
    pub(crate) apply_cfl: ac_strategy::ApplyCflFn,
    pub(crate) gradient_region_stats: ac_strategy::GradientRegionStatsFn,
    pub(crate) gradient_region_stats_with_chroma: ac_strategy::GradientRegionStatsFn,
    pub(crate) cfl_regression: color_correlation::CflRegressionFn,
    pub(crate) cfl_rdo_block: color_correlation::CflRdoBlockFn,
    pub(crate) cfl_rdo_stats: color_correlation::CflRdoStatsFn,
    pub(crate) x_gradient_sums: frame::XGradientSumsFn,
    pub(crate) fill_ytob_row: color_correlation::FillYtobRowFn,
    pub(crate) accumulate_ytob_weights: color_correlation::AccumulateYtobWeightsFn,
    pub(crate) fill_ytob_residuals: color_correlation::FillYtobResidualsFn,

    pub(crate) idct: &'static dct::IdctMethods,
    pub(crate) dct8x8: &'static dct::DctFn<8, 8, 64>,
    pub(crate) dct8x16: &'static dct::DctFn<16, 8, 128>,
    pub(crate) dct16x8: &'static dct::DctFn<8, 16, 128>,
    pub(crate) dct16x16: &'static dct::DctFn<16, 16, 256>,
    pub(crate) dct4x4: &'static dct::DctFn<8, 8, 64>,
    pub(crate) dct4x8: &'static dct::DctFn<8, 8, 64>,
    pub(crate) dct8x4: &'static dct::DctFn<8, 8, 64>,
    pub(crate) dct32x32: &'static dct::DctFn<32, 32, 1024>,
    pub(crate) dct64x64: &'static dct::DctFn<64, 64, 4096>,
    pub(crate) dct64x32: &'static dct::DctFn<32, 64, 2048>,
    pub(crate) dct32x64: &'static dct::DctFn<64, 32, 2048>,
    pub(crate) dct32x16: &'static dct::DctFn<16, 32, 512>,
    pub(crate) dct16x32: &'static dct::DctFn<32, 16, 512>,
    pub(crate) dc_from_dct32x32: dct::DcFromDct32x32Fn,
    pub(crate) dc_from_dct32x16: dct::DcFromDct32x16Fn,
    pub(crate) dc_from_dct16x32: dct::DcFromDct16x32Fn,
    pub(crate) dc_from_dct64x64: dct::DcFromDct64x64Fn,
    pub(crate) dc_from_dct64x32: dct::DcFromDct64x32Fn,
    pub(crate) dc_from_dct32x64: dct::DcFromDct32x64Fn,
    pub(crate) afv0: afv::AfvFn,
    pub(crate) afv1: afv::AfvFn,
    pub(crate) afv2: afv::AfvFn,
    pub(crate) afv3: afv::AfvFn,
}

impl EncodingContext {
    /// The dequant tables for the current frame: the distance-tier default,
    /// the saturated-content variant when the chroma gate fired, or the plain
    /// sub-d=0.3 set when the X-gradient gate fired (it outranks the chroma
    /// gate: both fire on saturated red content, where flattened B bands are
    /// a strict RD loss).
    #[inline]
    pub(crate) fn matrices(&self) -> &'static DequantMatrices {
        if self.x_heavy.load(std::sync::atomic::Ordering::Relaxed) {
            self.plain_hq_matrices
        } else if self.chroma_heavy.load(std::sync::atomic::Ordering::Relaxed) {
            self.sat_matrices
        } else {
            self.base_matrices
        }
    }

    pub(crate) fn set_chroma_heavy(&self, heavy: bool) {
        self.chroma_heavy
            .store(heavy, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn x_heavy(&self) -> bool {
        self.x_heavy.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn set_x_heavy(&self, heavy: bool) {
        self.x_heavy
            .store(heavy, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn b_heavy(&self) -> bool {
        self.b_heavy.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn set_b_heavy(&self, heavy: bool) {
        self.b_heavy
            .store(heavy, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn raise_b_qm_scale(&self, scale: u32) {
        self.b_qm_scale
            .fetch_max(scale.clamp(2, 7), std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn b_qm_scale(&self) -> u32 {
        self.b_qm_scale.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn b_qm_mul(&self) -> f32 {
        1.25f32.powf(self.b_qm_scale() as f32 - 2.0)
    }

    /// Distortion weights fitted at the two validated opsin endpoints and
    /// interpolated for the signalled B row. Keeping the spec weights while
    /// using the blue-biased row under-prices the channel whose energy that
    /// row deliberately increases.
    #[inline]
    pub(crate) fn channel_weights(&self) -> [f32; 3] {
        channel_weights_for_bias(self.xyb.fwd[8])
    }

    #[inline]
    pub(crate) fn channel_weight(&self, channel: usize) -> f32 {
        self.channel_weights()[channel]
    }

    pub(crate) fn new(
        speed: Speed,
        boost: Option<DarkAqConfig>,
        xyb: xyb::XybMatrix,
        distance: f32,
        num_threads: usize,
    ) -> Self {
        let quantize_dc = group::selected_quantize_dc_methods();
        let dc_from_dct = dct::selected_dc_from_dct_methods();
        let afv = afv::selected_afv_methods();
        Self {
            thread_pool: ThreadPool::new(num_threads),
            speed,
            boost,
            xyb,
            merge: ac_strategy::MergeTuning::new(distance),
            base_matrices: DequantMatrices::new(distance),
            sat_matrices: DequantMatrices::new_saturated(distance),
            plain_hq_matrices: DequantMatrices::new(0.0),
            chroma_heavy: std::sync::atomic::AtomicBool::new(false),
            x_heavy: std::sync::atomic::AtomicBool::new(false),
            b_heavy: std::sync::atomic::AtomicBool::new(false),
            b_qm_scale: std::sync::atomic::AtomicU32::new(2),
            to_xyb_band: xyb::selected_to_xyb_band_fn(),
            fill_quant_field: adaptive_quant::selected_fill_quant_field_fn(),
            sse_and_rate: inflated_cost::selected_sse_and_rate_fn(),
            recon_dist_and_rate: inflated_cost::select_recon_dist_and_rate_fn(),
            recon_error_kernels: inflated_cost::ReconErrorKernels {
                gradient_energy: inflated_cost::select_error_gradient_energy_fn(),
                gradient_peak_energy: inflated_cost::select_error_gradient_peak_energy_fn(),
                combine: inflated_cost::select_combine_error_fn(),
                rgb_hue_chroma_edge_loss: inflated_cost::select_rgb_hue_chroma_edge_loss_fn(),
            },
            rate_log2_lut: inflated_cost::rate_log2_lut(),
            quantize_block_ac: group::selected_quantize_block_ac_fn(),
            quantize_dc: quantize_dc.quantize,
            quantize_dc_cfl: quantize_dc.quantize_cfl,
            apply_quant_field_gain: dark_aq::select_apply_quant_field_gain_fn(),
            dark_structure_stats: dark_aq::select_dark_structure_stats_fn(),
            fill_blue_tile: dark_aq::select_fill_blue_tile_fn(),
            block_features: structure_aq::select_block_features_fn(),
            apply_structure_corrections: structure_aq::select_apply_corrections_fn(),
            apply_cfl: ac_strategy::selected_apply_cfl_fn(),
            gradient_region_stats: ac_strategy::select_gradient_region_stats_fn(),
            gradient_region_stats_with_chroma:
                ac_strategy::select_gradient_region_stats_with_chroma_fn(),
            cfl_regression: color_correlation::selected_cfl_regression_fn(),
            cfl_rdo_block: color_correlation::selected_cfl_rdo_block_fn(),
            cfl_rdo_stats: color_correlation::selected_cfl_rdo_stats_fn(),
            x_gradient_sums: frame::selected_x_gradient_sums_fn(),
            fill_ytob_row: color_correlation::selected_fill_ytob_row_fn(),
            accumulate_ytob_weights: color_correlation::selected_accumulate_ytob_weights_fn(),
            fill_ytob_residuals: color_correlation::selected_fill_ytob_residuals_fn(),

            idct: dct::selected_idct_methods(),
            dct8x8: dct::selected_dct8x8(),
            dct8x16: dct::selected_dct8x16(),
            dct16x8: dct::selected_dct16x8(),
            dct16x16: dct::selected_dct16x16(),
            dct4x4: dct::selected_dct4x4(),
            dct4x8: dct::selected_dct4x8(),
            dct8x4: dct::selected_dct8x4(),
            dct32x32: dct::selected_dct32x32(),
            dct64x64: dct::selected_dct64x64(),
            dct64x32: dct::selected_dct64x32(),
            dct32x64: dct::selected_dct32x64(),
            dct32x16: dct::selected_dct32x16(),
            dct16x32: dct::selected_dct16x32(),
            dc_from_dct32x32: dc_from_dct.dct32x32,
            dc_from_dct32x16: dc_from_dct.dct32x16,
            dc_from_dct16x32: dc_from_dct.dct16x32,
            dc_from_dct64x64: dc_from_dct.dct64x64,
            dc_from_dct64x32: dc_from_dct.dct64x32,
            dc_from_dct32x64: dc_from_dct.dct32x64,
            afv0: afv.afv0,
            afv1: afv.afv1,
            afv2: afv.afv2,
            afv3: afv.afv3,
        }
    }
}

impl Default for EncodingContext {
    #[inline]
    fn default() -> Self {
        Self::new(Speed::Fast, None, xyb::XybMatrix::SPEC, 1.0, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::channel_weights_for_bias;

    #[test]
    fn channel_weights_follow_the_opsin_b_row() {
        assert_eq!(
            channel_weights_for_bias(crate::xyb::B_BIAS),
            [0.30, 1.0, 0.28]
        );
        let strong = channel_weights_for_bias(0.85);
        assert!((strong[0] - 0.10).abs() < 1e-6);
        assert_eq!(strong[1], 1.0);
        assert!((strong[2] - 0.83).abs() < 1e-6);

        let mid = channel_weights_for_bias((crate::xyb::B_BIAS + 0.85) * 0.5);
        assert!((mid[0] - 0.20).abs() < 1e-6);
        assert!((mid[2] - 0.555).abs() < 1e-6);
    }
}
