/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Content-dependent quantizers for transform selection only.
//!
//! Scaling the whole RD cost also scales metadata and misses coefficient
//! threshold crossings. Instead, adjust the candidate's quantizer using the
//! energy of fine luma detail that its coding quantizer would erase. Coding
//! matrices, entropy quant classes and quant-field refinement stay unchanged.

use super::*;

/// The neutral region is essential: a large erased *fraction* of a smooth
/// residual is not necessarily visible detail. Normalize absolute energy by
/// distance squared so the threshold follows the quantization noise floor.
const VISIBLE_ENERGY: f32 = 2e-4;
const MAX_BOOST: f32 = 0.6;
/// ExtraSlow needs an area budget for the denser layouts this rule can choose.
/// Its first deployment is restricted to very busy, saturated content.
const BUSY_TEXTURE: f32 = 0.06;

pub(crate) fn enabled(ctx: &EncodingContext, distance: f32) -> bool {
    if ctx.arbiter != crate::Arbiter::Ssimulacra2
        || !ctx.coding.is_xyb()
        || !(1.0..=4.0).contains(&distance)
    {
        return false;
    }
    // The fast tiers compute no chroma statistics; the erased-energy test
    // gates by itself there.
    if matches!(ctx.speed, crate::Speed::Fast | crate::Speed::Fastest) {
        return true;
    }
    if !ctx.chroma_heavy() || ctx.screen_content() {
        return false;
    }
    match ctx.speed {
        crate::Speed::Medium | crate::Speed::MediumSlow => true,
        // Slow's fine-quality exact DP is already calibrated. The busy gate
        // also leaves smooth saturated art on its existing selection path.
        crate::Speed::Slow => {
            distance >= crate::quant_weights::QM_SS2_MIN_DISTANCE
                && ctx.luma_texture() >= BUSY_TEXTURE
        }
        #[cfg(feature = "extraslow")]
        crate::Speed::ExtraSlow => distance >= 1.5 && ctx.luma_texture() >= BUSY_TEXTURE,
        #[cfg(not(feature = "extraslow"))]
        crate::Speed::ExtraSlow => {
            distance >= crate::quant_weights::QM_SS2_MIN_DISTANCE
                && ctx.luma_texture() >= BUSY_TEXTURE
        }
        _ => false,
    }
}

pub(super) fn scale(
    ctx: &EncodingContext,
    strategy: u8,
    luma: &[f32],
    qac: f32,
    distance: f32,
    metadata: f32,
) -> f32 {
    // Zero-metadata calls measure/refine the actual coding quantizer. The
    // 16 family remains at stock calibration: it failed independent controls.
    if metadata == 0.0
        || !matches!(
            strategy,
            STRATEGY_DCT32X32 | STRATEGY_DCT32X16 | STRATEGY_DCT16X32
        )
        || !enabled(ctx, distance)
    {
        return 1.0;
    }
    let (cx, cy) = (
        AcStrategyImage::covered_blocks_x_of(strategy),
        AcStrategyImage::covered_blocks_y_of(strategy),
    );
    // Rectangular transforms store both orientations in the same wide
    // coefficient layout, irrespective of their footprint in the image.
    let erased = erased_detail(
        luma,
        inverse_matrix_for(ctx, strategy, 1),
        qac,
        cx.max(cy),
        cx.min(cy),
    );
    1.0 + MAX_BOOST * (erased / (VISIBLE_ENERGY * distance * distance) - 1.0).clamp(0.0, 1.0)
}

fn erased_detail(luma: &[f32], inverse: &[f32], qac: f32, cx: usize, cy: usize) -> f32 {
    let width = cx * 8;
    let mut energy = 0.0f64;
    for y in 0..cy * 8 {
        for x in 0..width {
            // The first quarter of each frequency axis, including every
            // LLF/DC coefficient, does not justify fine-detail protection.
            if x < cx * 2 && y < cy * 2 {
                continue;
            }
            let i = y * width + x;
            if (luma[i] * inverse[i] * qac).abs() < 0.5 {
                energy += f64::from(luma[i]).powi(2);
            }
        }
    }
    energy as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "extraslow"))]
    #[test]
    fn featureless_extraslow_retains_the_slow_selection_policy() {
        for distance in [1.0, 1.5, 2.25, 4.0] {
            let active = [crate::Speed::Slow, crate::Speed::ExtraSlow].map(|speed| {
                let ctx = EncodingContext::new(
                    speed,
                    crate::Arbiter::Ssimulacra2,
                    crate::xyb::XybMatrix::SPEC,
                    distance,
                    1,
                );
                ctx.set_chroma_heavy(true);
                ctx.set_luma_texture(0.07);
                enabled(&ctx, distance)
            });
            assert_eq!(active[0], active[1], "distance {distance}");
        }
    }

    #[test]
    fn rectangular_orientations_use_the_same_canonical_coefficient_layout() {
        let ctx = EncodingContext::new(
            crate::Speed::Medium,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            1.0,
            1,
        );
        ctx.set_chroma_heavy(true);
        assert!(enabled(&ctx, 1.0));
        let mut luma = [0.0; 512];
        // Both rectangle orientations store coefficients 32 wide. Index 36
        // is (4, 1), inside the protected low-frequency 8x4 rectangle. Reading
        // the tall footprint's 16-wide layout would misclassify it as detail.
        luma[36] = 0.02;
        let qac = 0.25 / (luma[36] * inverse_matrix_for(&ctx, STRATEGY_DCT32X16, 1)[36]);
        for strategy in [STRATEGY_DCT32X16, STRATEGY_DCT16X32] {
            assert_eq!(scale(&ctx, strategy, &luma, qac, 1.0, 1.0), 1.0);
        }
        // The first horizontal frequency outside that region must still
        // activate the feature, identically in the two orientations.
        luma[36] = 0.0;
        luma[8] = 0.02;
        let qac = 0.25 / (luma[8] * inverse_matrix_for(&ctx, STRATEGY_DCT32X16, 1)[8]);
        let tall = scale(&ctx, STRATEGY_DCT32X16, &luma, qac, 1.0, 1.0);
        let wide = scale(&ctx, STRATEGY_DCT16X32, &luma, qac, 1.0, 1.0);
        assert!(tall > 1.0);
        assert_eq!(tall, wide);
    }

    #[test]
    fn erased_detail_excludes_low_frequencies_and_surviving_coefficients() {
        for (cx, cy) in [(4, 4), (4, 2), (2, 4)] {
            let width = cx * 8;
            let mut luma = vec![0.0; cx * cy * 64];
            let inverse = vec![1.0; luma.len()];
            // Even an arbitrarily strong DC/ramp must not activate the rule.
            luma[0] = 100.0;
            luma[(cy * 2 - 1) * width + cx * 2 - 1] = 100.0;
            assert_eq!(erased_detail(&luma, &inverse, 1.0, cx, cy), 0.0);
            // Fine detail is eligible only while the coding quantizer erases it.
            luma[cx * 2] = 0.25;
            assert_eq!(erased_detail(&luma, &inverse, 1.0, cx, cy), 0.0625);
            assert_eq!(erased_detail(&luma, &inverse, 2.0, cx, cy), 0.0);
        }
    }
}
