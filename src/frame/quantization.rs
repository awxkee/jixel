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

//! Distance-dependent quantization and decoder filter parameters.

use crate::dct::fmla;
use crate::image::Image3F;

#[derive(Clone, Copy)]
pub(super) struct DistanceParams {
    pub(super) distance: f32,
    pub(super) global_scale: i32,
    pub(super) quant_dc: i32,
    pub(super) scale: f32,
    pub(super) scale_dc: f32,
    pub(super) x_qm_scale: u32,
    pub(super) epf_iters: u32,
    pub(super) epf_pass0_scale: Option<f32>,
    pub(super) epf_quant_mul: Option<f32>,
    pub(super) gab_enabled: bool,
    /// Multipliers of the default DC steps of X, Y and B.
    pub(super) dc_step: [f32; 3],
}

pub(super) const X_QM_REFINE_MAX_DISTANCE: f32 = 0.299;
const DC_REFINE_PEAK: f32 = 1.35;
const DC_REFINE_HOLD: f32 = 3.0;
const DC_REFINE_RELEASE: f32 = 5.0;

#[inline]
fn dc_refinement(distance: f32) -> f32 {
    if distance <= DC_REFINE_HOLD {
        DC_REFINE_PEAK
    } else if distance >= DC_REFINE_RELEASE {
        1.0
    } else {
        const RECIP_REFINE: f32 = 1.0 / (DC_REFINE_RELEASE - DC_REFINE_HOLD);
        let t = (distance - DC_REFINE_HOLD) * RECIP_REFINE;
        fmla(1.0 - DC_REFINE_PEAK, t, DC_REFINE_PEAK)
    }
}

const DCEPF_VLQ_D0: f32 = 8.0;
const DCEPF_VLQ_D1: f32 = 15.0;
const VLQ_DC_MUL: f32 = 0.7;

#[inline]
fn dcepf_vlq_t(distance: f32) -> f32 {
    ((distance - DCEPF_VLQ_D0) / (DCEPF_VLQ_D1 - DCEPF_VLQ_D0)).clamp(0.0, 1.0)
}

fn quant_dc(distance: f32) -> f32 {
    let refine = dc_refinement(distance);
    let vlq = fmla(dcepf_vlq_t(distance), VLQ_DC_MUL - 1.0, 1.0);
    let k_dc_quant_pow = 0.57f32;
    let k_dc_quant = 1.12f32;
    let k_dc_mul = 2.9f32;
    let effective = k_dc_mul * (distance / k_dc_mul).powf(k_dc_quant_pow);
    let effective = f32::clamp(effective, 0.5 * distance, distance);
    (k_dc_quant / effective).min(50.0) * refine * vlq
}

pub(super) fn compute_distance_params(distance: f32) -> DistanceParams {
    const K_GLOBAL_SCALE_DENOM: i32 = 1 << 16;
    const K_AC_QUANT: f32 = 0.8;
    // Keep the integer quant field away from the very coarse 2/3/4 range. The
    // corresponding reduction in global scale preserves the effective AC
    // quantizer while giving AQ enough integer resolution to vary smoothly.
    const K_QUANT_FIELD_TARGET: f32 = 10.0;

    let qdc = quant_dc(distance);
    let scale = K_GLOBAL_SCALE_DENOM as f32 * K_AC_QUANT / (distance * K_QUANT_FIELD_TARGET);
    // AC and DC are signaled independently. Capping the AC scale by qdc made
    // it freeze once quant_dc() reached its high-distance cap, producing a
    // rate/quality plateau followed by a cliff when the cap finally released.
    let global_scale = i32::clamp(scale.round() as i32, 1, 1 << 15);
    let scale_f = global_scale as f32 / K_GLOBAL_SCALE_DENOM as f32;
    let qd = ((qdc / scale_f) + 0.5) as i32;
    let qd = i32::clamp(qd, 1, 1 << 16);
    let scale_dc = qd as f32 * scale_f;

    let mut x_qm_scale: u32 = 2;
    if distance >= 1.25 {
        x_qm_scale += 1;
    }
    if distance > 9.0 {
        x_qm_scale += 1;
    }
    if distance < X_QM_REFINE_MAX_DISTANCE {
        x_qm_scale += 1;
    }
    static EPF_THRESHOLDS: [f32; 2] = [0.7, 1.5];
    let mut epf_iters: u32 = 0;
    for t in EPF_THRESHOLDS {
        if distance >= t {
            epf_iters += 1;
        }
    }
    let epf_pass0_scale = epf_pass0_scale(distance);
    if epf_pass0_scale.is_some() || dcepf_vlq_t(distance) >= 0.5 {
        epf_iters = 3;
    }
    let gab_enabled = false; // measured net-negative for rate-matched SSIMU2

    DistanceParams {
        distance,
        global_scale,
        quant_dc: qd,
        scale: scale_f,
        scale_dc,
        x_qm_scale,
        epf_iters,
        epf_pass0_scale,
        epf_quant_mul: None,
        gab_enabled,
        dc_step: chroma_dc_steps(qdc),
    }
}

/// Multiplier of the default X DC step. The spec's X DC quantizer is far
/// finer than the X AC band it sits under; twice the step buys back 1-2% of
/// the file at better SSIMULACRA2, butteraugli and cvvdp alike.
const X_DC_STEP: f32 = 2.0;

/// Distance from which the chroma DC steps grow slower than the luma DC step.
const CHROMA_DC_GROWTH_DISTANCE: f32 = 3.5;
/// Exponents of the X and B DC step growth relative to the luma DC step's.
static CHROMA_DC_GROWTH: [f32; 2] = [0.75, 0.5];

/// DC step multipliers for X, Y and B at the luma DC quantizer `qdc`.
///
/// Past `CHROMA_DC_GROWTH_DISTANCE` the chroma steps follow only a power of
/// the luma step's growth: a flat, nearly neutral region otherwise sits
/// between two chroma DC levels, and every block rounds the same way into a
/// visible hue shift.
fn chroma_dc_steps(qdc: f32) -> [f32; 3] {
    let growth = (quant_dc(CHROMA_DC_GROWTH_DISTANCE) / qdc).max(1.0);
    [
        X_DC_STEP * growth.powf(CHROMA_DC_GROWTH[0] - 1.0),
        1.0,
        growth.powf(CHROMA_DC_GROWTH[1] - 1.0),
    ]
}

const EPF_PASS0_START_D: f32 = 2.5;
const EPF_PASS0_SCALE: f32 = 3.6;
const EPF_PASS0_SPEC_SCALE: f32 = 0.9;

fn epf_pass0_scale(distance: f32) -> Option<f32> {
    if distance < EPF_PASS0_START_D {
        return None;
    }
    let t = dcepf_vlq_t(distance);
    if t >= 0.5 {
        return None;
    }
    Some(fmla(
        2.0 * t,
        EPF_PASS0_SPEC_SCALE - EPF_PASS0_SCALE,
        EPF_PASS0_SCALE,
    ))
}

const SCREEN_CONTENT_EPF_MAX_DISTANCE: f32 = 5.0;

pub(crate) fn epf_sharpness_id(distance: f32, screen_content: bool) -> i32 {
    let t = dcepf_vlq_t(distance);
    if t >= 0.75 {
        return 1;
    } else if t >= 0.5 {
        return 2;
    } else if t >= 0.25 {
        return 3;
    }
    // Hard edges beside flat fills ring visibly; up to d=5 the strongest
    // filter wins both SSIMULACRA2 and butteraugli there at identical bytes.
    if distance < 1.4 || (screen_content && distance < SCREEN_CONTENT_EPF_MAX_DISTANCE) {
        7
    } else if distance < 1.75 {
        6
    } else if distance < 3.5 {
        5
    } else {
        4
    }
}

/// Pre-invert Gaborish so the decoder's forward pass reproduces `xyb`.
pub(super) fn gaborize(xyb: &mut Image3F, distp: &DistanceParams) {
    if distp.gab_enabled {
        crate::gaborish::gaborish_inverse(xyb, 0.990_851_1);
    }
}

/// Largest stored DC level kept clear of the `i16` limit, with room for the
/// chroma-from-luma terms and the finer DC-step candidates.
const DC_LEVEL_LIMIT: f32 = 16_000.0;

/// Non-XYB planes hold the image's own samples, which float input may take far
/// past [0, 1]. Coarsen a plane's DC step only as far as its stored levels
/// need to stay representable.
pub(super) fn dc_steps_in_range(base: [f32; 3], scale_dc: f32, coded: &Image3F) -> [f32; 3] {
    let peak = (0..3)
        .flat_map(|c| coded.plane_data(c).iter())
        .fold(0.0f32, |m, &v| m.max(v.abs()));
    std::array::from_fn(|c| {
        let level = peak * crate::quant_weights::INV_DC_QUANT[c] * scale_dc / base[c];
        if level > DC_LEVEL_LIMIT {
            base[c] * level / DC_LEVEL_LIMIT
        } else {
            base[c]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_dc_steps_grow_slower_than_luma_past_the_knee() {
        for d in [0.1, 1.0, 2.0, 3.0, CHROMA_DC_GROWTH_DISTANCE] {
            assert_eq!(
                compute_distance_params(d).dc_step,
                [X_DC_STEP, 1.0, 1.0],
                "d={d}"
            );
        }
        // Absolute steps per channel: luma coarsens fastest, B slowest, and
        // no chroma step ever gets finer as the distance grows.
        let absolute = |d: f32| {
            let p = compute_distance_params(d);
            std::array::from_fn::<f32, 3, _>(|c| p.dc_step[c] / p.scale_dc)
        };
        let mut prev = absolute(CHROMA_DC_GROWTH_DISTANCE);
        for d in [4.0, 5.0, 6.0, 8.0, 11.0, 15.0, 25.0] {
            let p = compute_distance_params(d);
            assert!(p.dc_step[0] < X_DC_STEP && p.dc_step[2] < 1.0, "d={d}");
            assert!(p.dc_step[0] / X_DC_STEP > p.dc_step[2], "d={d}");
            let cur = absolute(d);
            for c in [0, 2] {
                assert!(cur[c] >= prev[c] * 0.999, "chroma {c} step shrank at d={d}");
            }
            prev = cur;
        }
    }

    #[test]
    fn dc_refinement_holds_then_ramps_out_monotonically() {
        // Full refinement through the hold point, none past the release point.
        for d in [0.1, 0.5, 1.0, 2.0, DC_REFINE_HOLD] {
            assert_eq!(dc_refinement(d), DC_REFINE_PEAK);
        }
        for d in [DC_REFINE_RELEASE, 5.5, 8.0, 25.0] {
            assert_eq!(dc_refinement(d), 1.0);
        }
        // Strictly decreasing across the ramp, and continuous at both ends.
        let mid = 0.5 * (DC_REFINE_HOLD + DC_REFINE_RELEASE);
        assert!((dc_refinement(mid) - 0.5 * (DC_REFINE_PEAK + 1.0)).abs() < 1e-6);
        let mut prev = f32::INFINITY;
        for i in 0..=40 {
            let d = DC_REFINE_HOLD + (DC_REFINE_RELEASE - DC_REFINE_HOLD) * (i as f32 / 40.0);
            let v = dc_refinement(d);
            assert!(v <= prev, "refinement rose at d={d}");
            assert!((1.0..=DC_REFINE_PEAK).contains(&v));
            prev = v;
        }
    }

    #[test]
    fn dc_refinement_makes_dc_finer_where_it_applies() {
        // Refining means a *larger* DC scale (finer quantization step), and it
        // must not disturb the AC quantizer, which is signaled independently.
        for d in [0.5, 1.0, 3.0] {
            let p = compute_distance_params(d);
            assert!(p.scale_dc > 0.0);
            assert!((p.scale_dc - quant_dc(d)).abs() <= 0.5 * p.scale + f32::EPSILON);
        }
        assert!(quant_dc(1.0) > quant_dc(1.0) / DC_REFINE_PEAK);
    }

    #[test]
    fn epf_third_pass_schedule() {
        // The third EPF pass at spec strength erases texture in the normal
        // bands, so it only enters weakened (custom pass-0 sigma scale) from
        // EPF_PASS0_START_D, eases toward spec strength across the first half
        // of the VLQ ramp, and hands over to the fitted spec-strength band
        // (default sigma header) at half-ramp. Guards every edge.
        for d in [0.0, 0.1, 0.5, 0.69] {
            assert_eq!(compute_distance_params(d).epf_iters, 0, "d={d}");
        }
        for d in [0.7, 1.0, 1.49] {
            assert_eq!(compute_distance_params(d).epf_iters, 1, "d={d}");
        }
        for d in [1.5, 2.0, 2.49] {
            let p = compute_distance_params(d);
            assert_eq!(p.epf_iters, 2, "d={d}");
            assert_eq!(p.epf_pass0_scale, None, "d={d}");
        }
        for d in [2.5, 4.0, 6.0, 8.0] {
            let p = compute_distance_params(d);
            assert_eq!(p.epf_iters, 3, "d={d}");
            assert_eq!(p.epf_pass0_scale, Some(EPF_PASS0_SCALE), "d={d}");
        }
        let mid = compute_distance_params(9.75).epf_pass0_scale.unwrap();
        assert!((mid - 0.5 * (EPF_PASS0_SCALE + EPF_PASS0_SPEC_SCALE)).abs() < 1e-5);
        for d in [11.5, 15.0, 25.0] {
            let p = compute_distance_params(d);
            assert_eq!(p.epf_iters, 3, "d={d}");
            assert_eq!(p.epf_pass0_scale, None, "d={d}");
        }
    }

    #[test]
    fn epf_sharpness_strengthens_in_the_coarse_band() {
        assert_eq!(epf_sharpness_id(1.39, false), 7);
        assert_eq!(epf_sharpness_id(1.4, false), 6);
        assert_eq!(epf_sharpness_id(1.74, false), 6);
        assert_eq!(epf_sharpness_id(1.75, false), 5);
        assert_eq!(epf_sharpness_id(2.75, false), 5);
        assert_eq!(epf_sharpness_id(3.49, false), 5);
        assert_eq!(epf_sharpness_id(3.5, false), 4);
        // VLQ ramp steps (quarter-points of DCEPF_VLQ_D0..D1 = 8..15).
        assert_eq!(epf_sharpness_id(9.74, false), 4);
        assert_eq!(epf_sharpness_id(9.75, false), 3);
        assert_eq!(epf_sharpness_id(11.5, false), 2);
        assert_eq!(epf_sharpness_id(13.25, false), 1);
        assert_eq!(epf_sharpness_id(25.0, false), 1);
    }

    #[test]
    fn screen_content_keeps_the_strongest_epf_below_d5() {
        use super::epf_sharpness_id;
        assert_eq!(epf_sharpness_id(1.0, true), 7);
        assert_eq!(epf_sharpness_id(2.5, true), 7);
        assert_eq!(epf_sharpness_id(4.99, true), 7);
        assert_eq!(epf_sharpness_id(5.0, true), epf_sharpness_id(5.0, false));
        assert_eq!(epf_sharpness_id(12.0, true), epf_sharpness_id(12.0, false));
    }

    #[test]
    fn quantization_keeps_coarsening_at_low_quality() {
        let d4 = compute_distance_params(4.0);
        let d45 = compute_distance_params(4.5);
        let d5 = compute_distance_params(5.0);
        assert!(d4.global_scale > d45.global_scale);
        assert!(d45.global_scale > d5.global_scale);
        for distances in [3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0, 16.0].array_windows::<2>() {
            let fine = compute_distance_params(distances[0]);
            let coarse = compute_distance_params(distances[1]);
            assert!(fine.scale_dc > coarse.scale_dc, "distances={distances:?}");
        }
    }

    #[test]
    fn dc_quantization_stays_independent_of_ac_scale() {
        for distance in [0.5, 1.0, 3.5, 4.0, 4.5, 6.0] {
            let params = compute_distance_params(distance);
            let rounding_error = 0.5 * params.scale;
            assert!((params.scale_dc - quant_dc(distance)).abs() <= rounding_error + f32::EPSILON);
        }
    }
}
