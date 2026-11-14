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

//! Transform, quantize and serialize the selected VarDCT frame.

use super::ac_entropy::{
    ORDER_RACE_REFINE_MARGIN, OrderArm, encode_ac_sections, encode_order_arm, finalize_ac_entropy,
};
#[cfg(not(feature = "extraslow"))]
use super::ac_groups::RefinedAcGroup;
use super::ac_groups::{
    AcGroupCapture, PendingAcGroup, ac_qf_threshold, paste_group, process_ac_group,
};
#[cfg(feature = "splines")]
use super::dc_groups::provisional_dc_group;
use super::dc_groups::{assemble_source_dc, merge_quant_dc, setup_dc_group};
use super::dc_smoothing::{
    choose_chroma_dc_coding, compensate_b_dc, flatten_chroma_dc, skip_dc_smoothing,
};
use super::extra_channels::{prepare_extra_sections, prepare_extra_sections_lossless};
use super::header::{
    used_quant_table_slots, write_ac_global, write_dc_global, write_frame_header_kind,
};
#[cfg(feature = "extraslow")]
use super::quant_field_loop::run_quant_field_loop;
#[cfg(not(feature = "extraslow"))]
use super::quantization::DistanceParams;
use super::quantization::{compute_distance_params, dc_steps_in_range, epf_sharpness_id};
use super::sections::{combine_sections_ordered, group_section_order};
use super::{
    ImageDim, K_BLOCK_DIM, K_DC_GROUP_DIM, K_GROUP_DIM_IN_BLOCKS, dc_rdo, dc_smoothing, extra_tree,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::color_correlation::choose_ytob_dc;
use crate::dc_group_data::DcGroupData;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::{EncodingContext, FrameRole};
use crate::entropy::write_token;
use crate::image::Image3F;
use crate::noise::GrainPass;
use crate::patches::VarDctFrameKind;
use crate::util::EncodeError;
use std::borrow::Cow;
use std::sync::Mutex;

#[cfg(test)]
#[path = "vardct_encoder_tests.rs"]
mod tests;

/// Share of the scan walk the custom coefficient orders must remove before
/// the refine pass's prices are rebuilt on them.
const RDOQ_FRESH_PRICES_MIN_SCAN_MOVE: f64 = 0.35;

/// The closed loop settles a frame's B quant scale in the shared context;
/// the next frame, or the next sampled estimate, starts from the configured one.
#[cfg(feature = "extraslow")]
struct RestoreBQmScale<'a>(&'a EncodingContext, u32);

#[cfg(feature = "extraslow")]
impl Drop for RestoreBQmScale<'_> {
    fn drop(&mut self) {
        self.0.set_b_qm_scale(self.1);
    }
}

/// Returns DC-group and AC-section bits, excluding the global headers, so
/// sampled payload can be scaled separately from the frame's fixed costs.
/// What the decoder shows of each pixel of this frame: alpha and an overlay
/// blended over a saved base scale its colors' error, and Replace patches
/// overwrite it before splines are drawn.
#[cfg(feature = "splines")]
fn edge_view(
    ctx: &EncodingContext,
    frame_kind: VarDctFrameKind,
    alpha: Option<&AlphaPlane>,
    width: usize,
    height: usize,
) -> crate::splines::EdgeView {
    let len = width * height;
    let mut weight: Option<Vec<f32>> = None;
    if let Some(alpha) = alpha.filter(|a| a.len() == len && !a.is_constant()) {
        let max = ((1u64 << alpha.bits().min(16)) - 1) as f32;
        weight = Some(
            (0..len)
                .map(|i| {
                    let a = if alpha.is_float() {
                        f32::from_bits(alpha.get_i32(i) as u32).clamp(0.0, 1.0)
                    } else {
                        alpha.get_i32(i) as f32 / max
                    };
                    a * a
                })
                .collect(),
        );
    }
    if let Some(cover) = ctx.overlay_coverage.as_ref().filter(|c| c.len() == len) {
        let weight = weight.get_or_insert_with(|| vec![1.0; len]);
        for (w, &c) in weight.iter_mut().zip(cover) {
            *w *= (1.0 - c) * (1.0 - c);
        }
    }
    let replaced = match frame_kind {
        VarDctFrameKind::Patched(references, ..) => {
            let mut replaced = vec![false; len];
            for reference in references
                .iter()
                .filter(|r| !r.alpha_only && !r.add && !r.mul)
            {
                for &(px, py) in &reference.positions {
                    for y in py.min(height)..(py + reference.height).min(height) {
                        replaced[y * width + px.min(width)
                            ..y * width + (px + reference.width).min(width)]
                            .fill(true);
                    }
                }
            }
            replaced.contains(&true).then_some(replaced)
        }
        _ => None,
    };
    crate::splines::EdgeView { weight, replaced }
}

/// Encode one VarDCT frame. `Grain::Auto` on a displayed frame first codes
/// the frame plainly and reads the measured grain's survival off that
/// reconstruction: when the quantizer keeps the grain the plain coding is
/// kept as well, otherwise the frame is coded again denoised with the
/// decoder's grain replacing what was lost.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_frame_core(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    opsin: Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    #[cfg(feature = "splines")] splines: Option<&crate::splines::SplineSet>,
    #[cfg(feature = "splines")] allow_edge_splines: bool,
    grain_pass: GrainPass,
    writer: &mut BitWriter,
) -> Result<usize, EncodeError> {
    {
        // Color patches keep their atlas in another reference slot and
        // leave the extra channel alone, so they combine with either form.
        let eligible = ctx.function_fields
            && match frame_kind {
                VarDctFrameKind::Regular => true,
                VarDctFrameKind::Patched(references, ..) => {
                    references.iter().all(|r| !r.alpha_only && r.ref_frame != 0)
                }
                VarDctFrameKind::ReferenceOnly { .. } => false,
            }
            && matches!(grain_pass, GrainPass::Inherit)
            && ctx.frame_role == FrameRole::Last
            && (ctx.coding.is_xyb() || ctx.extra_black)
            && ctx.speed >= crate::Speed::Medium
            && ctx.extra_step == 1
            && ctx.grain == crate::Grain::Off
            && !ctx.progressive_dc
            && coeff_shifts == [0]
            && opsin.xsize().saturating_mul(opsin.ysize()) <= crate::function_fields::MAX_PIXELS;
        #[cfg(feature = "splines")]
        let eligible = eligible
            && splines.is_none()
            && !ctx.splines
            && !ctx.edge_splines
            && !ctx.dots
            && ctx.overlay_coverage.is_none();
        if eligible && let Some(source @ (AlphaPlane::U8(_) | AlphaPlane::U16 { .. })) = alpha {
            let (width, height) = (opsin.xsize(), opsin.ysize());
            // Deeper planes take their geometry from the 8-bit fits and keep
            // their own profile, residual and judge.
            let deep = matches!(source, AlphaPlane::U16 { .. });
            let fits: Vec<_> = crate::function_fields::fit(source, width, height)
                .into_iter()
                .filter(|fit| {
                    fit.max_error <= 2
                        && (ctx.extra_squeeze_distance > 0.0 || fit.squared_error == 0)
                })
                .collect();
            // A function that only explains most of the plane: its misfit is
            // coded as an ordinary lossy residual in the displayed frame and
            // added to a base frame holding the function. K plates (no-ink
            // bias) keep the single-frame candidates.
            // A repeated tile is the same construction with an exactly coded
            // residual, so it also serves exact requests.
            let lossy = ctx.extra_squeeze_distance > 0.0;
            let residual_fit =
                (ctx.function_fields_layered && ctx.coding.is_xyb() && !ctx.extra_black)
                    .then(|| {
                        if deep {
                            return crate::function_fields_deep::fit(
                                &ctx.thread_pool,
                                source,
                                width,
                                height,
                                lossy,
                            );
                        }
                        let (periodic, smooth) = ctx.thread_pool.join(
                            || crate::function_fields::fit_periodic(source, width, height),
                            || {
                                lossy
                                    .then(|| {
                                        crate::function_fields::fit_residual(
                                            &ctx.thread_pool,
                                            source,
                                            width,
                                            height,
                                        )
                                    })
                                    .flatten()
                            },
                        );
                        periodic.or(smooth)
                    })
                    .flatten();
            if !fits.is_empty() || residual_fit.is_some() {
                let repeated = !ctx.extra_black
                    && crate::alpha_patches::find_alpha_patches(source, width, height).is_some();
                // The base frame holds the function; its size is part of
                // every candidate that adds a residual to it.
                let based = residual_fit.as_ref().and_then(|fit| {
                    let mut candidate_scratch = CoderScratch::default();
                    let mut based = writer.clone();
                    let start = based.bits_written();
                    // A modular frame of its own, black with the function
                    // in the extra channel: none of a VarDCT frame's
                    // overhead, and 1024-pixel groups. Its coordinate goes
                    // through a second Palette unless the file must stay
                    // readable without explicit deltas.
                    let model = &fit.base.model;
                    crate::modular::portable_profile::encode_frame(
                        &ctx.thread_pool,
                        &model.profile,
                        &model.u,
                        &model.v,
                        !ctx.function_fields_portable,
                        &mut candidate_scratch,
                        &mut based,
                    )?;
                    let bits = based.bits_written() - start;
                    Some((based, bits))
                });
                // (dead zone, distance scale). The function's own misfit and
                // the source's rounding noise are what a fine residual spends
                // its bits on: either drop residuals within the dead zone,
                // which leaves a sparse plane the exact coder takes, or keep
                // them all and quantize, a little finer when the nominal
                // distance scores below the incumbent.
                let smooth = [(2u8, 1.0f32), (0, 1.0), (1, 1.0), (0, 0.8), (0, 0.65)];
                let zoned: Vec<AlphaPlane> = match &residual_fit {
                    Some(fit) if !fit.periodic && based.is_some() => smooth
                        .iter()
                        .filter(|&&(zone, _)| zone != 0)
                        .map(|&(zone, _)| {
                            if deep {
                                crate::function_fields_deep::dead_zone(fit, source, zone)
                            } else {
                                fit.dead_zone(source, zone)
                            }
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                let mut alternatives: Vec<ExtraAlternative<'_>> =
                    fits.iter().map(ExtraAlternative::Nested).collect();
                let mut zone_indices = [None; 3];
                if let (Some(fit), Some(_)) = (&residual_fit, &based) {
                    if fit.periodic {
                        // What a tile misses is sparse and always coded
                        // exactly: a pattern mask the ordinary path keeps
                        // exact stays so.
                        alternatives.push(ExtraAlternative::Plane {
                            plane: &fit.residual,
                            lossless: true,
                            scale: 1.0,
                            based: true,
                        });
                    } else {
                        let mut zones = zoned.iter();
                        for (zone, scale) in smooth {
                            if scale == 1.0 {
                                zone_indices[[1, 2, 0][usize::from(zone)]] =
                                    Some(alternatives.len());
                            }
                            alternatives.push(ExtraAlternative::Plane {
                                plane: if zone == 0 {
                                    &fit.residual
                                } else {
                                    zones.next().expect("one plane per dead zone")
                                },
                                lossless: false,
                                scale,
                                based: true,
                            });
                        }
                    }
                }
                // The exact coding a repeated mask is raced against in the
                // ordinary path.
                let exact_source = (repeated && lossy && ctx.speed >= crate::Speed::Slow)
                    .then_some(alternatives.len());
                if exact_source.is_some() {
                    alternatives.push(ExtraAlternative::Plane {
                        plane: source,
                        lossless: true,
                        scale: 1.0,
                        based: false,
                    });
                }
                let color = &opsin;
                let incumbent_scores = super::function_quality::GateCache::default();
                let deep_scores = crate::function_fields_deep::GateCache::default();
                let judge = |index: usize, incumbent: &AlphaPlane, coded: &AlphaPlane| {
                    if Some(index) == exact_source {
                        return Some(0.0);
                    }
                    match (&alternatives[index], &residual_fit) {
                        (ExtraAlternative::Nested(fit), _) => super::function_quality::score(
                            ctx,
                            color,
                            source,
                            incumbent,
                            &fit.reconstructed,
                            false,
                            &incumbent_scores,
                        ),
                        (ExtraAlternative::Plane { .. }, Some(fit)) if deep => {
                            crate::function_fields_deep::compose(fit, coded).and_then(|decoded| {
                                crate::function_fields_deep::score(
                                    ctx,
                                    color,
                                    source,
                                    incumbent,
                                    &decoded,
                                    &deep_scores,
                                )
                            })
                        }
                        (ExtraAlternative::Plane { .. }, Some(fit)) => {
                            fit.compose(coded).and_then(|decoded| {
                                super::function_quality::score(
                                    ctx,
                                    color,
                                    source,
                                    incumbent,
                                    &decoded,
                                    true,
                                    &incumbent_scores,
                                )
                            })
                        }
                        _ => None,
                    }
                };
                let (based_writer, base_bits) = match based {
                    Some((writer, bits)) => (Some(writer), bits),
                    None => (None, 0),
                };
                let race = ExtraRace {
                    alternatives: &alternatives,
                    base_bits,
                    judge: &judge,
                    based: std::sync::Mutex::new(based_writer),
                    chosen: std::sync::Mutex::new(None),
                    zones: match zone_indices {
                        [Some(strong), Some(none), Some(weak)] => Some([strong, none, weak]),
                        _ => None,
                    },
                };
                // One frame encode: color is coded once and every extra
                // coding is costed against the same DC streams.
                let mut best = writer.clone();
                let (mut payload, _) = encode_frame_core_inner(
                    ctx,
                    scratch,
                    distance,
                    Cow::Borrowed(&opsin),
                    alpha,
                    ExtraCoding {
                        race: Some(&race),
                        ..ExtraCoding::default()
                    },
                    coeff_shifts,
                    frame_kind,
                    #[cfg(feature = "splines")]
                    splines,
                    #[cfg(feature = "splines")]
                    allow_edge_splines,
                    GrainPass::Off,
                    &mut best,
                )?;
                let chosen = race.chosen.lock().ok().and_then(|chosen| *chosen);
                if chosen.is_some_and(|index| alternatives[index].based()) {
                    payload += base_bits;
                }
                drop(race);
                drop(alternatives);
                drop((incumbent_scores, zoned, residual_fit, fits));
                // With alpha patches on, a repeated mask may do better as an
                // atlas: that needs its own frames.
                if repeated && ctx.alpha_patches {
                    let mut masked = writer.clone();
                    let masked_payload = encode_frame_core_masks(
                        ctx,
                        scratch,
                        distance,
                        opsin,
                        alpha,
                        coeff_shifts,
                        frame_kind,
                        #[cfg(feature = "splines")]
                        splines,
                        #[cfg(feature = "splines")]
                        allow_edge_splines,
                        GrainPass::Inherit,
                        &mut masked,
                    )?;
                    if masked.bits_written() < best.bits_written() {
                        best = masked;
                        payload = masked_payload;
                    }
                }
                *writer = best;
                return Ok(payload);
            }
        }
    }
    encode_frame_core_masks(
        ctx,
        scratch,
        distance,
        opsin,
        alpha,
        coeff_shifts,
        frame_kind,
        #[cfg(feature = "splines")]
        splines,
        #[cfg(feature = "splines")]
        allow_edge_splines,
        grain_pass,
        writer,
    )
}

/// The ordinary frame, raced against the exact and atlas codings of a
/// repeated coverage mask.
#[allow(clippy::too_many_arguments)]
fn encode_frame_core_masks(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    opsin: Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    #[cfg(feature = "splines")] splines: Option<&crate::splines::SplineSet>,
    #[cfg(feature = "splines")] allow_edge_splines: bool,
    grain_pass: GrainPass,
    writer: &mut BitWriter,
) -> Result<usize, EncodeError> {
    // These candidates own reference slot 2 until the final frame. Sampled
    // rate probes, layered bases and frames with other image features keep
    // the ordinary path. In particular, probes rescale payload estimates and
    // must never include an unscaled reference atlas.
    let eligible = matches!(frame_kind, VarDctFrameKind::Regular)
        && matches!(grain_pass, GrainPass::Inherit)
        && ctx.frame_role == FrameRole::Last
        && ctx.coding.is_xyb()
        && ctx.speed >= crate::Speed::Slow
        && !ctx.extra_black
        && ctx.extra_step == 1
        && (ctx.alpha_patches || ctx.extra_squeeze_distance > 0.0)
        && ctx
            .extra_plane_tree
            .is_some_and(|budget| !budget.gradient_only)
        && ctx.grain == crate::Grain::Off
        && !ctx.progressive_dc
        && coeff_shifts == [0]
        && opsin.xsize().saturating_mul(opsin.ysize()) <= 16 * 1024 * 1024;
    #[cfg(feature = "splines")]
    let eligible = eligible
        && splines.is_none()
        && !ctx.splines
        && !ctx.edge_splines
        && !ctx.dots
        && ctx.overlay_coverage.is_none();
    let plan = eligible
        .then(|| crate::alpha_patches::find_alpha_patches(alpha?, opsin.xsize(), opsin.ysize()))
        .flatten();
    let Some(plan) = plan else {
        return encode_frame_core_plain(
            ctx,
            scratch,
            distance,
            opsin,
            alpha,
            coeff_shifts,
            frame_kind,
            #[cfg(feature = "splines")]
            splines,
            #[cfg(feature = "splines")]
            allow_edge_splines,
            grain_pass,
            writer,
        );
    };

    // Keep the incumbent complete frame, including its incoming bit offset.
    // The candidates may change the shared DC entropy bundle, so comparing
    // extra-channel estimates alone is insufficient to prevent file growth.
    let mut best = writer.clone();
    let mut best_payload = encode_frame_core_plain(
        ctx,
        scratch,
        distance,
        opsin.clone(),
        alpha,
        coeff_shifts,
        frame_kind,
        #[cfg(feature = "splines")]
        splines,
        #[cfg(feature = "splines")]
        allow_edge_splines,
        grain_pass,
        &mut best,
    )?;
    let mut candidate_scratch = CoderScratch::default();
    if ctx.extra_squeeze_distance > 0.0 {
        // A repeated mask can violate the fixed fourfold bound used to
        // screen its learned exact tree. Compare the actual exact-alpha
        // frame, retaining the incumbent when exact coding costs more.
        let mut exact = writer.clone();
        if let Ok((payload, _)) = encode_frame_core_inner(
            ctx,
            &mut candidate_scratch,
            distance,
            Cow::Owned(opsin.clone()),
            alpha,
            ExtraCoding {
                alpha: None,
                lossless: true,
                ..ExtraCoding::default()
            },
            coeff_shifts,
            frame_kind,
            #[cfg(feature = "splines")]
            splines,
            #[cfg(feature = "splines")]
            false,
            GrainPass::Off,
            &mut exact,
        ) && exact.bits_written() < best.bits_written()
        {
            best = exact;
            best_payload = payload;
        }
    }

    if !ctx.alpha_patches {
        *writer = best;
        return Ok(best_payload);
    }
    let mut patched = writer.clone();
    let start = patched.bits_written();
    let channels = std::array::from_fn(|_| vec![0; plan.atlas_width * plan.atlas_height]);
    if crate::lossless::encode_modular_xyb_atlas_tree(
        &channels,
        plan.atlas_width,
        plan.atlas_height,
        Some(&plan.atlas_alpha),
        ctx.speed,
        &ctx.thread_pool,
        &mut candidate_scratch,
        &mut patched,
    ) {
        let atlas_bits = patched.bits_written() - start;
        // Color visibility always reads the target alpha. The substituted
        // base samples are only the entropy representation of that channel.
        if let Ok((payload, _)) = encode_frame_core_inner(
            ctx,
            &mut candidate_scratch,
            distance,
            Cow::Owned(opsin),
            alpha,
            ExtraCoding {
                alpha: Some(&plan.base_alpha),
                lossless: true,
                ..ExtraCoding::default()
            },
            coeff_shifts,
            VarDctFrameKind::Patched(&plan.references, None, None),
            #[cfg(feature = "splines")]
            None,
            #[cfg(feature = "splines")]
            false,
            GrainPass::Off,
            &mut patched,
        ) && patched.bits_written() < best.bits_written()
        {
            best = patched;
            best_payload = payload + atlas_bits;
        }
    }
    *writer = best;
    Ok(best_payload)
}

/// Grain handling shared by the ordinary frame and the incumbent of the
/// repeated-alpha representation race.
#[allow(clippy::too_many_arguments)]
fn encode_frame_core_plain(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    opsin: Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    #[cfg(feature = "splines")] splines: Option<&crate::splines::SplineSet>,
    #[cfg(feature = "splines")] allow_edge_splines: bool,
    grain_pass: GrainPass,
    writer: &mut BitWriter,
) -> Result<usize, EncodeError> {
    let displayed = matches!(
        frame_kind,
        VarDctFrameKind::Regular | VarDctFrameKind::Patched(..)
    );
    if matches!(grain_pass, GrainPass::Inherit)
        && ctx.grain == crate::Grain::Auto
        && displayed
        && let Some((measured, stats)) = crate::noise::auto_candidate(&opsin, ctx)
    {
        let original = opsin.clone();
        let mut probe = BitWriter::new();
        let (bits, survival) = encode_frame_core_inner(
            ctx,
            scratch,
            distance,
            Cow::Owned(opsin),
            alpha,
            ExtraCoding::default(),
            coeff_shifts,
            frame_kind,
            #[cfg(feature = "splines")]
            splines,
            #[cfg(feature = "splines")]
            allow_edge_splines,
            GrainPass::Probe(stats.clone()),
            &mut probe,
        )?;
        let survival = survival.unwrap_or(1.0);
        let weight = crate::noise::grain_weight(survival);
        if weight <= 0.0 {
            writer.append(&probe);
            return Ok(bits);
        }
        drop(probe);
        return encode_frame_core_inner(
            ctx,
            scratch,
            distance,
            Cow::Owned(original),
            alpha,
            ExtraCoding::default(),
            coeff_shifts,
            frame_kind,
            #[cfg(feature = "splines")]
            splines,
            #[cfg(feature = "splines")]
            allow_edge_splines,
            GrainPass::Apply(measured, stats, weight),
            writer,
        )
        .map(|(bits, _)| bits);
    }
    encode_frame_core_inner(
        ctx,
        scratch,
        distance,
        Cow::Owned(opsin),
        alpha,
        ExtraCoding::default(),
        coeff_shifts,
        frame_kind,
        #[cfg(feature = "splines")]
        splines,
        #[cfg(feature = "splines")]
        allow_edge_splines,
        grain_pass,
        writer,
    )
    .map(|(bits, _)| bits)
}

/// Largest plane whose extra-channel alternatives are costed side by side.
const PARALLEL_RACE_PIXELS: usize = 2 * 1024 * 1024;

/// Another coding of the frame's extra channel, costed beside the ordinary
/// one inside a single frame encode.
#[derive(Clone, Copy)]
enum ExtraAlternative<'a> {
    /// `plane` through the ordinary extra-channel coder; `based` planes are
    /// residuals the decoder adds to the function frame in slot 0.
    Plane {
        plane: &'a AlphaPlane,
        lossless: bool,
        scale: f32,
        based: bool,
    },
    /// A whole-plane function in this frame.
    Nested(&'a crate::function_fields::FittedProfile),
}

impl ExtraAlternative<'_> {
    fn based(&self) -> bool {
        matches!(self, Self::Plane { based: true, .. })
    }

    /// Only costed when no first-round alternative passes the judge.
    fn fallback(&self) -> bool {
        matches!(self, Self::Plane { scale, lossless: false, .. } if *scale != 1.0)
    }
}

struct ExtraRace<'a> {
    alternatives: &'a [ExtraAlternative<'a>],
    /// Size of the base frame every `based` alternative also pays.
    base_bits: usize,
    /// `(alternative, incumbent samples, alternative's coded samples)`:
    /// `None` when the alternative may not replace the ordinary coding,
    /// otherwise its distance from the source (lower is better).
    judge: &'a (dyn Fn(usize, &AlphaPlane, &AlphaPlane) -> Option<f64> + Sync),
    /// The incoming bits followed by the base frame; the displayed frame is
    /// written after it when a `based` alternative wins.
    based: std::sync::Mutex<Option<BitWriter>>,
    chosen: std::sync::Mutex<Option<usize>>,
    /// `[strong dead zone, none, weak dead zone]` among the alternatives: the
    /// weak one codes like the other two wherever those two code alike.
    zones: Option<[usize; 3]>,
}

#[derive(Clone, Copy, Default)]
struct ExtraCoding<'a> {
    alpha: Option<&'a AlphaPlane>,
    lossless: bool,
    profile: Option<&'a crate::function_fields::FittedProfile>,
    /// Frame header role when it differs from the context's.
    role: Option<FrameRole>,
    race: Option<&'a ExtraRace<'a>>,
}

fn color_is_unpatched(frame_kind: VarDctFrameKind<'_>) -> bool {
    match frame_kind {
        VarDctFrameKind::Regular => true,
        VarDctFrameKind::Patched(references, ..) => references.iter().all(|r| r.alpha_only),
        VarDctFrameKind::ReferenceOnly { .. } => false,
    }
}

pub(super) fn epf_fit_enabled(ctx: &EncodingContext, distance: f32) -> bool {
    // A saved base is fitted only when it knows what is laid over it.
    let role = ctx.frame_role == FrameRole::Last
        || (ctx.frame_role == FrameRole::SavedFirst && ctx.patch_epf && ctx.epf_overlay.is_some());
    ctx.speed.effort().epf_fit && ctx.coding.is_xyb() && role && (0.5..=24.0).contains(&distance)
}

fn encode_frame_core_inner(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    opsin: Cow<'_, Image3F>,
    alpha: Option<&AlphaPlane>,
    extra_coding: ExtraCoding<'_>,
    coeff_shifts: &[u32],
    frame_kind: VarDctFrameKind<'_>,
    #[cfg(feature = "splines")] splines: Option<&crate::splines::SplineSet>,
    #[cfg(feature = "splines")] allow_edge_splines: bool,
    grain_pass: GrainPass,
    writer: &mut BitWriter,
) -> Result<(usize, Option<f32>), EncodeError> {
    #[cfg(feature = "extraslow")]
    let _b_qm_scale = RestoreBQmScale(ctx, ctx.b_qm_scale());
    let num_threads = ctx.thread_pool.num_threads();
    let dim = ImageDim::new(opsin.xsize(), opsin.ysize());
    // Grain is measured on the source XYB before any frame processing; only
    // displayed frames carry it (an atlas frame's header has no flags).
    let mut opsin = opsin;
    let displayed = matches!(
        frame_kind,
        VarDctFrameKind::Regular | VarDctFrameKind::Patched(..)
    );
    let probe_stats = match &grain_pass {
        GrainPass::Probe(stats) | GrainPass::Apply(_, stats, _) => Some(stats),
        _ => None,
    };
    let mut noise = match &grain_pass {
        GrainPass::Off | GrainPass::Probe(_) => None,
        GrainPass::Apply(measured, _, weight) if displayed => {
            crate::noise::apply_measured(ctx, scratch, distance, opsin.to_mut(), *measured, *weight)
        }
        GrainPass::Apply(..) => None,
        GrainPass::Inherit if displayed => {
            crate::noise::NoiseParams::resolve(ctx.grain, ctx, scratch, distance, opsin.to_mut())
        }
        GrainPass::Inherit => None,
    };
    let mut distp = compute_distance_params(distance);
    if ctx.coding.is_xyb() {
        if ctx.x_heavy() && distp.x_qm_scale == 2 {
            distp.x_qm_scale = 3;
        }
        distp.x_qm_scale = distp.x_qm_scale.max(ctx.x_qm_scale_floor());
    } else {
        // Not serialized: the decoder's X multiplier is 1.
        distp.x_qm_scale = 2;
        distp.dc_step = dc_steps_in_range(ctx.coding.dc_step_base(), distp.scale_dc, &opsin);
    }

    // Progressive lossy splits each quantized AC coeff across `num_passes`
    // passes by a decreasing per-pass shift (last = 0). The decoder reconstructs
    // C = sum_p (sent_p << shift_p) (jxl-vardct hf_coeff.rs:185,191).
    let num_passes = coeff_shifts.len();
    // Progressive DC: the DC image goes out as its own squeezed DC frame,
    // which this frame references instead of coding a DC stream.
    let use_dc_frame = ctx.progressive_dc
        && matches!(
            frame_kind,
            VarDctFrameKind::Regular | VarDctFrameKind::Patched(..)
        )
        && alpha.is_none()
        && ctx.coding.is_xyb()
        && crate::lossy_modular::dc_frame_supported(dim.xsize_blocks, dim.ysize_blocks);

    let num_sections = 2 + dim.num_dc_groups + num_passes * dim.num_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // Set up every DC group, then encode every AC group. Keeping the
    // two apart lets AC groups from all DC groups share one steal_map instead of
    // one serialized burst per DC group. Merges back in (dc, gix) order, so the
    // output stays bit-identical to single-threaded.
    let group_coords: Vec<(usize, usize)> = (0..dim.ysize_dc_groups)
        .flat_map(|gy| (0..dim.xsize_dc_groups).map(move |gx| (gx, gy)))
        .collect();
    let mut opsin_image = Some(opsin);

    // Edge proposals are judged on native transform footprints with the real
    // AQ field, quantizers and CfL map, on a DCT8 layout: accepted splines
    // change the source, so the one full layout search runs after them.
    // Sample estimates and atlas/progressive/noisy frames never run this search.
    // Patched frames, the saved base of a layered stack and frames with alpha
    // take part: the view weights each pixel by what the decoder shows of it.
    #[cfg(feature = "splines")]
    let edge_set = if allow_edge_splines
        && ctx.edge_splines
        && ctx.coding.is_xyb()
        && matches!(
            frame_kind,
            VarDctFrameKind::Regular | VarDctFrameKind::Patched(..)
        )
        && noise.is_none()
        && !use_dc_frame
        && !distp.gab_enabled
        && num_passes == 1
        && distance > 1.0
        && distance <= 16.0
        && dim.xsize >= 64
        && dim.ysize >= 64
    {
        let source = opsin_image.as_ref().unwrap();
        let provisional: Vec<Result<DcGroupData, EncodeError>> =
            ctx.thread_pool
                .steal_map(scratch, group_coords.len(), |i, scratch| {
                    let (gx, gy) = group_coords[i];
                    provisional_dc_group(ctx, scratch, source, &dim, &distp, gx, gy)
                });
        let provisional: Vec<DcGroupData> = provisional.into_iter().collect::<Result<_, _>>()?;
        let view = edge_view(ctx, frame_kind, alpha, dim.xsize, dim.ysize);
        crate::splines::select_edges(
            ctx,
            scratch,
            opsin_image.as_mut().unwrap().to_mut(),
            &provisional,
            crate::splines::edge_model::Quantization {
                distance,
                scale: distp.scale,
                scale_dc: distp.scale_dc,
                x_qm_scale: distp.x_qm_scale,
                dc_step: distp.dc_step,
            },
            splines,
            &view,
        )
    } else {
        None
    };
    #[cfg(feature = "splines")]
    let splines = edge_set.as_ref().or(splines);
    let opsin = opsin_image.as_ref().unwrap();

    // Split the thread budget across nesting levels: `outer` lanes steal DC
    // groups, each parallelizing its AC-strategy bands with the remainder, so
    // the setup phase saturates all cores even with few (large) DC groups.
    let outer = group_coords.len().min(num_threads.max(1));
    let setup_budget = num_threads.max(1).div_ceil(outer);
    let setups: Vec<Result<(DcGroupData, usize, usize), EncodeError>> =
        ctx.thread_pool
            .steal_map(scratch, group_coords.len(), |i, scratch| {
                let (dc_gx, dc_gy) = group_coords[i];
                setup_dc_group(
                    ctx,
                    scratch,
                    opsin,
                    &dim,
                    &distp,
                    dc_gx,
                    dc_gy,
                    setup_budget,
                )
            });
    let setups: Vec<(DcGroupData, usize, usize)> = setups.into_iter().collect::<Result<_, _>>()?;

    let mut dc_datas: Vec<DcGroupData> = Vec::with_capacity(setups.len());
    let mut ac_tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (dc_idx, (dc_data, gxs, gys)) in setups.into_iter().enumerate() {
        ac_tasks.extend((0..gxs * gys).map(|g| (dc_idx, g % gxs, g / gxs)));
        dc_datas.push(dc_data);
    }

    crate::ac_strategy::account_matrix_headers(
        ctx,
        scratch,
        opsin,
        distance,
        distp.scale,
        distp.x_qm_scale,
        &group_coords,
        &mut dc_datas,
    );
    for dc in dc_datas.iter_mut() {
        dc.block_quant_field = None;
    }

    // The CfL slopes were estimated on fixed 8x8 DCTs so the transform search
    // could price candidates under them; the final layout and quant field
    // are known now, so re-estimate them on the coefficients that get coded.
    if crate::color_correlation::cfl_slope_search(ctx) {
        let dc_ref = &dc_datas;
        let maps = ctx
            .thread_pool
            .steal_map(scratch, group_coords.len(), |i, scratch| {
                let (dc_gx, dc_gy) = group_coords[i];
                crate::color_correlation::refit_cmap_on_layout(
                    ctx,
                    scratch,
                    opsin,
                    &dc_ref[i],
                    dc_gx * K_DC_GROUP_DIM / K_BLOCK_DIM,
                    dc_gy * K_DC_GROUP_DIM / K_BLOCK_DIM,
                    distp.scale,
                    distp.distance,
                    distp.x_qm_scale,
                )
            });
        for (dc, (ytox, ytob)) in dc_datas.iter_mut().zip(maps) {
            dc.ytox_map = ytox;
            dc.ytob_map = ytob;
        }
    }

    let patch_luma = match frame_kind {
        VarDctFrameKind::Patched(references, Some(atlases), _) => Some((references, atlases)),
        _ => None,
    };
    // Layer residuals and progressive/DC frames retain the EPF schedule.
    let fit_epf = epf_fit_enabled(ctx, distance)
        && (patch_luma.is_some() || color_is_unpatched(frame_kind))
        && !use_dc_frame
        && num_passes == 1
        && dim.xsize >= 16
        && dim.ysize >= 16;
    // The grain probe reads its survival off the same reconstruction.
    let want_recon = fit_epf || (probe_stats.is_some() && !use_dc_frame && num_passes == 1);

    // Per-image quant-field threshold for the fine AC block-context layout.
    // The median splits the field into halves with genuinely different
    // coefficient statistics; whether any split is *kept* is decided later
    // from real token stats, so a useless threshold costs nothing.
    let qf_threshold = ac_qf_threshold(&dc_datas);

    let dc_ref = &dc_datas;
    let natural_orders = crate::coeff_order::CoeffOrders::natural();
    // Tally coefficient positions only when the refine pass will run to consume
    // the derived orders: pass one alone can never adopt them (its tokens are
    // already written on the natural scan), so the work would be wasted and its
    // mere presence perturbs codegen enough to shift a few borderline
    // quantizer roundings on paths that should be untouched.
    let want_order_stats = ctx.speed.effort().coeff_orders;
    // Passes before the refined one only feed statistics and token prices,
    // which describe whole coefficients: they tokenize as one pass, and only
    // the refined pass splits the coefficients across the frame's passes.
    let (analysis_passes, analysis_shifts): (usize, &[u32]) = if want_order_stats {
        (1, &[0])
    } else {
        (num_passes, coeff_shifts)
    };
    // Only the final quantization produces pixels. Each worker pastes its
    // group on completion so the frame and all group captures never coexist.
    let mut epf_recon =
        (want_recon && !want_order_stats).then(|| Image3F::new(dim.xsize, dim.ysize));
    let results = {
        let target = epf_recon.as_mut().map(Mutex::new);
        ctx.thread_pool
            .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                let (dc_idx, gx, gy) = ac_tasks[t];
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                let (p, local, source_b, stats, capture) = process_ac_group(
                    ctx,
                    scratch,
                    opsin,
                    &dim,
                    &distp,
                    &dc_ref[dc_idx],
                    analysis_passes,
                    analysis_shifts,
                    dc_gx,
                    dc_gy,
                    gx,
                    gy,
                    0,
                    None,
                    &natural_orders,
                    &[],
                    want_order_stats || use_dc_frame || want_recon,
                    qf_threshold,
                    // Without custom orders this pass is the final quantization.
                    if want_recon && !want_order_stats {
                        AcGroupCapture::Pixels
                    } else {
                        AcGroupCapture::None
                    },
                    None,
                );
                if let (Some(target), Some((image, _))) = (&target, capture) {
                    paste_group(&mut target.lock().unwrap(), &dim, p.group_idx, &image);
                }
                (dc_idx, gx, gy, p, local, source_b, stats)
            })
    };

    let mut all_pending: Vec<PendingAcGroup> = Vec::with_capacity(results.len());
    // Adopt the first group's buffers instead of allocating an empty tally.
    // Fast and progressive frames never allocate aggregate order statistics.
    let mut order_stats: Option<crate::coeff_order::OrderStats> = None;
    for (dc_idx, gx, gy, p, local, source, stats) in results {
        merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
        if let Some(source) = source {
            let dc = &mut dc_datas[dc_idx];
            let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
            let ox = gx * K_GROUP_DIM_IN_BLOCKS;
            let oy = gy * K_GROUP_DIM_IN_BLOCKS;
            for (target, source) in [
                (&mut dc.source_dc_y, &source.y),
                (&mut dc.source_dc_b, &source.b),
                (&mut dc.source_dc_x, &source.x),
            ] {
                let target = target.get_or_insert_with(|| crate::image::Plane::new(w, h));
                for y in 0..source.ysize() {
                    target.row_mut(oy + y)[ox..ox + source.xsize()].copy_from_slice(source.row(y));
                }
            }
        }
        all_pending.push(p);
        if let Some(s) = stats {
            if let Some(total) = &mut order_stats {
                total.merge(&s);
            } else {
                order_stats = Some(s);
            }
        }
    }

    // The EPF fit adds the decoded-minus-source DC to its reconstruction.
    // Other DC tools key on the source planes, so only a copy outlives the
    // first pass when nothing else asked for them.
    let mut epf_source_dc = None;
    if want_recon && !want_order_stats && !use_dc_frame {
        epf_source_dc = Some(assemble_source_dc(&dc_datas, &group_coords, &dim));
        for dc in &mut dc_datas {
            dc.source_dc_b = None;
            dc.source_dc_y = None;
            dc.source_dc_x = None;
        }
    }

    // DC-level chroma-from-luma. The slope has to be folded into the DC
    // quantizer's rounding rather than subtracted from the stored integers
    // afterwards: `round(v) - round(s*y)` carries up to a full step of B DC
    // error against `round(v - s*y)`'s half step, and that extra chroma noise
    // costs far more quality than the slope saves in rate. So the search rides
    // on the rerank pass, which re-quantizes every DC group regardless -- and
    // is skipped entirely when that pass does not run.
    // Custom coefficient orders, derived from the first pass's nonzero tallies.
    let mut coeff_orders = natural_orders;
    let mut order_alternatives = Vec::new();

    let mut ytob_dc = 0i32;
    if want_order_stats {
        let scan_moved = order_stats.as_ref().map_or(0.0, |stats| {
            crate::coeff_order::derive_orders(stats, &mut coeff_orders)
        });
        // Keep the incumbent RDOQ scan: the final race changes coding only,
        // so all arms transmit exactly the same quantized coefficients.
        if ctx.speed.effort().coeff_order_entropy_race() && num_passes == 1 {
            if let Some(stats) = &order_stats {
                let mut proposed = crate::coeff_order::CoeffOrders::natural();
                crate::coeff_order::propose_orders(stats, &mut proposed);
                if proposed.used_mask != 0 && proposed.orders != coeff_orders.orders {
                    order_alternatives.push(proposed);
                }
            }
            if coeff_orders.used_mask != 0 {
                order_alternatives.push(crate::coeff_order::CoeffOrders::natural());
            }
        }
        drop(order_stats);
        ytob_dc = if use_dc_frame {
            0
        } else if ctx.speed.effort().dc_cfl_weighted() {
            crate::color_correlation::choose_ytob_dc_weighted(
                &dc_datas,
                ctx,
                distp.scale_dc,
                distp.dc_step,
                &mut scratch.dc_cfl_cur,
                &mut scratch.dc_cfl_prev,
            )
        } else {
            let candidate = choose_ytob_dc(
                &dc_datas,
                ctx.fill_ytob_row,
                ctx.accumulate_ytob_weights,
                ctx.fill_ytob_residuals,
                &mut scratch.dc_cfl_cur,
                &mut scratch.dc_cfl_prev,
                distp.dc_step,
                ctx.cfl_frame().color_factor,
                &ctx.thread_pool,
            );
            crate::color_correlation::validate_ytob_dc(
                &dc_datas,
                candidate,
                distp.scale_dc,
                distp.dc_step,
                ctx.cfl_frame(),
                ctx.quantize_dc_cfl,
            )
        };
        if !use_dc_frame {
            distp.dc_step = crate::color_correlation::choose_dc_steps(
                &dc_datas,
                ctx,
                distp.scale_dc,
                ytob_dc,
                distp.dc_step,
                ctx.cfl_frame(),
            );
        }
        if !ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
            if want_recon && epf_source_dc.is_none() {
                epf_source_dc = Some(assemble_source_dc(&dc_datas, &group_coords, &dim));
            }
            for dc in &mut dc_datas {
                dc.source_dc_b = None;
                dc.source_dc_y = None;
                dc.source_dc_x = None;
            }
        }
        // The first pass's tokens follow the natural scans. Once the custom
        // orders move most of the walk, they no longer describe what the
        // contexts of the refine pass will see, and the quantizer would price
        // its choices on them: tokenize again on the orders it codes with.
        if ctx.speed.effort().rdoq_fresh_prices() || scan_moved >= RDOQ_FRESH_PRICES_MIN_SCAN_MOVE {
            // These streams will be replaced; release their token buffers
            // before allocating the reordered streams for the whole frame.
            all_pending.clear();
            let dc_ref = &dc_datas;
            let reordered = ctx
                .thread_pool
                .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    let (dc_gx, dc_gy) = group_coords[dc_idx];
                    process_ac_group(
                        ctx,
                        scratch,
                        opsin,
                        &dim,
                        &distp,
                        &dc_ref[dc_idx],
                        analysis_passes,
                        analysis_shifts,
                        dc_gx,
                        dc_gy,
                        gx,
                        gy,
                        ytob_dc,
                        None,
                        &coeff_orders,
                        &[],
                        false,
                        qf_threshold,
                        AcGroupCapture::None,
                        None,
                    )
                    .0
                });
            all_pending.extend(reordered);
        }
        let prices = {
            let provisional_code = crate::entropy::optimize_entropy_code_ac_streams(
                all_pending.iter().map(|pg| pg.tokens[0].as_slice()),
                crate::ac_context::K_NUM_FINE_AC_CONTEXTS,
                &mut scratch.huffman_pool,
                // No config selection here: these prices feed RDOQ, and even a
                // one-token nudge can push the clustering off a knife-edge merge.
                // Selection applies to the final codes only.
                false,
                ctx.speed,
                Some(&ctx.thread_pool),
            );
            crate::entropy::FrozenTokenPrices::new(&provisional_code)
        };
        // The closed loop's reference and verification encodes are pass 2
        // itself (same RDOQ, prices and orders), so when it runs its final
        // groups replace the refined pass.
        all_pending.clear();
        #[cfg(feature = "extraslow")]
        let loop_groups = if ctx.speed.effort().aq_closed_loop()
            && ctx.coding.is_xyb()
            && num_passes == 1
        {
            // The loop's selection-scale race re-runs the DC-group setup
            // (quant field, layout, CfL) and its frame-level follow-ups
            // under selection-only scaled tables; coefficient orders,
            // prices and DC choices stay the first pass's.
            let mut reselect =
                |scratch: &mut CoderScratch| -> Result<Vec<DcGroupData>, EncodeError> {
                    let setups =
                        ctx.thread_pool
                            .steal_map(scratch, group_coords.len(), |i, scratch| {
                                let (dc_gx, dc_gy) = group_coords[i];
                                setup_dc_group(
                                    ctx,
                                    scratch,
                                    opsin,
                                    &dim,
                                    &distp,
                                    dc_gx,
                                    dc_gy,
                                    setup_budget,
                                )
                            });
                    let mut datas: Vec<DcGroupData> = setups
                        .into_iter()
                        .map(|r| r.map(|(data, _, _)| data))
                        .collect::<Result<_, _>>()?;
                    crate::ac_strategy::account_matrix_headers(
                        ctx,
                        scratch,
                        opsin,
                        distance,
                        distp.scale,
                        distp.x_qm_scale,
                        &group_coords,
                        &mut datas,
                    );
                    for dc in datas.iter_mut() {
                        dc.block_quant_field = None;
                    }
                    if crate::color_correlation::cfl_slope_search(ctx) {
                        let dc_ref = &datas;
                        let maps =
                            ctx.thread_pool
                                .steal_map(scratch, group_coords.len(), |i, scratch| {
                                    let (dc_gx, dc_gy) = group_coords[i];
                                    crate::color_correlation::refit_cmap_on_layout(
                                        ctx,
                                        scratch,
                                        opsin,
                                        &dc_ref[i],
                                        dc_gx * K_DC_GROUP_DIM / K_BLOCK_DIM,
                                        dc_gy * K_DC_GROUP_DIM / K_BLOCK_DIM,
                                        distp.scale,
                                        distp.distance,
                                        distp.x_qm_scale,
                                    )
                                });
                        for (dc, (ytox, ytob)) in datas.iter_mut().zip(maps) {
                            dc.ytox_map = ytox;
                            dc.ytob_map = ytob;
                        }
                    }
                    Ok(datas)
                };
            run_quant_field_loop(
                ctx,
                scratch,
                opsin,
                &dim,
                &distp,
                &mut dc_datas,
                &ac_tasks,
                &group_coords,
                num_passes,
                coeff_shifts,
                &coeff_orders,
                &order_alternatives,
                ytob_dc,
                qf_threshold,
                &prices,
                want_recon,
                // Patch atlases and reference frames are judged by the
                // frames that use them, not by their own reconstruction.
                if color_is_unpatched(frame_kind)
                    && !crate::ac_strategy::content_selection_enabled(ctx, distance)
                {
                    Some(&mut reselect)
                } else {
                    None
                },
            )
        } else {
            None
        };
        #[cfg(not(feature = "extraslow"))]
        let loop_groups: Option<(Vec<RefinedAcGroup>, DistanceParams, Option<Image3F>)> = None;
        let loop_groups = loop_groups.map(|(groups, chosen, recon)| {
            distp = chosen;
            (groups, recon)
        });
        let refined = if let Some((groups, recon)) = loop_groups {
            // The loop's final reconstruction replaces the pass-2 captures.
            epf_recon = recon;
            groups
        } else {
            let dc_ref = &dc_datas;
            epf_recon = want_recon.then(|| Image3F::new(dim.xsize, dim.ysize));
            let target = epf_recon.as_mut().map(Mutex::new);
            ctx.thread_pool
                .steal_map(scratch, ac_tasks.len(), |t, scratch| {
                    let (dc_idx, gx, gy) = ac_tasks[t];
                    let (dc_gx, dc_gy) = group_coords[dc_idx];
                    let (p, local, _, _, capture) = process_ac_group(
                        ctx,
                        scratch,
                        opsin,
                        &dim,
                        &distp,
                        &dc_ref[dc_idx],
                        num_passes,
                        coeff_shifts,
                        dc_gx,
                        dc_gy,
                        gx,
                        gy,
                        ytob_dc,
                        Some(&prices),
                        &coeff_orders,
                        &order_alternatives,
                        false,
                        qf_threshold,
                        if want_recon {
                            AcGroupCapture::Pixels
                        } else {
                            AcGroupCapture::None
                        },
                        None,
                    );
                    if let (Some(target), Some((image, _))) = (&target, capture) {
                        paste_group(&mut target.lock().unwrap(), &dim, p.group_idx, &image);
                    }
                    (dc_idx, gx, gy, p, local)
                })
        };
        for (dc_idx, gx, gy, p, local) in refined {
            merge_quant_dc(&mut dc_datas[dc_idx], gx, gy, &local);
            all_pending.push(p);
        }
    }

    // The retained source DC is invariant under AC reranking. Search against
    // the final Y levels and steps, then quantize X once from that source.
    let ytox_dc = if ctx.coding.is_xyb() && ctx.speed.effort().dc_x_cfl && !use_dc_frame {
        crate::color_correlation::choose_ytox_dc(
            &mut dc_datas,
            distp.scale_dc,
            distp.dc_step,
            ctx.cfl_frame(),
            ytob_dc != 0,
            ctx.quantize_dc_cfl,
            &ctx.thread_pool,
            scratch,
        )
    } else {
        0
    };
    if ctx.coding.is_xyb() && ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
        let cfl = crate::color_correlation::CflFrame {
            base_x: crate::color_correlation::y_to_x_ratio(ctx.cfl_frame(), ytox_dc as i8),
            ..ctx.cfl_frame()
        };
        distp.dc_step = dc_rdo::choose_x_dc_step(
            &mut dc_datas,
            &distp,
            cfl,
            ctx.quantize_dc_cfl,
            &ctx.thread_pool,
            scratch,
        );
    }
    // Only DC reconstruction uses the extra X slope. The AC map and the
    // header's base correlations retain the frame's original parameters.
    let dc_cfl = crate::color_correlation::CflFrame {
        base_x: crate::color_correlation::y_to_x_ratio(ctx.cfl_frame(), ytox_dc as i8),
        ..ctx.cfl_frame()
    };
    // Couple B rounding to the final luma error after AC decisions are fixed.
    // Flattening and Haar selection must use that same target as well.
    let compensate_b = ctx.coding.is_xyb()
        && ctx.speed.effort().chroma_dc_squeeze
        && !use_dc_frame
        && compensate_b_dc(
            &mut dc_datas,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.compensate_b,
        );
    if ctx.coding.is_xyb() && !use_dc_frame {
        flatten_chroma_dc(
            opsin,
            &dim,
            &mut dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
        )?;
    }
    if ctx.coding.is_xyb() && ctx.speed.effort().chroma_dc_squeeze && !use_dc_frame {
        choose_chroma_dc_coding(
            &dim,
            &mut dc_datas,
            &mut distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
            ctx.dc_rows.adjust_b,
            &ctx.thread_pool,
        )?;
    }
    if ctx.coding.is_xyb() && ctx.speed.effort().dc_rdoq && !use_dc_frame {
        dc_rdo::refine_dc_rounding(
            &mut dc_datas,
            &distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
            &mut scratch.dc_predictor,
            &ctx.thread_pool,
        )?;
    }
    let skip_dc_smoothing = !use_dc_frame
        && skip_dc_smoothing(
            &ctx.thread_pool,
            scratch,
            opsin,
            &dim,
            &dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.reconstruct,
        )?;
    let mut epf_sharpness = epf_sharpness_id(distance, ctx.screen_content());
    let mut survival = None;
    if let Some(mut recon) = epf_recon {
        let decoded_dc = dc_smoothing::decoded_dc(
            &dim,
            &dc_datas,
            &group_coords,
            &distp,
            ytob_dc,
            dc_cfl,
            ctx.dc_rows.reconstruct,
            skip_dc_smoothing,
        )?;
        let source_dc =
            epf_source_dc.unwrap_or_else(|| assemble_source_dc(&dc_datas, &group_coords, &dim));
        crate::epf_filter::add_dc_error(
            ctx,
            scratch,
            &mut recon,
            &decoded_dc,
            &source_dc,
            &dc_datas,
            &group_coords,
        );
        if distp.gab_enabled {
            crate::epf_filter::gaborish_in_place(&mut recon);
        }
        let blocks = crate::epf_filter::FrameBlocks::new(
            &dc_datas,
            &group_coords,
            dim.xsize_blocks,
            dim.ysize_blocks,
            distp.global_scale,
        );
        if fit_epf {
            let source = distp
                .gab_enabled
                .then(|| crate::epf_filter::gaborish(opsin));
            // Source pixel decisions are complete. The EPF proxy needs only Y;
            // release X/B before allocating its pyramid and candidate output.
            // A Gaborish source also replaces the original source altogether.
            let source_y = match (source, opsin_image.take().unwrap()) {
                (Some(source), _) | (None, Cow::Owned(source)) => {
                    let [_, y, _] = source.into_planes();
                    Cow::Owned(y)
                }
                (None, Cow::Borrowed(source)) => Cow::Borrowed(source.plane(1)),
            };
            #[cfg(feature = "splines")]
            // Splines are drawn after the filter: the fit sees them as a layer.
            let spline_layer =
                splines.map(|set| crate::splines::additive_y(ctx, set, dim.xsize, dim.ysize));
            let overlay = ctx
                .epf_overlay
                .as_ref()
                .filter(|o| o.len() == source_y.as_slice().len());
            let chosen = crate::epf_filter::fit(
                ctx,
                scratch,
                source_y,
                crate::epf_filter::LumaLayers {
                    #[cfg(feature = "splines")]
                    additive: spline_layer.as_deref(),
                    #[cfg(not(feature = "splines"))]
                    additive: None,
                    patches: patch_luma,
                    overlay,
                },
                &recon,
                &blocks,
                distance,
                crate::epf_filter::Choice {
                    iters: distp.epf_iters,
                    quant_mul: distp.epf_quant_mul,
                    pass0: distp.epf_pass0_scale,
                    sharpness: epf_sharpness,
                },
            );
            distp.epf_iters = chosen.iters;
            distp.epf_pass0_scale = chosen.pass0;
            distp.epf_quant_mul = chosen.quant_mul;
            epf_sharpness = chosen.sharpness;
        }
        if let Some(stats) = probe_stats {
            // Survival as the decoder shows it: after this frame's EPF.
            let lf = crate::epf_filter::LoopFilter::new(
                distp.epf_iters,
                distp.epf_pass0_scale,
                distp.epf_quant_mul,
                ctx.epf_channel_scale,
            );
            let mut epf = crate::epf_filter::Epf::new(&ctx.thread_pool, &recon, &blocks);
            let filtered = epf.apply(&ctx.thread_pool, scratch, &lf, epf_sharpness);
            let s = stats.survival(filtered, ctx.noise_kernels);
            survival = Some(s.overall);
            if let (GrainPass::Apply(_, _, weight), Some(params)) = (&grain_pass, noise.as_mut()) {
                params.fill_gap(&s, *weight);
                if !params.has_any() {
                    noise = None;
                }
            }
        }
    }
    // Pixel-domain decisions are finished; entropy coding only reads DC data
    // and tokens. Release the full-resolution planes before its working sets.
    drop(opsin_image);

    /// A weighted-predictor candidate whose estimate under the incumbent's
    /// code is this much worse (per 10 000) is not coded: on 30 encodes the
    /// closest candidate that then won sat at +0.8 ‰, the worst miss of the
    /// estimate against the coded candidate at 4 ‰ on losers only.
    const DC_WP_SKIP_MARGIN_PER_10K: u64 = 15;
    // The DC streams' weighted-predictor parameters: libjxl presets raced on
    // the complete DC coding (the MA-learned tree is learned once and
    // re-tokenized under each), then the parameters fitted from the winning
    // preset on a proxy and priced exactly once. Fast and Medium race the
    // second preset only, MediumSlow and Slow also fit when it wins (a fit
    // from the default never paid), ExtraSlow races all four and always fits.
    let mut dc_entropy = super::dc_entropy::prepare_dc_entropy(
        ctx,
        scratch,
        &dc_datas,
        dim.num_dc_groups,
        epf_sharpness,
        use_dc_frame,
        distance,
        crate::weighted_predictor::WpParams::DEFAULT,
        None,
    );
    let wp_effort = ctx.speed.effort();
    let wp_full = wp_effort.dc_wp_fit();
    if (wp_full || wp_effort.dc_wp_race) && !use_dc_frame {
        let mut best_bits = dc_entropy.bits(dim.num_dc_groups, scratch);
        let mut offer = |params: crate::weighted_predictor::WpParams,
                         dc_entropy: &mut super::dc_entropy::DcEntropy,
                         scratch: &mut CoderScratch| {
            // Under the MA arm the candidate parameters are priced first
            // under the incumbent's tree and code.
            let candidate = match dc_entropy.dc_tokens_under(ctx, scratch, &dc_datas, params) {
                Some(dc) => {
                    let estimate = dc_entropy.bits_with(dim.num_dc_groups, scratch, &dc, params);
                    if estimate * 10_000 >= best_bits * (10_000 + DC_WP_SKIP_MARGIN_PER_10K) {
                        return;
                    }
                    dc_entropy.warmed(ctx, scratch, dc, params)
                }
                None => None,
            }
            .unwrap_or_else(|| {
                super::dc_entropy::prepare_dc_entropy(
                    ctx,
                    scratch,
                    &dc_datas,
                    dim.num_dc_groups,
                    epf_sharpness,
                    use_dc_frame,
                    distance,
                    params,
                    dc_entropy.ma_tree.as_ref(),
                )
            });
            let bits = candidate.bits(dim.num_dc_groups, scratch);
            if bits < best_bits {
                best_bits = bits;
                *dc_entropy = candidate;
            }
        };
        let presets = &crate::weighted_predictor::WpParams::PRESETS;
        for &params in if wp_full {
            &presets[1..]
        } else {
            &presets[1..2]
        } {
            offer(params, &mut dc_entropy, scratch);
        }
        if wp_full
            || (wp_effort.dc_wp_fit_on_win
                && dc_entropy.wp_params != crate::weighted_predictor::WpParams::DEFAULT)
        {
            let fitted =
                super::dc_entropy::fit_dc_wp_params(ctx, scratch, &dc_datas, dc_entropy.wp_params);
            if fitted != dc_entropy.wp_params {
                offer(fitted, &mut dc_entropy, scratch);
            }
        }
    }
    // Refit: the chroma DC rounding priced by the written tree instead of
    // the frozen static contexts, then re-tokenized under the same tree.
    if ctx.coding.is_xyb()
        && ctx.speed.effort().dc_rdoq
        && ctx.speed.effort().dc_rdoq_refit
        && !use_dc_frame
        && !dc_entropy.ma_leaf_ctx.is_empty()
        && let Some(trees) = dc_entropy.ma_tree.as_ref()
    {
        let rates = dc_rdo::LearnedDcRates {
            tree: &trees.dc,
            leaf_ctx: &dc_entropy.ma_leaf_ctx,
            offsets: &dc_entropy.ma_offsets,
            code: &dc_entropy.code,
            wp_params: dc_entropy.wp_params,
        };
        let changed = dc_rdo::refine_dc_rounding_learned(
            &mut dc_datas,
            &distp,
            ytob_dc,
            dc_cfl,
            compensate_b,
            &rates,
            &ctx.thread_pool,
        )?;
        if changed {
            // The DC levels moved by a few steps: re-tokenize through the
            // same tree and warm-start the code from the pre-refit one
            // (recount, Slow refinement). On 30 encodes at Medium and
            // MediumSlow that matched a rebuild from scratch (bytes
            // −0.0006 % in all, −0.20..+0.03 % per image) at a third of
            // the call's time.
            let trees = trees.clone();
            let warmed = dc_entropy
                .dc_tokens_under(ctx, scratch, &dc_datas, dc_entropy.wp_params)
                .and_then(|dc| dc_entropy.warmed(ctx, scratch, dc, dc_entropy.wp_params));
            dc_entropy = match warmed {
                Some(entropy) => entropy,
                None => super::dc_entropy::prepare_dc_entropy(
                    ctx,
                    scratch,
                    &dc_datas,
                    dim.num_dc_groups,
                    epf_sharpness,
                    use_dc_frame,
                    distp.distance,
                    dc_entropy.wp_params,
                    Some(&trees),
                ),
            };
        }
    }
    let super::dc_entropy::DcEntropy {
        tree: dc_tree,
        dc: dc_tokens_per_group,
        meta: meta_tokens_per_group,
        code: mut dc_code_owned,
        refinement: ans_refinement,
        wp_params: dc_wp_params,
        ma_tree: _,
        ma_leaf_ctx: _,
        ma_offsets: _,
    } = dc_entropy;
    let (mut dc_tree, mut dc_tokens_per_group, mut meta_tokens_per_group) =
        (dc_tree, dc_tokens_per_group, meta_tokens_per_group);
    // The extra channel's sections only touch the DC streams and the AC
    // entropy plan never reads them, so the two run side by side (on its own
    // scratch); the extra channel's trees are a serial phase on small images.
    let mut extra_scratch = CoderScratch::default();
    let (extra, (ac_plan, ac_code_per_pass, raced_ac_sections, coeff_orders, all_pending)) =
        ctx.thread_pool.join(
            || {
                extra_coding.alpha.or(alpha).map(|plane| {
                    let snapshot = extra_coding.race.map(|_| {
                        (
                            dc_tree.clone(),
                            dc_code_owned.clone(),
                            dc_tokens_per_group.clone(),
                            meta_tokens_per_group.clone(),
                        )
                    });
                    // Ordinary extra codings can merge their own DC tree.
                    // Nested profiles use the original DC streams directly.
                    let race = extra_coding.race;
                    let levels_memo = super::extra_channels::LevelsMemo::default();
                    let cost = |indices: &[usize]| {
                        let Some((race, snapshot)) = race.zip(snapshot.as_ref()) else {
                            return Vec::new();
                        };
                        let job = |slot: usize| {
                            let index = indices[slot];
                            let alternative = &race.alternatives[index];
                            let mut scratch = CoderScratch::default();
                            let (candidate, streams) = match *alternative {
                                ExtraAlternative::Nested(fit) => (
                                    super::extra_channels::prepare_nested_sections(
                                        &ctx.thread_pool,
                                        &fit.model.profile,
                                        &fit.model.u,
                                        &fit.model.v,
                                        &dim,
                                        &fit.reconstructed,
                                        ctx.function_fields_portable,
                                        &mut scratch,
                                    )?,
                                    None,
                                ),
                                ExtraAlternative::Plane {
                                    plane,
                                    lossless,
                                    scale,
                                    ..
                                } => {
                                    let mut streams = snapshot.clone();
                                    let global = extra_tree::GlobalStreams {
                                        tree: &mut streams.0,
                                        code: &mut streams.1,
                                        dc: &mut streams.2,
                                        meta: &mut streams.3,
                                        dc_written: !use_dc_frame,
                                        refinement: ans_refinement,
                                    };
                                    let candidate = if lossless {
                                        prepare_extra_sections_lossless(
                                            ctx,
                                            plane,
                                            &dim,
                                            &group_coords,
                                            num_passes - 1,
                                            global,
                                            &mut scratch,
                                        )
                                    } else {
                                        super::extra_channels::prepare_extra_sections_scaled(
                                            ctx,
                                            plane,
                                            &dim,
                                            &group_coords,
                                            num_passes - 1,
                                            global,
                                            &mut scratch,
                                            scale,
                                            Some(&levels_memo),
                                        )
                                    };
                                    (candidate, Some(streams))
                                }
                            };
                            let bits = candidate.net_bits()
                                + if alternative.based() {
                                    race.base_bits
                                } else {
                                    0
                                };
                            Some((bits, index, candidate, streams))
                        };
                        let mut coded: Vec<_> = ctx
                            .thread_pool
                            .map(indices.len(), job)
                            .into_iter()
                            .flatten()
                            .collect();
                        // The first passing candidate sets the size window.
                        coded.sort_by_key(|&(bits, index, ..)| (bits, index));
                        coded
                    };
                    // The finer residuals only matter when nothing at the
                    // nominal distance passes, which is the rare case.
                    let (first, second): (Vec<usize>, Vec<usize>) = race
                        .map(|race| {
                            (0..race.alternatives.len())
                                .partition(|&index| !race.alternatives[index].fallback())
                        })
                        .unwrap_or_default();
                    // Costing an alternative takes working memory of the
                    // order of the frame's own (about 75 bytes a sample on
                    // top of the color state still held here): side by side
                    // with each other and the ordinary coding they tripled
                    // the encoder's peak on large planes. Above this size
                    // they are costed and judged one at a time, after the
                    // ordinary coding, and only the current pick is kept.
                    let streamed = dim.xsize * dim.ysize > PARALLEL_RACE_PIXELS;
                    let (mut prepared, first_round) = ctx.thread_pool.join(
                        || {
                            let nested = extra_coding.profile.and_then(|fit| {
                                super::extra_channels::prepare_nested_sections(
                                    &ctx.thread_pool,
                                    &fit.model.profile,
                                    &fit.model.u,
                                    &fit.model.v,
                                    &dim,
                                    &fit.reconstructed,
                                    ctx.function_fields_portable,
                                    &mut extra_scratch,
                                )
                            });
                            nested.unwrap_or_else(|| {
                                let streams = extra_tree::GlobalStreams {
                                    tree: &mut dc_tree,
                                    code: &mut dc_code_owned,
                                    dc: &mut dc_tokens_per_group,
                                    meta: &mut meta_tokens_per_group,
                                    dc_written: !use_dc_frame,
                                    refinement: ans_refinement,
                                };
                                if extra_coding.lossless {
                                    prepare_extra_sections_lossless(
                                        ctx,
                                        plane,
                                        &dim,
                                        &group_coords,
                                        num_passes - 1,
                                        streams,
                                        &mut extra_scratch,
                                    )
                                } else {
                                    prepare_extra_sections(
                                        ctx,
                                        plane,
                                        &dim,
                                        &group_coords,
                                        num_passes - 1,
                                        streams,
                                        &mut extra_scratch,
                                        race.is_some(),
                                    )
                                }
                            })
                        },
                        // A large plane costs nothing beside the ordinary
                        // coding: its alternatives follow one at a time.
                        || if streamed { Vec::new() } else { cost(&first) },
                    );
                    if let Some(race) = race {
                        let incumbent = prepared.reconstruction().unwrap_or(plane);
                        let incumbent_bits = prepared.net_bits();
                        let mut first_round = Some(first_round);
                        let mut winner = None;
                        // The same choice as the batch below, reached without
                        // holding the alternatives: sizes and distances of
                        // the passing ones are remembered, the coding of the
                        // current pick alone is kept, and a pick that falls
                        // out of the size window later hands over to one
                        // that is costed again if it was already let go.
                        for indices in [&first, &second] {
                            if !streamed {
                                break;
                            }
                            let mut seen = Vec::new();
                            let mut passing: Vec<(usize, f64, usize)> = Vec::new();
                            let mut kept = None;
                            let mut below = false;
                            let pick = |passing: &[(usize, f64, usize)]| {
                                let smallest = passing.iter().map(|p| p.0).min()?;
                                passing
                                    .iter()
                                    .filter(|p| p.0 * 10 <= smallest * 11)
                                    .min_by(|a, b| {
                                        a.1.total_cmp(&b.1).then((a.0, a.2).cmp(&(b.0, b.2)))
                                    })
                                    .map(|p| p.2)
                            };
                            let mut sizes: Vec<(usize, usize)> = Vec::new();
                            for &index in indices.iter() {
                                // Where the strong dead zone and none code to
                                // the same size, so does the weak one (27 of
                                // 27 such races): it is not costed.
                                let size =
                                    |of: usize| sizes.iter().find(|s| s.0 == of).map(|s| s.1);
                                if race.zones.is_some_and(|[strong, none, weak]| {
                                    index == weak
                                        && size(strong).is_some()
                                        && size(strong) == size(none)
                                }) {
                                    continue;
                                }
                                // The base frame alone already costs as much
                                // as the ordinary coding.
                                if race.alternatives[index].based()
                                    && race.base_bits >= incumbent_bits
                                {
                                    continue;
                                }
                                let Some((bits, index, candidate, streams)) =
                                    cost(&[index]).into_iter().next()
                                else {
                                    continue;
                                };
                                sizes.push((index, bits));
                                below |= bits < incumbent_bits;
                                let smallest = passing.iter().map(|p| p.0).min();
                                if bits >= incumbent_bits
                                    || seen.contains(&bits)
                                    || smallest.is_some_and(|size| bits * 10 > size * 11)
                                {
                                    continue;
                                }
                                seen.push(bits);
                                let decoded = candidate.reconstruction().unwrap_or(
                                    match &race.alternatives[index] {
                                        ExtraAlternative::Plane { plane, .. } => plane,
                                        ExtraAlternative::Nested(fit) => &fit.reconstructed,
                                    },
                                );
                                let Some(distance) = (race.judge)(index, incumbent, decoded) else {
                                    continue;
                                };
                                passing.push((bits, distance, index));
                                let picked = pick(&passing);
                                if picked == Some(index) {
                                    kept = Some((index, candidate, streams));
                                } else if kept
                                    .as_ref()
                                    .is_some_and(|k: &(usize, _, _)| Some(k.0) != picked)
                                {
                                    kept = None;
                                }
                            }
                            if let Some(picked) = pick(&passing) {
                                winner = kept.filter(|k| k.0 == picked).or_else(|| {
                                    cost(&[picked]).into_iter().next().map(
                                        |(_, index, candidate, streams)| {
                                            (index, candidate, streams)
                                        },
                                    )
                                });
                                break;
                            }
                            // The finer residuals are larger: with nothing
                            // below the ordinary coding's size at the nominal
                            // distance there is nothing for them to rescue
                            // (0 of 42 such rounds found a winner).
                            if !below {
                                break;
                            }
                        }
                        for _ in 0..2 {
                            if streamed {
                                break;
                            }
                            let costed = match first_round.take() {
                                Some(costed) => costed,
                                None => cost(&second),
                            };
                            // See the same test above.
                            let below = costed.first().is_some_and(|c| c.0 < incumbent_bits);
                            // The smallest that passes, unless one within a
                            // tenth of its size is closer to the source: a
                            // dead zone can save 5 % and double the error.
                            let mut smallest = None;
                            let mut previous_bits = None;
                            let mut best = None;
                            for (bits, index, candidate, streams) in costed {
                                if bits >= incumbent_bits
                                    || smallest.is_some_and(|size| bits * 10 > size * 11)
                                {
                                    break;
                                }
                                // Zoned and plain residuals often quantize alike.
                                if previous_bits.replace(bits) == Some(bits) {
                                    continue;
                                }
                                let decoded = candidate.reconstruction().unwrap_or(
                                    match &race.alternatives[index] {
                                        ExtraAlternative::Plane { plane, .. } => plane,
                                        ExtraAlternative::Nested(fit) => &fit.reconstructed,
                                    },
                                );
                                let Some(distance) = (race.judge)(index, incumbent, decoded) else {
                                    continue;
                                };
                                smallest.get_or_insert(bits);
                                if best.is_none_or(|score| distance.total_cmp(&score).is_lt()) {
                                    best = Some(distance);
                                    winner = Some((index, candidate, streams));
                                }
                            }
                            if winner.is_some() || !below {
                                break;
                            }
                        }
                        if let Some((index, candidate, streams)) = winner {
                            (
                                dc_tree,
                                dc_code_owned,
                                dc_tokens_per_group,
                                meta_tokens_per_group,
                            ) = streams
                                .or(snapshot)
                                .expect("the race retains its DC streams");
                            prepared = candidate;
                            if let Ok(mut chosen) = race.chosen.lock() {
                                *chosen = Some(index);
                            }
                        }
                    }
                    prepared
                })
            },
            || {
                let mut raced_ac_sections = None;
                // Separate scan arms before context planning: each gets its own final
                // context map, hybrid-uint configuration, clustering and ANS refinement.
                let mut alternative_pending: Vec<Vec<PendingAcGroup>> = (0..order_alternatives
                    .len())
                    .map(|_| Vec::with_capacity(all_pending.len()))
                    .collect();
                for p in &mut all_pending {
                    for (tokens, arm) in p.tokens.drain(num_passes..).zip(&mut alternative_pending)
                    {
                        arm.push(PendingAcGroup {
                            group_idx: p.group_idx,
                            tokens: vec![tokens],
                        });
                    }
                }
                let (ac_plan, ac_code_per_pass) = if order_alternatives.is_empty() {
                    finalize_ac_entropy(
                        ctx,
                        scratch,
                        &mut all_pending,
                        num_passes,
                        qf_threshold,
                        ans_refinement,
                    )
                } else {
                    let used_tables = used_quant_table_slots(&dc_datas);
                    let mut arms: Vec<OrderArm> = Vec::with_capacity(1 + order_alternatives.len());
                    // Stage 1: every arm gets its context plan, hybrid-uint configurations,
                    // clustering and the Slow refinement, then serializes.
                    for (orders, mut pending) in std::iter::once((coeff_orders, all_pending))
                        .chain(order_alternatives.into_iter().zip(alternative_pending))
                    {
                        let (plan, codes) = finalize_ac_entropy(
                            ctx,
                            scratch,
                            &mut pending,
                            num_passes,
                            qf_threshold,
                            ans_refinement.map(|_| crate::entropy::AnsRefinement::Slow),
                        );
                        let (bits, global, streams) = encode_order_arm(
                            ctx,
                            scratch,
                            &pending,
                            &plan,
                            &codes,
                            &orders,
                            &used_tables,
                            dim.num_groups,
                            dim.num_dc_groups,
                        );
                        arms.push(OrderArm {
                            orders,
                            pending,
                            plan,
                            codes,
                            bits,
                            global,
                            streams,
                        });
                    }
                    // Stage 2: the exhaustive table search moves every arm by nearly the
                    // same amount, so only arms within reach of the leader buy it.
                    let leader = arms.iter().map(|arm| arm.bits).min().unwrap_or(0);
                    let reach = leader + (leader as f64 * ORDER_RACE_REFINE_MARGIN) as usize;
                    if matches!(
                        ans_refinement,
                        Some(crate::entropy::AnsRefinement::ExtraSlow)
                    ) {
                        for arm in arms.iter_mut().filter(|arm| arm.bits <= reach) {
                            for (pass, code) in arm.codes.iter_mut().enumerate() {
                                crate::entropy::refine_ans_entropy_exhaustive(
                                    code,
                                    arm.pending.iter().map(|p| p.tokens[pass].as_slice()),
                                    &ctx.thread_pool,
                                    scratch,
                                );
                            }
                            (arm.bits, arm.global, arm.streams) = encode_order_arm(
                                ctx,
                                scratch,
                                &arm.pending,
                                &arm.plan,
                                &arm.codes,
                                &arm.orders,
                                &used_tables,
                                dim.num_groups,
                                dim.num_dc_groups,
                            );
                        }
                    }
                    // Prefer natural on ties, otherwise retain the incumbent.
                    let mut best = 0;
                    for (i, arm) in arms.iter().enumerate().skip(1) {
                        if arm.bits < arms[best].bits
                            || (arm.bits == arms[best].bits && arm.orders.used_mask == 0)
                        {
                            best = i;
                        }
                    }
                    let winner = arms.swap_remove(best);
                    coeff_orders = winner.orders;
                    all_pending = winner.pending;
                    // Retain the winning serialized sections; do not encode them again.
                    raced_ac_sections = Some((winner.global, winner.streams));
                    (winner.plan, winner.codes)
                };
                (
                    ac_plan,
                    ac_code_per_pass,
                    raced_ac_sections,
                    coeff_orders,
                    all_pending,
                )
            },
        );
    let dc_code = dc_code_owned.as_ref();

    // Write DC global with adaptive DC code.
    if let VarDctFrameKind::Patched(references, _, dictionary) = frame_kind {
        if let Some(dictionary) = dictionary.filter(|_| alpha.is_none()) {
            sections[0].append_bits(dictionary);
        } else {
            crate::lossless::write_patch_dictionary(
                references,
                alpha.is_some(),
                scratch,
                &mut sections[0],
            );
        }
    }
    #[cfg(feature = "splines")]
    if let Some(set) = splines {
        crate::splines::write_splines(set, scratch, &mut sections[0]);
    }
    if let Some(noise) = &noise {
        noise.write(&mut sections[0]);
    }
    write_dc_global(
        &distp,
        dim.num_dc_groups,
        &ac_plan,
        &dc_tree,
        &dc_code,
        ytox_dc,
        ytob_dc,
        ctx.cfl_frame(),
        scratch,
        &mut sections[0],
    );

    // Each DC group is an independent entropy stream. Encode the
    // streams in parallel, then place their writers back in raster order.
    let dc_sections = ctx
        .thread_pool
        .steal_map(scratch, dc_datas.len(), |i, _scratch| {
            let dc_data = &dc_datas[i];
            let mut w = BitWriter::new();
            // With a DC frame, the group carries no DC stream.
            if !use_dc_frame {
                w.write(2, 0); // extra_dc_precision
                w.write(1, 1); // use_global_tree
                crate::weighted_predictor::write_wp_header(dc_wp_params, &mut w);
                w.write(2, 0); // no transforms
                if dc_code.use_prefix_code {
                    for t in &dc_tokens_per_group[i] {
                        write_token(*t, &dc_code, &mut w);
                    }
                } else {
                    crate::entropy::write_ans_tokens(
                        &dc_tokens_per_group[i],
                        dc_code.context_map,
                        dc_code.ans_symbols,
                        dc_code.ans_reverse_maps,
                        dc_code.hybrid_uint_configs,
                        &mut w,
                    );
                }
            }
            // Keep the metadata suffix separate until the extra-channel
            // representation is selected using final section sizes.
            let prefix = w;
            let mut w = BitWriter::new();
            let num_blocks = dc_data.ac_strategy.xsize() * dc_data.ac_strategy.ysize();
            let num_ac_blocks = dc_data.ac_strategy.count_first_blocks();
            let nb_bits = if num_blocks <= 1 {
                0
            } else {
                32 - (num_blocks as u32).leading_zeros() as usize
                    - if num_blocks.is_power_of_two() { 1 } else { 0 }
            };
            if nb_bits != 0 {
                w.write(nb_bits, (num_ac_blocks - 1) as u64);
            }
            w.write(4, 3);
            if dc_code.use_prefix_code {
                for t in &meta_tokens_per_group[i] {
                    write_token(*t, &dc_code, &mut w);
                }
            } else {
                crate::entropy::write_ans_tokens(
                    &meta_tokens_per_group[i],
                    dc_code.context_map,
                    dc_code.ans_symbols,
                    dc_code.ans_reverse_maps,
                    dc_code.hybrid_uint_configs,
                    &mut w,
                );
            }
            (prefix, w)
        });
    let mut dc_suffixes = Vec::with_capacity(dc_sections.len());
    for (i, (prefix, suffix)) in dc_sections.into_iter().enumerate() {
        sections[1 + i] = prefix;
        dc_suffixes.push(suffix);
    }
    drop(dc_tokens_per_group);
    drop(meta_tokens_per_group);

    let (ac_global, ac_sections) = if let Some(raced) = raced_ac_sections {
        raced
    } else {
        let mut global = BitWriter::new();
        write_ac_global(
            ctx.matrices(),
            &used_quant_table_slots(&dc_datas),
            &coeff_orders,
            dim.num_groups,
            &ac_code_per_pass,
            scratch,
            &mut global,
        );
        let ac = encode_ac_sections(
            ctx,
            scratch,
            &all_pending,
            &ac_code_per_pass,
            dim.num_dc_groups,
        );
        (global, ac)
    };
    sections[1 + dim.num_dc_groups] = ac_global;
    for (section_idx, section) in ac_sections {
        sections[section_idx] = section;
    }
    drop(all_pending);
    // Color coding is complete. Compare exact section padding and TOC costs
    // before inserting the chosen extra channel at its three wire positions.
    let extra = extra.map(|prepared| {
        let mut color_bits: Vec<usize> = sections.iter().map(BitWriter::bits_written).collect();
        for (i, suffix) in dc_suffixes.iter().enumerate() {
            color_bits[1 + i] += suffix.bits_written();
        }
        prepared.finish(
            ctx,
            &dim,
            &group_coords,
            num_passes - 1,
            &color_bits,
            scratch,
        )
    });
    if let Some(extra) = &extra {
        sections[0].append(&extra.global);
    }
    for (i, suffix) in dc_suffixes.into_iter().enumerate() {
        if let Some(extra) = &extra {
            sections[1 + i].append(&extra.lf[i]);
        }
        sections[1 + i].append(&suffix);
    }
    // Modular alpha: extra-channel modular data is decoded in the last pass
    // (the only pass whose modular sub-image shift range is set when num_ds=0,
    // jxl-frame lib.rs:101-108 / pass_group.rs:93). Write it into that section.
    if let Some(extra) = &extra {
        let last_pass = num_passes - 1;
        for (abs_group_id, section) in extra.ac.iter().enumerate() {
            sections[2 + dim.num_dc_groups + last_pass * dim.num_groups + abs_group_id]
                .append(section);
        }
    }

    // The DC frame precedes the frame that uses it.
    let mut dc_frame_bits = 0;
    if use_dc_frame {
        let dc_image = assemble_source_dc(&dc_datas, &group_coords, &dim);
        let mut dc_writer = BitWriter::new();
        let coded = crate::lossy_modular::encode_dc_frame(
            &dc_image,
            std::array::from_fn(|c| {
                distp.dc_step[c] / (crate::quant_weights::INV_DC_QUANT[c] * distp.scale_dc)
            }),
            ctx.progressive_dc_levels,
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut dc_writer,
        );
        debug_assert!(coded, "DC frame size checked up front");
        dc_frame_bits = dc_writer.bits_written();
        writer.append(&dc_writer);
    }

    #[cfg(feature = "splines")]
    let has_splines = splines.is_some();
    #[cfg(not(feature = "splines"))]
    let has_splines = false;
    let role = extra_coding.role.unwrap_or(ctx.frame_role);
    // A residual alternative won: this frame follows the function frame and
    // adds its extra channel to it.
    let role = match extra_coding.race {
        Some(race)
            if race
                .chosen
                .lock()
                .ok()
                .and_then(|chosen| *chosen)
                .is_some_and(|index| race.alternatives[index].based()) =>
        {
            if let Some(based) = race.based.lock().ok().and_then(|mut based| based.take()) {
                *writer = based;
            }
            FrameRole::LastExtraAdd
        }
        _ => role,
    };
    write_frame_header_kind(
        distp.x_qm_scale,
        ctx.b_qm_scale(),
        distp.epf_iters,
        distp.epf_pass0_scale,
        distp.epf_quant_mul,
        distp.gab_enabled,
        alpha.is_some(),
        coeff_shifts,
        frame_kind,
        has_splines,
        noise.is_some(),
        skip_dc_smoothing,
        use_dc_frame,
        ctx.coding,
        ctx.epf_channel_scale,
        role,
        writer,
    );
    let payload_bits = dc_frame_bits
        + sections
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 0 && *i != 1 + dim.num_dc_groups)
            .map(|(_, section)| section.bits_written())
            .sum::<usize>();
    let order = (ctx.group_order != crate::GroupOrder::Scanline
        && matches!(
            frame_kind,
            VarDctFrameKind::Regular | VarDctFrameKind::Patched(..)
        ))
    .then(|| group_section_order(&dim, num_passes, ctx.group_order));
    combine_sections_ordered(
        &mut sections,
        order.as_deref().map(|o| (o, &mut scratch.huffman_pool)),
        writer,
    );
    Ok((payload_bits, survival))
}
