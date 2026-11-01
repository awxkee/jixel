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

//! Select the lossy modular or VarDCT encoding arm and prepare its input.

use super::modular::{LM_GATE_MARGIN, lm_plain_input, lm_texture_factor};
use super::quantization::compute_distance_params;
use super::vardct::encode_frame_vardct;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::util::EncodeError;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    linear: Image3F,
    coded: Option<Image3F>,
    is_achromatic: bool,
    alpha: Option<&AlphaPlane>,
    fill_invisible: bool,
    coeff_shifts: &[u32],
    patches: bool,
    integer_samples: bool,
    writer: &mut BitWriter,
) -> Result<bool, EncodeError> {
    let distp = compute_distance_params(distance);
    if !ctx.coding.is_xyb() {
        // Non-XYB planes skip every XYB content heuristic: they read as
        // achromatic to the classifiers, and patches quantize on the XYB
        // lattice.
        let mut coded = coded.expect("non-XYB frames carry their coded planes");
        if fill_invisible && let Some(alpha) = alpha {
            crate::invisible::fill_invisible(&mut coded, alpha);
        }
        // The lossy-modular arm codes the integer samples themselves (gray
        // as its one channel; alpha or K as an extra channel).
        let modular = |scratch: &mut CoderScratch| -> Option<(BitWriter, usize)> {
            if !integer_samples {
                return None;
            }
            let mut w = BitWriter::new();
            let original_bits = crate::lossy_modular::encode_plain_frame(
                &lm_plain_input(ctx, &coded, is_achromatic, alpha)?,
                coded.xsize(),
                coded.ysize(),
                crate::lossy_modular::plain_distance(ctx.coding, distance)
                    * lm_texture_factor(&linear),
                ctx.speed,
                &ctx.thread_pool,
                scratch,
                &mut w,
            )?;
            Some((w, original_bits))
        };
        let auto =
            ctx.lossy_modular == crate::LossyModular::Auto && ctx.speed.effort().lossy_modular_auto;
        let modular_writer = match ctx.lossy_modular {
            crate::LossyModular::Force => match modular(scratch) {
                Some((w, _)) => {
                    writer.append(&w);
                    return Ok(true);
                }
                None => None,
            },
            crate::LossyModular::Auto if auto => modular(scratch),
            _ => None,
        };
        let mut aq_source = to_xyb_image(ctx, scratch, linear);
        if fill_invisible && let Some(alpha) = alpha {
            crate::invisible::fill_invisible(&mut aq_source, alpha);
        }
        let _ = ctx.aq_source.set(aq_source);
        ctx.set_plain_matrices(&coded);
        ctx.set_cfl_frame(crate::coding::fit_cfl_frame(ctx.coding, &coded));
        let mut vardct_writer = BitWriter::new();
        encode_frame_vardct(
            ctx,
            scratch,
            distance,
            &distp,
            coded,
            true,
            alpha,
            coeff_shifts,
            false,
            &mut vardct_writer,
        )?;
        return Ok(match modular_writer {
            Some((m, original_bits))
                if (original_bits as f64) * LM_GATE_MARGIN
                    < vardct_writer.bits_written() as f64 =>
            {
                writer.append(&m);
                true
            }
            _ => {
                writer.append(&vardct_writer);
                false
            }
        });
    }
    let lossy_modular = alpha.is_none()
        && (ctx.lossy_modular == crate::LossyModular::Force
            || (ctx.lossy_modular == crate::LossyModular::Auto
                && ctx.speed.effort().lossy_modular_auto));
    let texture = lossy_modular.then(|| lm_texture_factor(&linear));
    let mut xyb = to_xyb_image(ctx, scratch, linear);
    if fill_invisible && let Some(alpha) = alpha {
        crate::invisible::fill_invisible(&mut xyb, alpha);
    }
    if is_achromatic {
        snap_achromatic_xyb(&mut xyb);
    }
    let screen = lossy_modular && crate::frame::is_screen_content(&xyb);
    let modular_distance = texture.map_or(0.0, |texture| {
        crate::lossy_modular::calibrated_distance(distance, screen) * texture
    });
    // Lossy-modular arm (quantized Squeeze): whole-frame modular encoding of
    // the XYB image at a distance calibrated to match the VarDCT arm's
    // quality. Force always takes the arm when the frame supports it; Auto
    // (Slow only) dual-encodes screen content and keeps the smaller frame.
    // On photos the byte gate only picks the arm where its calibration runs
    // coarse for that image: 16 Kodak images at Slow, Auto against VarDCT
    // alone SS2 -0.31 / BA +0.19 / cvvdp +1.38 % BD, so they skip it.
    if alpha.is_none()
        && ctx.lossy_modular == crate::LossyModular::Force
        && let Some((modular_writer, _)) = encode_modular_xyb(
            ctx,
            scratch,
            distance,
            modular_distance,
            screen,
            &xyb,
            patches,
        )
    {
        writer.append(&modular_writer);
        return Ok(false);
    }
    if screen && ctx.lossy_modular == crate::LossyModular::Auto {
        let modular = encode_modular_xyb(
            ctx,
            scratch,
            distance,
            modular_distance,
            screen,
            &xyb,
            patches,
        );
        let original_modular_bits = modular.as_ref().map(|(_, bits)| *bits);
        let modular_writer = modular.map(|(w, _)| w).unwrap_or_default();
        let mut vardct_writer = BitWriter::new();
        encode_frame_vardct(
            ctx,
            scratch,
            distance,
            &distp,
            xyb,
            is_achromatic,
            alpha,
            coeff_shifts,
            patches,
            &mut vardct_writer,
        )?;
        // Preserve the original arm decision: a better representation must
        // not swap VarDCT for a different lossy reconstruction in Auto mode.
        if original_modular_bits.is_some_and(|bits| {
            (bits as f64) * LM_GATE_MARGIN < vardct_writer.bits_written() as f64
        }) {
            writer.append(&modular_writer);
        } else {
            writer.append(&vardct_writer);
        }
        return Ok(false);
    }
    encode_frame_vardct(
        ctx,
        scratch,
        distance,
        &distp,
        xyb,
        is_achromatic,
        alpha,
        coeff_shifts,
        patches,
        writer,
    )?;
    Ok(false)
}

/// The lossy-modular arm of an XYB frame. With `patches`, repeated glyphs
/// go to an atlas reference frame and the frame codes the image less them:
/// on screenshots that is -11.7 % SS2 / -9.5 % BA BD at Slow, so a found
/// plan is taken without a second, unpatched encode (a byte race would also
/// prefer the plain frame where the patched one is slightly larger but
/// better). Unrepeated glyphs join as in VarDCT: freely up to
/// `GLYPH_SINGLETON_FREE_MAX_DISTANCE`, and up to the race distance when the
/// frame with them is smaller (text set in unrepeated glyphs: screen0 at d
/// 1.25 -15 % bytes and +1.6 SS2, where taking them freely there loses BA
/// +1.0 % BD over 22 screenshots and illustrations). Only a singleton atlas
/// of at least a quarter of the frame races: winners measure 0.33-0.58 of
/// it, scattered singletons on illustrations 0.02-0.13. Returns the writer and
/// the Auto gate's size (before the fold race, atlas included), or None
/// when the image is too small to squeeze.
fn encode_modular_xyb(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    modular_distance: f32,
    screen: bool,
    xyb: &Image3F,
    patches: bool,
) -> Option<(BitWriter, usize)> {
    use super::variants::{
        GLYPH_SINGLETON_FREE_MAX_DISTANCE, GLYPH_SINGLETON_RACE_MAX_DISTANCE, GlyphStage,
    };
    let encode = |glyphs: Option<GlyphStage>, scratch: &mut CoderScratch| {
        let (mut writer, image, references) = match glyphs {
            Some(glyphs) => {
                let (atlas, base, references) = glyphs.into_parts();
                (atlas, std::borrow::Cow::Owned(base), references)
            }
            None => (
                BitWriter::new(),
                std::borrow::Cow::Borrowed(xyb),
                Vec::new(),
            ),
        };
        let atlas_bits = writer.bits_written();
        let bits = crate::lossy_modular::encode_xyb_frame(
            &image,
            modular_distance,
            screen,
            &references,
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut writer,
        )?;
        Some((writer, atlas_bits + bits))
    };
    let free_singles = distance <= GLYPH_SINGLETON_FREE_MAX_DISTANCE;
    let glyphs = patches
        .then(|| GlyphStage::find(ctx, scratch, distance, xyb, None, free_singles))
        .flatten();
    let best = encode(glyphs, scratch)?;
    // The race encodes the frame twice (1.8x on the screenshots it runs on),
    // so it stays at Slow and above, where Auto runs.
    if !patches
        || free_singles
        || distance > GLYPH_SINGLETON_RACE_MAX_DISTANCE
        || ctx.speed < crate::Speed::Slow
    {
        return Some(best);
    }
    let best_bits = best.0.bits_written();
    match GlyphStage::find(ctx, scratch, distance, xyb, None, true) {
        Some(singles)
            if singles.atlas_bits() < best_bits && singles.atlas_bits() * 4 >= best_bits =>
        {
            match encode(Some(singles), scratch) {
                Some(arm) if arm.0.bits_written() < best_bits => Some(arm),
                _ => Some(best),
            }
        }
        _ => Some(best),
    }
}

/// Every opsin row sums to 1, so achromatic input gives L = M = S and hence
/// X = 0, B = Y, up to f32 rounding in the three mixes. Snap the planes exact
/// so chroma quantizes to zeros and CfL sees slope 0/1 instead of noise.
fn snap_achromatic_xyb(xyb: &mut Image3F) {
    for y in 0..xyb.ysize() {
        let [x_row, y_row, b_row] = xyb.all_plane_rows_mut(y);
        x_row.fill(0.0);
        b_row.copy_from_slice(y_row);
    }
}

fn to_xyb_image(ctx: &EncodingContext, scratch: &mut CoderScratch, mut linear: Image3F) -> Image3F {
    crate::xyb::to_xyb_in_place_with_fn(
        ctx.to_xyb_band,
        &ctx.xyb,
        &mut linear,
        &ctx.thread_pool,
        scratch,
    );
    linear
}
