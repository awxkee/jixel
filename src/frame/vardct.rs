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

//! Choose a regular, patched or spline-subtracted VarDCT frame.

use super::chroma::configure_chroma;
use super::quantization::DistanceParams;
use super::variants::{
    GLYPH_BITS_MARGIN, GLYPH_SINGLETON_FREE_MAX_DISTANCE, GLYPH_SINGLETON_RACE_MAX_DISTANCE,
    GlyphStage, LATTICE_BITS_MARGIN, PreparedVarDct, UNPATCHED_BITS_MARGIN, lattice_variants,
    prepare_vardct_variant,
};
#[cfg(feature = "splines")]
use super::variants::{SplineStage, spline_quant_field};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::util::EncodeError;
use std::borrow::Cow;

/// The VarDCT arm of lossy encoding (regular or patched), from a prepared XYB
/// image.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_frame_vardct(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    distp: &DistanceParams,
    xyb: Image3F,
    is_achromatic: bool,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    patches: bool,
    writer: &mut BitWriter,
) -> Result<(), EncodeError> {
    configure_chroma(ctx, distance, distp, &xyb, is_achromatic);
    #[cfg(feature = "splines")]
    let quant_field = if ctx.splines || ctx.dots {
        spline_quant_field(ctx, scratch, &xyb, distp)?
    } else {
        Vec::new()
    };
    #[cfg(feature = "splines")]
    let spline_candidates = if ctx.splines || ctx.dots {
        crate::splines::find_candidates(ctx, scratch, distance, &xyb, &quant_field)
    } else {
        None
    };
    #[cfg(feature = "splines")]
    let stage = spline_candidates.as_ref().map(|candidates| SplineStage {
        candidates,
        quant_field: &quant_field,
        before_tiles: false,
    });
    if !patches {
        let prepared = prepare_vardct_variant(
            ctx,
            scratch,
            distance,
            distp,
            Cow::Owned(xyb),
            alpha,
            coeff_shifts,
            false,
            None,
            true,
            #[cfg(feature = "splines")]
            stage.as_ref(),
        )?;
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        return prepared.encode(ctx, scratch, distance, alpha, coeff_shifts, writer);
    }

    // Compare bounded residual samples plus the actual atlas, dictionary and
    // spline costs. Only the selected plan receives a full-frame encode.
    let lattice = crate::lattice_patches::find(&xyb);
    let retain_epf_luma = ctx.patch_epf
        && super::vardct_encoder::epf_fit_enabled(ctx, distance)
        && !ctx.progressive_dc
        && coeff_shifts == [0]
        && xyb.xsize() >= 16
        && xyb.ysize() >= 16;
    let mut glyphs = GlyphStage::find(
        ctx,
        scratch,
        distance,
        &xyb,
        alpha,
        distance <= GLYPH_SINGLETON_FREE_MAX_DISTANCE,
        retain_epf_luma,
    );
    let mut best = prepare_vardct_variant(
        ctx,
        scratch,
        distance,
        distp,
        Cow::Borrowed(&xyb),
        alpha,
        coeff_shifts,
        true,
        glyphs.as_ref().map(Cow::Borrowed),
        true,
        #[cfg(feature = "splines")]
        stage.as_ref(),
    )?;
    // Text set in unrepeated glyphs: the atlas codes it near-exactly, so a
    // plan with them that is also smaller dominates the repeated-only one.
    // The race arm skips spline selection, which only costs it bits; a
    // winner is rebuilt with splines.
    if distance > GLYPH_SINGLETON_FREE_MAX_DISTANCE
        && distance <= GLYPH_SINGLETON_RACE_MAX_DISTANCE
        && let Some(singles) =
            GlyphStage::find(ctx, scratch, distance, &xyb, alpha, true, retain_epf_luma)
    {
        let best_bits = best.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
        if (singles.atlas_bits() as f64) < best_bits {
            let mut arm = prepare_vardct_variant(
                ctx,
                scratch,
                distance,
                distp,
                Cow::Borrowed(&xyb),
                alpha,
                coeff_shifts,
                true,
                Some(Cow::Borrowed(&singles)),
                true,
                #[cfg(feature = "splines")]
                None,
            )?;
            if arm.estimate(ctx, scratch, distance, alpha, coeff_shifts)? < best_bits {
                #[cfg(feature = "splines")]
                if stage.is_some() {
                    let full = prepare_vardct_variant(
                        ctx,
                        scratch,
                        distance,
                        distp,
                        Cow::Borrowed(&xyb),
                        alpha,
                        coeff_shifts,
                        true,
                        Some(Cow::Borrowed(&singles)),
                        true,
                        stage.as_ref(),
                    )?;
                    if full.has_splines() {
                        arm = full;
                    } else {
                        arm.tiles_touched = full.tiles_touched;
                    }
                }
                best = arm;
                glyphs = Some(singles);
            }
        }
    }
    // Glyph and tile plans race each other, and the frame without any
    // patches races the winner: on photo-heavy composites (store and
    // marketing screenshots, phone mock-ups with tiny perspective text)
    // thousands of near-glyph references cost more than they take out of
    // the residual. The unpatched frame wins on its estimate alone (see
    // `UNPATCHED_BITS_MARGIN`): a plan that is not smaller is not sharper
    // either.
    // Only plans that found something pay for the extra estimate. The base
    // frame under an ink layer keeps its patches: there the sampled estimate
    // prices the unpatched base far below its real size (an ink-stacked text
    // screenshot +10 % at d1 without them). A crop stack's base is the image
    // itself less some rectangles and races like a whole frame (one store
    // screen's base 109 -> 89 kB at d1.2).
    let race_unpatched = (ctx.frame_role == crate::encoding_context::FrameRole::Last
        || ctx.crop_base)
        && (glyphs.is_some() || best.used_tiles || lattice.is_some());
    // An unchanged unpatched arm can price the borrowed source and only
    // take ownership of the image if it wins the final rate comparison.
    let defer_unpatched = race_unpatched && glyphs.is_some() && !distp.gab_enabled;
    #[cfg(feature = "splines")]
    let defer_unpatched = defer_unpatched && stage.is_none();
    if !best.used_tiles || lattice.is_some() {
        // No more variants need the glyph residual or atlas.
        drop(glyphs.take());
    }
    let mut unpatched = if defer_unpatched {
        let bits =
            PreparedVarDct::estimate_regular(ctx, scratch, distance, &xyb, alpha, coeff_shifts)?;
        Some((None, bits))
    } else if race_unpatched {
        let mut arm = prepare_vardct_variant(
            ctx,
            scratch,
            distance,
            distp,
            Cow::Borrowed(&xyb),
            alpha,
            coeff_shifts,
            false,
            None,
            true,
            #[cfg(feature = "splines")]
            stage.as_ref(),
        )?;
        let bits = arm.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
        Some((Some(arm), bits))
    } else {
        None
    };
    let mut pick = |mut best: PreparedVarDct,
                    best_bits: Option<f64>,
                    original: Option<Image3F>,
                    scratch: &mut CoderScratch|
     -> Result<PreparedVarDct, EncodeError> {
        let Some((arm, bits)) = unpatched.take() else {
            return Ok(best);
        };
        let best_bits = match best_bits {
            Some(bits) => bits,
            None => best.estimate(ctx, scratch, distance, alpha, coeff_shifts)?,
        };
        Ok(if bits * UNPATCHED_BITS_MARGIN < best_bits {
            arm.unwrap_or_else(|| PreparedVarDct::regular(original.expect("unpatched source")))
        } else {
            best
        })
    };
    if !best.used_tiles && lattice.is_none() {
        let original = defer_unpatched.then_some(xyb);
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        let best = pick(best, None, original, scratch)?;
        return best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer);
    }
    #[allow(unused_mut)]
    let mut best_bits = best.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
    if let Some(plan) = lattice {
        let original = defer_unpatched.then_some(xyb);
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        // A lattice arm codes what the tile leaves at the frame's own distance,
        // where an exact-tile plan repeats it at the finer atlas distance, so
        // near-equal bits buy only a coarser remainder.
        let patched_bits = best_bits;
        best_bits *= LATTICE_BITS_MARGIN;
        let mut lattice_won = false;
        for mut arm in lattice_variants(ctx, scratch, distance, distp, plan, alpha)? {
            let bits = arm.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
            if bits < best_bits {
                best = arm;
                best_bits = bits;
                lattice_won = true;
            }
        }
        let best = pick(
            best,
            Some(if lattice_won { best_bits } else { patched_bits }),
            original,
            scratch,
        )?;
        return best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer);
    }
    #[cfg(feature = "splines")]
    if best.tiles_touched {
        let stage = SplineStage {
            before_tiles: true,
            ..stage.expect("candidates touched a tile")
        };
        let mut splines_first = prepare_vardct_variant(
            ctx,
            scratch,
            distance,
            distp,
            Cow::Borrowed(&xyb),
            alpha,
            coeff_shifts,
            true,
            glyphs.as_ref().map(Cow::Borrowed),
            true,
            Some(&stage),
        )?;
        let bits = splines_first.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
        if bits < best_bits {
            best = splines_first;
            best_bits = bits;
        }
    }
    let mut original = Some(xyb);
    let input = if defer_unpatched {
        Cow::Borrowed(original.as_ref().unwrap())
    } else {
        Cow::Owned(original.take().unwrap())
    };
    let mut no_tiles = prepare_vardct_variant(
        ctx,
        scratch,
        distance,
        distp,
        input,
        alpha,
        coeff_shifts,
        true,
        glyphs.map(Cow::Owned),
        false,
        #[cfg(feature = "splines")]
        stage.as_ref(),
    )?;
    let no_tile_bits = no_tiles.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
    // Tile atlases are finer than the residual, so dropping them must repay
    // the same quality margin as the previous full-bitstream comparison.
    if no_tile_bits * GLYPH_BITS_MARGIN < best_bits {
        best = no_tiles;
        best_bits = no_tile_bits;
    } else {
        drop(no_tiles);
    }
    #[cfg(feature = "splines")]
    drop(spline_candidates);
    let best = pick(best, Some(best_bits), original, scratch)?;
    best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer)
}
