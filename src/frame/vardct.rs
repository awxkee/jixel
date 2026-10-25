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
    GLYPH_BITS_MARGIN, GlyphStage, LATTICE_BITS_MARGIN, lattice_variants, prepare_vardct_variant,
};
#[cfg(feature = "splines")]
use super::variants::{SplineStage, spline_quant_field};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::util::EncodeError;

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
            xyb,
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
    let glyphs = GlyphStage::find(ctx, scratch, distance, &xyb, alpha);
    let mut best = prepare_vardct_variant(
        ctx,
        scratch,
        distance,
        distp,
        xyb.clone(),
        alpha,
        coeff_shifts,
        true,
        glyphs.as_ref(),
        true,
        #[cfg(feature = "splines")]
        stage.as_ref(),
    )?;
    if !best.used_tiles && lattice.is_none() {
        drop(xyb);
        drop(glyphs);
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        return best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer);
    }
    #[allow(unused_mut)]
    let mut best_bits = best.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
    if let Some(plan) = lattice {
        drop(xyb);
        drop(glyphs);
        #[cfg(feature = "splines")]
        drop(spline_candidates);
        // A lattice arm codes what the tile leaves at the frame's own distance,
        // where an exact-tile plan repeats it at the finer atlas distance, so
        // near-equal bits buy only a coarser remainder.
        best_bits *= LATTICE_BITS_MARGIN;
        for mut arm in lattice_variants(ctx, scratch, distance, distp, plan, alpha)? {
            let bits = arm.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
            if bits < best_bits {
                best = arm;
                best_bits = bits;
            }
        }
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
            xyb.clone(),
            alpha,
            coeff_shifts,
            true,
            glyphs.as_ref(),
            true,
            Some(&stage),
        )?;
        let bits = splines_first.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
        if bits < best_bits {
            best = splines_first;
            best_bits = bits;
        }
    }
    let mut no_tiles = prepare_vardct_variant(
        ctx,
        scratch,
        distance,
        distp,
        xyb,
        alpha,
        coeff_shifts,
        true,
        glyphs.as_ref(),
        false,
        #[cfg(feature = "splines")]
        stage.as_ref(),
    )?;
    let no_tile_bits = no_tiles.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
    // Tile atlases are finer than the residual, so dropping them must repay
    // the same quality margin as the previous full-bitstream comparison.
    if no_tile_bits * GLYPH_BITS_MARGIN < best_bits {
        best = no_tiles;
    } else {
        drop(no_tiles);
    }
    drop(glyphs);
    #[cfg(feature = "splines")]
    drop(spline_candidates);
    best.encode(ctx, scratch, distance, alpha, coeff_shifts, writer)
}
