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

//! Prepare patch atlases and spline variants, and estimate their residual rate.

use super::extra_channels::zero_alpha_for_lossy;
use super::quantization::{DistanceParams, compute_distance_params, gaborize};
use super::vardct_encoder::encode_frame_core;
#[cfg(feature = "splines")]
use super::{K_BLOCK_DIM, K_DC_GROUP_DIM};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::patches::{
    EpfPatchAtlases, GlyphAtlasLuma, MODULAR_PATCH_REF_ID, PATCH_REF_ID, VarDctFrameKind,
};
use crate::util::EncodeError;
use std::borrow::Cow;
use std::sync::Arc;

/// The kAdd and kMul lattice arms of a frame whose luma repeats on an integer
/// lattice; both race the tile/glyph plans on estimated bits.
#[allow(clippy::too_many_arguments)]
pub(super) fn lattice_variants(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    distp: &DistanceParams,
    plan: crate::lattice_patches::LatticePlan,
    alpha: Option<&AlphaPlane>,
) -> Result<Vec<PreparedVarDct>, EncodeError> {
    let (px, py) = plan.period;
    let add_references = plan.add_references();
    let mul_references = plan.mul_references();
    let atlas_alpha = zero_alpha_for_lossy(alpha, px * py);
    let modular_atlas = |atlas: &[Vec<i32>; 3], scratch: &mut CoderScratch| {
        let mut bits = BitWriter::new();
        crate::lossless::encode_modular_xyb_atlas_tree(
            atlas,
            px,
            py,
            atlas_alpha.as_ref(),
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut bits,
        );
        bits
    };
    let prepared =
        |prefix: BitWriter, mut base: Image3F, references: Vec<crate::patches::PatchReference>| {
            gaborize(&mut base, distp);
            PreparedVarDct {
                prefix,
                base,
                references,
                epf_atlases: None,
                #[cfg(feature = "splines")]
                splines: None,
                tiles_touched: false,
                used_tiles: false,
                estimated_bits: None,
            }
        };
    let add = prepared(
        modular_atlas(&plan.add_atlas, scratch),
        plan.add_base,
        add_references,
    );
    let mut mul_prefix = modular_atlas(&plan.mul_atlas, scratch);
    let (w, h) = (plan.residual.xsize(), plan.residual.ysize());
    let mut residual = plan.residual;
    gaborize(&mut residual, distp);
    let residual_alpha = zero_alpha_for_lossy(alpha, w * h);
    encode_frame_core(
        ctx,
        scratch,
        distance,
        residual,
        residual_alpha.as_ref(),
        &[0],
        VarDctFrameKind::ReferenceOnly {
            width: w,
            height: h,
        },
        #[cfg(feature = "splines")]
        None,
        #[cfg(feature = "splines")]
        false,
        crate::noise::GrainPass::Inherit,
        &mut mul_prefix,
    )?;
    let mul = prepared(mul_prefix, plan.gain, mul_references);
    Ok(vec![add, mul])
}

/// Spline selection of one frame variant.
#[cfg(feature = "splines")]
#[derive(Clone, Copy)]
pub(super) struct SplineStage<'a> {
    pub(super) candidates: &'a crate::splines::SplineCandidates,
    pub(super) quant_field: &'a [f32],
    /// Select before the Replace tiles are detected: splines may then cross
    /// tiles, which are cut from the spline-subtracted image.
    pub(super) before_tiles: bool,
}

/// A frame's glyph patches and their coded atlas. Every variant of the
/// patch race starts from the same glyphs, so they are found and coded once.
#[derive(Clone)]
pub(super) struct GlyphStage {
    atlas: BitWriter,
    base: Image3F,
    references: Vec<crate::patches::PatchReference>,
    luma: Option<Arc<GlyphAtlasLuma>>,
}

impl GlyphStage {
    pub(super) fn find(
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        distance: f32,
        xyb: &Image3F,
        alpha: Option<&AlphaPlane>,
        free_singles: bool,
        retain_epf_luma: bool,
    ) -> Option<Self> {
        let coarse = if distance < 1.5 {
            1
        } else if distance < 2.0 {
            2
        } else if distance < 6.0 {
            4
        } else {
            8
        };
        let mut plan = crate::patches::plan_lossy_glyph_patches(
            ctx,
            xyb,
            &crate::patches::GlyphParams {
                templates: ctx.glyph_templates
                    && matches!(alpha, Some(crate::encode_image::AlphaPlane::U8(_))),
                min_occurrences: 1,
                min_repeat_pixels: if free_singles {
                    0
                } else {
                    GLYPH_MIN_REPEAT_PIXELS
                },
                coarse,
                min_contrast: GLYPH_MIN_CONTRAST_STEPS * coarse,
                singles_min_cover: if free_singles {
                    0.0
                } else {
                    GLYPH_SINGLETON_MIN_COVER
                },
            },
        )?;
        let mut atlas = BitWriter::new();
        // Template coverage rides in the atlas alpha; kAdd entries leave it zero.
        let coverage = match alpha {
            Some(AlphaPlane::U8(_)) => plan.alpha_atlas.take().map(AlphaPlane::U8),
            _ => None,
        };
        let zero_alpha;
        let atlas_alpha = if let Some(coverage) = &coverage {
            Some(coverage)
        } else {
            zero_alpha = zero_alpha_for_lossy(alpha, plan.atlas_width * plan.atlas_height);
            zero_alpha.as_ref()
        };
        crate::lossless::encode_modular_xyb_atlas_tree(
            &plan.atlas,
            plan.atlas_width,
            plan.atlas_height,
            atlas_alpha,
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            &mut atlas,
        );
        let plan = plan.materialize(xyb);
        let luma = retain_epf_luma.then(|| {
            let [samples, _, _] = plan.atlas;
            Arc::new(GlyphAtlasLuma {
                width: plan.atlas_width,
                samples,
                coverage: coverage.map(|plane| {
                    let AlphaPlane::U8(samples) = plane else {
                        unreachable!("glyph coverage is 8-bit")
                    };
                    samples
                }),
            })
        });
        Some(Self {
            atlas,
            base: plan.base,
            references: plan.references,
            luma,
        })
    }

    pub(super) fn atlas_bits(&self) -> usize {
        self.atlas.bits_written()
    }

    /// The coded atlas frame, the image less the patches, and the dictionary.
    pub(super) fn into_parts(self) -> (BitWriter, Image3F, Vec<crate::patches::PatchReference>) {
        (self.atlas, self.base, self.references)
    }
}

/// One patched or regular VarDCT frame, with or without Replace `tiles`.
/// Atlas frames are small and encoded immediately; the full residual is deferred.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_vardct_variant(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    distp: &DistanceParams,
    #[allow(unused_mut)] mut xyb: Cow<'_, Image3F>,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
    patches: bool,
    glyphs: Option<Cow<'_, GlyphStage>>,
    tiles: bool,
    #[cfg(feature = "splines")] stage: Option<&SplineStage>,
) -> Result<PreparedVarDct, EncodeError> {
    #[allow(unused_mut)]
    let mut tiles_touched = false;
    #[allow(unused_mut)]
    let mut used_tiles = false;
    #[cfg(feature = "splines")]
    let before_tiles = stage.is_some_and(|s| s.before_tiles);
    #[cfg(feature = "splines")]
    let select_splines =
        |scratch: &mut CoderScratch, image: &mut Image3F, forbidden: Option<&[bool]>| {
            let stage = stage?;
            crate::splines::select_splines(
                ctx,
                scratch,
                distance,
                image,
                stage.quant_field,
                stage.candidates,
                forbidden,
            )
        };

    // Near-duplicate tiles beyond two 8-bit levels of RMS difference match
    // noise on smooth ground; the frame without them then wins the race and
    // the plan only adds atlas and sampling work.
    let near_tiles = if ctx.speed.effort().near_patch_tiles {
        distance.clamp(0.5, NEAR_TILES_MAX)
    } else {
        0.0
    };
    if let Some(glyphs) = glyphs {
        let (mut patched_writer, mut base, glyph_refs, luma) = match glyphs {
            Cow::Borrowed(glyphs) => (
                glyphs.atlas.clone(),
                Cow::Borrowed(&glyphs.base),
                Cow::Borrowed(glyphs.references.as_slice()),
                glyphs.luma.clone(),
            ),
            Cow::Owned(glyphs) => (
                glyphs.atlas,
                Cow::Owned(glyphs.base),
                Cow::Owned(glyphs.references),
                glyphs.luma,
            ),
        };

        // Repeated tiles of what is left (icons, rules, UI chrome). They are
        // Replace entries and must precede the additive glyph entries, which
        // then land on top of the replaced rectangles.
        #[cfg(feature = "splines")]
        let early_splines = if before_tiles {
            select_splines(scratch, base.to_mut(), None)
        } else {
            None
        };
        let mut references: Vec<crate::patches::PatchReference> = Vec::new();
        let mut epf_atlases = luma.map(|glyphs| EpfPatchAtlases {
            glyphs,
            tiles: None,
        });
        if tiles {
            let tiles16 = crate::patches::find_lossy_patches_sized(
                &base,
                crate::patches::PATCH_TILE,
                near_tiles,
                &ctx.thread_pool,
                scratch,
            );
            let after16 = tiles16.as_ref().map_or(base.as_ref(), |t| &t.base);
            let tiles8 = crate::patches::find_lossy_patch_groups_sized(
                after16,
                8,
                near_tiles,
                &ctx.thread_pool,
                scratch,
            );
            let covered = tiles16
                .as_ref()
                .map_or(0, |t| t.groups.iter().map(|g| g.len()).sum::<usize>() * 256)
                + tiles8.as_ref().map_or(0, |groups| {
                    groups.iter().map(|g| g.len()).sum::<usize>() * 64
                });
            if covered as f64 >= PATCH_TILE_MIN_COVER * (base.xsize() * base.ysize()) as f64 {
                let mut tile_refs = Vec::new();
                let mut tile_atlas: Option<Image3F> = None;
                let mut next_base: Option<Image3F> = None;
                if let Some(t) = tiles16 {
                    let (atlas, refs) =
                        crate::patches::pack_lossy_atlas(&base, t.groups, PATCH_REF_ID);
                    tile_atlas = Some(atlas);
                    tile_refs = refs;
                    next_base = Some(t.base);
                }
                if let Some(groups8) = tiles8 {
                    let residual = next_base.get_or_insert_with(|| base.as_ref().clone());
                    crate::patches::clear_patch_tiles(residual, 8, &groups8);
                    let (reused, groups8) =
                        crate::patches::reuse_tile_quadrants(&base, &tile_refs, groups8);
                    tile_refs.extend(reused);
                    if !groups8.is_empty() {
                        let (atlas8, mut refs8) =
                            crate::patches::pack_lossy_atlas_sized(&base, groups8, 8, PATCH_REF_ID);
                        tile_atlas = Some(match tile_atlas {
                            Some(top) => crate::patches::stack_atlases(top, atlas8, &mut refs8),
                            None => atlas8,
                        });
                        tile_refs.extend(refs8);
                    }
                }
                let tile_atlas = tile_atlas.expect("covered > 0");
                let atlas_distance = distance * ATLAS_DISTANCE_SCALE;
                let atlas_distp = compute_distance_params(atlas_distance);
                let tile_alpha = zero_alpha_for_lossy(
                    alpha,
                    tile_atlas.xsize().saturating_mul(tile_atlas.ysize()),
                );
                let (atlas_w, atlas_h) = (tile_atlas.xsize(), tile_atlas.ysize());
                let mut tile_atlas = tile_atlas;
                if let Some(atlases) = epf_atlases.as_mut() {
                    atlases.tiles = Some(tile_atlas.plane(1).clone());
                }
                let lattice = lattice_atlas(ctx, &tile_atlas, alpha.is_some(), scratch);
                gaborize(&mut tile_atlas, &atlas_distp);
                let mut vardct = BitWriter::new();
                encode_frame_core(
                    ctx,
                    scratch,
                    atlas_distance,
                    tile_atlas,
                    tile_alpha.as_ref(),
                    &[0],
                    VarDctFrameKind::ReferenceOnly {
                        width: atlas_w,
                        height: atlas_h,
                    },
                    #[cfg(feature = "splines")]
                    None,
                    #[cfg(feature = "splines")]
                    false,
                    crate::noise::GrainPass::Inherit,
                    &mut vardct,
                )?;
                patched_writer.append(&smaller_atlas(vardct, lattice));
                references.extend(tile_refs);
                base = Cow::Owned(next_base.expect("covered > 0"));
                used_tiles = true;
            }
        }
        let atlas_bits = patched_writer.bits_written();
        match glyph_refs {
            Cow::Owned(glyph_refs) if references.is_empty() => references = glyph_refs,
            Cow::Owned(glyph_refs) => references.extend(glyph_refs),
            Cow::Borrowed(glyph_refs) => references.extend_from_slice(glyph_refs),
        }
        let mut base = base.into_owned();

        // Additive glyph entries commute with splines; Replace tiles cut from
        // the unsubtracted image must stay clear of them.
        #[cfg(feature = "splines")]
        let base_splines = if before_tiles {
            early_splines
        } else if let Some(stage) = stage {
            let blocks_w = base.xsize().div_ceil(K_BLOCK_DIM);
            let mut in_patch = vec![false; blocks_w * base.ysize().div_ceil(K_BLOCK_DIM)];
            for reference in references.iter().filter(|r| !r.add) {
                for &(px, py) in &reference.positions {
                    for by in py / K_BLOCK_DIM..(py + reference.height).div_ceil(K_BLOCK_DIM) {
                        for bx in px / K_BLOCK_DIM..(px + reference.width).div_ceil(K_BLOCK_DIM) {
                            in_patch[by * blocks_w + bx] = true;
                        }
                    }
                }
            }
            tiles_touched = stage.candidates.touches(&in_patch, blocks_w);
            select_splines(scratch, &mut base, Some(&in_patch))
        } else {
            None
        };
        gaborize(&mut base, distp);
        let mut prepared = PreparedVarDct {
            prefix: patched_writer,
            base,
            references,
            epf_atlases,
            #[cfg(feature = "splines")]
            splines: base_splines,
            tiles_touched,
            used_tiles,
            estimated_bits: None,
        };
        if distance >= GLYPH_CHECK_MIN_DISTANCE {
            let bits = prepared.estimate(ctx, scratch, distance, alpha, coeff_shifts)?;
            if atlas_bits as f64 > bits * GLYPH_CHECK_ATLAS_SHARE {
                let mut regular = xyb.into_owned();
                #[cfg(feature = "splines")]
                let regular_splines = select_splines(scratch, &mut regular, None);
                gaborize(&mut regular, distp);
                let mut regular = PreparedVarDct {
                    prefix: BitWriter::new(),
                    base: regular,
                    references: Vec::new(),
                    epf_atlases: None,
                    #[cfg(feature = "splines")]
                    splines: regular_splines,
                    tiles_touched,
                    used_tiles,
                    estimated_bits: None,
                };
                if bits
                    >= regular.estimate(ctx, scratch, distance, alpha, coeff_shifts)?
                        * GLYPH_BITS_MARGIN
                {
                    return Ok(regular);
                }
            }
        }
        return Ok(prepared);
    }

    #[cfg(feature = "splines")]
    let early_splines = if before_tiles {
        select_splines(scratch, xyb.to_mut(), None)
    } else {
        None
    };
    // Coverage admits a tile plan; the frame-level rate model compares it
    // with a plan without tiles before either residual is encoded.
    let tile_plan = if patches && tiles {
        let plan16 = crate::patches::find_lossy_patches_sized(
            &xyb,
            crate::patches::PATCH_TILE,
            near_tiles,
            &ctx.thread_pool,
            scratch,
        );
        let after16 = plan16.as_ref().map_or(xyb.as_ref(), |p| &p.base);
        let tiles8 = crate::patches::find_lossy_patch_groups_sized(
            after16,
            8,
            near_tiles,
            &ctx.thread_pool,
            scratch,
        );
        let covered = plan16
            .as_ref()
            .map_or(0, |t| t.groups.iter().map(|g| g.len()).sum::<usize>() * 256)
            + tiles8.as_ref().map_or(0, |groups| {
                groups.iter().map(|g| g.len()).sum::<usize>() * 64
            });
        (covered as f64 >= PATCH_TILE_MIN_COVER * (xyb.xsize() * xyb.ysize()) as f64).then(|| {
            let plan = plan16.unwrap_or_else(|| crate::patches::LossyPatches {
                base: xyb.as_ref().clone(),
                groups: Vec::new(),
            });
            (plan, tiles8)
        })
    } else {
        None
    };
    if let Some((plan, tiles8)) = tile_plan {
        used_tiles = true;
        // Route each patch group to the atlas that codes it best: groups whose
        // quantized tiles fit the running 256-color palette budget go to the
        // modular atlas (measured strictly dominant when the palette hits),
        // the rest to a VarDCT atlas at a finer distance. Groups arrive most
        // frequent first, so high-value groups claim the palette budget first.
        // Either atlas is emitted only when it has content, and every
        // dictionary entry names its reference frame.
        let mut palette: std::collections::HashSet<[i32; 3]> = std::collections::HashSet::new();
        let groups = plan.groups;
        let (mut modular_idx, mut vardct_idx): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        {
            let tile_colors = scratch.patch_tile_colors.as_mut();
            for (i, group) in groups.iter().enumerate() {
                let (sx, sy) = group[0];
                crate::xyb::quantize_xyb_tile_colors(
                    &xyb,
                    sx,
                    sy,
                    MODULAR_ATLAS_LATTICE_SCALE,
                    tile_colors,
                );
                tile_colors.sort_unstable();
                let unique_colors = || {
                    tile_colors.iter().enumerate().filter_map(|(i, &color)| {
                        (i == 0 || tile_colors[i - 1] != color).then_some(color)
                    })
                };
                let new_colors = unique_colors().filter(|c| !palette.contains(c)).count();
                if palette.len() + new_colors <= 256 {
                    palette.extend(unique_colors());
                    modular_idx.push(i);
                } else {
                    vardct_idx.push(i);
                }
            }
        }
        let clone_groups = |idx: &[usize]| -> Vec<Vec<(usize, usize)>> {
            idx.iter().map(|&i| groups[i].clone()).collect()
        };

        let atlas_distance = distance * ATLAS_DISTANCE_SCALE;
        let atlas_distp = compute_distance_params(atlas_distance);
        let encode_vardct_atlas =
            |atlas: Image3F, scratch: &mut CoderScratch, out: &mut BitWriter| {
                let atlas_alpha =
                    zero_alpha_for_lossy(alpha, atlas.xsize().saturating_mul(atlas.ysize()));
                let (atlas_w, atlas_h) = (atlas.xsize(), atlas.ysize());
                let mut atlas = atlas;
                gaborize(&mut atlas, &atlas_distp);
                encode_frame_core(
                    ctx,
                    scratch,
                    atlas_distance,
                    atlas,
                    atlas_alpha.as_ref(),
                    &[0],
                    VarDctFrameKind::ReferenceOnly {
                        width: atlas_w,
                        height: atlas_h,
                    },
                    #[cfg(feature = "splines")]
                    None,
                    #[cfg(feature = "splines")]
                    false,
                    crate::noise::GrainPass::Inherit,
                    out,
                )
            };

        let mut patched_writer = BitWriter::new();
        let mut references: Vec<crate::patches::PatchReference> = Vec::new();

        // Palette viability routed the modular set, but viable is not the same
        // as compressible: a 256-pixel noise tile fits the palette budget and
        // still codes terribly (256 incompressible indices). Price the subset
        // both ways and keep the routing only when the modular frame is
        // actually smaller; otherwise everything folds back into one VarDCT
        // atlas in the original order.
        if !modular_idx.is_empty() {
            let (modular_atlas, modular_refs) = crate::patches::pack_lossy_atlas(
                &xyb,
                clone_groups(&modular_idx),
                MODULAR_PATCH_REF_ID,
            );
            let mut modular_bits = BitWriter::new();
            let modular_ok = crate::lossless::encode_modular_xyb_atlas(
                &modular_atlas,
                alpha.is_some(),
                MODULAR_ATLAS_LATTICE_SCALE,
                ctx.speed,
                true,
                scratch,
                &mut modular_bits,
            );
            let mut vardct_bits = BitWriter::new();
            encode_vardct_atlas(modular_atlas, scratch, &mut vardct_bits)?;
            if modular_ok && modular_bits.bits_written() < vardct_bits.bits_written() {
                patched_writer.append(&modular_bits);
                references.extend(modular_refs);
            } else {
                vardct_idx.append(&mut modular_idx);
                vardct_idx.sort_unstable();
            }
        }

        let mut base = plan.base;
        let mut vardct_atlas: Option<Image3F> = None;
        if !vardct_idx.is_empty() {
            let (atlas, vardct_refs) =
                crate::patches::pack_lossy_atlas(&xyb, clone_groups(&vardct_idx), PATCH_REF_ID);
            vardct_atlas = Some(atlas);
            references.extend(vardct_refs);
        }
        if let Some(groups8) = tiles8 {
            crate::patches::clear_patch_tiles(&mut base, 8, &groups8);
            let (reused, groups8) =
                crate::patches::reuse_tile_quadrants(&xyb, &references, groups8);
            references.extend(reused);
            if !groups8.is_empty() {
                let (atlas8, mut refs8) =
                    crate::patches::pack_lossy_atlas_sized(&xyb, groups8, 8, PATCH_REF_ID);
                vardct_atlas = Some(match vardct_atlas {
                    Some(top) => crate::patches::stack_atlases(top, atlas8, &mut refs8),
                    None => atlas8,
                });
                references.extend(refs8);
            }
        }
        if let Some(atlas) = vardct_atlas {
            let lattice = lattice_atlas(ctx, &atlas, alpha.is_some(), scratch);
            let mut vardct = BitWriter::new();
            encode_vardct_atlas(atlas, scratch, &mut vardct)?;
            patched_writer.append(&smaller_atlas(vardct, lattice));
        }

        // Patches replace their rectangles and the decoder draws splines after
        // them, so no spline may reach into a patch cut from the unsubtracted
        // image.
        #[cfg(feature = "splines")]
        let base_splines = if before_tiles {
            early_splines
        } else if let Some(stage) = stage {
            let blocks_w = base.xsize().div_ceil(K_BLOCK_DIM);
            let mut in_patch = vec![false; blocks_w * base.ysize().div_ceil(K_BLOCK_DIM)];
            for reference in &references {
                for &(px, py) in &reference.positions {
                    for by in py / K_BLOCK_DIM..(py + reference.height).div_ceil(K_BLOCK_DIM) {
                        for bx in px / K_BLOCK_DIM..(px + reference.width).div_ceil(K_BLOCK_DIM) {
                            in_patch[by * blocks_w + bx] = true;
                        }
                    }
                }
            }
            tiles_touched = stage.candidates.touches(&in_patch, blocks_w);
            select_splines(scratch, &mut base, Some(&in_patch))
        } else {
            None
        };
        gaborize(&mut base, distp);
        return Ok(PreparedVarDct {
            prefix: patched_writer,
            base,
            references,
            epf_atlases: None,
            #[cfg(feature = "splines")]
            splines: base_splines,
            tiles_touched,
            used_tiles,
            estimated_bits: None,
        });
    }

    let mut xyb = xyb.into_owned();
    #[cfg(feature = "splines")]
    let splines = if before_tiles {
        early_splines
    } else {
        select_splines(scratch, &mut xyb, None)
    };
    gaborize(&mut xyb, distp);
    Ok(PreparedVarDct {
        prefix: BitWriter::new(),
        base: xyb,
        references: Vec::new(),
        epf_atlases: None,
        #[cfg(feature = "splines")]
        splines,
        tiles_touched,
        used_tiles,
        estimated_bits: None,
    })
}

pub(super) struct PreparedVarDct {
    prefix: BitWriter,
    /// Selected splines/patches removed and inverse Gaborish already applied.
    base: Image3F,
    references: Vec<crate::patches::PatchReference>,
    epf_atlases: Option<EpfPatchAtlases>,
    #[cfg(feature = "splines")]
    splines: Option<crate::splines::SplineSet>,
    #[cfg_attr(not(feature = "splines"), allow(dead_code))]
    pub(super) tiles_touched: bool,
    pub(super) used_tiles: bool,
    estimated_bits: Option<f64>,
}

impl PreparedVarDct {
    pub(super) fn regular(base: Image3F) -> Self {
        Self {
            prefix: BitWriter::new(),
            base,
            references: Vec::new(),
            epf_atlases: None,
            #[cfg(feature = "splines")]
            splines: None,
            tiles_touched: false,
            used_tiles: false,
            estimated_bits: None,
        }
    }

    /// Price a regular frame whose source has no spline or Gaborish changes.
    pub(super) fn estimate_regular(
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        distance: f32,
        source: &Image3F,
        alpha: Option<&AlphaPlane>,
        coeff_shifts: &[u32],
    ) -> Result<f64, EncodeError> {
        Ok(noise_estimate_bits(ctx)
            + estimate_residual_bits(ctx, scratch, distance, source, alpha, coeff_shifts)?)
    }

    #[cfg(feature = "splines")]
    pub(super) fn has_splines(&self) -> bool {
        self.splines.is_some()
    }

    /// Moves the selected dots out of the spline set into a template atlas
    /// (a modular reference frame) and kAdd dictionary entries.
    #[cfg(feature = "splines")]
    fn materialize_dots(
        &mut self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        alpha: Option<&AlphaPlane>,
    ) {
        use crate::splines::dots;
        let Some(set) = self.splines.as_mut() else {
            return;
        };
        let encoding = set.dot_encoding.take();
        let selected: Vec<_> = set.splines.extract_if(.., |sp| dots::is_dot(sp)).collect();
        if set.splines.is_empty() {
            self.splines = None;
        }
        if selected.is_empty() {
            return;
        }
        if alpha.is_none()
            && let Some(encoding) = encoding
        {
            self.prefix.append_bits(&encoding.atlas);
            self.references.extend(encoding.references);
            return;
        }
        let plan = dots::build_dot_patches(&selected);
        let atlas_alpha = zero_alpha_for_lossy(alpha, plan.width * plan.height);
        crate::lossless::encode_modular_xyb_atlas_tree_slot(
            &plan.atlas,
            plan.width,
            plan.height,
            atlas_alpha.as_ref(),
            ctx.speed,
            &ctx.thread_pool,
            scratch,
            crate::patches::DOT_PATCH_REF_ID,
            &mut self.prefix,
        );
        self.references.extend(plan.references());
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "splines"), allow(unused_mut))]
    pub(super) fn encode(
        mut self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        distance: f32,
        alpha: Option<&AlphaPlane>,
        coeff_shifts: &[u32],
        writer: &mut BitWriter,
    ) -> Result<(), EncodeError> {
        #[cfg(feature = "splines")]
        self.materialize_dots(ctx, scratch, alpha);
        let epf_atlases = self.epf_atlases.filter(|atlases| {
            self.references.iter().all(|r| {
                r.alpha_only
                    || if r.mul {
                        // Blend-above glyph templates: color and coverage in the glyph atlas.
                        !r.add
                            && r.ref_frame == MODULAR_PATCH_REF_ID
                            && atlases.glyphs.coverage.is_some()
                    } else if r.add {
                        r.ref_frame == MODULAR_PATCH_REF_ID
                    } else {
                        r.ref_frame == PATCH_REF_ID && atlases.tiles.is_some()
                    }
            })
        });
        writer.append_bits(&self.prefix);
        drop(self.prefix);
        encode_frame_core(
            ctx,
            scratch,
            distance,
            self.base,
            alpha,
            coeff_shifts,
            if self.references.is_empty() {
                VarDctFrameKind::Regular
            } else {
                VarDctFrameKind::Patched(&self.references, epf_atlases.as_ref())
            },
            #[cfg(feature = "splines")]
            self.splines.as_ref(),
            #[cfg(feature = "splines")]
            true,
            crate::noise::GrainPass::Inherit,
            writer,
        )?;
        Ok(())
    }

    pub(super) fn estimate(
        &mut self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        distance: f32,
        alpha: Option<&AlphaPlane>,
        coeff_shifts: &[u32],
    ) -> Result<f64, EncodeError> {
        if let Some(bits) = self.estimated_bits {
            return Ok(bits);
        }
        #[cfg(feature = "splines")]
        self.materialize_dots(ctx, scratch, alpha);
        let mut extras = BitWriter::new();
        if !self.references.is_empty() {
            crate::lossless::write_patch_dictionary(
                &self.references,
                alpha.is_some(),
                scratch,
                &mut extras,
            );
        }
        #[cfg(feature = "splines")]
        if let Some(set) = &self.splines {
            crate::splines::write_splines(set, scratch, &mut extras);
        }
        let bits = self.prefix.bits_written() as f64
            + extras.bits_written() as f64
            + noise_estimate_bits(ctx)
            + estimate_residual_bits(ctx, scratch, distance, &self.base, alpha, coeff_shifts)?;
        self.estimated_bits = Some(bits);
        Ok(bits)
    }
}

fn noise_estimate_bits(ctx: &EncodingContext) -> f64 {
    if ctx.grain.enabled() {
        crate::noise::NoiseParams::BITS as f64
    } else {
        0.0
    }
}

/// Encode a bounded, spatially stratified sample with the real transform,
/// quantization and entropy choices. Scale only its data sections; global
/// headers and the actual patch/spline syntax are charged once.
/// Sampled statistics and boundaries can change RDO decisions, so this is a
/// rate estimate, not an exact predictor of the full bitstream.
fn estimate_residual_bits(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    source: &Image3F,
    alpha: Option<&AlphaPlane>,
    coeff_shifts: &[u32],
) -> Result<f64, EncodeError> {
    let (w, h) = (source.xsize(), source.ysize());
    let (sample, sample_alpha) = sample_vardct_residual(source, alpha);
    let (sw, sh) = (sample.xsize(), sample.ysize());
    let mut writer = BitWriter::new();
    let payload = encode_frame_core(
        ctx,
        scratch,
        distance,
        sample,
        sample_alpha.as_ref(),
        coeff_shifts,
        VarDctFrameKind::Regular,
        #[cfg(feature = "splines")]
        None,
        #[cfg(feature = "splines")]
        false,
        crate::noise::GrainPass::Off,
        &mut writer,
    )?;
    Ok((writer.bits_written() - payload) as f64
        + payload as f64 * (w * h) as f64 / (sw * sh) as f64)
}

/// Keep sample windows on the original pixel lattice and copy alpha through
/// the same positions. At most sixteen 128x128 windows enter the rate model.
fn sample_vardct_residual(
    source: &Image3F,
    alpha: Option<&AlphaPlane>,
) -> (Image3F, Option<AlphaPlane>) {
    let (w, h) = (source.xsize(), source.ysize());
    let side = (w.min(h) / 2).clamp(8, 128).next_power_of_two().min(128);
    let (tw, th) = (side.min(w), side.min(h));
    let (nx, ny) = (w.div_ceil(tw), h.div_ceil(th));
    let cells = nx * ny;
    let count = (cells / 4).clamp(1, 16);
    let columns = count.min(4);
    let rows = count.div_ceil(columns);
    let count = columns * rows;
    let (sw, sh) = (tw * columns, th * rows);
    let mut sample = Image3F::new(sw, sh);
    let mut sample_alpha = alpha.map(|alpha| match alpha {
        AlphaPlane::U8(_) => AlphaPlane::U8(vec![0; sw * sh]),
        AlphaPlane::U16 { bits, .. } => AlphaPlane::U16 {
            data: vec![0; sw * sh],
            bits: *bits,
        },
        AlphaPlane::F32(_) => AlphaPlane::F32(vec![0; sw * sh]),
    });
    for i in 0..count {
        let cell = ((2 * i + 1) * cells / (2 * count)).min(cells - 1);
        let sx = ((cell % nx) * tw).min(w - tw);
        let sy = ((cell / nx) * th).min(h - th);
        let (dx, dy) = ((i % columns) * tw, (i / columns) * th);
        for y in 0..th {
            for c in 0..3 {
                sample.plane_row_mut(c, dy + y)[dx..dx + tw]
                    .copy_from_slice(&source.plane_row(c, sy + y)[sx..sx + tw]);
            }
            if let Some((input, output)) = alpha.zip(sample_alpha.as_mut()) {
                let src = (sy + y) * w + sx;
                let dst = (dy + y) * sw + dx;
                match (input, output) {
                    (AlphaPlane::U8(input), AlphaPlane::U8(output)) => {
                        output[dst..dst + tw].copy_from_slice(&input[src..src + tw]);
                    }
                    (AlphaPlane::U16 { data: input, .. }, AlphaPlane::U16 { data: output, .. }) => {
                        output[dst..dst + tw].copy_from_slice(&input[src..src + tw]);
                    }
                    (AlphaPlane::F32(input), AlphaPlane::F32(output)) => {
                        output[dst..dst + tw].copy_from_slice(&input[src..src + tw]);
                    }
                    _ => unreachable!("sample alpha uses the source format"),
                }
            }
        }
    }
    (sample, sample_alpha)
}

/// Effective AC quant (`scale * q`) per 8x8 block, measured on the image before
/// any spline is removed; the spline RD gate prices blocks with it.
#[cfg(feature = "splines")]
pub(super) fn spline_quant_field(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
    distp: &DistanceParams,
) -> Result<Vec<f32>, EncodeError> {
    let blocks_w = xyb.xsize().div_ceil(K_BLOCK_DIM);
    let blocks_h = xyb.ysize().div_ceil(K_BLOCK_DIM);
    let mut field = vec![0f32; blocks_w * blocks_h];
    for y0 in (0..xyb.ysize()).step_by(K_DC_GROUP_DIM) {
        for x0 in (0..xyb.xsize()).step_by(K_DC_GROUP_DIM) {
            let gw = K_DC_GROUP_DIM.min(xyb.xsize() - x0).div_ceil(K_BLOCK_DIM);
            let gh = K_DC_GROUP_DIM.min(xyb.ysize() - y0).div_ceil(K_BLOCK_DIM);
            let mut raw = crate::image::ImageB::try_new_fill(gw, gh, 1)?;
            (ctx.fill_quant_field)(
                &mut scratch.aq_map,
                xyb,
                &mut raw,
                x0,
                y0,
                distp.distance,
                1.0 / distp.scale,
            );
            for y in 0..gh {
                let row = &mut field[(y0 / K_BLOCK_DIM + y) * blocks_w + x0 / K_BLOCK_DIM..];
                for (dst, &q) in row.iter_mut().zip(&raw.row(y)[..gw]) {
                    *dst = distp.scale * q as f32;
                }
            }
        }
    }
    Ok(field)
}

/// Unrepeated glyphs join the atlas freely up to this distance; above it
/// only when together they cover this share of the frame, or when the plan
/// with them wins the bit race up to the race distance.
pub(super) const GLYPH_SINGLETON_FREE_MAX_DISTANCE: f32 = 1.0;
pub(super) const GLYPH_SINGLETON_RACE_MAX_DISTANCE: f32 = 2.0;
const GLYPH_SINGLETON_MIN_COVER: f64 = 0.5;
/// A glyph must differ from its background by this many lattice rounding
/// steps somewhere; fainter shapes are shading the base codes for less.
const GLYPH_MIN_CONTRAST_STEPS: i32 = 2;
/// Box pixels a repeated glyph must save beyond its first occurrence.
const GLYPH_MIN_REPEAT_PIXELS: usize = 48;
/// A glyph plan is checked against the regular frame only from this distance
/// and when the atlas frames take more than this share of the patched bits.
const GLYPH_CHECK_MIN_DISTANCE: f32 = 6.0;
const GLYPH_CHECK_ATLAS_SHARE: f64 = 0.35;
/// Patched bits may exceed the regular frame's by this factor.
pub(super) const GLYPH_BITS_MARGIN: f64 = 1.10;
/// A lattice arm must stay under this share of the best tile/glyph plan's bits.
pub(super) const LATTICE_BITS_MARGIN: f64 = 0.5;
/// Tile patches are used only when they cover this share of the frame.
const PATCH_TILE_MIN_COVER: f64 = 0.03;
const NEAR_TILES_MAX: f32 = 2.0;

/// Power-of-two refinement of the modular atlas quantization lattice.
const MODULAR_ATLAS_LATTICE_SCALE: u32 = 8;

/// The VarDCT atlas is coded this much finer than the frame it serves.
const ATLAS_DISTANCE_SCALE: f32 = 0.45;

/// The VarDCT atlas's content as a modular frame on the finest XYB lattice,
/// in the same reference slot: sharper than the VarDCT atlas, so the smaller
/// of the two can be kept by bits alone. None when the atlas exceeds one
/// 1024-pixel group. Small graphics atlases (a few 8x8 tiles of flat UI
/// color) cost the VarDCT frame 2-4x the modular one; photographic tiles
/// keep VarDCT.
fn lattice_atlas(
    ctx: &EncodingContext,
    atlas: &Image3F,
    has_alpha: bool,
    scratch: &mut CoderScratch,
) -> Option<BitWriter> {
    let mut out = BitWriter::new();
    crate::lossless::encode_modular_xyb_atlas_ints(
        crate::xyb::quantize_xyb_channels(atlas, MODULAR_ATLAS_LATTICE_SCALE),
        atlas.xsize(),
        atlas.ysize(),
        has_alpha,
        MODULAR_ATLAS_LATTICE_SCALE,
        PATCH_REF_ID,
        ctx.speed,
        true,
        scratch,
        &mut out,
    )
    .then_some(out)
}

fn smaller_atlas(vardct: BitWriter, lattice: Option<BitWriter>) -> BitWriter {
    match lattice {
        Some(lattice) if lattice.bits_written() < vardct.bits_written() => lattice,
        _ => vardct,
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn residual_rate_sample_preserves_pixels_and_alpha_at_odd_edges() {
        use crate::encode_image::AlphaPlane;
        use crate::image::Image3F;
        for (w, h) in [(1, 1), (7, 19), (257, 385), (1025, 769)] {
            let mut source = Image3F::new(w, h);
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        source.plane_row_mut(c, y)[x] = (y * w + x) as f32 + c as f32 * 0.25;
                    }
                }
            }
            let alpha = AlphaPlane::U16 {
                data: (0..w * h).map(|i| (i % 4096) as u16).collect(),
                bits: 12,
            };
            let (sample, sampled_alpha) = super::sample_vardct_residual(&source, Some(&alpha));
            let Some(AlphaPlane::U16 { data, bits }) = sampled_alpha else {
                panic!("sample changed the alpha representation");
            };
            assert_eq!(bits, 12);
            assert!(sample.xsize() * sample.ysize() <= 16 * 128 * 128);
            assert_eq!(data.len(), sample.xsize() * sample.ysize());
            for y in 0..sample.ysize() {
                for x in 0..sample.xsize() {
                    let index = sample.plane_row(0, y)[x] as usize;
                    assert!(index < w * h);
                    assert_eq!(data[y * sample.xsize() + x], (index % 4096) as u16);
                    for c in 0..3 {
                        assert_eq!(
                            sample.plane_row(c, y)[x],
                            source.plane_row(c, index / w)[index % w]
                        );
                    }
                }
            }
        }
    }
}
