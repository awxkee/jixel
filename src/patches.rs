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
//! Patch dictionaries: repeated content stored once in a reference frame.
//!
//! - [`tiles`]: repeated square tiles of the lossy path.
//! - [`lossless`], [`lossless_glyphs`]: repeated tiles and glyphs of the
//!   lossless path.
//! - [`glyphs`]: text and UI glyphs of the lossy path.
use crate::image::{Image3F, Image3Si};

mod color_key;
mod glyphs;
mod lossless;
mod lossless_glyphs;
mod tiles;

pub(crate) use color_key::ContentHashMap;
pub(crate) use glyphs::{GlyphParams, plan_lossy_glyph_patches};
pub(crate) use lossless::find_lossless_patches;
pub(crate) use lossless_glyphs::find_lossless_glyph_patches;
pub(crate) use tiles::{
    clear_patch_tiles, find_lossy_patch_groups_sized, find_lossy_patches_sized, pack_lossy_atlas,
    pack_lossy_atlas_sized, reuse_tile_quadrants, stack_atlases,
};

pub(crate) struct LossyPatches {
    pub(crate) base: Image3F,
    /// Occurrence positions per group, first entry = the source tile. Packing
    /// into atlases is the caller's job so groups can be routed to more than
    /// one reference frame.
    pub(crate) groups: Vec<Vec<(usize, usize)>>,
}

#[derive(Clone, Copy)]
pub(crate) enum VarDctFrameKind<'a> {
    Regular,
    ReferenceOnly {
        width: usize,
        height: usize,
    },
    Patched(
        &'a [PatchReference],
        Option<&'a EpfPatchAtlases>,
        Option<&'a crate::bit_writer::BitWriter>,
    ),
}

pub(crate) struct GlyphAtlasLuma {
    pub(crate) width: usize,
    pub(crate) samples: Vec<i32>,
    /// 8-bit coverage of blend-above template tiles (atlas alpha).
    pub(crate) coverage: Option<Vec<u8>>,
}

pub(crate) struct EpfPatchAtlases {
    pub(crate) glyphs: std::sync::Arc<GlyphAtlasLuma>,
    /// The source luma of the finer Replace atlas, used for both sides of the
    /// comparison because its reconstruction is independent of the base EPF.
    pub(crate) tiles: Option<crate::image::Plane<f32>>,
}

pub(crate) const PATCH_TILE: usize = 16;

pub(crate) const PATCH_REF_ID: u32 = 3;
/// Reference slot for the modular atlas; the VarDCT atlas keeps slot 3, so a
/// hybrid plan can emit both and route each dictionary entry to either.
pub(crate) const MODULAR_PATCH_REF_ID: u32 = 2;
/// Reference slot for the star-dot template atlas.
#[cfg(feature = "splines")]
pub(crate) const DOT_PATCH_REF_ID: u32 = 1;
pub(crate) const NUM_PATCH_CONTEXTS: usize = 10;

#[derive(Clone)]
pub(crate) struct PatchReference {
    pub(crate) atlas_x: usize,
    pub(crate) atlas_y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Which saved reference frame this entry copies from.
    pub(crate) ref_frame: u32,
    /// Replace alpha from the atlas while leaving color untouched.
    pub(crate) alpha_only: bool,
    /// Blend with kAdd instead of kReplace.
    pub(crate) add: bool,
    /// Blend with kMul (the entry scales what is already there); `add` is
    /// then the hint that the entry is not a Replace rectangle. `mul`
    /// without `add` is kBlendAbove: the atlas color is mixed over the frame
    /// by the atlas alpha (glyph templates; needs an alpha channel).
    pub(crate) mul: bool,
    pub(crate) positions: Vec<(usize, usize)>,
}

pub(crate) struct LosslessPatches {
    pub(crate) atlas: Image3Si,
    pub(crate) base: Image3Si,
    pub(crate) references: Vec<PatchReference>,
}

#[derive(Clone, Copy)]
pub(crate) enum ModularFrameKind<'a> {
    Regular,
    ReferenceOnly {
        width: usize,
        height: usize,
    },
    /// Reference-only atlas inside an `xyb_encoded` codestream: channels are
    /// Y, X, B-Y on the default LF dequant lattice, saved to the modular slot.
    XybReferenceOnly {
        width: usize,
        height: usize,
        slot: u32,
    },
    /// Ink frame of an `xyb_encoded` codestream, alpha-blended (kBlend)
    /// over reference slot 0: channels are Y, X, B-Y on the default LF
    /// dequant lattice plus the coverage alpha (see `EncodeConfig::ink_layer`).
    XybBlend {
        /// False when further ink frames are stacked on this one: the
        /// blended canvas is saved back to reference slot 0.
        last: bool,
    },
    /// First frame of a lossless layered codestream (see
    /// `EncodeConfig::vector_layers`): the palette base, saved to reference
    /// slot 0 and not displayed on its own.
    RgbSavedFirst,
    /// Last frame of a lossless layered codestream: the coverage frame,
    /// alpha-blended (kBlend) over reference slot 0.
    RgbBlendLast,
    /// A region frame (see `region_frames`): a crop of the image at
    /// (`x0`, `y0`) replacing that rectangle of reference slot 0, saved back
    /// to the slot unless it is the last frame.
    RgbCrop {
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        /// The rectangle covers the whole image.
        full: bool,
        last: bool,
    },
    /// A crop frame of an `xyb_encoded` codestream (see `crop_frames`): the
    /// crop at (`x0`, `y0`) on the XYB lattice of the VarDCT base, replacing
    /// that rectangle of reference slot 0, saved back to the slot unless it
    /// is the last frame.
    XybCrop {
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        /// The rectangle covers the whole image.
        full: bool,
        last: bool,
    },
    /// The last frame of a cell-layer codestream (see `cell_layer`): a crop
    /// at (`x0`, `y0`) of `width` x `height` image pixels, coded at that
    /// size over the frame's upsampling factor, alpha-blended (kBlend with
    /// extra channel 0) over reference slot 0.
    RgbBlendCrop {
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    },
    Patched {
        references: &'a [PatchReference],
        /// The frame is the base of region frames: saved to reference slot
        /// 0 and not the last frame.
        saved_first: bool,
    },
}

impl ModularFrameKind<'_> {
    pub(crate) fn is_regular(self) -> bool {
        matches!(
            self,
            Self::Regular
                | Self::RgbSavedFirst
                | Self::RgbBlendLast
                | Self::RgbCrop { .. }
                | Self::XybCrop { .. }
                | Self::RgbBlendCrop { .. }
        )
    }

    /// Every frame kind may be coded by the learned-tree path: patched
    /// frames open their first section with the dictionary, and reference-
    /// only atlas frames only differ in their header (the flat coder spent
    /// 10 bits per pixel on a glyph atlas).
    pub(crate) fn allows_learned_tree(self) -> bool {
        true
    }
}
