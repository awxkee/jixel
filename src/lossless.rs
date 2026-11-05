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

//! Lossless modular encoding and shared group layout.
//!
//! `frame` decides which frames code the image (the whole frame, a region
//! split, a patched pair); `encoder` codes one frame, step by step: palette
//! transforms, Squeeze, the color transform, the learned-tree tournament
//! (`encoder::trees`) and the flat coder (`encoder::flat`), keeping the
//! smallest complete frame. The `ma_*` modules handle learned-tree sources, sampling,
//! learning, and encoding. Palette, predictor, LZ77, and Squeeze implementations
//! live in their own module trees; `header` and `tree` serialize shared syntax.
//! `near_lossless` is the bounded-error pre-pass applied before the exact coder.

#[cfg(test)]
#[path = "lossless/tests/constant_channel_tests.rs"]
mod constant_channel_tests;
mod context;
#[cfg(test)]
#[path = "lossless/tests/context_tree_tests.rs"]
mod context_tree_tests;
mod encoder;
pub(crate) use encoder::encode_frame_lossless_core;
mod entropy;
mod float;
mod frame;
mod groups;
mod header;
mod lz77;
#[cfg(test)]
#[path = "lossless/tests/lz_reach_probe.rs"]
mod lz_reach_probe;
mod ma_encoding;
mod ma_learning;
mod ma_local;
mod ma_lookup;
mod ma_multiplier;
mod ma_sampling;
#[cfg(test)]
#[path = "lossless/tests/ma_sampling_tests.rs"]
mod ma_sampling_tests;
mod ma_source;
mod ma_streams;
mod near_lossless;
pub(crate) use near_lossless::quantize_planes;
mod palette;
mod predictor;
#[cfg(test)]
#[path = "lossless/tests/rate_selection_tests.rs"]
mod rate_selection_tests;
mod rct;
mod squeeze;
mod tokens;
mod tree;
mod xyb;

pub(crate) use rct::{rct_from_ycocg_pixel, selected_rct_row_fn};
#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
))]
pub(crate) use {palette::palette_row_scalar, rct::rct_row_scalar};

pub(crate) use context::PickThresholdScratch;
pub(crate) use entropy::{EntropyOfHistFn, selected_entropy_of_hist_fn};
pub(crate) use float::encode_frame_lossless_float;
pub(crate) use frame::{encode_frame_lossless, encode_frame_lossless_base};
pub(crate) use groups::write_single_channel_lz77_section;
pub(crate) use header::write_patch_dictionary;
pub(crate) use lz77::LzToken;
pub(crate) use ma_sampling::{
    DcPlanes, sample_dc_planes, tokenize_dc_planes, walk_dc_planes_predictors,
};
pub(crate) use ma_source::ac_stream_id;
pub(crate) use palette::write_palette_transform;
pub(crate) use predictor::{
    GradPackInteriorFn, GradientScratch, pack_gradient_row, selected_grad_pack_interior_fn,
};
pub(crate) use rct::forward_ycocg;
pub(crate) use squeeze::{GlobalPlaneTree, PlaneTreeBudget, StreamCrop, learn_global_plane_tree};

/// The modular coding primitives [`crate::lossy_modular`] builds on: group
/// geometry, MA sampling and trees, predictors, LZ77 and entropy sections,
/// group and TOC writers, and the Squeeze transform entries.
pub(crate) mod kit {
    pub(crate) use super::GroupLayout;
    pub(crate) use super::groups::{group_lz_threads, write_lz_groups};
    pub(crate) use super::header::{write_toc_entry, write_u64};
    pub(crate) use super::lz77::{
        LZ77_MIN_SYMBOL, LzToken, LzTokenSource, build_lz_pixel_code_ans, lz77_compress_for_speed,
        lz77_literals, write_lz_section, write_tree_lz77,
    };
    pub(crate) use super::ma_multiplier::{
        SAFE_PREDICTORS, emit_forest_bfs, scale_ma_value_props, tokenize_channel_ma_mult,
    };
    pub(crate) use super::ma_sampling::{MaRefPlane, sample_channel_ma};
    pub(crate) use super::ma_source::{
        MA_MIN_NODE_SAMPLES, MA_PHOTO_CANDIDATES, MA_SPLIT_COST_BITS,
    };
    pub(crate) use super::predictor::{
        PREDICTOR_GRADIENT, SqueezePredictorCost, channel_to_context, choose_predictor_for_plane,
        fixed_predictor, selected_grad_pack_interior_fn, tokenize_plane_rows,
    };
    pub(crate) use super::rct::forward_ycocg;
    pub(crate) use super::squeeze::{
        GlobalPlaneTree, for_each_squeeze_group_crop, write_modular_transforms_rct_squeeze,
        write_modular_transforms_squeeze_only, write_modular_transforms_squeeze_palette,
    };
    pub(crate) use super::tree::{build_balanced_tree_tokens_mul, push_leaf_mul, push_split};
}

#[cfg(feature = "splines")]
pub(crate) use xyb::encode_modular_xyb_atlas_tree_slot;
pub(crate) use xyb::{
    encode_frame_lossless_layer, encode_frame_lossless_xyb_ink, encode_modular_xyb_atlas,
    encode_modular_xyb_atlas_tree,
};

use crate::encode_image::AlphaPlane;
use lz77::LZ77_MIN_SYMBOL;

/// Modular group layout: the frame header's `group_size_shift`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GroupLayout {
    pub(crate) shift: u32,
    /// The frame header's `upsampling` (1, 2, 4 or 8): the frame is coded
    /// at the image size over this factor and the decoder scales it back
    /// with the kernel signaled in the image metadata.
    upsampling: u32,
}

impl GroupLayout {
    /// 256-pixel groups: the codestream default and libjxl's lossless choice.
    pub(crate) const DEFAULT: Self = Self {
        shift: 1,
        upsampling: 1,
    };
    /// 1024-pixel groups. Every group edge costs its first row and column
    /// their north/west context and restarts the WP state, so Slow lossless
    /// prefers the largest groups whenever the learner's estimate agrees.
    pub(crate) const LARGE: Self = Self {
        shift: 3,
        upsampling: 1,
    };

    pub(crate) const fn with_upsampling(self, upsampling: u32) -> Self {
        Self { upsampling, ..self }
    }

    /// The frame header's U32(1, 2, 4, 8) selector of `upsampling`.
    pub(crate) const fn upsampling_selector(self) -> u64 {
        (self.upsampling.trailing_zeros()) as u64
    }

    pub(crate) const fn dim(self) -> usize {
        128 << self.shift
    }

    pub(crate) const fn lf_dim(self) -> usize {
        self.dim() * 8
    }
}

pub(crate) const GROUP_DIM: usize = GroupLayout::DEFAULT.dim();

pub(crate) const LF_GROUP_DIM: usize = GroupLayout::DEFAULT.lf_dim();

/// The frame's modular groups under one [`GroupLayout`].
#[derive(Clone, Copy, Debug)]
pub(super) struct GroupGrid {
    xsize: usize,
    ysize: usize,
    /// Side of an AC group in pixels.
    pub(super) dim: usize,
    pub(super) x_groups: usize,
    pub(super) y_groups: usize,
    pub(super) x_dc_groups: usize,
    pub(super) y_dc_groups: usize,
}

impl GroupGrid {
    pub(super) fn new(xsize: usize, ysize: usize, layout: GroupLayout) -> Self {
        Self {
            xsize,
            ysize,
            dim: layout.dim(),
            x_groups: xsize.div_ceil(layout.dim()),
            y_groups: ysize.div_ceil(layout.dim()),
            x_dc_groups: xsize.div_ceil(layout.lf_dim()),
            y_dc_groups: ysize.div_ceil(layout.lf_dim()),
        }
    }

    pub(super) fn num_groups(&self) -> usize {
        self.x_groups * self.y_groups
    }

    pub(super) fn num_dc_groups(&self) -> usize {
        self.x_dc_groups * self.y_dc_groups
    }

    /// The whole frame is one group: one section, no global tree.
    pub(super) fn single_group(&self) -> bool {
        self.num_groups() == 1
    }

    /// Group `index`'s rectangle in raster order: `(x0, y0, width, height)`.
    pub(super) fn rect(&self, index: usize) -> (usize, usize, usize, usize) {
        let x0 = (index % self.x_groups) * self.dim;
        let y0 = (index / self.x_groups) * self.dim;
        (
            x0,
            y0,
            self.dim.min(self.xsize - x0),
            self.dim.min(self.ysize - y0),
        )
    }
}

/// How a lossless frame is coded, apart from its pixels and its kind.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LosslessSettings {
    /// Bit depth of the widest channel, color or alpha.
    pub(crate) max_bits: u32,
    /// Squeeze-coded (progressive) frame.
    pub(crate) progressive: bool,
    /// 3 for RGB, coded from YCoCg planes; 1 for gray.
    pub(crate) num_color: usize,
    pub(crate) speed: crate::Speed,
    pub(crate) decoding_speed: crate::DecodingSpeed,
    /// The frame header's `upsampling` (1, 2, 4 or 8).
    pub(crate) upsampling: u32,
    /// The planes were quantized onto a `2k+1` grid (see `near_lossless`).
    pub(crate) near_lossless: bool,
}

impl LosslessSettings {
    /// One 8-bit RGB pass at full size: the frames of a layered codestream.
    pub(crate) fn rgb8(speed: crate::Speed, decoding_speed: crate::DecodingSpeed) -> Self {
        Self {
            max_bits: 8,
            progressive: false,
            num_color: 3,
            speed,
            decoding_speed,
            upsampling: 1,
            near_lossless: false,
        }
    }

    /// The same settings without Squeeze: reference, patched and region
    /// frames are never progressive.
    pub(crate) fn single_pass(self) -> Self {
        Self {
            progressive: false,
            ..self
        }
    }
}

fn lossless_min_symbol(max_bits: u32) -> u32 {
    if max_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * max_bits + 24
    }
}

/// The value of a solid alpha plane (every sample equal), else `None`.
fn constant_alpha_value(alpha: &AlphaPlane) -> Option<i32> {
    fn constant_of<T: Copy + PartialEq>(data: &[T]) -> Option<T> {
        let (&first, rest) = data.split_first()?;
        rest.iter().all(|&v| v == first).then_some(first)
    }
    match alpha {
        AlphaPlane::U8(data) => constant_of(data).map(i32::from),
        AlphaPlane::U16 { data, .. } => constant_of(data).map(i32::from),
        AlphaPlane::F32(data) => constant_of(data),
    }
}
