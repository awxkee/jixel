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
//! `frame` orchestrates displayed and reference frames; `encoder` selects coding
//! candidates. The `ma_*` modules handle learned-tree sources, sampling,
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
pub(crate) use ma_sampling::{DcPlanes, sample_dc_planes, tokenize_dc_planes};
pub(crate) use ma_source::ac_stream_id;
pub(crate) use palette::write_palette_transform;
pub(crate) use predictor::{
    GradPackInteriorFn, GradientScratch, pack_gradient_row, selected_grad_pack_interior_fn,
};
pub(crate) use rct::forward_ycocg;
pub(crate) use squeeze::{
    ExtraSqueeze, LmPlainInput, StreamCrop, dc_frame_supported, encode_dc_frame,
    encode_frame_lossy_modular_plain, encode_frame_lossy_modular_squeeze, learn_global_plane_tree,
    lm_calibrated_distance, lm_plain_distance,
};
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
    shift: u32,
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

const GROUP_DIM: usize = GroupLayout::DEFAULT.dim();

const LF_GROUP_DIM: usize = GroupLayout::DEFAULT.lf_dim();

#[allow(clippy::too_many_arguments)]
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
