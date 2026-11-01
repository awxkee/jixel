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

//! Lossy modular coding: an image, or one extra channel, as a quantized
//! Squeeze pyramid in modular streams instead of VarDCT.
//!
//! A lossy modular frame runs these stages, one module each:
//!
//! 1. [`calibrate`]: map the caller's distance to this arm's quantizer
//!    distance, so the modular and VarDCT arms land near the same quality.
//! 2. [`pyramid`]: put the source planes on the frame's integer [`layout`]
//!    (the XYB lattice, or the samples themselves through YCoCg), run the
//!    forward Squeeze, and divide every channel by its [`quantize`] step.
//! 3. [`fold`]: screen folding the pyramid's exact coarse tail back into
//!    full-scale averages: the same decoded image, another entropy
//!    representation.
//! 4. [`race`]: code the pyramid, and the screen's fold when it found one;
//!    keep the smaller frame.
//! 5. [`code`]: code one pyramid as a complete frame. A single-group frame is
//!    one section ([`single`]). Otherwise [`tags`] assigns every channel crop
//!    to a stream slot, [`trees`] learns one MA subtree per tag, [`forest`]
//!    merges them under the global tree, [`tokens`] tokenizes every stream
//!    and [`write`] emits the sections and TOC. [`cache`] carries unchanged
//!    trees and tokens from one race candidate to the next.
//!
//! [`alpha`] codes a lossy extra channel inside a VarDCT frame with the same
//! squeeze, quantizer and forest.

mod alpha;
#[cfg(test)]
mod alpha_tests;
mod cache;
mod calibrate;
mod code;
mod fold;
mod forest;
mod layout;
mod pyramid;
#[cfg(test)]
mod pyramid_tests;
mod quantize;
mod race;
mod single;
mod tags;
mod tokens;
mod trees;
mod write;

pub(crate) use alpha::ExtraSqueeze;
pub(crate) use calibrate::{calibrated_distance, plain_distance};
pub(crate) use pyramid::PlainInput;

use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::image::Image3F;
use crate::thread_pool::ThreadPool;
use pyramid::Pyramid;

/// Lossy modular frame of an XYB codestream. `distance` is the arm's
/// calibrated quantizer distance ([`calibrated_distance`] and the Auto gate).
/// Returns the frame size in bits before the fold race, for the Auto gate,
/// or None when the image is too small to squeeze.
pub(crate) fn encode_xyb_frame(
    xyb: &Image3F,
    distance: f32,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    let pyramid = Pyramid::xyb(xyb, distance)?;
    race::encode(pyramid, speed, pool, scratch, writer)
}

/// Lossy modular frame of a non-XYB codestream: the integer planes in
/// `input` at the plain-layout schedule ([`plain_distance`]). Returns the
/// frame size in bits before the fold race, for the Auto gate, or None when
/// the image is too small to squeeze.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_plain_frame(
    input: &PlainInput,
    xsize: usize,
    ysize: usize,
    distance: f32,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    let pyramid = Pyramid::plain(input, xsize, ysize, distance)?;
    race::encode(pyramid, speed, pool, scratch, writer)
}

/// Progressive DC: the 1:8 DC image of a VarDCT frame as its squeezed DC
/// frame (frame type kDCFrame, dc_level 1), which the VarDCT frame then
/// references instead of coding its own DC. Full-resolution chroma, no loop
/// filters. `levels` keeps that many of the finest squeeze levels (clamped
/// to at least one; `None` squeezes down to the 8x8 preview). Returns false
/// when the image is too small to squeeze.
pub(crate) fn encode_dc_frame(
    dc: &Image3F,
    vardct_steps: [f32; 3],
    levels: Option<u32>,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    match Pyramid::dc_frame(dc, vardct_steps, levels) {
        Some(pyramid) => race::encode(pyramid, speed, pool, scratch, writer).is_some(),
        None => false,
    }
}

/// Whether [`encode_dc_frame`] can code a DC image of this size.
pub(crate) fn dc_frame_supported(xsize: usize, ysize: usize) -> bool {
    !crate::squeeze::default_squeeze_steps(xsize, ysize, 3).is_empty()
}
