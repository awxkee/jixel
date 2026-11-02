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
//!    At ExtraSlow, `polish` then revisits the XYB frame's luma residuals
//!    against the exact SSIMULACRA2 and ColorVideoVDP arbiters.
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
mod closed_loop;
mod code;
mod fold;
mod forest;
mod layout;
#[cfg(feature = "extraslow")]
mod polish;
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
/// calibrated quantizer distance ([`calibrated_distance`] and the Auto gate)
/// and `screen` whether the image is screen content (its step rows);
/// `patches` are dictionary entries over an atlas frame the caller has
/// already written.
/// Returns the frame size in bits before the fold race, for the Auto gate,
/// or None when the image is too small to squeeze.
pub(crate) fn encode_xyb_frame(
    xyb: &Image3F,
    distance: f32,
    screen: bool,
    patches: &[crate::patches::PatchReference],
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    let mut pyramid = Pyramid::xyb(xyb, distance, screen, pool)?;
    pyramid.patches = patches.to_vec();
    pyramid.group_shift = group_shift(speed);
    #[cfg(feature = "extraslow")]
    if speed.effort().lm_residual_polish() {
        polish::polish_frame(&mut pyramid, xyb, distance, screen, pool, scratch);
    }
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
    // A lossless extra channel (alpha, say) codes the same samples squeezed
    // or not: unsqueezed wins on cutouts and stickers (to -27 %), the squeeze
    // on smooth fades (one vignette +7..+14 % without it). The colour pyramid
    // is the same either way, so from Slow the extras alone decide, coded
    // under an all-zero gray plane at Fastest (crops and tile samples break
    // the fades and pick wrong); faster tiers take unsqueezed, -9.7 % over
    // eight RGBA images and quicker to code.
    let unsqueeze = input.extra.last().is_some_and(|&(_, d)| d == 0.0)
        && (speed < crate::Speed::Slow || {
            let probe = PlainInput {
                color: vec![vec![0; xsize * ysize]],
                extra: input.extra.clone(),
                max: input.max,
                coding: input.coding,
            };
            let bits = |unsqueeze: bool, scratch: &mut CoderScratch| {
                let pyramid = Pyramid::plain(&probe, xsize, ysize, distance, unsqueeze, pool)?;
                let mut w = BitWriter::new();
                code::write_frame(&pyramid, crate::Speed::Fastest, pool, scratch, &mut w, None);
                Some(w.bits_written())
            };
            match (bits(false, scratch), bits(true, scratch)) {
                (Some(squeezed), Some(plain)) => plain < squeezed,
                _ => false,
            }
        });
    let mut pyramid = Pyramid::plain(input, xsize, ysize, distance, unsqueeze, pool)?;
    pyramid.group_shift = group_shift(speed);
    race::encode(pyramid, speed, pool, scratch, writer)
}

/// The group_size_shift of a regular frame. From Slow, 1024-pixel groups:
/// every group edge costs its first row and column their neighbours, so
/// larger groups code the same pixels in -1.3 % (Slow) / -1.5 % (Fast) over
/// 54 forced encodes, to -4.6 %. They cost 18-24 % encode time at 8
/// threads and the decoder's group parallelism, so faster tiers keep 256.
fn group_shift(speed: crate::Speed) -> u32 {
    if speed >= crate::Speed::Slow {
        crate::lossless::kit::GroupLayout::LARGE.shift
    } else {
        crate::lossless::kit::GroupLayout::DEFAULT.shift
    }
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
    match Pyramid::dc_frame(dc, vardct_steps, levels, pool) {
        Some(pyramid) => race::encode(pyramid, speed, pool, scratch, writer).is_some(),
        None => false,
    }
}

/// Whether [`encode_dc_frame`] can code a DC image of this size.
pub(crate) fn dc_frame_supported(xsize: usize, ysize: usize) -> bool {
    !crate::squeeze::default_squeeze_steps(xsize, ysize, 3).is_empty()
}
