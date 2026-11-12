/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! The frame's integer layout and the header fields it decides.
//!
//! [`Layout::Xyb`] stores [Y, X, B-Y] on the [`XYB_SCALE`] lattice and
//! signals that lattice through custom dc_quant steps. [`Layout::Plain`]
//! stores a non-XYB frame's own samples, through the YCoCg RCT for RGB. The
//! layout picks the frame header, the dc_quant field, the transform list and
//! the decoder EPF.

use crate::bit_writer::BitWriter;
use crate::lossless::kit::{
    GroupLayout, write_frame_dimension, write_modular_transforms_rct_squeeze,
    write_modular_transforms_squeeze_only, write_modular_transforms_squeeze_palette, write_u64,
};

/// XYB float -> integer scale factors in X, Y, B order. Signaled to the decoder
/// as custom dc_quant steps (their reciprocals); powers of two stay F16-exact.
pub(super) static XYB_SCALE: [f32; 3] = [65536.0, 4096.0, 4096.0];

/// How the arm's modular image maps onto the frame: the XYB lattice, or the
/// image's own integer samples (a non-XYB frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// [Y, X, B-Y] on the `XYB_SCALE` lattice, signaled through custom
    /// dc_quant steps.
    Xyb,
    /// The samples in bit-depth units; the decoder divides by the sample
    /// range. `ycbcr` sets the frame's do_YCbCr bit, `rct` codes the three
    /// color channels through the YCoCg RCT, `extra` counts the extra
    /// channels appended after the color ones.
    Plain {
        ycbcr: bool,
        rct: bool,
        extra: usize,
    },
}

impl Layout {
    /// `patched` sets the frame's kPatches flag (XYB frames only).
    /// `group_shift` is the frame's group_size_shift (groups of 128 << it).
    pub(super) fn write_frame_header(
        self,
        epf: (u32, f32),
        patched: bool,
        group_shift: u32,
        w: &mut BitWriter,
    ) {
        match self {
            Self::Xyb => write_xyb_frame_header(epf, patched, group_shift, w),
            Self::Plain { ycbcr, extra, .. } => {
                write_plain_frame_header(ycbcr, extra, epf, group_shift, w)
            }
        }
    }

    /// The LfGlobal dc_quant field.
    pub(super) fn write_dc_quant(self, w: &mut BitWriter) {
        match self {
            Self::Xyb => write_xyb_dc_quant(w),
            // Unused by the decoder outside XYB.
            Self::Plain { .. } => w.write(1, 1),
        }
    }

    /// The transform list: [RCT], Squeeze, then the extra channel's Palette
    /// as (begin_c in the squeezed list, colors) when `palette` is set.
    pub(super) fn write_transforms(
        self,
        steps: &[crate::squeeze::SqueezeStep],
        palette: Option<(usize, usize)>,
        w: &mut BitWriter,
    ) {
        let rct = matches!(self, Self::Plain { rct: true, .. });
        match palette {
            Some((begin_c, colors)) => write_modular_transforms_squeeze_palette(
                rct,
                steps,
                begin_c as u32,
                colors as u32,
                w,
            ),
            None if rct => write_modular_transforms_rct_squeeze(steps, w),
            None => write_modular_transforms_squeeze_only(steps, w),
        }
    }

    /// Decoder-side edge-preserving filter (iterations,
    /// `epf_sigma_for_modular`), sigma linear in the arm's distance.
    pub(super) fn epf(self, distance: f32) -> (u32, f32) {
        let d = distance;
        match self {
            Self::Xyb => (if d < 5.0 { 1 } else { 2 }, 0.69 + 0.25 * d),
            Self::Plain { rct: true, .. } => (2, 1.5 + 0.29 * d),
            Self::Plain { .. } => (if d < 10.0 { 1 } else { 2 }, 1.6 + 0.055 * d),
        }
    }
}

/// Regular is_last modular frame inside the lossy (`xyb_encoded`) codestream.
/// Identical to `write_frame_header_modular_flags` except the do_YCbCr bit is
/// absent: with xyb_encoded set the color transform is implicitly XYB.
fn write_xyb_frame_header(epf: (u32, f32), patched: bool, group_shift: u32, w: &mut BitWriter) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(if patched { 2 } else { 0 }, w); // flags: kPatches
    w.write(2, 0b00); // upsampling = 1
    w.write(2, group_shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    w.write(1, 1); // is_last
    w.write(2, 0b00); // name length = 0
    write_loop_filter(epf, w);
    w.write(2, 0b00); // no FH extensions
}

/// Cropped modular frame of the lossy codestream: the rectangle at `origin`
/// replaces reference slot 0 there and, unless `last`, the result goes back
/// to slot 0 (a crop frame over a saved base, see `crop_frames`).
pub(super) fn write_xyb_crop_frame_header(
    epf: (u32, f32),
    group_shift: u32,
    origin: [usize; 2],
    size: [usize; 2],
    last: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(2, 0b00); // upsampling = 1
    w.write(2, group_shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 1); // have_crop
    write_frame_dimension(2 * origin[0], w); // PackSigned(x0)
    write_frame_dimension(2 * origin[1], w);
    write_frame_dimension(size[0], w);
    write_frame_dimension(size[1], w);
    w.write(2, 0b00); // blending = Replace
    w.write(2, 0); // source = 0
    if last {
        w.write(1, 1); // is_last
    } else {
        w.write(1, 0); // not the last frame
        w.write(2, 0); // save_as_reference = 0
    }
    w.write(2, 0b00); // name length = 0
    write_loop_filter(epf, w);
    w.write(2, 0b00); // no FH extensions
}

/// Modular DC frame (kDCFrame, dc_level 1) of the lossy codestream: like
/// the regular header up to the passes, then the level, no loop filters.
pub(super) fn write_dc_frame_header(w: &mut BitWriter) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b01); // DC frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(2, 0b00); // upsampling = 1
    w.write(2, GroupLayout::DEFAULT.shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(2, 0b00); // dc_level = 1
    w.write(2, 0b00); // name length = 0
    write_loop_filter((0, 0.0), w);
    w.write(2, 0b00); // no FH extensions
}

/// Regular single-pass modular frame of a non-XYB codestream, no loop
/// filters (same shape as the lossless header plus the YCbCr signaling).
fn write_plain_frame_header(
    ycbcr: bool,
    extra: usize,
    epf: (u32, f32),
    group_shift: u32,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    w.write(1, u64::from(ycbcr)); // do_YCbCr
    if ycbcr {
        for _ in 0..3 {
            w.write(2, 0); // chroma subsampling 4:4:4
        }
    }
    w.write(2, 0b00); // upsampling = 1
    for _ in 0..extra {
        w.write(2, 0b00); // ec_upsampling = 1
    }
    w.write(2, group_shift as u64);
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    for _ in 0..extra {
        w.write(2, 0b00); // extra-channel blending = Replace
    }
    w.write(1, 1); // is_last
    w.write(2, 0b00); // name length = 0
    write_loop_filter(epf, w);
    w.write(2, 0b00); // no FH extensions
}

/// Loop-filter field of the modular frame header: no gaborish, EPF at the
/// arm's schedule (all defaults except the modular sigma).
fn write_loop_filter(epf: (u32, f32), w: &mut BitWriter) {
    let (iters, sigma) = epf;
    w.write(1, 0); // loop_filter not all_default
    w.write(1, 0); // no gaborish
    w.write(2, iters as u64); // EPF iters
    if iters > 0 {
        w.write(1, 0); // epf_weight_custom = false
        w.write(1, 0); // epf_sigma_custom = false
        w.write(16, crate::util::f32_to_f16_bits(sigma) as u64); // epf_sigma_for_modular
    }
    w.write(2, 0b00); // no LF extensions
}

/// Custom dc_quant field: the decoder multiplies stored X/Y/B channels by these
/// steps to recover XYB floats.
fn write_xyb_dc_quant(w: &mut BitWriter) {
    use crate::util::f32_to_f16_bits;
    w.write(1, 0); // dc_quant not all_default
    for scale in XYB_SCALE {
        let step_x128 = 128.0 / scale;
        w.write(16, f32_to_f16_bits(step_x128) as u64);
    }
}
