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

//! Stage 2: the quantized Squeeze pyramid.
//!
//! The source planes go onto the layout's integer lattice, the squeeze script
//! runs forward (a 4:2:0 chroma pre-squeeze, then alternating H/V levels down
//! to the 8 px preview) and every resulting channel is divided by its
//! quantization step.

use super::closed_loop::{Deadzone, requantize};
use super::layout::{Layout, XYB_SCALE, write_dc_frame_header};
use super::quantize::{
    Role, SCREEN_ROW_MUL, dc_frame_step, plain_step, quantize_channel_divide, xyb_step,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::coding::CodingTransform;
use crate::image::Image3F;
use crate::lossless::kit::GroupLayout;
use crate::patches::PatchReference;
use crate::squeeze::{
    Channel, SqueezeStep, apply_step_forward, default_squeeze_steps, lossy_squeeze_steps,
    lossy_squeeze_steps_chroma, squeeze_steps_levels,
};
use crate::thread_pool::ThreadPool;

/// The non-XYB frame's integer planes for the arm: `color` holds one (gray)
/// or three channels in the coded plane order, `extra` the extra channels
/// with their squeeze distances (0 = lossless), all in units of `max`.
pub(crate) struct PlainInput {
    pub color: Vec<Vec<i32>>,
    pub extra: Vec<(Vec<i32>, f32)>,
    pub max: f32,
    pub coding: CodingTransform,
}

/// The arm's quantized state: divided channels, per-channel steps, and the
/// squeeze script that produced them.
pub(super) struct Pyramid {
    pub(super) channels: Vec<Channel>,
    pub(super) quants: Vec<u32>,
    pub(super) steps: Vec<SqueezeStep>,
    pub(super) layout: Layout,
    /// Decoder EPF (iterations, modular sigma) for the frame header.
    pub(super) epf: (u32, f32),
    /// Code the pyramid as the 1:8 DC frame of a VarDCT frame.
    pub(super) dc_frame: bool,
    pub(super) xsize: usize,
    pub(super) ysize: usize,
    /// Patch dictionary entries the decoder adds after the frame (XYB frames;
    /// the atlas is a reference frame coded before this one).
    pub(super) patches: Vec<PatchReference>,
    /// The frame's group_size_shift: groups of 128 << it pixels, LF groups
    /// 8x that.
    pub(super) group_shift: u32,
}

impl Pyramid {
    /// The XYB image at the calibrated `distance`, on the screen-content rows
    /// when `screen`. None when the image is too small to squeeze.
    pub(super) fn xyb(
        xyb: &Image3F,
        distance: f32,
        screen: bool,
        pool: &ThreadPool,
    ) -> Option<Self> {
        let steps = lossy_squeeze_steps(xyb.xsize(), xyb.ysize(), 3);
        let row_mul = if screen { SCREEN_ROW_MUL } else { [1.0; 3] };
        let pyramid = Self::xyb_with(
            xyb,
            steps,
            Deadzone::COLOR,
            pool,
            |color, hshift, vshift| xyb_step(color, hshift, vshift, distance, row_mul),
        )?;
        Some(Self {
            epf: Layout::Xyb.epf(distance),
            ..pyramid
        })
    }

    /// The 1:8 DC image of a VarDCT frame, quantized relative to its
    /// `vardct_steps` and squeezed `levels` deep.
    pub(super) fn dc_frame(
        dc: &Image3F,
        vardct_steps: [f32; 3],
        levels: Option<u32>,
        pool: &ThreadPool,
    ) -> Option<Self> {
        // Stored color [Y, X, B-Y] -> the VarDCT DC step in lattice units.
        let unit = [
            vardct_steps[1] * XYB_SCALE[1],
            vardct_steps[0] * XYB_SCALE[0],
            vardct_steps[2] * XYB_SCALE[2],
        ];
        let levels = levels.map_or(usize::MAX, |n| n.max(1) as usize);
        let steps = squeeze_steps_levels(dc.xsize(), dc.ysize(), 3, levels);
        let pyramid = Self::xyb_with(dc, steps, Deadzone::NONE, pool, |color, hshift, vshift| {
            dc_frame_step(unit, color, hshift, vshift)
        })?;
        Some(Self {
            dc_frame: true,
            ..pyramid
        })
    }

    /// An XYB image over the squeeze script `steps` (optionally led by the
    /// chroma pre-squeeze), each channel at `q_of(stored color [Y, X, B-Y],
    /// hshift, vshift)`.
    fn xyb_with(
        xyb: &Image3F,
        steps: Vec<SqueezeStep>,
        deadzone: Deadzone,
        pool: &ThreadPool,
        q_of: impl Fn(usize, i32, i32) -> i32,
    ) -> Option<Self> {
        let xsize = xyb.xsize();
        let ysize = xyb.ysize();
        let num_c = 3usize;
        if default_squeeze_steps(xsize, ysize, num_c).is_empty() {
            // Tiny image: no squeeze means "lossy" would be plain color
            // quantization at the wrong scale. Not this arm's job.
            return None;
        }
        // With the chroma pre-squeeze, the last four channels are its residuals
        // (mapped to the chroma quantization row, like cjxl); everything before
        // them stays aligned to `num_c` blocks.
        let chroma_pre = steps.len() >= 2 && !steps[0].in_place;

        // XYB floats onto the integer lattice, stored as [Y, X, B-Y].
        let n = xsize * ysize;
        let input = [
            &xyb.plane_data(0)[..n],
            &xyb.plane_data(1)[..n],
            &xyb.plane_data(2)[..n],
        ];
        let mut planes = [vec![0i32; n], vec![0i32; n], vec![0i32; n]];
        {
            let [dst_y, dst_x, dst_b] = &mut planes;
            unsafe {
                crate::xyb::selected_quantize_xyb_channels_fn()(
                    input,
                    [dst_y, dst_x, dst_b],
                    XYB_SCALE,
                );
            }
        }
        let mut channels = full_size_channels(planes.into_iter(), xsize, ysize);
        for s in &steps {
            apply_step_forward(&mut channels, s);
        }
        let exact = channels.clone();
        // Quantize every channel on its (color, depth) schedule. Channels are
        // stored divided by their step; the tree leaf multiplier restores the
        // scale. The decoder's full-scale predictions stay consistent because
        // every predictor used on a q>1 channel is exactly scale-equivariant over
        // the divided values.
        let nb = channels.len();
        let color_of = |i: usize| {
            if chroma_pre && i >= nb - 4 {
                1 // trailing chroma pre-squeeze residuals -> chroma row
            } else {
                i % num_c
            }
        };
        let mut quants: Vec<u32> = Vec::with_capacity(nb);
        for (i, ch) in channels.iter_mut().enumerate() {
            let q = q_of(color_of(i), ch.hshift, ch.vshift);
            quantize_channel_divide(ch, q);
            quants.push(q as u32);
        }
        requantize(&mut channels, &quants, &steps, exact, deadzone, pool);
        Some(Self {
            channels,
            quants,
            steps,
            layout: Layout::Xyb,
            epf: (0, 0.0),
            dc_frame: false,
            xsize,
            ysize,
            patches: Vec::new(),
            group_shift: GroupLayout::DEFAULT.shift,
        })
    }

    /// A non-XYB frame's planes at the calibrated `distance`. With
    /// `unsqueeze_lossless`, trailing lossless extra channels stay out of the
    /// squeeze (same samples, another coding). None when the image is too
    /// small to squeeze.
    pub(super) fn plain(
        input: &PlainInput,
        xsize: usize,
        ysize: usize,
        distance: f32,
        unsqueeze_lossless: bool,
        pool: &ThreadPool,
    ) -> Option<Self> {
        let num_color = input.color.len();
        debug_assert!(num_color == 1 || num_color == 3);
        let num_c = num_color + input.extra.len();
        if default_squeeze_steps(xsize, ysize, num_c).is_empty() {
            return None;
        }
        let n = xsize * ysize;
        let mut planes: Vec<Vec<i32>> = Vec::with_capacity(num_c);
        let mut roles: Vec<Role> = Vec::with_capacity(num_c);
        let (ycbcr, rct, chroma): (bool, bool, Vec<usize>) = match (num_color, input.coding) {
            (3, CodingTransform::YCbCr) => (true, false, vec![0, 2]),
            (3, _) => (false, true, vec![1, 2]),
            _ => (false, false, Vec::new()),
        };
        if rct {
            let [r, g, b] = [&input.color[0], &input.color[1], &input.color[2]];
            let mut y = vec![0i32; n];
            let mut co = vec![0i32; n];
            let mut cg = vec![0i32; n];
            for (((y, co), cg), ((&r, &g), &b)) in y
                .iter_mut()
                .zip(&mut co)
                .zip(&mut cg)
                .zip(r[..n].iter().zip(&g[..n]).zip(&b[..n]))
            {
                (*y, *co, *cg) = crate::lossless::kit::forward_ycocg(r, g, b);
            }
            planes.extend([y, co, cg]);
        } else {
            planes.extend(input.color.iter().map(|p| p[..n].to_vec()));
        }
        roles.extend((0..num_color).map(|c| {
            if chroma.contains(&c) {
                Role::Chroma
            } else {
                Role::Luma
            }
        }));
        for (plane, distance) in &input.extra {
            planes.push(plane[..n].to_vec());
            roles.push(Role::Extra(*distance));
        }
        let squeezed = if unsqueeze_lossless {
            num_color
                + input
                    .extra
                    .iter()
                    .rposition(|&(_, d)| d > 0.0)
                    .map_or(0, |i| i + 1)
        } else {
            num_c
        };
        let unsqueezed = num_c - squeezed;
        let steps = lossy_squeeze_steps_chroma(xsize, ysize, squeezed, &chroma);
        let n_pre: usize = steps.iter().filter(|s| !s.in_place).map(|s| s.num_c).sum();
        let mut channels = full_size_channels(planes.into_iter(), xsize, ysize);
        for s in &steps {
            apply_step_forward(&mut channels, s);
        }
        let exact = channels.clone();
        let nb = channels.len();
        let mut quants: Vec<u32> = Vec::with_capacity(nb);
        for (i, ch) in channels.iter_mut().enumerate() {
            // The chroma pre-squeeze residuals come last, after the unsqueezed
            // extras.
            let role = if i >= nb - n_pre {
                Role::Chroma
            } else if i >= nb - n_pre - unsqueezed {
                Role::Extra(0.0)
            } else {
                roles[i % squeezed]
            };
            let q = plain_step(
                role,
                distance,
                ch.hshift,
                ch.vshift,
                input.max,
                input.coding,
            );
            quantize_channel_divide(ch, q);
            quants.push(q as u32);
        }
        requantize(&mut channels, &quants, &steps, exact, Deadzone::COLOR, pool);
        let layout = Layout::Plain {
            ycbcr,
            rct,
            extra: input.extra.len(),
        };
        Some(Self {
            channels,
            quants,
            steps,
            layout,
            epf: layout.epf(distance),
            dc_frame: false,
            xsize,
            ysize,
            patches: Vec::new(),
            group_shift: GroupLayout::DEFAULT.shift,
        })
    }

    /// The side of a group, in pixels.
    pub(super) fn group_dim(&self) -> usize {
        128 << self.group_shift
    }

    /// The side of an LF group, in pixels.
    pub(super) fn lf_group_dim(&self) -> usize {
        self.group_dim() * 8
    }

    /// The frame header: the DC frame's, or the layout's regular one.
    pub(super) fn write_frame_header(&self, w: &mut BitWriter) {
        if self.dc_frame {
            write_dc_frame_header(w);
        } else {
            self.layout
                .write_frame_header(self.epf, !self.patches.is_empty(), self.group_shift, w);
        }
    }

    /// The patch dictionary that opens the LfGlobal section, if any.
    pub(super) fn patch_dictionary(&self, scratch: &mut CoderScratch) -> Option<BitWriter> {
        if self.patches.is_empty() {
            return None;
        }
        let mut w = BitWriter::new();
        crate::lossless::write_patch_dictionary(&self.patches, false, scratch, &mut w);
        Some(w)
    }
}

/// Unshifted channels holding `planes`.
fn full_size_channels(
    planes: impl Iterator<Item = Vec<i32>>,
    xsize: usize,
    ysize: usize,
) -> Vec<Channel> {
    planes
        .map(|data| Channel {
            data,
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        })
        .collect()
}
