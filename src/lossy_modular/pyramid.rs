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

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The non-XYB frame's integer planes for the arm: `color` holds one (gray)
/// or three channels in the coded plane order, `extra` the extra channels
/// with their squeeze distances (0 = lossless), all in units of `max`.
pub(crate) struct PlainInput {
    pub(crate) color: Vec<Vec<i32>>,
    pub(crate) extra: Vec<(Vec<i32>, f32)>,
    pub(crate) max: f32,
    pub(crate) coding: CodingTransform,
}

/// The arm's quantized state: divided channels, per-channel steps, and the
/// squeeze script that produced them.
pub(crate) struct Pyramid {
    pub(crate) channels: Vec<Channel>,
    pub(crate) quants: Vec<u32>,
    pub(crate) steps: Vec<SqueezeStep>,
    pub(crate) layout: Layout,
    /// Decoder EPF (iterations, modular sigma) for the frame header.
    pub(crate) epf: (u32, f32),
    /// Code the pyramid as the 1:8 DC frame of a VarDCT frame.
    pub(crate) dc_frame: bool,
    pub(crate) xsize: usize,
    pub(crate) ysize: usize,
    /// Patch dictionary entries the decoder adds after the frame (XYB frames;
    /// the atlas is a reference frame coded before this one).
    pub(crate) patches: Vec<PatchReference>,
    /// The frame's group_size_shift: groups of 128 << it pixels, LF groups
    /// 8x that.
    pub(crate) group_shift: u32,
    /// An unsqueezed lossless extra channel coded through a Palette transform
    /// after the squeeze: its index in the squeezed channel list. Channel 0
    /// is then the palette's meta channel (the entries, ascending) and the
    /// channel at that index + 1 holds the indices.
    pub(crate) palette: Option<usize>,
    /// A crop frame over reference slot 0: the rectangle's origin in the
    /// image and whether the frame is the last one.
    pub(crate) crop: Option<([usize; 2], bool)>,
}

impl Pyramid {
    /// The XYB image at the calibrated `distance`, on the screen-content rows
    /// when `screen`. None when the image is too small to squeeze.
    pub(crate) fn xyb(
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
    pub(crate) fn dc_frame(
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
        let planes = crate::xyb::quantize_xyb_channels_with_scales(xyb, XYB_SCALE);
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
        let quants: Vec<u32> = channels
            .iter_mut()
            .enumerate()
            .map(|(i, ch)| {
                let q = q_of(color_of(i), ch.hshift, ch.vshift);
                quantize_channel_divide(ch, q);
                q as u32
            })
            .collect();
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
            palette: None,
            crop: None,
        })
    }

    /// A non-XYB frame's planes at the calibrated `distance`. With
    /// `unsqueeze_lossless`, trailing lossless extra channels stay out of the
    /// squeeze (same samples, another coding). None when the image is too
    /// small to squeeze.
    pub(crate) fn plain(
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
            crate::lossless::rct_simd::forward_ycocg_planes(
                [&r[..n], &g[..n], &b[..n]],
                [&mut y, &mut co, &mut cg],
            );
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
        planes.extend(input.extra.iter().map(|(plane, _)| plane[..n].to_vec()));
        roles.extend(
            input
                .extra
                .iter()
                .map(|&(_, distance)| Role::Extra(distance)),
        );
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
        let quants: Vec<u32> = channels
            .iter_mut()
            .enumerate()
            .map(|(i, ch)| {
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
                q as u32
            })
            .collect();
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
            palette: None,
            crop: None,
        })
    }

    /// [`Self::plain`] with its one extra channel lossless, unsqueezed and
    /// coded through a Palette transform after the squeeze: the channel's
    /// distinct values become a meta channel in front, the samples their
    /// indices. None when the plain pyramid is, when the input carries other
    /// than one lossless extra, or when its values do not fit a global-stream
    /// palette (2..=256 entries, at most a group wide).
    pub(crate) fn plain_palette(
        input: &PlainInput,
        xsize: usize,
        ysize: usize,
        distance: f32,
        pool: &ThreadPool,
    ) -> Option<Self> {
        if !matches!(input.extra.as_slice(), [(_, d)] if *d == 0.0) {
            return None;
        }
        let mut p = Self::plain(input, xsize, ysize, distance, true, pool)?;
        // The unsqueezed extra precedes the chroma pre-squeeze residuals.
        let n_pre: usize = p
            .steps
            .iter()
            .filter(|s| !s.in_place)
            .map(|s| s.num_c)
            .sum();
        let c = p.channels.len() - n_pre - 1;
        let mut entries = p.channels[c].data.clone();
        entries.sort_unstable();
        entries.dedup();
        // The entries are coded divided by their common step (257 for 8-bit
        // levels on the 16-bit lattice) under a leaf multiplier. Deep levels
        // (10- to 16-bit sources) have step 1: their 16-bit entries widen the
        // frame's literal alphabet, which [`min_symbol`] sizes from them.
        let step = entries
            .iter()
            .fold(0u32, |g, &v| gcd(g, v.unsigned_abs()))
            .max(1);
        if !(2..=256.min(p.group_dim())).contains(&entries.len()) || entries[0] < 0 {
            return None;
        }
        for v in &mut p.channels[c].data {
            *v = entries.binary_search(v).expect("an entry") as i32;
        }
        let colors = entries.len();
        p.channels.insert(
            0,
            Channel {
                data: entries.iter().map(|&v| v / step as i32).collect(),
                w: colors,
                h: 1,
                hshift: 0,
                vshift: 0,
            },
        );
        p.quants.insert(0, step);
        p.palette = Some(c);
        Some(p)
    }

    /// The decoded samples of extra channel `k` before the decoder's clamp
    /// (no palette): dequantize, then run the squeeze script backwards.
    pub(crate) fn decoded_extra(&self, k: usize) -> Vec<i32> {
        debug_assert!(self.palette.is_none());
        let Layout::Plain { extra, .. } = self.layout else {
            unreachable!("extra channels live in plain frames")
        };
        let mut channels: Vec<Channel> = self
            .channels
            .iter()
            .zip(&self.quants)
            .map(|(ch, &q)| Channel {
                data: ch.data.iter().map(|&v| v * q as i32).collect(),
                ..*ch
            })
            .collect();
        for s in self.steps.iter().rev() {
            crate::squeeze::apply_step_inverse(&mut channels, s);
        }
        let num_color = channels.len() - extra;
        channels.swap_remove(num_color + k).data
    }

    /// The side of a group, in pixels.
    pub(crate) fn group_dim(&self) -> usize {
        128 << self.group_shift
    }

    /// The side of an LF group, in pixels.
    pub(crate) fn lf_group_dim(&self) -> usize {
        self.group_dim() * 8
    }

    /// The frame header: the DC frame's, or the layout's regular one.
    pub(crate) fn write_frame_header(&self, w: &mut BitWriter) {
        if self.dc_frame {
            write_dc_frame_header(w);
        } else if let Some((origin, last)) = self.crop {
            super::layout::write_xyb_crop_frame_header(
                self.epf,
                self.group_shift,
                origin,
                [self.xsize, self.ysize],
                last,
                w,
            );
        } else {
            self.layout
                .write_frame_header(self.epf, !self.patches.is_empty(), self.group_shift, w);
        }
    }

    /// The patch dictionary that opens the LfGlobal section, if any.
    pub(crate) fn patch_dictionary(&self, scratch: &mut CoderScratch) -> Option<BitWriter> {
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
