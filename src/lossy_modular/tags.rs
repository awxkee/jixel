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

//! Which stream and slot codes each channel crop.
//!
//! Channels that fit one group live in the global (LfGlobal) stream. Larger
//! channels are cropped per DC group when their smaller shift is 3 or more,
//! and per AC group otherwise. A tag is one tree-carrying unit: a global
//! channel, a DC-group slot or an AC-group slot, where a slot is a channel's
//! rank among its group's channels (the value group streams see as property
//! 0). Full-frame channels never produce an empty crop, so the slots are the
//! same in every group.

use super::pyramid::Pyramid;
use crate::lossless::kit::{GROUP_DIM, LF_GROUP_DIM, MaRefPlane, for_each_squeeze_group_crop};
use crate::ma_tree::MA_REF_CHANNELS;
use crate::squeeze::Channel;

/// The frame's group grid.
pub(super) struct Grid {
    pub(super) xsize_groups: usize,
    pub(super) num_ac_groups: usize,
    pub(super) xsize_dc_groups: usize,
    pub(super) num_dc_groups: usize,
}

impl Grid {
    pub(super) fn new(xsize: usize, ysize: usize) -> Self {
        let xsize_groups = xsize.div_ceil(GROUP_DIM);
        let xsize_dc_groups = xsize.div_ceil(LF_GROUP_DIM);
        Self {
            xsize_groups,
            num_ac_groups: xsize_groups * ysize.div_ceil(GROUP_DIM),
            xsize_dc_groups,
            num_dc_groups: xsize_dc_groups * ysize.div_ceil(LF_GROUP_DIM),
        }
    }
}

/// One kind of group stream: its group size and grid, the shift range of the
/// channels it crops, and the tag of its first slot.
pub(super) struct GroupStreams {
    dim: usize,
    per_row: usize,
    pub(super) count: usize,
    min_shift: i32,
    max_shift: i32,
    pub(super) first_tag: usize,
}

impl GroupStreams {
    /// Visit group `group`'s crops as (slot, channel, x0, y0, width, height).
    pub(super) fn for_each_crop(
        &self,
        p: &Pyramid,
        split: usize,
        group: usize,
        visit: impl FnMut(usize, usize, usize, usize, usize, usize),
    ) {
        for_each_squeeze_group_crop(
            &p.channels,
            split,
            self.dim,
            group % self.per_row,
            group / self.per_row,
            self.min_shift,
            self.max_shift,
            visit,
        );
    }
}

/// The tags of a multi-group frame, ordered [global channels][DC slots][AC
/// slots].
pub(super) struct Tags {
    /// Channels before `split` live in the global stream; tag `t < split` is
    /// channel `t`.
    pub(super) split: usize,
    pub(super) n_dc_slots: usize,
    /// Each tag's channel.
    pub(super) channel: Vec<usize>,
    /// Each tag's quantization step (its leaves' multiplier).
    pub(super) quants: Vec<u32>,
    pub(super) dc: GroupStreams,
    pub(super) ac: GroupStreams,
}

impl Tags {
    pub(super) fn new(p: &Pyramid, grid: &Grid) -> Self {
        let channels = &p.channels;
        let nb = channels.len();
        let split = channels
            .iter()
            .position(|c| c.w > GROUP_DIM || c.h > GROUP_DIM)
            .unwrap_or(nb);
        let min_shift = |c: usize| channels[c].hshift.min(channels[c].vshift);
        let dc_slot_channels: Vec<usize> = (split..nb).filter(|&c| min_shift(c) >= 3).collect();
        let ac_slot_channels: Vec<usize> = (split..nb)
            .filter(|&c| (0..=2).contains(&min_shift(c)))
            .collect();
        let n_dc_slots = dc_slot_channels.len();
        let channel: Vec<usize> = (0..split)
            .chain(dc_slot_channels.iter().copied())
            .chain(ac_slot_channels.iter().copied())
            .collect();
        let quants = channel.iter().map(|&c| p.quants[c]).collect();
        Self {
            split,
            n_dc_slots,
            channel,
            quants,
            dc: GroupStreams {
                dim: LF_GROUP_DIM,
                per_row: grid.xsize_dc_groups,
                count: grid.num_dc_groups,
                min_shift: 3,
                max_shift: 1000,
                first_tag: split,
            },
            ac: GroupStreams {
                dim: GROUP_DIM,
                per_row: grid.xsize_groups,
                count: grid.num_ac_groups,
                min_shift: 0,
                max_shift: 2,
                first_tag: split + n_dc_slots,
            },
        }
    }

    pub(super) fn len(&self) -> usize {
        self.channel.len()
    }

    /// Tag `t`'s stream kind: 0 = global, 1 = DC group, 2 = AC group.
    pub(super) fn stream(&self, t: usize) -> usize {
        if t < self.split {
            0
        } else if t < self.ac.first_tag {
            1
        } else {
            2
        }
    }

    /// The channel value the decoder sees for tag `t` (property 0): the
    /// channel index in the global stream, the slot in group streams.
    pub(super) fn chan(&self, t: usize) -> u32 {
        (if t < self.split {
            t
        } else if t < self.ac.first_tag {
            t - self.dc.first_tag
        } else {
            t - self.ac.first_tag
        }) as u32
    }
}

/// Decoder reference properties: for each channel, the two previous
/// same-shaped channels of its stream at full scale. A DC frame references
/// within its global stream; a regular frame also within every group stream,
/// where same-shaped channels hold the same crop (the X and B residuals of a
/// level see its Y residual: 30 of 32 images smaller at Slow, screenshots
/// -2.7..-3.1 %, at 0-4 % encode time).
pub(super) struct References {
    /// Full-scale samples of every channel a later same-shaped one references.
    full_scale: Vec<Option<Vec<i32>>>,
    groups: bool,
}

impl References {
    pub(super) fn new(p: &Pyramid, split: usize) -> Self {
        let channels = &p.channels;
        let referenced = |j: usize| {
            let o = &channels[j];
            channels[j + 1..]
                .iter()
                .any(|c| (o.w, o.h, o.hshift, o.vshift) == (c.w, c.h, c.hshift, c.vshift))
        };
        let n = if p.dc_frame { split } else { channels.len() };
        let full_scale = (0..n)
            .map(|j| {
                referenced(j).then(|| {
                    let q = p.quants[j] as i32;
                    channels[j].data.iter().map(|&v| v * q).collect()
                })
            })
            .collect();
        Self {
            full_scale,
            groups: !p.dc_frame,
        }
    }

    /// The reference planes of global channel `c`.
    pub(super) fn of<'a>(&'a self, channels: &'a [Channel], c: usize) -> Vec<MaRefPlane<'a>> {
        if self.full_scale.is_empty() {
            return Vec::new();
        }
        self.previous(channels, 0, c, 0, 0)
    }

    /// The reference planes of group channel `c` (at or after `split`) for
    /// its crop at (`x0`, `y0`); a whole-channel walk passes (0, 0).
    pub(super) fn of_group<'a>(
        &'a self,
        channels: &'a [Channel],
        split: usize,
        c: usize,
        x0: usize,
        y0: usize,
    ) -> Vec<MaRefPlane<'a>> {
        if !self.groups {
            return Vec::new();
        }
        self.previous(channels, split, c, x0, y0)
    }

    fn previous<'a>(
        &'a self,
        channels: &'a [Channel],
        first: usize,
        c: usize,
        x0: usize,
        y0: usize,
    ) -> Vec<MaRefPlane<'a>> {
        let ch = &channels[c];
        (first..c)
            .rev()
            .filter(|&j| {
                let o = &channels[j];
                (o.w, o.h, o.hshift, o.vshift) == (ch.w, ch.h, ch.hshift, ch.vshift)
            })
            .take(MA_REF_CHANNELS)
            .map(|j| MaRefPlane {
                pixels: self.full_scale[j]
                    .as_deref()
                    .expect("a same-shaped later channel references it"),
                stride: channels[j].w,
                x0,
                y0,
            })
            .collect()
    }
}
