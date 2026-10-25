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

//! Palette, lossless and squeezed coding choices for extra channels.

use super::extra_tree;
use super::{ImageDim, K_GROUP_DIM};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;

/// Modular data of the extra channel, coded ahead of the frame sections:
/// the LfGlobal part, one part per DC group and one per AC group.
pub(super) struct ExtraSections {
    pub(super) global: BitWriter,
    pub(super) lf: Vec<BitWriter>,
    pub(super) ac: Vec<BitWriter>,
}

impl ExtraSections {
    pub(super) fn bits(&self) -> usize {
        self.global.bits_written()
            + self.lf.iter().map(BitWriter::bits_written).sum::<usize>()
            + self.ac.iter().map(BitWriter::bits_written).sum::<usize>()
    }
}

/// A lossless extra channel of few distinct values, coded through a Palette
/// transform: the values in index order and one index per sample.
pub(super) struct ExtraPalette {
    pub(super) entries: Vec<i32>,
    pub(super) indices: AlphaPlane,
}

/// Largest palette an extra channel is coded through.
const EXTRA_PALETTE_MAX_COLORS: usize = 256;
/// The global tree has not cut a plane's fixed lossless coding below a
/// quarter; past that the squeezed coding is out of its reach.
const GLOBAL_TREE_MAX_SAVING: usize = 4;

/// The palette of an integer extra channel whose distinct values number at
/// most `EXTRA_PALETTE_MAX_COLORS` and at most half its span, so that the
/// index image is narrower than the values.
fn extra_palette(plane: &AlphaPlane) -> Option<ExtraPalette> {
    let range = match plane {
        AlphaPlane::U8(_) => 1usize << 8,
        AlphaPlane::U16 { bits, .. } => 1usize << bits,
        AlphaPlane::F32(_) => return None,
    };
    let mut seen = vec![false; range];
    for i in 0..plane.len() {
        let v = plane.get_i32(i);
        if v < 0 || v as usize >= range {
            return None;
        }
        seen[v as usize] = true;
    }
    let entries: Vec<i32> = (0..range as i32).filter(|&v| seen[v as usize]).collect();
    let (Some(&lo), Some(&hi)) = (entries.first(), entries.last()) else {
        return None;
    };
    if entries.len() < 2
        || entries.len() > EXTRA_PALETTE_MAX_COLORS
        || 2 * entries.len() as i64 > i64::from(hi - lo) + 1
    {
        return None;
    }
    let mut index_of = vec![0u8; range];
    for (index, &v) in entries.iter().enumerate() {
        index_of[v as usize] = index as u8;
    }
    let indices = (0..plane.len())
        .map(|i| index_of[plane.get_i32(i) as usize])
        .collect();
    Some(ExtraPalette {
        entries,
        indices: AlphaPlane::U8(indices),
    })
}

/// Code the extra channel losslessly — through a Palette transform when it
/// has few distinct values, and through the frame's global tree where that
/// beats the fixed local trees — and, when the context asks for a lossy one,
/// also squeezed and quantized; keep the smaller. The lossless coding wins
/// wherever it is no larger, so solid and hard-edged planes stay exact. A
/// winning global-tree coding moves the frame's DC streams onto the merged
/// tree through `streams`.
pub(super) fn prepare_extra_sections(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    last_pass: usize,
    mut streams: extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
) -> ExtraSections {
    let single = dim.xsize <= K_GROUP_DIM && dim.ysize <= K_GROUP_DIM;
    let lossless_plane = ctx.extra_step == 1 && !plane.is_constant();
    let palette = lossless_plane.then(|| extra_palette(plane)).flatten();
    let mut lossless = ExtraSections {
        global: BitWriter::new(),
        lf: (0..dc_group_coords.len())
            .map(|_| BitWriter::new())
            .collect(),
        ac: Vec::new(),
    };
    // A multi-group palette-coded channel codes its palette locally in the
    // global stream and its indices per AC group. A single-group one keeps
    // the values' section as the local alternative of its global-tree
    // coding, which carries the palette.
    let coded = palette.as_ref().filter(|_| !single);
    match coded {
        Some(p) => crate::modular::write_global_palette_alpha_modular(
            &p.entries,
            plane.bits(),
            scratch,
            &mut lossless.global,
        ),
        None => crate::modular::write_global_alpha_modular(
            plane,
            dim.xsize,
            dim.ysize,
            ctx.extra_step,
            !streams.code.use_prefix_code,
            scratch,
            &mut lossless.global,
        ),
    }
    let ac_plane = coded.map_or(plane, |p| &p.indices);
    lossless.ac = ctx
        .thread_pool
        .steal_map(scratch, dim.num_groups, |group_id, scratch| {
            let x0 = (group_id % dim.xsize_groups) * K_GROUP_DIM;
            let y0 = (group_id / dim.xsize_groups) * K_GROUP_DIM;
            let mut w = BitWriter::new();
            crate::modular::write_ac_group_alpha(
                ac_plane,
                dim.xsize,
                dim.ysize,
                x0,
                y0,
                K_GROUP_DIM.min(dim.xsize - x0),
                K_GROUP_DIM.min(dim.ysize - y0),
                ctx.extra_step,
                scratch,
                &mut w,
            );
            w
        });
    let lossy = (ctx.extra_squeeze_distance > 0.0 && !plane.is_constant())
        .then(|| {
            crate::lossless::ExtraSqueeze::new(
                plane,
                dim.xsize,
                dim.ysize,
                ctx.extra_squeeze_distance,
                ctx.extra_learned_trees,
            )
        })
        .flatten()
        .map(|squeezed| {
            let mut lossy = ExtraSections {
                global: BitWriter::new(),
                lf: Vec::new(),
                ac: Vec::new(),
            };
            squeezed.write_global(scratch, &mut lossy.global);
            lossy.lf = ctx
                .thread_pool
                .steal_map(scratch, dc_group_coords.len(), |i, scratch| {
                    let (gx, gy) = dc_group_coords[i];
                    let mut w = BitWriter::new();
                    squeezed.write_lf_group(gx, gy, scratch, &mut w);
                    w
                });
            lossy.ac = ctx
                .thread_pool
                .steal_map(scratch, dim.num_groups, |group_id, scratch| {
                    let mut w = BitWriter::new();
                    squeezed.write_ac_group(
                        group_id % dim.xsize_groups,
                        group_id / dim.xsize_groups,
                        scratch,
                        &mut w,
                    );
                    w
                });
            lossy
        });
    let global_tree = (lossless_plane
        && ctx.extra_learned_trees
        && lossy
            .as_ref()
            .is_none_or(|lossy| lossless.bits() < GLOBAL_TREE_MAX_SAVING * lossy.bits()))
    .then(|| {
        extra_tree::plan_global_extra_tree(
            ctx,
            plane,
            palette.as_ref(),
            dim,
            last_pass,
            &streams,
            &lossless,
            scratch,
        )
    })
    .flatten();
    let lossless_bits = lossless.bits() - global_tree.as_ref().map_or(0, |t| t.saving());
    match lossy {
        Some(lossy) if lossy.bits() < lossless_bits => lossy,
        _ => {
            if let Some(tree) = global_tree {
                tree.apply(&mut streams, &mut lossless);
            }
            lossless
        }
    }
}

pub(super) fn zero_alpha_for_lossy(
    alpha: Option<&AlphaPlane>,
    pixels: usize,
) -> Option<AlphaPlane> {
    match alpha {
        Some(AlphaPlane::U8(_)) => Some(AlphaPlane::U8(vec![0; pixels])),
        Some(AlphaPlane::U16 { bits, .. }) => Some(AlphaPlane::U16 {
            data: vec![0; pixels],
            bits: *bits,
        }),
        Some(AlphaPlane::F32(_)) => Some(AlphaPlane::F32(vec![0; pixels])),
        None => None,
    }
}
