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
#[derive(Clone)]
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

    /// Final section padding and TOC size with this extra-channel encoding.
    fn serialized_size(&self, color_bits: &[usize], dim: &ImageDim, last_pass: usize) -> usize {
        let mut bits = color_bits.to_vec();
        bits[0] += self.global.bits_written();
        for (i, stream) in self.lf.iter().enumerate() {
            bits[1 + i] += stream.bits_written();
        }
        let ac_start = 2 + dim.num_dc_groups + last_pass * dim.num_groups;
        for (i, stream) in self.ac.iter().enumerate() {
            bits[ac_start + i] += stream.bits_written();
        }
        // A group permutation is independent of the extra representation;
        // its aligned header has the same size for every candidate.
        super::sections::combined_sections_size(&bits)
    }
}

/// Preserve the original lossless-versus-lossy choice before racing exact
/// representations. A smaller squeezed representation must not turn a plane
/// that previously stayed lossless into a lossy one.
pub(super) struct PreparedExtra {
    sections: ExtraSections,
    squeezed: Option<crate::lossy_modular::ExtraSqueeze>,
    /// The tail race's candidates when they are already coded.
    folds: Vec<ExtraSections>,
    global_code_ans: bool,
}

impl PreparedExtra {
    pub(crate) fn finish(
        self,
        ctx: &EncodingContext,
        dim: &ImageDim,
        dc_group_coords: &[(usize, usize)],
        last_pass: usize,
        color_bits: &[usize],
        scratch: &mut CoderScratch,
    ) -> ExtraSections {
        let original = self.sections;
        let Some(squeezed) = self.squeezed else {
            return original;
        };
        let Some(start) = squeezed.exact_tail_start() else {
            return original;
        };
        let mut best = None;
        let mut best_size = original.serialized_size(color_bits, dim, last_pass);
        if !self.folds.is_empty() {
            for sections in self.folds {
                let size = sections.serialized_size(color_bits, dim, last_pass);
                if size < best_size {
                    best = Some(sections);
                    best_size = size;
                }
            }
            return best.unwrap_or(original);
        }
        // The earliest exact cutoff and one more H/V pair give different
        // coarse prediction contexts. Keep this search bounded at all speeds.
        for retained in [start, start + 2] {
            let Some(candidate) = squeezed.without_exact_tail(retained) else {
                continue;
            };
            let sections = squeezed_sections(
                ctx,
                &candidate,
                dim,
                dc_group_coords,
                self.global_code_ans,
                Some((&squeezed, &original)),
                scratch,
            );
            let size = sections.serialized_size(color_bits, dim, last_pass);
            if size < best_size {
                best = Some(sections);
                best_size = size;
            }
        }
        best.unwrap_or(original)
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

/// The lossless sections of the extra channel, through `palette` when given:
/// its Palette transform and entries in the global stream and the indices per
/// AC group. Otherwise the values themselves.
fn lossless_sections(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    palette: Option<&ExtraPalette>,
    dim: &ImageDim,
    num_dc_groups: usize,
    global_code_ans: bool,
    scratch: &mut CoderScratch,
) -> ExtraSections {
    let mut sections = ExtraSections {
        global: BitWriter::new(),
        lf: (0..num_dc_groups).map(|_| BitWriter::new()).collect(),
        ac: Vec::new(),
    };
    match palette {
        Some(p) => crate::modular::write_global_palette_alpha_modular(
            &p.entries,
            plane.bits(),
            scratch,
            &mut sections.global,
        ),
        None => crate::modular::write_global_alpha_modular(
            plane,
            dim.xsize,
            dim.ysize,
            ctx.extra_step,
            global_code_ans,
            scratch,
            &mut sections.global,
        ),
    }
    let ac_plane = palette.map_or(plane, |p| &p.indices);
    sections.ac = ctx
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
    sections
}

fn squeezed_sections(
    ctx: &EncodingContext,
    squeezed: &crate::lossy_modular::ExtraSqueeze,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    global_code_ans: bool,
    previous: Option<(&crate::lossy_modular::ExtraSqueeze, &ExtraSections)>,
    scratch: &mut CoderScratch,
) -> ExtraSections {
    let mut global = BitWriter::new();
    squeezed.write_global(global_code_ans, scratch, &mut global);
    let lf = if let Some((_, previous)) =
        previous.filter(|(original, _)| original.exact_tail_groups_unchanged(squeezed))
    {
        previous.lf.clone()
    } else {
        ctx.thread_pool
            .steal_map(scratch, dc_group_coords.len(), |i, scratch| {
                let (gx, gy) = dc_group_coords[i];
                let mut w = BitWriter::new();
                squeezed.write_lf_group(gx, gy, scratch, &mut w);
                w
            })
    };
    let ac = if let Some((_, previous)) =
        previous.filter(|(original, _)| original.exact_tail_ac_unchanged(squeezed))
    {
        previous.ac.clone()
    } else {
        ctx.thread_pool
            .steal_map(scratch, dim.num_groups, |group_id, scratch| {
                let mut w = BitWriter::new();
                squeezed.write_ac_group(
                    group_id % dim.xsize_groups,
                    group_id / dim.xsize_groups,
                    scratch,
                    &mut w,
                );
                w
            })
    };
    ExtraSections { global, lf, ac }
}

/// The squeezed coding through the frame's global tree. The tail race picks
/// the representation by its local coding; its AC groups then move onto the
/// global tree where that beats every local coding. Otherwise the folds'
/// local sections go on to the exact tail race.
#[allow(clippy::too_many_arguments)]
fn joint_squeezed(
    ctx: &EncodingContext,
    squeezed: crate::lossy_modular::ExtraSqueeze,
    local: ExtraSections,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    ans: bool,
    streams: &mut extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
) -> PreparedExtra {
    let folds: Vec<(crate::lossy_modular::ExtraSqueeze, ExtraSections)> = squeezed
        .exact_tail_start()
        .map(|start| {
            [start, start + 2]
                .into_iter()
                .filter_map(|retained| squeezed.without_exact_tail(retained))
                .map(|fold| {
                    let sections = squeezed_sections(
                        ctx,
                        &fold,
                        dim,
                        dc_group_coords,
                        ans,
                        Some((&squeezed, &local)),
                        scratch,
                    );
                    (fold, sections)
                })
                .collect()
        })
        .unwrap_or_default();
    let best = folds
        .iter()
        .min_by_key(|(_, sections)| sections.bits())
        .filter(|(_, sections)| sections.bits() < local.bits());
    let (rep, rep_sections) = best.map_or((&squeezed, &local), |(f, s)| (f, s));
    if let Some(tree) =
        extra_tree::plan_global_squeeze_tree(ctx, rep, dim, streams, rep_sections, scratch)
    {
        let mut sections = rep_sections.clone();
        tree.apply(streams, &mut sections);
        return PreparedExtra {
            sections,
            squeezed: None,
            folds: Vec::new(),
            global_code_ans: ans,
        };
    }
    PreparedExtra {
        sections: local,
        squeezed: Some(squeezed),
        folds: folds.into_iter().map(|(_, sections)| sections).collect(),
        global_code_ans: ans,
    }
}

/// Code the extra channel losslessly — through a Palette transform when it
/// has few distinct values and that is smaller, and through the frame's
/// global tree where that beats the fixed local trees — and, when the context
/// asks for a lossy one, also squeezed and quantized; keep the smaller. The
/// lossless coding wins wherever it is no larger, so solid and hard-edged
/// planes stay exact. A winning global-tree coding moves the frame's DC
/// streams onto the merged tree through `streams`.
pub(super) fn prepare_extra_sections(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    last_pass: usize,
    mut streams: extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
) -> PreparedExtra {
    let single = dim.xsize <= K_GROUP_DIM && dim.ysize <= K_GROUP_DIM;
    let lossless_plane = ctx.extra_step == 1 && !plane.is_constant();
    let mut palette = lossless_plane.then(|| extra_palette(plane)).flatten();
    let ans = !streams.code.use_prefix_code;
    let groups = dc_group_coords.len();
    // A multi-group palette-coded channel codes its palette locally in the
    // global stream and its indices per AC group. Renumbering can break the
    // gradients the predictor follows, so without a global tree it races the
    // values coded directly. A global tree can reverse that order and costs
    // more to plan twice than the race saves; it keeps the palette.
    // A single-group one keeps the values' section as the local alternative
    // of its global-tree coding, which carries the palette.
    let mut lossless = match palette.as_ref().filter(|_| !single) {
        Some(p) => {
            let indexed = lossless_sections(ctx, plane, Some(p), dim, groups, ans, scratch);
            if ctx.extra_learned_trees {
                indexed
            } else {
                let plain = lossless_sections(ctx, plane, None, dim, groups, ans, scratch);
                if plain.bits() < indexed.bits() {
                    palette = None;
                    plain
                } else {
                    indexed
                }
            }
        }
        None => lossless_sections(ctx, plane, None, dim, groups, ans, scratch),
    };
    let squeezed = (ctx.extra_squeeze_distance > 0.0 && !plane.is_constant())
        .then(|| {
            crate::lossy_modular::ExtraSqueeze::new(
                plane,
                dim.xsize,
                dim.ysize,
                ctx.extra_squeeze_distance,
                ctx.extra_learned_trees,
                ctx.extra_context_trees,
            )
        })
        .flatten();
    let lossy = squeezed
        .as_ref()
        .map(|s| squeezed_sections(ctx, s, dim, dc_group_coords, ans, None, scratch));
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
        Some(lossy) if lossy.bits() < lossless_bits => {
            let squeezed = squeezed.expect("lossy sections come from a squeeze");
            if ctx.speed.effort().extra_global_tree {
                return joint_squeezed(
                    ctx,
                    squeezed,
                    lossy,
                    dim,
                    dc_group_coords,
                    ans,
                    &mut streams,
                    scratch,
                );
            }
            PreparedExtra {
                sections: lossy,
                squeezed: squeezed.exact_tail_start().is_some().then_some(squeezed),
                folds: Vec::new(),
                global_code_ans: ans,
            }
        }
        _ => {
            if let Some(tree) = global_tree {
                tree.apply(&mut streams, &mut lossless);
            }
            PreparedExtra {
                sections: lossless,
                squeezed: None,
                folds: Vec::new(),
                global_code_ans: ans,
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn writer(bits: usize) -> BitWriter {
        let mut w = BitWriter::new();
        for _ in 0..bits {
            w.write(1, 1);
        }
        w
    }

    #[test]
    fn extra_size_matches_aligned_sections_and_last_pass() {
        for (width, height) in [(63, 61), (513, 17), (2051, 257)] {
            let dim = ImageDim::new(width, height);
            for passes in [1, 3] {
                let count = 2 + dim.num_dc_groups + passes * dim.num_groups;
                for boundary in [7, 8184, 8191, 139256] {
                    let color_bits: Vec<usize> = (0..count).map(|i| boundary + i % 8).collect();
                    let extra = ExtraSections {
                        global: writer(13),
                        lf: (0..dim.num_dc_groups).map(|i| writer(5 + i)).collect(),
                        ac: (0..dim.num_groups).map(|i| writer(9 + i)).collect(),
                    };
                    let mut sections: Vec<BitWriter> =
                        color_bits.iter().map(|&bits| writer(bits)).collect();
                    sections[0].append(&extra.global);
                    for (i, lf) in extra.lf.iter().enumerate() {
                        sections[1 + i].append(lf);
                    }
                    let last_ac = 2 + dim.num_dc_groups + (passes - 1) * dim.num_groups;
                    for (i, ac) in extra.ac.iter().enumerate() {
                        sections[last_ac + i].append(ac);
                    }
                    let mut output = BitWriter::new();
                    super::super::sections::combine_sections(&mut sections, &mut output);
                    assert_eq!(
                        extra.serialized_size(&color_bits, &dim, passes - 1),
                        output.bits_written() / 8,
                    );
                }
            }
        }
    }
}
