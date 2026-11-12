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
use std::borrow::Cow;

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
    /// Actual selected lossy samples, after the decoder's sample clamp.
    /// Only retained for the opt-in function-field race. `None` means the
    /// input stays exact, or reconstruction tracking was not requested.
    reconstruction: Option<AlphaPlane>,
    /// What this coding adds to the frame before section padding: its own
    /// sections, less what a shared global tree saves on the DC streams.
    net_bits: usize,
}

impl PreparedExtra {
    pub(super) fn reconstruction(&self) -> Option<&AlphaPlane> {
        self.reconstruction.as_ref()
    }

    pub(super) fn net_bits(&self) -> usize {
        self.net_bits
    }

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
/// A gradient-only tree has cut it by at most 52 % (32 RGBA images, every
/// tier up to Fast), so below 1 / 2.5 of it the squeeze always stays.
const GRADIENT_TREE_MAX_SAVING: f64 = 2.5;

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
    let indices = match plane {
        AlphaPlane::U8(data) => data.iter().map(|&v| index_of[usize::from(v)]).collect(),
        AlphaPlane::U16 { data, .. } => data.iter().map(|&v| index_of[usize::from(v)]).collect(),
        AlphaPlane::F32(_) => return None,
    };
    Some(ExtraPalette {
        entries,
        indices: AlphaPlane::U8(indices),
    })
}

/// The lossless sections of the extra channel, through `palette` when given:
/// its Palette transform and entries in the global stream and the indices per
/// AC group. Otherwise the values themselves.
#[allow(clippy::too_many_arguments)]
fn lossless_sections(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    palette: Option<&ExtraPalette>,
    constant_integer: bool,
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
    let encode_group = |group_id, scratch: &mut CoderScratch| {
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
    };
    sections.ac =
        if constant_integer && (dim.xsize / K_GROUP_DIM > 1 || dim.ysize / K_GROUP_DIM > 1) {
            // Prediction restarts at each group, so a constant integer plane's
            // local coding depends only on the group's width and height.
            let mut by_shape: [Option<BitWriter>; 4] = Default::default();
            (0..dim.num_groups)
                .map(|group_id| {
                    let x0 = (group_id % dim.xsize_groups) * K_GROUP_DIM;
                    let y0 = (group_id / dim.xsize_groups) * K_GROUP_DIM;
                    let narrow = usize::from(dim.xsize - x0 < K_GROUP_DIM);
                    let short = usize::from(dim.ysize - y0 < K_GROUP_DIM);
                    by_shape[narrow + 2 * short]
                        .get_or_insert_with(|| encode_group(group_id, scratch))
                        .clone()
                })
                .collect()
        } else {
            ctx.thread_pool
                .steal_map(scratch, dim.num_groups, encode_group)
        };
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
struct JointSqueeze {
    squeezed: crate::lossy_modular::ExtraSqueeze,
    local: ExtraSections,
    folds: Vec<(crate::lossy_modular::ExtraSqueeze, ExtraSections)>,
    /// The fold the global tree codes (None: the unfolded squeeze).
    rep: Option<usize>,
    tree: Option<extra_tree::GlobalExtraTree>,
    ans: bool,
}

impl JointSqueeze {
    #[allow(clippy::too_many_arguments)]
    fn plan(
        ctx: &EncodingContext,
        squeezed: crate::lossy_modular::ExtraSqueeze,
        local: ExtraSections,
        dim: &ImageDim,
        dc_group_coords: &[(usize, usize)],
        ans: bool,
        streams: &extra_tree::GlobalStreams<'_>,
        scratch: &mut CoderScratch,
    ) -> Self {
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
        let rep = (0..folds.len())
            .min_by_key(|&i| folds[i].1.bits())
            .filter(|&i| folds[i].1.bits() < local.bits());
        let (rep_squeeze, rep_sections) =
            rep.map_or((&squeezed, &local), |i| (&folds[i].0, &folds[i].1));
        let tree = extra_tree::plan_global_squeeze_tree(
            ctx,
            rep_squeeze,
            dim,
            streams,
            rep_sections,
            scratch,
        );
        Self {
            squeezed,
            local,
            folds,
            rep,
            tree,
            ans,
        }
    }

    /// The smallest coding the plan reaches, before section padding.
    fn bits(&self) -> usize {
        let rep = self.rep.map_or(&self.local, |i| &self.folds[i].1);
        rep.bits() - self.tree.as_ref().map_or(0, |t| t.saving())
    }

    fn finish(
        mut self,
        streams: &mut extra_tree::GlobalStreams<'_>,
        reconstruction: Option<AlphaPlane>,
    ) -> PreparedExtra {
        let net_bits = self.bits();
        if let Some(tree) = self.tree {
            let mut sections = self.rep.map_or(self.local, |i| self.folds.swap_remove(i).1);
            tree.apply(streams, &mut sections);
            return PreparedExtra {
                sections,
                squeezed: None,
                folds: Vec::new(),
                global_code_ans: self.ans,
                reconstruction,
                net_bits,
            };
        }
        PreparedExtra {
            net_bits,
            sections: self.local,
            squeezed: Some(self.squeezed),
            folds: self
                .folds
                .into_iter()
                .map(|(_, sections)| sections)
                .collect(),
            global_code_ans: self.ans,
            reconstruction,
        }
    }
}

/// Native nested Palette/profile sections. They carry their own trees and
/// leave the frame's shared DC entropy bundle unchanged.
pub(super) fn prepare_nested_sections(
    pool: &crate::thread_pool::ThreadPool,
    profile: &[i32],
    u: &[i32],
    v: &[i32],
    dim: &ImageDim,
    reconstruction: &AlphaPlane,
    portable: bool,
    scratch: &mut CoderScratch,
) -> Option<PreparedExtra> {
    if u.len() != dim.xsize
        || v.len() != dim.ysize
        || reconstruction.len() != dim.xsize.checked_mul(dim.ysize)?
    {
        return None;
    }
    // The single-palette coding costs the first row and column of every
    // group; it is the choice only where the file must stay readable by
    // decoders that cannot parse a palette's explicit deltas.
    let (global, ac) = if portable {
        crate::modular::portable_profile::encode(pool, profile, u, v, scratch)?
    } else {
        crate::modular::nested_profile::encode(pool, profile, u, v, scratch)?
    };
    if ac.len() != dim.num_groups {
        return None;
    }
    let net_bits = global.bits_written() + ac.iter().map(BitWriter::bits_written).sum::<usize>();
    Some(PreparedExtra {
        net_bits,
        sections: ExtraSections {
            global,
            lf: (0..dim.num_dc_groups).map(|_| BitWriter::new()).collect(),
            ac,
        },
        squeezed: None,
        folds: Vec::new(),
        global_code_ans: true,
        reconstruction: None,
    })
}

/// Code the extra channel losslessly — through a Palette transform when it
/// has few distinct values and that is smaller, and through the frame's
/// global tree where that beats the fixed local trees — and, when the context
/// asks for a lossy one, also squeezed and quantized; keep the smaller. The
/// lossless coding wins wherever it is no larger, so solid and hard-edged
/// planes stay exact. A winning global-tree coding moves the frame's DC
/// streams onto the merged tree through `streams`.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_extra_sections(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    last_pass: usize,
    streams: extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
    track_reconstruction: bool,
) -> PreparedExtra {
    prepare_extra_sections_impl(
        ctx,
        plane,
        dim,
        dc_group_coords,
        last_pass,
        streams,
        scratch,
        false,
        1.0,
        None,
        track_reconstruction,
    )
}

/// [`prepare_extra_sections`] with the context's lossy distance scaled.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_extra_sections_scaled(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    last_pass: usize,
    streams: extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
    distance_scale: f32,
    memo: Option<&LevelsMemo>,
) -> PreparedExtra {
    prepare_extra_sections_impl(
        ctx,
        plane,
        dim,
        dc_group_coords,
        last_pass,
        streams,
        scratch,
        false,
        distance_scale,
        memo,
        true,
    )
}

/// Code a strongly repeated alpha plane exactly, including its learned tree.
/// A repeated mask can have an exceptionally expensive fixed stream and a
/// tiny learned stream, beyond the lossy path's usual fourfold cost bound.
///
/// The caller screens for substantial nonconstant repetitions and races the
/// completed frame against the normal path. The shared DC entropy bundle can
/// change, so an extra-channel bit estimate alone cannot guarantee no growth.
/// The context must retain exact samples (`extra_step == 1`).
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_extra_sections_lossless(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    last_pass: usize,
    streams: extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
) -> PreparedExtra {
    debug_assert_eq!(ctx.extra_step, 1);
    prepare_extra_sections_impl(
        ctx,
        plane,
        dim,
        dc_group_coords,
        last_pass,
        streams,
        scratch,
        true,
        1.0,
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_extra_sections_impl(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    dim: &ImageDim,
    dc_group_coords: &[(usize, usize)],
    last_pass: usize,
    mut streams: extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
    force_lossless: bool,
    distance_scale: f32,
    memo: Option<&LevelsMemo>,
    track_reconstruction: bool,
) -> PreparedExtra {
    let single = dim.xsize <= K_GROUP_DIM && dim.ysize <= K_GROUP_DIM;
    let try_squeeze = !force_lossless && ctx.extra_squeeze_distance > 0.0;
    let constant = (ctx.extra_step == 1 || try_squeeze) && plane.is_constant();
    let lossless_plane = ctx.extra_step == 1 && !constant;
    let constant_integer = ctx.extra_step == 1 && constant && !matches!(plane, AlphaPlane::F32(_));
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
            let indexed = lossless_sections(ctx, plane, Some(p), false, dim, groups, ans, scratch);
            if ctx.extra_plane_tree.is_some() {
                indexed
            } else {
                let plain = lossless_sections(ctx, plane, None, false, dim, groups, ans, scratch);
                if plain.bits() < indexed.bits() {
                    palette = None;
                    plain
                } else {
                    indexed
                }
            }
        }
        None => lossless_sections(
            ctx,
            plane,
            None,
            constant_integer,
            dim,
            groups,
            ans,
            scratch,
        ),
    };
    let squeezed = (try_squeeze && !constant)
        .then(|| {
            crate::lossy_modular::ExtraSqueeze::new(
                plane,
                dim.xsize,
                dim.ysize,
                ctx.extra_squeeze_distance * distance_scale,
                ctx.extra_learned_trees,
                ctx.extra_context_trees,
                ctx.extra_black,
            )
        })
        .flatten();
    let lossy = squeezed
        .as_ref()
        .map(|s| squeezed_sections(ctx, s, dim, dc_group_coords, ans, None, scratch));
    // A constant plane (the opaque alpha a layered or templated RGB image
    // declares) shares one leaf through the global tree instead of carrying a
    // local tree and histogram in every group.
    let constant_global = ctx.extra_step == 1 && !single && constant;
    let global_tree = ((lossless_plane || constant_global)
        && ctx.extra_plane_tree.is_some_and(|budget| {
            lossy.as_ref().is_none_or(|lossy| {
                if budget.gradient_only {
                    (lossless.bits() as f64) < GRADIENT_TREE_MAX_SAVING * lossy.bits() as f64
                } else {
                    lossless.bits() < GLOBAL_TREE_MAX_SAVING * lossy.bits()
                }
            })
        }))
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
            drop((lossless, palette, global_tree));
            let squeezed = squeezed.expect("lossy sections come from a squeeze");
            let mut reconstruction = None;
            if ctx.speed.effort().extra_global_tree {
                let joint = JointSqueeze::plan(
                    ctx,
                    squeezed,
                    lossy,
                    dim,
                    dc_group_coords,
                    ans,
                    &streams,
                    scratch,
                );
                let levels = lossy_palette(
                    ctx,
                    plane,
                    &joint.squeezed,
                    joint.bits(),
                    dim,
                    last_pass,
                    &mut streams,
                    scratch,
                    memo,
                    track_reconstruction.then_some(&mut reconstruction),
                );
                return levels.unwrap_or_else(|| {
                    let reconstruction = reconstruction.or_else(|| {
                        track_reconstruction
                            .then(|| squeezed_reconstruction(plane, joint.squeezed.reconstruct()))
                    });
                    joint.finish(&mut streams, reconstruction)
                });
            }
            if let Some(levels) = lossy_palette(
                ctx,
                plane,
                &squeezed,
                lossy.bits(),
                dim,
                last_pass,
                &mut streams,
                scratch,
                memo,
                track_reconstruction.then_some(&mut reconstruction),
            ) {
                return levels;
            }
            let reconstruction = reconstruction.or_else(|| {
                track_reconstruction.then(|| squeezed_reconstruction(plane, squeezed.reconstruct()))
            });
            PreparedExtra {
                net_bits: lossy.bits(),
                sections: lossy,
                squeezed: squeezed.exact_tail_start().is_some().then_some(squeezed),
                folds: Vec::new(),
                global_code_ans: ans,
                reconstruction,
            }
        }
        _ => {
            if let Some(tree) = global_tree {
                tree.apply(&mut streams, &mut lossless);
                // The joint code is ANS: a large plane planned on a prefix-coded
                // frame re-writes its global header with the ANS signature. A
                // palette's global section has its own local tree.
                if !ans && palette.is_none() && !single {
                    lossless.global = BitWriter::new();
                    crate::modular::write_global_alpha_modular(
                        plane,
                        dim.xsize,
                        dim.ysize,
                        ctx.extra_step,
                        true,
                        scratch,
                        &mut lossless.global,
                    );
                }
            }
            PreparedExtra {
                sections: lossless,
                squeezed: None,
                folds: Vec::new(),
                global_code_ans: ans,
                reconstruction: None,
                net_bits: lossless_bits,
            }
        }
    }
}

/// Convert the selected squeeze to the samples actually exposed by a decoder.
/// K's no-ink bias may leave the raw inverse outside the nominal sample range.
fn squeezed_reconstruction(plane: &AlphaPlane, samples: Vec<i32>) -> AlphaPlane {
    match plane {
        AlphaPlane::U8(_) => {
            AlphaPlane::U8(samples.into_iter().map(|v| v.clamp(0, 255) as u8).collect())
        }
        AlphaPlane::U16 { bits, .. } => {
            let max = (1i32 << *bits) - 1;
            AlphaPlane::U16 {
                data: samples
                    .into_iter()
                    .map(|v| v.clamp(0, max) as u16)
                    .collect(),
                bits: *bits,
            }
        }
        AlphaPlane::F32(_) => unreachable!("float extra channels are not squeezed"),
    }
}

/// The exact coding of lossy-palette level planes, shared by codings that
/// start from identical DC streams (the alternatives of one extra-channel
/// race). Different planes often quantize to the same few levels, and the
/// learned tree of a level plane is most of what coding it costs.
#[derive(Default)]
pub(super) struct LevelsMemo {
    entries: std::sync::Mutex<Vec<LevelsEntry>>,
}

type LevelsCoding = (ExtraSections, Option<extra_tree::GlobalExtraTree>);
type LevelsEntry = (
    AlphaPlane,
    std::sync::Arc<std::sync::OnceLock<LevelsCoding>>,
);

impl LevelsMemo {
    /// The slot for `levels`; a coding already under way elsewhere is waited
    /// for rather than repeated.
    fn slot(
        &self,
        levels: &AlphaPlane,
    ) -> Option<std::sync::Arc<std::sync::OnceLock<LevelsCoding>>> {
        let same = |a: &AlphaPlane, b: &AlphaPlane| match (a, b) {
            (AlphaPlane::U8(a), AlphaPlane::U8(b)) => a == b,
            (
                AlphaPlane::U16 {
                    data: a,
                    bits: abits,
                },
                AlphaPlane::U16 {
                    data: b,
                    bits: bbits,
                },
            ) => abits == bbits && a == b,
            _ => false,
        };
        let mut entries = self.entries.lock().ok()?;
        if let Some((_, slot)) = entries.iter().find(|(known, _)| same(known, levels)) {
            return Some(slot.clone());
        }
        let slot = std::sync::Arc::default();
        entries.push((levels.clone(), std::sync::Arc::clone(&slot)));
        Some(slot)
    }
}

/// The lossy palette arm skips its exact coding when the order-0 estimate of
/// its indices ([`crate::lossy_modular::palette_index_bits`]) is above this
/// many times the squeeze's coded bits.
const PALETTE_SCREEN: f64 = 4.0;

/// The lossy palette arm: the fewest levels within part of the squeeze's
/// weighted squared error ([`super::extra_levels`]), coded losslessly like an
/// exact plane (palette, the learned plane tree, global tree), when that is
/// smaller than the squeeze's best coding. Only with a plane tree: the fixed
/// per-group coding codes palette indices worse than the squeeze.
#[allow(clippy::too_many_arguments)]
fn lossy_palette(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    squeezed: &crate::lossy_modular::ExtraSqueeze,
    squeezed_bits: usize,
    dim: &ImageDim,
    last_pass: usize,
    streams: &mut extra_tree::GlobalStreams<'_>,
    scratch: &mut CoderScratch,
    memo: Option<&LevelsMemo>,
    reconstruction: Option<&mut Option<AlphaPlane>>,
) -> Option<PreparedExtra> {
    if ctx.extra_plane_tree.is_none() || ctx.extra_step != 1 {
        return None;
    }
    let weights = super::extra_levels::error_weights(plane, dim.xsize, dim.ysize);
    let samples = squeezed.reconstruct();
    let budget = super::extra_levels::squeeze_budget(plane, &samples, &weights);
    let track_reconstruction = reconstruction.is_some();
    if let Some(reconstruction) = reconstruction {
        *reconstruction = Some(squeezed_reconstruction(plane, samples));
    } else {
        drop(samples);
    }
    let levels = super::extra_levels::fewest_levels(plane, &weights, budget, ctx.extra_black)?;
    let estimate = match &levels {
        AlphaPlane::U8(data) => {
            crate::lossy_modular::palette_index_bits_u8(data, dim.xsize, dim.ysize)
        }
        _ => crate::lossy_modular::palette_index_bits(
            &levels.to_i32_vec(levels.len()),
            dim.xsize,
            dim.ysize,
        ),
    };
    if estimate > PALETTE_SCREEN * squeezed_bits as f64 {
        return None;
    }
    let single = dim.xsize <= K_GROUP_DIM && dim.ysize <= K_GROUP_DIM;
    let ans = !streams.code.use_prefix_code;
    let groups = dim.num_dc_groups;
    let code = |scratch: &mut CoderScratch| {
        let palette = extra_palette(&levels);
        let local_palette = palette.as_ref().filter(|_| !single);
        let sections = lossless_sections(
            ctx,
            &levels,
            local_palette,
            false,
            dim,
            groups,
            ans,
            scratch,
        );
        let tree = extra_tree::plan_global_extra_tree(
            ctx,
            &levels,
            palette.as_ref(),
            dim,
            last_pass,
            streams,
            &sections,
            scratch,
        );
        (sections, tree)
    };
    let slot = memo.and_then(|memo| memo.slot(&levels));
    let coding = match slot.as_ref() {
        Some(slot) => Cow::Borrowed(slot.get_or_init(|| code(scratch))),
        None => Cow::Owned(code(scratch)),
    };
    let (sections, tree) = coding.as_ref();
    let net_bits = sections.bits() - tree.as_ref().map_or(0, |t| t.saving());
    if net_bits >= squeezed_bits {
        return None;
    }
    let (mut sections, tree) = coding.into_owned();
    if let Some(tree) = tree {
        tree.apply(streams, &mut sections);
    }
    Some(PreparedExtra {
        net_bits,
        sections,
        squeezed: None,
        folds: Vec::new(),
        global_code_ans: ans,
        reconstruction: track_reconstruction.then_some(levels),
    })
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

    #[test]
    fn tracked_black_reconstruction_clamps_the_no_ink_bias() {
        for plane in [
            AlphaPlane::U8(vec![255; 33 * 35]),
            AlphaPlane::U16 {
                data: vec![65535; 33 * 35],
                bits: 16,
            },
        ] {
            let squeeze =
                crate::lossy_modular::ExtraSqueeze::new(&plane, 33, 35, 1.5, false, false, true)
                    .unwrap();
            let max = (1i32 << plane.bits()) - 1;
            assert!(squeeze.reconstruct().iter().any(|&v| v > max));
            let tracked = squeezed_reconstruction(&plane, squeeze.reconstruct());
            assert_eq!(tracked.bits(), plane.bits());
            assert_eq!(tracked.len(), plane.len());
            assert!((0..tracked.len()).all(|i| tracked.get_i32(i) == max));
        }
    }
}
