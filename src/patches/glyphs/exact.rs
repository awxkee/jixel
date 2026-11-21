/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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
use super::templates::GlyphTemplates;
use super::{GLYPH_ATLAS_MAX_WIDTH, GlyphParams, GlyphShape, Lattice, LossyGlyphPatches};
use crate::image::Image3F;
use crate::patches::color_key::round_to_multiple;
use crate::patches::{MODULAR_PATCH_REF_ID, PatchReference};
use crate::shared_sort::SharedSort;
use std::collections::HashMap;

/// The lossy glyph atlas is a multi-group modular frame; only keep it sane.
pub(super) const GLYPH_ATLAS_MAX_HEIGHT: usize = 4096;

/// Share of the frame the glyph boxes must cover.
const GLYPH_MIN_COVER: f64 = 0.005;
/// Lattice step of the stored glyph values.
pub(super) const GLYPH_ATLAS_STEP: i32 = 6;

fn grow_atlas_plane<T: Copy + Default>(
    plane: &mut Vec<T>,
    old_width: usize,
    old_height: usize,
    width: usize,
    height: usize,
) {
    plane.reserve_exact(width * height - plane.len());
    plane.resize(width * height, T::default());
    if old_width != width {
        // Widen rows from the bottom up so their old pixels stay intact.
        for y in (0..old_height).rev() {
            let src = y * old_width;
            let dst = y * width;
            plane.copy_within(src..src + old_width, dst);
            plane[dst + old_width..dst + width].fill(T::default());
        }
    }
}

/// Indices into the shapes of the occurrences of one glyph; the first is
/// the leader whose pixels the atlas stores.
type Group = Vec<usize>;

/// Groups shapes whose coarse differences from their backgrounds are
/// identical, keeping the groups with enough occurrences.
fn group_identical_shapes(
    lattice: &Lattice,
    label: &[u32],
    shapes: &[GlyphShape],
    min_occurrences: usize,
) -> Vec<Group> {
    let same = |a: &GlyphShape, b: &GlyphShape| -> bool {
        a.w == b.w
            && a.h == b.h
            && (0..3).all(|c| {
                (0..a.h).all(|dy| {
                    (0..a.w).all(|dx| {
                        lattice.diff(label, a, c, dx, dy) == lattice.diff(label, b, c, dx, dy)
                    })
                })
            })
    };
    let mut buckets: HashMap<u64, Vec<Group>> = HashMap::new();
    for (i, s) in shapes.iter().enumerate() {
        let groups = buckets.entry(s.hash).or_default();
        match groups.iter_mut().find(|g| same(&shapes[g[0]], s)) {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    buckets
        .into_values()
        .flatten()
        .filter(|g| g.len() >= min_occurrences)
        .collect()
}

/// Drops the groups that do not pay for their atlas entry. Unrepeated
/// shapes over most of the frame make it a drawing, which the atlas takes
/// whole; a few stray ones are left with the base, and the repeated shapes
/// then have to pay for themselves.
fn drop_unpaying_groups(
    groups: &mut Vec<Group>,
    shapes: &[GlyphShape],
    params: &GlyphParams,
    frame_pixels: usize,
) {
    let pays = |g: &Group| {
        let s = &shapes[g[0]];
        (g.len().max(2) - 1) * s.w * s.h >= params.min_repeat_pixels
    };
    if params.singles_min_cover > 0.0 {
        let single_cover: usize = groups
            .iter()
            .filter(|g| g.len() < 2)
            .map(|g| shapes[g[0]].w * shapes[g[0]].h)
            .sum();
        if (single_cover as f64) < params.singles_min_cover * frame_pixels as f64 {
            groups.retain(|g| g.len() >= 2 && pays(g));
        }
    } else {
        groups.retain(pays);
    }
}

/// Where the leaders go in the exact atlas.
struct Shelves {
    /// Top-left corner per group.
    slots: Vec<(usize, usize)>,
    width: usize,
    height: usize,
}

/// Keeps the most valuable groups the atlas has room for and shelf-packs
/// their leaders, tallest first. `None` when fewer than two groups pay or
/// together they cover too little of the frame.
fn pack_groups(
    groups: &mut Vec<Group>,
    shapes: &[GlyphShape],
    frame_pixels: usize,
) -> Option<Shelves> {
    if groups.len() < 2 {
        return None;
    }
    // Most valuable first; the tail is dropped when the shapes do not fit.
    groups.shared_sort_by_key(|g| {
        let s = &shapes[g[0]];
        (std::cmp::Reverse(g.len() * s.w * s.h), s.y0, s.x0)
    });
    let budget = GLYPH_ATLAS_MAX_WIDTH * GLYPH_ATLAS_MAX_HEIGHT * 3 / 4;
    let mut area = 0usize;
    let keep = groups
        .iter()
        .take_while(|g| {
            let s = &shapes[g[0]];
            area += s.w * s.h;
            area <= budget
        })
        .count();
    groups.truncate(keep);
    let (covered, total_area, max_width) = groups.iter().fold((0, 0, 0), |(cv, ar, mw), g| {
        let s = &shapes[g[0]];
        (cv + g.len() * s.w * s.h, ar + s.w * s.h, mw.max(s.w))
    });
    if (covered as f64) < GLYPH_MIN_COVER * frame_pixels as f64 {
        return None;
    }
    groups.shared_sort_by_key(|g| {
        let s = &shapes[g[0]];
        (
            std::cmp::Reverse(s.h),
            std::cmp::Reverse(g.len() * s.w * s.h),
            s.y0,
            s.x0,
        )
    });
    let width = ((total_area as f64).sqrt() as usize * 3 / 2)
        .max(max_width)
        .clamp(16, GLYPH_ATLAS_MAX_WIDTH);
    let mut slots: Vec<(usize, usize)> = Vec::with_capacity(groups.len());
    let (mut cx, mut cy, mut shelf_h) = (0usize, 0usize, 0usize);
    for g in groups.iter() {
        let s = &shapes[g[0]];
        if cx + s.w > width {
            cx = 0;
            cy += shelf_h;
            shelf_h = 0;
        }
        slots.push((cx, cy));
        cx += s.w;
        shelf_h = shelf_h.max(s.h);
    }
    Some(Shelves {
        slots,
        width,
        height: cy + shelf_h,
    })
}

/// The atlas the exact tiles are filled into: the template atlas grown by
/// `exact_height` rows at the bottom, or a new one.
fn open_atlas(
    templates: Option<GlyphTemplates>,
    exact_width: usize,
    exact_height: usize,
    groups: usize,
) -> LossyGlyphPatches<Option<Image3F>> {
    let Some(templates) = templates else {
        return LossyGlyphPatches {
            atlas: std::array::from_fn(|_| vec![0; exact_width * exact_height]),
            atlas_width: exact_width,
            atlas_height: exact_height,
            base: None,
            references: Vec::with_capacity(groups),
            alpha_atlas: None,
        };
    };
    let mut patches = templates.into_patches();
    let (old_width, old_height) = (patches.atlas_width, patches.atlas_height);
    let width = old_width.max(exact_width);
    let height = old_height + exact_height;
    for plane in &mut patches.atlas {
        grow_atlas_plane(plane, old_width, old_height, width, height);
    }
    if let Some(alpha) = &mut patches.alpha_atlas {
        grow_atlas_plane(alpha, old_width, old_height, width, height);
    }
    patches.atlas_width = width;
    patches.atlas_height = height;
    patches.references.reserve(groups);
    patches
}

/// Stores one leader at an atlas position, on a lattice coarser than the
/// matching one. Every placement subtracts exactly these values from the
/// base.
fn fill_tile(
    lattice: &Lattice,
    label: &[u32],
    s: &GlyphShape,
    (ax, ay): (usize, usize),
    atlas_step: i32,
    atlas: &mut [Vec<i32>; 3],
    atlas_width: usize,
) {
    let [qy, qx, qb] = &lattice.planes;
    let [atlas_y, atlas_x, atlas_b] = atlas;
    let [bg_y, bg_x, bg_b] = s.bg;
    for dy in 0..s.h {
        let src = (s.y0 + dy) * lattice.width + s.x0;
        let dst = (ay + dy) * atlas_width + ax;
        let src_row = src..src + s.w;
        let dst_row = dst..dst + s.w;
        let samples = qy[src_row.clone()]
            .iter()
            .zip(&qx[src_row.clone()])
            .zip(&qb[src_row.clone()]);
        let pixels = atlas_y[dst_row.clone()]
            .iter_mut()
            .zip(&mut atlas_x[dst_row.clone()])
            .zip(&mut atlas_b[dst_row]);
        for ((&owner, ((&y, &x), &b)), ((out_y, out_x), out_b)) in
            label[src_row].iter().zip(samples).zip(pixels)
        {
            if owner == s.label {
                let y = round_to_multiple(y - bg_y, atlas_step);
                *out_y = y;
                *out_x = round_to_multiple(x - bg_x, atlas_step);
                *out_b = round_to_multiple(b - bg_b, atlas_step) - y;
            } else {
                *out_y = 0;
                *out_x = 0;
                *out_b = 0;
            }
        }
    }
}

/// Codes the glyphs the templates left as exact additive patches under the
/// template atlas. Where nothing is worth an atlas the templates stand
/// alone.
pub(super) fn plan_exact_glyphs(
    lattice: &Lattice,
    label: &[u32],
    shapes: &[GlyphShape],
    params: &GlyphParams,
    templates: Option<GlyphTemplates>,
) -> Option<LossyGlyphPatches<Option<Image3F>>> {
    let frame_pixels = lattice.width * lattice.height;
    let template_height = templates.as_ref().map_or(0, |t| t.height);
    let mut groups = group_identical_shapes(lattice, label, shapes, params.min_occurrences);
    drop_unpaying_groups(&mut groups, shapes, params, frame_pixels);
    let shelves = pack_groups(&mut groups, shapes, frame_pixels)
        .filter(|shelves| shelves.height + template_height <= GLYPH_ATLAS_MAX_HEIGHT);
    let Some(shelves) = shelves else {
        return templates.map(GlyphTemplates::into_patches);
    };
    let mut patches = open_atlas(templates, shelves.width, shelves.height, groups.len());
    let atlas_width = patches.atlas_width;
    let atlas_step = GLYPH_ATLAS_STEP.max(params.coarse);
    for (g, (ax, ay)) in groups.into_iter().zip(shelves.slots) {
        let ay = ay + template_height;
        let s = &shapes[g[0]];
        fill_tile(
            lattice,
            label,
            s,
            (ax, ay),
            atlas_step,
            &mut patches.atlas,
            atlas_width,
        );
        patches.references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: s.w,
            height: s.h,
            ref_frame: MODULAR_PATCH_REF_ID,
            alpha_only: false,
            add: true,
            mul: false,
            positions: g
                .into_iter()
                .map(|i| (shapes[i].x0, shapes[i].y0))
                .collect(),
        });
    }
    Some(patches)
}
