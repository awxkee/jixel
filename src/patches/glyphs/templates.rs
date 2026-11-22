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
use super::exact::GLYPH_ATLAS_MAX_HEIGHT;
use super::{GLYPH_ATLAS_MAX_WIDTH, GlyphParams, GlyphShape, Lattice, LossyGlyphPatches};
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::patches::{MODULAR_PATCH_REF_ID, PatchReference};
use crate::shared_sort::SharedSort;
use crate::util::FastRound;
use std::collections::HashMap;

/// Glyph templates: one atlas tile per glyph holds its color and its
/// coverage (atlas alpha) and is placed with kBlendAbove, so a glyph shared
/// over several backgrounds needs one tile instead of one per background.
/// Under a template the base is the source un-premultiplied by the
/// coverage, which keeps the composite exact where the coverage leaves at
/// least this much of the base; deeper inside the stroke the base takes
/// the occurrence's flat background.
const TEMPLATE_MIN_BASE_WEIGHT: f32 = 0.35;
/// Coverage levels a template is rounded to before its 8-bit atlas alpha.
const TEMPLATE_ALPHA_LEVELS: f32 = 128.0;
/// Mean coverage difference from the template a member may have; larger
/// mismatches turn into base noise once un-premultiplied.
const TEMPLATE_MAX_ALPHA_DIFF: f32 = 0.05;
/// Lattice tolerance (Y, X, B) of the stroke cores, where the template's
/// two-color model is all the decoder shows.
static TEMPLATE_CORE_TOLERANCE: [f32; 3] = [6.0, 24.0, 6.0];
/// ... and on smooth ground, where one coverage cannot put a mix on the
/// opsin curve of every background at once.
static TEMPLATE_SOFT_CORE_TOLERANCE: [f32; 3] = [12.0, 36.0, 12.0];
/// Lattice granularity (Y, X, B) of the foreground colors one template shares.
static TEMPLATE_COLOR_STEP: [f32; 3] = [16.0, 64.0, 16.0];

/// Template atlas, placements and base of the glyphs taken by templates.
pub(super) struct GlyphTemplates {
    atlas: [Vec<i32>; 3],
    alpha: Vec<u8>,
    width: usize,
    pub(super) height: usize,
    base: Image3F,
    references: Vec<PatchReference>,
}

impl GlyphTemplates {
    pub(super) fn into_patches(self) -> LossyGlyphPatches<Option<Image3F>> {
        LossyGlyphPatches {
            atlas: self.atlas,
            atlas_width: self.width,
            atlas_height: self.height,
            base: Some(self.base),
            references: self.references,
            alpha_atlas: Some(self.alpha),
        }
    }
}

/// Lattice units (`INV_DC_QUANT`) of the planes in Y, X, B order.
static TEMPLATE_LATTICE: [f32; 3] = [512.0, 4096.0, 256.0];
/// Weights that make a lattice step of each channel comparable.
const TEMPLATE_CHANNEL_WEIGHTS: [f32; 3] = [64.0, 4096.0, 16.0];

/// One glyph occurrence as a template member: its foreground (mean of the
/// strongest fifth of its pixels) and every pixel's coverage between its
/// background and that foreground.
struct Occurrence {
    fg: [f32; 3],
    fg_delta: [f32; 3],
    alpha: Vec<f32>,
    /// On smooth ground: the coverage relative to the occurrence's own
    /// strongest pixels, kept while `alpha` is fitted against the foreground
    /// of one candidate group after another.
    relative: Vec<f32>,
}

/// Glyph groups that share a blend-above template (see
/// [`TEMPLATE_MIN_BASE_WEIGHT`]): same mask and foreground color, coverage
/// that agrees, two or more backgrounds.
struct TemplatePlan<'a> {
    shapes: &'a [GlyphShape],
    occurrences: Vec<Occurrence>,
    groups: Vec<Vec<usize>>,
}

/// A template: the members' mean coverage, rounded, and mean foreground.
fn template_of(
    ctx: &EncodingContext,
    shapes: &[GlyphShape],
    occurrences: &[Occurrence],
    members: &[usize],
    alpha: &mut Vec<f32>,
) -> [f32; 3] {
    let GlyphShape { w, h, .. } = shapes[members[0]];
    let inv = 1.0 / members.len() as f32;
    alpha.clear();
    alpha.resize(w * h, 0.0);
    let mut fg = [0.0f32; 3];
    for &i in members {
        let coverage = &occurrences[i].alpha;
        let len = alpha.len().min(coverage.len());
        (ctx.add_scaled)(&mut alpha[..len], coverage, inv);
        for c in 0..3 {
            fg[c] += occurrences[i].fg[c] * inv;
        }
    }
    for t in alpha {
        *t = (*t * TEMPLATE_ALPHA_LEVELS).fast_round() / TEMPLATE_ALPHA_LEVELS;
    }
    fg
}

impl TemplatePlan<'_> {
    /// Atlas area the templates save over exact patches: every exact group
    /// a template absorbs beyond one is a tile the atlas no longer stores.
    fn saved_area(&self) -> usize {
        let mut hashes = Vec::new();
        self.groups
            .iter()
            .map(|g| {
                hashes.clear();
                hashes.extend(g.iter().map(|&i| self.shapes[i].hash));
                hashes.shared_sort();
                hashes.dedup();
                let s = &self.shapes[g[0]];
                (hashes.len() - 1) * s.w * s.h
            })
            .sum()
    }
}

/// Mean coverage difference between occurrences of one glyph on smooth
/// ground: the opsin curve bends the same coverage a little differently
/// over every background.
const TEMPLATE_MAX_SOFT_ALPHA_DIFF: f32 = 0.09;
/// Spread of the backgrounds (squared, in weighted lattice units per
/// member) below which they say nothing about the foreground.
const TEMPLATE_MIN_GROUND_SPREAD: f32 = 1e-4;
/// A group on smooth ground stands when one template keeps this share of
/// it; otherwise it is cut in two by ground, this many times at most.
const TEMPLATE_SPAN_KEEP: (usize, usize) = (4, 5);
const TEMPLATE_MAX_SPAN_DEPTH: u32 = 5;
/// The strongest pixels of an antialiased glyph hold at least this much of
/// its foreground.
const TEMPLATE_MIN_PEAK_COVERAGE: f32 = 0.4;

/// The foreground the occurrences of one glyph share, from how its
/// strongest pixels move with the background. Those pixels are a mix
/// `bg + s (fg - bg)` with one peak coverage `s`, so over several
/// backgrounds their difference from the ground falls by `s` per unit of
/// ground: the slope gives `s`, and the foreground follows. Every member
/// then takes that foreground and its coverage toward it. Members on one
/// ground keep their own estimate (`s` = 1).
fn share_foreground(
    lattice: &Lattice,
    label: &[u32],
    shapes: &[GlyphShape],
    occurrences: &mut [Occurrence],
    group: &[usize],
) {
    let weights = TEMPLATE_CHANNEL_WEIGHTS;
    let inv = 1.0 / group.len() as f32;
    let ground = |i: usize, c: usize| shapes[i].bg[c] as f32 / TEMPLATE_LATTICE[c];
    // Back to every member's own estimate: an earlier, wider group may
    // have shared another.
    for &i in group {
        let occurrence = &mut occurrences[i];
        occurrence.fg = std::array::from_fn(|c| ground(i, c) + occurrence.fg_delta[c]);
        occurrence.alpha.clone_from(&occurrence.relative);
    }
    let mut mean_bg = [0.0f32; 3];
    let mut mean_delta = [0.0f32; 3];
    for &i in group {
        for c in 0..3 {
            mean_bg[c] += ground(i, c) * inv;
            mean_delta[c] += occurrences[i].fg_delta[c] * inv;
        }
    }
    let (mut covariance, mut variance) = (0.0f32, 0.0f32);
    for &i in group {
        for c in 0..3 {
            let b = ground(i, c) - mean_bg[c];
            covariance += weights[c] * b * (occurrences[i].fg_delta[c] - mean_delta[c]);
            variance += weights[c] * b * b;
        }
    }
    if variance * inv < TEMPLATE_MIN_GROUND_SPREAD {
        return;
    }
    let peak = (-covariance / variance).clamp(TEMPLATE_MIN_PEAK_COVERAGE, 1.0);
    let fg: [f32; 3] = std::array::from_fn(|c| mean_bg[c] + mean_delta[c] / peak);
    // Coverage against the shared foreground, pixel by pixel: the opsin
    // curve bends a mix away from the straight line between its ends, and
    // the coverage that lands each pixel on that line is what the decoder
    // needs, not the one the renderer used.
    for &i in group {
        let s = &shapes[i];
        let toward: [f32; 3] = std::array::from_fn(|c| fg[c] - ground(i, c));
        let norm = (0..3)
            .map(|c| weights[c] * toward[c] * toward[c])
            .sum::<f32>()
            .max(1e-12);
        let occurrence = &mut occurrences[i];
        occurrence.fg = fg;
        for (dy, row) in occurrence.alpha.chunks_exact_mut(s.w).enumerate() {
            let start = (s.y0 + dy) * lattice.width + s.x0;
            for (dx, a) in row.iter_mut().enumerate() {
                let idx = start + dx;
                if label[idx] == s.label {
                    let d = pixel_delta(&lattice.planes, idx, s.bg);
                    let coverage =
                        (0..3).map(|c| weights[c] * d[c] * toward[c]).sum::<f32>() / norm;
                    *a = coverage.clamp(0.0, 1.0);
                }
            }
        }
    }
}

#[inline]
fn pixel_delta(fine: &[Vec<i32>; 3], idx: usize, bg: [i32; 3]) -> [f32; 3] {
    std::array::from_fn(|c| (fine[c][idx] - bg[c]) as f32 / TEMPLATE_LATTICE[c])
}

/// Hashes the glyph mask and estimates the foreground from its strongest pixels.
fn estimate_foreground(
    lattice: &Lattice,
    label: &[u32],
    s: &GlyphShape,
    strength: &mut Vec<(f32, usize)>,
) -> (u64, Occurrence) {
    let weights = TEMPLATE_CHANNEL_WEIGHTS;
    strength.clear();
    let mut mask_hash = 0x9e37_79b9_7f4a_7c15 ^ ((s.w as u64) << 32 | s.h as u64);
    for dy in 0..s.h {
        let start = (s.y0 + dy) * lattice.width + s.x0;
        for (dx, &owner) in label[start..][..s.w].iter().enumerate() {
            let inside = owner == s.label;
            mask_hash = (mask_hash ^ u64::from(inside)).wrapping_mul(0xff51_afd7_ed55_8ccd);
            mask_hash ^= mask_hash >> 29;
            if inside {
                let idx = start + dx;
                let d = pixel_delta(&lattice.planes, idx, s.bg);
                strength.push(((0..3).map(|c| weights[c] * d[c] * d[c]).sum(), idx));
            }
        }
    }
    let keep = (strength.len() / 5).max(1);
    // Keep the row-major order of tied pixels from the full stable sort.
    let by_strength =
        |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
    strength.select_nth_unstable_by(keep - 1, by_strength);
    strength[..keep - 1].shared_sort_unstable_by(by_strength);
    let mut fd = [0.0f32; 3];
    for &(_, idx) in &strength[..keep] {
        let d = pixel_delta(&lattice.planes, idx, s.bg);
        for c in 0..3 {
            fd[c] += d[c] / keep as f32;
        }
    }
    let fg = std::array::from_fn(|c| s.bg[c] as f32 / TEMPLATE_LATTICE[c] + fd[c]);
    (
        mask_hash,
        Occurrence {
            fg,
            fg_delta: fd,
            alpha: Vec::new(),
            relative: Vec::new(),
        },
    )
}

/// Groups glyphs by template. `label` holds the component of every pixel.
/// `None` when no group shares a template.
fn plan_glyph_templates<'a>(
    ctx: &EncodingContext,
    lattice: &Lattice,
    label: &[u32],
    shapes: &'a [GlyphShape],
) -> Option<TemplatePlan<'a>> {
    let width = lattice.width;
    let fine = &lattice.planes;
    let weights = TEMPLATE_CHANNEL_WEIGHTS;
    let mut strength = Vec::new();
    let mut occurrences = Vec::with_capacity(shapes.len());
    let mut buckets: HashMap<(u64, [i32; 3]), Vec<usize>> = HashMap::new();
    for (i, s) in shapes.iter().enumerate() {
        let (mask_hash, occurrence) = estimate_foreground(lattice, label, s, &mut strength);
        // On smooth ground neither the mask nor the foreground estimate of
        // an antialiased glyph survives a change of background (faint edge
        // pixels come and go, and the strongest pixels are a mix that leans
        // on the ground): such shapes meet by size and polarity, and their
        // coverage decides.
        let (bucket, key) = if s.soft {
            let polarity = if occurrence.fg_delta[0] >= 0.0 { 1 } else { -1 };
            (
                0x5bd1_e995_u64 ^ ((s.w as u64) << 32 | s.h as u64),
                [polarity, i32::MAX, i32::MAX],
            )
        } else {
            (
                mask_hash,
                std::array::from_fn(|c| {
                    (occurrence.fg[c] * TEMPLATE_LATTICE[c] / TEMPLATE_COLOR_STEP[c]).round() as i32
                }),
            )
        };
        occurrences.push(occurrence);
        buckets.entry((bucket, key)).or_default().push(i);
    }
    drop(strength);
    // A member fits when its coverage agrees with the template and its
    // cores, which the decoder shows as the template's mix, stay close.
    let fits = |s: &GlyphShape, coverage: &[f32], alpha: &[f32], fg: &[f32; 3]| -> bool {
        let max_diff = if s.soft {
            TEMPLATE_MAX_SOFT_ALPHA_DIFF
        } else {
            TEMPLATE_MAX_ALPHA_DIFF
        };
        let GlyphShape {
            x0, y0, w, h, bg, ..
        } = *s;
        let (mut diff, mut count) = (0.0f32, 0usize);
        for (t, a) in alpha.iter().zip(coverage) {
            if *t > 0.0 || *a > 0.0 {
                diff += (t - a).abs();
                count += 1;
            }
        }
        let core_tolerance = if s.soft {
            &TEMPLATE_SOFT_CORE_TOLERANCE
        } else {
            &TEMPLATE_CORE_TOLERANCE
        };
        if count > 0 && diff > max_diff * count as f32 {
            return false;
        }
        for dy in 0..h {
            for dx in 0..w {
                let a = alpha[dy * w + dx];
                if a <= 0.0 || 1.0 - a >= TEMPLATE_MIN_BASE_WEIGHT {
                    continue;
                }
                let idx = (y0 + dy) * width + x0 + dx;
                for c in 0..3 {
                    let mix = bg[c] as f32 * (1.0 - a) + fg[c] * TEMPLATE_LATTICE[c] * a;
                    if (fine[c][idx] as f32 - mix).abs() > core_tolerance[c] {
                        return false;
                    }
                }
            }
        }
        true
    };
    // Shapes with one support and color may still differ inside (coverage):
    // split each bucket into clusters whose coverage agrees with their seed.
    let agrees = |a: &[f32], b: &[f32], max_diff: f32| -> bool {
        let (mut diff, mut count) = (0.0f32, 0usize);
        for (x, y) in a.iter().zip(b) {
            if *x > 0.0 || *y > 0.0 {
                diff += (x - y).abs();
                count += 1;
            }
        }
        diff <= max_diff * count.max(1) as f32
    };
    let has_multiple_backgrounds = |group: &[usize]| {
        group
            .split_first()
            .is_some_and(|(&first, rest)| rest.iter().any(|&i| shapes[i].bg != shapes[first].bg))
    };
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut groups = Vec::new();
    let mut template_alpha = Vec::new();
    let mut spans: Vec<(Vec<usize>, u32)> = Vec::new();
    for members in buckets
        .into_values()
        .filter(|g| has_multiple_backgrounds(g))
    {
        for i in members {
            let s = &shapes[i];
            let fd = occurrences[i].fg_delta;
            let norm = (0..3)
                .map(|c| weights[c] * fd[c] * fd[c])
                .sum::<f32>()
                .max(1e-12);
            let mut alpha = vec![0.0f32; s.w * s.h];
            for (dy, row) in alpha.chunks_exact_mut(s.w).enumerate() {
                let start = (s.y0 + dy) * width + s.x0;
                for (dx, a) in row.iter_mut().enumerate() {
                    let idx = start + dx;
                    if label[idx] == s.label {
                        let d = pixel_delta(fine, idx, s.bg);
                        let coverage =
                            (0..3).map(|c| weights[c] * d[c] * fd[c]).sum::<f32>() / norm;
                        *a = coverage.clamp(0.0, 1.0);
                    }
                }
            }
            let max_diff = if s.soft {
                TEMPLATE_MAX_SOFT_ALPHA_DIFF
            } else {
                TEMPLATE_MAX_ALPHA_DIFF
            };
            match clusters
                .iter_mut()
                .find(|c| agrees(&occurrences[c[0]].alpha, &alpha, max_diff))
            {
                Some(cluster) => cluster.push(i),
                None => clusters.push(vec![i]),
            }
            occurrences[i].alpha = alpha;
        }
        for mut group in clusters.drain(..) {
            if shapes[group[0]].soft {
                for &i in &group {
                    let occurrence = &mut occurrences[i];
                    occurrence.relative = std::mem::take(&mut occurrence.alpha);
                }
                // Ordered by ground, so that a group one template cannot
                // hold is cut where the ground changes.
                group.shared_sort_by_key(|&i| (shapes[i].bg, shapes[i].y0, shapes[i].x0));
                spans.push((group, 0));
                while let Some((span, depth)) = spans.pop() {
                    if !has_multiple_backgrounds(&span) {
                        continue;
                    }
                    share_foreground(lattice, label, shapes, &mut occurrences, &span);
                    let mut kept = span.clone();
                    for _ in 0..2 {
                        if !has_multiple_backgrounds(&kept) {
                            break;
                        }
                        let fg = template_of(ctx, shapes, &occurrences, &kept, &mut template_alpha);
                        kept.retain(|&i| {
                            fits(&shapes[i], &occurrences[i].alpha, &template_alpha, &fg)
                        });
                    }
                    let enough =
                        kept.len() * TEMPLATE_SPAN_KEEP.1 >= span.len() * TEMPLATE_SPAN_KEEP.0;
                    if (enough || depth == TEMPLATE_MAX_SPAN_DEPTH || span.len() < 4)
                        && has_multiple_backgrounds(&kept)
                    {
                        for &i in &kept {
                            occurrences[i].relative = Vec::new();
                        }
                        groups.push(kept);
                    } else if depth < TEMPLATE_MAX_SPAN_DEPTH && span.len() >= 4 {
                        let (low, high) = span.split_at(span.len() / 2);
                        spans.push((high.to_vec(), depth + 1));
                        spans.push((low.to_vec(), depth + 1));
                    }
                }
                continue;
            }
            for _ in 0..2 {
                if !has_multiple_backgrounds(&group) {
                    break;
                }
                let fg = template_of(ctx, shapes, &occurrences, &group, &mut template_alpha);
                group.retain(|&i| {
                    let accepted = fits(&shapes[i], &occurrences[i].alpha, &template_alpha, &fg);
                    if !accepted {
                        occurrences[i].alpha = Vec::new();
                    }
                    accepted
                });
            }
            if has_multiple_backgrounds(&group) {
                groups.push(group);
            } else {
                for i in group {
                    occurrences[i].alpha = Vec::new();
                }
            }
        }
    }
    // Soft members no group kept.
    let mut grouped = vec![false; shapes.len()];
    for &i in groups.iter().flatten() {
        grouped[i] = true;
    }
    for (occurrence, &kept) in occurrences.iter_mut().zip(&grouped) {
        if !kept {
            occurrence.alpha = Vec::new();
        }
        occurrence.relative = Vec::new();
    }
    (!groups.is_empty()).then_some(TemplatePlan {
        shapes,
        occurrences,
        groups,
    })
}

/// The template atlas (color tiles, coverage in alpha), placements and the
/// base with the template glyphs un-premultiplied out of it.
fn build_glyph_templates(
    ctx: &EncodingContext,
    plan: TemplatePlan<'_>,
    xyb: &Image3F,
) -> Option<GlyphTemplates> {
    let lattice = TEMPLATE_LATTICE;
    let image_plane = [1usize, 0, 2];
    let TemplatePlan {
        shapes,
        occurrences,
        mut groups,
    } = plan;
    groups.shared_sort_by_key(|g| {
        let GlyphShape { x0, y0, w, h, .. } = shapes[g[0]];
        (
            std::cmp::Reverse(h),
            std::cmp::Reverse(g.len() * w * h),
            y0,
            x0,
        )
    });
    let budget = GLYPH_ATLAS_MAX_WIDTH * GLYPH_ATLAS_MAX_HEIGHT / 4;
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
    let total_area: usize = groups.iter().map(|g| shapes[g[0]].w * shapes[g[0]].h).sum();
    let max_width = groups.iter().map(|g| shapes[g[0]].w).max().unwrap_or(16);
    let atlas_width = ((total_area as f64).sqrt() as usize * 3 / 2)
        .max(max_width)
        .clamp(16, GLYPH_ATLAS_MAX_WIDTH);
    let mut slots = Vec::with_capacity(groups.len());
    let (mut cx, mut cy, mut shelf_h) = (0usize, 0usize, 0usize);
    for g in &groups {
        let GlyphShape { w, h, .. } = shapes[g[0]];
        if cx + w > atlas_width {
            cx = 0;
            cy += shelf_h;
            shelf_h = 0;
        }
        slots.push((cx, cy));
        cx += w;
        shelf_h = shelf_h.max(h);
    }
    let atlas_height = cy + shelf_h;
    if atlas_height > GLYPH_ATLAS_MAX_HEIGHT / 2 {
        return None;
    }
    let pixels = atlas_width * atlas_height;
    let mut atlas = [vec![0i32; pixels], vec![0i32; pixels], vec![0i32; pixels]];
    let mut atlas_alpha = vec![0u8; pixels];
    let mut base = xyb.clone();
    let mut references = Vec::with_capacity(groups.len());
    let mut coverage = Vec::new();
    for (g, (ax, ay)) in groups.iter().zip(slots) {
        let GlyphShape { w, h, .. } = shapes[g[0]];
        let fg = template_of(ctx, shapes, &occurrences, g, &mut coverage);
        // The tile: the foreground color everywhere, the coverage in alpha.
        let color: [i32; 3] = std::array::from_fn(|c| (fg[c] * lattice[c]).round() as i32);
        let decoded_color: [f32; 3] = std::array::from_fn(|c| color[c] as f32 / lattice[c]);
        for (dy, row) in coverage.chunks_exact_mut(w).enumerate() {
            let start = (ay + dy) * atlas_width + ax;
            atlas[0][start..][..w].fill(color[0]);
            atlas[1][start..][..w].fill(color[1]);
            atlas[2][start..][..w].fill(color[2] - color[0]);
            for (a, sample) in row.iter_mut().zip(&mut atlas_alpha[start..][..w]) {
                *sample = (*a * 255.0).fast_round().clamp(0.0, 255.0) as u8;
                *a = f32::from(*sample) / 255.0;
            }
        }
        let mut positions = Vec::with_capacity(g.len());
        for &i in g {
            let GlyphShape { x0, y0, bg, .. } = shapes[i];
            positions.push((x0, y0));
            for c in 0..3 {
                let flat = bg[c] as f32 / lattice[c];
                for dy in 0..h {
                    let row = base.plane_row_mut(image_plane[c], y0 + dy);
                    for dx in 0..w {
                        let a = coverage[dy * w + dx];
                        if a <= 0.0 {
                            continue;
                        }
                        let keep = 1.0 - a;
                        let v = &mut row[x0 + dx];
                        *v = if keep >= TEMPLATE_MIN_BASE_WEIGHT {
                            (*v - decoded_color[c] * a) / keep
                        } else {
                            flat
                        };
                    }
                }
            }
        }
        positions.shared_sort_unstable_by_key(|&(x, y)| (y, x));
        references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: w,
            height: h,
            ref_frame: MODULAR_PATCH_REF_ID,
            alpha_only: false,
            add: false,
            mul: true,
            positions,
        });
    }
    Some(GlyphTemplates {
        atlas,
        alpha: atlas_alpha,
        width: atlas_width,
        height: atlas_height,
        base,
        references,
    })
}

/// Moves the glyphs that share a blend-above template out of `shapes` and
/// returns their atlas. With a template probe in the context the plan is
/// measured even when templates are off.
pub(super) fn take_glyph_templates(
    ctx: &EncodingContext,
    params: &GlyphParams,
    xyb: &Image3F,
    lattice: &Lattice,
    label: &[u32],
    shapes: &mut Vec<GlyphShape>,
) -> Option<GlyphTemplates> {
    if ctx.template_probe.is_none() && !params.templates {
        return None;
    }
    // The tiers that race templates for glyphs over gradients alone ask
    // nothing of a frame without smooth ground.
    if !params.templates && ctx.template_ground_only && !shapes.iter().any(|s| s.soft) {
        return None;
    }
    let plan = plan_glyph_templates(ctx, lattice, label, shapes);
    if let (Some(probe), Some(plan)) = (&ctx.template_probe, &plan) {
        probe.fetch_max(plan.saved_area(), std::sync::atomic::Ordering::Relaxed);
    }
    let templates = plan
        .filter(|_| params.templates)
        .and_then(|plan| build_glyph_templates(ctx, plan, xyb))?;
    let used: std::collections::HashSet<(usize, usize)> = templates
        .references
        .iter()
        .flat_map(|r| r.positions.iter().copied())
        .collect();
    shapes.retain(|s| !used.contains(&(s.x0, s.y0)));
    Some(templates)
}
