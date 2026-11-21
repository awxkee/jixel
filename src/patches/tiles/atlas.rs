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
use super::hash_tile;
use crate::image::Image3F;
use crate::patches::{PATCH_TILE, PatchReference};
use crate::shared_sort::SharedSort;
use std::collections::HashMap;

/// Pack a set of groups into a fresh atlas image whose dictionary entries all
/// name `ref_frame`. With at most 256 groups the atlas never exceeds 256x256,
/// i.e. a single 256-pixel modular group.
pub(crate) fn pack_lossy_atlas(
    linear: &Image3F,
    groups: Vec<Vec<(usize, usize)>>,
    ref_frame: u32,
) -> (Image3F, Vec<PatchReference>) {
    pack_lossy_atlas_sized(linear, groups, PATCH_TILE, ref_frame)
}

/// Put `bottom` under `top` in one atlas; `bottom_refs` move down with it.
pub(crate) fn stack_atlases(
    top: Image3F,
    bottom: Image3F,
    bottom_refs: &mut [PatchReference],
) -> Image3F {
    // Rows of whole 8x8 blocks keep either half off the other's blocks.
    let top_h = top.ysize().next_multiple_of(16);
    let width = top.xsize().max(bottom.xsize());
    let mut atlas = Image3F::new(width, top_h + bottom.ysize());
    for c in 0..3 {
        for y in 0..top.ysize() {
            atlas.plane_row_mut(c, y)[..top.xsize()]
                .copy_from_slice(&top.plane_row(c, y)[..top.xsize()]);
        }
        for y in 0..bottom.ysize() {
            atlas.plane_row_mut(c, top_h + y)[..bottom.xsize()]
                .copy_from_slice(&bottom.plane_row(c, y)[..bottom.xsize()]);
        }
    }
    for r in bottom_refs {
        r.atlas_y += top_h;
    }
    atlas
}

/// 8x8 groups whose tile equals an aligned quadrant of an already packed
/// 16x16 tile need no atlas pixels: they become entries pointing into that
/// tile. Returns those entries and the groups that still need packing.
/// `image` must hold both the 16x16 sources and the 8x8 tiles untouched.
pub(crate) fn reuse_tile_quadrants(
    image: &Image3F,
    packed: &[PatchReference],
    groups: Vec<Vec<(usize, usize)>>,
) -> (Vec<PatchReference>, Vec<Vec<(usize, usize)>>) {
    const HALF: usize = PATCH_TILE / 2;
    let same = |a: (usize, usize), b: (usize, usize)| {
        (0..3).all(|c| {
            (0..HALF).all(|dy| {
                image.plane_row(c, a.1 + dy)[a.0..a.0 + HALF]
                    == image.plane_row(c, b.1 + dy)[b.0..b.0 + HALF]
            })
        })
    };
    struct Quadrant {
        /// Index into `packed` and the quadrant's offset inside that tile.
        tile: usize,
        offset: (usize, usize),
        source: (usize, usize),
    }
    let mut quadrants: HashMap<u64, Vec<Quadrant>> = HashMap::new();
    for (index, reference) in packed.iter().enumerate() {
        if reference.add || reference.width != PATCH_TILE || reference.height != PATCH_TILE {
            continue;
        }
        let (sx, sy) = reference.positions[0];
        for (qx, qy) in [(0, 0), (HALF, 0), (0, HALF), (HALF, HALF)] {
            let source = (sx + qx, sy + qy);
            quadrants
                .entry(hash_tile(image, source.0, source.1, HALF))
                .or_default()
                .push(Quadrant {
                    tile: index,
                    offset: (qx, qy),
                    source,
                });
        }
    }
    let mut reused = Vec::new();
    let mut remaining = Vec::with_capacity(groups.len());
    for positions in groups {
        let first = positions[0];
        let found = quadrants
            .get(&hash_tile(image, first.0, first.1, HALF))
            .and_then(|candidates| candidates.iter().find(|q| same(q.source, first)));
        match found {
            Some(q) => reused.push(PatchReference {
                atlas_x: packed[q.tile].atlas_x + q.offset.0,
                atlas_y: packed[q.tile].atlas_y + q.offset.1,
                width: HALF,
                height: HALF,
                ref_frame: packed[q.tile].ref_frame,
                alpha_only: false,
                add: false,
                mul: false,
                positions,
            }),
            None => remaining.push(positions),
        }
    }
    (reused, remaining)
}

/// Visit order that keeps look-alike tiles next to each other in the atlas:
/// a greedy nearest-neighbor chain over each tile's mean X, Y, B.
pub(super) fn similarity_order(
    linear: &Image3F,
    groups: &[Vec<(usize, usize)>],
    tile: usize,
) -> Vec<usize> {
    let means: Vec<[f32; 3]> = groups
        .iter()
        .map(|g| {
            let (x0, y0) = g[0];
            std::array::from_fn(|c| {
                (0..tile)
                    .map(|dy| {
                        linear.plane_row(c, y0 + dy)[x0..x0 + tile]
                            .iter()
                            .sum::<f32>()
                    })
                    .sum::<f32>()
                    / (tile * tile) as f32
            })
        })
        .collect();
    if means.len() > 256 {
        return morton_similarity_order(&means);
    }
    greedy_similarity_order(&means)
}

pub(super) fn greedy_similarity_order(means: &[[f32; 3]]) -> Vec<usize> {
    // X spans a fraction of the Y and B ranges.
    let distance = |a: &[f32; 3], b: &[f32; 3]| {
        let d = [(a[0] - b[0]) * 8.0, a[1] - b[1], a[2] - b[2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    let mut left: Vec<usize> = (1..means.len()).collect();
    let mut order = Vec::with_capacity(means.len());
    let mut current = 0;
    order.push(current);
    while !left.is_empty() {
        let (slot, _) = left
            .iter()
            .enumerate()
            .map(|(slot, &i)| (slot, distance(&means[current], &means[i])))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty");
        current = left.swap_remove(slot);
        order.push(current);
    }
    order
}

/// A spatial sort followed by a bounded local nearest-neighbor chain.
/// Linked neighbors keep removals O(1); each step compares at most 64 points.
pub(super) fn morton_similarity_order(means: &[[f32; 3]]) -> Vec<usize> {
    let n = means.len();
    if n == 0 {
        return Vec::new();
    }
    let scaled: Vec<[f32; 3]> = means.iter().map(|m| [m[0] * 8.0, m[1], m[2]]).collect();
    let lo: [f32; 3] =
        std::array::from_fn(|c| scaled.iter().map(|m| m[c]).fold(f32::INFINITY, f32::min));
    let hi: [f32; 3] = std::array::from_fn(|c| {
        scaled
            .iter()
            .map(|m| m[c])
            .fold(f32::NEG_INFINITY, f32::max)
    });
    let span = (0..3)
        .map(|c| hi[c] - lo[c])
        .fold(0.0f32, f32::max)
        .max(1e-20);
    let key = |i: usize| {
        let q: [u32; 3] =
            std::array::from_fn(|c| (((scaled[i][c] - lo[c]) / span * 1023.0) as u32).min(1023));
        let mut result = 0;
        for bit in 0..10 {
            for c in 0..3 {
                result |= ((q[c] >> bit) & 1) << (3 * bit + c);
            }
        }
        result
    };
    let mut spatial: Vec<usize> = (0..n).collect();
    spatial.shared_sort_unstable_by_key(|&i| (key(i), i));
    let mut slot = vec![0; n];
    for (s, &i) in spatial.iter().enumerate() {
        slot[i] = s;
    }
    let mut prev: Vec<usize> = (0..n).map(|s| if s == 0 { n } else { s - 1 }).collect();
    let mut next: Vec<usize> = (0..n).map(|s| s + 1).collect();
    let mut current = 0;
    let mut order = Vec::with_capacity(n);
    while order.len() < n {
        order.push(current);
        let s = slot[current];
        let (mut left, mut right) = (prev[s], next[s]);
        if left != n {
            next[left] = right;
        }
        if right != n {
            prev[right] = left;
        }
        if order.len() == n {
            break;
        }
        let mut best = (f32::INFINITY, usize::MAX);
        for _ in 0..32 {
            for candidate in [left, right] {
                if candidate == n {
                    continue;
                }
                let id = spatial[candidate];
                let a = means[current];
                let b = means[id];
                let d = [(a[0] - b[0]) * 8.0, a[1] - b[1], a[2] - b[2]];
                let cost = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if cost.total_cmp(&best.0).then(id.cmp(&best.1)).is_lt() {
                    best = (cost, id);
                }
            }
            if left != n {
                left = prev[left];
            }
            if right != n {
                right = next[right];
            }
        }
        current = best.1;
    }
    order
}

pub(crate) fn pack_lossy_atlas_sized(
    linear: &Image3F,
    groups: Vec<Vec<(usize, usize)>>,
    tile: usize,
    ref_frame: u32,
) -> (Image3F, Vec<PatchReference>) {
    let mut groups = groups;
    if groups.len() > 2 {
        let order = similarity_order(linear, &groups, tile);
        let mut taken: Vec<Option<Vec<(usize, usize)>>> = groups.into_iter().map(Some).collect();
        groups = order
            .into_iter()
            .map(|i| taken[i].take().expect("once"))
            .collect();
    }
    let atlas_cols = groups.len().min(256 / tile);
    let atlas_rows = groups.len().div_ceil(atlas_cols);
    let mut atlas = Image3F::new(atlas_cols * tile, atlas_rows * tile);
    let mut references = Vec::with_capacity(groups.len());
    for (i, positions) in groups.into_iter().enumerate() {
        let atlas_x = (i % atlas_cols) * tile;
        let atlas_y = (i / atlas_cols) * tile;
        let (sx, sy) = positions[0];
        for c in 0..3 {
            for dy in 0..tile {
                atlas.plane_row_mut(c, atlas_y + dy)[atlas_x..atlas_x + tile]
                    .copy_from_slice(&linear.plane_row(c, sy + dy)[sx..sx + tile]);
            }
        }
        references.push(PatchReference {
            width: tile,
            height: tile,
            atlas_x,
            atlas_y,
            ref_frame,
            alpha_only: false,
            add: false,
            mul: false,
            positions,
        });
    }
    (atlas, references)
}
