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

//! Exact repeated coverage masks, independent of the image's color patches.
//!
//! A plan is only a representation candidate. Its atlas, dictionary and base
//! must be encoded and compared with the ordinary complete frame before use.

use crate::encode_image::AlphaPlane;
use crate::patches::{MODULAR_PATCH_REF_ID, PatchReference};
use std::cmp::Ordering;
use std::collections::HashMap;

const TILE: usize = 32;
const TILE_PIXELS: usize = TILE * TILE;
const ATLAS_COLUMNS: usize = 8;
const MAX_ATLAS_TILES: usize = 256;
const MAX_SCAN_TILES: usize = 65_536;
const MAX_OCCURRENCES: usize = 16_384;
const MAX_HASH_COLLISIONS: usize = 8;
const MIN_OCCURRENCES: usize = 32;
const MIN_DUPLICATES: usize = 16;

pub(crate) struct AlphaPatchPlan {
    pub(crate) atlas_alpha: AlphaPlane,
    pub(crate) atlas_width: usize,
    pub(crate) atlas_height: usize,
    pub(crate) base_alpha: AlphaPlane,
    pub(crate) references: Vec<PatchReference>,
}

struct TileGroup {
    first: (usize, usize),
    /// Additional matches, allocated only after the second occurrence.
    copies: Vec<(usize, usize)>,
    /// Another representative with the same hash, not necessarily equal data.
    next: Option<usize>,
}

/// A cheap screen and exact tile finder for an 8-bit coverage plane. Constant
/// tiles are left to the ordinary predictor/LZ coder. Small matches and plans
/// covering less than an eighth of the image do not justify a second encode.
///
/// All retained locations are on the same nonoverlapping 32-pixel grid. Larger
/// images or unusually large dictionaries simply keep the ordinary encoding.
pub(crate) fn find_alpha_patches(
    alpha: &AlphaPlane,
    width: usize,
    height: usize,
) -> Option<AlphaPatchPlan> {
    find_with_signature(alpha, width, height, tile_signature)
}

fn find_with_signature(
    alpha: &AlphaPlane,
    width: usize,
    height: usize,
    signature: impl Fn(&[u8], usize, usize, usize) -> Option<u64>,
) -> Option<AlphaPatchPlan> {
    let AlphaPlane::U8(data) = alpha else {
        return None;
    };
    let pixels = width.checked_mul(height)?;
    if pixels != data.len() {
        return None;
    }
    let tiles_x = width / TILE;
    let tiles_y = height / TILE;
    let tiles = tiles_x.checked_mul(tiles_y)?;
    if !(MIN_OCCURRENCES..=MAX_SCAN_TILES).contains(&tiles) {
        return None;
    }

    let mut heads = HashMap::<u64, usize>::with_capacity(tiles.min(4096));
    let mut groups = Vec::<TileGroup>::new();
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let (x, y) = (tx * TILE, ty * TILE);
            let Some(hash) = signature(data, width, x, y) else {
                continue;
            };
            let head = heads.get(&hash).copied();
            let mut current = head;
            let mut collisions = 0;
            let mut found = false;
            while let Some(index) = current {
                let group = &mut groups[index];
                if tile_eq(data, width, group.first, (x, y)) {
                    // Further occurrences beyond the complete plan's budget
                    // need not be stored. They remain ordinary base samples.
                    if group.copies.len() + 1 < MAX_OCCURRENCES {
                        group.copies.push((x, y));
                    }
                    found = true;
                    break;
                }
                collisions += 1;
                current = group.next;
            }
            if !found && collisions < MAX_HASH_COLLISIONS {
                let index = groups.len();
                groups.push(TileGroup {
                    first: (x, y),
                    copies: Vec::new(),
                    next: head,
                });
                heads.insert(hash, index);
            }
        }
    }

    let mut selected: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, group)| (!group.copies.is_empty()).then_some(i))
        .collect();
    // Spend the bounded dictionary on the most reused masks. The data tie
    // break makes selection independent of HashMap's randomized iteration.
    selected.sort_unstable_by(|&a, &b| {
        groups[b]
            .copies
            .len()
            .cmp(&groups[a].copies.len())
            .then_with(|| tile_cmp(data, width, groups[a].first, groups[b].first))
    });
    selected.truncate(MAX_ATLAS_TILES);
    let occurrences = bound_occurrences(&mut groups, &mut selected, MAX_OCCURRENCES);
    if occurrences < MIN_OCCURRENCES
        || occurrences.saturating_sub(selected.len()) < MIN_DUPLICATES
        || occurrences * TILE_PIXELS < pixels.div_ceil(8)
    {
        return None;
    }

    // Lexicographic atlas order matches the one-channel palette convention
    // and keeps neighboring coverage patterns spatially predictable.
    selected.sort_unstable_by(|&a, &b| tile_cmp(data, width, groups[a].first, groups[b].first));
    let columns = selected.len().min(ATLAS_COLUMNS);
    let atlas_width = columns * TILE;
    let atlas_height = selected.len().div_ceil(columns) * TILE;
    let mut atlas = vec![0u8; atlas_width * atlas_height];
    let mut base = data.clone();
    let mut references = Vec::with_capacity(selected.len());
    for (slot, index) in selected.into_iter().enumerate() {
        let group = &mut groups[index];
        let (sx, sy) = group.first;
        let (ax, ay) = ((slot % columns) * TILE, (slot / columns) * TILE);
        let mut sum = 0u32;
        for row in 0..TILE {
            let input = &data[(sy + row) * width + sx..][..TILE];
            atlas[(ay + row) * atlas_width + ax..][..TILE].copy_from_slice(input);
            sum += input.iter().map(|&v| u32::from(v)).sum::<u32>();
        }
        let mean = rounded_mean(sum);
        let mut positions = Vec::with_capacity(group.copies.len() + 1);
        positions.push(group.first);
        positions.append(&mut group.copies);
        for &(x, y) in &positions {
            for row in 0..TILE {
                base[(y + row) * width + x..][..TILE].fill(mean);
            }
        }
        references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: TILE,
            height: TILE,
            ref_frame: MODULAR_PATCH_REF_ID,
            alpha_only: true,
            add: false,
            mul: false,
            positions,
        });
    }
    Some(AlphaPatchPlan {
        atlas_alpha: AlphaPlane::U8(atlas),
        atlas_width,
        atlas_height,
        base_alpha: AlphaPlane::U8(base),
        references,
    })
}

fn bound_occurrences(groups: &mut [TileGroup], selected: &mut Vec<usize>, budget: usize) -> usize {
    let mut remaining = budget;
    selected.retain(|&index| {
        if remaining < 2 {
            return false;
        }
        let count = (groups[index].copies.len() + 1).min(remaining);
        groups[index].copies.truncate(count - 1);
        remaining -= count;
        true
    });
    budget - remaining
}

/// A fixed word hash; equality is always checked on the original samples.
fn tile_signature(data: &[u8], width: usize, x: usize, y: usize) -> Option<u64> {
    let repeated = u64::from_le_bytes([data[y * width + x]; 8]);
    let mut varied = false;
    let mut hash = 0x9e37_79b9_7f4a_7c15u64;
    for row in 0..TILE {
        for bytes in data[(y + row) * width + x..][..TILE].as_chunks::<8>().0 {
            let word = u64::from_le_bytes(*bytes);
            varied |= word != repeated;
            hash = (hash ^ word).wrapping_mul(0xff51_afd7_ed55_8ccd);
            hash ^= hash >> 32;
        }
    }
    varied.then_some(hash)
}

fn tile_eq(data: &[u8], width: usize, a: (usize, usize), b: (usize, usize)) -> bool {
    tile_cmp(data, width, a, b) == Ordering::Equal
}

fn tile_cmp(data: &[u8], width: usize, a: (usize, usize), b: (usize, usize)) -> Ordering {
    for row in 0..TILE {
        let left = &data[(a.1 + row) * width + a.0..][..TILE];
        let right = &data[(b.1 + row) * width + b.0..][..TILE];
        let order = left.cmp(right);
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

/// Ties-to-even agrees with a rounded arithmetic-mean coverage placeholder.
fn rounded_mean(sum: u32) -> u8 {
    let quotient = sum / TILE_PIXELS as u32;
    let remainder = sum % TILE_PIXELS as u32;
    let round_up = remainder * 2 > TILE_PIXELS as u32
        || (remainder * 2 == TILE_PIXELS as u32 && !quotient.is_multiple_of(2));
    (quotient + u32::from(round_up)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated_image(width: usize, height: usize) -> Vec<u8> {
        let mut alpha = vec![251; width * height];
        for ty in 0..height / TILE {
            for tx in 0..width / TILE {
                for y in 0..TILE {
                    for x in 0..TILE {
                        alpha[(ty * TILE + y) * width + tx * TILE + x] =
                            ((x * 5 + y * 3 + (tx % 2) * 17) % 256) as u8;
                    }
                }
            }
        }
        alpha
    }

    fn reconstruct(plan: &AlphaPatchPlan, width: usize) -> Vec<u8> {
        let AlphaPlane::U8(mut base) = plan.base_alpha.clone() else {
            unreachable!()
        };
        let AlphaPlane::U8(atlas) = &plan.atlas_alpha else {
            unreachable!()
        };
        for reference in &plan.references {
            assert!(reference.alpha_only);
            assert!(!reference.add && !reference.mul);
            for &(x, y) in &reference.positions {
                for row in 0..reference.height {
                    base[(y + row) * width + x..][..reference.width].copy_from_slice(
                        &atlas[(reference.atlas_y + row) * plan.atlas_width + reference.atlas_x..]
                            [..reference.width],
                    );
                }
            }
        }
        base
    }

    #[test]
    fn repeated_coverage_reconstructs_exactly_with_odd_edges() {
        let (width, height) = (257, 259);
        let data = repeated_image(width, height);
        let plan = find_alpha_patches(&AlphaPlane::U8(data.clone()), width, height).unwrap();
        assert_eq!(plan.references.len(), 2);
        assert_eq!(plan.atlas_width, 64);
        assert_eq!(plan.atlas_height, TILE);
        assert_eq!(reconstruct(&plan, width), data);
        let AlphaPlane::U8(base) = &plan.base_alpha else {
            unreachable!()
        };
        assert_eq!(base[256], 251);
        assert_eq!(base[(height - 1) * width], 251);
    }

    #[test]
    fn constant_unique_sparse_and_unsupported_planes_skip() {
        let (width, height) = (256, 256);
        assert!(
            find_alpha_patches(&AlphaPlane::U8(vec![255; width * height]), width, height).is_none()
        );
        assert!(
            find_alpha_patches(
                &AlphaPlane::U16 {
                    data: vec![1; width * height],
                    bits: 16
                },
                width,
                height
            )
            .is_none()
        );
        assert!(
            find_alpha_patches(&AlphaPlane::F32(vec![0; width * height]), width, height).is_none()
        );
        assert!(
            find_alpha_patches(&AlphaPlane::U8(vec![0; width * height - 1]), width, height)
                .is_none()
        );
        let mut unique = repeated_image(width, height);
        for tile in 0..64 {
            unique[(tile / 8) * TILE * width + (tile % 8) * TILE] = tile as u8;
        }
        assert!(find_alpha_patches(&AlphaPlane::U8(unique), width, height).is_none());
        let mut sparse = vec![0; 1024 * 1024];
        let repeated = repeated_image(256, 128);
        for y in 0..128 {
            sparse[y * 1024..][..256].copy_from_slice(&repeated[y * 256..][..256]);
        }
        assert!(find_alpha_patches(&AlphaPlane::U8(sparse), 1024, 1024).is_none());
    }

    #[test]
    fn hash_collisions_do_not_merge_different_masks() {
        let data = repeated_image(256, 256);
        let alpha = AlphaPlane::U8(data.clone());
        let plan = find_with_signature(&alpha, 256, 256, |_, _, _, _| Some(0)).unwrap();
        assert_eq!(plan.references.len(), 2);
        assert_eq!(reconstruct(&plan, 256), data);
    }

    #[test]
    fn atlas_and_occurrence_budgets_remain_bounded() {
        // One of 257 distinct twice-repeated masks must remain ordinary base
        // samples so the atlas stays within its 256-entry bound.
        let (width, height) = (257 * TILE, 2 * TILE);
        let mut data = repeated_image(width, height);
        for ty in 0..2 {
            for tx in 0..257 {
                data[ty * TILE * width + tx * TILE] = tx as u8;
                data[ty * TILE * width + tx * TILE + 1] = (tx >> 8) as u8;
            }
        }
        let plan = find_alpha_patches(&AlphaPlane::U8(data.clone()), width, height).unwrap();
        assert_eq!(plan.references.len(), MAX_ATLAS_TILES);
        assert!(plan.atlas_width <= 256 && plan.atlas_height <= 1024);
        assert_eq!(
            plan.references
                .iter()
                .map(|r| r.positions.len())
                .sum::<usize>(),
            512
        );
        assert_eq!(reconstruct(&plan, width), data);

        let mut groups: Vec<_> = [4, 7, 3]
            .into_iter()
            .map(|count| TileGroup {
                first: (0, 0),
                copies: vec![(0, 0); count - 1],
                next: None,
            })
            .collect();
        let mut selected = vec![0, 1, 2];
        assert_eq!(bound_occurrences(&mut groups, &mut selected, 9), 9);
        assert_eq!(selected, [0, 1]);
        assert_eq!(groups[1].copies.len(), 4);
    }

    #[test]
    fn placeholders_round_half_to_even() {
        assert_eq!(rounded_mean(100 * TILE_PIXELS as u32 + 512), 100);
        assert_eq!(rounded_mean(101 * TILE_PIXELS as u32 + 512), 102);
        assert_eq!(rounded_mean(255 * TILE_PIXELS as u32), 255);
    }
}
