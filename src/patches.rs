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
use crate::coder_scratch::CoderScratch;
use crate::image::{Image3F, Image3Si};
use crate::thread_pool::ThreadPool;
use std::collections::HashMap;

pub(crate) struct LossyPatches {
    pub(crate) base: Image3F,
    /// Occurrence positions per group, first entry = the source tile. Packing
    /// into atlases is the caller's job so groups can be routed to more than
    /// one reference frame.
    pub(crate) groups: Vec<Vec<(usize, usize)>>,
}

#[derive(Clone, Copy)]
pub(crate) enum VarDctFrameKind<'a> {
    Regular,
    ReferenceOnly { width: usize, height: usize },
    Patched(&'a [PatchReference]),
}

/// Order-sensitive bulk hash of one tile.
fn hash_tile(linear: &Image3F, x0: usize, y0: usize) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in 0..3 {
        for y in y0..y0 + PATCH_TILE {
            let row = &linear.plane_row(c, y)[x0..x0 + PATCH_TILE];
            for pair in row.as_chunks::<2>().0 {
                let v = u64::from(pair[0].to_bits()) | (u64::from(pair[1].to_bits()) << 32);
                h = (h ^ v).wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 29;
            }
        }
    }
    h
}

/// Mean absolute deviation of a tile, summed over channels: a cheap stand-in
/// for what the tile costs to code as ordinary blocks.
fn tile_energy(img: &Image3F, x0: usize, y0: usize) -> f32 {
    let mut energy = 0.0;
    for c in 0..3 {
        let mut sum = 0.0;
        for y in y0..y0 + PATCH_TILE {
            sum += img.plane_row(c, y)[x0..x0 + PATCH_TILE].iter().sum::<f32>();
        }
        let mean = sum / (PATCH_TILE * PATCH_TILE) as f32;
        for y in y0..y0 + PATCH_TILE {
            energy += img.plane_row(c, y)[x0..x0 + PATCH_TILE]
                .iter()
                .map(|v| (v - mean).abs())
                .sum::<f32>();
        }
    }
    energy / (PATCH_TILE * PATCH_TILE) as f32
}

/// Minimum per-tile energy worth spending a patch on.
const MIN_PATCH_ENERGY: f32 = 0.017;

/// Minimum occurrences before a tile group becomes a patch. The study
/// preferred 5 to the historical 3 at every good configuration: rare groups
/// pay dictionary positions without amortizing their atlas tile.
const MIN_PATCH_OCCURRENCES: usize = 5;

pub(crate) fn find_lossy_patches(
    linear: &Image3F,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<LossyPatches> {
    let tile = PATCH_TILE;
    let width = linear.xsize();
    let height = linear.ysize();
    if width < tile || height < tile {
        return None;
    }

    let (tiles_x, tiles_y) = (width / tile, height / tile);
    // Tile rows are independent; the merge below is over tile counts, not
    // pixels, so it stays negligible.
    let rows = pool.steal_map(scratch, tiles_y, |ty, _scratch| {
        (0..tiles_x)
            .map(|tx| hash_tile(linear, tx * tile, ty * tile))
            .collect::<Vec<u64>>()
    });
    let mut buckets: HashMap<u64, Vec<(usize, usize)>> =
        HashMap::with_capacity(tiles_x.saturating_mul(tiles_y));
    for (ty, row) in rows.into_iter().enumerate() {
        for (tx, hash) in row.into_iter().enumerate() {
            buckets
                .entry(hash)
                .or_default()
                .push((tx * tile, ty * tile));
        }
    }

    let mut groups = Vec::new();
    for candidates in buckets.into_values() {
        if candidates.len() < MIN_PATCH_OCCURRENCES {
            continue;
        }
        let mut exact_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        for pos in candidates {
            let matching = exact_groups.iter().position(|group| {
                let first = group[0];
                (0..3).all(|c| {
                    (0..tile).all(|dy| {
                        linear.plane_row(c, first.1 + dy)[first.0..first.0 + tile]
                            == linear.plane_row(c, pos.1 + dy)[pos.0..pos.0 + tile]
                    })
                })
            });
            if let Some(i) = matching {
                exact_groups[i].push(pos);
            } else {
                exact_groups.push(vec![pos]);
            }
        }
        groups.extend(
            exact_groups
                .into_iter()
                .filter(|g| g.len() >= MIN_PATCH_OCCURRENCES),
        );
    }

    groups.retain(|g| tile_energy(linear, g[0].0, g[0].1) >= MIN_PATCH_ENERGY);
    groups.sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
    groups.truncate(256);
    if groups.is_empty() {
        return None;
    }

    let mut base = linear.clone();
    for positions in &groups {
        for &(x, y) in positions {
            for c in 0..3 {
                for dy in 0..tile {
                    base.plane_row_mut(c, y + dy)[x..x + tile].fill(0.0);
                }
            }
        }
    }
    Some(LossyPatches { base, groups })
}

/// Pack a set of groups into a fresh atlas image whose dictionary entries all
/// name `ref_frame`. With at most 256 groups the atlas never exceeds 256x256,
/// i.e. a single 256-pixel modular group.
pub(crate) fn pack_lossy_atlas(
    linear: &Image3F,
    groups: Vec<Vec<(usize, usize)>>,
    ref_frame: u32,
) -> (Image3F, Vec<PatchReference>) {
    let tile = PATCH_TILE;
    let atlas_cols = groups.len().min(16);
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
            atlas_x,
            atlas_y,
            ref_frame,
            positions,
        });
    }
    (atlas, references)
}

pub(crate) const PATCH_TILE: usize = 16;
pub(crate) const PATCH_REF_ID: u32 = 3;
/// Reference slot for the modular atlas; the VarDCT atlas keeps slot 3, so a
/// hybrid plan can emit both and route each dictionary entry to either.
pub(crate) const MODULAR_PATCH_REF_ID: u32 = 2;
pub(crate) const NUM_PATCH_CONTEXTS: usize = 10;

#[derive(Clone)]
pub(crate) struct PatchReference {
    pub(crate) atlas_x: usize,
    pub(crate) atlas_y: usize,
    /// Which saved reference frame this entry copies from.
    pub(crate) ref_frame: u32,
    pub(crate) positions: Vec<(usize, usize)>,
}

pub(crate) struct LosslessPatches {
    pub(crate) atlas: Image3Si,
    pub(crate) base: Image3Si,
    pub(crate) references: Vec<PatchReference>,
}

#[derive(Clone, Copy)]
pub(crate) enum ModularFrameKind<'a> {
    Regular,
    ReferenceOnly { width: usize, height: usize },
    Patched(&'a [PatchReference]),
}

impl ModularFrameKind<'_> {
    pub(crate) fn is_regular(self) -> bool {
        matches!(self, Self::Regular)
    }
}

/// Order-sensitive bulk hash of one lossless tile. Packing pairs cuts the
/// number of mixing rounds in half compared with hashing individual `i32`s.
fn hash_lossless_tile(linear: &Image3Si, x0: usize, y0: usize) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in 0..3 {
        for y in y0..y0 + PATCH_TILE {
            let row = &linear.plane_row(c, y)[x0..x0 + PATCH_TILE];
            for pair in row.as_chunks::<2>().0 {
                let v = u64::from(pair[0] as u32) | (u64::from(pair[1] as u32) << 32);
                h = (h ^ v).wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 29;
            }
        }
    }
    h
}

pub(crate) fn find_lossless_patches(
    linear: &Image3Si,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<LosslessPatches> {
    let width = linear.xsize();
    let height = linear.ysize();
    if width < PATCH_TILE || height < PATCH_TILE {
        return None;
    }

    let tiles_x = width / PATCH_TILE;
    let tiles_y = height / PATCH_TILE;
    let rows = pool.steal_map(scratch, tiles_y, |ty, _scratch| {
        (0..tiles_x)
            .map(|tx| hash_lossless_tile(linear, tx * PATCH_TILE, ty * PATCH_TILE))
            .collect::<Vec<u64>>()
    });
    let mut buckets: HashMap<u64, Vec<(usize, usize)>> =
        HashMap::with_capacity(tiles_x.saturating_mul(tiles_y));
    for (ty, row) in rows.into_iter().enumerate() {
        for (tx, hash) in row.into_iter().enumerate() {
            buckets
                .entry(hash)
                .or_default()
                .push((tx * PATCH_TILE, ty * PATCH_TILE));
        }
    }

    let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
    for candidates in buckets.into_values() {
        if candidates.len() < 3 {
            continue;
        }
        // Hash collisions are split into exact-equality groups. The normal case
        // takes the first branch and performs one comparison per occurrence.
        let mut exact_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        for pos in candidates {
            let mut matching = None;
            for (i, group) in exact_groups.iter().enumerate() {
                let a = group[0];
                let equal = (0..3).all(|c| {
                    (0..PATCH_TILE).all(|dy| {
                        linear.plane_row(c, a.1 + dy)[a.0..a.0 + PATCH_TILE]
                            == linear.plane_row(c, pos.1 + dy)[pos.0..pos.0 + PATCH_TILE]
                    })
                });
                if equal {
                    matching = Some(i);
                    break;
                }
            }
            if let Some(i) = matching {
                exact_groups[i].push(pos);
            } else {
                exact_groups.push(vec![pos]);
            }
        }
        groups.extend(exact_groups.into_iter().filter(|g| g.len() >= 3));
    }
    groups.sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
    groups.truncate(256);
    if groups.is_empty() {
        return None;
    }

    let atlas_cols = groups.len().min(16);
    let atlas_rows = groups.len().div_ceil(atlas_cols);
    let atlas_width = atlas_cols * PATCH_TILE;
    let atlas_height = atlas_rows * PATCH_TILE;

    let mut atlas = Image3Si::new(atlas_width, atlas_height);
    let mut base = linear.clone();
    let mut references = Vec::with_capacity(groups.len());
    for (i, positions) in groups.into_iter().enumerate() {
        let atlas_x = (i % atlas_cols) * PATCH_TILE;
        let atlas_y = (i / atlas_cols) * PATCH_TILE;
        let src = positions[0];
        for c in 0..3 {
            for dy in 0..PATCH_TILE {
                atlas.plane_row_mut(c, atlas_y + dy)[atlas_x..atlas_x + PATCH_TILE]
                    .copy_from_slice(&linear.plane_row(c, src.1 + dy)[src.0..src.0 + PATCH_TILE]);
                for &(x, y) in &positions {
                    base.plane_row_mut(c, y + dy)[x..x + PATCH_TILE].fill(0);
                }
            }
        }
        references.push(PatchReference {
            atlas_x,
            atlas_y,
            ref_frame: PATCH_REF_ID,
            positions,
        });
    }
    Some(LosslessPatches {
        atlas,
        base,
        references,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Six distinct 16x16 glyphs, each repeated exactly six times (comfortably
    /// past MIN_PATCH_OCCURRENCES), so every group has the same count and only
    /// the tie-break can order them.
    fn equal_length_groups() -> Image3F {
        let tile = PATCH_TILE;
        let (cols, rows) = (6usize, 6usize);
        let mut img = Image3F::new(cols * tile, rows * tile);
        for ty in 0..rows {
            for tx in 0..cols {
                for c in 0..3 {
                    for dy in 0..tile {
                        let row = img.plane_row_mut(c, ty * tile + dy);
                        for dx in 0..tile {
                            row[tx * tile + dx] = ((tx * 31 + dy * 3 + dx + c) % 17) as f32 / 17.0;
                        }
                    }
                }
            }
        }
        img
    }

    fn equal_length_lossless_groups() -> Image3Si {
        let tile = PATCH_TILE;
        let (cols, rows) = (6usize, 4usize);
        let mut img = Image3Si::new(cols * tile, rows * tile);
        for ty in 0..rows {
            for tx in 0..cols {
                for c in 0..3 {
                    for dy in 0..tile {
                        let row = img.plane_row_mut(c, ty * tile + dy);
                        for dx in 0..tile {
                            row[tx * tile + dx] = ((tx * 31 + dy * 3 + dx + c) % 17) as i32 - 8;
                        }
                    }
                }
            }
        }
        img
    }

    #[test]
    fn equal_length_groups_get_a_stable_atlas_order() {
        let pool = ThreadPool::new(4);
        let mut scratch = CoderScratch::default();
        let img = equal_length_groups();

        let signature =
            |p: &LossyPatches| -> Vec<(usize, usize)> { p.groups.iter().map(|g| g[0]).collect() };

        let plan = find_lossy_patches(&img, &pool, &mut scratch).expect("groups");
        let first = signature(&plan);
        assert_eq!(first.len(), 6);
        assert!(
            first.windows(2).all(|w| w[0] < w[1]),
            "groups must follow the first occurrence in raster order: {first:?}"
        );
        // Packing preserves that order in atlas slots and stamps the frame id.
        let (atlas, refs) = pack_lossy_atlas(&img, plan.groups, PATCH_REF_ID);
        assert_eq!((atlas.xsize(), atlas.ysize()), (6 * PATCH_TILE, PATCH_TILE));
        for (i, r) in refs.iter().enumerate() {
            assert_eq!((r.atlas_x, r.atlas_y), (i * PATCH_TILE, 0));
            assert_eq!(r.ref_frame, PATCH_REF_ID);
            assert_eq!(r.positions[0], first[i]);
        }
        // Within one process the map is seeded once, so repeats here only guard
        // the sort itself; the raster-order assertion above is what pins the
        // layout across processes.
        for _ in 0..4 {
            let again = find_lossy_patches(&img, &pool, &mut scratch).expect("groups");
            assert_eq!(first, signature(&again));
        }
    }

    #[test]
    fn lossless_patch_discovery_is_thread_deterministic() {
        let img = equal_length_lossless_groups();
        let run = |threads| {
            let pool = ThreadPool::new(threads);
            let mut scratch = CoderScratch::default();
            find_lossless_patches(&img, &pool, &mut scratch).expect("groups")
        };

        let single = run(1);
        let parallel = run(4);
        assert_eq!(single.references.len(), 6);
        assert_eq!(single.references.len(), parallel.references.len());
        for (a, b) in single.references.iter().zip(&parallel.references) {
            assert_eq!((a.atlas_x, a.atlas_y), (b.atlas_x, b.atlas_y));
            assert_eq!(a.positions, b.positions);
        }
        for c in 0..3 {
            assert_eq!(single.atlas.plane_data(c), parallel.atlas.plane_data(c));
            assert_eq!(single.base.plane_data(c), parallel.base.plane_data(c));
        }
    }
}
