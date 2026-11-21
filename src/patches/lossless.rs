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
use super::tiles::sort_patch_groups;
use super::{ContentHashMap, LosslessPatches, PATCH_REF_ID, PATCH_TILE, PatchReference};
use crate::coder_scratch::CoderScratch;
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;

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
    let mut buckets: ContentHashMap<Vec<(usize, usize)>> = ContentHashMap::with_capacity_and_hasher(
        tiles_x.saturating_mul(tiles_y),
        Default::default(),
    );
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
    sort_patch_groups(&mut groups);
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
            width: PATCH_TILE,
            height: PATCH_TILE,
            atlas_x,
            atlas_y,
            ref_frame: PATCH_REF_ID,
            alpha_only: false,
            add: false,
            mul: false,
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
