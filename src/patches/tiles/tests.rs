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
use super::atlas::*;
use super::*;
use crate::patches::{PATCH_REF_ID, PATCH_TILE, PatchReference};

#[test]
fn patch_buckets_preserve_floor_and_saturating_cast() {
    let check = |value: f32| {
        for shift in [0.0f32, 0.5] {
            let shifted = value + shift;
            let expected = shifted.floor() as i64;
            assert_eq!(
                floor_patch_bucket_from_truncation(shifted),
                expected,
                "bits={:08x}, shift={shift}",
                value.to_bits()
            );
            assert_eq!(floor_patch_bucket(shifted), expected);
        }
    };
    // Every exponent, both signs, and mantissa boundaries cover subnormals,
    // infinities, NaNs, and both ends of the saturating i64 cast.
    for exponent in 0..=255u32 {
        for mantissa in [0, 1, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff] {
            for sign in [0, 0x80000000] {
                check(f32::from_bits(sign | exponent << 23 | mantissa));
            }
        }
    }
    for integer in -64..=64 {
        let value = integer as f32;
        for neighbor in [value.next_down(), value, value.next_up()] {
            check(neighbor);
        }
    }
    let mut bits = 0x4139b812u32;
    for _ in 0..65536 {
        bits ^= bits << 13;
        bits ^= bits >> 17;
        bits ^= bits << 5;
        check(f32::from_bits(bits));
    }
}

#[test]
fn cached_solid_hashes_preserve_float_bits_and_detect_nonuniform_tiles() {
    for tile in [8, 16] {
        // Nonzero padding checks that classification respects the tile bounds.
        let mut image = Image3F::new(tile + 3, tile + 2);
        for c in 0..3 {
            for y in 0..image.ysize() {
                image.plane_row_mut(c, y).fill(0.75);
            }
        }
        let mut cache = None;
        for bits in [
            [0u32; 3],
            [0u32; 3],
            [0x80000000, 0, 0],
            [0x3e800000, 0x3f000000, 0x3f400000],
            [0x7fc00001, 0, 0],
        ] {
            for c in 0..3 {
                for y in 1..tile + 1 {
                    image.plane_row_mut(c, y)[1..tile + 1].fill(f32::from_bits(bits[c]));
                }
            }
            let (hash, solid) = hash_tile_cached(&image, 1, 1, tile, &mut cache);
            assert!(solid);
            assert_eq!(hash, hash_tile(&image, 1, 1, tile));
            assert_eq!(hash_tile_cached(&image, 1, 1, tile, &mut cache).0, hash);
            for c in 0..3 {
                for (x, y) in [
                    (0, 0),
                    (tile - 1, 0),
                    (0, tile - 1),
                    (tile - 1, tile - 1),
                    (tile / 2, tile / 2),
                ] {
                    image.plane_row_mut(c, y + 1)[x + 1] = f32::from_bits(bits[c] ^ 1);
                    let (actual, solid) = hash_tile_cached(&image, 1, 1, tile, &mut cache);
                    assert!(!solid, "missed change in channel {c} at ({x}, {y})");
                    assert_eq!(actual, hash_tile(&image, 1, 1, tile));
                    image.plane_row_mut(c, y + 1)[x + 1] = f32::from_bits(bits[c]);
                }
            }
        }
    }
}

#[test]
fn exact_patch_plan_and_residual_preserve_order_and_borders() {
    let mut scratch = CoderScratch::default();
    let serial_pool = ThreadPool::new(1);
    let parallel_pool = ThreadPool::new(4);
    for tile in [8, 16] {
        let mut img = Image3F::new(64 * tile + 3, 33 * tile + 5);
        for c in 0..3 {
            for y in 0..img.ysize() {
                for x in 0..img.xsize() {
                    let id = y / tile * 64 + x / tile;
                    let pattern = if id % 5 == 0 { 10000 + id } else { id % 31 };
                    img.plane_row_mut(c, y)[x] = pattern as f32 * 0.001
                        + ((x % tile * 7 + y % tile * 3 + c * 5) % 17) as f32 * 0.025;
                }
            }
        }
        // Uncovered edge samples must be copied bit for bit, including
        // signed zero and NaN payloads, even when bands cut through tiles.
        img.plane_row_mut(0, 0)[64 * tile] = -0.0;
        img.plane_row_mut(1, 33 * tile)[0] = f32::from_bits(0x7fc0_0123);
        let serial = find_lossy_patches_sized(&img, tile, 0.0, &serial_pool, &mut scratch).unwrap();
        let parallel =
            find_lossy_patches_sized(&img, tile, 0.0, &parallel_pool, &mut scratch).unwrap();
        assert!(serial.groups.len() >= 20);
        assert_eq!(serial.groups, parallel.groups);
        let groups =
            find_lossy_patch_groups_sized(&img, tile, 0.0, &parallel_pool, &mut scratch).unwrap();
        assert_eq!(groups, serial.groups);
        let mut in_place = img.clone();
        clear_patch_tiles(&mut in_place, tile, &groups);
        let mut expected = img.clone();
        for group in &serial.groups {
            for &(x, y) in group {
                for c in 0..3 {
                    for dy in 0..tile {
                        expected.plane_row_mut(c, y + dy)[x..x + tile].fill(0.0);
                    }
                }
            }
        }
        for c in 0..3 {
            for ((want, serial), parallel) in expected
                .plane_data(c)
                .iter()
                .zip(serial.base.plane_data(c))
                .zip(parallel.base.plane_data(c))
            {
                assert_eq!(want.to_bits(), serial.to_bits());
                assert_eq!(want.to_bits(), parallel.to_bits());
            }
            for (got, want) in in_place.plane_data(c).iter().zip(serial.base.plane_data(c)) {
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
    }
}

#[test]
fn exact_group_consolidation_does_not_chain_tolerance() {
    let pool = ThreadPool::new(2);
    let mut scratch = CoderScratch::default();
    let tile = 16;
    let mut img = Image3F::new(15 * tile, tile);
    for tx in 0..15 {
        for c in 0..3 {
            for y in 0..tile {
                for x in 0..tile {
                    img.plane_row_mut(c, y)[tx * tile + x] = if x < 8 { 0.1 } else { 0.5 };
                    if c == 1 {
                        img.plane_row_mut(c, y)[tx * tile + x] +=
                            (tx / 5) as f32 * NEAR_TILE_UNIT[1] * 0.75;
                    }
                }
            }
        }
    }
    let base = find_lossy_patches_sized(&img, tile, 0.0, &pool, &mut scratch).unwrap();
    assert_eq!(base.groups.len(), 3);
    let merged = find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).unwrap();
    assert_eq!(merged.groups.len(), 2);
    assert_eq!(merged.groups[0].len(), 10);
    assert_eq!(merged.groups[1].len(), 5);
    for group in &merged.groups {
        let (sx, sy) = group[0];
        for &(px, py) in group {
            for c in 0..3 {
                for y in 0..tile {
                    for x in 0..tile {
                        assert!(
                            (img.plane_row(c, sy + y)[sx + x] - img.plane_row(c, py + y)[px + x])
                                .abs()
                                <= NEAR_TILE_UNIT[c]
                        );
                    }
                }
            }
        }
    }
    for _ in 0..4 {
        let repeated = find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).unwrap();
        assert_eq!(merged.groups, repeated.groups);
    }
}

#[test]
fn morton_order_is_complete_and_deterministic_with_duplicate_points() {
    assert!(morton_similarity_order(&[]).is_empty());
    assert_eq!(morton_similarity_order(&[[0.0; 3]]), [0]);
    for n in [2, 64, 257, 4096] {
        let points: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                [
                    ((i * 17) % 73) as f32 / 1024.0,
                    ((i * 11) % 29) as f32 / 32.0,
                    ((i * 7) % 13) as f32 / 16.0,
                ]
            })
            .collect();
        for points in [points, vec![[0.125; 3]; n]] {
            let order = morton_similarity_order(&points);
            assert_eq!(order[0], 0);
            assert_eq!(order, morton_similarity_order(&points));
            let mut sorted = order;
            sorted.shared_sort();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
        }
    }
}

#[test]
fn quadrants_of_packed_tiles_are_reused() {
    let tile = PATCH_TILE;
    let half = tile / 2;
    // One 16x16 tile whose four quadrants differ, and three 8x8 tiles:
    // its lower-left quadrant, its upper-right quadrant, and a stranger.
    let mut img = Image3F::new(4 * tile, 2 * tile);
    let value = |x: usize, y: usize, c: usize| ((x * 7 + y * 13 + c * 5) % 23) as f32 / 23.0;
    for c in 0..3 {
        for y in 0..tile {
            for x in 0..tile {
                img.plane_row_mut(c, y)[x] = value(x, y, c);
            }
        }
        for y in 0..half {
            for x in 0..half {
                img.plane_row_mut(c, tile + y)[x] = value(x, half + y, c);
                img.plane_row_mut(c, tile + y)[tile + x] = value(half + x, y, c);
                img.plane_row_mut(c, tile + y)[2 * tile + x] = value(x, y, c) * 0.5 + 0.3;
            }
        }
    }
    let packed = [PatchReference {
        atlas_x: 32,
        atlas_y: 48,
        width: tile,
        height: tile,
        ref_frame: PATCH_REF_ID,
        alpha_only: false,
        add: false,
        mul: false,
        positions: vec![(0, 0)],
    }];
    let groups = vec![
        vec![(0, tile), (40, 40)],
        vec![(tile, tile)],
        vec![(2 * tile, tile)],
    ];
    let (reused, remaining) = reuse_tile_quadrants(&img, &packed, groups);
    assert_eq!(remaining, vec![vec![(2 * tile, tile)]]);
    assert_eq!(reused.len(), 2);
    assert_eq!((reused[0].atlas_x, reused[0].atlas_y), (32, 48 + half));
    assert_eq!((reused[1].atlas_x, reused[1].atlas_y), (32 + half, 48));
    for r in &reused {
        assert_eq!(
            (r.width, r.height, r.ref_frame, r.add),
            (half, half, PATCH_REF_ID, false)
        );
    }
    assert_eq!(reused[0].positions, vec![(0, tile), (40, 40)]);
}

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
        first.array_windows::<2>().all(|w| w[0] < w[1]),
        "groups must follow the first occurrence in raster order: {first:?}"
    );
    // Packing chains look-alike tiles, so slots need not follow the group
    // order; every group must still get its own slot, in raster order of
    // the returned entries, with the frame id stamped.
    let (atlas, refs) = pack_lossy_atlas(&img, plan.groups, PATCH_REF_ID);
    assert_eq!((atlas.xsize(), atlas.ysize()), (6 * PATCH_TILE, PATCH_TILE));
    for (i, r) in refs.iter().enumerate() {
        assert_eq!((r.atlas_x, r.atlas_y), (i * PATCH_TILE, 0));
        assert_eq!(r.ref_frame, PATCH_REF_ID);
    }
    let mut packed: Vec<(usize, usize)> = refs.iter().map(|r| r.positions[0]).collect();
    let layout = packed.clone();
    packed.shared_sort();
    assert_eq!(packed, first);
    // Within one process the map is seeded once, so repeats here only guard
    // the sort itself; the raster-order assertion above is what pins the
    // layout across processes.
    for _ in 0..4 {
        let again = find_lossy_patches(&img, &pool, &mut scratch).expect("groups");
        assert_eq!(first, signature(&again));
        let (_, refs) = pack_lossy_atlas(&img, again.groups, PATCH_REF_ID);
        let repeat: Vec<(usize, usize)> = refs.iter().map(|r| r.positions[0]).collect();
        assert_eq!(layout, repeat);
    }
}
