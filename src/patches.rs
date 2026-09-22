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
fn hash_tile(linear: &Image3F, x0: usize, y0: usize, tile: usize) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in 0..3 {
        for y in y0..y0 + tile {
            let row = &linear.plane_row(c, y)[x0..x0 + tile];
            for pair in row.as_chunks::<2>().0 {
                let v = u64::from(pair[0].to_bits()) | (u64::from(pair[1].to_bits()) << 32);
                h = (h ^ v).wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 29;
            }
        }
    }
    h
}

/// Reuse the full hash of the most recent solid tile in this tile row.
/// Bitwise tests preserve the original treatment of signed zero and NaNs.
fn hash_tile_cached(
    image: &Image3F,
    x: usize,
    y: usize,
    tile: usize,
    cache: &mut Option<([u32; 3], u64)>,
) -> (u64, bool) {
    let colors: [u32; 3] = std::array::from_fn(|c| image.plane_row(c, y)[x].to_bits());
    let solid = [1, 0, 2].into_iter().all(|c| {
        (y..y + tile).all(|dy| {
            image.plane_row(c, dy)[x..x + tile]
                .iter()
                .all(|v| v.to_bits() == colors[c])
        })
    });
    if solid
        && let Some((previous, hash)) = cache
        && *previous == colors
    {
        return (*hash, true);
    }
    let hash = hash_tile(image, x, y, tile);
    if solid {
        *cache = Some((colors, hash));
    }
    (hash, solid)
}

/// Mean absolute deviation of a tile, summed over channels: a cheap stand-in
/// for what the tile costs to code as ordinary blocks.
fn tile_energy(img: &Image3F, x0: usize, y0: usize, tile: usize) -> f32 {
    let mut energy = 0.0;
    for c in 0..3 {
        let mut sum = 0.0;
        for y in y0..y0 + tile {
            sum += img.plane_row(c, y)[x0..x0 + tile].iter().sum::<f32>();
        }
        let mean = sum / (tile * tile) as f32;
        for y in y0..y0 + tile {
            energy += img.plane_row(c, y)[x0..x0 + tile]
                .iter()
                .map(|v| (v - mean).abs())
                .sum::<f32>();
        }
    }
    energy / (tile * tile) as f32
}

/// Minimum per-tile energy worth spending a patch on.
const MIN_PATCH_ENERGY: f32 = 0.017;

/// Minimum occurrences before a tile group becomes a patch. The study
/// preferred 5 to the historical 3 at every good configuration: rare groups
/// pay dictionary positions without amortizing their atlas tile.
const MIN_PATCH_OCCURRENCES: usize = 5;

/// Most tile groups one plan keeps.
const MAX_PATCH_GROUPS: usize = 16384;

#[cfg(test)]
pub(crate) fn find_lossy_patches(
    linear: &Image3F,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<LossyPatches> {
    find_lossy_patches_sized(linear, PATCH_TILE, 0.0, pool, scratch)
}

/// Per-channel (X, Y, B) tolerance of one unit of `near`: about one 8-bit
/// level of a mid-gray pixel.
static NEAR_TILE_UNIT: [f32; 3] = [0.0005, 0.0035, 0.006];

const NEAR_TILE_BUCKET_SCAN: usize = 24;
/// Textured tiles (at most this share of flat luma samples) may differ from
/// their leader by this many tolerances per sample; the RMS bound stays one.
const NEAR_TILE_MAX_FLAT_SHARE: f32 = 0.55;
const NEAR_TILE_TEXTURED_PEAK: f32 = 4.0;

#[inline]
fn floor_patch_bucket(value: f32) -> i64 {
    if cfg!(all(target_arch = "x86_64", not(target_feature = "sse4.1"))) {
        floor_patch_bucket_from_truncation(value)
    } else {
        value.floor() as i64
    }
}

#[inline]
fn floor_patch_bucket_from_truncation(value: f32) -> i64 {
    let truncated = value as i64;
    // Only negative fractions need correction; saturation also preserves the
    // floor-then-cast result for values below i64::MIN (including -infinity).
    truncated.saturating_sub(i64::from(value < truncated as f32))
}

/// `near > 0` also groups tiles close to a group's first tile: per channel the
/// RMS difference is within `near * NEAR_TILE_UNIT` and every sample within
/// that (flat tiles) or a few times that (textured tiles). They are replaced
/// by the first tile, so the substitution error is bounded per sample.
pub(crate) fn find_lossy_patches_sized(
    linear: &Image3F,
    tile: usize,
    near: f32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<LossyPatches> {
    let width = linear.xsize();
    let height = linear.ysize();
    if width < tile || height < tile {
        return None;
    }

    let (tiles_x, tiles_y) = (width / tile, height / tile);
    // Tile rows are independent; the merge below is over tile counts, not
    // pixels, so it stays negligible.
    let rows = pool.steal_map(scratch, tiles_y, |ty, _scratch| {
        let mut cache = None;
        (0..tiles_x)
            .map(|tx| hash_tile_cached(linear, tx * tile, ty * tile, tile, &mut cache))
            .unzip::<_, _, Vec<_>, Vec<_>>()
    });
    let solid: Vec<bool> = rows
        .iter()
        .flat_map(|(_, flags)| flags.iter().copied())
        .collect();
    let candidates: Vec<Vec<(usize, usize)>> = {
        let mut hashes: Vec<(u64, usize)> = rows
            .into_iter()
            .enumerate()
            .flat_map(|(ty, (row, _))| {
                row.into_iter()
                    .enumerate()
                    .map(move |(tx, hash)| (hash, ty * tiles_x + tx))
            })
            .collect();
        // Position breaks equal-hash ties to preserve the raster-order source.
        hashes.sort_unstable();
        hashes
            .chunk_by(|a, b| a.0 == b.0)
            .filter(|bucket| bucket.len() >= MIN_PATCH_OCCURRENCES)
            .map(|bucket| {
                bucket
                    .iter()
                    .map(|&(_, i)| ((i % tiles_x) * tile, (i / tiles_x) * tile))
                    .collect()
            })
            .collect()
    };
    let mut groups = Vec::new();
    let min_occurrences = MIN_PATCH_OCCURRENCES;
    for candidates in candidates {
        if candidates.len() < min_occurrences {
            continue;
        }
        let mut exact_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        for pos in candidates {
            let matching = exact_groups.iter().position(|group| {
                let first = group[0];
                // Both tiles were proved constant across all three planes.
                // Compare their samples with the original float equality, so
                // NaN remains unequal and ordinary hash collisions stay safe.
                if solid[(first.1 / tile) * tiles_x + first.0 / tile]
                    && solid[(pos.1 / tile) * tiles_x + pos.0 / tile]
                {
                    return (0..3).all(|c| {
                        linear.plane_row(c, first.1)[first.0] == linear.plane_row(c, pos.1)[pos.0]
                    });
                }
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
                .filter(|g| g.len() >= min_occurrences),
        );
    }

    if near > 0.0 {
        // The exact groups arrive in hash order; the near pass registers and
        // scans leaders in sequence, so fix the order first.
        sort_patch_groups(&mut groups);
        let tol = NEAR_TILE_UNIT.map(|u| u * near);
        let mut taken = vec![false; tiles_x * tiles_y];
        for g in &groups {
            for &(x, y) in g {
                taken[(y / tile) * tiles_x + x / tile] = true;
            }
        }
        // Coarse signature: 2x2 cell means of Y plus whole-tile X and B means
        // in buckets 16 tolerances wide. Two grids half a bucket apart: a
        // value on a boundary of one sits mid-bucket in the other.
        let cell = tile / 2;
        let signature = |x0: usize, y0: usize| -> ([u64; 2], [f32; 6]) {
            let mut means = [0f32; 6];
            for cy in 0..2 {
                for cx in 0..2 {
                    let mut sum = 0.0f32;
                    for dy in 0..cell {
                        let row = linear.plane_row(1, y0 + cy * cell + dy);
                        sum += row[x0 + cx * cell..x0 + (cx + 1) * cell]
                            .iter()
                            .sum::<f32>();
                    }
                    means[cy * 2 + cx] = sum / (cell * cell) as f32 / (16.0 * tol[1]);
                }
            }
            for (slot, c) in [(4usize, 0usize), (5, 2)] {
                let mut sum = 0.0f32;
                for dy in 0..tile {
                    sum += linear.plane_row(c, y0 + dy)[x0..x0 + tile]
                        .iter()
                        .sum::<f32>();
                }
                means[slot] = sum / (tile * tile) as f32 / (16.0 * tol[c]);
            }
            let keys = [0.0f32, 0.5].map(|shift| {
                let mut h: u64 = 0x9e37_79b9_7f4a_7c15 ^ shift.to_bits() as u64;
                for m in means {
                    h = (h ^ floor_patch_bucket(m + shift) as u64)
                        .wrapping_mul(0xff51_afd7_ed55_8ccd);
                    h ^= h >> 29;
                }
                h
            });
            (keys, means)
        };
        // An RMS difference within one tolerance bounds a quarter-tile mean
        // within two and a whole-tile mean within one (means are in units of
        // 16 tolerances): bucket-mates that fail this cannot match.
        let may_match = |a: &[f32; 6], b: &[f32; 6]| {
            a.iter()
                .zip(b)
                .enumerate()
                .all(|(i, (p, q))| (p - q).abs() <= if i < 4 { 2.02 / 16.0 } else { 1.01 / 16.0 })
        };
        // Share of luma samples that equal their left neighbor: UI and text
        // tiles are mostly flat, where a substituted sample shows; texture
        // masks it.
        let is_textured = |p: (usize, usize)| -> bool {
            let flat: usize = (0..tile)
                .map(|dy| {
                    linear.plane_row(1, p.1 + dy)[p.0..p.0 + tile]
                        .array_windows::<2>()
                        .filter(|w| (w[0] - w[1]).abs() <= NEAR_TILE_UNIT[1] * 0.5)
                        .count()
                })
                .sum();
            flat as f32 <= NEAR_TILE_MAX_FLAT_SHARE * (tile * (tile - 1)) as f32
        };
        // Every sample within `peak` tolerances of the leader's and the
        // channel's RMS difference within one.
        let within = |a: (usize, usize), b: (usize, usize), peak: f32| -> bool {
            // Reject shape mismatches before reading whole channels. This is
            // only a necessary peak check: survivors still need the complete
            // peak/RMS check below, with the original accumulation order.
            for c in [1, 0, 2] {
                for i in 0..8 {
                    let y = i * tile / 8;
                    let x = (i * 5 + tile / 2) % tile;
                    let d = linear.plane_row(c, a.1 + y)[a.0 + x]
                        - linear.plane_row(c, b.1 + y)[b.0 + x];
                    if d.abs() > tol[c] * peak {
                        return false;
                    }
                }
            }
            (0..3).all(|c| {
                let peak = tol[c] * peak;
                let budget = tol[c] * tol[c] * (tile * tile) as f32;
                let mut energy = 0.0f32;
                (0..tile).all(|dy| {
                    let ra = &linear.plane_row(c, a.1 + dy)[a.0..a.0 + tile];
                    let rb = &linear.plane_row(c, b.1 + dy)[b.0..b.0 + tile];
                    // Most candidates fail on their first samples.
                    ra.iter().zip(rb).all(|(p, q)| {
                        let d = p - q;
                        energy += d * d;
                        d.abs() <= peak && energy <= budget
                    })
                })
            })
        };
        let mut table: HashMap<u64, Vec<usize>> = HashMap::new();
        let peak_of = |p: (usize, usize)| {
            if is_textured(p) {
                NEAR_TILE_TEXTURED_PEAK
            } else {
                1.0
            }
        };
        let mut peaks: Vec<f32> = Vec::with_capacity(groups.len());
        let mut means: Vec<[f32; 6]> = Vec::with_capacity(groups.len());
        let exact_groups = std::mem::take(&mut groups);
        groups.reserve(exact_groups.len());
        for g in exact_groups {
            let (keys, mean) = signature(g[0].0, g[0].1);
            // Exact members all equal this source, so one comparison proves
            // the bound for the entire group. Leaders never change; matches
            // cannot accumulate error through a chain of representatives.
            let found = keys.iter().find_map(|key| {
                table
                    .get(key)?
                    .iter()
                    .take(NEAR_TILE_BUCKET_SCAN)
                    .copied()
                    .filter(|&gi| may_match(&means[gi], &mean))
                    .find(|&gi| within(groups[gi][0], g[0], peaks[gi]))
            });
            if let Some(gi) = found {
                groups[gi].extend(g);
            } else {
                let gi = groups.len();
                peaks.push(peak_of(g[0]));
                means.push(mean);
                for key in keys {
                    table.entry(key).or_default().push(gi);
                }
                groups.push(g);
            }
        }
        let exact_count = groups.len();
        // Descriptor construction is independent of the deterministic greedy
        // matching below, so only this image-reading work runs in parallel.
        let features = (pool.num_threads() > 1 && tiles_x * tiles_y >= 4096).then(|| {
            pool.steal_map(scratch, tiles_y, |ty, _| {
                (0..tiles_x)
                    .map(|tx| {
                        let pos = (tx * tile, ty * tile);
                        if taken[ty * tiles_x + tx]
                            || tile_energy(linear, pos.0, pos.1, tile) < MIN_PATCH_ENERGY
                        {
                            None
                        } else {
                            Some(signature(pos.0, pos.1))
                        }
                    })
                    .collect::<Vec<_>>()
            })
        });
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                if taken[ty * tiles_x + tx] {
                    continue;
                }
                let pos = (tx * tile, ty * tile);
                let (keys, mean) = if let Some(rows) = &features {
                    let Some(feature) = rows[ty][tx] else {
                        continue;
                    };
                    feature
                } else {
                    if tile_energy(linear, pos.0, pos.1, tile) < MIN_PATCH_ENERGY {
                        continue;
                    }
                    signature(pos.0, pos.1)
                };
                // Textured content fills buckets with unrelated leaders; a
                // bounded scan keeps the pass linear.
                let found = keys.iter().find_map(|key| {
                    table
                        .get(key)?
                        .iter()
                        .take(NEAR_TILE_BUCKET_SCAN)
                        .copied()
                        .filter(|&gi| may_match(&means[gi], &mean))
                        .find(|&gi| within(groups[gi][0], pos, peaks[gi]))
                });
                match found {
                    Some(gi) => groups[gi].push(pos),
                    None => {
                        let mut registered = false;
                        for key in keys {
                            let bucket = table.entry(key).or_default();
                            if bucket.len() < NEAR_TILE_BUCKET_SCAN {
                                bucket.push(groups.len());
                                registered = true;
                            }
                        }
                        if registered {
                            groups.push(vec![pos]);
                            peaks.push(peak_of(pos));
                            means.push(mean);
                        }
                    }
                }
            }
        }
        // Provisional near groups must reach the floor on their own.
        let mut index = 0;
        groups.retain(|g| {
            index += 1;
            index <= exact_count || g.len() >= min_occurrences
        });
    }

    groups.retain(|g| tile_energy(linear, g[0].0, g[0].1, tile) >= MIN_PATCH_ENERGY);
    sort_patch_groups(&mut groups);
    groups.truncate(MAX_PATCH_GROUPS);
    if groups.is_empty() {
        return None;
    }

    // Preserve the selected source at index zero; order the remaining
    // occurrences spatially after concatenating exact groups.
    for g in &mut groups {
        g[1..].sort_unstable_by_key(|&(x, y)| (y, x));
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
                add: false,
                positions,
            }),
            None => remaining.push(positions),
        }
    }
    (reused, remaining)
}

/// Visit order that keeps look-alike tiles next to each other in the atlas:
/// a greedy nearest-neighbor chain over each tile's mean X, Y, B.
fn similarity_order(linear: &Image3F, groups: &[Vec<(usize, usize)>], tile: usize) -> Vec<usize> {
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

fn greedy_similarity_order(means: &[[f32; 3]]) -> Vec<usize> {
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
fn morton_similarity_order(means: &[[f32; 3]]) -> Vec<usize> {
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
    spatial.sort_unstable_by_key(|&i| (key(i), i));
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
            add: false,
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
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Which saved reference frame this entry copies from.
    pub(crate) ref_frame: u32,
    /// Blend with kAdd instead of kReplace.
    pub(crate) add: bool,
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
    ReferenceOnly {
        width: usize,
        height: usize,
    },
    /// Reference-only atlas inside an `xyb_encoded` codestream: channels are
    /// Y, X, B-Y on the default LF dequant lattice, saved to the modular slot.
    XybReferenceOnly {
        width: usize,
        height: usize,
    },
    Patched(&'a [PatchReference]),
}

impl ModularFrameKind<'_> {
    pub(crate) fn is_regular(self) -> bool {
        matches!(self, Self::Regular)
    }

    /// Every frame kind may be coded by the learned-tree path: patched
    /// frames open their first section with the dictionary, and reference-
    /// only atlas frames only differ in their header (the flat coder spent
    /// 10 bits per pixel on a glyph atlas).
    pub(crate) fn allows_learned_tree(self) -> bool {
        true
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
            add: false,
            positions,
        });
    }
    Some(LosslessPatches {
        atlas,
        base,
        references,
    })
}

/// Largest glyph box side the glyph detector keeps.
const GLYPH_MAX_SIDE: usize = 64;
/// Background is decided per tile of this size (the tile's most frequent
/// color, or that of its eight neighbors, so boxes and headers whose
/// edges cross a tile still count as background on both sides).
const GLYPH_BG_TILE: usize = 32;
/// A shape must repeat: (occurrences - 1) * box pixels at or above these pays
/// for its atlas entry and per-occurrence position tokens.
const GLYPH_MIN_REPEAT_PIXELS: usize = 48;
/// Widest atlas the shapes are shelf-packed into.
const GLYPH_ATLAS_MAX_WIDTH: usize = 1024;

/// Three signed samples packed into 21-bit biased fields (planes may hold
/// negative values, e.g. after a color transform); values outside ±2^20
/// alias, which only weakens the background vote.
const GLYPH_KEY_BIAS: i64 = 1 << 20;
const GLYPH_KEY_MASK: u64 = (1 << 21) - 1;

#[inline]
fn color_key(c0: i32, c1: i32, c2: i32) -> u64 {
    let f = |v: i32| ((v as i64 + GLYPH_KEY_BIAS) as u64) & GLYPH_KEY_MASK;
    f(c0) << 42 | f(c1) << 21 | f(c2)
}

#[inline]
fn key_color(key: u64) -> [i32; 3] {
    let f = |shift: u32| ((key >> shift) & GLYPH_KEY_MASK) as i64 - GLYPH_KEY_BIAS;
    [f(42) as i32, f(21) as i32, f(0) as i32]
}

/// Ties go to the smallest key, as in the per-pixel vote.
fn most_frequent_color(keys: &mut [u64]) -> u64 {
    keys.sort_unstable();
    let (mut best, mut best_run) = (keys[0], 0);
    for run in keys.chunk_by(|a, b| a == b) {
        if run.len() > best_run {
            best = run[0];
            best_run = run.len();
        }
    }
    best
}

/// Glyph-level patches for text and UI: connected components of
/// non-background pixels (libjxl's `FindTextLikePatches` approach), keyed by
/// their exact bounding-box content and kept when the same shape occurs at
/// least twice. Unlike the tile detector, occurrences sit at arbitrary
/// positions, which is what repeated glyphs need. The base image gets the
/// local background under every patch (Replace blending ignores it), so the
/// remaining page is nearly blank.
pub(crate) fn find_lossless_glyph_patches(linear: &Image3Si) -> Option<LosslessPatches> {
    let width = linear.xsize();
    let height = linear.ysize();
    if width < GLYPH_BG_TILE || height < GLYPH_BG_TILE {
        return None;
    }
    // Per-tile background color: the most frequent color of the tile.
    let tiles_x = width.div_ceil(GLYPH_BG_TILE);
    let tiles_y = height.div_ceil(GLYPH_BG_TILE);
    let mut modes = vec![0u64; tiles_x * tiles_y];
    let mut keys: Vec<u64> = Vec::with_capacity(GLYPH_BG_TILE * GLYPH_BG_TILE);
    for (ty, mode_row) in modes.chunks_exact_mut(tiles_x).enumerate() {
        for (tx, mode) in mode_row.iter_mut().enumerate() {
            let x0 = tx * GLYPH_BG_TILE;
            let x1 = (x0 + GLYPH_BG_TILE).min(width);
            keys.clear();
            for y in ty * GLYPH_BG_TILE..((ty + 1) * GLYPH_BG_TILE).min(height) {
                let c0 = &linear.plane_row(0, y)[x0..x1];
                let c1 = &linear.plane_row(1, y)[x0..x1];
                let c2 = &linear.plane_row(2, y)[x0..x1];
                keys.extend(
                    c0.iter()
                        .zip(c1)
                        .zip(c2)
                        .map(|((&a, &b), &c)| color_key(a, b, c)),
                );
            }
            *mode = most_frequent_color(&mut keys);
        }
    }

    // Ink map, consumed by the flood fill. Neighboring background colors
    // are shared by the whole tile; collect them once, removing duplicates.
    let mut ink = vec![false; width * height];
    for (ty, band) in ink.chunks_mut(width * GLYPH_BG_TILE).enumerate() {
        let ny0 = ty.saturating_sub(1);
        let ny1 = (ty + 2).min(tiles_y);
        let neighbors = &modes[ny0 * tiles_x..ny1 * tiles_x];
        for tx in 0..tiles_x {
            let nx0 = tx.saturating_sub(1);
            let nx1 = (tx + 2).min(tiles_x);
            let mut backgrounds = [0; 9];
            let mut num_backgrounds = 0;
            for row in neighbors.chunks_exact(tiles_x) {
                for &key in &row[nx0..nx1] {
                    if !backgrounds[..num_backgrounds].contains(&key) {
                        backgrounds[num_backgrounds] = key;
                        num_backgrounds += 1;
                    }
                }
            }
            let backgrounds = &backgrounds[..num_backgrounds];
            let x0 = tx * GLYPH_BG_TILE;
            let x1 = (x0 + GLYPH_BG_TILE).min(width);
            for (dy, row) in band.chunks_exact_mut(width).enumerate() {
                let y = ty * GLYPH_BG_TILE + dy;
                let c0 = &linear.plane_row(0, y)[x0..x1];
                let c1 = &linear.plane_row(1, y)[x0..x1];
                let c2 = &linear.plane_row(2, y)[x0..x1];
                for (((pixel, &a), &b), &c) in row[x0..x1].iter_mut().zip(c0).zip(c1).zip(c2) {
                    *pixel = !backgrounds.contains(&color_key(a, b, c));
                }
            }
        }
    }
    // Connected components (8-neighbor) with bounding boxes.
    struct Shape {
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        hash: u64,
    }
    let mut shapes: Vec<Shape> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut next_pixel = 0;
    while let Some(offset) = ink[next_pixel..].iter().position(|&pixel| pixel) {
        let index = next_pixel + offset;
        next_pixel = index + 1;
        let (x, y) = (index % width, index / width);
        let (mut x0, mut x1, mut y0, mut y1) = (x, x, y, y);
        let mut count = 0usize;
        stack.push((x, y));
        ink[index] = false;
        while let Some((cx, cy)) = stack.pop() {
            count += 1;
            x0 = x0.min(cx);
            x1 = x1.max(cx);
            y0 = y0.min(cy);
            y1 = y1.max(cy);
            let nx0 = cx.saturating_sub(1);
            let nx1 = (cx + 2).min(width);
            let ny0 = cy.saturating_sub(1);
            let ny1 = (cy + 2).min(height);
            for (dy, row) in ink[ny0 * width..ny1 * width]
                .chunks_exact_mut(width)
                .enumerate()
            {
                for (dx, pixel) in row[nx0..nx1].iter_mut().enumerate() {
                    if *pixel {
                        *pixel = false;
                        stack.push((nx0 + dx, ny0 + dy));
                    }
                }
            }
        }
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        if count < 2 || w > GLYPH_MAX_SIDE || h > GLYPH_MAX_SIDE {
            continue;
        }
        let mut hash: u64 = 0x9e37_79b9_7f4a_7c15 ^ ((w as u64) << 32 | h as u64);
        for c in 0..3 {
            for row in linear.plane_data(c)[y0 * width..(y1 + 1) * width].chunks_exact(width) {
                for &v in &row[x0..=x1] {
                    hash = (hash ^ (v as u32 as u64)).wrapping_mul(0xff51_afd7_ed55_8ccd);
                    hash ^= hash >> 29;
                }
            }
        }
        shapes.push(Shape { x0, y0, w, h, hash });
    }
    if shapes.len() < 4 {
        return None;
    }
    // Group by exact content.
    let same_box = |a: &Shape, b: &Shape| -> bool {
        a.w == b.w
            && a.h == b.h
            && (0..3).all(|c| {
                let plane = linear.plane_data(c);
                let a_rows = plane[a.y0 * width..(a.y0 + a.h) * width].chunks_exact(width);
                let b_rows = plane[b.y0 * width..(b.y0 + b.h) * width].chunks_exact(width);
                a_rows
                    .zip(b_rows)
                    .all(|(ar, br)| ar[a.x0..a.x0 + a.w] == br[b.x0..b.x0 + b.w])
            })
    };
    struct ShapeGroup<'a> {
        shape: &'a Shape,
        positions: Vec<(usize, usize)>,
    }
    let mut buckets: HashMap<u64, Vec<ShapeGroup<'_>>> = HashMap::new();
    for s in &shapes {
        let groups = buckets.entry(s.hash).or_default();
        match groups.iter_mut().find(|g| same_box(g.shape, s)) {
            Some(g) => g.positions.push((s.x0, s.y0)),
            None => groups.push(ShapeGroup {
                shape: s,
                positions: vec![(s.x0, s.y0)],
            }),
        }
    }
    let mut groups: Vec<_> = buckets
        .into_values()
        .flatten()
        .filter(|g| {
            let s = g.shape;
            g.positions.len() >= 2 && (g.positions.len() - 1) * s.w * s.h >= GLYPH_MIN_REPEAT_PIXELS
        })
        .collect();
    if groups.len() < 2 {
        return None;
    }
    let (covered, total_area, max_width) =
        groups.iter().fold((0, 0, 0), |(covered, area, w), g| {
            let s = g.shape;
            (
                covered + g.positions.len() * s.w * s.h,
                area + s.w * s.h,
                w.max(s.w),
            )
        });
    if covered * 200 < width * height {
        return None;
    }
    // Shelf-pack tallest first, then by value within each height. Boxes may
    // overlap (kerned glyphs): every box equals the image, so Replace
    // blending reproduces the image in any order. The groups arrive in hash
    // order, which differs between processes, so ties are broken by the
    // group's first occurrence (unique per group) to keep the atlas layout
    // and the dictionary deterministic.
    groups.sort_by_key(|g| {
        let s = g.shape;
        (
            std::cmp::Reverse(s.h),
            std::cmp::Reverse((g.positions.len() - 1) * s.w * s.h),
            s.y0,
            s.x0,
        )
    });
    let atlas_width = ((total_area as f64).sqrt() as usize * 3 / 2)
        .max(max_width)
        .clamp(16, GLYPH_ATLAS_MAX_WIDTH);
    let mut slots: Vec<(usize, usize)> = Vec::with_capacity(groups.len());
    let (mut cx, mut cy, mut shelf_h) = (0usize, 0usize, 0usize);
    for g in &groups {
        let s = g.shape;
        if cx + s.w > atlas_width {
            cx = 0;
            cy += shelf_h;
            shelf_h = 0;
        }
        slots.push((cx, cy));
        cx += s.w;
        shelf_h = shelf_h.max(s.h);
    }
    let atlas_height = cy + shelf_h;
    // Padding between shelf-packed shapes takes the page's dominant
    // background, so glyph edges in the atlas meet the color they meet on
    // the page rather than zero.
    let mut atlas = Image3Si::try_new(atlas_width, atlas_height).ok()?;
    keys.clear();
    keys.extend_from_slice(&modes);
    let fill = key_color(most_frequent_color(&mut keys));
    for (c, value) in fill.into_iter().enumerate() {
        atlas.plane_mut(c).as_mut_slice().fill(value);
    }
    let mut base = linear.clone();
    let mut references = Vec::with_capacity(groups.len());
    for (g, (ax, ay)) in groups.into_iter().zip(slots) {
        let s = g.shape;
        for c in 0..3 {
            let src = &linear.plane_data(c)[s.y0 * width..(s.y0 + s.h) * width];
            let dst =
                &mut atlas.plane_mut(c).as_mut_slice()[ay * atlas_width..(ay + s.h) * atlas_width];
            for (dst_row, src_row) in dst
                .chunks_exact_mut(atlas_width)
                .zip(src.chunks_exact(width))
            {
                dst_row[ax..ax + s.w].copy_from_slice(&src_row[s.x0..s.x0 + s.w]);
            }
        }
        for &(px, py) in &g.positions {
            let fill = key_color(modes[(py / GLYPH_BG_TILE) * tiles_x + px / GLYPH_BG_TILE]);
            for (c, value) in fill.into_iter().enumerate() {
                let rows = &mut base.plane_mut(c).as_mut_slice()[py * width..(py + s.h) * width];
                for row in rows.chunks_exact_mut(width) {
                    row[px..px + s.w].fill(value);
                }
            }
        }
        // Row-major occurrence order keeps the dictionary's position deltas
        // small (dy mostly 0, dx the glyph pitch).
        let mut positions = g.positions;
        positions.sort_unstable_by_key(|&(x, y)| (y, x));
        references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: s.w,
            height: s.h,
            ref_frame: PATCH_REF_ID,
            add: false,
            positions,
        });
    }
    Some(LosslessPatches {
        atlas,
        base,
        references,
    })
}

const GLYPH_MAX_INK_SHARE: f64 = 0.6;
const GLYPH_SCREEN_STRIDE: usize = 64;
const GLYPH_SCREEN_MAX_INK: f64 = 0.6;
/// The lossy glyph atlas is a multi-group modular frame; only keep it sane.
const GLYPH_ATLAS_MAX_HEIGHT: usize = 4096;

/// Glyph patches for the lossy path, libjxl style: connected ink components
/// over a flat background, stored as the difference from that background on
/// the modular XYB lattice and blended with kAdd. The base keeps the image
/// minus the decoded patch, i.e. the flat background plus lattice rounding.
pub(crate) struct LossyGlyphPatches {
    /// Lattice channels Y, X, B-Y of the atlas.
    pub(crate) atlas: [Vec<i32>; 3],
    pub(crate) atlas_width: usize,
    pub(crate) atlas_height: usize,
    pub(crate) base: Image3F,
    pub(crate) references: Vec<PatchReference>,
}

pub(crate) struct GlyphParams {
    pub(crate) min_occurrences: usize,
    /// (occurrences - 1) * box pixels a group must reach.
    pub(crate) min_repeat_pixels: usize,
    /// Lattice values are rounded to multiples of this.
    pub(crate) coarse: i32,
}

/// Share of the frame the glyph boxes must cover.
const GLYPH_MIN_COVER: f64 = 0.005;

pub(crate) fn find_lossy_glyph_patches(
    xyb: &Image3F,
    params: &GlyphParams,
) -> Option<LossyGlyphPatches> {
    use crate::quant_weights::INV_DC_QUANT;
    let (width, height) = (xyb.xsize(), xyb.ysize());
    let bg_tile = GLYPH_BG_TILE;
    if width < bg_tile || height < bg_tile {
        return None;
    }
    // Cheap screen: on a sparse sample of 16-pixel tiles, how many pixels
    // differ from their tile's most frequent color. Photos are nearly all
    // "ink" and leave before the full-frame quantization.
    {
        let mut colors = [[0i32; 3]; PATCH_TILE * PATCH_TILE];
        let mut sample_keys = [0u64; PATCH_TILE * PATCH_TILE];
        let (mut ink, mut total) = (0usize, 0usize);
        for y0 in (0..height.saturating_sub(PATCH_TILE)).step_by(GLYPH_SCREEN_STRIDE) {
            for x0 in (0..width.saturating_sub(PATCH_TILE)).step_by(GLYPH_SCREEN_STRIDE) {
                crate::xyb::quantize_xyb_tile_colors(xyb, x0, y0, 1, &mut colors);
                let k = [params.coarse; 3];
                for (key, c) in sample_keys.iter_mut().zip(&colors) {
                    // Same granularity as the full pass; tile colors are Y, X, B-Y.
                    let r = |v: i32, k: i32| (v as f32 / k as f32).round() as i32 * k;
                    *key = color_key(r(c[0], k[0]), r(c[1], k[1]), r(c[2] + c[0], k[2]));
                }
                let mut sorted = sample_keys;
                let mode = most_frequent_color(&mut sorted);
                ink += sample_keys.iter().filter(|&&key| key != mode).count();
                total += sample_keys.len();
            }
        }
        if ink as f64 > GLYPH_SCREEN_MAX_INK * total as f64 {
            return None;
        }
    }
    let [qy, qx, mut qb] = crate::xyb::quantize_xyb_channels(xyb, 1);
    let (mut qy, mut qx) = (qy, qx);
    // Work with the rounded B itself; the atlas goes back to B-Y at the end.
    for (b, &y) in qb.iter_mut().zip(&qy) {
        *b += y;
    }
    // Coarser atlas values on the same lattice: multiples of `coarse[c]`.
    for (plane, k) in [&mut qy, &mut qx, &mut qb]
        .into_iter()
        .zip([params.coarse; 3])
    {
        if k > 1 {
            for v in plane.iter_mut() {
                *v = (*v as f32 / k as f32).round() as i32 * k;
            }
        }
    }
    let planes = [&qy, &qx, &qb];
    let keys: Vec<u64> = (0..width * height)
        .map(|i| color_key(qy[i], qx[i], qb[i]))
        .collect();

    let tiles_x = width.div_ceil(bg_tile);
    let tiles_y = height.div_ceil(bg_tile);
    let mut modes = vec![0u64; tiles_x * tiles_y];
    let mut tile_keys: Vec<u64> = Vec::with_capacity(bg_tile * bg_tile);
    // A band is one row of tiles: `bg_tile` image rows (fewer at the bottom).
    let band_len = width * bg_tile;
    for (band, mode_row) in keys.chunks(band_len).zip(modes.chunks_exact_mut(tiles_x)) {
        for (tx, mode) in mode_row.iter_mut().enumerate() {
            let x0 = tx * bg_tile;
            let x1 = (x0 + bg_tile).min(width);
            tile_keys.clear();
            tile_keys.extend(band.chunks_exact(width).flat_map(|row| &row[x0..x1]));
            *mode = most_frequent_color(&mut tile_keys);
        }
    }
    let mut ink = vec![false; width * height];
    for (ty, (ink_band, key_band)) in ink
        .chunks_mut(band_len)
        .zip(keys.chunks(band_len))
        .enumerate()
    {
        let neighbor_rows = &modes[ty.saturating_sub(1) * tiles_x..(ty + 2).min(tiles_y) * tiles_x];
        for tx in 0..tiles_x {
            let (nx0, nx1) = (tx.saturating_sub(1), (tx + 2).min(tiles_x));
            let mut backgrounds = [0u64; 9];
            let mut n = 0;
            for &key in neighbor_rows
                .chunks_exact(tiles_x)
                .flat_map(|row| &row[nx0..nx1])
            {
                if !backgrounds[..n].contains(&key) {
                    backgrounds[n] = key;
                    n += 1;
                }
            }
            let backgrounds = &backgrounds[..n];
            let x0 = tx * bg_tile;
            let x1 = (x0 + bg_tile).min(width);
            for (ink_row, key_row) in ink_band
                .chunks_exact_mut(width)
                .zip(key_band.chunks_exact(width))
            {
                for (pixel, key) in ink_row[x0..x1].iter_mut().zip(&key_row[x0..x1]) {
                    *pixel = !backgrounds.contains(key);
                }
            }
        }
    }

    let ink_pixels = ink.iter().filter(|&&p| p).count();
    // Text and UI sit on flat backgrounds; when most pixels are ink there is
    // no page to subtract and the component scan is wasted time.
    if ink_pixels as f64 > GLYPH_MAX_INK_SHARE * (width * height) as f64 {
        return None;
    }

    struct Shape {
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        label: u32,
        bg: [i32; 3],
        hash: u64,
    }
    let mut label = vec![0u32; width * height];
    let mut shapes: Vec<Shape> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut next_label = 0u32;
    let mut next_pixel = 0;
    while let Some(offset) = ink[next_pixel..]
        .iter()
        .zip(&label[next_pixel..])
        .position(|(&pixel, &owner)| pixel && owner == 0)
    {
        let start = next_pixel + offset;
        next_pixel = start + 1;
        next_label += 1;
        let (x, y) = (start % width, start / width);
        let (mut x0, mut x1, mut y0, mut y1) = (x, x, y, y);
        let mut count = 0usize;
        let mut bg_key: Option<u64> = None;
        let mut one_bg = true;
        stack.push((x, y));
        label[start] = next_label;
        while let Some((cx, cy)) = stack.pop() {
            count += 1;
            x0 = x0.min(cx);
            x1 = x1.max(cx);
            y0 = y0.min(cy);
            y1 = y1.max(cy);
            let (nx0, nx1) = (cx.saturating_sub(1), (cx + 2).min(width));
            let (ny0, ny1) = (cy.saturating_sub(1), (cy + 2).min(height));
            let window = ny0 * width..ny1 * width;
            let rows = ink[window.clone()]
                .chunks_exact(width)
                .zip(label[window.clone()].chunks_exact_mut(width))
                .zip(keys[window].chunks_exact(width));
            for (dy, ((ink_row, label_row), key_row)) in rows.enumerate() {
                let pixels = ink_row[nx0..nx1]
                    .iter()
                    .zip(&mut label_row[nx0..nx1])
                    .zip(&key_row[nx0..nx1]);
                for (dx, ((&is_ink, owner), &key)) in pixels.enumerate() {
                    if is_ink {
                        if *owner == 0 {
                            *owner = next_label;
                            stack.push((nx0 + dx, ny0 + dy));
                        }
                    } else {
                        match bg_key {
                            None => bg_key = Some(key),
                            Some(k) => one_bg &= k == key,
                        }
                    }
                }
            }
        }
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        let Some(bg_key) = bg_key else { continue };
        if !one_bg || count < 2 || w > GLYPH_MAX_SIDE || h > GLYPH_MAX_SIDE {
            continue;
        }
        let bg = key_color(bg_key);
        let mut hash: u64 = 0x9e37_79b9_7f4a_7c15 ^ ((w as u64) << 32 | h as u64);
        let rows = y0 * width..(y1 + 1) * width;
        for (plane, &bg) in planes.iter().zip(&bg) {
            let plane_rows = plane[rows.clone()].chunks_exact(width);
            let label_rows = label[rows.clone()].chunks_exact(width);
            for (values, owners) in plane_rows.zip(label_rows) {
                for (&value, &owner) in values[x0..=x1].iter().zip(&owners[x0..=x1]) {
                    let v = if owner == next_label { value - bg } else { 0 };
                    hash = (hash ^ (v as u32 as u64)).wrapping_mul(0xff51_afd7_ed55_8ccd);
                    hash ^= hash >> 29;
                }
            }
        }
        shapes.push(Shape {
            x0,
            y0,
            w,
            h,
            label: next_label,
            bg,
            hash,
        });
    }
    if shapes.len() < 4 {
        return None;
    }
    let diff = |s: &Shape, c: usize, dx: usize, dy: usize| -> i32 {
        let i = (s.y0 + dy) * width + s.x0 + dx;
        if label[i] == s.label {
            planes[c][i] - s.bg[c]
        } else {
            0
        }
    };
    let same = |a: &Shape, b: &Shape| -> bool {
        a.w == b.w
            && a.h == b.h
            && (0..3).all(|c| {
                (0..a.h).all(|dy| (0..a.w).all(|dx| diff(a, c, dx, dy) == diff(b, c, dx, dy)))
            })
    };
    let mut buckets: HashMap<u64, Vec<Vec<usize>>> = HashMap::new();
    for (i, s) in shapes.iter().enumerate() {
        let groups = buckets.entry(s.hash).or_default();
        match groups.iter_mut().find(|g| same(&shapes[g[0]], s)) {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let mut groups: Vec<Vec<usize>> = buckets
        .into_values()
        .flatten()
        .filter(|g| {
            let s = &shapes[g[0]];
            g.len() >= params.min_occurrences
                && (g.len().max(2) - 1) * s.w * s.h >= params.min_repeat_pixels
        })
        .collect();
    if groups.len() < 2 {
        return None;
    }
    // Most valuable first; the atlas is one 1024-pixel modular group, so the
    // tail is dropped when the shapes do not fit.
    groups.sort_by_key(|g| {
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
    if (covered as f64) < GLYPH_MIN_COVER * (width * height) as f64 {
        return None;
    }
    groups.sort_by_key(|g| {
        let s = &shapes[g[0]];
        (
            std::cmp::Reverse(s.h),
            std::cmp::Reverse(g.len() * s.w * s.h),
            s.y0,
            s.x0,
        )
    });
    let atlas_width = ((total_area as f64).sqrt() as usize * 3 / 2)
        .max(max_width)
        .clamp(16, GLYPH_ATLAS_MAX_WIDTH);
    let mut slots: Vec<(usize, usize)> = Vec::with_capacity(groups.len());
    let (mut cx, mut cy, mut shelf_h) = (0usize, 0usize, 0usize);
    for g in &groups {
        let s = &shapes[g[0]];
        if cx + s.w > atlas_width {
            cx = 0;
            cy += shelf_h;
            shelf_h = 0;
        }
        slots.push((cx, cy));
        cx += s.w;
        shelf_h = shelf_h.max(s.h);
    }
    let atlas_height = cy + shelf_h;
    if atlas_height > GLYPH_ATLAS_MAX_HEIGHT {
        return None;
    }
    let mut atlas = [
        vec![0i32; atlas_width * atlas_height],
        vec![0i32; atlas_width * atlas_height],
        vec![0i32; atlas_width * atlas_height],
    ];
    // Plane order of `planes` is Y, X, B; image planes are X, Y, B.
    let steps = [
        1.0 / INV_DC_QUANT[1],
        1.0 / INV_DC_QUANT[0],
        1.0 / INV_DC_QUANT[2],
    ];
    let image_plane = [1usize, 0, 2];
    let mut base = xyb.clone();
    let mut references = Vec::with_capacity(groups.len());
    for (g, (ax, ay)) in groups.into_iter().zip(slots) {
        let s = &shapes[g[0]];
        for dy in 0..s.h {
            for dx in 0..s.w {
                let at = (ay + dy) * atlas_width + ax + dx;
                let d = [diff(s, 0, dx, dy), diff(s, 1, dx, dy), diff(s, 2, dx, dy)];
                atlas[0][at] = d[0];
                atlas[1][at] = d[1];
                atlas[2][at] = d[2] - d[0];
            }
        }
        let mut positions = Vec::with_capacity(g.len());
        for &i in &g {
            let o = &shapes[i];
            positions.push((o.x0, o.y0));
            for c in 0..3 {
                for dy in 0..o.h {
                    let row = base.plane_row_mut(image_plane[c], o.y0 + dy);
                    for dx in 0..o.w {
                        let d = diff(o, c, dx, dy);
                        if d != 0 {
                            row[o.x0 + dx] -= d as f32 * steps[c];
                        }
                    }
                }
            }
        }
        positions.sort_unstable_by_key(|&(x, y)| (y, x));
        references.push(PatchReference {
            atlas_x: ax,
            atlas_y: ay,
            width: s.w,
            height: s.h,
            ref_frame: MODULAR_PATCH_REF_ID,
            add: true,
            positions,
        });
    }
    Some(LossyGlyphPatches {
        atlas,
        atlas_width,
        atlas_height,
        base,
        references,
    })
}

// Both patch searches use this ordering; share the comparator's sort code.
fn sort_patch_groups(groups: &mut [Vec<(usize, usize)>]) {
    groups.sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn near_patch_plan_is_identical_across_thread_counts() {
        let tile = 16;
        let mut img = Image3F::new(1024, 1024);
        for ty in 0..64 {
            for tx in 0..64 {
                let id = ty * 64 + tx;
                for c in 0..3 {
                    for y in 0..tile {
                        for x in 0..tile {
                            img.plane_row_mut(c, ty * tile + y)[tx * tile + x] =
                                ((x * 7 + y * 3 + id % 23) % 17) as f32 * 0.025
                                    + (id % 127) as f32 * NEAR_TILE_UNIT[c] / 254.0;
                        }
                    }
                }
            }
        }
        let mut scratch = CoderScratch::default();
        let serial =
            find_lossy_patches_sized(&img, tile, 1.0, &ThreadPool::new(1), &mut scratch).unwrap();
        let parallel =
            find_lossy_patches_sized(&img, tile, 1.0, &ThreadPool::new(4), &mut scratch).unwrap();
        assert_eq!(serial.groups, parallel.groups);
        for c in 0..3 {
            assert_eq!(serial.base.plane_data(c), parallel.base.plane_data(c));
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
                                (img.plane_row(c, sy + y)[sx + x]
                                    - img.plane_row(c, py + y)[px + x])
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
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            }
        }
    }

    fn assert_glyph_reconstruction(img: &Image3Si, plan: &LosslessPatches) {
        let mut restored = plan.base.clone();
        for r in &plan.references {
            for &(x, y) in &r.positions {
                for c in 0..3 {
                    for dy in 0..r.height {
                        restored.plane_row_mut(c, y + dy)[x..x + r.width].copy_from_slice(
                            &plan.atlas.plane_row(c, r.atlas_y + dy)
                                [r.atlas_x..r.atlas_x + r.width],
                        );
                    }
                }
            }
        }
        for c in 0..3 {
            assert_eq!(restored.plane_data(c), img.plane_data(c));
        }
    }

    #[test]
    fn glyph_patches_preserve_signed_pixels_at_image_and_tile_edges() {
        let mut img = Image3Si::new(131, 99);
        let background = [-17, 255, -(1 << 20)];
        for (c, value) in background.into_iter().enumerate() {
            img.plane_mut(c).as_mut_slice().fill(value);
        }
        let diagonals = [(0, 0), (29, 25), (124, 92)];
        let corners = [(126, 0), (61, 59), (0, 90)];
        for (c, value) in [-(1 << 20), (1 << 20) - 1, 0].into_iter().enumerate() {
            for &(x, y) in &diagonals {
                // Only diagonal neighbors connect these seven pixels.
                for d in 0..7 {
                    img.plane_row_mut(c, y + d)[x + d] = value;
                }
            }
            for &(x, y) in &corners {
                for dy in 0..9 {
                    img.plane_row_mut(c, y + dy)[x] = value;
                }
                img.plane_row_mut(c, y + 8)[x..x + 5].fill(value);
            }
        }
        let plan = find_lossless_glyph_patches(&img).expect("two repeated glyphs");
        assert_eq!(plan.references.len(), 2);
        for r in &plan.references {
            match (r.width, r.height) {
                (7, 7) => assert_eq!(r.positions, diagonals),
                (5, 9) => assert_eq!(r.positions, corners),
                dimensions => panic!("unexpected glyph box: {dimensions:?}"),
            }
        }
        for (c, value) in background.into_iter().enumerate() {
            assert!(plan.base.plane_data(c).iter().all(|&v| v == value));
        }
        assert_glyph_reconstruction(&img, &plan);
    }

    #[test]
    fn glyph_patches_preserve_neighboring_background_colors() {
        let mut img = Image3Si::new(99, 99);
        // Nine different tile modes exercise the complete neighborhood.
        for c in 0..3 {
            for y in 0..img.ysize() {
                for (x, pixel) in img.plane_row_mut(c, y).iter_mut().enumerate() {
                    *pixel = ((y / 32).min(2) * 3 + (x / 32).min(2) + c * 10) as i32;
                }
            }
        }
        for (w, h, positions) in [
            (5, 7, [(2, 3), (35, 4), (69, 5)]),
            (3, 9, [(3, 68), (36, 69), (70, 70)]),
        ] {
            for (x, y) in positions {
                for c in 0..3 {
                    for dy in 0..h {
                        img.plane_row_mut(c, y + dy)[x..x + w].fill(-100 - c as i32);
                    }
                }
            }
        }
        // This color belongs to a diagonal neighbor. It is background,
        // even though these repeated boxes are surrounded by another mode.
        for (x, y) in [(35, 35), (49, 49)] {
            for c in 0..3 {
                for dy in 0..7 {
                    img.plane_row_mut(c, y + dy)[x..x + 5].fill((c * 10) as i32);
                }
            }
        }
        let plan = find_lossless_glyph_patches(&img).expect("two repeated glyphs");
        assert_eq!(plan.references.len(), 2);
        assert!(plan.references.iter().all(|r| r.positions.len() == 3));
        assert_glyph_reconstruction(&img, &plan);
    }

    /// Six distinct 16x16 glyphs, each repeated exactly six times (comfortably
    /// past MIN_PATCH_OCCURRENCES), so every group has the same count and only
    /// the tie-break can order them.
    fn glyph_params(min_occurrences: usize) -> GlyphParams {
        GlyphParams {
            min_occurrences,
            min_repeat_pixels: 0,
            coarse: 1,
        }
    }

    /// A page of two flat backgrounds with two antialiased marks stamped at
    /// unaligned positions on both, plus one unrepeated mark. Every value sits
    /// on the XYB lattice so a mark differs between backgrounds only by place.
    fn glyph_page() -> Image3F {
        let (w, h) = (192usize, 128usize);
        let mut img = Image3F::new(w, h);
        for y in 0..h {
            let bg = if y < 64 {
                [4.0f32 / 4096.0, 307.0 / 512.0, 140.0 / 256.0]
            } else {
                [-8.0 / 4096.0, 102.0 / 512.0, 64.0 / 256.0]
            };
            for (c, &value) in bg.iter().enumerate() {
                img.plane_row_mut(c, y)[..w].fill(value);
            }
        }
        let mut stamp = |x0: usize, y0: usize, w: usize, h: usize, unit: f32| {
            for dy in 0..h {
                for dx in 0..w {
                    let ink = ((dx * 3 + dy * 5 + w) % 4) as f32 * unit;
                    img.plane_row_mut(1, y0 + dy)[x0 + dx] -= ink * 16.0 / 512.0;
                    img.plane_row_mut(2, y0 + dy)[x0 + dx] -= ink * 4.0 / 256.0;
                    img.plane_row_mut(0, y0 + dy)[x0 + dx] += ink * 2.0 / 4096.0;
                }
            }
        };
        for &(x, y) in &[(3, 5), (41, 9), (77, 22), (130, 37), (11, 75), (99, 101)] {
            stamp(x, y, 5, 7, 2.0);
        }
        for &(x, y) in &[(20, 40), (160, 12), (60, 90), (140, 110)] {
            stamp(x, y, 4, 4, 2.0);
        }
        stamp(150, 80, 6, 6, 1.0);
        img
    }

    /// kAdd semantics: base + decoded patches must give the image back up to
    /// float rounding, wherever the patches land.
    fn assert_glyph_plan_reconstructs(img: &Image3F, plan: &LossyGlyphPatches) {
        use crate::quant_weights::INV_DC_QUANT;
        let mut recon = plan.base.clone();
        let [ay, ax, ab] = &plan.atlas;
        for r in &plan.references {
            assert!(r.add);
            for &(px, py) in &r.positions {
                for dy in 0..r.height {
                    for dx in 0..r.width {
                        let at = (r.atlas_y + dy) * plan.atlas_width + r.atlas_x + dx;
                        recon.plane_row_mut(1, py + dy)[px + dx] += ay[at] as f32 / INV_DC_QUANT[1];
                        recon.plane_row_mut(0, py + dy)[px + dx] += ax[at] as f32 / INV_DC_QUANT[0];
                        recon.plane_row_mut(2, py + dy)[px + dx] +=
                            (ab[at] + ay[at]) as f32 / INV_DC_QUANT[2];
                    }
                }
            }
        }
        for c in 0..3 {
            for y in 0..img.ysize() {
                for (a, b) in img.plane_row(c, y)[..img.xsize()]
                    .iter()
                    .zip(&recon.plane_row(c, y)[..img.xsize()])
                {
                    assert!((a - b).abs() < 1e-5, "plane {c} row {y}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn lossy_glyph_patches_reconstruct_and_flatten_the_base() {
        let img = glyph_page();
        let repeated = find_lossy_glyph_patches(&img, &glyph_params(2)).expect("plan");
        // The mark differs between the two backgrounds only by where it sits.
        assert_eq!(repeated.references.len(), 2);
        let mut counts: Vec<usize> = repeated
            .references
            .iter()
            .map(|r| r.positions.len())
            .collect();
        counts.sort_unstable();
        assert_eq!(counts, [4, 6]);
        assert_glyph_plan_reconstructs(&img, &repeated);
        // Under the repeated marks the base is the background again, to within
        // the lattice rounding of the patch.
        let step = 1.0 / crate::quant_weights::INV_DC_QUANT[1];
        assert!((repeated.base.plane_row(1, 8)[5] - 307.0 / 512.0).abs() <= step);

        let all = find_lossy_glyph_patches(&img, &glyph_params(1)).expect("plan");
        assert_eq!(all.references.len(), 3);
        assert_glyph_plan_reconstructs(&img, &all);
    }

    #[test]
    fn near_tiles_join_within_the_tolerance_only() {
        let pool = ThreadPool::new(2);
        let mut scratch = CoderScratch::default();
        let tile = PATCH_TILE;
        let mut img = Image3F::new(8 * tile, 2 * tile);
        let mut state = 7u32;
        // Top row: six copies of one busy tile, three of them nudged by a
        // third of the tolerance; bottom row: unrelated noise.
        let mut proto = vec![0f32; tile * tile];
        for v in proto.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (state >> 24) as f32 / 255.0 * 0.5;
        }
        for ty in 0..2 {
            for tx in 0..8 {
                for dy in 0..tile {
                    for dx in 0..tile {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let noise = (state >> 24) as f32 / 255.0 * 0.5;
                        let value = if ty == 0 && tx < 6 {
                            proto[dy * tile + dx]
                                + if tx >= 3 && (dx + dy) % 2 == 0 {
                                    NEAR_TILE_UNIT[1] / 3.0
                                } else {
                                    0.0
                                }
                        } else {
                            noise
                        };
                        img.plane_row_mut(1, ty * tile + dy)[tx * tile + dx] = value;
                        img.plane_row_mut(2, ty * tile + dy)[tx * tile + dx] = value;
                    }
                }
            }
        }
        assert!(find_lossy_patches_sized(&img, tile, 0.0, &pool, &mut scratch).is_none());
        let plan = find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).expect("near");
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].len(), 6);
        assert_eq!(plan.groups[0][0], (0, 0));
        // Hash maps are reseeded per instance; the plan must not notice.
        for _ in 0..4 {
            let again =
                find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).expect("near");
            assert_eq!(again.groups, plan.groups);
        }
        // The tile is noise, i.e. textured: one sample may sit a few tolerances
        // off as long as the RMS bound holds ...
        img.plane_row_mut(1, 3)[4 * tile + 3] += NEAR_TILE_UNIT[1] * 3.0;
        let plan = find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).expect("near");
        assert_eq!(plan.groups[0].len(), 6);
        // ... but not past the textured peak.

        img.plane_row_mut(1, 3)[5 * tile + 3] += NEAR_TILE_UNIT[1] * 6.0;
        let plan = find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).expect("near");
        assert_eq!(plan.groups[0].len(), 5);
    }

    #[test]
    fn flat_tiles_keep_the_strict_near_peak() {
        let pool = ThreadPool::new(2);
        let mut scratch = CoderScratch::default();
        let tile = PATCH_TILE;
        // Six copies of a two-tone (UI-like, mostly flat) tile; one carries a
        // single sample three tolerances off, which texture would have hidden.
        let mut img = Image3F::new(6 * tile, tile);
        for tx in 0..6 {
            for c in 1..3 {
                for dy in 0..tile {
                    let row = img.plane_row_mut(c, dy);
                    row[tx * tile..tx * tile + tile / 2].fill(0.2);
                    row[tx * tile + tile / 2..(tx + 1) * tile].fill(0.6);
                }
            }
        }
        img.plane_row_mut(1, 5)[5 * tile + 2] += NEAR_TILE_UNIT[1] * 3.0;
        let plan = find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).expect("exact");
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].len(), 5);
        // Within the strict peak it does join.
        img.plane_row_mut(1, 5)[5 * tile + 2] -= NEAR_TILE_UNIT[1] * 2.5;
        let plan = find_lossy_patches_sized(&img, tile, 1.0, &pool, &mut scratch).expect("near");
        assert_eq!(plan.groups[0].len(), 6);
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
            add: false,
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

    #[test]
    fn atlas_order_keeps_every_group_and_chains_neighbors() {
        let tile = PATCH_TILE;
        // Tile brightness 0.9, 0.1, 0.8, 0.2: the chain from the first tile
        // must visit 0.8 before the dark pair.
        let mut img = Image3F::new(4 * tile, tile);
        for (i, level) in [0.9f32, 0.1, 0.8, 0.2].into_iter().enumerate() {
            for c in 0..3 {
                for y in 0..tile {
                    img.plane_row_mut(c, y)[i * tile..(i + 1) * tile].fill(level);
                }
            }
        }
        let groups: Vec<Vec<(usize, usize)>> = (0..4).map(|i| vec![(i * tile, 0)]).collect();
        assert_eq!(similarity_order(&img, &groups, tile), [0, 2, 3, 1]);
        let (_, refs) = pack_lossy_atlas(&img, groups, PATCH_REF_ID);
        let firsts: Vec<usize> = refs.iter().map(|r| r.positions[0].0 / tile).collect();
        assert_eq!(firsts, [0, 2, 3, 1]);
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
        packed.sort_unstable();
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
