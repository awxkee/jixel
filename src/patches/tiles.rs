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
use super::LossyPatches;
#[cfg(test)]
use super::PATCH_TILE;
use crate::coder_scratch::CoderScratch;
use crate::image::Image3F;
use crate::shared_sort::SharedSort;
use crate::thread_pool::ThreadPool;
use std::collections::HashMap;

mod atlas;

pub(crate) use atlas::{
    pack_lossy_atlas, pack_lossy_atlas_sized, reuse_tile_quadrants, stack_atlases,
};

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
    let groups = find_lossy_patch_groups_sized(linear, tile, near, pool, scratch)?;
    let base = tile_patch_residual(linear, tile, &groups, pool, scratch);
    Some(LossyPatches { base, groups })
}

/// Detect groups without allocating a full-frame residual. The caller can
/// check coverage and pack atlases before clearing tiles in its owned image.
pub(crate) fn find_lossy_patch_groups_sized(
    linear: &Image3F,
    tile: usize,
    near: f32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<Vec<Vec<(usize, usize)>>> {
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
    let min_occurrences = MIN_PATCH_OCCURRENCES;
    // Hash buckets are independent, but every candidate still needs the full
    // equality check: a hash collision must never substitute unequal pixels.
    // The map keeps bucket order and each bucket keeps raster-order leaders.
    let exact = pool.steal_map(scratch, candidates.len(), |i, _| {
        let mut exact_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        for &pos in &candidates[i] {
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
        exact_groups.retain(|g| g.len() >= min_occurrences);
        exact_groups
    });
    let mut groups: Vec<_> = exact.into_iter().flatten().collect();

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
        let rcp_tol1 = 1.0 / (16.0 * tol[1]);
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
                    means[cy * 2 + cx] = sum / (cell * cell) as f32 * rcp_tol1;
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
        g[1..].shared_sort_unstable_by_key(|&(x, y)| (y, x));
    }
    Some(groups)
}

/// Apply a detected plan after its atlas has been copied from the source.
pub(crate) fn clear_patch_tiles(base: &mut Image3F, tile: usize, groups: &[Vec<(usize, usize)>]) {
    for positions in groups {
        for &(x, y) in positions {
            for c in 0..3 {
                for dy in 0..tile {
                    base.plane_row_mut(c, y + dy)[x..x + tile].fill(0.0);
                }
            }
        }
    }
}

/// Copy only the uncovered spans into an initially zero residual. Copying the
/// whole image and then clearing occurrences in group order writes patched
/// pixels twice and jumps between distant rows.
fn tile_patch_residual(
    linear: &Image3F,
    tile: usize,
    groups: &[Vec<(usize, usize)>],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Image3F {
    let (width, height) = (linear.xsize(), linear.ysize());
    let (tiles_x, tiles_y) = (width / tile, height / tile);
    let mut patched = vec![false; tiles_x * tiles_y];
    for positions in groups {
        for &(x, y) in positions {
            patched[y / tile * tiles_x + x / tile] = true;
        }
    }
    let spans: Vec<Vec<_>> = (0..height.div_ceil(tile))
        .map(|ty| {
            let mut spans = Vec::new();
            let mut start = 0;
            if ty < tiles_y {
                for tx in 0..tiles_x {
                    if patched[ty * tiles_x + tx] {
                        let x = tx * tile;
                        if start < x {
                            spans.push(start..x);
                        }
                        start = x + tile;
                    }
                }
            }
            if start < width {
                spans.push(start..width);
            }
            spans
        })
        .collect();
    let mut base = Image3F::new(width, height);
    let num_bands = (pool.num_threads() * 4).min(height);
    let band_height = height.div_ceil(num_bands);
    let mut bands = base.row_bands_mut(num_bands);
    pool.steal_for_each_mut(scratch, &mut bands, |i, planes, _| {
        let y0 = i * band_height;
        for (c, plane) in planes.iter_mut().enumerate() {
            for (dy, row) in plane.chunks_exact_mut(width).enumerate() {
                let y = y0 + dy;
                let source = linear.plane_row(c, y);
                for span in &spans[y / tile] {
                    row[span.clone()].copy_from_slice(&source[span.clone()]);
                }
            }
        }
    });
    base
}

// Both patch searches use this ordering; share the comparator's sort code.
pub(super) fn sort_patch_groups(groups: &mut [Vec<(usize, usize)>]) {
    groups.shared_sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
}

#[cfg(test)]
mod tests;
