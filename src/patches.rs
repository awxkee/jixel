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
use crate::encoding_context::EncodingContext;
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
    Patched(&'a [PatchReference], Option<&'a EpfPatchAtlases>),
}

pub(crate) struct GlyphAtlasLuma {
    pub(crate) width: usize,
    pub(crate) samples: Vec<i32>,
    /// 8-bit coverage of blend-above template tiles (atlas alpha).
    pub(crate) coverage: Option<Vec<u8>>,
}

pub(crate) struct EpfPatchAtlases {
    pub(crate) glyphs: std::sync::Arc<GlyphAtlasLuma>,
    /// The source luma of the finer Replace atlas, used for both sides of the
    /// comparison because its reconstruction is independent of the base EPF.
    pub(crate) tiles: Option<crate::image::Plane<f32>>,
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
        g[1..].sort_unstable_by_key(|&(x, y)| (y, x));
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
            alpha_only: false,
            add: false,
            mul: false,
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
/// Reference slot for the star-dot template atlas.
#[cfg(feature = "splines")]
pub(crate) const DOT_PATCH_REF_ID: u32 = 1;
pub(crate) const NUM_PATCH_CONTEXTS: usize = 10;

#[derive(Clone)]
pub(crate) struct PatchReference {
    pub(crate) atlas_x: usize,
    pub(crate) atlas_y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Which saved reference frame this entry copies from.
    pub(crate) ref_frame: u32,
    /// Replace alpha from the atlas while leaving color untouched.
    pub(crate) alpha_only: bool,
    /// Blend with kAdd instead of kReplace.
    pub(crate) add: bool,
    /// Blend with kMul (the entry scales what is already there); `add` is
    /// then the hint that the entry is not a Replace rectangle. `mul`
    /// without `add` is kBlendAbove: the atlas color is mixed over the frame
    /// by the atlas alpha (glyph templates; needs an alpha channel).
    pub(crate) mul: bool,
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
        slot: u32,
    },
    /// Ink frame of an `xyb_encoded` codestream, alpha-blended (kBlend)
    /// over reference slot 0: channels are Y, X, B-Y on the default LF
    /// dequant lattice plus the coverage alpha (see `EncodeConfig::ink_layer`).
    XybBlend {
        /// False when further ink frames are stacked on this one: the
        /// blended canvas is saved back to reference slot 0.
        last: bool,
    },
    /// First frame of a lossless layered codestream (see
    /// `EncodeConfig::vector_layers`): the palette base, saved to reference
    /// slot 0 and not displayed on its own.
    RgbSavedFirst,
    /// Last frame of a lossless layered codestream: the coverage frame,
    /// alpha-blended (kBlend) over reference slot 0.
    RgbBlendLast,
    /// A region frame (see `region_frames`): a crop of the image at
    /// (`x0`, `y0`) replacing that rectangle of reference slot 0, saved back
    /// to the slot unless it is the last frame.
    RgbCrop {
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        last: bool,
    },
    /// A crop frame of an `xyb_encoded` codestream (see `crop_frames`): the
    /// crop at (`x0`, `y0`) on the XYB lattice of the VarDCT base, replacing
    /// that rectangle of reference slot 0, saved back to the slot unless it
    /// is the last frame.
    XybCrop {
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        last: bool,
    },
    /// The last frame of a cell-layer codestream (see `cell_layer`): a crop
    /// at (`x0`, `y0`) of `width` x `height` image pixels, coded at that
    /// size over the frame's upsampling factor, alpha-blended (kBlend with
    /// extra channel 0) over reference slot 0.
    RgbBlendCrop {
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    },
    Patched {
        references: &'a [PatchReference],
        /// The frame is the base of region frames: saved to reference slot
        /// 0 and not the last frame.
        saved_first: bool,
    },
}

impl ModularFrameKind<'_> {
    pub(crate) fn is_regular(self) -> bool {
        matches!(
            self,
            Self::Regular
                | Self::RgbSavedFirst
                | Self::RgbBlendLast
                | Self::RgbCrop { .. }
                | Self::XybCrop { .. }
                | Self::RgbBlendCrop { .. }
        )
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
fn round_to_multiple(v: i32, k: i32) -> i32 {
    let k = k.unsigned_abs();
    let r = ((v.unsigned_abs() + k / 2) / k * k) as i32;
    if v < 0 { -r } else { r }
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

const GLYPH_MAX_INK_SHARE: f64 = 0.6;
const GLYPH_SCREEN_STRIDE: usize = 64;
const GLYPH_SCREEN_MAX_INK: f64 = 0.6;
/// The lossy glyph atlas is a multi-group modular frame; only keep it sane.
const GLYPH_ATLAS_MAX_HEIGHT: usize = 4096;

/// Glyph patches for the lossy path, libjxl style: connected ink components
/// over a flat background, stored as the difference from that background on
/// the modular XYB lattice and blended with kAdd. The base keeps the image
/// minus the decoded patch, i.e. the flat background plus lattice rounding.
pub(crate) struct LossyGlyphPatches<Base> {
    /// Lattice channels Y, X, B-Y of the atlas.
    pub(crate) atlas: [Vec<i32>; 3],
    pub(crate) atlas_width: usize,
    pub(crate) atlas_height: usize,
    pub(crate) base: Base,
    pub(crate) references: Vec<PatchReference>,
    /// 8-bit atlas alpha: template coverage, zero under the kAdd entries.
    /// `None` when no template was placed.
    pub(crate) alpha_atlas: Option<Vec<u8>>,
}

impl LossyGlyphPatches<Option<Image3F>> {
    /// Materialize the additive residual after the atlas has been coded.
    /// The template base, if present, already has its blend-above patches
    /// removed. Additive placements keep their original order until the
    /// subtraction finishes, since their bounding boxes can overlap.
    pub(crate) fn materialize(self, xyb: &Image3F) -> LossyGlyphPatches<Image3F> {
        use crate::quant_weights::INV_DC_QUANT;

        let Self {
            atlas,
            atlas_width,
            atlas_height,
            base,
            mut references,
            alpha_atlas,
        } = self;
        let mut base = base.unwrap_or_else(|| xyb.clone());
        // Atlas order is Y, X, B-Y, while the image planes are X, Y, B.
        let image_planes = [1usize, 0, 2];
        let steps = image_planes.map(|c| 1.0 / INV_DC_QUANT[c]);
        for r in references.iter_mut().filter(|r| r.add) {
            for &(x, y) in &r.positions {
                for c in 0..3 {
                    for dy in 0..r.height {
                        let row = base.plane_row_mut(image_planes[c], y + dy);
                        let start = (r.atlas_y + dy) * atlas_width + r.atlas_x;
                        let pixels = row[x..][..r.width]
                            .iter_mut()
                            .zip(&atlas[c][start..][..r.width]);
                        for (dx, (pixel, &sample)) in pixels.enumerate() {
                            let d = if c == 2 {
                                sample + atlas[0][start + dx]
                            } else {
                                sample
                            };
                            if d != 0 {
                                *pixel -= d as f32 * steps[c];
                            }
                        }
                    }
                }
            }
            r.positions.sort_unstable_by_key(|&(x, y)| (y, x));
        }
        LossyGlyphPatches {
            atlas,
            atlas_width,
            atlas_height,
            base,
            references,
            alpha_atlas,
        }
    }
}

pub(crate) struct GlyphParams {
    /// Share glyphs recurring over different backgrounds as blend-above
    /// templates; the frame must carry an 8-bit alpha channel.
    pub(crate) templates: bool,
    pub(crate) min_occurrences: usize,
    /// (occurrences - 1) * box pixels a group must reach.
    pub(crate) min_repeat_pixels: usize,
    /// Lattice values are rounded to multiples of this.
    pub(crate) coarse: i32,
    /// A shape whose largest lattice difference from its background is
    /// below this is left to VarDCT.
    pub(crate) min_contrast: i32,
    /// Unrepeated groups are kept only when together they cover this share
    /// of the frame (0 keeps them unconditionally).
    pub(crate) singles_min_cover: f64,
}

/// Share of the frame the glyph boxes must cover.
const GLYPH_MIN_COVER: f64 = 0.005;
/// Lattice step of the stored glyph values.
const GLYPH_ATLAS_STEP: i32 = 6;

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
/// Lattice granularity (Y, X, B) of the foreground colors one template shares.
static TEMPLATE_COLOR_STEP: [f32; 3] = [16.0, 64.0, 16.0];

struct GlyphShape {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    label: u32,
    bg: [i32; 3],
    hash: u64,
}

/// Template atlas, placements and base of the glyphs taken by templates.
struct GlyphTemplates {
    atlas: [Vec<i32>; 3],
    alpha: Vec<u8>,
    width: usize,
    height: usize,
    base: Image3F,
    references: Vec<PatchReference>,
}

impl GlyphTemplates {
    fn into_patches(self) -> LossyGlyphPatches<Option<Image3F>> {
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

pub(crate) fn plan_lossy_glyph_patches(
    ctx: &EncodingContext,
    xyb: &Image3F,
    params: &GlyphParams,
) -> Option<LossyGlyphPatches<Option<Image3F>>> {
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
                    let r = round_to_multiple;
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
    // Work with the rounded B itself; the atlas goes back to B-Y at the end.
    for (b, &y) in qb.iter_mut().zip(&qy) {
        *b += y;
    }
    let fine = [qy, qx, qb];
    // Full-frame color keys use the same rounding kernel on bounded chunks.
    // Matching reads coarse samples only within glyph boxes.
    debug_assert!(params.coarse > 0 && (params.coarse as u32).is_power_of_two());
    let shift = params.coarse.trailing_zeros();
    let coarse_value = |v| crate::xyb::round_lattice_pow2_value(v, shift);
    let [qy, qx, qb] = &fine;
    let mut keys = Vec::with_capacity(width * height);
    if shift == 0 {
        keys.extend(
            qy.iter()
                .zip(qx)
                .zip(qb)
                .map(|((&y, &x), &b)| color_key(y, x, b)),
        );
    } else {
        const BLOCK: usize = 512;
        let mut rounded = [[0; BLOCK]; 3];
        for ((y, x), b) in qy.chunks(BLOCK).zip(qx.chunks(BLOCK)).zip(qb.chunks(BLOCK)) {
            let len = y.len();
            for (dest, src) in rounded.each_mut().into_iter().zip([y, x, b]) {
                let dest = &mut dest[..len];
                dest.copy_from_slice(src);
                // SAFETY: The context selects a supported rounding kernel.
                unsafe { (ctx.round_lattice_pow2)(dest, shift) };
            }
            let [y, x, b] = &rounded;
            keys.extend(
                y[..len]
                    .iter()
                    .zip(&x[..len])
                    .zip(&b[..len])
                    .map(|((&y, &x), &b)| color_key(y, x, b)),
            );
        }
    }

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

    let mut label = vec![0u32; width * height];
    let mut shapes = Vec::new();
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
        let mut contrast: i32 = 0;
        let rows = y0 * width..(y1 + 1) * width;
        for (plane, &bg) in fine.iter().zip(&bg) {
            let plane_rows = plane[rows.clone()].chunks_exact(width);
            let label_rows = label[rows.clone()].chunks_exact(width);
            for (values, owners) in plane_rows.zip(label_rows) {
                for (&value, &owner) in values[x0..=x1].iter().zip(&owners[x0..=x1]) {
                    let v = if owner == next_label {
                        coarse_value(value) - bg
                    } else {
                        0
                    };
                    contrast = contrast.max(v.abs());
                    hash = (hash ^ (v as u32 as u64)).wrapping_mul(0xff51_afd7_ed55_8ccd);
                    hash ^= hash >> 29;
                }
            }
        }
        if contrast < params.min_contrast {
            continue;
        }
        shapes.push(GlyphShape {
            x0,
            y0,
            w,
            h,
            label: next_label,
            bg,
            hash,
        });
    }
    drop((ink, keys, stack));
    if shapes.len() < 4 {
        return None;
    }
    let templates = if ctx.template_probe.is_some() || params.templates {
        let plan = plan_glyph_templates(width, &label, &fine, &shapes);
        if let (Some(probe), Some(plan)) = (&ctx.template_probe, &plan) {
            probe.fetch_max(plan.saved_area(), std::sync::atomic::Ordering::Relaxed);
        }
        plan.filter(|_| params.templates)
            .and_then(|plan| build_glyph_templates(plan, xyb))
    } else {
        None
    };
    if let Some(t) = &templates {
        let used: std::collections::HashSet<(usize, usize)> = t
            .references
            .iter()
            .flat_map(|r| r.positions.iter().copied())
            .collect();
        shapes.retain(|s| !used.contains(&(s.x0, s.y0)));
    }
    // From here on the exact path codes the remaining glyphs; where it finds
    // nothing worth an atlas, the templates stand alone.
    macro_rules! no_exact_plan {
        () => {
            return templates.map(GlyphTemplates::into_patches)
        };
    }
    let diff = |s: &GlyphShape, c: usize, dx: usize, dy: usize| -> i32 {
        let i = (s.y0 + dy) * width + s.x0 + dx;
        if label[i] == s.label {
            coarse_value(fine[c][i]) - s.bg[c]
        } else {
            0
        }
    };
    // Atlas values: the leader's own pixels on a lattice coarser than the
    // matching one. Every placement subtracts exactly these from the base.
    let atlas_step = GLYPH_ATLAS_STEP.max(params.coarse);
    let atlas_value = |s: &GlyphShape, c: usize, dx: usize, dy: usize| -> i32 {
        let i = (s.y0 + dy) * width + s.x0 + dx;
        if label[i] == s.label {
            round_to_multiple(fine[c][i] - s.bg[c], atlas_step)
        } else {
            0
        }
    };
    let same = |a: &GlyphShape, b: &GlyphShape| -> bool {
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
        .filter(|g| g.len() >= params.min_occurrences)
        .collect();
    let pays = |g: &Vec<usize>| {
        let s = &shapes[g[0]];
        (g.len().max(2) - 1) * s.w * s.h >= params.min_repeat_pixels
    };
    // Unrepeated shapes over most of the frame make it a drawing, which the
    // atlas takes whole; a few stray ones are left with the base, and the
    // repeated shapes then have to pay for themselves.
    if params.singles_min_cover > 0.0 {
        let single_cover: usize = groups
            .iter()
            .filter(|g| g.len() < 2)
            .map(|g| shapes[g[0]].w * shapes[g[0]].h)
            .sum();
        if (single_cover as f64) < params.singles_min_cover * (width * height) as f64 {
            groups.retain(|g| g.len() >= 2 && pays(g));
        }
    } else {
        groups.retain(pays);
    }
    if groups.len() < 2 {
        no_exact_plan!();
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
        no_exact_plan!();
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
    let exact_height = cy + shelf_h;
    let template_height = templates.as_ref().map_or(0, |t| t.height);
    let atlas_height = exact_height + template_height;
    if atlas_height > GLYPH_ATLAS_MAX_HEIGHT {
        no_exact_plan!();
    }
    // Fill the exact tiles straight into the bottom of the final atlas.
    let mut patches = match templates {
        Some(t) => {
            let mut patches = t.into_patches();
            let old_width = patches.atlas_width;
            let width = old_width.max(atlas_width);
            for plane in &mut patches.atlas {
                grow_atlas_plane(plane, old_width, template_height, width, atlas_height);
            }
            if let Some(alpha) = &mut patches.alpha_atlas {
                grow_atlas_plane(alpha, old_width, template_height, width, atlas_height);
            }
            patches.atlas_width = width;
            patches.atlas_height = atlas_height;
            patches.references.reserve(groups.len());
            patches
        }
        None => LossyGlyphPatches {
            atlas: std::array::from_fn(|_| vec![0; atlas_width * atlas_height]),
            atlas_width,
            atlas_height,
            base: None,
            references: Vec::with_capacity(groups.len()),
            alpha_atlas: None,
        },
    };
    let atlas_width = patches.atlas_width;

    let LossyGlyphPatches {
        atlas, references, ..
    } = &mut patches;
    for (g, (ax, ay)) in groups.into_iter().zip(slots) {
        let ay = ay + template_height;
        let s = &shapes[g[0]];
        for dy in 0..s.h {
            for dx in 0..s.w {
                let at = (ay + dy) * atlas_width + ax + dx;
                let d = [
                    atlas_value(s, 0, dx, dy),
                    atlas_value(s, 1, dx, dy),
                    atlas_value(s, 2, dx, dy),
                ];
                atlas[0][at] = d[0];
                atlas[1][at] = d[1];
                atlas[2][at] = d[2] - d[0];
            }
        }
        references.push(PatchReference {
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

/// Lattice units (`INV_DC_QUANT`) of the planes in Y, X, B order.
const TEMPLATE_LATTICE: [f32; 3] = [512.0, 4096.0, 256.0];

/// One glyph occurrence as a template member: its foreground (mean of the
/// strongest fifth of its pixels) and every pixel's coverage between its
/// background and that foreground.
struct Occurrence {
    fg: [f32; 3],
    fg_delta: [f32; 3],
    alpha: Vec<f32>,
}

/// Glyph groups that share a blend-above template (see
/// [`TEMPLATE_MIN_BASE_WEIGHT`]): same mask and foreground colour, coverage
/// that agrees, two or more backgrounds.
struct TemplatePlan<'a> {
    shapes: &'a [GlyphShape],
    occurrences: Vec<Occurrence>,
    groups: Vec<Vec<usize>>,
}

/// A template: the members' mean coverage, rounded, and mean foreground.
fn template_of(
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
        for (t, a) in alpha.iter_mut().zip(&occurrences[i].alpha) {
            *t += a * inv;
        }
        for c in 0..3 {
            fg[c] += occurrences[i].fg[c] * inv;
        }
    }
    for t in alpha {
        *t = (*t * TEMPLATE_ALPHA_LEVELS).round() / TEMPLATE_ALPHA_LEVELS;
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
                hashes.sort_unstable();
                hashes.dedup();
                let s = &self.shapes[g[0]];
                (hashes.len() - 1) * s.w * s.h
            })
            .sum()
    }
}

/// Template groups for the glyph shapes. `fine` holds the lattice planes
/// Y, X, B; `label` the component of every pixel. `None` when no group
/// shares a template.
fn plan_glyph_templates<'a>(
    width: usize,
    label: &[u32],
    fine: &[Vec<i32>; 3],
    shapes: &'a [GlyphShape],
) -> Option<TemplatePlan<'a>> {
    let lattice = TEMPLATE_LATTICE;
    // Weights that make a lattice step of each channel comparable.
    let weights = [64.0f32, 4096.0, 16.0];
    let delta = |i: usize, bg: [i32; 3]| -> [f32; 3] {
        std::array::from_fn(|c| (fine[c][i] - bg[c]) as f32 / lattice[c])
    };
    let mut strength: Vec<(f32, usize)> = Vec::new();
    let mut occurrences = Vec::with_capacity(shapes.len());
    let mut buckets: HashMap<(u64, [i32; 3]), Vec<usize>> = HashMap::new();
    for (i, s) in shapes.iter().enumerate() {
        strength.clear();
        let mut mask_hash = 0x9e37_79b9_7f4a_7c15 ^ ((s.w as u64) << 32 | s.h as u64);
        for dy in 0..s.h {
            let start = (s.y0 + dy) * width + s.x0;
            for (dx, &owner) in label[start..][..s.w].iter().enumerate() {
                let inside = owner == s.label;
                mask_hash = (mask_hash ^ u64::from(inside)).wrapping_mul(0xff51_afd7_ed55_8ccd);
                mask_hash ^= mask_hash >> 29;
                if inside {
                    let idx = start + dx;
                    let d = delta(idx, s.bg);
                    strength.push(((0..3).map(|c| weights[c] * d[c] * d[c]).sum(), idx));
                }
            }
        }
        let keep = (strength.len() / 5).max(1);
        // Keep the row-major order of tied pixels from the full stable sort.
        let by_strength =
            |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
        strength.select_nth_unstable_by(keep - 1, by_strength);
        strength[..keep - 1].sort_unstable_by(by_strength);
        let mut fd = [0.0f32; 3];
        for &(_, idx) in &strength[..keep] {
            let d = delta(idx, s.bg);
            for c in 0..3 {
                fd[c] += d[c] / keep as f32;
            }
        }
        let fg = std::array::from_fn(|c| s.bg[c] as f32 / lattice[c] + fd[c]);
        occurrences.push(Occurrence {
            fg,
            fg_delta: fd,
            alpha: Vec::new(),
        });
        let key: [i32; 3] =
            std::array::from_fn(|c| (fg[c] * lattice[c] / TEMPLATE_COLOR_STEP[c]).round() as i32);
        buckets.entry((mask_hash, key)).or_default().push(i);
    }
    drop(strength);
    // A member fits when its coverage agrees with the template and its
    // cores, which the decoder shows as the template's mix, stay close.
    let fits = |s: &GlyphShape, coverage: &[f32], alpha: &[f32], fg: &[f32; 3]| -> bool {
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
        if count > 0 && diff > TEMPLATE_MAX_ALPHA_DIFF * count as f32 {
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
                    let mix = bg[c] as f32 * (1.0 - a) + fg[c] * lattice[c] * a;
                    if (fine[c][idx] as f32 - mix).abs() > TEMPLATE_CORE_TOLERANCE[c] {
                        return false;
                    }
                }
            }
        }
        true
    };
    // Shapes with one support and colour may still differ inside (coverage):
    // split each bucket into clusters whose coverage agrees with their seed.
    let agrees = |a: &[f32], b: &[f32]| -> bool {
        let (mut diff, mut count) = (0.0f32, 0usize);
        for (x, y) in a.iter().zip(b) {
            if *x > 0.0 || *y > 0.0 {
                diff += (x - y).abs();
                count += 1;
            }
        }
        diff <= TEMPLATE_MAX_ALPHA_DIFF * count.max(1) as f32
    };
    let has_multiple_backgrounds = |group: &[usize]| {
        group
            .split_first()
            .is_some_and(|(&first, rest)| rest.iter().any(|&i| shapes[i].bg != shapes[first].bg))
    };
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut groups = Vec::new();
    let mut template_alpha = Vec::new();
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
                        let d = delta(idx, s.bg);
                        let coverage =
                            (0..3).map(|c| weights[c] * d[c] * fd[c]).sum::<f32>() / norm;
                        *a = coverage.clamp(0.0, 1.0);
                    }
                }
            }
            match clusters
                .iter_mut()
                .find(|c| agrees(&occurrences[c[0]].alpha, &alpha))
            {
                Some(cluster) => cluster.push(i),
                None => clusters.push(vec![i]),
            }
            occurrences[i].alpha = alpha;
        }
        for mut group in clusters.drain(..) {
            for _ in 0..2 {
                if !has_multiple_backgrounds(&group) {
                    break;
                }
                let fg = template_of(shapes, &occurrences, &group, &mut template_alpha);
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
    (!groups.is_empty()).then_some(TemplatePlan {
        shapes,
        occurrences,
        groups,
    })
}

/// The template atlas (colour tiles, coverage in alpha), placements and the
/// base with the template glyphs un-premultiplied out of it.
fn build_glyph_templates(plan: TemplatePlan<'_>, xyb: &Image3F) -> Option<GlyphTemplates> {
    let lattice = TEMPLATE_LATTICE;
    let image_plane = [1usize, 0, 2];
    let TemplatePlan {
        shapes,
        occurrences,
        mut groups,
    } = plan;
    groups.sort_by_key(|g| {
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
        let fg = template_of(shapes, &occurrences, g, &mut coverage);
        // The tile: the foreground color everywhere, the coverage in alpha.
        let color: [i32; 3] = std::array::from_fn(|c| (fg[c] * lattice[c]).round() as i32);
        let decoded_color: [f32; 3] = std::array::from_fn(|c| color[c] as f32 / lattice[c]);
        for (dy, row) in coverage.chunks_exact_mut(w).enumerate() {
            let start = (ay + dy) * atlas_width + ax;
            atlas[0][start..][..w].fill(color[0]);
            atlas[1][start..][..w].fill(color[1]);
            atlas[2][start..][..w].fill(color[2] - color[0]);
            for (a, sample) in row.iter_mut().zip(&mut atlas_alpha[start..][..w]) {
                *sample = (*a * 255.0).round().clamp(0.0, 255.0) as u8;
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
        positions.sort_unstable_by_key(|&(x, y)| (y, x));
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
            let serial =
                find_lossy_patches_sized(&img, tile, 0.0, &serial_pool, &mut scratch).unwrap();
            let parallel =
                find_lossy_patches_sized(&img, tile, 0.0, &parallel_pool, &mut scratch).unwrap();
            assert!(serial.groups.len() >= 20);
            assert_eq!(serial.groups, parallel.groups);
            let groups =
                find_lossy_patch_groups_sized(&img, tile, 0.0, &parallel_pool, &mut scratch)
                    .unwrap();
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
    fn glyph_ctx() -> EncodingContext {
        EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            1.0,
            1,
        )
    }

    fn glyph_params(min_occurrences: usize) -> GlyphParams {
        GlyphParams {
            templates: false,
            min_occurrences,
            min_repeat_pixels: 0,
            coarse: 1,
            min_contrast: 0,
            singles_min_cover: 0.0,
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
    fn assert_glyph_plan_reconstructs(img: &Image3F, plan: &LossyGlyphPatches<Image3F>) {
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

    fn find_lossy_glyph_patches(
        ctx: &EncodingContext,
        xyb: &Image3F,
        params: &GlyphParams,
    ) -> Option<LossyGlyphPatches<Image3F>> {
        plan_lossy_glyph_patches(ctx, xyb, params).map(|plan| plan.materialize(xyb))
    }

    #[test]
    fn glyph_base_subtracts_overlapping_placements_before_sorting() {
        let mut img = Image3F::new(3, 1);
        img.plane_row_mut(1, 0)[1] = 0.1;
        let plan = LossyGlyphPatches {
            atlas: [vec![-6, 120], vec![0; 2], vec![6, -120]],
            atlas_width: 2,
            atlas_height: 1,
            base: None,
            references: vec![PatchReference {
                atlas_x: 0,
                atlas_y: 0,
                width: 2,
                height: 1,
                ref_frame: MODULAR_PATCH_REF_ID,
                alpha_only: false,
                add: true,
                mul: false,
                positions: vec![(1, 0), (0, 0)],
            }],
            alpha_atlas: None,
        }
        .materialize(&img);
        let expected = (0.1f32 - (-6.0 / 512.0)) - 120.0 / 512.0;
        let reversed = (0.1f32 - 120.0 / 512.0) - (-6.0 / 512.0);
        assert_ne!(expected.to_bits(), reversed.to_bits());
        assert_eq!(plan.base.plane_row(1, 0)[1].to_bits(), expected.to_bits());
        assert_eq!(plan.references[0].positions, [(0, 0), (1, 0)]);
    }

    #[test]
    fn lattice_rounding_matches_float_round() {
        let values: Vec<i32> = (-4099..=4099).chain([i32::MIN / 4, i32::MAX / 4]).collect();
        for shift in 1..=4u32 {
            let k = 1i32 << shift;
            let kernels: Vec<(&str, crate::xyb::RoundLatticePow2Fn)> = vec![
                ("dispatch", crate::xyb::selected_round_lattice_pow2_fn()),
                ("scalar", crate::xyb::round_lattice_pow2_scalar),
            ];
            #[cfg(any(
                all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"),
                all(target_arch = "x86_64", feature = "avx"),
                all(target_arch = "aarch64", feature = "neon"),
                all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
            ))]
            let mut kernels = kernels;
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
            if std::is_x86_feature_detected!("sse2") {
                kernels.push(("sse2", crate::sse::round_lattice_pow2_sse2));
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::is_x86_feature_detected!("avx2") {
                kernels.push(("avx2", crate::avx::round_lattice_pow2_avx2));
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            kernels.push(("neon", crate::neon::round_lattice_pow2_neon));
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
            kernels.push(("wasm", crate::wasm::round_lattice_pow2_wasm));
            for (name, f) in kernels {
                let mut out = values.clone();
                unsafe { f(&mut out, shift) };
                for (&v, &got) in values.iter().zip(&out) {
                    let want = (v as f64 / k as f64).round() as i32 * k;
                    assert_eq!(got, want, "{name}: v={v} k={k}");
                }
            }
            for &v in &values {
                let want = (v as f64 / k as f64).round() as i32 * k;
                assert_eq!(round_to_multiple(v, k), want, "v={v} k={k}");
            }
        }
        for v in -100..=100 {
            assert_eq!(
                round_to_multiple(v, 6),
                (v as f64 / 6.0).round() as i32 * 6,
                "v={v}"
            );
        }
    }

    /// One dark ink alpha-blended over two different flat backgrounds: the
    /// same glyph differs between them in every antialiased pixel, so exact
    /// patches need one atlas entry per background.
    fn blended_glyph_page() -> Image3F {
        let (w, h) = (192usize, 128usize);
        let mut img = Image3F::new(w, h);
        let backgrounds = [
            [4.0f32 / 4096.0, 300.0 / 512.0, 140.0 / 256.0],
            [-12.0 / 4096.0, 260.0 / 512.0, 200.0 / 256.0],
        ];
        let ink = [2.0f32 / 4096.0, 40.0 / 512.0, 30.0 / 256.0];
        for y in 0..h {
            let bg = backgrounds[usize::from(y >= 64)];
            for (c, &value) in bg.iter().enumerate() {
                img.plane_row_mut(c, y)[..w].fill(value);
            }
        }
        for (k, &(x0, y0)) in [
            (3, 5),
            (41, 9),
            (77, 22),
            (130, 37),
            (11, 75),
            (99, 101),
            (150, 80),
        ]
        .iter()
        .enumerate()
        {
            let bg = backgrounds[usize::from(y0 >= 64)];
            for dy in 0..7 {
                for dx in 0..5 {
                    // Coverage 0.25..1 everywhere in the box: one component.
                    let a = 0.25 + ((dx * 3 + dy * 5 + k % 2) % 4) as f32 * 0.25;
                    let a = if k % 2 == 0 { a } else { 1.25 - a };
                    for c in 0..3 {
                        img.plane_row_mut(c, y0 + dy)[x0 + dx] = bg[c] * (1.0 - a) + ink[c] * a;
                    }
                }
            }
        }
        img
    }

    #[test]
    fn glyph_templates_share_blended_glyphs_across_backgrounds() {
        use crate::quant_weights::INV_DC_QUANT;
        let img = blended_glyph_page();
        let exact = find_lossy_glyph_patches(&glyph_ctx(), &img, &glyph_params(2)).expect("plan");
        assert!(exact.alpha_atlas.is_none());
        // Glyph A twice on each background, glyph B twice on the first and
        // once on the second: three exact entries (B's single is dropped).
        assert_eq!(exact.references.len(), 3);

        let mut ctx = glyph_ctx();
        let probe = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        ctx.template_probe = Some(std::sync::Arc::clone(&probe));
        let params = GlyphParams {
            templates: true,
            ..glyph_params(2)
        };
        let plan = find_lossy_glyph_patches(&ctx, &img, &params).expect("plan");
        // Each glyph's two exact groups become one template: one 5x7 tile saved each.
        assert_eq!(probe.load(std::sync::atomic::Ordering::Relaxed), 2 * 35);
        let alpha = plan.alpha_atlas.as_ref().expect("template alpha");
        assert_eq!(alpha.len(), plan.atlas_width * plan.atlas_height);
        // One blend-above template per glyph, each placed on both backgrounds.
        assert_eq!(plan.references.len(), 2);
        assert!(plan.references.iter().all(|r| r.mul && !r.add));
        assert!(plan.references.iter().all(|r| {
            r.positions.iter().any(|p| p.1 < 64) && r.positions.iter().any(|p| p.1 >= 64)
        }));

        // kBlendAbove over an opaque base: base * (1 - a) + color * a.
        let mut recon = plan.base.clone();
        let [ay, ax, ab] = &plan.atlas;
        for r in &plan.references {
            for &(px, py) in &r.positions {
                for dy in 0..r.height {
                    for dx in 0..r.width {
                        let at = (r.atlas_y + dy) * plan.atlas_width + r.atlas_x + dx;
                        let a = f32::from(alpha[at]) / 255.0;
                        let color = [
                            ax[at] as f32 / INV_DC_QUANT[0],
                            ay[at] as f32 / INV_DC_QUANT[1],
                            (ab[at] + ay[at]) as f32 / INV_DC_QUANT[2],
                        ];
                        for (c, &color) in color.iter().enumerate() {
                            let v = &mut recon.plane_row_mut(c, py + dy)[px + dx];
                            *v = *v * (1.0 - a) + color * a;
                        }
                    }
                }
            }
        }
        for c in 0..3 {
            let tolerance = 4.0 / INV_DC_QUANT[c];
            for y in 0..img.ysize() {
                for (x, (a, b)) in img.plane_row(c, y)[..img.xsize()]
                    .iter()
                    .zip(&recon.plane_row(c, y)[..img.xsize()])
                    .enumerate()
                {
                    assert!(
                        (a - b).abs() <= tolerance,
                        "plane {c} ({x}, {y}): {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn exact_glyph_tiles_widen_the_template_atlas() {
        let ctx = glyph_ctx();
        let params = GlyphParams {
            templates: true,
            ..glyph_params(2)
        };
        let mut img = blended_glyph_page();
        let templates = find_lossy_glyph_patches(&ctx, &img, &params).expect("templates");
        assert_eq!(templates.atlas_width, 16);

        for (x, y, w, h) in [
            (2, 47, 21, 4),
            (30, 47, 21, 4),
            (70, 56, 18, 3),
            (110, 56, 18, 3),
        ] {
            for py in y..y + h {
                img.plane_row_mut(1, py)[x..x + w].fill(40.0 / 512.0);
            }
        }
        let mixed = find_lossy_glyph_patches(&ctx, &img, &params).expect("mixed atlas");
        assert_eq!(mixed.atlas_width, 21);
        assert_eq!(mixed.atlas_height, templates.atlas_height + 7);
        let top = mixed.atlas_width * templates.atlas_height;
        for (old, new) in templates.atlas.iter().zip(&mixed.atlas) {
            for (src, dst) in old.chunks_exact(16).zip(new[..top].chunks_exact(21)) {
                assert_eq!(src, &dst[..16]);
                assert_eq!(&dst[16..], &[0; 5]);
            }
        }
        let alpha = mixed.alpha_atlas.as_ref().unwrap();
        for (src, dst) in templates
            .alpha_atlas
            .as_ref()
            .unwrap()
            .chunks_exact(16)
            .zip(alpha[..top].chunks_exact(21))
        {
            assert_eq!(src, &dst[..16]);
            assert_eq!(&dst[16..], &[0; 5]);
        }
        assert!(alpha[top..].iter().all(|&a| a == 0));
        assert_eq!(mixed.references.len(), 4);
        for r in &mixed.references[2..] {
            assert!(r.add && !r.mul);
            assert!(r.atlas_y >= templates.atlas_height);
            let at = r.atlas_y * mixed.atlas_width + r.atlas_x;
            for &(x, y) in &r.positions {
                let decoded = mixed.base.plane_row(1, y)[x] + mixed.atlas[0][at] as f32 / 512.0;
                assert_eq!(decoded, img.plane_row(1, y)[x]);
            }
        }
    }

    #[test]
    fn lossy_glyph_patches_reconstruct_and_flatten_the_base() {
        let img = glyph_page();
        let repeated =
            find_lossy_glyph_patches(&glyph_ctx(), &img, &glyph_params(2)).expect("plan");
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
        let step = GLYPH_ATLAS_STEP as f32 / 2.0 / crate::quant_weights::INV_DC_QUANT[1];
        assert!((repeated.base.plane_row(1, 8)[5] - 307.0 / 512.0).abs() <= step);

        let all = find_lossy_glyph_patches(&glyph_ctx(), &img, &glyph_params(1)).expect("plan");
        assert_eq!(all.references.len(), 3);
        assert_glyph_plan_reconstructs(&img, &all);
    }

    /// The repeated marks reach 96 lattice steps on Y, the unrepeated one 48:
    /// a contrast floor between them drops only the faint mark, and a floor
    /// above both leaves no plan.
    #[test]
    fn faint_shapes_are_left_to_vardct() {
        let img = glyph_page();
        let params = GlyphParams {
            min_contrast: 64,
            ..glyph_params(1)
        };
        let plan = find_lossy_glyph_patches(&glyph_ctx(), &img, &params).expect("plan");
        assert_eq!(plan.references.len(), 2);
        assert!(plan.references.iter().all(|r| r.positions.len() >= 4));
        assert_glyph_plan_reconstructs(&img, &plan);
        let params = GlyphParams {
            min_contrast: 128,
            ..glyph_params(1)
        };
        assert!(find_lossy_glyph_patches(&glyph_ctx(), &img, &params).is_none());
    }

    /// One 36-pixel unrepeated mark on a 192x128 page is far below any
    /// coverage share, so it is dropped while the repeated marks stay; a
    /// zero share keeps it.
    #[test]
    fn unrepeated_shapes_need_frame_coverage() {
        let img = glyph_page();
        let params = GlyphParams {
            singles_min_cover: 0.5,
            ..glyph_params(1)
        };
        let plan = find_lossy_glyph_patches(&glyph_ctx(), &img, &params).expect("plan");
        assert_eq!(plan.references.len(), 2);
        assert!(plan.references.iter().all(|r| r.positions.len() >= 4));
        let params = GlyphParams {
            singles_min_cover: 0.001,
            ..glyph_params(1)
        };
        let plan = find_lossy_glyph_patches(&glyph_ctx(), &img, &params).expect("plan");
        assert_eq!(plan.references.len(), 3);
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
