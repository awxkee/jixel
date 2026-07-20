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
use crate::image::{Image3F, Image3Si};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub(crate) struct LossyPatches {
    pub(crate) atlas: Image3F,
    pub(crate) base: Image3F,
    pub(crate) references: Vec<PatchReference>,
}

#[derive(Clone, Copy)]
pub(crate) enum VarDctFrameKind<'a> {
    Regular,
    ReferenceOnly { width: usize, height: usize },
    Patched(&'a [PatchReference]),
}

pub(crate) fn find_lossy_patches(linear: &Image3F) -> Option<LossyPatches> {
    let tile = PATCH_TILE;
    let width = linear.xsize();
    let height = linear.ysize();
    if width < tile || height < tile {
        return None;
    }

    let mut buckets: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    for ty in 0..height / tile {
        for tx in 0..width / tile {
            let (x0, y0) = (tx * tile, ty * tile);
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for c in 0..3 {
                for y in y0..y0 + tile {
                    for &sample in &linear.plane_row(c, y)[x0..x0 + tile] {
                        sample.to_bits().hash(&mut hasher);
                    }
                }
            }
            buckets.entry(hasher.finish()).or_default().push((x0, y0));
        }
    }

    let mut groups = Vec::new();
    for candidates in buckets.into_values() {
        if candidates.len() < 3 {
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
        groups.extend(exact_groups.into_iter().filter(|g| g.len() >= 3));
    }
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    groups.truncate(256);
    if groups.is_empty() {
        return None;
    }

    let atlas_cols = groups.len().min(16);
    let atlas_rows = groups.len().div_ceil(atlas_cols);
    let atlas_width = atlas_cols * tile;
    let atlas_height = atlas_rows * tile;
    let mut atlas = Image3F::new(atlas_width, atlas_height);
    let mut base = linear.clone();
    let mut references = Vec::with_capacity(groups.len());
    for (i, positions) in groups.into_iter().enumerate() {
        let atlas_x = (i % atlas_cols) * tile;
        let atlas_y = (i / atlas_cols) * tile;
        let src = positions[0];
        for c in 0..3 {
            for dy in 0..tile {
                atlas.plane_row_mut(c, atlas_y + dy)[atlas_x..atlas_x + tile]
                    .copy_from_slice(&linear.plane_row(c, src.1 + dy)[src.0..src.0 + tile]);
                for &(x, y) in &positions {
                    base.plane_row_mut(c, y + dy)[x..x + tile].fill(0.0);
                }
            }
        }
        references.push(PatchReference {
            atlas_x,
            atlas_y,
            positions,
        });
    }
    Some(LossyPatches {
        atlas,
        base,
        references,
    })
}

pub(crate) const PATCH_TILE: usize = 16;
pub(crate) const PATCH_REF_ID: u32 = 3;
pub(crate) const NUM_PATCH_CONTEXTS: usize = 10;

#[derive(Clone)]
pub(crate) struct PatchReference {
    pub(crate) atlas_x: usize,
    pub(crate) atlas_y: usize,
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

pub(crate) fn find_lossless_patches(linear: &Image3Si) -> Option<LosslessPatches> {
    let width = linear.xsize();
    let height = linear.ysize();
    if width < PATCH_TILE || height < PATCH_TILE {
        return None;
    }

    let tiles_x = width / PATCH_TILE;
    let tiles_y = height / PATCH_TILE;
    let mut buckets: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * PATCH_TILE;
            let y0 = ty * PATCH_TILE;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for c in 0..3 {
                for y in y0..y0 + PATCH_TILE {
                    linear.plane_row(c, y)[x0..x0 + PATCH_TILE].hash(&mut hasher);
                }
            }
            buckets.entry(hasher.finish()).or_default().push((x0, y0));
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
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
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
            positions,
        });
    }
    Some(LosslessPatches {
        atlas,
        base,
        references,
    })
}
