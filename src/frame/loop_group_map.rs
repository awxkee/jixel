/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

//! Direct geometry lookups for the quant-field loop and its decoded DC.

use super::{K_BLOCK_DIM, K_DC_GROUP_DIM, K_GROUP_DIM_IN_BLOCKS};
use crate::dc_ops::ReconstructDcFn;
use crate::image::{Image3F, Image3S};
use crate::ssimulacra2::Rect;

pub(super) struct LoopGroupMap {
    width_blocks: usize,
    height_blocks: usize,
    dc_groups_x: usize,
    dc_indices: Vec<usize>,
    pub(super) task_of_block: Vec<usize>,
    task_origins: Vec<(usize, usize)>,
}

impl LoopGroupMap {
    pub(super) fn new(
        width: usize,
        height: usize,
        group_coords: &[(usize, usize)],
        ac_tasks: &[(usize, usize, usize)],
    ) -> Self {
        let width_blocks = width.div_ceil(K_BLOCK_DIM);
        let height_blocks = height.div_ceil(K_BLOCK_DIM);
        let dc_groups_x = width.div_ceil(K_DC_GROUP_DIM);
        let mut dc_indices = vec![usize::MAX; dc_groups_x * height.div_ceil(K_DC_GROUP_DIM)];
        for (i, &(gx, gy)) in group_coords.iter().enumerate() {
            dc_indices[gy * dc_groups_x + gx] = i;
        }
        let blocks_per_dc = K_DC_GROUP_DIM / K_BLOCK_DIM;
        let task_origins: Vec<_> = ac_tasks
            .iter()
            .map(|&(dc_idx, gx, gy)| {
                let (dc_gx, dc_gy) = group_coords[dc_idx];
                (
                    dc_gx * blocks_per_dc + gx * K_GROUP_DIM_IN_BLOCKS,
                    dc_gy * blocks_per_dc + gy * K_GROUP_DIM_IN_BLOCKS,
                )
            })
            .collect();
        let mut task_of_block = vec![usize::MAX; width_blocks * height_blocks];
        // Fill each task's rectangle once; avoid a task-list search per block.
        for (t, &(x, y)) in task_origins.iter().enumerate() {
            for by in y..(y + K_GROUP_DIM_IN_BLOCKS).min(height_blocks) {
                task_of_block[by * width_blocks + x
                    ..by * width_blocks + (x + K_GROUP_DIM_IN_BLOCKS).min(width_blocks)]
                    .fill(t);
            }
        }
        debug_assert!(!task_of_block.contains(&usize::MAX));
        Self {
            width_blocks,
            height_blocks,
            dc_groups_x,
            dc_indices,
            task_of_block,
            task_origins,
        }
    }

    pub(super) fn locate(&self, bx: usize, by: usize) -> (usize, usize, usize) {
        let blocks_per_dc = K_DC_GROUP_DIM / K_BLOCK_DIM;
        let dc = self.dc_indices[(by / blocks_per_dc) * self.dc_groups_x + bx / blocks_per_dc];
        (dc, bx % blocks_per_dc, by % blocks_per_dc)
    }

    /// Reconstruct only DC cells read by a pixel rectangle. Each cell uses
    /// exactly the same row kernel and arithmetic as full-frame reconstruction.
    /// Values outside the rectangle's cells are deliberately left untouched.
    pub(super) fn reconstruct_dc(
        &self,
        quant_dc: &[Option<Image3S>],
        cfl: [f32; 2],
        steps: [f32; 3],
        reconstruct: ReconstructDcFn,
        rect: Rect,
        out: &mut Image3F,
    ) {
        debug_assert_eq!(
            (out.xsize(), out.ysize()),
            (self.width_blocks, self.height_blocks)
        );
        let (x0, y0) = (rect.x0 / K_BLOCK_DIM, rect.y0 / K_BLOCK_DIM);
        let (x1, y1) = (
            rect.x1.div_ceil(K_BLOCK_DIM).min(self.width_blocks),
            rect.y1.div_ceil(K_BLOCK_DIM).min(self.height_blocks),
        );
        if rect.x0 >= rect.x1 || rect.y0 >= rect.y1 || x0 >= x1 || y0 >= y1 {
            return;
        }
        // An absent task contributes zero, just as in a fresh full-frame plane.
        for c in 0..3 {
            for y in y0..y1 {
                out.plane_row_mut(c, y)[x0..x1].fill(0.0);
            }
        }
        for gy in y0 / K_GROUP_DIM_IN_BLOCKS..y1.div_ceil(K_GROUP_DIM_IN_BLOCKS) {
            for gx in x0 / K_GROUP_DIM_IN_BLOCKS..x1.div_ceil(K_GROUP_DIM_IN_BLOCKS) {
                let t = self.task_of_block
                    [gy * K_GROUP_DIM_IN_BLOCKS * self.width_blocks + gx * K_GROUP_DIM_IN_BLOCKS];
                let Some(q) = &quant_dc[t] else { continue };
                let (bx0, by0) = self.task_origins[t];
                let lo = x0.saturating_sub(bx0);
                let hi = (x1 - bx0).min(q.xsize());
                for ly in y0.saturating_sub(by0)..(y1 - by0).min(q.ysize()) {
                    let [rx, ry, rb] = out.all_plane_rows_mut(by0 + ly);
                    reconstruct(
                        std::array::from_fn(|c| &q.plane_row(c, ly)[lo..hi]),
                        cfl,
                        steps,
                        [
                            &mut rx[bx0 + lo..bx0 + hi],
                            &mut ry[bx0 + lo..bx0 + hi],
                            &mut rb[bx0 + lo..bx0 + hi],
                        ],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Fixture = (Vec<(usize, usize)>, Vec<(usize, usize, usize)>);

    fn fixture(width: usize, height: usize) -> Fixture {
        let mut groups: Vec<_> = (0..height.div_ceil(K_DC_GROUP_DIM))
            .flat_map(|gy| (0..width.div_ceil(K_DC_GROUP_DIM)).map(move |gx| (gx, gy)))
            .collect();
        groups.reverse();
        let mut tasks = Vec::new();
        for (i, &(dx, dy)) in groups.iter().enumerate() {
            for gy in 0..8 {
                for gx in 0..8 {
                    if dx * K_DC_GROUP_DIM + gx * 256 < width
                        && dy * K_DC_GROUP_DIM + gy * 256 < height
                    {
                        tasks.push((i, gx, gy));
                    }
                }
            }
        }
        tasks.reverse();
        (groups, tasks)
    }

    #[test]
    fn direct_group_maps_match_linear_search_with_reordered_edge_groups() {
        let (width, height) = (2337usize, 2129usize);
        let (groups, tasks) = fixture(width, height);
        let map = LoopGroupMap::new(width, height, &groups, &tasks);
        for by in 0..height.div_ceil(8) {
            for bx in 0..width.div_ceil(8) {
                let dc = groups
                    .iter()
                    .position(|&g| g == (bx / 256, by / 256))
                    .unwrap();
                let (x, y) = (bx % 256, by % 256);
                assert_eq!(map.locate(bx, by), (dc, x, y));
                let task = tasks
                    .iter()
                    .position(|&t| t == (dc, x / 32, y / 32))
                    .unwrap();
                assert_eq!(map.task_of_block[by * map.width_blocks + bx], task);
            }
        }
    }
}
