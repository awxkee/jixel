/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
#![forbid(unsafe_code)]

use super::{DistanceParams, ImageDim, K_BLOCK_DIM, K_DC_GROUP_DIM};
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::dct::fmla;
use crate::image::Image3F;
use crate::thread_pool::ThreadPool;
use crate::util::{EncodeError, try_vec};

/// Keep the decoder's adaptive DC smoothing only when it improves luma DC
/// error against the source block means. Stitch the DC groups before filtering
/// so their boundaries have the same neighbors as in the decoder.
#[allow(clippy::too_many_arguments)]
pub(super) fn skip_dc_smoothing(
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dim: &ImageDim,
    dc_datas: &[DcGroupData],
    group_coords: &[(usize, usize)],
    distp: &DistanceParams,
    ytob_dc: i32,
) -> Result<bool, EncodeError> {
    const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
    let (w, h) = (dim.xsize_blocks, dim.ysize_blocks);
    // The decoder never smooths the outermost blocks.
    if w <= 2 || h <= 2 {
        return Ok(true);
    }
    let steps: [f32; 3] = std::array::from_fn(|c| {
        distp.dc_step[c] / (crate::quant_weights::INV_DC_QUANT[c] * distp.scale_dc)
    });
    let cfl_b = crate::color_correlation::dc_cfl_factor(distp.dc_step, ytob_dc);
    let mut recon = Image3F::try_new(w, h)?;
    for (dc, &(gx, gy)) in dc_datas.iter().zip(group_coords) {
        let (ox, oy) = (gx * DC_GROUP_BLOCKS, gy * DC_GROUP_BLOCKS);
        let q = &dc.quant_dc;
        for ly in 0..q.ysize() {
            let [rx, ry, rb] = recon.all_plane_rows_mut(oy + ly);
            for (dst, &v) in rx[ox..ox + q.xsize()].iter_mut().zip(q.plane_row(0, ly)) {
                *dst = v as f32 * steps[0];
            }
            for (dst, &v) in ry[ox..ox + q.xsize()].iter_mut().zip(q.plane_row(1, ly)) {
                *dst = v as f32 * steps[1];
            }
            for ((dst, &b), &y) in rb[ox..ox + q.xsize()]
                .iter_mut()
                .zip(q.plane_row(2, ly))
                .zip(q.plane_row(1, ly))
            {
                *dst = fmla(y as f32, cfl_b, b as f32) * steps[2];
            }
        }
    }

    // Fixed bands and ordered reduction keep the decision independent of the
    // worker count. Avoid dispatch overhead on small images.
    let band_rows = 16.max(1024usize.div_ceil(w - 2));
    let bands = (h - 2).div_ceil(band_rows);
    let score_band = |i, _: &mut CoderScratch| {
        let start = 1 + i * band_rows;
        score_rows(opsin, &recon, steps, start..(start + band_rows).min(h - 1))
    };
    let delta: f32 = if pool.num_threads() > 1 && (w - 2) * (h - 2) >= 8192 {
        let mut scores = try_vec![Ok(0.0); bands]?;
        pool.steal_for_each_mut(scratch, &mut scores, |i, score, scratch| {
            *score = score_band(i, scratch);
        });
        scores.into_iter().sum::<Result<f32, EncodeError>>()?
    } else {
        (0..bands)
            .map(|i| score_band(i, scratch))
            .sum::<Result<f32, EncodeError>>()?
    };
    Ok(delta >= 0.0)
}

#[inline]
fn smooth(up: &[f32; 3], mid: &[f32; 3], down: &[f32; 3]) -> f32 {
    const W1: f32 = 0.203_451_4;
    const W2: f32 = 0.033_482_92;
    const W0: f32 = 1.0 - 4.0 * (W1 + W2);
    let side = up[1] + down[1] + mid[0] + mid[2];
    let corner = up[0] + up[2] + down[0] + down[2];
    corner * W2 + side * W1 + mid[1] * W0
}

#[inline]
fn neighborhoods(image: &Image3F, c: usize, y: usize) -> impl Iterator<Item = [&[f32; 3]; 3]> {
    image
        .plane_row(c, y - 1)
        .array_windows::<3>()
        .zip(image.plane_row(c, y).array_windows::<3>())
        .zip(image.plane_row(c, y + 1).array_windows::<3>())
        .map(|((up, mid), down)| [up, mid, down])
}

fn score_rows(
    opsin: &Image3F,
    recon: &Image3F,
    steps: [f32; 3],
    rows: std::ops::Range<usize>,
) -> Result<f32, EncodeError> {
    let n = recon.xsize() - 2;
    let inv_steps = steps.map(f32::recip);
    let mut gaps = try_vec![0.5f32; n]?;
    let mut smoothed = try_vec![0.0f32; n]?;
    let mut delta = 0.0;
    for y in rows {
        gaps.fill(0.5);
        // Separate contiguous channel passes let LLVM vectorize the stencil
        // without keeping all nine neighborhood rows live during averaging.
        for c in [0, 2] {
            for (gap, [up, mid, down]) in gaps.iter_mut().zip(neighborhoods(recon, c, y)) {
                *gap = gap.max((mid[1] - smooth(up, mid, down)).abs() * inv_steps[c]);
            }
        }
        for ((s, gap), [up, mid, down]) in smoothed
            .iter_mut()
            .zip(&mut gaps)
            .zip(neighborhoods(recon, 1, y))
        {
            let sy = smooth(up, mid, down);
            *gap = gap.max((mid[1] - sy).abs() * inv_steps[1]);
            *s = mid[1] + (sy - mid[1]) * (3.0 - 4.0 * *gap).max(0.0);
        }
        // Every interior block is a full 8x8, even in odd-sized images.
        let source: [&[[f32; K_BLOCK_DIM]]; K_BLOCK_DIM] = std::array::from_fn(|dy| {
            opsin.plane_row(1, y * K_BLOCK_DIM + dy)[K_BLOCK_DIM..(n + 1) * K_BLOCK_DIM]
                .as_chunks::<K_BLOCK_DIM>()
                .0
        });
        for ((&s, &m), x) in smoothed
            .iter()
            .zip(&recon.plane_row(1, y)[1..n + 1])
            .zip(0..source[0].len())
        {
            if s == m {
                continue;
            }
            let sums = source.map(|row| sum8(&row[x]));
            let target = sum8(&sums) * (1.0 / 64.0);
            // Difference of squares avoids two large, nearly equal error
            // totals. All pixel arithmetic and the band reduction stay in f32.
            delta += (s - m) * ((s - target) + (m - target));
        }
    }
    Ok(delta)
}

// Balanced sums shorten the dependency chain and limit f32 rounding error.
#[inline]
fn sum8(v: &[f32; K_BLOCK_DIM]) -> f32 {
    ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))
}
