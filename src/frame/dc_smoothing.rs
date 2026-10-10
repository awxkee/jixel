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
use crate::dc_ops::{AdjustBdcFn, BdcCompensation, CompensateBdcFn, ReconstructDcFn};
use crate::dct::fmla;
use crate::haar::{CHROMA_DC_SQUEEZE_STEP, haar_smooth};
use crate::image::Image3F;
use crate::thread_pool::ThreadPool;
use crate::util::{EncodeError, try_vec};

impl BdcCompensation {
    fn new(distp: &DistanceParams) -> Self {
        use crate::quant_weights::{DC_QUANT, INV_DC_QUANT};
        Self {
            scale_b: INV_DC_QUANT[2] / distp.dc_step[2] * distp.scale_dc,
            y_step_b: DC_QUANT[1] * distp.dc_step[1] * INV_DC_QUANT[2] / distp.dc_step[2],
        }
    }
}

/// Requantize B from the retained fractional source under the final Y levels.
/// Return whether the later chroma stages should use compensated targets too.
/// Keep the original source planes intact for those stages.
pub(super) fn compensate_b_dc(
    dc_datas: &mut [DcGroupData],
    distp: &DistanceParams,
    ytob_dc: i32,
    cfl: crate::color_correlation::CflFrame,
    compensate_row: CompensateBdcFn,
) -> bool {
    if dc_datas
        .iter()
        .any(|dc| dc.source_dc_y.is_none() || dc.source_dc_b.is_none())
    {
        return false;
    }
    let compensation = BdcCompensation::new(distp);
    let cfl_b = crate::color_correlation::dc_cfl_factor(distp.dc_step, ytob_dc, cfl);
    for dc in dc_datas {
        let source_b = dc.source_dc_b.as_ref().unwrap();
        let source_y = dc.source_dc_y.as_ref().unwrap();
        for row in 0..dc.quant_dc.ysize() {
            let [_, y, b] = dc.quant_dc.all_plane_rows_mut(row);
            compensate_row(
                source_b.row(row),
                source_y.row(row),
                y,
                compensation,
                cfl_b,
                b,
            );
        }
    }
    true
}

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
    cfl: crate::color_correlation::CflFrame,
    reconstruct_row: ReconstructDcFn,
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
    let cfl_b = crate::color_correlation::dc_cfl_factor(distp.dc_step, ytob_dc, cfl);
    let cfl_x = crate::color_correlation::dc_cfl_factor_x(distp.dc_step, cfl.base_x);
    let mut recon = Image3F::try_new(w, h)?;
    for (dc, &(gx, gy)) in dc_datas.iter().zip(group_coords) {
        let (ox, oy) = (gx * DC_GROUP_BLOCKS, gy * DC_GROUP_BLOCKS);
        let q = &dc.quant_dc;
        for ly in 0..q.ysize() {
            let [rx, ry, rb] = recon.all_plane_rows_mut(oy + ly);
            reconstruct_row(
                std::array::from_fn(|c| q.plane_row(c, ly)),
                [cfl_x, cfl_b],
                steps,
                [
                    &mut rx[ox..ox + q.xsize()],
                    &mut ry[ox..ox + q.xsize()],
                    &mut rb[ox..ox + q.xsize()],
                ],
            );
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

/// Largest spread of a 3x3 neighborhood of source block means, in DC
/// steps, that still counts as one flat value.
const FLAT_DC_SPREAD: f32 = 0.35;

/// Requantize the X and B DC of flat neighborhoods from the 3x3 mean of the
/// source block means. Where a flat area's value sits near a DC level
/// boundary, its source noise rounds neighboring blocks to different levels,
/// a block mosaic of tint that the decoder's DC smoothing cannot undo. One
/// shared target per neighborhood puts the blocks on the same level.
#[allow(clippy::too_many_arguments)]
pub(super) fn flatten_chroma_dc(
    opsin: &Image3F,
    dim: &ImageDim,
    dc_datas: &mut [DcGroupData],
    group_coords: &[(usize, usize)],
    distp: &DistanceParams,
    ytob_dc: i32,
    cfl: crate::color_correlation::CflFrame,
    compensate_b: bool,
) -> Result<(), EncodeError> {
    const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
    let (w, h) = (dim.xsize_blocks, dim.ysize_blocks);
    if w < 3 || h < 3 {
        return Ok(());
    }
    let steps: [f32; 3] = std::array::from_fn(|c| {
        distp.dc_step[c] / (crate::quant_weights::INV_DC_QUANT[c] * distp.scale_dc)
    });
    let cfl_factors = [
        crate::color_correlation::dc_cfl_factor_x(distp.dc_step, cfl.base_x),
        0.0,
        crate::color_correlation::dc_cfl_factor(distp.dc_step, ytob_dc, cfl),
    ];
    let mut means = try_vec![0.0f32; w * h]?;
    // Column sums of a 3-row window of means and of their squares.
    let mut col_sum = try_vec![0.0f32; w]?;
    let mut col_sq = try_vec![0.0f32; w]?;
    let compensation = BdcCompensation::new(distp);
    for c in [0, 2] {
        for by in 0..h {
            let row = &mut means[by * w..(by + 1) * w];
            row.fill(0.0);
            for dy in 0..K_BLOCK_DIM {
                let src = opsin.plane_row(c, (by * K_BLOCK_DIM + dy).min(opsin.ysize() - 1));
                let (chunks, tail) = src.as_chunks::<K_BLOCK_DIM>();
                for (m, chunk) in row.iter_mut().zip(chunks) {
                    *m += sum8(chunk);
                }
                if chunks.len() < w {
                    // A partial last block: repeat its last sample.
                    let last = *src.last().unwrap_or(&0.0);
                    let partial =
                        tail.iter().sum::<f32>() + last * (K_BLOCK_DIM - tail.len()) as f32;
                    row[chunks.len()] += partial;
                }
            }
            row.iter_mut().for_each(|m| *m *= 1.0 / 64.0);
        }
        let max_var = (FLAT_DC_SPREAD * steps[c]) * (FLAT_DC_SPREAD * steps[c]);
        let inv_step = steps[c].recip();
        for by in 1..h - 1 {
            for (bx, (s, q)) in col_sum.iter_mut().zip(col_sq.iter_mut()).enumerate() {
                let (a, b, d) = (
                    means[(by - 1) * w + bx],
                    means[by * w + bx],
                    means[(by + 1) * w + bx],
                );
                *s = a + b + d;
                *q = a * a + b * b + d * d;
            }
            let (gy, ly) = (by / DC_GROUP_BLOCKS, by % DC_GROUP_BLOCKS);
            for bx in 1..w - 1 {
                let sum = col_sum[bx - 1] + col_sum[bx] + col_sum[bx + 1];
                let sq = col_sq[bx - 1] + col_sq[bx] + col_sq[bx + 1];
                let mean = sum * (1.0 / 9.0);
                if sq * (1.0 / 9.0) - mean * mean >= max_var {
                    continue;
                }
                let (gx, lx) = (bx / DC_GROUP_BLOCKS, bx % DC_GROUP_BLOCKS);
                let dc = &mut dc_datas[gy * dim.xsize_dc_groups + gx];
                debug_assert_eq!(group_coords[gy * dim.xsize_dc_groups + gx], (gx, gy));
                let yq = dc.quant_dc.plane_row(1, ly)[lx];
                let mut target = fmla(mean, inv_step, -f32::from(yq) * cfl_factors[c]);
                if c == 2 && compensate_b {
                    target = compensation.adjust(
                        target,
                        dc.source_dc_y.as_ref().unwrap().row(ly)[lx],
                        yq,
                    );
                }
                dc.quant_dc.plane_row_mut(c, ly)[lx] = target.round() as i16;
            }
        }
    }
    Ok(())
}

/// Candidate high-band steps of the squeezed chroma DC, in units of the
/// channel's current DC step.
static CHROMA_DC_SQUEEZE_HF_STEPS: [f32; 4] = [0.5, 0.75, 1.0, 1.5];
/// Required saving of the combined rate + distortion score.
const CHROMA_DC_SQUEEZE_MARGIN: f64 = 0.98;

/// Choose, per chroma channel, between the DC plane as quantized and a 2x
/// finer DC step holding a one-level Haar reconstruction of the source DC
/// whose high band is deadzoned. Where the chroma varies by less than a step
/// the smooth plane is closer to the source than nearest rounding and codes
/// cheaper under the predictive DC coder; a candidate is taken only when its
/// estimated bits and its error both fall. Bits are the DC coder's own price
/// (`price_dc_plane`, see the note at `walk_dc_channel` in frame.rs: that
/// price must follow any change to the DC coding). The chosen step is
/// signaled with the frame's
/// DC steps and the planes are requantized from the unrounded DC.
pub(super) fn choose_chroma_dc_coding(
    dim: &ImageDim,
    dc_datas: &mut [DcGroupData],
    distp: &mut DistanceParams,
    ytob_dc: i32,
    cfl: crate::color_correlation::CflFrame,
    compensate_b: bool,
    adjust_row: AdjustBdcFn,
) -> Result<(), EncodeError> {
    const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
    let (w, h) = (dim.xsize_blocks, dim.ysize_blocks);
    if w < 2 || h < 2 {
        return Ok(());
    }
    if dc_datas
        .iter()
        .any(|dc| dc.source_dc_x.is_none() || dc.source_dc_b.is_none())
    {
        return Ok(());
    }
    let mut t = try_vec![0.0f32; w * h]?;
    let mut rec = try_vec![0.0f32; w * h]?;
    let mut levels = try_vec![0i32; w * h]?;
    let mut best_levels = try_vec![0i32; w * h]?;
    // Luma is unchanged by the chroma search. Stitch it once, then price each
    // candidate in raster order without resolving DC-group coordinates again.
    let mut y_levels = try_vec![0i16; w * h]?;
    for c in [0usize, 2] {
        let step = distp.dc_step;
        let cfl_factor = |step: [f32; 3]| {
            if c == 0 {
                crate::color_correlation::dc_cfl_factor_x(step, cfl.base_x)
            } else {
                crate::color_correlation::dc_cfl_factor(step, ytob_dc, cfl)
            }
        };
        // Targets in units of the channel's current step, CfL folded out.
        let inv = crate::quant_weights::INV_DC_QUANT[c] * distp.scale_dc / step[c];
        let cfl_coarse = cfl_factor(step);
        let compensation = BdcCompensation::new(distp);
        for (i, dc) in dc_datas.iter().enumerate() {
            let ox = (i % dim.xsize_dc_groups) * DC_GROUP_BLOCKS;
            let oy = (i / dim.xsize_dc_groups) * DC_GROUP_BLOCKS;
            let width = dc.quant_dc.xsize().min(w - ox);
            let height = dc.quant_dc.ysize().min(h - oy);
            let source = if c == 0 {
                &dc.source_dc_x
            } else {
                &dc.source_dc_b
            };
            let source = source.as_ref().unwrap();
            for ly in 0..height {
                let begin = (oy + ly) * w + ox;
                let end = begin + width;
                let y_row = &mut y_levels[begin..end];
                if c == 0 {
                    y_row.copy_from_slice(&dc.quant_dc.plane_row(1, ly)[..width]);
                }
                for ((target, &src), &y) in t[begin..end]
                    .iter_mut()
                    .zip(&source.row(ly)[..width])
                    .zip(y_row.iter())
                {
                    *target = fmla(src, inv, -f32::from(y) * cfl_coarse);
                }
                if c == 2 && compensate_b {
                    adjust_row(
                        &mut t[begin..end],
                        &dc.source_dc_y.as_ref().unwrap().row(ly)[..width],
                        y_row,
                        compensation,
                    );
                }
                for (level, &q) in levels[begin..end]
                    .iter_mut()
                    .zip(&dc.quant_dc.plane_row(c, ly)[..width])
                {
                    *level = i32::from(q);
                }
            }
        }
        let base_bits = plane_price(&levels, w, dim, dc_datas);
        let base_d: f64 = t
            .iter()
            .zip(&levels)
            .map(|(&tv, &l)| f64::from(tv - l as f32).powi(2))
            .sum();
        if base_d <= 0.0 {
            continue;
        }
        let lambda = base_bits / base_d;
        let base_score = base_bits + lambda * base_d;
        // The finer step as the decoder will read it back.
        let mut fine = step;
        fine[c] = crate::color_correlation::signaled_dc_step(c, step[c] * CHROMA_DC_SQUEEZE_STEP);
        let ratio = fine[c] / step[c];
        let cfl_fine = cfl_factor(fine);
        // Level on the fine lattice of a value given in current-step units.
        let fine_level = |value: f32, y_level: f32| -> i32 {
            ((value + y_level * cfl_coarse) / ratio - y_level * cfl_fine).round() as i32
        };
        let mut best = None;
        for &q_hf in &CHROMA_DC_SQUEEZE_HF_STEPS {
            haar_smooth(&t, w, h, q_hf, &mut rec);
            let mut d = 0.0f64;
            for (((level, &value), &target), &y) in
                levels.iter_mut().zip(&rec).zip(&t).zip(&y_levels)
            {
                let y_level = f32::from(y);
                let l = fine_level(value, y_level);
                *level = l;
                let back = (l as f32 + y_level * cfl_fine) * ratio - y_level * cfl_coarse;
                d += f64::from(target - back).powi(2);
            }
            if d > base_d {
                continue;
            }
            let bits = plane_price(&levels, w, dim, dc_datas);
            if bits > base_bits {
                continue;
            }
            let score = bits + lambda * d;
            if best.is_none_or(|b| score < b) {
                best = Some(score);
                // Preserve the winner without another Haar and quantization
                // pass. The next candidate overwrites every scratch level.
                std::mem::swap(&mut levels, &mut best_levels);
            }
        }
        let Some(score) = best else {
            continue;
        };
        if score >= base_score * CHROMA_DC_SQUEEZE_MARGIN {
            continue;
        }
        distp.dc_step[c] *= CHROMA_DC_SQUEEZE_STEP;
        for (i, dc) in dc_datas.iter_mut().enumerate() {
            let ox = (i % dim.xsize_dc_groups) * DC_GROUP_BLOCKS;
            let oy = (i / dim.xsize_dc_groups) * DC_GROUP_BLOCKS;
            let width = dc.quant_dc.xsize().min(w - ox);
            let height = dc.quant_dc.ysize().min(h - oy);
            for ly in 0..height {
                let begin = (oy + ly) * w + ox;
                for (dst, &level) in dc.quant_dc.plane_row_mut(c, ly)[..width]
                    .iter_mut()
                    .zip(&best_levels[begin..begin + width])
                {
                    *dst = level as i16;
                }
            }
        }
    }
    Ok(())
}

/// Bits to code a stitched `w`×`h` plane of DC levels, priced DC group by DC
/// group with the coder's own tokenization (`price_dc_plane`), so the
/// squeeze decision compares what the frame will really pay.
fn plane_price(levels: &[i32], w: usize, dim: &ImageDim, dc_datas: &[DcGroupData]) -> f64 {
    const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
    let h = levels.len() / w.max(1);
    let mut bits = 0.0;
    for (i, dc) in dc_datas.iter().enumerate() {
        let ox = (i % dim.xsize_dc_groups) * DC_GROUP_BLOCKS;
        let oy = (i / dim.xsize_dc_groups) * DC_GROUP_BLOCKS;
        let width = dc.quant_dc.xsize().min(w.saturating_sub(ox));
        let height = dc.quant_dc.ysize().min(h.saturating_sub(oy));
        if width == 0 || height == 0 {
            continue;
        }
        let begin = oy * w + ox;
        let end = begin + (height - 1) * w + width;
        bits += super::price_dc_plane(&levels[begin..end], width, height, w);
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_correlation::CflFrame;
    use crate::dc_ops::{B_DC_Y_COMPENSATION, selected_dc_row_kernels};
    use crate::haar::CHROMA_DC_SQUEEZE_DEADZONE;
    use crate::image::Plane;

    #[test]
    fn compensation_can_improve_b_minus_y_while_increasing_b_error() {
        let mut distp = crate::frame::compute_distance_params(2.0);
        distp.scale_dc = 1.0;
        distp.dc_step = [1.0; 3];
        let compensation = BdcCompensation::new(&distp);
        let (b, y, yq) = (0.99 / 256.0, 0.51 / 512.0, 1);
        let residual = fmla(b, 256.0, -f32::from(yq) * 0.5);
        let target = compensation.target(b, y, yq, 0.5);
        let full_target = residual + 0.5 * f32::from(yq) - y * 256.0;
        assert_eq!(residual.round(), 0.0);
        assert_eq!(target.round(), 1.0);
        assert!((target.round() - residual).abs() > (residual.round() - residual).abs());
        assert!((target.round() - full_target).abs() < (residual.round() - full_target).abs());
    }

    #[test]
    fn compensation_preserves_sources_and_feeds_the_haar_search() {
        let (dim, mut actual, mut params, slope, cfl) = fixture(5, 3, 0, false);
        let (_, mut expected, mut expected_params, _, _) = fixture(5, 3, 0, false);
        params.scale_dc = 1.0;
        expected_params.scale_dc = 1.0;
        // Uniform source values and exact binary phases make the independent
        // source adjustment identical to compensation in B-step units.
        for dc in actual.iter_mut().chain(expected.iter_mut()) {
            let mut sy = Plane::new(dc.quant_dc.xsize(), dc.quant_dc.ysize());
            sy.as_mut_slice().fill(-0.375 / 512.0);
            dc.source_dc_y = Some(sy);
            dc.source_dc_b
                .as_mut()
                .unwrap()
                .as_mut_slice()
                .fill(1.375 / 256.0);
        }
        let original_b = actual[0].source_dc_b.as_ref().unwrap().as_slice().to_vec();
        for dc in &mut expected {
            for row in 0..dc.quant_dc.ysize() {
                let source = dc.source_dc_b.as_mut().unwrap().row_mut(row);
                for v in source.iter_mut() {
                    *v += B_DC_Y_COMPENSATION * (0.375 / 512.0);
                }
                for (dst, &src) in dc
                    .quant_dc
                    .plane_row_mut(2, row)
                    .iter_mut()
                    .zip(source.iter())
                {
                    *dst = (src * 256.0).round() as i16;
                }
            }
        }
        assert!(compensate_b_dc(
            &mut actual,
            &params,
            slope,
            cfl,
            selected_dc_row_kernels().compensate_b
        ));
        assert_eq!(
            actual[0].source_dc_b.as_ref().unwrap().as_slice(),
            original_b
        );
        reference_choose_chroma_dc_coding(&dim, &mut expected, &mut expected_params, slope, cfl)
            .unwrap();
        choose_chroma_dc_coding(
            &dim,
            &mut actual,
            &mut params,
            slope,
            cfl,
            true,
            selected_dc_row_kernels().adjust_b,
        )
        .unwrap();
        assert_eq!(params.dc_step, expected_params.dc_step);
        for (a, b) in actual.iter().zip(&expected) {
            for c in 0..3 {
                assert_eq!(a.quant_dc.plane_data(c), b.quant_dc.plane_data(c));
            }
        }
    }

    #[test]
    fn flattening_uses_the_compensated_b_target() {
        let (dim, mut groups, mut params, slope, cfl) = fixture(5, 5, 0, false);
        params.scale_dc = 1.0;
        let mut opsin = Image3F::try_new(dim.xsize, dim.ysize).unwrap();
        opsin.plane_mut(2).as_mut_slice().fill(0.99 / 256.0);
        for dc in &mut groups {
            dc.quant_dc.plane_mut(1).as_mut_slice().fill(1);
            dc.source_dc_y = Some(Plane::new_fill(5, 5, 0.51 / 512.0));
            dc.source_dc_b = Some(Plane::new_fill(5, 5, 0.99 / 256.0));
        }
        assert!(compensate_b_dc(
            &mut groups,
            &params,
            slope,
            cfl,
            selected_dc_row_kernels().compensate_b
        ));
        flatten_chroma_dc(
            &opsin,
            &dim,
            &mut groups,
            &[(0, 0)],
            &params,
            slope,
            cfl,
            true,
        )
        .unwrap();
        assert_eq!(groups[0].quant_dc.plane_row(2, 2)[2], 1);
        flatten_chroma_dc(
            &opsin,
            &dim,
            &mut groups,
            &[(0, 0)],
            &params,
            slope,
            cfl,
            false,
        )
        .unwrap();
        assert_eq!(groups[0].quant_dc.plane_row(2, 2)[2], 0);
        assert!(groups[0].quant_dc.plane_data(1).iter().all(|&q| q == 1));
    }

    #[test]
    fn missing_luma_source_skips_compensation_for_all_groups() {
        let (_, mut groups, params, slope, cfl) = fixture(513, 3, 0, false);
        let dc = &mut groups[0];
        dc.source_dc_y = Some(Plane::new_fill(
            dc.quant_dc.xsize(),
            dc.quant_dc.ysize(),
            -0.01,
        ));
        let before: Vec<_> = groups.iter().map(|dc| dc.quant_dc.clone()).collect();
        assert!(!compensate_b_dc(
            &mut groups,
            &params,
            slope,
            cfl,
            selected_dc_row_kernels().compensate_b
        ));
        for (dc, before) in groups.iter().zip(before) {
            for c in 0..3 {
                assert_eq!(dc.quant_dc.plane_data(c), before.plane_data(c));
            }
        }
    }

    // The original search is an independent oracle for rate decisions, CfL,
    // group boundaries and the final quantized planes.
    fn reference_choose_chroma_dc_coding(
        dim: &ImageDim,
        dc_datas: &mut [DcGroupData],
        distp: &mut DistanceParams,
        ytob_dc: i32,
        cfl: crate::color_correlation::CflFrame,
    ) -> Result<(), EncodeError> {
        const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
        let (w, h) = (dim.xsize_blocks, dim.ysize_blocks);
        if w < 2 || h < 2 {
            return Ok(());
        }
        if dc_datas
            .iter()
            .any(|dc| dc.source_dc_x.is_none() || dc.source_dc_b.is_none())
        {
            return Ok(());
        }
        let mut t = try_vec![0.0f32; w * h]?;
        let mut rec = try_vec![0.0f32; w * h]?;
        let mut levels = try_vec![0i32; w * h]?;
        for c in [0usize, 2] {
            let step = distp.dc_step;
            let cfl_factor = |step: [f32; 3]| {
                if c == 0 {
                    crate::color_correlation::dc_cfl_factor_x(step, cfl.base_x)
                } else {
                    crate::color_correlation::dc_cfl_factor(step, ytob_dc, cfl)
                }
            };
            // Targets in units of the channel's current step, CfL folded out.
            let inv = crate::quant_weights::INV_DC_QUANT[c] * distp.scale_dc / step[c];
            let cfl_coarse = cfl_factor(step);
            for by in 0..h {
                for bx in 0..w {
                    let (gx, gy) = (bx / DC_GROUP_BLOCKS, by / DC_GROUP_BLOCKS);
                    let dc = &dc_datas[gy * dim.xsize_dc_groups + gx];
                    let (lx, ly) = (bx % DC_GROUP_BLOCKS, by % DC_GROUP_BLOCKS);
                    let y_level = f32::from(dc.quant_dc.plane_row(1, ly)[lx]);
                    let src = if c == 0 {
                        &dc.source_dc_x
                    } else {
                        &dc.source_dc_b
                    };
                    t[by * w + bx] = fmla(
                        src.as_ref().unwrap().row(ly)[lx],
                        inv,
                        -y_level * cfl_coarse,
                    );
                    levels[by * w + bx] = i32::from(dc.quant_dc.plane_row(c, ly)[lx]);
                }
            }
            let base_bits = plane_price(&levels, w, dim, dc_datas);
            let base_d: f64 = t
                .iter()
                .zip(&levels)
                .map(|(&tv, &l)| f64::from(tv - l as f32).powi(2))
                .sum();
            if base_d <= 0.0 {
                continue;
            }
            let lambda = base_bits / base_d;
            let base_score = base_bits + lambda * base_d;
            // The finer step as the decoder will read it back.
            let mut fine = step;
            fine[c] =
                crate::color_correlation::signaled_dc_step(c, step[c] * CHROMA_DC_SQUEEZE_STEP);
            let ratio = fine[c] / step[c];
            let cfl_fine = cfl_factor(fine);
            // Level on the fine lattice of a value given in current-step units.
            let fine_level = |value: f32, y_level: f32| -> i32 {
                ((value + y_level * cfl_coarse) / ratio - y_level * cfl_fine).round() as i32
            };
            let mut best: Option<(f64, f32)> = None;
            for &q_hf in &CHROMA_DC_SQUEEZE_HF_STEPS {
                reference_haar_smooth(&t, w, h, q_hf, &mut rec);
                let mut d = 0.0f64;
                for by in 0..h {
                    for bx in 0..w {
                        let (gx, gy) = (bx / DC_GROUP_BLOCKS, by / DC_GROUP_BLOCKS);
                        let dc = &dc_datas[gy * dim.xsize_dc_groups + gx];
                        let (lx, ly) = (bx % DC_GROUP_BLOCKS, by % DC_GROUP_BLOCKS);
                        let y_level = f32::from(dc.quant_dc.plane_row(1, ly)[lx]);
                        let l = fine_level(rec[by * w + bx], y_level);
                        levels[by * w + bx] = l;
                        let back = (l as f32 + y_level * cfl_fine) * ratio - y_level * cfl_coarse;
                        d += f64::from(t[by * w + bx] - back).powi(2);
                    }
                }
                if d > base_d {
                    continue;
                }
                let bits = plane_price(&levels, w, dim, dc_datas);
                if bits > base_bits {
                    continue;
                }
                let score = bits + lambda * d;
                if best.is_none_or(|b| score < b.0) {
                    best = Some((score, q_hf));
                }
            }
            let Some((score, q_hf)) = best else {
                continue;
            };
            if score >= base_score * CHROMA_DC_SQUEEZE_MARGIN {
                continue;
            }
            reference_haar_smooth(&t, w, h, q_hf, &mut rec);
            distp.dc_step[c] *= CHROMA_DC_SQUEEZE_STEP;
            for by in 0..h {
                for bx in 0..w {
                    let (gx, gy) = (bx / DC_GROUP_BLOCKS, by / DC_GROUP_BLOCKS);
                    let dc = &mut dc_datas[gy * dim.xsize_dc_groups + gx];
                    let (lx, ly) = (bx % DC_GROUP_BLOCKS, by % DC_GROUP_BLOCKS);
                    let y_level = f32::from(dc.quant_dc.plane_row(1, ly)[lx]);
                    dc.quant_dc.plane_row_mut(c, ly)[lx] =
                        fine_level(rec[by * w + bx], y_level) as i16;
                }
            }
        }
        Ok(())
    }

    /// One-level 2x2 Haar of `t`: the low band rounded to the fine lattice, the
    /// three high bands deadzone-quantized at `q_hf`; odd edges pass through.
    fn reference_haar_smooth(t: &[f32], w: usize, h: usize, q_hf: f32, out: &mut [f32]) {
        let dzq = |x: f32| -> f32 {
            ((x.abs() / q_hf + (1.0 - CHROMA_DC_SQUEEZE_DEADZONE)).floor() * q_hf).copysign(x)
        };
        out.copy_from_slice(t);
        for y in (0..h / 2 * 2).step_by(2) {
            for x in (0..w / 2 * 2).step_by(2) {
                let (a, b) = (t[y * w + x], t[y * w + x + 1]);
                let (c, d) = (t[(y + 1) * w + x], t[(y + 1) * w + x + 1]);
                let ll = ((a + b + c + d) * (0.25 / CHROMA_DC_SQUEEZE_STEP)).round()
                    * CHROMA_DC_SQUEEZE_STEP;
                let hl = dzq((a - b + c - d) * 0.25);
                let lh = dzq((a + b - c - d) * 0.25);
                let hh = dzq((a - b - c + d) * 0.25);
                out[y * w + x] = ll + hl + lh + hh;
                out[y * w + x + 1] = ll - hl + lh - hh;
                out[(y + 1) * w + x] = ll + hl - lh - hh;
                out[(y + 1) * w + x + 1] = ll - hl - lh + hh;
            }
        }
    }

    fn fixture(
        w: usize,
        h: usize,
        family: usize,
        custom_cfl: bool,
    ) -> (ImageDim, Vec<DcGroupData>, DistanceParams, i32, CflFrame) {
        let dim = ImageDim::new(w * K_BLOCK_DIM - 1, h * K_BLOCK_DIM - 1);
        let mut params = crate::frame::compute_distance_params(2.0);
        params.dc_step = if custom_cfl {
            [0.731, 1.173, 1.319]
        } else {
            [1.0; 3]
        };
        let cfl = if custom_cfl {
            CflFrame {
                base_x: 0.1875,
                base_b: 0.9375,
                ..CflFrame::XYB
            }
        } else {
            CflFrame::XYB
        };
        let ytob_dc = if custom_cfl { -17 } else { 0 };
        let factors = [
            crate::color_correlation::dc_cfl_factor_x(params.dc_step, cfl.base_x),
            0.0,
            crate::color_correlation::dc_cfl_factor(params.dc_step, ytob_dc, cfl),
        ];
        let inv: [f32; 3] = std::array::from_fn(|c| {
            crate::quant_weights::INV_DC_QUANT[c] * params.scale_dc / params.dc_step[c]
        });
        let group_blocks = K_DC_GROUP_DIM / K_BLOCK_DIM;
        let mut groups = Vec::new();
        for gy in 0..dim.ysize_dc_groups {
            for gx in 0..dim.xsize_dc_groups {
                let ox = gx * group_blocks;
                let oy = gy * group_blocks;
                let width = (w - ox).min(group_blocks);
                let height = (h - oy).min(group_blocks);
                let mut dc = DcGroupData::new(width, height).unwrap();
                let mut sx = Plane::new(width, height);
                let mut sb = Plane::new(width, height);
                for ly in 0..height {
                    for lx in 0..width {
                        let (x, y) = (ox + lx, oy + ly);
                        let hash = (x as u32)
                            .wrapping_mul(17_171)
                            .wrapping_add((y as u32).wrapping_mul(1_913));
                        let luma = if family == 0 {
                            0
                        } else {
                            (hash % 23) as i16 - 11
                        };
                        dc.quant_dc.plane_row_mut(1, ly)[lx] = luma;
                        for c in [0usize, 2] {
                            let target = match family {
                                0 => 1.49 + c as f32, // Both channels should accept the finer step.
                                1 => ((x + y + c) % 9) as f32,
                                2 => {
                                    if (x + y) % 2 == 0 {
                                        -20.3
                                    } else {
                                        19.9
                                    }
                                }
                                3 => {
                                    ((x / 8 + y / 8 + c) % 7) as f32
                                        + 0.49
                                        + (hash % 31) as f32 * 0.008
                                        - 0.12
                                }
                                _ => (hash % 65_521) as f32 / 128.0 - 256.0,
                            };
                            let source = (target + f32::from(luma) * factors[c]) / inv[c];
                            if c == 0 {
                                sx.row_mut(ly)[lx] = source;
                            } else {
                                sb.row_mut(ly)[lx] = source;
                            }
                            dc.quant_dc.plane_row_mut(c, ly)[lx] = target.round() as i16;
                        }
                    }
                }
                dc.source_dc_x = Some(sx);
                dc.source_dc_b = Some(sb);
                groups.push(dc);
            }
        }
        (dim, groups, params, ytob_dc, cfl)
    }

    fn assert_search_matches(w: usize, h: usize, family: usize, custom_cfl: bool) -> [f32; 3] {
        let (dim, mut expected, mut expected_params, ytob_dc, cfl) =
            fixture(w, h, family, custom_cfl);
        let (_, mut actual, mut actual_params, _, _) = fixture(w, h, family, custom_cfl);
        reference_choose_chroma_dc_coding(&dim, &mut expected, &mut expected_params, ytob_dc, cfl)
            .unwrap();
        choose_chroma_dc_coding(
            &dim,
            &mut actual,
            &mut actual_params,
            ytob_dc,
            cfl,
            false,
            selected_dc_row_kernels().adjust_b,
        )
        .unwrap();
        assert_eq!(
            actual_params.dc_step.map(f32::to_bits),
            expected_params.dc_step.map(f32::to_bits),
            "steps: {w}x{h}, family={family}, custom_cfl={custom_cfl}"
        );
        for (a, b) in actual.iter().zip(&expected) {
            for c in 0..3 {
                assert_eq!(
                    a.quant_dc.plane_data(c),
                    b.quant_dc.plane_data(c),
                    "levels: {w}x{h}, family={family}, custom_cfl={custom_cfl}, c={c}"
                );
            }
        }
        actual_params.dc_step
    }

    #[test]
    fn squeeze_search_preserves_reference_decisions_and_levels() {
        for (w, h) in [(2, 2), (3, 5), (16, 17), (65, 33), (513, 5), (5, 513)] {
            for family in 0..5 {
                for custom_cfl in [false, true] {
                    assert_search_matches(w, h, family, custom_cfl);
                }
            }
        }
        // Cross both DC-group boundaries, including the odd bottom-right group.
        let accepted = assert_search_matches(513, 513, 0, false);
        assert_eq!(accepted, [0.5, 1.0, 0.5]);
    }

    #[test]
    fn squeeze_search_skips_missing_source_planes_and_thin_images() {
        for (w, h, missing) in [(1, 5, false), (5, 1, false), (513, 3, true)] {
            let (dim, mut groups, mut params, ytob_dc, cfl) = fixture(w, h, 0, false);
            if missing {
                groups.last_mut().unwrap().source_dc_b = None;
            }
            let before: Vec<_> = groups.iter().map(|dc| dc.quant_dc.clone()).collect();
            choose_chroma_dc_coding(
                &dim,
                &mut groups,
                &mut params,
                ytob_dc,
                cfl,
                false,
                selected_dc_row_kernels().adjust_b,
            )
            .unwrap();
            assert_eq!(params.dc_step, [1.0; 3]);
            for (dc, before) in groups.iter().zip(before) {
                for c in 0..3 {
                    assert_eq!(dc.quant_dc.plane_data(c), before.plane_data(c));
                }
            }
        }
    }

    #[test]
    fn price_dc_plane_is_window_invariant_and_orders_smoothness() {
        // A window priced through a wider stride costs exactly what the same
        // samples cost as a contiguous plane: the pricer walks rows, not buffers.
        let (w, h, stride) = (37, 23, 64);
        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as i32 - 128
        };
        let big: Vec<i32> = (0..stride * h).map(|_| next()).collect();
        let window: Vec<i32> = (0..h)
            .flat_map(|y| big[y * stride..y * stride + w].iter().copied())
            .collect();
        let windowed = super::super::price_dc_plane(&big[..(h - 1) * stride + w], w, h, stride);
        let contiguous = super::super::price_dc_plane(&window, w, h, w);
        assert_eq!(windowed.to_bits(), contiguous.to_bits());
        // Empty windows are free; a constant plane costs almost nothing; noise
        // costs more than a ramp, which costs more than a constant.
        assert_eq!(super::super::price_dc_plane(&[], 0, 0, 0), 0.0);
        let constant = vec![5i32; w * h];
        let ramp: Vec<i32> = (0..w * h).map(|i| (i % w) as i32 / 4).collect();
        let flat = super::super::price_dc_plane(&constant, w, h, w);
        let sloped = super::super::price_dc_plane(&ramp, w, h, w);
        assert!(
            flat < 0.05 * (w * h) as f64,
            "constant plane priced {flat} bits"
        );
        assert!(
            flat < sloped && sloped < contiguous,
            "{flat} {sloped} {contiguous}"
        );
    }

    #[test]
    fn haar_matches_reference_at_deadzone_boundaries_and_odd_edges() {
        let mut state = 0x5632_91adu32;
        for (w, h) in [
            (0, 0),
            (1, 13),
            (13, 1),
            (2, 2),
            (3, 3),
            (8, 7),
            (17, 16),
            (513, 5),
        ] {
            for &q_hf in &CHROMA_DC_SQUEEZE_HF_STEPS {
                let source: Vec<_> = (0..w * h)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        match i % 7 {
                            0 => -0.0,
                            1 => q_hf * CHROMA_DC_SQUEEZE_DEADZONE,
                            2 => -q_hf * CHROMA_DC_SQUEEZE_DEADZONE,
                            _ => (state % 10_001) as f32 * 0.001 - 5.0,
                        }
                    })
                    .collect();
                let mut expected = vec![f32::NAN; w * h];
                let mut actual = expected.clone();
                reference_haar_smooth(&source, w, h, q_hf, &mut expected);
                haar_smooth(&source, w, h, q_hf, &mut actual);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{w}x{h}, q_hf={q_hf}"
                );
            }
        }
    }

    #[test]
    #[ignore = "DC squeeze benchmark; run in release mode with --ignored --nocapture"]
    fn benchmark_dc_squeeze_search() {
        use std::hint::black_box;
        use std::time::Instant;
        type Search = fn(
            &ImageDim,
            &mut [DcGroupData],
            &mut DistanceParams,
            i32,
            CflFrame,
        ) -> Result<(), EncodeError>;
        for (name, family) in [("smooth", 0), ("patchy", 3), ("textured", 4)] {
            let mut timings = [Vec::new(), Vec::new()];
            for repeat in 0..8 {
                let order = if repeat % 2 == 0 { [0, 1] } else { [1, 0] };
                for mode in order {
                    let (dim, mut groups, mut params, ytob_dc, cfl) =
                        fixture(600, 450, family, true);
                    let search: Search = if mode == 0 {
                        reference_choose_chroma_dc_coding
                    } else {
                        |dim, groups, params, ytob_dc, cfl| {
                            choose_chroma_dc_coding(
                                dim,
                                groups,
                                params,
                                ytob_dc,
                                cfl,
                                false,
                                selected_dc_row_kernels().adjust_b,
                            )
                        }
                    };
                    let start = Instant::now();
                    search(
                        black_box(&dim),
                        black_box(&mut groups),
                        black_box(&mut params),
                        ytob_dc,
                        cfl,
                    )
                    .unwrap();
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    black_box(groups);
                    black_box(params);
                    if repeat != 0 {
                        timings[mode].push(ms);
                    }
                }
            }
            for times in &mut timings {
                times.sort_by(f64::total_cmp);
            }
            let baseline = timings[0][timings[0].len() / 2];
            let optimized = timings[1][timings[1].len() / 2];
            eprintln!(
                "DC squeeze {name} (600x450 blocks): baseline {baseline:.3} ms, optimized {optimized:.3} ms, speedup {:.2}x",
                baseline / optimized
            );
        }
    }
}
