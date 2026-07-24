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

use crate::adaptive_quant::fast_exp2;
use crate::dct::{DctFn, fmla};
use crate::image::{Image3F, ImageB};
use crate::util::FastRound;

#[derive(Clone, Copy, Debug, Default)]
struct Features {
    persistence: f32,
    coherence: f32,
    predictability: f32,
    mid_share: f32,
    high_share: f32,
    gradient_presence: f32,
}

#[inline]
fn strength(distance: f32) -> f32 {
    static POINTS: &[(f32, f32)] = &[
        (1.5, 0.0),
        (2.0, 0.08),
        (3.0, 0.13),
        (4.0, 0.12),
        (4.5, 0.06),
        (5.0, 0.0),
    ];
    if distance <= POINTS[0].0 {
        return 0.0;
    }
    for pair in POINTS.array_windows::<2>() {
        let (d0, v0) = pair[0];
        let (d1, v1) = pair[1];
        if distance <= d1 {
            return v0 + (distance - d0) * (v1 - v0) / (d1 - d0);
        }
    }
    POINTS[POINTS.len() - 1].1
}

fn block_features(block: &[f32; 64], dct8x8: &DctFn<64>) -> Features {
    const EPS: f32 = 1.0e-10;
    let rows = block.as_chunks::<8>().0;
    let mean = block.iter().sum::<f32>() * (1.0 / 64.0);
    let variance = block
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        * (1.0 / 64.0);

    let (mut jxx, mut jxy, mut jyy) = (0.0f32, 0.0f32, 0.0f32);
    let mut tensor_n = 0.0f32;
    for row_window in rows.array_windows::<3>() {
        let [top, middle, bottom] = row_window;
        for ((&left, &right), (&top, &bottom)) in middle[..6]
            .iter()
            .zip(&middle[2..])
            .zip(top[1..7].iter().zip(&bottom[1..7]))
        {
            let gx = 0.5 * (right - left);
            let gy = 0.5 * (bottom - top);
            jxx = fmla(gx, gx, jxx);
            jxy = fmla(gx, gy, jxy);
            jyy = fmla(gy, gy, jyy);
            tensor_n += 1.0;
        }
    }
    let trace = jxx + jyy;
    let coherence =
        (((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy).sqrt() / (trace + EPS)).clamp(0.0, 1.0);
    let gradient_presence = (trace / (tensor_n * variance + EPS)).clamp(0.0, 1.0);

    let mut errors = [0.0f32; 5];
    let mut pred_n = 0.0f32;
    for row_window in rows.array_windows::<3>() {
        let [top_row, middle, bottom_row] = row_window;
        for ((((&left, &v), &top), &top_left), &bottom_left) in middle[..6]
            .iter()
            .zip(&middle[1..7])
            .zip(&top_row[1..7])
            .zip(&top_row[..6])
            .zip(&bottom_row[..6])
        {
            for (slot, prediction) in [left, top, top_left, bottom_left, left + top - top_left]
                .into_iter()
                .enumerate()
            {
                let e = v - prediction;
                errors[slot] = fmla(e, e, errors[slot]);
            }
            pred_n += 1.0;
        }
    }
    let min_error = errors.into_iter().fold(f32::INFINITY, f32::min) / pred_n;
    let predictability = (1.0 - min_error / (variance + EPS)).clamp(0.0, 1.0);

    let mut energy_1x = 0.0f32;
    for row in rows {
        for pair in row.array_windows::<2>() {
            let d = pair[1] - pair[0];
            energy_1x = fmla(d, d, energy_1x);
        }
    }
    for pair in rows.array_windows::<2>() {
        for (&top, &bottom) in pair[0].iter().zip(&pair[1]) {
            let d = bottom - top;
            energy_1x = fmla(d, d, energy_1x);
        }
    }
    energy_1x *= 1.0 / 112.0;
    let mut half = [0.0f32; 16];
    for (src_rows, dst) in rows
        .as_chunks::<2>()
        .0
        .iter()
        .zip(half.as_chunks_mut::<4>().0.iter_mut())
    {
        for ((top, bottom), out) in src_rows[0]
            .as_chunks::<2>()
            .0
            .iter()
            .zip(src_rows[1].as_chunks::<2>().0.iter())
            .zip(dst.iter_mut())
        {
            *out = 0.25 * (top[0] + top[1] + bottom[0] + bottom[1]);
        }
    }
    let half_rows = half.as_chunks::<4>().0;
    let mut energy_2x = 0.0f32;
    for row in half_rows {
        for pair in row.array_windows::<2>() {
            let d = pair[1] - pair[0];
            energy_2x = fmla(d, d, energy_2x);
        }
    }
    for pair in half_rows.array_windows::<2>() {
        for (&top, &bottom) in pair[0].iter().zip(&pair[1]) {
            let d = bottom - top;
            energy_2x = fmla(d, d, energy_2x);
        }
    }
    energy_2x *= 1.0 / 24.0;
    let persistence_ratio = energy_2x / (energy_1x + EPS);
    let persistence = (persistence_ratio / (1.0 + persistence_ratio)).clamp(0.0, 1.0);

    let mut coeffs = [0.0f32; 64];
    dct8x8(block, &mut coeffs);
    let (mut mid, mut high) = (0.0f32, 0.0f32);
    for (y, row) in coeffs.as_chunks::<8>().0.iter().enumerate() {
        for (x, &coeff) in row.iter().enumerate() {
            if x == 0 && y == 0 {
                continue;
            }
            let e = coeff * coeff;
            let band = x + y;
            if (2..=7).contains(&band) {
                mid += e;
            } else if band >= 8 {
                high += e;
            }
        }
    }
    let spectral = mid + high + EPS;
    Features {
        persistence,
        coherence,
        predictability,
        mid_share: mid / spectral,
        high_share: high / spectral,
        gradient_presence,
    }
}

#[inline]
fn correction_score(f: Features, distance: f32) -> f32 {
    let structure =
        0.30 * f.persistence + 0.25 * f.coherence + 0.25 * f.predictability + 0.20 * f.mid_share;
    let noise =
        f.high_share * (1.0 - f.coherence) * (1.0 - f.predictability) * (1.0 - f.persistence);
    let weak_edge = f.coherence * f.predictability * f.gradient_presence;
    let banding = weak_edge * f.mid_share * (1.0 - f.high_share);
    let low_quality_mix = ((distance - 3.5) / 2.5).clamp(0.0, 1.0);
    structure + (0.20 + 0.35 * low_quality_mix) * weak_edge + 0.35 * low_quality_mix * banding
        - (0.90 - 0.15 * low_quality_mix) * noise
}

pub(crate) fn apply(
    corrections: &mut Vec<f32>,
    opsin: &Image3F,
    field: &mut ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
    dct8x8: &DctFn<64>,
) {
    let amount = strength(distance);
    if amount == 0.0 || field.xsize() == 0 || field.ysize() == 0 {
        return;
    }
    let field_width = field.xsize();
    let field_height = field.ysize();
    if corrections.len() < field_width * field_height {
        corrections.resize(field_width * field_height, 0.0);
    }
    let corrections = &mut corrections[..field_width * field_height];
    let mut weighted_sum = 0.0f32;
    let mut total_weight = 0.0f32;
    for (by, correction_row) in corrections.chunks_exact_mut(field_width).enumerate() {
        let py = y0 + by * 8;
        let h = opsin.ysize().saturating_sub(py).min(8);
        for (bx, correction_out) in correction_row.iter_mut().enumerate() {
            let px = x0 + bx * 8;
            let w = opsin.xsize().saturating_sub(px).min(8);
            let mut block = [0.0f32; 64];
            let block_rows = block.as_chunks_mut::<8>().0;
            if w == 8 && h == 8 {
                for (dst, src_y) in block_rows.iter_mut().zip(py..py + 8) {
                    let source = &opsin.plane_row(1, src_y)[px..];
                    dst.copy_from_slice(&source.as_chunks::<8>().0[0]);
                }
            } else {
                for (dy, dst) in block_rows.iter_mut().enumerate() {
                    let row = opsin.plane_row(1, (py + dy).min(opsin.ysize() - 1));
                    for (dx, out) in dst.iter_mut().enumerate() {
                        *out = row[(px + dx).min(opsin.xsize() - 1)];
                    }
                }
            }
            let correction = correction_score(block_features(&block, dct8x8), distance);
            *correction_out = correction;
            let weight = (w * h) as f32;
            weighted_sum = fmla(weight, correction, weighted_sum);
            total_weight += weight;
        }
    }
    let center = if total_weight == 0.0 {
        0.0
    } else {
        weighted_sum / total_weight
    };
    let mut weighted_variance = 0.0f32;
    for (by, correction_row) in corrections.chunks_exact(field_width).enumerate() {
        let py = y0 + by * 8;
        let h = opsin.ysize().saturating_sub(py).min(8);
        for (bx, &correction) in correction_row.iter().enumerate() {
            let px = x0 + bx * 8;
            let w = opsin.xsize().saturating_sub(px).min(8);
            let d = correction - center;
            weighted_variance += (w * h) as f32 * d * d;
        }
    }
    let inv_stddev = if weighted_variance == 0.0 {
        0.0
    } else {
        (total_weight / weighted_variance).sqrt()
    };
    for (by, correction_row) in corrections.chunks_exact(field_width).enumerate() {
        for (q, &correction) in field.row_mut(by).iter_mut().zip(correction_row) {
            let delta = (-amount * (correction - center) * inv_stddev).clamp(-0.18, 0.22);
            *q = (*q as f32 * fast_exp2(delta))
                .fast_round()
                .clamp(1.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherent_line_scores_above_checker_noise() {
        let mut line = [0.0f32; 64];
        let mut checker = [0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                line[y * 8 + x] = if x < 4 { 0.2 } else { 0.8 };
                checker[y * 8 + x] = if (x + y) & 1 == 0 { 0.2 } else { 0.8 };
            }
        }
        let dct = crate::dct::selected_dct8x8();
        let line_f = block_features(&line, dct);
        let noise_f = block_features(&checker, dct);
        assert!(line_f.coherence > noise_f.coherence);
        assert!(correction_score(line_f, 3.0) > correction_score(noise_f, 3.0));
    }
}
