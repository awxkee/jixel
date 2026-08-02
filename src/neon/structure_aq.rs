/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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
use crate::dct::{DctFn, DctInput};
use crate::structure_aq::{Features, finish_block_features};
use std::arch::aarch64::*;

#[inline(never)]
#[target_feature(enable = "neon")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<4>().0;
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    for group in 0..4 {
        let chunk = group * 4;
        sum0 = vaddq_f32(sum0, unsafe { vld1q_f32(chunks[chunk].as_ptr()) });
        sum1 = vaddq_f32(sum1, unsafe { vld1q_f32(chunks[chunk + 1].as_ptr()) });
        sum2 = vaddq_f32(sum2, unsafe { vld1q_f32(chunks[chunk + 2].as_ptr()) });
        sum3 = vaddq_f32(sum3, unsafe { vld1q_f32(chunks[chunk + 3].as_ptr()) });
    }
    let mean = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3))) * (1.0 / 64.0);
    let mean_v = vdupq_n_f32(mean);
    let mut variance0 = vdupq_n_f32(0.0);
    let mut variance1 = vdupq_n_f32(0.0);
    let mut variance2 = vdupq_n_f32(0.0);
    let mut variance3 = vdupq_n_f32(0.0);
    for group in 0..4 {
        let chunk = group * 4;
        let d0 = vsubq_f32(unsafe { vld1q_f32(chunks[chunk].as_ptr()) }, mean_v);
        let d1 = vsubq_f32(unsafe { vld1q_f32(chunks[chunk + 1].as_ptr()) }, mean_v);
        let d2 = vsubq_f32(unsafe { vld1q_f32(chunks[chunk + 2].as_ptr()) }, mean_v);
        let d3 = vsubq_f32(unsafe { vld1q_f32(chunks[chunk + 3].as_ptr()) }, mean_v);
        variance0 = vfmaq_f32(variance0, d0, d0);
        variance1 = vfmaq_f32(variance1, d1, d1);
        variance2 = vfmaq_f32(variance2, d2, d2);
        variance3 = vfmaq_f32(variance3, d3, d3);
    }
    let variance = vaddq_f32(
        vaddq_f32(variance0, variance1),
        vaddq_f32(variance2, variance3),
    );
    (mean, vaddvq_f32(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "neon")]
fn load2(row: &[f32], x: usize) -> float32x4_t {
    vcombine_f32(unsafe { vld1_f32(row.as_ptr().add(x)) }, vdup_n_f32(0.0))
}

#[inline]
#[target_feature(enable = "neon")]
fn tensor_gradients(
    left: float32x4_t,
    right: float32x4_t,
    top: float32x4_t,
    bottom: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    (
        vmulq_n_f32(vsubq_f32(right, left), 0.5),
        vmulq_n_f32(vsubq_f32(bottom, top), 0.5),
    )
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mut jxx = vdupq_n_f32(0.0);
    let mut jxy = vdupq_n_f32(0.0);
    let mut jyy = vdupq_n_f32(0.0);
    for y in 1..7 {
        let (gx, gy) = tensor_gradients(
            unsafe { vld1q_f32(rows[y].as_ptr()) },
            unsafe { vld1q_f32(rows[y].as_ptr().add(2)) },
            unsafe { vld1q_f32(rows[y - 1].as_ptr().add(1)) },
            unsafe { vld1q_f32(rows[y + 1].as_ptr().add(1)) },
        );
        jxx = vfmaq_f32(jxx, gx, gx);
        jxy = vfmaq_f32(jxy, gx, gy);
        jyy = vfmaq_f32(jyy, gy, gy);
        let (gx, gy) = tensor_gradients(
            load2(&rows[y], 4),
            load2(&rows[y], 6),
            load2(&rows[y - 1], 5),
            load2(&rows[y + 1], 5),
        );
        jxx = vfmaq_f32(jxx, gx, gx);
        jxy = vfmaq_f32(jxy, gx, gy);
        jyy = vfmaq_f32(jyy, gy, gy);
    }
    [vaddvq_f32(jxx), vaddvq_f32(jxy), vaddvq_f32(jyy)]
}

#[inline]
#[target_feature(enable = "neon")]
fn predictor_residuals(
    value: float32x4_t,
    left: float32x4_t,
    top: float32x4_t,
    top_left: float32x4_t,
    bottom_left: float32x4_t,
) -> (
    float32x4_t,
    float32x4_t,
    float32x4_t,
    float32x4_t,
    float32x4_t,
) {
    (
        vsubq_f32(value, left),
        vsubq_f32(value, top),
        vsubq_f32(value, top_left),
        vsubq_f32(value, bottom_left),
        vsubq_f32(value, vsubq_f32(vaddq_f32(left, top), top_left)),
    )
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mut error0 = vdupq_n_f32(0.0);
    let mut error1 = vdupq_n_f32(0.0);
    let mut error2 = vdupq_n_f32(0.0);
    let mut error3 = vdupq_n_f32(0.0);
    let mut error4 = vdupq_n_f32(0.0);
    for y in 1..7 {
        let left = unsafe { vld1q_f32(rows[y].as_ptr()) };
        let value = unsafe { vld1q_f32(rows[y].as_ptr().add(1)) };
        let top = unsafe { vld1q_f32(rows[y - 1].as_ptr().add(1)) };
        let top_left = unsafe { vld1q_f32(rows[y - 1].as_ptr()) };
        let bottom_left = unsafe { vld1q_f32(rows[y + 1].as_ptr()) };
        let (e0, e1, e2, e3, e4) = predictor_residuals(value, left, top, top_left, bottom_left);
        error0 = vfmaq_f32(error0, e0, e0);
        error1 = vfmaq_f32(error1, e1, e1);
        error2 = vfmaq_f32(error2, e2, e2);
        error3 = vfmaq_f32(error3, e3, e3);
        error4 = vfmaq_f32(error4, e4, e4);
        let (e0, e1, e2, e3, e4) = predictor_residuals(
            load2(&rows[y], 5),
            load2(&rows[y], 4),
            load2(&rows[y - 1], 5),
            load2(&rows[y - 1], 4),
            load2(&rows[y + 1], 4),
        );
        error0 = vfmaq_f32(error0, e0, e0);
        error1 = vfmaq_f32(error1, e1, e1);
        error2 = vfmaq_f32(error2, e2, e2);
        error3 = vfmaq_f32(error3, e3, e3);
        error4 = vfmaq_f32(error4, e4, e4);
    }
    [
        vaddvq_f32(error0),
        vaddvq_f32(error1),
        vaddvq_f32(error2),
        vaddvq_f32(error3),
        vaddvq_f32(error4),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn horizontal_energy(row: &[f32], width: usize) -> float32x4_t {
    let tail = vsetq_lane_f32::<2>(
        row[width - 1] - row[width - 2],
        vsetq_lane_f32::<1>(
            row[width - 2] - row[width - 3],
            vsetq_lane_f32::<0>(row[width - 3] - row[width - 4], vdupq_n_f32(0.0)),
        ),
    );
    let mut energy = vmulq_f32(tail, tail);
    if width == 8 {
        let d = vsubq_f32(unsafe { vld1q_f32(row.as_ptr().add(1)) }, unsafe {
            vld1q_f32(row.as_ptr())
        });
        energy = vfmaq_f32(energy, d, d);
    }
    energy
}

#[inline]
#[target_feature(enable = "neon")]
fn vertical_energy(top: &[f32], bottom: &[f32], width: usize) -> float32x4_t {
    let d0 = vsubq_f32(unsafe { vld1q_f32(bottom.as_ptr()) }, unsafe {
        vld1q_f32(top.as_ptr())
    });
    let mut energy = vmulq_f32(d0, d0);
    if width == 8 {
        let d1 = vsubq_f32(unsafe { vld1q_f32(bottom.as_ptr().add(4)) }, unsafe {
            vld1q_f32(top.as_ptr().add(4))
        });
        energy = vfmaq_f32(energy, d1, d1);
    }
    energy
}

#[inline]
#[target_feature(enable = "neon")]
fn gradient_energy(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    for base in (0..height).step_by(4) {
        let row0 = &values[base * stride..][..width];
        let row1 = &values[(base + 1) * stride..][..width];
        let row2 = &values[(base + 2) * stride..][..width];
        let row3 = &values[(base + 3) * stride..][..width];
        sum0 = vaddq_f32(sum0, horizontal_energy(row0, width));
        sum1 = vaddq_f32(sum1, horizontal_energy(row1, width));
        sum2 = vaddq_f32(sum2, horizontal_energy(row2, width));
        sum3 = vaddq_f32(sum3, horizontal_energy(row3, width));
    }
    let vertical_rows = height - 1;
    for base in (0..vertical_rows).step_by(4) {
        sum0 = vaddq_f32(
            sum0,
            vertical_energy(
                &values[base * stride..],
                &values[(base + 1) * stride..],
                width,
            ),
        );
        if base + 1 < vertical_rows {
            sum1 = vaddq_f32(
                sum1,
                vertical_energy(
                    &values[(base + 1) * stride..],
                    &values[(base + 2) * stride..],
                    width,
                ),
            );
        }
        if base + 2 < vertical_rows {
            sum2 = vaddq_f32(
                sum2,
                vertical_energy(
                    &values[(base + 2) * stride..],
                    &values[(base + 3) * stride..],
                    width,
                ),
            );
        }
        if base + 3 < vertical_rows {
            sum3 = vaddq_f32(
                sum3,
                vertical_energy(
                    &values[(base + 3) * stride..],
                    &values[(base + 4) * stride..],
                    width,
                ),
            );
        }
    }
    vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)))
}

#[inline]
#[target_feature(enable = "neon")]
fn downsample_8x8(block: &[f32; 64]) -> [f32; 16] {
    let rows = block.as_chunks::<8>().0;
    let mut half = [0.0; 16];
    for y in 0..4 {
        let top0 = unsafe { vld1q_f32(rows[2 * y].as_ptr()) };
        let top1 = unsafe { vld1q_f32(rows[2 * y].as_ptr().add(4)) };
        let bottom0 = unsafe { vld1q_f32(rows[2 * y + 1].as_ptr()) };
        let bottom1 = unsafe { vld1q_f32(rows[2 * y + 1].as_ptr().add(4)) };
        let value = vmulq_n_f32(
            vaddq_f32(vpaddq_f32(top0, top1), vpaddq_f32(bottom0, bottom1)),
            0.25,
        );
        unsafe { vst1q_f32(half[y * 4..].as_mut_ptr(), value) };
    }
    half
}

#[inline]
#[target_feature(enable = "neon")]
fn energy_sum(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut sum = vdupq_n_f32(0.0);
    for chunk in chunks {
        let value = unsafe { vld1q_f32(chunk.as_ptr()) };
        sum = vfmaq_f32(sum, value, value);
    }
    if !tail.is_empty() {
        let mut padded = [0.0; 4];
        padded[..tail.len()].copy_from_slice(tail);
        let value = unsafe { vld1q_f32(padded.as_ptr()) };
        sum = vfmaq_f32(sum, value, value);
    }
    vaddvq_f32(sum)
}

#[inline]
#[target_feature(enable = "neon")]
fn spectral_energy(coeffs: &[f32; 64]) -> (f32, f32) {
    let mut mid = 0.0;
    let mut high = 0.0;
    for (y, row) in coeffs.as_chunks::<8>().0.iter().enumerate() {
        let mid_start = 2usize.saturating_sub(y);
        let mid_end = (8 - y).min(8);
        mid += energy_sum(&row[mid_start..mid_end]);
        high += energy_sum(&row[8usize.saturating_sub(y)..]);
    }
    (mid, high)
}

/// # Safety
/// AArch64 NEON must be available.
#[target_feature(enable = "neon")]
pub(crate) fn block_features_neon(block: &[f32; 64], dct8x8: &DctFn<8, 8, 64>) -> Features {
    let (_, variance) = moments(block);
    let tensor = tensor(block);
    let errors = predictor_errors(block);
    let energy_1x = gradient_energy(block, 8, 8, 8);
    let half = downsample_8x8(block);
    let energy_2x = gradient_energy(&half, 4, 4, 4);
    let mut coeffs = [0.0; 64];
    dct8x8(DctInput::from_flat(block), &mut coeffs);
    let (mid, high) = spectral_energy(&coeffs);
    finish_block_features(variance, tensor, errors, energy_1x, energy_2x, mid, high)
}
