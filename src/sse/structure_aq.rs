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
use crate::sse::adaptive_quant::hsum;
use crate::structure_aq::{Features, finish_block_features};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(never)]
#[target_feature(enable = "sse4.1")]
fn block_mean(block: &[f32; 64]) -> f32 {
    let chunks = block.as_chunks::<4>().0;
    let mut sum0 = _mm_setzero_ps();
    let mut sum1 = _mm_setzero_ps();
    for group in 0..8 {
        let chunk = group * 2;
        sum0 = _mm_add_ps(sum0, unsafe { _mm_loadu_ps(chunks[chunk].as_ptr()) });
        sum1 = _mm_add_ps(sum1, unsafe { _mm_loadu_ps(chunks[chunk + 1].as_ptr()) });
    }
    hsum(_mm_add_ps(sum0, sum1)) * (1.0 / 64.0)
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
fn moments(block: &[f32; 64]) -> (f32, f32) {
    let chunks = block.as_chunks::<4>().0;
    let mean = block_mean(block);
    let mean_v = _mm_set1_ps(mean);
    let mut variance0 = _mm_setzero_ps();
    let mut variance1 = _mm_setzero_ps();
    for group in 0..8 {
        let chunk = group * 2;
        let d0 = _mm_sub_ps(unsafe { _mm_loadu_ps(chunks[chunk].as_ptr()) }, mean_v);
        let d1 = _mm_sub_ps(unsafe { _mm_loadu_ps(chunks[chunk + 1].as_ptr()) }, mean_v);
        variance0 = _mm_add_ps(variance0, _mm_mul_ps(d0, d0));
        variance1 = _mm_add_ps(variance1, _mm_mul_ps(d1, d1));
    }
    let variance = _mm_add_ps(variance0, variance1);
    (mean, hsum(variance) * (1.0 / 64.0))
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86_64")]
fn load2<const X: usize>(row: &[f32; 8]) -> __m128 {
    _mm_setr_ps(row[X], row[X + 1], 0.0, 0.0)
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86_64")]
fn tensor_gradients(left: __m128, right: __m128, top: __m128, bottom: __m128) -> (__m128, __m128) {
    (
        _mm_mul_ps(_mm_sub_ps(right, left), _mm_set1_ps(0.5)),
        _mm_mul_ps(_mm_sub_ps(bottom, top), _mm_set1_ps(0.5)),
    )
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86_64")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    let rows = block.as_chunks::<8>().0;
    let mut jxx = _mm_setzero_ps();
    let mut jxy = _mm_setzero_ps();
    let mut jyy = _mm_setzero_ps();
    for y in 1..7 {
        let (gx, gy) = tensor_gradients(
            unsafe { _mm_loadu_ps(rows[y].as_ptr()) },
            unsafe { _mm_loadu_ps(rows[y].as_ptr().add(2)) },
            unsafe { _mm_loadu_ps(rows[y - 1].as_ptr().add(1)) },
            unsafe { _mm_loadu_ps(rows[y + 1].as_ptr().add(1)) },
        );
        jxx = _mm_add_ps(jxx, _mm_mul_ps(gx, gx));
        jxy = _mm_add_ps(jxy, _mm_mul_ps(gx, gy));
        jyy = _mm_add_ps(jyy, _mm_mul_ps(gy, gy));
    }
    for y in 1..7 {
        let (gx, gy) = tensor_gradients(
            load2::<4>(&rows[y]),
            load2::<6>(&rows[y]),
            load2::<5>(&rows[y - 1]),
            load2::<5>(&rows[y + 1]),
        );
        jxx = _mm_add_ps(jxx, _mm_mul_ps(gx, gx));
        jxy = _mm_add_ps(jxy, _mm_mul_ps(gx, gy));
        jyy = _mm_add_ps(jyy, _mm_mul_ps(gy, gy));
    }
    [hsum(jxx), hsum(jxy), hsum(jyy)]
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86")]
fn tensor_metric_row(block: &[f32; 64], y: usize, metric: usize) -> f32 {
    let rows = block.as_chunks::<8>().0;
    let left = unsafe { _mm_loadu_ps(rows[y].as_ptr()) };
    let right = unsafe { _mm_loadu_ps(rows[y].as_ptr().add(2)) };
    let top = unsafe { _mm_loadu_ps(rows[y - 1].as_ptr().add(1)) };
    let bottom = unsafe { _mm_loadu_ps(rows[y + 1].as_ptr().add(1)) };
    let gx = _mm_mul_ps(_mm_sub_ps(right, left), _mm_set1_ps(0.5));
    let gy = _mm_mul_ps(_mm_sub_ps(bottom, top), _mm_set1_ps(0.5));
    let value = match metric {
        0 => _mm_mul_ps(gx, gx),
        1 => _mm_mul_ps(gx, gy),
        _ => _mm_mul_ps(gy, gy),
    };
    let mut result = hsum(value);
    for x in 5..7 {
        let gx = (rows[y][x + 1] - rows[y][x - 1]) * 0.5;
        let gy = (rows[y + 1][x] - rows[y - 1][x]) * 0.5;
        result += match metric {
            0 => gx * gx,
            1 => gx * gy,
            _ => gy * gy,
        };
    }
    result
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86")]
fn tensor_metric(block: &[f32; 64], metric: usize) -> f32 {
    let mut sum = 0.0;
    for y in 1..7 {
        sum += tensor_metric_row(block, y, metric);
    }
    sum
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86")]
fn tensor(block: &[f32; 64]) -> [f32; 3] {
    [
        tensor_metric(block, 0),
        tensor_metric(block, 1),
        tensor_metric(block, 2),
    ]
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86_64")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    let rows = block.as_chunks::<8>().0;
    let mut error0 = _mm_setzero_ps();
    let mut error1 = _mm_setzero_ps();
    let mut error2 = _mm_setzero_ps();
    let mut error3 = _mm_setzero_ps();
    let mut error4 = _mm_setzero_ps();
    for y in 1..7 {
        let left = unsafe { _mm_loadu_ps(rows[y].as_ptr()) };
        let value = unsafe { _mm_loadu_ps(rows[y].as_ptr().add(1)) };
        let top = unsafe { _mm_loadu_ps(rows[y - 1].as_ptr().add(1)) };
        let top_left = unsafe { _mm_loadu_ps(rows[y - 1].as_ptr()) };
        let bottom_left = unsafe { _mm_loadu_ps(rows[y + 1].as_ptr()) };
        let e0 = _mm_sub_ps(value, left);
        error0 = _mm_add_ps(error0, _mm_mul_ps(e0, e0));
        let e1 = _mm_sub_ps(value, top);
        error1 = _mm_add_ps(error1, _mm_mul_ps(e1, e1));
        let e2 = _mm_sub_ps(value, top_left);
        error2 = _mm_add_ps(error2, _mm_mul_ps(e2, e2));
        let e3 = _mm_sub_ps(value, bottom_left);
        error3 = _mm_add_ps(error3, _mm_mul_ps(e3, e3));
        let e4 = _mm_sub_ps(value, _mm_sub_ps(_mm_add_ps(left, top), top_left));
        error4 = _mm_add_ps(error4, _mm_mul_ps(e4, e4));

        let left = load2::<4>(&rows[y]);
        let value = load2::<5>(&rows[y]);
        let top = load2::<5>(&rows[y - 1]);
        let top_left = load2::<4>(&rows[y - 1]);
        let bottom_left = load2::<4>(&rows[y + 1]);
        let e0 = _mm_sub_ps(value, left);
        error0 = _mm_add_ps(error0, _mm_mul_ps(e0, e0));
        let e1 = _mm_sub_ps(value, top);
        error1 = _mm_add_ps(error1, _mm_mul_ps(e1, e1));
        let e2 = _mm_sub_ps(value, top_left);
        error2 = _mm_add_ps(error2, _mm_mul_ps(e2, e2));
        let e3 = _mm_sub_ps(value, bottom_left);
        error3 = _mm_add_ps(error3, _mm_mul_ps(e3, e3));
        let e4 = _mm_sub_ps(value, _mm_sub_ps(_mm_add_ps(left, top), top_left));
        error4 = _mm_add_ps(error4, _mm_mul_ps(e4, e4));
    }
    [
        hsum(error0),
        hsum(error1),
        hsum(error2),
        hsum(error3),
        hsum(error4),
    ]
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86")]
fn predictor_error_row(block: &[f32; 64], y: usize, predictor: usize) -> f32 {
    let rows = block.as_chunks::<8>().0;
    let left = unsafe { _mm_loadu_ps(rows[y].as_ptr()) };
    let value = unsafe { _mm_loadu_ps(rows[y].as_ptr().add(1)) };
    let top = unsafe { _mm_loadu_ps(rows[y - 1].as_ptr().add(1)) };
    let top_left = unsafe { _mm_loadu_ps(rows[y - 1].as_ptr()) };
    let prediction = match predictor {
        0 => left,
        1 => top,
        2 => top_left,
        3 => unsafe { _mm_loadu_ps(rows[y + 1].as_ptr()) },
        _ => _mm_sub_ps(_mm_add_ps(left, top), top_left),
    };
    let error = _mm_sub_ps(value, prediction);
    let mut result = hsum(_mm_mul_ps(error, error));
    for x in 5..7 {
        let left = rows[y][x - 1];
        let value = rows[y][x];
        let top = rows[y - 1][x];
        let top_left = rows[y - 1][x - 1];
        let prediction = match predictor {
            0 => left,
            1 => top,
            2 => top_left,
            3 => rows[y + 1][x - 1],
            _ => left + top - top_left,
        };
        let error = value - prediction;
        result += error * error;
    }
    result
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86")]
fn predictor_error(block: &[f32; 64], predictor: usize) -> f32 {
    let mut sum = 0.0;
    for y in 1..7 {
        sum += predictor_error_row(block, y, predictor);
    }
    sum
}

#[inline(never)]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86")]
fn predictor_errors(block: &[f32; 64]) -> [f32; 5] {
    [
        predictor_error(block, 0),
        predictor_error(block, 1),
        predictor_error(block, 2),
        predictor_error(block, 3),
        predictor_error(block, 4),
    ]
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn horizontal_energy(row: &[f32], width: usize) -> __m128 {
    if width == 8 {
        let d0 = _mm_sub_ps(unsafe { _mm_loadu_ps(row.as_ptr().add(1)) }, unsafe {
            _mm_loadu_ps(row.as_ptr())
        });
        let d1 = _mm_setr_ps(row[5] - row[4], row[6] - row[5], row[7] - row[6], 0.0);
        _mm_add_ps(_mm_mul_ps(d0, d0), _mm_mul_ps(d1, d1))
    } else {
        let d = _mm_setr_ps(row[1] - row[0], row[2] - row[1], row[3] - row[2], 0.0);
        _mm_mul_ps(d, d)
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn vertical_energy(top: &[f32], bottom: &[f32], width: usize) -> __m128 {
    let d0 = _mm_sub_ps(unsafe { _mm_loadu_ps(bottom.as_ptr()) }, unsafe {
        _mm_loadu_ps(top.as_ptr())
    });
    let mut energy = _mm_mul_ps(d0, d0);
    if width == 8 {
        let d1 = _mm_sub_ps(unsafe { _mm_loadu_ps(bottom.as_ptr().add(4)) }, unsafe {
            _mm_loadu_ps(top.as_ptr().add(4))
        });
        energy = _mm_add_ps(energy, _mm_mul_ps(d1, d1));
    }
    energy
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86_64")]
fn gradient_energy(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let mut sum0 = _mm_setzero_ps();
    let mut sum1 = _mm_setzero_ps();
    let mut sum2 = _mm_setzero_ps();
    let mut sum3 = _mm_setzero_ps();
    for base in (0..height).step_by(4) {
        let row0 = &values[base * stride..][..width];
        let row1 = &values[(base + 1) * stride..][..width];
        let row2 = &values[(base + 2) * stride..][..width];
        let row3 = &values[(base + 3) * stride..][..width];
        sum0 = _mm_add_ps(sum0, horizontal_energy(row0, width));
        sum1 = _mm_add_ps(sum1, horizontal_energy(row1, width));
        sum2 = _mm_add_ps(sum2, horizontal_energy(row2, width));
        sum3 = _mm_add_ps(sum3, horizontal_energy(row3, width));
    }
    let vertical_rows = height - 1;
    for base in (0..vertical_rows).step_by(4) {
        sum0 = _mm_add_ps(
            sum0,
            vertical_energy(
                &values[base * stride..],
                &values[(base + 1) * stride..],
                width,
            ),
        );
        if base + 1 < vertical_rows {
            sum1 = _mm_add_ps(
                sum1,
                vertical_energy(
                    &values[(base + 1) * stride..],
                    &values[(base + 2) * stride..],
                    width,
                ),
            );
        }
        if base + 2 < vertical_rows {
            sum2 = _mm_add_ps(
                sum2,
                vertical_energy(
                    &values[(base + 2) * stride..],
                    &values[(base + 3) * stride..],
                    width,
                ),
            );
        }
        if base + 3 < vertical_rows {
            sum3 = _mm_add_ps(
                sum3,
                vertical_energy(
                    &values[(base + 3) * stride..],
                    &values[(base + 4) * stride..],
                    width,
                ),
            );
        }
    }
    hsum(_mm_add_ps(_mm_add_ps(sum0, sum1), _mm_add_ps(sum2, sum3)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[cfg(target_arch = "x86")]
fn gradient_energy(values: &[f32], stride: usize, width: usize, height: usize) -> f32 {
    let mut sum0 = _mm_setzero_ps();
    let mut sum1 = _mm_setzero_ps();
    for base in (0..height).step_by(2) {
        sum0 = _mm_add_ps(sum0, horizontal_energy(&values[base * stride..], width));
        sum1 = _mm_add_ps(
            sum1,
            horizontal_energy(&values[(base + 1) * stride..], width),
        );
    }
    let vertical_rows = height - 1;
    for base in (0..vertical_rows).step_by(2) {
        sum0 = _mm_add_ps(
            sum0,
            vertical_energy(
                &values[base * stride..],
                &values[(base + 1) * stride..],
                width,
            ),
        );
        if base + 1 < vertical_rows {
            sum1 = _mm_add_ps(
                sum1,
                vertical_energy(
                    &values[(base + 1) * stride..],
                    &values[(base + 2) * stride..],
                    width,
                ),
            );
        }
    }
    hsum(_mm_add_ps(sum0, sum1))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn downsample_8x8(block: &[f32; 64]) -> [f32; 16] {
    let rows = block.as_chunks::<8>().0;
    let mut half = [0.0; 16];
    for y in 0..4 {
        let top0 = unsafe { _mm_loadu_ps(rows[2 * y].as_ptr()) };
        let top1 = unsafe { _mm_loadu_ps(rows[2 * y].as_ptr().add(4)) };
        let bottom0 = unsafe { _mm_loadu_ps(rows[2 * y + 1].as_ptr()) };
        let bottom1 = unsafe { _mm_loadu_ps(rows[2 * y + 1].as_ptr().add(4)) };
        let value = _mm_mul_ps(
            _mm_add_ps(_mm_hadd_ps(top0, top1), _mm_hadd_ps(bottom0, bottom1)),
            _mm_set1_ps(0.25),
        );
        unsafe { _mm_storeu_ps(half[y * 4..].as_mut_ptr(), value) };
    }
    half
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn energy_sum(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut sum = _mm_setzero_ps();
    for chunk in chunks {
        let value = unsafe { _mm_loadu_ps(chunk.as_ptr()) };
        sum = _mm_add_ps(sum, _mm_mul_ps(value, value));
    }
    if !tail.is_empty() {
        let mut padded = [0.0; 4];
        padded[..tail.len()].copy_from_slice(tail);
        let value = unsafe { _mm_loadu_ps(padded.as_ptr()) };
        sum = _mm_add_ps(sum, _mm_mul_ps(value, value));
    }
    hsum(sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
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
/// The caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn block_features_sse41(block: &[f32; 64], dct8x8: &DctFn<8, 8, 64>) -> Features {
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
