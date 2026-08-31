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
use crate::avx::ac_strategy::hsum256;
use crate::dark_aq::{BLUE_OFFSET, INV_BLUE_FULL, Y_TO_LUMA8};
use crate::image::Image3F;
use std::arch::x86_64::*;

/// # Safety
/// The caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) fn fill_blue_tile_avx2(
    opsin: &Image3F,
    tile: &mut [f32],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> f32 {
    assert!(w <= 64 && tile.len() >= 64 * h);
    if w == 0 || h == 0 {
        return 0.0;
    }
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let offset = _mm256_set1_ps(BLUE_OFFSET);
    let inv_full = _mm256_set1_ps(INV_BLUE_FULL);
    let scale = _mm256_set1_ps(Y_TO_LUMA8);
    let sign = _mm256_set1_ps(-0.0);
    let mut sum0 = zero;
    let mut sum1 = zero;
    let mut sum2 = zero;
    let mut sum3 = zero;
    let groups = w / 32;
    let vector_tail = groups * 32;
    let full_tail = w % 32 / 8;
    let tail = vector_tail + full_tail * 8;
    let remainder = w - tail;
    macro_rules! accumulate_blue_x8 {
        ($sum:ident, $x:expr, $xr:ident, $yr:ident, $br:ident, $dst:ident) => {{
            let x = $x;
            let px = x0 + x;
            let xv = unsafe { _mm256_loadu_ps($xr.as_ptr().add(px)) };
            let yv = unsafe { _mm256_loadu_ps($yr.as_ptr().add(px)) };
            let bv = unsafe { _mm256_loadu_ps($br.as_ptr().add(px)) };
            let by = _mm256_sub_ps(bv, yv);
            let abs_x = _mm256_andnot_ps(sign, xv);
            let excess = _mm256_sub_ps(by, _mm256_add_ps(abs_x, offset));
            let risk = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(excess, inv_full), zero), one);
            $sum = _mm256_add_ps($sum, risk);
            unsafe { _mm256_storeu_ps($dst.as_mut_ptr().add(x), _mm256_mul_ps(by, scale)) };
        }};
    }
    for (r, dst) in tile.as_chunks_mut::<64>().0.iter_mut().take(h).enumerate() {
        let xr = opsin.plane_row(0, y0 + r);
        let yr = opsin.plane_row(1, y0 + r);
        let br = opsin.plane_row(2, y0 + r);
        for group in 0..groups {
            let x = group * 32;
            accumulate_blue_x8!(sum0, x, xr, yr, br, dst);
            accumulate_blue_x8!(sum1, x + 8, xr, yr, br, dst);
            accumulate_blue_x8!(sum2, x + 16, xr, yr, br, dst);
            accumulate_blue_x8!(sum3, x + 24, xr, yr, br, dst);
        }
        if full_tail > 0 {
            accumulate_blue_x8!(sum0, vector_tail, xr, yr, br, dst);
        }
        if full_tail > 1 {
            accumulate_blue_x8!(sum1, vector_tail + 8, xr, yr, br, dst);
        }
        if full_tail > 2 {
            accumulate_blue_x8!(sum2, vector_tail + 16, xr, yr, br, dst);
        }
        if remainder != 0 {
            let px = x0 + tail;
            let mask = first_lanes_mask(remainder);
            let xv = unsafe { _mm256_maskload_ps(xr.as_ptr().add(px), mask) };
            let yv = unsafe { _mm256_maskload_ps(yr.as_ptr().add(px), mask) };
            let bv = unsafe { _mm256_maskload_ps(br.as_ptr().add(px), mask) };
            let by = _mm256_sub_ps(bv, yv);
            let abs_x = _mm256_andnot_ps(sign, xv);
            let excess = _mm256_sub_ps(by, _mm256_add_ps(abs_x, offset));
            let risk = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(excess, inv_full), zero), one);
            match full_tail {
                0 => sum0 = _mm256_add_ps(sum0, risk),
                1 => sum1 = _mm256_add_ps(sum1, risk),
                2 => sum2 = _mm256_add_ps(sum2, risk),
                3 => sum3 = _mm256_add_ps(sum3, risk),
                _ => unreachable!(),
            }
            unsafe {
                _mm256_maskstore_ps(dst.as_mut_ptr().add(tail), mask, _mm256_mul_ps(by, scale))
            };
        }
    }
    let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    hsum256(sum) / (w * h) as f32
}

#[inline]
#[target_feature(enable = "avx2")]
fn first_lanes_mask(n: usize) -> __m256i {
    _mm256_cmpgt_epi32(
        _mm256_set1_epi32(n as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_x8(sum: __m256, row: &[f32], x: usize) -> __m256 {
    _mm256_add_ps(sum, unsafe { _mm256_loadu_ps(row.as_ptr().add(x)) })
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_masked(sum: __m256, row: &[f32], x: usize, mask: __m256i) -> __m256 {
    _mm256_add_ps(sum, unsafe {
        _mm256_maskload_ps(row.as_ptr().add(x), mask)
    })
}

#[inline]
#[target_feature(enable = "avx2")]
fn sum_rows(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let groups = w / 32;
    let full_tail = w % 32 / 8;
    let remainder = w % 8;
    let tail_mask = first_lanes_mask(remainder);
    for row in buf.chunks_exact(stride).take(h) {
        for group in 0..groups {
            let x = group * 32;
            sum0 = add_x8(sum0, row, x);
            sum1 = add_x8(sum1, row, x + 8);
            sum2 = add_x8(sum2, row, x + 16);
            sum3 = add_x8(sum3, row, x + 24);
        }
        let mut x = groups * 32;
        if full_tail > 0 {
            sum0 = add_x8(sum0, row, x);
            x += 8;
        }
        if full_tail > 1 {
            sum1 = add_x8(sum1, row, x);
            x += 8;
        }
        if full_tail > 2 {
            sum2 = add_x8(sum2, row, x);
            x += 8;
        }
        if remainder != 0 {
            match full_tail {
                0 => sum0 = add_masked(sum0, row, x, tail_mask),
                1 => sum1 = add_masked(sum1, row, x, tail_mask),
                2 => sum2 = add_masked(sum2, row, x, tail_mask),
                _ => sum3 = add_masked(sum3, row, x, tail_mask),
            }
        }
    }
    let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    hsum256(sum)
}

#[inline]
#[target_feature(enable = "avx2")]
fn laplacian_abs_x8(
    sum: __m256,
    top: &[f32],
    middle: &[f32],
    bottom: &[f32],
    x: usize,
    four: __m256,
    sign: __m256,
) -> __m256 {
    let up = unsafe { _mm256_loadu_ps(top.as_ptr().add(x)) };
    let down = unsafe { _mm256_loadu_ps(bottom.as_ptr().add(x)) };
    let left = unsafe { _mm256_loadu_ps(middle.as_ptr().add(x - 1)) };
    let center = unsafe { _mm256_loadu_ps(middle.as_ptr().add(x)) };
    let right = unsafe { _mm256_loadu_ps(middle.as_ptr().add(x + 1)) };
    let neighbors = _mm256_add_ps(_mm256_add_ps(up, down), _mm256_add_ps(left, right));
    let lap = _mm256_sub_ps(_mm256_mul_ps(center, four), neighbors);
    _mm256_add_ps(sum, _mm256_andnot_ps(sign, lap))
}

#[inline]
#[target_feature(enable = "avx2")]
fn laplacian_abs_masked(
    sum: __m256,
    top: &[f32],
    middle: &[f32],
    bottom: &[f32],
    x: usize,
    mask: __m256i,
    four: __m256,
    sign: __m256,
) -> __m256 {
    let up = unsafe { _mm256_maskload_ps(top.as_ptr().add(x), mask) };
    let down = unsafe { _mm256_maskload_ps(bottom.as_ptr().add(x), mask) };
    let left = unsafe { _mm256_maskload_ps(middle.as_ptr().add(x - 1), mask) };
    let center = unsafe { _mm256_maskload_ps(middle.as_ptr().add(x), mask) };
    let right = unsafe { _mm256_maskload_ps(middle.as_ptr().add(x + 1), mask) };
    let neighbors = _mm256_add_ps(_mm256_add_ps(up, down), _mm256_add_ps(left, right));
    let lap = _mm256_sub_ps(_mm256_mul_ps(center, four), neighbors);
    _mm256_add_ps(sum, _mm256_andnot_ps(sign, lap))
}

#[inline]
#[target_feature(enable = "avx2")]
fn laplacian_abs_sum(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let four = _mm256_set1_ps(4.0);
    let sign = _mm256_set1_ps(-0.0);
    let interior = w - 2;
    let groups = interior / 32;
    let full_tail = interior % 32 / 8;
    let remainder = interior % 8;
    let tail_mask = first_lanes_mask(remainder);
    for y in 1..h - 1 {
        let top = &buf[(y - 1) * stride..];
        let middle = &buf[y * stride..];
        let bottom = &buf[(y + 1) * stride..];
        for group in 0..groups {
            let x = 1 + group * 32;
            sum0 = laplacian_abs_x8(sum0, top, middle, bottom, x, four, sign);
            sum1 = laplacian_abs_x8(sum1, top, middle, bottom, x + 8, four, sign);
            sum2 = laplacian_abs_x8(sum2, top, middle, bottom, x + 16, four, sign);
            sum3 = laplacian_abs_x8(sum3, top, middle, bottom, x + 24, four, sign);
        }
        let mut x = 1 + groups * 32;
        if full_tail > 0 {
            sum0 = laplacian_abs_x8(sum0, top, middle, bottom, x, four, sign);
            x += 8;
        }
        if full_tail > 1 {
            sum1 = laplacian_abs_x8(sum1, top, middle, bottom, x, four, sign);
            x += 8;
        }
        if full_tail > 2 {
            sum2 = laplacian_abs_x8(sum2, top, middle, bottom, x, four, sign);
            x += 8;
        }
        if remainder != 0 {
            match full_tail {
                0 => {
                    sum0 = laplacian_abs_masked(sum0, top, middle, bottom, x, tail_mask, four, sign)
                }
                1 => {
                    sum1 = laplacian_abs_masked(sum1, top, middle, bottom, x, tail_mask, four, sign)
                }
                2 => {
                    sum2 = laplacian_abs_masked(sum2, top, middle, bottom, x, tail_mask, four, sign)
                }
                _ => {
                    sum3 = laplacian_abs_masked(sum3, top, middle, bottom, x, tail_mask, four, sign)
                }
            }
        }
    }
    let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    hsum256(sum)
}

#[inline]
#[target_feature(enable = "avx2")]
fn downsample_x4(top: &[f32; 8], bottom: &[f32; 8], dst: &mut [f32; 4]) {
    let top = unsafe { _mm256_loadu_ps(top.as_ptr()) };
    let bottom = unsafe { _mm256_loadu_ps(bottom.as_ptr()) };
    let pairs = _mm256_hadd_ps(top, bottom);
    let top_pairs = _mm256_permutevar8x32_ps(pairs, _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0));
    let bottom_pairs = _mm256_permutevar8x32_ps(pairs, _mm256_setr_epi32(2, 3, 6, 7, 0, 0, 0, 0));
    let value = _mm256_mul_ps(_mm256_add_ps(top_pairs, bottom_pairs), _mm256_set1_ps(0.25));
    unsafe { _mm_storeu_ps(dst.as_mut_ptr(), _mm256_castps256_ps128(value)) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn box_downsample_2x(
    src: &[f32],
    src_stride: usize,
    h: usize,
    w: usize,
    dst: &mut [f32],
    dst_stride: usize,
) -> (usize, usize) {
    let (hh, ww) = (h / 2, w / 2);
    for y in 0..hh {
        let top = &src[(2 * y) * src_stride..];
        let bottom = &src[(2 * y + 1) * src_stride..];
        let dst_row = &mut dst[y * dst_stride..][..ww];
        let (dst4, tail) = dst_row.as_chunks_mut::<4>();
        for (chunk, out) in dst4.iter_mut().enumerate() {
            let x = chunk * 8;
            downsample_x4(
                top[x..].first_chunk().unwrap(),
                bottom[x..].first_chunk().unwrap(),
                out,
            );
        }
        if !tail.is_empty() {
            let x = dst4.len() * 8;
            let source_len = tail.len() * 2;
            let mut top_pad = [0.0; 8];
            let mut bottom_pad = [0.0; 8];
            let mut out = [0.0; 4];
            top_pad[..source_len].copy_from_slice(&top[x..x + source_len]);
            bottom_pad[..source_len].copy_from_slice(&bottom[x..x + source_len]);
            downsample_x4(&top_pad, &bottom_pad, &mut out);
            tail.copy_from_slice(&out[..tail.len()]);
        }
    }
    (hh, ww)
}

/// # Safety
/// The caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) fn dark_structure_stats_avx2(buf: &[[f32; 64]], h: usize, w: usize) -> (f32, f32) {
    assert!(h <= 64 && w <= 64 && buf.len() >= h);
    if h == 0 || w == 0 {
        return (0.0, 0.0);
    }
    let flat = buf.as_flattened();
    let mean = sum_rows(flat, 64, h, w) / (h * w) as f32;
    if h < 3 || w < 3 {
        return (mean, 0.0);
    }
    let lap_full = laplacian_abs_sum(flat, 64, h, w) / ((h - 2) * (w - 2)) as f32;
    let mut half = [[0.0f32; 32]; 32];
    let (hh, ww) = box_downsample_2x(flat, 64, h, w, half.as_flattened_mut(), 32);
    if hh < 3 || ww < 3 {
        return (mean, 0.0);
    }
    let lap_half =
        laplacian_abs_sum(half.as_flattened(), 32, hh, ww) / ((hh - 2) * (ww - 2)) as f32;
    (mean, (lap_full * lap_half).sqrt())
}
