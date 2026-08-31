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

use crate::dark_aq::{BLUE_FULL, BLUE_OFFSET, Y_TO_LUMA8};
use crate::image::Image3F;
use std::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub(crate) fn fill_blue_tile_wasm(
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
    let zero = f32x4_splat(0.0);
    let one = f32x4_splat(1.0);
    let offset = f32x4_splat(BLUE_OFFSET);
    let inv_full = f32x4_splat(1.0 / BLUE_FULL);
    let scale = f32x4_splat(Y_TO_LUMA8);
    let mut sum0 = zero;
    let mut sum1 = zero;
    let mut sum2 = zero;
    let mut sum3 = zero;
    let mut scalar_sum = 0.0f32;
    let groups = w / 16;
    let vector_tail = groups * 16;
    let full_tail = w % 16 / 4;
    let tail = vector_tail + full_tail * 4;
    macro_rules! accumulate_blue_x4 {
        ($sum:ident, $x:expr, $xr:ident, $yr:ident, $br:ident, $dst:ident) => {{
            let x = $x;
            let px = x0 + x;
            let xv = unsafe { v128_load($xr.as_ptr().add(px).cast()) };
            let yv = unsafe { v128_load($yr.as_ptr().add(px).cast()) };
            let bv = unsafe { v128_load($br.as_ptr().add(px).cast()) };
            let by = f32x4_sub(bv, yv);
            let excess = f32x4_sub(by, f32x4_add(f32x4_abs(xv), offset));
            let risk = f32x4_min(f32x4_max(f32x4_mul(excess, inv_full), zero), one);
            $sum = f32x4_add($sum, risk);
            unsafe { v128_store($dst.as_mut_ptr().add(x).cast(), f32x4_mul(by, scale)) };
        }};
    }
    for (r, dst) in tile.chunks_exact_mut(64).take(h).enumerate() {
        let xr = opsin.plane_row(0, y0 + r);
        let yr = opsin.plane_row(1, y0 + r);
        let br = opsin.plane_row(2, y0 + r);
        for group in 0..groups {
            let x = group * 16;
            accumulate_blue_x4!(sum0, x, xr, yr, br, dst);
            accumulate_blue_x4!(sum1, x + 4, xr, yr, br, dst);
            accumulate_blue_x4!(sum2, x + 8, xr, yr, br, dst);
            accumulate_blue_x4!(sum3, x + 12, xr, yr, br, dst);
        }
        if full_tail > 0 {
            accumulate_blue_x4!(sum0, vector_tail, xr, yr, br, dst);
        }
        if full_tail > 1 {
            accumulate_blue_x4!(sum1, vector_tail + 4, xr, yr, br, dst);
        }
        if full_tail > 2 {
            accumulate_blue_x4!(sum2, vector_tail + 8, xr, yr, br, dst);
        }
        let xr = &xr[x0 + tail..x0 + w];
        let yr = &yr[x0 + tail..x0 + w];
        let br = &br[x0 + tail..x0 + w];
        for (((d, &x), &y), &b) in dst[tail..w].iter_mut().zip(xr).zip(yr).zip(br) {
            let by = b - y;
            scalar_sum += ((by - x.abs() - BLUE_OFFSET).max(0.0) / BLUE_FULL).min(1.0);
            *d = by * Y_TO_LUMA8;
        }
    }
    let sum = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
    (horizontal_sum_x4(sum) + scalar_sum) / (w * h) as f32
}

#[inline]
fn horizontal_sum_x4(value: v128) -> f32 {
    f32x4_extract_lane::<0>(value)
        + f32x4_extract_lane::<1>(value)
        + f32x4_extract_lane::<2>(value)
        + f32x4_extract_lane::<3>(value)
}

#[inline]
#[target_feature(enable = "simd128")]
fn add_x4(sum: v128, row: &[f32], x: usize) -> v128 {
    f32x4_add(sum, unsafe { v128_load(row.as_ptr().add(x).cast()) })
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_tail(row: &[f32], x: usize, n: usize) -> v128 {
    let value = f32x4_replace_lane::<0>(f32x4_splat(0.0), row[x]);
    let value = if n > 1 {
        f32x4_replace_lane::<1>(value, row[x + 1])
    } else {
        value
    };
    if n > 2 {
        f32x4_replace_lane::<2>(value, row[x + 2])
    } else {
        value
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn sum_rows(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);
    let groups = w / 16;
    let full_tail = w % 16 / 4;
    let remainder = w % 4;
    for row in buf.chunks_exact(stride).take(h) {
        for group in 0..groups {
            let x = group * 16;
            sum0 = add_x4(sum0, row, x);
            sum1 = add_x4(sum1, row, x + 4);
            sum2 = add_x4(sum2, row, x + 8);
            sum3 = add_x4(sum3, row, x + 12);
        }
        let mut x = groups * 16;
        if full_tail > 0 {
            sum0 = add_x4(sum0, row, x);
            x += 4;
        }
        if full_tail > 1 {
            sum1 = add_x4(sum1, row, x);
            x += 4;
        }
        if full_tail > 2 {
            sum2 = add_x4(sum2, row, x);
            x += 4;
        }
        if remainder != 0 {
            let tail = load_tail(row, x, remainder);
            match full_tail {
                0 => sum0 = f32x4_add(sum0, tail),
                1 => sum1 = f32x4_add(sum1, tail),
                2 => sum2 = f32x4_add(sum2, tail),
                _ => sum3 = f32x4_add(sum3, tail),
            }
        }
    }
    let sum = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
    horizontal_sum_x4(sum)
}

#[inline]
#[target_feature(enable = "simd128")]
fn laplacian_abs_x4(
    sum: v128,
    top: &[f32],
    middle: &[f32],
    bottom: &[f32],
    x: usize,
    four: v128,
) -> v128 {
    let up = unsafe { v128_load(top.as_ptr().add(x).cast()) };
    let down = unsafe { v128_load(bottom.as_ptr().add(x).cast()) };
    let left = unsafe { v128_load(middle.as_ptr().add(x - 1).cast()) };
    let center = unsafe { v128_load(middle.as_ptr().add(x).cast()) };
    let right = unsafe { v128_load(middle.as_ptr().add(x + 1).cast()) };
    let neighbors = f32x4_add(f32x4_add(up, down), f32x4_add(left, right));
    f32x4_add(
        sum,
        f32x4_abs(f32x4_sub(f32x4_mul(center, four), neighbors)),
    )
}

#[inline]
#[target_feature(enable = "simd128")]
fn laplacian_abs_sum(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);
    let four = f32x4_splat(4.0);
    let interior = w - 2;
    let groups = interior / 16;
    let full_tail = interior % 16 / 4;
    let remainder = interior % 4;
    for y in 1..h - 1 {
        let top = &buf[(y - 1) * stride..];
        let middle = &buf[y * stride..];
        let bottom = &buf[(y + 1) * stride..];
        for group in 0..groups {
            let x = 1 + group * 16;
            sum0 = laplacian_abs_x4(sum0, top, middle, bottom, x, four);
            sum1 = laplacian_abs_x4(sum1, top, middle, bottom, x + 4, four);
            sum2 = laplacian_abs_x4(sum2, top, middle, bottom, x + 8, four);
            sum3 = laplacian_abs_x4(sum3, top, middle, bottom, x + 12, four);
        }
        let mut x = 1 + groups * 16;
        if full_tail > 0 {
            sum0 = laplacian_abs_x4(sum0, top, middle, bottom, x, four);
            x += 4;
        }
        if full_tail > 1 {
            sum1 = laplacian_abs_x4(sum1, top, middle, bottom, x, four);
            x += 4;
        }
        if full_tail > 2 {
            sum2 = laplacian_abs_x4(sum2, top, middle, bottom, x, four);
            x += 4;
        }
        if remainder != 0 {
            let neighbors = f32x4_add(
                load_tail(top, x, remainder),
                load_tail(bottom, x, remainder),
            );
            let neighbors = f32x4_add(neighbors, load_tail(middle, x - 1, remainder));
            let neighbors = f32x4_add(neighbors, load_tail(middle, x + 1, remainder));
            let center = load_tail(middle, x, remainder);
            let lap = f32x4_abs(f32x4_sub(f32x4_mul(center, four), neighbors));
            match full_tail {
                0 => sum0 = f32x4_add(sum0, lap),
                1 => sum1 = f32x4_add(sum1, lap),
                2 => sum2 = f32x4_add(sum2, lap),
                _ => sum3 = f32x4_add(sum3, lap),
            }
        }
    }
    let sum = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
    horizontal_sum_x4(sum)
}

#[inline]
#[target_feature(enable = "simd128")]
fn downsample_x4(top: &[f32; 8], bottom: &[f32; 8], dst: &mut [f32; 4]) {
    let top0 = unsafe { v128_load(top.as_ptr().cast()) };
    let top1 = unsafe { v128_load(top.as_ptr().add(4).cast()) };
    let bottom0 = unsafe { v128_load(bottom.as_ptr().cast()) };
    let bottom1 = unsafe { v128_load(bottom.as_ptr().add(4).cast()) };
    let top_pairs = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(top0, top1),
        i32x4_shuffle::<1, 3, 5, 7>(top0, top1),
    );
    let bottom_pairs = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(bottom0, bottom1),
        i32x4_shuffle::<1, 3, 5, 7>(bottom0, bottom1),
    );
    let value = f32x4_mul(f32x4_add(top_pairs, bottom_pairs), f32x4_splat(0.25));
    unsafe { v128_store(dst.as_mut_ptr().cast(), value) };
}

#[inline]
#[target_feature(enable = "simd128")]
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

#[target_feature(enable = "simd128")]
pub(crate) fn dark_structure_stats_wasm(buf: &[[f32; 64]], h: usize, w: usize) -> (f32, f32) {
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
