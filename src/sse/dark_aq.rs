use crate::sse::adaptive_quant::hsum;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn sum_rows(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sums = [_mm_setzero_ps(); 4];
    for row in buf.chunks_exact(stride).take(h) {
        let (chunks, tail) = row[..w].as_chunks::<4>();
        for (i, chunk) in chunks.iter().enumerate() {
            sums[i & 3] = _mm_add_ps(sums[i & 3], unsafe { _mm_loadu_ps(chunk.as_ptr()) });
        }
        if !tail.is_empty() {
            let mut padded = [0.0; 4];
            padded[..tail.len()].copy_from_slice(tail);
            let lane = chunks.len() & 3;
            sums[lane] = _mm_add_ps(sums[lane], unsafe { _mm_loadu_ps(padded.as_ptr()) });
        }
    }
    let sum = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
    hsum(sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn laplacian_abs_sum(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sums = [_mm_setzero_ps(); 4];
    let four = _mm_set1_ps(4.0);
    let sign = _mm_set1_ps(-0.0);
    let interior = w - 2;
    for y in 1..h - 1 {
        let top = &buf[(y - 1) * stride..];
        let middle = &buf[y * stride..];
        let bottom = &buf[(y + 1) * stride..];
        let full = interior / 4;
        for chunk in 0..full {
            let x = 1 + chunk * 4;
            let up = unsafe { _mm_loadu_ps(top.as_ptr().add(x)) };
            let down = unsafe { _mm_loadu_ps(bottom.as_ptr().add(x)) };
            let left = unsafe { _mm_loadu_ps(middle.as_ptr().add(x - 1)) };
            let center = unsafe { _mm_loadu_ps(middle.as_ptr().add(x)) };
            let right = unsafe { _mm_loadu_ps(middle.as_ptr().add(x + 1)) };
            let neighbors = _mm_add_ps(_mm_add_ps(up, down), _mm_add_ps(left, right));
            let lap = _mm_sub_ps(_mm_mul_ps(center, four), neighbors);
            let lane = chunk & 3;
            sums[lane] = _mm_add_ps(sums[lane], _mm_andnot_ps(sign, lap));
        }
        let remainder = interior % 4;
        if remainder != 0 {
            let x = 1 + full * 4;
            let mut up = [0.0; 4];
            let mut down = [0.0; 4];
            let mut left = [0.0; 4];
            let mut center = [0.0; 4];
            let mut right = [0.0; 4];
            up[..remainder].copy_from_slice(&top[x..x + remainder]);
            down[..remainder].copy_from_slice(&bottom[x..x + remainder]);
            left[..remainder].copy_from_slice(&middle[x - 1..x - 1 + remainder]);
            center[..remainder].copy_from_slice(&middle[x..x + remainder]);
            right[..remainder].copy_from_slice(&middle[x + 1..x + 1 + remainder]);
            let up = unsafe { _mm_loadu_ps(up.as_ptr()) };
            let down = unsafe { _mm_loadu_ps(down.as_ptr()) };
            let left = unsafe { _mm_loadu_ps(left.as_ptr()) };
            let center = unsafe { _mm_loadu_ps(center.as_ptr()) };
            let right = unsafe { _mm_loadu_ps(right.as_ptr()) };
            let neighbors = _mm_add_ps(_mm_add_ps(up, down), _mm_add_ps(left, right));
            let lap = _mm_sub_ps(_mm_mul_ps(center, four), neighbors);
            let lane = full & 3;
            sums[lane] = _mm_add_ps(sums[lane], _mm_andnot_ps(sign, lap));
        }
    }
    let sum = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
    hsum(sum)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn downsample_x4(top: &[f32; 8], bottom: &[f32; 8], dst: &mut [f32; 4]) {
    let top0 = unsafe { _mm_loadu_ps(top.as_ptr()) };
    let top1 = unsafe { _mm_loadu_ps(top.as_ptr().add(4)) };
    let bottom0 = unsafe { _mm_loadu_ps(bottom.as_ptr()) };
    let bottom1 = unsafe { _mm_loadu_ps(bottom.as_ptr().add(4)) };
    let pairs = _mm_add_ps(_mm_hadd_ps(top0, top1), _mm_hadd_ps(bottom0, bottom1));
    unsafe { _mm_storeu_ps(dst.as_mut_ptr(), _mm_mul_ps(pairs, _mm_set1_ps(0.25))) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
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
/// The caller must ensure SSE4.1 is available.
#[target_feature(enable = "sse4.1")]
pub(crate) fn dark_structure_stats_sse41(buf: &[[f32; 64]], h: usize, w: usize) -> (f32, f32) {
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
