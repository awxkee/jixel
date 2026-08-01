use std::arch::wasm32::*;

#[inline]
fn horizontal_sum_x4(value: v128) -> f32 {
    f32x4_extract_lane::<0>(value)
        + f32x4_extract_lane::<1>(value)
        + f32x4_extract_lane::<2>(value)
        + f32x4_extract_lane::<3>(value)
}

#[inline]
#[target_feature(enable = "simd128")]
fn sum_rows(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sums = [f32x4_splat(0.0); 4];
    for row in buf.chunks_exact(stride).take(h) {
        let (chunks, tail) = row[..w].as_chunks::<4>();
        for (i, chunk) in chunks.iter().enumerate() {
            sums[i & 3] = f32x4_add(sums[i & 3], unsafe { v128_load(chunk.as_ptr().cast()) });
        }
        if !tail.is_empty() {
            let mut padded = [0.0; 4];
            padded[..tail.len()].copy_from_slice(tail);
            let lane = chunks.len() & 3;
            sums[lane] = f32x4_add(sums[lane], unsafe { v128_load(padded.as_ptr().cast()) });
        }
    }
    let sum = f32x4_add(f32x4_add(sums[0], sums[1]), f32x4_add(sums[2], sums[3]));
    horizontal_sum_x4(sum)
}

#[inline]
#[target_feature(enable = "simd128")]
fn laplacian_abs_sum(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sums = [f32x4_splat(0.0); 4];
    let four = f32x4_splat(4.0);
    let interior = w - 2;
    for y in 1..h - 1 {
        let top = &buf[(y - 1) * stride..];
        let middle = &buf[y * stride..];
        let bottom = &buf[(y + 1) * stride..];
        let full = interior / 4;
        for chunk in 0..full {
            let x = 1 + chunk * 4;
            let up = unsafe { v128_load(top.as_ptr().add(x).cast()) };
            let down = unsafe { v128_load(bottom.as_ptr().add(x).cast()) };
            let left = unsafe { v128_load(middle.as_ptr().add(x - 1).cast()) };
            let center = unsafe { v128_load(middle.as_ptr().add(x).cast()) };
            let right = unsafe { v128_load(middle.as_ptr().add(x + 1).cast()) };
            let neighbors = f32x4_add(f32x4_add(up, down), f32x4_add(left, right));
            let lap = f32x4_sub(f32x4_mul(center, four), neighbors);
            let lane = chunk & 3;
            sums[lane] = f32x4_add(sums[lane], f32x4_abs(lap));
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
            let up = unsafe { v128_load(up.as_ptr().cast()) };
            let down = unsafe { v128_load(down.as_ptr().cast()) };
            let left = unsafe { v128_load(left.as_ptr().cast()) };
            let center = unsafe { v128_load(center.as_ptr().cast()) };
            let right = unsafe { v128_load(right.as_ptr().cast()) };
            let neighbors = f32x4_add(f32x4_add(up, down), f32x4_add(left, right));
            let lap = f32x4_sub(f32x4_mul(center, four), neighbors);
            let lane = full & 3;
            sums[lane] = f32x4_add(sums[lane], f32x4_abs(lap));
        }
    }
    let sum = f32x4_add(f32x4_add(sums[0], sums[1]), f32x4_add(sums[2], sums[3]));
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
