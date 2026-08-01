use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn sum_rows(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sums = [vdupq_n_f32(0.0); 4];
    for row in buf.chunks_exact(stride).take(h) {
        let (chunks, tail) = row[..w].as_chunks::<4>();
        for (i, chunk) in chunks.iter().enumerate() {
            sums[i & 3] = vaddq_f32(sums[i & 3], unsafe { vld1q_f32(chunk.as_ptr()) });
        }
        if !tail.is_empty() {
            let mut padded = [0.0; 4];
            padded[..tail.len()].copy_from_slice(tail);
            let lane = chunks.len() & 3;
            sums[lane] = vaddq_f32(sums[lane], unsafe { vld1q_f32(padded.as_ptr()) });
        }
    }
    let sum = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
    vaddvq_f32(sum)
}

#[inline]
#[target_feature(enable = "neon")]
fn laplacian_abs_sum(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sums = [vdupq_n_f32(0.0); 4];
    let interior = w - 2;
    for y in 1..h - 1 {
        let top = &buf[(y - 1) * stride..];
        let middle = &buf[y * stride..];
        let bottom = &buf[(y + 1) * stride..];
        let full = interior / 4;
        for chunk in 0..full {
            let x = 1 + chunk * 4;
            let up = unsafe { vld1q_f32(top.as_ptr().add(x)) };
            let down = unsafe { vld1q_f32(bottom.as_ptr().add(x)) };
            let left = unsafe { vld1q_f32(middle.as_ptr().add(x - 1)) };
            let center = unsafe { vld1q_f32(middle.as_ptr().add(x)) };
            let right = unsafe { vld1q_f32(middle.as_ptr().add(x + 1)) };
            let neighbors = vaddq_f32(vaddq_f32(up, down), vaddq_f32(left, right));
            let lane = chunk & 3;
            sums[lane] = vaddq_f32(
                sums[lane],
                vabsq_f32(vsubq_f32(vmulq_n_f32(center, 4.0), neighbors)),
            );
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
            let up = unsafe { vld1q_f32(up.as_ptr()) };
            let down = unsafe { vld1q_f32(down.as_ptr()) };
            let left = unsafe { vld1q_f32(left.as_ptr()) };
            let center = unsafe { vld1q_f32(center.as_ptr()) };
            let right = unsafe { vld1q_f32(right.as_ptr()) };
            let neighbors = vaddq_f32(vaddq_f32(up, down), vaddq_f32(left, right));
            let lane = full & 3;
            sums[lane] = vaddq_f32(
                sums[lane],
                vabsq_f32(vsubq_f32(vmulq_n_f32(center, 4.0), neighbors)),
            );
        }
    }
    let sum = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
    vaddvq_f32(sum)
}

#[inline]
#[target_feature(enable = "neon")]
fn downsample_x4(top: &[f32; 8], bottom: &[f32; 8], dst: &mut [f32; 4]) {
    let top0 = unsafe { vld1q_f32(top.as_ptr()) };
    let top1 = unsafe { vld1q_f32(top[4..].as_ptr()) };
    let bottom0 = unsafe { vld1q_f32(bottom.as_ptr()) };
    let bottom1 = unsafe { vld1q_f32(bottom[4..].as_ptr()) };
    let pairs = vaddq_f32(vpaddq_f32(top0, top1), vpaddq_f32(bottom0, bottom1));
    unsafe { vst1q_f32(dst.as_mut_ptr(), vmulq_n_f32(pairs, 0.25)) };
}

#[inline]
#[target_feature(enable = "neon")]
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
/// AArch64 NEON must be available.
#[target_feature(enable = "neon")]
pub(crate) fn dark_structure_stats_neon(buf: &[[f32; 64]], h: usize, w: usize) -> (f32, f32) {
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
