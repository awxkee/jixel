/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

//! Spline detection, transforms, rendering and geometry kernels.

use crate::splines::{
    ALONG_PENALTY, BlobWeightRow, FilterBounds, MAX_SUBPIXEL_OFFSET, Point, RidgeRow, Sample,
    Segments, ridge_row_scalar, vertical_row_scalar_tail,
};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
unsafe fn vertical_block<const VECTORS: usize>(
    src: &[f32],
    w: usize,
    bounds: FilterBounds,
    kernel: &[f32],
    x: usize,
    dst: &mut [[f32; 4]; VECTORS],
) {
    let mut sums = [vdupq_n_f32(0.0); VECTORS];
    for (tap, &weight) in kernel.iter().enumerate() {
        let start = bounds.index(tap) * w + x;
        // SAFETY: the caller checked every source row contains this block.
        let pixels = unsafe { src.get_unchecked(start..start + 4 * VECTORS) };
        let (pixels, _) = pixels.as_chunks::<4>();
        for (sum, pixels) in sums.iter_mut().zip(pixels) {
            // SAFETY: each chunk contains four readable pixels.
            let pixels = unsafe { vld1q_f32(pixels.as_ptr()) };
            *sum = vaddq_f32(*sum, vmulq_n_f32(pixels, weight));
        }
    }
    for (block, sum) in dst.iter_mut().zip(sums) {
        // SAFETY: each chunk contains four writable pixels.
        unsafe { vst1q_f32(block.as_mut_ptr(), sum) };
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn vertical_row_neon(
    src: &[f32],
    w: usize,
    bounds: FilterBounds,
    kernel: &[f32],
    dst: &mut [f32],
) {
    if dst.is_empty() {
        return;
    }
    assert!(dst.len() <= w && bounds.index(kernel.len().saturating_sub(1)) < src.len() / w);
    let (blocks, tail) = dst.as_chunks_mut::<4>();
    let (groups, blocks) = blocks.as_chunks_mut::<4>();
    let mut x = 0;
    for group in groups {
        // SAFETY: validated above, and each block contains 16 pixels.
        unsafe { vertical_block(src, w, bounds, kernel, x, group) };
        x += 16;
    }
    for block in blocks {
        // SAFETY: validated above, and each block contains four pixels.
        unsafe { vertical_block(src, w, bounds, kernel, x, std::array::from_mut(block)) };
        x += 4;
    }
    if !tail.is_empty() {
        vertical_row_scalar_tail(src, w, bounds, kernel, x, tail);
    }
}

/// Evaluate four independent Hessians without changing scalar rounding or
/// scale tie-breaking. Rejected lanes retain all of their previous attributes.
#[target_feature(enable = "neon")]
pub(crate) fn spline_ridge_row_neon(s: f32, derivatives: [&[f32]; 5], output: RidgeRow<'_>) {
    let n = output.validate(derivatives);
    let end = n / 4 * 4;
    let [hxx, hyy, hxy, gx, gy] = derivatives;
    let s2 = s * s;
    let zero = vdupq_n_f32(0.0);
    let eps = vdupq_n_f32(1e-12);
    for i in (0..end).step_by(4) {
        // SAFETY: validation checks every input/output length; i..i+4 is in
        // bounds. All output slices are disjoint mutable borrows.
        unsafe {
            let xx = vld1q_f32(hxx.as_ptr().add(i));
            let yy = vld1q_f32(hyy.as_ptr().add(i));
            let xy = vld1q_f32(hxy.as_ptr().add(i));
            let tr = vmulq_n_f32(vaddq_f32(xx, yy), 0.5);
            let half_diff = vmulq_n_f32(vsubq_f32(xx, yy), 0.5);
            let df = vsqrtq_f32(vaddq_f32(
                vmulq_f32(half_diff, half_diff),
                vmulq_f32(xy, xy),
            ));
            let la = vaddq_f32(tr, df);
            let lb = vsubq_f32(tr, df);
            let a_larger = vcgtq_f32(vabsq_f32(la), vabsq_f32(lb));
            let big = vbslq_f32(a_larger, la, lb);
            let small = vbslq_f32(a_larger, lb, la);
            let abs_big = vabsq_f32(big);
            let strength = vmulq_n_f32(
                vmaxnmq_f32(
                    vsubq_f32(abs_big, vmulq_n_f32(vabsq_f32(small), ALONG_PENALTY)),
                    zero,
                ),
                s2,
            );
            let old_strength = vld1q_f32(output.strength.as_ptr().add(i));
            let mut accept = vmvnq_u32(vcleq_f32(strength, old_strength));
            if vmaxvq_u32(accept) == 0 {
                continue;
            }
            let gx = vld1q_f32(gx.as_ptr().add(i));
            let gy = vld1q_f32(gy.as_ptr().add(i));
            let gradient = vsqrtq_f32(vaddq_f32(vmulq_f32(gx, gx), vmulq_f32(gy, gy)));
            accept = vandq_u32(
                accept,
                vmvnq_u32(vcgtq_f32(
                    vmulq_n_f32(gradient, s),
                    vmulq_n_f32(abs_big, s2),
                )),
            );
            if vmaxvq_u32(accept) == 0 {
                continue;
            }
            let vy = vsubq_f32(big, xx);
            let norm = vsqrtq_f32(vaddq_f32(vmulq_f32(xy, xy), vmulq_f32(vy, vy)));
            let degenerate = vcltq_f32(norm, eps);
            let vx = vbslq_f32(degenerate, vdupq_n_f32(1.0), vdivq_f32(xy, norm));
            let vy = vbslq_f32(degenerate, zero, vdivq_f32(vy, norm));
            let t = vdivq_f32(
                vnegq_f32(vaddq_f32(vmulq_f32(gx, vx), vmulq_f32(gy, vy))),
                big,
            );
            // Comparisons preserve signed zero and NaNs like scalar clamp.
            let lower = vdupq_n_f32(-MAX_SUBPIXEL_OFFSET);
            let upper = vdupq_n_f32(MAX_SUBPIXEL_OFFSET);
            let t = vbslq_f32(vcltq_f32(t, lower), lower, t);
            let t = vbslq_f32(vcgtq_f32(t, upper), upper, t);
            let t = vbslq_f32(vcltq_f32(abs_big, eps), zero, t);
            let store = |row: &mut [f32], value| {
                let ptr = row.as_mut_ptr().add(i);
                vst1q_f32(ptr, vbslq_f32(accept, value, vld1q_f32(ptr)));
            };
            store(output.strength, strength);
            store(output.nx, vx);
            store(output.ny, vy);
            store(output.scale, vdupq_n_f32(s));
            store(output.offset, t);
            // Pack the four byte attributes and their masks; a 32-bit access
            // touches exactly these four pixels, including at the row end.
            let polarity = vbslq_s32(vcgtq_f32(big, zero), vdupq_n_s32(-1), vdupq_n_s32(1));
            let polarity = vmovn_s32(polarity);
            let polarity = vmovn_s16(vcombine_s16(polarity, polarity));
            let mask = vmovn_u32(accept);
            let mask = vmovn_u16(vcombine_u16(mask, mask));
            let polarity = vget_lane_u32::<0>(vreinterpret_u32_s8(polarity));
            let mask = vget_lane_u32::<0>(vreinterpret_u32_u8(mask));
            let ptr = output.polarity.as_mut_ptr().add(i).cast::<u32>();
            ptr.write_unaligned((ptr.read_unaligned() & !mask) | (polarity & mask));
        }
    }
    ridge_row_scalar(
        s,
        derivatives.map(|row| &row[end..]),
        RidgeRow {
            strength: &mut output.strength[end..],
            nx: &mut output.nx[end..],
            ny: &mut output.ny[end..],
            scale: &mut output.scale[end..],
            polarity: &mut output.polarity[end..],
            offset: &mut output.offset[end..],
        },
    );
}

#[target_feature(enable = "neon")]
fn fast_cos_neon(x: float32x4_t) -> float32x4_t {
    use std::f32::consts::{PI, SQRT_2};
    let pi2 = vdupq_n_f32(PI * 2.0);
    let periods = vrndmq_f32(vmulq_n_f32(x, 0.5 / PI));
    let xmod = vsubq_f32(x, vmulq_f32(periods, pi2));
    let x_pi = vminq_f32(xmod, vsubq_f32(pi2, xmod));
    let above = vcgeq_f32(x_pi, vdupq_n_f32(PI / 2.0));
    let xh = vbslq_f32(above, vsubq_f32(vdupq_n_f32(PI), x_pi), x_pi);
    let xs = vmulq_n_f32(xh, 0.25);
    let x2 = vmulq_f32(xs, xs);
    let x4 = vmulq_f32(x2, x2);
    // Separate multiply/add operations preserve the decoder's approximation.
    let pre = vaddq_f32(
        vmulq_n_f32(x4, 0.06960438),
        vaddq_f32(vmulq_n_f32(x2, -0.84087373), vdupq_n_f32(1.68179268)),
    );
    let s1 = vsubq_f32(vmulq_f32(pre, pre), vdupq_n_f32(SQRT_2));
    let s2 = vsubq_f32(vmulq_f32(s1, s1), vdupq_n_f32(1.0));
    vbslq_f32(above, vnegq_f32(s2), s2)
}

#[target_feature(enable = "neon")]
pub(crate) fn continuous_idct_neon(dct: &[[f32; 4]], t: f32) -> [f32; 4] {
    if dct.len() == 32 {
        continuous_idct_impl(&dct[..32], t)
    } else {
        continuous_idct_impl(dct, t)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn continuous_idct_impl(dct: &[[f32; 4]], t: f32) -> [f32; 4] {
    let mut result = vdupq_n_f32(0.0);
    let indices = [0.0, 1.0, 2.0, 3.0];
    let mut indices = unsafe { vld1q_f32(indices.as_ptr()) };
    let cosine = |indices| {
        let args = vmulq_n_f32(vmulq_n_f32(indices, std::f32::consts::PI / 32.0), t + 0.5);
        let cos = fast_cos_neon(args);
        [
            vdupq_laneq_f32::<0>(cos),
            vdupq_laneq_f32::<1>(cos),
            vdupq_laneq_f32::<2>(cos),
            vdupq_laneq_f32::<3>(cos),
        ]
    };
    // Accumulate frequencies in order, retaining scalar rounding in each channel.
    let mut accumulate = |row: &[f32; 4], cos| {
        let values = unsafe { vld1q_f32(row.as_ptr()) };
        result = vaddq_f32(result, vmulq_f32(values, cos));
    };
    let (blocks, tail) = dct.as_chunks::<4>();
    for rows in blocks {
        rows.iter()
            .zip(cosine(indices))
            .for_each(|(row, cos)| accumulate(row, cos));
        indices = vaddq_f32(indices, vdupq_n_f32(4.0));
    }
    if !tail.is_empty() {
        tail.iter()
            .zip(cosine(indices))
            .for_each(|(row, cos)| accumulate(row, cos));
    }
    let mut out = [0.0; 4];
    unsafe { vst1q_f32(out.as_mut_ptr(), result) };
    out
}

#[target_feature(enable = "neon")]
pub(crate) fn spline_distance_neon(p: Point<f32>, blocks: &[Segments]) -> f32 {
    let (px, py) = (vdupq_n_f32(p.x), vdupq_n_f32(p.y));
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let mut best = vdupq_n_f32(f32::INFINITY);
    for block in blocks {
        // SAFETY: each field is a complete, initialized four-element array.
        let (x, y, dx, dy, len2) = unsafe {
            (
                vld1q_f32(block.x.as_ptr()),
                vld1q_f32(block.y.as_ptr()),
                vld1q_f32(block.dx.as_ptr()),
                vld1q_f32(block.dy.as_ptr()),
                vld1q_f32(block.len2.as_ptr()),
            )
        };
        let nonzero = vcgtq_f32(len2, vdupq_n_f32(1e-12));
        let numerator = vaddq_f32(
            vmulq_f32(vsubq_f32(px, x), dx),
            vmulq_f32(vsubq_f32(py, y), dy),
        );
        let t = vdivq_f32(numerator, vbslq_f32(nonzero, len2, one));
        let t = vbslq_f32(nonzero, vminq_f32(vmaxq_f32(t, zero), one), zero);
        let ex = vsubq_f32(px, vaddq_f32(x, vmulq_f32(t, dx)));
        let ey = vsubq_f32(py, vaddq_f32(y, vmulq_f32(t, dy)));
        best = vminnmq_f32(best, vaddq_f32(vmulq_f32(ex, ex), vmulq_f32(ey, ey)));
    }
    vminnmvq_f32(best)
}

#[inline]
#[target_feature(enable = "neon")]
fn horner_step<const FUSED: bool>(a: float32x4_t, b: float32x4_t, c: f32) -> float32x4_t {
    if FUSED {
        vfmaq_f32(vdupq_n_f32(c), a, b)
    } else {
        vaddq_f32(vmulq_f32(a, b), vdupq_n_f32(c))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn erf<const FUSED: bool>(x: float32x4_t) -> float32x4_t {
    let a = vabsq_f32(x);
    let d1 = horner_step::<FUSED>(a, vdupq_n_f32(7.77394369e-02), 2.05260015e-04);
    let d2 = horner_step::<FUSED>(d1, a, 2.32120216e-01);
    let d3 = horner_step::<FUSED>(d2, a, 2.77820801e-01);
    let d4 = horner_step::<FUSED>(d3, a, 1.0);
    let d5 = vmulq_f32(d4, d4);
    let inv = vdivq_f32(vdupq_n_f32(1.0), d5);
    let r = vsubq_f32(vdupq_n_f32(1.0), vmulq_f32(inv, inv));
    vbslq_f32(vcleq_f32(x, vdupq_n_f32(0.0)), vnegq_f32(r), r)
}

#[inline]
#[target_feature(enable = "neon")]
fn dx_lanes(x: i32, center: f32) -> float32x4_t {
    // Add in integer lanes before conversion to preserve rounding above 2^24.
    debug_assert!(x <= i32::MAX - 3);
    // SAFETY: the constant array contains four readable lanes.
    let offsets = unsafe { vld1q_s32([0, 1, 2, 3].as_ptr()) };
    let indices = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(x), offsets));
    vsubq_f32(indices, vdupq_n_f32(center))
}

#[inline]
#[target_feature(enable = "neon")]
fn blob_weights<const FUSED: bool>(
    dx: float32x4_t,
    dy: f32,
    inv_sigma: f32,
    amp: f32,
) -> float32x4_t {
    let dist = vsqrtq_f32(vaddq_f32(vmulq_f32(dx, dx), vdupq_n_f32(dy * dy)));
    let half = vmulq_n_f32(dist, 0.5);
    let a = vmulq_n_f32(vaddq_f32(half, vdupq_n_f32(0.353_553_39)), inv_sigma);
    let b = vmulq_n_f32(vsubq_f32(half, vdupq_n_f32(0.353_553_39)), inv_sigma);
    let e = vsubq_f32(erf::<FUSED>(a), erf::<FUSED>(b));
    vmulq_f32(vmulq_n_f32(e, amp), e)
}

#[inline]
#[target_feature(enable = "neon")]
fn intensity(sample: &Sample, dy: f32, x: usize, amp: f32) -> float32x4_t {
    // Image dimensions are at most 30 bits, so padded coordinates fit in i32.
    debug_assert!(x <= i32::MAX as usize - 3);
    let dx = dx_lanes(x as i32, sample.position.x);
    // Match the scalar renderer's polynomial rounding.
    blob_weights::<false>(dx, dy, sample.inv_sigma, amp)
}

#[inline]
#[target_feature(enable = "neon")]
fn row_weights(row: BlobWeightRow, x: i32) -> float32x4_t {
    let dx = vmulq_n_f32(dx_lanes(x, row.center), row.step);
    blob_weights::<true>(dx, row.dy, row.inv_sigma, row.amp)
}

#[target_feature(enable = "neon")]
pub(crate) fn blob_weight_row_neon(row: BlobWeightRow, x0: i32, weights: &mut [f32]) {
    let (blocks, tail) = weights.as_chunks_mut::<4>();
    for (i, block) in blocks.iter_mut().enumerate() {
        // SAFETY: each block has room for all four lanes.
        unsafe { vst1q_f32(block.as_mut_ptr(), row_weights(row, x0 + 4 * i as i32)) };
    }
    if !tail.is_empty() {
        let mut padded = [0.0; 4];
        // SAFETY: write the full vector to the local array and copy only the tail.
        unsafe {
            vst1q_f32(
                padded.as_mut_ptr(),
                row_weights(row, x0 + 4 * blocks.len() as i32),
            )
        };
        tail.copy_from_slice(&padded[..tail.len()]);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dot_contrast_neon(
    src: &[f32],
    stride: usize,
    dst: &mut [f32],
    side: usize,
    bg: f32,
    slope: [f32; 2],
) {
    let radius = (side / 2) as i32;
    let [sx, sy] = slope;
    let background = vdupq_n_f32(bg);
    let horizontal = [
        vmulq_n_f32(dx_lanes(-radius, 0.0), sx),
        vmulq_n_f32(dx_lanes(4 - radius, 0.0), sx),
    ];
    let rows = crate::splines::dots::contrast_patch_rows(src, stride, dst, side);
    for (y, (src, dst)) in rows.enumerate() {
        let row_slope = sy * (y as i32 - radius) as f32;
        let vertical = vdupq_n_f32(row_slope);
        let (src, src_tail) = src.as_chunks::<4>();
        let (dst, dst_tail) = dst.as_chunks_mut::<4>();
        for ((src, dst), horizontal) in src.iter().zip(dst).zip(horizontal) {
            // SAFETY: each array contains four readable or writable pixels.
            unsafe {
                let pixels = vld1q_f32(src.as_ptr());
                let contrast = vsubq_f32(
                    vsubq_f32(vsubq_f32(pixels, background), horizontal),
                    vertical,
                );
                vst1q_f32(dst.as_mut_ptr(), contrast);
            }
        }
        let x0 = 4 * src.len() as i32 - radius;
        for ((dst, &pixel), dx) in dst_tail.iter_mut().zip(src_tail).zip(x0..) {
            *dst = pixel - bg - sx * dx as f32 - row_slope;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn spline_render_row_neon(
    sample: &Sample,
    dy: f32,
    x0: usize,
    rows: [&mut [f32]; 3],
    amp: f32,
) {
    let [rx, ry, rb] = rows;
    assert_eq!(rx.len(), ry.len());
    assert_eq!(rx.len(), rb.len());
    let [(rx, tx), (ry, ty), (rb, tb)] = [rx, ry, rb].map(|row| row.as_chunks_mut::<4>());
    let n = rx.len() * 4;
    for (i, ((rx, ry), rb)) in rx.iter_mut().zip(ry).zip(rb).enumerate() {
        let li = intensity(sample, dy, x0 + i * 4, amp);
        for (block, color) in [rx, ry, rb].into_iter().zip(sample.color) {
            // SAFETY: each block contains four readable and writable pixels.
            unsafe {
                let ptr = block.as_mut_ptr();
                vst1q_f32(ptr, vaddq_f32(vld1q_f32(ptr), vmulq_n_f32(li, color)));
            }
        }
    }
    if tx.len() >= 2 {
        let mut weights = [0.0; 4];
        // SAFETY: the incomplete vector writes only to a local array; no
        // image load or store extends past the end of a row.
        unsafe { vst1q_f32(weights.as_mut_ptr(), intensity(sample, dy, x0 + n, amp)) };
        for (tail, color) in [tx, ty, tb].into_iter().zip(sample.color) {
            for (out, li) in tail.iter_mut().zip(weights) {
                *out += color * li;
            }
        }
    } else {
        sample.scalar_row(dy, x0 + n, [tx, ty, tb], amp);
    }
}
