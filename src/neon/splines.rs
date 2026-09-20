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
    ALONG_PENALTY, MAX_SUBPIXEL_OFFSET, Point, RidgeRow, Sample, Segments, ridge_row_scalar,
};
use std::arch::aarch64::*;

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
pub(crate) fn continuous_idct_neon(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
    let mut result = vdupq_n_f32(0.0);
    let indices = [0.0, 1.0, 2.0, 3.0];
    let mut indices = unsafe { vld1q_f32(indices.as_ptr()) };
    for rows in dct.as_chunks::<4>().0 {
        let args = vmulq_n_f32(vmulq_n_f32(indices, std::f32::consts::PI / 32.0), t + 0.5);
        let cos = fast_cos_neon(args);
        let cosines = [
            vdupq_laneq_f32::<0>(cos),
            vdupq_laneq_f32::<1>(cos),
            vdupq_laneq_f32::<2>(cos),
            vdupq_laneq_f32::<3>(cos),
        ];
        // Accumulate frequencies in order, retaining scalar rounding in each channel.
        for (row, cos) in rows.iter().zip(cosines) {
            let values = unsafe { vld1q_f32(row.as_ptr()) };
            result = vaddq_f32(result, vmulq_f32(values, cos));
        }
        indices = vaddq_f32(indices, vdupq_n_f32(4.0));
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
fn erf(x: float32x4_t) -> float32x4_t {
    let a = vabsq_f32(x);
    // Deliberately separate multiplication and addition, including the
    // reciprocal, to match the scalar renderer's rounding exactly.
    let d1 = vaddq_f32(vmulq_n_f32(a, 7.77394369e-02), vdupq_n_f32(2.05260015e-04));
    let d2 = vaddq_f32(vmulq_f32(d1, a), vdupq_n_f32(2.32120216e-01));
    let d3 = vaddq_f32(vmulq_f32(d2, a), vdupq_n_f32(2.77820801e-01));
    let d4 = vaddq_f32(vmulq_f32(d3, a), vdupq_n_f32(1.0));
    let d5 = vmulq_f32(d4, d4);
    let inv = vdivq_f32(vdupq_n_f32(1.0), d5);
    let r = vsubq_f32(vdupq_n_f32(1.0), vmulq_f32(inv, inv));
    vbslq_f32(vcleq_f32(x, vdupq_n_f32(0.0)), vnegq_f32(r), r)
}

#[inline]
#[target_feature(enable = "neon")]
fn intensity(sample: &Sample, dy: f32, x: usize, amp: f32) -> float32x4_t {
    // Image dimensions are at most 30 bits, so padded coordinates fit in i32.
    // Add in integer lanes before conversion to preserve rounding above 2^24.
    debug_assert!(x <= i32::MAX as usize - 3);
    let offsets = unsafe { vld1q_s32([0, 1, 2, 3].as_ptr()) };
    let indices = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(x as i32), offsets));
    let dx = vsubq_f32(indices, vdupq_n_f32(sample.position.x));
    let dist = vsqrtq_f32(vaddq_f32(vmulq_f32(dx, dx), vdupq_n_f32(dy * dy)));
    let half = vmulq_n_f32(dist, 0.5);
    let a = vmulq_n_f32(vaddq_f32(half, vdupq_n_f32(0.353_553_39)), sample.inv_sigma);
    let b = vmulq_n_f32(vsubq_f32(half, vdupq_n_f32(0.353_553_39)), sample.inv_sigma);
    let e = vsubq_f32(erf(a), erf(b));
    vmulq_f32(vmulq_n_f32(e, amp), e)
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
    let n = rx.len() / 4 * 4;
    let mut x = 0;
    while x < n {
        let li = intensity(sample, dy, x0 + x, amp);
        // SAFETY: n is rounded down to full vectors; all three rows have
        // equal lengths.
        unsafe {
            let add = |row: &mut [f32], color| {
                let ptr = row.as_mut_ptr().add(x);
                vst1q_f32(ptr, vaddq_f32(vld1q_f32(ptr), vmulq_n_f32(li, color)));
            };
            add(rx, sample.color[0]);
            add(ry, sample.color[1]);
            add(rb, sample.color[2]);
        }
        x += 4;
    }
    if rx.len() - n >= 2 {
        let mut weights = [0.0; 4];
        // SAFETY: the incomplete vector writes only to a local array; no
        // image load or store extends past the end of a row.
        unsafe { vst1q_f32(weights.as_mut_ptr(), intensity(sample, dy, x0 + n, amp)) };
        for x in n..rx.len() {
            let li = weights[x - n];
            rx[x] += sample.color[0] * li;
            ry[x] += sample.color[1] * li;
            rb[x] += sample.color[2] * li;
        }
    } else {
        sample.scalar_row(dy, x0 + n, [&mut rx[n..], &mut ry[n..], &mut rb[n..]], amp);
    }
}
