/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! NEON interpolation for the interior points of a Catmull–Rom span.

use super::{Point, SPAN_STEPS};
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(super) fn sample_span_neon(
    controls: &[Point<f32>; 4],
    distances: [f32; 3],
    knots: [f32; 4],
    out: &mut Vec<Point<f32>>,
) {
    let [d0, d1, d2] = distances;
    let sums = [vdupq_n_f32(d0 + d1), vdupq_n_f32(d1 + d2)];
    let distances = distances.map(|d| vdupq_n_f32(d));
    let knots = knots.map(|t| vdupq_n_f32(t));
    let q = controls.map(|p| [vdupq_n_f32(p.x), vdupq_n_f32(p.y)]);
    let mut points = [Point::default(); SPAN_STEPS.len()];

    let lerp = |a: [float32x4_t; 2], b: [float32x4_t; 2], f| {
        [
            vaddq_f32(a[0], vmulq_f32(f, vsubq_f32(b[0], a[0]))),
            vaddq_f32(a[1], vmulq_f32(f, vsubq_f32(b[1], a[1]))),
        ]
    };
    for (points, steps) in points
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(SPAN_STEPS.as_chunks::<4>().0)
    {
        // SAFETY: the step array holds four elements.
        let step = unsafe { vld1q_f32(steps.as_ptr()) };
        let tt = vaddq_f32(distances[0], vmulq_f32(step, distances[1]));
        let dt0 = vsubq_f32(tt, knots[0]);
        let dt1 = vsubq_f32(tt, knots[1]);
        let dt2 = vsubq_f32(tt, knots[2]);
        // Keep the scalar division and interpolation rounding in each lane.
        let f1 = vdivq_f32(dt1, distances[1]);
        let a0 = lerp(q[0], q[1], vdivq_f32(dt0, distances[0]));
        let a1 = lerp(q[1], q[2], f1);
        let a2 = lerp(q[2], q[3], vdivq_f32(dt2, distances[2]));
        let b0 = lerp(a0, a1, vdivq_f32(dt0, sums[0]));
        let b1 = lerp(a1, a2, vdivq_f32(dt1, sums[1]));
        let [x, y] = lerp(b0, b1, f1);
        let mut xy = [0.0; 8];
        // SAFETY: the array holds the two interleaved vectors.
        unsafe { vst2q_f32(xy.as_mut_ptr(), float32x4x2_t(x, y)) };
        for (point, &[x, y]) in points.iter_mut().zip(xy.as_chunks::<2>().0) {
            *point = Point::new(x, y);
        }
    }

    // The caller writes the span's first control point directly.
    out.extend_from_slice(&points[1..]);
}
