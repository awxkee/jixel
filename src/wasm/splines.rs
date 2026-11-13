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

//! Spline interpolation, continuous IDCT, and tiled rendering with SIMD128.

use crate::splines::{Point, SPAN_STEPS, Sample};
use std::arch::wasm32::*;

pub(crate) fn sample_span_wasm(
    controls: &[Point<f32>; 4],
    distances: [f32; 3],
    knots: [f32; 4],
    out: &mut Vec<Point<f32>>,
) {
    let [d0, d1, d2] = distances;
    let sums = [f32x4_splat(d0 + d1), f32x4_splat(d1 + d2)];
    let distances = distances.map(|d| f32x4_splat(d));
    let knots = knots.map(|t| f32x4_splat(t));
    let q = controls.map(|p| [f32x4_splat(p.x), f32x4_splat(p.y)]);
    let mut points = [Point::default(); SPAN_STEPS.len()];

    let lerp = |a: [v128; 2], b: [v128; 2], f| {
        [
            f32x4_add(a[0], f32x4_mul(f, f32x4_sub(b[0], a[0]))),
            f32x4_add(a[1], f32x4_mul(f, f32x4_sub(b[1], a[1]))),
        ]
    };
    for (points, steps) in points
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(SPAN_STEPS.as_chunks::<4>().0)
    {
        // SAFETY: the step array holds four elements.
        let step = unsafe { v128_load(steps.as_ptr().cast()) };
        let tt = f32x4_add(distances[0], f32x4_mul(step, distances[1]));
        let dt0 = f32x4_sub(tt, knots[0]);
        let dt1 = f32x4_sub(tt, knots[1]);
        let dt2 = f32x4_sub(tt, knots[2]);
        // Keep the scalar division and interpolation rounding in each lane.
        let f1 = f32x4_div(dt1, distances[1]);
        let a0 = lerp(q[0], q[1], f32x4_div(dt0, distances[0]));
        let a1 = lerp(q[1], q[2], f1);
        let a2 = lerp(q[2], q[3], f32x4_div(dt2, distances[2]));
        let b0 = lerp(a0, a1, f32x4_div(dt0, sums[0]));
        let b1 = lerp(a1, a2, f32x4_div(dt1, sums[1]));
        let [x, y] = lerp(b0, b1, f1);
        let mut xy = [[0.0; 4]; 2];
        // SAFETY: each array holds one vector of interleaved coordinates.
        unsafe {
            v128_store(xy[0].as_mut_ptr().cast(), i32x4_shuffle::<0, 4, 1, 5>(x, y));
            v128_store(xy[1].as_mut_ptr().cast(), i32x4_shuffle::<2, 6, 3, 7>(x, y));
        }
        for (point, &[x, y]) in points.iter_mut().zip(xy.as_flattened().as_chunks::<2>().0) {
            *point = Point { x, y };
        }
    }

    // The caller writes the span's first control point directly.
    out.extend_from_slice(&points[1..]);
}

fn fast_cos_wasm(x: v128) -> v128 {
    use std::f32::consts::{PI, SQRT_2};
    let pi2 = f32x4_splat(PI * 2.0);
    let periods = f32x4_floor(f32x4_mul(x, f32x4_splat(0.5 / PI)));
    let xmod = f32x4_sub(x, f32x4_mul(periods, pi2));
    let x_pi = f32x4_min(xmod, f32x4_sub(pi2, xmod));
    let above = f32x4_ge(x_pi, f32x4_splat(PI / 2.0));
    let xh = v128_bitselect(f32x4_sub(f32x4_splat(PI), x_pi), x_pi, above);
    let xs = f32x4_mul(xh, f32x4_splat(0.25));
    let x2 = f32x4_mul(xs, xs);
    let x4 = f32x4_mul(x2, x2);
    let pre = f32x4_add(
        f32x4_mul(x4, f32x4_splat(0.06960438)),
        f32x4_add(
            f32x4_mul(x2, f32x4_splat(-0.84087373)),
            f32x4_splat(1.68179268),
        ),
    );
    let s1 = f32x4_sub(f32x4_mul(pre, pre), f32x4_splat(SQRT_2));
    let s2 = f32x4_sub(f32x4_mul(s1, s1), f32x4_splat(1.0));
    v128_bitselect(f32x4_neg(s2), s2, above)
}

pub(crate) fn continuous_idct_wasm(dct: &[[f32; 4]], t: f32) -> [f32; 4] {
    if dct.len() == 32 {
        continuous_idct_impl(&dct[..32], t)
    } else {
        continuous_idct_impl(dct, t)
    }
}

#[inline]
fn continuous_idct_impl(dct: &[[f32; 4]], t: f32) -> [f32; 4] {
    let mut result = f32x4_splat(0.0);
    let mut indices = f32x4(0.0, 1.0, 2.0, 3.0);
    let cosine = |indices| {
        let args = f32x4_mul(
            f32x4_mul(indices, f32x4_splat(std::f32::consts::PI / 32.0)),
            f32x4_splat(t + 0.5),
        );
        let cos = fast_cos_wasm(args);
        [
            f32x4_splat(f32x4_extract_lane::<0>(cos)),
            f32x4_splat(f32x4_extract_lane::<1>(cos)),
            f32x4_splat(f32x4_extract_lane::<2>(cos)),
            f32x4_splat(f32x4_extract_lane::<3>(cos)),
        ]
    };
    let mut accumulate = |row: &[f32; 4], cos| {
        let values = unsafe { v128_load(row.as_ptr().cast()) };
        result = f32x4_add(result, f32x4_mul(values, cos));
    };
    let (blocks, tail) = dct.as_chunks::<4>();
    for rows in blocks {
        rows.iter()
            .zip(cosine(indices))
            .for_each(|(row, cos)| accumulate(row, cos));
        indices = f32x4_add(indices, f32x4_splat(4.0));
    }
    if !tail.is_empty() {
        tail.iter()
            .zip(cosine(indices))
            .for_each(|(row, cos)| accumulate(row, cos));
    }
    let mut out = [0.0; 4];
    unsafe { v128_store(out.as_mut_ptr().cast(), result) };
    out
}

#[inline]
fn fast_erf_wasm(x: v128) -> v128 {
    let a = f32x4_abs(x);
    let d1 = f32x4_add(
        f32x4_mul(a, f32x4_splat(7.77394369e-02)),
        f32x4_splat(2.05260015e-04),
    );
    let d2 = f32x4_add(f32x4_mul(d1, a), f32x4_splat(2.32120216e-01));
    let d3 = f32x4_add(f32x4_mul(d2, a), f32x4_splat(2.77820801e-01));
    let one = f32x4_splat(1.0);
    let d4 = f32x4_add(f32x4_mul(d3, a), one);
    let inv = f32x4_div(one, f32x4_mul(d4, d4));
    let r = f32x4_sub(one, f32x4_mul(inv, inv));
    v128_bitselect(f32x4_neg(r), r, f32x4_le(x, f32x4_splat(0.0)))
}

#[inline]
fn blob_weights(dx_squared: v128, dy_squared: f32, inv_sigma: v128, amp: v128) -> v128 {
    let dist = f32x4_sqrt(f32x4_add(dx_squared, f32x4_splat(dy_squared)));
    let half = f32x4_mul(dist, f32x4_splat(0.5));
    let offset = f32x4_splat(0.353_553_39);
    let a = f32x4_mul(f32x4_add(half, offset), inv_sigma);
    let b = f32x4_mul(f32x4_sub(half, offset), inv_sigma);
    let e = f32x4_sub(fast_erf_wasm(a), fast_erf_wasm(b));
    f32x4_mul(f32x4_mul(amp, e), e)
}

/// Draw one clipped sample into an 8x8 tile, sharing x distances across rows.
pub(crate) fn spline_render_tile_wasm(
    sample: &Sample,
    tx: usize,
    ty: usize,
    (x0, x1, y0, y1): (usize, usize, usize, usize),
    pixels: &mut [[f32; 64]; 3],
    sign: f32,
) {
    debug_assert!(x0 >= tx && x1 <= tx + 8 && x0 < x1);
    debug_assert!(y0 >= ty && y1 <= ty + 8 && y0 < y1);
    let local_x0 = x0 - tx;
    let local_x1 = x1 - tx;
    let local_y = y0 - ty..y1 - ty;
    let amp = f32x4_splat(sample.amp * sign);
    let inv_sigma = f32x4_splat(sample.inv_sigma);
    let colors = sample.color.map(|color| f32x4_splat(color));
    let [rx, ry, rb] = pixels.each_mut().map(|plane| plane.as_chunks_mut::<8>().0);
    let offsets = i32x4(0, 1, 2, 3);
    for start in (local_x0..local_x1).step_by(4) {
        let end = (start + 4).min(local_x1);
        let col = start.min(4);
        let columns = i32x4_add(i32x4_splat(col as i32), offsets);
        // Convert absolute integer coordinates so rounding is correct above 2^24.
        let xs = f32x4_convert_i32x4(i32x4_add(i32x4_splat(tx as i32), columns));
        let dx = f32x4_sub(xs, f32x4_splat(sample.position.x));
        let dx_squared = f32x4_mul(dx, dx);
        let active = v128_and(
            i32x4_ge(columns, i32x4_splat(start as i32)),
            i32x4_lt(columns, i32x4_splat(end as i32)),
        );
        let clipped = col < start || col + 4 > end;
        for (((rx, ry), rb), y) in rx[local_y.clone()]
            .iter_mut()
            .zip(&mut ry[local_y.clone()])
            .zip(&mut rb[local_y.clone()])
            .zip(y0..)
        {
            let dy = y as f32 - sample.position.y;
            let weights = blob_weights(dx_squared, dy * dy, inv_sigma, amp);
            for (row, color) in [rx, ry, rb].into_iter().zip(colors) {
                let block = row[col..].first_chunk_mut::<4>().unwrap();
                // SAFETY: the block contains four readable and writable pixels.
                unsafe {
                    let ptr = block.as_mut_ptr().cast();
                    let original = v128_load(ptr);
                    let drawn = f32x4_add(original, f32x4_mul(color, weights));
                    let out = if clipped {
                        v128_bitselect(drawn, original, active)
                    } else {
                        drawn
                    };
                    v128_store(ptr, out);
                }
            }
        }
    }
}
