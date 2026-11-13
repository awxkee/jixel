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

//! Spline rendering and continuous IDCT with SSE2 and SSE4.1.

use crate::splines::Sample;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse2")]
fn fast_cos(x: __m128, periods: __m128) -> __m128 {
    use std::f32::consts::{PI, SQRT_2};
    let pi2 = _mm_set1_ps(PI * 2.0);
    let xmod = _mm_sub_ps(x, _mm_mul_ps(periods, pi2));
    let x_pi = _mm_min_ps(xmod, _mm_sub_ps(pi2, xmod));
    let above = _mm_cmpge_ps(x_pi, _mm_set1_ps(PI / 2.0));
    let xh = _mm_or_ps(
        _mm_and_ps(above, _mm_sub_ps(_mm_set1_ps(PI), x_pi)),
        _mm_andnot_ps(above, x_pi),
    );
    let xs = _mm_mul_ps(xh, _mm_set1_ps(0.25));
    let x2 = _mm_mul_ps(xs, xs);
    let x4 = _mm_mul_ps(x2, x2);
    let pre = _mm_add_ps(
        _mm_mul_ps(x4, _mm_set1_ps(0.06960438)),
        _mm_add_ps(
            _mm_mul_ps(x2, _mm_set1_ps(-0.84087373)),
            _mm_set1_ps(1.68179268),
        ),
    );
    let s1 = _mm_sub_ps(_mm_mul_ps(pre, pre), _mm_set1_ps(SQRT_2));
    let s2 = _mm_sub_ps(_mm_mul_ps(s1, s1), _mm_set1_ps(1.0));
    _mm_xor_ps(s2, _mm_and_ps(above, _mm_set1_ps(-0.0)))
}

// Generate both target-feature bodies so the rounding and polynomial stay
// inlined in the frequency loop, with no per-vector function-pointer call.
macro_rules! continuous_idct {
    ($name:ident, $feature:literal, $floor:expr) => {
        #[target_feature(enable = $feature)]
        pub(crate) fn $name(dct: &[[f32; 4]], t: f32) -> [f32; 4] {
            #[inline]
            #[target_feature(enable = $feature)]
            fn eval(dct: &[[f32; 4]], t: f32) -> [f32; 4] {
                let mut result = _mm_setzero_ps();
                let mut indices = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
                let cosine = |indices| {
                    let args = _mm_mul_ps(
                        _mm_mul_ps(indices, _mm_set1_ps(std::f32::consts::PI / 32.0)),
                        _mm_set1_ps(t + 0.5),
                    );
                    let periods =
                        ($floor)(_mm_mul_ps(args, _mm_set1_ps(0.5 / std::f32::consts::PI)));
                    let cos = fast_cos(args, periods);
                    [
                        _mm_shuffle_ps::<0x00>(cos, cos),
                        _mm_shuffle_ps::<0x55>(cos, cos),
                        _mm_shuffle_ps::<0xaa>(cos, cos),
                        _mm_shuffle_ps::<0xff>(cos, cos),
                    ]
                };
                let mut accumulate = |row: &[f32; 4], cos| {
                    let values = unsafe { _mm_loadu_ps(row.as_ptr()) };
                    result = _mm_add_ps(result, _mm_mul_ps(values, cos));
                };
                let (blocks, tail) = dct.as_chunks::<4>();
                for rows in blocks {
                    rows.iter()
                        .zip(cosine(indices))
                        .for_each(|(row, cos)| accumulate(row, cos));
                    indices = _mm_add_ps(indices, _mm_set1_ps(4.0));
                }
                if !tail.is_empty() {
                    tail.iter()
                        .zip(cosine(indices))
                        .for_each(|(row, cos)| accumulate(row, cos));
                }
                let mut out = [0.0; 4];
                unsafe { _mm_storeu_ps(out.as_mut_ptr(), result) };
                out
            }

            if dct.len() == 32 {
                eval(&dct[..32], t)
            } else {
                eval(dct, t)
            }
        }
    };
}

continuous_idct!(continuous_idct_sse41, "sse4.1", _mm_floor_ps);
// Spline arguments are finite, nonnegative and bounded by PI * 31 * 31.5 / 32.
// The pre-SSE4.1 fallback can therefore use truncation for floor.
continuous_idct!(continuous_idct_sse2, "sse2", |x| _mm_cvtepi32_ps(
    _mm_cvttps_epi32(x)
));

#[inline]
#[target_feature(enable = "sse2")]
fn erf(x: __m128) -> __m128 {
    let sign = _mm_set1_ps(-0.0);
    let abs = _mm_andnot_ps(sign, x);
    let mul_add = |a, b, c| _mm_add_ps(_mm_mul_ps(a, b), _mm_set1_ps(c));
    let d1 = mul_add(abs, _mm_set1_ps(7.77394369e-02), 2.05260015e-04);
    let d2 = mul_add(d1, abs, 2.32120216e-01);
    let d3 = mul_add(d2, abs, 2.77820801e-01);
    let d4 = mul_add(d3, abs, 1.0);
    let one = _mm_set1_ps(1.0);
    let inv = _mm_div_ps(one, _mm_mul_ps(d4, d4));
    let result = _mm_sub_ps(one, _mm_mul_ps(inv, inv));
    let nonpositive = _mm_cmple_ps(x, _mm_setzero_ps());
    _mm_xor_ps(result, _mm_and_ps(nonpositive, sign))
}

#[inline]
#[target_feature(enable = "sse2")]
fn blob_weights(dx_squared: __m128, dy: f32, inv_sigma: __m128, amp: __m128) -> __m128 {
    let distance = _mm_sqrt_ps(_mm_add_ps(dx_squared, _mm_set1_ps(dy * dy)));
    let half = _mm_mul_ps(distance, _mm_set1_ps(0.5));
    let offset = _mm_set1_ps(0.353_553_39);
    let a = _mm_mul_ps(_mm_add_ps(half, offset), inv_sigma);
    let b = _mm_mul_ps(_mm_sub_ps(half, offset), inv_sigma);
    let e = _mm_sub_ps(erf(a), erf(b));
    _mm_mul_ps(_mm_mul_ps(amp, e), e)
}

/// Draw a clipped sample, reusing each vector's x distances across the tile's rows.
#[target_feature(enable = "sse2")]
pub(crate) fn spline_render_tile_sse2(
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
    let [rx, ry, rb] = pixels.each_mut().map(|plane| plane.as_chunks_mut::<8>().0);
    let offsets = _mm_setr_epi32(0, 1, 2, 3);
    let center = _mm_set1_ps(sample.position.x);
    let inv_sigma = _mm_set1_ps(sample.inv_sigma);
    let amp = _mm_set1_ps(sample.amp * sign);
    let colors = sample.color.map(|color| _mm_set1_ps(color));
    let split = (local_x0 + 4).min(local_x1);
    for (start, end) in [(local_x0, split), (split, local_x1)] {
        if start == end {
            break;
        }
        let col = start.min(4);
        // Convert absolute integer coordinates together to preserve rounding above 2^24.
        let x = _mm_add_epi32(_mm_set1_epi32((tx + col) as i32), offsets);
        let dx = _mm_sub_ps(_mm_cvtepi32_ps(x), center);
        let dx_squared = _mm_mul_ps(dx, dx);
        let columns = _mm_add_epi32(_mm_set1_epi32(col as i32), offsets);
        let mask = _mm_castsi128_ps(_mm_and_si128(
            _mm_cmpgt_epi32(columns, _mm_set1_epi32(start as i32 - 1)),
            _mm_cmpgt_epi32(_mm_set1_epi32(end as i32), columns),
        ));
        let clipped = col < start || col + 4 > end;
        for (((rx, ry), rb), y) in rx[local_y.clone()]
            .iter_mut()
            .zip(&mut ry[local_y.clone()])
            .zip(&mut rb[local_y.clone()])
            .zip(y0..)
        {
            let weights = blob_weights(dx_squared, y as f32 - sample.position.y, inv_sigma, amp);
            for (row, color) in [rx, ry, rb].into_iter().zip(colors) {
                let block = row[col..].first_chunk_mut::<4>().unwrap();
                // SAFETY: each block has four readable and writable pixels.
                unsafe {
                    let ptr = block.as_mut_ptr();
                    let original = _mm_loadu_ps(ptr);
                    let drawn = _mm_add_ps(original, _mm_mul_ps(color, weights));
                    let out = if clipped {
                        _mm_or_ps(_mm_and_ps(mask, drawn), _mm_andnot_ps(mask, original))
                    } else {
                        drawn
                    };
                    _mm_storeu_ps(ptr, out);
                }
            }
        }
    }
}
