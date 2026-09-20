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

//! Continuous spline IDCT with SSE2 and SSE4.1.

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
        pub(crate) fn $name(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
            let mut result = _mm_setzero_ps();
            let mut indices = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
            for rows in dct.as_chunks::<4>().0 {
                let args = _mm_mul_ps(
                    _mm_mul_ps(indices, _mm_set1_ps(std::f32::consts::PI / 32.0)),
                    _mm_set1_ps(t + 0.5),
                );
                let periods = ($floor)(_mm_mul_ps(args, _mm_set1_ps(0.5 / std::f32::consts::PI)));
                let cos = fast_cos(args, periods);
                let cosines = [
                    _mm_shuffle_ps::<0x00>(cos, cos),
                    _mm_shuffle_ps::<0x55>(cos, cos),
                    _mm_shuffle_ps::<0xaa>(cos, cos),
                    _mm_shuffle_ps::<0xff>(cos, cos),
                ];
                for (row, cos) in rows.iter().zip(cosines) {
                    let values = unsafe { _mm_loadu_ps(row.as_ptr()) };
                    result = _mm_add_ps(result, _mm_mul_ps(values, cos));
                }
                indices = _mm_add_ps(indices, _mm_set1_ps(4.0));
            }
            let mut out = [0.0; 4];
            unsafe { _mm_storeu_ps(out.as_mut_ptr(), result) };
            out
        }
    };
}

continuous_idct!(continuous_idct_sse41, "sse4.1", _mm_floor_ps);
// Spline arguments are finite, nonnegative and bounded by PI * 31 * 31.5 / 32.
// The pre-SSE4.1 fallback can therefore use truncation for floor.
continuous_idct!(continuous_idct_sse2, "sse2", |x| _mm_cvtepi32_ps(
    _mm_cvttps_epi32(x)
));
