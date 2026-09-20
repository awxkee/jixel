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

//! Continuous spline IDCT with SIMD128.

fn fast_cos_wasm(x: std::arch::wasm32::v128) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;
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

pub(crate) fn continuous_idct_wasm(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
    use std::arch::wasm32::*;
    let mut result = f32x4_splat(0.0);
    let mut indices = f32x4(0.0, 1.0, 2.0, 3.0);
    for rows in dct.as_chunks::<4>().0 {
        let args = f32x4_mul(
            f32x4_mul(indices, f32x4_splat(std::f32::consts::PI / 32.0)),
            f32x4_splat(t + 0.5),
        );
        let cos = fast_cos_wasm(args);
        let cosines = [
            f32x4_splat(f32x4_extract_lane::<0>(cos)),
            f32x4_splat(f32x4_extract_lane::<1>(cos)),
            f32x4_splat(f32x4_extract_lane::<2>(cos)),
            f32x4_splat(f32x4_extract_lane::<3>(cos)),
        ];
        for (row, cos) in rows.iter().zip(cosines) {
            let values = unsafe { v128_load(row.as_ptr().cast()) };
            result = f32x4_add(result, f32x4_mul(values, cos));
        }
        indices = f32x4_add(indices, f32x4_splat(4.0));
    }
    let mut out = [0.0; 4];
    unsafe { v128_store(out.as_mut_ptr().cast(), result) };
    out
}
