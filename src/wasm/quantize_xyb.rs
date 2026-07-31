/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn round_ties_away_i32(v: v128) -> v128 {
    let truncated = f32x4_trunc(v);
    let fraction = f32x4_sub(v, truncated);
    let at_least_half = f32x4_ge(f32x4_abs(fraction), f32x4_splat(0.5));
    let signed_one = v128_or(f32x4_splat(1.0), v128_and(v, f32x4_splat(-0.0)));
    i32x4_trunc_sat_f32x4(f32x4_add(truncated, v128_and(signed_one, at_least_half)))
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_interleaved3_i32x4(output: &mut [[i32; 3]; 4], y: v128, x: v128, b: v128) {
    let yb01 = i32x4_shuffle::<0, 4, 1, 5>(y, b);
    let xb12 = i32x4_shuffle::<1, 5, 2, 6>(x, b);
    let bx23 = i32x4_shuffle::<2, 6, 3, 7>(b, x);
    // Store [Y0 X0 B0 Y1], [X1 B1 Y2 X2], [B2 Y3 X3 B3].
    let out0 = i32x4_shuffle::<0, 4, 1, 2>(yb01, x);
    let out1 = i32x4_shuffle::<0, 1, 6, 2>(xb12, y);
    let out2 = i32x4_shuffle::<0, 7, 3, 2>(bx23, y);
    let dst = output.as_mut_ptr().cast::<i32>();
    unsafe {
        v128_store(dst.cast(), out0);
        v128_store(dst.add(4).cast(), out1);
        v128_store(dst.add(8).cast(), out2);
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn quantize_xyb_channels_wasm(
    input: [&[f32]; 3],
    output: [&mut [i32]; 3],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [dst_y, dst_x, dst_b] = output;
    let [scale_x, scale_y, scale_b] = scales;
    let (x4s, x_tail) = src_x.as_chunks::<4>();
    let (y4s, y_tail) = src_y.as_chunks::<4>();
    let (b4s, b_tail) = src_b.as_chunks::<4>();
    let (yq4s, yq_tail) = dst_y.as_chunks_mut::<4>();
    let (xq4s, xq_tail) = dst_x.as_chunks_mut::<4>();
    let (bq4s, bq_tail) = dst_b.as_chunks_mut::<4>();
    let sx = f32x4_splat(scale_x);
    let sy = f32x4_splat(scale_y);
    let sb = f32x4_splat(scale_b);

    for (((((x4, y4), b4), yq4), xq4), bq4) in
        x4s.iter().zip(y4s).zip(b4s).zip(yq4s).zip(xq4s).zip(bq4s)
    {
        let x = unsafe { v128_load(x4.as_ptr().cast()) };
        let y = unsafe { v128_load(y4.as_ptr().cast()) };
        let b = unsafe { v128_load(b4.as_ptr().cast()) };
        let yq = round_ties_away_i32(f32x4_mul(y, sy));
        let xq = round_ties_away_i32(f32x4_mul(x, sx));
        let bq = round_ties_away_i32(f32x4_mul(b, sb));
        unsafe {
            v128_store(yq4.as_mut_ptr().cast(), yq);
            v128_store(xq4.as_mut_ptr().cast(), xq);
            v128_store(bq4.as_mut_ptr().cast(), i32x4_sub(bq, yq));
        }
    }

    let src = x_tail.iter().zip(y_tail).zip(b_tail);
    let dst = yq_tail.iter_mut().zip(xq_tail).zip(bq_tail);
    for (((x, y), b), ((yq, xq), bq)) in src.zip(dst) {
        let quantized_y = (*y * scale_y).round() as i32;
        *yq = quantized_y;
        *xq = (*x * scale_x).round() as i32;
        *bq = (*b * scale_b).round() as i32 - quantized_y;
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn quantize_xyb_tile_colors_wasm(
    input: [&[f32]; 3],
    output: &mut [[i32; 3]],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [scale_x, scale_y, scale_b] = scales;
    let (x4s, x_tail) = src_x.as_chunks::<4>();
    let (y4s, y_tail) = src_y.as_chunks::<4>();
    let (b4s, b_tail) = src_b.as_chunks::<4>();
    let (out4s, out_tail) = output.as_chunks_mut::<4>();
    let sx = f32x4_splat(scale_x);
    let sy = f32x4_splat(scale_y);
    let sb = f32x4_splat(scale_b);

    for (((x4, y4), b4), out4) in x4s.iter().zip(y4s).zip(b4s).zip(out4s) {
        let x = unsafe { v128_load(x4.as_ptr().cast()) };
        let y = unsafe { v128_load(y4.as_ptr().cast()) };
        let b = unsafe { v128_load(b4.as_ptr().cast()) };
        let yq = round_ties_away_i32(f32x4_mul(y, sy));
        let xq = round_ties_away_i32(f32x4_mul(x, sx));
        let bq = round_ties_away_i32(f32x4_mul(b, sb));
        store_interleaved3_i32x4(out4, yq, xq, bq);
    }

    let src = x_tail.iter().zip(y_tail).zip(b_tail);
    for (((x, y), b), out) in src.zip(out_tail) {
        *out = [
            (*y * scale_y).round() as i32,
            (*x * scale_x).round() as i32,
            (*b * scale_b).round() as i32,
        ];
    }
}
