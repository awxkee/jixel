/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::xyb::*;
use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn halley_cbrt_wasm(x: v128, a: v128) -> v128 {
    let tx = f32x4_mul(f32x4_mul(x, x), x);
    let two = f32x4_splat(2.0);
    let num = f32x4_add(tx, f32x4_mul(a, two));
    let den = f32x4_add(a, f32x4_mul(tx, two));

    f32x4_mul(x, f32x4_div(num, den))
}

#[inline]
#[target_feature(enable = "simd128")]
fn integer_pow_1_3_wasm(hx: v128) -> v128 {
    let scale = u32x4_splat(341);

    let lo64 = u64x2_shr(u64x2_extmul_low_u32x4(hx, scale), 10);
    let hi64 = u64x2_shr(u64x2_extmul_high_u32x4(hx, scale), 10);
    i8x16_shuffle::<
        0,
        1,
        2,
        3, // lo lane 0 low u32
        8,
        9,
        10,
        11, // lo lane 1 low u32
        16,
        17,
        18,
        19, // hi lane 0 low u32
        24,
        25,
        26,
        27, // hi lane 1 low u32
    >(lo64, hi64)
}

#[inline]
#[target_feature(enable = "simd128")]
fn cbrt_seed_positive_wasm(x: v128) -> v128 {
    i32x4_add(integer_pow_1_3_wasm(x), i32x4_splat(709958130))
}

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) fn vcbrt_fast3_positive_wasm(a0: v128, a1: v128, a2: v128) -> (v128, v128, v128) {
    let mut x0 = cbrt_seed_positive_wasm(a0);
    let mut x1 = cbrt_seed_positive_wasm(a1);
    let mut x2 = cbrt_seed_positive_wasm(a2);

    // Interleave the three cube roots to expose independent work.
    x0 = halley_cbrt_wasm(x0, a0);
    x1 = halley_cbrt_wasm(x1, a1);
    x2 = halley_cbrt_wasm(x2, a2);

    x0 = halley_cbrt_wasm(x0, a0);
    x1 = halley_cbrt_wasm(x1, a1);
    x2 = halley_cbrt_wasm(x2, a2);

    let zero = f32x4_splat(0.0);
    x0 = v128_bitselect(zero, x0, f32x4_eq(a0, zero));
    x1 = v128_bitselect(zero, x1, f32x4_eq(a1, zero));
    x2 = v128_bitselect(zero, x2, f32x4_eq(a2, zero));

    (x0, x1, x2)
}

#[inline]
#[target_feature(enable = "simd128")]
fn rgb_to_xyb_f32x4_wasm(m: &XybMatrix, r: v128, g: v128, b: v128) -> (v128, v128, v128) {
    let bias = f32x4_splat(OPSIN_BIAS);

    // mixed rows: m.fwd[r0..r2] per channel + OPSIN_BIAS
    let mut mixed0 = f32x4_add(bias, f32x4_mul(b, f32x4_splat(m.fwd[2])));
    mixed0 = f32x4_add(mixed0, f32x4_mul(g, f32x4_splat(m.fwd[1])));
    mixed0 = f32x4_add(mixed0, f32x4_mul(r, f32x4_splat(m.fwd[0])));

    // mixed rows: m.fwd[r0..r2] per channel + OPSIN_BIAS
    let mut mixed1 = f32x4_add(bias, f32x4_mul(b, f32x4_splat(m.fwd[5])));
    mixed1 = f32x4_add(mixed1, f32x4_mul(g, f32x4_splat(m.fwd[4])));
    mixed1 = f32x4_add(mixed1, f32x4_mul(r, f32x4_splat(m.fwd[3])));

    // mixed2 = m20*r + m21*g + m22*b + OPSIN_BIAS (B row selected by RED)
    let mut mixed2 = f32x4_add(bias, f32x4_mul(b, f32x4_splat(m.fwd[8])));
    mixed2 = f32x4_add(mixed2, f32x4_mul(g, f32x4_splat(m.fwd[7])));
    mixed2 = f32x4_add(mixed2, f32x4_mul(r, f32x4_splat(m.fwd[6])));

    let zero = f32x4_splat(0.0);
    mixed0 = f32x4_max(mixed0, zero);
    mixed1 = f32x4_max(mixed1, zero);
    mixed2 = f32x4_max(mixed2, zero);

    let (tm0, tm1, tm2) = vcbrt_fast3_positive_wasm(mixed0, mixed1, mixed2);

    let neg_bias = f32x4_splat(NEG_BIAS_CBRT);

    let tm0 = f32x4_add(tm0, neg_bias);
    let tm1 = f32x4_add(tm1, neg_bias);
    let tm2 = f32x4_add(tm2, neg_bias);

    let half = f32x4_splat(0.5);

    let x = f32x4_mul(f32x4_sub(tm0, tm1), half);
    let y = f32x4_mul(f32x4_add(tm0, tm1), half);

    (x, y, tm2)
}

/// Transform one row-band into separate output planes.
#[target_feature(enable = "simd128")]
pub(crate) fn to_xyb_wasm_band(
    m: &XybMatrix,
    input: [&[f32]; 3],
    output: [&mut [f32]; 3],
    w: usize,
) {
    let [rp, gp, bp] = input;
    let [xp, yp, out_bp] = output;
    for (((((r_row, g_row), b_row), x_row), y_row), out_b_row) in rp
        .chunks_exact(w)
        .zip(gp.chunks_exact(w))
        .zip(bp.chunks_exact(w))
        .zip(xp.chunks_exact_mut(w))
        .zip(yp.chunks_exact_mut(w))
        .zip(out_bp.chunks_exact_mut(w))
    {
        let (r_chunks, r_tail) = r_row.as_chunks::<4>();
        let (g_chunks, g_tail) = g_row.as_chunks::<4>();
        let (b_chunks, b_tail) = b_row.as_chunks::<4>();
        let (x_chunks, x_tail) = x_row.as_chunks_mut::<4>();
        let (y_chunks, y_tail) = y_row.as_chunks_mut::<4>();
        let (out_b_chunks, out_b_tail) = out_b_row.as_chunks_mut::<4>();

        for (((((r4, g4), b4), x4), y4), out_b4) in r_chunks
            .iter()
            .zip(g_chunks.iter())
            .zip(b_chunks.iter())
            .zip(x_chunks.iter_mut())
            .zip(y_chunks.iter_mut())
            .zip(out_b_chunks.iter_mut())
        {
            let r = unsafe { v128_load(r4.as_ptr().cast()) };
            let g = unsafe { v128_load(g4.as_ptr().cast()) };
            let b = unsafe { v128_load(b4.as_ptr().cast()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_wasm(m, r, g, b);

            unsafe {
                v128_store(x4.as_mut_ptr().cast(), xv);
                v128_store(y4.as_mut_ptr().cast(), yv);
                v128_store(out_b4.as_mut_ptr().cast(), bv);
            }
        }

        if !r_tail.is_empty() {
            let mut r4 = [0.0f32; 4];
            let mut g4 = [0.0f32; 4];
            let mut b4 = [0.0f32; 4];

            r4[..r_tail.len()].copy_from_slice(r_tail);
            g4[..g_tail.len()].copy_from_slice(g_tail);
            b4[..b_tail.len()].copy_from_slice(b_tail);

            let r = unsafe { v128_load(r4.as_ptr().cast()) };
            let g = unsafe { v128_load(g4.as_ptr().cast()) };
            let b = unsafe { v128_load(b4.as_ptr().cast()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_wasm(m, r, g, b);

            unsafe {
                v128_store(r4.as_mut_ptr().cast(), xv);
                v128_store(g4.as_mut_ptr().cast(), yv);
                v128_store(b4.as_mut_ptr().cast(), bv);
            }

            x_tail.copy_from_slice(&r4[..r_tail.len()]);
            y_tail.copy_from_slice(&g4[..g_tail.len()]);
            out_b_tail.copy_from_slice(&b4[..b_tail.len()]);
        }
    }
}
