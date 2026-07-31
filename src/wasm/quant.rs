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
use core::arch::wasm32::*;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "simd128")]
pub(crate) fn quantize_block_ac_wasm(
    block_in: &[f32],
    c: usize,
    qm: &[f32],
    quant: i32,
    scale: f32,
    qm_multiplier: f32,
    distance: f32,
    xsize: usize,
    ysize: usize,
    block_out: &mut [i32],
) {
    let width = xsize * 8;
    let height = ysize * 8;
    let n = width * height;

    debug_assert_eq!(qm.len(), n);
    debug_assert!(block_in.len() >= n, "block_in too small");
    debug_assert!(block_out.len() >= n, "block_out too small");
    debug_assert!(width.is_multiple_of(4));

    let qm = &qm[..n];
    let block_in = &block_in[..n];
    let block_out = &mut block_out[..n];

    let thr = crate::enc_group::quantize_ac_thresholds(c, xsize, ysize, distance);
    let q_scaled = crate::enc_group::quantize_ac_q_scaled(quant, scale, qm_multiplier);

    let half = width / 2;
    let qs = f32x4_splat(q_scaled);
    let zero_i = i32x4_splat(0);

    for (y, ((qm_row, in_row), out_row)) in qm
        .chunks_exact(width)
        .zip(block_in.chunks_exact(width))
        .zip(block_out.chunks_exact_mut(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];

        for (x0, ((qm4, in4), out4)) in qm_row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(in_row.as_chunks::<4>().0.iter())
            .zip(out_row.as_chunks_mut::<4>().0.iter_mut())
            .enumerate()
        {
            let x = x0 * 4;
            let threshold = if x >= half { thr_hi } else { thr_lo };
            let qmv = unsafe { v128_load(qm4.as_ptr().cast()) };
            let inv = unsafe { v128_load(in4.as_ptr().cast()) };
            let val = f32x4_mul(f32x4_mul(qmv, qs), inv);
            let keep = f32x4_ge(f32x4_abs(val), f32x4_splat(threshold));
            let truncated = f32x4_trunc(val);
            let frac = f32x4_sub(val, truncated);
            let ge_half = f32x4_ge(f32x4_abs(frac), f32x4_splat(0.5));
            let signed_one = v128_or(f32x4_splat(1.0), v128_and(val, f32x4_splat(-0.0)));
            let q = i32x4_trunc_sat_f32x4(f32x4_add(truncated, v128_and(signed_one, ge_half)));
            let q = v128_bitselect(q, zero_i, keep);
            unsafe { v128_store(out4.as_mut_ptr().cast(), q) };
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn round_ties_away_i32x4(value: v128) -> v128 {
    let truncated = f32x4_trunc(value);
    let fraction = f32x4_sub(value, truncated);
    let at_least_half = f32x4_ge(f32x4_abs(fraction), f32x4_splat(0.5));
    let signed_one = v128_or(f32x4_splat(1.0), v128_and(value, f32x4_splat(-0.0)));
    i32x4_trunc_sat_f32x4(f32x4_add(truncated, v128_and(signed_one, at_least_half)))
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_i16x4(output: &mut [i16; 4], value: v128) {
    let packed = i16x8_narrow_i32x4(value, value);
    unsafe { v128_store64_lane::<0>(packed, output.as_mut_ptr().cast()) };
}

#[target_feature(enable = "simd128")]
pub(crate) fn quantize_dc_wasm(input: &[f32], scale: f32, output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len());
    let (input4, input_tail) = input.as_chunks::<4>();
    let (output4, output_tail) = output.as_chunks_mut::<4>();
    let scale = f32x4_splat(scale);

    for (source, target) in input4.iter().zip(output4) {
        let value = unsafe { v128_load(source.as_ptr().cast()) };
        store_i16x4(target, round_ties_away_i32x4(f32x4_mul(value, scale)));
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 4];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let value = unsafe { v128_load(source.as_ptr().cast()) };
        let mut target = [0i16; 4];
        store_i16x4(&mut target, round_ties_away_i32x4(f32x4_mul(value, scale)));
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn quantize_dc_cfl_wasm(
    input: &[f32],
    y_quant: &[i16],
    scale: f32,
    cfl: f32,
    output: &mut [i16],
) {
    debug_assert_eq!(input.len(), y_quant.len());
    debug_assert_eq!(input.len(), output.len());
    let (input4, input_tail) = input.as_chunks::<4>();
    let (y4, y_tail) = y_quant.as_chunks::<4>();
    let (output4, output_tail) = output.as_chunks_mut::<4>();
    let scale = f32x4_splat(scale);
    let negative_cfl = f32x4_splat(-cfl);

    for ((source, y), target) in input4.iter().zip(y4).zip(output4) {
        let value = unsafe { v128_load(source.as_ptr().cast()) };
        let y = unsafe { v128_load64_zero(y.as_ptr().cast()) };
        let y = f32x4_convert_i32x4(i32x4_extend_low_i16x8(y));
        let value = f32x4_add(f32x4_mul(value, scale), f32x4_mul(y, negative_cfl));
        store_i16x4(target, round_ties_away_i32x4(value));
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 4];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let mut y = [0i16; 4];
        y[..y_tail.len()].copy_from_slice(y_tail);
        let value = unsafe { v128_load(source.as_ptr().cast()) };
        let y = unsafe { v128_load64_zero(y.as_ptr().cast()) };
        let y = f32x4_convert_i32x4(i32x4_extend_low_i16x8(y));
        let value = f32x4_add(f32x4_mul(value, scale), f32x4_mul(y, negative_cfl));
        let mut target = [0i16; 4];
        store_i16x4(&mut target, round_ties_away_i32x4(value));
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}
