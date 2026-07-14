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
            let q = i32x4_trunc_sat_f32x4(f32x4_nearest(val));
            let q = v128_bitselect(q, zero_i, keep);
            unsafe { v128_store(out4.as_mut_ptr().cast(), q) };
        }
    }
}
