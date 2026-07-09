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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn quantize_block_ac_sse41(
    block_in: &[f32],
    c: usize,
    qm: &[f32],
    quant: i32,
    scale: f32,
    qm_multiplier: f32,
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

    let thr = crate::enc_group::quantize_ac_thresholds(c, xsize, ysize);
    let q_scaled = crate::enc_group::quantize_ac_q_scaled(quant, scale, qm_multiplier);

    let half = width / 2;
    let qs = _mm_set1_ps(q_scaled);
    let sign = _mm_set1_ps(-0.0);
    let zero_i = _mm_setzero_si128();

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
            let qmv = unsafe { _mm_loadu_ps(qm4.as_ptr()) };
            let inv = unsafe { _mm_loadu_ps(in4.as_ptr()) };
            let val = _mm_mul_ps(_mm_mul_ps(qmv, qs), inv);
            let abs = _mm_andnot_ps(sign, val);
            let keep = _mm_cmpge_ps(abs, _mm_set1_ps(threshold));
            let q = _mm_cvtps_epi32(val);
            let q = _mm_blendv_epi8(zero_i, q, _mm_castps_si128(keep));
            unsafe { _mm_storeu_si128(out4.as_mut_ptr().cast::<__m128i>(), q) };
        }
    }
}
