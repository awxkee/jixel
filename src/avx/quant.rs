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
use std::arch::x86_64::*;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn quantize_block_ac_avx2(
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
    debug_assert!(width.is_multiple_of(8));

    let qm = &qm[..n];
    let block_in = &block_in[..n];
    let block_out = &mut block_out[..n];

    let thr = crate::enc_group::quantize_ac_thresholds(c, xsize, ysize);
    let q_scaled = crate::enc_group::quantize_ac_q_scaled(quant, scale, qm_multiplier);

    let half = width / 2;
    let qs = _mm256_set1_ps(q_scaled);
    let sign = _mm256_set1_ps(-0.0);
    let zero_i = _mm256_setzero_si256();
    let lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

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

        for (x0, ((qm8, in8), out8)) in qm_row
            .as_chunks::<8>()
            .0
            .iter()
            .zip(in_row.as_chunks::<8>().0.iter())
            .zip(out_row.as_chunks_mut::<8>().0.iter_mut())
            .enumerate()
        {
            let x = x0 * 8;
            let thr_v = if x + 8 <= half {
                _mm256_set1_ps(thr_lo)
            } else if x >= half {
                _mm256_set1_ps(thr_hi)
            } else {
                // Width 8 is the only current shape where one AVX vector crosses
                // the low/high threshold boundary (half = 4).
                let lane_x = _mm256_add_epi32(_mm256_set1_epi32(x as i32), lane_ids);
                let hi = _mm256_cmpgt_epi32(lane_x, _mm256_set1_epi32(half as i32 - 1));
                _mm256_blendv_ps(
                    _mm256_set1_ps(thr_lo),
                    _mm256_set1_ps(thr_hi),
                    _mm256_castsi256_ps(hi),
                )
            };

            let qmv = unsafe { _mm256_loadu_ps(qm8.as_ptr()) };
            let inv = unsafe { _mm256_loadu_ps(in8.as_ptr()) };
            let val = _mm256_mul_ps(_mm256_mul_ps(qmv, qs), inv);
            let abs = _mm256_andnot_ps(sign, val);
            let keep = _mm256_cmp_ps::<_CMP_GE_OQ>(abs, thr_v);
            let q = _mm256_cvttps_epi32(val);
            let q = _mm256_blendv_epi8(zero_i, q, _mm256_castps_si256(keep));
            unsafe { _mm256_storeu_si256(out8.as_mut_ptr().cast::<__m256i>(), q) };
        }
    }
}
