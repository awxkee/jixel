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
    debug_assert!(width.is_multiple_of(8));

    let qm = &qm[..n];
    let block_in = &block_in[..n];
    let block_out = &mut block_out[..n];

    let thr = crate::enc_group::quantize_ac_thresholds(c, xsize, ysize, distance);
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
            let truncated = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(val);
            let frac = _mm256_sub_ps(val, truncated);
            let abs_frac = _mm256_andnot_ps(sign, frac);
            let ge_half = _mm256_cmp_ps::<_CMP_GE_OQ>(abs_frac, _mm256_set1_ps(0.5));
            let signed_one = _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(val, sign));
            let q =
                _mm256_cvttps_epi32(_mm256_add_ps(truncated, _mm256_and_ps(signed_one, ge_half)));
            let q = _mm256_blendv_epi8(zero_i, q, _mm256_castps_si256(keep));
            unsafe { _mm256_storeu_si256(out8.as_mut_ptr().cast::<__m256i>(), q) };
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_ties_away_i32x8(value: __m256) -> __m256i {
    let sign = _mm256_set1_ps(-0.0);
    let truncated = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(value);
    let fraction = _mm256_sub_ps(value, truncated);
    let at_least_half =
        _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_andnot_ps(sign, fraction), _mm256_set1_ps(0.5));
    let signed_one = _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(value, sign));
    let rounded = _mm256_add_ps(truncated, _mm256_and_ps(signed_one, at_least_half));
    let quantized = _mm256_cvttps_epi32(rounded);
    let positive_overflow = _mm256_cmp_ps::<_CMP_GE_OQ>(rounded, _mm256_set1_ps(2_147_483_648.0));
    let quantized = _mm256_blendv_epi8(
        quantized,
        _mm256_set1_epi32(i32::MAX),
        _mm256_castps_si256(positive_overflow),
    );
    let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(value, value);
    _mm256_blendv_epi8(quantized, _mm256_setzero_si256(), _mm256_castps_si256(nan))
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8(output: &mut [i16; 8], value: __m256i) {
    let packed = _mm_packs_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256::<1>(value),
    );
    unsafe { _mm_storeu_si128(output.as_mut_ptr().cast(), packed) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn quantize_dc_value_x8(input: __m256, scale: __m256) -> __m256i {
    round_ties_away_i32x8(_mm256_mul_ps(input, scale))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn quantize_dc_cfl_value_x8(
    input: __m256,
    y_quant: __m128i,
    scale: __m256,
    negative_cfl: __m256,
) -> __m256i {
    let y = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(y_quant));
    let correction = _mm256_mul_ps(y, negative_cfl);
    let value = _mm256_fmadd_ps(input, scale, correction);
    round_ties_away_i32x8(value)
}

#[target_feature(enable = "avx2")]
pub(crate) fn quantize_dc_avx2(input: &[f32], scale: f32, output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len());
    let (input8, input_tail) = input.as_chunks::<8>();
    let (output8, output_tail) = output.as_chunks_mut::<8>();
    let scale = _mm256_set1_ps(scale);

    for (source, target) in input8.iter().zip(output8) {
        let value = unsafe { _mm256_loadu_ps(source.as_ptr()) };
        store_i16x8(target, quantize_dc_value_x8(value, scale));
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 8];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let value = unsafe { _mm256_loadu_ps(source.as_ptr()) };
        let mut target = [0i16; 8];
        store_i16x8(&mut target, quantize_dc_value_x8(value, scale));
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn quantize_dc_cfl_avx2(
    input: &[f32],
    y_quant: &[i16],
    scale: f32,
    cfl: f32,
    output: &mut [i16],
) {
    debug_assert_eq!(input.len(), y_quant.len());
    debug_assert_eq!(input.len(), output.len());
    let (input8, input_tail) = input.as_chunks::<8>();
    let (y8, y_tail) = y_quant.as_chunks::<8>();
    let (output8, output_tail) = output.as_chunks_mut::<8>();
    let scale = _mm256_set1_ps(scale);
    let negative_cfl = _mm256_set1_ps(-cfl);

    for ((source, y), target) in input8.iter().zip(y8).zip(output8) {
        let value = unsafe { _mm256_loadu_ps(source.as_ptr()) };
        let y = unsafe { _mm_loadu_si128(y.as_ptr().cast()) };
        store_i16x8(
            target,
            quantize_dc_cfl_value_x8(value, y, scale, negative_cfl),
        );
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 8];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let mut y = [0i16; 8];
        y[..y_tail.len()].copy_from_slice(y_tail);
        let value = unsafe { _mm256_loadu_ps(source.as_ptr()) };
        let y = unsafe { _mm_loadu_si128(y.as_ptr().cast()) };
        let mut target = [0i16; 8];
        store_i16x8(
            &mut target,
            quantize_dc_cfl_value_x8(value, y, scale, negative_cfl),
        );
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}

#[cfg(test)]
mod tests {
    use crate::enc_group::quantize_block_ac_scalar;

    #[test]
    fn quantize_block_ac_avx2_matches_scalar() {
        // Values straddling every rounding and deadzone boundary, including
        // exact halves where ties-to-even and ties-away disagree.
        static EDGES: [f32; 20] = [
            0.0, 0.49, 0.5, 0.51, 0.58, 0.75, 0.99, 1.0, 1.01, 1.49, 1.5, 1.51, 1.99, 2.0, 2.5,
            3.5, -0.5, -0.75, -1.5, -2.5,
        ];
        let mut state = 0x5eed_4321_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        for &(xsize, ysize) in &[
            (1usize, 1usize),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
        ] {
            let n = xsize * 8 * ysize * 8;
            for c in 0..3 {
                for &distance in &[0.5f32, 1.0, 2.5, 4.0] {
                    for &quant in &[1i32, 3, 11] {
                        let qm: Vec<f32> = (0..n).map(|i| 0.3 + 0.01 * (i % 17) as f32).collect();
                        // Half crafted edge values (scaled so the product lands
                        // on the boundary), half random.
                        let block: Vec<f32> = (0..n)
                            .map(|i| {
                                if i % 2 == 0 {
                                    let e = EDGES[(i / 2) % EDGES.len()];
                                    e / (qm[i] * quant as f32)
                                } else {
                                    next() * 8.0
                                }
                            })
                            .collect();
                        let mut want = vec![0i32; n];
                        let mut got = vec![0i32; n];
                        quantize_block_ac_scalar(
                            &block, c, &qm, quant, 1.0, 1.0, distance, xsize, ysize, &mut want,
                        );
                        // SAFETY: the test binary is built with +avx2.
                        unsafe {
                            super::quantize_block_ac_avx2(
                                &block, c, &qm, quant, 1.0, 1.0, distance, xsize, ysize, &mut got,
                            );
                        }
                        for i in 0..n {
                            assert_eq!(
                                got[i], want[i],
                                "mismatch at {i} (shape {xsize}x{ysize}, c={c}, d={distance}, \
                                 q={quant}): scalar {} avx2 {}, input {}",
                                want[i], got[i], block[i]
                            );
                        }
                    }
                }
            }
        }
    }
}
