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

    let thr = crate::group::quantize_ac_thresholds(c, xsize, ysize, distance);
    let q_scaled = crate::group::quantize_ac_q_scaled(quant, scale, qm_multiplier);

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
            let truncated = _mm_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(val);
            let frac = _mm_sub_ps(val, truncated);
            let abs_frac = _mm_andnot_ps(sign, frac);
            let ge_half = _mm_cmpge_ps(abs_frac, _mm_set1_ps(0.5));
            let signed_one = _mm_or_ps(_mm_set1_ps(1.0), _mm_and_ps(val, sign));
            let q = _mm_cvttps_epi32(_mm_add_ps(truncated, _mm_and_ps(signed_one, ge_half)));
            let q = _mm_blendv_epi8(zero_i, q, _mm_castps_si128(keep));
            unsafe { _mm_storeu_si128(out4.as_mut_ptr().cast::<__m128i>(), q) };
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn round_ties_away_i32x4(value: __m128) -> __m128i {
    let sign = _mm_set1_ps(-0.0);
    let truncated = _mm_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(value);
    let fraction = _mm_sub_ps(value, truncated);
    let at_least_half = _mm_cmpge_ps(_mm_andnot_ps(sign, fraction), _mm_set1_ps(0.5));
    let signed_one = _mm_or_ps(_mm_set1_ps(1.0), _mm_and_ps(value, sign));
    let rounded = _mm_add_ps(truncated, _mm_and_ps(signed_one, at_least_half));
    let quantized = _mm_cvttps_epi32(rounded);
    let positive_overflow = _mm_cmpge_ps(rounded, _mm_set1_ps(2_147_483_648.0));
    let quantized = _mm_blendv_epi8(
        quantized,
        _mm_set1_epi32(i32::MAX),
        _mm_castps_si128(positive_overflow),
    );
    let nan = _mm_cmpunord_ps(value, value);
    _mm_blendv_epi8(quantized, _mm_setzero_si128(), _mm_castps_si128(nan))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_quant_field_x4(source: &mut [u8; 4], value: __m128) {
    let rounded = round_ties_away_i32x4(value);
    let clamped = _mm_min_epi32(
        _mm_max_epi32(rounded, _mm_set1_epi32(1)),
        _mm_set1_epi32(255),
    );
    let nan = _mm_cmpunord_ps(value, value);
    let clamped = _mm_blendv_epi8(clamped, _mm_setzero_si128(), _mm_castps_si128(nan));
    let packed16 = _mm_packus_epi32(clamped, clamped);
    let packed8 = _mm_packus_epi16(packed16, packed16);
    *source = (_mm_cvtsi128_si32(packed8) as u32).to_ne_bytes();
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_quant_field_gain_x4(source: &mut [u8; 4], gain: __m128) {
    let bytes = _mm_cvtsi32_si128(i32::from_ne_bytes(*source));
    let value = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(bytes)), gain);
    store_quant_field_x4(source, value);
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn apply_quant_field_gain_sse41(
    image: &mut crate::image::ImageB,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    gain: f32,
) {
    let gain = _mm_set1_ps(gain);
    for y in y0..y0 + height {
        let values = &mut image.row_mut(y)[x0..x0 + width];
        let (values4, tail) = values.as_chunks_mut::<4>();
        for values in values4 {
            apply_quant_field_gain_x4(values, gain);
        }
        if !tail.is_empty() {
            let mut values = [0u8; 4];
            values[..tail.len()].copy_from_slice(tail);
            apply_quant_field_gain_x4(&mut values, gain);
            tail.copy_from_slice(&values[..tail.len()]);
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_structure_aq_x4(
    corrections: &[f32; 4],
    dest: &mut [u8; 4],
    scale: __m128,
    center: __m128,
) {
    let correction = unsafe { _mm_loadu_ps(corrections.as_ptr()) };
    let delta = _mm_mul_ps(_mm_sub_ps(correction, center), scale);
    let clamped = _mm_max_ps(_mm_min_ps(delta, _mm_set1_ps(0.22)), _mm_set1_ps(-0.18));
    let valid = _mm_cmpord_ps(delta, delta);
    let delta = _mm_blendv_ps(delta, clamped, valid);
    let gain = super::adaptive_quant::fast_exp2_x4(delta);
    let bytes = _mm_cvtsi32_si128(i32::from_ne_bytes(*dest));
    let values = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(bytes));
    store_quant_field_x4(dest, _mm_mul_ps(values, gain));
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn apply_structure_aq_sse41(
    corrections: &[f32],
    field: &mut crate::image::ImageB,
    amount: f32,
    center: f32,
    inv_stddev: f32,
) {
    let values = field.as_mut_slice();
    debug_assert_eq!(corrections.len(), values.len());
    let (corrections4, correction_tail) = corrections.as_chunks::<4>();
    let (values4, value_tail) = values.as_chunks_mut::<4>();
    let scale = _mm_set1_ps(-amount * inv_stddev);
    let center_vector = _mm_set1_ps(center);
    for (corrections, values) in corrections4.iter().zip(values4) {
        apply_structure_aq_x4(corrections, values, scale, center_vector);
    }
    if !correction_tail.is_empty() {
        let mut corrections = [center; 4];
        corrections[..correction_tail.len()].copy_from_slice(correction_tail);
        let mut values = [0u8; 4];
        values[..value_tail.len()].copy_from_slice(value_tail);
        apply_structure_aq_x4(&corrections, &mut values, scale, center_vector);
        value_tail.copy_from_slice(&values[..value_tail.len()]);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i16x4(output: &mut [i16; 4], value: __m128i) {
    let packed = _mm_packs_epi32(value, value);
    unsafe { _mm_storel_epi64(output.as_mut_ptr().cast(), packed) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn quantize_dc_cfl_value_x4(
    input: __m128,
    y_quant: __m128i,
    scale: __m128,
    negative_cfl: __m128,
) -> __m128i {
    let y = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(y_quant));
    let correction = _mm_mul_ps(y, negative_cfl);
    #[cfg(target_feature = "fma")]
    let value = _mm_fmadd_ps(input, scale, correction);
    #[cfg(not(target_feature = "fma"))]
    let value = _mm_add_ps(_mm_mul_ps(input, scale), correction);
    round_ties_away_i32x4(value)
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn quantize_dc_sse41(input: &[f32], scale: f32, output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len());
    let (input4, input_tail) = input.as_chunks::<4>();
    let (output4, output_tail) = output.as_chunks_mut::<4>();
    let scale = _mm_set1_ps(scale);

    for (source, target) in input4.iter().zip(output4) {
        let value = unsafe { _mm_loadu_ps(source.as_ptr()) };
        store_i16x4(target, round_ties_away_i32x4(_mm_mul_ps(value, scale)));
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 4];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let value = unsafe { _mm_loadu_ps(source.as_ptr()) };
        let mut target = [0i16; 4];
        store_i16x4(&mut target, round_ties_away_i32x4(_mm_mul_ps(value, scale)));
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn quantize_dc_cfl_sse41(
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
    let scale = _mm_set1_ps(scale);
    let negative_cfl = _mm_set1_ps(-cfl);

    for ((source, y), target) in input4.iter().zip(y4).zip(output4) {
        let value = unsafe { _mm_loadu_ps(source.as_ptr()) };
        let y = unsafe { _mm_loadl_epi64(y.as_ptr().cast()) };
        store_i16x4(
            target,
            quantize_dc_cfl_value_x4(value, y, scale, negative_cfl),
        );
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 4];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let mut y = [0i16; 4];
        y[..y_tail.len()].copy_from_slice(y_tail);
        let value = unsafe { _mm_loadu_ps(source.as_ptr()) };
        let y = unsafe { _mm_loadl_epi64(y.as_ptr().cast()) };
        let mut target = [0i16; 4];
        store_i16x4(
            &mut target,
            quantize_dc_cfl_value_x4(value, y, scale, negative_cfl),
        );
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}

#[cfg(test)]
mod tests {
    use crate::group::quantize_block_ac_scalar;

    #[test]
    fn quantize_block_ac_sse41_matches_scalar() {
        static EDGES: [f32; 12] = [
            0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 0.75, -0.75, 1.25, -1.25,
        ];
        for &(xsize, ysize) in &[(1usize, 1usize), (2, 2), (4, 4)] {
            let n = xsize * 8 * ysize * 8;
            for c in 0..3 {
                for &distance in &[0.5f32, 2.5] {
                    let qm: Vec<f32> = vec![0.5; n];
                    let block: Vec<f32> = (0..n).map(|i| EDGES[i % EDGES.len()] / 0.5).collect();
                    let mut want = vec![0i32; n];
                    let mut got = vec![0i32; n];
                    quantize_block_ac_scalar(
                        &block, c, &qm, 1, 1.0, 1.0, distance, xsize, ysize, &mut want,
                    );
                    unsafe {
                        super::quantize_block_ac_sse41(
                            &block, c, &qm, 1, 1.0, 1.0, distance, xsize, ysize, &mut got,
                        );
                    }
                    for i in 0..n {
                        assert_eq!(
                            got[i], want[i],
                            "mismatch at {i} ({xsize}x{ysize}, c={c}, d={distance}): \
                             scalar {} sse {}, input {}",
                            want[i], got[i], block[i]
                        );
                    }
                }
            }
        }
    }
}
