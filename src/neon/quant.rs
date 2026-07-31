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
use std::arch::aarch64::*;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn quantize_block_ac_neon(
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
    let qs = vdupq_n_f32(q_scaled);
    let zero = vdupq_n_s32(0);

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
            let qmv = unsafe { vld1q_f32(qm4.as_ptr()) };
            let inv = unsafe { vld1q_f32(in4.as_ptr()) };
            let val = vmulq_f32(vmulq_f32(qmv, qs), inv);
            let keep = vcgeq_f32(vabsq_f32(val), vdupq_n_f32(threshold));
            let q = vcvtaq_s32_f32(val);
            let q = vbslq_s32(keep, q, zero);
            unsafe { vst1q_s32(out4.as_mut_ptr(), q) };
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x4(output: &mut [i16; 4], value: int32x4_t) {
    unsafe { vst1_s16(output.as_mut_ptr(), vqmovn_s32(value)) };
}

#[target_feature(enable = "neon")]
pub(crate) fn quantize_dc_neon(input: &[f32], scale: f32, output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len());
    let (input4, input_tail) = input.as_chunks::<4>();
    let (output4, output_tail) = output.as_chunks_mut::<4>();
    let scale = vdupq_n_f32(scale);

    for (source, target) in input4.iter().zip(output4) {
        let value = unsafe { vld1q_f32(source.as_ptr()) };
        store_i16x4(target, vcvtaq_s32_f32(vmulq_f32(value, scale)));
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 4];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let value = unsafe { vld1q_f32(source.as_ptr()) };
        let mut target = [0i16; 4];
        store_i16x4(&mut target, vcvtaq_s32_f32(vmulq_f32(value, scale)));
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn quantize_dc_cfl_neon(
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
    let negative_cfl = vdupq_n_f32(-cfl);

    for ((source, y), target) in input4.iter().zip(y4).zip(output4) {
        let value = unsafe { vld1q_f32(source.as_ptr()) };
        let y = unsafe { vld1_s16(y.as_ptr()) };
        let correction = vmulq_f32(vcvtq_f32_s32(vmovl_s16(y)), negative_cfl);
        let value = vfmaq_n_f32(correction, value, scale);
        store_i16x4(target, vcvtaq_s32_f32(value));
    }

    if !input_tail.is_empty() {
        let mut source = [0.0; 4];
        source[..input_tail.len()].copy_from_slice(input_tail);
        let mut y = [0i16; 4];
        y[..y_tail.len()].copy_from_slice(y_tail);
        let value = unsafe { vld1q_f32(source.as_ptr()) };
        let y = unsafe { vld1_s16(y.as_ptr()) };
        let correction = vmulq_f32(vcvtq_f32_s32(vmovl_s16(y)), negative_cfl);
        let value = vfmaq_n_f32(correction, value, scale);
        let mut target = [0i16; 4];
        store_i16x4(&mut target, vcvtaq_s32_f32(value));
        output_tail.copy_from_slice(&target[..input_tail.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::quantize_block_ac_neon;
    use crate::enc_group::quantize_block_ac_scalar;

    fn rng_f32(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    #[test]
    fn quantize_block_ac_neon_matches_scalar() {
        let mut state = 0x5eed_1234_u64;
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
                for &distance in &[0.05f32, 0.5, 1.0, 2.5, 6.0] {
                    for &quant in &[1i32, 7, 64, 255] {
                        let block: Vec<f32> = (0..n)
                            .map(|i| {
                                let v = rng_f32(&mut state);
                                // Mix in magnitudes near the deadzone edge.
                                if i % 5 == 0 { v * 0.02 } else { v * 40.0 }
                            })
                            .collect();
                        let qm: Vec<f32> = (0..n)
                            .map(|_| 0.05 + rng_f32(&mut state).abs() * 2.0)
                            .collect();
                        let mut want = vec![0i32; n];
                        let mut got = vec![0i32; n];
                        quantize_block_ac_scalar(
                            &block, c, &qm, quant, 0.7, 1.25, distance, xsize, ysize, &mut want,
                        );
                        unsafe {
                            quantize_block_ac_neon(
                                &block, c, &qm, quant, 0.7, 1.25, distance, xsize, ysize, &mut got,
                            )
                        };
                        assert_eq!(
                            want, got,
                            "mismatch at {xsize}x{ysize} c={c} d={distance} quant={quant}"
                        );
                    }
                }
            }
        }
    }
}
