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

//! Double-precision NEON arithmetic for spline fitting.

use super::{FitMatrix, FitTargets};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn add_scaled_pair(dst: &mut [f64; 2], src: &[f64; 2], scale: float64x2_t) {
    // SAFETY: each array contains one complete vector.
    unsafe {
        let out = vld1q_f64(dst.as_ptr());
        let input = vld1q_f64(src.as_ptr());
        vst1q_f64(dst.as_mut_ptr(), vfmaq_f64(out, input, scale));
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn add_scaled_neon(dst: &mut [f64], src: &[f64], scale: f64) {
    assert_eq!(dst.len(), src.len());
    let gain = vdupq_n_f64(scale);
    let (blocks, dst) = dst.as_chunks_mut::<2>();
    let (inputs, src) = src.as_chunks::<2>();
    let (pairs, blocks) = blocks.as_chunks_mut::<2>();
    let (input_pairs, inputs) = inputs.as_chunks::<2>();
    for ([out0, out1], [input0, input1]) in pairs.iter_mut().zip(input_pairs) {
        add_scaled_pair(out0, input0, gain);
        add_scaled_pair(out1, input1, gain);
    }
    if let ([out], [input]) = (blocks, inputs) {
        add_scaled_pair(out, input, gain);
    }
    if let ([out], &[input]) = (dst, src) {
        *out = input.mul_add(scale, *out);
    }
}

#[target_feature(enable = "neon")]
pub(super) fn accumulate_fit_neon(
    ata: &mut FitMatrix,
    atb: &mut FitTargets,
    btb: &mut [f64; 3],
    row: &[f64],
    target: [f32; 3],
) {
    match row.len() {
        3 => accumulate_fit_fixed_neon(ata, atb, btb, row.first_chunk::<3>().unwrap(), target),
        6 => accumulate_fit_fixed_neon(ata, atb, btb, row.first_chunk::<6>().unwrap(), target),
        _ => {
            let target = accumulate_energy(btb, target);
            accumulate_fit_rows(ata, atb, row, target);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_energy(btb: &mut [f64; 3], target: [f32; 3]) -> [f64; 3] {
    let target = target.map(f64::from);
    // A finite f32 square is exact in f64, so fusion preserves energy rounding.
    // SAFETY: both arrays have at least two lanes.
    unsafe {
        let values = vld1q_f64(target.as_ptr());
        let energy = vld1q_f64(btb.as_ptr());
        vst1q_f64(btb.as_mut_ptr(), vfmaq_f64(energy, values, values));
    }
    btb[2] = target[2].mul_add(target[2], btb[2]);
    target
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn accumulate_fit_fixed_neon<const N: usize>(
    ata: &mut FitMatrix,
    atb: &mut FitTargets,
    btb: &mut [f64; 3],
    row: &[f64; N],
    target: [f32; 3],
) {
    let target = accumulate_energy(btb, target);
    for i in 0..N {
        add_scaled_neon(&mut ata[i][..=i], &row[..=i], row[i]);
    }
    for (out, value) in atb.iter_mut().zip(target) {
        add_scaled_neon(&mut out[..N], row, value);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_fit_rows(ata: &mut FitMatrix, atb: &mut FitTargets, row: &[f64], target: [f64; 3]) {
    for (i, (out, &value)) in ata.iter_mut().zip(row).enumerate() {
        add_scaled_neon(&mut out[..=i], &row[..=i], value);
    }
    for (out, value) in atb.iter_mut().zip(target) {
        add_scaled_neon(&mut out[..row.len()], row, value);
    }
}

#[target_feature(enable = "neon")]
pub(super) fn fit_score_neon(ata: &FitMatrix, rhs: &[f64], sol: &[f64]) -> (f64, f64) {
    assert_eq!(rhs.len(), sol.len());
    let rows = &ata[..sol.len()];
    let (sol_vectors, sol_tail) = sol.as_chunks::<2>();
    let (sol_pairs, sol_vector_tail) = sol_vectors.as_chunks::<2>();
    let (rhs_vectors, rhs_tail) = rhs.as_chunks::<2>();
    let (rhs_pairs, rhs_vector_tail) = rhs_vectors.as_chunks::<2>();

    let zero = vdupq_n_f64(0.0);
    let (mut lin0, mut lin1) = (zero, zero);
    for ([s0, s1], [r0, r1]) in sol_pairs.iter().zip(rhs_pairs) {
        // SAFETY: each array contains two lanes.
        unsafe {
            lin0 = vfmaq_f64(lin0, vld1q_f64(s0.as_ptr()), vld1q_f64(r0.as_ptr()));
            lin1 = vfmaq_f64(lin1, vld1q_f64(s1.as_ptr()), vld1q_f64(r1.as_ptr()));
        }
    }
    if let ([s], [r]) = (sol_vector_tail, rhs_vector_tail) {
        // SAFETY: each array contains two lanes.
        unsafe { lin0 = vfmaq_f64(lin0, vld1q_f64(s.as_ptr()), vld1q_f64(r.as_ptr())) };
    }
    let mut lin = vaddvq_f64(vaddq_f64(lin0, lin1));
    if let (&[s], &[r]) = (sol_tail, rhs_tail) {
        lin = s.mul_add(r, lin);
    }

    let (mut quad0, mut quad1) = (zero, zero);
    let mut quad_tail = 0.0;
    for (&value, row) in sol.iter().zip(rows) {
        let (row_vectors, row_tail) = row[..sol.len()].as_chunks::<2>();
        let (row_pairs, row_vector_tail) = row_vectors.as_chunks::<2>();
        for ([a0, a1], [b0, b1]) in row_pairs.iter().zip(sol_pairs) {
            // SAFETY: each array contains two lanes.
            unsafe {
                // Round value * a before multiplying by b, as in the scalar score.
                let scaled0 = vmulq_n_f64(vld1q_f64(a0.as_ptr()), value);
                let scaled1 = vmulq_n_f64(vld1q_f64(a1.as_ptr()), value);
                quad0 = vfmaq_f64(quad0, scaled0, vld1q_f64(b0.as_ptr()));
                quad1 = vfmaq_f64(quad1, scaled1, vld1q_f64(b1.as_ptr()));
            }
        }
        if let ([a], [b]) = (row_vector_tail, sol_vector_tail) {
            // SAFETY: each array contains two lanes.
            unsafe {
                let scaled = vmulq_n_f64(vld1q_f64(a.as_ptr()), value);
                quad0 = vfmaq_f64(quad0, scaled, vld1q_f64(b.as_ptr()));
            }
        }
        if let (&[a], &[b]) = (row_tail, sol_tail) {
            quad_tail = (value * a).mul_add(b, quad_tail);
        }
    }
    (lin, vaddvq_f64(vaddq_f64(quad0, quad1)) + quad_tail)
}

#[target_feature(enable = "neon")]
pub(super) fn sub_dot_neon(initial: f64, left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    match left.len() {
        0 => return initial,
        1 => return (-left[0]).mul_add(right[0], initial),
        _ => {}
    }

    let (left_blocks, left) = left.as_chunks::<2>();
    let (right_blocks, right) = right.as_chunks::<2>();
    let (left_pairs, left_tail) = left_blocks.as_chunks::<2>();
    let (right_pairs, right_tail) = right_blocks.as_chunks::<2>();
    let zero = vdupq_n_f64(0.0);
    let mut sum0 = vsetq_lane_f64::<0>(initial, zero);
    let mut sum1 = zero;
    for ([left0, left1], [right0, right1]) in left_pairs.iter().zip(right_pairs) {
        // SAFETY: each array contains one complete vector.
        unsafe {
            sum0 = vfmsq_f64(sum0, vld1q_f64(left0.as_ptr()), vld1q_f64(right0.as_ptr()));
            sum1 = vfmsq_f64(sum1, vld1q_f64(left1.as_ptr()), vld1q_f64(right1.as_ptr()));
        }
    }
    if let ([left], [right]) = (left_tail, right_tail) {
        // SAFETY: each array contains one complete vector.
        unsafe {
            sum0 = vfmsq_f64(sum0, vld1q_f64(left.as_ptr()), vld1q_f64(right.as_ptr()));
        }
    }
    let sum = vaddvq_f64(vaddq_f64(sum0, sum1));
    if let (&[left], &[right]) = (left, right) {
        (-left).mul_add(right, sum)
    } else {
        sum
    }
}
