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

use crate::splines::{FitMatrix, FitTargets};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn add_scaled_four(dst: &mut [f64; 4], src: &[f64; 4], scale: __m256d) {
    // SAFETY: each array contains one complete vector.
    unsafe {
        let out = _mm256_loadu_pd(dst.as_ptr());
        let input = _mm256_loadu_pd(src.as_ptr());
        _mm256_storeu_pd(dst.as_mut_ptr(), _mm256_fmadd_pd(input, scale, out));
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn add_scaled_f64_avx2(dst: &mut [f64], src: &[f64], scale: f64) {
    assert_eq!(dst.len(), src.len());
    let scale4 = _mm256_set1_pd(scale);
    let scale2 = _mm256_castpd256_pd128(scale4);
    let (blocks, dst) = dst.as_chunks_mut::<4>();
    let (inputs, src) = src.as_chunks::<4>();
    let (pairs, blocks) = blocks.as_chunks_mut::<2>();
    let (input_pairs, inputs) = inputs.as_chunks::<2>();
    for ([out0, out1], [input0, input1]) in pairs.iter_mut().zip(input_pairs) {
        add_scaled_four(out0, input0, scale4);
        add_scaled_four(out1, input1, scale4);
    }
    if let ([out], [input]) = (blocks, inputs) {
        add_scaled_four(out, input, scale4);
    }
    let (blocks, dst) = dst.as_chunks_mut::<2>();
    let (inputs, src) = src.as_chunks::<2>();
    if let ([out], [input]) = (blocks, inputs) {
        // SAFETY: each array contains one complete vector.
        unsafe {
            let old = _mm_loadu_pd(out.as_ptr());
            let input = _mm_loadu_pd(input.as_ptr());
            _mm_storeu_pd(out.as_mut_ptr(), _mm_fmadd_pd(input, scale2, old));
        }
    }
    if let ([out], &[input]) = (dst, src) {
        *out = _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(input), scale2, _mm_set_sd(*out)));
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn accumulate_fit_avx2(
    ata: &mut FitMatrix,
    atb: &mut FitTargets,
    btb: &mut [f64; 3],
    row: &[f64],
    target: [f32; 3],
) {
    let target = target.map(f64::from);
    // A finite f32 square is exact in f64, so fusion preserves energy rounding.
    // SAFETY: both arrays have at least two lanes.
    unsafe {
        let values = _mm_loadu_pd(target.as_ptr());
        let energy = _mm_loadu_pd(btb.as_ptr());
        _mm_storeu_pd(btb.as_mut_ptr(), _mm_fmadd_pd(values, values, energy));
    }
    btb[2] = target[2].mul_add(target[2], btb[2]);
    match row.len() {
        3 => accumulate_fit_fixed(ata, atb, row.first_chunk::<3>().unwrap(), target),
        6 => accumulate_fit_fixed(ata, atb, row.first_chunk::<6>().unwrap(), target),
        _ => accumulate_fit_rows(ata, atb, row, target),
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn accumulate_fit_fixed<const N: usize>(
    ata: &mut FitMatrix,
    atb: &mut FitTargets,
    row: &[f64; N],
    target: [f64; 3],
) {
    for i in 0..N {
        add_scaled_f64_avx2(&mut ata[i][..=i], &row[..=i], row[i]);
    }
    for (dst, scale) in atb.iter_mut().zip(target) {
        add_scaled_f64_avx2(&mut dst[..N], row, scale);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn accumulate_fit_rows(ata: &mut FitMatrix, atb: &mut FitTargets, row: &[f64], target: [f64; 3]) {
    for (i, (dst, &scale)) in ata.iter_mut().zip(row).enumerate() {
        add_scaled_f64_avx2(&mut dst[..=i], &row[..=i], scale);
    }
    for (dst, scale) in atb.iter_mut().zip(target) {
        add_scaled_f64_avx2(&mut dst[..row.len()], row, scale);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn sum_lanes(value: __m256d) -> f64 {
    let halves = _mm_add_pd(
        _mm256_castpd256_pd128(value),
        _mm256_extractf128_pd::<1>(value),
    );
    _mm_cvtsd_f64(_mm_add_sd(halves, _mm_unpackhi_pd(halves, halves)))
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn fit_score_avx2(ata: &FitMatrix, rhs: &[f64], sol: &[f64]) -> (f64, f64) {
    assert_eq!(rhs.len(), sol.len());
    let rows = &ata[..sol.len()];
    let (sol_blocks, sol_tail) = sol.as_chunks::<4>();
    let (rhs_blocks, rhs_tail) = rhs.as_chunks::<4>();
    let mut lin = _mm256_setzero_pd();
    for (values, rhs) in sol_blocks.iter().zip(rhs_blocks) {
        // SAFETY: each array contains one complete vector.
        unsafe {
            lin = _mm256_fmadd_pd(
                _mm256_loadu_pd(values.as_ptr()),
                _mm256_loadu_pd(rhs.as_ptr()),
                lin,
            );
        }
    }

    let (sol_pairs, sol_block) = sol_blocks.as_chunks::<2>();
    let mut quad0 = _mm256_setzero_pd();
    let mut quad1 = _mm256_setzero_pd();
    let mut quad_tail = 0.0;
    for (&value, row) in sol.iter().zip(rows) {
        let scale = _mm256_set1_pd(value);
        let (row_blocks, row_tail) = row[..sol.len()].as_chunks::<4>();
        let (row_pairs, row_block) = row_blocks.as_chunks::<2>();
        // SAFETY: each array contains one complete vector.
        unsafe {
            for ([a0, a1], [b0, b1]) in row_pairs.iter().zip(sol_pairs) {
                let scaled0 = _mm256_mul_pd(scale, _mm256_loadu_pd(a0.as_ptr()));
                let scaled1 = _mm256_mul_pd(scale, _mm256_loadu_pd(a1.as_ptr()));
                quad0 = _mm256_fmadd_pd(scaled0, _mm256_loadu_pd(b0.as_ptr()), quad0);
                quad1 = _mm256_fmadd_pd(scaled1, _mm256_loadu_pd(b1.as_ptr()), quad1);
            }
            if let ([a], [b]) = (row_block, sol_block) {
                let scaled = _mm256_mul_pd(scale, _mm256_loadu_pd(a.as_ptr()));
                quad0 = _mm256_fmadd_pd(scaled, _mm256_loadu_pd(b.as_ptr()), quad0);
            }
        }
        for (&a, &b) in row_tail.iter().zip(sol_tail) {
            quad_tail = (value * a).mul_add(b, quad_tail);
        }
    }
    let mut lin = sum_lanes(lin);
    for (&value, &rhs) in sol_tail.iter().zip(rhs_tail) {
        lin = value.mul_add(rhs, lin);
    }
    (lin, sum_lanes(_mm256_add_pd(quad0, quad1)) + quad_tail)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn sub_dot_avx2(initial: f64, left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    let (left_blocks, left) = left.as_chunks::<4>();
    let (right_blocks, right) = right.as_chunks::<4>();
    let mut result = _mm_set_sd(initial);
    // SAFETY: each load reads a fixed-size array that holds all its lanes.
    unsafe {
        if let Some(((left_head, left_blocks), (right_head, right_blocks))) =
            left_blocks.split_first().zip(right_blocks.split_first())
        {
            let mut sum0 = _mm256_mul_pd(
                _mm256_loadu_pd(left_head.as_ptr()),
                _mm256_loadu_pd(right_head.as_ptr()),
            );
            let mut sum1 = _mm256_setzero_pd();
            let (left_pairs, left_tail) = left_blocks.as_chunks::<2>();
            let (right_pairs, right_tail) = right_blocks.as_chunks::<2>();
            for ([left0, left1], [right0, right1]) in left_pairs.iter().zip(right_pairs) {
                sum1 = _mm256_fmadd_pd(
                    _mm256_loadu_pd(left0.as_ptr()),
                    _mm256_loadu_pd(right0.as_ptr()),
                    sum1,
                );
                sum0 = _mm256_fmadd_pd(
                    _mm256_loadu_pd(left1.as_ptr()),
                    _mm256_loadu_pd(right1.as_ptr()),
                    sum0,
                );
            }
            if let ([left], [right]) = (left_tail, right_tail) {
                sum1 = _mm256_fmadd_pd(
                    _mm256_loadu_pd(left.as_ptr()),
                    _mm256_loadu_pd(right.as_ptr()),
                    sum1,
                );
            }
            let sum = sum_lanes(_mm256_add_pd(sum0, sum1));
            result = _mm_sub_sd(result, _mm_set_sd(sum));
        }
        let (left_blocks, left) = left.as_chunks::<2>();
        let (right_blocks, right) = right.as_chunks::<2>();
        for (left, right) in left_blocks.iter().zip(right_blocks) {
            let pair = _mm_fnmadd_pd(
                _mm_loadu_pd(left.as_ptr()),
                _mm_loadu_pd(right.as_ptr()),
                result,
            );
            result = _mm_add_sd(pair, _mm_unpackhi_pd(pair, pair));
        }
        if let (&[left], &[right]) = (left, right) {
            result = _mm_fnmadd_sd(_mm_set_sd(left), _mm_set_sd(right), result);
        }
    }
    _mm_cvtsd_f64(result)
}
