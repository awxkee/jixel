/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

use super::ac_strategy::{hsum, sse_dequantized_level_f32, sse_log2p1_f32};
use crate::color_correlation::{CFL_MAGNITUDE_BITS, CFL_NONZERO_BITS, cfl_rd_cost_scalar};

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_rdo_refill_sse41(
    m_x: &mut [f32],
    s_x: &mut [f32],
    m_b: &mut [f32],
    s_b: &mut [f32],
    y: &[f32],
    x: &[f32],
    b: &[f32],
    qm_x: &[f32],
    qm_b: &[f32],
    params: crate::color_correlation::CflRdoRefillParams,
) {
    let len = m_x.len();
    assert_eq!(s_x.len(), len);
    assert_eq!(m_b.len(), len);
    assert_eq!(s_b.len(), len);
    assert_eq!(y.len(), len);
    assert_eq!(x.len(), len);
    assert_eq!(b.len(), len);
    assert_eq!(qm_x.len(), len);
    assert_eq!(qm_b.len(), len);
    let q_block = _mm_set1_ps(params.q_block);
    let x_mul = _mm_set1_ps(params.x_mul);
    let b_mul = _mm_set1_ps(params.b_mul);
    let shift_x = _mm_set1_ps(params.shift_x);
    let shift_b = _mm_set1_ps(params.shift_b);
    let mut i = 0;
    while i + 4 <= len {
        let yv = unsafe { _mm_loadu_ps(y.as_ptr().add(i)) };
        let xv = unsafe { _mm_loadu_ps(x.as_ptr().add(i)) };
        let bv = unsafe { _mm_loadu_ps(b.as_ptr().add(i)) };
        let qx = unsafe { _mm_loadu_ps(qm_x.as_ptr().add(i)) };
        let qb = unsafe { _mm_loadu_ps(qm_b.as_ptr().add(i)) };
        // Keep the two quantizer multiplications in the scalar order.
        let qx = _mm_mul_ps(_mm_mul_ps(qx, q_block), x_mul);
        let qb = _mm_mul_ps(_mm_mul_ps(qb, q_block), b_mul);
        let sx = _mm_add_ps(_mm_mul_ps(shift_x, yv), xv);
        let sb = _mm_add_ps(_mm_mul_ps(shift_b, yv), bv);
        unsafe {
            _mm_storeu_ps(m_x.as_mut_ptr().add(i), _mm_mul_ps(yv, qx));
            _mm_storeu_ps(s_x.as_mut_ptr().add(i), _mm_mul_ps(sx, qx));
            _mm_storeu_ps(m_b.as_mut_ptr().add(i), _mm_mul_ps(yv, qb));
            _mm_storeu_ps(s_b.as_mut_ptr().add(i), _mm_mul_ps(sb, qb));
        }
        i += 4;
    }
    let src = y[i..]
        .iter()
        .zip(&x[i..])
        .zip(&b[i..])
        .zip(qm_x[i..].iter().zip(&qm_b[i..]));
    let dst = m_x[i..]
        .iter_mut()
        .zip(&mut s_x[i..])
        .zip(m_b[i..].iter_mut().zip(&mut s_b[i..]));
    for ((((&y, &x), &b), (&qx, &qb)), ((m_x, s_x), (m_b, s_b))) in src.zip(dst) {
        let qx = qx * params.q_block * params.x_mul;
        let qb = qb * params.q_block * params.b_mul;
        *m_x = y * qx;
        *s_x = (params.shift_x * y + x) * qx;
        *m_b = y * qb;
        *s_b = (params.shift_b * y + b) * qb;
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_rd_cost_sse41(
    m: &[f32],
    s: &[f32],
    factor: f32,
    weight: f32,
    lambda: f32,
    bias: crate::group::DequantBias,
) -> f32 {
    let len = m.len().min(s.len());
    let (m_chunks, m_tail) = m[..len].as_chunks::<4>();
    let (s_chunks, s_tail) = s[..len].as_chunks::<4>();
    let factor_v = _mm_set1_ps(factor);
    let weight_v = _mm_set1_ps(weight);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let half = _mm_set1_ps(0.5);
    let sign = _mm_set1_ps(-0.0);
    let nonzero_bits = _mm_set1_ps(lambda * CFL_NONZERO_BITS);
    let magnitude_bits = _mm_set1_ps(lambda * CFL_MAGNITUDE_BITS);
    let mut acc = zero;

    for (m, s) in m_chunks.iter().zip(s_chunks) {
        let mv = unsafe { _mm_loadu_ps(m.as_ptr()) };
        let sv = unsafe { _mm_loadu_ps(s.as_ptr()) };
        let residual = _mm_sub_ps(sv, _mm_mul_ps(factor_v, mv));
        let zero_cost = _mm_mul_ps(_mm_mul_ps(weight_v, residual), residual);

        // Match f32::round (ties away from zero), including values just below
        // half-integers that adding 0.5 before truncation would round up.
        let truncated = _mm_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(residual);
        let fraction = _mm_andnot_ps(sign, _mm_sub_ps(residual, truncated));
        let at_least_half = _mm_cmpge_ps(fraction, half);
        let signed_one = _mm_or_ps(one, _mm_and_ps(sign, residual));
        let level = _mm_add_ps(truncated, _mm_and_ps(signed_one, at_least_half));

        let coded_err = _mm_sub_ps(residual, sse_dequantized_level_f32(level, bias));
        let abs_level = _mm_andnot_ps(sign, level);
        // Clamp zero levels before taking log2; their coded cost is masked out.
        let log2_level = sse_log2p1_f32(_mm_max_ps(_mm_sub_ps(abs_level, one), zero));
        let bits = _mm_add_ps(nonzero_bits, _mm_mul_ps(magnitude_bits, log2_level));
        let coded = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(weight_v, coded_err), coded_err), bits);
        let nonzero = _mm_cmpgt_ps(abs_level, zero);
        let best = _mm_min_ps(zero_cost, _mm_blendv_ps(zero_cost, coded, nonzero));
        acc = _mm_add_ps(acc, best);
    }

    hsum(acc) + cfl_rd_cost_scalar(m_tail, s_tail, factor, weight, lambda, bias)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn cost_term(
    mv: __m128,
    sv: __m128,
    factor: f32,
    weight: f32,
    lambda: f32,
    bias: crate::group::DequantBias,
) -> __m128 {
    let factor_v = _mm_set1_ps(factor);
    let weight_v = _mm_set1_ps(weight);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let half = _mm_set1_ps(0.5);
    let sign = _mm_set1_ps(-0.0);
    let nonzero_bits = _mm_set1_ps(lambda * CFL_NONZERO_BITS);
    let magnitude_bits = _mm_set1_ps(lambda * CFL_MAGNITUDE_BITS);
    let residual = _mm_sub_ps(sv, _mm_mul_ps(factor_v, mv));
    let zero_cost = _mm_mul_ps(_mm_mul_ps(weight_v, residual), residual);

    // Match f32::round (ties away from zero), including values just below
    // half-integers that adding 0.5 before truncation would round up.
    let truncated = _mm_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(residual);
    let fraction = _mm_andnot_ps(sign, _mm_sub_ps(residual, truncated));
    let at_least_half = _mm_cmpge_ps(fraction, half);
    let signed_one = _mm_or_ps(one, _mm_and_ps(sign, residual));
    let level = _mm_add_ps(truncated, _mm_and_ps(signed_one, at_least_half));

    let coded_err = _mm_sub_ps(residual, sse_dequantized_level_f32(level, bias));
    let abs_level = _mm_andnot_ps(sign, level);
    // Clamp zero levels before taking log2; their coded cost is masked out.
    let log2_level = sse_log2p1_f32(_mm_max_ps(_mm_sub_ps(abs_level, one), zero));
    let bits = _mm_add_ps(nonzero_bits, _mm_mul_ps(magnitude_bits, log2_level));
    let coded = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(weight_v, coded_err), coded_err), bits);
    let nonzero = _mm_cmpgt_ps(abs_level, zero);
    _mm_min_ps(zero_cost, _mm_blendv_ps(zero_cost, coded, nonzero))
}
#[target_feature(enable = "sse4.1")]
fn cost_load(v: &[f32; 4]) -> __m128 {
    unsafe { _mm_loadu_ps(v.as_ptr()) }
}
crate::color_correlation::rd_costs::batched_costs!(
    "sse4.1",
    4,
    cfl_rd_costs_sse41,
    cfl_rd_cost_sse41,
    cost_term,
    cost_load,
    _mm_set1_ps,
    _mm_add_ps,
    hsum
);
