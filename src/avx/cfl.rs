/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

#[inline]
#[target_feature(enable = "avx2")]
fn round_ties_away_x8(v: __m256) -> __m256 {
    let sign = _mm256_set1_ps(-0.0);
    let abs = _mm256_andnot_ps(sign, v);
    let rounded_abs = _mm256_floor_ps(_mm256_add_ps(abs, _mm256_set1_ps(0.5)));
    _mm256_or_ps(_mm256_and_ps(sign, v), rounded_abs)
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequantized_level_x8(q: __m256) -> __m256 {
    let sign_mask = _mm256_set1_ps(-0.0);
    let absq = _mm256_andnot_ps(sign_mask, q);
    let big = _mm256_sub_ps(
        q,
        _mm256_div_ps(_mm256_set1_ps(crate::group::DEFAULT_QUANT_BIAS_3), q),
    );
    let one = _mm256_or_ps(
        _mm256_and_ps(sign_mask, q),
        _mm256_set1_ps(crate::group::DEFAULT_QUANT_BIAS_1),
    );
    let use_big = _mm256_cmp_ps::<_CMP_GE_OQ>(absq, _mm256_set1_ps(1.125));
    let dq = _mm256_blendv_ps(one, big, use_big);
    let nonzero = _mm256_cmp_ps::<_CMP_GT_OQ>(absq, _mm256_setzero_ps());
    _mm256_and_ps(dq, nonzero)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn closed_loop_cost_x8(
    mv: __m256,
    sv: __m256,
    factor: __m256,
    threshold: __m256,
) -> (__m256, __m256) {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let sign = _mm256_set1_ps(-0.0);
    let residual = _mm256_sub_ps(sv, _mm256_mul_ps(factor, mv));
    let abs_residual = _mm256_andnot_ps(sign, residual);
    let active = _mm256_cmp_ps::<_CMP_GE_OQ>(abs_residual, threshold);
    let level = _mm256_and_ps(active, round_ties_away_x8(residual));
    let reconstructed = dequantized_level_x8(level);
    let quant_error = _mm256_andnot_ps(sign, _mm256_sub_ps(residual, reconstructed));
    let dist = _mm256_add_ps(
        quant_error,
        _mm256_mul_ps(_mm256_set1_ps(0.15), abs_residual),
    );
    let abs_level = _mm256_andnot_ps(sign, level);
    let nonzero = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_level, zero);
    let bits = _mm256_and_ps(
        nonzero,
        _mm256_add_ps(one, super::ac_strategy::avx2_log2p1_f32(abs_level)),
    );
    (dist, bits)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn cfl_closed_loop_cost_avx2(
    m: &[f32],
    s: &[f32],
    factor: f32,
    thresholds: &[f32; 63],
) -> [f32; 2] {
    let len = m.len().min(s.len());
    let factor = _mm256_set1_ps(factor);
    let mut distortion = 0.0f32;
    let mut coeff_bits = 0.0f32;
    let mut dist_acc = _mm256_setzero_ps();
    let mut bits_acc = _mm256_setzero_ps();
    let (m_chunks, m_tail) = m[..len].as_chunks::<8>();
    let (s_chunks, s_tail) = s[..len].as_chunks::<8>();

    for (chunk_index, (m, s)) in m_chunks.iter().zip(s_chunks).enumerate() {
        let mv = unsafe { _mm256_loadu_ps(m.as_ptr()) };
        let sv = unsafe { _mm256_loadu_ps(s.as_ptr()) };
        let phase = (chunk_index * 8) % 63;
        let threshold = if phase + 8 <= 63 {
            unsafe { _mm256_loadu_ps(thresholds.as_ptr().add(phase)) }
        } else {
            let wrapped = [
                thresholds[phase],
                thresholds[(phase + 1) % 63],
                thresholds[(phase + 2) % 63],
                thresholds[(phase + 3) % 63],
                thresholds[(phase + 4) % 63],
                thresholds[(phase + 5) % 63],
                thresholds[(phase + 6) % 63],
                thresholds[(phase + 7) % 63],
            ];
            unsafe { _mm256_loadu_ps(wrapped.as_ptr()) }
        };
        let (dist, bits) = closed_loop_cost_x8(mv, sv, factor, threshold);
        dist_acc = _mm256_add_ps(dist_acc, dist);
        bits_acc = _mm256_add_ps(bits_acc, bits);
    }

    if !m_tail.is_empty() {
        let tail_len = m_tail.len() as i32;
        let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(tail_len), lanes);
        let mv = unsafe { _mm256_maskload_ps(m_tail.as_ptr(), mask) };
        let sv = unsafe { _mm256_maskload_ps(s_tail.as_ptr(), mask) };
        let tail_start = m_chunks.len() * 8;
        let tail_thresholds: [f32; 8] =
            std::array::from_fn(|lane| thresholds[(tail_start + lane) % 63]);
        let threshold = unsafe { _mm256_loadu_ps(tail_thresholds.as_ptr()) };
        let (dist, bits) = closed_loop_cost_x8(mv, sv, factor, threshold);
        dist_acc = _mm256_add_ps(dist_acc, dist);
        bits_acc = _mm256_add_ps(bits_acc, bits);
    }

    distortion += super::ac_strategy::hsum256(dist_acc);
    coeff_bits += super::ac_strategy::hsum256(bits_acc);
    [distortion, coeff_bits]
}

#[inline]
#[target_feature(enable = "avx2")]
fn reduce_transposed_4x8(a: __m256, b: __m256, c: __m256, d: __m256) -> [f32; 4] {
    // Transpose the accumulators so each four-lane group contains
    // [ca_x, cb_x, ca_b, cb_b] for one input lane.
    let ab_lo = _mm256_unpacklo_ps(a, b);
    let ab_hi = _mm256_unpackhi_ps(a, b);
    let cd_lo = _mm256_unpacklo_ps(c, d);
    let cd_hi = _mm256_unpackhi_ps(c, d);
    let lane_0_4 = _mm256_shuffle_ps::<0x44>(ab_lo, cd_lo);
    let lane_1_5 = _mm256_shuffle_ps::<0xee>(ab_lo, cd_lo);
    let lane_2_6 = _mm256_shuffle_ps::<0x44>(ab_hi, cd_hi);
    let lane_3_7 = _mm256_shuffle_ps::<0xee>(ab_hi, cd_hi);

    let sums_0_3_and_4_7 = _mm256_add_ps(
        _mm256_add_ps(lane_0_4, lane_1_5),
        _mm256_add_ps(lane_2_6, lane_3_7),
    );
    let low = _mm256_castps256_ps128(sums_0_3_and_4_7);
    let high = _mm256_extractf128_ps::<1>(sums_0_3_and_4_7);
    let sums = _mm_add_ps(low, high);
    let mut out = [0.0f32; 4];
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), sums) };
    out
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn cfl_regression_avx2(
    y: &[f32; 64],
    x: &[f32; 64],
    b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
) -> [f32; 4] {
    let inv_color = _mm256_set1_ps(1.0 / 84.0);
    let mut ca_x = _mm256_setzero_ps();
    let mut cb_x = _mm256_setzero_ps();
    let mut ca_b = _mm256_setzero_ps();
    let mut cb_b = _mm256_setzero_ps();

    let y_chunks = y.as_chunks::<8>().0;
    let x_chunks = x.as_chunks::<8>().0;
    let b_chunks = b.as_chunks::<8>().0;
    let qm_x_chunks = qm_x.as_chunks::<8>().0;
    let qm_b_chunks = qm_b.as_chunks::<8>().0;
    for ((((y, x), b), qm_x), qm_b) in y_chunks
        .iter()
        .zip(x_chunks)
        .zip(b_chunks)
        .zip(qm_x_chunks)
        .zip(qm_b_chunks)
    {
        let yv = unsafe { _mm256_loadu_ps(y.as_ptr()) };
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr()) };
        let bv = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        let qx = unsafe { _mm256_loadu_ps(qm_x.as_ptr()) };
        let qb = unsafe { _mm256_loadu_ps(qm_b.as_ptr()) };

        let mx = _mm256_mul_ps(yv, qx);
        let sx = _mm256_mul_ps(xv, qx);
        let ax = _mm256_mul_ps(inv_color, mx);
        ca_x = _mm256_fmadd_ps(ax, ax, ca_x);
        cb_x = _mm256_fnmadd_ps(ax, sx, cb_x);

        let mb = _mm256_mul_ps(yv, qb);
        let sb = _mm256_mul_ps(bv, qb);
        let ab = _mm256_mul_ps(inv_color, mb);
        let residual_b = _mm256_sub_ps(mb, sb);
        ca_b = _mm256_fmadd_ps(ab, ab, ca_b);
        cb_b = _mm256_fmadd_ps(ab, residual_b, cb_b);
    }

    reduce_transposed_4x8(ca_x, cb_x, ca_b, cb_b)
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_rdo_block_avx2(
    m_x: &mut [f32; 63],
    s_x: &mut [f32; 63],
    m_b: &mut [f32; 63],
    s_b: &mut [f32; 63],
    block_y: &[f32; 64],
    block_x: &[f32; 64],
    block_b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
    q_block: f32,
) {
    let q_block_v = _mm256_set1_ps(q_block);
    for i in (0..56).step_by(8) {
        let coeff = i + 1;
        let y = unsafe { _mm256_loadu_ps(block_y.as_ptr().add(coeff)) };
        let x = unsafe { _mm256_loadu_ps(block_x.as_ptr().add(coeff)) };
        let b = unsafe { _mm256_loadu_ps(block_b.as_ptr().add(coeff)) };
        let qx = unsafe { _mm256_loadu_ps(qm_x.as_ptr().add(coeff)) };
        let qb = unsafe { _mm256_loadu_ps(qm_b.as_ptr().add(coeff)) };
        let qx = _mm256_mul_ps(qx, q_block_v);
        let qb = _mm256_mul_ps(qb, q_block_v);
        unsafe {
            _mm256_storeu_ps(m_x.as_mut_ptr().add(i), _mm256_mul_ps(y, qx));
            _mm256_storeu_ps(s_x.as_mut_ptr().add(i), _mm256_mul_ps(x, qx));
            _mm256_storeu_ps(m_b.as_mut_ptr().add(i), _mm256_mul_ps(y, qb));
            _mm256_storeu_ps(s_b.as_mut_ptr().add(i), _mm256_mul_ps(b, qb));
        }
    }

    // Seven AC coefficients remain. Mask off lane 7 so neither the loads nor
    // stores cross the fixed-size input/output arrays.
    let i = 56;
    let coeff = i + 1;
    let mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0);
    let y = unsafe { _mm256_maskload_ps(block_y.as_ptr().add(coeff), mask) };
    let x = unsafe { _mm256_maskload_ps(block_x.as_ptr().add(coeff), mask) };
    let b = unsafe { _mm256_maskload_ps(block_b.as_ptr().add(coeff), mask) };
    let qx = unsafe { _mm256_maskload_ps(qm_x.as_ptr().add(coeff), mask) };
    let qb = unsafe { _mm256_maskload_ps(qm_b.as_ptr().add(coeff), mask) };
    let qx = _mm256_mul_ps(qx, q_block_v);
    let qb = _mm256_mul_ps(qb, q_block_v);
    unsafe {
        _mm256_maskstore_ps(m_x.as_mut_ptr().add(i), mask, _mm256_mul_ps(y, qx));
        _mm256_maskstore_ps(s_x.as_mut_ptr().add(i), mask, _mm256_mul_ps(x, qx));
        _mm256_maskstore_ps(m_b.as_mut_ptr().add(i), mask, _mm256_mul_ps(y, qb));
        _mm256_maskstore_ps(s_b.as_mut_ptr().add(i), mask, _mm256_mul_ps(b, qb));
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn cfl_rdo_stats_avx2(m: &[f32], s: &[f32]) -> [f32; 4] {
    let len = m.len().min(s.len());
    let (m_chunks, m_tail) = m[..len].as_chunks::<8>();
    let (s_chunks, s_tail) = s[..len].as_chunks::<8>();
    let mut dot_ms = _mm256_setzero_ps();
    let mut dot_mm = _mm256_setzero_ps();
    let mut dot_ss = _mm256_setzero_ps();
    let mut sum_abs_s = _mm256_setzero_ps();
    let sign = _mm256_set1_ps(-0.0);

    for (m, s) in m_chunks.iter().zip(s_chunks) {
        let mv = unsafe { _mm256_loadu_ps(m.as_ptr()) };
        let sv = unsafe { _mm256_loadu_ps(s.as_ptr()) };
        dot_ms = _mm256_fmadd_ps(mv, sv, dot_ms);
        dot_mm = _mm256_fmadd_ps(mv, mv, dot_mm);
        dot_ss = _mm256_fmadd_ps(sv, sv, dot_ss);
        sum_abs_s = _mm256_add_ps(sum_abs_s, _mm256_andnot_ps(sign, sv));
    }

    let tail_len = m_tail.len();
    if tail_len != 0 {
        let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(tail_len as i32), lanes);
        let mv = unsafe { _mm256_maskload_ps(m_tail.as_ptr(), mask) };
        let sv = unsafe { _mm256_maskload_ps(s_tail.as_ptr(), mask) };
        dot_ms = _mm256_fmadd_ps(mv, sv, dot_ms);
        dot_mm = _mm256_fmadd_ps(mv, mv, dot_mm);
        dot_ss = _mm256_fmadd_ps(sv, sv, dot_ss);
        sum_abs_s = _mm256_add_ps(sum_abs_s, _mm256_andnot_ps(sign, sv));
    }

    reduce_transposed_4x8(dot_ms, dot_mm, dot_ss, sum_abs_s)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn apply_cfl_avx2(x: &mut [f32], y: &[f32], b: &mut [f32], cmap_factor: [f32; 3]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), b.len());

    let cx = _mm256_set1_ps(-cmap_factor[0]);
    let cb = _mm256_set1_ps(-cmap_factor[2]);
    let (x_chunks, x_tail) = x.as_chunks_mut::<8>();
    let (y_chunks, y_tail) = y.as_chunks::<8>();
    let (b_chunks, b_tail) = b.as_chunks_mut::<8>();
    let (x_groups, x_chunks_tail) = x_chunks.as_chunks_mut::<4>();
    let (y_groups, y_chunks_tail) = y_chunks.as_chunks::<4>();
    let (b_groups, b_chunks_tail) = b_chunks.as_chunks_mut::<4>();

    for ((x, y), b) in x_groups.iter_mut().zip(y_groups).zip(b_groups) {
        let y0 = unsafe { _mm256_loadu_ps(y[0].as_ptr()) };
        let y1 = unsafe { _mm256_loadu_ps(y[1].as_ptr()) };
        let y2 = unsafe { _mm256_loadu_ps(y[2].as_ptr()) };
        let y3 = unsafe { _mm256_loadu_ps(y[3].as_ptr()) };
        let x0 = unsafe { _mm256_loadu_ps(x[0].as_ptr()) };
        let x1 = unsafe { _mm256_loadu_ps(x[1].as_ptr()) };
        let x2 = unsafe { _mm256_loadu_ps(x[2].as_ptr()) };
        let x3 = unsafe { _mm256_loadu_ps(x[3].as_ptr()) };
        let b0 = unsafe { _mm256_loadu_ps(b[0].as_ptr()) };
        let b1 = unsafe { _mm256_loadu_ps(b[1].as_ptr()) };
        let b2 = unsafe { _mm256_loadu_ps(b[2].as_ptr()) };
        let b3 = unsafe { _mm256_loadu_ps(b[3].as_ptr()) };

        let x0 = _mm256_fmadd_ps(cx, y0, x0);
        let x1 = _mm256_fmadd_ps(cx, y1, x1);
        let x2 = _mm256_fmadd_ps(cx, y2, x2);
        let x3 = _mm256_fmadd_ps(cx, y3, x3);
        let b0 = _mm256_fmadd_ps(cb, y0, b0);
        let b1 = _mm256_fmadd_ps(cb, y1, b1);
        let b2 = _mm256_fmadd_ps(cb, y2, b2);
        let b3 = _mm256_fmadd_ps(cb, y3, b3);

        unsafe {
            _mm256_storeu_ps(x[0].as_mut_ptr(), x0);
            _mm256_storeu_ps(x[1].as_mut_ptr(), x1);
            _mm256_storeu_ps(x[2].as_mut_ptr(), x2);
            _mm256_storeu_ps(x[3].as_mut_ptr(), x3);
            _mm256_storeu_ps(b[0].as_mut_ptr(), b0);
            _mm256_storeu_ps(b[1].as_mut_ptr(), b1);
            _mm256_storeu_ps(b[2].as_mut_ptr(), b2);
            _mm256_storeu_ps(b[3].as_mut_ptr(), b3);
        }
    }

    for ((x, y), b) in x_chunks_tail
        .iter_mut()
        .zip(y_chunks_tail)
        .zip(b_chunks_tail)
    {
        let yv = unsafe { _mm256_loadu_ps(y.as_ptr()) };
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr()) };
        let bv = unsafe { _mm256_loadu_ps(b.as_ptr()) };
        unsafe {
            _mm256_storeu_ps(x.as_mut_ptr(), _mm256_fmadd_ps(cx, yv, xv));
            _mm256_storeu_ps(b.as_mut_ptr(), _mm256_fmadd_ps(cb, yv, bv));
        }
    }

    for ((x, &y), b) in x_tail.iter_mut().zip(y_tail).zip(b_tail) {
        *x = (-cmap_factor[0]).mul_add(y, *x);
        *b = (-cmap_factor[2]).mul_add(y, *b);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn apply_cfl_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        for size in [0, 1, 7, 8, 15, 32, 64, 1024, 4096] {
            let y: Vec<f32> = (0..size)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03125)
                .collect();
            let mut x: Vec<f32> = (0..size).map(|i| i as f32 * 0.015625 - 2.0).collect();
            let mut b: Vec<f32> = (0..size).map(|i| 3.0 - i as f32 * 0.0078125).collect();
            let mut want_x = x.clone();
            let mut want_b = b.clone();
            let factors: [f32; 3] = [0.3125, 0.0, -0.1875];
            for i in 0..size {
                want_x[i] = (-factors[0]).mul_add(y[i], want_x[i]);
                want_b[i] = (-factors[2]).mul_add(y[i], want_b[i]);
            }
            unsafe { super::apply_cfl_avx2(&mut x, &y, &mut b, factors) };
            assert_eq!(x, want_x, "X mismatch at size {size}");
            assert_eq!(b, want_b, "B mismatch at size {size}");
        }
    }
}
