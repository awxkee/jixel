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

use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn cfl_regression_neon(
    y: &[f32; 64],
    x: &[f32; 64],
    b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
) -> [f32; 4] {
    let inv_color = vdupq_n_f32(1.0 / 84.0);
    let mut ca_x = vdupq_n_f32(0.0);
    let mut cb_x = vdupq_n_f32(0.0);
    let mut ca_b = vdupq_n_f32(0.0);
    let mut cb_b = vdupq_n_f32(0.0);

    let y_chunks = y.as_chunks::<4>().0;
    let x_chunks = x.as_chunks::<4>().0;
    let b_chunks = b.as_chunks::<4>().0;
    let qm_x_chunks = qm_x.as_chunks::<4>().0;
    let qm_b_chunks = qm_b.as_chunks::<4>().0;
    for ((((y, x), b), qm_x), qm_b) in y_chunks
        .iter()
        .zip(x_chunks)
        .zip(b_chunks)
        .zip(qm_x_chunks)
        .zip(qm_b_chunks)
    {
        let yv = unsafe { vld1q_f32(y.as_ptr()) };
        let xv = unsafe { vld1q_f32(x.as_ptr()) };
        let bv = unsafe { vld1q_f32(b.as_ptr()) };
        let qx = unsafe { vld1q_f32(qm_x.as_ptr()) };
        let qb = unsafe { vld1q_f32(qm_b.as_ptr()) };

        let mx = vmulq_f32(yv, qx);
        let sx = vmulq_f32(xv, qx);
        let ax = vmulq_f32(inv_color, mx);
        ca_x = vfmaq_f32(ca_x, ax, ax);
        cb_x = vfmsq_f32(cb_x, ax, sx);

        let mb = vmulq_f32(yv, qb);
        let sb = vmulq_f32(bv, qb);
        let ab = vmulq_f32(inv_color, mb);
        let residual_b = vsubq_f32(mb, sb);
        ca_b = vfmaq_f32(ca_b, ab, ab);
        cb_b = vfmaq_f32(cb_b, ab, residual_b);
    }

    [
        vaddvq_f32(ca_x),
        vaddvq_f32(cb_x),
        vaddvq_f32(ca_b),
        vaddvq_f32(cb_b),
    ]
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_rdo_block_neon(
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
    let q_block_v = vdupq_n_f32(q_block);
    for i in (0..60).step_by(4) {
        let coeff = i + 1;
        let y = unsafe { vld1q_f32(block_y.as_ptr().add(coeff)) };
        let x = unsafe { vld1q_f32(block_x.as_ptr().add(coeff)) };
        let b = unsafe { vld1q_f32(block_b.as_ptr().add(coeff)) };
        let qx = unsafe { vld1q_f32(qm_x.as_ptr().add(coeff)) };
        let qb = unsafe { vld1q_f32(qm_b.as_ptr().add(coeff)) };
        let qx = vmulq_f32(qx, q_block_v);
        let qb = vmulq_f32(qb, q_block_v);
        unsafe {
            vst1q_f32(m_x.as_mut_ptr().add(i), vmulq_f32(y, qx));
            vst1q_f32(s_x.as_mut_ptr().add(i), vmulq_f32(x, qx));
            vst1q_f32(m_b.as_mut_ptr().add(i), vmulq_f32(y, qb));
            vst1q_f32(s_b.as_mut_ptr().add(i), vmulq_f32(b, qb));
        }
    }

    // NEON has no masked load/store. Recompute output 59 and use the other
    // three lanes for the tail, keeping every access within the arrays.
    let i = 59;
    let coeff = i + 1;
    let y = unsafe { vld1q_f32(block_y.as_ptr().add(coeff)) };
    let x = unsafe { vld1q_f32(block_x.as_ptr().add(coeff)) };
    let b = unsafe { vld1q_f32(block_b.as_ptr().add(coeff)) };
    let qx = unsafe { vld1q_f32(qm_x.as_ptr().add(coeff)) };
    let qb = unsafe { vld1q_f32(qm_b.as_ptr().add(coeff)) };
    let qx = vmulq_f32(qx, q_block_v);
    let qb = vmulq_f32(qb, q_block_v);
    unsafe {
        vst1q_f32(m_x.as_mut_ptr().add(i), vmulq_f32(y, qx));
        vst1q_f32(s_x.as_mut_ptr().add(i), vmulq_f32(x, qx));
        vst1q_f32(m_b.as_mut_ptr().add(i), vmulq_f32(y, qb));
        vst1q_f32(s_b.as_mut_ptr().add(i), vmulq_f32(b, qb));
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_rdo_stats_neon(m: &[f32], s: &[f32]) -> [f32; 4] {
    let len = m.len().min(s.len());
    let (m_chunks, m_tail) = m[..len].as_chunks::<4>();
    let (s_chunks, s_tail) = s[..len].as_chunks::<4>();
    let mut dot_ms = vdupq_n_f32(0.0);
    let mut dot_mm = vdupq_n_f32(0.0);
    let mut dot_ss = vdupq_n_f32(0.0);
    let mut sum_abs_s = vdupq_n_f32(0.0);

    for (m, s) in m_chunks.iter().zip(s_chunks) {
        let mv = unsafe { vld1q_f32(m.as_ptr()) };
        let sv = unsafe { vld1q_f32(s.as_ptr()) };
        dot_ms = vfmaq_f32(dot_ms, mv, sv);
        dot_mm = vfmaq_f32(dot_mm, mv, mv);
        dot_ss = vfmaq_f32(dot_ss, sv, sv);
        sum_abs_s = vaddq_f32(sum_abs_s, vabsq_f32(sv));
    }

    let mut stats = [
        vaddvq_f32(dot_ms),
        vaddvq_f32(dot_mm),
        vaddvq_f32(dot_ss),
        vaddvq_f32(sum_abs_s),
    ];
    for (&mv, &sv) in m_tail.iter().zip(s_tail) {
        stats[0] = mv.mul_add(sv, stats[0]);
        stats[1] = mv.mul_add(mv, stats[1]);
        stats[2] = sv.mul_add(sv, stats[2]);
        stats[3] += sv.abs();
    }
    stats
}

#[target_feature(enable = "neon")]
pub(crate) fn apply_cfl_neon(x: &mut [f32], y: &[f32], b: &mut [f32], cmap_factor: [f32; 3]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), b.len());

    let cx = vdupq_n_f32(-cmap_factor[0]);
    let cb = vdupq_n_f32(-cmap_factor[2]);
    let (x_chunks, x_tail) = x.as_chunks_mut::<4>();
    let (y_chunks, y_tail) = y.as_chunks::<4>();
    let (b_chunks, b_tail) = b.as_chunks_mut::<4>();

    let (x_groups, x_chunks_tail) = x_chunks.as_chunks_mut::<4>();
    let (y_groups, y_chunks_tail) = y_chunks.as_chunks::<4>();
    let (b_groups, b_chunks_tail) = b_chunks.as_chunks_mut::<4>();
    for ((x, y), b) in x_groups.iter_mut().zip(y_groups).zip(b_groups) {
        let y0 = unsafe { vld1q_f32(y[0].as_ptr()) };
        let y1 = unsafe { vld1q_f32(y[1].as_ptr()) };
        let y2 = unsafe { vld1q_f32(y[2].as_ptr()) };
        let y3 = unsafe { vld1q_f32(y[3].as_ptr()) };
        let x0 = unsafe { vld1q_f32(x[0].as_ptr()) };
        let x1 = unsafe { vld1q_f32(x[1].as_ptr()) };
        let x2 = unsafe { vld1q_f32(x[2].as_ptr()) };
        let x3 = unsafe { vld1q_f32(x[3].as_ptr()) };
        let b0 = unsafe { vld1q_f32(b[0].as_ptr()) };
        let b1 = unsafe { vld1q_f32(b[1].as_ptr()) };
        let b2 = unsafe { vld1q_f32(b[2].as_ptr()) };
        let b3 = unsafe { vld1q_f32(b[3].as_ptr()) };

        let x0 = vfmaq_f32(x0, cx, y0);
        let x1 = vfmaq_f32(x1, cx, y1);
        let x2 = vfmaq_f32(x2, cx, y2);
        let x3 = vfmaq_f32(x3, cx, y3);
        let b0 = vfmaq_f32(b0, cb, y0);
        let b1 = vfmaq_f32(b1, cb, y1);
        let b2 = vfmaq_f32(b2, cb, y2);
        let b3 = vfmaq_f32(b3, cb, y3);

        unsafe {
            vst1q_f32(x[0].as_mut_ptr(), x0);
            vst1q_f32(x[1].as_mut_ptr(), x1);
            vst1q_f32(x[2].as_mut_ptr(), x2);
            vst1q_f32(x[3].as_mut_ptr(), x3);
            vst1q_f32(b[0].as_mut_ptr(), b0);
            vst1q_f32(b[1].as_mut_ptr(), b1);
            vst1q_f32(b[2].as_mut_ptr(), b2);
            vst1q_f32(b[3].as_mut_ptr(), b3);
        }
    }

    for ((x, y), b) in x_chunks_tail
        .iter_mut()
        .zip(y_chunks_tail)
        .zip(b_chunks_tail)
    {
        let yv = unsafe { vld1q_f32(y.as_ptr()) };
        let xv = unsafe { vld1q_f32(x.as_ptr()) };
        let bv = unsafe { vld1q_f32(b.as_ptr()) };
        unsafe {
            vst1q_f32(x.as_mut_ptr(), vfmaq_f32(xv, cx, yv));
            vst1q_f32(b.as_mut_ptr(), vfmaq_f32(bv, cb, yv));
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
    fn apply_cfl_neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        for size in [0, 1, 3, 4, 7, 64, 1024, 4096] {
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
            unsafe { super::apply_cfl_neon(&mut x, &y, &mut b, factors) };

            assert_eq!(x, want_x, "X mismatch at size {size}");
            assert_eq!(b, want_b, "B mismatch at size {size}");
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::cfl_regression_neon;
    use crate::color_correlation::cfl_regression_scalar;

    fn rng(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// The regression sums drive the per-tile CfL slopes, i.e. how much luma is
    /// subtracted from chroma in every block of the frame. NEON accumulates in a
    /// different order than the scalar loop, so compare relative to magnitude.
    #[test]
    fn cfl_regression_neon_matches_scalar() {
        let mut state = 0x1234_abcd_u64;
        for case in 0..12 {
            let mut y = [0.0f32; 64];
            let mut x = [0.0f32; 64];
            let mut b = [0.0f32; 64];
            let mut qm_x = [0.0f32; 64];
            let mut qm_b = [0.0f32; 64];
            for i in 0..64 {
                y[i] = rng(&mut state) * 30.0;
                // A perfectly correlated case and a decorrelated one.
                x[i] = if case % 3 == 0 {
                    0.3 * y[i]
                } else {
                    rng(&mut state) * 10.0
                };
                b[i] = rng(&mut state) * 20.0;
                qm_x[i] = 0.1 + rng(&mut state).abs() * 3.0;
                qm_b[i] = 0.1 + rng(&mut state).abs() * 3.0;
            }
            let want = cfl_regression_scalar(&y, &x, &b, &qm_x, &qm_b);
            let got = unsafe { cfl_regression_neon(&y, &x, &b, &qm_x, &qm_b) };
            for k in 0..4 {
                let scale = want[k].abs().max(1.0);
                assert!(
                    (want[k] - got[k]).abs() <= 1e-4 * scale,
                    "case {case} sum {k}: neon {} vs scalar {}",
                    got[k],
                    want[k]
                );
            }
        }
    }
}
