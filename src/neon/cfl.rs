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
