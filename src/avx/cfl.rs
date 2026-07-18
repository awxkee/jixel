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
