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

use crate::dc_ops::{B_DC_Y_COMPENSATION, BdcCompensation};
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
struct CompensationVectors {
    negative_scale: __m256,
    y_step: __m256,
    weight: __m256,
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load_f32(ptr: *const f32) -> __m256 {
    unsafe { _mm256_loadu_ps(ptr) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load_i16(ptr: *const i16) -> __m256 {
    let value = unsafe { _mm_loadu_si128(ptr.cast()) };
    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(value))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn store_f32(ptr: *mut f32, value: __m256) {
    unsafe { _mm256_storeu_ps(ptr, value) };
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn store_rounded_i16(ptr: *mut i16, value: __m256) {
    let value = super::quant::round_ties_away_i32x8(value);
    let packed = _mm_packs_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256::<1>(value),
    );
    unsafe { _mm_storeu_si128(ptr.cast(), packed) };
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn adjust_value(target: __m256, source_y: __m256, yq: __m256, p: CompensationVectors) -> __m256 {
    let correction = _mm256_mul_ps(source_y, p.negative_scale);
    let error = _mm256_fmadd_ps(yq, p.y_step, correction);
    _mm256_fmadd_ps(p.weight, error, target)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn compensate_block(
    b: *const f32,
    y: *const f32,
    yq: *const i16,
    out: *mut i16,
    scale: __m256,
    negative_cfl: __m256,
    p: CompensationVectors,
) {
    unsafe {
        let b = load_f32(b);
        let y = load_f32(y);
        let yq = load_i16(yq);
        let correction = _mm256_mul_ps(yq, negative_cfl);
        let target = _mm256_fmadd_ps(b, scale, correction);
        store_rounded_i16(out, adjust_value(target, y, yq, p));
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn adjust_block(t: *mut f32, y: *const f32, yq: *const i16, p: CompensationVectors) {
    unsafe {
        let target = load_f32(t);
        let y = load_f32(y);
        let yq = load_i16(yq);
        store_f32(t, adjust_value(target, y, yq, p));
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn reconstruct_block(
    x: *const i16,
    y: *const i16,
    b: *const i16,
    rx: *mut f32,
    ry: *mut f32,
    rb: *mut f32,
    cfl: [__m256; 2],
    steps: [__m256; 3],
) {
    unsafe {
        let x = load_i16(x);
        let y = load_i16(y);
        let b = load_i16(b);
        store_f32(rx, _mm256_mul_ps(_mm256_fmadd_ps(y, cfl[0], x), steps[0]));
        store_f32(ry, _mm256_mul_ps(y, steps[1]));
        store_f32(rb, _mm256_mul_ps(_mm256_fmadd_ps(y, cfl[1], b), steps[2]));
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn compensate_b_dc_avx2(
    source_b: &[f32],
    source_y: &[f32],
    y_quant: &[i16],
    compensation: BdcCompensation,
    cfl_b: f32,
    output: &mut [i16],
) {
    let n = output.len();
    assert_eq!(source_b.len(), n);
    assert_eq!(source_y.len(), n);
    assert_eq!(y_quant.len(), n);
    let scale = _mm256_set1_ps(compensation.scale_b);
    let negative_cfl = _mm256_set1_ps(-cfl_b);
    let p = CompensationVectors {
        negative_scale: _mm256_set1_ps(-compensation.scale_b),
        y_step: _mm256_set1_ps(compensation.y_step_b),
        weight: _mm256_set1_ps(B_DC_Y_COMPENSATION),
    };
    let mut i = 0;
    // All rows have the checked length; each full block accesses 8 lanes.
    unsafe {
        while n - i >= 32 {
            compensate_block(
                source_b.as_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                output.as_mut_ptr().add(i),
                scale,
                negative_cfl,
                p,
            );
            compensate_block(
                source_b.as_ptr().add(i + 8),
                source_y.as_ptr().add(i + 8),
                y_quant.as_ptr().add(i + 8),
                output.as_mut_ptr().add(i + 8),
                scale,
                negative_cfl,
                p,
            );
            compensate_block(
                source_b.as_ptr().add(i + 16),
                source_y.as_ptr().add(i + 16),
                y_quant.as_ptr().add(i + 16),
                output.as_mut_ptr().add(i + 16),
                scale,
                negative_cfl,
                p,
            );
            compensate_block(
                source_b.as_ptr().add(i + 24),
                source_y.as_ptr().add(i + 24),
                y_quant.as_ptr().add(i + 24),
                output.as_mut_ptr().add(i + 24),
                scale,
                negative_cfl,
                p,
            );
            i += 32;
        }
        while n - i >= 8 {
            compensate_block(
                source_b.as_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                output.as_mut_ptr().add(i),
                scale,
                negative_cfl,
                p,
            );
            i += 8;
        }
    }
    if i < n {
        // Keep tails on the SIMD rounding path, without reading past a row.
        let tail = n - i;
        let mut b = [0.0; 8];
        let mut y = [0.0; 8];
        let mut yq = [0i16; 8];
        let mut out = [0i16; 8];
        b[..tail].copy_from_slice(&source_b[i..]);
        y[..tail].copy_from_slice(&source_y[i..]);
        yq[..tail].copy_from_slice(&y_quant[i..]);
        unsafe {
            compensate_block(
                b.as_ptr(),
                y.as_ptr(),
                yq.as_ptr(),
                out.as_mut_ptr(),
                scale,
                negative_cfl,
                p,
            );
        }
        output[i..].copy_from_slice(&out[..tail]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn adjust_b_dc_avx2(
    targets: &mut [f32],
    source_y: &[f32],
    y_quant: &[i16],
    compensation: BdcCompensation,
) {
    let n = targets.len();
    assert_eq!(source_y.len(), n);
    assert_eq!(y_quant.len(), n);
    let p = CompensationVectors {
        negative_scale: _mm256_set1_ps(-compensation.scale_b),
        y_step: _mm256_set1_ps(compensation.y_step_b),
        weight: _mm256_set1_ps(B_DC_Y_COMPENSATION),
    };
    let mut i = 0;
    // All rows have the checked length; each full block accesses 8 lanes.
    unsafe {
        while n - i >= 32 {
            adjust_block(
                targets.as_mut_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                p,
            );
            adjust_block(
                targets.as_mut_ptr().add(i + 8),
                source_y.as_ptr().add(i + 8),
                y_quant.as_ptr().add(i + 8),
                p,
            );
            adjust_block(
                targets.as_mut_ptr().add(i + 16),
                source_y.as_ptr().add(i + 16),
                y_quant.as_ptr().add(i + 16),
                p,
            );
            adjust_block(
                targets.as_mut_ptr().add(i + 24),
                source_y.as_ptr().add(i + 24),
                y_quant.as_ptr().add(i + 24),
                p,
            );
            i += 32;
        }
        while n - i >= 8 {
            adjust_block(
                targets.as_mut_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                p,
            );
            i += 8;
        }
    }
    if i < n {
        let tail = n - i;
        let mut t = [0.0; 8];
        let mut y = [0.0; 8];
        let mut yq = [0i16; 8];
        t[..tail].copy_from_slice(&targets[i..]);
        y[..tail].copy_from_slice(&source_y[i..]);
        yq[..tail].copy_from_slice(&y_quant[i..]);
        unsafe {
            adjust_block(t.as_mut_ptr(), y.as_ptr(), yq.as_ptr(), p);
        }
        targets[i..].copy_from_slice(&t[..tail]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn reconstruct_dc_avx2(
    input: [&[i16]; 3],
    cfl: [f32; 2],
    steps: [f32; 3],
    output: [&mut [f32]; 3],
) {
    let [x, y, b] = input;
    let [rx, ry, rb] = output;
    let n = y.len();
    for len in [x.len(), b.len(), rx.len(), ry.len(), rb.len()] {
        assert_eq!(len, n);
    }
    let cfl = cfl.map(|v| _mm256_set1_ps(v));
    let steps = steps.map(|v| _mm256_set1_ps(v));
    let mut i = 0;
    // All rows have the checked length; each full block accesses 8 lanes.
    unsafe {
        while n - i >= 32 {
            reconstruct_block(
                x.as_ptr().add(i),
                y.as_ptr().add(i),
                b.as_ptr().add(i),
                rx.as_mut_ptr().add(i),
                ry.as_mut_ptr().add(i),
                rb.as_mut_ptr().add(i),
                cfl,
                steps,
            );
            reconstruct_block(
                x.as_ptr().add(i + 8),
                y.as_ptr().add(i + 8),
                b.as_ptr().add(i + 8),
                rx.as_mut_ptr().add(i + 8),
                ry.as_mut_ptr().add(i + 8),
                rb.as_mut_ptr().add(i + 8),
                cfl,
                steps,
            );
            reconstruct_block(
                x.as_ptr().add(i + 16),
                y.as_ptr().add(i + 16),
                b.as_ptr().add(i + 16),
                rx.as_mut_ptr().add(i + 16),
                ry.as_mut_ptr().add(i + 16),
                rb.as_mut_ptr().add(i + 16),
                cfl,
                steps,
            );
            reconstruct_block(
                x.as_ptr().add(i + 24),
                y.as_ptr().add(i + 24),
                b.as_ptr().add(i + 24),
                rx.as_mut_ptr().add(i + 24),
                ry.as_mut_ptr().add(i + 24),
                rb.as_mut_ptr().add(i + 24),
                cfl,
                steps,
            );
            i += 32;
        }
        while n - i >= 8 {
            reconstruct_block(
                x.as_ptr().add(i),
                y.as_ptr().add(i),
                b.as_ptr().add(i),
                rx.as_mut_ptr().add(i),
                ry.as_mut_ptr().add(i),
                rb.as_mut_ptr().add(i),
                cfl,
                steps,
            );
            i += 8;
        }
    }
    if i < n {
        let tail = n - i;
        let mut tx = [0i16; 8];
        let mut ty = [0i16; 8];
        let mut tb = [0i16; 8];
        let mut ox = [0.0; 8];
        let mut oy = [0.0; 8];
        let mut ob = [0.0; 8];
        tx[..tail].copy_from_slice(&x[i..]);
        ty[..tail].copy_from_slice(&y[i..]);
        tb[..tail].copy_from_slice(&b[i..]);
        unsafe {
            reconstruct_block(
                tx.as_ptr(),
                ty.as_ptr(),
                tb.as_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                ob.as_mut_ptr(),
                cfl,
                steps,
            );
        }
        rx[i..].copy_from_slice(&ox[..tail]);
        ry[i..].copy_from_slice(&oy[..tail]);
        rb[i..].copy_from_slice(&ob[..tail]);
    }
}

#[target_feature(enable = "avx2")]
fn search_fmla(a: __m256, b: __m256, c: __m256) -> __m256 {
    #[cfg(target_feature = "fma")]
    {
        _mm256_fmadd_ps(a, b, c)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        _mm256_add_ps(_mm256_mul_ps(a, b), c)
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dc_error_row_avx2(
    src: &[f32],
    luma: &[i16],
    levels: &[i16],
    scale: f32,
    factor: f32,
    multiplier: f32,
    out: &mut [f32],
) {
    assert!(src.len() >= out.len() && luma.len() >= out.len() && levels.len() >= out.len());
    let end = out.len() / 8 * 8;
    for x in (0..end).step_by(8) {
        let y: [f32; 8] = std::array::from_fn(|i| -(luma[x + i] as f32));
        let q: [f32; 8] = std::array::from_fn(|i| levels[x + i] as f32);
        unsafe {
            let error = _mm256_sub_ps(
                search_fmla(
                    _mm256_loadu_ps(src.as_ptr().add(x)),
                    _mm256_set1_ps(scale),
                    _mm256_mul_ps(_mm256_loadu_ps(y.as_ptr()), _mm256_set1_ps(factor)),
                ),
                _mm256_loadu_ps(q.as_ptr()),
            );
            _mm256_storeu_ps(
                out.as_mut_ptr().add(x),
                _mm256_mul_ps(error, _mm256_set1_ps(multiplier)),
            );
        }
    }
    crate::dc_ops::dc_error_row_scalar(
        &src[end..],
        &luma[end..],
        &levels[end..],
        scale,
        factor,
        multiplier,
        &mut out[end..],
    );
}

#[target_feature(enable = "avx2")]
pub(crate) fn dc_distortion_row_avx2(
    src: &[f32],
    luma: &[i16],
    levels: &[i16],
    scale: f32,
    factor: f32,
    mut sum: f64,
) -> f64 {
    assert_eq!(src.len(), luma.len());
    assert_eq!(src.len(), levels.len());
    // Only the independent error terms are vectorized. Carry the f64
    // sum through every row in its original coefficient order.
    let mut errors = [0.0; 32];
    for start in (0..src.len()).step_by(errors.len()) {
        let end = (start + errors.len()).min(src.len());
        let errors = &mut errors[..end - start];
        dc_error_row_avx2(
            &src[start..end],
            &luma[start..end],
            &levels[start..end],
            scale,
            factor,
            1.0,
            errors,
        );
        for &error in errors.iter() {
            sum += f64::from(error).powi(2);
        }
    }
    sum
}

#[target_feature(enable = "avx2")]
pub(crate) fn add_scaled_avx2(out: &mut [f32], src: &[f32], factor: f32) {
    assert!(src.len() >= out.len());
    let end = out.len() / 8 * 8;
    for x in (0..end).step_by(8) {
        unsafe {
            _mm256_storeu_ps(
                out.as_mut_ptr().add(x),
                _mm256_add_ps(
                    _mm256_loadu_ps(out.as_ptr().add(x)),
                    _mm256_mul_ps(_mm256_set1_ps(factor), _mm256_loadu_ps(src.as_ptr().add(x))),
                ),
            );
        }
    }
    crate::dc_ops::add_scaled_scalar(&mut out[end..], &src[end..], factor);
}
