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

//! Continuous IDCT at fractional arc positions. The block IDCT kernels sample
//! integer positions, so spline rendering instead packs X, Y, B and sigma into
//! four SIMD lanes and shares each cosine between them. Cosines are evaluated
//! four frequencies at a time, with the original accumulation order preserved.

fn fast_cos(x: f32) -> f32 {
    use std::f32::consts::PI;
    let pi2 = PI * 2.0;
    let xmod = x - (x * (0.5 / PI)).floor() * pi2;
    let x_pi = xmod.min(pi2 - xmod);
    let above = x_pi >= PI / 2.0;
    let xh = if above { PI - x_pi } else { x_pi };
    let xs = xh * 0.25;
    let x2 = xs * xs;
    let x4 = x2 * x2;
    let pre = x4 * 0.06960438 + (x2 * -0.84087373 + 1.68179268);
    let s1 = pre * pre - std::f32::consts::SQRT_2;
    let s2 = s1 * s1 - 1.0;
    if above { -s2 } else { s2 }
}

/// Coefficients are interleaved by channel and already multiplied by sqrt(2).
pub(super) type ContinuousIdctFn = unsafe fn(&[[f32; 4]; 32], f32) -> [f32; 4];

#[inline]
fn select_continuous_idct() -> ContinuousIdctFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        // NEON is part of the AArch64 baseline.
        return continuous_idct_neon;
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "sse", feature = "avx")
    ))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            return x86::continuous_idct_sse41;
        }
        if std::is_x86_feature_detected!("sse2") {
            return x86::continuous_idct_sse2;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        return continuous_idct_wasm;
    }
    #[allow(unreachable_code)]
    continuous_idct_scalar
}

/// Resolve once before the arc-sample loop, never once per cosine or sample.
#[inline]
pub(super) fn selected_continuous_idct() -> ContinuousIdctFn {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "sse", feature = "avx")
    ))]
    {
        static IDCT: std::sync::OnceLock<ContinuousIdctFn> = std::sync::OnceLock::new();
        *IDCT.get_or_init(select_continuous_idct)
    }
    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "sse", feature = "avx")
    )))]
    select_continuous_idct()
}

fn continuous_idct_scalar(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
    let mut result = [0.0; 4];
    for (i, row) in dct.iter().enumerate() {
        let arg = std::f32::consts::PI / 32.0 * i as f32 * (t + 0.5);
        let cos = fast_cos(arg);
        for (out, &v) in result.iter_mut().zip(row) {
            *out += v * cos;
        }
    }
    result
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[target_feature(enable = "neon")]
fn fast_cos_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    use std::f32::consts::{PI, SQRT_2};
    let pi2 = vdupq_n_f32(PI * 2.0);
    let periods = vrndmq_f32(vmulq_n_f32(x, 0.5 / PI));
    let xmod = vsubq_f32(x, vmulq_f32(periods, pi2));
    let x_pi = vminq_f32(xmod, vsubq_f32(pi2, xmod));
    let above = vcgeq_f32(x_pi, vdupq_n_f32(PI / 2.0));
    let xh = vbslq_f32(above, vsubq_f32(vdupq_n_f32(PI), x_pi), x_pi);
    let xs = vmulq_n_f32(xh, 0.25);
    let x2 = vmulq_f32(xs, xs);
    let x4 = vmulq_f32(x2, x2);
    // Separate multiply/add operations preserve the decoder's approximation.
    let pre = vaddq_f32(
        vmulq_n_f32(x4, 0.06960438),
        vaddq_f32(vmulq_n_f32(x2, -0.84087373), vdupq_n_f32(1.68179268)),
    );
    let s1 = vsubq_f32(vmulq_f32(pre, pre), vdupq_n_f32(SQRT_2));
    let s2 = vsubq_f32(vmulq_f32(s1, s1), vdupq_n_f32(1.0));
    vbslq_f32(above, vnegq_f32(s2), s2)
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[target_feature(enable = "neon")]
fn continuous_idct_neon(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mut result = vdupq_n_f32(0.0);
    let indices = [0.0, 1.0, 2.0, 3.0];
    let mut indices = unsafe { vld1q_f32(indices.as_ptr()) };
    for rows in dct.as_chunks::<4>().0 {
        let args = vmulq_n_f32(vmulq_n_f32(indices, std::f32::consts::PI / 32.0), t + 0.5);
        let cos = fast_cos_neon(args);
        let cosines = [
            vdupq_laneq_f32::<0>(cos),
            vdupq_laneq_f32::<1>(cos),
            vdupq_laneq_f32::<2>(cos),
            vdupq_laneq_f32::<3>(cos),
        ];
        // Accumulate frequencies in order, retaining scalar rounding in each channel.
        for (row, cos) in rows.iter().zip(cosines) {
            let values = unsafe { vld1q_f32(row.as_ptr()) };
            result = vaddq_f32(result, vmulq_f32(values, cos));
        }
        indices = vaddq_f32(indices, vdupq_n_f32(4.0));
    }
    let mut out = [0.0; 4];
    unsafe { vst1q_f32(out.as_mut_ptr(), result) };
    out
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "sse", feature = "avx")
))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "sse2")]
    fn fast_cos(x: __m128, periods: __m128) -> __m128 {
        use std::f32::consts::{PI, SQRT_2};
        let pi2 = _mm_set1_ps(PI * 2.0);
        let xmod = _mm_sub_ps(x, _mm_mul_ps(periods, pi2));
        let x_pi = _mm_min_ps(xmod, _mm_sub_ps(pi2, xmod));
        let above = _mm_cmpge_ps(x_pi, _mm_set1_ps(PI / 2.0));
        let xh = _mm_or_ps(
            _mm_and_ps(above, _mm_sub_ps(_mm_set1_ps(PI), x_pi)),
            _mm_andnot_ps(above, x_pi),
        );
        let xs = _mm_mul_ps(xh, _mm_set1_ps(0.25));
        let x2 = _mm_mul_ps(xs, xs);
        let x4 = _mm_mul_ps(x2, x2);
        let pre = _mm_add_ps(
            _mm_mul_ps(x4, _mm_set1_ps(0.06960438)),
            _mm_add_ps(
                _mm_mul_ps(x2, _mm_set1_ps(-0.84087373)),
                _mm_set1_ps(1.68179268),
            ),
        );
        let s1 = _mm_sub_ps(_mm_mul_ps(pre, pre), _mm_set1_ps(SQRT_2));
        let s2 = _mm_sub_ps(_mm_mul_ps(s1, s1), _mm_set1_ps(1.0));
        _mm_xor_ps(s2, _mm_and_ps(above, _mm_set1_ps(-0.0)))
    }

    // Generate both target-feature bodies so the rounding and polynomial stay
    // inlined in the frequency loop, with no per-vector function-pointer call.
    macro_rules! continuous_idct {
        ($name:ident, $feature:literal, $floor:expr) => {
            #[target_feature(enable = $feature)]
            pub(super) fn $name(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
                let mut result = _mm_setzero_ps();
                let mut indices = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
                for rows in dct.as_chunks::<4>().0 {
                    let args = _mm_mul_ps(
                        _mm_mul_ps(indices, _mm_set1_ps(std::f32::consts::PI / 32.0)),
                        _mm_set1_ps(t + 0.5),
                    );
                    let periods =
                        ($floor)(_mm_mul_ps(args, _mm_set1_ps(0.5 / std::f32::consts::PI)));
                    let cos = fast_cos(args, periods);
                    let cosines = [
                        _mm_shuffle_ps::<0x00>(cos, cos),
                        _mm_shuffle_ps::<0x55>(cos, cos),
                        _mm_shuffle_ps::<0xaa>(cos, cos),
                        _mm_shuffle_ps::<0xff>(cos, cos),
                    ];
                    for (row, cos) in rows.iter().zip(cosines) {
                        let values = unsafe { _mm_loadu_ps(row.as_ptr()) };
                        result = _mm_add_ps(result, _mm_mul_ps(values, cos));
                    }
                    indices = _mm_add_ps(indices, _mm_set1_ps(4.0));
                }
                let mut out = [0.0; 4];
                unsafe { _mm_storeu_ps(out.as_mut_ptr(), result) };
                out
            }
        };
    }

    continuous_idct!(continuous_idct_sse41, "sse4.1", _mm_floor_ps);
    // Spline arguments are finite, nonnegative and bounded by PI * 31 * 31.5 / 32.
    // The pre-SSE4.1 fallback can therefore use truncation for floor.
    continuous_idct!(continuous_idct_sse2, "sse2", |x| _mm_cvtepi32_ps(
        _mm_cvttps_epi32(x)
    ));
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
fn fast_cos_wasm(x: std::arch::wasm32::v128) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;
    use std::f32::consts::{PI, SQRT_2};
    let pi2 = f32x4_splat(PI * 2.0);
    let periods = f32x4_floor(f32x4_mul(x, f32x4_splat(0.5 / PI)));
    let xmod = f32x4_sub(x, f32x4_mul(periods, pi2));
    let x_pi = f32x4_min(xmod, f32x4_sub(pi2, xmod));
    let above = f32x4_ge(x_pi, f32x4_splat(PI / 2.0));
    let xh = v128_bitselect(f32x4_sub(f32x4_splat(PI), x_pi), x_pi, above);
    let xs = f32x4_mul(xh, f32x4_splat(0.25));
    let x2 = f32x4_mul(xs, xs);
    let x4 = f32x4_mul(x2, x2);
    let pre = f32x4_add(
        f32x4_mul(x4, f32x4_splat(0.06960438)),
        f32x4_add(
            f32x4_mul(x2, f32x4_splat(-0.84087373)),
            f32x4_splat(1.68179268),
        ),
    );
    let s1 = f32x4_sub(f32x4_mul(pre, pre), f32x4_splat(SQRT_2));
    let s2 = f32x4_sub(f32x4_mul(s1, s1), f32x4_splat(1.0));
    v128_bitselect(f32x4_neg(s2), s2, above)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
fn continuous_idct_wasm(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
    use std::arch::wasm32::*;
    let mut result = f32x4_splat(0.0);
    let mut indices = f32x4(0.0, 1.0, 2.0, 3.0);
    for rows in dct.as_chunks::<4>().0 {
        let args = f32x4_mul(
            f32x4_mul(indices, f32x4_splat(std::f32::consts::PI / 32.0)),
            f32x4_splat(t + 0.5),
        );
        let cos = fast_cos_wasm(args);
        let cosines = [
            f32x4_splat(f32x4_extract_lane::<0>(cos)),
            f32x4_splat(f32x4_extract_lane::<1>(cos)),
            f32x4_splat(f32x4_extract_lane::<2>(cos)),
            f32x4_splat(f32x4_extract_lane::<3>(cos)),
        ];
        for (row, cos) in rows.iter().zip(cosines) {
            let values = unsafe { v128_load(row.as_ptr().cast()) };
            result = f32x4_add(result, f32x4_mul(values, cos));
        }
        indices = f32x4_add(indices, f32x4_splat(4.0));
    }
    let mut out = [0.0; 4];
    unsafe { v128_store(out.as_mut_ptr().cast(), result) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_idct_matches_separate_channels() {
        // Exercise every frequency, mixed signs, DC-only sigma, and fractional
        // positions: a conventional 32-point IDCT cannot substitute for this.
        let channels: [[f32; 32]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| {
                if c == 3 && i != 0 {
                    0.0
                } else {
                    ((i * 17 + c * 7) % 41) as f32 * 0.01 - 0.2
                }
            })
        });
        let interleaved = std::array::from_fn(|i| {
            std::array::from_fn(|c| std::f32::consts::SQRT_2 * channels[c][i])
        });
        let selected = selected_continuous_idct();
        for k in 0..=256 {
            let t = 31.0 * k as f32 / 256.0;
            let expected = channels.map(|channel| {
                let mut result = 0.0;
                for (i, v) in channel.into_iter().enumerate() {
                    let arg = std::f32::consts::PI / 32.0 * i as f32 * (t + 0.5);
                    result += std::f32::consts::SQRT_2 * v * fast_cos(arg);
                }
                result
            });
            assert_eq!(continuous_idct_scalar(&interleaved, t), expected);
            assert_eq!(unsafe { selected(&interleaved, t) }, expected, "t={t}");
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                any(feature = "sse", feature = "avx")
            ))]
            {
                if std::is_x86_feature_detected!("sse2") {
                    assert_eq!(
                        unsafe { x86::continuous_idct_sse2(&interleaved, t) },
                        expected
                    );
                }
                if std::is_x86_feature_detected!("sse4.1") {
                    assert_eq!(
                        unsafe { x86::continuous_idct_sse41(&interleaved, t) },
                        expected
                    );
                }
            }
        }
    }
}
