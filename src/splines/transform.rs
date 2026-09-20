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
pub(crate) type ContinuousIdctFn = fn(&[[f32; 4]; 32], f32) -> [f32; 4];

pub(crate) fn select_continuous_idct_fn() -> ContinuousIdctFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |dct, t| unsafe { crate::neon::continuous_idct_neon(dct, t) };
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |dct, t| unsafe { crate::avx::continuous_idct_avx2(dct, t) };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            return |dct, t| unsafe { crate::sse::continuous_idct_sse41(dct, t) };
        }
        if std::is_x86_feature_detected!("sse2") {
            return |dct, t| unsafe { crate::sse::continuous_idct_sse2(dct, t) };
        }
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::continuous_idct_wasm;
    #[allow(unreachable_code)]
    continuous_idct_scalar
}

pub(super) fn continuous_idct_scalar(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
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
        let selected = crate::encoding_context::EncodingContext::default().spline_continuous_idct;
        let steps = if cfg!(miri) { 8 } else { 256 };
        for k in 0..=steps {
            let t = 31.0 * k as f32 / steps as f32;
            let expected = channels.map(|channel| {
                let mut result = 0.0;
                for (i, v) in channel.into_iter().enumerate() {
                    let arg = std::f32::consts::PI / 32.0 * i as f32 * (t + 0.5);
                    result += std::f32::consts::SQRT_2 * v * fast_cos(arg);
                }
                result
            });
            assert_eq!(continuous_idct_scalar(&interleaved, t), expected);
            assert_eq!(selected(&interleaved, t), expected, "t={t}");
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
            {
                if std::is_x86_feature_detected!("sse2") {
                    assert_eq!(
                        unsafe { crate::sse::continuous_idct_sse2(&interleaved, t) },
                        expected
                    );
                }
                if std::is_x86_feature_detected!("sse4.1") {
                    assert_eq!(
                        unsafe { crate::sse::continuous_idct_sse41(&interleaved, t) },
                        expected
                    );
                }
            }
        }
    }
}
