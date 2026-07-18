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

use crate::dct::{WC4, WC8, WC16, WC32, WC64};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn transpose_4x4(a: __m128, b: __m128, c: __m128, d: __m128) -> [__m128; 4] {
    let t0 = _mm_unpacklo_ps(a, b);
    let t1 = _mm_unpackhi_ps(a, b);
    let t2 = _mm_unpacklo_ps(c, d);
    let t3 = _mm_unpackhi_ps(c, d);
    [
        _mm_movelh_ps(t0, t2),
        _mm_movehl_ps(t2, t0),
        _mm_movelh_ps(t1, t3),
        _mm_movehl_ps(t3, t1),
    ]
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn dct1d_4_s(c: &mut [__m128; 4]) {
    let t0 = _mm_add_ps(c[0], c[3]);
    let t1 = _mm_add_ps(c[1], c[2]);
    let d2 = _mm_mul_ps(_mm_sub_ps(c[0], c[3]), _mm_set1_ps(WC4[0]));
    let d3 = _mm_mul_ps(_mm_sub_ps(c[1], c[2]), _mm_set1_ps(WC4[1]));
    let op = _mm_add_ps(d2, d3);
    let om = _mm_sub_ps(d2, d3);
    c[0] = _mm_add_ps(t0, t1);
    c[1] = _mm_add_ps(_mm_mul_ps(op, _mm_set1_ps(std::f32::consts::SQRT_2)), om);
    c[2] = _mm_sub_ps(t0, t1);
    c[3] = om;
}

macro_rules! recursive_dct {
    ($name:ident, $n:expr, $half:expr, $weights:ident, $child:ident) => {
        #[inline]
        #[target_feature(enable = "sse4.1")]
        fn $name(c: &mut [__m128; $n]) {
            let mut e = [c[0]; $half];
            let mut o = [c[0]; $half];
            for i in 0..$half {
                e[i] = _mm_add_ps(c[i], c[$n - 1 - i]);
                o[i] = _mm_mul_ps(_mm_sub_ps(c[i], c[$n - 1 - i]), _mm_set1_ps($weights[i]));
            }
            $child(&mut e);
            $child(&mut o);
            o[0] = _mm_add_ps(
                _mm_mul_ps(o[0], _mm_set1_ps(std::f32::consts::SQRT_2)),
                o[1],
            );
            for i in 1..($half - 1) {
                o[i] = _mm_add_ps(o[i], o[i + 1]);
            }
            for i in 0..$half {
                c[2 * i] = e[i];
                c[2 * i + 1] = o[i];
            }
        }
    };
}

recursive_dct!(dct1d_8_s, 8, 4, WC8, dct1d_4_s);
recursive_dct!(dct1d_16_s, 16, 8, WC16, dct1d_8_s);
recursive_dct!(dct1d_32_s, 32, 16, WC32, dct1d_16_s);
recursive_dct!(dct1d_64_s, 64, 32, WC64, dct1d_32_s);

#[target_feature(enable = "sse4.1")]
pub(crate) fn dct64x64_sse41(input: &[f32; 4096], output: &mut [f32; 4096]) {
    let mut scratch = [0.0f32; 4096];
    for strip in 0..16 {
        let mut c: [__m128; 64] = std::array::from_fn(|row| unsafe {
            _mm_loadu_ps(input[row * 64 + strip * 4..].as_ptr())
        });
        dct1d_64_s(&mut c);
        for tile_idx in 0..16 {
            let base = tile_idx * 4;
            let tile = transpose_4x4(c[base], c[base + 1], c[base + 2], c[base + 3]);
            for (lane, value) in tile.iter().enumerate() {
                unsafe {
                    _mm_storeu_ps(
                        scratch[(strip * 4 + lane) * 64 + tile_idx * 4..].as_mut_ptr(),
                        *value,
                    );
                }
            }
        }
    }

    let scale = _mm_set1_ps(1.0 / 4096.0);
    for strip in 0..16 {
        let mut c: [__m128; 64] = std::array::from_fn(|col| unsafe {
            _mm_loadu_ps(scratch[col * 64 + strip * 4..].as_ptr())
        });
        dct1d_64_s(&mut c);
        for u in 0..64 {
            unsafe {
                _mm_storeu_ps(
                    output[u * 64 + strip * 4..].as_mut_ptr(),
                    _mm_mul_ps(c[u], scale),
                );
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn dct64x32_sse41(input: &[f32; 2048], output: &mut [f32; 2048]) {
    let mut scratch = [0.0f32; 2048];
    for strip in 0..8 {
        let mut c: [__m128; 64] = std::array::from_fn(|row| unsafe {
            _mm_loadu_ps(input[row * 32 + strip * 4..].as_ptr())
        });
        dct1d_64_s(&mut c);
        for tile_idx in 0..16 {
            let base = tile_idx * 4;
            let tile = transpose_4x4(c[base], c[base + 1], c[base + 2], c[base + 3]);
            for (lane, value) in tile.iter().enumerate() {
                unsafe {
                    _mm_storeu_ps(
                        scratch[(strip * 4 + lane) * 64 + tile_idx * 4..].as_mut_ptr(),
                        *value,
                    );
                }
            }
        }
    }

    let scale = _mm_set1_ps(1.0 / 2048.0);
    for strip in 0..16 {
        let mut c: [__m128; 32] = std::array::from_fn(|col| unsafe {
            _mm_loadu_ps(scratch[col * 64 + strip * 4..].as_ptr())
        });
        dct1d_32_s(&mut c);
        for u in 0..32 {
            unsafe {
                _mm_storeu_ps(
                    output[u * 64 + strip * 4..].as_mut_ptr(),
                    _mm_mul_ps(c[u], scale),
                );
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn dct32x64_sse41(input: &[f32; 2048], output: &mut [f32; 2048]) {
    let mut scratch = [0.0f32; 2048];
    for row_strip in 0..8 {
        let mut c = [_mm_setzero_ps(); 64];
        for tile_idx in 0..16 {
            let tile = transpose_4x4(
                unsafe { _mm_loadu_ps(input[(row_strip * 4) * 64 + tile_idx * 4..].as_ptr()) },
                unsafe { _mm_loadu_ps(input[(row_strip * 4 + 1) * 64 + tile_idx * 4..].as_ptr()) },
                unsafe { _mm_loadu_ps(input[(row_strip * 4 + 2) * 64 + tile_idx * 4..].as_ptr()) },
                unsafe { _mm_loadu_ps(input[(row_strip * 4 + 3) * 64 + tile_idx * 4..].as_ptr()) },
            );
            c[tile_idx * 4..tile_idx * 4 + 4].copy_from_slice(&tile);
        }
        dct1d_64_s(&mut c);
        for u in 0..64 {
            unsafe { _mm_storeu_ps(scratch[u * 32 + row_strip * 4..].as_mut_ptr(), c[u]) };
        }
    }

    let scale = _mm_set1_ps(1.0 / 2048.0);
    for freq_strip in 0..16 {
        let mut c = [_mm_setzero_ps(); 32];
        for row_tile in 0..8 {
            let tile = transpose_4x4(
                unsafe { _mm_loadu_ps(scratch[(freq_strip * 4) * 32 + row_tile * 4..].as_ptr()) },
                unsafe {
                    _mm_loadu_ps(scratch[(freq_strip * 4 + 1) * 32 + row_tile * 4..].as_ptr())
                },
                unsafe {
                    _mm_loadu_ps(scratch[(freq_strip * 4 + 2) * 32 + row_tile * 4..].as_ptr())
                },
                unsafe {
                    _mm_loadu_ps(scratch[(freq_strip * 4 + 3) * 32 + row_tile * 4..].as_ptr())
                },
            );
            c[row_tile * 4..row_tile * 4 + 4].copy_from_slice(&tile);
        }
        dct1d_32_s(&mut c);
        for v in 0..32 {
            unsafe {
                _mm_storeu_ps(
                    output[v * 64 + freq_strip * 4..].as_mut_ptr(),
                    _mm_mul_ps(c[v], scale),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    fn fill<const N: usize>(mut state: u64) -> [f32; N] {
        std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
    }

    #[test]
    fn dct64x64_sse41_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut cases = Vec::with_capacity(11);
        cases.push([0.5f32; 4096]);
        let mut alternating = [0.0f32; 4096];
        for (i, value) in alternating.iter_mut().enumerate() {
            *value = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
        }
        cases.push(alternating);
        let mut impulse = [0.0f32; 4096];
        impulse[4095] = 1.0;
        cases.push(impulse);
        for seed in 0..8 {
            cases.push(fill(0x6464 + seed));
        }

        for (case, input) in cases.iter().enumerate() {
            let mut got = [0.0f32; 4096];
            let mut want = [0.0f32; 4096];
            unsafe { super::dct64x64_sse41(input, &mut got) };
            crate::dct::dct64x64_scalar(input, &mut want);
            let max_error = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_error < 1e-4, "case {case}: max error {max_error}");
        }
    }

    fn check_rectangular<const N: usize>(
        simd: unsafe fn(&[f32; N], &mut [f32; N]),
        scalar: fn(&[f32; N], &mut [f32; N]),
        seed: u64,
        label: &str,
    ) {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        for case in 0..8 {
            let input = fill(seed + case);
            let mut got = [0.0f32; N];
            let mut want = [0.0f32; N];
            unsafe { simd(&input, &mut got) };
            scalar(&input, &mut want);
            let max_error = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error < 1e-4,
                "{label} case {case}: max error {max_error}"
            );
        }
    }

    #[test]
    fn dct64x32_sse41_matches_scalar() {
        check_rectangular(
            super::dct64x32_sse41,
            crate::dct::dct64x32_scalar,
            0x6432,
            "dct64x32",
        );
    }

    #[test]
    fn dct32x64_sse41_matches_scalar() {
        check_rectangular(
            super::dct32x64_sse41,
            crate::dct::dct32x64_scalar,
            0x3264,
            "dct32x64",
        );
    }
}
