/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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
#![allow(clippy::excessive_precision)]

use crate::dct::DctInput;
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
use crate::dct::{dct4x4_2d, dct4x8_2d, fmla};
use std::sync::OnceLock;

/// libjxl `k4x4AFVBasisTranspose`: `AFV_BASIS_TRANSPOSE[j][i]` is basis vector
/// `i` sampled at pixel `j`, so the forward contraction runs down the columns:
/// `coeffs[i] = Σ_j pixels[j] · AFV_BASIS_TRANSPOSE[j][i]`. The basis is
/// orthonormal; column 0 is the constant 0.25, so `coeffs[0]` is 4× the
/// quadrant mean.
#[rustfmt::skip]
pub(crate) static AFV_BASIS_TRANSPOSE: [[f32; 16]; 16] = [
    [
        0.2500000000000000, 0.8769029297991420, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000, -0.4105377591765233,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000,
    ],
    [
        0.2500000000000000, 0.2206518106944235, 0.0000000000000000,
        0.0000000000000000, -0.7071067811865474, 0.6235485373547691,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000,
    ],
    [
        0.2500000000000000, -0.1014005039375376, 0.4067007583026075,
        -0.2125574805828875, 0.0000000000000000, -0.0643507165794627,
        -0.4517556589999482, -0.3046847507248690, 0.3017929516615495,
        0.4082482904638627, 0.1747866975480809, -0.2110560104933578,
        -0.1426608480880726, -0.1381354035075859, -0.1743760259965107,
        0.1135498731499434,
    ],
    [
        0.2500000000000000, -0.1014005039375375, 0.4444481661973445,
        0.3085497062849767, 0.0000000000000000, -0.0643507165794627,
        0.1585450355184006, 0.5112616136591823, 0.2579236279634118,
        0.0000000000000000, 0.0812611176717539, 0.1856718091610980,
        -0.3416446842253372, 0.3302282550303788, 0.0702790691196284,
        -0.0741750459581035,
    ],
    [
        0.2500000000000000, 0.2206518106944236, 0.0000000000000000,
        0.0000000000000000, 0.7071067811865476, 0.6235485373547694,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.0000000000000000,
    ],
    [
        0.2500000000000000, -0.1014005039375378, 0.0000000000000000,
        0.4706702258572536, 0.0000000000000000, -0.0643507165794628,
        -0.0403851516082220, 0.0000000000000000, 0.1627234014286620,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.7367497537172237, 0.0875511500058708, -0.2921026642334881,
        0.1940289303259434,
    ],
    [
        0.2500000000000000, -0.1014005039375377, 0.1957439937204294,
        -0.1621205195722993, 0.0000000000000000, -0.0643507165794628,
        0.0074182263792424, -0.2904801297289980, 0.0952002265347504,
        0.0000000000000000, -0.3675398009862027, 0.4921585901373873,
        0.2462710772207515, -0.0794670660590957, 0.3623817333531167,
        -0.4351904965232280,
    ],
    [
        0.2500000000000000, -0.1014005039375376, 0.2929100136981264,
        0.0000000000000000, 0.0000000000000000, -0.0643507165794627,
        0.3935103426921017, -0.0657870154914280, 0.0000000000000000,
        -0.4082482904638628, -0.3078822139579090, -0.3852501370925192,
        -0.0857401903551931, -0.4613374887461511, 0.0000000000000000,
        0.2191868483885747,
    ],
    [
        0.2500000000000000, -0.1014005039375376, -0.4067007583026072,
        -0.2125574805828705, 0.0000000000000000, -0.0643507165794627,
        -0.4517556589999464, 0.3046847507248840, 0.3017929516615503,
        -0.4082482904638635, -0.1747866975480813, 0.2110560104933581,
        -0.1426608480880734, -0.1381354035075829, -0.1743760259965108,
        0.1135498731499426,
    ],
    [
        0.2500000000000000, -0.1014005039375377, -0.1957439937204287,
        -0.1621205195722833, 0.0000000000000000, -0.0643507165794628,
        0.0074182263792444, 0.2904801297290076, 0.0952002265347505,
        0.0000000000000000, 0.3675398009862011, -0.4921585901373891,
        0.2462710772207514, -0.0794670660591026, 0.3623817333531165,
        -0.4351904965232251,
    ],
    [
        0.2500000000000000, -0.1014005039375375, 0.0000000000000000,
        -0.4706702258572528, 0.0000000000000000, -0.0643507165794627,
        0.1107416575309343, 0.0000000000000000, -0.1627234014286617,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        0.1488339922711357, 0.4972464710953509, 0.2921026642334879,
        0.5550443808910661,
    ],
    [
        0.2500000000000000, -0.1014005039375377, 0.1137907446044809,
        -0.1464291867126764, 0.0000000000000000, -0.0643507165794628,
        0.0829816309488205, -0.2388977352334460, -0.3531238544981630,
        -0.4082482904638630, 0.4826689115059883, 0.1741941265991622,
        -0.0476868035022925, 0.1253805944856366, -0.4326608024727445,
        -0.2546827712406646,
    ],
    [
        0.2500000000000000, -0.1014005039375377, -0.4444481661973438,
        0.3085497062849487, 0.0000000000000000, -0.0643507165794628,
        0.1585450355183970, -0.5112616136592012, 0.2579236279634129,
        0.0000000000000000, -0.0812611176717504, -0.1856718091610990,
        -0.3416446842253373, 0.3302282550303805, 0.0702790691196282,
        -0.0741750459581023,
    ],
    [
        0.2500000000000000, -0.1014005039375376, -0.2929100136981264,
        0.0000000000000000, 0.0000000000000000, -0.0643507165794627,
        0.3935103426921022, 0.0657870154914254, 0.0000000000000000,
        0.4082482904638634, 0.3078822139579031, 0.3852501370925211,
        -0.0857401903551927, -0.4613374887461554, 0.0000000000000000,
        0.2191868483885728,
    ],
    [
        0.2500000000000000, -0.1014005039375376, -0.1137907446044814,
        -0.1464291867126654, 0.0000000000000000, -0.0643507165794627,
        0.0829816309488214, 0.2388977352334547, -0.3531238544981624,
        0.4082482904638630, -0.4826689115059858, -0.1741941265991621,
        -0.0476868035022928, 0.1253805944856431, -0.4326608024727457,
        -0.2546827712406641,
    ],
    [
        0.2500000000000000, -0.1014005039375374, 0.0000000000000000,
        0.4251149611657548, 0.0000000000000000, -0.0643507165794626,
        -0.4517556589999480, 0.0000000000000000, -0.6035859033230976,
        0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
        -0.1426608480880724, -0.1381354035075845, 0.3487520519930227,
        0.1135498731499429,
    ],
];

/// libjxl `AFVDCT4x4`: contract the mirrored 4×4 quadrant down the basis
/// columns.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
fn afv_dct4x4(pixels: &[f32; 16]) -> [f32; 16] {
    std::array::from_fn(|i| {
        let mut acc = 0.0f32;
        for (&px, row) in pixels.iter().zip(AFV_BASIS_TRANSPOSE.iter()) {
            acc = fmla(px, row[i], acc);
        }
        acc
    })
}

/// libjxl `AFVTransformFromPixels<afv_kind>`: forward AFV of one 8×8 block.
/// `kind` selects the AFV quadrant: bit 0 = right half, bit 1 = bottom half.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn afv_from_pixels(kind: u8, input: DctInput<'_, 8, 8>, out: &mut [f32; 64]) {
    assert!(kind < 4, "invalid AFV kind {kind}");
    let afv_x = (kind & 1) as usize;
    let afv_y = (kind >> 1) as usize;

    // AFV quadrant, mirrored so the block's outer corner maps to (0, 0) of the
    // basis whatever the variant.
    let quad = std::array::from_fn(|i| {
        let dy = i / 4;
        let dx = i % 4;
        let sy = if afv_y == 1 { 3 - dy } else { dy };
        let sx = if afv_x == 1 { 3 - dx } else { dx };
        input.row(sy + 4 * afv_y)[sx + 4 * afv_x]
    });
    let coeff = afv_dct4x4(&quad);
    for iy in 0..4 {
        for ix in 0..4 {
            out[iy * 2 * 8 + ix * 2] = coeff[iy * 4 + ix];
        }
    }

    // 4×4 DCT of the horizontally adjacent quadrant into (even, odd) positions.
    let qx = if afv_x == 1 { 0 } else { 4 };
    let quad = std::array::from_fn(|i| input.row(i / 4 + 4 * afv_y)[qx + i % 4]);
    let d4x4 = dct4x4_2d(&quad);
    for iy in 0..4 {
        for ix in 0..4 {
            out[iy * 2 * 8 + ix * 2 + 1] = d4x4[iy * 4 + ix];
        }
    }

    // 4×8 DCT of the other vertical half into the odd rows.
    let hy = if afv_y == 1 { 0 } else { 4 };
    let half = std::array::from_fn(|i| input.row(hy + i / 8)[i % 8]);
    let d4x8 = dct4x8_2d(&half);
    for iy in 0..4 {
        for ix in 0..8 {
            out[(1 + iy * 2) * 8 + ix] = d4x8[iy * 8 + ix];
        }
    }

    // Rewrite the three sub-part DCs so coefficient 0 becomes the overall
    // block mean (AFV's coeff 0 is 4x the quadrant mean, hence the 0.25).
    let block00 = out[0] * 0.25;
    let block01 = out[1];
    let block10 = out[8];
    out[0] = (block00 + block01 + 2.0 * block10) * 0.25;
    out[1] = (block00 - block01) * 0.5;
    out[8] = (block00 + block01 - 2.0 * block10) * 0.25;
}

pub(crate) type AfvFn = for<'a> fn(DctInput<'a, 8, 8>, &mut [f32; 64]);

#[derive(Clone, Copy)]
pub(crate) struct AfvMethods {
    pub(crate) afv0: AfvFn,
    pub(crate) afv1: AfvFn,
    pub(crate) afv2: AfvFn,
    pub(crate) afv3: AfvFn,
}

static AFV_METHODS: OnceLock<AfvMethods> = OnceLock::new();

fn select_afv_methods() -> AfvMethods {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        AfvMethods {
            afv0: |input, out| unsafe { crate::neon::afv0_neon(input, out) },
            afv1: |input, out| unsafe { crate::neon::afv1_neon(input, out) },
            afv2: |input, out| unsafe { crate::neon::afv2_neon(input, out) },
            afv3: |input, out| unsafe { crate::neon::afv3_neon(input, out) },
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        AfvMethods {
            afv0: crate::wasm::afv0_wasm,
            afv1: crate::wasm::afv1_wasm,
            afv2: crate::wasm::afv2_wasm,
            afv3: crate::wasm::afv3_wasm,
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return AfvMethods {
                afv0: |input, out| unsafe { crate::avx::afv0_avx2(input, out) },
                afv1: |input, out| unsafe { crate::avx::afv1_avx2(input, out) },
                afv2: |input, out| unsafe { crate::avx::afv2_avx2(input, out) },
                afv3: |input, out| unsafe { crate::avx::afv3_avx2(input, out) },
            };
        }
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            return AfvMethods {
                afv0: |input, out| unsafe { crate::sse::afv0_sse41(input, out) },
                afv1: |input, out| unsafe { crate::sse::afv1_sse41(input, out) },
                afv2: |input, out| unsafe { crate::sse::afv2_sse41(input, out) },
                afv3: |input, out| unsafe { crate::sse::afv3_sse41(input, out) },
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    AfvMethods {
        afv0: |input, out| afv_from_pixels(0, input, out),
        afv1: |input, out| afv_from_pixels(1, input, out),
        afv2: |input, out| afv_from_pixels(2, input, out),
        afv3: |input, out| afv_from_pixels(3, input, out),
    }
}

#[inline]
pub(crate) fn selected_afv_methods() -> AfvMethods {
    *AFV_METHODS.get_or_init(select_afv_methods)
}

macro_rules! afv_flat {
    ($name:ident, $method:ident) => {
        pub(crate) fn $name(input: &[f32; 64], output: &mut [f32; 64]) {
            (selected_afv_methods().$method)(DctInput::from_flat(input), output);
        }
    };
}

afv_flat!(afv0, afv0);
afv_flat!(afv1, afv1);
afv_flat!(afv2, afv2);
afv_flat!(afv3, afv3);

#[cfg(test)]
mod tests {
    use super::*;

    fn check_methods(methods: [AfvFn; 4]) {
        for (kind, method) in methods.into_iter().enumerate() {
            for seed in 0..32u32 {
                let flat: [f32; 64] = std::array::from_fn(|i| {
                    let bits = (i as u32)
                        .wrapping_mul(747_796_405)
                        .wrapping_add(seed.wrapping_mul(2_891_336_453));
                    ((bits >> 9) as f32 / (1u32 << 23) as f32) * 2.0 - 1.0
                });
                let mut strided = [f32::NAN; 8 * 13];
                for y in 0..8 {
                    strided[y * 13..y * 13 + 8].copy_from_slice(&flat[y * 8..y * 8 + 8]);
                }
                let mut got = [f32::NAN; 64];
                let mut want = [f32::NAN; 64];
                method(DctInput::new(&strided, 13), &mut got);
                afv_from_pixels(kind as u8, DctInput::from_flat(&flat), &mut want);
                for i in 0..64 {
                    assert!(
                        (got[i] - want[i]).abs() < 1e-4,
                        "AFV{kind} seed={seed} coeff={i}: SIMD {} != scalar {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn selected_methods_match_scalar_and_context() {
        use crate::encoding_context::EncodingContext;

        let methods = selected_afv_methods();
        let ctx = EncodingContext::default();
        let selected = [methods.afv0, methods.afv1, methods.afv2, methods.afv3];
        let context = [ctx.afv0, ctx.afv1, ctx.afv2, ctx.afv3];
        for kind in 0..4 {
            assert_eq!(
                selected[kind] as usize,
                context[kind] as usize,
                "AFV{} context dispatch is not the selected kernel",
                kind.clone()
            );
        }
        check_methods(selected);
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    #[test]
    fn sse41_methods_match_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let methods: [AfvFn; 4] = [
            |input, out| unsafe { crate::sse::afv0_sse41(input, out) },
            |input, out| unsafe { crate::sse::afv1_sse41(input, out) },
            |input, out| unsafe { crate::sse::afv2_sse41(input, out) },
            |input, out| unsafe { crate::sse::afv3_sse41(input, out) },
        ];
        check_methods(methods);
    }

    #[test]
    fn afv_basis_is_orthonormal() {
        for a in 0..16 {
            for b in 0..16 {
                let dot: f32 = (0..16)
                    .map(|j| AFV_BASIS_TRANSPOSE[j][a] * AFV_BASIS_TRANSPOSE[j][b])
                    .sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "basis columns {a}x{b}: dot {dot}"
                );
            }
        }
    }

    #[test]
    fn flat_block_yields_pure_dc() {
        let input = [0.375f32; 64];
        for kind in 0..4u8 {
            let mut out = [f32::NAN; 64];
            afv_from_pixels(kind, DctInput::from_flat(&input), &mut out);
            assert!(
                (out[0] - 0.375).abs() < 1e-6,
                "kind {kind}: DC {} != mean",
                out[0]
            );
            for (k, &v) in out.iter().enumerate().skip(1) {
                assert!(v.abs() < 1e-5, "kind {kind}: AC[{k}] = {v}");
            }
        }
    }

    #[test]
    fn dc_equals_block_mean() {
        let mut s = 123456789u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for kind in 0..4u8 {
            let mut input = [0.0f32; 64];
            input.iter_mut().for_each(|v| *v = rnd());
            let mean = input.iter().sum::<f32>() / 64.0;
            let mut out = [0.0f32; 64];
            afv_from_pixels(kind, DctInput::from_flat(&input), &mut out);
            assert!(
                (out[0] - mean).abs() < 1e-5,
                "kind {kind}: DC {} != mean {mean}",
                out[0]
            );
        }
    }

    /// Invert a linear operator given as a `forward(input) -> output` closure
    /// over `n`-vectors, via Gauss-Jordan elimination in f64.
    fn numeric_inverse(n: usize, forward: impl Fn(&[f32], &mut [f32])) -> Vec<f64> {
        // aug = [F^T | I] row-reduced; F columns from impulse responses.
        let mut aug = vec![0.0f64; n * 2 * n];
        for i in 0..n {
            let mut input = vec![0.0f32; n];
            let mut output = vec![0.0f32; n];
            input[i] = 1.0;
            forward(&input, &mut output);
            for r in 0..n {
                aug[r * 2 * n + i] = output[r] as f64;
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&a, &b| {
                    aug[a * 2 * n + col]
                        .abs()
                        .total_cmp(&aug[b * 2 * n + col].abs())
                })
                .unwrap();
            for k in 0..2 * n {
                aug.swap(col * 2 * n + k, pivot * 2 * n + k);
            }
            let d = aug[col * 2 * n + col];
            assert!(d.abs() > 1e-9, "singular forward operator");
            for k in 0..2 * n {
                aug[col * 2 * n + k] /= d;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let f = aug[r * 2 * n + col];
                for k in 0..2 * n {
                    aug[r * 2 * n + k] -= f * aug[col * 2 * n + k];
                }
            }
        }
        let mut inv = vec![0.0f64; n * n];
        for r in 0..n {
            inv[r * n..r * n + n].copy_from_slice(&aug[r * 2 * n + n..r * 2 * n + 2 * n]);
        }
        inv
    }

    fn apply(matrix: &[f64], n: usize, input: &[f32], output: &mut [f32]) {
        for (r, out) in output.iter_mut().enumerate() {
            *out = (0..n)
                .map(|k| matrix[r * n + k] * input[k] as f64)
                .sum::<f64>() as f32;
        }
    }

    /// Port of the decoder's `AFVTransformToPixels` (libjxl
    /// `dec_transforms-inl.h`), with the DCT4x4/DCT4x8 sub-inverses obtained
    /// by numerically inverting jixel's forward primitives. Round-tripping
    /// through this checks the quadrant mirroring, the scatter positions and
    /// the DC rewrite against what a conforming decoder will do.
    fn afv_to_pixels(kind: u8, coefficients: &[f32; 64], pixels: &mut [f32; 64]) {
        let afv_x = (kind & 1) as usize;
        let afv_y = (kind >> 1) as usize;
        let block00 = coefficients[0];
        let block01 = coefficients[1];
        let block10 = coefficients[8];
        let dcs = [
            (block00 + block10 + block01) * 4.0,
            block00 + block10 - block01,
            block00 - block10,
        ];
        // IAFV from the (even, even) positions.
        let mut coeff = [0.0f32; 16];
        coeff[0] = dcs[0];
        for iy in 0..4 {
            for ix in 0..4 {
                if ix == 0 && iy == 0 {
                    continue;
                }
                coeff[iy * 4 + ix] = coefficients[iy * 2 * 8 + ix * 2];
            }
        }
        // Orthonormal basis: pixels[j] = sum_i coeff[i] * basis[j][i].
        let mut quad = [0.0f32; 16];
        for (j, px) in quad.iter_mut().enumerate() {
            *px = (0..16)
                .map(|i| coeff[i] * AFV_BASIS_TRANSPOSE[j][i])
                .sum::<f32>();
        }
        for iy in 0..4 {
            for ix in 0..4 {
                let sy = if afv_y == 1 { 3 - iy } else { iy };
                let sx = if afv_x == 1 { 3 - ix } else { ix };
                pixels[(iy + afv_y * 4) * 8 + afv_x * 4 + ix] = quad[sy * 4 + sx];
            }
        }
        // IDCT4x4 from the (even, odd) positions.
        let inv4x4 = numeric_inverse(16, |i, o| {
            o.copy_from_slice(&dct4x4_2d(i.first_chunk::<16>().unwrap()))
        });
        let mut block = [0.0f32; 16];
        block[0] = dcs[1];
        for iy in 0..4 {
            for ix in 0..4 {
                if ix == 0 && iy == 0 {
                    continue;
                }
                block[iy * 4 + ix] = coefficients[iy * 2 * 8 + ix * 2 + 1];
            }
        }
        let mut quad_pixels = [0.0f32; 16];
        apply(&inv4x4, 16, &block, &mut quad_pixels);
        let qx = if afv_x == 1 { 0 } else { 4 };
        for iy in 0..4 {
            for ix in 0..4 {
                pixels[(afv_y * 4 + iy) * 8 + qx + ix] = quad_pixels[iy * 4 + ix];
            }
        }
        // IDCT4x8 for the other vertical half.
        let inv4x8 = numeric_inverse(32, |i, o| {
            o.copy_from_slice(&dct4x8_2d(i.first_chunk::<32>().unwrap()))
        });
        let mut half = [0.0f32; 32];
        half[0] = dcs[2];
        for iy in 0..4 {
            for ix in 0..8 {
                if ix == 0 && iy == 0 {
                    continue;
                }
                half[iy * 8 + ix] = coefficients[(1 + iy * 2) * 8 + ix];
            }
        }
        let mut half_pixels = [0.0f32; 32];
        apply(&inv4x8, 32, &half, &mut half_pixels);
        let hy = if afv_y == 1 { 0 } else { 4 };
        for iy in 0..4 {
            pixels[(hy + iy) * 8..(hy + iy) * 8 + 8]
                .copy_from_slice(&half_pixels[iy * 8..iy * 8 + 8]);
        }
    }

    #[test]
    fn round_trips_through_the_decoder_reference() {
        let mut s = 987654321u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for kind in 0..4u8 {
            let mut input = [0.0f32; 64];
            input.iter_mut().for_each(|v| *v = rnd());
            let mut coeffs = [0.0f32; 64];
            afv_from_pixels(kind, DctInput::from_flat(&input), &mut coeffs);
            let mut recon = [f32::NAN; 64];
            afv_to_pixels(kind, &coeffs, &mut recon);
            for k in 0..64 {
                assert!(
                    (recon[k] - input[k]).abs() < 1e-4,
                    "kind {kind}: pixel {k}: {} vs {}",
                    recon[k],
                    input[k]
                );
            }
        }
    }

    #[test]
    fn strided_input_matches_flat_input() {
        let mut flat = [0.0f32; 64];
        for (i, value) in flat.iter_mut().enumerate() {
            *value = (i as f32 * 0.29).sin();
        }
        let mut strided = [f32::NAN; 8 * 13];
        for y in 0..8 {
            strided[y * 13..y * 13 + 8].copy_from_slice(&flat[y * 8..y * 8 + 8]);
        }
        for kind in 0..4u8 {
            let mut expected = [0.0f32; 64];
            let mut actual = [0.0f32; 64];
            afv_from_pixels(kind, DctInput::from_flat(&flat), &mut expected);
            afv_from_pixels(kind, DctInput::new(&strided, 13), &mut actual);
            assert_eq!(actual, expected, "kind {}", kind.clone());
        }
    }
}
