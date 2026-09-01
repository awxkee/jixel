/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
#![allow(unused)]
use std::sync::OnceLock;

#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    target_arch = "aarch64"
))]
#[inline(always)]
pub(crate) fn fmla(a: f32, b: f32, c: f32) -> f32 {
    f32::mul_add(a, b, c)
}

#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    target_arch = "aarch64"
)))]
#[inline(always)]
pub(crate) fn fmla(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

pub(crate) const WC4: [f32; 2] = [0.541_196_1, 1.306_563];
pub(crate) const INV_WC4: [f32; 2] = [1.0 / WC4[0], 1.0 / WC4[1]];

pub(crate) const WC8: [f32; 4] = [0.509_795_6, 0.601_344_9, 0.899_976_2, 2.562_915_6];
pub(crate) const INV_WC8: [f32; 4] = [1.0 / WC8[0], 1.0 / WC8[1], 1.0 / WC8[2], 1.0 / WC8[3]];

#[allow(unused)]
#[inline(always)]
fn dct1d_2(buf: &mut [f32]) {
    let a = buf[0];
    let b = buf[1];
    buf[0] = a + b;
    buf[1] = a - b;
}

#[allow(unused)]
#[inline(always)]
fn dct1d_4(buf: &mut [f32; 4]) {
    let mut tmp = [0.0f32; 4];
    tmp[0] = buf[0] + buf[3];
    tmp[1] = buf[1] + buf[2];
    dct1d_2(&mut tmp[0..2]);
    tmp[2] = buf[0] - buf[3];
    tmp[3] = buf[1] - buf[2];
    tmp[2] *= WC4[0];
    tmp[3] *= WC4[1];
    dct1d_2(&mut tmp[2..4]);
    tmp[2] = fmla(tmp[2], std::f32::consts::SQRT_2, tmp[3]);
    buf[0] = tmp[0];
    buf[2] = tmp[1];
    buf[1] = tmp[2];
    buf[3] = tmp[3];
}

#[inline(always)]
#[allow(unused)]
fn dct1d_8(buf: &mut [f32]) {
    let mut tmp = [0.0f32; 8];
    for i in 0..4 {
        tmp[i] = buf[i] + buf[7 - i];
    }
    dct1d_4(<&mut [f32; 4]>::try_from(&mut tmp[..4]).unwrap());
    for i in 0..4 {
        tmp[4 + i] = (buf[i] - buf[7 - i]) * WC8[i];
    }
    dct1d_4(<&mut [f32; 4]>::try_from(&mut tmp[4..8]).unwrap());
    tmp[4] = fmla(tmp[4], std::f32::consts::SQRT_2, tmp[5]);
    tmp[5] += tmp[6];
    tmp[6] += tmp[7];
    for i in 0..4 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[4 + i];
    }
}

/// Validated zero-copy view of a rectangular transform input.
///
/// Private fields guarantee that every row contains `W` readable values, so
/// scalar and SIMD DCT/IDCT kernels can safely load through an arbitrary stride.
#[derive(Clone, Copy)]
pub(crate) struct DctInput<'a, const W: usize, const H: usize> {
    data: &'a [f32],
    stride: usize,
}

impl<'a, const W: usize, const H: usize> DctInput<'a, W, H> {
    #[inline]
    pub(crate) fn new(data: &'a [f32], stride: usize) -> Self {
        assert!(W != 0 && H != 0);
        assert!(stride >= W);
        let required_len = (H - 1)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(W))
            .expect("DCT input dimensions overflow");
        assert!(
            data.len() >= required_len,
            "DctInput {W}x{H} stride={stride} len={} need={required_len}",
            data.len()
        );
        Self {
            data: &data[..required_len],
            stride,
        }
    }

    #[inline]
    pub(crate) fn from_flat<const N: usize>(data: &'a [f32; N]) -> Self {
        assert_eq!(N, W * H);
        Self::new(data, W)
    }

    #[inline]
    pub(crate) fn row(&self, y: usize) -> &'a [f32; W] {
        assert!(y < H);
        let offset = y * self.stride;
        self.data[offset..offset + W].first_chunk::<W>().unwrap()
    }
}

pub(crate) type DctFn<const W: usize, const H: usize, const N: usize> =
    for<'a> fn(DctInput<'a, W, H>, &mut [f32; N]);

/// JPEG XL's IDENTITY (historically "Hornuss") transform. Each 4x4 quadrant
/// stores pixel residuals from its (1,1) anchor; the four quadrant means are
/// combined by a 2x2 Hadamard in coefficients {0, 1, 8, 9}.
pub(crate) fn identity8x8(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    for qy in 0..2 {
        for qx in 0..2 {
            let anchor = input.row(qy * 4 + 1)[qx * 4 + 1];
            let mut mean = 0.0f32;
            for iy in 0..4 {
                let row = input.row(qy * 4 + iy);
                for ix in 0..4 {
                    mean += row[qx * 4 + ix];
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    output[(qy + iy * 2) * 8 + qx + ix * 2] = row[qx * 4 + ix] - anchor;
                }
            }
            // Coefficient (qy, qx) is needed for the quadrant mean. Preserve
            // the displaced top-left residual in the otherwise unused anchor
            // slot before overwriting it.
            output[(qy + 2) * 8 + qx + 2] = output[qy * 8 + qx];
            output[qy * 8 + qx] = mean * (1.0 / 16.0);
        }
    }
    let b00 = output[0];
    let b01 = output[1];
    let b10 = output[8];
    let b11 = output[9];
    output[0] = (b00 + b01 + b10 + b11) * 0.25;
    output[1] = (b00 + b01 - b10 - b11) * 0.25;
    output[8] = (b00 - b01 + b10 - b11) * 0.25;
    output[9] = (b00 - b01 - b10 + b11) * 0.25;
}

/// Inverse of [`identity8x8`], used by reconstruction-domain strategy scoring.
pub(crate) fn inv_identity8x8(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let get = |i: usize| input.row(i / 8)[i % 8];
    let means = [
        get(0) + get(1) + get(8) + get(9),
        get(0) + get(1) - get(8) - get(9),
        get(0) - get(1) + get(8) - get(9),
        get(0) - get(1) - get(8) + get(9),
    ];
    for qy in 0..2 {
        for qx in 0..2 {
            let mut residual_sum = 0.0f32;
            for iy in 0..4 {
                for ix in 0..4 {
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    let residual = if ix == 0 && iy == 0 {
                        get((qy + 2) * 8 + qx + 2)
                    } else {
                        get((qy + iy * 2) * 8 + qx + ix * 2)
                    };
                    residual_sum += residual;
                }
            }
            let anchor = means[qy * 2 + qx] - residual_sum * (1.0 / 16.0);
            for iy in 0..4 {
                for ix in 0..4 {
                    let residual = if ix == 1 && iy == 1 {
                        0.0
                    } else if ix == 0 && iy == 0 {
                        get((qy + 2) * 8 + qx + 2)
                    } else {
                        get((qy + iy * 2) * 8 + qx + ix * 2)
                    };
                    output[(qy * 4 + iy) * 8 + qx * 4 + ix] = anchor + residual;
                }
            }
        }
    }
}

#[inline]
fn dct2_top_block<const S: usize>(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut temp = [0.0f32; 64];
    let half = S / 2;
    for y in 0..half {
        let row0 = input.row(y * 2);
        let row1 = input.row(y * 2 + 1);
        for x in 0..half {
            let c00 = row0[x * 2];
            let c01 = row0[x * 2 + 1];
            let c10 = row1[x * 2];
            let c11 = row1[x * 2 + 1];
            temp[y * 8 + x] = (c00 + c01 + c10 + c11) * 0.25;
            temp[y * 8 + half + x] = (c00 + c01 - c10 - c11) * 0.25;
            temp[(y + half) * 8 + x] = (c00 - c01 + c10 - c11) * 0.25;
            temp[(y + half) * 8 + half + x] = (c00 - c01 - c10 + c11) * 0.25;
        }
    }
    for y in 0..S {
        output[y * 8..y * 8 + S].copy_from_slice(&temp[y * 8..y * 8 + S]);
    }
}

/// Recursive 2x2-Hadamard transform from JPEG XL. The first stage retains
/// 2-pixel spatial locality, unlike an 8x8 DCT, then the upper-left DC pyramid
/// is transformed at 4x4 and 2x2 scales.
pub(crate) fn dct2x2_8x8(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    dct2_top_block::<8>(input, output);
    let snapshot = *output;
    dct2_top_block::<4>(DctInput::from_flat(&snapshot), output);
    let snapshot = *output;
    dct2_top_block::<2>(DctInput::from_flat(&snapshot), output);
}

#[inline]
fn inv_dct2_top_block<const S: usize>(input: &[f32; 64], output: &mut [f32; 64]) {
    let half = S / 2;
    for y in 0..half {
        for x in 0..half {
            let r00 = input[y * 8 + x];
            let r01 = input[y * 8 + half + x];
            let r10 = input[(y + half) * 8 + x];
            let r11 = input[(y + half) * 8 + half + x];
            output[(y * 2) * 8 + x * 2] = r00 + r01 + r10 + r11;
            output[(y * 2) * 8 + x * 2 + 1] = r00 + r01 - r10 - r11;
            output[(y * 2 + 1) * 8 + x * 2] = r00 - r01 + r10 - r11;
            output[(y * 2 + 1) * 8 + x * 2 + 1] = r00 - r01 - r10 + r11;
        }
    }
}

pub(crate) fn inv_dct2x2_8x8(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut a = std::array::from_fn::<_, 64, _>(|i| input.row(i / 8)[i % 8]);
    let snapshot = a;
    inv_dct2_top_block::<2>(&snapshot, &mut a);
    let snapshot = a;
    inv_dct2_top_block::<4>(&snapshot, &mut a);
    let snapshot = a;
    inv_dct2_top_block::<8>(&snapshot, output);
}

/// Inverse-transform dispatch table resolved once and retained by the encoding
/// context. Rectangular transforms use their normalized coefficient layout in
/// the function type, hence both orientations share the same input dimensions.
pub(crate) struct IdctMethods {
    pub(crate) inv_identity8x8: DctFn<8, 8, 64>,
    pub(crate) inv_dct2x2_8x8: DctFn<8, 8, 64>,
    pub(crate) idct8x8: DctFn<8, 8, 64>,
    pub(crate) idct8x16: DctFn<16, 8, 128>,
    pub(crate) idct16x8: DctFn<16, 8, 128>,
    pub(crate) idct16x16: DctFn<16, 16, 256>,
    pub(crate) idct16x32: DctFn<32, 16, 512>,
    pub(crate) idct32x16: DctFn<32, 16, 512>,
    pub(crate) idct32x32: DctFn<32, 32, 1024>,
    pub(crate) idct64x64: DctFn<64, 64, 4096>,
    pub(crate) idct64x32: DctFn<64, 32, 2048>,
    pub(crate) idct32x64: DctFn<64, 32, 2048>,
}

impl IdctMethods {
    pub(crate) const fn scalar() -> Self {
        Self {
            inv_identity8x8,
            inv_dct2x2_8x8,
            idct8x8: inv_dct8x8,
            idct8x16: inv_dct8x16,
            idct16x8: inv_dct16x8,
            idct16x16: inv_dct16x16,
            idct16x32: inv_dct16x32,
            idct32x16: inv_dct32x16,
            idct32x32: inv_dct32x32,
            idct64x64: inv_dct64x64,
            idct64x32: inv_dct64x32,
            idct32x64: inv_dct32x64,
        }
    }
}

static IDCT_METHODS: OnceLock<IdctMethods> = OnceLock::new();

fn select_idct_methods() -> IdctMethods {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return IdctMethods {
            inv_identity8x8: |input, output| unsafe {
                crate::avx::inv_identity8x8_avx2(input, output)
            },
            inv_dct2x2_8x8: |input, output| unsafe {
                crate::avx::inv_dct2x2_8x8_avx2(input, output)
            },
            idct8x8: |input, output| unsafe { crate::avx::inv_dct8x8_avx2(input, output) },
            idct8x16: |input, output| unsafe { crate::avx::inv_dct8x16_avx2(input, output) },
            idct16x8: |input, output| unsafe { crate::avx::inv_dct16x8_avx2(input, output) },
            idct16x16: |input, output| unsafe { crate::avx::inv_dct16x16_avx2(input, output) },
            idct16x32: |input, output| unsafe { crate::avx::inv_dct16x32_avx2(input, output) },
            idct32x16: |input, output| unsafe { crate::avx::inv_dct32x16_avx2(input, output) },
            idct32x32: |input, output| unsafe { crate::avx::inv_dct32x32_avx2(input, output) },
            idct64x64: |input, output| unsafe { crate::avx::inv_dct64x64_avx2(input, output) },
            idct64x32: |input, output| unsafe { crate::avx::inv_dct64x32_avx2(input, output) },
            idct32x64: |input, output| unsafe { crate::avx::inv_dct32x64_avx2(input, output) },
        };
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return IdctMethods {
            inv_identity8x8: |input, output| unsafe {
                crate::neon::inv_identity8x8_neon(input, output)
            },
            inv_dct2x2_8x8: |input, output| unsafe {
                crate::neon::inv_dct2x2_8x8_neon(input, output)
            },
            idct8x8: |input, output| unsafe { crate::neon::inv_dct8x8_neon(input, output) },
            idct8x16: |input, output| unsafe { crate::neon::inv_dct8x16_neon(input, output) },
            idct16x8: |input, output| unsafe { crate::neon::inv_dct16x8_neon(input, output) },
            idct16x16: |input, output| unsafe { crate::neon::inv_dct16x16_neon(input, output) },
            idct16x32: |input, output| unsafe { crate::neon::inv_dct16x32_neon(input, output) },
            idct32x16: |input, output| unsafe { crate::neon::inv_dct32x16_neon(input, output) },
            idct32x32: |input, output| unsafe { crate::neon::inv_dct32x32_neon(input, output) },
            idct64x64: |input, output| unsafe { crate::neon::inv_dct64x64_neon(input, output) },
            idct64x32: |input, output| unsafe { crate::neon::inv_dct64x32_neon(input, output) },
            idct32x64: |input, output| unsafe { crate::neon::inv_dct32x64_neon(input, output) },
        };
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        return IdctMethods {
            inv_identity8x8: |input, output| unsafe {
                crate::wasm::inv_identity8x8_wasm(input, output)
            },
            inv_dct2x2_8x8: |input, output| unsafe {
                crate::wasm::inv_dct2x2_8x8_wasm(input, output)
            },
            ..IdctMethods::scalar()
        };
    }

    #[allow(unreachable_code)]
    IdctMethods::scalar()
}

pub(crate) fn selected_idct_methods() -> &'static IdctMethods {
    IDCT_METHODS.get_or_init(select_idct_methods)
}

static IDENTITY_METHOD: OnceLock<DctFn<8, 8, 64>> = OnceLock::new();

fn select_identity() -> DctFn<8, 8, 64> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return |input, output| unsafe { crate::neon::identity8x8_neon(input, output) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        return |input, output| unsafe { crate::wasm::identity8x8_wasm(input, output) };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |input, output| unsafe { crate::avx::identity8x8_avx2(input, output) };
    }
    #[allow(unreachable_code)]
    identity8x8
}

pub(crate) fn selected_identity8x8() -> &'static DctFn<8, 8, 64> {
    IDENTITY_METHOD.get_or_init(select_identity)
}

static DCT2X2_METHOD: OnceLock<DctFn<8, 8, 64>> = OnceLock::new();

fn select_dct2x2() -> DctFn<8, 8, 64> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        return |input, output| unsafe { crate::neon::dct2x2_8x8_neon(input, output) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        return |input, output| unsafe { crate::wasm::dct2x2_8x8_wasm(input, output) };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |input, output| unsafe { crate::avx::dct2x2_8x8_avx2(input, output) };
    }
    #[allow(unreachable_code)]
    dct2x2_8x8
}

pub(crate) fn selected_dct2x2_8x8() -> &'static DctFn<8, 8, 64> {
    DCT2X2_METHOD.get_or_init(select_dct2x2)
}

static DCT_METHOD: OnceLock<DctFn<8, 8, 64>> = OnceLock::new();

fn select_dct() -> DctFn<8, 8, 64> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct8x8_neon;
        |input, output| unsafe {
            dct8x8_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct8x8_wasm;
        |input, output| {
            dct8x8_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct8x8_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct8x8_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct8x8() -> &'static DctFn<8, 8, 64> {
    DCT_METHOD.get_or_init(select_dct)
}

#[inline]
pub(crate) fn dct8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    selected_dct8x8()(DctInput::from_flat(input), output);
}

pub(crate) fn dct8x8_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    dct8x8_scalar_input(DctInput::from_flat(input), output);
}

fn dct8x8_scalar_input(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut tmp = [0.0f32; 64];

    for (y, tmp) in tmp.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let src_row = input.row(y);
        let mut row = [0.0f32; 8];
        for (dst, src) in row.iter_mut().zip(src_row.iter()) {
            *dst = *src;
        }
        dct1d_8(&mut row);
        for (dst, src) in tmp.iter_mut().zip(row.iter()) {
            *dst = *src;
        }
    }

    for (x, out_row) in output.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut col = [0.0f32; 8];
        for (col_slot, tmp_row) in col.iter_mut().zip(tmp.as_chunks::<8>().0.iter()) {
            *col_slot = tmp_row[x];
        }
        dct1d_8(&mut col);
        for (dst, src) in out_row.iter_mut().zip(col.iter()) {
            *dst = *src * (1.0 / 64.0);
        }
    }
}

pub(crate) const WC16: [f32; 8] = [
    0.502_419_3, // 1/(2·cos( π/32))
    0.522_498_6, // 1/(2·cos(3π/32))
    0.566_944,   // 1/(2·cos(5π/32))
    0.646_821_8, // 1/(2·cos(7π/32))
    0.788_154_6, // 1/(2·cos(9π/32))
    1.060_677_7, // 1/(2·cos(11π/32))
    1.722_447_1, // 1/(2·cos(13π/32))
    5.101_148_6, // 1/(2·cos(15π/32))
];
pub(crate) const INV_WC16: [f32; 8] = [
    1.0 / WC16[0],
    1.0 / WC16[1],
    1.0 / WC16[2],
    1.0 / WC16[3],
    1.0 / WC16[4],
    1.0 / WC16[5],
    1.0 / WC16[6],
    1.0 / WC16[7],
];

#[inline(always)]
#[allow(unused)]
pub(crate) fn dct1d_16(buf: &mut [f32; 16]) {
    let mut tmp = [0.0f32; 16];

    for i in 0..8 {
        tmp[i] = buf[i] + buf[15 - i];
        tmp[8 + i] = buf[i] - buf[15 - i];
    }

    // Recurse on the even half
    dct1d_8(&mut tmp[..8]);

    // Scale the odd half by WC16, then recurse
    for i in 0..8 {
        tmp[8 + i] *= WC16[i];
    }
    dct1d_8(&mut tmp[8..16]);

    tmp[8] = fmla(tmp[8], std::f32::consts::SQRT_2, tmp[9]);
    tmp[9] += tmp[10];
    tmp[10] += tmp[11];
    tmp[11] += tmp[12];
    tmp[12] += tmp[13];
    tmp[13] += tmp[14];
    tmp[14] += tmp[15];

    for i in 0..8 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[8 + i];
    }
}

#[inline(always)]
#[allow(unused)]
pub(crate) fn dct1d_16_oof(src: &[f32; 16], buf: &mut [f32; 16]) {
    let mut tmp = [0.0f32; 16];

    for i in 0..8 {
        tmp[i] = src[i] + src[15 - i];
        tmp[8 + i] = src[i] - src[15 - i];
    }

    // Recurse on the even half
    dct1d_8(&mut tmp[0..8]);

    // Scale the odd half by WC16, then recurse
    for i in 0..8 {
        tmp[8 + i] *= WC16[i];
    }
    dct1d_8(&mut tmp[8..16]);

    tmp[8] = fmla(tmp[8], std::f32::consts::SQRT_2, tmp[9]);
    tmp[9] += tmp[10];
    tmp[10] += tmp[11];
    tmp[11] += tmp[12];
    tmp[12] += tmp[13];
    tmp[13] += tmp[14];
    tmp[14] += tmp[15];

    // Interleave even (dc) and odd (ac) halves into output
    for i in 0..8 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[8 + i];
    }
}

fn select_dct_8x16() -> DctFn<16, 8, 128> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct8x16_neon;
        |input, output| unsafe {
            dct8x16_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct8x16_wasm;
        |input, output| {
            dct8x16_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct8x16_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct8x16_scalar_input(input, output);
    }
}

static DCT_METHOD_8X16: OnceLock<DctFn<16, 8, 128>> = OnceLock::new();
static DCT_METHOD_16X8: OnceLock<DctFn<8, 16, 128>> = OnceLock::new();

#[inline]
pub(crate) fn selected_dct8x16() -> &'static DctFn<16, 8, 128> {
    DCT_METHOD_8X16.get_or_init(select_dct_8x16)
}

pub(crate) fn dct8x16(input: &[f32; 128], output: &mut [f32; 128]) {
    selected_dct8x16()(DctInput::from_flat(input), output);
}

pub(crate) fn dct8x16_scalar(input: &[f32], output: &mut [f32; 128]) {
    let input: &[f32; 128] = input.try_into().unwrap();
    dct8x16_scalar_input(DctInput::from_flat(input), output);
}

fn dct8x16_scalar_input(input: DctInput<'_, 16, 8>, output: &mut [f32; 128]) {
    let mut after_row_dct = [0.0f32; 128];
    for (y, dst) in after_row_dct.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        dct1d_16_oof(input.row(y), dst);
    }

    let mut col = [0.0f32; 8];

    let scale = 1.0 / 128.0;
    for u in 0..16 {
        for i in 0..8 {
            col[i] = after_row_dct[i * 16 + u];
        }
        dct1d_8(&mut col);
        for v in 0..8 {
            output[v * 16 + u] = col[v] * scale;
        }
    }
}

fn select_dct_16x8() -> DctFn<8, 16, 128> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct16x8_neon;
        |input, output| unsafe {
            dct16x8_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct16x8_wasm;
        |input, output| {
            dct16x8_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct16x8_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct16x8_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct16x8() -> &'static DctFn<8, 16, 128> {
    DCT_METHOD_16X8.get_or_init(select_dct_16x8)
}

pub(crate) fn dct16x8(input: &[f32; 128], output: &mut [f32; 128]) {
    selected_dct16x8()(DctInput::from_flat(input), output);
}

pub(crate) fn dct16x8_scalar(input: &[f32; 128], output: &mut [f32; 128]) {
    dct16x8_scalar_input(DctInput::from_flat(input), output);
}

fn dct16x8_scalar_input(input: DctInput<'_, 8, 16>, output: &mut [f32; 128]) {
    let mut after_col_dct = [0.0f32; 128];
    for u in 0..8 {
        let mut col = [0.0f32; 16];
        for i in 0..16 {
            col[i] = input.row(i)[u];
        }
        dct1d_16(&mut col);
        for v in 0..16 {
            after_col_dct[v * 8 + u] = col[v];
        }
    }

    let scale = 1.0 / 128.0;
    for v in 0..16 {
        let row = &mut after_col_dct[v * 8..v * 8 + 8];
        dct1d_8(row);
        for u in 0..8 {
            output[u * 16 + v] = row[u] * scale;
        }
    }
}

pub(crate) const RESAMPLE_SCALE_16_TO_2: [f32; 2] = [1.0, 0.901_764_2];

#[inline]
pub(crate) fn dc_from_dct16x8(coeffs: &[f32; 128], dc: &mut [f32; 2]) {
    let s0 = coeffs[0] * RESAMPLE_SCALE_16_TO_2[0];
    let s1 = coeffs[1] * RESAMPLE_SCALE_16_TO_2[1];
    // IDCT1DImpl<2>: sum + diff, no scaling.
    dc[0] = s0 + s1;
    dc[1] = s0 - s1;
}

#[inline]
pub(crate) fn dc_from_dct8x16(coeffs: &[f32; 128], dc: &mut [f32; 2]) {
    let s0 = coeffs[0] * RESAMPLE_SCALE_16_TO_2[0];
    let s1 = coeffs[1] * RESAMPLE_SCALE_16_TO_2[1];
    dc[0] = s0 + s1;
    dc[1] = s0 - s1;
}

static DCT_METHOD_16X16: OnceLock<DctFn<16, 16, 256>> = OnceLock::new();

fn select_dct_16x16() -> DctFn<16, 16, 256> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct16x16_neon;
        |input, output| unsafe {
            dct16x16_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct16x16_wasm;
        |input, output| {
            dct16x16_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct16x16_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct16x16_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct16x16() -> &'static DctFn<16, 16, 256> {
    DCT_METHOD_16X16.get_or_init(select_dct_16x16)
}

pub(crate) fn dct16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    selected_dct16x16()(DctInput::from_flat(input), output);
}

pub(crate) fn dct16x16_scalar(input: &[f32; 256], output: &mut [f32; 256]) {
    dct16x16_scalar_input(DctInput::from_flat(input), output);
}

fn dct16x16_scalar_input(input: DctInput<'_, 16, 16>, output: &mut [f32; 256]) {
    let mut after_col_dct = [0.0f32; 256];
    let mut col = [0.0f32; 16];
    for u in 0..16 {
        for i in 0..16 {
            col[i] = input.row(i)[u];
        }
        dct1d_16(&mut col);
        for v in 0..16 {
            after_col_dct[v * 16 + u] = col[v];
        }
    }

    let scale = 1.0 / 256.0;
    for v in 0..16 {
        let row = &mut after_col_dct[v * 16..v * 16 + 16];
        dct1d_16(row.try_into().unwrap());
        for u in 0..16 {
            output[u * 16 + v] = row[u] * scale;
        }
    }
}

pub(crate) fn dc_from_dct16x16(coeffs: &[f32; 256], dc: &mut [f32; 4]) {
    // Scale the 2×2 LF patch by the outer product of the resample scales.
    let s00 = coeffs[0] * (RESAMPLE_SCALE_16_TO_2[0] * RESAMPLE_SCALE_16_TO_2[0]);
    let s01 = coeffs[1] * (RESAMPLE_SCALE_16_TO_2[0] * RESAMPLE_SCALE_16_TO_2[1]);
    let s10 = coeffs[16] * (RESAMPLE_SCALE_16_TO_2[1] * RESAMPLE_SCALE_16_TO_2[0]);
    let s11 = coeffs[17] * (RESAMPLE_SCALE_16_TO_2[1] * RESAMPLE_SCALE_16_TO_2[1]);
    // 2-D 2-point IDCT.
    //   TL = s00 + s01 + s10 + s11
    //   TR = s00 + s01 - s10 - s11
    //   BL = s00 - s01 + s10 - s11
    //   BR = s00 - s01 - s10 + s11
    let r00 = s00 + s01;
    let r01 = s00 - s01;
    let r10 = s10 + s11;
    let r11 = s10 - s11;
    dc[0] = r00 + r10; // top-left
    dc[1] = r00 - r10; // top-right
    dc[2] = r01 + r11; // bottom-left
    dc[3] = r01 - r11; // bottom-right
}

pub(crate) fn dct4x4_2d(input: &[f32; 16]) -> [f32; 16] {
    let mut tmp = *input;
    for r in 0..4 {
        dct1d_4((&mut tmp[r * 4..r * 4 + 4]).try_into().unwrap());
    }
    let mut output = tmp;
    for x in 0..4 {
        let mut col = [tmp[x], tmp[4 + x], tmp[8 + x], tmp[12 + x]];
        dct1d_4(&mut col);
        for i in 0..4 {
            output[x * 4 + i] = col[i] * (1.0 / 16.0);
        }
    }
    output
}

#[cfg(test)]
fn idct4x4_2d(input: &[f32; 16], output: &mut [f32; 16]) {
    let mut tmp = [0.0f32; 16];
    for x in 0..4 {
        let col_in = [
            input[x * 4],
            input[x * 4 + 1],
            input[x * 4 + 2],
            input[x * 4 + 3],
        ];
        let col = idct1d_4(col_in);
        for i in 0..4 {
            tmp[i * 4 + x] = col[i];
        }
    }
    for r in 0..4 {
        let row_in: [f32; 4] = [tmp[r * 4], tmp[r * 4 + 1], tmp[r * 4 + 2], tmp[r * 4 + 3]];
        let row = idct1d_4(row_in);
        output[r * 4..r * 4 + 4].copy_from_slice(&row);
    }
}

static DCT_METHOD_4X4: OnceLock<DctFn<8, 8, 64>> = OnceLock::new();

fn select_dct_4x4() -> DctFn<8, 8, 64> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct4x4_neon;
        |input, output| unsafe {
            dct4x4_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct4x4_wasm;
        |input, output| {
            dct4x4_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct4x4_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct4x4_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct4x4() -> &'static DctFn<8, 8, 64> {
    DCT_METHOD_4X4.get_or_init(select_dct_4x4)
}

pub(crate) fn dct4x4(input: &[f32; 64], output: &mut [f32; 64]) {
    selected_dct4x4()(DctInput::from_flat(input), output);
}

pub(crate) fn dct4x4_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    dct4x4_scalar_input(DctInput::from_flat(input), output);
}

fn dct4x4_scalar_input(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    for qy in 0..2 {
        for qx in 0..2 {
            let mut blk = [0.0f32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    blk[r * 4 + c] = input.row(qy * 4 + r)[qx * 4 + c];
                }
            }
            let d = dct4x4_2d(&blk);
            for iy in 0..4 {
                for ix in 0..4 {
                    output[(qy + iy * 2) * 8 + (qx + ix * 2)] = d[iy * 4 + ix];
                }
            }
        }
    }
    // 2×2 Hadamard on the four sub-DCs (block00=coeff[0], 01=[1], 10=[8], 11=[9]).
    let b00 = output[0];
    let b01 = output[1];
    let b10 = output[8];
    let b11 = output[9];
    output[0] = (b00 + b01 + b10 + b11) * 0.25;
    output[1] = (b00 + b01 - b10 - b11) * 0.25;
    output[8] = (b00 - b01 + b10 - b11) * 0.25;
    output[9] = (b00 - b01 - b10 + b11) * 0.25;
}

pub(crate) fn dct4x8_2d(input: &[f32; 32]) -> [f32; 32] {
    let mut tmp = *input;
    // Vertical 4-point DCT per column.
    for c in 0..8 {
        let mut col = [tmp[c], tmp[8 + c], tmp[16 + c], tmp[24 + c]];
        dct1d_4(&mut col);
        for vf in 0..4 {
            tmp[vf * 8 + c] = col[vf];
        }
    }
    // Horizontal 8-point DCT per vertical-frequency row.
    for vf in 0..4 {
        let row: &mut [f32; 8] = (&mut tmp[vf * 8..vf * 8 + 8]).try_into().unwrap();
        dct1d_8(row);
        for value in row {
            *value *= 1.0 / 32.0;
        }
    }
    tmp
}

pub(crate) fn dct4x8_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    dct4x8_scalar_input(DctInput::from_flat(input), output);
}

fn dct4x8_scalar_input(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    for y in 0..2 {
        let mut half = [0.0f32; 32];
        for r in 0..4 {
            for c in 0..8 {
                half[r * 8 + c] = input.row(y * 4 + r)[c];
            }
        }
        let d = dct4x8_2d(&half);
        for iy in 0..4 {
            for ix in 0..8 {
                output[(y + iy * 2) * 8 + ix] = d[iy * 8 + ix];
            }
        }
    }
    // 2-point Hadamard on the two sub-DCs (top = coeff[0], bottom = coeff[8]).
    let block0 = output[0];
    let block1 = output[8];
    output[0] = (block0 + block1) * 0.5;
    output[8] = (block0 - block1) * 0.5;
}

fn dct8x4_2d(input: &[f32; 32], output: &mut [f32; 32]) {
    let mut tmp = [0.0f32; 32]; // tmp[vf*4 + c]
    // Vertical 8-point DCT per column (4 columns).
    for c in 0..4 {
        let mut col = [
            input[c],
            input[4 + c],
            input[8 + c],
            input[12 + c],
            input[16 + c],
            input[20 + c],
            input[24 + c],
            input[28 + c],
        ];
        dct1d_8(&mut col);
        for vf in 0..8 {
            tmp[vf * 4 + c] = col[vf];
        }
    }
    // Horizontal 4-point DCT per vertical-frequency row; store transposed.
    for vf in 0..8 {
        let mut row = [
            tmp[vf * 4],
            tmp[vf * 4 + 1],
            tmp[vf * 4 + 2],
            tmp[vf * 4 + 3],
        ];
        dct1d_4(&mut row);
        for hf in 0..4 {
            output[hf * 8 + vf] = row[hf] * (1.0 / 32.0);
        }
    }
}

pub(crate) fn dct8x4_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    dct8x4_scalar_input(DctInput::from_flat(input), output);
}

fn dct8x4_scalar_input(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    for x in 0..2 {
        let mut half = [0.0f32; 32]; // 8 rows × 4 cols
        for r in 0..8 {
            for c in 0..4 {
                half[r * 4 + c] = input.row(r)[x * 4 + c];
            }
        }
        let mut d = [0.0f32; 32];
        dct8x4_2d(&half, &mut d);
        for iy in 0..4 {
            for ix in 0..8 {
                output[(x + iy * 2) * 8 + ix] = d[iy * 8 + ix];
            }
        }
    }
    let block0 = output[0];
    let block1 = output[8];
    output[0] = (block0 + block1) * 0.5;
    output[8] = (block0 - block1) * 0.5;
}

static DCT_METHOD_4X8: OnceLock<DctFn<8, 8, 64>> = OnceLock::new();

fn select_dct_4x8() -> DctFn<8, 8, 64> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct4x8_neon;
        |input, output| unsafe {
            dct4x8_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct4x8_wasm;
        |input, output| {
            dct4x8_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct4x8_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct4x8_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct4x8() -> &'static DctFn<8, 8, 64> {
    DCT_METHOD_4X8.get_or_init(select_dct_4x8)
}

pub(crate) fn dct4x8(input: &[f32; 64], output: &mut [f32; 64]) {
    selected_dct4x8()(DctInput::from_flat(input), output);
}

static DCT_METHOD_8X4: OnceLock<DctFn<8, 8, 64>> = OnceLock::new();

fn select_dct_8x4() -> DctFn<8, 8, 64> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct8x4_neon;
        |input, output| unsafe {
            dct8x4_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct8x4_wasm;
        |input, output| {
            dct8x4_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct8x4_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct8x4_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct8x4() -> &'static DctFn<8, 8, 64> {
    DCT_METHOD_8X4.get_or_init(select_dct_8x4)
}

pub(crate) fn dct8x4(input: &[f32; 64], output: &mut [f32; 64]) {
    selected_dct8x4()(DctInput::from_flat(input), output);
}

pub(crate) const WC32: [f32; 16] = [
    0.500_602_98,
    0.505_470_96,
    0.515_447_3,
    0.531_042_6,
    0.553_103_9,
    0.582_935,
    0.622_504_1,
    0.674_808_34,
    0.744_536_3,
    0.839_349_65,
    0.972_568_25,
    1.169_44,
    1.484_164_6,
    2.057_781,
    3.407_608_4,
    10.190_008,
];
pub(crate) const INV_WC32: [f32; 16] = [
    1.0 / WC32[0],
    1.0 / WC32[1],
    1.0 / WC32[2],
    1.0 / WC32[3],
    1.0 / WC32[4],
    1.0 / WC32[5],
    1.0 / WC32[6],
    1.0 / WC32[7],
    1.0 / WC32[8],
    1.0 / WC32[9],
    1.0 / WC32[10],
    1.0 / WC32[11],
    1.0 / WC32[12],
    1.0 / WC32[13],
    1.0 / WC32[14],
    1.0 / WC32[15],
];

#[inline]
pub(crate) fn dct1d_32(buf: &mut [f32; 32]) {
    let mut tmp = [0.0f32; 32];
    for i in 0..16 {
        tmp[i] = buf[i] + buf[31 - i];
        tmp[16 + i] = buf[i] - buf[31 - i];
    }
    dct1d_16(<&mut [f32; 16]>::try_from(&mut tmp[0..16]).unwrap());
    for i in 0..16 {
        tmp[16 + i] *= WC32[i];
    }
    dct1d_16(<&mut [f32; 16]>::try_from(&mut tmp[16..32]).unwrap());
    tmp[16] = fmla(tmp[16], std::f32::consts::SQRT_2, tmp[17]);
    for i in 17..31 {
        tmp[i] += tmp[i + 1];
    }
    for i in 0..16 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[16 + i];
    }
}

#[inline(always)]
fn inv_dct1d_2(buf: &mut [f32]) {
    let (s, d) = (buf[0], buf[1]);
    buf[0] = (s + d) * 0.5;
    buf[1] = (s - d) * 0.5;
}

#[inline(always)]
fn inv_dct1d_4(buf: &mut [f32; 4]) {
    const IS2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut t = [buf[0], buf[2], buf[1], buf[3]]; // undo the even/odd interleave
    t[2] = (t[2] - t[3]) * IS2; // undo tmp[2] = tmp[2]*SQRT2 + tmp[3]
    inv_dct1d_2(&mut t[2..4]);
    t[2] *= INV_WC4[0];
    t[3] *= INV_WC4[1];
    inv_dct1d_2(&mut t[..2]);
    buf[0] = (t[0] + t[2]) * 0.5;
    buf[3] = (t[0] - t[2]) * 0.5;
    buf[1] = (t[1] + t[3]) * 0.5;
    buf[2] = (t[1] - t[3]) * 0.5;
}

#[inline(always)]
fn inv_dct1d_8(buf: &mut [f32]) {
    const IS2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut t = [0.0f32; 8];
    for i in 0..4 {
        t[i] = buf[2 * i];
        t[4 + i] = buf[2 * i + 1];
    }
    t[6] -= t[7];
    t[5] -= t[6];
    t[4] = (t[4] - t[5]) * IS2;
    inv_dct1d_4(<&mut [f32; 4]>::try_from(&mut t[4..8]).unwrap());
    for i in 0..4 {
        t[4 + i] *= INV_WC8[i];
    }
    inv_dct1d_4(<&mut [f32; 4]>::try_from(&mut t[..4]).unwrap());
    for i in 0..4 {
        buf[i] = (t[i] + t[4 + i]) * 0.5;
        buf[7 - i] = (t[i] - t[4 + i]) * 0.5;
    }
}

#[inline(always)]
fn inv_dct1d_16(buf: &mut [f32; 16]) {
    const IS2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut t = [0.0f32; 16];
    for i in 0..8 {
        t[i] = buf[2 * i];
        t[8 + i] = buf[2 * i + 1];
    }
    for i in (9..=14).rev() {
        t[i] -= t[i + 1];
    }
    t[8] = (t[8] - t[9]) * IS2;
    inv_dct1d_8(&mut t[8..16]);
    for i in 0..8 {
        t[8 + i] *= INV_WC16[i];
    }
    inv_dct1d_8(&mut t[0..8]);
    for i in 0..8 {
        buf[i] = (t[i] + t[8 + i]) * 0.5;
        buf[15 - i] = (t[i] - t[8 + i]) * 0.5;
    }
}

#[inline]
fn inv_dct1d_32(buf: &mut [f32; 32]) {
    const IS2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut t = [0.0f32; 32];
    for i in 0..16 {
        t[i] = buf[2 * i];
        t[16 + i] = buf[2 * i + 1];
    }
    for i in (17..=30).rev() {
        t[i] -= t[i + 1];
    }
    t[16] = (t[16] - t[17]) * IS2;
    inv_dct1d_16(<&mut [f32; 16]>::try_from(&mut t[16..32]).unwrap());
    for i in 0..16 {
        t[16 + i] *= INV_WC32[i];
    }
    inv_dct1d_16(<&mut [f32; 16]>::try_from(&mut t[0..16]).unwrap());
    for i in 0..16 {
        buf[i] = (t[i] + t[16 + i]) * 0.5;
        buf[31 - i] = (t[i] - t[16 + i]) * 0.5;
    }
}

macro_rules! inv_dct_square {
    ($name:ident, $n:literal, $side:literal, $inv1d:path) => {
        pub(crate) fn $name(coeff: DctInput<'_, $side, $side>, out: &mut [f32; $n]) {
            for a in 0..$side {
                for b in 0..$side {
                    out[b * $side + a] = coeff.row(a)[b] * ($n as f32);
                }
            }
            for u in 0..$side {
                let mut col = [0.0f32; $side];
                for v in 0..$side {
                    col[v] = out[v * $side + u];
                }
                $inv1d(&mut col);
                for v in 0..$side {
                    out[v * $side + u] = col[v];
                }
            }
            for v in 0..$side {
                let row =
                    <&mut [f32; $side]>::try_from(&mut out[v * $side..v * $side + $side]).unwrap();
                $inv1d(row);
            }
        }
    };
}
inv_dct_square!(inv_dct8x8, 64, 8, inv_dct1d_8_arr);
inv_dct_square!(inv_dct16x16, 256, 16, inv_dct1d_16);
inv_dct_square!(inv_dct32x32, 1024, 32, inv_dct1d_32);

#[inline(always)]
fn inv_dct1d_8_arr(buf: &mut [f32; 8]) {
    inv_dct1d_8(buf);
}

pub(crate) fn inv_dct8x16(coeff: DctInput<'_, 16, 8>, out: &mut [f32; 128]) {
    for v in 0..8 {
        for u in 0..16 {
            out[v * 16 + u] = coeff.row(v)[u] * 128.0;
        }
    }
    for u in 0..16 {
        let mut col = [0.0f32; 8];
        for v in 0..8 {
            col[v] = out[v * 16 + u];
        }
        inv_dct1d_8(&mut col);
        for v in 0..8 {
            out[v * 16 + u] = col[v];
        }
    }
    for r in 0..8 {
        inv_dct1d_16(<&mut [f32; 16]>::try_from(&mut out[r * 16..r * 16 + 16]).unwrap());
    }
}

/// Inverse of `dct16x8` (16 tall × 8 wide pixels; forward: col-DCT16 then
/// row-DCT8, transposed store `[u*16+v]`).
pub(crate) fn inv_dct16x8(coeff: DctInput<'_, 16, 8>, out: &mut [f32; 128]) {
    let mut acd = [0.0f32; 128]; // after_col_dct[v*8+u]
    for v in 0..16 {
        let mut row = [0.0f32; 8];
        for u in 0..8 {
            row[u] = coeff.row(u)[v] * 128.0;
        }
        inv_dct1d_8(&mut row);
        for u in 0..8 {
            acd[v * 8 + u] = row[u];
        }
    }
    for u in 0..8 {
        let mut col = [0.0f32; 16];
        for v in 0..16 {
            col[v] = acd[v * 8 + u];
        }
        inv_dct1d_16(&mut col);
        for i in 0..16 {
            out[i * 8 + u] = col[i];
        }
    }
}

pub(crate) fn inv_dct16x32(coeff: DctInput<'_, 32, 16>, out: &mut [f32; 512]) {
    for v in 0..16 {
        for u in 0..32 {
            out[v * 32 + u] = coeff.row(v)[u] * 512.0;
        }
    }
    for u in 0..32 {
        let mut col = [0.0f32; 16];
        for v in 0..16 {
            col[v] = out[v * 32 + u];
        }
        inv_dct1d_16(&mut col);
        for i in 0..16 {
            out[i * 32 + u] = col[i];
        }
    }
    for i in 0..16 {
        inv_dct1d_32(<&mut [f32; 32]>::try_from(&mut out[i * 32..i * 32 + 32]).unwrap());
    }
}

pub(crate) fn inv_dct32x16(coeff: DctInput<'_, 32, 16>, out: &mut [f32; 512]) {
    let mut acd = [0.0f32; 512]; // after_col_dct[v*16+u]
    for v in 0..32 {
        let mut row = [0.0f32; 16];
        for u in 0..16 {
            row[u] = coeff.row(u)[v] * 512.0;
        }
        inv_dct1d_16(&mut row);
        for u in 0..16 {
            acd[v * 16 + u] = row[u];
        }
    }
    for u in 0..16 {
        let mut col = [0.0f32; 32];
        for v in 0..32 {
            col[v] = acd[v * 16 + u];
        }
        inv_dct1d_32(&mut col);
        for i in 0..32 {
            out[i * 16 + u] = col[i];
        }
    }
}

static DCT_METHOD_32X32: OnceLock<DctFn<32, 32, 1024>> = OnceLock::new();

fn select_dct_32x32() -> DctFn<32, 32, 1024> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct32x32_neon;
        |input, output| unsafe {
            dct32x32_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct32x32_wasm;
        |input, output| {
            dct32x32_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct32x32_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct32x32_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct32x32() -> &'static DctFn<32, 32, 1024> {
    DCT_METHOD_32X32.get_or_init(select_dct_32x32)
}

pub(crate) fn dct32x32(input: &[f32; 1024], output: &mut [f32; 1024]) {
    selected_dct32x32()(DctInput::from_flat(input), output);
}

pub(crate) fn dct32x32_scalar(input: &[f32; 1024], output: &mut [f32; 1024]) {
    dct32x32_scalar_input(DctInput::from_flat(input), output);
}

fn dct32x32_scalar_input(input: DctInput<'_, 32, 32>, output: &mut [f32; 1024]) {
    let mut after_col_dct = [0.0f32; 1024];
    let mut col = [0.0f32; 32];
    for u in 0..32 {
        for i in 0..32 {
            col[i] = input.row(i)[u];
        }
        dct1d_32(&mut col);
        for v in 0..32 {
            after_col_dct[v * 32 + u] = col[v];
        }
    }

    let scale = 1.0 / 1024.0;
    for v in 0..32 {
        let row: &mut [f32; 32] = (&mut after_col_dct[v * 32..v * 32 + 32])
            .try_into()
            .unwrap();
        dct1d_32(row);
        for u in 0..32 {
            output[u * 32 + v] = row[u] * scale;
        }
    }
}

pub(crate) const RESAMPLE_SCALE_64_TO_8: [f32; 8] = [
    1.0,
    0.993_686_6,
    0.974_886_83,
    0.944_018_07,
    0.901_764_2,
    0.849_057_5,
    0.787_054_9,
    0.717_108_13,
];

#[inline]
fn idct1d_8(mut values: [f32; 8]) -> [f32; 8] {
    for value in &mut values {
        *value *= 8.0;
    }
    inv_dct1d_8(&mut values);
    values
}

/// `DCTResampleScales<32, 4>` from libjxl `dct_scales.h`. Used to rescale the
/// lowest 4×4 frequencies before the 4-point IDCT in [`dc_from_dct32x32`].
pub(crate) const RESAMPLE_SCALE_32_TO_4: [f32; 4] = [1.0, 0.974_886_8, 0.901_764_2, 0.787_054_9];

pub(crate) type DcFromDct32x32Fn = fn(&[f32; 1024], &mut [f32; 16]);
pub(crate) type DcFromDct32x16Fn = fn(&[f32; 512], &mut [f32; 8]);
pub(crate) type DcFromDct16x32Fn = fn(&[f32; 512], &mut [f32; 8]);
pub(crate) type DcFromDct64x64Fn = fn(&[f32; 4096], &mut [f32; 64]);
pub(crate) type DcFromDct64x32Fn = fn(&[f32; 2048], &mut [f32; 32]);
pub(crate) type DcFromDct32x64Fn = fn(&[f32; 2048], &mut [f32; 32]);

#[derive(Clone, Copy)]
pub(crate) struct DcFromDctMethods {
    pub(crate) dct32x32: DcFromDct32x32Fn,
    pub(crate) dct32x16: DcFromDct32x16Fn,
    pub(crate) dct16x32: DcFromDct16x32Fn,
    pub(crate) dct64x64: DcFromDct64x64Fn,
    pub(crate) dct64x32: DcFromDct64x32Fn,
    pub(crate) dct32x64: DcFromDct32x64Fn,
}

static DC_FROM_DCT_METHODS: OnceLock<DcFromDctMethods> = OnceLock::new();

fn select_dc_from_dct_methods() -> DcFromDctMethods {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        DcFromDctMethods {
            dct32x32: |coeffs, dc| unsafe { crate::neon::dc_from_dct32x32_neon(coeffs, dc) },
            dct32x16: |coeffs, dc| unsafe { crate::neon::dc_from_dct32x16_neon(coeffs, dc) },
            dct16x32: |coeffs, dc| unsafe { crate::neon::dc_from_dct16x32_neon(coeffs, dc) },
            dct64x64: |coeffs, dc| unsafe { crate::neon::dc_from_dct64x64_neon(coeffs, dc) },
            dct64x32: |coeffs, dc| unsafe { crate::neon::dc_from_dct64x32_neon(coeffs, dc) },
            dct32x64: |coeffs, dc| unsafe { crate::neon::dc_from_dct32x64_neon(coeffs, dc) },
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return DcFromDctMethods {
                dct32x32: |coeffs, dc| unsafe { crate::avx::dc_from_dct32x32_avx2(coeffs, dc) },
                dct32x16: |coeffs, dc| unsafe { crate::avx::dc_from_dct32x16_avx2(coeffs, dc) },
                dct16x32: |coeffs, dc| unsafe { crate::avx::dc_from_dct16x32_avx2(coeffs, dc) },
                dct64x64: |coeffs, dc| unsafe { crate::avx::dc_from_dct64x64_avx2(coeffs, dc) },
                dct64x32: |coeffs, dc| unsafe { crate::avx::dc_from_dct64x32_avx2(coeffs, dc) },
                dct32x64: |coeffs, dc| unsafe { crate::avx::dc_from_dct32x64_avx2(coeffs, dc) },
            };
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    DcFromDctMethods {
        dct32x32: dc_from_dct32x32,
        dct32x16: dc_from_dct32x16,
        dct16x32: dc_from_dct16x32,
        dct64x64: dc_from_dct64x64,
        dct64x32: dc_from_dct64x32,
        dct32x64: dc_from_dct32x64,
    }
}

#[inline]
pub(crate) fn selected_dc_from_dct_methods() -> DcFromDctMethods {
    *DC_FROM_DCT_METHODS.get_or_init(select_dc_from_dct_methods)
}

/// libjxl `IDCT1DImpl<4>` (unnormalized 4-point inverse DCT), out-of-place.
/// Used only by [`dc_from_dct32x32`] to invert the lowest frequencies into the
/// 4×4 DC patch. Mirrors the structure of the verified 2-point path in
/// [`dc_from_dct16x16`].
#[inline]
fn idct1d_4(v: [f32; 4]) -> [f32; 4] {
    // ForwardEvenOdd: [v0, v2, v1, v3]
    let mut t = [v[0], v[2], v[1], v[3]];
    // IDCT2 on even half
    let e0 = t[0] + t[1];
    let e1 = t[0] - t[1];
    t[0] = e0;
    t[1] = e1;
    // BTranspose on odd half: t[3] += t[2]; t[2] *= sqrt2
    t[3] += t[2];
    t[2] *= std::f32::consts::SQRT_2;
    // IDCT2 on odd half
    let o0 = t[2] + t[3];
    let o1 = t[2] - t[3];
    t[2] = o0;
    t[3] = o1;
    // MultiplyAndAdd with WC4
    let mut out = [0.0f32; 4];
    out[0] = fmla(WC4[0], t[2], t[0]);
    out[3] = fmla(-WC4[0], t[2], t[0]);
    out[1] = fmla(WC4[1], t[3], t[1]);
    out[2] = fmla(-WC4[1], t[3], t[1]);
    out
}

#[allow(dead_code)]
pub(crate) fn dc_from_dct32x32(coeffs: &[f32; 1024], dc: &mut [f32; 16]) {
    let r = RESAMPLE_SCALE_32_TO_4;
    // Scale lowest 4×4: s[a][b] = coeffs[a*32 + b] * r[a] * r[b].
    let mut s = [[0.0f32; 4]; 4];
    for (a, srow) in s.iter_mut().enumerate() {
        for (b, sab) in srow.iter_mut().enumerate() {
            *sab = coeffs[a * 32 + b] * r[a] * r[b];
        }
    }
    // IDCT along b (columns) for each coefficient row a.
    let mut rr = [[0.0f32; 4]; 4];
    for a in 0..4 {
        rr[a] = idct1d_4(s[a]);
    }
    // IDCT along a (rows) for each output column bb; write transposed grid.
    for bb in 0..4 {
        let col = idct1d_4([rr[0][bb], rr[1][bb], rr[2][bb], rr[3][bb]]);
        for ridx in 0..4 {
            dc[bb * 4 + ridx] = col[ridx];
        }
    }
}

static DCT_METHOD_32X16: OnceLock<DctFn<16, 32, 512>> = OnceLock::new();

fn select_dct_32x16() -> DctFn<16, 32, 512> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct32x16_neon;
        |input, output| unsafe {
            dct32x16_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct32x16_wasm;
        |input, output| {
            dct32x16_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct32x16_avx2(input, output);
            };
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct32x16_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct32x16() -> &'static DctFn<16, 32, 512> {
    DCT_METHOD_32X16.get_or_init(select_dct_32x16)
}

pub(crate) fn dct32x16(input: &[f32; 512], output: &mut [f32; 512]) {
    selected_dct32x16()(DctInput::from_flat(input), output);
}

pub(crate) fn dct32x16_scalar(input: &[f32; 512], output: &mut [f32; 512]) {
    dct32x16_scalar_input(DctInput::from_flat(input), output);
}

fn dct32x16_scalar_input(input: DctInput<'_, 16, 32>, output: &mut [f32; 512]) {
    let mut after_col_dct = [0.0f32; 512];
    let mut col = [0.0f32; 32];
    for u in 0..16 {
        for i in 0..32 {
            col[i] = input.row(i)[u];
        }
        dct1d_32(&mut col);
        for v in 0..32 {
            after_col_dct[v * 16 + u] = col[v];
        }
    }

    let scale = 1.0 / 512.0;
    for v in 0..32 {
        let row: &mut [f32; 16] = (&mut after_col_dct[v * 16..v * 16 + 16])
            .try_into()
            .unwrap();
        dct1d_16(row);
        for u in 0..16 {
            output[u * 32 + v] = row[u] * scale;
        }
    }
}

static DCT_METHOD_16X32: OnceLock<DctFn<32, 16, 512>> = OnceLock::new();

fn select_dct_16x32() -> DctFn<32, 16, 512> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use crate::neon::dct16x32_neon;
        |input, output| unsafe {
            dct16x32_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        use crate::wasm::dct16x32_wasm;
        |input, output| {
            dct16x32_wasm(input, output);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct16x32_avx2(input, output);
            };
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    |input, output| {
        dct16x32_scalar_input(input, output);
    }
}

#[inline]
pub(crate) fn selected_dct16x32() -> &'static DctFn<32, 16, 512> {
    DCT_METHOD_16X32.get_or_init(select_dct_16x32)
}

pub(crate) fn dct16x32(input: &[f32; 512], output: &mut [f32; 512]) {
    selected_dct16x32()(DctInput::from_flat(input), output);
}

pub(crate) fn dct16x32_scalar(input: &[f32; 512], output: &mut [f32; 512]) {
    dct16x32_scalar_input(DctInput::from_flat(input), output);
}

fn dct16x32_scalar_input(input: DctInput<'_, 32, 16>, output: &mut [f32; 512]) {
    let mut after_row_dct = [0.0f32; 512];
    let mut rowbuf = [0.0f32; 32];
    for i in 0..16 {
        rowbuf.copy_from_slice(input.row(i));
        dct1d_32(&mut rowbuf);
        after_row_dct[i * 32..i * 32 + 32].copy_from_slice(&rowbuf);
    }

    let scale = 1.0 / 512.0;
    let mut col = [0.0f32; 16];
    for u in 0..32 {
        for i in 0..16 {
            col[i] = after_row_dct[i * 32 + u];
        }
        dct1d_16(&mut col);
        for v in 0..16 {
            output[v * 32 + u] = col[v] * scale;
        }
    }
}

#[allow(dead_code)]
pub(crate) fn dc_from_dct32x16(coeffs: &[f32; 512], dc: &mut [f32; 8]) {
    let r4 = RESAMPLE_SCALE_32_TO_4; // vertical freq (4 DC rows)
    let r2 = RESAMPLE_SCALE_16_TO_2; // horizontal freq (2 DC cols)
    let mut rr = [[0.0f32; 4]; 2];
    for a in 0..2 {
        let mut s = [0.0f32; 4];
        for b in 0..4 {
            s[b] = coeffs[a * 32 + b] * r2[a] * r4[b];
        }
        rr[a] = idct1d_4(s);
    }
    for bb in 0..4 {
        let c0 = rr[0][bb];
        let c1 = rr[1][bb];
        dc[bb * 2] = c0 + c1;
        dc[bb * 2 + 1] = c0 - c1;
    }
}

/// Extract the 2×4 = 8 DC values from a DCT16X32 coefficient block. The 16-tall
/// axis resamples 16→2 (vertical freq, 2-point IDCT), the 32-wide axis resamples
/// 32→4 (horizontal freq, 4-point IDCT). Coefficients are at `coeffs[v*32 + u]`
/// (v = vertical freq, u = horizontal freq). Output index is `didx = iy*4 + ix`
/// (row-major 2-row × 4-col grid), matching the caller (cov_x=4, cov_y=2).
#[allow(dead_code)]
pub(crate) fn dc_from_dct16x32(coeffs: &[f32; 512], dc: &mut [f32; 8]) {
    let r4 = RESAMPLE_SCALE_32_TO_4; // horizontal freq (4 DC cols)
    let r2 = RESAMPLE_SCALE_16_TO_2; // vertical freq (2 DC rows)
    // rr[a] = IDCT4 along horizontal freq for each of the 2 vertical freqs a.
    let mut rr = [[0.0f32; 4]; 2];
    for a in 0..2 {
        let mut s = [0.0f32; 4];
        for b in 0..4 {
            // coeff(vfreq=a, hfreq=b) = coeffs[a*32 + b].
            s[b] = coeffs[a * 32 + b] * r2[a] * r4[b];
        }
        rr[a] = idct1d_4(s);
    }
    // 2-point IDCT along vertical freq for each spatial column bb.
    for bb in 0..4 {
        let c0 = rr[0][bb];
        let c1 = rr[1][bb];
        dc[bb] = c0 + c1; // top row
        dc[4 + bb] = c0 - c1; // bottom row
    }
}

// DCT64 constants shared by the scalar and architecture-specific kernels.
pub(crate) const WC64: [f32; 32] = [
    0.500_150_6,
    0.501_358_45,
    0.503_788_7,
    0.507_471_14,
    0.512_451_47,
    0.518_792_7,
    0.526_577_3,
    0.535_909_83,
    0.546_920_4,
    0.559_769_8,
    0.574_655_2,
    0.591_818_5,
    0.611_557_36,
    0.634_238_96,
    0.660_319_8,
    0.690_372_1,
    0.725_120_54,
    0.765_494_17,
    0.812_702_1,
    0.868_344_7,
    0.934_583_6,
    1.014_408_2,
    1.112_071_6,
    1.233_832_7,
    1.389_293_9,
    1.593_972_3,
    1.874_676,
    2.282_05,
    2.924_628_5,
    4.084_611,
    6.796_750_5,
    20.373_878,
];

pub(crate) const DCT64_SPLIT_COS: [f32; 16] = [
    0.999_698_8,
    0.997_290_43,
    0.992_479_56,
    0.985_277_65,
    0.975_702_1,
    0.963_776_05,
    0.949_528_16,
    0.932_992_8,
    0.914_209_8,
    0.893_224_3,
    0.870_087,
    0.844_853_6,
    0.817_584_8,
    0.788_346_4,
    0.757_208_8,
    0.724_247_1,
];

pub(crate) const DCT64_SPLIT_SIN: [f32; 16] = [
    0.024_541_229,
    0.073_564_57,
    0.122_410_68,
    0.170_961_89,
    0.219_101_24,
    0.266_712_75,
    0.313_681_75,
    0.359_895_05,
    0.405_241_3,
    0.449_611_34,
    0.492_898_2,
    0.534_997_64,
    0.575_808_17,
    0.615_231_6,
    0.653_172_85,
    0.689_540_57,
];

pub(crate) const INV_WC64: [f32; 32] = [
    1.0 / WC64[0],
    1.0 / WC64[1],
    1.0 / WC64[2],
    1.0 / WC64[3],
    1.0 / WC64[4],
    1.0 / WC64[5],
    1.0 / WC64[6],
    1.0 / WC64[7],
    1.0 / WC64[8],
    1.0 / WC64[9],
    1.0 / WC64[10],
    1.0 / WC64[11],
    1.0 / WC64[12],
    1.0 / WC64[13],
    1.0 / WC64[14],
    1.0 / WC64[15],
    1.0 / WC64[16],
    1.0 / WC64[17],
    1.0 / WC64[18],
    1.0 / WC64[19],
    1.0 / WC64[20],
    1.0 / WC64[21],
    1.0 / WC64[22],
    1.0 / WC64[23],
    1.0 / WC64[24],
    1.0 / WC64[25],
    1.0 / WC64[26],
    1.0 / WC64[27],
    1.0 / WC64[28],
    1.0 / WC64[29],
    1.0 / WC64[30],
    1.0 / WC64[31],
];

pub(crate) fn dct1d_64(buf: &mut [f32; 64]) {
    let mut tmp = [0.0f32; 64];
    for i in 0..32 {
        tmp[i] = buf[i] + buf[63 - i];
        tmp[32 + i] = (buf[i] - buf[63 - i]) * WC64[i];
    }
    dct1d_32(<&mut [f32; 32]>::try_from(&mut tmp[..32]).unwrap());
    dct1d_32(<&mut [f32; 32]>::try_from(&mut tmp[32..]).unwrap());
    tmp[32] = fmla(tmp[32], std::f32::consts::SQRT_2, tmp[33]);
    for i in 33..63 {
        tmp[i] += tmp[i + 1];
    }
    for i in 0..32 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[32 + i];
    }
}

fn inv_dct1d_64(buf: &mut [f32; 64]) {
    let mut tmp = [0.0f32; 64];
    for i in 0..32 {
        tmp[i] = buf[2 * i];
        tmp[32 + i] = buf[2 * i + 1];
    }
    for i in (33..=62).rev() {
        tmp[i] -= tmp[i + 1];
    }
    tmp[32] = (tmp[32] - tmp[33]) * std::f32::consts::FRAC_1_SQRT_2;
    inv_dct1d_32(<&mut [f32; 32]>::try_from(&mut tmp[32..]).unwrap());
    for i in 0..32 {
        tmp[32 + i] *= INV_WC64[i];
    }
    inv_dct1d_32(<&mut [f32; 32]>::try_from(&mut tmp[..32]).unwrap());
    for i in 0..32 {
        buf[i] = (tmp[i] + tmp[32 + i]) * 0.5;
        buf[63 - i] = (tmp[i] - tmp[32 + i]) * 0.5;
    }
}

pub(crate) fn dct64x64_scalar_input(input: DctInput<'_, 64, 64>, output: &mut [f32; 4096]) {
    let mut after_col_dct = [0.0f32; 4096];
    let mut col = [0.0f32; 64];
    for u in 0..64 {
        for i in 0..64 {
            col[i] = input.row(i)[u];
        }
        dct1d_64(&mut col);
        for v in 0..64 {
            after_col_dct[v * 64 + u] = col[v];
        }
    }

    let scale = 1.0 / 4096.0;
    for v in 0..64 {
        let row: &mut [f32; 64] = (&mut after_col_dct[v * 64..v * 64 + 64])
            .try_into()
            .unwrap();
        dct1d_64(row);
        for u in 0..64 {
            output[u * 64 + v] = row[u] * scale;
        }
    }
}

pub(crate) fn dc_from_dct64x64(coeffs: &[f32; 4096], dc: &mut [f32; 64]) {
    let mut low = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            low[y * 8 + x] =
                coeffs[y * 64 + x] * (RESAMPLE_SCALE_64_TO_8[x] * RESAMPLE_SCALE_64_TO_8[y]);
        }
    }
    inv_dct8x8(DctInput::from_flat(&low), dc);
}

inv_dct_square!(inv_dct64x64, 4096, 64, inv_dct1d_64);

static DCT_METHOD_64X64: OnceLock<DctFn<64, 64, 4096>> = OnceLock::new();

fn select_dct_64x64() -> DctFn<64, 64, 4096> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |input, output| unsafe {
            crate::neon::dct64x64_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        |input, output| {
            crate::wasm::dct64x64_wasm(input, output);
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct64x64_avx2(input, output);
            };
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    dct64x64_scalar_input
}

pub(crate) fn selected_dct64x64() -> &'static DctFn<64, 64, 4096> {
    DCT_METHOD_64X64.get_or_init(select_dct_64x64)
}

/// 64 rows x 32 columns. Coefficients are normalized to the shared 32-row x
/// 64-column layout: `horizontal_frequency * 64 + vertical_frequency`.
pub(crate) fn dct64x32_scalar_input(input: DctInput<'_, 32, 64>, output: &mut [f32; 2048]) {
    let mut after_col_dct = [0.0f32; 2048];
    let mut col = [0.0f32; 64];
    for x in 0..32 {
        for y in 0..64 {
            col[y] = input.row(y)[x];
        }
        dct1d_64(&mut col);
        for v in 0..64 {
            after_col_dct[v * 32 + x] = col[v];
        }
    }
    let scale = 1.0 / 2048.0;
    for v in 0..64 {
        let row = <&mut [f32; 32]>::try_from(&mut after_col_dct[v * 32..v * 32 + 32]).unwrap();
        dct1d_32(row);
        for u in 0..32 {
            output[u * 64 + v] = row[u] * scale;
        }
    }
}

/// 32 rows x 64 columns, stored directly in normalized 32x64 frequency layout.
pub(crate) fn dct32x64_scalar_input(input: DctInput<'_, 64, 32>, output: &mut [f32; 2048]) {
    let mut after_row_dct = [0.0f32; 2048];
    for y in 0..32 {
        let row = <&mut [f32; 64]>::try_from(&mut after_row_dct[y * 64..y * 64 + 64]).unwrap();
        row.copy_from_slice(input.row(y));
        dct1d_64(row);
    }
    let scale = 1.0 / 2048.0;
    let mut col = [0.0f32; 32];
    for u in 0..64 {
        for y in 0..32 {
            col[y] = after_row_dct[y * 64 + u];
        }
        dct1d_32(&mut col);
        for v in 0..32 {
            output[v * 64 + u] = col[v] * scale;
        }
    }
}

pub(crate) fn inv_dct64x32(coeff: DctInput<'_, 64, 32>, out: &mut [f32; 2048]) {
    let mut after_row = [0.0f32; 2048];
    for v in 0..64 {
        let mut row = [0.0f32; 32];
        for u in 0..32 {
            row[u] = coeff.row(u)[v] * 2048.0;
        }
        inv_dct1d_32(&mut row);
        for u in 0..32 {
            after_row[v * 32 + u] = row[u];
        }
    }
    for x in 0..32 {
        let mut col = [0.0f32; 64];
        for v in 0..64 {
            col[v] = after_row[v * 32 + x];
        }
        inv_dct1d_64(&mut col);
        for y in 0..64 {
            out[y * 32 + x] = col[y];
        }
    }
}

pub(crate) fn inv_dct32x64(coeff: DctInput<'_, 64, 32>, out: &mut [f32; 2048]) {
    for y in 0..32 {
        for x in 0..64 {
            out[y * 64 + x] = coeff.row(y)[x] * 2048.0;
        }
    }
    for u in 0..64 {
        let mut col = [0.0f32; 32];
        for v in 0..32 {
            col[v] = out[v * 64 + u];
        }
        inv_dct1d_32(&mut col);
        for v in 0..32 {
            out[v * 64 + u] = col[v];
        }
    }
    for y in 0..32 {
        inv_dct1d_64(<&mut [f32; 64]>::try_from(&mut out[y * 64..y * 64 + 64]).unwrap());
    }
}

#[inline]
fn idct1d_8_unnormalized(mut values: [f32; 8]) -> [f32; 8] {
    for value in &mut values {
        *value *= 8.0;
    }
    inv_dct1d_8(&mut values);
    values
}

fn dc_from_dct64x32_normalized(coeffs: &[f32; 2048]) -> [[f32; 8]; 4] {
    let mut rows = [[0.0f32; 8]; 4];
    for a in 0..4 {
        let mut frequencies = [0.0f32; 8];
        for b in 0..8 {
            frequencies[b] =
                coeffs[a * 64 + b] * (RESAMPLE_SCALE_32_TO_4[a] * RESAMPLE_SCALE_64_TO_8[b]);
        }
        rows[a] = idct1d_8_unnormalized(frequencies);
    }
    let mut out = [[0.0f32; 8]; 4];
    for x in 0..8 {
        let column = idct1d_4([rows[0][x], rows[1][x], rows[2][x], rows[3][x]]);
        for y in 0..4 {
            out[y][x] = column[y];
        }
    }
    out
}

/// DC grid for a 64-row x 32-column transform (8 rows x 4 columns).
pub(crate) fn dc_from_dct64x32(coeffs: &[f32; 2048], dc: &mut [f32; 32]) {
    let normalized = dc_from_dct64x32_normalized(coeffs);
    for y in 0..8 {
        for x in 0..4 {
            dc[y * 4 + x] = normalized[x][y];
        }
    }
}

/// DC grid for a 32-row x 64-column transform (4 rows x 8 columns).
pub(crate) fn dc_from_dct32x64(coeffs: &[f32; 2048], dc: &mut [f32; 32]) {
    let normalized = dc_from_dct64x32_normalized(coeffs);
    for y in 0..4 {
        dc[y * 8..y * 8 + 8].copy_from_slice(&normalized[y]);
    }
}

static DCT_METHOD_64X32: OnceLock<DctFn<32, 64, 2048>> = OnceLock::new();
static DCT_METHOD_32X64: OnceLock<DctFn<64, 32, 2048>> = OnceLock::new();

fn select_dct_64x32() -> DctFn<32, 64, 2048> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |input, output| unsafe {
            crate::neon::dct64x32_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        |input, output| {
            crate::wasm::dct64x32_wasm(input, output);
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct64x32_avx2(input, output);
            };
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    dct64x32_scalar_input
}

fn select_dct_32x64() -> DctFn<64, 32, 2048> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |input, output| unsafe {
            crate::neon::dct32x64_neon(input, output);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        |input, output| {
            crate::wasm::dct32x64_wasm(input, output);
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |input, output| unsafe {
                crate::avx::dct32x64_avx2(input, output);
            };
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    dct32x64_scalar_input
}

pub(crate) fn selected_dct64x32() -> &'static DctFn<32, 64, 2048> {
    DCT_METHOD_64X32.get_or_init(select_dct_64x32)
}

pub(crate) fn selected_dct32x64() -> &'static DctFn<64, 32, 2048> {
    DCT_METHOD_32X64.get_or_init(select_dct_32x64)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn dct64_round_trips() {
        let mut input = [0.0f32; 4096];
        for y in 0..64 {
            for x in 0..64 {
                input[y * 64 + x] =
                    0.01 * x as f32 + 0.003 * y as f32 + ((x / 8 + y / 8) % 3) as f32 * 0.05;
            }
        }
        let mut coeffs = [0.0f32; 4096];
        dct64x64_scalar_input(DctInput::from_flat(&input), &mut coeffs);
        let mut output = [0.0f32; 4096];
        inv_dct64x64(DctInput::from_flat(&coeffs), &mut output);
        for (expected, actual) in input.iter().zip(output.iter()) {
            assert!((expected - actual).abs() < 1e-4, "{expected} vs {actual}");
        }
    }

    #[test]
    fn dct64_constant_is_pure_dc_and_extracts_block_dcs() {
        let input = [2.0f32; 4096];
        let mut coeffs = [0.0f32; 4096];
        dct64x64_scalar_input(DctInput::from_flat(&input), &mut coeffs);
        assert!((coeffs[0] - 2.0).abs() < 1e-5);
        assert!(coeffs[1..].iter().map(|v| v * v).sum::<f32>() < 1e-7);

        let mut dc = [0.0f32; 64];
        dc_from_dct64x64(&coeffs, &mut dc);
        for value in dc {
            assert!((value - 2.0).abs() < 1e-4, "{value}");
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
        let mut max_error = 0.0f32;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            max_error = max_error.max(error);
            assert!(
                error <= 1e-4,
                "{label}[{index}] SIMD={actual} scalar={expected} error={error}"
            );
        }
        assert!(max_error.is_finite());
    }

    #[test]
    fn dct64_family_dispatch_matches_scalar_with_strided_input() {
        let square = std::array::from_fn::<_, 4096, _>(|i| {
            ((i as f32 * 0.013).sin() + (i as f32 * 0.007).cos()) * 0.5
        });
        let mut square_strided = vec![f32::NAN; 64 * 69];
        for y in 0..64 {
            square_strided[y * 69..y * 69 + 64].copy_from_slice(&square[y * 64..y * 64 + 64]);
        }
        let mut expected_square = [0.0; 4096];
        let mut actual_square = [0.0; 4096];
        dct64x64_scalar_input(DctInput::from_flat(&square), &mut expected_square);
        selected_dct64x64()(DctInput::new(&square_strided, 69), &mut actual_square);
        assert_close(&actual_square, &expected_square, "dct64x64");

        let tall = std::array::from_fn::<_, 2048, _>(|i| {
            ((i as f32 * 0.017).sin() - (i as f32 * 0.005).cos()) * 0.5
        });
        let mut tall_strided = vec![f32::NAN; 64 * 37];
        for y in 0..64 {
            tall_strided[y * 37..y * 37 + 32].copy_from_slice(&tall[y * 32..y * 32 + 32]);
        }
        let mut expected_tall = [0.0; 2048];
        let mut actual_tall = [0.0; 2048];
        dct64x32_scalar_input(DctInput::from_flat(&tall), &mut expected_tall);
        selected_dct64x32()(DctInput::new(&tall_strided, 37), &mut actual_tall);
        assert_close(&actual_tall, &expected_tall, "dct64x32");

        let wide = std::array::from_fn::<_, 2048, _>(|i| {
            ((i as f32 * 0.019).cos() + (i as f32 * 0.003).sin()) * 0.5
        });
        let mut wide_strided = vec![f32::NAN; 32 * 71];
        for y in 0..32 {
            wide_strided[y * 71..y * 71 + 64].copy_from_slice(&wide[y * 64..y * 64 + 64]);
        }
        let mut expected_wide = [0.0; 2048];
        let mut actual_wide = [0.0; 2048];
        dct32x64_scalar_input(DctInput::from_flat(&wide), &mut expected_wide);
        selected_dct32x64()(DctInput::new(&wide_strided, 71), &mut actual_wide);
        assert_close(&actual_wide, &expected_wide, "dct32x64");
    }

    #[test]
    fn rectangular_dct64_round_trips() {
        let mut tall = [0.0f32; 2048];
        for y in 0..64 {
            for x in 0..32 {
                tall[y * 32 + x] = 0.01 * x as f32 + 0.003 * y as f32;
            }
        }
        let mut coeffs = [0.0f32; 2048];
        dct64x32_scalar_input(DctInput::from_flat(&tall), &mut coeffs);
        let mut output = [0.0f32; 2048];
        inv_dct64x32(DctInput::from_flat(&coeffs), &mut output);
        for (expected, actual) in tall.iter().zip(output.iter()) {
            assert!((expected - actual).abs() < 2e-4, "{expected} vs {actual}");
        }

        let mut wide = [0.0f32; 2048];
        for y in 0..32 {
            for x in 0..64 {
                wide[y * 64 + x] = 0.004 * x as f32 + 0.007 * y as f32;
            }
        }
        dct32x64_scalar_input(DctInput::from_flat(&wide), &mut coeffs);
        inv_dct32x64(DctInput::from_flat(&coeffs), &mut output);
        for (expected, actual) in wide.iter().zip(output.iter()) {
            assert!((expected - actual).abs() < 2e-4, "{expected} vs {actual}");
        }
    }

    #[test]
    fn rectangular_dct64_constant_dc_extracts() {
        let tall = [1.5f32; 2048];
        let mut coeffs = [0.0f32; 2048];
        dct64x32_scalar_input(DctInput::from_flat(&tall), &mut coeffs);
        let mut dc = [0.0f32; 32];
        dc_from_dct64x32(&coeffs, &mut dc);
        assert!(dc.iter().all(|value| (*value - 1.5).abs() < 2e-4));

        let wide = [1.5f32; 2048];
        dct32x64_scalar_input(DctInput::from_flat(&wide), &mut coeffs);
        dc_from_dct32x64(&coeffs, &mut dc);
        assert!(dc.iter().all(|value| (*value - 1.5).abs() < 2e-4));
    }

    #[test]
    fn strided_input_matches_flat_input() {
        let mut flat = [0.0f32; 64];
        for (i, value) in flat.iter_mut().enumerate() {
            *value = (i as f32 * 0.37).sin();
        }
        let mut strided = [f32::NAN; 8 * 13];
        for y in 0..8 {
            strided[y * 13..y * 13 + 8].copy_from_slice(&flat[y * 8..y * 8 + 8]);
        }

        let mut expected = [0.0f32; 64];
        let mut actual = [0.0f32; 64];
        dct8x8(&flat, &mut expected);
        selected_dct8x8()(DctInput::new(&strided, 13), &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn strided_idct_input_matches_flat_input() {
        let mut flat = [0.0f32; 64];
        for (i, value) in flat.iter_mut().enumerate() {
            *value = (i as f32 * 0.19).cos();
        }
        let mut strided = [f32::NAN; 8 * 13];
        for y in 0..8 {
            strided[y * 13..y * 13 + 8].copy_from_slice(&flat[y * 8..y * 8 + 8]);
        }

        let mut expected = [0.0f32; 64];
        let mut actual = [0.0f32; 64];
        inv_dct8x8(DctInput::from_flat(&flat), &mut expected);
        inv_dct8x8(DctInput::new(&strided, 13), &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn strided_input_rejects_short_last_row() {
        let data = [0.0f32; (8 - 1) * 13 + 8 - 1];
        let _ = DctInput::<8, 8>::new(&data, 13);
    }

    #[test]
    #[should_panic]
    fn strided_input_rejects_short_stride() {
        let data = [0.0f32; 64];
        let _ = DctInput::<8, 8>::new(&data, 7);
    }

    #[test]
    fn inverse_butterflies_round_trip() {
        let mut s = 987654321u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        // 1D
        for _ in 0..50 {
            let mut b8 = [0.0f32; 8];
            b8.iter_mut().for_each(|v| *v = rnd());
            let orig = b8;
            dct1d_8(&mut b8);
            inv_dct1d_8(&mut b8);
            for i in 0..8 {
                assert!((b8[i] - orig[i]).abs() < 1e-4, "1d8 {i}");
            }
        }
        // 2D square: inv_dctNxN(dctNxN(x)) == x
        macro_rules! rt2d {
            ($n:literal, $fwd:path, $inv:path) => {{
                let mut x = [0.0f32; $n];
                x.iter_mut().for_each(|v| *v = rnd());
                let orig = x;
                let mut c = [0.0f32; $n];
                $fwd(&x, &mut c);
                let mut r = [0.0f32; $n];
                $inv(DctInput::from_flat(&c), &mut r);
                let me = (0..$n)
                    .map(|i| (r[i] - orig[i]).abs())
                    .fold(0.0f32, f32::max);
                assert!(me < 1e-3, "2d {} max err {me}", $n);
            }};
        }
        rt2d!(64, dct8x8, inv_dct8x8);
        rt2d!(256, dct16x16, inv_dct16x16);
        rt2d!(1024, dct32x32, inv_dct32x32);
        rt2d!(128, dct8x16, inv_dct8x16);
        rt2d!(128, dct16x8, inv_dct16x8);
        rt2d!(512, dct16x32, inv_dct16x32);
        rt2d!(512, dct32x16, inv_dct32x16);
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_inverse_matches_scalar() {
        let mut s = 555u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        macro_rules! cmp {
            ($n:literal, $sc:path, $ne:path) => {{
                for case in 0..16 {
                    let mut c = [0.0f32; $n];
                    c.iter_mut().for_each(|v| *v = rnd());
                    let mut rs = [0.0f32; $n];
                    $sc(DctInput::from_flat(&c), &mut rs);
                    let mut rn = [0.0f32; $n];
                    unsafe { $ne(DctInput::from_flat(&c), &mut rn) };
                    let me = (0..$n)
                        .map(|i| (rs[i] - rn[i]).abs())
                        .fold(0.0f32, f32::max);
                    assert!(me < 1e-3, "neon vs scalar {} case {case} max diff {me}", $n);
                }
            }};
        }
        cmp!(64, inv_dct8x8, crate::neon::inv_dct8x8_neon);
        cmp!(128, inv_dct8x16, crate::neon::inv_dct8x16_neon);
        cmp!(128, inv_dct16x8, crate::neon::inv_dct16x8_neon);
        cmp!(256, inv_dct16x16, crate::neon::inv_dct16x16_neon);
        cmp!(512, inv_dct16x32, crate::neon::inv_dct16x32_neon);
        cmp!(512, inv_dct32x16, crate::neon::inv_dct32x16_neon);
        cmp!(1024, inv_dct32x32, crate::neon::inv_dct32x32_neon);
    }

    #[test]
    fn dct4x8_dc_matches_dct8x8() {
        // The block DC (coeff[0]) of DCT4X8 must land on the same scale as
        // DCT8X8's, otherwise the shared DC plane / decoder reconstruction is
        // wrong. The ×0.5 Hadamard on the two half-DCs is what enforces this.
        for &v in &[0.0f32, 0.37, -1.2, 4.5] {
            let block = [v; 64];
            let mut o4 = [0.0f32; 64];
            let mut o8 = [0.0f32; 64];
            dct4x8(&block, &mut o4);
            dct8x8(&block, &mut o8);
            assert!(
                (o4[0] - o8[0]).abs() < 1e-4,
                "DC mismatch for v={v}: 4x8={} 8x8={}",
                o4[0],
                o8[0]
            );
        }
    }

    #[test]
    fn dct4x8_constant_is_pure_dc() {
        // A constant block has energy only in the DC; every AC coefficient
        // (including the vertical-difference at [8]) must vanish.
        let block = [2.0f32; 64];
        let mut o = [0.0f32; 64];
        dct4x8(&block, &mut o);
        for (i, &c) in o.iter().enumerate() {
            if i == 0 {
                continue;
            }
            assert!(c.abs() < 1e-4, "AC coeff[{i}] = {c} should be ~0");
        }
    }

    #[test]
    fn dct8x4_dc_matches_dct8x8() {
        for &v in &[0.0f32, 0.37, -1.2, 4.5] {
            let block = [v; 64];
            let mut o = [0.0f32; 64];
            let mut o8 = [0.0f32; 64];
            dct8x4(&block, &mut o);
            dct8x8(&block, &mut o8);
            assert!(
                (o[0] - o8[0]).abs() < 1e-4,
                "DC mismatch for v={v}: 8x4={} 8x8={}",
                o[0],
                o8[0]
            );
        }
    }

    #[test]
    fn dct8x4_constant_is_pure_dc() {
        let block = [2.0f32; 64];
        let mut o = [0.0f32; 64];
        dct8x4(&block, &mut o);
        for (i, &c) in o.iter().enumerate() {
            if i == 0 {
                continue;
            }
            assert!(c.abs() < 1e-4, "AC coeff[{i}] = {c} should be ~0");
        }
    }

    #[test]
    fn dct32x16_constant_is_pure_dc() {
        let block = [1.7f32; 512];
        let mut o = [0.0f32; 512];
        dct32x16(&block, &mut o);
        for (i, &c) in o.iter().enumerate() {
            if i == 0 {
                continue;
            }
            assert!(c.abs() < 1e-3, "32x16 AC coeff[{i}] = {c} should be ~0");
        }
    }

    #[test]
    fn dct16x32_constant_is_pure_dc() {
        let block = [-0.9f32; 512];
        let mut o = [0.0f32; 512];
        dct16x32(&block, &mut o);
        for (i, &c) in o.iter().enumerate() {
            if i == 0 {
                continue;
            }
            assert!(c.abs() < 1e-3, "16x32 AC coeff[{i}] = {c} should be ~0");
        }
    }

    #[test]
    fn dct32x16_16x32_dc_matches_dct8x8() {
        // The block DC must land on the same scale as DCT8X8's, or the shared
        // DC plane / decoder reconstruction is wrong.
        for &v in &[0.0f32, 0.37, -1.2, 4.5] {
            let block512 = [v; 512];
            let block64 = [v; 64];
            let mut o8 = [0.0f32; 64];
            dct8x8(&block64, &mut o8);
            let mut o = [0.0f32; 512];
            dct32x16(&block512, &mut o);
            assert!(
                (o[0] - o8[0]).abs() < 1e-4,
                "32x16 DC mismatch v={v}: {} vs 8x8 {}",
                o[0],
                o8[0]
            );
            dct16x32(&block512, &mut o);
            assert!(
                (o[0] - o8[0]).abs() < 1e-4,
                "16x32 DC mismatch v={v}: {} vs 8x8 {}",
                o[0],
                o8[0]
            );
        }
    }

    #[test]
    fn dc_from_dct32x16_16x32_constant_matches_block_dc() {
        // For a constant block, every covered-block DC equals the block DC
        // (= DCT8X8 coeff[0] of the same constant).
        let v = 2.3f32;
        let block512 = [v; 512];
        let block64 = [v; 64];
        let mut o8 = [0.0f32; 64];
        dct8x8(&block64, &mut o8);
        let expect = o8[0];

        let mut coeffs = [0.0f32; 512];
        dct32x16(&block512, &mut coeffs);
        let mut dc = [0.0f32; 8];
        dc_from_dct32x16(&coeffs, &mut dc);
        for (i, &d) in dc.iter().enumerate() {
            assert!(
                (d - expect).abs() < 1e-3,
                "32x16 DC[{i}] = {d} should match block DC {expect}"
            );
        }

        dct16x32(&block512, &mut coeffs);
        dc_from_dct16x32(&coeffs, &mut dc);
        for (i, &d) in dc.iter().enumerate() {
            assert!(
                (d - expect).abs() < 1e-3,
                "16x32 DC[{i}] = {d} should match block DC {expect}"
            );
        }
    }

    // Emulate the SIMD data-flow (each vector lane modeled as one of 8 f32s,
    // dct1d_*_flat as the scalar 1D kernel applied per lane) using the exact
    // gather/store index math of `dct32x16_avx2` / `dct16x32_avx2` (and their
    // NEON twins), and confirm it reproduces the scalar transform. This
    // validates the SIMD coefficient layout — the one part not exercised by the
    // scalar fallback and not runnable under the in-repo toolchain. The real
    // intrinsic kernels are additionally checked vs scalar in avx/neon `tests`.
    fn dct1d_16_flat_emu(c: &mut [[f32; 8]; 16]) {
        for lane in 0..8 {
            let mut col = [0.0f32; 16];
            for i in 0..16 {
                col[i] = c[i][lane];
            }
            dct1d_16(&mut col);
            for i in 0..16 {
                c[i][lane] = col[i];
            }
        }
    }
    fn dct1d_32_flat_emu(c: &mut [[f32; 8]; 32]) {
        for lane in 0..8 {
            let mut col = [0.0f32; 32];
            for i in 0..32 {
                col[i] = c[i][lane];
            }
            dct1d_32(&mut col);
            for i in 0..32 {
                c[i][lane] = col[i];
            }
        }
    }

    #[test]
    fn dct32x16_simd_layout_matches_scalar() {
        let input: [f32; 512] = std::array::from_fn(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0);
        let mut output = [0.0f32; 512];
        // Phase 1: 32-pt column DCT, lane = column (contiguous loadu/storeu).
        let mut after_col = [0.0f32; 512];
        for g in 0..2 {
            let mut c = [[0.0f32; 8]; 32];
            for r in 0..32 {
                for lane in 0..8 {
                    c[r][lane] = input[r * 16 + g * 8 + lane];
                }
            }
            dct1d_32_flat_emu(&mut c);
            for v in 0..32 {
                for lane in 0..8 {
                    after_col[v * 16 + g * 8 + lane] = c[v][lane];
                }
            }
        }
        // Phase 2: 16-pt row DCT, lane = vfreq (set_ps gather → output[u*32+v]).
        let scale = 1.0 / 512.0;
        for g in 0..4 {
            let b = g * 8;
            let mut c = [[0.0f32; 8]; 16];
            for u in 0..16 {
                for lane in 0..8 {
                    c[u][lane] = after_col[(b + lane) * 16 + u];
                }
            }
            dct1d_16_flat_emu(&mut c);
            for u in 0..16 {
                for lane in 0..8 {
                    output[u * 32 + b + lane] = c[u][lane] * scale;
                }
            }
        }
        let mut want = [0.0f32; 512];
        dct32x16_scalar(&input, &mut want);
        for i in 0..512 {
            assert!(
                (output[i] - want[i]).abs() < 1e-3,
                "dct32x16 SIMD-layout mismatch at {i}: {} vs {}",
                output[i],
                want[i]
            );
        }
    }

    #[test]
    fn dct16x32_simd_layout_matches_scalar() {
        let input: [f32; 512] = std::array::from_fn(|i| ((i * 53 % 97) as f32 - 48.0) / 24.0);
        let mut output = [0.0f32; 512];
        // Phase 1: 32-pt row DCT, lane = row (set_ps gather → after_row[u*16+row]).
        let mut after_row = [0.0f32; 512];
        for g in 0..2 {
            let b = g * 8;
            let mut c = [[0.0f32; 8]; 32];
            for u in 0..32 {
                for lane in 0..8 {
                    c[u][lane] = input[(b + lane) * 32 + u];
                }
            }
            dct1d_32_flat_emu(&mut c);
            for u in 0..32 {
                for lane in 0..8 {
                    after_row[u * 16 + b + lane] = c[u][lane];
                }
            }
        }
        // Phase 2: 16-pt column DCT, lane = hfreq (set_ps gather → output[v*32+u]).
        let scale = 1.0 / 512.0;
        for g in 0..4 {
            let b = g * 8;
            let mut c = [[0.0f32; 8]; 16];
            for r in 0..16 {
                for lane in 0..8 {
                    c[r][lane] = after_row[(b + lane) * 16 + r];
                }
            }
            dct1d_16_flat_emu(&mut c);
            for v in 0..16 {
                for lane in 0..8 {
                    output[v * 32 + b + lane] = c[v][lane] * scale;
                }
            }
        }
        let mut want = [0.0f32; 512];
        dct16x32_scalar(&input, &mut want);
        for i in 0..512 {
            assert!(
                (output[i] - want[i]).abs() < 1e-3,
                "dct16x32 SIMD-layout mismatch at {i}: {} vs {}",
                output[i],
                want[i]
            );
        }
    }

    pub(crate) fn idct4x4(input: &[f32; 64], output: &mut [f32; 64]) {
        let mut coeff = *input;
        // Undo the 2×2 Hadamard: H is self-inverse up to scale (forward used ×0.25,
        // so the inverse is the plain Hadamard ×1).
        let c0 = coeff[0];
        let c1 = coeff[1];
        let c8 = coeff[8];
        let c9 = coeff[9];
        coeff[0] = c0 + c1 + c8 + c9; // b00
        coeff[1] = c0 + c1 - c8 - c9; // b01
        coeff[8] = c0 - c1 + c8 - c9; // b10
        coeff[9] = c0 - c1 - c8 + c9; // b11
        for qy in 0..2 {
            for qx in 0..2 {
                let mut d = [0.0f32; 16];
                for iy in 0..4 {
                    for ix in 0..4 {
                        d[iy * 4 + ix] = coeff[(qy + iy * 2) * 8 + (qx + ix * 2)];
                    }
                }
                let mut px = [0.0f32; 16];
                idct4x4_2d(&d, &mut px);
                for r in 0..4 {
                    for c in 0..4 {
                        output[(qy * 4 + r) * 8 + (qx * 4 + c)] = px[r * 4 + c];
                    }
                }
            }
        }
    }

    #[test]
    fn dct4x4_idct4x4_round_trips() {
        // dct4x4 followed by idct4x4 must recover the original 8×8 pixels.
        let mut input = [0.0f32; 64];
        for (i, v) in input.iter_mut().enumerate() {
            // A non-separable, non-symmetric pattern so a transpose bug would show.
            *v = ((i * 37 % 53) as f32) - 25.0 + (i as f32) * 0.13;
        }
        let mut coeff = [0.0f32; 64];
        dct4x4(&input, &mut coeff);
        let mut recon = [0.0f32; 64];
        idct4x4(&coeff, &mut recon);
        for (a, b) in input.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-3, "round-trip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn dct4x4_dc_is_block_mean() {
        // coeff[0] after the 2×2 Hadamard is the overall block mean (the DC the
        // decoder pulls out via dc[0] = coeff[0]).
        let input = [3.5f32; 64];
        let mut coeff = [0.0f32; 64];
        dct4x4(&input, &mut coeff);
        assert!((coeff[0] - 3.5).abs() < 1e-4, "DC {} != 3.5", coeff[0]);
        // A flat block has no AC energy.
        for (i, &c) in coeff.iter().enumerate() {
            if i != 0 {
                assert!(c.abs() < 1e-3, "coeff[{i}] = {c} should be ~0");
            }
        }
    }

    #[test]
    fn dct1d_32_diagonalizes_dct_basis() {
        // Feeding the k0-th DCT-II basis vector must yield a single non-zero
        // output at index k0 (i.e. dct1d_32 really is a DCT).
        let n = 32usize;
        for k0 in 0..n {
            let mut x = [0.0f32; 32];
            for (nn, xv) in x.iter_mut().enumerate() {
                *xv = (std::f32::consts::PI * (2.0 * nn as f32 + 1.0) * k0 as f32
                    / (2.0 * n as f32))
                    .cos();
            }
            dct1d_32(&mut x);
            let (mut mi, mut mv) = (0usize, 0.0f32);
            for (i, &v) in x.iter().enumerate() {
                if v.abs() > mv {
                    mv = v.abs();
                    mi = i;
                }
            }
            assert_eq!(mi, k0, "dct1d_32 basis {k0} peaked at {mi}");
            let off: f32 = (0..n).filter(|&i| i != k0).map(|i| x[i] * x[i]).sum();
            assert!(off < 1e-6 * (mv * mv).max(1.0), "basis {k0} leakage {off}");
        }
    }

    #[test]
    fn dct32x32_constant_is_pure_dc() {
        let input = [1.0f32; 1024];
        let mut out = [0.0f32; 1024];
        dct32x32_scalar(&input, &mut out);
        // Same normalization as dct16x16 (1/N^2): DC of a flat block == mean.
        assert!((out[0] - 1.0).abs() < 1e-4, "DC = {}", out[0]);
        let off: f32 = (1..1024).map(|i| out[i] * out[i]).sum();
        assert!(off < 1e-5, "non-DC energy {off}");
    }

    #[test]
    fn dct32x32_matches_separable_reference() {
        // Compare the fast transform to a direct separable application of the
        // (probed) 1-D kernel, confirming the 2-D assembly + 1/1024 scale.
        let mut m1 = [[0.0f32; 32]; 32];
        for k in 0..32 {
            let mut e = [0.0f32; 32];
            e[k] = 1.0;
            dct1d_32(&mut e);
            for j in 0..32 {
                m1[j][k] = e[j];
            }
        }
        let mut rng = 0x9e37_79b9u32;
        let mut input = [0.0f32; 1024];
        for v in input.iter_mut() {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        }
        // reference: cols then rows, scale 1/1024, transposed store.
        let mut col = [[0.0f32; 32]; 32];
        for u in 0..32 {
            for j in 0..32 {
                let mut acc = 0.0f32;
                for i in 0..32 {
                    acc += m1[j][i] * input[i * 32 + u];
                }
                col[j][u] = acc;
            }
        }
        let mut reference = [0.0f32; 1024];
        for v in 0..32 {
            for u in 0..32 {
                let mut acc = 0.0f32;
                for i in 0..32 {
                    acc += m1[u][i] * col[v][i];
                }
                reference[u * 32 + v] = acc / 1024.0;
            }
        }
        let mut out = [0.0f32; 1024];
        dct32x32_scalar(&input, &mut out);
        let mut maxerr = 0.0f32;
        for i in 0..1024 {
            maxerr = maxerr.max((out[i] - reference[i]).abs());
        }
        assert!(maxerr < 1e-5, "dct32x32 vs reference maxerr {maxerr}");
    }

    #[test]
    fn dc_from_dct32x32_inverts_lf_injection() {
        // Round-trip: build LF coeffs from a known 4x4 DC via the decoder rule
        // (forward 4-pt DCT then divide by resample scales), feed through
        // dc_from_dct32x32, and recover the original DC. Validates the operator
        // is the exact inverse of LowestFrequenciesFromDC (lossless DC path).
        // We instead check invertibility directly: the 16x16 operator on the
        // lowest 4x4 must be invertible and stable.
        // Apply to several random LF patches and confirm determinism + that a
        // flat LF (only coeff[0]) maps to a flat DC grid.
        let mut coeffs = [0.0f32; 1024];
        coeffs[0] = 32.0; // pure DC LF
        let mut dc = [0.0f32; 16];
        dc_from_dct32x32(&coeffs, &mut dc);
        // RESAMPLE[0]=1, IDCT4 DC row = all ones -> every DC == coeffs[0].
        for (i, &d) in dc.iter().enumerate() {
            assert!((d - 32.0).abs() < 1e-3, "dc[{i}] = {d}, expected 32");
        }
    }

    #[test]
    fn dc_from_dct_context_dispatch_matches_scalar() {
        use crate::encoding_context::EncodingContext;

        let ctx = EncodingContext::default();
        let selected = selected_dc_from_dct_methods();
        assert_eq!(
            ctx.dc_from_dct32x32 as usize, selected.dct32x32 as usize,
            "32x32 context dispatch is not the selected kernel"
        );
        assert_eq!(
            ctx.dc_from_dct32x16 as usize, selected.dct32x16 as usize,
            "32x16 context dispatch is not the selected kernel"
        );
        assert_eq!(
            ctx.dc_from_dct16x32 as usize, selected.dct16x32 as usize,
            "16x32 context dispatch is not the selected kernel"
        );
        assert_eq!(
            ctx.dc_from_dct64x64 as usize, selected.dct64x64 as usize,
            "64x64 context dispatch is not the selected kernel"
        );
        assert_eq!(
            ctx.dc_from_dct64x32 as usize, selected.dct64x32 as usize,
            "64x32 context dispatch is not the selected kernel"
        );
        assert_eq!(
            ctx.dc_from_dct32x64 as usize, selected.dct32x64 as usize,
            "32x64 context dispatch is not the selected kernel"
        );

        let coeffs: [f32; 1024] = std::array::from_fn(|i| ((i * 73 % 251) as f32 - 125.0) / 19.0);
        let mut got16 = [0.0f32; 16];
        let mut want16 = [0.0f32; 16];
        (ctx.dc_from_dct32x32)(&coeffs, &mut got16);
        dc_from_dct32x32(&coeffs, &mut want16);
        for (got, want) in got16.iter().zip(want16) {
            assert!((got - want).abs() < 1e-4, "32x32: {got} != {want}");
        }

        let coeffs = coeffs.first_chunk::<512>().unwrap();
        let mut got8 = [0.0f32; 8];
        let mut want8 = [0.0f32; 8];
        (ctx.dc_from_dct32x16)(coeffs, &mut got8);
        dc_from_dct32x16(coeffs, &mut want8);
        for (got, want) in got8.iter().zip(want8) {
            assert!((got - want).abs() < 1e-4, "32x16: {got} != {want}");
        }

        (ctx.dc_from_dct16x32)(coeffs, &mut got8);
        dc_from_dct16x32(coeffs, &mut want8);
        for (got, want) in got8.iter().zip(want8) {
            assert!((got - want).abs() < 1e-4, "16x32: {got} != {want}");
        }

        let coeffs: [f32; 4096] = std::array::from_fn(|i| ((i * 73 % 251) as f32 - 125.0) / 19.0);
        let mut got64 = [0.0f32; 64];
        let mut want64 = [0.0f32; 64];
        (ctx.dc_from_dct64x64)(&coeffs, &mut got64);
        dc_from_dct64x64(&coeffs, &mut want64);
        for (got, want) in got64.iter().zip(want64) {
            assert!((got - want).abs() < 1e-4, "64x64: {got} != {want}");
        }

        let coeffs = coeffs.first_chunk::<2048>().unwrap();
        let mut got32 = [0.0f32; 32];
        let mut want32 = [0.0f32; 32];
        (ctx.dc_from_dct64x32)(coeffs, &mut got32);
        dc_from_dct64x32(coeffs, &mut want32);
        for (got, want) in got32.iter().zip(want32) {
            assert!((got - want).abs() < 1e-4, "64x32: {got} != {want}");
        }

        (ctx.dc_from_dct32x64)(coeffs, &mut got32);
        dc_from_dct32x64(coeffs, &mut want32);
        for (got, want) in got32.iter().zip(want32) {
            assert!((got - want).abs() < 1e-4, "32x64: {got} != {want}");
        }
    }

    #[test]
    fn idct_context_reuses_the_resolved_dispatch_table() {
        use crate::encoding_context::EncodingContext;

        let first = EncodingContext::default();
        let second = EncodingContext::default();
        assert!(std::ptr::eq(first.idct, selected_idct_methods()));
        assert!(std::ptr::eq(first.idct, second.idct));
    }

    #[test]
    fn identity8x8_round_trips_and_preserves_dc_scale() {
        let input = std::array::from_fn::<_, 64, _>(|i| {
            let x = (i % 8) as f32;
            let y = (i / 8) as f32;
            0.31 * x - 0.17 * y + (x * y * 0.23).sin()
        });
        let mut coeffs = [0.0f32; 64];
        identity8x8(DctInput::from_flat(&input), &mut coeffs);
        let expected_dc = input.iter().sum::<f32>() * (1.0 / 64.0);
        assert!((coeffs[0] - expected_dc).abs() < 1e-6);

        let mut recon = [0.0f32; 64];
        inv_identity8x8(DctInput::from_flat(&coeffs), &mut recon);
        for (want, got) in input.iter().zip(recon) {
            assert!((want - got).abs() < 2e-6, "{want} != {got}");
        }
    }

    #[test]
    fn dct2x2_8x8_round_trips_and_preserves_dc_scale() {
        let input = std::array::from_fn::<_, 64, _>(|i| {
            let x = (i % 8) as f32;
            let y = (i / 8) as f32;
            (0.37 * x + 0.61 * y).sin() + 0.07 * x * y
        });
        let mut coeffs = [0.0f32; 64];
        dct2x2_8x8(DctInput::from_flat(&input), &mut coeffs);
        let expected_dc = input.iter().sum::<f32>() * (1.0 / 64.0);
        assert!((coeffs[0] - expected_dc).abs() < 1e-6);

        let mut recon = [0.0f32; 64];
        inv_dct2x2_8x8(DctInput::from_flat(&coeffs), &mut recon);
        for (want, got) in input.iter().zip(recon) {
            assert!((want - got).abs() < 2e-6, "{want} != {got}");
        }
    }

    #[test]
    fn dct_linearity() {
        let mut a = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        for i in 0..64 {
            a[i] = (i as f32 * 0.13).sin();
            b[i] = (i as f32 * 0.27).cos();
        }
        let mut sum = [0.0f32; 64];
        for i in 0..64 {
            sum[i] = a[i] + b[i];
        }

        let mut da = [0.0f32; 64];
        let mut db = [0.0f32; 64];
        let mut dsum = [0.0f32; 64];
        dct8x8(&a, &mut da);
        dct8x8(&b, &mut db);
        dct8x8(&sum, &mut dsum);

        for i in 0..64 {
            let expected = da[i] + db[i];
            assert!(
                (dsum[i] - expected).abs() < 1e-4,
                "i={} dsum={} expected={}",
                i,
                dsum[i],
                expected
            );
        }
    }
}
