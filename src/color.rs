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

use crate::color_encoding::{Primaries, TransferFunction};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn srgb_to_linear_u8_ref(v: u8) -> f32 {
    let v = v as f64 * (1. / 255.0);
    (if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }) as f32
}

#[inline]
pub(crate) fn srgb_to_linear_f32(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

type TransferFn = fn(f32) -> f32;

#[inline]
fn linear_to_linear(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[inline]
fn pq_to_linear(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 32.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 128.0;
    const C3: f32 = 2392.0 / 128.0;
    let p = v.powf(1.0 / M2);
    let relative_to_10k = ((p - C1).max(0.0) / (C2 - C3 * p).max(f32::MIN_POSITIVE)).powf(1.0 / M1);
    // JPEG XL's opsin domain uses 255 nits as linear 1.0, whereas PQ
    // is normalized to 10,000 nits.
    relative_to_10k * (10_000.0 / 255.0)
}

#[inline]
fn hlg_to_linear(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    const A: f32 = 0.178_832_77;
    const B: f32 = 0.284_668_92;
    const C: f32 = 0.559_910_7;
    if v <= 0.5 {
        v * v / 3.0
    } else {
        (((v - C) / A).exp() + B) / 12.0
    }
}

#[inline]
fn bt709_to_linear(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    const ALPHA: f32 = 1.099_296_8;
    const BETA: f32 = 0.018_053_97;
    if v < 4.5 * BETA {
        v / 4.5
    } else {
        ((v + ALPHA - 1.0) / ALPHA).powf(1.0 / 0.45)
    }
}

#[inline]
fn bt470m_to_linear(v: f32) -> f32 {
    v.clamp(0.0, 1.0).powf(2.2)
}

#[inline]
fn bt470bg_to_linear(v: f32) -> f32 {
    v.clamp(0.0, 1.0).powf(2.8)
}

#[inline]
fn smpte240_to_linear(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v < 0.0912 {
        v / 4.0
    } else {
        ((v + 0.1115) / 1.1115).powf(1.0 / 0.45)
    }
}

#[inline]
fn smpte428_to_linear(v: f32) -> f32 {
    v.clamp(0.0, 1.0).powf(2.6)
}

fn select_transfer(transfer: TransferFunction) -> TransferFn {
    match transfer {
        TransferFunction::Linear => linear_to_linear,
        TransferFunction::Smpte2084 => pq_to_linear,
        TransferFunction::Hlg => hlg_to_linear,
        TransferFunction::Bt709
        | TransferFunction::Bt601
        | TransferFunction::Bt202010bit
        | TransferFunction::Bt202012bit => bt709_to_linear,
        TransferFunction::Bt470M => bt470m_to_linear,
        TransferFunction::Bt470Bg => bt470bg_to_linear,
        TransferFunction::Smpte240 => smpte240_to_linear,
        TransferFunction::Smpte428 => smpte428_to_linear,
        TransferFunction::Reserved
        | TransferFunction::Unspecified
        | TransferFunction::Log100
        | TransferFunction::Log100sqrt10
        | TransferFunction::Iec61966
        | TransferFunction::Bt1361
        | TransferFunction::Srgb => srgb_to_linear_f32,
    }
}

/// Decode a normalized source sample to linear light according to its declared
/// transfer function. Selection is factored out for block-oriented hot paths.
#[inline]
#[cfg(test)]
pub(crate) fn transfer_to_linear_f32(v: f32, transfer: TransferFunction) -> f32 {
    select_transfer(transfer)(v)
}

pub(crate) trait TransferDecoder<T>: Sync {
    fn decode(&self, sample: T) -> f32;
}

pub(crate) trait LutSample: Copy {
    const DOMAIN_SIZE: usize;

    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx"),
        all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse")
    ))]
    fn as_index(self) -> usize;
}

impl LutSample for u8 {
    const DOMAIN_SIZE: usize = 1 << u8::BITS;

    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx"),
        all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse")
    ))]
    #[inline(always)]
    fn as_index(self) -> usize {
        self as usize
    }
}

impl LutSample for u16 {
    const DOMAIN_SIZE: usize = 1 << u16::BITS;

    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx"),
        all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse")
    ))]
    #[inline(always)]
    fn as_index(self) -> usize {
        self as usize
    }
}

pub(crate) struct TransferLut {
    values: Arc<[f32]>,
}

impl TransferLut {
    pub(crate) fn new(bits: u32, transfer: TransferFunction) -> Self {
        type Cache = HashMap<(u32, u32), Arc<[f32]>>;
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

        let key = (bits, transfer as u32);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(values) = cache.lock().unwrap().get(&key).cloned() {
            return Self { values };
        }

        let size = 1usize << bits;
        let max = (size - 1) as f32;
        let domain_size = if bits <= 8 {
            <u8 as LutSample>::DOMAIN_SIZE
        } else {
            <u16 as LutSample>::DOMAIN_SIZE
        };
        let mut built = vec![0.0f32; domain_size];
        if bits == 8 && transfer == TransferFunction::Srgb {
            for (v, slot) in built[..size].iter_mut().enumerate() {
                *slot = srgb_to_linear_u8_ref(v as u8);
            }
        } else {
            let decode = select_transfer(transfer);
            for (v, slot) in built[..size].iter_mut().enumerate() {
                *slot = decode(v as f32 / max);
            }
        }
        let built: Arc<[f32]> = built.into();

        let mut cache = cache.lock().unwrap();
        let values = cache.entry(key).or_insert_with(|| built.clone()).clone();
        Self { values }
    }

    #[inline]
    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx"),
        all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse")
    ))]
    fn as_array<const N: usize>(&self) -> &[f32; N] {
        self.values
            .as_ref()
            .try_into()
            .expect("transfer LUT has the sample type's complete domain")
    }
}

impl TransferDecoder<u8> for TransferLut {
    #[inline(always)]
    fn decode(&self, sample: u8) -> f32 {
        self.values[sample as usize]
    }
}

impl TransferDecoder<u16> for TransferLut {
    #[inline(always)]
    fn decode(&self, sample: u16) -> f32 {
        self.values[sample as usize]
    }
}

pub(crate) struct FloatTransferDecoder {
    decode: TransferFn,
}

impl FloatTransferDecoder {
    pub(crate) fn new(transfer: TransferFunction) -> Self {
        Self {
            decode: select_transfer(transfer),
        }
    }
}

impl TransferDecoder<f32> for FloatTransferDecoder {
    #[inline(always)]
    fn decode(&self, sample: f32) -> f32 {
        (self.decode)(sample)
    }
}

#[derive(Clone, Copy)]
enum PrimariesTransform {
    Identity,
    Matrix([f32; 9]),
}

impl PrimariesTransform {
    fn select(primaries: Primaries) -> Self {
        match primaries {
            Primaries::Bt2020 => Self::Matrix([
                1.660_491,
                -0.587_641_1,
                -0.072_849_86,
                -0.124_550_5,
                1.132_899_9,
                -0.008_349_42,
                -0.018_150_76,
                -0.100_578_9,
                1.118_729_7,
            ]),
            Primaries::Smpte431 | Primaries::Smpte432 => Self::Matrix([
                1.224_745_3,
                -0.224_904_4,
                -0.000_000_04,
                -0.042_057_9,
                1.042_081,
                -0.000_000_08,
                -0.019_642_3,
                -0.078_654_9,
                1.098_537_2,
            ]),
            _ => Self::Identity,
        }
    }
}

pub(crate) struct ColorTransform<D> {
    decoder: D,
    primaries: PrimariesTransform,
}

impl ColorTransform<TransferLut> {
    pub(crate) fn for_integer(bits: u32, transfer: TransferFunction, primaries: Primaries) -> Self {
        Self {
            decoder: TransferLut::new(bits, transfer),
            primaries: PrimariesTransform::select(primaries),
        }
    }
}

impl ColorTransform<FloatTransferDecoder> {
    pub(crate) fn for_float(transfer: TransferFunction, primaries: Primaries) -> Self {
        Self {
            decoder: FloatTransferDecoder::new(transfer),
            primaries: PrimariesTransform::select(primaries),
        }
    }
}

pub(crate) trait RgbBlockTransform<T>: Sync {
    fn transform_block<const CHANNELS: usize>(
        &self,
        src: &[[T; CHANNELS]],
        output: [&mut [f32]; 3],
    );
}

#[inline]
fn validate_block<T, const CHANNELS: usize>(src: &[[T; CHANNELS]], output: &[&mut [f32]; 3]) {
    debug_assert!(CHANNELS >= 3);
    debug_assert_eq!(src.len(), output[0].len());
    debug_assert_eq!(output[0].len(), output[1].len());
    debug_assert_eq!(output[1].len(), output[2].len());
}

#[inline]
fn transform_identity<T: Copy, D: TransferDecoder<T>, const CHANNELS: usize>(
    decoder: &D,
    src: &[[T; CHANNELS]],
    output: [&mut [f32]; 3],
) {
    let [r, g, b] = output;
    for (i, &pixel) in src.iter().enumerate() {
        r[i] = decoder.decode(pixel[0]);
        g[i] = decoder.decode(pixel[1]);
        b[i] = decoder.decode(pixel[2]);
    }
}

#[inline]
fn transform_matrix_scalar<T: Copy, D: TransferDecoder<T>, const CHANNELS: usize>(
    decoder: &D,
    matrix: &[f32; 9],
    src: &[[T; CHANNELS]],
    output: [&mut [f32]; 3],
) {
    let [r, g, b] = output;
    for (i, &pixel) in src.iter().enumerate() {
        let lr = decoder.decode(pixel[0]);
        let lg = decoder.decode(pixel[1]);
        let lb = decoder.decode(pixel[2]);
        r[i] = matrix[0] * lr + matrix[1] * lg + matrix[2] * lb;
        g[i] = matrix[3] * lr + matrix[4] * lg + matrix[5] * lb;
        b[i] = matrix[6] * lr + matrix[7] * lg + matrix[8] * lb;
    }
}

fn transform_integer_matrix<T: LutSample, const LUT_SIZE: usize, const CHANNELS: usize>(
    decoder: &TransferLut,
    matrix: &[f32; 9],
    src: &[[T; CHANNELS]],
    output: [&mut [f32]; 3],
) where
    TransferLut: TransferDecoder<T>,
{
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        crate::neon::color_matrix_shaper_neon(decoder.as_array::<LUT_SIZE>(), matrix, src, output);
        return;
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        crate::avx::color_matrix_shaper_avx2(decoder.as_array::<LUT_SIZE>(), matrix, src, output);
        return;
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        crate::sse::color_matrix_shaper_sse41(decoder.as_array::<LUT_SIZE>(), matrix, src, output);
        return;
    }

    #[allow(unreachable_code)]
    transform_matrix_scalar(decoder, matrix, src, output);
}

macro_rules! impl_integer_block_transform {
    ($sample:ty, $lut_size:expr) => {
        impl RgbBlockTransform<$sample> for ColorTransform<TransferLut> {
            fn transform_block<const CHANNELS: usize>(
                &self,
                src: &[[$sample; CHANNELS]],
                output: [&mut [f32]; 3],
            ) {
                validate_block(src, &output);
                match self.primaries {
                    PrimariesTransform::Identity => transform_identity(&self.decoder, src, output),
                    PrimariesTransform::Matrix(matrix) => transform_integer_matrix::<
                        $sample,
                        $lut_size,
                        CHANNELS,
                    >(
                        &self.decoder, &matrix, src, output
                    ),
                }
            }
        }
    };
}

impl_integer_block_transform!(u8, 256);
impl_integer_block_transform!(u16, 65_536);

impl RgbBlockTransform<f32> for ColorTransform<FloatTransferDecoder> {
    fn transform_block<const CHANNELS: usize>(
        &self,
        src: &[[f32; CHANNELS]],
        output: [&mut [f32]; 3],
    ) {
        validate_block(src, &output);
        match self.primaries {
            PrimariesTransform::Identity => transform_identity(&self.decoder, src, output),
            PrimariesTransform::Matrix(matrix) => {
                transform_matrix_scalar(&self.decoder, &matrix, src, output)
            }
        }
    }
}

/// Convert source linear RGB into the linear-sRGB space used by the default
/// JPEG XL opsin transform.
#[inline]
#[cfg(test)]
pub(crate) fn primaries_to_linear_srgb(primaries: Primaries, r: f32, g: f32, b: f32) -> [f32; 3] {
    match PrimariesTransform::select(primaries) {
        PrimariesTransform::Identity => [r, g, b],
        PrimariesTransform::Matrix(m) => [
            m[0] * r + m[1] * g + m[2] * b,
            m[3] * r + m[4] * g + m[5] * b,
            m[6] * r + m[7] * g + m[8] * b,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_reference() {
        let lut = TransferLut::new(8, TransferFunction::Srgb);
        for i in 0..=255u8 {
            let r = srgb_to_linear_u8_ref(i);
            let l = lut.decode(i);
            assert_eq!(
                r.to_bits(),
                l.to_bits(),
                "srgb LUT differs from reference at {i}: ref={r} lut={l}"
            );
        }
    }

    #[test]
    fn pq_100_nits_uses_opsin_255_nit_scale() {
        // ST 2084 code value for 100 cd/m².
        let linear = transfer_to_linear_f32(0.508_078_4, TransferFunction::Smpte2084);
        assert!((linear - 100.0 / 255.0).abs() < 2e-5, "{linear}");
    }

    #[test]
    fn bt2020_neutral_stays_neutral() {
        let rgb = primaries_to_linear_srgb(Primaries::Bt2020, 0.25, 0.25, 0.25);
        for channel in rgb {
            assert!((channel - 0.25).abs() < 2e-6, "{rgb:?}");
        }
    }

    #[test]
    fn fused_pq_bt2020_block_matches_scalar_transform() {
        // Seven pixels exercise both the four-pixel NEON body and its tail,
        // as well as the two-pixel AVX body and its tail.
        let src = [
            [0u16, 32768, 65535],
            [40000, 30000, 20000],
            [65535, 65535, 65535],
            [1, 2, 3],
            [12000, 44000, 61000],
            [50000, 25000, 10000],
            [1024, 4096, 16384],
        ];
        let transform =
            ColorTransform::for_integer(16, TransferFunction::Smpte2084, Primaries::Bt2020);
        let mut r = [0.0; 7];
        let mut g = [0.0; 7];
        let mut b = [0.0; 7];
        transform.transform_block(&src, [&mut r, &mut g, &mut b]);

        for (i, px) in src.iter().enumerate() {
            let linear =
                px.map(|v| transfer_to_linear_f32(v as f32 / 65535.0, TransferFunction::Smpte2084));
            let expected =
                primaries_to_linear_srgb(Primaries::Bt2020, linear[0], linear[1], linear[2]);
            for (actual, expected) in [r[i], g[i], b[i]].into_iter().zip(expected) {
                let tolerance = 2e-6 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn integer_transfer_luts_are_reused() {
        let first = TransferLut::new(16, TransferFunction::Smpte2084);
        let second = TransferLut::new(16, TransferFunction::Smpte2084);
        assert!(Arc::ptr_eq(&first.values, &second.values));
    }
}
