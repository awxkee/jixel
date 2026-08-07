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
use crate::ColorSpace;
use crate::encode_image::MAX_DIMENSION;
use std::fmt;
use std::ops::{Deref, DerefMut};

pub(crate) fn heap_array<T: Clone, const N: usize>(value: T) -> Box<[T; N]> {
    let slice = vec![value; N].into_boxed_slice();
    boxed_slice_to_array(slice)
}

pub(crate) fn heap_array_from_fn<T, const N: usize>(make: impl FnMut(usize) -> T) -> Box<[T; N]> {
    let slice = (0..N).map(make).collect::<Vec<_>>().into_boxed_slice();
    boxed_slice_to_array(slice)
}

fn boxed_slice_to_array<T, const N: usize>(slice: Box<[T]>) -> Box<[T; N]> {
    match slice.try_into() {
        Ok(array) => array,
        Err(_) => unreachable!("boxed slice length is the requested array length"),
    }
}

/// A fixed-size two-dimensional array whose elements are initialized directly
/// in one flat heap allocation.
#[derive(Clone)]
pub(crate) struct HeapMatrix<T, const ROWS: usize, const COLS: usize> {
    data: Box<[T]>,
}

impl<T: Clone, const ROWS: usize, const COLS: usize> HeapMatrix<T, ROWS, COLS> {
    pub(crate) fn new(value: T) -> Self {
        Self {
            data: vec![value; ROWS * COLS].into_boxed_slice(),
        }
    }

    pub(crate) fn from_rows(rows: &[[T; COLS]; ROWS]) -> Self {
        Self {
            data: rows
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

impl<T, const ROWS: usize, const COLS: usize> Deref for HeapMatrix<T, ROWS, COLS> {
    type Target = [[T; COLS]; ROWS];

    fn deref(&self) -> &Self::Target {
        let (rows, remainder) = self.data.as_chunks::<COLS>();
        debug_assert!(remainder.is_empty());
        rows.try_into()
            .unwrap_or_else(|_| unreachable!("heap matrix has its declared dimensions"))
    }
}

impl<T, const ROWS: usize, const COLS: usize> DerefMut for HeapMatrix<T, ROWS, COLS> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let (rows, remainder) = self.data.as_chunks_mut::<COLS>();
        debug_assert!(remainder.is_empty());
        rows.try_into()
            .unwrap_or_else(|_| unreachable!("heap matrix has its declared dimensions"))
    }
}

/// Errors that can occur during JXL encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum EncodeError {
    /// Image width or height is zero.
    EmptyImage,
    /// The input buffer length does not match `width * height * channels`.
    InputSizeMismatch {
        expected: usize,
        actual: usize,
    },
    /// The alpha plane length does not match `width * height`.
    AlphaSizeMismatch {
        expected: usize,
        actual: usize,
    },
    /// `distance` must be a positive finite number.
    InvalidDistance(f32),
    /// Quality value was NaN.
    QualityIsNaN,
    /// One or both image dimensions exceed the maximum representable value
    /// (2^30 − 1).
    DimensionTooLarge {
        width: usize,
        height: usize,
    },
    /// An alpha bit depth other than 8, 10, or 12 was supplied.
    UnsupportedAlphaBitDepth(u8),
    /// ICC profile injection is not yet implemented.
    IccProfileNotSupported,
    Unsupported(&'static str),
    BadChannelCount(usize),
    BadBitDepth(u32),
    IccNotSupported,
    SizeOverflow,
    InputLength {
        expected: usize,
        got: usize,
    },
    UnsupportedColorSpace(ColorSpace),
    /// The input JPEG could not be transcoded losslessly.
    Jpeg(String),
    /// A caller-provided Brotli backend could not compress metadata.
    Brotli(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => write!(f, "image dimensions must be non-zero"),
            Self::InputSizeMismatch { expected, actual } => write!(
                f,
                "input buffer size mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::AlphaSizeMismatch { expected, actual } => write!(
                f,
                "alpha plane size mismatch: expected {expected} pixels, got {actual}"
            ),
            Self::InvalidDistance(d) => write!(
                f,
                "butteraugli distance must be a positive finite number, got {d}"
            ),
            Self::QualityIsNaN => write!(f, "quality value must not be NaN"),
            Self::DimensionTooLarge { width, height } => write!(
                f,
                "image dimensions {width}×{height} exceed the maximum ({})",
                MAX_DIMENSION
            ),
            Self::UnsupportedAlphaBitDepth(bits) => {
                write!(
                    f,
                    "unsupported alpha bit depth: {bits} (must be 8, 10, or 12)"
                )
            }
            Self::IccProfileNotSupported => {
                write!(f, "ICC profile injection is not yet supported by jixel")
            }
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::BadChannelCount(n) => write!(f, "channel count {} not in 1..=4", n),
            Self::BadBitDepth(b) => write!(f, "bits_per_sample {} not in 1..=16", b),
            Self::IccNotSupported => write!(
                f,
                "embedded ICC not yet supported; use an enumerated color space"
            ),
            Self::SizeOverflow => write!(f, "image size overflows usize"),
            Self::InputLength { expected, got } => {
                write!(f, "input length {} != expected {}", got, expected)
            }
            Self::Jpeg(msg) => write!(f, "JPEG transcoding failed: {msg}"),
            Self::Brotli(msg) => write!(f, "Brotli compression failed: {msg}"),
            EncodeError::UnsupportedColorSpace(colorspace) => {
                f.write_fmt(format_args!("unsupported color space: {:?}", colorspace))
            }
        }
    }
}

impl std::error::Error for EncodeError {}

pub(crate) trait FastRound {
    fn fast_round(self) -> Self;
}

impl FastRound for f32 {
    fn fast_round(self) -> Self {
        self.round()
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[inline]
fn f32_to_f16_bits_impl(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    let e16 = exp - 112; // rebias 127 -> 15
    if e16 >= 31 {
        return sign | 0x7BFF; // clamp to max finite; NaN/Inf unsupported on the wire
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e16) as u32;
        let rounded = (m + (1 << (shift - 1)) - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let m = mant + 0xFFF + ((mant >> 13) & 1);
    let mut e16 = e16 as u32;
    let mut m16 = m >> 13;
    if m16 == 0x400 {
        m16 = 0;
        e16 += 1;
        if e16 >= 31 {
            return sign | 0x7BFF;
        }
    }
    sign | ((e16 as u16) << 10) | (m16 as u16)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "f16c")]
fn f32_to_f16_bits_f16c(v: f32) -> u16 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_MM_FROUND_TO_NEAREST_INT, _mm_cvtps_ph, _mm_cvtsi128_si32, _mm_set_ss};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _MM_FROUND_TO_NEAREST_INT, _mm_cvtps_ph, _mm_cvtsi128_si32, _mm_set_ss,
    };

    _mm_cvtsi128_si32(_mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_set_ss(v))) as u16
}

#[inline]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use std::arch::aarch64::{vcvt_f16_f32, vdupq_n_f32, vget_lane_u16, vreinterpret_u16_f16};
        unsafe { vget_lane_u16::<0>(vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(v)))) }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let converter = F16_CONVERTERS.get_or_init(select_f16_converters);
        unsafe { (converter.f32_to_f16_bits)(v) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        target_arch = "x86",
        target_arch = "x86_64"
    )))]
    {
        f32_to_f16_bits_impl(v)
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[inline]
fn f16_bits_to_f32_impl(b: u16) -> f32 {
    let sign = (b >> 15) as u32;
    let biased_exp = ((b >> 10) & 0x1F) as u32;
    let mantissa = (b & 0x3FF) as u32;
    if biased_exp == 0 {
        let v = (1.0f32 / 16384.0) * (mantissa as f32 * (1.0 / 1024.0));
        return if sign != 0 { -v } else { v };
    }
    f32::from_bits((sign << 31) | ((biased_exp + 112) << 23) | (mantissa << 13))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "f16c")]
fn f16_bits_to_f32_f16c(b: u16) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_cvtph_ps, _mm_cvtsi32_si128, _mm_cvtss_f32};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_cvtph_ps, _mm_cvtsi32_si128, _mm_cvtss_f32};

    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(b as i32)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct F16Converters {
    f32_to_f16_bits: unsafe fn(f32) -> u16,
    f16_bits_to_f32: unsafe fn(u16) -> f32,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn select_f16_converters() -> F16Converters {
    if std::is_x86_feature_detected!("f16c") {
        return F16Converters {
            f32_to_f16_bits: f32_to_f16_bits_f16c,
            f16_bits_to_f32: f16_bits_to_f32_f16c,
        };
    }
    F16Converters {
        f32_to_f16_bits: f32_to_f16_bits_impl,
        f16_bits_to_f32: f16_bits_to_f32_impl,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static F16_CONVERTERS: std::sync::OnceLock<F16Converters> = std::sync::OnceLock::new();

/// Mirror of libjxl `F16Coder::Read`: the value the decoder reconstructs from
/// a binary16 bit pattern.
#[inline]
pub(crate) fn f16_bits_to_f32(b: u16) -> f32 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use std::arch::aarch64::{vcvt_f32_f16, vdup_n_u16, vgetq_lane_f32, vreinterpret_f16_u16};
        unsafe { vgetq_lane_f32::<0>(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(b)))) }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let converter = F16_CONVERTERS.get_or_init(select_f16_converters);
        unsafe { (converter.f16_bits_to_f32)(b) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        target_arch = "x86",
        target_arch = "x86_64"
    )))]
    {
        f16_bits_to_f32_impl(b)
    }
}

#[cfg(test)]
mod f16_tests {
    use super::{f16_bits_to_f32, f32_to_f16_bits};

    #[test]
    fn known_hdr_luminances() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(255.0), 0x5BF8);
        assert_eq!(f32_to_f16_bits(1000.0), 0x63D0);
        assert_eq!(f32_to_f16_bits(4000.0), 0x6BD0);
        assert_eq!(f32_to_f16_bits(10000.0), 0x70E2);
    }

    #[test]
    fn known_binary16_values() {
        assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_bits_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0x5BF8), 255.0);
        assert_eq!(f16_bits_to_f32(0xFBFF), -65504.0);
    }
}
