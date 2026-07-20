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
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse4.1"
        ))]
        {
            const MAGIC: f32 = ((1u32 << 23) + (1u32 << 22)) as f32;
            (f32::from_bits(self.to_bits() + 1) + MAGIC) - MAGIC
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.round()
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse4.1"
            )
        )))]
        {
            const MAGIC: f32 = ((1u32 << 23) + (1u32 << 22)) as f32;
            (f32::from_bits(self.to_bits() + 1) + MAGIC) - MAGIC
        }
    }
}

/// Convert an `f32` to IEEE-754 binary16 (half-float) bits, matching JXL's F16
/// coder (sign | 5-bit biased exponent | 10-bit mantissa, round-to-nearest-even).
/// Used for HDR tone-mapping metadata (`intensity_target`, `min_nits`, ...).
/// Values out of the finite half range are clamped to the max finite half;
/// inf/NaN are not expected here (JXL rejects them in F16 fields).
pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127; // unbiased
    let mant = bits & 0x7F_FFFF; // 23-bit mantissa
    if value == 0.0 {
        return sign;
    }
    if exp > 15 {
        return sign | 0x7BFF; // clamp to max finite half (65504)
    }
    if exp < -24 {
        return sign; // underflow to zero
    }
    if exp < -14 {
        // Subnormal half: shift the implicit-1 mantissa down.
        let shift = (14 - exp) as u32; // 0..=10 extra
        let full = mant | 0x80_0000;
        let half_mant = full >> (shift + 13);
        let round = (full >> (shift + 12)) & 1;
        return sign | ((half_mant as u16) + round as u16);
    }
    // Normalized: pack and round-to-nearest-even; a mantissa carry naturally
    // increments the exponent via the addition.
    let e16 = (exp + 15) as u16;
    let mant16 = (mant >> 13) as u16;
    let remainder = mant & 0x1FFF; // dropped low 13 bits
    let half = 0x1000u32;
    let mut out = (e16 << 10) | mant16;
    if remainder > half || (remainder == half && (mant16 & 1) == 1) {
        out += 1; // round up (carry into exponent if mantissa overflows)
    }
    if out >= 0x7C00 {
        out = 0x7BFF; // never emit inf from rounding
    }
    sign | out
}

#[cfg(test)]
mod f16_tests {
    use super::f32_to_f16;
    #[test]
    fn known_hdr_luminances() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(255.0), 0x5BF8);
        assert_eq!(f32_to_f16(1000.0), 0x63D0);
        assert_eq!(f32_to_f16(4000.0), 0x6BD0);
        assert_eq!(f32_to_f16(10000.0), 0x70E2);
    }
}
