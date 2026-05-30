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
