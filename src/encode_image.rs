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
use crate::bit_writer::BitWriter;
use crate::color::{lut_high_bit, srgb_to_linear_f32, srgb_to_linear_u8};
use crate::color_encoding::write_color_encoding_with_icc;
use crate::enc_frame::encode_frame;
use crate::enc_lossless::{encode_frame_lossless, encode_frame_lossless_float, forward_ycocg};
use crate::image::{Image3F, Image3Si};
use crate::{ColorEncoding, EncodeError};

fn checked_buffer_size<T>(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<usize, EncodeError> {
    let pixel_size = size_of::<T>();
    let total_size = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(channels));

    _ = total_size
        .and_then(|v| v.checked_mul(pixel_size))
        .and_then(|v| isize::try_from(v).ok())
        .map(|v| v as usize)
        .ok_or(EncodeError::DimensionTooLarge {
            width: height,
            height,
        })?;

    total_size
        .and_then(|v| isize::try_from(v).ok())
        .map(|v| v as usize)
        .ok_or(EncodeError::DimensionTooLarge {
            width: height,
            height,
        })
}

/// 8-bit alpha plane (row-major, stride = `xsize`).
#[derive(Debug, Clone)]
pub(crate) enum AlphaPlane {
    /// 8-bit alpha, values 0..=255.
    U8(Vec<u8>),
    /// 10-bit (`bits=10`, values 0..=1023) or 12-bit (`bits=12`, values 0..=4095) alpha.
    U16 { data: Vec<u16>, bits: u8 },
    /// IEEE-754 single-precision alpha, stored as the raw 32-bit float bits
    /// reinterpreted as `i32` (matching libjxl's float_to_int for 32-bit float).
    F32(Vec<i32>),
}

/// Codestream marker byte that follows the leading 0xFF. Identifies this as
/// a raw JXL codestream (vs an ISOBMFF-wrapped one).
const CODESTREAM_MARKER: u8 = 0x0A;

/// Distances below this give larger files than lossless on photographic
/// content; we clamp up to this value.
const MIN_DISTANCE: f32 = 0.03;

/// JXL's image dimension field encodes (size - 1) in either 9, 13, 18, or
/// 30 bits, so 2^30 is the largest representable dimension.
pub(crate) const MAX_DIMENSION: usize = 0x3FFF_FFFF;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum BitsPerSample {
    #[default]
    Eight,
    Ten,
    Twelve,
    Sixteen,
    /// IEEE-754 half (16-bit, 5 exponent bits). Lossy only.
    F16,
    /// IEEE-754 single (32-bit, 8 exponent bits). Lossy only.
    F32,
}

impl BitsPerSample {
    pub(crate) fn bits(self) -> u32 {
        match self {
            BitsPerSample::Eight => 8,
            BitsPerSample::Ten => 10,
            BitsPerSample::Twelve => 12,
            BitsPerSample::Sixteen => 16,
            BitsPerSample::F16 => 16,
            BitsPerSample::F32 => 32,
        }
    }

    /// True for IEEE-754 float sample formats (f16/f32).
    pub(crate) fn is_float(self) -> bool {
        matches!(self, BitsPerSample::F16 | BitsPerSample::F32)
    }

    /// Number of exponent bits (5 for f16, 8 for f32); 0 for integer formats.
    pub(crate) fn exp_bits(self) -> u32 {
        match self {
            BitsPerSample::F16 => 5,
            BitsPerSample::F32 => 8,
            _ => 0,
        }
    }
}

impl AlphaPlane {
    /// Create an 8-bit alpha plane.
    #[inline]
    pub(crate) fn from_u8(data: Vec<u8>) -> Self {
        Self::U8(data)
    }

    /// Create a 10-bit alpha plane (values 0..=1023).
    #[inline]
    pub(crate) fn from_u16_10bit(data: Vec<u16>) -> Self {
        Self::U16 { data, bits: 10 }
    }

    /// Create a 12-bit alpha plane (values 0..=4095).
    #[inline]
    pub(crate) fn from_u16_12bit(data: Vec<u16>) -> Self {
        Self::U16 { data, bits: 12 }
    }

    /// Create a 16-bit alpha plane (values 0..=65535).
    #[inline]
    pub(crate) fn from_u16_16bit(data: Vec<u16>) -> Self {
        Self::U16 { data, bits: 16 }
    }

    /// Number of pixels.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16 { data, .. } => data.len(),
            Self::F32(v) => v.len(),
        }
    }

    /// Bit depth of the alpha samples (8, 10, or 12).
    #[inline]
    pub(crate) fn bits(&self) -> u8 {
        match self {
            Self::U8(_) => 8,
            Self::U16 { bits, .. } => *bits,
            Self::F32(_) => 32,
        }
    }

    /// True if the alpha samples are IEEE-754 float (stored as raw bits).
    #[inline]
    pub(crate) fn is_float(&self) -> bool {
        matches!(self, Self::F32(_))
    }

    /// Read pixel `idx` as `i32`.  Encoder hot path — kept tiny for inlining.
    #[inline]
    pub(crate) fn get_i32(&self, idx: usize) -> i32 {
        match self {
            Self::U8(v) => v[idx] as i32,
            Self::U16 { data, .. } => data[idx] as i32,
            Self::F32(v) => v[idx],
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodeConfig {
    pub distance: f32,
    pub color_encoding: ColorEncoding,
    pub icc_profile: Option<Vec<u8>>,
    /// If true, encode losslessly via the modular encoder. `distance` is then
    /// ignored. RGB and alpha both round-trip bit-perfectly.
    pub lossless: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct EncodeConfigImpl {
    pub(crate) distance: f32,
    pub(crate) color_encoding: ColorEncoding,
    pub(crate) icc_profile: Option<Vec<u8>>,
    pub(crate) alpha: Option<AlphaPlane>,
    /// Bit depth declared in the codestream (default: 8).
    pub(crate) bits_per_sample: BitsPerSample,
    /// If true, encode losslessly via the modular encoder. `distance` is then
    /// ignored. RGB and alpha both round-trip bit-perfectly.
    pub(crate) lossless: bool,
    /// If true, the image is grayscale: the codestream declares a Gray color
    /// space so the decoder emits a single-channel (L / LA) image. Internally
    /// the data still flows through the XYB pipeline with R=G=B, so the X and B
    /// chroma channels are ~constant and cost almost nothing.
    pub(crate) grayscale: bool,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            distance: 1.0,
            color_encoding: ColorEncoding::default(),
            icc_profile: None,
            lossless: false,
        }
    }
}

impl Default for EncodeConfigImpl {
    fn default() -> Self {
        Self {
            distance: 1.0,
            color_encoding: ColorEncoding::default(),
            icc_profile: None,
            alpha: None,
            bits_per_sample: BitsPerSample::Eight,
            lossless: false,
            grayscale: false,
        }
    }
}

impl EncodeConfigImpl {
    /// Convenience builder with the given butteraugli distance and otherwise
    /// default settings (sRGB primaries, linear transfer).
    pub(crate) fn with_distance(distance: f32) -> Self {
        Self {
            distance,
            ..Self::default()
        }
    }

    /// Attach an ICC profile. **Panics at encode time** — see field docs.
    pub(crate) fn with_icc_profile(mut self, icc: Option<Vec<u8>>) -> Self {
        self.icc_profile = icc;
        self
    }

    /// Attach an 8-bit alpha plane to be encoded losslessly via Modular.
    /// Length must equal `xsize * ysize` of the image passed to encode.
    pub(crate) fn with_alpha(mut self, alpha: AlphaPlane) -> Self {
        self.alpha = Some(alpha);
        self
    }

    pub(crate) fn with_bits_per_sample(mut self, bps: BitsPerSample) -> Self {
        self.bits_per_sample = bps;
        self
    }

    pub(crate) fn with_lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Mark the image as grayscale (declares a Gray color space).
    pub(crate) fn with_grayscale(mut self, grayscale: bool) -> Self {
        self.grayscale = grayscale;
        self
    }

    /// Replace the color encoding (white point / primaries / transfer / intent).
    pub(crate) fn with_color_encoding(mut self, enc: ColorEncoding) -> Self {
        self.color_encoding = enc;
        self
    }
}

impl EncodeConfig {
    /// Convenience builder with the given butteraugli distance and otherwise
    /// default settings (sRGB primaries, linear transfer).
    pub fn with_distance(mut self, distance: f32) -> Self {
        self.distance = distance;
        self
    }

    /// Convenience builder with quality on a libjpeg-like 0..=100 scale.
    /// See [`distance_from_quality`] for the mapping.
    pub fn with_quality(self, quality: f32) -> Self {
        self.with_distance(distance_from_quality(quality))
    }

    /// Replace the color encoding (white point / primaries / transfer / intent).
    pub fn with_color_encoding(mut self, enc: ColorEncoding) -> Self {
        self.color_encoding = enc;
        self
    }

    /// Attach an ICC profile. **Panics at encode time** — see field docs.
    pub fn with_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.icc_profile = Some(icc);
        self
    }
    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }
}

pub fn distance_from_quality(quality: f32) -> f32 {
    assert!(!quality.is_nan(), "quality must not be NaN");
    // Clamp at 100 from above (lossless isn't supported anyway; you'll
    // hit the MIN_DISTANCE floor below 0.03).
    let q = quality.min(100.0);
    let d = if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        6.24 + 2.5f32.powf((30.0 - q) / 5.0) / 6.25
    };
    d.min(25.0)
}

/// Encode a linear-light RGB `Image3F` at the given butteraugli distance,
/// using the default color encoding (sRGB primaries, linear transfer).
///
/// Shorthand for [`encode_with_config`] with default settings.
pub fn encode_image(
    input: &[u8],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }
    let expected = checked_buffer_size::<u8>(width, height, 3)?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected,
            actual: input.len(),
        });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }
    if config.lossless {
        return encode_with_config_loseless(
            input,
            width,
            height,
            false,
            8,
            &EncodeConfigImpl::with_distance(config.distance)
                .with_lossless(config.lossless)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);
    for (y, row) in input.chunks_exact(width * 3).enumerate() {
        let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
        for (((r, g), b), src) in r_row
            .iter_mut()
            .zip(g_row.iter_mut())
            .zip(b_row.iter_mut())
            .zip(row.as_chunks::<3>().0.iter())
        {
            *r = srgb_to_linear_u8(src[0]);
            *g = srgb_to_linear_u8(src[1]);
            *b = srgb_to_linear_u8(src[2]);
        }
    }
    encode_with_config(
        &linear,
        &EncodeConfigImpl::with_distance(distance)
            .with_icc_profile(config.icc_profile.clone())
            .with_color_encoding(config.color_encoding),
    )
}

/// Encode a linear-light RGB `Image3F` at the given butteraugli distance,
/// using the default color encoding (sRGB primaries, linear transfer).
///
/// Shorthand for [`encode_with_config`] with default settings.
pub fn encode_image_with_alpha(
    input: &[u8],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }
    let expected = checked_buffer_size::<u8>(width, height, 4)?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected,
            actual: input.len(),
        });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }

    if config.lossless {
        return encode_with_config_loseless(
            input,
            width,
            height,
            true,
            8,
            &EncodeConfigImpl::with_distance(config.distance)
                .with_lossless(config.lossless)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);
    let mut alpha_plane = vec![0u8; width * height];
    for (y, (row, alpha_row)) in input
        .chunks_exact(width * 4)
        .zip(alpha_plane.chunks_exact_mut(width))
        .enumerate()
    {
        let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
        for ((((r, g), b), src), alpha) in r_row
            .iter_mut()
            .zip(g_row.iter_mut())
            .zip(b_row.iter_mut())
            .zip(row.as_chunks::<4>().0.iter())
            .zip(alpha_row.iter_mut())
        {
            *r = srgb_to_linear_u8(src[0]);
            *g = srgb_to_linear_u8(src[1]);
            *b = srgb_to_linear_u8(src[2]);
            *alpha = src[3];
        }
    }
    encode_with_config(
        &linear,
        &EncodeConfigImpl::with_distance(distance)
            .with_alpha(AlphaPlane::from_u8(alpha_plane))
            .with_icc_profile(config.icc_profile.clone())
            .with_color_encoding(config.color_encoding),
    )
}

pub fn encode_image_with_alpha_10bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, true, config, BitsPerSample::Ten)
}

pub fn encode_image_with_alpha_12bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, true, config, BitsPerSample::Twelve)
}

/// Encode a 16-bit RGBA image. `input` is interleaved `[R, G, B, A]`,
/// `width * height * 4` samples, each in 0..=65535.
pub fn encode_image_with_alpha_16bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, true, config, BitsPerSample::Sixteen)
}

/// Encode a 16-bit RGB image. `input` is interleaved `[R, G, B]`,
/// `width * height * 3` samples, each in 0..=65535.
pub fn encode_image_16bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, false, config, BitsPerSample::Sixteen)
}

pub fn encode_image_10bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, false, config, BitsPerSample::Ten)
}

pub fn encode_image_12bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, false, config, BitsPerSample::Twelve)
}

/// Encode an 8-bit grayscale image. `input` is `width * height` luma bytes.
/// The codestream declares a Gray color space, so the decoder emits a
/// single-channel (L) image. Internally the luma is run through the XYB
/// pipeline with R=G=B; the chroma channels are ~constant and cost almost
/// nothing.
pub fn encode_image_gray(
    input: &[u8],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_gray_impl(input, None, width, height, config)
}

pub fn encode_image_gray_alpha(
    input: &[u8],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let expected = checked_buffer_size::<u8>(width, height, 2)?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height * 2,
            actual: input.len(),
        });
    }
    let (luma, alpha): (Vec<u8>, Vec<u8>) = input
        .as_chunks::<2>()
        .0
        .iter()
        .map(|px| (px[0], px[1]))
        .unzip();
    encode_gray_impl(&luma, Some(alpha), width, height, config)
}

/// Shared grayscale encode path. `luma` is `width * height` bytes; `alpha`, if
/// present, is the same length.
fn encode_gray_impl(
    luma: &[u8],
    alpha: Option<Vec<u8>>,
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }
    let expected = checked_buffer_size::<u8>(width, height, 1)?;
    if luma.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height,
            actual: luma.len(),
        });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }
    if config.lossless {
        // Lossless grayscale: route through the modular path as an RGB triplet
        // (R=G=B). The Gray color space still makes the decoder emit L/LA.
        let nchan = if alpha.is_some() { 4 } else { 3 };
        let mut interleaved = vec![0u8; width * height * nchan];
        match alpha.as_ref() {
            None => {
                // 3-channel: interleaved = [R, G, B] = [v, v, v]
                for (out, &v) in interleaved
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(luma.iter())
                {
                    out[0] = v;
                    out[1] = v;
                    out[2] = v;
                }
            }
            Some(a) => {
                // 4-channel: interleaved = [R, G, B, A] = [v, v, v, a]
                for (out, (&v, &av)) in interleaved
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(luma.iter().zip(a.iter()))
                {
                    out[0] = v;
                    out[1] = v;
                    out[2] = v;
                    out[3] = av;
                }
            }
        }
        return encode_with_config_loseless(
            &interleaved,
            width,
            height,
            alpha.is_some(),
            8,
            &EncodeConfigImpl::with_distance(config.distance)
                .with_lossless(true)
                .with_grayscale(true)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);
    for (y, row) in luma.chunks_exact(width).enumerate() {
        let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
        for (((r, g), b), &v) in r_row
            .iter_mut()
            .zip(g_row.iter_mut())
            .zip(b_row.iter_mut())
            .zip(row.iter())
        {
            let lin = srgb_to_linear_u8(v);
            *r = lin;
            *g = lin;
            *b = lin;
        }
    }
    let mut cfg = EncodeConfigImpl::with_distance(distance)
        .with_grayscale(true)
        .with_icc_profile(config.icc_profile.clone())
        .with_color_encoding(config.color_encoding);
    if let Some(a) = alpha {
        cfg = cfg.with_alpha(AlphaPlane::from_u8(a));
    }
    encode_with_config(&linear, &cfg)
}

/// Encode a 10-bit grayscale image. `input` is `width * height` luma samples (0..=1023).
pub fn encode_image_gray_10bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_gray_high_depth_impl(input, None, width, height, config, BitsPerSample::Ten)
}

/// Encode a 12-bit grayscale image. `input` is `width * height` luma samples (0..=4095).
pub fn encode_image_gray_12bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_gray_high_depth_impl(input, None, width, height, config, BitsPerSample::Twelve)
}

/// Encode a 10-bit grayscale+alpha image. `input` is interleaved `[L, A]` pairs,
/// `width * height * 2` samples total, each in 0..=1023.
pub fn encode_image_gray_alpha_10bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let expected = checked_buffer_size::<u16>(width, height, 2)?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height * 2,
            actual: input.len(),
        });
    }
    let (luma, alpha): (Vec<u16>, Vec<u16>) = input
        .as_chunks::<2>()
        .0
        .iter()
        .map(|px| (px[0], px[1]))
        .unzip();
    encode_gray_high_depth_impl(
        &luma,
        Some(alpha),
        width,
        height,
        config,
        BitsPerSample::Ten,
    )
}

/// Encode a 12-bit grayscale+alpha image. `input` is interleaved `[L, A]` pairs,
/// `width * height * 2` samples total, each in 0..=4095.
pub fn encode_image_gray_alpha_12bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let expected = checked_buffer_size::<u16>(width, height, 2)?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height * 2,
            actual: input.len(),
        });
    }
    let (luma, alpha): (Vec<u16>, Vec<u16>) = input
        .as_chunks::<2>()
        .0
        .iter()
        .map(|px| (px[0], px[1]))
        .unzip();
    encode_gray_high_depth_impl(
        &luma,
        Some(alpha),
        width,
        height,
        config,
        BitsPerSample::Twelve,
    )
}

/// Encode a 16-bit grayscale image. `input` is `width * height` luma samples (0..=65535).
pub fn encode_image_gray_16bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_gray_high_depth_impl(input, None, width, height, config, BitsPerSample::Sixteen)
}

/// Encode a 16-bit grayscale+alpha image. `input` is interleaved `[L, A]` pairs,
/// `width * height * 2` samples total, each in 0..=65535.
pub fn encode_image_gray_alpha_16bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let expected = checked_buffer_size::<u16>(width, height, 2)?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height * 2,
            actual: input.len(),
        });
    }
    let (luma, alpha): (Vec<u16>, Vec<u16>) = input
        .as_chunks::<2>()
        .0
        .iter()
        .map(|px| (px[0], px[1]))
        .unzip();
    encode_gray_high_depth_impl(
        &luma,
        Some(alpha),
        width,
        height,
        config,
        BitsPerSample::Sixteen,
    )
}

/// Shared high-bit-depth grayscale encode path.
/// `luma` is `width * height` samples; `alpha`, if present, is the same length.
fn encode_gray_high_depth_impl(
    luma: &[u16],
    alpha: Option<Vec<u16>>,
    width: usize,
    height: usize,
    config: &EncodeConfig,
    bps: BitsPerSample,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }
    let expected = checked_buffer_size::<u16>(width, height, 1)?;
    if luma.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height,
            actual: luma.len(),
        });
    }
    if let Some(alpha) = alpha.as_ref()
        && alpha.len() != expected
    {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height,
            actual: luma.len(),
        });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }

    if config.lossless {
        // Re-interleave as [L, L, L] or [L, L, L, A] so the existing
        // lossless RGB path can handle it.
        let nchan = if alpha.is_some() { 4 } else { 3 };
        let mut interleaved = vec![0u16; width * height * nchan];
        match alpha.as_ref() {
            None => {
                for (out, &v) in interleaved
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(luma.iter())
                {
                    out[0] = v;
                    out[1] = v;
                    out[2] = v;
                }
            }
            Some(a) => {
                for (out, (&v, &av)) in interleaved
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(luma.iter().zip(a.iter()))
                {
                    out[0] = v;
                    out[1] = v;
                    out[2] = v;
                    out[3] = av;
                }
            }
        }
        return encode_with_config_loseless(
            &interleaved,
            width,
            height,
            alpha.is_some(),
            bps.bits() as u8,
            &EncodeConfigImpl::with_distance(config.distance)
                .with_lossless(true)
                .with_grayscale(true)
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        );
    }

    let distance = config.distance.max(MIN_DISTANCE);
    let lut = &lut_high_bit(bps.bits() as u8).table;
    let mut linear = Image3F::new(width, height);

    for (y, row) in luma.chunks_exact(width).enumerate() {
        let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
        for (((r, g), b), &v) in r_row
            .iter_mut()
            .zip(g_row.iter_mut())
            .zip(b_row.iter_mut())
            .zip(row.iter())
        {
            let lin = lut[v as usize];
            *r = lin;
            *g = lin;
            *b = lin;
        }
    }

    let alpha_plane = alpha.map(|a| match bps {
        BitsPerSample::Ten => AlphaPlane::from_u16_10bit(a),
        BitsPerSample::Twelve => AlphaPlane::from_u16_12bit(a),
        BitsPerSample::Sixteen => AlphaPlane::from_u16_16bit(a),
        BitsPerSample::Eight => unreachable!("high-depth gray path called with 8-bit bps"),
        BitsPerSample::F16 | BitsPerSample::F32 => {
            unreachable!("float path does not use the integer alpha match")
        }
    });

    let mut cfg = EncodeConfigImpl::with_distance(distance)
        .with_grayscale(true)
        .with_bits_per_sample(bps)
        .with_icc_profile(config.icc_profile.clone())
        .with_color_encoding(config.color_encoding);
    if let Some(ap) = alpha_plane {
        cfg = cfg.with_alpha(ap);
    }
    encode_with_config(&linear, &cfg)
}

/// Shared implementation for 10-bit and 12-bit RGBA encoding.
fn encode_high_depth_rgba(
    input: &[u16],
    width: usize,
    height: usize,
    has_alpha: bool,
    config: &EncodeConfig,
    bps: BitsPerSample,
) -> Result<Vec<u8>, EncodeError> {
    let expected = checked_buffer_size::<u16>(width, height, if has_alpha { 4 } else { 3 })?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected,
            actual: input.len(),
        });
    }

    if config.lossless {
        return encode_with_config_loseless(
            input,
            width,
            height,
            has_alpha,
            bps.bits() as u8,
            &EncodeConfigImpl::with_distance(config.distance)
                .with_lossless(config.lossless)
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);

    let lut = &lut_high_bit(bps.bits() as u8).table;

    // For 16-bit, (1 << 16) - 1 overflows u16's shift; compute in u32 and cap.
    let bp_max: u16 = if bps.bits() >= 16 {
        u16::MAX
    } else {
        ((1u32 << bps.bits()) - 1) as u16
    };

    if has_alpha {
        let mut alpha_plane = vec![0u16; width * height];
        for (y, (row, alpha_row)) in input
            .chunks_exact(width * 4)
            .zip(alpha_plane.chunks_exact_mut(width))
            .enumerate()
        {
            let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
            for ((((r, g), b), src), alpha) in r_row
                .iter_mut()
                .zip(g_row.iter_mut())
                .zip(b_row.iter_mut())
                .zip(row.as_chunks::<4>().0.iter())
                .zip(alpha_row.iter_mut())
            {
                *r = lut[src[0] as usize];
                *g = lut[src[1] as usize];
                *b = lut[src[2] as usize];
                *alpha = src[3].min(bp_max);
            }
        }

        encode_with_config(
            &linear,
            &EncodeConfigImpl::with_distance(distance)
                .with_alpha(match bps {
                    BitsPerSample::Ten => AlphaPlane::from_u16_10bit(alpha_plane),
                    BitsPerSample::Twelve => AlphaPlane::from_u16_12bit(alpha_plane),
                    BitsPerSample::Sixteen => AlphaPlane::from_u16_16bit(alpha_plane),
                    BitsPerSample::Eight => unreachable!("high-depth path called with 8-bit bps"),
                    BitsPerSample::F16 | BitsPerSample::F32 => {
                        unreachable!("float path does not use the integer alpha match")
                    }
                })
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        )
    } else {
        for (y, row) in input.chunks_exact(width * 3).enumerate() {
            let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
            for (((r, g), b), src) in r_row
                .iter_mut()
                .zip(g_row.iter_mut())
                .zip(b_row.iter_mut())
                .zip(row.as_chunks::<3>().0.iter())
            {
                *r = lut[src[0] as usize];
                *g = lut[src[1] as usize];
                *b = lut[src[2] as usize];
            }
        }

        encode_with_config(
            &linear,
            &EncodeConfigImpl::with_distance(distance)
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        )
    }
}

/// Shared float (f16/f32) RGB(A) **lossy** encode path. `input` is interleaved
/// f32 samples (sRGB-encoded, normalized to [0, 1]), 3 or 4 per pixel. Alpha,
/// if present, is linear opacity quantized to a 16-bit integer extra channel
/// (alpha rarely needs float precision, and this reuses the 16-bit alpha path).
/// Lossless float is not supported here; `config.lossless` is ignored.
/// f32 lossless (v1): non-negative float RGB only. Reinterprets each float's
/// IEEE-754 bits as an int32 modular channel (matching libjxl's float_to_int
/// for 32-bit float) and codes them with no RCT and no LZ77.
fn encode_f32_lossless_rgba(
    input: &[f32],
    width: usize,
    height: usize,
    has_alpha: bool,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }

    // Guard once: v1 supports only finite, non-negative floats (negatives map to
    // large-magnitude negative int32 bits whose residuals overflow i32, which
    // needs the future 64-bit token path). Applies to RGB and alpha alike.
    for &v in input.iter() {
        if !v.is_finite() || v < 0.0 {
            return Err(EncodeError::Unsupported(
                "f32 lossless v1 supports only finite non-negative values",
            ));
        }
    }

    let nchan = if has_alpha { 4 } else { 3 };
    let mut image3s = Image3Si::new(width, height);
    let mut alpha_bits: Vec<i32> = if has_alpha {
        Vec::with_capacity(width * height)
    } else {
        Vec::new()
    };
    for (y, row) in input.chunks_exact(width * nchan).enumerate() {
        let [r_row, g_row, b_row] = image3s.all_plane_rows_mut(y);
        for (x, src) in row.chunks_exact(nchan).enumerate() {
            r_row[x] = src[0].to_bits() as i32;
            g_row[x] = src[1].to_bits() as i32;
            b_row[x] = src[2].to_bits() as i32;
            if has_alpha {
                alpha_bits.push(src[3].to_bits() as i32);
            }
        }
    }
    let alpha = if has_alpha {
        Some(AlphaPlane::F32(alpha_bits))
    } else {
        None
    };

    let mut w = BitWriter::new();
    w.write(8, 0xFF);
    w.write(8, CODESTREAM_MARKER as u64);
    write_size_header(width, height, &mut w);
    write_image_metadata(
        &config.color_encoding,
        alpha.as_ref(),
        config.icc_profile.as_deref(),
        BitsPerSample::F32,
        true,
        false,
        &mut w,
    );
    encode_frame_lossless_float(&image3s, alpha.as_ref(), &mut w);
    let codestream = w.into_bytes();
    let alpha_bits_md = if has_alpha { 32 } else { 0 };
    if needs_level_10(32, true, alpha_bits_md) {
        Ok(wrap_jxl_container(codestream, 10))
    } else {
        Ok(codestream)
    }
}

fn encode_float_rgba(
    input: &[f32],
    width: usize,
    height: usize,
    has_alpha: bool,
    config: &EncodeConfig,
    bps: BitsPerSample,
) -> Result<Vec<u8>, EncodeError> {
    let expected = checked_buffer_size::<f32>(width, height, if has_alpha { 4 } else { 3 })?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected,
            actual: input.len(),
        });
    }
    if config.lossless && matches!(bps, BitsPerSample::F32) {
        return encode_f32_lossless_rgba(input, width, height, has_alpha, config);
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);

    if has_alpha {
        let mut alpha_plane = vec![0u16; width * height];
        for (y, (row, alpha_row)) in input
            .chunks_exact(width * 4)
            .zip(alpha_plane.chunks_exact_mut(width))
            .enumerate()
        {
            let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
            for ((((r, g), b), src), a) in r_row
                .iter_mut()
                .zip(g_row.iter_mut())
                .zip(b_row.iter_mut())
                .zip(row.as_chunks::<4>().0.iter())
                .zip(alpha_row.iter_mut())
            {
                *r = srgb_to_linear_f32(src[0]);
                *g = srgb_to_linear_f32(src[1]);
                *b = srgb_to_linear_f32(src[2]);
                *a = (src[3].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
        }
        encode_with_config(
            &linear,
            &EncodeConfigImpl::with_distance(distance)
                .with_alpha(AlphaPlane::from_u16_16bit(alpha_plane))
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        )
    } else {
        for (y, row) in input.chunks_exact(width * 3).enumerate() {
            let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
            for (((r, g), b), src) in r_row
                .iter_mut()
                .zip(g_row.iter_mut())
                .zip(b_row.iter_mut())
                .zip(row.as_chunks::<3>().0.iter())
            {
                *r = srgb_to_linear_f32(src[0]);
                *g = srgb_to_linear_f32(src[1]);
                *b = srgb_to_linear_f32(src[2]);
            }
        }
        encode_with_config(
            &linear,
            &EncodeConfigImpl::with_distance(distance)
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        )
    }
}

/// Float (f16/f32) grayscale **lossy** encode path. `luma` is `width * height`
/// f32 samples (sRGB-encoded, [0, 1]).
fn encode_float_gray(
    luma: &[f32],
    width: usize,
    height: usize,
    config: &EncodeConfig,
    bps: BitsPerSample,
) -> Result<Vec<u8>, EncodeError> {
    let expected = checked_buffer_size::<f32>(width, height, 1)?;
    if luma.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected: width * height,
            actual: luma.len(),
        });
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);
    for (y, row) in luma.chunks_exact(width).enumerate() {
        let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
        for (((r, g), b), &v) in r_row
            .iter_mut()
            .zip(g_row.iter_mut())
            .zip(b_row.iter_mut())
            .zip(row.iter())
        {
            let lin = srgb_to_linear_f32(v);
            *r = lin;
            *g = lin;
            *b = lin;
        }
    }
    encode_with_config(
        &linear,
        &EncodeConfigImpl::with_distance(distance)
            .with_grayscale(true)
            .with_bits_per_sample(bps)
            .with_icc_profile(config.icc_profile.clone())
            .with_color_encoding(config.color_encoding),
    )
}

/// Encode a 32-bit float RGB image (lossy). `input` is interleaved `[R, G, B]`,
/// `width * height * 3` samples, each sRGB-encoded in [0, 1].
pub fn encode_image_f32(
    input: &[f32],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_float_rgba(input, width, height, false, config, BitsPerSample::F32)
}

/// Encode a 16-bit-half float RGB image (lossy). Input is f32 in [0, 1]; the
/// stream is declared as f16.
pub fn encode_image_f16(
    input: &[f32],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_float_rgba(input, width, height, false, config, BitsPerSample::F16)
}

/// Encode a 32-bit float RGBA image (lossy). `input` is interleaved
/// `[R, G, B, A]`; RGB sRGB-encoded in [0, 1], A linear opacity in [0, 1]
/// (quantized to a 16-bit alpha channel).
pub fn encode_image_with_alpha_f32(
    input: &[f32],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_float_rgba(input, width, height, true, config, BitsPerSample::F32)
}

/// Encode a 16-bit-half float RGBA image (lossy).
pub fn encode_image_with_alpha_f16(
    input: &[f32],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_float_rgba(input, width, height, true, config, BitsPerSample::F16)
}

/// Encode a 32-bit float grayscale image (lossy). `input` is `width * height`
/// luma samples, sRGB-encoded in [0, 1].
pub fn encode_image_gray_f32(
    input: &[f32],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_float_gray(input, width, height, config, BitsPerSample::F32)
}

/// Encode a 16-bit-half float grayscale image (lossy).
pub fn encode_image_gray_f16(
    input: &[f32],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_float_gray(input, width, height, config, BitsPerSample::F16)
}

/// Encode a linear-light RGB `Image3F` with the supplied configuration.
pub(crate) fn encode_with_config(
    input: &Image3F,
    config: &EncodeConfigImpl,
) -> Result<Vec<u8>, EncodeError> {
    if input.xsize() == 0 || input.ysize() == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if input.xsize() > MAX_DIMENSION || input.ysize() > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge {
            width: input.xsize(),
            height: input.ysize(),
        });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }

    if let Some(alpha) = config.alpha.as_ref() {
        let expected = match &alpha {
            AlphaPlane::U8(_) => checked_buffer_size::<u8>(input.xsize(), input.ysize(), 1)?,
            AlphaPlane::U16 { .. } => checked_buffer_size::<u16>(input.xsize(), input.ysize(), 1)?,
            AlphaPlane::F32(_) => checked_buffer_size::<f32>(input.xsize(), input.ysize(), 1)?,
        };
        if alpha.len() != expected {
            return Err(EncodeError::AlphaSizeMismatch {
                expected,
                actual: alpha.len(),
            });
        }
    }

    let distance = config.distance.max(MIN_DISTANCE);
    let mut w = BitWriter::new();
    w.write(8, 0xFF);
    w.write(8, CODESTREAM_MARKER as u64);
    write_size_header(input.xsize(), input.ysize(), &mut w);
    write_image_metadata(
        &config.color_encoding,
        config.alpha.as_ref(),
        config.icc_profile.as_deref(),
        config.bits_per_sample,
        config.lossless,
        config.grayscale,
        &mut w,
    );
    encode_frame(distance, input, config.alpha.as_ref(), &mut w);
    let codestream = w.into_bytes();
    let alpha_bits = config.alpha.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    if needs_level_10(config.bits_per_sample.bits(), config.lossless, alpha_bits) {
        Ok(wrap_jxl_container(codestream, 10))
    } else {
        Ok(codestream)
    }
}

pub(crate) trait AsSignedInt {
    fn to_signed_int(self, max_bp: u8) -> i32;
}

impl AsSignedInt for u8 {
    #[inline]
    fn to_signed_int(self, _: u8) -> i32 {
        self as i32
    }
}

impl AsSignedInt for u16 {
    #[inline]
    fn to_signed_int(self, max_bp: u8) -> i32 {
        let max_colors = ((1u32 << max_bp) - 1) as i32;
        (self as i32).min(max_colors)
    }
}

/// Encode a linear-light RGB `Image3F` with the supplied configuration.
fn encode_with_config_loseless<T: AsSignedInt + Copy>(
    input: &[T],
    width: usize,
    height: usize,
    has_alpha: bool,
    max_bp: u8,
    config: &EncodeConfigImpl,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    let expected = checked_buffer_size::<T>(width, height, if has_alpha { 4 } else { 3 })?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected,
            actual: input.len(),
        });
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }

    let mut image3s = Image3Si::new(width, height);
    let mut alpha_plane: Option<AlphaPlane> = None;

    if has_alpha {
        if max_bp > 8 {
            let mut new_alpha_plane = vec![0u16; width * height];
            for (y, (row, alpha_row)) in input
                .chunks_exact(width * 4)
                .zip(new_alpha_plane.chunks_exact_mut(width))
                .enumerate()
            {
                let [r_row, g_row, b_row] = image3s.all_plane_rows_mut(y);
                for ((((r, g), b), src), alpha) in r_row
                    .iter_mut()
                    .zip(g_row.iter_mut())
                    .zip(b_row.iter_mut())
                    .zip(row.as_chunks::<4>().0.iter())
                    .zip(alpha_row.iter_mut())
                {
                    let ycocg = forward_ycocg(
                        src[0].to_signed_int(max_bp),
                        src[1].to_signed_int(max_bp),
                        src[2].to_signed_int(max_bp),
                    );
                    *r = ycocg.0;
                    *g = ycocg.1;
                    *b = ycocg.2;
                    *alpha = src[3].to_signed_int(max_bp) as u16;
                }
            }

            alpha_plane = Some(AlphaPlane::U16 {
                data: new_alpha_plane,
                bits: max_bp,
            });
        } else {
            let mut new_alpha_plane = vec![0u8; width * height];
            for (y, (row, alpha_row)) in input
                .chunks_exact(width * 4)
                .zip(new_alpha_plane.chunks_exact_mut(width))
                .enumerate()
            {
                let [r_row, g_row, b_row] = image3s.all_plane_rows_mut(y);
                for ((((r, g), b), src), alpha) in r_row
                    .iter_mut()
                    .zip(g_row.iter_mut())
                    .zip(b_row.iter_mut())
                    .zip(row.as_chunks::<4>().0.iter())
                    .zip(alpha_row.iter_mut())
                {
                    let ycocg = forward_ycocg(
                        src[0].to_signed_int(max_bp),
                        src[1].to_signed_int(max_bp),
                        src[2].to_signed_int(max_bp),
                    );
                    *r = ycocg.0;
                    *g = ycocg.1;
                    *b = ycocg.2;
                    *alpha = src[3].to_signed_int(max_bp) as u8;
                }
            }
            alpha_plane = Some(AlphaPlane::U8(new_alpha_plane));
        }
    } else {
        for (y, row) in input.chunks_exact(width * 3).enumerate() {
            let [r_row, g_row, b_row] = image3s.all_plane_rows_mut(y);
            for (((r, g), b), src) in r_row
                .iter_mut()
                .zip(g_row.iter_mut())
                .zip(b_row.iter_mut())
                .zip(row.as_chunks::<3>().0.iter())
            {
                let ycocg = forward_ycocg(
                    src[0].to_signed_int(max_bp),
                    src[1].to_signed_int(max_bp),
                    src[2].to_signed_int(max_bp),
                );
                *r = ycocg.0;
                *g = ycocg.1;
                *b = ycocg.2;
            }
        }
    }

    let mut w = BitWriter::new();
    w.write(8, 0xFF);
    w.write(8, CODESTREAM_MARKER as u64);
    write_size_header(width, height, &mut w);
    write_image_metadata(
        &config.color_encoding,
        alpha_plane.as_ref(),
        config.icc_profile.as_deref(),
        config.bits_per_sample,
        config.lossless,
        config.grayscale,
        &mut w,
    );
    let alpha_bits = alpha_plane.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    let eff_bits = (max_bp as u32).max(alpha_bits);
    encode_frame_lossless(&image3s, alpha_plane.as_ref(), eff_bits, &mut w);
    let codestream = w.into_bytes();
    let alpha_bits = alpha_plane.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    if needs_level_10(max_bp as u32, true, alpha_bits) {
        Ok(wrap_jxl_container(codestream, 10))
    } else {
        Ok(codestream)
    }
}

/// Write a single dimension using JXL's 4-bucket variable-length encoding.
fn write_size(size: u32, w: &mut BitWriter) {
    let size_minus_one = size - 1;
    const BUCKET_BITS: [u32; 4] = [9, 13, 18, 30];
    for (selector, &bits) in BUCKET_BITS.iter().enumerate() {
        if size_minus_one < (1 << bits) {
            w.write(2, selector as u64);
            w.write(bits as usize, size_minus_one as u64);
            return;
        }
    }
    unreachable!("dimension was bounds-checked against MAX_DIMENSION");
}

fn write_size_header(xsize: usize, ysize: usize, w: &mut BitWriter) {
    assert!(
        xsize <= MAX_DIMENSION && ysize <= MAX_DIMENSION,
        "image too large: max dimension is {MAX_DIMENSION}"
    );
    w.write(1, 0); // small = false (use full dimension fields)
    write_size(ysize as u32, w);
    w.write(3, 0); // ratio = 0 (no fixed aspect)
    write_size(xsize as u32, w);
}

/// Whether the content requires codestream **level 10**: a modular channel
/// exceeding 16 bits. That happens for a 16-bit (or wider) alpha extra channel,
/// or for >=16-bit lossless (YCoCg-R color residuals are 17-bit). Level 5
/// (the implicit level of a bare codestream) caps modular at 16 bits, so such
/// files MUST be wrapped in a container declaring level 10 or a conformant
/// decoder rejects them.
fn needs_level_10(bits: u32, lossless: bool, alpha_bits: u32) -> bool {
    (lossless && bits >= 16) || alpha_bits >= 16
}

/// Wrap a bare codestream in a minimal JXL (ISO BMFF) container that declares
/// `level` via a `jxll` box. Box order: signature, ftyp, jxll, jxlc.
fn wrap_jxl_container(codestream: Vec<u8>, level: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(codestream.len() + 41);
    // JXL signature box.
    out.extend_from_slice(&[
        0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
    ]);
    // ftyp box (major brand "jxl ", minor 0, compatible "jxl ").
    out.extend_from_slice(&[
        0, 0, 0, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0, 0, 0, 0, b'j', b'x',
        b'l', b' ',
    ]);
    // jxll level box.
    out.extend_from_slice(&[0, 0, 0, 0x09, b'j', b'x', b'l', b'l', level]);
    // jxlc codestream box (64-bit largesize form if it would overflow u32).
    let payload = 8u64 + codestream.len() as u64;
    if payload <= u32::MAX as u64 {
        out.extend_from_slice(&(payload as u32).to_be_bytes());
        out.extend_from_slice(b"jxlc");
    } else {
        out.extend_from_slice(&1u32.to_be_bytes()); // size == 1 → largesize follows
        out.extend_from_slice(b"jxlc");
        out.extend_from_slice(&(payload + 8).to_be_bytes());
    }
    out.extend_from_slice(&codestream);
    out
}

fn write_int_bit_depth(bits: u32, w: &mut BitWriter) {
    w.write(1, 0); // floating_point_sample = false
    match bits {
        8 => w.write(2, 0),
        10 => w.write(2, 1),
        12 => w.write(2, 2),
        _ => {
            w.write(2, 3); // selector 3 → BitsOffset(6, 1)
            w.write(6, (bits - 1) as u64);
        }
    }
}

/// Write a JXL `BitDepth` field for an **IEEE-754 float** sample.
///
/// Layout: `floating_point_sample` (1 bit, = 1) followed by
/// `U32(Val(32), Val(16), Val(24), BitsOffset(6, 1))` for the total bit width
/// (f32 → selector 0, f16 → selector 1), then 4 bits = `exp_bits - 1`
/// (f32: 8 → 7 = 0b0111; f16: 5 → 4 = 0b0100).
fn write_float_bit_depth(bits: u32, exp_bits: u32, w: &mut BitWriter) {
    w.write(1, 1); // floating_point_sample = true
    match bits {
        32 => w.write(2, 0),
        16 => w.write(2, 1),
        24 => w.write(2, 2),
        _ => {
            w.write(2, 3);
            w.write(6, (bits - 1) as u64);
        }
    }
    w.write(4, (exp_bits - 1) as u64);
}

fn write_image_metadata(
    color_encoding: &ColorEncoding,
    alpha: Option<&AlphaPlane>,
    icc_profile: Option<&[u8]>,
    bps: BitsPerSample,
    lossless: bool,
    grayscale: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(1, 0); // extra_fields = false
    if bps.is_float() {
        write_float_bit_depth(bps.bits(), bps.exp_bits(), w);
    } else {
        write_int_bit_depth(bps.bits(), w);
    }
    // modular_16_bit_buffer_sufficient: a modular integer channel of N bits needs
    // N+1 signed buffer bits (libjxl enc_modular). So any 16-bit modular channel
    // needs 17 bits → 16-bit buffers are NOT sufficient. Modular channels here:
    //   * lossless: the color image (YCoCg-R residuals, 17-bit at 16-bit input);
    //   * lossy or lossless: a 16-bit alpha extra channel (values up to 65535).
    let alpha_bits = alpha.map(|a| a.bits() as u32).unwrap_or(0);
    let needs_32 = (lossless && bps.bits() >= 16) || alpha_bits >= 16;
    w.write(1, if needs_32 { 0 } else { 1 });

    if let Some(alpha) = alpha {
        w.write(2, 1); // num_extra_channels = 1
        // Alpha ECI bit depth MUST match the actual stored sample width.
        // - 8-bit alpha: U8 (all_default=true gives U8 inherently)
        // - 10-bit alpha: declare ECI as 10-bit explicitly
        // - 12-bit alpha: declare ECI as 12-bit explicitly
        //
        // Mismatches show as wrong opacity: e.g. declaring 10-bit while storing
        // u8 makes the decoder compute opacity = stored/1023 ≈ 0.25 for fully-
        // opaque pixels.
        match alpha.bits() {
            8 => {
                w.write(1, 1); // all_default = true → U8 alpha
            }
            bits => {
                w.write(1, 0); // all_default = false
                w.write(2, 0); // ec_type = Alpha (selector 0)
                if alpha.is_float() {
                    write_float_bit_depth(32, 8, w); // f32 alpha bit depth
                } else {
                    write_int_bit_depth(bits as u32, w);
                }
                w.write(2, 0); // dim_shift = 0
                w.write(2, 0); // name length = 0
                w.write(1, 0); // alpha_associated = false
            }
        }
    } else {
        w.write(2, 0); // num_extra_channels = 0
    }

    // For lossy VarDCT we use XYB (=1). For lossless modular, the codestream
    // carries un-transformed pixel values (xyb_encoded = 0).
    w.write(1, if lossless { 0 } else { 1 }); // xyb_encoded
    let want_icc = icc_profile.is_some();
    write_color_encoding_with_icc(color_encoding, want_icc, grayscale, w);
    // tone_mapping is conditional on extra_fields (which we set to 0), so it's absent.
    w.write(2, 0); // extensions: U64 selector = 0 (no extensions)
    // End of ImageMetadata. Now CustomTransformData (part of FileHeader, but kept here for
    // backward-compatible bit alignment with the no-ICC path).
    w.write(1, 1); // CustomTransformData.all_default = 1
    // ICC stream goes AFTER FileHeader, before the zero-pad to byte boundary.
    if let Some(icc) = icc_profile {
        crate::icc_codec::write_icc_stream(icc, w);
    }
    w.zero_pad_to_byte();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn quality_mapping_anchor_points() {
        // Matches the published libjxl mapping breakpoints.
        assert!(close(distance_from_quality(100.0), 0.1));
        assert!(close(distance_from_quality(90.0), 1.0));
        assert!(close(distance_from_quality(30.0), 6.4));
        // Below 30, the formula is quadratic-ish; q=25 must come in just above 6.4.
        let d25 = distance_from_quality(25.0);
        assert!(d25 > 6.4 && d25 < 7.0, "q=25 -> {d25}");
        // Far below 30, it climbs and clamps at 25.
        assert!(close(distance_from_quality(0.0), 25.0));
        assert!(close(distance_from_quality(-50.0), 25.0));
        // Above 100, clamped: 110 -> 100 -> 0.1.
        assert!(close(distance_from_quality(110.0), 0.1));
    }

    #[test]
    fn quality_mapping_monotone() {
        // Strictly decreasing in quality.
        let mut prev = distance_from_quality(0.0);
        for q in 1..=100 {
            let d = distance_from_quality(q as f32);
            assert!(d <= prev, "non-monotonic at q={q}: prev={prev}, d={d}");
            prev = d;
        }
    }

    #[test]
    #[should_panic(expected = "quality must not be NaN")]
    fn quality_nan_panics() {
        let _ = distance_from_quality(f32::NAN);
    }
}

#[cfg(test)]
mod encode_smoke_tests {
    use super::*;

    const W: usize = 16;
    const H: usize = 16;

    fn rgb8() -> Vec<u8> {
        (0..W * H * 3).map(|i| (i % 256) as u8).collect()
    }

    fn rgba8() -> Vec<u8> {
        (0..W * H * 4).map(|i| (i % 256) as u8).collect()
    }

    fn rgb16() -> Vec<u16> {
        (0..W * H * 3).map(|i| (i % 65536) as u16).collect()
    }

    fn rgba16() -> Vec<u16> {
        (0..W * H * 4).map(|i| (i % 65536) as u16).collect()
    }

    fn gray8() -> Vec<u8> {
        (0..W * H).map(|i| (i % 256) as u8).collect()
    }

    fn gray_alpha8() -> Vec<u8> {
        (0..W * H * 2).map(|i| (i % 256) as u8).collect()
    }

    fn gray16() -> Vec<u16> {
        (0..W * H).map(|i| (i % 65536) as u16).collect()
    }

    fn gray_alpha16() -> Vec<u16> {
        (0..W * H * 2).map(|i| (i % 65536) as u16).collect()
    }

    fn rgb10() -> Vec<u16> {
        (0..W * H * 3).map(|i| (i % 1024) as u16).collect()
    }

    fn rgba10() -> Vec<u16> {
        (0..W * H * 4).map(|i| (i % 1024) as u16).collect()
    }

    fn rgb12() -> Vec<u16> {
        (0..W * H * 3).map(|i| (i % 4096) as u16).collect()
    }

    fn rgba12() -> Vec<u16> {
        (0..W * H * 4).map(|i| (i % 4096) as u16).collect()
    }

    fn gray10() -> Vec<u16> {
        (0..W * H).map(|i| (i % 1024) as u16).collect()
    }

    fn gray_alpha10() -> Vec<u16> {
        (0..W * H * 2).map(|i| (i % 1024) as u16).collect()
    }

    fn gray12() -> Vec<u16> {
        (0..W * H).map(|i| (i % 4096) as u16).collect()
    }

    fn gray_alpha12() -> Vec<u16> {
        (0..W * H * 2).map(|i| (i % 4096) as u16).collect()
    }

    fn rgb_f32() -> Vec<f32> {
        (0..W * H * 3).map(|i| (i % 256) as f32 / 255.0).collect()
    }

    fn rgba_f32() -> Vec<f32> {
        (0..W * H * 4).map(|i| (i % 256) as f32 / 255.0).collect()
    }

    fn gray_f32() -> Vec<f32> {
        (0..W * H).map(|i| (i % 256) as f32 / 255.0).collect()
    }

    fn lossy() -> EncodeConfig {
        EncodeConfig::default().with_distance(1.0)
    }

    fn lossless() -> EncodeConfig {
        EncodeConfig::default().with_lossless(true)
    }

    fn ok(r: Result<Vec<u8>, EncodeError>) {
        let bytes = r.expect("encode failed");
        assert!(!bytes.is_empty(), "encoded output is empty");
    }

    // --- encode_image (RGB u8) ---

    #[test]
    fn rgb8_lossy() {
        ok(encode_image(&rgb8(), W, H, &lossy()));
    }

    #[test]
    fn rgb8_lossless() {
        ok(encode_image(&rgb8(), W, H, &lossless()));
    }

    #[test]
    fn rgb8_quality() {
        ok(encode_image(
            &rgb8(),
            W,
            H,
            &EncodeConfig::default().with_quality(85.0),
        ));
    }

    // --- encode_image_with_alpha (RGBA u8) ---

    #[test]
    fn rgba8_lossy() {
        ok(encode_image_with_alpha(&rgba8(), W, H, &lossy()));
    }

    #[test]
    fn rgba8_lossless() {
        ok(encode_image_with_alpha(&rgba8(), W, H, &lossless()));
    }

    // --- 16-bit RGB / RGBA ---

    #[test]
    fn rgb16_lossy() {
        ok(encode_image_16bit(&rgb16(), W, H, &lossy()));
    }

    #[test]
    fn rgb16_lossless() {
        ok(encode_image_16bit(&rgb16(), W, H, &lossless()));
    }

    #[test]
    fn rgba16_lossy() {
        ok(encode_image_with_alpha_16bit(&rgba16(), W, H, &lossy()));
    }

    #[test]
    fn rgba16_lossless() {
        ok(encode_image_with_alpha_16bit(&rgba16(), W, H, &lossless()));
    }

    // --- 10-bit RGB / RGBA ---

    #[test]
    fn rgb10_lossy() {
        ok(encode_image_10bit(&rgb10(), W, H, &lossy()));
    }

    #[test]
    fn rgb10_lossless() {
        ok(encode_image_10bit(&rgb10(), W, H, &lossless()));
    }

    #[test]
    fn rgba10_lossy() {
        ok(encode_image_with_alpha_10bit(&rgba10(), W, H, &lossy()));
    }

    #[test]
    fn rgba10_lossless() {
        ok(encode_image_with_alpha_10bit(&rgba10(), W, H, &lossless()));
    }

    // --- 12-bit RGB / RGBA ---

    #[test]
    fn rgb12_lossy() {
        ok(encode_image_12bit(&rgb12(), W, H, &lossy()));
    }

    #[test]
    fn rgb12_lossless() {
        ok(encode_image_12bit(&rgb12(), W, H, &lossless()));
    }

    #[test]
    fn rgba12_lossy() {
        ok(encode_image_with_alpha_12bit(&rgba12(), W, H, &lossy()));
    }

    #[test]
    fn rgba12_lossless() {
        ok(encode_image_with_alpha_12bit(&rgba12(), W, H, &lossless()));
    }

    // --- float RGB / RGBA ---

    #[test]
    fn rgb_f32_lossy() {
        ok(encode_image_f32(&rgb_f32(), W, H, &lossy()));
    }

    #[test]
    fn rgb_f16_lossy() {
        ok(encode_image_f16(&rgb_f32(), W, H, &lossy()));
    }

    #[test]
    fn rgba_f32_lossy() {
        ok(encode_image_with_alpha_f32(&rgba_f32(), W, H, &lossy()));
    }

    #[test]
    fn rgba_f16_lossy() {
        ok(encode_image_with_alpha_f16(&rgba_f32(), W, H, &lossy()));
    }

    // --- grayscale u8 ---

    #[test]
    fn gray8_lossy() {
        ok(encode_image_gray(&gray8(), W, H, &lossy()));
    }

    #[test]
    fn gray8_lossless() {
        ok(encode_image_gray(&gray8(), W, H, &lossless()));
    }

    #[test]
    fn gray_alpha8_lossy() {
        ok(encode_image_gray_alpha(&gray_alpha8(), W, H, &lossy()));
    }

    #[test]
    fn gray_alpha8_lossless() {
        ok(encode_image_gray_alpha(&gray_alpha8(), W, H, &lossless()));
    }

    // --- grayscale 10-bit ---

    #[test]
    fn gray10_lossy() {
        ok(encode_image_gray_10bit(&gray10(), W, H, &lossy()));
    }

    #[test]
    fn gray10_lossless() {
        ok(encode_image_gray_10bit(&gray10(), W, H, &lossless()));
    }

    #[test]
    fn gray_alpha10_lossy() {
        ok(encode_image_gray_alpha_10bit(
            &gray_alpha10(),
            W,
            H,
            &lossy(),
        ));
    }

    #[test]
    fn gray_alpha10_lossless() {
        ok(encode_image_gray_alpha_10bit(
            &gray_alpha10(),
            W,
            H,
            &lossless(),
        ));
    }

    // --- grayscale 12-bit ---

    #[test]
    fn gray12_lossy() {
        ok(encode_image_gray_12bit(&gray12(), W, H, &lossy()));
    }

    #[test]
    fn gray12_lossless() {
        ok(encode_image_gray_12bit(&gray12(), W, H, &lossless()));
    }

    #[test]
    fn gray_alpha12_lossy() {
        ok(encode_image_gray_alpha_12bit(
            &gray_alpha12(),
            W,
            H,
            &lossy(),
        ));
    }

    #[test]
    fn gray_alpha12_lossless() {
        ok(encode_image_gray_alpha_12bit(
            &gray_alpha12(),
            W,
            H,
            &lossless(),
        ));
    }

    // --- grayscale 16-bit ---

    #[test]
    fn gray16_lossy() {
        ok(encode_image_gray_16bit(&gray16(), W, H, &lossy()));
    }

    #[test]
    fn gray16_lossless() {
        ok(encode_image_gray_16bit(&gray16(), W, H, &lossless()));
    }

    #[test]
    fn gray_alpha16_lossy() {
        ok(encode_image_gray_alpha_16bit(
            &gray_alpha16(),
            W,
            H,
            &lossy(),
        ));
    }

    #[test]
    fn gray_alpha16_lossless() {
        ok(encode_image_gray_alpha_16bit(
            &gray_alpha16(),
            W,
            H,
            &lossless(),
        ));
    }

    // --- grayscale float ---

    #[test]
    fn gray_f32_lossy() {
        ok(encode_image_gray_f32(&gray_f32(), W, H, &lossy()));
    }

    #[test]
    fn gray_f16_lossy() {
        ok(encode_image_gray_f16(&gray_f32(), W, H, &lossy()));
    }

    // --- error paths ---

    #[test]
    fn empty_width_rejected() {
        assert!(matches!(
            encode_image(&[], 0, H, &lossy()),
            Err(EncodeError::EmptyImage)
        ));
    }

    #[test]
    fn empty_height_rejected() {
        assert!(matches!(
            encode_image(&[], W, 0, &lossy()),
            Err(EncodeError::EmptyImage)
        ));
    }

    #[test]
    fn wrong_buffer_size_rejected() {
        let short = vec![0u8; W * H]; // too short for RGB
        assert!(matches!(
            encode_image(&short, W, H, &lossy()),
            Err(EncodeError::InputSizeMismatch { .. })
        ));
    }

    #[test]
    fn invalid_distance_rejected() {
        assert!(matches!(
            encode_image(&rgb8(), W, H, &EncodeConfig::default().with_distance(-1.0)),
            Err(EncodeError::InvalidDistance(_))
        ));
    }

    #[test]
    fn dimension_too_large_rejected() {
        let huge = MAX_DIMENSION + 1;
        assert!(matches!(
            encode_image(&[], huge, H, &lossy()),
            Err(EncodeError::DimensionTooLarge { .. })
        ));
    }

    // --- 1x1 edge cases ---

    #[test]
    fn one_by_one_rgb8() {
        ok(encode_image(&[128u8, 64, 32], 1, 1, &lossy()));
    }

    #[test]
    fn one_by_one_rgba8_lossless() {
        ok(encode_image_with_alpha(
            &[128u8, 64, 32, 255],
            1,
            1,
            &lossless(),
        ));
    }

    #[test]
    fn one_by_one_gray8() {
        ok(encode_image_gray(&[200u8], 1, 1, &lossy()));
    }
}
