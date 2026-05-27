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
use crate::EncodeError;
use crate::bit_writer::BitWriter;
use crate::color::{lut_high_bit, srgb_to_linear_u8};
use crate::color_encoding::{ColorEncoding, write_color_encoding_with_icc};
use crate::enc_frame::encode_frame;
use crate::enc_lossless::{encode_frame_lossless, forward_ycocg};
use crate::image::{Image3F, Image3Si};

/// 8-bit alpha plane (row-major, stride = `xsize`).
#[derive(Debug, Clone)]
pub enum AlphaPlane {
    /// 8-bit alpha, values 0..=255.
    U8(Vec<u8>),
    /// 10-bit (`bits=10`, values 0..=1023) or 12-bit (`bits=12`, values 0..=4095) alpha.
    U16 { data: Vec<u16>, bits: u8 },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitsPerSample {
    Eight,
    Ten,
    Twelve,
}

impl Default for BitsPerSample {
    fn default() -> Self {
        Self::Eight
    }
}

impl BitsPerSample {
    pub(crate) fn bits(self) -> u32 {
        match self {
            BitsPerSample::Eight => 8,
            BitsPerSample::Ten => 10,
            BitsPerSample::Twelve => 12,
        }
    }
}

impl AlphaPlane {
    /// Create an 8-bit alpha plane.
    #[inline]
    pub fn from_u8(data: Vec<u8>) -> Self {
        Self::U8(data)
    }

    /// Create a 10-bit alpha plane (values 0..=1023).
    #[inline]
    pub fn from_u16_10bit(data: Vec<u16>) -> Self {
        Self::U16 { data, bits: 10 }
    }

    /// Create a 12-bit alpha plane (values 0..=4095).
    #[inline]
    pub fn from_u16_12bit(data: Vec<u16>) -> Self {
        Self::U16 { data, bits: 12 }
    }

    /// Number of pixels.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16 { data, .. } => data.len(),
        }
    }

    /// Bit depth of the alpha samples (8, 10, or 12).
    #[inline]
    pub fn bits(&self) -> u8 {
        match self {
            Self::U8(_) => 8,
            Self::U16 { bits, .. } => *bits,
        }
    }

    /// Read pixel `idx` as `i32`.  Encoder hot path — kept tiny for inlining.
    #[inline]
    pub fn get_i32(&self, idx: usize) -> i32 {
        match self {
            Self::U8(v) => v[idx] as i32,
            Self::U16 { data, .. } => data[idx] as i32,
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
    pub distance: f32,
    pub color_encoding: ColorEncoding,
    pub icc_profile: Option<Vec<u8>>,
    pub alpha: Option<AlphaPlane>,
    /// Bit depth declared in the codestream (default: 8).
    pub bits_per_sample: BitsPerSample,
    /// If true, encode losslessly via the modular encoder. `distance` is then
    /// ignored. RGB and alpha both round-trip bit-perfectly.
    pub lossless: bool,
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
        }
    }
}

impl EncodeConfigImpl {
    /// Convenience builder with the given butteraugli distance and otherwise
    /// default settings (sRGB primaries, linear transfer).
    pub fn with_distance(distance: f32) -> Self {
        Self {
            distance,
            ..Self::default()
        }
    }

    /// Attach an ICC profile. **Panics at encode time** — see field docs.
    pub fn with_icc_profile(mut self, icc: Option<Vec<u8>>) -> Self {
        self.icc_profile = icc;
        self
    }

    /// Attach an 8-bit alpha plane to be encoded losslessly via Modular.
    /// Length must equal `xsize * ysize` of the image passed to encode.
    pub fn with_alpha(mut self, alpha: AlphaPlane) -> Self {
        self.alpha = Some(alpha);
        self
    }

    pub fn with_bits_per_sample(mut self, bps: BitsPerSample) -> Self {
        self.bits_per_sample = bps;
        self
    }

    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Replace the color encoding (white point / primaries / transfer / intent).
    pub fn with_color_encoding(mut self, enc: ColorEncoding) -> Self {
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

/// Convert a JPEG-style quality value (0..=100, higher = better) to a
/// butteraugli distance (smaller = better).
///
/// The mapping matches libjxl's `JxlEncoderDistanceFromQuality`:
///   * `quality == 100` → distance ≈ 0.1
///   * `quality == 90`  → distance == 1.0 (the published "visually lossless" point)
///   * `quality == 30`  → distance == 6.4 (knee of the curve)
///   * `quality == 0`   → distance == 25.0 (clamp ceiling)
///
/// For `quality >= 30` the mapping is linear: `distance = 0.1 + (100 - q) * 0.09`.
/// For `quality < 30` it transitions to a faster ramp using `2.5^((30 - q) / 5)`.
/// Out-of-range inputs are clamped (`quality > 100` becomes 100, `quality < 0`
/// becomes 0 by way of the 25.0 clamp).
///
/// `NaN` is rejected.
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
    assert!(width > 0 && height > 0, "empty image");
    assert_eq!(
        input.len(),
        width * height * 3,
        "input buffer size mismatch"
    );
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
    assert!(width > 0 && height > 0, "empty image");
    assert_eq!(
        input.len(),
        width * height * 4,
        "input buffer size mismatch"
    );
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

/// Encode a 10-bit RGBA image (`u16` channels, values 0..=1023).
/// Input is interleaved RGBA, each channel a `u16` in native byte order.
pub fn encode_image_with_alpha_10bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, true, config, BitsPerSample::Ten)
}

/// Encode a 12-bit RGBA image (`u16` channels, values 0..=4095).
/// Input is interleaved RGBA, each channel a `u16` in native byte order.
pub fn encode_image_with_alpha_12bit(
    input: &[u16],
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_high_depth_rgba(input, width, height, true, config, BitsPerSample::Twelve)
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
    let expected = width * height * 4;
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
                .with_icc_profile(config.icc_profile.clone())
                .with_color_encoding(config.color_encoding),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);
    let mut alpha_plane = vec![0u16; width * height];

    let lut = &lut_high_bit(bps.bits() as u8).table;

    let bp_max = (1 << bps.bits()) - 1;

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
                BitsPerSample::Eight => unreachable!("high-depth path called with 8-bit bps"),
            })
            .with_bits_per_sample(bps)
            .with_icc_profile(config.icc_profile.clone())
            .with_color_encoding(config.color_encoding),
    )
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
        let expected = input.xsize() * input.ysize();
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
        &mut w,
    );
    encode_frame(distance, input, config.alpha.as_ref(), &mut w);
    encode_frame(distance, input, config.alpha.as_ref(), &mut w);
    Ok(w.into_bytes())
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
    let expected = width * height * if has_alpha { 4 } else { 3 };
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
        &mut w,
    );
    encode_frame_lossless(&image3s, alpha_plane.as_ref(), &mut w);
    Ok(w.into_bytes())
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

fn write_image_metadata(
    color_encoding: &ColorEncoding,
    alpha: Option<&AlphaPlane>,
    icc_profile: Option<&[u8]>,
    bps: BitsPerSample,
    lossless: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(1, 0); // extra_fields = false
    w.write(1, 0); // floating_point_sample = false
    match bps {
        BitsPerSample::Eight => w.write(2, 0),  // selector 0 → 8
        BitsPerSample::Ten => w.write(2, 1),    // selector 1 → 10
        BitsPerSample::Twelve => w.write(2, 2), // selector 2 → 12
    }
    w.write(1, 1); // modular_16_bit_buffer_sufficient = true

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
                w.write(1, 0); // floating_point_sample = false
                match bits {
                    10 => w.write(2, 1), // selector 1 → 10
                    12 => w.write(2, 2), // selector 2 → 12
                    _ => panic!("unsupported alpha bit depth: {bits}"),
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
    write_color_encoding_with_icc(color_encoding, want_icc, w);
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
