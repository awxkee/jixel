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
use crate::coder_scratch::CoderScratch;
use crate::color::{lut_high_bit, srgb_lut, srgb_to_linear_f32};
use crate::color_encoding::write_color_encoding_with_icc;
use crate::dark_aq::DarkAqConfig;
use crate::encoding_context::EncodingContext;
use crate::frame::encode_frame;
use crate::image::{Image3F, Image3Si};
use crate::jpeg::BrotliCompression;
use crate::lossless::{encode_frame_lossless, encode_frame_lossless_float, forward_ycocg};
use crate::orientation::Orientation;
use crate::xyb::XybMatrix;
use crate::{ColorEncoding, EncodeError};
use std::num::NonZero;
use std::sync::Arc;
use std::thread::available_parallelism;

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

/// The JXL ToneMapping bundle (HDR luminance metadata). All fields are
/// optional refinements on top of the SDR defaults; set them for fuller
/// mastering metadata. Constraints (enforced by decoders): `intensity_target`
/// must be > 0 and `0 <= min_nits <= intensity_target`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ToneMappingParams {
    /// Peak display luminance in nits. `None` keeps the 255-nit SDR default.
    pub intensity_target: Option<f32>,
    /// Minimum display luminance (mastering black level) in nits. Default 0.
    pub min_nits: f32,
    /// If true, sample values are relative to the max display luminance. Default false.
    pub relative_to_max_display: bool,
    /// Luminance below which the transfer is linear. Relative (0..1) when
    /// `relative_to_max_display`, otherwise in nits. Default 0.
    pub linear_below: f32,
}

impl ToneMappingParams {
    /// True when every field is at its JXL default (no ToneMapping bundle needed).
    fn is_default(&self) -> bool {
        self.intensity_target.is_none()
            && self.min_nits == 0.0
            && !self.relative_to_max_display
            && self.linear_below == 0.0
    }
}

/// Encoder speed/transform-search tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Speed {
    /// No transform search at all: every block is coded as a plain 8×8 DCT.
    /// Skips everything `Fast` skips, plus the square-merge selection.
    Fastest,
    #[default]
    Fast,
    Slow,
}

/// Decode-speed/density tradeoff for **lossless** encoding. Ignored for lossy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodingSpeed {
    /// No Weighted Predictor and no meta-adaptive context trees: the decoder
    /// takes its fixed-predictor fast path. Roughly 8-10x faster decode than
    /// `Slow` for `Speed::Slow` encodes, at ~+15..30% size.
    Fastest,
    /// No Weighted Predictor. Roughly 1.15-1.2x faster decode for
    /// `Speed::Slow` encodes (~+1..3% size), ~3.5x for `Speed::Fast` encodes
    /// (~+4% size).
    Fast,
    /// All coding tools; densest output, slowest to decode.
    #[default]
    Slow,
}

impl DecodingSpeed {
    /// Whether the emitted stream may use the Weighted Predictor (and the
    /// WP-error tree property, which also forces decoder-side WP state).
    pub(crate) fn use_weighted_predictor(self) -> bool {
        matches!(self, DecodingSpeed::Slow)
    }

    /// Whether the encoder may signal per-pixel meta-adaptive context trees.
    pub(crate) fn use_ma_trees(self) -> bool {
        !matches!(self, DecodingSpeed::Fastest)
    }
}

#[derive(Debug, Clone)]
pub struct EncodeConfig {
    pub distance: f32,
    pub color_encoding: ColorEncoding,
    pub icc_profile: Option<Vec<u8>>,
    /// EXIF/TIFF metadata to embed via an `Exif` container box. Forces the
    /// output into the JXL container form. Raw TIFF bytes (no "Exif\0\0" prefix).
    pub exif: Option<Vec<u8>>,
    /// XMP packet to embed via an `xml ` container box, or a `brob` box when a
    /// Brotli implementation is supplied. Forces the output into container form.
    pub xmp: Option<Vec<u8>>,
    /// Optional Brotli encoder for EXIF and XMP metadata boxes.
    ///
    /// When set, metadata is emitted in `brob` boxes. When absent, it remains
    /// in uncompressed `Exif` and `xml ` boxes.
    pub brotli_compression: Option<Arc<dyn BrotliCompression>>,
    /// Image orientation: how the decoder should rotate/flip the stored pixels
    /// for display (default [`Orientation::Normal`]).
    pub orientation: Orientation,
    /// If true, encode losslessly via the modular encoder. `distance` is then
    /// ignored. RGB and alpha both round-trip bit-perfectly.
    pub lossless: bool,
    /// If true (and lossless), use the progressive Squeeze transform: a low-res
    /// preview that refines to bit-exact. Works at any size (single- and
    /// multi-group) and with alpha.
    ///
    /// For lossy (`!lossless`) encoding, `true` selects 2-pass progressive
    /// VarDCT (equivalent to `progressive_passes = Some(2)`).
    pub progressive: bool,
    /// Detect exact repeated, 16x16-aligned regions and encode them through the
    /// JPEG XL patch dictionary. Supported for both lossless Modular and lossy
    /// VarDCT encoding. The encoder measures the complete normal and patched
    /// representations and keeps patches only when they reduce the final rate.
    pub patches: bool,
    /// Number of VarDCT passes for **lossy** progressive encoding. `None` falls
    /// back to `progressive` (2 passes if set, else 1). `Some(1)` = single pass;
    /// `Some(n)` for n in 2..=4 = n-pass progressive with an automatic
    /// coarse-to-fine bit-shift schedule (`[n-1, .., 1, 0]`). Values above 4 are
    /// clamped (the per-pass shift field is 2 bits with no downsampling).
    /// Ignored when `lossless`. Overridden by `progressive_shifts` if set.
    pub progressive_passes: Option<u32>,
    /// Explicit per-pass coefficient-shift schedule for lossy progressive
    /// encoding, overriding `progressive_passes`. The length is the pass count;
    /// each entry is the decoder left-shift for that pass (0..=3). A larger
    /// pass-0 shift makes the first (preview) pass smaller and coarser. Must end
    /// in 0 and contain only values 0..=3; otherwise it is ignored. Examples:
    /// `[1,0]` (default 2-pass), `[2,0]` / `[3,0]` (coarser thumbnail then full),
    /// `[2,1,0]` (3-step ramp). Ignored when `lossless`.
    pub progressive_shifts: Option<Vec<u32>>,
    /// HDR luminance signaling. When `Some(nits)`, the codestream's tone-mapping
    /// metadata declares this peak display luminance (`intensity_target`), e.g.
    /// `10000.0` for PQ/HDR10 or a measured mastering peak. `None` leaves the
    /// SDR default. Must be > 0.
    pub intensity_target: Option<f32>,
    /// HDR mastering black level in nits (ToneMapping `min_nits`). Default 0.
    pub min_nits: f32,
    /// ToneMapping `relative_to_max_display`. Default false.
    pub relative_to_max_display: bool,
    /// ToneMapping `linear_below` (nits, or 0..1 if relative). Default 0.
    pub linear_below: f32,
    pub num_threads: usize,
    pub speed: Speed,
    /// Decode-speed/density tradeoff for lossless encoding (see
    /// [`DecodingSpeed`]). Ignored for lossy.
    pub decoding_speed: DecodingSpeed,
    /// Optional superblock Variance-Boost / Dark-AQ modulation of the quant field
    /// (see [`DarkAqConfig`]). `None` (default) leaves the quant field untouched. Ignored
    /// for lossless. `Some(BoostCfg::default())` enables the validated Dark-AQ preset.
    pub boost: Option<DarkAqConfig>,
}

#[derive(Debug, Clone)]
pub(crate) struct EncodeConfigImpl {
    pub(crate) distance: f32,
    pub(crate) color_encoding: ColorEncoding,
    pub(crate) icc_profile: Option<Vec<u8>>,
    pub(crate) exif: Option<Vec<u8>>,
    pub(crate) xmp: Option<Vec<u8>>,
    pub(crate) brotli_compression: Option<Arc<dyn BrotliCompression>>,
    pub(crate) orientation: Orientation,
    pub(crate) alpha: Option<AlphaPlane>,
    /// Bit depth declared in the codestream (default: 8).
    pub(crate) bits_per_sample: BitsPerSample,
    /// If true, encode losslessly via the modular encoder. `distance` is then
    /// ignored. RGB and alpha both round-trip bit-perfectly.
    pub(crate) lossless: bool,
    pub(crate) progressive: bool,
    pub(crate) patches: bool,
    /// Number of lossy VarDCT passes (see `EncodeConfig::progressive_passes`).
    pub(crate) progressive_passes: Option<u32>,
    /// Explicit per-pass shift schedule (see `EncodeConfig::progressive_shifts`).
    pub(crate) progressive_shifts: Option<Vec<u32>>,
    /// If true, the image is grayscale: the codestream declares a Gray color
    /// space so the decoder emits a single-channel (L / LA) image. Internally
    /// the data still flows through the XYB pipeline with R=G=B, so the X and B
    /// chroma channels are ~constant and cost almost nothing.
    pub(crate) grayscale: bool,
    /// HDR peak display luminance in nits (see `EncodeConfig::intensity_target`).
    pub(crate) intensity_target: Option<f32>,
    pub(crate) min_nits: f32,
    pub(crate) relative_to_max_display: bool,
    pub(crate) linear_below: f32,
    /// Worker-thread count for VarDCT encoding (see `EncodeConfig::num_threads`).
    pub(crate) num_threads: usize,
    pub(crate) speed: Speed,
    /// Decode-speed/density tradeoff for lossless (see `EncodeConfig::decoding_speed`).
    pub(crate) decoding_speed: DecodingSpeed,
    /// Superblock Variance-Boost / Dark-AQ config (see `EncodeConfig::boost`).
    pub(crate) dark_aq: Option<DarkAqConfig>,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            distance: 1.0,
            color_encoding: ColorEncoding::default(),
            icc_profile: None,
            exif: None,
            xmp: None,
            brotli_compression: None,
            orientation: Orientation::Normal,
            lossless: false,
            progressive: false,
            patches: false,
            progressive_passes: None,
            progressive_shifts: None,
            intensity_target: None,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
            num_threads: available_parallelism()
                .unwrap_or(NonZero::new(1).unwrap())
                .get(),
            speed: Speed::Fast,
            decoding_speed: DecodingSpeed::Slow,
            boost: Some(DarkAqConfig::default()),
        }
    }
}

impl Default for EncodeConfigImpl {
    fn default() -> Self {
        Self {
            distance: 1.0,
            color_encoding: ColorEncoding::default(),
            icc_profile: None,
            exif: None,
            xmp: None,
            brotli_compression: None,
            orientation: Orientation::Normal,
            alpha: None,
            bits_per_sample: BitsPerSample::Eight,
            lossless: false,
            progressive: false,
            patches: false,
            grayscale: false,
            progressive_passes: None,
            progressive_shifts: None,
            intensity_target: None,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
            num_threads: available_parallelism()
                .unwrap_or(NonZero::new(1).unwrap())
                .get(),
            speed: Speed::Fast,
            decoding_speed: DecodingSpeed::Slow,
            dark_aq: Some(DarkAqConfig::default()),
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

    /// Set the worker-thread count (see `EncodeConfig::num_threads`).
    pub(crate) fn with_num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Attach an ICC profile. **Panics at encode time** — see field docs.
    pub(crate) fn with_icc_profile(mut self, icc: Option<Vec<u8>>) -> Self {
        self.icc_profile = icc;
        self
    }
    pub(crate) fn with_exif(mut self, exif: Option<Vec<u8>>) -> Self {
        self.exif = exif;
        self
    }
    pub(crate) fn with_xmp(mut self, xmp: Option<Vec<u8>>) -> Self {
        self.xmp = xmp;
        self
    }
    pub(crate) fn with_brotli_compression(
        mut self,
        compressor: Option<Arc<dyn BrotliCompression>>,
    ) -> Self {
        self.brotli_compression = compressor;
        self
    }
    pub(crate) fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
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

    pub(crate) fn with_progressive(mut self, progressive: bool) -> Self {
        self.progressive = progressive;
        self
    }

    pub(crate) fn with_patches(mut self, patches: bool) -> Self {
        self.patches = patches;
        self
    }

    pub(crate) fn with_progressive_passes(mut self, passes: Option<u32>) -> Self {
        self.progressive_passes = passes;
        self
    }

    pub(crate) fn with_progressive_shifts(mut self, shifts: Option<Vec<u32>>) -> Self {
        self.progressive_shifts = shifts;
        self
    }

    /// Copy all lossy-progressive settings from a public `EncodeConfig`.
    pub(crate) fn with_progressive_from(self, config: &EncodeConfig) -> Self {
        self.with_progressive(config.progressive)
            .with_patches(config.patches)
            .with_progressive_passes(config.progressive_passes)
            .with_progressive_shifts(config.progressive_shifts.clone())
            .with_speed(config.speed)
            .with_decoding_speed(config.decoding_speed)
            .with_boost(config.boost)
    }

    pub(crate) fn with_boost(mut self, boost: Option<DarkAqConfig>) -> Self {
        self.dark_aq = boost;
        self
    }

    pub(crate) fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    pub(crate) fn with_decoding_speed(mut self, decoding_speed: DecodingSpeed) -> Self {
        self.decoding_speed = decoding_speed;
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

    pub(crate) fn tone_mapping(&self) -> ToneMappingParams {
        ToneMappingParams {
            intensity_target: self.intensity_target,
            min_nits: self.min_nits,
            relative_to_max_display: self.relative_to_max_display,
            linear_below: self.linear_below,
        }
    }

    pub(crate) fn with_intensity_target(mut self, nits: Option<f32>) -> Self {
        self.intensity_target = nits;
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

    /// Enable superblock Variance-Boost / Dark-AQ quant-field modulation (lossy only).
    /// [`DarkAqConfig::default`] is the validated Dark-AQ preset. See [`DarkAqConfig`].
    pub fn with_dark_aq_config(mut self, boost: DarkAqConfig) -> Self {
        self.boost = Some(boost);
        self
    }

    /// Enable Dark-AQ with the validated defaults (equivalent to
    /// `with_boost(BoostCfg::default())`).
    pub fn with_dark_aq(self) -> Self {
        self.with_dark_aq_config(DarkAqConfig::default())
    }

    /// Replace the color encoding (white point / primaries / transfer / intent).
    pub fn with_color_encoding(mut self, enc: ColorEncoding) -> Self {
        self.color_encoding = enc;
        self
    }

    /// Signal HDR peak display luminance (nits) in the tone-mapping metadata.
    pub fn with_intensity_target(mut self, nits: f32) -> Self {
        self.intensity_target = Some(nits);
        self
    }

    /// HDR mastering black level in nits (ToneMapping `min_nits`).
    pub fn with_min_nits(mut self, nits: f32) -> Self {
        self.min_nits = nits;
        self
    }

    /// Set ToneMapping `relative_to_max_display`.
    pub fn with_relative_to_max_display(mut self, rel: bool) -> Self {
        self.relative_to_max_display = rel;
        self
    }

    /// Set ToneMapping `linear_below` (nits, or 0..1 when relative).
    pub fn with_linear_below(mut self, v: f32) -> Self {
        self.linear_below = v;
        self
    }

    pub(crate) fn tone_mapping(&self) -> ToneMappingParams {
        ToneMappingParams {
            intensity_target: self.intensity_target,
            min_nits: self.min_nits,
            relative_to_max_display: self.relative_to_max_display,
            linear_below: self.linear_below,
        }
    }

    /// Attach an ICC profile. **Panics at encode time** — see field docs.
    pub fn with_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.icc_profile = Some(icc);
        self
    }
    /// Embed EXIF/TIFF metadata (raw TIFF bytes) via an `Exif` container box.
    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.exif = Some(exif);
        self
    }
    /// Embed an XMP packet via an uncompressed `xml ` container box.
    pub fn with_xmp(mut self, xmp: Vec<u8>) -> Self {
        self.xmp = Some(xmp);
        self
    }
    /// Compress EXIF and XMP metadata using a caller-provided Brotli encoder.
    pub fn with_brotli_compression(mut self, compressor: Box<dyn BrotliCompression>) -> Self {
        self.brotli_compression = Some(Arc::from(compressor));
        self
    }
    /// Set the image [`Orientation`].
    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }
    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Enable progressive lossless. Only effective when `lossless`.
    pub fn with_progressive(mut self, progressive: bool) -> Self {
        self.progressive = progressive;
        self
    }

    /// Enable exact patch-dictionary matching with full rate comparison.
    pub fn with_patches(mut self, patches: bool) -> Self {
        self.patches = patches;
        self
    }

    /// Set the worker-thread count (see `EncodeConfig::num_threads`).
    pub fn with_num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Select the transform-search speed/effort tradeoff.
    pub fn with_decoding_speed(mut self, decoding_speed: DecodingSpeed) -> Self {
        self.decoding_speed = decoding_speed;
        self
    }

    pub fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        if speed == Speed::Slow {
            self.patches = true;
        }
        self
    }
}

pub fn distance_from_quality(quality: f32) -> f32 {
    assert!(!quality.is_nan(), "quality must not be NaN");
    let q = quality.clamp(0.0, 100.0);
    let d = if q >= 99.0 {
        // Reserve the final quality point for the practical VarDCT ceiling:
        // q99 = 0.10, q100 = 0.05.
        0.05 * 2.0f32.powf(100.0 - q)
    } else if q >= 90.0 {
        // Logarithmic upper range: q90 = 1.0, q99 = 0.1.
        10.0f32.powf((99.0 - q) / 9.0 - 1.0)
    } else if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        6.24 + 2.5f32.powf((30.0 - q) / 5.0) / 6.25
    };
    d.min(25.0)
}

/// Adaptive B-bias opsin swap for yellow content that quantization would
/// desaturate (two tiers, `Speed::Slow` only — see `yellow_opsin`). Must run
/// after linearization and before any XYB conversion; the non-spec matrix is
/// signaled via the explicit CustomTransformData bundle in `write_headers`.
fn apply_yellow_opsin(ctx: &mut EncodingContext, linear: &Image3F, distance: f32) {
    if ctx.speed != Speed::Slow {
        return;
    }
    let selection = crate::yellow_opsin::select_yellow(linear, distance);
    if let Some(m) = selection.matrix {
        ctx.xyb = m;
    }
    // The yellow selector runs before XYB conversion, so the post-XYB
    // `x_heavy` flag is not available yet. Apply only its own staged B scale
    // here; `frame::encode_frame` ORs in the X-gradient decision later.
    ctx.raise_b_qm_scale(selection.b_qm_scale);
}

fn lossy_context(
    config: &EncodeConfig,
    distance: f32,
    xyb: XybMatrix,
    pixels: usize,
) -> EncodingContext {
    // Fastest's work per pixel is deliberately small, so oversubscribing tiny
    // images spends more time constructing and synchronizing workers than it
    // saves. Keep the requested count as a maximum and give each active lane
    // roughly 64K pixels; larger images still receive the full thread budget.
    let num_threads = if config.speed == Speed::Fastest {
        config.num_threads.min(pixels.div_ceil(64 * 1024).max(1))
    } else {
        config.num_threads
    };
    EncodingContext::new(config.speed, config.boost, xyb, distance, num_threads)
}

fn for_each_linear_band<F>(
    image: &mut Image3F,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    f: F,
) where
    F: Fn(usize, [&mut [f32]; 3]) + Sync,
{
    let mut pixel_start = 0;
    let mut jobs: Vec<_> = image
        .row_bands_mut(ctx.thread_pool.num_threads())
        .into_iter()
        .map(|band| {
            let start = pixel_start;
            pixel_start += band[0].len();
            Some((start, band))
        })
        .collect();

    if jobs.len() <= 1 {
        for job in jobs {
            let (start, band) = job.unwrap();
            f(start, band);
        }
    } else {
        ctx.thread_pool
            .steal_for_each_mut(scratch, &mut jobs, |_i, job, _scratch| {
                let (start, band) = job.take().unwrap();
                f(start, band);
            });
    }
}

fn linearize_rgb<T, F, const CHANNELS: usize>(
    input: &[T],
    width: usize,
    height: usize,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    convert: F,
) -> Image3F
where
    T: Copy + Sync,
    F: Fn(T) -> f32 + Sync,
{
    debug_assert!(CHANNELS >= 3);
    let mut linear = Image3F::new(width, height);
    for_each_linear_band(&mut linear, ctx, scratch, |pixel_start, [r, g, b]| {
        let src = &input[pixel_start * CHANNELS..][..r.len() * CHANNELS];
        for (((r, g), b), px) in r
            .iter_mut()
            .zip(g.iter_mut())
            .zip(b.iter_mut())
            .zip(src.as_chunks::<CHANNELS>().0.iter())
        {
            *r = convert(px[0]);
            *g = convert(px[1]);
            *b = convert(px[2]);
        }
    });
    linear
}

fn linearize_gray<T, F>(
    input: &[T],
    width: usize,
    height: usize,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    convert: F,
) -> Image3F
where
    T: Copy + Sync,
    F: Fn(T) -> f32 + Sync,
{
    let mut linear = Image3F::new(width, height);
    for_each_linear_band(&mut linear, ctx, scratch, |pixel_start, [r, g, b]| {
        let src = &input[pixel_start..][..r.len()];
        for (((r, g), b), &v) in r
            .iter_mut()
            .zip(g.iter_mut())
            .zip(b.iter_mut())
            .zip(src.iter())
        {
            let linear = convert(v);
            *r = linear;
            *g = linear;
            *b = linear;
        }
    });
    linear
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
                .with_progressive(config.progressive)
                .with_patches(config.patches)
                .with_icc_profile(config.icc_profile.clone())
                .with_exif(config.exif.clone())
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_num_threads(config.num_threads),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let lut = srgb_lut();
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_rgb::<_, _, 3>(input, width, height, &ctx, &mut scratch, |v| {
        lut[v as usize]
    });
    let mut ctx = ctx;
    apply_yellow_opsin(&mut ctx, &linear, distance);
    let cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    encode_with_context(&linear, &cfg, &ctx, &mut scratch)
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
                .with_progressive(config.progressive)
                .with_patches(config.patches)
                .with_icc_profile(config.icc_profile.clone())
                .with_exif(config.exif.clone())
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_num_threads(config.num_threads),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let lut = srgb_lut();
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_rgb::<_, _, 4>(input, width, height, &ctx, &mut scratch, |v| {
        lut[v as usize]
    });
    let mut ctx = ctx;
    apply_yellow_opsin(&mut ctx, &linear, distance);
    let alpha_plane = input.as_chunks::<4>().0.iter().map(|px| px[3]).collect();
    let cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_alpha(AlphaPlane::from_u8(alpha_plane))
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    encode_with_context(&linear, &cfg, &ctx, &mut scratch)
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
                .with_patches(config.patches)
                .with_icc_profile(config.icc_profile.clone())
                .with_exif(config.exif.clone())
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_num_threads(config.num_threads),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let srgb_lut = srgb_lut();
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_gray(luma, width, height, &ctx, &mut scratch, |v| {
        srgb_lut[v as usize]
    });
    let mut cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_grayscale(true)
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    if let Some(a) = alpha {
        cfg = cfg.with_alpha(AlphaPlane::from_u8(a));
    }
    encode_with_context(&linear, &cfg, &ctx, &mut scratch)
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
                .with_patches(config.patches)
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_exif(config.exif.clone())
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_num_threads(config.num_threads),
        );
    }

    let distance = config.distance.max(MIN_DISTANCE);
    let lut = &lut_high_bit(bps.bits() as u8).table;
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_gray(luma, width, height, &ctx, &mut scratch, |v| lut[v as usize]);

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
        .with_progressive_from(config)
        .with_grayscale(true)
        .with_bits_per_sample(bps)
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    if let Some(ap) = alpha_plane {
        cfg = cfg.with_alpha(ap);
    }
    encode_with_context(&linear, &cfg, &ctx, &mut scratch)
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
                .with_patches(config.patches)
                .with_icc_profile(config.icc_profile.clone())
                .with_exif(config.exif.clone())
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_num_threads(config.num_threads),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let lut = &lut_high_bit(bps.bits() as u8).table;
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();

    // For 16-bit, (1 << 16) - 1 overflows u16's shift; compute in u32 and cap.
    let bp_max: u16 = if bps.bits() >= 16 {
        u16::MAX
    } else {
        ((1u32 << bps.bits()) - 1) as u16
    };

    if has_alpha {
        let linear = linearize_rgb::<_, _, 4>(input, width, height, &ctx, &mut scratch, |v| {
            lut[v as usize]
        });
        let mut ctx = ctx;
        apply_yellow_opsin(&mut ctx, &linear, distance);
        let alpha_plane = input
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| px[3].min(bp_max))
            .collect();
        let cfg = EncodeConfigImpl::with_distance(distance)
            .with_progressive_from(config)
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
            .with_exif(config.exif.clone())
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_with_context(&linear, &cfg, &ctx, &mut scratch)
    } else {
        let linear = linearize_rgb::<_, _, 3>(input, width, height, &ctx, &mut scratch, |v| {
            lut[v as usize]
        });
        let mut ctx = ctx;
        apply_yellow_opsin(&mut ctx, &linear, distance);
        let cfg = EncodeConfigImpl::with_distance(distance)
            .with_progressive_from(config)
            .with_bits_per_sample(bps)
            .with_icc_profile(config.icc_profile.clone())
            .with_exif(config.exif.clone())
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_with_context(&linear, &cfg, &ctx, &mut scratch)
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
    let mut metadata_scratch = Box::<CoderScratch>::default();
    write_image_metadata(
        config.tone_mapping(),
        &config.color_encoding,
        alpha.as_ref(),
        config.icc_profile.as_deref(),
        BitsPerSample::F32,
        true,
        &XybMatrix::SPEC,
        false,
        config.orientation,
        &mut metadata_scratch,
        &mut w,
    );
    encode_frame_lossless_float(&image3s, alpha.as_ref(), config.num_threads, &mut w);
    let codestream = w.into_bytes();
    let alpha_bits_md = if has_alpha { 32 } else { 0 };
    finalize_container(
        codestream,
        config.exif.as_deref(),
        config.xmp.as_deref(),
        config.brotli_compression.as_deref(),
        needs_level_10(32, true, alpha_bits_md),
    )
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
    // Float input skips the red-dominance classifier (integer-domain sampling).
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();

    if has_alpha {
        let linear =
            linearize_rgb::<_, _, 4>(input, width, height, &ctx, &mut scratch, srgb_to_linear_f32);
        let mut ctx = ctx;
        apply_yellow_opsin(&mut ctx, &linear, distance);
        let alpha_plane = input
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| (px[3].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
            .collect();
        let cfg = EncodeConfigImpl::with_distance(distance)
            .with_progressive_from(config)
            .with_alpha(AlphaPlane::from_u16_16bit(alpha_plane))
            .with_bits_per_sample(bps)
            .with_icc_profile(config.icc_profile.clone())
            .with_exif(config.exif.clone())
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_with_context(&linear, &cfg, &ctx, &mut scratch)
    } else {
        let linear =
            linearize_rgb::<_, _, 3>(input, width, height, &ctx, &mut scratch, srgb_to_linear_f32);
        let mut ctx = ctx;
        apply_yellow_opsin(&mut ctx, &linear, distance);
        let cfg = EncodeConfigImpl::with_distance(distance)
            .with_progressive_from(config)
            .with_bits_per_sample(bps)
            .with_icc_profile(config.icc_profile.clone())
            .with_exif(config.exif.clone())
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_with_context(&linear, &cfg, &ctx, &mut scratch)
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
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_gray(luma, width, height, &ctx, &mut scratch, srgb_to_linear_f32);
    let cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_grayscale(true)
        .with_bits_per_sample(bps)
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    encode_with_context(&linear, &cfg, &ctx, &mut scratch)
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

/// Resolve the per-pass VarDCT coefficient-shift schedule for lossy encoding.
/// Returns `[shift_pass0, .., shift_passN-1]` with the last element 0. A single
/// pass is `[0]`. The decoder reconstructs each AC coefficient by summing
/// `(sent_p << shift_p)` over passes, so the schedule must end in 0.
///
/// An explicit `shifts` is used when valid (non-empty, ends in 0, all entries
/// 0..=3, non-increasing); otherwise it is ignored and we fall back to a count:
/// `passes` (clamped 1..=4) or the `progressive` bool, yielding the
/// coarse-to-fine schedule `[n-1, .., 1, 0]`.
fn progressive_schedule(
    progressive: bool,
    passes: Option<u32>,
    shifts: Option<&[u32]>,
) -> Vec<u32> {
    if let Some(s) = shifts {
        let valid = !s.is_empty()
            && s.len() <= 11
            && *s.last().unwrap() == 0
            && s.iter().all(|&v| v <= 3)
            && s.windows(2).all(|w| w[0] >= w[1]);
        if valid {
            return s.to_vec();
        }
    }
    let n = passes
        .map(|p| p.clamp(1, 4))
        .unwrap_or(if progressive { 2 } else { 1 }) as usize;
    (0..n).rev().map(|p| p as u32).collect()
}

fn encode_with_context(
    input: &Image3F,
    config: &EncodeConfigImpl,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
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
    let coeff_shifts = progressive_schedule(
        config.progressive,
        config.progressive_passes,
        config.progressive_shifts.as_deref(),
    );
    write_image_metadata(
        config.tone_mapping(),
        &config.color_encoding,
        config.alpha.as_ref(),
        config.icc_profile.as_deref(),
        config.bits_per_sample,
        config.lossless,
        &ctx.xyb,
        config.grayscale,
        config.orientation,
        scratch,
        &mut w,
    );
    encode_frame(
        ctx,
        scratch,
        distance,
        input,
        config.grayscale,
        config.alpha.as_ref(),
        &coeff_shifts,
        config.patches,
        &mut w,
    )?;
    let codestream = w.into_bytes();
    let alpha_bits = config.alpha.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    finalize_container(
        codestream,
        config.exif.as_deref(),
        config.xmp.as_deref(),
        config.brotli_compression.as_deref(),
        needs_level_10(config.bits_per_sample.bits(), config.lossless, alpha_bits),
    )
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
    let mut metadata_scratch = Box::<CoderScratch>::default();
    write_image_metadata(
        config.tone_mapping(),
        &config.color_encoding,
        alpha_plane.as_ref(),
        config.icc_profile.as_deref(),
        config.bits_per_sample,
        config.lossless,
        &XybMatrix::SPEC,
        config.grayscale,
        config.orientation,
        &mut metadata_scratch,
        &mut w,
    );
    let alpha_bits = alpha_plane.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    let eff_bits = (max_bp as u32).max(alpha_bits);
    let num_color = if config.grayscale { 1 } else { 3 };
    encode_frame_lossless(
        &image3s,
        alpha_plane.as_ref(),
        eff_bits,
        config.progressive,
        config.patches,
        num_color,
        config.speed,
        config.decoding_speed,
        config.num_threads,
        &mut w,
    );
    let codestream = w.into_bytes();
    let alpha_bits = alpha_plane.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    finalize_container(
        codestream,
        config.exif.as_deref(),
        config.xmp.as_deref(),
        config.brotli_compression.as_deref(),
        needs_level_10(max_bp as u32, true, alpha_bits),
    )
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
pub(crate) fn needs_level_10(bits: u32, lossless: bool, alpha_bits: u32) -> bool {
    (lossless && bits >= 16) || alpha_bits >= 16
}

fn push_container_box(out: &mut Vec<u8>, kind: &[u8; 4], contents: &[u8]) {
    let size = 8u64 + contents.len() as u64;
    if size <= u32::MAX as u64 {
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(kind);
    } else {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(&(size + 8).to_be_bytes());
    }
    out.extend_from_slice(contents);
}

fn push_metadata_box(
    out: &mut Vec<u8>,
    kind: &[u8; 4],
    contents: &[u8],
    compressor: Option<&dyn BrotliCompression>,
) -> Result<(), EncodeError> {
    if let Some(compressor) = compressor {
        let compressed = compressor.compress(contents)?;
        let mut brob = Vec::with_capacity(4 + compressed.len());
        brob.extend_from_slice(kind);
        brob.extend_from_slice(&compressed);
        push_container_box(out, b"brob", &brob);
    } else {
        push_container_box(out, kind, contents);
    }
    Ok(())
}

/// Wrap a bare codestream in a minimal JXL (ISO BMFF) container that declares
/// `level` via a `jxll` box. Box order: signature, ftyp, jxll, jxlc, then
/// optional metadata boxes.
///
/// When `exif` is `Some`, an `Exif` box is appended after the codestream. Its
/// payload is a 4-byte big-endian TIFF-header offset (0) followed by the raw
/// EXIF/TIFF byte stream (the bytes that begin with the "II"/"MM" byte-order
/// mark — *not* the JPEG "Exif\0\0" APP1 prefix). `xmp`, when present, is
/// written as the payload of an `xml ` box. If `compressor` is supplied, EXIF
/// and XMP are instead wrapped in Brotli-compressed `brob` boxes.
pub(crate) fn wrap_jxl_container(
    codestream: Vec<u8>,
    level: u8,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    compressor: Option<&dyn BrotliCompression>,
) -> Result<Vec<u8>, EncodeError> {
    let exif_extra = exif.map(|e| e.len() + 12).unwrap_or(0);
    let xmp_extra = xmp.map(|x| x.len() + 8).unwrap_or(0);
    let mut out = Vec::with_capacity(codestream.len() + 41 + exif_extra + xmp_extra);
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
    push_container_box(&mut out, b"jxlc", &codestream);
    // Exif metadata box (after the codestream; libjxl convention).
    if let Some(e) = exif {
        let mut contents = Vec::with_capacity(4 + e.len());
        contents.extend_from_slice(&0u32.to_be_bytes()); // TIFF-header offset = 0
        contents.extend_from_slice(e);
        push_metadata_box(&mut out, b"Exif", &contents, compressor)?;
    }
    if let Some(x) = xmp {
        push_metadata_box(&mut out, b"xml ", x, compressor)?;
    }
    Ok(out)
}

/// Decide final output form: wrap in a container when level 10 is required or
/// when an EXIF/XMP box must be carried (a bare codestream cannot hold it).
fn finalize_container(
    codestream: Vec<u8>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    compressor: Option<&dyn BrotliCompression>,
    need_l10: bool,
) -> Result<Vec<u8>, EncodeError> {
    if need_l10 || exif.is_some() || xmp.is_some() {
        wrap_jxl_container(
            codestream,
            if need_l10 { 10 } else { 5 },
            exif,
            xmp,
            compressor,
        )
    } else {
        Ok(codestream)
    }
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
    tm: ToneMappingParams,
    color_encoding: &ColorEncoding,
    alpha: Option<&AlphaPlane>,
    icc_profile: Option<&[u8]>,
    bps: BitsPerSample,
    lossless: bool,
    xyb: &XybMatrix,
    grayscale: bool,
    orientation: Orientation,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    // tone_mapping (HDR luminance) is gated by extra_fields; a non-identity
    // orientation also lives in the extra_fields block, so either forces it on.
    let extra_fields = !tm.is_default() || orientation != Orientation::Normal;
    w.write(1, if extra_fields { 1 } else { 0 }); // extra_fields
    if extra_fields {
        w.write(3, orientation.to_u3()); // orientation: 1 + u(3)
        w.write(1, 0); // have_intrinsic_size = false
        w.write(1, 0); // have_preview = false
        w.write(1, 0); // have_animation = false
    }
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
    // tone_mapping bundle (gated by extra_fields). ToneMapping fields:
    // intensity_target (F16), min_nits (F16), relative_to_max_display (bool),
    // linear_below (F16). We signal intensity_target and leave the rest default.
    if extra_fields {
        // ToneMapping bundle, all fields explicit (all_default = false).
        w.write(1, 0);
        let it = tm.intensity_target.unwrap_or(255.0);
        w.write(16, crate::util::f32_to_f16_bits(it) as u64); // intensity_target
        w.write(16, crate::util::f32_to_f16_bits(tm.min_nits) as u64); // min_nits
        w.write(1, if tm.relative_to_max_display { 1 } else { 0 }); // relative_to_max_display
        w.write(16, crate::util::f32_to_f16_bits(tm.linear_below) as u64); // linear_below
    }
    w.write(2, 0); // extensions: U64 selector = 0 (no extensions)
    // End of ImageMetadata. Now CustomTransformData (part of FileHeader, but kept here for
    // backward-compatible bit alignment with the no-ICC path).
    if lossless || xyb.is_decoder_default() {
        w.write(1, 1); // CustomTransformData.all_default = 1
    } else {
        // A non-spec forward opsin matrix needs its matching inverse in the
        // codestream, so the bundle is explicit. Layout: all_default,
        // [xyb_encoded] OpsinInverseMatrix, custom_weights_mask.
        w.write(1, 0); // CustomTransformData.all_default = 0
        w.write(1, 0); // OpsinInverseMatrix.all_default = 0
        for &v in &xyb.inv {
            w.write(16, crate::util::f32_to_f16_bits(v) as u64);
        }
        for _ in 0..3 {
            // Opsin biases (defaults; the forward bias is unchanged).
            w.write(16, crate::util::f32_to_f16_bits(-0.003_793_073_4) as u64);
        }
        // Per-channel quant biases + numerator. X/Y keep the libjxl defaults;
        // B's ±1 reconstruction is raised to a full step: the explicit bundle
        // is only written for the chroma-gated content classes, where the
        // dying ±1 B residuals are exactly the desaturating ones. Measured
        // free (+0.03..0.05 SS2 at byte-identical streams) on every class
        // probed — fractal, blue-rotated victim, smooth-yellow photos;
        // overshoot past 1.0 was noise. The RDOQ dequant model keeps the
        // shared Y bias (its mismatch grows 0.05→0.07, same scale as the
        // pre-existing X mismatch).
        for v in [1.0 - 0.054_650_075, 1.0 - 0.070_054_5, 1.0, 0.145] {
            w.write(16, crate::util::f32_to_f16_bits(v) as u64);
        }
        w.write(3, 0); // custom_weights_mask = 0 (default upsampling kernels)
    }
    // ICC stream goes AFTER FileHeader, before the zero-pad to byte boundary.
    if let Some(icc) = icc_profile {
        crate::icc_codec::write_icc_stream(icc, &mut scratch.huffman_pool, w);
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
        assert!(close(distance_from_quality(100.0), 0.05));
        assert!(close(distance_from_quality(99.0), 0.1));
        assert!(close(distance_from_quality(90.0), 1.0));
        assert!(close(distance_from_quality(30.0), 6.4));
        // Below 30, the formula is quadratic-ish; q=25 must come in just above 6.4.
        let d25 = distance_from_quality(25.0);
        assert!(d25 > 6.4 && d25 < 7.0, "q=25 -> {d25}");
        // Far below 30, it climbs and clamps at 25.
        assert!(close(distance_from_quality(0.0), 25.0));
        assert!(close(distance_from_quality(-50.0), 25.0));
        // Above 100, clamped to the practical VarDCT endpoint.
        assert!(close(distance_from_quality(110.0), 0.05));
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
    fn default_speed_is_fast() {
        assert_eq!(Speed::default(), Speed::Fast);
        assert_eq!(EncodeConfig::default().speed, Speed::Fast);
    }

    #[test]
    fn srgb_linearization_uses_requested_threads() {
        use std::collections::HashSet;
        use std::sync::{Barrier, Mutex};

        let config = EncodeConfig::default().with_num_threads(4);
        let ctx = lossy_context(&config, 1.0, XybMatrix::SPEC, 1);
        let mut scratch = Box::<CoderScratch>::default();
        let mut image = Image3F::new(8, 4);
        let barrier = Barrier::new(4);
        let threads = Mutex::new(HashSet::new());

        for_each_linear_band(&mut image, &ctx, &mut scratch, |_start, _band| {
            threads.lock().unwrap().insert(std::thread::current().id());
            barrier.wait();
        });

        assert_eq!(threads.into_inner().unwrap().len(), 4);
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

    fn box_payload<'a>(container: &'a [u8], wanted: &[u8; 4]) -> Option<&'a [u8]> {
        let mut pos = 0usize;
        while pos + 8 <= container.len() {
            let small = u32::from_be_bytes(container[pos..pos + 4].try_into().unwrap()) as usize;
            let kind: &[u8; 4] = container[pos + 4..pos + 8].try_into().unwrap();
            let (header, size) = if small == 1 {
                if pos + 16 > container.len() {
                    return None;
                }
                (
                    16,
                    u64::from_be_bytes(container[pos + 8..pos + 16].try_into().unwrap()) as usize,
                )
            } else {
                (8, small)
            };
            if size < header || pos.checked_add(size)? > container.len() {
                return None;
            }
            if kind == wanted {
                return Some(&container[pos + header..pos + size]);
            }
            pos += size;
        }
        None
    }

    #[test]
    fn xmp_is_written_to_an_xml_box() {
        let xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>";
        let encoded = encode_image(&rgb8(), W, H, &lossy().with_xmp(xmp.to_vec())).unwrap();
        assert_eq!(box_payload(&encoded, b"xml "), Some(xmp.as_slice()));
    }

    #[test]
    fn custom_brotli_writes_brob_metadata() {
        struct MarkerCompressor;
        impl BrotliCompression for MarkerCompressor {
            fn compress(&self, data: &[u8]) -> Result<Vec<u8>, EncodeError> {
                assert_eq!(data, b"<x:xmpmeta/>");
                Ok(vec![0xAA, 0xBB])
            }
        }

        let encoded = encode_image(
            &rgb8(),
            W,
            H,
            &lossy()
                .with_xmp(b"<x:xmpmeta/>".to_vec())
                .with_brotli_compression(Box::new(MarkerCompressor)),
        )
        .unwrap();
        assert_eq!(box_payload(&encoded, b"brob"), Some(&b"xml \xAA\xBB"[..]));
        assert!(box_payload(&encoded, b"xml ").is_none());
    }

    #[test]
    fn custom_brotli_error_is_returned() {
        struct FailingCompressor;
        impl BrotliCompression for FailingCompressor {
            fn compress(&self, _data: &[u8]) -> Result<Vec<u8>, EncodeError> {
                Err(EncodeError::Brotli("backend failed".into()))
            }
        }

        assert_eq!(
            encode_image(
                &rgb8(),
                W,
                H,
                &lossy()
                    .with_xmp(b"<x:xmpmeta/>".to_vec())
                    .with_brotli_compression(Box::new(FailingCompressor)),
            ),
            Err(EncodeError::Brotli("backend failed".into()))
        );
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
    fn lossy_patches_are_rate_safe() {
        let plain = encode_image(&rgb8(), W, H, &lossy()).unwrap();
        let checked = encode_image(&rgb8(), W, H, &lossy().with_patches(true)).unwrap();
        assert!(checked.len() <= plain.len());
    }

    #[test]
    fn lossy_patches_reduce_repeated_regions() {
        const PW: usize = 256;
        const PH: usize = 256;
        let mut state = 0x89ab_cdefu32;
        let mut pixels = vec![0u8; PW * PH * 3];
        for sample in &mut pixels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as u8;
        }
        let tile: Vec<u8> = (0..16)
            .flat_map(|y| pixels[(y * PW * 3)..(y * PW * 3 + 48)].iter().copied())
            .collect();
        for ty in 0..16 {
            for tx in 0..16 {
                if (tx + ty) % 2 != 0 {
                    continue;
                }
                for y in 0..16 {
                    let dst = ((ty * 16 + y) * PW + tx * 16) * 3;
                    pixels[dst..dst + 48].copy_from_slice(&tile[y * 48..(y + 1) * 48]);
                }
            }
        }
        let plain = encode_image(&pixels, PW, PH, &lossy().with_patches(false)).unwrap();
        let patched = encode_image(&pixels, PW, PH, &lossy().with_patches(true)).unwrap();
        assert!(
            patched.len() < plain.len(),
            "lossy patches should win on repeated complex regions: {} >= {}",
            patched.len(),
            plain.len()
        );
    }

    #[test]
    fn rgb8_slow_low_quality_lossy() {
        const SIDE: usize = 64;
        let pixels: Vec<u8> = (0..SIDE * SIDE * 3).map(|i| (i % 251) as u8).collect();
        ok(encode_image(
            &pixels,
            SIDE,
            SIDE,
            &EncodeConfig::default()
                .with_distance(3.0)
                .with_speed(Speed::Slow),
        ));
    }

    #[test]
    fn rgb8_slow_low_quality_rect_merge_lossy() {
        let config = EncodeConfig::default()
            .with_distance(2.5)
            .with_speed(Speed::Slow);
        for (width, height) in [(32, 64), (64, 32)] {
            let pixels = vec![128u8; width * height * 3];
            ok(encode_image(&pixels, width, height, &config));
        }
    }

    #[test]
    fn lossy_output_is_independent_of_thread_count() {
        const WIDTH: usize = 257;
        const HEIGHT: usize = 65;
        let input: Vec<u8> = (0..WIDTH * HEIGHT * 3)
            .map(|i| i.wrapping_mul(37).wrapping_add((i / 7).wrapping_mul(13)) as u8)
            .collect();
        let config = lossy().with_speed(Speed::Slow);

        let single = encode_image(&input, WIDTH, HEIGHT, &config.clone().with_num_threads(1))
            .expect("single-threaded lossy encode failed");
        let threaded = encode_image(&input, WIDTH, HEIGHT, &config.with_num_threads(4))
            .expect("multi-threaded lossy encode failed");

        assert_eq!(single, threaded);
    }

    #[test]
    fn rgba_lossy_patches_reduce_repeated_regions() {
        const PW: usize = 256;
        const PH: usize = 256;
        let mut pixels = vec![0u8; PW * PH * 4];
        for (i, px) in pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (x, y) = (i % PW, i / PW);
            let (tx, ty) = (x % 16, y % 16);
            // Glyph-like repeated tiles on 3/4 of the grid, noise elsewhere.
            if (x / 16 + y / 16) % 4 != 0 {
                // Few distinct colors, so the tile is palette-friendly and the
                // routing sends it to the modular atlas.
                px[0] = ((tx / 4) * 40 + (ty / 4) * 17) as u8;
                px[1] = ((tx / 4) * 25 + (ty / 4) * 31) as u8;
                px[2] = ((tx / 4) * 11 + (ty / 4) * 45) as u8;
            } else {
                let s = (i as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                px[0] = (s >> 24) as u8;
                px[1] = (s >> 16) as u8;
                px[2] = (s >> 8) as u8;
            }
            px[3] = 255;
        }
        let cfg = |patches: bool| {
            EncodeConfig::default()
                .with_lossless(false)
                .with_distance(1.0)
                .with_patches(patches)
        };
        let plain = encode_image_with_alpha(&pixels, PW, PH, &cfg(false)).unwrap();
        let patched = encode_image_with_alpha(&pixels, PW, PH, &cfg(true)).unwrap();
        assert!(
            patched.len() < plain.len(),
            "alpha image patches should win: {} >= {}",
            patched.len(),
            plain.len()
        );
    }

    #[test]
    fn rgb8_lossless() {
        ok(encode_image(&rgb8(), W, H, &lossless()));
    }

    #[test]
    fn rgb8_lossless_patches_reduce_repeated_tiles() {
        const PW: usize = 96;
        const PH: usize = 64;
        let mut pixels = vec![0u8; PW * PH * 3];
        for y in 0..PH {
            for x in 0..PW {
                let tx = x % 16;
                let ty = y % 16;
                let p = (y * PW + x) * 3;
                pixels[p] = (tx * 13 + ty * 3) as u8;
                pixels[p + 1] = (tx * 5 + ty * 11) as u8;
                pixels[p + 2] = (tx * 7 + ty * 9) as u8;
            }
        }
        let plain = encode_image(&pixels, PW, PH, &lossless().with_patches(false)).unwrap();
        let patched = encode_image(&pixels, PW, PH, &lossless().with_patches(true)).unwrap();
        assert!(
            patched.len() < plain.len(),
            "patch dictionary should reduce a repeated-tile image: {} >= {}",
            patched.len(),
            plain.len()
        );
    }

    #[test]
    fn rgb8_lossless_patches_fall_back_when_rate_is_worse() {
        const PW: usize = 64;
        const PH: usize = 64;
        let mut state = 0x1234_5678u32;
        let mut pixels = vec![0u8; PW * PH * 3];
        for sample in &mut pixels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as u8;
        }
        for y in 0..16 {
            for x in 0..48 {
                let pixel = (y * PW + x) * 3;
                for sample in &mut pixels[pixel..pixel + 3] {
                    *sample = 0;
                }
            }
        }

        let plain = encode_image(&pixels, PW, PH, &lossless()).unwrap();
        let checked = encode_image(&pixels, PW, PH, &lossless().with_patches(true)).unwrap();
        assert_eq!(checked, plain, "a losing patch candidate must be discarded");
    }

    #[test]
    fn rgb8_lossless_slow_adaptive() {
        ok(encode_image(
            &rgb8(),
            W,
            H,
            &lossless().with_speed(Speed::Slow),
        ));
    }

    #[test]
    fn fastest_speed_encodes_dct8_only() {
        // A smooth gradient makes Fast/Slow merge into large transforms, so a
        // Fastest stream (all 8x8) must differ; both must be valid encodes.
        const WIDTH: usize = 128;
        const HEIGHT: usize = 128;
        let input: Vec<u8> = (0..WIDTH * HEIGHT * 3)
            .map(|i| (((i / 3) % WIDTH) / 2 + ((i / 3) / WIDTH) / 2) as u8)
            .collect();
        let config = lossy().with_distance(2.0);
        let fastest = encode_image(
            &input,
            WIDTH,
            HEIGHT,
            &config.clone().with_speed(Speed::Fastest),
        )
        .expect("Fastest encode failed");
        let fast = encode_image(
            &input,
            WIDTH,
            HEIGHT,
            &config.clone().with_speed(Speed::Fast),
        )
        .expect("Fast encode failed");
        assert!(!fastest.is_empty());
        assert_ne!(
            fastest, fast,
            "Fastest should skip the transform search Fast performs"
        );
    }

    #[test]
    fn lossless_output_is_independent_of_thread_count() {
        const WIDTH: usize = 257;
        const HEIGHT: usize = 257;
        let input: Vec<u8> = (0..WIDTH * HEIGHT * 3)
            .map(|i| i.wrapping_mul(37).wrapping_add((i / 7).wrapping_mul(13)) as u8)
            .collect();

        for speed in [Speed::Fastest, Speed::Fast, Speed::Slow] {
            let single = encode_image(
                &input,
                WIDTH,
                HEIGHT,
                &lossless().with_speed(speed).with_num_threads(1),
            )
            .expect("single-threaded lossless encode failed");
            let threaded = encode_image(
                &input,
                WIDTH,
                HEIGHT,
                &lossless().with_speed(speed).with_num_threads(8),
            )
            .expect("multi-threaded lossless encode failed");

            assert_eq!(single, threaded, "output changed for {speed:?}");
        }
    }

    #[test]
    fn decoding_speed_tiers_encode_and_restrict_tools() {
        const WIDTH: usize = 96;
        const HEIGHT: usize = 96;
        // Smooth per-channel gradients with edges and mild texture; enough
        // distinct colors to keep the palette path out, so WP and the learned
        // tree both normally engage.
        let input: Vec<u8> = (0..WIDTH * HEIGHT * 3)
            .map(|i| {
                let (px, c) = (i / 3, i % 3);
                let (x, y) = (px % WIDTH, px / WIDTH);
                let (x, y, c) = (x as i64, y as i64, c as i64);
                let base = if (x / 24 + y / 24) % 2 == 0 {
                    x + 2 * y + 31 * c
                } else {
                    255 - (2 * x + y + 17 * c)
                };
                (base as u8).wrapping_add(((x * y) % 5) as u8)
            })
            .collect();

        for speed in [Speed::Fast, Speed::Slow] {
            let encode = |ds: DecodingSpeed| {
                encode_image(
                    &input,
                    WIDTH,
                    HEIGHT,
                    &lossless()
                        .with_speed(speed)
                        .with_decoding_speed(ds)
                        .with_num_threads(1),
                )
                .expect("lossless encode failed")
            };
            let slow_tier = encode(DecodingSpeed::Slow);
            let fast_tier = encode(DecodingSpeed::Fast);
            let fastest_tier = encode(DecodingSpeed::Fastest);

            // Unset == Slow (the default: all tools).
            let unset = encode_image(
                &input,
                WIDTH,
                HEIGHT,
                &lossless().with_speed(speed).with_num_threads(1),
            )
            .expect("lossless encode failed");
            assert_eq!(slow_tier, unset, "{speed:?}: Slow must equal unset");

            // Dropping WP must change the stream on this content.
            assert_ne!(slow_tier, fast_tier, "{speed:?}: Fast tier inert");
            // Fastest differs from Slow too (no WP, no trees).
            assert_ne!(slow_tier, fastest_tier, "{speed:?}: Fastest tier inert");
        }
    }

    #[test]
    fn local_palette_rgb_and_rgba_are_thread_deterministic() {
        const WIDTH: usize = 257;
        const HEIGHT: usize = 33;
        let rgb: Vec<u8> = (0..WIDTH * HEIGHT)
            .flat_map(|i| {
                let x = i % WIDTH;
                let y = i / WIDTH;
                let c = ((x / 8 + y / 8) & 15) as u8;
                [c * 13, c * 7, c * 3]
            })
            .collect();
        let rgba: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255 - pixel[0]])
            .collect();
        let one = lossless().with_speed(Speed::Slow).with_num_threads(1);
        let many = lossless().with_speed(Speed::Slow).with_num_threads(4);

        assert_eq!(
            encode_image(&rgb, WIDTH, HEIGHT, &one).unwrap(),
            encode_image(&rgb, WIDTH, HEIGHT, &many).unwrap()
        );
        assert_eq!(
            encode_image_with_alpha(&rgba, WIDTH, HEIGHT, &one).unwrap(),
            encode_image_with_alpha(&rgba, WIDTH, HEIGHT, &many).unwrap()
        );
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
