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
use crate::coding::CodingTransform;
use crate::color::{
    ColorTransform, FloatTransferDecoder, RgbBlockTransform, TransferDecoder, TransferLut,
};
use crate::color_encoding::write_color_encoding_with_icc;
use crate::encoding_context::EncodingContext;
use crate::frame::encode_frame;
use crate::gain_map::{EncodedGainMap, GainMap, encode_gain_map};
use crate::image::{Image3F, Image3Si};
use crate::jpeg::BrotliCompression;
use crate::lossless::{encode_frame_lossless, encode_frame_lossless_float, forward_ycocg};
use crate::orientation::Orientation;
use crate::xyb::XybMatrix;
use crate::{ColorEncoding, EncodeError, Grain};
use std::num::NonZero;
use std::sync::Arc;
use std::thread::available_parallelism;

pub(crate) fn checked_buffer_size<T>(
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

/// Practical lower bound for lossy encoding; lossless bypasses this clamp.
pub(crate) const MIN_DISTANCE: f32 = 0.05;

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

    /// True if every sample holds the same value.
    pub(crate) fn is_constant(&self) -> bool {
        fn same<T: PartialEq>(v: &[T]) -> bool {
            v.split_first()
                .is_none_or(|(first, rest)| rest.iter().all(|x| x == first))
        }
        match self {
            Self::U8(v) => same(v),
            Self::U16 { data, .. } => same(data),
            Self::F32(v) => same(v),
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

/// Encoder speed/density tradeoff, ordered from fastest to densest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum Speed {
    /// No transform search at all: every block is coded as a plain 8×8 DCT
    /// with the cheapest entropy coding. Skips everything `Fastest` skips.
    UltraFast,
    /// 16×16 merges on quiet blocks only, prefix-clustered contexts and the
    /// DC predictor choice; none of `Fast`'s quantizer refinement, learned DC
    /// tree, or entropy-code refinement. Lossless is coded as at `Fast`.
    Fastest,
    /// Square transforms only; lossless uses no learned context trees.
    #[default]
    Fast,
    /// `Fast`'s square transforms and coefficient-model layout search with
    /// `Slow`'s custom coefficient orders, DC coding tools, chroma-from-luma
    /// search, color tools and ink layer; lossless learns its context tree
    /// from a smaller sample and skips the predictor-preset ranking.
    Medium,
    /// The full transform search of `Slow` with its reconstruction rerank and
    /// exact layout search over pieces up to 16x16, learned rate prices, the
    /// learned DC tree, entropy-code refinement and the EPF strength fit; no
    /// content layers (patches, splines, vector layers, region frames) by
    /// default. Lossless learns its context tree from `Slow`'s full sample
    /// but keeps `Medium`'s single group layout, so it decodes as fast.
    MediumSlow,
    /// Every coding tool with the standard search budget.
    Slow,
    /// All `Slow` tools with stronger final ANS entropy search for lossy and
    /// lossless encoding, additional lossless LZ77 candidates, and wider
    /// large-transform quantization search at distance >= 3. Without the
    /// `extraslow` cargo feature it encodes exactly as `Slow`.
    ExtraSlow,
}

/// The perceptual metric whose preferences settle the **lossy** encoder's
/// decisions where metrics disagree. Ignored for lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Arbiter {
    /// SSIMULACRA2.
    #[default]
    Ssimulacra2,
    /// ColorVideoVDP
    Cvvdp,
}

/// Decode-speed/density tradeoff for **lossless** encoding. Ignored for lossy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodingSpeed {
    /// No Weighted Predictor and no meta-adaptive context trees
    Fastest,
    /// No Weighted Predictor.
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
    /// Positive, finite lossy distance, clamped to a minimum of 0.05.
    /// Ignored when `lossless` is true.
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
    /// Smooth color samples underneath fully transparent pixels before lossy
    /// encoding, using libjxl's neighbor fill. Alpha and partially transparent
    /// source samples are unchanged by this preprocessing. Defaults to true.
    /// Ignored for lossless encoding and CMYK images.
    pub fill_invisible: bool,
    /// Distance of the alpha channel in lossy frames, in the units of
    /// `distance`. `None` (the default) codes alpha at the frame distance;
    /// `Some(0.0)` keeps it lossless. Lossy alpha is a quantized Squeeze
    /// pyramid; the encoder keeps lossless alpha wherever that is no larger,
    /// so solid and hard-edged alpha stays exact. Values are clamped to
    /// `0.0..=25.0`; NaN means lossless. Ignored when `lossless` is true.
    pub alpha_distance: Option<f32>,
    /// If true (and lossless), use the progressive Squeeze transform: a low-res
    /// preview that refines to bit-exact. Works at any size (single- and
    /// multi-group) and with alpha.
    ///
    /// For lossy (`!lossless`) encoding, `true` selects 2-pass progressive
    /// VarDCT (equivalent to `progressive_passes = Some(2)`).
    pub progressive: bool,
    /// Encode repeated content through the JPEG XL patch dictionary: glyphs
    /// and other small marks on flat backgrounds at any position, and
    /// repeated grid-aligned tiles. Supported for both lossless Modular and
    /// lossy VarDCT encoding. Lossless keeps patches only when they reduce
    /// the final size; lossy takes a plan on its coverage, since the patched
    /// frame is also the sharper one, and photographic content finds none.
    pub patches: bool,
    /// Experimental: detect thin curvilinear structures (wires, veins, ink
    /// lines) and code them as JPEG XL splines next to the lossy VarDCT image.
    /// Every spline passes a rate-distortion test against the VarDCT cost of
    /// the same pixels. Slow speed and the default decoding speed only; costs
    /// encode time and some decode time.
    /// Requires the `splines` Cargo feature. Defaults to false.
    #[cfg(feature = "splines")]
    pub splines: bool,
    /// Experimental: detect isolated bright or dark spots and code them as
    /// signed Gaussian dots drawn from a small template atlas (additive
    /// patches), next to the lossy VarDCT image. Dots, templates and the whole
    /// set pass rate-distortion tests; unprofitable sets fall back to VarDCT.
    /// Slow speed and the default decoding speed only; costs encode time.
    /// Requires the `splines` Cargo feature. Defaults to false.
    #[cfg(feature = "splines")]
    pub dots: bool,
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
    /// Metric the lossy encoder tunes for (see [`Arbiter`]). Default
    /// [`Arbiter::Ssimulacra2`].
    pub arbiter: Arbiter,
    /// Decode-speed/density tradeoff for lossless encoding (see
    /// [`DecodingSpeed`]). Ignored for lossy.
    pub decoding_speed: DecodingSpeed,
    /// Near-lossless bound for `lossless` encodes: the maximum absolute error
    /// per channel, in input code values. `0` (default) keeps every sample
    /// exact. Alpha always stays exact. Ignored for lossy.
    pub near_lossless: u16,
    /// Code integer-upscaled content at its cell grid and let the decoder
    /// scale it back with a nearest-neighbor kernel: a `lossless` image whose
    /// 2x2, 4x4 or 8x8 cells are all one color (pixel art, a 2x asset) is
    /// coded at a half, quarter or eighth of its size, and such a region
    /// inside an image becomes a layer blended over the rest, which then gains
    /// an opaque alpha channel. Exact. The search costs about 30 ms per
    /// megapixel. Defaults to false. Ignored for lossy and progressive encodes.
    pub upsampling: bool,
    /// Lossy encoding arm selection (see [`LossyModular`]). Default `Off`.
    pub lossy_modular: LossyModular,
    /// Experimental: lift line/text colors into one palette frame with
    /// coverage alpha, blended over a VarDCT base that is free under the ink.
    /// Source color and edge statistics select the decomposition before the
    /// base and ink frames are encoded once. Supports 8-bit RGB/RGBA in XYB;
    /// non-opaque pixels stay in the base and retain their original alpha.
    /// Layered RGB declares an opaque alpha channel. Lossless and progressive
    /// DC ignore it. Defaults to false.
    pub ink_layer: bool,
    /// Experimental: flat-color art with antialiased edges is split in one
    /// pass into a lossless palette base and a lossless coverage frame
    /// (kBlend) that carries the antialiasing; other images are coded as
    /// usual. 8-bit RGB input only; the output is near-lossless and
    /// independent of `distance`. Defaults to false.
    pub vector_layers: bool,
    /// Progressive DC for lossy VarDCT: the 1:8 DC image is coded as its own
    /// squeezed modular frame ahead of the main frame, so a decoder can show
    /// previews refining from 1:256 to 1:8 before any AC data arrives.
    /// Costs some size. XYB frames without alpha only; other frames ignore it.
    /// Defaults to false.
    pub progressive_dc: bool,
    /// Squeeze levels of the progressive DC frame, finest first; each level
    /// halves the preview resolution. `None` (default) squeezes down to an
    /// 8x8 preview. `Some(n)` keeps the `n` finest levels, clamped to between
    /// one and the most the image allows; fewer levels give fewer, larger
    /// previews. A single level costs noticeably more size; two or more cost
    /// about the same as the full pyramid. Used only with `progressive_dc`.
    pub progressive_dc_levels: Option<u32>,
    /// Codestream order of the lossy VarDCT groups (see [`GroupOrder`]).
    /// Default [`GroupOrder::Scanline`].
    pub group_order: GroupOrder,
    /// Choose VarDCT transforms by what each candidate costs under the
    /// image's own coefficient statistics instead of a fixed rate model.
    /// Patterned and synthetic content compresses much better; photographs
    /// are about unchanged. Takes effect at [`Speed::MediumSlow`] and above, where it
    /// costs encode time. Defaults to true.
    pub learned_rate: bool,
    /// Optional HDR gain map (see [`GainMap`]). When set, the gain map is
    /// encoded as a second JPEG XL codestream and shipped in a `jhgm`
    /// container box together with its ISO 21496-1 metadata. Forces the
    /// output into the JXL container form.
    pub gain_map: Option<GainMap>,
    /// Color transform of lossy frames (see [`CodingTransform`]).
    pub coding_transform: CodingTransform,
    /// Decoder-side grain synthesis over lossy VarDCT frames (see
    /// [`Grain`]). The decoder adds fine high-passed noise scaled per
    /// intensity; the frame carries only the 8-point strength table. Lossy
    /// modular frames and lossless encodes ignore it. Default [`Grain::Off`].
    pub grain: Grain,
}

/// Order in which a lossy VarDCT frame stores its groups. A progressive
/// decoder renders groups as they arrive, so this is the order the image
/// fills in. Costs a few bytes for the signaled permutation; the pixels are
/// identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupOrder {
    /// Row by row from the top left (the default).
    #[default]
    Scanline,
    /// Rings around the image center, each swept clockwise from the top;
    /// with several passes, every pass fills in from the center again.
    CenterFirst,
    /// Both corner-to-corner diagonals first, growing out from the center,
    /// then bands ever farther from them: the image draws an X that
    /// thickens until it fills the frame.
    Diagonals,
}

/// Which arm the **lossy** encoder uses. Besides VarDCT, jixel carries a lossy
/// modular arm (quantized Squeeze pyramid): multi-resolution residual coding
/// that outperforms block DCT on smooth/gradient-heavy content, most clearly
/// from medium quality down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossyModular {
    /// VarDCT only (the default).
    #[default]
    Off,
    /// Encode BOTH arms per frame — the modular arm at a distance calibrated
    /// to match the VarDCT arm's quality — and keep the smaller frame.
    /// [`Speed::Slow`] and above (falls back to VarDCT otherwise).
    Auto,
    /// Always use the modular arm when the frame supports it (XYB: RGB
    /// without alpha; other coding transforms: any integer image, gray, CMYK
    /// and alpha included; all above tiny sizes); other frames fall back to
    /// VarDCT.
    Force,
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
    /// The `alpha` plane is the K channel of a CMYK image (kBlack).
    pub(crate) black: bool,
    /// Bit depth declared in the codestream (default: 8).
    pub(crate) bits_per_sample: BitsPerSample,
    /// If true, encode losslessly via the modular encoder. `distance` is then
    /// ignored. RGB and alpha both round-trip bit-perfectly.
    pub(crate) lossless: bool,
    /// Invisible-color preprocessing (see `EncodeConfig::fill_invisible`).
    pub(crate) fill_invisible: bool,
    pub(crate) progressive: bool,
    pub(crate) patches: bool,
    #[cfg(feature = "splines")]
    pub(crate) splines: bool,
    #[cfg(feature = "splines")]
    pub(crate) dots: bool,
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
    /// Near-lossless error bound (see `EncodeConfig::near_lossless`).
    pub(crate) near_lossless: u16,
    /// Nearest-neighbor upsampling of cell content (see `EncodeConfig::upsampling`).
    pub(crate) upsampling: bool,
    /// Pre-encoded `jhgm` gain map bundle (see `EncodeConfig::gain_map`).
    pub(crate) gain_map: Option<EncodedGainMap>,
    /// Decoder-side grain synthesis (see `EncodeConfig::grain`).
    pub(crate) grain: Grain,
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
            fill_invisible: true,
            alpha_distance: None,
            progressive: false,
            patches: false,
            upsampling: false,
            #[cfg(feature = "splines")]
            splines: false,
            #[cfg(feature = "splines")]
            dots: false,
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
            arbiter: Arbiter::Ssimulacra2,
            decoding_speed: DecodingSpeed::Slow,
            near_lossless: 0,
            lossy_modular: LossyModular::Off,
            ink_layer: false,
            vector_layers: false,
            progressive_dc: false,
            progressive_dc_levels: None,
            group_order: GroupOrder::Scanline,
            learned_rate: true,
            gain_map: None,
            coding_transform: CodingTransform::Xyb,
            grain: Grain::Off,
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
            black: false,
            bits_per_sample: BitsPerSample::Eight,
            lossless: false,
            fill_invisible: true,
            progressive: false,
            patches: false,
            upsampling: false,
            #[cfg(feature = "splines")]
            splines: false,
            #[cfg(feature = "splines")]
            dots: false,
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
            near_lossless: 0,
            gain_map: None,
            grain: Grain::Off,
        }
    }
}

impl EncodeConfigImpl {
    /// Convenience builder with the given butteraugli distance and otherwise
    /// default settings (sRGB primaries and transfer).
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

    /// Embed an ICC profile in the codestream (predicted + entropy-coded ICC stream).
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

    /// Attach the alpha plane, coded in the frame's modular streams.
    /// Length must equal `xsize * ysize` of the image passed to encode.
    pub(crate) fn with_alpha(mut self, alpha: AlphaPlane) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Declare the extra plane as the K channel of a CMYK image.
    pub(crate) fn with_black(mut self, black: bool) -> Self {
        self.black = black;
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

    pub(crate) fn with_fill_invisible(mut self, fill_invisible: bool) -> Self {
        self.fill_invisible = fill_invisible;
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

    #[cfg(feature = "splines")]
    pub(crate) fn with_splines(mut self, splines: bool) -> Self {
        self.splines = splines;
        self
    }

    #[cfg(feature = "splines")]
    pub(crate) fn with_dots(mut self, dots: bool) -> Self {
        self.dots = dots;
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

    pub(crate) fn with_grain(mut self, grain: Grain) -> Self {
        self.grain = grain;
        self
    }

    /// Copy frame settings from a public `EncodeConfig`.
    pub(crate) fn with_progressive_from(self, config: &EncodeConfig) -> Self {
        #[cfg(feature = "splines")]
        let this = self.with_splines(config.splines).with_dots(config.dots);
        #[cfg(not(feature = "splines"))]
        let this = self;
        this.with_progressive(config.progressive)
            .with_grain(config.grain)
            .with_fill_invisible(config.fill_invisible)
            .with_patches(config.patches)
            .with_progressive_passes(config.progressive_passes)
            .with_progressive_shifts(config.progressive_shifts.clone())
            .with_speed(config.speed)
            .with_decoding_speed(config.decoding_speed)
    }

    pub(crate) fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    pub(crate) fn with_decoding_speed(mut self, decoding_speed: DecodingSpeed) -> Self {
        self.decoding_speed = decoding_speed;
        self
    }

    pub(crate) fn with_near_lossless(mut self, max_error: u16) -> Self {
        self.near_lossless = max_error;
        self
    }

    pub(crate) fn with_upsampling(mut self, upsampling: bool) -> Self {
        self.upsampling = upsampling;
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

    /// Attach an already-encoded gain map bundle to be written as a `jhgm` box.
    pub(crate) fn with_gain_map(mut self, gain_map: Option<EncodedGainMap>) -> Self {
        self.gain_map = gain_map;
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

    /// Embed an ICC profile in the codestream (predicted + entropy-coded ICC stream).
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

    /// Enable invisible-color preprocessing (see [`EncodeConfig::fill_invisible`]).
    /// Enabled by default; set to false to retain the original hidden colors
    /// as input to lossy encoding.
    pub fn with_fill_invisible(mut self, fill_invisible: bool) -> Self {
        self.fill_invisible = fill_invisible;
        self
    }

    /// Alpha distance of lossy frames (see [`EncodeConfig::alpha_distance`]);
    /// `0.0` keeps alpha lossless.
    pub fn with_alpha_distance(mut self, alpha_distance: f32) -> Self {
        self.alpha_distance = Some(alpha_distance);
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

    /// Experimental spline coding for lossy VarDCT (see [`EncodeConfig::splines`]).
    /// Requires the `splines` Cargo feature.
    #[cfg(feature = "splines")]
    pub fn with_splines(mut self, splines: bool) -> Self {
        self.splines = splines;
        self
    }

    /// Experimental bright/dark dot coding for lossy VarDCT (see [`EncodeConfig::dots`]).
    /// Requires the `splines` Cargo feature.
    #[cfg(feature = "splines")]
    pub fn with_dots(mut self, dots: bool) -> Self {
        self.dots = dots;
        self
    }

    /// Code lossy frames in `transform` (see [`CodingTransform`]).
    pub fn with_coding_transform(mut self, transform: CodingTransform) -> Self {
        self.coding_transform = transform;
        self
    }

    /// Decoder-side grain synthesis (see [`EncodeConfig::grain`]).
    pub fn with_grain(mut self, grain: Grain) -> Self {
        self.grain = grain;
        self
    }

    /// Select the lossy encoding arm (see [`LossyModular`]).
    pub fn with_lossy_modular(mut self, mode: LossyModular) -> Self {
        self.lossy_modular = mode;
        self
    }

    /// Experimental ink layer for cel and line art (see
    /// [`EncodeConfig::ink_layer`]).
    pub fn with_ink_layer(mut self, ink_layer: bool) -> Self {
        self.ink_layer = ink_layer;
        self
    }

    /// Experimental vector layers for flat-color art (see
    /// [`EncodeConfig::vector_layers`]).
    pub fn with_vector_layers(mut self, vector_layers: bool) -> Self {
        self.vector_layers = vector_layers;
        self
    }

    /// Code the lossy DC as a progressive DC frame (see
    /// [`EncodeConfig::progressive_dc`]).
    pub fn with_progressive_dc(mut self, progressive_dc: bool) -> Self {
        self.progressive_dc = progressive_dc;
        self
    }

    /// Store the lossy VarDCT groups in `order` (see [`GroupOrder`]).
    pub fn with_group_order(mut self, order: GroupOrder) -> Self {
        self.group_order = order;
        self
    }

    /// Limit the progressive DC frame to its `levels` finest squeeze levels
    /// (see [`EncodeConfig::progressive_dc_levels`]).
    pub fn with_progressive_dc_levels(mut self, levels: u32) -> Self {
        self.progressive_dc_levels = Some(levels);
        self
    }

    /// Price VarDCT transform candidates by the image's own coefficient
    /// statistics (see [`EncodeConfig::learned_rate`]).
    pub fn with_learned_rate(mut self, learned_rate: bool) -> Self {
        self.learned_rate = learned_rate;
        self
    }

    /// Attach an HDR gain map (see [`GainMap`]). It is encoded as its own
    /// JPEG XL codestream and written into a `jhgm` container box.
    pub fn with_gain_map(mut self, gain_map: GainMap) -> Self {
        self.gain_map = Some(gain_map);
        self
    }

    /// Set the worker-thread count (see `EncodeConfig::num_threads`).
    pub fn with_num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Select the lossless decode-speed/density tradeoff (see [`DecodingSpeed`]).
    pub fn with_decoding_speed(mut self, decoding_speed: DecodingSpeed) -> Self {
        self.decoding_speed = decoding_speed;
        self
    }

    /// Bound the per-channel error of a lossless encode (see
    /// [`EncodeConfig::near_lossless`]).
    pub fn with_near_lossless(mut self, max_error: u16) -> Self {
        self.near_lossless = max_error;
        self
    }

    /// Decoder-side nearest-neighbor upsampling of integer-upscaled content
    /// (see [`EncodeConfig::upsampling`]).
    pub fn with_upsampling(mut self, upsampling: bool) -> Self {
        self.upsampling = upsampling;
        self
    }

    /// Select the metric the lossy encoder tunes for (see [`Arbiter`]).
    pub fn with_arbiter(mut self, arbiter: Arbiter) -> Self {
        self.arbiter = arbiter;
        self
    }

    /// Select the encode-speed/density tradeoff (see [`Speed`]).
    /// [`Speed::Slow`] and [`Speed::ExtraSlow`] also turn on [`EncodeConfig::patches`]
    /// and [`EncodeConfig::vector_layers`]; [`Speed::Medium`] and above turn on
    /// [`EncodeConfig::ink_layer`].
    pub fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        if speed.effort().patches_default {
            self.patches = true;
        }
        if speed.effort().ink_layer_default {
            self.ink_layer = true;
        }
        if speed.effort().vector_layers_default {
            self.vector_layers = true;
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
    if !ctx.speed.effort().yellow_opsin || !ctx.coding.is_xyb() {
        return;
    }
    let selection = crate::yellow_opsin::select_yellow(linear, distance);
    if let Some(m) = selection.matrix {
        ctx.set_xyb_matrix(m, distance, selection.default_channel_weights);
    }
    // The yellow selector runs before XYB conversion, so the post-XYB
    // `x_heavy` flag is not available yet. Apply only its own staged B scale
    // here; `frame::encode_frame` ORs in the X-gradient decision later.
    ctx.raise_b_qm_scale(selection.b_qm_scale);
}

/// Squeeze quantizer distance of lossy alpha per unit of alpha distance, in
/// the lossy-modular arm's luma units.
const ALPHA_SQUEEZE_SCALE: f32 = 4.0;
const MAX_ALPHA_DISTANCE: f32 = 25.0;

/// Whether lossy alpha learns its context trees. The trees use no weighted
/// predictor, so the tree-free decoding tier rules them out; they halve the
/// alpha bits but cost encode time, so only the slow tiers learn them.
fn alpha_learned_trees(config: &EncodeConfig) -> bool {
    config.decoding_speed != DecodingSpeed::Fastest
        && matches!(config.speed, Speed::Slow | Speed::ExtraSlow)
}

pub(crate) fn lossy_context(
    config: &EncodeConfig,
    distance: f32,
    xyb: XybMatrix,
    pixels: usize,
) -> EncodingContext {
    // The fastest tiers' work per pixel is deliberately small, so
    // oversubscribing tiny images spends more time constructing and
    // synchronizing workers than it saves. Keep the requested count as a maximum and give each active lane
    // roughly 64K pixels; larger images still receive the full thread budget.
    let num_threads = if config.speed.effort().full_threads {
        config.num_threads
    } else {
        config.num_threads.min(pixels.div_ceil(64 * 1024).max(1))
    };
    let mut ctx = EncodingContext::new(config.speed, config.arbiter, xyb, distance, num_threads);
    ctx.lossy_modular = config.lossy_modular;
    ctx.progressive_dc = config.progressive_dc;
    ctx.progressive_dc_levels = config.progressive_dc_levels;
    ctx.group_order = config.group_order;
    ctx.learned_rate = config.learned_rate && config.speed.effort().learned_rate;
    ctx.grain = config.grain;
    let alpha_distance = match config.alpha_distance {
        None => distance,
        Some(d) if d.is_nan() => 0.0,
        Some(d) => d.clamp(0.0, MAX_ALPHA_DISTANCE),
    };
    ctx.extra_squeeze_distance = ALPHA_SQUEEZE_SCALE * alpha_distance;
    ctx.extra_learned_trees = alpha_learned_trees(config);
    #[cfg(feature = "splines")]
    {
        let slow = config.speed.effort().splines && config.decoding_speed == DecodingSpeed::Slow;
        ctx.splines = config.splines && slow;
        ctx.dots = config.dots && slow;
    }
    ctx
}

pub(crate) fn for_each_linear_band<F>(
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

fn linearize_rgb<T, X, const CHANNELS: usize>(
    input: &[T],
    width: usize,
    height: usize,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    transform: &X,
) -> Image3F
where
    T: Copy + Sync,
    X: RgbBlockTransform<T>,
{
    debug_assert!(CHANNELS >= 3);
    let mut linear = Image3F::new(width, height);
    for_each_linear_band(&mut linear, ctx, scratch, |pixel_start, [r, g, b]| {
        let src = &input[pixel_start * CHANNELS..][..r.len() * CHANNELS];
        transform.transform_block(src.as_chunks::<CHANNELS>().0, [r, g, b]);
    });
    linear
}

fn linearize_gray<T, D>(
    input: &[T],
    width: usize,
    height: usize,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    decoder: &D,
) -> Image3F
where
    T: Copy + Sync,
    D: TransferDecoder<T>,
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
            let linear = decoder.decode(v);
            *r = linear;
            *g = linear;
            *b = linear;
        }
    });
    linear
}

/// Encode interleaved 8-bit RGB samples. Lossy encoding interprets the samples
/// using `config.color_encoding`; the default is sRGB.
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
                .with_gain_map(encode_gain_map(config)?)
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_near_lossless(config.near_lossless)
                .with_upsampling(config.upsampling)
                .with_num_threads(config.num_threads),
        );
    }
    // The ink plan serves both layer paths: the ink stack codes it, and the
    // vector race reads its glyph-text flag.
    let layers = config.coding_transform == CodingTransform::Xyb && !config.progressive_dc;
    let ink_candidates = (layers && (config.ink_layer || (config.vector_layers && config.patches)))
        .then(|| crate::ink_layer::InkCandidates::new_fast(input, None, width, height))
        .flatten();
    let ink_plan = ink_candidates.and_then(|candidates| candidates.select(config.distance));
    let plain_encode = || {
        let mut plain_config = config.clone();
        plain_config.ink_layer = false;
        plain_config.vector_layers = false;
        encode_image(input, width, height, &plain_config)
    };
    let mut plain = None;
    let mut vector = None;
    // Vector layers are near-lossless and cost the same at every distance:
    // they replace the lossy frame only where its own size would be close.
    // The vector encode does not depend on the distance; it stands wherever
    // it is no larger than the VarDCT encode the distance asks for, taken
    // from the edge-cost model instead of a second encode.
    // The vector encode spends what VarDCT would at this distance: its
    // band precision follows the distance, then drops until the budget
    // covers it, but never below the precision of twice the distance; the
    // bare palette base only stands where that floor reaches it.
    if config.vector_layers {
        let steps = crate::vector_layers::coverage_steps(config.distance);
        if let Some(split) = crate::vector_layers::split(input, width, height, steps) {
            // The model sizes the VarDCT encode within 0.78..1.39 of its
            // measure; the layers, far above VarDCT in quality, are kept up
            // to that margin. Repeated-glyph patches code thin text far
            // below the model, which does not see them: there the plain
            // encode sets the budget and the vector encode is one candidate
            // among the layer stacks.
            let glyph_text =
                config.patches && ink_plan.as_ref().is_some_and(|plan| plan.glyph_text);
            if glyph_text {
                plain = Some(plain_encode()?);
            }
            let budget = match &plain {
                Some(plain) => plain.len() as f64,
                None => {
                    crate::vector_layers::vardct_bytes(input, width, height, config.distance)
                        * crate::vector_layers::VARDCT_BYTES_MARGIN
                }
            };
            let encoded = encode_vector_layers(split, width, height, config)?;
            let mut fitted = None;
            if encoded.bytes.len() as f64 <= budget {
                fitted = Some(encoded.bytes);
            } else if (encoded.base as f64) < budget {
                let room = ((budget - encoded.base as f64) / encoded.band.max(1) as f64
                    * crate::vector_layers::band_bytes_ratio(steps))
                .max(0.0);
                let fit = (5.0 * room - 1.0).exp2().floor().clamp(1.0, steps as f64) as u32;
                let floor = crate::vector_layers::coverage_steps(2.0 * config.distance);
                if (floor..steps).contains(&fit)
                    && let Some(split) = crate::vector_layers::split(input, width, height, fit)
                {
                    fitted = Some(encode_vector_layers(split, width, height, config)?.bytes);
                }
            }
            match (fitted, &plain) {
                (Some(bytes), None) => return Ok(bytes),
                (Some(bytes), Some(_)) => vector = Some(bytes),
                (None, _) => {}
            }
        }
    }
    // On a tie the vector encode stands: exact where the others are lossy.
    let smaller = |vector: Vec<u8>, other: Vec<u8>| {
        if vector.len() <= other.len() {
            vector
        } else {
            other
        }
    };
    if config.ink_layer && ink_plan.is_some() {
        let inked = select_ink_layered(
            ink_plan,
            width,
            height,
            config,
            None,
            crate::vector_layers::vardct_bytes(input, width, height, config.distance),
            || match plain {
                Some(plain) => Ok(plain),
                None => plain_encode(),
            },
        )?;
        return Ok(match vector {
            Some(vector) => smaller(vector, inked),
            None => inked,
        });
    }
    match (vector, plain) {
        (Some(vector), Some(plain)) => return Ok(smaller(vector, plain)),
        (Some(vector), None) => return Ok(vector),
        (None, Some(plain)) => return Ok(plain),
        (None, None) => {}
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let transform = ColorTransform::for_integer(
        8,
        config.color_encoding.transfer,
        config.color_encoding.primaries,
    );
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_rgb::<_, _, 3>(input, width, height, &ctx, &mut scratch, &transform);
    let mut ctx = ctx;
    ctx.set_coding(config.coding_transform, distance);
    apply_yellow_opsin(&mut ctx, &linear, distance);
    let cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_gain_map(encode_gain_map(config)?)
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    encode_samples_lossy(
        Samples::U8 {
            data: input,
            stride: 3,
        },
        false,
        linear,
        &cfg,
        &ctx,
        &mut scratch,
    )
}

/// Encode the selected decomposition once, or use the ordinary path.
fn select_ink_layered(
    plan: Option<crate::ink_layer::InkLayers>,
    width: usize,
    height: usize,
    config: &EncodeConfig,
    alpha: Option<&AlphaPlane>,
    budget: f64,
    ordinary: impl FnOnce() -> Result<Vec<u8>, EncodeError>,
) -> Result<Vec<u8>, EncodeError> {
    let Some(plan) = plan else {
        return ordinary();
    };
    // Repeated-glyph patches may code thin text better than the ink frame,
    // and the model does not see them: the plain encode decides, and the
    // stack has no quality edge over patched text to pay for.
    let glyph_text = plan.glyph_text && config.patches;
    let stack = InkStack::new(plan, width, height, config, alpha, budget)?;
    if !glyph_text && stack.bytes.len() as f64 <= budget {
        return Ok(stack.bytes);
    }
    // Past the modeled VarDCT size the plain encode decides: the stack, exact
    // on its strokes, may cost one quality step more than it.
    let plain = ordinary()?;
    let step = if glyph_text { 1.0 } else { INK_STACK_STEP };
    Ok(if stack.bytes.len() as f64 <= plain.len() as f64 * step {
        stack.bytes
    } else {
        plain
    })
}

/// Bytes an ink stack may cost relative to the plain encode it replaces: one
/// quality step of the rate curve.
const INK_STACK_STEP: f64 = 1.15;

/// Coverage levels of the ink frame at a distance: 7 at distance 1, more
/// below it, fewer above, within the palette's room for the inks.
fn ink_levels(distance: f32, inks: usize) -> u32 {
    (7.0 / distance.max(0.01))
        .round()
        .clamp(2.0, (255 / inks.max(1)) as f32) as u32
}

/// One VarDCT base and its palette coverage frame.
struct InkStack {
    width: usize,
    height: usize,
    ctx: EncodingContext,
    base_frame: BitWriter,
    /// Ink colors and coverage levels; index zero is transparent.
    inks: Vec<[u8; 3]>,
    ink: Vec<u8>,
    coverage: Vec<u8>,
    bytes: Vec<u8>,
}

impl InkStack {
    fn new(
        plan: crate::ink_layer::InkLayers,
        width: usize,
        height: usize,
        config: &EncodeConfig,
        alpha: Option<&AlphaPlane>,
        budget: f64,
    ) -> Result<Self, EncodeError> {
        let mut scratch = Box::<CoderScratch>::default();
        let (ctx, base_frame) =
            encode_ink_base(&plan.base, width, height, config, alpha, &mut scratch)?;
        drop(plan.base);
        let mut stack = Self {
            width,
            height,
            ctx,
            base_frame,
            inks: plan.inks,
            ink: plan.ink,
            coverage: plan.coverage,
            bytes: Vec::new(),
        };
        // The coverage precision follows the distance, then drops until the
        // modeled VarDCT size, within the model's margin, covers the stack;
        // the hard mask stands last.
        let budget = budget * crate::vector_layers::VARDCT_BYTES_MARGIN;
        let mut levels = ink_levels(config.distance, stack.inks.len());
        loop {
            stack.bytes = stack.assemble(config, alpha, levels, &mut scratch)?;
            if stack.bytes.len() as f64 <= budget || levels == 1 {
                return Ok(stack);
            }
            levels = [7, 5, 3, 2, 1]
                .into_iter()
                .find(|&l| l < levels)
                .unwrap_or(1);
        }
    }

    /// Image header, saved base frame and final palette ink frame with the
    /// coverage on `levels` steps above zero.
    fn assemble(
        &self,
        config: &EncodeConfig,
        alpha: Option<&AlphaPlane>,
        levels: u32,
        scratch: &mut CoderScratch,
    ) -> Result<Vec<u8>, EncodeError> {
        let mut palette = vec![([0u8; 3], 0u8)];
        let mut entries = vec![vec![0u8; levels as usize + 1]; self.inks.len()];
        for (slot, &ink) in self.inks.iter().enumerate() {
            for step in 1..=levels {
                entries[slot][step as usize] = palette.len() as u8;
                palette.push((ink, ((step * 255 + levels / 2) / levels) as u8));
            }
        }
        let indices: Vec<u8> = self
            .ink
            .iter()
            .zip(&self.coverage)
            .map(|(&k, &a)| {
                if k == 0 {
                    0
                } else {
                    entries[k as usize - 1][((a as u32 * levels + 127) / 255) as usize]
                }
            })
            .collect();
        let pixels = self.width * self.height;
        let opaque = alpha
            .is_none()
            .then(|| AlphaPlane::from_u8(vec![255; pixels]));
        let base_alpha = alpha.or(opaque.as_ref()).expect("base alpha");
        let mut w = BitWriter::new();
        w.write(8, 0xFF);
        w.write(8, CODESTREAM_MARKER as u64);
        write_size_header(self.width, self.height, &mut w);
        write_image_metadata(
            &ImageMetadataParams {
                tone_mapping: config.tone_mapping(),
                color_encoding: &config.color_encoding,
                alpha: Some(base_alpha),
                black: false,
                icc_profile: config.icc_profile.as_deref(),
                bps: BitsPerSample::Eight,
                lossless: false,
                xyb_encoded: true,
                xyb: &self.ctx.xyb,
                grayscale: false,
                orientation: config.orientation,
                lossy_modular_16: false,
                extra_16: false,
                upsampling_kernel: None,
            },
            scratch,
            &mut w,
        );
        w.append(&self.base_frame);
        let transform = ColorTransform::for_integer(
            8,
            config.color_encoding.transfer,
            config.color_encoding.primaries,
        );
        w.append(&encode_ink_frame(
            &self.ctx,
            &transform,
            &palette,
            &indices,
            self.width,
            self.height,
            config,
            scratch,
        )?);
        finalize_container(
            w.into_bytes(),
            config.exif.as_deref(),
            config.xmp.as_deref(),
            config.brotli_compression.as_deref(),
            false,
            encode_gain_map(config)?.as_ref(),
        )
    }
}

/// The VarDCT base of a layered codestream, saved to reference slot 0, with
/// the context whose opsin matrix every ink frame must share.
fn encode_ink_base(
    base: &[u8],
    width: usize,
    height: usize,
    config: &EncodeConfig,
    original_alpha: Option<&AlphaPlane>,
    scratch: &mut CoderScratch,
) -> Result<(EncodingContext, BitWriter), EncodeError> {
    use crate::encoding_context::FrameRole;
    let distance = config.distance.max(MIN_DISTANCE);
    let transform = ColorTransform::for_integer(
        8,
        config.color_encoding.transfer,
        config.color_encoding.primaries,
    );
    let pixels = width * height;
    let opaque = original_alpha
        .is_none()
        .then(|| AlphaPlane::from_u8(vec![255; pixels]));
    let base_alpha = original_alpha.or(opaque.as_ref()).expect("base alpha");
    let mut ctx = lossy_context(config, distance, XybMatrix::SPEC, pixels);
    ctx.lossy_modular = LossyModular::Off;
    // Coverage is zero wherever the source alpha is not opaque. Keeping the
    // reference alpha exact makes kBlend preserve it at those pixels.
    ctx.extra_squeeze_distance = 0.0;
    ctx.extra_step = 1;
    ctx.frame_role = FrameRole::SavedFirst;
    let linear = linearize_rgb::<_, _, 3>(base, width, height, &ctx, scratch, &transform);
    ctx.set_coding(CodingTransform::Xyb, distance);
    apply_yellow_opsin(&mut ctx, &linear, distance);
    let mut frame = BitWriter::new();
    encode_frame(
        &ctx,
        scratch,
        distance,
        linear,
        None,
        false,
        Some(base_alpha),
        config.fill_invisible,
        &[0],
        config.patches,
        true,
        &mut frame,
    )?;
    Ok((ctx, frame))
}

/// A colour on the XYB lattice of the VarDCT base, as the YCoCg triple the
/// signaled RCT restores exactly: Y, X and B-Y on the default LF dequant
/// steps.
fn xyb_lattice_ycocg(
    ctx: &EncodingContext,
    transform: &ColorTransform<crate::color::TransferLut>,
    rgb: [u8; 3],
) -> (i32, i32, i32) {
    let (mut r, mut g, mut b) = ([0.0], [0.0], [0.0]);
    transform.transform_block(&[rgb], [&mut r, &mut g, &mut b]);
    let (x, y, b) = crate::xyb::rgb_to_xyb_pixel_f32(&ctx.xyb, r[0], g[0], b[0]);
    let y = (y * crate::quant_weights::INV_DC_QUANT[1]).round() as i32;
    forward_ycocg(
        y,
        (x * crate::quant_weights::INV_DC_QUANT[0]).round() as i32,
        (b * crate::quant_weights::INV_DC_QUANT[2]).round() as i32 - y,
    )
}

/// The palette of inks on the base frame's XYB lattice, with exact coverage.
#[allow(clippy::too_many_arguments)]
fn encode_ink_frame(
    ctx: &EncodingContext,
    transform: &ColorTransform<crate::color::TransferLut>,
    colors: &[([u8; 3], u8)],
    indices: &[u8],
    width: usize,
    height: usize,
    config: &EncodeConfig,
    scratch: &mut CoderScratch,
) -> Result<BitWriter, EncodeError> {
    let palette: Vec<_> = colors
        .iter()
        .map(|&(ink, alpha)| (xyb_lattice_ycocg(ctx, transform, ink), alpha))
        .collect();
    let mut image = Image3Si::new(width, height);
    let mut alpha = vec![0u8; width * height];
    for (y, row) in indices.chunks_exact(width).enumerate() {
        for c in 0..3 {
            let dst = image.plane_row_mut(c, y);
            for (d, &k) in dst.iter_mut().zip(row) {
                let v = palette[k as usize].0;
                *d = match c {
                    0 => v.0,
                    1 => v.1,
                    _ => v.2,
                };
            }
        }
        for (a, &k) in alpha[y * width..][..width].iter_mut().zip(row) {
            *a = palette[k as usize].1;
        }
    }
    let alpha = AlphaPlane::from_u8(alpha);
    // The ink frame is a full-resolution learned-tree frame: Slow's LZ77 and
    // refinement take 14% off it on text and the base already paid the time.
    let speed = if config.speed >= Speed::Slow {
        Speed::Slow
    } else {
        Speed::Medium
    };
    let pool = crate::thread_pool::ThreadPool::new_lossless(config.num_threads);
    let mut frame = BitWriter::new();
    if !crate::lossless::encode_frame_lossless_xyb_ink(
        &image,
        &alpha,
        speed,
        config.decoding_speed,
        &pool,
        scratch,
        &mut frame,
    ) {
        return Err(EncodeError::EmptyImage);
    }
    Ok(frame)
}

/// Two lossless modular frames for a vector split: the palette base saved
/// to reference slot 0, then the coverage frame blended over it (kBlend).
struct VectorEncode {
    bytes: Vec<u8>,
    /// Bytes of the palette base frame and of the coverage band frame.
    base: usize,
    band: usize,
}

fn encode_vector_layers(
    split: crate::vector_layers::VectorSplit,
    width: usize,
    height: usize,
    config: &EncodeConfig,
) -> Result<VectorEncode, EncodeError> {
    use crate::patches::ModularFrameKind;
    let pixels = width * height;
    let planes = |rgb: &[u8]| {
        let mut image = Image3Si::new(width, height);
        let mut row = vec![(0i32, 0i32, 0i32); width];
        for (y, src) in rgb.as_chunks::<3>().0.chunks_exact(width).enumerate() {
            for (out, &[r, g, b]) in row.iter_mut().zip(src) {
                *out = forward_ycocg(r as i32, g as i32, b as i32);
            }
            for c in 0..3 {
                let dst = image.plane_row_mut(c, y);
                for (d, v) in dst.iter_mut().zip(&row) {
                    *d = match c {
                        0 => v.0,
                        1 => v.1,
                        _ => v.2,
                    };
                }
            }
        }
        image
    };
    let base = planes(&split.base);
    let layer = planes(&split.layer);
    let opaque = AlphaPlane::from_u8(vec![255; pixels]);
    let banded = split.coverage.iter().any(|&a| a != 0);
    let coverage = AlphaPlane::from_u8(split.coverage);
    let mut scratch = Box::new(CoderScratch::lossless());
    let mut w = BitWriter::new();
    w.write(8, 0xFF);
    w.write(8, CODESTREAM_MARKER as u64);
    write_size_header(width, height, &mut w);
    write_image_metadata(
        &ImageMetadataParams {
            tone_mapping: config.tone_mapping(),
            color_encoding: &config.color_encoding,
            alpha: Some(&opaque),
            black: false,
            icc_profile: config.icc_profile.as_deref(),
            bps: BitsPerSample::Eight,
            lossless: true,
            xyb_encoded: false,
            xyb: &XybMatrix::SPEC,
            grayscale: false,
            orientation: config.orientation,
            lossy_modular_16: false,
            extra_16: false,
            upsampling_kernel: None,
        },
        &mut scratch,
        &mut w,
    );
    // The frames hold flat regions and a thin band: the Medium lossless
    // effort codes them within 12% of Slow at a third of the time; the
    // fastest tiers have no palette or learned tree and triple the size.
    let speed = if cfg!(feature = "extraslow") && config.speed >= Speed::ExtraSlow {
        Speed::Slow
    } else {
        Speed::Medium
    };
    // The Weighted Predictor costs these flat frames decode time and gains
    // them nothing: 1-2% larger with it.
    let decoding_speed = if config.decoding_speed == DecodingSpeed::Fastest {
        DecodingSpeed::Fastest
    } else {
        DecodingSpeed::Fast
    };
    let pool = crate::thread_pool::ThreadPool::new_lossless(config.num_threads);
    let start = w.bits_written();
    crate::lossless::encode_frame_lossless_layer(
        &base,
        &opaque,
        if banded {
            ModularFrameKind::RgbSavedFirst
        } else {
            ModularFrameKind::Regular
        },
        speed,
        decoding_speed,
        &pool,
        &mut scratch,
        &mut w,
    );
    let base_bytes = (w.bits_written() - start) / 8;
    if banded {
        crate::lossless::encode_frame_lossless_layer(
            &layer,
            &coverage,
            ModularFrameKind::RgbBlendLast,
            speed,
            decoding_speed,
            &pool,
            &mut scratch,
            &mut w,
        );
    }
    let band_bytes = (w.bits_written() - start) / 8 - base_bytes;
    Ok(VectorEncode {
        bytes: finalize_container(
            w.into_bytes(),
            config.exif.as_deref(),
            config.xmp.as_deref(),
            config.brotli_compression.as_deref(),
            false,
            encode_gain_map(config)?.as_ref(),
        )?,
        base: base_bytes,
        band: band_bytes,
    })
}

/// Encode interleaved 8-bit RGBA samples. Lossy RGB samples use
/// `config.color_encoding`; alpha is linear opacity.
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
                .with_gain_map(encode_gain_map(config)?)
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_near_lossless(config.near_lossless)
                .with_upsampling(config.upsampling)
                .with_num_threads(config.num_threads),
        );
    }
    if config.ink_layer
        && config.coding_transform == CodingTransform::Xyb
        && !config.progressive_dc
        && width * height >= 64 * 64
    {
        let pixels = input.as_chunks::<4>().0;
        let rgb: Vec<_> = pixels.iter().flat_map(|&[r, g, b, _]| [r, g, b]).collect();
        let alpha = AlphaPlane::from_u8(pixels.iter().map(|p| p[3]).collect());
        let AlphaPlane::U8(alpha_bytes) = &alpha else {
            unreachable!()
        };
        if let Some(candidates) =
            crate::ink_layer::InkCandidates::new_fast(&rgb, Some(alpha_bytes), width, height)
        {
            let plan = candidates.select(config.distance);
            let budget = crate::vector_layers::vardct_bytes(&rgb, width, height, config.distance);
            drop(rgb);
            return select_ink_layered(plan, width, height, config, Some(&alpha), budget, || {
                let mut plain_config = config.clone();
                plain_config.ink_layer = false;
                plain_config.vector_layers = false;
                encode_image_with_alpha(input, width, height, &plain_config)
            });
        }
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let transform = ColorTransform::for_integer(
        8,
        config.color_encoding.transfer,
        config.color_encoding.primaries,
    );
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_rgb::<_, _, 4>(input, width, height, &ctx, &mut scratch, &transform);
    let mut ctx = ctx;
    ctx.set_coding(config.coding_transform, distance);
    apply_yellow_opsin(&mut ctx, &linear, distance);
    let alpha_plane = input.as_chunks::<4>().0.iter().map(|px| px[3]).collect();
    let cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_alpha(AlphaPlane::from_u8(alpha_plane))
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_gain_map(encode_gain_map(config)?)
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    encode_samples_lossy(
        Samples::U8 {
            data: input,
            stride: 4,
        },
        false,
        linear,
        &cfg,
        &ctx,
        &mut scratch,
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
        let mut image = Image3Si::try_new_per_plane([(width, height), (0, 0), (0, 0)])?;
        for (out, &value) in image.plane_mut(0).as_mut_slice().iter_mut().zip(luma) {
            *out = value.to_signed_int(8);
        }
        crate::lossless::quantize_planes(&mut image, 255, i32::from(config.near_lossless));
        let alpha_plane = alpha.map(AlphaPlane::U8);
        return encode_lossless_planes(
            &image,
            &alpha_plane,
            8,
            &EncodeConfigImpl::with_distance(config.distance)
                .with_lossless(true)
                .with_grayscale(true)
                .with_patches(config.patches)
                .with_icc_profile(config.icc_profile.clone())
                .with_exif(config.exif.clone())
                .with_gain_map(encode_gain_map(config)?)
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_near_lossless(config.near_lossless)
                .with_upsampling(config.upsampling)
                .with_num_threads(config.num_threads),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let decoder = TransferLut::new(8, config.color_encoding.transfer);
    let mut ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    ctx.set_coding(config.coding_transform, distance);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_gray(luma, width, height, &ctx, &mut scratch, &decoder);
    let mut cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_grayscale(true)
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_gain_map(encode_gain_map(config)?)
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    if let Some(a) = alpha {
        cfg = cfg.with_alpha(AlphaPlane::from_u8(a));
    }
    encode_samples_lossy(
        Samples::U8 {
            data: luma,
            stride: 1,
        },
        true,
        linear,
        &cfg,
        &ctx,
        &mut scratch,
    )
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
        let mut image = Image3Si::try_new_per_plane([(width, height), (0, 0), (0, 0)])?;
        for (out, &value) in image.plane_mut(0).as_mut_slice().iter_mut().zip(luma) {
            *out = value.to_signed_int(bps.bits() as u8);
        }
        let alpha_plane = alpha.map(|mut data| {
            let bits = bps.bits() as u8;
            for value in &mut data {
                *value = (*value).to_signed_int(bits) as u16;
            }
            AlphaPlane::U16 { data, bits }
        });
        return encode_lossless_planes(
            &image,
            &alpha_plane,
            bps.bits() as u8,
            &EncodeConfigImpl::with_distance(config.distance)
                .with_lossless(true)
                .with_grayscale(true)
                .with_patches(config.patches)
                .with_bits_per_sample(bps)
                .with_icc_profile(config.icc_profile.clone())
                .with_exif(config.exif.clone())
                .with_gain_map(encode_gain_map(config)?)
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_near_lossless(config.near_lossless)
                .with_upsampling(config.upsampling)
                .with_num_threads(config.num_threads),
        );
    }

    let distance = config.distance.max(MIN_DISTANCE);
    let decoder = TransferLut::new(bps.bits(), config.color_encoding.transfer);
    let mut ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    ctx.set_coding(config.coding_transform, distance);
    let mut scratch = Box::<CoderScratch>::default();
    let linear = linearize_gray(luma, width, height, &ctx, &mut scratch, &decoder);

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
        .with_gain_map(encode_gain_map(config)?)
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    if let Some(ap) = alpha_plane {
        cfg = cfg.with_alpha(ap);
    }
    encode_samples_lossy(
        Samples::U16 {
            data: luma,
            stride: 1,
            max: ((1u32 << bps.bits()) - 1) as u16,
        },
        true,
        linear,
        &cfg,
        &ctx,
        &mut scratch,
    )
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
                .with_gain_map(encode_gain_map(config)?)
                .with_xmp(config.xmp.clone())
                .with_brotli_compression(config.brotli_compression.clone())
                .with_orientation(config.orientation)
                .with_color_encoding(config.color_encoding)
                .with_intensity_target(config.intensity_target)
                .with_speed(config.speed)
                .with_decoding_speed(config.decoding_speed)
                .with_near_lossless(config.near_lossless)
                .with_upsampling(config.upsampling)
                .with_num_threads(config.num_threads),
        );
    }
    let distance = config.distance.max(MIN_DISTANCE);
    let transform = ColorTransform::for_integer(
        bps.bits(),
        config.color_encoding.transfer,
        config.color_encoding.primaries,
    );
    let ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    let mut scratch = Box::<CoderScratch>::default();

    // For 16-bit, (1 << 16) - 1 overflows u16's shift; compute in u32 and cap.
    let bp_max: u16 = if bps.bits() >= 16 {
        u16::MAX
    } else {
        ((1u32 << bps.bits()) - 1) as u16
    };

    if has_alpha {
        let linear = linearize_rgb::<_, _, 4>(input, width, height, &ctx, &mut scratch, &transform);
        let mut ctx = ctx;
        ctx.set_coding(config.coding_transform, distance);
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
            .with_gain_map(encode_gain_map(config)?)
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_samples_lossy(
            Samples::U16 {
                data: input,
                stride: 4,
                max: bp_max,
            },
            false,
            linear,
            &cfg,
            &ctx,
            &mut scratch,
        )
    } else {
        let linear = linearize_rgb::<_, _, 3>(input, width, height, &ctx, &mut scratch, &transform);
        let mut ctx = ctx;
        ctx.set_coding(config.coding_transform, distance);
        apply_yellow_opsin(&mut ctx, &linear, distance);
        let cfg = EncodeConfigImpl::with_distance(distance)
            .with_progressive_from(config)
            .with_bits_per_sample(bps)
            .with_icc_profile(config.icc_profile.clone())
            .with_exif(config.exif.clone())
            .with_gain_map(encode_gain_map(config)?)
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_samples_lossy(
            Samples::U16 {
                data: input,
                stride: 3,
                max: bp_max,
            },
            false,
            linear,
            &cfg,
            &ctx,
            &mut scratch,
        )
    }
}

/// Shared float (f16/f32) RGB(A) **lossy** encode path. `input` is interleaved
/// f32 samples (encoded per `config.color_encoding`, normalized to [0, 1]), 3
/// or 4 per pixel. Alpha,
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
    {
        let mut metadata_scratch = Box::new(CoderScratch::lossless());
        write_image_metadata(
            &ImageMetadataParams {
                tone_mapping: config.tone_mapping(),
                color_encoding: &config.color_encoding,
                alpha: alpha.as_ref(),
                black: false,
                icc_profile: config.icc_profile.as_deref(),
                bps: BitsPerSample::F32,
                lossless: true,
                xyb_encoded: false,
                xyb: &XybMatrix::SPEC,
                grayscale: false,
                orientation: config.orientation,
                lossy_modular_16: false,
                extra_16: false,
                upsampling_kernel: None,
            },
            &mut metadata_scratch,
            &mut w,
        );
    }
    encode_frame_lossless_float(&image3s, alpha.as_ref(), config.num_threads, &mut w);
    let codestream = w.into_bytes();
    let alpha_bits_md = if has_alpha { 32 } else { 0 };
    let gain_map = encode_gain_map(config)?;
    finalize_container(
        codestream,
        config.exif.as_deref(),
        config.xmp.as_deref(),
        config.brotli_compression.as_deref(),
        needs_level_10(32, true, alpha_bits_md),
        gain_map.as_ref(),
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
    let transform = ColorTransform::for_float(
        config.color_encoding.transfer,
        config.color_encoding.primaries,
    );

    if has_alpha {
        let linear = linearize_rgb::<_, _, 4>(input, width, height, &ctx, &mut scratch, &transform);
        let mut ctx = ctx;
        ctx.set_coding(config.coding_transform, distance);
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
            .with_gain_map(encode_gain_map(config)?)
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_samples_lossy(
            Samples::F32 {
                data: input,
                stride: 4,
            },
            false,
            linear,
            &cfg,
            &ctx,
            &mut scratch,
        )
    } else {
        let linear = linearize_rgb::<_, _, 3>(input, width, height, &ctx, &mut scratch, &transform);
        let mut ctx = ctx;
        ctx.set_coding(config.coding_transform, distance);
        apply_yellow_opsin(&mut ctx, &linear, distance);
        let cfg = EncodeConfigImpl::with_distance(distance)
            .with_progressive_from(config)
            .with_bits_per_sample(bps)
            .with_icc_profile(config.icc_profile.clone())
            .with_exif(config.exif.clone())
            .with_gain_map(encode_gain_map(config)?)
            .with_xmp(config.xmp.clone())
            .with_brotli_compression(config.brotli_compression.clone())
            .with_orientation(config.orientation)
            .with_color_encoding(config.color_encoding)
            .with_intensity_target(config.intensity_target)
            .with_num_threads(config.num_threads);
        encode_samples_lossy(
            Samples::F32 {
                data: input,
                stride: 3,
            },
            false,
            linear,
            &cfg,
            &ctx,
            &mut scratch,
        )
    }
}

/// Float (f16/f32) grayscale **lossy** encode path. `luma` is `width * height`
/// f32 samples (encoded per `config.color_encoding`, [0, 1]).
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
    let mut ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    ctx.set_coding(config.coding_transform, distance);
    let mut scratch = Box::<CoderScratch>::default();
    let decoder = FloatTransferDecoder::new(config.color_encoding.transfer);
    let linear = linearize_gray(luma, width, height, &ctx, &mut scratch, &decoder);
    let cfg = EncodeConfigImpl::with_distance(distance)
        .with_progressive_from(config)
        .with_grayscale(true)
        .with_bits_per_sample(bps)
        .with_icc_profile(config.icc_profile.clone())
        .with_exif(config.exif.clone())
        .with_gain_map(encode_gain_map(config)?)
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_color_encoding(config.color_encoding)
        .with_intensity_target(config.intensity_target)
        .with_num_threads(config.num_threads);
    encode_samples_lossy(
        Samples::F32 {
            data: luma,
            stride: 1,
        },
        true,
        linear,
        &cfg,
        &ctx,
        &mut scratch,
    )
}

/// Encode a 32-bit float RGB image (lossy). `input` is interleaved `[R, G, B]`,
/// `width * height * 3` samples encoded per `config.color_encoding` in [0, 1].
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
/// `[R, G, B, A]`; RGB uses `config.color_encoding` in [0, 1], A is linear opacity in [0, 1]
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
/// luma samples encoded per `config.color_encoding` in [0, 1].
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
pub(crate) fn progressive_schedule(
    progressive: bool,
    passes: Option<u32>,
    shifts: Option<&[u32]>,
) -> Vec<u32> {
    if let Some(s) = shifts {
        let valid = !s.is_empty()
            && s.len() <= 11
            && *s.last().unwrap() == 0
            && s.iter().all(|&v| v <= 3)
            && s.array_windows::<2>().all(|w| w[0] >= w[1]);
        if valid {
            return s.to_vec();
        }
    }
    let n = passes
        .map(|p| p.clamp(1, 4))
        .unwrap_or(if progressive { 2 } else { 1 }) as usize;
    (0..n).rev().map(|p| p as u32).collect()
}

/// Integer or float samples of one lossy entry point, as the frame codes
/// them under a non-XYB transform.
enum Samples<'a> {
    /// Interleaved integer channels (`stride` per pixel), at most `max`.
    U8 { data: &'a [u8], stride: usize },
    U16 {
        data: &'a [u16],
        stride: usize,
        max: u16,
    },
    /// Interleaved float channels, coded as given.
    F32 { data: &'a [f32], stride: usize },
}

impl Samples<'_> {
    /// Channel `c` of pixel `px` in nominal [0, 1] units.
    #[inline]
    fn get(&self, px: usize, c: usize) -> f32 {
        match *self {
            Samples::U8 { data, stride } => f32::from(data[px * stride + c]) * (1.0 / 255.0),
            Samples::U16 { data, stride, max } => {
                f32::from(data[px * stride + c].min(max)) / f32::from(max)
            }
            Samples::F32 { data, stride } => data[px * stride + c],
        }
    }
}

/// Lossy encode from the input samples. Non-XYB transforms code them as given,
/// in the image's declared color space (gray as three equal channels); `linear`
/// then only steers adaptive quantization.
fn encode_samples_lossy(
    samples: Samples<'_>,
    gray: bool,
    linear: Image3F,
    config: &EncodeConfigImpl,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
) -> Result<Vec<u8>, EncodeError> {
    if ctx.coding.is_xyb() {
        return encode_with_context(linear, config, ctx, scratch);
    }
    let (width, height) = (linear.xsize(), linear.ysize());
    let mut coded = Image3F::try_new(width, height)?;
    for y in 0..height {
        let [p0, p1, p2] = coded.all_plane_rows_mut(y);
        for x in 0..width {
            let px = y * width + x;
            let [r, g, b] = if gray {
                [samples.get(px, 0); 3]
            } else {
                [samples.get(px, 0), samples.get(px, 1), samples.get(px, 2)]
            };
            [p0[x], p1[x], p2[x]] = ctx.coding.code(r, g, b);
        }
    }
    encode_with_coded_planes(linear, Some(coded), config, ctx, scratch)
}

fn encode_with_context(
    input: Image3F,
    config: &EncodeConfigImpl,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
) -> Result<Vec<u8>, EncodeError> {
    encode_with_coded_planes(input, None, config, ctx, scratch)
}

/// `coded` carries the planes of a non-XYB frame when they are not the
/// sRGB rendition of `input`, which then only steers adaptive quantization.
pub(crate) fn encode_with_coded_planes(
    input: Image3F,
    coded: Option<Image3F>,
    config: &EncodeConfigImpl,
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
) -> Result<Vec<u8>, EncodeError> {
    let (width, height) = (input.xsize(), input.ysize());
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }

    if let Some(alpha) = config.alpha.as_ref() {
        let expected = match &alpha {
            AlphaPlane::U8(_) => checked_buffer_size::<u8>(width, height, 1)?,
            AlphaPlane::U16 { .. } => checked_buffer_size::<u16>(width, height, 1)?,
            AlphaPlane::F32(_) => checked_buffer_size::<f32>(width, height, 1)?,
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
    write_size_header(width, height, &mut w);
    let coeff_shifts = progressive_schedule(
        config.progressive,
        config.progressive_passes,
        config.progressive_shifts.as_deref(),
    );
    // The frame first: a lossy modular frame of a non-XYB codestream changes
    // the declared bit depth.
    let mut frame = BitWriter::new();
    let lossy_modular_16 = encode_frame(
        ctx,
        scratch,
        distance,
        input,
        coded,
        config.grayscale,
        config.alpha.as_ref(),
        config.fill_invisible && !config.lossless && !config.black,
        &coeff_shifts,
        config.patches,
        !config.bits_per_sample.is_float(),
        &mut frame,
    )?;
    write_image_metadata(
        &ImageMetadataParams {
            tone_mapping: config.tone_mapping(),
            color_encoding: &config.color_encoding,
            alpha: config.alpha.as_ref(),
            black: config.black,
            icc_profile: config.icc_profile.as_deref(),
            bps: config.bits_per_sample,
            lossless: config.lossless,
            xyb_encoded: !config.lossless && ctx.coding.is_xyb(),
            xyb: &ctx.xyb,
            grayscale: config.grayscale,
            orientation: config.orientation,
            lossy_modular_16,
            extra_16: lossy_modular_16 && ctx.extra_squeeze_distance > 0.0,
            upsampling_kernel: None,
        },
        scratch,
        &mut w,
    );
    w.append(&frame);
    let codestream = w.into_bytes();
    let alpha_bits = config.alpha.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    finalize_container(
        codestream,
        config.exif.as_deref(),
        config.xmp.as_deref(),
        config.brotli_compression.as_deref(),
        needs_level_10(config.bits_per_sample.bits(), config.lossless, alpha_bits) || config.black,
        config.gain_map.as_ref(),
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
pub(crate) fn encode_with_config_loseless<T: AsSignedInt + Copy>(
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
                    *r = src[0].to_signed_int(max_bp);
                    *g = src[1].to_signed_int(max_bp);
                    *b = src[2].to_signed_int(max_bp);
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
                    *r = src[0].to_signed_int(max_bp);
                    *g = src[1].to_signed_int(max_bp);
                    *b = src[2].to_signed_int(max_bp);
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
                *r = src[0].to_signed_int(max_bp);
                *g = src[1].to_signed_int(max_bp);
                *b = src[2].to_signed_int(max_bp);
            }
        }
    }

    if config.near_lossless > 0 {
        let max_value = ((1u32 << max_bp) - 1) as i32;
        crate::lossless::quantize_planes(&mut image3s, max_value, i32::from(config.near_lossless));
    }
    for y in 0..height {
        let [r_row, g_row, b_row] = image3s.all_plane_rows_mut(y);
        for ((r, g), b) in r_row.iter_mut().zip(g_row.iter_mut()).zip(b_row.iter_mut()) {
            (*r, *g, *b) = forward_ycocg(*r, *g, *b);
        }
    }

    encode_lossless_planes(&image3s, &alpha_plane, max_bp, config)
}

fn encode_lossless_planes(
    image3s: &Image3Si,
    alpha_plane: &Option<AlphaPlane>,
    max_bp: u8,
    config: &EncodeConfigImpl,
) -> Result<Vec<u8>, EncodeError> {
    let (width, height) = (image3s.xsize(), image3s.ysize());
    let mut w = BitWriter::new();
    w.write(8, 0xFF);
    w.write(8, CODESTREAM_MARKER as u64);
    write_size_header(width, height, &mut w);
    // An integer-upscaled source (pixel art, a 2x asset) is coded at its
    // cell grid and scaled back by the decoder's nearest-neighbor kernel,
    // exactly. Progressive frames keep their own headers at full size.
    let upsampling = if config.upsampling && !config.progressive {
        crate::upsampling::constant_cell_factor(image3s, alpha_plane.as_ref())
    } else {
        1
    };
    let downsampled = (upsampling > 1)
        .then(|| crate::upsampling::downsample(image3s, alpha_plane.as_ref(), upsampling));
    let kernel = (upsampling > 1).then(|| crate::upsampling::nearest_weights(upsampling));
    let (image3s, alpha_plane) = match &downsampled {
        Some((base, base_alpha)) => (base, base_alpha),
        None => (image3s, alpha_plane),
    };
    let write_metadata = |alpha: Option<&AlphaPlane>, kernel: Option<&[f32]>, w: &mut BitWriter| {
        let mut metadata_scratch = Box::new(CoderScratch::lossless());
        write_image_metadata(
            &ImageMetadataParams {
                tone_mapping: config.tone_mapping(),
                color_encoding: &config.color_encoding,
                alpha,
                black: config.black,
                icc_profile: config.icc_profile.as_deref(),
                bps: config.bits_per_sample,
                lossless: config.lossless,
                xyb_encoded: !config.lossless,
                xyb: &XybMatrix::SPEC,
                grayscale: config.grayscale,
                orientation: config.orientation,
                lossy_modular_16: false,
                extra_16: false,
                upsampling_kernel: kernel,
            },
            &mut metadata_scratch,
            w,
        );
    };
    write_metadata(alpha_plane.as_ref(), kernel.as_deref(), &mut w);
    let alpha_bits = alpha_plane.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    let eff_bits = (max_bp as u32).max(alpha_bits);
    let num_color = if config.grayscale { 1 } else { 3 };
    encode_frame_lossless(
        image3s,
        alpha_plane.as_ref(),
        eff_bits,
        config.progressive,
        config.patches,
        num_color,
        config.speed,
        config.decoding_speed,
        upsampling,
        config.near_lossless > 0,
        config.num_threads,
        &mut w,
    );
    // The integer-upscaled part of a mixed image as a cell layer (see
    // `cell_layer`): the base with the covered cells filled, saved to slot
    // 0, then the cells coded at their grid and blended over it with their
    // coverage as alpha; the output gains an opaque alpha channel, so an
    // input whose alpha is solid opaque qualifies as well. The complete
    // stream competes with the plain encode on size.
    let opaque =
        |alpha: &AlphaPlane| matches!(alpha, AlphaPlane::U8(a) if a.iter().all(|&v| v == 255));
    let cell_plan = (config.upsampling
        && upsampling == 1
        && alpha_plane.as_ref().is_none_or(opaque)
        && max_bp == 8
        && !config.grayscale
        && !config.progressive)
        .then(|| crate::cell_layer::plan(image3s))
        .flatten();
    if let Some(plan) = cell_plan {
        let (base, base_alpha) = crate::cell_layer::base(image3s, &plan);
        let (layer, layer_alpha) = crate::cell_layer::layer(image3s, &plan);
        let layer_kernel = crate::upsampling::nearest_weights(plan.factor as u32);
        let mut layered = BitWriter::new();
        layered.write(8, 0xFF);
        layered.write(8, CODESTREAM_MARKER as u64);
        write_size_header(width, height, &mut layered);
        write_metadata(Some(&base_alpha), Some(&layer_kernel), &mut layered);
        let pool = crate::thread_pool::ThreadPool::new_lossless(config.num_threads);
        let mut scratch = Box::new(CoderScratch::lossless());
        crate::lossless::encode_frame_lossless_base(
            &base,
            &base_alpha,
            config.patches,
            config.speed,
            config.decoding_speed,
            config.near_lossless > 0,
            &pool,
            &mut scratch,
            &mut layered,
        );
        crate::lossless::encode_frame_lossless_core(
            &layer,
            Some(&layer_alpha),
            8,
            false,
            3,
            config.speed,
            config.decoding_speed,
            plan.factor as u32,
            config.near_lossless > 0,
            &pool,
            &mut scratch,
            crate::patches::ModularFrameKind::RgbBlendCrop {
                x0: plan.x0,
                y0: plan.y0,
                width: plan.width,
                height: plan.height,
            },
            true,
            &mut layered,
        );
        if layered.bits_written() < w.bits_written() {
            w = layered;
        }
    }
    let codestream = w.into_bytes();
    let alpha_bits = alpha_plane.as_ref().map(|a| a.bits() as u32).unwrap_or(0);
    finalize_container(
        codestream,
        config.exif.as_deref(),
        config.xmp.as_deref(),
        config.brotli_compression.as_deref(),
        needs_level_10(max_bp as u32, true, alpha_bits) || config.black,
        config.gain_map.as_ref(),
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
/// `level` via a `jxll` box. Box order: signature, ftyp, jxll, jxlc, then the
/// optional `jhgm` gain map bundle and metadata boxes.
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
    gain_map: Option<&[u8]>,
) -> Result<Vec<u8>, EncodeError> {
    let exif_extra = exif.map(|e| e.len() + 12).unwrap_or(0);
    let xmp_extra = xmp.map(|x| x.len() + 8).unwrap_or(0);
    let gain_map_extra = gain_map.map(|g| g.len() + 8).unwrap_or(0);
    let mut out =
        Vec::with_capacity(codestream.len() + 41 + exif_extra + xmp_extra + gain_map_extra);
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
    // Gain map bundle (`jhgm`), right after the codestream it belongs to.
    if let Some(g) = gain_map {
        push_container_box(&mut out, b"jhgm", g);
    }
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
/// when an EXIF/XMP/gain-map box must be carried (a bare codestream cannot
/// hold them).
fn finalize_container(
    codestream: Vec<u8>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    compressor: Option<&dyn BrotliCompression>,
    need_l10: bool,
    gain_map: Option<&EncodedGainMap>,
) -> Result<Vec<u8>, EncodeError> {
    // A level-10 gain map codestream raises the whole file's level.
    let need_l10 = need_l10 || gain_map.is_some_and(|g| g.needs_level_10);
    if need_l10 || exif.is_some() || xmp.is_some() || gain_map.is_some() {
        wrap_jxl_container(
            codestream,
            if need_l10 { 10 } else { 5 },
            exif,
            xmp,
            compressor,
            gain_map.map(|g| g.bundle.as_slice()),
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

/// The image-level fields `write_image_metadata` serializes.
#[derive(Clone, Copy)]
struct ImageMetadataParams<'a> {
    tone_mapping: ToneMappingParams,
    color_encoding: &'a ColorEncoding,
    /// The single extra channel: alpha, or K when `black`.
    alpha: Option<&'a AlphaPlane>,
    black: bool,
    icc_profile: Option<&'a [u8]>,
    bps: BitsPerSample,
    lossless: bool,
    xyb_encoded: bool,
    /// Opsin matrix of an XYB codestream; a non-default one is signaled.
    xyb: &'a XybMatrix,
    grayscale: bool,
    orientation: Orientation,
    /// The frame is a lossy modular frame on the 16-bit lattice: the image
    /// declares 16 bits whatever the input depth was.
    lossy_modular_16: bool,
    /// The extra channel (K, or lossy alpha) sits on that lattice too.
    extra_16: bool,
    /// Frames are coded at the image size over 2, 4 or 8 and scaled back
    /// with this kernel (15, 55 or 210 weights), signaled here.
    upsampling_kernel: Option<&'a [f32]>,
}

fn write_image_metadata(
    params: &ImageMetadataParams<'_>,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    let ImageMetadataParams {
        tone_mapping: tm,
        color_encoding,
        alpha,
        black,
        icc_profile,
        bps,
        lossless,
        xyb_encoded,
        xyb,
        grayscale,
        orientation,
        lossy_modular_16,
        extra_16,
        upsampling_kernel,
    } = *params;
    let bps = if lossy_modular_16 {
        BitsPerSample::Sixteen
    } else {
        bps
    };
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
    let needs_32 = (lossless && bps.bits() >= 16) || alpha_bits >= 16 || lossy_modular_16;
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
            bits if black => {
                // kBlack: no all-default form (that one is 8-bit alpha) and
                // no type-specific fields. A lossy modular frame codes K on
                // the 16-bit lattice too.
                w.write(1, 0); // all_default = false
                w.write(2, 2); // ec_type: BitsOffset(4, 2) selector
                w.write(4, 4 - 2); // ec_type = Black (4)
                write_int_bit_depth(if extra_16 { 16 } else { bits as u32 }, w);
                w.write(2, 0); // dim_shift = 0
                w.write(2, 0); // name length = 0
            }
            8 if !extra_16 => {
                w.write(1, 1); // all_default = true → U8 alpha
            }
            bits => {
                w.write(1, 0); // all_default = false
                w.write(2, 0); // ec_type = Alpha (selector 0)
                if alpha.is_float() {
                    write_float_bit_depth(32, 8, w); // f32 alpha bit depth
                } else {
                    write_int_bit_depth(if extra_16 { 16 } else { bits as u32 }, w);
                }
                w.write(2, 0); // dim_shift = 0
                w.write(2, 0); // name length = 0
                w.write(1, 0); // alpha_associated = false
            }
        }
    } else {
        w.write(2, 0); // num_extra_channels = 0
    }

    // Lossy frames are XYB unless a non-XYB coding transform was chosen;
    // lossless modular carries un-transformed pixel values.
    w.write(1, u64::from(xyb_encoded)); // xyb_encoded
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
    let explicit_opsin = xyb_encoded && !xyb.is_decoder_default();
    let upsampling = upsampling_kernel.map_or(1, |kernel| match kernel.len() {
        15 => 2,
        55 => 4,
        _ => 8,
    });
    if !explicit_opsin && upsampling == 1 {
        w.write(1, 1); // CustomTransformData.all_default = 1
    } else {
        // A non-spec forward opsin matrix needs its matching inverse in the
        // codestream, and an upsampled frame its kernel, so the bundle is
        // explicit. Layout: all_default, [xyb_encoded] OpsinInverseMatrix,
        // custom_weights_mask, then the weights of each set mask bit.
        w.write(1, 0); // CustomTransformData.all_default = 0
        if xyb_encoded && !explicit_opsin {
            w.write(1, 1); // OpsinInverseMatrix.all_default = 1
        } else if xyb_encoded {
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
            // overshoot past 1.0 was noise. The encoder's dequantization model
            // follows via `DequantBias::channels`.
            for v in crate::group::EXPLICIT_QUANT_BIASES {
                w.write(16, crate::util::f32_to_f16_bits(v) as u64);
            }
        }
        if upsampling == 1 {
            w.write(3, 0); // custom_weights_mask = 0 (default upsampling kernels)
        } else {
            // Bit 0, 1, 2 of the mask select the 2x, 4x and 8x kernel; the
            // frame header's `upsampling` picks which one the decoder runs.
            w.write(3, (upsampling >> 1) as u64);
            for &weight in upsampling_kernel.unwrap_or_default() {
                w.write(16, crate::util::f32_to_f16_bits(weight) as u64);
            }
        }
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
    fn slow_encodes_short_wide_strips() {
        for (width, height) in [(1598usize, 9usize), (1598, 1), (9, 1598), (777, 15)] {
            let mut pixels = Vec::with_capacity(width * height * 3);
            for y in 0..height {
                for x in 0..width {
                    let v = ((x * 37 + y * 101) ^ (x * y)) as u8;
                    pixels.extend_from_slice(&[v, v.wrapping_mul(3), 255 - v]);
                }
            }
            for distance in [0.5f32, 1.0, 3.0] {
                let config = EncodeConfig::default()
                    .with_distance(distance)
                    .with_speed(Speed::Slow);
                let bytes = encode_image(&pixels, width, height, &config)
                    .unwrap_or_else(|e| panic!("{width}x{height} d={distance}: {e:?}"));
                assert!(!bytes.is_empty());
            }
        }
    }

    #[test]
    fn default_speed_is_fast() {
        assert_eq!(Speed::default(), Speed::Fast);
        assert_eq!(EncodeConfig::default().speed, Speed::Fast);
    }

    #[test]
    fn invisible_fill_defaults_on_and_can_be_disabled() {
        assert!(EncodeConfig::default().fill_invisible);
        let config = EncodeConfig::default().with_fill_invisible(false);
        assert!(!config.fill_invisible);
        assert!(
            !EncodeConfigImpl::default()
                .with_progressive_from(&config)
                .fill_invisible
        );
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

    #[test]
    fn palette_ink_is_deterministic_across_workers_and_preserves_plain_fallback() {
        let (width, height) = (1100, 130);
        let rgb: Vec<_> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    if (8..122).contains(&y) && (x % 40 < 2) {
                        [16; 3]
                    } else if y % 31 < 2 {
                        [150, 20, 40]
                    } else {
                        [240; 3]
                    }
                })
            })
            .collect();
        let config = EncodeConfig::default()
            .with_distance(1.5)
            .with_ink_layer(true)
            .with_patches(false);
        let single =
            encode_image(&rgb, width, height, &config.clone().with_num_threads(1)).unwrap();
        let parallel =
            encode_image(&rgb, width, height, &config.clone().with_num_threads(4)).unwrap();
        assert_eq!(single, parallel);
        let flat = vec![200; 64 * 64 * 3];
        assert_eq!(
            encode_image(&flat, 64, 64, &config).unwrap(),
            encode_image(&flat, 64, 64, &config.with_ink_layer(false)).unwrap()
        );
    }

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

    fn box_kinds(container: &[u8]) -> Vec<[u8; 4]> {
        let mut kinds = Vec::new();
        let mut pos = 0usize;
        while pos + 8 <= container.len() {
            let size = u32::from_be_bytes(container[pos..pos + 4].try_into().unwrap()) as usize;
            kinds.push(container[pos + 4..pos + 8].try_into().unwrap());
            assert!(size >= 8, "large boxes not expected in these tests");
            pos += size;
        }
        assert_eq!(pos, container.len());
        kinds
    }

    fn gain_map_metadata() -> crate::IsoGainMap {
        crate::IsoGainMap::from_floats(&crate::GainMapFloats {
            max: [2.0; 3],
            alternate_hdr_headroom: 2.0,
            ..crate::GainMapFloats::default()
        })
        .unwrap()
    }

    fn gray_gain_map(w: usize, h: usize) -> crate::GainMap {
        let gain: Vec<u8> = (0..w * h).map(|i| (i * 4) as u8).collect();
        crate::GainMap::gray8(gain, w, h, gain_map_metadata())
    }

    #[test]
    fn gain_map_is_written_to_a_jhgm_box_after_the_codestream() {
        let md = gain_map_metadata();
        let gm = gray_gain_map(8, 8)
            .with_lossless(true)
            .with_alternate_color_encoding(ColorEncoding::bt2020_pq());
        let encoded = encode_image(&rgb8(), W, H, &lossy().with_gain_map(gm)).unwrap();

        assert_eq!(
            box_kinds(&encoded),
            [*b"JXL ", *b"ftyp", *b"jxll", *b"jxlc", *b"jhgm"]
        );
        assert_eq!(box_payload(&encoded, b"jxll"), Some(&[5u8][..]));
        let jhgm = box_payload(&encoded, b"jhgm").expect("jhgm box");
        let p = crate::gain_map::parse_jhgm_bundle(jhgm).unwrap();
        assert_eq!(p.version, 0);
        assert_eq!(crate::IsoGainMap::from_metadata(p.metadata).unwrap(), md);
        assert!(!p.color_encoding.is_empty());
        assert!(p.alt_icc.is_empty());
        assert_eq!(&p.codestream[..2], &[0xFF, 0x0A]);
        // A naked codestream: no container signature inside the bundle.
        assert_ne!(&p.codestream[..4], &[0, 0, 0, 0x0C]);
    }

    #[test]
    fn gain_map_box_precedes_metadata_boxes() {
        let exif = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
        let encoded = encode_image(
            &rgb8(),
            W,
            H,
            &lossy()
                .with_exif(exif.clone())
                .with_xmp(b"<x/>".to_vec())
                .with_gain_map(gray_gain_map(4, 4)),
        )
        .unwrap();
        assert_eq!(
            box_kinds(&encoded),
            [
                *b"JXL ", *b"ftyp", *b"jxll", *b"jxlc", *b"jhgm", *b"Exif", *b"xml "
            ]
        );
        assert_eq!(box_payload(&encoded, b"xml "), Some(&b"<x/>"[..]));
    }

    #[test]
    fn level_10_gain_map_raises_the_file_level() {
        let gain: Vec<u16> = (0..4 * 4).map(|i| (i * 4000) as u16).collect();
        let gm = crate::GainMap::gray16(gain, 16, 4, 4, gain_map_metadata()).with_lossless(true);
        let encoded = encode_image(&rgb8(), W, H, &lossy().with_gain_map(gm)).unwrap();
        assert_eq!(box_payload(&encoded, b"jxll"), Some(&[10u8][..]));
        let jhgm = box_payload(&encoded, b"jhgm").unwrap();
        let p = crate::gain_map::parse_jhgm_bundle(jhgm).unwrap();
        assert_eq!(&p.codestream[..2], &[0xFF, 0x0A]);
    }

    #[test]
    fn gain_map_rides_along_every_primary_encoder_path() {
        let cfg = lossy().with_gain_map(
            crate::GainMap::rgb8(
                (0..4 * 4 * 3).map(|i| i as u8).collect(),
                4,
                4,
                gain_map_metadata(),
            )
            .with_distance(3.0),
        );
        let rgba: Vec<u8> = (0..W * H * 4).map(|i| i as u8).collect();
        let gray: Vec<u8> = (0..W * H).map(|i| i as u8).collect();
        let rgb16: Vec<u16> = (0..W * H * 3).map(|i| (i * 7) as u16).collect();
        let f32s: Vec<f32> = (0..W * H * 3).map(|i| (i % 255) as f32 / 255.0).collect();
        let outputs = [
            encode_image_with_alpha(&rgba, W, H, &cfg).unwrap(),
            encode_image_gray(&gray, W, H, &cfg).unwrap(),
            encode_image_16bit(&rgb16, W, H, &cfg).unwrap(),
            encode_image_f32(&f32s, W, H, &cfg).unwrap(),
            encode_image(
                &rgb8(),
                W,
                H,
                &lossless().with_gain_map(gray_gain_map(4, 4)),
            )
            .unwrap(),
            encode_image_16bit(&rgb16, W, H, &lossless().with_gain_map(gray_gain_map(4, 4)))
                .unwrap(),
            encode_image_f32(&f32s, W, H, &lossless().with_gain_map(gray_gain_map(4, 4))).unwrap(),
        ];
        for out in &outputs {
            let jhgm = box_payload(out, b"jhgm").expect("every path writes the jhgm box");
            let p = crate::gain_map::parse_jhgm_bundle(jhgm).unwrap();
            assert_eq!(&p.codestream[..2], &[0xFF, 0x0A]);
        }
    }

    #[test]
    fn invalid_gain_map_fails_the_encode() {
        let gm = crate::GainMap::gray8(vec![0; 15], 4, 4, gain_map_metadata());
        assert!(matches!(
            encode_image(&rgb8(), W, H, &lossy().with_gain_map(gm)),
            Err(EncodeError::GainMapSizeMismatch {
                expected: 16,
                actual: 15
            })
        ));
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
    fn progressive_dc_only_where_supported() {
        // 160x96 has a 20x12 DC image, large enough to squeeze.
        let (w, h) = (160, 96);
        let rgb: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [(x + y) as u8, (2 * x) as u8, ((x * y) % 251) as u8]
            })
            .collect();
        let rgba: Vec<u8> = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 200])
            .collect();
        for speed in [Speed::Fast, Speed::Slow] {
            let off = lossy().with_speed(speed).with_num_threads(1);
            let on = off.clone().with_progressive_dc(true);
            let full = encode_image(&rgb, w, h, &on).unwrap();
            assert_ne!(encode_image(&rgb, w, h, &off).unwrap(), full);
            // Level counts clamp to the 1..=2 this DC image allows.
            let levels = |n| encode_image(&rgb, w, h, &on.clone().with_progressive_dc_levels(n));
            assert_eq!(levels(99).unwrap(), full);
            assert_eq!(levels(0).unwrap(), levels(1).unwrap());
            assert_ne!(levels(1).unwrap(), full);
            // A 2x2 DC image cannot be squeezed; alpha frames keep their DC.
            assert_eq!(
                encode_image(&rgb8(), W, H, &off).unwrap(),
                encode_image(&rgb8(), W, H, &on).unwrap()
            );
            assert_eq!(
                encode_image_with_alpha(&rgba, w, h, &off).unwrap(),
                encode_image_with_alpha(&rgba, w, h, &on).unwrap()
            );
        }
    }

    #[test]
    fn invisible_fill_preserves_lossless_and_nontransparent_inputs() {
        let on = lossy().with_num_threads(1);
        let off = on.clone().with_fill_invisible(false);
        assert_eq!(
            encode_image(&rgb8(), W, H, &on).unwrap(),
            encode_image(&rgb8(), W, H, &off).unwrap()
        );
        let mut rgba = rgba8();
        for px in rgba.as_chunks_mut::<4>().0 {
            px[3] = 128; // Partial alpha must not be premultiplied or filled.
        }
        for coding in [
            CodingTransform::Xyb,
            CodingTransform::Rgb,
            CodingTransform::YCbCr,
        ] {
            let on = on.clone().with_coding_transform(coding);
            let off = off.clone().with_coding_transform(coding);
            assert_eq!(
                encode_image_with_alpha(&rgba, W, H, &on).unwrap(),
                encode_image_with_alpha(&rgba, W, H, &off).unwrap()
            );
        }
        let lossless_on = on.with_lossless(true);
        let lossless_off = off.with_lossless(true);
        assert_eq!(
            encode_image_with_alpha(&rgba8(), W, H, &lossless_on).unwrap(),
            encode_image_with_alpha(&rgba8(), W, H, &lossless_off).unwrap()
        );
    }

    #[test]
    fn public_lossy_paths_share_the_minimum_distance() {
        for speed in [Speed::Fast, Speed::Medium, Speed::Slow] {
            let at_floor = EncodeConfig::default()
                .with_speed(speed)
                .with_distance(0.05);
            for distance in [0.001, 0.03, 0.049] {
                let below = at_floor.clone().with_distance(distance);
                macro_rules! same_output {
                    ($encode:ident, $pixels:expr) => {
                        assert_eq!(
                            $encode(&$pixels, W, H, &below).unwrap(),
                            $encode(&$pixels, W, H, &at_floor).unwrap(),
                            "{}: speed={speed:?} d={distance}",
                            stringify!($encode)
                        );
                    };
                }
                same_output!(encode_image, rgb8());
                same_output!(encode_image_with_alpha, rgba8());
                same_output!(encode_image_16bit, rgb16());
                same_output!(encode_image_gray, gray8());
                same_output!(encode_image_gray_16bit, gray16());
                same_output!(encode_image_f32, rgb_f32());
                same_output!(encode_image_gray_f32, gray_f32());
            }
        }
    }

    /// Thin anti-aliased dark curves on a smooth gradient: the content class
    /// splines exist for.
    #[cfg(feature = "splines")]
    fn line_art(size: usize) -> Vec<u8> {
        let mut px = vec![0u8; size * size * 3];
        for y in 0..size {
            for x in 0..size {
                let mut v = 150.0 + 60.0 * (x + y) as f32 / (2 * size) as f32;
                for k in 0..6 {
                    let phase = k as f32 * 0.9;
                    let cy = size as f32 * (0.15 + 0.13 * k as f32)
                        + 18.0 * (x as f32 * 0.035 + phase).sin();
                    let d = (y as f32 - cy).abs();
                    v -= 120.0 * (-0.5 * (d / 0.8).powi(2)).exp();
                }
                let i = (y * size + x) * 3;
                px[i] = v.clamp(0.0, 255.0) as u8;
                px[i + 1] = (v * 0.92).clamp(0.0, 255.0) as u8;
                px[i + 2] = (v * 0.85).clamp(0.0, 255.0) as u8;
            }
        }
        px
    }

    #[cfg(feature = "splines")]
    fn slow_lossy(distance: f32) -> EncodeConfig {
        EncodeConfig::default()
            .with_distance(distance)
            .with_speed(Speed::Slow)
    }

    #[test]
    #[cfg(feature = "splines")]
    fn splines_reduce_thin_line_art() {
        const S: usize = 256;
        let pixels = line_art(S);
        let plain = encode_image(&pixels, S, S, &slow_lossy(3.0)).unwrap();
        let with = encode_image(&pixels, S, S, &slow_lossy(3.0).with_splines(true)).unwrap();
        assert!(
            with.len() * 100 < plain.len() * 90,
            "splines should win clearly on thin line art: {} vs {}",
            with.len(),
            plain.len()
        );
    }

    #[test]
    #[cfg(feature = "splines")]
    fn splines_join_crossed_grid_lines() {
        const S: usize = 256;
        let mut pixels = Vec::with_capacity(S * S * 3);
        for y in 0..S {
            for x in 0..S {
                let mut v = 170.0 + 70.0 * (x + y) as f32 / (2 * S) as f32;
                if (12..S - 12).contains(&x) && (12..S - 12).contains(&y) {
                    for axis in [x, y] {
                        let d = ((axis - 12 + 18) % 36) as f32 - 18.0;
                        v -= 65.0 * (-0.5 * (d / 0.65).powi(2)).exp();
                    }
                }
                pixels.extend([v as u8, (v * 0.95) as u8, (v * 0.9) as u8]);
            }
        }
        let config = slow_lossy(2.0)
            .with_patches(false)
            .with_ink_layer(false)
            .with_num_threads(1);
        let plain = encode_image(&pixels, S, S, &config).unwrap();
        let config = config.with_splines(true);
        let joined = encode_image(&pixels, S, S, &config).unwrap();
        assert!(
            joined.len() * 100 < plain.len() * 80,
            "whole grid lines should save more than isolated fragments: {} vs {}",
            joined.len(),
            plain.len()
        );
        assert_eq!(
            joined,
            encode_image(&pixels, S, S, &config.with_num_threads(4)).unwrap()
        );
    }

    #[test]
    #[cfg(feature = "splines")]
    fn splines_are_inert_without_lines() {
        const S: usize = 128;
        let mut pixels = vec![0u8; S * S * 3];
        for y in 0..S {
            for x in 0..S {
                let i = (y * S + x) * 3;
                pixels[i] = (x * 2) as u8;
                pixels[i + 1] = (y * 2) as u8;
                pixels[i + 2] = (x + y) as u8;
            }
        }
        let plain = encode_image(&pixels, S, S, &slow_lossy(2.0)).unwrap();
        let with = encode_image(&pixels, S, S, &slow_lossy(2.0).with_splines(true)).unwrap();
        assert_eq!(plain, with);
    }

    /// Gaussian point sources on a dark, smooth sky: the content class dots
    /// exist for.
    #[cfg(feature = "splines")]
    fn star_field(size: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32
        };
        let mut sky = vec![[0f32; 3]; size * size];
        for (i, p) in sky.iter_mut().enumerate() {
            let (x, y) = ((i % size) as f32, (i / size) as f32);
            *p = [8.0 + 0.03 * x, 10.0 + 0.02 * y, 22.0 + 0.04 * (x + y)];
        }
        for _ in 0..size * size / 120 {
            let (cx, cy) = (
                6.0 + next() * (size - 12) as f32,
                6.0 + next() * (size - 12) as f32,
            );
            let amp = 40.0 + 200.0 * next() * next();
            let sigma = 0.6 + 0.4 * next();
            let tint = [0.8 + 0.2 * next(), 0.9, 1.0];
            for y in cy as usize - 4..=cy as usize + 4 {
                for x in cx as usize - 4..=cx as usize + 4 {
                    let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                    let v = amp * (-0.5 * d2 / (sigma * sigma)).exp();
                    for (c, t) in sky[y * size + x].iter_mut().zip(tint) {
                        *c += v * t;
                    }
                }
            }
        }
        sky.iter()
            .flat_map(|p| p.map(|c| c.clamp(0.0, 255.0) as u8))
            .collect()
    }

    #[test]
    #[cfg(feature = "splines")]
    fn dots_shrink_star_fields() {
        const S: usize = 256;
        let pixels = star_field(S);
        let config = slow_lossy(3.0).with_num_threads(1);
        let plain = encode_image(&pixels, S, S, &config).unwrap();
        let config = config.with_dots(true);
        let with = encode_image(&pixels, S, S, &config).unwrap();
        // The synthetic field's margin sits at ~5%; sparser fields flip on
        // how the nearly flat sky blocks quantize.
        assert!(
            with.len() < plain.len(),
            "dots should win on a star field: {} vs {}",
            with.len(),
            plain.len()
        );
        assert_eq!(
            with,
            encode_image(&pixels, S, S, &config.with_num_threads(4)).unwrap()
        );
    }

    #[test]
    #[cfg(feature = "splines")]
    fn dots_shrink_dark_spot_fields_and_are_thread_deterministic() {
        const S: usize = 256;
        let pixels: Vec<_> = star_field(S).into_iter().map(|v| 255 - v).collect();
        let config = slow_lossy(2.0).with_patches(false).with_num_threads(1);
        let plain = encode_image(&pixels, S, S, &config).unwrap();
        let config = config.with_dots(true);
        let with = encode_image(&pixels, S, S, &config).unwrap();
        assert!(
            with.len() < plain.len(),
            "dark spots: {} vs {}",
            with.len(),
            plain.len()
        );
        assert_eq!(
            with,
            encode_image(&pixels, S, S, &config.with_num_threads(4)).unwrap()
        );
    }

    #[test]
    #[cfg(feature = "splines")]
    fn dots_are_inert_without_points() {
        const S: usize = 128;
        let pixels = line_art(S);
        let config = slow_lossy(2.0).with_splines(true);
        assert_eq!(
            encode_image(&pixels, S, S, &config).unwrap(),
            encode_image(&pixels, S, S, &config.clone().with_dots(true)).unwrap()
        );
    }

    #[test]
    #[cfg(feature = "splines")]
    fn dots_with_alpha_are_thread_deterministic() {
        const S: usize = 256;
        let rgba: Vec<_> = star_field(S)
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, p)| [p[0], p[1], p[2], (64 + i % 192) as u8])
            .collect();
        let config = slow_lossy(3.0).with_num_threads(1).with_dots(true);
        let with = encode_image_with_alpha(&rgba, S, S, &config).unwrap();
        assert_eq!(
            with,
            encode_image_with_alpha(&rgba, S, S, &config.clone().with_num_threads(4)).unwrap()
        );
        assert_ne!(
            with,
            encode_image_with_alpha(&rgba, S, S, &config.with_dots(false)).unwrap()
        );
    }

    #[test]
    #[cfg(feature = "splines")]
    fn dots_only_run_at_slow_speed_and_default_decoding_speed() {
        const S: usize = 128;
        let pixels = star_field(S);
        for config in [
            EncodeConfig::default()
                .with_distance(3.0)
                .with_speed(Speed::Fast),
            slow_lossy(3.0).with_decoding_speed(DecodingSpeed::Fast),
        ] {
            assert_eq!(
                encode_image(&pixels, S, S, &config).unwrap(),
                encode_image(&pixels, S, S, &config.clone().with_dots(true)).unwrap()
            );
        }
    }

    #[test]
    #[cfg(feature = "splines")]
    fn splines_only_run_at_slow_speed_and_default_decoding_speed() {
        const S: usize = 256;
        let pixels = line_art(S);
        let fast = EncodeConfig::default()
            .with_distance(3.0)
            .with_speed(Speed::Fast);
        assert_eq!(
            encode_image(&pixels, S, S, &fast).unwrap(),
            encode_image(&pixels, S, S, &fast.clone().with_splines(true)).unwrap()
        );
        let quick_decode = slow_lossy(3.0).with_decoding_speed(DecodingSpeed::Fast);
        assert_eq!(
            encode_image(&pixels, S, S, &quick_decode).unwrap(),
            encode_image(&pixels, S, S, &quick_decode.clone().with_splines(true)).unwrap()
        );
    }

    #[test]
    #[cfg(feature = "splines")]
    fn splines_coexist_with_patches() {
        const S: usize = 256;
        let pixels = line_art(S);
        let both = slow_lossy(3.0).with_splines(true).with_patches(true);
        let plain = encode_image(&pixels, S, S, &slow_lossy(3.0).with_patches(true)).unwrap();
        let with = encode_image(&pixels, S, S, &both).unwrap();
        assert!(with.len() < plain.len());
    }

    #[test]
    #[cfg(feature = "splines")]
    fn splines_cross_repeated_tiles() {
        // Graph paper with a curve. The grid period does not divide the tile
        // size, so tiles repeat only here and there and every line crosses
        // some; the noise band keeps the glyph detector away.
        const S: usize = 256;
        const PERIOD: f32 = 12.3;
        let rule = |t: usize| {
            let d = ((t as f32 + PERIOD / 2.0) % PERIOD - PERIOD / 2.0).abs();
            (1.0 - d).max(0.0)
        };
        let mut pixels = vec![0u8; S * S * 3];
        let mut state = 0x1234_5678u32;
        for y in 0..S {
            for x in 0..S {
                let mut v = 236.0 - 46.0 * rule(x).max(rule(y));
                let cy = 48.0 + 30.0 * (x as f32 * 0.021).sin();
                v -= 170.0 * (-0.5 * ((y as f32 - cy) / 0.9).powi(2)).exp();
                if y >= 96 {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    v = (state >> 24) as f32;
                }
                let i = (y * S + x) * 3;
                pixels[i] = (v * 0.35).clamp(0.0, 255.0) as u8;
                pixels[i + 1] = (v * 0.45).clamp(0.0, 255.0) as u8;
                pixels[i + 2] = v.clamp(0.0, 255.0) as u8;
            }
        }
        let patched = encode_image(&pixels, S, S, &slow_lossy(2.0).with_patches(true)).unwrap();
        let both = slow_lossy(2.0).with_splines(true).with_patches(true);
        let with = encode_image(&pixels, S, S, &both).unwrap();
        assert!(
            with.len() < patched.len(),
            "splines must survive next to tile patches: {} >= {}",
            with.len(),
            patched.len()
        );
    }

    #[test]
    fn lossy_patches_are_rate_safe() {
        let plain = encode_image(&rgb8(), W, H, &lossy()).unwrap();
        let checked = encode_image(&rgb8(), W, H, &lossy().with_patches(true)).unwrap();
        assert!(checked.len() <= plain.len());
    }

    #[test]
    fn fast_lossy_patches_are_rate_safe_on_unrepeated_ink() {
        // Saturated line drawings in separate 60-pixel tiles, each broken
        // by its own gaps: every tile is a singleton glyph, so the whole
        // image lands in the atlas.
        const SIZE: usize = 256;
        let mut pixels = vec![0u8; SIZE * SIZE * 3];
        for y in 0..SIZE {
            for x in 0..SIZE {
                if (x % 8 == 3 || y % 8 == 3) && x % 64 < 60 && y % 64 < 60 {
                    let blue = if (x / 64 + y / 64) % 2 == 0 { 254 } else { 0 };
                    pixels[(y * SIZE + x) * 3..][..3].copy_from_slice(&[252, 0, blue]);
                }
            }
        }
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as usize
        };
        for _ in 0..60 {
            let (x0, y0) = (next() % (SIZE - 8), next() % (SIZE - 8));
            let (w, h) = (2 + next() % 6, 1 + next() % 5);
            for y in y0..y0 + h {
                pixels[(y * SIZE + x0) * 3..][..w * 3].fill(0);
            }
        }
        for distance in [1.0f32, 3.0] {
            let plain = EncodeConfig::default().with_distance(distance);
            let patched = plain.clone().with_patches(true);
            let plain = encode_image(&pixels, SIZE, SIZE, &plain).unwrap();
            let patched = encode_image(&pixels, SIZE, SIZE, &patched).unwrap();
            assert!(
                patched.len() * 10 <= plain.len() * 11,
                "d={distance}: {} vs {}",
                patched.len(),
                plain.len()
            );
        }
    }

    /// Sixty dark 5x7 glyphs on white, set as lines of text off the block
    /// grid.
    fn glyph_text_pixels(size: usize) -> Vec<u8> {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as usize
        };
        let glyphs: Vec<[[bool; 5]; 7]> = (0..60)
            .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| next() % 3 < 2)))
            .collect();
        let mut pixels = vec![255u8; size * size * 3];
        for line in 0..size / 11 {
            for column in 0..size / 7 - 1 {
                let glyph = &glyphs[next() % glyphs.len()];
                let (x0, y0) = (3 + column * 7 + line % 3, 2 + line * 11);
                for (dy, row) in glyph.iter().enumerate() {
                    for (dx, &ink) in row.iter().enumerate() {
                        if ink {
                            pixels[((y0 + dy) * size + x0 + dx) * 3..][..3].fill(20);
                        }
                    }
                }
            }
        }
        pixels
    }

    #[test]
    fn fast_lossy_glyph_patches_shrink_repeated_text() {
        // The glyph atlas is too large to pay off unless it is coded with
        // a learned tree and palette, which Fast now uses too.
        const SIZE: usize = 256;
        let pixels = glyph_text_pixels(SIZE);
        let plain = EncodeConfig::default().with_distance(1.0);
        let patched = plain.clone().with_patches(true);
        let plain = encode_image(&pixels, SIZE, SIZE, &plain).unwrap();
        let patched = encode_image(&pixels, SIZE, SIZE, &patched).unwrap();
        assert!(
            patched.len() * 5 < plain.len(),
            "{} vs {}",
            patched.len(),
            plain.len()
        );
    }

    /// Noise with one 16x16 noise tile repeated on half of the grid.
    fn repeated_region_pixels() -> Vec<u8> {
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
        pixels
    }

    #[test]
    fn lossy_patches_reduce_repeated_regions() {
        const PW: usize = 256;
        const PH: usize = 256;
        let pixels = repeated_region_pixels();
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

    /// A 4-pixel red checkerboard under a slow brightness drift.
    fn checkerboard_rgb(size: usize) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(size * size * 3);
        for y in 0..size {
            for x in 0..size {
                let lit = (x / 4 + y / 4) % 2 == 1;
                let drift = (x * 24 / size + y * 12 / size) as u8;
                pixels.extend_from_slice(&if lit {
                    [210 - drift, 8, 6]
                } else {
                    [40 - drift.min(40), 0, 0]
                });
            }
        }
        pixels
    }

    fn learned_rate_lossy(distance: f32) -> EncodeConfig {
        EncodeConfig::default()
            .with_distance(distance)
            .with_speed(Speed::Slow)
            .with_patches(false)
            .with_ink_layer(false)
            .with_vector_layers(false)
            .with_learned_rate(true)
    }

    #[test]
    fn learned_rate_encodes_images_smaller_than_a_region() {
        for size in [8usize, 40, 63, 100] {
            let pixels = checkerboard_rgb(size);
            for distance in [1.0f32, 3.0] {
                let bytes = encode_image(&pixels, size, size, &learned_rate_lossy(distance))
                    .expect("encode failed");
                assert!(!bytes.is_empty(), "{size} at d={distance}");
            }
        }
    }

    #[test]
    fn learned_rate_shrinks_a_block_periodic_pattern() {
        const SIZE: usize = 256;
        let pixels = checkerboard_rgb(SIZE);
        for distance in [2.0f32, 3.0] {
            let learned = learned_rate_lossy(distance).with_num_threads(1);
            let fixed = learned.clone().with_learned_rate(false);
            let learned = encode_image(&pixels, SIZE, SIZE, &learned).unwrap();
            let fixed = encode_image(&pixels, SIZE, SIZE, &fixed).unwrap();
            assert!(
                learned.len() * 10 < fixed.len() * 8,
                "d={distance}: {} vs {}",
                learned.len(),
                fixed.len()
            );
        }
    }

    #[test]
    fn learned_rate_prices_every_transform_of_a_small_image() {
        // Nine 64x64 positions: too few for a table by count, enough by
        // the blocks they cover. Left on the fixed model, the 64x64
        // transform undercuts the learned ones and takes the image.
        const SIZE: usize = 192;
        let pixels = checkerboard_rgb(SIZE);
        let learned = learned_rate_lossy(1.0).with_num_threads(1);
        let fixed = learned.clone().with_learned_rate(false);
        let learned = encode_image(&pixels, SIZE, SIZE, &learned).unwrap();
        let fixed = encode_image(&pixels, SIZE, SIZE, &fixed).unwrap();
        // The bug this guards took 2.6x the fixed size; a shared-DC saving
        // shrinks both sides equally, so the bound must not sit at the ratio.
        assert!(
            learned.len() < fixed.len() * 2,
            "{} vs {}",
            learned.len(),
            fixed.len()
        );
    }

    #[test]
    fn learned_rate_lets_fine_transforms_compete_with_merges() {
        // One-pixel lines on black every 8 pixels. IDENTITY codes them in
        // a few tokens per block; under the fixed model they are only
        // tried on the blocks the merges left.
        const SIZE: usize = 256;
        let mut pixels = Vec::with_capacity(SIZE * SIZE * 3);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let ink = x % 8 == 3 || y % 8 == 3;
                let magenta = (x / 64 + y / 64) % 2 == 0;
                pixels.extend_from_slice(&match (ink, magenta) {
                    (false, _) => [0, 0, 0],
                    (true, true) => [252, 0, 254],
                    (true, false) => [252, 0, 0],
                });
            }
        }
        let learned = learned_rate_lossy(2.0).with_num_threads(1);
        let fixed = learned.clone().with_learned_rate(false);
        let learned = encode_image(&pixels, SIZE, SIZE, &learned).unwrap();
        let fixed = encode_image(&pixels, SIZE, SIZE, &fixed).unwrap();
        // The source-feature quant field already codes the lines well under
        // the fixed model (1.6 KB vs 4 KB on the flat field); the learned
        // prices still take a third off.
        assert!(
            learned.len() * 4 < fixed.len() * 3,
            "{} vs {}",
            learned.len(),
            fixed.len()
        );
    }

    #[test]
    fn learned_rate_is_a_slow_speed_tool() {
        const SIZE: usize = 128;
        let pixels = checkerboard_rgb(SIZE);
        for speed in [Speed::UltraFast, Speed::Fastest, Speed::Fast, Speed::Medium] {
            let base = EncodeConfig::default()
                .with_distance(2.0)
                .with_speed(speed)
                .with_learned_rate(false)
                .with_num_threads(1);
            assert_eq!(
                encode_image(&pixels, SIZE, SIZE, &base).unwrap(),
                encode_image(&pixels, SIZE, SIZE, &base.clone().with_learned_rate(true)).unwrap(),
                "{speed:?}"
            );
        }
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

#[cfg(test)]
mod lossy_alpha_tests {
    use super::*;

    const W: usize = 320;
    const H: usize = 280;

    fn rgba(alpha: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        (0..W * H)
            .flat_map(|i| {
                let (x, y) = (i % W, i / W);
                [(x * 255 / W) as u8, (y * 255 / H) as u8, 96, alpha(x, y)]
            })
            .collect()
    }

    fn disc_radius(x: usize, y: usize) -> f32 {
        (x as f32 - W as f32 / 2.0).hypot(y as f32 - H as f32 / 2.0) / H as f32
    }

    fn soft(x: usize, y: usize) -> u8 {
        (255.0 * ((0.45 - disc_radius(x, y)) / 0.25).clamp(0.0, 1.0)).round() as u8
    }

    fn hard(x: usize, y: usize) -> u8 {
        if disc_radius(x, y) < 0.4 { 255 } else { 0 }
    }

    fn encode(pixels: &[u8], config: &EncodeConfig) -> Vec<u8> {
        encode_image_with_alpha(pixels, W, H, config).unwrap()
    }

    fn fast() -> EncodeConfig {
        EncodeConfig::default()
            .with_distance(1.5)
            .with_num_threads(1)
    }

    #[test]
    fn solid_alpha_is_coded_as_before() {
        for value in [255, 77, 0] {
            let pixels = rgba(|_, _| value);
            assert_eq!(
                encode(&pixels, &fast()),
                encode(&pixels, &fast().with_alpha_distance(0.0)),
                "constant alpha {value}"
            );
        }
    }

    #[test]
    fn soft_alpha_is_smaller_than_lossless() {
        let pixels = rgba(soft);
        for config in [fast(), fast().with_speed(Speed::Slow)] {
            let lossy = encode(&pixels, &config).len();
            let lossless = encode(&pixels, &config.clone().with_alpha_distance(0.0)).len();
            assert!(
                lossy * 10 < lossless * 9,
                "{:?}: lossy alpha {lossy} vs lossless {lossless}",
                config.speed
            );
        }
    }

    #[test]
    fn lossy_alpha_never_grows_the_file() {
        let noisy = |x: usize, y: usize| {
            let h = (x * 7919 + y * 104_729) % 251;
            if h < 5 { 200 + h as u8 } else { 255 }
        };
        for alpha in [hard as fn(usize, usize) -> u8, noisy] {
            let pixels = rgba(alpha);
            assert!(
                encode(&pixels, &fast()).len()
                    <= encode(&pixels, &fast().with_alpha_distance(0.0)).len()
            );
        }
    }

    #[test]
    fn alpha_distance_follows_the_frame_by_default() {
        let pixels = rgba(soft);
        assert_eq!(
            encode(&pixels, &fast()),
            encode(&pixels, &fast().with_alpha_distance(1.5))
        );
        let lossless = encode(&pixels, &fast().with_alpha_distance(0.0));
        for odd in [f32::NAN, -1.0] {
            assert_eq!(encode(&pixels, &fast().with_alpha_distance(odd)), lossless);
        }
        assert!(encode(&pixels, &fast().with_alpha_distance(4.0)).len() < lossless.len());
    }

    #[test]
    fn gray_and_deep_alpha_go_lossy_too() {
        let gray: Vec<u8> = rgba(soft)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|px| [px[0], px[3]])
            .collect();
        let gray_lossy = encode_image_gray_alpha(&gray, W, H, &fast()).unwrap();
        let gray_lossless =
            encode_image_gray_alpha(&gray, W, H, &fast().with_alpha_distance(0.0)).unwrap();
        assert!(gray_lossy.len() < gray_lossless.len());

        let deep: Vec<u16> = rgba(soft).iter().map(|&v| u16::from(v) * 257).collect();
        let deep_lossy = encode_image_with_alpha_16bit(&deep, W, H, &fast()).unwrap();
        let deep_lossless =
            encode_image_with_alpha_16bit(&deep, W, H, &fast().with_alpha_distance(0.0)).unwrap();
        assert!(deep_lossy.len() < deep_lossless.len());
    }
}
