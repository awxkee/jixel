/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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
//! Gain map ("HDR gain map") support: attach a second image to an encode and
//! ship it in the JPEG XL `jhgm` container box, the way libjxl's
//! `JxlGainMapBundle` lays it out:
//!
//! ```text
//! u8      jhgm_version (0)
//! u16 BE  gain_map_metadata_size
//! [..]    gain_map_metadata           ISO 21496-1 binary blob (IsoGainMap)
//! u8      color_encoding_size         0 = absent
//! [..]    color_encoding              JXL ColorEncoding bundle, byte padded
//! u32 BE  alt_icc_size                0 = absent
//! [..]    alt_icc                     ICC profile, JXL ICC-codec compressed
//! [..]    gain_map                    naked JPEG XL codestream (to the end)
//! ```
//!
//! The gain map itself is a JPEG XL *codestream* (not a container), encoded
//! with the same engine as the primary image. Its dimensions are independent
//! of the primary image (quarter-resolution gain maps are common).

use crate::bit_writer::BitWriter;
use crate::color_encoding::write_color_encoding_with_icc;
use crate::encode_image::{
    EncodeConfig, encode_image, encode_image_10bit, encode_image_12bit, encode_image_16bit,
    encode_image_gray, encode_image_gray_10bit, encode_image_gray_12bit, encode_image_gray_16bit,
};
use crate::iso_gain_map::IsoGainMap;
use crate::{ColorEncoding, EncodeError};
use std::fmt;

/// Version byte written at the start of every `jhgm` bundle.
pub(crate) const JHGM_VERSION: u8 = 0;

/// Sample storage of a gain map image.
///
/// Integer samples only: 8-bit, or 10/12/16-bit in `u16` (values `0..2^bits`).
/// Gray maps are single-channel luminance gain; RGB maps carry per-channel gain.
#[derive(Clone)]
pub enum GainMapPixels {
    /// `width * height` 8-bit gray samples.
    Gray8(Vec<u8>),
    /// `width * height * 3` interleaved 8-bit RGB samples.
    Rgb8(Vec<u8>),
    /// `width * height` gray samples of `bits` (10, 12 or 16) bits each.
    Gray16 { data: Vec<u16>, bits: u8 },
    /// `width * height * 3` interleaved RGB samples of `bits` (10, 12 or 16) bits each.
    Rgb16 { data: Vec<u16>, bits: u8 },
}

impl GainMapPixels {
    /// color channels stored per pixel (1 or 3).
    pub fn channels(&self) -> usize {
        match self {
            Self::Gray8(_) | Self::Gray16 { .. } => 1,
            Self::Rgb8(_) | Self::Rgb16 { .. } => 3,
        }
    }

    /// Bit depth of the samples.
    pub fn bits(&self) -> u8 {
        match self {
            Self::Gray8(_) | Self::Rgb8(_) => 8,
            Self::Gray16 { bits, .. } | Self::Rgb16 { bits, .. } => *bits,
        }
    }

    /// Number of samples in the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Gray8(v) | Self::Rgb8(v) => v.len(),
            Self::Gray16 { data, .. } | Self::Rgb16 { data, .. } => data.len(),
        }
    }

    /// True when the buffer holds no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for GainMapPixels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Self::Gray8(_) => "Gray8",
            Self::Rgb8(_) => "Rgb8",
            Self::Gray16 { .. } => "Gray16",
            Self::Rgb16 { .. } => "Rgb16",
        };
        write!(
            f,
            "GainMapPixels::{kind} {{ samples: {}, bits: {} }}",
            self.len(),
            self.bits()
        )
    }
}

/// A gain map to attach to an encode via [`EncodeConfig::with_gain_map`].
///
/// The primary image stays the base rendition; the gain map is written as a
/// second JPEG XL codestream inside a `jhgm` box together with its
/// [`IsoGainMap`] metadata and, optionally, the color description of the
/// alternate (usually HDR) rendition it reconstructs.
///
/// ```no_run
/// use jixel::{EncodeConfig, GainMap, GainMapFloats, IsoGainMap};
/// # let (rgb, w, h): (Vec<u8>, usize, usize) = (vec![], 0, 0);
/// # let gain: Vec<u8> = vec![];
/// let metadata = IsoGainMap::from_floats(&GainMapFloats {
///     max: [2.0; 3],                 // log2 of the largest gain
///     alternate_hdr_headroom: 2.0,   // log2 of the HDR rendition's headroom
///     ..GainMapFloats::default()
/// })?;
/// let gain_map = GainMap::gray8(gain, w / 4, h / 4, metadata).with_distance(2.0);
/// let jxl = jixel::encode_image(&rgb, w, h, &EncodeConfig::default().with_gain_map(gain_map))?;
/// # Ok::<(), jixel::EncodeError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GainMap {
    /// Gain map width in pixels (independent of the primary image).
    pub width: usize,
    /// Gain map height in pixels.
    pub height: usize,
    /// Gain map samples.
    pub pixels: GainMapPixels,
    /// ISO 21496-1 metadata describing how to apply the map.
    pub metadata: IsoGainMap,
    /// color encoding the gain map samples are declared in inside their own
    /// codestream (default sRGB). Decoders return the samples in this space,
    /// so it only needs to match what the map was authored in.
    pub color_encoding: ColorEncoding,
    /// color encoding of the alternate rendition (e.g. BT.2020 PQ for an HDR
    /// alternate). Written into the bundle when set; omitted otherwise.
    pub alternate_color_encoding: Option<ColorEncoding>,
    /// ICC profile of the alternate rendition. Stored compressed with the
    /// JPEG XL ICC codec (libjxl `JxlICCProfileDecode` restores it).
    pub alternate_icc_profile: Option<Vec<u8>>,
    /// Butteraugli distance for the gain map codestream. `None` inherits the
    /// primary image's distance (1.0 when the primary is lossless).
    pub distance: Option<f32>,
    /// Encode the gain map losslessly (Modular). Default false.
    pub lossless: bool,
}

impl GainMap {
    fn new(pixels: GainMapPixels, width: usize, height: usize, metadata: IsoGainMap) -> Self {
        Self {
            width,
            height,
            pixels,
            metadata,
            color_encoding: ColorEncoding::default(),
            alternate_color_encoding: None,
            alternate_icc_profile: None,
            distance: None,
            lossless: false,
        }
    }

    /// 8-bit single-channel gain map (`width * height` samples).
    pub fn gray8(pixels: Vec<u8>, width: usize, height: usize, metadata: IsoGainMap) -> Self {
        Self::new(GainMapPixels::Gray8(pixels), width, height, metadata)
    }

    /// 8-bit RGB gain map (`width * height * 3` interleaved samples).
    pub fn rgb8(pixels: Vec<u8>, width: usize, height: usize, metadata: IsoGainMap) -> Self {
        Self::new(GainMapPixels::Rgb8(pixels), width, height, metadata)
    }

    /// Single-channel gain map with `bits` (10, 12 or 16) bits per sample.
    pub fn gray16(
        pixels: Vec<u16>,
        bits: u8,
        width: usize,
        height: usize,
        metadata: IsoGainMap,
    ) -> Self {
        Self::new(
            GainMapPixels::Gray16 { data: pixels, bits },
            width,
            height,
            metadata,
        )
    }

    /// RGB gain map with `bits` (10, 12 or 16) bits per sample.
    pub fn rgb16(
        pixels: Vec<u16>,
        bits: u8,
        width: usize,
        height: usize,
        metadata: IsoGainMap,
    ) -> Self {
        Self::new(
            GainMapPixels::Rgb16 { data: pixels, bits },
            width,
            height,
            metadata,
        )
    }

    /// Set the gain map codestream's own butteraugli distance.
    pub fn with_distance(mut self, distance: f32) -> Self {
        self.distance = Some(distance);
        self
    }

    /// Encode the gain map losslessly.
    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// color encoding the gain map samples are declared in (default sRGB).
    pub fn with_color_encoding(mut self, enc: ColorEncoding) -> Self {
        self.color_encoding = enc;
        self
    }

    /// color encoding of the alternate rendition, stored in the bundle.
    pub fn with_alternate_color_encoding(mut self, enc: ColorEncoding) -> Self {
        self.alternate_color_encoding = Some(enc);
        self
    }

    /// ICC profile of the alternate rendition, stored compressed in the bundle.
    pub fn with_alternate_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.alternate_icc_profile = Some(icc);
        self
    }

    fn validate(&self) -> Result<(), EncodeError> {
        if self.width == 0 || self.height == 0 {
            return Err(EncodeError::GainMap("gain map dimensions must be non-zero"));
        }
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(self.pixels.channels()))
            .ok_or(EncodeError::SizeOverflow)?;
        if self.pixels.len() != expected {
            return Err(EncodeError::GainMapSizeMismatch {
                expected,
                actual: self.pixels.len(),
            });
        }
        if !matches!(self.pixels.bits(), 8 | 10 | 12 | 16) {
            return Err(EncodeError::GainMap(
                "gain map bit depth must be 8, 10, 12 or 16",
            ));
        }
        if let Some(d) = self.distance
            && (!d.is_finite() || d <= 0.0)
        {
            return Err(EncodeError::InvalidDistance(d));
        }
        if matches!(self.alternate_icc_profile.as_deref(), Some(&[])) {
            return Err(EncodeError::GainMap("alternate ICC profile is empty"));
        }
        Ok(())
    }
}

/// A gain map already serialized into its `jhgm` bundle, ready to be placed
/// in the container.
#[derive(Debug, Clone)]
pub(crate) struct EncodedGainMap {
    /// Complete `jhgm` box payload.
    pub(crate) bundle: Vec<u8>,
    /// The gain map codestream needs codestream level 10 (see
    /// `needs_level_10`); the enclosing file's `jxll` box must say so too.
    pub(crate) needs_level_10: bool,
}

/// Encode the gain map attached to `base` (if any) into a `jhgm` bundle.
///
/// The gain map codestream inherits the primary image's speed, thread count,
/// orientation, dark-AQ and lossy-modular settings; it never carries EXIF,
/// XMP, an ICC profile or a nested gain map, and is never progressive.
pub(crate) fn encode_gain_map(base: &EncodeConfig) -> Result<Option<EncodedGainMap>, EncodeError> {
    let Some(gm) = base.gain_map.as_ref() else {
        return Ok(None);
    };
    gm.validate()?;

    let distance = gm
        .distance
        .unwrap_or(if base.lossless { 1.0 } else { base.distance });
    let cfg = EncodeConfig {
        distance,
        color_encoding: gm.color_encoding,
        icc_profile: None,
        exif: None,
        xmp: None,
        brotli_compression: None,
        orientation: base.orientation,
        lossless: gm.lossless,
        progressive: false,
        patches: base.patches,
        progressive_passes: None,
        progressive_shifts: None,
        intensity_target: None,
        min_nits: 0.0,
        relative_to_max_display: false,
        linear_below: 0.0,
        num_threads: base.num_threads,
        speed: base.speed,
        decoding_speed: base.decoding_speed,
        boost: base.boost,
        lossy_modular: base.lossy_modular,
        gain_map: None,
    };

    let (w, h) = (gm.width, gm.height);
    let encoded = match &gm.pixels {
        GainMapPixels::Gray8(p) => encode_image_gray(p, w, h, &cfg)?,
        GainMapPixels::Rgb8(p) => encode_image(p, w, h, &cfg)?,
        GainMapPixels::Gray16 { data, bits } => match bits {
            10 => encode_image_gray_10bit(data, w, h, &cfg)?,
            12 => encode_image_gray_12bit(data, w, h, &cfg)?,
            _ => encode_image_gray_16bit(data, w, h, &cfg)?,
        },
        GainMapPixels::Rgb16 { data, bits } => match bits {
            10 => encode_image_10bit(data, w, h, &cfg)?,
            12 => encode_image_12bit(data, w, h, &cfg)?,
            _ => encode_image_16bit(data, w, h, &cfg)?,
        },
    };
    let (codestream, needs_level_10) = strip_container(encoded)?;

    let metadata = gm.metadata.to_metadata()?;
    let bundle = write_jhgm_bundle(
        &metadata,
        gm.alternate_color_encoding.as_ref(),
        gm.alternate_icc_profile.as_deref(),
        &codestream,
    )?;
    Ok(Some(EncodedGainMap {
        bundle,
        needs_level_10,
    }))
}

/// JXL signature box that opens every container file.
static CONTAINER_SIGNATURE: [u8; 12] = [
    0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
];

/// Reduce an encoder output to its naked codestream. A bare codestream is
/// returned as is; a container is scanned for its `jxlc` box and the level
/// declared by `jxll` (the encoder only wraps when level 10 is required, or
/// for metadata boxes, which the gain map config never requests).
fn strip_container(bytes: Vec<u8>) -> Result<(Vec<u8>, bool), EncodeError> {
    if !bytes.starts_with(&CONTAINER_SIGNATURE) {
        return Ok((bytes, false));
    }
    let mut level_10 = false;
    let mut codestream = None;
    let mut pos = 0usize;
    while pos + 8 <= bytes.len() {
        let small = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let kind: &[u8; 4] = bytes[pos + 4..pos + 8].try_into().unwrap();
        let (header, size) = if small == 1 {
            if pos + 16 > bytes.len() {
                break;
            }
            let large = u64::from_be_bytes(bytes[pos + 8..pos + 16].try_into().unwrap());
            (
                16usize,
                usize::try_from(large).map_err(|_| EncodeError::SizeOverflow)?,
            )
        } else if small == 0 {
            (8usize, bytes.len() - pos) // box extends to end of file
        } else {
            (8usize, small)
        };
        if size < header || pos + size > bytes.len() {
            break;
        }
        let payload = &bytes[pos + header..pos + size];
        match kind {
            b"jxll" => level_10 = payload.first().copied() == Some(10),
            b"jxlc" => codestream = Some(payload.to_vec()),
            _ => {}
        }
        pos += size;
    }
    match codestream {
        Some(c) => Ok((c, level_10)),
        None => Err(EncodeError::GainMap(
            "internal: gain map container without a jxlc box",
        )),
    }
}

/// Serialize a JXL `ColorEncoding` bundle on its own, zero-padded to a byte
/// boundary (what libjxl's `Bundle::Write` + `ZeroPadToByte` produce).
fn color_encoding_bytes(enc: &ColorEncoding) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_color_encoding_with_icc(enc, false, false, &mut w);
    w.zero_pad_to_byte();
    w.into_bytes()
}

/// Compress an ICC profile with the JPEG XL ICC codec (ISO/IEC 18181-1
/// Annex, the same bit stream a codestream header carries), byte-padded.
/// This is the format libjxl's `JxlICCProfileEncode`/`Decode` pair uses.
pub(crate) fn compress_icc(icc: &[u8]) -> Vec<u8> {
    let mut pool = Vec::with_capacity(1024);
    let mut w = BitWriter::new();
    crate::icc_codec::write_icc_stream(icc, &mut pool, &mut w);
    w.zero_pad_to_byte();
    w.into_bytes()
}

/// Assemble the `jhgm` box payload from its parts (layout in the module docs).
pub(crate) fn write_jhgm_bundle(
    metadata: &[u8],
    color_encoding: Option<&ColorEncoding>,
    alt_icc: Option<&[u8]>,
    codestream: &[u8],
) -> Result<Vec<u8>, EncodeError> {
    let metadata_size = u16::try_from(metadata.len())
        .map_err(|_| EncodeError::GainMap("gain map metadata exceeds 65535 bytes"))?;
    let enc_bytes = color_encoding.map(color_encoding_bytes).unwrap_or_default();
    let enc_size = u8::try_from(enc_bytes.len())
        .map_err(|_| EncodeError::GainMap("gain map color encoding exceeds 255 bytes"))?;
    let icc_bytes = alt_icc.map(compress_icc).unwrap_or_default();
    let icc_size = u32::try_from(icc_bytes.len())
        .map_err(|_| EncodeError::GainMap("compressed alternate ICC exceeds 4 GiB"))?;

    let mut out = Vec::with_capacity(
        1 + 2 + metadata.len() + 1 + enc_bytes.len() + 4 + icc_bytes.len() + codestream.len(),
    );
    out.push(JHGM_VERSION);
    out.extend_from_slice(&metadata_size.to_be_bytes());
    out.extend_from_slice(metadata);
    out.push(enc_size);
    out.extend_from_slice(&enc_bytes);
    out.extend_from_slice(&icc_size.to_be_bytes());
    out.extend_from_slice(&icc_bytes);
    out.extend_from_slice(codestream);
    Ok(out)
}

/// Parsed view of a `jhgm` bundle (test/debug helper mirroring
/// libjxl's `JxlGainMapReadBundle`).
#[cfg(test)]
pub(crate) struct ParsedBundle<'a> {
    pub version: u8,
    pub metadata: &'a [u8],
    pub color_encoding: &'a [u8],
    pub alt_icc: &'a [u8],
    pub codestream: &'a [u8],
}

#[cfg(test)]
pub(crate) fn parse_jhgm_bundle(b: &[u8]) -> Option<ParsedBundle<'_>> {
    let mut pos = 0usize;
    let version = *b.get(pos)?;
    pos += 1;
    let md_size = u16::from_be_bytes(b.get(pos..pos + 2)?.try_into().ok()?) as usize;
    pos += 2;
    let metadata = b.get(pos..pos + md_size)?;
    pos += md_size;
    let ce_size = *b.get(pos)? as usize;
    pos += 1;
    let color_encoding = b.get(pos..pos + ce_size)?;
    pos += ce_size;
    let icc_size = u32::from_be_bytes(b.get(pos..pos + 4)?.try_into().ok()?) as usize;
    pos += 4;
    let alt_icc = b.get(pos..pos + icc_size)?;
    pos += icc_size;
    let codestream = b.get(pos..)?;
    Some(ParsedBundle {
        version,
        metadata,
        color_encoding,
        alt_icc,
        codestream,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GainMapFloats;

    fn metadata() -> IsoGainMap {
        IsoGainMap::from_floats(&GainMapFloats {
            max: [2.0; 3],
            alternate_hdr_headroom: 2.0,
            ..GainMapFloats::default()
        })
        .unwrap()
    }

    /// Minimal, structurally valid ICC-like blob (128-byte header + 0 tags).
    fn fake_icc() -> Vec<u8> {
        let mut icc = vec![0u8; 132];
        icc[0..4].copy_from_slice(&(132u32).to_be_bytes());
        icc[4..8].copy_from_slice(b"jixl");
        icc[12..16].copy_from_slice(b"mntr");
        icc[16..20].copy_from_slice(b"RGB ");
        icc[20..24].copy_from_slice(b"XYZ ");
        icc[36..40].copy_from_slice(b"acsp");
        icc
    }

    #[test]
    fn bundle_layout_matches_libjxl() {
        let md = metadata().to_metadata().unwrap();
        let enc = ColorEncoding::bt2020_pq();
        let icc = fake_icc();
        let codestream = vec![0xFF, 0x0A, 1, 2, 3, 4, 5];
        let bundle = write_jhgm_bundle(&md, Some(&enc), Some(&icc), &codestream).unwrap();
        let p = parse_jhgm_bundle(&bundle).unwrap();
        assert_eq!(p.version, JHGM_VERSION);
        assert_eq!(p.metadata, md.as_slice());
        assert_eq!(p.color_encoding, color_encoding_bytes(&enc).as_slice());
        assert!(!p.color_encoding.is_empty() && p.color_encoding.len() < 8);
        assert_eq!(p.alt_icc, compress_icc(&icc).as_slice());
        assert!(!p.alt_icc.is_empty());
        assert_eq!(p.codestream, codestream.as_slice());
        // Fixed-size fields: 1 + 2 + md + 1 + enc + 4 + icc + codestream.
        assert_eq!(
            bundle.len(),
            8 + md.len() + p.color_encoding.len() + p.alt_icc.len() + codestream.len()
        );
    }

    #[test]
    fn absent_optional_parts_are_zero_sized() {
        let md = metadata().to_metadata().unwrap();
        let bundle = write_jhgm_bundle(&md, None, None, &[0xFF, 0x0A]).unwrap();
        let p = parse_jhgm_bundle(&bundle).unwrap();
        assert!(p.color_encoding.is_empty());
        assert!(p.alt_icc.is_empty());
        assert_eq!(p.codestream, &[0xFF, 0x0A]);
        assert_eq!(bundle[1 + 2 + md.len()], 0); // color_encoding_size
        assert_eq!(&bundle[1 + 2 + md.len() + 1..][..4], &[0, 0, 0, 0]); // alt_icc_size
    }

    #[test]
    fn strip_container_handles_bare_and_wrapped_streams() {
        let bare = vec![0xFF, 0x0A, 9, 9];
        assert_eq!(
            strip_container(bare.clone()).unwrap(),
            (bare.clone(), false)
        );
        let wrapped =
            crate::encode_image::wrap_jxl_container(bare.clone(), 10, None, None, None, None)
                .unwrap();
        assert_eq!(strip_container(wrapped).unwrap(), (bare.clone(), true));
        let wrapped5 =
            crate::encode_image::wrap_jxl_container(bare.clone(), 5, None, None, None, None)
                .unwrap();
        assert_eq!(strip_container(wrapped5).unwrap(), (bare, false));
    }

    #[test]
    fn validation_catches_bad_gain_maps() {
        let m = metadata();
        assert!(matches!(
            GainMap::gray8(vec![0; 3], 2, 2, m).validate(),
            Err(EncodeError::GainMapSizeMismatch {
                expected: 4,
                actual: 3
            })
        ));
        assert!(GainMap::rgb8(vec![0; 12], 2, 2, m).validate().is_ok());
        assert!(GainMap::gray16(vec![0; 4], 9, 2, 2, m).validate().is_err());
        assert!(GainMap::gray8(vec![], 0, 2, m).validate().is_err());
        assert!(
            GainMap::gray8(vec![0; 4], 2, 2, m)
                .with_distance(0.0)
                .validate()
                .is_err()
        );
        assert!(
            GainMap::gray8(vec![0; 4], 2, 2, m)
                .with_alternate_icc_profile(vec![])
                .validate()
                .is_err()
        );
    }
}
