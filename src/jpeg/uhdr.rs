/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Binary ISO 21496-1 gain maps in two-image Ultra HDR JPEGs.
//!
//! As in gainforge and libultrahdr, the gain map is the JPEG following the
//! primary image's EOI. The existing JPEG parser keeps all those original
//! bytes in `tail_data`, so `jbrd` can still reconstruct the complete file.

use super::{JpegData, JpegTranscodeConfig, encode, extract_icc, icc_is_well_formed, parse_jpeg};
use crate::gain_map::write_jhgm_bundle;
use crate::{EncodeError, IsoGainMap, Orientation};

const ISO_TAG: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";

/// Find the ISO APP2 in a secondary JPEG's header before allocating its
/// coefficient planes. Unrelated JPEGs and arbitrary trailing data keep their
/// existing behavior. Segment lengths skip embedded thumbnails and marker-like
/// bytes in other APP payloads.
fn header_metadata(jpeg: &[u8]) -> Option<&[u8]> {
    if !jpeg.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut pos = 2;
    loop {
        if *jpeg.get(pos)? != 0xFF {
            return None;
        }
        while *jpeg.get(pos)? == 0xFF {
            pos += 1;
        }
        let marker = *jpeg.get(pos)?;
        pos += 1;
        // Ultra HDR puts the metadata before the first scan.
        if matches!(marker, 0x00 | 0xD8 | 0xD9 | 0xDA) {
            return None;
        }
        if matches!(marker, 0x01 | 0xD0..=0xD7) {
            continue;
        }
        let length = usize::from(u16::from_be_bytes(jpeg.get(pos..pos + 2)?.try_into().ok()?));
        if length < 2 {
            return None;
        }
        let payload = jpeg.get(pos + 2..pos.checked_add(length)?)?;
        if marker == 0xE2
            && let Some(metadata) = payload.strip_prefix(ISO_TAG)
        {
            return Some(metadata);
        }
        pos += length;
    }
}

pub(super) fn encode_gain_map(
    primary: &JpegData,
    orientation: Orientation,
    config: &JpegTranscodeConfig,
) -> Result<Option<Vec<u8>>, EncodeError> {
    let primary_version = primary
        .app_data
        .iter()
        .find_map(|app| {
            if app.first() != Some(&0xE2) {
                return None;
            }
            app.get(3..)?.strip_prefix(ISO_TAG)
        })
        // A standalone gain map JPEG has full metadata here. It remains an
        // ordinary image when transcoded on its own, without a nested map.
        .filter(|metadata| metadata.len() <= 4);
    if let Some(version) = primary_version {
        // The primary APP2 carries only minimum_version and writer_version;
        // the full metadata is in the gain map's APP2, never in this marker.
        if version.len() < 4 {
            return Err(EncodeError::GainMap(
                "Ultra HDR primary version is truncated",
            ));
        }
        if version[..2] != [0, 0] {
            return Err(EncodeError::GainMap(
                "unsupported Ultra HDR minimum version",
            ));
        }
    }

    let secondary = primary
        .tail_data
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0xD8])
        .map(|start| &primary.tail_data[start..]);
    let metadata = secondary.and_then(header_metadata);
    let Some((secondary, metadata)) = secondary.zip(metadata) else {
        return if primary_version.is_some() {
            Err(EncodeError::GainMap(
                "Ultra HDR gain map or binary metadata is missing",
            ))
        } else {
            Ok(None)
        };
    };

    IsoGainMap::from_metadata(metadata)?;
    let gain_map = parse_jpeg(secondary).map_err(|e| EncodeError::Jpeg(e.to_string()))?;
    let icc = extract_icc(&gain_map).filter(|icc| icc_is_well_formed(&icc.profile));
    let profile = icc.as_ref().map(|icc| icc.profile.as_slice());
    // Gain map orientation follows the primary image, as it does for maps
    // attached to a pixel encode. No nested container or reconstruction data.
    let codestream = encode::encode_jpeg_codestream(
        &gain_map,
        profile,
        orientation,
        config.speed,
        config.num_threads,
    )?;

    // The secondary ICC describes the alternate rendition, independently of
    // use_base_color_space (which selects the gain map's application space).
    // As in libultrahdr's applyGainMap, no secondary ICC means inherit the base.
    write_jhgm_bundle(metadata, None, profile, &codestream).map(Some)
}
