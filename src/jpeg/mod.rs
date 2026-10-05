/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

mod brotli;
mod coeff_order;
mod encode;
mod jbrd;
mod parse;

#[allow(unused_imports)]
pub(crate) use parse::{JpegError, parse_jpeg};
use std::num::NonZeroUsize;

use crate::Speed;
use crate::util::EncodeError;
pub use brotli::BrotliCompression;

/// Appends an ISOBMFF box with the given type and payload.
fn push_box(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    let size = 8 + payload.len();
    if let Ok(small) = u32::try_from(size) {
        out.extend_from_slice(&small.to_be_bytes());
        out.extend_from_slice(kind);
    } else {
        // 64-bit "largesize" escape.
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(&((size + 8) as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
}

/// An ICC profile reassembled from the APP2 `ICC_PROFILE` chunks.
#[derive(Debug, PartialEq)]
struct JpegIcc {
    profile: Vec<u8>,
    /// The chunks' `app_data` indices when they appear in file order as chunk
    /// 1, 2, .., N. Only then can the decoder regenerate them from the
    /// codestream's profile, so `jbrd` need not store them a second time.
    in_order_apps: Option<Vec<usize>>,
}

/// Reassembles an ICC profile from the APP2 `ICC_PROFILE` chunks.
fn extract_icc(jpg: &JpegData) -> Option<JpegIcc> {
    const TAG: &[u8] = b"ICC_PROFILE\0";
    // Each entry is [marker, len_hi, len_lo, payload..].
    let mut chunks: Vec<(u8, usize, &[u8])> = Vec::new();
    let mut expected = 0u8;
    for (index, app) in jpg.app_data.iter().enumerate() {
        if app.first() != Some(&0xE2) || app.len() < 3 + TAG.len() + 2 {
            continue;
        }
        let payload = &app[3..];
        if !payload.starts_with(TAG) {
            continue;
        }
        let seq = payload[TAG.len()];
        let count = payload[TAG.len() + 1];
        if seq == 0 || count == 0 || (expected != 0 && count != expected) {
            return None;
        }
        expected = count;
        chunks.push((seq, index, &payload[TAG.len() + 2..]));
    }
    if chunks.is_empty() || chunks.len() != usize::from(expected) {
        return None;
    }
    let in_order = chunks
        .iter()
        .enumerate()
        .all(|(i, &(seq, _, _))| usize::from(seq) == i + 1);
    chunks.sort_by_key(|&(seq, _, _)| seq);
    if chunks
        .iter()
        .enumerate()
        .any(|(i, &(seq, _, _))| usize::from(seq) != i + 1)
    {
        return None;
    }
    let profile: Vec<u8> = chunks
        .iter()
        .flat_map(|&(_, _, data)| data)
        .copied()
        .collect();
    (!profile.is_empty()).then(|| JpegIcc {
        profile,
        in_order_apps: in_order.then(|| chunks.iter().map(|&(_, index, _)| index).collect()),
    })
}

/// Structural checks a profile must pass before it is embedded, the ones PNG
/// writers and color managers also apply: header size field, `acsp`
/// signature, and every tag inside the profile. Anything else (a corrupted or
/// foreign APP2 payload) would be handed on to every decoder of the image.
fn icc_is_well_formed(icc: &[u8]) -> bool {
    const HEADER: usize = 128;
    let be32 = |at: usize| -> Option<usize> {
        let b = icc.get(at..at + 4)?;
        Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    if icc.len() < HEADER + 4 || be32(0) != Some(icc.len()) || &icc[36..40] != b"acsp" {
        return false;
    }
    let Some(num_tags) = be32(HEADER) else {
        return false;
    };
    let table_end = (num_tags as u64) * 12 + HEADER as u64 + 4;
    if table_end > icc.len() as u64 {
        return false;
    }
    (0..num_tags).all(|i| {
        let entry = HEADER + 4 + i * 12;
        match (be32(entry + 4), be32(entry + 8)) {
            (Some(offset), Some(size)) => (offset as u64 + size as u64) <= icc.len() as u64,
            _ => false,
        }
    })
}

/// Extracts the first standard XMP APP1 segment.
///
/// Returns both its index in `app_data` (for JPEG reconstruction tagging) and
/// the XML packet without the JPEG XMP namespace prefix.
fn extract_xmp(jpg: &JpegData) -> Option<(usize, Vec<u8>)> {
    const TAG: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
    jpg.app_data.iter().enumerate().find_map(|(index, app)| {
        if app.first() != Some(&0xE1) || app.len() < 3 + TAG.len() {
            return None;
        }
        let payload = &app[3..];
        payload
            .starts_with(TAG)
            .then(|| (index, payload[TAG.len()..].to_vec()))
    })
}

/// Options for [`encode_jpeg_lossless_with_config`].
pub struct JpegTranscodeConfig {
    /// Whether to embed the data needed to rebuild the original JPEG bytes.
    ///
    /// Defaults to `true`. See [`Self::with_jpeg_reconstruction`].
    pub jpeg_reconstruction: bool,
    /// Worker threads used while building the codestream. Defaults to 1.
    ///
    /// See [`Self::with_num_threads`].
    pub num_threads: usize,
    /// Optional Brotli implementation for JPEG reconstruction data.
    ///
    /// When absent, Jixel emits a conforming stored-block (uncompressed)
    /// Brotli stream and does not require a Brotli dependency.
    pub brotli_compression: Option<Box<dyn BrotliCompression>>,
    /// Entropy-coding effort. The coefficients, and so the decoded pixels and
    /// the reconstructed JPEG, are the same at every speed. Defaults to
    /// [`Speed::Fast`].
    pub speed: Speed,
}

impl std::fmt::Debug for JpegTranscodeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JpegTranscodeConfig")
            .field("jpeg_reconstruction", &self.jpeg_reconstruction)
            .field("num_threads", &self.num_threads)
            .field("speed", &self.speed)
            .field(
                "brotli_compression",
                &self.brotli_compression.as_ref().map(|_| "custom"),
            )
            .finish()
    }
}

impl Default for JpegTranscodeConfig {
    fn default() -> Self {
        Self {
            jpeg_reconstruction: true,
            num_threads: std::thread::available_parallelism()
                .unwrap_or(NonZeroUsize::new(1).unwrap())
                .get(),
            brotli_compression: None,
            speed: Speed::default(),
        }
    }
}

impl JpegTranscodeConfig {
    /// Controls whether the original JPEG bytes can be recovered.
    pub fn with_jpeg_reconstruction(mut self, enabled: bool) -> Self {
        self.jpeg_reconstruction = enabled;
        self
    }

    /// Sets the number of worker threads, which does not affect the output.
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads.max(1);
        self
    }

    /// Sets the entropy-coding effort; see [`Self::speed`].
    pub fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    /// Uses a caller-provided Brotli encoder for JPEG reconstruction data.
    pub fn with_brotli_compression(mut self, compressor: Box<dyn BrotliCompression>) -> Self {
        self.brotli_compression = Some(compressor);
        self
    }
}

/// Losslessly transcodes a JPEG file into JPEG XL, keeping the ability to
/// reconstruct the original bytes.
pub fn encode_jpeg_lossless(jpeg: &[u8]) -> Result<Vec<u8>, EncodeError> {
    encode_jpeg_lossless_with_config(jpeg, &JpegTranscodeConfig::default())
}

pub fn encode_jpeg_lossless_with_config(
    jpeg: &[u8],
    config: &JpegTranscodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let parsed = parse_jpeg(jpeg).map_err(|e| EncodeError::Jpeg(e.to_string()))?;

    // Embedded in the codestream either way: without it, dropping the `jbrd`
    // box would silently discard the profile and leave the image mislabeled.
    // A malformed one is left out, as libjxl does for broken chunking: the
    // image reads as sRGB and its APP2 bytes still travel in `jbrd`.
    let icc = extract_icc(&parsed).filter(|icc| icc_is_well_formed(&icc.profile));
    let xmp = extract_xmp(&parsed);
    let codestream = encode::encode_jpeg_codestream(
        &parsed,
        icc.as_ref().map(|icc| icc.profile.as_slice()),
        config.speed,
        config.num_threads,
    )?;

    if !config.jpeg_reconstruction {
        // Dropping reconstruction also drops container-only JPEG metadata.
        return Ok(codestream);
    }

    let reconstruction = jbrd::encode_jbrd(
        &parsed,
        xmp.as_ref().map(|(index, _)| *index),
        icc.as_ref()
            .and_then(|icc| icc.in_order_apps.as_deref())
            .unwrap_or_default(),
        config.brotli_compression.as_deref(),
    )?;
    let mut out = Vec::with_capacity(codestream.len() + reconstruction.len() + 64);
    out.extend_from_slice(&[
        0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
    ]);
    push_box(
        &mut out,
        b"ftyp",
        &[b'j', b'x', b'l', b' ', 0, 0, 0, 0, b'j', b'x', b'l', b' '],
    );
    // The reconstruction data has to precede the codestream box.
    push_box(&mut out, b"jbrd", &reconstruction);
    push_box(&mut out, b"jxlc", &codestream);
    if let Some((_, xmp)) = xmp {
        push_box(&mut out, b"xml ", &xmp);
    }
    Ok(out)
}

/// Number of coefficients in one DCT block.
pub(crate) const DCT_BLOCK_SIZE: usize = 64;
/// Longest Huffman code permitted by the JPEG spec.
pub(crate) const HUFF_MAX_BIT_LENGTH: usize = 16;
/// Size of the Huffman alphabet (values 0..=255).
pub(crate) const HUFF_ALPHABET_SIZE: usize = 256;
/// Maximum number of components we accept (the JPEG spec allows 4).
pub(crate) const MAX_COMPONENTS: usize = 4;

pub(crate) static NATURAL_ORDER: [usize; DCT_BLOCK_SIZE] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// A quantization table as it appeared in a DQT segment. `values` are held in
/// natural (raster) order, de-zig-zagged at parse time.
#[derive(Debug, Clone)]
pub(crate) struct JpegQuantTable {
    pub(crate) values: [i32; DCT_BLOCK_SIZE],
    /// 0 for 8-bit tables, 1 for 16-bit tables.
    pub(crate) precision: u32,
    /// Destination slot (`Tq`), 0..=3.
    pub(crate) index: u32,
    /// Whether this table was the last one in its DQT segment.
    pub(crate) is_last: bool,
}

/// A Huffman table as it appeared in a DHT segment.
#[derive(Debug, Clone)]
pub(crate) struct JpegHuffmanCode {
    /// `counts[i]` = number of codes of length `i`, for `i` in 1..=16.
    pub(crate) counts: [u32; HUFF_MAX_BIT_LENGTH + 1],
    /// Symbol values in canonical order.
    pub(crate) values: [u32; HUFF_ALPHABET_SIZE + 1],
    /// `Tc << 4 | Th` — class in the high nibble, destination in the low one.
    pub(crate) slot_id: u32,
    /// Whether this table was the last one in its DHT segment.
    pub(crate) is_last: bool,
}

impl Default for JpegHuffmanCode {
    fn default() -> Self {
        Self {
            counts: [0; HUFF_MAX_BIT_LENGTH + 1],
            values: [0; HUFF_ALPHABET_SIZE + 1],
            slot_id: 0,
            is_last: true,
        }
    }
}

/// Per-component entry inside a scan header.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct JpegComponentScanInfo {
    pub(crate) comp_idx: u32,
    pub(crate) dc_tbl_idx: u32,
    pub(crate) ac_tbl_idx: u32,
}

/// An AC scan position where the encoder emitted a longer zero run than needed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExtraZeroRunInfo {
    pub(crate) block_idx: u32,
    pub(crate) num_extra_zero_runs: u32,
}

/// One scan (SOS segment) plus the extra bookkeeping needed to rebuild it.
#[derive(Debug, Clone, Default)]
pub(crate) struct JpegScanInfo {
    /// Spectral selection start.
    pub(crate) ss: u32,
    /// Spectral selection end.
    pub(crate) se: u32,
    /// Successive approximation high bit.
    pub(crate) ah: u32,
    /// Successive approximation low bit.
    pub(crate) al: u32,
    pub(crate) num_components: u32,
    pub(crate) components: [JpegComponentScanInfo; MAX_COMPONENTS],
    /// Block indices at which a non-minimal zero run was emitted.
    pub(crate) extra_zero_runs: Vec<ExtraZeroRunInfo>,
    /// Block indices where one end-of-block run immediately followed another,
    /// which the re-encoder must reproduce. Nothing to do with restart markers.
    pub(crate) reset_points: Vec<u32>,
    /// Always zero: libjxl declares the field but never sets or reads it.
    pub(crate) last_needed_pass: u32,
}

/// One image component, with its fully-reconstructed coefficient plane.
#[derive(Debug, Clone, Default)]
pub(crate) struct JpegComponent {
    /// Component identifier (`Ci`) from the frame header.
    pub(crate) id: u32,
    pub(crate) h_samp_factor: usize,
    pub(crate) v_samp_factor: usize,
    /// Index into `JpegData::quant`.
    pub(crate) quant_idx: u32,
    pub(crate) width_in_blocks: usize,
    pub(crate) height_in_blocks: usize,
    /// Quantized coefficients
    pub(crate) coeffs: Vec<i16>,
}

/// Everything recovered from a JPEG file, sufficient to rebuild it exactly.
#[derive(Debug, Clone, Default)]
pub(crate) struct JpegData {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) restart_interval: u32,
    /// Raw APPn segments, each as `[marker_byte, len_hi, len_lo, payload..]`.
    pub(crate) app_data: Vec<Vec<u8>>,
    /// Raw COM segments, same layout as `app_data`.
    pub(crate) com_data: Vec<Vec<u8>>,
    pub(crate) quant: Vec<JpegQuantTable>,
    pub(crate) huffman_code: Vec<JpegHuffmanCode>,
    pub(crate) components: Vec<JpegComponent>,
    pub(crate) scan_info: Vec<JpegScanInfo>,
    /// The marker byte of every segment encountered, in file order.
    pub(crate) marker_order: Vec<u8>,
    /// Bytes found between segments that belong to no marker.
    pub(crate) inter_marker_data: Vec<Vec<u8>>,
    /// Bytes following EOI.
    pub(crate) tail_data: Vec<u8>,
    /// Set when some entropy-coded segment padded with something other than
    /// the all-ones fill the JPEG standard prescribes. Only then does
    /// `padding_bits` have to be transmitted; otherwise the reconstruction
    /// regenerates the padding itself.
    pub(crate) has_zero_padding_bit: bool,
    /// Every padding bit observed, MSB-first within each segment and
    /// concatenated in scan order. Always collected, conditionally serialized.
    pub(crate) padding_bits: Vec<u8>,
    /// True if the frame used SOF2 (progressive) rather than SOF0/SOF1.
    pub(crate) is_progressive: bool,
}

impl JpegData {
    /// Maximum horizontal sampling factor across all components.
    pub(crate) fn max_h_samp(&self) -> usize {
        self.components
            .iter()
            .map(|c| c.h_samp_factor)
            .max()
            .unwrap_or(1)
    }

    /// Maximum vertical sampling factor across all components.
    pub(crate) fn max_v_samp(&self) -> usize {
        self.components
            .iter()
            .map(|c| c.v_samp_factor)
            .max()
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an APP2 segment carrying one chunk of an ICC profile.
    fn icc_app2(seq: u8, count: u8, body: &[u8]) -> Vec<u8> {
        let mut payload = b"ICC_PROFILE\0".to_vec();
        payload.push(seq);
        payload.push(count);
        payload.extend_from_slice(body);
        let mut seg = vec![0xE2u8];
        seg.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        seg.extend_from_slice(&payload);
        seg
    }

    fn xmp_app1(body: &[u8]) -> Vec<u8> {
        let mut payload = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
        payload.extend_from_slice(body);
        let mut seg = vec![0xE1u8];
        seg.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        seg.extend_from_slice(&payload);
        seg
    }

    fn with_app(segments: Vec<Vec<u8>>) -> JpegData {
        JpegData {
            app_data: segments,
            ..Default::default()
        }
    }

    #[test]
    fn reassembles_a_split_icc_profile() {
        // Deliberately out of order: chunks carry their own sequence numbers.
        let jpg = with_app(vec![
            icc_app2(2, 3, b"world"),
            icc_app2(1, 3, b"hello "),
            icc_app2(3, 3, b"!"),
        ]);
        let icc = extract_icc(&jpg).unwrap();
        assert_eq!(icc.profile, b"hello world!");
        // Out of file order, so `jbrd` has to keep the segments verbatim.
        assert_eq!(icc.in_order_apps, None);
    }

    #[test]
    fn reads_a_single_chunk_profile() {
        let jpg = with_app(vec![icc_app2(1, 1, b"profile")]);
        let icc = extract_icc(&jpg).unwrap();
        assert_eq!(icc.profile, b"profile");
        assert_eq!(icc.in_order_apps.as_deref(), Some(&[0][..]));
    }

    #[test]
    fn rejects_incomplete_or_inconsistent_profiles() {
        // A chunk is missing.
        let missing = with_app(vec![icc_app2(1, 2, b"half")]);
        assert_eq!(extract_icc(&missing), None);

        // Chunk counts disagree.
        let inconsistent = with_app(vec![icc_app2(1, 2, b"a"), icc_app2(2, 3, b"b")]);
        assert_eq!(extract_icc(&inconsistent), None);

        // Sequence numbers are 1-based and contiguous.
        let bad_seq = with_app(vec![icc_app2(0, 1, b"a")]);
        assert_eq!(extract_icc(&bad_seq), None);
    }

    #[test]
    fn in_order_chunks_are_reported_for_regeneration() {
        let mut exif = vec![0xE1u8, 0x00, 0x05];
        exif.extend_from_slice(b"Exi");
        let jpg = with_app(vec![
            icc_app2(1, 2, b"hello "),
            exif,
            icc_app2(2, 2, b"world"),
        ]);
        let icc = extract_icc(&jpg).unwrap();
        assert_eq!(icc.profile, b"hello world");
        assert_eq!(icc.in_order_apps.as_deref(), Some(&[0, 2][..]));
    }

    /// A minimal structurally valid profile: header plus one tag.
    fn tiny_profile() -> Vec<u8> {
        let mut icc = vec![0u8; 128];
        icc[36..40].copy_from_slice(b"acsp");
        icc.extend_from_slice(&1u32.to_be_bytes());
        icc.extend_from_slice(b"desc");
        icc.extend_from_slice(&144u32.to_be_bytes());
        icc.extend_from_slice(&4u32.to_be_bytes());
        icc.extend_from_slice(b"data");
        let len = icc.len() as u32;
        icc[..4].copy_from_slice(&len.to_be_bytes());
        icc
    }

    #[test]
    fn well_formed_profiles_pass_and_damaged_ones_do_not() {
        let good = tiny_profile();
        assert!(icc_is_well_formed(&good));

        assert!(!icc_is_well_formed(b"hello world"));
        let mut bad_magic = good.clone();
        bad_magic[36] = b'x';
        assert!(!icc_is_well_formed(&bad_magic));
        // Size field disagrees with the reassembled length.
        let mut truncated = good.clone();
        truncated.pop();
        assert!(!icc_is_well_formed(&truncated));
        // A tag pointing past the end.
        let mut bad_tag = good.clone();
        bad_tag[136..140].copy_from_slice(&200u32.to_be_bytes());
        assert!(!icc_is_well_formed(&bad_tag));
        // More tags than the table can hold.
        let mut bad_count = good;
        bad_count[128..132].copy_from_slice(&1000u32.to_be_bytes());
        assert!(!icc_is_well_formed(&bad_count));
    }

    #[test]
    fn ignores_unrelated_app_segments() {
        let mut jfif = vec![0xE0u8, 0x00, 0x10];
        jfif.extend_from_slice(b"JFIF\0\x01\x02\x00\x00\x01\x00\x01\x00\x00");
        assert_eq!(extract_icc(&with_app(vec![jfif])), None);
    }

    #[test]
    fn extracts_the_first_standard_xmp_packet() {
        let jpg = with_app(vec![
            vec![0xE1, 0, 5, b'n', b'o', b'p'],
            xmp_app1(b"<x:xmpmeta>first</x:xmpmeta>"),
            xmp_app1(b"<x:xmpmeta>second</x:xmpmeta>"),
        ]);
        let (index, xmp) = extract_xmp(&jpg).expect("XMP packet");
        assert_eq!(index, 1);
        assert_eq!(xmp, b"<x:xmpmeta>first</x:xmpmeta>");
    }

    #[test]
    fn reconstruction_is_on_by_default() {
        let default = JpegTranscodeConfig::default();
        assert!(default.jpeg_reconstruction);
        assert!(default.brotli_compression.is_none());
        assert!(
            !JpegTranscodeConfig::default()
                .with_jpeg_reconstruction(false)
                .jpeg_reconstruction
        );
    }
}
