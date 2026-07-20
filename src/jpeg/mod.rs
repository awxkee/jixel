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

//! Bit-exact JPEG bitstream parsing for lossless JXL transcoding.
//!
//! These structures mirror libjxl's `JPEGData`, since the `jbrd` box is a
//! direct serialization of them. The apparently redundant detail — marker
//! order, raw APP/COM bytes, padding bits, trailing garbage — is all required
//! to re-emit the original file byte for byte.

mod brotli;
mod encode;
mod jbrd;
mod parse;

#[allow(unused_imports)]
pub(crate) use parse::{JpegError, parse_jpeg};

use crate::util::EncodeError;

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

/// Losslessly transcodes a JPEG file into JPEG XL.
pub fn encode_jpeg_lossless(jpeg: &[u8]) -> Result<Vec<u8>, EncodeError> {
    let parsed = parse_jpeg(jpeg).map_err(|e| EncodeError::Jpeg(e.to_string()))?;
    let codestream =
        encode::encode_jpeg_codestream(&parsed).map_err(|e| EncodeError::Jpeg(e.to_string()))?;
    let reconstruction =
        jbrd::encode_jbrd(&parsed).map_err(|e| EncodeError::Jpeg(e.to_string()))?;

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
    /// Quantized coefficients, `width_in_blocks * height_in_blocks` blocks of
    /// 64, each block in **natural** order.
    pub(crate) coeffs: Vec<i32>,
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
