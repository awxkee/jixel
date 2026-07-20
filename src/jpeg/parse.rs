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

//! The JPEG marker/entropy parser.
//!
//! Only the subset of JPEG that is actually transcodable to JXL is accepted:
//! 8-bit Huffman-coded baseline (SOF0/SOF1) and progressive (SOF2) images with
//! at most four components. Arithmetic coding, hierarchical mode, 12-bit
//! samples and lossless JPEG are all rejected up front rather than parsed
//! incorrectly.

use super::{
    DCT_BLOCK_SIZE, ExtraZeroRunInfo, HUFF_ALPHABET_SIZE, HUFF_MAX_BIT_LENGTH, JpegComponent,
    JpegComponentScanInfo, JpegData, JpegHuffmanCode, JpegQuantTable, JpegScanInfo, MAX_COMPONENTS,
    NATURAL_ORDER,
};

/// Reasons a JPEG may be rejected for transcoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JpegError {
    /// The file does not begin with an SOI marker.
    NotAJpeg,
    /// The bitstream ended in the middle of a segment.
    Truncated,
    /// A marker segment declared an implausible length.
    BadSegmentLength(u8),
    /// A coding mode we deliberately do not support (arithmetic, hierarchical,
    /// lossless, 12-bit, ...).
    UnsupportedMode(&'static str),
    /// A frame header was malformed or contradicted an earlier one.
    BadFrameHeader,
    /// A scan header was malformed.
    BadScanHeader,
    /// A DQT/DHT segment was malformed.
    BadTable(&'static str),
    /// The entropy-coded data did not decode cleanly.
    BadEntropyData,
    /// More than one SOF, or an SOS before any SOF.
    UnexpectedMarker(u8),
}

impl core::fmt::Display for JpegError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotAJpeg => write!(f, "input does not start with a JPEG SOI marker"),
            Self::Truncated => write!(f, "JPEG data ended unexpectedly"),
            Self::BadSegmentLength(m) => {
                write!(f, "marker 0xFF{m:02X} declared an invalid segment length")
            }
            Self::UnsupportedMode(what) => write!(f, "unsupported JPEG mode: {what}"),
            Self::BadFrameHeader => write!(f, "malformed JPEG frame header"),
            Self::BadScanHeader => write!(f, "malformed JPEG scan header"),
            Self::BadTable(what) => write!(f, "malformed JPEG {what} table"),
            Self::BadEntropyData => write!(f, "malformed JPEG entropy-coded data"),
            Self::UnexpectedMarker(m) => write!(f, "unexpected marker 0xFF{m:02X}"),
        }
    }
}

/// Number of bits resolved by the direct lookup table.
///
/// Eight covers the overwhelming majority of real codes; longer ones fall back
/// to the canonical walk below.
const HUFF_LUT_BITS: u32 = 8;
const HUFF_LUT_SIZE: usize = 1 << HUFF_LUT_BITS;

/// A decoding table built from a DHT segment, in the classic
/// min-code/max-code/value-pointer form.
#[derive(Clone)]
struct HuffTable {
    /// `(symbol, code length)` for every `HUFF_LUT_BITS`-bit prefix. A length of
    /// zero means the prefix needs the slow path — either the code is longer
    /// than the table resolves, or it is not a valid code at all.
    lut: [(u8, u8); HUFF_LUT_SIZE],
    /// `min_code[l]` / `max_code[l]` bound the codes of length `l+1`.
    /// `max_code[l] < 0` means no codes of that length exist.
    min_code: [i32; HUFF_MAX_BIT_LENGTH],
    max_code: [i32; HUFF_MAX_BIT_LENGTH],
    /// Index into `values` of the first code of length `l+1`.
    val_ptr: [usize; HUFF_MAX_BIT_LENGTH],
    values: Vec<u8>,
    present: bool,
}

impl Default for HuffTable {
    fn default() -> Self {
        Self {
            lut: [(0, 0); HUFF_LUT_SIZE],
            min_code: [0; HUFF_MAX_BIT_LENGTH],
            max_code: [-1; HUFF_MAX_BIT_LENGTH],
            val_ptr: [0; HUFF_MAX_BIT_LENGTH],
            values: Vec::new(),
            present: false,
        }
    }
}

impl HuffTable {
    fn build(counts: &[u32; HUFF_MAX_BIT_LENGTH + 1], values: Vec<u8>) -> Result<Self, JpegError> {
        let mut table = HuffTable {
            values,
            present: true,
            ..Default::default()
        };
        let mut code: i32 = 0;
        let mut k: usize = 0;
        for l in 0..HUFF_MAX_BIT_LENGTH {
            let n = counts[l + 1] as i32;
            if n == 0 {
                table.max_code[l] = -1;
                code <<= 1;
                continue;
            }
            table.val_ptr[l] = k;
            table.min_code[l] = code;
            k += n as usize;
            code += n;
            table.max_code[l] = code - 1;
            // A canonical code of length l+1 may not overflow its bit width.
            if code > (1 << (l + 1)) {
                return Err(JpegError::BadTable("Huffman"));
            }
            code <<= 1;
        }
        if k != table.values.len() {
            return Err(JpegError::BadTable("Huffman"));
        }

        // Fill the direct lookup table. A code of length `l` fixes the top `l`
        // bits, so it owns every prefix that starts with it.
        let mut code: u32 = 0;
        let mut idx = 0usize;
        for l in 1..=HUFF_LUT_BITS as usize {
            for _ in 0..counts[l] {
                let shift = HUFF_LUT_BITS as usize - l;
                let base = (code as usize) << shift;
                let sym = table.values[idx];
                for slot in &mut table.lut[base..base + (1 << shift)] {
                    *slot = (sym, l as u8);
                }
                idx += 1;
                code += 1;
            }
            code <<= 1;
        }

        Ok(table)
    }
}

/// Reads MSB-first bits out of an entropy-coded segment, undoing the `FF 00`
/// byte stuffing as it goes.
struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute byte position within `data`.
    pos: usize,
    /// Bit accumulator, MSB-aligned within `bits_left`.
    accum: u64,
    bits_left: u32,
    /// Bits fabricated past the end of the segment by `fill`. They sit at the
    /// bottom of `accum` and must not be mistaken for real buffered data when
    /// reporting positions or padding.
    invented: u32,
    /// Set once a marker (an unstuffed `FF xx`) has been reached.
    hit_marker: bool,
    /// Position of the `FF` byte that terminated the segment.
    marker_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            accum: 0,
            bits_left: 0,
            invented: 0,
            hit_marker: false,
            marker_pos: pos,
        }
    }

    /// Pulls one unstuffed byte out of the stream, or `None` at a marker.
    fn next_byte(&mut self) -> Option<u8> {
        if self.hit_marker || self.pos >= self.data.len() {
            return None;
        }
        let b = self.data[self.pos];
        if b == 0xFF {
            let next = self.data.get(self.pos + 1).copied();
            match next {
                // Stuffed literal 0xFF.
                Some(0x00) => {
                    self.pos += 2;
                    Some(0xFF)
                }
                // A fill byte run is legal padding before a marker.
                Some(0xFF) => {
                    self.hit_marker = true;
                    self.marker_pos = self.pos;
                    None
                }
                Some(_) => {
                    self.hit_marker = true;
                    self.marker_pos = self.pos;
                    None
                }
                None => {
                    self.hit_marker = true;
                    self.marker_pos = self.pos;
                    None
                }
            }
        } else {
            self.pos += 1;
            Some(b)
        }
    }

    fn fill(&mut self, need: u32) {
        while self.bits_left < need {
            match self.next_byte() {
                Some(b) => {
                    self.accum = (self.accum << 8) | b as u64;
                    self.bits_left += 8;
                }
                // Past the end of the segment the spec lets us invent 1 bits;
                // a conforming stream never actually consumes them.
                None => {
                    self.accum = (self.accum << 8) | 0xFF;
                    self.bits_left += 8;
                    self.invented += 8;
                }
            }
        }
    }

    fn get_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        self.fill(n);
        self.bits_left -= n;
        ((self.accum >> self.bits_left) & ((1u64 << n) - 1)) as u32
    }

    fn get_bit(&mut self) -> u32 {
        self.get_bits(1)
    }

    fn decode_huff(&mut self, table: &HuffTable) -> Result<u8, JpegError> {
        if !table.present {
            return Err(JpegError::BadEntropyData);
        }

        // Fast path: resolve the code from its first `HUFF_LUT_BITS` bits.
        // Nothing is consumed unless the lookup succeeds, so the fallback below
        // still sees the code from its first bit.
        self.fill(HUFF_LUT_BITS);
        let peek = ((self.accum >> (self.bits_left - HUFF_LUT_BITS)) & 0xFF) as usize;
        let (sym, len) = table.lut[peek];
        if len != 0 {
            self.bits_left -= len as u32;
            return Ok(sym);
        }

        let mut code: i32 = 0;
        for l in 0..HUFF_MAX_BIT_LENGTH {
            code = (code << 1) | self.get_bit() as i32;
            if table.max_code[l] >= code && code >= table.min_code[l] {
                let idx = table.val_ptr[l] + (code - table.min_code[l]) as usize;
                return table
                    .values
                    .get(idx)
                    .copied()
                    .ok_or(JpegError::BadEntropyData);
            }
        }
        Err(JpegError::BadEntropyData)
    }

    /// Sign-extends an `n`-bit magnitude as JPEG's `EXTEND` procedure does.
    fn receive_extend(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        let v = self.get_bits(n) as i32;
        if v < (1 << (n - 1)) {
            v - (1 << n) + 1
        } else {
            v
        }
    }

    /// Number of whole bits still buffered but unconsumed, and the value of the
    /// padding bits that would round the stream up to a byte boundary.
    fn padding_bits(&self) -> (u32, u32) {
        let pad = self.real_bits() % 8;
        if pad == 0 {
            (0, 0)
        } else {
            (
                pad,
                ((self.accum >> (self.bits_left - pad)) & ((1u64 << pad) - 1)) as u32,
            )
        }
    }

    /// Discards buffered bits back to a byte boundary, returning the position
    /// of the next unread byte.
    fn byte_position(&self) -> usize {
        self.pos - (self.real_bits() / 8) as usize
    }

    /// Buffered bits that actually came from the stream.
    fn real_bits(&self) -> u32 {
        self.bits_left.saturating_sub(self.invented)
    }
}

/// Largest image area accepted, matching the JPEG XL level-5 limit.
const MAX_PIXELS: usize = 1 << 28;

const M_SOF0: u8 = 0xC0;
const M_SOF1: u8 = 0xC1;
const M_SOF2: u8 = 0xC2;
const M_DHT: u8 = 0xC4;
const M_SOI: u8 = 0xD8;
const M_EOI: u8 = 0xD9;
const M_SOS: u8 = 0xDA;
const M_DQT: u8 = 0xDB;
const M_DRI: u8 = 0xDD;
const M_COM: u8 = 0xFE;

/// `IS_VALID_MARKER[m - 0xC0]` is true when `m` may begin a segment.
///
/// Reconstruction encodes each marker as six bits of `marker - 0xC0`, so only
/// this range is representable at all; anything else has to be carried as
/// inter-marker data instead.
static IS_VALID_MARKER: [bool; 64] = {
    let mut t = [false; 64];
    let valid = [
        0xC0u8, 0xC1, 0xC2, 0xC4, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD9, 0xDA, 0xDB,
        0xDD, 0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED,
        0xEE, 0xEF, 0xFE,
    ];
    let mut i = 0;
    while i < valid.len() {
        t[(valid[i] - 0xC0) as usize] = true;
        i += 1;
    }
    t
};

/// Counts the bytes from `pos` up to the next valid marker.
fn find_next_marker(data: &[u8], mut pos: usize) -> usize {
    let start = pos;
    while pos + 1 < data.len() {
        let is_marker = data[pos] == 0xFF
            && data[pos + 1] >= 0xC0
            && IS_VALID_MARKER[(data[pos + 1] - 0xC0) as usize];
        if is_marker {
            break;
        }
        pos += 1;
    }
    pos - start
}

/// Parses a complete JPEG file into a [`JpegData`].
pub(crate) fn parse_jpeg(data: &[u8]) -> Result<JpegData, JpegError> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != M_SOI {
        return Err(JpegError::NotAJpeg);
    }

    // The SOI is implicit and is deliberately absent from `marker_order`.
    let mut jpg = JpegData::default();

    // Working Huffman table slots: 0..3 DC, 4..7 AC.
    let mut dc_tables: [HuffTable; 4] = Default::default();
    let mut ac_tables: [HuffTable; 4] = Default::default();
    let mut seen_sof = false;
    let mut seen_sos = false;
    let mut pos = 2usize;

    loop {
        // Anything that is not a valid marker is arbitrary in-between data,
        // flagged by a synthetic 0xFF entry in the marker order.
        let skipped = find_next_marker(data, pos);
        if skipped > 0 {
            jpg.marker_order.push(0xFF);
            jpg.inter_marker_data
                .push(data[pos..pos + skipped].to_vec());
            pos += skipped;
        }
        if pos + 1 >= data.len() {
            return Err(JpegError::Truncated);
        }
        let marker = data[pos + 1];
        pos += 2;

        match marker {
            M_EOI => {
                jpg.marker_order.push(marker);
                jpg.tail_data = data[pos..].to_vec();
                break;
            }
            M_SOF0 | M_SOF1 | M_SOF2 => {
                if seen_sof {
                    return Err(JpegError::UnexpectedMarker(marker));
                }
                seen_sof = true;
                jpg.is_progressive = marker == M_SOF2;
                pos = process_sof(data, pos, &mut jpg)?;
            }
            M_DHT => pos = process_dht(data, pos, &mut jpg, &mut dc_tables, &mut ac_tables)?,
            M_DQT => pos = process_dqt(data, pos, &mut jpg)?,
            M_DRI => pos = process_dri(data, pos, &mut jpg)?,
            M_SOS => {
                if !seen_sof {
                    return Err(JpegError::UnexpectedMarker(marker));
                }
                seen_sos = true;
                pos = process_sos(data, pos, &mut jpg, &dc_tables, &ac_tables)?;
            }
            0xE0..=0xEF => pos = capture_segment(data, pos, marker, &mut jpg.app_data)?,
            M_COM => pos = capture_segment(data, pos, marker, &mut jpg.com_data)?,
            // Restart markers outside a scan carry no payload.
            0xD0..=0xD7 => {}
            other => return Err(JpegError::UnexpectedMarker(other)),
        }
        jpg.marker_order.push(marker);
    }

    if !seen_sof || jpg.components.is_empty() {
        return Err(JpegError::BadFrameHeader);
    }
    if !seen_sos {
        return Err(JpegError::UnexpectedMarker(M_SOS));
    }
    if jpg.huffman_code.is_empty() {
        return Err(JpegError::BadTable("Huffman"));
    }

    // Components reference quantization tables by DQT slot while parsing, but
    // everything downstream indexes the `quant` vector directly. Rewrite them
    // in place, resolving each slot to the first table that claims it.
    for i in 0..jpg.components.len() {
        let slot = jpg.components[i].quant_idx;
        let resolved = jpg
            .quant
            .iter()
            .position(|q| q.index == slot)
            .ok_or(JpegError::BadTable("quantization"))?;
        jpg.components[i].quant_idx = resolved as u32;
    }

    Ok(jpg)
}

/// Reads a length-prefixed segment and stores it as
/// `[marker, len_hi, len_lo, payload..]`.
fn capture_segment(
    data: &[u8],
    pos: usize,
    marker: u8,
    sink: &mut Vec<Vec<u8>>,
) -> Result<usize, JpegError> {
    let len = read_u16(data, pos)? as usize;
    if len < 2 {
        return Err(JpegError::BadSegmentLength(marker));
    }
    if pos + len > data.len() {
        return Err(JpegError::Truncated);
    }
    let mut seg = Vec::with_capacity(len + 1);
    seg.push(marker);
    seg.extend_from_slice(&data[pos..pos + len]);
    sink.push(seg);
    Ok(pos + len)
}

fn read_u16(data: &[u8], pos: usize) -> Result<u16, JpegError> {
    if pos + 1 >= data.len() {
        return Err(JpegError::Truncated);
    }
    Ok(((data[pos] as u16) << 8) | data[pos + 1] as u16)
}

fn process_sof(data: &[u8], pos: usize, jpg: &mut JpegData) -> Result<usize, JpegError> {
    let len = read_u16(data, pos)? as usize;
    let end = pos + len;
    if len < 8 || end > data.len() {
        return Err(JpegError::BadFrameHeader);
    }
    let mut p = pos + 2;
    let precision = data[p];
    if precision != 8 {
        return Err(JpegError::UnsupportedMode(
            "sample precision other than 8-bit",
        ));
    }
    p += 1;
    let height = read_u16(data, p)? as usize;
    let width = read_u16(data, p + 2)? as usize;
    p += 4;
    let num_comp = data[p] as usize;
    p += 1;
    if width == 0 || height == 0 {
        return Err(JpegError::BadFrameHeader);
    }
    // Coefficient planes are sized from these numbers alone, long before we
    // know whether any entropy-coded data actually follows. A seven-byte frame
    // header claiming 65535x65535 would otherwise ask for tens of gigabytes, so
    // cap the area at JXL's own level-5 limit rather than trusting the input.
    if width.saturating_mul(height) > MAX_PIXELS {
        return Err(JpegError::UnsupportedMode("image area exceeds 2^28 pixels"));
    }
    if num_comp == 0 || num_comp > MAX_COMPONENTS {
        return Err(JpegError::UnsupportedMode("component count outside 1..=4"));
    }
    if p + num_comp * 3 > end {
        return Err(JpegError::BadFrameHeader);
    }

    jpg.width = width;
    jpg.height = height;
    jpg.components = Vec::with_capacity(num_comp);
    for _ in 0..num_comp {
        let id = data[p] as u32;
        let h = (data[p + 1] >> 4) as usize;
        let v = (data[p + 1] & 0x0F) as usize;
        let tq = data[p + 2] as u32;
        p += 3;
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err(JpegError::BadFrameHeader);
        }
        if tq > 3 {
            return Err(JpegError::BadFrameHeader);
        }
        jpg.components.push(JpegComponent {
            id,
            h_samp_factor: h,
            v_samp_factor: v,
            quant_idx: tq,
            ..Default::default()
        });
    }
    // Duplicate component ids would make scan headers ambiguous.
    for i in 0..jpg.components.len() {
        let c_component = &jpg.components[i];
        for j in jpg.components[(i + 1)..jpg.components.len()].iter() {
            if c_component.id == j.id {
                return Err(JpegError::BadFrameHeader);
            }
        }
    }

    // Allocate coefficient planes, sized in whole MCUs.
    let max_h = jpg.max_h_samp();
    let max_v = jpg.max_v_samp();
    let mcus_x = width.div_ceil(max_h * 8);
    let mcus_y = height.div_ceil(max_v * 8);
    for c in jpg.components.iter_mut() {
        c.width_in_blocks = mcus_x * c.h_samp_factor;
        c.height_in_blocks = mcus_y * c.v_samp_factor;
        let n = c
            .width_in_blocks
            .checked_mul(c.height_in_blocks)
            .and_then(|b| b.checked_mul(DCT_BLOCK_SIZE))
            .ok_or(JpegError::BadFrameHeader)?;
        c.coeffs = vec![0i16; n];
    }
    Ok(end)
}

fn process_dqt(data: &[u8], pos: usize, jpg: &mut JpegData) -> Result<usize, JpegError> {
    let len = read_u16(data, pos)? as usize;
    let end = pos + len;
    if len < 2 || end > data.len() {
        return Err(JpegError::BadTable("quantization"));
    }
    let mut p = pos + 2;
    let first = jpg.quant.len();
    while p < end {
        let pq = (data[p] >> 4) as u32;
        let tq = (data[p] & 0x0F) as u32;
        p += 1;
        if pq > 1 || tq > 3 {
            return Err(JpegError::BadTable("quantization"));
        }
        let need = if pq == 1 { 128 } else { 64 };
        if p + need > end {
            return Err(JpegError::BadTable("quantization"));
        }
        let mut values = [0i32; DCT_BLOCK_SIZE];
        for k in 0..DCT_BLOCK_SIZE {
            let v = if pq == 1 {
                let hi = data[p + 2 * k] as i32;
                let lo = data[p + 2 * k + 1] as i32;
                (hi << 8) | lo
            } else {
                data[p + k] as i32
            };
            if v == 0 {
                return Err(JpegError::BadTable("quantization"));
            }
            // Stored zig-zagged in the file; keep natural order internally.
            values[NATURAL_ORDER[k]] = v;
        }
        p += need;
        jpg.quant.push(JpegQuantTable {
            values,
            precision: pq,
            index: tq,
            is_last: false,
        });
    }
    if jpg.quant.len() == first {
        return Err(JpegError::BadTable("quantization"));
    }
    if let Some(last) = jpg.quant.last_mut() {
        last.is_last = true;
    }
    Ok(end)
}

fn process_dht(
    data: &[u8],
    pos: usize,
    jpg: &mut JpegData,
    dc_tables: &mut [HuffTable; 4],
    ac_tables: &mut [HuffTable; 4],
) -> Result<usize, JpegError> {
    let len = read_u16(data, pos)? as usize;
    let end = pos + len;
    if len < 2 || end > data.len() {
        return Err(JpegError::BadTable("Huffman"));
    }
    let mut p = pos + 2;
    let first = jpg.huffman_code.len();
    while p < end {
        if p + 17 > end {
            return Err(JpegError::BadTable("Huffman"));
        }
        let slot_id = data[p] as u32;
        let tc = slot_id >> 4;
        let th = slot_id & 0x0F;
        if tc > 1 || th > 3 {
            return Err(JpegError::BadTable("Huffman"));
        }
        p += 1;
        let mut code = JpegHuffmanCode {
            slot_id,
            is_last: false,
            ..Default::default()
        };
        let mut total = 0usize;
        for l in 1..=HUFF_MAX_BIT_LENGTH {
            let n = data[p + l - 1] as u32;
            code.counts[l] = n;
            total += n as usize;
        }
        p += HUFF_MAX_BIT_LENGTH;
        if total == 0 || total > HUFF_ALPHABET_SIZE || p + total > end {
            return Err(JpegError::BadTable("Huffman"));
        }
        let mut values = Vec::with_capacity(total);
        for (i, &v) in data[p..p + total].iter().enumerate() {
            // DC values are magnitudes 0..=15; AC symbols use the full byte.
            if tc == 0 && v > 15 {
                return Err(JpegError::BadTable("Huffman"));
            }
            code.values[i] = v as u32;
            values.push(v);
        }
        p += total;

        // Decoding uses the table exactly as it appeared in the file.
        let built = HuffTable::build(&code.counts, values)?;
        if tc == 0 {
            dc_tables[th as usize] = built;
        } else {
            ac_tables[th as usize] = built;
        }

        // Reconstruction, however, expects libjxl's convention: a sentinel
        // symbol occupying the all-ones code is appended at the deepest
        // populated level, so the code space is provably full.
        let max_depth = (1..=HUFF_MAX_BIT_LENGTH)
            .rev()
            .find(|&l| code.counts[l] != 0)
            .ok_or(JpegError::BadTable("Huffman"))?;
        code.counts[max_depth] += 1;
        code.values[total] = HUFF_ALPHABET_SIZE as u32;

        jpg.huffman_code.push(code);
    }
    if jpg.huffman_code.len() == first {
        return Err(JpegError::BadTable("Huffman"));
    }
    if let Some(last) = jpg.huffman_code.last_mut() {
        last.is_last = true;
    }
    Ok(end)
}

fn process_dri(data: &[u8], pos: usize, jpg: &mut JpegData) -> Result<usize, JpegError> {
    let len = read_u16(data, pos)? as usize;
    if len != 4 || pos + len > data.len() {
        return Err(JpegError::BadSegmentLength(M_DRI));
    }
    jpg.restart_interval = read_u16(data, pos + 2)? as u32;
    Ok(pos + len)
}

/// Parses an SOS header and then decodes the entropy-coded segment that
/// follows it, accumulating coefficients into the component planes.
fn process_sos(
    data: &[u8],
    pos: usize,
    jpg: &mut JpegData,
    dc_tables: &[HuffTable; 4],
    ac_tables: &[HuffTable; 4],
) -> Result<usize, JpegError> {
    let len = read_u16(data, pos)? as usize;
    let end = pos + len;
    if len < 6 || end > data.len() {
        return Err(JpegError::BadScanHeader);
    }
    let mut p = pos + 2;
    let ns = data[p] as usize;
    p += 1;
    if ns == 0 || ns > MAX_COMPONENTS || p + ns * 2 + 3 > end {
        return Err(JpegError::BadScanHeader);
    }

    let mut scan = JpegScanInfo {
        num_components: ns as u32,
        ..Default::default()
    };
    for i in 0..ns {
        let cs = data[p] as u32;
        let td = (data[p + 1] >> 4) as u32;
        let ta = (data[p + 1] & 0x0F) as u32;
        p += 2;
        let comp_idx = jpg
            .components
            .iter()
            .position(|c| c.id == cs)
            .ok_or(JpegError::BadScanHeader)?;
        if td > 3 || ta > 3 {
            return Err(JpegError::BadScanHeader);
        }
        // A component may appear at most once per scan.
        if scan.components[..i]
            .iter()
            .any(|c| c.comp_idx == comp_idx as u32)
        {
            return Err(JpegError::BadScanHeader);
        }
        scan.components[i] = JpegComponentScanInfo {
            comp_idx: comp_idx as u32,
            dc_tbl_idx: td,
            ac_tbl_idx: ta,
        };
    }
    scan.ss = data[p] as u32;
    scan.se = data[p + 1] as u32;
    scan.ah = (data[p + 2] >> 4) as u32;
    scan.al = (data[p + 2] & 0x0F) as u32;
    p += 3;

    if scan.se >= DCT_BLOCK_SIZE as u32 || scan.ss > scan.se {
        return Err(JpegError::BadScanHeader);
    }
    if scan.al > 13 || scan.ah > 13 {
        return Err(JpegError::BadScanHeader);
    }
    if !jpg.is_progressive && (scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0) {
        return Err(JpegError::BadScanHeader);
    }
    // Progressive AC scans are always single-component.
    if jpg.is_progressive && scan.ss != 0 && ns != 1 {
        return Err(JpegError::BadScanHeader);
    }

    let next = decode_scan(data, p, jpg, &mut scan, dc_tables, ac_tables)?;
    jpg.scan_info.push(scan);
    Ok(next)
}

/// State carried across restart intervals.
struct ScanState {
    dc_pred: [i32; MAX_COMPONENTS],
    /// End-of-block run left to consume. A value of `-1` marks a fresh start
    /// after a restart marker: the next EOB run there is not "back to back"
    /// with a previous one, and so must not produce a reset point.
    eob_run: i32,
    /// Sequential index of the block being decoded, counted across every
    /// component in the scan. Reset points and extra zero runs are keyed by it.
    block_scan_index: u32,
}

fn decode_scan(
    data: &[u8],
    start: usize,
    jpg: &mut JpegData,
    scan: &mut JpegScanInfo,
    dc_tables: &[HuffTable; 4],
    ac_tables: &[HuffTable; 4],
) -> Result<usize, JpegError> {
    let ns = scan.num_components as usize;
    let max_h = jpg.max_h_samp();
    let max_v = jpg.max_v_samp();
    let mcus_x = jpg.width.div_ceil(max_h * 8);
    let mcus_y = jpg.height.div_ceil(max_v * 8);

    // A single-component scan is addressed in that component's own blocks,
    // not in MCUs; the spec calls these "non-interleaved" scans.
    let (units_x, units_y) = if ns == 1 {
        let c = &jpg.components[scan.components[0].comp_idx as usize];
        let bw = jpg.width.div_ceil(8 * (max_h / c.h_samp_factor));
        let bh = jpg.height.div_ceil(8 * (max_v / c.v_samp_factor));
        (bw, bh)
    } else {
        (mcus_x, mcus_y)
    };

    let mut br = BitReader::new(data, start);
    let mut state = ScanState {
        dc_pred: [0; MAX_COMPONENTS],
        // Starts "fresh", exactly as after a restart: the first end-of-block
        // run in a scan has no predecessor and so is not a reset point.
        eob_run: -1,
        block_scan_index: 0,
    };
    let restart_interval = jpg.restart_interval;
    let mut units_until_restart = restart_interval;

    for uy in 0..units_y {
        for ux in 0..units_x {
            if restart_interval > 0 && units_until_restart == 0 {
                // Consume padding, then the RSTn marker itself.
                record_padding(jpg, &mut br);
                let bpos = br.byte_position();
                if bpos + 1 >= data.len() || data[bpos] != 0xFF {
                    return Err(JpegError::BadEntropyData);
                }
                let m = data[bpos + 1];
                if !(0xD0..=0xD7).contains(&m) {
                    return Err(JpegError::BadEntropyData);
                }
                br = BitReader::new(data, bpos + 2);
                state.dc_pred = [0; MAX_COMPONENTS];
                state.eob_run = -1;
                units_until_restart = restart_interval;
            }

            if ns == 1 {
                let csi = scan.components[0];
                let comp_idx = csi.comp_idx as usize;
                let bw = jpg.components[comp_idx].width_in_blocks;
                let block = uy * bw + ux;
                decode_block(
                    &mut br, jpg, scan, &mut state, csi, comp_idx, block, dc_tables, ac_tables,
                )?;
                state.block_scan_index += 1;
            } else {
                for i in 0..ns {
                    let csi = scan.components[i];
                    let comp_idx = csi.comp_idx as usize;
                    let (h, v, bw) = {
                        let c = &jpg.components[comp_idx];
                        (c.h_samp_factor, c.v_samp_factor, c.width_in_blocks)
                    };
                    for by in 0..v {
                        for bx in 0..h {
                            let block = (uy * v + by) * bw + ux * h + bx;
                            decode_block(
                                &mut br, jpg, scan, &mut state, csi, comp_idx, block, dc_tables,
                                ac_tables,
                            )?;
                            state.block_scan_index += 1;
                        }
                    }
                }
            }

            if restart_interval > 0 {
                units_until_restart -= 1;
            }
        }
    }

    record_padding(jpg, &mut br);
    Ok(br.byte_position())
}

/// Captures the bits padding the tail of an entropy-coded segment.
///
/// The standard pads with 1-bits, so reconstruction regenerates an all-ones
/// fill by default. Bits are always accumulated but only transmitted when a
/// group deviates, which is what `has_zero_padding_bit` records.
fn record_padding(jpg: &mut JpegData, br: &mut BitReader<'_>) {
    let (n, bits) = br.padding_bits();
    if n == 0 {
        return;
    }
    let all_ones = (1u32 << n) - 1;
    if bits != all_ones {
        jpg.has_zero_padding_bit = true;
    }
    for i in (0..n).rev() {
        jpg.padding_bits.push(((bits >> i) & 1) as u8);
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_block(
    br: &mut BitReader<'_>,
    jpg: &mut JpegData,
    scan: &mut JpegScanInfo,
    state: &mut ScanState,
    csi: JpegComponentScanInfo,
    comp_idx: usize,
    block: usize,
    dc_tables: &[HuffTable; 4],
    ac_tables: &[HuffTable; 4],
) -> Result<(), JpegError> {
    let comp = &jpg.components[comp_idx];
    let nblocks = comp.width_in_blocks * comp.height_in_blocks;
    if block >= nblocks {
        return Err(JpegError::BadEntropyData);
    }
    let is_progressive = jpg.is_progressive;
    let base = block * DCT_BLOCK_SIZE;
    // One bounds check for the whole block instead of one per coefficient; this
    // loop is the hot path of every scan.
    let coeffs: &mut [i16] = &mut jpg.components[comp_idx].coeffs[base..base + DCT_BLOCK_SIZE];

    let ss = scan.ss;
    let se = scan.se;
    let ah = scan.ah;
    let al = scan.al;

    if ss == 0 {
        // ---- DC ----
        if ah == 0 {
            let t = br.decode_huff(&dc_tables[csi.dc_tbl_idx as usize])?;
            if t > 15 {
                return Err(JpegError::BadEntropyData);
            }
            let diff = br.receive_extend(t as u32);
            let pred = state.dc_pred[comp_idx_slot(scan, comp_idx)]
                .checked_add(diff)
                .ok_or(JpegError::BadEntropyData)?;
            state.dc_pred[comp_idx_slot(scan, comp_idx)] = pred;
            coeffs[0] = narrow(pred << al)?;
        } else {
            // Refinement: one bit at position `al`.
            if br.get_bit() != 0 {
                coeffs[0] |= 1i16 << al;
            }
        }
        if se == 0 {
            return Ok(());
        }
    }

    // ---- AC ----
    if !is_progressive {
        let mut k = 1usize;
        while k <= se as usize {
            let sym = br.decode_huff(&ac_tables[csi.ac_tbl_idx as usize])?;
            let r = (sym >> 4) as usize;
            let s = (sym & 0x0F) as u32;
            if s == 0 {
                if r != 15 {
                    break; // EOB
                }
                k += 16;
                continue;
            }
            k += r;
            if k > 63 {
                return Err(JpegError::BadEntropyData);
            }
            let v = br.receive_extend(s);
            coeffs[NATURAL_ORDER[k]] = narrow(v)?;
            k += 1;
        }
        return Ok(());
    }

    let ac_start = ss.max(1) as usize;
    if ah == 0 {
        if state.eob_run > 0 {
            state.eob_run -= 1;
            return Ok(());
        }
        let mut k = ac_start;
        // Count trailing ZRLs that produce no coefficient, so the writer can
        // reproduce a non-minimal encoding verbatim.
        let mut num_zero_runs = 0u32;
        while k <= se as usize {
            let sym = br.decode_huff(&ac_tables[csi.ac_tbl_idx as usize])?;
            let r = (sym >> 4) as u32;
            let s = (sym & 0x0F) as u32;
            if s == 0 {
                if r != 15 {
                    // Two EOB runs back to back tell the re-encoder to force a
                    // state reset here so it reproduces this exact structure.
                    if k == ac_start && state.eob_run == 0 {
                        scan.reset_points.push(state.block_scan_index);
                    }
                    // EOB run of length 2^r + extra, counting this block; the
                    // decrement below accounts for it.
                    state.eob_run = 1i32 << r;
                    if r > 0 {
                        state.eob_run += br.get_bits(r) as i32;
                    }
                    break;
                }
                k += 16;
                num_zero_runs += 1;
                continue;
            }
            k += r as usize;
            if k > se as usize {
                return Err(JpegError::BadEntropyData);
            }
            let v = br.receive_extend(s);
            coeffs[NATURAL_ORDER[k]] = narrow(v << al)?;
            num_zero_runs = 0;
            k += 1;
        }
        if num_zero_runs > 0 {
            scan.extra_zero_runs.push(ExtraZeroRunInfo {
                block_idx: state.block_scan_index,
                num_extra_zero_runs: num_zero_runs,
            });
        }
        // Every block consumes one unit of the run. With no run active this
        // drives the counter negative, which is what keeps the "two runs back
        // to back" test above from firing on ordinary blocks.
        state.eob_run -= 1;
    } else {
        let p1: i16 = 1i16 << al;
        let m1: i16 = -1i16 << al;
        let mut k = ac_start;

        if state.eob_run <= 0 {
            while k <= se as usize {
                let sym = br.decode_huff(&ac_tables[csi.ac_tbl_idx as usize])?;
                let mut r = (sym >> 4) as i32;
                let s = (sym & 0x0F) as u32;
                let mut value = 0i16;
                if s == 0 {
                    if r != 15 {
                        if k == ac_start && state.eob_run == 0 {
                            scan.reset_points.push(state.block_scan_index);
                        }
                        state.eob_run = 1i32 << r;
                        if r > 0 {
                            state.eob_run += br.get_bits(r as u32) as i32;
                        }
                        break;
                    }
                    // r == 15: skip 16 zero-valued coefficients.
                } else {
                    value = if br.get_bit() != 0 { p1 } else { m1 };
                }
                // Skip `r` zero-history coefficients, correcting non-zero
                // ones passed on the way.
                while k <= se as usize {
                    let idx = NATURAL_ORDER[k];
                    let cur = coeffs[idx];
                    if cur != 0 {
                        if br.get_bit() != 0 && (cur & p1) == 0 {
                            coeffs[idx] = if cur >= 0 { cur + p1 } else { cur + m1 };
                        }
                    } else {
                        if r == 0 {
                            if value != 0 {
                                coeffs[idx] = value;
                            }
                            k += 1;
                            break;
                        }
                        r -= 1;
                    }
                    k += 1;
                }
            }
        }

        if state.eob_run > 0 {
            // Within an EOB run only the correction bits for already-nonzero
            // coefficients are present.
            while k <= se as usize {
                let idx = NATURAL_ORDER[k];
                let cur = coeffs[idx];
                if cur != 0 && br.get_bit() != 0 && (cur & p1) == 0 {
                    coeffs[idx] = if cur >= 0 { cur + p1 } else { cur + m1 };
                }
                k += 1;
            }
        }
        state.eob_run -= 1;
    }
    Ok(())
}

#[inline]
fn narrow(v: i32) -> Result<i16, JpegError> {
    i16::try_from(v).map_err(|_| JpegError::BadEntropyData)
}

/// DC predictors are indexed per scan component, not per image component.
fn comp_idx_slot(scan: &JpegScanInfo, comp_idx: usize) -> usize {
    for i in 0..scan.num_components as usize {
        if scan.components[i].comp_idx as usize == comp_idx {
            return i;
        }
    }
    0
}
