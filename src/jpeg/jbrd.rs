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

//! Serialization of the `jbrd` (JPEG bitstream reconstruction data) box.

use super::brotli::brotli_store;
use super::{HUFF_ALPHABET_SIZE, HUFF_MAX_BIT_LENGTH, JpegData, JpegError};
use crate::bit_writer::BitWriter;

/// One branch of a `U32Enc` distribution.
#[derive(Clone, Copy)]
enum Dist {
    /// Encodes exactly this value in the two selector bits alone.
    Val(u32),
    /// Encodes `offset .. offset + 2^bits` using `bits` extra bits.
    BitsOffset(u32, u32),
}

use Dist::{BitsOffset, Val};

/// `Bits(n)` is just an offset-free `BitsOffset`.
const fn bits(n: u32) -> Dist {
    BitsOffset(n, 0)
}

/// Writes `value` under a four-way `U32Enc` distribution.
///
/// Selector choice must match libjxl's `ChooseSelector` exactly: the first
/// exact `Val` match wins immediately, otherwise the in-range branch with the
/// fewest total bits wins, ties going to the lowest selector index.
fn write_u32(w: &mut BitWriter, dists: [Dist; 4], value: u32) -> Result<(), JpegError> {
    let mut best: Option<(u32, u32)> = None; // (selector, extra_bits)
    let mut total_bits = u32::MAX;

    for (s, d) in dists.iter().enumerate() {
        match *d {
            Val(v) => {
                if v == value {
                    w.write(2, s as u64);
                    return Ok(());
                }
            }
            BitsOffset(extra, offset) => {
                if value < offset {
                    continue;
                }
                // 2^extra can reach 2^32, so widen before comparing.
                if (value - offset) as u64 >= (1u64 << extra) {
                    continue;
                }
                if 2 + extra < total_bits {
                    total_bits = 2 + extra;
                    best = Some((s as u32, extra));
                }
            }
        }
    }

    let Some((selector, extra)) = best else {
        return Err(JpegError::UnsupportedMode(
            "value not representable in reconstruction data",
        ));
    };
    let offset = match dists[selector as usize] {
        BitsOffset(_, o) => o,
        Val(_) => unreachable!("selector resolved to a direct branch"),
    };
    w.write(2, selector as u64);
    w.write(extra as usize, (value - offset) as u64);
    Ok(())
}

fn write_bool(w: &mut BitWriter, v: bool) {
    w.write(1, v as u64);
}

/// How a component set maps onto the compact component-type encoding.
const TYPE_GRAY: u64 = 0;
const TYPE_YCBCR: u64 = 1;
const TYPE_RGB: u64 = 2;
const TYPE_CUSTOM: u64 = 3;

/// Serializes `jpg` into a complete `jbrd` box payload.
pub(crate) fn encode_jbrd(jpg: &JpegData) -> Result<Vec<u8>, JpegError> {
    let mut out = encode_fields(jpg)?;

    // Order is fixed: unknown-typed APP segments, then COM, then inter-marker
    // chunks, then the tail. Since every APP segment is tagged kUnknown by
    // `encode_fields`, all of them are included here.
    let mut payload = Vec::new();
    for app in &jpg.app_data {
        payload.extend_from_slice(app);
    }
    for com in &jpg.com_data {
        payload.extend_from_slice(com);
    }
    for chunk in &jpg.inter_marker_data {
        payload.extend_from_slice(chunk);
    }
    payload.extend_from_slice(&jpg.tail_data);

    out.extend_from_slice(&brotli_store(&payload));
    Ok(out)
}

/// Emits the bit-packed field section, zero-padded to a byte boundary. Kept
/// separate so it can be diffed against libjxl's, whose Brotli tail differs.
fn encode_fields(jpg: &JpegData) -> Result<Vec<u8>, JpegError> {
    let mut w = BitWriter::new();

    write_bool(&mut w, jpg.components.len() == 1);

    // Self-terminating: the reader stops at the EOI (0xD9) entry, so the list
    // must contain exactly one and it must be last.
    if jpg.marker_order.last() != Some(&0xD9) {
        return Err(JpegError::UnexpectedMarker(0xD9));
    }
    for (i, &m) in jpg.marker_order.iter().enumerate() {
        if m < 0xC0 {
            return Err(JpegError::UnexpectedMarker(m));
        }
        if m == 0xD9 && i + 1 != jpg.marker_order.len() {
            return Err(JpegError::UnexpectedMarker(m));
        }
        w.write(6, (m - 0xC0) as u64);
    }

    // Every APP segment is tagged `kUnknown`, so its bytes go verbatim into the
    // Brotli section. libjxl instead tags ICC/Exif/XMP and rebuilds them from
    // the codestream; storing them is slightly larger but always correct.
    for app in &jpg.app_data {
        write_u32(
            &mut w,
            [Val(0), Val(1), BitsOffset(1, 2), BitsOffset(2, 4)],
            0,
        )?;
        let len = checked_seg_len(app)?;
        w.write(16, len as u64);
    }

    for com in &jpg.com_data {
        let len = checked_seg_len(com)?;
        w.write(16, len as u64);
    }

    // Only 1..=3 tables are usable: the encoding has a slot for 4 but the
    // decoder rejects it outright.
    let num_quant = jpg.quant.len();
    if !(1..=3).contains(&num_quant) {
        return Err(JpegError::BadTable("quantization"));
    }
    write_u32(&mut w, [Val(1), Val(2), Val(3), Val(4)], num_quant as u32)?;
    for q in &jpg.quant {
        if q.precision > 1 || q.index > 3 {
            return Err(JpegError::BadTable("quantization"));
        }
        w.write(1, q.precision as u64);
        w.write(2, q.index as u64);
        write_bool(&mut w, q.is_last);
    }

    let ids: Vec<u32> = jpg.components.iter().map(|c| c.id).collect();
    let component_type = match ids.as_slice() {
        [1] => TYPE_GRAY,
        [1, 2, 3] => TYPE_YCBCR,
        // 'R', 'G', 'B' as component ids.
        [0x52, 0x47, 0x42] => TYPE_RGB,
        _ => TYPE_CUSTOM,
    };
    w.write(2, component_type);
    if component_type == TYPE_CUSTOM {
        // Only 1- and 3-component images are accepted in this encoding.
        if ids.len() != 1 && ids.len() != 3 {
            return Err(JpegError::UnsupportedMode(
                "component count not reconstructible",
            ));
        }
        write_u32(&mut w, [Val(1), Val(2), Val(3), Val(4)], ids.len() as u32)?;
        for &id in &ids {
            if id > 0xFF {
                return Err(JpegError::BadFrameHeader);
            }
            w.write(8, id as u64);
        }
    }
    for c in &jpg.components {
        if c.quant_idx as usize >= jpg.quant.len() {
            return Err(JpegError::BadTable("quantization"));
        }
        w.write(2, c.quant_idx as u64);
    }

    let num_huff = jpg.huffman_code.len() as u32;
    write_u32(
        &mut w,
        [
            Val(4),
            BitsOffset(3, 2),
            BitsOffset(4, 10),
            BitsOffset(6, 26),
        ],
        num_huff,
    )?;
    for hc in &jpg.huffman_code {
        write_bool(&mut w, (hc.slot_id >> 4) != 0);
        w.write(2, (hc.slot_id & 0x0F) as u64);
        write_bool(&mut w, hc.is_last);

        let mut num_symbols = 0usize;
        for i in 0..=HUFF_MAX_BIT_LENGTH {
            write_u32(
                &mut w,
                [Val(0), Val(1), BitsOffset(3, 2), bits(8)],
                hc.counts[i],
            )?;
            num_symbols += hc.counts[i] as usize;
        }
        // A table with no symbols at all encodes an empty DHT marker and
        // carries no value list.
        if num_symbols == 0 {
            continue;
        }
        if num_symbols > HUFF_ALPHABET_SIZE + 1 {
            return Err(JpegError::BadTable("Huffman"));
        }
        for i in 0..num_symbols {
            write_u32(
                &mut w,
                [
                    bits(2),
                    BitsOffset(2, 4),
                    BitsOffset(4, 8),
                    BitsOffset(8, 1),
                ],
                hc.values[i],
            )?;
        }
        // The parser appends this sentinel; its absence means the table was
        // built wrong and the decoder would reject the file.
        if hc.values[num_symbols - 1] != HUFF_ALPHABET_SIZE as u32 {
            return Err(JpegError::BadTable("Huffman"));
        }
    }

    // Note the field order: Al precedes Ah, and AC precedes DC, both inverted
    // relative to the JPEG marker layout.
    for scan in &jpg.scan_info {
        if scan.num_components == 0 || scan.num_components >= 4 {
            return Err(JpegError::BadScanHeader);
        }
        write_u32(
            &mut w,
            [Val(1), Val(2), Val(3), Val(4)],
            scan.num_components,
        )?;
        w.write(6, scan.ss as u64);
        w.write(6, scan.se as u64);
        w.write(4, scan.al as u64);
        w.write(4, scan.ah as u64);
        for i in 0..scan.num_components as usize {
            let c = &scan.components[i];
            w.write(2, c.comp_idx as u64);
            w.write(2, c.ac_tbl_idx as u64);
            w.write(2, c.dc_tbl_idx as u64);
        }
        write_u32(
            &mut w,
            [Val(0), Val(1), Val(2), BitsOffset(3, 3)],
            scan.last_needed_pass,
        )?;
    }

    // --- restart interval --------------------------------------------------
    // Present only when a DRI marker was actually seen.
    if jpg.marker_order.contains(&0xDD) {
        w.write(16, jpg.restart_interval as u64);
    }

    // A second pass over the same scans, after everything above.
    static COUNT_DIST: [Dist; 4] = [
        Val(0),
        BitsOffset(2, 1),
        BitsOffset(4, 4),
        BitsOffset(16, 20),
    ];
    static BLOCK_DIST: [Dist; 4] = [
        Val(0),
        BitsOffset(3, 1),
        BitsOffset(5, 9),
        BitsOffset(28, 41),
    ];

    for scan in &jpg.scan_info {
        write_u32(&mut w, COUNT_DIST, scan.reset_points.len() as u32)?;
        let mut last: i64 = -1;
        for &block_idx in &scan.reset_points {
            let delta = (block_idx as i64) - (last + 1);
            if delta < 0 {
                return Err(JpegError::BadEntropyData);
            }
            write_u32(&mut w, BLOCK_DIST, delta as u32)?;
            last = block_idx as i64;
        }

        write_u32(&mut w, COUNT_DIST, scan.extra_zero_runs.len() as u32)?;
        let mut last: i64 = -1;
        for ezr in &scan.extra_zero_runs {
            // Count first, then the delta-coded block index.
            write_u32(
                &mut w,
                [
                    Val(1),
                    BitsOffset(2, 2),
                    BitsOffset(4, 5),
                    BitsOffset(8, 20),
                ],
                ezr.num_extra_zero_runs,
            )?;
            let delta = (ezr.block_idx as i64) - (last + 1);
            if delta < 0 {
                return Err(JpegError::BadEntropyData);
            }
            write_u32(&mut w, BLOCK_DIST, delta as u32)?;
            last = ezr.block_idx as i64;
        }
    }

    for chunk in &jpg.inter_marker_data {
        if chunk.len() > u16::MAX as usize {
            return Err(JpegError::UnsupportedMode("inter-marker chunk too large"));
        }
        w.write(16, chunk.len() as u64);
    }

    const MAX_TAIL: usize = 4_260_096;
    if jpg.tail_data.len() > MAX_TAIL {
        return Err(JpegError::UnsupportedMode("trailing data too large"));
    }
    write_u32(
        &mut w,
        [
            Val(0),
            BitsOffset(8, 1),
            BitsOffset(16, 257),
            BitsOffset(22, 65793),
        ],
        jpg.tail_data.len() as u32,
    )?;

    write_bool(&mut w, jpg.has_zero_padding_bit);
    if jpg.has_zero_padding_bit {
        if jpg.padding_bits.len() >= 1 << 24 {
            return Err(JpegError::UnsupportedMode("too many padding bits"));
        }
        w.write(24, jpg.padding_bits.len() as u64);
        for &bit in &jpg.padding_bits {
            write_bool(&mut w, bit != 0);
        }
    }

    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// APP/COM segments are stored with their marker byte, so the transmitted
/// length is one less than the stored size — and must fit 16 bits.
fn checked_seg_len(seg: &[u8]) -> Result<usize, JpegError> {
    if seg.len() < 3 {
        return Err(JpegError::BadSegmentLength(
            seg.first().copied().unwrap_or(0),
        ));
    }
    let len = seg.len() - 1;
    if len > u16::MAX as usize {
        return Err(JpegError::BadSegmentLength(seg[0]));
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads back a `U32Enc`-coded value, mirroring libjxl's reader.
    fn read_u32(bits_seq: &[u8], pos: &mut usize, dists: [Dist; 4]) -> u32 {
        let mut take = |n: usize| -> u32 {
            let mut v = 0u32;
            for i in 0..n {
                v |= (bits_seq[*pos + i] as u32) << i;
            }
            *pos += n;
            v
        };
        let selector = take(2) as usize;
        match dists[selector] {
            Val(v) => v,
            BitsOffset(extra, offset) => take(extra as usize) + offset,
        }
    }

    fn to_bit_seq(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for &b in bytes {
            for i in 0..8 {
                out.push((b >> i) & 1);
            }
        }
        out
    }

    /// The selector rules are subtle enough (direct-wins, then fewest bits,
    /// ties to the lowest index) that a round-trip check is worthwhile.
    #[test]
    fn u32_roundtrips_across_distributions() {
        let cases: Vec<([Dist; 4], Vec<u32>)> = vec![
            (
                [Val(0), Val(1), BitsOffset(3, 2), bits(8)],
                vec![0, 1, 2, 9, 10, 200, 255],
            ),
            (
                [
                    bits(2),
                    BitsOffset(2, 4),
                    BitsOffset(4, 8),
                    BitsOffset(8, 1),
                ],
                vec![0, 1, 3, 4, 7, 8, 23, 24, 256],
            ),
            (
                [
                    Val(4),
                    BitsOffset(3, 2),
                    BitsOffset(4, 10),
                    BitsOffset(6, 26),
                ],
                vec![2, 4, 5, 9, 10, 25, 26, 89],
            ),
            (
                [
                    Val(0),
                    BitsOffset(8, 1),
                    BitsOffset(16, 257),
                    BitsOffset(22, 65793),
                ],
                vec![0, 1, 256, 257, 65792, 65793, 4_260_096],
            ),
            (
                [
                    Val(0),
                    BitsOffset(3, 1),
                    BitsOffset(5, 9),
                    BitsOffset(28, 41),
                ],
                vec![0, 1, 8, 9, 40, 41, 1_000_000],
            ),
        ];

        for (dists, values) in cases {
            for v in values {
                let mut w = BitWriter::new();
                write_u32(&mut w, dists, v).expect("value should be representable");
                let nbits = w.bits_written();
                w.zero_pad_to_byte();
                let seq = to_bit_seq(&w.into_bytes());
                let mut pos = 0;
                let got = read_u32(&seq, &mut pos, dists);
                assert_eq!(got, v, "round-trip failed for {v}");
                assert_eq!(pos, nbits, "consumed {pos} bits, wrote {nbits}, for {v}");
            }
        }
    }

    /// Direct branches must win outright, even when a wider branch also covers
    /// the value — otherwise the bit count silently diverges from libjxl.
    #[test]
    fn u32_prefers_direct_then_shortest() {
        // 4 is covered by Val(4) and by BitsOffset(3,2); Val must win (2 bits).
        let mut w = BitWriter::new();
        write_u32(
            &mut w,
            [
                Val(4),
                BitsOffset(3, 2),
                BitsOffset(4, 10),
                BitsOffset(6, 26),
            ],
            4,
        )
        .unwrap();
        assert_eq!(w.bits_written(), 2);

        // 2 is covered by BitsOffset(3,2) (5 bits) and Bits(8) (10 bits).
        let mut w = BitWriter::new();
        write_u32(&mut w, [Val(0), Val(1), BitsOffset(3, 2), bits(8)], 2).unwrap();
        assert_eq!(w.bits_written(), 5);

        // 8 fits BitsOffset(4,8) (6 bits) and BitsOffset(8,1) (10 bits).
        let mut w = BitWriter::new();
        write_u32(
            &mut w,
            [
                bits(2),
                BitsOffset(2, 4),
                BitsOffset(4, 8),
                BitsOffset(8, 1),
            ],
            8,
        )
        .unwrap();
        assert_eq!(w.bits_written(), 6);
    }

    #[test]
    fn u32_rejects_unrepresentable() {
        assert!(write_u32(&mut BitWriter::new(), [Val(1), Val(2), Val(3), Val(4)], 7).is_err());
    }
}
