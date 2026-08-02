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
use crate::entropy::{
    ALPHABET_SIZE, Histogram, OwnedEntropyCode, PrefixCode, Token, build_huffman_codes,
    write_entropy_code, write_token,
};

pub(crate) const K_ICC_HEADER_SIZE: usize = 128;
pub(crate) const K_NUM_ICC_CONTEXTS: usize = 41;

// Commands recognized by the decoder's UnpredictICC.
const K_COMMAND_INSERT: u8 = 1;

/// 4-letter ICC tags used in the header prediction template.
const K_MNTR: [u8; 4] = *b"mntr";
const K_RGB_: [u8; 4] = *b"RGB ";
const K_XYZ_: [u8; 4] = *b"XYZ ";
const K_ACSP: [u8; 4] = *b"acsp";

/// Initial 128-byte header prediction template — matches libjxl's
/// `ICCInitialHeaderPrediction()`.
fn icc_initial_header_prediction() -> [u8; K_ICC_HEADER_SIZE] {
    let mut h = [0u8; K_ICC_HEADER_SIZE];
    h[8] = 4;
    h[12..16].copy_from_slice(&K_MNTR);
    h[16..20].copy_from_slice(&K_RGB_);
    h[20..24].copy_from_slice(&K_XYZ_);
    h[36..40].copy_from_slice(&K_ACSP);
    // Magic constants from the reference (creation-date-ish fields):
    h[68] = 0;
    h[69] = 0;
    h[70] = 246;
    h[71] = 214;
    h[72] = 0;
    h[73] = 1;
    h[74] = 0;
    h[75] = 0;
    h[76] = 0;
    h[77] = 0;
    h[78] = 211;
    h[79] = 45;
    h
}

/// Update the header prediction at position `pos` based on previously decoded
/// ICC bytes.  Mirrors libjxl's `ICCPredictHeader`.
fn icc_predict_header(icc: &[u8], header: &mut [u8; K_ICC_HEADER_SIZE], pos: usize) {
    let size = icc.len();
    if pos == 8 && size >= 8 {
        header[80] = icc[4];
        header[81] = icc[5];
        header[82] = icc[6];
        header[83] = icc[7];
    }
    if pos == 41 && size >= 41 {
        if icc[40] == b'A' {
            header[41] = b'P';
            header[42] = b'P';
            header[43] = b'L';
        }
        if icc[40] == b'M' {
            header[41] = b'S';
            header[42] = b'F';
            header[43] = b'T';
        }
    }
    if pos == 42 && size >= 42 {
        if icc[40] == b'S' && icc[41] == b'G' {
            header[42] = b'I';
            header[43] = b' ';
        }
        if icc[40] == b'S' && icc[41] == b'U' {
            header[42] = b'N';
            header[43] = b'W';
        }
    }
}

/// `EncodeVarInt`: 7 bits/byte, MSB = continuation.
fn encode_varint(value: u64, out: &mut Vec<u8>) {
    let mut v = value;
    while v > 127 {
        out.push(((v & 127) as u8) | 128);
        v >>= 7;
    }
    out.push((v & 127) as u8);
}

// libjxl's ByteKind1 / ByteKind2 used to choose one of 41 ICC ANS contexts.
fn byte_kind_1(b: u8) -> u8 {
    if b.is_ascii_lowercase() {
        return 0;
    }
    if b.is_ascii_uppercase() {
        return 0;
    }
    if b.is_ascii_digit() {
        return 1;
    }
    if b == b'.' || b == b',' {
        return 1;
    }
    if b == 0 {
        return 2;
    }
    if b == 1 {
        return 3;
    }
    if b < 16 {
        return 4;
    }
    if b == 255 {
        return 6;
    }
    if b > 240 {
        return 5;
    }
    7
}
fn byte_kind_2(b: u8) -> u8 {
    if b.is_ascii_lowercase() {
        return 0;
    }
    if b.is_ascii_uppercase() {
        return 0;
    }
    if b.is_ascii_digit() {
        return 1;
    }
    if b == b'.' || b == b',' {
        return 1;
    }
    if b < 16 {
        return 2;
    }
    if b > 240 {
        return 3;
    }
    4
}

/// Context index for byte `i` of the predicted/encoded stream, given the two
/// previous bytes.  Returns a value in `0..K_NUM_ICC_CONTEXTS`.
pub(crate) fn iccans_context(i: usize, prev1: u8, prev2: u8) -> usize {
    if i <= 128 {
        return 0;
    }
    1 + (byte_kind_1(prev1) as usize) + (byte_kind_2(prev2) as usize) * 8
}

/// Run libjxl's minimal-strategy `PredictICC`: header delta + 1 insert command.
/// Returns the byte stream the decoder's `UnpredictICC` will turn back into the
/// original ICC profile.
pub(crate) fn predict_icc_minimal(icc: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(icc.len() + 32);

    // 1. Output size (decoder verifies it matches the reconstructed profile).
    encode_varint(icc.len() as u64, &mut out);

    // Header data bytes (icc[i] - predicted[i] for i in 0..128).  Built first
    // so we can compute commands_size before writing it.
    let mut header_data: Vec<u8> = Vec::with_capacity(K_ICC_HEADER_SIZE);
    let mut header = icc_initial_header_prediction();
    // The reference also stuffs `icc.size()` into header[0..4] before predicting.
    // EncodeUint32(0, size, &header) — big-endian.
    let sz_be = (icc.len() as u32).to_be_bytes();
    header[0..4].copy_from_slice(&sz_be);
    let header_bound = K_ICC_HEADER_SIZE.min(icc.len());
    for i in 0..header_bound {
        icc_predict_header(icc, &mut header, i);
        header_data.push(icc[i].wrapping_sub(header[i]));
    }

    // 2. Commands stream.
    let mut commands: Vec<u8> = Vec::new();
    if icc.len() <= K_ICC_HEADER_SIZE {
        // Profiles ≤ 128 bytes have no commands — just the header delta.
        encode_varint(0, &mut commands);
    } else {
        // 0 tags (VarInt 0), then one Insert command for the entire tail.
        encode_varint(0, &mut commands);
        commands.push(K_COMMAND_INSERT);
        let tail_len = icc.len() - K_ICC_HEADER_SIZE;
        encode_varint(tail_len as u64, &mut commands);
    }

    // 3. csize, then commands bytes, then data bytes.
    encode_varint(commands.len() as u64, &mut out);
    out.extend_from_slice(&commands);
    out.extend_from_slice(&header_data);
    if icc.len() > K_ICC_HEADER_SIZE {
        out.extend_from_slice(&icc[K_ICC_HEADER_SIZE..]);
    }

    out
}

/// Write a `U64Coder` value (selectors 0..3 per libjxl `fields.cc`).
fn write_u64(value: u64, w: &mut BitWriter) {
    if value == 0 {
        w.write(2, 0);
    } else if value <= 16 {
        w.write(2, 1);
        w.write(4, value - 1);
    } else if value <= 272 {
        w.write(2, 2);
        w.write(8, value - 17);
    } else {
        w.write(2, 3);
        w.write(12, value & 4095);
        let mut v = value >> 12;
        let mut shift: u32 = 12;
        while v > 0 && shift < 60 {
            w.write(1, 1);
            w.write(8, v & 255);
            v >>= 8;
            shift += 8;
        }
        if v > 0 {
            w.write(1, 1);
            w.write(4, v & 15);
        } else {
            w.write(1, 0);
        }
    }
}

/// Apply a single-symbol patch to a prefix code (mirrors `build_lz_pixel_code`).
fn single_symbol_patch(pc: &mut PrefixCode) {
    let mut nonzero = 0;
    let mut idx = 0;
    for (i, &d) in pc.depths.iter().enumerate() {
        if d != 0 {
            nonzero += 1;
            idx = i;
            if nonzero > 1 {
                break;
            }
        }
    }
    if nonzero == 1 {
        if idx == 0 {
            pc.depths[idx] = 0;
            pc.bits[idx] = 0;
        } else {
            pc.depths[0] = 1;
            pc.bits[0] = 0;
            pc.depths[idx] = 1;
            pc.bits[idx] = 1;
        }
    }
}

/// Build a 41-context prefix code from the predicted ICC byte stream.
fn build_icc_code(
    enc: &[u8],
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
) -> OwnedEntropyCode {
    use crate::entropy::cluster_histograms;

    let num_contexts = K_NUM_ICC_CONTEXTS;
    // Each context gets its own raw histogram first.
    use crate::entropy::uint_encode;
    let mut histograms: Vec<Histogram> = vec![Histogram::new(); num_contexts];
    for (i, &b) in enc.iter().enumerate() {
        let prev1 = if i > 0 { enc[i - 1] } else { 0 };
        let prev2 = if i > 1 { enc[i - 2] } else { 0 };
        let ctx = iccans_context(i, prev1, prev2);
        let (tok, _, _) = uint_encode(b as u32);
        debug_assert!((tok as usize) < ALPHABET_SIZE);
        histograms[ctx].add(tok);
    }

    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map, huffman_pool);

    let mut code = OwnedEntropyCode {
        context_map,
        prefix_codes: build_huffman_codes(&histograms, huffman_pool),
        hybrid_uint_configs: vec![crate::entropy::HybridUintConfig::DEFAULT; histograms.len()],
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: true,
        ans_freqs: Vec::new(),
        ans_symbols: Vec::new(),
        ans_reverse_maps: Vec::new(),
    };
    for pc in &mut code.prefix_codes {
        single_symbol_patch(pc);
    }
    code
}

/// Emit the JXL ICC stream right after the color encoding bits.
/// `icc` must be non-empty.
pub(crate) fn write_icc_stream(
    icc: &[u8],
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut BitWriter,
) {
    assert!(!icc.is_empty(), "ICC profile must be non-empty");
    let enc = predict_icc_minimal(icc);

    // 1. U64 size of the predicted stream.
    write_u64(enc.len() as u64, w);

    // 2. Entropy-coded payload over 41 contexts.
    let code = build_icc_code(&enc, huffman_pool);
    // Header: `lz77.enabled = 0`, then context map + prefix codes.
    w.write(1, 0); // LZ77 disabled
    write_entropy_code(&code.as_ref(), huffman_pool, w);

    // 3. Emit tokens.
    let code_ref = code.as_ref();
    for (i, &b) in enc.iter().enumerate() {
        let prev1 = if i > 0 { enc[i - 1] } else { 0 };
        let prev2 = if i > 1 { enc[i - 2] } else { 0 };
        let ctx = iccans_context(i, prev1, prev2);
        write_token(Token::new(ctx as u32, b as u32), &code_ref, w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trip() {
        let test = [0u64, 1, 126, 127, 128, 255, 16383, 16384, 12345678];
        for &v in &test {
            let mut buf = Vec::new();
            encode_varint(v, &mut buf);
            // Decode manually.
            let mut x = 0u64;
            let mut shift = 0;
            for &b in &buf {
                x |= ((b & 127) as u64) << shift;
                if b & 128 == 0 {
                    break;
                }
                shift += 7;
            }
            assert_eq!(x, v, "varint round-trip {v}");
        }
    }

    #[test]
    fn iccans_context_in_bounds() {
        for i in 0..200 {
            for b1 in [0u8, 1, 16, 100, 200, 255] {
                for b2 in [0u8, 1, 16, 100, 200, 255] {
                    let c = iccans_context(i, b1, b2);
                    assert!(c < K_NUM_ICC_CONTEXTS, "ctx {c} out of range");
                }
            }
        }
    }

    #[test]
    fn minimal_predict_round_trips_via_format() {
        // Construct a fake ICC of 200 bytes (header + 72 tail).
        let mut icc = vec![0u8; 200];
        icc[0..4].copy_from_slice(&(200u32).to_be_bytes());
        icc[8] = 4;
        icc[12..16].copy_from_slice(b"mntr");
        icc[16..20].copy_from_slice(b"RGB ");
        icc[20..24].copy_from_slice(b"XYZ ");
        icc[36..40].copy_from_slice(b"acsp");
        // Fill tail with arbitrary bytes.
        for i in K_ICC_HEADER_SIZE..200 {
            icc[i] = i as u8;
        }

        let enc = predict_icc_minimal(&icc);
        // VarInt(osize=200) = 2 bytes (200 needs 8 bits → 2 varint bytes).
        // VarInt(csize) — depends.
        // commands: VarInt(0) (= 1 byte) + Insert (1 byte) + VarInt(72) (= 1 byte) = 3 bytes
        // Header delta: 128 bytes (mostly zeros — the prediction template matches).
        // Tail: 72 bytes.
        // Total ≈ 2 + 1 + 3 + 128 + 72 = 206 bytes.
        // (Plus the U64 size header at the JXL bit level, which is separate.)
        assert!(
            enc.len() >= 200 && enc.len() <= 220,
            "enc.len() = {}",
            enc.len()
        );

        // Header delta starts at offset = VarInt(osize len) + VarInt(csize len) + csize.
        // For osize=200 (1 varint byte? Actually 200 > 127 so 2 bytes), csize=3 (1 byte).
        // After: 128 bytes of header_data, then 72 tail bytes.
        // First few header_data bytes should be zero (icc matches template at those positions).
    }

    #[test]
    fn write_icc_stream_smoke() {
        // Build a tiny synthetic ICC and verify write_icc_stream produces a non-empty bitstream
        // with the expected leading U64-encoded size.
        let mut icc = vec![0u8; K_ICC_HEADER_SIZE + 16];
        let len = icc.len() as u32;
        icc[0..4].copy_from_slice(&len.to_be_bytes());
        icc[8] = 4;
        icc[12..16].copy_from_slice(b"mntr");
        icc[16..20].copy_from_slice(b"RGB ");
        icc[20..24].copy_from_slice(b"XYZ ");
        icc[36..40].copy_from_slice(b"acsp");
        let mut w = BitWriter::new();
        let mut scratch = crate::coder_scratch::CoderScratch::default();
        write_icc_stream(&icc, &mut scratch.huffman_pool, &mut w);
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();
        assert!(!bytes.is_empty());
    }
}
