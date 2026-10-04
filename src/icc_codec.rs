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
use std::collections::HashMap;

use crate::bit_writer::BitWriter;
use crate::entropy::{
    ALPHABET_SIZE, ANS_TAB_SIZE, AnsCoder, CLUSTERS_LIMIT, Histogram, HuffmanNode,
    HybridUintConfig, OwnedEntropyCode, build_ans_code_parts, build_huffman_codes,
    cluster_histograms, cluster_histograms_ans, uint_encode_with_config, write_entropy_code,
};
use crate::lz77_ac::lz77_length_encode;

pub(crate) const K_ICC_HEADER_SIZE: usize = 128;
pub(crate) const K_NUM_ICC_CONTEXTS: usize = 41;

const COMMAND_INSERT: u8 = 1;
const COMMAND_SHUFFLE2: u8 = 2;
const COMMAND_SHUFFLE4: u8 = 3;
const COMMAND_PREDICT: u8 = 4;
const COMMAND_XYZ: u8 = 10;
const COMMAND_TYPE_START_FIRST: u8 = 16;

const COMMAND_TAG_UNKNOWN: u8 = 1;
const COMMAND_TAG_TRC: u8 = 2;
const COMMAND_TAG_XYZ: u8 = 3;
const COMMAND_TAG_STRING_FIRST: u8 = 4;

const FLAG_BIT_OFFSET: u8 = 64;
const FLAG_BIT_SIZE: u8 = 128;

/// Larger profiles are stored with the minimal (insert-only) prediction.
const SIZE_LIMIT: u64 = (u32::MAX >> 2) as u64;

type Tag = [u8; 4];

/// Tag names with a dedicated tag-list command (RGB and gray monitor profiles).
static TAG_STRINGS: [Tag; 17] = [
    *b"cprt", *b"wtpt", *b"bkpt", *b"rXYZ", *b"gXYZ", *b"bXYZ", *b"kXYZ", *b"rTRC", *b"gTRC",
    *b"bTRC", *b"kTRC", *b"chad", *b"desc", *b"chrm", *b"dmnd", *b"dmdd", *b"lumi",
];

/// Tag types with a dedicated main-content command.
static TYPE_STRINGS: [Tag; 8] = [
    *b"XYZ ", *b"desc", *b"text", *b"mluc", *b"para", *b"curv", *b"sf32", *b"gbd ",
];

/// Tags whose size the tag list predicts as 20 (a single XYZNumber).
static XYZ_SIZED_TAGS: [Tag; 7] = [
    *b"rXYZ", *b"gXYZ", *b"bXYZ", *b"kXYZ", *b"wtpt", *b"bkpt", *b"lumi",
];

/// Initial 128-byte header prediction — libjxl's `ICCInitialHeaderPrediction`.
fn icc_initial_header_prediction(size: u64) -> [u8; K_ICC_HEADER_SIZE] {
    let mut h = [0u8; K_ICC_HEADER_SIZE];
    h[0..4].copy_from_slice(&(size as u32).to_be_bytes());
    h[8] = 4;
    h[12..16].copy_from_slice(b"mntr");
    h[16..20].copy_from_slice(b"RGB ");
    h[20..24].copy_from_slice(b"XYZ ");
    h[36..40].copy_from_slice(b"acsp");
    h[70] = 246;
    h[71] = 214;
    h[73] = 1;
    h[78] = 211;
    h[79] = 45;
    h
}

/// Update the header prediction at position `pos` from the bytes before it.
/// Mirrors libjxl's `ICCPredictHeader`.
fn icc_predict_header(icc: &[u8], header: &mut [u8; K_ICC_HEADER_SIZE], pos: usize) {
    let size = icc.len();
    if pos == 8 && size >= 8 {
        header[80..84].copy_from_slice(&icc[4..8]);
    }
    if pos == 41 && size >= 41 {
        if icc[40] == b'A' {
            header[41..44].copy_from_slice(b"PPL");
        }
        if icc[40] == b'M' {
            header[41..44].copy_from_slice(b"SFT");
        }
    }
    if pos == 42 && size >= 42 {
        if icc[40] == b'S' && icc[41] == b'G' {
            header[42..44].copy_from_slice(b"I ");
        }
        if icc[40] == b'S' && icc[41] == b'U' {
            header[42..44].copy_from_slice(b"NW");
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

fn decode_varint(input: &[u8], end: usize, pos: &mut usize) -> Option<u64> {
    let mut ret = 0u64;
    for i in 0..10 {
        if *pos >= end {
            return None;
        }
        let byte = input[*pos];
        *pos += 1;
        if i == 9 && byte & 0xFE != 0 {
            return None;
        }
        ret |= u64::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            return Some(ret);
        }
    }
    None
}

/// Big-endian u32 at `pos`, or 0 when it does not fit in `size`.
fn decode_u32(data: &[u8], size: usize, pos: usize) -> u32 {
    if pos + 4 > size {
        0
    } else {
        u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
    }
}

fn decode_keyword(data: &[u8], size: usize, pos: usize) -> Tag {
    if pos + 4 > size {
        *b"    "
    } else {
        [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]
    }
}

macro_rules! predict_value {
    ($p1:expr, $p2:expr, $p3:expr, $order:expr) => {
        match $order {
            0 => $p1,
            1 => $p1.wrapping_mul(2).wrapping_sub($p2),
            2 => $p1
                .wrapping_mul(3)
                .wrapping_sub($p2.wrapping_mul(3))
                .wrapping_add($p3),
            _ => 0,
        }
    };
}

/// libjxl's `LinearPredictICCValue`: order-0..2 prediction of byte `i` of a
/// run of `width`-byte big-endian integers starting at `start`, `stride`
/// bytes apart. Only bytes before `start + i` are read.
fn linear_predict_icc_value(
    data: &[u8],
    start: usize,
    i: usize,
    stride: usize,
    width: usize,
    order: u8,
) -> u8 {
    let pos = start + i;
    if width == 1 {
        let p1 = data[pos - stride];
        let p2 = data[pos - stride * 2];
        let p3 = data[pos - stride * 3];
        predict_value!(p1, p2, p3, order)
    } else if width == 2 {
        let p = start + (i & !1);
        let at = |o: usize| u16::from_be_bytes([data[p - o], data[p - o + 1]]);
        let (p1, p2, p3) = (at(stride), at(stride * 2), at(stride * 3));
        let pred: u16 = predict_value!(p1, p2, p3, order);
        if i & 1 != 0 {
            pred as u8
        } else {
            (pred >> 8) as u8
        }
    } else {
        let p = start + (i & !3);
        let p1 = decode_u32(data, pos, p - stride);
        let p2 = decode_u32(data, pos, p - stride * 2);
        let p3 = decode_u32(data, pos, p - stride * 3);
        let pred: u32 = predict_value!(p1, p2, p3, order);
        (pred >> ((3 - (i & 3)) * 8)) as u8
    }
}

/// De-interleave: width 2 turns "AaBbCc" into "ABCabc".
fn unshuffle(data: &mut [u8], width: usize) {
    let size = data.len();
    let height = size.div_ceil(width);
    let mut result = vec![0u8; size];
    let (mut s, mut j) = (0, 0);
    for &b in data.iter() {
        result[j] = b;
        j += height;
        if j >= size {
            s += 1;
            j = s;
        }
    }
    data.copy_from_slice(&result);
}

/// Inverse of [`unshuffle`].
fn shuffle(data: &mut [u8], width: usize) {
    let size = data.len();
    let height = size.div_ceil(width);
    let mut result = vec![0u8; size];
    let (mut s, mut j) = (0, 0);
    for r in result.iter_mut() {
        *r = data[j];
        j += height;
        if j >= size {
            s += 1;
            j = s;
        }
    }
    data.copy_from_slice(&result);
}

/// Predict `num` bytes at `*pos` and append the (unshuffled) residuals.
/// `None` when the decoder's stride rule would reject the command.
#[allow(clippy::too_many_arguments)]
fn predict_and_shuffle(
    stride: usize,
    width: usize,
    order: u8,
    num: usize,
    icc: &[u8],
    pos: &mut usize,
    out: &mut Vec<u8>,
) -> Option<()> {
    if *pos + num > icc.len() || *pos == 0 || ((*pos - 1) >> 2) < stride {
        return None;
    }
    let start = out.len();
    for i in 0..num {
        let predicted = linear_predict_icc_value(icc, *pos, i, stride, width, order);
        out.push(icc[*pos + i].wrapping_sub(predicted));
    }
    *pos += num;
    if width > 1 {
        unshuffle(&mut out[start..], width);
    }
    Some(())
}

/// Header delta + one Insert for the rest: valid for any byte string.
fn predict_icc_minimal(icc: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(icc.len() + 32);
    encode_varint(icc.len() as u64, &mut out);
    let mut header = icc_initial_header_prediction(icc.len() as u64);
    let mut header_data = Vec::with_capacity(K_ICC_HEADER_SIZE);
    for i in 0..K_ICC_HEADER_SIZE.min(icc.len()) {
        icc_predict_header(icc, &mut header, i);
        header_data.push(icc[i].wrapping_sub(header[i]));
    }
    let mut commands = Vec::new();
    if icc.len() > K_ICC_HEADER_SIZE {
        encode_varint(0, &mut commands); // no tag list
        commands.push(COMMAND_INSERT);
        encode_varint((icc.len() - K_ICC_HEADER_SIZE) as u64, &mut commands);
    }
    encode_varint(commands.len() as u64, &mut out);
    out.extend_from_slice(&commands);
    out.extend_from_slice(&header_data);
    if icc.len() > K_ICC_HEADER_SIZE {
        out.extend_from_slice(&icc[K_ICC_HEADER_SIZE..]);
    }
    out
}

/// Port of libjxl's `PredictICC`: tag-list commands plus typed predictions of
/// the tagged elements (curves, CLUTs, XYZ numbers, UTF-16 text).
fn predict_icc(icc: &[u8]) -> Option<Vec<u8>> {
    let size = icc.len();
    if size as u64 > SIZE_LIMIT {
        return None;
    }
    let mut result = Vec::with_capacity(size + 64);
    let mut commands: Vec<u8> = Vec::new();
    let mut data: Vec<u8> = Vec::with_capacity(size);

    encode_varint(size as u64, &mut result);
    let mut header = icc_initial_header_prediction(size as u64);
    for i in 0..K_ICC_HEADER_SIZE.min(size) {
        icc_predict_header(icc, &mut header, i);
        data.push(icc[i].wrapping_sub(header[i]));
    }
    if size <= K_ICC_HEADER_SIZE {
        encode_varint(0, &mut result); // no commands
        result.extend_from_slice(&data);
        return Some(result);
    }

    let mut tagstarts: Vec<u64> = Vec::new();
    let mut tagsizes: Vec<u64> = Vec::new();
    let mut tagmap: HashMap<u64, usize> = HashMap::new();

    // Tag list.
    let mut pos = K_ICC_HEADER_SIZE;
    if pos + 4 <= size {
        let numtags = u64::from(decode_u32(icc, size, pos));
        pos += 4;
        encode_varint(numtags + 1, &mut commands);
        let mut prevtagstart = K_ICC_HEADER_SIZE as u64 + numtags * 12;
        let mut prevtagsize = 0u64;
        let mut i = 0u64;
        while i < numtags {
            if pos + 12 > size {
                break;
            }
            let tag = decode_keyword(icc, size, pos);
            let tagstart = u64::from(decode_u32(icc, size, pos + 4));
            let tagsize = u64::from(decode_u32(icc, size, pos + 8));
            pos += 12;

            tagstarts.push(tagstart);
            tagsizes.push(tagsize);
            tagmap.insert(tagstart, tagstarts.len() - 1);

            let mut tagcode = TAG_STRINGS
                .iter()
                .position(|t| *t == tag)
                .map_or(COMMAND_TAG_UNKNOWN, |j| j as u8 + COMMAND_TAG_STRING_FIRST);

            if tag == *b"rTRC" && pos + 24 < size {
                let ok = decode_keyword(icc, size, pos) == *b"gTRC"
                    && decode_keyword(icc, size, pos + 12) == *b"bTRC"
                    && icc[pos - 8..pos] == icc[pos + 4..pos + 12]
                    && icc[pos - 8..pos] == icc[pos + 16..pos + 24];
                if ok {
                    tagcode = COMMAND_TAG_TRC;
                    pos += 24;
                    i += 2;
                }
            }

            if tag == *b"rXYZ" && pos + 24 < size {
                let offsetg = u64::from(decode_u32(icc, size, pos + 4));
                let offsetb = u64::from(decode_u32(icc, size, pos + 16));
                let sizeg = decode_u32(icc, size, pos + 8);
                let sizeb = decode_u32(icc, size, pos + 20);
                let ok = decode_keyword(icc, size, pos) == *b"gXYZ"
                    && decode_keyword(icc, size, pos + 12) == *b"bXYZ"
                    && tagsize == 20
                    && sizeg == 20
                    && sizeb == 20
                    && offsetg == tagstart + 20
                    && offsetb == tagstart + 40;
                if ok {
                    tagcode = COMMAND_TAG_XYZ;
                    pos += 24;
                    i += 2;
                }
            }

            let mut command = tagcode;
            if prevtagstart + prevtagsize != tagstart {
                command |= FLAG_BIT_OFFSET;
            }
            let predicted_tagsize = if XYZ_SIZED_TAGS.contains(&tag) {
                20
            } else {
                prevtagsize
            };
            if predicted_tagsize != tagsize {
                command |= FLAG_BIT_SIZE;
            }
            commands.push(command);
            if tagcode == COMMAND_TAG_UNKNOWN {
                data.extend_from_slice(&tag);
            }
            if command & FLAG_BIT_OFFSET != 0 {
                encode_varint(tagstart, &mut commands);
            }
            if command & FLAG_BIT_SIZE != 0 {
                encode_varint(tagsize, &mut commands);
            }
            prevtagstart = tagstart;
            prevtagsize = tagsize;
            i += 1;
        }
    }
    // End of the tag list (or a zero varint: no tag list).
    commands.push(0);

    // Main content: walk the tagged elements, emitting typed commands where
    // the element is recognized and Inserts for the bytes in between.
    let mut tag: Tag = [0; 4];
    let mut tagstart = 0u64;
    let mut tagsize = 0u64;
    let mut clutstart = 0u64;
    let tag_sane = |tagsize: u64| tagsize > 8 && tagsize < SIZE_LIMIT;

    let mut last0 = pos;
    while pos <= size {
        let mut last1 = pos;
        let mut commands_add: Vec<u8> = Vec::new();
        let mut data_add: Vec<u8> = Vec::new();
        let upos = pos as u64;

        if upos > tagstart + tagsize && tagsize < SIZE_LIMIT {
            tag = [0; 4];
        }

        if pos + 4 <= size
            && let Some(&index) = tagmap.get(&upos)
        {
            tag = decode_keyword(icc, size, pos);
            tagstart = tagstarts[index];
            tagsize = tagsizes[index];

            if tag == *b"mluc"
                && tag_sane(tagsize)
                && upos + tagsize <= size as u64
                && icc[pos + 4..pos + 8] == [0; 4]
            {
                let num = (tagsize - 8) as usize;
                commands_add.push(COMMAND_TYPE_START_FIRST + 3);
                pos += 8;
                commands_add.push(COMMAND_SHUFFLE2);
                encode_varint(num as u64, &mut commands_add);
                let start = data_add.len();
                data_add.extend_from_slice(&icc[pos..pos + num]);
                pos += num;
                unshuffle(&mut data_add[start..], 2);
            }

            if tag == *b"curv"
                && tag_sane(tagsize)
                && upos + tagsize <= size as u64
                && icc[pos + 4..pos + 8] == [0; 4]
            {
                let num = (tagsize - 8) as usize;
                if num > 16 && num < (1 << 28) && pos + num <= size {
                    commands_add.push(COMMAND_TYPE_START_FIRST + 5);
                    pos += 8;
                    commands_add.push(COMMAND_PREDICT);
                    let (order, width) = (1u8, 2usize);
                    commands_add.push((order << 2) | (width as u8 - 1));
                    encode_varint(num as u64, &mut commands_add);
                    predict_and_shuffle(width, width, order, num, icc, &mut pos, &mut data_add)?;
                }
            }
        }

        if tag == *b"mAB " || tag == *b"mBA " {
            let sub_tag = decode_keyword(icc, size, pos);
            if pos + 12 < size
                && (sub_tag == *b"curv" || sub_tag == *b"vcgt")
                && decode_u32(icc, size, pos + 4) == 0
            {
                let num = u64::from(decode_u32(icc, size, pos + 8)) * 2;
                if num > 16 && num < (1 << 28) && pos as u64 + 12 + num <= size as u64 {
                    pos += 12;
                    last1 = pos;
                    commands_add.push(COMMAND_PREDICT);
                    let (order, width) = (1u8, 2usize);
                    commands_add.push((order << 2) | (width as u8 - 1));
                    encode_varint(num, &mut commands_add);
                    predict_and_shuffle(
                        width,
                        width,
                        order,
                        num as usize,
                        icc,
                        &mut pos,
                        &mut data_add,
                    )?;
                }
            }

            if pos as u64 == tagstart + 24 && pos + 4 < size {
                // Remembered across iterations: the CLUT starts later.
                clutstart = tagstart + u64::from(decode_u32(icc, size, pos));
            }

            if pos as u64 == clutstart && clutstart + 16 < size as u64 && tagstart + 9 < size as u64
            {
                let clut = clutstart as usize;
                let numi = icc[tagstart as usize + 8] as usize;
                let numo = u64::from(icc[tagstart as usize + 9]);
                let width = u64::from(icc[clut + 16]);
                let stride = width * numo;
                let mut num = width * numo;
                for i in 0..numi {
                    if clut + i >= size {
                        break;
                    }
                    num = num.saturating_mul(u64::from(icc[clut + i]));
                }
                if (width == 1 || width == 2)
                    && num > 64
                    && num < (1 << 28)
                    && pos as u64 + num <= size as u64
                    && pos as u64 > stride * 4
                {
                    let order = 1u8;
                    let flags =
                        (order << 2) | (width as u8 - 1) | if stride == width { 0 } else { 16 };
                    commands_add.push(COMMAND_PREDICT);
                    commands_add.push(flags);
                    if flags & 16 != 0 {
                        encode_varint(stride, &mut commands_add);
                    }
                    encode_varint(num, &mut commands_add);
                    predict_and_shuffle(
                        stride as usize,
                        width as usize,
                        order,
                        num as usize,
                        icc,
                        &mut pos,
                        &mut data_add,
                    )?;
                }
            }
        }

        if commands_add.is_empty()
            && data_add.is_empty()
            && tag == *b"gbd "
            && tag_sane(tagsize)
            && pos as u64 == tagstart + 8
            && pos as u64 + tagsize - 8 <= size as u64
            && pos > 16
        {
            let (width, order) = (4usize, 0u8);
            let num = (tagsize - 8) as usize;
            commands_add.push(COMMAND_PREDICT);
            commands_add.push((order << 2) | (width as u8 - 1));
            encode_varint(num as u64, &mut commands_add);
            predict_and_shuffle(width, width, order, num, icc, &mut pos, &mut data_add)?;
        }

        if commands_add.is_empty()
            && data_add.is_empty()
            && pos + 20 <= size
            && decode_keyword(icc, size, pos) == *b"XYZ "
            && decode_u32(icc, size, pos + 4) == 0
        {
            commands_add.push(COMMAND_XYZ);
            pos += 8;
            data_add.extend_from_slice(&icc[pos..pos + 12]);
            pos += 12;
        }

        if commands_add.is_empty()
            && data_add.is_empty()
            && pos + 8 <= size
            && decode_u32(icc, size, pos + 4) == 0
        {
            let sub_tag = decode_keyword(icc, size, pos);
            if let Some(i) = TYPE_STRINGS.iter().position(|t| *t == sub_tag) {
                commands_add.push(COMMAND_TYPE_START_FIRST + i as u8);
                pos += 8;
            }
        }

        let added = !(commands_add.is_empty() && data_add.is_empty());
        if added || pos == size {
            if last0 < last1 {
                commands.push(COMMAND_INSERT);
                encode_varint((last1 - last0) as u64, &mut commands);
                data.extend_from_slice(&icc[last0..last1]);
            }
            commands.extend_from_slice(&commands_add);
            data.extend_from_slice(&data_add);
            last0 = pos;
        }
        if !added {
            pos += 1;
        }
    }

    encode_varint(commands.len() as u64, &mut result);
    result.extend_from_slice(&commands);
    result.extend_from_slice(&data);
    Some(result)
}

fn append_keyword(tag: Tag, out: &mut Vec<u8>) {
    out.extend_from_slice(&tag);
}

fn append_u32(value: u64, out: &mut Vec<u8>) -> Option<()> {
    out.extend_from_slice(&u32::try_from(value).ok()?.to_be_bytes());
    Some(())
}

/// Port of libjxl's `UnpredictICC`, the decoder side of [`predict_icc`]. The
/// encoder runs it to prove a prediction reproduces the profile exactly.
fn unpredict_icc(enc: &[u8]) -> Option<Vec<u8>> {
    let size = enc.len();
    let mut pos = 0usize;
    let osize = decode_varint(enc, size, &mut pos)?;
    u32::try_from(osize).ok()?;
    let csize = decode_varint(enc, size, &mut pos)?;
    u32::try_from(csize).ok()?;
    let mut cpos = pos;
    let commands_end = pos.checked_add(csize as usize)?;
    if commands_end > size {
        return None;
    }
    pos = commands_end;
    let osize = osize as usize;
    let mut result: Vec<u8> = Vec::with_capacity(osize);

    let mut header = icc_initial_header_prediction(osize as u64);
    for i in 0..=K_ICC_HEADER_SIZE {
        if result.len() == osize {
            return (cpos == commands_end && pos == size).then_some(result);
        }
        if i == K_ICC_HEADER_SIZE {
            break;
        }
        icc_predict_header(&result, &mut header, i);
        if pos >= size {
            return None;
        }
        result.push(enc[pos].wrapping_add(header[i]));
        pos += 1;
    }
    if cpos >= commands_end {
        return None;
    }

    // Tag list.
    let numtags = decode_varint(enc, commands_end, &mut cpos)?;
    if numtags != 0 {
        let numtags = numtags - 1;
        append_u32(numtags, &mut result)?;
        let mut prevtagstart = K_ICC_HEADER_SIZE as u64 + numtags * 12;
        let mut prevtagsize = 0u64;
        loop {
            if result.len() > osize || cpos > commands_end {
                return None;
            }
            if cpos == commands_end {
                break;
            }
            let command = enc[cpos];
            cpos += 1;
            let tagcode = command & 63;
            let tag: Tag = match tagcode {
                0 => break,
                COMMAND_TAG_UNKNOWN => {
                    if pos + 4 > size {
                        return None;
                    }
                    let t = decode_keyword(enc, size, pos);
                    pos += 4;
                    t
                }
                COMMAND_TAG_TRC => *b"rTRC",
                COMMAND_TAG_XYZ => *b"rXYZ",
                _ => *TAG_STRINGS.get((tagcode - COMMAND_TAG_STRING_FIRST) as usize)?,
            };
            append_keyword(tag, &mut result);

            let mut tagsize = if XYZ_SIZED_TAGS.contains(&tag) {
                20
            } else {
                prevtagsize
            };
            let tagstart = if command & FLAG_BIT_OFFSET != 0 {
                decode_varint(enc, commands_end, &mut cpos)?
            } else {
                u32::try_from(prevtagstart).ok()?;
                prevtagstart + prevtagsize
            };
            append_u32(tagstart, &mut result)?;
            if command & FLAG_BIT_SIZE != 0 {
                tagsize = decode_varint(enc, commands_end, &mut cpos)?;
            }
            append_u32(tagsize, &mut result)?;
            prevtagstart = tagstart;
            prevtagsize = tagsize;

            if tagcode == COMMAND_TAG_TRC {
                for t in [*b"gTRC", *b"bTRC"] {
                    append_keyword(t, &mut result);
                    append_u32(tagstart, &mut result)?;
                    append_u32(tagsize, &mut result)?;
                }
            }
            if tagcode == COMMAND_TAG_XYZ {
                u32::try_from(tagstart + tagsize * 2).ok()?;
                for (k, t) in [*b"gXYZ", *b"bXYZ"].into_iter().enumerate() {
                    append_keyword(t, &mut result);
                    append_u32(tagstart + tagsize * (k as u64 + 1), &mut result)?;
                    append_u32(tagsize, &mut result)?;
                }
            }
        }
    }

    // Main content.
    loop {
        if result.len() > osize || cpos > commands_end {
            return None;
        }
        if cpos == commands_end {
            break;
        }
        let command = enc[cpos];
        cpos += 1;
        match command {
            COMMAND_INSERT => {
                let num = decode_varint(enc, commands_end, &mut cpos)? as usize;
                let end = pos.checked_add(num).filter(|&e| e <= size)?;
                result.extend_from_slice(&enc[pos..end]);
                pos = end;
            }
            COMMAND_SHUFFLE2 | COMMAND_SHUFFLE4 => {
                let num = decode_varint(enc, commands_end, &mut cpos)? as usize;
                let end = pos.checked_add(num).filter(|&e| e <= size)?;
                let mut shuffled = enc[pos..end].to_vec();
                shuffle(
                    &mut shuffled,
                    if command == COMMAND_SHUFFLE2 { 2 } else { 4 },
                );
                result.extend_from_slice(&shuffled);
                pos = end;
            }
            COMMAND_PREDICT => {
                if cpos + 2 > commands_end {
                    return None;
                }
                let flags = enc[cpos];
                cpos += 1;
                let width = (flags & 3) as usize + 1;
                let order = (flags & 12) >> 2;
                if width == 3 || order == 3 {
                    return None;
                }
                let mut stride = width as u64;
                if flags & 16 != 0 {
                    stride = decode_varint(enc, commands_end, &mut cpos)?;
                    if stride < width as u64 {
                        return None;
                    }
                }
                if result.is_empty() || (((result.len() - 1) >> 2) as u64) < stride {
                    return None;
                }
                let num = decode_varint(enc, commands_end, &mut cpos)? as usize;
                let end = pos.checked_add(num).filter(|&e| e <= size)?;
                let mut shuffled = enc[pos..end].to_vec();
                if width > 1 {
                    shuffle(&mut shuffled, width);
                }
                let start = result.len();
                for (i, &s) in shuffled.iter().enumerate() {
                    let predicted =
                        linear_predict_icc_value(&result, start, i, stride as usize, width, order);
                    result.push(predicted.wrapping_add(s));
                }
                pos = end;
            }
            COMMAND_XYZ => {
                append_keyword(*b"XYZ ", &mut result);
                result.extend_from_slice(&[0; 4]);
                if pos + 12 > size {
                    return None;
                }
                result.extend_from_slice(&enc[pos..pos + 12]);
                pos += 12;
            }
            c if (COMMAND_TYPE_START_FIRST
                ..COMMAND_TYPE_START_FIRST + TYPE_STRINGS.len() as u8)
                .contains(&c) =>
            {
                append_keyword(
                    TYPE_STRINGS[(c - COMMAND_TYPE_START_FIRST) as usize],
                    &mut result,
                );
                result.extend_from_slice(&[0; 4]);
            }
            _ => return None,
        }
    }

    (pos == size && result.len() == osize).then_some(result)
}

// libjxl's ByteKind1 / ByteKind2 select one of the 41 ICC contexts.
fn byte_kind_1(b: u8) -> u8 {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0 => 2,
        1 => 3,
        2..=15 => 4,
        255 => 6,
        241..=254 => 5,
        _ => 7,
    }
}

fn byte_kind_2(b: u8) -> u8 {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=15 => 2,
        241..=255 => 3,
        _ => 4,
    }
}

/// Context of byte `i` of the predicted stream given the two bytes before it.
pub(crate) fn iccans_context(i: usize, prev1: u8, prev2: u8) -> usize {
    if i <= 128 {
        return 0;
    }
    1 + byte_kind_1(prev1) as usize + byte_kind_2(prev2) as usize * 8
}

// ---------------------------------------------------------------------------
// Entropy coding of the predicted stream. The ICC stream takes the general
// entropy code (LZ77, prefix or ANS, clustering), so every variant is written
// in full and the smallest one is kept.
// ---------------------------------------------------------------------------

/// The decoder appends this context for LZ77 distances (dist_multiplier = 0:
/// a distance is coded as `distance - 1`).
const DISTANCE_CONTEXT: usize = K_NUM_ICC_CONTEXTS;
const LZ77_MIN_LENGTH: usize = 3;
const LZ77_MAX_LENGTH: usize = 1 << 16;
/// libjxl's decoder window.
const LZ77_WINDOW: usize = 1 << 20;
/// Matches at least this long are taken without searching inside them.
const LZ77_NICE_LENGTH: usize = 256;
const LZ77_HASH_BITS: u32 = 15;
/// The parse tries every length up to this, and only the full length beyond.
const LZ77_DENSE_LENGTHS: usize = 64;
const LZ77_PASSES: usize = 4;
/// Price of a symbol the current code cannot represent.
const UNSEEN_SYMBOL_BITS: f32 = 15.0;

static ICC_UINT_CONFIGS: [HybridUintConfig; 5] = [
    HybridUintConfig::DEFAULT,
    HybridUintConfig {
        split_exponent: 4,
        msb_in_token: 1,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 4,
        msb_in_token: 0,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 5,
        msb_in_token: 2,
        lsb_in_token: 0,
    },
    HybridUintConfig {
        split_exponent: 6,
        msb_in_token: 1,
        lsb_in_token: 0,
    },
];

#[derive(Clone, Copy)]
struct IccSymbol {
    context: u8,
    symbol: u8,
    nbits: u8,
    bits: u32,
}

#[derive(Clone, Copy)]
struct IccMatch {
    pos: usize,
    len: usize,
    distance: usize,
}

/// Bit prices of one written code, used to steer the next LZ77 parse.
struct IccPrices {
    config: HybridUintConfig,
    min_symbol: u32,
    /// `[context][symbol]`, including the distance context.
    bits: Vec<[f32; ALPHABET_SIZE]>,
}

impl IccPrices {
    fn literal(&self, context: u8, value: u8) -> f32 {
        let (symbol, nbits, _) = uint_encode_with_config(u32::from(value), self.config);
        self.bits[context as usize][symbol as usize] + nbits as f32
    }

    /// Length part of a copy that starts in `context`.
    fn length(&self, context: u8, len: usize) -> f32 {
        let (len_token, len_nbits, _) = lz77_length_encode((len - LZ77_MIN_LENGTH) as u32);
        let len_symbol = (self.min_symbol + len_token) as usize;
        if len_symbol >= ALPHABET_SIZE {
            return f32::INFINITY;
        }
        self.bits[context as usize][len_symbol] + len_nbits as f32
    }

    fn distance(&self, distance: usize) -> f32 {
        let (symbol, nbits, _) = uint_encode_with_config((distance - 1) as u32, self.config);
        if symbol as usize >= ALPHABET_SIZE {
            return f32::INFINITY;
        }
        self.bits[DISTANCE_CONTEXT][symbol as usize] + nbits as f32
    }
}

fn icc_contexts(enc: &[u8]) -> Vec<u8> {
    (0..enc.len())
        .map(|i| {
            let prev1 = if i > 0 { enc[i - 1] } else { 0 };
            let prev2 = if i > 1 { enc[i - 2] } else { 0 };
            iccans_context(i, prev1, prev2) as u8
        })
        .collect()
}

/// Symbols for `enc` under `matches`, with the LZ77 `min_symbol` placed just
/// above the largest literal token. `None` when a symbol leaves the alphabet.
fn tokenize(
    enc: &[u8],
    contexts: &[u8],
    matches: &[IccMatch],
    config: HybridUintConfig,
) -> Option<(Vec<IccSymbol>, u32)> {
    let mut min_symbol = 8u32;
    let mut next = matches.iter().peekable();
    let mut pos = 0;
    while pos < enc.len() {
        if let Some(m) = next.next_if(|m| m.pos == pos) {
            pos += m.len;
        } else {
            let (symbol, _, _) = uint_encode_with_config(u32::from(enc[pos]), config);
            min_symbol = min_symbol.max(symbol + 1);
            pos += 1;
        }
    }

    let mut symbols = Vec::with_capacity(enc.len());
    let mut next = matches.iter().peekable();
    let mut pos = 0;
    while pos < enc.len() {
        if let Some(m) = next.next_if(|m| m.pos == pos) {
            let (len_token, nbits, bits) = lz77_length_encode((m.len - LZ77_MIN_LENGTH) as u32);
            let symbol = min_symbol + len_token;
            let (dist_symbol, dist_nbits, dist_bits) =
                uint_encode_with_config((m.distance - 1) as u32, config);
            if symbol as usize >= ALPHABET_SIZE || dist_symbol as usize >= ALPHABET_SIZE {
                return None;
            }
            symbols.push(IccSymbol {
                context: contexts[pos],
                symbol: symbol as u8,
                nbits: nbits as u8,
                bits,
            });
            symbols.push(IccSymbol {
                context: DISTANCE_CONTEXT as u8,
                symbol: dist_symbol as u8,
                nbits: dist_nbits as u8,
                bits: dist_bits,
            });
            pos += m.len;
        } else {
            let (symbol, nbits, bits) = uint_encode_with_config(u32::from(enc[pos]), config);
            symbols.push(IccSymbol {
                context: contexts[pos],
                symbol: symbol as u8,
                nbits: nbits as u8,
                bits,
            });
            pos += 1;
        }
    }
    Some((symbols, min_symbol))
}

/// Price-driven optimal parse (shortest path over byte positions).
fn optimal_parse(enc: &[u8], contexts: &[u8], prices: &IccPrices) -> Vec<IccMatch> {
    let n = enc.len();
    if n < LZ77_MIN_LENGTH + 1 {
        return Vec::new();
    }
    const NONE: u32 = u32::MAX;
    let max_chain = if n <= 1 << 16 { 64 } else { 32 };
    let hash_bits = (usize::BITS - n.leading_zeros()).clamp(8, LZ77_HASH_BITS);
    let mut head = vec![NONE; 1 << hash_bits];
    let mut prev = vec![NONE; n];
    let mut cost = vec![f32::INFINITY; n + 1];
    // (len, distance); distance 0 = literal.
    let mut from = vec![(0u32, 0u32); n + 1];
    cost[0] = 0.0;
    let hash = |i: usize| {
        let v = u32::from(enc[i]) | u32::from(enc[i + 1]) << 8 | u32::from(enc[i + 2]) << 16;
        (v.wrapping_mul(0x9E37_79B1) >> (32 - hash_bits)) as usize
    };
    // Dense length prices per context.
    let length_bits: Vec<[f32; LZ77_DENSE_LENGTHS + 1]> = (0..K_NUM_ICC_CONTEXTS as u8)
        .map(|context| {
            let mut row = [f32::INFINITY; LZ77_DENSE_LENGTHS + 1];
            for (l, b) in row.iter_mut().enumerate().skip(LZ77_MIN_LENGTH) {
                *b = prices.length(context, l);
            }
            row
        })
        .collect();
    let mut skip_until = 0;
    for i in 0..n {
        let base = cost[i];
        let literal = base + prices.literal(contexts[i], enc[i]);
        if literal < cost[i + 1] {
            cost[i + 1] = literal;
            from[i + 1] = (1, 0);
        }
        if i + LZ77_MIN_LENGTH > n {
            continue;
        }
        let h = hash(i);
        if i >= skip_until {
            let context = contexts[i];
            let length_bits = &length_bits[context as usize];
            let limit = (n - i).min(LZ77_MAX_LENGTH);
            let mut best_len = LZ77_MIN_LENGTH - 1;
            let mut candidate = head[h];
            let mut chain = 0;
            while candidate != NONE && chain < max_chain {
                let j = candidate as usize;
                let distance = i - j;
                if distance > LZ77_WINDOW {
                    break;
                }
                if best_len < limit && enc[j + best_len] == enc[i + best_len] {
                    let len = enc[j..j + limit]
                        .iter()
                        .zip(&enc[i..i + limit])
                        .take_while(|(a, b)| a == b)
                        .count();
                    if len > best_len {
                        let base = base + prices.distance(distance);
                        let dense_end = len.min(LZ77_DENSE_LENGTHS);
                        let dense = (best_len + 1..=dense_end).map(|l| (l, length_bits[l]));
                        let full = (len > dense_end).then(|| (len, prices.length(context, len)));
                        for (l, bits) in dense.chain(full) {
                            let c = base + bits;
                            if c < cost[i + l] {
                                cost[i + l] = c;
                                from[i + l] = (l as u32, distance as u32);
                            }
                        }
                        best_len = len;
                        if len >= LZ77_NICE_LENGTH || len == limit {
                            break;
                        }
                    }
                }
                candidate = prev[j];
                chain += 1;
            }
            if best_len >= LZ77_NICE_LENGTH {
                skip_until = i + best_len;
            }
        }
        prev[i] = head[h];
        head[h] = i as u32;
    }

    let mut matches = Vec::new();
    let mut p = n;
    while p > 0 {
        let (len, distance) = from[p];
        let len = len as usize;
        if distance != 0 {
            matches.push(IccMatch {
                pos: p - len,
                len,
                distance: distance as usize,
            });
        }
        p -= len;
    }
    matches.reverse();
    matches
}

/// LZ77Params with `min_length` 3 and the (4, 0, 0) length config of
/// [`lz77_length_encode`].
fn write_lz77_params(min_symbol: Option<u32>, w: &mut BitWriter) {
    let Some(min_symbol) = min_symbol else {
        w.write(1, 0);
        return;
    };
    w.write(1, 1);
    w.write(2, 3); // min_symbol: BitsOffset(15, 8)
    w.write(15, u64::from(min_symbol - 8));
    w.write(2, 0); // min_length: Val(3)
    w.write(4, 4); // length config: split 4
    w.write(3, 0); // msb_in_token
    w.write(3, 0); // lsb_in_token
}

/// Cluster, build and write one complete entropy-coded stream.
fn write_symbols(
    symbols: &[IccSymbol],
    min_symbol: Option<u32>,
    config: HybridUintConfig,
    use_ans: bool,
    fixed_map: Option<&[u8]>,
    huffman_pool: &mut Vec<HuffmanNode>,
) -> Option<(BitWriter, IccPrices)> {
    let num_contexts = K_NUM_ICC_CONTEXTS + usize::from(min_symbol.is_some());
    let mut context_map = vec![0u8; num_contexts];
    let mut histograms = match fixed_map {
        Some(map) => {
            context_map.copy_from_slice(&map[..num_contexts]);
            let n = *context_map.iter().max().unwrap() as usize + 1;
            vec![Histogram::new(); n]
        }
        None => vec![Histogram::new(); num_contexts],
    };
    for s in symbols {
        let h = if fixed_map.is_some() {
            context_map[s.context as usize] as usize
        } else {
            s.context as usize
        };
        histograms[h].add(u32::from(s.symbol));
    }
    if fixed_map.is_some() && histograms.iter().any(|h| h.total_count == 0) {
        return None;
    }
    let mut code = OwnedEntropyCode {
        context_map: Vec::new(),
        prefix_codes: Vec::new(),
        hybrid_uint_configs: Vec::new(),
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: !use_ans,
        ans_histograms: Vec::new(),
        ans_pricing_freqs: Vec::new(),
        ans_symbols: Vec::new(),
        ans_reverse_maps: Vec::new(),
    };
    let cluster_bits: Vec<[f32; ALPHABET_SIZE]>;
    if use_ans {
        if fixed_map.is_none() {
            let n = cluster_histograms_ans(
                &mut histograms,
                &mut context_map,
                None,
                true,
                CLUSTERS_LIMIT,
                2,
            );
            histograms.truncate(n);
        }
        let (ans_histograms, ans_symbols, ans_reverse_maps) =
            build_ans_code_parts(&histograms, true);
        cluster_bits = ans_histograms
            .iter()
            .map(|h| {
                let mut bits = [UNSEEN_SYMBOL_BITS; ALPHABET_SIZE];
                for (b, &f) in bits.iter_mut().zip(&h.freqs) {
                    if f != 0 {
                        *b = (ANS_TAB_SIZE as f32 / f32::from(f)).log2();
                    }
                }
                bits
            })
            .collect();
        code.ans_histograms = ans_histograms;
        code.ans_symbols = ans_symbols;
        code.ans_reverse_maps = ans_reverse_maps;
    } else {
        if fixed_map.is_none() {
            cluster_histograms(&mut histograms, &mut context_map, huffman_pool);
        }
        code.prefix_codes = build_huffman_codes(&histograms, huffman_pool);
        cluster_bits = code
            .prefix_codes
            .iter()
            .map(|pc| {
                let mut bits = [UNSEEN_SYMBOL_BITS; ALPHABET_SIZE];
                for (b, &d) in bits.iter_mut().zip(&pc.depths) {
                    if d != 0 {
                        *b = if pc.single_symbol { 0.0 } else { f32::from(d) };
                    }
                }
                bits
            })
            .collect();
    }
    code.hybrid_uint_configs = vec![config; histograms.len()];
    code.context_map = context_map;

    let mut w = BitWriter::new();
    write_lz77_params(min_symbol, &mut w);
    let code_ref = code.as_ref();
    write_entropy_code(&code_ref, huffman_pool, &mut w);

    if use_ans {
        let mut coder = AnsCoder::new();
        let mut words = Vec::with_capacity(symbols.len());
        for s in symbols.iter().rev() {
            let h = code.context_map[s.context as usize] as usize;
            let start = h * ANS_TAB_SIZE as usize;
            words.push(coder.put_symbol(
                &code.ans_symbols[h][s.symbol as usize],
                &code.ans_reverse_maps[start..start + ANS_TAB_SIZE as usize],
            ));
        }
        w.write(32, u64::from(coder.state()));
        for (s, word) in symbols.iter().zip(words.iter().rev()) {
            if let Some(word) = word {
                w.write(16, u64::from(*word));
            }
            w.write(s.nbits as usize, u64::from(s.bits));
        }
    } else {
        for s in symbols {
            let pc = &code.prefix_codes[code.context_map[s.context as usize] as usize];
            if pc.single_symbol {
                w.write(s.nbits as usize, u64::from(s.bits));
            } else {
                let d = pc.depths[s.symbol as usize] as usize;
                let data = u64::from(pc.bits[s.symbol as usize]) | (u64::from(s.bits) << d);
                w.write(d + s.nbits as usize, data);
            }
        }
    }

    let mut bits: Vec<[f32; ALPHABET_SIZE]> = code
        .context_map
        .iter()
        .map(|&c| cluster_bits[c as usize])
        .collect();
    if min_symbol.is_none() {
        // No distances yet: a neutral guess lets the first parse try copies.
        bits.push([6.0; ALPHABET_SIZE]);
    }
    let prices = IccPrices {
        config,
        min_symbol: min_symbol.unwrap_or_else(|| {
            let max_literal = symbols
                .iter()
                .map(|s| u32::from(s.symbol))
                .max()
                .unwrap_or(0);
            (max_literal + 1).max(8)
        }),
        bits,
    };
    Some((w, prices))
}

/// ANS tables only pay for themselves on longer streams.
const ANS_MIN_SYMBOLS: usize = 1000;
/// Coarse fixed context maps only win on short streams.
const FIXED_MAP_MAX_SYMBOLS: usize = 2000;

/// Coder choices `(use_ans, fixed context map)` for one tokenization:
/// automatic clustering, plus coarse fixed maps whose cheap context map pays
/// on small profiles.
fn variants(num_symbols: usize, exhaustive: bool) -> Vec<(bool, Option<Vec<u8>>)> {
    let n = K_NUM_ICC_CONTEXTS + 1;
    let maps = [
        vec![0u8; n],
        (0..n).map(|c| u8::from(c != 0)).collect(),
        (0..n)
            .map(|c| match c {
                0 => 0,
                DISTANCE_CONTEXT => 2,
                _ => 1,
            })
            .collect::<Vec<u8>>(),
    ];
    let short = num_symbols < FIXED_MAP_MAX_SYMBOLS;
    let mut out = vec![(false, None)];
    if short {
        out.extend(maps.iter().map(|m| (false, Some(m.clone()))));
    }
    if exhaustive || num_symbols >= ANS_MIN_SYMBOLS {
        out.push((true, None));
    }
    if exhaustive && short {
        out.extend(maps.iter().map(|m| (true, Some(m.clone()))));
    }
    out
}

/// Write the coder variants for one parse, keeping the smallest stream in
/// `best`. Returns the prices of the best automatically clustered code, whose
/// per-context detail steers the next parse; its config is `prices.config`.
fn evaluate_parse(
    enc: &[u8],
    contexts: &[u8],
    matches: &[IccMatch],
    configs: &[HybridUintConfig],
    exhaustive: bool,
    huffman_pool: &mut Vec<HuffmanNode>,
    best: &mut Option<BitWriter>,
) -> Option<IccPrices> {
    let mut seed: Option<(usize, IccPrices)> = None;
    for &config in configs {
        let Some((symbols, min_symbol)) = tokenize(enc, contexts, matches, config) else {
            continue;
        };
        let min_symbol = (!matches.is_empty()).then_some(min_symbol);
        for (use_ans, map) in variants(symbols.len(), exhaustive) {
            let Some((w, prices)) = write_symbols(
                &symbols,
                min_symbol,
                config,
                use_ans,
                map.as_deref(),
                huffman_pool,
            ) else {
                continue;
            };
            let bits = w.bits_written();
            if best.as_ref().is_none_or(|b| bits < b.bits_written()) {
                *best = Some(w);
            }
            if map.is_none() && seed.as_ref().is_none_or(|s| bits < s.0) {
                seed = Some((bits, prices));
            }
        }
    }
    seed.map(|(_, p)| p)
}

/// Smallest entropy-coded form of the predicted stream `enc`: literal-only
/// codes, then LZ77 parses steered by the prices of the previous codes.
fn encode_icc_tokens(enc: &[u8], huffman_pool: &mut Vec<HuffmanNode>) -> BitWriter {
    let contexts = icc_contexts(enc);
    let mut best = None;

    // Rank the uint configs by one exact literal-only prefix code each.
    let mut ranked: Vec<(usize, HybridUintConfig)> = ICC_UINT_CONFIGS
        .iter()
        .filter_map(|&config| {
            let (symbols, _) = tokenize(enc, &contexts, &[], config)?;
            let (w, _) = write_symbols(&symbols, None, config, false, None, huffman_pool)?;
            Some((w.bits_written(), config))
        })
        .collect();
    ranked.sort_by_key(|r| r.0);
    let mut configs: Vec<HybridUintConfig> = ranked.iter().take(2).map(|r| r.1).collect();

    let mut prices = evaluate_parse(
        enc,
        &contexts,
        &[],
        &configs[..1],
        true,
        huffman_pool,
        &mut best,
    )
    .map(|mut p| {
        // Unseen length symbols in a literal-only code: assume short copies.
        for row in p.bits.iter_mut().take(K_NUM_ICC_CONTEXTS) {
            for b in row.iter_mut().skip(p.min_symbol as usize) {
                *b = 5.0;
            }
        }
        p
    });
    for _ in 0..LZ77_PASSES {
        let Some(p) = prices.take() else { break };
        let matches = optimal_parse(enc, &contexts, &p);
        if matches.is_empty() {
            break;
        }
        let before = best.as_ref().map_or(usize::MAX, BitWriter::bits_written);
        prices = evaluate_parse(
            enc,
            &contexts,
            &matches,
            &configs,
            false,
            huffman_pool,
            &mut best,
        )
        .filter(|_| best.as_ref().is_some_and(|b| b.bits_written() < before));
        // Later passes keep the config the first parse settled on.
        if let Some(p) = &prices {
            configs = vec![p.config];
        }
    }
    best.expect("the default config always tokenizes")
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

/// Emit the JXL ICC stream right after the color encoding bits.
/// `icc` must be non-empty.
pub(crate) fn write_icc_stream(icc: &[u8], huffman_pool: &mut Vec<HuffmanNode>, w: &mut BitWriter) {
    assert!(!icc.is_empty(), "ICC profile must be non-empty");
    let enc = predict_icc(icc)
        .filter(|enc| unpredict_icc(enc).as_deref() == Some(icc))
        .unwrap_or_else(|| predict_icc_minimal(icc));
    write_u64(enc.len() as u64, w);
    w.append_bits(&encode_icc_tokens(&enc, huffman_pool));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small RGB display profile with the structures the predictor knows.
    fn synthetic_profile() -> Vec<u8> {
        let tags: [(Tag, Vec<u8>); 6] = [
            (*b"desc", {
                let mut v = b"mluc\0\0\0\0".to_vec();
                v.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 12]);
                v.extend_from_slice(b"enUS");
                v.extend_from_slice(&[0, 0, 0, 16, 0, 0, 0, 28]);
                for c in b"Synthetic RGB" {
                    v.extend_from_slice(&[0, *c]);
                }
                v
            }),
            (
                *b"wtpt",
                [
                    b"XYZ ".as_slice(),
                    &[0; 4],
                    &[0, 0, 0xF6, 0xD6, 0, 1, 0, 0, 0, 0, 0xD3, 0x2D],
                ]
                .concat(),
            ),
            (
                *b"rXYZ",
                [
                    b"XYZ ".as_slice(),
                    &[0; 4],
                    &[0, 0, 0x6F, 0xA2, 0, 0, 0x38, 0xF5, 0, 0, 3, 0x90],
                ]
                .concat(),
            ),
            (
                *b"gXYZ",
                [
                    b"XYZ ".as_slice(),
                    &[0; 4],
                    &[0, 0, 0x62, 0x99, 0, 0, 0xB7, 0x85, 0, 0, 0x18, 0xDA],
                ]
                .concat(),
            ),
            (
                *b"bXYZ",
                [
                    b"XYZ ".as_slice(),
                    &[0; 4],
                    &[0, 0, 0x24, 0xA0, 0, 0, 0x0F, 0x84, 0, 0, 0xB6, 0xCF],
                ]
                .concat(),
            ),
            (*b"rTRC", {
                let mut v = b"curv\0\0\0\0".to_vec();
                v.extend_from_slice(&256u32.to_be_bytes());
                for i in 0..256u32 {
                    let y = ((i as f64 / 255.0).powf(2.2) * 65535.0) as u16;
                    v.extend_from_slice(&y.to_be_bytes());
                }
                v
            }),
        ];
        let num_tags = tags.len() + 2; // gTRC and bTRC share rTRC's data
        let mut offset = K_ICC_HEADER_SIZE + 4 + num_tags * 12;
        let mut table = Vec::new();
        let mut body = Vec::new();
        let mut trc = (0, 0);
        for (tag, data) in &tags {
            table.extend_from_slice(tag);
            table.extend_from_slice(&(offset as u32).to_be_bytes());
            table.extend_from_slice(&(data.len() as u32).to_be_bytes());
            if tag == b"rTRC" {
                trc = (offset, data.len());
            }
            body.extend_from_slice(data);
            while body.len() % 4 != 0 {
                body.push(0);
            }
            offset = K_ICC_HEADER_SIZE + 4 + num_tags * 12 + body.len();
        }
        for tag in [b"gTRC", b"bTRC"] {
            table.extend_from_slice(tag);
            table.extend_from_slice(&(trc.0 as u32).to_be_bytes());
            table.extend_from_slice(&(trc.1 as u32).to_be_bytes());
        }
        let size = K_ICC_HEADER_SIZE + 4 + table.len() + body.len();
        let mut icc = icc_initial_header_prediction(size as u64).to_vec();
        icc[4..8].copy_from_slice(b"jixl");
        icc[40..44].copy_from_slice(b"APPL");
        icc.extend_from_slice(&(num_tags as u32).to_be_bytes());
        icc.extend_from_slice(&table);
        icc.extend_from_slice(&body);
        icc
    }

    #[test]
    fn varint_round_trip() {
        for v in [
            0u64,
            1,
            126,
            127,
            128,
            255,
            16383,
            16384,
            12345678,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            encode_varint(v, &mut buf);
            let mut pos = 0;
            assert_eq!(decode_varint(&buf, buf.len(), &mut pos), Some(v));
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn shuffle_inverts_unshuffle() {
        for len in 0..40 {
            for width in [2, 4] {
                let orig: Vec<u8> = (0..len as u8).collect();
                let mut v = orig.clone();
                unshuffle(&mut v, width);
                shuffle(&mut v, width);
                assert_eq!(v, orig, "len {len} width {width}");
            }
        }
    }

    #[test]
    fn iccans_context_in_bounds() {
        for i in 0..200 {
            for b1 in 0..=255u8 {
                for b2 in [0u8, 1, 16, 100, 200, 255] {
                    assert!(iccans_context(i, b1, b2) < K_NUM_ICC_CONTEXTS);
                }
            }
        }
    }

    #[test]
    fn prediction_round_trips() {
        let icc = synthetic_profile();
        let enc = predict_icc(&icc).expect("predictable");
        assert_eq!(unpredict_icc(&enc).as_deref(), Some(icc.as_slice()));
        // The tag list and curve prediction must actually engage.
        assert!(enc.iter().filter(|&&b| b == 0).count() > icc.iter().filter(|&&b| b == 0).count());
        let minimal = predict_icc_minimal(&icc);
        assert_eq!(unpredict_icc(&minimal).as_deref(), Some(icc.as_slice()));
    }

    #[test]
    fn prediction_survives_arbitrary_bytes() {
        let base = synthetic_profile();
        let mut state = 0x1234_5678u32;
        for round in 0..300 {
            let mut icc = base.clone();
            for _ in 0..1 + round % 8 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let at = state as usize % icc.len();
                icc[at] = (state >> 24) as u8;
            }
            icc.truncate(1 + (state as usize >> 7) % (icc.len() + 64).min(icc.len()));
            if let Some(enc) = predict_icc(&icc) {
                // Allowed to be rejected by the verifier, never to panic.
                if let Some(back) = unpredict_icc(&enc) {
                    assert_eq!(back, icc);
                }
            }
            let minimal = predict_icc_minimal(&icc);
            assert_eq!(unpredict_icc(&minimal).as_deref(), Some(icc.as_slice()));
            let mut w = BitWriter::new();
            write_icc_stream(&icc, &mut Vec::new(), &mut w);
        }
    }

    #[test]
    fn lz77_parse_covers_stream() {
        let enc = predict_icc(&synthetic_profile()).unwrap();
        let contexts = icc_contexts(&enc);
        let prices = IccPrices {
            config: HybridUintConfig::DEFAULT,
            min_symbol: 32,
            bits: vec![[4.0; ALPHABET_SIZE]; K_NUM_ICC_CONTEXTS + 1],
        };
        let matches = optimal_parse(&enc, &contexts, &prices);
        let mut end = 0;
        for m in &matches {
            assert!(m.pos >= end && m.distance >= 1 && m.distance <= m.pos);
            assert!(m.len >= LZ77_MIN_LENGTH);
            assert_eq!(
                enc[m.pos - m.distance..m.pos - m.distance + m.len].len(),
                m.len
            );
            for k in 0..m.len {
                assert_eq!(enc[m.pos + k], enc[m.pos - m.distance + k]);
            }
            end = m.pos + m.len;
        }
    }
}
