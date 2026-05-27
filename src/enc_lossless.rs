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

const TREE_CTX_SPLIT_VAL: u32 = 0;
const TREE_CTX_PROPERTY: u32 = 1;
const TREE_CTX_PREDICTOR: u32 = 2;
const TREE_CTX_OFFSET: u32 = 3;
const TREE_CTX_MULTIPLIER_LOG: u32 = 4;
const TREE_CTX_MULTIPLIER_BITS: u32 = 5;
const NUM_TREE_CONTEXTS: usize = 6;
const PREDICTOR_GRADIENT: u32 = 5;

const GROUP_DIM: usize = 256;
const LF_GROUP_DIM: usize = 2048;

// ---------------------------------------------------------------------------
// LZ77 parameters.
// ---------------------------------------------------------------------------
//
// JXL's prefix-code alphabet is shared between regular value tokens and LZ77
// length codes.  Length codes occupy alphabet slots `min_symbol .. min_symbol
// + kNumLZ77`.  The length value (`run_length - min_length`) is hybrid-encoded
// with `length_uint_config`, producing a token offset that's added to
// `min_symbol`.
//
// We pick:
//   min_symbol = 64 (safely above the largest regular token we emit, even for
//                    12-bit YCoCg residuals where the max is ~59).
//   min_length = 3  (smallest run length we'll back-reference).
//   length_uint_config = (4, 0, 0): values 0..15 use the symbol directly, with
//                       no extra bits; values 16+ are hybrid-encoded.
//
// We only emit distance=1 references (consecutive runs of identical tokens),
// which means the distance histogram has a single symbol (0).
//
// All alphabet symbols therefore fit within `ALPHABET_SIZE = 128`.

const LZ77_MIN_SYMBOL: u32 = 64;
const LZ77_MIN_LENGTH: u32 = 3;
// Distance value we emit for run-length encoding (distance = 1 → previous token).
const LZ77_DIST_VALUE: u32 = 1; // special_distance[1] = (dx=1, dy=0) → 1 token back

pub(crate) fn encode_frame_lossless(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    writer: &mut BitWriter,
) {
    let xsize = linear.xsize();
    let ysize = linear.ysize();
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };

    let xsize_groups = div_ceil(xsize, GROUP_DIM);
    let ysize_groups = div_ceil(ysize, GROUP_DIM);
    let num_ac_groups = xsize_groups * ysize_groups;
    let xsize_dc_groups = div_ceil(xsize, LF_GROUP_DIM);
    let ysize_dc_groups = div_ceil(ysize, LF_GROUP_DIM);
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
    let single_group = num_ac_groups == 1;

    write_frame_header_modular(alpha.is_some(), writer);

    if single_group {
        // Single section: GroupHeader + local tree + pixel histograms + pixels.
        let mut section = BitWriter::new();
        // 1 bit: dc_quant all_default = 1
        section.write(1, 1);
        // 1 bit: has_tree = 0  (no global tree; the local tree lives in the GroupHeader).
        section.write(1, 0);
        // GroupHeader: use_global_tree=0, wp_default=1, RCT transform on R/G/B.
        section.write(1, 0);
        section.write(1, 1);
        write_modular_transforms(nb_chans, &mut section);

        // Tokenize all channels (post-YCoCg, per-channel contexts).
        let tokens = tokenize_all(linear, alpha, xsize, ysize, 0, 0, xsize, ysize);

        // LZ77 layer: collapse runs of identical tokens into back-references.
        // The distance context is the (nb_chans)-th context, appended after the
        // per-channel ones.
        let distance_ctx = nb_chans as u32;
        let lz_tokens = lz77_compress(&tokens, distance_ctx);

        // Per-cluster prefix codes (nb_chans + 1 contexts), balanced N-leaf tree.
        let code = build_lz_pixel_code(&lz_tokens, nb_chans);
        write_local_tree_lz77(nb_chans, &code, &mut section);

        // Emit the LZ77'd token stream.
        for t in &lz_tokens {
            write_lz_token(*t, &code, &mut section);
        }
        section.zero_pad_to_byte();

        // TOC.
        writer.write(1, 0); // no permutation
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append(&section);
        writer.zero_pad_to_byte();
    } else {
        // Multi-group: a single global tree + histograms in DC global, then each
        // AC group emits its tokens against those codes.
        let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
        let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

        // Tokenize each AC group (sub-image-local neighbours, matching what
        // we'll emit below) and run LZ77 over each group's stream separately so
        // back-references stay within a group's modular sub-image.  Pool the
        // resulting LzToken streams to build a single global prefix code so
        // every per-group emission is guaranteed to be representable.
        let distance_ctx = nb_chans as u32;
        let mut group_lz_tokens: Vec<Vec<LzToken>> = Vec::with_capacity(num_ac_groups);
        let mut all_lz: Vec<LzToken> = Vec::new();
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);
                let toks = tokenize_all(linear, alpha, xsize, ysize, x0, y0, gw, gh);
                let lz = lz77_compress(&toks, distance_ctx);
                all_lz.extend_from_slice(&lz);
                group_lz_tokens.push(lz);
            }
        }
        let code = build_lz_pixel_code(&all_lz, nb_chans);

        // ----- Section 0: DC global -----
        sections[0].write(1, 1); // dc_quant all_default = 1
        sections[0].write(1, 1); // has_tree = 1
        write_local_tree_lz77(nb_chans, &code, &mut sections[0]);
        // GroupHeader for the global modular image: use_global_tree=1, wp=1, RCT transform.
        sections[0].write(1, 1);
        sections[0].write(1, 1);
        write_modular_transforms(nb_chans, &mut sections[0]);
        sections[0].zero_pad_to_byte();

        // ----- DC groups: empty GroupHeader only -----
        for i in 0..num_dc_groups {
            sections[1 + i].write(1, 1); // use_global_tree
            sections[1 + i].write(1, 1); // wp_default
            sections[1 + i].write(2, 0); // 0 transforms
            sections[1 + i].zero_pad_to_byte();
        }

        // ----- AC global: trivial (all_default flags) -----
        let ac_global_idx = 1 + num_dc_groups;
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].zero_pad_to_byte();

        // ----- AC groups: pixel data per group -----
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let group_index = gy * xsize_groups + gx;
                let section_idx = 2 + num_dc_groups + group_index;

                // GroupHeader: use_global_tree=1, wp=1, 0 transforms (the global
                // header already declared the RCT for the whole image).
                sections[section_idx].write(1, 1);
                sections[section_idx].write(1, 1);
                sections[section_idx].write(2, 0);

                for t in &group_lz_tokens[group_index] {
                    write_lz_token(*t, &code, &mut sections[section_idx]);
                }
                sections[section_idx].zero_pad_to_byte();
            }
        }

        // TOC.
        writer.write(1, 0);
        writer.zero_pad_to_byte();
        for s in &sections {
            write_toc_entry(s.bits_written() / 8, writer);
        }
        writer.zero_pad_to_byte();
        for s in &sections {
            writer.append(s);
            writer.zero_pad_to_byte();
        }
    }
}

// ---------------------------------------------------------------------------
// Frame header.
// ---------------------------------------------------------------------------

fn write_frame_header_modular(has_alpha: bool, w: &mut BitWriter) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    w.write(2, 0b00); // flags = u64(0)
    w.write(1, 0); // do_ycbcr = false   (xyb_encoded=0 so this is serialized)
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(2, 0b01); // group_size_shift = 1 (256-pixel groups)
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(1, 1); // is_last
    w.write(2, 0b00); // name length = 0
    w.write(1, 0); // loop_filter NOT all_default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // 0 EPF iters
    w.write(2, 0b00); // no LF extensions
    w.write(2, 0b00); // no FH extensions
}

// ---------------------------------------------------------------------------
// Modular transform header.
//
//   transforms count: u2S(0, Bits(2)+1, Bits(4)+5, Bits(8)+21)
//   per-Transform:
//     id            : Bits(2)       (0 = RCT, 1 = Palette, 2 = Squeeze)
//     begin_channel : u2S(Bits(3), Bits(6)+8, Bits(10)+72, Bits(13)+1096)
//     rct_type      : u2S(6, Bits(2), Bits(4)+2, Bits(6)+10)   (only if id == RCT)
//
// For nb_chans >= 3 we declare a single RCT transform with rct_type = 6
// (YCoCg) starting at channel 0.  For 1-channel (no transforms) we just
// emit `transforms=0` (2 bits).
// ---------------------------------------------------------------------------

fn write_modular_transforms(nb_chans: usize, w: &mut BitWriter) {
    if nb_chans >= 3 {
        // transforms count u2S(0, 1, Bits(4)+2, Bits(8)+18): selector 1 = Val(1) → 1 transform.
        w.write(2, 0b01);
        // Transform[0]:
        //   id Bits(2)            = 0 (RCT)
        //   begin_channel u2S(Bits(3), ...): selector 0 = Bits(3) → value 0 = 5 bits "00000"
        //   rct_type u2S(6, ...):  selector 0 = Val(6) → 2 bits "00"
        w.write(2, 0b00); // id = RCT (Bits(2))
        w.write(2, 0b00); // begin_channel selector 0
        w.write(3, 0); // begin_channel value (Bits(3)) = 0
        w.write(2, 0b00); // rct_type selector 0 = Val(6) = YCoCg
    } else {
        w.write(2, 0b00); // 0 transforms
    }
}

// ---------------------------------------------------------------------------
// TOC.
// ---------------------------------------------------------------------------

fn write_toc_entry(byte_len: usize, w: &mut BitWriter) {
    const OFFSETS: [usize; 4] = [0, 1024, 17_408, 4_211_712];
    const BITS: [usize; 4] = [10, 14, 22, 30];
    let mut bucket = 0usize;
    while bucket < 3 && byte_len >= OFFSETS[bucket + 1] {
        bucket += 1;
    }
    w.write(2, bucket as u64);
    w.write(BITS[bucket], (byte_len - OFFSETS[bucket]) as u64);
}

// ---------------------------------------------------------------------------
// Forward reversible YCoCg (RCT type 6, matches libjxl's InvRCTRow<6>).
//
// Encoder:
//   co  = r - b
//   tmp = b + (co >> 1)
//   cg  = g - tmp
//   y   = tmp + (cg >> 1)
//
// Decoder undoes this with the exact same shift sequence:
//   tmp = y - (cg >> 1);  g = cg + tmp;
//   y'  = tmp - (co >> 1); r = y' + co;  b = y'
//
// Reversible because every operation is invertible without rounding.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn forward_ycocg(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    (y, co, cg)
}

#[inline]
fn channel_to_context(chan: usize, nb_chans: usize) -> u32 {
    (nb_chans - 1 - chan) as u32
}

// ---------------------------------------------------------------------------
// LZ77 hybrid-uint encoding (length values: split_exp=4, msb=0, lsb=0).
// ---------------------------------------------------------------------------

/// Hybrid encode of `length_value` (= run_length - min_length).
/// Returns (alphabet_token, nbits, payload).  The actual alphabet symbol is
/// `LZ77_MIN_SYMBOL + alphabet_token`.
#[inline]
fn lz77_length_encode(length_value: u32) -> (u32, u32, u32) {
    // hybrid uint with split_exponent = 4, msb_in_token = 0, lsb_in_token = 0
    if length_value < 16 {
        (length_value, 0, 0)
    } else {
        let n = 31 - length_value.leading_zeros();
        let token = 16 + n - 4;
        let nbits = n;
        let bits = length_value - (1 << n);
        (token, nbits, bits)
    }
}

// ---------------------------------------------------------------------------
// LZ77 layer.
// ---------------------------------------------------------------------------

/// A unit of emission in the LZ77'd token stream.
#[derive(Clone, Copy)]
enum LzToken {
    /// A regular residual token, written via the normal `uint_encode` path.
    Pixel { context: u32, value: u32 },
    /// LZ77 back-reference: copy `length_value + LZ77_MIN_LENGTH` previous tokens.
    /// Emitted as a length symbol on the `pixel_context` channel, then a
    /// distance symbol on the distance context.
    Lz77 {
        pixel_context: u32,
        distance_context: u32,
        length_value: u32,
    },
}

/// Compact a sequence of `Token`s by back-referencing runs of identical
/// (context, value) pairs.  Returns the LZ77'd stream.
fn lz77_compress(tokens: &[Token], distance_context: u32) -> Vec<LzToken> {
    let mut out: Vec<LzToken> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        out.push(LzToken::Pixel {
            context: t.context,
            value: t.value,
        });
        // Find how many subsequent tokens are identical to `t`.
        let mut j = i + 1;
        while j < tokens.len() && tokens[j].context == t.context && tokens[j].value == t.value {
            j += 1;
        }
        let run_extra = (j - i - 1) as u32; // copies after the first one
        if run_extra >= LZ77_MIN_LENGTH {
            out.push(LzToken::Lz77 {
                pixel_context: t.context,
                distance_context,
                length_value: run_extra - LZ77_MIN_LENGTH,
            });
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn tokenize_all(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
) -> Vec<Token> {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };
    let mut out = Vec::with_capacity(gw * gh * nb_chans);

    for chan in 0..3usize {
        let ctx = channel_to_context(chan, nb_chans);
        for gy in 0..gh {
            let row = linear.plane_row(chan, y0 + gy);
            let prev_row = if gy > 0 {
                Some(linear.plane_row(chan, y0 + gy - 1))
            } else {
                None
            };
            for gx in 0..gw {
                let v = row[x0 + gx];
                let w_ = if gx > 0 { row[x0 + gx - 1] } else { 0 };
                let n_ = prev_row.map_or(0, |r| r[x0 + gx]);
                let nw_ = if gx > 0 {
                    prev_row.map_or(0, |r| r[x0 + gx - 1])
                } else {
                    0
                };
                let pred = gradient(w_, n_, nw_);
                out.push(Token::new(ctx, pack_signed(v - pred)));
            }
        }
    }

    // 3. Alpha (untransformed) under its own context.
    if let Some(a) = alpha {
        let ctx = channel_to_context(3, nb_chans);
        for gy in 0..gh {
            let img_y = y0 + gy;
            for gx in 0..gw {
                let img_x = x0 + gx;
                let v = a.get_i32(img_y * xsize + img_x);
                let w_ = if gx > 0 {
                    a.get_i32(img_y * xsize + img_x - 1)
                } else {
                    0
                };
                let n_ = if gy > 0 {
                    a.get_i32((img_y - 1) * xsize + img_x)
                } else {
                    0
                };
                let nw_ = if gx > 0 && gy > 0 {
                    a.get_i32((img_y - 1) * xsize + img_x - 1)
                } else {
                    0
                };
                let pred = gradient(w_, n_, nw_);
                out.push(Token::new(ctx, pack_signed(v - pred)));
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// LZ77-aware histogram building, code construction, and emission.
// ---------------------------------------------------------------------------

use crate::bit_writer::BitWriter;
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    Histogram, OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code,
    write_token,
};
use crate::image::Image3Si;
use crate::modular::gradient;

/// Build a frequency histogram for each cluster from an `LzToken` stream.
fn lz_build_histograms(
    toks: &[LzToken],
    context_map: &[u8],
    num_clusters: usize,
) -> Vec<Histogram> {
    let mut hs = vec![Histogram::new(); num_clusters];
    for t in toks {
        match *t {
            LzToken::Pixel { context, value } => {
                let (sym, _, _) = crate::entropy::uint_encode(value);
                let cluster = context_map[context as usize] as usize;
                hs[cluster].add(sym);
            }
            LzToken::Lz77 {
                pixel_context,
                distance_context,
                length_value,
            } => {
                let (len_tok, _, _) = lz77_length_encode(length_value);
                let pixel_cluster = context_map[pixel_context as usize] as usize;
                hs[pixel_cluster].add(LZ77_MIN_SYMBOL + len_tok);

                let dist_cluster = context_map[distance_context as usize] as usize;
                hs[dist_cluster].add(LZ77_DIST_VALUE);
            }
        }
    }
    hs
}

/// Build per-cluster prefix codes from an `LzToken` stream.
/// `nb_chans + 1` contexts: `nb_chans` channel leaves + 1 distance context.
fn build_lz_pixel_code(toks: &[LzToken], nb_chans: usize) -> OwnedEntropyCode {
    use crate::entropy::build_huffman_codes;
    use crate::entropy::cluster_histograms;

    let num_contexts = nb_chans + 1;
    let context_map_initial: Vec<u8> = (0..num_contexts).map(|i| i as u8).collect();
    let mut histograms = lz_build_histograms(toks, &context_map_initial, num_contexts);

    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map);

    let mut code = OwnedEntropyCode {
        context_map,
        prefix_codes: build_huffman_codes(&histograms),
        orig_context_map: None,
        orig_num_contexts: num_contexts,
    };

    // Apply the single-symbol patch (mirrors build_pixel_code) per cluster so
    // that contexts with one unique symbol still emit a parseable code.
    for pc in &mut code.prefix_codes {
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
    code
}

/// Emit one `LzToken` into the bitstream.
#[inline]
fn write_lz_token(t: LzToken, code: &OwnedEntropyCode, w: &mut BitWriter) {
    match t {
        LzToken::Pixel { context, value } => {
            let (sym, nbits, bits) = crate::entropy::uint_encode(value);
            let cluster = code.context_map[context as usize] as usize;
            let pc = &code.prefix_codes[cluster];
            let d = pc.depths[sym as usize] as usize;
            let data = (pc.bits[sym as usize] as u64) | ((bits as u64) << d);
            w.write(d + nbits as usize, data);
        }
        LzToken::Lz77 {
            pixel_context,
            distance_context,
            length_value,
        } => {
            let (len_tok, len_nbits, len_bits) = lz77_length_encode(length_value);
            let sym = LZ77_MIN_SYMBOL + len_tok;
            let pcluster = code.context_map[pixel_context as usize] as usize;
            let pc = &code.prefix_codes[pcluster];
            let d = pc.depths[sym as usize] as usize;
            debug_assert!(
                d > 0,
                "LZ77 length symbol {} unrepresented in histogram",
                sym
            );
            let data = (pc.bits[sym as usize] as u64) | ((len_bits as u64) << d);
            w.write(d + len_nbits as usize, data);

            // Distance symbol: value LZ77_DIST_VALUE = 0, no extra bits.
            let dcluster = code.context_map[distance_context as usize] as usize;
            let dc = &code.prefix_codes[dcluster];
            let dd = dc.depths[LZ77_DIST_VALUE as usize] as usize;
            // (Could be 0 if it's the only symbol in a single-symbol histogram.)
            if dd > 0 {
                w.write(dd, dc.bits[LZ77_DIST_VALUE as usize] as u64);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tree writing (balanced N-leaf, Gradient predictor).
//
// libjxl reads modular trees in BFS order: each node is one PROPERTY token
// (=0 for leaf, =prop+1 for split) optionally followed by a SPLIT_VAL token
// or 4 leaf-data tokens (predictor, offset, multiplier-log, multiplier-bits).
// All channels share the Gradient predictor with offset 0 and multiplier 1.
// ---------------------------------------------------------------------------

fn push_split(out: &mut Vec<Token>, property: u32, split_val: i32) {
    out.push(Token::new(TREE_CTX_PROPERTY, property + 1));
    out.push(Token::new(TREE_CTX_SPLIT_VAL, pack_signed(split_val)));
}

fn push_leaf(out: &mut Vec<Token>) {
    out.push(Token::new(TREE_CTX_PROPERTY, 0));
    out.push(Token::new(TREE_CTX_PREDICTOR, PREDICTOR_GRADIENT));
    out.push(Token::new(TREE_CTX_OFFSET, pack_signed(0)));
    out.push(Token::new(TREE_CTX_MULTIPLIER_LOG, 0));
    out.push(Token::new(TREE_CTX_MULTIPLIER_BITS, 0));
}

/// Build a balanced binary tree over `n_leaves` leaves splitting on property 0
/// (the channel index after RCT).  BFS leaf order is chan N-1, ..., chan 0.
fn build_balanced_tree_tokens(n_leaves: usize) -> Vec<Token> {
    let mut t = Vec::new();
    match n_leaves {
        1 => push_leaf(&mut t),
        2 => {
            push_split(&mut t, 0, 0);
            push_leaf(&mut t); // chan 1
            push_leaf(&mut t); // chan 0
        }
        3 => {
            // Tree:
            //        Split(0,1)
            //       /         \
            //   Leaf(c2)    Split(0,0)
            //               /        \
            //          Leaf(c1)   Leaf(c0)
            push_split(&mut t, 0, 1);
            push_leaf(&mut t); // chan 2
            push_split(&mut t, 0, 0);
            push_leaf(&mut t); // chan 1
            push_leaf(&mut t); // chan 0
        }
        4 => {
            // Tree (matches fjxl's BFS order chan3, chan2, chan1, chan0):
            //        Split(0,1)
            //       /         \
            //   Split(0,2)   Split(0,0)
            //   /    \       /     \
            // c3     c2     c1     c0
            push_split(&mut t, 0, 1);
            push_split(&mut t, 0, 2);
            push_split(&mut t, 0, 0);
            push_leaf(&mut t); // chan 3
            push_leaf(&mut t); // chan 2
            push_leaf(&mut t); // chan 1
            push_leaf(&mut t); // chan 0
        }
        _ => unreachable!("write_local_tree supports 1..=4 leaves"),
    }
    t
}

/// Write the LZ77 sub-bundle (matches `LZ77Params::VisitFields` and `DecodeUintConfig`):
///   1 bit:  enabled = 1
///   U32 min_symbol:  U32(Val(224), Val(512), Val(4096), BitsOffset(15, 8))
///                    For LZ77_MIN_SYMBOL = 64: selector 3 → "11" + 15 bits (64 - 8) = 56
///   U32 min_length:  U32(Val(3), Val(4), BitsOffset(2, 5), BitsOffset(8, 9))
///                    For LZ77_MIN_LENGTH = 3: selector 0 → "00"
///   length_uint_config: DecodeUintConfig(log_alpha_size = 8).
///                       split_exp = 4 (in CeilLog2(8+1) = 4 bits = "0010" LSB-first),
///                       msb_in_token = 0 (in CeilLog2(4+1) = 3 bits = "000"),
///                       lsb_in_token = 0 (in CeilLog2(4-0+1) = 3 bits = "000").
fn write_lz77_header(w: &mut BitWriter) {
    w.write(1, 1); // enabled
    // min_symbol: selector 3 (Bits(15) + 8), value 64 → payload = 56
    w.write(2, 0b11);
    w.write(15, (LZ77_MIN_SYMBOL - 8) as u64);
    // min_length: selector 0 (Val(3))
    w.write(2, 0b00);
    // length_uint_config (split=4, msb=0, lsb=0):
    w.write(4, 4);
    w.write(3, 0);
    w.write(3, 0);
}

/// Write the local tree + LZ77-enabled pixel histograms, then return.
/// The pixel `code` must have `nb_chans + 1` contexts (last = distance).
fn write_local_tree_lz77(n_leaves: usize, pixel_code: &OwnedEntropyCode, w: &mut BitWriter) {
    let tree_tokens = build_balanced_tree_tokens(n_leaves);
    let tree_code = optimize_entropy_code(&tree_tokens, NUM_TREE_CONTEXTS);
    let tree_code_ref = tree_code.as_ref();

    // Tree's entropy code: no LZ77 in the tree itself.
    w.write(1, 0);
    write_entropy_code(&tree_code_ref, w);
    for tok in &tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }

    // Pixel entropy code: LZ77 ENABLED for the main bitstream.
    write_lz77_header(w);
    // The decoder appends an extra context (distance) when LZ77 is on, so the
    // context map we write must already include it as its last entry.
    write_entropy_code(&pixel_code.as_ref(), w);
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Image3F;
    use crate::color::srgb_to_linear_u8;
    use crate::encode_image::{BitsPerSample, EncodeConfig, encode_with_config};

    fn make_image(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 3]) -> Image3F {
        let mut img = Image3F::new(w, h);
        for y in 0..h {
            let [r, g, b] = img.all_plane_rows_mut(y);
            for x in 0..w {
                let p = f(x, y);
                r[x] = srgb_to_linear_u8(p[0]);
                g[x] = srgb_to_linear_u8(p[1]);
                b[x] = srgb_to_linear_u8(p[2]);
            }
        }
        img
    }

    #[test]
    fn lossless_multi_group_produces_output() {
        // 300x300 > 256, so the encoder takes the multi-group path.
        let img = make_image(300, 300, |x, y| {
            [
                ((x + y) % 256) as u8,
                ((x * 3) % 256) as u8,
                ((y * 5) % 256) as u8,
            ]
        });
        let bytes = encode_with_config(&img, &EncodeConfig::default().with_lossless(true));
        assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
        assert!(
            bytes.len() > 100,
            "non-trivial frame should produce non-trivial output"
        );
    }

    #[test]
    fn lossless_10bit_emits_modular_frame() {
        use crate::color::srgb_to_linear_u16;
        let w = 32usize;
        let h = 32usize;
        let max = 1023.0_f32;
        let mut img = Image3F::new(w, h);
        for y in 0..h {
            let [r, g, b] = img.all_plane_rows_mut(y);
            for x in 0..w {
                r[x] = srgb_to_linear_u16(((x * 33) % 1024) as u16, max);
                g[x] = srgb_to_linear_u16(((y * 33) % 1024) as u16, max);
                b[x] = srgb_to_linear_u16(((x + y) * 16 % 1024) as u16, max);
            }
        }
        let cfg = EncodeConfig::default()
            .with_lossless(true)
            .with_bits_per_sample(BitsPerSample::Ten);
        let bytes = encode_with_config(&img, &cfg);
        assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
        assert!(bytes.len() > 100);
    }

    #[test]
    fn lossless_12bit_emits_modular_frame() {
        use crate::color::srgb_to_linear_u16;
        let w = 32usize;
        let h = 32usize;
        let max = 4095.0_f32;
        let mut img = Image3F::new(w, h);
        for y in 0..h {
            let [r, g, b] = img.all_plane_rows_mut(y);
            for x in 0..w {
                r[x] = srgb_to_linear_u16(((x * 131) % 4096) as u16, max);
                g[x] = srgb_to_linear_u16(((y * 131) % 4096) as u16, max);
                b[x] = srgb_to_linear_u16(((x + y) * 64 % 4096) as u16, max);
            }
        }
        let cfg = EncodeConfig::default()
            .with_lossless(true)
            .with_bits_per_sample(BitsPerSample::Twelve);
        let bytes = encode_with_config(&img, &cfg);
        assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
        assert!(bytes.len() > 100);
    }

    #[test]
    fn lz77_compress_collapses_run() {
        // 1 + 8 identical pixel tokens in the same context should collapse to:
        //   [first token, LZ77(length = 8 - 3 = 5)]
        let toks: Vec<Token> = (0..9).map(|_| Token::new(0, 0)).collect();
        let lz = lz77_compress(&toks, /*distance_context=*/ 1);
        assert_eq!(lz.len(), 2);
        match lz[0] {
            LzToken::Pixel { context, value } => assert_eq!((context, value), (0, 0)),
            _ => panic!(),
        }
        match lz[1] {
            LzToken::Lz77 {
                pixel_context,
                distance_context,
                length_value,
            } => {
                assert_eq!(pixel_context, 0);
                assert_eq!(distance_context, 1);
                assert_eq!(length_value, 8 - LZ77_MIN_LENGTH);
            }
            _ => panic!("expected an LZ77 token"),
        }
    }

    #[test]
    fn lz77_compress_leaves_short_runs_alone() {
        // Run of (1 + 2 = 3) is exactly at the LZ77_MIN_LENGTH threshold.
        // Our check is `run_extra >= LZ77_MIN_LENGTH`, run_extra = 2, so no compression.
        let toks: Vec<Token> = (0..3).map(|_| Token::new(0, 0)).collect();
        let lz = lz77_compress(&toks, 1);
        assert_eq!(lz.len(), 3);
    }

    #[test]
    fn lz77_length_encode_direct_vs_hybrid() {
        // Values 0..15: direct symbol, no extra bits.
        for v in 0..16u32 {
            let (tok, nbits, _) = lz77_length_encode(v);
            assert_eq!(tok, v);
            assert_eq!(nbits, 0);
        }
        // Values 16+: hybrid.  Value 16 → token 16, value 17 → token 16 with bits=1.
        let (tok16, nb16, b16) = lz77_length_encode(16);
        assert_eq!((tok16, nb16, b16), (16, 4, 0));
        let (tok31, nb31, b31) = lz77_length_encode(31);
        assert_eq!((tok31, nb31, b31), (16, 4, 15));
        let (tok32, nb32, _) = lz77_length_encode(32);
        assert_eq!(tok32, 17);
        assert_eq!(nb32, 5);
    }

    #[test]
    fn ycocg_is_reversible() {
        for r in 0..=255 {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(13) {
                    let (y, co, cg) = forward_ycocg(r, g, b);
                    let tmp = y - (cg >> 1);
                    let gg = cg + tmp;
                    let yy = tmp - (co >> 1);
                    let rr = yy + co;
                    let bb = yy;
                    assert_eq!((r, g, b), (rr, gg, bb), "YCoCg round-trip failed");
                }
            }
        }
    }
}
