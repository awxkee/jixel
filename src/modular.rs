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
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code,
    write_prefix_codes, write_token,
};

const TREE_CTX_SPLITVAL: u32 = 0;
const TREE_CTX_PROPERTY: u32 = 1;
const TREE_CTX_PREDICTOR: u32 = 2;
const TREE_CTX_OFFSET: u32 = 3;
const TREE_CTX_MULTIPLIER_LOG: u32 = 4;
const TREE_CTX_MULTIPLIER_BITS: u32 = 5;
const NUM_TREE_CONTEXTS: usize = 6;
const PREDICTOR_GRADIENT: u32 = 5;

/// AC group dimension in pixels.  Matches `K_GROUP_DIM` in enc_frame.rs.
const GROUP_DIM: usize = 256;

/// Write the LfGlobal modular section for the alpha channel.
///
/// * **Small images** (`xsize ≤ GROUP_DIM && ysize ≤ GROUP_DIM`): writes the
///   complete modular section — GroupHeader + local tree + pixel histograms +
///   all pixel tokens — into `w` (the LfGlobal section writer).
///
/// * **Large images**: writes only the 4-bit GroupHeader
///   (`use_global_tree=1`).  Per-AC-group pixel tokens are written later by
///   [`write_ac_group_alpha`].
pub(crate) fn write_lfglobal_alpha_section(
    alpha: &AlphaPlane,
    xsize: usize,
    ysize: usize,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    assert_eq!(alpha.len(), xsize * ysize);
    if xsize <= GROUP_DIM && ysize <= GROUP_DIM {
        // Small path: everything in the LfGlobal section.
        write_group_header_local_tree(w);
        let tokens = tokenize_channel(alpha, xsize, ysize, 0, 0, xsize, ysize, xsize);
        let pixel_code = build_pixel_code(&tokens, scratch);
        write_tree_and_pixel_histograms(&pixel_code, scratch, w);
        let code_ref = pixel_code.as_ref();
        for tok in &tokens {
            write_token(*tok, &code_ref, w);
        }
    } else {
        // Large path: just the GroupHeader; pixels come per AC group.
        write_group_header_global_tree(w);
    }
}

/// Convenience shim so callers that already pass `write_global_alpha_modular`
/// continue to compile without changes.  Delegates to
/// [`write_lfglobal_alpha_section`].
#[inline]
pub(crate) fn write_global_alpha_modular(
    alpha: &AlphaPlane,
    xsize: usize,
    ysize: usize,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    write_lfglobal_alpha_section(alpha, xsize, ysize, scratch, w);
}

pub(crate) fn write_ac_group_alpha(
    alpha: &AlphaPlane,
    full_xsize: usize,
    full_ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    // Small-image path: alpha lives entirely in LfGlobal; nothing per group.
    if full_xsize <= GROUP_DIM && full_ysize <= GROUP_DIM {
        return;
    }

    let mut tokens: Vec<Token> = Vec::with_capacity(gw * gh);
    for gy in 0..gh {
        let img_y = y0 + gy;
        for gx in 0..gw {
            let img_x = x0 + gx;
            let v = alpha.get_i32(img_y * full_xsize + img_x);
            let w_ = if gx > 0 {
                alpha.get_i32(img_y * full_xsize + img_x - 1)
            } else {
                0
            };
            let n_ = if gy > 0 {
                alpha.get_i32((img_y - 1) * full_xsize + img_x)
            } else {
                0
            };
            let nw_ = if gx > 0 && gy > 0 {
                alpha.get_i32((img_y - 1) * full_xsize + img_x - 1)
            } else {
                0
            };
            let pred = gradient(w_, n_, nw_);
            let ctx = if w_ + n_ - nw_ > 0 { 0u32 } else { 1u32 };
            tokens.push(Token::new(ctx, pack_signed(v - pred)));
        }
    }

    const NUM_CTX: usize = 2; // leaf 0 (grad > 0), leaf 1 (grad <= 0)
    let pixel_code = build_pixel_code_n(&tokens, NUM_CTX, scratch);
    write_group_header_local_tree(w); // use_global_tree=0, wp default, 0 transforms
    write_split_tree(scratch, w); // 2-leaf Gradient tree, split on property 9 at 0
    w.write(1, 0); // no LZ77 for the pixel entropy code
    write_entropy_code(&pixel_code.as_ref(), &mut scratch.huffman_pool, w); // context map + 2 prefix codes
    let code_ref = pixel_code.as_ref();
    for tok in &tokens {
        write_token(*tok, &code_ref, w);
    }
}

/// Collapse runs of identical packed residuals in RASTER order into LZ77 copies.
/// A run of L identical values is emitted as one literal (first pixel) followed,
/// when `L-1 >= LZ77_MIN_LENGTH`, by a copy of `L-1` covering the rest. The copy
/// length symbol is coded on the context of the first copied pixel (the pixel
/// right after the literal), matching how the decoder reads it.
#[allow(dead_code)]
fn lz77_compress_alpha(tokens: &[Token]) -> Vec<crate::enc_lz77_ac::AcLz> {
    use crate::enc_lz77_ac::{AcLz, LZ77_MIN_LENGTH};
    let mut out: Vec<AcLz> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let v = tokens[i].value;
        // Always emit the first element of the run as a literal (on its context).
        out.push(AcLz::Lit {
            context: tokens[i].context,
            value: v,
        });
        // Extend the run over identical *values* (context is irrelevant during a
        // copy — the decoder ignores it while replaying window values).
        let mut j = i + 1;
        while j < tokens.len() && tokens[j].value == v {
            j += 1;
        }
        let run_extra = (j - i - 1) as u32; // copied pixels after the literal
        if run_extra >= LZ77_MIN_LENGTH {
            // Copy covers pixels [i+1, j); length symbol coded on pixel (i+1)'s
            // context, distance 1 (repeat previous value).
            out.push(AcLz::Copy {
                context: tokens[i + 1].context,
                length_value: run_extra - LZ77_MIN_LENGTH,
            });
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// GroupHeader: use_global_tree=0, wp_header.all_default=1, 0 transforms.
/// (4 bits)  Used for the LfGlobal local-tree path.
pub(crate) fn write_group_header_local_tree(w: &mut BitWriter) {
    w.write(1, 0); // use_global_tree = false
    w.write(1, 1); // wp_header all_default
    w.write(2, 0); // 0 transforms
}

/// GroupHeader: use_global_tree=1, wp_header.all_default=1, 0 transforms.
/// (4 bits)  Used for LfGlobal and per-AC-group headers in the large-image path.
fn write_group_header_global_tree(w: &mut BitWriter) {
    w.write(1, 1); // use_global_tree = true
    w.write(1, 1); // wp_header all_default
    w.write(2, 0); // 0 transforms
}

pub(crate) fn write_tree_and_pixel_histograms(
    pixel_code: &OwnedEntropyCode,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    let tree_tokens = [
        Token::new(TREE_CTX_PROPERTY, 0),
        Token::new(TREE_CTX_PREDICTOR, PREDICTOR_GRADIENT),
        Token::new(TREE_CTX_OFFSET, pack_signed(0)),
        Token::new(TREE_CTX_MULTIPLIER_LOG, 0),
        Token::new(TREE_CTX_MULTIPLIER_BITS, 0),
    ];
    let tree_code =
        optimize_entropy_code(&tree_tokens, NUM_TREE_CONTEXTS, &mut scratch.huffman_pool);
    let tree_code_ref = tree_code.as_ref();

    w.write(1, 0); // no LZ77 for tree entropy code
    write_entropy_code(&tree_code_ref, &mut scratch.huffman_pool, w);
    for tok in &tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }

    // num_contexts = 1: decoder skips context map entirely — write prefix
    // codes directly (no write_context_map).
    w.write(1, 0); // no LZ77 for pixel entropy code
    write_prefix_codes(
        &pixel_code.prefix_codes,
        &pixel_code.hybrid_uint_configs,
        &mut scratch.huffman_pool,
        w,
    );
}

/// Like `write_tree_and_pixel_histograms` but emits a 2-leaf tree that splits on
/// modular property 9 (the raw gradient W+N-NW) at splitval 0. Pixels with
/// grad > 0 take leaf 0, others leaf 1; both leaves predict with Gradient. The
/// tree is decoded breadth-first: a split node (property+1, splitval) followed
/// by its two leaf nodes. Two pixel prefix codes follow (one per leaf), written
/// via the context map the decoder builds from num_contexts = 2.
/// Write the 2-leaf alpha tree: a split node on property 9 (raw gradient) at
/// splitval 0, followed by two Gradient-predictor leaves. The pixel entropy
/// code is written separately by the caller (so it can choose plain vs LZ77).
#[allow(dead_code)]
fn write_split_tree(scratch: &mut CoderScratch, w: &mut BitWriter) {
    // Property index 9 = raw gradient (W+N-NW) in libjxl's property order.
    const PROP_GRAD: u32 = 9;
    let tree_tokens = [
        // Split node: property = PROP_GRAD (encoded as property+1), splitval 0.
        Token::new(TREE_CTX_PROPERTY, PROP_GRAD + 1),
        Token::new(TREE_CTX_SPLITVAL, pack_signed(0)),
        // Leaf 0 (grad > 0): Gradient predictor, offset 0, multiplier 1.
        Token::new(TREE_CTX_PROPERTY, 0),
        Token::new(TREE_CTX_PREDICTOR, PREDICTOR_GRADIENT),
        Token::new(TREE_CTX_OFFSET, pack_signed(0)),
        Token::new(TREE_CTX_MULTIPLIER_LOG, 0),
        Token::new(TREE_CTX_MULTIPLIER_BITS, 0),
        // Leaf 1 (grad <= 0): same.
        Token::new(TREE_CTX_PROPERTY, 0),
        Token::new(TREE_CTX_PREDICTOR, PREDICTOR_GRADIENT),
        Token::new(TREE_CTX_OFFSET, pack_signed(0)),
        Token::new(TREE_CTX_MULTIPLIER_LOG, 0),
        Token::new(TREE_CTX_MULTIPLIER_BITS, 0),
    ];
    let tree_code =
        optimize_entropy_code(&tree_tokens, NUM_TREE_CONTEXTS, &mut scratch.huffman_pool);
    let tree_code_ref = tree_code.as_ref();

    w.write(1, 0); // no LZ77 for tree entropy code
    write_entropy_code(&tree_code_ref, &mut scratch.huffman_pool, w);
    for tok in &tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }
}

/// Estimate the encoded size in bits of the plain (non-LZ) pixel token stream.
#[allow(dead_code)]
fn estimate_plain_bits(tokens: &[Token], code: &OwnedEntropyCode) -> u64 {
    let code_ref = code.as_ref();
    let mut bits: u64 = 0;
    for t in tokens {
        let (sym, nbits, _) = crate::entropy::uint_encode(t.value);
        let cl = code_ref.context_map[t.context as usize] as usize;
        let pc = &code_ref.prefix_codes[cl];
        // Single-symbol contexts cost 0 code bits (only extra bits).
        let d = if pc.single_symbol {
            0
        } else {
            pc.depths[sym as usize] as u64
        };
        bits += d + nbits as u64;
    }
    bits
}

pub(crate) fn build_pixel_code(tokens: &[Token], scratch: &mut CoderScratch) -> OwnedEntropyCode {
    build_pixel_code_n(tokens, 1, scratch)
}

fn build_pixel_code_n(
    tokens: &[Token],
    num_contexts: usize,
    scratch: &mut CoderScratch,
) -> OwnedEntropyCode {
    let mut code = if num_contexts > 1 {
        // Keep contexts separate — the whole point of the multi-leaf alpha tree
        // is that the bulk-zero leaf must not be merged with the rare-corner
        // leaf, or it loses its single-symbol (zero-bit) property.
        crate::entropy::build_entropy_code_no_cluster(
            tokens,
            num_contexts,
            &mut scratch.huffman_pool,
        )
    } else {
        optimize_entropy_code(tokens, num_contexts, &mut scratch.huffman_pool)
    };
    for pc in &mut code.prefix_codes {
        // Count non-zero depths and remember the position of the only one (if any).
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
            // Single-symbol case.
            //
            // JXL has two distinct encodings for a degenerate prefix code:
            //
            //   (a) alphabet_size == 1: the decoder consumes ZERO bits per token
            //       and emits token value 0 every time. This only works if the
            //       single live symbol is token 0 — otherwise the decoder gets
            //       the wrong token AND leaves our extra-bit payload unconsumed,
            //       desynchronising the bitstream.
            //
            //   (b) A valid 2-symbol simple Huffman code where both symbols share
            //       a 1-bit code. The encoder always picks our real token (cost:
            //       1 wasted bit per pixel), the decoder synchronizes correctly.
            //
            // CreateHuffmanTree gives us depth=1 with bits=0 for the placeholder,
            // but the actual encoded value would be `0` — wrong if our token > 0.
            // We use (a) only when the symbol IS token 0, and (b) otherwise.
            if idx == 0 {
                // Plain (a): zero depth & bits so write_prefix_code emits the
                // "alphabet_size == 1" header (4 bits, value 1) and write_token
                // emits zero bits.
                pc.depths[idx] = 0;
                pc.bits[idx] = 0;
            } else {
                // (b): synthesize a second symbol at index 0 with depth 1 and
                // assign codes {0 → "0", idx → "1"} so the encoder always emits
                // "1" + extra bits, and the decoder follows the same logic.
                pc.depths[0] = 1;
                pc.bits[0] = 0;
                pc.depths[idx] = 1;
                pc.bits[idx] = 1;
            }
        }
        pc.update_single_symbol();
    }
    code
}

#[inline]
pub(crate) fn gradient(w: i32, n: i32, nw: i32) -> i32 {
    let lo = w.min(n);
    let hi = w.max(n);
    (w + n - nw).clamp(lo, hi)
}

/// Tokenize a rectangular slice [x0..x0+gw, y0..y0+gh] of `alpha` (which
/// has the given `full_xsize` stride) using context = 0.
fn tokenize_channel(
    alpha: &AlphaPlane,
    _full_xsize: usize,
    _full_ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    stride: usize,
) -> Vec<Token> {
    let _ = _full_ysize;
    let mut tokens = Vec::with_capacity(gw * gh);
    for y in y0..y0 + gh {
        for x in x0..x0 + gw {
            let v = alpha.get_i32(y * stride + x);
            let w_ = if x > 0 {
                alpha.get_i32(y * stride + x - 1)
            } else {
                0
            };
            let n_ = if y > 0 {
                alpha.get_i32((y - 1) * stride + x)
            } else {
                0
            };
            let nw_ = if x > 0 && y > 0 {
                alpha.get_i32((y - 1) * stride + x - 1)
            } else {
                0
            };
            let pred = gradient(w_, n_, nw_);
            tokens.push(Token::new(0, pack_signed(v - pred)));
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode_image::AlphaPlane;

    #[test]
    fn gradient_basic() {
        assert_eq!(gradient(50, 50, 0), 50);
        assert_eq!(gradient(50, 50, 100), 50);
        assert_eq!(gradient(10, 20, 5), 20);
        assert_eq!(gradient(10, 20, 25), 10);
        assert_eq!(gradient(10, 20, 15), 15);
    }

    #[test]
    fn tokenize_constant_zero() {
        let chan = vec![0u8; 16 * 16];
        let alpha = AlphaPlane::from_u8(chan);
        let toks = tokenize_channel(&alpha, 16, 16, 0, 0, 16, 16, 16);
        for t in &toks {
            assert_eq!(t.value, 0);
        }
    }

    #[test]
    fn write_alpha_small_emits_bytes() {
        let mut w = BitWriter::new();
        let chan = vec![128u8; 8 * 8];
        let alpha = AlphaPlane::from_u8(chan);
        let mut scratch = CoderScratch::default();
        write_global_alpha_modular(&alpha, 8, 8, &mut scratch, &mut w);
        let bits = w.bits_written();
        w.zero_pad_to_byte();
        assert!(w.into_bytes().len() > 0);
        assert!(bits > 0);
    }

    #[test]
    fn write_alpha_large_emits_header_only() {
        let mut w = BitWriter::new();
        let chan = vec![200u8; 512 * 400];
        let alpha = AlphaPlane::from_u8(chan);
        let mut scratch = CoderScratch::default();
        write_lfglobal_alpha_section(&alpha, 512, 400, &mut scratch, &mut w);
        // Large path: only 4 bits (GroupHeader).
        assert_eq!(w.bits_written(), 4);
    }
}
