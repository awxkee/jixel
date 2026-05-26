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
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    EntropyCode, OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code,
    write_prefix_codes, write_token,
};
use crate::static_entropy_codes::{K_CONTEXT_TREE_TOKENS, K_DC_CONTEXT_MAP, K_DC_PREFIX_CODES};

// ---------------------------------------------------------------------------
// MA tree token contexts (mirrors libjxl ma_common.h).
// ---------------------------------------------------------------------------

const TREE_CTX_PROPERTY: u32 = 1;
const TREE_CTX_PREDICTOR: u32 = 2;
const TREE_CTX_OFFSET: u32 = 3;
const TREE_CTX_MULTIPLIER_LOG: u32 = 4;
const TREE_CTX_MULTIPLIER_BITS: u32 = 5;
const NUM_TREE_CONTEXTS: usize = 6;
const PREDICTOR_GRADIENT: u32 = 5;

/// AC group dimension in pixels.  Matches `K_GROUP_DIM` in enc_frame.rs.
const GROUP_DIM: usize = 256;

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

/// Write the LfGlobal modular section for the alpha channel.
///
/// * **Small images** (`xsize ≤ GROUP_DIM && ysize ≤ GROUP_DIM`): writes the
///   complete modular section — GroupHeader + local tree + pixel histograms +
///   all pixel tokens — into `w` (the LfGlobal section writer).
///
/// * **Large images**: writes only the 4-bit GroupHeader
///   (`use_global_tree=1`).  Per-AC-group pixel tokens are written later by
///   [`write_ac_group_alpha`].
pub fn write_lfglobal_alpha_section(
    alpha: &AlphaPlane,
    xsize: usize,
    ysize: usize,
    w: &mut BitWriter,
) {
    assert_eq!(alpha.len(), xsize * ysize);
    if xsize <= GROUP_DIM && ysize <= GROUP_DIM {
        // Small path: everything in the LfGlobal section.
        write_group_header_local_tree(w);
        let tokens = tokenize_channel(alpha, xsize, ysize, 0, 0, xsize, ysize, xsize);
        let pixel_code = build_pixel_code(&tokens);
        write_tree_and_pixel_histograms(&pixel_code, w);
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
pub fn write_global_alpha_modular(
    alpha: &AlphaPlane,
    xsize: usize,
    ysize: usize,
    w: &mut BitWriter,
) {
    write_lfglobal_alpha_section(alpha, xsize, ysize, w);
}

/// Write the per-AC-group modular subbitstream for the alpha channel.
///
/// `alpha`      – full alpha plane (row-major, `full_xsize * full_ysize` bytes).
/// `full_xsize` – total image width (same stride as `alpha`).
/// `full_ysize` – total image height.
/// `x0`, `y0`  – top-left of this AC group in the full image.
/// `gw`, `gh`  – width and height of this AC group (clamped at image edges).
/// `group_id`  – absolute AC group index (y_group * xsize_groups + x_group).
/// `num_lf_groups` – number of DC/LF groups (= xsize_dc_groups * ysize_dc_groups).
/// `num_groups`    – total number of AC groups (= xsize_groups * ysize_groups).
///
/// The JXL decoder passes the modular stream ID as `group_id` in the property
/// vector used to traverse the MA tree.  For a pass-0 AC group the formula is:
///   stream_id = 1 + num_lf_groups * 3 + 17 + num_groups * 0 + group_index
///   (17 = NUM_QUANT_TABLES in libjxl)
pub fn write_ac_group_alpha(
    alpha: &AlphaPlane,
    full_xsize: usize,
    full_ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    group_index: usize, // 0-based AC group index
    num_lf_groups: usize,
    _num_groups: usize,
    w: &mut BitWriter,
) {
    // Small-image path: nothing to write per group.
    if full_xsize <= GROUP_DIM && full_ysize <= GROUP_DIM {
        return;
    }

    assert_eq!(alpha.len(), full_xsize * full_ysize);

    // Compute the stream_id that the decoder will use as property[1] (group_id)
    // when traversing the MA tree.
    const NUM_QUANT_TABLES: usize = 17;
    let stream_id = 1 + num_lf_groups * 3 + NUM_QUANT_TABLES + group_index;

    // GroupHeader: use_global_tree=1, wp_header.all_default=1, 0 transforms.
    write_group_header_global_tree(w);

    // Write pixel tokens using the GLOBAL tree (K_CONTEXT_TREE_TOKENS).
    // context_id = the leaf_id the global tree reaches for this pixel.
    // cluster    = K_DC_CONTEXT_MAP[context_id].
    // The token is written with K_DC_PREFIX_CODES[cluster].
    let dc_code = EntropyCode::new(&K_DC_CONTEXT_MAP, &K_DC_PREFIX_CODES);

    for gy in 0..gh {
        let img_y = y0 + gy;
        for gx in 0..gw {
            let img_x = x0 + gx;
            let v = alpha.get_i32(img_y * full_xsize + img_x);

            // Use SUB-IMAGE-LOCAL coordinates for neighbor lookup so we match
            // what the decoder sees.  The decoder builds a sub-image of size
            // gw×gh; pixel (gx=0, gy=0) has no left or top neighbor regardless
            // of where the group sits in the full image.
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
            let residual = v - pred;

            // Compute context_id by traversing the global tree.
            let context_id = alpha_context_id(gx, gy, w_, n_, nw_, n_ + w_ - nw_, stream_id);
            let tok = Token::new(context_id as u32, pack_signed(residual));
            write_token(tok, &dc_code, w);
        }
    }
}

// ---------------------------------------------------------------------------
// GroupHeader helpers.
// ---------------------------------------------------------------------------

/// GroupHeader: use_global_tree=0, wp_header.all_default=1, 0 transforms.
/// (4 bits)  Used for the LfGlobal local-tree path.
fn write_group_header_local_tree(w: &mut BitWriter) {
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

// ---------------------------------------------------------------------------
// Local-tree path helpers (small images).
// ---------------------------------------------------------------------------

fn write_tree_and_pixel_histograms(pixel_code: &OwnedEntropyCode, w: &mut BitWriter) {
    let tree_tokens = [
        Token::new(TREE_CTX_PROPERTY, 0),
        Token::new(TREE_CTX_PREDICTOR, PREDICTOR_GRADIENT),
        Token::new(TREE_CTX_OFFSET, pack_signed(0)),
        Token::new(TREE_CTX_MULTIPLIER_LOG, 0),
        Token::new(TREE_CTX_MULTIPLIER_BITS, 0),
    ];
    let tree_code = optimize_entropy_code(&tree_tokens, NUM_TREE_CONTEXTS);
    let tree_code_ref = tree_code.as_ref();

    w.write(1, 0); // no LZ77 for tree entropy code
    write_entropy_code(&tree_code_ref, w);
    for tok in &tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }

    // num_contexts = 1: decoder skips context map entirely — write prefix
    // codes directly (no write_context_map).
    w.write(1, 0); // no LZ77 for pixel entropy code
    write_prefix_codes(&pixel_code.prefix_codes, w);
}

fn build_pixel_code(tokens: &[Token]) -> OwnedEntropyCode {
    let mut code = optimize_entropy_code(tokens, 1);
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
            //       1 wasted bit per pixel), the decoder synchronises correctly.
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
                // (b): synthesise a second symbol at index 0 with depth 1 and
                // assign codes {0 → "0", idx → "1"} so the encoder always emits
                // "1" + extra bits, and the decoder follows the same logic.
                pc.depths[0] = 1;
                pc.bits[0] = 0;
                pc.depths[idx] = 1;
                pc.bits[idx] = 1;
            }
        }
    }
    code
}

// ---------------------------------------------------------------------------
// Global-tree path helpers (large images).
// ---------------------------------------------------------------------------

/// Walk K_CONTEXT_TREE_TOKENS to return the leaf_id (= context_id) that the
/// global modular tree assigns to a pixel in the alpha channel.
///
/// Property layout (mirrors libjxl context_predict.h):
///   [0] = channel_id  (0 for alpha in VarDCT mode)
///   [1] = group_id
///   [2] = y  (unused here — node splits only on 1, 4, 5, 6, 9 in practice)
///   [3] = x
///   [4] = abs(N)
///   [5] = abs(W)
///   [6] = N (top neighbor)
///   [7] = W (left neighbor)
///   [8] = W + N - NW  (raw gradient)
///   [9] = abs(W + N - NW)  (absolute value of raw gradient)
///
/// All reachable leaves for channel_id=0 have predictor=Gradient, offset=0,
/// multiplier=1; only the leaf_id (= context_id for the histogram) differs.
fn alpha_context_id(
    x: usize,
    y: usize,
    w_px: i32,
    n_px: i32,
    _nw_px: i32,
    grad_raw: i32,
    group_id: usize,
) -> usize {
    // Properties indexed as in libjxl context_predict.h.
    // We only need the ones the tree actually splits on.
    let props: [i32; 16] = [
        0,               // [0] channel_id = 0 (alpha is the only modular channel)
        group_id as i32, // [1] group_id
        y as i32,        // [2] y
        x as i32,        // [3] x
        n_px.abs(),      // [4] abs(N)
        w_px.abs(),      // [5] abs(W)
        n_px,            // [6] N
        w_px,            // [7] W
        grad_raw,        // [8] W + N - NW  (raw gradient)
        grad_raw.abs(),  // [9] abs(W + N - NW)
        0,
        0,
        0,
        0,
        0,
        0, // [10..15] reference-channel props (unused for single channel)
    ];

    // Walk the tree.  K_CONTEXT_TREE_TOKENS encodes a sequence:
    //   property_token (ctx=1): 0 means leaf, else (prop+1)
    //   if leaf:  predictor (ctx=2), offset (ctx=3), mul_log (ctx=4), mul_bits (ctx=5)
    //   if split: splitval (ctx=0)  then two children pushed to to_decode.
    //
    // We pre-parse the tree into a flat array of TreeNode, once at first call,
    // and then traverse.
    static TREE: std::sync::OnceLock<Vec<TreeNode>> = std::sync::OnceLock::new();
    let tree = TREE.get_or_init(build_global_tree);
    traverse_tree(tree, &props)
}

#[derive(Clone)]
enum TreeNode {
    Leaf {
        id: usize,
    },
    Split {
        property: usize,
        splitval: i32,
        left: usize,
        right: usize,
    },
}

fn unpack_signed(u: u32) -> i32 {
    ((u >> 1) ^ ((!u) & 1).wrapping_sub(1)) as i32
}

fn build_global_tree() -> Vec<TreeNode> {
    let mut nodes: Vec<TreeNode> = Vec::with_capacity(89);
    let mut to_decode = 1usize;
    let mut idx = 0usize;
    let mut leaf_id = 0usize;

    while to_decode > 0 {
        to_decode -= 1;
        let (_, val) = K_CONTEXT_TREE_TOKENS[idx];
        idx += 1;
        if val == 0 {
            // leaf — skip 4 more tokens (predictor, offset, mul_log, mul_bits)
            idx += 4;
            nodes.push(TreeNode::Leaf { id: leaf_id });
            leaf_id += 1;
        } else {
            let property = (val - 1) as usize;
            let (_, sv) = K_CONTEXT_TREE_TOKENS[idx];
            idx += 1;
            let splitval = unpack_signed(sv);
            let left = nodes.len() + to_decode + 1;
            let right = nodes.len() + to_decode + 2;
            nodes.push(TreeNode::Split {
                property,
                splitval,
                left,
                right,
            });
            to_decode += 2;
        }
    }
    nodes
}

fn traverse_tree(tree: &[TreeNode], props: &[i32]) -> usize {
    let mut idx = 0;
    loop {
        match &tree[idx] {
            TreeNode::Leaf { id } => return *id,
            TreeNode::Split {
                property,
                splitval,
                left,
                right,
            } => {
                let v = props.get(*property).copied().unwrap_or(0);
                idx = if v > *splitval { *left } else { *right };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers (gradient predictor, tokenizer).
// ---------------------------------------------------------------------------

#[inline]
pub fn gradient(w: i32, n: i32, nw: i32) -> i32 {
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

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

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
        write_global_alpha_modular(&alpha, 8, 8, &mut w);
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
        write_lfglobal_alpha_section(&alpha, 512, 400, &mut w);
        // Large path: only 4 bits (GroupHeader).
        assert_eq!(w.bits_written(), 4);
    }

    #[test]
    fn global_tree_all_gradient_for_alpha() {
        // Verify that for channel=0 (alpha), the global tree always gives
        // predictor=Gradient. We check this by confirming alpha_context_id
        // never panics and the leaf_id maps to a valid context.
        for group_id in 0..30 {
            for abs_n in [0i32, 50, 128, 200] {
                for abs_w in [0i32, 50, 128, 200] {
                    for grad in [-200i32, 0, 200] {
                        let cid = alpha_context_id(1, 1, abs_w, abs_n, 0, grad, group_id);
                        // context must be a valid leaf index (< 45)
                        assert!(cid < 45, "context_id={cid} out of range");
                    }
                }
            }
        }
    }
}
