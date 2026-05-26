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

// ---------------------------------------------------------------------------
// Tree contexts (match libjxl's ma_common.h enum values).
// ---------------------------------------------------------------------------

use crate::Image3F;
use crate::bit_writer::BitWriter;
use crate::color::linear_to_srgb_u_n;
use crate::enc_frame::div_ceil;
use crate::encode_image::{AlphaPlane, BitsPerSample};
use crate::entropy::{
    OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code,
    write_prefix_codes, write_token,
};
use crate::modular::gradient;

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

pub(crate) fn encode_frame_lossless(
    linear: &Image3F,
    alpha: Option<&AlphaPlane>,
    bps: BitsPerSample,
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
        let tokens = tokenize_all(linear, alpha, xsize, ysize, 0, 0, xsize, ysize, bps);

        // Per-channel prefix codes; balanced N-leaf tree.
        let code = build_pixel_code(&tokens, nb_chans);
        write_local_tree(nb_chans, &code, &mut section);

        // Write the pixel tokens.
        let code_ref = code.as_ref();
        for tok in &tokens {
            write_token(*tok, &code_ref, &mut section);
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

        // Tokenize ONCE per AC group (sub-image-local neighbours, matching what
        // we'll emit below) and pool to build the global prefix code so that
        // every per-group emission is guaranteed to be representable.
        let mut all_tokens: Vec<Token> = Vec::new();
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);
                let toks = tokenize_all(linear, alpha, xsize, ysize, x0, y0, gw, gh, bps);
                all_tokens.extend_from_slice(&toks);
            }
        }
        let code = build_pixel_code(&all_tokens, nb_chans);

        // ----- Section 0: DC global -----
        sections[0].write(1, 1); // dc_quant all_default = 1
        sections[0].write(1, 1); // has_tree = 1
        write_local_tree(nb_chans, &code, &mut sections[0]);
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
        let code_ref = code.as_ref();
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let group_index = gy * xsize_groups + gx;
                let section_idx = 2 + num_dc_groups + group_index;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);

                // GroupHeader: use_global_tree=1, wp=1, 0 transforms (the global
                // header already declared the RCT for the whole image).
                sections[section_idx].write(1, 1);
                sections[section_idx].write(1, 1);
                sections[section_idx].write(2, 0);

                let group_tokens = tokenize_all(linear, alpha, xsize, ysize, x0, y0, gw, gh, bps);
                for tok in &group_tokens {
                    write_token(*tok, &code_ref, &mut sections[section_idx]);
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

#[inline]
fn forward_ycocg(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
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

fn tokenize_all(
    linear: &Image3F,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    bps: BitsPerSample,
) -> Vec<Token> {
    let bits = bps.bits() as u8;
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };
    let mut out = Vec::with_capacity(gw * gh * nb_chans);

    // 1. Build the post-YCoCg Y, Co, Cg planes for this sub-rect.
    let mut ys: Vec<Vec<i32>> = Vec::with_capacity(gh);
    let mut cos: Vec<Vec<i32>> = Vec::with_capacity(gh);
    let mut cgs: Vec<Vec<i32>> = Vec::with_capacity(gh);
    for gy in 0..gh {
        let rr = linear.plane_row(0, y0 + gy);
        let gg = linear.plane_row(1, y0 + gy);
        let bb = linear.plane_row(2, y0 + gy);
        let mut yr = Vec::with_capacity(gw);
        let mut cor = Vec::with_capacity(gw);
        let mut cgr = Vec::with_capacity(gw);
        for gx in 0..gw {
            let r = linear_to_srgb_u_n(rr[x0 + gx], bits) as i32;
            let g = linear_to_srgb_u_n(gg[x0 + gx], bits) as i32;
            let b = linear_to_srgb_u_n(bb[x0 + gx], bits) as i32;
            let (y, co, cg) = forward_ycocg(r, g, b);
            yr.push(y);
            cor.push(co);
            cgr.push(cg);
        }
        ys.push(yr);
        cos.push(cor);
        cgs.push(cgr);
    }

    for chan in 0..3 {
        let plane: &Vec<Vec<i32>> = match chan {
            0 => &ys,
            1 => &cos,
            _ => &cgs,
        };
        let ctx = channel_to_context(chan, nb_chans);
        for gy in 0..gh {
            let row = &plane[gy];
            let prev_row = if gy > 0 { Some(&plane[gy - 1]) } else { None };
            for gx in 0..gw {
                let v = row[gx];
                let w_ = if gx > 0 { row[gx - 1] } else { 0 };
                let n_ = prev_row.map_or(0, |r| r[gx]);
                let nw_ = if gx > 0 {
                    prev_row.map_or(0, |r| r[gx - 1])
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

fn build_pixel_code(tokens: &[Token], num_contexts: usize) -> OwnedEntropyCode {
    let mut code = optimize_entropy_code(tokens, num_contexts);
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
            push_split(&mut t, 0, 1);
            push_leaf(&mut t); // chan 2
            push_split(&mut t, 0, 0);
            push_leaf(&mut t); // chan 1
            push_leaf(&mut t); // chan 0
        }
        4 => {
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

fn write_local_tree(n_leaves: usize, pixel_code: &OwnedEntropyCode, w: &mut BitWriter) {
    let tree_tokens = build_balanced_tree_tokens(n_leaves);
    let tree_code = optimize_entropy_code(&tree_tokens, NUM_TREE_CONTEXTS);
    let tree_code_ref = tree_code.as_ref();

    w.write(1, 0); // no LZ77 for tree
    write_entropy_code(&tree_code_ref, w);
    for tok in &tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }

    w.write(1, 0); // no LZ77 for pixel entropy code
    if n_leaves == 1 {
        write_prefix_codes(&pixel_code.prefix_codes, w);
    } else {
        write_entropy_code(&pixel_code.as_ref(), w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::srgb_to_linear_u8;
    use crate::encode_image::EncodeConfig;
    use crate::encode_with_config;

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
    fn lossless_single_group_emits_modular_frame() {
        let img = make_image(4, 4, |x, y| {
            [
                ((y * 4 + x) * 17 % 256) as u8,
                ((y * 4 + x) * 31 % 256) as u8,
                ((y * 4 + x) * 47 % 256) as u8,
            ]
        });
        let bytes = encode_with_config(&img, &EncodeConfig::default().with_lossless(true));
        assert_eq!(&bytes[..2], &[0xFF, 0x0A]);
        assert!(
            bytes.len() < 4 * 4 * 3 * 8,
            "lossless 4x4 should be reasonably small"
        );
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
    fn linear_to_srgb_u_n_round_trips() {
        use crate::color::{linear_to_srgb_u_n, srgb_to_linear_u16};
        for &bits in &[8u8, 10, 12] {
            let max = ((1u32 << bits) - 1) as f32;
            for v in 0..=((1u32 << bits) - 1) {
                let lin = srgb_to_linear_u16(v as u16, max);
                let back = linear_to_srgb_u_n(lin, bits) as u32;
                assert_eq!(v, back, "bits={bits} v={v} → linear → {back}");
            }
        }
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
