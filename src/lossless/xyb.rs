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

//! Modular XYB atlases and layered lossless frames.

use super::LosslessSettings;
use super::encoder::encode_frame_lossless_core;
use super::header::{write_frame_header_modular_xyb_reference, write_toc_entry};
use super::lz77::{
    LZ77_MIN_SYMBOL, build_lz_pixel_code, lz77_compress_for_speed, write_local_tree_lz77,
    write_lz_section,
};
use super::palette::write_palette_transform;
use super::predictor::{
    channel_to_context, choose_predictor_for_rows, fixed_predictor, selected_grad_pack_interior_fn,
    tokenize_plane, tokenize_plane_rows,
};
use super::rct::forward_ycocg;
use super::{GROUP_DIM, GroupLayout};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::Token;
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::xyb::quantize_xyb_channels;

/// Reference-only atlas frame for the lossy codestream, coded with the modular
/// machinery on the fixed XYB integer lattice: near-lossless, ringing-free, and
/// palette-compressed when the quantized atlas holds few distinct colors.
///
/// Returns false when the atlas cannot take this path (it must fit a single
/// 256-pixel modular group); the caller then keeps the VarDCT atlas. Because
/// this encoding is sharper than any lossy VarDCT atlas, selecting the smaller
/// of the two by bits alone cannot regress quality.
pub(crate) fn encode_modular_xyb_atlas(
    atlas: &crate::image::Image3F,
    has_alpha: bool,
    lattice_scale: u32,
    speed: crate::Speed,
    use_wp: bool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    // The 13-bit residual bound behind LZ77_MIN_SYMBOL caps the refinement.
    debug_assert!(lattice_scale.is_power_of_two() && lattice_scale <= 8);
    let (xsize, ysize) = (atlas.xsize(), atlas.ysize());
    if xsize == 0 || ysize == 0 || xsize > GROUP_DIM || ysize > GROUP_DIM {
        return false;
    }
    let ch = quantize_xyb_channels(atlas, lattice_scale);
    encode_modular_xyb_atlas_ints(
        ch,
        xsize,
        ysize,
        has_alpha,
        lattice_scale,
        crate::patches::MODULAR_PATCH_REF_ID,
        speed,
        use_wp,
        scratch,
        writer,
    )
}

/// Same as [`encode_modular_xyb_atlas`] for an atlas that is already on the
/// integer lattice (channels Y, X, B-Y), saved to reference `slot`. Accepts
/// up to one 1024-pixel group.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_modular_xyb_atlas_ints(
    ch: [Vec<i32>; 3],
    xsize: usize,
    ysize: usize,
    has_alpha: bool,
    lattice_scale: u32,
    slot: u32,
    speed: crate::Speed,
    use_wp: bool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    // Keep atlas costs used by lossy RDO at the standard budget.
    let speed = if speed == crate::Speed::ExtraSlow {
        crate::Speed::Slow
    } else {
        speed
    };
    use std::collections::HashMap;
    let large = GroupLayout::LARGE.dim();
    if xsize == 0 || ysize == 0 || xsize > large || ysize > large {
        return false;
    }
    let layout = if xsize > GROUP_DIM || ysize > GROUP_DIM {
        GroupLayout::LARGE
    } else {
        GroupLayout::DEFAULT
    };
    let grad_pack_fn = selected_grad_pack_interior_fn();
    let effort = speed.effort();
    let npx = xsize * ysize;

    // Distinct quantized XYB triples, palette-capped like the lossless path.
    let mut seen: HashMap<[i32; 3], ()> = HashMap::with_capacity(257);
    let [y, x, b] = &ch;
    let pixel_colors = || y[..npx].iter().zip(&x[..npx]).zip(&b[..npx]);
    for ((&y, &x), &b) in pixel_colors() {
        seen.entry([y, x, b]).or_insert(());
        if seen.len() > 256 {
            break;
        }
    }
    let use_palette = !seen.is_empty() && seen.len() <= 256;

    write_frame_header_modular_xyb_reference(xsize, ysize, slot, has_alpha, layout, writer);

    let mut section = BitWriter::new();
    // LfChannelDequant: the decoder multiplies each channel by these steps.
    // Custom F16 steps (each read as value/128) refine the lattice by
    // `lattice_scale`; powers of two keep every step F16-exact.
    if lattice_scale == 1 {
        section.write(1, 1); // dc_quant all_default = 1
    } else {
        use crate::quant_weights::INV_DC_QUANT;
        use crate::util::f32_to_f16_bits;
        section.write(1, 0);
        for c in 0..3 {
            let step_x128 = 128.0 / (INV_DC_QUANT[c] * lattice_scale as f32);
            section.write(16, f32_to_f16_bits(step_x128) as u64);
        }
    }
    section.write(1, 0); // has_tree = 0
    section.write(1, 0); // GroupHeader: use_global_tree = 0
    section.write(1, 1); // wp_default = 1

    // The atlas alpha is always the all-zero plane `zero_alpha_for_lossy`
    // produces (patch entries blend alpha with None, so its content is never
    // read); the palette absorbs it as a constant fourth component and the
    // plain path codes it as one all-zero channel.
    let num_c = 3 + usize::from(has_alpha);
    let mut tokens: Vec<Token> = Vec::new();
    let (preds, nb_chans): (Vec<u32>, usize) = if use_palette {
        let mut colors: Vec<[i32; 3]> = seen.keys().copied().collect();
        colors.sort_unstable();
        let nb_colors = colors.len();
        let mut idx_of: HashMap<[i32; 3], u32> = HashMap::with_capacity(nb_colors);
        for (i, c) in colors.iter().enumerate() {
            idx_of.insert(*c, i as u32);
        }
        // Component rows of the palette meta-channel; the alpha row (if any)
        // stays zero.
        let mut palette_ch = vec![0i32; num_c * nb_colors];
        for (c, row) in palette_ch[..3 * nb_colors]
            .chunks_exact_mut(nb_colors)
            .enumerate()
        {
            for (value, color) in row.iter_mut().zip(&colors) {
                *value = color[c];
            }
        }
        let index_img: Vec<i32> = pixel_colors()
            .map(|((&y, &x), &b)| idx_of[&[y, x, b]] as i32)
            .collect();

        write_palette_transform(num_c as u32, nb_colors as u32, &mut section);
        let preds = if effort.predictor_search {
            vec![
                choose_predictor_for_rows(
                    |y| &palette_ch[y * nb_colors..][..nb_colors],
                    nb_colors,
                    num_c,
                    use_wp,
                ),
                choose_predictor_for_rows(
                    |y| &index_img[y * xsize..][..xsize],
                    xsize,
                    ysize,
                    use_wp,
                ),
            ]
        } else {
            vec![fixed_predictor(use_wp); 2]
        };
        tokenize_plane_rows(
            channel_to_context(0, 2),
            |y| &palette_ch[y * nb_colors..][..nb_colors],
            nb_colors,
            num_c,
            preds[0],
            grad_pack_fn,
            &mut scratch.gradient,
            &mut tokens,
        );
        tokenize_plane_rows(
            channel_to_context(1, 2),
            |y| &index_img[y * xsize..][..xsize],
            xsize,
            ysize,
            preds[1],
            grad_pack_fn,
            &mut scratch.gradient,
            &mut tokens,
        );
        (preds, 2)
    } else {
        section.write(2, 0b00); // 0 transforms: Y/X/B-Y are already decorrelated
        let mut preds = Vec::with_capacity(num_c);
        for (c, data) in ch.iter().enumerate() {
            let pred = if effort.predictor_search {
                choose_predictor_for_rows(|y| &data[y * xsize..][..xsize], xsize, ysize, use_wp)
            } else {
                fixed_predictor(use_wp)
            };
            tokenize_plane_rows(
                channel_to_context(c, num_c),
                |y| &data[y * xsize..][..xsize],
                xsize,
                ysize,
                pred,
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
            preds.push(pred);
        }
        if has_alpha {
            // Constant zero plane: every predictor is exact, tokens are free.
            tokenize_plane(
                channel_to_context(3, num_c),
                |_, _| 0,
                xsize,
                ysize,
                fixed_predictor(use_wp),
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
            preds.push(fixed_predictor(use_wp));
        }
        (preds, num_c)
    };

    // Same guard as the lossless frame path: literals whose hybrid-uint token
    // would reach the LZ77 symbol range must push that range up, or the decoder
    // reads a large residual as a back-reference. The refined lattice makes the
    // channel values up to 13 bits, and residuals can double that.
    let max_abs = ch
        .iter()
        .flat_map(|plane| plane.iter())
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    let value_bits = 33 - (2 * max_abs).max(1).leading_zeros();
    let min_symbol = if value_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * value_bits + 24
    };

    let distance_ctx = nb_chans as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy(),
        std::iter::once(lz_tokens.as_slice()),
        nb_chans,
        min_symbol,
        effort.lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_local_tree_lz77(
        &preds,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();

    writer.write(1, 0); // TOC: no permutation
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(std::slice::from_mut(&mut section));
    writer.zero_pad_to_byte();
    true
}

/// Ink frame of the lossy codestream: XYB lattice ints (Y, X, B-Y, run
/// through YCoCg like RGB so the signaled RCT restores them exactly) and the
/// coverage alpha, blended over reference slot 0 with the learned-tree coder.
pub(crate) fn encode_frame_lossless_xyb_ink(
    linear: &Image3Si,
    alpha: &AlphaPlane,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    encode_frame_lossless_core(
        linear,
        Some(alpha),
        LosslessSettings {
            max_bits: 13,
            ..LosslessSettings::rgb8(speed, decoding_speed)
        },
        ModularFrameKind::XybBlend { last: true },
        true,
        pool,
        scratch,
        writer,
    )
}

/// One lossless 8-bit RGB(+alpha) frame of a layered codestream.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame_lossless_layer(
    linear: &Image3Si,
    alpha: &AlphaPlane,
    kind: ModularFrameKind<'_>,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    encode_frame_lossless_core(
        linear,
        Some(alpha),
        LosslessSettings::rgb8(speed, decoding_speed),
        kind,
        true,
        pool,
        scratch,
        writer,
    );
}

/// Glyph atlas on the default XYB lattice, coded by the full lossless
/// machinery (learned tree, palettes, RCT search).
pub(crate) fn encode_modular_xyb_atlas_tree(
    ch: &[Vec<i32>; 3],
    xsize: usize,
    ysize: usize,
    alpha: Option<&AlphaPlane>,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    encode_modular_xyb_atlas_tree_slot(
        ch,
        xsize,
        ysize,
        alpha,
        speed,
        pool,
        scratch,
        crate::patches::MODULAR_PATCH_REF_ID,
        writer,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_modular_xyb_atlas_tree_slot(
    ch: &[Vec<i32>; 3],
    xsize: usize,
    ysize: usize,
    alpha: Option<&AlphaPlane>,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    slot: u32,
    writer: &mut BitWriter,
) -> bool {
    // Keep atlas costs used by lossy RDO at the standard budget. Below Medium
    // the atlas has neither a learned tree nor a palette and outweighs the
    // frame it replaces, while the atlas itself is small.
    let speed = speed.clamp(crate::Speed::Medium, crate::Speed::Slow);
    let Ok(mut atlas) = Image3Si::try_new(xsize, ysize) else {
        return false;
    };
    // The core takes YCoCg planes (the lossless front end applies RCT 6) and
    // searches the other transforms from there.
    for y in 0..ysize {
        for x in 0..xsize {
            let i = y * xsize + x;
            let (a, b, c) = forward_ycocg(ch[0][i], ch[1][i], ch[2][i]);
            atlas.plane_row_mut(0, y)[x] = a;
            atlas.plane_row_mut(1, y)[x] = b;
            atlas.plane_row_mut(2, y)[x] = c;
        }
    }
    encode_frame_lossless_core(
        &atlas,
        alpha,
        LosslessSettings {
            max_bits: 14,
            ..LosslessSettings::rgb8(speed, crate::DecodingSpeed::Slow)
        },
        ModularFrameKind::XybReferenceOnly {
            width: xsize,
            height: ysize,
            slot,
        },
        false,
        pool,
        scratch,
        writer,
    );
    true
}
