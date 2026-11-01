/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! A frame that fits one group: one section carrying everything.
//!
//! Each channel gets one predictor (searched when the speed allows it,
//! scale-safe for quantized channels), a one-leaf-per-channel tree carrying
//! its step, and LZ77 over the whole token stream.

use super::pyramid::Pyramid;
use super::write::write_toc;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;
use crate::lossless::kit::{
    PREDICTOR_GRADIENT, SqueezePredictorCost, build_balanced_tree_tokens_mul, build_lz_pixel_code,
    channel_to_context, choose_predictor_for_plane, fixed_predictor, lz77_compress_for_speed,
    selected_grad_pack_interior_fn, tokenize_plane_rows, write_lz_section, write_tree_lz77,
};
use crate::thread_pool::ThreadPool;

pub(crate) fn write_frame(
    p: &Pyramid,
    min_symbol: u32,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    let channels = &p.channels;
    let quants = &p.quants;
    let nb = channels.len();
    let use_wp = true;
    let predictors: Vec<u32> = if speed.effort().predictor_search {
        pool.steal_map(scratch, nb, |c, _scratch| {
            let ch = &channels[c];
            let data = &ch.data;
            let w = ch.w;
            let get = move |gx: usize, gy: usize| data[gy * w + gx];
            if quants[c] > 1 {
                let mut costs = SqueezePredictorCost::with_zero();
                costs.add_crop(get, ch.w, ch.h, use_wp);
                costs.safe_predictor()
            } else {
                choose_predictor_for_plane(get, ch.w, ch.h, use_wp)
            }
        })
    } else {
        (0..nb)
            .map(|i| {
                if quants[i] > 1 {
                    PREDICTOR_GRADIENT
                } else {
                    fixed_predictor(use_wp)
                }
            })
            .collect()
    };
    let grad_pack_fn = selected_grad_pack_interior_fn();
    let channel_tokens = pool.steal_map(scratch, nb, |c, scratch| {
        let ch = &channels[c];
        let mut tokens = Vec::with_capacity(ch.w * ch.h);
        tokenize_plane_rows(
            channel_to_context(c, nb),
            |y| &ch.data[y * ch.w..][..ch.w],
            ch.w,
            ch.h,
            predictors[c],
            grad_pack_fn,
            &mut scratch.gradient,
            &mut tokens,
        );
        tokens
    });
    let mut tokens: Vec<Token> = Vec::new();
    for channel in channel_tokens {
        tokens.extend(channel);
    }

    p.write_frame_header(writer);
    let mut section = BitWriter::new();
    p.layout.write_dc_quant(&mut section);
    section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    p.layout.write_transforms(&p.steps, &mut section);

    let distance_ctx = nb as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy(),
        std::iter::once(lz_tokens.as_slice()),
        nb,
        min_symbol,
        speed.effort().lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let leaves: Vec<(u32, u32)> = predictors
        .iter()
        .copied()
        .zip(quants.iter().copied())
        .collect();
    write_tree_lz77(
        &build_balanced_tree_tokens_mul(&leaves),
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();
    write_toc(vec![section], writer);
}
