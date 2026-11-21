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

//! Tests for lz77.

use super::entropy::{build_lz_pixel_code_tails, lz77_literals};
use super::write::{write_lz_section, write_lz_section_tail};
use super::{ConstantTail, LzToken, lz77_compress_runs_channels};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::Token;

/// A pseudo-random literal stream over contexts `0..contexts`.
fn stream(seed: u32, len: usize, contexts: u32) -> Vec<Token> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            Token::new(state % contexts, (state >> 8) % 40)
        })
        .collect()
}

/// (thorough, refined, ans_cluster): ExtraSlow/Slow learned and flat codes,
/// plus Fast's prefix code. All must retain the decoder's constant fill.
const CODE_KINDS: [(bool, bool, bool); 5] = [
    (true, true, true),
    (true, true, false),
    (false, true, true),
    (false, true, false),
    (false, false, false),
];

#[test]
fn tail_context_is_pinned_to_a_single_symbol_cluster() {
    for (thorough, refined, ans_cluster) in CODE_KINDS {
        let tokens = stream(7, 4096, 3);
        let tail = ConstantTail {
            context: 3,
            count: 100_000,
        };
        let mut scratch = CoderScratch::default();
        let code = build_lz_pixel_code_tails(
            thorough,
            &[tokens.as_slice()],
            &[tail],
            4,
            64,
            refined,
            ans_cluster,
            None,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        let cluster = code.context_map[3] as usize;
        assert!(
            code.context_map
                .iter()
                .enumerate()
                .all(|(context, &c)| context == 3 || c as usize != cluster),
            "the tail shares no cluster ({refined}, {ans_cluster})"
        );
        let (symbol, nbits, _) =
            crate::entropy::uint_encode_with_config(0, code.hybrid_uint_configs[cluster]);
        assert_eq!(nbits, 0);
        if code.use_prefix_code {
            assert_eq!(code.prefix_codes[cluster].depths[symbol as usize], 0);
        } else {
            assert_eq!(
                code.ans_symbols[cluster][symbol as usize].freq as u32,
                crate::entropy::ANS_TAB_SIZE
            );
        }
    }
}

#[test]
fn tail_writes_the_same_bits_as_materialized_zeros() {
    for (thorough, refined, ans_cluster) in CODE_KINDS {
        for with_runs in [false, true] {
            let literals = stream(11, 3000, 3);
            let tail = ConstantTail {
                context: 3,
                count: 5000,
            };
            let tokens: Vec<LzToken> = if with_runs {
                lz77_compress_runs_channels(vec![literals.clone()])
            } else {
                lz77_literals(&literals)
            };
            let mut scratch = CoderScratch::default();
            let code = build_lz_pixel_code_tails(
                thorough,
                &[tokens.as_slice()],
                &[tail],
                4,
                64,
                refined,
                ans_cluster,
                None,
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            let mut with_tail = BitWriter::new();
            write_lz_section_tail(&tokens, tail, 4, &code, 64, &mut with_tail);
            let mut materialized = tokens.clone();
            materialized.extend((0..tail.count).map(|_| LzToken::pixel(3, 0)));
            let mut plain = BitWriter::new();
            write_lz_section(&materialized, 4, &code, 64, &mut plain);
            assert_eq!(with_tail.bits_written(), plain.bits_written());
            with_tail.zero_pad_to_byte();
            plain.zero_pad_to_byte();
            assert_eq!(with_tail.into_bytes(), plain.into_bytes());
        }
    }
}
