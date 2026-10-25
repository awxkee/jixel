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

//! AC strategy, quant-field, correlation and filter metadata tokens.

use super::dc_tokens::clamped_gradient;
use super::quantization::epf_sharpness_id;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::entropy::{Token, optimize_entropy_code, pack_signed};
use crate::static_entropy_codes::K_NUM_DC_CONTEXTS;

/// AC metadata: ytox/ytob CfL maps (all 0), AC strategy (all 0 = DCT-8x8),
/// quant field residuals (all 0), and EPF (token (0, PackSigned(4)) per block).
///
/// In libjxl-tiny ALL four sub-streams use the same shared dc_code.
/// Same as write_ac_metadata_tokens, but returns the tokens. Mirror of
/// collect_dc_tokens for the AC metadata (YtoX/B, ACS, QF, EPF).
#[inline]
pub(crate) fn ac_metadata_context(left: i32, base: u32) -> u32 {
    base + if left > 11 {
        0
    } else if left > 5 {
        1
    } else if left > 3 {
        2
    } else {
        3
    }
}

pub(crate) fn collect_ac_metadata_tokens(
    dc_data: &DcGroupData,
    props: &mut Vec<crate::dc_tree::DcProp>,
    epf_sharpness: i32,
    collect_props: bool,
) -> Vec<Token> {
    #[inline]
    fn wbin(w: i32) -> crate::dc_tree::DcProp {
        (512 + w).clamp(0, 1023) as crate::dc_tree::DcProp
    }
    let xsize_blocks = dc_data.ac_strategy.xsize();
    let ysize_blocks = dc_data.ac_strategy.ysize();
    let xtiles = dc_data.ytox_map.xsize();
    let ytiles = dc_data.ytox_map.ysize();
    let nblocks = xsize_blocks * ysize_blocks;
    let num_first_blocks = dc_data.ac_strategy.count_first_blocks();
    let cfl_tokens = dc_data.ytox_map.as_slice().len() + dc_data.ytob_map.as_slice().len();
    let mut tokens = Vec::with_capacity(cfl_tokens + 2 * num_first_blocks + nblocks);

    // (a) YtoX and YtoB tokens with gradient prediction.
    for (c, cfl_map) in [&dc_data.ytox_map, &dc_data.ytob_map]
        .into_iter()
        .enumerate()
    {
        debug_assert_eq!((cfl_map.xsize(), cfl_map.ysize()), (xtiles, ytiles));
        if xtiles == 0 || ytiles == 0 {
            continue;
        }

        let ctx_id = 2u32 - c as u32;
        let mut rows = cfl_map.as_slice().chunks_exact(xtiles);

        // The top row predicts from the value on the left.
        let first_row = rows.next().unwrap();
        let mut left = 0i32;
        for &here in first_row {
            let here = here as i32;
            if collect_props {
                props.push(wbin(left));
            }
            tokens.push(Token::new(ctx_id, pack_signed(here - left)));
            left = here;
        }

        let mut row_above = first_row;
        for row in rows {
            // First column replicates the value above for W and NW.
            if collect_props {
                props.push(wbin(row_above[0] as i32));
            }
            tokens.push(Token::new(
                ctx_id,
                pack_signed(row[0] as i32 - row_above[0] as i32),
            ));

            for (current, above) in row.array_windows::<2>().zip(row_above.array_windows::<2>()) {
                let prediction =
                    clamped_gradient(above[1] as i32, current[0] as i32, above[0] as i32);
                if collect_props {
                    props.push(wbin(current[0] as i32));
                }
                tokens.push(Token::new(
                    ctx_id,
                    pack_signed(current[1] as i32 - prediction),
                ));
            }
            row_above = row;
        }
    }

    // (b) AC strategy and (c) QF residual tokens. Their substreams are
    // contiguous, so reserve both ranges and fill them in one block scan.
    let strategy_base = tokens.len();
    let qf_base = strategy_base + num_first_blocks;
    tokens.resize(strategy_base + 2 * num_first_blocks, Token::new(0, 0));
    if collect_props {
        props.resize(strategy_base + 2 * num_first_blocks, 0);
    }
    let mut strategy_left = 0i32;
    let mut qf_left = if nblocks == 0 {
        0
    } else {
        dc_data.ac_strategy.strategy_code(0, 0) as i32
    };
    let mut first_idx = 0usize;
    for (y, row_qf) in (0..ysize_blocks)
        .map(|y| dc_data.raw_quant_field.row(y))
        .enumerate()
    {
        for (x, &qf) in row_qf[..xsize_blocks].iter().enumerate() {
            if !dc_data.ac_strategy.is_first_block(x, y) {
                continue;
            }

            let strategy = dc_data.ac_strategy.strategy_code(x, y) as i32;
            if collect_props {
                props[strategy_base + first_idx] = wbin(strategy_left);
            }
            tokens[strategy_base + first_idx] =
                Token::new(ac_metadata_context(strategy_left, 7), pack_signed(strategy));
            strategy_left = strategy;

            let qf = qf as i32 - 1;
            if collect_props {
                props[qf_base + first_idx] = wbin(qf_left);
            }
            tokens[qf_base + first_idx] =
                Token::new(ac_metadata_context(qf_left, 3), pack_signed(qf - qf_left));
            qf_left = qf;
            first_idx += 1;
        }
    }
    debug_assert_eq!(first_idx, num_first_blocks);

    // (d) EPF tokens (constant stream; refinement can never split it).
    if collect_props {
        props.resize(props.len() + nblocks, wbin(epf_sharpness));
    }
    tokens.resize(
        tokens.len() + nblocks,
        Token::new(0, pack_signed(epf_sharpness)),
    );
    tokens
}

/// Real prefix-code bit cost of a DC group's AC-metadata token stream under a
/// freshly optimized entropy code. Used by the sub-8x8 activation gate to weigh
/// the exact selected set's meta-stream cost against its RD benefit.
pub(super) fn meta_entropy_cost(
    dc_data: &DcGroupData,
    scratch: &mut CoderScratch,
    distance: f32,
) -> u64 {
    let toks = collect_ac_metadata_tokens(
        dc_data,
        &mut Vec::new(),
        epf_sharpness_id(distance, false),
        false,
    );
    let code_owned = optimize_entropy_code(&toks, K_NUM_DC_CONTEXTS, &mut scratch.huffman_pool);
    let code = code_owned.as_ref();
    let mut bits = 0u64;
    for t in &toks {
        let (tok, nbits, _b) = crate::entropy::uint_encode(t.value);
        let pc = &code.prefix_codes[code.context_map[t.context as usize] as usize];
        bits += if pc.single_symbol {
            nbits as u64
        } else {
            pc.depths[tok as usize] as u64 + nbits as u64
        };
    }
    bits
}
