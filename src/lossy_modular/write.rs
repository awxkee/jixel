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

//! A multi-group frame's header, sections and TOC.
//!
//! Section 0 (LfGlobal) carries the dc_quant field, the global tree with the
//! shared entropy code, the transform list and the global stream. DC-group
//! and AC-group sections carry their streams' tokens; the AC global section
//! is empty.

use super::pyramid::Pyramid;
use super::tokens::Streams;
use crate::bit_writer::BitWriter;
use crate::entropy::{EntropyCode, HuffmanNode, Token};
use crate::lossless::kit::{write_lz_groups, write_lz_section, write_toc_entry, write_tree_lz77};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

#[allow(clippy::too_many_arguments)]
pub(super) fn write_frame(
    p: &Pyramid,
    tree_tokens: &[Token],
    code: &EntropyCode<'_>,
    min_symbol: u32,
    distance_ctx: u32,
    streams: &Streams,
    dictionary: Option<BitWriter>,
    pool: &ThreadPool,
    huffman_pool: &mut Vec<HuffmanNode>,
    writer: &mut BitWriter,
) {
    let num_dc_groups = streams.dc_groups.len();
    let num_ac_groups = streams.ac_groups.len();
    p.write_frame_header(writer);

    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // ----- Section 0: LfGlobal = patches, global tree + global modular image -----
    if let Some(dictionary) = dictionary {
        sections[0] = dictionary;
    }
    p.layout.write_dc_quant(&mut sections[0]);
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(
        tree_tokens,
        code,
        min_symbol,
        huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree = 1
    sections[0].write(1, 1); // wp_default = 1
    p.layout.write_transforms(&p.steps, &mut sections[0]);
    write_lz_section(
        &streams.global,
        distance_ctx,
        code,
        min_symbol,
        &mut sections[0],
    );
    sections[0].zero_pad_to_byte();

    // ----- DC groups -----
    write_lz_groups(
        &streams.dc_groups,
        &[],
        code,
        distance_ctx,
        min_symbol,
        WpParams::DEFAULT,
        pool,
        &mut sections[1..num_dc_groups + 1],
    );

    // ----- AC global: trivial -----
    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    // ----- AC groups -----
    write_lz_groups(
        &streams.ac_groups,
        &[],
        code,
        distance_ctx,
        min_symbol,
        WpParams::DEFAULT,
        pool,
        &mut sections[2 + num_dc_groups..],
    );

    if num_ac_groups == 1 {
        // One TOC entry: every channel lives in the LfGlobal stream and the
        // group streams are empty, so the decoder reads nothing past it.
        debug_assert!(
            streams
                .dc_groups
                .iter()
                .chain(&streams.ac_groups)
                .all(Vec::is_empty)
        );
        sections.truncate(1);
    }
    write_toc(sections, writer);
}

/// The TOC (no permutation) and the byte-aligned sections.
pub(super) fn write_toc(mut sections: Vec<BitWriter>, writer: &mut BitWriter) {
    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
}
