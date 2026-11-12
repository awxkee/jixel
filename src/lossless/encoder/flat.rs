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

//! Step 8 of a lossless frame: the flat coder. One fixed predictor per
//! channel, per-channel contexts (WP-bucketed for displayed frames), LZ77
//! over the residual tokens and one prefix code for the frame.

use super::CoreFrame;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::image::Image3Si;
use crate::lossless::context::{WpContextTree, wp_context_tree, write_flat_tree};
use crate::lossless::groups::{DeepLzScratchPool, group_lz_threads, write_lz_groups};
use crate::lossless::header::{
    write_frame_header_modular_kind, write_modular_transforms, write_patch_dictionary,
    write_toc_entry,
};
use crate::lossless::lz77::{
    ConstantTail, LzToken, build_lz_pixel_code_tails, lz77_compress_channels_for_speed,
    lz77_compress_channels_for_speed_with_depth, write_lz_section, write_lz_section_tail,
};
use crate::lossless::predictor::{
    PREDICTOR_WEIGHTED, WpChannelContexts, channel_to_context, tokenize_channels_with_wp,
    tokenize_runs_with_wp,
};
use crate::lossless::{GroupGrid, GroupLayout};
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpParams, write_wp_header};

/// The flat coder's model of one frame, ready to write.
pub(super) struct FlatCoder<'f, 'a> {
    frame: &'f CoreFrame<'a>,
    /// The color planes under `rct_type`.
    planes: &'f Image3Si,
    rct_type: u32,
    layout: GroupLayout,
    grid: GroupGrid,
    /// One predictor per modular channel: the color planes (Y for gray;
    /// Y/Co/Cg for color), then alpha.
    predictors: Vec<u32>,
    /// Per-channel leaf offsets: a solid alpha plane's value.
    offsets: Vec<i32>,
    /// The fixed WP context tree (`flat_wp_contexts`), for displayed frames
    /// with a Weighted channel; else one context per channel.
    wp_tree: Option<WpContextTree>,
    /// The search's preset, or the default when no channel predicts with
    /// the Weighted Predictor.
    wp_params: WpParams,
}

impl<'f, 'a> FlatCoder<'f, 'a> {
    /// `predictors` holds the search's choice per color plane and, at index
    /// 3, for alpha.
    pub(super) fn new(
        frame: &'f CoreFrame<'a>,
        planes: &'f Image3Si,
        rct_type: u32,
        layout: GroupLayout,
        predictors: &[u32; 4],
        wp_params: WpParams,
    ) -> Self {
        let nb_chans = frame.nb_chans;
        let mut chan_preds: Vec<u32> = (0..frame.num_color).map(|c| predictors[c]).collect();
        if frame.alpha.is_some() {
            chan_preds.push(predictors[3]);
        }
        // A solid alpha plane: Zero predictor with the value as the leaf offset
        // (the predictor search's tie-break would pick the Weighted Predictor,
        // which the decoder then runs over the whole plane) and no tokens; its
        // channel is a constant tail of every stream (see `ConstantTail`).
        let mut offsets = vec![0i32; nb_chans];
        if let Some(value) = frame.alpha_constant {
            chan_preds[nb_chans - 1] = 0;
            offsets[nb_chans - 1] = value;
        }
        // A reference-only atlas keeps one context per channel.
        let displayed = matches!(
            frame.kind,
            ModularFrameKind::Regular
                | ModularFrameKind::Patched { .. }
                | ModularFrameKind::XybBlend { .. }
                | ModularFrameKind::RgbSavedFirst
                | ModularFrameKind::RgbBlendLast
                | ModularFrameKind::RgbCrop { .. }
                | ModularFrameKind::XybCrop { .. }
                | ModularFrameKind::RgbBlendCrop { .. }
        );
        let wp_tree = (frame.effort.flat_wp_contexts && frame.use_wp && displayed)
            .then(|| {
                let bucketed: Vec<bool> = (0..nb_chans)
                    .map(|chan| {
                        chan_preds[chan] == PREDICTOR_WEIGHTED
                            && !(chan >= frame.num_color && frame.alpha_constant.is_some())
                    })
                    .collect();
                bucketed.contains(&true).then(|| {
                    let shift = frame.settings.max_bits.saturating_sub(8);
                    wp_context_tree(&chan_preds, &offsets, &bucketed, shift)
                })
            })
            .flatten();
        let wp_params = if chan_preds.contains(&PREDICTOR_WEIGHTED) {
            wp_params
        } else {
            WpParams::DEFAULT
        };
        Self {
            frame,
            planes,
            rct_type,
            layout,
            grid: GroupGrid::new(frame.xsize, frame.ysize, layout),
            predictors: chan_preds,
            offsets,
            wp_tree,
            wp_params,
        }
    }

    fn num_contexts(&self) -> usize {
        self.wp_tree
            .as_ref()
            .map_or(self.frame.nb_chans, |tree| tree.num_ctx)
    }

    /// The LZ77 distance context, appended after the pixel contexts.
    fn distance_context(&self) -> u32 {
        self.num_contexts() as u32
    }

    fn buckets(&self) -> Option<&[WpChannelContexts]> {
        self.wp_tree.as_ref().map(|tree| tree.buckets.as_slice())
    }

    /// The solid alpha channel of a `w` x `h` stream, coded as a run of its
    /// one context's zero residuals.
    fn tail(&self, w: usize, h: usize) -> ConstantTail {
        let nb_chans = self.frame.nb_chans;
        match self.frame.alpha_constant {
            Some(_) => ConstantTail {
                context: self
                    .wp_tree
                    .as_ref()
                    .map_or(channel_to_context(nb_chans - 1, nb_chans), |tree| {
                        tree.buckets[nb_chans - 1].contexts[0]
                    }),
                count: w * h,
            },
            None => ConstantTail::NONE,
        }
    }

    /// The residual tokens of the `w` x `h` rectangle at (`x0`, `y0`), one
    /// stream per channel.
    #[allow(clippy::too_many_arguments)]
    fn tokenize(
        &self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Vec<Vec<crate::entropy::Token>> {
        let frame = self.frame;
        tokenize_channels_with_wp(
            self.planes,
            frame.alpha,
            frame.xsize,
            frame.ysize,
            x0,
            y0,
            w,
            h,
            frame.num_color,
            &self.predictors,
            frame.grad_pack_fn,
            pool,
            scratch,
            self.wp_params,
            frame.alpha_constant,
            self.buckets(),
        )
    }

    /// Writes the complete frame: header, TOC and sections.
    pub(super) fn write_frame(
        &self,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
        writer: &mut BitWriter,
    ) {
        let frame = self.frame;
        write_frame_header_modular_kind(frame.alpha.is_some(), frame.kind, self.layout, writer);
        if self.grid.single_group() {
            self.write_single_group(pool, scratch, writer);
        } else {
            self.write_multi_group(pool, scratch, writer);
        }
    }

    /// One section: GroupHeader, local tree, pixel histograms and pixels.
    fn write_single_group(
        &self,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
        writer: &mut BitWriter,
    ) {
        let frame = self.frame;
        let (xsize, ysize) = (frame.xsize, frame.ysize);
        let speed = frame.settings.speed;
        let mut section = BitWriter::new();
        if let ModularFrameKind::Patched { references, .. } = frame.kind {
            write_patch_dictionary(references, frame.alpha.is_some(), scratch, &mut section);
        }
        crate::lossless::header::write_dc_quant(frame.kind, &mut section);
        // 1 bit: has_tree = 0  (no global tree; the local tree lives in the GroupHeader).
        section.write(1, 0);
        // GroupHeader: use_global_tree=0, wp_default=1, RCT transform on R/G/B.
        section.write(1, 0);
        write_wp_header(self.wp_params, &mut section);
        write_modular_transforms(frame.nb_chans, self.rct_type, &mut section);

        // Tokenize all channels, then collapse runs of identical tokens into
        // LZ77 back-references.
        let channel_tokens = self.tokenize(0, 0, xsize, ysize, pool, scratch);
        let tail = self.tail(xsize, ysize);
        let distance_ctx = self.distance_context();
        let lz_tokens =
            lz77_compress_channels_for_speed(channel_tokens, distance_ctx, speed, scratch);

        // Per-cluster prefix codes, balanced N-leaf tree.
        let code = build_lz_pixel_code_tails(
            speed.effort().lz_exhaustive_entropy(),
            &[lz_tokens.as_slice()],
            &[tail],
            self.num_contexts(),
            frame.min_symbol,
            speed.effort().lz_refined_entropy,
            false,
            None,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        write_flat_tree(
            self.wp_tree.as_ref(),
            &self.predictors,
            &self.offsets,
            &code,
            frame.min_symbol,
            &mut scratch.huffman_pool,
            &mut section,
        );

        // Emit the LZ77'd token stream.
        write_lz_section_tail(
            &lz_tokens,
            tail,
            distance_ctx,
            &code,
            frame.min_symbol,
            &mut section,
        );
        section.zero_pad_to_byte();

        // TOC.
        writer.write(1, 0); // no permutation
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(std::slice::from_mut(&mut section));
        writer.zero_pad_to_byte();
    }

    /// One global tree and histograms in DC global, then each AC group
    /// emits its tokens against those codes.
    fn write_multi_group(
        &self,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
        writer: &mut BitWriter,
    ) {
        let frame = self.frame;
        let speed = frame.settings.speed;
        let grid = self.grid;
        let num_dc_groups = grid.num_dc_groups();
        let num_groups = grid.num_groups();
        let num_sections = 1 + num_dc_groups + 1 + num_groups;
        let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

        // Tokenize each AC group (sub-image-local neighbors, matching what
        // we'll emit below) and run LZ77 over each group's stream separately so
        // back-references stay within a group's modular sub-image. Pool the
        // resulting LzToken streams to build a single global prefix code so
        // every per-group emission is guaranteed to be representable.
        let distance_ctx = self.distance_context();
        let group_lz_tokens: Vec<Vec<LzToken>> = {
            let lz_threads = group_lz_threads(speed, pool);
            let deep_lz = speed
                .effort()
                .lz_deep
                .then(|| DeepLzScratchPool::new(lz_threads));
            pool.steal_map_with_threads(scratch, num_groups, lz_threads, |group_index, scratch| {
                let (x0, y0, gw, gh) = grid.rect(group_index);
                match &deep_lz {
                    Some(deep_lz) => {
                        let channel_tokens = self.tokenize(x0, y0, gw, gh, pool, scratch);
                        deep_lz.with_depth(|depth| {
                            lz77_compress_channels_for_speed_with_depth(
                                channel_tokens,
                                distance_ctx,
                                speed,
                                depth,
                                scratch,
                            )
                        })
                    }
                    None => tokenize_runs_with_wp(
                        self.planes,
                        frame.alpha,
                        frame.xsize,
                        x0,
                        y0,
                        gw,
                        gh,
                        frame.num_color,
                        &self.predictors,
                        frame.grad_pack_fn,
                        scratch,
                        self.wp_params,
                        frame.alpha_constant,
                        self.buckets(),
                    ),
                }
            })
        };
        let group_tails: Vec<ConstantTail> = (0..num_groups)
            .map(|group_index| {
                let (_, _, gw, gh) = grid.rect(group_index);
                self.tail(gw, gh)
            })
            .collect();

        // ----- Section 0: DC global -----
        if let ModularFrameKind::Patched { references, .. } = frame.kind {
            write_patch_dictionary(references, frame.alpha.is_some(), scratch, &mut sections[0]);
        }
        let group_slices: Vec<&[LzToken]> = group_lz_tokens.iter().map(Vec::as_slice).collect();
        let code = build_lz_pixel_code_tails(
            speed.effort().lz_exhaustive_entropy(),
            &group_slices,
            &group_tails,
            self.num_contexts(),
            frame.min_symbol,
            speed.effort().lz_refined_entropy,
            false,
            None,
            &mut scratch.lz_entropy,
            &mut scratch.huffman_pool,
        );
        crate::lossless::header::write_dc_quant(frame.kind, &mut sections[0]);
        sections[0].write(1, 1); // has_tree = 1
        write_flat_tree(
            self.wp_tree.as_ref(),
            &self.predictors,
            &self.offsets,
            &code,
            frame.min_symbol,
            &mut scratch.huffman_pool,
            &mut sections[0],
        );
        // GroupHeader for the global modular image: use_global_tree=1, wp=1, RCT transform.
        sections[0].write(1, 1);
        write_wp_header(self.wp_params, &mut sections[0]);
        write_modular_transforms(frame.nb_chans, self.rct_type, &mut sections[0]);
        // An empty global stream still carries its ANS state: jxl-oxide reads it.
        write_lz_section::<LzToken>(&[], distance_ctx, &code, frame.min_symbol, &mut sections[0]);
        sections[0].zero_pad_to_byte();

        // ----- DC groups: empty GroupHeader only -----
        for section in &mut sections[1..1 + num_dc_groups] {
            section.write(1, 1); // use_global_tree
            write_wp_header(self.wp_params, section);
            section.write(2, 0); // 0 transforms
            section.zero_pad_to_byte();
        }

        // ----- AC global: trivial (all_default flags) -----
        let ac_global_idx = 1 + num_dc_groups;
        sections[ac_global_idx].write(1, 1);
        write_wp_header(self.wp_params, &mut sections[ac_global_idx]);
        sections[ac_global_idx].zero_pad_to_byte();

        write_lz_groups(
            &group_lz_tokens,
            &group_tails,
            &code,
            distance_ctx,
            frame.min_symbol,
            self.wp_params,
            pool,
            &mut sections[2 + num_dc_groups..],
        );

        // TOC.
        writer.write(1, 0);
        writer.zero_pad_to_byte();
        for s in &sections {
            write_toc_entry(s.bits_written() / 8, writer);
        }
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(&mut sections);
        writer.zero_pad_to_byte();
    }
}
