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

//! One lossless frame, coded step by step.
//!
//! 1. Palette transform: an image of few colors (per frame or per group)
//!    codes as a palette and an index channel and is done.
//! 2. Squeeze: a progressive frame codes its Squeeze pyramid and is done.
//! 3. Color transform: the RCTs are ranked; YCoCg stays unless one wins.
//! 4. Weighted-predictor parameters for the chosen transform.
//! 5. Learned trees ([`trees`]): the RGB planes, a global palette and
//!    per-group palettes each grow a tree and write a complete frame. A tree
//!    estimated to win clearly ends the search here.
//! 6. Per-channel predictors for the remaining candidates.
//! 7. The v1 context tree (WP-activity contexts) writes its frame.
//! 8. The flat coder ([`flat`]) writes its frame.
//!
//! Below Slow only the flat frame is written; otherwise every written frame
//! competes on its complete byte-aligned size.

mod flat;
mod trees;

use super::context::{try_encode_context_tree_multi_group, try_encode_context_tree_single_group};
use super::frame::keep_smaller_writer;
use super::palette::{try_encode_local_palette_multi_group, try_encode_palette_single_group};
use super::predictor::{
    GradPackInteriorFn, choose_predictors_with_wp, choose_wp_params, fixed_predictor,
    selected_grad_pack_interior_fn,
};
use super::rct::{rank_rcts, rct_planes};
use super::squeeze::{encode_squeeze_multigroup, encode_squeeze_single_group};
use super::{GroupGrid, GroupLayout, LosslessSettings, constant_alpha_value, lossless_min_symbol};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::effort::Effort;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;
use flat::FlatCoder;

/// The learned-tree entropy proxy must beat its flat alternative decisively
/// before Slow mode trusts it enough to omit the context/flat final encodes.
/// Near decisions retain the exact byte-level tournament.
pub(super) const MA_DECISIVE_MIN_SAVINGS: f64 = 0.18;

/// Row steps of the RCT ranking under `rct_dense_rows` and below it.
const RCT_DENSE_ROW_STEP: usize = 4;

const RCT_SPARSE_ROW_STEP: usize = 16;

#[inline]
pub(super) fn learned_tree_is_decisive(estimated_savings: f64) -> bool {
    estimated_savings >= MA_DECISIVE_MIN_SAVINGS
}

/// Codes one lossless frame of `linear` (YCoCg planes, or one gray plane)
/// and `alpha`. Returns whether a learned tree won decisively, which tells
/// the caller a tile-patch plan cannot beat this frame.
pub(crate) fn encode_frame_lossless_core(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    settings: LosslessSettings,
    frame_kind: ModularFrameKind<'_>,
    local_trees: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    encode_frame_lossless_core_impl(
        linear,
        alpha,
        settings,
        frame_kind,
        true,
        local_trees,
        pool,
        scratch,
        writer,
    )
}

/// [`encode_frame_lossless_core`], with the palette-transform step
/// optional: the step's own size comparison codes the frame without it.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_frame_lossless_core_impl(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    settings: LosslessSettings,
    frame_kind: ModularFrameKind<'_>,
    allow_palette_candidates: bool,
    local_trees: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    let frame = CoreFrame::new(linear, alpha, settings, frame_kind, local_trees);

    // Step 1: a few-color image codes as a palette and an index channel.
    if allow_palette_candidates
        && let Some(decisive) = frame.encode_palette_transform(pool, scratch, writer)
    {
        return decisive;
    }

    // Step 2: a progressive frame codes its Squeeze pyramid. A multi-group
    // frame whose squeezed channels outgrow the global stream falls through
    // to the non-progressive path.
    if frame.encode_squeeze(pool, scratch, writer) {
        return false;
    }

    // Step 3: the color transform.
    let mut color = ColorTransform::rank(&frame, pool, scratch);

    // Step 4: the Weighted Predictor's parameters under that transform.
    let wp_params = frame.choose_wp_params(color.rct_type, pool, scratch);

    // Step 5: learned-tree candidates. Each one written is a complete frame;
    // the smallest so far is kept.
    let mut best = SmallestFrame::default();
    let trees =
        trees::learn_tree_candidates(&frame, &mut color, wp_params, &mut best, pool, scratch);
    if trees.decisive || trees.skip_alternatives {
        writer.append(best.get().expect("decisive learned-tree candidate"));
        return trees.decisive;
    }

    // Step 6: per-channel predictors for the v1 and flat candidates, under
    // the learned tree's group layout (deferred: a decisive tree never
    // needs them).
    let layout = trees.layout;
    let grid = GroupGrid::new(frame.xsize, frame.ysize, layout);
    let rct_type = color.rct_type;
    let planes = color.planes(pool, scratch);
    let predictors = if frame.effort.predictor_search {
        choose_predictors_with_wp(
            planes,
            frame.alpha,
            frame.xsize,
            frame.ysize,
            frame.num_color,
            pool,
            scratch,
            trees.wp_params,
            frame.use_wp,
        )
    } else {
        [fixed_predictor(frame.use_wp); 4]
    };
    let flat = FlatCoder::new(
        &frame,
        planes,
        rct_type,
        layout,
        &predictors,
        trees.wp_params,
    );

    // Step 7: the v1 context tree splits each channel's context on the WP
    // activity property; a big win on smooth+edge content.
    if frame.compare_tree_candidates && frame.effort.tree_ctx_v1 && frame.use_wp {
        let mut candidate = BitWriter::new();
        let written = if grid.single_group() {
            try_encode_context_tree_single_group(
                frame.settings.speed,
                frame.num_color,
                planes,
                frame.alpha,
                frame.alpha_constant,
                frame.xsize,
                frame.ysize,
                layout,
                frame.kind,
                &predictors,
                frame.min_symbol,
                rct_type,
                pool,
                scratch,
                trees.wp_params,
                &mut candidate,
            )
        } else {
            try_encode_context_tree_multi_group(
                frame.settings.speed,
                frame.num_color,
                planes,
                frame.alpha,
                frame.alpha_constant,
                frame.xsize,
                frame.ysize,
                layout,
                frame.kind,
                &predictors,
                grid.x_groups,
                grid.y_groups,
                grid.num_dc_groups(),
                frame.min_symbol,
                rct_type,
                pool,
                scratch,
                trees.wp_params,
                &mut candidate,
            )
        };
        if written {
            best.offer(candidate);
        }
    }

    // Step 8: the flat coder, the last candidate (and below Slow the only
    // one, written straight to the output).
    if frame.compare_tree_candidates {
        let mut candidate = BitWriter::new();
        flat.write_frame(pool, scratch, &mut candidate);
        best.offer(candidate);
        writer.append(best.get().expect("flat lossless candidate"));
    } else {
        flat.write_frame(pool, scratch, writer);
    }
    false
}

/// One frame's input and every setting derived from it before a candidate
/// is tried.
struct CoreFrame<'a> {
    /// The lossless front end's YCoCg planes (one plane for gray).
    ycocg: &'a Image3Si,
    alpha: Option<&'a AlphaPlane>,
    settings: LosslessSettings,
    effort: Effort,
    kind: ModularFrameKind<'a>,
    /// Learned-tree frames may give groups their own trees.
    local_trees: bool,
    xsize: usize,
    ysize: usize,
    num_color: usize,
    /// Modular channels: the color planes, then alpha.
    nb_chans: usize,
    /// The value of a solid alpha plane. It is coded as a constant channel
    /// (Zero predictor, offset = the value, one-symbol histogram): no bits,
    /// no predictor and no context-tree work in the decoder, which otherwise
    /// ran the Weighted Predictor over the whole plane.
    alpha_constant: Option<i32>,
    /// Channels that carry samples: size gates treat a solid-alpha image as
    /// its color image.
    coded_chans: usize,
    /// Near-lossless samples share a `2k+1` grid. The Weighted predictor's
    /// averaging leaves it, so the tiers without a learned tree keep their
    /// fixed predictor on the grid with Gradient instead.
    use_wp: bool,
    /// The first LZ77 symbol (see [`lossless_min_symbol`]).
    min_symbol: u32,
    /// The codestream-default (256-px) group grid.
    grid: GroupGrid,
    grad_pack_fn: GradPackInteriorFn,
    /// The learned-tree tournament runs: every candidate writes a complete
    /// frame and the smallest wins. Tree headers, clustering, hybrid uint
    /// choices and LZ77 can otherwise reverse the order predicted from
    /// residual entropy.
    compare_tree_candidates: bool,
}

impl<'a> CoreFrame<'a> {
    fn new(
        ycocg: &'a Image3Si,
        alpha: Option<&'a AlphaPlane>,
        settings: LosslessSettings,
        kind: ModularFrameKind<'a>,
        local_trees: bool,
    ) -> Self {
        let effort = *settings.speed.effort();
        let (xsize, ysize) = (ycocg.xsize(), ycocg.ysize());
        let num_color = settings.num_color;
        let alpha_constant = alpha.and_then(constant_alpha_value);
        Self {
            ycocg,
            alpha,
            settings,
            effort,
            kind,
            local_trees,
            xsize,
            ysize,
            num_color,
            nb_chans: num_color + usize::from(alpha.is_some()),
            alpha_constant,
            coded_chans: num_color + usize::from(alpha.is_some() && alpha_constant.is_none()),
            use_wp: settings.decoding_speed.use_weighted_predictor()
                && (effort.learned_tree || !settings.near_lossless),
            min_symbol: lossless_min_symbol(settings.max_bits),
            grid: GroupGrid::new(xsize, ysize, GroupLayout::DEFAULT),
            grad_pack_fn: selected_grad_pack_interior_fn(),
            compare_tree_candidates: kind.allows_learned_tree()
                && effort.learned_tree
                && (num_color == 3 || num_color == 1)
                && settings.decoding_speed.use_ma_trees(),
        }
    }

    /// The codestream-default layout at the frame's upsampling.
    fn default_layout(&self) -> GroupLayout {
        GroupLayout::DEFAULT.with_upsampling(self.settings.upsampling)
    }

    /// WP headers a frame signals: one per section that has a GroupHeader.
    fn wp_header_count(&self) -> usize {
        if self.grid.single_group() {
            1
        } else {
            2 + self.grid.num_dc_groups() + self.grid.num_groups()
        }
    }

    /// Bits the frame's WP headers cost over the default preset's.
    fn wp_header_bits(&self, params: WpParams) -> f64 {
        if params == WpParams::DEFAULT {
            0.0
        } else {
            (51 * self.wp_header_count()) as f64
        }
    }

    /// Step 1: the Palette transform, for regular RGB(A) frames whose single
    /// group holds at most 256 colors, or whose 256-px groups are each
    /// low-color although the frame is not (large graphics). High-color
    /// groups stay on the normal path. Returns `None` when it does not apply,
    /// else the decisive flag of the frame written.
    fn encode_palette_transform(
        &self,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
        writer: &mut BitWriter,
    ) -> Option<bool> {
        if !self.kind.is_regular() || self.num_color != 3 {
            return None;
        }
        let mut palette_writer = BitWriter::new();
        let applies = if self.grid.single_group() {
            try_encode_palette_single_group(
                self.ycocg,
                self.alpha,
                self.xsize,
                self.ysize,
                self.min_symbol,
                self.grad_pack_fn,
                self.settings.speed,
                self.use_wp,
                self.kind,
                self.default_layout(),
                scratch,
                &mut palette_writer,
            )
        } else {
            try_encode_local_palette_multi_group(
                self.ycocg,
                self.alpha,
                self.xsize,
                self.ysize,
                self.grid.x_groups,
                self.grid.y_groups,
                self.grid.num_dc_groups(),
                self.min_symbol,
                self.grad_pack_fn,
                self.settings.speed,
                self.use_wp,
                self.kind,
                self.default_layout(),
                pool,
                scratch,
                &mut palette_writer,
            )
        };
        if !applies {
            return None;
        }
        // `palette_dual`: the frame without the transform competes on size.
        if self.effort.palette_dual {
            let mut normal_writer = BitWriter::new();
            let normal_decisive = encode_frame_lossless_core_impl(
                self.ycocg,
                self.alpha,
                self.settings,
                self.kind,
                false,
                self.local_trees,
                pool,
                scratch,
                &mut normal_writer,
            );
            if normal_writer.bits_written() < palette_writer.bits_written() {
                writer.append(&normal_writer);
                return Some(normal_decisive);
            }
        }
        writer.append(&palette_writer);
        Some(false)
    }

    /// Step 2: the Squeeze pyramid of a progressive regular RGB(A) frame.
    /// Returns whether it was written: a multi-group frame needs every
    /// squeezed channel to fit the global stream.
    fn encode_squeeze(
        &self,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
        writer: &mut BitWriter,
    ) -> bool {
        if !self.settings.progressive || !self.kind.is_regular() || self.num_color != 3 {
            return false;
        }
        if self.grid.single_group() {
            encode_squeeze_single_group(
                self.ycocg,
                self.alpha,
                self.xsize,
                self.ysize,
                self.min_symbol,
                self.grad_pack_fn,
                self.settings.speed,
                self.use_wp,
                self.kind,
                self.default_layout(),
                pool,
                scratch,
                writer,
            );
            true
        } else {
            encode_squeeze_multigroup(
                self.ycocg,
                self.alpha,
                self.xsize,
                self.ysize,
                self.min_symbol,
                self.grid.x_groups,
                self.grid.y_groups,
                self.grid.x_dc_groups,
                self.grid.y_dc_groups,
                self.grid.num_dc_groups(),
                self.grid.num_groups(),
                self.grad_pack_fn,
                self.settings.speed,
                self.use_wp,
                self.kind,
                self.default_layout(),
                pool,
                scratch,
                writer,
            )
        }
    }

    /// Step 4: the WP preset searched under `wp_search`, else the default.
    fn choose_wp_params(
        &self,
        rct_type: u32,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> WpParams {
        if self.effort.wp_search && self.use_wp {
            choose_wp_params(
                self.ycocg,
                rct_type,
                self.alpha,
                self.num_color,
                self.wp_header_count(),
                pool,
                scratch,
            )
        } else {
            WpParams::DEFAULT
        }
    }
}

/// Step 3: the reversible color transform (RCT) the frame is coded under.
///
/// Slow ranks every transform over the same fixed-context gradient cost
/// libjxl uses. When a non-YCoCg transform wins, the color planes are
/// rebuilt (on first use: an ink frame's palette source usually wins
/// without them) and the frame headers signal it. A single gray plane has
/// no RCT. Near-lossless planes rank at every tier: their values share a
/// `2k+1` grid that only the subtract-type transforms preserve, and
/// YCoCg's shifts would scatter it.
struct ColorTransform<'a> {
    ycocg: &'a Image3Si,
    num_color: usize,
    /// The signaled RCT: 6 (YCoCg, the input itself) for color, 0 for gray.
    rct_type: u32,
    /// The color planes under a non-YCoCg `rct_type`, once built.
    planes: Option<Image3Si>,
    /// Every transform with its estimated cost, cheapest first; empty
    /// without the search.
    ranked: Vec<(u32, f32)>,
}

impl<'a> ColorTransform<'a> {
    fn rank(frame: &CoreFrame<'a>, pool: &ThreadPool, scratch: &mut CoderScratch) -> Self {
        let effort = frame.effort;
        let ranked = if frame.kind.allows_learned_tree()
            && (effort.rct_search || frame.settings.near_lossless)
            && frame.num_color == 3
        {
            let row_step = if effort.rct_every_row {
                1
            } else if effort.rct_dense_rows {
                RCT_DENSE_ROW_STEP
            } else {
                RCT_SPARSE_ROW_STEP
            };
            rank_rcts(
                frame.ycocg,
                frame.xsize,
                frame.ysize,
                row_step,
                pool,
                scratch,
            )
        } else {
            Vec::new()
        };
        let default_rct = if frame.num_color == 3 { 6 } else { 0 };
        Self {
            ycocg: frame.ycocg,
            num_color: frame.num_color,
            rct_type: ranked.first().map_or(default_rct, |&(t, _)| t),
            planes: None,
            ranked,
        }
    }

    /// The color planes under `rct_type`, built on first use.
    fn planes(&mut self, pool: &ThreadPool, scratch: &mut CoderScratch) -> &Image3Si {
        if self.num_color != 3 || self.rct_type == 6 {
            return self.ycocg;
        }
        let ycocg = self.ycocg;
        let rct_type = self.rct_type;
        self.planes.get_or_insert_with(|| {
            rct_planes(ycocg, ycocg.xsize(), ycocg.ysize(), rct_type, pool, scratch)
        })
    }

    /// Frees the transformed planes. A decisive learned tree never returns to
    /// the RCT-based v1/flat paths, and the palette candidates need only the
    /// YCoCg input.
    fn release_planes(&mut self) {
        self.planes = None;
    }

    /// Switches to `rct_type`, whose planes (`None` for YCoCg) were built by
    /// the caller.
    fn adopt(&mut self, rct_type: u32, planes: Option<Image3Si>) {
        self.rct_type = rct_type;
        self.planes = planes;
    }
}

/// The smallest complete frame among the candidates written so far.
#[derive(Default)]
struct SmallestFrame {
    writer: Option<BitWriter>,
}

impl SmallestFrame {
    fn offer(&mut self, candidate: BitWriter) {
        keep_smaller_writer(&mut self.writer, candidate);
    }

    fn bits(&self) -> Option<usize> {
        self.writer.as_ref().map(BitWriter::bits_written)
    }

    fn get(&self) -> Option<&BitWriter> {
        self.writer.as_ref()
    }
}
