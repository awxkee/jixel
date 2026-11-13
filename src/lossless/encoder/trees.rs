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

//! Step 5 of a lossless frame: the learned-tree tournament.
//!
//! Three sources may each grow a learned MA tree (greedy over the standard
//! property vector, per-leaf predictors) and write a complete frame:
//!
//! - the RGB planes under the chosen color transform;
//! - a global palette's index image, for frames of at most 1024 colors;
//! - per-group palettes, for screens whose 256-px groups are low-color.
//!
//! First a ranking ([`SourcePlan::rank`]) compares the RGB and palette
//! sources by one coarse learn each, ranks the WP presets per source and
//! tries the color transform's runner-up: the corner-sampled WP proxy and the
//! gradient RCT estimator both mis-rank on a third of images. Then each
//! surviving source is learned in full and written, cheapest first; a source
//! whose estimate trails the best so far is not written.

use super::{ColorTransform, CoreFrame, SmallestFrame, TreeSearch, learned_tree_is_decisive};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::lossless::GroupLayout;
use crate::lossless::ma_encoding::write_learned_tree_frame;
use crate::lossless::ma_learning::{
    RANK_COARSE_PRUNE, RCT_RANK_MARGIN, finish_ma_learn, learn_best_layout, learn_ma_candidate,
    rank_bits, rank_coarse, rank_presets,
};
use crate::lossless::ma_source::{
    CoarseLearn, DENSE_FINAL_MIN_BITS_PER_VALUE, MA_COARSE_TARGET_SAMPLES, MA_DENSE_SAMPLING,
    MA_FULL_SAMPLING, MA_LITE_SAMPLING, MaSource, PALETTE_LAYOUT_MARGIN, PER_GROUP_MARGIN,
    RANK_MIN_VALUES, RGB_SINGLE_LAYOUT_MIN_VALUES, mixed_ma_source, rgb_ma_source,
    source_values_of,
};
use crate::lossless::palette::{
    self, GlobalPalette, PALETTE_COARSE_MARGIN, PALETTE_FINAL_MARGIN, build_global_palette,
};
use crate::lossless::rct::rct_planes;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// A learned tree estimated to save at least this fraction over the flat
/// coder also skips the v1 context-tree and flat frame candidates.
const MA_ALTERNATIVES_MIN_SAVINGS: f64 = 0.05;

/// A vector frame's global palette must lead by at least 25% at the shared
/// first-stage budget before the full RGB tree is skipped.
const VECTOR_PALETTE_MAX_SHARE: f64 = 0.75;

/// What the tournament leaves to the later steps.
pub(super) struct TreeOutcome {
    /// The RGB learn's group layout (else the first candidate layout): the
    /// v1 and flat candidates code under it.
    pub(super) layout: GroupLayout,
    /// The WP preset the RGB ranking chose: the per-group, v1 and flat
    /// candidates' too.
    pub(super) wp_params: WpParams,
    /// A written tree was estimated to win decisively over the flat coder.
    pub(super) decisive: bool,
    /// The RGB tree saves enough that the v1 and flat candidates are skipped.
    pub(super) skip_alternatives: bool,
}

/// Learns and writes every learned-tree candidate into `best`.
pub(super) fn learn_tree_candidates(
    frame: &CoreFrame<'_>,
    color: &mut ColorTransform<'_>,
    wp_params: WpParams,
    best: &mut SmallestFrame,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> TreeOutcome {
    let layouts = LayoutPlan::new(frame);
    if !frame.compare_tree_candidates {
        return TreeOutcome {
            layout: layouts.all[0],
            wp_params,
            decisive: false,
            skip_alternatives: false,
        };
    }
    let effort = frame.effort;

    // 5a: the global palette, when the frame has one.
    let palette = if effort.tree_global_palette {
        build_global_palette(
            frame.ycocg,
            frame.alpha,
            frame.num_color,
            frame.xsize,
            frame.ysize,
            pool,
            scratch,
        )
    } else {
        None
    };

    // 5b: which sources to learn, under which presets.
    let mut sources = SourcePlan::new(frame, wp_params, palette.is_some());
    if effort.tree_rank && matches!(frame.settings.decoding_speed, crate::DecodingSpeed::Slow) {
        sources.rank(frame, color, palette.as_ref(), &layouts, pool, scratch);
    } else if matches!(frame.tree_search, TreeSearch::Vector)
        && let Some(palette) = &palette
    {
        sources.rank_vector(frame, color, palette, layouts.rank_layout(), pool, scratch);
    }
    sources.keep_reusable_stages(&layouts);

    // 5c-5e: learn and write each source.
    let mut tournament = Tournament::new(layouts.all[0]);
    if sources.rgb {
        tournament.learn_rgb(frame, color, &mut sources, &layouts, best, pool, scratch);
    }
    if tournament.decisive {
        color.release_planes();
    }
    let bits_before_palette = best.bits();
    if sources.palette
        && let Some(palette) = &palette
    {
        tournament.learn_global_palette(
            frame,
            color,
            &mut sources,
            palette,
            &layouts,
            best,
            pool,
            scratch,
        );
    }
    // A global palette that already won leaves nothing for per-group ones,
    // and building them would scan the image for nothing.
    let global_palette_won = best.bits() != bits_before_palette;
    if effort.tree_group_palette && frame.num_color == 3 && !global_palette_won {
        tournament.learn_group_palettes(frame, color, &sources, best, pool, scratch);
    }

    TreeOutcome {
        layout: tournament.layout,
        wp_params: sources.wp_params,
        decisive: tournament.decisive,
        skip_alternatives: tournament.skip_alternatives,
    }
}

/// The co-occurrence-ordered per-group palettes replace the color-sorted
/// ones only when their first-stage estimate leads by this factor.
const ORDERED_GROUP_PALETTE_MARGIN: f64 = 0.97;

/// The group layouts the learns try.
///
/// With every decoding tool available, Slow lossless runs the first learning
/// stage (sampling plus a 32-leaf tree) under both 1024- and 256-pixel
/// groups, one layout at a time so a single sample set is alive. Everything
/// else keeps the codestream default.
struct LayoutPlan {
    /// Every candidate layout; the ranking learns run under the first.
    all: Vec<GroupLayout>,
    /// How many of them the RGB source learns.
    rgb_count: usize,
}

impl LayoutPlan {
    fn new(frame: &CoreFrame<'_>) -> Self {
        let dual = frame.compare_tree_candidates
            && frame.effort.tree_dual_layout
            && matches!(frame.settings.decoding_speed, crate::DecodingSpeed::Slow)
            // An image inside one 256-px group learns the same thing under
            // both layouts.
            && (frame.xsize > GroupLayout::DEFAULT.dim() || frame.ysize > GroupLayout::DEFAULT.dim());
        let all: Vec<GroupLayout> = if dual {
            vec![GroupLayout::LARGE, GroupLayout::DEFAULT]
        } else {
            vec![GroupLayout::DEFAULT]
        }
        .into_iter()
        .map(|layout| layout.with_upsampling(frame.settings.upsampling))
        .collect();
        // RGB sources of more than a megapixel-channel learn the 1024-px
        // layout only: the 256 layout's first stage costs a sampling pass and
        // its full learn doubles the frame's learning time.
        let rgb_count = if dual
            && source_values_of(frame.xsize, frame.ysize, frame.coded_chans)
                > RGB_SINGLE_LAYOUT_MIN_VALUES
        {
            1
        } else {
            all.len()
        };
        Self { all, rgb_count }
    }

    fn rank_layout(&self) -> GroupLayout {
        self.all[0]
    }

    fn rgb(&self) -> &[GroupLayout] {
        &self.all[..self.rgb_count]
    }
}

/// Which sources the tournament learns, and how.
struct SourcePlan {
    /// Learn a tree over the RGB planes.
    rgb: bool,
    /// Learn one over the global palette's index image.
    palette: bool,
    /// The RGB source's WP preset (also the per-group, v1 and flat ones').
    wp_params: WpParams,
    /// The palette source's WP preset.
    palette_wp: WpParams,
    /// The ranking learns' sample budget.
    rank_sampling: (usize, usize),
    /// The final learns' sample budget: the ranking's, or dense for an RGB
    /// source costing many bits per value.
    final_sampling: (usize, usize),
    /// Ranking stages kept for the final learns (`tree_fresh_final_samples`
    /// off): a final learn may start from them instead of sampling again.
    rgb_stage: Option<CoarseLearn>,
    palette_stage: Option<CoarseLearn>,
}

impl SourcePlan {
    fn new(frame: &CoreFrame<'_>, wp_params: WpParams, has_palette: bool) -> Self {
        let sampling = if frame.effort.tree_full_sampling {
            MA_FULL_SAMPLING
        } else {
            MA_LITE_SAMPLING
        };
        Self {
            rgb: true,
            palette: has_palette,
            wp_params,
            palette_wp: wp_params,
            rank_sampling: sampling,
            final_sampling: sampling,
            rgb_stage: None,
            palette_stage: None,
        }
    }

    /// Vector frames use the same preset and RCT for the ranking and final
    /// learn. Skip a clearly trailing RGB tree and deepen the palette's
    /// first stage directly, with both complete encodes still racing close
    /// scores. The ranking's reduced leaf budget differs on small probes.
    fn rank_vector(
        &mut self,
        frame: &CoreFrame<'_>,
        color: &mut ColorTransform<'_>,
        palette: &GlobalPalette,
        layout: GroupLayout,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        let sample = |source: &MaSource<'_>, scratch: &mut CoderScratch| {
            rank_coarse(
                source,
                frame.xsize,
                frame.ysize,
                frame.min_symbol,
                self.wp_params,
                frame.use_wp,
                layout,
                self.rank_sampling,
                pool,
                scratch,
            )
        };
        let palette = palette.source(frame.xsize, frame.ysize);
        if palette.total_values() <= MA_COARSE_TARGET_SAMPLES {
            return;
        }
        let Some(palette_stage) = sample(&palette, scratch)
            .filter(|stage| stage.samples.len() > MA_COARSE_TARGET_SAMPLES)
        else {
            return;
        };
        let rct_type = color.rct_type;
        let planes = color.planes(pool, scratch);
        let rgb = rgb_ma_source(
            planes,
            frame.alpha,
            frame.xsize,
            rct_type,
            frame.alpha_constant,
            frame.num_color,
        );
        if let Some(rgb_stage) = sample(&rgb, scratch) {
            if palette_stage.candidate.est_real
                <= VECTOR_PALETTE_MAX_SHARE * rgb_stage.candidate.est_real
            {
                self.rgb = false;
            } else if rgb_stage.samples.len() > MA_COARSE_TARGET_SAMPLES {
                self.rgb_stage = Some(rgb_stage);
            }
        }
        self.palette_stage = Some(palette_stage);
    }

    /// Ranks the sources by one coarse learn each under the current preset,
    /// then ranks the WP presets of each surviving source and tries the color
    /// transform's runner-up.
    fn rank(
        &mut self,
        frame: &CoreFrame<'_>,
        color: &mut ColorTransform<'_>,
        palette: Option<&GlobalPalette>,
        layouts: &LayoutPlan,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        let (xsize, ysize) = (frame.xsize, frame.ysize);
        let effort = frame.effort;
        let rank_layout = layouts.rank_layout();
        let header_bits = |params: WpParams| frame.wp_header_bits(params);
        let keep_stages = !effort.tree_fresh_final_samples;
        let rank_presets_deep = effort.tree_rank_deep();
        // An ink frame is a palette image by construction: its color planes
        // hold one value per ink, so only the palette source is learned.
        let palette_only =
            matches!(frame.kind, ModularFrameKind::XybBlend { .. }) && palette.is_some();
        let rank_sampling = self.rank_sampling;
        let coarse = |source: &MaSource<'_>, wp_params: WpParams, scratch: &mut CoderScratch| {
            let mut stage = rank_coarse(
                source,
                xsize,
                ysize,
                frame.min_symbol,
                wp_params,
                frame.use_wp,
                rank_layout,
                rank_sampling,
                pool,
                scratch,
            )?;
            if !keep_stages && !rank_presets_deep {
                // Later ranking uses only the coarse estimate, and the final
                // learn samples afresh. Release these rows before the next
                // source allocates its sample set.
                stage.samples = crate::ma_tree::MaSamples::with_capacity(0);
            }
            Some(stage)
        };

        // One coarse learn per source decides which are learned in full.
        let wp_params = self.wp_params;
        let mut rgb_stage = if palette_only {
            None
        } else {
            let rct_type = color.rct_type;
            let planes = color.planes(pool, scratch);
            let source = rgb_ma_source(
                planes,
                frame.alpha,
                xsize,
                rct_type,
                frame.alpha_constant,
                frame.num_color,
            );
            coarse(&source, wp_params, scratch)
        };
        let mut palette_stage =
            palette.and_then(|p| coarse(&p.source(xsize, ysize), wp_params, scratch));
        let coarse_of = |stage: &Option<CoarseLearn>| {
            stage.as_ref().map_or(f64::INFINITY, |s| {
                s.candidate.est_real + header_bits(wp_params)
            })
        };
        let (rgb_coarse, palette_coarse) = (coarse_of(&rgb_stage), coarse_of(&palette_stage));
        if effort.tree_dense_sampling
            && rgb_stage.as_ref().is_some_and(|stage| {
                stage.candidate.est_real
                    >= DENSE_FINAL_MIN_BITS_PER_VALUE
                        * source_values_of(xsize, ysize, frame.coded_chans) as f64
            })
        {
            self.final_sampling = MA_DENSE_SAMPLING;
        }
        self.rgb = rgb_coarse <= palette_coarse * PALETTE_COARSE_MARGIN;
        self.palette =
            palette_stage.is_some() && palette_coarse <= rgb_coarse * PALETTE_COARSE_MARGIN;

        // Small images stop here: the preset ranking costs more than it finds.
        let rank_more = source_values_of(xsize, ysize, frame.coded_chans) >= RANK_MIN_VALUES;
        if self.rgb && rank_more {
            // The RGB source's WP presets.
            let (best_bits, best_coarse, params, stage) = {
                let rct_type = color.rct_type;
                let planes = color.planes(pool, scratch);
                let source = rgb_ma_source(
                    planes,
                    frame.alpha,
                    xsize,
                    rct_type,
                    frame.alpha_constant,
                    frame.num_color,
                );
                rank_presets(
                    &source,
                    xsize,
                    ysize,
                    frame.min_symbol,
                    frame.use_wp,
                    if palette_only {
                        coarse(&source, self.wp_params, scratch)
                    } else {
                        rgb_stage.take()
                    },
                    self.wp_params,
                    &header_bits,
                    rank_presets_deep,
                    rank_layout,
                    self.rank_sampling,
                    keep_stages,
                    pool,
                    scratch,
                )
            };
            self.wp_params = params;
            rgb_stage = stage;

            // The transform runner-up under the chosen preset, with the same
            // coarse prune.
            if let Some((runner_up, cost)) = color.runner_up()
                && cost <= color.ranked[0].1 * RCT_RANK_MARGIN
            {
                // The runner-up samples the same number of rows. Keep the
                // first stage when both sets fit the ranking sample budget
                // and the final learn can reuse it if the runner-up loses.
                rgb_stage = rgb_stage.filter(|stage| {
                    let rows = stage.samples.len();
                    self.final_sampling == self.rank_sampling
                        && rows > MA_COARSE_TARGET_SAMPLES
                        && rows <= self.rank_sampling.0 / 2
                });
                let planes = (runner_up != 6)
                    .then(|| rct_planes(color.ycocg, xsize, ysize, runner_up, pool, scratch));
                let source = rgb_ma_source(
                    planes.as_ref().unwrap_or(color.ycocg),
                    frame.alpha,
                    xsize,
                    runner_up,
                    frame.alpha_constant,
                    frame.num_color,
                );
                if let Some(stage) = coarse(&source, self.wp_params, scratch)
                    && stage.candidate.est_real + header_bits(self.wp_params)
                        <= best_coarse * RANK_COARSE_PRUNE
                    && rank_bits(&stage) + header_bits(self.wp_params) < best_bits
                {
                    color.adopt(runner_up, planes);
                    rgb_stage = keep_stages.then_some(stage);
                }
            }
        }
        if self.palette
            && rank_more
            && rank_presets_deep
            && let Some(p) = palette
        {
            // The palette source's WP presets.
            let (_, _, params, stage) = rank_presets(
                &p.source(xsize, ysize),
                xsize,
                ysize,
                frame.min_symbol,
                frame.use_wp,
                palette_stage.take(),
                self.wp_params,
                &header_bits,
                true,
                rank_layout,
                self.rank_sampling,
                keep_stages,
                pool,
                scratch,
            );
            self.palette_wp = params;
            palette_stage = stage;
        }
        if keep_stages {
            (self.rgb_stage, self.palette_stage) = (rgb_stage, palette_stage);
        }
    }

    /// Drops the ranking stages the final learns cannot start from. A stage
    /// stands in for a final learn's first stage when both sampled the same
    /// source under the same layout and budget, and the ranking did not stop
    /// at its direct small-set learn.
    fn keep_reusable_stages(&mut self, layouts: &LayoutPlan) {
        let same_budget = self.final_sampling == self.rank_sampling;
        let reusable = |stage: Option<CoarseLearn>, final_layout: GroupLayout| {
            stage.filter(|stage| {
                final_layout == layouts.rank_layout()
                    && same_budget
                    && stage.samples.len() > MA_COARSE_TARGET_SAMPLES
            })
        };
        self.rgb_stage = reusable(self.rgb_stage.take(), layouts.rgb()[0]);
        self.palette_stage = reusable(self.palette_stage.take(), layouts.all[0]);
    }
}

/// The tournament's running state across its sources.
struct Tournament {
    /// The RGB learn's layout, else the first candidate layout.
    layout: GroupLayout,
    /// A written candidate's tree was estimated decisive.
    decisive: bool,
    /// The RGB tree saves enough to skip the v1 and flat candidates.
    skip_alternatives: bool,
    /// The RGB source's first-stage (coarse) and final estimates.
    rgb_coarse: f64,
    rgb_final: f64,
    /// The best source's first-stage estimate so far: the bar a further
    /// candidate must clear before it is grown into a complete frame.
    best_source_coarse: f64,
}

impl Tournament {
    fn new(layout: GroupLayout) -> Self {
        Self {
            layout,
            decisive: false,
            skip_alternatives: false,
            rgb_coarse: f64::INFINITY,
            rgb_final: f64::INFINITY,
            best_source_coarse: f64::INFINITY,
        }
    }

    /// 5c: the RGB planes under the chosen color transform.
    #[allow(clippy::too_many_arguments)]
    fn learn_rgb(
        &mut self,
        frame: &CoreFrame<'_>,
        color: &mut ColorTransform<'_>,
        sources: &mut SourcePlan,
        layouts: &LayoutPlan,
        best: &mut SmallestFrame,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        let first_stage = sources.rgb_stage.take();
        let rct_type = color.rct_type;
        let planes = color.planes(pool, scratch);
        let source = rgb_ma_source(
            planes,
            frame.alpha,
            frame.xsize,
            rct_type,
            frame.alpha_constant,
            frame.num_color,
        );
        let Some((layout, cand, coarse)) = learn_best_layout(
            frame.settings.speed,
            &source,
            frame.xsize,
            frame.ysize,
            layouts.rgb(),
            frame.min_symbol,
            sources.wp_params,
            frame.use_wp,
            f64::INFINITY,
            f64::INFINITY,
            false,
            sources.final_sampling,
            first_stage,
            pool,
            scratch,
        ) else {
            return;
        };
        self.layout = layout;
        self.rgb_coarse = coarse;
        self.rgb_final = cand.est_real;
        self.best_source_coarse = coarse;

        let mut candidate = BitWriter::new();
        let estimated_savings = write_learned_tree_frame(
            frame.settings.speed,
            &source,
            frame.alpha.is_some(),
            frame.kind,
            frame.xsize,
            frame.ysize,
            layout,
            frame.min_symbol,
            pool,
            scratch,
            sources.wp_params,
            frame.use_wp,
            &cand,
            frame.tree_search,
            &mut candidate,
        );
        self.decisive = learned_tree_is_decisive(estimated_savings);
        self.skip_alternatives = estimated_savings >= MA_ALTERNATIVES_MIN_SAVINGS;
        best.offer(candidate);
    }

    /// 5d: the global palette's index image, with its own learned tree, WP
    /// preset, layout and literal/LZ77 choice. libjxl's e7 lossless gains
    /// ~10% from it on palette graphics; a decisive palette tree ends the
    /// search like a decisive RGB tree does.
    #[allow(clippy::too_many_arguments)]
    fn learn_global_palette(
        &mut self,
        frame: &CoreFrame<'_>,
        color: &mut ColorTransform<'_>,
        sources: &mut SourcePlan,
        palette: &GlobalPalette,
        layouts: &LayoutPlan,
        best: &mut SmallestFrame,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        let first_stage = sources.palette_stage.take();
        let source = palette.source(frame.xsize, frame.ysize);
        let Some((layout, cand, coarse)) = learn_best_layout(
            frame.settings.speed,
            &source,
            frame.xsize,
            frame.ysize,
            &layouts.all,
            frame.min_symbol,
            sources.palette_wp,
            frame.use_wp,
            self.rgb_coarse * PALETTE_COARSE_MARGIN,
            // Room for the 256 layout's ~4% on palette graphics.
            self.rgb_final * PALETTE_FINAL_MARGIN * PALETTE_LAYOUT_MARGIN,
            true,
            sources.final_sampling,
            first_stage,
            pool,
            scratch,
        ) else {
            return;
        };
        // The frame write (tokenize, code, sections) is the palette
        // candidate's most expensive step after the learn; a final estimate
        // clearly behind the RGB frame's cannot win the size comparison.
        if cand.est_real <= self.rgb_final * PALETTE_FINAL_MARGIN {
            let mut candidate = BitWriter::new();
            let estimated_savings = write_learned_tree_frame(
                frame.settings.speed,
                &source,
                frame.alpha.is_some(),
                frame.kind,
                frame.xsize,
                frame.ysize,
                layout,
                frame.min_symbol,
                pool,
                scratch,
                sources.palette_wp,
                frame.use_wp,
                &cand,
                frame.tree_search,
                &mut candidate,
            );
            if learned_tree_is_decisive(estimated_savings) {
                self.decisive = true;
                color.release_planes();
            }
            best.offer(candidate);
        }
        self.best_source_coarse = self.best_source_coarse.min(coarse);
    }

    /// 5e: per-group palettes. Screen content is a mosaic of low-color
    /// regions (a UI screen: 9 of 15 groups hold 221 to 870 colors) that no
    /// global palette can capture. Each qualifying group declares its own
    /// Palette transform while one learned tree still codes the whole frame,
    /// telling the regimes apart through the stream-id property.
    fn learn_group_palettes(
        &mut self,
        frame: &CoreFrame<'_>,
        color: &mut ColorTransform<'_>,
        sources: &SourcePlan,
        best: &mut SmallestFrame,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        let grid = frame.grid;
        if grid.single_group() {
            return;
        }
        let palettes: Vec<Option<palette::LocalPaletteGroup>> =
            pool.steal_map(scratch, grid.num_groups(), |group_index, _scratch| {
                let (x0, y0, w, h) = grid.rect(group_index);
                palette::build_local_palette_group_limited(
                    frame.ycocg,
                    frame.alpha,
                    frame.xsize,
                    x0,
                    y0,
                    w,
                    h,
                    palette::LOCAL_PALETTE_MAX_COLORS,
                )
            });
        let covered: usize = palettes
            .iter()
            .flatten()
            .map(|palette| palette.w * palette.h)
            .sum();
        // The groups that keep their color planes cost the same either way,
        // so the candidate is only worth its sampling and first-stage learn
        // when most of the frame palettises (a UI screen: 60%; a
        // screen-content photo mosaic: 44%, where the complete frame then
        // loses anyway).
        if covered * 2 < frame.xsize * frame.ysize {
            return;
        }
        // Two index orders per group race on their first-stage estimate: the
        // color sort (luma first), which the learned tree codes cheaper on
        // antialiased text and UI, and the spatial co-occurrence chain, which
        // wins on screen maps and quantised fields. The index-plane
        // estimator alone cannot tell them apart under a learned tree (it
        // picked the chain on screenshots at +2.7..+3.5 % of the frame).
        let ordered: Vec<Option<palette::LocalPaletteGroup>> =
            pool.steal_map(scratch, grid.num_groups(), |group_index, _scratch| {
                palette::ordered_local_palette(palettes[group_index].as_ref()?)
            });
        let layout = frame.default_layout();
        let mut raced: Option<(MaSource<'_>, CoarseLearn)> = None;
        let has_reordered = ordered.iter().any(Option::is_some);
        for use_reordered in [false, true]
            .into_iter()
            .take(1 + usize::from(has_reordered))
        {
            let groups = palettes.iter().zip(&ordered).map(|(original, changed)| {
                if use_reordered {
                    changed.as_ref().or(original.as_ref())
                } else {
                    original.as_ref()
                }
            });
            let mixed = mixed_ma_source(
                frame.ycocg,
                frame.alpha,
                frame.xsize,
                groups,
                frame.alpha_constant,
            );
            let stage = learn_ma_candidate(
                &mixed,
                frame.xsize,
                frame.ysize,
                layout,
                frame.min_symbol,
                sources.wp_params,
                frame.use_wp,
                mixed.max_leaves(),
                sources.final_sampling,
                pool,
                scratch,
            );
            let Some(stage) = stage else {
                continue;
            };
            // The chain's estimate runs ~2 % optimistic against its written
            // frame (the final tree recovers more from the color sort), so
            // it has to lead by more than that.
            if raced.as_ref().is_none_or(|(_, leader)| {
                stage.candidate.est_real < leader.candidate.est_real * ORDERED_GROUP_PALETTE_MARGIN
            }) {
                raced = Some((mixed, stage));
            }
        }
        // The first stage is grown into a frame only when it is close to the
        // best source's own first stage (a winning per-group plan lands at
        // 0.75, the losers at 1.08 and above; the margin covers a
        // screen-content tie).
        let Some((mixed, stage)) = raced else {
            return;
        };
        if stage.candidate.est_real >= self.best_source_coarse * PER_GROUP_MARGIN {
            return;
        }
        let cand = finish_ma_learn(
            frame.settings.speed,
            stage,
            frame.min_symbol,
            frame.use_wp,
            mixed.max_leaves(),
            pool,
            scratch,
        );
        let mut candidate = BitWriter::new();
        let estimated_savings = write_learned_tree_frame(
            frame.settings.speed,
            &mixed,
            frame.alpha.is_some(),
            frame.kind,
            frame.xsize,
            frame.ysize,
            layout,
            frame.min_symbol,
            pool,
            scratch,
            sources.wp_params,
            frame.use_wp,
            &cand,
            frame.tree_search,
            &mut candidate,
        );
        if learned_tree_is_decisive(estimated_savings) {
            self.decisive = true;
            color.release_planes();
        }
        best.offer(candidate);
    }
}
