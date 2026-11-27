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
//! sources by one coarse learn each and ranks the WP presets per source: the
//! corner-sampled WP proxy mis-ranks on a third of images. Then each
//! surviving source is learned in full and written, cheapest first; a source
//! whose estimate trails the best so far is not written.

use super::{ColorTransform, CoreFrame, SmallestFrame, TreeSearch, learned_tree_is_decisive};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::lossless::GroupLayout;
use crate::lossless::ma_encoding::write_learned_tree_frame;
use crate::lossless::ma_learning::{
    dense_no_wp_final, finish_ma_learn, learn_best_layout, learn_ma_candidate, rank_coarse,
    rank_presets,
};
use crate::lossless::ma_source::{
    CoarseLearn, DENSE_FINAL_MIN_BITS_PER_VALUE, LearnedCandidate, MA_COARSE_TARGET_SAMPLES,
    MA_DENSE_SAMPLING, MA_FULL_SAMPLING, MA_LITE_SAMPLING, MaSource, PALETTE_LAYOUT_MARGIN,
    PER_GROUP_MARGIN, RANK_MIN_VALUES, RGB_SINGLE_LAYOUT_MIN_VALUES, mixed_ma_source,
    rgb_ma_source, source_values_of,
};
use crate::lossless::palette::{
    self, GlobalPalette, LARGE_PALETTE_COARSE_MARGIN, OrderRace, PALETTE_COARSE_MARGIN,
    PALETTE_FINAL_MARGIN, PackedPalette, build_global_palettes,
};
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
    let vector = matches!(frame.tree_search, TreeSearch::Vector);
    let palettes = if effort.tree_global_palette {
        build_global_palettes(
            frame.ycocg,
            frame.alpha,
            frame.num_color,
            frame.xsize,
            frame.ysize,
            // Vector-layer frames race the order at their Medium effort too:
            // a base of 500-1000 colors under the estimate's order is what
            // loses the layered arm its race against the VarDCT stack (31
            // illustrations and comics, 7 distances: two gain 14.5 % and
            // 25.3 % at equal SSIMULACRA2, none loses over 0.6 %).
            (effort.palette_order_race || effort.palette_large || vector).then_some(OrderRace {
                min_symbol: frame.min_symbol,
                use_wp: frame.use_wp,
                // A vector-layer frame keeps its palette in place of
                // the RGB tree.
                large: effort.palette_large && !vector,
                small: effort.palette_order_race || vector,
                // A vector-layer frame is flat by construction, which
                // is what the ramp order is for (a manga page at
                // distance 0.5 and 1: 15 % and 9 % smaller; 116 other
                // lossy encodes unchanged at equal time).
                ramp: vector,
            }),
            pool,
            scratch,
        )
    } else {
        None
    };
    let (palette, raced_palette) = match palettes {
        Some((palette, raced)) => (Some(palette), raced),
        None => (None, None),
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
    let large_palette = palette.as_ref().is_some_and(GlobalPalette::is_large);
    let bits_before_palette = best.bits();
    if sources.palette
        && let Some(palette) = &palette
    {
        tournament.learn_global_palette(
            frame,
            color,
            &mut sources,
            palette,
            raced_palette.as_ref(),
            &layouts,
            best,
            pool,
            scratch,
        );
    }
    // A global palette that already won leaves nothing for per-group ones,
    // and building them would scan the image for nothing.
    let global_palette_won = best.bits() != bits_before_palette;
    // A large palette is one more candidate: it does not stand in for the
    // per-group ones.
    if effort.tree_group_palette && frame.num_color == 3 && (large_palette || !global_palette_won) {
        tournament.learn_group_palettes(frame, color, &sources, best, pool, scratch);
    }

    TreeOutcome {
        layout: tournament.layout,
        wp_params: sources.wp_params,
        decisive: tournament.decisive,
        skip_alternatives: tournament.skip_alternatives,
    }
}

/// In the palette order race, an order whose complete estimate under the
/// first layout trails the other order's by more than this is neither
/// learned further nor written. The estimate ratio of two orders follows
/// their coded ratio within 3 % on frames above
/// [`ORDER_RACE_BOTH_MAX_BITS`] (34 raced frames of 134 graphics, photos
/// and screenshots).
const ORDER_RACE_MARGIN: f64 = 1.02;

/// A packed index image of more than four colors is learned past its first
/// stage only while that stage's estimate stays within this of the plain
/// index image's. The packed form wins about one such frame in six, and the
/// frames it wins start at or below the plain estimate with few exceptions.
const PACKED_COARSE_MARGIN: f64 = 1.1;
const ORDER_RACE_BOTH_MAX_BITS: f64 = 65536.0;

/// A large palette's frame at most this share of the RGB frame's bits ends
/// the per-group palettes' race by its coarse learn.
const LARGE_PALETTE_CLEAR_LEAD: f64 = 0.92;

/// The co-occurrence-ordered per-group palettes replace the color-sorted
/// ones only when their first-stage estimate leads by this factor.
const ORDERED_GROUP_PALETTE_MARGIN: f64 = 0.97;

/// A per-group-palette frame wins while within 1 % of the smallest other
/// frame: its groups decode one index plane where the others decode every
/// color plane. A 6.7 MP screenshot whose palette frame trailed by 0.56 %
/// decodes at 13.4 MP/s on one thread with it and 4.9 without, for 0.27 % of
/// the file; of 49 screenshots, composites, illustrations and comics that is
/// the only file the credit changes. A wider credit buys less: at 4 % a
/// second file changes, 0.88 % larger for 9 % of decode speed.
const GROUP_PALETTE_CREDIT: f64 = 0.99;

/// The credit when the RGB source's final learns went dense without the
/// Weighted Predictor (`SourcePlan::rgb_dense_no_wp`) and nearly every group
/// has a palette (`GROUP_PALETTE_DENSE_COVERAGE`). The denser tree shrinks
/// the RGB frame more than the per-group-palette one: two UI screenshots
/// whose palette frames then trail by 1.1 % and 4.1 % decode 1.9-2.5 times
/// faster with them.
const GROUP_PALETTE_DENSE_CREDIT: f64 = 0.96;

/// Share of the frame's pixels in palettised groups that earns
/// `GROUP_PALETTE_DENSE_CREDIT`. Those screenshots are at 99 % and 100 %; a
/// web page at 93 % decodes 5 % slower from its palette frame, not faster.
const GROUP_PALETTE_DENSE_COVERAGE: f64 = 0.98;

/// Groups a frame keeps under its large layout. One modular group decodes
/// on one thread, so an image under ~4 MP in 1024-px groups decodes on one
/// to four. With 512-px groups there (256-px at Kodak size) djxl decodes
/// 2.5-3.8x faster on eight threads; 63 images cost +0.17% (Kodak), +0.10%
/// (screens), +0.02% (art), +0.05% (2-3 MP photos), inside the learner's
/// per-image noise except on Kodak, and encode 1-8% faster. Larger images
/// already have the groups and are untouched.
const PARALLEL_MIN_GROUPS: usize = 6;

/// The group layouts the learns try.
///
/// With every decoding tool available, Slow lossless runs the first learning
/// stage (sampling plus a 32-leaf tree) under both the large and 256-pixel
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
        // The largest groups that still leave the decoder
        // `PARALLEL_MIN_GROUPS` of them to run on its threads.
        let large = (GroupLayout::DEFAULT.shift..=GroupLayout::LARGE.shift)
            .rev()
            .map(|shift| GroupLayout {
                shift,
                ..GroupLayout::DEFAULT
            })
            .find(|layout| {
                frame.xsize.div_ceil(layout.dim()) * frame.ysize.div_ceil(layout.dim())
                    >= PARALLEL_MIN_GROUPS
            })
            // A frame too small for that many 256-px groups (0.4 MP, or a
            // cropped region frame) decodes in no time: it keeps the density.
            .unwrap_or(GroupLayout::LARGE);
        let all: Vec<GroupLayout> = if dual && large != GroupLayout::DEFAULT {
            vec![large, GroupLayout::DEFAULT]
        } else {
            vec![GroupLayout::DEFAULT]
        }
        .into_iter()
        .map(|layout| layout.with_upsampling(frame.settings.upsampling))
        .collect();
        // RGB sources of more than a megapixel-channel learn the large
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
    /// The RGB source's final learns (the per-group palettes' too) take the
    /// dense budget without the Weighted Predictor (`dense_no_wp_final`).
    rgb_dense_no_wp: bool,
    /// The same for the palette source.
    palette_dense_no_wp: bool,
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
        // A lossy encode's layer frame learns its final trees from the full
        // budget; its sources are still ranked at the tier's.
        let final_sampling = if frame.settings.lossy_layer && frame.effort.layer_full_final_sampling
        {
            MA_FULL_SAMPLING
        } else {
            sampling
        };
        Self {
            rgb: true,
            palette: has_palette,
            wp_params,
            palette_wp: wp_params,
            rank_sampling: sampling,
            final_sampling,
            rgb_dense_no_wp: false,
            palette_dense_no_wp: false,
            rgb_stage: None,
            palette_stage: None,
        }
    }

    fn dense_no_wp(&self, palette: bool) -> bool {
        if palette {
            self.palette_dense_no_wp
        } else {
            self.rgb_dense_no_wp
        }
    }

    /// The sample budget of a source's final learns.
    fn final_budget(&self, palette: bool) -> (usize, usize) {
        if self.dense_no_wp(palette) {
            MA_DENSE_SAMPLING
        } else {
            self.final_sampling
        }
    }

    /// Whether a source's final learns may use the Weighted Predictor.
    fn final_wp(&self, frame: &CoreFrame<'_>, palette: bool) -> bool {
        frame.use_wp && !self.dense_no_wp(palette)
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
    /// then ranks the WP presets of each surviving source.
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
        // hold one value per ink, so only the palette source is learned. A
        // two-color palette also stands alone: the RGB source never came
        // within the margin of it on 86 bi-level pages.
        let palette_only = (matches!(frame.kind, ModularFrameKind::XybBlend { .. })
            && palette.is_some())
            || palette.is_some_and(GlobalPalette::is_bilevel);
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

        let photo = |stage: &CoarseLearn| {
            stage.candidate.est_real
                >= DENSE_FINAL_MIN_BITS_PER_VALUE
                    * source_values_of(xsize, ysize, frame.coded_chans) as f64
        };
        let spare_wp = |source: &MaSource<'_>, wp_params: WpParams, scratch: &mut CoderScratch| {
            if !(effort.tree_dense_sampling && frame.use_wp) {
                return None;
            }
            dense_no_wp_final(
                source,
                xsize,
                ysize,
                frame.min_symbol,
                wp_params,
                rank_layout,
                rank_sampling,
                pool,
                scratch,
            )
        };

        // One coarse learn per source decides which are learned in full.
        let wp_params = self.wp_params;
        let mut rgb_spare_wp = None;
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
            let stage = coarse(&source, wp_params, scratch);
            if stage.as_ref().is_some_and(|stage| !photo(stage)) {
                rgb_spare_wp = spare_wp(&source, wp_params, scratch);
            }
            stage
        };
        let dense = effort.tree_dense_sampling && rgb_stage.as_ref().is_some_and(photo);
        let mut palette_spare_wp = None;
        // A two-color palette is learned whatever its coarse learn says:
        // nothing ranks against it, and without the preset ranking nothing
        // starts from that learn either (26 pages byte-identical, 5 % of
        // their encode).
        let unranked = !rank_presets_deep && palette.is_some_and(GlobalPalette::is_bilevel);
        let mut palette_stage = palette.filter(|_| !unranked).and_then(|p| {
            let source = p.source(xsize, ysize);
            let stage = coarse(&source, wp_params, scratch)?;
            // A two-color index keeps the predictor and the regular budget:
            // dense without it, 10 of 26 bi-level pages coded 0.4-5.3 %
            // larger and a scanned manga page 0.3 % smaller at 1.5 times
            // its encode.
            if !dense && !p.is_bilevel() {
                palette_spare_wp = spare_wp(&source, wp_params, scratch);
            }
            Some(stage)
        });
        let coarse_of = |stage: &Option<CoarseLearn>| {
            stage.as_ref().map_or(f64::INFINITY, |s| {
                s.candidate.est_real + header_bits(wp_params)
            })
        };
        let (rgb_coarse, palette_coarse) = (coarse_of(&rgb_stage), coarse_of(&palette_stage));
        if dense {
            self.final_sampling = MA_DENSE_SAMPLING;
        }
        // A source that needs the predictor keeps it for the other one too.
        if rgb_spare_wp != Some(false) && palette_spare_wp != Some(false) {
            self.rgb_dense_no_wp = rgb_spare_wp == Some(true);
            self.palette_dense_no_wp = palette_spare_wp == Some(true);
        }
        let large_palette = palette.is_some_and(GlobalPalette::is_large);
        self.rgb =
            large_palette || (!unranked && rgb_coarse <= palette_coarse * PALETTE_COARSE_MARGIN);
        let palette_margin = if large_palette {
            LARGE_PALETTE_COARSE_MARGIN
        } else {
            PALETTE_COARSE_MARGIN
        };
        self.palette =
            unranked || (palette_stage.is_some() && palette_coarse <= rgb_coarse * palette_margin);

        // Small images stop here: the preset ranking costs more than it finds.
        let rank_more = source_values_of(xsize, ysize, frame.coded_chans) >= RANK_MIN_VALUES;
        if self.rgb && rank_more {
            // The RGB source's WP presets.
            let (_, _, params, stage) = {
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
            // The estimator's first transform stands. Scoring its runner-up
            // with a ranking learn changed 12 of 74 photos, six each way
            // (Kodak +0.11%, the rest -0.03%), for 5-7% of the encode: a
            // 32-leaf tree favors transforms that decorrelate more and then
            // gain less from depth, and no estimate margin told its wins
            // from its losses.
            self.wp_params = params;
            rgb_stage = stage;
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
        let rank_sampling = self.rank_sampling;
        let reusable = |stage: Option<CoarseLearn>, final_layout: GroupLayout, budget| {
            stage.filter(|stage| {
                final_layout == layouts.rank_layout()
                    && budget == rank_sampling
                    && stage.samples.len() > MA_COARSE_TARGET_SAMPLES
            })
        };
        self.rgb_stage = reusable(
            self.rgb_stage.take(),
            layouts.rgb()[0],
            self.final_budget(false),
        );
        self.palette_stage = reusable(
            self.palette_stage.take(),
            layouts.all[0],
            self.final_budget(true),
        );
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
            sources.final_wp(frame, false),
            f64::INFINITY,
            f64::INFINITY,
            false,
            sources.final_budget(false),
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
            sources.final_wp(frame, false),
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
        raced: Option<&GlobalPalette>,
        layouts: &LayoutPlan,
        best: &mut SmallestFrame,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        let first_stage = sources.palette_stage.take();
        let source = palette.source(frame.xsize, frame.ysize);
        // A large palette learns the first layout only: the second changed
        // no file of thirteen and took a fifth of their encode time. So does
        // a two-color one, in each of its forms: over 56 bi-level pages the
        // second layout's estimate led in 4 of 91 learns, by 0.6-2.2 % and
        // in a form that lost to another, so no file changed.
        let palette_layouts = if palette.is_large() || palette.is_bilevel() {
            &layouts.all[..1]
        } else {
            &layouts.all[..]
        };
        // A frame of a few colors also codes its index image two pixels to a
        // sample; past four colors that form waits for the plain one's
        // estimate. A bi-level one holding a repeated screen codes
        // each form under a tree of Zero-predictor leaves as well, where LZ77
        // copies the screen's pixels. The smallest written frame stays.
        // The packed index image of a frame odd both ways lies in a frame
        // one column wider, cropped to the image by the decoder.
        let may_pad = matches!(frame.kind, ModularFrameKind::Regular) && frame.alpha.is_none();
        let packed = PackedPalette::new(palette, frame.xsize, frame.ysize, may_pad);
        let padded_frame = packed
            .as_ref()
            .and_then(PackedPalette::padded_xsize)
            .map(|xsize| CoreFrame {
                xsize,
                kind: ModularFrameKind::RgbCrop {
                    x0: 0,
                    y0: 0,
                    width: xsize,
                    height: frame.ysize,
                    full: true,
                    last: true,
                },
                ..*frame
            });
        let packed_frame = padded_frame.as_ref().unwrap_or(frame);
        let mut held: Option<(GroupLayout, LearnedCandidate)> = None;
        if packed.is_some() || palette.is_bilevel() {
            let screen = palette.is_bilevel()
                && palette.screen_share(frame.xsize, frame.ysize) >= palette::SCREEN_MIN_SHARE;
            let mut arms: Vec<MaSource<'_>> = Vec::with_capacity(3);
            if let Some(packed) = packed.as_ref().filter(|packed| !packed.races_coarse()) {
                arms.push(packed.source());
            }
            if screen {
                let mut values = palette.source(frame.xsize, frame.ysize);
                values.value_tree = true;
                arms.push(values);
                if let Some(packed) = &packed {
                    let mut values = packed.source();
                    values.value_tree = true;
                    arms.push(values);
                }
            }
            for arm in &arms {
                let frame = if arm.hshift + arm.vshift > 0 {
                    packed_frame
                } else {
                    frame
                };
                if let Some((layout, cand, _)) = self.learn_palette_layouts(
                    frame,
                    sources,
                    arm,
                    palette_layouts,
                    None,
                    f64::INFINITY,
                    pool,
                    scratch,
                ) {
                    // A two-color image writes its packed or its plain index
                    // image, whichever the final learn estimates smaller: at
                    // Slow the estimate named the smaller file on 25 of 26
                    // pages (the other by 21 bytes) and the second write was
                    // 15 % of their encode.
                    if palette.is_bilevel()
                        && !arm.value_tree
                        && raced.is_none()
                        && frame.effort.tree_dense_sampling
                    {
                        held = Some((layout, cand));
                        continue;
                    }
                    self.write_palette(
                        frame, color, sources, arm, layout, &cand, best, pool, scratch,
                    );
                }
            }
        }
        let Some(raced) = raced else {
            let plain = self.learn_palette_layouts(
                frame,
                sources,
                &source,
                palette_layouts,
                first_stage,
                f64::INFINITY,
                pool,
                scratch,
            );
            if let Some((_, _, coarse)) = &plain {
                self.learn_late_packed(
                    packed_frame,
                    color,
                    sources,
                    packed.as_ref(),
                    palette_layouts,
                    *coarse,
                    best,
                    pool,
                    scratch,
                );
            }
            let plain_est = plain
                .as_ref()
                .map_or(f64::INFINITY, |(_, cand, _)| cand.est_real);
            if let Some((held_layout, held_cand)) = held.take()
                && held_cand.est_real < plain_est
                && let Some(packed) = &packed
            {
                let packed_source = packed.source();
                self.write_palette(
                    packed_frame,
                    color,
                    sources,
                    &packed_source,
                    held_layout,
                    &held_cand,
                    best,
                    pool,
                    scratch,
                );
                if let Some((_, _, coarse)) = plain {
                    self.best_source_coarse = self.best_source_coarse.min(coarse);
                }
                return;
            }
            if let Some((layout, cand, coarse)) = plain {
                let before = best.bits();
                self.write_palette(
                    frame, color, sources, &source, layout, &cand, best, pool, scratch,
                );
                // The per-group palettes still race a large palette whose
                // frame leads the RGB one by little: its coarse learn pruned
                // them on a screenshot they coded 0.3 % smaller. Past that
                // lead (10-30 % on seven frames) they only cost time.
                let narrow = palette.is_large()
                    && before.zip(best.bits()).is_some_and(|(before, after)| {
                        after as f64 > before as f64 * LARGE_PALETTE_CLEAR_LEAD
                    });
                if !narrow {
                    self.best_source_coarse = self.best_source_coarse.min(coarse);
                }
            }
            return;
        };
        // The order race: both orders learn the first layout, and an order
        // whose complete estimate there trails the other's by more than
        // `ORDER_RACE_MARGIN` stops; the rest learn the other layouts and
        // are written.
        let raced_source = raced.source(frame.xsize, frame.ysize);
        let (head, tail) = palette_layouts.split_at(1);
        let mut learns = [(&source, first_stage), (&raced_source, None)].map(|(source, stage)| {
            let learn = self.learn_palette_layouts(
                frame,
                sources,
                source,
                head,
                stage,
                f64::INFINITY,
                pool,
                scratch,
            );
            (source, learn)
        });
        let plain_coarse = learns
            .iter()
            .filter_map(|(_, learn)| learn.as_ref().map(|learn| learn.2))
            .fold(f64::INFINITY, f64::min);
        self.learn_late_packed(
            packed_frame,
            color,
            sources,
            packed.as_ref(),
            palette_layouts,
            plain_coarse,
            best,
            pool,
            scratch,
        );
        let est_of = |learn: &Option<(GroupLayout, LearnedCandidate, f64)>| {
            learn
                .as_ref()
                .map_or(f64::INFINITY, |(_, cand, _)| cand.est_real)
        };
        let lead = est_of(&learns[0].1).min(est_of(&learns[1].1));
        // Estimates of a few kilobytes miss by more than the margin, and
        // such frames cost nothing to write twice.
        let limit = if lead <= ORDER_RACE_BOTH_MAX_BITS {
            f64::INFINITY
        } else {
            lead * ORDER_RACE_MARGIN
        };
        let final_limit = self.rgb_final * PALETTE_FINAL_MARGIN * PALETTE_LAYOUT_MARGIN;
        for (source, learn) in learns.iter_mut() {
            let Some((mut layout, mut cand, mut coarse)) = learn.take() else {
                continue;
            };
            if cand.est_real > limit {
                self.best_source_coarse = self.best_source_coarse.min(coarse);
                continue;
            }
            if !tail.is_empty()
                && cand.est_real <= final_limit
                && let Some((tail_layout, tail_cand, tail_coarse)) = self.learn_palette_layouts(
                    frame,
                    sources,
                    source,
                    tail,
                    None,
                    f64::INFINITY,
                    pool,
                    scratch,
                )
            {
                coarse = coarse.min(tail_coarse);
                if tail_cand.est_real < cand.est_real {
                    (layout, cand) = (tail_layout, tail_cand);
                }
            }
            self.write_palette(
                frame, color, sources, source, layout, &cand, best, pool, scratch,
            );
            self.best_source_coarse = self.best_source_coarse.min(coarse);
        }
    }

    /// The packed index image of a palette past four colors: learned once
    /// the plain index image's first-stage estimate is known, dropped when
    /// its own trails that by more than [`PACKED_COARSE_MARGIN`], and
    /// written otherwise.
    #[allow(clippy::too_many_arguments)]
    fn learn_late_packed(
        &mut self,
        frame: &CoreFrame<'_>,
        color: &mut ColorTransform<'_>,
        sources: &SourcePlan,
        packed: Option<&PackedPalette<'_>>,
        layouts: &[GroupLayout],
        plain_coarse: f64,
        best: &mut SmallestFrame,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        let Some(packed) = packed.filter(|packed| packed.races_coarse()) else {
            return;
        };
        if !plain_coarse.is_finite() {
            return;
        }
        let arm = packed.source();
        if let Some((layout, cand, _)) = self.learn_palette_layouts(
            frame,
            sources,
            &arm,
            layouts,
            None,
            plain_coarse * PACKED_COARSE_MARGIN,
            pool,
            scratch,
        ) {
            self.write_palette(
                frame, color, sources, &arm, layout, &cand, best, pool, scratch,
            );
        }
    }

    /// The palette source's learn under `layouts`, pruned against the RGB
    /// tree's estimates and against `coarse_cap`.
    #[allow(clippy::too_many_arguments)]
    fn learn_palette_layouts(
        &self,
        frame: &CoreFrame<'_>,
        sources: &SourcePlan,
        source: &MaSource<'_>,
        layouts: &[GroupLayout],
        first_stage: Option<CoarseLearn>,
        coarse_cap: f64,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Option<(GroupLayout, LearnedCandidate, f64)> {
        learn_best_layout(
            frame.settings.speed,
            source,
            frame.xsize,
            frame.ysize,
            layouts,
            frame.min_symbol,
            sources.palette_wp,
            sources.final_wp(frame, true),
            (self.rgb_coarse * PALETTE_COARSE_MARGIN).min(coarse_cap),
            // Room for the 256 layout's ~4% on palette graphics.
            self.rgb_final * PALETTE_FINAL_MARGIN * PALETTE_LAYOUT_MARGIN,
            true,
            sources.final_budget(true),
            first_stage,
            pool,
            scratch,
        )
    }

    /// Writes the learned palette frame and offers it to `best`. The frame
    /// write (tokenize, code, sections) is the palette candidate's most
    /// expensive step after the learn; a final estimate clearly behind the
    /// RGB frame's cannot win the size comparison.
    #[allow(clippy::too_many_arguments)]
    fn write_palette(
        &mut self,
        frame: &CoreFrame<'_>,
        color: &mut ColorTransform<'_>,
        sources: &SourcePlan,
        source: &MaSource<'_>,
        layout: GroupLayout,
        cand: &LearnedCandidate,
        best: &mut SmallestFrame,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) {
        if cand.est_real > self.rgb_final * PALETTE_FINAL_MARGIN {
            return;
        }
        let mut candidate = BitWriter::new();
        let estimated_savings = write_learned_tree_frame(
            frame.settings.speed,
            source,
            frame.alpha.is_some(),
            frame.kind,
            frame.xsize,
            frame.ysize,
            layout,
            frame.min_symbol,
            pool,
            scratch,
            sources.palette_wp,
            sources.final_wp(frame, true),
            cand,
            frame.tree_search,
            &mut candidate,
        );
        if learned_tree_is_decisive(estimated_savings) {
            self.decisive = true;
            color.release_planes();
        }
        best.offer(candidate);
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
        let pixels = frame.xsize * frame.ysize;
        if covered * 2 < pixels {
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
                sources.final_wp(frame, false),
                mixed.max_leaves(),
                sources.final_budget(false),
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
            sources.final_wp(frame, false),
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
            sources.final_wp(frame, false),
            &cand,
            frame.tree_search,
            &mut candidate,
        );
        if learned_tree_is_decisive(estimated_savings) {
            self.decisive = true;
            color.release_planes();
        }
        let palettised = covered as f64 >= GROUP_PALETTE_DENSE_COVERAGE * pixels as f64;
        let credit = if sources.rgb_dense_no_wp && palettised {
            GROUP_PALETTE_DENSE_CREDIT
        } else {
            GROUP_PALETTE_CREDIT
        };
        best.offer_credited(candidate, credit);
    }
}
