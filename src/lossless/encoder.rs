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

//! Lossless transform and coding candidate selection.

use super::context::{
    try_encode_context_tree_multi_group, try_encode_context_tree_single_group, wp_context_tree,
    write_flat_tree,
};
use super::frame::keep_smaller_writer;
use super::groups::{DeepLzScratchPool, group_lz_threads, write_lz_groups};
use super::header::{
    write_frame_header_modular_kind, write_modular_transforms, write_patch_dictionary,
    write_toc_entry,
};
use super::lz77::{
    ConstantTail, LzToken, build_lz_pixel_code_tails, lz77_compress_channels_for_speed,
    lz77_compress_channels_for_speed_with_depth, write_lz_section, write_lz_section_tail,
};
use super::ma_encoding::write_learned_tree_frame;
use super::ma_learning::{
    RANK_COARSE_PRUNE, RCT_RANK_MARGIN, finish_ma_learn, learn_best_layout, learn_ma_candidate,
    rank_bits, rank_coarse, rank_presets,
};
use super::ma_source::{
    CoarseLearn, DENSE_FINAL_MIN_BITS_PER_VALUE, MA_COARSE_TARGET_SAMPLES, MA_DENSE_SAMPLING,
    MA_FULL_SAMPLING, MA_LITE_SAMPLING, PALETTE_LAYOUT_MARGIN, PER_GROUP_MARGIN, RANK_MIN_VALUES,
    RGB_SINGLE_LAYOUT_MIN_VALUES, mixed_ma_source, rgb_ma_source, source_values_of,
};
use super::palette::{
    PALETTE_COARSE_MARGIN, PALETTE_FINAL_MARGIN, build_global_palette,
    try_encode_local_palette_multi_group, try_encode_palette_single_group,
};
use super::predictor::{
    PREDICTOR_WEIGHTED, channel_to_context, choose_predictors_with_wp, choose_wp_params,
    fixed_predictor, selected_grad_pack_interior_fn, tokenize_channels_with_wp,
    tokenize_runs_with_wp,
};
use super::rct::{rank_rcts, rct_planes as rct_planes_fn};
use super::squeeze::{encode_squeeze_multigroup, encode_squeeze_single_group};
use super::{
    GROUP_DIM, GroupLayout, LF_GROUP_DIM, constant_alpha_value, lossless_min_symbol, palette,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpParams, write_wp_header};

/// The learned-tree entropy proxy must beat its flat alternative decisively
/// before Slow mode trusts it enough to omit the context/flat final encodes.
/// Near decisions retain the exact byte-level tournament.
pub(super) const MA_DECISIVE_MIN_SAVINGS: f64 = 0.18;

/// Row steps of the RCT ranking under `rct_dense_rows` and below it.
const RCT_DENSE_ROW_STEP: usize = 4;

const RCT_SPARSE_ROW_STEP: usize = 16;

/// A learned tree estimated to save at least this fraction over the flat
/// coder also skips the v1 context-tree and flat frame candidates.
const MA_ALTERNATIVES_MIN_SAVINGS: f64 = 0.05;

#[inline]
pub(super) fn learned_tree_is_decisive(estimated_savings: f64) -> bool {
    estimated_savings >= MA_DECISIVE_MIN_SAVINGS
}

/// The color planes under `rct_type`: the YCoCg (or gray) input itself, or
/// its transform, built on first use and kept in `planes`.
fn rct_planes_of<'a>(
    planes: &'a mut Option<Image3Si>,
    ycocg: &'a Image3Si,
    num_color: usize,
    rct_type: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> &'a Image3Si {
    if num_color != 3 || rct_type == 6 {
        return ycocg;
    }
    planes.get_or_insert_with(|| {
        rct_planes_fn(ycocg, ycocg.xsize(), ycocg.ysize(), rct_type, pool, scratch)
    })
}

pub(crate) fn encode_frame_lossless_core(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    num_color: usize,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    upsampling: u32,
    near_lossless: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    frame_kind: ModularFrameKind<'_>,
    local_trees: bool,
    writer: &mut BitWriter,
) -> bool {
    encode_frame_lossless_core_impl(
        linear,
        alpha,
        max_bits,
        progressive,
        num_color,
        speed,
        decoding_speed,
        upsampling,
        near_lossless,
        pool,
        scratch,
        frame_kind,
        true,
        local_trees,
        writer,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_frame_lossless_core_impl(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    num_color: usize,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    upsampling: u32,
    near_lossless: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    frame_kind: ModularFrameKind<'_>,
    allow_palette_candidates: bool,
    local_trees: bool,
    writer: &mut BitWriter,
) -> bool {
    let effort = speed.effort();
    // Near-lossless samples share a `2k+1` grid. The Weighted predictor's
    // averaging leaves it, so the tiers without a learned tree keep their
    // fixed predictor on the grid with Gradient instead.
    let use_wp = decoding_speed.use_weighted_predictor() && (effort.learned_tree || !near_lossless);
    // The hybrid-uint token of a modular residual can reach ~4*B+11 for B-bit
    // input. For B<=13 this stays below 64, so keep the tight default; for higher
    // bit depths raise LZ77_MIN_SYMBOL above the max literal token so the decoder
    // never mistakes a large residual for an LZ77 back-reference (the cause of
    // the old ~12-bit lossless ceiling). Stays under the 128-symbol prefix-code
    // alphabet through ~18-bit.
    let min_symbol = lossless_min_symbol(max_bits);
    let xsize = linear.xsize();
    let ysize = linear.ysize();
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };
    // A solid alpha plane is coded as a constant channel (Zero predictor,
    // offset = the value, one-symbol histogram): no bits, no predictor and
    // no context-tree work in the decoder, which otherwise ran the Weighted
    // Predictor over the whole plane.
    let alpha_constant = alpha.and_then(constant_alpha_value);
    // Channels that carry samples: size gates treat a solid-alpha image as
    // its color image.
    let coded_chans = num_color + usize::from(alpha.is_some() && alpha_constant.is_none());

    let xsize_groups = xsize.div_ceil(GROUP_DIM);
    let ysize_groups = ysize.div_ceil(GROUP_DIM);
    let num_ac_groups = xsize_groups * ysize_groups;
    let xsize_dc_groups = xsize.div_ceil(LF_GROUP_DIM);
    let ysize_dc_groups = ysize.div_ceil(LF_GROUP_DIM);
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
    let single_group = num_ac_groups == 1;
    let grad_pack_fn = selected_grad_pack_interior_fn();

    // Palette (v2): single-group, RGB/RGBA, <=256 distinct colors. Encodes
    // the image as a small palette meta-channel + an index channel, which is a
    // large win for low-color/graphic content. Falls through to the normal
    // RCT+predictor path when it doesn't apply.
    if allow_palette_candidates && frame_kind.is_regular() && single_group && num_color == 3 {
        let mut palette_writer = BitWriter::new();
        if try_encode_palette_single_group(
            linear,
            alpha,
            xsize,
            ysize,
            min_symbol,
            grad_pack_fn,
            speed,
            use_wp,
            frame_kind,
            GroupLayout::DEFAULT.with_upsampling(upsampling),
            scratch,
            &mut palette_writer,
        ) {
            if speed.effort().palette_dual {
                let mut normal_writer = BitWriter::new();
                let normal_decisive = encode_frame_lossless_core_impl(
                    linear,
                    alpha,
                    max_bits,
                    progressive,
                    num_color,
                    speed,
                    decoding_speed,
                    upsampling,
                    near_lossless,
                    pool,
                    scratch,
                    frame_kind,
                    false,
                    local_trees,
                    &mut normal_writer,
                );
                if normal_writer.bits_written() < palette_writer.bits_written() {
                    writer.append(&normal_writer);
                    return normal_decisive;
                }
                writer.append(&palette_writer);
            } else {
                writer.append(&palette_writer);
            }
            return false;
        }
    }

    // Large graphics often exceed 256 colors globally while each 256x256 group
    // remains low-color. Encode qualifying RGB/RGBA groups through an exact
    // local Palette transform and leave high-color groups on the normal path.
    if allow_palette_candidates && frame_kind.is_regular() && !single_group && num_color == 3 {
        let mut palette_writer = BitWriter::new();
        if try_encode_local_palette_multi_group(
            linear,
            alpha,
            xsize,
            ysize,
            xsize_groups,
            ysize_groups,
            num_dc_groups,
            min_symbol,
            grad_pack_fn,
            speed,
            use_wp,
            frame_kind,
            GroupLayout::DEFAULT.with_upsampling(upsampling),
            pool,
            scratch,
            &mut palette_writer,
        ) {
            if speed.effort().palette_dual {
                let mut normal_writer = BitWriter::new();
                let normal_decisive = encode_frame_lossless_core_impl(
                    linear,
                    alpha,
                    max_bits,
                    progressive,
                    num_color,
                    speed,
                    decoding_speed,
                    upsampling,
                    near_lossless,
                    pool,
                    scratch,
                    frame_kind,
                    false,
                    local_trees,
                    &mut normal_writer,
                );
                if normal_writer.bits_written() < palette_writer.bits_written() {
                    writer.append(&normal_writer);
                    return normal_decisive;
                }
                writer.append(&palette_writer);
            } else {
                writer.append(&palette_writer);
            }
            return false;
        }
    }

    // Stage-2 progressive lossless (opt-in): Squeeze pyramid (RGB + optional alpha).
    if frame_kind.is_regular() && single_group && num_color == 3 && progressive {
        encode_squeeze_single_group(
            linear,
            alpha,
            xsize,
            ysize,
            min_symbol,
            grad_pack_fn,
            speed,
            use_wp,
            frame_kind,
            GroupLayout::DEFAULT.with_upsampling(upsampling),
            pool,
            scratch,
            writer,
        );
        return false;
    }

    // Progressive multi-group (Stage A: all squeezed channels fit the global
    // stream). Falls through to the non-progressive path if a channel is still
    // larger than a group (Stage B).
    if frame_kind.is_regular()
        && !single_group
        && num_color == 3
        && progressive
        && encode_squeeze_multigroup(
            linear,
            alpha,
            xsize,
            ysize,
            min_symbol,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_dc_groups,
            num_ac_groups,
            grad_pack_fn,
            speed,
            use_wp,
            frame_kind,
            GroupLayout::DEFAULT.with_upsampling(upsampling),
            pool,
            scratch,
            writer,
        )
    {
        return false;
    }

    // Fast uses fixed Weighted prediction and skips all adaptive analysis. Slow
    // searches the supported Slow predictor set per channel by estimated cost
    // (one global choice, used for the tree and every group).

    // RCT re-selection (Slow, regular RGB frames): estimated over the same
    // fixed-context gradient cost libjxl uses. When a non-YCoCg transform
    // wins, the color planes are rebuilt and the frame headers signal it.
    // A single gray plane has no RCT. Near-lossless planes rank at every
    // tier: their values share a `2k+1` grid that only the subtract-type
    // transforms preserve, and YCoCg's shifts would scatter it.
    let mut rct_type = if num_color == 3 { 6u32 } else { 0 };
    let mut rct_planes: Option<Image3Si> = None;
    let rct_ranked: Vec<(u32, f32)> = if frame_kind.allows_learned_tree()
        && (effort.rct_search || near_lossless)
        && num_color == 3
    {
        let row_step = if effort.rct_every_row {
            1
        } else if effort.rct_dense_rows {
            RCT_DENSE_ROW_STEP
        } else {
            RCT_SPARSE_ROW_STEP
        };
        rank_rcts(linear, xsize, ysize, row_step, pool, scratch)
    } else {
        Vec::new()
    };
    if let Some(&(t, _)) = rct_ranked.first()
        && t != 6
    {
        rct_type = t;
    }
    // The palette candidate below needs the untransformed YCoCg planes. The
    // transformed ones are built on first use: an ink frame's palette source
    // usually wins without them.
    let ycocg = linear;

    let wp_header_count = if single_group {
        1
    } else {
        2 + num_dc_groups + num_ac_groups
    };
    let mut wp_params = if effort.wp_search && use_wp {
        choose_wp_params(
            ycocg,
            rct_type,
            alpha,
            num_color,
            wp_header_count,
            pool,
            scratch,
        )
    } else {
        WpParams::DEFAULT
    };
    // The estimates below remain useful prefilters, but Slow mode keeps every
    // viable finalist and compares its complete byte-aligned frame against the
    // v1 and flat alternatives. Tree headers, clustering, hybrid uint choices,
    // and LZ77 can otherwise reverse the order predicted from residual entropy.
    let compare_tree_candidates = frame_kind.allows_learned_tree()
        && effort.learned_tree
        && (num_color == 3 || num_color == 1)
        && decoding_speed.use_ma_trees();
    let mut best_tree_writer: Option<BitWriter> = None;

    // Sources: the RGB(A) planes under the best transform and, for frames
    // with <= 1024 colors, a global palette index image. The corner-sampled
    // WP proxy and the gradient RCT estimator both mis-rank on a third of
    // so a reduced-leaf learner ranks every WP preset per source (and the transform runner-up for RGB).
    // First the two sources are compared by one first-stage learn each under the proxy
    // preset.
    let (layout, learned_tree_decisive, skip_alternatives) = {
        let palette = if compare_tree_candidates && effort.tree_global_palette {
            build_global_palette(ycocg, alpha, num_color, xsize, ysize, pool, scratch)
        } else {
            None
        };
        let header_bits = |params: WpParams| {
            if params == WpParams::DEFAULT {
                0.0
            } else {
                (51 * wp_header_count) as f64
            }
        };
        let mut rank_rgb = compare_tree_candidates;
        // Final learned estimate of the RGB frame; gates the palette write.
        let rgb_final_est: f64;
        // Coarse estimate of the best source so far: the bar a further
        // candidate must clear before it is grown into a complete frame.
        let mut best_source_coarse;
        let mut rank_pal = compare_tree_candidates && palette.is_some();
        // Group layout. With every decoding tool available, Slow lossless runs
        // the first learning stage (sampling plus a 32-leaf tree) under both
        // 1024- and 256-pixel groups.
        // Layouts are processed one at a time so a single sample set is alive.
        // Everything else keeps the codestream default.
        let slow_layouts = compare_tree_candidates
            && effort.tree_dual_layout
            && matches!(decoding_speed, crate::DecodingSpeed::Slow)
            // An image inside one 256-px group learns the same thing under
            // both layouts.
            && (xsize > GroupLayout::DEFAULT.dim() || ysize > GroupLayout::DEFAULT.dim());
        let layouts: Vec<GroupLayout> = if slow_layouts {
            vec![GroupLayout::LARGE, GroupLayout::DEFAULT]
        } else {
            vec![GroupLayout::DEFAULT]
        }
        .into_iter()
        .map(|layout| layout.with_upsampling(upsampling))
        .collect();
        let layouts: &[GroupLayout] = &layouts;
        // RGB sources of more than a megapixel-channel learn the 1024-px
        // layout only: the 256 layout's first stage costs a sampling pass and
        // its full learn doubles the frame's learning time.
        let rgb_layouts: &[GroupLayout] = if slow_layouts
            && source_values_of(xsize, ysize, coded_chans) > RGB_SINGLE_LAYOUT_MIN_VALUES
        {
            &layouts[..1]
        } else {
            layouts
        };
        // The ranking learns run under the first layout.
        let rank_layout = layouts[0];
        // An ink frame is a palette image by construction: its color planes
        // hold one value per ink, so only the palette source is learned.
        let palette_only =
            matches!(frame_kind, ModularFrameKind::XybBlend { .. }) && palette.is_some();
        let mut final_sampling = if effort.tree_full_sampling {
            MA_FULL_SAMPLING
        } else {
            MA_LITE_SAMPLING
        };
        let rank_sampling = final_sampling;
        let mut pal_wp = wp_params;
        // Ranking stages kept for the final learns (`tree_fresh_final_samples`).
        let keep_stages = !effort.tree_fresh_final_samples;
        let mut rgb_rank_stage: Option<CoarseLearn> = None;
        let mut pal_rank_stage: Option<CoarseLearn> = None;
        if compare_tree_candidates
            && effort.tree_rank
            && matches!(decoding_speed, crate::DecodingSpeed::Slow)
        {
            let mut rgb_stage = if palette_only {
                None
            } else {
                let linear =
                    rct_planes_of(&mut rct_planes, ycocg, num_color, rct_type, pool, scratch);
                let source =
                    rgb_ma_source(linear, alpha, xsize, rct_type, alpha_constant, num_color);
                rank_coarse(
                    &source,
                    xsize,
                    ysize,
                    min_symbol,
                    wp_params,
                    use_wp,
                    rank_layout,
                    rank_sampling,
                    pool,
                    scratch,
                )
            };
            let mut pal_stage = palette.as_ref().and_then(|p| {
                let source = p.source(xsize, ysize);
                rank_coarse(
                    &source,
                    xsize,
                    ysize,
                    min_symbol,
                    wp_params,
                    use_wp,
                    rank_layout,
                    rank_sampling,
                    pool,
                    scratch,
                )
            });
            let coarse_of = |stage: &Option<CoarseLearn>| {
                stage.as_ref().map_or(f64::INFINITY, |s| {
                    s.candidate.est_real + header_bits(wp_params)
                })
            };
            let (rgb_coarse, pal_coarse) = (coarse_of(&rgb_stage), coarse_of(&pal_stage));
            if effort.tree_dense_sampling
                && rgb_stage.as_ref().is_some_and(|stage| {
                    stage.candidate.est_real
                        >= DENSE_FINAL_MIN_BITS_PER_VALUE
                            * source_values_of(xsize, ysize, coded_chans) as f64
                })
            {
                final_sampling = MA_DENSE_SAMPLING;
            }
            rank_rgb = rgb_coarse <= pal_coarse * PALETTE_COARSE_MARGIN;
            rank_pal = pal_stage.is_some() && pal_coarse <= rgb_coarse * PALETTE_COARSE_MARGIN;

            let rank_more = source_values_of(xsize, ysize, coded_chans) >= RANK_MIN_VALUES;
            if rank_rgb && rank_more {
                let (mut best_bits, mut best_coarse, params, stage) = {
                    let linear =
                        rct_planes_of(&mut rct_planes, ycocg, num_color, rct_type, pool, scratch);
                    let source =
                        rgb_ma_source(linear, alpha, xsize, rct_type, alpha_constant, num_color);
                    rank_presets(
                        &source,
                        xsize,
                        ysize,
                        min_symbol,
                        use_wp,
                        rgb_stage.take(),
                        wp_params,
                        &header_bits,
                        effort.tree_rank_deep(),
                        rank_layout,
                        rank_sampling,
                        keep_stages,
                        pool,
                        scratch,
                    )
                };
                wp_params = params;
                rgb_stage = stage;
                // Transform runner-up with the chosen preset, same coarse prune.
                if let Some(&(runner_up, cost)) = rct_ranked.get(1)
                    && cost <= rct_ranked[0].1 * RCT_RANK_MARGIN
                {
                    // One RGB sample set alive at a time: a losing runner-up
                    // leaves the final learn to sample afresh.
                    rgb_stage = None;
                    let planes = (runner_up != 6)
                        .then(|| rct_planes_fn(ycocg, xsize, ysize, runner_up, pool, scratch));
                    let source = rgb_ma_source(
                        planes.as_ref().unwrap_or(ycocg),
                        alpha,
                        xsize,
                        runner_up,
                        alpha_constant,
                        num_color,
                    );
                    if let Some(stage) = rank_coarse(
                        &source,
                        xsize,
                        ysize,
                        min_symbol,
                        wp_params,
                        use_wp,
                        rank_layout,
                        rank_sampling,
                        pool,
                        scratch,
                    ) && stage.candidate.est_real + header_bits(wp_params)
                        <= best_coarse * RANK_COARSE_PRUNE
                    {
                        best_coarse =
                            best_coarse.min(stage.candidate.est_real + header_bits(wp_params));
                        let bits = rank_bits(&stage) + header_bits(wp_params);
                        if bits < best_bits {
                            best_bits = bits;
                            rct_type = runner_up;
                            rct_planes = planes;
                            rgb_stage = keep_stages.then_some(stage);
                        }
                    }
                }
                let _ = (best_bits, best_coarse);
            }
            if rank_pal
                && rank_more
                && effort.tree_rank_deep()
                && let Some(p) = &palette
            {
                let source = p.source(xsize, ysize);
                let (_, _, params, stage) = rank_presets(
                    &source,
                    xsize,
                    ysize,
                    min_symbol,
                    use_wp,
                    pal_stage.take(),
                    wp_params,
                    &header_bits,
                    true,
                    rank_layout,
                    rank_sampling,
                    keep_stages,
                    pool,
                    scratch,
                );
                pal_wp = params;
                pal_stage = stage;
            }
            if keep_stages {
                (rgb_rank_stage, pal_rank_stage) = (rgb_stage, pal_stage);
            }
        }
        let linear = if rank_rgb {
            rct_planes_of(&mut rct_planes, ycocg, num_color, rct_type, pool, scratch)
        } else {
            rct_planes.as_ref().unwrap_or(ycocg)
        };

        // A ranking stage stands in for a final learn's first stage when both
        // sampled the same source under the same layout and budget, and the
        // ranking did not stop at its direct small-set learn.
        let reusable = |stage: Option<CoarseLearn>, final_layout: GroupLayout| {
            stage.filter(|stage| {
                final_layout == rank_layout
                    && final_sampling == rank_sampling
                    && stage.samples.len() > MA_COARSE_TARGET_SAMPLES
            })
        };
        let mut rgb_rank_stage = reusable(rgb_rank_stage, rgb_layouts[0]);
        let mut pal_rank_stage = reusable(pal_rank_stage, layouts[0]);
        let (layout, rgb_coarse_est, mut learned_tree_decisive, skip_alternatives) = {
            let rgb_source =
                rgb_ma_source(linear, alpha, xsize, rct_type, alpha_constant, num_color);
            let mut learned_estimated_savings: Option<f64> = None;
            let learned = if rank_rgb {
                learn_best_layout(
                    speed,
                    &rgb_source,
                    xsize,
                    ysize,
                    rgb_layouts,
                    min_symbol,
                    wp_params,
                    use_wp,
                    f64::INFINITY,
                    f64::INFINITY,
                    false,
                    final_sampling,
                    rgb_rank_stage.take(),
                    pool,
                    scratch,
                )
            } else {
                None
            };
            let rgb_coarse_est = learned.as_ref().map_or(f64::INFINITY, |(_, _, e)| *e);
            best_source_coarse = rgb_coarse_est;
            rgb_final_est = learned
                .as_ref()
                .map_or(f64::INFINITY, |(_, c, _)| c.est_real);
            let layout = learned.as_ref().map_or(layouts[0], |(l, _, _)| *l);

            // Learned MA context tree (v2): greedy learned tree over the standard
            // property vector with per-leaf predictors.
            if let Some((_, cand, _)) = &learned {
                let mut candidate = BitWriter::new();
                let estimated_savings = write_learned_tree_frame(
                    speed,
                    &rgb_source,
                    alpha.is_some(),
                    frame_kind,
                    xsize,
                    ysize,
                    layout,
                    min_symbol,
                    pool,
                    scratch,
                    wp_params,
                    use_wp,
                    cand,
                    local_trees,
                    &mut candidate,
                );
                learned_estimated_savings = Some(estimated_savings);
                keep_smaller_writer(&mut best_tree_writer, candidate);
            }
            (
                layout,
                rgb_coarse_est,
                learned_estimated_savings.is_some_and(learned_tree_is_decisive),
                learned_estimated_savings.is_some_and(|s| s >= MA_ALTERNATIVES_MIN_SAVINGS),
            )
        };
        if learned_tree_decisive {
            // Only the original YCoCg planes are needed for the remaining palette
            // candidate. A decisive learned tree never returns to the RCT-based
            // v1/flat paths, so release those full-image planes before palette work.
            rct_planes = None;
        }

        // Global palette (<= 1024 colors) with its own learned tree, WP preset,
        // layout and literal/LZ77 choice over the index image. libjxl's e7
        // lossless gains ~10% from it on palette graphics; the complete frame
        // competes on size, and a decisive palette tree ends the search like a
        // decisive RGB tree does.
        let bytes_before_palette = best_tree_writer.as_ref().map(BitWriter::bits_written);
        if rank_pal && let Some(palette) = &palette {
            let palette_source = palette.source(xsize, ysize);
            let coarse_limit = rgb_coarse_est * PALETTE_COARSE_MARGIN;
            if let Some((palette_layout, cand, palette_coarse)) = learn_best_layout(
                speed,
                &palette_source,
                xsize,
                ysize,
                layouts,
                min_symbol,
                pal_wp,
                use_wp,
                coarse_limit,
                // Room for the 256 layout's ~4% on palette graphics.
                rgb_final_est * PALETTE_FINAL_MARGIN * PALETTE_LAYOUT_MARGIN,
                true,
                final_sampling,
                pal_rank_stage.take(),
                pool,
                scratch,
            ) {
                let palette_can_win = cand.est_real <= rgb_final_est * PALETTE_FINAL_MARGIN;
                // The frame write (tokenize, code, sections) is the palette candidate's
                // most expensive step after the learn; a final estimate clearly behind
                // the RGB frame's cannot win the size comparison.
                if palette_can_win {
                    let mut candidate = BitWriter::new();
                    let estimated_savings = write_learned_tree_frame(
                        speed,
                        &palette_source,
                        alpha.is_some(),
                        frame_kind,
                        xsize,
                        ysize,
                        palette_layout,
                        min_symbol,
                        pool,
                        scratch,
                        pal_wp,
                        use_wp,
                        &cand,
                        local_trees,
                        &mut candidate,
                    );
                    if learned_tree_is_decisive(estimated_savings) {
                        learned_tree_decisive = true;
                        rct_planes = None;
                    }
                    keep_smaller_writer(&mut best_tree_writer, candidate);
                }
                best_source_coarse = best_source_coarse.min(palette_coarse);
            }
        }
        // Per-group palettes: screen content is a mosaic of low-color
        // regions (a UI screen: 9 of 15 groups hold 221 to 870 colors) that
        // no global palette can capture. Each qualifying group declares its
        // own Palette transform while one learned tree still codes the whole
        // frame, telling the regimes apart through the stream-id property.
        // A global palette that already won leaves nothing for per-group
        // ones, and building them would scan the image for nothing.
        let global_palette_won =
            best_tree_writer.as_ref().map(BitWriter::bits_written) != bytes_before_palette;
        if compare_tree_candidates
            && effort.tree_group_palette
            && num_color == 3
            && !global_palette_won
        {
            let gdim = GroupLayout::DEFAULT.dim();
            let groups_x = xsize.div_ceil(gdim);
            let groups_y = ysize.div_ceil(gdim);
            let groups = groups_x * groups_y;
            if groups > 1 {
                let palettes: Vec<Option<palette::LocalPaletteGroup>> =
                    pool.steal_map(scratch, groups, |group_index, _scratch| {
                        let x0 = (group_index % groups_x) * gdim;
                        let y0 = (group_index / groups_x) * gdim;
                        palette::build_local_palette_group_limited(
                            ycocg,
                            alpha,
                            xsize,
                            x0,
                            y0,
                            gdim.min(xsize - x0),
                            gdim.min(ysize - y0),
                            palette::LOCAL_PALETTE_MAX_COLORS,
                        )
                    });
                let covered: usize = palettes
                    .iter()
                    .flatten()
                    .map(|palette| palette.w * palette.h)
                    .sum();
                // The groups that keep their color planes cost the same
                // either way, so the candidate is only worth its sampling
                // and first-stage learn when most of the frame palettises
                // (a UI screen: 60%; a screen-content photo mosaic: 44%,
                // where the complete frame then loses anyway).
                if covered * 2 >= xsize * ysize {
                    let mixed = mixed_ma_source(ycocg, alpha, xsize, &palettes, alpha_constant);
                    let stage = learn_ma_candidate(
                        &mixed,
                        xsize,
                        ysize,
                        GroupLayout::DEFAULT.with_upsampling(upsampling),
                        min_symbol,
                        wp_params,
                        use_wp,
                        mixed.max_leaves(),
                        final_sampling,
                        pool,
                        scratch,
                    );
                    // Only a first stage that already beats the best source
                    // earns the deepening and the complete frame: a global
                    // palette usually leaves nothing for per-group ones.
                    // The first stage is grown into a frame only when it is
                    // close to the best source's own first stage (a winning
                    // per-group plan lands at 0.75, the losers at 1.08 and
                    // above; the margin covers a screen-content tie).
                    if let Some(stage) = stage
                        .filter(|s| s.candidate.est_real < best_source_coarse * PER_GROUP_MARGIN)
                    {
                        let cand = finish_ma_learn(
                            speed,
                            stage,
                            min_symbol,
                            use_wp,
                            mixed.max_leaves(),
                            pool,
                            scratch,
                        );
                        let mut candidate = BitWriter::new();
                        let estimated_savings = write_learned_tree_frame(
                            speed,
                            &mixed,
                            alpha.is_some(),
                            frame_kind,
                            xsize,
                            ysize,
                            GroupLayout::DEFAULT.with_upsampling(upsampling),
                            min_symbol,
                            pool,
                            scratch,
                            wp_params,
                            use_wp,
                            &cand,
                            local_trees,
                            &mut candidate,
                        );
                        if learned_tree_is_decisive(estimated_savings) {
                            learned_tree_decisive = true;
                            rct_planes = None;
                        }
                        keep_smaller_writer(&mut best_tree_writer, candidate);
                    }
                }
            }
        }
        (layout, learned_tree_decisive, skip_alternatives)
    };

    if learned_tree_decisive || skip_alternatives {
        writer.append(
            best_tree_writer
                .as_ref()
                .expect("decisive learned-tree candidate"),
        );
        return learned_tree_decisive;
    }

    let gdim = layout.dim();
    let xsize_groups = xsize.div_ceil(gdim);
    let ysize_groups = ysize.div_ceil(gdim);
    let num_ac_groups = xsize_groups * ysize_groups;
    let num_dc_groups = xsize.div_ceil(layout.lf_dim()) * ysize.div_ceil(layout.lf_dim());
    let single_group = num_ac_groups == 1;

    let linear = rct_planes_of(&mut rct_planes, ycocg, num_color, rct_type, pool, scratch);
    // Per-channel predictor search for the v1 context tree and flat paths
    // (deferred: a decisive learned tree never needs it).
    let predictors = if effort.predictor_search {
        choose_predictors_with_wp(
            linear, alpha, xsize, ysize, num_color, pool, scratch, wp_params, use_wp,
        )
    } else {
        [fixed_predictor(use_wp); 4]
    };
    // Contiguous per-modular-channel predictors: the `num_color` color channels
    // (Y for gray; Y/Co/Cg for color) followed by alpha. For 3-color this is just
    // predictors[..nb_chans]; for gray it is [Y_pred, (alpha_pred)].
    let mut chan_preds: Vec<u32> = {
        let mut v: Vec<u32> = (0..num_color).map(|c| predictors[c]).collect();
        if alpha.is_some() {
            v.push(predictors[3]);
        }
        v
    };
    // A solid alpha plane: Zero predictor with the value as the leaf offset
    // (the predictor search's tie-break would pick the Weighted Predictor,
    // which the decoder then runs over the whole plane) and no tokens; its
    // channel is a constant tail of every stream (see `ConstantTail`).
    let mut chan_offsets = vec![0i32; nb_chans];
    if let Some(value) = alpha_constant {
        chan_preds[nb_chans - 1] = 0;
        chan_offsets[nb_chans - 1] = value;
    }
    // The fixed WP context tree of the flat path (`flat_wp_contexts`), for
    // displayed frames: a reference-only atlas keeps one context per channel.
    let displayed = matches!(
        frame_kind,
        ModularFrameKind::Regular
            | ModularFrameKind::Patched { .. }
            | ModularFrameKind::XybBlend { .. }
            | ModularFrameKind::RgbSavedFirst
            | ModularFrameKind::RgbBlendLast
            | ModularFrameKind::RgbCrop { .. }
            | ModularFrameKind::RgbBlendCrop { .. }
    );
    let wp_tree = (effort.flat_wp_contexts && use_wp && displayed)
        .then(|| {
            let bucketed: Vec<bool> = (0..nb_chans)
                .map(|chan| {
                    chan_preds[chan] == PREDICTOR_WEIGHTED
                        && !(chan >= num_color && alpha_constant.is_some())
                })
                .collect();
            bucketed.contains(&true).then(|| {
                let shift = max_bits.saturating_sub(8);
                wp_context_tree(&chan_preds, &chan_offsets, &bucketed, shift)
            })
        })
        .flatten();
    let flat_buckets = wp_tree.as_ref().map(|tree| tree.buckets.as_slice());
    let flat_contexts = wp_tree.as_ref().map_or(nb_chans, |tree| tree.num_ctx);
    let flat_tail = |w: usize, h: usize| -> ConstantTail {
        match alpha_constant {
            Some(_) => ConstantTail {
                context: wp_tree
                    .as_ref()
                    .map_or(channel_to_context(nb_chans - 1, nb_chans), |tree| {
                        tree.buckets[nb_chans - 1].contexts[0]
                    }),
                count: w * h,
            },
            None => ConstantTail::NONE,
        }
    };

    // Context tree (v1): single-group. Splits each channel's entropy context on
    // the WP activity property; a big win on smooth+edge content. Falls through
    // to the flat path when it isn't estimated to help.
    if compare_tree_candidates && effort.tree_ctx_v1 && use_wp {
        let mut candidate = BitWriter::new();
        if single_group {
            if try_encode_context_tree_single_group(
                speed,
                num_color,
                linear,
                alpha,
                alpha_constant,
                xsize,
                ysize,
                layout,
                frame_kind,
                &predictors,
                min_symbol,
                rct_type,
                pool,
                scratch,
                wp_params,
                &mut candidate,
            ) {
                keep_smaller_writer(&mut best_tree_writer, candidate);
            }
        } else if try_encode_context_tree_multi_group(
            speed,
            num_color,
            linear,
            alpha,
            alpha_constant,
            xsize,
            ysize,
            layout,
            frame_kind,
            &predictors,
            xsize_groups,
            ysize_groups,
            num_dc_groups,
            min_symbol,
            rct_type,
            pool,
            scratch,
            wp_params,
            &mut candidate,
        ) {
            keep_smaller_writer(&mut best_tree_writer, candidate);
        }
    }

    if !chan_preds.contains(&PREDICTOR_WEIGHTED) {
        wp_params = WpParams::DEFAULT;
    }

    let mut flat_writer = BitWriter::new();
    {
        let writer = if compare_tree_candidates {
            &mut flat_writer
        } else {
            &mut *writer
        };

        write_frame_header_modular_kind(alpha.is_some(), frame_kind, layout, writer);

        if single_group {
            // Single section: GroupHeader + local tree + pixel histograms + pixels.
            let mut section = BitWriter::new();
            if let ModularFrameKind::Patched { references, .. } = frame_kind {
                write_patch_dictionary(references, alpha.is_some(), scratch, &mut section);
            }
            // 1 bit: dc_quant all_default = 1
            section.write(1, 1);
            // 1 bit: has_tree = 0  (no global tree; the local tree lives in the GroupHeader).
            section.write(1, 0);
            // GroupHeader: use_global_tree=0, wp_default=1, RCT transform on R/G/B.
            section.write(1, 0);
            write_wp_header(wp_params, &mut section);
            write_modular_transforms(nb_chans, rct_type, &mut section);

            // Tokenize all channels (post-YCoCg, per-channel contexts).
            let channel_tokens = tokenize_channels_with_wp(
                linear,
                alpha,
                xsize,
                ysize,
                0,
                0,
                xsize,
                ysize,
                num_color,
                &chan_preds,
                grad_pack_fn,
                pool,
                scratch,
                wp_params,
                alpha_constant,
                flat_buckets,
            );
            let tail = flat_tail(xsize, ysize);

            // LZ77 layer: collapse runs of identical tokens into back-references.
            // The distance context is the (nb_chans)-th context, appended after the
            // per-channel ones.
            let distance_ctx = flat_contexts as u32;
            let lz_tokens =
                lz77_compress_channels_for_speed(channel_tokens, distance_ctx, speed, scratch);

            // Per-cluster prefix codes (nb_chans + 1 contexts), balanced N-leaf tree.
            let code = build_lz_pixel_code_tails(
                speed.effort().lz_exhaustive_entropy(),
                &[lz_tokens.as_slice()],
                &[tail],
                flat_contexts,
                min_symbol,
                speed.effort().lz_refined_entropy,
                false,
                None,
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            write_flat_tree(
                wp_tree.as_ref(),
                &chan_preds,
                &chan_offsets,
                &code,
                min_symbol,
                &mut scratch.huffman_pool,
                &mut section,
            );

            // Emit the LZ77'd token stream.
            write_lz_section_tail(
                &lz_tokens,
                tail,
                distance_ctx,
                &code,
                min_symbol,
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
        } else {
            // Multi-group: a single global tree + histograms in DC global, then each
            // AC group emits its tokens against those codes.
            let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
            let mut sections: Vec<BitWriter> =
                (0..num_sections).map(|_| BitWriter::new()).collect();

            // Tokenize each AC group (sub-image-local neighbors, matching what
            // we'll emit below) and run LZ77 over each group's stream separately so
            // back-references stay within a group's modular sub-image.  Pool the
            // resulting LzToken streams to build a single global prefix code so
            // every per-group emission is guaranteed to be representable.
            let distance_ctx = flat_contexts as u32;
            let group_lz_tokens: Vec<Vec<LzToken>> = {
                let deep_lz = speed
                    .effort()
                    .lz_deep
                    .then(|| DeepLzScratchPool::new(group_lz_threads(speed, pool)));
                pool.steal_map_with_threads(
                    scratch,
                    num_ac_groups,
                    group_lz_threads(speed, pool),
                    |group_index, scratch| {
                        let gx = group_index % xsize_groups;
                        let gy = group_index / xsize_groups;
                        let x0 = gx * gdim;
                        let y0 = gy * gdim;
                        let gw = gdim.min(xsize - x0);
                        let gh = gdim.min(ysize - y0);
                        if speed.effort().lz_deep {
                            let channel_tokens = tokenize_channels_with_wp(
                                linear,
                                alpha,
                                xsize,
                                ysize,
                                x0,
                                y0,
                                gw,
                                gh,
                                num_color,
                                &chan_preds,
                                grad_pack_fn,
                                pool,
                                scratch,
                                wp_params,
                                alpha_constant,
                                flat_buckets,
                            );
                            deep_lz.as_ref().unwrap().with_depth(|depth| {
                                lz77_compress_channels_for_speed_with_depth(
                                    channel_tokens,
                                    distance_ctx,
                                    speed,
                                    depth,
                                    scratch,
                                )
                            })
                        } else {
                            tokenize_runs_with_wp(
                                linear,
                                alpha,
                                xsize,
                                x0,
                                y0,
                                gw,
                                gh,
                                num_color,
                                &chan_preds,
                                grad_pack_fn,
                                scratch,
                                wp_params,
                                alpha_constant,
                                flat_buckets,
                            )
                        }
                    },
                )
            };
            let group_tails: Vec<ConstantTail> = (0..num_ac_groups)
                .map(|group_index| {
                    let x0 = (group_index % xsize_groups) * gdim;
                    let y0 = (group_index / xsize_groups) * gdim;
                    flat_tail(gdim.min(xsize - x0), gdim.min(ysize - y0))
                })
                .collect();
            // ----- Section 0: DC global -----
            if let ModularFrameKind::Patched { references, .. } = frame_kind {
                write_patch_dictionary(references, alpha.is_some(), scratch, &mut sections[0]);
            }
            let group_slices: Vec<&[LzToken]> = group_lz_tokens.iter().map(Vec::as_slice).collect();
            let code = build_lz_pixel_code_tails(
                speed.effort().lz_exhaustive_entropy(),
                &group_slices,
                &group_tails,
                flat_contexts,
                min_symbol,
                speed.effort().lz_refined_entropy,
                false,
                None,
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            sections[0].write(1, 1); // dc_quant all_default = 1
            sections[0].write(1, 1); // has_tree = 1
            write_flat_tree(
                wp_tree.as_ref(),
                &chan_preds,
                &chan_offsets,
                &code,
                min_symbol,
                &mut scratch.huffman_pool,
                &mut sections[0],
            );
            // GroupHeader for the global modular image: use_global_tree=1, wp=1, RCT transform.
            sections[0].write(1, 1);
            write_wp_header(wp_params, &mut sections[0]);
            write_modular_transforms(nb_chans, rct_type, &mut sections[0]);
            // An empty global stream still carries its ANS state: jxl-oxide reads it.
            write_lz_section::<LzToken>(&[], distance_ctx, &code, min_symbol, &mut sections[0]);
            sections[0].zero_pad_to_byte();

            // ----- DC groups: empty GroupHeader only -----
            for section in &mut sections[1..1 + num_dc_groups] {
                section.write(1, 1); // use_global_tree
                write_wp_header(wp_params, section);
                section.write(2, 0); // 0 transforms
                section.zero_pad_to_byte();
            }

            // ----- AC global: trivial (all_default flags) -----
            let ac_global_idx = 1 + num_dc_groups;
            sections[ac_global_idx].write(1, 1);
            write_wp_header(wp_params, &mut sections[ac_global_idx]);
            sections[ac_global_idx].zero_pad_to_byte();

            write_lz_groups(
                &group_lz_tokens,
                &group_tails,
                &code,
                distance_ctx,
                min_symbol,
                wp_params,
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

    if compare_tree_candidates {
        keep_smaller_writer(&mut best_tree_writer, flat_writer);
        writer.append(best_tree_writer.as_ref().expect("flat lossless candidate"));
    }
    false
}
