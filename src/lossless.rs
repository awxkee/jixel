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

#[path = "enc_lossless_lz77.rs"]
mod lz77;

use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS};
use crate::thread_pool::ThreadPool;
pub(crate) use lz77::LzToken;
use lz77::{
    LZ77_MIN_SYMBOL, RunLzWriter, build_lz_pixel_code, lz77_compress_channels_for_speed,
    lz77_compress_channels_for_speed_with_depth, lz77_compress_for_speed,
    lz77_compress_for_speed_with_depth, lz77_compress_runs, write_local_tree_lz77,
    write_lz_section, write_tree_lz77,
};
use std::sync::{Condvar, Mutex};

const TREE_CTX_SPLIT_VAL: u32 = 0;
const TREE_CTX_PROPERTY: u32 = 1;
const TREE_CTX_PREDICTOR: u32 = 2;
const TREE_CTX_OFFSET: u32 = 3;
const TREE_CTX_MULTIPLIER_LOG: u32 = 4;
const TREE_CTX_MULTIPLIER_BITS: u32 = 5;
const NUM_TREE_CONTEXTS: usize = 6;
const PREDICTOR_ZERO: u32 = 0;
const PREDICTOR_LEFT: u32 = 1;
const PREDICTOR_TOP: u32 = 2;
const PREDICTOR_AVERAGE0: u32 = 3;
const PREDICTOR_SELECT: u32 = 4;
const PREDICTOR_GRADIENT: u32 = 5;
const PREDICTOR_WEIGHTED: u32 = 6;
const PREDICTOR_TOP_RIGHT: u32 = 7;
const PREDICTOR_TOP_LEFT: u32 = 8;
const PREDICTOR_LEFT_LEFT: u32 = 9;
const PREDICTOR_AVERAGE1: u32 = 10;
const PREDICTOR_AVERAGE2: u32 = 11;
const PREDICTOR_AVERAGE3: u32 = 12;
const PREDICTOR_AVERAGE4: u32 = 13;
static SLOW_PREDICTORS: [u32; 6] = [
    PREDICTOR_WEIGHTED,
    PREDICTOR_GRADIENT,
    PREDICTOR_AVERAGE4,
    PREDICTOR_SELECT,
    PREDICTOR_LEFT,
    PREDICTOR_TOP,
];

#[derive(Clone, Copy)]
struct PredictorNeighbors {
    left: i64,
    top: i64,
    top_left: i64,
    top_right: i64,
    left_left: i64,
    top_top: i64,
    top_right_right: i64,
}

#[inline]
fn predictor_neighbors<F: Fn(usize, usize) -> i32 + ?Sized>(
    get: &F,
    x: usize,
    y: usize,
    width: usize,
) -> PredictorNeighbors {
    let left = if x > 0 {
        get(x - 1, y) as i64
    } else if y > 0 {
        get(x, y - 1) as i64
    } else {
        0
    };
    let top = if y > 0 { get(x, y - 1) as i64 } else { left };
    let top_left = if x > 0 && y > 0 {
        get(x - 1, y - 1) as i64
    } else {
        left
    };
    let top_right = if x + 1 < width && y > 0 {
        get(x + 1, y - 1) as i64
    } else {
        top
    };
    PredictorNeighbors {
        left,
        top,
        top_left,
        top_right,
        left_left: if x > 1 { get(x - 2, y) as i64 } else { left },
        top_top: if y > 1 { get(x, y - 2) as i64 } else { top },
        top_right_right: if x + 2 < width && y > 0 {
            get(x + 2, y - 1) as i64
        } else {
            top_right
        },
    }
}

#[inline]
fn predictor_value(pred_id: u32, n: PredictorNeighbors, weighted: i64) -> i64 {
    match pred_id {
        PREDICTOR_ZERO => 0,
        PREDICTOR_LEFT => n.left,
        PREDICTOR_TOP => n.top,
        PREDICTOR_AVERAGE0 => (n.left + n.top) / 2,
        PREDICTOR_SELECT => {
            let projected = n.left + n.top - n.top_left;
            if (projected - n.left).abs() < (projected - n.top).abs() {
                n.left
            } else {
                n.top
            }
        }
        PREDICTOR_GRADIENT => clamped_gradient(n.left, n.top, n.top_left),
        PREDICTOR_WEIGHTED => weighted,
        PREDICTOR_TOP_RIGHT => n.top_right,
        PREDICTOR_TOP_LEFT => n.top_left,
        PREDICTOR_LEFT_LEFT => n.left_left,
        PREDICTOR_AVERAGE1 => (n.left + n.top_left) / 2,
        PREDICTOR_AVERAGE2 => (n.top_left + n.top) / 2,
        PREDICTOR_AVERAGE3 => (n.top + n.top_right) / 2,
        PREDICTOR_AVERAGE4 => {
            (6 * n.top - 2 * n.top_top
                + 7 * n.left
                + n.left_left
                + n.top_right_right
                + 3 * n.top_right
                + 8)
                / 16
        }
        _ => unreachable!("unsupported modular predictor {pred_id}"),
    }
}

fn wp_sample_cost(
    get: impl Fn(usize, usize) -> i32,
    width: usize,
    height: usize,
    params: WpParams,
) -> (u64, usize) {
    let cw = width.min(128);
    let ch = height.min(128);
    let xs = if width > cw { [0, width - cw] } else { [0, 0] };
    let ys = if height > ch {
        [0, height - ch]
    } else {
        [0, 0]
    };
    let nx = 1 + usize::from(xs[1] != xs[0]);
    let ny = 1 + usize::from(ys[1] != ys[0]);
    let mut cost = 0u64;
    let mut count = 0usize;
    for &y0 in &ys[..ny] {
        for &x0 in &xs[..nx] {
            let local_get = |x: usize, y: usize| get(x0 + x, y0 + y);
            let mut wp = WpState::with_params(cw, params);
            for y in 0..ch {
                for x in 0..cw {
                    let neighbors = predictor_neighbors(&local_get, x, y, cw);
                    let value = local_get(x, y);
                    let prediction = wp.predict(
                        x,
                        y,
                        neighbors.top,
                        neighbors.left,
                        neighbors.top_right,
                        neighbors.top_left,
                        neighbors.top_top,
                    );
                    let packed = pack_signed(value.wrapping_sub(prediction as i32));
                    let (_, extra_bits, _) = crate::entropy::uint_encode(packed);
                    let magnitude_bits = packed
                        .checked_add(1)
                        .map_or(32, |value| 32 - value.leading_zeros());
                    cost += extra_bits as u64 + u64::from(magnitude_bits);
                    count += 1;
                    wp.update(value as i64, x, y);
                }
            }
        }
    }
    (cost, count)
}

fn choose_wp_params(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    num_color: usize,
    header_count: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> WpParams {
    let width = linear.xsize();
    let height = linear.ysize();
    let nb_chans = num_color + usize::from(alpha.is_some());
    let total_values = width.saturating_mul(height).saturating_mul(nb_chans);
    let scores = pool.steal_map(scratch, WpParams::PRESETS.len(), |preset, _scratch| {
        let params = WpParams::PRESETS[preset];
        let mut sample_cost = 0u64;
        let mut samples = 0usize;
        for channel in 0..num_color {
            let plane = linear.plane_data(channel);
            let (cost, count) = wp_sample_cost(|x, y| plane[y * width + x], width, height, params);
            sample_cost += cost;
            samples += count;
        }
        if let Some(alpha) = alpha {
            let (cost, count) =
                wp_sample_cost(|x, y| alpha.get_i32(y * width + x), width, height, params);
            sample_cost += cost;
            samples += count;
        }
        let scaled = sample_cost as f64 * total_values as f64 / samples.max(1) as f64;
        // A non-default header costs 51 additional bits per modular group.
        scaled
            + if preset == 0 {
                0.0
            } else {
                (51 * header_count) as f64
            }
    });
    let best = scores
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index);
    WpParams::PRESETS[best]
}

const GROUP_DIM: usize = 256;
const LF_GROUP_DIM: usize = 2048;
/// Each active deep-LZ lane owns about 4.7 MiB of hash lookup storage. Eight
/// lanes retain useful parallelism without allowing a many-core encoder to
/// multiply that fixed working set across every worker.
const SLOW_DEEP_LZ_MAX_THREADS: usize = 8;

/// A sparse set of local-palette groups forces Slow mode to build a complete
/// alternate frame while most groups lose the learned-tree model. Require a
/// useful frame footprint before paying for that exact candidate.
const LOCAL_PALETTE_MIN_COVERAGE_NUM: usize = 1;
const LOCAL_PALETTE_MIN_COVERAGE_DEN: usize = 4;

/// The learned-tree entropy proxy must beat its flat alternative decisively
/// before Slow mode trusts it enough to omit the context/flat final encodes.
/// Near decisions retain the exact byte-level tournament.
const MA_DECISIVE_MIN_SAVINGS: f64 = 0.18;

#[inline]
fn local_palette_coverage_is_sufficient(palette_pixels: usize, total_pixels: usize) -> bool {
    palette_pixels.saturating_mul(LOCAL_PALETTE_MIN_COVERAGE_DEN)
        >= total_pixels.saturating_mul(LOCAL_PALETTE_MIN_COVERAGE_NUM)
}

#[inline]
fn learned_tree_is_decisive(estimated_savings: f64) -> bool {
    estimated_savings >= MA_DECISIVE_MIN_SAVINGS
}

#[inline]
fn group_lz_threads(speed: crate::Speed, pool: &ThreadPool) -> usize {
    if speed == crate::Speed::Slow {
        pool.num_threads().min(SLOW_DEEP_LZ_MAX_THREADS)
    } else {
        pool.num_threads()
    }
}

/// Fixed set of reusable deep-LZ lookup tables for one group-compression
/// phase. Unlike worker-owned scratch, these buffers are dropped as soon as
/// the phase finishes and cannot spread across more than the configured lanes.
struct DeepLzScratchPool {
    available: Mutex<Vec<Vec<u32>>>,
    activity: Condvar,
}

impl DeepLzScratchPool {
    fn new(slots: usize) -> Self {
        Self {
            available: Mutex::new((0..slots.max(1)).map(|_| Vec::new()).collect()),
            activity: Condvar::new(),
        }
    }

    fn with_depth<T>(&self, f: impl FnOnce(&mut Vec<u32>) -> T) -> T {
        let mut available = self.available.lock().unwrap();
        while available.is_empty() {
            available = self.activity.wait(available).unwrap();
        }
        let depth = available.pop().unwrap();
        drop(available);

        let mut guard = DeepLzScratchGuard {
            owner: self,
            depth: Some(depth),
        };
        f(guard.depth.as_mut().unwrap())
    }
}

struct DeepLzScratchGuard<'a> {
    owner: &'a DeepLzScratchPool,
    depth: Option<Vec<u32>>,
}

impl Drop for DeepLzScratchGuard<'_> {
    fn drop(&mut self) {
        let depth = self.depth.take().unwrap();
        self.owner.available.lock().unwrap().push(depth);
        self.owner.activity.notify_one();
    }
}

fn zero_alpha_like(alpha: Option<&AlphaPlane>, pixels: usize) -> Option<AlphaPlane> {
    match alpha {
        Some(AlphaPlane::U8(_)) => Some(AlphaPlane::U8(vec![0; pixels])),
        Some(AlphaPlane::U16 { bits, .. }) => Some(AlphaPlane::U16 {
            data: vec![0; pixels],
            bits: *bits,
        }),
        Some(AlphaPlane::F32(_)) => Some(AlphaPlane::F32(vec![0; pixels])),
        None => None,
    }
}

fn keep_smaller_writer(best: &mut Option<BitWriter>, candidate: BitWriter) {
    if best
        .as_ref()
        .is_none_or(|current| candidate.bits_written() < current.bits_written())
    {
        *best = Some(candidate);
    }
}

pub(crate) fn encode_frame_lossless(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    patches: bool,
    num_color: usize,
    speed: crate::Speed,
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let pool = ThreadPool::new_lossless(num_threads);
    let mut scratch = Box::new(CoderScratch::lossless());
    encode_frame_lossless_with_pool(
        linear,
        alpha,
        max_bits,
        progressive,
        patches,
        num_color,
        speed,
        &pool,
        &mut scratch,
        writer,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_frame_lossless_with_pool(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    patches: bool,
    num_color: usize,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    if patches
        && num_color == 3
        && let Some(plan) = find_lossless_patches(linear, pool, scratch)
    {
        // Patches change prediction, LZ77, entropy histograms, and add both a
        // reference-only frame and a dictionary. Their true cost cannot be
        // inferred reliably from covered pixel area, so encode both complete
        // alternatives and select by their final byte-aligned bit count.
        let mut regular_writer = BitWriter::new();
        encode_frame_lossless_core(
            linear,
            alpha,
            max_bits,
            progressive,
            num_color,
            speed,
            pool,
            scratch,
            ModularFrameKind::Regular,
            &mut regular_writer,
        );

        let mut patched_writer = BitWriter::new();
        let atlas_alpha =
            zero_alpha_like(alpha, plan.atlas.xsize().saturating_mul(plan.atlas.ysize()));
        encode_frame_lossless_core(
            &plan.atlas,
            atlas_alpha.as_ref(),
            max_bits,
            false,
            num_color,
            speed,
            pool,
            scratch,
            ModularFrameKind::ReferenceOnly {
                width: plan.atlas.xsize(),
                height: plan.atlas.ysize(),
            },
            &mut patched_writer,
        );
        encode_frame_lossless_core(
            &plan.base,
            alpha,
            max_bits,
            false,
            num_color,
            speed,
            pool,
            scratch,
            ModularFrameKind::Patched(&plan.references),
            &mut patched_writer,
        );

        if patched_writer.bits_written() < regular_writer.bits_written() {
            writer.append(&patched_writer);
        } else {
            writer.append(&regular_writer);
        }
        return;
    }
    encode_frame_lossless_core(
        linear,
        alpha,
        max_bits,
        progressive,
        num_color,
        speed,
        pool,
        scratch,
        ModularFrameKind::Regular,
        writer,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_frame_lossless_core(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    num_color: usize,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    frame_kind: ModularFrameKind<'_>,
    writer: &mut BitWriter,
) {
    encode_frame_lossless_core_impl(
        linear,
        alpha,
        max_bits,
        progressive,
        num_color,
        speed,
        pool,
        scratch,
        frame_kind,
        true,
        writer,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_frame_lossless_core_impl(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    num_color: usize,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    frame_kind: ModularFrameKind<'_>,
    allow_palette_candidates: bool,
    writer: &mut BitWriter,
) {
    // The hybrid-uint token of a modular residual can reach ~4*B+11 for B-bit
    // input. For B<=13 this stays below 64, so keep the tight default; for higher
    // bit depths raise LZ77_MIN_SYMBOL above the max literal token so the decoder
    // never mistakes a large residual for an LZ77 back-reference (the cause of
    // the old ~12-bit lossless ceiling). Stays under the 128-symbol prefix-code
    // alphabet through ~18-bit.
    let min_symbol: u32 = if max_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * max_bits + 24
    };
    let xsize = linear.xsize();
    let ysize = linear.ysize();
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };

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
            scratch,
            &mut palette_writer,
        ) {
            if speed == crate::Speed::Slow {
                let mut normal_writer = BitWriter::new();
                encode_frame_lossless_core_impl(
                    linear,
                    alpha,
                    max_bits,
                    progressive,
                    num_color,
                    speed,
                    pool,
                    scratch,
                    frame_kind,
                    false,
                    &mut normal_writer,
                );
                if normal_writer.bits_written() < palette_writer.bits_written() {
                    writer.append(&normal_writer);
                } else {
                    writer.append(&palette_writer);
                }
            } else {
                writer.append(&palette_writer);
            }
            return;
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
            pool,
            scratch,
            &mut palette_writer,
        ) {
            if speed == crate::Speed::Slow {
                let mut normal_writer = BitWriter::new();
                encode_frame_lossless_core_impl(
                    linear,
                    alpha,
                    max_bits,
                    progressive,
                    num_color,
                    speed,
                    pool,
                    scratch,
                    frame_kind,
                    false,
                    &mut normal_writer,
                );
                if normal_writer.bits_written() < palette_writer.bits_written() {
                    writer.append(&normal_writer);
                } else {
                    writer.append(&palette_writer);
                }
            } else {
                writer.append(&palette_writer);
            }
            return;
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
            pool,
            scratch,
            writer,
        );
        return;
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
            pool,
            scratch,
            writer,
        )
    {
        return;
    }

    // Fast uses fixed Weighted prediction and skips all adaptive analysis. Slow
    // searches the supported Slow predictor set per channel by estimated cost
    // (one global choice, used for the tree and every group).
    let adaptive_search = speed == crate::Speed::Slow;

    // RCT re-selection (Slow, regular RGB frames): estimated over the same
    // fixed-context gradient cost libjxl uses. When a non-YCoCg transform
    // wins, the color planes are rebuilt and the frame headers signal it.
    let mut rct_type = 6u32;
    let mut rct_planes: Option<Image3Si> = None;
    if frame_kind.is_regular()
        && adaptive_search
        && num_color == 3
        && let Some((t, planes)) = select_rct(linear, xsize, ysize, pool, scratch)
    {
        rct_type = t;
        rct_planes = Some(planes);
    }
    let linear = rct_planes.as_ref().unwrap_or(linear);

    let wp_header_count = if single_group {
        1
    } else {
        2 + num_dc_groups + num_ac_groups
    };
    let mut wp_params = if adaptive_search {
        choose_wp_params(linear, alpha, num_color, wp_header_count, pool, scratch)
    } else {
        WpParams::DEFAULT
    };
    let predictors = if adaptive_search {
        choose_predictors_with_wp(linear, alpha, xsize, ysize, pool, scratch, wp_params)
    } else {
        [PREDICTOR_WEIGHTED; 4]
    };
    // Contiguous per-modular-channel predictors: the `num_color` color channels
    // (Y for gray; Y/Co/Cg for color) followed by alpha. For 3-color this is just
    // predictors[..nb_chans]; for gray it is [Y_pred, (alpha_pred)].
    let chan_preds: Vec<u32> = {
        let mut v: Vec<u32> = (0..num_color).map(|c| predictors[c]).collect();
        if alpha.is_some() {
            v.push(predictors[3]);
        }
        v
    };

    // The estimates below remain useful prefilters, but Slow mode keeps every
    // viable finalist and compares its complete byte-aligned frame against the
    // v1 and flat alternatives. Tree headers, clustering, hybrid uint choices,
    // and LZ77 can otherwise reverse the order predicted from residual entropy.
    let compare_tree_candidates = frame_kind.is_regular() && adaptive_search && num_color == 3;
    let mut best_tree_writer: Option<BitWriter> = None;
    let mut learned_estimated_savings: Option<f64> = None;

    // Learned MA context tree (v2): greedy learned tree over the standard
    // property vector with per-leaf predictors.
    if compare_tree_candidates {
        let mut candidate = BitWriter::new();
        if single_group {
            if let Some(estimated_savings) = try_encode_learned_tree_single_group(
                linear,
                alpha,
                xsize,
                ysize,
                min_symbol,
                rct_type,
                pool,
                scratch,
                wp_params,
                &mut candidate,
            ) {
                learned_estimated_savings = Some(estimated_savings);
                keep_smaller_writer(&mut best_tree_writer, candidate);
            }
        } else if let Some(estimated_savings) = try_encode_learned_tree_multi_group(
            linear,
            alpha,
            xsize,
            ysize,
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
            learned_estimated_savings = Some(estimated_savings);
            keep_smaller_writer(&mut best_tree_writer, candidate);
        }
    }

    if learned_estimated_savings.is_some_and(learned_tree_is_decisive) {
        writer.append(
            best_tree_writer
                .as_ref()
                .expect("decisive learned-tree candidate"),
        );
        return;
    }

    // Context tree (v1): single-group. Splits each channel's entropy context on
    // the WP activity property; a big win on smooth+edge content. Falls through
    // to the flat path when it isn't estimated to help.
    if compare_tree_candidates {
        let mut candidate = BitWriter::new();
        if single_group {
            if try_encode_context_tree_single_group(
                linear,
                alpha,
                xsize,
                ysize,
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
            linear,
            alpha,
            xsize,
            ysize,
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

        write_frame_header_modular_kind(alpha.is_some(), frame_kind, writer);

        if single_group {
            // Single section: GroupHeader + local tree + pixel histograms + pixels.
            let mut section = BitWriter::new();
            if let ModularFrameKind::Patched(references) = frame_kind {
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
            );

            // LZ77 layer: collapse runs of identical tokens into back-references.
            // The distance context is the (nb_chans)-th context, appended after the
            // per-channel ones.
            let distance_ctx = nb_chans as u32;
            let lz_tokens =
                lz77_compress_channels_for_speed(channel_tokens, distance_ctx, speed, scratch);

            // Per-cluster prefix codes (nb_chans + 1 contexts), balanced N-leaf tree.
            let code = build_lz_pixel_code(
                std::iter::once(lz_tokens.as_slice()),
                nb_chans,
                min_symbol,
                speed == crate::Speed::Slow,
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            write_local_tree_lz77(
                &chan_preds,
                &code,
                min_symbol,
                &mut scratch.huffman_pool,
                &mut section,
            );

            // Emit the LZ77'd token stream.
            write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
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
            let distance_ctx = nb_chans as u32;
            let deep_lz = (speed == crate::Speed::Slow)
                .then(|| DeepLzScratchPool::new(group_lz_threads(speed, pool)));
            let group_lz_tokens: Vec<Vec<LzToken>> = pool.steal_map_with_threads(
                scratch,
                num_ac_groups,
                group_lz_threads(speed, pool),
                |group_index, scratch| {
                    let gx = group_index % xsize_groups;
                    let gy = group_index / xsize_groups;
                    let x0 = gx * GROUP_DIM;
                    let y0 = gy * GROUP_DIM;
                    let gw = GROUP_DIM.min(xsize - x0);
                    let gh = GROUP_DIM.min(ysize - y0);
                    if speed == crate::Speed::Slow {
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
                        )
                    }
                },
            );
            drop(deep_lz);
            // ----- Section 0: DC global -----
            if let ModularFrameKind::Patched(references) = frame_kind {
                write_patch_dictionary(references, alpha.is_some(), scratch, &mut sections[0]);
            }
            let code = build_lz_pixel_code(
                group_lz_tokens.iter().map(Vec::as_slice),
                nb_chans,
                min_symbol,
                speed == crate::Speed::Slow,
                &mut scratch.lz_entropy,
                &mut scratch.huffman_pool,
            );
            sections[0].write(1, 1); // dc_quant all_default = 1
            sections[0].write(1, 1); // has_tree = 1
            write_local_tree_lz77(
                &chan_preds,
                &code,
                min_symbol,
                &mut scratch.huffman_pool,
                &mut sections[0],
            );
            // GroupHeader for the global modular image: use_global_tree=1, wp=1, RCT transform.
            sections[0].write(1, 1);
            write_wp_header(wp_params, &mut sections[0]);
            write_modular_transforms(nb_chans, rct_type, &mut sections[0]);
            sections[0].zero_pad_to_byte();

            // ----- DC groups: empty GroupHeader only -----
            for i in 0..num_dc_groups {
                sections[1 + i].write(1, 1); // use_global_tree
                write_wp_header(wp_params, &mut sections[1 + i]);
                sections[1 + i].write(2, 0); // 0 transforms
                sections[1 + i].zero_pad_to_byte();
            }

            // ----- AC global: trivial (all_default flags) -----
            let ac_global_idx = 1 + num_dc_groups;
            sections[ac_global_idx].write(1, 1);
            write_wp_header(wp_params, &mut sections[ac_global_idx]);
            sections[ac_global_idx].zero_pad_to_byte();

            // ----- AC groups: pixel data per group -----
            for gy in 0..ysize_groups {
                for gx in 0..xsize_groups {
                    let group_index = gy * xsize_groups + gx;
                    let section_idx = 2 + num_dc_groups + group_index;

                    // GroupHeader: use_global_tree=1, wp=1, 0 transforms (the global
                    // header already declared the RCT for the whole image).
                    sections[section_idx].write(1, 1);
                    write_wp_header(wp_params, &mut sections[section_idx]);
                    sections[section_idx].write(2, 0);

                    write_lz_section(
                        &group_lz_tokens[group_index],
                        distance_ctx,
                        &code,
                        min_symbol,
                        &mut sections[section_idx],
                    );
                    sections[section_idx].zero_pad_to_byte();
                }
            }

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
}

fn write_u64(value: u64, w: &mut BitWriter) {
    match value {
        0 => w.write(2, 0),
        1..=16 => {
            w.write(2, 1);
            w.write(4, value - 1);
        }
        17..=272 => {
            w.write(2, 2);
            w.write(8, value - 17);
        }
        _ => unreachable!("lossless frame flags fit the short U64 form"),
    }
}

fn write_frame_dimension(value: usize, w: &mut BitWriter) {
    let value = value as u64;
    if value < 256 {
        w.write(2, 0);
        w.write(8, value);
    } else if value < 2304 {
        w.write(2, 1);
        w.write(11, value - 256);
    } else if value < 18688 {
        w.write(2, 2);
        w.write(14, value - 2304);
    } else {
        w.write(2, 3);
        w.write(30, value - 18688);
    }
}

fn write_frame_header_modular_kind(has_alpha: bool, kind: ModularFrameKind<'_>, w: &mut BitWriter) {
    match kind {
        ModularFrameKind::Regular => write_frame_header_modular(has_alpha, w),
        ModularFrameKind::Patched(_) => write_frame_header_modular_flags(has_alpha, 2, w),
        ModularFrameKind::ReferenceOnly { width, height } => {
            w.write(1, 0); // all_default = false
            w.write(2, 0b10); // reference-only frame
            w.write(1, 1); // Modular
            write_u64(0, w); // flags
            w.write(1, 0); // color transform = None
            w.write(2, 0); // upsampling = 1
            if has_alpha {
                w.write(2, 0); // extra-channel upsampling = 1
            }
            w.write(2, 1); // group_size_shift = 1
            // Reference-only frames do not serialize Passes.
            w.write(1, 1); // custom size
            write_frame_dimension(width, w);
            write_frame_dimension(height, w);
            // No blending and no is_last for reference-only frames.
            w.write(2, PATCH_REF_ID as u64); // save_as_reference = 3
            w.write(1, 1); // save_before_color_transform = true
            w.write(2, 0); // empty name
            w.write(1, 0); // loop filter not all-default
            w.write(1, 0); // no gaborish
            w.write(2, 0); // no EPF
            w.write(2, 0); // no loop-filter extensions
            w.write(2, 0); // no frame-header extensions
        }
    }
}

/// Frame header for a modular reference-only frame inside the lossy
/// (`xyb_encoded`) codestream. Identical to the lossless variant except that
/// the container sets `xyb_encoded`, so the do_YCbCr bit is absent and the
/// frame's color transform is implicitly XYB. Saved to the modular slot so a
/// hybrid plan can also emit the VarDCT atlas in slot 3.
fn write_frame_header_modular_xyb_reference(
    width: usize,
    height: usize,
    has_alpha: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b10); // reference-only frame
    w.write(1, 1); // encoding = Modular
    write_u64(0, w); // flags
    // No do_YCbCr bit: with xyb_encoded set the color transform is XYB.
    w.write(2, 0); // upsampling = 1
    if has_alpha {
        w.write(2, 0); // ec_upsampling[0] = 1
    }
    w.write(2, 1); // group_size_shift = 1 (256-pixel groups)
    // Reference-only frames do not serialize Passes.
    w.write(1, 1); // custom size
    write_frame_dimension(width, w);
    write_frame_dimension(height, w);
    // No blending and no is_last for reference-only frames.
    w.write(2, crate::patches::MODULAR_PATCH_REF_ID as u64); // save_as_reference = 2
    w.write(1, 1); // save_before_color_transform = true
    w.write(2, 0); // empty name
    // Loop filter off: the atlas is stored exactly as coded; gaborish or EPF
    // would smear the glyph edges patches exist to preserve.
    w.write(1, 0); // loop filter not all-default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // no EPF
    w.write(2, 0); // no loop-filter extensions
    w.write(2, 0); // no frame-header extensions
}

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
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    use std::collections::HashMap;
    // The 13-bit residual bound behind LZ77_MIN_SYMBOL caps the refinement.
    debug_assert!(lattice_scale.is_power_of_two() && lattice_scale <= 8);
    let (xsize, ysize) = (atlas.xsize(), atlas.ysize());
    if xsize == 0 || ysize == 0 || xsize > GROUP_DIM || ysize > GROUP_DIM {
        return false;
    }
    let ch = quantize_xyb_channels(atlas, lattice_scale);
    let npx = xsize * ysize;
    let grad_pack_fn = selected_grad_pack_interior_fn();
    let slow = speed == crate::Speed::Slow;

    // Distinct quantized XYB triples, palette-capped like the lossless path.
    let mut seen: HashMap<[i32; 3], ()> = HashMap::with_capacity(257);
    for i in 0..npx {
        seen.entry([ch[0][i], ch[1][i], ch[2][i]]).or_insert(());
        if seen.len() > 256 {
            break;
        }
    }
    let use_palette = !seen.is_empty() && seen.len() <= 256;

    write_frame_header_modular_xyb_reference(xsize, ysize, has_alpha, writer);

    let mut section = BitWriter::new();
    // LfChannelDequant: the decoder multiplies each channel by these steps.
    // Custom F16 steps (each read as value/128) refine the lattice by
    // `lattice_scale`; powers of two keep every step F16-exact.
    if lattice_scale == 1 {
        section.write(1, 1); // dc_quant all_default = 1
    } else {
        use crate::quant_weights::{INV_DC_QUANT, f32_to_f16_bits};
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
        for (i, color) in colors.iter().enumerate() {
            for c in 0..3 {
                palette_ch[c * nb_colors + i] = color[c];
            }
        }
        let index_img: Vec<i32> = (0..npx)
            .map(|i| idx_of[&[ch[0][i], ch[1][i], ch[2][i]]] as i32)
            .collect();

        write_palette_transform(num_c as u32, nb_colors as u32, &mut section);
        let pget = |gx: usize, gy: usize| palette_ch[gy * nb_colors + gx];
        let iget = |gx: usize, gy: usize| index_img[gy * xsize + gx];
        let preds = if slow {
            vec![
                choose_predictor_for_plane(pget, nb_colors, num_c),
                choose_predictor_for_plane(iget, xsize, ysize),
            ]
        } else {
            vec![PREDICTOR_WEIGHTED; 2]
        };
        tokenize_plane(
            channel_to_context(0, 2),
            pget,
            nb_colors,
            num_c,
            preds[0],
            grad_pack_fn,
            &mut scratch.gradient,
            &mut tokens,
        );
        tokenize_plane(
            channel_to_context(1, 2),
            iget,
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
            let get = |gx: usize, gy: usize| data[gy * xsize + gx];
            let pred = if slow {
                choose_predictor_for_plane(get, xsize, ysize)
            } else {
                PREDICTOR_WEIGHTED
            };
            tokenize_plane(
                channel_to_context(c, num_c),
                get,
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
                PREDICTOR_WEIGHTED,
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
            preds.push(PREDICTOR_WEIGHTED);
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
        std::iter::once(lz_tokens.as_slice()),
        nb_chans,
        min_symbol,
        slow,
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

fn write_frame_header_modular(has_alpha: bool, w: &mut BitWriter) {
    write_frame_header_modular_flags(has_alpha, 0, w);
}

fn write_frame_header_modular_flags(has_alpha: bool, flags: u64, w: &mut BitWriter) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(flags, w);
    w.write(1, 0); // do_ycbcr = false   (xyb_encoded=0 so this is serialized)
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(2, 0b01); // group_size_shift = 1 (256-pixel groups)
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(1, 1); // is_last
    w.write(2, 0b00); // name length = 0
    w.write(1, 0); // loop_filter NOT all_default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // 0 EPF iters
    w.write(2, 0b00); // no LF extensions
    w.write(2, 0b00); // no FH extensions
}

pub(crate) fn write_patch_dictionary(
    references: &[PatchReference],
    has_alpha: bool,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    const NUM_REF: u32 = 0;
    const REFERENCE_FRAME: u32 = 1;
    const PATCH_SIZE: u32 = 2;
    const REF_POSITION: u32 = 3;
    const POSITION: u32 = 4;
    const BLEND_MODE: u32 = 5;
    const OFFSET: u32 = 6;
    const COUNT: u32 = 7;

    let mut tokens = Vec::new();
    tokens.push(Token::new(NUM_REF, references.len() as u32));
    for reference in references {
        tokens.push(Token::new(REFERENCE_FRAME, reference.ref_frame));
        tokens.push(Token::new(REF_POSITION, reference.atlas_x as u32));
        tokens.push(Token::new(REF_POSITION, reference.atlas_y as u32));
        tokens.push(Token::new(PATCH_SIZE, (PATCH_TILE - 1) as u32));
        tokens.push(Token::new(PATCH_SIZE, (PATCH_TILE - 1) as u32));
        tokens.push(Token::new(COUNT, (reference.positions.len() - 1) as u32));
        for (i, &(x, y)) in reference.positions.iter().enumerate() {
            if i == 0 {
                tokens.push(Token::new(POSITION, x as u32));
                tokens.push(Token::new(POSITION, y as u32));
            } else {
                let (px, py) = reference.positions[i - 1];
                tokens.push(Token::new(OFFSET, pack_signed(x as i32 - px as i32)));
                tokens.push(Token::new(OFFSET, pack_signed(y as i32 - py as i32)));
            }
            tokens.push(Token::new(BLEND_MODE, 1)); // color = Replace
            if has_alpha {
                tokens.push(Token::new(BLEND_MODE, 0)); // alpha = None
            }
        }
    }
    let code = optimize_entropy_code(&tokens, NUM_PATCH_CONTEXTS, &mut scratch.huffman_pool);
    let code_ref = code.as_ref();
    w.write(1, 0); // patch dictionary entropy stream has no LZ77
    write_entropy_code(&code_ref, &mut scratch.huffman_pool, w);
    for token in tokens {
        write_token(token, &code_ref, w);
    }
}

/// rct_type: U32(Val(6), Bits(2), BitsOffset(4,2), BitsOffset(6,10)), 0..41.
fn write_rct_type(rct_type: u32, w: &mut BitWriter) {
    debug_assert!(rct_type < 42);
    if rct_type == 6 {
        w.write(2, 0b00);
    } else if rct_type < 4 {
        w.write(2, 0b01);
        w.write(2, rct_type as u64);
    } else if rct_type < 18 {
        w.write(2, 0b10);
        w.write(4, (rct_type - 2) as u64);
    } else {
        w.write(2, 0b11);
        w.write(6, (rct_type - 10) as u64);
    }
}

fn write_modular_transforms(nb_chans: usize, rct_type: u32, w: &mut BitWriter) {
    if nb_chans >= 3 {
        // transforms count u2S(0, 1, Bits(4)+2, Bits(8)+18): selector 1 = Val(1) → 1 transform.
        w.write(2, 0b01);
        // Transform[0]:
        //   id Bits(2)            = 0 (RCT)
        //   begin_channel u2S(Bits(3), ...): selector 0 = Bits(3) → value 0 = 5 bits "00000"
        w.write(2, 0b00); // id = RCT (Bits(2))
        w.write(2, 0b00); // begin_channel selector 0
        w.write(3, 0); // begin_channel value (Bits(3)) = 0
        write_rct_type(rct_type, w);
    } else {
        w.write(2, 0b00); // 0 transforms
    }
}

/// Transform list for RCT + a Squeeze with the given step sequence.
/// Byte-exact to libjxl Transform/SqueezeParams field encoding.
fn write_modular_transforms_rct_squeeze(steps: &[crate::squeeze::SqueezeStep], w: &mut BitWriter) {
    // transforms count = 2 -> selector 2 + Bits(4)=0.
    w.write(2, 0b10);
    w.write(4, 0);
    // Transform[0] = RCT (YCoCg), begin_c=0.
    w.write(2, 0b00);
    w.write(2, 0b00);
    w.write(3, 0);
    w.write(2, 0b00);
    // Transform[1] = Squeeze.
    w.write(2, 0b10); // id = kSqueeze
    // num_squeezes: U32(Val0, BitsOffset(4,1), BitsOffset(6,9), BitsOffset(8,41)).
    let n = steps.len() as u32;
    if n == 0 {
        w.write(2, 0b00);
    } else if n <= 16 {
        w.write(2, 0b01);
        w.write(4, (n - 1) as u64);
    } else if n <= 72 {
        w.write(2, 0b10);
        w.write(6, (n - 9) as u64);
    } else {
        w.write(2, 0b11);
        w.write(8, (n - 41) as u64);
    }
    for s in steps {
        w.write(1, if s.horizontal { 1 } else { 0 });
        w.write(1, if s.in_place { 1 } else { 0 });
        // begin_c: U32(Bits3, ...): for small values use selector 0 = Bits(3).
        debug_assert!(s.begin_c < 8);
        w.write(2, 0b00);
        w.write(3, s.begin_c as u64);
        // num_c: U32(Val1,Val2,Val3,BitsOffset(4,4)).
        match s.num_c {
            1 => w.write(2, 0b00),
            2 => w.write(2, 0b01),
            3 => w.write(2, 0b10),
            n => {
                w.write(2, 0b11);
                w.write(4, (n - 4) as u64);
            }
        }
    }
}

/// Progressive-lossless single-group path. Applies the alternating Squeeze
/// pyramid to the RCT'd channels, then tokenizes the resulting channels with
/// fixed Weighted prediction in Fast mode or adaptive prediction in Slow mode.
/// The decoder reconstructs the input exactly through inverse-Squeeze + inverse-RCT.
fn encode_squeeze_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
    // Lift the 3 RCT'd planes (and alpha, if present) into channels.
    let num_c = if alpha.is_some() { 4 } else { 3 };
    let mut channels: Vec<Channel> = Vec::with_capacity(num_c);
    for c in 0..3usize {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            let row = linear.plane_row(c, y);
            ch.data[y * xsize..y * xsize + xsize].copy_from_slice(&row[..xsize]);
        }
        channels.push(ch);
    }
    if let Some(a) = alpha {
        let mut ch = Channel::new(xsize, ysize);
        match a {
            AlphaPlane::U8(data) => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    for (dst, &src) in row[..xsize].iter_mut().zip(src_row.iter()) {
                        *dst = src as i32;
                    }
                }
            }
            AlphaPlane::U16 { data, bits: _ } => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    for (dst, &src) in row[..xsize].iter_mut().zip(src_row.iter()) {
                        *dst = src as i32;
                    }
                }
            }
            AlphaPlane::F32(data) => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    row[..xsize].copy_from_slice(&src_row[..xsize]);
                }
            }
        }
        channels.push(ch);
    }
    // Apply the alternating H/V pyramid over all channels (explicit sequence).
    let steps = default_squeeze_steps(xsize, ysize, num_c);
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();

    // Slow searches the useful general and directional predictors per channel;
    // Fast avoids the analysis pass and uses fixed Weighted prediction.
    let predictors: Vec<u32> = if speed == crate::Speed::Slow {
        pool.steal_map(scratch, channels.len(), |c, _scratch| {
            let ch = &channels[c];
            let data = &ch.data;
            let w = ch.w;
            let get = move |gx: usize, gy: usize| data[gy * w + gx];
            choose_predictor_for_plane(get, ch.w, ch.h)
        })
    } else {
        vec![PREDICTOR_WEIGHTED; nb]
    };
    let channel_tokens = pool.steal_map(scratch, nb, |c, scratch| {
        let ch = &channels[c];
        let mut tokens = Vec::with_capacity(ch.w * ch.h);
        tokenize_plane(
            channel_to_context(c, nb),
            |x, y| ch.data[y * ch.w + x],
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

    write_frame_header_modular(alpha.is_some(), writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut section);

    let distance_ctx = nb as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
    let code = build_lz_pixel_code(
        std::iter::once(lz_tokens.as_slice()),
        nb,
        min_symbol,
        speed == crate::Speed::Slow,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_local_tree_lz77(
        &predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();

    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(std::slice::from_mut(&mut section));
    writer.zero_pad_to_byte();
}

/// Stage-2 progressive lossless, multi-group (image > one AC group).
///
/// Applies the Squeeze pyramid to the whole frame, then splits channels per the
/// JXL rule: channels that fit a group form the global modular image (LfGlobal /
/// section 0); larger channels are partitioned into AC groups, each carrying the
/// `frame_rect >> shift` crop of every large channel. A single global tree (chain
/// on the channel property) serves all streams; the within-group channel index is
/// what the tree sees, so group crops share contexts with the leading globals.
#[derive(Default)]
struct SqueezePredictorCost {
    costs: PredictorCosts,
}

impl SqueezePredictorCost {
    /// Add one independently predicted modular crop. Weighted prediction state
    /// resets here because the decoder resets it for every modular sub-image.
    fn add_crop(&mut self, get: impl Fn(usize, usize) -> i32, w: usize, h: usize) {
        if w == 0 || h == 0 {
            return;
        }
        let mut wp = WpState::new(w);
        for y in 0..h {
            for x in 0..w {
                let value = get(x, y) as i64;
                let neighbors = predictor_neighbors(&get, x, y, w);
                let weighted = wp.predict(
                    x,
                    y,
                    neighbors.top,
                    neighbors.left,
                    neighbors.top_right,
                    neighbors.top_left,
                    neighbors.top_top,
                );
                self.costs.add(value, neighbors, weighted);
                wp.update(value, x, y);
            }
        }
    }

    fn predictor(&self) -> u32 {
        self.costs.best_predictor()
    }
}

#[allow(clippy::too_many_arguments)]
fn for_each_squeeze_group_crop(
    channels: &[crate::squeeze::Channel],
    split: usize,
    gdim: usize,
    gx: usize,
    gy: usize,
    minsh: i32,
    maxsh: i32,
    mut visit: impl FnMut(usize, usize, usize, usize, usize, usize),
) {
    let mut within = 0usize;
    for c in split..channels.len() {
        let ch = &channels[c];
        let msh = ch.hshift.min(ch.vshift);
        if msh < minsh || msh > maxsh {
            continue;
        }
        let hs = ch.hshift as usize;
        let vs = ch.vshift as usize;
        let rx0 = (gx * gdim) >> hs;
        let ry0 = (gy * gdim) >> vs;
        if rx0 >= ch.w || ry0 >= ch.h {
            continue;
        }
        let rw = (gdim >> hs).min(ch.w - rx0);
        let rh = (gdim >> vs).min(ch.h - ry0);
        if rw == 0 || rh == 0 {
            continue;
        }
        visit(within, c, rx0, ry0, rw, rh);
        within += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_squeeze_multigroup(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    xsize_groups: usize,
    ysize_groups: usize,
    xsize_dc_groups: usize,
    ysize_dc_groups: usize,
    num_dc_groups: usize,
    num_ac_groups: usize,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
    let num_c = if alpha.is_some() { 4 } else { 3 };
    let mut channels: Vec<Channel> = Vec::with_capacity(num_c);
    for c in 0..3usize {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            let row = linear.plane_row(c, y);
            ch.data[y * xsize..y * xsize + xsize].copy_from_slice(&row[..xsize]);
        }
        channels.push(ch);
    }
    if let Some(a) = alpha {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            for x in 0..xsize {
                ch.data[y * xsize + x] = a.get_i32(y * xsize + x);
            }
        }
        channels.push(ch);
    }
    let steps = default_squeeze_steps(xsize, ysize, num_c);
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();

    // Channel -> stream split (JXL rule): the leading run of channels that fit a
    // group is the global modular image; the first channel still larger than a
    // group starts the suffix that is partitioned into AC groups.
    let split = channels
        .iter()
        .position(|c| c.w > GROUP_DIM || c.h > GROUP_DIM)
        .unwrap_or(nb);

    let predictors: Vec<u32> = if speed == crate::Speed::Slow {
        // The MA tree sees a channel's position inside each modular sub-image,
        // not its index in `channels`. Pool costs by that decoder-visible slot.
        let mut costs: Vec<SqueezePredictorCost> =
            (0..nb).map(|_| SqueezePredictorCost::default()).collect();
        for c in 0..split {
            let ch = &channels[c];
            costs[c].add_crop(|x, y| ch.data[y * ch.w + x], ch.w, ch.h);
        }
        for gy in 0..ysize_dc_groups {
            for gx in 0..xsize_dc_groups {
                for_each_squeeze_group_crop(
                    &channels,
                    split,
                    LF_GROUP_DIM,
                    gx,
                    gy,
                    3,
                    1000,
                    |within, c, x0, y0, w, h| {
                        let ch = &channels[c];
                        costs[within].add_crop(|x, y| ch.data[(y0 + y) * ch.w + x0 + x], w, h);
                    },
                );
            }
        }
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                for_each_squeeze_group_crop(
                    &channels,
                    split,
                    GROUP_DIM,
                    gx,
                    gy,
                    0,
                    2,
                    |within, c, x0, y0, w, h| {
                        let ch = &channels[c];
                        costs[within].add_crop(|x, y| ch.data[(y0 + y) * ch.w + x0 + x], w, h);
                    },
                );
            }
        }
        costs.iter().map(SqueezePredictorCost::predictor).collect()
    } else {
        vec![PREDICTOR_WEIGHTED; nb]
    };

    let distance_ctx = nb as u32;

    // Global stream: the small channels [0, split), whole.
    let global_channel_tokens = pool.steal_map(scratch, split, |c, scratch| {
        let ch = &channels[c];
        let ctx = channel_to_context(c, nb);
        let data = &ch.data;
        let w = ch.w;
        let get = move |gx: usize, gy: usize| data[gy * w + gx];
        let mut tokens = Vec::with_capacity(ch.w * ch.h);
        tokenize_plane(
            ctx,
            get,
            ch.w,
            ch.h,
            predictors[c],
            grad_pack_fn,
            &mut scratch.gradient,
            &mut tokens,
        );
        tokens
    });
    let mut global_tokens: Vec<Token> = Vec::new();
    for channel in global_channel_tokens {
        global_tokens.extend(channel);
    }
    let global_lz = lz77_compress_for_speed(&global_tokens, distance_ctx, speed, scratch);

    // One group's worth of cropped large-channel tokens. `gdim` is the group's
    // frame-space size (GROUP_DIM for AC, LF_GROUP_DIM for DC); a channel is
    // included when min(hshift,vshift) is inside [minsh,maxsh], and contributes
    // its `(group_rect >> shift)` crop. The decoder rebuilds the same scan, so
    // the within-group index (sequential over non-empty crops) is exactly the
    // `chan` property the global tree keys on.
    let deep_lz = (speed == crate::Speed::Slow)
        .then(|| DeepLzScratchPool::new(group_lz_threads(speed, pool)));
    let crop_group = |gdim: usize,
                      gx: usize,
                      gy: usize,
                      minsh: i32,
                      maxsh: i32,
                      scratch: &mut CoderScratch|
     -> Vec<LzToken> {
        let mut gtok: Vec<Token> = Vec::new();
        for_each_squeeze_group_crop(
            &channels,
            split,
            gdim,
            gx,
            gy,
            minsh,
            maxsh,
            |within, c, rx0, ry0, rw, rh| {
                let ch = &channels[c];
                let ctx = channel_to_context(within, nb);
                let pred = predictors[within];
                let data = &ch.data;
                let w = ch.w;
                let get = move |lx: usize, ly: usize| data[(ry0 + ly) * w + (rx0 + lx)];
                tokenize_plane(
                    ctx,
                    get,
                    rw,
                    rh,
                    pred,
                    grad_pack_fn,
                    &mut scratch.gradient,
                    &mut gtok,
                );
            },
        );
        if let Some(deep_lz) = &deep_lz {
            deep_lz.with_depth(|depth| {
                lz77_compress_for_speed_with_depth(&gtok, distance_ctx, speed, depth, scratch)
            })
        } else {
            lz77_compress_for_speed(&gtok, distance_ctx, speed, scratch)
        }
    };

    // DC (LF) groups carry the deeply-squeezed large channels (min shift >= 3),
    // partitioned into LF_GROUP_DIM rects. Empty unless the image is large enough
    // that a >=3x-squeezed channel still exceeds a group (dimension > ~2048).
    let dc_group_lz = pool.steal_map_with_threads(
        scratch,
        num_dc_groups,
        group_lz_threads(speed, pool),
        |group_index, scratch| {
            let gx = group_index % xsize_dc_groups;
            let gy = group_index / xsize_dc_groups;
            crop_group(LF_GROUP_DIM, gx, gy, 3, 1000, scratch)
        },
    );

    // AC groups carry the shallow large channels (min shift <= 2), in GROUP_DIM rects.
    let ac_group_lz = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(speed, pool),
        |group_index, scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            crop_group(GROUP_DIM, gx, gy, 0, 2, scratch)
        },
    );
    drop(deep_lz);

    let code = build_lz_pixel_code(
        std::iter::once(global_lz.as_slice())
            .chain(dc_group_lz.iter().map(Vec::as_slice))
            .chain(ac_group_lz.iter().map(Vec::as_slice)),
        nb,
        min_symbol,
        speed == crate::Speed::Slow,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    write_frame_header_modular(alpha.is_some(), writer);

    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // ----- Section 0: LfGlobal = global tree + global modular image -----
    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_local_tree_lz77(
        &predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree = 1
    sections[0].write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut sections[0]);
    write_lz_section(
        &global_lz,
        distance_ctx,
        &code,
        min_symbol,
        &mut sections[0],
    );
    sections[0].zero_pad_to_byte();

    // ----- DC groups: GroupHeader + any min-shift>=3 large-channel crops -----
    for (k, section) in sections[1..num_dc_groups + 1].iter_mut().enumerate() {
        let i = k + 1;
        section.write(1, 1); // use_global_tree
        section.write(1, 1); // wp_default
        section.write(2, 0); // 0 transforms (declared globally)
        write_lz_section(&dc_group_lz[i], distance_ctx, &code, min_symbol, section);
        section.zero_pad_to_byte();
    }

    // ----- AC global: trivial -----
    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    // ----- AC groups: GroupHeader + the cropped large-channel tokens -----
    for g in 0..num_ac_groups {
        let idx = 2 + num_dc_groups + g;
        sections[idx].write(1, 1); // use_global_tree
        sections[idx].write(1, 1); // wp_default
        sections[idx].write(2, 0); // 0 transforms (declared globally)
        write_lz_section(
            &ac_group_lz[g],
            distance_ctx,
            &code,
            min_symbol,
            &mut sections[idx],
        );
        sections[idx].zero_pad_to_byte();
    }

    // TOC + sections.
    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}

/// Serialize a single Palette transform (mirrors `Transform::VisitFields` for
/// `TransformId::kPalette`). v1: begin_c=0, num_c=3, nb_deltas=0, predictor=Zero.
fn write_palette_transform(num_c: u32, nb_colors: u32, w: &mut BitWriter) {
    debug_assert!(matches!(num_c, 3 | 4));
    // transforms count U32(Val(0),Val(1),...): selector 1 = Val(1) = 1 transform.
    w.write(2, 0b01);
    // id U32(Val(RCT=0),Val(Palette=1),...): selector 1 = Palette.
    w.write(2, 0b01);
    // begin_c U32(Bits(3),...): selector 0 = Bits(3), value 0.
    w.write(2, 0b00);
    w.write(3, 0);
    // num_c U32(Val(1),Val(3),Val(4),BitsOffset(13,1)).
    w.write(2, if num_c == 3 { 0b01 } else { 0b10 });
    // nb_colors U32(BitsOffset(8,0),BitsOffset(10,256),...).
    if nb_colors <= 255 {
        w.write(2, 0b00); // selector 0 = BitsOffset(8, 0)
        w.write(8, nb_colors as u64);
    } else {
        // nb_colors == 256 (our cap): selector 1 = BitsOffset(10, 256).
        w.write(2, 0b01);
        w.write(10, (nb_colors - 256) as u64);
    }
    // nb_deltas U32(Val(0),...): selector 0 = Val(0) = 0.
    w.write(2, 0b00);
    // predictor Bits(4): Predictor::Zero = 0.
    w.write(4, 0);
}

/// Single-group RGB/RGBA exact palette path. Encodes a palette meta-channel plus
/// an index channel, declaring a Palette transform so `InvPalette` reconstructs
/// the original channels directly (no RCT).
fn try_encode_palette_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    use std::collections::HashMap;
    let npx = xsize * ysize;
    let num_c = 3 + usize::from(alpha.is_some());

    // 1) Reconstruct RGB from YCoCg, collect distinct tuples (bail past 256).
    let plane0 = linear.plane_data(0);
    let plane1 = linear.plane_data(1);
    let plane2 = linear.plane_data(2);
    let color_at = |gx: usize, gy: usize| {
        let (r, g, b) = inverse_ycocg(
            plane0[gy * xsize + gx],
            plane1[gy * xsize + gx],
            plane2[gy * xsize + gx],
        );
        [r, g, b, alpha.map_or(0, |a| a.get_i32(gy * xsize + gx))]
    };
    let mut seen: HashMap<[i32; 4], ()> = HashMap::with_capacity(257);
    for gy in 0..ysize {
        for gx in 0..xsize {
            seen.entry(color_at(gx, gy)).or_insert(());
            if seen.len() > 256 {
                return false;
            }
        }
    }
    let nb_colors = seen.len();
    if nb_colors == 0 {
        return false;
    }

    // 2) Sorted palette + color->index map.
    let mut colors: Vec<[i32; 4]> = seen.keys().copied().collect();
    colors.sort_unstable();
    let mut idx_of: HashMap<[i32; 4], u32> = HashMap::with_capacity(nb_colors);
    for (i, c) in colors.iter().enumerate() {
        idx_of.insert(*c, i as u32);
    }

    // 3) Palette meta-channel (row c = component c of each color) + index channel.
    let mut palette_ch = vec![0i32; num_c * nb_colors];
    for (i, color) in colors.iter().enumerate() {
        for c in 0..num_c {
            palette_ch[c * nb_colors + i] = color[c];
        }
    }
    let mut index_img = Vec::with_capacity(npx);
    for gy in 0..ysize {
        for gx in 0..xsize {
            index_img.push(idx_of[&color_at(gx, gy)] as i32);
        }
    }

    // 4) Channel accessors (channel 0 = palette, channel 1 = index).
    let pget = |gx: usize, gy: usize| palette_ch[gy * nb_colors + gx];
    let iget = |gx: usize, gy: usize| index_img[gy * xsize + gx];

    // 5) Slow searches predictors per channel; Fast stays fixed Weighted.
    let preds = if speed == crate::Speed::Slow {
        [
            choose_predictor_for_plane(pget, nb_colors, num_c),
            choose_predictor_for_plane(iget, xsize, ysize),
        ]
    } else {
        [PREDICTOR_WEIGHTED; 2]
    };

    // 6) Frame header + single section (mirrors the RGB single-group layout).
    write_frame_header_modular(alpha.is_some(), writer);

    let nb_chans = 2usize;
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_palette_transform(num_c as u32, nb_colors as u32, &mut section);

    let mut tokens: Vec<Token> = Vec::with_capacity(num_c * nb_colors + npx);
    tokenize_plane(
        channel_to_context(0, nb_chans),
        pget,
        nb_colors,
        num_c,
        preds[0],
        grad_pack_fn,
        &mut scratch.gradient,
        &mut tokens,
    );
    tokenize_plane(
        channel_to_context(1, nb_chans),
        iget,
        xsize,
        ysize,
        preds[1],
        grad_pack_fn,
        &mut scratch.gradient,
        &mut tokens,
    );

    let distance_ctx = nb_chans as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch);
    let code = build_lz_pixel_code(
        std::iter::once(lz_tokens.as_slice()),
        nb_chans,
        min_symbol,
        speed == crate::Speed::Slow,
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

    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(std::slice::from_mut(&mut section));
    writer.zero_pad_to_byte();
    true
}

struct LocalPaletteGroup {
    palette: Vec<i32>,
    indices: Vec<i32>,
    nb_colors: usize,
    w: usize,
    h: usize,
}

fn estimated_local_stream_bits(
    tokens: &[Token],
    predictors: &[u32],
    num_contexts: usize,
    min_symbol: u32,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> usize {
    let lz = lz77_compress_runs(tokens);
    estimated_local_lz_stream_bits(&lz, predictors, num_contexts, min_symbol, speed, scratch)
}

fn estimated_local_lz_stream_bits(
    lz: &[LzToken],
    predictors: &[u32],
    num_contexts: usize,
    min_symbol: u32,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> usize {
    let code = build_lz_pixel_code(
        std::iter::once(lz),
        num_contexts,
        min_symbol,
        speed == crate::Speed::Slow,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let mut writer = BitWriter::new();
    write_local_tree_lz77(
        predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut writer,
    );
    write_lz_section(lz, num_contexts as u32, &code, min_symbol, &mut writer);
    writer.bits_written()
}

#[allow(clippy::too_many_arguments)]
fn local_palette_is_better(
    palette: &LocalPaletteGroup,
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> bool {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let palette_predictors = if speed == crate::Speed::Slow {
        [
            choose_predictor_for_plane(
                |x, y| palette.palette[y * palette.nb_colors + x],
                palette.nb_colors,
                nb_chans,
            ),
            choose_predictor_for_plane(
                |x, y| palette.indices[y * palette.w + x],
                palette.w,
                palette.h,
            ),
        ]
    } else {
        [PREDICTOR_WEIGHTED; 2]
    };
    let mut palette_tokens =
        Vec::with_capacity(nb_chans * palette.nb_colors + palette.w * palette.h);
    tokenize_plane(
        channel_to_context(0, 2),
        |x, y| palette.palette[y * palette.nb_colors + x],
        palette.nb_colors,
        nb_chans,
        palette_predictors[0],
        grad_pack_fn,
        &mut scratch.gradient,
        &mut palette_tokens,
    );
    tokenize_plane(
        channel_to_context(1, 2),
        |x, y| palette.indices[y * palette.w + x],
        palette.w,
        palette.h,
        palette_predictors[1],
        grad_pack_fn,
        &mut scratch.gradient,
        &mut palette_tokens,
    );
    let mut palette_transform = BitWriter::new();
    write_palette_transform(
        nb_chans as u32,
        palette.nb_colors as u32,
        &mut palette_transform,
    );
    let palette_bits = estimated_local_stream_bits(
        &palette_tokens,
        &palette_predictors,
        2,
        min_symbol,
        speed,
        scratch,
    ) + palette_transform.bits_written();

    let plain_predictors: Vec<u32> = if speed == crate::Speed::Slow {
        (0..nb_chans)
            .map(|chan| {
                if chan < 3 {
                    let plane = linear.plane_data(chan);
                    choose_predictor_for_plane(
                        |x, y| plane[(y0 + y) * xsize + x0 + x],
                        palette.w,
                        palette.h,
                    )
                } else {
                    let alpha = alpha.expect("alpha channel must exist");
                    choose_predictor_for_plane(
                        |x, y| alpha.get_i32((y0 + y) * xsize + x0 + x),
                        palette.w,
                        palette.h,
                    )
                }
            })
            .collect()
    } else {
        vec![PREDICTOR_WEIGHTED; nb_chans]
    };
    let plain_lz = tokenize_runs_with_wp(
        linear,
        alpha,
        xsize,
        x0,
        y0,
        palette.w,
        palette.h,
        3,
        &plain_predictors,
        grad_pack_fn,
        scratch,
        WpParams::DEFAULT,
    );
    let plain_bits = estimated_local_lz_stream_bits(
        &plain_lz,
        &plain_predictors,
        nb_chans,
        min_symbol,
        speed,
        scratch,
    ) + 2; // zero-transform count

    palette_bits < plain_bits
}

fn build_local_palette_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Option<LocalPaletteGroup> {
    use std::collections::HashMap;

    let num_c = 3 + usize::from(alpha.is_some());
    let num_pixels = w * h;
    let plane0 = linear.plane_data(0);
    let plane1 = linear.plane_data(1);
    let plane2 = linear.plane_data(2);
    let tuple_at = |gx: usize, gy: usize| {
        [
            plane0[gy * xsize + gx],
            plane1[gy * xsize + gx],
            plane2[gy * xsize + gx],
            alpha.map_or(0, |a| a.get_i32(gy * xsize + gx)),
        ]
    };

    // Most photographic groups exceed the palette limit almost immediately.
    // Probe only the distinct set first, avoiding a full-group tuple allocation
    // on that overwhelmingly common rejection path.
    let mut seen = HashMap::<[i32; 4], ()>::with_capacity(257);
    for y in 0..h {
        for x in 0..w {
            let gx = x0 + x;
            let gy = y0 + y;
            seen.entry(tuple_at(gx, gy)).or_insert(());
            if seen.len() > 256 {
                return None;
            }
        }
    }

    let nb_colors = seen.len();
    // Palette coding replaces `num_c * num_pixels` samples with one index per
    // pixel plus `num_c * nb_colors` palette samples. Leave a small margin for
    // the transform header and altered entropy statistics.
    let palette_samples = num_pixels + num_c * nb_colors + 16;
    if nb_colors == 0 || palette_samples >= num_c * num_pixels {
        return None;
    }

    let mut colors: Vec<[i32; 4]> = seen.keys().copied().collect();
    colors.sort_unstable();
    let mut index_of = HashMap::<[i32; 4], i32>::with_capacity(nb_colors);
    let mut palette = vec![0i32; num_c * nb_colors];
    for (index, color) in colors.iter().enumerate() {
        index_of.insert(*color, index as i32);
        for c in 0..num_c {
            palette[c * nb_colors + index] = color[c];
        }
    }
    let mut indices = Vec::with_capacity(num_pixels);
    for y in 0..h {
        for x in 0..w {
            indices.push(index_of[&tuple_at(x0 + x, y0 + y)]);
        }
    }

    Some(LocalPaletteGroup {
        palette,
        indices,
        nb_colors,
        w,
        h,
    })
}

#[allow(clippy::too_many_arguments)]
fn try_encode_local_palette_multi_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    xsize_groups: usize,
    ysize_groups: usize,
    num_dc_groups: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let num_ac_groups = xsize_groups * ysize_groups;
    let palettes = if speed == crate::Speed::Slow {
        let mut palettes = pool.steal_map(scratch, num_ac_groups, |group_index, _scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let w = GROUP_DIM.min(xsize - x0);
            let h = GROUP_DIM.min(ysize - y0);
            build_local_palette_group(linear, alpha, xsize, x0, y0, w, h)
        });
        let palette_pixels: usize = palettes
            .iter()
            .flatten()
            .map(|palette| palette.w * palette.h)
            .sum();
        if !local_palette_coverage_is_sufficient(palette_pixels, xsize * ysize) {
            return false;
        }

        let palette_is_better = pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
            let Some(palette) = &palettes[group_index] else {
                return false;
            };
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            local_palette_is_better(
                palette,
                linear,
                alpha,
                xsize,
                gx * GROUP_DIM,
                gy * GROUP_DIM,
                min_symbol,
                grad_pack_fn,
                speed,
                scratch,
            )
        });
        for (palette, keep) in palettes.iter_mut().zip(palette_is_better) {
            if !keep {
                *palette = None;
            }
        }
        palettes
    } else {
        pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let w = GROUP_DIM.min(xsize - x0);
            let h = GROUP_DIM.min(ysize - y0);
            let palette = build_local_palette_group(linear, alpha, xsize, x0, y0, w, h)?;
            local_palette_is_better(
                &palette,
                linear,
                alpha,
                xsize,
                x0,
                y0,
                min_symbol,
                grad_pack_fn,
                speed,
                scratch,
            )
            .then_some(palette)
        })
    };
    if !palettes.iter().any(Option::is_some) {
        return false;
    }

    // The global MA tree sees group-local channel slots. Pool predictor costs
    // for palette/index channels and ordinary YCoCg(A) channels by those slots.
    let predictors: Vec<u32> = if speed == crate::Speed::Slow {
        pool.steal_map(scratch, nb_chans, |slot, _scratch| {
            let mut cost = SqueezePredictorCost::default();
            for (group_index, palette) in palettes.iter().enumerate() {
                if let Some(palette) = palette {
                    if slot == 0 {
                        cost.add_crop(
                            |x, y| palette.palette[y * palette.nb_colors + x],
                            palette.nb_colors,
                            nb_chans,
                        );
                    } else if slot == 1 {
                        cost.add_crop(
                            |x, y| palette.indices[y * palette.w + x],
                            palette.w,
                            palette.h,
                        );
                    }
                    continue;
                }

                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let w = GROUP_DIM.min(xsize - x0);
                let h = GROUP_DIM.min(ysize - y0);
                if slot < 3 {
                    let plane = linear.plane_data(slot);
                    cost.add_crop(|x, y| plane[(y0 + y) * xsize + x0 + x], w, h);
                } else {
                    let alpha = alpha.expect("alpha slot requires alpha channel");
                    cost.add_crop(|x, y| alpha.get_i32((y0 + y) * xsize + x0 + x), w, h);
                }
            }
            cost.predictor()
        })
    } else {
        vec![PREDICTOR_WEIGHTED; nb_chans]
    };

    let distance_ctx = nb_chans as u32;
    let deep_lz = (speed == crate::Speed::Slow)
        .then(|| DeepLzScratchPool::new(group_lz_threads(speed, pool)));
    let group_lz_tokens: Vec<Vec<LzToken>> = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(speed, pool),
        |group_index, scratch| {
            if let Some(palette) = &palettes[group_index] {
                let mut tokens =
                    Vec::with_capacity(nb_chans * palette.nb_colors + palette.w * palette.h);
                tokenize_plane(
                    channel_to_context(0, nb_chans),
                    |x, y| palette.palette[y * palette.nb_colors + x],
                    palette.nb_colors,
                    nb_chans,
                    predictors[0],
                    grad_pack_fn,
                    &mut scratch.gradient,
                    &mut tokens,
                );
                tokenize_plane(
                    channel_to_context(1, nb_chans),
                    |x, y| palette.indices[y * palette.w + x],
                    palette.w,
                    palette.h,
                    predictors[1],
                    grad_pack_fn,
                    &mut scratch.gradient,
                    &mut tokens,
                );
                if let Some(deep_lz) = &deep_lz {
                    deep_lz.with_depth(|depth| {
                        lz77_compress_for_speed_with_depth(
                            &tokens,
                            distance_ctx,
                            speed,
                            depth,
                            scratch,
                        )
                    })
                } else {
                    lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch)
                }
            } else {
                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let w = GROUP_DIM.min(xsize - x0);
                let h = GROUP_DIM.min(ysize - y0);
                if speed == crate::Speed::Slow {
                    let channel_tokens = tokenize_channels_with_wp(
                        linear,
                        alpha,
                        xsize,
                        ysize,
                        x0,
                        y0,
                        w,
                        h,
                        3,
                        &predictors,
                        grad_pack_fn,
                        pool,
                        scratch,
                        WpParams::DEFAULT,
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
                        w,
                        h,
                        3,
                        &predictors,
                        grad_pack_fn,
                        scratch,
                        WpParams::DEFAULT,
                    )
                }
            }
        },
    );
    drop(deep_lz);

    let code = build_lz_pixel_code(
        group_lz_tokens.iter().map(Vec::as_slice),
        nb_chans,
        min_symbol,
        speed == crate::Speed::Slow,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    write_frame_header_modular(alpha.is_some(), writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default
    sections[0].write(1, 1); // has global tree
    write_local_tree_lz77(
        &predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    sections[0].write(1, 1); // wp_default
    write_modular_transforms(nb_chans, 6, &mut sections[0]);
    sections[0].zero_pad_to_byte();

    for section in sections[1..num_dc_groups + 1].iter_mut() {
        section.write(1, 1);
        section.write(1, 1);
        section.write(2, 0);
        section.zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    for group_index in 0..num_ac_groups {
        let section_idx = 2 + num_dc_groups + group_index;
        sections[section_idx].write(1, 1); // use_global_tree
        sections[section_idx].write(1, 1); // wp_default
        if let Some(palette) = &palettes[group_index] {
            write_palette_transform(
                nb_chans as u32,
                palette.nb_colors as u32,
                &mut sections[section_idx],
            );
        } else {
            sections[section_idx].write(2, 0); // no local transforms
        }
        write_lz_section(
            &group_lz_tokens[group_index],
            distance_ctx,
            &code,
            min_symbol,
            &mut sections[section_idx],
        );
        sections[section_idx].zero_pad_to_byte();
    }

    writer.write(1, 0); // no TOC permutation
    writer.zero_pad_to_byte();
    for section in &sections {
        write_toc_entry(section.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}

fn write_toc_entry(byte_len: usize, w: &mut BitWriter) {
    static OFFSETS: [usize; 4] = [0, 1024, 17_408, 4_211_712];
    static BITS: [usize; 4] = [10, 14, 22, 30];
    let mut bucket = 0usize;
    while bucket < 3 && byte_len >= OFFSETS[bucket + 1] {
        bucket += 1;
    }
    w.write(2, bucket as u64);
    w.write(BITS[bucket], (byte_len - OFFSETS[bucket]) as u64);
}

// ---------------------------------------------------------------------------
// Forward reversible YCoCg (RCT type 6, matches libjxl's InvRCTRow<6>).
//
// Encoder:
//   co  = r - b
//   tmp = b + (co >> 1)
//   cg  = g - tmp
//   y   = tmp + (cg >> 1)
//
// Decoder undoes this with the exact same shift sequence:
//   tmp = y - (cg >> 1);  g = cg + tmp;
//   y'  = tmp - (co >> 1); r = y' + co;  b = y'
//
// Reversible because every operation is invertible without rounding.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn forward_ycocg(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    (y, co, cg)
}

/// Exact integer inverse of `forward_ycocg` (undoes the steps in reverse).
#[inline]
pub(crate) fn inverse_ycocg(y: i32, co: i32, cg: i32) -> (i32, i32, i32) {
    let tmp = y - (cg >> 1);
    let g = cg + tmp;
    let b = tmp - (co >> 1);
    let r = b + co;
    (r, g, b)
}

#[inline]
fn channel_to_context(chan: usize, nb_chans: usize) -> u32 {
    (nb_chans - 1 - chan) as u32
}

fn tokenize_all(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<Token> {
    tokenize_all_with_wp(
        linear,
        alpha,
        xsize,
        _ysize,
        x0,
        y0,
        gw,
        gh,
        num_color,
        predictors,
        grad_pack_fn,
        pool,
        scratch,
        WpParams::DEFAULT,
    )
}

fn tokenize_all_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
) -> Vec<Token> {
    let channel_tokens = tokenize_channels_with_wp(
        linear,
        alpha,
        xsize,
        _ysize,
        x0,
        y0,
        gw,
        gh,
        num_color,
        predictors,
        grad_pack_fn,
        pool,
        scratch,
        wp_params,
    );
    let total_len = channel_tokens.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total_len);
    for channel in channel_tokens {
        out.extend(channel);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn tokenize_channels_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
) -> Vec<Vec<Token>> {
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };
    pool.steal_map(scratch, nb_chans, |chan, scratch| {
        let mut out = Vec::with_capacity(gw * gh);
        tokenize_channel_with_wp(
            linear,
            alpha,
            xsize,
            x0,
            y0,
            gw,
            gh,
            num_color,
            predictors,
            chan,
            grad_pack_fn,
            &mut scratch.gradient,
            &mut out,
            wp_params,
        );
        out
    })
}

/// Tokenizes and run-compresses one channel at a time. This preserves channel
/// boundaries (and therefore the bitstream) while retaining only one raw-token
/// plane per worker. Multi-group encoding already supplies group-level
/// parallelism, so nested channel tasks would only increase live storage.
#[allow(clippy::too_many_arguments)]
fn tokenize_runs_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
) -> Vec<LzToken> {
    let nb_chans = num_color + usize::from(alpha.is_some());
    let mut out = RunLzWriter::with_capacity(gw * gh * nb_chans);
    for chan in 0..nb_chans {
        tokenize_channel_with_wp(
            linear,
            alpha,
            xsize,
            x0,
            y0,
            gw,
            gh,
            num_color,
            predictors,
            chan,
            grad_pack_fn,
            &mut scratch.gradient,
            &mut out,
            wp_params,
        );
        out.finish_channel();
    }
    out.finish()
}

#[allow(clippy::too_many_arguments)]
fn tokenize_channel_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    chan: usize,
    grad_pack_fn: GradPackInteriorFn,
    gradient: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    let nb_chans = num_color + usize::from(alpha.is_some());
    let ctx = channel_to_context(chan, nb_chans);
    if chan < num_color {
        if predictors[chan] == PREDICTOR_WEIGHTED {
            tokenize_wp_row_slices(
                ctx,
                |gy| &linear.plane_row(chan, y0 + gy)[x0..x0 + gw],
                gw,
                gh,
                out,
                wp_params,
            );
        } else {
            let get = |gx: usize, gy: usize| linear.plane_row(chan, y0 + gy)[x0 + gx];
            tokenize_plane_with_wp(
                ctx,
                get,
                gw,
                gh,
                predictors[chan],
                grad_pack_fn,
                gradient,
                out,
                wp_params,
            );
        }
        return;
    }

    let alpha = alpha.expect("alpha channel must exist");
    if predictors[chan] == PREDICTOR_WEIGHTED {
        match alpha {
            AlphaPlane::U8(data) => tokenize_wp_row_slices(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                out,
                wp_params,
            ),
            AlphaPlane::U16 { data, .. } => tokenize_wp_row_slices(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                out,
                wp_params,
            ),
            AlphaPlane::F32(data) => tokenize_wp_row_slices(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                out,
                wp_params,
            ),
        }
    } else {
        let get = |gx: usize, gy: usize| alpha.get_i32((y0 + gy) * xsize + (x0 + gx));
        tokenize_plane_with_wp(
            ctx,
            get,
            gw,
            gh,
            predictors[chan],
            grad_pack_fn,
            gradient,
            out,
            wp_params,
        );
    }
}

#[derive(Default)]
pub(crate) struct GradientScratch {
    pub(crate) cur: Vec<i32>,
    pub(crate) prev: Vec<i32>,
    pub(crate) prev_prev: Vec<i32>,
    pub(crate) buf: Vec<u32>,
}

/// Tokenize one channel's group-local rectangle with the chosen predictor
/// (`PREDICTOR_GRADIENT` or `PREDICTOR_WEIGHTED`). Neighbors use libjxl's exact
/// border conventions (see weighted::State / Predict()):
///   left = x>0 ? W : (y>0 ? N : 0); top = y>0 ? N : left;
///   topleft = (x&&y) ? NW : left; topright = (x+1<w && y) ? NE : top;
///   toptop  = y>1 ? NN : top.
fn tokenize_plane(
    ctx: u32,
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut Vec<Token>,
) {
    tokenize_plane_with_wp(
        ctx,
        get,
        gw,
        gh,
        pred_id,
        grad_pack_fn,
        scratch,
        out,
        WpParams::DEFAULT,
    );
}

trait TokenSink {
    fn push_token(&mut self, token: Token);
}

impl TokenSink for Vec<Token> {
    #[inline(always)]
    fn push_token(&mut self, token: Token) {
        self.push(token);
    }
}

impl TokenSink for RunLzWriter {
    #[inline(always)]
    fn push_token(&mut self, token: Token) {
        self.push(token);
    }
}

#[inline(always)]
fn push_wp_token(ctx: u32, value: i64, prediction: i64, out: &mut impl TokenSink) {
    out.push_token(Token::new(ctx, pack_signed((value - prediction) as i32)));
}

/// The first, second, and interior row kernels deliberately have separate
/// border handling. Besides keeping branches out of the interior loop, this is
/// the dispatch seam for a future vector implementation of the row arithmetic.
fn tokenize_wp_first_row<T: Copy>(
    ctx: u32,
    current: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    let row = wp.row_offsets(0);
    let mut left = 0i64;
    for (x, &value) in current.iter().enumerate() {
        let value = i64::from(value);
        let prediction = wp.predict_and_update(
            value,
            x,
            row,
            WpNeighbors {
                north: left,
                west: left,
                north_east: left,
                north_west: left,
                north_north: left,
            },
        );
        push_wp_token(ctx, value, prediction, out);
        left = value;
    }
}

fn tokenize_wp_second_row<T: Copy>(
    ctx: u32,
    current: &[T],
    north: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    let row = wp.row_offsets(1);
    for (x, &value) in current.iter().enumerate() {
        let value = i64::from(value);
        let n = i64::from(north[x]);
        let w = if x == 0 { n } else { i64::from(current[x - 1]) };
        let prediction = wp.predict_and_update(
            value,
            x,
            row,
            WpNeighbors {
                north: n,
                west: w,
                north_east: north.get(x + 1).map_or(n, |&v| i64::from(v)),
                north_west: if x == 0 { w } else { i64::from(north[x - 1]) },
                north_north: n,
            },
        );
        push_wp_token(ctx, value, prediction, out);
    }
}

fn tokenize_wp_interior_row<T: Copy>(
    ctx: u32,
    y: usize,
    current: &[T],
    north: &[T],
    north_north: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    let row = wp.row_offsets(y);
    for (x, &value) in current.iter().enumerate() {
        let value = i64::from(value);
        let n = i64::from(north[x]);
        let w = if x == 0 { n } else { i64::from(current[x - 1]) };
        let prediction = wp.predict_and_update(
            value,
            x,
            row,
            WpNeighbors {
                north: n,
                west: w,
                north_east: north.get(x + 1).map_or(n, |&v| i64::from(v)),
                north_west: if x == 0 { w } else { i64::from(north[x - 1]) },
                north_north: i64::from(north_north[x]),
            },
        );
        push_wp_token(ctx, value, prediction, out);
    }
}

fn tokenize_wp_row_slices<'a, T: Copy + 'a>(
    ctx: u32,
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) where
    i64: From<T>,
{
    if gw == 0 || gh == 0 {
        return;
    }
    let mut wp = WpState::with_params(gw, wp_params);
    let first = get_row(0);
    debug_assert_eq!(first.len(), gw);
    tokenize_wp_first_row(ctx, first, &mut wp, out);
    if gh == 1 {
        return;
    }
    let second = get_row(1);
    debug_assert_eq!(second.len(), gw);
    tokenize_wp_second_row(ctx, second, first, &mut wp, out);
    let mut north_north = first;
    let mut north = second;
    for y in 2..gh {
        let current = get_row(y);
        debug_assert_eq!(current.len(), gw);
        tokenize_wp_interior_row(ctx, y, current, north, north_north, &mut wp, out);
        north_north = north;
        north = current;
    }
}

fn tokenize_wp_rows_from_get(
    ctx: u32,
    get: &impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    if gw == 0 || gh == 0 {
        return;
    }
    if scratch.cur.len() < gw {
        scratch.cur.resize(gw, 0);
    }
    if scratch.prev.len() < gw {
        scratch.prev.resize(gw, 0);
    }
    if scratch.prev_prev.len() < gw {
        scratch.prev_prev.resize(gw, 0);
    }

    let mut current = &mut scratch.cur[..gw];
    let mut north = &mut scratch.prev[..gw];
    let mut north_north = &mut scratch.prev_prev[..gw];
    let mut wp = WpState::with_params(gw, wp_params);
    for y in 0..gh {
        if y != 0 {
            std::mem::swap(&mut north_north, &mut north);
            std::mem::swap(&mut north, &mut current);
        }
        for (x, value) in current.iter_mut().enumerate() {
            *value = get(x, y);
        }
        match y {
            0 => tokenize_wp_first_row(ctx, current, &mut wp, out),
            1 => tokenize_wp_second_row(ctx, current, north, &mut wp, out),
            _ => tokenize_wp_interior_row(ctx, y, current, north, north_north, &mut wp, out),
        }
    }
}

fn tokenize_plane_with_wp(
    ctx: u32,
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    if pred_id == PREDICTOR_WEIGHTED {
        tokenize_wp_rows_from_get(ctx, &get, gw, gh, scratch, out, wp_params);
    } else if pred_id == PREDICTOR_GRADIENT {
        // Gradient (ClampedGradient): per-pixel independent, pure integer ->
        // vectorized over the interior of each row.
        if scratch.buf.len() < gw {
            scratch.buf.resize(gw, 0);
        }
        if scratch.cur.len() < gw {
            scratch.cur.resize(gw, 0);
        }
        if scratch.prev.len() < gw {
            scratch.prev.resize(gw, 0);
        }
        let mut cur = &mut scratch.cur[..gw];
        let mut prev = &mut scratch.prev[..gw];
        let buf = &mut scratch.buf[..gw];
        for gy in 0..gh {
            std::mem::swap(&mut cur, &mut prev); // prev = last row's cur
            for (gx, c) in cur.iter_mut().enumerate() {
                *c = get(gx, gy);
            }
            if gy == 0 {
                buf[0] = pack_signed(cur[0]); // gx 0: pred = 0
                for gx in 1..gw {
                    buf[gx] = pack_signed(cur[gx].wrapping_sub(cur[gx - 1]));
                    // pred = W
                }
            } else {
                buf[0] = pack_signed(cur[0].wrapping_sub(prev[0])); // gx 0: pred = N
                grad_pack_fn(cur, prev, buf, gw); // gx in 1..gw
            }
            for &b in buf.iter().take(gw) {
                out.push_token(Token::new(ctx, b));
            }
        }
    } else {
        debug_assert!(matches!(
            pred_id,
            PREDICTOR_AVERAGE4 | PREDICTOR_SELECT | PREDICTOR_LEFT | PREDICTOR_TOP
        ));
        for gy in 0..gh {
            for gx in 0..gw {
                let value = get(gx, gy) as i64;
                let neighbors = predictor_neighbors(&get, gx, gy, gw);
                let pred = predictor_value(pred_id, neighbors, 0);
                out.push_token(Token::new(ctx, pack_signed((value - pred) as i32)));
            }
        }
    }
}
type GradPackInteriorFn = fn(&[i32], &[i32], &mut [u32], usize);
fn select_grad_pack_interior_fn() -> GradPackInteriorFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |c, p, o, g| unsafe { crate::avx::grad_pack_interior(c, p, o, g) };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |c, p, o, g| unsafe { crate::sse::grad_pack_interior(c, p, o, g) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |c, p, o, g| unsafe { crate::neon::grad_pack_interior(c, p, o, g) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::grad_pack_interior
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        grad_pack_interior_scalar
    }
}

static GRAD_PACK_INTERIOR_FN: OnceLock<GradPackInteriorFn> = OnceLock::new();

#[inline]
fn selected_grad_pack_interior_fn() -> GradPackInteriorFn {
    *GRAD_PACK_INTERIOR_FN.get_or_init(select_grad_pack_interior_fn)
}

#[allow(unused)]
fn grad_pack_interior_scalar(cur: &[i32], prev: &[i32], out: &mut [u32], gw: usize) {
    for gx in 1..gw {
        let w = cur[gx - 1];
        let n = prev[gx];
        let nw = prev[gx - 1];
        let ac = w.wrapping_sub(nw);
        let bc = n.wrapping_sub(nw);
        let grad = ac.wrapping_add(n);
        let clamp = if (w.wrapping_sub(n) ^ bc) < 0 { n } else { w };
        let pred = if (ac ^ bc) < 0 { grad } else { clamp };
        out[gx] = pack_signed(cur[gx].wrapping_sub(pred));
    }
}

#[derive(Default)]
struct PredictorCosts {
    histograms: [Vec<u64>; SLOW_PREDICTORS.len()],
    total: u64,
}

impl PredictorCosts {
    #[inline]
    fn add(&mut self, value: i64, neighbors: PredictorNeighbors, weighted: i64) {
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate() {
            let pred = predictor_value(pred_id, neighbors, weighted);
            let symbol = pack_signed((value - pred) as i32) as usize;
            let hist = &mut self.histograms[candidate];
            if hist.len() <= symbol {
                hist.resize(symbol + 1, 0);
            }
            hist[symbol] += 1;
        }
        self.total += 1;
    }

    fn best_predictor(&self) -> u32 {
        let mut best_id = SLOW_PREDICTORS[0];
        let mut best_bits = entropy_of_hist(&self.histograms[0], self.total);
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate().skip(1) {
            let bits = entropy_of_hist(&self.histograms[candidate], self.total);
            if bits < best_bits {
                best_bits = bits;
                best_id = pred_id;
            }
        }
        best_id
    }
}

/// Evaluate all Slow-mode predictors in one traversal and choose the lowest
/// order-0 residual entropy. Weighted remains the deterministic tie-breaker.
fn choose_predictor_for_plane(get: impl Fn(usize, usize) -> i32, w: usize, h: usize) -> u32 {
    choose_predictor_for_plane_with_wp(get, w, h, WpParams::DEFAULT)
}

fn choose_predictor_for_plane_with_wp(
    get: impl Fn(usize, usize) -> i32,
    w: usize,
    h: usize,
    wp_params: WpParams,
) -> u32 {
    if w == 0 || h == 0 {
        return PREDICTOR_WEIGHTED;
    }
    let mut wp = WpState::with_params(w, wp_params);
    let mut costs = PredictorCosts::default();
    for gy in 0..h {
        for gx in 0..w {
            let value = get(gx, gy) as i64;
            let neighbors = predictor_neighbors(&get, gx, gy, w);
            let weighted = wp.predict(
                gx,
                gy,
                neighbors.top,
                neighbors.left,
                neighbors.top_right,
                neighbors.top_left,
                neighbors.top_top,
            );
            costs.add(value, neighbors, weighted);
            wp.update(value, gx, gy);
        }
    }
    costs.best_predictor()
}

fn choose_predictors_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
) -> [u32; 4] {
    let mut preds = [PREDICTOR_WEIGHTED; 4];
    let num_channels = 3 + usize::from(alpha.is_some());
    let selected = pool.steal_map(scratch, num_channels, |chan, _scratch| {
        if chan < 3 {
            let pd = linear.plane_data(chan);
            choose_predictor_for_plane_with_wp(|x, y| pd[y * xsize + x], xsize, ysize, wp_params)
        } else {
            let a = alpha.expect("alpha channel must exist");
            choose_predictor_for_plane_with_wp(
                |x, y| a.get_i32(y * xsize + x),
                xsize,
                ysize,
                wp_params,
            )
        }
    });
    preds[..num_channels].copy_from_slice(&selected);
    if alpha.is_none() {
        preds[3] = PREDICTOR_WEIGHTED;
    }
    preds
}

// ---------------------------------------------------------------------------
// Context tree (MA tree): split each channel's entropy context on the WP error
// property kWPProp (p[15]) into 3 activity buckets. The decoder must run the WP
// state for every pixel of a channel whose subtree references p[15], so we run
// WP for all channels here regardless of the selected leaf predictor.
// ---------------------------------------------------------------------------

const PROP_WP: u32 = 15; // kNumStaticProperties(2) + 13

/// Encoder-side tree: Split(property, splitval, gt-branch, le-branch) routes
/// `prop > splitval` to the gt-branch. Leaf carries (predictor, tag) where
/// tag = channel*3 + bucket.
enum CtTree {
    Split(u32, i32, Box<CtTree>, Box<CtTree>),
    Leaf(u32, u32),
}

/// BFS-emit the tree (matches libjxl's FIFO tree decode) and return the context
/// id assigned to each leaf tag. Context id == leaf's order of appearance.
fn emit_ct_tree(root: &CtTree, out: &mut Vec<Token>) -> std::collections::HashMap<u32, u32> {
    use std::collections::{HashMap, VecDeque};
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut q: VecDeque<&CtTree> = VecDeque::new();
    q.push_back(root);
    let mut ctx = 0u32;
    while let Some(node) = q.pop_front() {
        match node {
            CtTree::Split(prop, val, gt, le) => {
                push_split(out, *prop, *val);
                q.push_back(gt);
                q.push_back(le);
            }
            CtTree::Leaf(pred, tag) => {
                push_leaf(out, *pred);
                map.insert(*tag, ctx);
                ctx += 1;
            }
        }
    }
    map
}

#[inline]
fn bucket_of(prop: i64, t: i64) -> u32 {
    if prop > t {
        2
    } else if prop > -t - 1 {
        1
    } else {
        0
    }
}

/// 3-leaf activity subtree for channel `c` with predictor `pred` and threshold `t`.
fn act_sub(c: u32, pred: u32, t: i32) -> CtTree {
    CtTree::Split(
        PROP_WP,
        t,
        Box::new(CtTree::Leaf(pred, c * 3 + 2)),
        Box::new(CtTree::Split(
            PROP_WP,
            -t - 1,
            Box::new(CtTree::Leaf(pred, c * 3 + 1)),
            Box::new(CtTree::Leaf(pred, c * 3)),
        )),
    )
}

/// Channel-split tree (same shape as build_balanced_tree_tokens) with each
/// channel-leaf replaced by its activity subtree.
fn build_context_tree(nb_chans: usize, preds: &[u32], t: &[i32]) -> CtTree {
    let a = |c: usize| act_sub(c as u32, preds[c], t[c]);
    match nb_chans {
        1 => a(0),
        2 => CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0))),
        3 => CtTree::Split(
            0,
            1,
            Box::new(a(2)),
            Box::new(CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0)))),
        ),
        4 => CtTree::Split(
            0,
            1,
            Box::new(CtTree::Split(0, 2, Box::new(a(3)), Box::new(a(2)))),
            Box::new(CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0)))),
        ),
        _ => unreachable!("context tree supports 1..=4 channels"),
    }
}

#[inline]
fn clamped_gradient(w: i64, n: i64, nw: i64) -> i64 {
    let lo = w.min(n);
    let hi = w.max(n);
    (w + n - nw).clamp(lo, hi)
}

fn order0_entropy(vals: &[u32], cell: &mut Vec<u64>) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    // Direct-indexed frequency histogram (residual symbols are small-range), in
    // place of a HashMap: no hashing, and a deterministic accumulation order.
    let max = vals.iter().copied().max().unwrap_or(0) as usize;
    if cell.len() < max + 1 {
        cell.resize(max + 1, 0);
    }
    let hist = &mut cell[..max + 1];
    hist.fill(0);
    for &v in vals {
        hist[v as usize] += 1;
    }
    let total = vals.len() as f32;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c != 0 {
            let p = c as f32 / total;
            bits -= c as f32 * dirty_log2f(p);
        }
    }
    bits
}

/// Run WP over one channel's group rectangle, returning per-pixel
/// (packed residual under `pred_id`, WP property p[15]) in row-major order.
fn collect_channel(
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    wp_params: WpParams,
) -> (Vec<u32>, Vec<i64>) {
    let mut wp = WpState::with_params(gw, wp_params);
    let mut res: Vec<u32> = Vec::with_capacity(gw * gh);
    let mut prp = Vec::with_capacity(gw * gh);
    for gy in 0..gh {
        for gx in 0..gw {
            let v = get(gx, gy) as i64;
            let neighbors = predictor_neighbors(&get, gx, gy, gw);
            let wp_pred = wp.predict(
                gx,
                gy,
                neighbors.top,
                neighbors.left,
                neighbors.top_right,
                neighbors.top_left,
                neighbors.top_top,
            );
            prp.push(wp.wp_prop);
            let pred = predictor_value(pred_id, neighbors, wp_pred);
            res.push(pack_signed((v - pred) as i32));
            wp.update(v, gx, gy);
        }
    }
    (res, prp)
}

/// Pick the best activity threshold for a channel among candidates, returning
/// (best_t, best_bucketed_bits, flat_bits).
fn entropy_of_hist(hist: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f32;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c != 0 {
            let p = c as f32 / t;
            bits -= c as f32 * p.log2();
        }
    }
    bits
}

#[derive(Default)]
pub(crate) struct PickThresholdScratch {
    pub(crate) hist_scratch: Vec<u64>,
}

impl PickThresholdScratch {
    fn make_scratches(&mut self, size: usize) -> (&mut [u64], &mut [u64], &mut [u64]) {
        let bucket = size + 1;
        if self.hist_scratch.len() < bucket * 3 {
            self.hist_scratch.resize(bucket * 3, 0);
        }
        let (b0, r0) = self.hist_scratch.split_at_mut(bucket);
        let (b1, b2) = r0.split_at_mut(bucket);
        (b0, b1, &mut b2[..bucket])
    }
}

fn pick_threshold(res: &[u32], prp: &[i64], scratch: &mut CoderScratch) -> (i32, f32, f32) {
    let flat = order0_entropy(res, &mut scratch.order0_entropy);
    let max = res.iter().copied().max().unwrap_or(0) as usize;
    let (h0, h1, h2) = scratch.threshold.make_scratches(max + 1);
    let mut best_t = 0i32;
    let mut best_bits = f32::INFINITY;
    for &t in &[8i64, 16, 24, 32, 48, 64, 96] {
        h0.fill(0);
        h1.fill(0);
        h2.fill(0);
        let (mut n0, mut n1, mut n2) = (0u64, 0u64, 0u64);
        for (&r, &p) in res.iter().zip(prp.iter()) {
            match bucket_of(p, t) {
                0 => {
                    h0[r as usize] += 1;
                    n0 += 1;
                }
                1 => {
                    h1[r as usize] += 1;
                    n1 += 1;
                }
                _ => {
                    h2[r as usize] += 1;
                    n2 += 1;
                }
            }
        }
        let bits = entropy_of_hist(h0, n0) + entropy_of_hist(h1, n1) + entropy_of_hist(h2, n2);
        if bits < best_bits {
            best_bits = bits;
            best_t = t as i32;
        }
    }
    (best_t, best_bits, flat)
}

/// Multi-group form of `pick_threshold` that scans the already-collected
/// group slices directly. This avoids rebuilding full-image residual and
/// property vectors for every channel solely to calculate histograms.
fn pick_threshold_grouped(
    groups: &[Vec<(Vec<u32>, Vec<i64>)>],
    channel: usize,
    scratch: &mut CoderScratch,
) -> (i32, f32, f32) {
    let total: usize = groups.iter().map(|group| group[channel].0.len()).sum();
    if total == 0 {
        return (0, 0.0, 0.0);
    }
    let max = groups
        .iter()
        .flat_map(|group| group[channel].0.iter().copied())
        .max()
        .unwrap_or(0) as usize;

    if scratch.order0_entropy.len() < max + 1 {
        scratch.order0_entropy.resize(max + 1, 0);
    }
    let flat_hist = &mut scratch.order0_entropy[..max + 1];
    flat_hist.fill(0);
    for group in groups {
        for &residual in &group[channel].0 {
            flat_hist[residual as usize] += 1;
        }
    }
    let total_f = total as f32;
    let mut flat = 0.0;
    for &count in flat_hist.iter() {
        if count != 0 {
            let p = count as f32 / total_f;
            flat -= count as f32 * dirty_log2f(p);
        }
    }

    let (h0, h1, h2) = scratch.threshold.make_scratches(max + 1);
    let mut best_t = 0i32;
    let mut best_bits = f32::INFINITY;
    for &t in &[8i64, 16, 24, 32, 48, 64, 96] {
        h0.fill(0);
        h1.fill(0);
        h2.fill(0);
        let (mut n0, mut n1, mut n2) = (0u64, 0u64, 0u64);
        for group in groups {
            let (residuals, properties) = &group[channel];
            for (&residual, &property) in residuals.iter().zip(properties.iter()) {
                match bucket_of(property, t) {
                    0 => {
                        h0[residual as usize] += 1;
                        n0 += 1;
                    }
                    1 => {
                        h1[residual as usize] += 1;
                        n1 += 1;
                    }
                    _ => {
                        h2[residual as usize] += 1;
                        n2 += 1;
                    }
                }
            }
        }
        let bits = entropy_of_hist(h0, n0) + entropy_of_hist(h1, n1) + entropy_of_hist(h2, n2);
        if bits < best_bits {
            best_bits = bits;
            best_t = t as i32;
        }
    }
    (best_t, best_bits, flat)
}

// ---------------------------------------------------------------------------
// Learned MA context tree (v2): greedy learned tree over the standard modular
// property vector (libjxl ids 0..=15) with per-leaf predictors. The learner
// lives in ma_tree.rs; this section computes the property vectors (bit-exact
// with libjxl's PredictImpl, which the decoder re-runs), samples the image,
// and routes every pixel through the learned tree.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RCT selection: YCoCg (type 6) is a strong default, but content with weak or
// atypical channel correlation prefers one of the other reversible transforms
// (subtract-green and half-average families). Candidates and cost model follow
// libjxl: fixed 18-context gradient-residual token entropy per channel.
// ---------------------------------------------------------------------------

/// Candidate rct_types in libjxl's try order (first 7 = cjxl -e7's set):
/// none, YCoCg, G-half-avg, subtract-green, and half-average variants.
const RCT_CANDIDATES: [u32; 7] = [0, 6, 5, 10, 26, 40, 12];
const RCT_CONTEXT_CUTOFFS: [u32; 17] = [
    0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];
const RCT_CONTEXT_MAX_DIFF: u32 = 500;
const RCT_CONTEXT_LUT: [u8; RCT_CONTEXT_MAX_DIFF as usize + 1] = {
    let mut lut = [0u8; RCT_CONTEXT_MAX_DIFF as usize + 1];
    let mut diff = 0usize;
    while diff < lut.len() {
        let mut cutoff = 0usize;
        let mut context = 0u8;
        while cutoff < RCT_CONTEXT_CUTOFFS.len() {
            if (diff as u32) < RCT_CONTEXT_CUTOFFS[cutoff] {
                context += 1;
            }
            cutoff += 1;
        }
        lut[diff] = context;
        diff += 1;
    }
    lut
};

#[inline(always)]
fn rct_context(max_diff: u32) -> usize {
    RCT_CONTEXT_LUT[max_diff.min(RCT_CONTEXT_MAX_DIFF) as usize] as usize
}

/// Channel order fed to the elementary transform for permutation `perm`
/// (0=RGB, 1=GBR, 2=BRG, 3=RBG, 4=GRB, 5=BGR).
#[inline]
fn rct_perm(perm: usize) -> [usize; 3] {
    [
        perm % 3,
        (perm + 1 + perm / 3) % 3,
        (perm + 2 - perm / 3) % 3,
    ]
}

/// Forward elementary RCT `t` (rct_type % 7) on already-permuted values.
#[inline]
fn rct_forward_pixel(t: u32, first: i32, second: i32, third: i32) -> (i32, i32, i32) {
    if t == 6 {
        let o1 = first - third;
        let tmp = third + (o1 >> 1);
        let o2 = second - tmp;
        (tmp + (o2 >> 1), o1, o2)
    } else {
        let s = match t >> 1 {
            1 => second - first,
            2 => second - ((first + third) >> 1),
            _ => second,
        };
        let th = if t & 1 == 1 { third - first } else { third };
        (first, s, th)
    }
}

#[inline]
fn rct_from_ycocg_pixel(y: i32, co: i32, cg: i32, rct: u32) -> (i32, i32, i32) {
    if rct == 6 {
        return (y, co, cg);
    }
    let rgb = inverse_ycocg(y, co, cg);
    let rgb = [rgb.0, rgb.1, rgb.2];
    let perm = rct_perm((rct / 7) as usize);
    rct_forward_pixel(rct % 7, rgb[perm[0]], rgb[perm[1]], rgb[perm[2]])
}

/// Port of libjxl's EstimateCost: gradient-predictor residual token entropy
/// over 18 local-activity contexts per channel, plus raw bits. Candidate pixels
/// are derived directly from the input YCoCg planes, avoiding a temporary RGB
/// image shared by the seven scoring passes.
fn estimate_rct_cost(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    rct: u32,
    row_scratch: &mut GradientScratch,
) -> f32 {
    const NCTX: usize = 18;
    const ALPHA: usize = 64;
    let y_plane = linear.plane_data(0);
    let co_plane = linear.plane_data(1);
    let cg_plane = linear.plane_data(2);
    let mut hist = [0u64; 3 * NCTX * ALPHA];
    let mut extra_bits: u64 = 0;
    row_scratch.prev.resize(3 * xsize, 0);
    row_scratch.cur.resize(3 * xsize, 0);
    row_scratch.prev.fill(0);
    let GradientScratch { prev, cur, .. } = row_scratch;
    let (mut prev, mut cur) = (prev, cur);
    for y in 0..ysize {
        let yy = &y_plane[y * xsize..][..xsize];
        let co = &co_plane[y * xsize..][..xsize];
        let cg = &cg_plane[y * xsize..][..xsize];
        for x in 0..xsize {
            let (a, b, c) = rct_from_ycocg_pixel(yy[x], co[x], cg[x], rct);
            cur[x] = a;
            cur[xsize + x] = b;
            cur[2 * xsize + x] = c;
        }
        for ch in 0..3usize {
            let crow = &cur[ch * xsize..][..xsize];
            let prow = &prev[ch * xsize..][..xsize];
            let chist = &mut hist[ch * NCTX * ALPHA..][..NCTX * ALPHA];
            for x in 0..xsize {
                let left = if x > 0 {
                    crow[x - 1]
                } else if y > 0 {
                    prow[x]
                } else {
                    0
                };
                let top = if y > 0 { prow[x] } else { left };
                let topleft = if x > 0 && y > 0 { prow[x - 1] } else { left };
                let mx = left.max(top).max(topleft);
                let mn = left.min(top).min(topleft);
                let max_diff = (mx - mn) as u32;
                let ctx = rct_context(max_diff);
                let res =
                    crow[x] as i64 - clamped_gradient(left as i64, top as i64, topleft as i64);
                let (tok, nb, _) = uint_encode(pack_signed(res as i32));
                chist[ctx * ALPHA + (tok as usize).min(ALPHA - 1)] += 1;
                extra_bits += nb as u64;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let mut cost = extra_bits as f32;
    for h in hist.as_chunks::<ALPHA>().0 {
        let total: u64 = h.iter().sum();
        cost += entropy_of_hist(h, total);
    }
    cost
}

/// Estimate every candidate RCT and return the winner's planes when it is not
/// YCoCg (in which case the caller keeps the input planes unchanged).
fn select_rct(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<(u32, Image3Si)> {
    let costs = pool.steal_map(scratch, RCT_CANDIDATES.len(), |i, scratch| {
        estimate_rct_cost(
            linear,
            xsize,
            ysize,
            RCT_CANDIDATES[i],
            &mut scratch.gradient,
        )
    });
    let mut best = 0usize;
    for (i, &c) in costs.iter().enumerate() {
        if c < costs[best] {
            best = i;
        }
    }
    let rct = RCT_CANDIDATES[best];
    if rct == 6 {
        return None;
    }
    let mut out = Image3Si::new(xsize, ysize);
    for y in 0..ysize {
        let [o0, o1, o2] = out.all_plane_rows_mut(y);
        let yy = &linear.plane_data(0)[y * xsize..][..xsize];
        let co = &linear.plane_data(1)[y * xsize..][..xsize];
        let cg = &linear.plane_data(2)[y * xsize..][..xsize];
        for x in 0..xsize {
            let (a, b, c) = rct_from_ycocg_pixel(yy[x], co[x], cg[x], rct);
            o0[x] = a;
            o1[x] = b;
            o2[x] = c;
        }
    }
    Some((rct, out))
}

/// Target sample count for tree learning (across all channels/groups). Above
/// this size the learner's best-first search dominates Slow encoding;
/// 256K samples preserve a dense image-wide probe without letting its cost
/// scale with the full pixel count.
const MA_TARGET_SAMPLES: usize = 1 << 19;
/// Preserve full sampling for genuinely small images. Above the former 256K
/// target, a stride of three is both cheaper and less prone to fitting local
/// pixel-phase noise than switching abruptly to a full-image sample.
const MA_FULL_SAMPLE_LIMIT: usize = 1 << 18;
/// First-stage learner budget. Most unhelpful or simple trees terminate here;
/// only a saturated tree with a clear rate win is retrained on the full probe.
const MA_COARSE_TARGET_SAMPLES: usize = 1 << 16;
const MA_COARSE_MAX_LEAVES: usize = 32;
const MA_DEEPEN_MIN_SAVINGS: f64 = 0.02;
/// Bits a split must save (image domain) before it's kept. Growth is
/// best-first, so this mostly prunes the tail once the leaf budget is spent.
const MA_SPLIT_COST_BITS: f32 = 100.0;
/// Leaf cap: pixel contexts + the LZ77 distance context must fit the
/// LZ77_MAX_CONTEXTS scratch; clustering reduces them to <= 64 histograms.
/// Best-first growth adds exactly one leaf per split, so the cap is hard.
const MA_MAX_LEAVES: usize = LZ77_MAX_CONTEXTS - 1;
/// Minimum samples in a node before it must become a leaf.
const MA_MIN_NODE_SAMPLES: usize = 128;

/// Walk one channel rectangle in scan order, feeding the visitor the property
/// vector (libjxl ids 0..=15), the neighborhood, and the WP prediction of
/// every pixel.
fn walk_channel_ma<F: Fn(usize, usize) -> i32>(
    get: &F,
    gw: usize,
    gh: usize,
    chan: u32,
    wp_params: WpParams,
    mut visit: impl FnMut(usize, usize, i64, &[i32; NUM_MA_PROPS], PredictorNeighbors, i64),
) {
    let mut wp = WpState::with_params(gw, wp_params);
    let mut p = [0i32; NUM_MA_PROPS];
    p[0] = chan as i32;
    for y in 0..gh {
        let row = wp.row_offsets(y);
        let mut west = 0i64;
        let mut west_west = 0i64;
        p[2] = y as i32;
        p[9] = 0; // "local gradient" carry, reset per row (InitPropsRow)
        for x in 0..gw {
            let value = get(x, y) as i64;
            let top = if y > 0 { get(x, y - 1) as i64 } else { west };
            let left = if x > 0 { west } else { top };
            let top_left = if x > 0 && y > 0 {
                get(x - 1, y - 1) as i64
            } else {
                left
            };
            let top_right = if x + 1 < gw && y > 0 {
                get(x + 1, y - 1) as i64
            } else {
                top
            };
            let top_top = if y > 1 { get(x, y - 2) as i64 } else { top };
            let n = PredictorNeighbors {
                left,
                top,
                top_left,
                top_right,
                left_left: if x > 1 { west_west } else { left },
                top_top,
                top_right_right: if x + 2 < gw && y > 0 {
                    get(x + 2, y - 1) as i64
                } else {
                    top_right
                },
            };
            p[3] = x as i32;
            p[4] = n.top.abs() as i32;
            p[5] = n.left.abs() as i32;
            p[6] = n.top as i32;
            p[7] = n.left as i32;
            // p[8] reads the previous pixel's p[9] (0 at row start).
            p[8] = (n.left - p[9] as i64) as i32;
            p[9] = (n.left + n.top - n.top_left) as i32;
            p[10] = (n.left - n.top_left) as i32;
            p[11] = (n.top_left - n.top) as i32;
            p[12] = (n.top - n.top_right) as i32;
            p[13] = (n.top - n.top_top) as i32;
            p[14] = (n.left - n.left_left) as i32;
            let wp_pred = wp.predict_and_update(
                value,
                x,
                row,
                WpNeighbors {
                    north: n.top,
                    west: n.left,
                    north_east: n.top_right,
                    north_west: n.top_left,
                    north_north: n.top_top,
                },
            );
            p[15] = wp.wp_prop as i32;
            visit(x, y, value, &p, n, wp_pred);
            west_west = west;
            west = value;
        }
    }
}

/// Append every `stride`-th pixel of the channel rectangle to `samples`,
/// with the residual token under every decoder predictor.
fn sample_channel_ma<F: Fn(usize, usize) -> i32>(
    get: &F,
    gw: usize,
    gh: usize,
    chan: u32,
    wp_params: WpParams,
    stride: usize,
    samples: &mut MaSamples,
) {
    debug_assert!(stride != 0);
    let mut wp = WpState::with_params(gw, wp_params);
    let mut until_sample = stride;
    for y in 0..gh {
        let row = wp.row_offsets(y);
        let mut west = 0i64;
        let mut west_west = 0i64;
        let mut previous_local_gradient = 0i32;
        for x in 0..gw {
            let value = get(x, y) as i64;
            let north = if y > 0 { get(x, y - 1) as i64 } else { west };
            let left = if x > 0 { west } else { north };
            let top_left = if x > 0 && y > 0 {
                get(x - 1, y - 1) as i64
            } else {
                left
            };
            let top_right = if x + 1 < gw && y > 0 {
                get(x + 1, y - 1) as i64
            } else {
                north
            };
            let top_top = if y > 1 { get(x, y - 2) as i64 } else { north };
            let wp_neighbors = WpNeighbors {
                north,
                west: left,
                north_east: top_right,
                north_west: top_left,
                north_north: top_top,
            };
            let wp_pred = wp.predict_and_update(value, x, row, wp_neighbors);
            let local_gradient = (left + north - top_left) as i32;

            until_sample -= 1;
            if until_sample == 0 {
                until_sample = stride;
                let neighbors = PredictorNeighbors {
                    left,
                    top: north,
                    top_left,
                    top_right,
                    left_left: if x > 1 { west_west } else { left },
                    top_top,
                    top_right_right: if x + 2 < gw && y > 0 {
                        get(x + 2, y - 1) as i64
                    } else {
                        top_right
                    },
                };
                let mut props = [0i32; NUM_MA_PROPS];
                props[0] = chan as i32;
                props[2] = y as i32;
                props[3] = x as i32;
                props[4] = north.abs() as i32;
                props[5] = left.abs() as i32;
                props[6] = north as i32;
                props[7] = left as i32;
                props[8] = (left - previous_local_gradient as i64) as i32;
                props[9] = local_gradient;
                props[10] = (left - top_left) as i32;
                props[11] = (top_left - north) as i32;
                props[12] = (north - top_right) as i32;
                props[13] = (north - top_top) as i32;
                props[14] = (left - neighbors.left_left) as i32;
                props[15] = wp.wp_prop as i32;

                let mut tok = [0u8; NUM_MA_PREDS];
                let mut nbits = [0u8; NUM_MA_PREDS];
                for pred in 0..NUM_MA_PREDS as u32 {
                    let pv = predictor_value(pred, neighbors, wp_pred);
                    let (t, nb, _) = uint_encode(pack_signed((value - pv) as i32));
                    tok[pred as usize] = t.min(u8::MAX as u32) as u8;
                    nbits[pred as usize] = nb.min(u8::MAX as u32) as u8;
                }
                samples.push(props, tok, nbits);
            }
            previous_local_gradient = local_gradient;
            west_west = west;
            west = value;
        }
    }
}

/// Tokenize the channel rectangle through the learned tree: every pixel is
/// routed to its leaf's context and coded with its leaf's predictor.
fn tokenize_channel_ma<F: Fn(usize, usize) -> i32>(
    get: &F,
    gw: usize,
    gh: usize,
    chan: u32,
    wp_params: WpParams,
    tree: &LearnedTree,
    leaf_ctx: &[u32],
    out: &mut Vec<Token>,
) {
    walk_channel_ma(get, gw, gh, chan, wp_params, |_x, _y, v, p, n, wp_pred| {
        let (node, pred) = tree.lookup(p);
        let pv = predictor_value(pred, n, wp_pred);
        out.push(Token::new(
            leaf_ctx[node as usize],
            pack_signed((v - pv) as i32),
        ));
    });
}

/// BFS-emit a learned tree (matches libjxl's FIFO tree decode). Returns
/// (tree tokens, context id per node index — leaves only, context count).
fn emit_learned_tree(tree: &LearnedTree) -> (Vec<Token>, Vec<u32>, u32) {
    use std::collections::VecDeque;
    let mut tokens = Vec::new();
    let mut leaf_ctx = vec![u32::MAX; tree.nodes.len()];
    let mut q: VecDeque<u32> = VecDeque::new();
    q.push_back(0);
    let mut ctx = 0u32;
    while let Some(i) = q.pop_front() {
        match tree.nodes[i as usize] {
            MaNode::Split { prop, val, gt, le } => {
                push_split(&mut tokens, prop as u32, val);
                q.push_back(gt);
                q.push_back(le);
            }
            MaNode::Leaf { pred } => {
                push_leaf(&mut tokens, pred);
                leaf_ctx[i as usize] = ctx;
                ctx += 1;
            }
        }
    }
    (tokens, leaf_ctx, ctx)
}

/// Sampling stride for tree learning; odd to avoid column aliasing.
fn ma_sample_stride(total_px: usize) -> usize {
    let mut stride = total_px.div_ceil(MA_TARGET_SAMPLES).max(1);
    if stride == 1 && total_px > MA_FULL_SAMPLE_LIMIT {
        stride = 3;
    }
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    stride
}

/// Learn the tree from merged samples and decide whether it beats the flat
/// path estimate. Returns the tree with its BFS emission on success.
#[allow(clippy::type_complexity)]
fn learn_and_gate_ma_tree(
    samples: &MaSamples,
    stride: usize,
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<(LearnedTree, Vec<Token>, Vec<u32>, u32)> {
    if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
        return None;
    }

    struct GatedTree {
        tree: LearnedTree,
        tree_tokens: Vec<Token>,
        leaf_ctx: Vec<u32>,
        num_ctx: u32,
        est_real: f64,
        flat_real: f64,
    }

    let gate = |tree: LearnedTree, sample_scale: f64| {
        let (tree_tokens, leaf_ctx, num_ctx) = emit_learned_tree(&tree);
        // Header overhead: tree tokens plus the extra per-context histogram /
        // context-map cost (conservative estimate; clustering merges most of it).
        let overhead_bits = tree_tokens.len() as f64 * 10.0 + num_ctx as f64 * 200.0;
        let est_real = tree.est_bits * sample_scale + overhead_bits;
        let flat_real = tree.flat_bits * sample_scale;
        (est_real < flat_real).then_some(GatedTree {
            tree,
            tree_tokens,
            leaf_ctx,
            num_ctx,
            est_real,
            flat_real,
        })
    };

    if samples.len() <= MA_COARSE_TARGET_SAMPLES {
        let tree = learn_ma_tree(
            samples,
            MaLearnParams {
                alphabet: min_symbol as usize,
                max_leaves: MA_MAX_LEAVES,
                split_cost_bits: MA_SPLIT_COST_BITS / stride as f32,
                min_node: MA_MIN_NODE_SAMPLES,
            },
            pool,
            scratch,
        );
        let selected = gate(tree, stride as f64)?;
        return Some((
            selected.tree,
            selected.tree_tokens,
            selected.leaf_ctx,
            selected.num_ctx,
        ));
    }

    let coarse_indices = samples.evenly_sampled_indices(MA_COARSE_TARGET_SAMPLES);
    let coarse_len = coarse_indices.len();
    let coarse_scale = stride as f64 * samples.len() as f64 / coarse_len as f64;
    let coarse_tree = learn_ma_tree_indexed(
        samples,
        coarse_indices,
        MaLearnParams {
            alphabet: min_symbol as usize,
            max_leaves: MA_COARSE_MAX_LEAVES,
            split_cost_bits: MA_SPLIT_COST_BITS / coarse_scale as f32,
            min_node: MA_MIN_NODE_SAMPLES,
        },
        pool,
        scratch,
    );
    let coarse = gate(coarse_tree, coarse_scale)?;
    let coarse_leaves = coarse.tree.nodes.len().div_ceil(2);
    let coarse_savings = 1.0 - coarse.est_real / coarse.flat_real;
    let should_deepen = samples.len() > MA_COARSE_TARGET_SAMPLES
        && coarse_leaves == MA_COARSE_MAX_LEAVES
        && coarse_savings >= MA_DEEPEN_MIN_SAVINGS;

    let selected = if should_deepen {
        let deep_tree = deepen_ma_tree(
            samples,
            MaLearnParams {
                alphabet: min_symbol as usize,
                max_leaves: MA_MAX_LEAVES,
                split_cost_bits: MA_SPLIT_COST_BITS / stride as f32,
                min_node: MA_MIN_NODE_SAMPLES,
            },
            coarse.tree.clone(),
            pool,
            scratch,
        );
        match gate(deep_tree, stride as f64) {
            Some(deep) => deep,
            _ => coarse,
        }
    } else {
        coarse
    };
    Some((
        selected.tree,
        selected.tree_tokens,
        selected.leaf_ctx,
        selected.num_ctx,
    ))
}

/// Learned-tree lossless path, single group. Returns its estimated fractional
/// saving over the flat path after writing the complete candidate.
#[allow(clippy::too_many_arguments)]
fn try_encode_learned_tree_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    rct_type: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    writer: &mut BitWriter,
) -> Option<f64> {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };
    let stride = ma_sample_stride(xsize * ysize * nb_chans);

    let mut per_chan = pool.steal_map(scratch, nb_chans, |chan, _scratch| {
        let mut samples = MaSamples::with_capacity(xsize * ysize / stride);
        if chan < 3 {
            let pd = linear.plane_data(chan);
            sample_channel_ma(
                &|x, y| pd[y * xsize + x],
                xsize,
                ysize,
                chan as u32,
                wp_params,
                stride,
                &mut samples,
            );
        } else {
            let a = alpha.expect("alpha channel must exist");
            sample_channel_ma(
                &|x, y| a.get_i32(y * xsize + x),
                xsize,
                ysize,
                chan as u32,
                wp_params,
                stride,
                &mut samples,
            );
        }
        samples
    });
    let total_samples = per_chan.iter().map(MaSamples::len).sum();
    let mut samples = MaSamples::with_capacity(total_samples);
    for s in per_chan.iter_mut() {
        samples.append(s);
    }

    let (tree, tree_tokens, leaf_ctx, num_ctx) =
        learn_and_gate_ma_tree(&samples, stride, min_symbol, pool, scratch)?;
    let estimated_savings = 1.0 - tree.est_bits / tree.flat_bits.max(f64::MIN_POSITIVE);
    drop(samples);

    let channel_tokens = pool.steal_map(scratch, nb_chans, |chan, _scratch| {
        let mut tokens: Vec<Token> = Vec::with_capacity(xsize * ysize);
        if chan < 3 {
            let pd = linear.plane_data(chan);
            tokenize_channel_ma(
                &|x, y| pd[y * xsize + x],
                xsize,
                ysize,
                chan as u32,
                wp_params,
                &tree,
                &leaf_ctx,
                &mut tokens,
            );
        } else {
            let a = alpha.expect("alpha channel must exist");
            tokenize_channel_ma(
                &|x, y| a.get_i32(y * xsize + x),
                xsize,
                ysize,
                chan as u32,
                wp_params,
                &tree,
                &leaf_ctx,
                &mut tokens,
            );
        }
        tokens
    });
    let mut tokens: Vec<Token> = Vec::with_capacity(xsize * ysize * nb_chans);
    for channel in channel_tokens {
        tokens.extend(channel);
    }

    write_frame_header_modular(alpha.is_some(), writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    section.write(1, 0); // use_global_tree = 0
    write_wp_header(wp_params, &mut section);
    write_modular_transforms(nb_chans, rct_type, &mut section);

    let distance_ctx = num_ctx;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, crate::Speed::Slow, scratch);
    let code = build_lz_pixel_code(
        std::iter::once(lz_tokens.as_slice()),
        num_ctx as usize,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_tree_lz77(
        &tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(std::slice::from_mut(&mut section));
    writer.zero_pad_to_byte();
    Some(estimated_savings)
}

/// Learned-tree lossless path, multi-group: one global tree learned from
/// group-local samples; every AC group routes its pixels through it (fresh
/// WP and group-local coordinates per group, matching the decoder).
#[allow(clippy::too_many_arguments)]
fn try_encode_learned_tree_multi_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    xsize_groups: usize,
    ysize_groups: usize,
    num_dc_groups: usize,
    min_symbol: u32,
    rct_type: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    writer: &mut BitWriter,
) -> Option<f64> {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };
    let num_ac_groups = xsize_groups * ysize_groups;
    let stride = ma_sample_stride(xsize * ysize * nb_chans);

    let mut group_samples = pool.steal_map(scratch, num_ac_groups, |group_index, _scratch| {
        let gx = group_index % xsize_groups;
        let gy = group_index / xsize_groups;
        let x0 = gx * GROUP_DIM;
        let y0 = gy * GROUP_DIM;
        let gw = GROUP_DIM.min(xsize - x0);
        let gh = GROUP_DIM.min(ysize - y0);
        let mut samples = MaSamples::with_capacity(nb_chans * (gw * gh / stride));
        for chan in 0..3usize {
            let pd = linear.plane_data(chan);
            let get = |lx: usize, ly: usize| pd[(y0 + ly) * xsize + (x0 + lx)];
            sample_channel_ma(&get, gw, gh, chan as u32, wp_params, stride, &mut samples);
        }
        if let Some(a) = alpha {
            let get = |lx: usize, ly: usize| a.get_i32((y0 + ly) * xsize + (x0 + lx));
            sample_channel_ma(&get, gw, gh, 3, wp_params, stride, &mut samples);
        }
        samples
    });
    let total_samples = group_samples.iter().map(MaSamples::len).sum();
    let mut samples = MaSamples::with_capacity(total_samples);
    for s in group_samples.iter_mut() {
        samples.append(s);
    }
    drop(group_samples);

    let (tree, tree_tokens, leaf_ctx, num_ctx) =
        learn_and_gate_ma_tree(&samples, stride, min_symbol, pool, scratch)?;
    let estimated_savings = 1.0 - tree.est_bits / tree.flat_bits.max(f64::MIN_POSITIVE);
    drop(samples);

    let distance_ctx = num_ctx;
    let deep_lz = DeepLzScratchPool::new(group_lz_threads(crate::Speed::Slow, pool));
    let group_lz_tokens: Vec<Vec<LzToken>> = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(crate::Speed::Slow, pool),
        |group_index, scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let gw = GROUP_DIM.min(xsize - x0);
            let gh = GROUP_DIM.min(ysize - y0);
            let mut toks: Vec<Token> = Vec::with_capacity(gw * gh * nb_chans);
            for chan in 0..3usize {
                let pd = linear.plane_data(chan);
                let get = |lx: usize, ly: usize| pd[(y0 + ly) * xsize + (x0 + lx)];
                tokenize_channel_ma(
                    &get,
                    gw,
                    gh,
                    chan as u32,
                    wp_params,
                    &tree,
                    &leaf_ctx,
                    &mut toks,
                );
            }
            if let Some(a) = alpha {
                let get = |lx: usize, ly: usize| a.get_i32((y0 + ly) * xsize + (x0 + lx));
                tokenize_channel_ma(&get, gw, gh, 3, wp_params, &tree, &leaf_ctx, &mut toks);
            }
            deep_lz.with_depth(|depth| {
                lz77_compress_for_speed_with_depth(
                    &toks,
                    distance_ctx,
                    crate::Speed::Slow,
                    depth,
                    scratch,
                )
            })
        },
    );
    drop(deep_lz);
    let code = build_lz_pixel_code(
        group_lz_tokens.iter().map(Vec::as_slice),
        num_ctx as usize,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    write_frame_header_modular(alpha.is_some(), writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(
        &tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    write_wp_header(wp_params, &mut sections[0]);
    write_modular_transforms(nb_chans, rct_type, &mut sections[0]);
    sections[0].zero_pad_to_byte();

    for section in sections[1..num_dc_groups + 1].iter_mut() {
        section.write(1, 1);
        write_wp_header(wp_params, section);
        section.write(2, 0);
        section.zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    write_wp_header(wp_params, &mut sections[ac_global_idx]);
    sections[ac_global_idx].zero_pad_to_byte();

    for group_index in 0..num_ac_groups {
        let section_idx = 2 + num_dc_groups + group_index;
        sections[section_idx].write(1, 1);
        write_wp_header(wp_params, &mut sections[section_idx]);
        sections[section_idx].write(2, 0);
        write_lz_section(
            &group_lz_tokens[group_index],
            distance_ctx,
            &code,
            min_symbol,
            &mut sections[section_idx],
        );
        sections[section_idx].zero_pad_to_byte();
    }

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    Some(estimated_savings)
}

/// v1 context tree: single-group only. Returns true (and writes the full frame)
/// if the context tree is estimated to help; false otherwise (caller falls
/// through to the flat path, having written nothing).
fn try_encode_context_tree_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    predictors: &[u32],
    min_symbol: u32,
    rct_type: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };

    // Collect residuals + WP property per channel (WP runs for every channel).
    let collected = pool.steal_map(scratch, nb_chans, |chan, _scratch| {
        if chan < 3 {
            let pd = linear.plane_data(chan);
            collect_channel(
                |x, y| pd[y * xsize + x],
                xsize,
                ysize,
                predictors[chan],
                wp_params,
            )
        } else {
            let a = alpha.expect("alpha channel must exist");
            collect_channel(
                |x, y| a.get_i32(y * xsize + x),
                xsize,
                ysize,
                predictors[chan],
                wp_params,
            )
        }
    });
    let (chan_res, chan_prp): (Vec<Vec<u32>>, Vec<Vec<i64>>) = collected.into_iter().unzip();

    // Per-channel threshold + cost comparison.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    let threshold_costs = pool.steal_map(scratch, nb_chans, |chan, scratch| {
        pick_threshold(&chan_res[chan], &chan_prp[chan], scratch)
    });
    for (chan, (t, cb, fb)) in threshold_costs.into_iter().enumerate() {
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    // Guard: require the context tree to beat the flat path by more than the
    // extra-context header overhead (~64 bytes per added context, conservative).
    let overhead_bits = (2 * nb_chans) as f32 * 64.0 * 8.0;
    if ctx_bits + overhead_bits >= flat_bits {
        return false;
    }

    // Build tree + context map.
    let tree = build_context_tree(nb_chans, predictors, &ts);
    let mut tree_tokens: Vec<Token> = Vec::new();
    let ctx_map = emit_ct_tree(&tree, &mut tree_tokens);
    // Flat lookup over the dense (chan*3+bucket) property space, replacing a
    // per-pixel HashMap probe in the tokenize loop.
    let ctx_lut: Vec<u32> = (0..(nb_chans as u32) * 3).map(|k| ctx_map[&k]).collect();
    let num_pixel_ctx = nb_chans * 3;

    // Tokenize: each pixel routed to context (channel,bucket).
    let channel_tokens = pool.steal_map(scratch, nb_chans, |chan, _scratch| {
        let mut tokens = Vec::with_capacity(xsize * ysize);
        let res = &chan_res[chan];
        let prp = &chan_prp[chan];
        let t = ts[chan] as i64;
        for (&prp, &res) in prp[..res.len()].iter().zip(res.iter()) {
            let bucket = bucket_of(prp, t);
            let ctx = ctx_lut[chan * 3 + bucket as usize];
            tokens.push(Token::new(ctx, res));
        }
        tokens
    });
    let mut tokens: Vec<Token> = Vec::with_capacity(xsize * ysize * nb_chans);
    for channel in channel_tokens {
        tokens.extend(channel);
    }

    // Frame header + single section.
    write_frame_header_modular(alpha.is_some(), writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0
    section.write(1, 0); // use_global_tree = 0
    write_wp_header(wp_params, &mut section);
    write_modular_transforms(nb_chans, rct_type, &mut section);

    let distance_ctx = num_pixel_ctx as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, crate::Speed::Slow, scratch);
    let code = build_lz_pixel_code(
        std::iter::once(lz_tokens.as_slice()),
        num_pixel_ctx,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_tree_lz77(
        &tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(std::slice::from_mut(&mut section));
    writer.zero_pad_to_byte();
    true
}

/// v1 context tree, multi-group. Global context tree in LfGlobal; each AC group
/// routes its group-local pixels through it (fresh WP per group, matching the
/// decoder). Returns true (and writes the full frame) if it helps; else false.
fn try_encode_context_tree_multi_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    predictors: &[u32],
    xsize_groups: usize,
    ysize_groups: usize,
    num_dc_groups: usize,
    min_symbol: u32,
    rct_type: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };
    let num_ac_groups = xsize_groups * ysize_groups;

    // 1) Collect (residual, WP property) per group per channel (group-local WP).
    let groups: Vec<Vec<(Vec<u32>, Vec<i64>)>> =
        pool.steal_map(scratch, num_ac_groups, |group_index, _scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let gw = GROUP_DIM.min(xsize - x0);
            let gh = GROUP_DIM.min(ysize - y0);
            let mut chans: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(nb_chans);
            for chan in 0..3usize {
                let pd = linear.plane_data(chan);
                let get = |lx: usize, ly: usize| pd[(y0 + ly) * xsize + (x0 + lx)];
                chans.push(collect_channel(get, gw, gh, predictors[chan], wp_params));
            }
            if let Some(a) = alpha {
                let get = |lx: usize, ly: usize| a.get_i32((y0 + ly) * xsize + (x0 + lx));
                chans.push(collect_channel(get, gw, gh, predictors[3], wp_params));
            }
            chans
        });

    // 2) Global per-channel threshold from aggregated stats.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    let threshold_costs = pool.steal_map(scratch, nb_chans, |chan, scratch| {
        pick_threshold_grouped(&groups, chan, scratch)
    });
    for (chan, (t, cb, fb)) in threshold_costs.into_iter().enumerate() {
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    let overhead_bits = (2 * nb_chans) as f32 * 64.0 * 8.0;
    if ctx_bits + overhead_bits >= flat_bits {
        return false;
    }

    // 3) Global context tree + map.
    let tree = build_context_tree(nb_chans, predictors, &ts);
    let mut tree_tokens: Vec<Token> = Vec::new();
    let ctx_map = emit_ct_tree(&tree, &mut tree_tokens);
    // Flat lookup over the dense (chan*3+bucket) property space, replacing a
    // per-pixel HashMap probe in the tokenize loop.
    let ctx_lut: Vec<u32> = (0..(nb_chans as u32) * 3).map(|k| ctx_map[&k]).collect();
    let num_pixel_ctx = nb_chans * 3;
    let distance_ctx = num_pixel_ctx as u32;

    // 4) Per-group tokens (reusing collected res/prop) + per-group LZ77.
    let deep_lz = DeepLzScratchPool::new(group_lz_threads(crate::Speed::Slow, pool));
    let group_lz_tokens: Vec<Vec<LzToken>> = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(crate::Speed::Slow, pool),
        |group_index, scratch| {
            let g = &groups[group_index];
            let token_count = g.iter().map(|(res, _)| res.len()).sum();
            let mut toks: Vec<Token> = Vec::with_capacity(token_count);
            for chan in 0..nb_chans {
                let (res, prp) = &g[chan];
                let t = ts[chan] as i64;
                for (&prp, &res) in prp[..res.len()].iter().zip(res.iter()) {
                    let bucket = bucket_of(prp, t);
                    let ctx = ctx_lut[chan * 3 + bucket as usize];
                    toks.push(Token::new(ctx, res));
                }
            }
            deep_lz.with_depth(|depth| {
                lz77_compress_for_speed_with_depth(
                    &toks,
                    distance_ctx,
                    crate::Speed::Slow,
                    depth,
                    scratch,
                )
            })
        },
    );
    drop(deep_lz);
    let code = build_lz_pixel_code(
        group_lz_tokens.iter().map(Vec::as_slice),
        num_pixel_ctx,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    // 5) Sections (same layout as the flat multi-group path).
    write_frame_header_modular(alpha.is_some(), writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(
        &tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    write_wp_header(wp_params, &mut sections[0]);
    write_modular_transforms(nb_chans, rct_type, &mut sections[0]);
    sections[0].zero_pad_to_byte();

    for section in sections[1..num_dc_groups + 1].iter_mut() {
        section.write(1, 1);
        write_wp_header(wp_params, section);
        section.write(2, 0);
        section.zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    write_wp_header(wp_params, &mut sections[ac_global_idx]);
    sections[ac_global_idx].zero_pad_to_byte();

    for group_index in 0..num_ac_groups {
        let section_idx = 2 + num_dc_groups + group_index;
        sections[section_idx].write(1, 1);
        write_wp_header(wp_params, &mut sections[section_idx]);
        sections[section_idx].write(2, 0);
        write_lz_section(
            &group_lz_tokens[group_index],
            distance_ctx,
            &code,
            min_symbol,
            &mut sections[section_idx],
        );
        sections[section_idx].zero_pad_to_byte();
    }

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}

use crate::adaptive_quant::dirty_log2f;
use crate::bit_writer::BitWriter;
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, uint_encode, write_entropy_code,
    write_token,
};
use crate::image::Image3Si;
use crate::ma_tree::{
    LearnedTree, MaLearnParams, MaNode, MaSamples, NUM_MA_PREDS, NUM_MA_PROPS, deepen_ma_tree,
    learn_ma_tree, learn_ma_tree_indexed,
};
use crate::patches::{
    ModularFrameKind, NUM_PATCH_CONTEXTS, PATCH_REF_ID, PATCH_TILE, PatchReference,
    find_lossless_patches,
};
use crate::weighted_predictor::{WpNeighbors, WpParams, WpState, write_wp_header};
use crate::xyb::quantize_xyb_channels;
use std::sync::OnceLock;
// ---------------------------------------------------------------------------
// Tree writing (balanced N-leaf, Gradient predictor).
//
// libjxl reads modular trees in BFS order: each node is one PROPERTY token
// (=0 for leaf, =prop+1 for split) optionally followed by a SPLIT_VAL token
// or 4 leaf-data tokens (predictor, offset, multiplier-log, multiplier-bits).
// All channels share the Gradient predictor with offset 0 and multiplier 1.
// ---------------------------------------------------------------------------

fn push_split(out: &mut Vec<Token>, property: u32, split_val: i32) {
    out.push(Token::new(TREE_CTX_PROPERTY, property + 1));
    out.push(Token::new(TREE_CTX_SPLIT_VAL, pack_signed(split_val)));
}

fn push_leaf(out: &mut Vec<Token>, predictor: u32) {
    out.push(Token::new(TREE_CTX_PROPERTY, 0));
    out.push(Token::new(TREE_CTX_PREDICTOR, predictor));
    out.push(Token::new(TREE_CTX_OFFSET, pack_signed(0)));
    out.push(Token::new(TREE_CTX_MULTIPLIER_LOG, 0));
    out.push(Token::new(TREE_CTX_MULTIPLIER_BITS, 0));
}

/// Build a balanced binary tree over `n_leaves` leaves splitting on property 0
/// (the channel index after RCT).  BFS leaf order is chan N-1, ..., chan 0.
fn build_balanced_tree_tokens(predictors: &[u32]) -> Vec<Token> {
    let n_leaves = predictors.len();
    let mut t = Vec::new();
    // Leaves are emitted in BFS order chan(N-1), ..., chan0 (matches fjxl), so
    // index them with predictors[chan].
    match n_leaves {
        1 => push_leaf(&mut t, predictors[0]),
        2 => {
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        3 => {
            push_split(&mut t, 0, 1);
            push_leaf(&mut t, predictors[2]); // chan 2
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        4 => {
            push_split(&mut t, 0, 1);
            push_split(&mut t, 0, 2);
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[3]); // chan 3
            push_leaf(&mut t, predictors[2]); // chan 2
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        _ => {
            // N>=5: right-leaning chain. BFS emission yields leaves in order
            // chan(N-1)..chan0 -> contexts 0..N-1, matching channel_to_context.
            // Peels off the highest channel at each split: split(0,k) sends
            // channel>k (== k+1) to a leaf, else continue.
            for k in (1..=(n_leaves - 2)).rev() {
                push_split(&mut t, 0, k as i32);
                push_leaf(&mut t, predictors[k + 1]);
            }
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]);
            push_leaf(&mut t, predictors[0]);
        }
    }
    t
}

// ---------------------------------------------------------------------------
// f32 lossless (v1): non-negative float, RGB, no alpha. The 32-bit float bits
// are reinterpreted as int32 channels (matching libjxl float_to_int for
// bits==32: a raw memcpy), then coded as a modular image with NO RCT and NO
// LZ77. LZ77 must be off because float residual tokens can reach 127, leaving
// no room below the 128-symbol alphabet for LZ77 length symbols. Per-channel
// gradient/WP prediction and the channel-split tree are reused unchanged.
// ---------------------------------------------------------------------------

/// Write a channel-split MA tree (no LZ77 in the tree) followed by the pixel
/// entropy code header with LZ77 DISABLED. Mirrors the tree-writing convention
/// but flips the pixel stream's LZ77 flag off.
fn write_tree_and_pixel_code_nolz(
    tree_tokens: &[Token],
    pixel_code: &OwnedEntropyCode,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    let tree_code =
        optimize_entropy_code(tree_tokens, NUM_TREE_CONTEXTS, &mut scratch.huffman_pool);
    let tree_code_ref = tree_code.as_ref();
    w.write(1, 0); // tree entropy code: no LZ77
    write_entropy_code(&tree_code_ref, &mut scratch.huffman_pool, w);
    for tok in tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }
    // Pixel entropy code: LZ77 DISABLED (1 bit = 0), then the code itself.
    w.write(1, 0);
    write_entropy_code(&pixel_code.as_ref(), &mut scratch.huffman_pool, w);
}

pub(crate) fn encode_frame_lossless_float(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let pool = ThreadPool::new_lossless(num_threads);
    let mut scratch = Box::new(CoderScratch::lossless());
    encode_frame_lossless_float_with_pool(linear, alpha, &pool, &mut scratch, writer);
}

fn encode_frame_lossless_float_with_pool(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    let xsize = linear.xsize();
    let ysize = linear.ysize();
    let nb_chans = 3usize + if alpha.is_some() { 1 } else { 0 };

    let xsize_groups = xsize.div_ceil(GROUP_DIM);
    let ysize_groups = ysize.div_ceil(GROUP_DIM);
    let num_ac_groups = xsize_groups * ysize_groups;
    let xsize_dc_groups = xsize.div_ceil(LF_GROUP_DIM);
    let ysize_dc_groups = ysize.div_ceil(LF_GROUP_DIM);
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
    let single_group = num_ac_groups == 1;
    let grad_pack_fn = selected_grad_pack_interior_fn();

    // Float-bit values reach ~2^30, where the Weighted Predictor's internal
    // weight/divlookup arithmetic is not bit-verified against libjxl. Gradient
    // (clamp(W+N-NW), computed in i64) is exact at any magnitude and the
    // decoder reconstructs it identically, so the float path uses gradient on
    // every channel. (Predictor id 5 = gradient.)
    const GRADIENT_PRED: u32 = 5;
    let predictors = [GRADIENT_PRED; 4];
    let tree_tokens = build_balanced_tree_tokens(&predictors[..nb_chans]);

    write_frame_header_modular(alpha.is_some(), writer);

    if single_group {
        let mut section = BitWriter::new();
        section.write(1, 1); // dc_quant all_default = 1
        section.write(1, 0); // has_tree = 0
        section.write(1, 0); // use_global_tree = 0
        section.write(1, 1); // wp_default = 1
        section.write(2, 0b00); // 0 transforms (no RCT for float bits)

        let tokens = tokenize_all(
            linear,
            alpha,
            xsize,
            ysize,
            0,
            0,
            xsize,
            ysize,
            3,
            &predictors,
            grad_pack_fn,
            pool,
            scratch,
        );
        let code = optimize_entropy_code(&tokens, nb_chans, &mut scratch.huffman_pool);
        write_tree_and_pixel_code_nolz(&tree_tokens, &code, scratch, &mut section);
        for t in &tokens {
            write_token(*t, &code.as_ref(), &mut section);
        }
        section.zero_pad_to_byte();

        writer.write(1, 0);
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(std::slice::from_mut(&mut section));
        writer.zero_pad_to_byte();
    } else {
        let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
        let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

        // Tokenize each AC group (group-local) and pool for one global code.
        let group_tokens: Vec<Vec<Token>> =
            pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);
                tokenize_all(
                    linear,
                    alpha,
                    xsize,
                    ysize,
                    x0,
                    y0,
                    gw,
                    gh,
                    3,
                    &predictors,
                    grad_pack_fn,
                    pool,
                    scratch,
                )
            });
        let mut all_tokens: Vec<Token> = Vec::new();
        for tokens in &group_tokens {
            all_tokens.extend_from_slice(tokens);
        }
        let code = optimize_entropy_code(&all_tokens, nb_chans, &mut scratch.huffman_pool);

        // Section 0: DC global (tree + pixel code, both no-LZ77) + GroupHeader.
        sections[0].write(1, 1); // dc_quant
        sections[0].write(1, 1); // has_tree = 1
        write_tree_and_pixel_code_nolz(&tree_tokens, &code, scratch, &mut sections[0]);
        sections[0].write(1, 1); // use_global_tree
        sections[0].write(1, 1); // wp_default
        sections[0].write(2, 0b00); // 0 transforms
        sections[0].zero_pad_to_byte();

        for i in 0..num_dc_groups {
            sections[1 + i].write(1, 1);
            sections[1 + i].write(1, 1);
            sections[1 + i].write(2, 0);
            sections[1 + i].zero_pad_to_byte();
        }

        let ac_global_idx = 1 + num_dc_groups;
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].zero_pad_to_byte();

        for group_index in 0..num_ac_groups {
            let section_idx = 2 + num_dc_groups + group_index;
            sections[section_idx].write(1, 1);
            sections[section_idx].write(1, 1);
            sections[section_idx].write(2, 0);
            for t in &group_tokens[group_index] {
                write_token(*t, &code.as_ref(), &mut sections[section_idx]);
            }
            sections[section_idx].zero_pad_to_byte();
        }

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

#[cfg(test)]
mod rate_selection_tests {
    use super::*;

    #[test]
    fn shortlist_thresholds_keep_ambiguous_candidates() {
        assert!(!local_palette_coverage_is_sufficient(24, 100));
        assert!(local_palette_coverage_is_sufficient(25, 100));
        assert!(!learned_tree_is_decisive(
            MA_DECISIVE_MIN_SAVINGS - f64::EPSILON
        ));
        assert!(learned_tree_is_decisive(MA_DECISIVE_MIN_SAVINGS));
    }

    fn image_from_rgb(
        width: usize,
        height: usize,
        mut rgb: impl FnMut(usize, usize) -> [i32; 3],
    ) -> Image3Si {
        let mut image = Image3Si::new(width, height);
        for y in 0..height {
            let [p0, p1, p2] = image.all_plane_rows_mut(y);
            for x in 0..width {
                let [r, g, b] = rgb(x, y);
                let (yc, co, cg) = forward_ycocg(r, g, b);
                p0[x] = yc;
                p1[x] = co;
                p2[x] = cg;
            }
        }
        image
    }

    fn encode_core(image: &Image3Si, allow_palettes: bool) -> BitWriter {
        let pool = ThreadPool::new_lossless(1);
        let mut scratch = CoderScratch::lossless();
        let mut writer = BitWriter::new();
        encode_frame_lossless_core_impl(
            image,
            None,
            8,
            false,
            3,
            crate::Speed::Slow,
            &pool,
            &mut scratch,
            ModularFrameKind::Regular,
            allow_palettes,
            &mut writer,
        );
        writer
    }

    #[test]
    fn slow_single_group_palette_is_rate_safe() {
        let image = image_from_rgb(64, 64, |x, y| {
            let c = ((x / 8 + y / 8) & 3) as i32;
            [c * 61, c * 37, c * 19]
        });
        let selected = encode_core(&image, true);
        let normal = encode_core(&image, false);
        assert!(selected.bits_written() <= normal.bits_written());
    }

    #[test]
    fn slow_local_palette_is_rate_safe_for_mixed_content() {
        let image = image_from_rgb(257, 257, |x, y| {
            if x < 256 && y < 128 {
                let c = ((x / 16 + y / 16) & 7) as i32;
                [c * 31, c * 17, c * 7]
            } else {
                let v = x
                    .wrapping_mul(1_664_525)
                    .wrapping_add(y.wrapping_mul(1_013_904_223));
                [
                    ((v >> 8) & 255) as i32,
                    ((v >> 16) & 255) as i32,
                    ((v >> 24) & 255) as i32,
                ]
            }
        });
        let selected = encode_core(&image, true);
        let normal = encode_core(&image, false);
        assert!(selected.bits_written() <= normal.bits_written());
    }
}

#[cfg(test)]
mod ma_sampling_tests {
    use super::*;

    fn reference_samples(
        plane: &[i32],
        width: usize,
        height: usize,
        stride: usize,
        params: WpParams,
    ) -> MaSamples {
        let get = |x: usize, y: usize| plane[y * width + x];
        let mut samples = MaSamples::new();
        let mut counter = 0usize;
        walk_channel_ma(
            &get,
            width,
            height,
            2,
            params,
            |_x, _y, value, p, n, wp_pred| {
                counter += 1;
                if !counter.is_multiple_of(stride) {
                    return;
                }
                let mut tok = [0u8; NUM_MA_PREDS];
                let mut nbits = [0u8; NUM_MA_PREDS];
                for pred in 0..NUM_MA_PREDS as u32 {
                    let prediction = predictor_value(pred, n, wp_pred);
                    let (token, extra_bits, _) =
                        uint_encode(pack_signed((value - prediction) as i32));
                    tok[pred as usize] = token.min(u8::MAX as u32) as u8;
                    nbits[pred as usize] = extra_bits.min(u8::MAX as u32) as u8;
                }
                samples.push(*p, tok, nbits);
            },
        );
        samples
    }

    #[test]
    fn stride_aware_sampling_matches_full_property_walk() {
        for &(width, height) in &[(0, 3), (1, 1), (1, 7), (2, 5), (9, 7), (17, 4)] {
            let plane: Vec<i32> = (0..width * height)
                .map(|i| {
                    let x = i % width.max(1);
                    let y = i / width.max(1);
                    (((x * 977 + y * 619) ^ (x * y * 37)) as i32 & 0xffff) - 0x7fff
                })
                .collect();
            let get = |x: usize, y: usize| plane[y * width + x];
            for &stride in &[1, 2, 3, 5, 11, width * height + 1] {
                for &params in &WpParams::PRESETS {
                    let expected = reference_samples(&plane, width, height, stride, params);
                    let mut actual = MaSamples::new();
                    sample_channel_ma(&get, width, height, 2, params, stride, &mut actual);
                    assert_eq!(
                        actual.props, expected.props,
                        "properties: {width}x{height}, stride={stride}, params={params:?}"
                    );
                    assert_eq!(
                        actual.tok, expected.tok,
                        "tokens: {width}x{height}, stride={stride}, params={params:?}"
                    );
                    assert_eq!(
                        actual.nbits, expected.nbits,
                        "extra bits: {width}x{height}, stride={stride}, params={params:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod rct_tests {
    use super::*;

    #[test]
    fn context_lookup_matches_linear_cutoff_classification() {
        let reference = |max_diff: u32| {
            RCT_CONTEXT_CUTOFFS
                .iter()
                .filter(|&&cutoff| max_diff < cutoff)
                .count()
        };
        for max_diff in 0..=u16::MAX as u32 {
            assert_eq!(rct_context(max_diff), reference(max_diff), "{max_diff}");
        }
        assert_eq!(rct_context(u32::MAX), reference(u32::MAX));
    }

    #[test]
    fn direct_ycocg_candidate_conversion_matches_rgb_round_trip() {
        for &(y, co, cg) in &[(0, 0, 0), (127, -91, 53), (-400, 723, -255)] {
            let rgb = inverse_ycocg(y, co, cg);
            let rgb = [rgb.0, rgb.1, rgb.2];
            for &rct in &RCT_CANDIDATES {
                let expected = if rct == 6 {
                    (y, co, cg)
                } else {
                    let perm = rct_perm((rct / 7) as usize);
                    rct_forward_pixel(rct % 7, rgb[perm[0]], rgb[perm[1]], rgb[perm[2]])
                };
                assert_eq!(rct_from_ycocg_pixel(y, co, cg, rct), expected);
            }
        }
    }
}

#[cfg(test)]
mod predictor_tests {
    use super::*;

    #[test]
    fn grouped_threshold_scoring_matches_concatenated_reference() {
        let groups = vec![
            vec![(vec![0, 3, 7, 3], vec![-20, -4, 12, 80])],
            vec![(vec![7, 1, 1, 9, 2], vec![-100, -9, 0, 17, 110])],
        ];
        let residuals: Vec<u32> = groups
            .iter()
            .flat_map(|group| group[0].0.iter().copied())
            .collect();
        let properties: Vec<i64> = groups
            .iter()
            .flat_map(|group| group[0].1.iter().copied())
            .collect();
        let expected = pick_threshold(&residuals, &properties, &mut CoderScratch::default());
        let actual = pick_threshold_grouped(&groups, 0, &mut CoderScratch::default());
        assert_eq!(actual, expected);
    }

    #[test]
    fn added_predictors_match_jxl_formulas() {
        let n = PredictorNeighbors {
            left: 11,
            top: 7,
            top_left: 5,
            top_right: 13,
            left_left: 3,
            top_top: 2,
            top_right_right: 17,
        };
        assert_eq!(predictor_value(PREDICTOR_LEFT, n, 99), 11);
        assert_eq!(predictor_value(PREDICTOR_TOP, n, 99), 7);
        assert_eq!(predictor_value(PREDICTOR_SELECT, n, 99), 11);
        assert_eq!(predictor_value(PREDICTOR_AVERAGE4, n, 99), 11);

        // Select resolves equal distances toward Top, matching libjxl's
        // `pa < pb ? left : top`.
        let tie = PredictorNeighbors {
            left: 9,
            top: 3,
            top_left: 6,
            ..n
        };
        assert_eq!(predictor_value(PREDICTOR_SELECT, tie, 99), 3);
    }

    #[test]
    fn slow_search_selects_directional_predictors() {
        const W: usize = 64;
        const H: usize = 64;
        let mut horizontal = vec![0i32; W * H];
        for y in 0..H {
            horizontal[y * W] = ((y * 73) & 255) as i32;
            for x in 1..W {
                let delta = (((x * 17 + y * 31) ^ (x * y * 3)) % 7) as i32 - 3;
                horizontal[y * W + x] = horizontal[y * W + x - 1] + delta;
            }
        }
        let mut vertical = vec![0i32; W * H];
        for x in 0..W {
            vertical[x] = ((x * 73) & 255) as i32;
            for y in 1..H {
                let delta = (((y * 17 + x * 31) ^ (x * y * 3)) % 7) as i32 - 3;
                vertical[y * W + x] = vertical[(y - 1) * W + x] + delta;
            }
        }

        assert_eq!(
            choose_predictor_for_plane(|x, y| horizontal[y * W + x], W, H),
            PREDICTOR_SELECT
        );
        assert_eq!(
            choose_predictor_for_plane(|x, y| vertical[y * W + x], W, H),
            PREDICTOR_SELECT
        );
    }

    #[test]
    fn slow_search_selects_average4_for_its_recurrence() {
        const W: usize = 64;
        const H: usize = 64;
        let mut plane = vec![0i32; W * H];
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                if y < 2 || x < 2 {
                    plane[i] = (((x * 97 + y * 53) ^ (x * y * 11)) & 1023) as i32 - 512;
                    continue;
                }
                let top_right = if x + 1 < W {
                    plane[(y - 1) * W + x + 1]
                } else {
                    plane[(y - 1) * W + x]
                };
                let top_right_right = if x + 2 < W {
                    plane[(y - 1) * W + x + 2]
                } else {
                    top_right
                };
                plane[i] = (6 * plane[(y - 1) * W + x] - 2 * plane[(y - 2) * W + x]
                    + 7 * plane[i - 1]
                    + plane[i - 2]
                    + top_right_right
                    + 3 * top_right
                    + 8)
                    / 16;
            }
        }

        assert_eq!(
            choose_predictor_for_plane(|x, y| plane[y * W + x], W, H),
            PREDICTOR_AVERAGE4
        );
    }

    #[test]
    fn directional_wp_presets_are_distinct() {
        const W: usize = 128;
        const H: usize = 128;
        let mut west = vec![0i32; W * H];
        let mut north = vec![0i32; W * H];
        for y in 0..H {
            west[y * W] = ((y * 977) & 65535) as i32;
            for x in 1..W {
                west[y * W + x] = west[y * W + x - 1] + ((x * 13 + y * 7) % 5) as i32 - 2;
            }
        }
        for x in 0..W {
            north[x] = ((x * 977) & 65535) as i32;
            for y in 1..H {
                north[y * W + x] = north[(y - 1) * W + x] + ((y * 13 + x * 7) % 5) as i32 - 2;
            }
        }
        let west_costs: Vec<_> = WpParams::PRESETS
            .iter()
            .map(|&params| wp_sample_cost(|x, y| west[y * W + x], W, H, params).0)
            .collect();
        let north_costs: Vec<_> = WpParams::PRESETS
            .iter()
            .map(|&params| wp_sample_cost(|x, y| north[y * W + x], W, H, params).0)
            .collect();
        assert_ne!(west_costs[2], west_costs[3]);
        assert_ne!(north_costs[2], north_costs[3]);
    }

    #[test]
    fn fused_wp_row_kernels_match_point_fetched_reference() {
        for &(width, height) in &[(0, 3), (1, 1), (1, 4), (2, 2), (7, 5), (17, 4)] {
            let plane: Vec<i32> = (0..width * height)
                .map(|i| {
                    let x = i % width.max(1);
                    let y = i / width.max(1);
                    (((x * 977 + y * 619) ^ (x * y * 37)) as i32 & 0x7fff) - 0x3fff
                })
                .collect();
            for &params in &WpParams::PRESETS {
                let get = |x: usize, y: usize| plane[y * width + x];
                let mut reference = Vec::with_capacity(width * height);
                let mut wp = WpState::with_params(width, params);
                for y in 0..height {
                    for x in 0..width {
                        let value = get(x, y) as i64;
                        let n = predictor_neighbors(&get, x, y, width);
                        let prediction =
                            wp.predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
                        wp.update(value, x, y);
                        reference.push((11, pack_signed((value - prediction) as i32)));
                    }
                }

                let mut scratch = GradientScratch::default();
                let mut fused = Vec::with_capacity(width * height);
                tokenize_plane_with_wp(
                    11,
                    get,
                    width,
                    height,
                    PREDICTOR_WEIGHTED,
                    grad_pack_interior_scalar,
                    &mut scratch,
                    &mut fused,
                    params,
                );
                let fused: Vec<_> = fused
                    .iter()
                    .map(|token| (token.context, token.value))
                    .collect();
                assert_eq!(fused, reference, "{width}x{height}, params={params:?}");
            }
        }
    }
}

#[cfg(test)]
mod lz_reach_probe {

    use super::quantize_xyb_channels;
    use crate::image::Image3F;
    use crate::quant_weights::INV_DC_QUANT;

    #[test]
    fn slow_lossless_encode_of_repetitive_image() {
        // Highly repetitive => lz_has_repetition() is true => deep path runs.
        let (w, h) = (512usize, 512usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = (((x / 8) + (y / 8)) % 2) as u8 * 255;
                let i = (y * w + x) * 3;
                rgb[i] = v;
                rgb[i + 1] = v;
                rgb[i + 2] = v;
            }
        }
        let cfg = crate::EncodeConfig::default()
            .with_lossless(true)
            .with_speed(crate::Speed::Slow)
            .with_num_threads(1);
        let out = crate::encode_image(&rgb, w, h, &cfg).expect("encode failed");
        eprintln!("PROBE encoded {} bytes", out.len());
        assert!(!out.is_empty());
    }

    #[test]
    fn packed_quantization_matches_the_channel_formula() {
        let mut atlas = Image3F::new(5, 3);
        for i in 0..15 {
            atlas.plane_row_mut(0, i / 5)[i % 5] = (i as f32 - 7.0) / 41.0;
            atlas.plane_row_mut(1, i / 5)[i % 5] = i as f32 / 29.0;
            atlas.plane_row_mut(2, i / 5)[i % 5] = (15 - i) as f32 / 31.0;
        }

        for lattice_scale in [1, 2, 8] {
            let got = quantize_xyb_channels(&atlas, lattice_scale);
            let m = lattice_scale as f32;
            for i in 0..15 {
                let yq = (atlas.plane_data(1)[i] * INV_DC_QUANT[1] * m).round() as i32;
                assert_eq!(got[0][i], yq);
                assert_eq!(
                    got[1][i],
                    (atlas.plane_data(0)[i] * INV_DC_QUANT[0] * m).round() as i32
                );
                assert_eq!(
                    got[2][i],
                    (atlas.plane_data(2)[i] * INV_DC_QUANT[2] * m).round() as i32 - yq
                );
            }
        }
    }
}
