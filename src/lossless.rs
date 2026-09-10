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

#[path = "lossless_lz77.rs"]
mod lz77;
#[path = "lossless_ma_lookup.rs"]
mod ma_lookup;
#[path = "lossless_palette.rs"]
mod palette;
#[path = "lossless_predictor.rs"]
mod predictor;
#[path = "lossless_rct.rs"]
mod rct;
#[path = "lossless_squeeze.rs"]
mod squeeze;
#[path = "lossless_tokens.rs"]
mod tokens;
use ma_lookup::MaLookup;
use tokens::{RawTokenStreams, RawTokens};

use crate::adaptive_quant::dirty_log2f;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::{CoderScratch, LZ77_MAX_CONTEXTS};
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, uint_encode, write_entropy_code,
    write_token,
};
use crate::image::Image3Si;
use crate::ma_tree::{
    LearnedTree, MA_REF_CHANNELS, MaLearnParams, MaNode, MaSamples, NUM_MA_PREDS, NUM_MA_PROPS,
    deepen_ma_tree, learn_ma_tree, learn_ma_tree_indexed,
};
use crate::patches::{
    ModularFrameKind, NUM_PATCH_CONTEXTS, PATCH_REF_ID, PATCH_TILE, PatchReference,
    find_lossless_patches,
};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpNeighbors, WpParams, WpState, write_wp_header};
use crate::xyb::quantize_xyb_channels;
pub(crate) use lz77::LzToken;
use lz77::{
    LZ77_MIN_SYMBOL, RunLzWriter, build_lz_pixel_code, build_lz_pixel_code_threads,
    estimate_coded_bits, estimate_literal_and_run_bits, estimate_streams_bits,
    lz77_compress_channels_for_speed, lz77_compress_channels_for_speed_with_depth,
    lz77_compress_for_speed, lz77_compress_for_speed_with_depth, lz77_run_count,
    write_local_tree_lz77, write_lz_section, write_tree_lz77,
};
use palette::{
    PALETTE_COARSE_MARGIN, PALETTE_FINAL_MARGIN, build_global_palette,
    try_encode_local_palette_multi_group, try_encode_palette_single_group, write_palette_transform,
};
pub(crate) use predictor::{GradPackInteriorFn, GradientScratch, selected_grad_pack_interior_fn};
use predictor::{
    PREDICTOR_GRADIENT, PREDICTOR_WEIGHTED, PredictorNeighbors, channel_to_context,
    choose_predictor_for_plane, choose_predictors_with_wp, choose_wp_params, fixed_predictor,
    predictor_neighbors, predictor_value, tokenize_all, tokenize_channels_with_wp, tokenize_plane,
    tokenize_plane_rows, tokenize_runs_with_wp,
};
pub(crate) use rct::forward_ycocg;
use rct::{rank_rcts, rct_planes as rct_planes_fn, write_rct_transform};
pub(crate) use squeeze::{encode_frame_lossy_modular_squeeze, lm_calibrated_distance};
use squeeze::{encode_squeeze_multigroup, encode_squeeze_single_group};
use std::sync::{Condvar, Mutex};

const TREE_CTX_SPLIT_VAL: u32 = 0;
const TREE_CTX_PROPERTY: u32 = 1;
const TREE_CTX_PREDICTOR: u32 = 2;
const TREE_CTX_OFFSET: u32 = 3;
const TREE_CTX_MULTIPLIER_LOG: u32 = 4;
const TREE_CTX_MULTIPLIER_BITS: u32 = 5;
const NUM_TREE_CONTEXTS: usize = 6;

/// Modular group layout: the frame header's `group_size_shift`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GroupLayout {
    shift: u32,
}

impl GroupLayout {
    /// 256-pixel groups: the codestream default and libjxl's lossless choice.
    pub(crate) const DEFAULT: Self = Self { shift: 1 };
    /// 1024-pixel groups. Every group edge costs its first row and column
    /// their north/west context and restarts the WP state, so Slow lossless
    /// prefers the largest groups whenever the learner's estimate agrees.
    pub(crate) const LARGE: Self = Self { shift: 3 };

    pub(crate) const fn dim(self) -> usize {
        128 << self.shift
    }

    pub(crate) const fn lf_dim(self) -> usize {
        self.dim() * 8
    }
}

const GROUP_DIM: usize = GroupLayout::DEFAULT.dim();
const LF_GROUP_DIM: usize = GroupLayout::DEFAULT.lf_dim();
/// Each active deep-LZ lane owns about 4.7 MiB of hash lookup storage. Eight
/// lanes retain useful parallelism without allowing a many-core encoder to
/// multiply that fixed working set across every worker.
const SLOW_DEEP_LZ_MAX_THREADS: usize = 8;

/// The learned-tree entropy proxy must beat its flat alternative decisively
/// before Slow mode trusts it enough to omit the context/flat final encodes.
/// Near decisions retain the exact byte-level tournament.
const MA_DECISIVE_MIN_SAVINGS: f64 = 0.18;

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
        let depth = {
            let mut available = self.available.lock().unwrap();
            while available.is_empty() {
                available = self.activity.wait(available).unwrap();
            }
            available.pop().unwrap()
        };

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

/// Encode one single-channel modular group section from precomputed
/// gradient-predictor residual tokens (token contexts are ignored): a local
/// single-leaf Gradient tree, LZ77 run collapse, and the rANS-capable pixel
/// entropy code shared with the lossless path. Used by the lossy extra-channel
/// (alpha) path, where near-constant planes otherwise pay the 1 bit/pixel
/// prefix-code floor. Returns false (writing nothing) when a residual symbol
/// would collide with the LZ77 length alphabet (huge residuals, e.g. raw-float
/// alpha bits).
pub(crate) fn write_single_channel_lz77_section(
    tokens: &[Token],
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) -> bool {
    if tokens
        .iter()
        .any(|t| crate::entropy::uint_encode(t.value).0 >= LZ77_MIN_SYMBOL)
    {
        return false;
    }
    let mut runs = RunLzWriter::with_capacity(tokens.len());
    for t in tokens {
        runs.push(Token::new(0, t.value));
    }
    let lz = runs.finish();
    let code = build_lz_pixel_code(
        std::iter::once(lz.as_slice()),
        1,
        LZ77_MIN_SYMBOL,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    // GroupHeader: use_global_tree=0, wp all_default=1, 0 transforms.
    w.write(1, 0);
    w.write(1, 1);
    w.write(2, 0);
    write_local_tree_lz77(
        &[PREDICTOR_GRADIENT],
        &code,
        LZ77_MIN_SYMBOL,
        &mut scratch.huffman_pool,
        w,
    );
    write_lz_section(&lz, 1, &code, LZ77_MIN_SYMBOL, w);
    true
}

/// Independent streams share frozen entropy tables but retain their own ANS
/// state. Bound simultaneous staging memory as well as the worker count.
fn write_lz_groups(
    groups: &[Vec<LzToken>],
    code: &crate::entropy::EntropyCode<'_>,
    distance_context: u32,
    min_symbol: u32,
    wp_params: WpParams,
    pool: &ThreadPool,
    sections: &mut [BitWriter],
) {
    write_lz_groups_with_header(
        groups,
        code,
        distance_context,
        min_symbol,
        pool,
        sections,
        |_, section| {
            section.write(1, 1);
            write_wp_header(wp_params, section);
            section.write(2, 0);
        },
    );
}

fn write_lz_groups_with_header(
    groups: &[Vec<LzToken>],
    code: &crate::entropy::EntropyCode<'_>,
    distance_context: u32,
    min_symbol: u32,
    pool: &ThreadPool,
    sections: &mut [BitWriter],
    write_header: impl Fn(usize, &mut BitWriter) + Sync,
) {
    debug_assert_eq!(groups.len(), sections.len());
    let max_staging = if code.use_prefix_code {
        0
    } else {
        groups
            .iter()
            .map(|g| g.len().saturating_mul(8))
            .max()
            .unwrap_or(0)
    };
    #[allow(clippy::manual_clamp)]
    let lanes = (32 * 1024 * 1024 / max_staging.max(1))
        .max(1)
        .min(SLOW_DEEP_LZ_MAX_THREADS)
        .min(pool.num_threads())
        .min(sections.len());
    let write_group = |i: usize, section: &mut BitWriter| {
        write_header(i, section);
        write_lz_section(&groups[i], distance_context, code, min_symbol, section);
        section.zero_pad_to_byte();
    };
    if lanes <= 1 {
        for (i, section) in sections.iter_mut().enumerate() {
            write_group(i, section);
        }
        return;
    }
    // The caller's entropy scratch is borrowed by `code`. This lightweight
    // scratch supplies the caller lane; section writers need no worker tables.
    let mut writer_scratch = CoderScratch::lossless();
    pool.steal_for_each_mut_with_threads(&mut writer_scratch, sections, lanes, |i, section, _| {
        write_group(i, section);
    });
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
    decoding_speed: crate::DecodingSpeed,
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
        decoding_speed,
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
    decoding_speed: crate::DecodingSpeed,
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
        let decisive_tree = encode_frame_lossless_core(
            linear,
            alpha,
            max_bits,
            progressive,
            num_color,
            speed,
            decoding_speed,
            pool,
            scratch,
            ModularFrameKind::Regular,
            &mut regular_writer,
        );
        // Patched frames are coded by the flat path only, so they cannot beat
        // a decisive learned tree, and a plan covering little of the image
        // cannot pay for its dictionary and reference frame.
        let covered: usize = plan
            .references
            .iter()
            .map(|r| r.positions.len())
            .sum::<usize>()
            * crate::patches::PATCH_TILE
            * crate::patches::PATCH_TILE;
        let coverage = covered as f64 / (linear.xsize() * linear.ysize()) as f64;
        if decisive_tree || coverage < PATCH_MIN_COVERAGE {
            writer.append(&regular_writer);
            return;
        }

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
            decoding_speed,
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
            decoding_speed,
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
        decoding_speed,
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
    decoding_speed: crate::DecodingSpeed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    frame_kind: ModularFrameKind<'_>,
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
        pool,
        scratch,
        frame_kind,
        true,
        writer,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_frame_lossless_core_impl(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    num_color: usize,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    frame_kind: ModularFrameKind<'_>,
    allow_palette_candidates: bool,
    writer: &mut BitWriter,
) -> bool {
    let use_wp = decoding_speed.use_weighted_predictor();
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
            use_wp,
            scratch,
            &mut palette_writer,
        ) {
            if speed == crate::Speed::Slow {
                let mut normal_writer = BitWriter::new();
                let normal_decisive = encode_frame_lossless_core_impl(
                    linear,
                    alpha,
                    max_bits,
                    progressive,
                    num_color,
                    speed,
                    decoding_speed,
                    pool,
                    scratch,
                    frame_kind,
                    false,
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
            pool,
            scratch,
            &mut palette_writer,
        ) {
            if speed == crate::Speed::Slow {
                let mut normal_writer = BitWriter::new();
                let normal_decisive = encode_frame_lossless_core_impl(
                    linear,
                    alpha,
                    max_bits,
                    progressive,
                    num_color,
                    speed,
                    decoding_speed,
                    pool,
                    scratch,
                    frame_kind,
                    false,
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
    let adaptive_search = speed == crate::Speed::Slow;

    // RCT re-selection (Slow, regular RGB frames): estimated over the same
    // fixed-context gradient cost libjxl uses. When a non-YCoCg transform
    // wins, the color planes are rebuilt and the frame headers signal it.
    let mut rct_type = 6u32;
    let mut rct_planes: Option<Image3Si> = None;
    let rct_ranked: Vec<(u32, f32)> =
        if frame_kind.is_regular() && adaptive_search && num_color == 3 {
            rank_rcts(linear, xsize, ysize, pool, scratch)
        } else {
            Vec::new()
        };
    if let Some(&(t, _)) = rct_ranked.first()
        && t != 6
    {
        rct_type = t;
        rct_planes = Some(rct_planes_fn(linear, xsize, ysize, t));
    }
    // The palette candidate below needs the untransformed YCoCg planes.
    let ycocg = linear;
    let linear = rct_planes.as_ref().unwrap_or(linear);

    let wp_header_count = if single_group {
        1
    } else {
        2 + num_dc_groups + num_ac_groups
    };
    let mut wp_params = if adaptive_search && use_wp {
        choose_wp_params(linear, alpha, num_color, wp_header_count, pool, scratch)
    } else {
        WpParams::DEFAULT
    };
    // The estimates below remain useful prefilters, but Slow mode keeps every
    // viable finalist and compares its complete byte-aligned frame against the
    // v1 and flat alternatives. Tree headers, clustering, hybrid uint choices,
    // and LZ77 can otherwise reverse the order predicted from residual entropy.
    let compare_tree_candidates = frame_kind.is_regular()
        && adaptive_search
        && num_color == 3
        && decoding_speed.use_ma_trees();
    let mut best_tree_writer: Option<BitWriter> = None;

    // Sources: the RGB(A) planes under the best transform and, for frames
    // with <= 1024 colors, a global palette index image. The corner-sampled
    // WP proxy and the gradient RCT estimator both mis-rank on a third of
    // Kodak (up to 0.7% of the file), so a reduced-leaf learner ranks every
    // WP preset per source (and the transform runner-up for RGB). First the
    // two sources are compared by one first-stage learn each under the proxy
    // preset: a source trailing the leader by more than
    // PALETTE_COARSE_MARGIN is dropped entirely (on palette graphics the RGB
    // tree is ~25% behind and was paying the whole ranking for nothing).
    // Each ranking winner seeds its final learn.
    let (layout, learned_tree_decisive) = {
        let palette = if compare_tree_candidates {
            build_global_palette(ycocg, alpha, xsize, ysize, pool, scratch)
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
        let mut rank_pal = compare_tree_candidates && palette.is_some();
        let mut rgb_seed: Option<RankedLearn> = None;
        let mut pal_seed: Option<RankedLearn> = None;
        let mut pal_wp = wp_params;
        if compare_tree_candidates && matches!(decoding_speed, crate::DecodingSpeed::Slow) {
            let rgb_stage = {
                let source = rgb_ma_source(linear, alpha, xsize, rct_type);
                rank_coarse(
                    &source, xsize, ysize, min_symbol, wp_params, use_wp, pool, scratch,
                )
            };
            let pal_stage = palette.as_ref().and_then(|p| {
                let source = p.source(xsize, ysize);
                rank_coarse(
                    &source, xsize, ysize, min_symbol, wp_params, use_wp, pool, scratch,
                )
            });
            let coarse_of = |stage: &Option<CoarseLearn>| {
                stage.as_ref().map_or(f64::INFINITY, |s| {
                    s.candidate.est_real + header_bits(wp_params)
                })
            };
            let (rgb_coarse, pal_coarse) = (coarse_of(&rgb_stage), coarse_of(&pal_stage));
            rank_rgb = rgb_coarse <= pal_coarse * PALETTE_COARSE_MARGIN;
            rank_pal = pal_stage.is_some() && pal_coarse <= rgb_coarse * PALETTE_COARSE_MARGIN;

            if rank_rgb {
                let (mut best_bits, mut best_coarse, params, seed) = {
                    let source = rgb_ma_source(linear, alpha, xsize, rct_type);
                    rank_presets(
                        &source,
                        xsize,
                        ysize,
                        min_symbol,
                        use_wp,
                        rgb_stage,
                        wp_params,
                        &header_bits,
                        pool,
                        scratch,
                    )
                };
                wp_params = params;
                rgb_seed = seed;
                // Transform runner-up with the chosen preset, same coarse prune.
                if let Some(&(runner_up, cost)) = rct_ranked.get(1)
                    && cost <= rct_ranked[0].1 * RCT_RANK_MARGIN
                {
                    let planes =
                        (runner_up != 6).then(|| rct_planes_fn(ycocg, xsize, ysize, runner_up));
                    let source =
                        rgb_ma_source(planes.as_ref().unwrap_or(ycocg), alpha, xsize, runner_up);
                    if let Some(stage) = rank_coarse(
                        &source, xsize, ysize, min_symbol, wp_params, use_wp, pool, scratch,
                    ) && stage.candidate.est_real + header_bits(wp_params)
                        <= best_coarse * RANK_COARSE_PRUNE
                    {
                        best_coarse =
                            best_coarse.min(stage.candidate.est_real + header_bits(wp_params));
                        let (bits, ranked) = rank_finish(stage, min_symbol, use_wp, pool, scratch);
                        if bits + header_bits(wp_params) < best_bits {
                            best_bits = bits + header_bits(wp_params);
                            rct_type = runner_up;
                            rct_planes = planes;
                            rgb_seed = Some(ranked);
                        }
                    }
                }
                let _ = (best_bits, best_coarse);
            }
            if rank_pal && let Some(p) = &palette {
                let source = p.source(xsize, ysize);
                let (_, _, params, seed) = rank_presets(
                    &source,
                    xsize,
                    ysize,
                    min_symbol,
                    use_wp,
                    pal_stage,
                    wp_params,
                    &header_bits,
                    pool,
                    scratch,
                );
                pal_wp = params;
                pal_seed = seed;
            }
        }
        let linear = rct_planes.as_ref().unwrap_or(ycocg);

        // Group layout. With every decoding tool available, Slow lossless runs
        // the first learning stage (sampling plus a 32-leaf tree) under both
        // 1024- and 256-pixel groups. The 1024 layout wins on 22/24 Kodak images
        // and is always deepened; the 256 layout is deepened only when its first
        // stage already estimates smaller, and the full estimates then decide
        // (23/24 correct there, where the first stage alone is right 18/24).
        // Layouts are processed one at a time so a single sample set is alive.
        // Everything else keeps the codestream default.
        let layouts: &[GroupLayout] =
            if compare_tree_candidates && matches!(decoding_speed, crate::DecodingSpeed::Slow) {
                &[GroupLayout::LARGE, GroupLayout::DEFAULT]
            } else {
                &[GroupLayout::DEFAULT]
            };
        let (layout, rgb_coarse_est, mut learned_tree_decisive) = {
            let rgb_source = rgb_ma_source(linear, alpha, xsize, rct_type);
            let mut learned_estimated_savings: Option<f64> = None;
            let learned = if rank_rgb {
                learn_best_layout(
                    &rgb_source,
                    xsize,
                    ysize,
                    layouts,
                    min_symbol,
                    wp_params,
                    use_wp,
                    f64::INFINITY,
                    false,
                    rgb_seed,
                    pool,
                    scratch,
                )
            } else {
                None
            };
            let rgb_coarse_est = learned.as_ref().map_or(f64::INFINITY, |(_, _, e)| *e);
            rgb_final_est = learned
                .as_ref()
                .map_or(f64::INFINITY, |(_, c, _)| c.est_real);
            let layout = learned.as_ref().map_or(layouts[0], |(l, _, _)| *l);

            // Learned MA context tree (v2): greedy learned tree over the standard
            // property vector with per-leaf predictors.
            if let Some((_, cand, _)) = &learned {
                let mut candidate = BitWriter::new();
                let estimated_savings = write_learned_tree_frame(
                    &rgb_source,
                    alpha.is_some(),
                    xsize,
                    ysize,
                    layout,
                    min_symbol,
                    pool,
                    scratch,
                    wp_params,
                    use_wp,
                    cand,
                    &mut candidate,
                );
                learned_estimated_savings = Some(estimated_savings);
                keep_smaller_writer(&mut best_tree_writer, candidate);
            }
            (
                layout,
                rgb_coarse_est,
                learned_estimated_savings.is_some_and(learned_tree_is_decisive),
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
        if rank_pal && let Some(palette) = &palette {
            let palette_source = palette.source(xsize, ysize);
            let coarse_limit = rgb_coarse_est * PALETTE_COARSE_MARGIN;
            if let Some((palette_layout, cand, _)) = learn_best_layout(
                &palette_source,
                xsize,
                ysize,
                layouts,
                min_symbol,
                pal_wp,
                use_wp,
                coarse_limit,
                true,
                pal_seed,
                pool,
                scratch,
            ) {
                let palette_can_win = cand.est_real <= rgb_final_est * PALETTE_FINAL_MARGIN;
                // The frame write (tokenize, code, sections) is the palette candidate's
                // most expensive step after the learn; a final estimate clearly behind
                // the RGB frame's cannot win the size comparison (a 3,195-color
                // screenshot: ratio 1.24, frame +19%; a gray-image tie at 1.000 still writes).
                if palette_can_win {
                    let mut candidate = BitWriter::new();
                    let estimated_savings = write_learned_tree_frame(
                        &palette_source,
                        alpha.is_some(),
                        xsize,
                        ysize,
                        palette_layout,
                        min_symbol,
                        pool,
                        scratch,
                        pal_wp,
                        use_wp,
                        &cand,
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
        (layout, learned_tree_decisive)
    };

    if learned_tree_decisive {
        writer.append(
            best_tree_writer
                .as_ref()
                .expect("decisive learned-tree candidate"),
        );
        return true;
    }

    let gdim = layout.dim();
    let xsize_groups = xsize.div_ceil(gdim);
    let ysize_groups = ysize.div_ceil(gdim);
    let num_ac_groups = xsize_groups * ysize_groups;
    let num_dc_groups = xsize.div_ceil(layout.lf_dim()) * ysize.div_ceil(layout.lf_dim());
    let single_group = num_ac_groups == 1;

    let linear = rct_planes.as_ref().unwrap_or(ycocg);
    // Per-channel predictor search for the v1 context tree and flat paths
    // (deferred: a decisive learned tree never needs it).
    let predictors = if adaptive_search {
        choose_predictors_with_wp(
            linear, alpha, xsize, ysize, num_color, pool, scratch, wp_params, use_wp,
        )
    } else {
        [fixed_predictor(use_wp); 4]
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

    // Context tree (v1): single-group. Splits each channel's entropy context on
    // the WP activity property; a big win on smooth+edge content. Falls through
    // to the flat path when it isn't estimated to help.
    if compare_tree_candidates && use_wp {
        let mut candidate = BitWriter::new();
        if single_group {
            if try_encode_context_tree_single_group(
                linear,
                alpha,
                xsize,
                ysize,
                layout,
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
            layout,
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
            let group_lz_tokens: Vec<Vec<LzToken>> = {
                let deep_lz = (speed == crate::Speed::Slow)
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
                )
            };
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

fn write_frame_header_modular_kind(
    has_alpha: bool,
    kind: ModularFrameKind<'_>,
    layout: GroupLayout,
    w: &mut BitWriter,
) {
    match kind {
        ModularFrameKind::Regular => write_frame_header_modular(has_alpha, layout, w),
        ModularFrameKind::Patched(_) => write_frame_header_modular_flags(has_alpha, 2, layout, w),
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
            w.write(2, layout.shift as u64);
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
    w.write(2, GroupLayout::DEFAULT.shift as u64);
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
    use_wp: bool,
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
    let [y, x, b] = &ch;
    let pixel_colors = || y[..npx].iter().zip(&x[..npx]).zip(&b[..npx]);
    for ((&y, &x), &b) in pixel_colors() {
        seen.entry([y, x, b]).or_insert(());
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
        use crate::quant_weights::INV_DC_QUANT;
        use crate::util::f32_to_f16_bits;
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
        for (c, row) in palette_ch[..3 * nb_colors]
            .chunks_exact_mut(nb_colors)
            .enumerate()
        {
            for (value, color) in row.iter_mut().zip(&colors) {
                *value = color[c];
            }
        }
        let index_img: Vec<i32> = pixel_colors()
            .map(|((&y, &x), &b)| idx_of[&[y, x, b]] as i32)
            .collect();

        write_palette_transform(num_c as u32, nb_colors as u32, &mut section);
        let pget = |gx: usize, gy: usize| palette_ch[gy * nb_colors + gx];
        let iget = |gx: usize, gy: usize| index_img[gy * xsize + gx];
        let preds = if slow {
            vec![
                choose_predictor_for_plane(pget, nb_colors, num_c, use_wp),
                choose_predictor_for_plane(iget, xsize, ysize, use_wp),
            ]
        } else {
            vec![fixed_predictor(use_wp); 2]
        };
        tokenize_plane_rows(
            channel_to_context(0, 2),
            |y| &palette_ch[y * nb_colors..][..nb_colors],
            nb_colors,
            num_c,
            preds[0],
            grad_pack_fn,
            &mut scratch.gradient,
            &mut tokens,
        );
        tokenize_plane_rows(
            channel_to_context(1, 2),
            |y| &index_img[y * xsize..][..xsize],
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
                choose_predictor_for_plane(get, xsize, ysize, use_wp)
            } else {
                fixed_predictor(use_wp)
            };
            tokenize_plane_rows(
                channel_to_context(c, num_c),
                |y| &data[y * xsize..][..xsize],
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
                fixed_predictor(use_wp),
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
            preds.push(fixed_predictor(use_wp));
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

fn write_frame_header_modular(has_alpha: bool, layout: GroupLayout, w: &mut BitWriter) {
    write_frame_header_modular_flags(has_alpha, 0, layout, w);
}

fn write_frame_header_modular_flags(
    has_alpha: bool,
    flags: u64,
    layout: GroupLayout,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    write_u64(flags, w);
    w.write(1, 0); // do_ycbcr = false   (xyb_encoded=0 so this is serialized)
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(2, layout.shift as u64);
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

fn write_modular_transforms(nb_chans: usize, rct_type: u32, w: &mut BitWriter) {
    if nb_chans >= 3 {
        // transforms count u2S(0, 1, Bits(4)+2, Bits(8)+18): selector 1 = Val(1) → 1 transform.
        w.write(2, 0b01);
        write_rct_transform(rct_type, w);
    } else {
        w.write(2, 0b00); // 0 transforms
    }
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

/// Target sample count for tree learning (across all channels/groups). Photos
/// stop gaining well below this (Kodak: stride 1 is not better than 3), but
/// graphics with many near-deterministic contexts keep gaining: the 17 MP
/// Burning Ship fractal is −3.4% at 2M samples over 512K. Only one layout's
/// sample set is alive at a time (110 B per sample with two reference channels).
const MA_TARGET_SAMPLES: usize = 1 << 21;
/// Preserve full sampling for genuinely small images. Above the former 256K
/// target, a stride of three is both cheaper and less prone to fitting local
/// pixel-phase noise than switching abruptly to a full-image sample.
const MA_FULL_SAMPLE_LIMIT: usize = 1 << 22;
/// First-stage learner budget. Most unhelpful or simple trees terminate here;
/// only a saturated tree with a clear rate win is retrained on the full probe.
const MA_COARSE_TARGET_SAMPLES: usize = 1 << 16;
const MA_COARSE_MAX_LEAVES: usize = 32;
const MA_DEEPEN_MIN_SAVINGS: f64 = 0.02;
/// Bits a split must save (image domain) before it's kept. Growth is
/// best-first, so this mostly prunes the tail once the leaf budget is spent.
const MA_SPLIT_COST_BITS: f32 = 100.0;
/// Leaf cap: pixel contexts + the LZ77 distance context must fit the
/// LZ77_MAX_CONTEXTS scratch; ANS clustering reduces them to <= 128
/// histograms. Best-first growth adds exactly one leaf per split, so the cap
/// is hard.
const MA_MAX_LEAVES: usize = LZ77_MAX_CONTEXTS - 1;
/// Minimum samples in a node before it must become a leaf.
const MA_MIN_NODE_SAMPLES: usize = 128;
/// Predictors each side of a candidate split may choose from (the node's
/// cheapest ones).
const MA_SIDE_PREDS: usize = 4;

/// Walk one channel rectangle in scan order, feeding the visitor the property
/// vector (libjxl ids 0..=15), the neighborhood, and the WP prediction of
/// every pixel.
/// A previously coded channel of the same size as the one being walked,
/// cropped like it: the source of the decoder's reference properties.
#[derive(Clone, Copy)]
struct MaRefPlane<'a> {
    pixels: &'a [i32],
    stride: usize,
    x0: usize,
    y0: usize,
}

/// libjxl `PrecomputeReferences`: per reference channel |v|, v, |v − g|,
/// v − g, with left = 0 at the crop's left edge, top = left on its first row
/// and top-left = left on either edge, g the clamped gradient.
#[inline]
fn ref_props(refs: &[MaRefPlane<'_>], x: usize, y: usize, p: &mut [i32; NUM_MA_PROPS]) {
    for (k, r) in refs.iter().take(MA_REF_CHANNELS).enumerate() {
        let row = &r.pixels[(r.y0 + y) * r.stride + r.x0..];
        let v = row[x] as i64;
        let left = if x > 0 { row[x - 1] as i64 } else { 0 };
        let (top, top_left) = if y > 0 {
            let above = &r.pixels[(r.y0 + y - 1) * r.stride + r.x0..];
            (
                above[x] as i64,
                if x > 0 { above[x - 1] as i64 } else { left },
            )
        } else {
            (left, left)
        };
        let g = (left + top - top_left).clamp(left.min(top), left.max(top));
        let base = 16 + 4 * k;
        p[base] = v.abs() as i32;
        p[base + 1] = v as i32;
        p[base + 2] = (v - g).abs() as i32;
        p[base + 3] = (v - g) as i32;
    }
}

fn walk_channel_ma<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    refs: &[MaRefPlane<'a>],
    wp_params: WpParams,
    use_wp: bool,
    mut visit: impl FnMut(usize, usize, i64, &[i32; NUM_MA_PROPS], PredictorNeighbors, i64),
) where
    i64: From<T>,
{
    let disabled = !use_wp;
    let mut wp = WpState::with_params(gw, wp_params);
    let mut p = [0i32; NUM_MA_PROPS];
    p[0] = chan as i32;
    p[1] = stream;
    let mut north_row: &[T] = &[];
    let mut north_north_row: &[T] = &[];
    for y in 0..gh {
        let current_row = get_row(y);
        assert_eq!(current_row.len(), gw);
        // Bind valid rows even at the top border, so all three lengths are
        // established before entering the pixel loop.
        if y == 0 {
            north_row = current_row;
        }
        if y < 2 {
            north_north_row = north_row;
        }
        assert_eq!(north_row.len(), gw);
        assert_eq!(north_north_row.len(), gw);
        let row = wp.row_offsets(y);
        p[2] = y as i32;
        p[9] = 0; // "local gradient" carry, reset per row (InitPropsRow)
        // Keep the visitor in each loop body. Passing this large WP/property
        // update through a row callback can outline a call for every pixel.
        macro_rules! visit_pixel {
            ($x:expr, $value:expr, $neighbors:expr) => {{
                let x = $x;
                let value = $value;
                let n = $neighbors;
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
                let wp_pred = if disabled {
                    0
                } else {
                    wp.predict_and_update_flat(
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
                    )
                };
                p[15] = if disabled { 0 } else { wp.wp_prop as i32 };
                ref_props(refs, x, y, &mut p);
                visit(x, y, value, &p, n, wp_pred);
            }};
        }
        if y == 0 {
            if let Some((&first, tail)) = current_row.split_first() {
                let first = i64::from(first);
                visit_pixel!(
                    0,
                    first,
                    PredictorNeighbors {
                        left: 0,
                        top: 0,
                        top_left: 0,
                        top_right: 0,
                        left_left: 0,
                        top_top: 0,
                        top_right_right: 0,
                    }
                );
                let mut left = first;
                let mut left_left = first;
                for (x, &value) in tail.iter().enumerate() {
                    let value = i64::from(value);
                    visit_pixel!(
                        x + 1,
                        value,
                        PredictorNeighbors {
                            left,
                            top: left,
                            top_left: left,
                            top_right: left,
                            left_left,
                            top_top: left,
                            top_right_right: left,
                        }
                    );
                    left_left = left;
                    left = value;
                }
            }
        } else {
            let edge = |x: usize| {
                let top = i64::from(north_row[x]);
                let left = if x > 0 {
                    i64::from(current_row[x - 1])
                } else {
                    top
                };
                let top_right = north_row.get(x + 1).map_or(top, |&v| i64::from(v));
                PredictorNeighbors {
                    left,
                    top,
                    top_left: if x > 0 {
                        i64::from(north_row[x - 1])
                    } else {
                        left
                    },
                    top_right,
                    left_left: if x > 1 {
                        i64::from(current_row[x - 2])
                    } else {
                        left
                    },
                    top_top: i64::from(north_north_row[x]),
                    top_right_right: north_row.get(x + 2).map_or(top_right, |&v| i64::from(v)),
                }
            };
            for (x, &value) in current_row.iter().take(2).enumerate() {
                visit_pixel!(x, i64::from(value), edge(x));
            }
            if gw > 4 {
                let mut left = i64::from(current_row[1]);
                let mut left_left = i64::from(current_row[0]);
                let pixels = current_row[2..gw - 2]
                    .iter()
                    .zip(north_row[1..].array_windows::<4>())
                    .zip(&north_north_row[2..]);
                for (x, ((&value, &[nw, n, ne, nee]), &nn)) in pixels.enumerate() {
                    let value = i64::from(value);
                    visit_pixel!(
                        x + 2,
                        value,
                        PredictorNeighbors {
                            left,
                            top: i64::from(n),
                            top_left: i64::from(nw),
                            top_right: i64::from(ne),
                            left_left,
                            top_top: i64::from(nn),
                            top_right_right: i64::from(nee),
                        }
                    );
                    left_left = left;
                    left = value;
                }
            }
            for x in gw.saturating_sub(2).max(2)..gw {
                visit_pixel!(x, i64::from(current_row[x]), edge(x));
            }
        }
        north_north_row = north_row;
        north_row = current_row;
    }
}

/// Exact sample count for the rotating row phase.
fn ma_channel_sample_count(w: usize, h: usize, stride: usize) -> usize {
    // A complete cycle of row phases selects exactly w samples. In the
    // remaining rows, the first w % stride phases select one extra sample.
    (h / stride) * w + (h % stride) * (w / stride) + (h % stride).min(w % stride)
}

/// Visit the selected pixels with residual tokens under every decoder predictor.
fn sample_channel_ma<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    refs: &[MaRefPlane<'a>],
    wp_params: WpParams,
    use_wp: bool,
    stride: usize,
    mut push: impl FnMut([i32; NUM_MA_PROPS], [u8; NUM_MA_PREDS]),
) where
    i64: From<T>,
{
    debug_assert!(stride != 0);
    let disabled = !use_wp;
    let mut wp = WpState::with_params(gw, wp_params);
    let mut north_row: &[T] = &[];
    let mut north_north_row: &[T] = &[];
    for y in 0..gh {
        let current_row = get_row(y);
        assert_eq!(current_row.len(), gw);
        // Bind valid rows even at the top border, so all three lengths are
        // established before entering the pixel loop.
        if y == 0 {
            north_row = current_row;
        }
        if y < 2 {
            north_north_row = north_row;
        }
        assert_eq!(north_row.len(), gw);
        assert_eq!(north_north_row.len(), gw);
        let row = wp.row_offsets(y);
        let mut west = 0i64;
        let mut west_west = 0i64;
        let mut previous_local_gradient = 0i32;
        // Rotate the sampling phase every row so the probe never locks onto
        // one column residue (a row width divisible by the stride would
        // otherwise sample a single x phase for the whole channel).
        let mut until_sample = 1 + y % stride;
        for x in 0..gw {
            let value = i64::from(current_row[x]);
            let north = if y > 0 { i64::from(north_row[x]) } else { west };
            let left = if x > 0 { west } else { north };
            let top_left = if x > 0 && y > 0 {
                i64::from(north_row[x - 1])
            } else {
                left
            };
            let top_right = if x + 1 < gw && y > 0 {
                i64::from(north_row[x + 1])
            } else {
                north
            };
            let top_top = if y > 1 {
                i64::from(north_north_row[x])
            } else {
                north
            };
            let wp_pred = if disabled {
                0
            } else {
                let wp_neighbors = WpNeighbors {
                    north,
                    west: left,
                    north_east: top_right,
                    north_west: top_left,
                    north_north: top_top,
                };
                wp.predict_and_update_flat(value, x, row, wp_neighbors)
            };
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
                        i64::from(north_row[x + 2])
                    } else {
                        top_right
                    },
                };
                let mut props = [0i32; NUM_MA_PROPS];
                props[0] = chan as i32;
                props[1] = stream;
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
                props[15] = if disabled { 0 } else { wp.wp_prop as i32 };
                ref_props(refs, x, y, &mut props);

                let mut tok = [0u8; NUM_MA_PREDS];
                for pred in 0..NUM_MA_PREDS as u32 {
                    let pv = predictor_value(pred, neighbors, wp_pred);
                    let (t, _, _) = uint_encode(pack_signed((value - pv) as i32));
                    tok[pred as usize] = t.min(u8::MAX as u32) as u8;
                }
                push(props, tok);
            }
            previous_local_gradient = local_gradient;
            west_west = west;
            west = value;
        }
        north_north_row = north_row;
        north_row = current_row;
    }
}

/// Tokenize the channel rectangle through the learned tree: every pixel is
/// routed to its leaf's context and coded with its leaf's predictor.
fn tokenize_channel_ma<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    refs: &[MaRefPlane<'a>],
    wp_params: WpParams,
    use_wp: bool,
    lookup: &MaLookup,
    offsets: &[i32],
    out: &mut RawTokens,
) where
    i64: From<T>,
{
    // Keep the traversal's stores simple, then validate and pack one row at
    // a time. The temporary row fits in cache and is reused for the channel.
    let mut row = Vec::with_capacity(gw);
    walk_channel_ma(
        get_row,
        gw,
        gh,
        chan,
        stream,
        refs,
        wp_params,
        use_wp && lookup.needs_wp(),
        |x, _y, v, p, n, wp_pred| {
            let (context, pred) = lookup.lookup(p);
            let pv = predictor_value(pred, n, wp_pred);
            row.push(Token::new(
                context,
                pack_signed((v - pv) as i32 - offsets[context as usize]),
            ));
            if x + 1 == gw {
                out.extend(&row);
                row.clear();
            }
        },
    );
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

/// Largest |offset| a leaf may carry. The learner keeps hybrid-uint symbols,
/// which are exact packed residuals only below the split (symbols < 16 =
/// residuals −8..=7); larger symbols carry their sign in raw bits the sample
/// does not keep, so they only count towards the zero fraction.
const MA_LEAF_OFFSET_LIMIT: i32 = 8;
const MA_LEAF_EXACT_SYMBOLS: u32 = 16;
/// Leaves with fewer routed samples keep offset 0.
const MA_LEAF_OFFSET_MIN_SAMPLES: u32 = 16;
/// No offsets at all when more than this share of residuals is exactly zero:
/// such content (screenshots, fractals, flat graphics) has no predictor bias
/// to cancel.
const MA_LEAF_OFFSET_MAX_ZERO_FRACTION: f64 = 0.9;

fn unpack_signed(token: u32) -> i32 {
    if token & 1 == 1 {
        -(((token + 1) >> 1) as i32)
    } else {
        (token >> 1) as i32
    }
}

/// Per-context predictor offsets: the median residual of each leaf's chosen
/// predictor over the samples the finished tree routes to it. Shifting a
/// leaf's residuals does not change its own entropy, but it centres the
/// distribution, which lets ANS clustering merge leaves by shape rather than
/// by bias and keeps small residuals inside the hybrid-uint's exact symbols.
fn leaf_offsets(
    tree: &LearnedTree,
    leaf_ctx: &[u32],
    num_ctx: u32,
    samples: &MaSamples,
) -> Vec<i32> {
    const BINS: usize = 2 * MA_LEAF_OFFSET_LIMIT as usize + 1;
    let mut hist = vec![[0u32; BINS]; num_ctx as usize];
    // Every fourth sample: leaves keep >100 samples on average and the
    // routing pass drops below the noise of the encode time.
    let mut routed = 0u64;
    for (props, tok) in samples.props.iter().zip(&samples.tok).step_by(4) {
        let (node, pred) = tree.lookup(props);
        routed += 1;
        let symbol = tok[pred as usize] as u32;
        if symbol < MA_LEAF_EXACT_SYMBOLS {
            let r = unpack_signed(symbol);
            hist[leaf_ctx[node as usize] as usize][(r + MA_LEAF_OFFSET_LIMIT) as usize] += 1;
        }
    }
    let zeros: u64 = hist
        .iter()
        .map(|h| h[MA_LEAF_OFFSET_LIMIT as usize] as u64)
        .sum();
    if zeros as f64 > MA_LEAF_OFFSET_MAX_ZERO_FRACTION * routed as f64 {
        return vec![0; num_ctx as usize];
    }
    hist.iter()
        .map(|h| {
            let total: u32 = h.iter().sum();
            if total < MA_LEAF_OFFSET_MIN_SAMPLES {
                return 0;
            }
            let half = total.div_ceil(2);
            let mut acc = 0u32;
            for (bin, &c) in h.iter().enumerate() {
                acc += c;
                if acc >= half {
                    return bin as i32 - MA_LEAF_OFFSET_LIMIT;
                }
            }
            0
        })
        .collect()
}

/// Sampling stride for tree learning; odd to avoid column aliasing.
fn ma_sample_stride(total_px: usize, full_sample_limit: usize) -> usize {
    let mut stride = total_px.div_ceil(MA_TARGET_SAMPLES).max(1);
    if stride == 1 && total_px > full_sample_limit {
        stride = 3;
    }
    if stride > 1 && stride.is_multiple_of(2) {
        stride += 1;
    }
    stride
}

/// Header overhead model for a learned tree: tree tokens plus the extra
/// per-context histogram / context-map cost (conservative; clustering merges
/// most of it). Returns the candidate only when it beats the flat estimate.
/// `samples` is `Some` only for candidates that may be written: routing every
/// sample through the tree for the offsets is not worth doing for the
/// ranking stage's throwaway candidates.
fn gate_ma_tree(
    tree: LearnedTree,
    sample_scale: f64,
    samples: Option<&MaSamples>,
) -> Option<LearnedCandidate> {
    let (mut tree_tokens, leaf_ctx, num_ctx) = emit_learned_tree(&tree);
    let overhead_bits = tree_tokens.len() as f64 * 10.0 + num_ctx as f64 * 200.0;
    let est_real = tree.est_bits * sample_scale + overhead_bits;
    let flat_real = tree.flat_bits * sample_scale;
    if est_real.partial_cmp(&flat_real) != Some(std::cmp::Ordering::Less) {
        return None;
    }
    // Offsets do not affect the token-count gate. Route only accepted trees,
    // then fill their existing offset tokens in BFS context order.
    let leaf_offset = match samples {
        Some(samples) => leaf_offsets(&tree, &leaf_ctx, num_ctx, samples),
        None => vec![0; num_ctx as usize],
    };
    for (token, &offset) in tree_tokens
        .iter_mut()
        .filter(|token| token.context == TREE_CTX_OFFSET)
        .zip(&leaf_offset)
    {
        token.value = pack_signed(offset);
    }
    let estimated_savings = 1.0 - tree.est_bits / tree.flat_bits.max(f64::MIN_POSITIVE);
    Some(LearnedCandidate {
        tree,
        tree_tokens,
        leaf_ctx,
        leaf_offset,
        num_ctx,
        est_real,
        flat_real,
        estimated_savings,
    })
}

fn ma_learn_params(
    min_symbol: u32,
    max_leaves: usize,
    max_candidates: usize,
    split_scale: f64,
    use_wp: bool,
) -> MaLearnParams {
    MaLearnParams {
        alphabet: min_symbol as usize,
        max_leaves,
        split_cost_bits: MA_SPLIT_COST_BITS / split_scale as f32,
        min_node: MA_MIN_NODE_SAMPLES,
        allow_wp: use_wp,
        allowed_preds: u16::MAX,
        side_preds: MA_SIDE_PREDS,
        max_candidates,
    }
}

/// First learning stage: the merged samples with a gated coarse tree (or the
/// complete tree for small sample sets). Cheap enough to run once per group
/// layout; only the winning layout is deepened.
struct CoarseLearn {
    samples: MaSamples,
    stride: usize,
    max_candidates: usize,
    /// Leaf predictor offsets apply (RCT sources; palette indices lose).
    leaf_offsets: bool,
    candidate: LearnedCandidate,
    /// The coarse tree saturated its leaf budget with a clear win, so the
    /// full-probe second stage is worth its cost.
    deepen: bool,
}

fn learn_ma_coarse(
    samples: MaSamples,
    stride: usize,
    min_symbol: u32,
    use_wp: bool,
    max_leaves: usize,
    max_candidates: usize,
    leaf_offsets: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<CoarseLearn> {
    if samples.len() < 4 * MA_MIN_NODE_SAMPLES {
        return None;
    }
    if samples.len() <= MA_COARSE_TARGET_SAMPLES {
        let tree = learn_ma_tree(
            &samples,
            ma_learn_params(
                min_symbol,
                max_leaves,
                max_candidates,
                stride as f64,
                use_wp,
            ),
            pool,
            scratch,
        );
        let candidate = gate_ma_tree(tree, stride as f64, leaf_offsets.then_some(&samples))?;
        return Some(CoarseLearn {
            samples,
            stride,
            max_candidates,
            leaf_offsets,
            candidate,
            deepen: false,
        });
    }
    let coarse_indices = samples.evenly_sampled_indices(MA_COARSE_TARGET_SAMPLES);
    let coarse_scale = stride as f64 * samples.len() as f64 / coarse_indices.len() as f64;
    let coarse_tree = learn_ma_tree_indexed(
        &samples,
        coarse_indices,
        ma_learn_params(
            min_symbol,
            MA_COARSE_MAX_LEAVES,
            max_candidates,
            coarse_scale,
            use_wp,
        ),
        pool,
        scratch,
    );
    let candidate = gate_ma_tree(coarse_tree, coarse_scale, None)?;
    let coarse_leaves = candidate.tree.nodes.len().div_ceil(2);
    let coarse_savings = 1.0 - candidate.est_real / candidate.flat_real;
    let deepen = coarse_leaves == MA_COARSE_MAX_LEAVES && coarse_savings >= MA_DEEPEN_MIN_SAVINGS;
    Some(CoarseLearn {
        samples,
        stride,
        max_candidates,
        leaf_offsets,
        candidate,
        deepen,
    })
}

/// Second stage: re-score the coarse tree on the full probe and keep growing
/// it, falling back to the coarse tree when the deep one fails its gate.
fn finish_ma_learn(
    coarse: CoarseLearn,
    min_symbol: u32,
    use_wp: bool,
    max_leaves: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> LearnedCandidate {
    if !coarse.deepen {
        return coarse.candidate;
    }
    let (deep_tree, samples) = deepen_ma_tree(
        coarse.samples,
        ma_learn_params(
            min_symbol,
            max_leaves,
            coarse.max_candidates,
            coarse.stride as f64,
            use_wp,
        ),
        coarse.candidate.tree.clone(),
        pool,
        scratch,
    );
    gate_ma_tree(
        deep_tree,
        coarse.stride as f64,
        coarse.leaf_offsets.then_some(&samples),
    )
    .unwrap_or(coarse.candidate)
}

/// Rank the WP presets of `source` by the reduced-leaf learner, starting
/// from `first` (already learned under `first_params`), the rest in order
/// of how often they win, each deepened only when its first stage is within
/// `RANK_COARSE_PRUNE` of the best seen. Returns (best bits, best coarse
/// bits, preset, the winner's learn as a seed).
#[allow(clippy::too_many_arguments)]
fn rank_presets(
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    use_wp: bool,
    first: Option<CoarseLearn>,
    first_params: WpParams,
    header_bits: &dyn Fn(WpParams) -> f64,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (f64, f64, WpParams, Option<RankedLearn>) {
    let mut best_bits = f64::INFINITY;
    let mut best_coarse = f64::INFINITY;
    let mut best_params = first_params;
    let mut seed = None;
    let mut order: Vec<WpParams> = vec![first_params];
    if use_wp {
        for &preset in [1usize, 2, 3, 0].iter().map(|&i| &WpParams::PRESETS[i]) {
            if preset != first_params {
                order.push(preset);
            }
        }
    }
    let mut first = first;
    for preset in order {
        let stage = match first.take() {
            Some(stage) => Some(stage),
            None => rank_coarse(
                source, xsize, ysize, min_symbol, preset, use_wp, pool, scratch,
            ),
        };
        let Some(stage) = stage else {
            continue;
        };
        let coarse = stage.candidate.est_real + header_bits(preset);
        if coarse > best_coarse * RANK_COARSE_PRUNE {
            continue;
        }
        best_coarse = best_coarse.min(coarse);
        let (bits, ranked) = rank_finish(stage, min_symbol, use_wp, pool, scratch);
        let bits = bits + header_bits(preset);
        if bits < best_bits {
            best_bits = bits;
            best_params = preset;
            seed = Some(ranked);
        }
    }
    (best_bits, best_coarse, best_params, seed)
}

/// A ranking learn kept alive for the winner: its reduced-leaf tree seeds the
/// final learn, which then only grows from `MA_RANK_LEAVES` to the full
/// budget on the same (already compacted) sample set.
struct RankedLearn {
    candidate: LearnedCandidate,
    samples: MaSamples,
    stride: usize,
    max_candidates: usize,
    leaf_offsets: bool,
    coarse_est: f64,
}

impl RankedLearn {
    /// Continue best-first growth to the full leaf budget.
    fn finish(
        self,
        min_symbol: u32,
        use_wp: bool,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> LearnedCandidate {
        let (tree, samples) = deepen_ma_tree(
            self.samples,
            ma_learn_params(
                min_symbol,
                MA_MAX_LEAVES,
                self.max_candidates,
                self.stride as f64,
                use_wp,
            ),
            self.candidate.tree.clone(),
            pool,
            scratch,
        );
        gate_ma_tree(
            tree,
            self.stride as f64,
            self.leaf_offsets.then_some(&samples),
        )
        .unwrap_or(self.candidate)
    }
}

/// A gated learned tree with its BFS emission and rate estimate.
struct LearnedCandidate {
    tree: LearnedTree,
    tree_tokens: Vec<Token>,
    leaf_ctx: Vec<u32>,
    /// Predictor offset per context (see `leaf_offsets`).
    leaf_offset: Vec<i32>,
    num_ctx: u32,
    /// Estimated image-domain bits of the tree-coded frame, headers included.
    est_real: f64,
    /// Estimated image-domain bits of the flat alternative.
    flat_real: f64,
    /// Estimated fractional saving over the flat path.
    estimated_savings: f64,
}

/// One modular channel as the learner and tokenizer see it.
struct MaChannel<'a> {
    w: usize,
    h: usize,
    /// Meta channels (palette) always live in the global stream, whatever
    /// their size.
    meta: bool,
    pixels: MaPixels<'a>,
}

/// Borrow the original sample representation, including palette indices and
/// alpha, without expanding or copying rows for MA traversal.
enum MaPixels<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    I32(&'a [i32]),
}

// Resolve the sample type once per rectangle so the row kernels remain
// monomorphized. All variants provide the same cropped, borrowed row interface.
macro_rules! with_ma_pixels {
    ($channel:expr, |$pixels:ident| $body:expr) => {
        match &$channel.pixels {
            MaPixels::U8($pixels) => $body,
            MaPixels::U16($pixels) => $body,
            MaPixels::I32($pixels) => $body,
        }
    };
}

/// Split candidates for a photographic (RGB/RCT) source; palette index
/// images overfit above `MA_PALETTE_CANDIDATES`.
pub(super) const MA_PHOTO_CANDIDATES: usize = 32;
/// Split candidates for a palette index image: more than this overfits the
/// near-deterministic index.
pub(super) const MA_PALETTE_CANDIDATES: usize = 16;

/// The single global transform a learned-tree frame declares.
#[derive(Clone, Copy)]
enum MaTransform {
    /// RCT on the first three channels (`6` = YCoCg).
    Rct(u32),
    /// Global palette over `num_c` components: channel 0 is the palette
    /// meta-channel (`nb_colors` x `num_c`), channel 1 the index image.
    Palette { num_c: u32, nb_colors: u32 },
}

/// The modular channels of a frame plus their transform.
struct MaSource<'a> {
    channels: Vec<MaChannel<'a>>,
    transform: MaTransform,
}

/// Quant tables reserved in the decoder's modular stream numbering
/// (`kNumQuantTables`); AC group streams are numbered after them.
const MODULAR_NUM_QUANT_TABLES: usize = 17;

/// The decoder's stream id (MA property 1) of AC group `group_index`:
/// `ModularStreamId::ModularAC(group, pass 0)` = 1 + 3 * DC groups + quant
/// tables + group. The global stream is id 0.
fn ac_stream_id(num_dc_groups: usize, group_index: usize) -> i32 {
    (1 + 3 * num_dc_groups + MODULAR_NUM_QUANT_TABLES + group_index) as i32
}

/// Where a channel is coded under a layout: in the single section / the
/// DC-global stream, or per AC group. Channel ids restart per stream, as the
/// decoder's channel property does.
struct MaPlacement {
    channel: usize,
    chan_id: u32,
    global: bool,
}

impl MaSource<'_> {
    /// Split candidates the learner should evaluate for this source.
    fn max_candidates(&self) -> usize {
        match self.transform {
            MaTransform::Palette { .. } => MA_PALETTE_CANDIDATES,
            MaTransform::Rct(_) => MA_PHOTO_CANDIDATES,
        }
    }

    fn write_transforms(&self, w: &mut BitWriter) {
        match self.transform {
            MaTransform::Rct(rct_type) => {
                write_modular_transforms(self.channels.len(), rct_type, w)
            }
            MaTransform::Palette { num_c, nb_colors } => {
                write_palette_transform(num_c, nb_colors, w)
            }
        }
    }

    /// Channels no larger than a group are coded in the global stream (the
    /// decoder stops the global decode at the first bigger channel); in a
    /// single-group frame everything shares one stream.
    fn placement(&self, layout: GroupLayout, single_group: bool) -> Vec<MaPlacement> {
        let gdim = layout.dim();
        let mut global_id = 0u32;
        let mut group_id = 0u32;
        self.channels
            .iter()
            .enumerate()
            .map(|(channel, ch)| {
                let global = single_group || ch.meta || (ch.w <= gdim && ch.h <= gdim);
                let chan_id = if single_group {
                    channel as u32
                } else if global {
                    let id = global_id;
                    global_id += 1;
                    id
                } else {
                    let id = group_id;
                    group_id += 1;
                    id
                };
                MaPlacement {
                    channel,
                    chan_id,
                    global,
                }
            })
            .collect()
    }

    fn total_values(&self) -> usize {
        self.channels.iter().map(|c| c.w * c.h).sum()
    }

    /// The reference planes of placement `p`'s channel: earlier channels of
    /// the same stream and size, most recent first, cropped at (x0, y0).
    fn ref_planes(
        &self,
        placement: &[MaPlacement],
        p: &MaPlacement,
        x0: usize,
        y0: usize,
    ) -> Vec<MaRefPlane<'_>> {
        let ch = &self.channels[p.channel];
        placement[..p.channel]
            .iter()
            .rev()
            .filter(|q| q.global == p.global)
            .filter_map(|q| {
                let other = &self.channels[q.channel];
                match other.pixels {
                    MaPixels::I32(pixels) if other.w == ch.w && other.h == ch.h => {
                        Some(MaRefPlane {
                            pixels,
                            stride: other.w,
                            x0,
                            y0,
                        })
                    }
                    _ => None,
                }
            })
            .take(MA_REF_CHANNELS)
            .collect()
    }
}

/// The YCoCg(A) planes as a learned-tree source.
fn rgb_ma_source<'a>(
    linear: &'a Image3Si,
    alpha: Option<&'a AlphaPlane>,
    xsize: usize,
    rct_type: u32,
) -> MaSource<'a> {
    let ysize = linear.ysize();
    let mut channels: Vec<MaChannel<'a>> = (0..3)
        .map(|c| {
            let pd = linear.plane_data(c);
            MaChannel {
                w: xsize,
                h: ysize,
                meta: false,
                pixels: MaPixels::I32(pd),
            }
        })
        .collect();
    if let Some(a) = alpha {
        channels.push(MaChannel {
            w: xsize,
            h: ysize,
            meta: false,
            pixels: match a {
                AlphaPlane::U8(pixels) => MaPixels::U8(pixels),
                AlphaPlane::U16 { data, .. } => MaPixels::U16(data),
                AlphaPlane::F32(pixels) => MaPixels::I32(pixels),
            },
        });
    }
    MaSource {
        channels,
        transform: MaTransform::Rct(rct_type),
    }
}

/// Below this covered-area fraction the patched alternative is not encoded.
const PATCH_MIN_COVERAGE: f64 = 0.05;

/// Sample every stream of `source` under `layout` (group-local coordinates
/// and a fresh WP state per group, exactly as the decoder will see them) and
/// run the first learning stage. `None` when no tree is estimated to beat
/// the flat path.
#[allow(clippy::too_many_arguments)]
fn learn_ma_candidate(
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
    min_symbol: u32,
    wp_params: WpParams,
    use_wp: bool,
    max_leaves: usize,
    full_sample_limit: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<CoarseLearn> {
    let gdim = layout.dim();
    let xsize_groups = xsize.div_ceil(gdim);
    let num_groups = xsize_groups * ysize.div_ceil(gdim);
    let placement = source.placement(layout, num_groups == 1);
    let max_candidates = source.max_candidates();
    let stride = ma_sample_stride(source.total_values(), full_sample_limit);

    // Sampling jobs: global streams first, then group-major / channel-minor.
    struct Job<'a> {
        channel: usize,
        chan_id: u32,
        stream: i32,
        refs: Vec<MaRefPlane<'a>>,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
    }
    let num_dc_groups = xsize.div_ceil(layout.lf_dim()) * ysize.div_ceil(layout.lf_dim());
    let mut jobs: Vec<Job> = Vec::new();
    for p in placement.iter().filter(|p| p.global) {
        let ch = &source.channels[p.channel];
        jobs.push(Job {
            channel: p.channel,
            chan_id: p.chan_id,
            stream: 0,
            refs: source.ref_planes(&placement, p, 0, 0),
            x0: 0,
            y0: 0,
            w: ch.w,
            h: ch.h,
        });
    }
    if num_groups > 1 {
        for group_index in 0..num_groups {
            let x0 = (group_index % xsize_groups) * gdim;
            let y0 = (group_index / xsize_groups) * gdim;
            for p in placement.iter().filter(|p| !p.global) {
                let ch = &source.channels[p.channel];
                jobs.push(Job {
                    channel: p.channel,
                    chan_id: p.chan_id,
                    stream: ac_stream_id(num_dc_groups, group_index),
                    refs: source.ref_planes(&placement, p, x0, y0),
                    x0,
                    y0,
                    w: gdim.min(ch.w - x0),
                    h: gdim.min(ch.h - y0),
                });
            }
        }
    }

    // Each job owns a disjoint range in the final sample arrays. The rotating
    // row phase gives this exact count, including narrow and partial groups.
    let counts: Vec<usize> = jobs
        .iter()
        .map(|job| ma_channel_sample_count(job.w, job.h, stride))
        .collect();
    let total_samples = counts.iter().sum();
    let mut samples = MaSamples {
        props: vec![[0; NUM_MA_PROPS]; total_samples],
        tok: vec![[0; NUM_MA_PREDS]; total_samples],
    };
    let mut props = samples.props.as_mut_slice();
    let mut tok = samples.tok.as_mut_slice();
    let mut ranges = Vec::with_capacity(jobs.len());
    for count in counts {
        let (p, rest_p) = props.split_at_mut(count);
        let (t, rest_t) = tok.split_at_mut(count);
        ranges.push((p, t));
        (props, tok) = (rest_p, rest_t);
    }
    pool.steal_for_each_mut(scratch, &mut ranges, |i, range, _scratch| {
        let job = &jobs[i];
        let ch = &source.channels[job.channel];
        let mut rows = range.0.iter_mut().zip(range.1.iter_mut());
        with_ma_pixels!(ch, |pixels| sample_channel_ma(
            |y| &pixels[(job.y0 + y) * ch.w + job.x0..][..job.w],
            job.w,
            job.h,
            job.chan_id,
            job.stream,
            &job.refs,
            wp_params,
            use_wp,
            stride,
            |p, t| {
                let (props, tok) = rows.next().expect("sample count");
                *props = p;
                *tok = t;
            },
        ));
        debug_assert!(rows.next().is_none());
    });
    learn_ma_coarse(
        samples,
        stride,
        min_symbol,
        use_wp,
        max_leaves,
        max_candidates,
        matches!(source.transform, MaTransform::Rct(_)),
        pool,
        scratch,
    )
}

/// Leaf budget of the ranking learner that scores WP presets and the RCT
/// runner-up: 96 leaves rank exactly like 128 on Kodak.
const MA_RANK_LEAVES: usize = 96;
/// The estimator's RCT runner-up is scored by the ranking learner when its
/// estimated cost is within this factor of the best.
const RCT_RANK_MARGIN: f32 = 1.02;

/// Sampling budget of the ranking learners. It must match the final learn's:
/// the best preset for a stride-3 learner is not the best for a stride-1
/// learner.
const MA_RANK_SAMPLE_LIMIT: usize = MA_FULL_SAMPLE_LIMIT;
/// A ranking candidate whose 32-leaf coarse estimate trails the best coarse
/// estimate seen so far by more than this is not deepened.
const RANK_COARSE_PRUNE: f64 = 1.005;

/// First ranking stage of `source`: its coarse learn under the 1024-px
/// layout at the ranking sample budget.
#[allow(clippy::too_many_arguments)]
fn rank_coarse(
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    wp_params: WpParams,
    use_wp: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<CoarseLearn> {
    learn_ma_candidate(
        source,
        xsize,
        ysize,
        GroupLayout::LARGE,
        min_symbol,
        wp_params,
        use_wp,
        MA_RANK_LEAVES,
        MA_RANK_SAMPLE_LIMIT,
        pool,
        scratch,
    )
}

/// Second ranking stage: deepen to the ranking leaf budget and return the
/// estimated bits (the flat alternatives when the tree does not qualify)
/// with to learn itself, so the winner can seed the final learn.
fn rank_finish(
    stage: CoarseLearn,
    min_symbol: u32,
    use_wp: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> (f64, RankedLearn) {
    let coarse_est = stage.candidate.est_real;
    let flat = stage.candidate.flat_real;
    let stride = stage.stride;
    let max_candidates = stage.max_candidates;
    let leaf_offsets = stage.leaf_offsets;
    let (candidate, samples) = if stage.deepen {
        let (tree, samples) = deepen_ma_tree(
            stage.samples,
            ma_learn_params(
                min_symbol,
                MA_RANK_LEAVES,
                max_candidates,
                stride as f64,
                use_wp,
            ),
            stage.candidate.tree.clone(),
            pool,
            scratch,
        );
        (
            gate_ma_tree(tree, stride as f64, None).unwrap_or(stage.candidate),
            samples,
        )
    } else {
        (stage.candidate, stage.samples)
    };
    let bits = candidate.est_real.min(flat);
    (
        bits,
        RankedLearn {
            candidate,
            samples,
            stride,
            max_candidates,
            leaf_offsets,
            coarse_est,
        },
    )
}

/// First learning stage under every layout, second stage for the preferred
/// (first) layout and for any later layout whose first stage already
/// estimates smaller; the full estimates pick the winner. Layouts are
/// processed one at a time so a single sample set is alive.
#[allow(clippy::too_many_arguments)]
fn learn_best_layout(
    source: &MaSource<'_>,
    xsize: usize,
    ysize: usize,
    layouts: &[GroupLayout],
    min_symbol: u32,
    wp_params: WpParams,
    use_wp: bool,
    coarse_limit: f64,
    deepen_all: bool,
    seed: Option<RankedLearn>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<(GroupLayout, LearnedCandidate, f64)> {
    let mut best_coarse_est = f64::INFINITY;
    let mut learned: Option<(GroupLayout, LearnedCandidate)> = None;
    let mut seed = seed;
    for (index, &layout) in layouts.iter().enumerate() {
        // The ranking winner already sampled and partly grew the first layout.
        if index == 0
            && let Some(ranked) = seed.take()
        {
            best_coarse_est = ranked.coarse_est;
            let cand = ranked.finish(min_symbol, use_wp, pool, scratch);
            learned = Some((layout, cand));
            continue;
        }
        let Some(stage) = learn_ma_candidate(
            source,
            xsize,
            ysize,
            layout,
            min_symbol,
            wp_params,
            use_wp,
            MA_MAX_LEAVES,
            MA_FULL_SAMPLE_LIMIT,
            pool,
            scratch,
        ) else {
            continue;
        };
        let coarse_est = stage.candidate.est_real;
        // A source whose first stage already trails `coarse_limit` is not
        // deepened (the palette candidate against the RGB one). Between this
        // source's own layouts the coarse estimate mis-ranks palette index
        // images (it prefers 1024 where the accurate estimate prefers 256, a
        // ~4% loss), so `deepen_all` sources deepen every layout and pick by
        // the final estimate; photos keep the cheap coarse prune.
        let layout_pruned = index > 0 && !deepen_all && coarse_est > best_coarse_est;
        if coarse_est > coarse_limit || layout_pruned {
            best_coarse_est = best_coarse_est.min(coarse_est);
            continue;
        }
        best_coarse_est = best_coarse_est.min(coarse_est);
        let cand = finish_ma_learn(stage, min_symbol, use_wp, MA_MAX_LEAVES, pool, scratch);
        if learned
            .as_ref()
            .is_none_or(|(_, best)| cand.est_real < best.est_real)
        {
            learned = Some((layout, cand));
        }
    }
    learned.map(|(layout, cand)| (layout, cand, best_coarse_est))
}

// Compile both representations once; use the compact path only when every
// stream fits. The wide fallback retains the same stream order and values.
macro_rules! with_raw_streams {
    ($raw:expr, |$streams:ident| $body:block) => {
        match RawTokenStreams::new($raw) {
            RawTokenStreams::Compact($streams) => $body,
            RawTokenStreams::Wide($streams) => $body,
        }
    };
}

/// Write the complete learned-tree frame under `layout`: a local tree in the
/// single section when the frame is one group, otherwise a global tree in the
/// DC-global section (followed by any global-stream channels) with every AC
/// group routed through it (fresh WP state and group-local coordinates per
/// group, matching the decoder). Returns the candidate's estimated fractional
/// saving over the flat path.
#[allow(clippy::too_many_arguments)]
fn write_learned_tree_frame(
    source: &MaSource<'_>,
    has_alpha: bool,
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    use_wp: bool,
    cand: &LearnedCandidate,
    writer: &mut BitWriter,
) -> f64 {
    let gdim = layout.dim();
    let xsize_groups = xsize.div_ceil(gdim);
    let ysize_groups = ysize.div_ceil(gdim);
    let num_ac_groups = xsize_groups * ysize_groups;
    let num_dc_groups = xsize.div_ceil(layout.lf_dim()) * ysize.div_ceil(layout.lf_dim());
    let single_group = num_ac_groups == 1;
    let placement = source.placement(layout, single_group);
    let tree = &cand.tree;
    let leaf_ctx = &cand.leaf_ctx;
    let num_ctx = cand.num_ctx;
    let distance_ctx = num_ctx;
    // Placement is in source-channel order; its channel ids already account
    // for the independent global and group streams (including palettes).
    // The compiled lookup folds the channel and stream ids, so it is built
    // per (channel, stream): cheap next to tokenizing the stream itself.
    let tokenize = |p: &MaPlacement,
                    stream: i32,
                    x0: usize,
                    y0: usize,
                    w: usize,
                    h: usize,
                    out: &mut RawTokens| {
        let ch = &source.channels[p.channel];
        let lookup = MaLookup::new(tree, leaf_ctx, p.chan_id, stream);
        let refs = source.ref_planes(&placement, p, x0, y0);
        with_ma_pixels!(ch, |pixels| tokenize_channel_ma(
            |y| &pixels[(y0 + y) * ch.w + x0..][..w],
            w,
            h,
            p.chan_id,
            stream,
            &refs,
            wp_params,
            use_wp,
            &lookup,
            &cand.leaf_offset,
            out,
        ));
    };

    write_frame_header_modular(has_alpha, layout, writer);

    if single_group {
        let channel_tokens = pool.steal_map(scratch, placement.len(), |i, _scratch| {
            let p = &placement[i];
            let ch = &source.channels[p.channel];
            let mut tokens = RawTokens::with_capacity(ch.w * ch.h);
            tokenize(p, 0, 0, 0, ch.w, ch.h, &mut tokens);
            tokens
        });
        let mut section = with_raw_streams!(channel_tokens, |channel_tokens| {
            let mut tokens = Vec::with_capacity(source.total_values());
            for channel in channel_tokens {
                tokens.extend(channel);
            }

            let variants = choose_tree_streams(
                vec![tokens],
                distance_ctx,
                num_ctx as usize + 1,
                min_symbol,
                pool,
                scratch,
            );
            let mut best: Option<BitWriter> = None;
            let mut best_estimate = f64::INFINITY;
            for variant in variants {
                match variant {
                    TreeStreams::Literals(streams) => {
                        let stream = streams.into_iter().next().expect("one stream");
                        write_learned_single_variant(
                            &stream,
                            source,
                            cand,
                            min_symbol,
                            wp_params,
                            pool,
                            scratch,
                            &mut best,
                            &mut best_estimate,
                        );
                    }
                    TreeStreams::Lz(streams) => {
                        let stream = streams.into_iter().next().expect("one stream");
                        write_learned_single_variant(
                            &stream,
                            source,
                            cand,
                            min_symbol,
                            wp_params,
                            pool,
                            scratch,
                            &mut best,
                            &mut best_estimate,
                        );
                    }
                }
            }
            best.expect("learned-tree section")
        });

        writer.write(1, 0);
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append_byte_aligned(std::slice::from_mut(&mut section));
        writer.zero_pad_to_byte();
        return cand.estimated_savings;
    }

    // Global stream: channels that fit a group, coded in the DC-global section.
    let mut global_tokens = RawTokens::with_capacity(0);
    for p in placement.iter().filter(|p| p.global) {
        let ch = &source.channels[p.channel];
        tokenize(p, 0, 0, 0, ch.w, ch.h, &mut global_tokens);
    }
    let has_global_stream = !global_tokens.is_empty();

    let group_placement: Vec<&MaPlacement> = placement.iter().filter(|p| !p.global).collect();
    let group_tokens: Vec<RawTokens> = pool.steal_map_with_threads(
        scratch,
        num_ac_groups,
        group_lz_threads(crate::Speed::Slow, pool),
        |group_index, _scratch| {
            let x0 = (group_index % xsize_groups) * gdim;
            let y0 = (group_index / xsize_groups) * gdim;
            let num_values = group_placement
                .iter()
                .map(|p| {
                    let ch = &source.channels[p.channel];
                    gdim.min(ch.w - x0) * gdim.min(ch.h - y0)
                })
                .sum();
            let mut toks = RawTokens::with_capacity(num_values);
            for p in &group_placement {
                let ch = &source.channels[p.channel];
                let gw = gdim.min(ch.w - x0);
                let gh = gdim.min(ch.h - y0);
                tokenize(
                    p,
                    ac_stream_id(num_dc_groups, group_index),
                    x0,
                    y0,
                    gw,
                    gh,
                    &mut toks,
                );
            }
            toks
        },
    );
    // Stream 0 is the global stream (possibly empty), then one per AC group.
    let mut all_tokens: Vec<RawTokens> = Vec::with_capacity(1 + num_ac_groups);
    all_tokens.push(global_tokens);
    all_tokens.extend(group_tokens);
    let mut sections = with_raw_streams!(all_tokens, |all_tokens| {
        let variants = choose_tree_streams(
            all_tokens,
            distance_ctx,
            num_ctx as usize + 1,
            min_symbol,
            pool,
            scratch,
        );
        let mut best: Option<Vec<BitWriter>> = None;
        let mut best_bits = usize::MAX;
        let mut best_estimate = f64::INFINITY;
        for variant in variants {
            match variant {
                TreeStreams::Literals(streams) => {
                    write_learned_grouped_variant(
                        &streams,
                        source,
                        cand,
                        min_symbol,
                        wp_params,
                        pool,
                        scratch,
                        num_ac_groups,
                        num_dc_groups,
                        has_global_stream,
                        &mut best,
                        &mut best_bits,
                        &mut best_estimate,
                    );
                }
                TreeStreams::Lz(streams) => {
                    write_learned_grouped_variant(
                        &streams,
                        source,
                        cand,
                        min_symbol,
                        wp_params,
                        pool,
                        scratch,
                        num_ac_groups,
                        num_dc_groups,
                        has_global_stream,
                        &mut best,
                        &mut best_bits,
                        &mut best_estimate,
                    );
                }
            }
        }
        best.expect("learned-tree sections")
    });

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    cand.estimated_savings
}

#[allow(clippy::too_many_arguments)]
fn write_learned_single_variant<T: lz77::LzTokenSource>(
    stream: &[T],
    source: &MaSource<'_>,
    cand: &LearnedCandidate,
    min_symbol: u32,
    wp_params: WpParams,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    best: &mut Option<BitWriter>,
    best_estimate: &mut f64,
) {
    let num_ctx = cand.num_ctx;
    let distance_ctx = num_ctx;
    let code = build_lz_pixel_code_threads(
        std::iter::once(stream),
        num_ctx as usize,
        min_symbol,
        true,
        true,
        Some(pool),
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let estimate = estimate_coded_bits(
        &[stream],
        distance_ctx,
        &code,
        min_symbol,
        pool.num_threads(),
    );
    if estimate >= *best_estimate * VARIANT_WRITE_TOLERANCE {
        return;
    }
    *best_estimate = (*best_estimate).min(estimate);
    let mut body = BitWriter::new();
    body.write(1, 1); // dc_quant all_default = 1
    body.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    body.write(1, 0); // use_global_tree = 0
    write_wp_header(wp_params, &mut body);
    source.write_transforms(&mut body);
    write_tree_lz77(
        &cand.tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut body,
    );
    write_lz_section(stream, distance_ctx, &code, min_symbol, &mut body);
    body.zero_pad_to_byte();
    keep_smaller_writer(best, body);
}

#[allow(clippy::too_many_arguments)]
fn write_learned_grouped_variant<T: lz77::LzTokenSource>(
    streams: &[Vec<T>],
    source: &MaSource<'_>,
    cand: &LearnedCandidate,
    min_symbol: u32,
    wp_params: WpParams,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    num_ac_groups: usize,
    num_dc_groups: usize,
    has_global_stream: bool,
    best: &mut Option<Vec<BitWriter>>,
    best_bits: &mut usize,
    best_estimate: &mut f64,
) {
    let num_ctx = cand.num_ctx;
    let distance_ctx = num_ctx;
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let code = build_lz_pixel_code_threads(
        streams.iter().map(Vec::as_slice),
        num_ctx as usize,
        min_symbol,
        true,
        true,
        Some(pool),
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let estimate = {
        let slices: Vec<&[T]> = streams.iter().map(Vec::as_slice).collect();
        estimate_coded_bits(&slices, distance_ctx, &code, min_symbol, pool.num_threads())
    };
    if estimate >= *best_estimate * VARIANT_WRITE_TOLERANCE {
        return;
    }
    *best_estimate = (*best_estimate).min(estimate);
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(
        &cand.tree_tokens,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    write_wp_header(wp_params, &mut sections[0]);
    source.write_transforms(&mut sections[0]);
    if has_global_stream {
        write_lz_section(
            &streams[0],
            distance_ctx,
            &code,
            min_symbol,
            &mut sections[0],
        );
    }
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

    // Every AC group section is independent given the code (scoped
    // threads: the code borrows the worker scratch the pool would need).
    let code_ref = &code;
    let group_streams = &streams[1..];
    let threads = pool.num_threads().clamp(1, num_ac_groups.max(1));
    let chunk = num_ac_groups.div_ceil(threads).max(1);
    let group_sections: Vec<BitWriter> = std::thread::scope(|scope| {
        let handles: Vec<_> = group_streams
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|stream| {
                            let mut section = BitWriter::new();
                            section.write(1, 1);
                            write_wp_header(wp_params, &mut section);
                            section.write(2, 0);
                            write_lz_section(
                                stream,
                                distance_ctx,
                                code_ref,
                                min_symbol,
                                &mut section,
                            );
                            section.zero_pad_to_byte();
                            section
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("group section writer"))
            .collect()
    });
    for (group_index, section) in group_sections.into_iter().enumerate() {
        sections[2 + num_dc_groups + group_index] = section;
    }
    let bits: usize = sections.iter().map(BitWriter::bits_written).sum();
    if bits < *best_bits {
        *best_bits = bits;
        *best = Some(sections);
    }
}

enum TreeStreams<T> {
    /// Retain the original tokens without expanding them to LzToken records.
    Literals(Vec<Vec<T>>),
    Lz(Vec<Vec<LzToken>>),
}

/// Runs and matches collapse a token stream, but a learned tree routes flat
/// content into near-deterministic contexts where a literal costs almost
/// nothing under ANS while every run token still pays for its length and
/// distance (the Burning Ship fractal codes 24% smaller as literals) — yet
/// a palette index image can prefer long matches. Streams that barely
/// collapse keep the LZ77 layer outright; otherwise the order-0 estimates
/// order the literal and LZ77 variants (the deep matcher is skipped when
/// runs alone trail literals by `LZ_DEEP_MATCHER_MAX_RATIO`), and the
/// writer only codes a later variant when its code-based estimate beats
/// the one already written.
const LZ_LITERAL_ALTERNATIVE_MAX_RATIO: f64 = 0.98;
/// Order-0 estimates are optimistic for literal streams whose contexts get
/// merged by clustering (up to ~30% on palette indices), so the matcher is
/// only skipped on a wide margin.
const LZ_DEEP_MATCHER_MAX_RATIO: f64 = 1.6;
/// A later variant is written when its code-based estimate is within this
/// factor of the best written one (the estimate itself is ~0.5% accurate).
const VARIANT_WRITE_TOLERANCE: f64 = 1.005;

/// Stream variants to try for one learned-tree frame, most promising first:
/// literal variants reuse the input, and LZ variants own their match streams.
fn choose_tree_streams<T: lz77::LiteralToken>(
    tokens: Vec<Vec<T>>,
    distance_ctx: u32,
    num_contexts: usize,
    min_symbol: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<TreeStreams<T>> {
    let raw: usize = tokens.iter().map(Vec::len).sum();
    // Keep the per-stream counts from the existing gate pass so the run
    // candidates can allocate exactly once, without retaining growth slack.
    let run_counts = pool.steal_map(scratch, tokens.len(), |k, _scratch| {
        lz77_run_count(&tokens[k])
    });
    let run_len: usize = run_counts.iter().sum();
    let deep = |scratch: &mut CoderScratch| -> Vec<Vec<LzToken>> {
        let deep_lz = DeepLzScratchPool::new(group_lz_threads(crate::Speed::Slow, pool));
        pool.steal_map_with_threads(
            scratch,
            tokens.len(),
            group_lz_threads(crate::Speed::Slow, pool),
            |k, scratch| {
                deep_lz.with_depth(|depth| {
                    lz77_compress_for_speed_with_depth(
                        &tokens[k],
                        distance_ctx,
                        crate::Speed::Slow,
                        depth,
                        Some(run_counts[k]),
                        scratch,
                    )
                })
            },
        )
    };
    if (run_len as f64) >= LZ_LITERAL_ALTERNATIVE_MAX_RATIO * raw as f64 {
        return vec![TreeStreams::Lz(deep(scratch))];
    }
    let raw_slices: Vec<&[T]> = tokens.iter().map(Vec::as_slice).collect();
    let (e_lit, e_run) =
        estimate_literal_and_run_bits(&raw_slices, num_contexts, min_symbol, pool, scratch);
    if e_run > e_lit * LZ_DEEP_MATCHER_MAX_RATIO {
        return vec![TreeStreams::Literals(tokens)];
    }
    let lz = deep(scratch);
    let slices: Vec<&[LzToken]> = lz.iter().map(Vec::as_slice).collect();
    let e_lz = estimate_streams_bits(&slices, num_contexts, min_symbol, pool, scratch);
    if e_lz < e_lit {
        vec![TreeStreams::Lz(lz), TreeStreams::Literals(tokens)]
    } else {
        vec![TreeStreams::Literals(tokens), TreeStreams::Lz(lz)]
    }
}

/// v1 context tree: single-group only. Returns true (and writes the full frame)
/// if the context tree is estimated to help; false otherwise (caller falls
/// through to the flat path, having written nothing).
fn try_encode_context_tree_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    layout: GroupLayout,
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
    write_frame_header_modular(alpha.is_some(), layout, writer);
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
    layout: GroupLayout,
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
            let x0 = gx * layout.dim();
            let y0 = gy * layout.dim();
            let gw = layout.dim().min(xsize - x0);
            let gh = layout.dim().min(ysize - y0);
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
    let group_lz_tokens: Vec<Vec<LzToken>> = {
        let deep_lz = DeepLzScratchPool::new(group_lz_threads(crate::Speed::Slow, pool));
        pool.steal_map_with_threads(
            scratch,
            num_ac_groups,
            group_lz_threads(crate::Speed::Slow, pool),
            |group_index, scratch| {
                let g = &groups[group_index];
                let token_count = g.iter().map(|(res, _)| res.len()).sum();
                let mut toks: Vec<Token> = Vec::with_capacity(token_count);
                for (((res, prp), &threshold), contexts) in g[..nb_chans]
                    .iter()
                    .zip(&ts[..nb_chans])
                    .zip(ctx_lut.as_chunks::<3>().0)
                {
                    let t = threshold as i64;
                    for (&prp, &res) in prp[..res.len()].iter().zip(res.iter()) {
                        let bucket = bucket_of(prp, t);
                        let ctx = contexts[bucket as usize];
                        toks.push(Token::new(ctx, res));
                    }
                }
                deep_lz.with_depth(|depth| {
                    lz77_compress_for_speed_with_depth(
                        &toks,
                        distance_ctx,
                        crate::Speed::Slow,
                        depth,
                        None,
                        scratch,
                    )
                })
            },
        )
    };
    let code = build_lz_pixel_code(
        group_lz_tokens.iter().map(Vec::as_slice),
        num_pixel_ctx,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    // 5) Sections (same layout as the flat multi-group path).
    write_frame_header_modular(alpha.is_some(), layout, writer);
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

    write_lz_groups(
        &group_lz_tokens,
        &code,
        distance_ctx,
        min_symbol,
        wp_params,
        pool,
        &mut sections[2 + num_dc_groups..],
    );

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
    push_leaf_mul(out, predictor, 1);
}

/// Leaf with a residual multiplier: the decoder reconstructs
/// `pred + multiplier * residual` (multiplier = (mul_bits+1) << mul_log).
fn push_leaf_mul(out: &mut Vec<Token>, predictor: u32, multiplier: u32) {
    debug_assert!(multiplier >= 1);
    let mul_log = multiplier.trailing_zeros();
    let mul_bits = (multiplier >> mul_log) - 1;
    out.push(Token::new(TREE_CTX_PROPERTY, 0));
    out.push(Token::new(TREE_CTX_PREDICTOR, predictor));
    out.push(Token::new(TREE_CTX_OFFSET, pack_signed(0)));
    out.push(Token::new(TREE_CTX_MULTIPLIER_LOG, mul_log));
    out.push(Token::new(TREE_CTX_MULTIPLIER_BITS, mul_bits));
}

/// Balanced channel-split tree whose leaves carry per-channel residual
/// multipliers (the lossy-modular quantization steps).
fn build_balanced_tree_tokens_mul(leaves: &[(u32, u32)]) -> Vec<Token> {
    let n_leaves = leaves.len();
    let mut t = Vec::new();
    match n_leaves {
        1 => push_leaf_mul(&mut t, leaves[0].0, leaves[0].1),
        2 => {
            push_split(&mut t, 0, 0);
            push_leaf_mul(&mut t, leaves[1].0, leaves[1].1);
            push_leaf_mul(&mut t, leaves[0].0, leaves[0].1);
        }
        _ => {
            // Right-leaning chain, BFS leaves chan(N-1)..chan0 (same shape as
            // `build_balanced_tree_tokens`).
            for k in (1..=(n_leaves - 2)).rev() {
                push_split(&mut t, 0, k as i32);
                push_leaf_mul(&mut t, leaves[k + 1].0, leaves[k + 1].1);
            }
            push_split(&mut t, 0, 0);
            push_leaf_mul(&mut t, leaves[1].0, leaves[1].1);
            push_leaf_mul(&mut t, leaves[0].0, leaves[0].1);
        }
    }
    t
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

    write_frame_header_modular(alpha.is_some(), GroupLayout::DEFAULT, writer);

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

        for section in &mut sections[1..1 + num_dc_groups] {
            section.write(1, 1);
            section.write(1, 1);
            section.write(2, 0);
            section.zero_pad_to_byte();
        }

        let ac_global_idx = 1 + num_dc_groups;
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].zero_pad_to_byte();

        for (section, tokens) in sections[2 + num_dc_groups..][..num_ac_groups]
            .iter_mut()
            .zip(&group_tokens[..num_ac_groups])
        {
            section.write(1, 1);
            section.write(1, 1);
            section.write(2, 0);
            for t in tokens {
                write_token(*t, &code.as_ref(), section);
            }
            section.zero_pad_to_byte();
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
    use super::palette::local_palette_coverage_is_sufficient;
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
            crate::DecodingSpeed::Slow,
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

    #[test]
    fn row_walk_matches_point_fetched_properties_and_predictions() {
        for width in [0usize, 1, 2, 3, 4, 5, 17, 257] {
            for height in [0usize, 1, 2, 7] {
                for flat in [false, true] {
                    let plane: Vec<i32> = (0..width * height)
                        .map(|i| {
                            if flat {
                                8192
                            } else {
                                [i32::MIN, i32::MAX, -65536, 65535, -8192, 0, 8192]
                                    [(i * 17 + i / width) % 7]
                            }
                        })
                        .collect();
                    let get = |x: usize, y: usize| plane[y * width + x];
                    for &params in &WpParams::PRESETS {
                        for use_wp in [false, true] {
                            let mut wp = WpState::with_params(width, params);
                            let mut previous_gradient = 0i32;
                            let mut count = 0;
                            walk_channel_ma(
                                |y| &plane[y * width..][..width],
                                width,
                                height,
                                2,
                                0,
                                &[],
                                params,
                                use_wp,
                                |x, y, value, props, neighbors, weighted| {
                                    assert_eq!(count, y * width + x);
                                    assert_eq!(value, get(x, y) as i64);
                                    count += 1;
                                    if x == 0 {
                                        previous_gradient = 0;
                                    }
                                    let n = predictor_neighbors(&get, x, y, width);
                                    let expected_wp = if use_wp {
                                        wp.predict(
                                            x,
                                            y,
                                            n.top,
                                            n.left,
                                            n.top_right,
                                            n.top_left,
                                            n.top_top,
                                        )
                                    } else {
                                        0
                                    };
                                    let gradient = (n.left + n.top - n.top_left) as i32;
                                    // No reference planes: the extra properties stay 0.
                                    assert!(props[16..].iter().all(|&v| v == 0));
                                    assert_eq!(
                                        props[..16],
                                        [
                                            2,
                                            0,
                                            y as i32,
                                            x as i32,
                                            n.top.abs() as i32,
                                            n.left.abs() as i32,
                                            n.top as i32,
                                            n.left as i32,
                                            (n.left - previous_gradient as i64) as i32,
                                            gradient,
                                            (n.left - n.top_left) as i32,
                                            (n.top_left - n.top) as i32,
                                            (n.top - n.top_right) as i32,
                                            (n.top - n.top_top) as i32,
                                            (n.left - n.left_left) as i32,
                                            if use_wp { wp.wp_prop as i32 } else { 0 },
                                        ]
                                    );
                                    assert_eq!(weighted, expected_wp);
                                    for pred in 0..NUM_MA_PREDS as u32 {
                                        assert_eq!(
                                            predictor_value(pred, neighbors, weighted),
                                            predictor_value(pred, n, expected_wp),
                                        );
                                    }
                                    previous_gradient = gradient;
                                    if use_wp {
                                        wp.update(value, x, y);
                                    }
                                },
                            );
                            assert_eq!(count, width * height);
                        }
                    }
                }
            }
        }
    }

    fn reference_samples(
        plane: &[i32],
        width: usize,
        height: usize,
        stride: usize,
        params: WpParams,
    ) -> MaSamples {
        let mut samples = MaSamples::new();
        walk_channel_ma(
            |y| &plane[y * width..][..width],
            width,
            height,
            2,
            0,
            &[],
            params,
            true,
            |x, y, value, p, n, wp_pred| {
                // Row-rotated phase: row y samples x == y (mod stride).
                if x % stride != y % stride {
                    return;
                }
                let mut tok = [0u8; NUM_MA_PREDS];
                for pred in 0..NUM_MA_PREDS as u32 {
                    let prediction = predictor_value(pred, n, wp_pred);
                    let (token, _, _) = uint_encode(pack_signed((value - prediction) as i32));
                    tok[pred as usize] = token.min(u8::MAX as u32) as u8;
                }
                samples.push(*p, tok);
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
            let get_row = |y: usize| &plane[y * width..][..width];
            for &stride in &[1, 2, 3, 5, 11, width * height + 1] {
                for &params in &WpParams::PRESETS {
                    let expected = reference_samples(&plane, width, height, stride, params);
                    let mut actual = MaSamples::new();
                    sample_channel_ma(
                        get_row,
                        width,
                        height,
                        2,
                        0,
                        &[],
                        params,
                        true,
                        stride,
                        |p, t| actual.push(p, t),
                    );
                    assert_eq!(actual.len(), ma_channel_sample_count(width, height, stride));
                    assert_eq!(
                        actual.props, expected.props,
                        "properties: {width}x{height}, stride={stride}, params={params:?}"
                    );
                    assert_eq!(
                        actual.tok, expected.tok,
                        "tokens: {width}x{height}, stride={stride}, params={params:?}"
                    );
                }
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
#[cfg(test)]
mod context_tree_tests {
    use super::*;

    #[test]
    fn gated_tree_keeps_bfs_tokens_with_leaf_offsets() {
        let tree = LearnedTree {
            nodes: vec![
                MaNode::Split {
                    prop: 0,
                    val: 0,
                    gt: 1,
                    le: 2,
                },
                MaNode::Leaf { pred: 1 },
                MaNode::Split {
                    prop: 3,
                    val: 3,
                    gt: 3,
                    le: 4,
                },
                MaNode::Leaf { pred: 6 },
                MaNode::Leaf { pred: 0 },
            ],
            est_bits: 100.,
            flat_bits: 10000.,
        };
        let mut samples = MaSamples::new();
        for (channel, x, residual) in [(1, 0, -3), (0, 4, 5), (0, 3, 0)] {
            let mut props = [0; NUM_MA_PROPS];
            props[0] = channel;
            props[3] = x;
            for _ in 0..64 {
                samples.push(props, [pack_signed(residual) as u8; NUM_MA_PREDS]);
            }
        }
        for with_offsets in [false, true] {
            let candidate =
                gate_ma_tree(tree.clone(), 1., with_offsets.then_some(&samples)).unwrap();
            let offsets = if with_offsets { [-3, 5, 0] } else { [0; 3] };
            assert_eq!(candidate.leaf_offset, offsets);
            assert_eq!(candidate.leaf_ctx, [u32::MAX, 0, u32::MAX, 1, 2]);
            assert_eq!(candidate.num_ctx, 3);
            let leaf = |pred, offset| {
                [
                    (TREE_CTX_PROPERTY, 0),
                    (TREE_CTX_PREDICTOR, pred),
                    (TREE_CTX_OFFSET, pack_signed(offset)),
                    (TREE_CTX_MULTIPLIER_LOG, 0),
                    (TREE_CTX_MULTIPLIER_BITS, 0),
                ]
            };
            let mut expected = vec![(TREE_CTX_PROPERTY, 1), (TREE_CTX_SPLIT_VAL, 0)];
            expected.extend(leaf(1, offsets[0]));
            expected.extend([(TREE_CTX_PROPERTY, 4), (TREE_CTX_SPLIT_VAL, 6)]);
            expected.extend(leaf(6, offsets[1]));
            expected.extend(leaf(0, offsets[2]));
            let actual: Vec<_> = candidate
                .tree_tokens
                .iter()
                .map(|token| (token.context, token.value))
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(candidate.est_real, 890.);
        }
        // Admission uses only the unchanged token-count estimate.
        for est_bits in [10000., f64::NAN] {
            assert!(
                gate_ma_tree(
                    LearnedTree {
                        est_bits,
                        ..tree.clone()
                    },
                    1.,
                    Some(&samples)
                )
                .is_none()
            );
        }
    }

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
    fn compiled_ma_tokenization_matches_full_tree_with_and_without_wp() {
        let tree = LearnedTree {
            nodes: vec![
                MaNode::Split {
                    prop: 0,
                    val: 0,
                    gt: 1,
                    le: 4,
                },
                MaNode::Split {
                    prop: 15,
                    val: 0,
                    gt: 2,
                    le: 3,
                },
                MaNode::Leaf { pred: 6 },
                MaNode::Leaf { pred: 1 },
                MaNode::Split {
                    prop: 3,
                    val: 3,
                    gt: 5,
                    le: 6,
                },
                MaNode::Leaf { pred: 5 },
                MaNode::Leaf { pred: 1 },
            ],
            est_bits: 0.,
            flat_bits: 0.,
        };
        let (_, contexts, _) = emit_learned_tree(&tree);
        for channel in [0, 1, 7, u32::MAX] {
            let lookup = MaLookup::new(&tree, &contexts, channel, 0);
            assert_eq!(lookup.needs_wp(), (channel as i32) > 0);
            for width in [0, 1, 2, 3, 4, 5, 9, 257] {
                for height in [0, 1, 2, 3, 17] {
                    let values: Vec<i32> = (0..width * height)
                        .map(|i| ((i * 7919 ^ (i / 7 * 1237)) % 65536) as i32 - 32768)
                        .collect();
                    let get_row = |y| &values[y * width..][..width];
                    for use_wp in [false, true] {
                        let mut expected = Vec::new();
                        walk_channel_ma(
                            get_row,
                            width,
                            height,
                            channel,
                            0,
                            &[],
                            WpParams::DEFAULT,
                            use_wp,
                            |_x, _y, value, props, n, wp| {
                                let (node, pred) = tree.lookup(props);
                                expected.push((
                                    contexts[node as usize],
                                    pack_signed((value - predictor_value(pred, n, wp)) as i32),
                                ));
                            },
                        );
                        let mut actual = RawTokens::with_capacity(values.len());
                        tokenize_channel_ma(
                            get_row,
                            width,
                            height,
                            channel,
                            0,
                            &[],
                            WpParams::DEFAULT,
                            use_wp,
                            &lookup,
                            &vec![0; contexts.len()],
                            &mut actual,
                        );
                        let actual: Vec<_> = match RawTokenStreams::new(vec![actual]) {
                            RawTokenStreams::Compact(streams) => streams
                                .into_iter()
                                .flatten()
                                .map(|t| {
                                    let t = t.unpack();
                                    (t.context, t.value)
                                })
                                .collect(),
                            RawTokenStreams::Wide(streams) => streams
                                .into_iter()
                                .flatten()
                                .map(|t| (t.context, t.value))
                                .collect(),
                        };
                        assert_eq!(
                            actual, expected,
                            "channel={channel}, width={width}, height={height}, wp={use_wp}"
                        );
                    }
                }
            }
        }
    }
}
