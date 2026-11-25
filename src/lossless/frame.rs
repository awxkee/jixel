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

//! Which frames code a lossless image.
//!
//! 1. Region split: a screenshot with gradient or photo-like rectangles (see
//!    `region_frames`) may code as a base saved to reference slot 0 plus one
//!    cropped kReplace frame per rectangle. When the first-stage estimates
//!    favor the split, both it and the whole frame are encoded and the
//!    smaller is kept. Otherwise a second plan that also takes in flat
//!    illustrations is ranked by complete `Fast` encodes instead.
//! 2. Displayed frame: the image itself or, with patches, a reference-only
//!    atlas plus the patched image; both compete on complete size.
//!
//! Every frame is coded by [`encode_frame_lossless_core`].

use super::encoder::encode_frame_lossless_core;
use super::header::write_patch_dictionary;
use super::ma_learning::rank_coarse;
use super::ma_source::{MA_FULL_SAMPLING, PATCH_MIN_COVERAGE, rgb_ma_source};
use super::{GroupLayout, LosslessSettings, constant_alpha_value, lossless_min_symbol};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::patches::{
    LosslessPatches, ModularFrameKind, find_lossless_additive_glyph_patches,
    find_lossless_glyph_patches, find_lossless_patches,
};
use crate::region_frames::Region;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// Header, TOC and tree of one region frame.
const REGION_FRAME_OVERHEAD_BITS: f64 = 1600.0;

/// A rectangle stays in the base when the whole frame's glyph patches save
/// this share of its own-frame estimate (a nested screenshot 15%, a chat
/// header block 19%; an image thumbnail 2%, art and avatars 0).
const REGION_GLYPH_SAVING: f64 = 0.1;

/// The region split is encoded when its first-stage estimate is under this
/// share of the whole frame's: estimated gains of 1.8% and more won 2-20%,
/// one of 0.6% lost 3.7%.
const REGION_SPLIT_MARGIN: f64 = 0.99;

/// The second region plan (with flat illustrations) is ranked by complete
/// encodes at this speed: the first-stage estimate cannot price small crops
/// (no tree beats the flat coder) and missed every illustration split that
/// won (−1.7..−7.3 % on 5 of 64 screenshots), while `Fast` ranked them all
/// right at 0.02-0.8 s. It is only trusted to say yes: its no lost known
/// wins of the first plan, which is why that plan keeps the estimate.
const REGION_PROXY_SPEED: crate::Speed = crate::Speed::Fast;

/// The proxy's split must code under this share of its whole frame: a
/// 0.35 % proxy margin bought a 0.37 % win at twice the encode time.
const REGION_PROXY_MARGIN: f64 = 0.995;

/// A split whose first-stage estimate is at most this share of the whole
/// frame's, or whose proxy codes at most [`REGION_SURE_PROXY`] of its whole
/// frame, is written without the whole frame's complete encode. Of 24
/// raced splits the 9 this covers all won, by 2.7 to 29 %, and the whole
/// frame was 30 to 55 % of their encode; the 5 that lost had estimates of
/// 0.95 and more and proxies of 0.98 and more.
const REGION_SURE_ESTIMATE: f64 = 0.93;
const REGION_SURE_PROXY: f64 = 0.96;

/// A glyph plan's regular frame is encoded only when its first-stage
/// estimate is within this factor of the patched frame's (base estimate plus
/// the atlas frame and dictionary): the coarse ratio between the two frames
/// tracks the final one within a few percent (1.25 coarse vs 1.19 final on a
/// UI screenshot), and a clear loser's complete encode was a third of the
/// total time.
const GLYPH_REGULAR_COARSE_MARGIN: f64 = 1.10;

/// A patched pair that codes below this share of its own first-stage
/// estimate shows the estimate missing what the complete learn finds, and
/// the regular frame's estimate is then no closer: the regular frame is
/// encoded up to [`GLYPH_REGULAR_UNSURE_MARGIN`] instead.
const GLYPH_UNSURE_CODED_SHARE: f64 = 0.55;
const GLYPH_REGULAR_UNSURE_MARGIN: f64 = 1.25;

/// A glyph plan covering this share of the image races the regular frame
/// whatever the estimates say.
const GLYPH_REGULAR_COVERAGE: f64 = 0.6;

pub(super) fn keep_smaller_writer(best: &mut Option<BitWriter>, candidate: BitWriter) {
    if best
        .as_ref()
        .is_none_or(|current| candidate.bits_written() < current.bits_written())
    {
        *best = Some(candidate);
    }
}

/// Appends the smaller of two complete encodes; `first` on a tie.
fn append_smaller(first: &BitWriter, second: &BitWriter, writer: &mut BitWriter) {
    if second.bits_written() < first.bits_written() {
        writer.append(second);
    } else {
        writer.append(first);
    }
}

/// The lossless image's frames: the displayed frame, or a region split.
pub(crate) fn encode_frame_lossless(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    settings: LosslessSettings,
    patches: bool,
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let pool = ThreadPool::new_lossless(num_threads);
    let mut scratch = Box::new(CoderScratch::lossless());
    FrameCoder {
        settings,
        patches,
        pool: &pool,
        scratch: &mut scratch,
    }
    .encode_image(linear, alpha, writer);
}

/// The base of a cell layer (see `cell_layer`): the 8-bit RGB image with
/// its opaque alpha, coded with the displayed frame's patches and saved to
/// reference slot 0 for the layer blended over it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame_lossless_base(
    linear: &Image3Si,
    alpha: &AlphaPlane,
    patches: bool,
    speed: crate::Speed,
    decoding_speed: crate::DecodingSpeed,
    near_lossless: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) {
    FrameCoder {
        settings: LosslessSettings {
            near_lossless,
            ..LosslessSettings::rgb8(speed, decoding_speed)
        },
        patches,
        pool,
        scratch,
    }
    .encode_displayed(linear, Some(alpha), true, None, writer);
}

/// Everything the frames of one lossless image share.
struct FrameCoder<'p, 's> {
    settings: LosslessSettings,
    /// Patch plans (glyph or tile) compete with the plain frame.
    patches: bool,
    pool: &'p ThreadPool,
    scratch: &'s mut CoderScratch,
}

/// A screenshot cut into a base and one cropped frame per rectangle.
struct RegionSplit {
    regions: Vec<Region>,
    /// Each rectangle's pixels, coded as its own kReplace frame.
    crops: Vec<(Image3Si, Option<AlphaPlane>)>,
    /// The image with the rectangles cut out, saved to reference slot 0.
    base: Image3Si,
    base_alpha: Option<AlphaPlane>,
}

impl FrameCoder<'_, '_> {
    /// Codes one frame (see [`encode_frame_lossless_core`]).
    fn core(
        &mut self,
        image: &Image3Si,
        alpha: Option<&AlphaPlane>,
        settings: LosslessSettings,
        kind: ModularFrameKind<'_>,
        local_trees: bool,
        writer: &mut BitWriter,
    ) -> bool {
        encode_frame_lossless_core(
            image,
            alpha,
            settings,
            kind,
            local_trees,
            self.pool,
            self.scratch,
            writer,
        )
    }

    /// See [`coarse_frame_bits`].
    fn coarse_bits(&mut self, image: &Image3Si, alpha: Option<&AlphaPlane>) -> Option<f64> {
        coarse_frame_bits(
            image,
            alpha,
            lossless_min_symbol(self.settings.max_bits),
            self.settings.decoding_speed.use_weighted_predictor(),
            self.pool,
            self.scratch,
        )
    }

    fn encode_image(
        &mut self,
        linear: &Image3Si,
        alpha: Option<&AlphaPlane>,
        writer: &mut BitWriter,
    ) {
        // The whole frame's glyph plan, found once for both steps.
        let mut whole_glyphs: Option<Option<LosslessPatches>> = None;

        // Step 1: the region split, when the estimates favor it.
        if let Some((split, sure)) = self.plan_region_split(linear, alpha, &mut whole_glyphs) {
            if sure {
                drop(whole_glyphs);
                self.encode_region_split(&split, writer);
                return;
            }
            let mut split_writer = BitWriter::new();
            self.encode_region_split(&split, &mut split_writer);
            drop(split);
            // The estimate ranks the two within a few percent and misses
            // what the cut does to the base's palette and tree, so the
            // whole frame competes on complete size.
            let mut whole_writer = BitWriter::new();
            self.encode_displayed(linear, alpha, false, whole_glyphs.take(), &mut whole_writer);
            append_smaller(&split_writer, &whole_writer, writer);
            return;
        }

        // Step 2: the displayed frame.
        self.encode_displayed(linear, alpha, false, whole_glyphs, writer);
    }

    /// The region split worth encoding, else `None`: the rare-color plan
    /// when its first-stage estimate beats the whole frame's, else the plan
    /// with flat illustrations when complete `Fast` encodes favor it. The
    /// flag says the split is sure to win (`REGION_SURE_ESTIMATE`); without
    /// it the whole frame races on complete size. Finds the whole frame's
    /// glyph plan into `whole_glyphs` along the way.
    fn plan_region_split(
        &mut self,
        linear: &Image3Si,
        alpha: Option<&AlphaPlane>,
        whole_glyphs: &mut Option<Option<LosslessPatches>>,
    ) -> Option<(RegionSplit, bool)> {
        let s = self.settings;
        if !s.speed.effort().region_frames
            || s.upsampling != 1
            || s.progressive
            || s.num_color != 3
            || s.max_bits != 8
        {
            return None;
        }
        let mut whole = None;
        let mut rejected = None;
        if let Some(regions) = crate::region_frames::plan(linear, alpha, self.pool)
            && let Some((split, estimate)) =
                self.region_candidate(linear, alpha, regions, whole_glyphs, &mut whole)
        {
            if let Some(estimate) = estimate.filter(|&ratio| ratio < REGION_SPLIT_MARGIN) {
                let sure = estimate <= REGION_SURE_ESTIMATE
                    || self.proxy_ratio(&split, linear, alpha) <= REGION_SURE_PROXY;
                return Some((split, sure));
            }
            rejected = Some(split);
        }
        let regions = crate::region_frames::plan_local(linear, alpha, self.pool)?;
        let split = match rejected {
            // The same rectangles, none of them text-like: the estimate
            // already said no.
            Some(split) if split.regions == regions => split,
            // The estimate's yes is not taken for this plan. Of the twelve
            // frames it sent to the race (flags, emoji, flat illustrations,
            // a page of contour bands) ten lost it, at 1.3-2.7 times their
            // encode; the proxy turns nine of those away, and of the two
            // wins it keeps one and loses a color-bar chart 4.5 %.
            rejected => {
                drop(rejected);
                self.region_candidate(linear, alpha, regions, whole_glyphs, &mut whole)?
                    .0
            }
        };
        let proxy = self.proxy_ratio(&split, linear, alpha);
        (proxy < REGION_PROXY_MARGIN).then_some((split, proxy <= REGION_SURE_PROXY))
    }

    /// The split's size over the whole frame's, both as complete encodes at
    /// [`REGION_PROXY_SPEED`].
    fn proxy_ratio(
        &mut self,
        split: &RegionSplit,
        linear: &Image3Si,
        alpha: Option<&AlphaPlane>,
    ) -> f64 {
        let mut proxy = FrameCoder {
            settings: LosslessSettings {
                speed: REGION_PROXY_SPEED,
                ..self.settings
            },
            patches: self.patches,
            pool: self.pool,
            scratch: &mut *self.scratch,
        };
        let split_bits = {
            let mut split_writer = BitWriter::new();
            proxy.encode_region_split(split, &mut split_writer);
            split_writer.bits_written()
        };
        let mut whole_writer = BitWriter::new();
        proxy.encode_displayed(linear, alpha, false, None, &mut whole_writer);
        split_bits as f64 / whole_writer.bits_written().max(1) as f64
    }

    /// The split of `regions` once text-like rectangles stay in the base,
    /// and its first-stage estimate over the whole frame's (cached in
    /// `whole`) where both exist; `None` when no rectangle is left.
    fn region_candidate(
        &mut self,
        linear: &Image3Si,
        alpha: Option<&AlphaPlane>,
        mut regions: Vec<Region>,
        whole_glyphs: &mut Option<Option<LosslessPatches>>,
        whole: &mut Option<Option<f64>>,
    ) -> Option<(RegionSplit, Option<f64>)> {
        let mut crops: Vec<_> = regions
            .iter()
            .map(|r| crate::region_frames::crop(linear, alpha, r))
            .collect();
        let mut own: Vec<Option<f64>> = crops
            .iter()
            .map(|(c, a)| self.coarse_bits(c, a.as_ref()))
            .collect();

        // A rectangle of text (a nested screenshot) codes worse as its own
        // frame than inside the whole, whose glyph patches cover it: it
        // stays in the base when the patches save a share of its estimate.
        if self.patches
            && let Some(glyphs) =
                whole_glyphs.get_or_insert_with(|| find_lossless_glyph_patches(linear, self.pool))
        {
            let keep: Vec<bool> = regions
                .iter()
                .zip(&own)
                .map(|(r, &own)| {
                    let (patched, patched_alpha) =
                        crate::region_frames::crop(&glyphs.base, alpha, r);
                    let text_like = match (own, self.coarse_bits(&patched, patched_alpha.as_ref()))
                    {
                        (Some(own), Some(patched)) => patched < own * (1.0 - REGION_GLYPH_SAVING),
                        _ => false,
                    };
                    !text_like
                })
                .collect();
            retain_kept(&mut regions, &keep);
            retain_kept(&mut crops, &keep);
            retain_kept(&mut own, &keep);
        }

        // The split's estimate: the base, every crop and their headers.
        if regions.is_empty() {
            return None;
        }
        let (base, base_alpha) = crate::region_frames::cut(linear, alpha, &regions);
        let whole = match *whole {
            Some(bits) => bits,
            None => *whole.insert(self.coarse_bits(linear, alpha)),
        };
        let base_bits = self.coarse_bits(&base, base_alpha.as_ref());
        let split = own
            .iter()
            .copied()
            .chain([base_bits])
            .sum::<Option<f64>>()
            .map(|bits| bits + REGION_FRAME_OVERHEAD_BITS * regions.len() as f64);
        let estimate = match (whole, split) {
            (Some(whole), Some(split)) if whole > 0.0 => Some(split / whole),
            _ => None,
        };
        Some((
            RegionSplit {
                regions,
                crops,
                base,
                base_alpha,
            },
            estimate,
        ))
    }

    /// The base saved to slot 0, then each crop replacing its rectangle.
    fn encode_region_split(&mut self, split: &RegionSplit, writer: &mut BitWriter) {
        self.encode_displayed(&split.base, split.base_alpha.as_ref(), true, None, writer);
        for (i, ((crop, crop_alpha), r)) in split.crops.iter().zip(&split.regions).enumerate() {
            self.core(
                crop,
                crop_alpha.as_ref(),
                self.settings.single_pass(),
                ModularFrameKind::RgbCrop {
                    x0: r.x0,
                    y0: r.y0,
                    width: r.width,
                    height: r.height,
                    full: r.width == split.base.xsize() && r.height == split.base.ysize(),
                    last: i + 1 == split.regions.len(),
                },
                true,
                writer,
            );
        }
    }

    /// The displayed frame: the image itself, or a patched pair (an atlas
    /// frame and the image with the patches cut out) when that is smaller.
    /// `saved_first` saves the frame to reference slot 0 for frames coded
    /// over it. `glyph_plan` is the caller's glyph search, when it ran one.
    fn encode_displayed(
        &mut self,
        linear: &Image3Si,
        alpha: Option<&AlphaPlane>,
        saved_first: bool,
        glyph_plan: Option<Option<LosslessPatches>>,
        writer: &mut BitWriter,
    ) {
        let regular = if saved_first {
            ModularFrameKind::RgbSavedFirst
        } else {
            ModularFrameKind::Regular
        };
        let with_patches = self.patches && self.settings.num_color == 3;

        // Step 1: the patch plan. Glyph plans (arbitrary-position repeated
        // shapes: text, UI) come first; tile plans otherwise.
        let glyph_plan = match glyph_plan {
            Some(plan) => plan,
            // Exact glyphs first; where they find nothing, glyphs over
            // smooth ground as additive patches. The depths are those of
            // images: the XYB lattice frames of the lossy path come through
            // this coder at 13 and 14 bits and are no RGB.
            None if with_patches => find_lossless_glyph_patches(linear, self.pool).or_else(|| {
                let bits = self.settings.max_bits;
                matches!(bits, 8 | 10 | 12 | 16)
                    .then(|| find_lossless_additive_glyph_patches(linear, bits))?
            }),
            None => None,
        };
        // A bi-level page keeps the plain frame unless its glyphs cover
        // much of it: of 12 such pages with a glyph plan the patched pair
        // won two, by 0.8 and 1.1 %, and lost the rest by 1 to 25 % at 45 %
        // of their encode. The first-stage estimates do not rank the two
        // there (they miss the LZ77 a screen is coded by).
        if let Some(plan) = &glyph_plan
            && bilevel_keeps_regular(linear, alpha, plan)
        {
            drop(glyph_plan);
            self.core(linear, alpha, self.settings, regular, false, writer);
            return;
        }
        let is_glyph_plan = glyph_plan.is_some();
        let plan = if with_patches {
            glyph_plan.or_else(|| find_lossless_patches(linear, self.pool, self.scratch))
        } else {
            None
        };
        let Some(plan) = plan else {
            self.core(linear, alpha, self.settings, regular, true, writer);
            return;
        };

        // Patches change prediction, LZ77, entropy histograms, and add both a
        // reference-only frame and a dictionary. Their true cost cannot be
        // inferred reliably from covered pixel area, so both complete
        // alternatives are compared by their final byte-aligned bit count.
        let mut regular_writer = BitWriter::new();

        // Step 2 (tile plans): the regular frame first. A decisive learned
        // tree (never beaten by a tile plan) or a plan covering little of the
        // image (cannot pay for its dictionary and reference frame) skips
        // the patched encode.
        if !is_glyph_plan {
            // A plan that cannot be used is released first: its base is a
            // copy of the image (12 bytes per pixel) that would otherwise
            // stay allocated through the whole regular encode.
            if patch_coverage(&plan, linear.xsize() * linear.ysize()) < PATCH_MIN_COVERAGE {
                drop(plan);
                self.core(linear, alpha, self.settings, regular, true, writer);
                return;
            }
            let decisive_tree = self.core(
                linear,
                alpha,
                self.settings,
                regular,
                true,
                &mut regular_writer,
            );
            if decisive_tree {
                writer.append(&regular_writer);
                return;
            }
        }

        // Step 3: the patched pair.
        let mut patched_writer = BitWriter::new();
        let atlas_bits = self.encode_patched(&plan, alpha, saved_first, &mut patched_writer);

        // Step 4 (glyph plans): they win whenever the plan is any good, so
        // the regular frame (a complete second encode) is written only when
        // the first-stage estimates leave it a chance; the dense local-tree
        // learn is spent on the patched frame only.
        // An additive plan always races: what it leaves in the base is not
        // what the estimates were fitted on, and its gain turned into a 5 %
        // loss on one ramp page with the order its tiles were matched in.
        // So does a plan covering most of the image: the little it leaves in
        // the base is not what the estimates rank either.
        let additive = plan.references.iter().any(|r| r.add);
        let covering =
            patch_coverage(&plan, linear.xsize() * linear.ysize()) >= GLYPH_REGULAR_COVERAGE;
        let regular_competes = is_glyph_plan
            && (additive
                || covering
                || self.regular_competes(
                    linear,
                    alpha,
                    &plan,
                    atlas_bits,
                    patched_writer.bits_written(),
                ));
        drop(plan);
        if regular_competes {
            self.core(
                linear,
                alpha,
                self.settings,
                regular,
                false,
                &mut regular_writer,
            );
        }

        // Step 5: the smaller complete alternative.
        if regular_writer.bits_written() > 0
            && regular_writer.bits_written() < patched_writer.bits_written()
        {
            writer.append(&regular_writer);
        } else {
            writer.append(&patched_writer);
        }
    }

    /// The reference-only atlas frame, then the base coded with the patch
    /// dictionary. Returns the atlas frame's bits.
    fn encode_patched(
        &mut self,
        plan: &LosslessPatches,
        alpha: Option<&AlphaPlane>,
        saved_first: bool,
        writer: &mut BitWriter,
    ) -> usize {
        let (atlas_w, atlas_h) = (plan.atlas.xsize(), plan.atlas.ysize());
        let atlas_alpha = zero_alpha_like(alpha, atlas_w.saturating_mul(atlas_h));
        self.core(
            &plan.atlas,
            atlas_alpha.as_ref(),
            self.settings.single_pass(),
            ModularFrameKind::ReferenceOnly {
                width: atlas_w,
                height: atlas_h,
            },
            true,
            writer,
        );
        let atlas_bits = writer.bits_written();
        self.core(
            &plan.base,
            alpha,
            self.settings.single_pass(),
            ModularFrameKind::Patched {
                references: &plan.references,
                saved_first,
            },
            true,
            writer,
        );
        atlas_bits
    }

    /// The regular frame's first-stage estimate is within
    /// [`GLYPH_REGULAR_COARSE_MARGIN`] of the patched pair's (base estimate
    /// plus the atlas frame and dictionary), or within
    /// [`GLYPH_REGULAR_UNSURE_MARGIN`] when the pair's `patched_bits` show
    /// the estimates unreliable; `true` when either estimate is missing.
    fn regular_competes(
        &mut self,
        linear: &Image3Si,
        alpha: Option<&AlphaPlane>,
        plan: &LosslessPatches,
        atlas_bits: usize,
        patched_bits: usize,
    ) -> bool {
        let regular_est = self.coarse_bits(linear, alpha);
        let base_est = self.coarse_bits(&plan.base, alpha);
        let (Some(regular), Some(base)) = (regular_est, base_est) else {
            return true;
        };
        let mut dictionary = BitWriter::new();
        write_patch_dictionary(
            &plan.references,
            alpha.is_some(),
            self.scratch,
            &mut dictionary,
        );
        let patched = base + (atlas_bits + dictionary.bits_written()) as f64;
        let margin = if (patched_bits as f64) < patched * GLYPH_UNSURE_CODED_SHARE {
            GLYPH_REGULAR_UNSURE_MARGIN
        } else {
            GLYPH_REGULAR_COARSE_MARGIN
        };
        regular <= patched * margin
    }
}

/// Keeps the items whose `keep` flag is set.
fn retain_kept<T>(items: &mut Vec<T>, keep: &[bool]) {
    let mut kept = keep.iter();
    items.retain(|_| *kept.next().unwrap_or(&true));
}

/// The share of a `pixels`-pixel image the plan's patches cover.
fn patch_coverage(plan: &LosslessPatches, pixels: usize) -> f64 {
    let covered: usize = plan
        .references
        .iter()
        .map(|r| r.positions.len() * r.width * r.height)
        .sum();
    covered as f64 / pixels as f64
}

/// An all-zero alpha plane of `alpha`'s type, for the atlas frame.
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

/// First-stage learned-tree estimate of a frame's pixel bits: the ranking
/// stage's coarse learn (32 leaves, YCoCg planes, default WP preset, 1024-px
/// layout), or `None` when no tree beats the flat coder there. Frames of one
/// image relate to their complete encodes by nearly the same factor (a UI
/// screenshot and its glyph-patched base: 0.77 and 0.80), so it can rank two
/// competing frames before either is fully encoded.
fn coarse_frame_bits(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    min_symbol: u32,
    use_wp: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<f64> {
    let (xsize, ysize) = (linear.xsize(), linear.ysize());
    let source = rgb_ma_source(
        linear,
        alpha,
        xsize,
        6,
        alpha.and_then(constant_alpha_value),
        3,
    );
    let stage = rank_coarse(
        &source,
        xsize,
        ysize,
        min_symbol,
        WpParams::DEFAULT,
        use_wp,
        GroupLayout::LARGE,
        MA_FULL_SAMPLING,
        pool,
        scratch,
    )?;
    Some(stage.candidate.est_real.min(stage.candidate.flat_real))
}

/// Glyph plans covering less than this share of a bi-level page are not
/// encoded.
const BILEVEL_GLYPH_MIN_COVERAGE: f64 = 0.10;

/// A bi-level image whose glyph plan is not worth its encode: one with a
/// repeated screen (its regular frame codes the screen by LZ77 under
/// Zero-predictor leaves while the patched base has lost the repeats to the
/// atlas: a scanned manga page came out 15 % smaller as a regular frame),
/// or one whose glyphs cover little of it.
fn bilevel_keeps_regular(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    plan: &LosslessPatches,
) -> bool {
    alpha.is_none()
        && super::palette::bilevel_screen_share(linear).is_some_and(|share| {
            share >= super::palette::SCREEN_MIN_SHARE
                || patch_coverage(plan, linear.xsize() * linear.ysize())
                    < BILEVEL_GLYPH_MIN_COVERAGE
        })
}
