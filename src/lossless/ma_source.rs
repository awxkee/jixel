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

//! Learned-tree sources, layouts, candidates, and sampling budgets.

use super::header::write_modular_transforms;
use super::lz77::ConstantTail;
use super::ma_learning::leaf_offsets;
use super::ma_sampling::MaRefPlane;
use super::ma_streams::{LZ_DEEP_MATCHER_MAX_RATIO_PALETTE, LZ_DEEP_MATCHER_MAX_RATIO_RCT};
use super::palette::write_palette_transform;
use super::tree::{TREE_CTX_OFFSET, emit_learned_tree};
use super::{GroupLayout, palette};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::LZ77_MAX_CONTEXTS;
use crate::encode_image::AlphaPlane;
use crate::entropy::{Token, pack_signed};
use crate::image::Image3Si;
use crate::ma_tree::{LearnedTree, MA_REF_CHANNELS, MaNode, MaSamples};

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

const MA_FINAL_TARGET_SAMPLES: usize = 1 << 23;

const MA_LITE_TARGET_SAMPLES: usize = 1 << 19;

const MA_LITE_FULL_SAMPLE_LIMIT: usize = 1 << 19;

const MA_FINAL_FULL_SAMPLE_LIMIT: usize = 1 << 24;

/// (target samples, full-sample limit) of `tree_full_sampling`, the lite
/// budget and the dense final learn.
pub(super) const MA_FULL_SAMPLING: (usize, usize) = (MA_TARGET_SAMPLES, MA_FULL_SAMPLE_LIMIT);

pub(super) const MA_LITE_SAMPLING: (usize, usize) =
    (MA_LITE_TARGET_SAMPLES, MA_LITE_FULL_SAMPLE_LIMIT);

pub(super) const MA_DENSE_SAMPLING: (usize, usize) =
    (MA_FINAL_TARGET_SAMPLES, MA_FINAL_FULL_SAMPLE_LIMIT);

/// The dense final budget applies to sources whose first-stage estimate
/// costs at least this many bits per value: photos and paintings (2-3 b/v).
/// Screen and text content (0.3 b/v and below) gets nothing from it — its
/// gains come from local trees and patches — while a 7.7 MP UI screenshot
/// would spend 2.5x the CPU and 2.5 GB, and a web render came out 7% larger.
pub(super) const DENSE_FINAL_MIN_BITS_PER_VALUE: f64 = 1.0;

/// First-stage learner budget. Most unhelpful or simple trees terminate here;
/// only a saturated tree with a clear rate win is retrained on the full probe.
pub(super) const MA_COARSE_TARGET_SAMPLES: usize = 1 << 16;

pub(super) const MA_COARSE_MAX_LEAVES: usize = 32;

pub(super) const MA_DEEPEN_MIN_SAVINGS: f64 = 0.02;

/// Bits a split must save (image domain) before it's kept. Growth is
/// best-first, so this mostly prunes the tail once the leaf budget is spent.
pub(super) const MA_SPLIT_COST_BITS: f32 = 100.0;

/// Split cost of the final learns. Half the first-stage cost grows trees
/// that code 0.4-2.2% smaller on nearly every photo, painting and UI screen
/// (Kodak 24/24, -0.72% in total) for 15-25% more encode time there. Ranking
/// and coarse learns keep the full cost: their estimates gate candidate
/// sources and frames, and a text render lost 33% when a lower cost flipped
/// one of those gates. Below 50 the trees overfit (25: Kodak +9%).
pub(super) const MA_FINAL_SPLIT_COST_BITS: f32 = 50.0;

/// Leaf cap: pixel contexts + the LZ77 distance context must fit the
/// LZ77_MAX_CONTEXTS scratch; ANS clustering reduces them to <= 128
/// histograms. Best-first growth adds exactly one leaf per split, so the cap
/// is hard.
pub(super) const MA_MAX_LEAVES: usize = LZ77_MAX_CONTEXTS - 1;

/// Minimum samples in a node before it must become a leaf.
pub(super) const MA_MIN_NODE_SAMPLES: usize = 128;

/// Grow-then-prune of palette index trees (`with_ma_prune`): growth split
/// cost and minimum node, and the bits the prune charges per kept leaf.
pub(super) const MA_PRUNE_SPLIT_COST_BITS: f32 = 20.0;

pub(super) const MA_PRUNE_MIN_NODE_SAMPLES: usize = 16;

pub(super) const MA_PRUNE_LEAF_BITS: f32 = 40.0;

/// Predictors each side of a candidate split may choose from (the node's
/// cheapest ones).
pub(super) const MA_SIDE_PREDS: usize = 4;

/// `MA_SIDE_PREDS` of the final RCT tree learn under `tree_wide_side_preds`.
pub(super) const MA_WIDE_SIDE_PREDS: usize = 7;

/// Leaf cap of sub-sampled final learns without `tree_deep_final`.
pub(super) const MA_LITE_MAX_LEAVES: usize = 1024;

/// Split candidates of sub-sampled final learns without `tree_deep_final`.
pub(super) const MA_LITE_CANDIDATES: usize = 16;

/// First learning stage: the merged samples with a gated coarse tree (or the
/// complete tree for small sample sets). Cheap enough to run once per group
/// layout; only the winning layout is deepened.
pub(super) struct CoarseLearn {
    pub(super) samples: MaSamples,
    pub(super) stride: usize,
    pub(super) max_candidates: usize,
    /// Leaf predictor offsets apply (RCT sources; palette indices lose).
    pub(super) leaf_offsets: bool,
    pub(super) candidate: LearnedCandidate,
    /// The coarse tree saturated its leaf budget with a clear win, so the
    /// full-probe second stage is worth its cost.
    pub(super) deepen: bool,
}

/// A gated learned tree with its BFS emission and rate estimate.
pub(super) struct LearnedCandidate {
    pub(super) tree: LearnedTree,
    pub(super) tree_tokens: Vec<Token>,
    pub(super) leaf_ctx: Vec<u32>,
    /// Predictor offset per context (see `leaf_offsets`).
    pub(super) leaf_offset: Vec<i32>,
    pub(super) num_ctx: u32,
    /// Estimated image-domain bits of the tree-coded frame, headers included.
    pub(super) est_real: f64,
    /// Estimated image-domain bits of the flat alternative.
    pub(super) flat_real: f64,
    /// Estimated fractional saving over the flat path.
    pub(super) estimated_savings: f64,
    /// The context of the constant channel's leaf, once `with_constant_channel`
    /// wrapped the tree (always 0: the leaf is the root's first child).
    pub(super) constant_ctx: Option<u32>,
}

impl LearnedCandidate {
    /// The tree routed through a channel split whose upper branch is a
    /// Zero-predictor leaf with offset `value` for channel `chan_id` (the
    /// last channel of its stream); the learned tree keeps every other
    /// channel. Leaf contexts shift by one; the new leaf is context 0.
    pub(super) fn with_constant_channel(&self, chan_id: u32, value: i32) -> LearnedCandidate {
        assert!(
            chan_id >= 1,
            "a constant channel follows the color channels"
        );
        assert!(self.constant_ctx.is_none(), "one constant channel per tree");
        let mut nodes = Vec::with_capacity(self.tree.nodes.len() + 2);
        nodes.push(MaNode::Split {
            prop: 0,
            val: chan_id as i32 - 1,
            gt: 1,
            le: 2,
        });
        nodes.push(MaNode::Leaf {
            pred: 0,
            offset: value,
        });
        nodes.extend(self.tree.nodes.iter().map(|node| match *node {
            MaNode::Split { prop, val, gt, le } => MaNode::Split {
                prop,
                val,
                gt: gt + 2,
                le: le + 2,
            },
            leaf @ MaNode::Leaf { .. } => leaf,
        }));
        let tree = LearnedTree {
            nodes,
            est_bits: self.tree.est_bits,
            flat_bits: self.tree.flat_bits,
        };
        let (mut tree_tokens, leaf_ctx, num_ctx) = emit_learned_tree(&tree);
        debug_assert_eq!(num_ctx, self.num_ctx + 1);
        debug_assert_eq!(leaf_ctx[1], 0);
        let mut leaf_offset = Vec::with_capacity(num_ctx as usize);
        leaf_offset.push(value);
        leaf_offset.extend_from_slice(&self.leaf_offset);
        for (token, &offset) in tree_tokens
            .iter_mut()
            .filter(|token| token.context == TREE_CTX_OFFSET)
            .zip(&leaf_offset)
        {
            token.value = pack_signed(offset);
        }
        LearnedCandidate {
            tree,
            tree_tokens,
            leaf_ctx,
            leaf_offset,
            num_ctx,
            est_real: self.est_real,
            flat_real: self.flat_real,
            estimated_savings: self.estimated_savings,
            constant_ctx: Some(0),
        }
    }

    /// The tail of a `w` x `h` crop of the constant channel.
    pub(super) fn constant_tail(&self, w: usize, h: usize) -> ConstantTail {
        ConstantTail {
            context: self.constant_ctx.expect("constant channel leaf"),
            count: w * h,
        }
    }

    /// Fill offset tokens in BFS context order from the retained offset sample.
    pub(super) fn set_leaf_offsets(&mut self, samples: &MaSamples, step: usize) {
        self.leaf_offset = crate::ma_tree::with_ma_properties!(&samples.props, |rows| {
            leaf_offsets(
                &self.tree,
                &self.leaf_ctx,
                self.num_ctx,
                rows.iter().zip(&samples.tok).step_by(step),
            )
        });
        for (token, &offset) in self
            .tree_tokens
            .iter_mut()
            .filter(|token| token.context == TREE_CTX_OFFSET)
            .zip(&self.leaf_offset)
        {
            token.value = pack_signed(offset);
        }
    }
}

/// One modular channel as the learner and tokenizer see it.
pub(super) struct MaChannel<'a> {
    pub(super) w: usize,
    pub(super) h: usize,
    /// Meta channels (palette) always live in the global stream, whatever
    /// their size.
    pub(super) meta: bool,
    pub(super) pixels: MaPixels<'a>,
    /// `Some(value)` for a channel whose every sample is `value`: it is
    /// neither sampled nor tokenized, and the written tree routes it to a
    /// Zero-predictor leaf with that offset (see `ConstantTail`).
    pub(super) constant: Option<i32>,
}

/// Borrow the original sample representation, including palette indices and
/// alpha, without expanding or copying rows for MA traversal.
pub(super) enum MaPixels<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    I32(&'a [i32]),
}

// Resolve the sample type once per rectangle so the row kernels remain
// monomorphized. All variants provide the same cropped, borrowed row interface.
macro_rules! with_ma_pixels {
    ($channel:expr, |$pixels:ident| $body:expr) => {
        match &$channel.pixels {
            crate::lossless::ma_source::MaPixels::U8($pixels) => $body,
            crate::lossless::ma_source::MaPixels::U16($pixels) => $body,
            crate::lossless::ma_source::MaPixels::I32($pixels) => $body,
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
pub(super) enum MaTransform {
    /// RCT on the first three channels (`6` = YCoCg).
    Rct(u32),
    /// Global palette over `num_c` components: channel 0 is the palette
    /// meta-channel (`nb_colors` x `num_c`), channel 1 the index image.
    Palette { num_c: u32, nb_colors: u32 },
}

/// The modular channels of a frame plus their transform.
pub(super) struct MaSource<'a> {
    pub(super) channels: Vec<MaChannel<'a>>,
    pub(super) transform: MaTransform,
    /// Per-AC-group channel overrides: a group whose entry is `Some` codes
    /// those channels (a group-local Palette transform declared in its own
    /// GroupHeader) instead of crops of `channels`. Empty means none.
    pub(super) group_channels: Vec<Option<MaGroupChannels<'a>>>,
}

/// One AC group's own channels: the palette meta-channel followed by the
/// index channel, as the decoder reads them after that group's transform.
pub(super) struct MaGroupChannels<'a> {
    pub(super) channels: Vec<MaChannel<'a>>,
    pub(super) nb_colors: u32,
    pub(super) num_c: u32,
}

/// Quant tables reserved in the decoder's modular stream numbering
/// (`kNumQuantTables`); AC group streams are numbered after them.
const MODULAR_NUM_QUANT_TABLES: usize = 17;

/// The decoder's stream id (MA property 1) of AC group `group_index`:
/// `ModularStreamId::ModularAC(group, pass 0)` = 1 + 3 * DC groups + quant
/// tables + group. The global stream is id 0.
pub(crate) fn ac_stream_id(num_dc_groups: usize, group_index: usize) -> i32 {
    (1 + 3 * num_dc_groups + MODULAR_NUM_QUANT_TABLES + group_index) as i32
}

/// Where a channel is coded under a layout: in the single section / the
/// DC-global stream, or per AC group. Channel ids restart per stream, as the
/// decoder's channel property does.
pub(super) struct MaPlacement {
    pub(super) channel: usize,
    pub(super) chan_id: u32,
    pub(super) global: bool,
}

impl MaSource<'_> {
    /// Leaf budget of a final learn: the constant channel's leaf, added when
    /// the tree is written, must fit the context limit too.
    pub(super) fn max_leaves(&self) -> usize {
        MA_MAX_LEAVES - usize::from(self.channels.iter().any(|ch| ch.constant.is_some()))
    }

    /// Split candidates the learner should evaluate for this source.
    pub(super) fn max_candidates(&self) -> usize {
        match self.transform {
            MaTransform::Palette { .. } => MA_PALETTE_CANDIDATES,
            MaTransform::Rct(_) => MA_PHOTO_CANDIDATES,
        }
    }

    /// How far the run-token estimate may trail literals before the deep
    /// LZ77 matcher is skipped (see `LZ_DEEP_MATCHER_MAX_RATIO_RCT`); any
    /// palette index stream, global or per group, keeps the wide margin.
    pub(super) fn deep_lz_max_ratio(&self) -> f64 {
        match self.transform {
            MaTransform::Rct(_) if self.group_channels.iter().all(Option::is_none) => {
                LZ_DEEP_MATCHER_MAX_RATIO_RCT
            }
            _ => LZ_DEEP_MATCHER_MAX_RATIO_PALETTE,
        }
    }

    pub(super) fn write_transforms(&self, w: &mut BitWriter) {
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
    pub(super) fn placement(&self, layout: GroupLayout, single_group: bool) -> Vec<MaPlacement> {
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

    /// Values the learner samples and the tokenizer codes (constant
    /// channels have none).
    pub(super) fn total_values(&self) -> usize {
        self.channels
            .iter()
            .filter(|c| c.constant.is_none())
            .map(|c| c.w * c.h)
            .sum()
    }

    /// The constant channel of this source with its value, if it has one.
    /// It is the last channel of its stream, so the root channel split of
    /// `LearnedCandidate::with_constant_channel` isolates it.
    pub(super) fn constant_placement<'p>(
        &self,
        placement: &'p [MaPlacement],
    ) -> Option<(&'p MaPlacement, i32)> {
        let mut found: Option<(&MaPlacement, i32)> = None;
        for p in placement {
            if let Some(value) = self.channels[p.channel].constant {
                assert!(found.is_none(), "one constant channel per source");
                found = Some((p, value));
            }
        }
        let (p, value) = found?;
        assert!(
            placement
                .iter()
                .filter(|q| q.global == p.global)
                .all(|q| q.chan_id <= p.chan_id),
            "the constant channel is the last of its stream"
        );
        Some((p, value))
    }

    /// The override of AC group `group_index`, if it has one.
    pub(super) fn group_override(&self, group_index: usize) -> Option<&MaGroupChannels<'_>> {
        self.group_channels
            .get(group_index)
            .and_then(Option::as_ref)
    }

    /// The channel a job or placement refers to: an override's own channel
    /// when `group_override` is set, else one of the source's channels.
    pub(super) fn channel_at(
        &self,
        group_override: Option<usize>,
        channel: usize,
    ) -> &MaChannel<'_> {
        match group_override {
            Some(group_index) => {
                &self
                    .group_override(group_index)
                    .expect("override group")
                    .channels[channel]
            }
            None => &self.channels[channel],
        }
    }

    /// The reference planes of placement `p`'s channel: earlier channels of
    /// the same stream and size, most recent first, cropped at (x0, y0).
    pub(super) fn ref_planes(
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
                    crate::lossless::ma_source::MaPixels::I32(pixels)
                        if other.w == ch.w && other.h == ch.h =>
                    {
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

/// The YCoCg(A) planes with per-group Palette transforms where a group's
/// color count allows one. Groups without a palette code the planes
/// directly, so the frame keeps its RCT and one learned tree serves both.
pub(super) fn mixed_ma_source<'a>(
    ycocg: &'a Image3Si,
    alpha: Option<&'a AlphaPlane>,
    xsize: usize,
    palettes: &'a [Option<palette::LocalPaletteGroup>],
    alpha_constant: Option<i32>,
) -> MaSource<'a> {
    let mut source = rgb_ma_source(ycocg, alpha, xsize, 6, alpha_constant, 3);
    source.group_channels = palettes
        .iter()
        .map(|palette| {
            let palette = palette.as_ref()?;
            let num_c = palette.palette.len() / palette.nb_colors;
            Some(MaGroupChannels {
                channels: vec![
                    MaChannel {
                        w: palette.nb_colors,
                        h: num_c,
                        meta: true,
                        pixels: crate::lossless::ma_source::MaPixels::I32(&palette.palette),
                        constant: None,
                    },
                    MaChannel {
                        w: palette.w,
                        h: palette.h,
                        meta: false,
                        pixels: crate::lossless::ma_source::MaPixels::U16(&palette.indices),
                        constant: None,
                    },
                ],
                nb_colors: palette.nb_colors as u32,
                num_c: num_c as u32,
            })
        })
        .collect();
    source
}

/// The YCoCg(A) planes, or the single gray plane, as a learned-tree source.
pub(super) fn rgb_ma_source<'a>(
    linear: &'a Image3Si,
    alpha: Option<&'a AlphaPlane>,
    xsize: usize,
    rct_type: u32,
    alpha_constant: Option<i32>,
    num_color: usize,
) -> MaSource<'a> {
    let ysize = linear.ysize();
    let mut channels: Vec<MaChannel<'a>> = (0..num_color)
        .map(|c| {
            let pd = linear.plane_data(c);
            MaChannel {
                w: xsize,
                h: ysize,
                meta: false,
                pixels: crate::lossless::ma_source::MaPixels::I32(pd),
                constant: None,
            }
        })
        .collect();
    if let Some(a) = alpha {
        channels.push(MaChannel {
            w: xsize,
            h: ysize,
            meta: false,
            pixels: match a {
                AlphaPlane::U8(pixels) => crate::lossless::ma_source::MaPixels::U8(pixels),
                AlphaPlane::U16 { data, .. } => crate::lossless::ma_source::MaPixels::U16(data),
                AlphaPlane::F32(pixels) => crate::lossless::ma_source::MaPixels::I32(pixels),
            },
            constant: alpha_constant,
        });
    }
    MaSource {
        channels,
        transform: MaTransform::Rct(rct_type),
        group_channels: Vec::new(),
    }
}

/// Below this covered-area fraction the patched alternative is not encoded.
pub(super) const PATCH_MIN_COVERAGE: f64 = 0.05;

/// A palette candidate whose first complete layout trails the RGB frame's
/// final estimate by more than `PALETTE_FINAL_MARGIN` times this does not
/// learn its 256-px layout.
pub(super) const PALETTE_LAYOUT_MARGIN: f64 = 1.04;

/// Below this many values (pixels x channels) the WP-preset and RCT
/// runner-up rankings are skipped: each is a complete learn of the image
/// (20-35 ms single-threaded for a 128x128 icon coded in under a kilobyte),
/// six to eight of them were most of a small icon's encode time, and their
/// absolute gain on such files is a few bytes.
pub(super) const RANK_MIN_VALUES: usize = 1 << 19;

/// RGB sources above this many values (pixels x channels) learn only the
/// 1024-px group layout; see the layout selection.
pub(super) const RGB_SINGLE_LAYOUT_MIN_VALUES: usize = 1 << 20;

pub(super) fn source_values_of(xsize: usize, ysize: usize, nb_chans: usize) -> usize {
    xsize * ysize * nb_chans
}

/// A per-group-palette first stage is grown into a complete frame only when
/// it already beats the best source's own first stage. The bottom of the
/// range separates cleanly (a winning plan lands at 0.75); the top does not
/// (losers at 1.019 and 1.022 with a 1.65% winner between them at 1.023), so
/// the strict bar gives up that small win to spare the losers a full encode.
pub(super) const PER_GROUP_MARGIN: f64 = 1.0;

pub(super) use with_ma_pixels;
