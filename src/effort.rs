/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Per-tier encoder tool switches. Every effort-dependent decision reads one
//! named gate here instead of comparing against a [`Speed`] variant, so a
//! tier is just a row of this table.

use crate::Speed;

macro_rules! effort_gates {
    (
        $($(#[$doc:meta])* $name:ident,)* ;
        $($(#[$xdoc:meta])* $xname:ident,)*
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) struct Effort {
            $($(#[$doc])* pub(crate) $name: bool,)*
            $($xname: bool,)*
        }

        impl Effort {
            const NONE: Self = Self { $($name: false,)* $($xname: false,)* };
            const ALL: Self = Self { $($name: true,)* $($xname: true,)* };

            $(
                $(#[$xdoc])*
                #[cfg_attr(not(feature = "extraslow"), allow(dead_code))]
                #[inline(always)]
                pub(crate) const fn $xname(&self) -> bool {
                    cfg!(feature = "extraslow") && self.$xname
                }
            )*

            #[cfg(test)]
            fn gates(&self) -> Vec<(&'static str, bool)> {
                vec![$((stringify!($name), self.$name),)* $((stringify!($xname), self.$xname),)*]
            }
        }
    };
}

effort_gates! {
    // --- lossy VarDCT ---
    /// Full thread budget on small images (otherwise one lane per 64K pixels).
    full_threads,
    /// Square merge search (16x16/32x32) over the plain 8x8 DCT grid, with
    /// the per-strategy block contexts and custom-matrix accounting it needs.
    square_merges,
    /// 32x32 merges in the square search (otherwise 16x16 only).
    merge_32,
    /// Evaluate merges on quads whose DCT8 incumbent already exceeds
    /// `MERGE_PRESCREEN_BITS`.
    merge_busy_quads,
    /// Per-block quantizer refinement after the transform search.
    quant_refine,
    /// Pixel-chromacity X steps outside the chroma policy.
    pixel_chromacity_x,
    /// CfL slope search for YCbCr frames.
    cfl_ycbcr_rdo,
    /// Weighted DC predictor with per-context gradient fallback.
    dc_predictor_search,
    /// Chroma DC rounding with frozen residual prices and a pooled rate check.
    dc_rdoq,
    /// Per-image learned DC context tree raced against the static tree.
    dc_learned_tree,
    /// Per-cluster hybrid-uint config selection.
    huc_select,
    /// Prefix-model AC context clustering (otherwise a coarse context merge).
    ac_prefix_clustering,
    /// Greedy context clustering may open every cluster the format allows
    /// (otherwise `FAST_CLUSTERS_LIMIT`).
    wide_clustering,
    /// Fast ANS cluster refinement of the DC and AC codes.
    ans_fast_refine,
    /// Below `ans_recluster`: recluster at distance >= 3 during the fast refinement.
    ans_recluster_coarse,
    /// Rectangular, sub-8x8 (DCT4/AFV/IDENTITY) and leaf-first transform search.
    rectangles,
    /// SSIM reconstruction rerank at every distance (otherwise high quality only).
    full_rerank,
    dct64,
    fine_mosaic,
    learned_rate,
    /// Chroma content analysis and every policy it drives: saturated/pair-B/x-heavy
    /// tables, point chroma, chroma HF protection, pixel-chromacity B steps.
    chroma_policy,
    /// Exact DP re-tiling of closed 64x64 tiles over 8/16/32-family pieces at
    /// any 8-pixel origin; skipped where `layout_loop` re-chooses layouts exactly.
    /// Where it installs its layout, the rerank's downgrade and upgrade passes
    /// are skipped: the DP covers both.
    layout_retile,
    /// 32-family pieces (strips, 32x16/16x32, 32x32) in the `layout_retile` DP;
    /// otherwise it re-tiles over DCT8, pairs and 16x16 only.
    layout_retile_32,
    /// Coefficient-model DP re-tiling of the merge search's tiles over DCT8,
    /// pairs and 16x16 at any 8-pixel origin; superseded by `layout_retile`.
    layout_dp_coefficient,
    cfl_rdo,
    /// Frame-constant EPF strength and pass count fitted on the final
    /// reconstruction by the luma proxy.
    epf_fit,
    yellow_opsin,
    coeff_orders,
    ac_ctx_plan,
    ma_dc_tree,
    /// Luma-error-compensated B DC targets and, per chroma channel, a 2x finer
    /// DC step holding a squeeze-smoothed plane when cheaper and closer.
    chroma_dc_squeeze,
    /// Frame-level X DC chroma-from-luma search on the unrounded source,
    /// under the final luma levels and DC quantization steps.
    dc_x_cfl,
    ans_refine,
    /// Below `ans_refine`: recluster at every distance (otherwise from d=3).
    ans_recluster,
    lossy_modular_auto,
    splines,
    patches_default,
    /// `EncodeConfig::ink_layer` is on by default at this speed.
    ink_layer_default,
    /// `EncodeConfig::vector_layers` is on by default at this speed.
    vector_layers_default,
    /// Saturated pixels take a free target beyond the clip limit before the
    /// forward transform.
    clip_continuation,
    /// Lossy tile patches also match near-duplicate tiles, not only exact ones.
    near_patch_tiles,
    /// Any config strictly cheaper than the default hybrid-uint wins (else a 0.5% bar).
    huc_strict,
    // --- modular (lossless, lossy squeeze, patch atlases) ---
    /// Ring-hash LZ77 matcher next to run-only LZ77.
    lz_deep,
    /// ANS, per-cluster hybrid-uint selection and refined clustering for LZ streams.
    lz_refined_entropy,
    predictor_search,
    rct_search,
    /// RCT ranking scores every row (otherwise one row in `RCT_DENSE_ROW_STEP`).
    rct_every_row,
    /// RCT ranking scores one row in `RCT_DENSE_ROW_STEP` (otherwise one in
    /// `RCT_SPARSE_ROW_STEP`).
    rct_dense_rows,
    wp_search,
    /// Parallel per-group palettes behind a global coverage gate.
    local_palette_select,
    /// Searched predictors for the local-palette frame (otherwise fixed).
    local_palette_predictors,
    /// Flat-path channels split their context on the WP error property at
    /// fixed cut-offs (`WP_CONTEXT_CUTOFFS`).
    flat_wp_contexts,
    /// Encode the normal frame too when a palette frame succeeded; keep the smaller.
    palette_dual,
    /// Meta-adaptive learned trees and the candidate-frame race.
    learned_tree,
    /// Coarse-learn comparison of the palette and RGB sources and the RCT runner-up.
    tree_rank,
    /// 8M/16M-sample final tree learn.
    tree_dense_sampling,
    /// 2M-sample tree learn (otherwise 512K).
    tree_full_sampling,
    /// Learn under both the 1024 px and the 256 px group layouts.
    tree_dual_layout,
    /// Final learns of sub-sampled sources keep the full leaf budget and split
    /// candidates (otherwise `MA_LITE_MAX_LEAVES` and `MA_LITE_CANDIDATES`).
    tree_deep_final,
    /// Final learns sample their source afresh (otherwise they deepen the
    /// ranking stage's sample set when it matches their own).
    tree_fresh_final_samples,
    tree_global_palette,
    tree_group_palette,
    tree_ctx_v1,
    /// Gradient and photo-like rectangles of a screenshot as cropped frames
    /// of their own (`region_frames`).
    region_frames,
    ;
    // --- ExtraSlow only: always off without the `extraslow` feature ---
    /// Closed-loop quant-field refinement on the reconstructed luma (MS-SSIM).
    aq_closed_loop,
    /// 128-family transforms placed by the closed loop's exact SSIMULACRA2 arbiter.
    dct128_loop,
    /// 16x16 transform layouts (DCT8 quad, DCT16, rectangle pairs) re-chosen by
    /// the closed loop's exact SSIMULACRA2 arbiter.
    layout_loop,
    /// Race frequency-derived coefficient orders against natural scans using
    /// serialized permutation headers and final entropy-coded AC streams.
    coeff_order_entropy_race,
    /// Rank B DC CfL slopes with weighted prediction in pooled WP-error contexts.
    dc_cfl_weighted,
    /// Re-tokenize on the custom orders before pricing RDOQ however little
    /// the scans moved (otherwise only past `RDOQ_FRESH_PRICES_MIN_SCAN_MOVE`).
    rdoq_fresh_prices,
    /// Broader final hybrid-uint and ANS-table search.
    ans_exhaustive,
    /// Price every hybrid-uint candidate under its fully searched ANS table.
    lz_exhaustive_entropy,
    /// Broader cost-aware LZ77 parses alongside the normal stream candidates.
    lz_priced_search,
    /// Ranking learns of every WP preset.
    tree_rank_deep,
    /// Final RCT tree learns score more predictors on each side of a split.
    tree_wide_side_preds,
}

const ULTRAFAST: Effort = Effort::NONE;

const FASTEST: Effort = Effort {
    square_merges: true,
    dc_predictor_search: true,
    ac_prefix_clustering: true,
    ..ULTRAFAST
};

const FAST: Effort = Effort {
    full_threads: true,
    square_merges: true,
    merge_32: true,
    merge_busy_quads: true,
    quant_refine: true,
    layout_dp_coefficient: true,
    pixel_chromacity_x: true,
    cfl_ycbcr_rdo: true,
    dc_predictor_search: true,
    dc_learned_tree: true,
    huc_select: true,
    ac_prefix_clustering: true,
    wide_clustering: true,
    ans_fast_refine: true,
    ans_recluster_coarse: true,
    rct_search: true,
    local_palette_predictors: true,
    flat_wp_contexts: true,
    ..Effort::NONE
};

const MEDIUM: Effort = Effort {
    layout_retile_32: false,
    tree_full_sampling: false,
    ..MEDIUM_SLOW
};

const MEDIUM_SLOW: Effort = Effort {
    learned_rate: false,
    vector_layers_default: false,
    clip_continuation: false,
    epf_fit: false,
    ac_ctx_plan: false,
    lossy_modular_auto: false,
    splines: false,
    rct_every_row: false,
    tree_dense_sampling: false,
    tree_dual_layout: false,
    tree_ctx_v1: false,
    tree_deep_final: false,
    tree_fresh_final_samples: false,
    patches_default: false,
    region_frames: false,
    ..SLOW
};

const SLOW: Effort = Effort {
    aq_closed_loop: false,
    dct128_loop: false,
    layout_loop: false,
    coeff_order_entropy_race: false,
    dc_cfl_weighted: false,
    rdoq_fresh_prices: false,
    ans_exhaustive: false,
    lz_exhaustive_entropy: false,
    lz_priced_search: false,
    tree_rank_deep: false,
    tree_wide_side_preds: false,
    ..Effort::ALL
};

#[cfg(feature = "extraslow")]
const EXTRASLOW: Effort = Effort::ALL;

impl Speed {
    pub(crate) const fn effort(self) -> &'static Effort {
        match self {
            Speed::UltraFast => &ULTRAFAST,
            Speed::Fastest => &FASTEST,
            Speed::Fast => &FAST,
            Speed::Medium => &MEDIUM,
            Speed::MediumSlow => &MEDIUM_SLOW,
            Speed::Slow => &SLOW,
            #[cfg(feature = "extraslow")]
            Speed::ExtraSlow => &EXTRASLOW,
            #[cfg(not(feature = "extraslow"))]
            Speed::ExtraSlow => &SLOW,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slower_tiers_keep_every_faster_tool() {
        let tiers = [
            Speed::UltraFast,
            Speed::Fastest,
            Speed::Fast,
            Speed::Medium,
            Speed::MediumSlow,
            Speed::Slow,
            Speed::ExtraSlow,
        ];
        for pair in tiers.array_windows::<2>() {
            let (faster, slower) = (pair[0].effort().gates(), pair[1].effort().gates());
            for ((name, on), (_, slower_on)) in faster.into_iter().zip(slower) {
                assert!(
                    !on || slower_on,
                    "{name} on at {:?} but off at {:?}",
                    pair[0],
                    pair[1]
                );
            }
        }
    }
}
