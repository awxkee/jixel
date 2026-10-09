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
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) struct Effort {
            $($(#[$doc])* pub(crate) $name: bool,)*
        }

        impl Effort {
            const NONE: Self = Self { $($name: false,)* };
            const ALL: Self = Self { $($name: true,)* };

            #[cfg(test)]
            fn gates(&self) -> Vec<(&'static str, bool)> {
                vec![$((stringify!($name), self.$name),)*]
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
    /// Wider AC trellis for 32/64-family transforms at distance >= 3.
    large_transform_rdoq,
    fine_mosaic,
    learned_rate,
    /// Chroma content analysis and every policy it drives: saturated/pair-B/x-heavy
    /// tables, point chroma, chroma HF protection, pixel-chromacity B steps.
    chroma_policy,
    cfl_rdo,
    chroma_deadzone,
    yellow_opsin,
    coeff_orders,
    /// Re-tokenize on the custom orders before pricing RDOQ however little
    /// the scans moved (otherwise only past `RDOQ_FRESH_PRICES_MIN_SCAN_MOVE`).
    rdoq_fresh_prices,
    ac_ctx_plan,
    ma_dc_tree,
    /// Luma-error-compensated B DC targets and, per chroma channel, a 2x finer
    /// DC step holding a squeeze-smoothed plane when cheaper and closer.
    chroma_dc_squeeze,
    /// Frame-level X DC chroma-from-luma search on the unrounded source,
    /// under the final luma levels and DC quantization steps.
    dc_x_cfl,
    ans_refine,
    /// Broader final hybrid-uint and ANS-table search.
    ans_exhaustive,
    /// Below `ans_refine`: recluster at every distance (otherwise from d=3).
    ans_recluster,
    lossy_modular_auto,
    splines,
    patches_default,
    /// Any config strictly cheaper than the default hybrid-uint wins (else a 0.5% bar).
    huc_strict,
    // --- modular (lossless, lossy squeeze, patch atlases) ---
    /// Ring-hash LZ77 matcher next to run-only LZ77.
    lz_deep,
    /// ANS, per-cluster hybrid-uint selection and refined clustering for LZ streams.
    lz_refined_entropy,
    /// Price every hybrid-uint candidate under its fully searched ANS table.
    lz_exhaustive_entropy,
    /// Broader cost-aware LZ77 parses alongside the normal stream candidates.
    lz_priced_search,
    predictor_search,
    rct_search,
    wp_search,
    /// Parallel per-group palettes behind a global coverage gate.
    local_palette_select,
    /// Encode the normal frame too when a palette frame succeeded; keep the smaller.
    palette_dual,
    /// Meta-adaptive learned trees and the candidate-frame race.
    learned_tree,
    /// Coarse-learn comparison of the palette and RGB sources and the RCT runner-up.
    tree_rank,
    /// Ranking learns of every WP preset.
    tree_rank_deep,
    /// 8M/16M-sample final tree learn.
    tree_dense_sampling,
    /// 2M-sample tree learn (otherwise 512K).
    tree_full_sampling,
    /// Learn under both the 1024 px and the 256 px group layouts.
    tree_dual_layout,
    /// Final RCT tree learns score more predictors on each side of a split.
    tree_wide_side_preds,
    tree_global_palette,
    tree_group_palette,
    tree_ctx_v1,
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
    pixel_chromacity_x: true,
    cfl_ycbcr_rdo: true,
    dc_predictor_search: true,
    dc_learned_tree: true,
    huc_select: true,
    ac_prefix_clustering: true,
    wide_clustering: true,
    ans_fast_refine: true,
    ans_recluster_coarse: true,
    ..Effort::NONE
};

const MEDIUM: Effort = Effort {
    full_rerank: false,
    learned_rate: false,
    coeff_orders: false,
    ac_ctx_plan: false,
    ma_dc_tree: false,
    chroma_dc_squeeze: false,
    dc_x_cfl: false,
    ans_refine: false,
    lossy_modular_auto: false,
    splines: false,
    tree_rank_deep: false,
    tree_dense_sampling: false,
    tree_full_sampling: false,
    tree_dual_layout: false,
    tree_ctx_v1: false,
    patches_default: false,
    ..SLOW
};

const SLOW: Effort = Effort {
    large_transform_rdoq: false,
    rdoq_fresh_prices: false,
    ans_exhaustive: false,
    lz_exhaustive_entropy: false,
    lz_priced_search: false,
    tree_wide_side_preds: false,
    ..Effort::ALL
};

const EXTRASLOW: Effort = Effort::ALL;

impl Speed {
    pub(crate) const fn effort(self) -> &'static Effort {
        match self {
            Speed::UltraFast => &ULTRAFAST,
            Speed::Fastest => &FASTEST,
            Speed::Fast => &FAST,
            Speed::Medium => &MEDIUM,
            Speed::Slow => &SLOW,
            Speed::ExtraSlow => &EXTRASLOW,
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
            Speed::Slow,
            Speed::ExtraSlow,
        ];
        for pair in tiers.windows(2) {
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
