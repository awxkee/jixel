/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::ac_strategy::{Chosen32Cost, SavedChild};
use crate::adaptive_quant::AqMapScratch;
use crate::dc_group_data::AcStrategyImage;
use crate::entropy::{
    ALPHABET_SIZE, CLUSTERS_LIMIT, FixedClusterScratch, Histogram, HuffmanNode, HybridUintConfig,
    PrefixCode, Token,
};
use crate::group::AcGroupScratch;
use crate::lossless::{GradientScratch, LzToken, PickThresholdScratch};
use crate::ma_tree::MaPropertyScratch;
use crate::patches::PATCH_TILE;
use crate::static_entropy_codes::K_NUM_DC_CONTEXTS;
use crate::util::{HeapMatrix, heap_array};
use std::ops::{Deref, DerefMut};

pub(crate) const LZ77_MAX_CONTEXTS: usize = 1024;
const DC_PREDICTOR_SLOTS: usize = 2 * K_NUM_DC_CONTEXTS;

pub(crate) struct DcPredictorScratch {
    pub(crate) counts: Box<[[u32; ALPHABET_SIZE]; DC_PREDICTOR_SLOTS]>,
    pub(crate) extra: Box<[u64; DC_PREDICTOR_SLOTS]>,
}

impl Default for DcPredictorScratch {
    fn default() -> Self {
        Self {
            counts: heap_array([0; ALPHABET_SIZE]),
            extra: heap_array(0),
        }
    }
}

pub(crate) struct LzEntropyScratch {
    pub(crate) histograms: Box<[Histogram; LZ77_MAX_CONTEXTS]>,
    pub(crate) prefix_codes: Box<[PrefixCode; LZ77_MAX_CONTEXTS]>,
    pub(crate) context_map: Box<[u8; LZ77_MAX_CONTEXTS]>,
    pub(crate) configs: Box<[HybridUintConfig; CLUSTERS_LIMIT]>,
    pub(crate) clustering: FixedClusterScratch<LZ77_MAX_CONTEXTS>,
    /// rANS tables for the pixel code (Slow lossless); empty on the prefix path.
    pub(crate) ans: Box<LzAnsScratch>,
}

#[derive(Default)]
pub(crate) struct LzAnsScratch {
    pub(crate) histograms: Vec<crate::entropy::AnsHistogram>,
    pub(crate) symbols: Vec<Vec<crate::entropy::AnsEncSymbolInfo>>,
    pub(crate) reverse_maps: Vec<u16>,
}

impl Default for LzEntropyScratch {
    fn default() -> Self {
        Self {
            histograms: heap_array(Histogram::new()),
            prefix_codes: heap_array(PrefixCode::zero()),
            context_map: heap_array(0),
            configs: heap_array(HybridUintConfig::DEFAULT),
            clustering: FixedClusterScratch::default(),
            ans: Box::default(),
        }
    }
}

/// Defers fixed-size worker storage until a pipeline actually uses it.
pub(crate) struct LazyScratch<T> {
    value: Option<T>,
    init: fn() -> T,
}

impl<T> LazyScratch<T> {
    fn new(init: fn() -> T) -> Self {
        Self { value: None, init }
    }
}

impl<T: Default> Default for LazyScratch<T> {
    fn default() -> Self {
        Self::new(T::default)
    }
}

impl<T> Deref for LazyScratch<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
            .as_ref()
            .expect("lazy scratch must be initialized through mutable access")
    }
}

impl<T> DerefMut for LazyScratch<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.get_or_insert_with(self.init)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QuantRefinement {
    pub(crate) bx: usize,
    pub(crate) by: usize,
    pub(crate) cov_x: usize,
    pub(crate) cov_y: usize,
    pub(crate) q: u8,
}

#[derive(Clone, Copy)]
pub(crate) struct CachedQuantCost {
    pub(crate) bx: usize,
    pub(crate) by: usize,
    pub(crate) cost: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct RerankDowngrade {
    pub(crate) bx: usize,
    pub(crate) by: usize,
    pub(crate) cov_x: usize,
    pub(crate) cov_y: usize,
    /// Layout to install over the footprint: strategy per (iy*4 + ix) offset,
    /// 0xFF = covered by a preceding first block. The all-DCT8 fallback is a
    /// grid of STRATEGY_DCT at every offset.
    pub(crate) restore: [u8; 16],
}

/// A large transform that was split only because an Identity/DCT2x2 mosaic
/// won the joint boundary-aware rerank. The frame-level metadata gate can
/// restore `strategy` exactly when the full fine-transform map does not repay
/// its measured entropy cost.
#[derive(Clone, Copy)]
pub(crate) struct FineMergeRollback {
    pub(crate) bx: usize,
    pub(crate) by: usize,
    pub(crate) cov_x: usize,
    pub(crate) cov_y: usize,
    pub(crate) strategy: u8,
    pub(crate) fine_grid: [u8; 16],
    pub(crate) benefit: f32,
}

pub(crate) struct AcStrategyBandScratch {
    pub(crate) strategy: AcStrategyImage,
    pub(crate) benefit: f32,
    pub(crate) chosen32: Vec<Chosen32Cost>,
    pub(crate) saved_children: Vec<SavedChild>,
    pub(crate) rerank_downgrades: Vec<RerankDowngrade>,
    pub(crate) fine_rollbacks: Vec<FineMergeRollback>,
    pub(crate) current_costs: Vec<CachedQuantCost>,
    pub(crate) quant_refinements: Vec<QuantRefinement>,
}

impl Default for AcStrategyBandScratch {
    fn default() -> Self {
        Self {
            strategy: AcStrategyImage::new(0, 0),
            benefit: 0.0,
            chosen32: Vec::new(),
            saved_children: Vec::new(),
            rerank_downgrades: Vec::new(),
            fine_rollbacks: Vec::new(),
            current_costs: Vec::new(),
            quant_refinements: Vec::new(),
        }
    }
}

impl AcStrategyBandScratch {
    fn prepare_selection(&mut self, xsize: usize, ysize: usize) {
        if self.strategy.xsize() == xsize && self.strategy.ysize() == ysize {
            self.strategy.reset();
        } else {
            self.strategy = AcStrategyImage::new(xsize, ysize);
        }
        self.benefit = 0.0;
    }

    fn clear(&mut self) {
        self.benefit = 0.0;
        self.chosen32.clear();
        self.saved_children.clear();
        self.rerank_downgrades.clear();
        self.fine_rollbacks.clear();
        self.current_costs.clear();
        self.quant_refinements.clear();
    }

    fn prepare_rerank(&mut self, max_blocks: usize) {
        self.rerank_downgrades.clear();
        self.fine_rollbacks.clear();
        self.current_costs.clear();
        if self.rerank_downgrades.capacity() < max_blocks {
            self.rerank_downgrades.reserve(max_blocks);
        }
        if self.current_costs.capacity() < max_blocks {
            self.current_costs.reserve(max_blocks);
        }
    }

    fn prepare_refinement(&mut self, max_blocks: usize) {
        self.quant_refinements.clear();
        if self.quant_refinements.capacity() < max_blocks {
            self.quant_refinements.reserve(max_blocks);
        }
    }
}

#[derive(Default)]
pub(crate) struct AcStrategyPipelineScratch {
    pub(crate) bands: Vec<(usize, usize)>,
    pub(crate) band_scratch: Vec<AcStrategyBandScratch>,
    pub(crate) current_costs: Vec<f32>,
    /// Full-image grid of committed 32x32-quadrant selection costs (indexed
    /// `(by/4)*qx + bx/4`), NaN where the quadrant never went through the
    /// 32-level selection path.
    pub(crate) chosen32_grid: Vec<f32>,
}

impl AcStrategyPipelineScratch {
    pub(crate) fn prepare_bands(&mut self, xsize: usize, ysize: usize, parallel_selection: bool) {
        if self.band_scratch.len() < self.bands.len() {
            self.band_scratch
                .resize_with(self.bands.len(), AcStrategyBandScratch::default);
        }
        for band in &mut self.band_scratch[..self.bands.len()] {
            band.clear();
            if parallel_selection {
                band.prepare_selection(xsize, ysize);
            }
        }
    }

    pub(crate) fn prepare_rerank(&mut self, xsize: usize, ysize: usize) {
        for (band, &(y0, y1)) in self.band_scratch.iter_mut().zip(&self.bands) {
            band.prepare_rerank(xsize * (y1 - y0));
        }
        let blocks = xsize * ysize;
        if self.current_costs.len() != blocks {
            self.current_costs.resize(blocks, f32::NAN);
        }
    }

    pub(crate) fn prepare_refinement(&mut self, xsize: usize) {
        for (band, &(y0, y1)) in self.band_scratch.iter_mut().zip(&self.bands) {
            band.prepare_refinement(xsize * (y1 - y0));
        }
    }
}

/// Fixed-layout reusable storage owned by one encoder worker.
pub(crate) struct CoderScratch {
    pub(crate) aq_map: AqMapScratch,
    pub(crate) structure_corrections: Vec<f32>,
    pub(crate) lz_repetitions: Vec<u32>,
    pub(crate) lz_depth: Vec<u32>,
    pub(crate) lz_candidate: Vec<LzToken>,
    /// Roughly 1 MiB of fixed entropy tables; Fast group workers never use it.
    pub(crate) lz_entropy: LazyScratch<LzEntropyScratch>,
    pub(crate) recon: LazyScratch<HeapMatrix<f32, 8, 1024>>,
    pub(crate) dark_octile: Vec<f32>,
    pub(crate) huffman_pool: Vec<HuffmanNode>,
    pub(crate) alpha_tokens: Vec<Token>,
    pub(crate) ac_group: LazyScratch<AcGroupScratch>,
    pub(crate) transform_gather: LazyScratch<Box<[f32; 4096]>>,
    pub(crate) strategy_coeffs: LazyScratch<HeapMatrix<f32, 3, 4096>>,
    pub(crate) gradient: GradientScratch,
    pub(crate) order0_entropy: Vec<u64>,
    pub(crate) threshold: PickThresholdScratch,
    pub(crate) ma_property: MaPropertyScratch,
    pub(crate) dct8_costs: Vec<f32>,
    pub(crate) ac_strategy: AcStrategyPipelineScratch,
    pub(crate) dc_cfl_cur: Vec<i32>,
    pub(crate) dc_cfl_prev: Vec<i32>,
    pub(crate) dc_predictor: LazyScratch<DcPredictorScratch>,
    pub(crate) patch_tile_colors: LazyScratch<Box<[[i32; 3]; PATCH_TILE * PATCH_TILE]>>,
    /// ~63KB of CfL RDO coefficient staging; only Slow lossy workers use it.
    pub(crate) cfl_rdo: LazyScratch<crate::color_correlation::CflRdoScratch>,
}

impl CoderScratch {
    fn new(reserve_lossy_buffers: bool) -> Self {
        let (aq_map, structure_corrections, dark_octile, gradient, order0_entropy, threshold) =
            if reserve_lossy_buffers {
                (
                    AqMapScratch {
                        aq_map: vec![0.0; 256 * 256],
                        secondary: vec![0.0; 2048 + 512 * 512],
                    },
                    vec![0.0; 256 * 256],
                    vec![0.0; 32 * 32],
                    GradientScratch {
                        cur: vec![0; 256],
                        prev: vec![0; 256],
                        prev_prev: vec![0; 256],
                        buf: vec![0; 256],
                    },
                    vec![0; 1024],
                    PickThresholdScratch {
                        hist_scratch: vec![0; 3 * 1025],
                    },
                )
            } else {
                (
                    AqMapScratch::default(),
                    Vec::new(),
                    Vec::new(),
                    GradientScratch::default(),
                    Vec::new(),
                    PickThresholdScratch::default(),
                )
            };

        Self {
            aq_map,
            structure_corrections,
            // Fast never enters the deep LZ path. Slow grows these buffers on
            // first use, and subsequent groups reuse the allocation.
            lz_repetitions: Vec::new(),
            lz_depth: Vec::new(),
            lz_candidate: Vec::new(),
            lz_entropy: LazyScratch::default(),
            recon: LazyScratch::new(|| HeapMatrix::new(0.0)),
            dark_octile,
            huffman_pool: Vec::with_capacity(1024),
            alpha_tokens: Vec::new(),
            ac_group: LazyScratch::default(),
            transform_gather: LazyScratch::new(|| heap_array(0.0)),
            strategy_coeffs: LazyScratch::new(|| HeapMatrix::new(0.0)),
            gradient,
            order0_entropy,
            threshold,
            ma_property: MaPropertyScratch::default(),
            dct8_costs: Vec::new(),
            ac_strategy: AcStrategyPipelineScratch::default(),
            dc_cfl_cur: Vec::new(),
            dc_cfl_prev: Vec::new(),
            dc_predictor: LazyScratch::default(),
            patch_tile_colors: LazyScratch::new(|| heap_array([0; 3])),
            cfl_rdo: LazyScratch::default(),
        }
    }

    pub(crate) fn lossless() -> Self {
        Self::new(false)
    }
}

impl Default for CoderScratch {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{CoderScratch, DcPredictorScratch, LazyScratch, LzEntropyScratch};
    use crate::group::AcGroupScratch;
    use std::mem::size_of;

    #[test]
    fn scratch_structs_are_only_small_heap_handles() {
        assert!(size_of::<CoderScratch>() <= 1104);
        assert!(size_of::<DcPredictorScratch>() <= 32);
        assert!(size_of::<LzEntropyScratch>() <= 128);
        assert!(size_of::<LazyScratch<LzEntropyScratch>>() <= 128);
        assert!(size_of::<AcGroupScratch>() <= 128);
    }

    #[test]
    fn slow_lz_buffers_are_not_reserved_by_default() {
        let scratch = CoderScratch::default();
        assert_eq!(scratch.lz_repetitions.capacity(), 0);
        assert_eq!(scratch.lz_depth.capacity(), 0);
        assert_eq!(scratch.lz_candidate.capacity(), 0);
    }

    #[test]
    fn lossless_scratch_does_not_reserve_lossy_buffers() {
        let scratch = CoderScratch::lossless();
        assert_eq!(scratch.aq_map.aq_map.capacity(), 0);
        assert_eq!(scratch.aq_map.secondary.capacity(), 0);
        assert_eq!(scratch.structure_corrections.capacity(), 0);
        assert_eq!(scratch.dark_octile.capacity(), 0);
        assert_eq!(scratch.gradient.cur.capacity(), 0);
        assert_eq!(scratch.gradient.prev.capacity(), 0);
        assert_eq!(scratch.gradient.prev_prev.capacity(), 0);
        assert_eq!(scratch.gradient.buf.capacity(), 0);
        assert_eq!(scratch.order0_entropy.capacity(), 0);
        assert_eq!(scratch.threshold.hist_scratch.capacity(), 0);
        assert!(scratch.lz_entropy.value.is_none());
        assert!(scratch.recon.value.is_none());
        assert!(scratch.ac_group.value.is_none());
        assert!(scratch.transform_gather.value.is_none());
        assert!(scratch.strategy_coeffs.value.is_none());
        assert!(scratch.dc_predictor.value.is_none());
        assert!(scratch.patch_tile_colors.value.is_none());
    }

    #[test]
    fn strategy_scratch_covers_dct64_and_reuses_allocations() {
        let mut scratch = CoderScratch::default();
        let gather = scratch.transform_gather.as_mut_ptr();
        let coeffs = scratch.strategy_coeffs[0].as_mut_ptr();
        assert_eq!(scratch.transform_gather.len(), 4096);
        assert_eq!(scratch.strategy_coeffs[0].len(), 4096);
        assert_eq!(scratch.transform_gather.as_ptr(), gather);
        assert_eq!(scratch.strategy_coeffs[0].as_ptr(), coeffs);
    }

    #[test]
    fn coder_scratch_constructs_on_a_small_stack() {
        std::thread::Builder::new()
            .name("small-stack-scratch-test".into())
            .stack_size(64 * 1024)
            .spawn(|| drop(CoderScratch::default()))
            .unwrap()
            .join()
            .unwrap();
    }
}
