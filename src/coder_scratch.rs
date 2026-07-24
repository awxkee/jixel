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

use crate::adaptive_quant::AqMapScratch;
use crate::enc_group::AcGroupScratch;
use crate::enc_lossless::{GradientScratch, LzToken, PickThresholdScratch};
use crate::entropy::{
    CLUSTERS_LIMIT, FixedClusterScratch, Histogram, HuffmanNode, HybridUintConfig, PrefixCode,
};
use crate::util::{HeapMatrix, heap_array};

const LZ77_CANDIDATE_CAPACITY: usize = 1 << 19;
pub(crate) const LZ77_MAX_CONTEXTS: usize = 221;

pub(crate) struct LzEntropyScratch {
    pub(crate) histograms: Box<[Histogram; LZ77_MAX_CONTEXTS]>,
    pub(crate) prefix_codes: Box<[PrefixCode; LZ77_MAX_CONTEXTS]>,
    pub(crate) context_map: Box<[u8; LZ77_MAX_CONTEXTS]>,
    pub(crate) configs: Box<[HybridUintConfig; CLUSTERS_LIMIT]>,
    pub(crate) clustering: FixedClusterScratch<LZ77_MAX_CONTEXTS>,
}

impl Default for LzEntropyScratch {
    fn default() -> Self {
        Self {
            histograms: heap_array(Histogram::new()),
            prefix_codes: heap_array(PrefixCode::zero()),
            context_map: heap_array(0),
            configs: heap_array(HybridUintConfig::DEFAULT),
            clustering: FixedClusterScratch::default(),
        }
    }
}

/// Fixed-layout reusable storage owned by one encoder worker.
///
/// The box containing this structure is created once with the worker. Hot paths
/// then borrow a concrete field directly: there is no allocation, lookup, or
/// type erasure involved in selecting a scratch slot.
pub(crate) struct CoderScratch {
    pub(crate) aq_map: AqMapScratch,
    pub(crate) structure_corrections: Vec<f32>,
    pub(crate) lz_repetitions: Vec<u32>,
    pub(crate) lz_depth: Vec<u32>,
    pub(crate) lz_candidate: Vec<LzToken>,
    pub(crate) lz_entropy: LzEntropyScratch,
    pub(crate) recon: HeapMatrix<f32, 8, 1024>,
    pub(crate) dark_octile: Vec<f32>,
    pub(crate) huffman_pool: Vec<HuffmanNode>,
    pub(crate) ac_group: AcGroupScratch,
    pub(crate) transform_gather: Box<[f32; 1024]>,
    pub(crate) strategy_coeffs: HeapMatrix<f32, 3, 1024>,
    pub(crate) gradient: GradientScratch,
    pub(crate) order0_entropy: Vec<u64>,
    pub(crate) threshold: PickThresholdScratch,
}

impl Default for CoderScratch {
    fn default() -> Self {
        Self {
            aq_map: AqMapScratch {
                aq_map: vec![0.0; 256 * 256],
                secondary: vec![0.0; 2048 + 512 * 512],
            },
            structure_corrections: vec![0.0; 256 * 256],
            lz_repetitions: vec![0; 1 << 14],
            lz_depth: vec![u32::MAX; 1 << 20],
            lz_candidate: Vec::with_capacity(LZ77_CANDIDATE_CAPACITY),
            lz_entropy: LzEntropyScratch::default(),
            recon: HeapMatrix::new(0.0),
            dark_octile: vec![0.0; 32 * 32],
            huffman_pool: Vec::with_capacity(1024),
            ac_group: AcGroupScratch::default(),
            transform_gather: heap_array(0.0),
            strategy_coeffs: HeapMatrix::new(0.0),
            gradient: GradientScratch {
                cur: vec![0; 256],
                prev: vec![0; 256],
                buf: vec![0; 256],
            },
            order0_entropy: vec![0; 1024],
            threshold: PickThresholdScratch {
                hist_scratch: vec![0; 3 * 1025],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CoderScratch, LzEntropyScratch};
    use crate::enc_group::AcGroupScratch;
    use std::mem::size_of;

    #[test]
    fn scratch_structs_are_only_small_heap_handles() {
        assert!(size_of::<CoderScratch>() <= 1024);
        assert!(size_of::<LzEntropyScratch>() <= 128);
        assert!(size_of::<AcGroupScratch>() <= 128);
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
