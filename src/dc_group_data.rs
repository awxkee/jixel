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
use crate::image::{Image3S, ImageB, ImageSB};

/// AC-strategy "image" — in jixel, all blocks are DCT-8x8 (raw strategy 0)
/// so this just records the dimensions for iteration.
pub struct AcStrategyImage {
    xsize: usize,
    ysize: usize,
}

impl AcStrategyImage {
    pub fn new(xsize: usize, ysize: usize) -> Self {
        Self { xsize, ysize }
    }
    #[inline]
    pub fn xsize(&self) -> usize {
        self.xsize
    }
    #[inline]
    pub fn ysize(&self) -> usize {
        self.ysize
    }

    /// Every block is its own first block (no multi-block transforms).
    #[inline]
    pub fn is_first_block(&self, _x: usize, _y: usize) -> bool {
        true
    }
    /// Raw strategy = 0 (DCT-8x8) everywhere.
    #[inline]
    pub fn raw_strategy(&self, _x: usize, _y: usize) -> u8 {
        0
    }
    /// Strategy code for context lookup = 0 everywhere.
    #[inline]
    pub fn strategy_code(&self, _x: usize, _y: usize) -> u8 {
        0
    }
    #[inline]
    pub fn covered_blocks_x(&self) -> usize {
        1
    }
    #[inline]
    pub fn covered_blocks_y(&self) -> usize {
        1
    }
}

pub struct DcGroupData {
    pub quant_dc: Image3S,
    pub raw_quant_field: ImageB,
    pub ac_strategy: AcStrategyImage,
    pub ytox_map: ImageSB,
    pub ytob_map: ImageSB,
}

// kTileDimInBlocks = 8 (a 64x64 tile contains 8x8 blocks).
// kColorTileDim = kBlockDim * kTileDimInBlocks = 8 * 8 = 64.
// xtiles = ceil(xsize_blocks * 8 / 64) = ceil(xsize_blocks / 8).
const TILE_DIM_IN_BLOCKS: usize = 8;

impl DcGroupData {
    pub fn new(xsize_blocks: usize, ysize_blocks: usize) -> Self {
        let xtiles = (xsize_blocks + TILE_DIM_IN_BLOCKS - 1) / TILE_DIM_IN_BLOCKS;
        let ytiles = (ysize_blocks + TILE_DIM_IN_BLOCKS - 1) / TILE_DIM_IN_BLOCKS;
        Self {
            quant_dc: Image3S::new(xsize_blocks, ysize_blocks),
            raw_quant_field: ImageB::new_fill(xsize_blocks, ysize_blocks, 1),
            ac_strategy: AcStrategyImage::new(xsize_blocks, ysize_blocks),
            ytox_map: ImageSB::new_fill(xtiles, ytiles, 0),
            ytob_map: ImageSB::new_fill(xtiles, ytiles, 0),
        }
    }
}
