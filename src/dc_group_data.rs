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

pub(crate) const STRATEGY_DCT: u8 = 0;
pub(crate) const STRATEGY_DCT16X8: u8 = 1;
pub(crate) const STRATEGY_DCT8X16: u8 = 2;
pub(crate) const NUM_STRATEGIES: usize = 3;

/// Map raw strategy -> JXL HfTransformType code (= what the bitstream stores).
pub(crate) const STRATEGY_CODE_LUT: [u8; NUM_STRATEGIES] = [0, 6, 7];

const FIRST_BLOCK_BIT: u8 = 1;

pub(crate) struct AcStrategyImage {
    xsize: usize,
    ysize: usize,
    /// Row-major xsize × ysize. Each entry: (raw_strategy << 1) | is_first_bit.
    cells: Vec<u8>,
}

impl AcStrategyImage {
    pub(crate) fn new(xsize: usize, ysize: usize) -> Self {
        // Default: every block is its own DCT8 first block: (0 << 1) | 1 = 1.
        Self {
            xsize,
            ysize,
            cells: vec![(STRATEGY_DCT << 1) | FIRST_BLOCK_BIT; xsize * ysize],
        }
    }

    #[inline]
    pub(crate) fn xsize(&self) -> usize {
        self.xsize
    }
    #[inline]
    pub(crate) fn ysize(&self) -> usize {
        self.ysize
    }

    #[inline]
    pub(crate) fn is_first_block(&self, x: usize, y: usize) -> bool {
        self.cells[y * self.xsize + x] & FIRST_BLOCK_BIT != 0
    }
    #[inline]
    pub(crate) fn raw_strategy(&self, x: usize, y: usize) -> u8 {
        self.cells[y * self.xsize + x] >> 1
    }
    /// JXL HfTransformType code used in the bitstream.
    #[inline]
    pub(crate) fn strategy_code(&self, x: usize, y: usize) -> u8 {
        STRATEGY_CODE_LUT[self.raw_strategy(x, y) as usize]
    }

    #[inline]
    pub(crate) fn covered_blocks_x_of(strategy: u8) -> usize {
        // {DCT: 1, DCT16X8: 1, DCT8X16: 2}
        const LUT: [u8; NUM_STRATEGIES] = [1, 1, 2];
        LUT[strategy as usize] as usize
    }
    #[inline]
    pub(crate) fn covered_blocks_y_of(strategy: u8) -> usize {
        // {DCT: 1, DCT16X8: 2, DCT8X16: 1}
        const LUT: [u8; NUM_STRATEGIES] = [1, 2, 1];
        LUT[strategy as usize] as usize
    }
    #[inline]
    pub(crate) fn covered_blocks_x_at(&self, x: usize, y: usize) -> usize {
        Self::covered_blocks_x_of(self.raw_strategy(x, y))
    }
    #[inline]
    pub(crate) fn covered_blocks_y_at(&self, x: usize, y: usize) -> usize {
        Self::covered_blocks_y_of(self.raw_strategy(x, y))
    }

    /// Legacy 1×1 accessor (used by old code paths).
    #[inline]
    pub(crate) fn covered_blocks_x(&self) -> usize {
        1
    }
    #[inline]
    pub(crate) fn covered_blocks_y(&self) -> usize {
        1
    }

    /// Mark (x, y) as the first block of a multi-block transform with the
    /// given `strategy`. Fills covered cells with non-first markers.
    pub(crate) fn set_first(&mut self, x: usize, y: usize, strategy: u8) {
        let cx = Self::covered_blocks_x_of(strategy);
        let cy = Self::covered_blocks_y_of(strategy);
        assert!(
            x + cx <= self.xsize && y + cy <= self.ysize,
            "transform out of bounds: ({x},{y}) +{cx}x{cy} on {}x{}",
            self.xsize,
            self.ysize
        );
        self.cells[y * self.xsize + x] = (strategy << 1) | FIRST_BLOCK_BIT;
        for iy in 0..cy {
            for ix in 0..cx {
                if iy == 0 && ix == 0 {
                    continue;
                }
                self.cells[(y + iy) * self.xsize + (x + ix)] = strategy << 1;
            }
        }
    }

    /// True if a multi-block transform of `strategy` can be placed at (x, y).
    /// Checks image bounds and 32-block AC-group boundaries.
    pub(crate) fn can_place_strategy(&self, x: usize, y: usize, strategy: u8) -> bool {
        let cx = Self::covered_blocks_x_of(strategy);
        let cy = Self::covered_blocks_y_of(strategy);
        if x + cx > self.xsize || y + cy > self.ysize {
            return false;
        }
        const GROUP: usize = 32;
        if (x / GROUP) != ((x + cx - 1) / GROUP) {
            return false;
        }
        if (y / GROUP) != ((y + cy - 1) / GROUP) {
            return false;
        }
        true
    }

    /// Iterate first blocks in raster order, yielding (x, y, raw_strategy).
    pub(crate) fn iter_first_blocks(&self) -> impl Iterator<Item = (usize, usize, u8)> + '_ {
        let xs = self.xsize;
        let cells = &self.cells;
        (0..self.ysize).flat_map(move |y| {
            (0..xs).filter_map(move |x| {
                let cell = cells[y * xs + x];
                if cell & FIRST_BLOCK_BIT != 0 {
                    Some((x, y, cell >> 1))
                } else {
                    None
                }
            })
        })
    }

    /// Count first blocks (== modular channel[2] width).
    pub(crate) fn count_first_blocks(&self) -> usize {
        self.cells
            .iter()
            .filter(|&&c| c & FIRST_BLOCK_BIT != 0)
            .count()
    }
}

pub(crate) struct DcGroupData {
    pub(crate) quant_dc: Image3S,
    pub(crate) raw_quant_field: ImageB,
    pub(crate) ac_strategy: AcStrategyImage,
    pub(crate) ytox_map: ImageSB,
    pub(crate) ytob_map: ImageSB,
}

const TILE_DIM_IN_BLOCKS: usize = 8;

impl DcGroupData {
    pub(crate) fn new(xsize_blocks: usize, ysize_blocks: usize) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_dct8_first_blocks() {
        let a = AcStrategyImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                assert!(a.is_first_block(x, y));
                assert_eq!(a.raw_strategy(x, y), STRATEGY_DCT);
                assert_eq!(a.strategy_code(x, y), 0); // JXL HfTransformType for DCT
            }
        }
        assert_eq!(a.count_first_blocks(), 16);
    }

    #[test]
    fn dct16x8_covers_1x2_blocks() {
        // DCT16X8: 1 col × 2 rows of 8x8 blocks.
        let mut a = AcStrategyImage::new(4, 4);
        a.set_first(0, 0, STRATEGY_DCT16X8);
        assert!(a.is_first_block(0, 0));
        assert_eq!(a.raw_strategy(0, 0), STRATEGY_DCT16X8);
        assert_eq!(a.strategy_code(0, 0), 6); // JXL DCT16X8
        // Covered block at (0, 1) — same column, next row down.
        assert!(!a.is_first_block(0, 1));
        assert_eq!(a.raw_strategy(0, 1), STRATEGY_DCT16X8);
        // Other blocks still DCT8 first.
        assert!(a.is_first_block(1, 0));
        assert!(a.is_first_block(0, 2));
        // 16 default - 1 covered = 15 first blocks.
        assert_eq!(a.count_first_blocks(), 15);
    }

    #[test]
    fn dct8x16_covers_2x1_blocks() {
        // DCT8X16: 2 cols × 1 row of 8x8 blocks.
        let mut a = AcStrategyImage::new(4, 4);
        a.set_first(0, 0, STRATEGY_DCT8X16);
        assert!(a.is_first_block(0, 0));
        assert_eq!(a.strategy_code(0, 0), 7); // JXL DCT8X16
        assert!(!a.is_first_block(1, 0));
        assert_eq!(a.raw_strategy(1, 0), STRATEGY_DCT8X16);
        assert!(a.is_first_block(0, 1));
        assert_eq!(a.count_first_blocks(), 15);
    }

    #[test]
    fn can_place_respects_bounds_and_groups() {
        let a = AcStrategyImage::new(4, 4);
        assert!(a.can_place_strategy(0, 0, STRATEGY_DCT16X8)); // 1×2 fits
        assert!(a.can_place_strategy(3, 0, STRATEGY_DCT16X8)); // 1×2 fits at right edge (col 3)
        assert!(!a.can_place_strategy(0, 3, STRATEGY_DCT16X8)); // 1×2 doesn't fit at row 3
        assert!(a.can_place_strategy(0, 3, STRATEGY_DCT8X16)); // 2×1 fits at bottom edge
        assert!(!a.can_place_strategy(3, 0, STRATEGY_DCT8X16)); // 2×1 doesn't fit at col 3

        // Group boundary check (32-block AC groups):
        let big = AcStrategyImage::new(35, 35);
        assert!(!big.can_place_strategy(0, 31, STRATEGY_DCT16X8)); // 1×2 would straddle y=32
        assert!(big.can_place_strategy(0, 30, STRATEGY_DCT16X8)); // 1×2 stays within group 0
        assert!(!big.can_place_strategy(31, 0, STRATEGY_DCT8X16)); // 2×1 would straddle x=32
        assert!(big.can_place_strategy(30, 0, STRATEGY_DCT8X16)); // 2×1 stays within group 0
    }
}
