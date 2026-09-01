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
use crate::image::{Image3F, Image3S, ImageB, ImageSB};
use crate::util::{EncodeError, try_vec};

pub(crate) const STRATEGY_DCT: u8 = 0;
pub(crate) const STRATEGY_DCT16X8: u8 = 1;
pub(crate) const STRATEGY_DCT8X16: u8 = 2;
pub(crate) const STRATEGY_DCT16X16: u8 = 3;
pub(crate) const STRATEGY_DCT32X32: u8 = 4;
pub(crate) const STRATEGY_DCT4X4: u8 = 5;
pub(crate) const STRATEGY_DCT4X8: u8 = 6;
pub(crate) const STRATEGY_DCT8X4: u8 = 7;
pub(crate) const STRATEGY_DCT32X16: u8 = 8;
pub(crate) const STRATEGY_DCT16X32: u8 = 9;
pub(crate) const STRATEGY_AFV0: u8 = 10;
pub(crate) const STRATEGY_AFV1: u8 = 11;
pub(crate) const STRATEGY_AFV2: u8 = 12;
pub(crate) const STRATEGY_AFV3: u8 = 13;
pub(crate) const STRATEGY_DCT64X64: u8 = 14;
pub(crate) const STRATEGY_DCT64X32: u8 = 15;
pub(crate) const STRATEGY_DCT32X64: u8 = 16;
pub(crate) const STRATEGY_IDENTITY: u8 = 17;
pub(crate) const STRATEGY_DCT2X2: u8 = 18;
pub(crate) const NUM_STRATEGIES: usize = 19;

#[inline]
pub(crate) fn is_sub8_strategy(strategy: u8) -> bool {
    matches!(
        strategy,
        STRATEGY_IDENTITY
            | STRATEGY_DCT2X2
            | STRATEGY_DCT4X4
            | STRATEGY_DCT4X8
            | STRATEGY_DCT8X4
            | STRATEGY_AFV0
            | STRATEGY_AFV1
            | STRATEGY_AFV2
            | STRATEGY_AFV3
    )
}

/// Map raw strategy -> JXL HfTransformType code (= what the bitstream stores).
/// DCT8=0, DCT16X16=4, DCT32X32=5, DCT16X8=6, DCT8X16=7, DCT4X4=3, DCT4X8=12, DCT8X4=13,
/// DCT32X16=10, DCT16X32=11, AFV0..3=14..17, DCT64 family=18..20,
/// IDENTITY=1, DCT2X2=2.
pub(crate) static STRATEGY_CODE_LUT: [u8; NUM_STRATEGIES] = [
    0, 6, 7, 4, 5, 3, 12, 13, 10, 11, 14, 15, 16, 17, 18, 19, 20, 1, 2,
];

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

    pub(crate) fn try_new(xsize: usize, ysize: usize) -> Result<Self, EncodeError> {
        let len = xsize
            .checked_mul(ysize)
            .ok_or(EncodeError::AllocationFailed { size: usize::MAX })?;
        Ok(Self {
            xsize,
            ysize,
            cells: try_vec![(STRATEGY_DCT << 1) | FIRST_BLOCK_BIT; len]?,
        })
    }

    #[inline]
    pub(crate) fn xsize(&self) -> usize {
        self.xsize
    }
    #[inline]
    pub(crate) fn ysize(&self) -> usize {
        self.ysize
    }

    /// Restore the default all-DCT8 layout while retaining the cell allocation.
    pub(crate) fn reset(&mut self) {
        self.cells.fill((STRATEGY_DCT << 1) | FIRST_BLOCK_BIT);
    }

    /// Copy block rows `y0..y1` from `src` into `self` (both must share
    /// dimensions). Used to merge per-band selection results computed on
    /// independent worker threads back into the group's strategy image.
    pub(crate) fn copy_rows_from(&mut self, src: &AcStrategyImage, y0: usize, y1: usize) {
        debug_assert_eq!(self.xsize, src.xsize);
        let a = y0 * self.xsize;
        let b = y1 * self.xsize;
        self.cells[a..b].copy_from_slice(&src.cells[a..b]);
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
        // {DCT: 1, DCT16X8: 1, DCT8X16: 2, DCT16X16: 2, DCT32X32: 4, DCT4X4: 1,
        //  DCT4X8: 1, DCT8X4: 1, DCT32X16: 2, DCT16X32: 4, AFV0..3: 1}
        static LUT: [u8; NUM_STRATEGIES] =
            [1, 1, 2, 2, 4, 1, 1, 1, 2, 4, 1, 1, 1, 1, 8, 4, 8, 1, 1];
        LUT[strategy as usize] as usize
    }
    #[inline]
    pub(crate) fn covered_blocks_y_of(strategy: u8) -> usize {
        // {DCT: 1, DCT16X8: 2, DCT8X16: 1, DCT16X16: 2, DCT32X32: 4, DCT4X4: 1,
        //  DCT4X8: 1, DCT8X4: 1, DCT32X16: 4, DCT16X32: 2, AFV0..3: 1}
        static LUT: [u8; NUM_STRATEGIES] =
            [1, 2, 1, 2, 4, 1, 1, 1, 4, 2, 1, 1, 1, 1, 8, 8, 4, 1, 1];
        LUT[strategy as usize] as usize
    }

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
        // A multi-block transform must also stay within a single 8-block
        // (64 px) tile, since `write_ac_group` processes one tile-tall stripe
        // at a time. For 1×/2×-block transforms this is implied by the 2-block
        // alignment of selection; for the 4×4 DCT32X32 it is the binding
        // constraint.
        const TILE: usize = 8;
        if (x / TILE) != ((x + cx - 1) / TILE) {
            return false;
        }
        if (y / TILE) != ((y + cy - 1) / TILE) {
            return false;
        }
        true
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
        let v = strategy << 1;

        let mut rows = self.cells.chunks_exact_mut(self.xsize).skip(y).take(cy);

        if let Some(row) = rows.next() {
            row[x + 1..x + cx].fill(v);
        }

        for row in rows {
            row[x..x + cx].fill(v);
        }
    }

    /// True if a multi-block transform of `strategy` can be placed at (x, y).
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
    /// Unquantized DC targets in XYB (X and B are true DC, pre-CfL-slope):
    /// what the decoder's smoothed DC recon is compared against. Empty (0×0)
    /// unless the DC-smoothing rounding pass is going to run — the VarDCT
    /// path sizes it after construction; capture sites check for emptiness.
    pub(crate) dc_float: Image3F,
    pub(crate) raw_quant_field: ImageB,
    pub(crate) ac_strategy: AcStrategyImage,
    pub(crate) ytox_map: ImageSB,
    pub(crate) ytob_map: ImageSB,
    /// Accumulated RD benefit of all sub-8x8 strategies from
    /// `fill_ac_strategy`, used by the frame-level activation gate. 0 until
    /// selection runs.
    pub(crate) sub8_benefit: f32,
}

const TILE_DIM_IN_BLOCKS: usize = 8;

impl DcGroupData {
    pub(crate) fn new(xsize_blocks: usize, ysize_blocks: usize) -> Result<Self, EncodeError> {
        let xtiles = xsize_blocks.div_ceil(TILE_DIM_IN_BLOCKS);
        let ytiles = ysize_blocks.div_ceil(TILE_DIM_IN_BLOCKS);
        Ok(Self {
            quant_dc: Image3S::try_new(xsize_blocks, ysize_blocks)?,
            dc_float: Image3F::new(0, 0),
            raw_quant_field: ImageB::try_new_fill(xsize_blocks, ysize_blocks, 1)?,
            ac_strategy: AcStrategyImage::try_new(xsize_blocks, ysize_blocks)?,
            ytox_map: ImageSB::try_new_fill(xtiles, ytiles, 0)?,
            ytob_map: ImageSB::try_new_fill(xtiles, ytiles, 0)?,
            sub8_benefit: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_group_allocation_failure_is_reported() {
        assert!(matches!(
            DcGroupData::new(usize::MAX, 1),
            Err(EncodeError::AllocationFailed { size: usize::MAX })
        ));
    }

    #[test]
    fn sub8_strategy_set_is_complete() {
        for strategy in 0..NUM_STRATEGIES as u8 {
            assert_eq!(
                is_sub8_strategy(strategy),
                matches!(
                    strategy,
                    STRATEGY_IDENTITY
                        | STRATEGY_DCT2X2
                        | STRATEGY_DCT4X4
                        | STRATEGY_DCT4X8
                        | STRATEGY_DCT8X4
                        | STRATEGY_AFV0
                        | STRATEGY_AFV1
                        | STRATEGY_AFV2
                        | STRATEGY_AFV3
                )
            );
        }
    }

    #[test]
    fn fine_strategies_use_the_spec_wire_codes() {
        assert_eq!(STRATEGY_CODE_LUT[STRATEGY_IDENTITY as usize], 1);
        assert_eq!(STRATEGY_CODE_LUT[STRATEGY_DCT2X2 as usize], 2);
        assert_eq!(AcStrategyImage::covered_blocks_x_of(STRATEGY_IDENTITY), 1);
        assert_eq!(AcStrategyImage::covered_blocks_y_of(STRATEGY_IDENTITY), 1);
        assert_eq!(AcStrategyImage::covered_blocks_x_of(STRATEGY_DCT2X2), 1);
        assert_eq!(AcStrategyImage::covered_blocks_y_of(STRATEGY_DCT2X2), 1);
    }

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
