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

pub const K_NON_ZERO_BUCKETS: usize = 37;
pub const K_ZERO_DENSITY_CONTEXT_COUNT: usize = 458;
pub const K_NUM_BLOCK_CTXS: usize = 4;
pub const K_NUM_AC_CONTEXTS: usize =
    K_NUM_BLOCK_CTXS * (K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT); // 1980

pub const K_NUM_AC_STRATEGY_CODES: usize = 27;

/// Frequency-band context per zigzag position. Index 0 is unused (DC).
pub const K_COEFF_FREQ_CONTEXT: [u16; 64] = [
    0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

/// Remaining-nonzeros context bucket.
pub const K_COEFF_NUM_NONZERO_CONTEXT: [u16; 64] = [
    0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206,
];

/// Static block context map signaled in WriteDCGlobal.
pub const K_COMPACT_BLOCK_CONTEXT_MAP: [u8; 39] = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Y
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // X
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // B
];

/// 3 channels x 27 strategy codes — only entry [c * 27 + 0] is ever read in
/// jixel since we only use AcStrategy::DCT (code 0).
#[rustfmt::skip]
pub const K_BLOCK_CONTEXT_MAP: [u8; 81] = [
    // X
    2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Y
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // B
    2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[inline]
pub fn block_context(c: usize, ac_strategy_code: u8) -> u32 {
    K_BLOCK_CONTEXT_MAP[c * K_NUM_AC_STRATEGY_CODES + ac_strategy_code as usize] as u32
}

#[inline]
pub fn non_zero_context(non_zeros: u32, block_ctx: u32) -> u32 {
    let bucket = if non_zeros < 8 {
        non_zeros
    } else if non_zeros >= 64 {
        36
    } else {
        4 + non_zeros / 2
    };
    bucket * K_NUM_BLOCK_CTXS as u32 + block_ctx
}

/// 8x8 specialization: covered_blocks = 1, log2_covered_blocks = 0, so
/// nonzeros_left and k pass through directly.
#[inline]
pub fn zero_density_context_8x8(nonzeros_left: usize, k: usize, prev: usize) -> usize {
    (K_COEFF_NUM_NONZERO_CONTEXT[nonzeros_left] as usize + K_COEFF_FREQ_CONTEXT[k] as usize) * 2
        + prev
}

#[inline]
pub const fn zero_density_contexts_offset(block_ctx: u32) -> u32 {
    K_NUM_BLOCK_CTXS as u32 * K_NON_ZERO_BUCKETS as u32
        + K_ZERO_DENSITY_CONTEXT_COUNT as u32 * block_ctx
}

/// 8x8 zigzag order. Coefficient at zigzag position k is at raw index
/// K_COEFF_ORDER_8X8[k].
#[rustfmt::skip]
pub const K_COEFF_ORDER_8X8: [u8; 64] = [
    0,   1,   8,   16, 9,   2,   3,   10,
    17,  24,  32,  25, 18,  11,  4,   5,
    12,  19,  26,  33, 40,  48,  41,  34,
    27,  20,  13,  6,  7,   14,  21,  28,
    35,  42,  49,  56, 57,  50,  43,  36,
    29,  22,  15,  23, 30,  37,  44,  51,
    58,  59,  52,  45, 38,  31,  39,  46,
    53,  60,  61,  54, 47,  55,  62,  63,
];
