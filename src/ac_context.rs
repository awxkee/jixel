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

pub(crate) const K_NON_ZERO_BUCKETS: usize = 37;
pub(crate) const K_ZERO_DENSITY_CONTEXT_COUNT: usize = 458;
pub(crate) const K_NUM_BLOCK_CTXS: usize = 4;
pub(crate) const K_NUM_AC_CONTEXTS: usize =
    K_NUM_BLOCK_CTXS * (K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT); // 1980

pub(crate) const K_NUM_AC_STRATEGY_CODES: usize = 27;

/// Frequency-band context per zigzag position. Index 0 is unused (DC).
pub(crate) static K_COEFF_FREQ_CONTEXT: [u16; 64] = [
    0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

/// Remaining-nonzeros context bucket.
pub(crate) static K_COEFF_NUM_NONZERO_CONTEXT: [u16; 64] = [
    0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206,
];

/// Static block context map signaled in WriteDCGlobal.
pub(crate) static K_COMPACT_BLOCK_CONTEXT_MAP: [u8; 39] = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // Y
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // X
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, // B
];

#[rustfmt::skip]
pub(crate) static K_BLOCK_CONTEXT_MAP: [u8; 81] = [
    // X row. Position 4 = DCT16X16 (decoder ctx 2). Positions 6, 7 = DCT16X8/8X16 (ctx 3).
    2, 0, 0, 0, 2, 0, 3, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Y row. Position 4 = DCT16X16 (decoder ctx 0). Positions 6, 7 = DCT16X8/8X16 (ctx 1).
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // B row. Position 4 = DCT16X16 (decoder ctx 2). Positions 6, 7 = DCT16X8/8X16 (ctx 3).
    2, 0, 0, 0, 2, 0, 3, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[inline]
pub(crate) fn block_context(c: usize, ac_strategy_code: u8) -> u32 {
    K_BLOCK_CONTEXT_MAP[c * K_NUM_AC_STRATEGY_CODES + ac_strategy_code as usize] as u32
}

#[inline]
pub(crate) fn non_zero_context(non_zeros: u32, block_ctx: u32) -> u32 {
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
pub(crate) fn zero_density_context_8x8(nonzeros_left: usize, k: usize, prev: usize) -> usize {
    (K_COEFF_NUM_NONZERO_CONTEXT[nonzeros_left] as usize + K_COEFF_FREQ_CONTEXT[k] as usize) * 2
        + prev
}

/// General `zero_density_context` from libjxl-tiny. For 8x8 use
/// `zero_density_context_8x8`. For multi-block transforms, `covered_blocks` is
/// the number of 8x8 sub-blocks (2 for DCT16X8/DCT8X16) and `log2_covered_blocks`
/// is `log2(covered_blocks)` (1 for the rectangular pair).
#[inline]
pub(crate) fn zero_density_context(
    nonzeros_left: usize,
    k: usize,
    covered_blocks: usize,
    log2_covered_blocks: usize,
    prev: usize,
) -> usize {
    let nz = (nonzeros_left + covered_blocks - 1) >> log2_covered_blocks;
    let kk = k >> log2_covered_blocks;
    (K_COEFF_NUM_NONZERO_CONTEXT[nz] as usize + K_COEFF_FREQ_CONTEXT[kk] as usize) * 2 + prev
}

#[inline]
pub(crate) const fn zero_density_contexts_offset(block_ctx: u32) -> u32 {
    K_NUM_BLOCK_CTXS as u32 * K_NON_ZERO_BUCKETS as u32
        + K_ZERO_DENSITY_CONTEXT_COUNT as u32 * block_ctx
}

/// 8x8 zigzag order. Coefficient at zigzag position k is at raw index
/// K_COEFF_ORDER_8X8[k].
#[rustfmt::skip]
pub(crate) static K_COEFF_ORDER_8X8: [u8; 64] = [
    0,   1,   8,   16, 9,   2,   3,   10,
    17,  24,  32,  25, 18,  11,  4,   5,
    12,  19,  26,  33, 40,  48,  41,  34,
    27,  20,  13,  6,  7,   14,  21,  28,
    35,  42,  49,  56, 57,  50,  43,  36,
    29,  22,  15,  23, 30,  37,  44,  51,
    58,  59,  52,  45, 38,  31,  39,  46,
    53,  60,  61,  54, 47,  55,  62,  63,
];

/// 16x8 / 8x16 coefficient order (shared, libjxl-tiny `kCoeffOrders` offset 64).
/// 128 entries. Positions 0 and 1 are LF positions; HF positions are zigzagged
/// over the 16x8 grid.
#[rustfmt::skip]
pub(crate) static K_COEFF_ORDER_16X8: [u8; 128] = [
    0,   1,   16,  2,   3,   17,  32,  18,  4,   5,   19,
    33,  48,  34,  20,  6,   7,   21,  35,  49,  64,  50,  36,  22,  8,   9,
    23,  37,  51,  65,  80,  66,  52,  38,  24,  10,  11,  25,  39,  53,  67,
    81,  96,  82,  68,  54,  40,  26,  12,  13,  27,  41,  55,  69,  83,  97,
    112, 98,  84,  70,  56,  42,  28,  14,  15,  29,  43,  57,  71,  85,  99,
    113, 114, 100, 86,  72,  58,  44,  30,  31,  45,  59,  73,  87,  101, 115,
    116, 102, 88,  74,  60,  46,  47,  61,  75,  89,  103, 117, 118, 104, 90,
    76,  62,  63,  77,  91,  105, 119, 120, 106, 92,  78,  79,  93,  107, 121,
    122, 108, 94,  95,  109, 123, 124, 110, 111, 125, 126, 127,
];

#[rustfmt::skip]
pub(crate) static  K_COEFF_ORDER_16X16: [u8; 256] = [
       0,    1,   16,   17,   32,    2,    3,   18,
      33,   48,   64,   49,   34,   19,    4,    5,
      20,   35,   50,   65,   80,   96,   81,   66,
      51,   36,   21,    6,    7,   22,   37,   52,
      67,   82,   97,  112,  128,  113,   98,   83,
      68,   53,   38,   23,    8,    9,   24,   39,
      54,   69,   84,   99,  114,  129,  144,  160,
     145,  130,  115,  100,   85,   70,   55,   40,
      25,   10,   11,   26,   41,   56,   71,   86,
     101,  116,  131,  146,  161,  176,  192,  177,
     162,  147,  132,  117,  102,   87,   72,   57,
      42,   27,   12,   13,   28,   43,   58,   73,
      88,  103,  118,  133,  148,  163,  178,  193,
     208,  224,  209,  194,  179,  164,  149,  134,
     119,  104,   89,   74,   59,   44,   29,   14,
      15,   30,   45,   60,   75,   90,  105,  120,
     135,  150,  165,  180,  195,  210,  225,  240,
     241,  226,  211,  196,  181,  166,  151,  136,
     121,  106,   91,   76,   61,   46,   31,   47,
      62,   77,   92,  107,  122,  137,  152,  167,
     182,  197,  212,  227,  242,  243,  228,  213,
     198,  183,  168,  153,  138,  123,  108,   93,
      78,   63,   79,   94,  109,  124,  139,  154,
     169,  184,  199,  214,  229,  244,  245,  230,
     215,  200,  185,  170,  155,  140,  125,  110,
      95,  111,  126,  141,  156,  171,  186,  201,
     216,  231,  246,  247,  232,  217,  202,  187,
     172,  157,  142,  127,  143,  158,  173,  188,
     203,  218,  233,  248,  249,  234,  219,  204,
     189,  174,  159,  175,  190,  205,  220,  235,
     250,  251,  236,  221,  206,  191,  207,  222,
     237,  252,  253,  238,  223,  239,  254,  255,
];
