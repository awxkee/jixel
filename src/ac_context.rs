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
use crate::entropy::f_log2;

pub(crate) const K_NON_ZERO_BUCKETS: usize = 37;
pub(crate) const K_ZERO_DENSITY_CONTEXT_COUNT: usize = 458;
pub(crate) const K_NUM_BLOCK_CTXS: usize = 15;
pub(crate) const K_NUM_AC_CONTEXTS: usize =
    K_NUM_BLOCK_CTXS * (K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT); // 15 * 495

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

/// Static block context map signaled in WriteDCGlobal, indexed by
/// `channel_row * K_NUM_ORDERS + strategy_order`. This is libjxl's default map:
/// transform *classes* get their own contexts rather than being collapsed into
/// "small" and "large" buckets. jixel previously used a 4-context reduction
/// inherited from libjxl-tiny, which lumped DCT32X32 with DCT8X8 and every
/// larger shape together (see docs/dct64-findings.md).
#[rustfmt::skip]
pub(crate) static K_COMPACT_BLOCK_CONTEXT_MAP: [u8; 39] = [
    // Y
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6,
    // X
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
    // B
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
];

/// Number of transform *orders* (shape classes) the block context map is keyed
/// by; libjxl `kNumOrders`.
pub(crate) const K_NUM_ORDERS: usize = 13;

/// Strategy code -> transform order, libjxl `kStrategyOrder`. Codes are the
/// wire values (0 = DCT8X8, 4 = DCT16X16, 5 = DCT32X32, 18..20 = the 64px
/// family), so this covers the whole JPEG XL strategy space, not only the
/// subset jixel emits.
#[rustfmt::skip]
pub(crate) static K_STRATEGY_ORDER: [u8; K_NUM_AC_STRATEGY_CODES] = [
    0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1,
    1, 1, 1, 1, 7, 8, 8, 9, 9, 9, 10, 10, 10,
];

/// Local channel order is X, Y, B; the block context map's row order is
/// Y, X, B.
static CHANNEL_ROW: [usize; 3] = [1, 0, 2];

/// The block-context map actually signaled in DC global.
#[inline]
pub(crate) fn compact_block_context_map() -> &'static [u8; 39] {
    &K_COMPACT_BLOCK_CONTEXT_MAP
}

/// Derived from the signaled map so the encoder and decoder cannot disagree —
/// the previous code-indexed duplicate silently defaulted unsupported entries
/// to context 0.
#[inline]
pub(crate) fn block_context(c: usize, ac_strategy_code: u8) -> u32 {
    let order = K_STRATEGY_ORDER[ac_strategy_code as usize] as usize;
    K_COMPACT_BLOCK_CONTEXT_MAP[CHANNEL_ROW[c] * K_NUM_ORDERS + order] as u32
}

#[inline]
fn non_zero_bucket(non_zeros: u32) -> u32 {
    if non_zeros < 8 {
        non_zeros
    } else if non_zeros >= 64 {
        36
    } else {
        4 + non_zeros / 2
    }
}

#[inline]
pub(crate) fn non_zero_context(non_zeros: u32, block_ctx: u32) -> u32 {
    non_zero_bucket(non_zeros) * K_NUM_BLOCK_CTXS as u32 + block_ctx
}

/// Fine tokenization layout for the lossy path: every base block context is
/// split by a per-image quant-field threshold into two classes. The final
/// signaled map is decided *after* tokenization from real token statistics
/// (splits kept only where they pay, within the spec's 16-context budget) and
/// tokens are remapped; this layout only has to be a superset of every final
/// map. The JPEG-recompression path stays on the legacy 15-context layout.
pub(crate) const K_NUM_QF_CLASSES: usize = 2;
pub(crate) const K_NUM_FINE_BLOCK_CTXS: usize = K_NUM_BLOCK_CTXS * K_NUM_QF_CLASSES;
pub(crate) const K_NUM_FINE_AC_CONTEXTS: usize =
    K_NUM_FINE_BLOCK_CTXS * (K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT);

#[inline]
pub(crate) fn fine_block_context(c: usize, ac_strategy_code: u8, qf_hi: bool) -> u32 {
    block_context(c, ac_strategy_code) * K_NUM_QF_CLASSES as u32 + qf_hi as u32
}

#[inline]
pub(crate) fn fine_non_zero_context(non_zeros: u32, fine_block_ctx: u32) -> u32 {
    non_zero_bucket(non_zeros) * K_NUM_FINE_BLOCK_CTXS as u32 + fine_block_ctx
}

#[inline]
pub(crate) const fn fine_zero_density_contexts_offset(fine_block_ctx: u32) -> u32 {
    K_NUM_FINE_BLOCK_CTXS as u32 * K_NON_ZERO_BUCKETS as u32
        + K_ZERO_DENSITY_CONTEXT_COUNT as u32 * fine_block_ctx
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

/// Natural (zig-zag) coefficient order for DCT32X32, 1024 entries. Generated by
/// the libjxl `CoeffOrderAndLut` algorithm (`is_lut=false`) with cx=cy=4; the
/// first 16 slots are the 4×4 LLF block. Needs `u16` since indices reach 1023.
pub(crate) static K_COEFF_ORDER_32X32: [u16; 1024] = [
    0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99, 128, 4, 5, 36, 129, 160, 192, 161,
    130, 68, 37, 6, 7, 38, 69, 100, 131, 162, 193, 224, 256, 225, 194, 163, 132, 101, 70, 39, 8, 9,
    40, 71, 102, 133, 164, 195, 226, 257, 288, 320, 289, 258, 227, 196, 165, 134, 103, 72, 41, 10,
    11, 42, 73, 104, 135, 166, 197, 228, 259, 290, 321, 352, 384, 353, 322, 291, 260, 229, 198,
    167, 136, 105, 74, 43, 12, 13, 44, 75, 106, 137, 168, 199, 230, 261, 292, 323, 354, 385, 416,
    448, 417, 386, 355, 324, 293, 262, 231, 200, 169, 138, 107, 76, 45, 14, 15, 46, 77, 108, 139,
    170, 201, 232, 263, 294, 325, 356, 387, 418, 449, 480, 512, 481, 450, 419, 388, 357, 326, 295,
    264, 233, 202, 171, 140, 109, 78, 47, 16, 17, 48, 79, 110, 141, 172, 203, 234, 265, 296, 327,
    358, 389, 420, 451, 482, 513, 544, 576, 545, 514, 483, 452, 421, 390, 359, 328, 297, 266, 235,
    204, 173, 142, 111, 80, 49, 18, 19, 50, 81, 112, 143, 174, 205, 236, 267, 298, 329, 360, 391,
    422, 453, 484, 515, 546, 577, 608, 640, 609, 578, 547, 516, 485, 454, 423, 392, 361, 330, 299,
    268, 237, 206, 175, 144, 113, 82, 51, 20, 21, 52, 83, 114, 145, 176, 207, 238, 269, 300, 331,
    362, 393, 424, 455, 486, 517, 548, 579, 610, 641, 672, 704, 673, 642, 611, 580, 549, 518, 487,
    456, 425, 394, 363, 332, 301, 270, 239, 208, 177, 146, 115, 84, 53, 22, 23, 54, 85, 116, 147,
    178, 209, 240, 271, 302, 333, 364, 395, 426, 457, 488, 519, 550, 581, 612, 643, 674, 705, 736,
    768, 737, 706, 675, 644, 613, 582, 551, 520, 489, 458, 427, 396, 365, 334, 303, 272, 241, 210,
    179, 148, 117, 86, 55, 24, 25, 56, 87, 118, 149, 180, 211, 242, 273, 304, 335, 366, 397, 428,
    459, 490, 521, 552, 583, 614, 645, 676, 707, 738, 769, 800, 832, 801, 770, 739, 708, 677, 646,
    615, 584, 553, 522, 491, 460, 429, 398, 367, 336, 305, 274, 243, 212, 181, 150, 119, 88, 57,
    26, 27, 58, 89, 120, 151, 182, 213, 244, 275, 306, 337, 368, 399, 430, 461, 492, 523, 554, 585,
    616, 647, 678, 709, 740, 771, 802, 833, 864, 896, 865, 834, 803, 772, 741, 710, 679, 648, 617,
    586, 555, 524, 493, 462, 431, 400, 369, 338, 307, 276, 245, 214, 183, 152, 121, 90, 59, 28, 29,
    60, 91, 122, 153, 184, 215, 246, 277, 308, 339, 370, 401, 432, 463, 494, 525, 556, 587, 618,
    649, 680, 711, 742, 773, 804, 835, 866, 897, 928, 960, 929, 898, 867, 836, 805, 774, 743, 712,
    681, 650, 619, 588, 557, 526, 495, 464, 433, 402, 371, 340, 309, 278, 247, 216, 185, 154, 123,
    92, 61, 30, 31, 62, 93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465, 496, 527,
    558, 589, 620, 651, 682, 713, 744, 775, 806, 837, 868, 899, 930, 961, 992, 993, 962, 931, 900,
    869, 838, 807, 776, 745, 714, 683, 652, 621, 590, 559, 528, 497, 466, 435, 404, 373, 342, 311,
    280, 249, 218, 187, 156, 125, 94, 63, 95, 126, 157, 188, 219, 250, 281, 312, 343, 374, 405,
    436, 467, 498, 529, 560, 591, 622, 653, 684, 715, 746, 777, 808, 839, 870, 901, 932, 963, 994,
    995, 964, 933, 902, 871, 840, 809, 778, 747, 716, 685, 654, 623, 592, 561, 530, 499, 468, 437,
    406, 375, 344, 313, 282, 251, 220, 189, 158, 127, 159, 190, 221, 252, 283, 314, 345, 376, 407,
    438, 469, 500, 531, 562, 593, 624, 655, 686, 717, 748, 779, 810, 841, 872, 903, 934, 965, 996,
    997, 966, 935, 904, 873, 842, 811, 780, 749, 718, 687, 656, 625, 594, 563, 532, 501, 470, 439,
    408, 377, 346, 315, 284, 253, 222, 191, 223, 254, 285, 316, 347, 378, 409, 440, 471, 502, 533,
    564, 595, 626, 657, 688, 719, 750, 781, 812, 843, 874, 905, 936, 967, 998, 999, 968, 937, 906,
    875, 844, 813, 782, 751, 720, 689, 658, 627, 596, 565, 534, 503, 472, 441, 410, 379, 348, 317,
    286, 255, 287, 318, 349, 380, 411, 442, 473, 504, 535, 566, 597, 628, 659, 690, 721, 752, 783,
    814, 845, 876, 907, 938, 969, 1000, 1001, 970, 939, 908, 877, 846, 815, 784, 753, 722, 691,
    660, 629, 598, 567, 536, 505, 474, 443, 412, 381, 350, 319, 351, 382, 413, 444, 475, 506, 537,
    568, 599, 630, 661, 692, 723, 754, 785, 816, 847, 878, 909, 940, 971, 1002, 1003, 972, 941,
    910, 879, 848, 817, 786, 755, 724, 693, 662, 631, 600, 569, 538, 507, 476, 445, 414, 383, 415,
    446, 477, 508, 539, 570, 601, 632, 663, 694, 725, 756, 787, 818, 849, 880, 911, 942, 973, 1004,
    1005, 974, 943, 912, 881, 850, 819, 788, 757, 726, 695, 664, 633, 602, 571, 540, 509, 478, 447,
    479, 510, 541, 572, 603, 634, 665, 696, 727, 758, 789, 820, 851, 882, 913, 944, 975, 1006,
    1007, 976, 945, 914, 883, 852, 821, 790, 759, 728, 697, 666, 635, 604, 573, 542, 511, 543, 574,
    605, 636, 667, 698, 729, 760, 791, 822, 853, 884, 915, 946, 977, 1008, 1009, 978, 947, 916,
    885, 854, 823, 792, 761, 730, 699, 668, 637, 606, 575, 607, 638, 669, 700, 731, 762, 793, 824,
    855, 886, 917, 948, 979, 1010, 1011, 980, 949, 918, 887, 856, 825, 794, 763, 732, 701, 670,
    639, 671, 702, 733, 764, 795, 826, 857, 888, 919, 950, 981, 1012, 1013, 982, 951, 920, 889,
    858, 827, 796, 765, 734, 703, 735, 766, 797, 828, 859, 890, 921, 952, 983, 1014, 1015, 984,
    953, 922, 891, 860, 829, 798, 767, 799, 830, 861, 892, 923, 954, 985, 1016, 1017, 986, 955,
    924, 893, 862, 831, 863, 894, 925, 956, 987, 1018, 1019, 988, 957, 926, 895, 927, 958, 989,
    1020, 1021, 990, 959, 991, 1022, 1023,
];

/// Natural coefficient order for DCT32X16 and DCT16X32 (shared, libjxl
/// `QuantTable::DCT16X32` geometry).
pub(crate) static K_COEFF_ORDER_32X16: [u16; 512] = [
    0, 1, 2, 3, 32, 33, 34, 35, 64, 4, 5, 65, 96, 66, 36, 6, 7, 37, 67, 97, 128, 98, 68, 38, 8, 9,
    39, 69, 99, 129, 160, 130, 100, 70, 40, 10, 11, 41, 71, 101, 131, 161, 192, 162, 132, 102, 72,
    42, 12, 13, 43, 73, 103, 133, 163, 193, 224, 194, 164, 134, 104, 74, 44, 14, 15, 45, 75, 105,
    135, 165, 195, 225, 256, 226, 196, 166, 136, 106, 76, 46, 16, 17, 47, 77, 107, 137, 167, 197,
    227, 257, 288, 258, 228, 198, 168, 138, 108, 78, 48, 18, 19, 49, 79, 109, 139, 169, 199, 229,
    259, 289, 320, 290, 260, 230, 200, 170, 140, 110, 80, 50, 20, 21, 51, 81, 111, 141, 171, 201,
    231, 261, 291, 321, 352, 322, 292, 262, 232, 202, 172, 142, 112, 82, 52, 22, 23, 53, 83, 113,
    143, 173, 203, 233, 263, 293, 323, 353, 384, 354, 324, 294, 264, 234, 204, 174, 144, 114, 84,
    54, 24, 25, 55, 85, 115, 145, 175, 205, 235, 265, 295, 325, 355, 385, 416, 386, 356, 326, 296,
    266, 236, 206, 176, 146, 116, 86, 56, 26, 27, 57, 87, 117, 147, 177, 207, 237, 267, 297, 327,
    357, 387, 417, 448, 418, 388, 358, 328, 298, 268, 238, 208, 178, 148, 118, 88, 58, 28, 29, 59,
    89, 119, 149, 179, 209, 239, 269, 299, 329, 359, 389, 419, 449, 480, 450, 420, 390, 360, 330,
    300, 270, 240, 210, 180, 150, 120, 90, 60, 30, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301,
    331, 361, 391, 421, 451, 481, 482, 452, 422, 392, 362, 332, 302, 272, 242, 212, 182, 152, 122,
    92, 62, 63, 93, 123, 153, 183, 213, 243, 273, 303, 333, 363, 393, 423, 453, 483, 484, 454, 424,
    394, 364, 334, 304, 274, 244, 214, 184, 154, 124, 94, 95, 125, 155, 185, 215, 245, 275, 305,
    335, 365, 395, 425, 455, 485, 486, 456, 426, 396, 366, 336, 306, 276, 246, 216, 186, 156, 126,
    127, 157, 187, 217, 247, 277, 307, 337, 367, 397, 427, 457, 487, 488, 458, 428, 398, 368, 338,
    308, 278, 248, 218, 188, 158, 159, 189, 219, 249, 279, 309, 339, 369, 399, 429, 459, 489, 490,
    460, 430, 400, 370, 340, 310, 280, 250, 220, 190, 191, 221, 251, 281, 311, 341, 371, 401, 431,
    461, 491, 492, 462, 432, 402, 372, 342, 312, 282, 252, 222, 223, 253, 283, 313, 343, 373, 403,
    433, 463, 493, 494, 464, 434, 404, 374, 344, 314, 284, 254, 255, 285, 315, 345, 375, 405, 435,
    465, 495, 496, 466, 436, 406, 376, 346, 316, 286, 287, 317, 347, 377, 407, 437, 467, 497, 498,
    468, 438, 408, 378, 348, 318, 319, 349, 379, 409, 439, 469, 499, 500, 470, 440, 410, 380, 350,
    351, 381, 411, 441, 471, 501, 502, 472, 442, 412, 382, 383, 413, 443, 473, 503, 504, 474, 444,
    414, 415, 445, 475, 505, 506, 476, 446, 447, 477, 507, 508, 478, 479, 509, 510, 511,
];

#[cfg(test)]
mod tests {
    use super::{
        CHANNEL_ROW, K_COMPACT_BLOCK_CONTEXT_MAP, K_NUM_AC_STRATEGY_CODES, K_NUM_BLOCK_CTXS,
        K_NUM_ORDERS, K_STRATEGY_ORDER, block_context,
    };

    /// The encoder must derive exactly the contexts the signaled map implies, or
    /// the decoder reads different histograms. Channel order differs between the
    /// two (local X,Y,B vs the map's Y,X,B), which is the easy thing to get wrong.
    #[test]
    fn derived_block_context_matches_the_signaled_map() {
        for code in 0..K_NUM_AC_STRATEGY_CODES as u8 {
            let order = K_STRATEGY_ORDER[code as usize] as usize;
            for (c, row) in CHANNEL_ROW.iter().enumerate() {
                assert_eq!(
                    block_context(c, code),
                    K_COMPACT_BLOCK_CONTEXT_MAP[row * K_NUM_ORDERS + order] as u32,
                    "channel {c} code {code}"
                );
            }
        }
        // Y contexts live below the chroma ones and every context is in range.
        for code in 0..K_NUM_AC_STRATEGY_CODES as u8 {
            let luma_ctxs = K_NUM_BLOCK_CTXS as u32 / 2;
            assert!(block_context(1, code) < luma_ctxs);
            for c in [0usize, 2] {
                assert!((luma_ctxs..K_NUM_BLOCK_CTXS as u32).contains(&block_context(c, code)));
            }
        }
    }

    /// Properties the block-context map must satisfy whatever grouping it uses,
    /// so that changing `K_COMPACT_BLOCK_CONTEXT_MAP` cannot silently break the
    /// invariants the AC context arithmetic relies on.
    #[test]
    fn block_context_map_invariants() {
        // Luma and chroma never share a context, and chroma X and B always do.
        for code in 0..K_NUM_AC_STRATEGY_CODES as u8 {
            let (x, y, b) = (
                block_context(0, code),
                block_context(1, code),
                block_context(2, code),
            );
            assert_eq!(x, b, "X and B differ for code {code}");
            assert_ne!(x, y, "luma and chroma share a context for code {code}");
            assert!(y < K_NUM_BLOCK_CTXS as u32 && x < K_NUM_BLOCK_CTXS as u32);
        }
        // Codes of the same transform order are indistinguishable by design.
        for a in 0..K_NUM_AC_STRATEGY_CODES as u8 {
            for b in 0..K_NUM_AC_STRATEGY_CODES as u8 {
                if K_STRATEGY_ORDER[a as usize] == K_STRATEGY_ORDER[b as usize] {
                    for c in 0..3 {
                        assert_eq!(block_context(c, a), block_context(c, b));
                    }
                }
            }
        }
        // Every declared context is reachable, or the count is overstated and
        // the AC context arithmetic reserves histograms that never get tokens.
        let used: std::collections::BTreeSet<u32> = (0..K_NUM_AC_STRATEGY_CODES as u8)
            .flat_map(|code| (0..3).map(move |c| block_context(c, code)))
            .collect();
        assert_eq!(
            used.len(),
            K_NUM_BLOCK_CTXS,
            "unreachable block contexts: {used:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Per-image block-context planning
// ---------------------------------------------------------------------------

/// The signaled block-context layout decided per image after tokenization.
///
/// Tokens are produced on the fine layout (every base context split by the
/// quant-field threshold); this plan says which splits are kept and which
/// chroma contexts are merged to stay inside the spec's 16-id budget. A plan
/// equal to [`AcCtxPlan::baseline`] signals byte-for-byte the legacy map.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct AcCtxPlan {
    pub(crate) nbc: u8,
    pub(crate) fine_to_final: [u8; K_NUM_FINE_BLOCK_CTXS],
    /// `Some(t)` when at least one kept split needs the threshold signaled.
    pub(crate) qf_threshold: Option<u32>,
}

impl AcCtxPlan {
    /// The legacy 15-context map: qf classes collapsed, nothing merged.
    pub(crate) fn baseline() -> Self {
        let mut fine_to_final = [0u8; K_NUM_FINE_BLOCK_CTXS];
        for (fine, out) in fine_to_final.iter_mut().enumerate() {
            *out = (fine / K_NUM_QF_CLASSES) as u8;
        }
        AcCtxPlan {
            nbc: K_NUM_BLOCK_CTXS as u8,
            fine_to_final,
            qf_threshold: None,
        }
    }

    #[inline]
    pub(crate) fn num_ac_contexts(&self) -> usize {
        self.nbc as usize * (K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT)
    }

    /// Remap a fine-layout token context into this plan's final layout.
    #[inline]
    pub(crate) fn remap(&self, ctx: u32) -> u32 {
        const NZ_SPAN: u32 = (K_NUM_FINE_BLOCK_CTXS * K_NON_ZERO_BUCKETS) as u32;
        let nbc = self.nbc as u32;
        if ctx < NZ_SPAN {
            let bucket = ctx / K_NUM_FINE_BLOCK_CTXS as u32;
            let fine = ctx % K_NUM_FINE_BLOCK_CTXS as u32;
            bucket * nbc + u32::from(self.fine_to_final[fine as usize])
        } else {
            let rel = ctx - NZ_SPAN;
            let fine = rel / K_ZERO_DENSITY_CONTEXT_COUNT as u32;
            let z = rel % K_ZERO_DENSITY_CONTEXT_COUNT as u32;
            nbc * K_NON_ZERO_BUCKETS as u32
                + u32::from(self.fine_to_final[fine as usize]) * K_ZERO_DENSITY_CONTEXT_COUNT as u32
                + z
        }
    }

    /// The signaled ctx_map entries: channel-row major, order next, qf inner.
    pub(crate) fn ctx_map_entries(&self) -> Vec<u8> {
        let qf_classes = if self.qf_threshold.is_some() {
            K_NUM_QF_CLASSES
        } else {
            1
        };
        let mut out = Vec::with_capacity(3 * K_NUM_ORDERS * qf_classes);
        for base in K_COMPACT_BLOCK_CONTEXT_MAP.iter() {
            for qf_idx in 0..qf_classes {
                out.push(self.fine_to_final[*base as usize * K_NUM_QF_CLASSES + qf_idx]);
            }
        }
        out
    }
}

/// Shannon cost model constants for plan decisions. A kept split grows the
/// signaled AC context map by one block context's worth of entries (495) and
/// typically one histogram cluster; a split paired with a chroma merge keeps
/// the map size unchanged and only pays churn. Margins hedge the proxy, per
/// the recurring clustering lesson.
const SPLIT_OVERHEAD_BITS: f64 = 800.0;
const PAIRED_OVERHEAD_BITS: f64 = 300.0;
const PLAN_GAIN_MARGIN: f64 = 1.3;
const MAX_BLOCK_CTXS: usize = 16;
const PLAN_NUM_SYMBOLS: usize = 128;
const PLAN_SAMPLE_STEP: usize = 4;

/// The proposal must predict at least this many bits of net gain before the
/// expensive dual-arm verification runs. Below it, marginal proposals were
/// overwhelmingly rejected by the gate anyway — pure encode-time waste.
const PROPOSE_MIN_NET_BITS: f64 = 4096.0;

/// Chroma merge candidates, cheapest-first per the offline study; each frees
/// one id for a split when the 16-id budget is exhausted.
const CHROMA_MERGE_CANDIDATES: [(u8, u8); 4] = [(10, 11), (8, 9), (12, 13), (13, 14)];

struct PlanStats {
    /// [fine ac context][symbol] counts plus raw extra bits.
    counts: Vec<u32>,
    extra: Vec<u64>,
}

impl PlanStats {
    fn new() -> Self {
        PlanStats {
            counts: vec![0u32; K_NUM_FINE_AC_CONTEXTS * PLAN_NUM_SYMBOLS],
            extra: vec![0u64; K_NUM_FINE_AC_CONTEXTS],
        }
    }
    fn add_stream(&mut self, tokens: &[crate::entropy::Token]) {
        // Subsampled: the stats only feed a proposal that a real-bits gate
        // verifies, and a 1-in-4 sample keeps this pass off the profile.
        for t in tokens.iter().step_by(PLAN_SAMPLE_STEP) {
            let (sym, nbits, _) = crate::entropy::uint_encode(t.value);
            self.counts[t.context as usize * PLAN_NUM_SYMBOLS + sym as usize] += 1;
            self.extra[t.context as usize] += u64::from(nbits);
        }
    }
    /// Cost of coding one final context made of `members` fine block contexts,
    /// summed over every sub-context slot (nonzero buckets + zero-density).
    fn group_cost(&self, members: &[u8]) -> f64 {
        let mut total = 0.0f64;
        let mut merged = [0u64; PLAN_NUM_SYMBOLS];
        let sub_slots = K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT;
        for slot in 0..sub_slots {
            let mut any = false;
            for v in merged.iter_mut() {
                *v = 0;
            }
            let mut extra = 0u64;
            for &fine in members {
                let ctx = if slot < K_NON_ZERO_BUCKETS {
                    slot * K_NUM_FINE_BLOCK_CTXS + fine as usize
                } else {
                    K_NUM_FINE_BLOCK_CTXS * K_NON_ZERO_BUCKETS
                        + fine as usize * K_ZERO_DENSITY_CONTEXT_COUNT
                        + (slot - K_NON_ZERO_BUCKETS)
                };
                let row = &self.counts[ctx * PLAN_NUM_SYMBOLS..(ctx + 1) * PLAN_NUM_SYMBOLS];
                for (dst, &src) in merged.iter_mut().zip(row) {
                    *dst += u64::from(src);
                    any |= src != 0;
                }
                // Extra bits accrue once per token regardless of grouping; they
                // cancel in comparisons but keep costs absolute for clarity.
                if slot == 0 {
                    extra += self.extra[ctx];
                }
            }
            if any {
                let n: u64 = merged.iter().sum();
                let nf = n as f64;
                total += merged
                    .iter()
                    .filter(|&&c| c != 0)
                    .map(|&c| c as f64 * f_log2(nf / c as f64))
                    .sum::<f64>();
            }
            total += extra as f64;
        }
        total
    }
}

/// Decide the per-image block-context map from real token statistics.
pub(crate) fn plan_block_ctx_map<'a, I>(streams: I, qf_threshold: u32) -> AcCtxPlan
where
    I: IntoIterator<Item = &'a [crate::entropy::Token]>,
{
    let mut stats = PlanStats::new();
    for s in streams {
        stats.add_stream(s);
    }

    // Groups of fine ids per final context; starts as the baseline pairing.
    // A group is "splittable" while it is exactly one base context's pair.
    let mut groups: Vec<Vec<u8>> = (0..K_NUM_BLOCK_CTXS as u8)
        .map(|b| vec![b * 2, b * 2 + 1])
        .collect();
    let mut group_cost: Vec<f64> = groups.iter().map(|g| stats.group_cost(g)).collect();
    let splittable = |g: &[u8]| g.len() == 2 && g[0].is_multiple_of(2) && g[1] == g[0] + 1;
    // Merge candidates may only combine unsplit groups.
    let unsplit = |g: &[u8]| {
        g.len().is_multiple_of(2)
            && g.chunks(2)
                .all(|p| p[0].is_multiple_of(2) && p[1] == p[0] + 1)
    };

    let mut merges_used = [false; CHROMA_MERGE_CANDIDATES.len()];
    let mut any_split = false;
    let mut total_net = 0.0f64;
    loop {
        let mut best: Option<(f64, usize)> = None;
        for (i, g) in groups.iter().enumerate() {
            if !splittable(g) {
                continue;
            }
            let gain = group_cost[i] - stats.group_cost(&[g[0]]) - stats.group_cost(&[g[1]]);
            if gain > 0.0 && best.is_none_or(|(bg, _)| gain > bg) {
                best = Some((gain, i));
            }
        }
        let Some((gain, split_pos)) = best else { break };
        let gain = gain * PLAN_SAMPLE_STEP as f64;
        let split_fines = groups[split_pos].clone();

        let merge_choice = if groups.len() < MAX_BLOCK_CTXS {
            None
        } else {
            let mut cheapest: Option<(f64, usize, Vec<u8>, Vec<u8>)> = None;
            for (mi, &(a, b)) in CHROMA_MERGE_CANDIDATES.iter().enumerate() {
                if merges_used[mi] {
                    continue;
                }
                let pa = groups.iter().position(|g| g.contains(&(a * 2)));
                let pb = groups.iter().position(|g| g.contains(&(b * 2)));
                let (Some(pa), Some(pb)) = (pa, pb) else {
                    continue;
                };
                if pa == pb
                    || !unsplit(&groups[pa])
                    || !unsplit(&groups[pb])
                    || pa == split_pos
                    || pb == split_pos
                {
                    continue;
                }
                let mut union = groups[pa].clone();
                union.extend_from_slice(&groups[pb]);
                let cost = (stats.group_cost(&union) - group_cost[pa] - group_cost[pb])
                    * PLAN_SAMPLE_STEP as f64;
                if cheapest.as_ref().is_none_or(|(c, ..)| cost < *c) {
                    cheapest = Some((cost, mi, groups[pa].clone(), groups[pb].clone()));
                }
            }
            match cheapest {
                Some(c) => Some(c),
                None => break, // budget full, nothing mergeable
            }
        };

        let net = match &merge_choice {
            None => gain - SPLIT_OVERHEAD_BITS * PLAN_GAIN_MARGIN,
            Some((cost, ..)) => gain - cost - PAIRED_OVERHEAD_BITS * PLAN_GAIN_MARGIN,
        };
        if net <= 0.0 {
            break;
        }
        total_net += net;

        if let Some((_, mi, ga, gb)) = merge_choice {
            merges_used[mi] = true;
            let pa = groups.iter().position(|g| *g == ga).unwrap();
            groups.remove(pa);
            group_cost.remove(pa);
            let pb = groups.iter().position(|g| *g == gb).unwrap();
            groups.remove(pb);
            group_cost.remove(pb);
            let mut union = ga;
            union.extend(gb);
            group_cost.push(stats.group_cost(&union));
            groups.push(union);
        }
        let pos = groups.iter().position(|g| *g == split_fines).unwrap();
        groups.remove(pos);
        group_cost.remove(pos);
        for fine in split_fines {
            group_cost.push(stats.group_cost(&[fine]));
            groups.push(vec![fine]);
        }
        any_split = true;
    }
    if total_net < PROPOSE_MIN_NET_BITS {
        return AcCtxPlan::baseline();
    }

    // Final numbering: stable by smallest fine id.
    groups.sort_by_key(|g| g.iter().copied().min().unwrap());
    let mut fine_to_final = [0u8; K_NUM_FINE_BLOCK_CTXS];
    for (id, g) in groups.iter().enumerate() {
        for &fine in g {
            fine_to_final[fine as usize] = id as u8;
        }
    }
    AcCtxPlan {
        nbc: groups.len() as u8,
        fine_to_final,
        qf_threshold: any_split.then_some(qf_threshold),
    }
}
