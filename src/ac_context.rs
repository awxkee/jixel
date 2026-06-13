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
    // X row. Position 4 = DCT16X16 (decoder ctx 2). Position 5 = DCT32X32
    // (ord 3, ctx 2). Position 3 = DCT4X4, positions 12,13 = DCT4X8/DCT8X4 (all ord 1, ctx 2). Positions 6, 7 = DCT16X8/8X16 (ctx 3).
    2, 0, 0, 2, 2, 2, 3, 3, 0, 0, 0, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Y row. Position 4 = DCT16X16 (decoder ctx 0). Positions 6, 7 = DCT16X8/8X16 (ctx 1).
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // B row. Position 4 = DCT16X16 (decoder ctx 2). Position 5 = DCT32X32
    // (ord 3, ctx 2). Position 3 = DCT4X4, positions 12,13 = DCT4X8/DCT8X4 (all ord 1, ctx 2). Positions 6, 7 = DCT16X8/8X16 (ctx 3).
    2, 0, 0, 2, 2, 2, 3, 3, 0, 0, 0, 0, 2, 2,
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
