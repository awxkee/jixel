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
#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub context: u32,
    pub value: u32,
}

impl Token {
    #[inline]
    pub const fn new(context: u32, value: u32) -> Self {
        Self { context, value }
    }
}

/// Hybrid-uint encode: split a u32 into (token, nbits, bits).
/// With config (4, 2, 0):
///   value < 16              -> (value, 0, 0)
///   value in [2^n, 2^(n+1)) -> token = 4n + top-2-bits-below-leading-1,
///                              nbits = n - 2,
///                              bits  = low (n - 2) bits of value.
#[inline]
pub fn uint_encode(value: u32) -> (u32, u32, u32) {
    if value < 16 {
        (value, 0, 0)
    } else {
        let n = 31 - value.leading_zeros(); // floor_log2_nonzero
        debug_assert!(n >= 4);
        let m = value - (1 << n);
        let token = (n << 2) + (m >> (n - 2));
        let nbits = n - 2;
        let bits = value & ((1u32 << nbits) - 1);
        (token, nbits, bits)
    }
}

/// Map a signed integer to a non-negative one via "zigzag" encoding.
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
#[inline]
pub fn pack_signed(value: i32) -> u32 {
    ((value << 1) ^ (value >> 31)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint_encode_small_values() {
        for v in 0..16u32 {
            assert_eq!(uint_encode(v), (v, 0, 0));
        }
    }

    #[test]
    fn uint_encode_boundary_16() {
        // 16 = 10000, n=4, m=0, token = 16+0 = 16, nbits = 2
        assert_eq!(uint_encode(16), (16, 2, 0));
        // 17 = 10001, m=1, token=16, bits=01
        assert_eq!(uint_encode(17), (16, 2, 1));
        // 20 = 10100, m=4, m>>2 = 1, token=17, bits=00
        assert_eq!(uint_encode(20), (17, 2, 0));
        // 31 = 11111, m=15, m>>2 = 3, token=19, bits=11
        assert_eq!(uint_encode(31), (19, 2, 3));
    }

    #[test]
    fn uint_encode_n5() {
        // 32 = 100000, n=5, m=0, token=20, nbits=3, bits=0
        assert_eq!(uint_encode(32), (20, 3, 0));
    }

    #[test]
    fn pack_signed_roundtrip() {
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(-2), 3);
        assert_eq!(pack_signed(2), 4);
    }
}
