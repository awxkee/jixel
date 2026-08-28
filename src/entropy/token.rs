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
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Token {
    pub(crate) context: u32,
    pub(crate) value: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HybridUintConfig {
    pub(crate) split_exponent: u8,
    pub(crate) msb_in_token: u8,
    pub(crate) lsb_in_token: u8,
}

impl HybridUintConfig {
    pub(crate) const DEFAULT: Self = Self {
        split_exponent: 4,
        msb_in_token: 2,
        lsb_in_token: 0,
    };
}

impl Token {
    #[inline]
    pub(crate) const fn new(context: u32, value: u32) -> Self {
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
pub(crate) fn uint_encode(value: u32) -> (u32, u32, u32) {
    if value < 16 {
        (value, 0, 0)
    } else {
        let n = 31 - value.leading_zeros();
        let m = value - (1 << n);
        let token = (n << 2) + (m >> (n - 2));
        let nbits = n - 2;
        let bits = value & ((1u32 << nbits) - 1);
        (token, nbits, bits)
    }
}

/// Hybrid-uint encoding for an arbitrary JPEG XL UintConfig.
#[inline]
pub(crate) fn uint_encode_with_config(value: u32, config: HybridUintConfig) -> (u32, u32, u32) {
    if config == HybridUintConfig::DEFAULT {
        return uint_encode(value);
    }
    let split = config.split_exponent as u32;
    let msb = config.msb_in_token as u32;
    let lsb = config.lsb_in_token as u32;
    debug_assert!(msb + lsb <= split);
    if value < (1u32 << split) {
        return (value, 0, 0);
    }

    let n = 31 - value.leading_zeros();
    let nbits = n - msb - lsb;
    let low_token = value & ((1u32 << lsb) - 1);
    let high_token = (value >> (n - msb)) & ((1u32 << msb) - 1);
    let token = (1u32 << split) + ((n - split) << (msb + lsb)) + (high_token << lsb) + low_token;
    let bits = (value >> lsb) & ((1u32 << nbits) - 1);
    (token, nbits, bits)
}

/// Map a signed integer to a non-negative one via "zigzag" encoding.
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
/// Computed in i64 so it is exact across the full i32 range (the naive
/// `value << 1` overflows i32 for |value| >= 2^30, which occurs with the large
/// residuals produced when coding raw float bits). Identical result to the
/// naive form for all values that don't overflow, so 8/16-bit output is
/// unchanged. Matches libjxl's PackSigned (which uses int64 internally).
#[inline]
pub(crate) fn pack_signed(value: i32) -> u32 {
    let v = value as i64;
    ((v << 1) ^ (v >> 63)) as u32
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

    fn uint_decode_for_test(token: u32, _nbits: u32, bits: u32, c: HybridUintConfig) -> u32 {
        let split = c.split_exponent as u32;
        if token < (1 << split) {
            return token;
        }
        let msb = c.msb_in_token as u32;
        let lsb = c.lsb_in_token as u32;
        let rest = token - (1 << split);
        let token_bits = msb + lsb;
        let n = split + (rest >> token_bits);
        let low = rest & ((1 << lsb) - 1);
        let high = (rest >> lsb) & ((1 << msb) - 1);
        (1 << n) | (high << (n - msb)) | (bits << lsb) | low
    }

    #[test]
    fn configurable_uint_round_trips() {
        let configs = [
            HybridUintConfig {
                split_exponent: 0,
                msb_in_token: 0,
                lsb_in_token: 0,
            },
            HybridUintConfig {
                split_exponent: 3,
                msb_in_token: 0,
                lsb_in_token: 0,
            },
            HybridUintConfig {
                split_exponent: 4,
                msb_in_token: 1,
                lsb_in_token: 1,
            },
            HybridUintConfig::DEFAULT,
            HybridUintConfig {
                split_exponent: 6,
                msb_in_token: 1,
                lsb_in_token: 0,
            },
        ];
        let values = [0, 1, 2, 15, 16, 17, 31, 32, 255, 65_535, u32::MAX];
        for config in configs {
            for value in values {
                let (token, nbits, bits) = uint_encode_with_config(value, config);
                assert_eq!(uint_decode_for_test(token, nbits, bits, config), value);
            }
        }
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
