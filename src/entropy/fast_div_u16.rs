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

const FAST_DIV_ADD_MARKER: u8 = 0x40;
const FAST_DIV_SHIFT_MASK: u8 = 0x1f;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FastDivU16 {
    magic: u32,
    more: u8,
}

impl FastDivU16 {
    #[inline]
    pub(crate) fn new(divisor: u16) -> Self {
        assert_ne!(divisor, 0, "division by zero");
        Self::new_u32(divisor as u32)
    }

    #[inline]
    pub(crate) fn new_or_one(divisor: u16) -> Self {
        if divisor == 0 {
            Self { magic: 0, more: 0 }
        } else {
            Self::new(divisor)
        }
    }

    fn new_u32(divisor: u32) -> Self {
        let floor_log_2_d = 31 - divisor.leading_zeros();

        if divisor.is_power_of_two() {
            return Self {
                magic: 0,
                more: floor_log_2_d as u8,
            };
        }

        let numerator = 1u64 << (32 + floor_log_2_d);
        let mut proposed_m = numerator / divisor as u64;
        let rem = (numerator % divisor as u64) as u32;
        let e = divisor - rem;

        let more = if e < (1u32 << floor_log_2_d) {
            floor_log_2_d as u8
        } else {
            proposed_m += proposed_m;

            let twice_rem = rem.wrapping_add(rem);
            if twice_rem >= divisor || twice_rem < rem {
                proposed_m += 1;
            }

            floor_log_2_d as u8 | FAST_DIV_ADD_MARKER
        };

        Self {
            magic: proposed_m.wrapping_add(1) as u32,
            more,
        }
    }

    #[inline(always)]
    fn mul_hi_u32(a: u32, b: u32) -> u32 {
        (((a as u64) * (b as u64)) >> 32) as u32
    }

    #[inline(always)]
    pub(crate) fn div_fast(self, n: u32) -> u32 {
        if self.magic == 0 {
            return n >> self.more;
        }

        let q = Self::mul_hi_u32(n, self.magic);

        if (self.more & FAST_DIV_ADD_MARKER) != 0 {
            let shift = self.more & FAST_DIV_SHIFT_MASK;
            let t = ((n.wrapping_sub(q)) >> 1).wrapping_add(q);
            t >> shift
        } else {
            q >> self.more
        }
    }

    #[inline(always)]
    pub(crate) fn div_rem_fast(self, n: u32, divisor: u32) -> (u32, u32) {
        let q = self.div_fast(n);
        let r = n - q * divisor;
        debug_assert!(r < divisor);
        (q, r)
    }
}

#[cfg(test)]
mod fast_div_u16_tests {
    use super::FastDivU16;

    #[test]
    fn fast_div_u16_edges() {
        let divisors = [
            1u16,
            2,
            3,
            4,
            5,
            7,
            8,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            255,
            256,
            511,
            512,
            1023,
            1024,
            2047,
            2048,
            4095,
            4096,
            u16::MAX,
        ];
        let numerators = [
            0u32,
            1,
            2,
            3,
            4,
            5,
            7,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            255,
            256,
            1023,
            1024,
            65535,
            65536,
            123_456_789,
            u32::MAX - 1,
            u32::MAX,
        ];

        for &d in &divisors {
            let fd = FastDivU16::new(d);
            for &n in &numerators {
                let (q, r) = fd.div_rem_fast(n, d as u32);
                assert_eq!(q, n / d as u32, "bad quotient: {n} / {d}");
                assert_eq!(r, n % d as u32, "bad remainder: {n} % {d}");
            }
        }
    }

    #[test]
    fn fast_div_u16_random_full_numerators() {
        let mut x = 0x1234_5678u32;
        for _ in 0..1_000_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = x;
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let d = ((x & 0xffff) as u16).max(1);
            let fd = FastDivU16::new(d);
            let (q, r) = fd.div_rem_fast(n, d as u32);
            assert_eq!(q, n / d as u32, "bad quotient: {n} / {d}");
            assert_eq!(r, n % d as u32, "bad remainder: {n} % {d}");
        }
    }
}
