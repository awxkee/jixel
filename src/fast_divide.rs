/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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

use std::ops::Div;
use std::sync::OnceLock;

#[inline(always)]
fn mulhi_i32(x: i32, y: i32) -> i32 {
    let xl = x as i64;
    let yl = y as i64;
    ((xl * yl) >> 32) as i32
}

/// A precomputed signed 32-bit divisor; `x / divider` truncates toward zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DividerI32 {
    magic: i32,
    more: u8,
}

impl DividerI32 {
    /// `divisor` must not be zero.
    pub(crate) fn new(divisor: i32) -> DividerI32 {
        debug_assert!(divisor != 0);
        let ud: u32 = divisor.unsigned_abs();
        let floor_log_2_d = 31 - ud.leading_zeros();
        if (ud & (ud - 1)) == 0 {
            // Branchfree and non-branchfree cases are the same
            let mut more = floor_log_2_d as u8;
            if divisor < 0 {
                more |= 0x80;
            }
            return DividerI32 { magic: 0, more };
        }
        // the dividend here is 2**(floor_log_2_d + 31), so the low 32 bit word
        // is 0 and the high word is floor_log_2_d - 1
        let num = (1u64 << (floor_log_2_d - 1)) << 32;
        let u_div = ud as u64;
        let (mut proposed_m, rem) = ((num / u_div) as u32, (num % u_div) as u32);
        let e = ud - rem;

        // We are going to start with a power of floor_log_2_d - 1.
        // This works if e < 2**floor_log_2_d.
        let mut more: u8 = if e < (1u32 << floor_log_2_d) {
            // This power works
            (floor_log_2_d - 1) as u8
        } else {
            // We need to go one higher. This should not make proposed_m
            // overflow, but it will make it negative when interpreted as an
            // int32_t.
            proposed_m += proposed_m;
            let twice_rem = rem.wrapping_add(rem);
            if twice_rem >= ud || twice_rem < rem {
                proposed_m += 1;
            }
            (floor_log_2_d | 0x40) as u8
        };
        proposed_m = proposed_m.wrapping_add(1);
        let mut magic: i32 = proposed_m as i32;

        // Mark if we are negative
        if divisor < 0 {
            more |= 0x80;
            magic = magic.wrapping_neg();
        }
        DividerI32 { magic, more }
    }
}

// Accessors for the SIMD dividers that rebuild the scalar algorithm.
#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(target_arch = "aarch64", feature = "neon"),
    all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
))]
impl DividerI32 {
    /// The multiplier; 0 for a power of two, which is a plain shift.
    #[inline(always)]
    pub(crate) fn magic(self) -> i32 {
        self.magic
    }

    #[inline(always)]
    pub(crate) fn shift(self) -> u32 {
        (self.more & 0x3F) as u32
    }

    /// Whether the numerator is added to the multiply-high (a magic that
    /// needed 33 bits).
    #[inline(always)]
    pub(crate) fn adds_numerator(self) -> bool {
        self.more & 0x40 != 0
    }
}

impl Div<DividerI32> for i32 {
    type Output = i32;

    #[inline(always)]
    fn div(self, denom: DividerI32) -> Self::Output {
        let shift = denom.more & 0x3F;
        // All ones for a negative divisor, else zero.
        let sign: i32 = ((denom.more as i8) >> 7) as i32;

        if denom.magic == 0 {
            // shift path
            let mask = (1u32 << shift) - 1;
            let uq: u32 = (self as u32).wrapping_add((self >> 31) as u32 & mask);
            let mut q = uq as i32;
            q >>= shift;
            q = (q ^ sign) - sign;
            q
        } else {
            let mut uq = mulhi_i32(self, denom.magic);
            if (denom.more & 0x40) != 0 {
                // q += (more < 0 ? -numer : numer)
                uq = uq.wrapping_add((self ^ sign).wrapping_sub(sign));
            }
            let mut q = uq;
            q >>= shift;
            q += (q < 0) as i32;
            q
        }
    }
}

/// An exact unsigned 64-bit reciprocal divider. Like zaft's divider, this
/// uses the high half of a 128-bit product, but one native division sets up
/// the reciprocal instead of a 128-bit division.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DividerU64 {
    magic: u64,
    divisor: u64,
}

impl DividerU64 {
    pub(crate) const fn new(divisor: u64) -> Self {
        assert!(divisor != 0, "division by zero");
        Self {
            magic: u64::MAX / divisor,
            divisor,
        }
    }

    // floor((2^64 - 1) / d) differs from 2^64/d by at most one.
    // Its multiply-high quotient is therefore exact or one too small,
    // including d = 1. The product q*d cannot exceed the numerator.
    #[inline(always)]
    fn quotient_remainder(self, numerator: u64) -> (u64, u64) {
        let q = ((numerator as u128 * self.magic as u128) >> 64) as u64;
        (q, numerator - q * self.divisor)
    }

    #[inline(always)]
    pub(crate) fn div_rem(self, numerator: u64) -> (u64, u64) {
        let (q, r) = self.quotient_remainder(numerator);
        if r >= self.divisor {
            (q + 1, r - self.divisor)
        } else {
            (q, r)
        }
    }
}

impl Div<DividerU64> for u64 {
    type Output = u64;

    #[inline(always)]
    fn div(self, denom: DividerU64) -> Self::Output {
        let (q, r) = denom.quotient_remainder(self);
        q + u64::from(r >= denom.divisor)
    }
}

/// A precomputed signed 64-bit divisor, adapted from osclet with the signed
/// power-of-two and wrapping arithmetic fixes used by [`DividerI32`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct DividerI64 {
    magic: i64,
    more: u8,
}

impl DividerI64 {
    pub(crate) fn new(divisor: i64) -> Self {
        assert!(divisor != 0, "division by zero");
        let ud = divisor.unsigned_abs();
        let floor_log = 63 - ud.leading_zeros();
        let sign = if divisor < 0 { 0x80 } else { 0 };
        if ud.is_power_of_two() {
            return Self {
                magic: 0,
                more: floor_log as u8 | sign,
            };
        }
        let numerator = 1u128 << (floor_log + 63);
        let mut proposed = (numerator / ud as u128) as u64;
        let remainder = (numerator % ud as u128) as u64;
        let mut more = if ud - remainder < 1u64 << floor_log {
            (floor_log - 1) as u8
        } else {
            proposed = proposed.wrapping_add(proposed);
            let twice_remainder = remainder.wrapping_add(remainder);
            if twice_remainder >= ud || twice_remainder < remainder {
                proposed = proposed.wrapping_add(1);
            }
            floor_log as u8 | 0x40
        };
        let mut magic = proposed.wrapping_add(1) as i64;
        if divisor < 0 {
            more |= sign;
            magic = magic.wrapping_neg();
        }
        Self { magic, more }
    }
}

impl Div<DividerI64> for i64 {
    type Output = i64;

    #[inline(always)]
    fn div(self, denom: DividerI64) -> Self::Output {
        let shift = denom.more & 0x3f;
        let sign = ((denom.more as i8) >> 7) as i64;
        if denom.magic == 0 {
            assert!(self != i64::MIN || denom.more != 0x80, "division overflow");
            let mask = (1u64 << shift) - 1;
            let q = (self as u64).wrapping_add((self >> 63) as u64 & mask) as i64;
            ((q >> shift) ^ sign).wrapping_sub(sign)
        } else {
            let mut q = ((self as i128 * denom.magic as i128) >> 64) as i64;
            if denom.more & 0x40 != 0 {
                q = q.wrapping_add((self ^ sign).wrapping_sub(sign));
            }
            q >>= shift;
            q + i64::from(q < 0)
        }
    }
}

/// A plane kernel: rounds every sample to its nearest multiple of `q > 1`
/// (ties away from zero) and stores the quotient. `|sample| + q / 2` must
/// fit in an `i32`.
pub(crate) type RoundedQuotientsFn = unsafe fn(&mut [i32], i32);

/// [`RoundedQuotientsFn`] without SIMD; also the tails of the SIMD kernels.
pub(crate) fn rounded_quotients_scalar(plane: &mut [i32], q: i32) {
    debug_assert!(q > 1);
    let divider = DividerI32::new(q);
    let half = q / 2;
    for v in plane.iter_mut() {
        let sign = *v >> 31;
        let magnitude = (*v ^ sign).wrapping_sub(sign);
        *v = ((magnitude.wrapping_add(half) / divider) ^ sign).wrapping_sub(sign);
    }
}

fn select_rounded_quotients_fn() -> RoundedQuotientsFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        return crate::avx::rounded_quotients_avx2;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::rounded_quotients_sse41;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return crate::neon::rounded_quotients_neon;
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::rounded_quotients_wasm;
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    rounded_quotients_scalar
}

static ROUNDED_QUOTIENTS_FN: OnceLock<RoundedQuotientsFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_rounded_quotients_fn() -> RoundedQuotientsFn {
    *ROUNDED_QUOTIENTS_FN.get_or_init(select_rounded_quotients_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_u64(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn unsigned_64_matches_full_width_division() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        let mut divisors: Vec<u64> = (1..=1024).collect();
        for shift in 1..64 {
            let power = 1u64 << shift;
            divisors.extend([power - 1, power, power + 1]);
        }
        divisors.extend([u64::MAX, u64::MAX - 1]);
        divisors.extend((0..4096).map(|_| next_u64(&mut state).max(1)));
        for d in divisors {
            let divider = DividerU64::new(d);
            for x in [0, 1, d - 1, d, d.saturating_add(1), u64::MAX] {
                assert_eq!(x / divider, x / d, "x={x}, divisor={d}");
                assert_eq!(divider.div_rem(x), (x / d, x % d));
            }
            for _ in 0..128 {
                let x = next_u64(&mut state);
                assert_eq!(x / divider, x / d, "x={x}, divisor={d}");
                assert_eq!(divider.div_rem(x), (x / d, x % d));
            }
        }
    }

    #[test]
    fn signed_64_matches_full_width_division() {
        let mut state = 0x6a09_e667_f3bc_c909;
        let mut divisors = vec![i64::MIN, i64::MIN + 1, i64::MAX];
        for shift in 0..63 {
            let power = 1i64 << shift;
            for d in [power - 1, power, power + 1] {
                if d != 0 {
                    divisors.extend([d, -d]);
                }
            }
        }
        divisors.extend((0..4096).map(|_| {
            let d = next_u64(&mut state) as i64;
            if d == 0 { 1 } else { d }
        }));
        for d in divisors {
            let divider = DividerI64::new(d);
            for x in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX, d] {
                if let Some(want) = x.checked_div(d) {
                    assert_eq!(x / divider, want, "x={x}, divisor={d}");
                }
            }
            for _ in 0..128 {
                let x = next_u64(&mut state) as i64;
                if let Some(want) = x.checked_div(d) {
                    assert_eq!(x / divider, want, "x={x}, divisor={d}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn unsigned_64_rejects_zero() {
        DividerU64::new(0);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn signed_64_rejects_zero() {
        DividerI64::new(0);
    }

    #[test]
    #[should_panic(expected = "division overflow")]
    fn signed_64_rejects_overflow() {
        let _ = i64::MIN / DividerI64::new(-1);
    }

    #[test]
    fn matches_truncating_division() {
        let divisors = [
            1,
            2,
            3,
            7,
            15,
            64,
            150,
            255,
            1000,
            65535,
            1 << 20,
            i32::MAX,
            i32::MIN + 1,
            -1,
            -2,
            -3,
            -150,
            -(1 << 20),
        ];
        let values = [
            -500,
            -150,
            -15,
            -1,
            0,
            1,
            15,
            150,
            500,
            123_456_789,
            -987_654_321,
            i32::MAX,
            i32::MIN + 1,
        ];
        for &d in &divisors {
            let divider = DividerI32::new(d);
            for &x in &values {
                assert_eq!(x / divider, x / d, "x = {x}, divisor = {d}");
            }
        }
    }

    #[test]
    fn matches_truncating_division_densely() {
        for d in (1..2000).chain([4095, 4096, 4097, 99_991]) {
            let divider = DividerI32::new(d);
            for x in (-70_000..70_000).step_by(7) {
                assert_eq!(x / divider, x / d, "x = {x}, divisor = {d}");
            }
        }
    }

    #[test]
    fn plane_kernels_match_the_rounding_definition() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut values: Vec<i32> = (-5000..=5000).collect();
        values.extend((0..20_000).map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as i32) >> (1 + seed % 8)
        }));
        values.extend([
            i32::MAX / 2,
            i32::MIN / 2,
            i32::MAX / 2 - 1,
            i32::MIN / 2 + 1,
        ]);
        // Odd length: the SIMD kernels' scalar tails run too.
        values.push(7);
        let divisors = (2..300).chain([
            511,
            512,
            513,
            1000,
            4095,
            4096,
            65535,
            65536,
            99_991,
            1 << 20,
            3 << 20,
        ]);
        let kernels: Vec<(&str, RoundedQuotientsFn)> = vec![
            ("dispatch", selected_rounded_quotients_fn()),
            ("scalar", rounded_quotients_scalar),
        ];
        #[cfg(any(
            all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"),
            all(target_arch = "x86_64", feature = "avx"),
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
        ))]
        let mut kernels = kernels;
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse4.1") {
            kernels.push(("sse4.1", crate::sse::rounded_quotients_sse41));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", crate::avx::rounded_quotients_avx2));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        kernels.push(("neon", crate::neon::rounded_quotients_neon));
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
        kernels.push(("wasm", crate::wasm::rounded_quotients_wasm));
        for q in divisors {
            let want: Vec<i32> = values
                .iter()
                .map(|&a| {
                    if a < 0 {
                        -((-a + q / 2) / q)
                    } else {
                        (a + q / 2) / q
                    }
                })
                .collect();
            for &(name, f) in &kernels {
                for len in [values.len(), values.len() - 1, 3] {
                    let mut out = values[..len].to_vec();
                    unsafe { f(&mut out, q) };
                    assert_eq!(out, want[..len], "{name}: q = {q}, len = {len}");
                }
            }
        }
    }
}
