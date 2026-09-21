/*
 * // Copyright (c) Radzivon Bartoshyk 8/2025. All rights reserved.
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

#[inline(always)]
pub(crate) fn f_fmla(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    ))]
    {
        f64::mul_add(a, b, c)
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    )))]
    {
        a * b + c
    }
}

#[cfg(any(
    target_arch = "aarch64",
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    )
))]
#[inline(always)]
pub(crate) fn f_fmlaf(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    ))]
    {
        f32::mul_add(a, b, c)
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    )))]
    {
        a * b + c
    }
}

#[inline]
pub(crate) const fn get_exponent_f32(x: f32) -> i32 {
    let bits = x.to_bits();
    (((bits >> 23) & 0xFF) as i32).wrapping_sub(127)
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse4.1"
    )
)))]
#[inline]
const fn round(x: f64) -> f64 {
    let mut i0: i64 = x.to_bits() as i64;
    let j0: i32 = (((i0 >> 52) & 0x7ff) - 0x3ff) as i32;
    if j0 < 52 {
        if j0 < 0 {
            i0 &= 0x8000000000000000u64 as i64;
            if j0 == -1 {
                i0 |= 0x3ff0000000000000u64 as i64;
            }
        } else {
            let i = (0x000fffffffffffffu64 >> j0) as i64;
            if (i0 & i) == 0 {
                /* X is integral.  */
                return x;
            }

            i0 += (0x0008000000000000u64 >> j0) as i64;
            i0 &= !i;
        }
    } else {
        return if j0 == 0x400 {
            /* Inf or NaN.  */
            x + x
        } else {
            x
        };
    }
    f64::from_bits(i0 as u64)
}

#[inline]
pub(crate) fn fround_finite(x: f64) -> f64 {
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse4.1"
        ),
        target_arch = "aarch64"
    ))]
    {
        x.round()
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse4.1"
        ),
        target_arch = "aarch64"
    )))]
    {
        round(x)
    }
}

pub(super) trait CpuRound {
    fn cpu_round(self) -> Self;
}
impl CpuRound for f64 {
    #[inline(always)]
    fn cpu_round(self) -> Self {
        fround_finite(self)
    }
}
