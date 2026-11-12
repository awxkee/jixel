/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

use std::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) fn deinterleave_rgb8(pixels: &[[u8; 3]; 16]) -> [v128; 3] {
    let src = pixels.as_ptr().cast::<u8>();
    let loads: [_; 3] = std::array::from_fn(|i| unsafe { v128_load(src.add(16 * i).cast()) });
    super::RGB8_SHUFFLE.each_ref().map(|channel| {
        let masks = channel
            .each_ref()
            .map(|part| unsafe { v128_load(part.as_ptr().cast()) });
        v128_or(
            v128_or(
                i8x16_swizzle(loads[0], masks[0]),
                i8x16_swizzle(loads[1], masks[1]),
            ),
            i8x16_swizzle(loads[2], masks[2]),
        )
    })
}

pub(crate) mod i32 {
    pub(crate) type Vector = v128;

    use std::arch::wasm32::*;
    pub(crate) use std::arch::wasm32::{
        i32x4_abs as abs, i32x4_add as add, i32x4_eq as eq, i32x4_gt as gt, i32x4_splat as splat,
        i32x4_sub as sub, v128_and as and, v128_or as or,
    };

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn load(p: *const i32) -> v128 {
        unsafe { v128_load(p.cast()) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn store(p: *mut i32, v: v128) {
        unsafe { v128_store(p.cast(), v) };
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn shr<const N: i32>(v: v128) -> v128 {
        i32x4_shr(v, N as u32)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn select(m: v128, a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, m)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn load_u8(p: *const u8) -> v128 {
        i32x4_extend_low_u16x8(i16x8_extend_low_u8x16(unsafe {
            v128_load32_zero(p.cast())
        }))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn load_u16(p: *const u16) -> v128 {
        i32x4_extend_low_u16x8(unsafe { v128_load64_zero(p.cast()) })
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn shl<const N: i32>(v: v128) -> v128 {
        i32x4_shl(v, N as u32)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn xor(a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn min(a: v128, b: v128) -> v128 {
        i32x4_min(a, b)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn max(a: v128, b: v128) -> v128 {
        i32x4_max(a, b)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn lt_unsigned(a: v128, b: v128) -> v128 {
        u32x4_lt(a, b)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn shr_unsigned<const N: i32>(v: v128) -> v128 {
        u32x4_shr(v, N as u32)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn to_f32_bits(v: v128) -> v128 {
        f32x4_convert_i32x4(v)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn load_pairs(p: *const i32) -> (v128, v128) {
        let a = unsafe { load(p) };
        let b = unsafe { load(p.add(4)) };
        (
            i32x4_shuffle::<0, 2, 4, 6>(a, b),
            i32x4_shuffle::<1, 3, 5, 7>(a, b),
        )
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn mulhi(a: v128, b: v128) -> v128 {
        i32x4_shuffle::<1, 3, 5, 7>(i64x2_extmul_low_i32x4(a, b), i64x2_extmul_high_i32x4(a, b))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn bits_clear(v: v128, mask: v128) -> bool {
        !v128_any_true(and(v, mask))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn mul(a: v128, b: v128) -> v128 {
        i32x4_mul(a, b)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn shr_by(v: v128, count: u32) -> v128 {
        i32x4_shr(v, count)
    }
}

pub(crate) mod f32 {
    use std::arch::wasm32::*;
    pub(crate) use std::arch::wasm32::{
        f32x4_abs as abs, f32x4_add as add, f32x4_div as div, f32x4_mul as mul,
        f32x4_splat as splat, f32x4_sub as sub,
    };

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn load(p: *const f32) -> v128 {
        unsafe { v128_load(p.cast()) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn store(p: *mut f32, v: v128) {
        unsafe { v128_store(p.cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn min(a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, f32x4_lt(a, b))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn max(a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, f32x4_gt(a, b))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn mask_nonzero(v: v128, a: v128, b: v128) -> v128 {
        let z = f32x4_splat(0.0);
        v128_bitselect(z, v, v128_or(f32x4_eq(a, z), f32x4_eq(b, z)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn from_i32(v: v128) -> v128 {
        f32x4_convert_i32x4(v)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn trunc_to_i32(v: v128) -> v128 {
        i32x4_trunc_sat_f32x4(v)
    }
}
