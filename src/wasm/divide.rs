/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Plane division by a loop-invariant step through [`DividerI32`]: every
//! sample becomes its nearest multiple of `q` (ties away from zero), stored
//! divided by `q`. Powers of two take a plain shift; other steps a 32x32->64
//! multiply-high, plus the numerator when the magic needs 33 bits.

use crate::fast_divide::{DividerI32, rounded_quotients_scalar};
use core::arch::wasm32::*;

/// `m * magic >> 32`, lane-wise.
#[inline]
#[target_feature(enable = "simd128")]
fn mulhi_i32x4(m: v128, magic: v128) -> v128 {
    let lo = i64x2_extmul_low_i32x4(m, magic);
    let hi = i64x2_extmul_high_i32x4(m, magic);
    i32x4_shuffle::<1, 3, 5, 7>(lo, hi)
}

#[inline]
#[target_feature(enable = "simd128")]
fn magic_quotients<const ADD: bool>(v4s: &mut [[i32; 4]], half: i32, divider: DividerI32) {
    let half = i32x4_splat(half);
    let magic = i32x4_splat(divider.magic());
    let shift = divider.shift();
    for v4 in v4s {
        let v = unsafe { v128_load(v4.as_ptr().cast()) };
        let sign = i32x4_shr(v, 31);
        let m = i32x4_add(i32x4_abs(v), half);
        let mut uq = mulhi_i32x4(m, magic);
        if ADD {
            uq = i32x4_add(uq, m);
        }
        let q = i32x4_shr(uq, shift);
        let q = i32x4_sub(q, i32x4_shr(q, 31));
        let out = i32x4_sub(v128_xor(q, sign), sign);
        unsafe { v128_store(v4.as_mut_ptr().cast(), out) };
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn rounded_quotients_wasm(plane: &mut [i32], q: i32) {
    debug_assert!(q > 1);
    let divider = DividerI32::new(q);
    let (v4s, tail) = plane.as_chunks_mut::<4>();
    if divider.magic() == 0 {
        let half = i32x4_splat(q / 2);
        let shift = divider.shift();
        for v4 in v4s.iter_mut() {
            let v = unsafe { v128_load(v4.as_ptr().cast()) };
            let sign = i32x4_shr(v, 31);
            let r = i32x4_shr(i32x4_add(i32x4_abs(v), half), shift);
            let out = i32x4_sub(v128_xor(r, sign), sign);
            unsafe { v128_store(v4.as_mut_ptr().cast(), out) };
        }
    } else if divider.adds_numerator() {
        magic_quotients::<true>(v4s, q / 2, divider);
    } else {
        magic_quotients::<false>(v4s, q / 2, divider);
    }
    rounded_quotients_scalar(tail, q);
}
