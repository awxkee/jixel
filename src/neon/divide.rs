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
use std::arch::aarch64::*;

/// `magnitude * magic >> 32`, lane-wise.
#[inline]
#[target_feature(enable = "neon")]
fn mulhi_s32(m: int32x4_t, magic: int32x4_t) -> int32x4_t {
    let lo = vmull_s32(vget_low_s32(m), vget_low_s32(magic));
    let hi = vmull_high_s32(m, magic);
    vuzp2q_s32(vreinterpretq_s32_s64(lo), vreinterpretq_s32_s64(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn magic_quotients<const ADD: bool>(v4s: &mut [[i32; 4]], half: i32, divider: DividerI32) {
    let half = vdupq_n_s32(half);
    let magic = vdupq_n_s32(divider.magic());
    let shift = vdupq_n_s32(-(divider.shift() as i32));
    for v4 in v4s {
        let v = unsafe { vld1q_s32(v4.as_ptr()) };
        let m = vaddq_s32(vabsq_s32(v), half);
        let mut uq = mulhi_s32(m, magic);
        if ADD {
            uq = vaddq_s32(uq, m);
        }
        let q = vshlq_s32(uq, shift);
        let q = vsubq_s32(q, vshrq_n_s32::<31>(q));
        let out = vbslq_s32(vcltzq_s32(v), vnegq_s32(q), q);
        unsafe { vst1q_s32(v4.as_mut_ptr(), out) };
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn rounded_quotients_neon(plane: &mut [i32], q: i32) {
    debug_assert!(q > 1);
    let divider = DividerI32::new(q);
    let (v4s, tail) = plane.as_chunks_mut::<4>();
    if divider.magic() == 0 {
        let half = vdupq_n_s32(q / 2);
        let shift = vdupq_n_s32(-(divider.shift() as i32));
        for v4 in v4s.iter_mut() {
            let v = unsafe { vld1q_s32(v4.as_ptr()) };
            let r = vshlq_s32(vaddq_s32(vabsq_s32(v), half), shift);
            let out = vbslq_s32(vcltzq_s32(v), vnegq_s32(r), r);
            unsafe { vst1q_s32(v4.as_mut_ptr(), out) };
        }
    } else if divider.adds_numerator() {
        magic_quotients::<true>(v4s, q / 2, divider);
    } else {
        magic_quotients::<false>(v4s, q / 2, divider);
    }
    rounded_quotients_scalar(tail, q);
}
