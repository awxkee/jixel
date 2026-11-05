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

pub(crate) mod i32 {
    pub(crate) type Vector = int32x4_t;

    use std::arch::aarch64::*;
    pub(crate) use std::arch::aarch64::{
        vabsq_s32 as abs, vaddq_s32 as add, vandq_s32 as and, vdupq_n_s32 as splat,
        vorrq_s32 as or, vsubq_s32 as sub,
    };

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn load(p: *const i32) -> int32x4_t {
        unsafe { vld1q_s32(p) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn store(p: *mut i32, v: int32x4_t) {
        unsafe { vst1q_s32(p, v) };
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn shr<const N: i32>(v: int32x4_t) -> int32x4_t {
        vshrq_n_s32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn gt(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcgtq_s32(a, b))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn select(m: int32x4_t, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vbslq_s32(vreinterpretq_u32_s32(m), a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn load_u8(p: *const u8) -> int32x4_t {
        let bytes = vcreate_u8(u64::from(unsafe { p.cast::<u32>().read_unaligned() }));
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes))))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn load_u16(p: *const u16) -> int32x4_t {
        vreinterpretq_s32_u32(vmovl_u16(unsafe { vld1_u16(p) }))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn shl<const N: i32>(v: int32x4_t) -> int32x4_t {
        vshlq_n_s32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn xor(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        veorq_s32(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn min(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vminq_s32(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn max(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vmaxq_s32(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn lt_unsigned(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vcltq_u32(
            vreinterpretq_u32_s32(a),
            vreinterpretq_u32_s32(b),
        ))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn shr_unsigned<const N: i32>(v: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(v)))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn to_f32_bits(v: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_f32(vcvtq_f32_s32(v))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn load_pairs(p: *const i32) -> (int32x4_t, int32x4_t) {
        let v = unsafe { vld2q_s32(p) };
        (v.0, v.1)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn eq(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vreinterpretq_s32_u32(vceqq_s32(a, b))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn mulhi(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        let lo = vmull_s32(vget_low_s32(a), vget_low_s32(b));
        let hi = vmull_high_s32(a, b);
        vuzp2q_s32(vreinterpretq_s32_s64(lo), vreinterpretq_s32_s64(hi))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn bits_clear(v: int32x4_t, mask: int32x4_t) -> bool {
        vmaxvq_u32(vreinterpretq_u32_s32(and(v, mask))) == 0
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn mul(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vmulq_s32(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn shr_by(v: int32x4_t, count: u32) -> int32x4_t {
        vshlq_s32(v, vdupq_n_s32(-(count as i32)))
    }
}

pub(crate) mod f32 {
    use std::arch::aarch64::*;
    pub(crate) use std::arch::aarch64::{
        vabsq_f32 as abs, vaddq_f32 as add, vdivq_f32 as div, vdupq_n_f32 as splat,
        vld1q_f32 as load, vmaxnmq_f32 as max, vminnmq_f32 as min, vmulq_f32 as mul,
        vst1q_f32 as store, vsubq_f32 as sub,
    };

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn mask_nonzero(v: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vbslq_f32(vorrq_u32(vceqzq_f32(a), vceqzq_f32(b)), vdupq_n_f32(0.0), v)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn from_i32(v: int32x4_t) -> float32x4_t {
        vcvtq_f32_s32(v)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn trunc_to_i32(v: float32x4_t) -> int32x4_t {
        vcvtq_s32_f32(v)
    }
}
