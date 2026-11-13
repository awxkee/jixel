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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "ssse3")]
pub(crate) fn deinterleave_rgb8(pixels: &[[u8; 3]; 16]) -> [__m128i; 3] {
    let src = pixels.as_ptr().cast::<u8>();
    let loads: [_; 3] = std::array::from_fn(|i| unsafe { _mm_loadu_si128(src.add(16 * i).cast()) });
    super::RGB8_SHUFFLE.each_ref().map(|channel| {
        let masks = channel
            .each_ref()
            .map(|part| unsafe { _mm_loadu_si128(part.as_ptr().cast()) });
        _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(loads[0], masks[0]),
                _mm_shuffle_epi8(loads[1], masks[1]),
            ),
            _mm_shuffle_epi8(loads[2], masks[2]),
        )
    })
}

#[cfg(feature = "sse")]
pub(crate) mod i32 {
    pub(crate) type Vector = __m128i;

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn load(p: *const i32) -> __m128i {
        unsafe { _mm_loadu_si128(p.cast()) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn store(p: *mut i32, v: __m128i) {
        unsafe { _mm_storeu_si128(p.cast(), v) };
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn shr<const N: i32>(v: __m128i) -> __m128i {
        _mm_srai_epi32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn select(m: __m128i, a: __m128i, b: __m128i) -> __m128i {
        _mm_blendv_epi8(b, a, m)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn splat(v: i32) -> __m128i {
        _mm_set1_epi32(v)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn add(a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn sub(a: __m128i, b: __m128i) -> __m128i {
        _mm_sub_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn and(a: __m128i, b: __m128i) -> __m128i {
        _mm_and_si128(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn or(a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn gt(a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpgt_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn abs(v: __m128i) -> __m128i {
        _mm_abs_epi32(v)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn load_u8(p: *const u8) -> __m128i {
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(unsafe {
            p.cast::<i32>().read_unaligned()
        }))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn load_u16(p: *const u16) -> __m128i {
        _mm_cvtepu16_epi32(unsafe { _mm_loadl_epi64(p.cast()) })
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn shl<const N: i32>(v: __m128i) -> __m128i {
        _mm_slli_epi32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn xor(a: __m128i, b: __m128i) -> __m128i {
        _mm_xor_si128(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn min(a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn max(a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn lt_unsigned(a: __m128i, b: __m128i) -> __m128i {
        let sign = _mm_set1_epi32(i32::MIN);
        _mm_cmpgt_epi32(xor(b, sign), xor(a, sign))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn shr_unsigned<const N: i32>(v: __m128i) -> __m128i {
        _mm_srli_epi32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn to_f32_bits(v: __m128i) -> __m128i {
        _mm_castps_si128(_mm_cvtepi32_ps(v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn load_pairs(p: *const i32) -> (__m128i, __m128i) {
        let a = _mm_castsi128_ps(unsafe { load(p) });
        let b = _mm_castsi128_ps(unsafe { load(p.add(4)) });
        (
            _mm_castps_si128(_mm_shuffle_ps::<0x88>(a, b)),
            _mm_castps_si128(_mm_shuffle_ps::<0xdd>(a, b)),
        )
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn mulhi(a: __m128i, b: __m128i) -> __m128i {
        let even = _mm_srli_epi64::<32>(_mm_mul_epi32(a, b));
        let odd = _mm_mul_epi32(_mm_srli_epi64::<32>(a), _mm_srli_epi64::<32>(b));
        _mm_blend_epi16::<0xcc>(even, odd)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn eq(a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn bits_clear(v: __m128i, mask: __m128i) -> bool {
        _mm_testz_si128(v, mask) != 0
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn mul(a: __m128i, b: __m128i) -> __m128i {
        _mm_mullo_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn shr_by(v: __m128i, count: u32) -> __m128i {
        _mm_sra_epi32(v, _mm_cvtsi32_si128(count as i32))
    }
}

pub(crate) mod f32 {
    #[cfg(all(target_arch = "x86", feature = "sse"))]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86")]
    pub(crate) use std::arch::x86::{
        _mm_add_ps as add, _mm_loadu_ps as load, _mm_mul_ps as mul, _mm_set1_ps as splat,
        _mm_storeu_ps as store, _mm_sub_ps as sub,
    };
    #[cfg(all(target_arch = "x86", feature = "sse"))]
    pub(crate) use std::arch::x86::{_mm_div_ps as div, _mm_max_ps as max, _mm_min_ps as min};
    #[cfg(all(target_arch = "x86_64", feature = "sse"))]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86_64")]
    pub(crate) use std::arch::x86_64::{
        _mm_add_ps as add, _mm_loadu_ps as load, _mm_mul_ps as mul, _mm_set1_ps as splat,
        _mm_storeu_ps as store, _mm_sub_ps as sub,
    };
    #[cfg(all(target_arch = "x86_64", feature = "sse"))]
    pub(crate) use std::arch::x86_64::{_mm_div_ps as div, _mm_max_ps as max, _mm_min_ps as min};

    #[cfg(feature = "sse")]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn abs(v: __m128) -> __m128 {
        _mm_andnot_ps(_mm_set1_ps(-0.0), v)
    }

    #[cfg(feature = "sse")]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn mask_nonzero(v: __m128, a: __m128, b: __m128) -> __m128 {
        _mm_and_ps(
            v,
            _mm_and_ps(
                _mm_cmpneq_ps(a, _mm_setzero_ps()),
                _mm_cmpneq_ps(b, _mm_setzero_ps()),
            ),
        )
    }

    #[cfg(feature = "sse")]
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn from_i32(v: __m128i) -> __m128 {
        _mm_cvtepi32_ps(v)
    }

    #[cfg(feature = "sse")]
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn trunc_to_i32(v: __m128) -> __m128i {
        _mm_cvttps_epi32(v)
    }
}

#[cfg(feature = "sse")]
#[target_feature(enable = "sse4.1")]
pub(crate) fn add_scaled_sse41(out: &mut [f32], src: &[f32], factor: f32) {
    let (src_blocks, src) = src[..out.len()].as_chunks::<4>();
    let (out_blocks, out) = out.as_chunks_mut::<4>();
    let scale = _mm_set1_ps(factor);
    for (dst, src) in out_blocks.iter_mut().zip(src_blocks) {
        let dst = dst.as_mut_ptr();
        // SAFETY: both arrays contain four elements.
        unsafe {
            _mm_storeu_ps(
                dst,
                _mm_add_ps(
                    _mm_loadu_ps(dst),
                    _mm_mul_ps(scale, _mm_loadu_ps(src.as_ptr())),
                ),
            );
        }
    }
    crate::simd::scaled_add::add_scaled_scalar(out, src, factor);
}
