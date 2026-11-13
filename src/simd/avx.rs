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
    pub(crate) type Vector = __m256i;

    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn load(p: *const i32) -> __m256i {
        unsafe { _mm256_loadu_si256(p.cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn store(p: *mut i32, v: __m256i) {
        unsafe { _mm256_storeu_si256(p.cast(), v) };
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn shr<const N: i32>(v: __m256i) -> __m256i {
        _mm256_srai_epi32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn select(m: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_blendv_epi8(b, a, m)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn splat(v: i32) -> __m256i {
        _mm256_set1_epi32(v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn add(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn sub(a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn and(a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn or(a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn gt(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpgt_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn abs(v: __m256i) -> __m256i {
        _mm256_abs_epi32(v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn load_u8(p: *const u8) -> __m256i {
        _mm256_cvtepu8_epi32(unsafe { _mm_loadl_epi64(p.cast()) })
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn load_u16(p: *const u16) -> __m256i {
        _mm256_cvtepu16_epi32(unsafe { _mm_loadu_si128(p.cast()) })
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn shl<const N: i32>(v: __m256i) -> __m256i {
        _mm256_slli_epi32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn xor(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn min(a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn max(a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn lt_unsigned(a: __m256i, b: __m256i) -> __m256i {
        let sign = _mm256_set1_epi32(i32::MIN);
        _mm256_cmpgt_epi32(xor(b, sign), xor(a, sign))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn shr_unsigned<const N: i32>(v: __m256i) -> __m256i {
        _mm256_srli_epi32::<N>(v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn to_f32_bits(v: __m256i) -> __m256i {
        _mm256_castps_si256(_mm256_cvtepi32_ps(v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn load_pairs(p: *const i32) -> (__m256i, __m256i) {
        let a = _mm256_castsi256_ps(unsafe { load(p) });
        let b = _mm256_castsi256_ps(unsafe { load(p.add(8)) });
        let order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        (
            _mm256_permutevar8x32_epi32(
                _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(a, b)),
                order,
            ),
            _mm256_permutevar8x32_epi32(
                _mm256_castps_si256(_mm256_shuffle_ps::<0xdd>(a, b)),
                order,
            ),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn mulhi(a: __m256i, b: __m256i) -> __m256i {
        let even = _mm256_srli_epi64::<32>(_mm256_mul_epi32(a, b));
        let odd = _mm256_mul_epi32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(b));
        _mm256_blend_epi32::<0xaa>(even, odd)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn eq(a: __m256i, b: __m256i) -> __m256i {
        _mm256_cmpeq_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn bits_clear(v: __m256i, mask: __m256i) -> bool {
        _mm256_testz_si256(v, mask) != 0
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn mul(a: __m256i, b: __m256i) -> __m256i {
        _mm256_mullo_epi32(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn shr_by(v: __m256i, count: u32) -> __m256i {
        _mm256_sra_epi32(v, _mm_cvtsi32_si128(count as i32))
    }
}

pub(crate) mod f32 {
    use std::arch::x86_64::*;
    pub(crate) use std::arch::x86_64::{
        _mm256_add_ps as add, _mm256_div_ps as div, _mm256_loadu_ps as load, _mm256_max_ps as max,
        _mm256_min_ps as min, _mm256_mul_ps as mul, _mm256_set1_ps as splat,
        _mm256_storeu_ps as store, _mm256_sub_ps as sub,
    };

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn abs(v: __m256) -> __m256 {
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn mask_nonzero(v: __m256, a: __m256, b: __m256) -> __m256 {
        _mm256_and_ps(
            v,
            _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_NEQ_UQ>(a, _mm256_setzero_ps()),
                _mm256_cmp_ps::<_CMP_NEQ_UQ>(b, _mm256_setzero_ps()),
            ),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn from_i32(v: __m256i) -> __m256 {
        _mm256_cvtepi32_ps(v)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn trunc_to_i32(v: __m256) -> __m256i {
        _mm256_cvttps_epi32(v)
    }
}

macro_rules! add_scaled {
    ($name:ident, $features:literal, $update8:expr, $update4:expr, $update1:expr) => {
        #[target_feature(enable = $features)]
        pub(crate) fn $name(out: &mut [f32], src: &[f32], factor: f32) {
            use std::arch::x86_64::*;

            let src = &src[..out.len()];
            let scale = _mm256_set1_ps(factor);
            let scale4 = _mm256_castps256_ps128(scale);
            let update8 = |dst: &mut [core::primitive::f32; 8], src: &[core::primitive::f32; 8]| {
                let d = dst.as_mut_ptr();
                // SAFETY: both arrays contain eight elements.
                unsafe {
                    let x = _mm256_loadu_ps(src.as_ptr());
                    _mm256_storeu_ps(d, $update8(scale, x, _mm256_loadu_ps(d)));
                }
            };
            let (out_blocks, out) = out.as_chunks_mut::<8>();
            let (src_blocks, src) = src.as_chunks::<8>();
            let (out_pairs, out_blocks) = out_blocks.as_chunks_mut::<2>();
            let (src_pairs, src_blocks) = src_blocks.as_chunks::<2>();
            for ([dst0, dst1], [src0, src1]) in out_pairs.iter_mut().zip(src_pairs) {
                update8(dst0, src0);
                update8(dst1, src1);
            }
            for (dst, src) in out_blocks.iter_mut().zip(src_blocks) {
                update8(dst, src);
            }
            let (out_blocks, out) = out.as_chunks_mut::<4>();
            let (src_blocks, src) = src.as_chunks::<4>();
            for (dst, src) in out_blocks.iter_mut().zip(src_blocks) {
                let d = dst.as_mut_ptr();
                // SAFETY: both arrays contain four elements.
                unsafe {
                    let x = _mm_loadu_ps(src.as_ptr());
                    _mm_storeu_ps(d, $update4(scale4, x, _mm_loadu_ps(d)));
                }
            }
            for (dst, &input) in out.iter_mut().zip(src) {
                *dst = $update1(factor, input, *dst);
            }
        }
    };
}

add_scaled!(
    add_scaled_avx2,
    "avx2",
    |scale, src, dst| _mm256_add_ps(dst, _mm256_mul_ps(scale, src)),
    |scale, src, dst| _mm_add_ps(dst, _mm_mul_ps(scale, src)),
    |scale, src, dst| dst + scale * src
);

#[cfg(feature = "splines")]
add_scaled!(
    add_scaled_fma_avx2,
    "avx2,fma",
    _mm256_fmadd_ps,
    _mm_fmadd_ps,
    core::primitive::f32::mul_add
);
