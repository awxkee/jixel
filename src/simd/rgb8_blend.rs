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

//! Coverage compositing in interleaved RGB8.

use std::sync::OnceLock;

type BlendFn = unsafe fn(&mut [[u8; 3]], &[[u8; 3]], &[u8], Option<u8>) -> usize;

/// Blend the layer over the destination with one coverage byte per pixel.
pub(crate) fn blend_in_place(dst: &mut [[u8; 3]], layer: &[[u8; 3]], coverage: &[u8]) {
    blend(dst, layer, coverage, None);
}

/// Blend the layer over a uniform gray background.
#[cfg(feature = "extraslow")]
pub(crate) fn blend_over_gray(dst: &mut [[u8; 3]], layer: &[[u8; 3]], coverage: &[u8], gray: u8) {
    blend(dst, layer, coverage, Some(gray));
}

fn blend(dst: &mut [[u8; 3]], layer: &[[u8; 3]], coverage: &[u8], background: Option<u8>) {
    assert_eq!(dst.len(), layer.len());
    assert_eq!(dst.len(), coverage.len());
    static BLEND: OnceLock<BlendFn> = OnceLock::new();
    let kernel = *BLEND.get_or_init(select);
    // SAFETY: dispatch selects a supported ISA and the three slices have
    // equal pixel counts. Each SIMD load and store stays inside a full block.
    let done = unsafe { kernel(dst, layer, coverage, background) };
    for ((dst, &layer), &alpha) in dst[done..]
        .iter_mut()
        .zip(&layer[done..])
        .zip(&coverage[done..])
    {
        let base = match background {
            None => *dst,
            Some(gray) => [gray; 3],
        };
        let a = u16::from(alpha);
        for ((dst, b), l) in dst.iter_mut().zip(base).zip(layer) {
            *dst = ((u16::from(b) * (255 - a) + u16::from(l) * a + 127) / 255) as u8;
        }
    }
}

fn select() -> BlendFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        return avx::blend;
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return sse::blend;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return neon::blend;
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return wasm::blend;
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    |_, _, _, _| 0
}

#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
))]
const COVERAGE_SHUFFLES: [[u8; 16]; 3] = {
    // The same alpha feeds each of a pixel's three consecutive RGB bytes.
    let mut indices = [[0; 16]; 3];
    let mut i = 0;
    while i < 48 {
        indices[i / 16][i % 16] = (i / 3) as u8;
        i += 1;
    }
    indices
};

#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
))]
macro_rules! packed_backend {
    ($feature:literal) => {
        #[target_feature(enable = $feature)]
        pub(super) fn blend(
            dst: &mut [[u8; 3]],
            layer: &[[u8; 3]],
            coverage: &[u8],
            background: Option<u8>,
        ) -> usize {
            let solid = background.map(|gray| splat(gray));
            let dst = dst.as_chunks_mut::<16>().0;
            let layer = layer.as_chunks::<16>().0;
            let coverage = coverage.as_chunks::<16>().0;
            let masks = super::COVERAGE_SHUFFLES.each_ref().map(|mask| load(mask));
            for ((dst, layer), coverage) in dst.iter_mut().zip(layer).zip(coverage) {
                let alpha = load(coverage);
                let dst = dst.as_flattened_mut().as_chunks_mut::<16>().0;
                let layer = layer.as_flattened().as_chunks::<16>().0;
                for ((dst, layer), mask) in dst.iter_mut().zip(layer).zip(masks) {
                    let base = solid.unwrap_or_else(|| load(dst));
                    let coverage = shuffle(alpha, mask);
                    store(dst, blend_bytes(base, load(layer), coverage));
                }
            }
            dst.len() * 16
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    fn blend_bytes(b: uint8x16_t, l: uint8x16_t, a: uint8x16_t, inv: uint8x16_t) -> uint8x16_t {
        let half = vdupq_n_u16(128);
        let lo = vmlal_u8(
            vmlal_u8(half, vget_low_u8(b), vget_low_u8(inv)),
            vget_low_u8(l),
            vget_low_u8(a),
        );
        let hi = vmlal_high_u8(vmlal_high_u8(half, b, inv), l, a);
        // For n <= 255 * 255, round(n / 255) = (p + (p >> 8)) >> 8, p = n + 128.
        let lo = vaddq_u16(lo, vshrq_n_u16::<8>(lo));
        let hi = vaddq_u16(hi, vshrq_n_u16::<8>(hi));
        vshrn_high_n_u16::<8>(vshrn_n_u16::<8>(lo), hi)
    }

    #[target_feature(enable = "neon")]
    pub(super) fn blend(
        dst: &mut [[u8; 3]],
        layer: &[[u8; 3]],
        coverage: &[u8],
        background: Option<u8>,
    ) -> usize {
        let solid = match background {
            None => None,
            Some(gray) => {
                let v = vdupq_n_u8(gray);
                Some(uint8x16x3_t(v, v, v))
            }
        };
        let dst = dst.as_chunks_mut::<16>().0;
        let layer = layer.as_chunks::<16>().0;
        let coverage = coverage.as_chunks::<16>().0;
        for ((dst, layer), alpha) in dst.iter_mut().zip(layer).zip(coverage) {
            let b = solid.unwrap_or_else(|| unsafe { vld3q_u8(dst.as_ptr().cast()) });
            let l = unsafe { vld3q_u8(layer.as_ptr().cast()) };
            let a = unsafe { vld1q_u8(alpha.as_ptr()) };
            let inv = vmvnq_u8(a);
            let blended = uint8x16x3_t(
                blend_bytes(b.0, l.0, a, inv),
                blend_bytes(b.1, l.1, a, inv),
                blend_bytes(b.2, l.2, a, inv),
            );
            unsafe { vst3q_u8(dst.as_mut_ptr().cast(), blended) };
        }
        dst.len() * 16
    }
}

#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse")
))]
mod x86_bytes {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "ssse3")]
    pub(super) fn load(bytes: &[u8; 16]) -> __m128i {
        unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    pub(super) fn store(bytes: &mut [u8; 16], value: __m128i) {
        unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), value) };
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    pub(super) fn splat(value: u8) -> __m128i {
        _mm_set1_epi8(value as i8)
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    pub(super) fn shuffle(alpha: __m128i, mask: __m128i) -> __m128i {
        _mm_shuffle_epi8(alpha, mask)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use super::x86_bytes::*;
    use std::arch::x86_64::*;

    packed_backend!("avx2");

    #[inline]
    #[target_feature(enable = "avx2")]
    fn blend_bytes(base: __m128i, layer: __m128i, alpha: __m128i) -> __m128i {
        let a = _mm256_cvtepu8_epi16(alpha);
        let inv = _mm256_xor_si256(a, _mm256_set1_epi16(255));
        let b = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(base), inv);
        let l = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(layer), a);
        let p = _mm256_add_epi16(_mm256_add_epi16(b, l), _mm256_set1_epi16(128));
        let result = _mm256_srli_epi16::<8>(_mm256_add_epi16(p, _mm256_srli_epi16::<8>(p)));
        _mm_packus_epi16(
            _mm256_castsi256_si128(result),
            _mm256_extracti128_si256::<1>(result),
        )
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse {
    use super::x86_bytes::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    packed_backend!("sse4.1");

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn blend_bytes(b: __m128i, l: __m128i, a: __m128i) -> __m128i {
        let zero = _mm_setzero_si128();
        let part = |b, l, a| {
            let inv = _mm_xor_si128(a, _mm_set1_epi16(255));
            let sum = _mm_add_epi16(_mm_mullo_epi16(b, inv), _mm_mullo_epi16(l, a));
            let p = _mm_add_epi16(sum, _mm_set1_epi16(128));
            _mm_srli_epi16::<8>(_mm_add_epi16(p, _mm_srli_epi16::<8>(p)))
        };
        _mm_packus_epi16(
            part(
                _mm_unpacklo_epi8(b, zero),
                _mm_unpacklo_epi8(l, zero),
                _mm_unpacklo_epi8(a, zero),
            ),
            part(
                _mm_unpackhi_epi8(b, zero),
                _mm_unpackhi_epi8(l, zero),
                _mm_unpackhi_epi8(a, zero),
            ),
        )
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;

    packed_backend!("simd128");

    #[inline]
    fn load(bytes: &[u8; 16]) -> v128 {
        unsafe { v128_load(bytes.as_ptr().cast()) }
    }

    #[inline]
    fn store(bytes: &mut [u8; 16], value: v128) {
        unsafe { v128_store(bytes.as_mut_ptr().cast(), value) };
    }

    use std::arch::wasm32::{i8x16_swizzle as shuffle, u8x16_splat as splat};

    #[inline]
    fn blend_bytes(b: v128, l: v128, a: v128) -> v128 {
        let part = |b, l, a| {
            let inv = v128_xor(a, u16x8_splat(255));
            let sum = u16x8_add(u16x8_mul(b, inv), u16x8_mul(l, a));
            let p = u16x8_add(sum, u16x8_splat(128));
            u16x8_shr(u16x8_add(p, u16x8_shr(p, 8)), 8)
        };
        u8x16_narrow_i16x8(
            part(
                u16x8_extend_low_u8x16(b),
                u16x8_extend_low_u8x16(l),
                u16x8_extend_low_u8x16(a),
            ),
            part(
                u16x8_extend_high_u8x16(b),
                u16x8_extend_high_u8x16(l),
                u16x8_extend_high_u8x16(a),
            ),
        )
    }
}
