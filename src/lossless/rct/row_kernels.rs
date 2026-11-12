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

//! Dispatched planar and packed RGB kernels for modular RCT 6.

use super::forward_ycocg;
use std::sync::OnceLock;

type ForwardPlanes = unsafe fn([&[i32]; 3], [&mut [i32]; 3]);
type ForwardInPlace = unsafe fn([&mut [i32]; 3]);
type ForwardRgb8 = unsafe fn(&[[u8; 3]], [&mut [i32]; 3]);

struct Kernels {
    planes: ForwardPlanes,
    in_place: ForwardInPlace,
    rgb8: ForwardRgb8,
}

fn selected() -> &'static Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    KERNELS.get_or_init(select)
}

fn select() -> Kernels {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        return avx::kernels();
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return sse::kernels();
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return neon::kernels();
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return wasm::kernels();
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    Kernels {
        planes: planes_scalar,
        in_place: in_place_scalar,
        rgb8: rgb8_scalar,
    }
}

pub(crate) fn forward_ycocg_planes(src: [&[i32]; 3], dst: [&mut [i32]; 3]) {
    assert!(src.iter().all(|p| p.len() == dst[0].len()));
    assert!(dst.iter().all(|p| p.len() == src[0].len()));
    // SAFETY: dispatch selects a supported ISA and all slices have equal lengths.
    unsafe { (selected().planes)(src, dst) };
}

pub(crate) fn forward_ycocg_in_place(planes: [&mut [i32]; 3]) {
    assert!(planes.iter().all(|p| p.len() == planes[0].len()));
    // SAFETY: dispatch selects a supported ISA and all slices have equal lengths.
    unsafe { (selected().in_place)(planes) };
}

pub(crate) fn rgb8_to_ycocg(src: &[[u8; 3]], dst: [&mut [i32]; 3]) {
    assert!(dst.iter().all(|p| p.len() == src.len()));
    // SAFETY: dispatch selects a supported ISA and the source has a pixel for
    // every element in each destination plane.
    unsafe { (selected().rgb8)(src, dst) };
}

fn planes_scalar([r, g, b]: [&[i32]; 3], [y, co, cg]: [&mut [i32]; 3]) {
    let src = r.iter().zip(g).zip(b);
    for (((&r, &g), &b), ((y, co), cg)) in src.zip(y.iter_mut().zip(co).zip(cg)) {
        (*y, *co, *cg) = forward_ycocg(r, g, b);
    }
}

fn in_place_scalar([r, g, b]: [&mut [i32]; 3]) {
    for ((r, g), b) in r.iter_mut().zip(g).zip(b) {
        (*r, *g, *b) = forward_ycocg(*r, *g, *b);
    }
}

fn rgb8_scalar(src: &[[u8; 3]], [y, co, cg]: [&mut [i32]; 3]) {
    for (&pixel, ((y, co), cg)) in src.iter().zip(y.iter_mut().zip(co).zip(cg)) {
        let [r, g, b] = pixel.map(i32::from);
        (*y, *co, *cg) = forward_ycocg(r, g, b);
    }
}

#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
))]
macro_rules! backend {
    ($feature:literal, $lanes:literal, [$($offset:literal),+]) => {
        pub(super) fn kernels() -> super::Kernels {
            super::Kernels { planes, in_place, rgb8 }
        }

        #[inline]
        #[target_feature(enable = $feature)]
        pub(crate) fn forward_vec(r: Vector, g: Vector, b: Vector) -> [Vector; 3] {
            let co = sub(r, b);
            let tmp = add(b, shr::<1>(co));
            let cg = sub(g, tmp);
            [add(tmp, shr::<1>(cg)), co, cg]
        }

        #[target_feature(enable = $feature)]
        unsafe fn planes(src: [&[i32]; 3], dst: [&mut [i32]; 3]) {
            let [y, co, cg] = dst;
            let end = y.len() / $lanes * $lanes;
            for i in (0..end).step_by($lanes) {
                let [r, g, b] = src.map(|p| unsafe { load(p.as_ptr().add(i)) });
                let [luma, orange, green] = forward_vec(r, g, b);
                unsafe {
                    store(y.as_mut_ptr().add(i), luma);
                    store(co.as_mut_ptr().add(i), orange);
                    store(cg.as_mut_ptr().add(i), green);
                }
            }
            super::planes_scalar(src.map(|p| &p[end..]), [&mut y[end..], &mut co[end..], &mut cg[end..]]);
        }

        #[target_feature(enable = $feature)]
        unsafe fn in_place([r, g, b]: [&mut [i32]; 3]) {
            let end = r.len() / $lanes * $lanes;
            for i in (0..end).step_by($lanes) {
                let ptrs = [&mut *r, &mut *g, &mut *b].map(|p| unsafe { p.as_mut_ptr().add(i) });
                let [rv, gv, bv] = ptrs.map(|p| unsafe { load(p) });
                for (p, value) in ptrs.into_iter().zip(forward_vec(rv, gv, bv)) {
                    unsafe { store(p, value) };
                }
            }
            super::in_place_scalar([&mut r[end..], &mut g[end..], &mut b[end..]]);
        }

        #[target_feature(enable = $feature)]
        unsafe fn rgb8(src: &[[u8; 3]], [y, co, cg]: [&mut [i32]; 3]) {
            let (blocks, tail) = src.as_chunks::<16>();
            let end = blocks.len() * 16;
            for (block, pixels) in blocks.iter().enumerate() {
                let channels = deinterleave(pixels);
                let i = block * 16;
                $({
                    let [r, g, b] = channels.map(|channel| widen::<$offset>(channel));
                    let [luma, orange, green] = forward_vec(r, g, b);
                    unsafe {
                        store(y.as_mut_ptr().add(i + $offset), luma);
                        store(co.as_mut_ptr().add(i + $offset), orange);
                        store(cg.as_mut_ptr().add(i + $offset), green);
                    }
                })+
            }
            super::rgb8_scalar(tail, [&mut y[end..], &mut co[end..], &mut cg[end..]]);
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) mod neon {
    use crate::simd::neon::i32::*;
    use std::arch::aarch64::*;

    backend!("neon", 4, [0, 4, 8, 12]);

    #[inline]
    #[target_feature(enable = "neon")]
    fn deinterleave(pixels: &[[u8; 3]; 16]) -> [uint8x16_t; 3] {
        let uint8x16x3_t(r, g, b) = unsafe { vld3q_u8(pixels.as_ptr().cast()) };
        [r, g, b]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn widen<const LANE: usize>(channel: uint8x16_t) -> Vector {
        let half = if LANE < 8 {
            vmovl_u8(vget_low_u8(channel))
        } else {
            vmovl_high_u8(channel)
        };
        let words = if LANE.is_multiple_of(8) {
            vmovl_u16(vget_low_u16(half))
        } else {
            vmovl_high_u16(half)
        };
        vreinterpretq_s32_u32(words)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub(crate) mod avx {
    use crate::simd::avx::i32::*;
    use crate::simd::sse::deinterleave_rgb8 as deinterleave;
    use std::arch::x86_64::*;

    backend!("avx2", 8, [0, 8]);

    #[inline]
    #[target_feature(enable = "avx2")]
    fn widen<const LANE: usize>(channel: __m128i) -> Vector {
        let bytes = if LANE == 0 {
            channel
        } else {
            _mm_srli_si128::<8>(channel)
        };
        _mm256_cvtepu8_epi32(bytes)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
pub(crate) mod sse {
    use crate::simd::sse::deinterleave_rgb8 as deinterleave;
    use crate::simd::sse::i32::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    backend!("sse4.1", 4, [0, 4, 8, 12]);

    #[inline]
    #[target_feature(enable = "sse4.1")]
    fn widen<const LANE: usize>(channel: __m128i) -> Vector {
        let bytes = match LANE {
            0 => channel,
            4 => _mm_srli_si128::<4>(channel),
            8 => _mm_srli_si128::<8>(channel),
            _ => _mm_srli_si128::<12>(channel),
        };
        _mm_cvtepu8_epi32(bytes)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
pub(crate) mod wasm {
    use crate::simd::wasm::deinterleave_rgb8 as deinterleave;
    use crate::simd::wasm::i32::*;
    use std::arch::wasm32::*;

    backend!("simd128", 4, [0, 4, 8, 12]);

    #[inline]
    #[target_feature(enable = "simd128")]
    fn widen<const LANE: usize>(channel: v128) -> Vector {
        let half = if LANE < 8 {
            u16x8_extend_low_u8x16(channel)
        } else {
            u16x8_extend_high_u8x16(channel)
        };
        if LANE.is_multiple_of(8) {
            u32x4_extend_low_u16x8(half)
        } else {
            u32x4_extend_high_u16x8(half)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_and_planar_ycocg_match_scalar_with_tails() {
        let bytes: Vec<_> = (0..140)
            .map(|i| [(i * 73) as u8, (i * 137 + 19) as u8, (i * 53 + 251) as u8])
            .collect();
        let signed: [Vec<_>; 3] = std::array::from_fn(|c| {
            (0..bytes.len())
                .map(|i| i32::from(bytes[i][c]) * 511 - 65_535)
                .collect()
        });
        const CANARY: i32 = i32::MIN;
        for len in 0..=bytes.len() - 2 {
            let range = 1..1 + len;
            let mut packed = std::array::from_fn(|_| vec![CANARY; len + 2]);
            rgb8_to_ycocg(
                &bytes[range.clone()],
                packed.each_mut().map(|p| &mut p[range.clone()]),
            );
            let src = signed.each_ref().map(|p| &p[range.clone()]);
            let mut planar = std::array::from_fn(|_| vec![CANARY; len + 2]);
            forward_ycocg_planes(src, planar.each_mut().map(|p| &mut p[range.clone()]));
            let mut in_place = signed.clone();
            forward_ycocg_in_place(in_place.each_mut().map(|p| &mut p[range.clone()]));
            for i in range {
                let [r, g, b] = bytes[i].map(i32::from);
                let (y, co, cg) = forward_ycocg(r, g, b);
                assert_eq!(packed.each_ref().map(|p| p[i]), [y, co, cg]);
                let [r, g, b] = signed.each_ref().map(|p| p[i]);
                let (y, co, cg) = forward_ycocg(r, g, b);
                assert_eq!(planar.each_ref().map(|p| p[i]), [y, co, cg]);
                assert_eq!(in_place.each_ref().map(|p| p[i]), [y, co, cg]);
            }
            for p in packed.iter().chain(&planar) {
                assert_eq!((p[0], p[len + 1]), (CANARY, CANARY));
            }
            for (p, original) in in_place.iter().zip(&signed) {
                assert_eq!((p[0], p[len + 1]), (original[0], original[len + 1]));
            }
        }
    }
}
