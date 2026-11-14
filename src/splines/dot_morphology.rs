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

//! Separable seven-sample extrema for the spline dot detector.

pub(super) const R: usize = 3;
const WINDOW: usize = 2 * R + 1;

pub(super) type RowExtremaFn = fn(&[f32], &mut [f32], bool);
/// Reduce packed source rows to a single output row of the same width.
pub(super) type ColumnExtremaFn = fn(&[f32], &mut [f32], bool);

#[inline]
fn pick<const MAX: bool>(a: f32, b: f32) -> f32 {
    if MAX { a.max(b) } else { a.min(b) }
}

fn row_edges<'a, const MAX: bool>(src: &[f32], out: &'a mut [f32]) -> &'a mut [f32] {
    assert_eq!(src.len(), out.len());
    let left = R.min(src.len());
    let right = left.max(src.len().saturating_sub(R));
    let (prefix, rest) = out.split_at_mut(left);
    let (interior, suffix) = rest.split_at_mut(right - left);
    // Padding is neutral unless every sample in a clipped window is NaN.
    let fill = if MAX {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    for (x, dst) in prefix.iter_mut().enumerate().chain((right..).zip(suffix)) {
        let lo = x.saturating_sub(R);
        let hi = (x + R + 1).min(src.len());
        *dst = src[lo..hi].iter().copied().fold(fill, pick::<MAX>);
    }
    interior
}

#[inline]
fn row_interior<const MAX: bool>(src: &[f32], out: &mut [f32]) {
    for (dst, &[a, b, c, d, e, f, g]) in out.iter_mut().zip(src.array_windows::<WINDOW>()) {
        let left = pick::<MAX>(pick::<MAX>(a, b), pick::<MAX>(c, d));
        let right = pick::<MAX>(pick::<MAX>(d, e), pick::<MAX>(f, g));
        *dst = pick::<MAX>(left, right);
    }
}

pub(super) fn row_scalar(src: &[f32], out: &mut [f32], max: bool) {
    if max {
        row_interior::<true>(src, row_edges::<true>(src, out));
    } else {
        row_interior::<false>(src, row_edges::<false>(src, out));
    }
}

fn column_tail<const MAX: bool>(rows: &[f32], first: &[f32], x: usize, tail: &mut [f32]) {
    let w = first.len();
    tail.copy_from_slice(&first[x..]);
    for row in rows.chunks_exact(w) {
        for (dst, &v) in tail.iter_mut().zip(&row[x..]) {
            *dst = pick::<MAX>(*dst, v);
        }
    }
}

pub(super) fn column_scalar(rows: &[f32], out: &mut [f32], max: bool) {
    if out.is_empty() {
        return;
    }
    let (first, rows) = rows.split_at(out.len());
    if max {
        column_tail::<true>(rows, first, 0, out);
    } else {
        column_tail::<false>(rows, first, 0, out);
    }
}

#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
))]
macro_rules! extrema_kernels {
    ($feature:literal, $lanes:literal, $group:literal) => {
        #[target_feature(enable = $feature)]
        pub(in super::super) fn row(src: &[f32], out: &mut [f32], max: bool) {
            if max {
                row_simd::<true>(src, out);
            } else {
                row_simd::<false>(src, out);
            }
        }

        #[target_feature(enable = $feature)]
        fn row_simd<const MAX: bool>(src: &[f32], out: &mut [f32]) {
            let out = super::row_edges::<MAX>(src, out);
            let done = row_blocks::<MAX>(src, out);
            super::row_interior::<MAX>(&src[done..], &mut out[done..]);
        }

        #[target_feature(enable = $feature)]
        pub(in super::super) fn column(rows: &[f32], out: &mut [f32], max: bool) {
            if max {
                column_simd::<true>(rows, out);
            } else {
                column_simd::<false>(rows, out);
            }
        }

        #[target_feature(enable = $feature)]
        fn column_simd<const MAX: bool>(rows: &[f32], out: &mut [f32]) {
            let w = out.len();
            if w == 0 {
                return;
            }
            let (first, rows) = rows.split_at(w);
            let (blocks, tail) = out.as_chunks_mut::<$lanes>();
            let (groups, blocks) = blocks.as_chunks_mut::<$group>();
            let mut x = 0;
            for group in groups {
                column_block::<MAX, $group>(first, rows, x, group);
                x += $lanes * $group;
            }
            for block in blocks {
                column_block::<MAX, 1>(first, rows, x, std::array::from_mut(block));
                x += $lanes;
            }
            if !tail.is_empty() {
                super::column_tail::<MAX>(rows, first, x, tail);
            }
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn column_block<const MAX: bool, const N: usize>(
            first: &[f32],
            rows: &[f32],
            x: usize,
            out: &mut [[f32; $lanes]; N],
        ) {
            let w = first.len();
            // SAFETY: the caller processes only full output blocks within w,
            // and every complete source row contains w floats.
            unsafe {
                let p = first.as_ptr().add(x);
                let mut values: [_; N] = std::array::from_fn(|i| load(p.add(i * $lanes)));
                for row in rows.chunks_exact(w) {
                    let p = row.as_ptr().add(x);
                    for (i, v) in values.iter_mut().enumerate() {
                        *v = pick::<MAX>(*v, load(p.add(i * $lanes)));
                    }
                }
                for (block, v) in out.iter_mut().zip(values) {
                    store(block.as_mut_ptr(), v);
                }
            }
        }
    };
}

#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
))]
macro_rules! shifted_row_blocks {
    ($feature:literal, $lanes:literal) => {
        #[target_feature(enable = $feature)]
        fn row_blocks<const MAX: bool>(src: &[f32], out: &mut [f32]) -> usize {
            let (blocks, _) = out.as_chunks_mut::<$lanes>();
            let mut x = 0;
            for block in blocks {
                // SAFETY: a block of interior outputs has six additional readable
                // samples in src, so every shifted load stays within the source.
                unsafe {
                    let p = src.as_ptr().add(x);
                    let d = load(p.add(3));
                    let ab = pick::<MAX>(load(p), load(p.add(1)));
                    let cd = pick::<MAX>(load(p.add(2)), d);
                    let de = pick::<MAX>(d, load(p.add(4)));
                    let fg = pick::<MAX>(load(p.add(5)), load(p.add(6)));
                    let value = pick::<MAX>(pick::<MAX>(ab, cd), pick::<MAX>(de, fg));
                    store(block.as_mut_ptr(), value);
                }
                x += $lanes;
            }
            x
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(super) mod neon {
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(p: *const f32) -> float32x4_t {
        let mut v = unsafe { vld1q_f32(p) };
        // Self-min quiets signaling NaNs before the number min/max reduction.
        // LLVM folds the equivalent intrinsic to its input.
        unsafe {
            std::arch::asm!(
                "fminnm {v:v}.4s, {v:v}.4s, {v:v}.4s",
                v = inout(vreg) v,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        v
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn pick<const MAX: bool>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        if MAX {
            vmaxnmq_f32(a, b)
        } else {
            vminnmq_f32(a, b)
        }
    }

    use std::arch::aarch64::vst1q_f32 as store;
    extrema_kernels!("neon", 4, 4);

    #[target_feature(enable = "neon")]
    fn row_blocks<const MAX: bool>(src: &[f32], out: &mut [f32]) -> usize {
        let (blocks, _) = out.as_chunks_mut::<4>();
        let count = blocks.len().min((src.len() / 4).saturating_sub(3));
        if count == 0 {
            return 0;
        }
        // Carry the two- and four-sample extrema into the next block.
        // Each iteration loads four new samples to finish four outputs.
        // SAFETY: count accounts for the three initial vectors and bounds
        // each subsequent input load and output store.
        unsafe {
            let p = src.as_ptr();
            let a = load(p);
            let b = load(p.add(4));
            let mut c = load(p.add(8));
            let ab = pick::<MAX>(a, vextq_f32::<1>(a, b));
            let mut bc = pick::<MAX>(b, vextq_f32::<1>(b, c));
            let mut left = pick::<MAX>(ab, vextq_f32::<2>(ab, bc));
            for (i, block) in blocks[..count].iter_mut().enumerate() {
                let next = load(p.add(4 * i + 12));
                let pair = pick::<MAX>(c, vextq_f32::<1>(c, next));
                let right = pick::<MAX>(bc, vextq_f32::<2>(bc, pair));
                let value = pick::<MAX>(left, vextq_f32::<3>(left, right));
                store(block.as_mut_ptr(), value);
                c = next;
                bc = pair;
                left = right;
            }
        }
        count * 4
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub(super) mod avx {
    use std::arch::x86_64::*;
    use std::arch::x86_64::{_mm256_loadu_ps as load, _mm256_storeu_ps as store};

    #[inline]
    #[target_feature(enable = "avx2")]
    fn pick<const MAX: bool>(a: __m256, b: __m256) -> __m256 {
        let value = if MAX {
            _mm256_max_ps(a, b)
        } else {
            _mm256_min_ps(a, b)
        };
        // x86 min/max return the second operand for unordered lanes.
        _mm256_blendv_ps(value, a, _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b))
    }

    extrema_kernels!("avx2", 8, 2);
    shifted_row_blocks!("avx2", 8);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
pub(super) mod sse {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_loadu_ps as load, _mm_storeu_ps as store, *};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_loadu_ps as load, _mm_storeu_ps as store, *};

    #[inline]
    #[target_feature(enable = "sse2")]
    fn pick<const MAX: bool>(a: __m128, b: __m128) -> __m128 {
        let value = if MAX {
            _mm_max_ps(a, b)
        } else {
            _mm_min_ps(a, b)
        };
        let unordered = _mm_cmpunord_ps(b, b);
        _mm_or_ps(_mm_and_ps(unordered, a), _mm_andnot_ps(unordered, value))
    }

    extrema_kernels!("sse2", 4, 4);
    shifted_row_blocks!("sse2", 4);
}

#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
pub(super) mod wasm {
    use std::arch::wasm32::*;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn load(p: *const f32) -> v128 {
        unsafe { v128_load(p.cast()) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn store(p: *mut f32, v: v128) {
        unsafe { v128_store(p.cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    fn pick<const MAX: bool>(a: v128, b: v128) -> v128 {
        let value = if MAX {
            f32x4_pmax(a, b)
        } else {
            f32x4_pmin(a, b)
        };
        // Pseudo min/max return the first operand for unordered lanes.
        v128_bitselect(b, value, f32x4_ne(a, a))
    }

    extrema_kernels!("simd128", 4, 4);
    shifted_row_blocks!("simd128", 4);
}
