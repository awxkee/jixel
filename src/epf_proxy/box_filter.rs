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

//! Streaming reflected 5x5 moments. SIMD lanes follow independent columns
//! vertically and independent rows horizontally, preserving both f32 recurrences.

const ROWS: usize = 4;
type Accumulate = fn([&[f32]; 2], [&mut [f32]; 3]);
type Step = fn([&[f32]; 2], [&[f32]; 2], [&mut [f32]; 3], [&mut [f32]; 3]);
type Horizontal = fn(&[f32], usize, &mut [f32]);

#[derive(Clone, Copy)]
struct MomentKernels {
    accumulate: Accumulate,
    advance: Step,
    finish: Step,
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    reference: MomentKernels,
    comparison: MomentKernels,
    horizontal: Horizontal,
}

pub(super) fn selected_kernels() -> Kernels {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return avx::kernels();
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse2") {
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
    scalar_kernels()
}

#[derive(Default)]
pub(super) struct Scratch {
    // Three vertical sums plus four rows per moment on either side of the
    // horizontal pass. Storage follows the widest row, never image area.
    sums: [Vec<f32>; 3],
    vertical: [Vec<f32>; 3],
    output: [Vec<f32>; 3],
}

impl Scratch {
    /// `CROSS` adds E[reference * test] to E[test] and E[test^2]. Products
    /// are formed in registers as rows enter/leave the vertical window.
    pub(super) fn run<const CROSS: bool>(
        &mut self,
        kernels: Kernels,
        test: &[f32],
        reference: &[f32],
        w: usize,
        h: usize,
        mut consume: impl FnMut(usize, [&[f32]; 3]),
    ) {
        assert!(w > 0 && h > 0);
        assert_eq!(test.len(), w * h);
        if CROSS {
            assert_eq!(reference.len(), test.len());
        }
        let count = if CROSS { 3 } else { 2 };
        let k = if CROSS {
            kernels.comparison
        } else {
            kernels.reference
        };
        for c in 0..count {
            // The quarter scale must not truncate these buffers: otherwise
            // the next half scale would zero-fill memory it immediately overwrites.
            for (buffer, len) in [
                (&mut self.sums[c], w),
                (&mut self.vertical[c], w * ROWS.min(h)),
                (&mut self.output[c], w * ROWS.min(h)),
            ] {
                if buffer.len() < len {
                    buffer.resize(len, 0.0);
                }
            }
            self.sums[c][..w].fill(-0.0);
        }
        let input = |y| {
            let start = super::mirror(y, h) * w;
            [
                &test[start..start + w],
                if CROSS {
                    &reference[start..start + w]
                } else {
                    &[]
                },
            ]
        };
        for y in -2..=2 {
            (k.accumulate)(input(y), moment_rows::<CROSS>(&mut self.sums, 0, w));
        }
        for first in (0..h).step_by(ROWS) {
            let rows = ROWS.min(h - first);
            for r in 0..rows {
                let y = first + r;
                let (step, enter, leave) = if y + 1 < h {
                    (k.advance, input(y as isize + 3), input(y as isize - 2))
                } else {
                    (k.finish, [&[][..]; 2], [&[][..]; 2])
                };
                step(
                    enter,
                    leave,
                    moment_rows::<CROSS>(&mut self.sums, 0, w),
                    moment_rows::<CROSS>(&mut self.vertical, r, w),
                );
            }
            for c in 0..count {
                let src = &self.vertical[c][..rows * w];
                let dst = &mut self.output[c][..rows * w];
                if rows == ROWS {
                    (kernels.horizontal)(src, w, dst);
                } else {
                    horizontal_scalar(src, w, dst);
                }
            }
            for r in 0..rows {
                consume(
                    first + r,
                    std::array::from_fn(|c| {
                        if c < count {
                            &self.output[c][r * w..(r + 1) * w]
                        } else {
                            &[]
                        }
                    }),
                );
            }
        }
    }
}

fn moment_rows<const CROSS: bool>(
    buffers: &mut [Vec<f32>; 3],
    row: usize,
    w: usize,
) -> [&mut [f32]; 3] {
    let [mean, square, cross] = buffers;
    let start = row * w;
    [
        &mut mean[start..start + w],
        &mut square[start..start + w],
        if CROSS {
            &mut cross[start..start + w]
        } else {
            &mut []
        },
    ]
}

fn accumulate_scalar<const CROSS: bool>(src: [&[f32]; 2], sums: [&mut [f32]; 3]) {
    let [mean, square, cross] = sums;
    for x in 0..mean.len() {
        let v = src[0][x];
        mean[x] += v;
        square[x] += v * v;
        if CROSS {
            cross[x] += src[1][x] * v;
        }
    }
}

fn step_scalar<const CROSS: bool, const UPDATE: bool>(
    enter: [&[f32]; 2],
    leave: [&[f32]; 2],
    sums: [&mut [f32]; 3],
    output: [&mut [f32]; 3],
) {
    let [mean, square, cross] = sums;
    let [out_mean, out_square, out_cross] = output;
    for x in 0..mean.len() {
        out_mean[x] = mean[x] * 0.2;
        out_square[x] = square[x] * 0.2;
        if CROSS {
            out_cross[x] = cross[x] * 0.2;
        }
        if UPDATE {
            let (a, b) = (enter[0][x], leave[0][x]);
            mean[x] += a - b;
            square[x] += a * a - b * b;
            if CROSS {
                cross[x] += enter[1][x] * a - leave[1][x] * b;
            }
        }
    }
}

fn horizontal_scalar(src: &[f32], w: usize, dst: &mut [f32]) {
    for (row, dst) in src.chunks_exact(w).zip(dst.chunks_exact_mut(w)) {
        let mut sum: f32 = (-2..=2).map(|x| row[super::mirror(x, w)]).sum();
        for x in 0..w {
            dst[x] = sum * 0.2;
            if x + 1 < w {
                sum +=
                    row[super::mirror(x as isize + 3, w)] - row[super::mirror(x as isize - 2, w)];
            }
        }
    }
}

#[allow(dead_code)]
fn scalar_kernels() -> Kernels {
    Kernels {
        reference: MomentKernels {
            accumulate: accumulate_scalar::<false>,
            advance: step_scalar::<false, true>,
            finish: step_scalar::<false, false>,
        },
        comparison: MomentKernels {
            accumulate: accumulate_scalar::<true>,
            advance: step_scalar::<true, true>,
            finish: step_scalar::<true, false>,
        },
        horizontal: horizontal_scalar,
    }
}

// Only the vector operations differ between backends. Keep the moment formulas
// and their scalar tails in one place; never reassociate or fuse these sums.
#[allow(unused_macros)]
macro_rules! moments {
    ($feature:literal, $lanes:literal, $load:ident, $store:ident, $splat:ident,
     $add:ident, $sub:ident, $mul:ident, $horizontal:path) => {
        #[target_feature(enable = $feature)]
        fn accumulate<const CROSS: bool>(src: [&[f32]; 2], sums: [&mut [f32]; 3]) {
            let n = sums[0].len();
            let end = n / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let v = $load(src[0].as_ptr().add(x));
                    let mean = $add($load(sums[0].as_ptr().add(x)), v);
                    let square = $add($load(sums[1].as_ptr().add(x)), $mul(v, v));
                    $store(sums[0].as_mut_ptr().add(x), mean);
                    $store(sums[1].as_mut_ptr().add(x), square);
                    if CROSS {
                        let product = $mul($load(src[1].as_ptr().add(x)), v);
                        let cross = $add($load(sums[2].as_ptr().add(x)), product);
                        $store(sums[2].as_mut_ptr().add(x), cross);
                    }
                }
            }
            super::accumulate_scalar::<CROSS>(
                src.map(|s| if s.is_empty() { s } else { &s[end..] }),
                sums.map(|s| {
                    let len = s.len();
                    &mut s[end.min(len)..]
                }),
            );
        }

        #[target_feature(enable = $feature)]
        fn step<const CROSS: bool, const UPDATE: bool>(
            enter: [&[f32]; 2],
            leave: [&[f32]; 2],
            sums: [&mut [f32]; 3],
            output: [&mut [f32]; 3],
        ) {
            let n = sums[0].len();
            let end = n / $lanes * $lanes;
            // The box mean: multiply by 1/5 instead of dividing.
            let fifth = $splat(0.2);
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let mean = $load(sums[0].as_ptr().add(x));
                    let square = $load(sums[1].as_ptr().add(x));
                    $store(output[0].as_mut_ptr().add(x), $mul(mean, fifth));
                    $store(output[1].as_mut_ptr().add(x), $mul(square, fifth));
                    if CROSS {
                        $store(
                            output[2].as_mut_ptr().add(x),
                            $mul($load(sums[2].as_ptr().add(x)), fifth),
                        );
                    }
                    if UPDATE {
                        let a = $load(enter[0].as_ptr().add(x));
                        let b = $load(leave[0].as_ptr().add(x));
                        $store(sums[0].as_mut_ptr().add(x), $add(mean, $sub(a, b)));
                        $store(
                            sums[1].as_mut_ptr().add(x),
                            $add(square, $sub($mul(a, a), $mul(b, b))),
                        );
                        if CROSS {
                            let a_ref = $load(enter[1].as_ptr().add(x));
                            let b_ref = $load(leave[1].as_ptr().add(x));
                            let cross = $load(sums[2].as_ptr().add(x));
                            $store(
                                sums[2].as_mut_ptr().add(x),
                                $add(cross, $sub($mul(a_ref, a), $mul(b_ref, b))),
                            );
                        }
                    }
                }
            }
            let tail = |s: &[f32]| if s.is_empty() { 0 } else { end };
            super::step_scalar::<CROSS, UPDATE>(
                enter.map(|s| &s[tail(s)..]),
                leave.map(|s| &s[tail(s)..]),
                sums.map(|s| {
                    let len = s.len();
                    &mut s[end.min(len)..]
                }),
                output.map(|s| {
                    let len = s.len();
                    &mut s[end.min(len)..]
                }),
            );
        }

        // A target feature enabled for the whole build (wasm simd128) makes
        // these calls safe; the other backends still need the blocks.
        #[allow(unused_unsafe)]
        pub(super) fn kernels() -> super::Kernels {
            // The selector checks CPU support before publishing these safe
            // wrappers. Scratch::run validates the inputs and sizes every
            // active moment buffer before entering an unsafe SIMD kernel.
            super::Kernels {
                reference: super::MomentKernels {
                    accumulate: |src, sums| unsafe { accumulate::<false>(src, sums) },
                    advance: |a, b, sums, out| unsafe { step::<false, true>(a, b, sums, out) },
                    finish: |a, b, sums, out| unsafe { step::<false, false>(a, b, sums, out) },
                },
                comparison: super::MomentKernels {
                    accumulate: |src, sums| unsafe { accumulate::<true>(src, sums) },
                    advance: |a, b, sums, out| unsafe { step::<true, true>(a, b, sums, out) },
                    finish: |a, b, sums, out| unsafe { step::<true, false>(a, b, sums, out) },
                },
                horizontal: |src, w, out| unsafe { $horizontal(src, w, out) },
            }
        }
    };
}

// Four lanes carry four independent horizontal recurrences. All loads stay in
// the four-row working set, including reflected edges and widths below a lane.
#[allow(unused_macros)]
macro_rules! horizontal {
    ($feature:literal, $load:ident, $store:ident, $splat:ident, $add:ident, $sub:ident, $mul:ident) => {
        #[target_feature(enable = $feature)]
        pub(super) fn horizontal(src: &[f32], w: usize, dst: &mut [f32]) {
            unsafe {
                let column =
                    |x| $load([src[x], src[w + x], src[2 * w + x], src[3 * w + x]].as_ptr());
                let mut sum = $splat(-0.0);
                for x in -2..=2 {
                    sum = $add(sum, column(super::super::mirror(x, w)));
                }
                let fifth = $splat(0.2);
                for x in 0..w {
                    let mut values = [0.0; 4];
                    $store(values.as_mut_ptr(), $mul(sum, fifth));
                    for r in 0..4 {
                        dst[r * w + x] = values[r];
                    }
                    if x + 1 < w {
                        sum = $add(
                            sum,
                            $sub(
                                column(super::super::mirror(x as isize + 3, w)),
                                column(super::super::mirror(x as isize - 2, w)),
                            ),
                        );
                    }
                }
            }
        }
    };
}

#[cfg(any(
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse")
))]
mod sse {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    horizontal!(
        "sse2",
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_set1_ps,
        _mm_add_ps,
        _mm_sub_ps,
        _mm_mul_ps
    );
    #[cfg(feature = "sse")]
    moments!(
        "sse2",
        4,
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_set1_ps,
        _mm_add_ps,
        _mm_sub_ps,
        _mm_mul_ps,
        horizontal
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use std::arch::x86_64::*;
    moments!(
        "avx2",
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_add_ps,
        _mm256_sub_ps,
        _mm256_mul_ps,
        super::sse::horizontal
    );
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use std::arch::aarch64::*;
    horizontal!(
        "neon",
        vld1q_f32,
        vst1q_f32,
        vdupq_n_f32,
        vaddq_f32,
        vsubq_f32,
        vmulq_f32
    );
    moments!(
        "neon",
        4,
        vld1q_f32,
        vst1q_f32,
        vdupq_n_f32,
        vaddq_f32,
        vsubq_f32,
        vmulq_f32,
        horizontal
    );
}

#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn load(p: *const f32) -> v128 {
        unsafe { v128_load(p.cast()) }
    }
    #[target_feature(enable = "simd128")]
    unsafe fn store(p: *mut f32, v: v128) {
        unsafe { v128_store(p.cast(), v) }
    }
    horizontal!(
        "simd128",
        load,
        store,
        f32x4_splat,
        f32x4_add,
        f32x4_sub,
        f32x4_mul
    );
    moments!(
        "simd128",
        4,
        load,
        store,
        f32x4_splat,
        f32x4_add,
        f32x4_sub,
        f32x4_mul,
        horizontal
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available_kernels() -> Vec<(&'static str, Kernels)> {
        #[allow(unused_mut)]
        let mut kernels = vec![
            ("scalar", scalar_kernels()),
            ("selected", selected_kernels()),
        ];
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", avx::kernels()));
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse2") {
            kernels.push(("sse2", sse::kernels()));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        kernels.push(("neon", neon::kernels()));
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        kernels.push(("wasm", wasm::kernels()));
        kernels
    }

    // Original full-plane box filter: an independent oracle for the streaming
    // layout, the fused products and the exact order of both recurrences.
    fn original_box5(src: &[f32], w: usize, h: usize) -> Vec<f32> {
        let mut tmp = vec![0.; w * h];
        let mut dst = vec![0.; w * h];
        for x in 0..w {
            let mut sum: f32 = (-2..=2)
                .map(|y| src[super::super::mirror(y, h) * w + x])
                .sum();
            for y in 0..h {
                tmp[y * w + x] = sum * 0.2;
                sum += src[super::super::mirror(y as isize + 3, h) * w + x]
                    - src[super::super::mirror(y as isize - 2, h) * w + x];
            }
        }
        for y in 0..h {
            let row = &tmp[y * w..(y + 1) * w];
            let mut sum: f32 = (-2..=2).map(|x| row[super::super::mirror(x, w)]).sum();
            for x in 0..w {
                dst[y * w + x] = sum * 0.2;
                sum += row[super::super::mirror(x as isize + 3, w)]
                    - row[super::super::mirror(x as isize - 2, w)];
            }
        }
        dst
    }

    #[test]
    fn streamed_moments_match_original_bits_for_all_backends_and_tails() {
        for (name, kernels) in available_kernels() {
            let mut scratch = Scratch::default();
            for w in (1..=19).chain([31, 32, 33, 127, 257]) {
                for h in (1..=9).chain([17, 65]) {
                    for flat in [false, true] {
                        let src: Vec<_> = (0..w * h)
                            .map(|i| {
                                if flat {
                                    0.21
                                } else {
                                    ((i * 6719 + 941) % 65521) as f32 * 0.00001 - 0.3
                                }
                            })
                            .collect();
                        let reference: Vec<_> = (0..w * h)
                            .map(|i| ((i * 1877 + 11) % 3253) as f32 * 0.0001)
                            .collect();
                        let square: Vec<_> = src.iter().map(|v| v * v).collect();
                        let cross: Vec<_> =
                            src.iter().zip(&reference).map(|(&a, &b)| b * a).collect();
                        let expected = [
                            original_box5(&src, w, h),
                            original_box5(&square, w, h),
                            original_box5(&cross, w, h),
                        ];
                        let check = |count: usize, y: usize, rows: [&[f32]; 3]| {
                            for c in 0..count {
                                for x in 0..w {
                                    assert_eq!(
                                        rows[c][x].to_bits(),
                                        expected[c][y * w + x].to_bits(),
                                        "{name} {w}x{h}, flat={flat}, moment={c}, ({x},{y})"
                                    );
                                }
                            }
                        };
                        scratch.run::<false>(kernels, &src, &[], w, h, |y, rows| check(2, y, rows));
                        scratch.run::<true>(kernels, &src, &reference, w, h, |y, rows| {
                            check(3, y, rows)
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn retained_box_storage_is_independent_of_image_height() {
        let w = 33;
        let mut scratch = Scratch::default();
        let kernels = selected_kernels();
        let mut capacities = Vec::new();
        for h in [4, 1024, 1, 129] {
            let src = vec![0.21; w * h];
            scratch.run::<true>(kernels, &src, &src, w, h, |_, _| {});
            let current: Vec<_> = scratch
                .sums
                .iter()
                .chain(&scratch.vertical)
                .chain(&scratch.output)
                .map(|v| (v.len(), v.capacity()))
                .collect();
            if capacities.is_empty() {
                capacities = current;
            } else {
                assert_eq!(current, capacities);
            }
        }
    }
}
