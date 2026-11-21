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

//! EPF row operations. Selection happens once per renderer, never per pixel.
//! Products feed their sums through `mul_add`: one fused rounding on AVX2+FMA
//! and NEON, a plain multiply then add on scalar, SSE2 and wasm. Backends agree
//! to a few ULPs, not bit for bit.
use super::MIN_SIGMA;

pub(crate) type Diff = fn([&[f32]; 3], [&[f32]; 3], [f32; 3], &mut [f32]);
pub(crate) type Patch = fn([&[f32]; 5], &mut [f32]);
#[derive(Clone, Copy)]
pub(crate) struct Neighbor<'a> {
    pub(crate) pixels: [&'a [f32]; 3],
    pub(crate) sad: &'a [f32],
}
pub(crate) struct FilterRow<'a> {
    pub(crate) center: [&'a [f32]; 3],
    pub(crate) neighbors: &'a [Neighbor<'a>],
    pub(crate) sigma: &'a [f32],
    pub(crate) border: bool,
    pub(crate) sm: f32,
    pub(crate) bsm: f32,
}
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) difference: Diff,
    pub(crate) patch: Patch,
    pub(crate) filter: fn(FilterRow<'_>, [&mut [f32]; 3]),
    pub(crate) gaborish: fn([&[f32]; 3], [f32; 3], &mut [f32]),
    pub(crate) sigma: fn(&[f32], f32, f32, &mut [f32]),
    pub(crate) add: fn(&mut [f32], &[f32]),
    pub(crate) add_constant: fn(&mut [f32], f32),
    pub(crate) project: fn(&[f32], &[f32], usize, &mut [f32]),
}

pub(crate) fn selected() -> Kernels {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
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
    scalar()
}

#[inline(always)]
fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

fn difference(a: [&[f32]; 3], b: [&[f32]; 3], scale: [f32; 3], out: &mut [f32]) {
    for x in 0..out.len() {
        let mut v = (a[0][x] - b[0][x]).abs() * scale[0];
        v = mul_add((a[1][x] - b[1][x]).abs(), scale[1], v);
        out[x] = mul_add((a[2][x] - b[2][x]).abs(), scale[2], v);
    }
}
fn patch(src: [&[f32]; 5], out: &mut [f32]) {
    for x in 0..out.len() {
        out[x] = src[0][x] + (src[1][x] + src[2][x]) + (src[3][x] + src[4][x]);
    }
}
#[inline]
fn pixel(row: &FilterRow<'_>, x: usize) -> [f32; 3] {
    let s = row.sigma[x / 8];
    let mut acc = row.center.map(|p| p[x]);
    // Unfiltered blocks give every neighbor zero weight: the center, exactly.
    if s < MIN_SIGMA {
        return acc;
    }
    let inv = s * if row.border || x.is_multiple_of(8) || x % 8 == 7 {
        row.bsm
    } else {
        row.sm
    };
    let mut sum = 1.0;
    for n in row.neighbors {
        let weight = mul_add(n.sad[x], inv, 1.0).max(0.0);
        sum += weight;
        for c in 0..3 {
            acc[c] = mul_add(weight, n.pixels[c][x], acc[c]);
        }
    }
    // One reciprocal shared by the three channels.
    let inv_sum = 1.0 / sum;
    acc.map(|v| v * inv_sum)
}
fn filter(row: FilterRow<'_>, out: [&mut [f32]; 3]) {
    for x in 0..out[0].len() {
        let p = pixel(&row, x);
        for c in 0..3 {
            out[c][x] = p[c];
        }
    }
}
#[inline]
fn gaborish_pixel(src: [&[f32]; 3], weights: [f32; 3], x: usize) -> f32 {
    let [t, m, b] = src;
    let left = x.saturating_sub(1);
    let right = (x + 1).min(m.len() - 1);
    let side = (m[left] + m[right]) + (t[x] + b[x]);
    let corner = (t[left] + t[right]) + (b[left] + b[right]);
    mul_add(
        corner,
        weights[2],
        mul_add(side, weights[1], m[x] * weights[0]),
    )
}
fn gaborish(src: [&[f32]; 3], weights: [f32; 3], out: &mut [f32]) {
    for (x, v) in out.iter_mut().enumerate() {
        *v = gaborish_pixel(src, weights, x);
    }
}
fn sigma(src: &[f32], lut: f32, quant_mul: f32, out: &mut [f32]) {
    for (out, &s) in out.iter_mut().zip(src) {
        *out = 1.0 / (quant_mul / s * lut).min(-1e-4);
    }
}
fn add(out: &mut [f32], src: &[f32]) {
    for (v, &s) in out.iter_mut().zip(src) {
        *v += s;
    }
}
fn add_constant(out: &mut [f32], value: f32) {
    for v in out {
        *v += value;
    }
}
/// `matrix` is column-major (`k * n + j`), so lanes over `j` load contiguously.
fn project(matrix: &[f32], delta: &[f32], cx: usize, coeffs: &mut [f32]) {
    project_rows(matrix, delta, cx, 0, coeffs);
}
/// Rows `from..n` of the projection, summing `k` in ascending order.
#[inline]
fn project_rows(matrix: &[f32], delta: &[f32], cx: usize, from: usize, coeffs: &mut [f32]) {
    let n = delta.len();
    for j in from..n {
        let mut sum = -0.0;
        for (k, &d) in delta.iter().enumerate() {
            sum = mul_add(matrix[k * n + j], d, sum);
        }
        coeffs[j / cx * cx * 8 + j % cx] = sum;
    }
}
#[allow(dead_code)]
pub(crate) fn scalar() -> Kernels {
    Kernels {
        difference,
        patch,
        filter,
        gaborish,
        sigma,
        add,
        add_constant,
        project,
    }
}

/// Calls `$f::<$unroll>(args.., x)` while `$unroll` vectors fit before `$to`,
/// then `$f::<1>`, and yields where the scalar tail starts.
#[allow(unused_macros)]
macro_rules! unrolled {
    ($lanes:literal, $unroll:literal, $from:expr, $to:expr, $f:ident($($arg:expr),*)) => {{
        let to = $to;
        let mut x = $from;
        while x + $lanes * $unroll <= to {
            $f::<$unroll>($($arg,)* x);
            x += $lanes * $unroll;
        }
        while x + $lanes <= to {
            $f::<1>($($arg,)* x);
            x += $lanes;
        }
        x
    }};
}

/// Every row kernel runs `$unroll` independent vectors per step: the bodies
/// are short dependency chains (divisions, multiply-add sums), and a single
/// vector per iteration leaves the FP pipes idle on their latency.
#[allow(unused_macros)]
macro_rules! backend {
    ($feature:literal, $lanes:literal, $unroll:literal) => {
        #[target_feature(enable=$feature)]
        #[inline]
        fn difference_at<const N: usize>(
            a: [&[f32]; 3],
            b: [&[f32]; 3],
            scale: [f32; 3],
            out: &mut [f32],
            x: usize,
        ) {
            debug_assert!(x + N * $lanes <= out.len());
            let scales = scale.map(|s| v::splat(s));
            for i in 0..N {
                let at = x + i * $lanes;
                // SAFETY: the driver keeps `at + lanes` within every span.
                unsafe {
                    let d = std::array::from_fn::<_, 3, _>(|c| {
                        v::abs_diff(
                            v::load(a[c].as_ptr().add(at)),
                            v::load(b[c].as_ptr().add(at)),
                        )
                    });
                    let sum = v::mul_add(
                        d[2],
                        scales[2],
                        v::mul_add(d[1], scales[1], v::mul(d[0], scales[0])),
                    );
                    v::store(out.as_mut_ptr().add(at), sum);
                }
            }
        }
        #[target_feature(enable=$feature)]
        fn difference(a: [&[f32]; 3], b: [&[f32]; 3], scale: [f32; 3], out: &mut [f32]) {
            let n = out.len();
            let a = a.map(|s| &s[..n]);
            let b = b.map(|s| &s[..n]);
            let end = unrolled!($lanes, $unroll, 0, n, difference_at(a, b, scale, out));
            super::difference(
                a.map(|s| &s[end..]),
                b.map(|s| &s[end..]),
                scale,
                &mut out[end..],
            );
        }
        #[target_feature(enable=$feature)]
        #[inline]
        fn patch_at<const N: usize>(src: [&[f32]; 5], out: &mut [f32], x: usize) {
            debug_assert!(x + N * $lanes <= out.len());
            for i in 0..N {
                let at = x + i * $lanes;
                // SAFETY: the driver keeps `at + lanes` within every span.
                unsafe {
                    let [m, l, r, u, d] = src.map(|s| v::load(s.as_ptr().add(at)));
                    v::store(
                        out.as_mut_ptr().add(at),
                        v::add(v::add(m, v::add(l, r)), v::add(u, d)),
                    );
                }
            }
        }
        #[target_feature(enable=$feature)]
        fn patch(src: [&[f32]; 5], out: &mut [f32]) {
            let n = out.len();
            let src = src.map(|s| &s[..n]);
            let end = unrolled!($lanes, $unroll, 0, n, patch_at(src, out));
            super::patch(src.map(|s| &s[end..]), &mut out[end..]);
        }
        /// `N` adjacent vectors at once: each output's accumulators form a
        /// dependent chain through every neighbor, so interleaving independent
        /// vectors hides the multiply-add latency.
        #[target_feature(enable=$feature)]
        #[inline]
        fn vectors<const N: usize>(
            row: &super::FilterRow<'_>,
            mul: &[f32; 8],
            out: [*mut f32; 3],
            x: usize,
        ) {
            unsafe {
                let zero = v::splat(0.0);
                let one = v::splat(1.0);
                // An unfiltered block gets inverse sigma and bias zero: every
                // neighbor weighs exactly zero and the center passes through.
                let mut inv = [zero; N];
                let mut bias = [zero; N];
                let mut active = false;
                for i in 0..N {
                    let at = x + i * $lanes;
                    let s = row.sigma[at / 8];
                    if s >= super::MIN_SIGMA {
                        // Lanes never straddle a block; only edge multipliers vary.
                        inv[i] = v::mul(v::splat(s), v::load(mul.as_ptr().add(at % 8)));
                        bias[i] = one;
                        active = true;
                    }
                }
                let mut acc: [[_; 3]; N] = std::array::from_fn(|i| {
                    row.center.map(|p| v::load(p.as_ptr().add(x + i * $lanes)))
                });
                if !active {
                    for i in 0..N {
                        for c in 0..3 {
                            v::store(out[c].add(x + i * $lanes), acc[i][c]);
                        }
                    }
                    return;
                }
                let mut sum = [one; N];
                for n in row.neighbors {
                    for i in 0..N {
                        let at = x + i * $lanes;
                        let weight = v::max(
                            v::mul_add(v::load(n.sad.as_ptr().add(at)), inv[i], bias[i]),
                            zero,
                        );
                        sum[i] = v::add(sum[i], weight);
                        for c in 0..3 {
                            acc[i][c] = v::mul_add(
                                weight,
                                v::load(n.pixels[c].as_ptr().add(at)),
                                acc[i][c],
                            );
                        }
                    }
                }
                for i in 0..N {
                    // One reciprocal shared by the three channels.
                    let inv_sum = v::div(one, sum[i]);
                    for c in 0..3 {
                        v::store(out[c].add(x + i * $lanes), v::mul(acc[i][c], inv_sum));
                    }
                }
            }
        }
        #[target_feature(enable=$feature)]
        fn filter(row: super::FilterRow<'_>, out: [&mut [f32]; 3]) {
            let width = out[0].len();
            // Eight pixels share sigma. Only block-edge multipliers vary by lane.
            let mul: [f32; 8] = std::array::from_fn(|i| {
                if row.border || i == 0 || i == 7 {
                    row.bsm
                } else {
                    row.sm
                }
            });
            let [r, g, b] = out;
            let dst = [r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr()];
            let end = unrolled!($lanes, $unroll, 0, width, vectors(&row, &mul, dst));
            for x in end..width {
                let p = super::pixel(&row, x);
                for c in 0..3 {
                    unsafe { *dst[c].add(x) = p[c] };
                }
            }
        }
        #[target_feature(enable=$feature)]
        #[inline]
        fn gaborish_at<const N: usize>(
            src: [&[f32]; 3],
            weights: [f32; 3],
            out: &mut [f32],
            x: usize,
        ) {
            debug_assert!(x >= 1 && x + N * $lanes < out.len());
            let [w0, w1, w2] = weights.map(|w| v::splat(w));
            let [t, m, b] = src;
            for i in 0..N {
                let at = x + i * $lanes;
                // SAFETY: the driver keeps `at - 1` and `at + lanes` in bounds.
                unsafe {
                    let side = v::add(
                        v::add(
                            v::load(m.as_ptr().add(at - 1)),
                            v::load(m.as_ptr().add(at + 1)),
                        ),
                        v::add(v::load(t.as_ptr().add(at)), v::load(b.as_ptr().add(at))),
                    );
                    let corner = v::add(
                        v::add(
                            v::load(t.as_ptr().add(at - 1)),
                            v::load(t.as_ptr().add(at + 1)),
                        ),
                        v::add(
                            v::load(b.as_ptr().add(at - 1)),
                            v::load(b.as_ptr().add(at + 1)),
                        ),
                    );
                    let value = v::mul_add(
                        corner,
                        w2,
                        v::mul_add(side, w1, v::mul(v::load(m.as_ptr().add(at)), w0)),
                    );
                    v::store(out.as_mut_ptr().add(at), value);
                }
            }
        }
        #[target_feature(enable=$feature)]
        fn gaborish(src: [&[f32]; 3], weights: [f32; 3], out: &mut [f32]) {
            let n = out.len();
            if n == 0 {
                return;
            }
            let src = src.map(|s| &s[..n]);
            out[0] = super::gaborish_pixel(src, weights, 0);
            // Vectors need one pixel on either side: interior lanes only.
            let end = unrolled!($lanes, $unroll, 1, n - 1, gaborish_at(src, weights, out));
            for x in end..n {
                out[x] = super::gaborish_pixel(src, weights, x);
            }
        }
        #[target_feature(enable=$feature)]
        #[inline]
        fn sigma_at<const N: usize>(
            src: &[f32],
            lut: f32,
            quant_mul: f32,
            out: &mut [f32],
            x: usize,
        ) {
            debug_assert!(x + N * $lanes <= out.len());
            for i in 0..N {
                let at = x + i * $lanes;
                // SAFETY: the driver keeps `at + lanes` within both spans.
                unsafe {
                    let s = v::load(src.as_ptr().add(at));
                    v::store(
                        out.as_mut_ptr().add(at),
                        v::div(
                            v::splat(1.0),
                            v::min(
                                v::mul(v::div(v::splat(quant_mul), s), v::splat(lut)),
                                v::splat(-1e-4),
                            ),
                        ),
                    );
                }
            }
        }
        #[target_feature(enable=$feature)]
        fn sigma(src: &[f32], lut: f32, quant_mul: f32, out: &mut [f32]) {
            let n = out.len();
            let src = &src[..n];
            let end = unrolled!($lanes, $unroll, 0, n, sigma_at(src, lut, quant_mul, out));
            super::sigma(&src[end..], lut, quant_mul, &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        #[inline]
        fn add_at<const N: usize>(out: &mut [f32], src: &[f32], x: usize) {
            debug_assert!(x + N * $lanes <= out.len());
            for i in 0..N {
                let at = x + i * $lanes;
                // SAFETY: the driver keeps `at + lanes` within both spans.
                unsafe {
                    v::store(
                        out.as_mut_ptr().add(at),
                        v::add(v::load(out.as_ptr().add(at)), v::load(src.as_ptr().add(at))),
                    );
                }
            }
        }
        #[target_feature(enable=$feature)]
        fn add(out: &mut [f32], src: &[f32]) {
            let n = out.len();
            let src = &src[..n];
            let end = unrolled!($lanes, $unroll, 0, n, add_at(out, src));
            super::add(&mut out[end..], &src[end..]);
        }
        #[target_feature(enable=$feature)]
        #[inline]
        fn add_constant_at<const N: usize>(out: &mut [f32], value: f32, x: usize) {
            debug_assert!(x + N * $lanes <= out.len());
            let value = v::splat(value);
            for i in 0..N {
                let at = x + i * $lanes;
                // SAFETY: the driver keeps `at + lanes` within the span.
                unsafe {
                    v::store(
                        out.as_mut_ptr().add(at),
                        v::add(v::load(out.as_ptr().add(at)), value),
                    );
                }
            }
        }
        #[target_feature(enable=$feature)]
        fn add_constant(out: &mut [f32], value: f32) {
            let n = out.len();
            let end = unrolled!($lanes, $unroll, 0, n, add_constant_at(out, value));
            super::add_constant(&mut out[end..], value);
        }
        /// Rows `j..j + N * lanes`: independent sums over `k`, contiguous
        /// loads from the column-major matrix.
        #[target_feature(enable=$feature)]
        #[inline]
        fn project_at<const N: usize>(
            matrix: &[f32],
            delta: &[f32],
            cx: usize,
            coeffs: &mut [f32],
            j: usize,
        ) {
            let n = delta.len();
            debug_assert!(j + N * $lanes <= n && matrix.len() >= n * n);
            let mut sum = [v::splat(-0.0); N];
            for (k, &d) in delta.iter().enumerate() {
                let d = v::splat(d);
                for i in 0..N {
                    // SAFETY: k * n + j + (i + 1) * lanes <= n * n.
                    unsafe {
                        let column = v::load(matrix.as_ptr().add(k * n + j + i * $lanes));
                        sum[i] = v::mul_add(column, d, sum[i]);
                    }
                }
            }
            let mut values = [0.0f32; 64];
            for i in 0..N {
                // SAFETY: N * lanes <= 64.
                unsafe { v::store(values.as_mut_ptr().add(i * $lanes), sum[i]) };
            }
            for (lane, &value) in values[..N * $lanes].iter().enumerate() {
                let r = j + lane;
                coeffs[r / cx * cx * 8 + r % cx] = value;
            }
        }
        #[target_feature(enable=$feature)]
        fn project(matrix: &[f32], delta: &[f32], cx: usize, coeffs: &mut [f32]) {
            let n = delta.len();
            let matrix = &matrix[..n * n];
            let end = unrolled!($lanes, $unroll, 0, n, project_at(matrix, delta, cx, coeffs));
            super::project_rows(matrix, delta, cx, end, coeffs);
        }
        // Targets built with the feature enabled call these safely.
        #[allow(unused_unsafe)]
        pub(crate) fn kernels() -> super::Kernels {
            // Only CPU-checked selectors publish these wrappers. The row driver
            // supplies equal-sized spans and complete eight-pixel sigma blocks.
            super::Kernels {
                difference: |a, b, s, o| unsafe { difference(a, b, s, o) },
                patch: |s, o| unsafe { patch(s, o) },
                filter: |r, o| unsafe { filter(r, o) },
                gaborish: |s, w, o| unsafe { gaborish(s, w, o) },
                sigma: |s, l, q, o| unsafe { sigma(s, l, q, o) },
                add: |o, s| unsafe { add(o, s) },
                add_constant: |o, v| unsafe { add_constant(o, v) },
                project: |m, d, c, o| unsafe { project(m, d, c, o) },
            }
        }
    };
}
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse {
    use crate::simd::sse::f32 as v;
    backend!("sse2", 4, 2);
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use crate::simd::avx::f32 as v;
    backend!("avx2,fma", 8, 2);
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use crate::simd::neon::f32 as v;
    backend!("neon", 4, 4);
}
#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm {
    use crate::simd::wasm::f32 as v;
    backend!("simd128", 4, 2);
}

#[cfg(test)]
pub(crate) fn available() -> Vec<(&'static str, Kernels)> {
    #[allow(unused_mut)]
    let mut k = vec![("scalar", scalar()), ("selected", selected())];
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        k.push(("avx2+fma", avx::kernels()));
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        k.push(("sse2", sse::kernels()));
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    k.push(("neon", neon::kernels()));
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    k.push(("wasm", wasm::kernels()));
    k
}
