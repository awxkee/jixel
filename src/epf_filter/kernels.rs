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
//! Separate multiplies and adds retain the decoder model's f32 operation order.
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
    scalar()
}

fn difference(a: [&[f32]; 3], b: [&[f32]; 3], scale: [f32; 3], out: &mut [f32]) {
    for x in 0..out.len() {
        out[x] = 0.0
            + (a[0][x] - b[0][x]).abs() * scale[0]
            + (a[1][x] - b[1][x]).abs() * scale[1]
            + (a[2][x] - b[2][x]).abs() * scale[2];
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
    let keep = if s >= MIN_SIGMA { 1.0 } else { 0.0 };
    let inv = if s >= MIN_SIGMA {
        s * if row.border || x.is_multiple_of(8) || x % 8 == 7 {
            row.bsm
        } else {
            row.sm
        }
    } else {
        0.0
    };
    let mut acc = row.center.map(|p| p[x]);
    let mut sum = 1.0;
    for n in row.neighbors {
        let weight = (n.sad[x] * inv + 1.0).max(0.0) * keep;
        sum += weight;
        for c in 0..3 {
            acc[c] += weight * n.pixels[c][x];
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
    corner * weights[2] + (side * weights[1] + m[x] * weights[0])
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
fn project(matrix: &[f32], delta: &[f32], cx: usize, coeffs: &mut [f32]) {
    let n = delta.len();
    for j in 0..n {
        coeffs[j / cx * cx * 8 + j % cx] = matrix[j * n..(j + 1) * n]
            .iter()
            .zip(delta)
            .map(|(a, b)| a * b)
            .sum();
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

#[allow(unused_macros)]
macro_rules! backend {
    ($feature:literal, $lanes:literal, $load:ident, $store:ident, $splat:ident,
     $add:ident, $sub:ident, $mul:ident, $div:ident, $abs:ident, $min:ident, $max:ident) => {
        #[target_feature(enable=$feature)]
        unsafe fn difference(a: [&[f32]; 3], b: [&[f32]; 3], scale: [f32; 3], out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                let scales = scale.map(|s| $splat(s));
                for x in (0..end).step_by($lanes) {
                    let mut v = $splat(0.0);
                    for c in 0..3 {
                        v = $add(
                            v,
                            $mul(
                                $abs($sub(
                                    $load(a[c].as_ptr().add(x)),
                                    $load(b[c].as_ptr().add(x)),
                                )),
                                scales[c],
                            ),
                        );
                    }
                    $store(out.as_mut_ptr().add(x), v);
                }
            }
            super::difference(
                a.map(|s| &s[end..]),
                b.map(|s| &s[end..]),
                scale,
                &mut out[end..],
            );
        }
        #[target_feature(enable=$feature)]
        unsafe fn patch(src: [&[f32]; 5], out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let [m, l, r, u, d] = src.map(|s| $load(s.as_ptr().add(x)));
                    $store(
                        out.as_mut_ptr().add(x),
                        $add($add(m, $add(l, r)), $add(u, d)),
                    );
                }
            }
            super::patch(src.map(|s| &s[end..]), &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        unsafe fn filter(row: super::FilterRow<'_>, out: [&mut [f32]; 3]) {
            let width = out[0].len();
            let end = width / $lanes * $lanes;
            unsafe {
                let one = $splat(1.0);
                let zero = $splat(0.0);
                // Eight pixels share sigma. Only block-edge multipliers vary by lane.
                let mul: [f32; 8] = std::array::from_fn(|i| {
                    if row.border || i == 0 || i == 7 {
                        row.bsm
                    } else {
                        row.sm
                    }
                });
                for x in (0..end).step_by($lanes) {
                    let s = row.sigma[x / 8];
                    let active = s >= super::MIN_SIGMA;
                    let inv = if active {
                        $mul($splat(s), $load(mul.as_ptr().add(x % 8)))
                    } else {
                        zero
                    };
                    let keep = if active { one } else { zero };
                    let mut acc = row.center.map(|p| $load(p.as_ptr().add(x)));
                    let mut sum = one;
                    for n in row.neighbors {
                        let weight = $mul(
                            $max($add($mul($load(n.sad.as_ptr().add(x)), inv), one), zero),
                            keep,
                        );
                        sum = $add(sum, weight);
                        for c in 0..3 {
                            acc[c] = $add(acc[c], $mul(weight, $load(n.pixels[c].as_ptr().add(x))));
                        }
                    }
                    // One reciprocal shared by the three channels.
                    let inv_sum = $div(one, sum);
                    for c in 0..3 {
                        $store(out[c].as_mut_ptr().add(x), $mul(acc[c], inv_sum));
                    }
                }
            }
            for x in end..width {
                let p = super::pixel(&row, x);
                for c in 0..3 {
                    out[c][x] = p[c];
                }
            }
        }
        #[target_feature(enable=$feature)]
        unsafe fn gaborish(src: [&[f32]; 3], weights: [f32; 3], out: &mut [f32]) {
            if out.is_empty() {
                return;
            }
            out[0] = super::gaborish_pixel(src, weights, 0);
            let mut x = 1;
            unsafe {
                let [w0, w1, w2] = weights.map(|w| $splat(w));
                while x + $lanes < out.len() {
                    let [t, m, b] = src;
                    let side = $add(
                        $add($load(m.as_ptr().add(x - 1)), $load(m.as_ptr().add(x + 1))),
                        $add($load(t.as_ptr().add(x)), $load(b.as_ptr().add(x))),
                    );
                    let corner = $add(
                        $add($load(t.as_ptr().add(x - 1)), $load(t.as_ptr().add(x + 1))),
                        $add($load(b.as_ptr().add(x - 1)), $load(b.as_ptr().add(x + 1))),
                    );
                    let value = $add(
                        $mul(corner, w2),
                        $add($mul(side, w1), $mul($load(m.as_ptr().add(x)), w0)),
                    );
                    $store(out.as_mut_ptr().add(x), value);
                    x += $lanes;
                }
            }
            for x in x..out.len() {
                out[x] = super::gaborish_pixel(src, weights, x);
            }
        }
        #[target_feature(enable=$feature)]
        unsafe fn sigma(src: &[f32], lut: f32, quant_mul: f32, out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let s = $load(src.as_ptr().add(x));
                    $store(
                        out.as_mut_ptr().add(x),
                        $div(
                            $splat(1.0),
                            $min($mul($div($splat(quant_mul), s), $splat(lut)), $splat(-1e-4)),
                        ),
                    );
                }
            }
            super::sigma(&src[end..], lut, quant_mul, &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        unsafe fn add(out: &mut [f32], src: &[f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    $store(
                        out.as_mut_ptr().add(x),
                        $add($load(out.as_ptr().add(x)), $load(src.as_ptr().add(x))),
                    );
                }
            }
            super::add(&mut out[end..], &src[end..]);
        }
        #[target_feature(enable=$feature)]
        unsafe fn add_constant(out: &mut [f32], value: f32) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    $store(
                        out.as_mut_ptr().add(x),
                        $add($load(out.as_ptr().add(x)), $splat(value)),
                    );
                }
            }
            super::add_constant(&mut out[end..], value);
        }
        #[target_feature(enable=$feature)]
        unsafe fn project(matrix: &[f32], delta: &[f32], cx: usize, coeffs: &mut [f32]) {
            let n = delta.len();
            let end = n / $lanes * $lanes;
            unsafe {
                for j in (0..end).step_by($lanes) {
                    let mut sum = $splat(-0.0);
                    for k in 0..n {
                        let column: [f32; $lanes] =
                            std::array::from_fn(|lane| matrix[(j + lane) * n + k]);
                        sum = $add(sum, $mul($load(column.as_ptr()), $splat(delta[k])));
                    }
                    let mut values = [0.0; $lanes];
                    $store(values.as_mut_ptr(), sum);
                    for lane in 0..$lanes {
                        let i = j + lane;
                        coeffs[i / cx * cx * 8 + i % cx] = values[lane];
                    }
                }
            }
            for j in end..n {
                coeffs[j / cx * cx * 8 + j % cx] = matrix[j * n..(j + 1) * n]
                    .iter()
                    .zip(delta)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
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
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "sse2")]
    fn abs(v: __m128) -> __m128 {
        _mm_andnot_ps(_mm_set1_ps(-0.0), v)
    }
    backend!(
        "sse2",
        4,
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_set1_ps,
        _mm_add_ps,
        _mm_sub_ps,
        _mm_mul_ps,
        _mm_div_ps,
        abs,
        _mm_min_ps,
        _mm_max_ps
    );
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx2")]
    fn abs(v: __m256) -> __m256 {
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), v)
    }
    backend!(
        "avx2",
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_add_ps,
        _mm256_sub_ps,
        _mm256_mul_ps,
        _mm256_div_ps,
        abs,
        _mm256_min_ps,
        _mm256_max_ps
    );
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use std::arch::aarch64::*;
    backend!(
        "neon",
        4,
        vld1q_f32,
        vst1q_f32,
        vdupq_n_f32,
        vaddq_f32,
        vsubq_f32,
        vmulq_f32,
        vdivq_f32,
        vabsq_f32,
        vminnmq_f32,
        vmaxnmq_f32
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
    // Wasm's ordinary min/max propagate NaNs; Rust's min/max select the other operand.
    #[target_feature(enable = "simd128")]
    fn min(a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, f32x4_lt(a, b))
    }
    #[target_feature(enable = "simd128")]
    fn max(a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, f32x4_gt(a, b))
    }
    backend!(
        "simd128",
        4,
        load,
        store,
        f32x4_splat,
        f32x4_add,
        f32x4_sub,
        f32x4_mul,
        f32x4_div,
        f32x4_abs,
        min,
        max
    );
}

#[cfg(test)]
pub(crate) fn available() -> Vec<(&'static str, Kernels)> {
    #[allow(unused_mut)]
    let mut k = vec![("scalar", scalar()), ("selected", selected())];
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        k.push(("avx2", avx::kernels()));
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
