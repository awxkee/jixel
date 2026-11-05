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
    ($feature:literal, $lanes:literal) => {
        #[target_feature(enable=$feature)]
        unsafe fn difference(a: [&[f32]; 3], b: [&[f32]; 3], scale: [f32; 3], out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                let scales = scale.map(|s| v::splat(s));
                for x in (0..end).step_by($lanes) {
                    let mut v = v::splat(0.0);
                    for c in 0..3 {
                        v = v::add(
                            v,
                            v::mul(
                                v::abs(v::sub(
                                    v::load(a[c].as_ptr().add(x)),
                                    v::load(b[c].as_ptr().add(x)),
                                )),
                                scales[c],
                            ),
                        );
                    }
                    v::store(out.as_mut_ptr().add(x), v);
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
                    let [m, l, r, u, d] = src.map(|s| v::load(s.as_ptr().add(x)));
                    v::store(
                        out.as_mut_ptr().add(x),
                        v::add(v::add(m, v::add(l, r)), v::add(u, d)),
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
                let one = v::splat(1.0);
                let zero = v::splat(0.0);
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
                        v::mul(v::splat(s), v::load(mul.as_ptr().add(x % 8)))
                    } else {
                        zero
                    };
                    let keep = if active { one } else { zero };
                    let mut acc = row.center.map(|p| v::load(p.as_ptr().add(x)));
                    let mut sum = one;
                    for n in row.neighbors {
                        let weight = v::mul(
                            v::max(
                                v::add(v::mul(v::load(n.sad.as_ptr().add(x)), inv), one),
                                zero,
                            ),
                            keep,
                        );
                        sum = v::add(sum, weight);
                        for c in 0..3 {
                            acc[c] = v::add(
                                acc[c],
                                v::mul(weight, v::load(n.pixels[c].as_ptr().add(x))),
                            );
                        }
                    }
                    // One reciprocal shared by the three channels.
                    let inv_sum = v::div(one, sum);
                    for c in 0..3 {
                        v::store(out[c].as_mut_ptr().add(x), v::mul(acc[c], inv_sum));
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
                let [w0, w1, w2] = weights.map(|w| v::splat(w));
                while x + $lanes < out.len() {
                    let [t, m, b] = src;
                    let side = v::add(
                        v::add(
                            v::load(m.as_ptr().add(x - 1)),
                            v::load(m.as_ptr().add(x + 1)),
                        ),
                        v::add(v::load(t.as_ptr().add(x)), v::load(b.as_ptr().add(x))),
                    );
                    let corner = v::add(
                        v::add(
                            v::load(t.as_ptr().add(x - 1)),
                            v::load(t.as_ptr().add(x + 1)),
                        ),
                        v::add(
                            v::load(b.as_ptr().add(x - 1)),
                            v::load(b.as_ptr().add(x + 1)),
                        ),
                    );
                    let value = v::add(
                        v::mul(corner, w2),
                        v::add(v::mul(side, w1), v::mul(v::load(m.as_ptr().add(x)), w0)),
                    );
                    v::store(out.as_mut_ptr().add(x), value);
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
                    let s = v::load(src.as_ptr().add(x));
                    v::store(
                        out.as_mut_ptr().add(x),
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
            super::sigma(&src[end..], lut, quant_mul, &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        unsafe fn add(out: &mut [f32], src: &[f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    v::store(
                        out.as_mut_ptr().add(x),
                        v::add(v::load(out.as_ptr().add(x)), v::load(src.as_ptr().add(x))),
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
                    v::store(
                        out.as_mut_ptr().add(x),
                        v::add(v::load(out.as_ptr().add(x)), v::splat(value)),
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
                    let mut sum = v::splat(-0.0);
                    for k in 0..n {
                        let column: [f32; $lanes] =
                            std::array::from_fn(|lane| matrix[(j + lane) * n + k]);
                        sum = v::add(sum, v::mul(v::load(column.as_ptr()), v::splat(delta[k])));
                    }
                    let mut values = [0.0; $lanes];
                    v::store(values.as_mut_ptr(), sum);
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
    use crate::simd::sse::f32 as v;
    backend!("sse2", 4);
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use crate::simd::avx::f32 as v;
    backend!("avx2", 8);
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use crate::simd::neon::f32 as v;
    backend!("neon", 4);
}
#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm {
    use crate::simd::wasm::f32 as v;
    backend!("simd128", 4);
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
