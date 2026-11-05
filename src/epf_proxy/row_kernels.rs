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

//! Pyramid reduction and scoring without full-plane intermediate terms.
//! Pool vector results in their original pixel order to keep scores bit-exact.
#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub half: fn([&[f32]; 2], &mut [f32]),
    pub half_y: fn([&[f32]; 2], &mut [f32]),
    pub half_in_place: fn(&mut [f32], usize, usize),
    pub variance: fn(&[f32], &[f32], &mut [f32]),
    pub pool: fn([&[f32]; 5], [f32; 2]) -> [f32; 2],
    pub gradients: fn([&[f32]; 4], [f32; 3]) -> [f32; 3],
}
pub(super) fn selected() -> Kernels {
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
fn half<const BIAS: bool>(src: [&[f32]; 2], out: &mut [f32]) {
    for (x, v) in out.iter_mut().enumerate() {
        let tap = |row: usize, i| {
            if BIAS {
                src[row][i] + 0.01
            } else {
                src[row][i]
            }
        };
        *v = 0.25 * (tap(0, 2 * x) + tap(1, 2 * x) + tap(0, 2 * x + 1) + tap(1, 2 * x + 1));
    }
}
fn half_in_place(values: &mut [f32], w: usize, h: usize) {
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            let i = 2 * y * w + 2 * x;
            values[y * (w / 2) + x] =
                0.25 * (values[i] + values[i + w] + values[i + 1] + values[i + w + 1]);
        }
    }
}
fn variance(mean: &[f32], square: &[f32], out: &mut [f32]) {
    for ((out, &m), &s) in out.iter_mut().zip(mean).zip(square) {
        *out = (s - m * m).max(0.0);
    }
}
fn pool(src: [&[f32]; 5], mut sum: [f32; 2]) -> [f32; 2] {
    let [mean, square, cross, reference, variance] = src;
    for x in 0..mean.len() {
        let (a, b) = (reference[x], mean[x]);
        let var = (square[x] - b * b).max(0.0);
        let cov = if variance[x] == 0.0 || var == 0.0 {
            0.0
        } else {
            cross[x] - a * b
        };
        let s = (1.0
            - (1.0 - (a - b) * (a - b)) * (2.0 * cov + 0.0009) / (variance[x] + var + 0.0009))
            .max(0.0);
        sum[0] += s;
        sum[1] += s * s * s * s;
    }
    sum
}
fn gradients(src: [&[f32]; 4], mut sum: [f32; 3]) -> [f32; 3] {
    for x in 0..src[0].len() {
        let a = src[1][x] - src[0][x];
        let b = src[3][x] - src[2][x];
        sum[0] += (b - a).abs();
        sum[1] += (b.abs() - a.abs()).max(0.0);
        let loss = (a.abs() - b.abs()).max(0.0);
        sum[2] += loss * loss * loss * loss;
    }
    sum
}
#[allow(dead_code)]
pub(super) fn scalar() -> Kernels {
    Kernels {
        half: half::<false>,
        half_y: half::<true>,
        half_in_place,
        variance,
        pool,
        gradients,
    }
}
#[allow(unused_macros)]
macro_rules! backend {
    ($feature:literal, $lanes:literal) => {
        #[target_feature(enable=$feature)]
        fn half<const BIAS: bool>(src: [&[f32]; 2], out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let mut taps = [v::splat(0.0); 4];
                    for t in 0..4 {
                        let v: [f32; $lanes] =
                            std::array::from_fn(|i| src[t % 2][2 * (x + i) + t / 2]);
                        taps[t] = v::load(v.as_ptr());
                        if BIAS {
                            taps[t] = v::add(taps[t], v::splat(0.01));
                        }
                    }
                    v::store(
                        out.as_mut_ptr().add(x),
                        v::mul(
                            v::splat(0.25),
                            v::add(v::add(v::add(taps[0], taps[1]), taps[2]), taps[3]),
                        ),
                    );
                }
            }
            super::half::<BIAS>(src.map(|s| &s[2 * end..]), &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        fn half_in_place(values: &mut [f32], w: usize, h: usize) {
            let ow = w / 2;
            let end = ow / $lanes * $lanes;
            for y in 0..h / 2 {
                unsafe {
                    for x in (0..end).step_by($lanes) {
                        let mut taps = [v::splat(0.0); 4];
                        for t in 0..4 {
                            let v: [f32; $lanes] = std::array::from_fn(|i| {
                                values[(2 * y + t % 2) * w + 2 * (x + i) + t / 2]
                            });
                            taps[t] = v::load(v.as_ptr());
                        }
                        v::store(
                            values.as_mut_ptr().add(y * ow + x),
                            v::mul(
                                v::splat(0.25),
                                v::add(v::add(v::add(taps[0], taps[1]), taps[2]), taps[3]),
                            ),
                        );
                    }
                }
                for x in end..ow {
                    let i = 2 * y * w + 2 * x;
                    values[y * ow + x] =
                        0.25 * (values[i] + values[i + w] + values[i + 1] + values[i + w + 1]);
                }
            }
        }
        #[target_feature(enable=$feature)]
        unsafe fn variance(mean: &[f32], square: &[f32], out: &mut [f32]) {
            let end = out.len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let m = v::load(mean.as_ptr().add(x));
                    v::store(
                        out.as_mut_ptr().add(x),
                        v::max(
                            v::sub(v::load(square.as_ptr().add(x)), v::mul(m, m)),
                            v::splat(0.0),
                        ),
                    );
                }
            }
            super::variance(&mean[end..], &square[end..], &mut out[end..]);
        }
        #[target_feature(enable=$feature)]
        unsafe fn pool(src: [&[f32]; 5], mut sum: [f32; 2]) -> [f32; 2] {
            let end = src[0].len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let [b, square, cross, a, variance] = src.map(|s| v::load(s.as_ptr().add(x)));
                    let zero = v::splat(0.0);
                    let one = v::splat(1.0);
                    let c = v::splat(0.0009);
                    let var = v::max(v::sub(square, v::mul(b, b)), zero);
                    let cov = v::mask_nonzero(v::sub(cross, v::mul(a, b)), variance, var);
                    let delta = v::sub(a, b);
                    let s = v::max(
                        v::sub(
                            one,
                            v::div(
                                v::mul(
                                    v::sub(one, v::mul(delta, delta)),
                                    v::add(v::mul(v::splat(2.0), cov), c),
                                ),
                                v::add(v::add(variance, var), c),
                            ),
                        ),
                        zero,
                    );
                    let fourth = v::mul(v::mul(v::mul(s, s), s), s);
                    let mut terms = [0.0; $lanes];
                    let mut powers = [0.0; $lanes];
                    v::store(terms.as_mut_ptr(), s);
                    v::store(powers.as_mut_ptr(), fourth);
                    for i in 0..$lanes {
                        sum[0] += terms[i];
                        sum[1] += powers[i];
                    }
                }
            }
            super::pool(src.map(|s| &s[end..]), sum)
        }
        #[target_feature(enable=$feature)]
        unsafe fn gradients(src: [&[f32]; 4], mut sum: [f32; 3]) -> [f32; 3] {
            let end = src[0].len() / $lanes * $lanes;
            unsafe {
                for x in (0..end).step_by($lanes) {
                    let [ai, aj, bi, bj] = src.map(|s| v::load(s.as_ptr().add(x)));
                    let a = v::sub(aj, ai);
                    let b = v::sub(bj, bi);
                    let zero = v::splat(0.0);
                    let error = v::abs(v::sub(b, a));
                    let excess = v::max(v::sub(v::abs(b), v::abs(a)), zero);
                    let loss = v::max(v::sub(v::abs(a), v::abs(b)), zero);
                    let loss4 = v::mul(v::mul(v::mul(loss, loss), loss), loss);
                    let mut e = [0.0; $lanes];
                    let mut ex = [0.0; $lanes];
                    let mut l = [0.0; $lanes];
                    v::store(e.as_mut_ptr(), error);
                    v::store(ex.as_mut_ptr(), excess);
                    v::store(l.as_mut_ptr(), loss4);
                    for i in 0..$lanes {
                        sum[0] += e[i];
                        sum[1] += ex[i];
                        sum[2] += l[i];
                    }
                }
            }
            super::gradients(src.map(|s| &s[end..]), sum)
        }
        // A target feature enabled for the whole build (wasm simd128) makes
        // these calls safe; the other backends still need the blocks.
        #[allow(unused_unsafe)]
        pub(super) fn kernels() -> super::Kernels {
            // The selector checks CPU features. The proxy supplies complete
            // source pairs for reduction and equal-length spans for scoring.
            super::Kernels {
                half: |s, o| unsafe { half::<false>(s, o) },
                half_y: |s, o| unsafe { half::<true>(s, o) },
                half_in_place: |v, w, h| unsafe { half_in_place(v, w, h) },
                variance: |m, s, o| unsafe { variance(m, s, o) },
                pool: |s, p| unsafe { pool(s, p) },
                gradients: |s, p| unsafe { gradients(s, p) },
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
pub(super) fn available() -> Vec<(&'static str, Kernels)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }
    #[test]
    fn pyramid_and_pooling_match_scalar_bits_for_every_tail() {
        let scalar = scalar();
        for (name, k) in available() {
            for n in 1..=35 {
                let w = 2 * n + 1;
                let h = 7;
                let src: Vec<_> = (0..w * h)
                    .map(|i| ((i * 1877 + 941) % 3253) as f32 * 0.0001 - 0.03)
                    .collect();
                for (a, b) in [(scalar.half, k.half), (scalar.half_y, k.half_y)] {
                    let mut expected = vec![0.; n];
                    let mut actual = vec![0.; n];
                    a([&src[..w], &src[w..2 * w]], &mut expected);
                    b([&src[..w], &src[w..2 * w]], &mut actual);
                    assert_eq!(bits(&actual), bits(&expected), "{name}, half n={n}");
                }
                let mut a = src.clone();
                let mut b = src.clone();
                (scalar.half_in_place)(&mut a, w, h);
                (k.half_in_place)(&mut b, w, h);
                assert_eq!(bits(&a), bits(&b), "{name}, in place n={n}");
                let reference: Vec<_> = (0..n).map(|i| [0.21, 0.19, 0.0, 0.31][i % 4]).collect();
                let mean: Vec<_> = reference.iter().map(|v| v + 0.01).collect();
                let square: Vec<_> = mean
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * v + if i % 3 == 0 { 0.0 } else { 0.0001 })
                    .collect();
                let cross: Vec<_> = reference
                    .iter()
                    .zip(&mean)
                    .map(|(a, b)| a * b + 0.00005)
                    .collect();
                let variance: Vec<_> = (0..n)
                    .map(|i| if i % 5 == 0 { 0.0 } else { 0.0001 })
                    .collect();
                let mut a = vec![0.; n];
                let mut b = vec![0.; n];
                (scalar.variance)(&mean, &square, &mut a);
                (k.variance)(&mean, &square, &mut b);
                assert_eq!(bits(&a), bits(&b), "{name}, variance n={n}");
                let rows = [&mean[..], &square, &cross, &reference, &variance];
                assert_eq!(
                    (scalar.pool)(rows, [0.25, 0.75]).map(f32::to_bits),
                    (k.pool)(rows, [0.25, 0.75]).map(f32::to_bits),
                    "{name}, pool n={n}"
                );
                let rows = [
                    &src[..n],
                    &src[1..n + 1],
                    &src[w..w + n],
                    &src[w + 1..w + n + 1],
                ];
                assert_eq!(
                    (scalar.gradients)(rows, [0.25, 0.75, 0.125]).map(f32::to_bits),
                    (k.gradients)(rows, [0.25, 0.75, 0.125]).map(f32::to_bits),
                    "{name}, gradients n={n}"
                );
            }
        }
    }
}
