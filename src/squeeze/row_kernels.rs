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

//! Exact squeeze rows. The narrow vector path checks every input block before
//! using i32 tendency arithmetic; wider samples retain the scalar i64 path.

use super::{average, smooth_tendency};
use crate::fast_divide::DividerI32;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub(super) struct ForwardRow<'a> {
    pub(super) pixels: [&'a [i32]; 2],
    pub(super) top: Option<&'a [i32]>,
    pub(super) next: [&'a [i32]; 2],
}

#[derive(Clone, Copy)]
pub(crate) struct InverseRow<'a> {
    pub(crate) average: &'a [i32],
    pub(crate) residual: &'a [i32],
    pub(crate) top: Option<&'a [i32]>,
    pub(crate) next: &'a [i32],
}

impl ForwardRow<'_> {
    fn slice(&self, range: std::ops::Range<usize>) -> ForwardRow<'_> {
        ForwardRow {
            pixels: self.pixels.map(|r| &r[range.clone()]),
            top: self.top.map(|r| &r[range.clone()]),
            next: self.next.map(|r| &r[range.clone()]),
        }
    }

    fn validate(&self, len: usize) {
        assert!(
            self.pixels
                .into_iter()
                .chain(self.next)
                .all(|r| r.len() >= len)
        );
        assert!(self.top.is_none_or(|r| r.len() >= len));
    }
}

impl InverseRow<'_> {
    fn slice(&self, range: std::ops::Range<usize>) -> InverseRow<'_> {
        InverseRow {
            average: &self.average[range.clone()],
            residual: &self.residual[range.clone()],
            top: self.top.map(|r| &r[range.clone()]),
            next: &self.next[range],
        }
    }

    fn validate(&self, len: usize) {
        assert!(
            [self.average, self.residual, self.next]
                .into_iter()
                .all(|r| r.len() >= len)
        );
        assert!(self.top.is_none_or(|r| r.len() >= len));
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ClosedRow<'a> {
    pub(crate) exact: InverseRow<'a>,
    pub(crate) average: &'a [i32],
    pub(crate) top: &'a [i32],
    pub(crate) next: &'a [i32],
}

impl ClosedRow<'_> {
    fn slice(&self, range: std::ops::Range<usize>) -> ClosedRow<'_> {
        ClosedRow {
            exact: self.exact.slice(range.clone()),
            average: &self.average[range.clone()],
            top: &self.top[range.clone()],
            next: &self.next[range],
        }
    }

    fn validate(&self, len: usize) {
        self.exact.validate(len);
        assert!(
            [self.average, self.top, self.next]
                .into_iter()
                .all(|r| r.len() >= len)
        );
    }
}

pub(crate) type ClosedVerticalFn = fn(ClosedRow<'_>, ResidualRounding, [&mut [i32]; 5]);

/// Deadzone rounding for one step `q`, without float division when
/// there is no deadzone: `floor(|r| / q + 1/2)` is then the integer rounded
/// quotient, exact through a precomputed divider for every `|r| < 2^23`
/// (beyond that the f32 path is kept, bit for bit).
#[derive(Clone, Copy)]
pub(crate) struct ResidualRounding {
    q: i64,
    half: i32,
    divider: DividerI32,
    deadzone: f32,
}

impl ResidualRounding {
    const EXACT_MAGNITUDE: u64 = 1 << 23;

    pub(crate) fn new(q: i64, deadzone: f32) -> Self {
        let step = q.clamp(1, i64::from(i32::MAX)) as i32;
        Self {
            q,
            half: step / 2,
            divider: DividerI32::new(step),
            deadzone,
        }
    }

    #[inline(always)]
    pub(crate) fn apply(self, r: i64) -> i64 {
        let magnitude = r.unsigned_abs();
        if self.q <= 1 || self.deadzone != 0.0 || magnitude >= Self::EXACT_MAGNITUDE {
            if self.q <= 1 {
                return r;
            }
            let f = magnitude as f32 / self.q as f32;
            let m = (f + 0.5 - self.deadzone).max(0.0) as i64;
            return if r < 0 { -m } else { m };
        }
        let m = i64::from((magnitude as i32 + self.half) / self.divider);
        if r < 0 { -m } else { m }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub(super) forward_horizontal: fn(&[i32], &mut [i32], &mut [i32]),
    pub(super) forward_vertical: fn(ForwardRow<'_>, [&mut [i32]; 2]),
    pub(super) closed_vertical: ClosedVerticalFn,
    pub(super) inverse_vertical: fn(InverseRow<'_>, [&mut [i32]; 2]),
}

pub(super) fn selected() -> &'static Kernels {
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
    scalar()
}

fn horizontal_span(row: &[i32], start: usize, avg: &mut [i32], res: &mut [i32]) {
    let (pairs, tail) = row.as_chunks::<2>();
    for (x, (avg, res)) in (start..).zip(avg.iter_mut().zip(res)) {
        let [a, b] = pairs[x];
        let av = average(a, b);
        let next = pairs.get(x + 1).map_or_else(
            || tail.first().copied().unwrap_or(av),
            |&[a, b]| average(a, b),
        );
        let left = if x == 0 { av } else { pairs[x - 1][1] };
        *avg = av;
        // Cast only after subtraction, retaining release wrapping for wide
        // differences without overflowing an intermediate i32 in debug builds.
        *res = (i64::from(a)
            - i64::from(b)
            - smooth_tendency(i64::from(left), i64::from(av), i64::from(next)))
            as i32;
    }
}

fn forward_horizontal(row: &[i32], avg: &mut [i32], res: &mut [i32]) {
    assert_eq!(avg.len(), row.len().div_ceil(2));
    assert_eq!(res.len(), row.len() / 2);
    horizontal_span(row, 0, avg, res);
    avg[res.len()..].copy_from_slice(row.as_chunks::<2>().1);
}

fn forward_vertical(row: ForwardRow<'_>, [avg, res]: [&mut [i32]; 2]) {
    assert_eq!(avg.len(), res.len());
    row.validate(avg.len());
    let mut top = row.top.into_iter().flatten();
    let inputs = row.pixels[0]
        .iter()
        .zip(row.pixels[1])
        .zip(row.next[0].iter().zip(row.next[1]));
    for (((&a, &b), (&na, &nb)), (avg, res)) in inputs.zip(avg.iter_mut().zip(res)) {
        let av = average(a, b);
        let top = top.next().copied().unwrap_or(av);
        *avg = av;
        *res = (i64::from(a)
            - i64::from(b)
            - smooth_tendency(i64::from(top), i64::from(av), i64::from(average(na, nb))))
            as i32;
    }
}

fn inverse_vertical(row: InverseRow<'_>, [a, b]: [&mut [i32]; 2]) {
    assert_eq!(a.len(), b.len());
    row.validate(a.len());
    let mut top = row.top.into_iter().flatten();
    for (((&av, &res), &next), (a, b)) in row
        .average
        .iter()
        .zip(row.residual)
        .zip(row.next)
        .zip(a.iter_mut().zip(b))
    {
        let av = i64::from(av);
        let top = top.next().map_or(av, |&v| i64::from(v));
        let diff = i64::from(res) + smooth_tendency(top, av, i64::from(next));
        let first = av + diff / 2;
        *a = first as i32;
        *b = (first - diff) as i32;
    }
}

fn closed_vertical(
    row: ClosedRow<'_>,
    rounding: ResidualRounding,
    [x0, x1, residual, out0, out1]: [&mut [i32]; 5],
) {
    let len = residual.len();
    row.validate(len);
    assert!(
        [x0.len(), x1.len(), out0.len(), out1.len()]
            .into_iter()
            .all(|n| n == len)
    );
    let mut top_exact = row.exact.top.into_iter().flatten();
    let exact = row
        .exact
        .average
        .iter()
        .zip(row.exact.next)
        .zip(row.exact.residual);
    let decoded = row.average.iter().zip(row.next).zip(row.top);
    let outputs = x0
        .iter_mut()
        .zip(x1)
        .zip(residual)
        .zip(out0.iter_mut().zip(out1));
    for ((exact, decoded), outputs) in exact.zip(decoded).zip(outputs) {
        let ((&av, &next), &res) = exact;
        let ((&av_dec, &next_dec), &top_dec) = decoded;
        let (((x0, x1), residual), (out0, out1)) = outputs;
        let av = i64::from(av);
        let top = top_exact.next().map_or(av, |&v| i64::from(v));
        let diff = i64::from(res) + smooth_tendency(top, av, i64::from(next));
        let first = av + diff / 2;
        *x0 = first as i32;
        *x1 = (first - diff) as i32;
        let tendency = smooth_tendency(i64::from(top_dec), i64::from(av_dec), i64::from(next_dec));
        let quantized = rounding.apply(i64::from(*x0) - i64::from(*x1) - tendency);
        *residual = quantized as i32;
        let diff = quantized * rounding.q + tendency;
        let first = i64::from(av_dec) + diff / 2;
        *out0 = first as i32;
        *out1 = (first - diff) as i32;
    }
}

pub(super) fn scalar() -> Kernels {
    Kernels {
        forward_horizontal,
        forward_vertical,
        inverse_vertical,
        closed_vertical,
    }
}

// All inputs have magnitude below 2^26 in the vector arm. The tendency's
// largest numerator is then below 2^29 + 6; averages, clamps, residuals and
// inverse reconstruction also fit i32. Every other block uses i64 above.
#[allow(unused_macros)]
macro_rules! backend {
    ($feature:literal, $lanes:literal, $vector:ty) => {
        #[inline]
        #[target_feature(enable = $feature)]
        fn narrow<const N: usize>(values: [$vector; N]) -> bool {
            bits_clear(
                values
                    .into_iter()
                    .fold(splat(0), |bits, v| or(bits, abs(v))),
                splat(!((1 << 26) - 1)),
            )
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn average(a: $vector, b: $vector) -> $vector {
            shr::<1>(add(add(a, b), and(gt(a, b), splat(1))))
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn tendency(b: $vector, a: $vector, n: $vector) -> $vector {
            let one = splat(1);
            let ge = |a, b| or(gt(a, b), eq(a, b));
            let down = and(ge(b, a), ge(a, n));
            let up = and(ge(a, b), ge(n, a));
            let numerator = sub(sub(add(add(b, b), add(b, b)), add(add(n, n), n)), a);
            let numerator = add(numerator, select(down, splat(6), splat(-6)));
            // Signed division by 12, truncated toward zero, through the high
            // half of a signed product and its sign correction.
            let diff = sub(
                shr::<1>(mulhi(numerator, splat(0x2aaa_aaab))),
                shr::<31>(numerator),
            );
            let ba = sub(b, a);
            let an = sub(a, n);
            let ba = add(ba, ba);
            let an = add(an, an);
            let d = select(gt(sub(diff, and(diff, one)), ba), add(ba, one), diff);
            let d = select(gt(add(d, and(d, one)), an), an, d);
            let u = select(gt(ba, add(diff, and(diff, one))), sub(ba, one), diff);
            let u = select(gt(an, sub(u, and(u, one))), an, u);
            select(down, d, select(up, u, splat(0)))
        }

        #[target_feature(enable = $feature)]
        fn forward_horizontal(row: &[i32], avg: &mut [i32], res: &mut [i32]) {
            assert_eq!(avg.len(), row.len().div_ceil(2));
            let rw = row.len() / 2;
            assert_eq!(res.len(), rw);
            if rw < $lanes + 2 {
                super::forward_horizontal(row, avg, res);
                return;
            }
            super::horizontal_span(row, 0, &mut avg[..1], &mut res[..1]);
            let (averages, avg_tail) = avg[1..rw - 1].as_chunks_mut::<$lanes>();
            let (residuals, res_tail) = res[1..rw - 1].as_chunks_mut::<$lanes>();
            for (i, (avg, res)) in averages.iter_mut().zip(residuals).enumerate() {
                let x = 1 + i * $lanes;
                // SAFETY: every interior block has a complete preceding and
                // following pair; each load reads exactly 2 * lanes samples.
                let (a, b) = unsafe { load_pairs(row.as_ptr().add(2 * x)) };
                let (na, nb) = unsafe { load_pairs(row.as_ptr().add(2 * (x + 1))) };
                let (_, left) = unsafe { load_pairs(row.as_ptr().add(2 * (x - 1))) };
                if narrow([a, b, na, nb, left]) {
                    let av = average(a, b);
                    let r = sub(sub(a, b), tendency(left, av, average(na, nb)));
                    unsafe {
                        store(avg.as_mut_ptr(), av);
                        store(res.as_mut_ptr(), r);
                    }
                } else {
                    super::horizontal_span(row, x, avg, res);
                }
            }
            let start = rw - 1 - res_tail.len();
            super::horizontal_span(row, start, avg_tail, res_tail);
            super::horizontal_span(row, rw - 1, &mut avg[rw - 1..rw], &mut res[rw - 1..]);
            avg[rw..].copy_from_slice(row.as_chunks::<2>().1);
        }

        #[target_feature(enable = $feature)]
        fn forward_vertical(row: super::ForwardRow<'_>, [avg, res]: [&mut [i32]; 2]) {
            if avg.len() < $lanes {
                super::forward_vertical(row, [avg, res]);
                return;
            }
            assert_eq!(avg.len(), res.len());
            row.validate(avg.len());
            let (averages, avg_tail) = avg.as_chunks_mut::<$lanes>();
            let (residuals, res_tail) = res.as_chunks_mut::<$lanes>();
            let end = averages.len() * $lanes;
            for (i, (avg, res)) in averages.iter_mut().zip(residuals).enumerate() {
                let x = i * $lanes;
                // SAFETY: validate and as_chunks establish complete, unaligned
                // source and destination vectors for this block.
                let [a, b] = row.pixels.map(|r| unsafe { load(r.as_ptr().add(x)) });
                let [na, nb] = row.next.map(|r| unsafe { load(r.as_ptr().add(x)) });
                let av = average(a, b);
                let top = row.top.map_or(av, |r| unsafe { load(r.as_ptr().add(x)) });
                if narrow([a, b, na, nb, top]) {
                    let r = sub(sub(a, b), tendency(top, av, average(na, nb)));
                    unsafe {
                        store(avg.as_mut_ptr(), av);
                        store(res.as_mut_ptr(), r);
                    }
                } else {
                    super::forward_vertical(row.slice(x..x + $lanes), [avg, res]);
                }
            }
            super::forward_vertical(row.slice(end..end + res_tail.len()), [avg_tail, res_tail]);
        }

        #[target_feature(enable = $feature)]
        fn inverse_vertical(row: super::InverseRow<'_>, [a, b]: [&mut [i32]; 2]) {
            if a.len() < $lanes {
                super::inverse_vertical(row, [a, b]);
                return;
            }
            assert_eq!(a.len(), b.len());
            row.validate(a.len());
            let (first, a_tail) = a.as_chunks_mut::<$lanes>();
            let (second, b_tail) = b.as_chunks_mut::<$lanes>();
            let end = first.len() * $lanes;
            for (i, (a, b)) in first.iter_mut().zip(second).enumerate() {
                let x = i * $lanes;
                // SAFETY: validate and as_chunks bound every vector access.
                let av = unsafe { load(row.average.as_ptr().add(x)) };
                let res = unsafe { load(row.residual.as_ptr().add(x)) };
                let next = unsafe { load(row.next.as_ptr().add(x)) };
                let top = row.top.map_or(av, |r| unsafe { load(r.as_ptr().add(x)) });
                if narrow([av, res, next, top]) {
                    let diff = add(res, tendency(top, av, next));
                    let half = shr::<1>(add(diff, and(shr::<31>(diff), splat(1))));
                    let first = add(av, half);
                    unsafe {
                        store(a.as_mut_ptr(), first);
                        store(b.as_mut_ptr(), sub(first, diff));
                    }
                } else {
                    super::inverse_vertical(row.slice(x..x + $lanes), [a, b]);
                }
            }
            super::inverse_vertical(row.slice(end..end + a_tail.len()), [a_tail, b_tail]);
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn half(v: $vector) -> $vector {
            shr::<1>(add(v, and(shr::<31>(v), splat(1))))
        }

        #[target_feature(enable = $feature)]
        fn closed_vertical(
            row: super::ClosedRow<'_>,
            rounding: super::ResidualRounding,
            outputs: [&mut [i32]; 5],
        ) {
            let len = outputs[2].len();
            row.validate(len);
            assert!(outputs.iter().all(|r| r.len() == len));
            // With |input| < 2^24 and q <= 2^24, both tendencies, the
            // reconstructed samples, rounding and dequantization fit i32.
            // Larger steps/samples retain the scalar i64 reconstruction.
            if rounding.q < 1
                || rounding.q > 1 << 24
                || (rounding.q > 1 && !(0.0..=0.5).contains(&rounding.deadzone))
            {
                super::closed_vertical(row, rounding, outputs);
                return;
            }
            let [(a, at), (b, bt), (res, rt), (out0, ot0), (out1, ot1)] =
                outputs.map(|r| r.as_chunks_mut::<$lanes>());
            let end = res.len() * $lanes;
            let vectors = a.iter_mut().zip(b).zip(res).zip(out0.iter_mut().zip(out1));
            for (i, (((a, b), residual), (out0, out1))) in vectors.enumerate() {
                let x = i * $lanes;
                let [av, next, exact_res, av_dec, next_dec, top_dec] = [
                    row.exact.average,
                    row.exact.next,
                    row.exact.residual,
                    row.average,
                    row.next,
                    row.top,
                ]
                .map(|r| unsafe { load(r.as_ptr().add(x)) });
                let top = row
                    .exact
                    .top
                    .map_or(av, |r| unsafe { load(r.as_ptr().add(x)) });
                let bits = [av, next, exact_res, top, av_dec, next_dec, top_dec]
                    .into_iter()
                    .fold(splat(0), |bits, v| or(bits, abs(v)));
                if !bits_clear(bits, splat(!((1 << 24) - 1))) {
                    super::closed_vertical(
                        row.slice(x..x + $lanes),
                        rounding,
                        [a, b, residual, out0, out1],
                    );
                    continue;
                }
                let diff = add(exact_res, tendency(top, av, next));
                let first = add(av, half(diff));
                let second = sub(first, diff);
                let tendency = tendency(top_dec, av_dec, next_dec);
                let r = sub(sub(first, second), tendency);
                let quantized = if rounding.q <= 1 {
                    r
                } else {
                    let magnitude = abs(r);
                    let m = if rounding.deadzone == 0.0 {
                        // The integer shortcut has the same 2^23 bound as
                        // scalar rounding. Preserve its float path beyond it.
                        if !bits_clear(magnitude, splat(!((1 << 23) - 1))) {
                            super::closed_vertical(
                                row.slice(x..x + $lanes),
                                rounding,
                                [a, b, residual, out0, out1],
                            );
                            continue;
                        }
                        let numerator = add(magnitude, splat(rounding.half));
                        let divider = rounding.divider;
                        if divider.magic() == 0 {
                            shr_by(numerator, divider.shift())
                        } else {
                            let mut quotient = mulhi(numerator, splat(divider.magic()));
                            if divider.adds_numerator() {
                                quotient = add(quotient, numerator);
                            }
                            let quotient = shr_by(quotient, divider.shift());
                            sub(quotient, shr::<31>(quotient))
                        }
                    } else {
                        let scaled = f::div(f::from_i32(magnitude), f::splat(rounding.q as f32));
                        let shifted =
                            f::sub(f::add(scaled, f::splat(0.5)), f::splat(rounding.deadzone));
                        f::trunc_to_i32(f::max(shifted, f::splat(0.0)))
                    };
                    select(gt(splat(0), r), sub(splat(0), m), m)
                };
                let decoded_diff = add(mul(quantized, splat(rounding.q as i32)), tendency);
                let decoded_first = add(av_dec, half(decoded_diff));
                unsafe {
                    store(a.as_mut_ptr(), first);
                    store(b.as_mut_ptr(), second);
                    store(residual.as_mut_ptr(), quantized);
                    store(out0.as_mut_ptr(), decoded_first);
                    store(out1.as_mut_ptr(), sub(decoded_first, decoded_diff));
                }
            }
            super::closed_vertical(row.slice(end..len), rounding, [at, bt, rt, ot0, ot1]);
        }

        #[allow(unused_unsafe)]
        pub(super) fn kernels() -> super::Kernels {
            // The selector checks this backend's CPU features once.
            super::Kernels {
                forward_horizontal: |r, a, b| unsafe { forward_horizontal(r, a, b) },
                forward_vertical: |r, o| unsafe { forward_vertical(r, o) },
                inverse_vertical: |r, o| unsafe { inverse_vertical(r, o) },
                closed_vertical: |r, q, o| unsafe { closed_vertical(r, q, o) },
            }
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use crate::simd::neon::f32 as f;
    use crate::simd::neon::i32::*;
    backend!("neon", 4, Vector);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse {
    use crate::simd::sse::f32 as f;
    use crate::simd::sse::i32::*;
    backend!("sse4.1", 4, Vector);
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use crate::simd::avx::f32 as f;
    use crate::simd::avx::i32::*;
    backend!("avx2", 8, Vector);
}

#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm {
    use crate::simd::wasm::f32 as f;
    use crate::simd::wasm::i32::*;
    backend!("simd128", 4, Vector);
}

#[cfg(test)]
pub(super) fn available() -> Vec<(&'static str, Kernels)> {
    #[allow(unused_mut)]
    let mut result = vec![("scalar", scalar()), ("selected", *selected())];
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        result.push(("avx2", avx::kernels()));
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        result.push(("sse4.1", sse::kernels()));
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    result.push(("neon", neon::kernels()));
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    result.push(("wasm", wasm::kernels()));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(seed: &mut u64) -> i32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as i32
    }

    fn check_closed(row: ClosedRow<'_>, q: i64, deadzone: f32) {
        let rounding = ResidualRounding::new(q, deadzone);
        let len = row.average.len();
        let mut expected: [Vec<i32>; 5] = std::array::from_fn(|_| vec![0; len]);
        (scalar().closed_vertical)(row, rounding, expected.each_mut().map(Vec::as_mut_slice));
        for (name, kernels) in available() {
            let mut actual = expected.clone();
            actual.iter_mut().for_each(|r| r.fill(i32::MIN));
            (kernels.closed_vertical)(row, rounding, actual.each_mut().map(Vec::as_mut_slice));
            assert_eq!(
                actual, expected,
                "{name}, len={len}, q={q}, deadzone={deadzone}"
            );
        }
    }

    #[test]
    fn closed_vertical_matches_i64_for_all_tails_and_wide_fallbacks() {
        let mut seed = 1234567;
        for len in (0..=35).chain([63, 64, 65, 127]) {
            for offset in 0..4 {
                for shift in [20, 7, 0] {
                    let data: [Vec<i32>; 7] = std::array::from_fn(|_| {
                        (0..len + offset)
                            .map(|_| random(&mut seed) >> shift)
                            .collect()
                    });
                    let [av, res, next, top, dec, next_dec, top_dec] =
                        data.each_ref().map(|r| &r[offset..]);
                    for top in [None, Some(top)] {
                        let row = ClosedRow {
                            exact: InverseRow {
                                average: av,
                                residual: res,
                                top,
                                next,
                            },
                            average: dec,
                            top: top_dec,
                            next: next_dec,
                        };
                        for q in [
                            1,
                            2,
                            3,
                            7,
                            40,
                            256,
                            65535,
                            1 << 20,
                            1 << 24,
                            (1 << 24) + 1,
                            i64::from(i32::MAX),
                        ] {
                            for deadzone in [0.0, 0.15, 0.5] {
                                check_closed(row, q, deadzone);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn closed_vertical_preserves_rounding_thresholds_and_scalar_special_cases() {
        let limit = 1 << 24;
        for q in [
            1i64,
            2,
            3,
            7,
            12,
            255,
            256,
            65535,
            1 << 20,
            1 << 24,
            i64::from(i32::MAX),
        ] {
            let mut residuals = vec![i32::MIN, i32::MAX];
            for magnitude in [
                (1 << 23) - 1,
                1 << 23,
                (1 << 23) + 1,
                limit - 1,
                limit,
                limit + 1,
            ] {
                residuals.extend([magnitude, -magnitude]);
            }
            for k in [0, 1, 2, 7, 100] {
                for delta in [-1, 0, 1] {
                    let v = k * q + q / 2 + delta;
                    if v >= 0 && v <= i64::from(i32::MAX) {
                        residuals.extend([v as i32, -v as i32]);
                    }
                }
            }
            for offset in 0..8 {
                let residuals = &residuals[offset..];
                let zero = vec![0; residuals.len()];
                let row = ClosedRow {
                    exact: InverseRow {
                        average: &zero,
                        residual: residuals,
                        top: None,
                        next: &zero,
                    },
                    average: &zero,
                    top: &zero,
                    next: &zero,
                };
                for deadzone in [0.0, -0.0, 0.15, 0.5, -0.1, 0.75, f32::INFINITY, f32::NAN] {
                    check_closed(row, q, deadzone);
                }
            }
        }
    }

    #[test]
    fn rows_match_wide_scalar_for_tails_and_fallbacks() {
        let reference = scalar();
        let mut seed = 1234567;
        for len in (0..=67).chain([127, 128, 129, 257]) {
            for offset in 0..4 {
                for wide in [false, true] {
                    let rows: [Vec<i32>; 5] = std::array::from_fn(|_| {
                        (0..len + offset)
                            .map(|_| {
                                let v = random(&mut seed);
                                if wide { v } else { v >> 17 }
                            })
                            .collect()
                    });
                    let [a, b, top, na, nb] = rows.each_ref().map(|r| &r[offset..]);
                    for top in [None, Some(top)] {
                        let forward = ForwardRow {
                            pixels: [a, b],
                            top,
                            next: [na, nb],
                        };
                        let inverse = InverseRow {
                            average: a,
                            residual: b,
                            top,
                            next: na,
                        };
                        let mut expected_f = [vec![0; len], vec![0; len]];
                        let mut expected_i = expected_f.clone();
                        let [ea, eb] = &mut expected_f;
                        (reference.forward_vertical)(forward, [ea, eb]);
                        let [ea, eb] = &mut expected_i;
                        (reference.inverse_vertical)(inverse, [ea, eb]);
                        for (name, kernels) in available() {
                            let mut actual = [vec![0; len], vec![0; len]];
                            let [aa, ab] = &mut actual;
                            (kernels.forward_vertical)(forward, [aa, ab]);
                            assert_eq!(actual, expected_f, "{name} forward {len}/{offset}/{wide}");
                            let [aa, ab] = &mut actual;
                            (kernels.inverse_vertical)(inverse, [aa, ab]);
                            assert_eq!(actual, expected_i, "{name} inverse {len}/{offset}/{wide}");
                        }
                    }
                    let mut expected_a = vec![0; len.div_ceil(2)];
                    let mut expected_r = vec![0; len / 2];
                    (reference.forward_horizontal)(a, &mut expected_a, &mut expected_r);
                    for (name, kernels) in available() {
                        let mut actual_a = expected_a.clone();
                        let mut actual_r = expected_r.clone();
                        actual_a.fill(i32::MIN);
                        actual_r.fill(i32::MIN);
                        (kernels.forward_horizontal)(a, &mut actual_a, &mut actual_r);
                        assert_eq!(
                            actual_a, expected_a,
                            "{name} horizontal average {len}/{offset}/{wide}"
                        );
                        assert_eq!(
                            actual_r, expected_r,
                            "{name} horizontal residual {len}/{offset}/{wide}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tendency_clamps_and_narrow_bounds_match_i64() {
        let limit = 1 << 26;
        let values = [
            i32::MIN,
            -limit - 1,
            -limit,
            -limit + 1,
            -65535,
            -19,
            -2,
            -1,
            0,
            1,
            2,
            19,
            65535,
            limit - 1,
            limit,
            limit + 1,
            i32::MAX,
        ];
        let mut top = Vec::new();
        let mut average = Vec::new();
        let mut next = Vec::new();
        for b in values {
            for a in values {
                for n in values {
                    top.push(b);
                    average.push(a);
                    next.push(n);
                }
            }
        }
        let residual: Vec<i32> = (0..top.len()).map(|i| values[i % values.len()]).collect();
        let row = InverseRow {
            average: &average,
            residual: &residual,
            top: Some(&top),
            next: &next,
        };
        let mut expected = [vec![0; top.len()], vec![0; top.len()]];
        let [a, b] = &mut expected;
        (scalar().inverse_vertical)(row, [a, b]);
        for (name, kernels) in available() {
            let mut actual = expected.clone();
            actual.iter_mut().for_each(|r| r.fill(0));
            let [a, b] = &mut actual;
            (kernels.inverse_vertical)(row, [a, b]);
            assert_eq!(actual, expected, "{name}");
        }
    }
}
