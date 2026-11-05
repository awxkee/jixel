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

//! Fixed modular predictors and RCT residual rows. Weighted prediction keeps
//! its ordered state walk; histogram updates remain scalar after each vector.

use super::{PredictorNeighbors, predictor_value};
use crate::entropy::pack_signed;
use crate::lossless::rct::RctHistogram;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub(crate) struct Rows<'a, T> {
    pub(in crate::lossless) current: &'a [T],
    pub(in crate::lossless) north: Option<&'a [T]>,
    pub(in crate::lossless) north_north: Option<&'a [T]>,
}

impl<T: Copy> Rows<'_, T>
where
    i64: From<T>,
{
    fn validate(&self, len: usize) {
        assert_eq!(self.current.len(), len);
        assert!(self.north.is_none_or(|r| r.len() == len));
        assert!(self.north_north.is_none_or(|r| r.len() == len));
        assert!(self.north.is_some() || self.north_north.is_none());
    }

    fn neighbors(&self, x: usize) -> PredictorNeighbors {
        let left = if x > 0 {
            i64::from(self.current[x - 1])
        } else {
            self.north.map_or(0, |r| i64::from(r[x]))
        };
        let top = self.north.map_or(left, |r| i64::from(r[x]));
        let top_right = self
            .north
            .and_then(|r| r.get(x + 1))
            .map_or(top, |&v| i64::from(v));
        PredictorNeighbors {
            left,
            top,
            top_left: if x > 0 {
                self.north.map_or(left, |r| i64::from(r[x - 1]))
            } else {
                left
            },
            top_right,
            left_left: if x > 1 {
                i64::from(self.current[x - 2])
            } else {
                left
            },
            top_top: self.north_north.map_or(top, |r| i64::from(r[x])),
            top_right_right: self
                .north
                .and_then(|r| r.get(x + 2))
                .map_or(top_right, |&v| i64::from(v)),
        }
    }
}

pub(crate) trait Sample: Copy {
    fn pack(kernels: &Kernels, rows: Rows<'_, Self>, pred: u32, out: &mut [u32]);
    fn costs(
        kernels: &Kernels,
        rows: Rows<'_, Self>,
        hist: &mut [Vec<u64>],
        zero: Option<&mut Vec<u64>>,
        weighted_zero: bool,
    );
}

macro_rules! sample {
    ($sample:ty, $field:ident, $costs:ident) => {
        impl Sample for $sample {
            fn pack(kernels: &Kernels, rows: Rows<'_, Self>, pred: u32, out: &mut [u32]) {
                (kernels.$field)(rows, pred, out);
            }
            fn costs(
                kernels: &Kernels,
                rows: Rows<'_, Self>,
                hist: &mut [Vec<u64>],
                zero: Option<&mut Vec<u64>>,
                weighted_zero: bool,
            ) {
                (kernels.$costs)(rows, hist, zero, weighted_zero);
            }
        }
    };
}
sample!(i32, i32, costs_i32);
sample!(u8, u8, costs_u8);
sample!(u16, u16, costs_u16);

type CostRowFn<T> = fn(Rows<'_, T>, &mut [Vec<u64>], Option<&mut Vec<u64>>, bool);

#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    i32: fn(Rows<'_, i32>, u32, &mut [u32]),
    u8: fn(Rows<'_, u8>, u32, &mut [u32]),
    u16: fn(Rows<'_, u16>, u32, &mut [u32]),
    costs_i32: CostRowFn<i32>,
    costs_u8: CostRowFn<u8>,
    costs_u16: CostRowFn<u16>,
    pub(in crate::lossless) rct: fn(&[i32], &[i32], bool, &mut RctHistogram) -> u64,
}

pub(in crate::lossless) fn selected() -> &'static Kernels {
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

fn pack_span<T: Copy>(rows: Rows<'_, T>, pred: u32, start: usize, out: &mut [u32])
where
    i64: From<T>,
{
    for (x, out) in (start..).zip(out) {
        let prediction = predictor_value(pred, rows.neighbors(x), 0);
        *out = pack_signed((i64::from(rows.current[x]) - prediction) as i32);
    }
}

fn pack_scalar<T: Copy>(rows: Rows<'_, T>, pred: u32, out: &mut [u32])
where
    i64: From<T>,
{
    rows.validate(out.len());
    assert!(pred < 14 && pred != 6, "weighted prediction requires state");
    pack_span(rows, pred, 0, out);
}

fn cost_span<T: Copy>(
    rows: Rows<'_, T>,
    range: std::ops::Range<usize>,
    hist: &mut [Vec<u64>],
    zero: &mut Option<&mut Vec<u64>>,
    weighted_zero: bool,
) where
    i64: From<T>,
{
    for x in range {
        let value = i64::from(rows.current[x]);
        let n = rows.neighbors(x);
        let raw = pack_signed(value as i32);
        if weighted_zero {
            super::cost::add_symbol(&mut hist[0], raw);
        }
        if let Some(zero) = zero.as_deref_mut() {
            super::cost::add_symbol(zero, raw);
        }
        for (hist, &pred) in hist[1..].iter_mut().zip(&super::SLOW_PREDICTORS[1..]) {
            super::cost::add_symbol(
                hist,
                pack_signed((value - predictor_value(pred, n, 0)) as i32),
            );
        }
    }
}

fn cost_scalar<T: Copy>(
    rows: Rows<'_, T>,
    hist: &mut [Vec<u64>],
    mut zero: Option<&mut Vec<u64>>,
    weighted_zero: bool,
) where
    i64: From<T>,
{
    rows.validate(rows.current.len());
    assert_eq!(hist.len(), super::SLOW_PREDICTORS.len());
    cost_span(rows, 0..rows.current.len(), hist, &mut zero, weighted_zero);
}

#[allow(dead_code)]
pub(in crate::lossless) fn scalar() -> Kernels {
    Kernels {
        i32: pack_scalar,
        u8: pack_scalar,
        u16: pack_scalar,
        costs_i32: cost_scalar,
        costs_u8: cost_scalar,
        costs_u16: cost_scalar,
        rct: crate::lossless::rct::rct_gradient_row_scalar,
    }
}

#[allow(unused_macros)]
macro_rules! backend {
    ($feature:literal, $lanes:literal, $vector:ty) => {
        #[inline]
        #[target_feature(enable = $feature)]
        fn gradient(w: $vector, n: $vector, nw: $vector) -> $vector {
            let lo = min(w, n);
            let hi = max(w, n);
            // When NW lies between W and N, W + N - NW also lies between
            // them. Otherwise the prediction is an endpoint. Wrapped vector
            // arithmetic therefore reproduces the widened clamp for all i32.
            select(gt(nw, hi), lo, select(gt(lo, nw), hi, sub(add(w, n), nw)))
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn average(a: $vector, b: $vector) -> $vector {
            let different = xor(a, b);
            let floor = add(and(a, b), shr::<1>(different));
            add(floor, and(and(shr::<31>(floor), different), splat(1)))
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn prediction<const PRED: u32>([w, n, nw, ne, ww, nn, nee]: [$vector; 7]) -> $vector {
            match PRED {
                0 => splat(0),
                1 => w,
                2 => n,
                3 => average(w, n),
                4 => {
                    let left_distance = sub(max(n, nw), min(n, nw));
                    let top_distance = sub(max(w, nw), min(w, nw));
                    select(lt_unsigned(left_distance, top_distance), w, n)
                }
                5 => gradient(w, n, nw),
                7 => ne,
                8 => nw,
                9 => ww,
                10 => average(w, nw),
                11 => average(nw, n),
                12 => average(n, ne),
                13 => {
                    let sum = add(
                        add(
                            sub(sub(shl::<3>(n), shl::<1>(n)), shl::<1>(nn)),
                            sub(shl::<3>(w), w),
                        ),
                        add(add(ww, nee), add(add(shl::<1>(ne), ne), splat(8))),
                    );
                    shr::<4>(add(sum, and(shr::<31>(sum), splat(15))))
                }
                _ => unreachable!(),
            }
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn packed(value: $vector, prediction: $vector) -> $vector {
            let residual = sub(value, prediction);
            xor(shl::<1>(residual), shr::<31>(residual))
        }

        #[inline]
        #[target_feature(enable = $feature)]
        fn symbols(value: $vector) -> ($vector, $vector) {
            // Retain at most 24 leading bits before conversion. Every chosen
            // integer is represented exactly by f32, so its exponent and top
            // two fraction bits give the original integer's log2 and mantissa
            // without rounding at powers of two, including u32::MAX.
            let high = lt_unsigned(splat((1 << 24) - 1), value);
            let exact = select(high, shr_unsigned::<8>(value), max(value, splat(1)));
            let bits = to_f32_bits(exact);
            let log = add(
                sub(shr_unsigned::<23>(bits), splat(127)),
                and(high, splat(8)),
            );
            let large = lt_unsigned(splat(15), value);
            let token = add(shl::<2>(log), and(shr_unsigned::<21>(bits), splat(3)));
            (select(large, token, value), and(large, sub(log, splat(2))))
        }

        #[target_feature(enable = $feature)]
        fn pack_impl<T: Copy, const CHECK: bool, const PRED: u32>(
            rows: super::Rows<'_, T>,
            out: &mut [u32],
            load: impl Fn(*const T) -> $vector,
        ) where
            i64: From<T>,
        {
            rows.validate(out.len());
            let len = out.len();
            if len < $lanes + 4 {
                super::pack_scalar(rows, PRED, out);
                return;
            }
            super::pack_span(rows, PRED, 0, &mut out[..2]);
            let (vectors, tail) = out[2..len - 2].as_chunks_mut::<$lanes>();
            let end = 2 + vectors.len() * $lanes;
            for (i, output) in vectors.iter_mut().enumerate() {
                let x = 2 + i * $lanes;
                // SAFETY: the two-pixel borders and as_chunks bound every
                // neighbor load, including NE/NEE and WW. Typed loads read
                // exactly lanes samples, without extending byte/word rows.
                let v = load(unsafe { rows.current.as_ptr().add(x) });
                let w = load(unsafe { rows.current.as_ptr().add(x - 1) });
                let ww = load(unsafe { rows.current.as_ptr().add(x - 2) });
                let n = rows.north.map_or(w, |r| load(unsafe { r.as_ptr().add(x) }));
                let nw = rows
                    .north
                    .map_or(w, |r| load(unsafe { r.as_ptr().add(x - 1) }));
                let ne = rows
                    .north
                    .map_or(w, |r| load(unsafe { r.as_ptr().add(x + 1) }));
                let nee = rows
                    .north
                    .map_or(w, |r| load(unsafe { r.as_ptr().add(x + 2) }));
                let nn = rows
                    .north_north
                    .map_or(n, |r| load(unsafe { r.as_ptr().add(x) }));
                if CHECK
                    && PRED == 13
                    && !bits_clear(
                        [w, n, ww, nn, ne, nee]
                            .into_iter()
                            .fold(splat(0), |bits, v| or(bits, abs(v))),
                        splat(!((1 << 26) - 1)),
                    )
                {
                    // Average4's weighted numerator needs i64 for wide input.
                    super::pack_span(rows, PRED, x, output);
                } else {
                    let value = packed(v, prediction::<PRED>([w, n, nw, ne, ww, nn, nee]));
                    unsafe { store(output.as_mut_ptr().cast(), value) };
                }
            }
            super::pack_span(rows, PRED, end, tail);
            super::pack_span(rows, PRED, len - 2, &mut out[len - 2..]);
        }

        #[target_feature(enable = $feature)]
        fn pack<T: Copy, const CHECK: bool>(
            rows: super::Rows<'_, T>,
            pred: u32,
            out: &mut [u32],
            load: impl Fn(*const T) -> $vector,
        ) where
            i64: From<T>,
        {
            // Specialize once per row; each loop loads only the neighbors its
            // predictor actually reads after inlining and dead-code removal.
            match pred {
                0 => pack_impl::<T, CHECK, 0>(rows, out, load),
                1 => pack_impl::<T, CHECK, 1>(rows, out, load),
                2 => pack_impl::<T, CHECK, 2>(rows, out, load),
                3 => pack_impl::<T, CHECK, 3>(rows, out, load),
                4 => pack_impl::<T, CHECK, 4>(rows, out, load),
                5 => pack_impl::<T, CHECK, 5>(rows, out, load),
                7 => pack_impl::<T, CHECK, 7>(rows, out, load),
                8 => pack_impl::<T, CHECK, 8>(rows, out, load),
                9 => pack_impl::<T, CHECK, 9>(rows, out, load),
                10 => pack_impl::<T, CHECK, 10>(rows, out, load),
                11 => pack_impl::<T, CHECK, 11>(rows, out, load),
                12 => pack_impl::<T, CHECK, 12>(rows, out, load),
                13 => pack_impl::<T, CHECK, 13>(rows, out, load),
                _ => panic!("unsupported fixed predictor {pred}"),
            }
        }

        #[target_feature(enable = $feature)]
        fn costs<T: Copy, const CHECK: bool>(
            rows: super::Rows<'_, T>,
            hist: &mut [Vec<u64>],
            mut zero: Option<&mut Vec<u64>>,
            weighted_zero: bool,
            load: impl Fn(*const T) -> $vector,
        ) where
            i64: From<T>,
        {
            let len = rows.current.len();
            rows.validate(len);
            assert_eq!(hist.len(), super::super::SLOW_PREDICTORS.len());
            if len < $lanes + 4 {
                super::cost_scalar(rows, hist, zero, weighted_zero);
                return;
            }
            super::cost_span(rows, 0..2, hist, &mut zero, weighted_zero);
            let (vectors, _) = rows.current[2..len - 2].as_chunks::<$lanes>();
            for (i, _) in vectors.iter().enumerate() {
                let x = 2 + i * $lanes;
                // The complete interior block has two valid neighbor samples
                // on each side. Load them once for all five fixed predictors.
                let v = load(unsafe { rows.current.as_ptr().add(x) });
                let w = load(unsafe { rows.current.as_ptr().add(x - 1) });
                let ww = load(unsafe { rows.current.as_ptr().add(x - 2) });
                let n = rows.north.map_or(w, |r| load(unsafe { r.as_ptr().add(x) }));
                let nw = rows
                    .north
                    .map_or(w, |r| load(unsafe { r.as_ptr().add(x - 1) }));
                let ne = rows
                    .north
                    .map_or(w, |r| load(unsafe { r.as_ptr().add(x + 1) }));
                let nee = rows
                    .north
                    .map_or(w, |r| load(unsafe { r.as_ptr().add(x + 2) }));
                let nn = rows
                    .north_north
                    .map_or(n, |r| load(unsafe { r.as_ptr().add(x) }));
                if CHECK
                    && !bits_clear(
                        [w, n, ww, nn, ne, nee]
                            .into_iter()
                            .fold(splat(0), |bits, v| or(bits, abs(v))),
                        splat(!((1 << 26) - 1)),
                    )
                {
                    super::cost_span(rows, x..x + $lanes, hist, &mut zero, weighted_zero);
                    continue;
                }
                let neighbors = [w, n, nw, ne, ww, nn, nee];
                let predictions = [
                    prediction::<5>(neighbors),
                    prediction::<13>(neighbors),
                    prediction::<4>(neighbors),
                    prediction::<1>(neighbors),
                    prediction::<2>(neighbors),
                ];
                let mut values = [0i32; $lanes];
                for (hist, pred) in hist[1..].iter_mut().zip(predictions) {
                    unsafe { store(values.as_mut_ptr(), packed(v, pred)) };
                    for value in values {
                        super::super::cost::add_symbol(hist, value as u32);
                    }
                }
                if weighted_zero || zero.is_some() {
                    unsafe { store(values.as_mut_ptr(), packed(v, splat(0))) };
                    if weighted_zero {
                        for value in values {
                            super::super::cost::add_symbol(&mut hist[0], value as u32);
                        }
                    }
                    if let Some(zero) = zero.as_deref_mut() {
                        for value in values {
                            super::super::cost::add_symbol(zero, value as u32);
                        }
                    }
                }
            }
            let end = 2 + vectors.len() * $lanes;
            super::cost_span(rows, end..len, hist, &mut zero, weighted_zero);
        }

        #[target_feature(enable = $feature)]
        fn rct_impl<const FIRST: bool>(
            cur: &[i32],
            prev: &[i32],
            hist: &mut crate::lossless::rct::RctHistogram,
        ) -> u64 {
            assert_eq!(cur.len(), prev.len());
            if cur.is_empty() {
                return 0;
            }
            let mut bits = crate::lossless::rct::rct_gradient_span(cur, prev, FIRST, 0..1, hist);
            let (values, _) = cur[1..].as_chunks::<$lanes>();
            let (left, _) = cur[..cur.len() - 1].as_chunks::<$lanes>();
            let (north, _) = prev[1..].as_chunks::<$lanes>();
            let (north_west, _) = prev[..prev.len() - 1].as_chunks::<$lanes>();
            for (((v, w), n), nw) in values.iter().zip(left).zip(north).zip(north_west) {
                let v = unsafe { load(v.as_ptr()) };
                let w = unsafe { load(w.as_ptr()) };
                let n = if FIRST {
                    w
                } else {
                    unsafe { load(n.as_ptr()) }
                };
                let nw = if FIRST {
                    w
                } else {
                    unsafe { load(nw.as_ptr()) }
                };
                let prediction = if FIRST { w } else { gradient(w, n, nw) };
                let activity = sub(max(max(w, n), nw), min(min(w, n), nw));
                let (tokens, extras) = symbols(packed(v, prediction));
                let mut tokens_out = [0i32; $lanes];
                let mut extras_out = [0i32; $lanes];
                let mut activities = [0i32; $lanes];
                unsafe {
                    store(tokens_out.as_mut_ptr(), tokens);
                    store(extras_out.as_mut_ptr(), extras);
                    store(activities.as_mut_ptr(), activity);
                }
                for ((symbol, extra), activity) in
                    tokens_out.into_iter().zip(extras_out).zip(activities)
                {
                    let context = crate::lossless::rct::rct_context(activity as u32);
                    hist[context * 64 + (symbol as usize).min(63)] += 1;
                    bits += extra as u64;
                }
            }
            let end = 1 + values.len() * $lanes;
            bits + crate::lossless::rct::rct_gradient_span(cur, prev, FIRST, end..cur.len(), hist)
        }

        #[target_feature(enable = $feature)]
        fn rct(
            cur: &[i32],
            prev: &[i32],
            first: bool,
            hist: &mut crate::lossless::rct::RctHistogram,
        ) -> u64 {
            if first {
                rct_impl::<true>(cur, prev, hist)
            } else {
                rct_impl::<false>(cur, prev, hist)
            }
        }

        #[allow(unused_unsafe)]
        pub(super) fn kernels() -> super::Kernels {
            super::Kernels {
                i32: |r, p, o| unsafe { pack::<i32, true>(r, p, o, |v| load(v)) },
                u8: |r, p, o| unsafe { pack::<u8, false>(r, p, o, |v| load_u8(v)) },
                u16: |r, p, o| unsafe { pack::<u16, false>(r, p, o, |v| load_u16(v)) },
                costs_i32: |r, h, z, w| unsafe { costs::<i32, true>(r, h, z, w, |v| load(v)) },
                costs_u8: |r, h, z, w| unsafe { costs::<u8, false>(r, h, z, w, |v| load_u8(v)) },
                costs_u16: |r, h, z, w| unsafe { costs::<u16, false>(r, h, z, w, |v| load_u16(v)) },
                rct: |c, p, f, h| unsafe { rct(c, p, f, h) },
            }
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon {
    use crate::simd::neon::i32::*;
    backend!("neon", 4, Vector);
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
mod sse {
    use crate::simd::sse::i32::*;
    backend!("sse4.1", 4, Vector);
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx {
    use crate::simd::avx::i32::*;
    backend!("avx2", 8, Vector);
}

#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm {
    use crate::simd::wasm::i32::*;
    backend!("simd128", 4, Vector);
}

#[cfg(test)]
pub(in crate::lossless) fn available() -> Vec<(&'static str, Kernels)> {
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
    use crate::lossless::predictor::{clamped_gradient, predictor_neighbors};

    fn random(seed: &mut u64) -> i32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as i32
    }

    fn check_rows<T: Sample>(data: &[Vec<T>; 3], offset: usize)
    where
        i32: From<T>,
        i64: From<T>,
    {
        let rows = data.each_ref().map(|r| &r[offset..]);
        let width = rows[0].len();
        for y in 0..3 {
            let row = Rows {
                current: rows[y],
                north: y.checked_sub(1).map(|y| rows[y]),
                north_north: y.checked_sub(2).map(|y| rows[y]),
            };
            let get = |x: usize, y: usize| i32::from(rows[y][x]);
            for pred in (0..14).filter(|&p| p != 6) {
                let expected: Vec<u32> = (0..width)
                    .map(|x| {
                        let n = predictor_neighbors(&get, x, y, width);
                        pack_signed(((get(x, y) as i64) - predictor_value(pred, n, 0)) as i32)
                    })
                    .collect();
                for (name, kernels) in available() {
                    let mut actual = vec![u32::MAX; width];
                    T::pack(&kernels, row, pred, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "{name}, predictor {pred}, width {width}, row {y}, offset {offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn typed_rows_match_point_predictors_at_all_borders_and_tails() {
        let mut seed = 1234567;
        for width in (0..=35).chain([63, 64, 65, 127, 128, 129, 257]) {
            for offset in 0..4 {
                let bytes: [Vec<u8>; 3] = std::array::from_fn(|_| {
                    (0..width + offset)
                        .map(|_| random(&mut seed) as u8)
                        .collect()
                });
                let words: [Vec<u16>; 3] = std::array::from_fn(|_| {
                    (0..width + offset)
                        .map(|_| random(&mut seed) as u16)
                        .collect()
                });
                check_rows(&bytes, offset);
                check_rows(&words, offset);
                for wide in [false, true] {
                    let ints: [Vec<i32>; 3] = std::array::from_fn(|_| {
                        (0..width + offset)
                            .map(|_| {
                                let v = random(&mut seed);
                                if wide { v } else { v >> 16 }
                            })
                            .collect()
                    });
                    check_rows(&ints, offset);
                }
            }
        }
    }

    #[test]
    fn full_range_rounding_select_ties_and_average4_fallback_match_i64() {
        let limit = 1 << 26;
        let values = [
            i32::MIN,
            i32::MIN + 1,
            -limit - 1,
            -limit,
            -limit + 1,
            -65535,
            -17,
            -2,
            -1,
            0,
            1,
            2,
            17,
            65535,
            limit - 1,
            limit,
            limit + 1,
            i32::MAX - 1,
            i32::MAX,
        ];
        for phase in 0..values.len() {
            let data = std::array::from_fn(|row| {
                (0..257)
                    .map(|x| values[(x + phase + row * 7) % values.len()])
                    .collect()
            });
            check_rows::<i32>(&data, 1);
        }
    }

    #[test]
    fn fixed_costs_preserve_wide_average4_fallback() {
        for base in [i32::MIN + 512, -(1 << 26), 1 << 26, i32::MAX - 512] {
            let data: [Vec<i32>; 3] = std::array::from_fn(|y| {
                (0..65)
                    .map(|x| base + ((x * 31 + y * 17) % 127) as i32 - 63)
                    .collect()
            });
            let rows = Rows {
                current: &data[2],
                north: Some(&data[1]),
                north_north: Some(&data[0]),
            };
            let mut expected: [Vec<u64>; 6] = std::array::from_fn(|_| Vec::new());
            i32::costs(&scalar(), rows, &mut expected, None, false);
            for (name, kernels) in available() {
                let mut actual: [Vec<u64>; 6] = std::array::from_fn(|_| Vec::new());
                i32::costs(&kernels, rows, &mut actual, None, false);
                assert_eq!(actual, expected, "{name}/{base}");
            }
        }
    }

    #[test]
    fn rct_rows_preserve_histogram_collisions_and_wide_activity() {
        let mut seed = 987654321;
        for width in (0..=35).chain([63, 64, 65, 127, 128, 129, 257]) {
            for offset in 0..4 {
                for pattern in 0..4 {
                    let rows: [Vec<i32>; 2] = std::array::from_fn(|row| {
                        (0..width + offset)
                            .map(|x| match pattern {
                                0 => 37,
                                1 => random(&mut seed) >> 16,
                                2 => random(&mut seed),
                                _ => [i32::MIN, i32::MAX, -1, 0, 1][(x + row) % 5],
                            })
                            .collect()
                    });
                    let [prev, cur] = rows.each_ref().map(|r| &r[offset..]);
                    for first in [false, true] {
                        let mut expected: RctHistogram = std::array::from_fn(|i| (i % 17) as u64);
                        let initial = expected;
                        let mut bits = 0;
                        for x in 0..width {
                            let left = if x > 0 {
                                cur[x - 1]
                            } else if first {
                                0
                            } else {
                                prev[x]
                            };
                            let top = if first { left } else { prev[x] };
                            let nw = if x > 0 && !first { prev[x - 1] } else { left };
                            let activity = (i64::from(left.max(top).max(nw))
                                - i64::from(left.min(top).min(nw)))
                                as u32;
                            let context = crate::lossless::rct::rct_context(activity);
                            let prediction =
                                clamped_gradient(i64::from(left), i64::from(top), i64::from(nw));
                            let (symbol, extra, _) = crate::entropy::uint_encode(pack_signed(
                                (i64::from(cur[x]) - prediction) as i32,
                            ));
                            expected[context * 64 + (symbol as usize).min(63)] += 1;
                            bits += u64::from(extra);
                        }
                        for (name, kernels) in available() {
                            let mut actual = initial;
                            let actual_bits = (kernels.rct)(cur, prev, first, &mut actual);
                            assert_eq!(actual_bits, bits, "{name}, width {width}, first {first}");
                            assert_eq!(actual, expected, "{name}, width {width}, first {first}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rct_symbolization_matches_all_hybrid_boundaries() {
        let mut packed: Vec<u32> = (0..=17).collect();
        for log in 4..32 {
            for mantissa in 0..4 {
                let value = (1u64 << log) + (mantissa << (log - 2));
                for delta in [-1i64, 0, 1] {
                    packed.push((value as i64 + delta).clamp(0, i64::from(u32::MAX)) as u32);
                }
            }
        }
        packed.extend([u32::MAX - 1, u32::MAX]);
        for offset in 0..8 {
            let mut current = vec![0i32; offset];
            let mut previous = 0i32;
            for &value in &packed {
                let residual = ((value >> 1) as i32) ^ -((value & 1) as i32);
                previous = previous.wrapping_add(residual);
                current.push(previous);
            }
            let north = vec![0; current.len()];
            let mut expected = [0; 18 * 64];
            let bits = (scalar().rct)(&current, &north, true, &mut expected);
            for (name, kernels) in available() {
                let mut actual = [0; 18 * 64];
                assert_eq!(
                    (kernels.rct)(&current, &north, true, &mut actual),
                    bits,
                    "{name}/{offset}"
                );
                assert_eq!(actual, expected, "{name}/{offset}");
            }
        }
    }
}
