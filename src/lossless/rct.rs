/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

//! Reversible color transforms, their modular headers, and transform selection.

use super::EntropyOfHistFn;
use super::predictor::{GradientScratch, clamped_gradient};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{pack_signed, uint_encode};
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;

// ---------------------------------------------------------------------------
// Forward reversible YCoCg (RCT type 6, matches libjxl's InvRCTRow<6>).
//
// Encoder:
//   co  = r - b
//   tmp = b + (co >> 1)
//   cg  = g - tmp
//   y   = tmp + (cg >> 1)
//
// Decoder undoes this with the exact same shift sequence:
//   tmp = y - (cg >> 1);  g = cg + tmp;
//   y'  = tmp - (co >> 1); r = y' + co;  b = y'
//
// Reversible because every operation is invertible without rounding.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn forward_ycocg(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    (y, co, cg)
}

/// Exact integer inverse of `forward_ycocg` (undoes the steps in reverse).
#[inline]
pub(super) fn inverse_ycocg(y: i32, co: i32, cg: i32) -> (i32, i32, i32) {
    let tmp = y - (cg >> 1);
    let g = cg + tmp;
    let b = tmp - (co >> 1);
    let r = b + co;
    (r, g, b)
}

/// rct_type: U32(Val(6), Bits(2), BitsOffset(4,2), BitsOffset(6,10)), 0..41.
fn write_rct_type(rct_type: u32, w: &mut BitWriter) {
    debug_assert!(rct_type < 42);
    if rct_type == 6 {
        w.write(2, 0b00);
    } else if rct_type < 4 {
        w.write(2, 0b01);
        w.write(2, rct_type as u64);
    } else if rct_type < 18 {
        w.write(2, 0b10);
        w.write(4, (rct_type - 2) as u64);
    } else {
        w.write(2, 0b11);
        w.write(6, (rct_type - 10) as u64);
    }
}

/// One RCT transform entry operating on the first three channels.
pub(super) fn write_rct_transform(rct_type: u32, w: &mut BitWriter) {
    w.write(2, 0b00); // id = RCT (Bits(2))
    w.write(2, 0b00); // begin_channel selector 0
    w.write(3, 0); // begin_channel value (Bits(3)) = 0
    write_rct_type(rct_type, w);
}

// ---------------------------------------------------------------------------
// RCT selection: YCoCg (type 6) is a strong default, but content with weak or
// atypical channel correlation prefers one of the other reversible transforms
// (subtract-green and half-average families). Candidates and cost model follow
// libjxl: fixed 18-context gradient-residual token entropy per channel.
// ---------------------------------------------------------------------------

/// Candidate rct_types in libjxl's try order (first 7 = cjxl -e7's set):
/// none, YCoCg, G-half-avg, subtract-green, and half-average variants.
static RCT_CANDIDATES: [u32; 7] = [0, 6, 5, 10, 26, 40, 12];
static RCT_CONTEXT_CUTOFFS: [u32; 17] = [
    0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];
const RCT_NUM_CONTEXTS: usize = 18;
const RCT_ALPHABET: usize = 64;
const RCT_CONTEXT_MAX_DIFF: u32 = 500;

static RCT_CONTEXT_LUT: [u8; RCT_CONTEXT_MAX_DIFF as usize + 1] = {
    assert!(RCT_CONTEXT_CUTOFFS.len() < RCT_NUM_CONTEXTS);
    let mut lut = [0u8; RCT_CONTEXT_MAX_DIFF as usize + 1];
    let mut diff = 0usize;
    while diff < lut.len() {
        let mut cutoff = 0usize;
        let mut context = 0u8;
        while cutoff < RCT_CONTEXT_CUTOFFS.len() {
            if (diff as u32) < RCT_CONTEXT_CUTOFFS[cutoff] {
                context += 1;
            }
            cutoff += 1;
        }
        lut[diff] = context;
        diff += 1;
    }
    lut
};

fn context_lut() -> [u8; RCT_CONTEXT_MAX_DIFF as usize + 1] {
    RCT_CONTEXT_LUT
}

#[inline(always)]
pub(super) fn rct_context(max_diff: u32) -> usize {
    context_lut()[max_diff.min(RCT_CONTEXT_MAX_DIFF) as usize] as usize
}

/// Channel order fed to the elementary transform for permutation `perm`
/// (0=RGB, 1=GBR, 2=BRG, 3=RBG, 4=GRB, 5=BGR).
#[inline]
fn rct_perm(perm: usize) -> [usize; 3] {
    [
        perm % 3,
        (perm + 1 + perm / 3) % 3,
        (perm + 2 - perm / 3) % 3,
    ]
}

/// Forward elementary RCT `t` (rct_type % 7) on already-permuted values.
#[inline]
fn rct_forward_pixel(t: u32, first: i32, second: i32, third: i32) -> (i32, i32, i32) {
    if t == 6 {
        let o1 = first - third;
        let tmp = third + (o1 >> 1);
        let o2 = second - tmp;
        (tmp + (o2 >> 1), o1, o2)
    } else {
        let s = match t >> 1 {
            1 => second - first,
            2 => second - ((first + third) >> 1),
            _ => second,
        };
        let th = if t & 1 == 1 { third - first } else { third };
        (first, s, th)
    }
}

#[inline]
pub(crate) fn rct_from_ycocg_pixel(y: i32, co: i32, cg: i32, rct: u32) -> (i32, i32, i32) {
    if rct == 6 {
        return (y, co, cg);
    }
    let rgb = inverse_ycocg(y, co, cg);
    let rgb = [rgb.0, rgb.1, rgb.2];
    let perm = rct_perm((rct / 7) as usize);
    rct_forward_pixel(rct % 7, rgb[perm[0]], rgb[perm[1]], rgb[perm[2]])
}

#[inline]
pub(crate) fn rct_gradient_row(
    crow: &[i32],
    prow: &[i32],
    first_row: bool,
    hist: &mut [u64; RCT_NUM_CONTEXTS * RCT_ALPHABET],
) -> u64 {
    assert_eq!(crow.len(), prow.len());
    let Some((&first, tail)) = crow.split_first() else {
        return 0;
    };
    let mut extra_bits = 0u64;
    let mut record = |value: i32, prediction: i64, ctx: usize| {
        let (tok, nb, _) = uint_encode(pack_signed((value as i64 - prediction) as i32));
        hist[ctx * RCT_ALPHABET + (tok as usize).min(RCT_ALPHABET - 1)] += 1;
        extra_bits += nb as u64;
    };
    let edge_ctx = rct_context(0);
    if first_row {
        // With no north row all three neighbors equal the west sample.
        record(first, 0, edge_ctx);
        for (&value, &left) in tail.iter().zip(crow) {
            record(value, left as i64, edge_ctx);
        }
    } else {
        let (&top, north_tail) = prow.split_first().expect("equal nonempty rows");
        // At x = 0 all three neighbors equal the north sample.
        record(first, top as i64, edge_ctx);
        for (((&value, &left), &top), &topleft) in tail.iter().zip(crow).zip(north_tail).zip(prow) {
            let mx = left.max(top).max(topleft);
            let mn = left.min(top).min(topleft);
            let ctx = rct_context((mx - mn) as u32);
            let prediction = clamped_gradient(left as i64, top as i64, topleft as i64);
            record(value, prediction, ctx);
        }
    }
    extra_bits
}

/// Port of libjxl's EstimateCost: gradient-predictor residual token entropy
/// over 18 local-activity contexts per channel, plus raw bits. Candidate pixels
/// are derived directly from the input YCoCg planes, avoiding a temporary RGB
/// image shared by the seven scoring passes.
fn estimate_rct_cost(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    rct: u32,
    row_step: usize,
    row_scratch: &mut GradientScratch,
    entropy_of_hist: EntropyOfHistFn,
) -> f32 {
    let y_plane = linear.plane_data(0);
    let co_plane = linear.plane_data(1);
    let cg_plane = linear.plane_data(2);
    let mut hist = [[0u64; RCT_NUM_CONTEXTS * RCT_ALPHABET]; 3];
    let mut extra_bits: u64 = 0;
    let transform = crate::lossless::selected_rct_row_fn();
    row_scratch.prev.resize(3 * xsize, 0);
    row_scratch.cur.resize(3 * xsize, 0);
    row_scratch.prev.fill(0);
    let GradientScratch { prev, cur, .. } = row_scratch;
    let (mut prev, mut cur) = (prev, cur);
    for y in 0..ysize {
        // A scored row needs its predecessor transformed as well.
        let scored = y.is_multiple_of(row_step);
        if !scored && !(y + 1).is_multiple_of(row_step) {
            continue;
        }
        let yy = &y_plane[y * xsize..][..xsize];
        let co = &co_plane[y * xsize..][..xsize];
        let cg = &cg_plane[y * xsize..][..xsize];
        let (out0, rest) = cur.split_at_mut(xsize);
        let (out1, out2) = rest.split_at_mut(xsize);
        transform([yy, co, cg], rct, [out0, out1, out2]);
        if scored {
            for (ch, chist) in hist.iter_mut().enumerate() {
                let crow = &cur[ch * xsize..][..xsize];
                let prow = &prev[ch * xsize..][..xsize];
                extra_bits += rct_gradient_row(crow, prow, y == 0, chist);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let mut cost = extra_bits as f32;
    for h in hist.as_flattened().as_chunks::<RCT_ALPHABET>().0 {
        let total: u64 = h.iter().sum();
        cost += entropy_of_hist(h, total);
    }
    cost
}

/// Estimate every candidate RCT and return the winner's planes when it is not
/// YCoCg (in which case the caller keeps the input planes unchanged).
/// Every candidate RCT with its estimated cost over every `row_step`-th row,
/// cheapest first.
pub(super) fn rank_rcts(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    row_step: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<(u32, f32)> {
    let costs = pool.steal_map(scratch, RCT_CANDIDATES.len(), |i, scratch| {
        estimate_rct_cost(
            linear,
            xsize,
            ysize,
            RCT_CANDIDATES[i],
            row_step,
            &mut scratch.gradient,
            scratch.entropy_of_hist,
        )
    });
    let mut ranked: Vec<(u32, f32)> = RCT_CANDIDATES.iter().copied().zip(costs).collect();
    crate::util::sort_small_by(&mut ranked, |a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    ranked
}

/// The YCoCg planes re-expressed under `rct`, in parallel row bands.
pub(super) fn rct_planes(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    rct: u32,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Image3Si {
    let mut out = Image3Si::new(xsize, ysize);
    let mut bands = out.row_bands_mut(pool.num_threads());
    let mut starts = Vec::with_capacity(bands.len());
    let mut start = 0;
    for band in &bands {
        starts.push(start);
        start += band[0].len();
    }
    let transform = crate::lossless::selected_rct_row_fn();
    pool.steal_for_each_mut(scratch, &mut bands, |i, [o0, o1, o2], _scratch| {
        let range = starts[i]..starts[i] + o0.len();
        transform(
            [0, 1, 2].map(|c| &linear.plane_data(c)[range.clone()]),
            rct,
            [o0, o1, o2],
        );
    });
    out
}

/// Estimate every candidate RCT and return the winner's planes when it is not
/// YCoCg (in which case the caller keeps the input planes unchanged).
#[allow(dead_code)]
pub(super) fn select_rct(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<(u32, Image3Si)> {
    let rct = rank_rcts(linear, xsize, ysize, 1, pool, scratch)[0].0;
    if rct == 6 {
        return None;
    }
    Some((rct, rct_planes(linear, xsize, ysize, rct, pool, scratch)))
}

pub(crate) type RctRowFn = fn([&[i32]; 3], u32, [&mut [i32]; 3]);

#[allow(unreachable_code)]
pub(crate) fn selected_rct_row_fn() -> RctRowFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |a, b, c| unsafe { crate::avx::rct_row_avx2(a, b, c) };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        return |a, b, c| unsafe { crate::sse::rct_row_sse41(a, b, c) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |a, b, c| unsafe { crate::neon::rct_row_neon(a, b, c) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::rct_row_wasm;
    rct_row_scalar
}

#[allow(dead_code)]
pub(crate) fn rct_row_scalar(src: [&[i32]; 3], kind: u32, out: [&mut [i32]; 3]) {
    let [a, b, c] = out;
    for i in 0..a.len() {
        (a[i], b[i], c[i]) =
            crate::lossless::rct_from_ycocg_pixel(src[0][i], src[1][i], src[2][i], kind);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_gradient_rows_match_pixelwise_histograms() {
        for width in (0..=33).chain([63, 64, 65, 127, 128, 129, 4800]) {
            for pattern in 0..4 {
                let sample = |x: usize, row: usize| match pattern {
                    0 => 37 - row as i32 * 56,
                    1 => x as i32 * 7 - row as i32 * 11,
                    2 => {
                        if (x + row).is_multiple_of(2) {
                            -65_535
                        } else {
                            65_535
                        }
                    }
                    _ => ((x * 977 + row * 619) ^ (x * x * 37)) as i32 & 0xffff,
                };
                let crow: Vec<i32> = (0..width).map(|x| sample(x, 1)).collect();
                let prow: Vec<i32> = (0..width).map(|x| sample(x, 0)).collect();
                for first_row in [false, true] {
                    let mut expected = std::array::from_fn(|i| (i % 17) as u64);
                    let mut actual = expected;
                    let mut expected_bits = 0u64;
                    for x in 0..width {
                        let left = if x > 0 {
                            crow[x - 1]
                        } else if !first_row {
                            prow[x]
                        } else {
                            0
                        };
                        let top = if !first_row { prow[x] } else { left };
                        let topleft = if x > 0 && !first_row {
                            prow[x - 1]
                        } else {
                            left
                        };
                        let diff = (left.max(top).max(topleft) - left.min(top).min(topleft)) as u32;
                        let ctx = rct_context(diff);
                        let prediction = clamped_gradient(left as i64, top as i64, topleft as i64);
                        let (token, nbits, _) =
                            uint_encode(pack_signed((crow[x] as i64 - prediction) as i32));
                        expected[ctx * RCT_ALPHABET + (token as usize).min(RCT_ALPHABET - 1)] += 1;
                        expected_bits += nbits as u64;
                    }
                    let actual_bits = rct_gradient_row(&crow, &prow, first_row, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "width={width}, pattern={pattern}, first_row={first_row}"
                    );
                    assert_eq!(actual_bits, expected_bits);
                }
            }
        }
    }

    #[test]
    fn context_lookup_matches_linear_cutoff_classification() {
        let reference = |max_diff: u32| {
            RCT_CONTEXT_CUTOFFS
                .iter()
                .filter(|&&cutoff| max_diff < cutoff)
                .count()
        };
        for max_diff in 0..=u16::MAX as u32 {
            assert_eq!(rct_context(max_diff), reference(max_diff), "{max_diff}");
        }
        assert_eq!(rct_context(u32::MAX), reference(u32::MAX));
    }

    #[test]
    fn direct_ycocg_candidate_conversion_matches_rgb_round_trip() {
        for &(y, co, cg) in &[(0, 0, 0), (127, -91, 53), (-400, 723, -255)] {
            let rgb = inverse_ycocg(y, co, cg);
            let rgb = [rgb.0, rgb.1, rgb.2];
            for &rct in &RCT_CANDIDATES {
                let expected = if rct == 6 {
                    (y, co, cg)
                } else {
                    let perm = rct_perm((rct / 7) as usize);
                    rct_forward_pixel(rct % 7, rgb[perm[0]], rgb[perm[1]], rgb[perm[2]])
                };
                assert_eq!(rct_from_ycocg_pixel(y, co, cg, rct), expected);
            }
        }
    }
}

#[cfg(test)]
mod search_row_tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Rows {
        rct: RctRowFn,
    }

    fn backends() -> Vec<Rows> {
        #[allow(unused_mut)]
        let mut rows = vec![Rows {
            rct: rct_row_scalar,
        }];
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            rows.push(Rows {
                rct: |a, b, c| unsafe { crate::avx::rct_row_avx2(a, b, c) },
            });
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            rows.push(Rows {
                rct: |a, b, c| unsafe { crate::sse::rct_row_sse41(a, b, c) },
            });
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            rows.push(Rows {
                rct: |a, b, c| unsafe { crate::neon::rct_row_neon(a, b, c) },
            });
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            rows.push(Rows {
                rct: crate::wasm::rct_row_wasm,
            });
        }
        rows.push(Rows {
            rct: selected_rct_row_fn(),
        });
        rows
    }

    #[test]
    fn every_rct_permutation_and_tail_matches_scalar() {
        for len in 0..35 {
            let src: [Vec<i32>; 3] = std::array::from_fn(|c| {
                (0..len + 1)
                    .map(|i| ((i * 377 + c * 997) % 8191) as i32 - 4095)
                    .collect()
            });
            for kind in 0..42 {
                let mut expected: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0; len]);
                let [a, b, c] = &mut expected;
                rct_row_scalar(src.each_ref().map(|p| &p[1..]), kind, [a, b, c]);
                for backend in backends() {
                    let mut actual: [Vec<i32>; 3] = std::array::from_fn(|_| vec![0; len + 2]);
                    let [a, b, c] = &mut actual;
                    (backend.rct)(
                        src.each_ref().map(|p| &p[1..]),
                        kind,
                        [&mut a[1..1 + len], &mut b[1..1 + len], &mut c[1..1 + len]],
                    );
                    for (a, e) in actual.iter().zip(&expected) {
                        assert_eq!(&a[1..1 + len], e, "kind={kind}, len={len}");
                        assert_eq!(a[0], 0);
                        assert_eq!(a[len + 1], 0);
                    }
                }
            }
        }
    }
}
