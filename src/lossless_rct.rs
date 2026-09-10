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

use super::entropy_of_hist;
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
fn rct_from_ycocg_pixel(y: i32, co: i32, cg: i32, rct: u32) -> (i32, i32, i32) {
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
    row_scratch: &mut GradientScratch,
) -> f32 {
    let y_plane = linear.plane_data(0);
    let co_plane = linear.plane_data(1);
    let cg_plane = linear.plane_data(2);
    let mut hist = [[0u64; RCT_NUM_CONTEXTS * RCT_ALPHABET]; 3];
    let mut extra_bits: u64 = 0;
    row_scratch.prev.resize(3 * xsize, 0);
    row_scratch.cur.resize(3 * xsize, 0);
    row_scratch.prev.fill(0);
    let GradientScratch { prev, cur, .. } = row_scratch;
    let (mut prev, mut cur) = (prev, cur);
    for y in 0..ysize {
        let yy = &y_plane[y * xsize..][..xsize];
        let co = &co_plane[y * xsize..][..xsize];
        let cg = &cg_plane[y * xsize..][..xsize];
        let (out0, rest) = cur.split_at_mut(xsize);
        let (out1, out2) = rest.split_at_mut(xsize);
        let inputs = yy.iter().zip(co).zip(cg);
        let outputs = out0.iter_mut().zip(out1).zip(out2);
        for (((&y_value, &co_value), &cg_value), ((a, b), c)) in inputs.zip(outputs) {
            (*a, *b, *c) = rct_from_ycocg_pixel(y_value, co_value, cg_value, rct);
        }
        for (ch, chist) in hist.iter_mut().enumerate() {
            let crow = &cur[ch * xsize..][..xsize];
            let prow = &prev[ch * xsize..][..xsize];
            extra_bits += rct_gradient_row(crow, prow, y == 0, chist);
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
/// Every candidate RCT with its estimated cost, cheapest first.
pub(super) fn rank_rcts(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<(u32, f32)> {
    let costs = pool.steal_map(scratch, RCT_CANDIDATES.len(), |i, scratch| {
        estimate_rct_cost(
            linear,
            xsize,
            ysize,
            RCT_CANDIDATES[i],
            &mut scratch.gradient,
        )
    });
    let mut ranked: Vec<(u32, f32)> = RCT_CANDIDATES.iter().copied().zip(costs).collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    ranked
}

/// The YCoCg planes re-expressed under `rct`.
pub(super) fn rct_planes(linear: &Image3Si, xsize: usize, ysize: usize, rct: u32) -> Image3Si {
    let mut out = Image3Si::new(xsize, ysize);
    for y in 0..ysize {
        let [o0, o1, o2] = out.all_plane_rows_mut(y);
        let yy = &linear.plane_data(0)[y * xsize..][..xsize];
        let co = &linear.plane_data(1)[y * xsize..][..xsize];
        let cg = &linear.plane_data(2)[y * xsize..][..xsize];
        let inputs = yy.iter().zip(co).zip(cg);
        let outputs = o0.iter_mut().zip(o1).zip(o2);
        for (((&y_value, &co_value), &cg_value), ((a, b), c)) in inputs.zip(outputs) {
            (*a, *b, *c) = rct_from_ycocg_pixel(y_value, co_value, cg_value, rct);
        }
    }
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
    let rct = rank_rcts(linear, xsize, ysize, pool, scratch)[0].0;
    if rct == 6 {
        return None;
    }
    Some((rct, rct_planes(linear, xsize, ysize, rct)))
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
                        if (x + row) % 2 == 0 {
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
