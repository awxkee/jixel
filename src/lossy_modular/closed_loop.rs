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

//! Stage 2c: closed-loop residuals.
//!
//! The decoder rebuilds every squeeze step from the averages it already
//! decoded, through their tendency, not from the exact averages. Quantizing a
//! residual against the exact averages lets those errors stack up level by
//! level, worst at edges. Here each residual is chosen, coarse to fine,
//! against the decoded averages and the decoded left/top neighbor, exactly
//! as `squeeze::inv_h_squeeze` / `inv_v_squeeze` rebuild them.

use crate::adaptive_quant::dirty_log2f;
use crate::fast_divide::DividerI32;
use crate::squeeze::{Channel, SqueezeStep};
use crate::thread_pool::ThreadPool;

/// How a residual's magnitude rounds once divided by its step.
#[derive(Clone, Copy)]
pub(crate) enum Rounding {
    /// To nearest after taking this much of a step off the magnitude.
    Deadzone(f32),
    /// To nearest, or one step toward zero where that saves more estimated
    /// bits, times this lambda, than it adds in squared error (in steps²).
    /// The estimate: about 0.3 bits for a zero, 2.5 + 2·log2|m| otherwise.
    RateAware(f32),
}

/// `r / q` rounded as `rounding` says.
#[inline]
fn quantize_residual(r: i64, q: i64, rounding: Rounding) -> i64 {
    if q <= 1 {
        return r;
    }
    let f = r.unsigned_abs() as f32 / q as f32;
    let m = match rounding {
        // Casting a nonnegative float truncates toward zero, which is floor.
        Rounding::Deadzone(deadzone) => (f + 0.5 - deadzone).max(0.0) as i64,
        Rounding::RateAware(lambda) => {
            let nearest = (f + 0.5) as i64;
            let bits = |m: i64| {
                if m == 0 {
                    0.3
                } else {
                    2.5 + 2.0 * dirty_log2f(m as f32)
                }
            };
            let cost = |m: i64| 0.5 * (f - m as f32).powi(2) + lambda * bits(m);
            if nearest > 0 && cost(nearest - 1) < cost(nearest) {
                nearest - 1
            } else {
                nearest
            }
        }
    };
    if r < 0 { -m } else { m }
}

/// [`quantize_residual`] for one step `q`, without its float division when
/// there is no deadzone: `floor(|r| / q + 1/2)` is then the integer rounded
/// quotient, exact through a precomputed divider for every `|r| < 2^23`
/// (beyond that the f32 path is kept, bit for bit).
#[derive(Clone, Copy)]
struct ResidualRounding {
    q: i64,
    half: i32,
    divider: DividerI32,
    deadzone: f32,
}

impl ResidualRounding {
    const EXACT_MAGNITUDE: u64 = 1 << 23;

    fn new(q: i64, deadzone: f32) -> Self {
        let step = q.clamp(1, i64::from(i32::MAX)) as i32;
        Self {
            q,
            half: step / 2,
            divider: DividerI32::new(step),
            deadzone,
        }
    }

    #[inline(always)]
    fn apply(self, r: i64) -> i64 {
        let magnitude = r.unsigned_abs();
        if self.q <= 1 || self.deadzone != 0.0 || magnitude >= Self::EXACT_MAGNITUDE {
            return quantize_residual(r, self.q, Rounding::Deadzone(self.deadzone));
        }
        let m = i64::from((magnitude as i32 + self.half) / self.divider);
        if r < 0 { -m } else { m }
    }
}

/// One horizontal squeeze step quantized in closed loop: the residuals of
/// `x` (`w`×`h`, the step's exact input) under the tendency the decoder
/// derives from the decoded averages `avg`, and the decoded `x`. Mirrors
/// `squeeze::inv_h_squeeze`, so the decoder's left neighbor is the decoded
/// one.
pub(crate) fn closed_h_squeeze(
    x: &[i32],
    w: usize,
    h: usize,
    avg: &[i32],
    q: i32,
    rounding: Rounding,
) -> (Vec<i32>, Vec<i32>) {
    use crate::squeeze::smooth_tendency;
    let ow = w.div_ceil(2);
    let rw = w - ow;
    let q = i64::from(q);
    let mut res = vec![0i32; rw * h];
    let mut out = vec![0i32; w * h];
    for y in 0..h {
        let avg_row = &avg[y * ow..(y + 1) * ow];
        let x_row = &x[y * w..(y + 1) * w];
        let out_row = &mut out[y * w..(y + 1) * w];
        let res_row = &mut res[y * rw..(y + 1) * rw];
        let (x_pairs, _) = x_row.as_chunks::<2>();
        let (out_pairs, out_tail) = out_row.as_chunks_mut::<2>();
        let next_avg = avg_row.iter().skip(1).chain(avg_row.last());
        let mut left = None;
        for ((([x0, x1], [out0, out1]), residual), (&av, &next)) in x_pairs
            .iter()
            .zip(out_pairs)
            .zip(res_row)
            .zip(avg_row.iter().zip(next_avg))
        {
            let av = i64::from(av);
            let next = i64::from(next);
            let tendency = smooth_tendency(left.unwrap_or(av), av, next);
            let diff = i64::from(*x0) - i64::from(*x1);
            let r = quantize_residual(diff - tendency, q, rounding);
            *residual = r as i32;
            let diff = r * q + tendency;
            let first = av + diff / 2;
            *out0 = first as i32;
            *out1 = (first - diff) as i32;
            left = Some(first - diff);
        }
        out_tail.copy_from_slice(&avg_row[rw..]);
    }
    (res, out)
}

/// The vertical counterpart of [`closed_h_squeeze`] (mirrors
/// `squeeze::inv_v_squeeze`).
pub(crate) fn closed_v_squeeze(
    x: &[i32],
    w: usize,
    h: usize,
    avg: &[i32],
    q: i32,
    rounding: Rounding,
) -> (Vec<i32>, Vec<i32>) {
    use crate::squeeze::smooth_tendency;
    let oh = h.div_ceil(2);
    let rh = h - oh;
    let q = i64::from(q);
    let mut res = vec![0i32; w * rh];
    let mut out = vec![0i32; w * h];
    for y in 0..rh {
        let avg_row = &avg[y * w..(y + 1) * w];
        let next_row = if y + 1 < oh {
            &avg[(y + 1) * w..(y + 2) * w]
        } else {
            avg_row
        };
        let x0 = &x[2 * y * w..(2 * y + 1) * w];
        let x1 = &x[(2 * y + 1) * w..(2 * y + 2) * w];
        let res_row = &mut res[y * w..(y + 1) * w];
        let (decoded, remaining) = out.split_at_mut(2 * y * w);
        let top_row = if y > 0 {
            &decoded[decoded.len() - w..]
        } else {
            avg_row
        };
        let (out0, remaining) = remaining.split_at_mut(w);
        let out1 = &mut remaining[..w];
        let neighbors = avg_row.iter().zip(next_row).zip(top_row);
        let inputs = x0.iter().zip(x1);
        let outputs = out0.iter_mut().zip(out1).zip(res_row);
        for ((neighbors, (&x0, &x1)), outputs) in neighbors.zip(inputs).zip(outputs) {
            let ((&av, &next), &top) = neighbors;
            let ((out0, out1), residual) = outputs;
            let av = i64::from(av);
            let next = i64::from(next);
            let top = i64::from(top);
            let tendency = smooth_tendency(top, av, next);
            let diff = i64::from(x0) - i64::from(x1);
            let r = quantize_residual(diff - tendency, q, rounding);
            *residual = r as i32;
            let diff = r * q + tendency;
            let first = av + diff / 2;
            *out0 = first as i32;
            *out1 = (first - diff) as i32;
        }
    }
    if oh > rh {
        out[(h - 1) * w..h * w].copy_from_slice(&avg[(oh - 1) * w..oh * w]);
    }
    (res, out)
}

/// Rows per parallel band of a horizontal step, and columns per stripe of a
/// vertical one.
const ROWS_PER_BAND: usize = 32;
const COLUMNS_PER_STRIPE: usize = 64;

/// One horizontal step in closed loop, fed by the exact pyramid: rebuilds the
/// exact input from `avg`/`res` as `squeeze::inv_h_squeeze` does and, in the
/// same pass, quantizes it against the `decoded` averages as
/// [`closed_h_squeeze`] does. Returns (exact input, residuals, decoded input).
/// Rows are independent, so bands run in parallel.
fn closed_h_from_exact(
    avg: &Channel,
    res: &Channel,
    decoded: &[i32],
    q: i32,
    deadzone: f32,
    pool: &ThreadPool,
) -> (Channel, Vec<i32>, Vec<i32>) {
    use crate::squeeze::smooth_tendency;
    let (ow, rw, h) = (avg.w, res.w, avg.h);
    let w = ow + rw;
    let q = i64::from(q);
    let rounding = ResidualRounding::new(q, deadzone);
    let mut input = vec![0i32; w * h];
    let mut residuals = vec![0i32; rw * h];
    let mut out = vec![0i32; w * h];
    if w > 0 && rw > 0 {
        let band = ROWS_PER_BAND;
        let mut bands: Vec<(&mut [i32], &mut [i32], &mut [i32])> = input
            .chunks_mut(band * w)
            .zip(residuals.chunks_mut(band * rw))
            .zip(out.chunks_mut(band * w))
            .map(|((x, r), o)| (x, r, o))
            .collect();
        pool.for_each_mut(&mut bands, |b, (x_band, r_band, o_band)| {
            for (row, ((x_row, r_row), o_row)) in x_band
                .chunks_exact_mut(w)
                .zip(r_band.chunks_exact_mut(rw))
                .zip(o_band.chunks_exact_mut(w))
                .enumerate()
            {
                let y = b * band + row;
                let exact_avg = &avg.data[y * ow..(y + 1) * ow];
                let exact_res = &res.data[y * rw..(y + 1) * rw];
                let dec_avg = &decoded[y * ow..(y + 1) * ow];
                let next_exact = exact_avg.iter().skip(1).chain(exact_avg.last());
                let next_dec = dec_avg.iter().skip(1).chain(dec_avg.last());
                let exact = exact_avg.iter().zip(next_exact).zip(exact_res);
                let decoded = dec_avg.iter().zip(next_dec);
                let (x_pairs, x_tail) = x_row.as_chunks_mut::<2>();
                let (o_pairs, o_tail) = o_row.as_chunks_mut::<2>();
                let outputs = x_pairs.iter_mut().zip(r_row).zip(o_pairs);
                let (mut left_exact, mut left) = (None, None);
                for ((exact, (&av_dec, &next_dec)), outputs) in exact.zip(decoded).zip(outputs) {
                    let ((&av, &next), &exact_res) = exact;
                    let (([x0, x1], residual), [out0, out1]) = outputs;
                    let av = i64::from(av);
                    let tendency = smooth_tendency(left_exact.unwrap_or(av), av, i64::from(next));
                    let diff = i64::from(exact_res) + tendency;
                    let first = av + diff / 2;
                    *x0 = first as i32;
                    *x1 = (first - diff) as i32;
                    left_exact = Some(i64::from(*x1));

                    let av = i64::from(av_dec);
                    let tendency = smooth_tendency(left.unwrap_or(av), av, i64::from(next_dec));
                    let diff = i64::from(*x0) - i64::from(*x1);
                    let r = rounding.apply(diff - tendency);
                    *residual = r as i32;
                    let diff = r * q + tendency;
                    let first = av + diff / 2;
                    *out0 = first as i32;
                    *out1 = (first - diff) as i32;
                    left = Some(first - diff);
                }
                x_tail.copy_from_slice(&exact_avg[rw..]);
                o_tail.copy_from_slice(&dec_avg[rw..]);
            }
        });
    } else {
        // No residual columns: the input is its average.
        input.copy_from_slice(&avg.data);
        out.copy_from_slice(decoded);
    }
    let input = Channel {
        data: input,
        w,
        h,
        hshift: avg.hshift - 1,
        vshift: avg.vshift,
    };
    (input, residuals, out)
}

/// The vertical counterpart of [`closed_h_from_exact`] (mirrors
/// `squeeze::inv_v_squeeze` and [`closed_v_squeeze`]). Columns are
/// independent, so stripes run in parallel.
fn closed_v_from_exact(
    avg: &Channel,
    res: &Channel,
    decoded: &[i32],
    q: i32,
    deadzone: f32,
    pool: &ThreadPool,
) -> (Channel, Vec<i32>, Vec<i32>) {
    use crate::squeeze::smooth_tendency;
    let (w, oh, rh) = (avg.w, avg.h, res.h);
    let h = oh + rh;
    let q = i64::from(q);
    let rounding = ResidualRounding::new(q, deadzone);
    let stripe = COLUMNS_PER_STRIPE;
    // Each stripe codes its columns into its own buffers, `sw` columns wide,
    // scattered back afterwards.
    let stripes = pool.map(w.div_ceil(stripe), |s| {
        let c0 = s * stripe;
        let sw = (w - c0).min(stripe);
        let mut x = vec![0i32; sw * h];
        let mut r = vec![0i32; sw * rh];
        let mut o = vec![0i32; sw * h];
        for y in 0..rh {
            let next_y = if y + 1 < oh { y + 1 } else { y };
            let exact_avg = &avg.data[y * w + c0..][..sw];
            let next_exact = &avg.data[next_y * w + c0..][..sw];
            let exact_res = &res.data[y * w + c0..][..sw];
            let dec_avg = &decoded[y * w + c0..][..sw];
            let next_dec = &decoded[next_y * w + c0..][..sw];
            let (exact_rows, remaining) = x.split_at_mut(2 * y * sw);
            let top_exact = if y > 0 {
                &exact_rows[exact_rows.len() - sw..]
            } else {
                exact_avg
            };
            let (x0, remaining) = remaining.split_at_mut(sw);
            let x1 = &mut remaining[..sw];
            let (decoded_rows, remaining) = o.split_at_mut(2 * y * sw);
            let top_dec = if y > 0 {
                &decoded_rows[decoded_rows.len() - sw..]
            } else {
                dec_avg
            };
            let (out0, remaining) = remaining.split_at_mut(sw);
            let out1 = &mut remaining[..sw];
            let r_row = &mut r[y * sw..(y + 1) * sw];
            let exact = exact_avg
                .iter()
                .zip(next_exact)
                .zip(top_exact)
                .zip(exact_res);
            let decoded = dec_avg.iter().zip(next_dec).zip(top_dec);
            let outputs = x0
                .iter_mut()
                .zip(x1)
                .zip(r_row)
                .zip(out0.iter_mut().zip(out1));
            for ((exact, decoded), outputs) in exact.zip(decoded).zip(outputs) {
                let (((&av, &next), &top), &exact_res) = exact;
                let ((&av_dec, &next_dec), &top_dec) = decoded;
                let (((x0, x1), residual), (out0, out1)) = outputs;
                let av = i64::from(av);
                let tendency = smooth_tendency(i64::from(top), av, i64::from(next));
                let diff = i64::from(exact_res) + tendency;
                let first = av + diff / 2;
                *x0 = first as i32;
                *x1 = (first - diff) as i32;

                let av = i64::from(av_dec);
                let tendency = smooth_tendency(i64::from(top_dec), av, i64::from(next_dec));
                let diff = i64::from(*x0) - i64::from(*x1);
                let quantized = rounding.apply(diff - tendency);
                *residual = quantized as i32;
                let diff = quantized * q + tendency;
                let first = av + diff / 2;
                *out0 = first as i32;
                *out1 = (first - diff) as i32;
            }
        }
        if oh > rh {
            x[(h - 1) * sw..].copy_from_slice(&avg.data[(oh - 1) * w + c0..][..sw]);
            o[(h - 1) * sw..].copy_from_slice(&decoded[(oh - 1) * w + c0..][..sw]);
        }
        (x, r, o)
    });
    let mut input = vec![0i32; w * h];
    let mut residuals = vec![0i32; w * rh];
    let mut out = vec![0i32; w * h];
    for (s, (x, r, o)) in stripes.iter().enumerate() {
        let c0 = s * stripe;
        let sw = (w - c0).min(stripe);
        for ((input_row, out_row), (x_row, o_row)) in input
            .chunks_exact_mut(w)
            .zip(out.chunks_exact_mut(w))
            .zip(x.chunks_exact(sw).zip(o.chunks_exact(sw)))
        {
            input_row[c0..][..sw].copy_from_slice(x_row);
            out_row[c0..][..sw].copy_from_slice(o_row);
        }
        for (res_row, r_row) in residuals.chunks_exact_mut(w).zip(r.chunks_exact(sw)) {
            res_row[c0..][..sw].copy_from_slice(r_row);
        }
    }
    let input = Channel {
        data: input,
        w,
        h,
        hshift: avg.hshift,
        vshift: avg.vshift - 1,
    };
    (input, residuals, out)
}

/// The residual deadzone of a pyramid: magnitudes lose `finest_pair` of a
/// step before rounding on the finest horizontal and vertical residuals
/// (shift sum 1 and 2), where an error stays local; coarser residuals, whose
/// errors every finer level inherits, round to nearest.
#[derive(Clone, Copy)]
pub(super) struct Deadzone {
    finest_pair: f32,
}

impl Deadzone {
    /// No deadzone anywhere (the DC frame: +1.6 % BA BD with the colour one).
    pub(super) const NONE: Self = Self { finest_pair: 0.0 };

    pub(super) const COLOR: Self = Self { finest_pair: 0.15 };

    /// The deadzone of a residual with this shift sum.
    pub(super) fn at(self, shift: i32) -> f32 {
        if shift <= 2 { self.finest_pair } else { 0.0 }
    }
}

/// Re-quantize every residual of a quantized pyramid in closed loop.
/// `channels` and `quants` are the open-loop pyramid, whose final averages
/// stay as rounded; `exact` is the same pyramid unquantized, from which the
/// lossless inverse rebuilds each step's exact input. Walks the decoder's
/// inverse over the whole channel list: an in-place step's residuals follow
/// its channels, an out-of-place step's sit at the end of the list. Returns
/// the decoded planes of the input.
pub(super) fn requantize(
    channels: &mut [Channel],
    quants: &[u32],
    steps: &[SqueezeStep],
    mut exact: Vec<Channel>,
    deadzone: Deadzone,
    pool: &ThreadPool,
) -> Vec<Vec<i32>> {
    debug_assert_eq!(exact.len(), channels.len());
    let dequantized =
        |ch: &Channel, q: u32| -> Vec<i32> { ch.data.iter().map(|&v| v * q as i32).collect() };
    // Per list position: its final channel index and, once decoded, its
    // full-scale samples.
    let mut ids: Vec<usize> = (0..channels.len()).collect();
    let mut decoded: Vec<Option<Vec<i32>>> = vec![None; channels.len()];
    for step in steps.iter().rev() {
        let (begin, num) = (step.begin_c, step.num_c);
        let offset = if step.in_place {
            begin + num
        } else {
            ids.len() - num
        };
        for j in 0..num {
            let (avg, residual) = (begin + j, offset + j);
            let average = decoded[avg]
                .take()
                .unwrap_or_else(|| dequantized(&channels[ids[avg]], quants[ids[avg]]));
            let id = ids[residual];
            let closed = if step.horizontal {
                closed_h_from_exact
            } else {
                closed_v_from_exact
            };
            let (input, res, out) = closed(
                &exact[avg],
                &exact[residual],
                &average,
                quants[id] as i32,
                deadzone.at(channels[id].hshift + channels[id].vshift),
                pool,
            );
            debug_assert_eq!(res.len(), channels[id].data.len());
            channels[id].data = res;
            decoded[avg] = Some(out);
            exact[avg] = input;
        }
        ids.drain(offset..offset + num);
        decoded.drain(offset..offset + num);
        exact.drain(offset..offset + num);
    }
    decoded
        .into_iter()
        .zip(&ids)
        .map(|(plane, &id)| plane.unwrap_or_else(|| dequantized(&channels[id], quants[id])))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_rounding_matches_the_float_formula() {
        let mut seed = 3u64;
        for q in (2..3000).chain([4095, 4096, 65535, 1 << 20]) {
            let rounding = ResidualRounding::new(q, 0.0);
            let check = |r: i64| {
                assert_eq!(
                    rounding.apply(r),
                    quantize_residual(r, q, Rounding::Deadzone(0.0)),
                    "r={r} q={q}"
                )
            };
            for r in (-40_000..40_000).step_by(7) {
                check(r);
            }
            for _ in 0..2000 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                check(((seed >> 40) as i64) - (1 << 23));
            }
            for k in 0..64 {
                check(k * q + q / 2);
                check(-(k * q + q / 2));
                check(k * q + (q - 1) / 2);
            }
        }
    }
    use crate::squeeze::{
        apply_step_forward, apply_step_inverse, lossy_squeeze_steps, lossy_squeeze_steps_chroma,
    };

    fn pyramid(
        w: usize,
        h: usize,
        steps: &[SqueezeStep],
        num_c: usize,
        seed: &mut u64,
    ) -> (Vec<Vec<i32>>, Vec<Channel>) {
        let planes: Vec<Vec<i32>> = (0..num_c)
            .map(|c| {
                (0..w * h)
                    .map(|i| {
                        *seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let (x, y) = ((i % w) as i32, (i / w) as i32);
                        x * 37 + y * 23 + c as i32 * 500 + ((*seed >> 40) as i32 % 400) - 200
                    })
                    .collect()
            })
            .collect();
        let mut channels: Vec<Channel> = planes
            .iter()
            .map(|data| Channel {
                data: data.clone(),
                w,
                h,
                hshift: 0,
                vshift: 0,
            })
            .collect();
        for s in steps {
            apply_step_forward(&mut channels, s);
        }
        (planes, channels)
    }

    fn decode(channels: &[Channel], quants: &[u32], steps: &[SqueezeStep]) -> Vec<Vec<i32>> {
        let mut channels: Vec<Channel> = channels
            .iter()
            .zip(quants)
            .map(|(ch, &q)| Channel {
                data: ch.data.iter().map(|&v| v * q as i32).collect(),
                ..ch.clone()
            })
            .collect();
        for s in steps.iter().rev() {
            apply_step_inverse(&mut channels, s);
        }
        channels.into_iter().map(|ch| ch.data).collect()
    }

    #[test]
    fn requantized_pyramids_decode_to_what_the_walk_predicts() {
        let mut seed = 7u64;
        for (w, h) in [(37, 29), (64, 48), (9, 90), (130, 17), (1, 33)] {
            let scripts = [
                (lossy_squeeze_steps(w, h, 3), 3),
                (lossy_squeeze_steps_chroma(w, h, 3, &[1, 2]), 3),
                (lossy_squeeze_steps_chroma(w, h, 5, &[0, 2]), 5),
                (lossy_squeeze_steps_chroma(w, h, 1, &[]), 1),
            ];
            for (steps, num_c) in scripts {
                let (planes, channels) = pyramid(w, h, &steps, num_c, &mut seed);
                for unit in [true, false] {
                    let quants: Vec<u32> = (0..channels.len())
                        .map(|i| if unit { 1 } else { 1 + (i as u32 * 7) % 29 })
                        .collect();
                    let mut stored = channels.clone();
                    for (ch, &q) in stored.iter_mut().zip(&quants) {
                        super::super::quantize::quantize_channel_divide(ch, q as i32);
                    }
                    for deadzone in [0.0, 0.15] {
                        let mut out = stored.clone();
                        let pool = ThreadPool::new(3);
                        let walked = requantize(
                            &mut out,
                            &quants,
                            &steps,
                            channels.clone(),
                            Deadzone {
                                finest_pair: deadzone,
                            },
                            &pool,
                        );
                        let decoded = decode(&out, &quants, &steps);
                        assert_eq!(walked, decoded, "{w}x{h} {num_c}ch unit={unit}");
                        if unit {
                            assert_eq!(decoded, planes, "{w}x{h} {num_c}ch lossless");
                        }
                    }
                }
            }
        }
    }
}
