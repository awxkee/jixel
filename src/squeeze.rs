/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

#![allow(dead_code)]

mod row_kernels;

pub(crate) use row_kernels::{ClosedRow, ClosedVerticalFn, InverseRow, ResidualRounding};

pub(crate) fn selected_closed_vertical_fn() -> ClosedVerticalFn {
    row_kernels::selected().closed_vertical
}

/// A modular channel: integer samples in row-major `w`×`h`, plus shifts.
#[derive(Clone)]
pub(crate) struct Channel {
    pub(crate) data: Vec<i32>,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) hshift: i32,
    pub(crate) vshift: i32,
}
impl Channel {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        Channel {
            data: vec![0; w * h],
            w,
            h,
            hshift: 0,
            vshift: 0,
        }
    }
    #[inline]
    pub(crate) fn at(&self, x: usize, y: usize) -> i32 {
        self.data[y * self.w + x]
    }
    #[inline]
    pub(crate) fn set(&mut self, x: usize, y: usize, v: i32) {
        self.data[y * self.w + x] = v;
    }
}

#[inline]
fn average(a: i32, b: i32) -> i32 {
    // libjxl: ((X + Y + ((X > Y) ? 1 : 0)) >> 1)   (arithmetic shift)
    ((a as i64 + b as i64 + if a > b { 1 } else { 0 }) >> 1) as i32
}

#[inline]
pub(crate) fn smooth_tendency(b: i64, a: i64, n: i64) -> i64 {
    // byte-exact libjxl SmoothTendency(B,a,n)
    let mut diff = 0i64;
    if b >= a && a >= n {
        diff = (4 * b - 3 * n - a + 6) / 12;
        if diff - (diff & 1) > 2 * (b - a) {
            diff = 2 * (b - a) + 1;
        }
        if diff + (diff & 1) > 2 * (a - n) {
            diff = 2 * (a - n);
        }
    } else if b <= a && a <= n {
        diff = (4 * b - 3 * n - a - 6) / 12;
        if diff + (diff & 1) < 2 * (b - a) {
            diff = 2 * (b - a) - 1;
        }
        if diff - (diff & 1) < 2 * (a - n) {
            diff = 2 * (a - n);
        }
    }
    diff
}

/// Forward horizontal squeeze: returns (avg, residual) channels (FwdHSqueeze).
pub(crate) fn fwd_h_squeeze(chin: &Channel) -> (Channel, Channel) {
    let w = chin.w;
    let h = chin.h;
    let ow = w.div_ceil(2); // avg width
    let rw = w - ow; // residual width
    let mut avg = Channel {
        data: vec![0; ow * h],
        w: ow,
        h,
        hshift: chin.hshift + 1,
        vshift: chin.vshift,
    };
    let mut res = Channel {
        data: vec![0; rw * h],
        w: rw,
        h,
        hshift: chin.hshift + 1,
        vshift: chin.vshift,
    };
    if w == 0 {
        return (avg, res);
    }
    let forward = row_kernels::selected().forward_horizontal;
    for (y, (row, avg_row)) in chin
        .data
        .chunks_exact(w)
        .zip(avg.data.chunks_exact_mut(ow))
        .enumerate()
    {
        forward(row, avg_row, &mut res.data[y * rw..(y + 1) * rw]);
    }
    (avg, res)
}

/// Inverse horizontal squeeze (InvHSqueeze scalar path) — for self-verification.
pub(crate) fn inv_h_squeeze(avg: &Channel, res: &Channel) -> Channel {
    let ow = avg.w;
    let rw = res.w;
    let w = ow + rw;
    let h = avg.h;
    let mut out = Channel {
        data: vec![0; w * h],
        w,
        h,
        hshift: avg.hshift - 1,
        vshift: avg.vshift,
    };
    for (y, out_row) in out.data.chunks_exact_mut(w.max(1)).enumerate() {
        let avg_row = &avg.data[y * ow..(y + 1) * ow];
        let res_row = &res.data[y * rw..(y + 1) * rw];
        let mut next_averages = avg_row.iter().skip(1);
        let mut left = None;
        let (pairs, _) = out_row.as_chunks_mut::<2>();
        for ((&av, &residual), [a_pix, b_pix]) in avg_row.iter().zip(res_row).zip(pairs.iter_mut())
        {
            let av = i64::from(av);
            let next_avg = next_averages.next().map_or(av, |&next| i64::from(next));
            let tendency = smooth_tendency(left.unwrap_or(av), av, next_avg);
            let diff = i64::from(residual) + tendency;
            let first = av + (diff / 2);
            *a_pix = first as i32;
            *b_pix = (first - diff) as i32;
            left = Some(i64::from(*b_pix));
        }
        if ow > rw {
            *out_row.last_mut().unwrap() = *avg_row.last().unwrap();
        }
    }
    out
}

/// Forward vertical squeeze: returns (avg, residual) (FwdVSqueeze).
pub(crate) fn fwd_v_squeeze(chin: &Channel) -> (Channel, Channel) {
    let w = chin.w;
    let h = chin.h;
    let oh = h.div_ceil(2);
    let rh = h - oh;
    let mut avg = Channel {
        data: vec![0; w * oh],
        w,
        h: oh,
        hshift: chin.hshift,
        vshift: chin.vshift + 1,
    };
    let mut res = Channel {
        data: vec![0; w * rh],
        w,
        h: rh,
        hshift: chin.hshift,
        vshift: chin.vshift + 1,
    };
    let forward = row_kernels::selected().forward_vertical;
    for y in 0..rh {
        let row_a = &chin.data[(2 * y) * w..(2 * y) * w + w];
        let row_b = &chin.data[(2 * y + 1) * w..(2 * y + 1) * w + w];
        let row_top: Option<&[i32]> = if y > 0 {
            Some(&chin.data[(2 * y - 1) * w..(2 * y - 1) * w + w])
        } else {
            None
        };
        // next-average source: a full pair (2y+2,2y+3), a lone row (2y+2), or none.
        let next_pair: Option<(&[i32], &[i32])> = if 2 * y + 2 < 2 * rh {
            Some((
                &chin.data[(2 * y + 2) * w..(2 * y + 2) * w + w],
                &chin.data[(2 * y + 3) * w..(2 * y + 3) * w + w],
            ))
        } else {
            None
        };
        let next_lone: Option<&[i32]> = if next_pair.is_none() && h & 1 == 1 {
            Some(&chin.data[(2 * y + 2) * w..(2 * y + 2) * w + w])
        } else {
            None
        };
        let next = next_pair.map_or_else(
            || next_lone.map_or([row_a, row_b], |row| [row, row]),
            |(a, b)| [a, b],
        );
        forward(
            row_kernels::ForwardRow {
                pixels: [row_a, row_b],
                top: row_top,
                next,
            },
            [
                &mut avg.data[y * w..(y + 1) * w],
                &mut res.data[y * w..(y + 1) * w],
            ],
        );
    }
    if h & 1 == 1 {
        let src = &chin.data[(2 * (oh - 1)) * w..(2 * (oh - 1)) * w + w];
        avg.data[(oh - 1) * w..(oh - 1) * w + w].copy_from_slice(src);
    }
    (avg, res)
}

pub(crate) fn inv_v_squeeze(avg: &Channel, res: &Channel) -> Channel {
    let w = avg.w;
    let oh = avg.h;
    let rh = res.h;
    let h = oh + rh;
    let mut out = Channel {
        data: vec![0; w * h],
        w,
        h,
        hshift: avg.hshift,
        vshift: avg.vshift - 1,
    };
    if w == 0 || h == 0 {
        return out;
    }
    let inverse = row_kernels::selected().inverse_vertical;
    let avg_rows = avg.data.chunks_exact(w);
    let mut next_avg_rows = avg_rows.clone().skip(1);
    let mut top_row: Option<&[i32]> = None;
    for ((avg_row, res_row), out_pair) in avg_rows
        .zip(res.data.chunks_exact(w))
        .zip(out.data.chunks_exact_mut(2 * w))
    {
        let next_avg_row = next_avg_rows.next().unwrap_or(avg_row);
        let (a_row, b_row) = out_pair.split_at_mut(w);
        inverse(
            row_kernels::InverseRow {
                average: avg_row,
                residual: res_row,
                top: top_row,
                next: next_avg_row,
            },
            [a_row, b_row],
        );
        top_row = Some(b_row);
    }
    if oh > rh {
        out.data[(h - 1) * w..h * w].copy_from_slice(&avg.data[(oh - 1) * w..oh * w]);
    }
    out
}

/// One squeeze step (matches libjxl SqueezeParams).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SqueezeStep {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_c: usize,
    pub num_c: usize,
}

/// Apply a forward squeeze step to a channel list (matches MetaSqueeze ordering:
/// channel[c] becomes its average; the residual is inserted at offset+(c-begin_c),
/// offset = endc+1 for in-place, else end of list).
pub(crate) fn apply_step_forward(channels: &mut Vec<Channel>, s: &SqueezeStep) {
    let endc = s.begin_c + s.num_c - 1;
    let offset = if s.in_place { endc + 1 } else { channels.len() };
    for i in 0..s.num_c {
        let c = s.begin_c + i;
        let (avg, res) = if s.horizontal {
            fwd_h_squeeze(&channels[c])
        } else {
            fwd_v_squeeze(&channels[c])
        };
        channels[c] = avg;
        channels.insert(offset + i, res);
    }
}

/// Invert one squeeze step (for self-verification; reverses apply_step_forward).
pub(crate) fn apply_step_inverse(channels: &mut Vec<Channel>, s: &SqueezeStep) {
    let endc = s.begin_c + s.num_c - 1;
    let offset = if s.in_place {
        endc + 1
    } else {
        channels.len() - s.num_c
    };
    // Remove residuals from the back so indices stay valid, recombining each.
    for i in (0..s.num_c).rev() {
        let c = s.begin_c + i;
        let res = channels.remove(offset + i);
        let avg = &channels[c];
        let full = if s.horizontal {
            inv_h_squeeze(avg, &res)
        } else {
            inv_v_squeeze(avg, &res)
        };
        channels[c] = full;
    }
}

/// Fold one step of a quantized pyramid into its decoded averages. Only the
/// step's averages and residuals are materialized at full scale; all other
/// coefficients retain their values and multipliers. Invalid geometry or
/// overflowing dequantization leaves both vectors unchanged.
pub(crate) fn fold_quantized_step(
    channels: &mut Vec<Channel>,
    quants: &mut Vec<u32>,
    step: &SqueezeStep,
) -> bool {
    if channels.len() != quants.len() || step.num_c == 0 {
        return false;
    }
    let Some(end) = step.begin_c.checked_add(step.num_c) else {
        return false;
    };
    let Some(offset) = (if step.in_place {
        Some(end)
    } else {
        channels.len().checked_sub(step.num_c)
    }) else {
        return false;
    };
    let Some(res_end) = offset.checked_add(step.num_c) else {
        return false;
    };
    if end > offset || res_end > channels.len() {
        return false;
    }

    let dequantize = |ch: &Channel, q: u32| -> Option<Channel> {
        if q == 0 || ch.w.checked_mul(ch.h)? != ch.data.len() {
            return None;
        }
        let data: Option<Vec<i32>> = ch
            .data
            .iter()
            .map(|&v| i32::try_from(i64::from(v) * i64::from(q)).ok())
            .collect();
        Some(Channel {
            data: data?,
            w: ch.w,
            h: ch.h,
            hshift: ch.hshift,
            vshift: ch.vshift,
        })
    };
    let mut merged = Vec::with_capacity(step.num_c);
    for i in 0..step.num_c {
        let avg_index = step.begin_c + i;
        let res_index = offset + i;
        let avg = &channels[avg_index];
        let res = &channels[res_index];
        let same_shift = avg.hshift == res.hshift && avg.vshift == res.vshift;
        let valid_shape = if step.horizontal {
            avg.w
                .checked_add(res.w)
                .is_some_and(|w| avg.h == res.h && avg.w == w.div_ceil(2) && avg.hshift > 0)
        } else {
            avg.h
                .checked_add(res.h)
                .is_some_and(|h| avg.w == res.w && avg.h == h.div_ceil(2) && avg.vshift > 0)
        };
        if !same_shift || !valid_shape || avg.hshift < 0 || avg.vshift < 0 {
            return false;
        }
        let Some(avg) = dequantize(avg, quants[avg_index]) else {
            return false;
        };
        let Some(res) = dequantize(res, quants[res_index]) else {
            return false;
        };
        // Reuse the decoder-matching inverse, including its defined i64 to
        // i32 output casts for signed intermediate values.
        merged.push(if step.horizontal {
            inv_h_squeeze(&avg, &res)
        } else {
            inv_v_squeeze(&avg, &res)
        });
    }

    // Commit only after every pair was checked and rebuilt successfully.
    for (i, channel) in merged.into_iter().enumerate() {
        channels[step.begin_c + i] = channel;
        quants[step.begin_c + i] = 1;
    }
    channels.drain(offset..res_end);
    quants.drain(offset..res_end);
    true
}

/// The XYB lossy pyramid over channels [0,num_c): a 4:2:0 chroma pre-squeeze,
/// then [`lossy_shared_steps`] until both dimensions are <= 8
/// (kMaxFirstPreviewSize). Valid, explicit (signaled) sequence.
pub(crate) fn lossy_squeeze_steps(w: usize, h: usize, num_c: usize) -> Vec<SqueezeStep> {
    let mut steps = Vec::new();
    if num_c >= 3 {
        for horizontal in [true, false] {
            steps.push(SqueezeStep {
                horizontal,
                in_place: false,
                begin_c: 1,
                num_c: 2,
            });
        }
    }
    steps.extend(lossy_shared_steps(w, h, num_c));
    steps
}

/// The lossy pyramid over `num_c` channels with the listed chroma channels
/// pre-squeezed once (4:2:0) ahead of the shared steps; each pre-squeeze
/// appends its residual after every other channel.
pub(crate) fn lossy_squeeze_steps_chroma(
    w: usize,
    h: usize,
    num_c: usize,
    chroma: &[usize],
) -> Vec<SqueezeStep> {
    let mut steps = Vec::new();
    for &c in chroma {
        for horizontal in [true, false] {
            steps.push(SqueezeStep {
                horizontal,
                in_place: false,
                begin_c: c,
                num_c: 1,
            });
        }
    }
    steps.extend(lossy_shared_steps(w, h, num_c));
    steps
}

pub(crate) fn default_squeeze_steps(w: usize, h: usize, num_c: usize) -> Vec<SqueezeStep> {
    squeeze_steps_levels(w, h, num_c, usize::MAX)
}

/// The shared steps of a lossy pyramid: [`default_squeeze_steps`], led by
/// one vertical step on images taller than wide (libjxl's order).
pub(crate) fn lossy_shared_steps(w: usize, h: usize, num_c: usize) -> Vec<SqueezeStep> {
    const MAX_PREVIEW: usize = 8;
    if h <= w || h <= MAX_PREVIEW {
        return default_squeeze_steps(w, h, num_c);
    }
    let mut steps = vec![SqueezeStep {
        horizontal: false,
        in_place: true,
        begin_c: 0,
        num_c,
    }];
    steps.extend(default_squeeze_steps(w, h.div_ceil(2), num_c));
    steps
}

/// [`default_squeeze_steps`] cut to its `levels` finest levels; a level
/// halves every dimension still above the preview size.
pub(crate) fn squeeze_steps_levels(
    mut w: usize,
    mut h: usize,
    num_c: usize,
    levels: usize,
) -> Vec<SqueezeStep> {
    let mut steps = Vec::new();
    const MAX_PREVIEW: usize = 8;
    for _ in 0..levels {
        if w <= MAX_PREVIEW && h <= MAX_PREVIEW {
            break;
        }
        if w > MAX_PREVIEW {
            steps.push(SqueezeStep {
                horizontal: true,
                in_place: true,
                begin_c: 0,
                num_c,
            });
            w = w.div_ceil(2);
        }
        if h > MAX_PREVIEW {
            steps.push(SqueezeStep {
                horizontal: false,
                in_place: true,
                begin_c: 0,
                num_c,
            });
            h = h.div_ceil(2);
        }
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossy_pyramids_start_along_the_longer_side_of_portraits() {
        let preview = |steps: &[SqueezeStep], mut w: usize, mut h: usize| {
            for s in steps.iter().filter(|s| s.in_place) {
                if s.horizontal {
                    w = w.div_ceil(2);
                } else {
                    h = h.div_ceil(2);
                }
            }
            (w, h)
        };
        for (w, h) in [
            (512, 768),
            (100, 1000),
            (9, 17),
            (768, 512),
            (512, 512),
            (4, 40),
        ] {
            let steps = lossy_squeeze_steps(w, h, 3);
            let first = steps.iter().find(|s| s.in_place).unwrap();
            assert_eq!(first.horizontal, h <= w, "{w}x{h}");
            let (pw, ph) = preview(&steps, w, h);
            assert!(pw <= 8 && ph <= 8, "{w}x{h} -> {pw}x{ph}");
            let plain = lossy_squeeze_steps_chroma(w, h, 3, &[1, 2]);
            assert_eq!(
                plain.iter().find(|s| s.in_place).unwrap().horizontal,
                h <= w,
                "{w}x{h}"
            );
            // Lossless and lossy-alpha pyramids keep the horizontal start.
            assert!(
                default_squeeze_steps(w, h, 1)[0].horizontal || w <= 8,
                "{w}x{h}"
            );
        }
    }
    fn rnd(seed: &mut u64) -> i32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as i32 % 1001) - 500
    }
    #[test]
    fn h_squeeze_roundtrips_identity() {
        let mut s = 12345u64;
        for _ in 0..400 {
            let w = 1 + (rnd(&mut s).unsigned_abs() as usize % 40);
            let h = 1 + (rnd(&mut s).unsigned_abs() as usize % 8);
            let mut c = Channel::new(w, h);
            for v in c.data.iter_mut() {
                *v = rnd(&mut s);
            }
            let (avg, res) = fwd_h_squeeze(&c);
            let rec = inv_h_squeeze(&avg, &res);
            assert_eq!(
                rec.data, c.data,
                "H squeeze roundtrip failed w={} h={}",
                w, h
            );
        }
    }
    #[test]
    fn v_squeeze_roundtrips_identity() {
        let mut s = 999u64;
        for _ in 0..400 {
            let w = 1 + (rnd(&mut s).unsigned_abs() as usize % 8);
            let h = 1 + (rnd(&mut s).unsigned_abs() as usize % 40);
            let mut c = Channel::new(w, h);
            for v in c.data.iter_mut() {
                *v = rnd(&mut s);
            }
            let (avg, res) = fwd_v_squeeze(&c);
            let rec = inv_v_squeeze(&avg, &res);
            assert_eq!(
                rec.data, c.data,
                "V squeeze roundtrip failed w={} h={}",
                w, h
            );
        }
    }
    #[test]
    fn default_pyramid_roundtrips_identity() {
        // Apply the full alternating H/V pyramid to 3 channels, then invert the
        // whole sequence in reverse, and check every channel is bit-identical.
        let mut s = 424242u64;
        for &(w, h) in &[
            (256usize, 256usize),
            (255, 257),
            (40, 9),
            (9, 40),
            (17, 17),
            (8, 8),
        ] {
            let orig: Vec<Channel> = (0..3)
                .map(|_| {
                    let mut c = Channel::new(w, h);
                    for v in c.data.iter_mut() {
                        *v = rnd(&mut s);
                    }
                    c
                })
                .collect();
            let steps = default_squeeze_steps(w, h, 3);
            let mut chans = orig.clone();
            for st in &steps {
                apply_step_forward(&mut chans, st);
            }
            // invert in reverse order
            for st in steps.iter().rev() {
                apply_step_inverse(&mut chans, st);
            }
            assert_eq!(chans.len(), 3, "channel count not restored ({}x{})", w, h);
            for c in 0..3 {
                assert_eq!(
                    chans[c].data, orig[c].data,
                    "pyramid roundtrip failed ch{} {}x{}",
                    c, w, h
                );
                assert_eq!(
                    (chans[c].w, chans[c].h),
                    (w, h),
                    "dims not restored ch{} {}x{}",
                    c,
                    w,
                    h
                );
            }
        }
    }

    fn decoded_channels(
        channels: &[Channel],
        quants: &[u32],
        steps: &[SqueezeStep],
    ) -> Vec<Channel> {
        let mut result = channels.to_vec();
        for (channel, &quant) in result.iter_mut().zip(quants) {
            for value in &mut channel.data {
                *value = value.wrapping_mul(quant as i32);
            }
        }
        for step in steps.iter().rev() {
            apply_step_inverse(&mut result, step);
        }
        result
    }

    fn assert_channels_equal(actual: &[Channel], expected: &[Channel]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(
                (a.w, a.h, a.hshift, a.vshift),
                (b.w, b.h, b.hshift, b.vshift)
            );
            assert_eq!(a.data, b.data);
        }
    }

    #[test]
    fn folding_quantized_steps_preserves_mixed_channel_reconstruction() {
        let mut seed = 1234567;
        for (w, h) in [(37, 29), (1, 33), (33, 1), (65, 18)] {
            for pre_chroma in [false, true] {
                let mut channels: Vec<Channel> = (0..4)
                    .map(|_| {
                        let mut channel = Channel::new(w, h);
                        for value in &mut channel.data {
                            *value = rnd(&mut seed);
                        }
                        channel
                    })
                    .collect();
                let steps = if pre_chroma {
                    lossy_squeeze_steps_chroma(w, h, 4, &[1, 2])
                } else {
                    default_squeeze_steps(w, h, 4)
                };
                for step in &steps {
                    apply_step_forward(&mut channels, step);
                }
                let mut quants: Vec<u32> =
                    (0..channels.len()).map(|i| [1, 2, 3, 7][i % 4]).collect();
                for (channel, &q) in channels.iter_mut().zip(&quants) {
                    for value in &mut channel.data {
                        *value /= q as i32;
                    }
                }
                let expected = decoded_channels(&channels, &quants, &steps);
                for retained in (0..steps.len()).rev() {
                    let step = &steps[retained];
                    let previous = channels.clone();
                    let old_quants = quants.clone();
                    let end = step.begin_c + step.num_c;
                    let offset = if step.in_place {
                        end
                    } else {
                        previous.len() - step.num_c
                    };
                    assert!(fold_quantized_step(&mut channels, &mut quants, step));
                    assert_channels_equal(
                        &decoded_channels(&channels, &quants, &steps[..retained]),
                        &expected,
                    );
                    // Unrelated color, extra, and previously appended bands
                    // retain both coefficients and quantizers verbatim.
                    for i in 0..previous.len() {
                        if (step.begin_c..end).contains(&i)
                            || (offset..offset + step.num_c).contains(&i)
                        {
                            continue;
                        }
                        let target = if i >= offset + step.num_c {
                            i - step.num_c
                        } else {
                            i
                        };
                        assert_channels_equal(&channels[target..=target], &previous[i..=i]);
                        assert_eq!(quants[target], old_quants[i]);
                    }
                    assert!(quants[step.begin_c..end].iter().all(|&q| q == 1));
                }
            }
        }
    }

    #[test]
    fn folding_quantized_steps_rejects_invalid_state_transactionally() {
        let step = SqueezeStep {
            horizontal: true,
            in_place: true,
            begin_c: 1,
            num_c: 2,
        };
        let mut channels: Vec<Channel> = (0..4)
            .map(|i| {
                let mut ch = Channel::new(9, 7);
                ch.data.fill(i + 1);
                ch
            })
            .collect();
        apply_step_forward(&mut channels, &step);
        let quants = vec![1; channels.len()];
        let reject = |mut c: Vec<Channel>, mut q: Vec<u32>, s: SqueezeStep| {
            let original = c.clone();
            let original_q = q.clone();
            assert!(!fold_quantized_step(&mut c, &mut q, &s));
            assert_channels_equal(&c, &original);
            assert_eq!(q, original_q);
        };
        for invalid in [
            SqueezeStep { num_c: 0, ..step },
            SqueezeStep {
                begin_c: usize::MAX,
                ..step
            },
            SqueezeStep {
                num_c: usize::MAX,
                ..step
            },
            SqueezeStep { begin_c: 5, ..step },
            SqueezeStep {
                begin_c: 3,
                in_place: false,
                ..step
            },
        ] {
            reject(channels.clone(), quants.clone(), invalid);
        }
        reject(channels.clone(), quants[..quants.len() - 1].to_vec(), step);
        let mut invalid_q = quants.clone();
        invalid_q[4] = 0;
        reject(channels.clone(), invalid_q, step);
        for value in [i32::MAX, i32::MIN] {
            let mut c = channels.clone();
            let mut q = quants.clone();
            // The first pair is valid; failure in the second must not commit
            // the already reconstructed first pair.
            c[4].data[0] = value;
            q[4] = 2;
            reject(c, q, step);
        }
        let mut invalid = channels.clone();
        invalid[4].data.pop();
        reject(invalid, quants.clone(), step);
        let mut invalid = channels.clone();
        invalid[4].vshift += 1;
        reject(invalid, quants.clone(), step);
        let mut invalid = channels.clone();
        invalid[4].h += 1;
        reject(invalid, quants.clone(), step);
        let mut invalid = channels.clone();
        invalid[4].w += 2;
        reject(invalid, quants, step);
    }

    #[test]
    fn folding_quantized_steps_matches_inverse_output_casts() {
        for horizontal in [true, false] {
            let (w, h, hshift, vshift) = if horizontal {
                (2, 1, 1, 0)
            } else {
                (1, 2, 0, 1)
            };
            let channel = Channel {
                data: vec![i32::MAX; w * h],
                w,
                h,
                hshift,
                vshift,
            };
            let step = SqueezeStep {
                horizontal,
                in_place: true,
                begin_c: 0,
                num_c: 1,
            };
            let mut channels = vec![channel.clone(), channel];
            let mut quants = vec![1, 1];
            let expected = decoded_channels(&channels, &quants, &[step]);
            assert!(expected[0].data.iter().any(|&v| v < 0));
            assert!(fold_quantized_step(&mut channels, &mut quants, &step));
            assert_channels_equal(&channels, &expected);
            assert_eq!(quants, vec![1]);
        }
    }
}
