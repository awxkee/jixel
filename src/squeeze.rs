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
fn smooth_tendency(b: i64, a: i64, n: i64) -> i64 {
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
    for y in 0..h {
        let row = &chin.data[y * w..y * w + w];
        for x in 0..rw {
            let a_pix = row[2 * x];
            let b_pix = row[2 * x + 1];
            let av = average(a_pix, b_pix);
            avg.data[y * ow + x] = av;
            let diff = a_pix - b_pix;
            let next_avg = if 2 * x + 2 < 2 * rw {
                average(row[2 * x + 2], row[2 * x + 3])
            } else if w & 1 == 1 {
                row[2 * x + 2]
            } else {
                av
            };
            let left = if x > 0 { row[2 * x - 1] } else { av };
            let tendency = smooth_tendency(left as i64, av as i64, next_avg as i64);
            res.data[y * rw + x] = (diff as i64 - tendency) as i32;
        }
        if w & 1 == 1 {
            avg.data[y * ow + (ow - 1)] = row[2 * (ow - 1)];
        }
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
    for y in 0..h {
        for x in 0..rw {
            let av = avg.data[y * ow + x] as i64;
            let next_avg = if x + 1 < ow {
                avg.data[y * ow + x + 1] as i64
            } else {
                av
            };
            let left = if x > 0 {
                out.data[y * w + 2 * x - 1] as i64
            } else {
                av
            };
            let tendency = smooth_tendency(left, av, next_avg);
            let diff = res.data[y * rw + x] as i64 + tendency;
            let a_pix = av + (diff / 2);
            out.data[y * w + 2 * x] = a_pix as i32;
            out.data[y * w + 2 * x + 1] = (a_pix - diff) as i32;
        }
        if ow > rw {
            out.data[y * w + (w - 1)] = avg.data[y * ow + (ow - 1)];
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
        let avg_out = &mut avg.data[y * w..y * w + w];
        let res_out = &mut res.data[y * w..y * w + w];
        for x in 0..w {
            let a_pix = row_a[x];
            let b_pix = row_b[x];
            let av = average(a_pix, b_pix);
            avg_out[x] = av;
            let diff = a_pix - b_pix;
            let next_avg = match next_pair {
                Some((r2, r3)) => average(r2[x], r3[x]),
                None => match next_lone {
                    Some(r2) => r2[x],
                    None => av,
                },
            };
            let top = match row_top {
                Some(rt) => rt[x],
                None => av,
            };
            let tendency = smooth_tendency(top as i64, av as i64, next_avg as i64);
            res_out[x] = (diff as i64 - tendency) as i32;
        }
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
    for x in 0..w {
        for y in 0..rh {
            let av = avg.data[y * w + x] as i64;
            let next_avg = if y + 1 < oh {
                avg.data[(y + 1) * w + x] as i64
            } else {
                av
            };
            let top = if y > 0 {
                out.data[(2 * y - 1) * w + x] as i64
            } else {
                av
            };
            let tendency = smooth_tendency(top, av, next_avg);
            let diff = res.data[y * w + x] as i64 + tendency;
            let a_pix = av + (diff / 2);
            out.data[(2 * y) * w + x] = a_pix as i32;
            out.data[(2 * y + 1) * w + x] = (a_pix - diff) as i32;
        }
        if oh > rh {
            out.data[(h - 1) * w + x] = avg.data[(oh - 1) * w + x];
        }
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

/// Build a simple alternating H/V pyramid over channels [0,num_c), squeezing
/// until both dimensions are <= 8 (kMaxFirstPreviewSize). Valid, explicit
/// (signaled) sequence — need not match libjxl's chroma-first default.
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
    steps.extend(default_squeeze_steps(w, h, num_c));
    steps
}

pub(crate) fn default_squeeze_steps(mut w: usize, mut h: usize, num_c: usize) -> Vec<SqueezeStep> {
    let mut steps = Vec::new();
    const MAX_PREVIEW: usize = 8;
    while w > MAX_PREVIEW || h > MAX_PREVIEW {
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
}
