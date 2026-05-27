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

use crate::dct::{WC4, WC8, WC16};
use std::arch::aarch64::*;

#[derive(Clone, Copy)]
struct NeonDoubledVector {
    lo: float32x4_t,
    hi: float32x4_t,
}

impl NeonDoubledVector {
    #[inline]
    #[target_feature(enable = "neon")]
    fn add(self, rhs: NeonDoubledVector) -> NeonDoubledVector {
        NeonDoubledVector {
            lo: vaddq_f32(self.lo, rhs.lo),
            hi: vaddq_f32(self.hi, rhs.hi),
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn sub(self, rhs: NeonDoubledVector) -> NeonDoubledVector {
        NeonDoubledVector {
            lo: vsubq_f32(self.lo, rhs.lo),
            hi: vsubq_f32(self.hi, rhs.hi),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn muls(self, s: f32) -> NeonDoubledVector {
        NeonDoubledVector {
            lo: vmulq_n_f32(self.lo, s),
            hi: vmulq_n_f32(self.hi, s),
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn fma(self, b: NeonDoubledVector, s: f32) -> NeonDoubledVector {
        NeonDoubledVector {
            lo: vfmaq_n_f32(self.lo, b.lo, s),
            hi: vfmaq_n_f32(self.hi, b.hi, s),
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dct1d_4_v(c: &mut [NeonDoubledVector; 4]) {
    let t0 = c[0].add(c[3]);
    let t1 = c[1].add(c[2]);
    let sum = t0.add(t1);
    let diff = t0.sub(t1);

    let t2 = c[0].sub(c[3]).muls(WC4[0]);
    let t3 = c[1].sub(c[2]).muls(WC4[1]);
    let t2p = t2.add(t3);
    let t3p = t2.sub(t3);
    let t2pp = t3p.fma(t2p, std::f32::consts::SQRT_2);

    c[0] = sum;
    c[1] = t2pp;
    c[2] = diff;
    c[3] = t3p;
}

#[inline]
#[target_feature(enable = "neon")]
fn dct1d_8_v(c: &mut [NeonDoubledVector; 8]) {
    let mut evens = [
        c[0].add(c[7]),
        c[1].add(c[6]),
        c[2].add(c[5]),
        c[3].add(c[4]),
    ];
    dct1d_4_v(&mut evens);

    let mut odds = [
        c[0].sub(c[7]).muls(WC8[0]),
        c[1].sub(c[6]).muls(WC8[1]),
        c[2].sub(c[5]).muls(WC8[2]),
        c[3].sub(c[4]).muls(WC8[3]),
    ];
    dct1d_4_v(&mut odds);

    odds[0] = odds[1].fma(odds[0], std::f32::consts::SQRT_2);
    odds[1] = odds[1].add(odds[2]);
    odds[2] = odds[2].add(odds[3]);

    c[0] = evens[0];
    c[1] = odds[0];
    c[2] = evens[1];
    c[3] = odds[1];
    c[4] = evens[2];
    c[5] = odds[2];
    c[6] = evens[3];
    c[7] = odds[3];
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4(
    r0: float32x4_t,
    r1: float32x4_t,
    r2: float32x4_t,
    r3: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    let v0 = vtrn1q_f32(r0, r1);
    let v1 = vtrn2q_f32(r0, r1);
    let v2 = vtrn1q_f32(r2, r3);
    let v3 = vtrn2q_f32(r2, r3);
    let c0 = vreinterpretq_f32_f64(vtrn1q_f64(
        vreinterpretq_f64_f32(v0),
        vreinterpretq_f64_f32(v2),
    ));
    let c1 = vreinterpretq_f32_f64(vtrn1q_f64(
        vreinterpretq_f64_f32(v1),
        vreinterpretq_f64_f32(v3),
    ));
    let c2 = vreinterpretq_f32_f64(vtrn2q_f64(
        vreinterpretq_f64_f32(v0),
        vreinterpretq_f64_f32(v2),
    ));
    let c3 = vreinterpretq_f32_f64(vtrn2q_f64(
        vreinterpretq_f64_f32(v1),
        vreinterpretq_f64_f32(v3),
    ));
    (c0, c1, c2, c3)
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_8x8(c: &mut [NeonDoubledVector; 8]) {
    let (a0, a1, a2, a3) = transpose_4x4(c[0].lo, c[1].lo, c[2].lo, c[3].lo);
    let (b0, b1, b2, b3) = transpose_4x4(c[0].hi, c[1].hi, c[2].hi, c[3].hi);
    let (cc0, cc1, cc2, cc3) = transpose_4x4(c[4].lo, c[5].lo, c[6].lo, c[7].lo);
    let (d0, d1, d2, d3) = transpose_4x4(c[4].hi, c[5].hi, c[6].hi, c[7].hi);

    c[0] = NeonDoubledVector { lo: a0, hi: cc0 };
    c[1] = NeonDoubledVector { lo: a1, hi: cc1 };
    c[2] = NeonDoubledVector { lo: a2, hi: cc2 };
    c[3] = NeonDoubledVector { lo: a3, hi: cc3 };
    c[4] = NeonDoubledVector { lo: b0, hi: d0 };
    c[5] = NeonDoubledVector { lo: b1, hi: d1 };
    c[6] = NeonDoubledVector { lo: b2, hi: d2 };
    c[7] = NeonDoubledVector { lo: b3, hi: d3 };
}

#[inline]
#[target_feature(enable = "neon")]
fn load<const N: usize>(ptr: &[f32; N], stride: usize) -> [NeonDoubledVector; 8] {
    let row = |y: usize| -> NeonDoubledVector {
        unsafe {
            let p = &ptr[y * stride..];
            NeonDoubledVector {
                lo: vld1q_f32(p.as_ptr()),
                hi: vld1q_f32(p[4..].as_ptr()),
            }
        }
    };
    [
        row(0),
        row(1),
        row(2),
        row(3),
        row(4),
        row(5),
        row(6),
        row(7),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn scale_and_store(cols: &[NeonDoubledVector; 8], scale: f32, out: &mut [f32; 64]) {
    for (k, col) in cols.iter().enumerate() {
        unsafe {
            vst1q_f32(out[k * 8..].as_mut_ptr(), vmulq_n_f32(col.lo, scale));
            vst1q_f32(out[k * 8 + 4..].as_mut_ptr(), vmulq_n_f32(col.hi, scale));
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct8x8_neon(input: &[f32; 64], output: &mut [f32; 64]) {
    let mut cols = load(input, 8);
    dct1d_8_v(&mut cols);
    transpose_8x8(&mut cols);
    dct1d_8_v(&mut cols);
    scale_and_store(&cols, 1.0 / 64.0, output);
}

#[inline]
#[target_feature(enable = "neon")]
fn dct1d_16_v(c: &mut [NeonDoubledVector; 16]) {
    let mut evens = [
        c[0].add(c[15]),
        c[1].add(c[14]),
        c[2].add(c[13]),
        c[3].add(c[12]),
        c[4].add(c[11]),
        c[5].add(c[10]),
        c[6].add(c[9]),
        c[7].add(c[8]),
    ];
    let mut odds = [
        c[0].sub(c[15]).muls(WC16[0]),
        c[1].sub(c[14]).muls(WC16[1]),
        c[2].sub(c[13]).muls(WC16[2]),
        c[3].sub(c[12]).muls(WC16[3]),
        c[4].sub(c[11]).muls(WC16[4]),
        c[5].sub(c[10]).muls(WC16[5]),
        c[6].sub(c[9]).muls(WC16[6]),
        c[7].sub(c[8]).muls(WC16[7]),
    ];

    dct1d_8_v(&mut evens);
    dct1d_8_v(&mut odds);

    odds[0] = odds[1].fma(odds[0], std::f32::consts::SQRT_2);
    odds[1] = odds[1].add(odds[2]);
    odds[2] = odds[2].add(odds[3]);
    odds[3] = odds[3].add(odds[4]);
    odds[4] = odds[4].add(odds[5]);
    odds[5] = odds[5].add(odds[6]);
    odds[6] = odds[6].add(odds[7]);

    c[0] = evens[0];
    c[1] = odds[0];
    c[2] = evens[1];
    c[3] = odds[1];
    c[4] = evens[2];
    c[5] = odds[2];
    c[6] = evens[3];
    c[7] = odds[3];
    c[8] = evens[4];
    c[9] = odds[4];
    c[10] = evens[5];
    c[11] = odds[5];
    c[12] = evens[6];
    c[13] = odds[6];
    c[14] = evens[7];
    c[15] = odds[7];
}

#[inline]
#[target_feature(enable = "neon")]
fn load8_128(ptr: &[f32], stride: usize) -> [NeonDoubledVector; 8] {
    let row = |y: usize| unsafe {
        let p = ptr.get_unchecked(y * stride..);
        NeonDoubledVector {
            lo: vld1q_f32(p.as_ptr()),
            hi: vld1q_f32(p.get_unchecked(4..).as_ptr()),
        }
    };
    [
        row(0),
        row(1),
        row(2),
        row(3),
        row(4),
        row(5),
        row(6),
        row(7),
    ]
}

#[target_feature(enable = "neon")]
pub(crate) fn dct8x16_neon(input: &[f32; 128], output: &mut [f32; 128]) {
    let left = load8_128(input, 16);
    let right = load8_128(&input[8..], 16);

    let mut c = [NeonDoubledVector {
        lo: vdupq_n_f32(0.0),
        hi: vdupq_n_f32(0.0),
    }; 16];
    c[0..8].copy_from_slice(&left);
    c[8..16].copy_from_slice(&right);

    dct1d_16_v(&mut c);

    let mut cl: [NeonDoubledVector; 8] = c[0..8].try_into().unwrap();
    let mut cr: [NeonDoubledVector; 8] = c[8..16].try_into().unwrap();
    transpose_8x8(&mut cl);
    transpose_8x8(&mut cr);
    dct1d_8_v(&mut cl);
    dct1d_8_v(&mut cr);

    let scale = 1.0 / 128.0;
    for m in 0..8 {
        let base = &mut output[m * 16..];
        unsafe {
            vst1q_f32(base.as_mut_ptr(), vmulq_n_f32(cl[m].lo, scale));
            vst1q_f32(&mut base[4], vmulq_n_f32(cl[m].hi, scale));
            vst1q_f32(&mut base[8], vmulq_n_f32(cr[m].lo, scale));
            vst1q_f32(&mut base[12], vmulq_n_f32(cr[m].hi, scale));
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct16x8_neon(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut c = [NeonDoubledVector {
        lo: vdupq_n_f32(0.0),
        hi: vdupq_n_f32(0.0),
    }; 16];
    for v in 0..16 {
        let p = &input[v * 8..];
        unsafe {
            c[v] = NeonDoubledVector {
                lo: vld1q_f32(p.as_ptr()),
                hi: vld1q_f32(p[4..].as_ptr()),
            };
        }
    }

    dct1d_16_v(&mut c);

    let mut top: [NeonDoubledVector; 8] = c[0..8].try_into().unwrap();
    let mut bot: [NeonDoubledVector; 8] = c[8..16].try_into().unwrap();
    transpose_8x8(&mut top);
    transpose_8x8(&mut bot);
    dct1d_8_v(&mut top);
    dct1d_8_v(&mut bot);

    let scale = 1.0 / 128.0;
    for m in 0..8 {
        let base = &mut output[m * 16..];
        unsafe {
            vst1q_f32(base.as_mut_ptr(), vmulq_n_f32(top[m].lo, scale)); // v = 0..4
            vst1q_f32(base[4..].as_mut_ptr(), vmulq_n_f32(top[m].hi, scale)); // v = 4..8
            vst1q_f32(base[8..].as_mut_ptr(), vmulq_n_f32(bot[m].lo, scale)); // v = 8..12
            vst1q_f32(base[12..].as_mut_ptr(), vmulq_n_f32(bot[m].hi, scale)); // v = 12..16
        }
    }
}
