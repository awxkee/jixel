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

use crate::dct::{
    DCT64_SPLIT_COS, DCT64_SPLIT_SIN, DctInput, INV_WC4, INV_WC8, INV_WC16, INV_WC32, INV_WC64,
    RESAMPLE_SCALE_16_TO_2, RESAMPLE_SCALE_32_TO_4, RESAMPLE_SCALE_64_TO_8, WC4, WC8, WC16, WC32,
};
use std::arch::aarch64::*;
use std::mem::MaybeUninit;

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
pub(super) fn transpose_4x4(
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
fn load(input: DctInput<'_, 8, 8>) -> [NeonDoubledVector; 8] {
    let row = |y: usize| -> NeonDoubledVector {
        unsafe {
            let p = input.row(y);
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

#[inline]
#[target_feature(enable = "neon")]
fn scale_and_store_uninit(
    cols: &[NeonDoubledVector; 8],
    scale: f32,
    out: &mut MaybeUninit<[f32; 64]>,
) {
    for (k, col) in cols.iter().enumerate() {
        unsafe {
            let dst_ptr = out.as_mut_ptr() as *mut f32;
            vst1q_f32(dst_ptr.add(k * 8), vmulq_n_f32(col.lo, scale));
            vst1q_f32(dst_ptr.add(k * 8 + 4), vmulq_n_f32(col.hi, scale));
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn identity8x8_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut means = [0.0f32; 4];
    for qy in 0..2 {
        let anchor_row = input.row(qy * 4 + 1);
        let anchor_lo = vdupq_n_f32(anchor_row[1]);
        let anchor_hi = vdupq_n_f32(anchor_row[5]);
        let mut sum_lo = vdupq_n_f32(0.0);
        let mut sum_hi = vdupq_n_f32(0.0);
        for iy in 0..4 {
            let row = input.row(qy * 4 + iy);
            let pixels_lo = unsafe { vld1q_f32(row.as_ptr()) };
            let pixels_hi = unsafe { vld1q_f32(row[4..].as_ptr()) };
            sum_lo = vaddq_f32(sum_lo, pixels_lo);
            sum_hi = vaddq_f32(sum_hi, pixels_hi);

            let residual_lo = vsubq_f32(pixels_lo, anchor_lo);
            let residual_hi = vsubq_f32(pixels_hi, anchor_hi);
            let packed_lo = vzip1q_f32(residual_lo, residual_hi);
            let packed_hi = vzip2q_f32(residual_lo, residual_hi);
            let dst = &mut output[(qy + iy * 2) * 8..];
            unsafe {
                vst1q_f32(dst.as_mut_ptr(), packed_lo);
                vst1q_f32(dst[4..].as_mut_ptr(), packed_hi);
            }
        }
        means[qy * 2] = vaddvq_f32(sum_lo) * (1.0 / 16.0);
        means[qy * 2 + 1] = vaddvq_f32(sum_hi) * (1.0 / 16.0);
    }

    for qy in 0..2 {
        for qx in 0..2 {
            output[(qy + 2) * 8 + qx + 2] = output[qy * 8 + qx];
        }
    }
    let [b00, b01, b10, b11] = means;
    output[0] = (b00 + b01 + b10 + b11) * 0.25;
    output[1] = (b00 + b01 - b10 - b11) * 0.25;
    output[8] = (b00 - b01 + b10 - b11) * 0.25;
    output[9] = (b00 - b01 - b10 + b11) * 0.25;
}

#[target_feature(enable = "neon")]
pub(crate) fn inv_identity8x8_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let c00 = input.row(0)[0];
    let c01 = input.row(0)[1];
    let c10 = input.row(1)[0];
    let c11 = input.row(1)[1];
    let means = [
        c00 + c01 + c10 + c11,
        c00 + c01 - c10 - c11,
        c00 - c01 + c10 - c11,
        c00 - c01 - c10 + c11,
    ];

    for qy in 0..2 {
        let mut left = [vdupq_n_f32(0.0); 4];
        let mut right = [vdupq_n_f32(0.0); 4];
        for iy in 0..4 {
            let row = input.row(qy + iy * 2);
            let packed_lo = unsafe { vld1q_f32(row.as_ptr()) };
            let packed_hi = unsafe { vld1q_f32(row[4..].as_ptr()) };
            left[iy] = vuzp1q_f32(packed_lo, packed_hi);
            right[iy] = vuzp2q_f32(packed_lo, packed_hi);
        }
        left[0] = vsetq_lane_f32::<0>(input.row(qy + 2)[2], left[0]);
        right[0] = vsetq_lane_f32::<0>(input.row(qy + 2)[3], right[0]);
        left[1] = vsetq_lane_f32::<1>(0.0, left[1]);
        right[1] = vsetq_lane_f32::<1>(0.0, right[1]);

        let left_sum = vaddq_f32(vaddq_f32(left[0], left[1]), vaddq_f32(left[2], left[3]));
        let right_sum = vaddq_f32(vaddq_f32(right[0], right[1]), vaddq_f32(right[2], right[3]));
        let anchor_lo = vdupq_n_f32(means[qy * 2] - vaddvq_f32(left_sum) * (1.0 / 16.0));
        let anchor_hi = vdupq_n_f32(means[qy * 2 + 1] - vaddvq_f32(right_sum) * (1.0 / 16.0));
        for iy in 0..4 {
            let dst = &mut output[(qy * 4 + iy) * 8..];
            unsafe {
                vst1q_f32(dst.as_mut_ptr(), vaddq_f32(left[iy], anchor_lo));
                vst1q_f32(dst[4..].as_mut_ptr(), vaddq_f32(right[iy], anchor_hi));
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dct2x2_forward(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
    d: float32x4_t,
) -> [float32x4_t; 4] {
    let scale = vdupq_n_f32(0.25);
    let ab = vaddq_f32(a, b);
    let amb = vsubq_f32(a, b);
    [
        vmulq_f32(vaddq_f32(vaddq_f32(ab, c), d), scale),
        vmulq_f32(vsubq_f32(vsubq_f32(ab, c), d), scale),
        vmulq_f32(vsubq_f32(vaddq_f32(amb, c), d), scale),
        vmulq_f32(vaddq_f32(vsubq_f32(amb, c), d), scale),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn dct2x2_inverse(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
    d: float32x4_t,
) -> [float32x4_t; 4] {
    let ab = vaddq_f32(a, b);
    let amb = vsubq_f32(a, b);
    [
        vaddq_f32(vaddq_f32(ab, c), d),
        vsubq_f32(vsubq_f32(ab, c), d),
        vsubq_f32(vaddq_f32(amb, c), d),
        vaddq_f32(vsubq_f32(amb, c), d),
    ]
}

#[target_feature(enable = "neon")]
pub(crate) fn dct2x2_8x8_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    for y in 0..4 {
        let row0 = input.row(y * 2);
        let row1 = input.row(y * 2 + 1);
        let r0lo = unsafe { vld1q_f32(row0.as_ptr()) };
        let r0hi = unsafe { vld1q_f32(row0[4..].as_ptr()) };
        let r1lo = unsafe { vld1q_f32(row1.as_ptr()) };
        let r1hi = unsafe { vld1q_f32(row1[4..].as_ptr()) };
        let q = dct2x2_forward(
            vuzp1q_f32(r0lo, r0hi),
            vuzp2q_f32(r0lo, r0hi),
            vuzp1q_f32(r1lo, r1hi),
            vuzp2q_f32(r1lo, r1hi),
        );
        unsafe {
            vst1q_f32(output[y * 8..].as_mut_ptr(), q[0]);
            vst1q_f32(output[y * 8 + 4..].as_mut_ptr(), q[1]);
            vst1q_f32(output[(y + 4) * 8..].as_mut_ptr(), q[2]);
            vst1q_f32(output[(y + 4) * 8 + 4..].as_mut_ptr(), q[3]);
        }
    }
    let mut stage4 = [[vdupq_n_f32(0.0); 4]; 2];
    for (y, q) in stage4.iter_mut().enumerate() {
        let r0 = unsafe { vld1q_f32(output[(y * 2) * 8..].as_ptr()) };
        let r1 = unsafe { vld1q_f32(output[(y * 2 + 1) * 8..].as_ptr()) };
        *q = dct2x2_forward(
            vuzp1q_f32(r0, r0),
            vuzp2q_f32(r0, r0),
            vuzp1q_f32(r1, r1),
            vuzp2q_f32(r1, r1),
        );
    }
    for (y, q) in stage4.into_iter().enumerate() {
        unsafe {
            vst1_f32(output[y * 8..].as_mut_ptr(), vget_low_f32(q[0]));
            vst1_f32(output[y * 8 + 2..].as_mut_ptr(), vget_low_f32(q[1]));
            vst1_f32(output[(y + 2) * 8..].as_mut_ptr(), vget_low_f32(q[2]));
            vst1_f32(output[(y + 2) * 8 + 2..].as_mut_ptr(), vget_low_f32(q[3]));
        }
    }
    let a = output[0];
    let b = output[1];
    let c = output[8];
    let d = output[9];
    output[0] = (a + b + c + d) * 0.25;
    output[1] = (a + b - c - d) * 0.25;
    output[8] = (a - b + c - d) * 0.25;
    output[9] = (a - b - c + d) * 0.25;
}

#[target_feature(enable = "neon")]
pub(crate) fn inv_dct2x2_8x8_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut a = std::array::from_fn::<_, 64, _>(|i| input.row(i / 8)[i % 8]);
    let c00 = a[0];
    let c01 = a[1];
    let c10 = a[8];
    let c11 = a[9];
    a[0] = c00 + c01 + c10 + c11;
    a[1] = c00 + c01 - c10 - c11;
    a[8] = c00 - c01 + c10 - c11;
    a[9] = c00 - c01 - c10 + c11;

    let mut stage4 = [[vdupq_n_f32(0.0); 4]; 2];
    for (y, q) in stage4.iter_mut().enumerate() {
        *q = dct2x2_inverse(
            unsafe { vld1q_f32(a[y * 8..].as_ptr()) },
            unsafe { vld1q_f32(a[y * 8 + 2..].as_ptr()) },
            unsafe { vld1q_f32(a[(y + 2) * 8..].as_ptr()) },
            unsafe { vld1q_f32(a[(y + 2) * 8 + 2..].as_ptr()) },
        );
    }
    for (y, q) in stage4.into_iter().enumerate() {
        let row0 = vzip1q_f32(q[0], q[1]);
        let row1 = vzip1q_f32(q[2], q[3]);
        unsafe {
            vst1q_f32(a[(y * 2) * 8..].as_mut_ptr(), row0);
            vst1q_f32(a[(y * 2 + 1) * 8..].as_mut_ptr(), row1);
        }
    }
    for y in 0..4 {
        let q = dct2x2_inverse(
            unsafe { vld1q_f32(a[y * 8..].as_ptr()) },
            unsafe { vld1q_f32(a[y * 8 + 4..].as_ptr()) },
            unsafe { vld1q_f32(a[(y + 4) * 8..].as_ptr()) },
            unsafe { vld1q_f32(a[(y + 4) * 8 + 4..].as_ptr()) },
        );
        let row0lo = vzip1q_f32(q[0], q[1]);
        let row0hi = vzip2q_f32(q[0], q[1]);
        let row1lo = vzip1q_f32(q[2], q[3]);
        let row1hi = vzip2q_f32(q[2], q[3]);
        unsafe {
            vst1q_f32(output[(y * 2) * 8..].as_mut_ptr(), row0lo);
            vst1q_f32(output[(y * 2) * 8 + 4..].as_mut_ptr(), row0hi);
            vst1q_f32(output[(y * 2 + 1) * 8..].as_mut_ptr(), row1lo);
            vst1q_f32(output[(y * 2 + 1) * 8 + 4..].as_mut_ptr(), row1hi);
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct8x8_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut cols = load(input);
    dct1d_8_v(&mut cols);
    transpose_8x8(&mut cols);
    dct1d_8_v(&mut cols);
    scale_and_store(&cols, 1.0 / 64.0, output);
}
#[target_feature(enable = "neon")]
pub(crate) fn dct8x16_neon(input: DctInput<'_, 16, 8>, output: &mut [f32; 128]) {
    // 16-pt row DCT then 8-pt col DCT, 4-wide strips; scratch is hfreq-major.
    let mut scratch = MaybeUninit::<[f32; 128]>::uninit();
    let dst = scratch.as_mut_ptr() as *mut f32;
    for s in 0..2 {
        let mut c = [vdupq_n_f32(0.0); 16];
        for ct in 0..4 {
            let (a, b, cc, d) = transpose_4x4(
                load_strip(input.row(s * 4)[ct * 4..].as_ptr()),
                load_strip(input.row(s * 4 + 1)[ct * 4..].as_ptr()),
                load_strip(input.row(s * 4 + 2)[ct * 4..].as_ptr()),
                load_strip(input.row(s * 4 + 3)[ct * 4..].as_ptr()),
            );
            c[ct * 4] = a;
            c[ct * 4 + 1] = b;
            c[ct * 4 + 2] = cc;
            c[ct * 4 + 3] = d;
        }
        dct1d_16_s(&mut c);
        for u in 0..16 {
            unsafe { vst1q_f32(dst.add(u * 8 + s * 4), c[u]) };
        }
    }
    let scratch = unsafe { scratch.assume_init() };
    let scale = 1.0 / 128.0;
    for q in 0..4 {
        let mut c = [vdupq_n_f32(0.0); 8];
        for rt in 0..2 {
            let (a, b, cc, d) = transpose_4x4(
                load_strip(unsafe { scratch.get_unchecked((q * 4) * 8 + rt * 4..) }.as_ptr()),
                load_strip(unsafe { scratch.get_unchecked((q * 4 + 1) * 8 + rt * 4..) }.as_ptr()),
                load_strip(unsafe { scratch.get_unchecked((q * 4 + 2) * 8 + rt * 4..) }.as_ptr()),
                load_strip(unsafe { scratch.get_unchecked((q * 4 + 3) * 8 + rt * 4..) }.as_ptr()),
            );
            c[rt * 4] = a;
            c[rt * 4 + 1] = b;
            c[rt * 4 + 2] = cc;
            c[rt * 4 + 3] = d;
        }
        dct1d_8_s(&mut c);
        for v in 0..8 {
            let p = unsafe { output.get_unchecked_mut(v * 16 + q * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[v], scale)) };
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct16x8_neon(input: DctInput<'_, 8, 16>, output: &mut [f32; 128]) {
    // 16-pt col DCT then 8-pt row DCT, 4-wide strips; scratch is column-major.
    let mut scratch = MaybeUninit::<[f32; 128]>::uninit();
    let dst = scratch.as_mut_ptr() as *mut f32;
    for s in 0..2 {
        let mut c: [float32x4_t; 16] =
            std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_16_s(&mut c);
        for t in 0..4 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                unsafe { vst1q_f32(dst.add((s * 4 + j) * 16 + t * 4), *v) };
            }
        }
    }
    let scratch = unsafe { scratch.assume_init() };
    let scale = 1.0 / 128.0;
    for q in 0..4 {
        let mut c: [float32x4_t; 8] = std::array::from_fn(|col| {
            load_strip(unsafe { scratch.get_unchecked(col * 16 + q * 4..) }.as_ptr())
        });
        dct1d_8_s(&mut c);
        for u in 0..8 {
            let p = unsafe { output.get_unchecked_mut(u * 16 + q * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[u], scale)) };
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct16x16_neon(input: DctInput<'_, 16, 16>, output: &mut [f32; 256]) {
    // 4-wide strips keep the live set at 16 vectors instead of 64; scratch is
    // column-major (`[col * 16 + vfreq]`) for gather-free reloads.
    let mut scratch = MaybeUninit::<[f32; 256]>::uninit();
    let dst = scratch.as_mut_ptr() as *mut f32;
    for s in 0..4 {
        let mut c: [float32x4_t; 16] =
            std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_16_s(&mut c);
        for t in 0..4 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                unsafe { vst1q_f32(dst.add((s * 4 + j) * 16 + t * 4), *v) };
            }
        }
    }
    let scratch = unsafe { scratch.assume_init() };
    let scale = 1.0 / 256.0;
    for q in 0..4 {
        let mut c: [float32x4_t; 16] = std::array::from_fn(|col| {
            load_strip(unsafe { scratch.get_unchecked(col * 16 + q * 4..) }.as_ptr())
        });
        dct1d_16_s(&mut c);
        for u in 0..16 {
            let p = unsafe { output.get_unchecked_mut(u * 16 + q * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[u], scale)) };
        }
    }
}

// Single-width (4 cols/lane) 1D kernels for the tall 32-point transforms:
// a 4-wide strip halves the live set vs. the doubled vector, then goes to scratch.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn dct1d_4_s(c: &mut [float32x4_t; 4]) {
    let s2 = std::f32::consts::SQRT_2;
    let t0 = vaddq_f32(c[0], c[3]);
    let t1 = vaddq_f32(c[1], c[2]);
    let d2 = vmulq_n_f32(vsubq_f32(c[0], c[3]), WC4[0]);
    let d3 = vmulq_n_f32(vsubq_f32(c[1], c[2]), WC4[1]);
    let op = vaddq_f32(d2, d3);
    let om = vsubq_f32(d2, d3);
    c[0] = vaddq_f32(t0, t1);
    c[1] = vfmaq_n_f32(om, op, s2);
    c[2] = vsubq_f32(t0, t1);
    c[3] = om;
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn dct1d_8_s(c: &mut [float32x4_t; 8]) {
    let s2 = std::f32::consts::SQRT_2;
    let mut e = [
        vaddq_f32(c[0], c[7]),
        vaddq_f32(c[1], c[6]),
        vaddq_f32(c[2], c[5]),
        vaddq_f32(c[3], c[4]),
    ];
    dct1d_4_s(&mut e);
    let mut o = [
        vmulq_n_f32(vsubq_f32(c[0], c[7]), WC8[0]),
        vmulq_n_f32(vsubq_f32(c[1], c[6]), WC8[1]),
        vmulq_n_f32(vsubq_f32(c[2], c[5]), WC8[2]),
        vmulq_n_f32(vsubq_f32(c[3], c[4]), WC8[3]),
    ];
    dct1d_4_s(&mut o);
    o[0] = vfmaq_n_f32(o[1], o[0], s2);
    o[1] = vaddq_f32(o[1], o[2]);
    o[2] = vaddq_f32(o[2], o[3]);
    c[0] = e[0];
    c[1] = o[0];
    c[2] = e[1];
    c[3] = o[1];
    c[4] = e[2];
    c[5] = o[2];
    c[6] = e[3];
    c[7] = o[3];
}

#[inline]
#[target_feature(enable = "neon")]
fn dct1d_16_s(c: &mut [float32x4_t; 16]) {
    let s2 = std::f32::consts::SQRT_2;
    let mut e = [c[0]; 8];
    let mut o = [c[0]; 8];
    for i in 0..8 {
        e[i] = vaddq_f32(c[i], c[15 - i]);
        o[i] = vmulq_n_f32(vsubq_f32(c[i], c[15 - i]), WC16[i]);
    }
    dct1d_8_s(&mut e);
    dct1d_8_s(&mut o);
    o[0] = vfmaq_n_f32(o[1], o[0], s2);
    for i in 1..7 {
        o[i] = vaddq_f32(o[i], o[i + 1]);
    }
    for i in 0..8 {
        c[2 * i] = e[i];
        c[2 * i + 1] = o[i];
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dct1d_32_s(c: &mut [float32x4_t; 32]) {
    let s2 = std::f32::consts::SQRT_2;
    let mut e = [c[0]; 16];
    let mut o = [c[0]; 16];
    for i in 0..16 {
        e[i] = vaddq_f32(c[i], c[31 - i]);
        o[i] = vmulq_n_f32(vsubq_f32(c[i], c[31 - i]), WC32[i]);
    }
    dct1d_16_s(&mut e);
    dct1d_16_s(&mut o);
    o[0] = vfmaq_n_f32(o[1], o[0], s2);
    for i in 1..15 {
        o[i] = vaddq_f32(o[i], o[i + 1]);
    }
    for i in 0..16 {
        c[2 * i] = e[i];
        c[2 * i + 1] = o[i];
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_strip(ptr: *const f32) -> float32x4_t {
    unsafe { vld1q_f32(ptr) }
}

// ---------------------------------------------------------------------------
// Inverse (synthesis) strip butterflies — 4-lane vector versions of the scalar
// `inv_dct1d_N`, each an exact reverse of the forward `dct1d_N_s`. Composed into
// SIMD 2D IDCTs for the reconstruction-based transform selector.

const IS2: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[inline]
#[target_feature(enable = "neon")]
fn idct2_s(a: float32x4_t, b: float32x4_t) -> (float32x4_t, float32x4_t) {
    (
        vmulq_n_f32(vaddq_f32(a, b), 0.5),
        vmulq_n_f32(vsubq_f32(a, b), 0.5),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct1d_4_s(c: &mut [float32x4_t; 4]) {
    let (t0, t1, mut t2, t3) = (c[0], c[2], c[1], c[3]);
    t2 = vmulq_n_f32(vsubq_f32(t2, t3), IS2);
    let (n2, n3) = idct2_s(t2, t3);
    let a2 = vmulq_n_f32(n2, INV_WC4[0]);
    let a3 = vmulq_n_f32(n3, INV_WC4[1]);
    let (m0, m1) = idct2_s(t0, t1);
    c[0] = vmulq_n_f32(vaddq_f32(m0, a2), 0.5);
    c[3] = vmulq_n_f32(vsubq_f32(m0, a2), 0.5);
    c[1] = vmulq_n_f32(vaddq_f32(m1, a3), 0.5);
    c[2] = vmulq_n_f32(vsubq_f32(m1, a3), 0.5);
}

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct1d_8_s(c: &mut [float32x4_t; 8]) {
    let mut t = [c[0]; 8];
    for i in 0..4 {
        t[i] = c[2 * i];
        t[4 + i] = c[2 * i + 1];
    }
    t[6] = vsubq_f32(t[6], t[7]);
    t[5] = vsubq_f32(t[5], t[6]);
    t[4] = vmulq_n_f32(vsubq_f32(t[4], t[5]), IS2);
    let mut o: [float32x4_t; 4] = [t[4], t[5], t[6], t[7]];
    inv_dct1d_4_s(&mut o);
    for i in 0..4 {
        o[i] = vmulq_n_f32(o[i], INV_WC8[i]);
    }
    let mut e: [float32x4_t; 4] = [t[0], t[1], t[2], t[3]];
    inv_dct1d_4_s(&mut e);
    for i in 0..4 {
        c[i] = vmulq_n_f32(vaddq_f32(e[i], o[i]), 0.5);
        c[7 - i] = vmulq_n_f32(vsubq_f32(e[i], o[i]), 0.5);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct1d_16_s(c: &mut [float32x4_t; 16]) {
    let mut t = [c[0]; 16];
    for i in 0..8 {
        t[i] = c[2 * i];
        t[8 + i] = c[2 * i + 1];
    }
    for i in (9..=14).rev() {
        t[i] = vsubq_f32(t[i], t[i + 1]);
    }
    t[8] = vmulq_n_f32(vsubq_f32(t[8], t[9]), IS2);
    let mut o: [float32x4_t; 8] = std::array::from_fn(|i| t[8 + i]);
    inv_dct1d_8_s(&mut o);
    for i in 0..8 {
        o[i] = vmulq_n_f32(o[i], INV_WC16[i]);
    }
    let mut e: [float32x4_t; 8] = std::array::from_fn(|i| t[i]);
    inv_dct1d_8_s(&mut e);
    for i in 0..8 {
        c[i] = vmulq_n_f32(vaddq_f32(e[i], o[i]), 0.5);
        c[15 - i] = vmulq_n_f32(vsubq_f32(e[i], o[i]), 0.5);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct1d_32_s(c: &mut [float32x4_t; 32]) {
    let mut t = [c[0]; 32];
    for i in 0..16 {
        t[i] = c[2 * i];
        t[16 + i] = c[2 * i + 1];
    }
    for i in (17..=30).rev() {
        t[i] = vsubq_f32(t[i], t[i + 1]);
    }
    t[16] = vmulq_n_f32(vsubq_f32(t[16], t[17]), IS2);
    let mut o: [float32x4_t; 16] = std::array::from_fn(|i| t[16 + i]);
    inv_dct1d_16_s(&mut o);
    for i in 0..16 {
        o[i] = vmulq_n_f32(o[i], INV_WC32[i]);
    }
    let mut e: [float32x4_t; 16] = std::array::from_fn(|i| t[i]);
    inv_dct1d_16_s(&mut e);
    for i in 0..16 {
        c[i] = vmulq_n_f32(vaddq_f32(e[i], o[i]), 0.5);
        c[31 - i] = vmulq_n_f32(vsubq_f32(e[i], o[i]), 0.5);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct1d_64_s(c: &mut [float32x4_t; 64]) {
    let mut t = [c[0]; 64];
    for i in 0..32 {
        t[i] = c[2 * i];
        t[32 + i] = c[2 * i + 1];
    }
    for i in (33..=62).rev() {
        t[i] = vsubq_f32(t[i], t[i + 1]);
    }
    t[32] = vmulq_n_f32(vsubq_f32(t[32], t[33]), std::f32::consts::FRAC_1_SQRT_2);
    let mut odd: [float32x4_t; 32] = std::array::from_fn(|i| t[32 + i]);
    inv_dct1d_32_s(&mut odd);
    for i in 0..32 {
        odd[i] = vmulq_n_f32(odd[i], INV_WC64[i]);
    }
    let mut even: [float32x4_t; 32] = std::array::from_fn(|i| t[i]);
    inv_dct1d_32_s(&mut even);
    for i in 0..32 {
        c[i] = vmulq_n_f32(vaddq_f32(even[i], odd[i]), 0.5);
        c[63 - i] = vmulq_n_f32(vsubq_f32(even[i], odd[i]), 0.5);
    }
}

macro_rules! horizontal_idct_pass {
    ($tmp:expr, $out:expr, $h:literal, $w:literal, $inv_h:path) => {{
        for y4 in (0..$h).step_by(4) {
            let mut c = [vdupq_n_f32(0.0); $w];
            for u4 in (0..$w).step_by(4) {
                let (a, b, cc, d) = transpose_4x4(
                    load_strip(unsafe { $tmp.get_unchecked(y4 * $w + u4..) }.as_ptr()),
                    load_strip(unsafe { $tmp.get_unchecked((y4 + 1) * $w + u4..) }.as_ptr()),
                    load_strip(unsafe { $tmp.get_unchecked((y4 + 2) * $w + u4..) }.as_ptr()),
                    load_strip(unsafe { $tmp.get_unchecked((y4 + 3) * $w + u4..) }.as_ptr()),
                );
                c[u4] = a;
                c[u4 + 1] = b;
                c[u4 + 2] = cc;
                c[u4 + 3] = d;
            }
            $inv_h(&mut c);
            for x4 in (0..$w).step_by(4) {
                let (r0, r1, r2, r3) = transpose_4x4(c[x4], c[x4 + 1], c[x4 + 2], c[x4 + 3]);
                unsafe {
                    vst1q_f32($out.as_mut_ptr().add(y4 * $w + x4), r0);
                    vst1q_f32($out.as_mut_ptr().add((y4 + 1) * $w + x4), r1);
                    vst1q_f32($out.as_mut_ptr().add((y4 + 2) * $w + x4), r2);
                    vst1q_f32($out.as_mut_ptr().add((y4 + 3) * $w + x4), r3);
                }
            }
        }
    }};
}

/// IDCT for a natural coefficient layout `[vfreq * W + hfreq]`.
macro_rules! inv_dct_natural_neon {
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path) => {
        #[target_feature(enable = "neon")]
        pub(crate) fn $name(coeff: DctInput<'_, $w, $h>, out: &mut [f32; $n]) {
            let mut tmp_u = MaybeUninit::<[f32; $n]>::uninit();
            let tmp_ptr = tmp_u.as_mut_ptr() as *mut f32;
            for u4 in (0..$w).step_by(4) {
                let mut c: [float32x4_t; $h] = std::array::from_fn(|v| {
                    vmulq_n_f32(load_strip(coeff.row(v)[u4..].as_ptr()), $n as f32)
                });
                $inv_v(&mut c);
                for y in 0..$h {
                    unsafe { vst1q_f32(tmp_ptr.add(y * $w + u4), c[y]) };
                }
            }
            let tmp = unsafe { tmp_u.assume_init() };
            horizontal_idct_pass!(tmp, out, $h, $w, $inv_h);
        }
    };
}

/// IDCT for a transposed coefficient layout `[hfreq * H + vfreq]`.
macro_rules! inv_dct_transposed_neon {
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path) => {
        #[target_feature(enable = "neon")]
        pub(crate) fn $name(coeff: DctInput<'_, $h, $w>, out: &mut [f32; $n]) {
            let mut tmp_u = MaybeUninit::<[f32; $n]>::uninit();
            let tmp_ptr = tmp_u.as_mut_ptr() as *mut f32;
            for u4 in (0..$w).step_by(4) {
                let mut c = [vdupq_n_f32(0.0); $h];
                for v4 in (0..$h).step_by(4) {
                    let (a, b, cc, d) = transpose_4x4(
                        load_strip(coeff.row(u4)[v4..].as_ptr()),
                        load_strip(coeff.row(u4 + 1)[v4..].as_ptr()),
                        load_strip(coeff.row(u4 + 2)[v4..].as_ptr()),
                        load_strip(coeff.row(u4 + 3)[v4..].as_ptr()),
                    );
                    c[v4] = vmulq_n_f32(a, $n as f32);
                    c[v4 + 1] = vmulq_n_f32(b, $n as f32);
                    c[v4 + 2] = vmulq_n_f32(cc, $n as f32);
                    c[v4 + 3] = vmulq_n_f32(d, $n as f32);
                }
                $inv_v(&mut c);
                for y in 0..$h {
                    unsafe { vst1q_f32(tmp_ptr.add(y * $w + u4), c[y]) };
                }
            }
            let tmp = unsafe { tmp_u.assume_init() };
            horizontal_idct_pass!(tmp, out, $h, $w, $inv_h);
        }
    };
}

inv_dct_transposed_neon!(inv_dct8x8_neon, 64, 8, 8, inv_dct1d_8_s, inv_dct1d_8_s);
inv_dct_transposed_neon!(
    inv_dct16x16_neon,
    256,
    16,
    16,
    inv_dct1d_16_s,
    inv_dct1d_16_s
);
inv_dct_transposed_neon!(
    inv_dct32x32_neon,
    1024,
    32,
    32,
    inv_dct1d_32_s,
    inv_dct1d_32_s
);
inv_dct_natural_neon!(inv_dct8x16_neon, 128, 8, 16, inv_dct1d_8_s, inv_dct1d_16_s);
inv_dct_transposed_neon!(inv_dct16x8_neon, 128, 16, 8, inv_dct1d_16_s, inv_dct1d_8_s);
inv_dct_natural_neon!(
    inv_dct16x32_neon,
    512,
    16,
    32,
    inv_dct1d_16_s,
    inv_dct1d_32_s
);
inv_dct_transposed_neon!(
    inv_dct32x16_neon,
    512,
    32,
    16,
    inv_dct1d_32_s,
    inv_dct1d_16_s
);
inv_dct_transposed_neon!(
    inv_dct64x64_neon,
    4096,
    64,
    64,
    inv_dct1d_64_s,
    inv_dct1d_64_s
);
inv_dct_transposed_neon!(
    inv_dct64x32_neon,
    2048,
    64,
    32,
    inv_dct1d_64_s,
    inv_dct1d_32_s
);
inv_dct_natural_neon!(
    inv_dct32x64_neon,
    2048,
    32,
    64,
    inv_dct1d_32_s,
    inv_dct1d_64_s
);

#[target_feature(enable = "neon")]
pub(crate) fn dct32x32_neon(input: DctInput<'_, 32, 32>, output: &mut [f32; 1024]) {
    // 4-wide strips; column pass writes transposed scratch (`[col * 32 + vfreq]`)
    // for gather-free contiguous reloads in the row pass.
    let mut colt = MaybeUninit::<[f32; 1024]>::uninit();
    let dst = colt.as_mut_ptr() as *mut f32;
    for s in 0..8 {
        let mut c: [float32x4_t; 32] =
            std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_32_s(&mut c);
        for t in 0..8 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                unsafe { vst1q_f32(dst.add((s * 4 + j) * 32 + t * 4), *v) };
            }
        }
    }
    let colt = unsafe { colt.assume_init() };
    let scale = 1.0 / 1024.0;
    for q in 0..8 {
        let mut c: [float32x4_t; 32] = std::array::from_fn(|col| {
            load_strip(unsafe { colt.get_unchecked(col * 32 + q * 4..) }.as_ptr())
        });
        dct1d_32_s(&mut c);
        for u in 0..32 {
            let p = unsafe { output.get_unchecked_mut(u * 32 + q * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[u], scale)) };
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dct1d_64_s(c: &mut [float32x4_t; 64]) {
    // Split-radix decomposition from pxdct. Each four-value symmetry group is
    // overwritten in place, so this needs no 32-vector even/odd temporaries.
    for i in 0..16 {
        let bottom = c[i];
        let top = c[63 - i];
        let half_bottom = c[31 - i];
        let half_top = c[32 + i];
        let lower = vsubq_f32(bottom, top);
        let upper = vsubq_f32(half_bottom, half_top);

        c[i] = vaddq_f32(bottom, top);
        c[31 - i] = vaddq_f32(half_bottom, half_top);
        c[32 + i] = vfmaq_n_f32(
            vmulq_n_f32(lower, DCT64_SPLIT_COS[i]),
            upper,
            DCT64_SPLIT_SIN[i],
        );
        let sin = vfmsq_n_f32(
            vmulq_n_f32(upper, DCT64_SPLIT_COS[i]),
            lower,
            DCT64_SPLIT_SIN[i],
        );
        c[63 - i] = if i & 1 == 0 { sin } else { vnegq_f32(sin) };
    }

    dct1d_32_s(<&mut [float32x4_t; 32]>::try_from(&mut c[..32]).unwrap());
    dct1d_16_s(<&mut [float32x4_t; 16]>::try_from(&mut c[32..48]).unwrap());
    dct1d_16_s(<&mut [float32x4_t; 16]>::try_from(&mut c[48..]).unwrap());

    for i in 1..16 {
        let cos = c[32 + i];
        let sin = if i & 1 == 0 {
            vnegq_f32(c[64 - i])
        } else {
            c[64 - i]
        };
        c[32 + i] = vaddq_f32(cos, sin);
        c[64 - i] = vsubq_f32(cos, sin);
    }
    // jixel's DCT convention scales every AC coefficient by sqrt(2), while
    // pxdct's split-radix butterflies leave the DC of each inner DCT unscaled.
    c[32] = vmulq_n_f32(c[32], std::f32::consts::SQRT_2);
    c[48] = vmulq_n_f32(c[48], -std::f32::consts::SQRT_2);
}

#[inline]
#[target_feature(enable = "neon")]
fn dct64_store_transposed4(
    dst: *mut f32,
    column: usize,
    frequency: usize,
    values: [float32x4_t; 4],
) {
    let (a, b, c, d) = transpose_4x4(values[0], values[1], values[2], values[3]);
    for (lane, value) in [a, b, c, d].iter().enumerate() {
        unsafe { vst1q_f32(dst.add((column + lane) * 64 + frequency), *value) };
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dct64_store4(
    dst: *mut f32,
    stride: usize,
    frequency: usize,
    values: [float32x4_t; 4],
    scale: f32,
) {
    for (offset, value) in values.iter().enumerate() {
        unsafe {
            vst1q_f32(
                dst.add((frequency + offset) * stride),
                vmulq_n_f32(*value, scale),
            )
        };
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct64x64_neon(input: DctInput<'_, 64, 64>, output: &mut [f32; 4096]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 4096]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for strip in 0..16 {
        let mut c: [float32x4_t; 64] =
            std::array::from_fn(|row| load_strip(input.row(row)[strip * 4..].as_ptr()));
        dct1d_64_s(&mut c);
        dct64_store_transposed4(dst, strip * 4, 0, [c[0], c[32], c[1], c[33]]);
        for group in 1..15 {
            dct64_store_transposed4(
                dst,
                strip * 4,
                group * 4,
                [c[group * 2], c[64 - group], c[group * 2 + 1], c[33 + group]],
            )
        }
        dct64_store_transposed4(dst, strip * 4, 60, [c[30], c[49], c[31], c[48]]);
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = 1.0 / 4096.0;
    for strip in 0..16 {
        let mut c: [float32x4_t; 64] = std::array::from_fn(|column| {
            load_strip(unsafe { scratch.get_unchecked(column * 64 + strip * 4..) }.as_ptr())
        });
        dct1d_64_s(&mut c);
        let dst = unsafe { output.as_mut_ptr().add(strip * 4) };
        dct64_store4(dst, 64, 0, [c[0], c[32], c[1], c[33]], scale);
        for group in 1..15 {
            dct64_store4(
                dst,
                64,
                group * 4,
                [c[group * 2], c[64 - group], c[group * 2 + 1], c[33 + group]],
                scale,
            )
        }
        dct64_store4(dst, 64, 60, [c[30], c[49], c[31], c[48]], scale);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct64x32_neon(input: DctInput<'_, 32, 64>, output: &mut [f32; 2048]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 2048]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for strip in 0..8 {
        let mut c: [float32x4_t; 64] =
            std::array::from_fn(|row| load_strip(input.row(row)[strip * 4..].as_ptr()));
        dct1d_64_s(&mut c);
        dct64_store_transposed4(dst, strip * 4, 0, [c[0], c[32], c[1], c[33]]);
        for group in 1..15 {
            dct64_store_transposed4(
                dst,
                strip * 4,
                group * 4,
                [c[group * 2], c[64 - group], c[group * 2 + 1], c[33 + group]],
            )
        }
        dct64_store_transposed4(dst, strip * 4, 60, [c[30], c[49], c[31], c[48]]);
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = 1.0 / 2048.0;
    for strip in 0..16 {
        let mut c: [float32x4_t; 32] = std::array::from_fn(|column| {
            load_strip(unsafe { scratch.get_unchecked(column * 64 + strip * 4..) }.as_ptr())
        });
        dct1d_32_s(&mut c);
        for u in 0..32 {
            let p = unsafe { output.get_unchecked_mut(u * 64 + strip * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[u], scale)) };
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct32x64_neon(input: DctInput<'_, 64, 32>, output: &mut [f32; 2048]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 2048]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for row_strip in 0..8 {
        let mut c = [vdupq_n_f32(0.0); 64];
        for column_tile in 0..16 {
            let (a, b, cc, d) = transpose_4x4(
                load_strip(input.row(row_strip * 4)[column_tile * 4..].as_ptr()),
                load_strip(input.row(row_strip * 4 + 1)[column_tile * 4..].as_ptr()),
                load_strip(input.row(row_strip * 4 + 2)[column_tile * 4..].as_ptr()),
                load_strip(input.row(row_strip * 4 + 3)[column_tile * 4..].as_ptr()),
            );
            c[column_tile * 4] = a;
            c[column_tile * 4 + 1] = b;
            c[column_tile * 4 + 2] = cc;
            c[column_tile * 4 + 3] = d;
        }
        dct1d_64_s(&mut c);
        unsafe {
            dct64_store4(
                dst.add(row_strip * 4),
                32,
                0,
                [c[0], c[32], c[1], c[33]],
                1.0,
            )
        };
        for group in 1..15 {
            unsafe {
                dct64_store4(
                    dst.add(row_strip * 4),
                    32,
                    group * 4,
                    [c[group * 2], c[64 - group], c[group * 2 + 1], c[33 + group]],
                    1.0,
                )
            };
        }
        unsafe {
            dct64_store4(
                dst.add(row_strip * 4),
                32,
                60,
                [c[30], c[49], c[31], c[48]],
                1.0,
            )
        };
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = 1.0 / 2048.0;
    for column_strip in 0..16 {
        let mut c = [vdupq_n_f32(0.0); 32];
        for row_tile in 0..8 {
            let (a, b, cc, d) = transpose_4x4(
                load_strip(
                    unsafe { scratch.get_unchecked((column_strip * 4) * 32 + row_tile * 4..) }
                        .as_ptr(),
                ),
                load_strip(
                    unsafe { scratch.get_unchecked((column_strip * 4 + 1) * 32 + row_tile * 4..) }
                        .as_ptr(),
                ),
                load_strip(
                    unsafe { scratch.get_unchecked((column_strip * 4 + 2) * 32 + row_tile * 4..) }
                        .as_ptr(),
                ),
                load_strip(
                    unsafe { scratch.get_unchecked((column_strip * 4 + 3) * 32 + row_tile * 4..) }
                        .as_ptr(),
                ),
            );
            c[row_tile * 4] = a;
            c[row_tile * 4 + 1] = b;
            c[row_tile * 4 + 2] = cc;
            c[row_tile * 4 + 3] = d;
        }
        dct1d_32_s(&mut c);
        for v in 0..32 {
            let p = unsafe { output.get_unchecked_mut(v * 64 + column_strip * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[v], scale)) };
        }
    }
}

#[target_feature(enable = "neon")]
fn dct1d_4_q(c: &mut [float32x4_t; 4]) {
    let t0 = vaddq_f32(c[0], c[3]);
    let t1 = vaddq_f32(c[1], c[2]);
    let sum = vaddq_f32(t0, t1);
    let diff = vsubq_f32(t0, t1);
    let t2 = vmulq_n_f32(vsubq_f32(c[0], c[3]), WC4[0]);
    let t3 = vmulq_n_f32(vsubq_f32(c[1], c[2]), WC4[1]);
    let t2p = vaddq_f32(t2, t3);
    let t3p = vsubq_f32(t2, t3);
    let t2pp = vfmaq_n_f32(t3p, t2p, std::f32::consts::SQRT_2);
    c[0] = sum;
    c[1] = t2pp;
    c[2] = diff;
    c[3] = t3p;
}

#[target_feature(enable = "neon")]
pub(crate) fn dct4x4_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    // Gather q[r*4+c].lane[k] = input[(qy*4+r)*8 + (qx*4+c)], k = qy*2+qx.
    let mut q = [vdupq_n_f32(0.0); 16];
    for r in 0..4 {
        for col in 0..4 {
            let lanes = [
                input.row(r)[col],         // k=0 (qy0,qx0)
                input.row(r)[4 + col],     // k=1 (qy0,qx1)
                input.row(4 + r)[col],     // k=2 (qy1,qx0)
                input.row(4 + r)[4 + col], // k=3 (qy1,qx1)
            ];
            q[r * 4 + col] = unsafe { vld1q_f32(lanes.as_ptr()) };
        }
    }
    // Row DCT.
    for r in 0..4 {
        let mut row = [q[r * 4], q[r * 4 + 1], q[r * 4 + 2], q[r * 4 + 3]];
        dct1d_4_q(&mut row);
        for col in 0..4 {
            q[r * 4 + col] = row[col];
        }
    }
    // Column DCT (×1/16) → d[x*4+i] = colDCT freq i of column x.
    let mut d = [[0.0f32; 4]; 16];
    for col in 0..4 {
        let mut cc = [q[col], q[4 + col], q[8 + col], q[12 + col]];
        dct1d_4_q(&mut cc);
        for i in 0..4 {
            unsafe { vst1q_f32(d[col * 4 + i].as_mut_ptr(), vmulq_n_f32(cc[i], 1.0 / 16.0)) };
        }
    }
    // Scatter d[iy*4+ix].lane[k] → output[(qy+iy*2)*8 + (qx+ix*2)].
    for iy in 0..4 {
        for ix in 0..4 {
            let dd = &d[iy * 4 + ix];
            for k in 0..4 {
                let qy = k / 2;
                let qx = k % 2;
                output[(qy + iy * 2) * 8 + (qx + ix * 2)] = dd[k];
            }
        }
    }
    // 2×2 Hadamard on the four sub-DCs.
    let b00 = output[0];
    let b01 = output[1];
    let b10 = output[8];
    let b11 = output[9];
    output[0] = (b00 + b01 + b10 + b11) * 0.25;
    output[1] = (b00 + b01 - b10 - b11) * 0.25;
    output[8] = (b00 - b01 + b10 - b11) * 0.25;
    output[9] = (b00 - b01 - b10 + b11) * 0.25;
}

#[target_feature(enable = "neon")]
pub(crate) fn dct4x8_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let rows = load(input);
    let mut top: [NeonDoubledVector; 4] = [rows[0], rows[1], rows[2], rows[3]];
    let mut bot: [NeonDoubledVector; 4] = [rows[4], rows[5], rows[6], rows[7]];
    dct1d_4_v(&mut top);
    dct1d_4_v(&mut bot);
    let mut r: [NeonDoubledVector; 8] = [
        top[0], top[1], top[2], top[3], bot[0], bot[1], bot[2], bot[3],
    ];
    transpose_8x8(&mut r);
    dct1d_8_v(&mut r);
    transpose_8x8(&mut r);

    let mut buf_u = MaybeUninit::<[f32; 64]>::uninit();
    scale_and_store_uninit(&r, 1.0 / 32.0, &mut buf_u);
    let buf = unsafe { buf_u.assume_init() };
    for k in 0..8 {
        let vf = k % 4;
        let half = k / 4;
        for hf in 0..8 {
            output[(half + vf * 2) * 8 + hf] = buf[k * 8 + hf];
        }
    }
    let b0 = output[0];
    let b1 = output[8];
    output[0] = (b0 + b1) * 0.5;
    output[8] = (b0 - b1) * 0.5;
}

#[target_feature(enable = "neon")]
pub(crate) fn dct8x4_neon(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut rows = load(input);
    dct1d_8_v(&mut rows);
    transpose_8x8(&mut rows);
    let mut left: [NeonDoubledVector; 4] = [rows[0], rows[1], rows[2], rows[3]];
    let mut right: [NeonDoubledVector; 4] = [rows[4], rows[5], rows[6], rows[7]];
    dct1d_4_v(&mut left);
    dct1d_4_v(&mut right);

    let combo: [NeonDoubledVector; 8] = [
        left[0], left[1], left[2], left[3], right[0], right[1], right[2], right[3],
    ];
    let mut buf_u = MaybeUninit::<[f32; 64]>::uninit();
    scale_and_store_uninit(&combo, 1.0 / 32.0, &mut buf_u);
    let buf = unsafe { buf_u.assume_init() };
    for hf in 0..4 {
        for vf in 0..8 {
            output[(hf * 2) * 8 + vf] = buf[hf * 8 + vf];
            output[(1 + hf * 2) * 8 + vf] = buf[(4 + hf) * 8 + vf];
        }
    }
    let b0 = output[0];
    let b1 = output[8];
    output[0] = (b0 + b1) * 0.5;
    output[8] = (b0 - b1) * 0.5;
}

#[target_feature(enable = "neon")]
pub(crate) fn dct32x16_neon(input: DctInput<'_, 16, 32>, output: &mut [f32; 512]) {
    // Column pass then row pass, 4-wide strips; scratch is column-major
    // (`[col * 32 + vfreq]`) so the row pass reloads contiguously.
    let mut scratch = MaybeUninit::<[f32; 512]>::uninit();
    let dst = scratch.as_mut_ptr() as *mut f32;
    for s in 0..4 {
        let mut c: [float32x4_t; 32] =
            std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_32_s(&mut c);
        for t in 0..8 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                unsafe { vst1q_f32(dst.add((s * 4 + j) * 32 + t * 4), *v) };
            }
        }
    }
    let scratch = unsafe { scratch.assume_init() };
    let scale = 1.0 / 512.0;
    for q in 0..8 {
        let mut c: [float32x4_t; 16] = std::array::from_fn(|col| {
            load_strip(unsafe { scratch.get_unchecked(col * 32 + q * 4..) }.as_ptr())
        });
        dct1d_16_s(&mut c);
        for u in 0..16 {
            let p = unsafe { output.get_unchecked_mut(u * 32 + q * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[u], scale)) };
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct16x32_neon(input: DctInput<'_, 32, 16>, output: &mut [f32; 512]) {
    // Row pass (32-pt) then column pass (16-pt), 4-wide strips; scratch is
    // hfreq-major (`[hfreq * 16 + row]`).
    let mut scratch = MaybeUninit::<[f32; 512]>::uninit();
    let dst = scratch.as_mut_ptr() as *mut f32;
    for s in 0..4 {
        let mut c = [vdupq_n_f32(0.0); 32];
        for ct in 0..8 {
            let (a, b, cc, d) = transpose_4x4(
                load_strip(input.row(s * 4)[ct * 4..].as_ptr()),
                load_strip(input.row(s * 4 + 1)[ct * 4..].as_ptr()),
                load_strip(input.row(s * 4 + 2)[ct * 4..].as_ptr()),
                load_strip(input.row(s * 4 + 3)[ct * 4..].as_ptr()),
            );
            c[ct * 4] = a;
            c[ct * 4 + 1] = b;
            c[ct * 4 + 2] = cc;
            c[ct * 4 + 3] = d;
        }
        dct1d_32_s(&mut c);
        for u in 0..32 {
            unsafe { vst1q_f32(dst.add(u * 16 + s * 4), c[u]) };
        }
    }
    let scratch = unsafe { scratch.assume_init() };
    let scale = 1.0 / 512.0;
    for q in 0..8 {
        let mut c = [vdupq_n_f32(0.0); 16];
        for rt in 0..4 {
            let (a, b, cc, d) = transpose_4x4(
                load_strip(unsafe { scratch.get_unchecked((q * 4) * 16 + rt * 4..) }.as_ptr()),
                load_strip(unsafe { scratch.get_unchecked((q * 4 + 1) * 16 + rt * 4..) }.as_ptr()),
                load_strip(unsafe { scratch.get_unchecked((q * 4 + 2) * 16 + rt * 4..) }.as_ptr()),
                load_strip(unsafe { scratch.get_unchecked((q * 4 + 3) * 16 + rt * 4..) }.as_ptr()),
            );
            c[rt * 4] = a;
            c[rt * 4 + 1] = b;
            c[rt * 4 + 2] = cc;
            c[rt * 4 + 3] = d;
        }
        dct1d_16_s(&mut c);
        for v in 0..16 {
            let p = unsafe { output.get_unchecked_mut(v * 32 + q * 4..) };
            unsafe { vst1q_f32(p.as_mut_ptr(), vmulq_n_f32(c[v], scale)) };
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dc_idct4_neon(v: float32x4_t) -> float32x4_t {
    let v0 = vdupq_laneq_f32::<0>(v);
    let v1 = vdupq_laneq_f32::<1>(v);
    let v2 = vdupq_laneq_f32::<2>(v);
    let v3 = vdupq_laneq_f32::<3>(v);

    let e0 = vaddq_f32(v0, v2);
    let e1 = vsubq_f32(v0, v2);
    let t2 = vmulq_n_f32(v1, std::f32::consts::SQRT_2);
    let t3 = vaddq_f32(v3, v1);
    let o0 = vaddq_f32(t2, t3);
    let o1 = vsubq_f32(t2, t3);

    let even01 = vzip1q_f32(e0, e1);
    let odd01 = vzip1q_f32(o0, o1);
    let even = vcombine_f32(vget_low_f32(even01), vget_high_f32(vrev64q_f32(even01)));
    let odd = vcombine_f32(vget_low_f32(odd01), vget_high_f32(vrev64q_f32(odd01)));
    let weights = load_strip([WC4[0], WC4[1], -WC4[1], -WC4[0]].as_ptr());
    vfmaq_f32(even, odd, weights)
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_from_dct32x32_neon(coeffs: &[f32; 1024], dc: &mut [f32; 16]) {
    let resample = load_strip(RESAMPLE_SCALE_32_TO_4.as_ptr());
    let mut rows: [float32x4_t; 4] = std::array::from_fn(|a| {
        let coeff = load_strip(coeffs[a * 32..].as_ptr());
        let scaled_a = vmulq_n_f32(coeff, RESAMPLE_SCALE_32_TO_4[a]);
        dc_idct4_neon(vmulq_f32(scaled_a, resample))
    });
    let (r0, r1, r2, r3) = transpose_4x4(rows[0], rows[1], rows[2], rows[3]);
    rows = [r0, r1, r2, r3];
    for (bb, row) in rows.iter().enumerate() {
        unsafe { vst1q_f32(dc[bb * 4..].as_mut_ptr(), dc_idct4_neon(*row)) };
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dc_from_dct_rect_neon(coeffs: &[f32; 512]) -> (float32x4_t, float32x4_t) {
    let resample = load_strip(RESAMPLE_SCALE_32_TO_4.as_ptr());
    let row0 = dc_idct4_neon(vmulq_f32(load_strip(coeffs.as_ptr()), resample));
    let row1 = dc_idct4_neon(vmulq_f32(
        vmulq_n_f32(load_strip(coeffs[32..].as_ptr()), RESAMPLE_SCALE_16_TO_2[1]),
        resample,
    ));
    (vaddq_f32(row0, row1), vsubq_f32(row0, row1))
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_from_dct32x16_neon(coeffs: &[f32; 512], dc: &mut [f32; 8]) {
    let (sum, diff) = dc_from_dct_rect_neon(coeffs);
    unsafe {
        vst1q_f32(dc.as_mut_ptr(), vzip1q_f32(sum, diff));
        vst1q_f32(dc[4..].as_mut_ptr(), vzip2q_f32(sum, diff));
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_from_dct16x32_neon(coeffs: &[f32; 512], dc: &mut [f32; 8]) {
    let (sum, diff) = dc_from_dct_rect_neon(coeffs);
    unsafe {
        vst1q_f32(dc.as_mut_ptr(), sum);
        vst1q_f32(dc[4..].as_mut_ptr(), diff);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_from_dct64x64_neon(coeffs: &[f32; 4096], dc: &mut [f32; 64]) {
    let resample_lo = load_strip(RESAMPLE_SCALE_64_TO_8.as_ptr());
    let resample_hi = load_strip(RESAMPLE_SCALE_64_TO_8[4..].as_ptr());
    let mut lo: [float32x4_t; 8] = std::array::from_fn(|y| {
        vmulq_n_f32(
            vmulq_f32(load_strip(coeffs[y * 64..].as_ptr()), resample_lo),
            64.0 * RESAMPLE_SCALE_64_TO_8[y],
        )
    });
    let mut hi: [float32x4_t; 8] = std::array::from_fn(|y| {
        vmulq_n_f32(
            vmulq_f32(load_strip(coeffs[y * 64 + 4..].as_ptr()), resample_hi),
            64.0 * RESAMPLE_SCALE_64_TO_8[y],
        )
    });
    inv_dct1d_8_s(&mut lo);
    inv_dct1d_8_s(&mut hi);

    for y4 in (0..8).step_by(4) {
        let (x0, x1, x2, x3) = transpose_4x4(lo[y4], lo[y4 + 1], lo[y4 + 2], lo[y4 + 3]);
        let (x4, x5, x6, x7) = transpose_4x4(hi[y4], hi[y4 + 1], hi[y4 + 2], hi[y4 + 3]);
        let mut horizontal = [x0, x1, x2, x3, x4, x5, x6, x7];
        inv_dct1d_8_s(&mut horizontal);
        for (x, row) in horizontal.iter().enumerate() {
            unsafe { vst1q_f32(dc[x * 8 + y4..].as_mut_ptr(), *row) };
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dc_from_dct64x32_normalized_neon(coeffs: &[f32; 2048]) -> [float32x4_t; 8] {
    let resample_lo = load_strip(RESAMPLE_SCALE_64_TO_8.as_ptr());
    let resample_hi = load_strip(RESAMPLE_SCALE_64_TO_8[4..].as_ptr());
    let mut lo: [float32x4_t; 4] = std::array::from_fn(|y| {
        vmulq_n_f32(
            vmulq_f32(load_strip(coeffs[y * 64..].as_ptr()), resample_lo),
            32.0 * RESAMPLE_SCALE_32_TO_4[y],
        )
    });
    let mut hi: [float32x4_t; 4] = std::array::from_fn(|y| {
        vmulq_n_f32(
            vmulq_f32(load_strip(coeffs[y * 64 + 4..].as_ptr()), resample_hi),
            32.0 * RESAMPLE_SCALE_32_TO_4[y],
        )
    });
    inv_dct1d_4_s(&mut lo);
    inv_dct1d_4_s(&mut hi);
    let (x0, x1, x2, x3) = transpose_4x4(lo[0], lo[1], lo[2], lo[3]);
    let (x4, x5, x6, x7) = transpose_4x4(hi[0], hi[1], hi[2], hi[3]);
    let mut horizontal = [x0, x1, x2, x3, x4, x5, x6, x7];
    inv_dct1d_8_s(&mut horizontal);
    horizontal
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_from_dct64x32_neon(coeffs: &[f32; 2048], dc: &mut [f32; 32]) {
    let rows = dc_from_dct64x32_normalized_neon(coeffs);
    for (y, row) in rows.iter().enumerate() {
        unsafe { vst1q_f32(dc[y * 4..].as_mut_ptr(), *row) };
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_from_dct32x64_neon(coeffs: &[f32; 2048], dc: &mut [f32; 32]) {
    let rows = dc_from_dct64x32_normalized_neon(coeffs);
    let (lo0, lo1, lo2, lo3) = transpose_4x4(rows[0], rows[1], rows[2], rows[3]);
    let (hi0, hi1, hi2, hi3) = transpose_4x4(rows[4], rows[5], rows[6], rows[7]);
    unsafe {
        vst1q_f32(dc.as_mut_ptr(), lo0);
        vst1q_f32(dc[4..].as_mut_ptr(), hi0);
        vst1q_f32(dc[8..].as_mut_ptr(), lo1);
        vst1q_f32(dc[12..].as_mut_ptr(), hi1);
        vst1q_f32(dc[16..].as_mut_ptr(), lo2);
        vst1q_f32(dc[20..].as_mut_ptr(), hi2);
        vst1q_f32(dc[24..].as_mut_ptr(), lo3);
        vst1q_f32(dc[28..].as_mut_ptr(), hi3);
    }
}

#[cfg(test)]
mod neon_dct_tests {
    use crate::dct::{DctInput, dct8x8_scalar, dct8x16_scalar, dct16x8_scalar};

    const ATOL: f32 = 1e-4;

    fn assert_close(neon: &[f32], scalar: &[f32], label: &str) {
        assert_eq!(neon.len(), scalar.len(), "{label}: length mismatch");
        let base_tolerance = if label.starts_with("idct") && label.contains("64") {
            5e-4
        } else {
            ATOL
        };
        let mut max_err: f32 = 0.0;
        let mut max_tolerance: f32 = base_tolerance;
        let mut worst = 0usize;
        for (i, (n, s)) in neon.iter().zip(scalar.iter()).enumerate() {
            let e = (n - s).abs();
            if e > max_err {
                max_err = e;
                max_tolerance = base_tolerance + 16.0 * f32::EPSILON * s.abs();
                worst = i;
            }
        }
        assert!(
            max_err < max_tolerance,
            "{label}: max error {max_err:.2e} at index {worst} \
             (neon={:.6}, scalar={:.6})",
            neon[worst],
            scalar[worst]
        );
    }

    fn rng_f32(seed: u64) -> f32 {
        // xorshift64
        let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        // map to [-1, 1]
        let u = (x >> 41) as f32; // 23-bit mantissa
        u / (1u32 << 23) as f32 * 2.0 - 1.0
    }

    fn fill<const N: usize>(seed: u64) -> [f32; N] {
        let mut buf = [0.0f32; N];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = rng_f32(seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005)));
        }
        buf
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dc_from_dct_neon_matches_scalar() {
        use crate::dct::{
            dc_from_dct16x32, dc_from_dct32x16, dc_from_dct32x32, dc_from_dct32x64,
            dc_from_dct64x32, dc_from_dct64x64,
        };
        use crate::neon::{
            dc_from_dct16x32_neon, dc_from_dct32x16_neon, dc_from_dct32x32_neon,
            dc_from_dct32x64_neon, dc_from_dct64x32_neon, dc_from_dct64x64_neon,
        };

        for seed in 0u64..32 {
            let coeffs = fill::<1024>(seed.wrapping_add(0xdc32_0032));
            let mut got = [0.0f32; 16];
            let mut want = [0.0f32; 16];
            unsafe { dc_from_dct32x32_neon(&coeffs, &mut got) };
            dc_from_dct32x32(&coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct32x32 seed={seed}"));

            let coeffs: &[f32; 512] = coeffs.first_chunk::<512>().unwrap();
            let mut got = [0.0f32; 8];
            let mut want = [0.0f32; 8];
            unsafe { dc_from_dct32x16_neon(coeffs, &mut got) };
            dc_from_dct32x16(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct32x16 seed={seed}"));

            unsafe { dc_from_dct16x32_neon(coeffs, &mut got) };
            dc_from_dct16x32(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct16x32 seed={seed}"));

            let coeffs = fill::<4096>(seed.wrapping_add(0xdc64_0064));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dc_from_dct64x64_neon(&coeffs, &mut got) };
            dc_from_dct64x64(&coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct64x64 seed={seed}"));

            let coeffs = coeffs.first_chunk::<2048>().unwrap();
            let mut got = [0.0f32; 32];
            let mut want = [0.0f32; 32];
            unsafe { dc_from_dct64x32_neon(coeffs, &mut got) };
            dc_from_dct64x32(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct64x32 seed={seed}"));

            unsafe { dc_from_dct32x64_neon(coeffs, &mut got) };
            dc_from_dct32x64(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct32x64 seed={seed}"));
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn assert_inverse_matches_scalar<const N: usize, const W: usize, const H: usize>(
        scalar: for<'a> fn(DctInput<'a, W, H>, &mut [f32; N]),
        neon: for<'a> unsafe fn(DctInput<'a, W, H>, &mut [f32; N]),
        label: &str,
    ) {
        let mut cases = Vec::with_capacity(35);
        cases.push([0.0f32; N]);

        let mut dc = [0.0f32; N];
        dc[0] = 0.75;
        cases.push(dc);

        let mut alternating = [0.0f32; N];
        for (i, v) in alternating.iter_mut().enumerate() {
            *v = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
        }
        cases.push(alternating);

        for seed in 0..32 {
            cases.push(fill(0x1dc7_0000 + seed));
        }

        for (case, input) in cases.iter().enumerate() {
            let mut got = [0.0f32; N];
            let mut want = [0.0f32; N];
            let stride = W + 3;
            let mut strided = vec![f32::NAN; H * stride];
            for y in 0..H {
                strided[y * stride..y * stride + W].copy_from_slice(&input[y * W..y * W + W]);
            }
            unsafe { neon(DctInput::new(&strided, stride), &mut got) };
            scalar(DctInput::from_flat(input), &mut want);
            assert_close(&got, &want, &format!("{label} case={case}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_vs_scalar_random() {
        use crate::neon::dct8x8_neon;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed);
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct8x8_neon(DctInput::from_flat(&input), &mut got) };
            dct8x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_identity8x8_neon_matches_scalar() {
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(0x1de0_0000 + seed);
            let mut strided = [f32::NAN; 88];
            for y in 0..8 {
                strided[y * 11..y * 11 + 8].copy_from_slice(&input[y * 8..y * 8 + 8]);
            }
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { crate::neon::identity8x8_neon(DctInput::new(&strided, 11), &mut got) };
            crate::dct::identity8x8(DctInput::from_flat(&input), &mut want);
            assert_close(&got, &want, &format!("identity8x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_inverse_identity8x8_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_identity8x8,
            crate::neon::inv_identity8x8_neon,
            "inverse_identity8x8",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct2x2_8x8_neon_matches_scalar() {
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(0xd2d2_0000 + seed);
            let mut strided = [f32::NAN; 88];
            for y in 0..8 {
                strided[y * 11..y * 11 + 8].copy_from_slice(&input[y * 8..y * 8 + 8]);
            }
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { crate::neon::dct2x2_8x8_neon(DctInput::new(&strided, 11), &mut got) };
            crate::dct::dct2x2_8x8(DctInput::from_flat(&input), &mut want);
            assert_close(&got, &want, &format!("dct2x2_8x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_inverse_dct2x2_8x8_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct2x2_8x8,
            crate::neon::inv_dct2x2_8x8_neon,
            "inverse_dct2x2_8x8",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_dc_only() {
        // All-constant input: only DC coefficient should be non-zero.
        use crate::neon::dct8x8_neon;
        let input = [0.5f32; 64];
        let mut got = [0.0f32; 64];
        let mut want = [0.0f32; 64];
        unsafe { dct8x8_neon(DctInput::from_flat(&input), &mut got) };
        dct8x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x8 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_zero() {
        use crate::neon::dct8x8_neon;
        let input = [0.0f32; 64];
        let mut got = [0.0f32; 64];
        let mut want = [0.0f32; 64];
        unsafe { dct8x8_neon(DctInput::from_flat(&input), &mut got) };
        dct8x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x8 zero");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_linearity() {
        // DCT is linear: DCT(a + b) == DCT(a) + DCT(b)
        use crate::neon::dct8x8_neon;
        let a: [f32; 64] = fill(100);
        let b: [f32; 64] = fill(200);
        let mut sum = [0.0f32; 64];
        for i in 0..64 {
            sum[i] = a[i] + b[i];
        }

        let mut da = [0.0f32; 64];
        let mut db = [0.0f32; 64];
        let mut dsum = [0.0f32; 64];
        unsafe {
            dct8x8_neon(DctInput::from_flat(&a), &mut da);
            dct8x8_neon(DctInput::from_flat(&b), &mut db);
            dct8x8_neon(DctInput::from_flat(&sum), &mut dsum);
        }
        let expected: Vec<f32> = (0..64).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct8x8 linearity");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_basis_vectors() {
        // Feed each of the 64 basis vectors (single 1.0) and compare to scalar.
        use crate::neon::dct8x8_neon;
        for k in 0..64 {
            let mut input = [0.0f32; 64];
            input[k] = 1.0;
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct8x8_neon(DctInput::from_flat(&input), &mut got) };
            dct8x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x8 basis[{k}]"));
        }
    }

    // ── dct8x16 ───────────────────────────────────────────────────────────────

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x16_neon_vs_scalar_random() {
        use crate::neon::dct8x16_neon;
        for seed in 0u64..32 {
            let input: [f32; 128] = fill(seed.wrapping_add(0xdead));
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            unsafe { dct8x16_neon(DctInput::from_flat(&input), &mut got) };
            dct8x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x16 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x16_neon_dc_only() {
        use crate::neon::dct8x16_neon;
        let input = [0.5f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        unsafe { dct8x16_neon(DctInput::from_flat(&input), &mut got) };
        dct8x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x16 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x16_neon_zero() {
        use crate::neon::dct8x16_neon;
        let input = [0.0f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        unsafe { dct8x16_neon(DctInput::from_flat(&input), &mut got) };
        dct8x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x16 zero");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x16_neon_basis_vectors() {
        use crate::neon::dct8x16_neon;
        for k in 0..128 {
            let mut input = [0.0f32; 128];
            input[k] = 1.0;
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            unsafe { dct8x16_neon(DctInput::from_flat(&input), &mut got) };
            dct8x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x16 basis[{k}]"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x16_neon_linearity() {
        use crate::neon::dct8x16_neon;
        let a: [f32; 128] = fill(300);
        let b: [f32; 128] = fill(400);
        let mut sum = [0.0f32; 128];
        for i in 0..128 {
            sum[i] = a[i] + b[i];
        }
        let mut da = [0.0f32; 128];
        let mut db = [0.0f32; 128];
        let mut dsum = [0.0f32; 128];
        unsafe {
            dct8x16_neon(DctInput::from_flat(&a), &mut da);
            dct8x16_neon(DctInput::from_flat(&b), &mut db);
            dct8x16_neon(DctInput::from_flat(&sum), &mut dsum);
        }
        let expected: Vec<f32> = (0..128).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct8x16 linearity");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_vs_scalar_random() {
        use crate::neon::dct16x8_neon;
        for seed in 0u64..32 {
            let input: [f32; 128] = fill(seed.wrapping_add(0xbeef));
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            unsafe { dct16x8_neon(DctInput::from_flat(&input), &mut got) };
            dct16x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct4x8_neon_vs_scalar_random() {
        use crate::dct::dct4x8_scalar;
        use crate::neon::dct4x8_neon;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x4a8));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct4x8_neon(DctInput::from_flat(&input), &mut got) };
            dct4x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct4x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x4_neon_vs_scalar_random() {
        use crate::dct::dct8x4_scalar;
        use crate::neon::dct8x4_neon;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x8a4));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct8x4_neon(DctInput::from_flat(&input), &mut got) };
            dct8x4_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x4 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_dc_only() {
        use crate::neon::dct16x8_neon;
        let input = [0.5f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        unsafe { dct16x8_neon(DctInput::from_flat(&input), &mut got) };
        dct16x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x8 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_zero() {
        use crate::neon::dct16x8_neon;
        let input = [0.0f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        unsafe { dct16x8_neon(DctInput::from_flat(&input), &mut got) };
        dct16x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x8 zero");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_basis_vectors() {
        use crate::neon::dct16x8_neon;
        for k in 0..128 {
            let mut input = [0.0f32; 128];
            input[k] = 1.0;
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            unsafe { dct16x8_neon(DctInput::from_flat(&input), &mut got) };
            dct16x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x8 basis[{k}]"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_linearity() {
        use crate::neon::dct16x8_neon;
        let a: [f32; 128] = fill(500);
        let b: [f32; 128] = fill(600);
        let mut sum = [0.0f32; 128];
        for i in 0..128 {
            sum[i] = a[i] + b[i];
        }
        let mut da = [0.0f32; 128];
        let mut db = [0.0f32; 128];
        let mut dsum = [0.0f32; 128];
        unsafe {
            dct16x8_neon(DctInput::from_flat(&a), &mut da);
            dct16x8_neon(DctInput::from_flat(&b), &mut db);
            dct16x8_neon(DctInput::from_flat(&sum), &mut dsum);
        }
        let expected: Vec<f32> = (0..128).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct16x8 linearity");
    }

    // ── Cross-shape consistency ────────────────────────────────────────────────
    // A 16x8 block is the transpose of an 8x16 block.
    // DCT(A^T)[u][v] relates to DCT(A)[v][u] by the separable 2-D DCT symmetry.
    // We test a weaker property: both kernels produce the same DC coefficient
    // for an all-constant input (DC is rotation-invariant).

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dc_coefficient_constant_input() {
        use crate::neon::{dct8x16_neon, dct16x8_neon};
        let input = [0.25f32; 128];

        let mut out8x16 = [0.0f32; 128];
        let mut out16x8 = [0.0f32; 128];
        unsafe {
            dct8x16_neon(DctInput::from_flat(&input), &mut out8x16);
            dct16x8_neon(DctInput::from_flat(&input), &mut out16x8);
        }
        // DC is at index 0 in both output layouts.
        assert!(
            (out8x16[0] - out16x8[0]).abs() < ATOL,
            "DC mismatch: 8x16={} 16x8={}",
            out8x16[0],
            out16x8[0]
        );
    }

    // ── Extreme values ────────────────────────────────────────────────────────

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_extreme_values() {
        use crate::neon::dct8x8_neon;
        // Alternating +1/-1 (highest-frequency input)
        let mut input = [0.0f32; 64];
        for i in 0..64 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 64];
        let mut want = [0.0f32; 64];
        unsafe { dct8x8_neon(DctInput::from_flat(&input), &mut got) };
        dct8x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x8 alternating +-1");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_extreme_values() {
        use crate::neon::dct16x8_neon;
        let mut input = [0.0f32; 128];
        for i in 0..128 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        unsafe { dct16x8_neon(DctInput::from_flat(&input), &mut got) };
        dct16x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x8 alternating +-1");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x16_neon_extreme_values() {
        use crate::neon::dct8x16_neon;
        let mut input = [0.0f32; 128];
        for i in 0..128 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        unsafe { dct8x16_neon(DctInput::from_flat(&input), &mut got) };
        dct8x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x16 alternating +-1");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x16_neon_vs_scalar_random() {
        use crate::dct::dct16x16_scalar;
        use crate::neon::dct16x16_neon;
        for seed in 0u64..32 {
            let input: [f32; 256] = fill(seed.wrapping_add(0xf00d));
            let mut got = [0.0f32; 256];
            let mut want = [0.0f32; 256];
            unsafe { dct16x16_neon(DctInput::from_flat(&input), &mut got) };
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct32x32_neon_vs_scalar_random() {
        use crate::dct::dct32x32_scalar;
        use crate::neon::dct32x32_neon;
        for seed in 0u64..16 {
            let input: [f32; 1024] = fill(seed.wrapping_add(0x3232));
            let mut got = [0.0f32; 1024];
            let mut want = [0.0f32; 1024];
            unsafe { dct32x32_neon(DctInput::from_flat(&input), &mut got) };
            dct32x32_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct32x32 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct32x16_neon_vs_scalar_random() {
        use crate::dct::dct32x16_scalar;
        use crate::neon::dct32x16_neon;
        for seed in 0u64..16 {
            let input: [f32; 512] = fill(seed.wrapping_add(0x3216));
            let mut got = [0.0f32; 512];
            let mut want = [0.0f32; 512];
            unsafe { dct32x16_neon(DctInput::from_flat(&input), &mut got) };
            dct32x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct32x16 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x32_neon_vs_scalar_random() {
        use crate::dct::dct16x32_scalar;
        use crate::neon::dct16x32_neon;
        for seed in 0u64..16 {
            let input: [f32; 512] = fill(seed.wrapping_add(0x1632));
            let mut got = [0.0f32; 512];
            let mut want = [0.0f32; 512];
            unsafe { dct16x32_neon(DctInput::from_flat(&input), &mut got) };
            dct16x32_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x32 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct4x4_neon_vs_scalar_random() {
        use crate::dct::dct4x4_scalar;
        use crate::neon::dct4x4_neon;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x4a4));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct4x4_neon(DctInput::from_flat(&input), &mut got) };
            dct4x4_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct4x4 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x16_neon_dc_only() {
        use crate::dct::dct16x16_scalar;
        use crate::neon::dct16x16_neon;
        let input = [0.5f32; 256];
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        unsafe { dct16x16_neon(DctInput::from_flat(&input), &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x16_neon_zero() {
        use crate::dct::dct16x16_scalar;
        use crate::neon::dct16x16_neon;
        let input = [0.0f32; 256];
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        unsafe { dct16x16_neon(DctInput::from_flat(&input), &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 zero");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x16_neon_basis_vectors() {
        use crate::dct::dct16x16_scalar;
        use crate::neon::dct16x16_neon;
        for k in 0..256 {
            let mut input = [0.0f32; 256];
            input[k] = 1.0;
            let mut got = [0.0f32; 256];
            let mut want = [0.0f32; 256];
            unsafe { dct16x16_neon(DctInput::from_flat(&input), &mut got) };
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 basis[{k}]"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x16_neon_linearity() {
        use crate::neon::dct16x16_neon;
        let a: [f32; 256] = fill(700);
        let b: [f32; 256] = fill(800);
        let mut sum = [0.0f32; 256];
        for i in 0..256 {
            sum[i] = a[i] + b[i];
        }
        let mut da = [0.0f32; 256];
        let mut db = [0.0f32; 256];
        let mut dsum = [0.0f32; 256];
        unsafe {
            dct16x16_neon(DctInput::from_flat(&a), &mut da);
            dct16x16_neon(DctInput::from_flat(&b), &mut db);
            dct16x16_neon(DctInput::from_flat(&sum), &mut dsum);
        }
        let expected: Vec<f32> = (0..256).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct16x16 linearity");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x16_neon_extreme_values() {
        use crate::dct::dct16x16_scalar;
        use crate::neon::dct16x16_neon;
        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        unsafe { dct16x16_neon(DctInput::from_flat(&input), &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 alternating +-1");
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct8x8_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct8x8,
            crate::neon::inv_dct8x8_neon,
            "idct8x8",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct8x16_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct8x16,
            crate::neon::inv_dct8x16_neon,
            "idct8x16",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct16x8_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct16x8,
            crate::neon::inv_dct16x8_neon,
            "idct16x8",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct16x16_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct16x16,
            crate::neon::inv_dct16x16_neon,
            "idct16x16",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct16x32_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct16x32,
            crate::neon::inv_dct16x32_neon,
            "idct16x32",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct32x16_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct32x16,
            crate::neon::inv_dct32x16_neon,
            "idct32x16",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct32x32_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct32x32,
            crate::neon::inv_dct32x32_neon,
            "idct32x32",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct64x64_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct64x64,
            crate::neon::inv_dct64x64_neon,
            "idct64x64",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct64x32_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct64x32,
            crate::neon::inv_dct64x32_neon,
            "idct64x32",
        );
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_idct32x64_neon_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct32x64,
            crate::neon::inv_dct32x64_neon,
            "idct32x64",
        );
    }
}
