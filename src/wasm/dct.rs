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
    DCT64_SPLIT_COS, DCT64_SPLIT_SIN, DctInput, INV_WC4, INV_WC8, INV_WC16, INV_WC32,
    RESAMPLE_SCALE_32_TO_4, WC4, WC8, WC16, WC32,
};
#[cfg(feature = "extraslow")]
use crate::dct::{INV_WC64, RESAMPLE_SCALE_64_TO_8};
#[cfg(feature = "extraslow")]
use crate::dct128::{DCT128_SPLIT_COS, DCT128_SPLIT_SIN, INV_WC128, RESAMPLE_SCALE_128_TO_16};
use core::arch::wasm32::*;
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
struct WasmDoubledVector {
    lo: v128,
    hi: v128,
}

impl WasmDoubledVector {
    #[inline]
    #[target_feature(enable = "simd128")]
    fn add(self, rhs: WasmDoubledVector) -> WasmDoubledVector {
        WasmDoubledVector {
            lo: f32x4_add(self.lo, rhs.lo),
            hi: f32x4_add(self.hi, rhs.hi),
        }
    }
    #[inline]
    #[target_feature(enable = "simd128")]
    fn sub(self, rhs: WasmDoubledVector) -> WasmDoubledVector {
        WasmDoubledVector {
            lo: f32x4_sub(self.lo, rhs.lo),
            hi: f32x4_sub(self.hi, rhs.hi),
        }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    fn muls(self, s: f32) -> WasmDoubledVector {
        WasmDoubledVector {
            lo: f32x4_mul(self.lo, f32x4_splat(s)),
            hi: f32x4_mul(self.hi, f32x4_splat(s)),
        }
    }
    #[inline]
    #[target_feature(enable = "simd128")]
    fn fma(self, b: WasmDoubledVector, s: f32) -> WasmDoubledVector {
        WasmDoubledVector {
            lo: f32x4_add(self.lo, f32x4_mul(b.lo, f32x4_splat(s))),
            hi: f32x4_add(self.hi, f32x4_mul(b.hi, f32x4_splat(s))),
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct1d_4_v(c: &mut [WasmDoubledVector; 4]) {
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
#[target_feature(enable = "simd128")]
fn dct1d_8_v(c: &mut [WasmDoubledVector; 8]) {
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
#[target_feature(enable = "simd128")]
pub(super) fn transpose_4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
    let v0 = i32x4_shuffle::<0, 4, 2, 6>(r0, r1);
    let v1 = i32x4_shuffle::<1, 5, 3, 7>(r0, r1);
    let v2 = i32x4_shuffle::<0, 4, 2, 6>(r2, r3);
    let v3 = i32x4_shuffle::<1, 5, 3, 7>(r2, r3);
    let c0 = i32x4_shuffle::<0, 1, 4, 5>(v0, v2);
    let c1 = i32x4_shuffle::<0, 1, 4, 5>(v1, v3);
    let c2 = i32x4_shuffle::<2, 3, 6, 7>(v0, v2);
    let c3 = i32x4_shuffle::<2, 3, 6, 7>(v1, v3);
    (c0, c1, c2, c3)
}

#[inline]
#[target_feature(enable = "simd128")]
fn transpose_8x8(c: &mut [WasmDoubledVector; 8]) {
    let (a0, a1, a2, a3) = transpose_4x4(c[0].lo, c[1].lo, c[2].lo, c[3].lo);
    let (b0, b1, b2, b3) = transpose_4x4(c[0].hi, c[1].hi, c[2].hi, c[3].hi);
    let (cc0, cc1, cc2, cc3) = transpose_4x4(c[4].lo, c[5].lo, c[6].lo, c[7].lo);
    let (d0, d1, d2, d3) = transpose_4x4(c[4].hi, c[5].hi, c[6].hi, c[7].hi);

    c[0] = WasmDoubledVector { lo: a0, hi: cc0 };
    c[1] = WasmDoubledVector { lo: a1, hi: cc1 };
    c[2] = WasmDoubledVector { lo: a2, hi: cc2 };
    c[3] = WasmDoubledVector { lo: a3, hi: cc3 };
    c[4] = WasmDoubledVector { lo: b0, hi: d0 };
    c[5] = WasmDoubledVector { lo: b1, hi: d1 };
    c[6] = WasmDoubledVector { lo: b2, hi: d2 };
    c[7] = WasmDoubledVector { lo: b3, hi: d3 };
}

#[inline]
#[target_feature(enable = "simd128")]
fn load(input: DctInput<'_, 8, 8>) -> [WasmDoubledVector; 8] {
    let row = |y: usize| -> WasmDoubledVector {
        unsafe {
            let p = input.row(y);
            WasmDoubledVector {
                lo: v128_load(p.as_ptr().cast()),
                hi: v128_load(p[4..].as_ptr().cast()),
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
#[target_feature(enable = "simd128")]
fn scale_and_store(cols: &[WasmDoubledVector; 8], scale: f32, out: &mut [f32; 64]) {
    for (k, col) in cols.iter().enumerate() {
        unsafe {
            v128_store(
                out[k * 8..].as_mut_ptr().cast(),
                f32x4_mul(col.lo, f32x4_splat(scale)),
            );
            v128_store(
                out[k * 8 + 4..].as_mut_ptr().cast(),
                f32x4_mul(col.hi, f32x4_splat(scale)),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn hsum(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

#[target_feature(enable = "simd128")]
pub(crate) fn identity8x8_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut means = [0.0f32; 4];
    for qy in 0..2 {
        let anchor_row = input.row(qy * 4 + 1);
        let anchor_lo = f32x4_splat(anchor_row[1]);
        let anchor_hi = f32x4_splat(anchor_row[5]);
        let mut sum_lo = f32x4_splat(0.0);
        let mut sum_hi = f32x4_splat(0.0);
        for iy in 0..4 {
            let row = input.row(qy * 4 + iy);
            let pixels_lo = unsafe { v128_load(row.as_ptr().cast()) };
            let pixels_hi = unsafe { v128_load(row[4..].as_ptr().cast()) };
            sum_lo = f32x4_add(sum_lo, pixels_lo);
            sum_hi = f32x4_add(sum_hi, pixels_hi);
            let residual_lo = f32x4_sub(pixels_lo, anchor_lo);
            let residual_hi = f32x4_sub(pixels_hi, anchor_hi);
            let packed_lo = i32x4_shuffle::<0, 4, 1, 5>(residual_lo, residual_hi);
            let packed_hi = i32x4_shuffle::<2, 6, 3, 7>(residual_lo, residual_hi);
            let dst = &mut output[(qy + iy * 2) * 8..];
            unsafe {
                v128_store(dst.as_mut_ptr().cast(), packed_lo);
                v128_store(dst[4..].as_mut_ptr().cast(), packed_hi);
            }
        }
        means[qy * 2] = hsum(sum_lo) * (1.0 / 16.0);
        means[qy * 2 + 1] = hsum(sum_hi) * (1.0 / 16.0);
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

#[target_feature(enable = "simd128")]
pub(crate) fn inv_identity8x8_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
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
        let mut left = [f32x4_splat(0.0); 4];
        let mut right = [f32x4_splat(0.0); 4];
        for iy in 0..4 {
            let row = input.row(qy + iy * 2);
            let packed_lo = unsafe { v128_load(row.as_ptr().cast()) };
            let packed_hi = unsafe { v128_load(row[4..].as_ptr().cast()) };
            left[iy] = i32x4_shuffle::<0, 2, 4, 6>(packed_lo, packed_hi);
            right[iy] = i32x4_shuffle::<1, 3, 5, 7>(packed_lo, packed_hi);
        }
        left[0] = f32x4_replace_lane::<0>(left[0], input.row(qy + 2)[2]);
        right[0] = f32x4_replace_lane::<0>(right[0], input.row(qy + 2)[3]);
        left[1] = f32x4_replace_lane::<1>(left[1], 0.0);
        right[1] = f32x4_replace_lane::<1>(right[1], 0.0);
        let left_sum = f32x4_add(f32x4_add(left[0], left[1]), f32x4_add(left[2], left[3]));
        let right_sum = f32x4_add(f32x4_add(right[0], right[1]), f32x4_add(right[2], right[3]));
        let anchor_lo = f32x4_splat(means[qy * 2] - hsum(left_sum) * (1.0 / 16.0));
        let anchor_hi = f32x4_splat(means[qy * 2 + 1] - hsum(right_sum) * (1.0 / 16.0));
        for iy in 0..4 {
            let dst = &mut output[(qy * 4 + iy) * 8..];
            unsafe {
                v128_store(dst.as_mut_ptr().cast(), f32x4_add(left[iy], anchor_lo));
                v128_store(
                    dst[4..].as_mut_ptr().cast(),
                    f32x4_add(right[iy], anchor_hi),
                );
            }
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct2x2_forward(a: v128, b: v128, c: v128, d: v128) -> [v128; 4] {
    let scale = f32x4_splat(0.25);
    let ab = f32x4_add(a, b);
    let amb = f32x4_sub(a, b);
    [
        f32x4_mul(f32x4_add(f32x4_add(ab, c), d), scale),
        f32x4_mul(f32x4_sub(f32x4_sub(ab, c), d), scale),
        f32x4_mul(f32x4_sub(f32x4_add(amb, c), d), scale),
        f32x4_mul(f32x4_add(f32x4_sub(amb, c), d), scale),
    ]
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct2x2_inverse(a: v128, b: v128, c: v128, d: v128) -> [v128; 4] {
    let ab = f32x4_add(a, b);
    let amb = f32x4_sub(a, b);
    [
        f32x4_add(f32x4_add(ab, c), d),
        f32x4_sub(f32x4_sub(ab, c), d),
        f32x4_sub(f32x4_add(amb, c), d),
        f32x4_add(f32x4_sub(amb, c), d),
    ]
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct2x2_8x8_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    for y in 0..4 {
        let row0 = input.row(y * 2);
        let row1 = input.row(y * 2 + 1);
        let r0lo = unsafe { v128_load(row0.as_ptr().cast()) };
        let r0hi = unsafe { v128_load(row0[4..].as_ptr().cast()) };
        let r1lo = unsafe { v128_load(row1.as_ptr().cast()) };
        let r1hi = unsafe { v128_load(row1[4..].as_ptr().cast()) };
        let q = dct2x2_forward(
            i32x4_shuffle::<0, 2, 4, 6>(r0lo, r0hi),
            i32x4_shuffle::<1, 3, 5, 7>(r0lo, r0hi),
            i32x4_shuffle::<0, 2, 4, 6>(r1lo, r1hi),
            i32x4_shuffle::<1, 3, 5, 7>(r1lo, r1hi),
        );
        unsafe {
            v128_store(output[y * 8..].as_mut_ptr().cast(), q[0]);
            v128_store(output[y * 8 + 4..].as_mut_ptr().cast(), q[1]);
            v128_store(output[(y + 4) * 8..].as_mut_ptr().cast(), q[2]);
            v128_store(output[(y + 4) * 8 + 4..].as_mut_ptr().cast(), q[3]);
        }
    }
    let mut stage4 = [[f32x4_splat(0.0); 4]; 2];
    for (y, q) in stage4.iter_mut().enumerate() {
        let r0 = unsafe { v128_load(output[(y * 2) * 8..].as_ptr().cast()) };
        let r1 = unsafe { v128_load(output[(y * 2 + 1) * 8..].as_ptr().cast()) };
        *q = dct2x2_forward(
            i32x4_shuffle::<0, 2, 0, 2>(r0, r0),
            i32x4_shuffle::<1, 3, 1, 3>(r0, r0),
            i32x4_shuffle::<0, 2, 0, 2>(r1, r1),
            i32x4_shuffle::<1, 3, 1, 3>(r1, r1),
        );
    }
    for (y, q) in stage4.into_iter().enumerate() {
        output[y * 8] = f32x4_extract_lane::<0>(q[0]);
        output[y * 8 + 1] = f32x4_extract_lane::<1>(q[0]);
        output[y * 8 + 2] = f32x4_extract_lane::<0>(q[1]);
        output[y * 8 + 3] = f32x4_extract_lane::<1>(q[1]);
        output[(y + 2) * 8] = f32x4_extract_lane::<0>(q[2]);
        output[(y + 2) * 8 + 1] = f32x4_extract_lane::<1>(q[2]);
        output[(y + 2) * 8 + 2] = f32x4_extract_lane::<0>(q[3]);
        output[(y + 2) * 8 + 3] = f32x4_extract_lane::<1>(q[3]);
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

#[target_feature(enable = "simd128")]
pub(crate) fn inv_dct2x2_8x8_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut a = std::array::from_fn::<_, 64, _>(|i| input.row(i / 8)[i % 8]);
    let c00 = a[0];
    let c01 = a[1];
    let c10 = a[8];
    let c11 = a[9];
    a[0] = c00 + c01 + c10 + c11;
    a[1] = c00 + c01 - c10 - c11;
    a[8] = c00 - c01 + c10 - c11;
    a[9] = c00 - c01 - c10 + c11;
    let mut stage4 = [[f32x4_splat(0.0); 4]; 2];
    for (y, q) in stage4.iter_mut().enumerate() {
        *q = dct2x2_inverse(
            unsafe { v128_load(a[y * 8..].as_ptr().cast()) },
            unsafe { v128_load(a[y * 8 + 2..].as_ptr().cast()) },
            unsafe { v128_load(a[(y + 2) * 8..].as_ptr().cast()) },
            unsafe { v128_load(a[(y + 2) * 8 + 2..].as_ptr().cast()) },
        );
    }
    for (y, q) in stage4.into_iter().enumerate() {
        unsafe {
            v128_store(
                a[(y * 2) * 8..].as_mut_ptr().cast(),
                i32x4_shuffle::<0, 4, 1, 5>(q[0], q[1]),
            );
            v128_store(
                a[(y * 2 + 1) * 8..].as_mut_ptr().cast(),
                i32x4_shuffle::<0, 4, 1, 5>(q[2], q[3]),
            );
        }
    }
    for y in 0..4 {
        let q = dct2x2_inverse(
            unsafe { v128_load(a[y * 8..].as_ptr().cast()) },
            unsafe { v128_load(a[y * 8 + 4..].as_ptr().cast()) },
            unsafe { v128_load(a[(y + 4) * 8..].as_ptr().cast()) },
            unsafe { v128_load(a[(y + 4) * 8 + 4..].as_ptr().cast()) },
        );
        unsafe {
            v128_store(
                output[(y * 2) * 8..].as_mut_ptr().cast(),
                i32x4_shuffle::<0, 4, 1, 5>(q[0], q[1]),
            );
            v128_store(
                output[(y * 2) * 8 + 4..].as_mut_ptr().cast(),
                i32x4_shuffle::<2, 6, 3, 7>(q[0], q[1]),
            );
            v128_store(
                output[(y * 2 + 1) * 8..].as_mut_ptr().cast(),
                i32x4_shuffle::<0, 4, 1, 5>(q[2], q[3]),
            );
            v128_store(
                output[(y * 2 + 1) * 8 + 4..].as_mut_ptr().cast(),
                i32x4_shuffle::<2, 6, 3, 7>(q[2], q[3]),
            );
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct8x8_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut cols = load(input);
    dct1d_8_v(&mut cols);
    transpose_8x8(&mut cols);
    dct1d_8_v(&mut cols);
    scale_and_store(&cols, 1.0 / 64.0, output);
}
#[target_feature(enable = "simd128")]
pub(crate) fn dct8x16_wasm(input: DctInput<'_, 16, 8>, output: &mut [f32; 128]) {
    // 16-pt row DCT then 8-pt col DCT, 4-wide strips; scratch is hfreq-major.
    let mut scratch_uninit = MaybeUninit::<[f32; 128]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for s in 0..2 {
        let mut c = [f32x4_splat(0.0); 16];
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
            unsafe { v128_store(dst.add(u * 8 + s * 4).cast(), c[u]) };
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 128.0);
    for q in 0..4 {
        let mut c = [f32x4_splat(0.0); 8];
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
            unsafe { v128_store(p.as_mut_ptr() as *mut v128, f32x4_mul(c[v], scale)) };
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct16x8_wasm(input: DctInput<'_, 8, 16>, output: &mut [f32; 128]) {
    // 16-pt col DCT then 8-pt row DCT, 4-wide strips; scratch is column-major.
    let mut scratch_uninit = MaybeUninit::<[f32; 128]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for s in 0..2 {
        let mut c: [v128; 16] = std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_16_s(&mut c);
        for t in 0..4 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                let p = unsafe { dst.add((s * 4 + j) * 16 + t * 4) };
                unsafe { v128_store(p.cast(), *v) };
            }
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 128.0);
    for q in 0..4 {
        let mut c: [v128; 8] = std::array::from_fn(|col| {
            load_strip(unsafe { scratch.get_unchecked(col * 16 + q * 4..) }.as_ptr())
        });
        dct1d_8_s(&mut c);
        for u in 0..8 {
            let p = unsafe { output.get_unchecked_mut(u * 16 + q * 4..) };
            unsafe { v128_store(p.as_mut_ptr().cast(), f32x4_mul(c[u], scale)) };
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct16x16_wasm(input: DctInput<'_, 16, 16>, output: &mut [f32; 256]) {
    // 4-wide strips keep the live set at 16 v128 instead of 64; scratch is
    // column-major (`[col * 16 + vfreq]`) for gather-free reloads.
    let mut scratch_uninit = MaybeUninit::<[f32; 256]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for s in 0..4 {
        let mut c: [v128; 16] = std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_16_s(&mut c);
        for t in 0..4 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                let p = unsafe { dst.add((s * 4 + j) * 16 + t * 4) };
                unsafe { v128_store(p.cast(), *v) };
            }
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 256.0);
    for q in 0..4 {
        let mut c: [v128; 16] = std::array::from_fn(|col| {
            load_strip(unsafe { scratch.get_unchecked(col * 16 + q * 4..) }.as_ptr())
        });
        dct1d_16_s(&mut c);
        for u in 0..16 {
            let p = unsafe { output.get_unchecked_mut(u * 16 + q * 4..) };
            unsafe { v128_store(p.as_mut_ptr() as *mut v128, f32x4_mul(c[u], scale)) };
        }
    }
}

// Single-width (v128, 4 cols/lane) 1D kernels for the tall 32-point transforms:
// a 4-wide strip halves the live set vs. the doubled vector, then goes to scratch.
#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn dct1d_4_s(c: &mut [v128; 4]) {
    let t0 = f32x4_add(c[0], c[3]);
    let t1 = f32x4_add(c[1], c[2]);
    let s2 = f32x4_splat(std::f32::consts::SQRT_2);
    let d2 = f32x4_mul(f32x4_sub(c[0], c[3]), f32x4_splat(WC4[0]));
    let d3 = f32x4_mul(f32x4_sub(c[1], c[2]), f32x4_splat(WC4[1]));
    let op = f32x4_add(d2, d3);
    let om = f32x4_sub(d2, d3);
    c[0] = f32x4_add(t0, t1);
    c[1] = f32x4_add(f32x4_mul(op, s2), om);
    c[2] = f32x4_sub(t0, t1);
    c[3] = om;
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn dct1d_8_s(c: &mut [v128; 8]) {
    let s2 = f32x4_splat(std::f32::consts::SQRT_2);
    let mut e = [
        f32x4_add(c[0], c[7]),
        f32x4_add(c[1], c[6]),
        f32x4_add(c[2], c[5]),
        f32x4_add(c[3], c[4]),
    ];
    dct1d_4_s(&mut e);
    let mut o = [
        f32x4_mul(f32x4_sub(c[0], c[7]), f32x4_splat(WC8[0])),
        f32x4_mul(f32x4_sub(c[1], c[6]), f32x4_splat(WC8[1])),
        f32x4_mul(f32x4_sub(c[2], c[5]), f32x4_splat(WC8[2])),
        f32x4_mul(f32x4_sub(c[3], c[4]), f32x4_splat(WC8[3])),
    ];
    dct1d_4_s(&mut o);
    o[0] = f32x4_add(f32x4_mul(o[0], s2), o[1]);
    o[1] = f32x4_add(o[1], o[2]);
    o[2] = f32x4_add(o[2], o[3]);
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
#[target_feature(enable = "simd128")]
fn dct1d_16_s(c: &mut [v128; 16]) {
    let s2 = f32x4_splat(std::f32::consts::SQRT_2);
    let mut e = [f32x4_splat(0.0); 8];
    let mut o = [f32x4_splat(0.0); 8];
    for i in 0..8 {
        e[i] = f32x4_add(c[i], c[15 - i]);
        o[i] = f32x4_mul(f32x4_sub(c[i], c[15 - i]), f32x4_splat(WC16[i]));
    }
    dct1d_8_s(&mut e);
    dct1d_8_s(&mut o);
    o[0] = f32x4_add(f32x4_mul(o[0], s2), o[1]);
    for i in 1..7 {
        o[i] = f32x4_add(o[i], o[i + 1]);
    }
    for i in 0..8 {
        c[2 * i] = e[i];
        c[2 * i + 1] = o[i];
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct1d_32_s(c: &mut [v128; 32]) {
    let s2 = f32x4_splat(std::f32::consts::SQRT_2);
    let mut e = [f32x4_splat(0.0); 16];
    let mut o = [f32x4_splat(0.0); 16];
    for i in 0..16 {
        e[i] = f32x4_add(c[i], c[31 - i]);
        o[i] = f32x4_mul(f32x4_sub(c[i], c[31 - i]), f32x4_splat(WC32[i]));
    }
    dct1d_16_s(&mut e);
    dct1d_16_s(&mut o);
    o[0] = f32x4_add(f32x4_mul(o[0], s2), o[1]);
    for i in 1..15 {
        o[i] = f32x4_add(o[i], o[i + 1]);
    }
    for i in 0..16 {
        c[2 * i] = e[i];
        c[2 * i + 1] = o[i];
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_strip(ptr: *const f32) -> v128 {
    unsafe { v128_load(ptr as *const v128) }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct32x32_wasm(input: DctInput<'_, 32, 32>, output: &mut [f32; 1024]) {
    // Both passes over 4-wide strips; the column pass writes a transposed scratch
    // (`[col * 32 + vfreq]`) so the row pass reloads contiguously.
    let mut scratch_uninit = MaybeUninit::<[f32; 1024]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for s in 0..8 {
        let mut c: [v128; 32] = std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_32_s(&mut c);
        for t in 0..8 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                let p = unsafe { dst.add((s * 4 + j) * 32 + t * 4) };
                unsafe { v128_store(p.cast(), *v) };
            }
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 1024.0);
    for q in 0..8 {
        let mut c: [v128; 32] = std::array::from_fn(|col| {
            load_strip(unsafe { scratch.get_unchecked(col * 32 + q * 4..) }.as_ptr())
        });
        dct1d_32_s(&mut c);
        for u in 0..32 {
            let p = unsafe { output.get_unchecked_mut(u * 32 + q * 4..) };
            unsafe { v128_store(p.as_mut_ptr() as *mut v128, f32x4_mul(c[u], scale)) };
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct1d_64_s(c: &mut [v128; 64]) {
    for i in 0..16 {
        let bottom = c[i];
        let top = c[63 - i];
        let half_bottom = c[31 - i];
        let half_top = c[32 + i];
        let lower = f32x4_sub(bottom, top);
        let upper = f32x4_sub(half_bottom, half_top);
        let cos = f32x4_splat(DCT64_SPLIT_COS[i]);
        let sin = f32x4_splat(DCT64_SPLIT_SIN[i]);

        c[i] = f32x4_add(bottom, top);
        c[31 - i] = f32x4_add(half_bottom, half_top);
        c[32 + i] = f32x4_add(f32x4_mul(lower, cos), f32x4_mul(upper, sin));
        let sin_input = f32x4_sub(f32x4_mul(upper, cos), f32x4_mul(lower, sin));
        c[63 - i] = if i & 1 == 0 {
            sin_input
        } else {
            f32x4_neg(sin_input)
        };
    }

    dct1d_32_s(<&mut [v128; 32]>::try_from(&mut c[..32]).unwrap());
    dct1d_16_s(<&mut [v128; 16]>::try_from(&mut c[32..48]).unwrap());
    dct1d_16_s(<&mut [v128; 16]>::try_from(&mut c[48..]).unwrap());

    for i in 1..16 {
        let cos = c[32 + i];
        let sin = if i & 1 == 0 {
            f32x4_neg(c[64 - i])
        } else {
            c[64 - i]
        };
        c[32 + i] = f32x4_add(cos, sin);
        c[64 - i] = f32x4_sub(cos, sin);
    }
    c[32] = f32x4_mul(c[32], f32x4_splat(std::f32::consts::SQRT_2));
    c[48] = f32x4_mul(c[48], f32x4_splat(-std::f32::consts::SQRT_2));
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct64_store_transposed4(dst: *mut f32, column: usize, frequency: usize, values: [v128; 4]) {
    let (a, b, c, d) = transpose_4x4(values[0], values[1], values[2], values[3]);
    for (lane, value) in [a, b, c, d].iter().enumerate() {
        unsafe { v128_store(dst.add((column + lane) * 64 + frequency).cast(), *value) };
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct64_store4(dst: *mut f32, stride: usize, frequency: usize, values: [v128; 4], scale: v128) {
    for (offset, value) in values.iter().enumerate() {
        unsafe {
            v128_store(
                dst.add((frequency + offset) * stride).cast(),
                f32x4_mul(*value, scale),
            )
        };
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct64x64_wasm(input: DctInput<'_, 64, 64>, output: &mut [f32; 4096]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 4096]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for strip in 0..16 {
        let mut c: [v128; 64] =
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
    let scale = f32x4_splat(1.0 / 4096.0);
    for strip in 0..16 {
        let mut c: [v128; 64] = std::array::from_fn(|column| {
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

#[target_feature(enable = "simd128")]
pub(crate) fn dct64x32_wasm(input: DctInput<'_, 32, 64>, output: &mut [f32; 2048]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 2048]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for strip in 0..8 {
        let mut c: [v128; 64] =
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
    let scale = f32x4_splat(1.0 / 2048.0);
    for strip in 0..16 {
        let mut c: [v128; 32] = std::array::from_fn(|column| {
            load_strip(unsafe { scratch.get_unchecked(column * 64 + strip * 4..) }.as_ptr())
        });
        dct1d_32_s(&mut c);
        for u in 0..32 {
            let p = unsafe { output.get_unchecked_mut(u * 64 + strip * 4..) };
            unsafe { v128_store(p.as_mut_ptr().cast(), f32x4_mul(c[u], scale)) };
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct32x64_wasm(input: DctInput<'_, 64, 32>, output: &mut [f32; 2048]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 2048]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for row_strip in 0..8 {
        let mut c = [f32x4_splat(0.0); 64];
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
        let one = f32x4_splat(1.0);
        unsafe {
            dct64_store4(
                dst.add(row_strip * 4),
                32,
                0,
                [c[0], c[32], c[1], c[33]],
                one,
            )
        };
        for group in 1..15 {
            unsafe {
                dct64_store4(
                    dst.add(row_strip * 4),
                    32,
                    group * 4,
                    [c[group * 2], c[64 - group], c[group * 2 + 1], c[33 + group]],
                    one,
                )
            };
        }
        unsafe {
            dct64_store4(
                dst.add(row_strip * 4),
                32,
                60,
                [c[30], c[49], c[31], c[48]],
                one,
            )
        };
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 2048.0);
    for column_strip in 0..16 {
        let mut c = [f32x4_splat(0.0); 32];
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
            unsafe { v128_store(p.as_mut_ptr().cast(), f32x4_mul(c[v], scale)) };
        }
    }
}

#[target_feature(enable = "simd128")]
fn dct1d_4_q(c: &mut [v128; 4]) {
    let t0 = f32x4_add(c[0], c[3]);
    let t1 = f32x4_add(c[1], c[2]);
    let sum = f32x4_add(t0, t1);
    let diff = f32x4_sub(t0, t1);
    let t2 = f32x4_mul(f32x4_sub(c[0], c[3]), f32x4_splat(WC4[0]));
    let t3 = f32x4_mul(f32x4_sub(c[1], c[2]), f32x4_splat(WC4[1]));
    let t2p = f32x4_add(t2, t3);
    let t3p = f32x4_sub(t2, t3);
    let t2pp = f32x4_add(t3p, f32x4_mul(t2p, f32x4_splat(std::f32::consts::SQRT_2)));
    c[0] = sum;
    c[1] = t2pp;
    c[2] = diff;
    c[3] = t3p;
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct4x4_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    // Gather q[r*4+c].lane[k] = input[(qy*4+r)*8 + (qx*4+c)], k = qy*2+qx.
    let mut q = [f32x4_splat(0.0); 16];
    for r in 0..4 {
        for col in 0..4 {
            let lanes = [
                input.row(r)[col],         // k=0 (qy0,qx0)
                input.row(r)[4 + col],     // k=1 (qy0,qx1)
                input.row(4 + r)[col],     // k=2 (qy1,qx0)
                input.row(4 + r)[4 + col], // k=3 (qy1,qx1)
            ];
            q[r * 4 + col] = unsafe { v128_load(lanes.as_ptr().cast()) };
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
            unsafe {
                v128_store(
                    d[col * 4 + i].as_mut_ptr().cast(),
                    f32x4_mul(cc[i], f32x4_splat(1.0 / 16.0)),
                )
            };
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

#[target_feature(enable = "simd128")]
pub(crate) fn dct4x8_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let rows = load(input);
    let mut top: [WasmDoubledVector; 4] = [rows[0], rows[1], rows[2], rows[3]];
    let mut bot: [WasmDoubledVector; 4] = [rows[4], rows[5], rows[6], rows[7]];
    dct1d_4_v(&mut top);
    dct1d_4_v(&mut bot);
    let mut r: [WasmDoubledVector; 8] = [
        top[0], top[1], top[2], top[3], bot[0], bot[1], bot[2], bot[3],
    ];
    transpose_8x8(&mut r);
    dct1d_8_v(&mut r);
    transpose_8x8(&mut r);

    let mut buf = [0.0f32; 64];
    scale_and_store(&r, 1.0 / 32.0, &mut buf);
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

#[target_feature(enable = "simd128")]
pub(crate) fn dct8x4_wasm(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut rows = load(input);
    dct1d_8_v(&mut rows);
    transpose_8x8(&mut rows);
    let mut left: [WasmDoubledVector; 4] = [rows[0], rows[1], rows[2], rows[3]];
    let mut right: [WasmDoubledVector; 4] = [rows[4], rows[5], rows[6], rows[7]];
    dct1d_4_v(&mut left);
    dct1d_4_v(&mut right);

    let combo: [WasmDoubledVector; 8] = [
        left[0], left[1], left[2], left[3], right[0], right[1], right[2], right[3],
    ];
    let mut buf = [0.0f32; 64];
    scale_and_store(&combo, 1.0 / 32.0, &mut buf);
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

// Transpose the long-axis frequencies into tiles for the 8-point pass.
#[inline]
#[target_feature(enable = "simd128")]
fn dct32x8_long_pass_wasm(c: &mut [v128; 32], tiles: &mut [[v128; 8]; 8], strip: usize) {
    dct1d_32_s(c);
    for (t, tile) in tiles.iter_mut().enumerate() {
        let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
        tile[strip * 4..strip * 4 + 4].copy_from_slice(&[a, b, cc, d]);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn dct32x8_short_pass_wasm(tiles: &mut [[v128; 8]; 8], output: &mut [f32; 256]) {
    for (t, tile) in tiles.iter_mut().enumerate() {
        dct1d_8_s(tile);
        for (u, value) in tile.iter().enumerate() {
            unsafe {
                v128_store(
                    output[u * 32 + t * 4..].as_mut_ptr().cast(),
                    f32x4_mul(*value, f32x4_splat(1.0 / 256.0)),
                );
            }
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct32x8_wasm(input: DctInput<'_, 8, 32>, output: &mut [f32; 256]) {
    let mut tiles = [[f32x4_splat(0.0); 8]; 8];
    for strip in 0..2 {
        let mut c: [v128; 32] =
            std::array::from_fn(|r| load_strip(input.row(r)[strip * 4..].as_ptr()));
        dct32x8_long_pass_wasm(&mut c, &mut tiles, strip);
    }
    dct32x8_short_pass_wasm(&mut tiles, output);
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct8x32_wasm(input: DctInput<'_, 32, 8>, output: &mut [f32; 256]) {
    let mut tiles = [[f32x4_splat(0.0); 8]; 8];
    for strip in 0..2 {
        let mut c = [f32x4_splat(0.0); 32];
        for t in 0..8 {
            let (a, b, cc, d) = transpose_4x4(
                load_strip(input.row(strip * 4)[t * 4..].as_ptr()),
                load_strip(input.row(strip * 4 + 1)[t * 4..].as_ptr()),
                load_strip(input.row(strip * 4 + 2)[t * 4..].as_ptr()),
                load_strip(input.row(strip * 4 + 3)[t * 4..].as_ptr()),
            );
            c[t * 4..t * 4 + 4].copy_from_slice(&[a, b, cc, d]);
        }
        dct32x8_long_pass_wasm(&mut c, &mut tiles, strip);
    }
    dct32x8_short_pass_wasm(&mut tiles, output);
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct32x16_wasm(input: DctInput<'_, 16, 32>, output: &mut [f32; 512]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 512]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for s in 0..4 {
        let mut c: [v128; 32] = std::array::from_fn(|r| load_strip(input.row(r)[s * 4..].as_ptr()));
        dct1d_32_s(&mut c);
        for t in 0..8 {
            let (a, b, cc, d) = transpose_4x4(c[t * 4], c[t * 4 + 1], c[t * 4 + 2], c[t * 4 + 3]);
            let tile = [a, b, cc, d];
            for (j, v) in tile.iter().enumerate() {
                let p = unsafe { dst.add((s * 4 + j) * 32 + t * 4) };
                unsafe { v128_store(p.cast(), *v) };
            }
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 512.0);
    for q in 0..8 {
        let mut c: [v128; 16] = std::array::from_fn(|col| {
            load_strip(unsafe { scratch.get_unchecked(col * 32 + q * 4..) }.as_ptr())
        });
        dct1d_16_s(&mut c);
        for u in 0..16 {
            let p = unsafe { output.get_unchecked_mut(u * 32 + q * 4..) };
            unsafe { v128_store(p.as_mut_ptr() as *mut v128, f32x4_mul(c[u], scale)) };
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dct16x32_wasm(input: DctInput<'_, 32, 16>, output: &mut [f32; 512]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 512]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for s in 0..4 {
        let mut c = [f32x4_splat(0.0); 32];
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
            let p = unsafe { dst.add(u * 16 + s * 4) };
            unsafe { v128_store(p.cast(), c[u]) };
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 512.0);
    for q in 0..8 {
        let mut c = [f32x4_splat(0.0); 16];
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
            unsafe { v128_store(p.as_mut_ptr() as *mut v128, f32x4_mul(c[v], scale)) };
        }
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn dc_from_dct8x32_wasm(coeffs: &[f32; 256], dc: &mut [f32; 4]) {
    let resample = load_strip(RESAMPLE_SCALE_32_TO_4.as_ptr());
    let v = f32x4_mul(load_strip(coeffs.as_ptr()), resample);
    let v0 = i32x4_shuffle::<0, 0, 0, 0>(v, v);
    let v1 = i32x4_shuffle::<1, 1, 1, 1>(v, v);
    let v2 = i32x4_shuffle::<2, 2, 2, 2>(v, v);
    let v3 = i32x4_shuffle::<3, 3, 3, 3>(v, v);
    let e0 = f32x4_add(v0, v2);
    let e1 = f32x4_sub(v0, v2);
    let t2 = f32x4_mul(v1, f32x4_splat(std::f32::consts::SQRT_2));
    let t3 = f32x4_add(v3, v1);
    let o0 = f32x4_add(t2, t3);
    let o1 = f32x4_sub(t2, t3);
    let even = i32x4_shuffle::<0, 4, 4, 0>(e0, e1);
    let odd = i32x4_shuffle::<0, 4, 4, 0>(o0, o1);
    let weights = f32x4(WC4[0], WC4[1], -WC4[1], -WC4[0]);
    unsafe {
        v128_store(
            dc.as_mut_ptr().cast(),
            f32x4_add(even, f32x4_mul(odd, weights)),
        )
    };
}

// Restore the natural frequency order of the 64-point split-radix kernel.
#[target_feature(enable = "simd128")]
#[cfg(feature = "extraslow")]
fn dct1d_64_natural(c: &mut [v128; 64]) {
    dct1d_64_s(c);
    let split = *c;
    for i in 0..64 {
        c[i] = split[crate::dct128::dct64_natural_index(i)];
    }
}

/// Split-radix DCT-II over 4 independent 128-point columns.
#[target_feature(enable = "simd128")]
#[cfg(feature = "extraslow")]
fn dct1d_128(c: &mut [v128; 128]) {
    for i in 0..32 {
        let bottom = c[i];
        let top = c[127 - i];
        let half_bottom = c[63 - i];
        let half_top = c[64 + i];
        let lower = f32x4_sub(bottom, top);
        let upper = f32x4_sub(half_bottom, half_top);
        c[i] = f32x4_add(bottom, top);
        c[63 - i] = f32x4_add(half_bottom, half_top);
        c[64 + i] = f32x4_add(
            f32x4_mul(lower, f32x4_splat(DCT128_SPLIT_COS[i])),
            f32x4_mul(upper, f32x4_splat(DCT128_SPLIT_SIN[i])),
        );
        let sin = f32x4_sub(
            f32x4_mul(upper, f32x4_splat(DCT128_SPLIT_COS[i])),
            f32x4_mul(lower, f32x4_splat(DCT128_SPLIT_SIN[i])),
        );
        c[127 - i] = if i & 1 == 0 { sin } else { f32x4_neg(sin) };
    }
    dct1d_64_natural((&mut c[..64]).try_into().unwrap());
    dct1d_32_s((&mut c[64..96]).try_into().unwrap());
    dct1d_32_s((&mut c[96..]).try_into().unwrap());
    for i in 1..32 {
        let cos = c[64 + i];
        let sin = if i & 1 == 0 {
            f32x4_neg(c[128 - i])
        } else {
            c[128 - i]
        };
        c[64 + i] = f32x4_add(cos, sin);
        c[128 - i] = f32x4_sub(cos, sin);
    }
    c[64] = f32x4_mul(c[64], f32x4_splat(std::f32::consts::SQRT_2));
    c[96] = f32x4_mul(c[96], f32x4_splat(-std::f32::consts::SQRT_2));
    let split = *c;
    for i in 0..64 {
        c[2 * i] = split[i];
        let odd = if i == 0 {
            64
        } else if i & 1 != 0 {
            65 + i / 2
        } else {
            128 - i / 2
        };
        c[2 * i + 1] = split[odd];
    }
}

// Transposed scratch makes both separable passes use contiguous SIMD loads.
#[cfg(feature = "extraslow")]
macro_rules! dct128_wasm {
    ($name:ident, $n:literal, $w:literal, $h:literal, $dct_v:path, $dct_h:path) => {
        #[target_feature(enable = "simd128")]
        pub(crate) fn $name(input: DctInput<'_, $w, $h>, output: &mut [f32; $n]) {
            let mut scratch = Box::<[f32; $n]>::new_uninit();
            let dst = scratch.as_mut_ptr() as *mut f32;
            for x in (0..$w).step_by(4) {
                let mut c: [v128; $h] =
                    std::array::from_fn(|y| load_strip(input.row(y)[x..].as_ptr()));
                $dct_v(&mut c);
                for v in (0..$h).step_by(4) {
                    let mut tile: [v128; 4] = c[v..v + 4].try_into().unwrap();
                    let (r0, r1, r2, r3) = transpose_4x4(tile[0], tile[1], tile[2], tile[3]);
                    tile = [r0, r1, r2, r3];
                    for (j, value) in tile.iter().enumerate() {
                        unsafe { v128_store(dst.add((x + j) * $h + v) as *mut v128, *value) };
                    }
                }
            }
            // Every scratch element was written by exactly one tile above.
            let scratch = unsafe { scratch.assume_init() };
            let scale = 1.0 / $n as f32;
            for v in (0..$h).step_by(4) {
                let mut c: [v128; $w] =
                    std::array::from_fn(|x| load_strip(scratch[x * $h + v..].as_ptr()));
                $dct_h(&mut c);
                if $h >= $w {
                    for u in 0..$w {
                        unsafe {
                            v128_store(
                                output.as_mut_ptr().add(u * $h + v) as *mut v128,
                                f32x4_mul(c[u], f32x4_splat(scale)),
                            )
                        };
                    }
                } else {
                    for u in (0..$w).step_by(4) {
                        let mut tile: [v128; 4] = c[u..u + 4].try_into().unwrap();
                        let (r0, r1, r2, r3) = transpose_4x4(tile[0], tile[1], tile[2], tile[3]);
                        tile = [r0, r1, r2, r3];
                        for (j, value) in tile.iter().enumerate() {
                            unsafe {
                                v128_store(
                                    output.as_mut_ptr().add((v + j) * $w + u) as *mut v128,
                                    f32x4_mul(*value, f32x4_splat(scale)),
                                )
                            };
                        }
                    }
                }
            }
        }
    };
}

#[cfg(feature = "extraslow")]
dct128_wasm!(dct128x128_wasm, 16384, 128, 128, dct1d_128, dct1d_128);
#[cfg(feature = "extraslow")]
dct128_wasm!(dct128x64_wasm, 8192, 64, 128, dct1d_128, dct1d_64_natural);
#[cfg(feature = "extraslow")]
dct128_wasm!(dct64x128_wasm, 8192, 128, 64, dct1d_64_natural, dct1d_128);

const IS2: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[inline]
#[target_feature(enable = "simd128")]
fn idct2_s(a: v128, b: v128) -> (v128, v128) {
    (
        f32x4_mul(f32x4_add(a, b), f32x4_splat(0.5)),
        f32x4_mul(f32x4_sub(a, b), f32x4_splat(0.5)),
    )
}

#[inline]
#[target_feature(enable = "simd128")]
fn inv_dct1d_4_s(c: &mut [v128; 4]) {
    let (t0, t1, mut t2, t3) = (c[0], c[2], c[1], c[3]);
    t2 = f32x4_mul(f32x4_sub(t2, t3), f32x4_splat(IS2));
    let (n2, n3) = idct2_s(t2, t3);
    let a2 = f32x4_mul(n2, f32x4_splat(INV_WC4[0]));
    let a3 = f32x4_mul(n3, f32x4_splat(INV_WC4[1]));
    let (m0, m1) = idct2_s(t0, t1);
    c[0] = f32x4_mul(f32x4_add(m0, a2), f32x4_splat(0.5));
    c[3] = f32x4_mul(f32x4_sub(m0, a2), f32x4_splat(0.5));
    c[1] = f32x4_mul(f32x4_add(m1, a3), f32x4_splat(0.5));
    c[2] = f32x4_mul(f32x4_sub(m1, a3), f32x4_splat(0.5));
}

#[inline]
#[target_feature(enable = "simd128")]
fn inv_dct1d_8_s(c: &mut [v128; 8]) {
    let mut t = [c[0]; 8];
    for i in 0..4 {
        t[i] = c[2 * i];
        t[4 + i] = c[2 * i + 1];
    }
    t[6] = f32x4_sub(t[6], t[7]);
    t[5] = f32x4_sub(t[5], t[6]);
    t[4] = f32x4_mul(f32x4_sub(t[4], t[5]), f32x4_splat(IS2));
    let mut o: [v128; 4] = [t[4], t[5], t[6], t[7]];
    inv_dct1d_4_s(&mut o);
    for i in 0..4 {
        o[i] = f32x4_mul(o[i], f32x4_splat(INV_WC8[i]));
    }
    let mut e: [v128; 4] = [t[0], t[1], t[2], t[3]];
    inv_dct1d_4_s(&mut e);
    for i in 0..4 {
        c[i] = f32x4_mul(f32x4_add(e[i], o[i]), f32x4_splat(0.5));
        c[7 - i] = f32x4_mul(f32x4_sub(e[i], o[i]), f32x4_splat(0.5));
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn inv_dct1d_16_s(c: &mut [v128; 16]) {
    let mut t = [c[0]; 16];
    for i in 0..8 {
        t[i] = c[2 * i];
        t[8 + i] = c[2 * i + 1];
    }
    for i in (9..=14).rev() {
        t[i] = f32x4_sub(t[i], t[i + 1]);
    }
    t[8] = f32x4_mul(f32x4_sub(t[8], t[9]), f32x4_splat(IS2));
    let mut o: [v128; 8] = std::array::from_fn(|i| t[8 + i]);
    inv_dct1d_8_s(&mut o);
    for i in 0..8 {
        o[i] = f32x4_mul(o[i], f32x4_splat(INV_WC16[i]));
    }
    let mut e: [v128; 8] = std::array::from_fn(|i| t[i]);
    inv_dct1d_8_s(&mut e);
    for i in 0..8 {
        c[i] = f32x4_mul(f32x4_add(e[i], o[i]), f32x4_splat(0.5));
        c[15 - i] = f32x4_mul(f32x4_sub(e[i], o[i]), f32x4_splat(0.5));
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn inv_dct1d_32_s(c: &mut [v128; 32]) {
    let mut t = [c[0]; 32];
    for i in 0..16 {
        t[i] = c[2 * i];
        t[16 + i] = c[2 * i + 1];
    }
    for i in (17..=30).rev() {
        t[i] = f32x4_sub(t[i], t[i + 1]);
    }
    t[16] = f32x4_mul(f32x4_sub(t[16], t[17]), f32x4_splat(IS2));
    let mut o: [v128; 16] = std::array::from_fn(|i| t[16 + i]);
    inv_dct1d_16_s(&mut o);
    for i in 0..16 {
        o[i] = f32x4_mul(o[i], f32x4_splat(INV_WC32[i]));
    }
    let mut e: [v128; 16] = std::array::from_fn(|i| t[i]);
    inv_dct1d_16_s(&mut e);
    for i in 0..16 {
        c[i] = f32x4_mul(f32x4_add(e[i], o[i]), f32x4_splat(0.5));
        c[31 - i] = f32x4_mul(f32x4_sub(e[i], o[i]), f32x4_splat(0.5));
    }
}

#[inline]
#[target_feature(enable = "simd128")]
#[cfg(feature = "extraslow")]
fn inv_dct1d_64_s(c: &mut [v128; 64]) {
    let mut t = [c[0]; 64];
    for i in 0..32 {
        t[i] = c[2 * i];
        t[32 + i] = c[2 * i + 1];
    }
    for i in (33..=62).rev() {
        t[i] = f32x4_sub(t[i], t[i + 1]);
    }
    t[32] = f32x4_mul(
        f32x4_sub(t[32], t[33]),
        f32x4_splat(std::f32::consts::FRAC_1_SQRT_2),
    );
    let mut odd: [v128; 32] = std::array::from_fn(|i| t[32 + i]);
    inv_dct1d_32_s(&mut odd);
    for i in 0..32 {
        odd[i] = f32x4_mul(odd[i], f32x4_splat(INV_WC64[i]));
    }
    let mut even: [v128; 32] = std::array::from_fn(|i| t[i]);
    inv_dct1d_32_s(&mut even);
    for i in 0..32 {
        c[i] = f32x4_mul(f32x4_add(even[i], odd[i]), f32x4_splat(0.5));
        c[63 - i] = f32x4_mul(f32x4_sub(even[i], odd[i]), f32x4_splat(0.5));
    }
}

macro_rules! horizontal_idct_pass_wasm {
    ($tmp:expr, $out:expr, $h:literal, $w:literal, $inv_h:path) => {{
        for y4 in (0..$h).step_by(4) {
            let mut c = [f32x4_splat(0.0); $w];
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
                    v128_store(($out.as_mut_ptr().add(y4 * $w + x4)).cast(), r0);
                    v128_store(($out.as_mut_ptr().add((y4 + 1) * $w + x4)).cast(), r1);
                    v128_store(($out.as_mut_ptr().add((y4 + 2) * $w + x4)).cast(), r2);
                    v128_store(($out.as_mut_ptr().add((y4 + 3) * $w + x4)).cast(), r3);
                }
            }
        }
    }};
}

/// IDCT for a natural coefficient layout `[vfreq * W + hfreq]`.
macro_rules! inv_dct_natural_wasm {
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path) => {
        inv_dct_natural_wasm!(
            $name,
            $n,
            $h,
            $w,
            $inv_v,
            $inv_h,
            MaybeUninit::<[f32; $n]>::uninit()
        );
    };
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path, $scratch:expr) => {
        #[target_feature(enable = "simd128")]
        pub(crate) fn $name(coeff: DctInput<'_, $w, $h>, out: &mut [f32; $n]) {
            let mut tmp_u = $scratch;
            let tmp_ptr = tmp_u.as_mut_ptr() as *mut f32;
            for u4 in (0..$w).step_by(4) {
                let mut c: [v128; $h] = std::array::from_fn(|v| {
                    f32x4_mul(
                        load_strip(coeff.row(v)[u4..].as_ptr()),
                        f32x4_splat($n as f32),
                    )
                });
                $inv_v(&mut c);
                for y in 0..$h {
                    unsafe { v128_store((tmp_ptr.add(y * $w + u4)).cast(), c[y]) };
                }
            }
            let tmp = unsafe { tmp_u.assume_init() };
            horizontal_idct_pass_wasm!(tmp, out, $h, $w, $inv_h);
        }
    };
}

/// IDCT for a transposed coefficient layout `[hfreq * H + vfreq]`.
macro_rules! inv_dct_transposed_wasm {
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path) => {
        inv_dct_transposed_wasm!(
            $name,
            $n,
            $h,
            $w,
            $inv_v,
            $inv_h,
            MaybeUninit::<[f32; $n]>::uninit()
        );
    };
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path, $scratch:expr) => {
        #[target_feature(enable = "simd128")]
        pub(crate) fn $name(coeff: DctInput<'_, $h, $w>, out: &mut [f32; $n]) {
            let mut tmp_u = $scratch;
            let tmp_ptr = tmp_u.as_mut_ptr() as *mut f32;
            for u4 in (0..$w).step_by(4) {
                let mut c = [f32x4_splat(0.0); $h];
                for v4 in (0..$h).step_by(4) {
                    let (a, b, cc, d) = transpose_4x4(
                        load_strip(coeff.row(u4)[v4..].as_ptr()),
                        load_strip(coeff.row(u4 + 1)[v4..].as_ptr()),
                        load_strip(coeff.row(u4 + 2)[v4..].as_ptr()),
                        load_strip(coeff.row(u4 + 3)[v4..].as_ptr()),
                    );
                    c[v4] = f32x4_mul(a, f32x4_splat($n as f32));
                    c[v4 + 1] = f32x4_mul(b, f32x4_splat($n as f32));
                    c[v4 + 2] = f32x4_mul(cc, f32x4_splat($n as f32));
                    c[v4 + 3] = f32x4_mul(d, f32x4_splat($n as f32));
                }
                $inv_v(&mut c);
                for y in 0..$h {
                    unsafe { v128_store((tmp_ptr.add(y * $w + u4)).cast(), c[y]) };
                }
            }
            let tmp = unsafe { tmp_u.assume_init() };
            horizontal_idct_pass_wasm!(tmp, out, $h, $w, $inv_h);
        }
    };
}

inv_dct_natural_wasm!(inv_dct8x32_wasm, 256, 8, 32, inv_dct1d_8_s, inv_dct1d_32_s);
inv_dct_transposed_wasm!(inv_dct32x8_wasm, 256, 32, 8, inv_dct1d_32_s, inv_dct1d_8_s);

#[target_feature(enable = "simd128")]
#[cfg(feature = "extraslow")]
fn inv_dct1d_128(c: &mut [v128; 128]) {
    let mut even: [v128; 64] = std::array::from_fn(|i| c[2 * i]);
    let mut odd: [v128; 64] = std::array::from_fn(|i| c[2 * i + 1]);
    for i in (1..63).rev() {
        odd[i] = f32x4_sub(odd[i], odd[i + 1]);
    }
    odd[0] = f32x4_mul(
        f32x4_sub(odd[0], odd[1]),
        f32x4_splat(std::f32::consts::FRAC_1_SQRT_2),
    );
    inv_dct1d_64_s(&mut odd);
    inv_dct1d_64_s(&mut even);
    for i in 0..64 {
        let o = f32x4_mul(odd[i], f32x4_splat(INV_WC128[i]));
        c[i] = f32x4_mul(f32x4_add(even[i], o), f32x4_splat(0.5));
        c[127 - i] = f32x4_mul(f32x4_sub(even[i], o), f32x4_splat(0.5));
    }
}

#[cfg(feature = "extraslow")]
inv_dct_transposed_wasm!(
    inv_dct128x128_wasm,
    16384,
    128,
    128,
    inv_dct1d_128,
    inv_dct1d_128,
    Box::<[f32; 16384]>::new_uninit()
);
#[cfg(feature = "extraslow")]
inv_dct_transposed_wasm!(
    inv_dct128x64_wasm,
    8192,
    128,
    64,
    inv_dct1d_128,
    inv_dct1d_64_s,
    Box::<[f32; 8192]>::new_uninit()
);
#[cfg(feature = "extraslow")]
inv_dct_natural_wasm!(
    inv_dct64x128_wasm,
    8192,
    64,
    128,
    inv_dct1d_64_s,
    inv_dct1d_128,
    Box::<[f32; 8192]>::new_uninit()
);

#[cfg(feature = "extraslow")]
macro_rules! dc_from_dct128_wasm {
    ($name:ident, $n:literal, $rows:literal, $inv_v:path) => {
        #[target_feature(enable = "simd128")]
        pub(crate) fn $name(coeffs: &[f32; $n], dc: &mut [f32; { $rows * 16 }]) {
            let mut scratch = [[0.0f32; 16]; $rows];
            for x in (0..16).step_by(4) {
                let resample = load_strip(RESAMPLE_SCALE_128_TO_16[x..].as_ptr());
                let mut c: [v128; $rows] = std::array::from_fn(|y| {
                    let row_scale = if $rows == 16 {
                        RESAMPLE_SCALE_128_TO_16[y]
                    } else {
                        RESAMPLE_SCALE_64_TO_8[y]
                    };
                    f32x4_mul(
                        f32x4_mul(load_strip(coeffs[y * 128 + x..].as_ptr()), resample),
                        f32x4_splat(($rows * 16) as f32 * row_scale),
                    )
                });
                $inv_v(&mut c);
                for y in 0..$rows {
                    unsafe { v128_store((scratch[y].as_mut_ptr().add(x)).cast(), c[y]) };
                }
            }
            for y in (0..$rows).step_by(4) {
                let mut c = [f32x4_splat(0.0); 16];
                for x in (0..16).step_by(4) {
                    let mut tile: [v128; 4] =
                        std::array::from_fn(|j| load_strip(scratch[y + j][x..].as_ptr()));
                    let (r0, r1, r2, r3) = transpose_4x4(tile[0], tile[1], tile[2], tile[3]);
                    tile = [r0, r1, r2, r3];
                    c[x..x + 4].copy_from_slice(&tile);
                }
                inv_dct1d_16_s(&mut c);
                // Square and tall footprints store the transposed DC grid.
                for x in 0..16 {
                    unsafe { v128_store((dc.as_mut_ptr().add(x * $rows + y)).cast(), c[x]) };
                }
            }
        }
    };
}

#[cfg(feature = "extraslow")]
dc_from_dct128_wasm!(dc_from_dct128x128_wasm, 16384, 16, inv_dct1d_16_s);
#[cfg(feature = "extraslow")]
dc_from_dct128_wasm!(dc_from_dct128x64_wasm, 8192, 8, inv_dct1d_8_s);

#[target_feature(enable = "simd128")]
#[cfg(feature = "extraslow")]
pub(crate) fn dc_from_dct64x128_wasm(coeffs: &[f32; 8192], dc: &mut [f32; 128]) {
    let mut transposed = [0.0f32; 128];
    dc_from_dct128x64_wasm(coeffs, &mut transposed);
    for y in (0..8).step_by(4) {
        for x in (0..16).step_by(4) {
            let mut tile: [v128; 4] =
                std::array::from_fn(|j| load_strip(transposed[(x + j) * 8 + y..].as_ptr()));
            let (r0, r1, r2, r3) = transpose_4x4(tile[0], tile[1], tile[2], tile[3]);
            tile = [r0, r1, r2, r3];
            for (j, row) in tile.iter().enumerate() {
                unsafe { v128_store((dc.as_mut_ptr().add((y + j) * 16 + x)).cast(), *row) };
            }
        }
    }
}

#[cfg(test)]
mod wasm_dct_tests {
    use crate::dct::{DctInput, dct8x8_scalar, dct8x16_scalar, dct16x8_scalar};

    const ATOL: f32 = 1e-4;

    fn assert_close(wasm: &[f32], scalar: &[f32], label: &str) {
        assert_eq!(wasm.len(), scalar.len(), "{label}: length mismatch");
        let mut max_err: f32 = 0.0;
        let mut worst = 0usize;
        for (i, (n, s)) in wasm.iter().zip(scalar.iter()).enumerate() {
            let e = (n - s).abs();
            if e > max_err {
                max_err = e;
                worst = i;
            }
        }
        assert!(
            max_err < ATOL,
            "{label}: max error {max_err:.2e} at index {worst} \
             (wasm={:.6}, scalar={:.6})",
            wasm[worst],
            scalar[worst]
        );
    }

    /// Deterministic pseudo-random f32 in [-1, 1] seeded by index.
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
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x8_neon_vs_scalar_random() {
        use crate::wasm::dct8x8_wasm;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed);
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            dct8x8_wasm(DctInput::from_flat(&input), &mut got);
            dct8x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x8_neon_dc_only() {
        // All-constant input: only DC coefficient should be non-zero.
        use crate::wasm::dct8x8_wasm;
        let input = [0.5f32; 64];
        let mut got = [0.0f32; 64];
        let mut want = [0.0f32; 64];
        dct8x8_wasm(DctInput::from_flat(&input), &mut got);
        dct8x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x8 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x8_neon_zero() {
        use crate::wasm::dct8x8_wasm;
        let input = [0.0f32; 64];
        let mut got = [0.0f32; 64];
        let mut want = [0.0f32; 64];
        dct8x8_wasm(DctInput::from_flat(&input), &mut got);
        dct8x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x8 zero");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x8_neon_linearity() {
        // DCT is linear: DCT(a + b) == DCT(a) + DCT(b)
        use crate::wasm::dct8x8_wasm;
        let a: [f32; 64] = fill(100);
        let b: [f32; 64] = fill(200);
        let mut sum = [0.0f32; 64];
        for i in 0..64 {
            sum[i] = a[i] + b[i];
        }

        let mut da = [0.0f32; 64];
        let mut db = [0.0f32; 64];
        let mut dsum = [0.0f32; 64];
        dct8x8_wasm(DctInput::from_flat(&a), &mut da);
        dct8x8_wasm(DctInput::from_flat(&b), &mut db);
        dct8x8_wasm(DctInput::from_flat(&sum), &mut dsum);
        let expected: Vec<f32> = (0..64).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct8x8 linearity");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x8_neon_basis_vectors() {
        // Feed each of the 64 basis vectors (single 1.0) and compare to scalar.
        use crate::wasm::dct8x8_wasm;
        for k in 0..64 {
            let mut input = [0.0f32; 64];
            input[k] = 1.0;
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            dct8x8_wasm(DctInput::from_flat(&input), &mut got);
            dct8x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x8 basis[{k}]"));
        }
    }

    // ── dct8x16 ───────────────────────────────────────────────────────────────

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x16_neon_vs_scalar_random() {
        use crate::wasm::dct8x16_wasm;
        for seed in 0u64..32 {
            let input: [f32; 128] = fill(seed.wrapping_add(0xdead));
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            dct8x16_wasm(DctInput::from_flat(&input), &mut got);
            dct8x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x16 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x16_neon_dc_only() {
        use crate::wasm::dct8x16_wasm;
        let input = [0.5f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        dct8x16_wasm(DctInput::from_flat(&input), &mut got);
        dct8x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x16 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x16_neon_zero() {
        use crate::wasm::dct8x16_wasm;
        let input = [0.0f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        dct8x16_wasm(DctInput::from_flat(&input), &mut got);
        dct8x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x16 zero");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x16_neon_basis_vectors() {
        use crate::wasm::dct8x16_wasm;
        for k in 0..128 {
            let mut input = [0.0f32; 128];
            input[k] = 1.0;
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            dct8x16_wasm(DctInput::from_flat(&input), &mut got);
            dct8x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x16 basis[{k}]"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x16_neon_linearity() {
        use crate::wasm::dct8x16_wasm;
        let a: [f32; 128] = fill(300);
        let b: [f32; 128] = fill(400);
        let mut sum = [0.0f32; 128];
        for i in 0..128 {
            sum[i] = a[i] + b[i];
        }
        let mut da = [0.0f32; 128];
        let mut db = [0.0f32; 128];
        let mut dsum = [0.0f32; 128];
        dct8x16_wasm(DctInput::from_flat(&a), &mut da);
        dct8x16_wasm(DctInput::from_flat(&b), &mut db);
        dct8x16_wasm(DctInput::from_flat(&sum), &mut dsum);
        let expected: Vec<f32> = (0..128).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct8x16 linearity");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x8_neon_vs_scalar_random() {
        use crate::wasm::dct16x8_wasm;
        for seed in 0u64..32 {
            let input: [f32; 128] = fill(seed.wrapping_add(0xbeef));
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            dct16x8_wasm(DctInput::from_flat(&input), &mut got);
            dct16x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct4x8_neon_vs_scalar_random() {
        use crate::dct::dct4x8_scalar;
        use crate::wasm::dct4x8_wasm;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x4a8));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            dct4x8_wasm(DctInput::from_flat(&input), &mut got);
            dct4x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct4x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x4_neon_vs_scalar_random() {
        use crate::dct::dct8x4_scalar;
        use crate::wasm::dct8x4_wasm;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x8a4));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            dct8x4_wasm(DctInput::from_flat(&input), &mut got);
            dct8x4_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x4 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x8_neon_dc_only() {
        use crate::wasm::dct16x8_wasm;
        let input = [0.5f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        dct16x8_wasm(DctInput::from_flat(&input), &mut got);
        dct16x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x8 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x8_neon_zero() {
        use crate::wasm::dct16x8_wasm;
        let input = [0.0f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        dct16x8_wasm(DctInput::from_flat(&input), &mut got);
        dct16x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x8 zero");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x8_neon_basis_vectors() {
        use crate::wasm::dct16x8_wasm;
        for k in 0..128 {
            let mut input = [0.0f32; 128];
            input[k] = 1.0;
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            dct16x8_wasm(DctInput::from_flat(&input), &mut got);
            dct16x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x8 basis[{k}]"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x8_neon_linearity() {
        use crate::wasm::dct16x8_wasm;
        let a: [f32; 128] = fill(500);
        let b: [f32; 128] = fill(600);
        let mut sum = [0.0f32; 128];
        for i in 0..128 {
            sum[i] = a[i] + b[i];
        }
        let mut da = [0.0f32; 128];
        let mut db = [0.0f32; 128];
        let mut dsum = [0.0f32; 128];
        dct16x8_wasm(DctInput::from_flat(&a), &mut da);
        dct16x8_wasm(DctInput::from_flat(&b), &mut db);
        dct16x8_wasm(DctInput::from_flat(&sum), &mut dsum);
        let expected: Vec<f32> = (0..128).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct16x8 linearity");
    }

    // ── Cross-shape consistency ────────────────────────────────────────────────
    // A 16x8 block is the transpose of an 8x16 block.
    // DCT(A^T)[u][v] relates to DCT(A)[v][u] by the separable 2-D DCT symmetry.
    // We test a weaker property: both kernels produce the same DC coefficient
    // for an all-constant input (DC is rotation-invariant).

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dc_coefficient_constant_input() {
        use crate::wasm::{dct8x16_wasm, dct16x8_wasm};
        let input = [0.25f32; 128];

        let mut out8x16 = [0.0f32; 128];
        let mut out16x8 = [0.0f32; 128];
        dct8x16_wasm(DctInput::from_flat(&input), &mut out8x16);
        dct16x8_wasm(DctInput::from_flat(&input), &mut out16x8);
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
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x8_neon_extreme_values() {
        use crate::wasm::dct8x8_wasm;
        // Alternating +1/-1 (highest-frequency input)
        let mut input = [0.0f32; 64];
        for i in 0..64 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 64];
        let mut want = [0.0f32; 64];
        dct8x8_wasm(DctInput::from_flat(&input), &mut got);
        dct8x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x8 alternating +-1");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x8_neon_extreme_values() {
        use crate::wasm::dct16x8_wasm;
        let mut input = [0.0f32; 128];
        for i in 0..128 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        dct16x8_wasm(DctInput::from_flat(&input), &mut got);
        dct16x8_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x8 alternating +-1");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct8x16_neon_extreme_values() {
        use crate::wasm::dct8x16_wasm;
        let mut input = [0.0f32; 128];
        for i in 0..128 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        dct8x16_wasm(DctInput::from_flat(&input), &mut got);
        dct8x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct8x16 alternating +-1");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x16_neon_vs_scalar_random() {
        use crate::dct::dct16x16_scalar;
        use crate::wasm::dct16x16_wasm;
        for seed in 0u64..32 {
            let input: [f32; 256] = fill(seed.wrapping_add(0xf00d));
            let mut got = [0.0f32; 256];
            let mut want = [0.0f32; 256];
            dct16x16_wasm(DctInput::from_flat(&input), &mut got);
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct32x32_neon_vs_scalar_random() {
        use crate::dct::dct32x32_scalar;
        use crate::wasm::dct32x32_wasm;
        for seed in 0u64..16 {
            let input: [f32; 1024] = fill(seed.wrapping_add(0x3232));
            let mut got = [0.0f32; 1024];
            let mut want = [0.0f32; 1024];
            dct32x32_wasm(DctInput::from_flat(&input), &mut got);
            dct32x32_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct32x32 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct32x16_neon_vs_scalar_random() {
        use crate::dct::dct32x16_scalar;
        use crate::wasm::dct32x16_wasm;
        for seed in 0u64..16 {
            let input: [f32; 512] = fill(seed.wrapping_add(0x3216));
            let mut got = [0.0f32; 512];
            let mut want = [0.0f32; 512];
            dct32x16_wasm(DctInput::from_flat(&input), &mut got);
            dct32x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct32x16 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x32_neon_vs_scalar_random() {
        use crate::dct::dct16x32_scalar;
        use crate::wasm::dct16x32_wasm;
        for seed in 0u64..16 {
            let input: [f32; 512] = fill(seed.wrapping_add(0x1632));
            let mut got = [0.0f32; 512];
            let mut want = [0.0f32; 512];
            dct16x32_wasm(DctInput::from_flat(&input), &mut got);
            dct16x32_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x32 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct4x4_neon_vs_scalar_random() {
        use crate::dct::dct4x4_scalar;
        use crate::wasm::dct4x4_wasm;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x4a4));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            dct4x4_wasm(DctInput::from_flat(&input), &mut got);
            dct4x4_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct4x4 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x16_neon_dc_only() {
        use crate::dct::dct16x16_scalar;
        use crate::wasm::dct16x16_wasm;
        let input = [0.5f32; 256];
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        dct16x16_wasm(DctInput::from_flat(&input), &mut got);
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 dc-only");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x16_neon_zero() {
        use crate::dct::dct16x16_scalar;
        use crate::wasm::dct16x16_wasm;
        let input = [0.0f32; 256];
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        dct16x16_wasm(DctInput::from_flat(&input), &mut got);
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 zero");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x16_neon_basis_vectors() {
        use crate::dct::dct16x16_scalar;
        use crate::wasm::dct16x16_wasm;
        for k in 0..256 {
            let mut input = [0.0f32; 256];
            input[k] = 1.0;
            let mut got = [0.0f32; 256];
            let mut want = [0.0f32; 256];
            dct16x16_wasm(DctInput::from_flat(&input), &mut got);
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 basis[{k}]"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x16_neon_linearity() {
        use crate::wasm::dct16x16_wasm;
        let a: [f32; 256] = fill(700);
        let b: [f32; 256] = fill(800);
        let mut sum = [0.0f32; 256];
        for i in 0..256 {
            sum[i] = a[i] + b[i];
        }
        let mut da = [0.0f32; 256];
        let mut db = [0.0f32; 256];
        let mut dsum = [0.0f32; 256];
        dct16x16_wasm(DctInput::from_flat(&a), &mut da);
        dct16x16_wasm(DctInput::from_flat(&b), &mut db);
        dct16x16_wasm(DctInput::from_flat(&sum), &mut dsum);
        let expected: Vec<f32> = (0..256).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct16x16 linearity");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dct16x16_neon_extreme_values() {
        use crate::dct::dct16x16_scalar;
        use crate::wasm::dct16x16_wasm;
        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        dct16x16_wasm(DctInput::from_flat(&input), &mut got);
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 alternating +-1");
    }
}
