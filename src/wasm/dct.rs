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

use crate::dct::{DctInput, WC4, WC8, WC16, WC32, WC64};
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
    let mut evens = [f32x4_splat(0.0); 32];
    let mut odds = [f32x4_splat(0.0); 32];
    for i in 0..32 {
        evens[i] = f32x4_add(c[i], c[63 - i]);
        odds[i] = f32x4_mul(f32x4_sub(c[i], c[63 - i]), f32x4_splat(WC64[i]));
    }
    dct1d_32_s(&mut evens);
    dct1d_32_s(&mut odds);
    odds[0] = f32x4_add(
        f32x4_mul(odds[0], f32x4_splat(std::f32::consts::SQRT_2)),
        odds[1],
    );
    for i in 1..31 {
        odds[i] = f32x4_add(odds[i], odds[i + 1]);
    }
    for i in 0..32 {
        c[2 * i] = evens[i];
        c[2 * i + 1] = odds[i];
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
        for tile_index in 0..16 {
            let (a, b, cc, d) = transpose_4x4(
                c[tile_index * 4],
                c[tile_index * 4 + 1],
                c[tile_index * 4 + 2],
                c[tile_index * 4 + 3],
            );
            for (lane, value) in [a, b, cc, d].iter().enumerate() {
                let p = unsafe { dst.add((strip * 4 + lane) * 64 + tile_index * 4) };
                unsafe { v128_store(p.cast(), *value) };
            }
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = f32x4_splat(1.0 / 4096.0);
    for strip in 0..16 {
        let mut c: [v128; 64] = std::array::from_fn(|column| {
            load_strip(unsafe { scratch.get_unchecked(column * 64 + strip * 4..) }.as_ptr())
        });
        dct1d_64_s(&mut c);
        for u in 0..64 {
            let p = unsafe { output.get_unchecked_mut(u * 64 + strip * 4..) };
            unsafe { v128_store(p.as_mut_ptr().cast(), f32x4_mul(c[u], scale)) };
        }
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
        for tile_index in 0..16 {
            let (a, b, cc, d) = transpose_4x4(
                c[tile_index * 4],
                c[tile_index * 4 + 1],
                c[tile_index * 4 + 2],
                c[tile_index * 4 + 3],
            );
            for (lane, value) in [a, b, cc, d].iter().enumerate() {
                let p = unsafe { dst.add((strip * 4 + lane) * 64 + tile_index * 4) };
                unsafe { v128_store(p.cast(), *value) };
            }
        }
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
        for u in 0..64 {
            let p = unsafe { dst.add(u * 32 + row_strip * 4) };
            unsafe { v128_store(p.cast(), c[u]) };
        }
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

#[cfg(target_feature = "simd128")]
#[cfg(test)]
mod neon_dct_tests {
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
            unsafe { dct8x8_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct8x8_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct8x8_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe {
            dct8x8_wasm(DctInput::from_flat(&a), &mut da);
            dct8x8_wasm(DctInput::from_flat(&b), &mut db);
            dct8x8_wasm(DctInput::from_flat(&sum), &mut dsum);
        }
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
            unsafe { dct8x8_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct8x16_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct8x16_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct8x16_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct8x16_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe {
            dct8x16_wasm(DctInput::from_flat(&a), &mut da);
            dct8x16_wasm(DctInput::from_flat(&b), &mut db);
            dct8x16_wasm(DctInput::from_flat(&sum), &mut dsum);
        }
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
            unsafe { dct16x8_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct4x8_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct8x4_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct16x8_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct16x8_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct16x8_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe {
            dct16x8_wasm(DctInput::from_flat(&a), &mut da);
            dct16x8_wasm(DctInput::from_flat(&b), &mut db);
            dct16x8_wasm(DctInput::from_flat(&sum), &mut dsum);
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
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    fn test_dc_coefficient_constant_input() {
        use crate::wasm::{dct8x16_wasm, dct16x8_wasm};
        let input = [0.25f32; 128];

        let mut out8x16 = [0.0f32; 128];
        let mut out16x8 = [0.0f32; 128];
        unsafe {
            dct8x16_wasm(DctInput::from_flat(&input), &mut out8x16);
            dct16x8_wasm(DctInput::from_flat(&input), &mut out16x8);
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
        unsafe { dct8x8_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct16x8_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct8x16_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct16x16_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct32x32_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct32x16_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct16x32_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct4x4_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct16x16_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe { dct16x16_wasm(DctInput::from_flat(&input), &mut got) };
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
            unsafe { dct16x16_wasm(DctInput::from_flat(&input), &mut got) };
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
        unsafe {
            dct16x16_wasm(DctInput::from_flat(&a), &mut da);
            dct16x16_wasm(DctInput::from_flat(&b), &mut db);
            dct16x16_wasm(DctInput::from_flat(&sum), &mut dsum);
        }
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
        unsafe { dct16x16_wasm(DctInput::from_flat(&input), &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 alternating +-1");
    }
}
