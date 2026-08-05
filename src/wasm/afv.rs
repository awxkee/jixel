/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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
use crate::afv::AFV_BASIS_TRANSPOSE;
use crate::dct::DctInput;
use std::arch::wasm32::*;

use super::dct::{dct1d_4_s, dct1d_8_s, transpose_4x4};

#[inline]
#[target_feature(enable = "simd128")]
fn load(ptr: *const f32) -> v128 {
    unsafe { v128_load(ptr as *const v128) }
}

#[inline]
#[target_feature(enable = "simd128")]
fn transpose(rows: &mut [v128; 4]) {
    let (a, b, c, d) = transpose_4x4(rows[0], rows[1], rows[2], rows[3]);
    *rows = [a, b, c, d];
}

#[inline]
#[target_feature(enable = "simd128")]
fn afv_basis<const KIND: usize>(input: DctInput<'_, 8, 8>) -> [v128; 4] {
    let afv_x = KIND & 1;
    let afv_y = KIND >> 1;
    let mut out = [f32x4_splat(0.0); 4];
    for j in 0..16 {
        let dy = j / 4;
        let dx = j % 4;
        let sy = if afv_y == 1 { 3 - dy } else { dy } + 4 * afv_y;
        let sx = if afv_x == 1 { 3 - dx } else { dx } + 4 * afv_x;
        let px = f32x4_splat(input.row(sy)[sx]);
        for (chunk, acc) in out.iter_mut().enumerate() {
            let basis = load(AFV_BASIS_TRANSPOSE[j][chunk * 4..].as_ptr());
            *acc = f32x4_add(*acc, f32x4_mul(px, basis));
        }
    }
    out
}

#[inline]
#[target_feature(enable = "simd128")]
fn adjacent_dct4x4<const KIND: usize>(input: DctInput<'_, 8, 8>) -> [v128; 4] {
    let afv_x = KIND & 1;
    let afv_y = KIND >> 1;
    let qx = if afv_x == 1 { 0 } else { 4 };
    let mut rows: [v128; 4] =
        std::array::from_fn(|y| load(input.row(y + 4 * afv_y)[qx..].as_ptr()));
    transpose(&mut rows);
    dct1d_4_s(&mut rows);
    transpose(&mut rows);
    dct1d_4_s(&mut rows);
    transpose(&mut rows);
    let scale = f32x4_splat(1.0 / 16.0);
    rows.map(|row| f32x4_mul(row, scale))
}

#[inline]
#[target_feature(enable = "simd128")]
fn opposite_dct4x8<const KIND: usize>(input: DctInput<'_, 8, 8>) -> ([v128; 4], [v128; 4]) {
    let afv_y = KIND >> 1;
    let hy = if afv_y == 1 { 0 } else { 4 };
    let mut left: [v128; 4] = std::array::from_fn(|y| load(input.row(hy + y).as_ptr()));
    let mut right: [v128; 4] = std::array::from_fn(|y| load(input.row(hy + y)[4..].as_ptr()));
    dct1d_4_s(&mut left);
    dct1d_4_s(&mut right);
    transpose(&mut left);
    transpose(&mut right);
    let mut freq = [left[0]; 8];
    freq[..4].copy_from_slice(&left);
    freq[4..].copy_from_slice(&right);
    dct1d_8_s(&mut freq);
    left.copy_from_slice(&freq[..4]);
    right.copy_from_slice(&freq[4..]);
    transpose(&mut left);
    transpose(&mut right);
    let scale = f32x4_splat(1.0 / 32.0);
    (
        left.map(|row| f32x4_mul(row, scale)),
        right.map(|row| f32x4_mul(row, scale)),
    )
}

#[inline]
#[target_feature(enable = "simd128")]
fn afv<const KIND: usize>(input: DctInput<'_, 8, 8>, out: &mut [f32; 64]) {
    let basis = afv_basis::<KIND>(input);
    let adjacent = adjacent_dct4x4::<KIND>(input);
    for y in 0..4 {
        let lo = i32x4_shuffle::<0, 4, 1, 5>(basis[y], adjacent[y]);
        let hi = i32x4_shuffle::<2, 6, 3, 7>(basis[y], adjacent[y]);
        unsafe {
            v128_store(out[y * 16..].as_mut_ptr() as *mut v128, lo);
            v128_store(out[y * 16 + 4..].as_mut_ptr() as *mut v128, hi);
        }
    }

    let (left, right) = opposite_dct4x8::<KIND>(input);
    for y in 0..4 {
        unsafe {
            v128_store(out[(2 * y + 1) * 8..].as_mut_ptr() as *mut v128, left[y]);
            v128_store(
                out[(2 * y + 1) * 8 + 4..].as_mut_ptr() as *mut v128,
                right[y],
            );
        }
    }

    let block00 = out[0] * 0.25;
    let block01 = out[1];
    let block10 = out[8];
    out[0] = (block00 + block01 + 2.0 * block10) * 0.25;
    out[1] = (block00 - block01) * 0.5;
    out[8] = (block00 + block01 - 2.0 * block10) * 0.25;
}

macro_rules! afv_kernel {
    ($name:ident, $kind:literal) => {
        #[target_feature(enable = "simd128")]
        pub(crate) fn $name(input: DctInput<'_, 8, 8>, out: &mut [f32; 64]) {
            afv::<$kind>(input, out);
        }
    };
}

afv_kernel!(afv0_wasm, 0);
afv_kernel!(afv1_wasm, 1);
afv_kernel!(afv2_wasm, 2);
afv_kernel!(afv3_wasm, 3);
