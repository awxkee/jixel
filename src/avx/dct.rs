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
    DctInput, INV_WC4, INV_WC8, INV_WC16, INV_WC32, INV_WC64, RESAMPLE_SCALE_16_TO_2,
    RESAMPLE_SCALE_32_TO_4, RESAMPLE_SCALE_64_TO_8, WC4, WC8, WC16, WC32, WC64,
};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8(c: &mut [__m256; 8]) {
    let t0 = _mm256_unpacklo_ps(c[0], c[1]);
    let t1 = _mm256_unpackhi_ps(c[0], c[1]);
    let t2 = _mm256_unpacklo_ps(c[2], c[3]);
    let t3 = _mm256_unpackhi_ps(c[2], c[3]);
    let t4 = _mm256_unpacklo_ps(c[4], c[5]);
    let t5 = _mm256_unpackhi_ps(c[4], c[5]);
    let t6 = _mm256_unpacklo_ps(c[6], c[7]);
    let t7 = _mm256_unpackhi_ps(c[6], c[7]);

    let s0 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t0),
        _mm256_castps_pd(t2),
    ));
    let s1 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t0),
        _mm256_castps_pd(t2),
    ));
    let s2 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t1),
        _mm256_castps_pd(t3),
    ));
    let s3 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t1),
        _mm256_castps_pd(t3),
    ));
    let s4 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t4),
        _mm256_castps_pd(t6),
    ));
    let s5 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t4),
        _mm256_castps_pd(t6),
    ));
    let s6 = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(t5),
        _mm256_castps_pd(t7),
    ));
    let s7 = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(t5),
        _mm256_castps_pd(t7),
    ));

    // permute 128-bit lanes
    c[0] = _mm256_permute2f128_ps::<0x20>(s0, s4);
    c[1] = _mm256_permute2f128_ps::<0x20>(s1, s5);
    c[2] = _mm256_permute2f128_ps::<0x20>(s2, s6);
    c[3] = _mm256_permute2f128_ps::<0x20>(s3, s7);
    c[4] = _mm256_permute2f128_ps::<0x31>(s0, s4);
    c[5] = _mm256_permute2f128_ps::<0x31>(s1, s5);
    c[6] = _mm256_permute2f128_ps::<0x31>(s2, s6);
    c[7] = _mm256_permute2f128_ps::<0x31>(s3, s7);
}

#[inline]
#[target_feature(enable = "avx2")]
fn load(input: DctInput<'_, 8, 8>) -> [__m256; 8] {
    unsafe { std::array::from_fn(|i| _mm256_loadu_ps(input.row(i).as_ptr())) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dct1d_8_flat(c: &mut [__m256; 8]) {
    let e0 = _mm256_add_ps(c[0], c[7]);
    let e1 = _mm256_add_ps(c[1], c[6]);
    let e2 = _mm256_add_ps(c[2], c[5]);
    let e3 = _mm256_add_ps(c[3], c[4]);

    let o0 = _mm256_mul_ps(_mm256_sub_ps(c[0], c[7]), _mm256_set1_ps(WC8[0]));
    let o1 = _mm256_mul_ps(_mm256_sub_ps(c[1], c[6]), _mm256_set1_ps(WC8[1]));
    let o2 = _mm256_mul_ps(_mm256_sub_ps(c[2], c[5]), _mm256_set1_ps(WC8[2]));
    let o3 = _mm256_mul_ps(_mm256_sub_ps(c[3], c[4]), _mm256_set1_ps(WC8[3]));

    let et0 = _mm256_add_ps(e0, e3);
    let et1 = _mm256_add_ps(e1, e2);
    let esum = _mm256_add_ps(et0, et1);
    let ediff = _mm256_sub_ps(et0, et1);
    let et2 = _mm256_mul_ps(_mm256_sub_ps(e0, e3), _mm256_set1_ps(WC4[0]));
    let et3 = _mm256_mul_ps(_mm256_sub_ps(e1, e2), _mm256_set1_ps(WC4[1]));
    let et2p = _mm256_add_ps(et2, et3);
    let et3p = _mm256_sub_ps(et2, et3);
    let et2pp = _mm256_fmadd_ps(et2p, _mm256_set1_ps(std::f32::consts::SQRT_2), et3p);
    let evens = [esum, et2pp, ediff, et3p];

    let ot0 = _mm256_add_ps(o0, o3);
    let ot1 = _mm256_add_ps(o1, o2);
    let osum = _mm256_add_ps(ot0, ot1);
    let odiff = _mm256_sub_ps(ot0, ot1);
    let ot2 = _mm256_mul_ps(_mm256_sub_ps(o0, o3), _mm256_set1_ps(WC4[0]));
    let ot3 = _mm256_mul_ps(_mm256_sub_ps(o1, o2), _mm256_set1_ps(WC4[1]));
    let ot2p = _mm256_add_ps(ot2, ot3);
    let ot3p = _mm256_sub_ps(ot2, ot3);
    let ot2pp = _mm256_fmadd_ps(ot2p, _mm256_set1_ps(std::f32::consts::SQRT_2), ot3p);
    let mut odds = [osum, ot2pp, odiff, ot3p];

    odds[0] = _mm256_fmadd_ps(odds[0], _mm256_set1_ps(std::f32::consts::SQRT_2), odds[1]);
    odds[1] = _mm256_add_ps(odds[1], odds[2]);
    odds[2] = _mm256_add_ps(odds[2], odds[3]);

    c[0] = evens[0];
    c[1] = odds[0];
    c[2] = evens[1];
    c[3] = odds[1];
    c[4] = evens[2];
    c[5] = odds[2];
    c[6] = evens[3];
    c[7] = odds[3];
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct8x8_avx2(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut rows = load(input);

    dct1d_8_flat(&mut rows);
    transpose_8x8(&mut rows);
    dct1d_8_flat(&mut rows);

    let scale = _mm256_set1_ps(1.0 / 64.0);
    for (k, row) in rows.iter().enumerate() {
        unsafe {
            _mm256_storeu_ps(output[k * 8..].as_mut_ptr(), _mm256_mul_ps(*row, scale));
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dct1d_16_flat(c: &mut [__m256; 16]) {
    let mut evens = [
        _mm256_add_ps(c[0], c[15]),
        _mm256_add_ps(c[1], c[14]),
        _mm256_add_ps(c[2], c[13]),
        _mm256_add_ps(c[3], c[12]),
        _mm256_add_ps(c[4], c[11]),
        _mm256_add_ps(c[5], c[10]),
        _mm256_add_ps(c[6], c[9]),
        _mm256_add_ps(c[7], c[8]),
    ];
    let mut odds = [
        _mm256_mul_ps(_mm256_sub_ps(c[0], c[15]), _mm256_set1_ps(WC16[0])),
        _mm256_mul_ps(_mm256_sub_ps(c[1], c[14]), _mm256_set1_ps(WC16[1])),
        _mm256_mul_ps(_mm256_sub_ps(c[2], c[13]), _mm256_set1_ps(WC16[2])),
        _mm256_mul_ps(_mm256_sub_ps(c[3], c[12]), _mm256_set1_ps(WC16[3])),
        _mm256_mul_ps(_mm256_sub_ps(c[4], c[11]), _mm256_set1_ps(WC16[4])),
        _mm256_mul_ps(_mm256_sub_ps(c[5], c[10]), _mm256_set1_ps(WC16[5])),
        _mm256_mul_ps(_mm256_sub_ps(c[6], c[9]), _mm256_set1_ps(WC16[6])),
        _mm256_mul_ps(_mm256_sub_ps(c[7], c[8]), _mm256_set1_ps(WC16[7])),
    ];

    dct1d_8_flat(&mut evens);
    dct1d_8_flat(&mut odds);

    odds[0] = _mm256_fmadd_ps(odds[0], _mm256_set1_ps(std::f32::consts::SQRT_2), odds[1]);
    odds[1] = _mm256_add_ps(odds[1], odds[2]);
    odds[2] = _mm256_add_ps(odds[2], odds[3]);
    odds[3] = _mm256_add_ps(odds[3], odds[4]);
    odds[4] = _mm256_add_ps(odds[4], odds[5]);
    odds[5] = _mm256_add_ps(odds[5], odds[6]);
    odds[6] = _mm256_add_ps(odds[6], odds[7]);

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

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct8x16_avx2(input: DctInput<'_, 16, 8>, output: &mut [f32; 128]) {
    let mut rows_lo: [__m256; 8] =
        std::array::from_fn(|k| unsafe { _mm256_loadu_ps(input.row(k).as_ptr()) });
    let mut rows_hi: [__m256; 8] =
        std::array::from_fn(|k| unsafe { _mm256_loadu_ps(input.row(k)[8..].as_ptr()) });

    transpose_8x8(&mut rows_lo);
    transpose_8x8(&mut rows_hi);

    let mut c = [_mm256_undefined_ps(); 16];
    c[0..8].copy_from_slice(&rows_lo);
    c[8..16].copy_from_slice(&rows_hi);

    dct1d_16_flat(&mut c);
    let mut cl: [__m256; 8] = c[..8].try_into().unwrap();
    let mut cr: [__m256; 8] = c[8..16].try_into().unwrap();
    transpose_8x8(&mut cl);
    transpose_8x8(&mut cr);
    dct1d_8_flat(&mut cl);
    dct1d_8_flat(&mut cr);

    let scale = _mm256_set1_ps(1.0 / 128.0);
    for m in 0..8 {
        let base = &mut output[m * 16..];
        unsafe {
            _mm256_storeu_ps(base.as_mut_ptr(), _mm256_mul_ps(cl[m], scale));
            _mm256_storeu_ps(base[8..].as_mut_ptr(), _mm256_mul_ps(cr[m], scale));
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct16x8_avx2(input: DctInput<'_, 8, 16>, output: &mut [f32; 128]) {
    let mut c: [__m256; 16] =
        std::array::from_fn(|v| unsafe { _mm256_loadu_ps(input.row(v).as_ptr()) });

    dct1d_16_flat(&mut c);

    let mut top: [__m256; 8] = c[0..8].try_into().unwrap();
    let mut bot: [__m256; 8] = c[8..16].try_into().unwrap();
    transpose_8x8(&mut top);
    transpose_8x8(&mut bot);
    dct1d_8_flat(&mut top);
    dct1d_8_flat(&mut bot);

    let scale = _mm256_set1_ps(1.0 / 128.0);
    for m in 0..8 {
        let base = &mut output[m * 16..];
        unsafe {
            _mm256_storeu_ps(base.as_mut_ptr(), _mm256_mul_ps(top[m], scale));
            _mm256_storeu_ps(base[8..].as_mut_ptr(), _mm256_mul_ps(bot[m], scale));
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct16x16_avx2(input: DctInput<'_, 16, 16>, output: &mut [f32; 256]) {
    // Two 8-col strips per pass through a column-major scratch keeps the live set
    // near 16 YMM instead of materializing all 32; row pass reloads contiguously.
    let mut scratch_uninit = MaybeUninit::<[f32; 256]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for g in 0..2 {
        let mut c: [__m256; 16] =
            std::array::from_fn(|r| unsafe { _mm256_loadu_ps(input.row(r)[g * 8..].as_ptr()) });
        dct1d_16_flat(&mut c);
        for t in 0..2 {
            let mut tile: [__m256; 8] = c[t * 8..t * 8 + 8].try_into().unwrap();
            transpose_8x8(&mut tile);
            for (j, v) in tile.iter().enumerate() {
                unsafe { _mm256_storeu_ps(dst.add((g * 8 + j) * 16 + t * 8), *v) };
            }
        }
    }
    let scale = _mm256_set1_ps(1.0 / 256.0);
    let scratch = unsafe { scratch_uninit.assume_init() };
    for g in 0..2 {
        let mut c: [__m256; 16] = std::array::from_fn(|col| unsafe {
            _mm256_loadu_ps(scratch[col * 16 + g * 8..].as_ptr())
        });
        dct1d_16_flat(&mut c);
        for u in 0..16 {
            unsafe {
                _mm256_storeu_ps(
                    output[u * 16 + g * 8..].as_mut_ptr(),
                    _mm256_mul_ps(c[u], scale),
                );
            }
        }
    }
}

/// Vectorized 32-point DCT-II over 8 columns (one per lane), mirroring the
/// scalar [`crate::dct::dct1d_32`]: even/odd split, recurse on two 16-point
/// halves via [`dct1d_16_flat`], then the odd-half combine. Verified lane-for-
/// lane identical to the scalar recursion.
#[target_feature(enable = "avx2,fma")]
fn dct1d_32_flat(c: &mut [__m256; 32]) {
    let mut evens = [_mm256_undefined_ps(); 16];
    let mut odds = [_mm256_undefined_ps(); 16];
    for i in 0..16 {
        evens[i] = _mm256_add_ps(c[i], c[31 - i]);
        odds[i] = _mm256_mul_ps(_mm256_sub_ps(c[i], c[31 - i]), _mm256_set1_ps(WC32[i]));
    }
    dct1d_16_flat(&mut evens);
    dct1d_16_flat(&mut odds);
    odds[0] = _mm256_fmadd_ps(odds[0], _mm256_set1_ps(std::f32::consts::SQRT_2), odds[1]);
    for i in 1..15 {
        odds[i] = _mm256_add_ps(odds[i], odds[i + 1]);
    }
    for i in 0..16 {
        c[2 * i] = evens[i];
        c[2 * i + 1] = odds[i];
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct32x32_avx2(input: DctInput<'_, 32, 32>, output: &mut [f32; 1024]) {
    // Column pass writes a transposed scratch (`[col*32 + vfreq]`) so the row pass
    // reloads contiguously instead of gathering 8 scalars per vector.
    let mut scratch_uninit = MaybeUninit::<[f32; 1024]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for g in 0..4 {
        let mut c: [__m256; 32] =
            std::array::from_fn(|r| unsafe { _mm256_loadu_ps(input.row(r)[g * 8..].as_ptr()) });
        dct1d_32_flat(&mut c);
        for t in 0..4 {
            let mut tile: [__m256; 8] = c[t * 8..t * 8 + 8].try_into().unwrap();
            transpose_8x8(&mut tile);
            for (j, v) in tile.iter().enumerate() {
                unsafe { _mm256_storeu_ps(dst.add((g * 8 + j) * 32 + t * 8), *v) };
            }
        }
    }
    let colt = unsafe { scratch_uninit.assume_init() };
    let scale = _mm256_set1_ps(1.0 / 1024.0);
    for g in 0..4 {
        let mut c: [__m256; 32] = std::array::from_fn(|col| unsafe {
            _mm256_loadu_ps(colt[col * 32 + g * 8..].as_ptr())
        });
        dct1d_32_flat(&mut c);
        for u in 0..32 {
            unsafe {
                _mm256_storeu_ps(
                    output[u * 32 + g * 8..].as_mut_ptr(),
                    _mm256_mul_ps(c[u], scale),
                );
            }
        }
    }
}

/// Vectorized 64-point DCT-II over eight independent columns.
#[target_feature(enable = "avx2,fma")]
fn dct1d_64_flat(c: &mut [__m256; 64]) {
    let mut evens = [_mm256_undefined_ps(); 32];
    let mut odds = [_mm256_undefined_ps(); 32];
    for i in 0..32 {
        evens[i] = _mm256_add_ps(c[i], c[63 - i]);
        odds[i] = _mm256_mul_ps(_mm256_sub_ps(c[i], c[63 - i]), _mm256_set1_ps(WC64[i]));
    }
    dct1d_32_flat(&mut evens);
    dct1d_32_flat(&mut odds);
    odds[0] = _mm256_fmadd_ps(odds[0], _mm256_set1_ps(std::f32::consts::SQRT_2), odds[1]);
    for i in 1..31 {
        odds[i] = _mm256_add_ps(odds[i], odds[i + 1]);
    }
    for i in 0..32 {
        c[2 * i] = evens[i];
        c[2 * i + 1] = odds[i];
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct64x64_avx2(input: DctInput<'_, 64, 64>, output: &mut [f32; 4096]) {
    // Column pass writes [column * 64 + vertical_frequency]. Both passes then
    // operate on contiguous eight-lane strips, avoiding gathers entirely.
    let mut scratch_uninit = MaybeUninit::<[f32; 4096]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for strip in 0..8 {
        let mut c: [__m256; 64] = std::array::from_fn(|row| unsafe {
            _mm256_loadu_ps(input.row(row)[strip * 8..].as_ptr())
        });
        dct1d_64_flat(&mut c);
        for tile_index in 0..8 {
            let mut tile: [__m256; 8] = c[tile_index * 8..tile_index * 8 + 8].try_into().unwrap();
            transpose_8x8(&mut tile);
            for (lane, value) in tile.iter().enumerate() {
                unsafe {
                    _mm256_storeu_ps(dst.add((strip * 8 + lane) * 64 + tile_index * 8), *value)
                };
            }
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = _mm256_set1_ps(1.0 / 4096.0);
    for strip in 0..8 {
        let mut c: [__m256; 64] = std::array::from_fn(|column| unsafe {
            _mm256_loadu_ps(scratch[column * 64 + strip * 8..].as_ptr())
        });
        dct1d_64_flat(&mut c);
        for u in 0..64 {
            unsafe {
                _mm256_storeu_ps(
                    output[u * 64 + strip * 8..].as_mut_ptr(),
                    _mm256_mul_ps(c[u], scale),
                )
            };
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct64x32_avx2(input: DctInput<'_, 32, 64>, output: &mut [f32; 2048]) {
    let mut scratch_uninit = MaybeUninit::<[f32; 2048]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for strip in 0..4 {
        let mut c: [__m256; 64] = std::array::from_fn(|row| unsafe {
            _mm256_loadu_ps(input.row(row)[strip * 8..].as_ptr())
        });
        dct1d_64_flat(&mut c);
        for tile_index in 0..8 {
            let mut tile: [__m256; 8] = c[tile_index * 8..tile_index * 8 + 8].try_into().unwrap();
            transpose_8x8(&mut tile);
            for (lane, value) in tile.iter().enumerate() {
                unsafe {
                    _mm256_storeu_ps(dst.add((strip * 8 + lane) * 64 + tile_index * 8), *value)
                };
            }
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = _mm256_set1_ps(1.0 / 2048.0);
    for strip in 0..8 {
        let mut c: [__m256; 32] = std::array::from_fn(|column| unsafe {
            _mm256_loadu_ps(scratch[column * 64 + strip * 8..].as_ptr())
        });
        dct1d_32_flat(&mut c);
        for u in 0..32 {
            unsafe {
                _mm256_storeu_ps(
                    output[u * 64 + strip * 8..].as_mut_ptr(),
                    _mm256_mul_ps(c[u], scale),
                )
            };
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct32x64_avx2(input: DctInput<'_, 64, 32>, output: &mut [f32; 2048]) {
    // Row DCT into [horizontal_frequency * 32 + row], then a 32-point
    // vertical DCT. Output is the natural [vertical_frequency][horizontal]
    // layout required by the shared rectangular quant matrix.
    let mut scratch_uninit = MaybeUninit::<[f32; 2048]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for row_strip in 0..4 {
        let mut c = [_mm256_undefined_ps(); 64];
        for column_tile in 0..8 {
            let mut tile: [__m256; 8] = std::array::from_fn(|lane| unsafe {
                _mm256_loadu_ps(input.row(row_strip * 8 + lane)[column_tile * 8..].as_ptr())
            });
            transpose_8x8(&mut tile);
            c[column_tile * 8..column_tile * 8 + 8].copy_from_slice(&tile);
        }
        dct1d_64_flat(&mut c);
        for u in 0..64 {
            unsafe { _mm256_storeu_ps(dst.add(u * 32 + row_strip * 8), c[u]) };
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = _mm256_set1_ps(1.0 / 2048.0);
    for column_strip in 0..8 {
        let mut c = [_mm256_undefined_ps(); 32];
        for row_tile in 0..4 {
            let mut tile: [__m256; 8] = std::array::from_fn(|lane| unsafe {
                _mm256_loadu_ps(scratch[(column_strip * 8 + lane) * 32 + row_tile * 8..].as_ptr())
            });
            transpose_8x8(&mut tile);
            c[row_tile * 8..row_tile * 8 + 8].copy_from_slice(&tile);
        }
        dct1d_32_flat(&mut c);
        for v in 0..32 {
            unsafe {
                _mm256_storeu_ps(
                    output[v * 64 + column_strip * 8..].as_mut_ptr(),
                    _mm256_mul_ps(c[v], scale),
                )
            };
        }
    }
}

const IS2: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn idct2_flat(a: __m256, b: __m256) -> (__m256, __m256) {
    let half = _mm256_set1_ps(0.5);
    (
        _mm256_mul_ps(_mm256_add_ps(a, b), half),
        _mm256_mul_ps(_mm256_sub_ps(a, b), half),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn inv_dct1d_4_flat(c: &mut [__m256; 4]) {
    let (t0, t1, mut t2, t3) = (c[0], c[2], c[1], c[3]);
    t2 = _mm256_mul_ps(_mm256_sub_ps(t2, t3), _mm256_set1_ps(IS2));
    let (n2, n3) = idct2_flat(t2, t3);
    let a2 = _mm256_mul_ps(n2, _mm256_set1_ps(INV_WC4[0]));
    let a3 = _mm256_mul_ps(n3, _mm256_set1_ps(INV_WC4[1]));
    let (m0, m1) = idct2_flat(t0, t1);
    let half = _mm256_set1_ps(0.5);
    c[0] = _mm256_mul_ps(_mm256_add_ps(m0, a2), half);
    c[3] = _mm256_mul_ps(_mm256_sub_ps(m0, a2), half);
    c[1] = _mm256_mul_ps(_mm256_add_ps(m1, a3), half);
    c[2] = _mm256_mul_ps(_mm256_sub_ps(m1, a3), half);
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn inv_dct1d_8_flat(c: &mut [__m256; 8]) {
    let mut t = [c[0]; 8];
    for i in 0..4 {
        t[i] = c[2 * i];
        t[4 + i] = c[2 * i + 1];
    }
    t[6] = _mm256_sub_ps(t[6], t[7]);
    t[5] = _mm256_sub_ps(t[5], t[6]);
    t[4] = _mm256_mul_ps(_mm256_sub_ps(t[4], t[5]), _mm256_set1_ps(IS2));
    let mut o = [t[4], t[5], t[6], t[7]];
    inv_dct1d_4_flat(&mut o);
    for i in 0..4 {
        o[i] = _mm256_mul_ps(o[i], _mm256_set1_ps(INV_WC8[i]));
    }
    let mut e = [t[0], t[1], t[2], t[3]];
    inv_dct1d_4_flat(&mut e);
    let half = _mm256_set1_ps(0.5);
    for i in 0..4 {
        c[i] = _mm256_mul_ps(_mm256_add_ps(e[i], o[i]), half);
        c[7 - i] = _mm256_mul_ps(_mm256_sub_ps(e[i], o[i]), half);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn inv_dct1d_16_flat(c: &mut [__m256; 16]) {
    let mut t = [c[0]; 16];
    for i in 0..8 {
        t[i] = c[2 * i];
        t[8 + i] = c[2 * i + 1];
    }
    for i in (9..=14).rev() {
        t[i] = _mm256_sub_ps(t[i], t[i + 1]);
    }
    t[8] = _mm256_mul_ps(_mm256_sub_ps(t[8], t[9]), _mm256_set1_ps(IS2));
    let mut o: [__m256; 8] = std::array::from_fn(|i| t[8 + i]);
    inv_dct1d_8_flat(&mut o);
    for i in 0..8 {
        o[i] = _mm256_mul_ps(o[i], _mm256_set1_ps(INV_WC16[i]));
    }
    let mut e: [__m256; 8] = std::array::from_fn(|i| t[i]);
    inv_dct1d_8_flat(&mut e);
    let half = _mm256_set1_ps(0.5);
    for i in 0..8 {
        c[i] = _mm256_mul_ps(_mm256_add_ps(e[i], o[i]), half);
        c[15 - i] = _mm256_mul_ps(_mm256_sub_ps(e[i], o[i]), half);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn inv_dct1d_32_flat(c: &mut [__m256; 32]) {
    let mut t = [c[0]; 32];
    for i in 0..16 {
        t[i] = c[2 * i];
        t[16 + i] = c[2 * i + 1];
    }
    for i in (17..=30).rev() {
        t[i] = _mm256_sub_ps(t[i], t[i + 1]);
    }
    t[16] = _mm256_mul_ps(_mm256_sub_ps(t[16], t[17]), _mm256_set1_ps(IS2));
    let mut o: [__m256; 16] = std::array::from_fn(|i| t[16 + i]);
    inv_dct1d_16_flat(&mut o);
    for i in 0..16 {
        o[i] = _mm256_mul_ps(o[i], _mm256_set1_ps(INV_WC32[i]));
    }
    let mut e: [__m256; 16] = std::array::from_fn(|i| t[i]);
    inv_dct1d_16_flat(&mut e);
    let half = _mm256_set1_ps(0.5);
    for i in 0..16 {
        c[i] = _mm256_mul_ps(_mm256_add_ps(e[i], o[i]), half);
        c[31 - i] = _mm256_mul_ps(_mm256_sub_ps(e[i], o[i]), half);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn inv_dct1d_64_flat(c: &mut [__m256; 64]) {
    let mut t = [c[0]; 64];
    for i in 0..32 {
        t[i] = c[2 * i];
        t[32 + i] = c[2 * i + 1];
    }
    for i in (33..=62).rev() {
        t[i] = _mm256_sub_ps(t[i], t[i + 1]);
    }
    t[32] = _mm256_mul_ps(_mm256_sub_ps(t[32], t[33]), _mm256_set1_ps(IS2));
    let mut odd: [__m256; 32] = std::array::from_fn(|i| t[32 + i]);
    inv_dct1d_32_flat(&mut odd);
    for i in 0..32 {
        odd[i] = _mm256_mul_ps(odd[i], _mm256_set1_ps(INV_WC64[i]));
    }
    let mut even: [__m256; 32] = std::array::from_fn(|i| t[i]);
    inv_dct1d_32_flat(&mut even);
    let half = _mm256_set1_ps(0.5);
    for i in 0..32 {
        c[i] = _mm256_mul_ps(_mm256_add_ps(even[i], odd[i]), half);
        c[63 - i] = _mm256_mul_ps(_mm256_sub_ps(even[i], odd[i]), half);
    }
}

macro_rules! horizontal_idct_pass_avx2 {
    ($tmp:expr, $out:expr, $h:literal, $w:literal, $inv_h:path) => {{
        for y8 in (0..$h).step_by(8) {
            let mut c = [_mm256_undefined_ps(); $w];
            for u8 in (0..$w).step_by(8) {
                let mut tile: [__m256; 8] = std::array::from_fn(|j| unsafe {
                    _mm256_loadu_ps($tmp.as_ptr().add((y8 + j) * $w + u8))
                });
                transpose_8x8(&mut tile);
                c[u8..u8 + 8].copy_from_slice(&tile);
            }
            $inv_h(&mut c);
            for x8 in (0..$w).step_by(8) {
                let mut tile: [__m256; 8] = c[x8..x8 + 8].try_into().unwrap();
                transpose_8x8(&mut tile);
                for (j, row) in tile.iter().enumerate() {
                    unsafe { _mm256_storeu_ps($out.as_mut_ptr().add((y8 + j) * $w + x8), *row) };
                }
            }
        }
    }};
}

macro_rules! inv_dct_natural_avx2 {
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path) => {
        #[target_feature(enable = "avx2,fma")]
        pub(crate) fn $name(coeff: DctInput<'_, $w, $h>, out: &mut [f32; $n]) {
            let mut scratch_uninit = MaybeUninit::<[f32; $n]>::uninit();
            let dst = scratch_uninit.as_mut_ptr() as *mut f32;
            let scale = _mm256_set1_ps($n as f32);
            for u8 in (0..$w).step_by(8) {
                let mut c: [__m256; $h] = std::array::from_fn(|v| unsafe {
                    _mm256_mul_ps(_mm256_loadu_ps(coeff.row(v)[u8..].as_ptr()), scale)
                });
                $inv_v(&mut c);
                for y in 0..$h {
                    unsafe { _mm256_storeu_ps(dst.add(y * $w + u8), c[y]) };
                }
            }
            let tmp = unsafe { scratch_uninit.assume_init() };
            horizontal_idct_pass_avx2!(tmp, out, $h, $w, $inv_h);
        }
    };
}

macro_rules! inv_dct_transposed_avx2 {
    ($name:ident, $n:literal, $h:literal, $w:literal, $inv_v:path, $inv_h:path) => {
        #[target_feature(enable = "avx2,fma")]
        pub(crate) fn $name(coeff: DctInput<'_, $h, $w>, out: &mut [f32; $n]) {
            let mut scratch_uninit = MaybeUninit::<[f32; $n]>::uninit();
            let dst = scratch_uninit.as_mut_ptr() as *mut f32;
            let scale = _mm256_set1_ps($n as f32);
            for u8 in (0..$w).step_by(8) {
                let mut c = [_mm256_undefined_ps(); $h];
                for v8 in (0..$h).step_by(8) {
                    let mut tile: [__m256; 8] = std::array::from_fn(|j| unsafe {
                        _mm256_loadu_ps(coeff.row(u8 + j)[v8..].as_ptr())
                    });
                    transpose_8x8(&mut tile);
                    for (j, v) in tile.iter().enumerate() {
                        c[v8 + j] = _mm256_mul_ps(*v, scale);
                    }
                }
                $inv_v(&mut c);
                for y in 0..$h {
                    unsafe { _mm256_storeu_ps(dst.add(y * $w + u8), c[y]) };
                }
            }
            let tmp = unsafe { scratch_uninit.assume_init() };
            horizontal_idct_pass_avx2!(tmp, out, $h, $w, $inv_h);
        }
    };
}

inv_dct_transposed_avx2!(
    inv_dct8x8_avx2,
    64,
    8,
    8,
    inv_dct1d_8_flat,
    inv_dct1d_8_flat
);
inv_dct_natural_avx2!(
    inv_dct8x16_avx2,
    128,
    8,
    16,
    inv_dct1d_8_flat,
    inv_dct1d_16_flat
);
inv_dct_transposed_avx2!(
    inv_dct16x8_avx2,
    128,
    16,
    8,
    inv_dct1d_16_flat,
    inv_dct1d_8_flat
);
inv_dct_transposed_avx2!(
    inv_dct16x16_avx2,
    256,
    16,
    16,
    inv_dct1d_16_flat,
    inv_dct1d_16_flat
);
inv_dct_natural_avx2!(
    inv_dct16x32_avx2,
    512,
    16,
    32,
    inv_dct1d_16_flat,
    inv_dct1d_32_flat
);
inv_dct_transposed_avx2!(
    inv_dct32x16_avx2,
    512,
    32,
    16,
    inv_dct1d_32_flat,
    inv_dct1d_16_flat
);
inv_dct_transposed_avx2!(
    inv_dct32x32_avx2,
    1024,
    32,
    32,
    inv_dct1d_32_flat,
    inv_dct1d_32_flat
);
inv_dct_transposed_avx2!(
    inv_dct64x64_avx2,
    4096,
    64,
    64,
    inv_dct1d_64_flat,
    inv_dct1d_64_flat
);
inv_dct_transposed_avx2!(
    inv_dct64x32_avx2,
    2048,
    64,
    32,
    inv_dct1d_64_flat,
    inv_dct1d_32_flat
);
inv_dct_natural_avx2!(
    inv_dct32x64_avx2,
    2048,
    32,
    64,
    inv_dct1d_32_flat,
    inv_dct1d_64_flat
);

/// 4-point DCT-II over 4 lanes (one per 4×4 quadrant), mirroring the scalar
/// [`crate::dct::dct1d_4`].
#[target_feature(enable = "avx2,fma")]
pub(super) fn dct1d_4_m128(c: &mut [__m128; 4]) {
    let t0 = _mm_add_ps(c[0], c[3]);
    let t1 = _mm_add_ps(c[1], c[2]);
    let e0 = _mm_add_ps(t0, t1);
    let e1 = _mm_sub_ps(t0, t1);
    let t2 = _mm_mul_ps(_mm_sub_ps(c[0], c[3]), _mm_set1_ps(WC4[0]));
    let t3 = _mm_mul_ps(_mm_sub_ps(c[1], c[2]), _mm_set1_ps(WC4[1]));
    let o0 = _mm_add_ps(t2, t3);
    let o1 = _mm_sub_ps(t2, t3);
    let m = _mm_fmadd_ps(o0, _mm_set1_ps(std::f32::consts::SQRT_2), o1);
    c[0] = e0;
    c[2] = e1;
    c[1] = m;
    c[3] = o1;
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) fn dct1d_8_m128(c: &mut [__m128; 8]) {
    let mut even = [
        _mm_add_ps(c[0], c[7]),
        _mm_add_ps(c[1], c[6]),
        _mm_add_ps(c[2], c[5]),
        _mm_add_ps(c[3], c[4]),
    ];
    let mut odd = [
        _mm_mul_ps(_mm_sub_ps(c[0], c[7]), _mm_set1_ps(WC8[0])),
        _mm_mul_ps(_mm_sub_ps(c[1], c[6]), _mm_set1_ps(WC8[1])),
        _mm_mul_ps(_mm_sub_ps(c[2], c[5]), _mm_set1_ps(WC8[2])),
        _mm_mul_ps(_mm_sub_ps(c[3], c[4]), _mm_set1_ps(WC8[3])),
    ];
    dct1d_4_m128(&mut even);
    dct1d_4_m128(&mut odd);
    odd[0] = _mm_fmadd_ps(odd[0], _mm_set1_ps(std::f32::consts::SQRT_2), odd[1]);
    odd[1] = _mm_add_ps(odd[1], odd[2]);
    odd[2] = _mm_add_ps(odd[2], odd[3]);
    for i in 0..4 {
        c[2 * i] = even[i];
        c[2 * i + 1] = odd[i];
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct4x4_avx2(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    // Gather q[r*4+c].lane[k] = input[(qy*4+r)*8 + (qx*4+c)], k = qy*2+qx.
    let mut q = [_mm_undefined_ps(); 16];
    for r in 0..4 {
        for col in 0..4 {
            q[r * 4 + col] = _mm_set_ps(
                input.row(4 + r)[4 + col], // k=3 (qy1,qx1)
                input.row(4 + r)[col],     // k=2 (qy1,qx0)
                input.row(r)[4 + col],     // k=1 (qy0,qx1)
                input.row(r)[col],         // k=0 (qy0,qx0)
            );
        }
    }
    // Row DCT.
    for r in 0..4 {
        let mut row = [q[r * 4], q[r * 4 + 1], q[r * 4 + 2], q[r * 4 + 3]];
        dct1d_4_m128(&mut row);
        for col in 0..4 {
            q[r * 4 + col] = row[col];
        }
    }
    // Column DCT (×1/16) → d[x*4+i] = colDCT freq i of column x.
    let inv16 = _mm_set1_ps(1.0 / 16.0);
    let mut d = [[0.0f32; 4]; 16];
    for col in 0..4 {
        let mut cc = [q[col], q[4 + col], q[8 + col], q[12 + col]];
        dct1d_4_m128(&mut cc);
        for i in 0..4 {
            unsafe { _mm_storeu_ps(d[col * 4 + i].as_mut_ptr(), _mm_mul_ps(cc[i], inv16)) };
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

/// Vectorized scalar `dct1d_4` across the 8 lanes of four `__m256` registers
/// (one "row" of the recursion per register). Mirrors `dct::dct1d_4` exactly.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn dct1d_4_flat(c: &mut [__m256; 4]) {
    // even half: dct1d_2(c0+c3, c1+c2)
    let s0 = _mm256_add_ps(c[0], c[3]);
    let s1 = _mm256_add_ps(c[1], c[2]);
    let e0 = _mm256_add_ps(s0, s1);
    let e1 = _mm256_sub_ps(s0, s1);
    // odd half: dct1d_2((c0-c3)*WC4[0], (c1-c2)*WC4[1]); then t2 = t2*SQRT2 + t3
    let d2 = _mm256_mul_ps(_mm256_sub_ps(c[0], c[3]), _mm256_set1_ps(WC4[0]));
    let d3 = _mm256_mul_ps(_mm256_sub_ps(c[1], c[2]), _mm256_set1_ps(WC4[1]));
    let osum = _mm256_add_ps(d2, d3);
    let odiff = _mm256_sub_ps(d2, d3);
    let o0 = _mm256_fmadd_ps(osum, _mm256_set1_ps(std::f32::consts::SQRT_2), odiff);
    c[0] = e0;
    c[1] = o0;
    c[2] = e1;
    c[3] = odiff;
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct4x8_avx2(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let rows = load(input);
    let mut top: [__m256; 4] = [rows[0], rows[1], rows[2], rows[3]];
    let mut bot: [__m256; 4] = [rows[4], rows[5], rows[6], rows[7]];
    dct1d_4_flat(&mut top);
    dct1d_4_flat(&mut bot);
    let mut r: [__m256; 8] = [
        top[0], top[1], top[2], top[3], bot[0], bot[1], bot[2], bot[3],
    ];
    transpose_8x8(&mut r);
    dct1d_8_flat(&mut r);
    transpose_8x8(&mut r);

    let scale = _mm256_set1_ps(1.0 / 32.0);
    let mut buf = [0.0f32; 64];
    for k in 0..8 {
        unsafe {
            _mm256_storeu_ps(buf[k * 8..].as_mut_ptr(), _mm256_mul_ps(r[k], scale));
        }
    }
    // buf[k*8 + hf]: k = vf for top (k<4) / bottom (k>=4), interleaved by row.
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

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct8x4_avx2(input: DctInput<'_, 8, 8>, output: &mut [f32; 64]) {
    let mut rows = load(input);
    dct1d_8_flat(&mut rows);
    transpose_8x8(&mut rows); // rows[col].lane[vf]
    let mut left: [__m256; 4] = [rows[0], rows[1], rows[2], rows[3]];
    let mut right: [__m256; 4] = [rows[4], rows[5], rows[6], rows[7]];
    dct1d_4_flat(&mut left);
    dct1d_4_flat(&mut right);

    let scale = _mm256_set1_ps(1.0 / 32.0);
    let mut lb = [0.0f32; 32];
    let mut rb = [0.0f32; 32];
    for hf in 0..4 {
        unsafe {
            _mm256_storeu_ps(lb[hf * 8..].as_mut_ptr(), _mm256_mul_ps(left[hf], scale));
            _mm256_storeu_ps(rb[hf * 8..].as_mut_ptr(), _mm256_mul_ps(right[hf], scale));
        }
    }
    for hf in 0..4 {
        for vf in 0..8 {
            output[(hf * 2) * 8 + vf] = lb[hf * 8 + vf];
            output[(1 + hf * 2) * 8 + vf] = rb[hf * 8 + vf];
        }
    }
    let b0 = output[0];
    let b1 = output[8];
    output[0] = (b0 + b1) * 0.5;
    output[8] = (b0 - b1) * 0.5;
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct32x16_avx2(input: DctInput<'_, 16, 32>, output: &mut [f32; 512]) {
    // Column pass, then transpose each vfreq tile so the row pass loads
    // contiguously instead of gathering 8 scalars per vector.
    let mut cols = [[_mm256_undefined_ps(); 16]; 4];
    for g in 0..2 {
        let mut c: [__m256; 32] =
            std::array::from_fn(|r| unsafe { _mm256_loadu_ps(input.row(r)[g * 8..].as_ptr()) });
        dct1d_32_flat(&mut c);
        for t in 0..4 {
            let mut tile: [__m256; 8] = c[t * 8..t * 8 + 8].try_into().unwrap();
            transpose_8x8(&mut tile);
            cols[t][g * 8..g * 8 + 8].copy_from_slice(&tile);
        }
    }
    let scale = _mm256_set1_ps(1.0 / 512.0);
    for (t, tile) in cols.iter_mut().enumerate() {
        dct1d_16_flat(tile);
        for u in 0..16 {
            unsafe {
                _mm256_storeu_ps(
                    output[u * 32 + t * 8..].as_mut_ptr(),
                    _mm256_mul_ps(tile[u], scale),
                );
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct16x32_avx2(input: DctInput<'_, 32, 16>, output: &mut [f32; 512]) {
    // Row pass (32-pt) over 8-row strips into hfreq-major scratch, then column
    // pass (16-pt); both use transposed contiguous loads instead of scalar gathers.
    let mut scratch_uninit = MaybeUninit::<[f32; 512]>::uninit();
    let dst = scratch_uninit.as_mut_ptr() as *mut f32;
    for g in 0..2 {
        let mut c = [_mm256_undefined_ps(); 32];
        for ct in 0..4 {
            let mut tile: [__m256; 8] = std::array::from_fn(|j| unsafe {
                _mm256_loadu_ps(input.row(g * 8 + j)[ct * 8..].as_ptr())
            });
            transpose_8x8(&mut tile);
            c[ct * 8..ct * 8 + 8].copy_from_slice(&tile);
        }
        dct1d_32_flat(&mut c);
        for u in 0..32 {
            unsafe { _mm256_storeu_ps(dst.add(u * 16 + g * 8), c[u]) };
        }
    }
    let scratch = unsafe { scratch_uninit.assume_init() };
    let scale = _mm256_set1_ps(1.0 / 512.0);
    for g in 0..4 {
        let mut c = [_mm256_undefined_ps(); 16];
        for rt in 0..2 {
            let mut tile: [__m256; 8] = std::array::from_fn(|j| unsafe {
                _mm256_loadu_ps(scratch[(g * 8 + j) * 16 + rt * 8..].as_ptr())
            });
            transpose_8x8(&mut tile);
            c[rt * 8..rt * 8 + 8].copy_from_slice(&tile);
        }
        dct1d_16_flat(&mut c);
        for v in 0..16 {
            unsafe {
                _mm256_storeu_ps(
                    output[v * 32 + g * 8..].as_mut_ptr(),
                    _mm256_mul_ps(c[v], scale),
                );
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dc_idct4_avx2(v: __m128) -> __m128 {
    let v0 = _mm_shuffle_ps::<0x00>(v, v);
    let v1 = _mm_shuffle_ps::<0x55>(v, v);
    let v2 = _mm_shuffle_ps::<0xaa>(v, v);
    let v3 = _mm_shuffle_ps::<0xff>(v, v);

    let e0 = _mm_add_ps(v0, v2);
    let e1 = _mm_sub_ps(v0, v2);
    let t2 = _mm_mul_ps(v1, _mm_set1_ps(std::f32::consts::SQRT_2));
    let t3 = _mm_add_ps(v3, v1);
    let o0 = _mm_add_ps(t2, t3);
    let o1 = _mm_sub_ps(t2, t3);

    let even01 = _mm_unpacklo_ps(e0, e1);
    let odd01 = _mm_unpacklo_ps(o0, o1);
    let even = _mm_shuffle_ps::<0xb4>(even01, even01);
    let odd = _mm_shuffle_ps::<0xb4>(odd01, odd01);
    let weights = _mm_setr_ps(WC4[0], WC4[1], -WC4[1], -WC4[0]);
    _mm_fmadd_ps(odd, weights, even)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn transpose_4x4_avx2(rows: &mut [__m128; 4]) {
    let t0 = _mm_unpacklo_ps(rows[0], rows[1]);
    let t1 = _mm_unpackhi_ps(rows[0], rows[1]);
    let t2 = _mm_unpacklo_ps(rows[2], rows[3]);
    let t3 = _mm_unpackhi_ps(rows[2], rows[3]);
    rows[0] = _mm_movelh_ps(t0, t2);
    rows[1] = _mm_movehl_ps(t2, t0);
    rows[2] = _mm_movelh_ps(t1, t3);
    rows[3] = _mm_movehl_ps(t3, t1);
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dc_from_dct32x32_avx2(coeffs: &[f32; 1024], dc: &mut [f32; 16]) {
    let resample = unsafe { _mm_loadu_ps(RESAMPLE_SCALE_32_TO_4.as_ptr()) };
    let mut rows: [__m128; 4] = std::array::from_fn(|a| {
        let coeff = unsafe { _mm_loadu_ps(coeffs[a * 32..].as_ptr()) };
        let scaled_a = _mm_mul_ps(coeff, _mm_set1_ps(RESAMPLE_SCALE_32_TO_4[a]));
        dc_idct4_avx2(_mm_mul_ps(scaled_a, resample))
    });
    transpose_4x4_avx2(&mut rows);
    for (bb, row) in rows.iter().enumerate() {
        unsafe { _mm_storeu_ps(dc[bb * 4..].as_mut_ptr(), dc_idct4_avx2(*row)) };
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dc_from_dct_rect_avx2(coeffs: &[f32; 512]) -> (__m128, __m128) {
    let resample = unsafe { _mm_loadu_ps(RESAMPLE_SCALE_32_TO_4.as_ptr()) };
    let row0 = unsafe { _mm_loadu_ps(coeffs.as_ptr()) };
    let row1 = unsafe { _mm_loadu_ps(coeffs[32..].as_ptr()) };
    let row0 = dc_idct4_avx2(_mm_mul_ps(row0, resample));
    let row1 = dc_idct4_avx2(_mm_mul_ps(
        _mm_mul_ps(row1, _mm_set1_ps(RESAMPLE_SCALE_16_TO_2[1])),
        resample,
    ));
    (_mm_add_ps(row0, row1), _mm_sub_ps(row0, row1))
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dc_from_dct32x16_avx2(coeffs: &[f32; 512], dc: &mut [f32; 8]) {
    let (sum, diff) = dc_from_dct_rect_avx2(coeffs);
    unsafe {
        _mm_storeu_ps(dc.as_mut_ptr(), _mm_unpacklo_ps(sum, diff));
        _mm_storeu_ps(dc[4..].as_mut_ptr(), _mm_unpackhi_ps(sum, diff));
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dc_from_dct16x32_avx2(coeffs: &[f32; 512], dc: &mut [f32; 8]) {
    let (sum, diff) = dc_from_dct_rect_avx2(coeffs);
    unsafe {
        _mm_storeu_ps(dc.as_mut_ptr(), sum);
        _mm_storeu_ps(dc[4..].as_mut_ptr(), diff);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dc_from_dct64x64_avx2(coeffs: &[f32; 4096], dc: &mut [f32; 64]) {
    let resample = unsafe { _mm256_loadu_ps(RESAMPLE_SCALE_64_TO_8.as_ptr()) };
    let mut rows: [__m256; 8] = std::array::from_fn(|y| {
        let coeff = unsafe { _mm256_loadu_ps(coeffs[y * 64..].as_ptr()) };
        _mm256_mul_ps(
            _mm256_mul_ps(coeff, resample),
            _mm256_set1_ps(64.0 * RESAMPLE_SCALE_64_TO_8[y]),
        )
    });
    inv_dct1d_8_flat(&mut rows);
    transpose_8x8(&mut rows);
    inv_dct1d_8_flat(&mut rows);
    for (y, row) in rows.iter().enumerate() {
        unsafe { _mm256_storeu_ps(dc[y * 8..].as_mut_ptr(), *row) };
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dc_from_dct64x32_normalized_avx2(coeffs: &[f32; 2048]) -> [__m256; 8] {
    let resample = unsafe { _mm256_loadu_ps(RESAMPLE_SCALE_64_TO_8.as_ptr()) };
    let mut vertical: [__m256; 4] = std::array::from_fn(|y| {
        let coeff = unsafe { _mm256_loadu_ps(coeffs[y * 64..].as_ptr()) };
        _mm256_mul_ps(
            _mm256_mul_ps(coeff, resample),
            _mm256_set1_ps(32.0 * RESAMPLE_SCALE_32_TO_4[y]),
        )
    });
    inv_dct1d_4_flat(&mut vertical);

    let mut rows = [_mm256_setzero_ps(); 8];
    rows[..4].copy_from_slice(&vertical);
    transpose_8x8(&mut rows);
    inv_dct1d_8_flat(&mut rows);
    rows
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dc_from_dct64x32_avx2(coeffs: &[f32; 2048], dc: &mut [f32; 32]) {
    let rows = dc_from_dct64x32_normalized_avx2(coeffs);
    for (y, row) in rows.iter().enumerate() {
        unsafe { _mm_storeu_ps(dc[y * 4..].as_mut_ptr(), _mm256_castps256_ps128(*row)) };
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dc_from_dct32x64_avx2(coeffs: &[f32; 2048], dc: &mut [f32; 32]) {
    let mut rows = dc_from_dct64x32_normalized_avx2(coeffs);
    transpose_8x8(&mut rows);
    for (y, row) in rows[..4].iter().enumerate() {
        unsafe { _mm256_storeu_ps(dc[y * 8..].as_mut_ptr(), *row) };
    }
}

#[cfg(test)]
mod tests {
    use crate::dct::DctInput;

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
    fn test_dc_from_dct_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::{
            dc_from_dct16x32_avx2, dc_from_dct32x16_avx2, dc_from_dct32x32_avx2,
            dc_from_dct32x64_avx2, dc_from_dct64x32_avx2, dc_from_dct64x64_avx2,
        };
        use crate::dct::{
            dc_from_dct16x32, dc_from_dct32x16, dc_from_dct32x32, dc_from_dct32x64,
            dc_from_dct64x32, dc_from_dct64x64,
        };

        for seed in 0u64..32 {
            let coeffs = fill::<1024>(seed.wrapping_add(0xdc32_0032));
            let mut got = [0.0f32; 16];
            let mut want = [0.0f32; 16];
            unsafe { dc_from_dct32x32_avx2(&coeffs, &mut got) };
            dc_from_dct32x32(&coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct32x32 seed={seed}"));

            let coeffs: &[f32; 512] = coeffs.first_chunk::<512>().unwrap();
            let mut got = [0.0f32; 8];
            let mut want = [0.0f32; 8];
            unsafe { dc_from_dct32x16_avx2(coeffs, &mut got) };
            dc_from_dct32x16(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct32x16 seed={seed}"));

            unsafe { dc_from_dct16x32_avx2(coeffs, &mut got) };
            dc_from_dct16x32(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct16x32 seed={seed}"));

            let coeffs = fill::<4096>(seed.wrapping_add(0xdc64_0064));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dc_from_dct64x64_avx2(&coeffs, &mut got) };
            dc_from_dct64x64(&coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct64x64 seed={seed}"));

            let coeffs = coeffs.first_chunk::<2048>().unwrap();
            let mut got = [0.0f32; 32];
            let mut want = [0.0f32; 32];
            unsafe { dc_from_dct64x32_avx2(coeffs, &mut got) };
            dc_from_dct64x32(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct64x32 seed={seed}"));

            unsafe { dc_from_dct32x64_avx2(coeffs, &mut got) };
            dc_from_dct32x64(coeffs, &mut want);
            assert_close(&got, &want, &format!("dc_from_dct32x64 seed={seed}"));
        }
    }

    fn assert_inverse_matches_scalar<const N: usize, const W: usize, const H: usize>(
        scalar: for<'a> fn(DctInput<'a, W, H>, &mut [f32; N]),
        avx2: for<'a> unsafe fn(DctInput<'a, W, H>, &mut [f32; N]),
        label: &str,
    ) {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
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
            cases.push(fill(0xa7_c2_0000 + seed));
        }

        for (case, input) in cases.iter().enumerate() {
            let mut got = [0.0f32; N];
            let mut want = [0.0f32; N];
            let stride = W + 3;
            let mut strided = vec![f32::NAN; H * stride];
            for y in 0..H {
                strided[y * stride..y * stride + W].copy_from_slice(&input[y * W..y * W + W]);
            }
            unsafe { avx2(DctInput::new(&strided, stride), &mut got) };
            scalar(DctInput::from_flat(input), &mut want);
            assert_close(&got, &want, &format!("{label} case={case}"));
        }
    }

    #[test]
    fn test_dct16x16_avx2_vs_scalar_random() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct16x16_avx2;
        use crate::dct::dct16x16_scalar;
        for seed in 0u64..32 {
            let input: [f32; 256] = fill(seed.wrapping_add(0xf00d));
            let mut got = [0.0f32; 256];
            let mut want = [0.0f32; 256];
            unsafe { dct16x16_avx2(DctInput::from_flat(&input), &mut got) };
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 seed={seed}"));
        }
    }

    #[test]
    fn test_dct32x32_avx2_vs_scalar_random() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct32x32_avx2;
        use crate::dct::dct32x32_scalar;
        for seed in 0u64..16 {
            let input: [f32; 1024] = fill(seed.wrapping_add(0x3232));
            let mut got = [0.0f32; 1024];
            let mut want = [0.0f32; 1024];
            unsafe { dct32x32_avx2(DctInput::from_flat(&input), &mut got) };
            dct32x32_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct32x32 seed={seed}"));
        }
    }

    #[test]
    fn test_dct32x16_avx2_vs_scalar_random() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct32x16_avx2;
        use crate::dct::dct32x16_scalar;
        for seed in 0u64..16 {
            let input: [f32; 512] = fill(seed.wrapping_add(0x3216));
            let mut got = [0.0f32; 512];
            let mut want = [0.0f32; 512];
            unsafe { dct32x16_avx2(DctInput::from_flat(&input), &mut got) };
            dct32x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct32x16 seed={seed}"));
        }
    }

    #[test]
    fn test_dct16x32_avx2_vs_scalar_random() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct16x32_avx2;
        use crate::dct::dct16x32_scalar;
        for seed in 0u64..16 {
            let input: [f32; 512] = fill(seed.wrapping_add(0x1632));
            let mut got = [0.0f32; 512];
            let mut want = [0.0f32; 512];
            unsafe { dct16x32_avx2(DctInput::from_flat(&input), &mut got) };
            dct16x32_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x32 seed={seed}"));
        }
    }

    #[test]
    fn test_dct4x4_avx2_vs_scalar_random() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct4x4_avx2;
        use crate::dct::dct4x4_scalar;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x4a4));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct4x4_avx2(DctInput::from_flat(&input), &mut got) };
            dct4x4_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct4x4 seed={seed}"));
        }
    }

    #[test]
    fn test_dct4x8_avx2_vs_scalar_random() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct4x8_avx2;
        use crate::dct::dct4x8_scalar;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x4a8));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct4x8_avx2(DctInput::from_flat(&input), &mut got) };
            dct4x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct4x8 seed={seed}"));
        }
    }

    #[test]
    fn test_dct8x4_avx2_vs_scalar_random() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct8x4_avx2;
        use crate::dct::dct8x4_scalar;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed.wrapping_add(0x8a4));
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct8x4_avx2(DctInput::from_flat(&input), &mut got) };
            dct8x4_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x4 seed={seed}"));
        }
    }

    #[test]
    fn test_dct16x16_avx2_dc_only() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct16x16_avx2;
        use crate::dct::dct16x16_scalar;
        let input = [0.5f32; 256];
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        unsafe { dct16x16_avx2(DctInput::from_flat(&input), &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 dc-only");
    }

    #[test]
    fn test_dct16x16_avx2_zero() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct16x16_avx2;
        use crate::dct::dct16x16_scalar;
        let input = [0.0f32; 256];
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        unsafe { dct16x16_avx2(DctInput::from_flat(&input), &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 zero");
    }

    #[test]
    fn test_dct16x16_avx2_basis_vectors() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct16x16_avx2;
        use crate::dct::dct16x16_scalar;
        for k in 0..256 {
            let mut input = [0.0f32; 256];
            input[k] = 1.0;
            let mut got = [0.0f32; 256];
            let mut want = [0.0f32; 256];
            unsafe { dct16x16_avx2(DctInput::from_flat(&input), &mut got) };
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 basis[{k}]"));
        }
    }

    #[test]
    fn test_dct16x16_avx2_linearity() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct16x16_avx2;
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
            dct16x16_avx2(DctInput::from_flat(&a), &mut da);
            dct16x16_avx2(DctInput::from_flat(&b), &mut db);
            dct16x16_avx2(DctInput::from_flat(&sum), &mut dsum);
        }
        let expected: Vec<f32> = (0..256).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct16x16 linearity");
    }

    #[test]
    fn test_dct16x16_avx2_extreme_values() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::avx::dct16x16_avx2;
        use crate::dct::dct16x16_scalar;
        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut got = [0.0f32; 256];
        let mut want = [0.0f32; 256];
        unsafe { dct16x16_avx2(DctInput::from_flat(&input), &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 alternating +-1");
    }

    #[test]
    fn test_idct8x8_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct8x8,
            crate::avx::inv_dct8x8_avx2,
            "idct8x8",
        );
    }

    #[test]
    fn test_idct8x16_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct8x16,
            crate::avx::inv_dct8x16_avx2,
            "idct8x16",
        );
    }

    #[test]
    fn test_idct16x8_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct16x8,
            crate::avx::inv_dct16x8_avx2,
            "idct16x8",
        );
    }

    #[test]
    fn test_idct16x16_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct16x16,
            crate::avx::inv_dct16x16_avx2,
            "idct16x16",
        );
    }

    #[test]
    fn test_idct16x32_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct16x32,
            crate::avx::inv_dct16x32_avx2,
            "idct16x32",
        );
    }

    #[test]
    fn test_idct32x16_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct32x16,
            crate::avx::inv_dct32x16_avx2,
            "idct32x16",
        );
    }

    #[test]
    fn test_idct32x32_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct32x32,
            crate::avx::inv_dct32x32_avx2,
            "idct32x32",
        );
    }

    #[test]
    fn test_idct64x64_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct64x64,
            crate::avx::inv_dct64x64_avx2,
            "idct64x64",
        );
    }

    #[test]
    fn test_idct64x32_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct64x32,
            crate::avx::inv_dct64x32_avx2,
            "idct64x32",
        );
    }

    #[test]
    fn test_idct32x64_avx2_matches_scalar() {
        assert_inverse_matches_scalar(
            crate::dct::inv_dct32x64,
            crate::avx::inv_dct32x64_avx2,
            "idct32x64",
        );
    }
}
