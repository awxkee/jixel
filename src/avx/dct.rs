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
use crate::dct::{WC4, WC8};
use std::arch::x86_64::*;

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
    c[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    c[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    c[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    c[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    c[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    c[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    c[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    c[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

#[inline]
#[target_feature(enable = "avx2")]
fn load(input: &[f32; 64]) -> [__m256; 8] {
    unsafe { std::array::from_fn(|i| _mm256_loadu_ps(input[i * 8..].as_ptr())) }
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

    // DCT4 on evens
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

    // DCT4 on odds
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
pub(crate) fn dct8x8_avx2(input: &[f32; 64], output: &mut [f32; 64]) {
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
