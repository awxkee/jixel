use crate::dct::{WC4, WC8};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
pub(super) fn transpose_4x4(rows: &mut [__m128; 4]) {
    let t0 = _mm_unpacklo_ps(rows[0], rows[1]);
    let t1 = _mm_unpackhi_ps(rows[0], rows[1]);
    let t2 = _mm_unpacklo_ps(rows[2], rows[3]);
    let t3 = _mm_unpackhi_ps(rows[2], rows[3]);
    rows[0] = _mm_movelh_ps(t0, t2);
    rows[1] = _mm_movehl_ps(t2, t0);
    rows[2] = _mm_movelh_ps(t1, t3);
    rows[3] = _mm_movehl_ps(t3, t1);
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(super) fn dct1d_4(c: &mut [__m128; 4]) {
    let s0 = _mm_add_ps(c[0], c[3]);
    let s1 = _mm_add_ps(c[1], c[2]);
    let d2 = _mm_mul_ps(_mm_sub_ps(c[0], c[3]), _mm_set1_ps(WC4[0]));
    let d3 = _mm_mul_ps(_mm_sub_ps(c[1], c[2]), _mm_set1_ps(WC4[1]));
    let osum = _mm_add_ps(d2, d3);
    let odiff = _mm_sub_ps(d2, d3);
    c[0] = _mm_add_ps(s0, s1);
    c[1] = _mm_add_ps(
        _mm_mul_ps(osum, _mm_set1_ps(std::f32::consts::SQRT_2)),
        odiff,
    );
    c[2] = _mm_sub_ps(s0, s1);
    c[3] = odiff;
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(super) fn dct1d_8(c: &mut [__m128; 8]) {
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
    dct1d_4(&mut even);
    dct1d_4(&mut odd);
    odd[0] = _mm_add_ps(
        _mm_mul_ps(odd[0], _mm_set1_ps(std::f32::consts::SQRT_2)),
        odd[1],
    );
    odd[1] = _mm_add_ps(odd[1], odd[2]);
    odd[2] = _mm_add_ps(odd[2], odd[3]);
    for i in 0..4 {
        c[2 * i] = even[i];
        c[2 * i + 1] = odd[i];
    }
}
