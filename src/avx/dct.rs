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

    // ── Interleave even/odd into output ───────────────────────────────────────
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
pub(crate) fn dct8x16_avx2(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut rows_lo: [__m256; 8] =
        std::array::from_fn(|k| unsafe { _mm256_loadu_ps(input[k * 16..].as_ptr()) });
    let mut rows_hi: [__m256; 8] =
        std::array::from_fn(|k| unsafe { _mm256_loadu_ps(input[k * 16 + 8..].as_ptr()) });

    transpose_8x8(&mut rows_lo);
    transpose_8x8(&mut rows_hi);

    let mut c = [_mm256_undefined_ps(); 16];
    c[0..8].copy_from_slice(&rows_lo);
    c[8..16].copy_from_slice(&rows_hi);

    dct1d_16_flat(&mut c);
    let mut cl: [__m256; 8] = c[0..8].try_into().unwrap();
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
pub(crate) fn dct16x8_avx2(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut c: [__m256; 16] =
        std::array::from_fn(|v| unsafe { _mm256_loadu_ps(input[v * 8..].as_ptr()) });

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

// ─── dct16x16 (AVX2) ──────────────────────────────────────────────────────────
//
// input  layout: input[row * 16 + col],  row, col ∈ 0..16
// output layout: output[u * 16 + v],     u = col-freq, v = row-freq
//
// Uses 12 × transpose_8x8 and 32 contiguous 256-bit stores — no scatter.
//
//  Phase 1 (col-DCT):
//   Load 4 sub-blocks → transpose each → assemble c_top/c_bot →
//   dct1d_16_flat × 2 → split → transpose × 4
//
//  Phase 2 (row-DCT):
//   Assemble d_a/d_b → dct1d_16_flat × 2 → split → transpose × 4 →
//   scale and store contiguously

#[target_feature(enable = "avx2,fma")]
pub(crate) fn dct16x16_avx2(input: &[f32; 256], output: &mut [f32; 256]) {
    unsafe {
        // ── Phase 1: col-DCT-16 ───────────────────────────────────────────────

        // Load four 8×8 sub-blocks.
        let mut top_lo: [__m256; 8] =   // rows 0..8,  cols 0..8
            std::array::from_fn(|k| _mm256_loadu_ps(input[k * 16..].as_ptr()));
        let mut bot_lo: [__m256; 8] =   // rows 8..16, cols 0..8
            std::array::from_fn(|k| _mm256_loadu_ps(input[(k + 8) * 16..].as_ptr()));
        let mut top_hi: [__m256; 8] =   // rows 0..8,  cols 8..16
            std::array::from_fn(|k| _mm256_loadu_ps(input[k * 16 + 8..].as_ptr()));
        let mut bot_hi: [__m256; 8] =   // rows 8..16, cols 8..16
            std::array::from_fn(|k| _mm256_loadu_ps(input[(k + 8) * 16 + 8..].as_ptr()));

        // Transpose: after this block[col].lane[row] = input[row][col].
        transpose_8x8(&mut top_lo);
        transpose_8x8(&mut bot_lo);
        transpose_8x8(&mut top_hi);
        transpose_8x8(&mut bot_hi);

        // Assemble 16-step arrays: index = original col (time-step), lane = row.
        //   c_top[k].lane[r] = input[r][k]     r = 0..8
        //   c_bot[k].lane[r] = input[r+8][k]   r = 0..8
        let mut c_top = [_mm256_undefined_ps(); 16];
        let mut c_bot = [_mm256_undefined_ps(); 16];
        c_top[0..8].copy_from_slice(&top_lo);
        c_top[8..16].copy_from_slice(&top_hi);
        c_bot[0..8].copy_from_slice(&bot_lo);
        c_bot[8..16].copy_from_slice(&bot_hi);

        // Col-DCT-16: for each lane (row 0..8), DCT-16 across the 16 cols.
        dct1d_16_flat(&mut c_top);
        dct1d_16_flat(&mut c_bot);

        // Split into col-freq halves and transpose for the row-DCT pass.
        let mut tlo: [__m256; 8] = c_top[0..8].try_into().unwrap();
        let mut thi: [__m256; 8] = c_top[8..16].try_into().unwrap();
        let mut blo: [__m256; 8] = c_bot[0..8].try_into().unwrap();
        let mut bhi: [__m256; 8] = c_bot[8..16].try_into().unwrap();
        transpose_8x8(&mut tlo);
        transpose_8x8(&mut thi);
        transpose_8x8(&mut blo);
        transpose_8x8(&mut bhi);

        // ── Phase 2: row-DCT-16 ───────────────────────────────────────────────

        // Assemble: index = row (time-step 0..16), lane = col-freq (0..8 or 8..16).
        //   d_a rows 0..8  from tlo, rows 8..16 from blo  → col-freqs 0..8  in lanes
        //   d_b rows 0..8  from thi, rows 8..16 from bhi  → col-freqs 8..16 in lanes
        let mut d_a = [_mm256_undefined_ps(); 16];
        let mut d_b = [_mm256_undefined_ps(); 16];
        d_a[0..8].copy_from_slice(&tlo);
        d_a[8..16].copy_from_slice(&blo);
        d_b[0..8].copy_from_slice(&thi);
        d_b[8..16].copy_from_slice(&bhi);

        // Row-DCT-16: for each lane (col-freq 0..8), DCT-16 across the 16 rows.
        dct1d_16_flat(&mut d_a);
        dct1d_16_flat(&mut d_b);

        // ── Store via final transpose → contiguous writes ─────────────────────
        //
        // After row-DCT:
        //   d_a[u].lane[v] = output[v*16 + u] * 256   u = row-freq 0..16, v = col-freq 0..8
        //   d_b[u].lane[v] = output[(v+8)*16 + u] * 256
        //
        // We want output[col_freq * 16 + 0..16] to be written as two sequential stores.
        // Transpose d_a and d_b (each 16×8 → two 8×8 blocks) to get row-major order.
        let mut d_a_lo: [__m256; 8] = d_a[0..8].try_into().unwrap(); // row-freqs 0..8
        let mut d_a_hi: [__m256; 8] = d_a[8..16].try_into().unwrap(); // row-freqs 8..16
        let mut d_b_lo: [__m256; 8] = d_b[0..8].try_into().unwrap();
        let mut d_b_hi: [__m256; 8] = d_b[8..16].try_into().unwrap();
        transpose_8x8(&mut d_a_lo);
        transpose_8x8(&mut d_a_hi);
        transpose_8x8(&mut d_b_lo);
        transpose_8x8(&mut d_b_hi);

        // After final transpose:
        //   d_a_lo[v].lane[u] = output[v*16 + u]       v = col-freq 0..8,  u = row-freq 0..8
        //   d_a_hi[v].lane[u] = output[v*16 + u+8]     v = col-freq 0..8,  u = row-freq 8..16
        //   d_b_lo[v].lane[u] = output[(v+8)*16 + u]   v = col-freq 0..8 → actual col-freq v+8
        //   d_b_hi[v].lane[u] = output[(v+8)*16 + u+8]

        let scale = _mm256_set1_ps(1.0 / 256.0);
        for v in 0..8 {
            let base_a = v * 16;
            _mm256_storeu_ps(
                output[base_a..].as_mut_ptr(),
                _mm256_mul_ps(d_a_lo[v], scale),
            );
            _mm256_storeu_ps(
                output[base_a + 8..].as_mut_ptr(),
                _mm256_mul_ps(d_a_hi[v], scale),
            );
            let base_b = (v + 8) * 16;
            _mm256_storeu_ps(
                output[base_b..].as_mut_ptr(),
                _mm256_mul_ps(d_b_lo[v], scale),
            );
            _mm256_storeu_ps(
                output[base_b + 8..].as_mut_ptr(),
                _mm256_mul_ps(d_b_hi[v], scale),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    const ATOL: f32 = 1e-4;

    fn assert_close(neon: &[f32], scalar: &[f32], label: &str) {
        assert_eq!(neon.len(), scalar.len(), "{label}: length mismatch");
        let mut max_err: f32 = 0.0;
        let mut worst = 0usize;
        for (i, (n, s)) in neon.iter().zip(scalar.iter()).enumerate() {
            let e = (n - s).abs();
            if e > max_err {
                max_err = e;
                worst = i;
            }
        }
        assert!(
            max_err < ATOL,
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
            unsafe { dct16x16_avx2(&input, &mut got) };
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 seed={seed}"));
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
        unsafe { dct16x16_avx2(&input, &mut got) };
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
        unsafe { dct16x16_avx2(&input, &mut got) };
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
            unsafe { dct16x16_avx2(&input, &mut got) };
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
            dct16x16_avx2(&a, &mut da);
            dct16x16_avx2(&b, &mut db);
            dct16x16_avx2(&sum, &mut dsum);
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
        unsafe { dct16x16_avx2(&input, &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 alternating +-1");
    }
}
