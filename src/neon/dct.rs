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
    let mut left = load8_128(input, 16);
    let mut right = load8_128(&input[8..], 16);

    let mut c = [NeonDoubledVector {
        lo: vdupq_n_f32(0.0),
        hi: vdupq_n_f32(0.0),
    }; 16];

    transpose_8x8(&mut left);
    transpose_8x8(&mut right);

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

// ─── Helper: load 16 rows × 8 cols with arbitrary stride ─────────────────────

#[inline]
#[target_feature(enable = "neon")]
fn load16_256(ptr: &[f32], stride: usize) -> [NeonDoubledVector; 16] {
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
        row(8),
        row(9),
        row(10),
        row(11),
        row(12),
        row(13),
        row(14),
        row(15),
    ]
}

#[target_feature(enable = "neon")]
pub(crate) fn dct16x16_neon(input: &[f32; 256], output: &mut [f32; 256]) {
    let mut c_l = load16_256(input.as_slice(), 16);
    let mut c_r = load16_256(&input[8..], 16);

    dct1d_16_v(&mut c_l);
    dct1d_16_v(&mut c_r);

    // Step 4: split into row-freq groups 0..8 and 8..16.
    let mut top_l: [NeonDoubledVector; 8] = c_l[0..8].try_into().unwrap();
    let mut bot_l: [NeonDoubledVector; 8] = c_l[8..16].try_into().unwrap();
    let mut top_r: [NeonDoubledVector; 8] = c_r[0..8].try_into().unwrap();
    let mut bot_r: [NeonDoubledVector; 8] = c_r[8..16].try_into().unwrap();

    transpose_8x8(&mut top_l);
    transpose_8x8(&mut bot_l);
    transpose_8x8(&mut top_r);
    transpose_8x8(&mut bot_r);

    let mut d_a = [NeonDoubledVector {
        lo: vdupq_n_f32(0.0),
        hi: vdupq_n_f32(0.0),
    }; 16];
    let mut d_b = d_a;
    d_a[0..8].copy_from_slice(&top_l);
    d_a[8..16].copy_from_slice(&top_r);
    d_b[0..8].copy_from_slice(&bot_l);
    d_b[8..16].copy_from_slice(&bot_r);

    // Step 7: DCT-16 along the column dimension.
    dct1d_16_v(&mut d_a);
    dct1d_16_v(&mut d_b);

    let scale = 1.0 / 256.0;
    for u in 0..16 {
        let base = &mut output[u * 16..];
        unsafe {
            vst1q_f32(base.as_mut_ptr(), vmulq_n_f32(d_a[u].lo, scale));
            vst1q_f32(base[4..].as_mut_ptr(), vmulq_n_f32(d_a[u].hi, scale));
            vst1q_f32(base[8..].as_mut_ptr(), vmulq_n_f32(d_b[u].lo, scale));
            vst1q_f32(base[12..].as_mut_ptr(), vmulq_n_f32(d_b[u].hi, scale));
        }
    }
}

#[cfg(test)]
mod neon_dct_tests {
    use crate::dct::{dct8x8_scalar, dct8x16_scalar, dct16x8_scalar};

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
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_vs_scalar_random() {
        use crate::neon::dct8x8_neon;
        for seed in 0u64..32 {
            let input: [f32; 64] = fill(seed);
            let mut got = [0.0f32; 64];
            let mut want = [0.0f32; 64];
            unsafe { dct8x8_neon(&input, &mut got) };
            dct8x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct8x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct8x8_neon_dc_only() {
        // All-constant input: only DC coefficient should be non-zero.
        use crate::neon::dct8x8_neon;
        let input = [0.5f32; 64];
        let mut got = [0.0f32; 64];
        let mut want = [0.0f32; 64];
        unsafe { dct8x8_neon(&input, &mut got) };
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
        unsafe { dct8x8_neon(&input, &mut got) };
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
            dct8x8_neon(&a, &mut da);
            dct8x8_neon(&b, &mut db);
            dct8x8_neon(&sum, &mut dsum);
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
            unsafe { dct8x8_neon(&input, &mut got) };
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
            unsafe { dct8x16_neon(&input, &mut got) };
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
        unsafe { dct8x16_neon(&input, &mut got) };
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
        unsafe { dct8x16_neon(&input, &mut got) };
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
            unsafe { dct8x16_neon(&input, &mut got) };
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
            dct8x16_neon(&a, &mut da);
            dct8x16_neon(&b, &mut db);
            dct8x16_neon(&sum, &mut dsum);
        }
        let expected: Vec<f32> = (0..128).map(|i| da[i] + db[i]).collect();
        assert_close(&dsum, &expected, "dct8x16 linearity");
    }

    // ── dct16x8 ───────────────────────────────────────────────────────────────

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_vs_scalar_random() {
        use crate::neon::dct16x8_neon;
        for seed in 0u64..32 {
            let input: [f32; 128] = fill(seed.wrapping_add(0xbeef));
            let mut got = [0.0f32; 128];
            let mut want = [0.0f32; 128];
            unsafe { dct16x8_neon(&input, &mut got) };
            dct16x8_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x8 seed={seed}"));
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn test_dct16x8_neon_dc_only() {
        use crate::neon::dct16x8_neon;
        let input = [0.5f32; 128];
        let mut got = [0.0f32; 128];
        let mut want = [0.0f32; 128];
        unsafe { dct16x8_neon(&input, &mut got) };
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
        unsafe { dct16x8_neon(&input, &mut got) };
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
            unsafe { dct16x8_neon(&input, &mut got) };
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
            dct16x8_neon(&a, &mut da);
            dct16x8_neon(&b, &mut db);
            dct16x8_neon(&sum, &mut dsum);
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
            dct8x16_neon(&input, &mut out8x16);
            dct16x8_neon(&input, &mut out16x8);
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
        unsafe { dct8x8_neon(&input, &mut got) };
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
        unsafe { dct16x8_neon(&input, &mut got) };
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
        unsafe { dct8x16_neon(&input, &mut got) };
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
            unsafe { dct16x16_neon(&input, &mut got) };
            dct16x16_scalar(&input, &mut want);
            assert_close(&got, &want, &format!("dct16x16 seed={seed}"));
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
        unsafe { dct16x16_neon(&input, &mut got) };
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
        unsafe { dct16x16_neon(&input, &mut got) };
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
            unsafe { dct16x16_neon(&input, &mut got) };
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
            dct16x16_neon(&a, &mut da);
            dct16x16_neon(&b, &mut db);
            dct16x16_neon(&sum, &mut dsum);
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
        unsafe { dct16x16_neon(&input, &mut got) };
        dct16x16_scalar(&input, &mut want);
        assert_close(&got, &want, "dct16x16 alternating +-1");
    }
}
