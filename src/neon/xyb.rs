/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::xyb::*;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn xyb_to_rgb_neon(
    m: &XybMatrix,
    x: float32x4_t,
    y: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let cube = |value: float32x4_t| {
        let value = vsubq_f32(value, vdupq_n_f32(NEG_BIAS_CBRT));
        // Keep the cube and bias subtraction separate, as in the scalar inverse.
        vsubq_f32(
            vmulq_f32(vmulq_f32(value, value), value),
            vdupq_n_f32(OPSIN_BIAS),
        )
    };
    let m0 = cube(vaddq_f32(y, x));
    let m1 = cube(vsubq_f32(y, x));
    let m2 = cube(b);
    // Match dct::fmla, including builds that do not fuse the inverse matrix.
    let row = |offset: usize| {
        let value = vmulq_n_f32(m2, m.inv[offset + 2]);
        let value = vfmaq_n_f32(value, m1, m.inv[offset + 1]);
        vfmaq_n_f32(value, m0, m.inv[offset])
    };
    (row(0), row(3), row(6))
}

/// # Safety
/// The CPU must support the enabled target features and all six slices must
/// have equal lengths. Only complete SIMD vectors are read and written.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn to_linear_rows_neon(
    m: &XybMatrix,
    input: [&[f32]; 3],
    output: [&mut [f32]; 3],
) -> usize {
    let [r, g, b] = output;
    let done = unsafe { to_linear_rows::<4>(m, input, [r, g, b]) };
    done + unsafe {
        to_linear_rows::<1>(
            m,
            input.map(|row| &row[done..]),
            [&mut r[done..], &mut g[done..], &mut b[done..]],
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn to_linear_rows<const N: usize>(
    m: &XybMatrix,
    [x, y, b]: [&[f32]; 3],
    [r, g, out_b]: [&mut [f32]; 3],
) -> usize {
    const LANES: usize = 4;
    let done = r.len() / (N * LANES) * (N * LANES);
    for start in (0..done).step_by(N * LANES) {
        for lane in 0..N {
            let i = start + lane * LANES;
            // The caller's equal-length contract and complete-vector loop
            // bound cover every unaligned load and store.
            unsafe {
                let xv = vld1q_f32(x.as_ptr().add(i));
                let yv = vld1q_f32(y.as_ptr().add(i));
                let bv = vld1q_f32(b.as_ptr().add(i));
                let (rv, gv, bv) = xyb_to_rgb_neon(m, xv, yv, bv);
                vst1q_f32(r.as_mut_ptr().add(i), rv);
                vst1q_f32(g.as_mut_ptr().add(i), gv);
                vst1q_f32(out_b.as_mut_ptr().add(i), bv);
            }
        }
    }
    done
}

#[inline]
#[target_feature(enable = "neon")]
fn halley_cbrt<const SCALAR_ORDER: bool>(x: float32x4_t, a: float32x4_t) -> float32x4_t {
    let tx = vmulq_f32(vmulq_f32(x, x), x);
    let num = vfmaq_n_f32(tx, a, 2.0);
    let den = vfmaq_n_f32(a, tx, 2.0);
    if SCALAR_ORDER {
        vdivq_f32(vmulq_f32(x, num), den)
    } else {
        vmulq_f32(x, vdivq_f32(num, den))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn integer_pow_1_3(hx: uint32x4_t) -> uint32x4_t {
    let scale = vdupq_n_u32(341);
    let hi = vshrq_n_u64::<10>(vmull_high_u32(hx, scale));
    let lo = vshrq_n_u64::<10>(vmull_u32(vget_low_u32(hx), vget_low_u32(scale)));
    vcombine_u32(vmovn_u64(lo), vmovn_u64(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn cbrt_seed_positive_f32<const SCALAR_ORDER: bool>(x: float32x4_t) -> float32x4_t {
    let hx = vreinterpretq_u32_f32(x);
    let third = if SCALAR_ORDER {
        // Exact unsigned /3, matching `xyb::cbrtf`'s seed.
        let magic = vdup_n_u32(0xaaaa_aaab);
        let lo = vmovn_u64(vshrq_n_u64::<33>(vmull_u32(vget_low_u32(hx), magic)));
        let hi = vmovn_u64(vshrq_n_u64::<33>(vmull_u32(vget_high_u32(hx), magic)));
        vcombine_u32(lo, hi)
    } else {
        integer_pow_1_3(hx)
    };
    let hx = vaddq_u32(third, vdupq_n_u32(709958130));
    vreinterpretq_f32_u32(hx)
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn vcbrtq_fast3_positive_f32(
    a0: float32x4_t,
    a1: float32x4_t,
    a2: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    cbrt3_positive_f32::<false>(a0, a1, a2)
}

#[inline]
#[target_feature(enable = "neon")]
fn cbrt3_positive_f32<const SCALAR_ORDER: bool>(
    a0: float32x4_t,
    a1: float32x4_t,
    a2: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let mut x0 = cbrt_seed_positive_f32::<SCALAR_ORDER>(a0);
    let mut x1 = cbrt_seed_positive_f32::<SCALAR_ORDER>(a1);
    let mut x2 = cbrt_seed_positive_f32::<SCALAR_ORDER>(a2);

    x0 = halley_cbrt::<SCALAR_ORDER>(x0, a0);
    x1 = halley_cbrt::<SCALAR_ORDER>(x1, a1);
    x2 = halley_cbrt::<SCALAR_ORDER>(x2, a2);

    x0 = halley_cbrt::<SCALAR_ORDER>(x0, a0);
    x1 = halley_cbrt::<SCALAR_ORDER>(x1, a1);
    x2 = halley_cbrt::<SCALAR_ORDER>(x2, a2);

    let zero = vdupq_n_f32(0.0);
    x0 = vbslq_f32(vceqq_f32(a0, zero), zero, x0);
    x1 = vbslq_f32(vceqq_f32(a1, zero), zero, x1);
    x2 = vbslq_f32(vceqq_f32(a2, zero), zero, x2);

    (x0, x1, x2)
}

/// `SCALAR_ORDER` preserves scalar seed/rounding for threshold classifiers;
/// full-frame conversion retains its existing fast arithmetic. The choice is
/// compile-time, so neither path pays for a runtime mode check.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn rgb_to_xyb_f32x4_neon<const SCALAR_ORDER: bool>(
    m: &XybMatrix,
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let bias = vdupq_n_f32(OPSIN_BIAS);

    let mut mixed0 = vfmaq_n_f32(bias, b, m.fwd[2]);
    mixed0 = vfmaq_n_f32(mixed0, g, m.fwd[1]);
    mixed0 = vfmaq_n_f32(mixed0, r, m.fwd[0]);

    let mut mixed1 = vfmaq_n_f32(bias, b, m.fwd[5]);
    mixed1 = vfmaq_n_f32(mixed1, g, m.fwd[4]);
    mixed1 = vfmaq_n_f32(mixed1, r, m.fwd[3]);

    let mut mixed2 = vfmaq_n_f32(bias, b, m.fwd[8]);
    mixed2 = vfmaq_n_f32(mixed2, g, m.fwd[7]);
    mixed2 = vfmaq_n_f32(mixed2, r, m.fwd[6]);

    let zero = vdupq_n_f32(0.0);
    if SCALAR_ORDER {
        mixed0 = vmaxnmq_f32(mixed0, zero);
        mixed1 = vmaxnmq_f32(mixed1, zero);
        mixed2 = vmaxnmq_f32(mixed2, zero);
    } else {
        mixed0 = vmaxq_f32(mixed0, zero);
        mixed1 = vmaxq_f32(mixed1, zero);
        mixed2 = vmaxq_f32(mixed2, zero);
    }

    let (tm0, tm1, tm2) = cbrt3_positive_f32::<SCALAR_ORDER>(mixed0, mixed1, mixed2);

    let neg_bias = vdupq_n_f32(NEG_BIAS_CBRT);
    let tm0 = vaddq_f32(tm0, neg_bias);
    let tm1 = vaddq_f32(tm1, neg_bias);
    let tm2 = vaddq_f32(tm2, neg_bias);

    let half = vdupq_n_f32(0.5);

    let x = vmulq_f32(vsubq_f32(tm0, tm1), half);
    let y = vmulq_f32(vaddq_f32(tm0, tm1), half);

    (x, y, tm2)
}

/// Convert the complete SIMD vectors of the modular XYB lattice.
#[target_feature(enable = "neon")]
pub(crate) fn xyb_lattice_neon(
    m: &XybMatrix,
    [r, g, b]: [&[f32]; 3],
    [luma, co, cg]: [&mut [i32]; 3],
) -> usize {
    use crate::lossless::rct_simd::neon::forward_vec;
    use crate::quant_weights::INV_DC_QUANT;

    let [sx, sy, sb] = [
        vdupq_n_f32(INV_DC_QUANT[0]),
        vdupq_n_f32(INV_DC_QUANT[1]),
        vdupq_n_f32(INV_DC_QUANT[2]),
    ];
    let r = r.as_chunks::<4>().0;
    let g = g.as_chunks::<4>().0;
    let b = b.as_chunks::<4>().0;
    let luma = luma.as_chunks_mut::<4>().0;
    let co = co.as_chunks_mut::<4>().0;
    let cg = cg.as_chunks_mut::<4>().0;
    for (((r, g), b), ((luma, co_out), cg_out)) in
        r.iter().zip(g).zip(b).zip(luma.iter_mut().zip(co).zip(cg))
    {
        let r = unsafe { vld1q_f32(r.as_ptr()) };
        let g = unsafe { vld1q_f32(g.as_ptr()) };
        let b = unsafe { vld1q_f32(b.as_ptr()) };
        let (x, y, b) = rgb_to_xyb_f32x4_neon::<true>(m, r, g, b);
        let y = vcvtaq_s32_f32(vmulq_f32(y, sy));
        let x = vcvtaq_s32_f32(vmulq_f32(x, sx));
        let b_minus_y = vsubq_s32(vcvtaq_s32_f32(vmulq_f32(b, sb)), y);
        let [luma_value, co, cg] = forward_vec(y, x, b_minus_y);
        unsafe {
            vst1q_s32(luma.as_mut_ptr(), luma_value);
            vst1q_s32(co_out.as_mut_ptr(), co);
            vst1q_s32(cg_out.as_mut_ptr(), cg);
        }
    }
    r.len() * 4
}

/// Transform one row-band into separate output planes.
#[target_feature(enable = "neon")]
pub(crate) fn to_xyb_neon_band(
    m: &XybMatrix,
    input: [&[f32]; 3],
    output: [&mut [f32]; 3],
    w: usize,
) {
    let [rp, gp, bp] = input;
    let [xp, yp, out_bp] = output;
    for (((((r_row, g_row), b_row), x_row), y_row), out_b_row) in rp
        .chunks_exact(w)
        .zip(gp.chunks_exact(w))
        .zip(bp.chunks_exact(w))
        .zip(xp.chunks_exact_mut(w))
        .zip(yp.chunks_exact_mut(w))
        .zip(out_bp.chunks_exact_mut(w))
    {
        let (r_chunks, r_tail) = r_row.as_chunks::<4>();
        let (g_chunks, g_tail) = g_row.as_chunks::<4>();
        let (b_chunks, b_tail) = b_row.as_chunks::<4>();
        let (x_chunks, x_tail) = x_row.as_chunks_mut::<4>();
        let (y_chunks, y_tail) = y_row.as_chunks_mut::<4>();
        let (out_b_chunks, out_b_tail) = out_b_row.as_chunks_mut::<4>();

        for (((((r4, g4), b4), x4), y4), out_b4) in r_chunks
            .iter()
            .zip(g_chunks.iter())
            .zip(b_chunks.iter())
            .zip(x_chunks.iter_mut())
            .zip(y_chunks.iter_mut())
            .zip(out_b_chunks.iter_mut())
        {
            let r = unsafe { vld1q_f32(r4.as_ptr()) };
            let g = unsafe { vld1q_f32(g4.as_ptr()) };
            let b = unsafe { vld1q_f32(b4.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_neon::<false>(m, r, g, b);

            unsafe {
                vst1q_f32(x4.as_mut_ptr(), xv);
                vst1q_f32(y4.as_mut_ptr(), yv);
                vst1q_f32(out_b4.as_mut_ptr(), bv);
            }
        }

        if !r_tail.is_empty() {
            let mut r4: [f32; 4] = [0.; 4];
            let mut g4: [f32; 4] = [0.; 4];
            let mut b4: [f32; 4] = [0.; 4];
            r4[..r_tail.len()].copy_from_slice(r_tail);
            g4[..g_tail.len()].copy_from_slice(g_tail);
            b4[..b_tail.len()].copy_from_slice(b_tail);
            let r = unsafe { vld1q_f32(r4.as_ptr()) };
            let g = unsafe { vld1q_f32(g4.as_ptr()) };
            let b = unsafe { vld1q_f32(b4.as_ptr()) };

            let (xv, yv, bv) = rgb_to_xyb_f32x4_neon::<false>(m, r, g, b);

            unsafe {
                vst1q_f32(r4.as_mut_ptr(), xv);
                vst1q_f32(g4.as_mut_ptr(), yv);
                vst1q_f32(b4.as_mut_ptr(), bv);
            }

            x_tail.copy_from_slice(&r4[..r_tail.len()]);
            y_tail.copy_from_slice(&g4[..g_tail.len()]);
            out_b_tail.copy_from_slice(&b4[..b_tail.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::to_xyb_neon_band;
    use crate::xyb::rgb_to_xyb_pixel_f32;

    fn rng(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 40) as f32 / (1u64 << 24) as f32
    }

    #[test]
    fn scalar_order_conversion_is_bit_exact() {
        use super::rgb_to_xyb_f32x4_neon;
        use std::arch::aarch64::{vld1q_f32, vst1q_f32};

        let mut state = 0x8476_12b9;
        let m = crate::xyb::XybMatrix::SPEC;
        for scale in [0.0, 1.0, 16.0] {
            for _ in 0..1024 {
                let rgb: [[f32; 4]; 3] = std::array::from_fn(|_| {
                    std::array::from_fn(|_| (rng(&mut state) - 0.1) * scale)
                });
                let mut actual = [[0.0; 4]; 3];
                unsafe {
                    let (x, y, b) = rgb_to_xyb_f32x4_neon::<true>(
                        &m,
                        vld1q_f32(rgb[0].as_ptr()),
                        vld1q_f32(rgb[1].as_ptr()),
                        vld1q_f32(rgb[2].as_ptr()),
                    );
                    for (dst, v) in actual.iter_mut().zip([x, y, b]) {
                        vst1q_f32(dst.as_mut_ptr(), v);
                    }
                }
                for i in 0..4 {
                    let (x, y, b) = rgb_to_xyb_pixel_f32(&m, rgb[0][i], rgb[1][i], rgb[2][i]);
                    assert_eq!(
                        actual.map(|c| c[i].to_bits()),
                        [x, y, b].map(f32::to_bits),
                        "RGB {:?}",
                        rgb.map(|c| c[i])
                    );
                }
            }
        }
    }

    #[test]
    fn to_xyb_neon_band_matches_scalar_pixels() {
        check_band_variant(&crate::xyb::XybMatrix::SPEC);
    }

    fn check_band_variant(m: &crate::xyb::XybMatrix) {
        let mut state = 0xc0ff_ee11_u64;
        for &w in &[4usize, 7, 16, 19] {
            for rows in 1..=3 {
                let n = w * rows;
                let src: Vec<[f32; 3]> = (0..n)
                    .map(|i| {
                        // Cover the dark end, mid-tones and clipping headroom,
                        // plus exact zeros where the cube root is least stable.
                        let s = if i % 11 == 0 { 0.0 } else { rng(&mut state) };
                        [s, rng(&mut state), rng(&mut state) * 0.5]
                    })
                    .collect();
                let r: Vec<f32> = src.iter().map(|p| p[0]).collect();
                let g: Vec<f32> = src.iter().map(|p| p[1]).collect();
                let b: Vec<f32> = src.iter().map(|p| p[2]).collect();
                let mut x = vec![0.0; n];
                let mut y = vec![0.0; n];
                let mut out_b = vec![0.0; n];
                unsafe { to_xyb_neon_band(m, [&r, &g, &b], [&mut x, &mut y, &mut out_b], w) };
                for (i, p) in src.iter().enumerate() {
                    let (want_x, want_y, want_b) = rgb_to_xyb_pixel_f32(m, p[0], p[1], p[2]);
                    for (got, want, name) in [
                        (x[i], want_x, "X"),
                        (y[i], want_y, "Y"),
                        (out_b[i], want_b, "B"),
                    ] {
                        assert!(
                            (got - want).abs() <= 2e-6,
                            "{name} mismatch at w={w} i={i}: neon {got} vs scalar {want}"
                        );
                    }
                }
            }
        }
    }
}
