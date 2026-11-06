/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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
use std::arch::aarch64::*;

/// NEON implementation of `sse_and_rate_scalar`.
///
/// # Safety
/// Caller must ensure `neon` is available (checked via [`supported`]); on
/// AArch64 NEON is part of the baseline ISA. All slice accesses are
/// bounds-validated against `width*height`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn sse_and_rate_neon<const BIASED: bool>(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    bias: crate::group::DequantBias,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    _rate_log2_lut: &crate::inflated_cost::RateLog2Lut,
    thr: &[f32; 4],
    scan_pos: &[u32],
) -> (f32, usize, f32, u32) {
    sse_and_rate_neon_impl::<BIASED, false>(
        coeff,
        inv_matrix,
        q_scaled,
        bias,
        width,
        height,
        half,
        cx,
        cy,
        _rate_log2_lut,
        thr,
        scan_pos,
        &mut [],
    )
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn sse_and_rate_neon_impl<const BIASED: bool, const SAVE_LEVELS: bool>(
    coeff: &[f32],
    inv_matrix: &[f32],
    q_scaled: f32,
    bias: crate::group::DequantBias,
    width: usize,
    height: usize,
    half: usize,
    cx: usize,
    cy: usize,
    _rate_log2_lut: &crate::inflated_cost::RateLog2Lut,
    thr: &[f32; 4],
    scan_pos: &[u32],
    levels: &mut [i32],
) -> (f32, usize, f32, u32) {
    let n = width * height;
    assert!(coeff.len() >= n && inv_matrix.len() >= n);
    assert!(if SAVE_LEVELS {
        levels.len() >= n
    } else {
        scan_pos.len() >= n
    });
    // Two vectors per step; `half` is a vector boundary.
    assert!(width.is_multiple_of(8) && half.is_multiple_of(4));

    let qs = vdupq_n_f32(q_scaled);
    let mut sse_acc = vdupq_n_f32(0.0);
    let mut nz_acc = vdupq_n_u32(0);
    let mut mag_acc = vdupq_n_f32(0.0);
    let mut scan_acc = vdupq_n_u32(0);
    let zero = vdupq_n_f32(0.0);
    let all_active = vdupq_n_u32(u32::MAX);
    let lane_ids = unsafe { vld1q_u32([0u32, 1, 2, 3].as_ptr()) };

    for (y, (coeffs, inv_matrix)) in coeff
        .chunks_exact(width)
        .zip(inv_matrix.chunks_exact(width))
        .take(height)
        .enumerate()
    {
        let yfix = if y >= height / 2 { 2 } else { 0 };
        let thr_lo = thr[yfix];
        let thr_hi = thr[yfix + 1];
        let llf_row = y < cy;

        for (x0, (coeff, inv_matrix)) in coeffs
            .as_chunks::<8>()
            .0
            .iter()
            .zip(inv_matrix.as_chunks::<8>().0.iter())
            .enumerate()
        {
            let x = x0 * 8;
            let thr0 = vdupq_n_f32(if x >= half { thr_hi } else { thr_lo });
            let thr1 = vdupq_n_f32(if x + 4 >= half { thr_hi } else { thr_lo });

            let c0 = unsafe { vld1q_f32(coeff.as_ptr()) };
            let c1 = unsafe { vld1q_f32(coeff.as_ptr().add(4)) };
            let m0 = unsafe { vld1q_f32(inv_matrix.as_ptr()) };
            let m1 = unsafe { vld1q_f32(inv_matrix.as_ptr().add(4)) };

            let a0 = vmulq_f32(vmulq_f32(m0, qs), c0);
            let a1 = vmulq_f32(vmulq_f32(m1, qs), c1);
            let keep0 = vcgeq_f32(vabsq_f32(a0), thr0);
            let keep1 = vcgeq_f32(vabsq_f32(a1), thr1);
            // Ties-away rounding, matching the real quantizer (vcvtaq/fast_round).
            let q0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vrndaq_f32(a0)), keep0));
            let q1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vrndaq_f32(a1)), keep1));
            let absq0 = vabsq_f32(q0);
            let absq1 = vabsq_f32(q1);
            let nz0 = vcgtq_f32(absq0, zero);
            let nz1 = vcgtq_f32(absq1, zero);

            let (active0, active1) = if llf_row && x < cx {
                let cxv = vdupq_n_u32(cx as u32);
                let lane_x = vaddq_u32(vdupq_n_u32(x as u32), lane_ids);
                (
                    vcgeq_u32(lane_x, cxv),
                    vcgeq_u32(vaddq_u32(lane_x, vdupq_n_u32(4)), cxv),
                )
            } else {
                (all_active, all_active)
            };
            let rate0 = vandq_u32(nz0, active0);
            let rate1 = vandq_u32(nz1, active1);

            // Most pairs quantize to all zeros (65-90 % of them in real
            // encodes): one lane-max per pair skips the dequantization and
            // the rate for those, where every term is zero anyway (masked
            // LLF lanes contribute nothing either way). A nonzero pair
            // gathers the biased level and its magnitude rate from the
            // 16-entry tables by the integer level, with the division and
            // the polynomial only for a level of 16 or more (a few percent
            // of the nonzero pairs).
            let (d0, d1) = if vmaxvq_u32(vorrq_u32(rate0, rate1)) != 0 {
                let tables = SmallLevelTables::new(&bias);
                let lk0 = SmallLevelIndex::new(absq0);
                let lk1 = SmallLevelIndex::new(absq1);
                let mut dq0 = if BIASED {
                    tables.dequantized(lk0, q0)
                } else {
                    q0
                };
                let mut dq1 = if BIASED {
                    tables.dequantized(lk1, q1)
                } else {
                    q1
                };
                let mut lg0 = tables.log2p1(lk0);
                let mut lg1 = tables.log2p1(lk1);
                if vmaxvq_u32(vorrq_u32(lk0.big, lk1.big)) != 0 {
                    if BIASED {
                        dq0 = vbslq_f32(lk0.big, neon_dequantized_level_f32(q0, bias), dq0);
                        dq1 = vbslq_f32(lk1.big, neon_dequantized_level_f32(q1, bias), dq1);
                    }
                    if !SAVE_LEVELS {
                        lg0 = vbslq_f32(lk0.big, neon_log2p1_f32(absq0), lg0);
                        lg1 = vbslq_f32(lk1.big, neon_log2p1_f32(absq1), lg1);
                    }
                }
                if !SAVE_LEVELS {
                    // Same lane order as one vector at a time, so the sums
                    // round identically.
                    mag_acc = vaddq_f32(mag_acc, vbslq_f32(rate0, lg0, zero));
                    mag_acc = vaddq_f32(mag_acc, vbslq_f32(rate1, lg1, zero));
                    // Scan position of the nonzeros (LLF slots are never
                    // nonzero here, so masked-to-zero lanes are neutral).
                    let sv = unsafe { scan_pos.as_ptr().add(y * width + x) };
                    let sv0 = unsafe { vld1q_u32(sv) };
                    let sv1 = unsafe { vld1q_u32(sv.add(4)) };
                    scan_acc = vmaxq_u32(scan_acc, vandq_u32(sv0, rate0));
                    scan_acc = vmaxq_u32(scan_acc, vandq_u32(sv1, rate1));
                }
                (vsubq_f32(a0, dq0), vsubq_f32(a1, dq1))
            } else if BIASED {
                // The biased level of every lane is +0 here: a - 0 = a.
                (a0, a1)
            } else {
                // A level may be -0 here, which a - q folds to +0.
                (vsubq_f32(a0, q0), vsubq_f32(a1, q1))
            };
            let d2_0 = vmulq_f32(d0, d0);
            let d2_1 = vmulq_f32(d1, d1);
            sse_acc = vaddq_f32(sse_acc, vbslq_f32(active0, d2_0, zero));
            sse_acc = vaddq_f32(sse_acc, vbslq_f32(active1, d2_1, zero));
            nz_acc = vaddq_u32(nz_acc, vshrq_n_u32::<31>(rate0));
            nz_acc = vaddq_u32(nz_acc, vshrq_n_u32::<31>(rate1));

            if SAVE_LEVELS {
                let keep0 = vreinterpretq_s32_u32(vandq_u32(keep0, active0));
                let keep1 = vreinterpretq_s32_u32(vandq_u32(keep1, active1));
                let out = unsafe { levels.as_mut_ptr().add(y * width + x) };
                unsafe { vst1q_s32(out, vandq_s32(vcvtaq_s32_f32(a0), keep0)) };
                unsafe { vst1q_s32(out.add(4), vandq_s32(vcvtaq_s32_f32(a1), keep1)) };
            }
        }
    }
    (
        vaddvq_f32(sse_acc),
        vaddvq_u32(nz_acc) as usize,
        vaddvq_f32(mag_acc),
        vmaxvq_u32(scan_acc),
    )
}

/// `log2(1 + k)` for `k < SMALL_LEVELS` as [`neon_log2p1_f32`] evaluates it,
/// so the table path is bit-identical with the polynomial path.
static LOG2P1_SMALL: std::sync::LazyLock<[f32; crate::group::SMALL_LEVELS]> =
    std::sync::LazyLock::new(|| {
        std::array::from_fn(|k| unsafe {
            vgetq_lane_f32::<0>(neon_log2p1_f32(vdupq_n_f32(k as f32)))
        })
    });

/// Byte indices of four integer levels into a 16-entry `f32` table, with the
/// lanes at or above 16 flagged (they index past the table and read zero).
#[derive(Clone, Copy)]
pub(crate) struct SmallLevelIndex {
    bytes: uint8x16_t,
    pub(crate) big: uint32x4_t,
}

impl SmallLevelIndex {
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn new(absq: float32x4_t) -> Self {
        const _: () = assert!(crate::group::SMALL_LEVELS == 16);
        let sixteen = vdupq_n_u32(16);
        // Integer-valued floats convert exactly; huge ones saturate past 16.
        let k = vminq_u32(vcvtq_u32_f32(absq), sixteen);
        // Lane k reads bytes 4k..4k+3: replicate 4k and add the byte offsets.
        let bytes = vorrq_u32(
            vmulq_u32(k, vdupq_n_u32(0x0404_0404)),
            vdupq_n_u32(0x0302_0100),
        );
        Self {
            bytes: vreinterpretq_u8_u32(bytes),
            big: vceqq_u32(k, sixteen),
        }
    }
}

/// The 16-entry dequantization and rate tables of one channel, loaded into
/// registers for `vqtbl4q_u8` gathers.
#[derive(Clone, Copy)]
pub(crate) struct SmallLevelTables {
    dequant: uint8x16x4_t,
    log2p1: uint8x16x4_t,
}

impl SmallLevelTables {
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn new(bias: &crate::group::DequantBias) -> Self {
        Self {
            dequant: unsafe { vld1q_u8_x4(bias.small_levels.as_ptr().cast()) },
            log2p1: unsafe { vld1q_u8_x4(LOG2P1_SMALL.as_ptr().cast()) },
        }
    }

    /// [`neon_dequantized_level_f32`] of `q` for lanes below 16 (zero above).
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn dequantized(self, index: SmallLevelIndex, q: float32x4_t) -> float32x4_t {
        let magnitude = vreinterpretq_u32_u8(vqtbl4q_u8(self.dequant, index.bytes));
        let sign = vandq_u32(vreinterpretq_u32_f32(q), vdupq_n_u32(0x8000_0000));
        vreinterpretq_f32_u32(vorrq_u32(magnitude, sign))
    }

    /// [`neon_log2p1_f32`] of `|q|` for lanes below 16 (zero above).
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn log2p1(self, index: SmallLevelIndex) -> float32x4_t {
        vreinterpretq_f32_u8(vqtbl4q_u8(self.log2p1, index.bytes))
    }
}

/// NEON counterpart of `DequantBias::level_f32`: the decoder's biased
/// dequant of an integer-valued float level (0 -> 0, +-1 -> +-bias.one,
/// q -> q - bias.numerator/q)
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn neon_dequantized_level_f32(
    q: float32x4_t,
    bias: crate::group::DequantBias,
) -> float32x4_t {
    let absq = vabsq_f32(q);
    // q - 0.145/q (q == 0 lanes produce non-finite values, masked out below).
    let big = vsubq_f32(q, vdivq_f32(vdupq_n_f32(bias.numerator), q));
    let sign = vandq_u32(vreinterpretq_u32_f32(q), vdupq_n_u32(0x8000_0000));
    let one = vreinterpretq_f32_u32(vorrq_u32(
        sign,
        vreinterpretq_u32_f32(vdupq_n_f32(bias.one)),
    ));
    let use_big = vcgeq_f32(absq, vdupq_n_f32(1.125));
    let dq = vbslq_f32(use_big, big, one);
    let nz = vcgtq_f32(absq, vdupq_n_f32(0.0));
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dq), nz))
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn neon_log2p1_f32(x: float32x4_t) -> float32x4_t {
    // y = 1 + x
    let y = vaddq_f32(x, vdupq_n_f32(1.0));

    // Decompose y = m * 2^e, m in [1, 2).
    let bits = vreinterpretq_u32_f32(y);

    let exp_u = vshrq_n_u32::<23>(bits);
    let exp_i = vsubq_s32(vreinterpretq_s32_u32(exp_u), vdupq_n_s32(127));
    let e = vcvtq_f32_s32(exp_i);

    let mant_bits = vorrq_u32(
        vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f80_0000),
    );
    let m = vreinterpretq_f32_u32(mant_bits);

    let t = vsubq_f32(m, vdupq_n_f32(1.0));

    // Generated by tools/log2p1.sollya.
    let c0 = vdupq_n_f32(1.4426934719085693359375);
    let c1 = vdupq_n_f32(-0.721179187297821044921875);
    let c2 = vdupq_n_f32(0.477900087833404541015625);
    let c3 = vdupq_n_f32(-0.340080082416534423828125);
    let c4 = vdupq_n_f32(0.21719777584075927734375);
    let c5 = vdupq_n_f32(-9.749893844127655029296875e-2);
    let c6 = vdupq_n_f32(2.096841670572757720947265625e-2);

    let mut p = c6;
    p = vfmaq_f32(c5, t, p);
    p = vfmaq_f32(c4, t, p);
    p = vfmaq_f32(c3, t, p);
    p = vfmaq_f32(c2, t, p);
    p = vfmaq_f32(c1, t, p);
    p = vfmaq_f32(c0, t, p);

    vfmaq_f32(e, t, p)
}

#[cfg(test)]
mod tests {
    use super::{neon_log2p1_f32, sse_and_rate_neon};
    use std::arch::aarch64::{vdupq_n_f32, vgetq_lane_f32};

    #[allow(clippy::too_many_arguments)]
    fn reference(
        coeff: &[f32],
        inv: &[f32],
        qs: f32,
        bias: crate::group::DequantBias,
        w: usize,
        h: usize,
        half: usize,
        cx: usize,
        cy: usize,
        thr: &[f32; 4],
        scan_pos: &[u32],
        biased: bool,
    ) -> (f32, usize, f32, u32) {
        let (mut sse, mut nzeros, mut mag_bits) = (0.0f32, 0usize, 0.0f32);
        let mut max_scan = 0u32;
        for y in 0..h {
            let yfix = if y >= h / 2 { 2 } else { 0 };
            for x in 0..w {
                if x < cx && y < cy {
                    continue;
                }
                let i = y * w + x;
                let threshold = if x >= half { thr[yfix + 1] } else { thr[yfix] };
                let a = inv[i] * qs * coeff[i];
                let q = if a.abs() >= threshold { a.round() } else { 0.0 };
                let d = if biased { a - bias.level_f32(q) } else { a - q };
                sse += d * d;
                if q != 0.0 {
                    nzeros += 1;
                    mag_bits += (1.0 + q.abs()).log2();
                    max_scan = max_scan.max(scan_pos[i]);
                }
            }
        }
        (sse, nzeros, mag_bits, max_scan)
    }

    #[test]
    fn test_sse_and_rate_neon_vs_reference() {
        let mut state = 0x5e5e_a11d_0f00_d00du64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32 as f32 / u32::MAX as f32
        };

        for &(w, h, half, cx, cy) in &[
            (8usize, 8usize, 4usize, 1usize, 1usize),
            (16, 8, 8, 2, 1),
            (8, 16, 4, 1, 2),
            (16, 16, 8, 2, 2),
            (32, 32, 16, 4, 4),
        ] {
            for case in 0..100 {
                let n = w * h;
                // Alternate sparse and dense inputs so the all-zero-vector
                // fast path and the logarithm path are both exercised.
                let coeff_scale = if case % 2 == 0 { 0.2 } else { 200.0 };
                let coeff: Vec<f32> = (0..n).map(|_| (random() - 0.5) * coeff_scale).collect();
                let inv: Vec<f32> = (0..n).map(|_| 0.001 + random() * 0.5).collect();
                let qs = 0.5 + random() * 3.0;
                let bias = crate::group::DequantBias::DEFAULT[case % 3];
                let thr = [
                    random() * 0.6,
                    random() * 0.6,
                    random() * 0.6,
                    random() * 0.6,
                ];
                let scan_pos = crate::coeff_order::scan_pos_lut(w, h);
                for biased in [false, true] {
                    let expected = reference(
                        &coeff, &inv, qs, bias, w, h, half, cx, cy, &thr, scan_pos, biased,
                    );
                    let kernel = if biased {
                        sse_and_rate_neon::<true>
                    } else {
                        sse_and_rate_neon::<false>
                    };
                    let actual = unsafe {
                        kernel(
                            &coeff,
                            &inv,
                            qs,
                            bias,
                            w,
                            h,
                            half,
                            cx,
                            cy,
                            crate::inflated_cost::rate_log2_lut(),
                            &thr,
                            scan_pos,
                        )
                    };

                    assert_eq!(actual.1, expected.1, "nzeros mismatch for {w}x{h}");
                    assert_eq!(actual.3, expected.3, "max-scan mismatch for {w}x{h}");
                    let sse_rel = (actual.0 - expected.0).abs() / expected.0.abs().max(1.0);
                    let mag_rel = (actual.2 - expected.2).abs() / expected.2.abs().max(1.0);
                    assert!(
                        sse_rel < 1e-4,
                        "SSE relative error {sse_rel} for {w}x{h} biased={biased}"
                    );
                    assert!(
                        mag_rel < 1e-5,
                        "magnitude-rate relative error {mag_rel} for {w}x{h} biased={biased}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_log2p1() {
        unsafe {
            for i in 0..5000 {
                let q1 = ((i as f32 / 5000.) + 1.).log2();
                let q2 = vgetq_lane_f32::<0>(neon_log2p1_f32(vdupq_n_f32(i as f32 / 5000.)));
                assert!((q1 - q2).abs() < 1e-5, "q1 {} q2 {}", q1, q2);
            }
        }
    }
}
