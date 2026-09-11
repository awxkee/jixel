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

use std::arch::aarch64::*;

use crate::entropy::ALPHABET_SIZE;

#[inline]
#[target_feature(enable = "neon")]
fn dirty_log2f_x4(d: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let mut ix = vreinterpretq_u32_f32(d);
    ix = vaddq_u32(ix, vdupq_n_u32(0x3f80_0000u32 - 0x3f35_04f3u32));
    let n = vreinterpretq_s32_u32(vsubq_u32(vshrq_n_u32(ix, 23), vdupq_n_u32(0x7f)));
    ix = vaddq_u32(
        vandq_u32(ix, vdupq_n_u32(0x007f_ffff)),
        vdupq_n_u32(0x3f35_04f3),
    );

    let a = vreinterpretq_f32_u32(ix);
    // Use architectural division: reciprocal-estimate precision is not part of
    // the NEON contract and can otherwise perturb clustering decisions.
    let x = vdivq_f32(vsubq_f32(a, one), vaddq_f32(a, one));
    let x2 = vmulq_f32(x, x);
    let mut u = vdupq_n_f32(0.412_198_57);
    u = vfmaq_f32(vdupq_n_f32(0.577_078_04), u, x2);
    u = vfmaq_f32(vdupq_n_f32(0.961_796_7), u, x2);
    let base = vfmaq_f32(vcvtq_f32_s32(n), x, vdupq_n_f32(2.885_390_1));
    vfmaq_f32(base, vmulq_f32(x2, x), u)
}

/// Order-0 entropy of a lossless residual histogram, with the scalar
/// division, logarithm approximation, and accumulation order preserved.
///
/// # Safety
/// The caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub(crate) fn entropy_of_hist_neon(hist: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f32;
    let t = vdupq_n_f32(total_f);
    let mut bits = 0.0f32;
    // Residual histograms often have long empty regions. Pack only occupied
    // bins into SIMD lanes instead of evaluating logarithms of zero bins.
    let mut counts = hist.iter().copied().filter(|&c| c != 0);
    while let Some(c0) = counts.next() {
        let Some(c1) = counts.next() else {
            bits -= c0 as f32 * crate::adaptive_quant::dirty_log2f(c0 as f32 / total_f);
            break;
        };
        let Some(c2) = counts.next() else {
            for c in [c0, c1] {
                bits -= c as f32 * crate::adaptive_quant::dirty_log2f(c as f32 / total_f);
            }
            break;
        };
        let Some(c3) = counts.next() else {
            for c in [c0, c1, c2] {
                bits -= c as f32 * crate::adaptive_quant::dirty_log2f(c as f32 / total_f);
            }
            break;
        };
        // Convert directly to f32: an intermediate f64 could round large
        // u64 counts twice and disagree with the scalar calculation.
        let floats = [c0 as f32, c1 as f32, c2 as f32, c3 as f32];
        let c = unsafe { vld1q_f32(floats.as_ptr()) };
        let p = vdivq_f32(c, t);
        let terms = vmulq_f32(c, dirty_log2f_x4(p));
        let mut lanes = [0.0f32; 4];
        unsafe { vst1q_f32(lanes.as_mut_ptr(), terms) };
        // Reassociating the sum or fusing these subtractions can change the
        // selected predictor or threshold, even with identical bin costs.
        for value in lanes {
            bits -= value;
        }
    }
    bits
}

/// Shannon population cost used by entropy histogram clustering.
///
/// # Safety
/// The caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub(crate) fn counts_bit_cost_neon(counts: &[u32; ALPHABET_SIZE], total_count: u32) -> f32 {
    debug_assert_ne!(total_count, 0);
    let log_total = vdupq_n_f32(crate::adaptive_quant::dirty_log2f(total_count as f32));
    let one = vdupq_n_f32(1.0);
    let mut cost0 = vdupq_n_f32(0.0);
    let mut cost1 = vdupq_n_f32(0.0);
    for counts8 in counts.as_chunks::<8>().0 {
        let count0 = vcvtq_f32_u32(unsafe { vld1q_u32(counts8.as_ptr()) });
        let count1 = vcvtq_f32_u32(unsafe { vld1q_u32(counts8.as_ptr().add(4)) });
        let positive0 = vmaxq_f32(count0, one);
        let positive1 = vmaxq_f32(count1, one);
        cost0 = vfmaq_f32(
            cost0,
            count0,
            vsubq_f32(log_total, dirty_log2f_x4(positive0)),
        );
        cost1 = vfmaq_f32(
            cost1,
            count1,
            vsubq_f32(log_total, dirty_log2f_x4(positive1)),
        );
    }
    vaddvq_f32(vaddq_f32(cost0, cost1)).max(0.0)
}
