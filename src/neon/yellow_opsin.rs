/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

//! Batched source conversions for the sampled yellow-structure classifier.

use super::xyb::rgb_to_xyb_f32x4_neon;
use crate::image::Image3F;
use crate::xyb::XybMatrix;
use std::arch::aarch64::*;

/// Convert center, right, and below together. Missing neighbors duplicate the
/// center, but are excluded from the sums, including for one-pixel dimensions.
#[target_feature(enable = "neon")]
pub(crate) fn sampled_b_gradient_ratio_neon(linear: &Image3F, stride: usize) -> f32 {
    let (w, h) = (linear.xsize(), linear.ysize());
    let [rp, gp, bp] = std::array::from_fn(|c| linear.plane_data(c));
    let (mut sum_y, mut sum_b) = (0.0, 0.0);
    for y in (0..h).step_by(stride) {
        for x in (0..w).step_by(stride) {
            let i = y * w + x;
            let right = if x + 1 < w { i + 1 } else { i };
            let below = if y + 1 < h { i + w } else { i };
            let gather = |p: &[f32]| unsafe { vld1q_f32([p[i], p[right], p[below], 0.0].as_ptr()) };
            let (_, yv, bv) =
                rgb_to_xyb_f32x4_neon::<true>(&XybMatrix::SPEC, gather(rp), gather(gp), gather(bp));
            let bv = vsubq_f32(bv, yv);
            let mut ys = [0.0; 4];
            let mut bs = [0.0; 4];
            unsafe {
                vst1q_f32(ys.as_mut_ptr(), yv);
                vst1q_f32(bs.as_mut_ptr(), bv);
            }
            // Keep the original horizontal-then-vertical accumulation order.
            if x + 1 < w {
                sum_y += (ys[1] - ys[0]).abs();
                sum_b += (bs[1] - bs[0]).abs();
            }
            if y + 1 < h {
                sum_y += (ys[2] - ys[0]).abs();
                sum_b += (bs[2] - bs[0]).abs();
            }
        }
    }
    if sum_y > f32::EPSILON {
        sum_b / sum_y
    } else {
        0.0
    }
}
