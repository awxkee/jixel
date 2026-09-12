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

use super::xyb::rgb_to_xyb_f32x8_avx2;
use crate::image::Image3F;
use crate::xyb::XybMatrix;
use std::arch::x86_64::*;

/// Two sample neighborhoods per vector: center/right/below in lanes 0..3 and
/// 4..7. Missing neighbors and the final unpaired sample use safe duplicates;
/// only real neighbors contribute, in the original scalar accumulation order.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn sampled_b_gradient_ratio_avx2(linear: &Image3F, stride: usize) -> f32 {
    let (w, h) = (linear.xsize(), linear.ysize());
    let [rp, gp, bp] = std::array::from_fn(|c| linear.plane_data(c));
    let (mut sum_y, mut sum_b) = (0.0, 0.0);
    for y in (0..h).step_by(stride) {
        let has_below = y + 1 < h;
        for x in (0..w).step_by(2 * stride) {
            let i = y * w + x;
            let right = if x + 1 < w { i + 1 } else { i };
            let below = if has_below { i + w } else { i };
            let next_x = x + stride;
            let paired = next_x < w;
            let j = if paired { i + stride } else { i };
            let next_right = if paired && next_x + 1 < w { j + 1 } else { j };
            let next_below = if paired && has_below { j + w } else { j };
            let gather = |p: &[f32]| {
                _mm256_setr_ps(
                    p[i],
                    p[right],
                    p[below],
                    0.0,
                    p[j],
                    p[next_right],
                    p[next_below],
                    0.0,
                )
            };
            let (_, yv, bv) =
                rgb_to_xyb_f32x8_avx2::<true>(&XybMatrix::SPEC, gather(rp), gather(gp), gather(bp));
            let bv = _mm256_sub_ps(bv, yv);
            let mut ys = [0.0; 8];
            let mut bs = [0.0; 8];
            unsafe {
                _mm256_storeu_ps(ys.as_mut_ptr(), yv);
                _mm256_storeu_ps(bs.as_mut_ptr(), bv);
            }
            for (lane, sx) in [(0, x), (4, next_x)] {
                if sx >= w {
                    break;
                }
                if sx + 1 < w {
                    sum_y += (ys[lane + 1] - ys[lane]).abs();
                    sum_b += (bs[lane + 1] - bs[lane]).abs();
                }
                if has_below {
                    sum_y += (ys[lane + 2] - ys[lane]).abs();
                    sum_b += (bs[lane + 2] - bs[lane]).abs();
                }
            }
        }
    }
    if sum_y > f32::EPSILON {
        sum_b / sum_y
    } else {
        0.0
    }
}
