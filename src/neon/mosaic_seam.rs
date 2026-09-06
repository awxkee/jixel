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
//! Adaptive transform (AC strategy) selection via rate-distortion optimization.

use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::mosaic_seam::{SeamSamples, mosaic_seam_stats_with};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn excess8(s: &SeamSamples, floor: f32, weight: f32) -> ([f32; 8], f32) {
    let mut out = [0.0; 8];
    let mut peaks = vdupq_n_f32(0.0);
    // SAFETY: each unaligned load/store stays within an eight-element array.
    unsafe {
        for i in [0, 4] {
            let error = vabsq_f32(vsubq_f32(
                vld1q_f32(s.right.as_ptr().add(i)),
                vld1q_f32(s.left.as_ptr().add(i)),
            ));
            let gradient = vabsq_f32(vsubq_f32(
                vld1q_f32(s.source_right.as_ptr().add(i)),
                vld1q_f32(s.source_left.as_ptr().add(i)),
            ));
            // Keep the two subtractions separate, as in the scalar formula.
            let raw = vsubq_f32(
                vsubq_f32(error, vmulq_n_f32(gradient, 0.5)),
                vdupq_n_f32(floor),
            );
            let excess = vmaxnmq_f32(raw, vdupq_n_f32(0.0));
            vst1q_f32(out.as_mut_ptr().add(i), excess);
            peaks = vmaxnmq_f32(peaks, vmulq_f32(vmulq_n_f32(excess, weight), excess));
        }
    }
    (out, vmaxvq_f32(peaks))
}

#[target_feature(enable = "neon")]
pub(crate) fn mosaic_seam_stats_neon(
    ctx: &EncodingContext,
    opsin: &Image3F,
    px: usize,
    py: usize,
    cxb: usize,
    cyb: usize,
    distance: f32,
    selected: &[&[[f32; 64]; 3]],
) -> (f32, f32) {
    mosaic_seam_stats_with(
        ctx,
        opsin,
        px,
        py,
        cxb,
        cyb,
        distance,
        selected,
        |s, floor, weight| excess8(s, floor, weight),
    )
}
