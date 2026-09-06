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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn excess8(s: &SeamSamples, floor: f32, weight: f32) -> ([f32; 8], f32) {
    let mut out = [0.0; 8];
    let mut peaks = _mm_setzero_ps();
    // SAFETY: each unaligned load/store stays within an eight-element array.
    unsafe {
        let sign = _mm_set1_ps(-0.0);
        for i in [0, 4] {
            let error = _mm_andnot_ps(
                sign,
                _mm_sub_ps(
                    _mm_loadu_ps(s.right.as_ptr().add(i)),
                    _mm_loadu_ps(s.left.as_ptr().add(i)),
                ),
            );
            let gradient = _mm_andnot_ps(
                sign,
                _mm_sub_ps(
                    _mm_loadu_ps(s.source_right.as_ptr().add(i)),
                    _mm_loadu_ps(s.source_left.as_ptr().add(i)),
                ),
            );
            let raw = _mm_sub_ps(
                _mm_sub_ps(error, _mm_mul_ps(gradient, _mm_set1_ps(0.5))),
                _mm_set1_ps(floor),
            );
            let excess = _mm_max_ps(raw, _mm_setzero_ps());
            _mm_storeu_ps(out.as_mut_ptr().add(i), excess);
            peaks = _mm_max_ps(
                peaks,
                _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(weight), excess), excess),
            );
        }
    }
    peaks = _mm_max_ps(peaks, _mm_movehl_ps(peaks, peaks));
    peaks = _mm_max_ss(peaks, _mm_shuffle_ps::<0x55>(peaks, peaks));
    (out, _mm_cvtss_f32(peaks))
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn mosaic_seam_stats_sse41(
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
