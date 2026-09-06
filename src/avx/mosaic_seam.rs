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
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn excess8(s: &SeamSamples, floor: f32, weight: f32) -> ([f32; 8], f32) {
    let mut out = [0.0; 8];
    let mut peaks;
    // SAFETY: each unaligned load/store stays within an eight-element array.
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        let error = _mm256_andnot_ps(
            sign,
            _mm256_sub_ps(
                _mm256_loadu_ps(s.right.as_ptr()),
                _mm256_loadu_ps(s.left.as_ptr()),
            ),
        );
        let gradient = _mm256_andnot_ps(
            sign,
            _mm256_sub_ps(
                _mm256_loadu_ps(s.source_right.as_ptr()),
                _mm256_loadu_ps(s.source_left.as_ptr()),
            ),
        );
        let raw = _mm256_sub_ps(
            _mm256_sub_ps(error, _mm256_mul_ps(gradient, _mm256_set1_ps(0.5))),
            _mm256_set1_ps(floor),
        );
        let excess = _mm256_max_ps(raw, _mm256_setzero_ps());
        _mm256_storeu_ps(out.as_mut_ptr(), excess);
        let weighted = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(weight), excess), excess);
        peaks = _mm_max_ps(
            _mm256_castps256_ps128(weighted),
            _mm256_extractf128_ps::<1>(weighted),
        );
    }
    peaks = _mm_max_ps(peaks, _mm_movehl_ps(peaks, peaks));
    peaks = _mm_max_ss(peaks, _mm_shuffle_ps::<0x55>(peaks, peaks));
    (out, _mm_cvtss_f32(peaks))
}

#[target_feature(enable = "avx2")]
pub(crate) fn mosaic_seam_stats_avx2(
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
