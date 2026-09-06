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
use std::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn excess8(s: &SeamSamples, floor: f32, weight: f32) -> ([f32; 8], f32) {
    let mut out = [0.0; 8];
    let mut peaks = f32x4_splat(0.0);
    // SAFETY: each unaligned load/store stays within an eight-element array.
    unsafe {
        for i in [0, 4] {
            let error = f32x4_abs(f32x4_sub(
                v128_load(s.right.as_ptr().add(i).cast()),
                v128_load(s.left.as_ptr().add(i).cast()),
            ));
            let gradient = f32x4_abs(f32x4_sub(
                v128_load(s.source_right.as_ptr().add(i).cast()),
                v128_load(s.source_left.as_ptr().add(i).cast()),
            ));
            let raw = f32x4_sub(
                f32x4_sub(error, f32x4_mul(gradient, f32x4_splat(0.5))),
                f32x4_splat(floor),
            );
            // Ordered comparison also maps NaN to zero, like f32::max(0.0).
            let excess = v128_and(raw, f32x4_gt(raw, f32x4_splat(0.0)));
            v128_store(out.as_mut_ptr().add(i).cast(), excess);
            peaks = f32x4_max(
                peaks,
                f32x4_mul(f32x4_mul(f32x4_splat(weight), excess), excess),
            );
        }
    }
    let peak = f32x4_extract_lane::<0>(peaks)
        .max(f32x4_extract_lane::<1>(peaks))
        .max(f32x4_extract_lane::<2>(peaks))
        .max(f32x4_extract_lane::<3>(peaks));
    (out, peak)
}

#[target_feature(enable = "simd128")]
pub(crate) fn mosaic_seam_stats_wasm(
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
