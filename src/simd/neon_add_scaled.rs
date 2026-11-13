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

use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn update<const FUSED: bool>(dst: &mut [f32; 4], src: &[f32; 4], scale: float32x4_t) {
    let dst = dst.as_mut_ptr();
    // SAFETY: both arrays contain four elements.
    unsafe {
        let out = vld1q_f32(dst);
        let input = vld1q_f32(src.as_ptr());
        let out = if FUSED {
            vfmaq_f32(out, scale, input)
        } else {
            vaddq_f32(out, vmulq_f32(scale, input))
        };
        vst1q_f32(dst, out);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn add_scaled<const FUSED: bool>(out: &mut [f32], src: &[f32], factor: f32) {
    let src = &src[..out.len()];
    let scale = vdupq_n_f32(factor);
    let (out_blocks, out) = out.as_chunks_mut::<4>();
    let (src_blocks, src) = src.as_chunks::<4>();
    let (out_pairs, out_blocks) = out_blocks.as_chunks_mut::<2>();
    let (src_pairs, src_blocks) = src_blocks.as_chunks::<2>();
    for ([dst0, dst1], [src0, src1]) in out_pairs.iter_mut().zip(src_pairs) {
        update::<FUSED>(dst0, src0, scale);
        update::<FUSED>(dst1, src1, scale);
    }
    for (dst, src) in out_blocks.iter_mut().zip(src_blocks) {
        update::<FUSED>(dst, src, scale);
    }
    if FUSED {
        for (dst, &input) in out.iter_mut().zip(src) {
            *dst = factor.mul_add(input, *dst);
        }
    } else {
        crate::simd::scaled_add::add_scaled_scalar(out, src, factor);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn add_scaled_neon(out: &mut [f32], src: &[f32], factor: f32) {
    add_scaled::<false>(out, src, factor);
}

#[cfg(feature = "splines")]
#[target_feature(enable = "neon")]
pub(crate) fn add_scaled_fma_neon(out: &mut [f32], src: &[f32], factor: f32) {
    add_scaled::<true>(out, src, factor);
}
