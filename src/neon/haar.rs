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
use crate::haar::{CHROMA_DC_SQUEEZE_DEADZONE, CHROMA_DC_SQUEEZE_STEP};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn dzq(x: float32x4_t, q: float32x4_t, offset: float32x4_t) -> float32x4_t {
    let magnitude = vmulq_f32(vrndmq_f32(vaddq_f32(vdivq_f32(vabsq_f32(x), q), offset)), q);
    vbslq_f32(vdupq_n_u32(0x8000_0000), x, magnitude)
}

#[target_feature(enable = "neon")]
pub(crate) fn haar_rows_neon(
    top: &[f32],
    bottom: &[f32],
    q_hf: f32,
    top_out: &mut [f32],
    bottom_out: &mut [f32],
) -> usize {
    let quarter = vdupq_n_f32(0.25);
    let low_scale = vdupq_n_f32(0.25 / CHROMA_DC_SQUEEZE_STEP);
    let step = vdupq_n_f32(CHROMA_DC_SQUEEZE_STEP);
    let q = vdupq_n_f32(q_hf);
    let offset = vdupq_n_f32(1.0 - CHROMA_DC_SQUEEZE_DEADZONE);
    let mut done = 0;
    for ((top, bottom), (top_out, bottom_out)) in top
        .as_chunks::<8>()
        .0
        .iter()
        .zip(bottom.as_chunks::<8>().0)
        .zip(
            top_out
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(bottom_out.as_chunks_mut::<8>().0),
        )
    {
        // Deinterleave four independent 2x2 blocks.
        let float32x4x2_t(a, b) = unsafe { vld2q_f32(top.as_ptr()) };
        let float32x4x2_t(c, d) = unsafe { vld2q_f32(bottom.as_ptr()) };
        let ll = vmulq_f32(
            vrndaq_f32(vmulq_f32(
                vaddq_f32(vaddq_f32(vaddq_f32(a, b), c), d),
                low_scale,
            )),
            step,
        );
        let hl = dzq(
            vmulq_f32(vsubq_f32(vaddq_f32(vsubq_f32(a, b), c), d), quarter),
            q,
            offset,
        );
        let lh = dzq(
            vmulq_f32(vsubq_f32(vsubq_f32(vaddq_f32(a, b), c), d), quarter),
            q,
            offset,
        );
        let hh = dzq(
            vmulq_f32(vaddq_f32(vsubq_f32(vsubq_f32(a, b), c), d), quarter),
            q,
            offset,
        );
        let a_out = vaddq_f32(vaddq_f32(vaddq_f32(ll, hl), lh), hh);
        let b_out = vsubq_f32(vaddq_f32(vsubq_f32(ll, hl), lh), hh);
        let c_out = vsubq_f32(vsubq_f32(vaddq_f32(ll, hl), lh), hh);
        let d_out = vaddq_f32(vsubq_f32(vsubq_f32(ll, hl), lh), hh);
        unsafe {
            vst2q_f32(top_out.as_mut_ptr(), float32x4x2_t(a_out, b_out));
            vst2q_f32(bottom_out.as_mut_ptr(), float32x4x2_t(c_out, d_out));
        }
        done += 8;
    }
    done
}
