/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn round_ties_away(x: __m128) -> __m128 {
    let sign = _mm_and_ps(x, _mm_set1_ps(-0.0));
    let truncated = _mm_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x);
    let fraction = _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(x, truncated));
    let adjust = _mm_cmpge_ps(fraction, _mm_set1_ps(0.5));
    let signed_one = _mm_or_ps(_mm_set1_ps(1.0), sign);
    // Adding +0 can erase -0; restore the original sign after rounding.
    _mm_or_ps(_mm_add_ps(truncated, _mm_and_ps(signed_one, adjust)), sign)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn dzq(x: __m128, q: __m128, offset: __m128) -> __m128 {
    let sign_mask = _mm_set1_ps(-0.0);
    let magnitude = _mm_andnot_ps(sign_mask, x);
    let magnitude = _mm_mul_ps(
        _mm_floor_ps(_mm_add_ps(_mm_div_ps(magnitude, q), offset)),
        q,
    );
    _mm_or_ps(
        _mm_andnot_ps(sign_mask, magnitude),
        _mm_and_ps(sign_mask, x),
    )
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn haar_rows_sse41(
    top: &[f32],
    bottom: &[f32],
    q_hf: f32,
    top_out: &mut [f32],
    bottom_out: &mut [f32],
) -> usize {
    let quarter = _mm_set1_ps(0.25);
    let low_scale = _mm_set1_ps(0.25 / CHROMA_DC_SQUEEZE_STEP);
    let step = _mm_set1_ps(CHROMA_DC_SQUEEZE_STEP);
    let q = _mm_set1_ps(q_hf);
    let offset = _mm_set1_ps(1.0 - CHROMA_DC_SQUEEZE_DEADZONE);
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
        let (top0, top1, bottom0, bottom1) = unsafe {
            (
                _mm_loadu_ps(top.as_ptr()),
                _mm_loadu_ps(top.as_ptr().add(4)),
                _mm_loadu_ps(bottom.as_ptr()),
                _mm_loadu_ps(bottom.as_ptr().add(4)),
            )
        };
        // Deinterleave four independent 2x2 blocks.
        let a = _mm_shuffle_ps::<0x88>(top0, top1);
        let b = _mm_shuffle_ps::<0xdd>(top0, top1);
        let c = _mm_shuffle_ps::<0x88>(bottom0, bottom1);
        let d = _mm_shuffle_ps::<0xdd>(bottom0, bottom1);
        let ll = _mm_mul_ps(
            round_ties_away(_mm_mul_ps(
                _mm_add_ps(_mm_add_ps(_mm_add_ps(a, b), c), d),
                low_scale,
            )),
            step,
        );
        let hl = dzq(
            _mm_mul_ps(_mm_sub_ps(_mm_add_ps(_mm_sub_ps(a, b), c), d), quarter),
            q,
            offset,
        );
        let lh = dzq(
            _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(_mm_add_ps(a, b), c), d), quarter),
            q,
            offset,
        );
        let hh = dzq(
            _mm_mul_ps(_mm_add_ps(_mm_sub_ps(_mm_sub_ps(a, b), c), d), quarter),
            q,
            offset,
        );
        let a_out = _mm_add_ps(_mm_add_ps(_mm_add_ps(ll, hl), lh), hh);
        let b_out = _mm_sub_ps(_mm_add_ps(_mm_sub_ps(ll, hl), lh), hh);
        let c_out = _mm_sub_ps(_mm_sub_ps(_mm_add_ps(ll, hl), lh), hh);
        let d_out = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(ll, hl), lh), hh);
        unsafe {
            _mm_storeu_ps(top_out.as_mut_ptr(), _mm_unpacklo_ps(a_out, b_out));
            _mm_storeu_ps(top_out.as_mut_ptr().add(4), _mm_unpackhi_ps(a_out, b_out));
            _mm_storeu_ps(bottom_out.as_mut_ptr(), _mm_unpacklo_ps(c_out, d_out));
            _mm_storeu_ps(
                bottom_out.as_mut_ptr().add(4),
                _mm_unpackhi_ps(c_out, d_out),
            );
        }
        done += 8;
    }
    done
}
