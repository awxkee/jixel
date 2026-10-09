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
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn round_ties_away(x: __m256) -> __m256 {
    let sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0));
    let truncated = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x);
    let fraction = _mm256_andnot_ps(_mm256_set1_ps(-0.0), _mm256_sub_ps(x, truncated));
    let adjust = _mm256_cmp_ps::<_CMP_GE_OQ>(fraction, _mm256_set1_ps(0.5));
    let signed_one = _mm256_or_ps(_mm256_set1_ps(1.0), sign);
    // Adding +0 can erase -0; restore the original sign after rounding.
    _mm256_or_ps(
        _mm256_add_ps(truncated, _mm256_and_ps(signed_one, adjust)),
        sign,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn dzq(x: __m256, q: __m256, offset: __m256) -> __m256 {
    let sign_mask = _mm256_set1_ps(-0.0);
    let magnitude = _mm256_andnot_ps(sign_mask, x);
    let magnitude = _mm256_mul_ps(
        _mm256_floor_ps(_mm256_add_ps(_mm256_div_ps(magnitude, q), offset)),
        q,
    );
    _mm256_or_ps(
        _mm256_andnot_ps(sign_mask, magnitude),
        _mm256_and_ps(sign_mask, x),
    )
}

#[target_feature(enable = "avx2")]
pub(crate) fn haar_rows_avx2(
    top: &[f32],
    bottom: &[f32],
    q_hf: f32,
    top_out: &mut [f32],
    bottom_out: &mut [f32],
) -> usize {
    let quarter = _mm256_set1_ps(0.25);
    let low_scale = _mm256_set1_ps(0.25 / CHROMA_DC_SQUEEZE_STEP);
    let step = _mm256_set1_ps(CHROMA_DC_SQUEEZE_STEP);
    let q = _mm256_set1_ps(q_hf);
    let offset = _mm256_set1_ps(1.0 - CHROMA_DC_SQUEEZE_DEADZONE);
    let mut done = 0;
    for ((top, bottom), (top_out, bottom_out)) in top
        .as_chunks::<16>()
        .0
        .iter()
        .zip(bottom.as_chunks::<16>().0)
        .zip(
            top_out
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(bottom_out.as_chunks_mut::<16>().0),
        )
    {
        let (top0, top1, bottom0, bottom1) = unsafe {
            (
                _mm256_loadu_ps(top.as_ptr()),
                _mm256_loadu_ps(top.as_ptr().add(8)),
                _mm256_loadu_ps(bottom.as_ptr()),
                _mm256_loadu_ps(bottom.as_ptr().add(8)),
            )
        };
        // AVX shuffles stay within 128-bit lanes; reconstruction uses the same order.
        let a = _mm256_shuffle_ps::<0x88>(top0, top1);
        let b = _mm256_shuffle_ps::<0xdd>(top0, top1);
        let c = _mm256_shuffle_ps::<0x88>(bottom0, bottom1);
        let d = _mm256_shuffle_ps::<0xdd>(bottom0, bottom1);
        let ll = _mm256_mul_ps(
            round_ties_away(_mm256_mul_ps(
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a, b), c), d),
                low_scale,
            )),
            step,
        );
        let hl = dzq(
            _mm256_mul_ps(
                _mm256_sub_ps(_mm256_add_ps(_mm256_sub_ps(a, b), c), d),
                quarter,
            ),
            q,
            offset,
        );
        let lh = dzq(
            _mm256_mul_ps(
                _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(a, b), c), d),
                quarter,
            ),
            q,
            offset,
        );
        let hh = dzq(
            _mm256_mul_ps(
                _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(a, b), c), d),
                quarter,
            ),
            q,
            offset,
        );
        let a_out = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(ll, hl), lh), hh);
        let b_out = _mm256_sub_ps(_mm256_add_ps(_mm256_sub_ps(ll, hl), lh), hh);
        let c_out = _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(ll, hl), lh), hh);
        let d_out = _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(ll, hl), lh), hh);
        unsafe {
            _mm256_storeu_ps(top_out.as_mut_ptr(), _mm256_unpacklo_ps(a_out, b_out));
            _mm256_storeu_ps(
                top_out.as_mut_ptr().add(8),
                _mm256_unpackhi_ps(a_out, b_out),
            );
            _mm256_storeu_ps(bottom_out.as_mut_ptr(), _mm256_unpacklo_ps(c_out, d_out));
            _mm256_storeu_ps(
                bottom_out.as_mut_ptr().add(8),
                _mm256_unpackhi_ps(c_out, d_out),
            );
        }
        done += 16;
    }
    done
}
