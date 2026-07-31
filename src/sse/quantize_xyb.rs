/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn round_ties_away_i32(v: __m128) -> __m128i {
    let sign = _mm_set1_ps(-0.0);
    let truncated = _mm_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(v);
    let fraction = _mm_sub_ps(v, truncated);
    let abs_fraction = _mm_andnot_ps(sign, fraction);
    let at_least_half = _mm_cmpge_ps(abs_fraction, _mm_set1_ps(0.5));
    let signed_one = _mm_or_ps(_mm_set1_ps(1.0), _mm_and_ps(v, sign));
    let rounded = _mm_add_ps(truncated, _mm_and_ps(signed_one, at_least_half));
    let q = _mm_cvttps_epi32(rounded);
    let positive_overflow = _mm_cmpge_ps(rounded, _mm_set1_ps(2_147_483_648.0));
    let q = _mm_blendv_epi8(
        q,
        _mm_set1_epi32(i32::MAX),
        _mm_castps_si128(positive_overflow),
    );
    let nan = _mm_cmpunord_ps(v, v);
    _mm_blendv_epi8(q, _mm_setzero_si128(), _mm_castps_si128(nan))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_interleaved3_i32x4(output: &mut [[i32; 3]; 4], y: __m128i, x: __m128i, b: __m128i) {
    let yx_lo = _mm_unpacklo_epi32(y, x);
    let yx_hi = _mm_unpackhi_epi32(y, x);
    // Store [Y0 X0 B0 Y1], [X1 B1 Y2 X2], [B2 Y3 X3 B3].
    let out0 = _mm_blend_epi16::<0x30>(
        _mm_shuffle_epi32::<0x84>(yx_lo),
        _mm_shuffle_epi32::<0x00>(b),
    );
    let out1 = _mm_blend_epi16::<0x0c>(
        _mm_castps_si128(_mm_shuffle_ps::<0x43>(
            _mm_castsi128_ps(yx_lo),
            _mm_castsi128_ps(yx_hi),
        )),
        _mm_shuffle_epi32::<0x55>(b),
    );
    let out2 = _mm_blend_epi16::<0xc3>(
        _mm_shuffle_epi32::<0x38>(yx_hi),
        _mm_shuffle_epi32::<0xc2>(b),
    );
    let dst = output.as_mut_ptr().cast::<i32>();
    unsafe {
        _mm_storeu_si128(dst.cast(), out0);
        _mm_storeu_si128(dst.add(4).cast(), out1);
        _mm_storeu_si128(dst.add(8).cast(), out2);
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn quantize_xyb_channels_sse41(
    input: [&[f32]; 3],
    output: [&mut [i32]; 3],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [dst_y, dst_x, dst_b] = output;
    let [scale_x, scale_y, scale_b] = scales;
    let (x4s, x_tail) = src_x.as_chunks::<4>();
    let (y4s, y_tail) = src_y.as_chunks::<4>();
    let (b4s, b_tail) = src_b.as_chunks::<4>();
    let (yq4s, yq_tail) = dst_y.as_chunks_mut::<4>();
    let (xq4s, xq_tail) = dst_x.as_chunks_mut::<4>();
    let (bq4s, bq_tail) = dst_b.as_chunks_mut::<4>();
    let sx = _mm_set1_ps(scale_x);
    let sy = _mm_set1_ps(scale_y);
    let sb = _mm_set1_ps(scale_b);

    for (((((x4, y4), b4), yq4), xq4), bq4) in
        x4s.iter().zip(y4s).zip(b4s).zip(yq4s).zip(xq4s).zip(bq4s)
    {
        let x = unsafe { _mm_loadu_ps(x4.as_ptr()) };
        let y = unsafe { _mm_loadu_ps(y4.as_ptr()) };
        let b = unsafe { _mm_loadu_ps(b4.as_ptr()) };
        let yq = round_ties_away_i32(_mm_mul_ps(y, sy));
        let xq = round_ties_away_i32(_mm_mul_ps(x, sx));
        let bq = round_ties_away_i32(_mm_mul_ps(b, sb));
        unsafe {
            _mm_storeu_si128(yq4.as_mut_ptr().cast(), yq);
            _mm_storeu_si128(xq4.as_mut_ptr().cast(), xq);
            _mm_storeu_si128(bq4.as_mut_ptr().cast(), _mm_sub_epi32(bq, yq));
        }
    }

    let src = x_tail.iter().zip(y_tail).zip(b_tail);
    let dst = yq_tail.iter_mut().zip(xq_tail).zip(bq_tail);
    for (((x, y), b), ((yq, xq), bq)) in src.zip(dst) {
        let quantized_y = (*y * scale_y).round() as i32;
        *yq = quantized_y;
        *xq = (*x * scale_x).round() as i32;
        *bq = (*b * scale_b).round() as i32 - quantized_y;
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn quantize_xyb_tile_colors_sse41(
    input: [&[f32]; 3],
    output: &mut [[i32; 3]],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [scale_x, scale_y, scale_b] = scales;
    let (x4s, x_tail) = src_x.as_chunks::<4>();
    let (y4s, y_tail) = src_y.as_chunks::<4>();
    let (b4s, b_tail) = src_b.as_chunks::<4>();
    let (out4s, out_tail) = output.as_chunks_mut::<4>();
    let sx = _mm_set1_ps(scale_x);
    let sy = _mm_set1_ps(scale_y);
    let sb = _mm_set1_ps(scale_b);

    for (((x4, y4), b4), out4) in x4s.iter().zip(y4s).zip(b4s).zip(out4s) {
        let x = unsafe { _mm_loadu_ps(x4.as_ptr()) };
        let y = unsafe { _mm_loadu_ps(y4.as_ptr()) };
        let b = unsafe { _mm_loadu_ps(b4.as_ptr()) };
        let yq = round_ties_away_i32(_mm_mul_ps(y, sy));
        let xq = round_ties_away_i32(_mm_mul_ps(x, sx));
        let bq = round_ties_away_i32(_mm_mul_ps(b, sb));
        store_interleaved3_i32x4(out4, yq, xq, bq);
    }

    let src = x_tail.iter().zip(y_tail).zip(b_tail);
    for (((x, y), b), out) in src.zip(out_tail) {
        *out = [
            (*y * scale_y).round() as i32,
            (*x * scale_x).round() as i32,
            (*b * scale_b).round() as i32,
        ];
    }
}
