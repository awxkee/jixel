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
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn round_ties_away_i32(v: __m256) -> __m256i {
    let sign = _mm256_set1_ps(-0.0);
    let truncated = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(v);
    let fraction = _mm256_sub_ps(v, truncated);
    let abs_fraction = _mm256_andnot_ps(sign, fraction);
    let at_least_half = _mm256_cmp_ps::<_CMP_GE_OQ>(abs_fraction, _mm256_set1_ps(0.5));
    let signed_one = _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_and_ps(v, sign));
    let rounded = _mm256_add_ps(truncated, _mm256_and_ps(signed_one, at_least_half));
    let q = _mm256_cvttps_epi32(rounded);
    let positive_overflow = _mm256_cmp_ps::<_CMP_GE_OQ>(rounded, _mm256_set1_ps(2_147_483_648.0));
    let q = _mm256_blendv_epi8(
        q,
        _mm256_set1_epi32(i32::MAX),
        _mm256_castps_si256(positive_overflow),
    );
    let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(v, v);
    _mm256_blendv_epi8(q, _mm256_setzero_si256(), _mm256_castps_si256(nan))
}

#[inline]
#[target_feature(enable = "avx2")]
fn interleave3_i32x8(y: __m256i, x: __m256i, b: __m256i) -> [__m256i; 3] {
    let yx_lo = _mm256_unpacklo_epi32(y, x);
    let yx_hi = _mm256_unpackhi_epi32(y, x);

    // First interleave each 128-bit lane independently.
    let lane0 = _mm256_blend_epi32::<0x44>(
        _mm256_shuffle_epi32::<0x84>(yx_lo),
        _mm256_shuffle_epi32::<0x00>(b),
    );
    let lane1 = _mm256_blend_epi32::<0x22>(
        _mm256_castps_si256(_mm256_shuffle_ps::<0x43>(
            _mm256_castsi256_ps(yx_lo),
            _mm256_castsi256_ps(yx_hi),
        )),
        _mm256_shuffle_epi32::<0x55>(b),
    );
    let lane2 = _mm256_blend_epi32::<0x99>(
        _mm256_shuffle_epi32::<0x38>(yx_hi),
        _mm256_shuffle_epi32::<0xc2>(b),
    );

    // Join [low lane0, low lane1], [low lane2, high lane0],
    // and [high lane1, high lane2] into the 24-word AoS stream.
    [
        _mm256_permute2x128_si256::<0x20>(lane0, lane1),
        _mm256_permute2x128_si256::<0x30>(lane2, lane0),
        _mm256_permute2x128_si256::<0x31>(lane1, lane2),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_interleaved3_i32x8(output: &mut [[i32; 3]; 8], y: __m256i, x: __m256i, b: __m256i) {
    let [out0, out1, out2] = interleave3_i32x8(y, x, b);
    let dst = output.as_mut_ptr().cast::<i32>();
    unsafe {
        _mm256_storeu_si256(dst.cast(), out0);
        _mm256_storeu_si256(dst.add(8).cast(), out1);
        _mm256_storeu_si256(dst.add(16).cast(), out2);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn active_lane_mask(len: usize) -> __m256i {
    _mm256_cmpgt_epi32(
        _mm256_set1_epi32(len as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_interleaved3_i32x8_tail(output: &mut [[i32; 3]], y: __m256i, x: __m256i, b: __m256i) {
    debug_assert!(!output.is_empty() && output.len() < 8);
    let [out0, out1, out2] = interleave3_i32x8(y, x, b);
    let words = output.len() * 3;
    let dst = output.as_mut_ptr().cast::<i32>();
    unsafe {
        if words < 8 {
            _mm256_maskstore_epi32(dst, active_lane_mask(words), out0);
            return;
        }
        _mm256_storeu_si256(dst.cast(), out0);
        if words < 16 {
            _mm256_maskstore_epi32(dst.add(8), active_lane_mask(words - 8), out1);
            return;
        }
        _mm256_storeu_si256(dst.add(8).cast(), out1);
        _mm256_maskstore_epi32(dst.add(16), active_lane_mask(words - 16), out2);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn quantize_i32x8(
    x: __m256,
    y: __m256,
    b: __m256,
    sx: __m256,
    sy: __m256,
    sb: __m256,
) -> [__m256i; 3] {
    [
        round_ties_away_i32(_mm256_mul_ps(y, sy)),
        round_ties_away_i32(_mm256_mul_ps(x, sx)),
        round_ties_away_i32(_mm256_mul_ps(b, sb)),
    ]
}

#[target_feature(enable = "avx2")]
pub(crate) fn quantize_xyb_channels_avx2(
    input: [&[f32]; 3],
    output: [&mut [i32]; 3],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [dst_y, dst_x, dst_b] = output;
    let [scale_x, scale_y, scale_b] = scales;
    let (x8s, x_tail) = src_x.as_chunks::<8>();
    let (y8s, y_tail) = src_y.as_chunks::<8>();
    let (b8s, b_tail) = src_b.as_chunks::<8>();
    let (yq8s, yq_tail) = dst_y.as_chunks_mut::<8>();
    let (xq8s, xq_tail) = dst_x.as_chunks_mut::<8>();
    let (bq8s, bq_tail) = dst_b.as_chunks_mut::<8>();
    let sx = _mm256_set1_ps(scale_x);
    let sy = _mm256_set1_ps(scale_y);
    let sb = _mm256_set1_ps(scale_b);

    for (((((x8, y8), b8), yq8), xq8), bq8) in
        x8s.iter().zip(y8s).zip(b8s).zip(yq8s).zip(xq8s).zip(bq8s)
    {
        let x = unsafe { _mm256_loadu_ps(x8.as_ptr()) };
        let y = unsafe { _mm256_loadu_ps(y8.as_ptr()) };
        let b = unsafe { _mm256_loadu_ps(b8.as_ptr()) };
        let [yq, xq, bq] = quantize_i32x8(x, y, b, sx, sy, sb);
        unsafe {
            _mm256_storeu_si256(yq8.as_mut_ptr().cast(), yq);
            _mm256_storeu_si256(xq8.as_mut_ptr().cast(), xq);
            _mm256_storeu_si256(bq8.as_mut_ptr().cast(), _mm256_sub_epi32(bq, yq));
        }
    }

    if !x_tail.is_empty() {
        debug_assert_eq!(x_tail.len(), y_tail.len());
        debug_assert_eq!(x_tail.len(), b_tail.len());
        let mask = active_lane_mask(x_tail.len());
        let x = unsafe { _mm256_maskload_ps(x_tail.as_ptr(), mask) };
        let y = unsafe { _mm256_maskload_ps(y_tail.as_ptr(), mask) };
        let b = unsafe { _mm256_maskload_ps(b_tail.as_ptr(), mask) };
        let [yq, xq, bq] = quantize_i32x8(x, y, b, sx, sy, sb);
        unsafe {
            _mm256_maskstore_epi32(yq_tail.as_mut_ptr(), mask, yq);
            _mm256_maskstore_epi32(xq_tail.as_mut_ptr(), mask, xq);
            _mm256_maskstore_epi32(bq_tail.as_mut_ptr(), mask, _mm256_sub_epi32(bq, yq));
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn quantize_xyb_tile_colors_avx2(
    input: [&[f32]; 3],
    output: &mut [[i32; 3]],
    scales: [f32; 3],
) {
    let [src_x, src_y, src_b] = input;
    let [scale_x, scale_y, scale_b] = scales;
    let (x8s, x_tail) = src_x.as_chunks::<8>();
    let (y8s, y_tail) = src_y.as_chunks::<8>();
    let (b8s, b_tail) = src_b.as_chunks::<8>();
    let (out8s, out_tail) = output.as_chunks_mut::<8>();
    let sx = _mm256_set1_ps(scale_x);
    let sy = _mm256_set1_ps(scale_y);
    let sb = _mm256_set1_ps(scale_b);

    for (((x8, y8), b8), out8) in x8s.iter().zip(y8s).zip(b8s).zip(out8s) {
        let x = unsafe { _mm256_loadu_ps(x8.as_ptr()) };
        let y = unsafe { _mm256_loadu_ps(y8.as_ptr()) };
        let b = unsafe { _mm256_loadu_ps(b8.as_ptr()) };
        let [yq, xq, bq] = quantize_i32x8(x, y, b, sx, sy, sb);
        store_interleaved3_i32x8(out8, yq, xq, bq);
    }

    if !x_tail.is_empty() {
        debug_assert_eq!(x_tail.len(), y_tail.len());
        debug_assert_eq!(x_tail.len(), b_tail.len());
        debug_assert_eq!(x_tail.len(), out_tail.len());
        let mask = active_lane_mask(x_tail.len());
        let x = unsafe { _mm256_maskload_ps(x_tail.as_ptr(), mask) };
        let y = unsafe { _mm256_maskload_ps(y_tail.as_ptr(), mask) };
        let b = unsafe { _mm256_maskload_ps(b_tail.as_ptr(), mask) };
        let [yq, xq, bq] = quantize_i32x8(x, y, b, sx, sy, sb);
        store_interleaved3_i32x8_tail(out_tail, yq, xq, bq);
    }
}
