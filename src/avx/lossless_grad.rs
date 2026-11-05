/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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
fn grad_pack_vector(
    left: &[i32; 8],
    top: &[i32; 8],
    topleft: &[i32; 8],
    px: &[i32; 8],
    out: &mut [u32; 8],
) {
    let zero = _mm256_setzero_si256();
    let left = unsafe { _mm256_loadu_si256(left.as_ptr() as *const __m256i) };
    let top = unsafe { _mm256_loadu_si256(top.as_ptr() as *const __m256i) };
    let topleft = unsafe { _mm256_loadu_si256(topleft.as_ptr() as *const __m256i) };
    let px = unsafe { _mm256_loadu_si256(px.as_ptr() as *const __m256i) };
    let ac = _mm256_sub_epi32(left, topleft);
    let ab = _mm256_sub_epi32(left, top);
    let bc = _mm256_sub_epi32(top, topleft);
    let grad = _mm256_add_epi32(ac, top);
    let d = _mm256_xor_si256(ab, bc);
    let clamp = _mm256_blendv_epi8(left, top, _mm256_cmpgt_epi32(zero, d)); // d<0 ? top : left
    let s = _mm256_xor_si256(ac, bc);
    let pred = _mm256_blendv_epi8(clamp, grad, _mm256_cmpgt_epi32(zero, s)); // s<0 ? grad : clamp
    let res = _mm256_sub_epi32(px, pred);
    let packed = _mm256_xor_si256(_mm256_slli_epi32::<1>(res), _mm256_srai_epi32::<31>(res));
    unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, packed) };
}

#[target_feature(enable = "avx2")]
pub(crate) fn grad_pack_interior(cur: &[i32], prev: &[i32], out: &mut [u32], gw: usize) {
    assert!(cur.len() >= gw && prev.len() >= gw && out.len() >= gw);
    if gw <= 1 {
        return;
    }
    let (left, left_tail) = cur[..gw - 1].as_chunks::<8>();
    let (top, top_tail) = prev[1..gw].as_chunks::<8>();
    let (topleft, topleft_tail) = prev[..gw - 1].as_chunks::<8>();
    let (px, px_tail) = cur[1..gw].as_chunks::<8>();
    let (out, out_tail) = out[1..gw].as_chunks_mut::<8>();

    let (left_pairs, left_single) = left.as_chunks::<2>();
    let (top_pairs, top_single) = top.as_chunks::<2>();
    let (topleft_pairs, topleft_single) = topleft.as_chunks::<2>();
    let (px_pairs, px_single) = px.as_chunks::<2>();
    let (out_pairs, out_single) = out.as_chunks_mut::<2>();
    for ((((left, top), topleft), px), out) in left_pairs
        .iter()
        .zip(top_pairs)
        .zip(topleft_pairs)
        .zip(px_pairs)
        .zip(out_pairs)
    {
        grad_pack_vector(&left[0], &top[0], &topleft[0], &px[0], &mut out[0]);
        grad_pack_vector(&left[1], &top[1], &topleft[1], &px[1], &mut out[1]);
    }
    for ((((left, top), topleft), px), out) in left_single
        .iter()
        .zip(top_single)
        .zip(topleft_single)
        .zip(px_single)
        .zip(out_single)
    {
        grad_pack_vector(left, top, topleft, px, out);
    }
    for ((((&w, &n), &nw), &value), out) in left_tail
        .iter()
        .zip(top_tail)
        .zip(topleft_tail)
        .zip(px_tail)
        .zip(out_tail)
    {
        let ac = w.wrapping_sub(nw);
        let bc = n.wrapping_sub(nw);
        let grad = ac.wrapping_add(n);
        let clamp = if (w.wrapping_sub(n) ^ bc) < 0 { n } else { w };
        let pred = if (ac ^ bc) < 0 { grad } else { clamp };
        let r = value.wrapping_sub(pred) as i64;
        *out = ((r << 1) ^ (r >> 63)) as u32;
    }
}
