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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub(crate) fn grad_pack_interior(cur: &[i32], prev: &[i32], out: &mut [u32], gw: usize) {
    assert!(cur.len() >= gw && prev.len() >= gw && out.len() >= gw);
    let zero = _mm_setzero_si128();
    let mut gx = 1usize;
    while gx + 4 <= gw {
        let left = unsafe { _mm_loadu_si128(cur.as_ptr().add(gx - 1) as *const __m128i) };
        let top = unsafe { _mm_loadu_si128(prev.as_ptr().add(gx) as *const __m128i) };
        let topleft = unsafe { _mm_loadu_si128(prev.as_ptr().add(gx - 1) as *const __m128i) };
        let px = unsafe { _mm_loadu_si128(cur.as_ptr().add(gx) as *const __m128i) };

        // fjxl PredictPixels: grad = (left-topleft) + top; clamp to the nearer of
        // {left, top} via the XOR-sign of the neighbor differences.
        let ac = _mm_sub_epi32(left, topleft);
        let ab = _mm_sub_epi32(left, top);
        let bc = _mm_sub_epi32(top, topleft);
        let grad = _mm_add_epi32(ac, top);
        let d = _mm_xor_si128(ab, bc);
        let clamp = _mm_blendv_epi8(left, top, _mm_cmpgt_epi32(zero, d)); // d<0 ? top : left
        let s = _mm_xor_si128(ac, bc);
        let pred = _mm_blendv_epi8(clamp, grad, _mm_cmpgt_epi32(zero, s)); // s<0 ? grad : clamp

        let res = _mm_sub_epi32(px, pred);
        // pack_signed: (res << 1) ^ (res >> 31)  (zig-zag), exact in i32.
        let packed = _mm_xor_si128(_mm_slli_epi32::<1>(res), _mm_srai_epi32::<31>(res));
        unsafe { _mm_storeu_si128(out.as_mut_ptr().add(gx) as *mut __m128i, packed) };
        gx += 4;
    }
    // Scalar tail (identical math).
    while gx < gw {
        let w = cur[gx - 1];
        let n = prev[gx];
        let nw = prev[gx - 1];
        let ac = w.wrapping_sub(nw);
        let bc = n.wrapping_sub(nw);
        let grad = ac.wrapping_add(n);
        let clamp = if (w.wrapping_sub(n) ^ bc) < 0 { n } else { w };
        let pred = if (ac ^ bc) < 0 { grad } else { clamp };
        let r = cur[gx].wrapping_sub(pred) as i64;
        out[gx] = ((r << 1) ^ (r >> 63)) as u32;
        gx += 1;
    }
}
