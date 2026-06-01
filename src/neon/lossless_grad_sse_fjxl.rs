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
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn grad_pack_interior(cur: &[i32], prev: &[i32], out: &mut [u32], gw: usize) {
    assert!(cur.len() >= gw && prev.len() >= gw && out.len() >= gw);
    let zero = vdupq_n_s32(0);
    let mut gx = 1usize;
    while gx + 4 <= gw {
        let left = unsafe { vld1q_s32(cur.as_ptr().add(gx - 1)) };
        let top = unsafe { vld1q_s32(prev.as_ptr().add(gx)) };
        let topleft = unsafe { vld1q_s32(prev.as_ptr().add(gx - 1)) };
        let px = unsafe { vld1q_s32(cur.as_ptr().add(gx)) };
        let ac = vsubq_s32(left, topleft);
        let ab = vsubq_s32(left, top);
        let bc = vsubq_s32(top, topleft);
        let grad = vaddq_s32(ac, top);
        let d = veorq_s32(ab, bc);
        let clamp = vbslq_s32(vcltq_s32(d, zero), top, left); // d<0 ? top : left
        let s = veorq_s32(ac, bc);
        let pred = vbslq_s32(vcltq_s32(s, zero), grad, clamp); // s<0 ? grad : clamp
        let res = vsubq_s32(px, pred);
        // pack_signed: (res << 1) ^ (res >> 31)
        let packed = veorq_s32(vshlq_n_s32::<1>(res), vshrq_n_s32::<31>(res));
        unsafe { vst1q_u32(out.as_mut_ptr().add(gx), vreinterpretq_u32_s32(packed)) };
        gx += 4;
    }
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
