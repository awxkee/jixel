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
use core::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub(crate) fn grad_pack_interior(cur: &[i32], prev: &[i32], out: &mut [u32], gw: usize) {
    assert!(cur.len() >= gw && prev.len() >= gw && out.len() >= gw);
    let zero = i32x4_splat(0);
    let mut gx = 1usize;
    while gx + 4 <= gw {
        let left = unsafe { v128_load(cur.as_ptr().add(gx - 1).cast()) };
        let top = unsafe { v128_load(prev.as_ptr().add(gx).cast()) };
        let topleft = unsafe { v128_load(prev.as_ptr().add(gx - 1).cast()) };
        let px = unsafe { v128_load(cur.as_ptr().add(gx).cast()) };
        let ac = i32x4_sub(left, topleft);
        let ab = i32x4_sub(left, top);
        let bc = i32x4_sub(top, topleft);
        let grad = i32x4_add(ac, top);
        let d = v128_xor(ab, bc);
        let clamp = v128_bitselect(top, left, i32x4_lt(d, zero)); // d<0 ? top : left
        let s = v128_xor(ac, bc);
        let pred = v128_bitselect(grad, clamp, i32x4_lt(s, zero)); // s<0 ? grad : clamp
        let res = i32x4_sub(px, pred);
        // pack_signed: (res << 1) ^ (res >> 31)
        let packed = v128_xor(i32x4_shl(res, 1), i32x4_shr(res, 31));
        unsafe { v128_store(out.as_mut_ptr().add(gx).cast(), packed) };
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
