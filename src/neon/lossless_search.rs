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

use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
fn shr<const N: i32>(v: int32x4_t) -> int32x4_t {
    vshrq_n_s32::<N>(v)
}

#[target_feature(enable = "neon")]
fn gt(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vreinterpretq_s32_u32(vcgtq_s32(a, b))
}

#[target_feature(enable = "neon")]
pub(crate) fn rct_row_neon(src: [&[i32]; 3], kind: u32, out: [&mut [i32]; 3]) {
    let [a, b, c] = out;
    let len = a.len();
    assert!(src.iter().all(|s| s.len() >= len) && b.len() >= len && c.len() >= len);
    assert!(kind < 42);
    if kind == 6 {
        for (out, src) in [a, b, c].into_iter().zip(src) {
            out.copy_from_slice(src);
        }
        return;
    }
    let perm = (kind / 7) as usize;
    let perm = [
        perm % 3,
        (perm + 1 + perm / 3) % 3,
        (perm + 2 - perm / 3) % 3,
    ];
    let t = kind % 7;
    let end = len / 4 * 4;
    for i in (0..end).step_by(4) {
        // All three source vectors are loaded before any output store.
        let [y, co, cg] = src.map(|s| unsafe { vld1q_s32(s.as_ptr().add(i)) });
        let tmp = vsubq_s32(y, shr::<1>(cg));
        let blue = vsubq_s32(tmp, shr::<1>(co));
        let rgb = [vaddq_s32(blue, co), vaddq_s32(cg, tmp), blue];
        let [first, second, third] = perm.map(|p| rgb[p]);
        let (v0, v1, v2) = if t == 6 {
            let v1 = vsubq_s32(first, third);
            let tmp = vaddq_s32(third, shr::<1>(v1));
            let v2 = vsubq_s32(second, tmp);
            (vaddq_s32(tmp, shr::<1>(v2)), v1, v2)
        } else {
            let v1 = match t >> 1 {
                1 => vsubq_s32(second, first),
                2 => vsubq_s32(second, shr::<1>(vaddq_s32(first, third))),
                _ => second,
            };
            (
                first,
                v1,
                if t & 1 == 1 {
                    vsubq_s32(third, first)
                } else {
                    third
                },
            )
        };
        unsafe {
            vst1q_s32(a.as_mut_ptr().add(i), v0);
            vst1q_s32(b.as_mut_ptr().add(i), v1);
            vst1q_s32(c.as_mut_ptr().add(i), v2);
        }
    }
    crate::lossless::rct_row_scalar(
        src.map(|s| &s[end..]),
        kind,
        [&mut a[end..], &mut b[end..], &mut c[end..]],
    );
}

#[target_feature(enable = "neon")]
pub(crate) fn palette_row_neon(above: &[i32], row: &[i32], out: &mut [[i32; 4]]) {
    assert!(above.len() >= out.len() + 3 && row.len() >= out.len() + 2);
    let end = out.len() / 4 * 4;
    for x in (0..end).step_by(4) {
        let [ll, l, v] = [0, 1, 2].map(|j| unsafe { vld1q_s32(row.as_ptr().add(x + j)) });
        let [tl, t, tr] = [1, 2, 3].map(|j| unsafe { vld1q_s32(above.as_ptr().add(x + j)) });
        // Comparisons are 0 or -1: 1 + positive - negative.
        let sign = |v| {
            vaddq_s32(
                vsubq_s32(vdupq_n_s32(1), gt(v, vdupq_n_s32(0))),
                gt(vdupq_n_s32(0), v),
            )
        };
        let ctx = vaddq_s32(
            vaddq_s32(
                vmulq_s32(sign(vsubq_s32(l, tl)), vdupq_n_s32(27)),
                vmulq_s32(sign(vsubq_s32(t, tl)), vdupq_n_s32(9)),
            ),
            vaddq_s32(
                vmulq_s32(sign(vsubq_s32(t, tr)), vdupq_n_s32(3)),
                sign(vsubq_s32(l, ll)),
            ),
        );
        // Palette indices are bounded by 1023, so this i32 expression
        // is exactly the original widened clamped gradient.
        let gradient = vminq_s32(
            vmaxq_s32(vsubq_s32(vaddq_s32(l, t), tl), vminq_s32(l, t)),
            vmaxq_s32(l, t),
        );
        let vectors = [
            ctx,
            vsubq_s32(v, l),
            vsubq_s32(v, t),
            vsubq_s32(v, gradient),
        ];
        let mut values = [[0; 4]; 4];
        for (dst, vector) in values.iter_mut().zip(vectors) {
            unsafe {
                vst1q_s32(dst.as_mut_ptr(), vector);
            }
        }
        for i in 0..4 {
            out[x + i] = std::array::from_fn(|j| values[j][i]);
        }
    }
    crate::lossless::palette_row_scalar(&above[end..], &row[end..], &mut out[end..]);
}
