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
fn any(v: int32x4_t) -> bool {
    vmaxvq_u32(vreinterpretq_u32_s32(v)) != 0
}

#[target_feature(enable = "neon")]
fn gt(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vreinterpretq_s32_u32(vcgtq_s32(a, b))
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_row_neon(rgb: &[[u8; 3]], out: &mut [u8]) {
    assert!(rgb.len() >= out.len());
    let end = out.len() / 16 * 16;
    for x in (0..end).step_by(16) {
        let uint8x16x3_t(r, g, b) = unsafe { vld3q_u8(rgb.as_ptr().add(x).cast()) };
        let part = |r, g, b| {
            let y = vmlal_u8(
                vmlal_u8(vmull_u8(r, vdup_n_u8(77)), g, vdup_n_u8(150)),
                b,
                vdup_n_u8(29),
            );
            vshrn_n_u16::<8>(vaddq_u16(y, vdupq_n_u16(128)))
        };
        let lo = part(vget_low_u8(r), vget_low_u8(g), vget_low_u8(b));
        let hi = part(vget_high_u8(r), vget_high_u8(g), vget_high_u8(b));
        unsafe {
            vst1q_u8(out.as_mut_ptr().add(x), vcombine_u8(lo, hi));
        }
    }
    crate::ink_layer::luma_row_scalar(&rgb[end..], &mut out[end..]);
}

#[target_feature(enable = "neon")]
pub(crate) fn ridge_row_neon(
    rgb: &[[u8; 3]],
    luma: &[u8],
    w: usize,
    h: usize,
    y: usize,
    out: &mut [u8],
) -> usize {
    assert!(rgb.len() >= w * h && luma.len() >= w * h && out.len() >= w && y < h);
    let mut x = y & 1;
    let mut count = 0;
    while x + 2 * (4 - 1) < w {
        let positions: [[usize; 5]; 4] =
            std::array::from_fn(|i| crate::ink_layer::ridge_positions(x + 2 * i, y, w, h));
        let gather = |p: usize, c: Option<usize>| {
            let v: [i32; 4] = std::array::from_fn(|i| {
                let j = positions[i][p];
                i32::from(c.map_or(luma[j], |c| rgb[j][c]))
            });
            unsafe { vld1q_s32(v.as_ptr()) }
        };
        let mut yes = vdupq_n_s32(0);
        let p = gather(0, None);
        for (a, b) in [(1, 2), (3, 4)] {
            let da = vsubq_s32(p, gather(a, None));
            let db = vsubq_s32(p, gather(b, None));
            yes = vorrq_s32(
                yes,
                vorrq_s32(
                    vandq_s32(
                        gt(da, vdupq_n_s32(crate::ink_layer::THIN_DEV)),
                        gt(db, vdupq_n_s32(crate::ink_layer::THIN_DEV)),
                    ),
                    vandq_s32(
                        gt(vdupq_n_s32(-crate::ink_layer::THIN_DEV), da),
                        gt(vdupq_n_s32(-crate::ink_layer::THIN_DEV), db),
                    ),
                ),
            );
        }
        // Preserve the scalar short circuit at vector granularity:
        // luma ridges and flat backgrounds need no further color work.
        if any(vsubq_s32(vdupq_n_s32(-1), yes)) {
            let center: [_; 3] = std::array::from_fn(|c| gather(0, Some(c)));
            let chroma = |d: [_; 3]| {
                vmaxq_s32(
                    vabsq_s32(vsubq_s32(d[0], d[1])),
                    vabsq_s32(vsubq_s32(d[2], d[1])),
                )
            };
            for (a, b) in [(1, 2), (3, 4)] {
                let ac: [_; 3] = std::array::from_fn(|c| vsubq_s32(center[c], gather(a, Some(c))));
                let possible = vandq_s32(
                    gt(chroma(ac), vdupq_n_s32(crate::ink_layer::THIN_DEV)),
                    vsubq_s32(vdupq_n_s32(-1), yes),
                );
                if any(possible) {
                    let bc: [_; 3] =
                        std::array::from_fn(|c| vsubq_s32(center[c], gather(b, Some(c))));
                    let mut background = vdupq_n_s32(-1);
                    for c in 0..3 {
                        background = vandq_s32(
                            background,
                            gt(vdupq_n_s32(16), vabsq_s32(vsubq_s32(bc[c], ac[c]))),
                        );
                    }
                    yes = vorrq_s32(
                        yes,
                        vandq_s32(
                            possible,
                            vandq_s32(
                                background,
                                gt(chroma(bc), vdupq_n_s32(crate::ink_layer::THIN_DEV)),
                            ),
                        ),
                    );
                }
            }
        }
        let mut values = [0; 4];
        unsafe {
            vst1q_s32(values.as_mut_ptr(), yes);
        }
        for i in 0..4 {
            out[x + 2 * i] = u8::from(values[i] != 0);
            count += usize::from(out[x + 2 * i]);
        }
        x += 2 * 4;
    }
    for x in (x..w).step_by(2) {
        out[x] =
            crate::ink_layer::ridge_pixel(rgb, luma, crate::ink_layer::ridge_positions(x, y, w, h));
        count += usize::from(out[x]);
    }
    count
}
