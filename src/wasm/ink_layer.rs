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

use std::arch::wasm32::*;

#[target_feature(enable = "simd128")]
unsafe fn iload(p: *const i32) -> v128 {
    unsafe { v128_load(p.cast()) }
}

#[target_feature(enable = "simd128")]
unsafe fn istore(p: *mut i32, v: v128) {
    unsafe { v128_store(p.cast(), v) }
}

#[target_feature(enable = "simd128")]
fn any(v: v128) -> bool {
    v128_any_true(v)
}

#[target_feature(enable = "simd128")]
pub(crate) fn luma_row_wasm(rgb: &[[u8; 3]], out: &mut [u8]) {
    assert!(rgb.len() >= out.len());
    let masks: [[v128; 3]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|part| {
            let indices: [u8; 16] = std::array::from_fn(|i| {
                if (3 * i + c) / 16 == part {
                    ((3 * i + c) % 16) as u8
                } else {
                    128
                }
            });
            unsafe { v128_load(indices.as_ptr().cast()) }
        })
    });
    let end = out.len() / 16 * 16;
    for x in (0..end).step_by(16) {
        let src: [v128; 3] = std::array::from_fn(|part| unsafe {
            v128_load(rgb.as_ptr().add(x).cast::<u8>().add(part * 16).cast())
        });
        let channels: [v128; 3] = std::array::from_fn(|c| {
            v128_or(
                v128_or(
                    i8x16_swizzle(src[0], masks[c][0]),
                    i8x16_swizzle(src[1], masks[c][1]),
                ),
                i8x16_swizzle(src[2], masks[c][2]),
            )
        });
        let half = |hi: bool| {
            let [r, g, b] = channels.map(|v| {
                if hi {
                    u16x8_extend_high_u8x16(v)
                } else {
                    u16x8_extend_low_u8x16(v)
                }
            });
            let y = i16x8_add(
                i16x8_add(
                    i16x8_mul(r, i16x8_splat(77)),
                    i16x8_mul(g, i16x8_splat(150)),
                ),
                i16x8_add(i16x8_mul(b, i16x8_splat(29)), i16x8_splat(128)),
            );
            u16x8_shr(y, 8)
        };
        let y = u8x16_narrow_i16x8(half(false), half(true));
        unsafe {
            v128_store(out.as_mut_ptr().add(x).cast(), y);
        }
    }
    crate::ink_layer::luma_row_scalar(&rgb[end..], &mut out[end..]);
}

#[target_feature(enable = "simd128")]
pub(crate) fn ridge_row_wasm(
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
            unsafe { iload(v.as_ptr()) }
        };
        let mut yes = i32x4_splat(0);
        let p = gather(0, None);
        for (a, b) in [(1, 2), (3, 4)] {
            let da = i32x4_sub(p, gather(a, None));
            let db = i32x4_sub(p, gather(b, None));
            yes = v128_or(
                yes,
                v128_or(
                    v128_and(
                        i32x4_gt(da, i32x4_splat(crate::ink_layer::THIN_DEV)),
                        i32x4_gt(db, i32x4_splat(crate::ink_layer::THIN_DEV)),
                    ),
                    v128_and(
                        i32x4_gt(i32x4_splat(-crate::ink_layer::THIN_DEV), da),
                        i32x4_gt(i32x4_splat(-crate::ink_layer::THIN_DEV), db),
                    ),
                ),
            );
        }
        // Preserve the scalar short circuit at vector granularity:
        // luma ridges and flat backgrounds need no further color work.
        if any(i32x4_sub(i32x4_splat(-1), yes)) {
            let center: [_; 3] = std::array::from_fn(|c| gather(0, Some(c)));
            let chroma = |d: [_; 3]| {
                i32x4_max(
                    i32x4_abs(i32x4_sub(d[0], d[1])),
                    i32x4_abs(i32x4_sub(d[2], d[1])),
                )
            };
            for (a, b) in [(1, 2), (3, 4)] {
                let ac: [_; 3] = std::array::from_fn(|c| i32x4_sub(center[c], gather(a, Some(c))));
                let possible = v128_and(
                    i32x4_gt(chroma(ac), i32x4_splat(crate::ink_layer::THIN_DEV)),
                    i32x4_sub(i32x4_splat(-1), yes),
                );
                if any(possible) {
                    let bc: [_; 3] =
                        std::array::from_fn(|c| i32x4_sub(center[c], gather(b, Some(c))));
                    let mut background = i32x4_splat(-1);
                    for c in 0..3 {
                        background = v128_and(
                            background,
                            i32x4_gt(i32x4_splat(16), i32x4_abs(i32x4_sub(bc[c], ac[c]))),
                        );
                    }
                    yes = v128_or(
                        yes,
                        v128_and(
                            possible,
                            v128_and(
                                background,
                                i32x4_gt(chroma(bc), i32x4_splat(crate::ink_layer::THIN_DEV)),
                            ),
                        ),
                    );
                }
            }
        }
        let mut values = [0; 4];
        unsafe {
            istore(values.as_mut_ptr(), yes);
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
