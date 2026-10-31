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

use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn iload(p: *const i32) -> __m256i {
    unsafe { _mm256_loadu_si256(p.cast()) }
}

#[target_feature(enable = "avx2")]
unsafe fn istore(p: *mut i32, v: __m256i) {
    unsafe { _mm256_storeu_si256(p.cast(), v) }
}

#[target_feature(enable = "avx2")]
fn any(v: __m256i) -> bool {
    _mm256_movemask_epi8(v) != 0
}

#[target_feature(enable = "avx2")]
pub(crate) fn luma_row_avx2(rgb: &[[u8; 3]], out: &mut [u8]) {
    assert!(rgb.len() >= out.len());
    let masks: [[__m128i; 3]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|part| {
            let indices: [i8; 16] = std::array::from_fn(|i| {
                if (3 * i + c) / 16 == part {
                    ((3 * i + c) % 16) as i8
                } else {
                    -128
                }
            });
            unsafe { _mm_loadu_si128(indices.as_ptr().cast()) }
        })
    });
    let end = out.len() / 16 * 16;
    for x in (0..end).step_by(16) {
        let src: [__m128i; 3] = std::array::from_fn(|part| unsafe {
            _mm_loadu_si128(rgb.as_ptr().add(x).cast::<u8>().add(part * 16).cast())
        });
        let channels: [__m128i; 3] = std::array::from_fn(|c| {
            _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(src[0], masks[c][0]),
                    _mm_shuffle_epi8(src[1], masks[c][1]),
                ),
                _mm_shuffle_epi8(src[2], masks[c][2]),
            )
        });
        let [r, g, b] = channels.map(|v| _mm256_cvtepu8_epi16(v));
        let y = _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_mullo_epi16(r, _mm256_set1_epi16(77)),
                _mm256_mullo_epi16(g, _mm256_set1_epi16(150)),
            ),
            _mm256_add_epi16(
                _mm256_mullo_epi16(b, _mm256_set1_epi16(29)),
                _mm256_set1_epi16(128),
            ),
        );
        let y = _mm256_srli_epi16::<8>(y);
        let y = _mm_packus_epi16(_mm256_castsi256_si128(y), _mm256_extracti128_si256::<1>(y));

        unsafe {
            _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y);
        }
    }
    crate::ink_layer::luma_row_scalar(&rgb[end..], &mut out[end..]);
}

#[target_feature(enable = "avx2")]
pub(crate) fn ridge_row_avx2(
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
    while x + 2 * (8 - 1) < w {
        let positions: [[usize; 5]; 8] =
            std::array::from_fn(|i| crate::ink_layer::ridge_positions(x + 2 * i, y, w, h));
        let gather = |p: usize, c: Option<usize>| {
            let v: [i32; 8] = std::array::from_fn(|i| {
                let j = positions[i][p];
                i32::from(c.map_or(luma[j], |c| rgb[j][c]))
            });
            unsafe { iload(v.as_ptr()) }
        };
        let mut yes = _mm256_set1_epi32(0);
        let p = gather(0, None);
        for (a, b) in [(1, 2), (3, 4)] {
            let da = _mm256_sub_epi32(p, gather(a, None));
            let db = _mm256_sub_epi32(p, gather(b, None));
            yes = _mm256_or_si256(
                yes,
                _mm256_or_si256(
                    _mm256_and_si256(
                        _mm256_cmpgt_epi32(da, _mm256_set1_epi32(crate::ink_layer::THIN_DEV)),
                        _mm256_cmpgt_epi32(db, _mm256_set1_epi32(crate::ink_layer::THIN_DEV)),
                    ),
                    _mm256_and_si256(
                        _mm256_cmpgt_epi32(_mm256_set1_epi32(-crate::ink_layer::THIN_DEV), da),
                        _mm256_cmpgt_epi32(_mm256_set1_epi32(-crate::ink_layer::THIN_DEV), db),
                    ),
                ),
            );
        }
        // Preserve the scalar short circuit at vector granularity:
        // luma ridges and flat backgrounds need no further color work.
        if any(_mm256_sub_epi32(_mm256_set1_epi32(-1), yes)) {
            let center: [_; 3] = std::array::from_fn(|c| gather(0, Some(c)));
            let chroma = |d: [_; 3]| {
                _mm256_max_epi32(
                    _mm256_abs_epi32(_mm256_sub_epi32(d[0], d[1])),
                    _mm256_abs_epi32(_mm256_sub_epi32(d[2], d[1])),
                )
            };
            for (a, b) in [(1, 2), (3, 4)] {
                let ac: [_; 3] =
                    std::array::from_fn(|c| _mm256_sub_epi32(center[c], gather(a, Some(c))));
                let possible = _mm256_and_si256(
                    _mm256_cmpgt_epi32(chroma(ac), _mm256_set1_epi32(crate::ink_layer::THIN_DEV)),
                    _mm256_sub_epi32(_mm256_set1_epi32(-1), yes),
                );
                if any(possible) {
                    let bc: [_; 3] =
                        std::array::from_fn(|c| _mm256_sub_epi32(center[c], gather(b, Some(c))));
                    let mut background = _mm256_set1_epi32(-1);
                    for c in 0..3 {
                        background = _mm256_and_si256(
                            background,
                            _mm256_cmpgt_epi32(
                                _mm256_set1_epi32(16),
                                _mm256_abs_epi32(_mm256_sub_epi32(bc[c], ac[c])),
                            ),
                        );
                    }
                    yes = _mm256_or_si256(
                        yes,
                        _mm256_and_si256(
                            possible,
                            _mm256_and_si256(
                                background,
                                _mm256_cmpgt_epi32(
                                    chroma(bc),
                                    _mm256_set1_epi32(crate::ink_layer::THIN_DEV),
                                ),
                            ),
                        ),
                    );
                }
            }
        }
        let mut values = [0; 8];
        unsafe {
            istore(values.as_mut_ptr(), yes);
        }
        for i in 0..8 {
            out[x + 2 * i] = u8::from(values[i] != 0);
            count += usize::from(out[x + 2 * i]);
        }
        x += 2 * 8;
    }
    for x in (x..w).step_by(2) {
        out[x] =
            crate::ink_layer::ridge_pixel(rgb, luma, crate::ink_layer::ridge_positions(x, y, w, h));
        count += usize::from(out[x]);
    }
    count
}
