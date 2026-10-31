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
fn shr<const N: i32>(v: __m256i) -> __m256i {
    _mm256_srai_epi32::<N>(v)
}

#[target_feature(enable = "avx2")]
pub(crate) fn rct_row_avx2(src: [&[i32]; 3], kind: u32, out: [&mut [i32]; 3]) {
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
    let end = len / 8 * 8;
    for i in (0..end).step_by(8) {
        // All three source vectors are loaded before any output store.
        let [y, co, cg] = src.map(|s| unsafe { iload(s.as_ptr().add(i)) });
        let tmp = _mm256_sub_epi32(y, shr::<1>(cg));
        let blue = _mm256_sub_epi32(tmp, shr::<1>(co));
        let rgb = [_mm256_add_epi32(blue, co), _mm256_add_epi32(cg, tmp), blue];
        let [first, second, third] = perm.map(|p| rgb[p]);
        let (v0, v1, v2) = if t == 6 {
            let v1 = _mm256_sub_epi32(first, third);
            let tmp = _mm256_add_epi32(third, shr::<1>(v1));
            let v2 = _mm256_sub_epi32(second, tmp);
            (_mm256_add_epi32(tmp, shr::<1>(v2)), v1, v2)
        } else {
            let v1 = match t >> 1 {
                1 => _mm256_sub_epi32(second, first),
                2 => _mm256_sub_epi32(second, shr::<1>(_mm256_add_epi32(first, third))),
                _ => second,
            };
            (
                first,
                v1,
                if t & 1 == 1 {
                    _mm256_sub_epi32(third, first)
                } else {
                    third
                },
            )
        };
        unsafe {
            istore(a.as_mut_ptr().add(i), v0);
            istore(b.as_mut_ptr().add(i), v1);
            istore(c.as_mut_ptr().add(i), v2);
        }
    }
    crate::lossless::rct_row_scalar(
        src.map(|s| &s[end..]),
        kind,
        [&mut a[end..], &mut b[end..], &mut c[end..]],
    );
}

#[target_feature(enable = "avx2")]
pub(crate) fn palette_row_avx2(above: &[i32], row: &[i32], out: &mut [[i32; 4]]) {
    assert!(above.len() >= out.len() + 3 && row.len() >= out.len() + 2);
    let end = out.len() / 8 * 8;
    for x in (0..end).step_by(8) {
        let [ll, l, v] = [0, 1, 2].map(|j| unsafe { iload(row.as_ptr().add(x + j)) });
        let [tl, t, tr] = [1, 2, 3].map(|j| unsafe { iload(above.as_ptr().add(x + j)) });
        // Comparisons are 0 or -1: 1 + positive - negative.
        let sign = |v| {
            _mm256_add_epi32(
                _mm256_sub_epi32(
                    _mm256_set1_epi32(1),
                    _mm256_cmpgt_epi32(v, _mm256_set1_epi32(0)),
                ),
                _mm256_cmpgt_epi32(_mm256_set1_epi32(0), v),
            )
        };
        let ctx = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(sign(_mm256_sub_epi32(l, tl)), _mm256_set1_epi32(27)),
                _mm256_mullo_epi32(sign(_mm256_sub_epi32(t, tl)), _mm256_set1_epi32(9)),
            ),
            _mm256_add_epi32(
                _mm256_mullo_epi32(sign(_mm256_sub_epi32(t, tr)), _mm256_set1_epi32(3)),
                sign(_mm256_sub_epi32(l, ll)),
            ),
        );
        // Palette indices are bounded by 1023, so this i32 expression
        // is exactly the original widened clamped gradient.
        let gradient = _mm256_min_epi32(
            _mm256_max_epi32(
                _mm256_sub_epi32(_mm256_add_epi32(l, t), tl),
                _mm256_min_epi32(l, t),
            ),
            _mm256_max_epi32(l, t),
        );
        let vectors = [
            ctx,
            _mm256_sub_epi32(v, l),
            _mm256_sub_epi32(v, t),
            _mm256_sub_epi32(v, gradient),
        ];
        let mut values = [[0; 8]; 4];
        for (dst, vector) in values.iter_mut().zip(vectors) {
            unsafe {
                istore(dst.as_mut_ptr(), vector);
            }
        }
        for i in 0..8 {
            out[x + i] = std::array::from_fn(|j| values[j][i]);
        }
    }
    crate::lossless::palette_row_scalar(&above[end..], &row[end..], &mut out[end..]);
}
