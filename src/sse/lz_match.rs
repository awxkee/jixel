/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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
use crate::entropy::{CompactToken, Token};
use crate::lz_match::{match_len_compact_scalar, match_len_scalar};
use std::arch::x86_64::*;

// Token is repr(C), two u32 fields, without padding. Every vector load below
// stays within both source slices. The sources may overlap: they are read-only.
#[target_feature(enable = "sse2")]
pub(crate) fn match_len_sse2(a: &[Token], b: &[Token]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    unsafe {
        while i + 2 <= n {
            let x = _mm_loadu_si128(a.as_ptr().add(i).cast());
            let y = _mm_loadu_si128(b.as_ptr().add(i).cast());
            let mask = _mm_movemask_epi8(_mm_cmpeq_epi32(x, y)) as u32;
            if mask != 0xffff {
                return i + ((!mask).trailing_zeros() as usize / 8);
            }
            i += 2;
        }
    }
    i + match_len_scalar(&a[i..n], &b[i..n])
}

// CompactToken is one u32 without padding. Every vector load below
// stays within both source slices. The sources may overlap: they are read-only.
#[target_feature(enable = "sse2")]
pub(crate) fn match_len_compact_sse2(a: &[CompactToken], b: &[CompactToken]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    unsafe {
        while i + 4 <= n {
            let x = _mm_loadu_si128(a.as_ptr().add(i).cast());
            let y = _mm_loadu_si128(b.as_ptr().add(i).cast());
            let mask = _mm_movemask_epi8(_mm_cmpeq_epi32(x, y)) as u32;
            if mask != 0xffff {
                return i + ((!mask).trailing_zeros() as usize / 4);
            }
            i += 4;
        }
    }
    i + match_len_compact_scalar(&a[i..n], &b[i..n])
}
