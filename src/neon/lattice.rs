/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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
//! Autocorrelation dot products of one line at adjacent lags.

use std::arch::aarch64::*;

const LAGS: usize = 8;

/// Dot products of `line` with itself shifted by `k..k + 8`.
#[inline]
#[target_feature(enable = "neon")]
fn group(line: &[f32], k: usize) -> [f32; LAGS] {
    let len = line.len();
    // Head samples every lag of the group can pair.
    let common = len - (k + LAGS - 1);
    let head = line.as_ptr();
    let tail = line[k..].as_ptr();
    let mut acc0 = [vdupq_n_f32(0.0); LAGS];
    let mut acc1 = [vdupq_n_f32(0.0); LAGS];
    let mut i = 0;
    // SAFETY: the furthest load ends at `k + i + 7 + LAGS - 1 + 1`, which is
    // at most `len` while `i + 8 <= common`; likewise for the 4-wide step.
    unsafe {
        while i + 8 <= common {
            let a0 = vld1q_f32(head.add(i));
            let a1 = vld1q_f32(head.add(i + 4));
            for j in 0..LAGS {
                acc0[j] = vfmaq_f32(acc0[j], a0, vld1q_f32(tail.add(i + j)));
                acc1[j] = vfmaq_f32(acc1[j], a1, vld1q_f32(tail.add(i + 4 + j)));
            }
            i += 8;
        }
        if i + 4 <= common {
            let a0 = vld1q_f32(head.add(i));
            for j in 0..LAGS {
                acc0[j] = vfmaq_f32(acc0[j], a0, vld1q_f32(tail.add(i + j)));
            }
            i += 4;
        }
    }
    let mut out = [0.0f32; LAGS];
    for (j, out) in out.iter_mut().enumerate() {
        let lag = k + j;
        let mut d = vaddvq_f32(vaddq_f32(acc0[j], acc1[j]));
        for (&a, &b) in line[i..len - lag].iter().zip(&line[i + lag..]) {
            d = a.mul_add(b, d);
        }
        *out = d;
    }
    out
}

/// `dots[k] = sum(line[i] * line[i + k])`; `dots.len() <= line.len() / 2 + 1`.
#[target_feature(enable = "neon")]
pub(crate) fn lag_dots_neon(line: &[f32], dots: &mut [f32]) {
    let count = dots.len();
    if count < LAGS {
        crate::lattice_patches::lag_dots_scalar(line, dots);
        return;
    }
    let (groups, rest) = dots.as_chunks_mut::<LAGS>();
    for (k, out) in (0..).step_by(LAGS).zip(groups) {
        *out = group(line, k);
    }
    if !rest.is_empty() {
        // The last, overlapping group covers the leftover lags.
        let last = group(line, count - LAGS);
        rest.copy_from_slice(&last[LAGS - rest.len()..]);
    }
}
