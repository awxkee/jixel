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
use crate::haar::{CHROMA_DC_SQUEEZE_DEADZONE, CHROMA_DC_SQUEEZE_STEP};
use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn round_ties_away(x: v128) -> v128 {
    let sign = v128_and(x, f32x4_splat(-0.0));
    let truncated = f32x4_trunc(x);
    let adjust = f32x4_ge(f32x4_abs(f32x4_sub(x, truncated)), f32x4_splat(0.5));
    let signed_one = v128_or(f32x4_splat(1.0), sign);
    v128_or(f32x4_add(truncated, v128_and(signed_one, adjust)), sign)
}

#[inline]
#[target_feature(enable = "simd128")]
fn dzq(x: v128, q: v128, offset: v128) -> v128 {
    let magnitude = f32x4_mul(
        f32x4_floor(f32x4_add(f32x4_div(f32x4_abs(x), q), offset)),
        q,
    );
    v128_bitselect(x, magnitude, f32x4_splat(-0.0))
}

#[target_feature(enable = "simd128")]
pub(crate) fn haar_rows_wasm(
    top: &[f32],
    bottom: &[f32],
    q_hf: f32,
    top_out: &mut [f32],
    bottom_out: &mut [f32],
) -> usize {
    let quarter = f32x4_splat(0.25);
    let low_scale = f32x4_splat(0.25 / CHROMA_DC_SQUEEZE_STEP);
    let step = f32x4_splat(CHROMA_DC_SQUEEZE_STEP);
    let q = f32x4_splat(q_hf);
    let offset = f32x4_splat(1.0 - CHROMA_DC_SQUEEZE_DEADZONE);
    let mut done = 0;
    for ((top, bottom), (top_out, bottom_out)) in top
        .as_chunks::<8>()
        .0
        .iter()
        .zip(bottom.as_chunks::<8>().0)
        .zip(
            top_out
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(bottom_out.as_chunks_mut::<8>().0),
        )
    {
        let (top0, top1, bottom0, bottom1) = unsafe {
            (
                v128_load(top.as_ptr().cast()),
                v128_load(top.as_ptr().add(4).cast()),
                v128_load(bottom.as_ptr().cast()),
                v128_load(bottom.as_ptr().add(4).cast()),
            )
        };
        let a = i32x4_shuffle::<0, 2, 4, 6>(top0, top1);
        let b = i32x4_shuffle::<1, 3, 5, 7>(top0, top1);
        let c = i32x4_shuffle::<0, 2, 4, 6>(bottom0, bottom1);
        let d = i32x4_shuffle::<1, 3, 5, 7>(bottom0, bottom1);
        let ll = f32x4_mul(
            round_ties_away(f32x4_mul(
                f32x4_add(f32x4_add(f32x4_add(a, b), c), d),
                low_scale,
            )),
            step,
        );
        let hl = dzq(
            f32x4_mul(f32x4_sub(f32x4_add(f32x4_sub(a, b), c), d), quarter),
            q,
            offset,
        );
        let lh = dzq(
            f32x4_mul(f32x4_sub(f32x4_sub(f32x4_add(a, b), c), d), quarter),
            q,
            offset,
        );
        let hh = dzq(
            f32x4_mul(f32x4_add(f32x4_sub(f32x4_sub(a, b), c), d), quarter),
            q,
            offset,
        );
        let a_out = f32x4_add(f32x4_add(f32x4_add(ll, hl), lh), hh);
        let b_out = f32x4_sub(f32x4_add(f32x4_sub(ll, hl), lh), hh);
        let c_out = f32x4_sub(f32x4_sub(f32x4_add(ll, hl), lh), hh);
        let d_out = f32x4_add(f32x4_sub(f32x4_sub(ll, hl), lh), hh);
        unsafe {
            v128_store(
                top_out.as_mut_ptr().cast(),
                i32x4_shuffle::<0, 4, 1, 5>(a_out, b_out),
            );
            v128_store(
                top_out.as_mut_ptr().add(4).cast(),
                i32x4_shuffle::<2, 6, 3, 7>(a_out, b_out),
            );
            v128_store(
                bottom_out.as_mut_ptr().cast(),
                i32x4_shuffle::<0, 4, 1, 5>(c_out, d_out),
            );
            v128_store(
                bottom_out.as_mut_ptr().add(4).cast(),
                i32x4_shuffle::<2, 6, 3, 7>(c_out, d_out),
            );
        }
        done += 8;
    }
    done
}
