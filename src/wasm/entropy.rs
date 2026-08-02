/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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

use crate::entropy::ALPHABET_SIZE;

#[inline]
#[target_feature(enable = "simd128")]
fn dirty_log2f_x4(d: v128) -> v128 {
    let one = f32x4_splat(1.0);
    let mut ix = d;
    ix = i32x4_add(ix, i32x4_splat((0x3f80_0000u32 - 0x3f35_04f3u32) as i32));
    let n = i32x4_sub(u32x4_shr(ix, 23), i32x4_splat(0x7f));
    ix = i32x4_add(
        v128_and(ix, i32x4_splat(0x007f_ffff)),
        i32x4_splat(0x3f35_04f3),
    );

    let a = ix;
    // WebAssembly defines vector division but not a reciprocal-estimate with a
    // portable precision contract, so retain division in the range reduction.
    let x = f32x4_div(f32x4_sub(a, one), f32x4_add(a, one));
    let x2 = f32x4_mul(x, x);
    let mut u = f32x4_splat(0.412_198_57);
    u = f32x4_add(f32x4_mul(u, x2), f32x4_splat(0.577_078_04));
    u = f32x4_add(f32x4_mul(u, x2), f32x4_splat(0.961_796_7));
    let base = f32x4_add(
        f32x4_mul(x, f32x4_splat(2.885_390_1)),
        f32x4_convert_i32x4(n),
    );
    f32x4_add(f32x4_mul(f32x4_mul(x2, x), u), base)
}

#[inline]
#[target_feature(enable = "simd128")]
fn horizontal_sum_x4(value: v128) -> f32 {
    f32x4_extract_lane::<0>(value)
        + f32x4_extract_lane::<1>(value)
        + f32x4_extract_lane::<2>(value)
        + f32x4_extract_lane::<3>(value)
}

/// Shannon population cost used by entropy histogram clustering.
///
/// # Safety
/// The caller must ensure SIMD128 is available.
#[target_feature(enable = "simd128")]
pub(crate) fn counts_bit_cost_wasm(counts: &[u32; ALPHABET_SIZE], total_count: u32) -> f32 {
    debug_assert_ne!(total_count, 0);
    let log_total = f32x4_splat(crate::adaptive_quant::dirty_log2f(total_count as f32));
    let one = f32x4_splat(1.0);
    let mut cost0 = f32x4_splat(0.0);
    let mut cost1 = f32x4_splat(0.0);
    for counts8 in counts.as_chunks::<8>().0 {
        let count0 = f32x4_convert_u32x4(unsafe { v128_load(counts8.as_ptr().cast()) });
        let count1 = f32x4_convert_u32x4(unsafe { v128_load(counts8.as_ptr().add(4).cast()) });
        let positive0 = f32x4_max(count0, one);
        let positive1 = f32x4_max(count1, one);
        cost0 = f32x4_add(
            cost0,
            f32x4_mul(count0, f32x4_sub(log_total, dirty_log2f_x4(positive0))),
        );
        cost1 = f32x4_add(
            cost1,
            f32x4_mul(count1, f32x4_sub(log_total, dirty_log2f_x4(positive1))),
        );
    }
    horizontal_sum_x4(f32x4_add(cost0, cost1)).max(0.0)
}
