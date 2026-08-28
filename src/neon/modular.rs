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

use crate::entropy::Token;
use crate::modular::{alpha_token_u8, alpha_token_u16};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn tokenize_eight_u8(
    values: uint8x8_t,
    west: uint8x8_t,
    north: uint8x8_t,
    northwest: uint8x8_t,
    out: &mut [Token],
) {
    debug_assert_eq!(out.len(), 8);
    let v = vreinterpretq_s16_u16(vmovl_u8(values));
    let w = vreinterpretq_s16_u16(vmovl_u8(west));
    let n = vreinterpretq_s16_u16(vmovl_u8(north));
    let nw = vreinterpretq_s16_u16(vmovl_u8(northwest));
    let grad = vsubq_s16(vaddq_s16(w, n), nw);
    let pred = vmaxq_s16(vminq_s16(w, n), vminq_s16(vmaxq_s16(w, n), grad));
    let residual = vsubq_s16(v, pred);
    let packed = vreinterpretq_u16_s16(veorq_s16(
        vshlq_n_s16::<1>(residual),
        vshrq_n_s16::<15>(residual),
    ));
    let contexts = vandq_u16(vcleq_s16(grad, vdupq_n_s16(0)), vdupq_n_u16(1));
    let low = uint32x4x2_t(
        vmovl_u16(vget_low_u16(contexts)),
        vmovl_u16(vget_low_u16(packed)),
    );
    let high = uint32x4x2_t(vmovl_high_u16(contexts), vmovl_high_u16(packed));
    unsafe {
        vst2q_u32(out[..4].as_mut_ptr().cast::<u32>(), low);
        vst2q_u32(out[4..].as_mut_ptr().cast::<u32>(), high);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn tokenize_alpha_u8_first_row_neon(row: &[u8], out: &mut [Token]) {
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<16>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<16>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<16>();
    let zero = vdup_n_u8(0);

    for ((values, west), out) in values.iter().zip(west.iter()).zip(out_chunks.iter_mut()) {
        let values = unsafe { vld1q_u8(values.as_ptr()) };
        let west = unsafe { vld1q_u8(west.as_ptr()) };
        tokenize_eight_u8(
            vget_low_u8(values),
            vget_low_u8(west),
            zero,
            zero,
            &mut out[..8],
        );
        tokenize_eight_u8(
            vget_high_u8(values),
            vget_high_u8(west),
            zero,
            zero,
            &mut out[8..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<8>();
    let (west_halves, west_tail) = west_tail.as_chunks::<8>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<8>();
    for ((values, west), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_eight_u8(
            unsafe { vld1_u8(values.as_ptr()) },
            unsafe { vld1_u8(west.as_ptr()) },
            zero,
            zero,
            out,
        );
    }

    for ((&value, &west), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u8(value, west, 0, 0);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn tokenize_alpha_u8_interior_neon(row: &[u8], north: &[u8], out: &mut [Token]) {
    assert_eq!(row.len(), north.len());
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<16>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<16>();
    let (north_chunks, north_tail) = north[1..].as_chunks::<16>();
    let (northwest, northwest_tail) = north[..north.len() - 1].as_chunks::<16>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<16>();

    for ((((values, west), north), northwest), out) in values
        .iter()
        .zip(west.iter())
        .zip(north_chunks.iter())
        .zip(northwest.iter())
        .zip(out_chunks.iter_mut())
    {
        let values = unsafe { vld1q_u8(values.as_ptr()) };
        let west = unsafe { vld1q_u8(west.as_ptr()) };
        let north = unsafe { vld1q_u8(north.as_ptr()) };
        let northwest = unsafe { vld1q_u8(northwest.as_ptr()) };
        tokenize_eight_u8(
            vget_low_u8(values),
            vget_low_u8(west),
            vget_low_u8(north),
            vget_low_u8(northwest),
            &mut out[..8],
        );
        tokenize_eight_u8(
            vget_high_u8(values),
            vget_high_u8(west),
            vget_high_u8(north),
            vget_high_u8(northwest),
            &mut out[8..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<8>();
    let (west_halves, west_tail) = west_tail.as_chunks::<8>();
    let (north_halves, north_tail) = north_tail.as_chunks::<8>();
    let (northwest_halves, northwest_tail) = northwest_tail.as_chunks::<8>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<8>();
    for ((((values, west), north), northwest), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(north_halves.iter())
        .zip(northwest_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_eight_u8(
            unsafe { vld1_u8(values.as_ptr()) },
            unsafe { vld1_u8(west.as_ptr()) },
            unsafe { vld1_u8(north.as_ptr()) },
            unsafe { vld1_u8(northwest.as_ptr()) },
            out,
        );
    }

    for ((((&value, &west), &north), &northwest), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(north_tail.iter())
        .zip(northwest_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u8(value, west, north, northwest);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn tokenize_four_u16(
    values: uint16x4_t,
    west: uint16x4_t,
    north: uint16x4_t,
    northwest: uint16x4_t,
    out: &mut [Token],
) {
    debug_assert_eq!(out.len(), 4);
    let v = vreinterpretq_s32_u32(vmovl_u16(values));
    let w = vreinterpretq_s32_u32(vmovl_u16(west));
    let n = vreinterpretq_s32_u32(vmovl_u16(north));
    let nw = vreinterpretq_s32_u32(vmovl_u16(northwest));
    let grad = vsubq_s32(vaddq_s32(w, n), nw);
    let pred = vmaxq_s32(vminq_s32(w, n), vminq_s32(vmaxq_s32(w, n), grad));
    let residual = vsubq_s32(v, pred);
    let packed = vreinterpretq_u32_s32(veorq_s32(
        vshlq_n_s32::<1>(residual),
        vshrq_n_s32::<31>(residual),
    ));
    let contexts = vandq_u32(vcleq_s32(grad, vdupq_n_s32(0)), vdupq_n_u32(1));
    unsafe {
        vst2q_u32(
            out.as_mut_ptr().cast::<u32>(),
            uint32x4x2_t(contexts, packed),
        );
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn tokenize_alpha_u16_first_row_neon(row: &[u16], out: &mut [Token]) {
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<8>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<8>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<8>();
    let zero = vdup_n_u16(0);

    for ((values, west), out) in values.iter().zip(west.iter()).zip(out_chunks.iter_mut()) {
        let values = unsafe { vld1q_u16(values.as_ptr()) };
        let west = unsafe { vld1q_u16(west.as_ptr()) };
        tokenize_four_u16(
            vget_low_u16(values),
            vget_low_u16(west),
            zero,
            zero,
            &mut out[..4],
        );
        tokenize_four_u16(
            vget_high_u16(values),
            vget_high_u16(west),
            zero,
            zero,
            &mut out[4..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<4>();
    let (west_halves, west_tail) = west_tail.as_chunks::<4>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<4>();
    for ((values, west), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_four_u16(
            unsafe { vld1_u16(values.as_ptr()) },
            unsafe { vld1_u16(west.as_ptr()) },
            zero,
            zero,
            out,
        );
    }

    for ((&value, &west), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u16(value, west, 0, 0);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn tokenize_alpha_u16_interior_neon(row: &[u16], north: &[u16], out: &mut [Token]) {
    assert_eq!(row.len(), north.len());
    assert_eq!(row.len(), out.len());
    debug_assert_eq!(size_of::<Token>(), 2 * size_of::<u32>());

    let (values, value_tail) = row[1..].as_chunks::<8>();
    let (west, west_tail) = row[..row.len() - 1].as_chunks::<8>();
    let (north_chunks, north_tail) = north[1..].as_chunks::<8>();
    let (northwest, northwest_tail) = north[..north.len() - 1].as_chunks::<8>();
    let (out_chunks, out_tail) = out[1..].as_chunks_mut::<8>();

    for ((((values, west), north), northwest), out) in values
        .iter()
        .zip(west.iter())
        .zip(north_chunks.iter())
        .zip(northwest.iter())
        .zip(out_chunks.iter_mut())
    {
        let values = unsafe { vld1q_u16(values.as_ptr()) };
        let west = unsafe { vld1q_u16(west.as_ptr()) };
        let north = unsafe { vld1q_u16(north.as_ptr()) };
        let northwest = unsafe { vld1q_u16(northwest.as_ptr()) };
        tokenize_four_u16(
            vget_low_u16(values),
            vget_low_u16(west),
            vget_low_u16(north),
            vget_low_u16(northwest),
            &mut out[..4],
        );
        tokenize_four_u16(
            vget_high_u16(values),
            vget_high_u16(west),
            vget_high_u16(north),
            vget_high_u16(northwest),
            &mut out[4..],
        );
    }

    let (value_halves, value_tail) = value_tail.as_chunks::<4>();
    let (west_halves, west_tail) = west_tail.as_chunks::<4>();
    let (north_halves, north_tail) = north_tail.as_chunks::<4>();
    let (northwest_halves, northwest_tail) = northwest_tail.as_chunks::<4>();
    let (out_halves, out_tail) = out_tail.as_chunks_mut::<4>();
    for ((((values, west), north), northwest), out) in value_halves
        .iter()
        .zip(west_halves.iter())
        .zip(north_halves.iter())
        .zip(northwest_halves.iter())
        .zip(out_halves.iter_mut())
    {
        tokenize_four_u16(
            unsafe { vld1_u16(values.as_ptr()) },
            unsafe { vld1_u16(west.as_ptr()) },
            unsafe { vld1_u16(north.as_ptr()) },
            unsafe { vld1_u16(northwest.as_ptr()) },
            out,
        );
    }

    for ((((&value, &west), &north), &northwest), out) in value_tail
        .iter()
        .zip(west_tail.iter())
        .zip(north_tail.iter())
        .zip(northwest_tail.iter())
        .zip(out_tail.iter_mut())
    {
        *out = alpha_token_u16(value, west, north, northwest);
    }
}
