/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
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

use core::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
fn fill_ytob_4(b: v128, y: v128, slope: v128) -> v128 {
    let adjusted = f32x4_mul(slope, f32x4_convert_i32x4(y));
    let adjusted = i32x4_trunc_sat_f32x4(f32x4_nearest(adjusted));
    i32x4_sub(b, adjusted)
}

#[inline]
#[target_feature(enable = "simd128")]
fn grad_residual_4(current: v128, north: v128, west: v128, northwest: v128) -> v128 {
    let prediction = i32x4_sub(i32x4_add(north, west), northwest);
    let prediction = i32x4_max(
        i32x4_min(north, west),
        i32x4_min(i32x4_max(north, west), prediction),
    );
    i32x4_sub(current, prediction)
}

#[inline]
fn grad_residual_scalar(current: i16, north: i16, west: i16, northwest: i16) -> i32 {
    let (current, north, west, northwest) =
        (current as i32, north as i32, west as i32, northwest as i32);
    current - (north + west - northwest).clamp(north.min(west), north.max(west))
}

#[inline]
#[target_feature(enable = "simd128")]
fn fill_grad_residuals_16(
    dst: &mut [i32; 16],
    current: &[i16; 16],
    north: &[i16; 16],
    west: &[i16; 16],
    northwest: &[i16; 16],
) {
    let current0 = unsafe { v128_load(current.as_ptr().cast()) };
    let current1 = unsafe { v128_load(current[8..].as_ptr().cast()) };
    let north0 = unsafe { v128_load(north.as_ptr().cast()) };
    let north1 = unsafe { v128_load(north[8..].as_ptr().cast()) };
    let west0 = unsafe { v128_load(west.as_ptr().cast()) };
    let west1 = unsafe { v128_load(west[8..].as_ptr().cast()) };
    let northwest0 = unsafe { v128_load(northwest.as_ptr().cast()) };
    let northwest1 = unsafe { v128_load(northwest[8..].as_ptr().cast()) };
    let result0 = grad_residual_4(
        i32x4_extend_low_i16x8(current0),
        i32x4_extend_low_i16x8(north0),
        i32x4_extend_low_i16x8(west0),
        i32x4_extend_low_i16x8(northwest0),
    );
    let result1 = grad_residual_4(
        i32x4_extend_high_i16x8(current0),
        i32x4_extend_high_i16x8(north0),
        i32x4_extend_high_i16x8(west0),
        i32x4_extend_high_i16x8(northwest0),
    );
    let result2 = grad_residual_4(
        i32x4_extend_low_i16x8(current1),
        i32x4_extend_low_i16x8(north1),
        i32x4_extend_low_i16x8(west1),
        i32x4_extend_low_i16x8(northwest1),
    );
    let result3 = grad_residual_4(
        i32x4_extend_high_i16x8(current1),
        i32x4_extend_high_i16x8(north1),
        i32x4_extend_high_i16x8(west1),
        i32x4_extend_high_i16x8(northwest1),
    );
    unsafe {
        v128_store(dst.as_mut_ptr().cast(), result0);
        v128_store(dst[4..].as_mut_ptr().cast(), result1);
        v128_store(dst[8..].as_mut_ptr().cast(), result2);
        v128_store(dst[12..].as_mut_ptr().cast(), result3);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn fill_grad_residuals_plane(dst: &mut [i32], row: &[i16], up: &[i16]) {
    let Some((dst_first, dst_rest)) = dst.split_first_mut() else {
        return;
    };
    let (&row_first, current) = row.split_first().unwrap();
    let (&up_first, north) = up.split_first().unwrap();
    *dst_first = row_first as i32 - up_first as i32;
    let (_, west) = row.split_last().unwrap();
    let (_, northwest) = up.split_last().unwrap();

    let (dst_chunks, dst_tail) = dst_rest.as_chunks_mut::<16>();
    let (current_chunks, current_tail) = current.as_chunks::<16>();
    let (north_chunks, north_tail) = north.as_chunks::<16>();
    let (west_chunks, west_tail) = west.as_chunks::<16>();
    let (northwest_chunks, northwest_tail) = northwest.as_chunks::<16>();

    for ((((dst, current), north), west), northwest) in dst_chunks
        .iter_mut()
        .zip(current_chunks)
        .zip(north_chunks)
        .zip(west_chunks)
        .zip(northwest_chunks)
    {
        fill_grad_residuals_16(dst, current, north, west, northwest);
    }

    for ((((dst, &current), &north), &west), &northwest) in dst_tail
        .iter_mut()
        .zip(current_tail)
        .zip(north_tail)
        .zip(west_tail)
        .zip(northwest_tail)
    {
        *dst = grad_residual_scalar(current, north, west, northwest);
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn fill_ytob_residuals_wasm(
    rb: &mut [i32],
    ry: &mut [i32],
    b_row: &[i16],
    y_row: &[i16],
    b_up: &[i16],
    y_up: &[i16],
) {
    let len = rb
        .len()
        .min(ry.len())
        .min(b_row.len())
        .min(y_row.len())
        .min(b_up.len())
        .min(y_up.len());
    if len == 0 {
        return;
    }

    fill_grad_residuals_plane(&mut rb[..len], &b_row[..len], &b_up[..len]);
    fill_grad_residuals_plane(&mut ry[..len], &y_row[..len], &y_up[..len]);
}

#[target_feature(enable = "simd128")]
pub(crate) fn fill_ytob_row_wasm(dst: &mut [i32], b: &[i16], y: &[i16], slope: f32) {
    let len = dst.len().min(b.len()).min(y.len());
    let (dst_chunks, dst_tail) = dst[..len].as_chunks_mut::<16>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<16>();
    let (y_chunks, y_tail) = y[..len].as_chunks::<16>();
    let slope_vec = f32x4_splat(slope);

    for ((dst, b), y) in dst_chunks.iter_mut().zip(b_chunks).zip(y_chunks) {
        let b0 = unsafe { v128_load(b.as_ptr().cast()) };
        let b1 = unsafe { v128_load(b[8..].as_ptr().cast()) };
        let y0 = unsafe { v128_load(y.as_ptr().cast()) };
        let y1 = unsafe { v128_load(y[8..].as_ptr().cast()) };
        let result0 = fill_ytob_4(
            i32x4_extend_low_i16x8(b0),
            i32x4_extend_low_i16x8(y0),
            slope_vec,
        );
        let result1 = fill_ytob_4(
            i32x4_extend_high_i16x8(b0),
            i32x4_extend_high_i16x8(y0),
            slope_vec,
        );
        let result2 = fill_ytob_4(
            i32x4_extend_low_i16x8(b1),
            i32x4_extend_low_i16x8(y1),
            slope_vec,
        );
        let result3 = fill_ytob_4(
            i32x4_extend_high_i16x8(b1),
            i32x4_extend_high_i16x8(y1),
            slope_vec,
        );
        unsafe {
            v128_store(dst.as_mut_ptr().cast(), result0);
            v128_store(dst[4..].as_mut_ptr().cast(), result1);
            v128_store(dst[8..].as_mut_ptr().cast(), result2);
            v128_store(dst[12..].as_mut_ptr().cast(), result3);
        }
    }

    if !dst_tail.is_empty() {
        let mut b_padded = [0i16; 16];
        let mut y_padded = [0i16; 16];
        b_padded[..b_tail.len()].copy_from_slice(b_tail);
        y_padded[..y_tail.len()].copy_from_slice(y_tail);

        let b0 = unsafe { v128_load(b_padded.as_ptr().cast()) };
        let b1 = unsafe { v128_load(b_padded[8..].as_ptr().cast()) };
        let y0 = unsafe { v128_load(y_padded.as_ptr().cast()) };
        let y1 = unsafe { v128_load(y_padded[8..].as_ptr().cast()) };
        let result0 = fill_ytob_4(
            i32x4_extend_low_i16x8(b0),
            i32x4_extend_low_i16x8(y0),
            slope_vec,
        );
        let result1 = fill_ytob_4(
            i32x4_extend_high_i16x8(b0),
            i32x4_extend_high_i16x8(y0),
            slope_vec,
        );
        let result2 = fill_ytob_4(
            i32x4_extend_low_i16x8(b1),
            i32x4_extend_low_i16x8(y1),
            slope_vec,
        );
        let result3 = fill_ytob_4(
            i32x4_extend_high_i16x8(b1),
            i32x4_extend_high_i16x8(y1),
            slope_vec,
        );

        let mut result_padded = [0i32; 16];
        unsafe {
            v128_store(result_padded.as_mut_ptr().cast(), result0);
            v128_store(result_padded[4..].as_mut_ptr().cast(), result1);
            v128_store(result_padded[8..].as_mut_ptr().cast(), result2);
            v128_store(result_padded[12..].as_mut_ptr().cast(), result3);
        }
        dst_tail.copy_from_slice(&result_padded[..dst_tail.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::{fill_ytob_residuals_wasm, fill_ytob_row_wasm};
    use crate::color_correlation::{fill_ytob_residuals_scalar, fill_ytob_row_scalar};

    #[test]
    fn fill_ytob_row_matches_scalar_ties_to_even() {
        let b = std::array::from_fn::<_, 33, _>(|i| i as i16 * 100 - 1600);
        let y = std::array::from_fn::<_, 33, _>(|i| i as i16 * 2 - 31);
        for len in 0..=b.len() {
            let mut expected = [0i32; 33];
            let mut actual = [0i32; 33];
            fill_ytob_row_scalar(&mut expected[..len], &b[..len], &y[..len], 0.5);
            fill_ytob_row_wasm(&mut actual[..len], &b[..len], &y[..len], 0.5);
            assert_eq!(&actual[..len], &expected[..len], "length {len}");
        }
    }

    #[test]
    fn fill_ytob_residuals_matches_scalar() {
        let b_row =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 7919).wrapping_add(16381) as i16);
        let y_row =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 3253).wrapping_sub(22003) as i16);
        let b_up =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 1297).wrapping_sub(31111) as i16);
        let y_up =
            std::array::from_fn::<_, 35, _>(|i| (i as i32 * 5347).wrapping_add(27011) as i16);
        for len in 0..=b_row.len() {
            let mut expected_b = [0i32; 35];
            let mut expected_y = [0i32; 35];
            let mut actual_b = [0i32; 35];
            let mut actual_y = [0i32; 35];
            fill_ytob_residuals_scalar(
                &mut expected_b[..len],
                &mut expected_y[..len],
                &b_row[..len],
                &y_row[..len],
                &b_up[..len],
                &y_up[..len],
            );
            fill_ytob_residuals_wasm(
                &mut actual_b[..len],
                &mut actual_y[..len],
                &b_row[..len],
                &y_row[..len],
                &b_up[..len],
                &y_up[..len],
            );
            assert_eq!(&actual_b[..len], &expected_b[..len], "B length {len}");
            assert_eq!(&actual_y[..len], &expected_y[..len], "Y length {len}");
        }
    }
}
