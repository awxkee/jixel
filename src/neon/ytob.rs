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

use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn fill_ytob_4(b: int16x4_t, y: int16x4_t, slope: float32x4_t) -> int32x4_t {
    let b = vmovl_s16(b);
    let y = vmovl_s16(y);
    let adjusted = vmulq_f32(slope, vcvtq_f32_s32(y));
    vsubq_s32(b, vcvtnq_s32_f32(adjusted))
}

#[inline]
#[target_feature(enable = "neon")]
fn grad_residual_4(
    current: int16x4_t,
    north: int16x4_t,
    west: int16x4_t,
    northwest: int16x4_t,
) -> int32x4_t {
    let current = vmovl_s16(current);
    let north = vmovl_s16(north);
    let west = vmovl_s16(west);
    let northwest = vmovl_s16(northwest);
    let prediction = vsubq_s32(vaddq_s32(north, west), northwest);
    let prediction = vmaxq_s32(
        vminq_s32(north, west),
        vminq_s32(vmaxq_s32(north, west), prediction),
    );
    vsubq_s32(current, prediction)
}

#[inline]
fn grad_residual_scalar(current: i16, north: i16, west: i16, northwest: i16) -> i32 {
    let (current, north, west, northwest) =
        (current as i32, north as i32, west as i32, northwest as i32);
    current - (north + west - northwest).clamp(north.min(west), north.max(west))
}

#[inline]
#[target_feature(enable = "neon")]
fn fill_grad_residuals_16(
    dst: &mut [i32; 16],
    current: &[i16; 16],
    north: &[i16; 16],
    west: &[i16; 16],
    northwest: &[i16; 16],
) {
    let current0 = unsafe { vld1q_s16(current.as_ptr()) };
    let current1 = unsafe { vld1q_s16(current[8..].as_ptr()) };
    let north0 = unsafe { vld1q_s16(north.as_ptr()) };
    let north1 = unsafe { vld1q_s16(north[8..].as_ptr()) };
    let west0 = unsafe { vld1q_s16(west.as_ptr()) };
    let west1 = unsafe { vld1q_s16(west[8..].as_ptr()) };
    let northwest0 = unsafe { vld1q_s16(northwest.as_ptr()) };
    let northwest1 = unsafe { vld1q_s16(northwest[8..].as_ptr()) };
    let result0 = grad_residual_4(
        vget_low_s16(current0),
        vget_low_s16(north0),
        vget_low_s16(west0),
        vget_low_s16(northwest0),
    );
    let result1 = grad_residual_4(
        vget_high_s16(current0),
        vget_high_s16(north0),
        vget_high_s16(west0),
        vget_high_s16(northwest0),
    );
    let result2 = grad_residual_4(
        vget_low_s16(current1),
        vget_low_s16(north1),
        vget_low_s16(west1),
        vget_low_s16(northwest1),
    );
    let result3 = grad_residual_4(
        vget_high_s16(current1),
        vget_high_s16(north1),
        vget_high_s16(west1),
        vget_high_s16(northwest1),
    );
    unsafe {
        vst1q_s32(dst.as_mut_ptr(), result0);
        vst1q_s32(dst[4..].as_mut_ptr(), result1);
        vst1q_s32(dst[8..].as_mut_ptr(), result2);
        vst1q_s32(dst[12..].as_mut_ptr(), result3);
    }
}

#[inline]
#[target_feature(enable = "neon")]
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

#[target_feature(enable = "neon")]
pub(crate) fn fill_ytob_residuals_neon(
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

#[inline]
#[target_feature(enable = "neon")]
fn ytob_weight_indices_4(
    rb: int32x4_t,
    ry: int32x4_t,
    step: float32x4_t,
) -> (int32x4_t, int32x4_t) {
    let ratio = vdivq_f32(vcvtq_f32_s32(rb), vmulq_f32(vcvtq_f32_s32(ry), step));
    let k = vcvtnq_s32_f32(ratio);
    let k = vmaxq_s32(vdupq_n_s32(-127), vminq_s32(vdupq_n_s32(127), k));
    (vaddq_s32(k, vdupq_n_s32(127)), vabsq_s32(ry))
}

#[inline(always)]
fn scatter_ytob_weights(
    indices: &[i32; 16],
    magnitudes: &[i32; 16],
    lanes: usize,
    weights: &mut [u64],
) {
    for (&idx, &magnitude) in indices[..lanes].iter().zip(&magnitudes[..lanes]) {
        if magnitude != 0 {
            weights[idx as usize] += magnitude as u64;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_ytob_weights_16(
    rb: &[i32; 16],
    ry: &[i32; 16],
    step: float32x4_t,
    lanes: usize,
    weights: &mut [u64],
) {
    let rb0 = unsafe { vld1q_s32(rb.as_ptr()) };
    let rb1 = unsafe { vld1q_s32(rb[4..].as_ptr()) };
    let rb2 = unsafe { vld1q_s32(rb[8..].as_ptr()) };
    let rb3 = unsafe { vld1q_s32(rb[12..].as_ptr()) };
    let ry0 = unsafe { vld1q_s32(ry.as_ptr()) };
    let ry1 = unsafe { vld1q_s32(ry[4..].as_ptr()) };
    let ry2 = unsafe { vld1q_s32(ry[8..].as_ptr()) };
    let ry3 = unsafe { vld1q_s32(ry[12..].as_ptr()) };
    let (indices0, magnitudes0) = ytob_weight_indices_4(rb0, ry0, step);
    let (indices1, magnitudes1) = ytob_weight_indices_4(rb1, ry1, step);
    let (indices2, magnitudes2) = ytob_weight_indices_4(rb2, ry2, step);
    let (indices3, magnitudes3) = ytob_weight_indices_4(rb3, ry3, step);

    let mut indices = [0i32; 16];
    let mut magnitudes = [0i32; 16];
    unsafe {
        vst1q_s32(indices.as_mut_ptr(), indices0);
        vst1q_s32(indices[4..].as_mut_ptr(), indices1);
        vst1q_s32(indices[8..].as_mut_ptr(), indices2);
        vst1q_s32(indices[12..].as_mut_ptr(), indices3);
        vst1q_s32(magnitudes.as_mut_ptr(), magnitudes0);
        vst1q_s32(magnitudes[4..].as_mut_ptr(), magnitudes1);
        vst1q_s32(magnitudes[8..].as_mut_ptr(), magnitudes2);
        vst1q_s32(magnitudes[12..].as_mut_ptr(), magnitudes3);
    }
    scatter_ytob_weights(&indices, &magnitudes, lanes, weights);
}

#[target_feature(enable = "neon")]
pub(crate) fn accumulate_ytob_weights_neon(rb: &[i32], ry: &[i32], step: f32, weights: &mut [u64]) {
    let len = rb.len().min(ry.len());
    let (rb_chunks, rb_tail) = rb[..len].as_chunks::<16>();
    let (ry_chunks, ry_tail) = ry[..len].as_chunks::<16>();
    let step = vdupq_n_f32(step);

    for (rb, ry) in rb_chunks.iter().zip(ry_chunks) {
        accumulate_ytob_weights_16(rb, ry, step, 16, weights);
    }

    if !rb_tail.is_empty() {
        let mut rb_padded = [0i32; 16];
        let mut ry_padded = [0i32; 16];
        rb_padded[..rb_tail.len()].copy_from_slice(rb_tail);
        ry_padded[..ry_tail.len()].copy_from_slice(ry_tail);
        accumulate_ytob_weights_16(&rb_padded, &ry_padded, step, rb_tail.len(), weights);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn fill_ytob_row_neon(dst: &mut [i32], b: &[i16], y: &[i16], slope: f32) {
    let len = dst.len().min(b.len()).min(y.len());
    let (dst_chunks, dst_tail) = dst[..len].as_chunks_mut::<16>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<16>();
    let (y_chunks, y_tail) = y[..len].as_chunks::<16>();
    let slope_vec = vdupq_n_f32(slope);

    for ((dst, b), y) in dst_chunks.iter_mut().zip(b_chunks).zip(y_chunks) {
        let b0 = unsafe { vld1q_s16(b.as_ptr()) };
        let b1 = unsafe { vld1q_s16(b[8..].as_ptr()) };
        let y0 = unsafe { vld1q_s16(y.as_ptr()) };
        let y1 = unsafe { vld1q_s16(y[8..].as_ptr()) };
        let result0 = fill_ytob_4(vget_low_s16(b0), vget_low_s16(y0), slope_vec);
        let result1 = fill_ytob_4(vget_high_s16(b0), vget_high_s16(y0), slope_vec);
        let result2 = fill_ytob_4(vget_low_s16(b1), vget_low_s16(y1), slope_vec);
        let result3 = fill_ytob_4(vget_high_s16(b1), vget_high_s16(y1), slope_vec);
        unsafe {
            vst1q_s32(dst.as_mut_ptr(), result0);
            vst1q_s32(dst[4..].as_mut_ptr(), result1);
            vst1q_s32(dst[8..].as_mut_ptr(), result2);
            vst1q_s32(dst[12..].as_mut_ptr(), result3);
        }
    }

    if !dst_tail.is_empty() {
        let mut b_padded = [0i16; 16];
        let mut y_padded = [0i16; 16];
        b_padded[..b_tail.len()].copy_from_slice(b_tail);
        y_padded[..y_tail.len()].copy_from_slice(y_tail);

        let b0 = unsafe { vld1q_s16(b_padded.as_ptr()) };
        let b1 = unsafe { vld1q_s16(b_padded[8..].as_ptr()) };
        let y0 = unsafe { vld1q_s16(y_padded.as_ptr()) };
        let y1 = unsafe { vld1q_s16(y_padded[8..].as_ptr()) };
        let result0 = fill_ytob_4(vget_low_s16(b0), vget_low_s16(y0), slope_vec);
        let result1 = fill_ytob_4(vget_high_s16(b0), vget_high_s16(y0), slope_vec);
        let result2 = fill_ytob_4(vget_low_s16(b1), vget_low_s16(y1), slope_vec);
        let result3 = fill_ytob_4(vget_high_s16(b1), vget_high_s16(y1), slope_vec);

        let mut result_padded = [0i32; 16];
        unsafe {
            vst1q_s32(result_padded.as_mut_ptr(), result0);
            vst1q_s32(result_padded[4..].as_mut_ptr(), result1);
            vst1q_s32(result_padded[8..].as_mut_ptr(), result2);
            vst1q_s32(result_padded[12..].as_mut_ptr(), result3);
        }
        dst_tail.copy_from_slice(&result_padded[..dst_tail.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::{accumulate_ytob_weights_neon, fill_ytob_residuals_neon, fill_ytob_row_neon};
    use crate::enc_color_correlation::{
        accumulate_ytob_weights_scalar, fill_ytob_residuals_scalar, fill_ytob_row_scalar,
    };

    #[test]
    fn fill_ytob_row_matches_scalar_ties_to_even() {
        let b = std::array::from_fn::<_, 33, _>(|i| i as i16 * 100 - 1600);
        let y = std::array::from_fn::<_, 33, _>(|i| i as i16 * 2 - 31);
        for len in 0..=b.len() {
            let mut expected = [0i32; 33];
            let mut actual = [0i32; 33];
            fill_ytob_row_scalar(&mut expected[..len], &b[..len], &y[..len], 0.5);
            unsafe {
                fill_ytob_row_neon(&mut actual[..len], &b[..len], &y[..len], 0.5);
            }
            assert_eq!(&actual[..len], &expected[..len], "length {len}");
        }
    }

    #[test]
    fn accumulate_ytob_weights_matches_scalar() {
        let rb = std::array::from_fn::<_, 33, _>(|i| (i as i32 - 16) * 173);
        let ry = std::array::from_fn::<_, 33, _>(|i| {
            if i.is_multiple_of(7) {
                0
            } else {
                (i as i32 - 15) * 11
            }
        });
        for &step in &[0.5f32, 1.0 / 168.0, 0.125] {
            for len in 0..=rb.len() {
                let mut expected = [0u64; 255];
                let mut actual = [0u64; 255];
                accumulate_ytob_weights_scalar(&rb[..len], &ry[..len], step, &mut expected);
                unsafe {
                    accumulate_ytob_weights_neon(&rb[..len], &ry[..len], step, &mut actual);
                }
                assert_eq!(actual, expected, "length {len}, step {step}");
            }
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
            unsafe {
                fill_ytob_residuals_neon(
                    &mut actual_b[..len],
                    &mut actual_y[..len],
                    &b_row[..len],
                    &y_row[..len],
                    &b_up[..len],
                    &y_up[..len],
                );
            }
            assert_eq!(&actual_b[..len], &expected_b[..len], "B length {len}");
            assert_eq!(&actual_y[..len], &expected_y[..len], "Y length {len}");
        }
    }
}
