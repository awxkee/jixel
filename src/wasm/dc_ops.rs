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

use crate::dc_ops::{B_DC_Y_COMPENSATION, BdcCompensation};
use std::arch::wasm32::*;

#[derive(Clone, Copy)]
struct CompensationVectors {
    negative_scale: v128,
    y_step: v128,
    weight: v128,
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn load_f32(ptr: *const f32) -> v128 {
    unsafe { v128_load(ptr.cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn load_i16(ptr: *const i16) -> v128 {
    let value = unsafe { v128_load64_zero(ptr.cast()) };
    f32x4_convert_i32x4(i32x4_extend_low_i16x8(value))
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn store_f32(ptr: *mut f32, value: v128) {
    unsafe { v128_store(ptr.cast(), value) };
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn store_rounded_i16(ptr: *mut i16, value: v128) {
    let value = super::quant::round_ties_away_i32x4(value);
    let packed = i16x8_narrow_i32x4(value, value);
    unsafe { v128_store64_lane::<0>(packed, ptr.cast()) };
}

#[inline]
#[target_feature(enable = "simd128")]
fn adjust_value(target: v128, source_y: v128, yq: v128, p: CompensationVectors) -> v128 {
    let correction = f32x4_mul(source_y, p.negative_scale);
    let error = f32x4_add(f32x4_mul(yq, p.y_step), correction);
    f32x4_add(f32x4_mul(p.weight, error), target)
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn compensate_block(
    b: *const f32,
    y: *const f32,
    yq: *const i16,
    out: *mut i16,
    scale: v128,
    negative_cfl: v128,
    p: CompensationVectors,
) {
    unsafe {
        let b = load_f32(b);
        let y = load_f32(y);
        let yq = load_i16(yq);
        let correction = f32x4_mul(yq, negative_cfl);
        let target = f32x4_add(f32x4_mul(b, scale), correction);
        store_rounded_i16(out, adjust_value(target, y, yq, p));
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn adjust_block(t: *mut f32, y: *const f32, yq: *const i16, p: CompensationVectors) {
    unsafe {
        let target = load_f32(t);
        let y = load_f32(y);
        let yq = load_i16(yq);
        store_f32(t, adjust_value(target, y, yq, p));
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn reconstruct_block(
    x: *const i16,
    y: *const i16,
    b: *const i16,
    rx: *mut f32,
    ry: *mut f32,
    rb: *mut f32,
    cfl: [v128; 2],
    steps: [v128; 3],
) {
    unsafe {
        let x = load_i16(x);
        let y = load_i16(y);
        let b = load_i16(b);
        store_f32(rx, f32x4_mul(f32x4_add(f32x4_mul(y, cfl[0]), x), steps[0]));
        store_f32(ry, f32x4_mul(y, steps[1]));
        store_f32(rb, f32x4_mul(f32x4_add(f32x4_mul(y, cfl[1]), b), steps[2]));
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn compensate_b_dc_wasm(
    source_b: &[f32],
    source_y: &[f32],
    y_quant: &[i16],
    compensation: BdcCompensation,
    cfl_b: f32,
    output: &mut [i16],
) {
    let n = output.len();
    assert_eq!(source_b.len(), n);
    assert_eq!(source_y.len(), n);
    assert_eq!(y_quant.len(), n);
    let scale = f32x4_splat(compensation.scale_b);
    let negative_cfl = f32x4_splat(-cfl_b);
    let p = CompensationVectors {
        negative_scale: f32x4_splat(-compensation.scale_b),
        y_step: f32x4_splat(compensation.y_step_b),
        weight: f32x4_splat(B_DC_Y_COMPENSATION),
    };
    let mut i = 0;
    // All rows have the checked length; each full block accesses 4 lanes.
    unsafe {
        while n - i >= 16 {
            compensate_block(
                source_b.as_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                output.as_mut_ptr().add(i),
                scale,
                negative_cfl,
                p,
            );
            compensate_block(
                source_b.as_ptr().add(i + 4),
                source_y.as_ptr().add(i + 4),
                y_quant.as_ptr().add(i + 4),
                output.as_mut_ptr().add(i + 4),
                scale,
                negative_cfl,
                p,
            );
            compensate_block(
                source_b.as_ptr().add(i + 8),
                source_y.as_ptr().add(i + 8),
                y_quant.as_ptr().add(i + 8),
                output.as_mut_ptr().add(i + 8),
                scale,
                negative_cfl,
                p,
            );
            compensate_block(
                source_b.as_ptr().add(i + 12),
                source_y.as_ptr().add(i + 12),
                y_quant.as_ptr().add(i + 12),
                output.as_mut_ptr().add(i + 12),
                scale,
                negative_cfl,
                p,
            );
            i += 16;
        }
        while n - i >= 4 {
            compensate_block(
                source_b.as_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                output.as_mut_ptr().add(i),
                scale,
                negative_cfl,
                p,
            );
            i += 4;
        }
    }
    if i < n {
        // Keep tails on the SIMD rounding path, without reading past a row.
        let tail = n - i;
        let mut b = [0.0; 4];
        let mut y = [0.0; 4];
        let mut yq = [0i16; 4];
        let mut out = [0i16; 4];
        b[..tail].copy_from_slice(&source_b[i..]);
        y[..tail].copy_from_slice(&source_y[i..]);
        yq[..tail].copy_from_slice(&y_quant[i..]);
        unsafe {
            compensate_block(
                b.as_ptr(),
                y.as_ptr(),
                yq.as_ptr(),
                out.as_mut_ptr(),
                scale,
                negative_cfl,
                p,
            );
        }
        output[i..].copy_from_slice(&out[..tail]);
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn adjust_b_dc_wasm(
    targets: &mut [f32],
    source_y: &[f32],
    y_quant: &[i16],
    compensation: BdcCompensation,
) {
    let n = targets.len();
    assert_eq!(source_y.len(), n);
    assert_eq!(y_quant.len(), n);
    let p = CompensationVectors {
        negative_scale: f32x4_splat(-compensation.scale_b),
        y_step: f32x4_splat(compensation.y_step_b),
        weight: f32x4_splat(B_DC_Y_COMPENSATION),
    };
    let mut i = 0;
    // All rows have the checked length; each full block accesses 4 lanes.
    unsafe {
        while n - i >= 16 {
            adjust_block(
                targets.as_mut_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                p,
            );
            adjust_block(
                targets.as_mut_ptr().add(i + 4),
                source_y.as_ptr().add(i + 4),
                y_quant.as_ptr().add(i + 4),
                p,
            );
            adjust_block(
                targets.as_mut_ptr().add(i + 8),
                source_y.as_ptr().add(i + 8),
                y_quant.as_ptr().add(i + 8),
                p,
            );
            adjust_block(
                targets.as_mut_ptr().add(i + 12),
                source_y.as_ptr().add(i + 12),
                y_quant.as_ptr().add(i + 12),
                p,
            );
            i += 16;
        }
        while n - i >= 4 {
            adjust_block(
                targets.as_mut_ptr().add(i),
                source_y.as_ptr().add(i),
                y_quant.as_ptr().add(i),
                p,
            );
            i += 4;
        }
    }
    if i < n {
        let tail = n - i;
        let mut t = [0.0; 4];
        let mut y = [0.0; 4];
        let mut yq = [0i16; 4];
        t[..tail].copy_from_slice(&targets[i..]);
        y[..tail].copy_from_slice(&source_y[i..]);
        yq[..tail].copy_from_slice(&y_quant[i..]);
        unsafe {
            adjust_block(t.as_mut_ptr(), y.as_ptr(), yq.as_ptr(), p);
        }
        targets[i..].copy_from_slice(&t[..tail]);
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn reconstruct_dc_wasm(
    input: [&[i16]; 3],
    cfl: [f32; 2],
    steps: [f32; 3],
    output: [&mut [f32]; 3],
) {
    let [x, y, b] = input;
    let [rx, ry, rb] = output;
    let n = y.len();
    for len in [x.len(), b.len(), rx.len(), ry.len(), rb.len()] {
        assert_eq!(len, n);
    }
    let cfl = cfl.map(|v| f32x4_splat(v));
    let steps = steps.map(|v| f32x4_splat(v));
    let mut i = 0;
    // All rows have the checked length; each full block accesses 4 lanes.
    unsafe {
        while n - i >= 16 {
            reconstruct_block(
                x.as_ptr().add(i),
                y.as_ptr().add(i),
                b.as_ptr().add(i),
                rx.as_mut_ptr().add(i),
                ry.as_mut_ptr().add(i),
                rb.as_mut_ptr().add(i),
                cfl,
                steps,
            );
            reconstruct_block(
                x.as_ptr().add(i + 4),
                y.as_ptr().add(i + 4),
                b.as_ptr().add(i + 4),
                rx.as_mut_ptr().add(i + 4),
                ry.as_mut_ptr().add(i + 4),
                rb.as_mut_ptr().add(i + 4),
                cfl,
                steps,
            );
            reconstruct_block(
                x.as_ptr().add(i + 8),
                y.as_ptr().add(i + 8),
                b.as_ptr().add(i + 8),
                rx.as_mut_ptr().add(i + 8),
                ry.as_mut_ptr().add(i + 8),
                rb.as_mut_ptr().add(i + 8),
                cfl,
                steps,
            );
            reconstruct_block(
                x.as_ptr().add(i + 12),
                y.as_ptr().add(i + 12),
                b.as_ptr().add(i + 12),
                rx.as_mut_ptr().add(i + 12),
                ry.as_mut_ptr().add(i + 12),
                rb.as_mut_ptr().add(i + 12),
                cfl,
                steps,
            );
            i += 16;
        }
        while n - i >= 4 {
            reconstruct_block(
                x.as_ptr().add(i),
                y.as_ptr().add(i),
                b.as_ptr().add(i),
                rx.as_mut_ptr().add(i),
                ry.as_mut_ptr().add(i),
                rb.as_mut_ptr().add(i),
                cfl,
                steps,
            );
            i += 4;
        }
    }
    if i < n {
        let tail = n - i;
        let mut tx = [0i16; 4];
        let mut ty = [0i16; 4];
        let mut tb = [0i16; 4];
        let mut ox = [0.0; 4];
        let mut oy = [0.0; 4];
        let mut ob = [0.0; 4];
        tx[..tail].copy_from_slice(&x[i..]);
        ty[..tail].copy_from_slice(&y[i..]);
        tb[..tail].copy_from_slice(&b[i..]);
        unsafe {
            reconstruct_block(
                tx.as_ptr(),
                ty.as_ptr(),
                tb.as_ptr(),
                ox.as_mut_ptr(),
                oy.as_mut_ptr(),
                ob.as_mut_ptr(),
                cfl,
                steps,
            );
        }
        rx[i..].copy_from_slice(&ox[..tail]);
        ry[i..].copy_from_slice(&oy[..tail]);
        rb[i..].copy_from_slice(&ob[..tail]);
    }
}
