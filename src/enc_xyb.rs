/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::image::Image3F;

const M00: f32 = 0.30;
const M02: f32 = 0.078;
const M01: f32 = 1.0 - M02 - M00;

const M10: f32 = 0.23;
const M12: f32 = 0.078;
const M11: f32 = 1.0 - M12 - M10;

const M20: f32 = 0.243_422_69;
const M21: f32 = 0.204_767_45;
const M22: f32 = 1.0 - M20 - M21;

const OPSIN_BIAS: f32 = 0.003_793_073_4;
const NEG_BIAS_CBRT: f32 = -0.155_954_2;

#[cfg_attr(feature = "fixed-xyb", allow(dead_code))]
#[inline(always)]
fn rgb_to_xyb_pixel_f32(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let mixed0 = M00 * r + M01 * g + M02 * b + OPSIN_BIAS;
    let mixed1 = M10 * r + M11 * g + M12 * b + OPSIN_BIAS;
    let mixed2 = M20 * r + M21 * g + M22 * b + OPSIN_BIAS;

    let tm0 = mixed0.max(0.0).cbrt() + NEG_BIAS_CBRT;
    let tm1 = mixed1.max(0.0).cbrt() + NEG_BIAS_CBRT;
    let tm2 = mixed2.max(0.0).cbrt() + NEG_BIAS_CBRT;

    (0.5 * (tm0 - tm1), 0.5 * (tm0 + tm1), tm2)
}

#[cfg_attr(feature = "fixed-xyb", allow(dead_code))]
fn to_xyb_f32(image: &mut Image3F) {
    let ysize = image.ysize();
    for y in 0..ysize {
        let [r_row, g_row, b_row] = image.all_plane_rows_mut(y);
        for ((r, g), b) in r_row.iter_mut().zip(g_row.iter_mut()).zip(b_row.iter_mut()) {
            let (xv, yv, bv) = rgb_to_xyb_pixel_f32(*r, *g, *b);
            *r = xv;
            *g = yv;
            *b = bv;
        }
    }
}

const FP_SHIFT: u32 = 20;
const FP_ONE: i32 = 1 << FP_SHIFT; // 1_048_576

const CBRT_LUT_MAX_F: f32 = 1.1;
const CBRT_LUT_LEN: usize = 128;

struct FixedTables {
    // Matrix rows in Q20.
    m00: i32,
    m01: i32,
    m02: i32,
    m10: i32,
    m11: i32,
    m12: i32,
    m20: i32,
    m21: i32,
    m22: i32,
    opsin_bias: i32,
    neg_bias_cbrt: i32,
    cbrt_lut: [i32; CBRT_LUT_LEN + 1],
    inv_step_q20: i32,
}

#[inline]
fn to_q20(x: f32) -> i32 {
    (x * FP_ONE as f32).round() as i32
}

fn build_tables() -> FixedTables {
    let step = CBRT_LUT_MAX_F / (CBRT_LUT_LEN - 1) as f32;
    let mut cbrt_lut = [0i32; CBRT_LUT_LEN + 1];
    for (i, slot) in cbrt_lut.iter_mut().enumerate() {
        let x = (i.min(CBRT_LUT_LEN - 1) as f32) * step;
        *slot = to_q20(x.cbrt());
    }
    let inv_step = (CBRT_LUT_LEN - 1) as f32 / CBRT_LUT_MAX_F;
    FixedTables {
        m00: to_q20(M00),
        m01: to_q20(M01),
        m02: to_q20(M02),
        m10: to_q20(M10),
        m11: to_q20(M11),
        m12: to_q20(M12),
        m20: to_q20(M20),
        m21: to_q20(M21),
        m22: to_q20(M22),
        opsin_bias: to_q20(OPSIN_BIAS),
        neg_bias_cbrt: to_q20(NEG_BIAS_CBRT),
        cbrt_lut,
        inv_step_q20: to_q20(inv_step),
    }
}

fn tables() -> &'static FixedTables {
    use std::sync::OnceLock;
    static T: OnceLock<FixedTables> = OnceLock::new();
    T.get_or_init(build_tables)
}

/// Cube root of `x_q20` (must be ≥ 0) via direct LUT + linear interpolation.
/// Returns result in Q20.
#[inline(always)]
fn cbrt_q20(x_q20: i32, t: &FixedTables) -> i32 {
    if x_q20 <= 0 {
        return 0;
    }
    let prod = (x_q20 as i64) * (t.inv_step_q20 as i64);
    let idx_full = (prod >> FP_SHIFT) as u32;
    let idx = (idx_full >> FP_SHIFT) as usize;
    let idx = idx.min(CBRT_LUT_LEN - 1);
    //
    let frac = (idx_full & (FP_ONE as u32 - 1)) as i32;

    let a = t.cbrt_lut[idx];
    let b = t.cbrt_lut[idx + 1];
    // Linear interpolation: a + (b-a)*frac/2^20, Q20.
    a + (((b - a) as i64 * frac as i64) >> FP_SHIFT) as i32
}

#[inline(always)]
fn mix_q20(r: i32, g: i32, b: i32, m0: i32, m1: i32, m2: i32) -> i32 {
    let s = (r as i64) * (m0 as i64) + (g as i64) * (m1 as i64) + (b as i64) * (m2 as i64);
    (s >> FP_SHIFT) as i32
}

#[inline(always)]
fn rgb_to_xyb_pixel_q20(r_q: i32, g_q: i32, b_q: i32, t: &FixedTables) -> (i32, i32, i32) {
    let mixed0 = mix_q20(r_q, g_q, b_q, t.m00, t.m01, t.m02) + t.opsin_bias;
    let mixed1 = mix_q20(r_q, g_q, b_q, t.m10, t.m11, t.m12) + t.opsin_bias;
    let mixed2 = mix_q20(r_q, g_q, b_q, t.m20, t.m21, t.m22) + t.opsin_bias;

    let tm0 = cbrt_q20(mixed0.max(0), t) + t.neg_bias_cbrt;
    let tm1 = cbrt_q20(mixed1.max(0), t) + t.neg_bias_cbrt;
    let tm2 = cbrt_q20(mixed2.max(0), t) + t.neg_bias_cbrt;

    let x = (tm0 - tm1) >> 1;
    let y = (tm0 + tm1) >> 1;
    (x, y, tm2)
}

fn to_xyb_fixed(image: &mut Image3F) {
    let t = tables();
    let scale_in = FP_ONE as f32;
    let scale_out = 1.0 / FP_ONE as f32;
    let ysize = image.ysize();
    for y in 0..ysize {
        let [r_row, g_row, b_row] = image.all_plane_rows_mut(y);
        for ((r, g), b) in r_row.iter_mut().zip(g_row.iter_mut()).zip(b_row.iter_mut()) {
            let r_q = (*r * scale_in) as i32;
            let g_q = (*g * scale_in) as i32;
            let b_q = (*b * scale_in) as i32;
            let (xv, yv, bv) = rgb_to_xyb_pixel_q20(r_q, g_q, b_q, t);
            *r = xv as f32 * scale_out;
            *g = yv as f32 * scale_out;
            *b = bv as f32 * scale_out;
        }
    }
}

/// Convert in-place from linear RGB (planes 0/1/2) to XYB.
pub(crate) fn to_xyb(image: &mut Image3F) {
    #[cfg(feature = "fixed-xyb")]
    {
        to_xyb_fixed(image);
    }
    #[cfg(not(feature = "fixed-xyb"))]
    {
        to_xyb_f32(image);
    }
}
