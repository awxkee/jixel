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
use crate::dct::fmla;
use crate::image::Image3F;
use std::sync::OnceLock;

pub(crate) const M00: f32 = 0.30;
pub(crate) const M02: f32 = 0.078;
pub(crate) const M01: f32 = 1.0 - M02 - M00;

pub(crate) const M10: f32 = 0.23;
pub(crate) const M12: f32 = 0.078;
pub(crate) const M11: f32 = 1.0 - M12 - M10;

pub(crate) const M20: f32 = 0.243_422_69;
pub(crate) const M21: f32 = 0.204_767_45;
pub(crate) const M22: f32 = 1.0 - M20 - M21;

pub(crate) const OPSIN_BIAS: f32 = 0.003_793_073_4;
pub(crate) const NEG_BIAS_CBRT: f32 = -0.155_954_2;

#[allow(unused)]
#[inline(always)]
fn halley_refine(x: f32, a: f32) -> f32 {
    let tx = x * x * x;
    x * fmla(2f32, a, tx) / fmla(2f32, tx, a)
}

#[allow(unused)]
#[inline]
fn cbrtf(x: f32) -> f32 {
    const B1: u32 = 709958130;
    let mut t: f32;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7fffffff;

    hx = (hx / 3).wrapping_add(B1);
    ui &= 0x80000000;
    ui |= hx;

    t = f32::from_bits(ui);
    t = halley_refine(t, x);
    halley_refine(t, x)
}

#[allow(unused)]
#[inline(always)]
fn rgb_to_xyb_pixel_f32(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let mixed0 = fmla(M00, r, fmla(M01, g, fmla(M02, b, OPSIN_BIAS)));
    let mixed1 = fmla(M10, r, fmla(M11, g, fmla(M12, b, OPSIN_BIAS)));
    let mixed2 = fmla(M20, r, fmla(M21, g, fmla(M22, b, OPSIN_BIAS)));

    let tm0 = cbrtf(mixed0.max(0.0)) + NEG_BIAS_CBRT;
    let tm1 = cbrtf(mixed1.max(0.0)) + NEG_BIAS_CBRT;
    let tm2 = cbrtf(mixed2.max(0.0)) + NEG_BIAS_CBRT;

    (0.5 * (tm0 - tm1), 0.5 * (tm0 + tm1), tm2)
}

#[allow(unused)]
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

type ToXybFn = unsafe fn(&mut Image3F);

/// Convert in-place from linear RGB (planes 0/1/2) to XYB.
pub(crate) fn to_xyb(image: &mut Image3F) {
    static TO_XYB_FN: OnceLock<ToXybFn> = OnceLock::new();
    unsafe {
        TO_XYB_FN.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    return crate::avx::to_xyb_avx2;
                }
            }
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
            {
                if std::is_x86_feature_detected!("sse4.1") {
                    return crate::sse::to_xyb_sse41;
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                use crate::neon::to_xyb_neon;
                to_xyb_neon
            }
            #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
            {
                crate::wasm::to_xyb_wasm
            }
            #[cfg(not(any(
                all(target_arch = "aarch64", feature = "neon"),
                all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
            )))]
            {
                to_xyb_f32
            }
        })(image)
    }
}
