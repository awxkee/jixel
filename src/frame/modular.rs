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

//! Input preparation and calibration for the lossy modular encoding arm.

use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

/// Auto gate of the lossy-modular arm: byte cushion for the calibration's
/// per-image quality noise.
pub(super) const LM_GATE_MARGIN: f64 = 1.08;

/// Texture scaling of the lossy-modular arm's calibrated distance: the
/// VarDCT-matching ratio tracks luma gradient energy (textured images want
/// the arm coarser, smooth ones finer). Reference gradient, exponents below
/// and above it.
const LM_GRAD_REF: f32 = 0.0231;
const LM_GRAD_EXP_SMOOTH: f32 = 0.16;
const LM_GRAD_EXP_TEXTURED: f32 = 0.08;

/// `(gradient / LM_GRAD_REF)^exp` from the mean absolute horizontal and
/// vertical luminance step of the linear image, clamped to the fitted range.
pub(super) fn lm_texture_factor(linear: &Image3F) -> f32 {
    let (w, h) = (linear.xsize(), linear.ysize());
    let n = w * h;
    let (r, g, b) = (
        &linear.plane_data(0)[..n],
        &linear.plane_data(1)[..n],
        &linear.plane_data(2)[..n],
    );
    let luma = |i: usize| 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
    let mut sum_h = 0.0f64;
    let mut sum_v = 0.0f64;
    for y in 0..h {
        let row = y * w;
        let mut prev = luma(row);
        for x in 1..w {
            let cur = luma(row + x);
            sum_h += (cur - prev).abs() as f64;
            prev = cur;
        }
        if y + 1 < h {
            for x in 0..w {
                sum_v += (luma(row + w + x) - luma(row + x)).abs() as f64;
            }
        }
    }
    let mean_h = if w > 1 {
        sum_h / ((w - 1) * h) as f64
    } else {
        0.0
    };
    let mean_v = if h > 1 {
        sum_v / (w * (h - 1)) as f64
    } else {
        0.0
    };
    let ratio = (0.5 * (mean_h + mean_v)) as f32 / LM_GRAD_REF;
    let exp = if ratio < 1.0 {
        LM_GRAD_EXP_SMOOTH
    } else {
        LM_GRAD_EXP_TEXTURED
    };
    ratio.clamp(0.13, 4.3).powf(exp)
}

/// Integer lattice of the plain lossy-modular arm. Squeeze rounds its
/// averages to the lattice at every level
const LM_PLAIN_MAX: f32 = 65535.0;

/// The non-XYB frame's integer planes for the lossy-modular arm on the
/// `LM_PLAIN_MAX` lattice: the coded color planes (gray: its one channel),
/// then K at its squeeze distance (rescaled to the lattice) or alpha as
/// stored (coded losslessly). None for float alpha.
pub(super) fn lm_plain_input(
    ctx: &EncodingContext,
    coded: &Image3F,
    gray: bool,
    alpha: Option<&AlphaPlane>,
) -> Option<crate::lossless::LmPlainInput> {
    let n = coded.xsize() * coded.ysize();
    let max = LM_PLAIN_MAX;
    let quantize = |input: &[f32]| {
        let mut levels = vec![0i32; input.len()];
        (ctx.quantize_dc_i32)(input, max, &mut levels);
        levels
    };
    let color: Vec<Vec<i32>> = if gray {
        // Gray is coded as three equal channels; the modular frame carries
        // the sample itself (YCbCr's luma plane sits 128/255 below it).
        let offset = match ctx.coding {
            crate::coding::CodingTransform::YCbCr => 128.0 / 255.0,
            _ => 0.0,
        };
        if offset == 0.0 {
            vec![quantize(&coded.plane_data(1)[..n])]
        } else {
            let width = coded.xsize().max(1);
            let mut source = vec![0.0f32; width];
            let mut levels = vec![0i32; n];
            for (row, target) in coded.plane_data(1)[..n]
                .chunks(width)
                .zip(levels.chunks_mut(width))
            {
                for (value, &input) in source.iter_mut().zip(row) {
                    *value = input + offset;
                }
                (ctx.quantize_dc_i32)(&source[..row.len()], max, target);
            }
            vec![levels]
        }
    } else {
        (0..3)
            .map(|c| quantize(&coded.plane_data(c)[..n]))
            .collect()
    };
    let extra: Vec<Vec<i32>> = match alpha {
        Some(AlphaPlane::F32(_)) => return None,
        Some(plane) if ctx.extra_squeeze_distance > 0.0 => {
            let scale = max / ((1u32 << plane.bits()) - 1) as f32;
            let width = coded.xsize().max(1);
            let mut source = vec![0.0f32; width];
            let mut levels = vec![0i32; n];
            for (y, target) in levels.chunks_mut(width).enumerate() {
                for (x, value) in source[..target.len()].iter_mut().enumerate() {
                    *value = plane.get_i32(y * width + x) as f32;
                }
                (ctx.quantize_dc_i32)(&source[..target.len()], scale, target);
            }
            vec![levels]
        }
        Some(plane) => vec![(0..n).map(|i| plane.get_i32(i)).collect()],
        None => Vec::new(),
    };
    Some(crate::lossless::LmPlainInput {
        color,
        extra: extra
            .into_iter()
            .map(|v| (v, ctx.extra_squeeze_distance))
            .collect(),
        max,
        coding: ctx.coding,
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn plain_modular_lattice_matches_scalar_rounding() {
        use crate::coding::CodingTransform;
        use crate::encode_image::AlphaPlane;
        use crate::encoding_context::EncodingContext;
        use crate::image::Image3F;

        for width in [1, 3, 4, 7, 8, 9, 17] {
            let mut coded = Image3F::new(width, 3);
            for c in 0..3 {
                for (i, value) in coded.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                    *value = (i as f32 - 8.5 + c as f32 * 0.25) / super::LM_PLAIN_MAX;
                }
            }
            for coding in [CodingTransform::Rgb, CodingTransform::YCbCr] {
                let mut ctx = EncodingContext::default();
                ctx.coding = coding;
                ctx.extra_squeeze_distance = 1.0;
                for alpha in [
                    AlphaPlane::U8((0..width * 3).map(|i| (i * 37) as u8).collect()),
                    AlphaPlane::U16 {
                        data: (0..width * 3).map(|i| (i * 977) as u16).collect(),
                        bits: 16,
                    },
                ] {
                    for gray in [false, true] {
                        let got = super::lm_plain_input(&ctx, &coded, gray, Some(&alpha)).unwrap();
                        let channels: &[usize] = if gray { &[1] } else { &[0, 1, 2] };
                        for (levels, &c) in got.color.iter().zip(channels) {
                            let offset = if gray && coding == CodingTransform::YCbCr {
                                128.0 / 255.0
                            } else {
                                0.0
                            };
                            let want: Vec<i32> = coded.plane_data(c)[..width * 3]
                                .iter()
                                .map(|&value| {
                                    ((value + offset) * super::LM_PLAIN_MAX).round() as i32
                                })
                                .collect();
                            assert_eq!(*levels, want);
                        }
                        let scale = super::LM_PLAIN_MAX / ((1u32 << alpha.bits()) - 1) as f32;
                        let want: Vec<i32> = (0..width * 3)
                            .map(|i| (alpha.get_i32(i) as f32 * scale).round() as i32)
                            .collect();
                        assert_eq!(got.extra[0].0, want);
                    }
                }
            }
        }
    }
}
