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

//! Stage 1: the arm's quantizer distance.
//!
//! The arm quantizes on cjxl's squeeze schedule, which does not land on
//! VarDCT's quality at the same distance. These piecewise-linear factors,
//! fitted per layout, map the caller's `d` to the distance that does; the
//! byte-only Auto gate depends on them.

use crate::coding::CodingTransform;

/// XYB layout factors:
static XYB_KNOTS: [(f32, f32); 7] = [
    (1.0, 1.247),
    (1.25, 1.311),
    (1.5, 1.546),
    (1.75, 1.902),
    (2.0, 2.211),
    (3.0, 2.533),
    (4.5, 2.570),
];

/// XYB layout factors on screen content ([`crate::frame::is_screen_content`],
/// coded on [`super::quantize::SCREEN_ROW_MUL`]): the same fit over 22
/// screenshots and illustrations, finer than photos at d 1 and coarser from
/// d 2.
static XYB_SCREEN_KNOTS: [(f32, f32); 7] = [
    (1.0, 1.153),
    (1.25, 1.340),
    (1.5, 1.684),
    (1.75, 2.029),
    (2.0, 2.479),
    (3.0, 2.625),
    (4.5, 2.923),
];

/// Plain-layout factors of RGB frames (coded through YCoCg).
static RGB_KNOTS: [(f32, f32); 6] = [
    (1.0, 1.527),
    (1.5, 2.099),
    (2.0, 2.599),
    (3.0, 2.701),
    (4.0, 2.745),
    (5.0, 2.641),
];

/// Plain-layout factors of YCbCr (and gray) frames.
static YCBCR_KNOTS: [(f32, f32); 6] = [
    (1.0, 2.383),
    (1.5, 3.324),
    (2.0, 3.905),
    (3.0, 3.952),
    (4.0, 3.895),
    (5.0, 3.766),
];

/// The XYB layout's quantizer distance for the caller's `d`, on `screen`
/// content or not.
pub(crate) fn calibrated_distance(d: f32, screen: bool) -> f32 {
    let knots: &[(f32, f32)] = if screen {
        &XYB_SCREEN_KNOTS
    } else {
        &XYB_KNOTS
    };
    d * interpolate_knots(knots, d)
}

/// The plain layout's quantizer distance for the caller's `d`, fitted at
/// Slow with the chroma multipliers of [`super::quantize`] and the frame's
/// EPF.
pub(crate) fn plain_distance(coding: CodingTransform, d: f32) -> f32 {
    let knots: &[(f32, f32)] = match coding {
        CodingTransform::Rgb => &RGB_KNOTS,
        _ => &YCBCR_KNOTS,
    };
    d * interpolate_knots(knots, d)
}

/// Piecewise-linear lookup through `knots` (sorted by x), flat outside.
pub(super) fn interpolate_knots(knots: &[(f32, f32)], x: f32) -> f32 {
    let (first, last) = (knots[0], knots[knots.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    for w in knots.array_windows::<2>() {
        let [(x0, y0), (x1, y1)] = [w[0], w[1]];
        if x >= x0 && x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    last.1
}
