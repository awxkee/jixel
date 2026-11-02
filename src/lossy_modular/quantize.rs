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

//! Stage 2b: the quantization step of every squeezed channel.
//!
//! Steps follow cjxl's per-squeeze-level tables scaled by the arm's distance:
//! one row per stored color on the XYB lattice, the luma row scaled to the
//! sample range on the plain layouts, and the VarDCT DC step on a DC frame.
//! A channel is stored divided by its step; the tree leaf multiplier
//! restores the scale in the decoder.

use super::layout::XYB_SCALE;
use crate::fast_divide::selected_rounded_quotients_fn;

/// cjxl's per-squeeze-level quantization steps for XYB modular, indexed by the
/// stored-channel color [Y, X, B-Y] and the (capped) squeeze depth. Values are
/// integer lattice steps relative to the `XYB_SCALE` scaling.
pub(super) static SQUEEZE_XYB_QTABLE: [[f32; 16]; 3] = [
    [
        163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02,
        0.01, 0.005,
    ], // Y
    [
        1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5,
    ], // X
    [
        2048.0, 1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5,
        0.5,
    ], // B-Y
];

pub(super) static SQUEEZE_QUALITY_FACTOR_XYB: f32 = 4.8;

/// Per-color multipliers on the cjxl base rows (stored order [Y, X, B-Y]),
/// Optuna study lossy_modular_v3 2026-09-02 (129 trials, crops+Kodak,
/// holdout-validated): B-Y much finer — coarse B-Y measurably desaturates
/// yellows (mean +2/255 blue shift) and SS2 punishes it; X detail is cheap to
/// discard; Y slightly coarser pays for the chroma.
static ROW_MUL: [f32; 3] = [1.265, 1.704, 0.546];

/// Further multipliers on screen content ([`crate::frame::is_screen_content`]):
/// colored text and UI keep their X and B edges. 11 screenshots at Slow,
/// SS2 / BA / cvvdp BD -2.29 / -9.40 / -2.63 %, no image worse on two
/// metrics; 11 held-out illustrations -2.63 / -0.43 / -2.25. Photos keep
/// `ROW_MUL`: there finer chroma only trades SS2 for BA and cvvdp.
pub(super) static SCREEN_ROW_MUL: [f32; 3] = [1.0, 0.5, 0.85];

/// DC-frame residual step at the finest squeeze level, in VarDCT DC steps,
/// per stored color [Y, X, B-Y].
static DC_FRAME_GAIN: [f32; 3] = [1.5, 2.5, 1.5];

/// DC-frame step ratio from one squeeze level to the next coarser one.
const DC_FRAME_LEVEL_RATIO: f32 = 0.6;

/// Table depth of a channel squeezed `hshift` + `vshift` times.
fn table_depth(hshift: i32, vshift: i32) -> usize {
    let mut shift = (hshift + vshift).min(16);
    if shift > 0 {
        shift -= 1;
    }
    shift as usize
}

/// XYB-layout step of one squeezed channel (cjxl's schedule, including the
/// float->int truncation), times `row_mul` per stored color.
pub(super) fn xyb_step(
    color: usize,
    hshift: i32,
    vshift: i32,
    distance: f32,
    row_mul: [f32; 3],
) -> i32 {
    let quantizer = 0.25f32 * distance;
    let q = quantizer
        * SQUEEZE_QUALITY_FACTOR_XYB
        * SQUEEZE_XYB_QTABLE[color][table_depth(hshift, vshift)]
        * ROW_MUL[color]
        * row_mul[color];
    (q as i32).max(1)
}

/// DC-frame step: `unit` is the VarDCT DC step of each stored color in
/// lattice units, finer by `DC_FRAME_LEVEL_RATIO` per coarser level.
pub(super) fn dc_frame_step(unit: [f32; 3], color: usize, hshift: i32, vshift: i32) -> i32 {
    let level = (hshift + vshift - 1).max(0);
    let q = unit[color] * DC_FRAME_GAIN[color] * DC_FRAME_LEVEL_RATIO.powi(level);
    (q.round() as i32).max(1)
}

/// Quantization role of one channel of a plain-layout pyramid.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Role {
    Luma,
    Chroma,
    /// Extra channel at its own (luma-row) distance; 0 = lossless.
    Extra(f32),
}

/// Chroma step relative to luma in the plain layouts: Cb/Cr, and the YCoCg
/// differences of RGB (twice their range). Slow Kodak sweeps (01/02/05/20,
/// d 1-4): YCbCr flat over 2-3, RGB flat over 3-6.
fn plain_chroma_mul(coding: crate::coding::CodingTransform) -> f32 {
    match coding {
        crate::coding::CodingTransform::Rgb => 4.0,
        _ => 2.5,
    }
}

/// Plain-layout step of a channel in its `role`, in units of `max`.
pub(super) fn plain_step(
    role: Role,
    distance: f32,
    hshift: i32,
    vshift: i32,
    max: f32,
    coding: crate::coding::CodingTransform,
) -> i32 {
    match role {
        Role::Luma => luma_row_step(distance, hshift, vshift, max, 1.0),
        Role::Chroma => luma_row_step(distance, hshift, vshift, max, plain_chroma_mul(coding)),
        Role::Extra(d) if d > 0.0 => luma_row_step(d, hshift, vshift, max, 1.0),
        Role::Extra(_) => 1,
    }
}

/// Luma-row quantization step of a channel in bit-depth units: the arm's
/// schedule scaled from the XYB luma lattice to `max`.
fn luma_row_step(distance: f32, hshift: i32, vshift: i32, max: f32, mul: f32) -> i32 {
    let unit = max / XYB_SCALE[1];
    let q = 0.25
        * distance
        * SQUEEZE_QUALITY_FACTOR_XYB
        * SQUEEZE_XYB_QTABLE[0][table_depth(hshift, vshift)];
    ((q * unit * mul) as i32).max(1)
}

/// Quantize every sample to its nearest multiple of `q` (ties away from zero)
/// and store the DIVIDED value. The channel is then coded at 1/q scale and the
/// signaled tree-leaf multiplier `q` restores the scale in the decoder.
pub(super) fn quantize_channel_divide(ch: &mut crate::squeeze::Channel, q: i32) {
    if q <= 1 {
        return;
    }
    // A precomputed divider on the dispatched SIMD kernel: no hardware
    // division in the loop.
    unsafe { selected_rounded_quotients_fn()(&mut ch.data, q) };
}
