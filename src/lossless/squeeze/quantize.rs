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

//! Lossy modular quantization, layouts, and distance calibration.

use super::frame::{
    write_frame_header_modular_plain, write_frame_header_modular_xyb_regular,
    write_lossy_modular_dc_quant,
};
use super::{write_modular_transforms_rct_squeeze, write_modular_transforms_squeeze_only};
use crate::bit_writer::BitWriter;

// ===================== Lossy modular (quantized Squeeze) =====================

/// cjxl's per-squeeze-level quantization steps for XYB modular, indexed by the
/// stored-channel color [Y, X, B-Y] and the (capped) squeeze depth. Values are
/// integer lattice steps relative to the `LOSSY_MODULAR_XYB_SCALE` scaling.
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
static LM_ROW_MUL: [f32; 3] = [1.265, 1.704, 0.546];

/// XYB float -> integer scale factors in X, Y, B order. Signaled to the decoder
/// as custom dc_quant steps (their reciprocals); powers of two stay F16-exact.
pub(super) static LOSSY_MODULAR_XYB_SCALE: [f32; 3] = [65536.0, 4096.0, 4096.0];

/// Quantization step for one squeezed channel (cjxl's schedule, including the
/// float->int truncation).
fn lossy_modular_q(color: usize, hshift: i32, vshift: i32, distance: f32) -> i32 {
    let mut shift = (hshift + vshift).min(16);
    if shift > 0 {
        shift -= 1;
    }
    let quantizer = 0.25f32 * distance;
    let q = quantizer
        * SQUEEZE_QUALITY_FACTOR_XYB
        * SQUEEZE_XYB_QTABLE[color][shift as usize]
        * LM_ROW_MUL[color];
    (q as i32).max(1)
}

/// Decoder predictors that are exactly scale-equivariant over an exact
/// division (safe under a leaf multiplier): Zero, Left, Top, Select, Gradient
/// and the plain neighbor predictors. Weighted and the averages are not.
/// Map the caller's distance to the modular arm's quantizer distance so both
/// arms land on comparable SS2 at the same `d` (fitted on the 8-image Kodak
/// pilot; the byte-only auto gate depends on this). Piecewise linear through
/// the measured knots, flat past the last one.
pub(crate) fn lm_calibrated_distance(d: f32) -> f32 {
    static KNOTS: [(f32, f32); 6] = [
        (1.0, 1.324),
        (1.5, 1.851),
        (2.0, 2.259),
        (3.0, 2.466),
        (4.0, 2.504),
        (5.0, 2.350),
    ];
    let knots = KNOTS;
    let k = if d <= knots[0].0 {
        knots[0].1
    } else if d >= knots[knots.len() - 1].0 {
        knots[knots.len() - 1].1
    } else {
        let mut k = knots[0].1;
        for w in knots.array_windows::<2>() {
            let [(d0, k0), (d1, k1)] = [w[0], w[1]];
            if d >= d0 && d <= d1 {
                k = k0 + (k1 - k0) * (d - d0) / (d1 - d0);
                break;
            }
        }
        k
    };
    d * k
}

/// Quantize every sample to its nearest multiple of `q` (ties away from zero)
/// and store the DIVIDED value. The channel is then coded at 1/q scale and the
/// signaled tree-leaf multiplier `q` restores the scale in the decoder.
pub(super) fn quantize_channel_divide(ch: &mut crate::squeeze::Channel, q: i32) {
    if q <= 1 {
        return;
    }
    for v in ch.data.iter_mut() {
        let a = *v;
        *v = if a < 0 {
            -((-a + q / 2) / q)
        } else {
            (a + q / 2) / q
        };
    }
}

/// Decoder-side edge-preserving filter of the lossy-modular frame
/// (iterations, `epf_sigma_for_modular`), sigma linear in the arm's
/// effective distance.
fn lm_epf_schedule(layout: LmLayout, effective_distance: f32) -> (u32, f32) {
    let eff = effective_distance;
    match layout {
        LmLayout::Xyb => (if eff < 5.0 { 1 } else { 2 }, 0.69 + 0.25 * eff),
        LmLayout::Plain { rct: true, .. } => (2, 1.5 + 0.29 * eff),
        LmLayout::Plain { .. } => (if eff < 10.0 { 1 } else { 2 }, 1.6 + 0.055 * eff),
    }
}

/// How the arm's modular image maps onto the frame: the XYB lattice, or the
/// image's own integer samples (a non-XYB frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LmLayout {
    /// [Y, X, B-Y] on the `LOSSY_MODULAR_XYB_SCALE` lattice, signaled
    /// through custom dc_quant steps.
    Xyb,
    /// The samples in bit-depth units; the decoder divides by the sample
    /// range. `ycbcr` sets the frame's do_YCbCr bit, `rct` codes the three
    /// color channels through the YCoCg RCT, `extra` counts the extra
    /// channels appended after the color ones.
    Plain {
        ycbcr: bool,
        rct: bool,
        extra: usize,
    },
}

impl LmLayout {
    pub(super) fn write_frame_header(self, epf: (u32, f32), w: &mut BitWriter) {
        match self {
            Self::Xyb => write_frame_header_modular_xyb_regular(epf, w),
            Self::Plain { ycbcr, extra, .. } => {
                write_frame_header_modular_plain(ycbcr, extra, epf, w);
            }
        }
    }

    /// The LfGlobal dc_quant field.
    pub(super) fn write_dc_quant(self, w: &mut BitWriter) {
        match self {
            Self::Xyb => write_lossy_modular_dc_quant(w),
            // Unused by the decoder outside XYB.
            Self::Plain { .. } => w.write(1, 1),
        }
    }

    pub(super) fn write_transforms(self, steps: &[crate::squeeze::SqueezeStep], w: &mut BitWriter) {
        match self {
            Self::Plain { rct: true, .. } => write_modular_transforms_rct_squeeze(steps, w),
            _ => write_modular_transforms_squeeze_only(steps, w),
        }
    }
}

/// The lossy-modular arm's quantized state: divided channels, per-channel
/// steps, and the squeeze script that produced them.
pub(crate) struct LmQuantized {
    pub(super) channels: Vec<crate::squeeze::Channel>,
    pub(super) quants: Vec<u32>,
    pub(super) steps: Vec<crate::squeeze::SqueezeStep>,
    pub(super) layout: LmLayout,
    /// Decoder EPF (iterations, modular sigma) for the frame header.
    pub(super) epf: (u32, f32),
    /// Code the pyramid as the 1:8 DC frame of a VarDCT frame.
    pub(super) dc_frame: bool,
    pub(super) xsize: usize,
    pub(super) ysize: usize,
}

impl LmQuantized {
    /// Fold one coarse step using the integers reconstructed by the decoder.
    /// Retained bands keep their original multipliers; the merged averages
    /// become full-scale integer channels. Layout, EPF and DC scaling stay put.
    pub(super) fn fold_last(&mut self) -> bool {
        if self.steps.len() <= 1 {
            return false;
        }
        let step = *self.steps.last().unwrap();
        // Keep the chroma pre-squeeze and at least one explicit step. The
        // shared coarse suffix always splits all original channels in place.
        if !step.in_place || step.begin_c != 0 {
            return false;
        }
        let offset = step.num_c;
        if offset == 0 || offset > self.channels.len() / 2 {
            return false;
        }
        for c in 0..step.num_c {
            let avg = &self.channels[c];
            let res = &self.channels[offset + c];
            let dims = if step.horizontal {
                avg.w.checked_add(res.w).map(|w| (w, avg.h))
            } else {
                avg.h.checked_add(res.h).map(|h| (avg.w, h))
            };
            // The current global forest requires global channels. Keeping
            // every changed average global also leaves all group crops stable.
            if dims.is_none_or(|(w, h)| w > super::GROUP_DIM || h > super::GROUP_DIM) {
                return false;
            }
        }
        if !crate::squeeze::fold_quantized_step(&mut self.channels, &mut self.quants, &step) {
            return false;
        }
        self.steps.pop();
        true
    }
}

/// Quantization role of one channel of a plain-layout pyramid.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum LmRole {
    Luma,
    Chroma,
    /// Extra channel at its own (luma-row) distance; 0 = lossless.
    Extra(f32),
}

/// Chroma step relative to luma in the plain layouts: Cb/Cr, and the YCoCg
/// differences of RGB (twice their range). Slow Kodak sweeps (01/02/05/20,
/// d 1-4): YCbCr flat over 2-3, RGB flat over 3-6.
fn lm_plain_chroma_mul(coding: crate::coding::CodingTransform) -> f32 {
    match coding {
        crate::coding::CodingTransform::Rgb => 4.0,
        _ => 2.5,
    }
}

/// Map the caller's distance to a plain-layout quantizer distance so the arm
/// lands on the VarDCT arm's SSIMULACRA2 at the same `d` (same purpose as
/// [`lm_calibrated_distance`]; fitted at Slow with the chroma multipliers
/// above and the frame's EPF). Piecewise linear, flat past the ends.
pub(crate) fn lm_plain_distance(coding: crate::coding::CodingTransform, d: f32) -> f32 {
    let knots: &[(f32, f32)] = match coding {
        crate::coding::CodingTransform::Rgb => &[
            (1.0, 1.527),
            (1.5, 2.099),
            (2.0, 2.599),
            (3.0, 2.701),
            (4.0, 2.745),
            (5.0, 2.641),
        ],
        _ => &[
            (1.0, 2.383),
            (1.5, 3.324),
            (2.0, 3.905),
            (3.0, 3.952),
            (4.0, 3.895),
            (5.0, 3.766),
        ],
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

/// Luma-row quantization step of a channel in bit-depth units: the arm's
/// schedule scaled from the XYB luma lattice to `max`.
fn lossy_modular_q_plain(distance: f32, hshift: i32, vshift: i32, max: f32, mul: f32) -> i32 {
    let mut shift = (hshift + vshift).min(16);
    if shift > 0 {
        shift -= 1;
    }
    let unit = max / LOSSY_MODULAR_XYB_SCALE[1];
    let q = 0.25 * distance * SQUEEZE_QUALITY_FACTOR_XYB * SQUEEZE_XYB_QTABLE[0][shift as usize];
    ((q * unit * mul) as i32).max(1)
}

/// The non-XYB frame's integer planes for the arm: `color` holds one (gray)
/// or three channels in the coded plane order, `extra` the extra channels
/// with their squeeze distances (0 = lossless), all in units of `max`.
pub(crate) struct LmPlainInput {
    pub color: Vec<Vec<i32>>,
    pub extra: Vec<(Vec<i32>, f32)>,
    pub max: f32,
    pub coding: crate::coding::CodingTransform,
}

/// Build and quantize a plain-layout pyramid at `effective_distance`
/// (already calibrated). None when the image is too small to squeeze.
pub(crate) fn lm_quantize_plain(
    input: &LmPlainInput,
    xsize: usize,
    ysize: usize,
    effective_distance: f32,
) -> Option<LmQuantized> {
    use crate::coding::CodingTransform;
    use crate::squeeze::{
        Channel, apply_step_forward, default_squeeze_steps, lossy_squeeze_steps_chroma,
    };
    let num_color = input.color.len();
    debug_assert!(num_color == 1 || num_color == 3);
    let num_c = num_color + input.extra.len();
    if default_squeeze_steps(xsize, ysize, num_c).is_empty() {
        return None;
    }
    let n = xsize * ysize;
    let mut planes: Vec<Vec<i32>> = Vec::with_capacity(num_c);
    let mut roles: Vec<LmRole> = Vec::with_capacity(num_c);
    let (ycbcr, rct, chroma): (bool, bool, Vec<usize>) = match (num_color, input.coding) {
        (3, CodingTransform::YCbCr) => (true, false, vec![0, 2]),
        (3, _) => (false, true, vec![1, 2]),
        _ => (false, false, Vec::new()),
    };
    if rct {
        let [r, g, b] = [&input.color[0], &input.color[1], &input.color[2]];
        let mut y = vec![0i32; n];
        let mut co = vec![0i32; n];
        let mut cg = vec![0i32; n];
        for i in 0..n {
            (y[i], co[i], cg[i]) = crate::lossless::forward_ycocg(r[i], g[i], b[i]);
        }
        planes.extend([y, co, cg]);
    } else {
        planes.extend(input.color.iter().map(|p| p[..n].to_vec()));
    }
    roles.extend((0..num_color).map(|c| {
        if chroma.contains(&c) {
            LmRole::Chroma
        } else {
            LmRole::Luma
        }
    }));
    for (plane, distance) in &input.extra {
        planes.push(plane[..n].to_vec());
        roles.push(LmRole::Extra(*distance));
    }
    let steps = lossy_squeeze_steps_chroma(xsize, ysize, num_c, &chroma);
    let n_pre: usize = steps.iter().filter(|s| !s.in_place).map(|s| s.num_c).sum();
    let mut channels: Vec<Channel> = planes
        .into_iter()
        .map(|data| Channel {
            data,
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        })
        .collect();
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();
    let mut quants: Vec<u32> = Vec::with_capacity(nb);
    for (i, ch) in channels.iter_mut().enumerate() {
        let role = if i >= nb - n_pre {
            LmRole::Chroma
        } else {
            roles[i % num_c]
        };
        let q = match role {
            LmRole::Luma => {
                lossy_modular_q_plain(effective_distance, ch.hshift, ch.vshift, input.max, 1.0)
            }
            LmRole::Chroma => lossy_modular_q_plain(
                effective_distance,
                ch.hshift,
                ch.vshift,
                input.max,
                lm_plain_chroma_mul(input.coding),
            ),
            LmRole::Extra(d) if d > 0.0 => {
                lossy_modular_q_plain(d, ch.hshift, ch.vshift, input.max, 1.0)
            }
            LmRole::Extra(_) => 1,
        };
        quantize_channel_divide(ch, q);
        quants.push(q as u32);
    }
    let layout = LmLayout::Plain {
        ycbcr,
        rct,
        extra: input.extra.len(),
    };
    Some(LmQuantized {
        channels,
        quants,
        steps,
        layout,
        epf: lm_epf_schedule(layout, effective_distance),
        dc_frame: false,
        xsize,
        ysize,
    })
}

/// Build and quantize the arm's channel pyramid at `effective_distance`
/// (already calibrated by the caller). None when the image is too small.
pub(crate) fn lm_quantize(
    xyb: &crate::image::Image3F,
    effective_distance: f32,
) -> Option<LmQuantized> {
    let steps = crate::squeeze::lossy_squeeze_steps(xyb.xsize(), xyb.ysize(), 3);
    lm_quantize_xyb(xyb, steps, |color, hshift, vshift| {
        lossy_modular_q(color, hshift, vshift, effective_distance)
    })
    .map(|quantized| LmQuantized {
        epf: lm_epf_schedule(LmLayout::Xyb, effective_distance),
        ..quantized
    })
}

/// [`lm_quantize`] over the squeeze script `steps` (optionally led by the
/// chroma pre-squeeze), each channel at `q_of(stored color [Y, X, B-Y],
/// hshift, vshift)`.
pub(super) fn lm_quantize_xyb(
    xyb: &crate::image::Image3F,
    steps: Vec<crate::squeeze::SqueezeStep>,
    q_of: impl Fn(usize, i32, i32) -> i32,
) -> Option<LmQuantized> {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
    let xsize = xyb.xsize();
    let ysize = xyb.ysize();
    let num_c = 3usize;
    if default_squeeze_steps(xsize, ysize, num_c).is_empty() {
        // Tiny image: no squeeze means "lossy" would be plain color
        // quantization at the wrong scale. Not this arm's job.
        return None;
    }
    // With the chroma pre-squeeze, the last four channels are its residuals
    // (mapped to the chroma quantization row, like cjxl); everything before
    // them stays aligned to `num_c` blocks.
    let chroma_pre = steps.len() >= 2 && !steps[0].in_place;

    // XYB floats onto the integer lattice, stored as [Y, X, B-Y].
    let n = xsize * ysize;
    let input = [
        &xyb.plane_data(0)[..n],
        &xyb.plane_data(1)[..n],
        &xyb.plane_data(2)[..n],
    ];
    let mut planes = [vec![0i32; n], vec![0i32; n], vec![0i32; n]];
    {
        let [dst_y, dst_x, dst_b] = &mut planes;
        let scales = [
            LOSSY_MODULAR_XYB_SCALE[0],
            LOSSY_MODULAR_XYB_SCALE[1],
            LOSSY_MODULAR_XYB_SCALE[2],
        ];
        unsafe {
            crate::xyb::selected_quantize_xyb_channels_fn()(input, [dst_y, dst_x, dst_b], scales);
        }
    }
    let mut channels: Vec<Channel> = planes
        .into_iter()
        .map(|data| Channel {
            data,
            w: xsize,
            h: ysize,
            hshift: 0,
            vshift: 0,
        })
        .collect();
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    // Quantize every channel on its (color, depth) schedule. Channels are
    // stored divided by their step; the tree leaf multiplier restores the
    // scale. The decoder's full-scale predictions stay consistent because
    // every predictor used on a q>1 channel is exactly scale-equivariant over
    // the divided values.
    let nb = channels.len();
    let color_of = |i: usize| {
        if chroma_pre && i >= nb - 4 {
            1 // trailing chroma pre-squeeze residuals -> chroma row
        } else {
            i % num_c
        }
    };
    let mut quants: Vec<u32> = Vec::with_capacity(nb);
    for (i, ch) in channels.iter_mut().enumerate() {
        let q = q_of(color_of(i), ch.hshift, ch.vshift);
        quantize_channel_divide(ch, q);
        quants.push(q as u32);
    }
    Some(LmQuantized {
        channels,
        quants,
        steps,
        layout: LmLayout::Xyb,
        epf: (0, 0.0),
        dc_frame: false,
        xsize,
        ysize,
    })
}
