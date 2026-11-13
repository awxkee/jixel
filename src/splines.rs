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

//! JPEG XL splines for the lossy VarDCT path (experimental, opt-in through
//! `EncodeConfig::with_splines`).
//!
//! Thin curvilinear structures are detected on the XYB luma plane, fitted with
//! the format's spline model, judged against the encoder's own rate model and,
//! when they pay, subtracted from the image before VarDCT. The decoder adds
//! them back after gaborish, EPF and patches, so what they carry never meets
//! the DCT quantizer or the smoothing filters.

mod detect;
pub(crate) mod dots;
mod edge_detect;
mod edge_fit;
pub(crate) mod edge_model;
mod edge_select;
mod extend;
mod filter;
mod fit;
#[cfg(target_arch = "aarch64")]
#[path = "neon/spline_fit.rs"]
mod fit_neon;
mod geometry;
mod lines;
#[cfg(target_arch = "aarch64")]
#[path = "neon/splines.rs"]
mod neon;
mod render;
mod select;
mod transform;
mod trig;

#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", feature = "avx")))]
pub(crate) use detect::{ALONG_PENALTY, MAX_SUBPIXEL_OFFSET, RidgeRow, ridge_row_scalar};
#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", feature = "avx")))]
pub(crate) use filter::{FilterBounds, vertical_row_scalar_tail};
pub(crate) use fit::FitScratch;
#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", feature = "avx")))]
pub(crate) use geometry::Segments;
#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", feature = "avx")))]
pub(crate) use render::Sample;

use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::entropy::{
    Token, optimize_entropy_code_ac, pack_signed, write_ans_tokens, write_entropy_code, write_token,
};
use crate::image::Image3F;
use crate::path_order;
use crate::simd::scaled_add::{AddScaledFn, selected_add_scaled_fn};
use crate::util::fast_hypot;
use std::sync::OnceLock;

pub(crate) type FitMatrix = [[f64; fit::MAX_COEFFS]; fit::MAX_COEFFS];
pub(crate) type FitTargets = [[f64; fit::MAX_COEFFS]; 3];

/// Spline kernels selected once for the process and shared by every encoder.
pub(crate) struct SplinesContext {
    ridge_row: detect::RidgeRowFn,
    segment_distance: geometry::SegmentDistanceFn,
    render_row: render::RenderRowFn,
    continuous_idct: transform::ContinuousIdctFn,
    /// Fitting and dot templates use a fused erf polynomial on SIMD targets.
    blob_weight_row: fn(BlobWeightRow, i32, &mut [f32]),
    add_scaled_f32: AddScaledFn,
    add_scaled_fma_f32: AddScaledFn,
    filter_vertical_row: filter::FilterVerticalRowFn,
    fit_add_scaled: fn(&mut [f64], &[f64], f64),
    fit_accumulate: fn(&mut FitMatrix, &mut FitTargets, &[f64], [f64; 3]),
    fit_sub_dot: fn(f64, &[f64], &[f64]) -> f64,
    cos: fn(f32) -> f32,
    sin_cos: fn(f32) -> (f32, f32),
    atan2: fn(f32, f32) -> f32,
}

impl SplinesContext {
    pub(crate) fn get() -> &'static Self {
        static KERNELS: OnceLock<SplinesContext> = OnceLock::new();
        KERNELS.get_or_init(Self::resolve)
    }

    fn resolve() -> Self {
        let add_scaled_f32 = selected_add_scaled_fn();
        let kernels = Self {
            ridge_row: detect::ridge_row_scalar,
            segment_distance: geometry::segment_distance_scalar,
            render_row: render::Sample::scalar_row,
            continuous_idct: transform::continuous_idct_scalar,
            blob_weight_row: BlobWeightRow::fill_scalar,
            add_scaled_f32,
            add_scaled_fma_f32: add_scaled_f32,
            filter_vertical_row: filter::vertical_row_scalar,
            fit_add_scaled: fit::add_scaled_scalar,
            fit_accumulate: fit::accumulate_fit_scalar,
            fit_sub_dot: fit::sub_dot_scalar,
            cos: trig::cos_scalar,
            sin_cos: trig::sin_cos_scalar,
            atan2: trig::atan2_scalar,
        };

        #[cfg(target_arch = "aarch64")]
        let kernels = Self {
            // SAFETY: NEON is part of the supported AArch64 target baseline.
            ridge_row: |s, derivatives, out| unsafe {
                neon::spline_ridge_row_neon(s, derivatives, out)
            },
            segment_distance: |p, blocks| unsafe { neon::spline_distance_neon(p, blocks) },
            render_row: |sample, dy, x, rows, amp| unsafe {
                neon::spline_render_row_neon(sample, dy, x, rows, amp)
            },
            continuous_idct: |dct, t| unsafe { neon::continuous_idct_neon(dct, t) },
            blob_weight_row: |row, x, out| unsafe { neon::blob_weight_row_neon(row, x, out) },
            add_scaled_fma_f32: |dst, src, scale| unsafe {
                crate::simd::neon_add_scaled::add_scaled_fma_neon(dst, src, scale)
            },
            filter_vertical_row: |src, w, bounds, taps, dst| unsafe {
                neon::vertical_row_neon(src, w, bounds, taps, dst)
            },
            fit_add_scaled: |dst, src, scale| unsafe { fit_neon::add_scaled_neon(dst, src, scale) },
            fit_accumulate: |ata, atb, row, target| unsafe {
                fit_neon::accumulate_fit_neon(ata, atb, row, target)
            },
            fit_sub_dot: |acc, left, right| unsafe { fit_neon::sub_dot_neon(acc, left, right) },
            ..kernels
        };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let avx_fma = std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("fma");
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let kernels = if avx_fma {
            // SAFETY: these callbacks require AVX and FMA.
            Self {
                cos: |x| unsafe { trig::cos_fma(x) },
                sin_cos: |x| unsafe { trig::sin_cos_fma(x) },
                atan2: |y, x| unsafe { trig::atan2_fma(y, x) },
                ..kernels
            }
        } else {
            kernels
        };

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        let kernels = if avx2 {
            Self {
                // SAFETY: the vertical kernel only requires AVX2.
                filter_vertical_row: |src, w, bounds, taps, dst| unsafe {
                    crate::avx::vertical_row_avx2(src, w, bounds, taps, dst)
                },
                ..kernels
            }
        } else {
            kernels
        };
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if avx_fma && avx2 {
            use crate::avx;
            // SAFETY: these callbacks require AVX2 and FMA.
            return Self {
                ridge_row: |s, derivatives, out| unsafe {
                    avx::spline_ridge_row_avx2(s, derivatives, out)
                },
                segment_distance: |p, blocks| unsafe { avx::spline_distance_avx2(p, blocks) },
                render_row: |sample, dy, x, rows, amp| unsafe {
                    avx::spline_render_row_avx2(sample, dy, x, rows, amp)
                },
                continuous_idct: |dct, t| unsafe { avx::continuous_idct_avx2(dct, t) },
                blob_weight_row: |row, x, out| unsafe { avx::blob_weight_row_avx2(row, x, out) },
                add_scaled_fma_f32: |dst, src, scale| unsafe {
                    crate::simd::avx::add_scaled_fma_avx2(dst, src, scale)
                },
                fit_add_scaled: |dst, src, scale| unsafe {
                    avx::add_scaled_f64_avx2(dst, src, scale)
                },
                fit_accumulate: |ata, atb, row, target| unsafe {
                    avx::accumulate_fit_avx2(ata, atb, row, target)
                },
                fit_sub_dot: |acc, left, right| unsafe { avx::sub_dot_avx2(acc, left, right) },
                ..kernels
            };
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                return Self {
                    continuous_idct: |dct, t| unsafe { crate::sse::continuous_idct_sse41(dct, t) },
                    ..kernels
                };
            }
            if std::arch::is_x86_feature_detected!("sse2") {
                return Self {
                    continuous_idct: |dct, t| unsafe { crate::sse::continuous_idct_sse2(dct, t) },
                    ..kernels
                };
            }
        }

        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        let kernels = Self {
            continuous_idct: crate::wasm::continuous_idct_wasm,
            ..kernels
        };
        kernels
    }
}

/// X, Y, B, sigma.
static CHANNEL_WEIGHT: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];
const Y_TO_X: f32 = 0.0;
const Y_TO_B: f32 = 1.0;

/// Natural log of the Gaussian render cutoff `10^-5`.
/// `ln((1/10)^5) = 5 * ln(1/10) = -5 * ln(10)`.
const LN_RENDER_CUTOFF: f32 = -5.0 * std::f32::consts::LN_10;

/// Frame-global `quantization_adjustment`: the sigma lattice step is
/// `0.3333 / (1 + adjust / 8)`.
pub(crate) const QUANT_ADJUST: i32 = 4;

const CTX_QUANT_ADJUST: u32 = 0;
const CTX_START_POS: u32 = 1;
const CTX_NUM_SPLINES: u32 = 2;
const CTX_NUM_POINTS: u32 = 3;
const CTX_POINTS: u32 = 4;
const CTX_DCT: u32 = 5;
const NUM_SPLINE_CONTEXTS: usize = 6;

const MIN_DIM: usize = 64;
const MAX_DISTANCE: f32 = 16.0;
const PALETTE_LIKE_COLORS: usize = 32;

/// A spline control point or sampled position.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Point<T> {
    pub(crate) x: T,
    pub(crate) y: T,
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl Point<i32> {
    fn as_f32(self) -> Point<f32> {
        Point::new(self.x as f32, self.y as f32)
    }
}

/// Prepare ties-away rounding for a truncating float-to-integer cast.
#[inline]
fn biased_for_rounding(value: f32) -> f32 {
    // Adding half would round the float immediately below 0.5 up to 1.0;
    // at 2^23 and above, f32 values are already integral.
    if value.abs() < 0.5 || value.abs() >= 8_388_608.0 {
        value
    } else {
        value + 0.5f32.copysign(value)
    }
}

#[inline]
fn round_i32(value: f32) -> i32 {
    // Baseline x86 lowers round() to a libm call. Other targets can use a
    // native rounding conversion (for example, FCVTAS on AArch64).
    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        biased_for_rounding(value) as i32
    } else {
        value.round() as i32
    }
}

#[inline]
fn round_isize(value: f32) -> isize {
    biased_for_rounding(value) as isize
}

#[derive(Clone, Debug)]
pub(crate) struct QuantizedSpline {
    pub(crate) points: Vec<Point<i32>>,
    /// X, Y, B, sigma.
    pub(crate) dct: [[i32; 32]; 4],
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SplineSet {
    pub(crate) adjust: i32,
    pub(crate) splines: Vec<QuantizedSpline>,
    pub(crate) dot_encoding: Option<dots::DotEncoding>,
}

#[inline]
pub(crate) fn adjusted_quant(adjust: i32) -> f32 {
    if adjust >= 0 {
        1.0 + 0.125 * adjust as f32
    } else {
        1.0 / (1.0 - 0.125 * adjust as f32)
    }
}

#[inline]
pub(crate) fn fast_erf(x: f32) -> f32 {
    let a = x.abs();
    let d1 = a * 7.77394369e-02 + 2.05260015e-04;
    let d2 = d1 * a + 2.32120216e-01;
    let d3 = d2 * a + 2.77820801e-01;
    let d4 = d3 * a + 1.0;
    let d5 = d4 * d4;
    let inv = 1.0 / d5;
    let r = 1.0 - inv * inv;
    if x <= 0.0 { -r } else { r }
}

pub(crate) fn catmull_rom(control: &[Point<f32>]) -> Vec<Point<f32>> {
    catmull_rom_spans(control, 0..control.len() - 1)
}

/// Samples only these spans, retaining the surrounding controls for their tangents.
fn catmull_rom_spans(control: &[Point<f32>], spans: std::ops::Range<usize>) -> Vec<Point<f32>> {
    if control.len() == 1 {
        return vec![control[0]];
    }
    let n = control.len();
    let mut p = Vec::with_capacity(n + 2);
    p.push(Point::new(
        control[0].x + (control[0].x - control[1].x),
        control[0].y + (control[0].y - control[1].y),
    ));
    p.extend_from_slice(control);
    p.push(Point::new(
        control[n - 1].x + (control[n - 1].x - control[n - 2].x),
        control[n - 1].y + (control[n - 1].y - control[n - 2].y),
    ));
    let mut out = Vec::with_capacity(spans.len() * 16 + 1);
    for q in p.array_windows::<4>().skip(spans.start).take(spans.len()) {
        out.push(q[1]);
        let mut d = [0f32; 3];
        let mut t = [0f32; 4];
        for k in 0..3 {
            d[k] = fast_hypot(q[k + 1].x - q[k].x, q[k + 1].y - q[k].y).sqrt();
            t[k + 1] = t[k] + d[k];
        }
        for i in 1..16 {
            let tt = d[0] + (i as f32 / 16.0) * d[1];
            let mut a = [Point::default(); 3];
            for k in 0..3 {
                let f = (tt - t[k]) / d[k];
                a[k] = Point::new(
                    q[k].x + f * (q[k + 1].x - q[k].x),
                    q[k].y + f * (q[k + 1].y - q[k].y),
                );
            }
            let mut b = [Point::default(); 2];
            for k in 0..2 {
                let f = (tt - t[k]) / (d[k] + d[k + 1]);
                b[k] = Point::new(
                    a[k].x + f * (a[k + 1].x - a[k].x),
                    a[k].y + f * (a[k + 1].y - a[k].y),
                );
            }
            let f = (tt - t[1]) / d[1];
            out.push(Point::new(
                b[0].x + f * (b[1].x - b[0].x),
                b[0].y + f * (b[1].y - b[0].y),
            ));
        }
    }
    out.push(control[spans.end]);
    out
}

/// One arc sample: position and intensity multiplier.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArcSample {
    pub(crate) position: Point<f32>,
    pub(crate) multiplier: f32,
}

fn equally_spaced(points: &[Point<f32>]) -> Vec<ArcSample> {
    let mut out = Vec::new();
    let mut current = points[0];
    out.push(ArcSample {
        position: current,
        multiplier: 1.0,
    });
    let mut next = 0usize;
    while next < points.len() {
        let mut previous = current;
        let mut from_prev = 0f32;
        loop {
            if next == points.len() {
                out.push(ArcSample {
                    position: previous,
                    multiplier: from_prev,
                });
                return out;
            }
            let nx = points[next];
            let to_next = fast_hypot(nx.x - previous.x, nx.y - previous.y);
            if from_prev + to_next >= 1.0 {
                let f = (1.0 - from_prev) / to_next;
                current = Point::new(
                    previous.x + f * (nx.x - previous.x),
                    previous.y + f * (nx.y - previous.y),
                );
                out.push(ArcSample {
                    position: current,
                    multiplier: 1.0,
                });
                break;
            }
            from_prev += to_next;
            previous = nx;
            next += 1;
        }
    }
    out
}

/// The decoder's 1 px arc-length samples of the curve through `points`, their
/// intensity multipliers, and the arc length.
pub(crate) fn sample_curve(points: &[Point<i32>]) -> (Vec<ArcSample>, f32) {
    let control: Vec<Point<f32>> = points.iter().map(|&p| p.as_f32()).collect();
    let samples = equally_spaced(&catmull_rom(&control));
    let arc = (samples.len() as f32 - 2.0) + samples.last().unwrap().multiplier;
    (samples, arc)
}

/// Per-unit-color weight of an arc sample with `amp = 0.25 * sigma * mult`.
#[inline]
pub(crate) fn blob_weight(dx: f32, dy: f32, inv_sigma: f32, amp: f32) -> f32 {
    let dist = (dx * dx + dy * dy).sqrt();
    let e = fast_erf((dist * 0.5 + 0.353_553_39) * inv_sigma)
        - fast_erf((dist * 0.5 - 0.353_553_39) * inv_sigma);
    amp * e * e
}

/// A horizontal batch with `dx = (x as f32 - center) * step`.
#[derive(Clone, Copy)]
pub(crate) struct BlobWeightRow {
    pub(crate) center: f32,
    pub(crate) step: f32,
    pub(crate) dy: f32,
    pub(crate) inv_sigma: f32,
    pub(crate) amp: f32,
}

impl BlobWeightRow {
    fn fill_scalar(self, x0: i32, weights: &mut [f32]) {
        for (x, out) in (x0..).zip(weights) {
            let dx = (x as f32 - self.center) * self.step;
            *out = blob_weight(dx, self.dy, self.inv_sigma, self.amp);
        }
    }
}

/// Half-open pixel window `(x0, x1, y0, y1)` of an arc sample with the given reach.
#[inline]
pub(crate) fn blob_window(
    cx: f32,
    cy: f32,
    reach: f32,
    w: usize,
    h: usize,
) -> (usize, usize, usize, usize) {
    let x0 = (round_i32(cx - reach).max(0) as usize).min(w);
    let x1 = (round_i32(cx + reach) as isize + 1).clamp(0, w as isize) as usize;
    let y0 = (round_i32(cy - reach).max(0) as usize).min(h);
    let y1 = (round_i32(cy + reach) as isize + 1).clamp(0, h as isize) as usize;
    (x0, x1, y0, y1)
}

/// Inclusive pixel box `(x0, y0, x1, y1)`.
pub(crate) type PixelBox = (usize, usize, usize, usize);

/// Adds `sign` times the decoder's render of one spline; returns the touched box.
#[cfg(test)]
pub(crate) fn render_spline(
    ctx: &EncodingContext,
    sp: &QuantizedSpline,
    adjust: i32,
    xyb: &mut Image3F,
    sign: f32,
) -> Option<PixelBox> {
    let plan = render::RenderPlan::new(ctx, sp, adjust, xyb.xsize(), xyb.ysize());
    plan.draw(xyb, sign);
    plan.bounds
}

pub(crate) fn spline_tokens(sp: &QuantizedSpline) -> Vec<(u32, u32)> {
    let mut t = vec![(CTX_NUM_POINTS, (sp.points.len() - 1) as u32)];
    let Point {
        x: mut px,
        y: mut py,
    } = sp.points[0];
    let (mut pdx, mut pdy) = (0i32, 0i32);
    for &Point { x, y } in &sp.points[1..] {
        let (dx, dy) = (x - px, y - py);
        t.push((CTX_POINTS, pack_signed(dx - pdx)));
        t.push((CTX_POINTS, pack_signed(dy - pdy)));
        (pdx, pdy, px, py) = (dx, dy, x, y);
    }
    t.extend(
        sp.dct
            .as_flattened()
            .iter()
            .map(|&v| (CTX_DCT, pack_signed(v))),
    );
    t
}

/// Greedy nearest-neighbor chain over the starting points. They are coded as
/// deltas from the previous spline and rendering is additive, so order is free.
fn coding_order(splines: &[QuantizedSpline]) -> Vec<&QuantizedSpline> {
    let mut left: Vec<&QuantizedSpline> = splines.iter().collect();
    let mut order = Vec::with_capacity(left.len());
    let mut at = Point::new(0, 0);
    while !left.is_empty() {
        let (k, _) = left
            .iter()
            .enumerate()
            .min_by_key(|(_, sp)| {
                let (dx, dy) = (sp.points[0].x - at.x, sp.points[0].y - at.y);
                let bits = |v: i32| (1 + v.unsigned_abs()).ilog2();
                (bits(dx) + bits(dy), dx.abs() + dy.abs())
            })
            .unwrap();
        let sp = left.swap_remove(k);
        at = sp.points[0];
        order.push(sp);
    }
    order
}

/// Start positions priced as the section codes them (the first absolute, then
/// deltas, all in one context), ordered as the cheapest open path from `chain`.
fn priced_order<'a>(chain: &[&'a QuantizedSpline]) -> Vec<&'a QuantizedSpline> {
    let points: Vec<path_order::Pos> = chain
        .iter()
        .map(|sp| (sp.points[0].x, sp.points[0].y))
        .collect();
    let given = vec![(0..points.len()).collect()];
    let order = path_order::order_sets(&[points], &[true], given, true, None);
    order[0].iter().map(|&i| chain[i]).collect()
}

fn spline_section_tokens(set: &SplineSet, splines: &[&QuantizedSpline]) -> Vec<Token> {
    let mut tokens = Vec::new();
    tokens.push(Token::new(CTX_NUM_SPLINES, (splines.len() - 1) as u32));
    let (mut lx, mut ly) = (0i32, 0i32);
    for (i, sp) in splines.iter().enumerate() {
        let Point { x, y } = sp.points[0];
        if i == 0 {
            tokens.push(Token::new(CTX_START_POS, x as u32));
            tokens.push(Token::new(CTX_START_POS, y as u32));
        } else {
            tokens.push(Token::new(CTX_START_POS, pack_signed(x - lx)));
            tokens.push(Token::new(CTX_START_POS, pack_signed(y - ly)));
        }
        (lx, ly) = (x, y);
    }
    tokens.push(Token::new(CTX_QUANT_ADJUST, pack_signed(set.adjust)));
    for sp in splines {
        tokens.extend(spline_tokens(sp).into_iter().map(|(c, v)| Token::new(c, v)));
    }
    tokens
}

fn encode_spline_section(tokens: &[Token], scratch: &mut CoderScratch) -> BitWriter {
    let mut w = BitWriter::new();
    let code = optimize_entropy_code_ac(tokens, NUM_SPLINE_CONTEXTS, &mut scratch.huffman_pool);
    let code_ref = code.as_ref();
    w.write(1, 0); // no LZ77
    write_entropy_code(&code_ref, &mut scratch.huffman_pool, &mut w);
    if code_ref.use_prefix_code {
        for &token in tokens {
            write_token(token, &code_ref, &mut w);
        }
    } else {
        write_ans_tokens(
            tokens,
            code_ref.context_map,
            code_ref.ans_symbols,
            code_ref.ans_reverse_maps,
            code_ref.hybrid_uint_configs,
            &mut w,
        );
    }
    w
}

/// LfGlobal spline section (after the patch dictionary, before the DC scales).
pub(crate) fn write_splines(set: &SplineSet, scratch: &mut CoderScratch, w: &mut BitWriter) {
    let chain = coding_order(&set.splines);
    let mut best = encode_spline_section(&spline_section_tokens(set, &chain), scratch);
    let priced = priced_order(&chain);
    if priced
        .iter()
        .zip(&chain)
        .any(|(a, b)| !std::ptr::eq(*a, *b))
    {
        let candidate = encode_spline_section(&spline_section_tokens(set, &priced), scratch);
        if candidate.bits_written() < best.bits_written() {
            best = candidate;
        }
    }
    w.append(&best);
}

/// Add edge detail only after the native transform layout is available.
/// A rejected trial restores the source exactly and returns no replacement set.
pub(crate) use edge_select::{EdgeView, select_edges};

/// The decoder adds this luma layer after filtering the VarDCT base. Render
/// sparse 8x8 tiles to avoid allocating a second three-channel image.
pub(crate) fn additive_y(
    ctx: &EncodingContext,
    set: &SplineSet,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut layer = vec![0.0; width * height];
    let blocks_w = width.div_ceil(8);
    for spline in &set.splines {
        let plan = render::RenderPlan::new(ctx, spline, set.adjust, width, height).tiled();
        for tile in &plan.tiles {
            let mut pixels = [[0.0; 64]; 3];
            plan.draw_tile(tile, &mut pixels, 1.0);
            let x0 = tile.cell % blocks_w * 8;
            let y0 = tile.cell / blocks_w * 8;
            let n = (width - x0).min(8);
            for y in 0..(height - y0).min(8) {
                for x in 0..n {
                    layer[(y0 + y) * width + x0 + x] += pixels[1][y * 8 + x];
                }
            }
        }
    }
    layer
}

/// Few distinct colors: bilevel / palette art whose hard aliased strokes the
/// spline profile cannot match.
fn palette_like(xyb: &Image3F) -> bool {
    let mut seen: Vec<[i32; 3]> = Vec::with_capacity(PALETTE_LIKE_COLORS + 1);
    for y in (0..xyb.ysize()).step_by(3) {
        let rows = [
            xyb.plane_row(0, y),
            xyb.plane_row(1, y),
            xyb.plane_row(2, y),
        ];
        for ((&x, &y), &b) in rows[0].iter().zip(rows[1]).zip(rows[2]).step_by(3) {
            let key = [
                round_i32(x * 4096.0),
                round_i32(y * 256.0),
                round_i32(b * 256.0),
            ];
            if !seen.contains(&key) {
                seen.push(key);
                if seen.len() > PALETTE_LIKE_COLORS {
                    return false;
                }
            }
        }
    }
    true
}

/// Fitted spline candidates of one image, ready for RD selection.
pub(crate) struct SplineCandidates(Vec<fit::Candidate>, Vec<fit::LineGroup>);

impl SplineCandidates {
    /// Whether any candidate passes through a marked 8x8 block.
    pub(crate) fn touches(&self, blocks: &[bool], blocks_w: usize) -> bool {
        self.0.iter().any(|candidate| {
            candidate.alts.first().is_some_and(|alt| {
                alt.points.array_windows::<2>().any(|[a, b]| {
                    let (a, b) = (a.as_f32(), b.as_f32());
                    let steps = ((b.x - a.x).abs().max((b.y - a.y).abs()) / 4.0).ceil() as usize;
                    (0..=steps).any(|k| {
                        let t = k as f32 / steps.max(1) as f32;
                        let x = (a.x + t * (b.x - a.x)) as usize / 8;
                        let y = (a.y + t * (b.y - a.y)) as usize / 8;
                        blocks.get(y * blocks_w + x).is_some_and(|&v| v)
                    })
                })
            })
        })
    }
}

/// Detects curvilinear structures (splines) and isolated bright/dark dots in
/// `xyb` and fits candidates to them.
pub(crate) fn find_candidates(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    xyb: &Image3F,
    quant_field: &[f32],
) -> Option<SplineCandidates> {
    if xyb.xsize() < MIN_DIM || xyb.ysize() < MIN_DIM || distance > MAX_DISTANCE {
        return None;
    }
    if palette_like(xyb) {
        return None;
    }
    let mut candidates = Vec::new();
    let mut groups = Vec::new();
    if ctx.splines {
        let chains = lines::split_corners(detect::detect_chains(ctx, scratch, xyb));
        if !chains.is_empty() {
            let extensions = extend::find_extensions(ctx, scratch, xyb, &chains);
            let long_lines = lines::find_long_lines(ctx.spline_kernels, &chains);
            (candidates, groups) = fit::fit_candidates(
                ctx,
                scratch,
                distance,
                xyb,
                quant_field,
                &chains,
                &extensions,
                &long_lines,
            );
        }
    }
    if ctx.dots {
        candidates.extend(dots::dot_candidates(
            ctx.spline_kernels,
            &ctx.thread_pool,
            scratch,
            xyb,
        ));
    }
    (!candidates.is_empty()).then_some(SplineCandidates(candidates, groups))
}

/// RD-selects candidates against `image` and subtracts the accepted splines
/// from it. `quant_field` is the effective AC quant (`scale * q`) per 8x8
/// block; splines touching a `forbidden` block are skipped (patch rectangles:
/// the decoder draws splines on top of the replaced patch pixels).
pub(crate) fn select_splines(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    distance: f32,
    image: &mut Image3F,
    quant_field: &[f32],
    candidates: &SplineCandidates,
    forbidden: Option<&[bool]>,
) -> Option<SplineSet> {
    // Lines first, exactly as without dots: dots see the line residual, and
    // dropping every dot leaves the line-only result untouched.
    let split = candidates
        .0
        .iter()
        .position(|c| c.dot)
        .unwrap_or(candidates.0.len());
    let (lines, dots) = candidates.0.split_at(split);
    let mut set = if lines.is_empty() {
        None
    } else {
        select::rd_select(
            ctx,
            scratch,
            distance,
            image,
            quant_field,
            lines,
            &candidates.1,
            forbidden,
        )
    };
    if !dots.is_empty()
        && let Some(dot_set) = select::rd_select(
            ctx,
            scratch,
            distance,
            image,
            quant_field,
            dots,
            &[],
            forbidden,
        )
    {
        match set.as_mut() {
            Some(set) => {
                set.splines.extend(dot_set.splines);
                set.dot_encoding = dot_set.dot_encoding;
            }
            None => set = Some(dot_set),
        }
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_scaled_add_kernels_cover_unaligned_rows_and_tails() {
        let kernels = SplinesContext::get();
        #[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
        let fused = cfg!(target_arch = "aarch64");
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        let fused = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        let scale = 0.91f32;
        let src: [f32; 67] = std::array::from_fn(|i| (i as f32 - 17.0) * 0.071);
        let dst: [f32; 66] = std::array::from_fn(|i| (i as f32 - 23.0) * 0.083);
        for len in 0..=64 {
            for (add, fused) in [
                (kernels.add_scaled_f32, false),
                (kernels.add_scaled_fma_f32, fused),
            ] {
                let mut actual = dst;
                let mut expected = dst;
                add(&mut actual[1..1 + len], &src[2..], scale);
                for (out, &value) in expected[1..1 + len].iter_mut().zip(&src[2..]) {
                    *out = if fused {
                        scale.mul_add(value, *out)
                    } else {
                        *out + scale * value
                    };
                }
                assert_eq!(
                    actual.map(f32::to_bits),
                    expected.map(f32::to_bits),
                    "len={len}, fused={fused}"
                );
            }
        }
    }

    #[test]
    fn blob_weight_rows_preserve_coordinates_and_tail_values() {
        let fill = SplinesContext::get().blob_weight_row;
        for (x0, center, step) in [
            (-11, 0.0, 1.0),
            (0, 0.0, 1.0 / 32.0),
            (16_777_213, 16_777_219.0, 1.0),
            (
                crate::encode_image::MAX_DIMENSION as i32 - 111,
                (crate::encode_image::MAX_DIMENSION - 128) as f32,
                1.0,
            ),
        ] {
            for inv_sigma in [0.29, 1.4, 4.5] {
                let row = BlobWeightRow {
                    center,
                    step,
                    dy: -0.75,
                    inv_sigma,
                    amp: 0.25 / inv_sigma,
                };
                let mut scalar = [0.0; 32];
                let mut full = [0.0; 32];
                row.fill_scalar(x0, &mut scalar);
                fill(row, x0, &mut full);
                for (i, (&actual, expected)) in full.iter().zip(scalar).enumerate() {
                    // Fused Horner steps can shift either erf result by a few ulps.
                    assert!(
                        (actual - expected).abs() <= 2e-6 * row.amp,
                        "x0={x0}, lane={i}, inv_sigma={inv_sigma}: {actual} vs {expected}"
                    );
                }
                for len in 0..full.len() {
                    let mut partial = [f32::NAN; 32];
                    fill(row, x0, &mut partial[..len]);
                    assert_eq!(partial[..len], full[..len], "x0={x0}, len={len}");
                    assert!(partial[len..].iter().all(|x| x.is_nan()));
                }
            }
        }
    }

    #[test]
    fn coding_order_chains_neighboring_starts() {
        let at = |x: i32, y: i32| QuantizedSpline {
            points: vec![Point::new(x, y), Point::new(x + 30, y)],
            dct: [[0; 32]; 4],
        };
        let splines = [at(200, 200), at(10, 10), at(210, 190), at(20, 12)];
        let starts: Vec<_> = coding_order(&splines)
            .iter()
            .map(|sp| (sp.points[0].x, sp.points[0].y))
            .collect();
        assert_eq!(starts, [(10, 10), (20, 12), (200, 200), (210, 190)]);
    }

    #[test]
    fn candidates_touch_only_the_blocks_they_pass_through() {
        let line = QuantizedSpline {
            points: vec![Point::new(4, 4), Point::new(60, 4)],
            dct: [[0; 32]; 4],
        };
        let candidates = SplineCandidates(
            vec![fit::Candidate {
                alts: vec![line],
                bits_factor: 1.0,
                dot: false,
            }],
            Vec::new(),
        );
        // 8 x 2 blocks: the line runs through the whole top row
        let mut blocks = vec![false; 16];
        assert!(!candidates.touches(&blocks, 8));
        blocks[8 + 3] = true;
        assert!(!candidates.touches(&blocks, 8));
        blocks[5] = true;
        assert!(candidates.touches(&blocks, 8));
    }

    #[test]
    fn local_catmull_rom_spans_match_the_full_curve() {
        let control = [
            Point::new(2.0, 7.0),
            Point::new(11.0, 19.0),
            Point::new(25.0, 4.0),
            Point::new(33.0, 18.0),
            Point::new(48.0, 11.0),
            Point::new(63.0, 23.0),
        ];
        let full = catmull_rom(&control);
        for start in 0..control.len() {
            for end in start..control.len() {
                assert_eq!(
                    catmull_rom_spans(&control, start..end),
                    full[start * 16..=end * 16],
                    "spans {start}..{end}",
                );
            }
        }
        assert_eq!(catmull_rom(&control[..1]), control[..1]);
    }

    #[test]
    fn signed_coordinates_round_at_half_and_integer_precision_boundaries() {
        for value in [
            0.5f32.next_down(),
            0.5,
            0.5f32.next_up(),
            1.5,
            8_388_607.5,
            8_388_609.0,
            2_147_483_648.0,
            9_223_372_036_854_775_808.0,
            0.0,
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NAN,
        ] {
            for value in [value, -value] {
                for value in [value.next_down(), value, value.next_up()] {
                    assert_eq!(round_i32(value), value.round() as i32);
                    assert_eq!(round_isize(value), value.round() as isize);
                }
            }
        }
    }

    #[test]
    fn blob_window_preserves_clipping_and_signed_half_ties() {
        for cx in [-32.0f32, -1.5, -0.5, 0.0, 0.5, 15.5, 31.5, 48.0] {
            for cy in [-16.0f32, -0.5, 0.0, 0.5, 7.5, 15.5, 24.0] {
                for reach in [0.0, 0.5, 1.0, 2.75, 16.0] {
                    let reference = (
                        ((cx - reach).round() as isize).clamp(0, 32) as usize,
                        ((cx + reach).round() as isize + 1).clamp(0, 32) as usize,
                        ((cy - reach).round() as isize).clamp(0, 16) as usize,
                        ((cy + reach).round() as isize + 1).clamp(0, 16) as usize,
                    );
                    assert_eq!(blob_window(cx, cy, reach, 32, 16), reference);
                }
            }
        }
    }
}
