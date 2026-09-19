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
mod extend;
mod fit;
mod lines;
mod select;
mod transform;

use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::entropy::{
    Token, optimize_entropy_code_ac, pack_signed, write_ans_tokens, write_entropy_code, write_token,
};
use crate::image::Image3F;

/// X, Y, B, sigma.
const CHANNEL_WEIGHT: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];
const Y_TO_X: f32 = 0.0;
const Y_TO_B: f32 = 1.0;

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

/// Signed coordinates and color residuals need the half offset in both directions.
#[inline]
fn round_i32(value: f32) -> i32 {
    // Adding half would round the float immediately below 0.5 up to 1.0;
    // at 2^23 and above, f32 values are already integral.
    if value.abs() < 0.5 || value.abs() >= 8_388_608.0 {
        value as i32
    } else {
        (value + 0.5f32.copysign(value)) as i32
    }
}

#[inline]
fn fast_hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
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
pub(crate) fn render_spline(
    sp: &QuantizedSpline,
    adjust: i32,
    xyb: &mut Image3F,
    sign: f32,
) -> Option<PixelBox> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let inv_quant = 1.0 / adjusted_quant(adjust);
    // Interleave channels so the continuous IDCT can accumulate four SIMD lanes.
    let mut dct = [[0f32; 4]; 32];
    for (i, row) in dct.iter_mut().enumerate() {
        let f = if i == 0 {
            std::f32::consts::FRAC_1_SQRT_2
        } else {
            1.0
        };
        for ((v, channel), weight) in row.iter_mut().zip(&sp.dct).zip(CHANNEL_WEIGHT) {
            *v = channel[i] as f32 * f * weight * inv_quant;
        }
        row[0] += Y_TO_X * row[1];
        row[2] += Y_TO_B * row[1];
        for v in row {
            *v *= std::f32::consts::SQRT_2;
        }
    }
    let (samples, arc) = sample_curve(&sp.points);
    if arc <= 0.0 {
        return None;
    }
    let mut touched: Option<PixelBox> = None;
    for (k, sample) in samples.iter().enumerate() {
        let Point { x: cx, y: cy } = sample.position;
        let mult = sample.multiplier;
        let t = 31.0 * (k as f32 / arc).min(1.0);
        let [x, y, b, sigma] = transform::continuous_idct(&dct, t);
        let color = [x, y, b];
        if !(sigma.is_finite() && sigma != 0.0 && (1.0 / sigma).is_finite()) {
            continue;
        }
        let mut max_color = 0.01f32;
        for c in color {
            max_color = max_color.max((c * mult).abs());
        }
        let reach = (-2.0 * sigma * sigma * (0.1f32.ln() * 5.0 - max_color.ln())).sqrt();
        let (x0, x1, y0, y1) = blob_window(cx, cy, reach, w, h);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        let (inv_sigma, amp) = (1.0 / sigma, 0.25 * sigma * mult * sign);
        for y in y0..y1 {
            let dy = y as f32 - cy;
            let [rx, ry, rb] = xyb.all_plane_rows_mut(y);
            for (x, ((rx, ry), rb)) in (x0..x1).zip(
                rx[x0..x1]
                    .iter_mut()
                    .zip(&mut ry[x0..x1])
                    .zip(&mut rb[x0..x1]),
            ) {
                let li = blob_weight(x as f32 - cx, dy, inv_sigma, amp);
                *rx += color[0] * li;
                *ry += color[1] * li;
                *rb += color[2] * li;
            }
        }
        touched = Some(match touched {
            None => (x0, y0, x1 - 1, y1 - 1),
            Some(b) => (b.0.min(x0), b.1.min(y0), b.2.max(x1 - 1), b.3.max(y1 - 1)),
        });
    }
    touched
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
    for row in &sp.dct {
        for &v in row {
            t.push((CTX_DCT, pack_signed(v)));
        }
    }
    t
}

/// LfGlobal spline section (after the patch dictionary, before the DC scales).
pub(crate) fn write_splines(set: &SplineSet, scratch: &mut CoderScratch, w: &mut BitWriter) {
    let mut tokens = Vec::new();
    tokens.push(Token::new(CTX_NUM_SPLINES, (set.splines.len() - 1) as u32));
    let (mut lx, mut ly) = (0i32, 0i32);
    for (i, sp) in set.splines.iter().enumerate() {
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
    for sp in &set.splines {
        tokens.extend(spline_tokens(sp).into_iter().map(|(c, v)| Token::new(c, v)));
    }
    let code = optimize_entropy_code_ac(&tokens, NUM_SPLINE_CONTEXTS, &mut scratch.huffman_pool);
    let code_ref = code.as_ref();
    w.write(1, 0); // no LZ77
    write_entropy_code(&code_ref, &mut scratch.huffman_pool, w);
    if code_ref.use_prefix_code {
        for token in tokens {
            write_token(token, &code_ref, w);
        }
    } else {
        write_ans_tokens(
            &tokens,
            code_ref.context_map,
            code_ref.ans_symbols,
            code_ref.ans_reverse_maps,
            code_ref.hybrid_uint_configs,
            w,
        );
    }
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
pub(crate) struct SplineCandidates(Vec<fit::Candidate>);

/// Detects curvilinear structures in `xyb` and fits spline candidates to them.
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
    let chains = detect::detect_chains(ctx, scratch, xyb);
    if chains.is_empty() {
        return None;
    }
    let extensions = extend::find_extensions(ctx, scratch, xyb, &chains);
    let long_lines = lines::find_long_lines(&chains);
    let candidates = fit::fit_candidates(
        ctx,
        scratch,
        distance,
        xyb,
        quant_field,
        &chains,
        &extensions,
        &long_lines,
    );
    (!candidates.is_empty()).then_some(SplineCandidates(candidates))
}

/// RD-selects candidates against `image` and subtracts the accepted splines
/// from it. `quant_field` is the effective AC quant (`scale * q`) per 8x8
/// block; splines touching a `forbidden` block are skipped (patch rectangles:
/// the decoder draws splines on top of the replaced patch pixels).
pub(crate) fn select_splines(
    ctx: &EncodingContext,
    distance: f32,
    image: &mut Image3F,
    quant_field: &[f32],
    candidates: &SplineCandidates,
    forbidden: Option<&[bool]>,
) -> Option<SplineSet> {
    select::rd_select(ctx, distance, image, quant_field, &candidates.0, forbidden)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        ] {
            for value in [value, -value] {
                assert_eq!(round_i32(value), value.round() as i32);
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
