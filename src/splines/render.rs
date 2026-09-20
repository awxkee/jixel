//! Prepared decoder rendering. Samples retain absolute coordinates and their
//! original order, including when a trial renders only a sparse set of blocks.

use super::{
    CHANNEL_WEIGHT, PixelBox, Point, QuantizedSpline, Y_TO_B, Y_TO_X, adjusted_quant, blob_weight,
    blob_window, sample_curve,
};
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

pub(crate) struct Sample {
    pub(crate) position: Point<f32>,
    pub(crate) color: [f32; 3],
    pub(crate) inv_sigma: f32,
    pub(crate) amp: f32,
    /// Half-open pixel window.
    window: (usize, usize, usize, usize),
}

impl Sample {
    #[inline]
    fn row(&self, kernel: RenderRowFn, y: usize, x0: usize, rows: [&mut [f32]; 3], sign: f32) {
        kernel(self, y as f32 - self.position.y, x0, rows, self.amp * sign);
    }

    #[inline]
    pub(crate) fn scalar_row(&self, dy: f32, x0: usize, rows: [&mut [f32]; 3], amp: f32) {
        let [rx, ry, rb] = rows;
        for (x, ((rx, ry), rb)) in (x0..).zip(rx.iter_mut().zip(ry).zip(rb)) {
            let li = blob_weight(x as f32 - self.position.x, dy, self.inv_sigma, amp);
            *rx += self.color[0] * li;
            *ry += self.color[1] * li;
            *rb += self.color[2] * li;
        }
    }
}

pub(crate) type RenderRowFn = fn(&Sample, f32, usize, [&mut [f32]; 3], f32);

pub(crate) fn select_render_row_fn() -> RenderRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |sample, dy, x0, rows, amp| unsafe {
        crate::neon::spline_render_row_neon(sample, dy, x0, rows, amp)
    };
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |sample, dy, x0, rows, amp| unsafe {
            crate::avx::spline_render_row_avx2(sample, dy, x0, rows, amp)
        };
    }
    #[allow(unreachable_code)]
    Sample::scalar_row
}

pub(super) struct RenderPlan {
    samples: Vec<Sample>,
    pub(super) bounds: Option<PixelBox>,
    width: usize,
    row: RenderRowFn,
}

impl RenderPlan {
    pub(super) fn new(
        ctx: &EncodingContext,
        sp: &QuantizedSpline,
        adjust: i32,
        w: usize,
        h: usize,
    ) -> Self {
        let inv_quant = 1.0 / adjusted_quant(adjust);
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
        let mut plan = Self {
            samples: Vec::new(),
            bounds: None,
            width: w,
            row: ctx.spline_render_row,
        };
        if arc <= 0.0 {
            return plan;
        }
        plan.samples.reserve(samples.len());
        let continuous_idct = ctx.spline_continuous_idct;
        let r = 0.1f32.ln() * 5.0;
        for (k, sample) in samples.iter().enumerate() {
            let Point { x: cx, y: cy } = sample.position;
            let mult = sample.multiplier;
            let t = 31.0 * (k as f32 / arc).min(1.0);
            let [x, y, b, sigma] = continuous_idct(&dct, t);
            let color = [x, y, b];
            if !(sigma.is_finite() && sigma != 0.0 && (1.0 / sigma).is_finite()) {
                continue;
            }
            let mut max_color = 0.01f32;
            for c in color {
                max_color = max_color.max((c * mult).abs());
            }
            let reach = (-2.0 * sigma * sigma * (r - max_color.ln())).sqrt();
            let (x0, x1, y0, y1) = blob_window(cx, cy, reach, w, h);
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            plan.samples.push(Sample {
                position: sample.position,
                color,
                inv_sigma: 1.0 / sigma,
                amp: 0.25 * sigma * mult,
                window: (x0, x1, y0, y1),
            });
            plan.bounds = Some(match plan.bounds {
                None => (x0, y0, x1 - 1, y1 - 1),
                Some(b) => (b.0.min(x0), b.1.min(y0), b.2.max(x1 - 1), b.3.max(y1 - 1)),
            });
        }
        plan
    }

    pub(super) fn draw(&self, image: &mut Image3F, sign: f32) {
        for sample in &self.samples {
            let (x0, x1, y0, y1) = sample.window;
            for y in y0..y1 {
                let [rx, ry, rb] = image.all_plane_rows_mut(y);
                sample.row(
                    self.row,
                    y,
                    x0,
                    [&mut rx[x0..x1], &mut ry[x0..x1], &mut rb[x0..x1]],
                    sign,
                );
            }
        }
    }

    pub(super) fn tiled(self) -> TiledSpline {
        let blocks_w = self.width.div_ceil(8);
        let mut coverage = Vec::new();
        for (index, sample) in self.samples.iter().enumerate() {
            let (x0, x1, y0, y1) = sample.window;
            for by in y0 / 8..=(y1 - 1) / 8 {
                for bx in x0 / 8..=(x1 - 1) / 8 {
                    coverage.push((by * blocks_w + bx, index));
                }
            }
        }
        // The sample index is a secondary key: each pixel sees additions in
        // decoder order even though different blocks can be rendered separately.
        coverage.sort_unstable();
        let mut tiles: Vec<Tile> = Vec::new();
        let mut sample_indices = Vec::with_capacity(coverage.len());
        for (cell, index) in coverage {
            if tiles.last().is_none_or(|tile| tile.cell != cell) {
                tiles.push(Tile {
                    cell,
                    start: sample_indices.len(),
                    end: sample_indices.len(),
                });
            }
            sample_indices.push(index);
            tiles.last_mut().unwrap().end += 1;
        }
        TiledSpline {
            plan: self,
            tiles,
            sample_indices,
        }
    }
}

pub(super) struct Tile {
    pub(super) cell: usize,
    start: usize,
    end: usize,
}

pub(super) struct TiledSpline {
    plan: RenderPlan,
    pub(super) tiles: Vec<Tile>,
    sample_indices: Vec<usize>,
}

impl TiledSpline {
    pub(super) fn bounds(&self) -> Option<PixelBox> {
        self.plan.bounds
    }

    pub(super) fn draw_tile(&self, tile: &Tile, pixels: &mut [[f32; 64]; 3], sign: f32) {
        let blocks_w = self.plan.width.div_ceil(8);
        let (tx, ty) = (tile.cell % blocks_w * 8, tile.cell / blocks_w * 8);
        for &index in &self.sample_indices[tile.start..tile.end] {
            let sample = &self.plan.samples[index];
            let (sx0, sx1, sy0, sy1) = sample.window;
            let (x0, x1) = (sx0.max(tx), sx1.min(tx + 8));
            for y in sy0.max(ty)..sy1.min(ty + 8) {
                let start = (y - ty) * 8 + x0 - tx;
                let end = start + x1 - x0;
                let [rx, ry, rb] = &mut *pixels;
                sample.row(
                    self.plan.row,
                    y,
                    x0,
                    [
                        &mut rx[start..end],
                        &mut ry[start..end],
                        &mut rb[start..end],
                    ],
                    sign,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splines::QUANT_ADJUST;

    #[test]
    fn render_row_matches_scalar_for_all_alignments_and_tails() {
        let ctx = EncodingContext::default();
        for x0 in [
            3,
            16_777_215,
            16_777_217,
            crate::encode_image::MAX_DIMENSION - 32,
        ] {
            for inv_sigma in [-1.7, 0.31] {
                for dy in [-0.0, 2.375] {
                    let sample = Sample {
                        position: Point::new(x0 as f32 + 0.35, 0.0),
                        color: [-0.013, 0.071, 0.017],
                        inv_sigma,
                        amp: -0.12,
                        window: (0, 0, 0, 0),
                    };
                    for start in 0..8 {
                        for len in 0..=25 {
                            let end = start + len;
                            let mut expected: [[f32; 40]; 3] = std::array::from_fn(|c| {
                                std::array::from_fn(|i| (i * 19 + c * 7) as f32 * 0.003 - 0.3)
                            });
                            let mut actual = expected;
                            let [rx, ry, rb] = &mut expected;
                            sample.scalar_row(
                                dy,
                                x0,
                                [
                                    &mut rx[start..end],
                                    &mut ry[start..end],
                                    &mut rb[start..end],
                                ],
                                sample.amp,
                            );
                            let [rx, ry, rb] = &mut actual;
                            (ctx.spline_render_row)(
                                &sample,
                                dy,
                                x0,
                                [
                                    &mut rx[start..end],
                                    &mut ry[start..end],
                                    &mut rb[start..end],
                                ],
                                sample.amp,
                            );
                            for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "x0={x0}, start={start}, len={len}, inv_sigma={inv_sigma}, dy={dy}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_render_matches_original_sample_arithmetic() {
        let kernels = crate::encoding_context::EncodingContext::default();
        for (w, h) in [(37, 29), (129, 87)] {
            for width in [0, 1, 4, -3] {
                let mut sp = QuantizedSpline {
                    points: vec![
                        Point::new(-2, 1),
                        Point::new(12, 23),
                        Point::new(w as i32 - 1, h as i32 - 2),
                    ],
                    dct: [[0; 32]; 4],
                };
                sp.dct[0][0] = -11;
                sp.dct[1][0] = 7;
                sp.dct[1][2] = -3;
                sp.dct[2][1] = 9;
                sp.dct[3][0] = width;
                sp.dct[3][3] = if width == 0 { 0 } else { 1 };
                for sign in [-1.0, 1.0] {
                    let mut original = Image3F::new(w, h);
                    for c in 0..3 {
                        for y in 0..h {
                            for (x, value) in original.plane_row_mut(c, y).iter_mut().enumerate() {
                                *value = ((x * 19 + y * 31 + c * 7) % 83) as f32 * 0.001 - 0.04;
                            }
                        }
                    }
                    let mut prepared = original.clone();
                    let bounds = reference_render(&sp, QUANT_ADJUST, &mut original, sign);
                    let plan = RenderPlan::new(&kernels, &sp, QUANT_ADJUST, w, h);
                    plan.draw(&mut prepared, sign);
                    assert_eq!(plan.bounds, bounds);
                    for c in 0..3 {
                        assert!(
                            original
                                .plane_data(c)
                                .iter()
                                .zip(prepared.plane_data(c))
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                        );
                    }
                }
            }
        }
    }
    fn reference_render(
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
        let continuous_idct = super::super::transform::continuous_idct_scalar;
        let mut touched: Option<PixelBox> = None;
        for (k, sample) in samples.iter().enumerate() {
            let Point { x: cx, y: cy } = sample.position;
            let mult = sample.multiplier;
            let t = 31.0 * (k as f32 / arc).min(1.0);
            let [x, y, b, sigma] = continuous_idct(&dct, t);
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
}
