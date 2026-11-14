/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! The decoder's view of a VarDCT frame after its loop filters, for the bounded EPF strength fitter.
//!
//! The encoder's reconstruction capture carries the source plus each block's
//! AC quantization error. Adding what the decoder rebuilds from the decoded
//! DC (the lowest frequencies of every transform) and the forward Gaborish
//! models the image the edge-preserving filter sees. The filter itself
//! follows libjxl's three render-pipeline stages.

use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::{AcStrategyImage, DcGroupData, STRATEGY_DCT};
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, Plane};
use crate::thread_pool::ThreadPool;
use std::borrow::Cow;
use std::sync::OnceLock;

mod kernels;
mod stream;
#[cfg(test)]
mod tests;

const INV_SIGMA_NUM: f32 = -1.171_572_9;
/// `INV_SIGMA_NUM / 0.3`: blocks whose inverse sigma falls below this are
/// left unfiltered.
const MIN_SIGMA: f32 = -3.905_243;
const SAD_MUL: f32 = 1.65;
const SHARPNESS_IDS: i32 = 8;

/// Loop-filter fields as the decoder reads them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoopFilter {
    pub(crate) epf_iters: u32,
    quant_mul: f32,
    pass0_sigma_scale: f32,
    pass2_sigma_scale: f32,
    border_sad_mul: f32,
    channel_scale: [f32; 3],
}

impl LoopFilter {
    /// The fields `write_loop_filter` serializes: a custom sigma bundle is
    /// read back through F16, the defaults are the spec's floats.
    pub(crate) fn new(
        epf_iters: u32,
        epf_pass0_scale: Option<f32>,
        quant_mul: Option<f32>,
        channel_scale: Option<[f32; 3]>,
    ) -> Self {
        let f16 = |v: f32| crate::util::f16_bits_to_f32(crate::util::f32_to_f16_bits(v));
        let (quant_mul, pass0, pass2, border) = match epf_pass0_scale.or(quant_mul.map(|_| 0.9)) {
            Some(s) => (
                f16(quant_mul.unwrap_or(0.46)),
                f16(s),
                f16(6.5),
                f16(2.0 / 3.0),
            ),
            None => (0.46, 0.9, 6.5, 2.0 / 3.0),
        };
        Self {
            epf_iters,
            quant_mul,
            pass0_sigma_scale: pass0,
            pass2_sigma_scale: pass2,
            border_sad_mul: border,
            channel_scale: channel_scale.map_or([40.0, 5.0, 3.5], |s| s.map(f16)),
        }
    }
}

/// Per 8x8 block: the transform's sigma denominator, from its first block's
/// quantizer. Candidate strength and sharpness are applied when rendering.
pub(crate) struct FrameBlocks {
    xsize: usize,
    ysize: usize,
    sigma_quant: Vec<f32>,
}

impl FrameBlocks {
    pub(crate) fn new(
        dc_datas: &[DcGroupData],
        group_coords: &[(usize, usize)],
        xsize_blocks: usize,
        ysize_blocks: usize,
        global_scale: i32,
    ) -> Self {
        const DC_GROUP_BLOCKS: usize = crate::frame::K_DC_GROUP_DIM / 8;
        let quant_scale = global_scale as f32 / 65536.0;
        let mut sigma_quant = vec![0.0; xsize_blocks * ysize_blocks];
        for (dc, &(gx, gy)) in dc_datas.iter().zip(group_coords) {
            let (ox, oy) = (gx * DC_GROUP_BLOCKS, gy * DC_GROUP_BLOCKS);
            let acs = &dc.ac_strategy;
            for by in 0..acs.ysize() {
                for bx in 0..acs.xsize() {
                    if !acs.is_first_block(bx, by) {
                        continue;
                    }
                    let raw = acs.raw_strategy(bx, by);
                    let q = f32::from(dc.raw_quant_field.row(by)[bx]);
                    let s = quant_scale * q * INV_SIGMA_NUM;
                    for iy in 0..AcStrategyImage::covered_blocks_y_of(raw) {
                        for ix in 0..AcStrategyImage::covered_blocks_x_of(raw) {
                            let (x, y) = (ox + bx + ix, oy + by + iy);
                            if x < xsize_blocks && y < ysize_blocks {
                                sigma_quant[y * xsize_blocks + x] = s;
                            }
                        }
                    }
                }
            }
        }
        Self {
            xsize: xsize_blocks,
            ysize: ysize_blocks,
            sigma_quant,
        }
    }
}

#[inline]
fn mirror(mut i: isize, n: isize) -> usize {
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - 1 - i;
        } else {
            return i as usize;
        }
    }
}

/// The decoder's Gaborish with the default weights. Source rows are borrowed;
/// no padded frame or intermediate image is needed.
pub(crate) fn gaborish(image: &Image3F) -> Image3F {
    const W1: f32 = 1.1 * 0.104_699_57;
    const W2: f32 = 1.1 * 0.055_680_54;
    let norm = 1.0 / (1.0 + 4.0 * (W1 + W2));
    let weights = [norm, W1 * norm, W2 * norm];
    let (w, h) = (image.xsize(), image.ysize());
    let k = kernels::selected();
    let mut out = Image3F::new(w, h);
    for c in 0..3 {
        for y in 0..h {
            (k.gaborish)(
                [
                    image.plane_row(c, y.saturating_sub(1)),
                    image.plane_row(c, y),
                    image.plane_row(c, (y + 1).min(h - 1)),
                ],
                weights,
                out.plane_row_mut(c, y),
            );
        }
    }
    out
}

/// The owned reconstruction needs no second image for Gaborish. Preserve just
/// three source rows before overwriting each plane, reusing them across channels.
pub(crate) fn gaborish_in_place(image: &mut Image3F) {
    const W1: f32 = 1.1 * 0.104_699_57;
    const W2: f32 = 1.1 * 0.055_680_54;
    let norm = 1.0 / (1.0 + 4.0 * (W1 + W2));
    let weights = [norm, W1 * norm, W2 * norm];
    let (w, h) = (image.xsize(), image.ysize());
    if w == 0 || h == 0 {
        return;
    }
    let k = kernels::selected();
    let mut rows: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; w]);
    for c in 0..3 {
        rows[0].copy_from_slice(image.plane_row(c, 0));
        for y in 0..h {
            if y + 1 < h {
                rows[(y + 1) % 3].copy_from_slice(image.plane_row(c, y + 1));
            }
            (k.gaborish)(
                [
                    &rows[y.saturating_sub(1) % 3],
                    &rows[y % 3],
                    &rows[(y + 1).min(h - 1) % 3],
                ],
                weights,
                image.plane_row_mut(c, y),
            );
        }
    }
}

/// One EPF stage: the neighbors it averages, each as a canonical difference
/// direction and whether the neighbor lies along it or against it.
struct Stage {
    border: usize,
    /// Patch SADs (plus-shaped, 5 pixels) or single-pixel differences.
    patch: bool,
    dirs: &'static [(isize, isize)],
    neighbors: &'static [(usize, bool)],
}

static STAGE0: Stage = Stage {
    border: 3,
    patch: true,
    dirs: &[(1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (-1, 1)],
    // libjxl's order: (0,-2) (-1,-1) (0,-1) (1,-1) (-2,0) (-1,0) (1,0) (2,0)
    // (-1,1) (0,1) (1,1) (0,2), as (dx, dy).
    neighbors: &[
        (3, false),
        (4, false),
        (2, false),
        (5, false),
        (1, false),
        (0, false),
        (0, true),
        (1, true),
        (5, true),
        (2, true),
        (4, true),
        (3, true),
    ],
};
// Up, left, right, down.
static STAGE1: Stage = Stage {
    border: 2,
    patch: true,
    dirs: &[(1, 0), (0, 1)],
    neighbors: &[(1, false), (0, false), (0, true), (1, true)],
};
static STAGE2: Stage = Stage {
    border: 1,
    patch: false,
    dirs: &[(1, 0), (0, 1)],
    neighbors: &[(1, false), (0, false), (0, true), (1, true)],
};

/// One renderer for every pass shape and strength of the same reconstruction.
/// Only the output image persists; SADs and differences live in row rings.
pub(crate) struct Epf<'a> {
    image: &'a Image3F,
    blocks: &'a FrameBlocks,
    output: Option<Image3F>,
    inv_sigma: Vec<f32>,
    work: Vec<stream::Scratch>,
    kernels: kernels::Kernels,
}

impl<'a> Epf<'a> {
    pub(crate) fn new(pool: &ThreadPool, image: &'a Image3F, blocks: &'a FrameBlocks) -> Self {
        Self {
            image,
            blocks,
            output: None,
            inv_sigma: Vec::new(),
            work: (0..pool.num_threads().min(image.ysize()).max(1))
                .map(|_| stream::Scratch::default())
                .collect(),
            kernels: kernels::selected(),
        }
    }

    pub(crate) fn apply(
        &mut self,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
        lf: &LoopFilter,
        sharpness: i32,
    ) -> &Image3F {
        let (w, h) = (self.image.xsize(), self.image.ysize());
        if lf.epf_iters == 0 || sharpness == 0 || w == 0 || h == 0 {
            return self.image;
        }
        debug_assert_eq!(self.blocks.xsize, w.div_ceil(8));
        debug_assert_eq!(self.blocks.ysize, h.div_ceil(8));
        grow(&mut self.inv_sigma, self.blocks.sigma_quant.len());
        (self.kernels.sigma)(
            &self.blocks.sigma_quant,
            sharpness as f32 / (SHARPNESS_IDS - 1) as f32,
            lf.quant_mul,
            &mut self.inv_sigma,
        );
        if !self.inv_sigma.iter().any(|&s| s >= MIN_SIGMA) {
            return self.image;
        }
        let stages: &[(&Stage, f32)] = match lf.epf_iters {
            1 => &[(&STAGE1, SAD_MUL)],
            2 => &[
                (&STAGE1, SAD_MUL),
                (&STAGE2, SAD_MUL * lf.pass2_sigma_scale),
            ],
            _ => &[
                (&STAGE0, SAD_MUL * lf.pass0_sigma_scale),
                (&STAGE1, SAD_MUL),
                (&STAGE2, SAD_MUL * lf.pass2_sigma_scale),
            ],
        };
        let output = self.output.get_or_insert_with(|| Image3F::new(w, h));
        for (i, &(stage, sm)) in stages.iter().enumerate() {
            stream::render(
                pool,
                scratch,
                if i == 0 { Some(self.image) } else { None },
                output,
                stage,
                lf,
                &self.inv_sigma,
                self.blocks.xsize,
                sm,
                self.kernels,
                &mut self.work,
            );
        }
        output
    }
}

/// Per raw strategy: `dc = M · llf` inverted, row-major over the lowest
/// frequencies (`v * cx + u`) by DC index (`iy * cov_x + ix`).
fn llf_from_dc(ctx: &EncodingContext, raw: u8) -> &'static [f32] {
    static TABLES: [OnceLock<Vec<f32>>; crate::dc_group_data::NUM_STRATEGIES] =
        [const { OnceLock::new() }; crate::dc_group_data::NUM_STRATEGIES];
    TABLES[raw as usize].get_or_init(|| {
        let (cov_x, cov_y) = (
            AcStrategyImage::covered_blocks_x_of(raw),
            AcStrategyImage::covered_blocks_y_of(raw),
        );
        let (cx, cy) = (cov_x.max(cov_y), cov_x.min(cov_y));
        let n = cx * cy;
        let mut coeffs = vec![0.0f32; n * 64];
        let mut dc = vec![0.0f32; n];
        // m[k][j]: DC k from a unit lowest frequency j.
        let mut m = vec![0.0f32; n * n];
        for j in 0..n {
            let (v, u) = (j / cx, j % cx);
            coeffs[v * cx * 8 + u] = 1.0;
            crate::group::dc_from_coefficients(ctx, raw, &coeffs, &mut dc);
            coeffs[v * cx * 8 + u] = 0.0;
            for k in 0..n {
                m[k * n + j] = dc[k];
            }
        }
        invert(&mut m, n)
    })
}

/// Gauss-Jordan inverse with partial pivoting.
fn invert(m: &mut [f32], n: usize) -> Vec<f32> {
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| m[a * n + col].abs().total_cmp(&m[b * n + col].abs()))
            .unwrap();
        for k in 0..n {
            m.swap(col * n + k, pivot * n + k);
            inv.swap(col * n + k, pivot * n + k);
        }
        let d = 1.0 / m[col * n + col];
        for k in 0..n {
            m[col * n + k] *= d;
            inv[col * n + k] *= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                m[r * n + k] -= f * m[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    inv
}

/// Add to the AC reconstruction capture (source plus AC error, frame-sized)
/// what each transform's lowest frequencies contribute from the decoded DC
/// instead of the source DC. `dc_decoded` and `dc_source` hold one value per
/// 8x8 block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_dc_error(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    recon: &mut Image3F,
    dc_decoded: &Image3F,
    dc_source: &Image3F,
    dc_datas: &[DcGroupData],
    group_coords: &[(usize, usize)],
) {
    const DC_GROUP_BLOCKS: usize = crate::frame::K_DC_GROUP_DIM / 8;
    // Transforms never cross an AC group, so its pixel rows are independent.
    const BAND_ROWS: usize = crate::frame::K_GROUP_DIM;
    let (w, h) = (recon.xsize(), recon.ysize());
    let kernels = kernels::selected();
    let mut bands: Vec<(usize, [&mut [f32]; 3])> = {
        let [p0, p1, p2] = recon.all_planes_mut().map(Plane::as_mut_slice);
        p0.chunks_mut(BAND_ROWS * w)
            .zip(p1.chunks_mut(BAND_ROWS * w))
            .zip(p2.chunks_mut(BAND_ROWS * w))
            .map(|((a, b), c)| [a, b, c])
            .enumerate()
            .collect()
    };
    ctx.thread_pool
        .steal_for_each_mut(scratch, &mut bands, |_, (band, planes), _| {
            let y_lo = *band * BAND_ROWS;
            // Most transforms need much less than the 128x128 maximum.
            // DCT8 only needs a constant row add and allocates neither buffer.
            let mut coeffs = Vec::new();
            let mut pix = Vec::new();
            let mut delta = [0.0f32; 256];
            for (dc, &(gx, gy)) in dc_datas.iter().zip(group_coords) {
                let (ox, oy) = (gx * DC_GROUP_BLOCKS, gy * DC_GROUP_BLOCKS);
                let acs = &dc.ac_strategy;
                let by_range = (y_lo / 8).saturating_sub(oy).min(acs.ysize())
                    ..((y_lo + BAND_ROWS) / 8).saturating_sub(oy).min(acs.ysize());
                for by in by_range {
                    for bx in 0..acs.xsize() {
                        if !acs.is_first_block(bx, by) {
                            continue;
                        }
                        let raw = acs.raw_strategy(bx, by);
                        let (fbx, fby) = (ox + bx, oy + by);
                        let (px0, py0) = (fbx * 8, fby * 8);
                        let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(raw);
                        debug_assert!(py0 + ph <= y_lo + BAND_ROWS || py0 + ph > h);
                        if raw == STRATEGY_DCT {
                            for c in 0..3 {
                                let d = dc_decoded.plane_row(c, fby)[fbx]
                                    - dc_source.plane_row(c, fby)[fbx];
                                for y in py0..(py0 + 8).min(h) {
                                    let row = &mut planes[c][(y - y_lo) * w..(y - y_lo + 1) * w];
                                    (kernels.add_constant)(&mut row[px0..(px0 + 8).min(w)], d);
                                }
                            }
                            continue;
                        }
                        let (cov_x, cov_y) = (
                            AcStrategyImage::covered_blocks_x_of(raw),
                            AcStrategyImage::covered_blocks_y_of(raw),
                        );
                        let (cx, cy) = (cov_x.max(cov_y), cov_x.min(cov_y));
                        let n = cx * cy;
                        let inv = llf_from_dc(ctx, raw);
                        grow(&mut coeffs, n * 64);
                        grow(&mut pix, n * 64);
                        for c in 0..3 {
                            for iy in 0..cov_y {
                                for ix in 0..cov_x {
                                    let (x, y) = (fbx + ix, fby + iy);
                                    delta[iy * cov_x + ix] = if x < dc_decoded.xsize()
                                        && y < dc_decoded.ysize()
                                    {
                                        dc_decoded.plane_row(c, y)[x] - dc_source.plane_row(c, y)[x]
                                    } else {
                                        0.0
                                    };
                                }
                            }
                            (kernels.project)(inv, &delta[..n], cx, &mut coeffs[..n * 64]);
                            crate::inflated_cost::reconstruct_error(
                                ctx.idct,
                                raw,
                                &coeffs[..n * 64],
                                &mut pix[..n * 64],
                            );
                            for y in 0..ph.min(h.saturating_sub(py0)) {
                                let row =
                                    &mut planes[c][(py0 + y - y_lo) * w..(py0 + y - y_lo + 1) * w];
                                let width = pw.min(w.saturating_sub(px0));
                                (kernels.add)(
                                    &mut row[px0..px0 + width],
                                    &pix[y * pw..y * pw + width],
                                );
                            }
                        }
                        // Projection only touches these LLF entries. Reset them
                        // once per transform, retaining zero AC coefficients for
                        // the next shape instead of clearing three whole blocks.
                        for j in 0..n {
                            coeffs[j / cx * cx * 8 + j % cx] = 0.0;
                        }
                    }
                }
            }
        });
}

/// Per raw strategy: the pixel image (row-major, the transform's pixel
/// width) of a unit change of each DC cell (`iy * cov_x + ix`) of one
/// transform, through its lowest frequencies.
#[cfg(feature = "extraslow")]
pub(crate) fn dc_cell_images(ctx: &EncodingContext, raw: u8) -> &'static [f32] {
    static TABLES: [OnceLock<Vec<f32>>; crate::dc_group_data::NUM_STRATEGIES] =
        [const { OnceLock::new() }; crate::dc_group_data::NUM_STRATEGIES];
    TABLES[raw as usize].get_or_init(|| {
        let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(raw);
        let pixels = pw * ph;
        if raw == STRATEGY_DCT {
            return vec![1.0; pixels];
        }
        let kernels = kernels::selected();
        let (cov_x, cov_y) = (
            AcStrategyImage::covered_blocks_x_of(raw),
            AcStrategyImage::covered_blocks_y_of(raw),
        );
        let cx = cov_x.max(cov_y);
        let n = cov_x * cov_y;
        let inv = llf_from_dc(ctx, raw);
        let mut images = vec![0.0f32; n * pixels];
        let mut coeffs = vec![0.0f32; pixels];
        let mut delta = vec![0.0f32; n];
        for (slot, image) in images.chunks_exact_mut(pixels).enumerate() {
            delta[slot] = 1.0;
            coeffs.fill(0.0);
            (kernels.project)(inv, &delta, cx, &mut coeffs);
            crate::inflated_cost::reconstruct_error(ctx.idct, raw, &coeffs, image);
            delta[slot] = 0.0;
        }
        images
    })
}

fn grow(buffer: &mut Vec<f32>, len: usize) {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Choice {
    pub(crate) iters: u32,
    pub(crate) quant_mul: Option<f32>,
    pub(crate) pass0: Option<f32>,
    pub(crate) sharpness: i32,
}

impl Choice {
    fn candidate(iters: u32, strength: f32, pass0: Option<f32>) -> Self {
        Self {
            iters,
            quant_mul: Some(0.46 * strength),
            pass0,
            sharpness: 7,
        }
    }
}

/// Baseline/off, four shape pilots, two endpoints and one log-quadratic vertex.
/// The callback supplies only a cheap proxy score; no exact-metric acceptance.
fn choose(distance: f32, baseline: Choice, mut score: impl FnMut(Choice) -> f32) -> Choice {
    fn trial(
        rows: &mut Vec<(Choice, f32)>,
        score: &mut impl FnMut(Choice) -> f32,
        choice: Choice,
    ) -> (Choice, f32) {
        if let Some(&r) = rows.iter().find(|r| r.0 == choice) {
            return r;
        }
        let loss = score(choice);
        let result = (
            choice,
            if loss.is_finite() {
                loss
            } else {
                f32::INFINITY
            },
        );
        rows.push(result);
        result
    }
    let mut rows = Vec::with_capacity(9);
    trial(&mut rows, &mut score, baseline);
    trial(
        &mut rows,
        &mut score,
        Choice {
            iters: 0,
            quant_mul: None,
            pass0: None,
            ..baseline
        },
    );
    let center = (2. / distance).clamp(0.0625, 4.);
    let mut pilots = Vec::with_capacity(4);
    for (iters, pass0) in [(1, None), (2, None), (3, Some(0.9)), (3, Some(3.6))] {
        pilots.push(trial(
            &mut rows,
            &mut score,
            Choice::candidate(iters, center, pass0),
        ));
    }
    let &(pilot, middle) = pilots.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    let lo = (center / 2.).max(0.0625);
    let hi = (center * 2.).min(4.);
    let low = trial(
        &mut rows,
        &mut score,
        Choice::candidate(pilot.iters, lo, pilot.pass0),
    )
    .1;
    let high = trial(
        &mut rows,
        &mut score,
        Choice::candidate(pilot.iters, hi, pilot.pass0),
    )
    .1;
    if lo < center && center < hi {
        let (x0, x1, x2) = (lo.log2(), center.log2(), hi.log2());
        let a = ((middle - low) / (x1 - x0) - (high - middle) / (x2 - x1)) / (x0 - x2);
        let b = (middle - low) / (x1 - x0) - a * (x0 + x1);
        if a > 1e-12 {
            let strength = 2f32.powf((-b / (2. * a)).clamp(x0, x2));
            trial(
                &mut rows,
                &mut score,
                Choice::candidate(pilot.iters, strength, pilot.pass0),
            );
        }
    }
    rows.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0
}

/// What later frames lay over a saved base frame, in luma: the decoder
/// shows `base * (1 - coverage) + luma * coverage` (an ink frame blended by
/// its coverage, crop frames replacing their boxes with coverage 1).
pub(crate) enum EpfOverlay {
    Ink {
        indices: Vec<u8>,
        coverage: Vec<u8>,
        luma: Vec<f32>,
    },
    Crops {
        coverage: Vec<u8>,
    },
}

const fn byte_coverage() -> [f32; 256] {
    let mut values = [0.0; 256];
    let mut i = 0;
    while i < values.len() {
        values[i] = i as f32 / 255.0;
        i += 1;
    }
    values
}

impl EpfOverlay {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Ink { coverage, .. } | Self::Crops { coverage } => coverage.len(),
        }
    }

    fn blend(&self, output: &mut [f32]) {
        match self {
            Self::Ink {
                indices,
                coverage,
                luma,
            } => {
                static ALPHA: [f32; 256] = byte_coverage();
                assert_eq!(output.len(), indices.len());
                assert_eq!(output.len(), coverage.len());
                for ((out, &k), &a) in output.iter_mut().zip(indices).zip(coverage) {
                    let c = ALPHA[usize::from(if k == 0 { 0 } else { a })];
                    *out = *out * (1.0 - c) + luma[usize::from(k)] * c;
                }
            }
            Self::Crops { coverage } => {
                assert_eq!(output.len(), coverage.len());
                for (out, &c) in output.iter_mut().zip(coverage) {
                    *out = *out * (1.0 - f32::from(c)) + 0.0;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LumaLayers<'a> {
    pub(crate) additive: Option<&'a [f32]>,
    pub(crate) patches: Option<(
        &'a [crate::patches::PatchReference],
        &'a crate::patches::EpfPatchAtlases,
    )>,
    /// Later frames over this one, applied last.
    pub(crate) overlay: Option<&'a EpfOverlay>,
}

impl LumaLayers<'_> {
    fn is_empty(self) -> bool {
        self.additive.is_none() && self.patches.is_none() && self.overlay.is_none()
    }

    fn compose(mut self, base: &[f32], width: usize) -> Vec<f32> {
        let mut output = if self.patches.is_none()
            && let Some(layer) = self.additive.take()
        {
            assert_eq!(base.len(), layer.len());
            base.iter()
                .zip(layer)
                .map(|(&base, &layer)| base + layer)
                .collect()
        } else {
            base.to_vec()
        };
        self.compose_in_place(width, &mut output);
        output
    }

    /// The decoder applies patches in dictionary order, then draws splines.
    /// The filter itself still receives the base reconstruction.
    fn compose_in_place(self, width: usize, output: &mut [f32]) {
        if let Some((references, atlases)) = self.patches {
            let glyph_step = 1.0 / crate::quant_weights::INV_DC_QUANT[1];
            for patch in references.iter().filter(|patch| !patch.alpha_only) {
                for &(px, py) in &patch.positions {
                    for dy in 0..patch.height {
                        let start = (py + dy) * width + px;
                        let dst = &mut output[start..start + patch.width];
                        let ay = patch.atlas_y + dy;
                        let ax = patch.atlas_x;
                        if patch.mul && !patch.add {
                            // kBlendAbove: the template color over the frame by its coverage.
                            let start = ay * atlases.glyphs.width + ax;
                            let src = &atlases.glyphs.samples[start..start + patch.width];
                            let coverage = atlases.glyphs.coverage.as_ref().unwrap();
                            let alpha = &coverage[start..start + patch.width];
                            for ((out, &value), &a) in dst.iter_mut().zip(src).zip(alpha) {
                                let a = f32::from(a) / 255.0;
                                *out = *out * (1.0 - a) + value as f32 * glyph_step * a;
                            }
                        } else if patch.add {
                            let start = ay * atlases.glyphs.width + ax;
                            let src = &atlases.glyphs.samples[start..start + patch.width];
                            for (out, &value) in dst.iter_mut().zip(src) {
                                *out += value as f32 * glyph_step;
                            }
                        } else {
                            let src = atlases.tiles.as_ref().unwrap().row(ay);
                            dst.copy_from_slice(&src[ax..ax + patch.width]);
                        }
                    }
                }
            }
        }
        if let Some(layer) = self.additive {
            assert_eq!(output.len(), layer.len());
            for (out, &value) in output.iter_mut().zip(layer) {
                *out += value;
            }
        }
        if let Some(overlay) = self.overlay {
            overlay.blend(output);
        }
    }
}

pub(crate) fn fit(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    mut source_y: Cow<'_, Plane<f32>>,
    layers: LumaLayers<'_>,
    reconstruction: &Image3F,
    blocks: &FrameBlocks,
    distance: f32,
    baseline: Choice,
) -> Choice {
    let (width, height) = (source_y.xsize(), source_y.ysize());
    let has_layers = !layers.is_empty();
    let composite_source;
    let reference_y = match (&mut source_y, has_layers) {
        (Cow::Owned(y), true) => {
            layers.compose_in_place(width, y.as_mut_slice());
            y.as_slice()
        }
        (Cow::Borrowed(y), true) => {
            composite_source = layers.compose(y.as_slice(), width);
            composite_source.as_slice()
        }
        (y, false) => y.as_slice(),
    };
    let reference = crate::epf_proxy::LumaProxy::new(reference_y, width, height, &ctx.thread_pool);
    let mut work = crate::epf_proxy::Scratch::default();
    let make = |choice: Choice| {
        LoopFilter::new(
            choice.iters,
            choice.pass0,
            choice.quant_mul,
            ctx.epf_channel_scale,
        )
    };
    let mut epf = Epf::new(&ctx.thread_pool, reconstruction, blocks);
    let mut unfiltered_score = None;
    choose(distance, baseline, |choice| {
        let unfiltered = choice.iters == 0
            || std::ptr::eq(
                epf.apply(&ctx.thread_pool, scratch, &make(choice), choice.sharpness),
                reconstruction,
            );
        if unfiltered && let Some(score) = unfiltered_score {
            return score;
        }
        let composite;
        let candidate_y = if unfiltered {
            let y = reconstruction.plane_data(1);
            if has_layers {
                composite = layers.compose(y, width);
                composite.as_slice()
            } else {
                y
            }
        } else {
            // The next active render overwrites every pixel from the base,
            // so this trial can use the renderer's luma as composition scratch.
            let y = epf.output.as_mut().unwrap().plane_mut(1).as_mut_slice();
            layers.compose_in_place(width, y);
            y
        };
        let score = reference.compare(candidate_y, &mut work).0;
        if unfiltered {
            unfiltered_score = Some(score);
        }
        score
    })
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    fn additive_scene() -> (Plane<f32>, Image3F, FrameBlocks, Vec<f32>) {
        let (w, h) = (32usize, 24usize);
        let mut source = Plane::new(w, h);
        let mut recon = Image3F::new(w, h);
        let mut layer = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let value = 0.2 + 0.003 * ((x * 7 + y * 3) % 23) as f32;
                source.row_mut(y)[x] = value;
                recon.plane_row_mut(1, y)[x] = value + 0.004 * ((x + y) % 3) as f32;
                recon.plane_row_mut(2, y)[x] = value;
                layer[y * w + x] = 0.045 * (-0.5 * ((x as f32 - 15.3) / 0.8).powi(2)).exp();
            }
        }
        let blocks = FrameBlocks {
            xsize: w.div_ceil(8),
            ysize: h.div_ceil(8),
            sigma_quant: vec![-1.0; w.div_ceil(8) * h.div_ceil(8)],
        };
        (source, recon, blocks, layer)
    }

    fn additive_baseline() -> Choice {
        Choice {
            iters: 2,
            quant_mul: None,
            pass0: None,
            sharpness: 5,
        }
    }

    #[test]
    fn luma_composite_lays_the_overlay_over_the_base_last() {
        // An ink overlay (half covered, then fully covered) over a base that
        // also carries an additive layer: the overlay comes last.
        let overlay = EpfOverlay::Ink {
            indices: vec![0, 1, 2],
            coverage: vec![255, 128, 255],
            luma: vec![0.0, 0.2, 0.1],
        };
        let layers = LumaLayers {
            additive: Some(&[0.1, 0.1, 0.1]),
            overlay: Some(&overlay),
            ..LumaLayers::default()
        };
        let out = layers.compose(&[0.4, 0.4, 0.4], 3);
        let half = 128.0 / 255.0;
        assert_eq!(out, [0.5, 0.5 * (1.0 - half) + 0.2 * half, 0.1]);
    }

    #[test]
    fn crop_overlay_matches_float_blend_including_signed_zero() {
        let coverage = vec![0, 0, 1, 1];
        let mut output = [0.4, -0.0, -0.8, -0.0];
        let expected = output.map(f32::to_bits);
        let overlay = EpfOverlay::Crops { coverage };
        overlay.blend(&mut output);
        assert_eq!(output.map(f32::to_bits), [expected[0], 0, 0, 0]);
    }

    #[test]
    fn luma_composite_blends_glyph_templates_by_their_coverage() {
        use crate::patches::{GlyphAtlasLuma, MODULAR_PATCH_REF_ID, PatchReference};

        // A blend-above template: color Y = 51 lattice steps, coverage 1 and 0.5.
        let template = PatchReference {
            atlas_x: 0,
            atlas_y: 0,
            width: 2,
            height: 1,
            ref_frame: MODULAR_PATCH_REF_ID,
            alpha_only: false,
            add: false,
            mul: true,
            positions: vec![(1, 0)],
        };
        let atlases = crate::patches::EpfPatchAtlases {
            glyphs: std::sync::Arc::new(GlyphAtlasLuma {
                width: 2,
                samples: vec![51, 51],
                coverage: Some(vec![255, 128]),
            }),
            tiles: None,
        };
        let layers = LumaLayers {
            patches: Some((std::slice::from_ref(&template), &atlases)),
            ..LumaLayers::default()
        };
        let out = layers.compose(&[0.5, 0.5, 0.5], 3);
        let color = 51.0 / crate::quant_weights::INV_DC_QUANT[1];
        let half = 128.0 / 255.0;
        assert_eq!(out[0], 0.5);
        assert!((out[1] - color).abs() < 1e-6);
        assert!((out[2] - (0.5 * (1.0 - half) + color * half)).abs() < 1e-6);
    }

    #[test]
    fn luma_composite_replaces_tiles_before_adding_glyphs_and_splines() {
        use crate::patches::{GlyphAtlasLuma, MODULAR_PATCH_REF_ID, PATCH_REF_ID, PatchReference};

        let tile = PatchReference {
            atlas_x: 1,
            atlas_y: 0,
            width: 2,
            height: 1,
            ref_frame: PATCH_REF_ID,
            alpha_only: false,
            add: false,
            mul: false,
            positions: vec![(0, 1)],
        };
        let glyph = PatchReference {
            atlas_x: 1,
            atlas_y: 1,
            width: 2,
            height: 1,
            ref_frame: MODULAR_PATCH_REF_ID,
            alpha_only: false,
            add: true,
            mul: false,
            positions: vec![(1, 0), (1, 1)],
        };
        let references = [
            tile.clone(),
            glyph,
            PatchReference {
                alpha_only: true,
                positions: vec![(0, 0)],
                ..tile
            },
        ];
        let mut tiles = Plane::new(3, 1);
        tiles.row_mut(0).copy_from_slice(&[0.2, 0.4, 0.6]);
        let atlases = crate::patches::EpfPatchAtlases {
            glyphs: std::sync::Arc::new(GlyphAtlasLuma {
                width: 3,
                samples: vec![0, 0, 0, 0, 9, -4],
                coverage: None,
            }),
            tiles: Some(tiles),
        };
        let layers = LumaLayers {
            additive: Some(&[0.0, 0.0, 0.0, 0.05, 0.01, 0.0]),
            patches: Some((&references, &atlases)),
            overlay: None,
        };
        let output = layers.compose(&[0.1; 6], 3);
        let step = 1.0 / crate::quant_weights::INV_DC_QUANT[1];
        assert_eq!(
            output,
            [
                0.1,
                0.1 + 9.0 * step,
                0.1 - 4.0 * step,
                0.4 + 0.05,
                0.6 + 9.0 * step + 0.01,
                0.1 - 4.0 * step,
            ]
        );
    }

    #[test]
    fn no_additive_luma_matches_zero_layer_choice() {
        let ctx = EncodingContext::default();
        let (source, recon, blocks, _) = additive_scene();
        let mut scratch = CoderScratch::default();
        let none = fit(
            &ctx,
            &mut scratch,
            Cow::Borrowed(&source),
            LumaLayers::default(),
            &recon,
            &blocks,
            2.0,
            additive_baseline(),
        );
        let zero = vec![0.0; source.as_slice().len()];
        let with_zero = fit(
            &ctx,
            &mut scratch,
            Cow::Borrowed(&source),
            LumaLayers {
                additive: Some(&zero),
                ..LumaLayers::default()
            },
            &recon,
            &blocks,
            2.0,
            additive_baseline(),
        );
        assert_eq!(none, with_zero);
    }

    #[test]
    fn additive_luma_is_scored_after_filtering_the_base() {
        let ctx = EncodingContext::default();
        let (source, recon, blocks, layer) = additive_scene();
        let mut scratch = CoderScratch::default();
        let actual = fit(
            &ctx,
            &mut scratch,
            Cow::Borrowed(&source),
            LumaLayers {
                additive: Some(&layer),
                ..LumaLayers::default()
            },
            &recon,
            &blocks,
            2.0,
            additive_baseline(),
        );
        let full_source: Vec<f32> = source
            .as_slice()
            .iter()
            .zip(&layer)
            .map(|(&base, &layer)| base + layer)
            .collect();
        let reference = crate::epf_proxy::LumaProxy::new(
            &full_source,
            source.xsize(),
            source.ysize(),
            &ctx.thread_pool,
        );
        let mut work = crate::epf_proxy::Scratch::default();
        let mut epf = Epf::new(&ctx.thread_pool, &recon, &blocks);
        let expected = choose(2.0, additive_baseline(), |choice| {
            let filtered = if choice.iters == 0 {
                &recon
            } else {
                epf.apply(
                    &ctx.thread_pool,
                    &mut scratch,
                    &LoopFilter::new(
                        choice.iters,
                        choice.pass0,
                        choice.quant_mul,
                        ctx.epf_channel_scale,
                    ),
                    choice.sharpness,
                )
            };
            let composite: Vec<f32> = filtered
                .plane_data(1)
                .iter()
                .zip(&layer)
                .map(|(&base, &layer)| base + layer)
                .collect();
            reference.compare(&composite, &mut work).0
        });
        assert_eq!(actual, expected);

        // Filtering the narrow additive line would change the decoder result.
        // This detects an accidental pre-filter composition of the two layers.
        let filter = LoopFilter::new(2, None, Some(1.8), ctx.epf_channel_scale);
        let base_filtered = epf
            .apply(&ctx.thread_pool, &mut scratch, &filter, 7)
            .clone();
        let mut precomposed = recon.clone();
        for y in 0..source.ysize() {
            for x in 0..source.xsize() {
                precomposed.plane_row_mut(1, y)[x] += layer[y * source.xsize() + x];
            }
        }
        let mut incorrect = Epf::new(&ctx.thread_pool, &precomposed, &blocks);
        let filtered_layer = incorrect.apply(&ctx.thread_pool, &mut scratch, &filter, 7);
        let largest_difference = base_filtered
            .plane_data(1)
            .iter()
            .zip(&layer)
            .zip(filtered_layer.plane_data(1))
            .map(|((&base, &layer), &wrong)| (base + layer - wrong).abs())
            .fold(0.0f32, f32::max);
        assert!(
            largest_difference > 1e-4,
            "test layer did not distinguish the pre-filter route"
        );
    }

    #[test]
    fn reused_renderers_match_fresh_stages_across_strengths_and_worker_counts() {
        let (w, h) = (35usize, 27usize);
        let mut image = Image3F::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    image.plane_row_mut(c, y)[x] = 0.2 + 0.003 * ((x * 7 + y * 3 + c) % 23) as f32;
                }
            }
        }
        let blocks = FrameBlocks {
            xsize: w.div_ceil(8),
            ysize: h.div_ceil(8),
            sigma_quant: vec![-1.; w.div_ceil(8) * h.div_ceil(8)],
        };
        let mut expected = Vec::new();
        for threads in [1, 4] {
            let pool = ThreadPool::new(threads);
            let mut scratch = CoderScratch::default();
            let mut results = Vec::new();
            for long in [false, true] {
                let mut reused = Epf::new(&pool, &image, &blocks);
                for (iters, strength, pass0) in if long {
                    [(3, 1., Some(3.6)), (3, 0.5, Some(0.9)), (3, 2., Some(0.9))]
                } else {
                    [(2, 1., None), (1, 0.5, None), (2, 2., None)]
                } {
                    let lf = LoopFilter::new(iters, pass0, Some(0.46 * strength), None);
                    let actual = reused.apply(&pool, &mut scratch, &lf, 7).clone();
                    let mut fresh = Epf::new(&pool, &image, &blocks);
                    let reference = fresh.apply(&pool, &mut scratch, &lf, 7);
                    for c in 0..3 {
                        assert_eq!(actual.plane_data(c), reference.plane_data(c));
                        results.extend_from_slice(actual.plane_data(c));
                    }
                }
            }
            if threads == 1 {
                expected = results;
            } else {
                assert_eq!(results, expected);
            }
        }
    }

    #[test]
    fn proxy_only_search_recovers_a_continuous_minimum_within_budget() {
        let baseline = Choice {
            iters: 2,
            quant_mul: None,
            pass0: None,
            sharpness: 5,
        };
        let mut count = 0;
        let winner = choose(2., baseline, |choice| {
            count += 1;
            match choice.quant_mul {
                None => 10.,
                Some(q) => {
                    (choice.iters.abs_diff(2) as f32).powi(2) + (q / 0.46 / 0.8).log2().powi(2)
                }
            }
        });
        assert_eq!(winner.iters, 2);
        assert!((winner.quant_mul.unwrap() / 0.46 - 0.8).abs() < 1e-5);
        assert!(count <= 9);
    }

    #[test]
    fn baseline_survives_ties_and_nonfinite_candidates() {
        let baseline = Choice {
            iters: 2,
            quant_mul: None,
            pass0: None,
            sharpness: 5,
        };
        assert_eq!(choose(24., baseline, |_| 1.), baseline);
        assert_eq!(
            choose(1., baseline, |c| if c == baseline { 1. } else { f32::NAN }),
            baseline
        );
    }
}
