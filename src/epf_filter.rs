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

pub(crate) fn fit(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    source_y: &Plane<f32>,
    reconstruction: &Image3F,
    blocks: &FrameBlocks,
    distance: f32,
    baseline: Choice,
) -> Choice {
    let reference =
        crate::epf_proxy::LumaProxy::new(source_y.as_slice(), source_y.xsize(), source_y.ysize());
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
    choose(distance, baseline, |choice| {
        let filtered = if choice.iters == 0 {
            reconstruction
        } else {
            epf.apply(&ctx.thread_pool, scratch, &make(choice), choice.sharpness)
        };
        reference.compare(filtered.plane_data(1), &mut work).0
    })
}

#[cfg(test)]
mod fit_tests {
    use super::*;

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
