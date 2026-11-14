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

use super::dc_rdo::{LearnedCost, LearnedDcRates, RefPlane, learned_row_bits};
use super::dc_smoothing::{dc_value_steps, decoded_cell, decoded_dc, dequantized_dc};
use super::quantization::DistanceParams;
use super::{ImageDim, K_BLOCK_DIM, K_DC_GROUP_DIM};
use crate::coder_scratch::CoderScratch;
use crate::color_correlation::{CflFrame, dc_cfl_factor, dc_cfl_factor_x};
use crate::cvvdp_arbiter::{CvvdpArbiter, CvvdpCheckpoint};
use crate::dc_group_data::{AcStrategyImage, DcGroupData};
use crate::encoding_context::EncodingContext;
use crate::image::{Image3F, Image3S};
use crate::ssimulacra2::{Rect, Ssimulacra2Checkpoint, Ssimulacra2Reference};
use crate::util::EncodeError;
use crate::weighted_predictor::WpState;

/// Moves costing more bits than this are not verified.
const MAX_COST_BITS: f64 = 8.0;
/// Largest distance of a moved level from the unrounded DC, in steps.
const MAX_ERROR: f32 = 1.0;
/// Cells per tile side. A tile's moves are judged on its own crop of the
/// frame, without the other tiles' moves: the cost and memory of a trial do
/// not grow with the frame, and scheduling cannot affect decisions.
const TILE_CELLS: usize = 32;
/// Pixels around a moved cell whose metrics are refreshed. The image of a
/// cell of a large transform is concentrated around it; all of it enters
/// the reconstruction, this window of it is judged.
const JUDGE_REACH: usize = 24;
/// Cells of context around a tile in its crop.
const MARGIN_CELLS: usize = 8;

/// The exchange rates the quant-field loop's coefficient polish judged by.
#[derive(Default, Clone, Copy)]
pub(super) struct Handoff {
    /// SSIMULACRA2 deficit per bit at the frame's exchange rate.
    pub(super) lambda_bits: f32,
    /// The same for ColorVideoVDP.
    pub(super) cv_lambda: f32,
}

impl Handoff {
    /// Both judges have a usable exchange rate.
    pub(crate) fn ready(&self) -> bool {
        let usable = |v: f32| v.is_finite() && v > 0.0;
        usable(self.lambda_bits) && usable(self.cv_lambda)
    }
}

/// Modular channel of each XYB channel (Y, X, B are coded in that order).
static MCHAN_OF: [usize; 3] = [1, 0, 2];
static CHAN_OF: [usize; 3] = [1, 0, 2];

struct Transform {
    fbx: usize,
    fby: usize,
    raw: u8,
}

/// Columns before and after a moved cell whose bits are re-read. Neighbors
/// and reference properties reach `x - 2..=x + 2`; the rest follows the
/// weighted predictor's error feedback along the row.
const PRICE_BEFORE: usize = 3;
const PRICE_AFTER: usize = 6;
/// Rows a committed move is followed past its reach while the predictor
/// state has not rejoined the stored one.
const COMMIT_EXTRA_ROWS: usize = 4;

/// The learned rate model's state of one DC group.
struct GroupRates {
    w: usize,
    h: usize,
    stream: i32,
    /// X and Y levels as the reference properties read them.
    wide: [Vec<i32>; 2],
    /// Rows `row0..row1` are held.
    row0: usize,
    row1: usize,
    /// Per modular channel: predictor state at the start of each held row,
    /// and after the last.
    snaps: [Vec<WpState>; 3],
    cell_bits: [Vec<f32>; 3],
}

impl GroupRates {
    fn refs(&self, m: usize) -> ([RefPlane<'_>; 2], usize) {
        let x = RefPlane {
            pixels: &self.wide[0],
            width: self.w,
        };
        let y = RefPlane {
            pixels: &self.wide[1],
            width: self.w,
        };
        match m {
            0 => ([x, y], 0),
            // X references Y; B references X, then Y.
            1 => ([y, x], 1),
            _ => ([x, y], 2),
        }
    }

    /// Rows of modular channel `m` the level at row `y` of the moved plane
    /// can reach through neighbors (own plane) or reference properties.
    fn span(&self, m: usize, moved: usize, y: usize) -> std::ops::Range<usize> {
        y..(y + if m == moved { 3 } else { 2 }).min(self.h)
    }

    /// Bit change of the group's DC after the level at (`x`, `y`) of modular
    /// channel `moved` changed (`levels` and `wide` hold the moved level):
    /// a window of columns in every row the move reaches, the predictor
    /// carried from row to row with the unchanged cells' records around it.
    #[allow(clippy::too_many_arguments)]
    fn price(
        &self,
        levels: &Image3S,
        moved: usize,
        x: usize,
        y: usize,
        rates: &LearnedDcRates<'_>,
        cost: &LearnedCost,
        wp: &mut WpState,
    ) -> f64 {
        let cols = x.saturating_sub(PRICE_BEFORE)..(x + PRICE_AFTER).min(self.w);
        let mut delta = 0.0f64;
        for m in moved..3 {
            let (refs, n) = self.refs(m);
            let snaps = &self.snaps[m];
            wp.clone_from(&snaps[y - self.row0]);
            for row in self.span(m, moved, y) {
                let next = &snaps[row + 1 - self.row0];
                wp.enter_row(next, row, cols.start);
                learned_row_bits(
                    levels.plane(CHAN_OF[m]).as_slice(),
                    self.w,
                    m as i32,
                    self.stream,
                    &refs[..n],
                    rates,
                    cost,
                    wp,
                    row,
                    cols.clone(),
                    |_, bits| delta += f64::from(bits),
                );
                wp.leave_row(next, row, cols.end);
                delta -= self.cell_bits[m][(row - self.row0) * self.w..][cols.clone()]
                    .iter()
                    .map(|&b| f64::from(b))
                    .sum::<f64>();
            }
        }
        delta
    }

    /// Refresh whole rows from row `first` of every plane from modular
    /// channel `moved` on. With `rejoin`, stop where the predictor state
    /// meets the stored one past the rows the move reaches, or a few rows on.
    fn walk(
        &mut self,
        levels: &Image3S,
        moved: usize,
        first: usize,
        rejoin: bool,
        rates: &LearnedDcRates<'_>,
        cost: &LearnedCost,
    ) {
        let (w, row0) = (self.w, self.row0);
        for m in moved..3 {
            let must = self.span(m, moved, first).end;
            let last = if rejoin {
                (must + COMMIT_EXTRA_ROWS).min(self.row1)
            } else {
                self.row1
            };
            let mut snaps = std::mem::take(&mut self.snaps[m]);
            let mut cell_bits = std::mem::take(&mut self.cell_bits[m]);
            let mut wp = snaps[first - row0].clone();
            let (refs, n) = self.refs(m);
            for row in first..last {
                let out = &mut cell_bits[(row - row0) * w..(row + 1 - row0) * w];
                learned_row_bits(
                    levels.plane(CHAN_OF[m]).as_slice(),
                    w,
                    m as i32,
                    self.stream,
                    &refs[..n],
                    rates,
                    cost,
                    &mut wp,
                    row,
                    0..w,
                    |x, bits| out[x] = bits,
                );
                if rejoin && row + 1 >= must && snaps[row + 1 - row0] == wp {
                    break;
                }
                snaps[row + 1 - row0].clone_from(&wp);
            }
            self.snaps[m] = snaps;
            self.cell_bits[m] = cell_bits;
        }
    }

    fn new(levels: &Image3S, stream: i32, rates: &LearnedDcRates<'_>, cost: &LearnedCost) -> Self {
        let (w, h) = (levels.xsize(), levels.ysize());
        let wide = |c: usize| -> Vec<i32> {
            levels
                .plane(c)
                .as_slice()
                .iter()
                .map(|&v| i32::from(v))
                .collect()
        };
        let mut this = Self {
            w,
            h,
            stream,
            wide: [wide(0), wide(1)],
            row0: 0,
            row1: h,
            snaps: std::array::from_fn(|_| vec![WpState::with_params(w, rates.wp_params); h + 1]),
            cell_bits: std::array::from_fn(|_| vec![0.0; w * h]),
        };
        this.walk(levels, 0, 0, false, rates, cost);
        this
    }

    /// The state a tile over the rows `rows` works on: those rows and the
    /// ones its moves reach.
    fn window(&self, rows: std::ops::Range<usize>) -> Self {
        let row1 = (rows.end + 2 + COMMIT_EXTRA_ROWS).min(self.h);
        let held = rows.start..row1;
        Self {
            w: self.w,
            h: self.h,
            stream: self.stream,
            wide: self.wide.clone(),
            row0: held.start,
            row1,
            snaps: std::array::from_fn(|m| self.snaps[m][held.start..=held.end].to_vec()),
            cell_bits: std::array::from_fn(|m| {
                self.cell_bits[m][held.start * self.w..held.end * self.w].to_vec()
            }),
        }
    }
}

/// The pixel image of one trial move.
#[derive(Default)]
struct Patch {
    /// Transform index, offset of its three channel images in `arena` and
    /// which of them changed.
    parts: Vec<(usize, usize, [bool; 3])>,
    arena: Vec<f32>,
    /// Per part and channel: the changed cells' slots and deltas.
    deltas: Vec<Vec<(usize, f32)>>,
    /// Changed DC cells with their new decoded values.
    cells: Vec<(usize, usize, [f32; 3])>,
}

/// Frame geometry and models shared by the bands.
struct Frame<'a> {
    ctx: &'a EncodingContext,
    source_dc: &'a Image3F,
    cell_transform: Vec<u32>,
    transforms: Vec<Transform>,
    steps: [f32; 3],
    factors: [f32; 2],
    skip_smoothing: bool,
    /// DC cells of the frame.
    cells: (usize, usize),
}

/// The rows `y0..y1` and columns `x0..x1` of `image`.
fn crop(image: &Image3F, x0: usize, y0: usize, x1: usize, y1: usize) -> Image3F {
    let mut out = Image3F::new(x1 - x0, y1 - y0);
    for c in 0..3 {
        for y in y0..y1 {
            out.plane_row_mut(c, y - y0)
                .copy_from_slice(&image.plane_row(c, y)[x0..x1]);
        }
    }
    out
}

/// One tile's crop of the DC planes and of the reconstruction.
struct Pixels {
    /// First DC cell of the crop; its first pixel is eight times that.
    cell0: (usize, usize),
    /// Dequantized DC before smoothing, and as the transforms receive it.
    raw: Image3F,
    dec: Image3F,
    recon: Image3F,
}

impl Pixels {
    /// A transform's pixels inside the crop: crop-local origin, the offset
    /// of that origin inside the transform, and the size.
    #[allow(clippy::type_complexity)]
    fn clip(&self, tf: &Transform) -> Option<((usize, usize), (usize, usize), (usize, usize))> {
        let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(tf.raw);
        let (cx, cy) = (self.cell0.0 * K_BLOCK_DIM, self.cell0.1 * K_BLOCK_DIM);
        let (tx, ty) = (tf.fbx * K_BLOCK_DIM, tf.fby * K_BLOCK_DIM);
        let (x0, y0) = (tx.max(cx), ty.max(cy));
        let x1 = (tx + pw).min(cx + self.recon.xsize());
        let y1 = (ty + ph).min(cy + self.recon.ysize());
        (x1 > x0 && y1 > y0).then(|| ((x0 - cx, y0 - cy), (x0 - tx, y0 - ty), (x1 - x0, y1 - y0)))
    }

    /// Fill `patch` for the dequantized values already written at frame cell
    /// (`fx`, `fy`) of `raw`. Returns the crop rectangle it covers.
    fn prepare(&self, frame: &Frame<'_>, fx: usize, fy: usize, patch: &mut Patch) -> Option<Rect> {
        let add_scaled = frame.ctx.add_scaled;
        let (bw, bh) = frame.cells;
        let (dx, dy) = self.cell0;
        patch.parts.clear();
        patch.cells.clear();
        let reach = usize::from(!frame.skip_smoothing);
        let mut rect: Option<Rect> = None;
        for y in fy.saturating_sub(reach)..(fy + reach + 1).min(bh) {
            for x in fx.saturating_sub(reach)..(fx + reach + 1).min(bw) {
                let value =
                    decoded_cell(&self.raw, x - dx, y - dy, frame.steps, frame.skip_smoothing);
                let delta: [f32; 3] =
                    std::array::from_fn(|c| value[c] - self.dec.plane_row(c, y - dy)[x - dx]);
                if delta == [0.0; 3] {
                    continue;
                }
                patch.cells.push((x, y, value));
                let t = frame.cell_transform[y * bw + x] as usize;
                let part = match patch.parts.iter().position(|&(index, ..)| index == t) {
                    Some(part) => part,
                    None => {
                        patch.parts.push((t, 0, [false; 3]));
                        let part = patch.parts.len() - 1;
                        if patch.deltas.len() < 3 * (part + 1) {
                            patch.deltas.resize_with(3 * (part + 1), Vec::new);
                        }
                        for c in 0..3 {
                            patch.deltas[part * 3 + c].clear();
                        }
                        part
                    }
                };
                let tf = &frame.transforms[t];
                let cov_x = AcStrategyImage::covered_blocks_x_of(tf.raw);
                let slot = (y - tf.fby) * cov_x + (x - tf.fbx);
                for c in 0..3 {
                    if delta[c] != 0.0 {
                        patch.deltas[part * 3 + c].push((slot, delta[c]));
                    }
                }
            }
        }
        patch.arena.clear();
        for part in 0..patch.parts.len() {
            let tf = &frame.transforms[patch.parts[part].0];
            let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(tf.raw);
            let n = pw * ph;
            let offset = patch.arena.len();
            patch.parts[part].1 = offset;
            let Some(((x0, y0), _, (w, h))) = self.clip(tf) else {
                continue;
            };
            patch.arena.resize(offset + 3 * n, 0.0);
            let images = crate::epf_filter::dc_cell_images(frame.ctx, tf.raw);
            for c in 0..3 {
                let Some((&(slot, delta), rest)) = patch.deltas[part * 3 + c].split_first() else {
                    continue;
                };
                let out = &mut patch.arena[offset + c * n..offset + (c + 1) * n];
                let image = &images[slot * n..(slot + 1) * n];
                // Assign the first contribution directly to preserve signed zero.
                for (out, &pixel) in out.iter_mut().zip(image) {
                    *out = delta * pixel;
                }
                for &(slot, delta) in rest {
                    add_scaled(out, &images[slot * n..(slot + 1) * n], delta);
                }
                patch.parts[part].2[c] = true;
            }
            let r = Rect::new(x0, y0, x0 + w, y0 + h);
            rect = Some(match rect {
                None => r,
                Some(a) => Rect::new(
                    a.x0.min(r.x0),
                    a.y0.min(r.y0),
                    a.x1.max(r.x1),
                    a.y1.max(r.y1),
                ),
            });
        }
        // The image of a moved cell is concentrated around it: judge that
        // window of a large transform, not all of it.
        let reach = JUDGE_REACH;
        let (mut fx0, mut fy0, mut fx1, mut fy1) = (usize::MAX, usize::MAX, 0, 0);
        for &(x, y, _) in &patch.cells {
            fx0 = fx0.min(((x - dx) * K_BLOCK_DIM).saturating_sub(reach));
            fy0 = fy0.min(((y - dy) * K_BLOCK_DIM).saturating_sub(reach));
            fx1 = fx1.max((x - dx + 1) * K_BLOCK_DIM + reach);
            fy1 = fy1.max((y - dy + 1) * K_BLOCK_DIM + reach);
        }
        rect.map(|r| Rect::new(r.x0.max(fx0), r.y0.max(fy0), r.x1.min(fx1), r.y1.min(fy1)))
            .filter(|r| r.x1 > r.x0 && r.y1 > r.y0)
    }

    fn apply(&mut self, frame: &Frame<'_>, patch: &Patch, sign: f32) {
        let add_scaled = frame.ctx.add_scaled;
        for &(t, offset, changed) in &patch.parts {
            let tf = &frame.transforms[t];
            let (pw, ph) = crate::inflated_cost::strategy_pixel_dims(tf.raw);
            let Some(((x0, y0), (ix, iy), (w, h))) = self.clip(tf) else {
                continue;
            };
            for c in (0..3).filter(|&c| changed[c]) {
                let image = &patch.arena[offset + c * pw * ph..][..pw * ph];
                for y in 0..h {
                    let row = &mut self.recon.plane_row_mut(c, y0 + y)[x0..x0 + w];
                    let from = &image[(iy + y) * pw + ix..][..w];
                    add_scaled(row, from, sign);
                }
            }
        }
    }
}

/// An accepted level: group, channel, cell and value.
type Move = (usize, usize, usize, usize, i16);

/// The frame before the pass: DC planes and reconstruction with DC error.
struct Baseline {
    raw: Image3F,
    dec: Image3F,
    recon: Image3F,
}

/// Polish the cells `tx` x `ty` of group `g` from the shared baseline, on a
/// crop of the surrounding frame. Returns the accepted levels in order.
#[allow(clippy::too_many_arguments)]
fn polish_tile(
    frame: &Frame<'_>,
    source: &Image3F,
    groups: &[DcGroupData],
    coords: &[(usize, usize)],
    baseline: &Baseline,
    base_rates: &[GroupRates],
    lambdas: (f64, f64),
    rates: &LearnedDcRates<'_>,
    cost: &LearnedCost,
    (g, tx, ty): (usize, std::ops::Range<usize>, std::ops::Range<usize>),
) -> Vec<Move> {
    const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
    let reconstruct = frame.ctx.dc_rows.reconstruct;
    let (ox, oy) = (coords[g].0 * DC_GROUP_BLOCKS, coords[g].1 * DC_GROUP_BLOCKS);
    let (bw, bh) = frame.cells;
    let cell0 = (
        (ox + tx.start).saturating_sub(MARGIN_CELLS),
        (oy + ty.start).saturating_sub(MARGIN_CELLS),
    );
    let cell1 = (
        (ox + tx.end + MARGIN_CELLS).min(bw),
        (oy + ty.end + MARGIN_CELLS).min(bh),
    );
    let pixel0 = (cell0.0 * K_BLOCK_DIM, cell0.1 * K_BLOCK_DIM);
    let pixel1 = (
        (cell1.0 * K_BLOCK_DIM).min(source.xsize()),
        (cell1.1 * K_BLOCK_DIM).min(source.ysize()),
    );
    let (reference, cv_reference) = {
        let source = crop(source, pixel0.0, pixel0.1, pixel1.0, pixel1.1);
        let Some(judges) = Ssimulacra2Reference::new(&source).zip(CvvdpArbiter::prepare(&source))
        else {
            return Vec::new();
        };
        judges
    };
    let mut pixels = Pixels {
        cell0,
        raw: crop(&baseline.raw, cell0.0, cell0.1, cell1.0, cell1.1),
        dec: crop(&baseline.dec, cell0.0, cell0.1, cell1.0, cell1.1),
        recon: crop(&baseline.recon, pixel0.0, pixel0.1, pixel1.0, pixel1.1),
    };
    let Some(mut judge) = CvvdpArbiter::new_shared(&pixels.recon, cv_reference) else {
        return Vec::new();
    };
    // The reconstruction carries the DC error itself.
    let no_dc_error = Image3F::new(cell1.0 - cell0.0, cell1.1 - cell0.1);
    let mut state = reference.state(&pixels.recon);
    judge.reset(&pixels.recon, &no_dc_error);
    // Both deficits are means over the judged pixels: a change weighs more
    // in the crop than in the frame the exchange rates were measured on.
    let share = (source.xsize() * source.ysize()) as f64
        / (pixels.recon.xsize() * pixels.recon.ysize()) as f64;
    let (lambda, cv_lambda) = (lambdas.0 * share, lambdas.1 * share);
    let mut base_d = state.deficit64();
    let mut cv_base = f64::from(judge.deficit());
    let mut cp = Ssimulacra2Checkpoint::default();
    let mut cv_cp = CvvdpCheckpoint::default();
    let mut patch = Patch::default();
    let mut wp = WpState::new(1);
    let mut moves = Vec::new();
    let w = groups[g].quant_dc.xsize();
    {
        let mut levels = groups[g].quant_dc.clone();
        let mut group_rates = base_rates[g].window(ty.clone());
        for ly in ty {
            for lx in tx.clone() {
                let (fx, fy) = (ox + lx, oy + ly);
                for c in [1usize, 0, 2] {
                    for dir in [-1i16, 1] {
                        let old = levels.plane_row(c, ly)[lx];
                        let Some(new) = old.checked_add(dir) else {
                            continue;
                        };
                        let cell: [[i16; 1]; 3] = std::array::from_fn(|k| {
                            [if k == c {
                                new
                            } else {
                                levels.plane_row(k, ly)[lx]
                            }]
                        });
                        let mut value = [[0.0f32; 1]; 3];
                        {
                            let [vx, vy, vb] = &mut value;
                            reconstruct(
                                [&cell[0], &cell[1], &cell[2]],
                                frame.factors,
                                frame.steps,
                                [vx, vy, vb],
                            );
                        }
                        let error = (value[c][0] - frame.source_dc.plane_row(c, fy)[fx]).abs()
                            / frame.steps[c];
                        if error > MAX_ERROR {
                            continue;
                        }
                        let set_level =
                            |levels: &mut Image3S, group_rates: &mut GroupRates, level: i16| {
                                levels.plane_row_mut(c, ly)[lx] = level;
                                if c < 2 {
                                    group_rates.wide[c][ly * w + lx] = i32::from(level);
                                }
                            };
                        set_level(&mut levels, &mut group_rates, new);
                        let m = MCHAN_OF[c];
                        let dbits = group_rates.price(&levels, m, lx, ly, rates, cost, &mut wp);
                        if dbits > MAX_COST_BITS {
                            set_level(&mut levels, &mut group_rates, old);
                            continue;
                        }
                        let saved: [f32; 3] = std::array::from_fn(|k| {
                            pixels.raw.plane_row(k, fy - cell0.1)[fx - cell0.0]
                        });
                        for k in 0..3 {
                            pixels.raw.plane_row_mut(k, fy - cell0.1)[fx - cell0.0] = value[k][0];
                        }
                        let mut accept = false;
                        let rect = pixels.prepare(frame, fx, fy, &mut patch);
                        if let Some(rect) = rect {
                            pixels.apply(frame, &patch, 1.0);
                            state.update_checkpointed(&pixels.recon, rect, &mut cp);
                            let d = state.deficit64();
                            // ColorVideoVDP only sees what SSIMULACRA2 accepts.
                            if d - base_d + lambda * dbits < 0.0 {
                                let cv_d = f64::from(judge.update_checkpointed(
                                    &pixels.recon,
                                    &no_dc_error,
                                    rect,
                                    &mut cv_cp,
                                ));
                                if cv_d - cv_base + cv_lambda * dbits < 0.0 {
                                    accept = true;
                                    base_d = d;
                                    cv_base = cv_d;
                                } else {
                                    judge.rollback(&cv_cp);
                                }
                            }
                            if !accept {
                                state.rollback(&cp);
                                pixels.apply(frame, &patch, -1.0);
                            }
                        }
                        if accept {
                            for &(x, y, v) in &patch.cells {
                                for k in 0..3 {
                                    pixels.dec.plane_row_mut(k, y - cell0.1)[x - cell0.0] = v[k];
                                }
                            }
                            group_rates.walk(&levels, m, ly, true, rates, cost);
                            moves.push((g, c, lx, ly, new));
                            break;
                        }
                        for k in 0..3 {
                            pixels.raw.plane_row_mut(k, fy - cell0.1)[fx - cell0.0] = saved[k];
                        }
                        set_level(&mut levels, &mut group_rates, old);
                    }
                }
            }
        }
    }
    moves
}

/// Polish the DC levels of `groups` in place on `ac_recon`, the frame
/// reconstruction without its DC error. Returns whether any level changed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    handoff: Handoff,
    source: &Image3F,
    ac_recon: Image3F,
    source_dc: &Image3F,
    dim: &ImageDim,
    groups: &mut [DcGroupData],
    coords: &[(usize, usize)],
    distp: &DistanceParams,
    ytob_dc: i32,
    cfl: CflFrame,
    skip_smoothing: bool,
    rates: &LearnedDcRates<'_>,
) -> Result<bool, EncodeError> {
    const DC_GROUP_BLOCKS: usize = K_DC_GROUP_DIM / K_BLOCK_DIM;
    if !handoff.ready() {
        return Ok(false);
    }
    let Some(cost) = LearnedCost::new(rates.code) else {
        return Ok(false);
    };
    let reconstruct = ctx.dc_rows.reconstruct;
    let raw = dequantized_dc(dim, groups, coords, distp, ytob_dc, cfl, reconstruct)?;
    let dec = decoded_dc(
        dim,
        groups,
        coords,
        distp,
        ytob_dc,
        cfl,
        reconstruct,
        skip_smoothing,
    )?;
    let mut recon = ac_recon;
    crate::epf_filter::add_dc_error(ctx, scratch, &mut recon, &dec, source_dc, groups, coords);
    let (bw, bh) = (raw.xsize(), raw.ysize());
    let mut transforms = Vec::new();
    let mut cell_transform = vec![0u32; bw * bh];
    for (dc, &(gx, gy)) in groups.iter().zip(coords) {
        let (ox, oy) = (gx * DC_GROUP_BLOCKS, gy * DC_GROUP_BLOCKS);
        for (bx, by, raw_strategy) in dc.ac_strategy.iter_first_blocks() {
            let index = transforms.len() as u32;
            let (fbx, fby) = (ox + bx, oy + by);
            transforms.push(Transform {
                fbx,
                fby,
                raw: raw_strategy,
            });
            for y in fby..(fby + AcStrategyImage::covered_blocks_y_of(raw_strategy)).min(bh) {
                for x in fbx..(fbx + AcStrategyImage::covered_blocks_x_of(raw_strategy)).min(bw) {
                    cell_transform[y * bw + x] = index;
                }
            }
        }
    }
    let frame = Frame {
        ctx,
        source_dc,
        cell_transform,
        transforms,
        steps: dc_value_steps(distp),
        factors: [
            dc_cfl_factor_x(distp.dc_step, cfl.base_x),
            dc_cfl_factor(distp.dc_step, ytob_dc, cfl),
        ],
        skip_smoothing,
        cells: (bw, bh),
    };
    let lambdas = (f64::from(handoff.lambda_bits), f64::from(handoff.cv_lambda));
    let baseline = Baseline { raw, dec, recon };
    let shared: &[DcGroupData] = groups;
    let base_rates = ctx.thread_pool.steal_map(scratch, shared.len(), |g, _| {
        GroupRates::new(&shared[g].quant_dc, 1 + g as i32, rates, &cost)
    });
    let mut tiles = Vec::new();
    for (g, dc) in shared.iter().enumerate() {
        let (w, h) = (dc.quant_dc.xsize(), dc.quant_dc.ysize());
        for y in (0..h).step_by(TILE_CELLS) {
            for x in (0..w).step_by(TILE_CELLS) {
                tiles.push((g, x..(x + TILE_CELLS).min(w), y..(y + TILE_CELLS).min(h)));
            }
        }
    }
    let work = ctx.thread_pool.steal_map(scratch, tiles.len(), |tile, _| {
        polish_tile(
            &frame,
            source,
            shared,
            coords,
            &baseline,
            &base_rates,
            lambdas,
            rates,
            &cost,
            tiles[tile].clone(),
        )
    });
    let mut changed = false;
    for (g, c, lx, ly, level) in work.into_iter().flatten() {
        groups[g].quant_dc.plane_row_mut(c, ly)[lx] = level;
        changed = true;
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dc_group_data::STRATEGY_DCT32X32;

    #[test]
    fn clipped_patch_accumulation_and_rollback_match_scalar() {
        let ctx = EncodingContext::default();
        let dec = Image3F::new(4, 4);
        let frame = Frame {
            ctx: &ctx,
            source_dc: &dec,
            cell_transform: vec![0; 16],
            transforms: vec![Transform {
                fbx: 0,
                fby: 0,
                raw: STRATEGY_DCT32X32,
            }],
            steps: [1.0; 3],
            factors: [0.0; 2],
            skip_smoothing: false,
            cells: (4, 4),
        };
        let mut pixels = Pixels {
            cell0: (0, 0),
            raw: dec.clone(),
            dec: dec.clone(),
            recon: Image3F::new(29, 27),
        };
        pixels.raw.plane_row_mut(1, 1)[1] = -0.375;
        let samples = [0.0, -0.0, -0.125, 0.875];
        for (i, pixel) in pixels
            .recon
            .plane_mut(1)
            .as_mut_slice()
            .iter_mut()
            .enumerate()
        {
            *pixel = samples[i % samples.len()];
        }
        let mut patch = Patch::default();
        assert!(pixels.prepare(&frame, 1, 1, &mut patch).is_some());
        assert_eq!(patch.parts, [(0, 0, [false, true, false])]);
        assert_eq!(patch.deltas[1].len(), 4);

        const AREA: usize = 32 * 32;
        let images = crate::epf_filter::dc_cell_images(&ctx, STRATEGY_DCT32X32);
        let delta: [f32; AREA] = std::array::from_fn(|i| {
            let (&(slot, scale), rest) = patch.deltas[1].split_first().unwrap();
            rest.iter()
                .fold(scale * images[slot * AREA + i], |sum, &(slot, scale)| {
                    sum + scale * images[slot * AREA + i]
                })
        });
        let mut expected = pixels.recon.clone();
        for sign in [1.0, -1.0] {
            pixels.apply(&frame, &patch, sign);
            for y in 0..expected.ysize() {
                for (x, out) in expected.plane_row_mut(1, y).iter_mut().enumerate() {
                    *out += sign * delta[y * 32 + x];
                }
                for c in 0..3 {
                    for (&actual, &expected) in pixels
                        .recon
                        .plane_row(c, y)
                        .iter()
                        .zip(expected.plane_row(c, y))
                    {
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }
}
