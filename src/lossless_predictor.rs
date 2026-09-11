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

//! Modular predictors, cost estimation, and channel tokenization.

use super::lz77::{LzToken, RunLzWriter};
use super::{EntropyOfHistFn, selected_entropy_of_hist_fn};
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::{Token, pack_signed};
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpNeighbors, WpParams, WpState};
use std::sync::OnceLock;

const PREDICTOR_ZERO: u32 = 0;
const PREDICTOR_LEFT: u32 = 1;
const PREDICTOR_TOP: u32 = 2;
const PREDICTOR_AVERAGE0: u32 = 3;
const PREDICTOR_SELECT: u32 = 4;
pub(super) const PREDICTOR_GRADIENT: u32 = 5;
pub(super) const PREDICTOR_WEIGHTED: u32 = 6;
const PREDICTOR_TOP_RIGHT: u32 = 7;
const PREDICTOR_TOP_LEFT: u32 = 8;
const PREDICTOR_LEFT_LEFT: u32 = 9;
const PREDICTOR_AVERAGE1: u32 = 10;
const PREDICTOR_AVERAGE2: u32 = 11;
const PREDICTOR_AVERAGE3: u32 = 12;
const PREDICTOR_AVERAGE4: u32 = 13;
static SLOW_PREDICTORS: [u32; 6] = [
    PREDICTOR_WEIGHTED,
    PREDICTOR_GRADIENT,
    PREDICTOR_AVERAGE4,
    PREDICTOR_SELECT,
    PREDICTOR_LEFT,
    PREDICTOR_TOP,
];

/// Fixed-predictor fallback used wherever Fast mode hardcodes Weighted;
/// Gradient when the decoding-speed setting excludes the Weighted Predictor.
#[inline]
pub(super) fn fixed_predictor(use_wp: bool) -> u32 {
    if use_wp {
        PREDICTOR_WEIGHTED
    } else {
        PREDICTOR_GRADIENT
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(super) struct PredictorNeighbors {
    pub(super) left: i64,
    pub(super) top: i64,
    pub(super) top_left: i64,
    pub(super) top_right: i64,
    pub(super) left_left: i64,
    pub(super) top_top: i64,
    pub(super) top_right_right: i64,
}

#[inline]
pub(super) fn predictor_neighbors<F: Fn(usize, usize) -> i32 + ?Sized>(
    get: &F,
    x: usize,
    y: usize,
    width: usize,
) -> PredictorNeighbors {
    let left = if x > 0 {
        get(x - 1, y) as i64
    } else if y > 0 {
        get(x, y - 1) as i64
    } else {
        0
    };
    let top = if y > 0 { get(x, y - 1) as i64 } else { left };
    let top_left = if x > 0 && y > 0 {
        get(x - 1, y - 1) as i64
    } else {
        left
    };
    let top_right = if x + 1 < width && y > 0 {
        get(x + 1, y - 1) as i64
    } else {
        top
    };
    PredictorNeighbors {
        left,
        top,
        top_left,
        top_right,
        left_left: if x > 1 { get(x - 2, y) as i64 } else { left },
        top_top: if y > 1 { get(x, y - 2) as i64 } else { top },
        top_right_right: if x + 2 < width && y > 0 {
            get(x + 2, y - 1) as i64
        } else {
            top_right
        },
    }
}

/// Visit a row with the same border values as `predictor_neighbors`. Only the
/// first/last two columns need border handling; the interior zips valid windows.
#[inline]
fn visit_predictor_row<T: Copy>(
    current: &[T],
    north: &[T],
    north_north: &[T],
    first_row: bool,
    mut visit: impl FnMut(usize, i64, PredictorNeighbors),
) where
    i64: From<T>,
{
    let width = current.len();
    if first_row {
        let Some((&first, tail)) = current.split_first() else {
            return;
        };
        let first = i64::from(first);
        visit(
            0,
            first,
            PredictorNeighbors {
                left: 0,
                top: 0,
                top_left: 0,
                top_right: 0,
                left_left: 0,
                top_top: 0,
                top_right_right: 0,
            },
        );
        let mut left = first;
        let mut left_left = first;
        for (x, &value) in tail.iter().enumerate() {
            let value = i64::from(value);
            visit(
                x + 1,
                value,
                PredictorNeighbors {
                    left,
                    top: left,
                    top_left: left,
                    top_right: left,
                    left_left,
                    top_top: left,
                    top_right_right: left,
                },
            );
            left_left = left;
            left = value;
        }
        return;
    }
    assert_eq!(north.len(), width);
    assert_eq!(north_north.len(), width);
    let edge = |x: usize| {
        let left = if x > 0 {
            i64::from(current[x - 1])
        } else {
            i64::from(north[x])
        };
        let top = i64::from(north[x]);
        let top_right = north.get(x + 1).map_or(top, |&v| i64::from(v));
        PredictorNeighbors {
            left,
            top,
            top_left: if x > 0 { i64::from(north[x - 1]) } else { left },
            top_right,
            left_left: if x > 1 {
                i64::from(current[x - 2])
            } else {
                left
            },
            top_top: i64::from(north_north[x]),
            top_right_right: north.get(x + 2).map_or(top_right, |&v| i64::from(v)),
        }
    };
    for (x, &current) in (0..width.min(2)).zip(current.iter()) {
        visit(x, i64::from(current), edge(x));
    }
    if width > 4 {
        let mut left = i64::from(current[1]);
        let mut left_left = i64::from(current[0]);
        let pixels = current[2..width - 2]
            .iter()
            .zip(north[1..].array_windows::<4>())
            .zip(&north_north[2..]);
        for (x, ((&value, &[nw, n, ne, nee]), &nn)) in pixels.enumerate() {
            let value = i64::from(value);
            visit(
                x + 2,
                value,
                PredictorNeighbors {
                    left,
                    top: i64::from(n),
                    top_left: i64::from(nw),
                    top_right: i64::from(ne),
                    left_left,
                    top_top: i64::from(nn),
                    top_right_right: i64::from(nee),
                },
            );
            left_left = left;
            left = value;
        }
    }
    for x in width.saturating_sub(2).max(2)..width {
        visit(x, i64::from(current[x]), edge(x));
    }
}

fn visit_predictor_rows<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    w: usize,
    h: usize,
    mut visit: impl FnMut(usize, usize, i64, PredictorNeighbors),
) where
    i64: From<T>,
{
    if w == 0 || h == 0 {
        return;
    }
    let first = get_row(0);
    assert_eq!(first.len(), w);
    visit_predictor_row(first, &[], &[], true, |x, value, n| visit(x, 0, value, n));
    let mut north = first;
    let mut north_north = first;
    for y in 1..h {
        let current = get_row(y);
        assert_eq!(current.len(), w);
        visit_predictor_row(current, north, north_north, false, |x, value, n| {
            visit(x, y, value, n);
        });
        north_north = north;
        north = current;
    }
}

#[inline]
pub(super) fn predictor_value(pred_id: u32, n: PredictorNeighbors, weighted: i64) -> i64 {
    match pred_id {
        PREDICTOR_ZERO => 0,
        PREDICTOR_LEFT => n.left,
        PREDICTOR_TOP => n.top,
        PREDICTOR_AVERAGE0 => (n.left + n.top) / 2,
        PREDICTOR_SELECT => {
            let projected = n.left + n.top - n.top_left;
            if (projected - n.left).abs() < (projected - n.top).abs() {
                n.left
            } else {
                n.top
            }
        }
        PREDICTOR_GRADIENT => clamped_gradient(n.left, n.top, n.top_left),
        PREDICTOR_WEIGHTED => weighted,
        PREDICTOR_TOP_RIGHT => n.top_right,
        PREDICTOR_TOP_LEFT => n.top_left,
        PREDICTOR_LEFT_LEFT => n.left_left,
        PREDICTOR_AVERAGE1 => (n.left + n.top_left) / 2,
        PREDICTOR_AVERAGE2 => (n.top_left + n.top) / 2,
        PREDICTOR_AVERAGE3 => (n.top + n.top_right) / 2,
        PREDICTOR_AVERAGE4 => {
            (6 * n.top - 2 * n.top_top
                + 7 * n.left
                + n.left_left
                + n.top_right_right
                + 3 * n.top_right
                + 8)
                / 16
        }
        _ => unreachable!("unsupported modular predictor {pred_id}"),
    }
}

fn wp_sample_cost(
    get: impl Fn(usize, usize) -> i32,
    width: usize,
    height: usize,
    params: WpParams,
) -> (u64, usize) {
    let cw = width.min(128);
    let ch = height.min(128);
    let xs = if width > cw { [0, width - cw] } else { [0, 0] };
    let ys = if height > ch {
        [0, height - ch]
    } else {
        [0, 0]
    };
    let nx = 1 + usize::from(xs[1] != xs[0]);
    let ny = 1 + usize::from(ys[1] != ys[0]);
    let mut cost = 0u64;
    let mut count = 0usize;
    let mut wp = WpState::with_params(cw, params);
    for &y0 in &ys[..ny] {
        for &x0 in &xs[..nx] {
            let local_get = |x: usize, y: usize| get(x0 + x, y0 + y);
            wp.reset(cw, params);
            for y in 0..ch {
                for x in 0..cw {
                    let neighbors = predictor_neighbors(&local_get, x, y, cw);
                    let value = local_get(x, y);
                    let prediction = wp.predict(
                        x,
                        y,
                        neighbors.top,
                        neighbors.left,
                        neighbors.top_right,
                        neighbors.top_left,
                        neighbors.top_top,
                    );
                    let packed = pack_signed(value.wrapping_sub(prediction as i32));
                    let (_, extra_bits, _) = crate::entropy::uint_encode(packed);
                    let magnitude_bits = packed
                        .checked_add(1)
                        .map_or(32, |value| 32 - value.leading_zeros());
                    cost += extra_bits as u64 + u64::from(magnitude_bits);
                    count += 1;
                    wp.update(value as i64, x, y);
                }
            }
        }
    }
    (cost, count)
}

pub(super) fn choose_wp_params(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    num_color: usize,
    header_count: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> WpParams {
    let width = linear.xsize();
    let height = linear.ysize();
    let nb_chans = num_color + usize::from(alpha.is_some());
    let total_values = width.saturating_mul(height).saturating_mul(nb_chans);
    let scores = pool.steal_map(scratch, WpParams::PRESETS.len(), |preset, _scratch| {
        let params = WpParams::PRESETS[preset];
        let mut sample_cost = 0u64;
        let mut samples = 0usize;
        for channel in 0..num_color {
            let plane = linear.plane_data(channel);
            let (cost, count) = wp_sample_cost(|x, y| plane[y * width + x], width, height, params);
            sample_cost += cost;
            samples += count;
        }
        if let Some(alpha) = alpha {
            let (cost, count) =
                wp_sample_cost(|x, y| alpha.get_i32(y * width + x), width, height, params);
            sample_cost += cost;
            samples += count;
        }
        let scaled = sample_cost as f64 * total_values as f64 / samples.max(1) as f64;
        // A non-default header costs 51 additional bits per modular group.
        scaled
            + if preset == 0 {
                0.0
            } else {
                (51 * header_count) as f64
            }
    });
    let best = scores
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index);
    WpParams::PRESETS[best]
}

/// Aggregated residual costs across independently predicted modular crops.
#[derive(Default)]
pub(super) struct SqueezePredictorCost {
    pub(super) costs: PredictorCosts,
}

impl SqueezePredictorCost {
    pub(super) fn with_zero() -> Self {
        let mut c = Self::default();
        c.costs.collect_zero = true;
        c
    }

    pub(super) fn safe_predictor(&self) -> u32 {
        self.costs.best_safe_predictor()
    }

    /// Add one independently predicted modular crop. Weighted prediction state
    /// resets here because the decoder resets it for every modular sub-image.
    pub(super) fn add_crop(
        &mut self,
        get: impl Fn(usize, usize) -> i32,
        w: usize,
        h: usize,
        use_wp: bool,
    ) {
        if w == 0 || h == 0 {
            return;
        }
        let disabled = !use_wp;
        let mut wp = WpState::new(w);
        for y in 0..h {
            for x in 0..w {
                let value = get(x, y) as i64;
                let neighbors = predictor_neighbors(&get, x, y, w);
                let weighted = if disabled {
                    0
                } else {
                    wp.predict(
                        x,
                        y,
                        neighbors.top,
                        neighbors.left,
                        neighbors.top_right,
                        neighbors.top_left,
                        neighbors.top_top,
                    )
                };
                self.costs.add(value, neighbors, weighted);
                if !disabled {
                    wp.update(value, x, y);
                }
            }
        }
    }

    pub(super) fn predictor(&self, use_wp: bool) -> u32 {
        self.costs.best_predictor(use_wp)
    }

    pub(super) fn add_rows<'a, T: Copy + 'a>(
        &mut self,
        get_row: impl Fn(usize) -> &'a [T],
        w: usize,
        h: usize,
        use_wp: bool,
    ) where
        i64: From<T>,
    {
        if use_wp {
            // Row-backed costs are used for palette candidates, where flat
            // spans are common. Keep every WP state update and histogram bump.
            let mut wp = WpState::new(w);
            visit_predictor_rows(get_row, w, h, |x, y, value, n| {
                let weighted = wp.predict_and_update_flat(
                    value,
                    x,
                    wp.row_offsets(y),
                    WpNeighbors {
                        north: n.top,
                        west: n.left,
                        north_east: n.top_right,
                        north_west: n.top_left,
                        north_north: n.top_top,
                    },
                );
                self.costs.add(value, n, weighted);
            });
        } else {
            visit_predictor_rows(get_row, w, h, |_, _, value, n| {
                self.costs.add(value, n, 0);
            });
        }
    }
}

#[inline]
pub(super) fn channel_to_context(chan: usize, nb_chans: usize) -> u32 {
    (nb_chans - 1 - chan) as u32
}

pub(super) fn tokenize_all(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<Token> {
    tokenize_all_with_wp(
        linear,
        alpha,
        xsize,
        _ysize,
        x0,
        y0,
        gw,
        gh,
        num_color,
        predictors,
        grad_pack_fn,
        pool,
        scratch,
        WpParams::DEFAULT,
    )
}

fn tokenize_all_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
) -> Vec<Token> {
    let channel_tokens = tokenize_channels_with_wp(
        linear,
        alpha,
        xsize,
        _ysize,
        x0,
        y0,
        gw,
        gh,
        num_color,
        predictors,
        grad_pack_fn,
        pool,
        scratch,
        wp_params,
    );
    let total_len = channel_tokens.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total_len);
    for channel in channel_tokens {
        out.extend(channel);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tokenize_channels_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
) -> Vec<Vec<Token>> {
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };
    pool.steal_map(scratch, nb_chans, |chan, scratch| {
        let mut out = Vec::with_capacity(gw * gh);
        tokenize_channel_with_wp(
            linear,
            alpha,
            xsize,
            x0,
            y0,
            gw,
            gh,
            num_color,
            predictors,
            chan,
            grad_pack_fn,
            &mut scratch.gradient,
            &mut out,
            wp_params,
        );
        out
    })
}

/// Tokenizes and run-compresses one channel at a time. This preserves channel
/// boundaries (and therefore the bitstream) while retaining only one raw-token
/// plane per worker. Multi-group encoding already supplies group-level
/// parallelism, so nested channel tasks would only increase live storage.
#[allow(clippy::too_many_arguments)]
pub(super) fn tokenize_runs_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
) -> Vec<LzToken> {
    let nb_chans = num_color + usize::from(alpha.is_some());
    let mut out = RunLzWriter::with_capacity(gw * gh * nb_chans);
    for chan in 0..nb_chans {
        tokenize_channel_with_wp(
            linear,
            alpha,
            xsize,
            x0,
            y0,
            gw,
            gh,
            num_color,
            predictors,
            chan,
            grad_pack_fn,
            &mut scratch.gradient,
            &mut out,
            wp_params,
        );
        out.finish_channel();
    }
    out.finish()
}

#[allow(clippy::too_many_arguments)]
// Preserve specialization at the group caller after adding the typed row paths.
#[inline(always)]
fn tokenize_channel_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
    chan: usize,
    grad_pack_fn: GradPackInteriorFn,
    gradient: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    let nb_chans = num_color + usize::from(alpha.is_some());
    let ctx = channel_to_context(chan, nb_chans);
    if chan < num_color {
        if predictors[chan] == PREDICTOR_WEIGHTED {
            tokenize_wp_row_slices(
                ctx,
                |gy| &linear.plane_row(chan, y0 + gy)[x0..x0 + gw],
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            );
        } else if predictors[chan] == PREDICTOR_GRADIENT {
            // Keep the existing staged SIMD path for color gradients.
            tokenize_plane_with_wp(
                ctx,
                |gx, gy| linear.plane_row(chan, y0 + gy)[x0 + gx],
                gw,
                gh,
                PREDICTOR_GRADIENT,
                grad_pack_fn,
                gradient,
                out,
                wp_params,
            );
        } else {
            tokenize_plane_rows(
                ctx,
                |gy| &linear.plane_row(chan, y0 + gy)[x0..x0 + gw],
                gw,
                gh,
                predictors[chan],
                grad_pack_fn,
                gradient,
                out,
            );
        }
        return;
    }

    let alpha = alpha.expect("alpha channel must exist");
    if predictors[chan] == PREDICTOR_WEIGHTED {
        match alpha {
            AlphaPlane::U8(data) => tokenize_wp_row_slices(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            ),
            AlphaPlane::U16 { data, .. } => tokenize_wp_row_slices(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            ),
            AlphaPlane::F32(data) => tokenize_wp_row_slices(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            ),
        }
    } else {
        match alpha {
            AlphaPlane::U8(data) => tokenize_sample_rows(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                predictors[chan],
                grad_pack_fn,
                gradient,
                out,
            ),
            AlphaPlane::U16 { data, .. } => tokenize_sample_rows(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                predictors[chan],
                grad_pack_fn,
                gradient,
                out,
            ),
            AlphaPlane::F32(data) => tokenize_plane_rows(
                ctx,
                |gy| {
                    let start = (y0 + gy) * xsize + x0;
                    &data[start..start + gw]
                },
                gw,
                gh,
                predictors[chan],
                grad_pack_fn,
                gradient,
                out,
            ),
        }
    }
}

#[derive(Default)]
pub(crate) struct GradientScratch {
    pub(crate) cur: Vec<i32>,
    pub(crate) prev: Vec<i32>,
    pub(crate) prev_prev: Vec<i32>,
    pub(crate) buf: Vec<u32>,
    pub(crate) wp: Option<Box<WpState>>,
}

fn reset_wp_scratch(
    slot: &mut Option<Box<WpState>>,
    width: usize,
    params: WpParams,
) -> &mut WpState {
    match slot {
        Some(wp) => {
            wp.reset(width, params);
            wp
        }
        None => slot.insert(Box::new(WpState::with_params(width, params))),
    }
}

/// Tokenize one channel's group-local rectangle with the chosen predictor
/// (`PREDICTOR_GRADIENT` or `PREDICTOR_WEIGHTED`). Neighbors use libjxl's exact
/// border conventions (see weighted::State / Predict()):
///   left = x>0 ? W : (y>0 ? N : 0); top = y>0 ? N : left;
///   topleft = (x&&y) ? NW : left; topright = (x+1<w && y) ? NE : top;
///   toptop  = y>1 ? NN : top.
pub(super) fn tokenize_plane(
    ctx: u32,
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut Vec<Token>,
) {
    tokenize_plane_with_wp(
        ctx,
        get,
        gw,
        gh,
        pred_id,
        grad_pack_fn,
        scratch,
        out,
        WpParams::DEFAULT,
    );
}

/// Slice-backed planes avoid point fetches and temporary input rows.
pub(super) fn tokenize_plane_rows<'a>(
    ctx: u32,
    get_row: impl Fn(usize) -> &'a [i32],
    w: usize,
    h: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
) {
    if w == 0 || h == 0 {
        return;
    }
    if pred_id == PREDICTOR_GRADIENT {
        scratch.buf.resize(w, 0);
        let buf = &mut scratch.buf[..w];
        let mut north = get_row(0);
        assert_eq!(north.len(), w);
        out.push_token(Token::new(ctx, pack_signed(north[0])));
        for pair in north.array_windows::<2>() {
            out.push_token(Token::new(ctx, pack_signed(pair[1].wrapping_sub(pair[0]))));
        }
        for y in 1..h {
            let current = get_row(y);
            assert_eq!(current.len(), w);
            buf[0] = pack_signed(current[0].wrapping_sub(north[0]));
            grad_pack_fn(current, north, buf, w);
            out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
            north = current;
        }
    } else {
        tokenize_predictor_rows(ctx, get_row, w, h, pred_id, scratch, out);
    }
}

/// Expand byte/word samples only for the i32 gradient kernel's two input rows;
/// other predictors read the original palette indices or alpha samples directly.
pub(super) fn tokenize_sample_rows<'a, T: Copy + 'a>(
    ctx: u32,
    get_row: impl Fn(usize) -> &'a [T],
    w: usize,
    h: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
) where
    i32: From<T>,
    i64: From<T>,
{
    if w == 0 || h == 0 {
        return;
    }
    if pred_id == PREDICTOR_GRADIENT {
        scratch.cur.resize(w, 0);
        scratch.prev.resize(w, 0);
        scratch.buf.resize(w, 0);
        let mut current = &mut scratch.cur[..w];
        let mut north = &mut scratch.prev[..w];
        let buf = &mut scratch.buf[..w];
        let first = get_row(0);
        assert_eq!(first.len(), w);
        out.push_token(Token::new(ctx, pack_signed(i32::from(first[0]))));
        for pair in first.array_windows::<2>() {
            out.push_token(Token::new(
                ctx,
                pack_signed(i32::from(pair[1]).wrapping_sub(i32::from(pair[0]))),
            ));
        }
        for (dest, &value) in north.iter_mut().zip(first) {
            *dest = i32::from(value);
        }
        for y in 1..h {
            let row = get_row(y);
            assert_eq!(row.len(), w);
            for (dest, &value) in current.iter_mut().zip(row) {
                *dest = i32::from(value);
            }
            buf[0] = pack_signed(current[0].wrapping_sub(north[0]));
            grad_pack_fn(current, north, buf, w);
            out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
            std::mem::swap(&mut current, &mut north);
        }
    } else {
        tokenize_predictor_rows(ctx, get_row, w, h, pred_id, scratch, out);
    }
}

fn tokenize_predictor_rows<'a, T: Copy + 'a>(
    ctx: u32,
    get_row: impl Fn(usize) -> &'a [T],
    w: usize,
    h: usize,
    pred_id: u32,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    if pred_id == PREDICTOR_WEIGHTED {
        let wp = reset_wp_scratch(&mut scratch.wp, w, WpParams::DEFAULT);
        visit_predictor_rows(get_row, w, h, |x, y, value, n| {
            let prediction = wp.predict_and_update_flat(
                value,
                x,
                wp.row_offsets(y),
                WpNeighbors {
                    north: n.top,
                    west: n.left,
                    north_east: n.top_right,
                    north_west: n.top_left,
                    north_north: n.top_top,
                },
            );
            push_wp_token(ctx, value, prediction, out);
        });
    } else {
        visit_predictor_rows(get_row, w, h, |_, _, value, n| {
            let prediction = predictor_value(pred_id, n, 0);
            out.push_token(Token::new(ctx, pack_signed((value - prediction) as i32)));
        });
    }
}

pub(super) trait TokenSink {
    fn push_token(&mut self, token: Token);

    fn extend_tokens(&mut self, tokens: impl Iterator<Item = Token>) {
        for token in tokens {
            self.push_token(token);
        }
    }
}

impl TokenSink for Vec<Token> {
    #[inline(always)]
    fn push_token(&mut self, token: Token) {
        self.push(token);
    }

    #[inline]
    fn extend_tokens(&mut self, tokens: impl Iterator<Item = Token>) {
        self.extend(tokens);
    }
}

impl TokenSink for RunLzWriter {
    #[inline(always)]
    fn push_token(&mut self, token: Token) {
        self.push(token);
    }
}

#[inline(always)]
fn push_wp_token(ctx: u32, value: i64, prediction: i64, out: &mut impl TokenSink) {
    out.push_token(Token::new(ctx, pack_signed((value - prediction) as i32)));
}

/// The first, second, and interior row kernels deliberately have separate
/// border handling. Besides keeping branches out of the interior loop, this is
/// the dispatch seam for a future vector implementation of the row arithmetic.
fn tokenize_wp_first_row<T: Copy>(
    ctx: u32,
    current: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    let row = wp.row_offsets(0);
    let mut left = 0i64;
    for (x, &value) in current.iter().enumerate() {
        let value = i64::from(value);
        let prediction = wp.predict_and_update(
            value,
            x,
            row,
            WpNeighbors {
                north: left,
                west: left,
                north_east: left,
                north_west: left,
                north_north: left,
            },
        );
        push_wp_token(ctx, value, prediction, out);
        left = value;
    }
}

fn tokenize_wp_second_row<T: Copy>(
    ctx: u32,
    current: &[T],
    north: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    tokenize_wp_interior_row(ctx, 1, current, north, north, wp, out);
}

fn tokenize_wp_interior_row<T: Copy>(
    ctx: u32,
    y: usize,
    current: &[T],
    north: &[T],
    north_north: &[T],
    wp: &mut WpState,
    out: &mut impl TokenSink,
) where
    i64: From<T>,
{
    let row = wp.row_offsets(y);
    visit_predictor_row(current, north, north_north, false, |x, value, n| {
        let prediction = wp.predict_and_update(
            value,
            x,
            row,
            WpNeighbors {
                north: n.top,
                west: n.left,
                north_east: n.top_right,
                north_west: n.top_left,
                north_north: n.top_top,
            },
        );
        push_wp_token(ctx, value, prediction, out);
    });
}

fn tokenize_wp_row_slices<'a, T: Copy + 'a>(
    ctx: u32,
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    out: &mut impl TokenSink,
    wp_params: WpParams,
    wp_scratch: &mut Option<Box<WpState>>,
) where
    i64: From<T>,
{
    if gw == 0 || gh == 0 {
        return;
    }
    let wp = reset_wp_scratch(wp_scratch, gw, wp_params);
    let first = get_row(0);
    debug_assert_eq!(first.len(), gw);
    tokenize_wp_first_row(ctx, first, wp, out);
    if gh == 1 {
        return;
    }
    let second = get_row(1);
    debug_assert_eq!(second.len(), gw);
    tokenize_wp_second_row(ctx, second, first, wp, out);
    let mut north_north = first;
    let mut north = second;
    for y in 2..gh {
        let current = get_row(y);
        debug_assert_eq!(current.len(), gw);
        tokenize_wp_interior_row(ctx, y, current, north, north_north, wp, out);
        north_north = north;
        north = current;
    }
}

fn tokenize_wp_rows_from_get(
    ctx: u32,
    get: &impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    if gw == 0 || gh == 0 {
        return;
    }
    if scratch.cur.len() < gw {
        scratch.cur.resize(gw, 0);
    }
    if scratch.prev.len() < gw {
        scratch.prev.resize(gw, 0);
    }
    if scratch.prev_prev.len() < gw {
        scratch.prev_prev.resize(gw, 0);
    }

    let mut current = &mut scratch.cur[..gw];
    let mut north = &mut scratch.prev[..gw];
    let mut north_north = &mut scratch.prev_prev[..gw];
    let wp = reset_wp_scratch(&mut scratch.wp, gw, wp_params);
    for y in 0..gh {
        if y != 0 {
            std::mem::swap(&mut north_north, &mut north);
            std::mem::swap(&mut north, &mut current);
        }
        for (x, value) in current.iter_mut().enumerate() {
            *value = get(x, y);
        }
        match y {
            0 => tokenize_wp_first_row(ctx, current, wp, out),
            1 => tokenize_wp_second_row(ctx, current, north, wp, out),
            _ => tokenize_wp_interior_row(ctx, y, current, north, north_north, wp, out),
        }
    }
}

fn tokenize_plane_with_wp(
    ctx: u32,
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
    wp_params: WpParams,
) {
    if pred_id == PREDICTOR_WEIGHTED {
        tokenize_wp_rows_from_get(ctx, &get, gw, gh, scratch, out, wp_params);
    } else if pred_id == PREDICTOR_GRADIENT {
        // Gradient (ClampedGradient): per-pixel independent, pure integer ->
        // vectorized over the interior of each row.
        if scratch.buf.len() < gw {
            scratch.buf.resize(gw, 0);
        }
        if scratch.cur.len() < gw {
            scratch.cur.resize(gw, 0);
        }
        if scratch.prev.len() < gw {
            scratch.prev.resize(gw, 0);
        }
        let mut cur = &mut scratch.cur[..gw];
        let mut prev = &mut scratch.prev[..gw];
        let buf = &mut scratch.buf[..gw];
        for gy in 0..gh {
            std::mem::swap(&mut cur, &mut prev); // prev = last row's cur
            for (gx, c) in cur.iter_mut().enumerate() {
                *c = get(gx, gy);
            }
            if gy == 0 {
                buf[0] = pack_signed(cur[0]); // gx 0: pred = 0
                for (out, &[left, value]) in buf[1..].iter_mut().zip(cur.array_windows::<2>()) {
                    *out = pack_signed(value.wrapping_sub(left)); // pred = W
                }
            } else {
                buf[0] = pack_signed(cur[0].wrapping_sub(prev[0])); // gx 0: pred = N
                grad_pack_fn(cur, prev, buf, gw); // gx in 1..gw
            }
            out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
        }
    } else {
        debug_assert!(matches!(
            pred_id,
            PREDICTOR_AVERAGE4 | PREDICTOR_SELECT | PREDICTOR_LEFT | PREDICTOR_TOP
        ));
        for gy in 0..gh {
            for gx in 0..gw {
                let value = get(gx, gy) as i64;
                let neighbors = predictor_neighbors(&get, gx, gy, gw);
                let pred = predictor_value(pred_id, neighbors, 0);
                out.push_token(Token::new(ctx, pack_signed((value - pred) as i32)));
            }
        }
    }
}
pub(crate) type GradPackInteriorFn = fn(&[i32], &[i32], &mut [u32], usize);
fn select_grad_pack_interior_fn() -> GradPackInteriorFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |c, p, o, g| unsafe { crate::avx::grad_pack_interior(c, p, o, g) };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |c, p, o, g| unsafe { crate::sse::grad_pack_interior(c, p, o, g) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |c, p, o, g| unsafe { crate::neon::grad_pack_interior(c, p, o, g) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::grad_pack_interior
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        grad_pack_interior_scalar
    }
}

static GRAD_PACK_INTERIOR_FN: OnceLock<GradPackInteriorFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_grad_pack_interior_fn() -> GradPackInteriorFn {
    *GRAD_PACK_INTERIOR_FN.get_or_init(select_grad_pack_interior_fn)
}

#[allow(unused)]
fn grad_pack_interior_scalar(cur: &[i32], prev: &[i32], out: &mut [u32], gw: usize) {
    if gw <= 1 {
        return;
    }
    let pixels = cur[..gw].array_windows::<2>();
    let north = prev[..gw].array_windows::<2>();
    for ((dest, &[w, value]), &[nw, n]) in out[1..gw].iter_mut().zip(pixels).zip(north) {
        let ac = w.wrapping_sub(nw);
        let bc = n.wrapping_sub(nw);
        let grad = ac.wrapping_add(n);
        let clamp = if (w.wrapping_sub(n) ^ bc) < 0 { n } else { w };
        let pred = if (ac ^ bc) < 0 { grad } else { clamp };
        *dest = pack_signed(value.wrapping_sub(pred));
    }
}

pub(super) struct PredictorCosts {
    entropy_of_hist: EntropyOfHistFn,
    histograms: [Vec<u64>; SLOW_PREDICTORS.len()],
    /// Residual histogram for the Zero predictor (the raw values). Collected
    /// only when `collect_zero` is set (the lossy-modular path); the lossless
    /// callers skip the extra bump.
    zero_hist: Vec<u64>,
    collect_zero: bool,
    total: u64,
}

impl Default for PredictorCosts {
    fn default() -> Self {
        Self {
            entropy_of_hist: selected_entropy_of_hist_fn(),
            histograms: Default::default(),
            zero_hist: Vec::new(),
            collect_zero: false,
            total: 0,
        }
    }
}

impl PredictorCosts {
    /// Merge independently predicted crops before evaluating entropy costs.
    pub(crate) fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.collect_zero, other.collect_zero);
        for (dest, src) in self.histograms.iter_mut().zip(&other.histograms) {
            if dest.len() < src.len() {
                dest.resize(src.len(), 0);
            }
            for (dest, &src) in dest.iter_mut().zip(src) {
                *dest += src;
            }
        }
        if self.zero_hist.len() < other.zero_hist.len() {
            self.zero_hist.resize(other.zero_hist.len(), 0);
        }
        for (dest, &src) in self.zero_hist.iter_mut().zip(&other.zero_hist) {
            *dest += src;
        }
        self.total += other.total;
    }
    #[inline]
    fn add(&mut self, value: i64, neighbors: PredictorNeighbors, weighted: i64) {
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate() {
            let pred = predictor_value(pred_id, neighbors, weighted);
            let symbol = pack_signed((value - pred) as i32) as usize;
            let hist = &mut self.histograms[candidate];
            if hist.len() <= symbol {
                hist.resize(symbol + 1, 0);
            }
            hist[symbol] += 1;
        }
        if self.collect_zero {
            let symbol = pack_signed(value as i32) as usize;
            if self.zero_hist.len() <= symbol {
                self.zero_hist.resize(symbol + 1, 0);
            }
            self.zero_hist[symbol] += 1;
        }
        self.total += 1;
    }

    /// Best predictor among Zero and the scale-equivariant subset — the only
    /// predictors a channel coded at 1/q scale with a leaf multiplier can use.
    fn best_safe_predictor(&self) -> u32 {
        let entropy_of_hist = self.entropy_of_hist;
        debug_assert!(self.collect_zero);
        let mut best_id = PREDICTOR_ZERO;
        let mut best_bits = entropy_of_hist(&self.zero_hist, self.total);
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate() {
            if !matches!(
                pred_id,
                PREDICTOR_GRADIENT | PREDICTOR_SELECT | PREDICTOR_LEFT | PREDICTOR_TOP
            ) {
                continue;
            }
            let bits = entropy_of_hist(&self.histograms[candidate], self.total);
            if bits < best_bits {
                best_bits = bits;
                best_id = pred_id;
            }
        }
        best_id
    }

    fn best_predictor(&self, use_wp: bool) -> u32 {
        let entropy_of_hist = self.entropy_of_hist;
        // Without WP the Weighted candidate (index 0) is skipped.
        let first = usize::from(!use_wp);
        let mut best_id = SLOW_PREDICTORS[first];
        let mut best_bits = entropy_of_hist(&self.histograms[first], self.total);
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate().skip(first + 1) {
            let bits = entropy_of_hist(&self.histograms[candidate], self.total);
            if bits < best_bits {
                best_bits = bits;
                best_id = pred_id;
            }
        }
        best_id
    }
}

/// Evaluate all Slow-mode predictors in one traversal and choose the lowest
/// order-0 residual entropy. Weighted remains the deterministic tie-breaker.
pub(super) fn choose_predictor_for_plane(
    get: impl Fn(usize, usize) -> i32,
    w: usize,
    h: usize,
    use_wp: bool,
) -> u32 {
    choose_predictor_for_plane_with_wp(get, w, h, WpParams::DEFAULT, use_wp)
}

pub(super) fn choose_predictor_for_rows<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    w: usize,
    h: usize,
    use_wp: bool,
) -> u32
where
    i64: From<T>,
{
    if w == 0 || h == 0 {
        return fixed_predictor(use_wp);
    }
    let mut costs = SqueezePredictorCost::default();
    costs.add_rows(get_row, w, h, use_wp);
    costs.predictor(use_wp)
}

fn choose_predictor_for_plane_with_wp(
    get: impl Fn(usize, usize) -> i32,
    w: usize,
    h: usize,
    wp_params: WpParams,
    use_wp: bool,
) -> u32 {
    if w == 0 || h == 0 {
        return fixed_predictor(use_wp);
    }
    let disabled = !use_wp;
    let mut wp = WpState::with_params(w, wp_params);
    let mut costs = PredictorCosts::default();
    for gy in 0..h {
        for gx in 0..w {
            let value = get(gx, gy) as i64;
            let neighbors = predictor_neighbors(&get, gx, gy, w);
            let weighted = if disabled {
                0
            } else {
                wp.predict(
                    gx,
                    gy,
                    neighbors.top,
                    neighbors.left,
                    neighbors.top_right,
                    neighbors.top_left,
                    neighbors.top_top,
                )
            };
            costs.add(value, neighbors, weighted);
            if !disabled {
                wp.update(value, gx, gy);
            }
        }
    }
    costs.best_predictor(use_wp)
}

pub(super) fn choose_predictors_with_wp(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    num_color: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    wp_params: WpParams,
    use_wp: bool,
) -> [u32; 4] {
    let mut preds = [fixed_predictor(use_wp); 4];
    let num_channels = num_color + usize::from(alpha.is_some());
    let selected = pool.steal_map(scratch, num_channels, |chan, _scratch| {
        if chan < num_color {
            let pd = linear.plane_data(chan);
            choose_predictor_for_plane_with_wp(
                |x, y| pd[y * xsize + x],
                xsize,
                ysize,
                wp_params,
                use_wp,
            )
        } else {
            let a = alpha.expect("alpha channel must exist");
            choose_predictor_for_plane_with_wp(
                |x, y| a.get_i32(y * xsize + x),
                xsize,
                ysize,
                wp_params,
                use_wp,
            )
        }
    });
    preds[..num_color].copy_from_slice(&selected[..num_color]);
    if alpha.is_some() {
        preds[3] = selected[num_color];
    }
    preds
}

#[inline]
pub(super) fn clamped_gradient(w: i64, n: i64, nw: i64) -> i64 {
    let lo = w.min(n);
    let hi = w.max(n);
    (w + n - nw).clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_neighbors_match_point_fetches_at_every_border() {
        for w in (0..=9).chain([17, 64, 257]) {
            for h in 0..=5 {
                let data: Vec<i32> = (0..w * h)
                    .map(|i| ((i * 977) % 65536) as i32 - 32768)
                    .collect();
                let get = |x, y| data[y * w + x];
                let mut count = 0;
                visit_predictor_rows(
                    |y| &data[y * w..][..w],
                    w,
                    h,
                    |x, y, value, n| {
                        assert_eq!(count, y * w + x);
                        assert_eq!(value, get(x, y) as i64);
                        assert_eq!(n, predictor_neighbors(&get, x, y, w), "{w}x{h}, ({x}, {y})");
                        count += 1;
                    },
                );
                assert_eq!(count, w * h);
            }
        }
    }

    #[test]
    fn row_costs_match_point_fetched_histograms_and_predictors() {
        for w in [1, 2, 3, 4, 5, 17, 257] {
            for h in [1, 2, 5] {
                let data: Vec<u16> = (0..w * h)
                    .map(|i| ((i * 977) ^ (i * i * 37)) as u16)
                    .collect();
                for use_wp in [false, true] {
                    let mut reference = SqueezePredictorCost::with_zero();
                    reference.add_crop(|x, y| data[y * w + x] as i32, w, h, use_wp);
                    let mut rows = SqueezePredictorCost::with_zero();
                    rows.add_rows(|y| &data[y * w..][..w], w, h, use_wp);
                    assert_eq!(
                        rows.costs.histograms, reference.costs.histograms,
                        "{w}x{h}, wp={use_wp}"
                    );
                    assert_eq!(rows.costs.zero_hist, reference.costs.zero_hist);
                    assert_eq!(rows.costs.total, reference.costs.total);
                    assert_eq!(rows.predictor(use_wp), reference.predictor(use_wp));
                }
            }
        }
    }

    #[test]
    fn palette_flat_spans_preserve_cost_histograms() {
        for w in [1, 17, 64] {
            let h = 8;
            for common in [-8193, -8192, -1, 0, 1, 8192, 8193, 65535] {
                let data: Vec<i32> = (0..w * h)
                    .map(|i| {
                        if i / w == 3 || i % w == w / 2 {
                            common + (i % 29) as i32 - 14
                        } else {
                            common
                        }
                    })
                    .collect();
                let mut reference = SqueezePredictorCost::with_zero();
                reference.add_crop(|x, y| data[y * w + x], w, h, true);
                let mut rows = SqueezePredictorCost::with_zero();
                rows.add_rows(|y| &data[y * w..][..w], w, h, true);
                assert_eq!(rows.costs.histograms, reference.costs.histograms);
                assert_eq!(rows.costs.zero_hist, reference.costs.zero_hist);
                assert_eq!(rows.costs.total, reference.costs.total);
                assert_eq!(rows.predictor(true), reference.predictor(true));
            }
        }
    }

    #[test]
    fn row_tokens_match_point_fetched_predictions() {
        for w in (0..=9).chain([15, 16, 17, 33, 257]) {
            for h in [0, 1, 2, 5] {
                let data: Vec<i32> = (0..w * h)
                    .map(|i| ((i * 977) % 65536) as i32 - 32768)
                    .collect();
                let get = |x, y| data[y * w + x];
                for pred in SLOW_PREDICTORS {
                    let mut wp = WpState::new(w);
                    let mut reference = Vec::new();
                    for y in 0..h {
                        for x in 0..w {
                            let n = predictor_neighbors(&get, x, y, w);
                            let value = get(x, y) as i64;
                            let weighted = if pred == PREDICTOR_WEIGHTED {
                                let p = wp.predict(
                                    x,
                                    y,
                                    n.top,
                                    n.left,
                                    n.top_right,
                                    n.top_left,
                                    n.top_top,
                                );
                                wp.update(value, x, y);
                                p
                            } else {
                                0
                            };
                            reference.push((
                                11,
                                pack_signed((value - predictor_value(pred, n, weighted)) as i32),
                            ));
                        }
                    }
                    for kernel in [
                        grad_pack_interior_scalar as GradPackInteriorFn,
                        selected_grad_pack_interior_fn(),
                    ] {
                        let mut scratch = GradientScratch::default();
                        let mut actual = Vec::new();
                        tokenize_plane_rows(
                            11,
                            |y| &data[y * w..][..w],
                            w,
                            h,
                            pred,
                            kernel,
                            &mut scratch,
                            &mut actual,
                        );
                        let actual: Vec<_> = actual.iter().map(|t| (t.context, t.value)).collect();
                        assert_eq!(actual, reference, "{w}x{h}, predictor={pred}");
                        assert!(
                            scratch.cur.is_empty()
                                && scratch.prev.is_empty()
                                && scratch.prev_prev.is_empty()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn channel_rows_match_point_fetches_for_crops_and_alpha_types() {
        use super::super::lz77::lz77_compress_runs;

        for w in [1, 2, 3, 4, 5, 17, 257] {
            for h in [1, 2, 5] {
                let (x0, y0) = (3, 2);
                let stride = w + 7;
                let height = h + 4;
                let mut image = Image3Si::new(stride, height);
                let values = [i32::MIN, i32::MAX, -65535, -8193, -1, 0, 1, 8192, 65535];
                for chan in 0..3 {
                    for y in 0..height {
                        for (x, value) in image.plane_row_mut(chan, y).iter_mut().enumerate() {
                            *value = if (x + y * stride) % 23 < 11 {
                                0
                            } else {
                                values[(x * 7 + y * 3 + chan) % values.len()]
                            };
                        }
                    }
                }
                let alpha8: Vec<_> = (0..stride * height)
                    .map(|i| if i % 13 < 5 { 0 } else { (i * 197) as u8 })
                    .collect();
                let alpha16: Vec<_> = (0..stride * height)
                    .map(|i| if i % 13 < 5 { 65535 } else { (i * 977) as u16 })
                    .collect();
                let alpha32: Vec<_> = (0..stride * height)
                    .map(|i| values[i % values.len()])
                    .collect();
                for alpha in [
                    AlphaPlane::U8(alpha8),
                    AlphaPlane::U16 {
                        data: alpha16,
                        bits: 16,
                    },
                    AlphaPlane::F32(alpha32),
                ] {
                    for num_color in [1, 3] {
                        let channels = num_color + 1;
                        for chan in 0..channels {
                            let ctx = channel_to_context(chan, channels);
                            for pred in SLOW_PREDICTORS {
                                for params in WpParams::PRESETS {
                                    let get = |x: usize, y: usize| {
                                        if chan < num_color {
                                            image.plane_row(chan, y0 + y)[x0 + x]
                                        } else {
                                            alpha.get_i32((y0 + y) * stride + x0 + x)
                                        }
                                    };
                                    let mut reference = Vec::new();
                                    tokenize_plane_with_wp(
                                        ctx,
                                        get,
                                        w,
                                        h,
                                        pred,
                                        grad_pack_interior_scalar,
                                        &mut GradientScratch::default(),
                                        &mut reference,
                                        params,
                                    );
                                    for kernel in [
                                        grad_pack_interior_scalar as GradPackInteriorFn,
                                        selected_grad_pack_interior_fn(),
                                    ] {
                                        let mut scratch = GradientScratch::default();
                                        let mut actual = Vec::new();
                                        tokenize_channel_with_wp(
                                            &image,
                                            Some(&alpha),
                                            stride,
                                            x0,
                                            y0,
                                            w,
                                            h,
                                            num_color,
                                            &[pred; 4],
                                            chan,
                                            kernel,
                                            &mut scratch,
                                            &mut actual,
                                            params,
                                        );
                                        assert_eq!(actual.len(), reference.len());
                                        assert!(
                                            actual.iter().zip(&reference).all(|(a, b)| (
                                                a.context, a.value
                                            ) == (
                                                b.context, b.value
                                            )),
                                            "{w}x{h}, channel={chan}, predictor={pred}, params={params:?}"
                                        );
                                        if pred != PREDICTOR_GRADIENT
                                            || (chan == num_color
                                                && matches!(&alpha, AlphaPlane::F32(_)))
                                        {
                                            assert!(
                                                scratch.cur.is_empty()
                                                    && scratch.prev.is_empty()
                                                    && scratch.prev_prev.is_empty()
                                            );
                                        }
                                        let mut runs = RunLzWriter::with_capacity(0);
                                        tokenize_channel_with_wp(
                                            &image,
                                            Some(&alpha),
                                            stride,
                                            x0,
                                            y0,
                                            w,
                                            h,
                                            num_color,
                                            &[pred; 4],
                                            chan,
                                            kernel,
                                            &mut scratch,
                                            &mut runs,
                                            params,
                                        );
                                        assert_eq!(runs.finish(), lz77_compress_runs(&reference));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn byte_indices_and_streamed_runs_match_point_fetched_tokens() {
        use super::super::lz77::lz77_compress_runs;

        let mut scratch = GradientScratch::default();
        for w in (0..=9).chain([15, 16, 17, 33, 257]) {
            for h in [0, 1, 2, 5] {
                for pattern in 0..3 {
                    let data: Vec<u8> = (0..w * h)
                        .map(|i| match pattern {
                            0 => 0,
                            1 => {
                                if i % 7 < 4 {
                                    255
                                } else {
                                    0
                                }
                            }
                            _ => ((i * 977) ^ (i * i * 37)) as u8,
                        })
                        .collect();
                    for pred in SLOW_PREDICTORS {
                        for kernel in [
                            grad_pack_interior_scalar as GradPackInteriorFn,
                            selected_grad_pack_interior_fn(),
                        ] {
                            let mut reference = Vec::new();
                            if w != 0 && h != 0 {
                                tokenize_plane(
                                    1,
                                    |x, y| data[y * w + x] as i32,
                                    w,
                                    h,
                                    pred,
                                    kernel,
                                    &mut scratch,
                                    &mut reference,
                                );
                            }
                            let mut actual = Vec::new();
                            tokenize_sample_rows(
                                1,
                                |y| &data[y * w..][..w],
                                w,
                                h,
                                pred,
                                kernel,
                                &mut scratch,
                                &mut actual,
                            );
                            assert!(actual.iter().zip(&reference).all(|(a, b)| (
                                a.context, a.value
                            ) == (
                                b.context, b.value
                            )));
                            assert_eq!(actual.len(), reference.len());

                            // Include both continuous runs and context changes between
                            // the palette meta-plane and the compact index plane.
                            for ctx in [0, 1] {
                                let meta = [0i32; 16];
                                let mut concatenated = Vec::new();
                                tokenize_plane(
                                    ctx,
                                    |x, y| meta[y * 4 + x],
                                    4,
                                    4,
                                    pred,
                                    kernel,
                                    &mut scratch,
                                    &mut concatenated,
                                );
                                concatenated.extend(reference.iter().copied());
                                let mut runs = RunLzWriter::with_capacity(0);
                                tokenize_plane_rows(
                                    ctx,
                                    |y| &meta[y * 4..][..4],
                                    4,
                                    4,
                                    pred,
                                    kernel,
                                    &mut scratch,
                                    &mut runs,
                                );
                                tokenize_sample_rows(
                                    1,
                                    |y| &data[y * w..][..w],
                                    w,
                                    h,
                                    pred,
                                    kernel,
                                    &mut scratch,
                                    &mut runs,
                                );
                                assert_eq!(
                                    runs.finish(),
                                    lz77_compress_runs(&concatenated),
                                    "{w}x{h}, predictor={pred}, pattern={pattern}, context={ctx}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn merged_crop_costs_match_sequential_histograms_and_predictors() {
        for use_wp in [false, true] {
            let mut sequential = SqueezePredictorCost::with_zero();
            let mut merged = SqueezePredictorCost::with_zero();
            for (width, height, bias) in [(1, 3, 0), (17, 13, -512), (5, 2, 2048)] {
                let get = |x: usize, y: usize| (x * 71 + y * 37) as i32 + bias;
                sequential.add_crop(get, width, height, use_wp);
                let mut part = SqueezePredictorCost::with_zero();
                part.add_crop(get, width, height, use_wp);
                merged.costs.merge(&part.costs);
            }
            assert_eq!(merged.costs.histograms, sequential.costs.histograms);
            assert_eq!(merged.costs.zero_hist, sequential.costs.zero_hist);
            assert_eq!(merged.costs.total, sequential.costs.total);
            assert_eq!(merged.predictor(use_wp), sequential.predictor(use_wp));
            assert_eq!(merged.safe_predictor(), sequential.safe_predictor());
        }
    }

    #[test]
    fn added_predictors_match_jxl_formulas() {
        let n = PredictorNeighbors {
            left: 11,
            top: 7,
            top_left: 5,
            top_right: 13,
            left_left: 3,
            top_top: 2,
            top_right_right: 17,
        };
        assert_eq!(predictor_value(PREDICTOR_LEFT, n, 99), 11);
        assert_eq!(predictor_value(PREDICTOR_TOP, n, 99), 7);
        assert_eq!(predictor_value(PREDICTOR_SELECT, n, 99), 11);
        assert_eq!(predictor_value(PREDICTOR_AVERAGE4, n, 99), 11);

        // Select resolves equal distances toward Top, matching libjxl's
        // `pa < pb ? left : top`.
        let tie = PredictorNeighbors {
            left: 9,
            top: 3,
            top_left: 6,
            ..n
        };
        assert_eq!(predictor_value(PREDICTOR_SELECT, tie, 99), 3);
    }

    #[test]
    fn slow_search_selects_directional_predictors() {
        const W: usize = 64;
        const H: usize = 64;
        let mut horizontal = vec![0i32; W * H];
        for y in 0..H {
            horizontal[y * W] = ((y * 73) & 255) as i32;
            for x in 1..W {
                let delta = (((x * 17 + y * 31) ^ (x * y * 3)) % 7) as i32 - 3;
                horizontal[y * W + x] = horizontal[y * W + x - 1] + delta;
            }
        }
        let mut vertical = vec![0i32; W * H];
        for x in 0..W {
            vertical[x] = ((x * 73) & 255) as i32;
            for y in 1..H {
                let delta = (((y * 17 + x * 31) ^ (x * y * 3)) % 7) as i32 - 3;
                vertical[y * W + x] = vertical[(y - 1) * W + x] + delta;
            }
        }

        assert_eq!(
            choose_predictor_for_plane(|x, y| horizontal[y * W + x], W, H, true),
            PREDICTOR_SELECT
        );
        assert_eq!(
            choose_predictor_for_plane(|x, y| vertical[y * W + x], W, H, true),
            PREDICTOR_SELECT
        );
    }

    #[test]
    fn slow_search_selects_average4_for_its_recurrence() {
        const W: usize = 64;
        const H: usize = 64;
        let mut plane = vec![0i32; W * H];
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                if y < 2 || x < 2 {
                    plane[i] = (((x * 97 + y * 53) ^ (x * y * 11)) & 1023) as i32 - 512;
                    continue;
                }
                let top_right = if x + 1 < W {
                    plane[(y - 1) * W + x + 1]
                } else {
                    plane[(y - 1) * W + x]
                };
                let top_right_right = if x + 2 < W {
                    plane[(y - 1) * W + x + 2]
                } else {
                    top_right
                };
                plane[i] = (6 * plane[(y - 1) * W + x] - 2 * plane[(y - 2) * W + x]
                    + 7 * plane[i - 1]
                    + plane[i - 2]
                    + top_right_right
                    + 3 * top_right
                    + 8)
                    / 16;
            }
        }

        assert_eq!(
            choose_predictor_for_plane(|x, y| plane[y * W + x], W, H, true),
            PREDICTOR_AVERAGE4
        );
    }

    #[test]
    fn directional_wp_presets_are_distinct() {
        const W: usize = 128;
        const H: usize = 128;
        let mut west = vec![0i32; W * H];
        let mut north = vec![0i32; W * H];
        for y in 0..H {
            west[y * W] = ((y * 977) & 65535) as i32;
            for x in 1..W {
                west[y * W + x] = west[y * W + x - 1] + ((x * 13 + y * 7) % 5) as i32 - 2;
            }
        }
        for x in 0..W {
            north[x] = ((x * 977) & 65535) as i32;
            for y in 1..H {
                north[y * W + x] = north[(y - 1) * W + x] + ((y * 13 + x * 7) % 5) as i32 - 2;
            }
        }
        let west_costs: Vec<_> = WpParams::PRESETS
            .iter()
            .map(|&params| wp_sample_cost(|x, y| west[y * W + x], W, H, params).0)
            .collect();
        let north_costs: Vec<_> = WpParams::PRESETS
            .iter()
            .map(|&params| wp_sample_cost(|x, y| north[y * W + x], W, H, params).0)
            .collect();
        assert_ne!(west_costs[2], west_costs[3]);
        assert_ne!(north_costs[2], north_costs[3]);
    }

    #[test]
    fn fused_wp_row_kernels_match_point_fetched_reference() {
        for &(width, height) in &[(0, 3), (1, 1), (1, 4), (2, 2), (7, 5), (17, 4)] {
            let plane: Vec<i32> = (0..width * height)
                .map(|i| {
                    let x = i % width.max(1);
                    let y = i / width.max(1);
                    (((x * 977 + y * 619) ^ (x * y * 37)) as i32 & 0x7fff) - 0x3fff
                })
                .collect();
            for &params in &WpParams::PRESETS {
                let get = |x: usize, y: usize| plane[y * width + x];
                let mut reference = Vec::with_capacity(width * height);
                let mut wp = WpState::with_params(width, params);
                for y in 0..height {
                    for x in 0..width {
                        let value = get(x, y) as i64;
                        let n = predictor_neighbors(&get, x, y, width);
                        let prediction =
                            wp.predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
                        wp.update(value, x, y);
                        reference.push((11, pack_signed((value - prediction) as i32)));
                    }
                }

                let mut scratch = GradientScratch::default();
                let mut fused = Vec::with_capacity(width * height);
                tokenize_plane_with_wp(
                    11,
                    get,
                    width,
                    height,
                    PREDICTOR_WEIGHTED,
                    grad_pack_interior_scalar,
                    &mut scratch,
                    &mut fused,
                    params,
                );
                let fused: Vec<_> = fused
                    .iter()
                    .map(|token| (token.context, token.value))
                    .collect();
                assert_eq!(fused, reference, "{width}x{height}, params={params:?}");
            }
        }
    }
}
