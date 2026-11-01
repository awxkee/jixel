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

//! Channel and plane tokenization with fixed or weighted predictors.

use super::super::lz77::{LzToken, RunLzWriter};
use super::weighted::{
    WpBucketContexts, WpChannelContexts, tokenize_plane_with_wp, tokenize_wp_row_slices,
};
use super::{
    GradPackInteriorFn, GradientScratch, PREDICTOR_GRADIENT, PREDICTOR_WEIGHTED, TokenSink,
    pack_gradient_row, predictor_value, push_wp_token, reset_wp_scratch, visit_predictor_rows,
};
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::{Token, pack_signed};
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpNeighbors, WpParams};

#[inline]
pub(crate) fn channel_to_context(chan: usize, nb_chans: usize) -> u32 {
    (nb_chans - 1 - chan) as u32
}

pub(in crate::lossless) fn tokenize_all(
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
        None,
        None,
    );
    let total_len = channel_tokens.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total_len);
    for channel in channel_tokens {
        out.extend(channel);
    }
    out
}

/// `alpha_constant`: a solid alpha plane is coded as a constant tail
/// (`ConstantTail`), so its channel yields no tokens here.
#[allow(clippy::too_many_arguments)]
pub(in crate::lossless) fn tokenize_channels_with_wp(
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
    alpha_constant: Option<i32>,
    buckets: Option<&[WpChannelContexts]>,
) -> Vec<Vec<Token>> {
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };
    pool.steal_map(scratch, nb_chans, |chan, scratch| {
        if chan >= num_color && alpha_constant.is_some() {
            return Vec::new();
        }
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
            buckets.map(|buckets| &buckets[chan]),
        );
        out
    })
}

/// Tokenizes and run-compresses one channel at a time. This preserves channel
/// boundaries (and therefore the bitstream) while retaining only one raw-token
/// plane per worker. Multi-group encoding already supplies group-level
/// parallelism, so nested channel tasks would only increase live storage.
#[allow(clippy::too_many_arguments)]
pub(in crate::lossless) fn tokenize_runs_with_wp(
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
    alpha_constant: Option<i32>,
    buckets: Option<&[WpChannelContexts]>,
) -> Vec<LzToken> {
    let nb_chans = num_color + usize::from(alpha.is_some());
    let mut out = RunLzWriter::with_capacity(gw * gh * nb_chans);
    for chan in 0..nb_chans {
        if chan >= num_color && alpha_constant.is_some() {
            continue;
        }
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
            buckets.map(|buckets| &buckets[chan]),
        );
        out.finish_channel();
    }
    out.finish()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tokenize_channel_with_wp(
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
    buckets: Option<&WpChannelContexts>,
) {
    let nb_chans = num_color + usize::from(alpha.is_some());
    // The shared WP tree also contains single leaves for non-WP channels.
    // Those channels keep their selected predictor and the leaf's context.
    let ctx = buckets.map_or_else(|| channel_to_context(chan, nb_chans), |b| b.contexts[0]);
    if predictors[chan] == PREDICTOR_WEIGHTED
        && let Some(buckets) = buckets
    {
        let ctx = WpBucketContexts(buckets);
        if chan < num_color {
            tokenize_wp_row_slices(
                ctx,
                |gy| &linear.plane_row(chan, y0 + gy)[x0..x0 + gw],
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            );
            return;
        }
        let alpha = alpha.expect("alpha channel must exist");
        let row = |gy: usize| (y0 + gy) * xsize + x0;
        match alpha {
            AlphaPlane::U8(data) => tokenize_wp_row_slices(
                ctx,
                |gy| &data[row(gy)..row(gy) + gw],
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            ),
            AlphaPlane::U16 { data, .. } => tokenize_wp_row_slices(
                ctx,
                |gy| &data[row(gy)..row(gy) + gw],
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            ),
            AlphaPlane::F32(data) => tokenize_wp_row_slices(
                ctx,
                |gy| &data[row(gy)..row(gy) + gw],
                gw,
                gh,
                out,
                wp_params,
                &mut gradient.wp,
            ),
        }
        return;
    }
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

/// Tokenize one channel's group-local rectangle with the chosen predictor
/// (`PREDICTOR_GRADIENT` or `PREDICTOR_WEIGHTED`). Neighbors use libjxl's exact
/// border conventions (see weighted::State / Predict()):
///   left = x>0 ? W : (y>0 ? N : 0); top = y>0 ? N : left;
///   topleft = (x&&y) ? NW : left; topright = (x+1<w && y) ? NE : top;
///   toptop  = y>1 ? NN : top.
pub(in crate::lossless) fn tokenize_plane(
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
pub(crate) fn tokenize_plane_rows<'a>(
    ctx: u32,
    get_row: impl Fn(usize) -> &'a [i32],
    w: usize,
    h: usize,
    pred_id: u32,
    grad_pack_fn: GradPackInteriorFn,
    scratch: &mut GradientScratch,
    out: &mut impl TokenSink,
) {
    tokenize_plane_rows_shared(ctx, &get_row, w, h, pred_id, grad_pack_fn, scratch, out);
}

// Keep the gradient kernel shared across row getter closure types too.
fn tokenize_plane_rows_shared<'a>(
    ctx: u32,
    get_row: &dyn Fn(usize) -> &'a [i32],
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
        pack_gradient_row(north, None, buf, grad_pack_fn);
        out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
        for y in 1..h {
            let current = get_row(y);
            assert_eq!(current.len(), w);
            pack_gradient_row(current, Some(north), buf, grad_pack_fn);
            out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
            north = current;
        }
    } else {
        tokenize_predictor_rows(ctx, get_row, w, h, pred_id, scratch, out);
    }
}

/// Expand byte/word samples only for the i32 gradient kernel's two input rows;
/// other predictors read the original palette indices or alpha samples directly.
pub(in crate::lossless) fn tokenize_sample_rows<'a, T: Copy + 'a>(
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
    tokenize_sample_rows_shared(ctx, &get_row, w, h, pred_id, grad_pack_fn, scratch, out);
}

fn tokenize_sample_rows_shared<'a, T: Copy + 'a>(
    ctx: u32,
    get_row: &dyn Fn(usize) -> &'a [T],
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
        for (dest, &value) in north.iter_mut().zip(first) {
            *dest = i32::from(value);
        }
        pack_gradient_row(north, None, buf, grad_pack_fn);
        out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
        for y in 1..h {
            let row = get_row(y);
            assert_eq!(row.len(), w);
            for (dest, &value) in current.iter_mut().zip(row) {
                *dest = i32::from(value);
            }
            pack_gradient_row(current, Some(north), buf, grad_pack_fn);
            out.extend_tokens(buf.iter().map(|&value| Token::new(ctx, value)));
            std::mem::swap(&mut current, &mut north);
        }
    } else {
        tokenize_predictor_rows(ctx, get_row, w, h, pred_id, scratch, out);
    }
}

fn tokenize_predictor_rows<'a, T: Copy + 'a>(
    ctx: u32,
    // Dispatch once per row so getter closures share the large pixel kernels.
    get_row: &dyn Fn(usize) -> &'a [T],
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
