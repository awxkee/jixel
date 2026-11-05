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

//! Modular predictors, neighborhoods, and gradient row kernels.

mod cost;
#[cfg(test)]
mod tests;
mod tokens;
mod weighted;

pub(crate) use cost::{SqueezePredictorCost, choose_predictor_for_plane};
pub(super) use cost::{choose_predictor_for_rows, choose_predictors_with_wp, choose_wp_params};
pub(crate) use tokens::{channel_to_context, tokenize_plane_rows};
pub(super) use tokens::{
    tokenize_all, tokenize_channels_with_wp, tokenize_plane, tokenize_runs_with_wp,
    tokenize_sample_rows,
};
pub(super) use weighted::{WP_CONTEXT_BUCKETS, WP_CONTEXT_CUTOFFS, WpChannelContexts};

use super::lz77::RunLzWriter;
use crate::entropy::{Token, pack_signed};
use crate::weighted_predictor::{WpParams, WpState};
use std::sync::OnceLock;

const PREDICTOR_ZERO: u32 = 0;

const PREDICTOR_LEFT: u32 = 1;

const PREDICTOR_TOP: u32 = 2;

const PREDICTOR_AVERAGE0: u32 = 3;

const PREDICTOR_SELECT: u32 = 4;

pub(crate) const PREDICTOR_GRADIENT: u32 = 5;

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
pub(crate) fn fixed_predictor(use_wp: bool) -> u32 {
    if use_wp {
        PREDICTOR_WEIGHTED
    } else {
        PREDICTOR_GRADIENT
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(crate) struct PredictorNeighbors {
    pub(crate) left: i64,
    pub(crate) top: i64,
    pub(crate) top_left: i64,
    pub(crate) top_right: i64,
    pub(crate) left_left: i64,
    pub(crate) top_top: i64,
    pub(crate) top_right_right: i64,
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
pub(crate) fn predictor_value(pred_id: u32, n: PredictorNeighbors, weighted: i64) -> i64 {
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

#[derive(Default)]
pub(crate) struct GradientScratch {
    pub(crate) cur: Vec<i32>,
    pub(crate) prev: Vec<i32>,
    pub(crate) prev_prev: Vec<i32>,
    pub(crate) buf: Vec<u32>,
    pub(crate) wp: Option<Box<WpState>>,
}

/// Pack one group-local gradient row. `None` starts a new group: its first
/// sample predicts zero and the rest predict west. Later rows predict north
/// in the first column and use the selected SIMD kernel for the interior.
pub(crate) fn pack_gradient_row(
    current: &[i32],
    north: Option<&[i32]>,
    out: &mut [u32],
    grad_pack_fn: GradPackInteriorFn,
) {
    assert_eq!(out.len(), current.len());
    let Some((&first, _)) = current.split_first() else {
        return;
    };
    if let Some(north) = north {
        assert_eq!(north.len(), current.len());
        out[0] = pack_signed(first.wrapping_sub(north[0]));
        grad_pack_fn(current, north, out, current.len());
    } else {
        out[0] = pack_signed(first);
        for (dest, &[left, value]) in out[1..].iter_mut().zip(current.array_windows::<2>()) {
            *dest = pack_signed(value.wrapping_sub(left));
        }
    }
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

pub(crate) trait TokenSink {
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

#[inline]
pub(super) fn clamped_gradient(w: i64, n: i64, nw: i64) -> i64 {
    let lo = w.min(n);
    let hi = w.max(n);
    (w + n - nw).clamp(lo, hi)
}
