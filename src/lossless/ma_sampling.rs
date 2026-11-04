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

//! Modular property traversal, sampling, and DC plane tokenization.

use super::ma_lookup::MaLookup;
use super::predictor::{PredictorNeighbors, predictor_value};
use super::tokens::RawTokens;
use crate::entropy::{Token, pack_signed, uint_encode};
use crate::ma_tree::{LearnedTree, MA_REF_CHANNELS, MaSamples, NUM_MA_PREDS, NUM_MA_PROPS};
use crate::weighted_predictor::{WpNeighbors, WpParams, WpState};

/// Walk one channel rectangle in scan order, feeding the visitor the property
/// vector (libjxl ids 0..=15), the neighborhood, and the WP prediction of
/// every pixel.
/// A previously coded channel of the same size as the one being walked,
/// cropped like it: the source of the decoder's reference properties.
#[derive(Clone, Copy)]
pub(crate) struct MaRefPlane<'a> {
    pub(crate) pixels: &'a [i32],
    pub(crate) stride: usize,
    pub(crate) x0: usize,
    pub(crate) y0: usize,
}

/// libjxl `PrecomputeReferences`: per reference channel |v|, v, |v − g|,
/// v − g, with left = 0 at the crop's left edge, top = left on its first row
/// and top-left = left on either edge, g the clamped gradient.
#[inline]
fn ref_props(refs: &[MaRefPlane<'_>], x: usize, y: usize, p: &mut [i32; NUM_MA_PROPS]) {
    for (r, props) in refs.iter().zip(p[16..].as_chunks_mut::<4>().0) {
        let row = &r.pixels[(r.y0 + y) * r.stride + r.x0..];
        let v = row[x] as i64;
        let left = if x > 0 { row[x - 1] as i64 } else { 0 };
        let (top, top_left) = if y > 0 {
            let above = &r.pixels[(r.y0 + y - 1) * r.stride + r.x0..];
            (
                above[x] as i64,
                if x > 0 { above[x - 1] as i64 } else { left },
            )
        } else {
            (left, left)
        };
        let g = (left + top - top_left).clamp(left.min(top), left.max(top));
        let residual = v - g;
        *props = [
            v.abs() as i32,
            v as i32,
            residual.abs() as i32,
            residual as i32,
        ];
    }
}

pub(super) fn walk_channel_ma<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    refs: &[MaRefPlane<'a>],
    wp_params: WpParams,
    use_wp: bool,
    mut visit: impl FnMut(usize, usize, i64, &[i32; NUM_MA_PROPS], PredictorNeighbors, i64),
) where
    i64: From<T>,
{
    let disabled = !use_wp;
    let mut wp = WpState::with_params(gw, wp_params);
    let mut p = [0i32; NUM_MA_PROPS];
    p[0] = chan as i32;
    p[1] = stream;
    let mut north_row: &[T] = &[];
    let mut north_north_row: &[T] = &[];
    for y in 0..gh {
        let current_row = get_row(y);
        assert_eq!(current_row.len(), gw);
        // Bind valid rows even at the top border, so all three lengths are
        // established before entering the pixel loop.
        if y == 0 {
            north_row = current_row;
        }
        if y < 2 {
            north_north_row = north_row;
        }
        assert_eq!(north_row.len(), gw);
        assert_eq!(north_north_row.len(), gw);
        let row = wp.row_offsets(y);
        p[2] = y as i32;
        p[9] = 0; // "local gradient" carry, reset per row (InitPropsRow)
        // Keep property updates in each loop, but fuse WP only in the hot
        // interior. First-row, border, and tail visits share the WP operation.
        macro_rules! visit_pixel {
            ($predict:ident, $x:expr, $value:expr, $neighbors:expr) => {{
                let x = $x;
                let value = $value;
                let n = $neighbors;
                p[3] = x as i32;
                p[4] = n.top.abs() as i32;
                p[5] = n.left.abs() as i32;
                p[6] = n.top as i32;
                p[7] = n.left as i32;
                // p[8] reads the previous pixel's p[9] (0 at row start).
                p[8] = (n.left - p[9] as i64) as i32;
                p[9] = (n.left + n.top - n.top_left) as i32;
                p[10] = (n.left - n.top_left) as i32;
                p[11] = (n.top_left - n.top) as i32;
                p[12] = (n.top - n.top_right) as i32;
                p[13] = (n.top - n.top_top) as i32;
                p[14] = (n.left - n.left_left) as i32;
                let wp_pred = if disabled {
                    0
                } else {
                    wp.$predict(
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
                    )
                };
                p[15] = if disabled { 0 } else { wp.wp_prop as i32 };
                ref_props(refs, x, y, &mut p);
                visit(x, y, value, &p, n, wp_pred);
            }};
        }
        if y == 0 {
            if let Some((&first, tail)) = current_row.split_first() {
                let first = i64::from(first);
                visit_pixel!(
                    predict_and_update_flat,
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
                    }
                );
                let mut left = first;
                let mut left_left = first;
                for (x, &value) in tail.iter().enumerate() {
                    let value = i64::from(value);
                    visit_pixel!(
                        predict_and_update_flat,
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
                        }
                    );
                    left_left = left;
                    left = value;
                }
            }
        } else {
            let edge = |x: usize| {
                let top = i64::from(north_row[x]);
                let left = if x > 0 {
                    i64::from(current_row[x - 1])
                } else {
                    top
                };
                let top_right = north_row.get(x + 1).map_or(top, |&v| i64::from(v));
                PredictorNeighbors {
                    left,
                    top,
                    top_left: if x > 0 {
                        i64::from(north_row[x - 1])
                    } else {
                        left
                    },
                    top_right,
                    left_left: if x > 1 {
                        i64::from(current_row[x - 2])
                    } else {
                        left
                    },
                    top_top: i64::from(north_north_row[x]),
                    top_right_right: north_row.get(x + 2).map_or(top_right, |&v| i64::from(v)),
                }
            };
            for (x, &value) in current_row.iter().take(2).enumerate() {
                visit_pixel!(predict_and_update_flat, x, i64::from(value), edge(x));
            }
            if gw > 4 {
                let mut left = i64::from(current_row[1]);
                let mut left_left = i64::from(current_row[0]);
                let pixels = current_row[2..gw - 2]
                    .iter()
                    .zip(north_row[1..].array_windows::<4>())
                    .zip(&north_north_row[2..]);
                for (x, ((&value, &[nw, n, ne, nee]), &nn)) in pixels.enumerate() {
                    let value = i64::from(value);
                    visit_pixel!(
                        predict_and_update_flat_inlined,
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
                        }
                    );
                    left_left = left;
                    left = value;
                }
            }
            for x in gw.saturating_sub(2).max(2)..gw {
                visit_pixel!(
                    predict_and_update_flat,
                    x,
                    i64::from(current_row[x]),
                    edge(x)
                );
            }
        }
        north_north_row = north_row;
        north_row = current_row;
    }
}

/// Exact sample count for the rotating row phase.
pub(super) fn ma_channel_sample_count(w: usize, h: usize, stride: usize) -> usize {
    // A complete cycle of row phases selects exactly w samples. In the
    // remaining rows, the first w % stride phases select one extra sample.
    (h / stride) * w + (h % stride) * (w / stride) + (h % stride).min(w % stride)
}

/// Visit the selected pixels with residual tokens under every decoder predictor.
pub(crate) fn sample_channel_ma<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    refs: &[MaRefPlane<'a>],
    wp_params: WpParams,
    use_wp: bool,
    stride: usize,
    mut push: impl FnMut([i32; NUM_MA_PROPS], [u8; NUM_MA_PREDS], i32),
) where
    i64: From<T>,
{
    if use_wp {
        sample_channel_ma_shared::<T, true>(
            &get_row, gw, gh, chan, stream, refs, wp_params, stride, &mut push,
        );
    } else {
        sample_channel_ma_shared::<T, false>(
            &get_row, gw, gh, chan, stream, refs, wp_params, stride, &mut push,
        );
    }
}

// Share the loop across callback types: one getter call per row and one output
// call per selected sample. Keep WP-on/off specializations so disabled WP work
// is removed, especially for lossy modular sampling.
fn sample_channel_ma_shared<'a, T: Copy + 'a, const USE_WP: bool>(
    get_row: &dyn Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    refs: &[MaRefPlane<'a>],
    wp_params: WpParams,
    stride: usize,
    push: &mut dyn FnMut([i32; NUM_MA_PROPS], [u8; NUM_MA_PREDS], i32),
) where
    i64: From<T>,
{
    debug_assert!(stride != 0);
    let disabled = !USE_WP;
    let mut wp = WpState::with_params(gw, wp_params);
    let mut north_row: &[T] = &[];
    let mut north_north_row: &[T] = &[];
    for y in 0..gh {
        let current_row = get_row(y);
        assert_eq!(current_row.len(), gw);
        // Bind valid rows even at the top border, so all three lengths are
        // established before entering the pixel loop.
        if y == 0 {
            north_row = current_row;
        }
        if y < 2 {
            north_north_row = north_row;
        }
        assert_eq!(north_row.len(), gw);
        assert_eq!(north_north_row.len(), gw);
        let row = wp.row_offsets(y);
        let mut west = 0i64;
        let mut west_west = 0i64;
        let mut previous_local_gradient = 0i32;
        // Rotate the sampling phase every row so the probe never locks onto
        // one column residue (a row width divisible by the stride would
        // otherwise sample a single x phase for the whole channel).
        let mut until_sample = 1 + y % stride;
        for x in 0..gw {
            let value = i64::from(current_row[x]);
            let north = if y > 0 { i64::from(north_row[x]) } else { west };
            let left = if x > 0 { west } else { north };
            let top_left = if x > 0 && y > 0 {
                i64::from(north_row[x - 1])
            } else {
                left
            };
            let top_right = if x + 1 < gw && y > 0 {
                i64::from(north_row[x + 1])
            } else {
                north
            };
            let top_top = if y > 1 {
                i64::from(north_north_row[x])
            } else {
                north
            };
            let wp_pred = if disabled {
                0
            } else {
                let wp_neighbors = WpNeighbors {
                    north,
                    west: left,
                    north_east: top_right,
                    north_west: top_left,
                    north_north: top_top,
                };
                // Sampling still advances WP on every pixel; this loop benefits
                // from fusion even when only a fraction of properties are kept.
                wp.predict_and_update_flat_inlined(value, x, row, wp_neighbors)
            };
            let local_gradient = (left + north - top_left) as i32;

            until_sample -= 1;
            if until_sample == 0 {
                until_sample = stride;
                let neighbors = PredictorNeighbors {
                    left,
                    top: north,
                    top_left,
                    top_right,
                    left_left: if x > 1 { west_west } else { left },
                    top_top,
                    top_right_right: if x + 2 < gw && y > 0 {
                        i64::from(north_row[x + 2])
                    } else {
                        top_right
                    },
                };
                let mut props = [0i32; NUM_MA_PROPS];
                props[0] = chan as i32;
                props[1] = stream;
                props[2] = y as i32;
                props[3] = x as i32;
                props[4] = north.abs() as i32;
                props[5] = left.abs() as i32;
                props[6] = north as i32;
                props[7] = left as i32;
                props[8] = (left - previous_local_gradient as i64) as i32;
                props[9] = local_gradient;
                props[10] = (left - top_left) as i32;
                props[11] = (top_left - north) as i32;
                props[12] = (north - top_right) as i32;
                props[13] = (north - top_top) as i32;
                props[14] = (left - neighbors.left_left) as i32;
                props[15] = if disabled { 0 } else { wp.wp_prop as i32 };
                ref_props(refs, x, y, &mut props);

                let mut tok = [0u8; NUM_MA_PREDS];
                for pred in 0..NUM_MA_PREDS as u32 {
                    let pv = predictor_value(pred, neighbors, wp_pred);
                    let (t, _, _) = uint_encode(pack_signed((value - pv) as i32));
                    tok[pred as usize] = t.min(u8::MAX as u32) as u8;
                }
                push(props, tok, value as i32);
            }
            previous_local_gradient = local_gradient;
            west_west = west;
            west = value;
        }
        north_north_row = north_row;
        north_row = current_row;
    }
}

/// Tokenize the channel rectangle through the learned tree: every pixel is
/// routed to its leaf's context and coded with its leaf's predictor.
pub(super) fn tokenize_channel_ma<'a, T: Copy + 'a>(
    get_row: impl Fn(usize) -> &'a [T],
    gw: usize,
    gh: usize,
    chan: u32,
    stream: i32,
    refs: &[MaRefPlane<'a>],
    wp_params: WpParams,
    use_wp: bool,
    lookup: &MaLookup,
    offsets: &[i32],
    out: &mut RawTokens,
) where
    i64: From<T>,
{
    // Keep the traversal's stores simple, then validate and pack one row at
    // a time. The temporary row fits in cache and is reused for the channel.
    let mut row = Vec::with_capacity(gw);
    walk_channel_ma(
        get_row,
        gw,
        gh,
        chan,
        stream,
        refs,
        wp_params,
        use_wp && lookup.needs_wp(),
        |x, _y, v, p, n, wp_pred| {
            let (context, pred) = lookup.lookup(p);
            let pv = predictor_value(pred, n, wp_pred);
            row.push(Token::new(
                context,
                pack_signed((v - pv) as i32 - offsets[context as usize]),
            ));
            if x + 1 == gw {
                out.extend(&row);
                row.clear();
            }
        },
    );
}

/// One VarDCT DC group's quantized planes in modular channel order (Y, X, B).
pub(crate) struct DcPlanes {
    pub(crate) planes: [Vec<i32>; 3],
    pub(crate) w: usize,
    pub(crate) h: usize,
}

impl DcPlanes {
    /// Previously coded channels of `c`, most recent first, as the decoder's
    /// reference properties see them.
    pub(super) fn refs(&self, c: usize) -> Vec<MaRefPlane<'_>> {
        (0..c)
            .rev()
            .take(MA_REF_CHANNELS)
            .map(|r| MaRefPlane {
                pixels: &self.planes[r],
                stride: self.w,
                x0: 0,
                y0: 0,
            })
            .collect()
    }
}

/// MA samples of every DC group (stream id `1 + group`, default WP), taking
/// every `stride`-th pixel.
pub(crate) fn sample_dc_planes(
    groups: &[DcPlanes],
    stride: usize,
    wp_params: WpParams,
) -> MaSamples {
    let capacity = groups
        .iter()
        .map(|g| 3 * ma_channel_sample_count(g.w, g.h, stride))
        .sum();
    let mut samples = MaSamples::with_capacity(capacity);
    for (group, g) in groups.iter().enumerate() {
        if g.w == 0 || g.h == 0 {
            continue;
        }
        for c in 0..3 {
            let refs = g.refs(c);
            let pixels = &g.planes[c];
            sample_channel_ma(
                |y| &pixels[y * g.w..(y + 1) * g.w],
                g.w,
                g.h,
                c as u32,
                1 + group as i32,
                &refs,
                wp_params,
                true,
                stride,
                |props, tok, _| samples.push(props, tok),
            );
        }
    }
    samples
}

/// Tokenize one DC group through a learned tree: every pixel takes its leaf's
/// context (`leaf_ctx`, indexed by tree node) and predictor.
pub(crate) fn tokenize_dc_planes(
    g: &DcPlanes,
    group: usize,
    tree: &LearnedTree,
    leaf_ctx: &[u32],
    wp_params: WpParams,
    out: &mut Vec<Token>,
) {
    if g.w == 0 || g.h == 0 {
        return;
    }
    for c in 0..3 {
        let refs = g.refs(c);
        let pixels = &g.planes[c];
        walk_channel_ma(
            |y| &pixels[y * g.w..(y + 1) * g.w],
            g.w,
            g.h,
            c as u32,
            1 + group as i32,
            &refs,
            wp_params,
            true,
            |_x, _y, v, p, n, wp_pred| {
                let (node, pred) = tree.lookup(p);
                let pv = predictor_value(pred, n, wp_pred);
                out.push(Token::new(
                    leaf_ctx[node as usize],
                    pack_signed((v - pv) as i32),
                ));
            },
        );
    }
}
