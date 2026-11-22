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

//! Tests for palette.

use super::super::rct::inverse_ycocg;
use super::global::{
    PackedPalette, SCREEN_MIN_SHARE, bilevel_screen_share, build_global_palette,
    cooccurrence_chain_path, estimate_index_plane_cost, improve_chain_tour, index_cost_rows,
    path_cooccurrence, screen_share,
};
use super::visit_palette_colors;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;

#[test]
fn global_palette_reconstructs_rgb_and_alpha_from_input_tuple_ids() {
    use super::super::rct::forward_ycocg;

    let (w, h) = (37, 19);
    let pool = ThreadPool::new_lossless(3);
    let mut scratch = CoderScratch::lossless();
    for alpha_only in [false, true] {
        let colors: Vec<[i32; 4]> = (0..17)
            .map(|i| {
                if alpha_only {
                    [11, 22, 33, i * 3855]
                } else {
                    [i * 3855, 65535 - i * 1997, i * 977, i * 3855]
                }
            })
            .collect();
        let mut linear = Image3Si::new(w, h);
        let mut alpha = Vec::with_capacity(w * h);
        let mut expected = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let color = colors[(x / 3 + y * 7) % colors.len()];
                let (yy, co, cg) = forward_ycocg(color[0], color[1], color[2]);
                for (c, value) in [yy, co, cg].into_iter().enumerate() {
                    linear.plane_row_mut(c, y)[x] = value;
                }
                alpha.push(color[3] as u16);
                expected.push(color);
            }
        }
        let alpha = AlphaPlane::U16 {
            data: alpha,
            bits: 16,
        };
        let palette =
            build_global_palette(&linear, Some(&alpha), 3, w, h, &pool, &mut scratch).unwrap();
        assert_eq!(palette.nb_colors, colors.len());
        for (&index, color) in palette.indices.iter().zip(expected) {
            for (c, value) in color.into_iter().enumerate() {
                assert_eq!(
                    palette.palette[c * palette.nb_colors + index as usize],
                    value
                );
            }
        }
    }
}

#[test]
fn pooled_index_cost_matches_serial_histograms() {
    let entropy_of_hist = super::super::selected_entropy_of_hist_fn();
    const COLORS: usize = 17;
    for (w, h) in [(0, 0), (0, 5), (1, 1), (1, 17), (3, 5), (257, 259)] {
        let indices: Vec<u16> = (0..w * h)
            .map(|i| ((i * 7 + i / w.max(1) * 11) % COLORS) as u16)
            .collect();
        for order in [
            (0..COLORS as i32).collect::<Vec<_>>(),
            (0..COLORS as i32).rev().collect(),
            (0..COLORS).map(|i| ((i * 7) % COLORS) as i32).collect(),
        ] {
            let whole = index_cost_rows(&indices, w, 0, h, &order, COLORS);
            // Bands sum to the whole plane's histograms.
            for begin in [0, h / 2, h] {
                let mut bands = index_cost_rows(&indices, w, 0, begin, &order, COLORS);
                bands.add(&index_cost_rows(&indices, w, begin, h, &order, COLORS));
                assert_eq!(bands.tokens, whole.tokens, "{w}x{h}, begin={begin}");
                assert_eq!(bands.raw_bits, whole.raw_bits);
                assert_eq!(bands.values, whole.values);
            }
            let reference = whole.cost(entropy_of_hist);
            for threads in [1, 2, 8] {
                let pool = ThreadPool::new_lossless(threads);
                let mut scratch = CoderScratch::lossless();
                let cost = estimate_index_plane_cost(&indices, w, h, &order, &pool, &mut scratch);
                assert_eq!(
                    cost.to_bits(),
                    reference.to_bits(),
                    "{w}x{h}, threads={threads}"
                );
            }
        }
    }
}

#[test]
fn chain_tour_never_lowers_the_path_cooccurrence() {
    let mut state = 0x9e37_79b9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut strictly_better = 0usize;
    for &n in &[1usize, 2, 3, 5, 17, 64] {
        for _ in 0..8 {
            // Sparse symmetric weights: most pairs never meet.
            let mut cooc = vec![0u32; n * n];
            for a in 0..n {
                for b in a + 1..n {
                    let v = next();
                    let weight = if v % 3 == 0 { (v >> 8) % 1000 } else { 0 };
                    cooc[a * n + b] = weight;
                    cooc[b * n + a] = weight;
                }
            }
            let chain = cooccurrence_chain_path(n, &cooc);
            let tour = improve_chain_tour(n, &cooc, chain.clone());
            let mut sorted = tour.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            let (before, after) = (
                path_cooccurrence(n, &cooc, &chain),
                path_cooccurrence(n, &cooc, &tour),
            );
            assert!(after >= before, "{after} < {before} at n={n}");
            strictly_better += usize::from(after > before);
        }
    }
    assert!(strictly_better > 0);
}

#[test]
fn palette_scan_matches_indexed_crops_and_early_rejection() {
    for stride in [1, 3, 33, 257] {
        let height = 5;
        let mut linear = Image3Si::new(stride, height);
        for c in 0..3 {
            for y in 0..height {
                for (x, value) in linear.plane_row_mut(c, y).iter_mut().enumerate() {
                    *value = ((x * 977 + y * 619 + c * 71) & 0xffff) as i32 - 32768;
                }
            }
        }
        let n = stride * height;
        let alphas = [
            None,
            Some(AlphaPlane::U8((0..n).map(|i| i as u8).collect())),
            Some(AlphaPlane::U16 {
                data: (0..n).map(|i| (i * 71) as u16).collect(),
                bits: 16,
            }),
            Some(AlphaPlane::F32((0..n).map(|i| !(i as i32)).collect())),
        ];
        for alpha in &alphas {
            for rgb in [false, true] {
                for (x0, y0, w, h) in [
                    (0, 0, stride, height),
                    (stride / 2, 2, stride - stride / 2, 2),
                    (0, 0, 0, height),
                    (0, 0, stride, 0),
                ] {
                    let mut expected = Vec::new();
                    for y in y0..y0 + h {
                        for x in x0..x0 + w {
                            let i = y * stride + x;
                            let (a, b, c) = (
                                linear.plane_data(0)[i],
                                linear.plane_data(1)[i],
                                linear.plane_data(2)[i],
                            );
                            let (a, b, c) = if rgb {
                                inverse_ycocg(a, b, c)
                            } else {
                                (a, b, c)
                            };
                            expected.push([a, b, c, alpha.as_ref().map_or(0, |a| a.get_i32(i))]);
                        }
                    }
                    for stop in [1, stride + 1, usize::MAX] {
                        let mut actual = Vec::new();
                        let visit = |color| {
                            actual.push(color);
                            actual.len() < stop
                        };
                        let complete = if rgb {
                            visit_palette_colors::<true>(
                                &linear,
                                alpha.as_ref(),
                                stride,
                                x0,
                                y0,
                                w,
                                h,
                                visit,
                            )
                        } else {
                            visit_palette_colors::<false>(
                                &linear,
                                alpha.as_ref(),
                                stride,
                                x0,
                                y0,
                                w,
                                h,
                                visit,
                            )
                        };
                        assert_eq!(actual, expected[..expected.len().min(stop)]);
                        assert_eq!(complete, expected.len() < stop);
                    }
                }
            }
        }
    }
}

/// A gray plane of `levels` values chosen by `level_at`.
fn gray_plane(w: usize, h: usize, level_at: impl Fn(usize, usize) -> i32) -> Image3Si {
    let mut linear = Image3Si::new(w, h);
    for y in 0..h {
        for x in 0..w {
            linear.plane_row_mut(0, y)[x] = level_at(x, y);
        }
    }
    linear
}

#[test]
fn packed_palette_inverts_through_the_decoder_squeeze() {
    let pool = ThreadPool::new_lossless(2);
    let mut scratch = CoderScratch::lossless();
    let (w, h) = (38, 11);
    let bilevel = |x: usize, y: usize| 255 * i32::from((x * x + 3 * y + x / 5) % 3 == 0);
    let linear = gray_plane(w, h, bilevel);
    let palette = build_global_palette(&linear, None, 1, w, h, &pool, &mut scratch).unwrap();
    assert_eq!(palette.nb_colors, 2);
    let packed = PackedPalette::new(&palette, w, h).unwrap();
    assert_eq!(packed.packed_colors, 4);
    assert_eq!(packed.indices.len(), w / 2 * h);
    // The decoder's inverse horizontal Squeeze with a zero smoothing term.
    let unpacked: Vec<u16> = packed
        .indices
        .iter()
        .flat_map(|&index| {
            let average = packed.tuples[usize::from(index)];
            let residual = packed.tuples[packed.packed_colors + usize::from(index)];
            let parity = if residual > 0 {
                -(residual & 1)
            } else {
                residual & 1
            };
            let left = (2 * average + residual + parity) >> 1;
            [left as u16, (left - residual) as u16]
        })
        .collect();
    assert_eq!(unpacked, palette.indices);
    let source = packed.source(w, h);
    assert!(packed.horizontal);
    assert_eq!((source.hshift, source.vshift), (1, 0));
    assert_eq!(
        source
            .channels
            .iter()
            .map(|c| (c.w, c.h, c.meta))
            .collect::<Vec<_>>(),
        [(4, 2, true), (2, 1, true), (w / 2, h, false)]
    );

    // An odd width pairs along columns when the height is even.
    let (w, h) = (37, 12);
    let linear = gray_plane(w, h, bilevel);
    let palette = build_global_palette(&linear, None, 1, w, h, &pool, &mut scratch).unwrap();
    let packed = PackedPalette::new(&palette, w, h).unwrap();
    assert!(!packed.horizontal);
    let source = packed.source(w, h);
    assert_eq!((source.hshift, source.vshift), (0, 1));
    assert_eq!((source.channels[2].w, source.channels[2].h), (w, h / 2));
    for (i, &index) in packed.indices.iter().enumerate() {
        let (x, y) = (i % w, i / w);
        let average = packed.tuples[usize::from(index)];
        let residual = packed.tuples[packed.packed_colors + usize::from(index)];
        let top = (2 * average
            + residual
            + if residual > 0 {
                -(residual & 1)
            } else {
                residual & 1
            })
            >> 1;
        assert_eq!(top as u16, palette.indices[2 * y * w + x]);
        assert_eq!(
            (top - residual) as u16,
            palette.indices[(2 * y + 1) * w + x]
        );
    }

    // Sizes odd both ways, and palettes of another size, stay unpacked.
    let (w, h) = (38, 11);
    let odd = gray_plane(w - 1, h, bilevel);
    let odd_palette = build_global_palette(&odd, None, 1, w - 1, h, &pool, &mut scratch).unwrap();
    assert!(PackedPalette::new(&odd_palette, w - 1, h).is_none());
    let three = gray_plane(w, h, |x, y| 100 * ((x + y) % 3) as i32);
    let three_palette = build_global_palette(&three, None, 1, w, h, &pool, &mut scratch).unwrap();
    assert!(PackedPalette::new(&three_palette, w, h).is_none());

    // A plane without one of the four pairs keeps a shorter second palette.
    let stripes = gray_plane(w, h, |x, _| 255 * (x % 2) as i32);
    let stripes_palette =
        build_global_palette(&stripes, None, 1, w, h, &pool, &mut scratch).unwrap();
    let packed = PackedPalette::new(&stripes_palette, w, h).unwrap();
    assert_eq!(packed.packed_colors, 1);
    assert!(packed.indices.iter().all(|&index| index == 0));
}

#[test]
fn screen_share_tells_a_dot_screen_from_strokes() {
    let (w, h) = (256, 96);
    // A 45-degree dot screen of pitch 8: rows repeat eight rows up.
    let dots = |x: usize, y: usize| (x + y) % 8 < 3 && (x + 8 - y % 8) % 8 < 3;
    assert!(screen_share(dots, w, h) > 0.9);
    // Vertical rules and a diagonal stroke repeat from the row above or not
    // at all.
    let rules = |x: usize, _: usize| x % 23 < 2;
    assert_eq!(screen_share(rules, w, h), 0.0);
    let stroke = |x: usize, y: usize| x.abs_diff(2 * y) < 3;
    assert!(screen_share(stroke, w, h) < SCREEN_MIN_SHARE);
    // Blank and narrow images have no busy span.
    assert_eq!(screen_share(|_, _| false, w, h), 0.0);
    assert_eq!(screen_share(dots, 63, h), 0.0);

    // Two colors qualify whatever they are; a third does not.
    let mut linear = Image3Si::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let value = if dots(x, y) { [7, -3, 2] } else { [200, 0, 0] };
            for (c, v) in value.into_iter().enumerate() {
                linear.plane_row_mut(c, y)[x] = v;
            }
        }
    }
    assert!(bilevel_screen_share(&linear).unwrap() > 0.9);
    linear.plane_row_mut(1, 5)[9] = 77;
    assert!(bilevel_screen_share(&linear).is_none());
    let flat = Image3Si::new(w, h);
    assert!(bilevel_screen_share(&flat).is_none());
}
