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
    ColorHasher, build_global_palette, cooccurrence_chain, cooccurrence_chain_path,
    estimate_index_plane_cost, improve_chain_tour, index_cooccurrence, index_cost_rows,
    path_cooccurrence,
};
use super::visit_palette_colors;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;

#[test]
fn color_hasher_accepts_array_hashing_and_uses_every_channel() {
    use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

    let builder = BuildHasherDefault::<ColorHasher>::default();
    let mut hashes = std::collections::HashSet::new();
    for channel in 0..4 {
        for value in [i32::MIN, -65535, -1, 1, 255, 65535, i32::MAX] {
            let mut color = [0i32; 4];
            color[channel] = value;
            let mut bytes = [0; 16];
            for (part, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(color) {
                *part = value.to_ne_bytes();
            }
            let mut raw = ColorHasher::default();
            raw.write(&bytes);
            let hash = builder.hash_one(color);
            // Exercise the standard library's actual array Hash protocol,
            // including its constant length prefix, against a raw payload.
            assert_eq!(hash, raw.finish());
            assert!(hashes.insert(hash), "channel={channel}, value={value}");
        }
    }
}

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
fn cooccurrence_chain_recovers_a_scrambled_ramp() {
    // A horizontal ramp of 40 levels whose canonical ids are a random
    // permutation of the ramp position: the chain must put the levels
    // back in ramp order (either direction), which luma order cannot.
    let (w, h, n) = (200usize, 8usize, 40usize);
    let mut perm: Vec<u16> = (0..n as u16).collect();
    let mut state = 0x2545_F491u32;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        perm.swap(i, state as usize % (i + 1));
    }
    let indices: Vec<u16> = (0..h)
        .flat_map(|_| (0..w).map(|x| perm[x * n / w]))
        .collect();
    let cooc = index_cooccurrence(&indices, w, h, n);
    let order = cooccurrence_chain(n, &cooc);
    let ramp: Vec<i32> = (0..n).map(|level| order[perm[level] as usize]).collect();
    let ascending = ramp.array_windows::<2>().all(|p| p[1] == p[0] + 1);
    let descending = ramp.array_windows::<2>().all(|p| p[1] == p[0] - 1);
    assert!(ascending || descending, "chain order {ramp:?}");
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
