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

//! Tests for predictor.

use super::super::lz77::RunLzWriter;
use super::cost::{SqueezePredictorCost, choose_predictor_for_plane, wp_sample_cost};
use super::tokens::{
    channel_to_context, tokenize_channel_with_wp, tokenize_plane, tokenize_plane_rows,
    tokenize_sample_rows,
};
use super::weighted::{WP_CONTEXT_BUCKETS, WpChannelContexts, tokenize_plane_with_wp};
use super::{
    GradPackInteriorFn, GradientScratch, PREDICTOR_AVERAGE4, PREDICTOR_GRADIENT, PREDICTOR_LEFT,
    PREDICTOR_SELECT, PREDICTOR_TOP, PREDICTOR_WEIGHTED, PredictorNeighbors, SLOW_PREDICTORS,
    grad_pack_interior_scalar, predictor_neighbors, predictor_value,
    selected_grad_pack_interior_fn, visit_predictor_rows,
};
use crate::encode_image::AlphaPlane;
use crate::entropy::pack_signed;
use crate::image::Image3Si;
use crate::weighted_predictor::{WpParams, WpState};

#[test]
fn mixed_wp_contexts_preserve_other_channels_predictors_and_leaf_contexts() {
    let (width, height) = (17, 11);
    let mut image = Image3Si::new(width, height);
    for c in 0..3 {
        for y in 0..height {
            for (x, v) in image.plane_row_mut(c, y).iter_mut().enumerate() {
                *v = ((x * 37 + y * 19 + c * 11) % 97) as i32 - 48;
            }
        }
    }
    let alpha = AlphaPlane::from_u8(
        (0..width * height)
            .map(|i| {
                if (i % width + i / width) % 7 < 3 {
                    0
                } else {
                    255
                }
            })
            .collect(),
    );
    let leaf_context = 42;
    let buckets = WpChannelContexts {
        contexts: [leaf_context; WP_CONTEXT_BUCKETS],
        shift: 6,
    };
    for chan in [0, 3] {
        for pred in SLOW_PREDICTORS
            .into_iter()
            .filter(|&p| p != PREDICTOR_WEIGHTED)
        {
            let mut predictors = [PREDICTOR_WEIGHTED; 4];
            predictors[chan] = pred;
            let mut actual = Vec::new();
            tokenize_channel_with_wp(
                &image,
                Some(&alpha),
                width,
                0,
                0,
                width,
                height,
                3,
                &predictors,
                chan,
                grad_pack_interior_scalar,
                &mut GradientScratch::default(),
                &mut actual,
                WpParams::DEFAULT,
                Some(&buckets),
            );
            let mut expected = Vec::new();
            tokenize_plane_with_wp(
                leaf_context,
                |x, y| {
                    if chan == 3 {
                        alpha.get_i32(y * width + x)
                    } else {
                        image.plane_row(chan, y)[x]
                    }
                },
                width,
                height,
                pred,
                grad_pack_interior_scalar,
                &mut GradientScratch::default(),
                &mut expected,
                WpParams::DEFAULT,
            );
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a.context, a.value) == (b.context, b.value)),
                "channel {chan}, predictor {pred}"
            );
        }
    }
}

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
fn row_costs_preserve_every_wp_preset_and_zero_histogram() {
    let (w, h) = (33, 7);
    let data: Vec<i32> = (0..w * h)
        .map(|i| {
            if i / w < 3 {
                37
            } else {
                ((i * 97 + i / w * 31) % 2048) as i32 - 1024
            }
        })
        .collect();
    let get = |x, y| data[y * w + x];
    for params in WpParams::PRESETS {
        for use_wp in [false, true] {
            let mut reference = SqueezePredictorCost::with_zero();
            let mut wp = WpState::with_params(w, params);
            for y in 0..h {
                for x in 0..w {
                    let n = predictor_neighbors(&get, x, y, w);
                    let value = get(x, y) as i64;
                    let weighted = if use_wp {
                        wp.predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top)
                    } else {
                        0
                    };
                    reference.costs.add(value, n, weighted);
                    if use_wp {
                        wp.update(value, x, y);
                    }
                }
            }
            for flat in [false, true] {
                let mut actual = SqueezePredictorCost::with_zero();
                if flat {
                    actual.add_rows_shared::<_, true>(
                        &|y| &data[y * w..][..w],
                        w,
                        h,
                        use_wp,
                        params,
                    );
                } else {
                    actual.add_rows_shared::<_, false>(
                        &|y| &data[y * w..][..w],
                        w,
                        h,
                        use_wp,
                        params,
                    );
                }
                assert_eq!(actual.costs.histograms, reference.costs.histograms);
                assert_eq!(actual.costs.zero_hist, reference.costs.zero_hist);
                assert_eq!(actual.costs.total, reference.costs.total);
                assert_eq!(actual.predictor(use_wp), reference.predictor(use_wp));
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
fn gradient_simd_matches_scalar_at_chunk_boundaries() {
    let mut kernels: Vec<(&str, GradPackInteriorFn)> = Vec::new();
    kernels.push(("selected", selected_grad_pack_interior_fn()));
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        kernels.push(("avx2", |c, p, o, g| unsafe {
            crate::avx::grad_pack_interior(c, p, o, g)
        }));
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        kernels.push(("sse4.1", |c, p, o, g| unsafe {
            crate::sse::grad_pack_interior(c, p, o, g)
        }));
    }

    let extremes = [i32::MIN, i32::MAX, -1, 0, 1, i32::MIN / 2, i32::MAX / 2];
    let mut seed = 0x1234_5678u32;
    for width in (0..=65).chain([127, 128, 129, 255, 256, 257]) {
        for offset in 0..4 {
            for padding in [0, 5] {
                for pattern in 0..3 {
                    let len = offset + width + padding;
                    let mut sample = |i: usize| match pattern {
                        0 => {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            seed as i32
                        }
                        1 => extremes[i % extremes.len()],
                        _ => (i as i32 * 977) % 65536 - 32768,
                    };
                    let cur: Vec<i32> = (0..len).map(&mut sample).collect();
                    let prev: Vec<i32> = (1..=len).map(&mut sample).collect();
                    let mut expected = vec![0xdead_beef; len];
                    grad_pack_interior_scalar(
                        &cur[offset..],
                        &prev[offset..],
                        &mut expected[offset..],
                        width,
                    );
                    for &(name, kernel) in &kernels {
                        let mut actual = vec![0xdead_beef; len];
                        kernel(
                            &cur[offset..],
                            &prev[offset..],
                            &mut actual[offset..],
                            width,
                        );
                        assert_eq!(
                            actual, expected,
                            "{name}: width={width}, offset={offset}, padding={padding}, pattern={pattern}"
                        );
                    }
                }
            }
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
                            let p =
                                wp.predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
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
                                        None,
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
                                        None,
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
                        assert!(
                            actual
                                .iter()
                                .zip(&reference)
                                .all(|(a, b)| (a.context, a.value) == (b.context, b.value))
                        );
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
