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

use super::cost::{SqueezePredictorCost, wp_sample_cost};
use super::tokens::{tokenize_channel_with_wp, tokenize_plane_rows};
use super::weighted::{WP_CONTEXT_BUCKETS, WpChannelContexts, tokenize_plane_with_wp};
use super::{
    GradPackInteriorFn, GradientScratch, PREDICTOR_AVERAGE4, PREDICTOR_LEFT, PREDICTOR_SELECT,
    PREDICTOR_TOP, PREDICTOR_WEIGHTED, PredictorNeighbors, SLOW_PREDICTORS,
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
fn gradient_simd_matches_scalar_at_chunk_boundaries() {
    // Only x86 builds push SIMD kernels after the portable one.
    #[allow(unused_mut)]
    let mut kernels: Vec<(&str, GradPackInteriorFn)> =
        vec![("selected", selected_grad_pack_interior_fn())];
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
fn wp_sample_cost_matches_point_fetched_corners() {
    for (width, height) in [(1, 4), (3, 129), (129, 3), (11, 13)] {
        let plane: Vec<i32> = (0..width * height)
            .map(|i| ((i * 977 + (i / width) * 619) % 65_536) as i32 - 32_768)
            .collect();
        let cw = width.min(128);
        let ch = height.min(128);
        for params in WpParams::PRESETS {
            let mut expected = (0, 0);
            for y0 in (0..=height - ch).step_by((height - ch).max(1)) {
                for x0 in (0..=width - cw).step_by((width - cw).max(1)) {
                    let get = |x, y| plane[(y0 + y) * width + x0 + x];
                    let mut wp = WpState::with_params(cw, params);
                    for y in 0..ch {
                        for x in 0..cw {
                            let n = predictor_neighbors(&get, x, y, cw);
                            let value = get(x, y);
                            let prediction =
                                wp.predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
                            let packed = pack_signed(value.wrapping_sub(prediction as i32));
                            let (_, extra, _) = crate::entropy::uint_encode(packed);
                            expected.0 +=
                                u64::from(extra + 32 - packed.saturating_add(1).leading_zeros());
                            expected.1 += 1;
                            wp.update(i64::from(value), x, y);
                        }
                    }
                }
            }
            assert_eq!(
                wp_sample_cost(|y| &plane[y * width..][..width], width, height, params),
                expected
            );
        }
    }
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
