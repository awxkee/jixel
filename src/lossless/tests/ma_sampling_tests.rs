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

use super::ma_sampling::walk_channel_ma;
use super::predictor::{predictor_neighbors, predictor_value};
use crate::ma_tree::NUM_MA_PREDS;
use crate::weighted_predictor::{WpParams, WpState};

#[test]
fn row_walk_matches_point_fetched_properties_and_predictions() {
    for width in [0usize, 1, 2, 3, 4, 5, 17, 257] {
        for height in [0usize, 1, 2, 7] {
            for flat in [false, true] {
                let plane: Vec<i32> = (0..width * height)
                    .map(|i| {
                        if flat {
                            8192
                        } else {
                            [i32::MIN, i32::MAX, -65536, 65535, -8192, 0, 8192]
                                [(i * 17 + i / width) % 7]
                        }
                    })
                    .collect();
                let get = |x: usize, y: usize| plane[y * width + x];
                for &params in &WpParams::PRESETS {
                    for use_wp in [false, true] {
                        let mut wp = WpState::with_params(width, params);
                        let mut previous_gradient = 0i32;
                        let mut count = 0;
                        walk_channel_ma(
                            |y| &plane[y * width..][..width],
                            width,
                            height,
                            2,
                            0,
                            &[],
                            params,
                            use_wp,
                            |x, y, value, props, neighbors, weighted| {
                                assert_eq!(count, y * width + x);
                                assert_eq!(value, get(x, y) as i64);
                                count += 1;
                                if x == 0 {
                                    previous_gradient = 0;
                                }
                                let n = predictor_neighbors(&get, x, y, width);
                                let expected_wp = if use_wp {
                                    wp.predict(
                                        x,
                                        y,
                                        n.top,
                                        n.left,
                                        n.top_right,
                                        n.top_left,
                                        n.top_top,
                                    )
                                } else {
                                    0
                                };
                                let gradient = (n.left + n.top - n.top_left) as i32;
                                // No reference planes: the extra properties stay 0.
                                assert!(props[16..].iter().all(|&v| v == 0));
                                assert_eq!(
                                    props[..16],
                                    [
                                        2,
                                        0,
                                        y as i32,
                                        x as i32,
                                        n.top.abs() as i32,
                                        n.left.abs() as i32,
                                        n.top as i32,
                                        n.left as i32,
                                        (n.left - previous_gradient as i64) as i32,
                                        gradient,
                                        (n.left - n.top_left) as i32,
                                        (n.top_left - n.top) as i32,
                                        (n.top - n.top_right) as i32,
                                        (n.top - n.top_top) as i32,
                                        (n.left - n.left_left) as i32,
                                        if use_wp { wp.wp_prop as i32 } else { 0 },
                                    ]
                                );
                                assert_eq!(weighted, expected_wp);
                                for pred in 0..NUM_MA_PREDS as u32 {
                                    assert_eq!(
                                        predictor_value(pred, neighbors, weighted),
                                        predictor_value(pred, n, expected_wp),
                                    );
                                }
                                previous_gradient = gradient;
                                if use_wp {
                                    wp.update(value, x, y);
                                }
                            },
                        );
                        assert_eq!(count, width * height);
                    }
                }
            }
        }
    }
}
