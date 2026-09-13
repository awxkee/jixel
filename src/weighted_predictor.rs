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

use crate::bit_writer::BitWriter;

// Self-correcting Weighted Predictor (WP), bit-faithful to libjxl's
// `weighted::State` in context_predict.h.
const WP_EXTRA_BITS: i64 = 3;
const WP_PRED_ROUND: i64 = ((1 << WP_EXTRA_BITS) >> 1) - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WpParams {
    w: [u32; 4],
    p: [i64; 7],
}

impl WpParams {
    pub(crate) const PRESETS: [Self; 4] = [
        // libjxl mode 0: lossless16 and the bitstream default.
        Self {
            w: [0xd, 0xc, 0xc, 0xc],
            p: [16, 10, 7, 7, 7, 0, 0],
        },
        // libjxl mode 1: default lossless8.
        Self {
            w: [0xd, 0xc, 0xc, 0xb],
            p: [8, 8, 4, 0, 3, 23, 2],
        },
        // libjxl mode 2: west-oriented lossless8.
        Self {
            w: [0xd, 0xc, 0xd, 0xc],
            p: [10, 9, 7, 0, 0, 16, 9],
        },
        // libjxl mode 3: north-oriented lossless8.
        Self {
            w: [0xd, 0xd, 0xc, 0xc],
            p: [16, 8, 0, 16, 0, 23, 0],
        },
    ];
    pub(crate) const DEFAULT: Self = Self::PRESETS[0];
}

pub(crate) fn write_wp_header(params: WpParams, writer: &mut BitWriter) {
    if params == WpParams::DEFAULT {
        writer.write(1, 1);
        return;
    }
    writer.write(1, 0);
    for &param in &params.p {
        writer.write(5, param as u64);
    }
    for &weight in &params.w {
        writer.write(4, weight as u64);
    }
}

// divlookup[i] = (1<<24)/(i+1)
static WP_DIV: [u32; 64] = [
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201, 1398101, 1290555, 1198372, 1118481, 1048576, 986895, 932067, 883011, 838860, 798915,
    762600, 729444, 699050, 671088, 645277, 621378, 599186, 578524, 559240, 541200, 524288, 508400,
    493447, 479349, 466033, 453438, 441505, 430185, 419430, 409200, 399457, 390167, 381300, 372827,
    364722, 356962, 349525, 342392, 335544, 328965, 322638, 316551, 310689, 305040, 299593, 294337,
    289262, 284359, 279620, 275036, 270600, 266305, 262144,
];

pub(crate) struct WpState {
    xsize: usize,
    pred_errors: [Vec<u32>; 4],
    error: Vec<i64>,
    prediction: [i64; 4],
    pred: i64,
    /// libjxl property kWPProp (p[15]): the signed neighbor WP-error with the
    /// largest absolute value among {W, N, NW, NE}. Set on each prediction.
    pub(crate) wp_prop: i64,
    params: WpParams,
}

#[derive(Clone, Copy)]
pub(crate) struct WpRowOffsets {
    current: usize,
    previous: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct WpNeighbors {
    pub(crate) north: i64,
    pub(crate) west: i64,
    pub(crate) north_east: i64,
    pub(crate) north_west: i64,
    pub(crate) north_north: i64,
}

impl WpState {
    pub(crate) fn new(xsize: usize) -> Self {
        Self::with_params(xsize, WpParams::DEFAULT)
    }

    pub(crate) fn with_params(xsize: usize, params: WpParams) -> Self {
        let n = (xsize + 2) * 2;
        Self {
            xsize,
            pred_errors: [vec![0u32; n], vec![0u32; n], vec![0u32; n], vec![0u32; n]],
            error: vec![0i64; n],
            prediction: [0; 4],
            pred: 0,
            wp_prop: 0,
            params,
        }
    }

    pub(crate) fn reset(&mut self, xsize: usize, params: WpParams) {
        let n = (xsize + 2) * 2;
        for errors in &mut self.pred_errors {
            errors.resize(n, 0);
            errors.fill(0);
        }
        self.error.resize(n, 0);
        self.error.fill(0);
        self.xsize = xsize;
        self.params = params;
        self.prediction = [0; 4];
        self.pred = 0;
        self.wp_prop = 0;
    }

    #[inline]
    fn add_bits(x: i64) -> i64 {
        x << WP_EXTRA_BITS
    }

    #[inline]
    fn floor_log2(x: u64) -> u32 {
        debug_assert!(x >= 1);
        63 - x.leading_zeros()
    }

    #[inline]
    fn error_weight(x: u64, maxweight: u32) -> u32 {
        let mut shift = Self::floor_log2(x + 1) as i32 - 5;
        if shift < 0 {
            shift = 0;
        }
        4 + (((maxweight as u64 * WP_DIV[(x >> shift) as usize] as u64) >> shift) as u32)
    }

    #[inline]
    fn weighted_average(pred: &[i64; 4], w_in: &[u32; 4]) -> i64 {
        let mut weight_sum: u32 = w_in.iter().sum();
        let log_weight = Self::floor_log2(weight_sum as u64);
        let mut w = [0u32; 4];
        weight_sum = 0;
        for i in 0..4 {
            w[i] = w_in[i] >> (log_weight - 4);
            weight_sum += w[i];
        }
        let mut sum: i64 = (weight_sum as i64 >> 1) - 1;
        for i in 0..4 {
            sum += pred[i] * w[i] as i64;
        }
        (sum * WP_DIV[(weight_sum - 1) as usize] as i64) >> 24
    }

    #[inline]
    pub(crate) fn row_offsets(&self, y: usize) -> WpRowOffsets {
        let second = self.xsize + 2;
        if y & 1 == 1 {
            WpRowOffsets {
                current: 0,
                previous: second,
            }
        } else {
            WpRowOffsets {
                current: second,
                previous: 0,
            }
        }
    }

    #[inline(always)]
    fn predict_with_row<const FLAT_SHORTCUT: bool>(
        &mut self,
        x: usize,
        row: WpRowOffsets,
        n: WpNeighbors,
    ) -> (i64, bool) {
        let pos_n = row.previous + x;
        let pos_ne = if x < self.xsize - 1 { pos_n + 1 } else { pos_n };
        let pos_nw = if x > 0 { pos_n - 1 } else { pos_n };
        // Equal neighbors and zero WP errors make every subprediction equal.
        // Normalized weights total 13..=31; within +/-65536 fixed-point units
        // the reciprocal's truncation error stays below its rounding bias, so
        // the weighted average and clamped prediction equal `common` exactly.
        // Use this shortcut for MA and palette traversals: checking it in general
        // prediction loops costs more than it saves on photographic inputs.
        if FLAT_SHORTCUT
            && n.north == n.west
            && n.north == n.north_east
            && n.north == n.north_west
            && n.north == n.north_north
            && (-8192..=8192).contains(&n.north)
            && (x == 0 || self.error[row.current + x - 1] == 0)
            && self.error[pos_n] == 0
            && self.error[pos_nw] == 0
            && self.error[pos_ne] == 0
        {
            let common = Self::add_bits(n.north);
            self.prediction = [common; 4];
            self.pred = common;
            self.wp_prop = 0;
            return (n.north, true);
        }
        let mut weights = [0u32; 4];
        for i in 0..4 {
            let s = self.pred_errors[i][pos_n] as u64
                + self.pred_errors[i][pos_ne] as u64
                + self.pred_errors[i][pos_nw] as u64;
            weights[i] = Self::error_weight(s, self.params.w[i]);
        }
        let an = Self::add_bits(n.north);
        let aw = Self::add_bits(n.west);
        let ane = Self::add_bits(n.north_east);
        let anw = Self::add_bits(n.north_west);
        let ann = Self::add_bits(n.north_north);
        let te_w = if x == 0 {
            0
        } else {
            self.error[row.current + x - 1]
        };
        let te_n = self.error[pos_n];
        let te_nw = self.error[pos_nw];
        let te_ne = self.error[pos_ne];
        let mut wp_prop = te_w;
        if te_n.abs() > wp_prop.abs() {
            wp_prop = te_n;
        }
        if te_nw.abs() > wp_prop.abs() {
            wp_prop = te_nw;
        }
        if te_ne.abs() > wp_prop.abs() {
            wp_prop = te_ne;
        }
        self.wp_prop = wp_prop;
        let s_wn = te_n + te_w;
        self.prediction[0] = aw + ane - an;
        self.prediction[1] = an - (((s_wn + te_ne) * self.params.p[0]) >> 5);
        self.prediction[2] = aw - (((s_wn + te_nw) * self.params.p[1]) >> 5);
        self.prediction[3] = an
            - ((te_nw * self.params.p[2]
                + te_n * self.params.p[3]
                + te_ne * self.params.p[4]
                + (ann - an) * self.params.p[5]
                + (anw - aw) * self.params.p[6])
                >> 5);
        let pred = Self::weighted_average(&self.prediction, &weights);
        if ((te_n ^ te_w) | (te_n ^ te_nw)) > 0 {
            self.pred = pred;
            ((pred + WP_PRED_ROUND) >> WP_EXTRA_BITS, false)
        } else {
            let mx = aw.max(ane).max(an);
            let mn = aw.min(ane).min(an);
            let predc = pred.max(mn).min(mx);
            self.pred = predc;
            ((predc + WP_PRED_ROUND) >> WP_EXTRA_BITS, false)
        }
    }

    #[inline(always)]
    fn update_with_row(&mut self, val: i64, x: usize, row: WpRowOffsets) {
        let valb = Self::add_bits(val);
        self.error[row.current + x] = self.pred - valb;
        for i in 0..4 {
            let e = ((self.prediction[i] - valb).abs() + WP_PRED_ROUND) >> WP_EXTRA_BITS;
            self.pred_errors[i][row.current + x] = e as u32;
            let slot = &mut self.pred_errors[i][row.previous + x + 1];
            *slot = slot.wrapping_add(e as u32);
        }
    }

    /// Complete one sequential WP state transition using row offsets already
    /// selected by the caller's row kernel.
    /// Share the full transition across typed row kernels and border handlers;
    /// the private helpers stay fused within this operation.
    pub(crate) fn predict_and_update(
        &mut self,
        val: i64,
        x: usize,
        row: WpRowOffsets,
        neighbors: WpNeighbors,
    ) -> i64 {
        let (prediction, _) = self.predict_with_row::<false>(x, row, neighbors);
        self.update_with_row(val, x, row);
        prediction
    }

    /// MA and palette traversals benefit from the exact flat-span shortcut.
    pub(crate) fn predict_and_update_flat(
        &mut self,
        val: i64,
        x: usize,
        row: WpRowOffsets,
        neighbors: WpNeighbors,
    ) -> i64 {
        self.predict_and_update_flat_inlined(val, x, row, neighbors)
    }

    #[inline(always)]
    pub(crate) fn predict_and_update_flat_inlined(
        &mut self,
        val: i64,
        x: usize,
        row: WpRowOffsets,
        neighbors: WpNeighbors,
    ) -> i64 {
        let (prediction, flat) = self.predict_with_row::<true>(x, row, neighbors);
        if flat && prediction == val {
            // The shortcut makes all four predictions exactly val << 3.
            // Clear the current errors; adding zero to the north-east
            // subpredictor errors would leave that state unchanged.
            self.error[row.current + x] = 0;
            for errors in &mut self.pred_errors {
                errors[row.current + x] = 0;
            }
        } else {
            self.update_with_row(val, x, row);
        }
        prediction
    }

    /// Compatibility API for callers that perform work between prediction and
    /// error-state update.
    pub(crate) fn predict(
        &mut self,
        x: usize,
        y: usize,
        north: i64,
        west: i64,
        north_east: i64,
        north_west: i64,
        north_north: i64,
    ) -> i64 {
        self.predict_with_row::<false>(
            x,
            self.row_offsets(y),
            WpNeighbors {
                north,
                west,
                north_east,
                north_west,
                north_north,
            },
        )
        .0
    }

    #[inline]
    pub(crate) fn update(&mut self, val: i64, x: usize, y: usize) {
        self.update_with_row(val, x, self.row_offsets(y));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_predictions_keep_the_exact_fixed_point_average() {
        for total in 13i64..=31 {
            let reciprocal = WP_DIV[total as usize - 1] as i64;
            for prediction in -65_536i64..=65_536 {
                let sum = prediction * total + (total >> 1) - 1;
                assert_eq!((sum * reciprocal) >> 24, prediction);
            }
        }
        let mut seed = 17u32;
        for _ in 0..10_000 {
            let weights: [u32; 4] = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                4 + seed % (15 << 24)
            });
            let sum: u32 = weights.iter().sum();
            let shift = WpState::floor_log2(sum as u64) - 4;
            let normalized: u32 = weights.iter().map(|w| w >> shift).sum();
            assert!((13..=31).contains(&normalized));
            for prediction in [-65_536, -1, 0, 1, 65_536] {
                assert_eq!(
                    WpState::weighted_average(&[prediction; 4], &weights),
                    prediction
                );
            }
        }
    }

    #[test]
    fn flat_shortcut_preserves_prediction_and_error_state() {
        for params in WpParams::PRESETS {
            for width in [1, 17] {
                for value in [-8193, -8192, -1, 0, 1, 8192, 8193, 65535] {
                    let mut general = WpState::with_params(width, params);
                    let mut ma = WpState::with_params(width, params);
                    for (i, errors) in general.pred_errors.iter_mut().enumerate() {
                        for (j, error) in errors.iter_mut().enumerate() {
                            *error = (i * 191 + j * 73) as u32;
                        }
                    }
                    ma.pred_errors.clone_from(&general.pred_errors);
                    for y in 0..8 {
                        for x in 0..width {
                            let delta = if (2..5).contains(&y) {
                                x as i64 * 31 + y as i64 * 17
                            } else {
                                0
                            };
                            let neighbors = WpNeighbors {
                                north: value,
                                west: value + delta,
                                north_east: value - delta,
                                north_west: value + delta * 2,
                                north_north: value - delta * 3,
                            };
                            assert_eq!(
                                ma.predict_and_update_flat(
                                    value + delta,
                                    x,
                                    ma.row_offsets(y),
                                    neighbors,
                                ),
                                general.predict_and_update(
                                    value + delta,
                                    x,
                                    general.row_offsets(y),
                                    neighbors,
                                )
                            );
                            assert_eq!(ma.pred, general.pred);
                            assert_eq!(ma.prediction, general.prediction);
                            assert_eq!(ma.wp_prop, general.wp_prop);
                            assert_eq!(ma.error, general.error);
                            assert_eq!(ma.pred_errors, general.pred_errors);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reset_matches_fresh_state_after_width_and_parameter_changes() {
        let mut reused = WpState::new(0);
        for (round, width) in [17, 3, 257, 1, 17].into_iter().enumerate() {
            let params = WpParams::PRESETS[round % WpParams::PRESETS.len()];
            reused.reset(width, params);
            let mut fresh = WpState::with_params(width, params);
            for y in 0..5 {
                for x in 0..width {
                    let value = (x * 37 + y * 101 + round * 17) as i64;
                    let neighbors = WpNeighbors {
                        north: value - 13,
                        west: value + 7,
                        north_east: value - 29,
                        north_west: value + 3,
                        north_north: value - 19,
                    };
                    assert_eq!(
                        reused.predict_and_update(value, x, reused.row_offsets(y), neighbors),
                        fresh.predict_and_update(value, x, fresh.row_offsets(y), neighbors)
                    );
                    assert_eq!(reused.wp_prop, fresh.wp_prop);
                }
            }
        }
    }

    #[test]
    fn wp_headers_have_expected_sizes() {
        let mut default = BitWriter::new();
        write_wp_header(WpParams::DEFAULT, &mut default);
        assert_eq!(default.bits_written(), 1);

        for &params in &WpParams::PRESETS[1..] {
            let mut custom = BitWriter::new();
            write_wp_header(params, &mut custom);
            assert_eq!(custom.bits_written(), 52);
        }
    }
}
