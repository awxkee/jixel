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

//! Predictor and weighted-predictor preset cost estimation.

use super::super::entropy::{EntropyOfHistFn, selected_entropy_of_hist_fn};
use super::row_kernels::{self, Rows, Sample};
use super::{
    PREDICTOR_GRADIENT, PREDICTOR_LEFT, PREDICTOR_SELECT, PREDICTOR_TOP, PREDICTOR_ZERO,
    SLOW_PREDICTORS, fixed_predictor, predictor_neighbors,
};
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::entropy::pack_signed;
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpNeighbors, WpParams, WpState};

/// Side of the corner crops `wp_sample_cost` scores.
const WP_SAMPLE_CROP: usize = 128;

/// The coordinates along one axis that `wp_sample_cost` reads, in order: a
/// packed plane of them holds the same crops.
fn wp_sample_coords(len: usize) -> impl Iterator<Item = usize> {
    let crop = len.min(WP_SAMPLE_CROP);
    let tail = if len > crop { len - crop..len } else { 0..0 };
    (0..crop).chain(tail)
}

pub(super) fn wp_sample_cost(
    get: impl Fn(usize, usize) -> i32,
    width: usize,
    height: usize,
    params: WpParams,
) -> (u64, usize) {
    let cw = width.min(WP_SAMPLE_CROP);
    let ch = height.min(WP_SAMPLE_CROP);
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

/// `ycocg` holds YCoCg planes (or one gray plane) and the color channels are
/// scored under `rct`: only the sampled corners are transformed.
pub(in crate::lossless) fn choose_wp_params(
    ycocg: &Image3Si,
    rct: u32,
    alpha: Option<&AlphaPlane>,
    num_color: usize,
    header_count: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> WpParams {
    let width = ycocg.xsize();
    let height = ycocg.ysize();
    let nb_chans = num_color + usize::from(alpha.is_some());
    let total_values = width.saturating_mul(height).saturating_mul(nb_chans);
    let packed_width = wp_sample_coords(width).count();
    let packed_height = wp_sample_coords(height).count();
    let corners = (num_color == 3 && rct != 6).then(|| {
        let mut planes = std::array::from_fn(|_| vec![0; packed_width * packed_height]);
        let [a, b, c] = &mut planes;
        let [p0, p1, p2] = [0, 1, 2].map(|c| ycocg.plane_data(c));
        let transform = crate::lossless::selected_rct_row_fn();
        let crop = width.min(WP_SAMPLE_CROP);
        let tail = if width > crop {
            width - crop..width
        } else {
            0..0
        };
        for (row, y) in wp_sample_coords(height).enumerate() {
            let mut offset = row * packed_width;
            for xs in [0..crop, tail.clone()] {
                let len = xs.len();
                let start = y * width + xs.start;
                transform(
                    [p0, p1, p2].map(|p| &p[start..start + len]),
                    rct,
                    [
                        &mut a[offset..offset + len],
                        &mut b[offset..offset + len],
                        &mut c[offset..offset + len],
                    ],
                );
                offset += len;
            }
        }
        planes
    });
    let scores = pool.steal_map(scratch, WpParams::PRESETS.len(), |preset, _scratch| {
        let params = WpParams::PRESETS[preset];
        let mut sample_cost = 0u64;
        let mut samples = 0usize;
        for channel in 0..num_color {
            let (cost, count) = match &corners {
                Some(planes) => {
                    let plane = &planes[channel];
                    wp_sample_cost(
                        |x, y| plane[y * packed_width + x],
                        packed_width,
                        packed_height,
                        params,
                    )
                }
                None => {
                    let plane = ycocg.plane_data(channel);
                    wp_sample_cost(|x, y| plane[y * width + x], width, height, params)
                }
            };
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
pub(crate) struct SqueezePredictorCost {
    pub(in crate::lossless) costs: PredictorCosts,
}

impl SqueezePredictorCost {
    pub(crate) fn with_zero() -> Self {
        let mut c = Self::default();
        c.costs.collect_zero = true;
        c
    }

    pub(crate) fn safe_predictor(&self) -> u32 {
        self.costs.best_safe_predictor()
    }

    /// Add one independently predicted modular crop. Weighted prediction state
    /// resets here because the decoder resets it for every modular sub-image.
    #[cfg(test)]
    pub(crate) fn add_crop(
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

    pub(in crate::lossless) fn predictor(&self, use_wp: bool) -> u32 {
        self.costs.best_predictor(use_wp)
    }

    pub(crate) fn add_rows<'a, T: Sample + 'a>(
        &mut self,
        get_row: impl Fn(usize) -> &'a [T],
        w: usize,
        h: usize,
        use_wp: bool,
    ) where
        i64: From<T>,
    {
        self.add_rows_shared::<_, true>(&get_row, w, h, use_wp, WpParams::DEFAULT);
    }

    // Getter dispatch is once per row, outside the specialized pixel loops.
    pub(super) fn add_rows_shared<'a, T: Sample + 'a, const FLAT_SHORTCUT: bool>(
        &mut self,
        get_row: &dyn Fn(usize) -> &'a [T],
        w: usize,
        h: usize,
        use_wp: bool,
        wp_params: WpParams,
    ) where
        i64: From<T>,
    {
        if w == 0 || h == 0 {
            return;
        }
        let kernels = row_kernels::selected();
        let mut wp = use_wp.then(|| WpState::with_params(w, wp_params));
        let (mut north, mut north_north) = (None, None);
        for y in 0..h {
            let current = get_row(y);
            let rows = Rows {
                current,
                north,
                north_north,
            };
            if let Some(wp) = &mut wp {
                let row = wp.row_offsets(y);
                super::visit_predictor_row(
                    current,
                    north.unwrap_or(current),
                    north_north.or(north).unwrap_or(current),
                    y == 0,
                    |x, value, n| {
                        let neighbors = WpNeighbors {
                            north: n.top,
                            west: n.left,
                            north_east: n.top_right,
                            north_west: n.top_left,
                            north_north: n.top_top,
                        };
                        let weighted = if FLAT_SHORTCUT {
                            wp.predict_and_update_flat(value, x, row, neighbors)
                        } else {
                            wp.predict_and_update(value, x, row, neighbors)
                        };
                        add_symbol(
                            &mut self.costs.histograms[0],
                            pack_signed((value - weighted) as i32),
                        );
                    },
                );
            }
            T::costs(
                kernels,
                rows,
                &mut self.costs.histograms,
                self.costs.collect_zero.then_some(&mut self.costs.zero_hist),
                wp.is_none(),
            );
            self.costs.total += w as u64;
            north_north = north;
            north = Some(current);
        }
    }
}

#[inline]
pub(super) fn add_symbol(hist: &mut Vec<u64>, symbol: u32) {
    let symbol = symbol as usize;
    if hist.len() <= symbol {
        hist.resize(symbol + 1, 0);
    }
    hist[symbol] += 1;
}

pub(in crate::lossless) struct PredictorCosts {
    pub(super) entropy_of_hist: EntropyOfHistFn,
    pub(super) histograms: [Vec<u64>; SLOW_PREDICTORS.len()],
    /// Residual histogram for the Zero predictor (the raw values). Collected
    /// only when `collect_zero` is set (the lossy-modular path); the lossless
    /// callers skip the extra bump.
    pub(super) zero_hist: Vec<u64>,
    pub(super) collect_zero: bool,
    pub(super) total: u64,
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
    #[cfg(test)]
    pub(super) fn add(&mut self, value: i64, neighbors: super::PredictorNeighbors, weighted: i64) {
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate() {
            let pred = super::predictor_value(pred_id, neighbors, weighted);
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
    pub(super) fn best_safe_predictor(&self) -> u32 {
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

    pub(super) fn best_predictor(&self, use_wp: bool) -> u32 {
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
#[cfg(test)]
pub(crate) fn choose_predictor_for_plane(
    get: impl Fn(usize, usize) -> i32,
    w: usize,
    h: usize,
    use_wp: bool,
) -> u32 {
    choose_predictor_for_plane_with_wp(get, w, h, WpParams::DEFAULT, use_wp)
}

pub(crate) fn choose_predictor_for_rows<'a, T: Sample + 'a>(
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

#[cfg(test)]
pub(super) fn choose_predictor_for_plane_with_wp(
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

pub(in crate::lossless) fn choose_predictors_with_wp(
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
        let mut costs = SqueezePredictorCost::default();
        if chan < num_color {
            let pd = linear.plane_data(chan);
            costs.add_rows_shared::<_, false>(
                &|y| &pd[y * xsize..][..xsize],
                xsize,
                ysize,
                use_wp,
                wp_params,
            )
        } else {
            let a = alpha.expect("alpha channel must exist");
            match a {
                AlphaPlane::U8(data) => costs.add_rows_shared::<_, false>(
                    &|y| &data[y * xsize..][..xsize],
                    xsize,
                    ysize,
                    use_wp,
                    wp_params,
                ),
                AlphaPlane::U16 { data, .. } => costs.add_rows_shared::<_, false>(
                    &|y| &data[y * xsize..][..xsize],
                    xsize,
                    ysize,
                    use_wp,
                    wp_params,
                ),
                AlphaPlane::F32(data) => costs.add_rows_shared::<_, false>(
                    &|y| &data[y * xsize..][..xsize],
                    xsize,
                    ysize,
                    use_wp,
                    wp_params,
                ),
            }
        }
        costs.predictor(use_wp)
    });
    preds[..num_color].copy_from_slice(&selected[..num_color]);
    if alpha.is_some() {
        preds[3] = selected[num_color];
    }
    preds
}
