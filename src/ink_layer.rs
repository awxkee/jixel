/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Lift a small palette of strokes and text into one coverage frame. Source
//! statistics select the colors before encoding, and the union is inpainted
//! once so the encoder writes one VarDCT base and one modular ink frame.

use crate::thread_pool::ThreadPool;
use crate::util::round_i32;

/// Distance to the opposite background samples that define a thin ridge.
pub(crate) const FEATURE_RADIUS: usize = 5;
/// Minimum luma/chroma separation from the opposite background samples.
pub(crate) const THIN_DEV: i32 = 24;
/// Max per-channel distance from the ink color for a pixel to join a stroke;
/// lighter antialiasing pixels become partial coverage instead.
const COLOR_TOL: i32 = 20;
/// The same for a local (non-dominant) ink.
const LOCAL_TOL: u8 = 9;
/// Inks per image; the ink frame's palette holds their coverage levels.
const MAX_INKS: usize = 16;
/// The antialiasing band reaches this far from a stroke.
const BAND_RADIUS: usize = 1;
/// A pixel beside the band within this of its surroundings is the fringe
/// of the antialiasing: left out of the fill under the strokes, kept exact.
const FRINGE_TOL: u32 = 24;
/// Strokes smaller than this are left to VarDCT.
const MIN_COMPONENT: usize = 16;
/// The dominant color must hold this share of the thin pixels.
const MIN_DOMINANT_SHARE: f32 = 0.3;
/// The ink must cover this share of the image.
const MIN_CORE_SHARE: f32 = 0.005;
/// Mean squared distance (summed over channels) of the strokes from the ink
/// color above which the lines are painted, textured or soft highlights, not
/// inked: inks with their antialiasing sit at 5-16, soft highlights at 40.
const MAX_INK_SQUARE: f32 = 24.0;
/// Mean squared residual (summed over channels) of the band pixels from the
/// mix of their ink and surroundings above which the antialiasing is not a
/// mix at all (soft highlights at 92) and VarDCT keeps it; inks sit at 0-12,
/// strokes over several fills at 69.
const MAX_BAND_RESIDUAL: f64 = 80.0;
const INK_COLOR_SAMPLES: usize = 4096;
/// Modeled bits saved per image pixel the strongest ink must reach to open
/// the layered path: painted strokes and photo ridges stay below 0.006,
/// inks and UI text above 0.012.
const MIN_GAIN_PER_PIXEL: f64 = 0.012;
/// Further inks only pay for their own palette entry.
const MIN_SECONDARY_GAIN_PER_PIXEL: f64 = 0.004;
/// Band pixels per stroke pixel above which the strokes are thin text
/// (0.9-1.3; cel art and doodles sit at 0.15-0.5).
const MAX_TEXT_BAND_RATIO: f32 = 0.7;
/// Mean stroke component size (pixels) up to which the strokes are glyphs.
const MAX_GLYPH_COMPONENT: usize = 256;
#[cfg(test)]
const DIFFUSION_ITERS: usize = 60;

/// A decomposition chosen before any frame is encoded.
pub(crate) struct InkLayers {
    pub(crate) base: Vec<u8>,
    pub(crate) inks: Vec<[u8; 3]>,
    /// Ink of each pixel, 1-based; 0 outside the strokes.
    pub(crate) ink: Vec<u8>,
    /// Coverage of the ink, 255 on the strokes and partial on the band.
    pub(crate) coverage: Vec<u8>,
    /// The strokes are mostly band or small components: text, which
    /// repeated-glyph patches may code better.
    pub(crate) glyph_text: bool,
}

const UNGROWN: u8 = 1;

#[derive(Clone, Copy)]
struct InkPlan {
    color: [u8; 3],
    local: bool,
    share: f32,
}

/// Analyze source colors and seeds, then construct their shared coverage
/// and filled base without encoding trial frames.
pub(crate) struct InkCandidates<'a> {
    rgb: &'a [[u8; 3]],
    alpha: Option<&'a [u8]>,
    width: usize,
    height: usize,
    thin: Vec<u8>,
    state: Vec<u8>,
    queue: Vec<usize>,
    plans: std::vec::IntoIter<InkPlan>,
}

impl<'a> InkCandidates<'a> {
    /// Scan ridge row bands on the shared pool.
    pub(crate) fn new_fast(
        rgb: &'a [u8],
        alpha: Option<&'a [u8]>,
        width: usize,
        height: usize,
        pool: &ThreadPool,
    ) -> Option<Self> {
        let n = width * height;
        if n < 64 * 64 {
            return None;
        }
        let rgb = rgb.as_chunks::<3>().0;
        let (mut thin, mut thin_count) = thin_ridges(rgb, width, height, pool);
        if let Some(alpha) = alpha {
            for (s, &a) in thin.iter_mut().zip(alpha) {
                if a != 255 {
                    thin_count -= usize::from(*s != 0);
                    *s = 0;
                }
            }
        }
        if thin_count < MIN_COMPONENT {
            return None;
        }
        let mut samples = Vec::new();
        sample_colors(rgb, &thin, thin_count, &mut samples);
        let plans: Vec<_> = ink_modes(&samples)
            .into_iter()
            .filter(|&(_, share)| share >= 0.01)
            .map(|(color, share)| InkPlan {
                color,
                local: true,
                share,
            })
            .collect();
        if plans.is_empty() {
            return None;
        }
        Some(Self {
            rgb,
            alpha,
            width,
            height,
            thin,
            state: vec![0; n],
            queue: Vec::new(),
            plans: plans.into_iter(),
        })
    }

    pub(crate) fn select(mut self, distance: f32) -> Option<InkLayers> {
        let mut ranked: Vec<_> = self.plans.by_ref().collect();
        // Background colors are common throughout the image, whereas useful
        // inks concentrate on ridges. Rank their excess ridge support.
        let stride = self.rgb.len().div_ceil(4096);
        let samples: Vec<_> = (0..self.rgb.len())
            .step_by(stride)
            .filter(|&i| self.alpha.is_none_or(|a| a[i] == 255))
            .collect();
        let sample_count = samples.len().max(1);
        for plan in &mut ranked {
            let global = samples
                .iter()
                .filter(|&&i| {
                    self.rgb[i]
                        .iter()
                        .zip(plan.color)
                        .all(|(&a, b)| a.abs_diff(b) < 12)
                })
                .count();
            plan.share -= global as f32 / sample_count as f32;
        }
        ranked.retain(|plan| plan.share > 0.005);
        ranked.sort_by(|a, b| a.local.cmp(&b.local).then(b.share.total_cmp(&a.share)));
        if let Some(first) = ranked.first_mut() {
            let support = first.share;
            if support >= MIN_DOMINANT_SHARE {
                first.local = false;
            }
            ranked.retain(|p| p.share >= support * 0.2);
        }
        let mut plans: Vec<InkPlan> = Vec::with_capacity(MAX_INKS);
        for plan in ranked {
            if plans.iter().any(|p| {
                p.color
                    .iter()
                    .zip(plan.color)
                    .all(|(&a, b)| a.abs_diff(b) < 20)
            }) {
                continue;
            }
            plans.push(plan);
            if plans.len() == MAX_INKS {
                break;
            }
        }
        if plans.is_empty() {
            return None;
        }
        joint_split(
            self.rgb,
            self.alpha,
            self.width,
            self.height,
            &mut self.thin,
            &mut self.state,
            &mut self.queue,
            &plans,
            distance,
        )
    }
}

/// Classify all selected inks together, then inpaint their union once. The
/// state holds the pixel's ink (1-based) above three growth flags, then the
/// ink alone with the top bit marking the component scan.
fn joint_split(
    rgb: &[[u8; 3]],
    alpha: Option<&[u8]>,
    width: usize,
    height: usize,
    thin: &mut [u8],
    state: &mut [u8],
    queue: &mut Vec<usize>,
    plans: &[InkPlan],
    distance: f32,
) -> Option<InkLayers> {
    let n = rgb.len();
    queue.clear();
    let mut buckets = vec![0u32; 32 * 32 * 32];
    for (k, plan) in plans.iter().enumerate() {
        let radius = if plan.local {
            LOCAL_TOL - 1
        } else {
            COLOR_TOL as u8 - 1
        };
        let lo = plan.color.map(|v| (v.saturating_sub(radius) >> 3) as usize);
        let hi = plan.color.map(|v| (v.saturating_add(radius) >> 3) as usize);
        for r in lo[0]..=hi[0] {
            for g in lo[1]..=hi[1] {
                for b in lo[2]..=hi[2] {
                    buckets[(r << 10) | (g << 5) | b] |= 1u32 << k;
                }
            }
        }
    }
    for (i, (p, s)) in rgb.iter().zip(state.iter_mut()).enumerate() {
        *s = u8::MAX;
        if alpha.is_some_and(|a| a[i] != 255) {
            continue;
        }
        let mut best = (u8::MAX, 0usize);
        let mut choices = buckets
            [((p[0] as usize >> 3) << 10) | ((p[1] as usize >> 3) << 5) | (p[2] as usize >> 3)];
        while choices != 0 {
            let k = choices.trailing_zeros() as usize;
            choices &= choices - 1;
            let plan = &plans[k];
            let d = p
                .iter()
                .zip(plan.color)
                .map(|(&a, b)| a.abs_diff(b))
                .max()
                .unwrap();
            if d < if plan.local {
                LOCAL_TOL
            } else {
                COLOR_TOL as u8
            } && d < best.0
            {
                best = (d, k);
            }
        }
        if best.0 == u8::MAX {
            continue;
        }
        let seed = thin[i] & if plans[best.1].local { 3 } else { 1 } != 0;
        *s = ((best.1 as u8 + 1) << 3) | if seed { 0 } else { UNGROWN };
        if seed {
            queue.push(i);
        }
    }
    // Strokes are 8-connected: a one-pixel diagonal touches only by corner.
    let mut head = 0;
    while let Some(&i) = queue.get(head) {
        head += 1;
        let color = state[i] & !7;
        for j in neighbors8(i, width, height) {
            if state[j] == color | UNGROWN {
                state[j] = color;
                queue.push(j);
            }
        }
    }
    for s in state.iter_mut() {
        *s = if *s & 7 == 0 { *s >> 3 } else { 0 };
    }
    const VISITED: u8 = 0x80;
    let mut counts = vec![0usize; plans.len()];
    let mut components = vec![0usize; plans.len()];
    // Squared error of drawing the strokes in the flat ink color.
    let mut squares = vec![0u64; plans.len()];
    let mut seed = 0;
    while let Some(offset) = state[seed..]
        .iter()
        .position(|&s| (1..=MAX_INKS as u8).contains(&s))
    {
        seed += offset;
        let color = state[seed];
        let k = color as usize - 1;
        queue.clear();
        queue.push(seed);
        state[seed] |= VISITED;
        let mut square = 0u64;
        let mut head = 0;
        while let Some(&i) = queue.get(head) {
            head += 1;
            square += rgb[i]
                .iter()
                .zip(plans[k].color)
                .map(|(&a, b)| (a.abs_diff(b) as u64).pow(2))
                .sum::<u64>();
            for j in neighbors8(i, width, height) {
                if state[j] == color {
                    state[j] |= VISITED;
                    queue.push(j);
                }
            }
        }
        // A stroke is one color: a component that drifts from the ink is a
        // fill or gradient the growth flooded, and stays in the base.
        if queue.len() < if plans[k].local { 4 } else { MIN_COMPONENT }
            || square as f32 > MAX_INK_SQUARE * distance.min(1.0).powi(2) * queue.len() as f32
        {
            for &i in queue.iter() {
                state[i] = 0;
            }
        } else {
            counts[k] += queue.len();
            components[k] += 1;
            squares[k] += square;
        }
        seed += 1;
    }
    for s in state.iter_mut() {
        *s &= !VISITED;
    }
    // An ink whose strokes mostly lie within the band of another ink's
    // strokes is that ink's antialiasing level, not a color of its own.
    let mut touching = vec![0usize; plans.len()];
    for i in 0..n {
        let color = state[i];
        if color == 0 {
            continue;
        }
        let (x, y) = (i % width, i / width);
        if neighborhood(state, width, height, x, y, BAND_RADIUS)
            .any(|row| row.iter().any(|&s| s != 0 && s != color))
        {
            touching[color as usize - 1] += 1;
        }
    }
    for k in 0..plans.len() {
        if touching[k] * 2 > counts[k] {
            for s in state.iter_mut() {
                if *s == k as u8 + 1 {
                    *s = 0;
                }
            }
            counts[k] = 0;
        }
    }
    // Whether the strokes are ink is a property of the image, not of the
    // distance: the edge costs are those of distance 1, and the coverage
    // precision follows the distance later.
    let costs: [f32; 256] = std::array::from_fn(|v| (1.0 + v as f32 / 4.0).log2());
    let mut gain = vec![0f64; plans.len()];
    for (i, &color) in state.iter().enumerate() {
        if color == 0 {
            continue;
        }
        let k = color as usize - 1;
        let x = i % width;
        for j in [
            x.checked_sub(1).map(|_| i - 1),
            (x + 1 < width).then_some(i + 1),
            i.checked_sub(width),
            (i + width < n).then_some(i + width),
        ]
        .into_iter()
        .flatten()
        {
            if state[j] == color {
                continue;
            }
            let edge: f64 = (0..3)
                .map(|c| costs[rgb[i][c].abs_diff(rgb[j][c]) as usize] as f64)
                .sum();
            gain[k] += edge * 0.35 - 2.0;
        }
    }
    // Below distance 1 the flat ink must match the finer quantizer.
    let precision = distance.min(1.0).powi(2);
    let valid = |k: usize| {
        let share = if plans[k].local {
            0.0005
        } else {
            MIN_CORE_SHARE
        };
        (counts[k] as f32) >= share * n as f32
            && squares[k] as f32 <= MAX_INK_SQUARE * precision * counts[k] as f32
    };
    if !(0..plans.len()).any(|k| valid(k) && gain[k] > MIN_GAIN_PER_PIXEL * n as f64) {
        return None;
    }
    let mut slots = vec![usize::MAX; plans.len()];
    let mut inks = Vec::new();
    for (k, plan) in plans.iter().enumerate() {
        if !valid(k) || gain[k] <= MIN_SECONDARY_GAIN_PER_PIXEL * n as f64 {
            continue;
        }
        slots[k] = inks.len();
        inks.push(plan.color);
    }
    if inks.is_empty() {
        return None;
    }
    for s in state.iter_mut() {
        if *s != 0 && slots[*s as usize - 1] == usize::MAX {
            *s = 0;
        }
    }
    let mut ink = vec![0u8; n];
    let mut coverage = vec![0u8; n];
    let mut base: Vec<[f32; 3]> = Vec::new();
    // Residual of the band pixels from the mix of their ink and surroundings,
    // per ink: an ink whose band is no such mix (soft highlights, strokes
    // over several fills) stays in the base, and the band is split again
    // without it.
    let mut residual = vec![0f64; plans.len()];
    let mut band = vec![0usize; plans.len()];
    let mut fringe = Vec::new();
    loop {
        ink.fill(0);
        coverage.fill(0);
        residual.fill(0.0);
        band.fill(0);
        fringe.clear();
        // The antialiasing band: within `BAND_RADIUS` of a stroke. Its pixels
        // mix an ink with the surroundings, which only pixels outside the band
        // show.
        let mut near = vec![false; n];
        for i in 0..n {
            if state[i] != 0 {
                let (x, y) = (i % width, i / width);
                let (x0, y0) = (x.saturating_sub(BAND_RADIUS), y.saturating_sub(BAND_RADIUS));
                for row in near[y0 * width..(y + BAND_RADIUS + 1).min(height) * width]
                    .chunks_exact_mut(width)
                {
                    row[x0..(x + BAND_RADIUS + 1).min(width)].fill(true);
                }
            }
        }
        thin.fill(1);
        base.clear();
        base.extend(rgb.iter().map(|p| p.map(f32::from)));
        for i in 0..n {
            let color = state[i];
            if color != 0 {
                ink[i] = slots[color as usize - 1] as u8 + 1;
                coverage[i] = 255;
                thin[i] = 0;
                continue;
            }
            if !near[i] || alpha.is_some_and(|a| a[i] != 255) {
                continue;
            }
            let (x, y) = (i % width, i / width);
            let (x0, y0) = (
                x.saturating_sub(BAND_RADIUS + 1),
                y.saturating_sub(BAND_RADIUS + 1),
            );
            let mut fill = [0f32; 3];
            let mut count = 0;
            // Inks within the band radius: the pixel mixes one of them.
            let mut adjacent = 0u32;
            for (dy, (pixels, states)) in neighborhood(rgb, width, height, x, y, BAND_RADIUS + 1)
                .zip(neighborhood(state, width, height, x, y, BAND_RADIUS + 1))
                .enumerate()
            {
                for (dx, (p, &s)) in pixels.iter().zip(states).enumerate() {
                    let (xx, yy) = (x0 + dx, y0 + dy);
                    let j = yy * width + xx;
                    if s != 0 {
                        if xx.abs_diff(x) <= BAND_RADIUS && yy.abs_diff(y) <= BAND_RADIUS {
                            adjacent |= 1 << (s - 1);
                        }
                        continue;
                    }
                    if near[j] || alpha.is_some_and(|a| a[j] != 255) {
                        continue;
                    }
                    for c in 0..3 {
                        fill[c] += p[c] as f32;
                    }
                    count += 1;
                }
            }
            if adjacent == 0 || count == 0 {
                continue;
            }
            fill = fill.map(|v| v / count as f32);
            let mut best = (f32::INFINITY, 0usize, 0f32, 0u8);
            for (k, plan) in plans.iter().enumerate() {
                if slots[k] == usize::MAX || adjacent & (1 << k) == 0 {
                    continue;
                }
                let ink = plan.color.map(f32::from);
                let d: [f32; 3] = std::array::from_fn(|c| fill[c] - ink[c]);
                let dd: f32 = d.iter().map(|v| v * v).sum();
                if dd < 100.0 {
                    continue;
                }
                let c = ((0..3)
                    .map(|j| (fill[j] - rgb[i][j] as f32) * d[j])
                    .sum::<f32>()
                    / dd)
                    .clamp(0.0, 1.0);
                let a = round_i32(c * 255.0) as u8;
                if a == 0 {
                    continue;
                }
                let error: f32 = (0..3)
                    .map(|j| {
                        let e = fill[j] - d[j] * c - rgb[i][j] as f32;
                        e * e
                    })
                    .sum();
                if error < best.0 {
                    best = (error, k, c, a);
                }
            }
            if !best.0.is_finite() {
                continue;
            }
            let (_, k, c, a) = best;
            residual[k] += best.0 as f64;
            band[k] += 1;
            ink[i] = slots[k] as u8 + 1;
            coverage[i] = a;
            // Mostly surroundings: the base keeps them with the ink taken out.
            if c < 0.5 {
                for j in 0..3 {
                    base[i][j] = ((rgb[i][j] as f32 - plans[k].color[j] as f32 * c) / (1.0 - c))
                        .clamp(0.0, 255.0);
                }
            } else {
                thin[i] = 0;
            }
        }
        // The base continues its surroundings under the strokes and their band.
        // The faint fringe one pixel further, which the mix does not claim, is
        // taken out of the fill (it would seep under every glyph) and put back
        // exact: it is what the strokes' edges look like.
        for i in 0..n {
            if coverage[i] == 0 {
                continue;
            }
            for j in neighbors8(i, width, height) {
                if coverage[j] != 0 || thin[j] == 0 || alpha.is_some_and(|a| a[j] != 255) {
                    continue;
                }
                let (x, y) = (j % width, j / width);
                let (x0, y0) = (x.saturating_sub(2), y.saturating_sub(2));
                let mut fill = [0i32; 3];
                let mut count = 0;
                for (dy, (pixels, states)) in neighborhood(rgb, width, height, x, y, 2)
                    .zip(neighborhood(state, width, height, x, y, 2))
                    .enumerate()
                {
                    for (dx, (p, &s)) in pixels.iter().zip(states).enumerate() {
                        let m = (y0 + dy) * width + x0 + dx;
                        if s != 0 || near[m] || alpha.is_some_and(|a| a[m] != 255) {
                            continue;
                        }
                        for c in 0..3 {
                            fill[c] += p[c] as i32;
                        }
                        count += 1;
                    }
                }
                // Deep inside the text no surroundings show: it is fringe too.
                if count == 0
                    || (0..3)
                        .all(|c| (rgb[j][c] as i32 - fill[c] / count).unsigned_abs() <= FRINGE_TOL)
                {
                    thin[j] = 0;
                    fringe.push(j);
                }
            }
        }
        let limit = MAX_BAND_RESIDUAL * precision as f64;
        let mut dropped = false;
        for k in 0..plans.len() {
            if slots[k] != usize::MAX && residual[k] > limit * band[k] as f64 {
                for s in state.iter_mut() {
                    if *s == k as u8 + 1 {
                        *s = 0;
                    }
                }
                slots[k] = usize::MAX;
                counts[k] = 0;
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
        inks.clear();
        for k in 0..plans.len() {
            if slots[k] != usize::MAX {
                slots[k] = inks.len();
                inks.push(plans[k].color);
            }
        }
        if inks.is_empty() {
            return None;
        }
    }
    let band: usize = band.iter().sum();
    let kept = || (0..plans.len()).filter(|&k| slots[k] != usize::MAX);
    let core: usize = kept().map(|k| counts[k]).sum();
    let components: usize = kept().map(|k| components[k]).sum();
    let glyph_text = (band as f32) >= MAX_TEXT_BAND_RATIO * core as f32
        || core <= MAX_GLYPH_COMPONENT * components;
    fill_dont_care_with_iters(&mut base, thin, width, height, queue, 8);
    for &i in &fringe {
        base[i] = rgb[i].map(f32::from);
    }
    Some(InkLayers {
        base: base
            .as_flattened()
            .iter()
            .map(|&v| round_i32(v).clamp(0, 255) as u8)
            .collect(),
        inks,
        ink,
        coverage,
        glyph_text,
    })
}

/// The up to eight neighbors of pixel `i`.
fn neighbors8(i: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % width, i / width);
    (y.saturating_sub(1)..(y + 2).min(height)).flat_map(move |yy| {
        (x.saturating_sub(1)..(x + 2).min(width))
            .map(move |xx| yy * width + xx)
            .filter(move |&j| j != i)
    })
}

/// Rows of a clipped square neighborhood, with bounds checked per row
/// instead of per channel of every pixel.
fn neighborhood<T>(
    pixels: &[T],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
) -> impl Iterator<Item = &[T]> {
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(height);
    pixels[y0 * width..y1 * width]
        .chunks_exact(width)
        .map(move |row| &row[x0..x1])
}

/// Opposite samples detect thin luma ridges in linear time. At a clipped
/// edge use the available background sample on both sides.
fn thin_ridges(
    rgb: &[[u8; 3]],
    width: usize,
    height: usize,
    pool: &ThreadPool,
) -> (Vec<u8>, usize) {
    let luma_row = selected_luma_row_fn();
    let ridges = selected_ridge_row_fn();
    let mut luma = vec![0; rgb.len()];
    luma_row(rgb, &mut luma);
    let mut thin = vec![0; rgb.len()];
    let ridge_row = |y: usize, row: &mut [u8]| ridges(rgb, &luma, width, height, y, row);
    let band = |y_begin: usize, rows: &mut [u8]| {
        let mut count = 0;
        for (dy, row) in rows.chunks_exact_mut(width).enumerate() {
            count += ridge_row(y_begin + dy, row);
        }
        count
    };
    let rows_per_band = height.div_ceil(pool.num_threads());
    let mut bands: Vec<_> = thin
        .chunks_mut(rows_per_band * width)
        .map(|rows| (rows, 0))
        .collect();
    pool.for_each_mut(&mut bands, |i, (rows, count)| {
        *count = band(i * rows_per_band, rows);
    });
    let count = bands.into_iter().map(|(_, count)| count).sum();
    (thin, count)
}

fn sample_colors(rgb: &[[u8; 3]], thin: &[u8], thin_count: usize, samples: &mut Vec<[f32; 3]>) {
    let stride = thin_count.div_ceil(INK_COLOR_SAMPLES).max(1);
    samples.clear();
    samples.reserve(thin_count.div_ceil(stride));
    samples.extend(
        rgb.iter()
            .zip(thin)
            .filter(|&(_, &t)| t != 0)
            .step_by(stride)
            .map(|(p, _)| p.map(f32::from)),
    );
}

/// Fixed color cells preserve actual foreground modes instead of averaging
/// a small, flat ink with a broad background/antialiasing cluster.
fn ink_modes(samples: &[[f32; 3]]) -> Vec<([u8; 3], f32)> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut counts = [0u32; 4096];
    let mut sums = [[0u32; 3]; 4096];
    for p in samples {
        let p = p.map(|v| v as u8);
        let key = ((p[0] as usize >> 4) << 8) | ((p[1] as usize >> 4) << 4) | (p[2] as usize >> 4);
        counts[key] += 1;
        for c in 0..3 {
            sums[key][c] += p[c] as u32;
        }
    }
    let mut order: Vec<_> = (0..4096).filter(|&k| counts[k] != 0).collect();
    order.sort_unstable_by_key(|&k| (std::cmp::Reverse(counts[k]), k));
    order
        .into_iter()
        .take(24)
        .map(|k| {
            (
                sums[k].map(|v| ((v + counts[k] / 2) / counts[k]) as u8),
                counts[k] as f32 / samples.len() as f32,
            )
        })
        .collect()
}

/// Replace the free pixels of `base` by a smooth continuation of their
/// surroundings: a boundary-inward fill followed by box diffusion.
/// `filled` starts at zero for free pixels and one for fixed pixels.
#[cfg(test)]
fn fill_dont_care(
    base: &mut [[f32; 3]],
    filled: &mut [u8],
    width: usize,
    height: usize,
    dont_care: &mut Vec<usize>,
) {
    fill_dont_care_with_iters(base, filled, width, height, dont_care, DIFFUSION_ITERS);
}

fn fill_dont_care_with_iters(
    base: &mut [[f32; 3]],
    filled: &mut [u8],
    width: usize,
    height: usize,
    dont_care: &mut Vec<usize>,
    iterations: usize,
) {
    dont_care.clear();
    dont_care.extend(
        filled
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| (s == 0).then_some(i)),
    );
    if dont_care.is_empty() {
        return;
    }
    let neighbors = |i: usize| {
        let (x, y) = (i % width, i / width);
        (y.saturating_sub(1)..(y + 2).min(height)).flat_map(move |yy| {
            (x.saturating_sub(1)..(x + 2).min(width)).map(move |xx| yy * width + xx)
        })
    };
    let mut frontier: Vec<usize> = dont_care
        .iter()
        .copied()
        .filter(|&i| neighbors(i).any(|j| filled[j] == 1))
        .collect();
    // Inward fill in rings: each ring takes the mean of its filled neighbors.
    // State 2 claims a pixel for the next ring without making its value visible.
    let mut next = Vec::new();
    while !frontier.is_empty() {
        for &i in &frontier {
            let mut sum = [0f32; 3];
            let mut count = 0usize;
            let (x, y) = (i % width, i / width);
            for (pixels, states) in neighborhood(base, width, height, x, y, 1)
                .zip(neighborhood(filled, width, height, x, y, 1))
            {
                for (p, &s) in pixels.iter().zip(states) {
                    if s == 1 {
                        for (sum, &v) in sum.iter_mut().zip(p) {
                            *sum += v;
                        }
                        count += 1;
                    }
                }
            }
            // The current ring is still unfilled, so its new values cannot
            // be read by another pixel until the whole ring is committed.
            base[i] = sum.map(|v| v / count.max(1) as f32);
        }
        for &i in &frontier {
            filled[i] = 1;
        }
        next.clear();
        for &i in &frontier {
            for j in neighbors(i) {
                if filled[j] == 0 {
                    filled[j] = 2;
                    next.push(j);
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next);
    }
    drop(frontier);
    drop(next);
    // Delay each row's writes until the following active row has been read.
    // A 3x3 stencil cannot see an earlier row after that point. This preserves
    // Jacobi's evaluation order with two rows of values instead of a second
    // float buffer covering every free pixel.
    let mut rows = Vec::new();
    let mut start = 0;
    let mut max_row = 0;
    while start < dont_care.len() {
        let y = dont_care[start] / width;
        let end = start + dont_care[start..].partition_point(|&i| i / width == y);
        rows.push((y, start, end));
        max_row = max_row.max(end - start);
        start = end;
    }
    let mut previous = vec![[0f32; 3]; max_row];
    let mut current = vec![[0f32; 3]; max_row];
    for _ in 0..iterations {
        let mut pending = &dont_care[..0];
        for &(y, start, end) in &rows {
            let row_start = y * width;
            let row_end = row_start + width;
            let indices = &dont_care[start..end];
            for (slot, &i) in current.iter_mut().zip(indices) {
                let mut sum = [0f32; 3];
                if y > 0 && y + 1 < height && i > row_start && i + 1 < row_end {
                    // The interior has a fixed stencil, without coordinate
                    // division or clipped-window setup in every iteration.
                    for row in [
                        &base[i - width - 1..][..3],
                        &base[i - 1..][..3],
                        &base[i + width - 1..][..3],
                    ] {
                        for p in row {
                            for (sum, &v) in sum.iter_mut().zip(p) {
                                *sum += v;
                            }
                        }
                    }
                    *slot = sum.map(|v| v / 9.0);
                } else {
                    let mut count = 0;
                    for row in neighborhood(base, width, height, i - row_start, y, 1) {
                        for p in row {
                            for (sum, &v) in sum.iter_mut().zip(p) {
                                *sum += v;
                            }
                        }
                        count += row.len();
                    }
                    *slot = sum.map(|v| v / count as f32);
                }
            }
            for (&i, &v) in pending.iter().zip(&previous) {
                base[i] = v;
            }
            std::mem::swap(&mut previous, &mut current);
            pending = indices;
        }
        for (&i, &v) in pending.iter().zip(&previous) {
            base[i] = v;
        }
    }
}

pub(crate) fn ridge_positions(x: usize, y: usize, w: usize, h: usize) -> [usize; 5] {
    const R: usize = crate::ink_layer::FEATURE_RADIUS;
    let left = x.saturating_sub(R);
    let right = (x + R).min(w - 1);
    let above = y.saturating_sub(R);
    let below = (y + R).min(h - 1);
    [
        y * w + x,
        y * w + if x < R { right } else { left },
        y * w + if x + R >= w { left } else { right },
        (if y < R { below } else { above }) * w + x,
        (if y + R >= h { above } else { below }) * w + x,
    ]
}

pub(crate) fn ridge_pixel(rgb: &[[u8; 3]], luma: &[u8], positions: [usize; 5]) -> u8 {
    let [p, a, b, c, d] = positions;
    let ridge = |a, b| {
        let da = luma[p] as i32 - luma[a] as i32;
        let db = luma[p] as i32 - luma[b] as i32;
        (da > THIN_DEV && db > THIN_DEV) || (da < -THIN_DEV && db < -THIN_DEV)
    };
    let chroma = |a: usize| {
        let dg = rgb[p][1] as i32 - rgb[a][1] as i32;
        (rgb[p][0] as i32 - rgb[a][0] as i32 - dg)
            .abs()
            .max((rgb[p][2] as i32 - rgb[a][2] as i32 - dg).abs())
    };
    let color = |a, b| {
        chroma(a) > THIN_DEV
            && chroma(b) > THIN_DEV
            && (0..3).all(|c| rgb[a][c].abs_diff(rgb[b][c]) < 16)
    };
    u8::from(ridge(a, b) || ridge(c, d) || color(a, b) || color(c, d))
}

pub(crate) type LumaRowFn = fn(&[[u8; 3]], &mut [u8]);

#[allow(unreachable_code)]
pub(crate) fn selected_luma_row_fn() -> LumaRowFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |a, b| unsafe { crate::avx::luma_row_avx2(a, b) };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        return |a, b| unsafe { crate::sse::luma_row_sse41(a, b) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |a, b| unsafe { crate::neon::luma_row_neon(a, b) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::luma_row_wasm;
    luma_row_scalar
}

#[allow(dead_code)]
pub(crate) fn luma_row_scalar(rgb: &[[u8; 3]], out: &mut [u8]) {
    for (&[r, g, b], out) in rgb.iter().zip(out) {
        *out = ((77 * r as u32 + 150 * g as u32 + 29 * b as u32 + 128) >> 8) as u8;
    }
}

pub(crate) type RidgeRowFn = fn(&[[u8; 3]], &[u8], usize, usize, usize, &mut [u8]) -> usize;

#[allow(unreachable_code)]
pub(crate) fn selected_ridge_row_fn() -> RidgeRowFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |a, b, c, d, e, f| unsafe { crate::avx::ridge_row_avx2(a, b, c, d, e, f) };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        return |a, b, c, d, e, f| unsafe { crate::sse::ridge_row_sse41(a, b, c, d, e, f) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |a, b, c, d, e, f| unsafe { crate::neon::ridge_row_neon(a, b, c, d, e, f) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::ridge_row_wasm;
    ridge_row_scalar
}

#[allow(dead_code)]
pub(crate) fn ridge_row_scalar(
    rgb: &[[u8; 3]],
    luma: &[u8],
    w: usize,
    h: usize,
    y: usize,
    out: &mut [u8],
) -> usize {
    let mut count = 0;
    for x in (y & 1..w).step_by(2) {
        out[x] = ridge_pixel(rgb, luma, ridge_positions(x, y, w, h));
        count += usize::from(out[x]);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_joint_palette_covers_border_strokes_and_preserves_nonopaque_pixels() {
        let (width, height) = (160, 96);
        let background = [240, 235, 230];
        let inks = [[20, 20, 20], [160, 20, 40]];
        let rgb: Vec<_> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    if (8..88).contains(&y) && !(2..158).contains(&x) {
                        inks[0]
                    } else if (8..88).contains(&y) && (77..79).contains(&x) {
                        inks[1]
                    } else {
                        background
                    }
                })
            })
            .collect();
        let mut alpha = vec![255; width * height];
        for (x, y, value) in [(0, 12, 0), (1, 24, 64), (77, 36, 128), (159, 48, 254)] {
            alpha[y * width + x] = value;
        }
        let plan = InkCandidates::new_fast(&rgb, Some(&alpha), width, height, &ThreadPool::new(1))
            .unwrap()
            .select(1.5)
            .unwrap();
        assert!(plan.inks.len() <= 4);
        for (x, y, ink) in [(0, 60, inks[0]), (159, 60, inks[0]), (77, 60, inks[1])] {
            let i = y * width + x;
            assert_eq!(plan.inks[plan.ink[i] as usize - 1], ink);
            assert_eq!(plan.coverage[i], 255);
        }
        for (i, &a) in alpha.iter().enumerate() {
            if a != 255 {
                assert_eq!(plan.ink[i], 0);
                assert_eq!(&plan.base[i * 3..][..3], &rgb[i * 3..][..3]);
            }
        }
        assert!(plan.ink.iter().all(|&i| (i as usize) <= plan.inks.len()));
    }

    #[test]
    fn fast_joint_palette_retains_equal_luma_chromatic_strokes() {
        let (width, height) = (128, 64);
        let rgb: Vec<_> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    if (8..56).contains(&y) && (32..34).contains(&x) {
                        [255, 0, 0]
                    } else {
                        [0, 131, 0]
                    }
                })
            })
            .collect();
        let plan = InkCandidates::new_fast(&rgb, None, width, height, &ThreadPool::new(1))
            .unwrap()
            .select(1.5)
            .unwrap();
        assert_eq!(
            plan.inks[plan.ink[16 * width + 32] as usize - 1],
            [255, 0, 0]
        );
        assert_eq!(plan.coverage[16 * width + 32], 255);
        assert_eq!(&plan.base[(16 * width + 32) * 3..][..3], &[0, 131, 0]);
    }

    #[test]
    fn fast_analysis_rejects_flat_and_nonopaque_sources_and_handles_narrow_images() {
        for (width, height) in [(1, 4096), (2, 2048), (8, 512), (64, 64)] {
            assert!(
                InkCandidates::new_fast(
                    &vec![200; width * height * 3],
                    None,
                    width,
                    height,
                    &ThreadPool::new(1)
                )
                .is_none()
            );
        }
        let (width, height) = (128, 64);
        let rgb: Vec<_> = (0..width * height)
            .flat_map(|i| [if i % width < 3 { 0 } else { 220 }; 3])
            .collect();
        assert!(
            InkCandidates::new_fast(
                &rgb,
                Some(&vec![127; width * height]),
                width,
                height,
                &ThreadPool::new(1)
            )
            .is_none()
        );
    }

    #[test]
    fn inward_fill_reaches_free_pixels_and_preserves_fixed_pixels() {
        for (width, height) in [(1, 17), (17, 1), (9, 7)] {
            let color = [17.0, 83.0, 201.0];
            let mut base = vec![[0.0; 3]; width * height];
            base[0] = color;
            let mut filled = vec![0; base.len()];
            filled[0] = 1;
            let mut work = Vec::new();
            fill_dont_care(&mut base, &mut filled, width, height, &mut work);
            assert!(base.iter().all(|&p| p == color));

            // The empty free set and a free set without a boundary are valid.
            filled.fill(1);
            fill_dont_care(&mut base, &mut filled, width, height, &mut work);
            assert!(work.is_empty());
            filled.fill(0);
            fill_dont_care(&mut base, &mut filled, width, height, &mut work);
            assert!(base.iter().all(|&p| p == color));
        }
    }

    #[test]
    fn fill_matches_synchronous_reference_for_sparse_and_clipped_rows() {
        // Deliberately use whole-image snapshots in the reference so that
        // writes from either a fill ring or a diffusion row cannot leak into
        // another pixel's input. Include gaps between active rows and edges.
        for (width, height) in [(1, 1), (1, 17), (17, 1), (2, 7), (7, 2), (17, 13), (19, 21)] {
            for style in 0..7 {
                let mut random = 7u64;
                let free: Vec<_> = (0..width * height)
                    .map(|i| {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        match style {
                            0 => true,
                            1 => false,
                            2 => i != 0,
                            3 => random as u8 > 127,
                            4 => random as u8 > 240,
                            5 => i / width % 3 == 0,
                            _ => i % width == 0 || i % width + 1 == width,
                        }
                    })
                    .collect();
                let mut actual: Vec<[f32; 3]> = (0..width * height)
                    .map(|i| std::array::from_fn(|c| ((i * 73 + c * 31) % 256) as f32 * 0.37))
                    .collect();
                let mut expected = actual.clone();
                let mut filled: Vec<_> = free.iter().map(|&v| u8::from(!v)).collect();
                let mut known: Vec<_> = free.iter().map(|&v| !v).collect();
                loop {
                    let mut ring = Vec::new();
                    for i in 0..expected.len() {
                        if known[i] {
                            continue;
                        }
                        let (x, y) = (i % width, i / width);
                        let mut sum = [0f32; 3];
                        let mut count = 0;
                        for yy in y.saturating_sub(1)..(y + 2).min(height) {
                            for xx in x.saturating_sub(1)..(x + 2).min(width) {
                                let j = yy * width + xx;
                                if known[j] {
                                    for c in 0..3 {
                                        sum[c] += expected[j][c];
                                    }
                                    count += 1;
                                }
                            }
                        }
                        if count != 0 {
                            ring.push((i, sum.map(|v| v / count as f32)));
                        }
                    }
                    if ring.is_empty() {
                        break;
                    }
                    for (i, value) in ring {
                        expected[i] = value;
                        known[i] = true;
                    }
                }
                for _ in 0..DIFFUSION_ITERS {
                    let before = expected.clone();
                    for (i, &free) in free.iter().enumerate() {
                        if !free {
                            continue;
                        }
                        let (x, y) = (i % width, i / width);
                        let mut sum = [0f32; 3];
                        let mut count = 0;
                        for yy in y.saturating_sub(1)..(y + 2).min(height) {
                            for xx in x.saturating_sub(1)..(x + 2).min(width) {
                                for c in 0..3 {
                                    sum[c] += before[yy * width + xx][c];
                                }
                                count += 1;
                            }
                        }
                        expected[i] = sum.map(|v| v / count as f32);
                    }
                }
                fill_dont_care(&mut actual, &mut filled, width, height, &mut Vec::new());
                assert_eq!(
                    actual
                        .iter()
                        .map(|p| p.map(f32::to_bits))
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|p| p.map(f32::to_bits))
                        .collect::<Vec<_>>(),
                    "{width}x{height}, style {style}"
                );
            }
        }
    }

    #[test]
    fn line_art_plan_is_deterministic_and_preserves_uncovered_pixels() {
        let (width, height) = (79, 83);
        let rgb: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let background = [180 + (x % 61) as u8, 160 + (y % 81) as u8, 210];
                    let ink = [12, 18, 24];
                    let d = (x + y / 3) % 31;
                    if d < 3 || y % 47 < 2 || (x % 101 < 5 && y % 113 < 5) {
                        ink
                    } else if d == 3 || y % 47 == 2 {
                        std::array::from_fn(|c| ((background[c] as u16 + ink[c] as u16) / 2) as u8)
                    } else {
                        background
                    }
                })
            })
            .collect();
        let a = InkCandidates::new_fast(&rgb, None, width, height, &ThreadPool::new(1))
            .unwrap()
            .select(1.5)
            .unwrap();
        let b = InkCandidates::new_fast(&rgb, None, width, height, &ThreadPool::new(4))
            .unwrap()
            .select(1.5)
            .unwrap();
        assert_eq!(a.base, b.base);
        assert_eq!(a.inks, b.inks);
        assert_eq!(a.ink, b.ink);
        assert_eq!(a.coverage, b.coverage);
        assert!(a.inks.contains(&[12, 18, 24]));
        for (i, &index) in a.ink.iter().enumerate() {
            if index == 0 {
                assert_eq!(&a.base[i * 3..][..3], &rgb[i * 3..][..3]);
            }
        }
    }

    #[test]
    fn periodic_colored_glyphs_do_not_collapse_into_one_cluster() {
        let colors = [
            [0, 0, 0],
            [150, 0, 0],
            [0, 90, 0],
            [0, 0, 150],
            [110, 70, 0],
            [0, 80, 140],
            [100, 0, 100],
            [90, 90, 90],
        ];
        let (width, height) = (256, 64);
        let rgb: Vec<_> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    if (8..56).contains(&y) {
                        colors
                            .iter()
                            .enumerate()
                            .find(|(k, _)| (16 + 30 * k..18 + 30 * k).contains(&x))
                            .map_or([255; 3], |(_, &c)| c)
                    } else {
                        [255; 3]
                    }
                })
            })
            .collect();
        let candidates =
            InkCandidates::new_fast(&rgb, None, width, height, &ThreadPool::new(1)).unwrap();
        let found: Vec<_> = candidates.plans.map(|p| p.color).collect();
        for color in colors {
            assert!(found.contains(&color), "missing {color:?}");
        }
    }

    #[test]
    fn semi_transparent_pixels_never_enter_the_ink_frame() {
        let (width, height) = (128, 64);
        let rgb: Vec<_> = (0..height)
            .flat_map(|_y| {
                (0..width).flat_map(move |x| [if (30..34).contains(&x) { 0 } else { 220 }; 3])
            })
            .collect();
        let mut alpha = vec![255; width * height];
        let excluded = [(30, 0, 0), (31, 10, 64), (32, 20, 128), (33, 63, 254)];
        for (x, y, a) in excluded {
            alpha[y * width + x] = a;
        }
        let plan = InkCandidates::new_fast(&rgb, Some(&alpha), width, height, &ThreadPool::new(1))
            .unwrap()
            .select(1.5)
            .unwrap();
        for (x, y, _) in excluded {
            let i = y * width + x;
            assert_eq!(plan.ink[i], 0);
            assert_eq!(&plan.base[i * 3..][..3], &rgb[i * 3..][..3]);
        }
    }

    #[test]
    fn split_rejects_small_images_flat_images_and_specks() {
        assert!(
            InkCandidates::new_fast(&vec![200; 32 * 32 * 3], None, 32, 32, &ThreadPool::new(1))
                .is_none()
        );
        assert!(
            InkCandidates::new_fast(&vec![200; 64 * 64 * 3], None, 64, 64, &ThreadPool::new(1))
                .is_none()
        );
        let rgb: Vec<u8> = (0..64)
            .flat_map(|y| {
                (0..64).flat_map(move |x| [if x % 8 == 0 && y % 8 == 0 { 0 } else { 200 }; 3])
            })
            .collect();
        assert!(
            InkCandidates::new_fast(&rgb, None, 64, 64, &ThreadPool::new(1))
                .and_then(|c| c.select(1.5))
                .is_none()
        );
    }
}

#[cfg(test)]
mod search_row_tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Rows {
        luma: LumaRowFn,
        ridges: RidgeRowFn,
    }

    fn backends() -> Vec<Rows> {
        #[allow(unused_mut)]
        let mut rows = vec![Rows {
            luma: luma_row_scalar,
            ridges: ridge_row_scalar,
        }];
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            rows.push(Rows {
                luma: |a, b| unsafe { crate::avx::luma_row_avx2(a, b) },
                ridges: |a, b, c, d, e, f| unsafe { crate::avx::ridge_row_avx2(a, b, c, d, e, f) },
            });
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            rows.push(Rows {
                luma: |a, b| unsafe { crate::sse::luma_row_sse41(a, b) },
                ridges: |a, b, c, d, e, f| unsafe { crate::sse::ridge_row_sse41(a, b, c, d, e, f) },
            });
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            rows.push(Rows {
                luma: |a, b| unsafe { crate::neon::luma_row_neon(a, b) },
                ridges: |a, b, c, d, e, f| unsafe { crate::neon::ridge_row_neon(a, b, c, d, e, f) },
            });
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            rows.push(Rows {
                luma: crate::wasm::luma_row_wasm,
                ridges: crate::wasm::ridge_row_wasm,
            });
        }
        rows.push(Rows {
            luma: selected_luma_row_fn(),
            ridges: selected_ridge_row_fn(),
        });
        rows
    }

    #[test]
    fn luma_and_clipped_ridges_match_scalar() {
        for w in 1..35 {
            for h in [1, 2, 5, 11, 19] {
                for pattern in 0..4 {
                    let rgb: Vec<[u8; 3]> = (0..w * h)
                        .map(|i| {
                            std::array::from_fn(|c| match pattern {
                                0 => 127,
                                1 => {
                                    if (i + i / w) % 7 == 0 {
                                        [255, 0, 255][c]
                                    } else {
                                        [0, 127, 0][c]
                                    }
                                }
                                _ => ((i * 37 + c * 53 + pattern * 83) % 256) as u8,
                            })
                        })
                        .collect();
                    let mut expected_luma = vec![0; w * h];
                    luma_row_scalar(&rgb, &mut expected_luma);
                    for backend in backends() {
                        let mut actual_luma = vec![0; w * h];
                        (backend.luma)(&rgb, &mut actual_luma);
                        assert_eq!(actual_luma, expected_luma);
                        for y in 0..h {
                            let mut expected = vec![0xa5; w];
                            let count =
                                ridge_row_scalar(&rgb, &expected_luma, w, h, y, &mut expected);
                            let mut actual = vec![0xa5; w];
                            assert_eq!(
                                (backend.ridges)(&rgb, &expected_luma, w, h, y, &mut actual),
                                count
                            );
                            assert_eq!(actual, expected, "w={w}, h={h}, y={y}, pattern={pattern}");
                        }
                    }
                }
            }
        }
    }
}
