/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::adaptive_quant::{dirty_log1pf, fast_exp2};
use crate::dct::fmla;
use crate::image::{Image3F, ImageB};

pub(crate) type ApplyQuantFieldGainFn = fn(&mut ImageB, usize, usize, usize, usize, f32);
pub(crate) type DarkStructureStatsFn = fn(&[[f32; 64]], usize, usize) -> (f32, f32);
pub(crate) type FillBlueTileFn = fn(&Image3F, &mut [f32], usize, usize, usize, usize) -> f32;

pub(crate) const BLUE_OFFSET: f32 = 0.003_199_477;
pub(crate) const BLUE_FULL: f32 = 0.010_474_085;
pub(crate) const INV_BLUE_FULL: f32 = 1.0 / BLUE_FULL;
pub(crate) const Y_TO_LUMA8: f32 = 300.0;

#[allow(dead_code)]
pub(crate) fn fill_blue_tile_scalar(
    opsin: &Image3F,
    tile: &mut [f32],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> f32 {
    assert!(w <= 64 && tile.len() >= 64 * h);
    if w == 0 || h == 0 {
        return 0.0;
    }
    let mut blue_sum = 0.0f32;
    for (r, dst) in tile.as_chunks_mut::<64>().0.iter_mut().take(h).enumerate() {
        let xr = &opsin.plane_row(0, y0 + r)[x0..x0 + w];
        let yr = &opsin.plane_row(1, y0 + r)[x0..x0 + w];
        let br = &opsin.plane_row(2, y0 + r)[x0..x0 + w];
        for (((d, &x), &y), &b) in dst[..w].iter_mut().zip(xr).zip(yr).zip(br) {
            let by = b - y;
            let excess = (by - x.abs() - BLUE_OFFSET).max(0.0);
            blue_sum += (excess * INV_BLUE_FULL).min(1.0);
            *d = by * Y_TO_LUMA8;
        }
    }
    blue_sum / (w * h) as f32
}

#[allow(dead_code)]
pub(crate) fn apply_quant_field_gain_scalar(
    image: &mut ImageB,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    gain: f32,
) {
    for y in y0..y0 + height {
        for value in &mut image.row_mut(y)[x0..x0 + width] {
            let q = *value as f32 * gain;
            *value = q.round().clamp(1.0, 255.0) as u8;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_apply_quant_field_gain_fn() -> ApplyQuantFieldGainFn {
    crate::wasm::apply_quant_field_gain_wasm
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_dark_structure_stats_fn() -> DarkStructureStatsFn {
    crate::wasm::dark_structure_stats_wasm
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
pub(crate) fn select_fill_blue_tile_fn() -> FillBlueTileFn {
    crate::wasm::fill_blue_tile_wasm
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_fill_blue_tile_fn() -> FillBlueTileFn {
    |opsin, tile, x0, y0, w, h| unsafe {
        crate::neon::fill_blue_tile_neon(opsin, tile, x0, y0, w, h)
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_fill_blue_tile_fn() -> FillBlueTileFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |opsin, tile, x0, y0, w, h| unsafe {
            crate::avx::fill_blue_tile_avx2(opsin, tile, x0, y0, w, h)
        };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |opsin, tile, x0, y0, w, h| unsafe {
            crate::sse::fill_blue_tile_sse41(opsin, tile, x0, y0, w, h)
        };
    }
    fill_blue_tile_scalar
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_dark_structure_stats_fn() -> DarkStructureStatsFn {
    |buf, h, w| unsafe { crate::neon::dark_structure_stats_neon(buf, h, w) }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_dark_structure_stats_fn() -> DarkStructureStatsFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |buf, h, w| unsafe { crate::avx::dark_structure_stats_avx2(buf, h, w) };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |buf, h, w| unsafe { crate::sse::dark_structure_stats_sse41(buf, h, w) };
    }
    dark_structure_stats_scalar
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) fn select_apply_quant_field_gain_fn() -> ApplyQuantFieldGainFn {
    |image, x0, y0, width, height, gain| unsafe {
        crate::neon::apply_quant_field_gain_neon(image, x0, y0, width, height, gain)
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"),
    all(target_arch = "aarch64", feature = "neon")
)))]
pub(crate) fn select_apply_quant_field_gain_fn() -> ApplyQuantFieldGainFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |image, x0, y0, width, height, gain| unsafe {
            crate::avx::apply_quant_field_gain_avx2(image, x0, y0, width, height, gain)
        };
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |image, x0, y0, width, height, gain| unsafe {
            crate::sse::apply_quant_field_gain_sse41(image, x0, y0, width, height, gain)
        };
    }

    apply_quant_field_gain_scalar
}

#[cfg(test)]
pub(crate) trait AqLuma: Copy {
    fn to_f32(self) -> f32;
}
#[cfg(test)]
impl AqLuma for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}
#[cfg(test)]
impl AqLuma for i32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// The representative variance of a superblock for Variance Boost: the value at the
/// requested `octile` (1..=8) of the 64 sorted 8x8 variances. Octile 1 = the most
/// low-variance-biased pick (boost readily), octile 8 = only the maximum (boost only
/// when the whole SB is low-variance). Octile 6 (index 47) is the SVT-AV1-PSY default.
pub(crate) fn sb_octile_variance(subvars: &mut [f32; 64], octile: u8) -> f32 {
    // Octile o in 1..=8 maps to sorted index o*8 - 1 (o=1 -> 7, o=4 -> 31 (median-ish),
    // o=6 -> 47 (SVT-AV1-PSY default), o=8 -> 63 (max)).
    let o = octile.clamp(1, 8) as usize;
    let idx = (o * 8 - 1).min(63);
    *subvars
        .select_nth_unstable_by(idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .1
}

/// Variance Boost qindex delta for one superblock. `picked_var` is the octile pick
/// (see [`sb_octile_variance`]); `ref_log` is the tile mean log-variance.
pub(crate) fn variance_boost_delta(
    picked_var: f32,
    ref_log: f32,
    strength: f32,
    boost_only: bool,
) -> i32 {
    // Work in log-variance: compresses the huge dynamic range of variance and matches
    // the reference (which is a mean of log-variances).
    let v_log = dirty_log1pf(picked_var);
    // Low-variance threshold (curve 0): ln(1 + 256).
    const LOW_LOG: f32 = 5.549_076; // (1.0 + 256.0).ln()
    const MAX_BOOST: f32 = 18.0; // max qindex *reduction* for the flattest SBs
    const MAX_CUT: f32 = 10.0; // max qindex *increase* for the busiest SBs
    // qindex per unit log-variance for each side.
    const BOOST_SLOPE: f32 = 5.0;
    const CUT_SLOPE: f32 = 3.0;

    if v_log < LOW_LOG {
        // Low contrast: boost (negative delta). Deeper below threshold => stronger.
        let d = ((LOW_LOG - v_log) * BOOST_SLOPE * strength).min(MAX_BOOST);
        -(d.round() as i32)
    } else if boost_only {
        0
    } else {
        // Higher contrast: coarsen relative to the tile reference, capped. Using the
        // reference (not the threshold) keeps well-textured frames near zero-mean.
        let over = (v_log - ref_log.max(LOW_LOG)).max(0.0);
        let d = (over * CUT_SLOPE * strength).min(MAX_CUT);
        d.round() as i32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DarkAq {
    pub enabled: bool,
    /// Only active at base_q >= this (qstep-extinction range, roughly q<=45 in 8-bit).
    pub min_q: i32,
    pub mean_floor: f32,
    pub dark_ref: f32,
    pub gamma: f32,
    pub max_weight: f32,
    /// qindex units per unit of `log1p(mid_energy * dark_weight)`.
    pub scale: f32,
    /// Cap on the extra boost (qindex reduction).
    pub max_qidx: i32,
}

impl Default for DarkAq {
    fn default() -> Self {
        DarkAq {
            enabled: false,
            min_q: 150,
            mean_floor: 16.0,
            dark_ref: 56.0,
            gamma: 1.2,
            // `darkness = max_weight - 1` is the effective multiplier for the darkest SBs.
            max_weight: 4.5,
            scale: 4.0,
            max_qidx: 16,
        }
    }
}

impl DarkAq {
    /// Enabled with the calibrated defaults.
    pub(crate) fn on() -> Self {
        DarkAq {
            enabled: true,
            ..DarkAq::default()
        }
    }
}

#[inline]
#[allow(dead_code)]
fn laplacian_abs_sum_scalar(buf: &[f32], stride: usize, h: usize, w: usize) -> f32 {
    let mut sum = 0.0f32;
    for y in 1..h - 1 {
        let top = &buf[(y - 1) * stride..][..w];
        let middle = &buf[y * stride..][..w];
        let bottom = &buf[(y + 1) * stride..][..w];
        for x in 1..w - 1 {
            let up = top[x];
            let down = bottom[x];
            let [left, center, right] = middle[x - 1..=x + 1] else {
                unreachable!()
            };
            let l = 4.0 * center - up - down - left - right;
            sum += l.abs();
        }
    }
    sum
}

#[inline]
#[allow(dead_code)]
fn box_downsample_2x_scalar(
    src: &[f32],
    src_stride: usize,
    h: usize,
    w: usize,
    dst: &mut [f32],
    dst_stride: usize,
) -> (usize, usize) {
    let (hh, ww) = (h / 2, w / 2);
    for y in 0..hh {
        let top = &src[(2 * y) * src_stride..][..w];
        let bottom = &src[(2 * y + 1) * src_stride..][..w];
        let dst_row = &mut dst[y * dst_stride..][..ww];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let sx = 2 * x;
            *out = 0.25 * (top[sx] + top[sx + 1] + bottom[sx] + bottom[sx + 1]);
        }
    }
    (hh, ww)
}

#[allow(dead_code)]
pub(crate) fn dark_structure_stats_scalar(buf: &[[f32; 64]], h: usize, w: usize) -> (f32, f32) {
    assert!(h <= 64 && w <= 64 && buf.len() >= h);
    if h == 0 || w == 0 {
        return (0.0, 0.0);
    }
    let flat = buf.as_flattened();
    let mut sum = 0.0f32;
    for row in flat.as_chunks::<64>().0.iter().take(h) {
        for &v in &row[..w] {
            sum += v;
        }
    }
    let mean = sum / (h * w) as f32;
    if h < 3 || w < 3 {
        return (mean, 0.0);
    }
    let nf = (h - 2) * (w - 2);
    let lap_full = laplacian_abs_sum_scalar(flat, 64, h, w) / nf as f32;
    let mut half = [[0f32; 32]; 32];
    let (hh, ww) = box_downsample_2x_scalar(flat, 64, h, w, half.as_flattened_mut(), 32);
    if hh < 3 || ww < 3 {
        return (mean, 0.0);
    }
    let nh = (hh - 2) * (ww - 2);
    let lap_half = laplacian_abs_sum_scalar(half.as_flattened(), 32, hh, ww) / nh as f32;
    (mean, (lap_full * lap_half).sqrt())
}

#[cfg(test)]
pub(crate) fn dark_structure_stats<T: AqLuma>(
    yp: &[T],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
    scale: f32,
) -> (f32, f32) {
    let h = height.saturating_sub(sb_y).min(64);
    let w = width.saturating_sub(sb_x).min(64);
    if h == 0 || w == 0 {
        return (0.0, 0.0);
    }
    let mut buf = [[0f32; 64]; 64];
    for (r, row) in buf.iter_mut().enumerate().take(h) {
        let base = (sb_y + r) * pw + sb_x;
        for (yp, dst) in yp[base..base + w].iter().zip(row.iter_mut()) {
            let v = yp.to_f32() * scale;
            *dst = v;
        }
    }
    dark_structure_stats_scalar(&buf, h, w)
}

#[inline]
fn dark_protection_from_stats(d: &DarkAq, base_q: i32, mean: f32, mid_energy: f32) -> i32 {
    if !d.enabled || base_q < d.min_q || mid_energy <= 0.0 {
        return 0;
    }
    let dark_weight = ((d.mean_floor + d.dark_ref) / (d.mean_floor + mean))
        .powf(d.gamma)
        .clamp(1.0, d.max_weight);
    let darkness = dark_weight - 1.0;
    if darkness <= 0.0 {
        return 0;
    }
    let dark_structure = dirty_log1pf(mid_energy * darkness);
    (dark_structure * d.scale)
        .min(d.max_qidx as f32)
        .max(0.0)
        .round() as i32
}

/// Extra qindex reduction (>= 0) for a dark, structured SB. 0 when disabled, out of the
/// gated quality range, or the SB carries no cross-scale structure. `scale` normalizes
/// the plane to 8-bit range (AV2: `1.0`; AV1: `1/(1<<(bd-8))`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dark_protection<T: AqLuma>(
    d: &DarkAq,
    base_q: i32,
    yp: &[T],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
    scale: f32,
) -> i32 {
    if !d.enabled || base_q < d.min_q {
        return 0;
    }
    let (mean, mid_energy) = dark_structure_stats(yp, pw, sb_y, sb_x, width, height, scale);
    dark_protection_from_stats(d, base_q, mean, mid_energy)
}

/// Superblock qindex-style AQ modulations, attached to an encode via
/// [`EncodeConfig::with_boost`](crate::EncodeConfig::with_dark_aq_config). [`Default`] is the
/// validated preset (Dark AQ on at all distances, scale 4; Variance Boost off).
///
/// - `vb_strength`  Variance Boost strength (0 ⇒ VB off). Default `0.0`.
/// - `octile`       1..=8 representative-variance octile. Default `6`.
/// - `qstep`        qindex→log2-gain factor: `gain = 2^(-Δqidx * qstep)`. Default `0.02`.
/// - `boost_only`   never coarsen busy SBs (pure additive boost). Default `false`.
/// - `dark`         Dark AQ sub-config (`enabled`, `scale`, …). Default on, scale `4.0`.
/// - `dark_min_d`   Min butteraugli distance for Dark AQ to engage. Default `0.0` (all).
/// - `vb_min_d`     Min butteraugli distance for Variance Boost to engage. Default `0.0`
///   (all distances). VB reallocates within a fixed budget, which only pays once the
///   fine-grained 8x8 masking AQ has faded out (d >~ 2).
/// - `vb_edge_t`    Edge/text content threshold: an SB whose sub-block variance
#[derive(Clone, Copy, Debug)]
pub struct DarkAqConfig {
    pub vb_strength: f32,
    pub octile: u8,
    pub qstep: f32,
    pub boost_only: bool,
    pub dark: DarkAq,
    pub dark_min_d: f32,
    pub vb_min_d: f32,
    pub vb_edge_t: f32,
}

/// Blue-axis structure contribution to Dark AQ. Dark AQ's
/// luminance-only Laplacian misses detail whose contrast lives mostly in B-Y;
/// this lets the existing darkness gate see that structure without changing
/// the behavior of smooth blue fields.
fn blue_dark_aq_strength(distance: f32) -> f32 {
    // The shared field stops being an efficient way to buy B-Y precision once
    // the fine AQ has mostly faded: the d=1.5 point spends primarily on Y.
    // Below d=0.4 the integer field is already fine enough that this either
    // rounds away or perturbs transform choice without a chroma payoff.
    if !(0.4..1.4).contains(&distance) {
        return 0.0;
    }
    1.0
}

impl Default for DarkAqConfig {
    /// The validated defaults: Dark AQ on at all distances (scale 4), Variance Boost off.
    fn default() -> Self {
        DarkAqConfig {
            vb_strength: 0.0,
            octile: 6,
            qstep: 0.02,
            boost_only: false,
            dark: DarkAq {
                scale: 4.0,
                ..DarkAq::on()
            },
            dark_min_d: 0.0,
            vb_min_d: 0.0,
            vb_edge_t: f32::INFINITY,
        }
    }
}

impl DarkAqConfig {
    /// Parse a config string (used by CLI front-ends). A bare `1`/`on`/`true` selects
    /// the validated defaults; otherwise a positional CSV
    /// `vb_strength,octile,qstep,boost_only,dark,dark_scale,dark_min_d` where empty
    /// fields keep their default. Returns `None` on a malformed field.
    pub fn parse(s: &str) -> Option<DarkAqConfig> {
        let mut cfg = DarkAqConfig::default();
        // Shortcut: a bare "1"/"on"/"true" ⇒ validated defaults (Dark AQ only).
        if matches!(s.to_ascii_lowercase().as_str(), "1" | "on" | "true") {
            return Some(cfg);
        }
        // Positional CSV; an empty field keeps that field's default.
        let mut it = s.split(',').map(|f| f.trim());
        let mut field = |set: &mut dyn FnMut(&str) -> Option<()>| -> Option<()> {
            match it.next() {
                None | Some("") => Some(()),
                Some(f) => set(f),
            }
        };
        field(&mut |f| {
            cfg.vb_strength = f.parse().ok()?;
            Some(())
        })?;
        field(&mut |f| {
            cfg.octile = f.parse().ok()?;
            Some(())
        })?;
        field(&mut |f| {
            cfg.qstep = f.parse().ok()?;
            Some(())
        })?;
        field(&mut |f| {
            cfg.boost_only = f.parse::<i32>().ok()? != 0;
            Some(())
        })?;
        field(&mut |f| {
            cfg.dark.enabled = f.parse::<i32>().ok()? != 0;
            Some(())
        })?;
        field(&mut |f| {
            cfg.dark.scale = f.parse().ok()?;
            Some(())
        })?;
        field(&mut |f| {
            cfg.dark_min_d = f.parse().ok()?;
            Some(())
        })?;
        field(&mut |f| {
            cfg.vb_min_d = f.parse().ok()?;
            Some(())
        })?;
        field(&mut |f| {
            cfg.vb_edge_t = f.parse().ok()?;
            Some(())
        })?;
        Some(cfg)
    }
}

/// The 64 per-8x8-block variances of a 64x64 luma superblock tile (row-major,
/// stride `pw`). Blocks that fall outside `[w, h]` are filled with the SB mean so
/// they neither create nor suppress boost at the image edge.
fn subblock_variances(tile: &[f32], pw: usize, w: usize, h: usize) -> [f32; 64] {
    debug_assert!(pw > 0 && w <= pw && tile.len() >= pw * h);
    let mut subvars = [0f32; 64];
    let valid_block_rows = h.div_ceil(8).min(8);
    let valid_block_cols = w.div_ceil(8).min(8);
    for (by, out_row) in subvars.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        if by >= valid_block_rows {
            out_row.fill(f32::NAN);
            continue;
        }
        let y0 = by * 8;
        let block_height = (h - y0).min(8);
        for (bx, out) in out_row.iter_mut().enumerate() {
            if bx >= valid_block_cols {
                *out = f32::NAN;
                continue;
            }
            let x0 = bx * 8;
            let x1 = (x0 + 8).min(w);
            let mut sum = 0f32;
            let mut sum2 = 0f32;
            let mut n = 0u32;
            for row in tile.chunks_exact(pw).skip(y0).take(block_height) {
                for &v in &row[x0..x1] {
                    sum += v;
                    sum2 = fmla(v, v, sum2);
                    n += 1;
                }
            }
            debug_assert!(n > 0);
            let m = sum / n as f32;
            *out = (sum2 / n as f32 - m * m).max(0.0);
        }
    }
    // Partial/empty blocks (NaN) get the SB-mean variance so they don't skew the octile.
    // Mean of the valid variances is a safer neutral than 0 for the octile pick.
    let (mut vsum, mut vn) = (0f32, 0u32);
    for &v in &subvars {
        if !v.is_nan() {
            vsum += v;
            vn += 1;
        }
    }
    let fill = if vn > 0 { vsum / vn as f32 } else { 0.0 };
    for v in &mut subvars {
        if v.is_nan() {
            *v = fill;
        }
    }
    subvars
}

/// Apply the Variance-Boost + Dark-AQ superblock modulations to `raw_quant_field`
/// in place. `x0`/`y0` are the DC group's pixel origin in `opsin`; the field is at
/// 8x8-block resolution for this DC group. Only called when a boost config is set on
/// the [`EncodeConfig`](crate::EncodeConfig).
pub(crate) fn apply_boost(
    cell: &mut Vec<f32>,
    cfg: &DarkAqConfig,
    opsin: &Image3F,
    raw_quant_field: &mut ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
    blue_heavy: bool,
    apply_quant_field_gain: ApplyQuantFieldGainFn,
    dark_structure_stats: DarkStructureStatsFn,
    fill_blue_tile: FillBlueTileFn,
) {
    let xblocks = raw_quant_field.xsize();
    let yblocks = raw_quant_field.ysize();
    if xblocks == 0 || yblocks == 0 {
        return;
    }
    let img_w = opsin.xsize();
    let img_h = opsin.ysize();

    // Superblocks are 8x8 blocks (64x64 px). Grid over the DC group.
    let sb_cols = xblocks.div_ceil(8);
    let sb_rows = yblocks.div_ceil(8);

    let vb_on = cfg.vb_strength > 0.0 && distance >= cfg.vb_min_d;
    let dark_on = cfg.dark.enabled && distance >= cfg.dark_min_d;
    if !vb_on && !dark_on {
        return;
    }
    // Synthetic base_q so the Dark-AQ internal gate (base_q >= min_q) passes exactly
    // when the distance gate above already allowed it.
    let base_q = if dark_on { cfg.dark.min_q } else { 0 };

    // First pass: octile variance per SB (+ its tile buffer reused in pass 2 via
    // recompute — SBs are cheap and this keeps memory flat). Also accumulate the
    // mean log-variance reference for the two-sided cut. `cell` holds the picked
    // variance in [0, n) and the edge/text dispersion score in [n, 2n).
    let n_sb = sb_cols * sb_rows;
    if vb_on && cell.len() < 2 * n_sb {
        cell.resize_with(2 * n_sb, Default::default);
    }
    let mut ref_acc = 0f32;
    let mut ref_n = 0u32;
    let mut tile = [0f32; 64 * 64];
    let blue_dark_strength = if blue_heavy {
        blue_dark_aq_strength(distance)
    } else {
        0.0
    };

    let fill_tile = |tile: &mut [f32], sb_x0: usize, sb_y0: usize| -> (usize, usize) {
        let w = img_w.saturating_sub(sb_x0).min(64);
        let h = img_h.saturating_sub(sb_y0).min(64);
        for (r, dst) in tile.as_chunks_mut::<64>().0.iter_mut().take(h).enumerate() {
            let row = opsin.plane_row(1, sb_y0 + r);
            for (d, &s) in dst.iter_mut().zip(row[sb_x0..sb_x0 + w].iter()) {
                *d = s * Y_TO_LUMA8;
            }
        }
        (w, h)
    };

    if vb_on {
        let (picked, edge) = cell[..2 * n_sb].split_at_mut(n_sb);
        for sby in 0..sb_rows {
            for sbx in 0..sb_cols {
                let i = sby * sb_cols + sbx;
                let sb_x0 = x0 + sbx * 64;
                let sb_y0 = y0 + sby * 64;
                let (w, h) = fill_tile(&mut tile, sb_x0, sb_y0);
                if w == 0 || h == 0 {
                    picked[i] = 0.0;
                    edge[i] = 0.0;
                    continue;
                }
                let mut subvars = subblock_variances(&tile, 64, w, h);
                // Edge/text score: dispersion between the busiest and the median
                // sub-block. Flat background + a few extreme edge blocks (text,
                // signs) scores high; uniform texture scores low.
                let v_max = subvars.iter().copied().fold(0f32, f32::max);
                let v_med = sb_octile_variance(&mut subvars.clone(), 4);
                edge[i] = dirty_log1pf(v_max) - dirty_log1pf(v_med);
                let pv = sb_octile_variance(&mut subvars, cfg.octile);
                picked[i] = pv;
                ref_acc += dirty_log1pf(pv);
                ref_n += 1;
            }
        }
    }
    let ref_log = if ref_n > 0 {
        ref_acc / ref_n as f32
    } else {
        0.0
    };

    // Zero-mean the VB deltas across the DC group so VB is a pure spatial
    // reallocation, never a global rate change. The boost side's low-variance
    // threshold is absolute, so on majority-flat content (sky, signs) almost
    // every SB qualifies and un-normalized VB turns into a blanket overspend.
    // Skipped under `boost_only`, whose semantics are an intentional one-sided
    // spend. Deltas are recomputed cheaply from `cell` in the apply loop.
    // Edge/text content classifier at DC-group granularity: if the MAJORITY of
    // SBs are edge-dominated (flat background + sharp strokes — text, signs,
    // line art), variance is a wrong importance proxy for this content and VB
    // is switched off for the whole group. Per-SB exclusion was measured worse
    // than no gate at all (it concentrates the zero-mean reallocation into a
    // small arbitrary residual pool); the classifier must act wholesale.
    let vb_group_on = vb_on && {
        let edge = &cell[n_sb..2 * n_sb];
        let edgy = edge.iter().filter(|&&e| e > cfg.vb_edge_t).count();
        2 * edgy <= n_sb
    };

    let mut vb_mean = 0f32;
    if vb_group_on && !cfg.boost_only {
        let picked = &cell[..n_sb];
        for &pv in picked {
            vb_mean += variance_boost_delta(pv, ref_log, cfg.vb_strength, false) as f32;
        }
        vb_mean /= n_sb as f32;
    }

    // Second pass: convert each SB's summed qindex delta into a field gain and apply.
    for sby in 0..sb_rows {
        for sbx in 0..sb_cols {
            let sb_x0 = x0 + sbx * 64;
            let sb_y0 = y0 + sby * 64;

            let mut delta = 0f32;
            if vb_group_on {
                let src = cell[sby * sb_cols + sbx];
                delta += variance_boost_delta(src, ref_log, cfg.vb_strength, cfg.boost_only) as f32
                    - vb_mean;
            }
            if dark_on {
                let (w, h) = fill_tile(&mut tile, sb_x0, sb_y0);
                if w >= 3 && h >= 3 {
                    // Tile already in 8-bit-luma units (scale=1.0 here).
                    let rows = tile.as_chunks::<64>().0;
                    let (mean, mut mid_energy) = dark_structure_stats(rows, h, w);
                    if blue_dark_strength > 0.0 {
                        let blue_area = fill_blue_tile(opsin, &mut tile, sb_x0, sb_y0, w, h);
                        if blue_area > 0.0 {
                            let rows = tile.as_chunks::<64>().0;
                            let (_, blue_energy) = dark_structure_stats(rows, h, w);
                            // Max rather than addition avoids charging the same
                            // edge twice when it is already visible in Y.
                            mid_energy =
                                mid_energy.max(blue_energy * blue_area * blue_dark_strength);
                        }
                    }
                    delta -= dark_protection_from_stats(&cfg.dark, base_q, mean, mid_energy) as f32;
                }
            }
            if delta == 0.0 {
                continue;
            }
            // qindex delta (negative = finer) → multiplicative field gain.
            let gain = fast_exp2(-delta * cfg.qstep);

            let bx0 = sbx * 8;
            let by0 = sby * 8;
            apply_quant_field_gain(
                raw_quant_field,
                bx0,
                by0,
                (xblocks - bx0).min(8),
                (yblocks - by0).min(8),
                gain,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_dark_aq_is_confined_to_the_rd_winning_band() {
        for distance in [0.0, 0.399_999, 1.4, 2.0, 10.0] {
            assert_eq!(blue_dark_aq_strength(distance), 0.0);
        }
        for distance in [0.4, 0.5, 1.0, 1.25, 1.399_999] {
            assert_eq!(blue_dark_aq_strength(distance), 1.0);
        }
    }

    fn check_fill_blue_tile(method: FillBlueTileFn) {
        let mut opsin = Image3F::new(83, 79);
        for y in 0..opsin.ysize() {
            for x in 0..opsin.xsize() {
                let t = (x * 37 + y * 53 + x * y * 3) as f32;
                opsin.plane_row_mut(0, y)[x] = (t * 0.017).sin() * 0.08;
                opsin.plane_row_mut(1, y)[x] = 0.03 + (t * 0.011).cos().abs() * 0.35;
                opsin.plane_row_mut(2, y)[x] =
                    opsin.plane_row(1, y)[x] + (t * 0.023).sin() * 0.12 + 0.025;
            }
        }
        for &(x0, y0, w, h) in &[
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (3, 5, 3, 7),
            (7, 9, 8, 8),
            (11, 13, 15, 17),
            (19, 7, 31, 63),
            (18, 15, 64, 64),
        ] {
            let mut expected = [f32::NAN; 64 * 64];
            let mut actual = expected;
            let expected_area = fill_blue_tile_scalar(&opsin, &mut expected, x0, y0, w, h);
            let actual_area = method(&opsin, &mut actual, x0, y0, w, h);
            let tolerance = 2e-6 * expected_area.abs().max(1.0);
            assert!(
                (actual_area - expected_area).abs() <= tolerance,
                "shape {w}x{h} at ({x0},{y0}): area {actual_area} vs {expected_area}"
            );
            for y in 0..h {
                assert_eq!(
                    &actual[y * 64..y * 64 + w],
                    &expected[y * 64..y * 64 + w],
                    "shape {w}x{h}, row {y}"
                );
            }
        }
    }

    #[test]
    fn selected_fill_blue_tile_matches_scalar() {
        check_fill_blue_tile(select_fill_blue_tile_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_fill_blue_tile_matches_scalar() {
        if is_x86_feature_detected!("avx2") {
            check_fill_blue_tile(|opsin, tile, x0, y0, w, h| unsafe {
                crate::avx::fill_blue_tile_avx2(opsin, tile, x0, y0, w, h)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_fill_blue_tile_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_fill_blue_tile(|opsin, tile, x0, y0, w, h| unsafe {
                crate::sse::fill_blue_tile_sse41(opsin, tile, x0, y0, w, h)
            });
        }
    }

    fn check_dark_structure_stats(method: DarkStructureStatsFn) {
        let mut tile = [[0.0f32; 64]; 64];
        let mut state = 0xa511_e9b3u32;
        for row in &mut tile {
            for value in row {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *value = (state >> 8) as f32 / (1u32 << 24) as f32 * Y_TO_LUMA8;
            }
        }
        for (h, w) in [
            (0, 0),
            (1, 1),
            (2, 17),
            (3, 3),
            (3, 7),
            (7, 3),
            (8, 8),
            (11, 17),
            (17, 31),
            (63, 64),
            (64, 63),
            (64, 64),
        ] {
            let expected = dark_structure_stats_scalar(&tile, h, w);
            let actual = method(&tile, h, w);
            for (label, actual, expected) in [
                ("mean", actual.0, expected.0),
                ("energy", actual.1, expected.1),
            ] {
                let tolerance = 2e-4f32.max(expected.abs() * 8e-6);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "shape {w}x{h} {label}: actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn selected_dark_structure_stats_matches_scalar() {
        check_dark_structure_stats(select_dark_structure_stats_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_dark_structure_stats_matches_scalar() {
        if is_x86_feature_detected!("avx2") {
            check_dark_structure_stats(|buf, h, w| unsafe {
                crate::avx::dark_structure_stats_avx2(buf, h, w)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_dark_structure_stats_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_dark_structure_stats(|buf, h, w| unsafe {
                crate::sse::dark_structure_stats_sse41(buf, h, w)
            });
        }
    }

    fn check_apply_quant_field_gain(method: ApplyQuantFieldGainFn) {
        let source = [
            0u8, 1, 2, 3, 7, 15, 31, 63, 127, 128, 254, 255, 17, 91, 149, 223, 5,
        ];
        let gains = [
            -1.0,
            0.0,
            0.25,
            0.5,
            0.999,
            1.0,
            1.5,
            2.0,
            300.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for len in 0..=source.len() {
            for &gain in &gains {
                let mut expected = ImageB::new(source.len() + 2, 5);
                for y in 0..5 {
                    for (x, value) in expected.row_mut(y).iter_mut().enumerate() {
                        *value = source[(y * (source.len() + 2) + x) % source.len()];
                    }
                }
                let mut actual = expected.clone();
                apply_quant_field_gain_scalar(&mut expected, 1, 1, len, 3, gain);
                method(&mut actual, 1, 1, len, 3, gain);
                assert_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                    "length {len}, gain {gain}"
                );
            }
        }
    }

    #[test]
    fn selected_quant_field_gain_matches_scalar() {
        check_apply_quant_field_gain(select_apply_quant_field_gain_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_quant_field_gain_matches_scalar() {
        if is_x86_feature_detected!("avx2") {
            check_apply_quant_field_gain(|image, x0, y0, width, height, gain| unsafe {
                crate::avx::apply_quant_field_gain_avx2(image, x0, y0, width, height, gain)
            });
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_quant_field_gain_matches_scalar() {
        if is_x86_feature_detected!("sse4.1") {
            check_apply_quant_field_gain(|image, x0, y0, width, height, gain| unsafe {
                crate::sse::apply_quant_field_gain_sse41(image, x0, y0, width, height, gain)
            });
        }
    }

    #[test]
    fn parse_shortcut_and_fields() {
        // Shortcut ⇒ validated defaults: VB off, dark on all-distances scale 4.
        let c = DarkAqConfig::parse("1").unwrap();
        assert_eq!(c.vb_strength, 0.0);
        assert!(c.dark.enabled);
        assert_eq!(c.dark.scale, 4.0);
        assert_eq!(c.dark_min_d, 0.0);
        assert_eq!(DarkAqConfig::parse("on").unwrap().dark.scale, 4.0);
        // Empty fields keep defaults; only the given ones override.
        let c = DarkAqConfig::parse("1.5,,,,0").unwrap();
        assert_eq!(c.vb_strength, 1.5);
        assert_eq!(c.octile, 6); // untouched
        assert!(!c.dark.enabled); // field 5 = 0
        // Full CSV: explicit variance-boost opt-in.
        let c = DarkAqConfig::parse("1,4,0.03,1,1,6,2.0,2.5").unwrap();
        assert_eq!(c.vb_strength, 1.0);
        assert_eq!(c.octile, 4);
        assert!(c.boost_only);
        assert_eq!(c.dark.scale, 6.0);
        assert_eq!(c.dark_min_d, 2.0);
        assert_eq!(c.vb_min_d, 2.5);
        // vb_min_d defaults to 0 (VB at all distances) when omitted.
        assert_eq!(DarkAqConfig::parse("1.5").unwrap().vb_min_d, 0.0);
        // vb_edge_t: 9th field; defaults to infinity (classifier off).
        let c = DarkAqConfig::parse("1,4,0.03,1,1,6,2.0,2.5,3.0").unwrap();
        assert_eq!(c.vb_edge_t, 3.0);
        assert_eq!(DarkAqConfig::parse("1.5").unwrap().vb_edge_t, f32::INFINITY);
        // Garbage field ⇒ None (pass stays disabled rather than mis-encoding).
        assert!(DarkAqConfig::parse("1,notanumber").is_none());
    }

    #[test]
    fn octile_indices() {
        let mut v: [f32; 64] = std::array::from_fn(|i| i as f32);
        assert_eq!(sb_octile_variance(&mut v.clone(), 1), 7.0);
        assert_eq!(sb_octile_variance(&mut v.clone(), 6), 47.0);
        assert_eq!(sb_octile_variance(&mut v, 8), 63.0);
    }

    #[test]
    fn octile_selection_matches_full_sort() {
        let mut values = [0.0f32; 64];
        let mut state = 0x6d2b_79f5u32;
        for value in &mut values {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = (state >> 12) as f32;
        }
        let mut sorted = values;
        sorted.sort_unstable_by(f32::total_cmp);
        for octile in 1..=8 {
            assert_eq!(
                sb_octile_variance(&mut values, octile),
                sorted[octile as usize * 8 - 1]
            );
        }
    }

    #[test]
    fn scalar_laplacian_matches_indexed_reference() {
        let mut buf = [[0.0f32; 64]; 64];
        let mut state = 0x1234_5678u32;
        for row in &mut buf {
            for value in row {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *value = (state >> 8) as f32 / (1u32 << 24) as f32;
            }
        }

        for (h, w) in [(3, 3), (3, 17), (11, 3), (17, 29), (64, 64)] {
            let mut expected = 0.0f32;
            for r in 1..h - 1 {
                for c in 1..w - 1 {
                    let l = 4.0 * buf[r][c]
                        - buf[r - 1][c]
                        - buf[r + 1][c]
                        - buf[r][c - 1]
                        - buf[r][c + 1];
                    expected += l.abs();
                }
            }
            let actual = laplacian_abs_sum_scalar(buf.as_flattened(), 64, h, w);
            assert_eq!(actual, expected, "shape {w}x{h}");
        }
    }

    #[test]
    fn scalar_downsample_matches_indexed_reference() {
        let mut src = [[0.0f32; 64]; 64];
        for (r, row) in src.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = (r * 67 + c * 13) as f32 * 0.125;
            }
        }

        for (h, w) in [(2, 2), (2, 17), (11, 2), (17, 29), (64, 64)] {
            let mut actual = [[f32::NAN; 32]; 32];
            let shape = box_downsample_2x_scalar(
                src.as_flattened(),
                64,
                h,
                w,
                actual.as_flattened_mut(),
                32,
            );
            assert_eq!(shape, (h / 2, w / 2));
            for r in 0..h / 2 {
                for c in 0..w / 2 {
                    let expected = 0.25
                        * (src[2 * r][2 * c]
                            + src[2 * r][2 * c + 1]
                            + src[2 * r + 1][2 * c]
                            + src[2 * r + 1][2 * c + 1]);
                    assert_eq!(actual[r][c], expected, "shape {w}x{h}, sample {c},{r}");
                }
            }
        }
    }

    #[test]
    fn sliced_subblock_variances_match_indexed_reference() {
        fn reference(tile: &[f32], pw: usize, w: usize, h: usize) -> [f32; 64] {
            let mut out = [f32::NAN; 64];
            for by in 0..8 {
                for bx in 0..8 {
                    let (mut sum, mut sum2, mut n) = (0.0f32, 0.0f32, 0u32);
                    for r in 0..8 {
                        let y = by * 8 + r;
                        if y >= h {
                            break;
                        }
                        for c in 0..8 {
                            let x = bx * 8 + c;
                            if x >= w {
                                break;
                            }
                            let v = tile[y * pw + x];
                            sum += v;
                            sum2 += v * v;
                            n += 1;
                        }
                    }
                    if n != 0 {
                        let mean = sum / n as f32;
                        out[by * 8 + bx] = (sum2 / n as f32 - mean * mean).max(0.0);
                    }
                }
            }
            let (mut sum, mut n) = (0.0f32, 0u32);
            for &v in &out {
                if !v.is_nan() {
                    sum += v;
                    n += 1;
                }
            }
            let fill = if n == 0 { 0.0 } else { sum / n as f32 };
            for v in &mut out {
                if v.is_nan() {
                    *v = fill;
                }
            }
            out
        }

        let mut tile = [0.0f32; 64 * 64];
        let mut state = 0x9e37_79b9u32;
        for value in &mut tile {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = (state >> 8) as f32 / (1u32 << 24) as f32;
        }
        for (h, w) in [(0, 0), (1, 1), (7, 9), (8, 8), (17, 31), (63, 64), (64, 64)] {
            subblock_variances(&tile, 64, w, h)
                .iter()
                .zip(reference(&tile, 64, w, h).iter())
                .for_each(|(&a, &b)| {
                    assert!(
                        (a - b).abs() < 1e-7,
                        "Dark AQ reference failed with {a} and {b}"
                    );
                });
        }
    }

    #[test]
    fn direct_dark_tile_stats_match_gathered_path() {
        let mut tile = [0.0f32; 64 * 64];
        let mut state = 0xa511_e9b3u32;
        for value in &mut tile {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = (state >> 8) as f32 / (1u32 << 24) as f32 * Y_TO_LUMA8;
        }
        let rows = tile.as_chunks::<64>().0;
        for (h, w) in [(1, 1), (3, 7), (11, 17), (63, 64), (64, 64)] {
            assert_eq!(
                dark_structure_stats_scalar(rows, h, w),
                dark_structure_stats(&tile, 64, 0, 0, w, h, 1.0),
                "shape {w}x{h}"
            );
        }
    }

    #[test]
    fn flat_sb_boosts_busy_sb_cuts() {
        // Flat SB (var ~0): strong boost (negative).
        let boost = variance_boost_delta(0.0, 8.0, 1.0, false);
        assert!(boost < 0, "flat SB should boost, got {boost}");
        // Busy SB well above the tile ref: coarsen (positive) when two-sided.
        let cut = variance_boost_delta(1.0e6, 6.0, 1.0, false);
        assert!(cut > 0, "busy SB should cut, got {cut}");
        // boost_only never coarsens.
        assert_eq!(variance_boost_delta(1.0e6, 6.0, 1.0, true), 0);
    }

    #[test]
    fn dark_protection_gated_and_darkness_only() {
        let d = DarkAq::on();
        // A dark, structured tile: 4x4-block checker (10/40) — structure survives the
        // 2x downsample, so cross-scale mid_energy > 0; mean ~25.
        let mut tile = [0f32; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                tile[y * 64 + x] = if ((x / 4) + (y / 4)) & 1 == 0 {
                    10.0
                } else {
                    40.0
                };
            }
        }
        let boost = dark_protection(&d, d.min_q, &tile, 64, 0, 0, 64, 64, 1.0);
        assert!(
            boost > 0,
            "dark structured SB should be protected, got {boost}"
        );
        // Below the quality gate: nothing.
        assert_eq!(
            dark_protection(&d, d.min_q - 1, &tile, 64, 0, 0, 64, 64, 1.0),
            0
        );
        // Bright version of the same structure (same cross-scale energy): no darkness
        // ⇒ no boost, isolating the darkness gate from the structure gate.
        let mut bright = [0f32; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                bright[y * 64 + x] = if ((x / 4) + (y / 4)) & 1 == 0 {
                    200.0
                } else {
                    230.0
                };
            }
        }
        assert_eq!(
            dark_protection(&d, d.min_q, &bright, 64, 0, 0, 64, 64, 1.0),
            0
        );
    }
}
