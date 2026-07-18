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
use crate::image::{Image3F, ImageB};
use crate::util::FastRound;
use std::cell::RefCell;

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
        -(d.fast_round() as i32)
    } else if boost_only {
        0
    } else {
        // Higher contrast: coarsen relative to the tile reference, capped. Using the
        // reference (not the threshold) keeps well-textured frames near zero-mean.
        let over = (v_log - ref_log.max(LOW_LOG)).max(0.0);
        let d = (over * CUT_SLOPE * strength).min(MAX_CUT);
        d.fast_round() as i32
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
fn laplacian_abs_sum<const STRIDE: usize>(buf: &[[f32; STRIDE]], h: usize, w: usize) -> (f32, u32) {
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for rows in buf[..h].array_windows::<3>() {
        let [top, middle, bottom] = rows;
        for ((&up, &down), &[left, center, right]) in top[1..w - 1]
            .iter()
            .zip(bottom[1..w - 1].iter())
            .zip(middle[..w].array_windows::<3>())
        {
            let l = 4.0 * center - up - down - left - right;
            sum += l.abs();
            n += 1;
        }
    }
    (sum, n)
}

#[inline]
fn box_downsample_2x<const SRC_STRIDE: usize, const DST_STRIDE: usize>(
    src: &[[f32; SRC_STRIDE]],
    h: usize,
    w: usize,
    dst: &mut [[f32; DST_STRIDE]],
) -> (usize, usize) {
    let (hh, ww) = (h / 2, w / 2);
    for (dst_row, rows) in dst[..hh]
        .iter_mut()
        .zip(src[..h].array_windows::<2>().step_by(2))
    {
        let [top, bottom] = rows;
        for (out, (&[top_left, top_right], &[bottom_left, bottom_right])) in
            dst_row[..ww].iter_mut().zip(
                top[..w]
                    .array_windows::<2>()
                    .step_by(2)
                    .zip(bottom[..w].array_windows::<2>().step_by(2)),
            )
        {
            *out = 0.25 * (top_left + top_right + bottom_left + bottom_right);
        }
    }
    (hh, ww)
}

#[inline]
fn dark_structure_stats_buf(buf: &[[f32; 64]], h: usize, w: usize) -> (f32, f32) {
    let mut sum = 0.0f32;
    for row in buf[..h].iter() {
        for &v in &row[..w] {
            sum += v;
        }
    }
    let mean = sum / (h * w) as f32;
    if h < 3 || w < 3 {
        return (mean, 0.0);
    }
    let (lap_full, nf) = laplacian_abs_sum(buf, h, w);
    let lap_full = lap_full / nf as f32;
    let mut half = [[0f32; 32]; 32];
    let (hh, ww) = box_downsample_2x(buf, h, w, &mut half);
    if hh < 3 || ww < 3 {
        return (mean, 0.0);
    }
    let (lap_half, nh) = laplacian_abs_sum(&half, hh, ww);
    let lap_half = lap_half / nh as f32;
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
    dark_structure_stats_buf(&buf, h, w)
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
        .fast_round() as i32
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
#[derive(Clone, Copy, Debug)]
pub struct DarkAqConfig {
    pub vb_strength: f32,
    pub octile: u8,
    pub qstep: f32,
    pub boost_only: bool,
    pub dark: DarkAq,
    pub dark_min_d: f32,
}

const Y_TO_LUMA8: f32 = 300.0;

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
                    sum2 += v * v;
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

thread_local! {
    static DARK_OCTILE: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Apply the Variance-Boost + Dark-AQ superblock modulations to `raw_quant_field`
/// in place. `x0`/`y0` are the DC group's pixel origin in `opsin`; the field is at
/// 8x8-block resolution for this DC group. Only called when a boost config is set on
/// the [`EncodeConfig`](crate::EncodeConfig).
pub(crate) fn apply_boost(
    cfg: &DarkAqConfig,
    opsin: &Image3F,
    raw_quant_field: &mut ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
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

    let vb_on = cfg.vb_strength > 0.0;
    let dark_on = cfg.dark.enabled && distance >= cfg.dark_min_d;
    if !vb_on && !dark_on {
        return;
    }
    // Synthetic base_q so the Dark-AQ internal gate (base_q >= min_q) passes exactly
    // when the distance gate above already allowed it.
    let base_q = if dark_on { cfg.dark.min_q } else { 0 };

    // First pass: octile variance per SB (+ its tile buffer reused in pass 2 via
    // recompute — SBs are cheap and this keeps memory flat). Also accumulate the
    // mean log-variance reference for the two-sided cut.
    DARK_OCTILE.with_borrow_mut(|cell| {
        if vb_on && cell.len() < sb_cols * sb_rows {
            cell.resize_with(sb_cols * sb_rows, Default::default);
        }
        let mut ref_acc = 0f32;
        let mut ref_n = 0u32;
        let mut tile = [0f32; 64 * 64];

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
            let picked = &mut cell[..sb_cols * sb_rows];
            for (sby, row) in picked.chunks_exact_mut(sb_cols).enumerate().take(sb_rows) {
                for (sbx, dst) in row.iter_mut().enumerate() {
                    let sb_x0 = x0 + sbx * 64;
                    let sb_y0 = y0 + sby * 64;
                    let (w, h) = fill_tile(&mut tile, sb_x0, sb_y0);
                    if w == 0 || h == 0 {
                        *dst = 0.0;
                        continue;
                    }
                    let mut subvars = subblock_variances(&tile, 64, w, h);
                    let pv = sb_octile_variance(&mut subvars, cfg.octile);
                    *dst = pv;
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

        // Second pass: convert each SB's summed qindex delta into a field gain and apply.
        for sby in 0..sb_rows {
            for sbx in 0..sb_cols {
                let sb_x0 = x0 + sbx * 64;
                let sb_y0 = y0 + sby * 64;

                let mut delta = 0i32;
                if vb_on {
                    let src = cell[sby * sb_cols + sbx];
                    delta += variance_boost_delta(src, ref_log, cfg.vb_strength, cfg.boost_only);
                }
                if dark_on {
                    let (w, h) = fill_tile(&mut tile, sb_x0, sb_y0);
                    if w >= 3 && h >= 3 {
                        // Tile already in 8-bit-luma units (scale=1.0 here).
                        let rows = tile.as_chunks::<64>().0;
                        let (mean, mid_energy) = dark_structure_stats_buf(rows, h, w);
                        delta -= dark_protection_from_stats(&cfg.dark, base_q, mean, mid_energy);
                    }
                }
                if delta == 0 {
                    continue;
                }
                // qindex delta (negative = finer) → multiplicative field gain.
                let gain = fast_exp2(-(delta as f32) * cfg.qstep);

                let bx0 = sbx * 8;
                let by0 = sby * 8;
                for by in by0..(by0 + 8).min(yblocks) {
                    let row = raw_quant_field.row_mut(by);
                    for dst in row[bx0..(bx0 + 8).min(xblocks)].iter_mut() {
                        let q = *dst as f32 * gain;
                        *dst = q.fast_round().clamp(1.0, 255.0) as u8;
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let c = DarkAqConfig::parse("1,4,0.03,1,1,6,2.0").unwrap();
        assert_eq!(c.vb_strength, 1.0);
        assert_eq!(c.octile, 4);
        assert!(c.boost_only);
        assert_eq!(c.dark.scale, 6.0);
        assert_eq!(c.dark_min_d, 2.0);
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
    fn array_windows_laplacian_matches_indexed_reference() {
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
            let mut expected_n = 0u32;
            for r in 1..h - 1 {
                for c in 1..w - 1 {
                    let l = 4.0 * buf[r][c]
                        - buf[r - 1][c]
                        - buf[r + 1][c]
                        - buf[r][c - 1]
                        - buf[r][c + 1];
                    expected += l.abs();
                    expected_n += 1;
                }
            }
            let actual = laplacian_abs_sum(&buf, h, w);
            assert_eq!(actual, (expected, expected_n), "shape {w}x{h}");
        }
    }

    #[test]
    fn array_windows_downsample_matches_indexed_reference() {
        let mut src = [[0.0f32; 64]; 64];
        for (r, row) in src.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = (r * 67 + c * 13) as f32 * 0.125;
            }
        }

        for (h, w) in [(2, 2), (2, 17), (11, 2), (17, 29), (64, 64)] {
            let mut actual = [[f32::NAN; 32]; 32];
            let shape = box_downsample_2x(&src, h, w, &mut actual);
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
            assert_eq!(
                subblock_variances(&tile, 64, w, h),
                reference(&tile, 64, w, h),
                "shape {w}x{h}"
            );
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
                dark_structure_stats_buf(rows, h, w),
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
