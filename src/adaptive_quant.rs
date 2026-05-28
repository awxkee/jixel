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

// AC-quantization field, ported faithfully from libjxl-tiny's
// `ComputeAdaptiveQuantFieldTile` (enc_adaptive_quantization.cc). Operates on
// the XYB `opsin` image (NOT linear RGB). Produces the per-8x8-block integer
// `raw_quant_field` via `clamp(round(aq_map * inv_scale), 1, 255)`, exactly as
// libjxl-tiny does. This replaces the earlier gradient heuristic, which
// produced values in 2..=6 (mostly the floor) and quantized ~2.6x too coarsely.

use crate::image::{Image3F, ImageB};

// ---- Sigmoid-like gamma constants (butteraugli <-> opsin space) ----
const K_SG_MUL: f32 = 226.0480446705883;
const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_LOG2: f32 = 0.693147181;
const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_LOG2;
const K_SG_V_OFFSET: f32 = 7.14672470003;

const K_AC_QUANT: f32 = 0.8294;
const MATCH_GAMMA_OFFSET: f32 = 0.019;
const K_X_MUL: f32 = 23.426802998210313;

/// Ratio of derivatives of cubic-root to simple-gamma; moves quantization from
/// jxl's opsin (cube-root-of-photons) space to butteraugli's log-gamma space.
#[inline]
fn ratio_cubic_to_simple_gamma(v: f32, invert: bool) -> f32 {
    let k_epsilon = 1e-2f32;
    let v = if v < 0.0 { 0.0 } else { v };
    let k_num_mul = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    let k_v_offset = K_SG_V_OFFSET * K_LOG2 + k_epsilon;
    let k_den_mul = K_LOG2 * K_SG_MUL;
    let v2 = v * v;
    let num = k_num_mul * v2 + k_epsilon;
    let den = (k_den_mul * v) * v2 + k_v_offset;
    if invert { num / den } else { den / num }
}

/// Visual-masking transform applied to the accumulated per-block difference.
#[inline]
fn compute_mask(out_val: f32) -> f32 {
    let k_base = -0.74174993f32;
    let k_mul4 = 3.2353257320940401f32;
    let k_mul2 = 12.906028311180409f32;
    let k_offset2 = 305.04035728311436f32;
    let k_mul3 = 5.0220313103171232f32;
    let k_offset3 = 2.1925739705298404f32;
    let k_offset4 = 0.25f32 * k_offset3;
    let k_mul0 = 0.74760422233706747f32;

    let v1 = (out_val * k_mul0).max(1e-3);
    let v2 = 1.0 / (v1 + k_offset2);
    let v3 = 1.0 / (v1 * v1 + k_offset3);
    let v4 = 1.0 / (v1 * v1 + k_offset4);
    k_base + k_mul4 * v4 + k_mul2 * v2 + k_mul3 * v3
}

/// `0.25 * sqrt(v * sqrt(kMul*1e8) + kLogOffset)`.
#[inline]
fn masking_sqrt(v: f32) -> f32 {
    let k_log_offset = 26.481471032459346f32;
    let k_mul = 211.50759899638012f32;
    let mul_v = k_mul * 1e8;
    0.25 * (v * mul_v.sqrt() + k_log_offset).sqrt()
}

/// HF modulation: sum of |right| and |below| abs differences in the 8x8 Y block.
fn hf_modulation(x: usize, y: usize, xyb_y: &Image3F, out_val: f32) -> f32 {
    let mut sum = 0.0f32;
    for dy in 0..8 {
        let row = xyb_y.plane_row(1, y + dy);
        let row_next = if dy == 7 {
            row
        } else {
            xyb_y.plane_row(1, y + dy + 1)
        };
        for dx in 0..8 {
            let p = row[x + dx];
            // right difference, skipping the last column (matches kMaskRight)
            if dx < 7 {
                sum += (p - row[x + dx + 1]).abs();
            }
            sum += (p - row_next[x + dx]).abs();
        }
    }
    sum * (-2.0052193233688884f32 / 112.0) + out_val
}

/// Color modulation: boost precision in strongly red or blue blocks.
fn color_modulation(
    x: usize,
    y: usize,
    xyb: &Image3F,
    butteraugli_target: f32,
    out_val: f32,
) -> f32 {
    let k_strength_mul = 2.177823400325309f32;
    let k_red_ramp_start = 0.0073200141118951231f32;
    let k_red_ramp_length = 0.019421555948474039f32;
    let k_blue_ramp_length = 0.086890611400405895f32;
    let k_blue_ramp_start = 0.26973418507870539f32;
    let strength = k_strength_mul * (1.0 - 0.25 * butteraugli_target);
    if strength < 0.0 {
        return out_val;
    }
    let red_strength = strength * 5.992297772961519f32;
    let blue_strength = strength;
    let offset = strength * -0.009174542291185913f32;
    let mut out_val = out_val + offset;

    let mut red_coverage = 0.0f32;
    let mut blue_coverage = 0.0f32;
    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        let row_b = xyb.plane_row(2, y + dy);
        for dx in 0..8 {
            let pixel_x = (row_x[x + dx] - k_red_ramp_start).max(0.0);
            let pixel_y = row_y[x + dx];
            let pixel_b = (row_b[x + dx] - (pixel_y + k_blue_ramp_start)).max(0.0);
            blue_coverage += pixel_b.min(k_blue_ramp_length);
            red_coverage += pixel_x.min(k_red_ramp_length);
        }
    }
    let ratio = 30.610615782142737f32; // out of 64 pixels
    let mut overall_red = red_coverage.min(ratio * k_red_ramp_length);
    overall_red *= red_strength / ratio;
    let mut overall_blue = blue_coverage.min(ratio * k_blue_ramp_length);
    overall_blue *= blue_strength / ratio;

    out_val = out_val + overall_red + overall_blue;
    out_val
}

/// Gamma modulation: accounts for the local gamma of the 8x8 block.
fn gamma_modulation(x: usize, y: usize, xyb: &Image3F, out_val: f32) -> f32 {
    let k_bias = 0.16f32;
    let mut overall_ratio = 0.0f32;
    for dy in 0..8 {
        let row_x = xyb.plane_row(0, y + dy);
        let row_y = xyb.plane_row(1, y + dy);
        for dx in 0..8 {
            let iny = row_y[x + dx] + k_bias;
            let inx = row_x[x + dx];
            let r = iny - inx;
            let g = iny + inx;
            let ratio_r = ratio_cubic_to_simple_gamma(r, true);
            let ratio_g = ratio_cubic_to_simple_gamma(g, true);
            overall_ratio += 0.5 * (ratio_r + ratio_g);
        }
    }
    overall_ratio *= 1.0 / 64.0;
    // ln(2) folded in: -0.15526878... * ln(2) * log2(x) == -0.15... * ln(x)
    let k_gam = -0.15526878023684174f32 * 0.693147180559945f32;
    k_gam * overall_ratio.log2() + out_val
}

#[inline]
fn store_min4(v: f32, mins: &mut [f32; 4]) {
    if v < mins[3] {
        if v < mins[0] {
            mins[3] = mins[2];
            mins[2] = mins[1];
            mins[1] = mins[0];
            mins[0] = v;
        } else if v < mins[1] {
            mins[3] = mins[2];
            mins[2] = mins[1];
            mins[1] = v;
        } else if v < mins[2] {
            mins[3] = mins[2];
            mins[2] = v;
        } else {
            mins[3] = v;
        }
    }
}

/// Fill `raw_quant_field` (block-resolution) for the DC-group region whose
/// top-left pixel is (x0, y0) in the full `opsin` (XYB) image.
///
/// `distance` is the butteraugli target; `inv_scale = 1.0 / distance_params.scale`.
pub fn fill_quant_field(
    opsin: &Image3F,
    raw_quant_field: &mut ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
    inv_scale: f32,
) {
    let xsize_blocks = raw_quant_field.xsize();
    let ysize_blocks = raw_quant_field.ysize();
    let img_xsize = opsin.xsize();
    let img_ysize = opsin.ysize();

    let scale = K_AC_QUANT / distance;

    // Pixel extent of this DC-group region (padded to whole blocks but clamped
    // to the image — the modulation loops read up to (bx*8+7, by*8+7)).
    let region_px_w = xsize_blocks * 8;
    let region_px_h = ysize_blocks * 8;

    // ---- Stage 1: per-pixel diff, accumulated in 4-row groups, subsampled 4x.
    // `pre` has resolution (region_px_w/4) x (region_px_h/4) == (2*xblocks) x (2*yblocks).
    let pre_w = region_px_w / 4;
    let pre_h = region_px_h / 4;
    let mut pre = vec![0.0f32; pre_w * pre_h];

    // Accumulator for the current group of 4 rows, width region_px_w.
    let mut row_acc = vec![0.0f32; region_px_w];

    let clampx = |x: isize| -> usize { x.max(0).min(img_xsize as isize - 1) as usize };
    let clampy = |y: isize| -> usize { y.max(0).min(img_ysize as isize - 1) as usize };

    for ry in 0..region_px_h {
        let gy = y0 + ry;
        // Skip work for rows entirely outside the image (replicate edge instead).
        let gy_c = clampy(gy as isize);
        let gy1 = clampy(gy as isize - 1);
        let gy2 = clampy(gy as isize + 1);
        let row_y = opsin.plane_row(1, gy_c);
        let row_y1 = opsin.plane_row(1, gy1);
        let row_y2 = opsin.plane_row(1, gy2);
        let row_x = opsin.plane_row(0, gy_c);
        let row_x1 = opsin.plane_row(0, gy1);
        let row_x2 = opsin.plane_row(0, gy2);

        assert!(row_y.len() >= img_xsize);
        assert!(row_y1.len() >= img_xsize);
        assert!(row_y2.len() >= img_xsize);
        assert!(row_x.len() >= img_xsize);
        assert!(row_x1.len() >= img_xsize);
        assert!(row_x2.len() >= img_xsize);

        for rx in 0..region_px_w {
            let gx = x0 + rx;
            let gx_c = clampx(gx as isize);
            let gx1 = clampx(gx as isize - 1);
            let gx2 = clampx(gx as isize + 1);

            let in_y = row_y[gx_c];
            let base = 0.25 * (row_y2[gx_c] + row_y1[gx_c] + row_y[gx1] + row_y[gx2]);
            let gammac = ratio_cubic_to_simple_gamma(in_y + MATCH_GAMMA_OFFSET, false);
            let mut diff = gammac * (in_y - base);
            diff *= diff;

            let in_x = row_x[gx_c];
            let base_x = 0.25 * (row_x2[gx_c] + row_x1[gx_c] + row_x[gx1] + row_x[gx2]);
            let mut diff_x = gammac * (in_x - base_x);
            diff_x *= diff_x;
            diff += K_X_MUL * diff_x;
            diff = masking_sqrt(diff);

            if (ry & 3) != 0 {
                row_acc[rx] += diff;
            } else {
                row_acc[rx] = diff;
            }
        }

        // Every 4th row, downsample the accumulated 4-row band by 4 in x.
        if ry % 4 == 3 {
            let out_y = ry / 4;
            let prow = &mut pre[out_y * pre_w..out_y * pre_w + pre_w];
            for px in 0..pre_w {
                prow[px] = (row_acc[px * 4]
                    + row_acc[px * 4 + 1]
                    + row_acc[px * 4 + 2]
                    + row_acc[px * 4 + 3])
                    * 0.25;
            }
        }
    }

    // ---- Stage 2: FuzzyErosion (3x3 weighted-min pooling) over `pre`, then
    // downsample 2x into the block-resolution aq_map.
    // pre is (2*xblocks) x (2*yblocks); aq_map is xblocks x yblocks.
    let mut aq_map = vec![0.0f32; xsize_blocks * ysize_blocks];
    let k_mul = 0.05f32; // same weight for center and all four mins
    for fy in 0..pre_h {
        let ym1 = if fy >= 1 { fy - 1 } else { fy };
        let yp1 = if fy + 1 < pre_h { fy + 1 } else { fy };
        let rowt = &pre[ym1 * pre_w..ym1 * pre_w + pre_w];
        let row = &pre[fy * pre_w..fy * pre_w + pre_w];
        let rowb = &pre[yp1 * pre_w..yp1 * pre_w + pre_w];
        let out_y = fy / 2;
        for fx in 0..pre_w {
            let xm1 = if fx >= 1 { fx - 1 } else { fx };
            let xp1 = if fx + 1 < pre_w { fx + 1 } else { fx };
            // First four values, sorted ascending into mins[0..4].
            let mut mins = [row[fx], row[xm1], row[xp1], rowt[xm1]];
            mins.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Remaining five neighbours.
            store_min4(rowt[fx], &mut mins);
            store_min4(rowt[xp1], &mut mins);
            store_min4(rowb[xm1], &mut mins);
            store_min4(rowb[fx], &mut mins);
            store_min4(rowb[xp1], &mut mins);
            let v = k_mul * row[fx]
                + k_mul * mins[0]
                + k_mul * mins[1]
                + k_mul * mins[2]
                + k_mul * mins[3];
            let out_x = fx / 2;
            let idx = out_y * xsize_blocks + out_x;
            if fx % 2 == 0 && fy % 2 == 0 {
                aq_map[idx] = v;
            } else {
                aq_map[idx] += v;
            }
        }
    }

    // ---- Stage 3: PerBlockModulations + write integer quant field.
    let mut base_level = 0.5 * scale;
    let k_dampen_ramp_start = 7.0f32;
    let k_dampen_ramp_end = 14.0f32;
    let mut dampen = 1.0f32;
    if distance >= k_dampen_ramp_start {
        dampen =
            1.0 - ((distance - k_dampen_ramp_start) / (k_dampen_ramp_end - k_dampen_ramp_start));
        if dampen < 0.0 {
            dampen = 0.0;
        }
    }
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;
    let _ = &mut base_level;

    for by in 0..ysize_blocks {
        let py = y0 + by * 8;
        // If the block's pixels are completely outside the image, leave field=1.
        for bx in 0..xsize_blocks {
            let px = x0 + bx * 8;
            // Pixels read by modulations are clamped to image edges below; but
            // libjxl pads to whole blocks. Our opsin is image-sized, so guard.
            if px >= img_xsize || py >= img_ysize {
                raw_quant_field.row_mut(by)[bx] = 1;
                continue;
            }
            // For blocks touching the right/bottom edge, libjxl operates on a
            // padded image; we clamp reads to valid rows/cols by using bounded
            // block coordinates. The modulation funcs index up to +7; clamp.
            let bx_px = px.min(img_xsize.saturating_sub(8));
            let by_px = py.min(img_ysize.saturating_sub(8));

            let mut out_val = aq_map[by * xsize_blocks + bx];
            out_val = compute_mask(out_val);
            out_val = hf_modulation(bx_px, by_px, opsin, out_val);
            out_val = color_modulation(bx_px, by_px, opsin, distance, out_val);
            out_val = gamma_modulation(bx_px, by_px, opsin, out_val);
            // Multiplicative field: exponent modulation done above.
            let qf = fast_pow2(out_val * 1.442695041) * mul + add;
            let qi = (qf * inv_scale + 0.5) as i32;
            raw_quant_field.row_mut(by)[bx] = qi.clamp(1, 255) as u8;
        }
    }
}

/// 2^x. std f32 powi/exp is fine here (not perf-critical vs DCT); use exp2.
#[inline]
fn fast_pow2(x: f32) -> f32 {
    x.exp2()
}
