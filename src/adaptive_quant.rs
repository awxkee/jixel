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

use crate::dct::fmla;
use crate::image::{Image3F, ImageB};
use std::sync::OnceLock;

const K_SG_MUL: f32 = 226.77216153508914;
const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
const K_LOG2: f32 = 0.693147181;
const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_LOG2;
const K_SG_V_OFFSET: f32 = 7.7825991679894591;

pub(crate) const K_AC_QUANT: f32 = 0.8294;
const MATCH_GAMMA_OFFSET: f32 = 0.019;

/// Ratio of derivatives of cubic-root to simple-gamma; moves quantization from
/// jxl's opsin (cube-root-of-photons) space to butteraugli's log-gamma space.
#[inline]
pub(crate) fn ratio_cubic_to_simple_gamma<const INVERT: bool>(v: f32) -> f32 {
    let k_epsilon = 1e-2f32;
    let v = if v < 0.0 { 0.0 } else { v };
    let k_num_mul = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    let k_v_offset = K_SG_V_OFFSET * K_LOG2 + k_epsilon;
    let k_den_mul = K_LOG2 * K_SG_MUL;
    let v2 = v * v;
    let num = k_num_mul * v2 + k_epsilon;
    let den = (k_den_mul * v) * v2 + k_v_offset;
    if INVERT { num / den } else { den / num }
}

/// Visual-masking transform applied to the accumulated per-block difference.
#[inline]
pub(crate) fn compute_mask(out_val: f32) -> f32 {
    let k_base = -0.7647f32;
    let k_mul4 = 9.4708735624378946f32;
    let k_mul2 = 17.35036561631863f32;
    let k_offset2 = 302.59587815579727f32;
    let k_mul3 = 6.7943250517376494f32;
    let k_offset3 = 3.7179635626140772f32;
    let k_offset4 = 0.25f32 * k_offset3;
    let k_mul0 = 0.80061762862741759f32;

    let v1 = (out_val * k_mul0).max(1e-3);
    let v2 = 1.0 / (v1 + k_offset2);
    let v3 = 1.0 / (v1 * v1 + k_offset3);
    let v4 = 1.0 / (v1 * v1 + k_offset4);
    fmla(k_mul4, v4, fmla(k_mul2, v2, fmla(k_mul3, v3, k_base)))
}

/// `0.25 * sqrt(v * sqrt(kMul*1e8) + kLogOffset)`.
#[inline]
pub(crate) fn masking_sqrt(v: f32) -> f32 {
    let k_log_offset = 27.505837037000106f32;
    let k_mul = 211.66567973503678f32;
    let mul_v = k_mul * 1e8;
    0.25 * fmla(v, mul_v.sqrt(), k_log_offset).sqrt()
}

/// HF modulation (current libjxl): per-difference clamp at `valmin_y`, summed
/// over the 8x8 Y block, then `sum * kMul_y + kOffset` added to `out_val`.
pub(crate) fn hf_modulation_strength(distance: f32) -> f32 {
    (0.5 * distance).clamp(0.1, 1.0)
}

const AQ_PEAK: f32 = 0.8;
const AQ_FADE_START: f32 = 1.0;
const AQ_FADE_END: f32 = 1.75;

pub(crate) fn aq_dampen(distance: f32) -> f32 {
    if distance < AQ_FADE_START {
        return AQ_PEAK;
    }
    let t = (distance - AQ_FADE_START) / (AQ_FADE_END - AQ_FADE_START);
    (AQ_PEAK * (1.0 - t)).clamp(0.0, 1.0)
}

pub(crate) fn hf_modulation(
    x: usize,
    y: usize,
    xyb_y: &Image3F,
    out_val: f32,
    strength: f32,
) -> f32 {
    let ys = xyb_y.ysize();
    let xs = xyb_y.xsize();
    let valmin_y = 0.0206f32;
    let mut sum_y = 0.0f32;
    for dy in 0..8 {
        let row = xyb_y.plane_row(1, (y + dy).min(ys - 1));
        let row_next = if dy == 7 {
            row
        } else {
            xyb_y.plane_row(1, (y + dy + 1).min(ys - 1))
        };
        for dx in 0..8 {
            let c0 = (x + dx).min(xs - 1);
            let p = row[c0];
            // right difference, skipping the last column (matches kMaskRight)
            if dx < 7 {
                sum_y += valmin_y.min((p - row[(x + dx + 1).min(xs - 1)]).abs());
            }
            sum_y += valmin_y.min((p - row_next[c0]).abs());
        }
    }
    let k_mul_y = -0.38 * strength;
    let k_offset = 0.42f32; // higher value -> more bpp
    fmla(sum_y, k_mul_y, k_offset) + out_val
}

/// Blue modulation (current libjxl): increase precision in 8x8 blocks with
/// significant blue content that is not near-solid blue. Combined into the
/// field via `min`, not addition.
pub(crate) fn blue_modulation(x: usize, y: usize, xyb: &Image3F, out_val: f32) -> f32 {
    let ys = xyb.ysize();
    let xs = xyb.xsize();
    let k_limit = 0.010474084867598155f32;
    let k_offset = 0.0031994768654636393f32;
    let mut sum = 0.0f32;
    for dy in 0..8 {
        let ry = (y + dy).min(ys - 1);
        let row_x = xyb.plane_row(0, ry);
        let row_y = xyb.plane_row(1, ry);
        let row_b = xyb.plane_row(2, ry);
        for dx in 0..8 {
            let cx = (x + dx).min(xs - 1);
            let p_x = row_x[cx];
            let p_b = row_b[cx];
            let p_y_effective = row_y[cx] + k_offset + p_x.abs();
            if p_b > p_y_effective {
                sum += (p_b - p_y_effective).min(k_limit);
            }
        }
    }
    let k_mul = 0.90590804735610064f32;
    let mut scalar_sum = sum;
    // If it is all blue, don't boost the quantization (avoid over-spending on sky).
    if scalar_sum >= 32.0 * k_limit {
        scalar_sum = 64.0 * k_limit - scalar_sum;
    }
    let k_max_limit = 15.463398341612438f32;
    if scalar_sum >= k_max_limit * k_limit {
        scalar_sum = k_max_limit * k_limit;
    }
    scalar_sum *= k_mul;
    scalar_sum + out_val
}
/// Gamma modulation: accounts for the local gamma of the 8x8 block.
pub(crate) fn gamma_modulation(x: usize, y: usize, xyb: &Image3F, out_val: f32) -> f32 {
    let k_bias = 0.16f32;
    let mut overall_ratio = 0.0f32;
    let ys = xyb.ysize();
    let xs = xyb.xsize();
    for dy in 0..8 {
        let ry = (y + dy).min(ys - 1);
        let row_x = xyb.plane_row(0, ry);
        let row_y = xyb.plane_row(1, ry);
        for dx in 0..8 {
            let cx = (x + dx).min(xs - 1);
            let iny = row_y[cx] + k_bias;
            let inx = row_x[cx];
            let r = iny - inx;
            let g = iny + inx;
            let ratio_r = ratio_cubic_to_simple_gamma::<true>(r);
            let ratio_g = ratio_cubic_to_simple_gamma::<true>(g);
            overall_ratio = fmla(0.5, ratio_r + ratio_g, overall_ratio);
        }
    }
    overall_ratio *= 1.0 / 64.0;
    // ideally -1.0, but optimal correction adds some entropy, so slightly less.
    let k_gam = 0.1005613337192697f32;
    fmla(k_gam, dirty_log2f(overall_ratio), out_val)
}

#[inline]
pub(crate) fn store_min4(v: f32, mins: &mut [f32; 4]) {
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

#[inline(always)]
pub(crate) fn sort4(a: &mut [f32; 4]) {
    // Optimal 5-comparison sorting network for 4 elements
    macro_rules! cswap {
        ($i:expr, $j:expr) => {
            if a[$i] > a[$j] {
                a.swap($i, $j);
            }
        };
    }
    cswap!(0, 1);
    cswap!(2, 3);
    cswap!(0, 2);
    cswap!(1, 3);
    cswap!(1, 2);
}

pub(crate) type FillQuantFieldFn =
    fn(&mut AqMapScratch, &Image3F, &mut ImageB, usize, usize, f32, f32);

fn select_fill_quant_field_fn() -> FillQuantFieldFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return |scratch, opsin, raw_quant_field, x0, y0, distance, inv_scale| unsafe {
            crate::avx::fill_quant_field(
                scratch,
                opsin,
                raw_quant_field,
                x0,
                y0,
                distance,
                inv_scale,
            );
        };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |scratch, opsin, raw_quant_field, x0, y0, distance, inv_scale| unsafe {
            crate::sse::fill_quant_field(
                scratch,
                opsin,
                raw_quant_field,
                x0,
                y0,
                distance,
                inv_scale,
            );
        };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |scratch, opsin, raw_quant_field, x0, y0, distance, inv_scale| unsafe {
            crate::neon::fill_quant_field(
                scratch,
                opsin,
                raw_quant_field,
                x0,
                y0,
                distance,
                inv_scale,
            );
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::fill_quant_field
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        fill_quant_field_scalar
    }
}

static FILL_QUANT_FIELD_FN: OnceLock<FillQuantFieldFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_fill_quant_field_fn() -> FillQuantFieldFn {
    *FILL_QUANT_FIELD_FN.get_or_init(select_fill_quant_field_fn)
}

#[derive(Default)]
pub(crate) struct AqMapScratch {
    pub(crate) aq_map: Vec<f32>,
    pub(crate) secondary: Vec<f32>,
}

#[allow(unused)]
fn fill_quant_field_scalar(
    scratch: &mut AqMapScratch,
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

    // Accumulator for the current group of 4 rows, width region_px_w.
    let total_secondary = region_px_w + pre_w * pre_h;
    if scratch.secondary.len() < total_secondary {
        scratch.secondary.resize(total_secondary, 0.);
    }
    let borrowed_secondary = &mut scratch.secondary[..total_secondary];
    let (row_acc, pre) = borrowed_secondary.split_at_mut(region_px_w);

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
        assert!(row_y.len() >= img_xsize);
        assert!(row_y1.len() >= img_xsize);
        assert!(row_y2.len() >= img_xsize);

        for rx in 0..region_px_w {
            let gx = x0 + rx;
            let gx_c = clampx(gx as isize);
            let gx1 = clampx(gx as isize - 1);
            let gx2 = clampx(gx as isize + 1);

            let in_y = row_y[gx_c];
            let base = 0.25 * (row_y2[gx_c] + row_y1[gx_c] + row_y[gx1] + row_y[gx2]);
            let gammac = ratio_cubic_to_simple_gamma::<false>(in_y + MATCH_GAMMA_OFFSET);
            let mut diff = gammac * (in_y - base);
            diff *= diff;
            // current libjxl: clamp the squared luma diff before masking.
            if diff >= 0.2 {
                diff = 0.2;
            }
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
    if scratch.aq_map.len() < xsize_blocks * ysize_blocks {
        scratch.aq_map.resize(xsize_blocks * ysize_blocks, 0.);
    }
    let aq_map = &mut scratch.aq_map[..xsize_blocks * ysize_blocks];
    // FuzzyErosion weights (current libjxl): four smallest neighbors, weighted,
    // butteraugli-target dependent, normalized to kTotal.
    let fe_mul = if distance < 2.0 {
        (2.0 - distance) * 0.5
    } else {
        0.0
    };
    let fe_base = [0.125f32, 0.1, 0.09, 0.06];
    let fe_add = [0.0f32, -0.1, -0.09, -0.06];
    let mut k_mul = [0.0f32; 4];
    let mut norm_sum = 0.0f32;
    for i in 0..4 {
        k_mul[i] = fe_base[i] + fe_mul * fe_add[i];
        norm_sum += k_mul[i];
    }
    let k_total = 0.29959705784054957f32;
    for w in &mut k_mul {
        *w *= k_total / norm_sum;
    }
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
            sort4(&mut mins);
            // Remaining five neighbors.
            store_min4(rowt[fx], &mut mins);
            store_min4(rowt[xp1], &mut mins);
            store_min4(rowb[xm1], &mut mins);
            store_min4(rowb[fx], &mut mins);
            store_min4(rowb[xp1], &mut mins);
            let v =
                k_mul[0] * mins[0] + k_mul[1] * mins[1] + k_mul[2] * mins[2] + k_mul[3] * mins[3];
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
    let base_level = 0.48 * scale;
    let dampen = aq_dampen(distance);
    let mul = scale * dampen;
    let add = (1.0 - dampen) * base_level;

    for by in 0..ysize_blocks {
        let py = y0 + by * 8;
        let aq_row = &aq_map[by * xsize_blocks..by * xsize_blocks + xsize_blocks];
        let qf_row = raw_quant_field.row_mut(by);
        // If the block's pixels are completely outside the image, leave field=1.
        for (bx, (qf_out, &aq)) in qf_row.iter_mut().zip(aq_row.iter()).enumerate() {
            let px = x0 + bx * 8;
            if px >= img_xsize || py >= img_ysize {
                *qf_out = 1;
                continue;
            }

            let bx_px = px.min(img_xsize.saturating_sub(8));
            let by_px = py.min(img_ysize.saturating_sub(8));

            let mask_val = compute_mask(aq);
            let mask_val = gamma_modulation(bx_px, by_px, opsin, mask_val);
            let out_val = hf_modulation(
                bx_px,
                by_px,
                opsin,
                mask_val,
                hf_modulation_strength(distance),
            );
            let out_val = out_val.min(blue_modulation(bx_px, by_px, opsin, mask_val));
            // Multiplicative field: exponent modulation done above.
            let qf = fast_exp2(out_val * 1.442695041) * mul + add;
            let qi = fmla(qf, inv_scale, 0.5) as i32;
            *qf_out = qi.clamp(1, 255) as u8;
        }
    }
}

const TBLSIZE: usize = 64;

#[repr(align(64))]
struct Exp2Table([(u32, u32); TBLSIZE]);

#[rustfmt::skip]
static EXP2FT: Exp2Table = Exp2Table([(0x3F3504F3, 0xB2D4175E),(0x3F36FD92, 0x3268D5EF),(0x3F38FBAF, 0xB30E8719),(0x3F3AFF5B, 0x3319E7DA),(0x3F3D08A4, 0x333CD82F),(0x3F3F179A, 0x330E1902),(0x3F412C4D, 0x32CCF4D7),(0x3F4346CD, 0x328F330E),(0x3F45672A, 0xB201B5B7),(0x3F478D75, 0x32CCCE34),(0x3F49B9BE, 0x335E937C),(0x3F4BEC15, 0x2FF41909),(0x3F4E248C, 0xB21760EA),(0x3F506334, 0x3283628B),(0x3F52A81E, 0x3340F500),(0x3F54F35B, 0x331202BD),(0x3F5744FD, 0x32B66A3E),(0x3F599D16, 0x32D0D9B1),(0x3F5BFBB8, 0x332ED93F),(0x3F5E60F5, 0x3350A709),(0x3F60CCDF, 0x32025744),(0x3F633F89, 0xB33A7C4D),(0x3F65B907, 0x321DA4E9),(0x3F68396A, 0xB2FF36A7),(0x3F6AC0C7, 0x3217E40E),(0x3F6D4F30, 0xB2400CBB),(0x3F6FE4BA, 0x331A2ACC),(0x3F728177, 0xB2B7D3E5),(0x3F75257D, 0xB1FED2BE),(0x3F77D0DF, 0xB32B73BA),(0x3F7A83B3, 0x32579081),(0x3F7D3E0C, 0xB19726B5),(0x3F800000, 0x00000000),(0x3F8164D2, 0x320C09FB),(0x3F82CD87, 0x3391E031),(0x3F843A29, 0x33287EEF),(0x3F85AAC3, 0xB38F6665),(0x3F871F62, 0x339004AB),(0x3F88980F, 0x33AC4561),(0x3F8A14D5, 0xB39CDAEA),(0x3F8B95C2, 0x32949D5C),(0x3F8D1ADF, 0xB36F79FA),(0x3F8EA43A, 0x33971DC2),(0x3F9031DC, 0xB32BD022),(0x3F91C3D3, 0xB3928952),(0x3F935A2B, 0xB2EBFECF),(0x3F94F4F0, 0x3357B8BB),(0x3F96942D, 0xB307353B),(0x3F9837F0, 0xB345DFE9),(0x3F99E046, 0x3382A804),(0x3F9B8D3A, 0x3326993E),(0x3F9D3EDA, 0x3350A029),(0x3F9EF532, 0xB3605F62),(0x3FA0B051, 0xB210909B),(0x3FA27043, 0xB0DDC369),(0x3FA43516, 0x33385844),(0x3FA5FED7, 0x33400757),(0x3FA7CD94, 0x3325446E),(0x3FA9A15B, 0x33237A50),(0x3FAB7A3A, 0x33201CA4),(0x3FAD583F, 0x32394687),(0x3FAF3B79, 0x332E1225),(0x3FB123F6, 0x33838969),(0x3FB311C4, 0xB219F2BA)]);

pub(crate) fn fast_exp2(d: f32) -> f32 {
    let redux = f32::from_bits(0x4b400000) / TBLSIZE as f32;

    let ui = f32::to_bits(d + redux);
    let mut i0 = ui;
    i0 = i0.wrapping_add(TBLSIZE as u32 / 2);
    let k = i0 / TBLSIZE as u32;
    i0 &= TBLSIZE as u32 - 1;
    let mut uf = f32::from_bits(ui);
    uf -= redux;

    let item = EXP2FT.0[i0 as usize];
    let z0: f32 = f32::from_bits(item.0);

    let f: f32 = d - uf;

    let mut u = 0.24022650695908768;
    u = fmla(u, f, 0.69314718055994973);
    u *= f;

    #[inline]
    fn pow2if(q: i32) -> f32 {
        f32::from_bits((q.wrapping_add(0x7f) as u32) << 23)
    }

    let i2 = pow2if(k as i32);
    fmla(u, z0, z0) * i2
}

pub(crate) fn dirty_log2f(d: f32) -> f32 {
    let mut ix = d.to_bits();
    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    ix = ix.wrapping_add(0x3f800000 - 0x3f3504f3);
    let n = (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007fffff).wrapping_add(0x3f3504f3);
    let a = f32::from_bits(ix);

    let x = (a - 1.) / (a + 1.);

    let x2 = x * x;
    let mut u = 0.4121985850084821691e+0;
    u = fmla(u, x2, 0.5770780163490337802e+0);
    u = fmla(u, x2, 0.9617966939259845749e+0);
    fmla(x2 * x, u, fmla(x, 0.2885390081777926802e+1, n as f32))
}

#[inline]
pub(crate) fn dirty_log1pf(d: f32) -> f32 {
    const LN_2: f32 = 0.6931471805599453;
    let ix = (1.0 + d).to_bits();
    let exponent = ix & 0x7f80_0000;
    let n = (exponent >> 23) as i32 - 0x7f;

    // Replacing the exponent with 127 normalizes 1+d to 1+t without division.
    // For n=0 use d itself so tiny inputs do not lose precision in (1+d)-1.
    let mantissa = f32::from_bits((ix & 0x007f_ffff) | 0x3f80_0000);
    let t = if n == 0 { d } else { mantissa - 1.0 };

    // Direct minimax ln(1+t) ~= t*P(t), t in [0, 1]. See tools/log1p.sollya.
    let mut p = 0.014539075084030628;
    p = fmla(p, t, -0.0675969123840332);
    p = fmla(p, t, 0.15056970715522766);
    p = fmla(p, t, -0.23573730885982513);
    p = fmla(p, t, 0.33125850558280945);
    p = fmla(p, t, -0.4998837411403656);
    p = fmla(p, t, 0.999998927116394);
    fmla(n as f32, LN_2, t * p)
}

#[cfg(test)]
mod tests {
    use super::{
        AQ_FADE_END, AQ_FADE_START, AQ_PEAK, AqMapScratch, Image3F, ImageB, aq_dampen,
        dirty_log1pf, dirty_log2f, fill_quant_field_scalar, hf_modulation_strength,
        selected_fill_quant_field_fn,
    };

    #[test]
    fn aq_dampen_holds_then_fades_to_a_flat_field() {
        for d in [0.0, 0.1, 0.5, 0.99] {
            assert_eq!(
                aq_dampen(d),
                AQ_PEAK,
                "AQ should be at full strength at d={d}"
            );
        }
        assert_eq!(aq_dampen(AQ_FADE_START), AQ_PEAK);
        // Fully faded at and beyond the end of the ramp — never negative, and
        // never re-activating at the very coarse distances.
        for d in [AQ_FADE_END, 2.0, 4.0, 10.0, 25.0] {
            assert_eq!(aq_dampen(d), 0.0, "AQ should be off at d={d}");
        }
        // Monotonically non-increasing and in range across the whole domain.
        let mut prev = f32::INFINITY;
        for i in 0..=200 {
            let d = i as f32 * 0.05;
            let v = aq_dampen(d);
            assert!(v <= prev + 1e-7, "aq_dampen rose at d={d}");
            assert!(
                (0.0..=AQ_PEAK).contains(&v),
                "aq_dampen out of range at d={d}"
            );
            prev = v;
        }
        // Halfway through the ramp is half strength.
        let mid = 0.5 * (AQ_FADE_START + AQ_FADE_END);
        assert!((aq_dampen(mid) - 0.5 * AQ_PEAK).abs() < 1e-6);
    }

    /// The SIMD backends take a wide vector path only for groups of blocks that
    /// are fully interior. Images narrower than one such group (e.g. 27x28 for
    /// the 4-block/33-column NEON and WASM gates) must fall through to the
    /// scalar tail instead of reading past the end of a row.
    #[test]
    fn fill_quant_field_matches_scalar_on_small_images() {
        for &(xsize, ysize) in &[
            (27usize, 28usize),
            (1, 1),
            (7, 9),
            (32, 32),
            (33, 33),
            (64, 17),
            (65, 65),
        ] {
            let mut opsin = Image3F::new(xsize, ysize);
            for c in 0..3 {
                for y in 0..ysize {
                    let row = opsin.plane_row_mut(c, y);
                    for (x, v) in row.iter_mut().enumerate() {
                        // Deterministic, non-flat content so the modulations
                        // actually differ block to block.
                        let t = (x * 7 + y * 13 + c * 29) as f32;
                        *v = (t * 0.017).sin() * 0.4 + c as f32 * 0.05;
                    }
                }
            }

            let xsize_blocks = xsize.div_ceil(8);
            let ysize_blocks = ysize.div_ceil(8);

            let mut got = ImageB::new(xsize_blocks, ysize_blocks);
            let mut want = ImageB::new(xsize_blocks, ysize_blocks);
            let mut got_scratch = AqMapScratch::default();
            let mut want_scratch = AqMapScratch::default();

            selected_fill_quant_field_fn()(&mut got_scratch, &opsin, &mut got, 0, 0, 1.0, 1.0);
            fill_quant_field_scalar(&mut want_scratch, &opsin, &mut want, 0, 0, 1.0, 1.0);

            for by in 0..ysize_blocks {
                assert_eq!(got.row(by), want.row(by), "{xsize}x{ysize}, block row {by}");
            }
        }
    }

    #[test]
    fn dirty_log2f_matches_log2() {
        for exponent in -20..=20 {
            let x = 2.0f32.powi(exponent);
            assert_eq!(dirty_log2f(x), exponent as f32, "log2({x})");
        }

        let mut max_abs = 0.0f32;
        let mut worst_x = 0.0f32;
        let mut x = 1.0e-6f32;
        while x <= 1.0e6 {
            let error = (dirty_log2f(x) - x.log2()).abs();
            if error > max_abs {
                max_abs = error;
                worst_x = x;
            }
            x *= 1.01;
        }
        assert!(
            max_abs < 2.0e-6,
            "max absolute error {max_abs} at x={worst_x}"
        );
    }

    #[test]
    fn dirty_log1pf_matches_ln_1p() {
        assert_eq!(dirty_log1pf(0.0), 0.0);
        let mut max_abs = 0f32;
        let mut max_rel = 0f32; // measured only for d >= 0.1
        let mut x = 1e-5f32;
        while x <= 5.0e6 {
            let got = dirty_log1pf(x);
            let want = x.ln_1p();
            max_abs = max_abs.max((got - want).abs());
            if x >= 0.1 {
                max_rel = max_rel.max((got - want).abs() / want.abs());
            }
            x *= 1.05;
        }
        assert!(max_abs < 1e-5, "max absolute error {max_abs} too large");
        assert!(max_rel < 1e-5, "max relative error {max_rel} too large");
        // A few spot checks against std ln_1p.
        for &v in &[0.5f32, 1.0, 42.0, 256.0, 1.0e6] {
            let rel = (dirty_log1pf(v) - v.ln_1p()).abs() / v.ln_1p();
            assert!(rel < 1e-5, "ln_1p({v}) rel err {rel}");
        }
    }

    #[test]
    fn hf_modulation_strength_protects_high_quality_detail() {
        assert_eq!(hf_modulation_strength(0.1), 0.1);
        assert_eq!(hf_modulation_strength(0.3), 0.15);
        assert_eq!(hf_modulation_strength(0.5), 0.25);
        assert_eq!(hf_modulation_strength(1.0), 0.5);
        assert_eq!(hf_modulation_strength(2.0), 1.0);
        assert_eq!(hf_modulation_strength(6.0), 1.0);
    }
}
