/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

use std::arch::x86_64::*;

const MATCH_GAMMA_OFFSET: f32 = 0.019;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn load8s(s: &[f32], i: usize) -> __m256 {
    assert!(s.len() >= 8);
    _mm256_setr_ps(
        s[i],
        s[i + 1],
        s[i + 2],
        s[i + 3],
        s[i + 4],
        s[i + 5],
        s[i + 6],
        s[i + 7],
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn store8(v: __m256, s: &mut [f32], i: usize) {
    assert!(s.len() >= 8);
    unsafe {
        _mm256_storeu_ps(s[i..].as_mut_ptr(), v);
    }
}

/// Horizontal sum of the 8 lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
fn hsum(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let mut shuf = _mm_movehdup_ps(s);
    let mut sums = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
fn abs_ps(v: __m256) -> __m256 {
    _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), v)
}

/// `a * b + c`, true fused multiply-add.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn _mm256_mlaf_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(a, b, c)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
fn _mm256_neg_epi32(n: __m256i) -> __m256i {
    _mm256_sub_epi32(_mm256_setzero_si256(), n)
}

/// Vectorised `ratio_cubic_to_simple_gamma`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn ratio_cubic_x8(v: __m256, invert: bool) -> __m256 {
    const K_SG_MUL: f32 = 226.77216153508914;
    const K_SG_MUL2: f32 = 1.0 / 73.377132366608819;
    const K_LOG2: f32 = 0.693147181;
    const K_SG_RET_MUL: f32 = K_SG_MUL2 * 18.6580932135 * K_LOG2;
    const K_SG_V_OFFSET: f32 = 7.7825991679894591;
    let k_epsilon = 1e-2f32;
    let k_num_mul = K_SG_RET_MUL * 3.0 * K_SG_MUL;
    let k_v_offset = K_SG_V_OFFSET * K_LOG2 + k_epsilon;
    let k_den_mul = K_LOG2 * K_SG_MUL;

    let v = _mm256_max_ps(v, _mm256_setzero_ps());
    let v2 = _mm256_mul_ps(v, v);
    let num = _mm256_mlaf_ps(_mm256_set1_ps(k_num_mul), v2, _mm256_set1_ps(k_epsilon));
    let den = _mm256_mlaf_ps(
        _mm256_mul_ps(_mm256_set1_ps(k_den_mul), v),
        v2,
        _mm256_set1_ps(k_v_offset),
    );
    if invert {
        _mm256_div_ps(num, den)
    } else {
        _mm256_div_ps(den, num)
    }
}

/// Vectorised `masking_sqrt`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn masking_sqrt_x8(v: __m256) -> __m256 {
    let k_log_offset = 27.505837037000106f32;
    let k_mul = 211.66567973503678f32;
    let mul_v = (k_mul * 1e8f32).sqrt();
    let inner = _mm256_mlaf_ps(v, _mm256_set1_ps(mul_v), _mm256_set1_ps(k_log_offset));
    _mm256_mul_ps(_mm256_set1_ps(0.25), _mm256_sqrt_ps(inner))
}

/// Fill `row_acc[0..region_px_w]` with this row's masking diff. Y-clamped row
/// slices come from the caller; we only split on x. Interior columns go 8-wide;
/// first/last/over-edge columns use the scalar formula.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
fn stage1_diff_row(
    set_mode: bool,
    row_y: &[f32],
    row_y1: &[f32],
    row_y2: &[f32],
    x0: usize,
    img_xsize: usize,
    region_px_w: usize,
    row_acc: &mut [f32],
) {
    let offset = _mm256_set1_ps(MATCH_GAMMA_OFFSET);
    let quarter = _mm256_set1_ps(0.25);
    let limit = _mm256_set1_ps(0.2);

    let lo_gx = 1isize;
    let hi_gx = img_xsize as isize - 9;
    let rx_lo = (lo_gx - x0 as isize).max(0) as usize;
    let rx_hi = if hi_gx >= x0 as isize {
        ((hi_gx - x0 as isize) as usize + 1).min(region_px_w)
    } else {
        0
    };

    let head_end = rx_lo.min(region_px_w);
    for rx in 0..head_end {
        scalar_pixel(set_mode, row_y, row_y1, row_y2, x0, img_xsize, rx, row_acc);
    }

    let mut rx = rx_lo;
    while rx + 8 <= rx_hi {
        let gx = x0 + rx;
        let cy = load8s(row_y, gx);
        let ly = load8s(row_y, gx - 1);
        let ry_ = load8s(row_y, gx + 1);
        let uy = load8s(row_y1, gx);
        let dy = load8s(row_y2, gx);
        let base_y = _mm256_mul_ps(
            quarter,
            _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(dy, uy), ly), ry_),
        );
        let gammac = ratio_cubic_x8(_mm256_add_ps(cy, offset), false);
        let dyv = _mm256_mul_ps(gammac, _mm256_sub_ps(cy, base_y));
        let mut diff = _mm256_mul_ps(dyv, dyv);
        // current libjxl: clamp the squared luma diff before masking.
        diff = _mm256_min_ps(diff, limit);
        diff = masking_sqrt_x8(diff);

        if set_mode {
            store8(diff, row_acc, rx);
        } else {
            store8(_mm256_add_ps(load8s(row_acc, rx), diff), row_acc, rx);
        }
        rx += 8;
    }

    for rx in rx..region_px_w {
        scalar_pixel(set_mode, row_y, row_y1, row_y2, x0, img_xsize, rx, row_acc);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn scalar_pixel(
    set_mode: bool,
    row_y: &[f32],
    row_y1: &[f32],
    row_y2: &[f32],
    x0: usize,
    img_xsize: usize,
    rx: usize,
    row_acc: &mut [f32],
) {
    let clampx = |x: isize| -> usize { x.max(0).min(img_xsize as isize - 1) as usize };
    let gx = x0 + rx;
    let gx_c = clampx(gx as isize);
    let gx1 = clampx(gx as isize - 1);
    let gx2 = clampx(gx as isize + 1);
    let in_y = row_y[gx_c];
    let base = 0.25 * (row_y2[gx_c] + row_y1[gx_c] + row_y[gx1] + row_y[gx2]);
    let gammac =
        crate::adaptive_quant::ratio_cubic_to_simple_gamma(in_y + MATCH_GAMMA_OFFSET, false);
    let mut diff = gammac * (in_y - base);
    diff *= diff;
    if diff >= 0.2 {
        diff = 0.2;
    }
    diff = crate::adaptive_quant::masking_sqrt(diff);
    if !set_mode {
        row_acc[rx] += diff;
    } else {
        row_acc[rx] = diff;
    }
}

/// Branchless "insert v, keep the 4 smallest, sorted ascending" — per-lane
/// min/max chain over 8 lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn insert8(m: &mut [__m256; 4], v: __m256) {
    let mut t = v;
    let n0 = _mm256_min_ps(m[0], t);
    t = _mm256_max_ps(m[0], t);
    m[0] = n0;
    let n1 = _mm256_min_ps(m[1], t);
    t = _mm256_max_ps(m[1], t);
    m[1] = n1;
    let n2 = _mm256_min_ps(m[2], t);
    t = _mm256_max_ps(m[2], t);
    m[2] = n2;
    m[3] = _mm256_min_ps(m[3], t);
}

/// Scalar fuzzy-erosion value for one `pre` pixel (row edges). Byte-identical to
/// the scalar Stage-2 inner body.
#[inline]
fn scalar_px(
    rowt: &[f32],
    row: &[f32],
    rowb: &[f32],
    pre_w: usize,
    fx: usize,
    kmul: &[f32; 4],
) -> f32 {
    let xm1 = if fx >= 1 { fx - 1 } else { fx };
    let xp1 = if fx + 1 < pre_w { fx + 1 } else { fx };
    let mut mins = [row[fx], row[xm1], row[xp1], rowt[xm1]];
    crate::adaptive_quant::sort4(&mut mins);
    crate::adaptive_quant::store_min4(rowt[fx], &mut mins);
    crate::adaptive_quant::store_min4(rowt[xp1], &mut mins);
    crate::adaptive_quant::store_min4(rowb[xm1], &mut mins);
    crate::adaptive_quant::store_min4(rowb[fx], &mut mins);
    crate::adaptive_quant::store_min4(rowb[xp1], &mut mins);
    kmul[0] * mins[0] + kmul[1] * mins[1] + kmul[2] * mins[2] + kmul[3] * mins[3]
}

/// Fill `vrow[0..pre_w]` with the FuzzyErosion value for one `pre` row.
#[target_feature(enable = "avx2,fma")]
fn fuzzy_erosion_row(
    rowt: &[f32],
    row: &[f32],
    rowb: &[f32],
    pre_w: usize,
    kmul: &[f32; 4],
    vrow: &mut [f32],
) {
    let k0 = _mm256_set1_ps(kmul[0]);
    let k1 = _mm256_set1_ps(kmul[1]);
    let k2 = _mm256_set1_ps(kmul[2]);
    let k3 = _mm256_set1_ps(kmul[3]);
    let inf = _mm256_set1_ps(f32::INFINITY);

    let mut fx = 0usize;
    if pre_w > 0 {
        vrow[0] = scalar_px(rowt, row, rowb, pre_w, 0, kmul);
        fx = 1;
    }
    // AVX2 interior: chunk covers output pixels fx..fx+7; loads span [fx-1, fx+8].
    while fx + 9 <= pre_w {
        let mut m = [inf, inf, inf, inf];
        insert8(&mut m, load8s(rowt, fx - 1));
        insert8(&mut m, load8s(rowt, fx));
        insert8(&mut m, load8s(rowt, fx + 1));
        insert8(&mut m, load8s(row, fx - 1));
        insert8(&mut m, load8s(row, fx));
        insert8(&mut m, load8s(row, fx + 1));
        insert8(&mut m, load8s(rowb, fx - 1));
        insert8(&mut m, load8s(rowb, fx));
        insert8(&mut m, load8s(rowb, fx + 1));
        // v = k0*m0 + k1*m1 + k2*m2 + k3*m3 (4 smallest, no center; scalar add order)
        let mut v = _mm256_mul_ps(k0, m[0]);
        v = _mm256_add_ps(v, _mm256_mul_ps(k1, m[1]));
        v = _mm256_add_ps(v, _mm256_mul_ps(k2, m[2]));
        v = _mm256_add_ps(v, _mm256_mul_ps(k3, m[3]));
        store8(v, vrow, fx);
        fx += 8;
    }
    while fx < pre_w {
        vrow[fx] = scalar_px(rowt, row, rowb, pre_w, fx, kmul);
        fx += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn fill_quant_field(
    opsin: &crate::image::Image3F,
    raw_quant_field: &mut crate::image::ImageB,
    x0: usize,
    y0: usize,
    distance: f32,
    inv_scale: f32,
) {
    let xsize_blocks = raw_quant_field.xsize();
    let ysize_blocks = raw_quant_field.ysize();
    let img_xsize = opsin.xsize();
    let img_ysize = opsin.ysize();

    let scale = crate::adaptive_quant::K_AC_QUANT / distance.powf(0.7934);

    let region_px_w = xsize_blocks * 8;
    let region_px_h = ysize_blocks * 8;

    // ---- Stage 1: per-pixel masking pre-pass.
    let pre_w = region_px_w / 4;
    let pre_h = region_px_h / 4;
    let mut pre = vec![0.0f32; pre_w * pre_h];
    let mut row_acc = vec![0.0f32; region_px_w];

    let clampy = |y: isize| -> usize { y.max(0).min(img_ysize as isize - 1) as usize };

    for ry in 0..region_px_h {
        let gy = y0 + ry;
        let gy_c = clampy(gy as isize);
        let gy1 = clampy(gy as isize - 1);
        let gy2 = clampy(gy as isize + 1);
        let row_y = opsin.plane_row(1, gy_c);
        let row_y1 = opsin.plane_row(1, gy1);
        let row_y2 = opsin.plane_row(1, gy2);

        let set_mode = (ry & 3) == 0;
        stage1_diff_row(
            set_mode,
            row_y,
            row_y1,
            row_y2,
            x0,
            img_xsize,
            region_px_w,
            &mut row_acc,
        );

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

    // ---- Stage 2: FuzzyErosion, then 2x downsample into block-resolution aq_map.
    // Weights (current libjxl): four smallest neighbours, butteraugli-dependent,
    // normalised to kTotal.
    let fe_mul = if distance < 2.0 {
        (2.0 - distance) * 0.5
    } else {
        0.0
    };
    let fe_base = [0.125f32, 0.1, 0.09, 0.06];
    let fe_add = [0.0f32, -0.1, -0.09, -0.06];
    let mut kmul = [0.0f32; 4];
    let mut norm_sum = 0.0f32;
    for i in 0..4 {
        kmul[i] = fe_base[i] + fe_mul * fe_add[i];
        norm_sum += kmul[i];
    }
    let k_total = 0.29959705784054957f32;
    for w in &mut kmul {
        *w *= k_total / norm_sum;
    }
    let mut aq_map = vec![0.0f32; xsize_blocks * ysize_blocks];
    let mut vrow = vec![0.0f32; pre_w];
    for fy in 0..pre_h {
        let ym1 = if fy >= 1 { fy - 1 } else { fy };
        let yp1 = if fy + 1 < pre_h { fy + 1 } else { fy };
        let rowt = &pre[ym1 * pre_w..ym1 * pre_w + pre_w];
        let row = &pre[fy * pre_w..fy * pre_w + pre_w];
        let rowb = &pre[yp1 * pre_w..yp1 * pre_w + pre_w];
        let out_y = fy / 2;

        fuzzy_erosion_row(rowt, row, rowb, pre_w, &kmul, &mut vrow);
        for fx in 0..pre_w {
            let out_x = fx / 2;
            let idx = out_y * xsize_blocks + out_x;
            if fx % 2 == 0 && fy % 2 == 0 {
                aq_map[idx] = vrow[fx];
            } else {
                aq_map[idx] += vrow[fx];
            }
        }
    }

    // ---- Stage 3: per-block modulations + integer quant field.
    let base_level = 0.48 * scale;
    let k_dampen_ramp_start = 2.0f32;
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

    for by in 0..ysize_blocks {
        let py = y0 + by * 8;
        let aq_row = &aq_map[by * xsize_blocks..by * xsize_blocks + xsize_blocks];
        let qf_row = raw_quant_field.row_mut(by);
        for (bx, (qf_out, &aq)) in qf_row.iter_mut().zip(aq_row.iter()).enumerate() {
            let px = x0 + bx * 8;
            if px >= img_xsize || py >= img_ysize {
                *qf_out = 1;
                continue;
            }
            let bx_px = px.min(img_xsize.saturating_sub(8));
            let by_px = py.min(img_ysize.saturating_sub(8));
            // Per-block modulations run through the (validated) scalar helpers:
            // mask -> gamma -> hf, then min with blue. This is block-resolution
            // work, not a hot path, so it need not be vectorised.
            let mask_val = crate::adaptive_quant::compute_mask(aq);
            let mask_val = crate::adaptive_quant::gamma_modulation(bx_px, by_px, opsin, mask_val);
            let out_val = crate::adaptive_quant::hf_modulation(bx_px, by_px, opsin, mask_val);
            let out_val = out_val.min(crate::adaptive_quant::blue_modulation(
                bx_px, by_px, opsin, mask_val,
            ));
            let qf = crate::adaptive_quant::fast_exp2(out_val * 1.442695041) * mul + add;
            let qi = f32::mul_add(qf, inv_scale, 0.5) as i32;
            *qf_out = qi.clamp(1, 255) as u8;
        }
    }
}
