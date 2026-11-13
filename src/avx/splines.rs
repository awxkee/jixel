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

//! AVX2/FMA spline kernels. Detector and decoder rendering arithmetic retains
//! the scalar order of operations.

use crate::splines::{
    ALONG_PENALTY, BlobWeightRow, FilterBounds, MAX_SUBPIXEL_OFFSET, Point, RidgeRow, Sample,
    Segments, ridge_row_scalar, vertical_row_scalar_tail,
};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn vertical_block<const VECTORS: usize>(
    src: &[f32],
    w: usize,
    bounds: FilterBounds,
    kernel: &[f32],
    x: usize,
    dst: &mut [[f32; 8]; VECTORS],
) {
    let mut sums = [_mm256_setzero_ps(); VECTORS];
    for (tap, &weight) in kernel.iter().enumerate() {
        let start = bounds.index(tap) * w + x;
        let weight = _mm256_set1_ps(weight);
        // SAFETY: the caller checked every source row contains this block.
        let pixels = unsafe { src.get_unchecked(start..start + 8 * VECTORS) };
        let (pixels, _) = pixels.as_chunks::<8>();
        for (sum, pixels) in sums.iter_mut().zip(pixels) {
            // SAFETY: each chunk contains eight readable pixels.
            let pixels = unsafe { _mm256_loadu_ps(pixels.as_ptr()) };
            *sum = _mm256_add_ps(*sum, _mm256_mul_ps(weight, pixels));
        }
    }
    for (block, sum) in dst.iter_mut().zip(sums) {
        // SAFETY: each chunk contains eight writable pixels.
        unsafe { _mm256_storeu_ps(block.as_mut_ptr(), sum) };
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn vertical_row_avx2(
    src: &[f32],
    w: usize,
    bounds: FilterBounds,
    kernel: &[f32],
    dst: &mut [f32],
) {
    if dst.is_empty() {
        return;
    }
    assert!(dst.len() <= w && bounds.index(kernel.len().saturating_sub(1)) < src.len() / w);
    let (blocks, tail) = dst.as_chunks_mut::<8>();
    let (pairs, blocks) = blocks.as_chunks_mut::<2>();
    let mut x = 0;
    for pair in pairs {
        // SAFETY: validated above, and each block contains 16 pixels.
        unsafe { vertical_block(src, w, bounds, kernel, x, pair) };
        x += 16;
    }
    for block in blocks {
        // SAFETY: validated above, and each block contains eight pixels.
        unsafe { vertical_block(src, w, bounds, kernel, x, std::array::from_mut(block)) };
        x += 8;
    }
    if !tail.is_empty() {
        vertical_row_scalar_tail(src, w, bounds, kernel, x, tail);
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn abs(x: __m256) -> __m256 {
    _mm256_andnot_ps(_mm256_set1_ps(-0.0), x)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn neg(x: __m256) -> __m256 {
    _mm256_xor_ps(x, _mm256_set1_ps(-0.0))
}

/// Pack eight masks or polarities, whose values are all in -1..=1.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn pack_bytes(x: __m256i) -> u64 {
    let x = _mm_packs_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256::<1>(x));
    _mm_cvtsi128_si64(_mm_packs_epi16(x, x)) as u64
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn spline_ridge_row_avx2(s: f32, derivatives: [&[f32]; 5], output: RidgeRow<'_>) {
    let n = output.validate(derivatives);
    let end = n / 8 * 8;
    let [hxx, hyy, hxy, gx, gy] = derivatives;
    let s2 = s * s;
    let zero = _mm256_setzero_ps();
    let eps = _mm256_set1_ps(1e-12);
    for i in (0..end).step_by(8) {
        // SAFETY: validate checks every slice; i..i+8 contains eight real
        // pixels. Output slices are disjoint mutable borrows.
        unsafe {
            let xx = _mm256_loadu_ps(hxx.as_ptr().add(i));
            let yy = _mm256_loadu_ps(hyy.as_ptr().add(i));
            let xy = _mm256_loadu_ps(hxy.as_ptr().add(i));
            let tr = _mm256_mul_ps(_mm256_add_ps(xx, yy), _mm256_set1_ps(0.5));
            let half_diff = _mm256_mul_ps(_mm256_sub_ps(xx, yy), _mm256_set1_ps(0.5));
            let df = _mm256_sqrt_ps(_mm256_add_ps(
                _mm256_mul_ps(half_diff, half_diff),
                _mm256_mul_ps(xy, xy),
            ));
            let la = _mm256_add_ps(tr, df);
            let lb = _mm256_sub_ps(tr, df);
            let a_larger = _mm256_cmp_ps::<_CMP_GT_OQ>(abs(la), abs(lb));
            let big = _mm256_blendv_ps(lb, la, a_larger);
            let small = _mm256_blendv_ps(la, lb, a_larger);
            let abs_big = abs(big);
            // MAXPS chooses its second operand for NaN, matching max(0.0).
            let strength = _mm256_mul_ps(
                _mm256_max_ps(
                    _mm256_sub_ps(
                        abs_big,
                        _mm256_mul_ps(abs(small), _mm256_set1_ps(ALONG_PENALTY)),
                    ),
                    zero,
                ),
                _mm256_set1_ps(s2),
            );
            let old_strength = _mm256_loadu_ps(output.strength.as_ptr().add(i));
            let mut accept = _mm256_cmp_ps::<_CMP_NLE_UQ>(strength, old_strength);
            if _mm256_movemask_ps(accept) == 0 {
                continue;
            }
            let gx = _mm256_loadu_ps(gx.as_ptr().add(i));
            let gy = _mm256_loadu_ps(gy.as_ptr().add(i));
            let gradient =
                _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(gx, gx), _mm256_mul_ps(gy, gy)));
            accept = _mm256_and_ps(
                accept,
                _mm256_cmp_ps::<_CMP_NGT_UQ>(
                    _mm256_mul_ps(gradient, _mm256_set1_ps(s)),
                    _mm256_mul_ps(abs_big, _mm256_set1_ps(s2)),
                ),
            );
            if _mm256_movemask_ps(accept) == 0 {
                continue;
            }
            let vy = _mm256_sub_ps(big, xx);
            let norm = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(xy, xy), _mm256_mul_ps(vy, vy)));
            let degenerate = _mm256_cmp_ps::<_CMP_LT_OQ>(norm, eps);
            let vx = _mm256_blendv_ps(_mm256_div_ps(xy, norm), _mm256_set1_ps(1.0), degenerate);
            let vy = _mm256_blendv_ps(_mm256_div_ps(vy, norm), zero, degenerate);
            let t = _mm256_div_ps(
                neg(_mm256_add_ps(_mm256_mul_ps(gx, vx), _mm256_mul_ps(gy, vy))),
                big,
            );
            let lower = _mm256_set1_ps(-MAX_SUBPIXEL_OFFSET);
            let upper = _mm256_set1_ps(MAX_SUBPIXEL_OFFSET);
            let t = _mm256_blendv_ps(t, lower, _mm256_cmp_ps::<_CMP_LT_OQ>(t, lower));
            let t = _mm256_blendv_ps(t, upper, _mm256_cmp_ps::<_CMP_GT_OQ>(t, upper));
            let t = _mm256_blendv_ps(t, zero, _mm256_cmp_ps::<_CMP_LT_OQ>(abs_big, eps));
            let store = |row: &mut [f32], value| {
                let ptr = row.as_mut_ptr().add(i);
                _mm256_storeu_ps(ptr, _mm256_blendv_ps(_mm256_loadu_ps(ptr), value, accept));
            };
            store(output.strength, strength);
            store(output.nx, vx);
            store(output.ny, vy);
            store(output.scale, _mm256_set1_ps(s));
            store(output.offset, t);
            let positive = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(big, zero));
            let polarity = pack_bytes(_mm256_or_si256(positive, _mm256_set1_epi32(1)));
            let mask = pack_bytes(_mm256_castps_si256(accept));
            // Exactly eight bytes, with no load or store past the row end.
            let ptr = output.polarity.as_mut_ptr().add(i).cast::<u64>();
            ptr.write_unaligned((ptr.read_unaligned() & !mask) | (polarity & mask));
        }
    }
    ridge_row_scalar(
        s,
        derivatives.map(|row| &row[end..]),
        RidgeRow {
            strength: &mut output.strength[end..],
            nx: &mut output.nx[end..],
            ny: &mut output.ny[end..],
            scale: &mut output.scale[end..],
            polarity: &mut output.polarity[end..],
            offset: &mut output.offset[end..],
        },
    );
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn fast_cos(x: __m256) -> __m256 {
    use std::f32::consts::{PI, SQRT_2};
    let pi2 = _mm256_set1_ps(PI * 2.0);
    let periods = _mm256_floor_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5 / PI)));
    let xmod = _mm256_sub_ps(x, _mm256_mul_ps(periods, pi2));
    let x_pi = _mm256_min_ps(xmod, _mm256_sub_ps(pi2, xmod));
    let above = _mm256_cmp_ps::<_CMP_GE_OQ>(x_pi, _mm256_set1_ps(PI / 2.0));
    let xh = _mm256_blendv_ps(x_pi, _mm256_sub_ps(_mm256_set1_ps(PI), x_pi), above);
    let xs = _mm256_mul_ps(xh, _mm256_set1_ps(0.25));
    let x2 = _mm256_mul_ps(xs, xs);
    let x4 = _mm256_mul_ps(x2, x2);
    let pre = _mm256_add_ps(
        _mm256_mul_ps(x4, _mm256_set1_ps(0.06960438)),
        _mm256_add_ps(
            _mm256_mul_ps(x2, _mm256_set1_ps(-0.84087373)),
            _mm256_set1_ps(1.68179268),
        ),
    );
    let s1 = _mm256_sub_ps(_mm256_mul_ps(pre, pre), _mm256_set1_ps(SQRT_2));
    let s2 = _mm256_sub_ps(_mm256_mul_ps(s1, s1), _mm256_set1_ps(1.0));
    _mm256_blendv_ps(s2, neg(s2), above)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn continuous_idct_avx2(dct: &[[f32; 4]; 32], t: f32) -> [f32; 4] {
    let mut result = _mm_setzero_ps();
    let mut indices = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    for rows in dct.as_chunks::<8>().0 {
        let args = _mm256_mul_ps(
            _mm256_mul_ps(indices, _mm256_set1_ps(std::f32::consts::PI / 32.0)),
            _mm256_set1_ps(t + 0.5),
        );
        let cos = fast_cos(args);
        for (i, pair) in rows.as_chunks::<2>().0.iter().enumerate() {
            let a = (i * 2) as i32;
            let b = a + 1;
            let cosines = _mm256_permutevar8x32_ps(cos, _mm256_setr_epi32(a, a, a, a, b, b, b, b));
            // SAFETY: two adjacent channel arrays contain exactly eight floats.
            let values = unsafe { _mm256_loadu_ps(pair.as_ptr().cast::<f32>()) };
            let product = _mm256_mul_ps(values, cosines);
            // Keep all 32 frequency additions in decoder order; reducing
            // separate even/odd accumulators would change rounding.
            result = _mm_add_ps(result, _mm256_castps256_ps128(product));
            result = _mm_add_ps(result, _mm256_extractf128_ps::<1>(product));
        }
        indices = _mm256_add_ps(indices, _mm256_set1_ps(8.0));
    }
    let mut out = [0.0; 4];
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), result) };
    out
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn spline_distance_avx2(p: Point<f32>, blocks: &[Segments]) -> f32 {
    let (px, py) = (_mm256_set1_ps(p.x), _mm256_set1_ps(p.y));
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let mut best = _mm256_set1_ps(f32::INFINITY);
    for pair in blocks.chunks(2) {
        let a = &pair[0];
        // Duplicate the last real block rather than introduce dummy segments.
        let b = pair.get(1).unwrap_or(a);
        // SAFETY: every field is a complete four-element array. Joining two
        // such arrays provides eight initialized lanes, including odd tails.
        unsafe {
            let join = |a: &[f32; 4], b: &[f32; 4]| {
                _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_loadu_ps(a.as_ptr())),
                    _mm_loadu_ps(b.as_ptr()),
                )
            };
            let (x, y, dx, dy, len2) = (
                join(&a.x, &b.x),
                join(&a.y, &b.y),
                join(&a.dx, &b.dx),
                join(&a.dy, &b.dy),
                join(&a.len2, &b.len2),
            );
            let nonzero = _mm256_cmp_ps::<_CMP_GT_OQ>(len2, _mm256_set1_ps(1e-12));
            let numerator = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(px, x), dx),
                _mm256_mul_ps(_mm256_sub_ps(py, y), dy),
            );
            let t = _mm256_div_ps(numerator, _mm256_blendv_ps(one, len2, nonzero));
            let t = _mm256_blendv_ps(t, zero, _mm256_cmp_ps::<_CMP_LT_OQ>(t, zero));
            let t = _mm256_blendv_ps(t, one, _mm256_cmp_ps::<_CMP_GT_OQ>(t, one));
            let t = _mm256_and_ps(t, nonzero);
            let ex = _mm256_sub_ps(px, _mm256_add_ps(x, _mm256_mul_ps(t, dx)));
            let ey = _mm256_sub_ps(py, _mm256_add_ps(y, _mm256_mul_ps(t, dy)));
            let distance = _mm256_add_ps(_mm256_mul_ps(ex, ex), _mm256_mul_ps(ey, ey));
            // A NaN distance must leave the incumbent minimum intact.
            best = _mm256_min_ps(distance, best);
        }
    }
    let best = _mm_min_ps(
        _mm256_castps256_ps128(best),
        _mm256_extractf128_ps::<1>(best),
    );
    let best = _mm_min_ps(best, _mm_movehl_ps(best, best));
    _mm_cvtss_f32(_mm_min_ss(best, _mm_shuffle_ps::<0x55>(best, best)))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn horner_step<const FUSED: bool>(a: __m256, b: __m256, c: f32) -> __m256 {
    let c = _mm256_set1_ps(c);
    if FUSED {
        _mm256_fmadd_ps(a, b, c)
    } else {
        _mm256_add_ps(_mm256_mul_ps(a, b), c)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn erf<const FUSED: bool>(x: __m256) -> __m256 {
    let a = abs(x);
    let d1 = horner_step::<FUSED>(a, _mm256_set1_ps(7.77394369e-02), 2.05260015e-04);
    let d2 = horner_step::<FUSED>(d1, a, 2.32120216e-01);
    let d3 = horner_step::<FUSED>(d2, a, 2.77820801e-01);
    let d4 = horner_step::<FUSED>(d3, a, 1.0);
    let d5 = _mm256_mul_ps(d4, d4);
    let inv = _mm256_div_ps(_mm256_set1_ps(1.0), d5);
    let r = _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(inv, inv));
    _mm256_blendv_ps(
        r,
        neg(r),
        _mm256_cmp_ps::<_CMP_LE_OQ>(x, _mm256_setzero_ps()),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn dx_lanes(x: i32, center: f32) -> __m256 {
    // Add in integer lanes before conversion to preserve rounding above 2^24.
    debug_assert!(x <= i32::MAX - 7);
    let indices = _mm256_cvtepi32_ps(_mm256_add_epi32(
        _mm256_set1_epi32(x),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    ));
    _mm256_sub_ps(indices, _mm256_set1_ps(center))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn blob_weights<const FUSED: bool>(dx: __m256, dy: f32, inv_sigma: f32, amp: f32) -> __m256 {
    let dist = _mm256_sqrt_ps(_mm256_add_ps(
        _mm256_mul_ps(dx, dx),
        _mm256_set1_ps(dy * dy),
    ));
    let half = _mm256_mul_ps(dist, _mm256_set1_ps(0.5));
    let a = _mm256_mul_ps(
        _mm256_add_ps(half, _mm256_set1_ps(0.353_553_39)),
        _mm256_set1_ps(inv_sigma),
    );
    let b = _mm256_mul_ps(
        _mm256_sub_ps(half, _mm256_set1_ps(0.353_553_39)),
        _mm256_set1_ps(inv_sigma),
    );
    let e = _mm256_sub_ps(erf::<FUSED>(a), erf::<FUSED>(b));
    _mm256_mul_ps(_mm256_mul_ps(e, _mm256_set1_ps(amp)), e)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn intensity(sample: &Sample, dy: f32, x: usize, amp: f32) -> __m256 {
    // Image dimensions are at most 30 bits, so padded coordinates fit in i32.
    debug_assert!(x <= i32::MAX as usize - 7);
    let dx = dx_lanes(x as i32, sample.position.x);
    // Match the scalar renderer's polynomial rounding.
    blob_weights::<false>(dx, dy, sample.inv_sigma, amp)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn row_weights(row: BlobWeightRow, x: i32) -> __m256 {
    let dx = _mm256_mul_ps(dx_lanes(x, row.center), _mm256_set1_ps(row.step));
    blob_weights::<true>(dx, row.dy, row.inv_sigma, row.amp)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn blob_weight_row_avx2(row: BlobWeightRow, x0: i32, weights: &mut [f32]) {
    let (blocks, tail) = weights.as_chunks_mut::<8>();
    for (i, block) in blocks.iter_mut().enumerate() {
        // SAFETY: each block has room for all eight lanes.
        unsafe { _mm256_storeu_ps(block.as_mut_ptr(), row_weights(row, x0 + 8 * i as i32)) };
    }
    if !tail.is_empty() {
        let mut padded = [0.0; 8];
        // SAFETY: write the full vector to the local array and copy only the tail.
        unsafe {
            _mm256_storeu_ps(
                padded.as_mut_ptr(),
                row_weights(row, x0 + 8 * blocks.len() as i32),
            )
        };
        tail.copy_from_slice(&padded[..tail.len()]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn spline_render_row_avx2(
    sample: &Sample,
    dy: f32,
    x0: usize,
    rows: [&mut [f32]; 3],
    amp: f32,
) {
    let [rx, ry, rb] = rows;
    assert_eq!(rx.len(), ry.len());
    assert_eq!(rx.len(), rb.len());
    let [(rx, tx), (ry, ty), (rb, tb)] = [rx, ry, rb].map(|row| row.as_chunks_mut::<8>());
    let n = rx.len() * 8;
    for (i, ((rx, ry), rb)) in rx.iter_mut().zip(ry).zip(rb).enumerate() {
        let li = intensity(sample, dy, x0 + i * 8, amp);
        for (block, color) in [rx, ry, rb].into_iter().zip(sample.color) {
            // SAFETY: each block contains eight readable and writable pixels.
            unsafe {
                let ptr = block.as_mut_ptr();
                _mm256_storeu_ps(
                    ptr,
                    _mm256_add_ps(
                        _mm256_loadu_ps(ptr),
                        _mm256_mul_ps(li, _mm256_set1_ps(color)),
                    ),
                );
            }
        }
    }
    if tx.len() >= 2 {
        let mut weights = [0.0; 8];
        // SAFETY: incomplete vectors write only to a local, full-sized array.
        unsafe { _mm256_storeu_ps(weights.as_mut_ptr(), intensity(sample, dy, x0 + n, amp)) };
        for (tail, color) in [tx, ty, tb].into_iter().zip(sample.color) {
            for (out, li) in tail.iter_mut().zip(weights) {
                *out += color * li;
            }
        }
    } else {
        sample.scalar_row(dy, x0 + n, [tx, ty, tb], amp);
    }
}
