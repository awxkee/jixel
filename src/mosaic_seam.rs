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
//! Adaptive transform (AC strategy) selection via rate-distortion optimization.

//! Boundary-error scoring for cached fine-mosaic assignments.

use crate::dct::fmla;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

pub(crate) type MosaicSeamStatsFn = fn(
    &EncodingContext,
    &Image3F,
    usize,
    usize,
    usize,
    usize,
    f32,
    &[&[[f32; 64]; 3]],
) -> (f32, f32);

pub(crate) fn select_mosaic_seam_stats_fn() -> MosaicSeamStatsFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |ctx, opsin, px, py, cxb, cyb, distance, selected| unsafe {
        crate::neon::mosaic_seam_stats_neon(ctx, opsin, px, py, cxb, cyb, distance, selected)
    };
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |ctx, opsin, px, py, cxb, cyb, distance, selected| unsafe {
            crate::avx::mosaic_seam_stats_avx2(ctx, opsin, px, py, cxb, cyb, distance, selected)
        };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |ctx, opsin, px, py, cxb, cyb, distance, selected| unsafe {
            crate::sse::mosaic_seam_stats_sse41(ctx, opsin, px, py, cxb, cyb, distance, selected)
        };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::mosaic_seam_stats_wasm;
    #[allow(unreachable_code)]
    mosaic_seam_stats_scalar
}

/// Boundary-error statistics of one candidate mosaic assignment, computed
/// straight from the cached per-child spatial error planes. Only the seams
/// between adjacent children exist in a mosaic's 8-grid, so this visits the
/// same pixels, in the same order, as an assembled-region scan, without
/// constructing that larger error plane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mosaic_seam_stats_scalar(
    ctx: &EncodingContext,
    opsin: &Image3F,
    px: usize,
    py: usize,
    cxb: usize,
    cyb: usize,
    distance: f32,
    selected: &[&[[f32; 64]; 3]],
) -> (f32, f32) {
    debug_assert_eq!(selected.len(), cxb * cyb);
    let coarse_mix = ((distance - 1.9) / 0.1).clamp(0.0, 1.0);
    let floor = distance * fmla(coarse_mix, 0.0045 - 0.0015, 0.0015);
    let mut energy = 0.0f32;
    let mut peak = 0.0f32;
    for c in 0..3 {
        let plane = opsin.plane(c);
        let weight = ctx.channel_weight(c);
        let mut channel = 0.0f32;
        for sx in 1..cxb {
            for ky in 0..cyb {
                let left = &selected[ky * cxb + sx - 1][c];
                let right = &selected[ky * cxb + sx][c];
                let left_x = (px + sx * 8 - 1).min(plane.xsize() - 1);
                let right_x = (px + sx * 8).min(plane.xsize() - 1);
                for y in 0..8 {
                    let sy = (py + ky * 8 + y).min(plane.ysize() - 1);
                    let source_gradient = (plane.row(sy)[right_x] - plane.row(sy)[left_x]).abs();
                    let excess =
                        ((right[y * 8] - left[y * 8 + 7]).abs() - 0.5 * source_gradient - floor)
                            .max(0.0);
                    channel = fmla(excess, excess, channel);
                    peak = peak.max(weight * excess * excess);
                }
            }
        }
        for sy_b in 1..cyb {
            for kx in 0..cxb {
                let top = &selected[(sy_b - 1) * cxb + kx][c];
                let bottom = &selected[sy_b * cxb + kx][c];
                let top_y = (py + sy_b * 8 - 1).min(plane.ysize() - 1);
                let bottom_y = (py + sy_b * 8).min(plane.ysize() - 1);
                for x in 0..8 {
                    let sx_abs = (px + kx * 8 + x).min(plane.xsize() - 1);
                    let source_gradient =
                        (plane.row(bottom_y)[sx_abs] - plane.row(top_y)[sx_abs]).abs();
                    let excess =
                        ((bottom[x] - top[7 * 8 + x]).abs() - 0.5 * source_gradient - floor)
                            .max(0.0);
                    channel = fmla(excess, excess, channel);
                    peak = peak.max(weight * excess * excess);
                }
            }
        }
        energy += weight * channel;
    }
    (energy, peak)
}

/// Eight seam pixels gathered in traversal order. Gathering keeps the SIMD
/// kernels independent of image strides and retains edge replication.
#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
))]
pub(crate) struct SeamSamples {
    pub(crate) left: [f32; 8],
    pub(crate) right: [f32; 8],
    pub(crate) source_left: [f32; 8],
    pub(crate) source_right: [f32; 8],
}

/// Shared traversal, inlined into each ISA entry point. Only the independent
/// per-pixel excess and peak calculations are vectorized: the FMA reduction must
/// match the scalar model exactly, including on non-FMA x86 and WASM builds.
#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"),
))]
#[inline(always)]
pub(crate) fn mosaic_seam_stats_with(
    ctx: &EncodingContext,
    opsin: &Image3F,
    px: usize,
    py: usize,
    cxb: usize,
    cyb: usize,
    distance: f32,
    selected: &[&[[f32; 64]; 3]],
    excess8: impl Fn(&SeamSamples, f32, f32) -> ([f32; 8], f32),
) -> (f32, f32) {
    assert_eq!(selected.len(), cxb * cyb);
    let coarse_mix = ((distance - 1.9) / 0.1).clamp(0.0, 1.0);
    let floor = distance * fmla(coarse_mix, 0.0045 - 0.0015, 0.0015);
    let mut energy = 0.0f32;
    let mut peak = 0.0f32;
    for (c, weight) in ctx.channel_weights().into_iter().enumerate() {
        let plane = opsin.plane(c);
        let mut channel = 0.0f32;
        let mut score = |samples: &SeamSamples| {
            let (excesses, seam_peak) = excess8(samples, floor, weight);
            for excess in excesses {
                channel = fmla(excess, excess, channel);
            }
            peak = peak.max(seam_peak);
        };
        for sx in 1..cxb {
            for (ky, children) in selected.chunks_exact(cxb).enumerate() {
                let left_rows = children[sx - 1][c].as_chunks::<8>().0;
                let right_rows = children[sx][c].as_chunks::<8>().0;
                let left_x = (px + sx * 8 - 1).min(plane.xsize() - 1);
                let right_x = (px + sx * 8).min(plane.xsize() - 1);
                let mut samples = SeamSamples {
                    left: [0.0; 8],
                    right: [0.0; 8],
                    source_left: [0.0; 8],
                    source_right: [0.0; 8],
                };
                for ((left, right), (left_row, right_row)) in samples
                    .left
                    .iter_mut()
                    .zip(&mut samples.right)
                    .zip(left_rows.iter().zip(right_rows))
                {
                    *left = left_row[7];
                    *right = right_row[0];
                }
                for (y, (left, right)) in samples
                    .source_left
                    .iter_mut()
                    .zip(&mut samples.source_right)
                    .enumerate()
                {
                    let sy = (py + ky * 8 + y).min(plane.ysize() - 1);
                    let row = plane.row(sy);
                    *left = row[left_x];
                    *right = row[right_x];
                }
                score(&samples);
            }
        }
        for (row_idx, (top_children, bottom_children)) in selected
            .chunks_exact(cxb)
            .zip(selected.chunks_exact(cxb).skip(1))
            .enumerate()
        {
            let sy_b = row_idx + 1;
            for (kx, (top, bottom)) in top_children.iter().zip(bottom_children).enumerate() {
                let top_y = (py + sy_b * 8 - 1).min(plane.ysize() - 1);
                let bottom_y = (py + sy_b * 8).min(plane.ysize() - 1);
                let mut samples = SeamSamples {
                    left: *top[c].last_chunk::<8>().unwrap(),
                    right: *bottom[c].first_chunk::<8>().unwrap(),
                    source_left: [0.0; 8],
                    source_right: [0.0; 8],
                };
                let top_row = plane.row(top_y);
                let bottom_row = plane.row(bottom_y);
                for (x, (left, right)) in samples
                    .source_left
                    .iter_mut()
                    .zip(&mut samples.source_right)
                    .enumerate()
                {
                    let sx_abs = (px + kx * 8 + x).min(plane.xsize() - 1);
                    *left = top_row[sx_abs];
                    *right = bottom_row[sx_abs];
                }
                score(&samples);
            }
        }
        energy += weight * channel;
    }
    (energy, peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernels() -> Vec<(&'static str, MosaicSeamStatsFn)> {
        #[allow(unused_mut)]
        let mut kernels: Vec<(&str, MosaicSeamStatsFn)> =
            vec![("dispatch", select_mosaic_seam_stats_fn())];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        kernels.push(
            ("neon", |ctx, opsin, px, py, cxb, cyb, d, selected| unsafe {
                crate::neon::mosaic_seam_stats_neon(ctx, opsin, px, py, cxb, cyb, d, selected)
            }),
        );
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if is_x86_feature_detected!("avx2") {
            kernels.push(
                ("avx2", |ctx, opsin, px, py, cxb, cyb, d, selected| unsafe {
                    crate::avx::mosaic_seam_stats_avx2(ctx, opsin, px, py, cxb, cyb, d, selected)
                }),
            );
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if is_x86_feature_detected!("sse4.1") {
            kernels.push((
                "sse41",
                |ctx, opsin, px, py, cxb, cyb, d, selected| unsafe {
                    crate::sse::mosaic_seam_stats_sse41(ctx, opsin, px, py, cxb, cyb, d, selected)
                },
            ));
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        kernels.push(("wasm", crate::wasm::mosaic_seam_stats_wasm));
        kernels
    }

    #[test]
    fn simd_matches_scalar_for_seams_edges_and_distance_ramp() {
        let kernels = kernels();
        let mut seed = 127u32;
        let mut random = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.25
        };
        for bias in [crate::xyb::B_BIAS, 0.85] {
            let ctx = EncodingContext::new(
                crate::Speed::Slow,
                None,
                crate::yellow_opsin::matrix_for_bias(bias),
                1.0,
                1,
            );
            for (width, height) in [(1, 1), (7, 9), (16, 16), (29, 31)] {
                let mut opsin = Image3F::new(width, height);
                for c in 0..3 {
                    for y in 0..height {
                        for v in opsin.plane_row_mut(c, y) {
                            *v = random();
                        }
                    }
                }
                let mut planes = [[[[0.0f32; 64]; 3]; 3]; 4];
                for child in &mut planes {
                    for candidate in child {
                        for channel in candidate {
                            for v in channel {
                                *v = random();
                            }
                        }
                    }
                }
                for (cxb, cyb) in [(1usize, 1usize), (2, 1), (1, 2), (2, 2)] {
                    for (px, py) in [(0, 0), (width - 1, height - 1)] {
                        for distance in [0.0, 0.5, 1.9, 1.95, 2.0, 3.0] {
                            for code in 0..3usize.pow((cxb * cyb) as u32) {
                                let mut rest = code;
                                let selected: Vec<_> = planes[..cxb * cyb]
                                    .iter()
                                    .map(|child| {
                                        let i = rest % 3;
                                        rest /= 3;
                                        &child[i]
                                    })
                                    .collect();
                                let expected = mosaic_seam_stats_scalar(
                                    &ctx, &opsin, px, py, cxb, cyb, distance, &selected,
                                );
                                for &(name, kernel) in &kernels {
                                    let got =
                                        kernel(&ctx, &opsin, px, py, cxb, cyb, distance, &selected);
                                    assert_eq!(
                                        (got.0.to_bits(), got.1.to_bits()),
                                        (expected.0.to_bits(), expected.1.to_bits()),
                                        "{name}: {width}x{height}, origin={px},{py}, grid={cxb}x{cyb}, d={distance}, code={code}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simd_preserves_zero_and_nonfinite_excess_handling() {
        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            None,
            crate::xyb::XybMatrix::SPEC,
            1.0,
            1,
        );
        let opsin = Image3F::new(16, 16);
        for value in [
            0.0,
            -0.0,
            0.0015,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let left = [[0.0; 64]; 3];
            let right = [[value; 64]; 3];
            for (cxb, cyb) in [(2, 1), (1, 2)] {
                let selected = [&left, &right];
                let expected =
                    mosaic_seam_stats_scalar(&ctx, &opsin, 0, 0, cxb, cyb, 1.0, &selected);
                for (name, kernel) in kernels() {
                    let got = kernel(&ctx, &opsin, 0, 0, cxb, cyb, 1.0, &selected);
                    assert_eq!(got, expected, "{name}: {value}");
                }
            }
        }
    }
}
