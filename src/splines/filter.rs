/*
 * Copyright (c) Radzivon Bartoshyk, 10/2024. All rights reserved.
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
//! Single-channel filtering plans adapted from pic-scale's FilterBounds and
//! floating_point_vertical scalar accumulators. Horizontal convolution keeps
//! the fixed-kernel tap-major path, which benchmarks faster at 1:1.
//!
//! The detector supplies signed Gaussian derivative kernels. Unlike resize
//! weights these must not be normalized, made symmetric, or skipped at 1:1.
//! Replicated border taps stay separate to retain the original summation order.
#![forbid(unsafe_code)]

use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;

#[derive(Clone, Copy)]
struct FilterBounds {
    start: usize,
    size: usize,
    /// Number of taps repeating the first sample before the contiguous span.
    left: usize,
}

impl FilterBounds {
    #[inline]
    fn index(self, tap: usize) -> usize {
        self.start + tap.saturating_sub(self.left).min(self.size - 1)
    }
}

/// One fixed kernel and precomputed source bounds, reusable across planes.
/// Interior positions share the kernel instead of storing a copy per pixel.
pub(super) struct FilterPlan {
    kernel: Vec<f32>,
    bounds: Vec<FilterBounds>,
}

impl FilterPlan {
    pub(super) fn new(size: usize, kernel: &[f32]) -> Self {
        assert!(size > 0 && !kernel.is_empty() && kernel.len() % 2 == 1);
        let radius = kernel.len() / 2;
        let bounds = (0..size)
            .map(|x| {
                let start = x.saturating_sub(radius);
                FilterBounds {
                    start,
                    size: (x + radius + 1).min(size) - start,
                    left: radius.saturating_sub(x),
                }
            })
            .collect();
        Self {
            kernel: kernel.to_vec(),
            bounds,
        }
    }

    #[inline]
    fn row_span<'a>(&self, src: &'a [f32], bounds: FilterBounds) -> &'a [f32] {
        &src[bounds.start..bounds.start + bounds.size]
    }

    #[inline]
    fn edge_samples<'a>(
        &self,
        span: &'a [f32],
        bounds: FilterBounds,
    ) -> impl Iterator<Item = &'a f32> {
        std::iter::repeat_n(&span[0], bounds.left)
            .chain(span.iter())
            .chain(std::iter::repeat_n(
                span.last().unwrap(),
                self.kernel.len() - bounds.left - bounds.size,
            ))
    }

    fn horizontal_row(&self, src: &[f32], dst: &mut [f32]) {
        let w = self.bounds.len();
        let radius = self.kernel.len() / 2;
        if w > 2 * radius {
            let inner = &mut dst[radius..w - radius];
            inner.fill(0.0);
            // For this fixed 1:1 kernel, tap-major iteration is faster than
            // pic-scale's four-row dot products. Both source and destination
            // walk contiguous slices with no indexing inside the tap loop.
            for (&weight, pixels) in self.kernel.iter().zip(src.windows(inner.len())) {
                for (out, &pixel) in inner.iter_mut().zip(pixels) {
                    *out += weight * pixel;
                }
            }
        }
        let edge = |out: &mut f32, &bounds: &FilterBounds| {
            let span = self.row_span(src, bounds);
            *out = self
                .kernel
                .iter()
                .zip(self.edge_samples(span, bounds))
                .map(|(&k, &v)| k * v)
                .sum();
        };
        if w > 2 * radius {
            for (out, bounds) in dst[..radius].iter_mut().zip(&self.bounds[..radius]) {
                edge(out, bounds);
            }
            for (out, bounds) in dst[w - radius..].iter_mut().zip(&self.bounds[w - radius..]) {
                edge(out, bounds);
            }
        } else {
            for (out, bounds) in dst.iter_mut().zip(&self.bounds) {
                edge(out, bounds);
            }
        }
    }

    pub(super) fn horizontal(
        &self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        src: &[f32],
        dst: &mut [f32],
    ) {
        let w = self.bounds.len();
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len() % w, 0);
        let h = src.len() / w;
        if h == 0 {
            return;
        }
        let bands = (ctx.thread_pool.num_threads() * 4).min(h);
        let rows_per = h.div_ceil(bands);
        let mut parts: Vec<_> = dst.chunks_mut(rows_per * w).collect();
        ctx.thread_pool
            .steal_for_each_mut(scratch, &mut parts, |b, rows, _| {
                let src = &src[b * rows_per * w..b * rows_per * w + rows.len()];
                for (src, dst) in src.chunks_exact(w).zip(rows.chunks_exact_mut(w)) {
                    self.horizontal_row(src, dst);
                }
            });
    }

    /// Pic-scale's scalar column accumulator groups: 16, then 4, then 1.
    /// Each output is stored once after visiting all taps; no explicit SIMD.
    #[inline]
    fn vertical_block<const N: usize>(
        &self,
        src: &[f32],
        w: usize,
        y: usize,
        x: usize,
        dst: &mut [f32],
    ) {
        let bounds = self.bounds[y];
        let mut sums = [0.0f32; N];
        for (j, &weight) in self.kernel.iter().enumerate() {
            let start = bounds.index(j) * w + x;
            let pixels = &src[start..start + N];
            for (sum, &pixel) in sums.iter_mut().zip(pixels) {
                *sum += weight * pixel;
            }
        }
        dst.copy_from_slice(&sums);
    }

    pub(super) fn vertical_row(&self, src: &[f32], w: usize, y: usize, dst: &mut [f32]) {
        debug_assert_eq!(src.len(), w * self.bounds.len());
        debug_assert_eq!(dst.len(), w);
        let mut x = 0;
        let (chunks, remainder) = dst.as_chunks_mut::<16>();
        for chunk in chunks {
            self.vertical_block::<16>(src, w, y, x, chunk);
            x += 16;
        }
        let (chunks, remainder) = remainder.as_chunks_mut::<4>();
        for chunk in chunks {
            self.vertical_block::<4>(src, w, y, x, chunk);
            x += 4;
        }
        for pixel in remainder {
            self.vertical_block::<1>(src, w, y, x, std::slice::from_mut(pixel));
            x += 1;
        }
    }

    pub(super) fn vertical(
        &self,
        ctx: &EncodingContext,
        scratch: &mut CoderScratch,
        src: &[f32],
        dst: &mut [f32],
    ) {
        let h = self.bounds.len();
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len() % h, 0);
        let w = src.len() / h;
        if w == 0 {
            return;
        }
        let bands = (ctx.thread_pool.num_threads() * 4).min(h);
        let rows_per = h.div_ceil(bands);
        let mut parts: Vec<_> = dst.chunks_mut(rows_per * w).collect();
        ctx.thread_pool
            .steal_for_each_mut(scratch, &mut parts, |b, rows, _| {
                for (y, row) in rows.chunks_exact_mut(w).enumerate() {
                    self.vertical_row(src, w, b * rows_per + y, row);
                }
            });
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use crate::{Speed, xyb::XybMatrix};
    use std::{hint::black_box, time::Instant};

    // The pre-port tap-major implementation, retained only for manual timing.
    fn reference(src: &[f32], dst: &mut [f32], w: usize, kernel: &[f32], vertical: bool) {
        let h = src.len() / w;
        let radius = kernel.len() / 2;
        for (y, out) in dst.chunks_exact_mut(w).enumerate() {
            out.fill(0.0);
            if vertical {
                for (j, &k) in kernel.iter().enumerate() {
                    let yi = (y + j).saturating_sub(radius).min(h - 1);
                    for (o, &v) in out.iter_mut().zip(&src[yi * w..(yi + 1) * w]) {
                        *o += k * v;
                    }
                }
            } else {
                let row = &src[y * w..(y + 1) * w];
                let inner = w - 2 * radius;
                for (j, &k) in kernel.iter().enumerate() {
                    for (o, &v) in out[radius..radius + inner]
                        .iter_mut()
                        .zip(&row[j..j + inner])
                    {
                        *o += k * v;
                    }
                }
                for x in (0..radius).chain(w - radius..w) {
                    out[x] = kernel
                        .iter()
                        .enumerate()
                        .map(|(j, &k)| k * row[(x + j).saturating_sub(radius).min(w - 1)])
                        .sum();
                }
            }
        }
    }

    #[test]
    #[ignore = "manual release-mode filter timing"]
    fn filter_timing() {
        let ctx = EncodingContext::new(Speed::Slow, XybMatrix::SPEC, 3.0, 1);
        let mut scratch = Box::<CoderScratch>::default();
        let (w, h) = (1680, 1120);
        let src: Vec<_> = (0..w * h).map(|i| (i % 37) as f32 * 0.013 - 0.2).collect();
        let mut dst = vec![0.; w * h];
        for sigma in [0.7, 2.2, 3.0] {
            let [kernel, _, _] = super::super::detect::gaussian_kernels(sigma);
            let hp = FilterPlan::new(w, &kernel);
            let vp = FilterPlan::new(h, &kernel);
            for vertical in [false, true] {
                for planned in [false, true] {
                    let mut elapsed = Vec::new();
                    for _ in 0..7 {
                        let start = Instant::now();
                        if planned {
                            if vertical {
                                vp.vertical(&ctx, &mut scratch, black_box(&src), &mut dst)
                            } else {
                                hp.horizontal(&ctx, &mut scratch, black_box(&src), &mut dst)
                            }
                        } else {
                            reference(black_box(&src), &mut dst, w, &kernel, vertical);
                        }
                        black_box(&dst);
                        elapsed.push(start.elapsed().as_secs_f64() * 1000.);
                    }
                    elapsed.sort_by(f64::total_cmp);
                    println!(
                        "sigma={sigma} vertical={vertical} planned={planned} median_ms={:.3}",
                        elapsed[3]
                    );
                }
            }
        }
    }
}
