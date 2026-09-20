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

use super::Point;
use crate::encoding_context::EncodingContext;
use std::ops::Range;

#[derive(Default)]
pub(crate) struct Segments {
    pub(crate) x: [f32; 4],
    pub(crate) y: [f32; 4],
    pub(crate) dx: [f32; 4],
    pub(crate) dy: [f32; 4],
    pub(crate) len2: [f32; 4],
}

pub(super) struct Polyline {
    blocks: Vec<Segments>,
    len: usize,
    distance_blocks: SegmentDistanceFn,
}

impl Polyline {
    pub(super) fn new(ctx: &EncodingContext, points: &[Point<f32>]) -> Self {
        let len = points.len().saturating_sub(1);
        let mut blocks = Vec::<Segments>::with_capacity(len.div_ceil(4));
        for (i, &[a, b]) in points.array_windows::<2>().enumerate() {
            if i % 4 == 0 {
                blocks.push(Segments::default());
            }
            let block = blocks.last_mut().unwrap();
            let lane = i % 4;
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            block.x[lane] = a.x;
            block.y[lane] = a.y;
            block.dx[lane] = dx;
            block.dy[lane] = dy;
            block.len2[lane] = dx * dx + dy * dy;
        }
        Self {
            blocks,
            len,
            distance_blocks: ctx.spline_distance,
        }
    }

    pub(super) fn distance(&self, p: Point<f32>) -> f32 {
        self.distance_range(p, 0..self.len)
    }

    /// `segments` indexes edges, so points[a..b] corresponds to a..b-1.
    pub(super) fn distance_range(&self, p: Point<f32>, segments: Range<usize>) -> f32 {
        assert!(segments.start <= segments.end && segments.end <= self.len);
        let mut best = f32::INFINITY;
        let prefix = segments.start.next_multiple_of(4).min(segments.end);
        for i in segments.start..prefix {
            best = best.min(self.scalar(p, i));
        }
        let end = segments.end / 4 * 4;
        if prefix < end {
            let blocks = &self.blocks[prefix / 4..end / 4];
            best = best.min((self.distance_blocks)(p, blocks));
        }
        for i in prefix.max(end)..segments.end {
            best = best.min(self.scalar(p, i));
        }
        best
    }

    fn scalar(&self, p: Point<f32>, index: usize) -> f32 {
        self.blocks[index / 4].distance(p, index % 4)
    }
}

impl Segments {
    #[inline]
    fn distance(&self, p: Point<f32>, lane: usize) -> f32 {
        let (x, y, dx, dy, len2) = (
            self.x[lane],
            self.y[lane],
            self.dx[lane],
            self.dy[lane],
            self.len2[lane],
        );
        let t = if len2 > 1e-12 {
            (((p.x - x) * dx + (p.y - y) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (ex, ey) = (p.x - (x + t * dx), p.y - (y + t * dy));
        ex * ex + ey * ey
    }
}

pub(crate) type SegmentDistanceFn = fn(Point<f32>, &[Segments]) -> f32;

pub(crate) fn select_segment_distance_fn() -> SegmentDistanceFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |point, blocks| unsafe { crate::neon::spline_distance_neon(point, blocks) };
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |point, blocks| unsafe { crate::avx::spline_distance_avx2(point, blocks) };
    }
    #[allow(unreachable_code)]
    segment_distance_scalar
}

fn segment_distance_scalar(p: Point<f32>, blocks: &[Segments]) -> f32 {
    let mut best = f32::INFINITY;
    for block in blocks {
        for lane in 0..4 {
            best = best.min(block.distance(p, lane));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(p: Point<f32>, points: &[Point<f32>]) -> f32 {
        let mut best = f32::INFINITY;
        for &[a, b] in points.array_windows::<2>() {
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            let len2 = dx * dx + dy * dy;
            let t = if len2 > 1e-12 {
                (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (ex, ey) = (p.x - (a.x + t * dx), p.y - (a.y + t * dy));
            best = best.min(ex * ex + ey * ey);
        }
        best
    }

    #[test]
    fn prepared_distances_match_segment_reference_for_every_alignment() {
        let kernels = crate::encoding_context::EncodingContext::default();
        let points: Vec<_> = (0..if cfg!(miri) { 17 } else { 71 })
            .map(|i| {
                // Include duplicate/degenerate edges as well as short bends.
                let i = i / 2;
                Point::new((i * 19 % 31) as f32 * 0.7, (i * 7 % 13) as f32 * 0.1234)
            })
            .collect();
        // Tiny edges cross the degenerate-segment threshold; large coordinates
        // exercise cancellation and coarse f32 spacing in the projections.
        for (scale, offset) in [(1.0, 0.0), (1e-7, 0.0), (1000.0, 1e7)] {
            let points: Vec<_> = points
                .iter()
                .map(|p| Point::new(p.x * scale + offset, p.y * scale + offset))
                .collect();
            let line = Polyline::new(&kernels, &points);
            for p in [
                Point::new(-1.0, -0.0),
                Point::new(14.3 * scale + offset, 8.9 * scale + offset),
                points[24.min(points.len() - 1)],
                Point::new(1e7, -1e5),
            ] {
                for start in 0..points.len() {
                    for end in start..points.len() {
                        assert_eq!(
                            line.distance_range(p, start..end).to_bits(),
                            reference(p, &points[start..=end]).to_bits(),
                            "scale {scale}, offset {offset}, range {start}..{end}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            Polyline::new(&kernels, &[]).distance(points[0]),
            f32::INFINITY
        );
    }
}
