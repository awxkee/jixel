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

//! Bounded row rings for all three EPF stages, fused into one pass. Each stage
//! filters straight into the next stage's input ring, so the frame is read
//! once and written once whatever the pass count, as a decoder's row pipeline
//! does. Workers own row bands; band edges recompute the later stages' halos.
//! Bands walk in column strips so every ring of the chain stays in L1.
use super::kernels::{FilterRow, Kernels, Neighbor};
use super::{LoopFilter, Stage, mirror};
use crate::coder_scratch::CoderScratch;
use crate::image::{Image3F, Plane};
use crate::thread_pool::ThreadPool;

fn grow(data: &mut Vec<f32>, len: usize) {
    if data.len() < len {
        data.resize(len, 0.0);
    }
}
fn slot(row: isize, rows: usize, stride: usize) -> usize {
    row.rem_euclid(rows as isize) as usize * stride
}

/// The columns one stage covers in the current strip.
#[derive(Clone, Copy, Default)]
struct Span {
    /// Output columns; `lo` is block aligned so lanes never straddle a block.
    lo: usize,
    hi: usize,
    /// Frame column of ring index 0: left of zero at the frame's left edge.
    origin: isize,
    stride: usize,
}

impl Span {
    fn width(&self) -> usize {
        self.hi - self.lo
    }
    /// Ring index of output column `lo`.
    fn offset(&self) -> usize {
        (self.lo as isize - self.origin) as usize
    }
}

/// One stage's rings: mirrored-padded input rows, per-direction differences
/// (patch stages) and SADs, all indexed by absolute frame row.
#[derive(Default)]
struct Rings {
    span: Span,
    pixels: [Vec<f32>; 3],
    difference: [Vec<f32>; 6],
    sad: [Vec<f32>; 6],
}

#[derive(Default)]
pub(super) struct Scratch {
    rings: [Rings; 3],
}

/// What every band shares: the stage chain and its per-frame inputs.
pub(super) struct Chain<'a> {
    pub(super) stages: &'a [(&'a Stage, f32)],
    pub(super) source: &'a Image3F,
    pub(super) lf: &'a LoopFilter,
    pub(super) sigma: &'a [f32],
    pub(super) blocks_w: usize,
    pub(super) kernels: Kernels,
    /// Target strip width in pixels, a multiple of the 8-pixel block.
    pub(super) strip: usize,
}

impl Rings {
    fn prepare(&mut self, stage: &Stage, span: Span) {
        self.span = span;
        let stride = span.stride;
        for p in &mut self.pixels {
            grow(p, (2 * stage.border + 1) * stride);
        }
        for (i, &(_, dy)) in stage.dirs.iter().enumerate() {
            if stage.patch {
                grow(&mut self.difference[i], 3 * stride);
            }
            grow(&mut self.sad[i], (dy as usize + 1) * stride);
        }
    }
    fn start(&self, stage: &Stage, row: isize) -> usize {
        slot(row, 2 * stage.border + 1, self.span.stride)
    }
    /// Frame columns `lo..hi` of input row `row`, for the previous stage to fill.
    fn row_mut(&mut self, stage: &Stage, row: isize, lo: usize, hi: usize) -> [&mut [f32]; 3] {
        let start = self.start(stage, row) + (lo as isize - self.span.origin) as usize;
        self.pixels
            .each_mut()
            .map(|p| &mut p[start..start + hi - lo])
    }
    /// Mirror the columns of a filled input row that lie outside the frame.
    fn pad(&mut self, stage: &Stage, row: isize, w: usize) {
        let Span { origin, stride, .. } = self.span;
        let end = origin + stride as isize;
        if origin >= 0 && end <= w as isize {
            return;
        }
        let start = self.start(stage, row);
        let index = |x: isize| (x - origin) as usize;
        for p in &mut self.pixels {
            let dst = &mut p[start..start + stride];
            for x in (origin..0).chain(w as isize..end) {
                dst[index(x)] = dst[index(mirror(x, w as isize) as isize)];
            }
        }
    }
    fn load_source(&mut self, stage: &Stage, image: &Image3F, row: isize) {
        let w = image.xsize();
        let y = mirror(row, image.ysize() as isize);
        let lo = self.span.origin.max(0) as usize;
        let hi = (self.span.origin + self.span.stride as isize).min(w as isize) as usize;
        for (c, dst) in self.row_mut(stage, row, lo, hi).into_iter().enumerate() {
            dst.copy_from_slice(&image.plane_row(c, y)[lo..hi]);
        }
        self.pad(stage, row, w);
    }
    /// A row outside the frame repeats its mirror, already in the window.
    fn reflect(&mut self, stage: &Stage, row: isize, h: usize) {
        let stride = self.span.stride;
        let from = self.start(stage, mirror(row, h as isize) as isize);
        let to = self.start(stage, row);
        for p in &mut self.pixels {
            p.copy_within(from..from + stride, to);
        }
    }
    fn difference(&mut self, stage: &Stage, dir: usize, row: isize, lf: &LoopFilter, k: Kernels) {
        let (dx, dy) = stage.dirs[dir];
        let stride = self.span.stride;
        let lo = (-dx).max(0) as usize;
        let hi = stride - dx.max(0) as usize;
        let a = slot(row, 2 * stage.border + 1, stride) + lo;
        let b = slot(row + dy, 2 * stage.border + 1, stride) + (lo as isize + dx) as usize;
        let n = hi - lo;
        let pixels = &self.pixels;
        let dst = if stage.patch {
            &mut self.difference[dir]
        } else {
            &mut self.sad[dir]
        };
        let start = slot(row, if stage.patch { 3 } else { dy as usize + 1 }, stride) + lo;
        (k.difference)(
            pixels.each_ref().map(|p| &p[a..a + n]),
            pixels.each_ref().map(|p| &p[b..b + n]),
            lf.channel_scale,
            &mut dst[start..start + n],
        );
    }
    fn patch(&mut self, stage: &Stage, dir: usize, row: isize, k: Kernels) {
        let (dx, dy) = stage.dirs[dir];
        let stride = self.span.stride;
        let lo = (-dx).max(0) as usize + 1;
        let hi = stride - dx.max(0) as usize - 1;
        let src = &self.difference[dir];
        let m = slot(row, 3, stride);
        let u = slot(row - 1, 3, stride);
        let d = slot(row + 1, 3, stride);
        let start = slot(row, dy as usize + 1, stride);
        (k.patch)(
            [
                &src[m + lo..m + hi],
                &src[m + lo - 1..m + hi - 1],
                &src[m + lo + 1..m + hi + 1],
                &src[u + lo..u + hi],
                &src[d + lo..d + hi],
            ],
            &mut self.sad[dir][start + lo..start + hi],
        );
    }
    /// SADs for output row `y`: the whole lead-in on a strip's first row,
    /// one new row of differences after that.
    fn sads(&mut self, stage: &Stage, y: isize, first: bool, lf: &LoopFilter, k: Kernels) {
        for (dir, &(_, dy)) in stage.dirs.iter().enumerate() {
            match (first, stage.patch) {
                (true, true) => {
                    self.difference(stage, dir, y - dy - 1, lf, k);
                    self.difference(stage, dir, y - dy, lf, k);
                    for r in y - dy..=y {
                        self.difference(stage, dir, r + 1, lf, k);
                        self.patch(stage, dir, r, k);
                    }
                }
                (true, false) => {
                    for r in y - dy..=y {
                        self.difference(stage, dir, r, lf, k);
                    }
                }
                (false, true) => {
                    self.difference(stage, dir, y + 1, lf, k);
                    self.patch(stage, dir, y, k);
                }
                (false, false) => self.difference(stage, dir, y, lf, k),
            }
        }
    }
    fn filter(&self, stage: &Stage, sm: f32, y: isize, chain: &Chain<'_>, out: [&mut [f32]; 3]) {
        let b = stage.border;
        let (stride, off, w) = (
            self.span.stride,
            self.span.offset() as isize,
            self.span.width(),
        );
        let center = slot(y, 2 * b + 1, stride) + off as usize;
        let neighbors: [Neighbor<'_>; 12] = std::array::from_fn(|i| {
            if i >= stage.neighbors.len() {
                return Neighbor {
                    pixels: [&[]; 3],
                    sad: &[],
                };
            }
            let (dir, along) = stage.neighbors[i];
            let (dx, dy) = stage.dirs[dir];
            let (ny, nx, sy, sx) = if along {
                (y + dy, off + dx, y, off)
            } else {
                (y - dy, off - dx, y - dy, off - dx)
            };
            let at = slot(ny, 2 * b + 1, stride) + nx as usize;
            let sad = slot(sy, dy as usize + 1, stride) + sx as usize;
            Neighbor {
                pixels: self.pixels.each_ref().map(|p| &p[at..at + w]),
                sad: &self.sad[dir][sad..sad + w],
            }
        });
        let gy = y as usize;
        let blocks = (gy / 8) * chain.blocks_w;
        let row = FilterRow {
            center: self.pixels.each_ref().map(|p| &p[center..center + w]),
            neighbors: &neighbors[..stage.neighbors.len()],
            sigma: &chain.sigma[blocks + self.span.lo / 8..blocks + self.span.hi.div_ceil(8)],
            border: gy.is_multiple_of(8) || gy % 8 == 7,
            sm,
            bsm: sm * chain.lf.border_sad_mul,
        };
        (chain.kernels.filter)(row, out);
    }
}

/// One band's walk through the chain. Stage `s` produces frame rows from
/// `begin[s]` in order; the last stage's rows are the band.
struct Band<'a, 'b> {
    chain: &'a Chain<'b>,
    rings: &'a mut [Rings; 3],
    begin: [usize; 3],
    next: [usize; 3],
    first: usize,
    out: [&'a mut [f32]; 3],
}

impl Band<'_, '_> {
    /// Produce stage `s` row `y`, pulling the input rows it needs first.
    fn produce(&mut self, s: usize, y: usize) {
        let chain = self.chain;
        let (stage, sm) = chain.stages[s];
        let (w, h) = (chain.source.xsize(), chain.source.ysize());
        let b = stage.border as isize;
        let yi = y as isize;
        let first = y == self.begin[s];
        // Rows y - b..=y + b of the stage input must be in the ring.
        let rows = if first {
            yi - b..=yi + b
        } else {
            yi + b..=yi + b
        };
        if s == 0 {
            for r in rows {
                self.rings[0].load_source(stage, chain.source, r);
            }
        } else {
            let last_real = (yi + b).min(h as isize - 1) as usize;
            while self.next[s - 1] <= last_real {
                let r = self.next[s - 1];
                self.produce(s - 1, r);
            }
            for r in rows {
                if r < 0 || r >= h as isize {
                    self.rings[s].reflect(stage, r, h);
                }
            }
        }
        self.rings[s].sads(stage, yi, first, chain.lf, chain.kernels);
        let (done, rest) = self.rings.split_at_mut(s + 1);
        let ring = &done[s];
        let Span { lo, hi, .. } = ring.span;
        if s + 1 == chain.stages.len() {
            let r = (y - self.first) * w;
            let out = self.out.each_mut().map(|p| &mut p[r + lo..r + hi]);
            ring.filter(stage, sm, yi, chain, out);
        } else {
            let (next_stage, _) = chain.stages[s + 1];
            let out = rest[0].row_mut(next_stage, yi, lo, hi);
            ring.filter(stage, sm, yi, chain, out);
            rest[0].pad(next_stage, yi, w);
        }
        self.next[s] = y + 1;
    }
}

/// Spans of every stage for the final-stage columns `x0..x1`: each earlier
/// stage widens by the later stage's border, out to whole blocks.
fn spans(stages: &[(&Stage, f32)], w: usize, x0: usize, x1: usize) -> [Span; 3] {
    let n = stages.len();
    let mut spans = [Span::default(); 3];
    let (mut lo, mut hi) = (x0, x1);
    for s in (0..n).rev() {
        spans[s].lo = lo;
        spans[s].hi = hi;
        let b = stages[s].0.border;
        lo = lo.saturating_sub(b) / 8 * 8;
        hi = (hi + b).next_multiple_of(8).min(w);
    }
    for s in 0..n {
        let b = stages[s].0.border as isize;
        let (lo, hi) = (spans[s].lo as isize, spans[s].hi as isize);
        // The ring holds the previous stage's whole output, plus mirrored
        // columns where this stage's border leaves the frame.
        let (start, end) = if s == 0 {
            (lo - b, hi + b)
        } else {
            let prev = spans[s - 1];
            (
                (prev.lo as isize).min(lo - b),
                (prev.hi as isize).max(hi + b),
            )
        };
        spans[s].origin = start;
        spans[s].stride = (end - start) as usize;
    }
    spans
}

pub(super) fn render(
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    chain: &Chain<'_>,
    output: &mut Image3F,
    work: &mut [Scratch],
) {
    let (w, h) = (output.xsize(), output.ysize());
    let n = chain.stages.len();
    let band_rows = h.div_ceil(work.len());
    // Strip starts are block aligned; a narrow remainder joins the last strip.
    let strips = ((w + chain.strip / 2) / chain.strip).max(1);
    let bounds: Vec<usize> = (0..=strips)
        .map(|i| {
            if i == strips {
                w
            } else {
                i * w / strips / 8 * 8
            }
        })
        .collect();
    let [a, b, c] = output.all_planes_mut().map(Plane::as_mut_slice);
    let mut bands: Vec<_> = a
        .chunks_mut(band_rows * w)
        .zip(b.chunks_mut(band_rows * w))
        .zip(c.chunks_mut(band_rows * w))
        .zip(work)
        .enumerate()
        .map(|(i, (((a, b), c), work))| (i * band_rows, [a, b, c], work))
        .collect();
    pool.steal_for_each_mut(scratch, &mut bands, |_, (first, out, work), _| {
        let first = *first;
        let last = first + out[0].len() / w;
        // Each earlier stage starts its later stage's border above the band.
        let mut begin = [0; 3];
        let mut lo = first;
        for s in (0..n).rev() {
            begin[s] = lo;
            lo = lo.saturating_sub(chain.stages[s].0.border);
        }
        let mut band = Band {
            chain,
            rings: &mut work.rings,
            begin,
            next: begin,
            first,
            out: out.each_mut().map(|p| &mut **p),
        };
        for strip in bounds.windows(2) {
            let spans = spans(chain.stages, w, strip[0], strip[1]);
            for ((rings, &(stage, _)), span) in band.rings.iter_mut().zip(chain.stages).zip(spans) {
                rings.prepare(stage, span);
            }
            band.next = begin;
            for y in first..last {
                band.produce(n - 1, y);
            }
        }
    });
}

#[cfg(test)]
impl Scratch {
    pub(super) fn floats(&self) -> usize {
        self.rings
            .iter()
            .flat_map(|r| r.pixels.iter().chain(&r.difference).chain(&r.sad))
            .map(Vec::capacity)
            .sum()
    }
}
