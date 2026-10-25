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
//! SSIMULACRA 2 of XYB images for the closed loop, on the `ssimu2-lite`
//! crate: the source is analysed once, candidates are scored in full or by
//! region-incremental updates in linear RGB.
use crate::image::Image3F;
use ssimu2_lite::{IncrementalState, PreparedReference, RgbImage};

pub(crate) use ssimu2_lite::{Checkpoint as Ssimulacra2Checkpoint, Rect};

/// Linear RGB planes of an XYB image (the encoder's inverse opsin transform).
pub(crate) fn linear_of_xyb(img: &Image3F) -> [Vec<f32>; 3] {
    let (w, h) = (img.xsize(), img.ysize());
    let mut out = [vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]];
    let [r, g, b] = &mut out;
    for y in 0..h {
        let span = y * w..(y + 1) * w;
        crate::xyb::xyb_to_rgb_rows(
            &crate::xyb::XybMatrix::SPEC,
            [
                img.plane_row(0, y),
                img.plane_row(1, y),
                img.plane_row(2, y),
            ],
            [&mut r[span.clone()], &mut g[span.clone()], &mut b[span]],
        );
    }
    out
}

fn planes(linear: &[Vec<f32>; 3], w: usize, h: usize) -> RgbImage<'_> {
    RgbImage::linear_rgb([&linear[0], &linear[1], &linear[2]], w, h, w)
        .expect("frame-sized linear planes")
}

/// The source frame, analyzed once.
pub(crate) struct Ssimulacra2Reference {
    prepared: PreparedReference,
    width: usize,
    height: usize,
}

impl Ssimulacra2Reference {
    /// `None` below the metric's 8x8 minimum.
    pub(crate) fn new(source: &Image3F) -> Option<Self> {
        let (width, height) = (source.xsize(), source.ysize());
        let linear = linear_of_xyb(source);
        let prepared = PreparedReference::new(planes(&linear, width, height)).ok()?;
        Some(Self {
            prepared,
            width,
            height,
        })
    }

    /// `100 - SSIMULACRA2` of `recon` and the per-pixel map whose mean is
    /// the (linearized) deficit. A non-finite reconstruction scores NaN.
    pub(crate) fn deficit_and_map(&self, recon: &Image3F) -> (f32, Vec<f32>) {
        let linear = linear_of_xyb(recon);
        match self
            .prepared
            .compare_with_map(planes(&linear, self.width, self.height))
        {
            Ok((score, map)) => ((100.0 - score) as f32, map),
            Err(_) => (f32::NAN, vec![0.0; self.width * self.height]),
        }
    }

    /// The identical pooled score when no per-pixel map is consumed.
    pub(crate) fn deficit(&self, recon: &Image3F) -> f32 {
        let linear = linear_of_xyb(recon);
        self.prepared
            .compare(planes(&linear, self.width, self.height))
            .map_or(f32::NAN, |score| (100.0 - score) as f32)
    }

    /// Incremental state following `recon`.
    pub(crate) fn state(&self, recon: &Image3F) -> Ssimulacra2State {
        let linear = linear_of_xyb(recon);
        let state = self
            .prepared
            .incremental(planes(&linear, self.width, self.height))
            .expect("finite reconstruction");
        Ssimulacra2State {
            state,
            linear,
            width: self.width,
            height: self.height,
        }
    }
}

/// Incremental SSIMULACRA 2 of a fixed source against a mutable reconstruction.
pub(crate) struct Ssimulacra2State {
    state: IncrementalState,
    /// Linear RGB staging of the reconstruction; an update converts and reads
    /// only its rect, so rolled-back pixels elsewhere are never read.
    linear: [Vec<f32>; 3],
    width: usize,
    height: usize,
}

impl Ssimulacra2State {
    /// `100 - score`.
    pub(crate) fn deficit(&self) -> f32 {
        (100.0 - self.state.score()) as f32
    }

    /// Refresh the linear RGB staging inside the clipped rectangle.
    fn write_linear(&mut self, recon: &Image3F, r: Rect) -> Option<Rect> {
        let (w, h) = (self.width, self.height);
        let r = Rect::new(r.x0, r.y0, r.x1.min(w), r.y1.min(h));
        if r.x1 <= r.x0 || r.y1 <= r.y0 {
            return None;
        }
        let [lr, lg, lb] = &mut self.linear;
        for y in r.y0..r.y1 {
            let span = y * w + r.x0..y * w + r.x1;
            crate::xyb::xyb_to_rgb_rows(
                &crate::xyb::XybMatrix::SPEC,
                [
                    &recon.plane_row(0, y)[r.x0..r.x1],
                    &recon.plane_row(1, y)[r.x0..r.x1],
                    &recon.plane_row(2, y)[r.x0..r.x1],
                ],
                [&mut lr[span.clone()], &mut lg[span.clone()], &mut lb[span]],
            );
        }
        Some(r)
    }

    /// Commit a reconstruction change without saving rollback data.
    pub(crate) fn update(&mut self, recon: &Image3F, r: Rect) {
        let Some(r) = self.write_linear(recon, r) else {
            return;
        };
        let result = self
            .state
            .update(planes(&self.linear, self.width, self.height), r);
        debug_assert!(result.is_ok(), "non-finite reconstruction");
    }

    /// The reconstruction changed inside `r`: refresh, saving what changes
    /// into `cp` for `rollback`.
    pub(crate) fn update_checkpointed(
        &mut self,
        recon: &Image3F,
        r: Rect,
        cp: &mut Ssimulacra2Checkpoint,
    ) {
        let Some(r) = self.write_linear(recon, r) else {
            return;
        };
        let result =
            self.state
                .update_checkpointed(planes(&self.linear, self.width, self.height), r, cp);
        debug_assert!(result.is_ok(), "non-finite reconstruction");
    }

    /// Undo the `update_checkpointed` that filled `cp`.
    pub(crate) fn rollback(&mut self, cp: &Ssimulacra2Checkpoint) {
        self.state.rollback(cp);
    }

    /// Per-pixel linearized map (mean = deficit).
    pub(crate) fn map(&self) -> Vec<f32> {
        self.state.error_map()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(seed: u32, w: usize, h: usize) -> Image3F {
        let mut img = Image3F::new(w, h);
        let mut rng = seed;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (rng >> 24) as f32 / 255.0;
                    let base = if c == 1 { 0.3 } else { 0.0 };
                    img.plane_row_mut(c, y)[x] = base
                        + 0.3 * ((x / 8 + y / 8) % 2) as f32 * (if c == 1 { 1.0 } else { 0.1 })
                        + 0.05 * noise;
                }
            }
        }
        img
    }

    #[test]
    fn score_only_preserves_the_map_scores_exact_bits() {
        for (w, h) in [(8, 8), (9, 17), (161, 97)] {
            let source = textured(7, w, h);
            let reference = Ssimulacra2Reference::new(&source).unwrap();
            for seed in [7, 8, 19] {
                let recon = textured(seed, w, h);
                assert_eq!(
                    reference.deficit(&recon).to_bits(),
                    reference.deficit_and_map(&recon).0.to_bits(),
                    "{w}x{h}, seed={seed}",
                );
            }
        }
    }

    #[test]
    fn incremental_updates_and_rollback_track_full_scores() {
        let (w, h) = (161, 97);
        let src = textured(7, w, h);
        let mut rec = textured(8, w, h);
        let reference = Ssimulacra2Reference::new(&src).unwrap();
        let mut state = reference.state(&rec);
        let (d0, map0) = reference.deficit_and_map(&rec);
        assert!((state.deficit() - d0).abs() < 1e-3);
        assert!(
            map0.iter()
                .zip(state.map())
                .all(|(a, b)| (a - b).abs() < 1e-3)
        );
        let r = Rect::new(40, 30, 90, 70);
        for y in r.y0..r.y1 {
            for v in &mut rec.plane_row_mut(1, y)[r.x0..r.x1] {
                *v += 0.02;
            }
        }
        let mut cp = Ssimulacra2Checkpoint::default();
        state.update_checkpointed(&rec, r, &mut cp);
        let (d1, _) = reference.deficit_and_map(&rec);
        assert!(
            (state.deficit() - d1).abs() < 1e-3,
            "{} vs {d1}",
            state.deficit()
        );
        assert!(d1 > d0);
        state.rollback(&cp);
        assert!((state.deficit() - d0).abs() < 1e-6);
    }

    #[test]
    fn committed_updates_match_checkpointed_updates_and_later_rollbacks() {
        let (w, h) = (65, 49);
        let source = textured(7, w, h);
        let mut recon = textured(8, w, h);
        let reference = Ssimulacra2Reference::new(&source).unwrap();
        let mut committed = reference.state(&recon);
        let mut checkpointed = reference.state(&recon);
        let mut cp = Ssimulacra2Checkpoint::default();
        let mut trial_cp = Ssimulacra2Checkpoint::default();
        for r in [
            Rect::new(7, 9, 24, 31),
            Rect::new(64, 48, 65, 49),
            Rect::new(0, 0, w + 5, h + 3),
            Rect::new(16, 8, 16, 8),
        ] {
            for c in 0..3 {
                for y in r.y0..r.y1.min(h) {
                    for v in &mut recon.plane_row_mut(c, y)[r.x0..r.x1.min(w)] {
                        *v += (c + 1) as f32 * 0.0007;
                    }
                }
            }
            committed.update(&recon, r);
            checkpointed.update_checkpointed(&recon, r, &mut cp);
            let base = committed.deficit().to_bits();
            assert_eq!(base, checkpointed.deficit().to_bits());
            assert_eq!(committed.linear, checkpointed.linear);
            assert_eq!(committed.map(), checkpointed.map());

            let mut trial = recon.clone();
            trial.plane_row_mut(1, 17)[23] += 0.003;
            let rect = Rect::new(23, 17, 24, 18);
            committed.update_checkpointed(&trial, rect, &mut trial_cp);
            checkpointed.update_checkpointed(&trial, rect, &mut cp);
            assert_eq!(
                committed.deficit().to_bits(),
                checkpointed.deficit().to_bits()
            );
            committed.rollback(&trial_cp);
            checkpointed.rollback(&cp);
            assert_eq!(committed.deficit().to_bits(), base);
            assert_eq!(checkpointed.deficit().to_bits(), base);
            assert_eq!(committed.map(), checkpointed.map());
        }
    }

    #[test]
    fn identical_frames_have_no_deficit() {
        let src = textured(3, 64, 48);
        let reference = Ssimulacra2Reference::new(&src).unwrap();
        let (d, map) = reference.deficit_and_map(&src);
        assert!(d.abs() < 1e-4 && map.iter().all(|v| v.abs() < 1e-6));
        assert!(Ssimulacra2Reference::new(&textured(3, 7, 48)).is_none());
    }
}
