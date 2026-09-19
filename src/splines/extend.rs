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

//! Recover long, fading continuations without consuming the original traces.
//! A short longitudinal average is matched to a Gaussian cross-section after
//! removing a constant and a linear background. Only bounded additional curves
//! are sent to fitting; all original candidates keep their original targets.

use super::detect::Chain;
use super::{Point, fast_hypot};
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

const CHUNK: usize = 8;
const MAX_CHUNKS: usize = 32;
const MAX_PROPOSALS: usize = 32;
const PROFILE_FLOOR: f32 = 0.003;
const MIN_NOVEL: usize = 24;
const SHIFTS: [f32; 5] = [0.0, -0.5, 0.5, -1.0, 1.0];
const G: [f32; 7] = [
    -0.3468838433,
    -0.2226575566,
    0.2485378199,
    0.6420071601,
    0.2485378199,
    -0.2226575566,
    -0.3468838433,
];

#[inline]
fn dot(a: Point<f32>, b: Point<f32>) -> f32 {
    a.x * b.x + a.y * b.y
}
#[inline]
fn offset(p: Point<f32>, t: Point<f32>, d: f32) -> Point<f32> {
    Point::new(p.x + t.x * d, p.y + t.y * d)
}
#[inline]
fn normal(t: Point<f32>) -> Point<f32> {
    Point::new(-t.y, t.x)
}
#[inline]
fn sample(data: &[f32], w: usize, h: usize, p: Point<f32>) -> f32 {
    let x = p.x.clamp(0.0, (w - 1) as f32);
    let y = p.y.clamp(0.0, (h - 1) as f32);
    let (ix, iy) = (x as usize, y as usize);
    let (jx, jy) = ((ix + 1).min(w - 1), (iy + 1).min(h - 1));
    let (fx, fy) = (x - ix as f32, y - iy as f32);
    let a = data[iy * w + ix] * (1.0 - fx) + data[iy * w + jx] * fx;
    let b = data[jy * w + ix] * (1.0 - fx) + data[jy * w + jx] * fx;
    a * (1.0 - fy) + b * fy
}
fn profile(data: &[f32], w: usize, h: usize, p: Point<f32>, n: Point<f32>, sigma: f32) -> [f32; 7] {
    std::array::from_fn(|i| sample(data, w, h, offset(p, n, (i as f32 - 3.0) * sigma)))
}
fn amplitude(values: &[f32; 7]) -> f32 {
    values.iter().zip(G).map(|(&v, g)| v * g).sum::<f32>() / G.iter().map(|g| g * g).sum::<f32>()
}
fn explained(values: &[f32; 7]) -> f32 {
    let dc = values.iter().sum::<f32>() / 7.0;
    let slope = values
        .iter()
        .enumerate()
        .map(|(i, &v)| v * (i as f32 - 3.0))
        .sum::<f32>()
        / 28.0;
    let a = amplitude(values);
    let (mut error, mut energy) = (0.0, 0.0);
    for i in 0..7 {
        let v = values[i] - dc - slope * (i as f32 - 3.0);
        error += (v - a * G[i]).powi(2);
        energy += v * v;
    }
    1.0 - error / energy.max(1e-12)
}

struct Proposal {
    chain: Chain,
    tail_start: usize,
    novel: usize,
    supported: usize,
}

fn trace(data: &[f32], w: usize, h: usize, seed: &[Point<f32>], scale: f32) -> Option<Proposal> {
    if seed.len() < 28 {
        return None;
    }
    let mut p = *seed.last()?;
    let tail = &seed[seed.len() - 12..];
    let d = Point::new(p.x - tail[0].x, p.y - tail[0].y);
    let length = fast_hypot(d.x, d.y);
    if length < 8.0 {
        return None;
    }
    let axis = Point::new(d.x / length, d.y / length);
    let mut t = axis;
    let n = normal(t);
    if tail
        .iter()
        .any(|q| dot(Point::new(q.x - p.x, q.y - p.y), n).abs() > 1.0)
    {
        return None;
    }
    let sigma = (0.86 * scale - 0.21).max(0.7);
    let seed_amp = tail
        .iter()
        .map(|&q| amplitude(&profile(data, w, h, q, n, sigma)))
        .sum::<f32>()
        / tail.len() as f32;
    let mut reference = seed_amp.abs();
    if reference < PROFILE_FLOOR {
        return None;
    }
    let polarity = seed_amp.signum();
    // Keep the approximately straight part of the seed, with a bounded overlap.
    let mut start = seed.len() - 12;
    while start > seed.len().saturating_sub(64) {
        let q = seed[start - 1];
        if dot(Point::new(q.x - p.x, q.y - p.y), n).abs() > 1.0 {
            break;
        }
        start -= 1;
    }
    let mut extra = Vec::new();
    let mut pending = Vec::new();
    let (mut gaps, mut supported) = (0usize, 0usize);
    for _ in 0..MAX_CHUNKS {
        let n = normal(t);
        let mut best: Option<(f32, Point<f32>, f32)> = None;
        for shift in SHIFTS {
            let mut average = [0.0f32; 7];
            let mut amps = [0.0f32; CHUNK];
            let mut end = p;
            let mut inside = true;
            for k in 1..=CHUNK {
                let q = offset(offset(p, t, k as f32), n, shift * k as f32 / CHUNK as f32);
                if q.x < 4.0 || q.y < 4.0 || q.x > w as f32 - 5.0 || q.y > h as f32 - 5.0 {
                    inside = false;
                    break;
                }
                let row = profile(data, w, h, q, n, sigma);
                amps[k - 1] = polarity * amplitude(&row);
                for j in 0..7 {
                    average[j] += row[j] / CHUNK as f32;
                }
                end = q;
            }
            if !inside {
                continue;
            }
            let mean = amps.iter().sum::<f32>() / CHUNK as f32;
            let variance = amps.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / CHUNK as f32;
            if mean <= PROFILE_FLOOR.max(0.12 * reference)
                || amps.iter().filter(|&&a| a > 0.0).count() < 6
                || mean <= variance.sqrt() * 0.7
                || explained(&average) <= 0.7
            {
                continue;
            }
            let score = mean - 0.00015 * shift.abs();
            if best.as_ref().is_none_or(|b| score > b.0) {
                best = Some((score, end, mean));
            }
        }
        let Some((_, end, response)) = best else {
            gaps += 1;
            if gaps > 2
                || (extra.len() + pending.len() + CHUNK) as f32
                    > ((supported + 2) * CHUNK) as f32 / 0.8
            {
                break;
            }
            for k in 1..=CHUNK {
                pending.push(offset(p, t, k as f32));
            }
            p = offset(p, t, CHUNK as f32);
            continue;
        };
        let d = Point::new(end.x - p.x, end.y - p.y);
        let length = fast_hypot(d.x, d.y);
        let next = Point::new(d.x / length, d.y / length);
        if dot(next, axis) < 0.94 {
            break;
        }
        extra.append(&mut pending);
        for k in 1..=CHUNK {
            extra.push(offset(p, d, k as f32 / CHUNK as f32));
        }
        supported += 1;
        gaps = 0;
        t = Point::new(t.x + next.x, t.y + next.y);
        let length = fast_hypot(t.x, t.y);
        t = Point::new(t.x / length, t.y / length);
        p = end;
        reference = 0.75 * reference + 0.25 * response;
    }
    if extra.len() < 24 || ((supported * CHUNK) as f32) < 0.75 * extra.len() as f32 {
        return None;
    }
    let mut points = seed[start..].to_vec();
    let tail_start = points.len();
    points.extend(extra);
    Some(Proposal {
        chain: Chain { points, scale },
        tail_start,
        novel: 0,
        supported: supported * CHUNK,
    })
}

fn mark(occupied: &mut [bool], w: usize, h: usize, points: &[Point<f32>]) {
    for p in points {
        let (x, y) = (p.x.round() as isize, p.y.round() as isize);
        for (dx, dy) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (xx, yy) = (x + dx, y + dy);
            if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                occupied[yy as usize * w + xx as usize] = true;
            }
        }
    }
}
fn novelty(occupied: &[bool], w: usize, h: usize, points: &[Point<f32>]) -> usize {
    points
        .iter()
        .filter(|p| {
            let (x, y) = (p.x.round() as isize, p.y.round() as isize);
            x >= 0
                && y >= 0
                && x < w as isize
                && y < h as isize
                && !occupied[y as usize * w + x as usize]
        })
        .count()
}

pub(super) fn find_extensions(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    xyb: &Image3F,
    chains: &[Chain],
) -> Vec<Chain> {
    let (w, h) = (xyb.xsize(), xyb.ysize());
    let mut occupied = vec![false; w * h];
    for chain in chains {
        mark(&mut occupied, w, h, &chain.points);
    }
    let mut proposals: Vec<Proposal> = ctx
        .thread_pool
        .steal_map(scratch, chains.len() * 2, |i, _| {
            let chain = &chains[i / 2];
            let reversed;
            let seed = if i % 2 == 1 {
                reversed = chain.points.iter().copied().rev().collect::<Vec<_>>();
                &reversed
            } else {
                &chain.points
            };
            let mut proposal = trace(xyb.plane_data(1), w, h, seed, chain.scale)?;
            proposal.novel = novelty(
                &occupied,
                w,
                h,
                &proposal.chain.points[proposal.tail_start..],
            );
            (proposal.novel >= MIN_NOVEL).then_some(proposal)
        })
        .into_iter()
        .flatten()
        .collect();
    proposals.sort_by(|a, b| b.novel.cmp(&a.novel).then(b.supported.cmp(&a.supported)));
    let mut out = Vec::new();
    for proposal in proposals {
        let novel = novelty(
            &occupied,
            w,
            h,
            &proposal.chain.points[proposal.tail_start..],
        );
        if novel < MIN_NOVEL || novel * 2 < proposal.novel {
            continue;
        }
        mark(&mut occupied, w, h, &proposal.chain.points);
        out.push(proposal.chain);
        if out.len() == MAX_PROPOSALS {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        polarity: f32,
        gap: std::ops::Range<usize>,
        end: usize,
        parallel: bool,
    ) -> (Vec<f32>, Vec<Point<f32>>) {
        let mut data = vec![0.0f32; 256 * 128];
        for y in 0..128 {
            for x in 0..256 {
                let faded = (1.0 - (x as f32 - 64.0).max(0.0) / 72.0).clamp(0.0, 1.0);
                let amplitude = if x < end && !gap.contains(&x) {
                    0.008 + 0.072 * faded
                } else {
                    0.0
                };
                let line = polarity * amplitude * (-0.5 * (y as f32 - 64.0).powi(2)).exp();
                let other = if parallel && x >= 112 {
                    polarity * 0.08 * (-0.5 * (y as f32 - 69.0).powi(2)).exp()
                } else {
                    0.0
                };
                data[y * 256 + x] = 0.35 + 0.0003 * x as f32 + 0.0005 * y as f32 + line + other;
            }
        }
        let seed = (16..=48).map(|x| Point::new(x as f32, 64.0)).collect();
        (data, seed)
    }

    #[test]
    fn matched_profile_removes_constant_and_linear_background() {
        let g: [f32; 7] = std::array::from_fn(|i| (-0.5 * (i as f32 - 3.0).powi(2)).exp());
        for slope in [-0.2, 0.0, 0.2] {
            for a in [-0.1, 0.1] {
                let p = std::array::from_fn(|i| 0.4 + slope * (i as f32 - 3.0) + a * g[i]);
                assert!((amplitude(&p) - a).abs() < 1e-6);
                assert!(explained(&p) > 0.99999);
            }
        }
    }

    #[test]
    fn follows_fading_bright_and_dark_lines_through_short_occlusion() {
        for polarity in [-1.0, 1.0] {
            let (data, seed) = fixture(polarity, 112..124, 224, false);
            let p = trace(&data, 256, 128, &seed, 1.4).expect("fading continuation lost");
            assert!(p.chain.points.last().unwrap().x >= 208.0);
            assert!(
                p.chain.points.last().unwrap().x <= 232.0,
                "continued past actual line end"
            );
            assert!(p.chain.points.iter().all(|q| (q.y - 64.0).abs() < 0.6));
            assert!(p.chain.points.len() <= seed.len() + MAX_CHUNKS * CHUNK);
            assert_eq!(seed.first().unwrap().x, 16.0);
            assert_eq!(seed.last().unwrap().x, 48.0);
        }
    }

    #[test]
    fn does_not_extrapolate_seed_into_empty_affine_background() {
        let (data, seed) = fixture(1.0, 0..0, 49, false);
        assert!(trace(&data, 256, 128, &seed, 1.4).is_none());
    }

    #[test]
    fn stops_at_large_gap_and_does_not_jump_to_parallel_line() {
        for parallel in [false, true] {
            let (data, seed) = fixture(1.0, 96..160, 224, parallel);
            let p = trace(&data, 256, 128, &seed, 1.4).expect("initial visible continuation lost");
            assert!(
                p.chain.points.last().unwrap().x < 120.0,
                "crossed a long unsupported interval"
            );
            assert!(
                p.chain.points.iter().all(|q| (q.y - 64.0).abs() < 0.6),
                "switched to another line"
            );
        }
    }
}
