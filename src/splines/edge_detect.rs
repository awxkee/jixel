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

//! Source-luma step contours for bounded paired edge proposals.
//!
//! A small separable blur, central gradient and subpixel nonmaximum suppression
//! trace coherent edges with signed dark-to-bright normals. Strong gradients
//! propose geometry only; the native coefficient selector decides coding value.

use super::Point;
use super::filter::FilterPlan;
use crate::coder_scratch::CoderScratch;
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;

const HIGH: f32 = 0.012;
const LOW: f32 = 0.004;
const FRAME_MARGIN: usize = 4;
const MIN_POINTS: usize = 20;
const MIN_NORMAL_COS: f32 = 0.65;
// A 90-degree corner can rotate gradually across a few blurred pixels. Check
// a short history as well as adjacent normals so it still splits there.
const BEND_LOOKBACK: usize = 4;
const MIN_BEND_COS: f32 = 0.6;
const MIN_FORWARD_COS: f32 = 0.35;
const NO_PIXEL: usize = usize::MAX;
static NEIGHBORS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Points are ordered along the contour, with roughly one sample per pixel.
/// Unit normals always point from lower to higher source luma. Their signed
/// local coherence is enforced during tracing; they are not unoriented axes.
pub(super) struct Contour {
    pub(super) points: Vec<Point<f32>>,
    pub(super) normals: Vec<Point<f32>>,
    /// Mean source-luma gradient magnitude, before nonmaximum suppression.
    pub(super) strength: f32,
}

#[derive(Clone, Copy)]
struct EdgePixel {
    position: Point<f32>,
    normal: Point<f32>,
    strength: f32,
}

/// Each edge pixel's 8-neighbors, in `NEIGHBORS` order, that pass the walk's
/// state-free tests (an edge pixel with a coherent normal, not on top of it).
/// The walk visits seeds in strength order, all over the frame; this keeps it
/// off the full-frame lookup map.
struct Links {
    start: Vec<usize>,
    next: Vec<usize>,
}

impl Links {
    #[inline]
    fn of(&self, pixel: usize) -> &[usize] {
        &self.next[self.start[pixel]..self.start[pixel + 1]]
    }
}

/// Whether the walk may step from `here` to `candidate` for some state.
#[inline]
fn linkable(here: &EdgePixel, candidate: &EdgePixel) -> bool {
    let delta = Point {
        x: candidate.position.x - here.position.x,
        y: candidate.position.y - here.position.y,
    };
    dot(here.normal, candidate.normal) >= MIN_NORMAL_COS
        && (delta.x * delta.x + delta.y * delta.y).sqrt() >= 0.1
}

#[inline]
fn dot(a: Point<f32>, b: Point<f32>) -> f32 {
    a.x * b.x + a.y * b.y
}

#[inline]
fn bilinear(data: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let a = data[y0 * w + x0] * (1.0 - fx) + data[y0 * w + x1] * fx;
    let b = data[y1 * w + x0] * (1.0 - fx) + data[y1 * w + x1] * fx;
    a * (1.0 - fy) + b * fy
}

/// Greedy tangent-following walk. The two directions from a seed share the
/// visited map. Ambiguous diverging continuations terminate a chain, rather
/// than allowing a spline to cut through a junction or a sharp corner.
fn walk(
    pixels: &[EdgePixel],
    links: &Links,
    visited: &mut [bool],
    seed: usize,
    mut tangent: Point<f32>,
    chain: &mut Vec<usize>,
) {
    chain.clear();
    let mut current = seed;
    let mut history = [pixels[seed].normal; BEND_LOOKBACK];
    let mut history_index = 0;
    loop {
        let here = pixels[current];
        let mut choices = [(0.0, NO_PIXEL, Point { x: 0.0, y: 0.0 }); 8];
        let mut count = 0;
        for &next in links.of(current) {
            if visited[next] {
                continue;
            }
            let candidate = pixels[next];
            let normal_cos = dot(here.normal, candidate.normal);
            if normal_cos < MIN_NORMAL_COS
                || dot(history[history_index], candidate.normal) < MIN_BEND_COS
            {
                continue;
            }
            let delta = Point {
                x: candidate.position.x - here.position.x,
                y: candidate.position.y - here.position.y,
            };
            let length = (delta.x * delta.x + delta.y * delta.y).sqrt();
            if length < 0.1 {
                continue;
            }
            let direction = Point {
                x: delta.x / length,
                y: delta.y / length,
            };
            let forward = dot(tangent, direction);
            if forward < MIN_FORWARD_COS {
                continue;
            }
            // Subpixel positions largely remove the staircase preference of
            // integer 8-neighbor directions. Ties retain raster-neighbor order.
            let score = 0.7 * forward
                + 0.2 * normal_cos
                + 0.1 * (candidate.strength / here.strength).min(1.0);
            choices[count] = (score, next, direction);
            count += 1;
        }
        let choices = &choices[..count];
        // The first highest score wins ties, as in a stable descending sort.
        let Some(best) = (0..count).reduce(|best, k| {
            if choices[k].0.total_cmp(&choices[best].0).is_gt() {
                k
            } else {
                best
            }
        }) else {
            break;
        };
        let (best_score, next, direction) = choices[best];
        if choices.iter().enumerate().any(|(k, &(score, _, other))| {
            k != best && score >= best_score - 0.08 && dot(direction, other) < 0.5
        }) {
            break;
        }
        visited[next] = true;
        chain.push(next);
        current = next;
        let normal = pixels[next].normal;
        history[history_index] = normal;
        history_index = (history_index + 1) % BEND_LOOKBACK;
        tangent = Point {
            x: -normal.y,
            y: normal.x,
        };
        if dot(tangent, direction) < 0.0 {
            tangent.x = -tangent.x;
            tangent.y = -tangent.y;
        }
    }
}

pub(super) fn detect(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    source: &Image3F,
    detectable: Option<&[bool]>,
    max_contours: usize,
) -> Vec<Contour> {
    let (w, h) = (source.xsize(), source.ysize());
    if max_contours == 0 || w <= 2 * FRAME_MARGIN || h <= 2 * FRAME_MARGIN {
        return Vec::new();
    }
    let kernel = [0.25, 0.5, 0.25];
    let mut rows = vec![0.0; w * h];
    let mut smooth = vec![0.0; w * h];
    FilterPlan::new(w, &kernel).horizontal(ctx, scratch, source.plane_data(1), &mut rows);
    FilterPlan::new(h, &kernel).vertical(ctx, scratch, &rows, &mut smooth);
    drop(rows);
    let mut magnitude = vec![0.0; w * h];
    let rows = smooth.chunks_exact(w);

    for (((top, mid), bottom), out) in rows
        .clone()
        .zip(rows.clone().skip(1))
        .zip(rows.skip(2))
        .zip(magnitude.chunks_exact_mut(w).skip(1))
    {
        for ((px, (&up, &down)), mag) in mid
            .array_windows::<3>()
            .zip(top.iter().zip(bottom).skip(1))
            .zip(out.iter_mut().skip(1))
        {
            let gx = 0.5 * (px[2] - px[0]);
            let gy = 0.5 * (down - up);
            *mag = (gx * gx + gy * gy).sqrt();
        }
    }

    // Nonmaximum suppression reads only fixed neighborhoods, so row bands run
    // in parallel and are concatenated in raster order.
    const BAND: usize = 64;
    let rows = FRAME_MARGIN..h - FRAME_MARGIN;
    let bands = rows.len().div_ceil(BAND);
    let found = ctx.thread_pool.steal_map(scratch, bands, |band, _| {
        let mut out = Vec::new();
        let y0 = rows.start + band * BAND;
        for y in y0..(y0 + BAND).min(rows.end) {
            for x in FRAME_MARGIN..w - FRAME_MARGIN {
                let i = y * w + x;
                let strength = magnitude[i];
                if !strength.is_finite() || strength < LOW || detectable.is_some_and(|d| !d[i]) {
                    continue;
                }
                let normal = Point {
                    x: 0.5 * (smooth[i + 1] - smooth[i - 1]) / strength,
                    y: 0.5 * (smooth[i + w] - smooth[i - w]) / strength,
                };
                let a = bilinear(&magnitude, w, h, x as f32 + normal.x, y as f32 + normal.y);
                let b = bilinear(&magnitude, w, h, x as f32 - normal.x, y as f32 - normal.y);
                // One strict comparison selects one side of a two-pixel plateau.
                // The parabolic offset then recovers its halfway edge position.
                if strength < a || strength <= b {
                    continue;
                }
                let curvature = a + b - 2.0 * strength;
                let offset = if curvature < -1e-12 {
                    (0.5 * (b - a) / curvature).clamp(-0.75, 0.75)
                } else {
                    0.0
                };
                out.push((
                    i,
                    EdgePixel {
                        position: Point {
                            x: x as f32 + offset * normal.x,
                            y: y as f32 + offset * normal.y,
                        },
                        normal,
                        strength,
                    },
                ));
            }
        }
        out
    });
    let mut lookup = vec![NO_PIXEL; w * h];
    // Edge pixels in raster order, so their indices order like positions.
    let count = found.iter().map(Vec::len).sum();
    let mut pixels = Vec::with_capacity(count);
    let mut at = Vec::with_capacity(count);
    // Seeds strongest first, ties in raster order. Strengths are positive and
    // finite, so their bit patterns order like the values.
    let mut seeds: Vec<u64> = Vec::new();
    for (i, pixel) in found.into_iter().flatten() {
        let index = pixels.len();
        lookup[i] = index;
        at.push(i);
        if pixel.strength >= HIGH {
            seeds.push(u64::from(!pixel.strength.to_bits()) << 32 | index as u64);
        }
        pixels.push(pixel);
    }
    // Every detected point is four pixels inside the frame.
    const LINK_CHUNK: usize = 4096;
    let chunks =
        ctx.thread_pool
            .steal_map(scratch, pixels.len().div_ceil(LINK_CHUNK), |chunk, _| {
                let range = chunk * LINK_CHUNK..((chunk + 1) * LINK_CHUNK).min(pixels.len());
                let mut counts = Vec::with_capacity(range.len());
                let mut next = Vec::new();
                for (here, &i) in pixels[range.clone()].iter().zip(&at[range]) {
                    let (x, y) = (i % w, i / w);
                    let before = next.len();
                    for (dx, dy) in NEIGHBORS {
                        let n = lookup[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                        if n != NO_PIXEL && linkable(here, &pixels[n]) {
                            next.push(n);
                        }
                    }
                    counts.push(next.len() - before);
                }
                (counts, next)
            });
    drop(lookup);
    drop(at);
    let mut links = Links {
        start: Vec::with_capacity(pixels.len() + 1),
        next: Vec::with_capacity(chunks.iter().map(|c| c.1.len()).sum()),
    };
    links.start.push(0);
    for (counts, next) in chunks {
        for count in counts {
            links.start.push(links.start.last().unwrap() + count);
        }
        links.next.extend(next);
    }
    drop(smooth);
    drop(magnitude);
    seeds.sort_unstable();
    let mut visited = vec![false; pixels.len()];
    // Kept chain pieces as (strength, first index, range in `members`); only the
    // strongest are turned into contours. Most seeds in texture end after a
    // step or two, so the walk buffers are reused.
    let mut pieces: Vec<(f32, usize, std::ops::Range<usize>)> = Vec::new();
    let mut members: Vec<usize> = Vec::new();
    let (mut forward, mut chain) = (Vec::new(), Vec::new());
    for seed in seeds {
        let seed = (seed & u64::from(u32::MAX)) as usize;
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        let normal = pixels[seed].normal;
        walk(
            &pixels,
            &links,
            &mut visited,
            seed,
            Point {
                x: -normal.y,
                y: normal.x,
            },
            &mut forward,
        );
        walk(
            &pixels,
            &links,
            &mut visited,
            seed,
            Point {
                x: normal.y,
                y: -normal.x,
            },
            &mut chain,
        );
        // Every piece is a part of the joined chain.
        if chain.len() + 1 + forward.len() < MIN_POINTS {
            continue;
        }
        chain.reverse();
        chain.push(seed);
        chain.extend_from_slice(&forward);
        // The seed can itself sit on the center of a corner. Each directional
        // walk may then rotate by only 45 degrees while their concatenation
        // crosses the full 90. Remove those short bend neighborhoods as well.
        let mut emit = |indices: &[usize]| {
            if indices.len() < MIN_POINTS {
                return;
            }
            let strength = indices.iter().map(|&i| pixels[i].strength).sum::<f32>();
            let start = members.len();
            members.extend_from_slice(indices);
            pieces.push((strength, indices[0], start..members.len()));
        };
        let mut start = 0;
        for i in 0..chain.len() {
            let left = chain[i.saturating_sub(BEND_LOOKBACK)];
            let right = chain[(i + BEND_LOOKBACK).min(chain.len() - 1)];
            if dot(pixels[left].normal, pixels[right].normal) < MIN_BEND_COS {
                emit(&chain[start..i]);
                start = i + 1;
            }
        }
        emit(&chain[start..]);
    }
    // Favor persistent supported edges, rather than very short high-contrast
    // fragments. The full discovery pass is linear apart from seed sorting.
    pieces.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    pieces.truncate(max_contours);
    pieces
        .into_iter()
        .map(|(strength, _, range)| {
            let indices = &members[range];
            Contour {
                points: indices.iter().map(|&i| pixels[i].position).collect(),
                normals: indices.iter().map(|&i| pixels[i].normal).collect(),
                strength: strength / indices.len() as f32,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_with(mut value: impl FnMut(f32, f32) -> f32) -> Image3F {
        let mut image = Image3F::new(160, 144);
        for y in 0..144 {
            for x in 0..160 {
                image.plane_row_mut(1, y)[x] = value(x as f32, y as f32);
            }
        }
        image
    }

    fn contours(image: &Image3F, budget: usize) -> Vec<Contour> {
        detect(
            &EncodingContext::default(),
            &mut CoderScratch::default(),
            image,
            None,
            budget,
        )
    }

    fn validate(contours: &[Contour]) {
        for contour in contours {
            assert_eq!(contour.points.len(), contour.normals.len());
            assert!(contour.points.len() >= MIN_POINTS);
            assert!(contour.strength >= LOW);
            for normal in &contour.normals {
                assert!((dot(*normal, *normal) - 1.0).abs() < 1e-5);
            }
            for normals in contour.normals.windows(2) {
                assert!(dot(normals[0], normals[1]) >= MIN_NORMAL_COS);
            }
            for point in &contour.points {
                assert!(point.x >= 3.0 && point.y >= 3.0);
                assert!(point.x <= 157.0 && point.y <= 141.0);
            }
        }
    }

    #[test]
    fn constant_source_has_no_contours() {
        assert!(contours(&image_with(|_, _| 0.5), 8).is_empty());
    }

    #[test]
    fn horizontal_step_has_subpixel_positions_and_signed_normals() {
        for sign in [-1.0, 1.0] {
            let source = image_with(|_, y| 0.5 + sign * if y >= 72.0 { 0.3 } else { -0.3 });
            let found = contours(&source, 8);
            validate(&found);
            let longest = found.iter().max_by_key(|c| c.points.len()).unwrap();
            assert!(longest.points.len() >= 140);
            for (point, normal) in longest.points.iter().zip(&longest.normals) {
                assert!((point.y - 71.5).abs() < 1e-5, "y={}", point.y);
                assert!(normal.x.abs() < 1e-5 && normal.y * sign > 0.99);
            }
        }
    }

    #[test]
    fn diagonal_step_has_a_long_coherent_contour() {
        let source = image_with(|x, y| if y > 33.3 + 0.43 * x { 0.8 } else { 0.2 });
        let found = contours(&source, 8);
        validate(&found);
        let longest = found.iter().max_by_key(|c| c.points.len()).unwrap();
        assert!(
            longest.points.len() >= 120,
            "{} points",
            longest.points.len()
        );
        for (point, normal) in longest.points.iter().zip(&longest.normals) {
            assert!((point.y - 33.3 - 0.43 * point.x).abs() < 0.8);
            assert!(normal.y > 0.75 && normal.x < -0.1);
        }
    }

    #[test]
    fn curved_step_remains_coherent_and_budget_is_deterministic() {
        let source = image_with(|x, y| {
            let radius = ((x - 80.0).powi(2) + (y - 72.0).powi(2)).sqrt();
            0.5 + 0.3 * ((radius - 45.0) * 1.5).tanh()
        });
        let first = contours(&source, 2);
        let second = contours(&source, 2);
        validate(&first);
        assert!(!first.is_empty() && first.len() <= 2);
        assert!(first.iter().any(|c| c.points.len() >= 60));
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second) {
            assert_eq!(a.points, b.points);
            assert_eq!(a.normals, b.normals);
            for (point, normal) in a.points.iter().zip(&a.normals) {
                let radial = Point {
                    x: point.x - 80.0,
                    y: point.y - 72.0,
                };
                assert!(dot(radial, *normal) > 40.0);
            }
        }
        assert!(contours(&source, 0).is_empty());
    }

    #[test]
    fn right_angle_step_splits_before_following_both_sides() {
        let source = image_with(|x, y| if x >= 40.0 && y >= 40.0 { 0.8 } else { 0.2 });
        let found = contours(&source, 8);
        validate(&found);
        assert!(found.len() >= 2);
        for contour in found {
            let horizontal = contour.normals.iter().any(|normal| normal.y > 0.99);
            let vertical = contour.normals.iter().any(|normal| normal.x > 0.99);
            assert!(
                !(horizontal && vertical),
                "a contour crossed the sharp corner"
            );
        }
    }
}
