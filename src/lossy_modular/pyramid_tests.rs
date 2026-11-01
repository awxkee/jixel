/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! Tests for the plain-layout pyramids and their calibration.

use super::calibrate::{interpolate_knots, plain_distance};
use super::layout::Layout;
use super::pyramid::{PlainInput, Pyramid};

use crate::coding::CodingTransform;

fn ramp(w: usize, h: usize, seed: i32) -> Vec<i32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            ((x * 523 + y * 311 + seed * 97) % 65536).max(0)
        })
        .collect()
}

fn input(coding: CodingTransform, color: usize, extra: Vec<(Vec<i32>, f32)>) -> PlainInput {
    PlainInput {
        color: (0..color).map(|c| ramp(64, 48, c as i32)).collect(),
        extra,
        max: 65535.0,
        coding,
    }
}

#[test]
fn ycbcr_layout_pre_squeezes_the_outer_planes() {
    let q = Pyramid::plain(&input(CodingTransform::YCbCr, 3, Vec::new()), 64, 48, 8.0).unwrap();
    assert_eq!(
        q.layout,
        Layout::Plain {
            ycbcr: true,
            rct: false,
            extra: 0
        }
    );
    let pre: Vec<(usize, usize)> = q
        .steps
        .iter()
        .filter(|s| !s.in_place)
        .map(|s| (s.begin_c, s.num_c))
        .collect();
    assert_eq!(pre, [(0, 1), (0, 1), (2, 1), (2, 1)]);
    assert!(q.steps.iter().filter(|s| s.in_place).all(|s| s.num_c == 3));
    // The chroma pre-squeeze residuals come last and sit on the coarser
    // chroma schedule.
    let nb = q.channels.len();
    assert!(q.quants[nb - 1] > q.quants[nb - 1 - 4]);
    assert!(q.quants.iter().all(|&v| v >= 1));
}

#[test]
fn rgb_layout_uses_the_rct_and_gray_neither() {
    let q = Pyramid::plain(&input(CodingTransform::Rgb, 3, Vec::new()), 64, 48, 8.0).unwrap();
    assert_eq!(
        q.layout,
        Layout::Plain {
            ycbcr: false,
            rct: true,
            extra: 0
        }
    );
    let pre: Vec<usize> = q
        .steps
        .iter()
        .filter(|s| !s.in_place)
        .map(|s| s.begin_c)
        .collect();
    assert_eq!(pre, [1, 1, 2, 2]);
    let g = Pyramid::plain(&input(CodingTransform::Rgb, 1, Vec::new()), 64, 48, 8.0).unwrap();
    assert_eq!(
        g.layout,
        Layout::Plain {
            ycbcr: false,
            rct: false,
            extra: 0
        }
    );
    assert!(g.steps.iter().all(|s| s.in_place && s.num_c == 1));
}

#[test]
fn extra_channels_follow_their_own_distance() {
    let lossless = (ramp(64, 48, 7), 0.0);
    let lossy = (ramp(64, 48, 9), 40.0);
    let q = Pyramid::plain(
        &input(CodingTransform::YCbCr, 3, vec![lossless, lossy]),
        64,
        48,
        8.0,
    )
    .unwrap();
    assert_eq!(
        q.layout,
        Layout::Plain {
            ycbcr: true,
            rct: false,
            extra: 2
        }
    );
    assert!(q.steps.iter().filter(|s| s.in_place).all(|s| s.num_c == 5));
    let nb = q.channels.len();
    // Channel i % 5: 3 = the lossless extra (step 1 everywhere), 4 = the
    // lossy one (coarser than luma at the same level).
    for i in 0..nb - 4 {
        match i % 5 {
            3 => assert_eq!(q.quants[i], 1, "lossless extra at {i}"),
            4 => assert!(q.quants[i] >= q.quants[i - 3], "lossy extra at {i}"),
            _ => {}
        }
    }
    // Tiny frames have no pyramid.
    assert!(Pyramid::plain(&input(CodingTransform::YCbCr, 3, Vec::new()), 8, 8, 8.0).is_none());
}

#[test]
fn plain_distance_is_monotonic_and_flat_outside_the_knots() {
    for coding in [CodingTransform::YCbCr, CodingTransform::Rgb] {
        let mut last = 0.0;
        for d in [0.1, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 10.0] {
            let e = plain_distance(coding, d);
            assert!(e > last, "{coding:?} d={d}");
            assert!(
                e > d,
                "{coding:?} d={d}: the arm quantizes coarser than its distance"
            );
            last = e;
        }
        assert_eq!(
            plain_distance(coding, 0.5) * 2.0,
            plain_distance(coding, 1.0)
        );
    }
    assert_eq!(interpolate_knots(&[(1.0, 2.0), (3.0, 4.0)], 2.0), 3.0);
}
