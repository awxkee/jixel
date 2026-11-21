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
fn lossless_extras_can_stay_out_of_the_squeeze() {
    let pool = crate::thread_pool::ThreadPool::new(1);
    let extra = vec![(ramp(64, 48, 11), 0.0)];
    let squeezed = Pyramid::plain(
        &input(CodingTransform::YCbCr, 3, extra.clone()),
        64,
        48,
        8.0,
        false,
        &pool,
    )
    .unwrap();
    let plain = Pyramid::plain(
        &input(CodingTransform::YCbCr, 3, extra),
        64,
        48,
        8.0,
        true,
        &pool,
    )
    .unwrap();
    // The extra is one full-size unit-step channel ahead of the four chroma
    // pre-squeeze residuals; the squeeze covers the color channels only.
    let nb = plain.channels.len();
    let alpha = &plain.channels[nb - 5];
    assert_eq!(
        (alpha.w, alpha.h, alpha.hshift, alpha.vshift),
        (64, 48, 0, 0)
    );
    assert_eq!(plain.quants[nb - 5], 1);
    assert_eq!(alpha.data, ramp(64, 48, 11));
    assert!(
        plain
            .steps
            .iter()
            .filter(|s| s.in_place)
            .all(|s| s.num_c == 3)
    );
    assert!(
        squeezed
            .steps
            .iter()
            .filter(|s| s.in_place)
            .all(|s| s.num_c == 4)
    );
    // Squeezed, the extra adds one residual channel per in-place step.
    let in_place = plain.steps.iter().filter(|s| s.in_place).count();
    assert_eq!(squeezed.channels.len(), plain.channels.len() + in_place);
}

#[test]
fn plain_palette_keeps_the_extra_as_indices_behind_its_meta_channel() {
    let pool = crate::thread_pool::ThreadPool::new(1);
    let levels = [0, 257 * 40, 257 * 128, 65535];
    let alpha: Vec<i32> = (0..64 * 48).map(|i| levels[(i * 7 / 64) % 4]).collect();
    for (coding, color) in [
        (CodingTransform::YCbCr, 3),
        (CodingTransform::Rgb, 3),
        (CodingTransform::Rgb, 1),
    ] {
        let source = input(coding, color, vec![(alpha.clone(), 0.0)]);
        let plain = Pyramid::plain(&source, 64, 48, 2.0, true, &pool).unwrap();
        let p = Pyramid::plain_palette(&source, 64, 48, 2.0, &pool).unwrap();
        let c = p.palette.expect("palette");
        assert_eq!(p.channels.len(), plain.channels.len() + 1);
        assert_eq!(p.quants[0], 257);
        assert_eq!(p.channels[0].data, [0, 40, 128, 255]);
        assert_eq!((p.channels[0].w, p.channels[0].h), (4, 1));
        // The color channels and their steps are the plain pyramid's.
        for (i, ch) in plain.channels.iter().enumerate().filter(|&(i, _)| i != c) {
            assert_eq!(p.channels[i + 1].data, ch.data);
            assert_eq!(p.quants[i + 1], plain.quants[i]);
        }
        let decoded: Vec<i32> = p.channels[c + 1]
            .data
            .iter()
            .map(|&k| p.channels[0].data[k as usize] * p.quants[0] as i32)
            .collect();
        assert_eq!(decoded, alpha);
        assert_eq!(plain.channels[c].data, alpha);
    }
    // Deep levels (no common step) keep their 16-bit entries.
    let deep: Vec<i32> = (0..64 * 48).map(|i| [0, 1, 40961, 65535][i % 4]).collect();
    let source = input(CodingTransform::Rgb, 3, vec![(deep.clone(), 0.0)]);
    let p = Pyramid::plain_palette(&source, 64, 48, 2.0, &pool).unwrap();
    let c = p.palette.expect("palette");
    assert_eq!(
        (p.quants[0], p.channels[0].data.as_slice()),
        (1, &[0, 1, 40961, 65535][..])
    );
    let decoded: Vec<i32> = p.channels[c + 1]
        .data
        .iter()
        .map(|&k| p.channels[0].data[k as usize])
        .collect();
    assert_eq!(decoded, deep);
    // A lossy extra, or none, has no palette coding.
    let lossy = input(CodingTransform::Rgb, 3, vec![(alpha.clone(), 1.0)]);
    assert!(Pyramid::plain_palette(&lossy, 64, 48, 2.0, &pool).is_none());
    assert!(
        Pyramid::plain_palette(
            &input(CodingTransform::Rgb, 3, Vec::new()),
            64,
            48,
            2.0,
            &pool
        )
        .is_none()
    );
}

#[test]
fn lossy_palette_keeps_the_full_value_only_for_a_black_plate() {
    let pool = crate::thread_pool::ThreadPool::new(1);
    // Mostly the full value beside light tints, at a coarse distance: one
    // cell can absorb them all. The values share no grid, so a merged level
    // cannot round back onto the full value.
    let plane: Vec<i32> = (0..64 * 48)
        .map(|i: usize| [65535, 65535, 64000, 64001, 63000][(i * i * 7 + i * 3) % 5])
        .collect();
    let source = input(CodingTransform::Rgb, 3, vec![(plane.clone(), 60.0)]);
    let squeezed = Pyramid::plain(&source, 64, 48, 2.0, false, &pool).unwrap();
    let levels = |black_plate| {
        super::lossy_palette_levels(&source, &squeezed, 64, 48, black_plate).expect("levels")
    };
    // Alpha has no such value: the full samples may merge.
    assert!(
        plane
            .iter()
            .zip(levels(false))
            .any(|(&s, l)| s == 65535 && l != 65535)
    );
    // K: exactly the no-ink samples stay at the full value.
    for (&s, l) in plane.iter().zip(levels(true)) {
        assert_eq!(s == 65535, l == 65535);
    }
}
