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

//! Tests for the lossy extra channel.

use super::alpha::ExtraSqueeze;
use super::closed_loop::{Rounding, closed_h_squeeze, closed_v_squeeze};
use crate::encode_image::AlphaPlane;

/// A soft-edged disc: the anti-aliased, feathered alpha lossy coding is for.
fn soft_disc(w: usize, h: usize) -> AlphaPlane {
    AlphaPlane::U8(
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let r = ((x - w as f32 / 2.0).hypot(y - h as f32 / 2.0)) / (w.min(h) as f32);
                (255.0 * ((0.45 - r) / 0.2).clamp(0.0, 1.0)).round() as u8
            })
            .collect(),
    )
}

#[test]
fn unit_steps_reconstruct_the_plane_exactly() {
    let (w, h) = (300, 270);
    let plane = soft_disc(w, h);
    let squeezed = ExtraSqueeze::new(&plane, w, h, 1e-3, false, false).unwrap();
    assert!(squeezed.quants.iter().all(|&q| q == 1));
    let want: Vec<i32> = (0..w * h).map(|i| plane.get_i32(i)).collect();
    assert_eq!(squeezed.reconstruct(), want);
}

#[test]
fn quantized_reconstruction_stays_close() {
    let (w, h) = (300, 270);
    let plane = soft_disc(w, h);
    let squeezed = ExtraSqueeze::new(&plane, w, h, 4.0, false, true).unwrap();
    assert!(squeezed.quants.iter().any(|&q| q > 1));
    let rec = squeezed.reconstruct();
    let err: Vec<i32> = rec
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.clamp(0, 255) - plane.get_i32(i)).abs())
        .collect();
    let mean = err.iter().sum::<i32>() as f32 / err.len() as f32;
    assert!(err.iter().any(|&e| e > 0), "distance 4 should be lossy");
    assert!(mean < 2.0, "mean alpha error {mean}");
}

/// The closed-loop quantizer tracks exactly what the decoder's inverse
/// squeeze rebuilds from the decoded averages and the scaled residuals.
#[test]
fn closed_loop_steps_mirror_the_inverse_squeeze() {
    use crate::squeeze::{Channel, fwd_h_squeeze, fwd_v_squeeze, inv_h_squeeze, inv_v_squeeze};
    for (w, h) in [(37, 29), (64, 64), (1, 9), (9, 1)] {
        let plane = soft_disc(w, h);
        let x: Vec<i32> = (0..w * h).map(|i| plane.get_i32(i)).collect();
        let input = Channel {
            data: x.clone(),
            w,
            h,
            hshift: 0,
            vshift: 0,
        };
        for horizontal in [true, false] {
            let (mut avg, _) = if horizontal {
                fwd_h_squeeze(&input)
            } else {
                fwd_v_squeeze(&input)
            };
            // Perturb the averages as a coarser level's quantization would.
            for (i, v) in avg.data.iter_mut().enumerate() {
                *v += (i as i32 % 5) - 2;
            }
            for (q, deadzone) in [(1, 0.0), (7, 0.0), (7, 0.1), (40, 0.25)] {
                let (res, out) = if horizontal {
                    closed_h_squeeze(&x, w, h, &avg.data, q, Rounding::Deadzone(deadzone))
                } else {
                    closed_v_squeeze(&x, w, h, &avg.data, q, Rounding::Deadzone(deadzone))
                };
                let (rw, rh) = if horizontal { (w / 2, h) } else { (w, h / 2) };
                let res = Channel {
                    data: res.iter().map(|&r| r * q).collect(),
                    w: rw,
                    h: rh,
                    hshift: avg.hshift,
                    vshift: avg.vshift,
                };
                let rebuilt = if horizontal {
                    inv_h_squeeze(&avg, &res)
                } else {
                    inv_v_squeeze(&avg, &res)
                };
                assert_eq!(rebuilt.data, out, "{w}x{h} h={horizontal} q={q}");
            }
        }
    }
}

/// Endpoints and high-contrast detail exercise signed inverse-squeeze values,
/// including excursions that a final alpha clamp would hide.
fn patterned_alpha(w: usize, h: usize, bits: u8, wide: bool) -> AlphaPlane {
    let max = (1u32 << bits) - 1;
    let data: Vec<u16> = (0..w * h)
        .map(|i| match i % 11 {
            0 => 0,
            1 => max as u16,
            2 => (max / 2) as u16,
            _ => {
                let mixed = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((mixed >> 32) % u64::from(max + 1)) as u16
            }
        })
        .collect();
    if wide {
        AlphaPlane::U16 { data, bits }
    } else {
        assert_eq!(bits, 8);
        AlphaPlane::U8(data.into_iter().map(|v| v as u8).collect())
    }
}

#[test]
fn exact_tail_candidates_preserve_unclamped_reconstruction() {
    let mut saw_partial_tail = false;
    let mut saw_unclamped_excursion = false;
    for (bits, wide) in [(8, false), (8, true), (10, true), (12, true), (16, true)] {
        let max = (1i32 << bits) - 1;
        for (w, h) in [(37, 29), (65, 18), (1, 33), (33, 1), (257, 129)] {
            let plane = patterned_alpha(w, h, bits, wide);
            for distance in [1e-6, 0.01, 1.0, 6.0] {
                let original = ExtraSqueeze::new(&plane, w, h, distance, false, true).unwrap();
                let Some(first) = original.exact_tail_start() else {
                    continue;
                };
                saw_partial_tail |= first > 0;
                let want = original.reconstruct();
                saw_unclamped_excursion |= want.iter().any(|&v| v < 0 || v > max);
                // Test every legal stopping point, including suffixes that end
                // between horizontal and vertical steps.
                for retained in first..original.steps.len() {
                    let candidate = original.without_exact_tail(retained).unwrap();
                    assert_eq!(
                        candidate.reconstruct(),
                        want,
                        "{w}x{h}, bits={bits}, wide={wide}, d={distance}, retained={retained}"
                    );
                }
                for retained in 0..first {
                    assert!(original.without_exact_tail(retained).is_none());
                }
            }
        }
    }
    assert!(
        saw_partial_tail,
        "exercise a mixture of lossy and exact steps"
    );
    assert!(
        saw_unclamped_excursion,
        "check signed reconstruction, not only values after alpha clamping"
    );
}

#[test]
fn an_entirely_exact_pyramid_can_be_removed() {
    for (bits, wide) in [(8, false), (8, true), (10, true), (12, true), (16, true)] {
        for (w, h) in [(1, 9), (9, 1), (19, 27), (513, 257)] {
            let plane = patterned_alpha(w, h, bits, wide);
            let original = ExtraSqueeze::new(&plane, w, h, 1e-6, false, true).unwrap();
            assert!(original.quants.iter().all(|&q| q == 1));
            assert_eq!(original.exact_tail_start(), Some(0));
            let candidate = original.without_exact_tail(0).unwrap();
            assert!(candidate.steps.is_empty());
            assert_eq!(candidate.channels.len(), 1);
            let want: Vec<i32> = (0..w * h).map(|i| plane.get_i32(i)).collect();
            assert_eq!(candidate.reconstruct(), want);
        }
    }
}

#[test]
fn exact_tail_removal_rejects_quantized_or_out_of_range_requests() {
    for (bits, wide) in [(8, false), (8, true), (10, true), (12, true), (16, true)] {
        let plane = patterned_alpha(17, 11, bits, wide);
        let quantized = ExtraSqueeze::new(&plane, 17, 11, 64.0, false, true).unwrap();
        assert!(quantized.quants.iter().all(|&q| q > 1));
        assert_eq!(quantized.exact_tail_start(), None);
        for retained in 0..=quantized.steps.len() + 1 {
            assert!(quantized.without_exact_tail(retained).is_none());
        }

        let exact = ExtraSqueeze::new(&plane, 17, 11, 1e-6, false, true).unwrap();
        assert!(exact.without_exact_tail(exact.steps.len()).is_none());
        assert!(exact.without_exact_tail(exact.steps.len() + 1).is_none());

        // Unit residual multipliers alone do not make a suffix reversible:
        // the coarse average must also retain its integer precision.
        let mut quantized_average = exact;
        quantized_average.quants[0] = 2;
        assert_eq!(quantized_average.exact_tail_start(), None);
        for retained in 0..quantized_average.steps.len() {
            assert!(quantized_average.without_exact_tail(retained).is_none());
        }
    }
}

/// The global forest codes every AC-group sample once, one token stream
/// per AC group.
#[test]
fn global_ac_forest_covers_every_ac_sample() {
    let (w, h) = (600, 520);
    let plane = soft_disc(w, h);
    let squeezed = ExtraSqueeze::new(&plane, w, h, 1.5, true, true).unwrap();
    let (xg, yg) = (w.div_ceil(256), h.div_ceil(256));
    let ac_px: usize = squeezed
        .channels
        .iter()
        .skip(squeezed.split)
        .filter(|ch| ch.hshift.min(ch.vshift) <= 2)
        .map(|ch| ch.w * ch.h)
        .sum();
    assert!(ac_px > 0);
    let forest = squeezed
        .global_ac_forest(xg * yg, xg, &crate::thread_pool::ThreadPool::new(2))
        .unwrap();
    assert_eq!(forest.crop_tokens.len(), xg * yg);
    let coded: usize = forest.crop_tokens.iter().map(Vec::len).sum();
    assert_eq!(coded, ac_px);
    assert!(!forest.tree_tokens.is_empty());
}

#[test]
fn repeated_share_marks_tiled_alpha_only() {
    use super::{TILED_SHARE, repeated_share};
    let (w, h) = (400, 400);
    // An antialiased ring per 40x40 tile, repeated over the plane.
    let ring = |x: usize, y: usize| {
        let (dx, dy) = (x as f32 - 19.5, y as f32 - 19.5);
        let r = (dx * dx + dy * dy).sqrt();
        ((1.0 - (r - 12.0).abs() / 4.0).clamp(0.0, 1.0) * 65535.0) as i32
    };
    let tiled: Vec<i32> = (0..w * h).map(|i| ring(i % w % 40, i / w % 40)).collect();
    assert!(repeated_share(&tiled, w) > TILED_SHARE);
    // One large soft ring fading downwards: its rows never recur (an exactly
    // mirrored shape repeats its rows, which LZ77 copies too).
    let single: Vec<i32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let (dx, dy) = (x as f32 - 199.5, y as f32 - 199.5);
            let r = (dx * dx + dy * dy).sqrt();
            let fade = 1.0 - y as f32 / (2 * h) as f32;
            ((1.0 - (r - 120.0).abs() / 40.0).clamp(0.0, 1.0) * fade * 65535.0) as i32
        })
        .collect();
    assert!(repeated_share(&single, w) < 0.1);
    // A straight vertical ramp repeats only from the row above.
    let ramp: Vec<i32> = (0..w * h).map(|i| (i % w * 163) as i32).collect();
    assert!(repeated_share(&ramp, w) < 0.1);
}
