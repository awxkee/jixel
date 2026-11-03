/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use super::LosslessSettings;
use super::encoder::{
    MA_DECISIVE_MIN_SAVINGS, encode_frame_lossless_core_impl, learned_tree_is_decisive,
};
use super::entropy::{EntropyOfHistFn, entropy_of_hist_scalar, selected_entropy_of_hist_fn};
use super::rct::forward_ycocg;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use crate::thread_pool::ThreadPool;

use super::palette::local_palette_coverage_is_sufficient;

/// AVX2 always uses the fused polynomial, even when the rest of an x86
/// build uses the portable non-FMA scalar logarithm.
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn hist_entropy_fma_reference(hist: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let mut bits = 0.0;
    for &count in hist.iter().filter(|&&c| c != 0) {
        let p = count as f32 / total as f32;
        let ix = p.to_bits().wrapping_add(0x3f80_0000 - 0x3f35_04f3);
        let exponent = (ix >> 23) as i32 - 127;
        let a = f32::from_bits((ix & 0x007f_ffff) + 0x3f35_04f3);
        let x = (a - 1.0) / (a + 1.0);
        let x2 = x * x;
        let u = 0.412_198_57f32
            .mul_add(x2, 0.577_078_04)
            .mul_add(x2, 0.961_796_7);
        let log = (x2 * x).mul_add(u, x.mul_add(2.885_390_1, exponent as f32));
        bits -= count as f32 * log;
    }
    bits
}

#[test]
fn selected_hist_entropy_matches_scalar_model_exactly() {
    let entropy_of_hist = selected_entropy_of_hist_fn();
    #[allow(unused_mut)]
    let mut reference: EntropyOfHistFn = entropy_of_hist_scalar;
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        reference = hist_entropy_fma_reference;
    }
    let check = |hist: &[u64]| {
        let total = hist.iter().sum();
        assert_eq!(
            entropy_of_hist(hist, total).to_bits(),
            reference(hist, total).to_bits(),
            "histogram length {}, total {total}",
            hist.len(),
        );
    };
    check(&[]);
    check(&[0; 65]);
    check(&[u64::MAX]);
    check(&[u64::MAX - 6, 1, 2, 3]);
    assert_eq!(entropy_of_hist(&[1, 2, 3], 0), 0.0);
    assert_eq!(entropy_of_hist(&[1, 1], 2), 2.0);
    // Exercise direct u64-to-f32 rounding around half-way values,
    // including counts too large to convert to f64 without rounding.
    for exponent in 24..64 {
        let midpoint = (1u64 << exponent) + (1u64 << (exponent - 24));
        for value in [midpoint - 1, midpoint, midpoint + 1] {
            check(&[value, 1, 3, 7, 0]);
            check(&[value, 1, 3, 7, 0, 11, 13, 17, 19]);
        }
    }
    for len in 0..129 {
        for stride in [1, 2, 3, 5, 16, 64] {
            for range in [1, 100, 1 << 24, 1 << 32, 1 << 54] {
                let mut state = (len + stride) as u64;
                let hist: Vec<_> = (0..len)
                    .map(|i| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        if i % stride == 0 {
                            1 + (state >> 10) % range
                        } else {
                            0
                        }
                    })
                    .collect();
                check(&hist);
            }
        }
    }
}

#[test]
fn shortlist_thresholds_keep_ambiguous_candidates() {
    assert!(!local_palette_coverage_is_sufficient(24, 100));
    assert!(local_palette_coverage_is_sufficient(25, 100));
    assert!(!learned_tree_is_decisive(
        MA_DECISIVE_MIN_SAVINGS - f64::EPSILON
    ));
    assert!(learned_tree_is_decisive(MA_DECISIVE_MIN_SAVINGS));
}

fn image_from_rgb(
    width: usize,
    height: usize,
    mut rgb: impl FnMut(usize, usize) -> [i32; 3],
) -> Image3Si {
    let mut image = Image3Si::new(width, height);
    for y in 0..height {
        let [p0, p1, p2] = image.all_plane_rows_mut(y);
        for x in 0..width {
            let [r, g, b] = rgb(x, y);
            let (yc, co, cg) = forward_ycocg(r, g, b);
            p0[x] = yc;
            p1[x] = co;
            p2[x] = cg;
        }
    }
    image
}

fn encode_core(image: &Image3Si, allow_palettes: bool) -> BitWriter {
    let pool = ThreadPool::new_lossless(1);
    let mut scratch = CoderScratch::lossless();
    let mut writer = BitWriter::new();
    encode_frame_lossless_core_impl(
        image,
        None,
        LosslessSettings::rgb8(crate::Speed::Slow, crate::DecodingSpeed::Slow),
        ModularFrameKind::Regular,
        allow_palettes,
        true,
        &pool,
        &mut scratch,
        &mut writer,
    );
    writer
}

#[test]
fn slow_single_group_palette_is_rate_safe() {
    let image = image_from_rgb(64, 64, |x, y| {
        let c = ((x / 8 + y / 8) & 3) as i32;
        [c * 61, c * 37, c * 19]
    });
    let selected = encode_core(&image, true);
    let normal = encode_core(&image, false);
    assert!(selected.bits_written() <= normal.bits_written());
}

#[test]
fn slow_local_palette_is_rate_safe_for_mixed_content() {
    let image = image_from_rgb(257, 160, |x, y| {
        if x < 256 && y < 128 {
            let c = ((x / 16 + y / 16) & 7) as i32;
            [c * 31, c * 17, c * 7]
        } else {
            let v = x
                .wrapping_mul(1_664_525)
                .wrapping_add(y.wrapping_mul(1_013_904_223));
            [
                ((v >> 8) & 255) as i32,
                ((v >> 16) & 255) as i32,
                ((v >> 24) & 255) as i32,
            ]
        }
    });
    let selected = encode_core(&image, true);
    let normal = encode_core(&image, false);
    assert!(selected.bits_written() <= normal.bits_written());
}
