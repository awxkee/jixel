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
use std::sync::OnceLock;

/// The finer DC step holding the squeezed plane, relative to the current one.
pub(crate) const CHROMA_DC_SQUEEZE_STEP: f32 = 0.5;
/// Deadzone of the high-band quantizer (fraction of a step kept as zero).
pub(crate) const CHROMA_DC_SQUEEZE_DEADZONE: f32 = 0.7;

// Kernels write an even prefix of the paired rows and return its length.
type HaarRowsFn = fn(&[f32], &[f32], f32, &mut [f32], &mut [f32]) -> usize;

fn selected_rows_fn() -> HaarRowsFn {
    static ROWS: OnceLock<HaarRowsFn> = OnceLock::new();
    *ROWS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            // No FMA: keep the scalar expression's arithmetic order.
            return |top, bottom, q, top_out, bottom_out| unsafe {
                crate::avx::haar_rows_avx2(top, bottom, q, top_out, bottom_out)
            };
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return |top, bottom, q, top_out, bottom_out| unsafe {
                crate::sse::haar_rows_sse41(top, bottom, q, top_out, bottom_out)
            };
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return |top, bottom, q, top_out, bottom_out| unsafe {
                crate::neon::haar_rows_neon(top, bottom, q, top_out, bottom_out)
            };
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            crate::wasm::haar_rows_wasm
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
        haar_rows_scalar
    })
}

/// One-level 2x2 Haar: round the low band to the fine lattice, deadzone-quantize
/// the three high bands at `q_hf`, and pass odd edges through unchanged.
pub(crate) fn haar_smooth(t: &[f32], w: usize, h: usize, q_hf: f32, out: &mut [f32]) {
    haar_smooth_with_rows(t, w, h, q_hf, out, selected_rows_fn());
}

fn haar_smooth_with_rows(
    t: &[f32],
    w: usize,
    h: usize,
    q_hf: f32,
    out: &mut [f32],
    rows: HaarRowsFn,
) {
    assert_eq!(t.len(), w * h);
    assert_eq!(out.len(), t.len());
    if w < 2 || h < 2 {
        out.copy_from_slice(t);
        return;
    }
    let mut input_rows = t.chunks_exact(2 * w);
    let mut output_rows = out.chunks_exact_mut(2 * w);
    for (input, output) in input_rows.by_ref().zip(output_rows.by_ref()) {
        let (top, bottom) = input.split_at(w);
        let (top_out, bottom_out) = output.split_at_mut(w);
        let done = rows(top, bottom, q_hf, top_out, bottom_out);
        if done < (w & !1) {
            haar_rows_scalar(
                &top[done..],
                &bottom[done..],
                q_hf,
                &mut top_out[done..],
                &mut bottom_out[done..],
            );
        }
        if !w.is_multiple_of(2) {
            top_out[w - 1] = top[w - 1];
            bottom_out[w - 1] = bottom[w - 1];
        }
    }
    output_rows
        .into_remainder()
        .copy_from_slice(input_rows.remainder());
}

fn haar_rows_scalar(
    top: &[f32],
    bottom: &[f32],
    q_hf: f32,
    top_out: &mut [f32],
    bottom_out: &mut [f32],
) -> usize {
    let dzq = |x: f32| -> f32 {
        ((x.abs() / q_hf + (1.0 - CHROMA_DC_SQUEEZE_DEADZONE)).floor() * q_hf).copysign(x)
    };
    for ((ab, cd), (top_dst, bottom_dst)) in top
        .as_chunks::<2>()
        .0
        .iter()
        .zip(bottom.as_chunks::<2>().0)
        .zip(
            top_out
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(bottom_out.as_chunks_mut::<2>().0),
        )
    {
        let [a, b] = *ab;
        let [c, d] = *cd;
        let ll =
            ((a + b + c + d) * (0.25 / CHROMA_DC_SQUEEZE_STEP)).round() * CHROMA_DC_SQUEEZE_STEP;
        let hl = dzq((a - b + c - d) * 0.25);
        let lh = dzq((a + b - c - d) * 0.25);
        let hh = dzq((a - b - c + d) * 0.25);
        top_dst[0] = ll + hl + lh + hh;
        top_dst[1] = ll - hl + lh - hh;
        bottom_dst[0] = ll + hl - lh - hh;
        bottom_dst[1] = ll - hl - lh + hh;
    }
    top.len() & !1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available_kernels() -> Vec<(&'static str, HaarRowsFn)> {
        #[allow(unused_mut)] // Architecture-specific entries depend on enabled features.
        let mut kernels: Vec<(&str, HaarRowsFn)> = vec![("dispatch", selected_rows_fn())];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            kernels.push(("neon", |a, b, q, c, d| unsafe {
                crate::neon::haar_rows_neon(a, b, q, c, d)
            }));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", |a, b, q, c, d| unsafe {
                crate::avx::haar_rows_avx2(a, b, q, c, d)
            }));
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            kernels.push(("sse4.1", |a, b, q, c, d| unsafe {
                crate::sse::haar_rows_sse41(a, b, q, c, d)
            }));
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        kernels.push(("wasm", crate::wasm::haar_rows_wasm));
        kernels
    }

    fn check_kernels(source: &[f32], w: usize, h: usize, q: f32) {
        let mut expected = vec![f32::NAN; source.len()];
        haar_smooth_with_rows(source, w, h, q, &mut expected, haar_rows_scalar);
        for (name, rows) in available_kernels() {
            let mut actual = vec![f32::NAN; source.len()];
            haar_smooth_with_rows(source, w, h, q, &mut actual, rows);
            for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{name}: {w}x{h}, q={q}, index={i}, actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn kernels_match_scalar_for_all_tails_and_odd_edges() {
        let mut state = 0x5632_91adu32;
        for w in (0..=65).chain([127, 128, 129, 513, 600]) {
            for h in [0, 1, 2, 3, 5, 16] {
                let source: Vec<_> = (0..w * h)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        match i % 8 {
                            0 => -0.0,
                            1 => f32::from_bits(state & 0x807f_ffff), // Subnormals.
                            2 => f32::from_bits((state & 0x807f_ffff) | ((state % 220) << 23)),
                            _ => (state % 100_001) as f32 * 0.001 - 50.0,
                        }
                    })
                    .collect();
                for q in [0.3, 0.5, 0.75, 1.0, 1.5, 2.75] {
                    check_kernels(&source, w, h, q);
                }
            }
        }
    }

    #[test]
    fn kernels_preserve_low_band_ties_and_deadzone_thresholds() {
        let w = 35;
        for q in [0.3, 0.5, 0.75, 1.0, 1.5, 2.75] {
            let threshold = q * CHROMA_DC_SQUEEZE_DEADZONE;
            let values = [
                0.0,
                -0.0,
                0.25,
                -0.25,
                0.75,
                -0.75,
                f32::from_bits(0.25f32.to_bits() - 1),
                f32::from_bits(0.25f32.to_bits() + 1),
                f32::from_bits(threshold.to_bits() - 1),
                threshold,
                f32::from_bits(threshold.to_bits() + 1),
            ];
            for value in values {
                for sign in [1.0, -1.0] {
                    let x = value * sign;
                    for block in [[x; 4], [x, -x, x, -x], [x, x, -x, -x], [x, -x, -x, x]] {
                        let mut source = vec![x; w * 3];
                        for col in (0..w - 1).step_by(2) {
                            source[col..col + 2].copy_from_slice(&block[..2]);
                            source[w + col..w + col + 2].copy_from_slice(&block[2..]);
                        }
                        check_kernels(&source, w, 3, q);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "Haar kernel benchmark; run in release mode with --ignored --nocapture"]
    fn benchmark_haar_smooth() {
        use std::hint::black_box;
        use std::time::Instant;
        let (w, h) = (600, 450);
        let source: Vec<_> = (0..w * h).map(|i| (i % 1001) as f32 * 0.01 - 5.0).collect();
        let mut output = vec![0.0; source.len()];
        let mut kernels = available_kernels();
        kernels.insert(0, ("scalar", haar_rows_scalar));
        for (name, rows) in kernels {
            let mut times = Vec::new();
            for _ in 0..9 {
                let start = Instant::now();
                for _ in 0..20 {
                    haar_smooth_with_rows(
                        black_box(&source),
                        w,
                        h,
                        black_box(0.75),
                        black_box(&mut output),
                        rows,
                    );
                }
                black_box(&output);
                times.push(start.elapsed().as_secs_f64() * 1000.0 / 20.0);
            }
            times.sort_by(f64::total_cmp);
            eprintln!("Haar {name} (600x450): {:.3} ms", times[times.len() / 2]);
        }
    }
}
