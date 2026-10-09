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
//! Row kernels for DC compensation and reconstruction, resolved once per encode.

use crate::dct::fmla;
use std::sync::OnceLock;

/// Weight of B−Y error in the DC target, with the remaining weight on B error.
pub(crate) const B_DC_Y_COMPENSATION: f32 = 0.75;

#[derive(Clone, Copy)]
pub(crate) struct BdcCompensation {
    pub(crate) scale_b: f32,
    pub(crate) y_step_b: f32,
}

impl BdcCompensation {
    /// Add the luma reconstruction error in B-step units, before rounding.
    #[inline]
    pub(crate) fn adjust(self, target: f32, source_y: f32, yq: i16) -> f32 {
        let y_error = fmla(f32::from(yq), self.y_step_b, -source_y * self.scale_b);
        fmla(B_DC_Y_COMPENSATION, y_error, target)
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn target(self, source_b: f32, source_y: f32, yq: i16, cfl_b: f32) -> f32 {
        self.adjust(
            fmla(source_b, self.scale_b, -f32::from(yq) * cfl_b),
            source_y,
            yq,
        )
    }
}

pub(crate) type CompensateBdcFn = fn(&[f32], &[f32], &[i16], BdcCompensation, f32, &mut [i16]);
pub(crate) type AdjustBdcFn = fn(&mut [f32], &[f32], &[i16], BdcCompensation);
pub(crate) type ReconstructDcFn = fn([&[i16]; 3], [f32; 2], [f32; 3], [&mut [f32]; 3]);

#[derive(Clone, Copy)]
pub(crate) struct DcRowKernels {
    pub(crate) compensate_b: CompensateBdcFn,
    pub(crate) adjust_b: AdjustBdcFn,
    pub(crate) reconstruct: ReconstructDcFn,
}

#[allow(dead_code)]
pub(crate) fn compensate_b_dc_scalar(
    source_b: &[f32],
    source_y: &[f32],
    y_quant: &[i16],
    compensation: BdcCompensation,
    cfl_b: f32,
    output: &mut [i16],
) {
    assert_eq!(source_b.len(), output.len());
    assert_eq!(source_y.len(), output.len());
    assert_eq!(y_quant.len(), output.len());
    for (((level, &yq), &sb), &sy) in output.iter_mut().zip(y_quant).zip(source_b).zip(source_y) {
        *level = compensation.target(sb, sy, yq, cfl_b).round() as i16;
    }
}

#[allow(dead_code)]
pub(crate) fn adjust_b_dc_scalar(
    targets: &mut [f32],
    source_y: &[f32],
    y_quant: &[i16],
    compensation: BdcCompensation,
) {
    assert_eq!(source_y.len(), targets.len());
    assert_eq!(y_quant.len(), targets.len());
    for ((target, &sy), &yq) in targets.iter_mut().zip(source_y).zip(y_quant) {
        *target = compensation.adjust(*target, sy, yq);
    }
}

#[allow(dead_code)]
pub(crate) fn reconstruct_dc_scalar(
    input: [&[i16]; 3],
    cfl: [f32; 2],
    steps: [f32; 3],
    output: [&mut [f32]; 3],
) {
    let [x, y, b] = input;
    let [rx, ry, rb] = output;
    let n = y.len();
    for len in [x.len(), b.len(), rx.len(), ry.len(), rb.len()] {
        assert_eq!(len, n);
    }
    for i in 0..n {
        let y = f32::from(y[i]);
        rx[i] = fmla(y, cfl[0], f32::from(x[i])) * steps[0];
        ry[i] = y * steps[1];
        rb[i] = fmla(y, cfl[1], f32::from(b[i])) * steps[2];
    }
}

fn select_dc_row_kernels() -> DcRowKernels {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return DcRowKernels {
            compensate_b: |b, y, yq, compensation, cfl, out| unsafe {
                crate::avx::compensate_b_dc_avx2(b, y, yq, compensation, cfl, out)
            },
            adjust_b: |t, y, yq, compensation| unsafe {
                crate::avx::adjust_b_dc_avx2(t, y, yq, compensation)
            },
            reconstruct: |input, cfl, steps, output| unsafe {
                crate::avx::reconstruct_dc_avx2(input, cfl, steps, output)
            },
        };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return DcRowKernels {
            compensate_b: |b, y, yq, compensation, cfl, out| unsafe {
                crate::sse::compensate_b_dc_sse41(b, y, yq, compensation, cfl, out)
            },
            adjust_b: |t, y, yq, compensation| unsafe {
                crate::sse::adjust_b_dc_sse41(t, y, yq, compensation)
            },
            reconstruct: |input, cfl, steps, output| unsafe {
                crate::sse::reconstruct_dc_sse41(input, cfl, steps, output)
            },
        };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return DcRowKernels {
        compensate_b: |b, y, yq, compensation, cfl, out| unsafe {
            crate::neon::compensate_b_dc_neon(b, y, yq, compensation, cfl, out)
        },
        adjust_b: |t, y, yq, compensation| unsafe {
            crate::neon::adjust_b_dc_neon(t, y, yq, compensation)
        },
        reconstruct: |input, cfl, steps, output| unsafe {
            crate::neon::reconstruct_dc_neon(input, cfl, steps, output)
        },
    };
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    return DcRowKernels {
        compensate_b: crate::wasm::compensate_b_dc_wasm,
        adjust_b: crate::wasm::adjust_b_dc_wasm,
        reconstruct: crate::wasm::reconstruct_dc_wasm,
    };
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    DcRowKernels {
        compensate_b: compensate_b_dc_scalar,
        adjust_b: adjust_b_dc_scalar,
        reconstruct: reconstruct_dc_scalar,
    }
}

pub(crate) fn selected_dc_row_kernels() -> DcRowKernels {
    static KERNELS: OnceLock<DcRowKernels> = OnceLock::new();
    *KERNELS.get_or_init(select_dc_row_kernels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernels() -> Vec<(&'static str, DcRowKernels, bool)> {
        let scalar_fma = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
        let mut kernels = vec![(
            "scalar",
            DcRowKernels {
                compensate_b: compensate_b_dc_scalar,
                adjust_b: adjust_b_dc_scalar,
                reconstruct: reconstruct_dc_scalar,
            },
            scalar_fma,
        )];
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            kernels.push((
                "avx2",
                DcRowKernels {
                    compensate_b: |b, y, yq, p, cfl, out| unsafe {
                        crate::avx::compensate_b_dc_avx2(b, y, yq, p, cfl, out)
                    },
                    adjust_b: |t, y, yq, p| unsafe { crate::avx::adjust_b_dc_avx2(t, y, yq, p) },
                    reconstruct: |input, cfl, steps, output| unsafe {
                        crate::avx::reconstruct_dc_avx2(input, cfl, steps, output)
                    },
                },
                true,
            ));
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if is_x86_feature_detected!("sse4.1") {
            kernels.push((
                "sse4.1",
                DcRowKernels {
                    compensate_b: |b, y, yq, p, cfl, out| unsafe {
                        crate::sse::compensate_b_dc_sse41(b, y, yq, p, cfl, out)
                    },
                    adjust_b: |t, y, yq, p| unsafe { crate::sse::adjust_b_dc_sse41(t, y, yq, p) },
                    reconstruct: |input, cfl, steps, output| unsafe {
                        crate::sse::reconstruct_dc_sse41(input, cfl, steps, output)
                    },
                },
                false,
            ));
        }
        let selected_fma = if kernels.iter().any(|(name, _, _)| *name == "avx2") {
            true
        } else if kernels.iter().any(|(name, _, _)| *name == "sse4.1") {
            false
        } else {
            scalar_fma
        };
        kernels.push(("selected", selected_dc_row_kernels(), selected_fma));
        kernels
    }

    fn madd(a: f32, b: f32, c: f32, fused: bool) -> f32 {
        if fused { a.mul_add(b, c) } else { a * b + c }
    }

    fn assert_float(actual: f32, expected: f32, context: &str) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{context}: expected NaN, got {actual}");
        } else {
            assert_eq!(actual, expected, "{context}");
        }
    }

    #[test]
    fn dc_rows_match_rounding_saturation_and_float_formulas() {
        let edges = [
            f32::NEG_INFINITY,
            -40_000.5,
            -32_768.5,
            -32_768.0,
            -3.5,
            -2.5,
            -0.5f32.next_up(),
            -0.5,
            -0.5f32.next_down(),
            -0.0,
            0.0,
            0.5f32.next_down(),
            0.5,
            0.5f32.next_up(),
            1.5,
            2.5,
            3.5,
            32_767.0,
            32_767.5,
            40_000.5,
            2_147_483_648.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        let sizes = (0..=65).chain([95, 96, 97, 127, 128, 129, 511, 512, 513]);
        for len in sizes {
            for offset in 0..4 {
                let n = offset + len + 3;
                let source_b: Vec<_> = (0..n)
                    .map(|i| {
                        if i < edges.len() || i % 4 == 0 {
                            edges[i % edges.len()]
                        } else {
                            ((i * 7919 % 65521) as f32 - 32760.0) / 16.0
                        }
                    })
                    .collect();
                let source_y: Vec<_> = (0..n)
                    .map(|i| ((i * 971 % 997) as f32 - 499.0) / 256.0)
                    .collect();
                let zero_y = vec![0.0; n];
                let quant: [Vec<i16>; 3] = std::array::from_fn(|c| {
                    (0..n)
                        .map(|i| i.wrapping_mul(7919 + 113 * c) as i16)
                        .collect()
                });
                for (name, kernels, fused) in kernels() {
                    let context = format!("{name}, len={len}, offset={offset}");
                    for (scale_b, y_step_b, cfl_b) in [(1.0, 0.0, 0.0), (0.93, 0.57, -0.23)] {
                        let source_y = if y_step_b == 0.0 { &zero_y } else { &source_y };
                        let p = BdcCompensation { scale_b, y_step_b };
                        let mut actual = vec![12345i16; n];
                        (kernels.compensate_b)(
                            &source_b[offset..offset + len],
                            &source_y[offset..offset + len],
                            &quant[1][offset..offset + len],
                            p,
                            cfl_b,
                            &mut actual[offset..offset + len],
                        );
                        for i in 0..n {
                            let expected = if (offset..offset + len).contains(&i) {
                                let yq = f32::from(quant[1][i]);
                                let target = madd(source_b[i], scale_b, -yq * cfl_b, fused);
                                let error = madd(yq, y_step_b, -source_y[i] * scale_b, fused);
                                madd(B_DC_Y_COMPENSATION, error, target, fused).round() as i16
                            } else {
                                12345
                            };
                            assert_eq!(actual[i], expected, "compensation: {context}, lane={i}");
                        }
                        let mut targets = source_b.clone();
                        (kernels.adjust_b)(
                            &mut targets[offset..offset + len],
                            &source_y[offset..offset + len],
                            &quant[1][offset..offset + len],
                            p,
                        );
                        for i in 0..n {
                            let expected = if (offset..offset + len).contains(&i) {
                                let error = madd(
                                    f32::from(quant[1][i]),
                                    y_step_b,
                                    -source_y[i] * scale_b,
                                    fused,
                                );
                                madd(B_DC_Y_COMPENSATION, error, source_b[i], fused)
                            } else {
                                source_b[i]
                            };
                            assert_float(
                                targets[i],
                                expected,
                                &format!("adjustment: {context}, lane={i}"),
                            );
                        }
                    }
                    let cfl = [-0.37, 0.83];
                    let steps = [0.031, 0.023, 0.051];
                    let mut output: [Vec<f32>; 3] = std::array::from_fn(|_| vec![12345.0; n]);
                    (kernels.reconstruct)(
                        quant.each_ref().map(|row| &row[offset..offset + len]),
                        cfl,
                        steps,
                        output.each_mut().map(|row| &mut row[offset..offset + len]),
                    );
                    for c in 0..3 {
                        for i in 0..n {
                            let expected = if (offset..offset + len).contains(&i) {
                                let y = f32::from(quant[1][i]);
                                match c {
                                    0 => madd(y, cfl[0], f32::from(quant[0][i]), fused) * steps[0],
                                    1 => y * steps[1],
                                    _ => madd(y, cfl[1], f32::from(quant[2][i]), fused) * steps[2],
                                }
                            } else {
                                12345.0
                            };
                            assert_float(
                                output[c][i],
                                expected,
                                &format!("reconstruction: {context}, c={c}, lane={i}"),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "DC row benchmark; run in release mode with --ignored --nocapture"]
    fn benchmark_dc_rows() {
        use std::hint::black_box;
        use std::time::Instant;

        let n = 512;
        let source_b: Vec<_> = (0..n).map(|i| (i as f32 - 256.0) / 17.0).collect();
        let source_y: Vec<_> = (0..n).map(|i| (i as f32 - 256.0) / 971.0).collect();
        let quant: [Vec<i16>; 3] =
            std::array::from_fn(|c| (0..n).map(|i| (i * (c + 1)) as i16 - 256).collect());
        let p = BdcCompensation {
            scale_b: 317.0,
            y_step_b: 0.53,
        };
        for (name, kernels, _) in kernels() {
            let mut output = vec![0i16; n];
            let mut targets = source_b.clone();
            let mut recon: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; n]);
            for operation in ["compensate", "adjust", "reconstruct"] {
                let mut times = Vec::new();
                for _ in 0..5 {
                    let start = Instant::now();
                    for _ in 0..10_000 {
                        match operation {
                            "compensate" => black_box(kernels.compensate_b)(
                                black_box(&source_b),
                                black_box(&source_y),
                                black_box(&quant[1]),
                                black_box(p),
                                black_box(0.37),
                                black_box(&mut output),
                            ),
                            "adjust" => black_box(kernels.adjust_b)(
                                black_box(&mut targets),
                                black_box(&source_y),
                                black_box(&quant[1]),
                                black_box(p),
                            ),
                            _ => black_box(kernels.reconstruct)(
                                black_box(quant.each_ref().map(Vec::as_slice)),
                                black_box([-0.23, 0.37]),
                                black_box([0.01, 0.02, 0.03]),
                                black_box(recon.each_mut().map(Vec::as_mut_slice)),
                            ),
                        }
                    }
                    times.push(start.elapsed().as_nanos() as f64 / 10_000.0);
                }
                times.sort_by(f64::total_cmp);
                eprintln!("{name} {operation}, {n} DCs: {:.1} ns/row", times[2]);
            }
        }
    }
}
