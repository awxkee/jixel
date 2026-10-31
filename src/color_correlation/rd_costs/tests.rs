use crate::color_correlation::{CflRdCostFn, CflRdCostsFn};

fn backends() -> Vec<(CflRdCostsFn, CflRdCostFn)> {
    #[allow(unused_mut)]
    let mut out: Vec<(CflRdCostsFn, CflRdCostFn)> = vec![];
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        out.push((
            |m, s, f, r, i, w, l, b, c| unsafe {
                crate::avx::cfl_rd_costs_avx2(m, s, f, r, i, w, l, b, c)
            },
            |m, s, f, w, l, b| unsafe { crate::avx::cfl_rd_cost_avx2(m, s, f, w, l, b) },
        ));
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        out.push((
            |m, s, f, r, i, w, l, b, c| unsafe {
                crate::sse::cfl_rd_costs_sse41(m, s, f, r, i, w, l, b, c)
            },
            |m, s, f, w, l, b| unsafe { crate::sse::cfl_rd_cost_sse41(m, s, f, w, l, b) },
        ));
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    out.push((
        crate::wasm::cfl_rd_costs_wasm,
        crate::color_correlation::cfl_rd_cost_scalar,
    ));
    out
}

fn winner(costs: &[f32], rates: &[f32]) -> usize {
    let mut best = (f32::MAX, 0);
    for (i, (&cost, &rate)) in costs.iter().zip(rates).enumerate() {
        if cost + rate < best.0 {
            best = (cost + rate, i);
        }
    }
    best.1
}

#[test]
fn batching_and_bounds_keep_backend_bits_and_winners() {
    for (batch, single) in backends() {
        for len in [0, 1, 3, 4, 7, 8, 15, 63, 64, 65, 127, 129, 1025, 4095] {
            for pattern in 0..3 {
                let m: Vec<_> = (0..len + 1)
                    .map(|i| {
                        if pattern == 1 || (pattern == 2 && i % 17 < 13) {
                            0.0
                        } else {
                            (i as f32 % 37. - 18.) * 0.03125
                        }
                    })
                    .collect();
                let s: Vec<_> = (0..len + 1)
                    .map(|i| (i as f32 % 71. - 35.) * 0.09375)
                    .collect();
                let factors: [f32; 28] = std::array::from_fn(|i| (i as f32 - 14.) * 0.125);
                let rates: [f32; 28] = std::array::from_fn(|i| (i % 5) as f32 * 0.031);
                for bias in crate::group::DequantBias::channels(true) {
                    for (w, l) in [
                        (0.3, 0.03),
                        (1.0, 0.0),
                        (0.0, 0.03),
                        (-0.1, 0.03),
                        (0.3, -0.03),
                    ] {
                        let expected = factors.map(|f| single(&m[1..], &s[1..], f, w, l, bias));
                        for n in [0usize, 1, 3, 4, 5, 7, 19, 28] {
                            for first in [0, n.saturating_sub(1)] {
                                let mut actual = [f32::NAN; 28];
                                batch(
                                    &m[1..],
                                    &s[1..],
                                    &factors[..n],
                                    &rates[..n],
                                    first,
                                    w,
                                    l,
                                    bias,
                                    &mut actual[..n],
                                );
                                for i in 0..n {
                                    assert!(
                                        actual[i] == f32::INFINITY
                                            || actual[i].to_bits() == expected[i].to_bits(),
                                        "len={len}, pattern={pattern}, n={n}, first={first}, i={i}, actual={}, expected={}",
                                        actual[i],
                                        expected[i]
                                    );
                                }
                                assert_eq!(
                                    winner(&actual[..n], &rates[..n]),
                                    winner(&expected[..n], &rates[..n])
                                );
                                assert!(actual[n..].iter().all(|v| v.is_nan()));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn batches_preserve_special_values_and_rounding() {
    let values = [
        0.0f32,
        -0.0,
        0.5f32.next_down(),
        0.5,
        0.5f32.next_up(),
        -1.5,
        1.5f32.next_up(),
        8_388_608.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for (batch, single) in backends() {
        for offset in 0..values.len() {
            let m: Vec<_> = (0..65)
                .map(|i| {
                    if offset % 2 == 0 {
                        0.0
                    } else {
                        values[(i + offset) % values.len()]
                    }
                })
                .collect();
            let s: Vec<_> = (0..65)
                .map(|i| values[(i * 3 + offset) % values.len()])
                .collect();
            let factors = [0.0, -0.0, 0.37, 1.0, -1.0, f32::INFINITY, f32::NAN];
            let rates = [0.0; 7];
            for bias in crate::group::DequantBias::channels(true) {
                for lambda in [0.0, 0.03, -0.03, f32::INFINITY] {
                    let expected = factors.map(|f| single(&m, &s, f, 0.3, lambda, bias));
                    let mut actual = [0.; 7];
                    batch(&m, &s, &factors, &rates, 0, 0.3, lambda, bias, &mut actual);
                    for i in 0..7 {
                        assert!(
                            actual[i] == f32::INFINITY
                                || actual[i].to_bits() == expected[i].to_bits(),
                            "i={i}, offset={offset}"
                        );
                    }
                    assert_eq!(winner(&actual, &rates), winner(&expected, &rates));
                }
            }
        }
    }
}

#[test]
fn finite_batches_keep_half_integer_rounding_and_overflow_behavior() {
    let values = [
        0.0f32,
        -0.0,
        0.5f32.next_down(),
        0.5,
        0.5f32.next_up(),
        1.5f32.next_down(),
        1.5,
        1.5f32.next_up(),
        -2.5,
        f32::MAX,
        -f32::MAX,
    ];
    let factors = [-1.0, 0.0, 0.37, 1.5];
    let rates = [0.0; 4];
    for (batch, single) in backends() {
        for offset in 0..values.len() {
            for zero_luma in [false, true] {
                let m: Vec<_> = (0..65)
                    .map(|i| {
                        if zero_luma {
                            0.0
                        } else {
                            values[(i + offset) % values.len()]
                        }
                    })
                    .collect();
                let s: Vec<_> = (0..65)
                    .map(|i| values[(i * 3 + offset) % values.len()])
                    .collect();
                let bias = crate::group::DequantBias::DEFAULT[0];
                let expected = factors.map(|f| single(&m, &s, f, 0.3, 0.03, bias));
                let mut actual = [0.0; 4];
                batch(&m, &s, &factors, &rates, 0, 0.3, 0.03, bias, &mut actual);
                for i in 0..4 {
                    assert!(
                        actual[i] == f32::INFINITY || actual[i].to_bits() == expected[i].to_bits()
                    );
                }
                assert_eq!(winner(&actual, &rates), winner(&expected, &rates));
            }
        }
    }
}
