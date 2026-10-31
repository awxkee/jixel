use core::arch::wasm32::*;

/// Match the scalar cost expression, including its unfused log polynomial.
#[inline]
#[target_feature(enable = "simd128")]
fn rd_terms(
    m: v128,
    s: v128,
    factor: v128,
    weight: f32,
    lambda: f32,
    bias: crate::group::DequantBias,
) -> v128 {
    use crate::color_correlation::{CFL_MAGNITUDE_BITS, CFL_NONZERO_BITS};
    let r = f32x4_sub(s, f32x4_mul(factor, m));
    let zero = f32x4_mul(f32x4_mul(f32x4_splat(weight), r), r);
    let truncated = f32x4_trunc(r);
    let half = f32x4_ge(f32x4_abs(f32x4_sub(r, truncated)), f32x4_splat(0.5));
    let signed_one = v128_or(f32x4_splat(1.0), v128_and(r, f32x4_splat(-0.0)));
    let level = f32x4_add(truncated, v128_and(signed_one, half));
    let error = f32x4_sub(
        r,
        super::ac_strategy::wasm_dequantized_level_f32(level, bias),
    );
    let log = super::adaptive_quant::dirty_log2f_x4(f32x4_abs(level));
    let bits = f32x4_mul(
        f32x4_splat(lambda),
        f32x4_add(
            f32x4_splat(CFL_NONZERO_BITS),
            f32x4_mul(f32x4_splat(CFL_MAGNITUDE_BITS), log),
        ),
    );
    let coded = f32x4_add(
        f32x4_mul(f32x4_mul(f32x4_splat(weight), error), error),
        bits,
    );
    // Rust's min ignores a single NaN; Wasm's ordinary min propagates it.
    let pick = v128_and(
        f32x4_ne(level, f32x4_splat(0.0)),
        v128_or(f32x4_lt(coded, zero), f32x4_ne(zero, zero)),
    );
    v128_bitselect(coded, zero, pick)
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "simd128")]
pub(crate) fn cfl_rd_costs_wasm(
    m: &[f32],
    s: &[f32],
    factors: &[f32],
    rates: &[f32],
    first: usize,
    weight: f32,
    lambda: f32,
    bias: crate::group::DequantBias,
    costs: &mut [f32],
) {
    assert_eq!(factors.len(), costs.len());
    assert_eq!(rates.len(), costs.len());
    assert!(factors.is_empty() || first < factors.len());
    let len = m.len().min(s.len());
    // Preserve scalar handling of NaN payloads and nonfinite parameters.
    if ![weight, lambda, bias.one, bias.numerator]
        .iter()
        .all(|v| v.is_finite())
        || !factors
            .iter()
            .chain(&m[..len])
            .chain(&s[..len])
            .all(|v| v.is_finite())
    {
        for (&f, out) in factors.iter().zip(costs) {
            *out = crate::color_correlation::cfl_rd_cost_scalar(m, s, f, weight, lambda, bias);
        }
        return;
    }
    let (batches, rest) = factors.as_chunks::<4>();
    let (outputs, tail) = costs.as_chunks_mut::<4>();
    for (factors, output) in batches.iter().zip(outputs) {
        let factor = load(factors);
        let mut acc = f32x4_splat(0.0);
        // Each lane is one slope. Every slope retains the original scalar
        // coefficient accumulation order, with no horizontal reduction.
        for (&m, &s) in m[..len].iter().zip(&s[..len]) {
            acc = f32x4_add(
                acc,
                rd_terms(f32x4_splat(m), f32x4_splat(s), factor, weight, lambda, bias),
            );
        }
        store(output, acc);
        if output.iter().any(|v| !v.is_finite()) {
            // Finite operands can still overflow; retain scalar NaN payloads.
            for (&factor, out) in factors.iter().zip(output) {
                *out = crate::color_correlation::cfl_rd_cost_scalar(
                    m, s, factor, weight, lambda, bias,
                );
            }
        }
    }
    for (&f, out) in rest.iter().zip(tail) {
        *out = crate::color_correlation::cfl_rd_cost_scalar(m, s, f, weight, lambda, bias);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn load(values: &[f32; 4]) -> v128 {
    unsafe { v128_load(values.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "simd128")]
fn store(values: &mut [f32; 4], vector: v128) {
    unsafe { v128_store(values.as_mut_ptr().cast(), vector) }
}

#[inline]
#[target_feature(enable = "simd128")]
fn reduce_transposed_4x4(a: v128, b: v128, c: v128, d: v128) -> [f32; 4] {
    let ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a, b);
    let ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a, b);
    let cd_lo = i32x4_shuffle::<0, 4, 1, 5>(c, d);
    let cd_hi = i32x4_shuffle::<2, 6, 3, 7>(c, d);
    let lane_0 = i32x4_shuffle::<0, 1, 4, 5>(ab_lo, cd_lo);
    let lane_1 = i32x4_shuffle::<2, 3, 6, 7>(ab_lo, cd_lo);
    let lane_2 = i32x4_shuffle::<0, 1, 4, 5>(ab_hi, cd_hi);
    let lane_3 = i32x4_shuffle::<2, 3, 6, 7>(ab_hi, cd_hi);
    let sums = f32x4_add(f32x4_add(lane_0, lane_1), f32x4_add(lane_2, lane_3));
    let mut out = [0.0f32; 4];
    store(&mut out, sums);
    out
}

#[target_feature(enable = "simd128")]
pub(crate) fn cfl_regression_wasm(
    y: &[f32; 64],
    x: &[f32; 64],
    b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
) -> [f32; 4] {
    let inv_color = f32x4_splat(1.0 / 84.0);
    let mut ca_x = f32x4_splat(0.0);
    let mut cb_x = f32x4_splat(0.0);
    let mut ca_b = f32x4_splat(0.0);
    let mut cb_b = f32x4_splat(0.0);

    let y_chunks = y.as_chunks::<4>().0;
    let x_chunks = x.as_chunks::<4>().0;
    let b_chunks = b.as_chunks::<4>().0;
    let qm_x_chunks = qm_x.as_chunks::<4>().0;
    let qm_b_chunks = qm_b.as_chunks::<4>().0;
    for ((((y, x), b), qm_x), qm_b) in y_chunks
        .iter()
        .zip(x_chunks)
        .zip(b_chunks)
        .zip(qm_x_chunks)
        .zip(qm_b_chunks)
    {
        let yv = unsafe { v128_load(y.as_ptr().cast()) };
        let xv = unsafe { v128_load(x.as_ptr().cast()) };
        let bv = unsafe { v128_load(b.as_ptr().cast()) };
        let qx = unsafe { v128_load(qm_x.as_ptr().cast()) };
        let qb = unsafe { v128_load(qm_b.as_ptr().cast()) };

        let mx = f32x4_mul(yv, qx);
        let sx = f32x4_mul(xv, qx);
        let ax = f32x4_mul(inv_color, mx);
        ca_x = f32x4_add(ca_x, f32x4_mul(ax, ax));
        cb_x = f32x4_sub(cb_x, f32x4_mul(ax, sx));

        let mb = f32x4_mul(yv, qb);
        let sb = f32x4_mul(bv, qb);
        let ab = f32x4_mul(inv_color, mb);
        let residual_b = f32x4_sub(mb, sb);
        ca_b = f32x4_add(ca_b, f32x4_mul(ab, ab));
        cb_b = f32x4_add(cb_b, f32x4_mul(ab, residual_b));
    }

    reduce_transposed_4x4(ca_x, cb_x, ca_b, cb_b)
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "simd128")]
pub(crate) fn cfl_rdo_refill_wasm(
    m_x: &mut [f32],
    s_x: &mut [f32],
    m_b: &mut [f32],
    s_b: &mut [f32],
    y: &[f32],
    x: &[f32],
    b: &[f32],
    qm_x: &[f32],
    qm_b: &[f32],
    params: crate::color_correlation::CflRdoRefillParams,
) {
    let len = m_x.len();
    assert_eq!(s_x.len(), len);
    assert_eq!(m_b.len(), len);
    assert_eq!(s_b.len(), len);
    assert_eq!(y.len(), len);
    assert_eq!(x.len(), len);
    assert_eq!(b.len(), len);
    assert_eq!(qm_x.len(), len);
    assert_eq!(qm_b.len(), len);
    let q_block = f32x4_splat(params.q_block);
    let x_mul = f32x4_splat(params.x_mul);
    let b_mul = f32x4_splat(params.b_mul);
    let shift_x = f32x4_splat(params.shift_x);
    let shift_b = f32x4_splat(params.shift_b);
    let mut i = 0;
    while i + 4 <= len {
        let yv = unsafe { v128_load(y.as_ptr().add(i).cast()) };
        let xv = unsafe { v128_load(x.as_ptr().add(i).cast()) };
        let bv = unsafe { v128_load(b.as_ptr().add(i).cast()) };
        let qx = unsafe { v128_load(qm_x.as_ptr().add(i).cast()) };
        let qb = unsafe { v128_load(qm_b.as_ptr().add(i).cast()) };
        // Keep the two quantizer multiplications in the scalar order.
        let qx = f32x4_mul(f32x4_mul(qx, q_block), x_mul);
        let qb = f32x4_mul(f32x4_mul(qb, q_block), b_mul);
        let sx = f32x4_add(f32x4_mul(shift_x, yv), xv);
        let sb = f32x4_add(f32x4_mul(shift_b, yv), bv);
        unsafe {
            v128_store(m_x.as_mut_ptr().add(i).cast(), f32x4_mul(yv, qx));
            v128_store(s_x.as_mut_ptr().add(i).cast(), f32x4_mul(sx, qx));
            v128_store(m_b.as_mut_ptr().add(i).cast(), f32x4_mul(yv, qb));
            v128_store(s_b.as_mut_ptr().add(i).cast(), f32x4_mul(sb, qb));
        }
        i += 4;
    }
    let src = y[i..]
        .iter()
        .zip(&x[i..])
        .zip(&b[i..])
        .zip(qm_x[i..].iter().zip(&qm_b[i..]));
    let dst = m_x[i..]
        .iter_mut()
        .zip(&mut s_x[i..])
        .zip(m_b[i..].iter_mut().zip(&mut s_b[i..]));
    for ((((&y, &x), &b), (&qx, &qb)), ((m_x, s_x), (m_b, s_b))) in src.zip(dst) {
        let qx = qx * params.q_block * params.x_mul;
        let qb = qb * params.q_block * params.b_mul;
        *m_x = y * qx;
        *s_x = (params.shift_x * y + x) * qx;
        *m_b = y * qb;
        *s_b = (params.shift_b * y + b) * qb;
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn cfl_rdo_block_wasm(
    m_x: &mut [f32; 63],
    s_x: &mut [f32; 63],
    m_b: &mut [f32; 63],
    s_b: &mut [f32; 63],
    block_y: &[f32; 64],
    block_x: &[f32; 64],
    block_b: &[f32; 64],
    qm_x: &[f32; 64],
    qm_b: &[f32; 64],
    q_block: f32,
) {
    let q_block_v = f32x4_splat(q_block);
    for i in (0..60).step_by(4) {
        let coeff = i + 1;
        let y = unsafe { v128_load(block_y.as_ptr().add(coeff).cast()) };
        let x = unsafe { v128_load(block_x.as_ptr().add(coeff).cast()) };
        let b = unsafe { v128_load(block_b.as_ptr().add(coeff).cast()) };
        let qx = unsafe { v128_load(qm_x.as_ptr().add(coeff).cast()) };
        let qb = unsafe { v128_load(qm_b.as_ptr().add(coeff).cast()) };
        let qx = f32x4_mul(qx, q_block_v);
        let qb = f32x4_mul(qb, q_block_v);
        unsafe {
            v128_store(m_x.as_mut_ptr().add(i).cast(), f32x4_mul(y, qx));
            v128_store(s_x.as_mut_ptr().add(i).cast(), f32x4_mul(x, qx));
            v128_store(m_b.as_mut_ptr().add(i).cast(), f32x4_mul(y, qb));
            v128_store(s_b.as_mut_ptr().add(i).cast(), f32x4_mul(b, qb));
        }
    }

    // SIMD128 has no masked load/store. Recompute output 59 and use the other
    // three lanes for the tail, keeping every access within the arrays.
    let i = 59;
    let coeff = i + 1;
    let y = unsafe { v128_load(block_y.as_ptr().add(coeff).cast()) };
    let x = unsafe { v128_load(block_x.as_ptr().add(coeff).cast()) };
    let b = unsafe { v128_load(block_b.as_ptr().add(coeff).cast()) };
    let qx = unsafe { v128_load(qm_x.as_ptr().add(coeff).cast()) };
    let qb = unsafe { v128_load(qm_b.as_ptr().add(coeff).cast()) };
    let qx = f32x4_mul(qx, q_block_v);
    let qb = f32x4_mul(qb, q_block_v);
    unsafe {
        v128_store(m_x.as_mut_ptr().add(i).cast(), f32x4_mul(y, qx));
        v128_store(s_x.as_mut_ptr().add(i).cast(), f32x4_mul(x, qx));
        v128_store(m_b.as_mut_ptr().add(i).cast(), f32x4_mul(y, qb));
        v128_store(s_b.as_mut_ptr().add(i).cast(), f32x4_mul(b, qb));
    }
}

#[target_feature(enable = "simd128")]
pub(crate) fn cfl_rdo_stats_wasm(m: &[f32], s: &[f32]) -> [f32; 4] {
    let len = m.len().min(s.len());
    let (m_chunks, m_tail) = m[..len].as_chunks::<4>();
    let (s_chunks, s_tail) = s[..len].as_chunks::<4>();
    let mut dot_ms = f32x4_splat(0.0);
    let mut dot_mm = f32x4_splat(0.0);
    let mut dot_ss = f32x4_splat(0.0);
    let mut sum_abs_s = f32x4_splat(0.0);

    for (m, s) in m_chunks.iter().zip(s_chunks) {
        let mv = unsafe { v128_load(m.as_ptr().cast()) };
        let sv = unsafe { v128_load(s.as_ptr().cast()) };
        dot_ms = f32x4_add(dot_ms, f32x4_mul(mv, sv));
        dot_mm = f32x4_add(dot_mm, f32x4_mul(mv, mv));
        dot_ss = f32x4_add(dot_ss, f32x4_mul(sv, sv));
        sum_abs_s = f32x4_add(sum_abs_s, f32x4_abs(sv));
    }

    let mut stats = reduce_transposed_4x4(dot_ms, dot_mm, dot_ss, sum_abs_s);
    for (&mv, &sv) in m_tail.iter().zip(s_tail) {
        stats[0] += mv * sv;
        stats[1] += mv * mv;
        stats[2] += sv * sv;
        stats[3] += sv.abs();
    }
    stats
}

#[target_feature(enable = "simd128")]
pub(crate) fn apply_cfl_wasm(x: &mut [f32], y: &[f32], b: &mut [f32], cmap_factor: [f32; 3]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), b.len());

    let cx = f32x4_splat(-cmap_factor[0]);
    let cb = f32x4_splat(-cmap_factor[2]);
    let (x_chunks, x_tail) = x.as_chunks_mut::<4>();
    let (y_chunks, y_tail) = y.as_chunks::<4>();
    let (b_chunks, b_tail) = b.as_chunks_mut::<4>();
    let (x_groups, x_chunks_tail) = x_chunks.as_chunks_mut::<4>();
    let (y_groups, y_chunks_tail) = y_chunks.as_chunks::<4>();
    let (b_groups, b_chunks_tail) = b_chunks.as_chunks_mut::<4>();

    for ((x, y), b) in x_groups.iter_mut().zip(y_groups).zip(b_groups) {
        let y0 = load(&y[0]);
        let y1 = load(&y[1]);
        let y2 = load(&y[2]);
        let y3 = load(&y[3]);
        let x0 = load(&x[0]);
        let x1 = load(&x[1]);
        let x2 = load(&x[2]);
        let x3 = load(&x[3]);
        let b0 = load(&b[0]);
        let b1 = load(&b[1]);
        let b2 = load(&b[2]);
        let b3 = load(&b[3]);

        let x0 = f32x4_add(x0, f32x4_mul(cx, y0));
        let x1 = f32x4_add(x1, f32x4_mul(cx, y1));
        let x2 = f32x4_add(x2, f32x4_mul(cx, y2));
        let x3 = f32x4_add(x3, f32x4_mul(cx, y3));
        let b0 = f32x4_add(b0, f32x4_mul(cb, y0));
        let b1 = f32x4_add(b1, f32x4_mul(cb, y1));
        let b2 = f32x4_add(b2, f32x4_mul(cb, y2));
        let b3 = f32x4_add(b3, f32x4_mul(cb, y3));

        store(&mut x[0], x0);
        store(&mut x[1], x1);
        store(&mut x[2], x2);
        store(&mut x[3], x3);
        store(&mut b[0], b0);
        store(&mut b[1], b1);
        store(&mut b[2], b2);
        store(&mut b[3], b3);
    }

    for ((x, y), b) in x_chunks_tail
        .iter_mut()
        .zip(y_chunks_tail)
        .zip(b_chunks_tail)
    {
        let yv = load(y);
        store(x, f32x4_add(load(x), f32x4_mul(cx, yv)));
        store(b, f32x4_add(load(b), f32x4_mul(cb, yv)));
    }

    for ((x, &y), b) in x_tail.iter_mut().zip(y_tail).zip(b_tail) {
        *x -= cmap_factor[0] * y;
        *b -= cmap_factor[2] * y;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn apply_cfl_wasm_matches_scalar() {
        for size in [0, 1, 3, 4, 7, 16, 64, 1024, 4096] {
            let y: Vec<f32> = (0..size)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03125)
                .collect();
            let mut x: Vec<f32> = (0..size).map(|i| i as f32 * 0.015625 - 2.0).collect();
            let mut b: Vec<f32> = (0..size).map(|i| 3.0 - i as f32 * 0.0078125).collect();
            let mut want_x = x.clone();
            let mut want_b = b.clone();
            let factors: [f32; 3] = [0.3125, 0.0, -0.1875];
            for i in 0..size {
                want_x[i] += -factors[0] * y[i];
                want_b[i] += -factors[2] * y[i];
            }
            super::apply_cfl_wasm(&mut x, &y, &mut b, factors);
            assert_eq!(x, want_x, "X mismatch at size {size}");
            assert_eq!(b, want_b, "B mismatch at size {size}");
        }
    }
}
