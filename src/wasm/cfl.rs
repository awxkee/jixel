use core::arch::wasm32::*;

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
                want_x[i] = -factors[0] * y[i] + want_x[i];
                want_b[i] = -factors[2] * y[i] + want_b[i];
            }
            super::apply_cfl_wasm(&mut x, &y, &mut b, factors);
            assert_eq!(x, want_x, "X mismatch at size {size}");
            assert_eq!(b, want_b, "B mismatch at size {size}");
        }
    }
}
