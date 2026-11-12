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
use crate::coder_scratch::CoderScratch;
use crate::dct::fmla;
use crate::image::Image3F;
use crate::thread_pool::ThreadPool;
use std::sync::OnceLock;

pub(crate) const M00: f32 = 0.30;
pub(crate) const M02: f32 = 0.078;
pub(crate) const M01: f32 = 1.0 - M02 - M00;

pub(crate) const M10: f32 = 0.23;
pub(crate) const M12: f32 = 0.078;
pub(crate) const M11: f32 = 1.0 - M12 - M10;

pub(crate) const B_BIAS: f32 = 0.551_809_86;
const B_R_RATIO: f32 = 0.243_422_69 / (0.243_422_69 + 0.204_767_45);
pub(crate) const M20: f32 = B_R_RATIO * (1.0 - B_BIAS);
pub(crate) const M21: f32 = (1.0 - B_R_RATIO) * (1.0 - B_BIAS);
pub(crate) const M22: f32 = 1.0 - M20 - M21;

pub(crate) const OPSIN_INVERSE_MATRIX: [f32; 9] = [
    11.031567,
    -9.866944,
    -0.16462299,
    -3.2541473,
    4.4187703,
    -0.16462299,
    -3.6588514,
    2.712923,
    1.9459282,
];

// Red-variant B row: the blue-biased row (b_bias = 0.85), used only for
// strongly red-dominant content at high quality.
// pub(crate) const B_BIAS_RED: f32 = 0.85;
// pub(crate) const M20R: f32 = B_R_RATIO * (1.0 - B_BIAS_RED);
// pub(crate) const M21R: f32 = (1.0 - B_R_RATIO) * (1.0 - B_BIAS_RED);
// pub(crate) const M22R: f32 = 1.0 - M20R - M21R;
//
// pub(crate) const OPSIN_INVERSE_MATRIX_RED: [f32; 9] = [
//     10.785613059997559,
//     -9.684576988220215,
//     -0.10103627294301987,
//     -3.50010085105896,
//     4.601137161254883,
//     -0.10103627294301987,
//     -0.751554548740387,
//     0.5572540163993835,
//     1.1943005323410034,
// ];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct XybMatrix {
    pub(crate) fwd: [f32; 9],
    pub(crate) inv: [f32; 9],
}

impl XybMatrix {
    /// The default matrix (spec opsin as long as `B_BIAS` is unpatched).
    pub(crate) const SPEC: XybMatrix = XybMatrix {
        fwd: [M00, M01, M02, M10, M11, M12, M20, M21, M22],
        inv: OPSIN_INVERSE_MATRIX,
    };
    // /// Red-variant matrix for strongly red-dominant high-quality content.
    // pub(crate) const RED: XybMatrix = XybMatrix {
    //     fwd: [M00, M01, M02, M10, M11, M12, M20R, M21R, M22R],
    //     inv: OPSIN_INVERSE_MATRIX_RED,
    // };

    pub(crate) fn is_decoder_default(&self) -> bool {
        static SPEC_FWD: [f32; 9] = [
            0.30,
            0.622,
            0.078,
            0.23,
            0.692,
            0.078,
            0.243_422_69,
            0.204_767_45,
            0.551_809_86,
        ];
        self.fwd
            .iter()
            .zip(SPEC_FWD.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    }
}

// /// The red-variant row only wins below this distance.
// pub(crate) const RED_OPSIN_MAX_DISTANCE: f32 = 1.5;
//
// /// Detect strongly red-dominant content.
// pub(crate) fn is_red_dominant<T: Copy + Into<u64>, const N: usize>(input: &[T]) -> bool {
//     let px = input.len() / N;
//     if px == 0 {
//         return false;
//     }
//     let step = (px / 65536).max(1);
//     let (mut sr, mut sg, mut sb, mut red_px, mut n) = (0u64, 0u64, 0u64, 0u64, 0u64);
//     for src in input.as_chunks::<N>().0.iter().step_by(step) {
//         let r: u64 = src[0].into();
//         let g: u64 = src[1].into();
//         let b: u64 = src[2].into();
//         sr += r;
//         sg += g;
//         sb += b;
//         red_px += u64::from(r > 2 * (g + b));
//         n += 1;
//     }
//     sr > 3 * (sg + sb) && red_px * 10 > n * 3
// }

pub(crate) const OPSIN_BIAS: f32 = 0.003_793_073_4;
pub(crate) const NEG_BIAS_CBRT: f32 = -0.155_954_2;

#[allow(unused)]
#[inline(always)]
fn halley_refine(x: f32, a: f32) -> f32 {
    let tx = x * x * x;
    x * fmla(2.0, a, tx) / fmla(2.0, tx, a)
}

#[inline]
pub(crate) fn cbrtf(x: f32) -> f32 {
    if x == 0.0 {
        return x;
    }
    const B1: u32 = 709958130;
    let mut t: f32;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7fffffff;

    hx = (hx / 3).wrapping_add(B1);
    ui &= 0x80000000;
    ui |= hx;

    t = f32::from_bits(ui);
    t = halley_refine(t, x);
    halley_refine(t, x)
}

#[allow(unused)]
#[inline(always)]
pub(crate) fn rgb_to_xyb_pixel_f32(m: &XybMatrix, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m.fwd;
    let mixed0 = fmla(m00, r, fmla(m01, g, fmla(m02, b, OPSIN_BIAS)));
    let mixed1 = fmla(m10, r, fmla(m11, g, fmla(m12, b, OPSIN_BIAS)));
    let mixed2 = fmla(m20, r, fmla(m21, g, fmla(m22, b, OPSIN_BIAS)));

    let tm0 = cbrtf(mixed0.max(0.0)) + NEG_BIAS_CBRT;
    let tm1 = cbrtf(mixed1.max(0.0)) + NEG_BIAS_CBRT;
    let tm2 = cbrtf(mixed2.max(0.0)) + NEG_BIAS_CBRT;

    (0.5 * (tm0 - tm1), 0.5 * (tm0 + tm1), tm2)
}

type XybLatticeFn = unsafe fn(&XybMatrix, [&[f32]; 3], [&mut [i32]; 3], u32) -> usize;

fn select_xyb_lattice_fn() -> XybLatticeFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return crate::avx::xyb_lattice_avx2;
    }
    // The SSE kernel matches the unfused scalar arithmetic of generic x86 builds.
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        feature = "sse",
        not(target_feature = "fma")
    ))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::xyb_lattice_sse41;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return crate::neon::xyb_lattice_neon;
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    {
        crate::wasm::xyb_lattice_wasm
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")))]
    {
        |_, _, _, _| 0
    }
}

/// Refinement of the default XYB lattice in crop and ink frames.
pub(crate) const CROP_LATTICE_SCALE: u32 = 8;

#[inline]
pub(crate) fn scaled_lattice_quantizers(scale: u32) -> [f32; 3] {
    // The base quantizers are powers of two, so folding in the refinement
    // preserves rounding before conversion to lattice integers.
    crate::quant_weights::INV_DC_QUANT.map(|quantizer| quantizer * scale as f32)
}

/// Convert linear RGB to the XYB lattice refined `scale` times. Apply YCoCg
/// to the quantized (Y, X, B-Y) planes for modular coding.
/// The input and output planes must have equal lengths.
pub(crate) fn xyb_lattice_ycocg_scaled(
    m: &XybMatrix,
    input: [&[f32]; 3],
    output: [&mut [i32]; 3],
    scale: u32,
) {
    use crate::lossless::forward_ycocg;

    static LATTICE: OnceLock<XybLatticeFn> = OnceLock::new();
    let lattice = *LATTICE.get_or_init(select_xyb_lattice_fn);
    let [r, g, b] = input;
    let [y_out, co_out, cg_out] = output;
    assert!(
        [g.len(), b.len(), y_out.len(), co_out.len(), cg_out.len()]
            .iter()
            .all(|&len| len == r.len())
    );
    // SAFETY: dispatch selects supported CPU features and all six planes
    // have equal lengths. The AVX2 kernel masks any partial vector.
    let done = unsafe { lattice(m, input, [y_out, co_out, cg_out], scale) };
    let [sx, sy, sb] = scaled_lattice_quantizers(scale);
    let src = r[done..].iter().zip(&g[done..]).zip(&b[done..]);
    let dst = y_out[done..]
        .iter_mut()
        .zip(&mut co_out[done..])
        .zip(&mut cg_out[done..]);
    for (((r, g), b), ((y_out, co_out), cg_out)) in src.zip(dst) {
        let (x, y, b) = rgb_to_xyb_pixel_f32(m, *r, *g, *b);
        let yq = (y * sy).round() as i32;
        (*y_out, *co_out, *cg_out) =
            forward_ycocg(yq, (x * sx).round() as i32, (b * sb).round() as i32 - yq);
    }
}

/// Inverse of [`rgb_to_xyb_pixel_f32`], using the active signalled matrix.
/// This matches the decoder's opsin inverse apart from its f16 matrix storage.
#[inline]
pub(crate) fn xyb_to_rgb_pixel_f32(m: &XybMatrix, x: f32, y: f32, b: f32) -> [f32; 3] {
    let tm0 = y + x;
    let tm1 = y - x;
    let tm2 = b;
    let cube = |t: f32| {
        let c = t - NEG_BIAS_CBRT;
        c * c * c - OPSIN_BIAS
    };
    let m0 = cube(tm0);
    let m1 = cube(tm1);
    let m2 = cube(tm2);
    let [i00, i01, i02, i10, i11, i12, i20, i21, i22] = m.inv;
    [
        fmla(i00, m0, fmla(i01, m1, i02 * m2)),
        fmla(i10, m0, fmla(i11, m1, i12 * m2)),
        fmla(i20, m0, fmla(i21, m1, i22 * m2)),
    ]
}

/// Row kernels return the SIMD prefix; the driver applies the scalar inverse
/// to the tail, preserving its operation order and fused/unfused arithmetic.
type ToLinearRowsFn = unsafe fn(&XybMatrix, [&[f32]; 3], [&mut [f32]; 3]) -> usize;

fn to_linear_rows_scalar(
    m: &XybMatrix,
    [x, y, b]: [&[f32]; 3],
    [r, g, out_b]: [&mut [f32]; 3],
) -> usize {
    for (((((x, y), b), r), g), out_b) in x.iter().zip(y).zip(b).zip(r).zip(g).zip(out_b) {
        [*r, *g, *out_b] = xyb_to_rgb_pixel_f32(m, *x, *y, *b);
    }
    x.len()
}

fn selected_to_linear_rows_fn() -> ToLinearRowsFn {
    static ROWS: OnceLock<ToLinearRowsFn> = OnceLock::new();
    *ROWS.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            return crate::avx::to_linear_rows_avx2;
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse4.1") {
            return crate::sse::to_linear_rows_sse41;
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            crate::neon::to_linear_rows_neon
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            crate::wasm::to_linear_rows_wasm
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
        )))]
        to_linear_rows_scalar
    })
}

#[cfg_attr(not(feature = "extraslow"), allow(dead_code))]
/// Planar XYB → linear RGB, with runtime SIMD dispatch and scalar tails.
/// Input and output rows may be cropped and unaligned, but must have equal lengths.
pub(crate) fn xyb_to_rgb_rows(m: &XybMatrix, input: [&[f32]; 3], output: [&mut [f32]; 3]) {
    let n = input[0].len();
    assert!(input.iter().all(|row| row.len() == n));
    assert!(output.iter().all(|row| row.len() == n));
    let rows = selected_to_linear_rows_fn();
    let [r, g, b] = output;
    // SAFETY: runtime dispatch selects supported CPU features, and all six
    // slices have `n` elements. The SIMD kernels process only complete vectors.
    let done = unsafe { rows(m, input, [r, g, b]) };
    if done < n {
        to_linear_rows_scalar(
            m,
            input.map(|row| &row[done..]),
            [&mut r[done..], &mut g[done..], &mut b[done..]],
        );
    }
}

pub(crate) type ToXybBandFn = unsafe fn(&XybMatrix, [&[f32]; 3], [&mut [f32]; 3], usize);

fn select_to_xyb_band_fn() -> ToXybBandFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return crate::avx::to_xyb_avx2_band;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::to_xyb_sse41_band;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return crate::neon::to_xyb_neon_band;
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::to_xyb_wasm_band;
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        to_xyb_f32_band
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
)))]
fn to_xyb_f32_band(m: &XybMatrix, input: [&[f32]; 3], output: [&mut [f32]; 3], _w: usize) {
    let [rp, gp, bp] = input;
    let [xp, yp, out_bp] = output;
    for (((((r, g), b), x), y), out_b) in rp
        .iter()
        .zip(gp.iter())
        .zip(bp.iter())
        .zip(xp.iter_mut())
        .zip(yp.iter_mut())
        .zip(out_bp.iter_mut())
    {
        (*x, *y, *out_b) = rgb_to_xyb_pixel_f32(m, *r, *g, *b);
    }
}

static TO_XYB_BAND_FN: OnceLock<ToXybBandFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_to_xyb_band_fn() -> ToXybBandFn {
    *TO_XYB_BAND_FN.get_or_init(select_to_xyb_band_fn)
}

#[cfg(test)]
pub(crate) fn to_xyb_with_fn(
    f: ToXybBandFn,
    m: &XybMatrix,
    linear: &Image3F,
    xyb: &mut Image3F,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) {
    debug_assert_eq!(linear.xsize(), xyb.xsize());
    debug_assert_eq!(linear.ysize(), xyb.ysize());

    let w = linear.xsize();
    let input = [
        linear.plane_data(0),
        linear.plane_data(1),
        linear.plane_data(2),
    ];
    let mut offset = 0;
    let mut jobs: Vec<_> = xyb
        .row_bands_mut(pool.num_threads())
        .into_iter()
        .map(|output| {
            let end = offset + output[0].len();
            let input = [
                &input[0][offset..end],
                &input[1][offset..end],
                &input[2][offset..end],
            ];
            offset = end;
            Some((input, output))
        })
        .collect();

    pool.steal_for_each_mut(scratch, &mut jobs, |_i, job, _scratch| {
        let (input, output) = job.take().unwrap();
        unsafe { f(m, input, output, w) };
    });
}

/// Reuse the owned RGB planes for XYB. A worker stages one input row so the
/// existing SIMD kernels read and write disjoint slices, including scalar tails.
pub(crate) fn to_xyb_in_place_with_fn(
    f: ToXybBandFn,
    m: &XybMatrix,
    image: &mut Image3F,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) {
    let w = image.xsize();
    let mut jobs: Vec<_> = image
        .row_bands_mut(pool.num_threads())
        .into_iter()
        .map(Some)
        .collect();
    pool.steal_for_each_mut(scratch, &mut jobs, |_i, job, _scratch| {
        let [r, g, b] = job.take().unwrap();
        let mut row = vec![0.0; 3 * w];
        let (r_in, rest) = row.split_at_mut(w);
        let (g_in, b_in) = rest.split_at_mut(w);
        for ((r, g), b) in r
            .chunks_exact_mut(w)
            .zip(g.chunks_exact_mut(w))
            .zip(b.chunks_exact_mut(w))
        {
            r_in.copy_from_slice(r);
            g_in.copy_from_slice(g);
            b_in.copy_from_slice(b);
            // SAFETY: dispatch selects supported CPU features. The input row
            // is staged separately from the three non-overlapping output rows.
            unsafe { f(m, [r_in, g_in, b_in], [r, g, b], w) };
        }
    });
}

pub(crate) type QuantizeXybChannelsFn = unsafe fn([&[f32]; 3], [&mut [i32]; 3], [f32; 3]);

#[cfg(not(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
)))]
fn quantize_xyb_channels_scalar(input: [&[f32]; 3], output: [&mut [i32]; 3], scales: [f32; 3]) {
    let [src_x, src_y, src_b] = input;
    let [dst_y, dst_x, dst_b] = output;
    let [scale_x, scale_y, scale_b] = scales;
    let src = src_x.iter().zip(src_y).zip(src_b);
    let dst = dst_y.iter_mut().zip(dst_x).zip(dst_b);
    for (((x, y), b), ((out_y, out_x), out_b)) in src.zip(dst) {
        let yq = (*y * scale_y).round() as i32;
        *out_y = yq;
        *out_x = (*x * scale_x).round() as i32;
        *out_b = (*b * scale_b).round() as i32 - yq;
    }
}

fn select_quantize_xyb_channels_fn() -> QuantizeXybChannelsFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        return crate::avx::quantize_xyb_channels_avx2;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::quantize_xyb_channels_sse41;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return crate::neon::quantize_xyb_channels_neon;
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::quantize_xyb_channels_wasm;
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    quantize_xyb_channels_scalar
}

static QUANTIZE_XYB_CHANNELS_FN: OnceLock<QuantizeXybChannelsFn> = OnceLock::new();

#[inline]
fn selected_quantize_xyb_channels_fn() -> QuantizeXybChannelsFn {
    *QUANTIZE_XYB_CHANNELS_FN.get_or_init(select_quantize_xyb_channels_fn)
}

pub(crate) type RoundLatticePow2Fn = unsafe fn(&mut [i32], u32);

/// `round_lattice_pow2` without SIMD; also the tails of the SIMD kernels.
pub(crate) fn round_lattice_pow2_scalar(plane: &mut [i32], shift: u32) {
    for v in plane.iter_mut() {
        *v = round_lattice_pow2_value(*v, shift);
    }
}

#[inline]
pub(crate) fn round_lattice_pow2_value(value: i32, shift: u32) -> i32 {
    let step = 1i32 << shift;
    let sign = value >> 31;
    let rounded = value.wrapping_abs().wrapping_add(step >> 1) & -step;
    (rounded ^ sign).wrapping_sub(sign)
}

fn select_round_lattice_pow2_fn() -> RoundLatticePow2Fn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        return crate::avx::round_lattice_pow2_avx2;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse2") {
        return crate::sse::round_lattice_pow2_sse2;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return crate::neon::round_lattice_pow2_neon;
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::round_lattice_pow2_wasm;
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    round_lattice_pow2_scalar
}

static ROUND_LATTICE_POW2_FN: OnceLock<RoundLatticePow2Fn> = OnceLock::new();

/// Rounds lattice values to multiples of `1 << shift`, halves away from zero
/// (`f32::round` semantics, without float conversion or libm).
#[inline]
pub(crate) fn selected_round_lattice_pow2_fn() -> RoundLatticePow2Fn {
    *ROUND_LATTICE_POW2_FN.get_or_init(select_round_lattice_pow2_fn)
}

/// XYB samples quantized onto the fixed modular-XYB integer lattice.
pub(crate) fn quantize_xyb_channels(atlas: &Image3F, lattice_scale: u32) -> [Vec<i32>; 3] {
    use crate::quant_weights::INV_DC_QUANT;
    let m = lattice_scale as f32;
    quantize_xyb_channels_with_scales(atlas, INV_DC_QUANT.map(|scale| scale * m))
}

/// Quantize contiguous X, Y and B planes into the stored Y, X and B-Y order.
/// The scales are given in input channel order.
pub(crate) fn quantize_xyb_channels_with_scales(
    image: &Image3F,
    scales: [f32; 3],
) -> [Vec<i32>; 3] {
    let n = image.xsize() * image.ysize();
    let input = [0, 1, 2].map(|c| &image.plane_data(c)[..n]);
    let mut output = [vec![0; n], vec![0; n], vec![0; n]];
    // SAFETY: the dispatcher selects a supported ISA, and all planes have n samples.
    unsafe {
        selected_quantize_xyb_channels_fn()(
            input,
            output.each_mut().map(Vec::as_mut_slice),
            scales,
        );
    }
    output
}

pub(crate) type QuantizeXybTileColorsFn = unsafe fn([&[f32]; 3], &mut [[i32; 3]], [f32; 3]);

#[cfg(not(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
)))]
fn quantize_xyb_tile_colors_scalar(input: [&[f32]; 3], output: &mut [[i32; 3]], scales: [f32; 3]) {
    let [src_x, src_y, src_b] = input;
    let [scale_x, scale_y, scale_b] = scales;
    let src = src_x.iter().zip(src_y).zip(src_b);
    for (((x, y), b), out) in src.zip(output) {
        *out = [
            (*y * scale_y).round() as i32,
            (*x * scale_x).round() as i32,
            (*b * scale_b).round() as i32,
        ];
    }
}

fn select_quantize_xyb_tile_colors_fn() -> QuantizeXybTileColorsFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        return crate::avx::quantize_xyb_tile_colors_avx2;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        return crate::sse::quantize_xyb_tile_colors_sse41;
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return crate::neon::quantize_xyb_tile_colors_neon;
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::quantize_xyb_tile_colors_wasm;
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128")
    )))]
    quantize_xyb_tile_colors_scalar
}

static QUANTIZE_XYB_TILE_COLORS_FN: OnceLock<QuantizeXybTileColorsFn> = OnceLock::new();

#[inline]
pub(crate) fn selected_quantize_xyb_tile_colors_fn() -> QuantizeXybTileColorsFn {
    *QUANTIZE_XYB_TILE_COLORS_FN.get_or_init(select_quantize_xyb_tile_colors_fn)
}

pub(crate) fn quantize_xyb_tile_colors(
    xyb: &Image3F,
    x0: usize,
    y0: usize,
    lattice_scale: u32,
    output: &mut [[i32; 3]; crate::patches::PATCH_TILE * crate::patches::PATCH_TILE],
) {
    use crate::patches::PATCH_TILE;
    use crate::quant_weights::INV_DC_QUANT;

    let m = lattice_scale as f32;
    let scales = [
        INV_DC_QUANT[0] * m,
        INV_DC_QUANT[1] * m,
        INV_DC_QUANT[2] * m,
    ];
    let quantize = selected_quantize_xyb_tile_colors_fn();
    for (dy, out_row) in output
        .as_chunks_mut::<PATCH_TILE>()
        .0
        .iter_mut()
        .enumerate()
    {
        let input = [
            &xyb.plane_row(0, y0 + dy)[x0..x0 + PATCH_TILE],
            &xyb.plane_row(1, y0 + dy)[x0..x0 + PATCH_TILE],
            &xyb.plane_row(2, y0 + dy)[x0..x0 + PATCH_TILE],
        ];
        unsafe { quantize(input, out_row, scales) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refined_lattice_matches_scalar_across_vector_tails() {
        use crate::lossless::forward_ycocg;
        use crate::quant_weights::INV_DC_QUANT;

        const GUARD: i32 = i32::MIN;
        let mut custom = XybMatrix::SPEC;
        custom.fwd[6..].copy_from_slice(&[0.1, 0.2, 0.7]);
        let rgb: [[f32; 258]; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((i * (97 + 4 * c) + c * 113) % 1031) as f32 / 255.0 - 0.5)
        });
        for matrix in [XybMatrix::SPEC, custom] {
            for scale in [0, 2, 3, CROP_LATTICE_SCALE] {
                for n in (0..=33).chain([255, 256, 257]) {
                    let input = rgb.each_ref().map(|p| &p[1..1 + n]);
                    let mut output = [[GUARD; 259]; 3];
                    xyb_lattice_ycocg_scaled(
                        &matrix,
                        input,
                        output.each_mut().map(|p| &mut p[1..1 + n]),
                        scale,
                    );
                    for i in 0..n {
                        let (x, y, b) =
                            rgb_to_xyb_pixel_f32(&matrix, input[0][i], input[1][i], input[2][i]);
                        let quantize =
                            |v: f32, c: usize| (v * INV_DC_QUANT[c] * scale as f32).round() as i32;
                        let y = quantize(y, 1);
                        let (luma, co, cg) = forward_ycocg(y, quantize(x, 0), quantize(b, 2) - y);
                        assert_eq!(
                            output.each_ref().map(|p| p[i + 1]),
                            [luma, co, cg],
                            "scale={scale}, len={n}, pixel={i}",
                        );
                    }
                    for p in output {
                        assert_eq!(p[0], GUARD);
                        assert!(p[1 + n..].iter().all(|&v| v == GUARD));
                    }
                }
            }
        }
    }

    fn inverse_row_kernels() -> Vec<(&'static str, ToLinearRowsFn)> {
        #[allow(unused_mut)]
        let mut kernels: Vec<(&str, ToLinearRowsFn)> = vec![
            ("dispatch", selected_to_linear_rows_fn()),
            ("scalar", to_linear_rows_scalar),
        ];
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            kernels.push(("neon", crate::neon::to_linear_rows_neon));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", crate::avx::to_linear_rows_avx2));
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse4.1") {
            kernels.push(("sse4.1", crate::sse::to_linear_rows_sse41));
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        kernels.push(("wasm", crate::wasm::to_linear_rows_wasm));
        kernels
    }

    fn check_inverse_rows(m: &XybMatrix, input: [&[f32]; 3], offset: usize) {
        const GUARD: f32 = -12345.0;
        let n = input[0].len();
        let expected: Vec<[f32; 3]> = (0..n)
            .map(|i| xyb_to_rgb_pixel_f32(m, input[0][i], input[1][i], input[2][i]))
            .collect();
        for (name, rows) in inverse_row_kernels() {
            let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![GUARD; offset + n + 4]);
            let [r, g, b] = &mut out;
            // SAFETY: the inventory checks the kernel's CPU features, and
            // all six slices have `n` elements, including empty rows.
            let done = unsafe {
                rows(
                    m,
                    input,
                    [
                        &mut r[offset..offset + n],
                        &mut g[offset..offset + n],
                        &mut b[offset..offset + n],
                    ],
                )
            };
            assert!(done <= n, "{name}: processed {done} of {n}");
            to_linear_rows_scalar(
                m,
                input.map(|p| &p[done..]),
                [
                    &mut r[offset + done..offset + n],
                    &mut g[offset + done..offset + n],
                    &mut b[offset + done..offset + n],
                ],
            );
            for c in 0..3 {
                for (i, rgb) in expected.iter().enumerate() {
                    let actual = out[c][offset + i];
                    if rgb[c].is_nan() {
                        assert!(
                            actual.is_nan(),
                            "{name}: n={n}, offset={offset}, channel={c}, pixel={i}"
                        );
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            rgb[c].to_bits(),
                            "{name}: n={n}, offset={offset}, channel={c}, pixel={i}"
                        );
                    }
                }
                assert!(
                    out[c][..offset]
                        .iter()
                        .chain(&out[c][offset + n..])
                        .all(|&v| v == GUARD),
                    "{name}: output guards"
                );
            }
        }
    }

    #[test]
    fn inverse_rows_match_scalar_for_all_tails_alignment_and_matrices() {
        let custom = XybMatrix {
            fwd: XybMatrix::SPEC.fwd,
            inv: std::array::from_fn(|i| XybMatrix::SPEC.inv[i] + (i as f32 - 4.0) * 0.125),
        };
        for m in [XybMatrix::SPEC, custom] {
            for n in (0..=65).chain([95, 127, 128, 129, 257]) {
                for offset in 0..4 {
                    let input: [Vec<f32>; 3] = std::array::from_fn(|c| {
                        (0..n + 4)
                            .map(|i| {
                                let noise = ((i * 71 + c * 113) % 101) as f32 / 101.0;
                                if c == 0 {
                                    0.16 * noise - 0.08
                                } else {
                                    1.5 * noise - 0.2
                                }
                            })
                            .collect()
                    });
                    check_inverse_rows(
                        &m,
                        input.each_ref().map(|p| &p[offset..offset + n]),
                        offset,
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_rows_preserve_signed_values_hdr_and_extreme_arithmetic() {
        let values = [
            -0.0,
            0.0,
            NEG_BIAS_CBRT,
            -NEG_BIAS_CBRT,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            -2.0,
            2.0,
            16.0,
            -16.0,
            1e12,
            -1e12,
            1e13,
            -1e13,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for c in 0..3 {
            let input: [Vec<f32>; 3] = std::array::from_fn(|channel| {
                (0..128)
                    .map(|i| {
                        if c == channel {
                            values[i % values.len()]
                        } else {
                            0.3
                        }
                    })
                    .collect()
            });
            check_inverse_rows(&XybMatrix::SPEC, input.each_ref().map(|p| &p[..]), 1);
        }
    }

    #[test]
    #[should_panic]
    fn inverse_rows_reject_mismatched_inputs() {
        xyb_to_rgb_rows(
            &XybMatrix::SPEC,
            [&[0.0; 8], &[0.0; 7], &[0.0; 8]],
            [&mut [0.0; 8], &mut [0.0; 8], &mut [0.0; 8]],
        );
    }

    #[test]
    #[should_panic]
    fn inverse_rows_reject_mismatched_outputs() {
        xyb_to_rgb_rows(
            &XybMatrix::SPEC,
            [&[0.0; 8]; 3],
            [&mut [0.0; 7], &mut [0.0; 8], &mut [0.0; 8]],
        );
    }

    unsafe fn scalar_band(m: &XybMatrix, input: [&[f32]; 3], output: [&mut [f32]; 3], _w: usize) {
        let [r, g, b] = input;
        let [x, y, out_b] = output;
        for i in 0..r.len() {
            (x[i], y[i], out_b[i]) = rgb_to_xyb_pixel_f32(m, r[i], g[i], b[i]);
        }
    }

    #[test]
    fn out_of_place_conversion_preserves_linear_input() {
        let mut linear = Image3F::new(3, 2);
        for i in 0..6 {
            linear.plane_row_mut(0, i / 3)[i % 3] = i as f32 / 7.0;
            linear.plane_row_mut(1, i / 3)[i % 3] = (i + 1) as f32 / 8.0;
            linear.plane_row_mut(2, i / 3)[i % 3] = (i + 2) as f32 / 9.0;
        }
        let mut xyb = Image3F::new(3, 2);
        let pool = ThreadPool::new(2);
        let mut scratch = CoderScratch::default();

        to_xyb_with_fn(
            scalar_band,
            &XybMatrix::SPEC,
            &linear,
            &mut xyb,
            &pool,
            &mut scratch,
        );

        for i in 0..6 {
            assert_eq!(linear.plane_data(0)[i], i as f32 / 7.0);
            assert_eq!(linear.plane_data(1)[i], (i + 1) as f32 / 8.0);
            assert_eq!(linear.plane_data(2)[i], (i + 2) as f32 / 9.0);
            let (x, y, b) = rgb_to_xyb_pixel_f32(
                &XybMatrix::SPEC,
                linear.plane_data(0)[i],
                linear.plane_data(1)[i],
                linear.plane_data(2)[i],
            );
            assert_eq!(xyb.plane_data(0)[i], x);
            assert_eq!(xyb.plane_data(1)[i], y);
            assert_eq!(xyb.plane_data(2)[i], b);
        }
    }

    #[test]
    fn in_place_conversion_matches_separate_planes_and_reuses_storage() {
        for threads in [1, 4] {
            let pool = ThreadPool::new(threads);
            let mut scratch = CoderScratch::default();
            for (w, h) in [(1, 1), (3, 7), (17, 3), (129, 19), (2049, 2)] {
                let mut linear = Image3F::new(w, h);
                for c in 0..3 {
                    for (i, v) in linear.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                        // Include signed and HDR inputs as well as SDR values.
                        *v = ((i * 97 + c * 13) % 1031) as f32 / 255.0 - 0.5;
                    }
                }
                for f in [scalar_band as ToXybBandFn, selected_to_xyb_band_fn()] {
                    let mut expected = Image3F::new(w, h);
                    to_xyb_with_fn(
                        f,
                        &XybMatrix::SPEC,
                        &linear,
                        &mut expected,
                        &pool,
                        &mut scratch,
                    );
                    let mut actual = linear.clone();
                    let pointers = [0, 1, 2].map(|c| actual.plane_data(c).as_ptr());
                    to_xyb_in_place_with_fn(f, &XybMatrix::SPEC, &mut actual, &pool, &mut scratch);
                    for (c, pointer) in pointers.into_iter().enumerate() {
                        assert_eq!(actual.plane_data(c).as_ptr(), pointer);
                        assert!(
                            actual
                                .plane_data(c)
                                .iter()
                                .zip(expected.plane_data(c))
                                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
                        );
                    }
                }
            }
        }
    }

    fn check_xyb_quantizer(f: QuantizeXybChannelsFn) {
        let values = [
            -8_388_609.0,
            -3.5,
            -2.5,
            -1.5,
            -0.51,
            -0.5,
            -0.49,
            -0.0,
            0.0,
            0.49,
            0.5,
            0.51,
            1.5,
            2.5,
            3.5,
            7.25,
            8_388_609.0,
        ];
        let src_x = values;
        let src_y = values.map(|v| v * 0.5);
        let src_b = values.map(|v| v * -0.25);
        let mut dst_y = [0; 17];
        let mut dst_x = [0; 17];
        let mut dst_b = [0; 17];

        unsafe {
            f(
                [&src_x, &src_y, &src_b],
                [&mut dst_y, &mut dst_x, &mut dst_b],
                [1.0, 1.0, 1.0],
            );
        }

        for i in 0..values.len() {
            let yq = src_y[i].round() as i32;
            assert_eq!(dst_y[i], yq, "Y lane {i}");
            assert_eq!(dst_x[i], src_x[i].round() as i32, "X lane {i}");
            assert_eq!(dst_b[i], src_b[i].round() as i32 - yq, "B-Y lane {i}");
        }

        let special_x = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2_147_483_648.0,
            -2_147_483_904.0,
            2_147_483_520.0,
            -2_147_483_648.0,
            123.5,
        ];
        let zero = [0.0; 8];
        let mut special_yq = [1; 8];
        let mut special_xq = [1; 8];
        let mut special_bq = [1; 8];
        unsafe {
            f(
                [&special_x, &zero, &zero],
                [&mut special_yq, &mut special_xq, &mut special_bq],
                [1.0, 1.0, 1.0],
            );
        }
        for i in 0..special_x.len() {
            assert_eq!(special_yq[i], 0, "special Y lane {i}");
            assert_eq!(
                special_xq[i],
                special_x[i].round() as i32,
                "special X lane {i}"
            );
            assert_eq!(special_bq[i], 0, "special B-Y lane {i}");
        }

        let tail_x = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5];
        let tail_y = [20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5];
        let tail_b = [30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5];
        for len in 1..8 {
            let mut tail_yq = [i32::MIN; 7];
            let mut tail_xq = [i32::MIN; 7];
            let mut tail_bq = [i32::MIN; 7];
            unsafe {
                f(
                    [&tail_x[..len], &tail_y[..len], &tail_b[..len]],
                    [
                        &mut tail_yq[..len],
                        &mut tail_xq[..len],
                        &mut tail_bq[..len],
                    ],
                    [1.0; 3],
                )
            };
            for i in 0..len {
                let yq = tail_y[i].round() as i32;
                assert_eq!(tail_yq[i], yq, "tail Y lane {i} of {len}");
                assert_eq!(
                    tail_xq[i],
                    tail_x[i].round() as i32,
                    "tail X lane {i} of {len}"
                );
                assert_eq!(
                    tail_bq[i],
                    tail_b[i].round() as i32 - yq,
                    "tail B-Y lane {i} of {len}"
                );
            }
            assert!(tail_yq[len..].iter().all(|&v| v == i32::MIN));
            assert!(tail_xq[len..].iter().all(|&v| v == i32::MIN));
            assert!(tail_bq[len..].iter().all(|&v| v == i32::MIN));
        }
    }

    fn check_xyb_tile_color_quantizer(f: QuantizeXybTileColorsFn) {
        let src_x = [
            -3.5, -2.5, -1.5, -0.51, -0.5, -0.49, -0.0, 0.0, 0.49, 0.5, 0.51, 1.5, 2.5, 3.5, 7.25,
            11.75, 19.5,
        ];
        let src_y = src_x.map(|v| v * 0.25);
        let src_b = src_x.map(|v| v * -0.75);
        let scales = [1.5, 2.0, 0.5];
        let mut output = [[0; 3]; 17];

        unsafe { f([&src_x, &src_y, &src_b], &mut output, scales) };

        for i in 0..src_x.len() {
            assert_eq!(
                output[i],
                [
                    (src_y[i] * scales[1]).round() as i32,
                    (src_x[i] * scales[0]).round() as i32,
                    (src_b[i] * scales[2]).round() as i32,
                ],
                "tile-color lane {i}"
            );
        }

        let tagged_x = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let tagged_y = [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0];
        let tagged_b = [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0];
        let mut tagged_output = [[0; 3]; 8];
        unsafe {
            f(
                [&tagged_x, &tagged_y, &tagged_b],
                &mut tagged_output,
                [1.0; 3],
            )
        };
        for i in 0..tagged_output.len() {
            assert_eq!(
                tagged_output[i],
                [tagged_y[i] as i32, tagged_x[i] as i32, tagged_b[i] as i32],
                "tagged tile-color lane {i}"
            );
        }

        for len in 1..8 {
            let mut tail_output = [[i32::MIN; 3]; 7];
            unsafe {
                f(
                    [&src_x[..len], &src_y[..len], &src_b[..len]],
                    &mut tail_output[..len],
                    scales,
                )
            };
            for i in 0..len {
                assert_eq!(
                    tail_output[i],
                    [
                        (src_y[i] * scales[1]).round() as i32,
                        (src_x[i] * scales[0]).round() as i32,
                        (src_b[i] * scales[2]).round() as i32,
                    ],
                    "tail tile-color lane {i} of {len}"
                );
            }
            assert!(
                tail_output[len..]
                    .iter()
                    .all(|&pixel| pixel == [i32::MIN; 3])
            );
        }
    }

    #[test]
    fn selected_xyb_quantizer_matches_round_ties_away() {
        check_xyb_quantizer(selected_quantize_xyb_channels_fn());
        check_xyb_tile_color_quantizer(selected_quantize_xyb_tile_colors_fn());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_xyb_quantizer_matches_round_ties_away() {
        if std::is_x86_feature_detected!("avx2") {
            check_xyb_quantizer(crate::avx::quantize_xyb_channels_avx2);
            check_xyb_tile_color_quantizer(crate::avx::quantize_xyb_tile_colors_avx2);
        }
    }

    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    #[test]
    fn sse41_xyb_quantizer_matches_round_ties_away() {
        if std::is_x86_feature_detected!("sse4.1") {
            check_xyb_quantizer(crate::sse::quantize_xyb_channels_sse41);
            check_xyb_tile_color_quantizer(crate::sse::quantize_xyb_tile_colors_sse41);
        }
    }
}
