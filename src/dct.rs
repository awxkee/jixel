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
use std::sync::{Arc, OnceLock};

#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    target_arch = "aarch64"
))]
#[inline(always)]
#[allow(unused)]
pub(crate) fn fmla(a: f32, b: f32, c: f32) -> f32 {
    f32::mul_add(a, b, c)
}

#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    target_arch = "aarch64"
)))]
#[inline(always)]
#[allow(unused)]
pub(crate) fn fmla(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

pub(crate) const WC4: [f32; 2] = [0.541_196_1, 1.306_563];

pub(crate) const WC8: [f32; 4] = [0.509_795_6, 0.601_344_9, 0.899_976_2, 2.562_915_6];

#[allow(unused)]
#[inline(always)]
fn dct1d_2(buf: &mut [f32]) {
    let a = buf[0];
    let b = buf[1];
    buf[0] = a + b;
    buf[1] = a - b;
}

#[allow(unused)]
#[inline(always)]
fn dct1d_4(buf: &mut [f32; 4]) {
    let mut tmp = [0.0f32; 4];
    tmp[0] = buf[0] + buf[3];
    tmp[1] = buf[1] + buf[2];
    dct1d_2(&mut tmp[0..2]);
    tmp[2] = buf[0] - buf[3];
    tmp[3] = buf[1] - buf[2];
    tmp[2] *= WC4[0];
    tmp[3] *= WC4[1];
    dct1d_2(&mut tmp[2..4]);
    tmp[2] = fmla(tmp[2], std::f32::consts::SQRT_2, tmp[3]);
    buf[0] = tmp[0];
    buf[2] = tmp[1];
    buf[1] = tmp[2];
    buf[3] = tmp[3];
}

#[inline(always)]
#[allow(unused)]
fn dct1d_8(buf: &mut [f32]) {
    let mut tmp = [0.0f32; 8];
    for i in 0..4 {
        tmp[i] = buf[i] + buf[7 - i];
    }
    dct1d_4(<&mut [f32; 4]>::try_from(&mut tmp[..4]).unwrap());
    for i in 0..4 {
        tmp[4 + i] = (buf[i] - buf[7 - i]) * WC8[i];
    }
    dct1d_4(<&mut [f32; 4]>::try_from(&mut tmp[4..8]).unwrap());
    tmp[4] = fmla(tmp[4], std::f32::consts::SQRT_2, tmp[5]);
    tmp[5] += tmp[6];
    tmp[6] += tmp[7];
    for i in 0..4 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[4 + i];
    }
}

pub(crate) type DctFn<const N: usize> = dyn Fn(&[f32; N], &mut [f32; N]) + Send + Sync;

static DCT_METHOD: OnceLock<Arc<DctFn<64>>> = OnceLock::new();

fn select_dct() -> Arc<DctFn<64>> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") {
            use crate::neon::dct8x8_neon;
            return Arc::new(|input, output| unsafe {
                dct8x8_neon(input, output);
            });
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Arc::new(|input, output| unsafe {
                crate::avx::dct8x8_avx2(input, output);
            });
        }
    }

    Arc::new(|input, output| {
        dct8x8_scalar(input, output);
    })
}

#[inline]
pub(crate) fn dct8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    DCT_METHOD.get_or_init(select_dct)(input, output);
}

pub(crate) fn dct8x8_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
    let mut tmp = [0.0f32; 64];

    for (src_row, tmp) in input
        .as_chunks::<8>()
        .0
        .iter()
        .zip(tmp.as_chunks_mut::<8>().0.iter_mut())
    {
        let mut row = [0.0f32; 8];
        for (dst, src) in row.iter_mut().zip(src_row.iter()) {
            *dst = *src;
        }
        dct1d_8(&mut row);
        for (dst, src) in tmp.iter_mut().zip(row.iter()) {
            *dst = *src;
        }
    }

    for (x, out_row) in output.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut col = [0.0f32; 8];
        for (col_slot, tmp_row) in col.iter_mut().zip(tmp.chunks_exact(8)) {
            *col_slot = tmp_row[x];
        }
        dct1d_8(&mut col);
        for (dst, src) in out_row.iter_mut().zip(col.iter()) {
            *dst = *src * (1.0 / 64.0);
        }
    }
}

pub(crate) const WC16: [f32; 8] = [
    0.502_419_3, // 1/(2·cos( π/32))
    0.522_498_6, // 1/(2·cos(3π/32))
    0.566_944,   // 1/(2·cos(5π/32))
    0.646_821_8, // 1/(2·cos(7π/32))
    0.788_154_6, // 1/(2·cos(9π/32))
    1.060_677_7, // 1/(2·cos(11π/32))
    1.722_447_1, // 1/(2·cos(13π/32))
    5.101_148_6, // 1/(2·cos(15π/32))
];

#[inline(always)]
#[allow(unused)]
pub(crate) fn dct1d_16(buf: &mut [f32; 16]) {
    let mut tmp = [0.0f32; 16];

    for i in 0..8 {
        tmp[i] = buf[i] + buf[15 - i];
        tmp[8 + i] = buf[i] - buf[15 - i];
    }

    // Recurse on the even half
    dct1d_8(&mut tmp[0..8]);

    // Scale the odd half by WC16, then recurse
    for i in 0..8 {
        tmp[8 + i] *= WC16[i];
    }
    dct1d_8(&mut tmp[8..16]);

    tmp[8] = fmla(tmp[8], std::f32::consts::SQRT_2, tmp[9]);
    tmp[9] += tmp[10];
    tmp[10] += tmp[11];
    tmp[11] += tmp[12];
    tmp[12] += tmp[13];
    tmp[13] += tmp[14];
    tmp[14] += tmp[15];

    for i in 0..8 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[8 + i];
    }
}

#[inline(always)]
#[allow(unused)]
pub(crate) fn dct1d_16_oof(src: &[f32; 16], buf: &mut [f32; 16]) {
    let mut tmp = [0.0f32; 16];

    for i in 0..8 {
        tmp[i] = src[i] + src[15 - i];
        tmp[8 + i] = src[i] - src[15 - i];
    }

    // Recurse on the even half
    dct1d_8(&mut tmp[0..8]);

    // Scale the odd half by WC16, then recurse
    for i in 0..8 {
        tmp[8 + i] *= WC16[i];
    }
    dct1d_8(&mut tmp[8..16]);

    tmp[8] = fmla(tmp[8], std::f32::consts::SQRT_2, tmp[9]);
    tmp[9] += tmp[10];
    tmp[10] += tmp[11];
    tmp[11] += tmp[12];
    tmp[12] += tmp[13];
    tmp[13] += tmp[14];
    tmp[14] += tmp[15];

    // Interleave even (dc) and odd (ac) halves into output
    for i in 0..8 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[8 + i];
    }
}

fn select_dct_8x16() -> Arc<DctFn<128>> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") {
            use crate::neon::dct8x16_neon;
            return Arc::new(|input, output| unsafe {
                dct8x16_neon(input, output);
            });
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Arc::new(|input, output| unsafe {
                crate::avx::dct8x16_avx2(input, output);
            });
        }
    }

    Arc::new(|input, output| {
        dct8x16_scalar(input, output);
    })
}

static DCT_METHOD_8X16: OnceLock<Arc<DctFn<128>>> = OnceLock::new();
static DCT_METHOD_16X8: OnceLock<Arc<DctFn<128>>> = OnceLock::new();

pub(crate) fn dct8x16(input: &[f32; 128], output: &mut [f32; 128]) {
    DCT_METHOD_8X16.get_or_init(select_dct_8x16)(input, output);
}

pub(crate) fn dct8x16_scalar(input: &[f32], output: &mut [f32; 128]) {
    let mut after_row_dct = [0.0f32; 128];
    for (src, dst) in input
        .as_chunks::<16>()
        .0
        .iter()
        .zip(after_row_dct.as_chunks_mut::<16>().0.iter_mut())
    {
        dct1d_16_oof(src, dst);
    }

    let mut col = [0.0f32; 8];

    let scale = 1.0 / 128.0;
    for u in 0..16 {
        for i in 0..8 {
            col[i] = after_row_dct[i * 16 + u];
        }
        dct1d_8(&mut col);
        for v in 0..8 {
            output[v * 16 + u] = col[v] * scale;
        }
    }
}

fn select_dct_16x8() -> Arc<DctFn<128>> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") {
            use crate::neon::dct16x8_neon;
            return Arc::new(|input, output| unsafe {
                dct16x8_neon(input, output);
            });
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Arc::new(|input, output| unsafe {
                crate::avx::dct16x8_avx2(input, output);
            });
        }
    }

    Arc::new(|input, output| {
        dct16x8_scalar(input, output);
    })
}

pub(crate) fn dct16x8(input: &[f32; 128], output: &mut [f32; 128]) {
    DCT_METHOD_16X8.get_or_init(select_dct_16x8)(input, output);
}

pub(crate) fn dct16x8_scalar(input: &[f32; 128], output: &mut [f32; 128]) {
    let mut after_col_dct = [0.0f32; 128];
    for u in 0..8 {
        let mut col = [0.0f32; 16];
        for i in 0..16 {
            col[i] = input[i * 8 + u];
        }
        dct1d_16(&mut col);
        for v in 0..16 {
            after_col_dct[v * 8 + u] = col[v];
        }
    }

    let scale = 1.0 / 128.0;
    for v in 0..16 {
        let row = &mut after_col_dct[v * 8..v * 8 + 8];
        dct1d_8(row);
        for u in 0..8 {
            output[u * 16 + v] = row[u] * scale;
        }
    }
}

const RESAMPLE_SCALE_16_TO_2: [f32; 2] = [1.0, 0.901_764_2];

pub(crate) fn dc_from_dct16x8(coeffs: &[f32; 128], dc: &mut [f32; 2]) {
    let s0 = coeffs[0] * RESAMPLE_SCALE_16_TO_2[0];
    let s1 = coeffs[1] * RESAMPLE_SCALE_16_TO_2[1];
    // IDCT1DImpl<2>: sum + diff, no scaling.
    dc[0] = s0 + s1;
    dc[1] = s0 - s1;
}

pub(crate) fn dc_from_dct8x16(coeffs: &[f32; 128], dc: &mut [f32; 2]) {
    let s0 = coeffs[0] * RESAMPLE_SCALE_16_TO_2[0];
    let s1 = coeffs[1] * RESAMPLE_SCALE_16_TO_2[1];
    dc[0] = s0 + s1;
    dc[1] = s0 - s1;
}

static DCT_METHOD_16X16: OnceLock<Arc<DctFn<256>>> = OnceLock::new();

fn select_dct_16x16() -> Arc<DctFn<256>> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        use std::arch::is_aarch64_feature_detected;
        if is_aarch64_feature_detected!("neon") {
            use crate::neon::dct16x16_neon;
            return Arc::new(|input, output| unsafe {
                dct16x16_neon(input, output);
            });
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Arc::new(|input, output| unsafe {
                crate::avx::dct16x16_avx2(input, output);
            });
        }
    }

    Arc::new(|input, output| {
        dct16x16_scalar(input, output);
    })
}

pub(crate) fn dct16x16(input: &[f32; 256], output: &mut [f32; 256]) {
    DCT_METHOD_16X16.get_or_init(select_dct_16x16)(input, output);
}

pub(crate) fn dct16x16_scalar(input: &[f32; 256], output: &mut [f32; 256]) {
    let mut after_col_dct = [0.0f32; 256];
    let mut col = [0.0f32; 16];
    for u in 0..16 {
        for i in 0..16 {
            col[i] = input[i * 16 + u];
        }
        dct1d_16(&mut col);
        for v in 0..16 {
            after_col_dct[v * 16 + u] = col[v];
        }
    }

    let scale = 1.0 / 256.0;
    for v in 0..16 {
        let row = &mut after_col_dct[v * 16..v * 16 + 16];
        dct1d_16(row.try_into().unwrap());
        for u in 0..16 {
            output[u * 16 + v] = row[u] * scale;
        }
    }
}

pub(crate) fn dc_from_dct16x16(coeffs: &[f32; 256], dc: &mut [f32; 4]) {
    // Scale the 2×2 LF patch by the outer product of the resample scales.
    let s00 = coeffs[0] * RESAMPLE_SCALE_16_TO_2[0] * RESAMPLE_SCALE_16_TO_2[0];
    let s01 = coeffs[1] * RESAMPLE_SCALE_16_TO_2[0] * RESAMPLE_SCALE_16_TO_2[1];
    let s10 = coeffs[16] * RESAMPLE_SCALE_16_TO_2[1] * RESAMPLE_SCALE_16_TO_2[0];
    let s11 = coeffs[17] * RESAMPLE_SCALE_16_TO_2[1] * RESAMPLE_SCALE_16_TO_2[1];
    // 2-D 2-point IDCT.
    //   TL = s00 + s01 + s10 + s11
    //   TR = s00 + s01 - s10 - s11
    //   BL = s00 - s01 + s10 - s11
    //   BR = s00 - s01 - s10 + s11
    let r00 = s00 + s01;
    let r01 = s00 - s01;
    let r10 = s10 + s11;
    let r11 = s10 - s11;
    dc[0] = r00 + r10; // top-left
    dc[1] = r00 - r10; // top-right
    dc[2] = r01 + r11; // bottom-left
    dc[3] = r01 - r11; // bottom-right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_linearity() {
        let mut a = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        for i in 0..64 {
            a[i] = (i as f32 * 0.13).sin();
            b[i] = (i as f32 * 0.27).cos();
        }
        let mut sum = [0.0f32; 64];
        for i in 0..64 {
            sum[i] = a[i] + b[i];
        }

        let mut da = [0.0f32; 64];
        let mut db = [0.0f32; 64];
        let mut dsum = [0.0f32; 64];
        dct8x8(&a, &mut da);
        dct8x8(&b, &mut db);
        dct8x8(&sum, &mut dsum);

        for i in 0..64 {
            let expected = da[i] + db[i];
            assert!(
                (dsum[i] - expected).abs() < 1e-4,
                "i={} dsum={} expected={}",
                i,
                dsum[i],
                expected
            );
        }
    }
}
