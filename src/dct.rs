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

pub(crate) const WC4: [f32; 2] = [0.541196100146197, 1.3065629648763764];

pub(crate) const WC8: [f32; 4] = [
    0.5097955791041592,
    0.6013448869350453,
    0.8999762231364156,
    2.5629154477415055,
];

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
fn dct1d_4(buf: &mut [f32]) {
    let mut tmp = [0.0f32; 4];
    // AddReverse: even half
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
    dct1d_4(&mut tmp[0..4]);
    for i in 0..4 {
        tmp[4 + i] = buf[i] - buf[7 - i];
    }
    for i in 0..4 {
        tmp[4 + i] *= WC8[i];
    }
    dct1d_4(&mut tmp[4..8]);
    tmp[4] = fmla(tmp[4], std::f32::consts::SQRT_2, tmp[5]);
    tmp[5] = tmp[5] + tmp[6];
    tmp[6] = tmp[6] + tmp[7];
    for i in 0..4 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[4 + i];
    }
}

pub(crate) type DctFn = dyn Fn(&[f32; 64], &mut [f32; 64]) + Send + Sync;

static DCT_METHOD: OnceLock<Arc<DctFn>> = OnceLock::new();

fn select_dct() -> Arc<DctFn> {
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

fn dct8x8_scalar(input: &[f32; 64], output: &mut [f32; 64]) {
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
