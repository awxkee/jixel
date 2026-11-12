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
//! 128-point transforms for the DCT128X128 / DCT128X64 / DCT64X128
//! family. Same conventions as the 64 family in `dct.rs`: unnormalized 1-D
//! butterflies, the 2-D result scaled by `1 / (rows * cols)`, and coefficients
//! stored in the normalized layout (short axis = rows, long axis = columns;
//! square blocks are stored transposed, `horizontal * 128 + vertical`).

use crate::dct::DctInput;
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
use crate::dct::{
    RESAMPLE_SCALE_64_TO_8, dct1d_64, fmla, inv_dct1d_8, inv_dct1d_16, inv_dct1d_64, inv_dct16x16,
};
use std::sync::OnceLock;

/// libjxl `DCTResampleScales<128, 16>`.
pub(crate) const RESAMPLE_SCALE_128_TO_16: [f32; 16] = [
    1.0,
    0.998_419_45,
    0.993_686_6,
    0.985_827_8,
    0.974_886_83,
    0.960_924_4,
    0.944_018_07,
    0.924_261_6,
    0.901_764_2,
    0.876_650_1,
    0.849_057_5,
    0.819_137_9,
    0.787_054_9,
    0.752_983_4,
    0.717_108_13,
    0.679_622_85,
];

/// `1 / (2 cos((2i + 1) pi / 256))`, the 128-point butterfly weights.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) static WC128: [f32; 64] = [
    0.50003767,
    0.500339031,
    0.500942707,
    0.501850545,
    0.503065169,
    0.504590452,
    0.506430984,
    0.508592427,
    0.511081576,
    0.5139063,
    0.517075658,
    0.520599842,
    0.524490535,
    0.528760731,
    0.533424914,
    0.538499415,
    0.544002235,
    0.549953401,
    0.556374967,
    0.563291669,
    0.570730567,
    0.578721881,
    0.58729893,
    0.596498787,
    0.606362462,
    0.61693573,
    0.628269434,
    0.640420318,
    0.65345192,
    0.667435229,
    0.682450116,
    0.698586643,
    0.715946436,
    0.73464483,
    0.754812956,
    0.776600659,
    0.800179899,
    0.825748801,
    0.853536725,
    0.883810997,
    0.916884422,
    0.953125894,
    0.99297297,
    1.03694904,
    1.08568501,
    1.13994873,
    1.20068324,
    1.26906121,
    1.34655762,
    1.43505514,
    1.5369941,
    1.65559649,
    1.79520524,
    1.96181786,
    2.16395783,
    2.41416001,
    2.73164511,
    3.14746213,
    3.71524262,
    4.53629112,
    5.82768822,
    8.15384865,
    13.5842905,
    40.7446899,
];

pub(crate) static INV_WC128: [f32; 64] = [
    1.99984944,
    1.99864471,
    1.99623621,
    1.99262524,
    1.98781395,
    1.98180532,
    1.97460282,
    1.96621096,
    1.95663476,
    1.94587994,
    1.93395293,
    1.92086101,
    1.90661204,
    1.89121461,
    1.87467802,
    1.85701215,
    1.83822775,
    1.81833601,
    1.79734898,
    1.77527928,
    1.75214016,
    1.72794569,
    1.70271039,
    1.67644942,
    1.64917862,
    1.62091434,
    1.59167385,
    1.56147444,
    1.53033447,
    1.49827278,
    1.46530855,
    1.43146169,
    1.39675248,
    1.361202,
    1.3248316,
    1.2876631,
    1.24971902,
    1.21102214,
    1.17159569,
    1.13146365,
    1.09064996,
    1.04917932,
    1.00707674,
    0.964367568,
    0.92107743,
    0.877232492,
    0.832859099,
    0.787984073,
    0.742634416,
    0.696837366,
    0.65062058,
    0.604011893,
    0.55703938,
    0.509731293,
    0.462116212,
    0.414222747,
    0.366079777,
    0.3177163,
    0.269161403,
    0.220444411,
    0.17159462,
    0.122641474,
    0.0736144483,
    0.0245430768,
];

// Split-radix rotations for the SIMD 128-point kernels.
#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
))]
#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
))]
pub(crate) static DCT128_SPLIT_COS: [f32; 32] = [
    0.999924719,
    0.999322355,
    0.998118103,
    0.996312618,
    0.993906975,
    0.990902662,
    0.987301409,
    0.983105481,
    0.97831738,
    0.972939968,
    0.966976464,
    0.960430503,
    0.953306019,
    0.945607305,
    0.937339008,
    0.928506076,
    0.919113874,
    0.909168005,
    0.898674488,
    0.887639642,
    0.876070082,
    0.863972843,
    0.851355195,
    0.838224709,
    0.824589312,
    0.81045717,
    0.795836926,
    0.780737221,
    0.765167236,
    0.749136388,
    0.732654274,
    0.715730846,
];

#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
))]
#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
))]
pub(crate) const DCT128_SPLIT_SIN: [f32; 32] = [
    0.0122715384,
    0.0368072242,
    0.061320737,
    0.0857973099,
    0.110222206,
    0.134580702,
    0.15885815,
    0.183039889,
    0.207111374,
    0.231058106,
    0.254865646,
    0.27851969,
    0.302005947,
    0.32531029,
    0.348418683,
    0.371317208,
    0.393992037,
    0.416429549,
    0.438616246,
    0.460538715,
    0.482183784,
    0.50353837,
    0.524589658,
    0.545324981,
    0.565731823,
    0.585797846,
    0.605511069,
    0.624859512,
    0.643831551,
    0.662415802,
    0.680601001,
    0.698376238,
];

/// Frequency location in the split-radix 64-point SIMD result.
#[cfg(any(
    all(target_arch = "aarch64", feature = "neon"),
    all(target_arch = "x86_64", feature = "avx"),
    all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
))]
#[inline]
pub(crate) const fn dct64_natural_index(frequency: usize) -> usize {
    if frequency & 1 == 0 {
        frequency / 2
    } else {
        let odd = frequency / 2;
        if odd == 0 {
            32
        } else if odd & 1 != 0 {
            33 + odd / 2
        } else {
            64 - odd / 2
        }
    }
}

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn dct1d_128(buf: &mut [f32; 128]) {
    let wc = &WC128;
    let mut tmp = [0.0f32; 128];
    for i in 0..64 {
        tmp[i] = buf[i] + buf[127 - i];
        tmp[64 + i] = (buf[i] - buf[127 - i]) * wc[i];
    }
    dct1d_64(<&mut [f32; 64]>::try_from(&mut tmp[..64]).unwrap());
    dct1d_64(<&mut [f32; 64]>::try_from(&mut tmp[64..]).unwrap());
    tmp[64] = fmla(tmp[64], std::f32::consts::SQRT_2, tmp[65]);
    for i in 65..127 {
        tmp[i] += tmp[i + 1];
    }
    for i in 0..64 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[64 + i];
    }
}

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn inv_dct1d_128(buf: &mut [f32; 128]) {
    let inv_wc = &INV_WC128;
    let mut tmp = [0.0f32; 128];
    for i in 0..64 {
        tmp[i] = buf[2 * i];
        tmp[64 + i] = buf[2 * i + 1];
    }
    for i in (65..=126).rev() {
        tmp[i] -= tmp[i + 1];
    }
    tmp[64] = (tmp[64] - tmp[65]) * std::f32::consts::FRAC_1_SQRT_2;
    inv_dct1d_64(<&mut [f32; 64]>::try_from(&mut tmp[64..]).unwrap());
    for i in 0..64 {
        tmp[64 + i] *= inv_wc[i];
    }
    inv_dct1d_64(<&mut [f32; 64]>::try_from(&mut tmp[..64]).unwrap());
    for i in 0..64 {
        buf[i] = (tmp[i] + tmp[64 + i]) * 0.5;
        buf[127 - i] = (tmp[i] - tmp[64 + i]) * 0.5;
    }
}

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn dct128x128_scalar_input(input: DctInput<'_, 128, 128>, output: &mut [f32; 16384]) {
    let mut after_col_dct = vec![0.0f32; 16384];
    let mut col = [0.0f32; 128];
    for u in 0..128 {
        for (i, value) in col.iter_mut().enumerate() {
            *value = input.row(i)[u];
        }
        dct1d_128(&mut col);
        for v in 0..128 {
            after_col_dct[v * 128 + u] = col[v];
        }
    }
    let scale = 1.0 / 16384.0;
    for v in 0..128 {
        let row = <&mut [f32; 128]>::try_from(&mut after_col_dct[v * 128..v * 128 + 128]).unwrap();
        dct1d_128(row);
        for u in 0..128 {
            output[u * 128 + v] = row[u] * scale;
        }
    }
}

/// 128 rows x 64 columns, stored as 64 rows x 128 columns
/// (`horizontal_frequency * 128 + vertical_frequency`).
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn dct128x64_scalar_input(input: DctInput<'_, 64, 128>, output: &mut [f32; 8192]) {
    let mut after_col_dct = vec![0.0f32; 8192];
    let mut col = [0.0f32; 128];
    for x in 0..64 {
        for (y, value) in col.iter_mut().enumerate() {
            *value = input.row(y)[x];
        }
        dct1d_128(&mut col);
        for v in 0..128 {
            after_col_dct[v * 64 + x] = col[v];
        }
    }
    let scale = 1.0 / 8192.0;
    for v in 0..128 {
        let row = <&mut [f32; 64]>::try_from(&mut after_col_dct[v * 64..v * 64 + 64]).unwrap();
        dct1d_64(row);
        for u in 0..64 {
            output[u * 128 + v] = row[u] * scale;
        }
    }
}

/// 64 rows x 128 columns, stored directly in the normalized layout.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn dct64x128_scalar_input(input: DctInput<'_, 128, 64>, output: &mut [f32; 8192]) {
    let mut after_row_dct = vec![0.0f32; 8192];
    for y in 0..64 {
        let row = <&mut [f32; 128]>::try_from(&mut after_row_dct[y * 128..y * 128 + 128]).unwrap();
        row.copy_from_slice(input.row(y));
        dct1d_128(row);
    }
    let scale = 1.0 / 8192.0;
    let mut col = [0.0f32; 64];
    for u in 0..128 {
        for y in 0..64 {
            col[y] = after_row_dct[y * 128 + u];
        }
        dct1d_64(&mut col);
        for v in 0..64 {
            output[v * 128 + u] = col[v] * scale;
        }
    }
}

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn inv_dct128x128(coeff: DctInput<'_, 128, 128>, out: &mut [f32; 16384]) {
    for a in 0..128 {
        for b in 0..128 {
            out[b * 128 + a] = coeff.row(a)[b] * 16384.0;
        }
    }
    let mut col = [0.0f32; 128];
    for u in 0..128 {
        for v in 0..128 {
            col[v] = out[v * 128 + u];
        }
        inv_dct1d_128(&mut col);
        for v in 0..128 {
            out[v * 128 + u] = col[v];
        }
    }
    for v in 0..128 {
        inv_dct1d_128(<&mut [f32; 128]>::try_from(&mut out[v * 128..v * 128 + 128]).unwrap());
    }
}

/// Inverse of [`dct128x64`]: 128 rows x 64 columns of pixels.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn inv_dct128x64(coeff: DctInput<'_, 128, 64>, out: &mut [f32; 8192]) {
    let mut after_row = vec![0.0f32; 8192];
    let mut row = [0.0f32; 64];
    for v in 0..128 {
        for u in 0..64 {
            row[u] = coeff.row(u)[v] * 8192.0;
        }
        inv_dct1d_64(&mut row);
        after_row[v * 64..v * 64 + 64].copy_from_slice(&row);
    }
    let mut col = [0.0f32; 128];
    for x in 0..64 {
        for v in 0..128 {
            col[v] = after_row[v * 64 + x];
        }
        inv_dct1d_128(&mut col);
        for y in 0..128 {
            out[y * 64 + x] = col[y];
        }
    }
}

/// Inverse of [`dct64x128`]: 64 rows x 128 columns of pixels.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn inv_dct64x128(coeff: DctInput<'_, 128, 64>, out: &mut [f32; 8192]) {
    for y in 0..64 {
        for x in 0..128 {
            out[y * 128 + x] = coeff.row(y)[x] * 8192.0;
        }
    }
    let mut col = [0.0f32; 64];
    for u in 0..128 {
        for v in 0..64 {
            col[v] = out[v * 128 + u];
        }
        inv_dct1d_64(&mut col);
        for v in 0..64 {
            out[v * 128 + u] = col[v];
        }
    }
    for y in 0..64 {
        inv_dct1d_128(<&mut [f32; 128]>::try_from(&mut out[y * 128..y * 128 + 128]).unwrap());
    }
}

/// The 16x16 grid of 8x8 DC values covered by a DCT128X128.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn dc_from_dct128x128(coeffs: &[f32; 16384], dc: &mut [f32; 256]) {
    let mut low = [0.0f32; 256];
    for y in 0..16 {
        for x in 0..16 {
            low[y * 16 + x] =
                coeffs[y * 128 + x] * (RESAMPLE_SCALE_128_TO_16[x] * RESAMPLE_SCALE_128_TO_16[y]);
        }
    }
    let dc: &mut [f32; 256] = (&mut dc[..256]).try_into().unwrap();
    inv_dct16x16(DctInput::from_flat(&low), dc);
}

#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
fn dc_from_dct128x64_normalized(coeffs: &[f32; 8192]) -> [[f32; 16]; 8] {
    let mut rows = [[0.0f32; 16]; 8];
    for (a, row) in rows.iter_mut().enumerate() {
        for b in 0..16 {
            row[b] = coeffs[a * 128 + b]
                * (RESAMPLE_SCALE_64_TO_8[a] * RESAMPLE_SCALE_128_TO_16[b])
                * 16.0;
        }
        inv_dct1d_16(row);
    }
    let mut out = [[0.0f32; 16]; 8];
    for x in 0..16 {
        let mut column = [0.0f32; 8];
        for y in 0..8 {
            column[y] = rows[y][x] * 8.0;
        }
        inv_dct1d_8(&mut column);
        for y in 0..8 {
            out[y][x] = column[y];
        }
    }
    out
}

/// DC grid for a 128-row x 64-column transform (16 rows x 8 columns).
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn dc_from_dct128x64(coeffs: &[f32; 8192], dc: &mut [f32; 128]) {
    let normalized = dc_from_dct128x64_normalized(coeffs);
    for y in 0..16 {
        for x in 0..8 {
            dc[y * 8 + x] = normalized[x][y];
        }
    }
}

/// DC grid for a 64-row x 128-column transform (8 rows x 16 columns).
#[cfg(any(
    test,
    not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    ))
))]
pub(crate) fn dc_from_dct64x128(coeffs: &[f32; 8192], dc: &mut [f32; 128]) {
    let normalized = dc_from_dct128x64_normalized(coeffs);
    for y in 0..8 {
        dc[y * 16..y * 16 + 16].copy_from_slice(&normalized[y]);
    }
}

static DCT_METHOD_128X128: OnceLock<crate::dct::DctFn<128, 128, 16384>> = OnceLock::new();

fn select_dct_128x128() -> crate::dct::DctFn<128, 128, 16384> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |input, output| unsafe { crate::neon::dct128x128_neon(input, output) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::dct128x128_wasm
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |input, output| unsafe { crate::avx::dct128x128_avx2(input, output) };
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    dct128x128_scalar_input
}

pub(crate) fn selected_dct128x128() -> &'static crate::dct::DctFn<128, 128, 16384> {
    DCT_METHOD_128X128.get_or_init(select_dct_128x128)
}

pub(crate) fn dct128x128(input: DctInput<'_, 128, 128>, output: &mut [f32]) {
    selected_dct128x128()(input, output.first_chunk_mut::<16384>().unwrap());
}

static DCT_METHOD_128X64: OnceLock<crate::dct::DctFn<64, 128, 8192>> = OnceLock::new();

fn select_dct_128x64() -> crate::dct::DctFn<64, 128, 8192> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |input, output| unsafe { crate::neon::dct128x64_neon(input, output) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::dct128x64_wasm
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |input, output| unsafe { crate::avx::dct128x64_avx2(input, output) };
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    dct128x64_scalar_input
}

pub(crate) fn selected_dct128x64() -> &'static crate::dct::DctFn<64, 128, 8192> {
    DCT_METHOD_128X64.get_or_init(select_dct_128x64)
}

pub(crate) fn dct128x64(input: DctInput<'_, 64, 128>, output: &mut [f32]) {
    selected_dct128x64()(input, output.first_chunk_mut::<8192>().unwrap());
}

static DCT_METHOD_64X128: OnceLock<crate::dct::DctFn<128, 64, 8192>> = OnceLock::new();

fn select_dct_64x128() -> crate::dct::DctFn<128, 64, 8192> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |input, output| unsafe { crate::neon::dct64x128_neon(input, output) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::dct64x128_wasm
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |input, output| unsafe { crate::avx::dct64x128_avx2(input, output) };
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    dct64x128_scalar_input
}

pub(crate) fn selected_dct64x128() -> &'static crate::dct::DctFn<128, 64, 8192> {
    DCT_METHOD_64X128.get_or_init(select_dct_64x128)
}

pub(crate) fn dct64x128(input: DctInput<'_, 128, 64>, output: &mut [f32]) {
    selected_dct64x128()(input, output.first_chunk_mut::<8192>().unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, label: &str) {
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let tolerance = atol + 32.0 * f32::EPSILON * e.abs();
            assert!(
                (a - e).abs() <= tolerance,
                "{label}[{i}]: {a} vs {e}, tolerance {tolerance}"
            );
        }
    }

    // Leave NaNs between rows and omit trailing padding on the last row.
    // The prefix also exercises unaligned SIMD loads.
    fn strided(input: &[f32], w: usize, h: usize) -> Vec<f32> {
        let stride = w + 7;
        let mut data = vec![f32::NAN; 3 + (h - 1) * stride + w];
        for y in 0..h {
            data[3 + y * stride..3 + y * stride + w].copy_from_slice(&input[y * w..(y + 1) * w]);
        }
        data
    }

    fn check_dispatch<
        const W: usize,
        const H: usize,
        const CH: usize,
        const N: usize,
        const DC: usize,
    >(
        forward: crate::dct::DctFn<W, H, N>,
        scalar_forward: crate::dct::DctFn<W, H, N>,
        inverse: crate::dct::DctFn<128, CH, N>,
        scalar_inverse: crate::dct::DctFn<128, CH, N>,
        dc: fn(&[f32; N], &mut [f32; DC]),
        scalar_dc: fn(&[f32; N], &mut [f32; DC]),
    ) {
        let simd_inverse = cfg!(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(
                target_arch = "wasm32",
                target_feature = "simd128",
                feature = "wasm"
            )
        ));
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        let simd_inverse = simd_inverse
            || (std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"));
        let simd_forward = simd_inverse;
        assert_eq!(std::ptr::fn_addr_eq(forward, scalar_forward), !simd_forward);
        assert_eq!(std::ptr::fn_addr_eq(inverse, scalar_inverse), !simd_inverse);
        assert_eq!(std::ptr::fn_addr_eq(dc, scalar_dc), !simd_inverse);

        let mut expected = vec![0.0f32; N];
        let mut actual = vec![f32::NAN; N + 4];
        let mut restored = vec![f32::NAN; N + 4];
        for seed in 0..6 {
            let pixels: Vec<f32> = (0..N)
                .map(|i| match seed {
                    0 => 0.75,
                    1 => {
                        if i == N - 1 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    2 => {
                        if (i % W + i / W) & 1 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ => {
                        let mut x = (i as u64 + 1 + seed * 719).wrapping_mul(0x9e3779b97f4a7c15);
                        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                        ((x ^ (x >> 31)) >> 41) as f32 / (1u32 << 22) as f32 - 1.0
                    }
                })
                .collect();
            let padded = strided(&pixels, W, H);
            scalar_forward(
                DctInput::new(&pixels, W),
                expected.first_chunk_mut::<N>().unwrap(),
            );
            forward(
                DctInput::new(&padded[3..], W + 7),
                actual.first_chunk_mut::<N>().unwrap(),
            );
            assert_close(&actual[..N], &expected, 2e-6, "forward");
            assert!(actual[N..].iter().all(|v| v.is_nan()));

            let padded = strided(&actual[..N], 128, CH);
            inverse(
                DctInput::new(&padded[3..], 135),
                restored.first_chunk_mut::<N>().unwrap(),
            );
            assert_close(&restored[..N], &pixels, 2e-4, "round trip");
            assert!(restored[N..].iter().all(|v| v.is_nan()));

            // Arbitrary coefficients catch inverse errors masked by a forward round trip.
            let coeffs: Vec<f32> = pixels.iter().map(|v| v * 0.01).collect();
            let padded = strided(&coeffs, 128, CH);
            scalar_inverse(
                DctInput::new(&coeffs, 128),
                expected.first_chunk_mut::<N>().unwrap(),
            );
            inverse(
                DctInput::new(&padded[3..], 135),
                actual.first_chunk_mut::<N>().unwrap(),
            );
            assert_close(&actual[..N], &expected, 2e-3, "inverse");

            let mut expected_dc = [0.0f32; DC];
            let mut actual_dc = [f32::NAN; DC];
            scalar_dc(coeffs.first_chunk::<N>().unwrap(), &mut expected_dc);
            dc(coeffs.first_chunk::<N>().unwrap(), &mut actual_dc);
            assert_close(&actual_dc, &expected_dc, 3e-5, "DC extraction");
        }

        // Independent cosine basis checks coefficient ordering and normalization,
        // including the highest odd frequencies of the split-radix decomposition.
        for (u, v) in [
            (0, 0),
            (1, 0),
            (0, 1),
            (W / 2, H / 2),
            (W - 3, H - 3),
            (W - 1, H - 1),
        ] {
            let pixels: Vec<f32> = (0..N)
                .map(|i| {
                    let x = (std::f64::consts::PI * (i % W) as f64 + std::f64::consts::FRAC_PI_2)
                        * u as f64
                        / W as f64;
                    let y = (std::f64::consts::PI * (i / W) as f64 + std::f64::consts::FRAC_PI_2)
                        * v as f64
                        / H as f64;
                    (x.cos() * y.cos()) as f32
                })
                .collect();
            forward(
                DctInput::new(&pixels, W),
                actual.first_chunk_mut::<N>().unwrap(),
            );
            expected.fill(0.0);
            let index = if H >= W { u * H + v } else { v * W + u };
            expected[index] = (if u == 0 {
                1.0
            } else {
                std::f32::consts::FRAC_1_SQRT_2
            }) * (if v == 0 {
                1.0
            } else {
                std::f32::consts::FRAC_1_SQRT_2
            });
            assert_close(&actual[..N], &expected, 2e-6, "cosine basis");
        }
    }

    #[test]
    fn context_dispatch_matches_scalar_for_strided_inputs() {
        let ctx = crate::encoding_context::EncodingContext::default();
        let dc = crate::dct::selected_dc_from_dct_methods();
        macro_rules! check {
            ($w:literal, $h:literal, $ch:literal, $n:literal, $dc:literal, $forward:ident, $select:ident, $scalar:ident, $inverse:ident, $scalar_inverse:ident, $dc_fn:ident) => {
                assert!(std::ptr::eq(ctx.$forward, $select()));
                assert_eq!(ctx.$dc_fn as usize, dc.$forward as usize);
                check_dispatch::<$w, $h, $ch, $n, $dc>(
                    *ctx.$forward,
                    $scalar,
                    ctx.idct.$inverse,
                    $scalar_inverse,
                    ctx.$dc_fn,
                    $dc_fn,
                );
            };
        }
        check!(
            128,
            128,
            128,
            16384,
            256,
            dct128x128,
            selected_dct128x128,
            dct128x128_scalar_input,
            idct128x128,
            inv_dct128x128,
            dc_from_dct128x128
        );
        check!(
            64,
            128,
            64,
            8192,
            128,
            dct128x64,
            selected_dct128x64,
            dct128x64_scalar_input,
            idct128x64,
            inv_dct128x64,
            dc_from_dct128x64
        );
        check!(
            128,
            64,
            64,
            8192,
            128,
            dct64x128,
            selected_dct64x128,
            dct64x128_scalar_input,
            idct64x128,
            inv_dct64x128,
            dc_from_dct64x128
        );
    }

    fn pattern(w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = 0.01 * x as f32
                    + 0.003 * y as f32
                    + ((x / 8 + 2 * (y / 8)) % 5) as f32 * 0.05
                    + ((x * 7 + y * 13) % 11) as f32 * 0.002;
            }
        }
        out
    }

    #[test]
    fn resample_scales_match_the_box_filter_response() {
        for (k, &s) in RESAMPLE_SCALE_128_TO_16.iter().enumerate() {
            let k = k as f64;
            let expected = if k == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * k / 32.0).sin()
                    / (8.0 * (std::f64::consts::PI * k / 256.0).sin())
            };
            assert!((s as f64 - expected).abs() < 1e-6, "{k}: {s} vs {expected}");
        }
    }

    #[test]
    fn transforms_round_trip() {
        let input = pattern(128, 128);
        let mut coeffs = vec![0.0f32; 16384];
        let mut output = vec![0.0f32; 16384];
        dct128x128(DctInput::new(&input, 128), &mut coeffs);
        inv_dct128x128(
            DctInput::new(&coeffs, 128),
            output.first_chunk_mut::<16384>().unwrap(),
        );
        for (e, a) in input.iter().zip(&output) {
            assert!((e - a).abs() < 2e-4, "{e} vs {a}");
        }

        let tall = pattern(64, 128);
        dct128x64(DctInput::new(&tall, 64), &mut coeffs);
        inv_dct128x64(
            DctInput::new(&coeffs, 128),
            output.first_chunk_mut::<8192>().unwrap(),
        );
        for (e, a) in tall.iter().zip(&output) {
            assert!((e - a).abs() < 2e-4, "{e} vs {a}");
        }

        let wide = pattern(128, 64);
        dct64x128(DctInput::new(&wide, 128), &mut coeffs);
        inv_dct64x128(
            DctInput::new(&coeffs, 128),
            output.first_chunk_mut::<8192>().unwrap(),
        );
        for (e, a) in wide.iter().zip(&output) {
            assert!((e - a).abs() < 2e-4, "{e} vs {a}");
        }
    }

    #[test]
    fn transposed_rectangles_agree() {
        let tall = pattern(64, 128);
        let mut wide = vec![0.0f32; 8192];
        for y in 0..128 {
            for x in 0..64 {
                wide[x * 128 + y] = tall[y * 64 + x];
            }
        }
        let mut a = vec![0.0f32; 8192];
        let mut b = vec![0.0f32; 8192];
        dct128x64(DctInput::new(&tall, 64), &mut a);
        dct64x128(DctInput::new(&wide, 128), &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn constant_is_pure_dc_and_extracts_block_dcs() {
        let input = vec![2.0f32; 16384];
        let mut coeffs = vec![0.0f32; 16384];
        dct128x128(DctInput::new(&input, 128), &mut coeffs);
        assert!((coeffs[0] - 2.0).abs() < 1e-5);
        assert!(coeffs[1..].iter().map(|v| v * v).sum::<f32>() < 1e-7);
        let mut dc = [0.0f32; 256];
        dc_from_dct128x128(coeffs.first_chunk::<16384>().unwrap(), &mut dc);
        for value in dc {
            assert!((value - 2.0).abs() < 1e-4, "{value}");
        }
        dct128x64(DctInput::new(&input[..8192], 64), &mut coeffs);
        let mut dc = [0.0f32; 128];
        dc_from_dct128x64(coeffs.first_chunk::<8192>().unwrap(), &mut dc);
        for value in dc {
            assert!((value - 2.0).abs() < 1e-4, "{value}");
        }
    }

    /// Smooth content: the LLF-derived DC must approximate each 8x8 mean.
    #[test]
    fn dc_grid_tracks_block_means_on_smooth_content() {
        let (w, h) = (128usize, 128usize);
        let input: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                (x * 0.021).sin() + (y * 0.013).cos() * 0.5
            })
            .collect();
        let mean = |bx: usize, by: usize, stride: usize, data: &[f32]| -> f32 {
            let mut s = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    s += data[(by * 8 + y) * stride + bx * 8 + x];
                }
            }
            s / 64.0
        };
        let mut coeffs = vec![0.0f32; 16384];
        dct128x128(DctInput::new(&input, 128), &mut coeffs);
        let mut dc = [0.0f32; 256];
        dc_from_dct128x128(coeffs.first_chunk::<16384>().unwrap(), &mut dc);
        for by in 0..16 {
            for bx in 0..16 {
                let m = mean(bx, by, 128, &input);
                assert!((dc[by * 16 + bx] - m).abs() < 0.02, "{bx},{by}");
            }
        }
        // Tall: rows 0..128, cols 0..64.
        let tall: Vec<f32> = (0..128 * 64)
            .map(|i| input[(i / 64) * 128 + i % 64])
            .collect();
        dct128x64(DctInput::new(&tall, 64), &mut coeffs);
        let mut dc = [0.0f32; 128];
        dc_from_dct128x64(coeffs.first_chunk::<8192>().unwrap(), &mut dc);
        for by in 0..16 {
            for bx in 0..8 {
                let m = mean(bx, by, 64, &tall);
                assert!((dc[by * 8 + bx] - m).abs() < 0.02, "{bx},{by}");
            }
        }
        // Wide: rows 0..64, cols 0..128.
        dct64x128(DctInput::new(&input[..8192], 128), &mut coeffs);
        dc_from_dct64x128(coeffs.first_chunk::<8192>().unwrap(), &mut dc);
        for by in 0..8 {
            for bx in 0..16 {
                let m = mean(bx, by, 128, &input);
                assert!((dc[by * 16 + bx] - m).abs() < 0.02, "{bx},{by}");
            }
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use crate::dc_group_data::NUM_STRATEGIES;
    use crate::inflated_cost::{forward_for, reconstruct_error, strategy_pixel_dims};

    /// The closed loop's reconstruction capture reads inverse output as
    /// row-major pixels `strategy_pixel_dims` wide; every transform must
    /// round-trip so.
    #[test]
    fn reconstruct_error_returns_row_major_footprints() {
        let idct = crate::dct::IdctMethods::scalar();
        for s in 0..NUM_STRATEGIES as u8 {
            let (w, h) = strategy_pixel_dims(s);
            let pix: Vec<f32> = (0..w * h)
                .map(|i| ((i % w) as f32 * 0.37 + (i / w) as f32 * 0.11).sin())
                .collect();
            let mut coeffs = vec![0.0f32; w * h];
            forward_for(s, &pix, &mut coeffs);
            let mut back = vec![0.0f32; w * h];
            reconstruct_error(&idct, s, &coeffs, &mut back);
            let err = pix
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(err < 2e-3, "strategy {s}: max err {err}");
        }
    }
}
