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
#![allow(clippy::excessive_precision)]

use crate::dct::fmla;
use crate::util::{HeapMatrix, f16_bits_to_f32, f32_to_f16_bits, heap_array_from_fn};

/// A signaled distance-band override for one quant table.
#[derive(Clone, Copy)]
pub(crate) struct BandOverride {
    pub(crate) num_bands: usize,
    /// Only the first `num_bands` entries of each channel row are meaningful.
    pub(crate) bands: [[f32; 16]; 3],
}

const QM_SS2_MIN_DISTANCE: f32 = 2.25;
const QM_DCT8_MIN_DISTANCE: f32 = 3.5;
const QM_FLAT_B8_MIN_DISTANCE: f32 = 0.3;
const QM_FLAT_B8_MID_MIN_DISTANCE: f32 = 1.25;
const QM_SS2_SCALE16: f32 = 0.78;
const QM_SS2_SCALE32: f32 = 0.89;
const QM_SS2_SCALE16X32: f32 = 1.20;
const QM_DCT8_Y_HF_SCALE: f32 = 1.25;

static DCT8_BANDS: [[f32; 6]; 3] = [
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
];

static FLAT_B8_BANDS_HQ: [f32; 6] = [512.0, -0.5, -0.25, 0.0, -0.25, -0.5];
static FLAT_B8_BANDS_MID: [f32; 6] = [512.0, -0.13, -0.235, -0.276, -0.609, -0.339];
static SAT_B8_BANDS: [f32; 6] = [512.0, 0.024, -0.517, -0.011, -0.357, -0.674];

fn default_dct8_override(use_coarse: bool, flat_b8: &[f32; 6]) -> BandOverride {
    let mut out = if use_coarse {
        coarse_dct8_override()
    } else {
        let mut base = BandOverride {
            num_bands: 6,
            bands: [[0.0; 16]; 3],
        };
        for c in 0..3 {
            base.bands[c][..6].copy_from_slice(&DCT8_BANDS[c]);
        }
        base
    };
    out.bands[2][..6].copy_from_slice(flat_b8);
    for c in 0..3 {
        for i in 0..6 {
            let v = out.bands[c][i];
            out.bands[c][i] = if i == 0 {
                f16_bits_to_f32(f32_to_f16_bits(v / 64.0)) * 64.0
            } else {
                f16_bits_to_f32(f32_to_f16_bits(v))
            };
        }
    }
    out
}

/// Retain 25% more luma precision in the outer radial band, spread
/// geometrically across the five band transitions. F16-round-trip every
/// parameter so the encoder matrix exactly matches the signaled decoder table.
fn coarse_dct8_override() -> BandOverride {
    let hf_step = QM_DCT8_Y_HF_SCALE.powf(1.0 / 5.0);
    let mut out = BandOverride {
        num_bands: 6,
        bands: [[0.0; 16]; 3],
    };
    for c in 0..3 {
        out.bands[c][..6].copy_from_slice(&DCT8_BANDS[c]);
    }
    for i in 1..6 {
        let ratio = band_mult(DCT8_BANDS[1][i]) * hf_step;
        out.bands[1][i] = if ratio > 1.0 {
            ratio - 1.0
        } else {
            1.0 - 1.0 / ratio
        };
    }
    for c in 0..3 {
        for i in 0..6 {
            let v = out.bands[c][i];
            out.bands[c][i] = if i == 0 {
                f16_bits_to_f32(f32_to_f16_bits(v / 64.0)) * 64.0
            } else {
                f16_bits_to_f32(f32_to_f16_bits(v))
            };
        }
    }
    out
}

/// Default bands with the base (band 0) scaled, F16-round-tripped exactly like
/// a parsed override so encoder and decoder agree bit-exactly.
fn scaled_override<const N: usize>(defaults: &[[f32; N]; 3], scale: f32) -> BandOverride {
    let mut out = BandOverride {
        num_bands: N,
        bands: [[0.0f32; 16]; 3],
    };
    for c in 0..3 {
        for i in 0..N {
            let v = if i == 0 {
                defaults[c][0] * scale
            } else {
                defaults[c][i]
            };
            out.bands[c][i] = if i == 0 {
                f16_bits_to_f32(f32_to_f16_bits(v / 64.0)) * 64.0
            } else {
                f16_bits_to_f32(f32_to_f16_bits(v))
            };
        }
    }
    out
}

pub(crate) static INV_DC_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];
pub(crate) static DC_QUANT: [f32; 3] = [1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0];

/// Per-channel dequant matrices for 8x8 DCT.
/// Indexed [channel][y * 8 + x]. Channel 0 = X, 1 = Y, 2 = B.
pub(crate) static DEQUANT_MATRIX_8X8: [[f32; 64]; 3] = [
    // Channel 0 (X): kQuantWeights[0..64]
    [
        3.1746033e-04,
        3.1746057e-04,
        3.1854658e-04,
        3.7755401e-04,
        4.4749113e-04,
        5.3038419e-04,
        6.2863121e-04,
        7.4507861e-04,
        3.1746057e-04,
        3.1746062e-04,
        3.3158599e-04,
        3.8811122e-04,
        4.5695182e-04,
        5.3938502e-04,
        6.3753547e-04,
        7.5413194e-04,
        3.1854658e-04,
        3.3158599e-04,
        3.6670428e-04,
        4.1847790e-04,
        4.8487642e-04,
        5.6626293e-04,
        6.6427846e-04,
        7.8140449e-04,
        3.7755401e-04,
        3.8811122e-04,
        4.1847790e-04,
        4.6632939e-04,
        5.3038419e-04,
        6.1082945e-04,
        7.0903177e-04,
        8.2727504e-04,
        4.4749113e-04,
        4.5695182e-04,
        4.8487642e-04,
        5.3038419e-04,
        5.9302151e-04,
        6.7320757e-04,
        7.7229418e-04,
        9.4286882e-04,
        5.3038419e-04,
        5.3938502e-04,
        5.6626293e-04,
        6.1082945e-04,
        6.7320757e-04,
        7.5413194e-04,
        8.5507357e-04,
        1.2723245e-03,
        6.2863121e-04,
        6.3753547e-04,
        6.6427846e-04,
        7.0903177e-04,
        7.7229418e-04,
        8.5507357e-04,
        1.1923184e-03,
        1.7919940e-03,
        7.4507861e-04,
        7.5413194e-04,
        7.8140449e-04,
        8.2727504e-04,
        9.4286882e-04,
        1.2723245e-03,
        1.7919940e-03,
        2.6133191e-03,
    ],
    // Channel 1 (Y): kQuantWeights[64..128]
    [
        1.7857145e-03,
        1.7857157e-03,
        1.7904768e-03,
        2.0441783e-03,
        2.3338278e-03,
        2.6645192e-03,
        3.0420676e-03,
        3.4731133e-03,
        1.7857157e-03,
        1.7857160e-03,
        1.8473724e-03,
        2.0886122e-03,
        2.3722122e-03,
        2.6997121e-03,
        3.0756146e-03,
        3.5059757e-03,
        1.7904768e-03,
        1.8473724e-03,
        1.9982266e-03,
        2.2149722e-03,
        2.4845072e-03,
        2.8040458e-03,
        3.1757555e-03,
        3.6044519e-03,
        2.0441783e-03,
        2.0886122e-03,
        2.2149722e-03,
        2.4100873e-03,
        2.6645192e-03,
        2.9746795e-03,
        3.3413812e-03,
        3.7683977e-03,
        2.3338278e-03,
        2.3722122e-03,
        2.4845072e-03,
        2.6645192e-03,
        2.9068382e-03,
        3.2089923e-03,
        3.5716419e-03,
        3.9980840e-03,
        2.6645192e-03,
        2.6997121e-03,
        2.8040458e-03,
        2.9746795e-03,
        3.2089923e-03,
        3.5059757e-03,
        3.8667743e-03,
        4.2947000e-03,
        3.0420676e-03,
        3.0756146e-03,
        3.1757555e-03,
        3.3413812e-03,
        3.5716419e-03,
        3.8667743e-03,
        4.2286036e-03,
        4.6607289e-03,
        3.4731133e-03,
        3.5059757e-03,
        3.6044519e-03,
        3.7683977e-03,
        3.9980840e-03,
        4.2947000e-03,
        4.6607289e-03,
        5.1001739e-03,
    ],
    // Channel 2 (B): kQuantWeights[128..192]
    [
        1.9531252e-03,
        3.4018266e-03,
        5.9007513e-03,
        8.3743408e-03,
        1.1718751e-02,
        1.1718759e-02,
        1.1968765e-02,
        1.6986061e-02,
        3.4018266e-03,
        4.2808522e-03,
        6.4091417e-03,
        8.8638803e-03,
        1.1718752e-02,
        1.1718759e-02,
        1.2320629e-02,
        1.7413978e-02,
        5.9007513e-03,
        6.4091417e-03,
        7.8861341e-03,
        1.0351914e-02,
        1.1718754e-02,
        1.1718762e-02,
        1.3408982e-02,
        1.8736197e-02,
        8.3743408e-03,
        8.8638803e-03,
        1.0351914e-02,
        1.1718752e-02,
        1.1718759e-02,
        1.1718766e-02,
        1.5336527e-02,
        2.1072537e-02,
        1.1718751e-02,
        1.1718752e-02,
        1.1718754e-02,
        1.1718759e-02,
        1.1718764e-02,
        1.3782934e-02,
        1.8288977e-02,
        2.5368163e-02,
        1.1718759e-02,
        1.1718759e-02,
        1.1718762e-02,
        1.1718766e-02,
        1.3782934e-02,
        1.7413978e-02,
        2.2557227e-02,
        3.4232263e-02,
        1.1968765e-02,
        1.2320629e-02,
        1.3408982e-02,
        1.5336527e-02,
        1.8288977e-02,
        2.2557227e-02,
        3.2079678e-02,
        4.8214123e-02,
        1.6986061e-02,
        1.7413978e-02,
        1.8736197e-02,
        2.1072537e-02,
        2.5368163e-02,
        3.4232263e-02,
        4.8214123e-02,
        7.0312120e-02,
    ],
];

/// DCT16X8 dequant weights, shared with DCT8X16 (libjxl-tiny convention).
/// 128 floats per channel = 2 8×8 blocks of storage.
/// DCT16X8 layout: weights[c][y*8 + x], y in 0..16, x in 0..8.
/// DCT8X16 reuses the same 128 floats; both have the same per-channel offset
/// in libjxl-tiny's `kQuantWeights` table (offset 3, 5, 7 in blocks).
pub(crate) static DEQUANT_MATRIX_16X8: [[f32; 128]; 3] = [
    // Channel 0 (X)
    [
        1.3810680e-04,
        1.6047071e-04,
        1.8645605e-04,
        2.1664926e-04,
        2.5173181e-04,
        2.9249521e-04,
        3.3985957e-04,
        3.9489369e-04,
        4.1871337e-04,
        4.4087201e-04,
        4.6420316e-04,
        4.8876996e-04,
        5.1463587e-04,
        5.4187077e-04,
        5.7054684e-04,
        6.0074159e-04,
        1.9049694e-04,
        1.9694651e-04,
        2.1442315e-04,
        2.4016941e-04,
        2.7289384e-04,
        3.1245520e-04,
        3.5932945e-04,
        4.0429863e-04,
        4.2484730e-04,
        4.4662904e-04,
        4.6966935e-04,
        4.9400958e-04,
        5.1969837e-04,
        5.4679497e-04,
        5.7536521e-04,
        6.0547784e-04,
        2.6276117e-04,
        2.6734054e-04,
        2.8085473e-04,
        3.0283103e-04,
        3.3291124e-04,
        3.7106971e-04,
        4.0540050e-04,
        4.2322354e-04,
        4.4259510e-04,
        4.6344541e-04,
        4.8574654e-04,
        5.0949713e-04,
        5.3471868e-04,
        5.6144706e-04,
        5.8973121e-04,
        6.1962701e-04,
        3.6243827e-04,
        3.6666830e-04,
        3.7935356e-04,
        3.9960333e-04,
        4.0956112e-04,
        4.2183659e-04,
        4.3620329e-04,
        4.5248115e-04,
        4.7053859e-04,
        4.9028790e-04,
        5.1167433e-04,
        5.3467450e-04,
        5.5928703e-04,
        5.8552966e-04,
        6.1343284e-04,
        6.4304151e-04,
        4.3123538e-04,
        4.3253010e-04,
        4.3638589e-04,
        4.4272337e-04,
        4.5142765e-04,
        4.6236772e-04,
        4.7541404e-04,
        4.9045112e-04,
        5.0738233e-04,
        5.2613625e-04,
        5.4666388e-04,
        5.6893763e-04,
        5.9295003e-04,
        6.1870721e-04,
        6.4623309e-04,
        6.7556335e-04,
        4.8162136e-04,
        4.8277923e-04,
        4.8623976e-04,
        4.9196521e-04,
        4.9989921e-04,
        5.0997391e-04,
        5.2211789e-04,
        5.3626322e-04,
        5.5235048e-04,
        5.7033246e-04,
        5.9017621e-04,
        6.1186077e-04,
        6.3538179e-04,
        6.6074729e-04,
        6.8797835e-04,
        7.5214857e-04,
        5.3789350e-04,
        5.3897168e-04,
        5.4219965e-04,
        5.4755906e-04,
        5.5502116e-04,
        5.6455150e-04,
        5.7611306e-04,
        5.8966759e-04,
        6.0518348e-04,
        6.2263483e-04,
        6.4200454e-04,
        6.6328526e-04,
        6.8647927e-04,
        7.3936180e-04,
        8.0337300e-04,
        8.7534159e-04,
        6.0074159e-04,
        6.0177385e-04,
        6.0486794e-04,
        6.1001495e-04,
        6.1720144e-04,
        6.2641077e-04,
        6.3762552e-04,
        6.5082888e-04,
        6.6600717e-04,
        6.8315107e-04,
        7.1794738e-04,
        7.6673855e-04,
        8.2213664e-04,
        8.8475435e-04,
        9.5529074e-04,
        1.0345384e-03,
    ],
    // Channel 1 (Y)
    [
        6.9053401e-04,
        7.7444571e-04,
        8.6855399e-04,
        9.7409816e-04,
        1.0924696e-03,
        1.2252233e-03,
        1.3741088e-03,
        1.5410866e-03,
        1.7283577e-03,
        1.9383827e-03,
        2.1739292e-03,
        2.3783136e-03,
        2.5041751e-03,
        2.6366978e-03,
        2.7762330e-03,
        2.9231580e-03,
        8.8290084e-04,
        9.0565201e-04,
        9.6644071e-04,
        1.0539154e-03,
        1.1619731e-03,
        1.2886107e-03,
        1.4338633e-03,
        1.5988132e-03,
        1.7851711e-03,
        1.9951246e-03,
        2.2312698e-03,
        2.4038092e-03,
        2.5288085e-03,
        2.6606584e-03,
        2.7996788e-03,
        2.9462043e-03,
        1.1288587e-03,
        1.1438611e-03,
        1.1877866e-03,
        1.2581701e-03,
        1.3525900e-03,
        1.4695247e-03,
        1.6085195e-03,
        1.7700332e-03,
        1.9552717e-03,
        2.1660449e-03,
        2.3636019e-03,
        2.4791702e-03,
        2.6018962e-03,
        2.7319542e-03,
        2.8695827e-03,
        3.0150530e-03,
        1.4433329e-03,
        1.4561869e-03,
        1.4945272e-03,
        1.5578135e-03,
        1.6454635e-03,
        1.7571596e-03,
        1.8930284e-03,
        2.0537286e-03,
        2.2404641e-03,
        2.3856999e-03,
        2.4897642e-03,
        2.6016813e-03,
        2.7214440e-03,
        2.8491386e-03,
        2.9849128e-03,
        3.1289863e-03,
        1.8454153e-03,
        1.8577600e-03,
        1.8947916e-03,
        1.9565322e-03,
        2.0431101e-03,
        2.1548597e-03,
        2.2924175e-03,
        2.3864941e-03,
        2.4688798e-03,
        2.5601350e-03,
        2.6600207e-03,
        2.7684029e-03,
        2.8852450e-03,
        3.0105773e-03,
        3.1445161e-03,
        3.2872346e-03,
        2.3435291e-03,
        2.3491632e-03,
        2.3660017e-03,
        2.3938618e-03,
        2.4324679e-03,
        2.4814904e-03,
        2.5405819e-03,
        2.6094120e-03,
        2.6876912e-03,
        2.7751899e-03,
        2.8717481e-03,
        2.9772632e-03,
        3.0917146e-03,
        3.2151414e-03,
        3.3476453e-03,
        3.4893905e-03,
        2.6173447e-03,
        2.6225911e-03,
        2.6382981e-03,
        2.6643763e-03,
        2.7006865e-03,
        2.7470603e-03,
        2.8033180e-03,
        2.8692731e-03,
        2.9447721e-03,
        3.0296890e-03,
        3.1239407e-03,
        3.2274905e-03,
        3.3403505e-03,
        3.4625907e-03,
        3.5943130e-03,
        3.7356857e-03,
        2.9231580e-03,
        2.9281813e-03,
        2.9432368e-03,
        2.9682817e-03,
        3.0032503e-03,
        3.0480621e-03,
        3.1026325e-03,
        3.1668788e-03,
        3.2407353e-03,
        3.3241559e-03,
        3.4171303e-03,
        3.5196650e-03,
        3.6318223e-03,
        3.7536959e-03,
        3.8854245e-03,
        4.0271855e-03,
    ],
    // Channel 2 (B)
    [
        1.9729543e-03,
        2.5272998e-03,
        3.2374004e-03,
        4.1470206e-03,
        4.8498721e-03,
        5.1065302e-03,
        5.3767711e-03,
        5.6613120e-03,
        6.3208523e-03,
        7.0889443e-03,
        7.9503711e-03,
        8.9164926e-03,
        9.9999988e-03,
        1.1215170e-02,
        1.2578006e-02,
        1.5967883e-02,
        3.3539708e-03,
        3.5433732e-03,
        4.0769530e-03,
        4.7721486e-03,
        4.9862624e-03,
        5.2236770e-03,
        5.4806760e-03,
        5.8470899e-03,
        6.5286267e-03,
        7.2964565e-03,
        8.1600742e-03,
        9.1304630e-03,
        1.0220082e-02,
        1.1443083e-02,
        1.2854187e-02,
        1.6610704e-02,
        4.9218577e-03,
        4.9511627e-03,
        5.0357706e-03,
        5.1678251e-03,
        5.3387447e-03,
        5.5415547e-03,
        5.8825868e-03,
        6.4732647e-03,
        7.1507092e-03,
        7.9215374e-03,
        8.7942975e-03,
        9.7792931e-03,
        1.0888627e-02,
        1.2136214e-02,
        1.4550334e-02,
        1.8655479e-02,
        5.4969229e-03,
        5.5188821e-03,
        5.5837538e-03,
        5.6971479e-03,
        6.0176966e-03,
        6.4261849e-03,
        6.9230762e-03,
        7.5107799e-03,
        8.1936987e-03,
        8.9781955e-03,
        9.8724691e-03,
        1.0886625e-02,
        1.2032620e-02,
        1.4036770e-02,
        1.7736901e-02,
        2.2478340e-02,
        6.7489492e-03,
        6.7940955e-03,
        6.9295247e-03,
        7.1553192e-03,
        7.4719470e-03,
        7.8806318e-03,
        8.3836997e-03,
        8.9848433e-03,
        9.6892491e-03,
        1.0503774e-02,
        1.1436986e-02,
        1.2499244e-02,
        1.4953869e-02,
        1.8516723e-02,
        2.3044668e-02,
        2.8803868e-02,
        8.6290650e-03,
        8.6752698e-03,
        8.8141672e-03,
        9.0466458e-03,
        9.3743140e-03,
        9.7996546e-03,
        1.0326200e-02,
        1.0958694e-02,
        1.1703256e-02,
        1.2567491e-02,
        1.4605597e-02,
        1.7509630e-02,
        2.1164555e-02,
        2.5766177e-02,
        3.1564441e-02,
        4.4291977e-02,
        1.1032925e-02,
        1.1082166e-02,
        1.1230315e-02,
        1.1478675e-02,
        1.1829470e-02,
        1.2285955e-02,
        1.2938387e-02,
        1.4542448e-02,
        1.6570158e-02,
        1.9115077e-02,
        2.2296766e-02,
        2.6267400e-02,
        3.1220267e-02,
        4.1523870e-02,
        5.6756895e-02,
        7.8389272e-02,
        1.5967883e-02,
        1.6106272e-02,
        1.6526788e-02,
        1.7245775e-02,
        1.8291343e-02,
        1.9704822e-02,
        2.1542856e-02,
        2.3880199e-02,
        2.6813647e-02,
        3.0466938e-02,
        3.7175436e-02,
        4.7613274e-02,
        6.1909460e-02,
        8.1609353e-02,
        1.0892317e-01,
        1.4702357e-01,
    ],
];

pub(crate) struct DequantMatrices {
    pub(crate) matrix: HeapMatrix<f32, 3, 64>,
    /// Per-channel inverse matrices (1/weight). Entry [c][0] is zeroed because
    /// DC is quantized separately via DC_QUANT.
    pub(crate) inv_matrix: HeapMatrix<f32, 3, 64>,
    /// 16×8 / 8×16 dequant matrix. Both rectangular transforms share these
    /// 128 floats per channel (libjxl-tiny convention).
    pub(crate) matrix_16x8: HeapMatrix<f32, 3, 128>,
    pub(crate) inv_matrix_16x8: HeapMatrix<f32, 3, 128>,
    /// 16×16 dequant matrix (256 floats per channel). Generated at
    /// construction time from the libjxl polynomial parameters since it isn't
    /// part of libjxl-tiny.
    pub(crate) matrix_16x16: HeapMatrix<f32, 3, 256>,
    pub(crate) inv_matrix_16x16: HeapMatrix<f32, 3, 256>,
    /// 32×32 dequant matrix (1024 floats per channel). Generated at
    /// construction time from the libjxl DCT32X32 polynomial parameters.
    pub(crate) matrix_32x32: HeapMatrix<f32, 3, 1024>,
    pub(crate) inv_matrix_32x32: HeapMatrix<f32, 3, 1024>,
    /// Spec-default DCT64X64 table used by the slow large-transform path.
    pub(crate) matrix_64x64: HeapMatrix<f32, 3, 4096>,
    pub(crate) inv_matrix_64x64: HeapMatrix<f32, 3, 4096>,
    /// Shared normalized 32-row x 64-column table for DCT64X32/DCT32X64.
    pub(crate) matrix_64x32: HeapMatrix<f32, 3, 2048>,
    pub(crate) inv_matrix_64x32: HeapMatrix<f32, 3, 2048>,
    /// Tables that differ from the spec defaults and must be signaled by
    /// `write_dequant_matrices`, in the fixed order
    pub(crate) custom_tables: Box<[Option<BandOverride>; 8]>,
    /// DCT4X4 dequant matrix (64 floats per channel, 8×8 grid). Generated from
    /// the libjxl DCT4X4 4-band parameters: 4×4 radial weights replicated to
    /// 2×2 cells. Used for the sub-8×8 DCT4X4 transform.
    pub(crate) matrix_4x4: HeapMatrix<f32, 3, 64>,
    pub(crate) inv_matrix_4x4: HeapMatrix<f32, 3, 64>,
    /// DCT4X8 dequant matrix (64 floats per channel). 4×8 radial weights with
    /// each row replicated to 2 rows of the 8×8 grid. Used for DCT4X8.
    pub(crate) matrix_4x8: HeapMatrix<f32, 3, 64>,
    pub(crate) inv_matrix_4x8: HeapMatrix<f32, 3, 64>,
    /// DCT32X16 / DCT16X32 dequant matrix (512 floats per channel). Both
    /// rectangular large transforms share these weights (libjxl
    /// `QuantTable::DCT16X32`), computed at the normalized 16-row × 32-col
    /// resolution so the same table applies to both orientations.
    pub(crate) matrix_32x16: HeapMatrix<f32, 3, 512>,
    pub(crate) inv_matrix_32x16: HeapMatrix<f32, 3, 512>,
    /// AFV dequant matrix (64 floats per channel, 8×8 grid), shared by all
    /// four AFV variants (libjxl `QuantTable::AFV0`).
    pub(crate) matrix_afv: HeapMatrix<f32, 3, 64>,
    pub(crate) inv_matrix_afv: HeapMatrix<f32, 3, 64>,
}

/// libjxl `DequantMatricesLibraryDef::DCT16X16()` parameters: 7 distance
/// bands per channel. Channel order is X, Y, B (jixel convention). Source:
/// libjxl `lib/jxl/quant_weights.cc`. The first value of each row is the
/// inverse step size at the DC (radius 0) position; subsequent values are
/// multiplicative deltas between successive bands. Same format as
/// jxl-rs `dct16x16()`.
static DCT16X16_BANDS: [[f32; 7]; 3] = [
    // X
    [
        8996.872_571_181_412,
        -1.300_077_739_335_380_4,
        -0.494_245_298_245_712_25,
        -0.439_093_774_457_103_44,
        -0.635_010_183_269_574_4,
        -0.901_772_640_508_276_1,
        -1.616_209_923_988_741_4,
    ],
    // Y
    [
        3191.483_662_968_442_3,
        -0.674_245_821_041_943_55,
        -0.807_458_134_284_710,
        -0.449_258_374_848_434_4,
        -0.358_654_409_810_334_03,
        -0.313_223_891_118_773_05,
        -0.376_150_253_157_254_83,
    ],
    // B
    [
        1157.504_081_454_872,
        -2.053_142_316_580_441_4,
        -1.4,
        -0.506_871_300_333_784,
        -0.427_087_306_247_339_04,
        -1.485_683_453_929_624_4,
        -4.920_914_288_440_160,
    ],
];

/// libjxl `DequantMatricesLibraryDef::DCT32X32()` parameters: 8 distance
/// bands per channel (X, Y, B). Same format as [`DCT16X16_BANDS`]. Source:
/// libjxl `lib/jxl/quant_weights.cc`.
static DCT32X32_BANDS: [[f32; 8]; 3] = [
    // X
    [
        15718.408_309_825_19,
        -1.025,
        -0.98,
        -0.901_2,
        -0.4,
        -0.488_193_95,
        -0.421_064,
        -0.27,
    ],
    // Y
    [
        7305.763_681_069_598,
        -0.804_195_82,
        -0.763_303_65,
        -0.556_603_8,
        -0.497_853_05,
        -0.436_995_93,
        -0.401_808_67,
        -0.273_216_83,
    ],
    // B
    [
        3803.531_737_212_15,
        -3.060_733_6,
        -2.041_327,
        -2.023_565,
        -0.549_538_95,
        -0.4,
        -0.4,
        -0.3,
    ],
];

/// libjxl band-step multiplicative helper. Positive → 1+v, negative → 1/(1-v).
/// Matches `jxl::DequantMatricesLibrary::Mult` and jxl-rs `mult`.
#[inline]
fn band_mult(v: f32) -> f32 {
    if v > 0.0 { 1.0 + v } else { 1.0 / (1.0 - v) }
}

#[cfg(test)]
static DCT16X8_BANDS: [[f32; 7]; 3] = [
    // X
    [7240.7734393502, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
    // Y
    [1448.15468787004, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
    // B
    [506.854002029, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
];

static DCT16X32_BANDS: [[f32; 8]; 3] = [
    // X
    [
        13844.97076442300573,
        -0.97113799999999995,
        -0.658,
        -0.42026,
        -0.22712,
        -0.2206,
        -0.226,
        -0.6,
    ],
    // Y
    [
        4798.964084220744293,
        -0.61125308982767057,
        -0.83770786552491361,
        -0.79014862079498627,
        -0.2692727459704829,
        -0.38272769465388551,
        -0.22924222653091453,
        -0.20719098826199578,
    ],
    // B
    [
        1807.236946760964614,
        -1.2,
        -1.2,
        -0.7,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
];

/// libjxl `DequantMatricesLibraryDef::DCT4X4()` parameters: 4 distance bands
/// per channel (X, Y, B). The 4×4 radial weights are computed from these, then
/// each is replicated to a 2×2 cell of the 8×8 block. The `dct4multipliers`
/// libjxl applies to the three 2×2-Hadamard DC positions are all 1.0, so they
/// are a no-op and omitted here. Source: libjxl `lib/jxl/quant_weights.cc`.
static DCT4X4_BANDS: [[f32; 4]; 3] = [
    [2200.0, 0.0, 0.0, 0.0],
    [392.0, 0.0, 0.0, 0.0],
    [112.0, -0.25, -0.25, -0.5],
];

/// libjxl `DequantMatricesLibraryDef::DCT4X8()` parameters: 4 distance bands per
/// channel. Computed as 4×8 radial weights (rows=4, cols=8), then each of the 4
/// rows is replicated to 2 rows of the 8×8 block. The `dct4x8multipliers` are
/// all 1.0 (no-op) and omitted. Source: libjxl `lib/jxl/quant_weights.cc`.
static DCT4X8_BANDS: [[f32; 4]; 3] = [
    [
        2198.050556016380522,
        -0.96269623020744692,
        -0.76194253026666783,
        -0.6551140670773547,
    ],
    [
        764.3655248643528689,
        -0.92630200888366945,
        -0.9675229603596517,
        -0.27845290869168118,
    ],
    [
        527.107573587542228,
        -1.4594385811273854,
        -1.450082094097871593,
        -1.5843722511996204,
    ],
];

static DCT64X64_BANDS: [[f32; 8]; 3] = [
    [
        0.9 * 26629.073922049845,
        -1.025,
        -0.78,
        -0.65012,
        -0.19041574084286472,
        -0.20819395464,
        -0.421064,
        -0.3273384553584867,
    ],
    [
        0.9 * 9311.323871001005,
        -0.3041958212306401,
        -0.3633036457487539,
        -0.35660379990111464,
        -0.3443074455424403,
        -0.33699592683512467,
        -0.3018086652624211,
        -0.27321683125358037,
    ],
    [
        0.9 * 4992.248644553863,
        -1.2,
        -1.2,
        -0.8,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
];

/// JPEG XL DCT32X64 parameters, shared by both orientations.
static DCT32X64_BANDS: [[f32; 8]; 3] = [
    [
        0.65 * 23629.073922049845,
        -1.025,
        -0.78,
        -0.65012,
        -0.19041574084286472,
        -0.20819395464,
        -0.421064,
        -0.3273384553584867,
    ],
    [
        0.65 * 8611.323871001005,
        -0.3041958212306401,
        -0.3633036457487539,
        -0.35660379990111464,
        -0.3443074455424403,
        -0.33699592683512467,
        -0.3018086652624211,
        -0.27321683125358037,
    ],
    [
        0.65 * 4492.248644553863,
        -1.2,
        -1.2,
        -0.8,
        -0.7,
        -0.7,
        -0.4,
        -0.5,
    ],
];

fn compute_dct64x64_matrix(override_: Option<&BandOverride>) -> HeapMatrix<f32, 3, 4096> {
    const NUM_BANDS: usize = 8;
    let mut src = DCT64X64_BANDS;
    if let Some(ov) = override_ {
        for c in 0..3 {
            src[c].copy_from_slice(&ov.bands[c][..NUM_BANDS]);
        }
    }
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = src[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(src[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcp = scale / 63.0;
        for y in 0..64 {
            let dy = y as f32 * rcp;
            for x in 0..64 {
                let dx = x as f32 * rcp;
                out[c][y * 64 + x] =
                    1.0 / interpolate_vec_bands(fmla(dx, dx, dy * dy).sqrt(), &bands);
            }
        }
    }
    out
}

fn compute_dct64x32_matrix(override_: Option<&BandOverride>) -> HeapMatrix<f32, 3, 2048> {
    const NUM_BANDS: usize = 8;
    let mut src = DCT32X64_BANDS;
    if let Some(ov) = override_ {
        for c in 0..3 {
            src[c].copy_from_slice(&ov.bands[c][..NUM_BANDS]);
        }
    }
    let mut out = HeapMatrix::new(0.0);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = src[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(src[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcprow = scale / 31.0;
        let rcpcol = scale / 63.0;
        for y in 0..32 {
            let dy = y as f32 * rcprow;
            for x in 0..64 {
                let dx = x as f32 * rcpcol;
                out[c][y * 64 + x] =
                    1.0 / interpolate_vec_bands(fmla(dx, dx, dy * dy).sqrt(), &bands);
            }
        }
    }
    out
}

/// libjxl `DequantMatricesLibraryDef::AFV0()` parameters: 9 `afv_weights` per
/// channel (X, Y, B), shared by all four AFV variants. Layout per channel:
/// [0..2) the 4x8/4x4 sub-part DC tendencies (coefficients 8 and 1), [2..5)
/// the fixed 3-pixel-corner weights (coefficients 16, 2 and 18), [5..9) the
/// 4 distance bands for the remaining AFV positions (same first-value +
/// multiplicative-delta format as the DCT band tables). Source: libjxl
/// `lib/jxl/quant_weights.cc`.
static AFV_BANDS: [[f32; 9]; 3] = [
    [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
    [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
    [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
];

/// libjxl `kFreqs`: the radial frequency assigned to each AFV basis function,
/// used to place the non-corner AFV weights on the 4-band curve. Positions
/// with fixed weights (the 2x2 low corner) are never read; libjxl fills them
/// with the 0xBAD sentinel, kept here verbatim.
#[rustfmt::skip]
static AFV_FREQS: [f32; 16] = [
    0xBAD as f32, 0xBAD as f32, 0.8517778890324296, 5.37778436506804,
    0xBAD as f32, 0xBAD as f32, 4.734747904497923, 5.449245381693219,
    1.6598270267479331, 4.0, 7.275749096817861, 10.423227632456525,
    2.662932286148962, 7.630657783650829, 8.962388608184032, 12.97166202570235,
];

const AFV_FREQ_LO: f32 = 0.8517778890324296;
const AFV_FREQ_HI: f32 = 12.97166202570235 - AFV_FREQ_LO + 1e-6;

/// libjxl interpolation between band weights. The two surrounding band values
/// `a`, `b` are interpolated geometrically (exponential) using the fractional
/// scaled distance.
#[inline]
fn interpolate_vec_bands(scaled_pos: f32, bands: &[f32]) -> f32 {
    let idx_f = scaled_pos.floor();
    let frac = scaled_pos - idx_f;
    let idx = idx_f as usize;
    let a = bands[idx];
    let b = bands[idx + 1];
    (b / a).powf(frac) * a
}

/// Reproduce libjxl's DCT4X8 quant table (`kQuantModeDCT4X8`): compute 4×8
/// radial weights via `GetQuantWeights(4, 8, bands, num_bands=4)` (rows=4,
/// cols=8 → separate row/col scaling), then expand to the 8×8 block with
/// `w8x8[y*8+x] = w4x8[(y/2)*8 + x]` (each of the 4 rows replicated to 2).
/// Returns inverse weights (matrix entry = 1/weight = step size).
fn compute_dct4x8_matrix(override_: Option<&BandOverride>) -> HeapMatrix<f32, 3, 64> {
    const NUM_BANDS: usize = 4;
    let mut src = DCT4X8_BANDS;
    if let Some(o) = override_ {
        for c in 0..3 {
            src[c].copy_from_slice(&o.bands[c][..NUM_BANDS]);
        }
    }
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = src[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(src[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcprow = scale / 3.0; // ROWS - 1 = 3
        let rcpcol = scale / 7.0; // COLS - 1 = 7
        // 4×8 radial weights (4 rows, 8 cols).
        let mut w4x8 = [0.0f32; 32];
        for y in 0..4 {
            let dy = y as f32 * rcprow;
            let dy2 = dy * dy;
            for x in 0..8 {
                let dx = x as f32 * rcpcol;
                let dist = fmla(dx, dx, dy2).sqrt();
                w4x8[y * 8 + x] = interpolate_vec_bands(dist, &bands);
            }
        }
        // Replicate each 4×8 row to two rows of the 8×8 grid; store 1/weight.
        for y in 0..8 {
            for x in 0..8 {
                let w = w4x8[(y / 2) * 8 + x];
                out[c][y * 8 + x] = 1.0 / w;
            }
        }
    }
    out
}

fn compute_dct4x4_matrix() -> HeapMatrix<f32, 3, 64> {
    const NUM_BANDS: usize = 4;
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = DCT4X4_BANDS[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT4X4_BANDS[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcp = scale / 3.0; // (4 - 1)
        // 4×4 radial weights.
        let mut w4x4 = [0.0f32; 16];
        for y in 0..4 {
            let dy = y as f32 * rcp;
            let dy2 = dy * dy;
            for x in 0..4 {
                let dx = x as f32 * rcp;
                let dist = fmla(dx, dx, dy2).sqrt();
                w4x4[y * 4 + x] = interpolate_vec_bands(dist, &bands);
            }
        }
        // Expand each 4×4 weight to a 2×2 cell of the 8×8 grid; store 1/weight.
        for y in 0..8 {
            for x in 0..8 {
                let w = w4x4[(y / 2) * 4 + (x / 2)];
                out[c][y * 8 + x] = 1.0 / w;
            }
        }
    }
    out
}

/// Reproduce libjxl's AFV quant table (`kQuantModeAFV`), shared by all four
/// AFV variants. The 8×8 block mixes three weight sources, mirroring the
/// coefficient layout of the transform: odd rows carry the 4×8 radial weights
/// (from the DCT4X8 library params), even rows / odd columns the 4×4 radial
/// weights (from the DCT4X4 library params), and the (even, even) AFV
/// positions get either fixed corner weights or a 4-band interpolation over
/// the basis-function frequencies. Returns step sizes (1/weight).
fn compute_afv_matrix() -> HeapMatrix<f32, 3, 64> {
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        // 4×8 radial weights, exactly as compute_dct4x8_matrix builds them.
        let mut bands = [0.0f32; 4];
        bands[0] = DCT4X8_BANDS[c][0];
        for i in 1..4 {
            bands[i] = bands[i - 1] * band_mult(DCT4X8_BANDS[c][i]);
        }
        let scale = 3.0 / (std::f32::consts::SQRT_2 + 1e-6);
        let mut w4x8 = [0.0f32; 32];
        for y in 0..4 {
            let dy = y as f32 * (scale / 3.0);
            let dy2 = dy * dy;
            for x in 0..8 {
                let dx = x as f32 * (scale / 7.0);
                let dist = fmla(dx, dx, dy2).sqrt();
                w4x8[y * 8 + x] = interpolate_vec_bands(dist, &bands);
            }
        }
        // 4×4 radial weights, exactly as compute_dct4x4_matrix builds them.
        bands[0] = DCT4X4_BANDS[c][0];
        for i in 1..4 {
            bands[i] = bands[i - 1] * band_mult(DCT4X4_BANDS[c][i]);
        }
        let mut w4x4 = [0.0f32; 16];
        for y in 0..4 {
            let dy = y as f32 * (scale / 3.0);
            let dy2 = dy * dy;
            for x in 0..4 {
                let dx = x as f32 * (scale / 3.0);
                let dist = fmla(dx, dx, dy2).sqrt();
                w4x4[y * 4 + x] = interpolate_vec_bands(dist, &bands);
            }
        }
        // AFV bands for the non-corner (even, even) positions.
        bands[0] = AFV_BANDS[c][5];
        for i in 1..4 {
            bands[i] = bands[i - 1] * band_mult(AFV_BANDS[c][i + 5]);
        }
        let mut weights = [0.0f32; 64];
        // Coefficient 0 is the block mean, quantized via the DC plane; libjxl
        // stores weight 1 in the unused slot.
        weights[0] = 1.0;
        // Sub-part DC tendencies: coefficient 8 (4x8 half) and 1 (4x4 quad).
        weights[8] = AFV_BANDS[c][0];
        weights[1] = AFV_BANDS[c][1];
        // Fixed weights for the 3-pixel AFV corner.
        weights[16] = AFV_BANDS[c][2];
        weights[2] = AFV_BANDS[c][3];
        weights[18] = AFV_BANDS[c][4];
        // Remaining AFV positions from the frequency interpolation.
        for y in 0..4 {
            for x in 0..4 {
                if x < 2 && y < 2 {
                    continue;
                }
                let pos = (AFV_FREQS[y * 4 + x] - AFV_FREQ_LO) * 3.0 / AFV_FREQ_HI;
                weights[(2 * y) * 8 + 2 * x] = interpolate_vec_bands(pos, &bands);
            }
        }
        // 4×8 weights in the odd rows, except position 8 (kept above).
        for y in 0..4 {
            for x in 0..8 {
                if x == 0 && y == 0 {
                    continue;
                }
                weights[(2 * y + 1) * 8 + x] = w4x8[y * 8 + x];
            }
        }
        // 4×4 weights in even rows / odd columns, except position 1.
        for y in 0..4 {
            for x in 0..4 {
                if x == 0 && y == 0 {
                    continue;
                }
                weights[(2 * y) * 8 + 2 * x + 1] = w4x4[y * 4 + x];
            }
        }
        for (o, &w) in out[c].iter_mut().zip(weights.iter()) {
            *o = 1.0 / w;
        }
    }
    out
}

fn compute_dct8x8_matrix(override_: &BandOverride) -> HeapMatrix<f32, 3, 64> {
    const NUM_BANDS: usize = 6;
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = override_.bands[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(override_.bands[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcp = scale / 7.0;
        for y in 0..8 {
            let dy = y as f32 * rcp;
            let dy2 = dy * dy;
            for x in 0..8 {
                let dx = x as f32 * rcp;
                let dist = fmla(dx, dx, dy2).sqrt();
                out[c][y * 8 + x] = 1.0 / interpolate_vec_bands(dist, &bands);
            }
        }
    }
    out
}

fn compute_dct16x16_matrix(override_: Option<&BandOverride>) -> HeapMatrix<f32, 3, 256> {
    const NUM_BANDS: usize = 7;
    let mut src = DCT16X16_BANDS;
    if let Some(o) = override_ {
        for c in 0..3 {
            src[c].copy_from_slice(&o.bands[c][..NUM_BANDS]);
        }
    }
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = src[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(src[c][i]);
        }
        // libjxl: `scale = (num_bands - 1) / (sqrt(2) + 1e-6)` — the (15,15)
        // corner radial distance scales to (num_bands - 1).
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcp = scale / 15.0;
        for y in 0..16 {
            let dy = y as f32 * rcp;
            let dy2 = dy * dy;
            for x in 0..16 {
                let dx = x as f32 * rcp;
                let dist = fmla(dx, dx, dy2).sqrt();
                let weight = interpolate_vec_bands(dist, &bands);
                // libjxl stores 1/weight as the matrix entry (step size).
                out[c][y * 16 + x] = 1.0 / weight;
            }
        }
    }
    out
}

fn compute_dct32x32_matrix(override_: Option<&BandOverride>) -> HeapMatrix<f32, 3, 1024> {
    const NUM_BANDS: usize = 8;
    let mut src = DCT32X32_BANDS;
    if let Some(o) = override_ {
        for c in 0..3 {
            src[c].copy_from_slice(&o.bands[c][..NUM_BANDS]);
        }
    }
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = src[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(src[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcp = scale / 31.0;
        for y in 0..32 {
            let dy = y as f32 * rcp;
            let dy2 = dy * dy;
            for x in 0..32 {
                let dx = x as f32 * rcp;
                let dist = fmla(dx, dx, dy2).sqrt();
                let weight = interpolate_vec_bands(dist, &bands);
                out[c][y * 32 + x] = 1.0 / weight;
            }
        }
    }
    out
}

#[cfg(test)]
fn compute_dct16x8_matrix(override_: Option<&BandOverride>) -> HeapMatrix<f32, 3, 128> {
    const NUM_BANDS: usize = 7;
    let mut src = DCT16X8_BANDS;
    if let Some(o) = override_ {
        for c in 0..3 {
            src[c].copy_from_slice(&o.bands[c][..NUM_BANDS]);
        }
    }
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = src[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(src[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcprow = scale / 7.0; // ROWS = 8
        let rcpcol = scale / 15.0; // COLS = 16
        for y in 0..8 {
            let dy = y as f32 * rcprow;
            let dy2 = dy * dy;
            for x in 0..16 {
                let dx = x as f32 * rcpcol;
                let dist = fmla(dx, dx, dy2).sqrt();
                let weight = interpolate_vec_bands(dist, &bands);
                out[c][y * 16 + x] = 1.0 / weight;
            }
        }
    }
    out
}

fn compute_dct32x16_matrix(override_: Option<&BandOverride>) -> HeapMatrix<f32, 3, 512> {
    const NUM_BANDS: usize = 8;
    let mut src = DCT16X32_BANDS;
    if let Some(o) = override_ {
        for c in 0..3 {
            src[c].copy_from_slice(&o.bands[c][..NUM_BANDS]);
        }
    }
    let mut out = HeapMatrix::new(0.0f32);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = src[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(src[c][i]);
        }
        let scale = (NUM_BANDS as f32 - 1.0) / (std::f32::consts::SQRT_2 + 1e-6);
        let rcprow = scale / 15.0; // ROWS = 16
        let rcpcol = scale / 31.0; // COLS = 32
        for y in 0..16 {
            let dy = y as f32 * rcprow;
            let dy2 = dy * dy;
            for x in 0..32 {
                let dx = x as f32 * rcpcol;
                let dist = fmla(dx, dx, dy2).sqrt();
                let weight = interpolate_vec_bands(dist, &bands);
                out[c][y * 32 + x] = 1.0 / weight;
            }
        }
    }
    out
}

/// Quant tables that do not depend on the SS2-retune distance gate, computed
/// once per process and cloned into both `DequantMatrices` variants.
struct SharedTables {
    matrix: HeapMatrix<f32, 3, 64>,
    inv_matrix: HeapMatrix<f32, 3, 64>,
    matrix_16x8: HeapMatrix<f32, 3, 128>,
    inv_matrix_16x8: HeapMatrix<f32, 3, 128>,
    matrix_4x4: HeapMatrix<f32, 3, 64>,
    inv_matrix_4x4: HeapMatrix<f32, 3, 64>,
    matrix_4x8: HeapMatrix<f32, 3, 64>,
    inv_matrix_4x8: HeapMatrix<f32, 3, 64>,
    matrix_afv: HeapMatrix<f32, 3, 64>,
    inv_matrix_afv: HeapMatrix<f32, 3, 64>,
    matrix_64x64: HeapMatrix<f32, 3, 4096>,
    inv_matrix_64x64: HeapMatrix<f32, 3, 4096>,
    matrix_64x32: HeapMatrix<f32, 3, 2048>,
    inv_matrix_64x32: HeapMatrix<f32, 3, 2048>,
}

fn shared_tables() -> &'static SharedTables {
    static SHARED: std::sync::OnceLock<Box<SharedTables>> = std::sync::OnceLock::new();
    SHARED.get_or_init(|| {
        let matrix = HeapMatrix::from_rows(&DEQUANT_MATRIX_8X8);
        let mut inv_matrix = HeapMatrix::new(0.);
        for c in 0..3 {
            for k in 1..64 {
                inv_matrix[c][k] = 1.0 / matrix[c][k];
            }
        }

        // Use the precomputed static rather than recomputing from bands: the
        // two differ by ~4e-6 relative (f32 rounding of the stored constants),
        // which is enough to shift a few quantizer decisions.
        let matrix_16x8 = HeapMatrix::from_rows(&DEQUANT_MATRIX_16X8);
        let mut inv_matrix_16x8 = HeapMatrix::new(0.);
        for c in 0..3 {
            for k in 1..128 {
                inv_matrix_16x8[c][k] = 1.0 / matrix_16x8[c][k];
            }
        }

        let matrix_4x4 = compute_dct4x4_matrix();
        let mut inv_matrix_4x4 = HeapMatrix::new(0.);
        for c in 0..3 {
            // DC slot (index 0) zeroed (handled by the DC plane). For DCT4X4 the
            // only LLF position is the DC; [1], [8], [9] are regular AC.
            for k in 1..64 {
                inv_matrix_4x4[c][k] = 1.0 / matrix_4x4[c][k];
            }
        }

        let matrix_4x8 = compute_dct4x8_matrix(None);
        let mut inv_matrix_4x8 = HeapMatrix::new(0.);
        for c in 0..3 {
            // Only [0] is the DC (handled by the DC plane); [8] (the vertical
            // half-difference after the Hadamard) and all others are regular AC.
            for k in 1..64 {
                inv_matrix_4x8[c][k] = 1.0 / matrix_4x8[c][k];
            }
        }

        let matrix_afv = compute_afv_matrix();
        let mut inv_matrix_afv = HeapMatrix::new(0.);
        for c in 0..3 {
            // Only [0] (the block mean) lives in the DC plane; [1] and [8]
            // (the sub-part DC differences) are regular AC coefficients.
            for k in 1..64 {
                inv_matrix_afv[c][k] = 1.0 / matrix_afv[c][k];
            }
        }

        let matrix_64x64 = compute_dct64x64_matrix(None);
        let mut inv_matrix_64x64 = HeapMatrix::new(0.0f32);
        for c in 0..3 {
            for k in 1..4096 {
                inv_matrix_64x64[c][k] = 1.0 / matrix_64x64[c][k];
            }
        }

        let matrix_64x32 = compute_dct64x32_matrix(None);
        let mut inv_matrix_64x32 = HeapMatrix::new(0.0f32);
        for c in 0..3 {
            for k in 1..2048 {
                inv_matrix_64x32[c][k] = 1.0 / matrix_64x32[c][k];
            }
        }

        Box::new(SharedTables {
            matrix,
            inv_matrix,
            matrix_16x8,
            inv_matrix_16x8,
            matrix_4x4,
            inv_matrix_4x4,
            matrix_4x8,
            inv_matrix_4x8,
            matrix_afv,
            inv_matrix_afv,
            matrix_64x64,
            inv_matrix_64x64,
            matrix_64x32,
            inv_matrix_64x32,
        })
    })
}

impl DequantMatrices {
    pub(crate) fn new(distance: f32) -> &'static Self {
        static HQ: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        static DEFAULT_HQ: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        static DEFAULT_MID: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        static SS2: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        static COARSE: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        if distance >= QM_DCT8_MIN_DISTANCE {
            COARSE.get_or_init(|| Self::compute(true, true, Some(&FLAT_B8_BANDS_MID)))
        } else if distance >= QM_SS2_MIN_DISTANCE {
            SS2.get_or_init(|| Self::compute(true, false, Some(&FLAT_B8_BANDS_MID)))
        } else if distance >= QM_FLAT_B8_MID_MIN_DISTANCE {
            DEFAULT_MID.get_or_init(|| Self::compute(false, false, Some(&FLAT_B8_BANDS_MID)))
        } else if distance >= QM_FLAT_B8_MIN_DISTANCE {
            DEFAULT_HQ.get_or_init(|| Self::compute(false, false, Some(&FLAT_B8_BANDS_HQ)))
        } else {
            HQ.get_or_init(|| Self::compute(false, false, None))
        }
    }

    pub(crate) fn new_saturated(distance: f32) -> &'static Self {
        static SAT_DEFAULT: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        static SAT_SS2: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        static SAT_COARSE: std::sync::OnceLock<DequantMatrices> = std::sync::OnceLock::new();
        if distance >= QM_DCT8_MIN_DISTANCE {
            SAT_COARSE.get_or_init(|| Self::compute(true, true, Some(&SAT_B8_BANDS)))
        } else if distance >= QM_SS2_MIN_DISTANCE {
            SAT_SS2.get_or_init(|| Self::compute(true, false, Some(&SAT_B8_BANDS)))
        } else if distance >= QM_FLAT_B8_MIN_DISTANCE {
            SAT_DEFAULT.get_or_init(|| Self::compute(false, false, Some(&SAT_B8_BANDS)))
        } else {
            Self::new(distance)
        }
    }

    fn compute(use_ss2: bool, use_coarse_dct8: bool, flat_b8: Option<&[f32; 6]>) -> Self {
        // The large-transform tables depend on the SS2 gate, and DCT8 gets a
        // separate coarser-quality variant. All other tables are shared.
        let shared = shared_tables();
        // The coarse DCT8 variant only exists above the flat-B gate, so the
        // no-flat (near-lossless) tier always carries the spec table.
        let o8 = flat_b8.map(|bands| default_dct8_override(use_coarse_dct8, bands));
        let o16 = use_ss2.then(|| scaled_override(&DCT16X16_BANDS, QM_SS2_SCALE16));
        let o32 = use_ss2.then(|| scaled_override(&DCT32X32_BANDS, QM_SS2_SCALE32));
        let o64: Option<BandOverride> = None;
        let o64r: Option<BandOverride> = None;
        let (matrix_64x64, inv_matrix_64x64) = match o64.as_ref() {
            None => (shared.matrix_64x64.clone(), shared.inv_matrix_64x64.clone()),
            Some(ov) => {
                let m = compute_dct64x64_matrix(Some(ov));
                let mut inv = HeapMatrix::new(0.0f32);
                for c in 0..3 {
                    for k in 1..4096 {
                        inv[c][k] = 1.0 / m[c][k];
                    }
                }
                (m, inv)
            }
        };
        let (matrix_64x32, inv_matrix_64x32) = match o64r.as_ref() {
            None => (shared.matrix_64x32.clone(), shared.inv_matrix_64x32.clone()),
            Some(ov) => {
                let m = compute_dct64x32_matrix(Some(ov));
                let mut inv = HeapMatrix::new(0.0f32);
                for c in 0..3 {
                    for k in 1..2048 {
                        inv[c][k] = 1.0 / m[c][k];
                    }
                }
                (m, inv)
            }
        };
        let o32x16 = use_ss2.then(|| scaled_override(&DCT16X32_BANDS, QM_SS2_SCALE16X32));
        let matrix_32x16 = compute_dct32x16_matrix(o32x16.as_ref());
        let mut inv_matrix_32x16 = HeapMatrix::new(0.);
        for c in 0..3 {
            // DC slot zeroed; non-DC LF positions (the 4×2 LLF) left populated
            // since the decoder overwrites them via LowestFrequenciesFromDC.
            for k in 1..512 {
                inv_matrix_32x16[c][k] = 1.0 / matrix_32x16[c][k];
            }
        }

        let matrix_16x16 = compute_dct16x16_matrix(o16.as_ref());
        let mut inv_16x16 = HeapMatrix::new(0.);
        for c in 0..3 {
            // Same convention as inv_matrix and inv_matrix_16x8: DC slot
            // (index 0) is zeroed (handled by DC plane / LF-from-DC). For
            // 16×16 the LLF region is 2×2: positions {0, 1, 16, 17}. We
            // leave non-DC LF positions populated because the decoder
            // will overwrite them via LowestFrequenciesFromDC anyway, just
            // like for 16×8 / 8×16.
            for k in 1..256 {
                inv_16x16[c][k] = 1.0 / matrix_16x16[c][k];
            }
        }

        let matrix_32x32 = compute_dct32x32_matrix(o32.as_ref());
        let mut inv_32x32 = HeapMatrix::new(0.);
        for c in 0..3 {
            // DC slot zeroed; non-DC LF positions (the 4×4 LLF) left populated
            // since the decoder overwrites them via LowestFrequenciesFromDC.
            for k in 1..1024 {
                inv_32x32[c][k] = 1.0 / matrix_32x32[c][k];
            }
        }

        let (matrix, inv_matrix) = if let Some(o8) = o8.as_ref() {
            let matrix = compute_dct8x8_matrix(o8);
            let mut inv_matrix = HeapMatrix::new(0.);
            for c in 0..3 {
                for k in 1..64 {
                    inv_matrix[c][k] = 1.0 / matrix[c][k];
                }
            }
            (matrix, inv_matrix)
        } else {
            (shared.matrix.clone(), shared.inv_matrix.clone())
        };

        Self {
            matrix,
            inv_matrix,
            matrix_16x8: shared.matrix_16x8.clone(),
            inv_matrix_16x8: shared.inv_matrix_16x8.clone(),
            matrix_16x16,
            inv_matrix_16x16: inv_16x16,
            matrix_32x32,
            inv_matrix_32x32: inv_32x32,
            matrix_64x64,
            inv_matrix_64x64,
            matrix_64x32,
            inv_matrix_64x32,
            custom_tables: heap_array_from_fn(|i| match i {
                0 => o8,
                // DCT8X16 and DCT4X8 measured best at the spec defaults, so
                // they remain unsignalled.
                3 | 5 => None,
                1 => o16,
                2 => o32,
                4 => o32x16,
                6 => o64r,
                7 => o64,
                _ => unreachable!(),
            }),
            matrix_4x4: shared.matrix_4x4.clone(),
            inv_matrix_4x4: shared.inv_matrix_4x4.clone(),
            matrix_4x8: shared.matrix_4x8.clone(),
            inv_matrix_4x8: shared.inv_matrix_4x8.clone(),
            matrix_32x16,
            inv_matrix_32x16,
            matrix_afv: shared.matrix_afv.clone(),
            inv_matrix_afv: shared.inv_matrix_afv.clone(),
        }
    }

    #[inline]
    pub(crate) fn matrix(&self, c: usize) -> &[f32; 64] {
        &self.matrix[c]
    }
    #[inline]
    pub(crate) fn inv_matrix(&self, c: usize) -> &[f32; 64] {
        &self.inv_matrix[c]
    }

    /// 16×8 dequant matrix (also used for 8×16 with reinterpreted indexing).
    #[inline]
    pub(crate) fn matrix_16x8(&self, c: usize) -> &[f32; 128] {
        &self.matrix_16x8[c]
    }
    /// 16×8 inverse dequant matrix (used during quantization).
    #[inline]
    pub(crate) fn inv_matrix_16x8(&self, c: usize) -> &[f32; 128] {
        &self.inv_matrix_16x8[c]
    }

    #[inline]
    pub(crate) fn matrix_16x16(&self, c: usize) -> &[f32; 256] {
        &self.matrix_16x16[c]
    }
    #[inline]
    pub(crate) fn inv_matrix_16x16(&self, c: usize) -> &[f32; 256] {
        &self.inv_matrix_16x16[c]
    }

    /// 32×32 dequant matrix (1024 floats per channel).
    #[inline]
    pub(crate) fn matrix_32x32(&self, c: usize) -> &[f32; 1024] {
        &self.matrix_32x32[c]
    }
    #[inline]
    pub(crate) fn inv_matrix_32x32(&self, c: usize) -> &[f32; 1024] {
        &self.inv_matrix_32x32[c]
    }

    #[inline]
    pub(crate) fn matrix_64x64(&self, c: usize) -> &[f32; 4096] {
        &self.matrix_64x64[c]
    }

    #[inline]
    pub(crate) fn inv_matrix_64x64(&self, c: usize) -> &[f32; 4096] {
        &self.inv_matrix_64x64[c]
    }

    #[inline]
    pub(crate) fn matrix_64x32(&self, c: usize) -> &[f32; 2048] {
        &self.matrix_64x32[c]
    }

    #[inline]
    pub(crate) fn inv_matrix_64x32(&self, c: usize) -> &[f32; 2048] {
        &self.inv_matrix_64x32[c]
    }

    /// DCT4X4 dequant matrix (64 floats per channel, 8×8 grid).
    #[inline]
    pub(crate) fn matrix_4x4(&self, c: usize) -> &[f32; 64] {
        &self.matrix_4x4[c]
    }
    /// DCT4X4 inverse dequant matrix (used during quantization).
    #[inline]
    pub(crate) fn inv_matrix_4x4(&self, c: usize) -> &[f32; 64] {
        &self.inv_matrix_4x4[c]
    }
    #[inline]
    pub(crate) fn matrix_4x8(&self, c: usize) -> &[f32; 64] {
        &self.matrix_4x8[c]
    }
    /// DCT4X8 inverse dequant matrix (used during quantization).
    #[inline]
    pub(crate) fn inv_matrix_4x8(&self, c: usize) -> &[f32; 64] {
        &self.inv_matrix_4x8[c]
    }
    /// DCT32X16 / DCT16X32 dequant matrix (shared; 512 floats per channel).
    #[inline]
    pub(crate) fn matrix_32x16(&self, c: usize) -> &[f32; 512] {
        &self.matrix_32x16[c]
    }
    /// DCT32X16 / DCT16X32 inverse dequant matrix (used during quantization).
    #[inline]
    pub(crate) fn inv_matrix_32x16(&self, c: usize) -> &[f32; 512] {
        &self.inv_matrix_32x16[c]
    }
    /// AFV dequant matrix (64 floats per channel, shared by AFV0..AFV3).
    #[inline]
    pub(crate) fn matrix_afv(&self, c: usize) -> &[f32; 64] {
        &self.matrix_afv[c]
    }
    /// AFV inverse dequant matrix (used during quantization).
    #[inline]
    pub(crate) fn inv_matrix_afv(&self, c: usize) -> &[f32; 64] {
        &self.inv_matrix_afv[c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_matrices_are_only_heap_handles() {
        // One handle (plus the custom-table box) per table, nothing inline: the
        // struct must stay small enough to live on a 64 KiB stack.
        let size = std::mem::size_of::<DequantMatrices>();
        // The DCT64 rectangle family adds one matrix/inverse pair (two Box
        // handles), taking the expected footprint from 312 to 328 bytes.
        assert!(size <= 328, "DequantMatrices grew to {size} bytes");
    }

    #[test]
    fn dequant_matrices_construct_on_a_small_stack() {
        std::thread::Builder::new()
            .name("small-stack-dequant-test".into())
            .stack_size(64 * 1024)
            .spawn(|| {
                drop(DequantMatrices::compute(
                    false,
                    false,
                    Some(&FLAT_B8_BANDS_MID),
                ))
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn coarse_dct8_preserves_more_luma_high_frequency() {
        // The DCT8 table is signaled (flat B bands) at every tier above the
        // flat-B quality gate.
        assert!(DequantMatrices::new(QM_DCT8_MIN_DISTANCE - 0.01).custom_tables[0].is_some());
        assert!(DequantMatrices::new(QM_DCT8_MIN_DISTANCE).custom_tables[0].is_some());

        let normal = DequantMatrices::compute(true, false, Some(&FLAT_B8_BANDS_MID));
        let coarse = DequantMatrices::compute(true, true, Some(&FLAT_B8_BANDS_MID));

        let hf_ratio = coarse.inv_matrix(1)[63] / normal.inv_matrix(1)[63];
        assert!((hf_ratio - QM_DCT8_Y_HF_SCALE).abs() < 0.01, "{hf_ratio}");
        let lf_ratio = coarse.inv_matrix(1)[1] / normal.inv_matrix(1)[1];
        assert!((1.0..1.03).contains(&lf_ratio), "{lf_ratio}");
    }

    #[test]
    fn saturated_tables_swap_only_above_the_flat_b_gate() {
        // Near-lossless tier ignores the content gate entirely (the tail pays
        // for any finer B coding, saturated or not).
        assert!(std::ptr::eq(
            DequantMatrices::new_saturated(QM_FLAT_B8_MIN_DISTANCE - 0.01),
            DequantMatrices::new(QM_FLAT_B8_MIN_DISTANCE - 0.01),
        ));
        // Every flat-B tier gets a distinct table set with the saturated B row.
        for d in [0.5, 1.5, 2.5, 4.0] {
            let sat = DequantMatrices::new_saturated(d);
            let base = DequantMatrices::new(d);
            assert!(!std::ptr::eq(sat, base), "d={d}");
            let table = sat.custom_tables[0].expect("saturated flat table");
            assert_eq!(
                table.bands[2][1],
                f16_bits_to_f32(f32_to_f16_bits(SAT_B8_BANDS[1])),
                "d={d}"
            );
        }
    }

    #[test]
    fn flat_b8_gated_off_at_very_high_quality() {
        // Near-lossless tier keeps the spec DCT8 table (nothing signaled);
        // the flat-B override kicks in at the gate.
        assert!(DequantMatrices::new(QM_FLAT_B8_MIN_DISTANCE - 0.01).custom_tables[0].is_none());
        assert!(DequantMatrices::new(QM_FLAT_B8_MIN_DISTANCE).custom_tables[0].is_some());
    }

    #[test]
    fn flat_b8_shape_splits_at_mid_distance() {
        // HQ knee below the split, Optuna mid shape at and above it.
        let below = DequantMatrices::new(QM_FLAT_B8_MID_MIN_DISTANCE - 0.01).custom_tables[0]
            .expect("hq flat table");
        let at = DequantMatrices::new(QM_FLAT_B8_MID_MIN_DISTANCE).custom_tables[0]
            .expect("mid flat table");
        assert_eq!(
            below.bands[2][1],
            f16_bits_to_f32(f32_to_f16_bits(FLAT_B8_BANDS_HQ[1]))
        );
        assert_eq!(
            at.bands[2][1],
            f16_bits_to_f32(f32_to_f16_bits(FLAT_B8_BANDS_MID[1]))
        );
        // X/Y rows identical across the split (only the B row changes).
        for c in [0usize, 1] {
            assert_eq!(below.bands[c][..6], at.bands[c][..6]);
        }
    }

    #[test]
    fn default_dct8_flattens_blue_high_frequency_only() {
        let m = DequantMatrices::compute(false, false, Some(&FLAT_B8_BANDS_HQ));
        // X and Y stay at the spec table within F16 signaling round-off.
        for c in [0usize, 1] {
            for k in 1..64 {
                let spec = DEQUANT_MATRIX_8X8[c][k];
                let got = m.matrix(c)[k];
                assert!(
                    (got - spec).abs() <= 2e-3 * spec.abs(),
                    "c={c} k={k}: {got} vs spec {spec}"
                );
            }
        }
        // B dequant steps never coarser than spec, and several times finer at
        // HF (flat bands raise the outer-band weights, shrinking the step).
        for k in 1..64 {
            let ratio = m.matrix(2)[k] / DEQUANT_MATRIX_8X8[2][k];
            assert!(ratio <= 1.001, "k={k}: B step ratio {ratio} > 1");
        }
        let b_hf = m.matrix(2)[63] / DEQUANT_MATRIX_8X8[2][63];
        assert!(b_hf < 0.35, "B HF step ratio {b_hf}, expected < 0.35");
    }

    #[test]
    fn dct8_matrix_from_bands_matches_static() {
        let identity = BandOverride {
            num_bands: 6,
            bands: {
                let mut b = [[0.0f32; 16]; 3];
                for c in 0..3 {
                    b[c][..6].copy_from_slice(&DCT8_BANDS[c]);
                }
                b
            },
        };
        let computed = compute_dct8x8_matrix(&identity);
        for c in 0..3 {
            for k in 0..64 {
                let expected = DEQUANT_MATRIX_8X8[c][k];
                let got = computed[c][k];
                assert!(
                    (got - expected).abs() <= 2e-6 * expected.abs(),
                    "c={c} k={k}: computed {got}, static {expected}"
                );
            }
        }
    }

    #[test]
    fn dct16x8_matrix_from_bands_matches_static() {
        let identity = BandOverride {
            num_bands: 7,
            bands: {
                let mut b = [[0.0f32; 16]; 3];
                for c in 0..3 {
                    b[c][..7].copy_from_slice(&DCT16X8_BANDS[c]);
                }
                b
            },
        };
        let computed = compute_dct16x8_matrix(Some(&identity));
        for c in 0..3 {
            for k in 0..128 {
                let expected = DEQUANT_MATRIX_16X8[c][k];
                let got = computed[c][k];
                assert!(
                    (got - expected).abs() <= 4e-6 * expected.abs(),
                    "c={c} k={k}: computed {got}, static {expected}"
                );
            }
        }
    }

    #[test]
    fn rect_tables_default_to_the_library_values() {
        let m = DequantMatrices::new(1.0);
        for (c, k) in (0..3).flat_map(|c| (0..128).map(move |k| (c, k))) {
            let expected = DEQUANT_MATRIX_16X8[c][k];
            let got = m.matrix_16x8(c)[k];
            assert!(
                (got - expected).abs() <= 4e-6 * expected.abs(),
                "16x8 c={c} k={k}: {got} vs {expected}"
            );
        }
        // 32x16 was already band-computed, so identity is exact by construction.
        let identity = BandOverride {
            num_bands: 8,
            bands: {
                let mut b = [[0.0f32; 16]; 3];
                for c in 0..3 {
                    b[c][..8].copy_from_slice(&DCT16X32_BANDS[c]);
                }
                b
            },
        };
        let computed = compute_dct32x16_matrix(Some(&identity));
        for c in 0..3 {
            for k in 0..512 {
                assert_eq!(computed[c][k], m.matrix_32x16(c)[k], "32x16 c={c} k={k}");
            }
        }
    }

    #[test]
    fn afv_matrix_fixed_positions_match_library_weights() {
        let m = DequantMatrices::new(1.0);
        for c in 0..3 {
            let mc = m.matrix_afv(c);
            // Sub-part DC tendencies and the fixed 3-pixel-corner weights are
            // stored verbatim in the library table.
            assert_eq!(mc[8], 1.0 / AFV_BANDS[c][0], "c={c} [8]");
            assert_eq!(mc[1], 1.0 / AFV_BANDS[c][1], "c={c} [1]");
            assert_eq!(mc[16], 1.0 / AFV_BANDS[c][2], "c={c} [16]");
            assert_eq!(mc[2], 1.0 / AFV_BANDS[c][3], "c={c} [2]");
            assert_eq!(mc[18], 1.0 / AFV_BANDS[c][4], "c={c} [18]");
            // Odd rows replicate the DCT4X8 weights, even rows / odd columns
            // the DCT4X4 weights.
            for x in 1..8 {
                assert_eq!(mc[8 + x], m.matrix_4x8(c)[x], "c={c} 4x8 col {x}");
            }
            for x in 1..4 {
                assert_eq!(mc[2 * x + 1], m.matrix_4x4(c)[2 * x], "c={c} 4x4 col {x}");
            }
            for (k, &v) in mc.iter().enumerate() {
                assert!(v > 0.0 && v.is_finite(), "c={c} k={k}: {v}");
            }
            // DC slot must not leak into AC quantization.
            assert_eq!(m.inv_matrix_afv(c)[0], 0.0);
        }
    }

    #[test]
    fn dct16x16_matrix_dc_matches_libjxl_polynomial() {
        let m = DequantMatrices::new(1.0);
        // bands[0] = 8996.87 for X → 1/bands[0] ≈ 1.112e-4 at radial position 0
        let x_dc = m.matrix_16x16(0)[0];
        assert!(
            (x_dc - 1.0 / 8996.872_57).abs() < 1e-7,
            "X[0,0]={}, expected ~{}",
            x_dc,
            1.0 / 8996.872_57
        );
        // bands[0] = 3191.48 for Y → 1/bands[0] ≈ 3.133e-4
        let y_dc = m.matrix_16x16(1)[0];
        assert!(
            (y_dc - 1.0 / 3191.483_66).abs() < 1e-7,
            "Y[0,0]={}, expected ~{}",
            y_dc,
            1.0 / 3191.483_66
        );
        // bands[0] = 1157.50 for B → 1/bands[0] ≈ 8.64e-4
        let b_dc = m.matrix_16x16(2)[0];
        assert!(
            (b_dc - 1.0 / 1157.504_08).abs() < 1e-7,
            "B[0,0]={}, expected ~{}",
            b_dc,
            1.0 / 1157.504_08
        );
        // Compare highest-frequency corner with reference (computed by hand /
        // jxl-rs golden): the actual values aren't pub(crate)lished as a static
        // table, but they should all be positive and increase along the
        // radial direction.
        for c in 0..3 {
            let m_c = m.matrix_16x16(c);
            for k in 0..256 {
                assert!(
                    m_c[k] > 0.0,
                    "matrix[{},{}] = {} not positive",
                    c,
                    k,
                    m_c[k]
                );
            }
            assert!(
                m_c[255] > m_c[0],
                "matrix[{}] HF should exceed DC: {} <= {}",
                c,
                m_c[255],
                m_c[0]
            );
        }
    }
}
