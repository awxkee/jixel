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
    pub(crate) matrix: [[f32; 64]; 3],
    /// Per-channel inverse matrices (1/weight). Entry [c][0] is zeroed because
    /// DC is quantized separately via DC_QUANT.
    pub(crate) inv_matrix: [[f32; 64]; 3],
    /// 16×8 / 8×16 dequant matrix. Both rectangular transforms share these
    /// 128 floats per channel (libjxl-tiny convention).
    pub(crate) matrix_16x8: [[f32; 128]; 3],
    pub(crate) inv_matrix_16x8: [[f32; 128]; 3],
    /// 16×16 dequant matrix (256 floats per channel). Generated at
    /// construction time from the libjxl polynomial parameters since it isn't
    /// part of libjxl-tiny.
    pub(crate) matrix_16x16: [[f32; 256]; 3],
    pub(crate) inv_matrix_16x16: [[f32; 256]; 3],
    /// 32×32 dequant matrix (1024 floats per channel). Generated at
    /// construction time from the libjxl DCT32X32 polynomial parameters.
    pub(crate) matrix_32x32: Box<[[f32; 1024]; 3]>,
    pub(crate) inv_matrix_32x32: Box<[[f32; 1024]; 3]>,
    /// 64x64 dequant matrix, used only by the slow large-transform path.
    pub(crate) matrix_64x64: Box<[[f32; 4096]; 3]>,
    pub(crate) inv_matrix_64x64: Box<[[f32; 4096]; 3]>,
    /// Shared normalized 32-row × 64-column table for DCT64X32/DCT32X64.
    pub(crate) matrix_64x32: Box<[[f32; 2048]; 3]>,
    pub(crate) inv_matrix_64x32: Box<[[f32; 2048]; 3]>,
    /// DCT4X4 dequant matrix (64 floats per channel, 8×8 grid). Generated from
    /// the libjxl DCT4X4 4-band parameters: 4×4 radial weights replicated to
    /// 2×2 cells. Used for the sub-8×8 DCT4X4 transform.
    pub(crate) matrix_4x4: [[f32; 64]; 3],
    pub(crate) inv_matrix_4x4: [[f32; 64]; 3],
    /// DCT4X8 dequant matrix (64 floats per channel). 4×8 radial weights with
    /// each row replicated to 2 rows of the 8×8 grid. Used for DCT4X8.
    pub(crate) matrix_4x8: [[f32; 64]; 3],
    pub(crate) inv_matrix_4x8: [[f32; 64]; 3],
    /// DCT32X16 / DCT16X32 dequant matrix (512 floats per channel). Both
    /// rectangular large transforms share these weights (libjxl
    /// `QuantTable::DCT16X32`), computed at the normalized 16-row × 32-col
    /// resolution so the same table applies to both orientations.
    pub(crate) matrix_32x16: Box<[[f32; 512]; 3]>,
    pub(crate) inv_matrix_32x16: Box<[[f32; 512]; 3]>,
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

/// JPEG XL default DCT64X64 quant-weight parameters.
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

/// JPEG XL default `QuantTable::DCT32X64` parameters, shared by both orientations.
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

/// libjxl band-step multiplicative helper. Positive → 1+v, negative → 1/(1-v).
/// Matches `jxl::DequantMatricesLibrary::Mult` and jxl-rs `mult`.
#[inline]
fn band_mult(v: f32) -> f32 {
    if v > 0.0 { 1.0 + v } else { 1.0 / (1.0 - v) }
}

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
fn compute_dct4x8_matrix() -> [[f32; 64]; 3] {
    const NUM_BANDS: usize = 4;
    let mut out = [[0.0f32; 64]; 3];
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = DCT4X8_BANDS[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT4X8_BANDS[c][i]);
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

fn compute_dct4x4_matrix() -> [[f32; 64]; 3] {
    const NUM_BANDS: usize = 4;
    let mut out = [[0.0f32; 64]; 3];
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

/// Returns the inverse weights (so caller's matrix entry = 1/band-interp,
/// matching the layout of `DEQUANT_MATRIX_8X8` etc).
fn compute_dct16x16_matrix() -> [[f32; 256]; 3] {
    const NUM_BANDS: usize = 7;
    let mut out = [[0.0f32; 256]; 3];
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = DCT16X16_BANDS[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT16X16_BANDS[c][i]);
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

fn compute_dct32x32_matrix() -> Box<[[f32; 1024]; 3]> {
    const NUM_BANDS: usize = 8;
    let mut out = Box::new([[0.0f32; 1024]; 3]);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = DCT32X32_BANDS[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT32X32_BANDS[c][i]);
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

fn compute_dct64x64_matrix() -> Box<[[f32; 4096]; 3]> {
    const NUM_BANDS: usize = 8;
    let mut out = Box::new([[0.0f32; 4096]; 3]);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = DCT64X64_BANDS[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT64X64_BANDS[c][i]);
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

fn compute_dct64x32_matrix() -> Box<[[f32; 2048]; 3]> {
    const NUM_BANDS: usize = 8;
    let mut out = Box::new([[0.0f32; 2048]; 3]);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = DCT32X64_BANDS[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT32X64_BANDS[c][i]);
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

fn compute_dct32x16_matrix() -> Box<[[f32; 512]; 3]> {
    const NUM_BANDS: usize = 8;
    let mut out = Box::new([[0.0f32; 512]; 3]);
    for c in 0..3 {
        let mut bands = [0.0f32; NUM_BANDS];
        bands[0] = DCT16X32_BANDS[c][0];
        for i in 1..NUM_BANDS {
            bands[i] = bands[i - 1] * band_mult(DCT16X32_BANDS[c][i]);
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

impl DequantMatrices {
    pub(crate) fn new() -> Self {
        let matrix = DEQUANT_MATRIX_8X8;
        let mut inv = [[0.0f32; 64]; 3];
        for c in 0..3 {
            for k in 1..64 {
                inv[c][k] = 1.0 / matrix[c][k];
            }
        }

        let matrix_16x8 = DEQUANT_MATRIX_16X8;
        let mut inv_16x8 = [[0.0f32; 128]; 3];
        for c in 0..3 {
            for k in 1..128 {
                inv_16x8[c][k] = 1.0 / matrix_16x8[c][k];
            }
        }

        let matrix_16x16 = compute_dct16x16_matrix();
        let mut inv_16x16 = [[0.0f32; 256]; 3];
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

        let matrix_32x32 = compute_dct32x32_matrix();
        let mut inv_32x32 = Box::new([[0.0f32; 1024]; 3]);
        for c in 0..3 {
            // DC slot zeroed; non-DC LF positions (the 4×4 LLF) left populated
            // since the decoder overwrites them via LowestFrequenciesFromDC.
            for k in 1..1024 {
                inv_32x32[c][k] = 1.0 / matrix_32x32[c][k];
            }
        }

        let matrix_64x64 = compute_dct64x64_matrix();
        let mut inv_64x64 = Box::new([[0.0f32; 4096]; 3]);
        for c in 0..3 {
            for k in 1..4096 {
                inv_64x64[c][k] = 1.0 / matrix_64x64[c][k];
            }
        }

        let matrix_64x32 = compute_dct64x32_matrix();
        let mut inv_64x32 = Box::new([[0.0f32; 2048]; 3]);
        for c in 0..3 {
            for k in 1..2048 {
                inv_64x32[c][k] = 1.0 / matrix_64x32[c][k];
            }
        }

        let matrix_4x4 = compute_dct4x4_matrix();
        let mut inv_4x4 = [[0.0f32; 64]; 3];
        for c in 0..3 {
            // DC slot (index 0) zeroed (handled by the DC plane). For DCT4X4 the
            // only LLF position is the DC; [1], [8], [9] are regular AC.
            for k in 1..64 {
                inv_4x4[c][k] = 1.0 / matrix_4x4[c][k];
            }
        }

        let matrix_4x8 = compute_dct4x8_matrix();
        let mut inv_4x8 = [[0.0f32; 64]; 3];
        for c in 0..3 {
            // Only [0] is the DC (handled by the DC plane); [8] (the vertical
            // half-difference after the Hadamard) and all others are regular AC.
            for k in 1..64 {
                inv_4x8[c][k] = 1.0 / matrix_4x8[c][k];
            }
        }

        let matrix_32x16 = compute_dct32x16_matrix();
        let mut inv_32x16 = Box::new([[0.0f32; 512]; 3]);
        for c in 0..3 {
            // DC slot zeroed; non-DC LF positions (the 4×2 LLF) left populated
            // since the decoder overwrites them via LowestFrequenciesFromDC.
            for k in 1..512 {
                inv_32x16[c][k] = 1.0 / matrix_32x16[c][k];
            }
        }

        Self {
            matrix,
            inv_matrix: inv,
            matrix_16x8,
            inv_matrix_16x8: inv_16x8,
            matrix_16x16,
            inv_matrix_16x16: inv_16x16,
            matrix_32x32,
            inv_matrix_32x32: inv_32x32,
            matrix_64x64,
            inv_matrix_64x64: inv_64x64,
            matrix_64x32,
            inv_matrix_64x32: inv_64x32,
            matrix_4x4,
            inv_matrix_4x4: inv_4x4,
            matrix_4x8,
            inv_matrix_4x8: inv_4x8,
            matrix_32x16,
            inv_matrix_32x16: inv_32x16,
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

    pub(crate) fn matrix_64x32(&self, c: usize) -> &[f32; 2048] {
        &self.matrix_64x32[c]
    }

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
}

impl Default for DequantMatrices {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test_16x16 {
    use super::*;
    #[test]
    fn dct16x16_matrix_dc_matches_libjxl_polynomial() {
        let m = DequantMatrices::new();
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
