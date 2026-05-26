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
pub(crate) const INV_DC_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];
pub(crate) const DC_QUANT: [f32; 3] = [1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0];

/// Per-channel dequant matrices for 8x8 DCT.
/// Indexed [channel][y * 8 + x]. Channel 0 = X, 1 = Y, 2 = B.
pub(crate) const DEQUANT_MATRIX_8X8: [[f32; 64]; 3] = [
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

pub(crate) struct DequantMatrices {
    pub(crate) matrix: [[f32; 64]; 3],
    /// Per-channel inverse matrices (1/weight). Entry [c][0] is zeroed because
    /// DC is quantized separately via DC_QUANT — using these for DC would
    /// silently produce wrong values, so zero it as a safety measure.
    pub(crate) inv_matrix: [[f32; 64]; 3],
}

impl DequantMatrices {
    pub(crate) fn new() -> Self {
        let matrix = DEQUANT_MATRIX_8X8;
        let mut inv = [[0.0f32; 64]; 3];
        for c in 0..3 {
            // DC slot stays 0.0 — AC quantizer should never touch it.
            for k in 1..64 {
                inv[c][k] = 1.0 / matrix[c][k];
            }
        }
        Self {
            matrix,
            inv_matrix: inv,
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
}

impl Default for DequantMatrices {
    fn default() -> Self {
        Self::new()
    }
}
