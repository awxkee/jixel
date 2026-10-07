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
/// The color transform lossy VarDCT frames are coded in.
///
/// `Xyb` converts the image to the perceptual XYB space; the decoder renders
/// it into any output space. The other two keep the samples in the image's own
/// color space, as declared by `color_encoding` or the ICC profile, so any
/// profile round-trips without color management; they cost a few percent at
/// equal quality. Float samples are coded in their own units, so linear-light
/// HDR is better served by `Xyb`. Gray input is coded as three equal
/// channels. CMYK never uses XYB (it would discard the separations) and takes
/// `YCbCr` for `Xyb`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CodingTransform {
    /// Perceptual XYB (the default).
    #[default]
    Xyb,
    /// The samples converted to Cb, Y, Cr with the JPEG matrix, which the
    /// decoder inverts.
    YCbCr,
    /// The samples unchanged: R, G, B (or C, M, Y), with G (M) predicting the
    /// other two.
    Rgb,
}

impl CodingTransform {
    #[inline]
    pub(crate) fn is_xyb(self) -> bool {
        self == Self::Xyb
    }

    /// `base_correlation_x` of the frame's chroma-from-luma: unchanged
    /// channels all track the middle one.
    #[inline]
    pub(crate) fn base_correlation_x(self) -> f32 {
        match self {
            Self::Rgb => 1.0,
            Self::Xyb | Self::YCbCr => 0.0,
        }
    }

    /// `base_correlation_b`: XYB's B tracks Y, the YCbCr chroma planes do
    /// not, and unchanged channels track the middle one.
    #[inline]
    pub(crate) fn base_correlation_b(self) -> f32 {
        match self {
            Self::Xyb | Self::Rgb => 1.0,
            Self::YCbCr => 0.0,
        }
    }

    /// Multipliers of the default (XYB) DC steps. The non-XYB planes all
    /// span about [0, 1]: the middle one takes the luma step `1/512`, the
    /// outer two coarser ones.
    #[inline]
    pub(crate) fn dc_step_base(self) -> [f32; 3] {
        match self {
            Self::Xyb => [1.0; 3],
            Self::YCbCr => [8.0 * 2.07, 1.0, 0.5 * 3.27],
            Self::Rgb => [8.0 * 2.32, 1.0, 0.5 * 3.68],
        }
    }

    /// Base-weight multipliers (larger = finer) applied to the luma table
    /// rows of each plane.
    #[inline]
    pub(crate) fn table_scale(self) -> [f32; 3] {
        match self {
            Self::Xyb => [1.0; 3],
            Self::YCbCr => [0.321, 1.0, 0.487],
            Self::Rgb => [0.486, 1.0, 0.271],
        }
    }

    /// Precision multiplier of each plane's outermost table band (larger =
    /// finer high frequencies).
    #[inline]
    pub(crate) fn table_hf_tilt(self) -> [f32; 3] {
        match self {
            Self::Xyb => [1.0; 3],
            Self::YCbCr => [0.768, 0.609, 1.561],
            Self::Rgb => [0.546, 1.542, 0.766],
        }
    }

    /// Per-plane distortion weights of the RD searches.
    #[inline]
    pub(crate) fn channel_weights(self) -> [f32; 3] {
        match self {
            Self::Xyb => [0.30, 1.0, 0.28],
            Self::YCbCr => [0.125, 1.0, 0.255],
            Self::Rgb => [1.1, 1.0, 0.159],
        }
    }

    /// Default EPF channel scales; `None` keeps the spec XYB ones. The non-XYB
    /// planes all span luma's range.
    #[inline]
    pub(crate) fn epf_channel_scale(self) -> Option<[f32; 3]> {
        match self {
            Self::Xyb => None,
            Self::YCbCr | Self::Rgb => Some([5.0; 3]),
        }
    }

    /// The coded planes of one sample from gamma-encoded [0, 1] channels.
    #[inline]
    pub(crate) fn code(self, r: f32, g: f32, b: f32) -> [f32; 3] {
        match self {
            Self::YCbCr => ycbcr_triple(r, g, b),
            Self::Rgb => [r, g, b],
            Self::Xyb => unreachable!("XYB planes come from the opsin transform"),
        }
    }
}

/// Least-squares slope of each outer plane's local detail against the middle
/// plane's: the frame's chroma-from-luma base correlations `[x, b]`, which the
/// per-tile maps refine within +-1.5. Saturated content can track luma well
/// past that window (red detail in Cr runs at ~1.7x luma).
pub(crate) fn fit_cfl_bases(coded: &crate::image::Image3F) -> [f32; 2] {
    const LIMIT: f32 = 4.0;
    let (w, h) = (coded.xsize(), coded.ysize());
    let (mut yy, mut xy, mut by) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        let rows = [0, 1, 2].map(|c| coded.plane_row(c, y));
        let below = (y + 1 < h).then(|| [0, 1, 2].map(|c| coded.plane_row(c, y + 1)));
        for x in 0..w {
            let mut add = |d: [f32; 3]| {
                let dy = f64::from(d[1]);
                yy += dy * dy;
                xy += f64::from(d[0]) * dy;
                by += f64::from(d[2]) * dy;
            };
            if x + 1 < w {
                add([0, 1, 2].map(|c| rows[c][x + 1] - rows[c][x]));
            }
            if let Some(below) = below {
                add([0, 1, 2].map(|c| below[c][x] - rows[c][x]));
            }
        }
    }
    if yy <= 0.0 {
        return [0.0; 2];
    }
    [xy / yy, by / yy].map(|slope| {
        let slope = (slope as f32).clamp(-LIMIT, LIMIT);
        crate::util::f16_bits_to_f32(crate::util::f32_to_f16_bits(slope))
    })
}

/// The 8-bit JPEG matrix, as the decoder inverts it (libjxl `stage_ycbcr`).
const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;

/// One YCbCr sample triple (Cb, Y − 128/255, Cr) from gamma-encoded [0, 1]
/// channel values.
#[inline]
fn ycbcr_triple(r: f32, g: f32, b: f32) -> [f32; 3] {
    let luma = KR * r + KG * g + KB * b;
    [
        (b - luma) / (2.0 * (1.0 - KB)),
        luma - 128.0 / 255.0,
        (r - luma) / (2.0 * (1.0 - KR)),
    ]
}

/// sRGB transfer, decoding direction.
#[inline]
pub(crate) fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodeConfig;

    /// The decoder's inverse of `ycbcr_triple` (libjxl `stage_ycbcr`).
    fn decode_ycbcr([cb, y, cr]: [f32; 3]) -> [f32; 3] {
        let y = y + 128.0 / 255.0;
        [
            y + 1.402 * cr,
            y - 0.114 * 1.772 / 0.587 * cb - 0.299 * 1.402 / 0.587 * cr,
            y + 1.772 * cb,
        ]
    }

    #[test]
    fn ycbcr_round_trips_through_the_decoder_matrix() {
        for &(r, g, b) in &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.9, 0.2, 0.4),
            (0.1, 0.7, 0.95),
        ] {
            let got = decode_ycbcr(CodingTransform::YCbCr.code(r, g, b));
            for (a, e) in got.iter().zip([r, g, b]) {
                assert!((a - e).abs() < 1e-5, "{got:?} vs {:?}", [r, g, b]);
            }
        }
        assert_eq!(
            CodingTransform::Rgb.code(0.25, 0.5, 0.75),
            [0.25, 0.5, 0.75]
        );
    }

    /// Every lossy entry point (integer, float, gray) codes all three
    /// transforms, and the non-XYB ones are real alternatives rather than a
    /// silent XYB fallback.
    #[test]
    fn rgb_entry_points_take_every_transform() {
        let (w, h) = (300, 260);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 7) % 251) as u8).collect();
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| ((i * 5) % 253) as u8).collect();
        let rgb16: Vec<u16> = rgb.iter().map(|&v| u16::from(v) << 8).collect();
        // Float input past [0, 1] exercises the DC range guard.
        let rgbf: Vec<f32> = rgb.iter().map(|&v| f32::from(v) / 25.0).collect();
        let gray: Vec<u8> = rgb.iter().step_by(3).copied().collect();
        let gray16: Vec<u16> = gray.iter().map(|&v| u16::from(v) * 257).collect();
        let grayf: Vec<f32> = gray.iter().map(|&v| f32::from(v) / 255.0).collect();
        for speed in [crate::Speed::Fast, crate::Speed::Slow] {
            let mut outputs = Vec::new();
            for transform in [
                CodingTransform::Xyb,
                CodingTransform::YCbCr,
                CodingTransform::Rgb,
            ] {
                let config = EncodeConfig::default()
                    .with_speed(speed)
                    .with_distance(1.5)
                    .with_coding_transform(transform);
                let a = crate::encode_image(&rgb, w, h, &config).unwrap();
                crate::encode_image_with_alpha(&rgba, w, h, &config).unwrap();
                crate::encode_image_16bit(&rgb16, w, h, &config).unwrap();
                crate::encode_image_f32(&rgbf, w, h, &config).unwrap();
                crate::encode_image_gray(&gray, w, h, &config).unwrap();
                crate::encode_image_gray_16bit(&gray16, w, h, &config).unwrap();
                crate::encode_image_gray_f32(&grayf, w, h, &config).unwrap();
                outputs.push(a);
            }
            assert_ne!(outputs[0], outputs[1]);
            assert_ne!(outputs[0], outputs[2]);
            assert_ne!(outputs[1], outputs[2]);
        }
    }
}
