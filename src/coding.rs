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

    /// Per-image multipliers of `table_scale`. The tables are fitted on
    /// images whose colour G (or luma) predicts, but where red or blue
    /// dominates, that colour carries the visible structure: in RGB on its own
    /// coarser plane, in YCbCr on a luma that sees a fraction of it. Such
    /// images take finer tables there.
    pub(crate) fn plane_scale(self, coded: &crate::image::Image3F) -> [f32; 3] {
        // Full-strength multipliers when red / blue dominates, and the
        // dominant share of the gradient energy at which they fade in and
        // saturate.
        let (red_full, blue_full, start, full) = match self {
            Self::Xyb => return [1.0; 3],
            Self::Rgb => ([1.375, 1.0, 1.0], [1.0, 1.0, 2.0], 0.04, 0.20),
            Self::YCbCr => ([1.375, 2.0, 1.375], [1.375, 2.8125, 1.0], 0.30, 0.80),
        };
        let [red, blue] = dominant_gradient_shares(coded, |p| self.decode(p))
            .map(|share: f32| ((share - start) / (full - start)).clamp(0.0, 1.0));
        std::array::from_fn(|c| {
            let m = (1.0 + (red_full[c] - 1.0) * red).max(1.0 + (blue_full[c] - 1.0) * blue);
            // Sixteenth steps keep the table cache small.
            (16.0 * m).round() / 16.0
        })
    }

    /// Gamma-encoded R, G, B of one coded triple (the decoder's inverse of
    /// [`Self::code`]).
    #[inline]
    fn decode(self, [p0, p1, p2]: [f32; 3]) -> [f32; 3] {
        match self {
            Self::YCbCr => {
                let y = p1 + 128.0 / 255.0;
                [
                    y + 2.0 * (1.0 - KR) * p2,
                    y - 2.0 * (1.0 - KB) * KB / KG * p0 - 2.0 * (1.0 - KR) * KR / KG * p2,
                    y + 2.0 * (1.0 - KB) * p0,
                ]
            }
            Self::Rgb | Self::Xyb => [p0, p1, p2],
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

/// The chroma-from-luma parameters of a non-XYB frame. Each outer plane's
/// base is the least-squares slope of its local detail against the middle
/// plane's; the per-tile maps refine it within about +-1.5. Saturated content
/// tracks luma far from the transform's default (red detail in Cr runs at
/// ~1.7x luma), and when a mix of such content leaves many textured tiles out
/// of the maps' reach, the YCbCr base moves to bring the most of them back.
/// (RGB tiles over a flat G have no meaningful slope to reach.)
pub(crate) fn fit_cfl_frame(
    coding: CodingTransform,
    coded: &crate::image::Image3F,
) -> crate::color_correlation::CflFrame {
    const LIMIT: f32 = 4.0;
    // CfL tile side and the reach of a tile's map around the base, short of
    // its +-127/84 so tiles at the edge keep some room.
    const TILE: usize = 64;
    const REACH: f32 = 1.4;
    // Tiles with at least this share of the mean luma detail have a
    // meaningful slope; the base moves once this share of them is out of reach.
    const TEXTURED: f64 = 0.05;
    const OUT_OF_REACH: f32 = 0.1;
    // Candidate base step.
    const STEP: f32 = 1.0 / 64.0;
    let (w, h) = (coded.xsize(), coded.ysize());
    let tiles_x = w.div_ceil(TILE);
    // Luma detail energy and its products with the outer planes, over the
    // whole image and per tile (differences within the tile only).
    let mut all = [0.0f64; 3];
    let mut tiles = vec![[0.0f64; 3]; tiles_x * h.div_ceil(TILE)];
    for y in 0..h {
        let rows = [0, 1, 2].map(|c| coded.plane_row(c, y));
        let below = (y + 1 < h).then(|| [0, 1, 2].map(|c| coded.plane_row(c, y + 1)));
        let tile_row = &mut tiles[(y / TILE) * tiles_x..][..tiles_x];
        for x in 0..w {
            let tile = &mut tile_row[x / TILE];
            let mut add = |d: [f32; 3], within_tile: bool| {
                let dy = f64::from(d[1]);
                let terms = [dy * dy, f64::from(d[0]) * dy, f64::from(d[2]) * dy];
                for i in 0..3 {
                    all[i] += terms[i];
                    if within_tile {
                        tile[i] += terms[i];
                    }
                }
            };
            if x + 1 < w {
                add(
                    [0, 1, 2].map(|c| rows[c][x + 1] - rows[c][x]),
                    (x + 1) % TILE != 0,
                );
            }
            if let Some(below) = below {
                add(
                    [0, 1, 2].map(|c| below[c][x] - rows[c][x]),
                    (y + 1) % TILE != 0,
                );
            }
        }
    }
    let yy = all[0];
    let textured = TEXTURED * tiles.iter().map(|t| t[0]).sum::<f64>() / tiles.len().max(1) as f64;
    let fit = |plane: usize| -> f32 {
        if yy <= 0.0 {
            return 0.0;
        }
        let least_squares = ((all[plane] / yy) as f32).clamp(-LIMIT, LIMIT);
        let slopes: Vec<f32> = tiles
            .iter()
            .filter(|t| t[0] > textured)
            .map(|t| (t[plane] / t[0]) as f32)
            .collect();
        let out_of_reach = |base: f32| slopes.iter().filter(|&&s| (s - base).abs() > REACH).count();
        let at_least_squares = out_of_reach(least_squares);
        if coding != CodingTransform::YCbCr
            || (at_least_squares as f32) <= OUT_OF_REACH * slopes.len() as f32
        {
            return least_squares;
        }
        let mut best = (at_least_squares, 0.0f32, least_squares);
        for i in 0..=(2.0 * LIMIT / STEP) as i32 {
            let base = -LIMIT + i as f32 * STEP;
            let key = (out_of_reach(base), (base - least_squares).abs());
            if (key.0, key.1) < (best.0, best.1) {
                best = (key.0, key.1, base);
            }
        }
        best.2
    };
    let [base_x, base_b] = [fit(1), fit(2)]
        .map(|slope| crate::util::f16_bits_to_f32(crate::util::f32_to_f16_bits(slope)));
    crate::color_correlation::CflFrame {
        base_x,
        base_b,
        color_factor: crate::color_correlation::K_COLOR_FACTOR,
    }
}

/// Shares of the image's gradient energy (all of R, G, B; horizontal and
/// vertical neighbour differences) carried by R and by B where that channel
/// dominates the pixel: above 0.15 and over 1.6x both others. `decode` maps
/// coded triples to R, G, B.
fn dominant_gradient_shares(
    coded: &crate::image::Image3F,
    decode: impl Fn([f32; 3]) -> [f32; 3],
) -> [f32; 2] {
    const FLOOR: f32 = 0.15;
    const RATIO: f32 = 1.6;
    let (w, h) = (coded.xsize(), coded.ysize());
    let decode_row = |y: usize, out: &mut Vec<[f32; 3]>| {
        let rows = [0, 1, 2].map(|c| coded.plane_row(c, y));
        out.clear();
        out.extend((0..w).map(|x| decode([rows[0][x], rows[1][x], rows[2][x]])));
    };
    let (mut row, mut below) = (Vec::with_capacity(w), Vec::with_capacity(w));
    if h > 0 {
        decode_row(0, &mut row);
    }
    let (mut total, mut dominant) = (0.0f64, [0.0f64; 2]);
    for y in 0..h {
        if y + 1 < h {
            decode_row(y + 1, &mut below);
        }
        for x in 0..w {
            let px = row[x];
            let mut energy = [0.0f32; 3];
            for c in 0..3 {
                if x + 1 < w {
                    let d = row[x + 1][c] - px[c];
                    energy[c] += d * d;
                }
                if y + 1 < h {
                    let d = below[x][c] - px[c];
                    energy[c] += d * d;
                }
            }
            total += f64::from(energy[0] + energy[1] + energy[2]);
            let [r, g, b] = px;
            if r > FLOOR && r > RATIO * g.max(b) {
                dominant[0] += f64::from(energy[0]);
            }
            if b > FLOOR && b > RATIO * r.max(g) {
                dominant[1] += f64::from(energy[2]);
            }
        }
        std::mem::swap(&mut row, &mut below);
    }
    if total > 0.0 {
        dominant.map(|e| (e / total) as f32)
    } else {
        [0.0; 2]
    }
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
    fn dominant_colour_refines_its_plane() {
        // A textured channel over two flat ones, coded with `coding`.
        let textured = |coding: CodingTransform, dominant: usize| {
            let mut img = crate::image::Image3F::new(32, 32);
            for y in 0..32 {
                let [p0, p1, p2] = img.all_plane_rows_mut(y);
                for x in 0..32 {
                    let rgb: [f32; 3] = std::array::from_fn(|c| {
                        if c == dominant {
                            0.5 + 0.4 * (((x + y) % 4) as f32 / 3.0 - 0.5)
                        } else {
                            0.05
                        }
                    });
                    [p0[x], p1[x], p2[x]] = coding.code(rgb[0], rgb[1], rgb[2]);
                }
            }
            img
        };
        let scale = |coding: CodingTransform, dominant: usize| {
            coding.plane_scale(&textured(coding, dominant))
        };
        assert_eq!(scale(CodingTransform::Rgb, 0), [1.375, 1.0, 1.0]);
        assert_eq!(scale(CodingTransform::Rgb, 2), [1.0, 1.0, 2.0]);
        assert_eq!(scale(CodingTransform::Rgb, 1), [1.0; 3]);
        assert_eq!(scale(CodingTransform::YCbCr, 0), [1.375, 2.0, 1.375]);
        assert_eq!(scale(CodingTransform::YCbCr, 2), [1.375, 2.8125, 1.0]);
        assert_eq!(scale(CodingTransform::YCbCr, 1), [1.0; 3]);
    }

    #[test]
    fn cfl_base_reaches_mixed_hue_tiles() {
        // One CfL tile of red detail (Cr at ~1.67x luma) beside one of green
        // detail (Cr at ~-0.71x luma): the least-squares base leaves the red
        // tile out of the maps' reach, so the base moves between the two.
        let coding = CodingTransform::YCbCr;
        let mut img = crate::image::Image3F::new(128, 64);
        for y in 0..64 {
            let [p0, p1, p2] = img.all_plane_rows_mut(y);
            for x in 0..128 {
                let t = 0.5 + 0.3 * (((x * 7 + y * 3) % 5) as f32 / 4.0 - 0.5);
                let [r, g, b] = if x < 64 { [t, 0.0, 0.0] } else { [0.2, t, 0.2] };
                [p0[x], p1[x], p2[x]] = coding.code(r, g, b);
            }
        }
        let base = fit_cfl_frame(coding, &img).base_b;
        assert!(
            (base - 1.67).abs() <= 1.4 && (base + 0.71).abs() <= 1.4,
            "{base}"
        );

        // A single hue keeps the least-squares slope.
        let red = |x: usize, y: usize| 0.5 + 0.3 * (((x * 7 + y * 3) % 5) as f32 / 4.0 - 0.5);
        for y in 0..64 {
            let [p0, p1, p2] = img.all_plane_rows_mut(y);
            for x in 0..128 {
                [p0[x], p1[x], p2[x]] = coding.code(red(x, y), 0.0, 0.0);
            }
        }
        let base = fit_cfl_frame(coding, &img).base_b;
        assert!((base - 0.5 / 0.299).abs() < 0.02, "{base}");
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
