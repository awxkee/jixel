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
use crate::bit_writer::BitWriter;
use crate::color_encoding::{ColorEncoding, write_color_encoding};
use crate::enc_frame::encode_frame;
use crate::image::Image3F;
use crate::srgb_to_linear_u8;

/// 8-bit alpha plane (row-major, stride = `xsize`).
pub type AlphaPlane = Vec<u8>;

// -----------------------------------------------------------------------------
// Codestream constants.
// -----------------------------------------------------------------------------

/// Codestream marker byte that follows the leading 0xFF. Identifies this as
/// a raw JXL codestream (vs an ISOBMFF-wrapped one).
const CODESTREAM_MARKER: u8 = 0x0A;

/// Distances below this give larger files than lossless on photographic
/// content; we clamp up to this value.
const MIN_DISTANCE: f32 = 0.03;

/// JXL's image dimension field encodes (size - 1) in either 9, 13, 18, or
/// 30 bits, so 2^30 is the largest representable dimension.
const MAX_DIMENSION: usize = 0x3FFF_FFFF;

// -----------------------------------------------------------------------------
// Encode configuration.
// -----------------------------------------------------------------------------

/// Configuration for a single encode call.
///
/// Defaults are tuned to match the legacy `encode_file(image, distance)`
/// behaviour: linear-RGB float input, sRGB primaries, linear transfer, D65.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// Butteraugli target distance. Smaller = higher quality, larger files.
    /// Clamped to a minimum of 0.03; lossless is not supported.
    pub distance: f32,

    /// Color encoding signalled in the codestream.
    pub color_encoding: ColorEncoding,

    /// Optional embedded ICC profile bytes.
    ///
    /// **Not yet implemented**: the JXL ICC stream uses a custom compression
    /// scheme (41-context arithmetic coding + tag-list commands) that hasn't
    /// been ported. Setting this currently panics on encode. The field is
    /// retained so the API doesn't need to change once support lands.
    pub icc_profile: Option<Vec<u8>>,

    /// Optional 8-bit alpha plane (row-major, stride = `image.xsize()`,
    /// length = `image.xsize() * image.ysize()`). When set, the encoded
    /// codestream declares one extra channel (Alpha, U8, unassociated) and
    /// the plane is stored losslessly via a minimum-viable Modular sub-
    /// bitstream (single-leaf MA tree, Gradient predictor).
    pub alpha: Option<AlphaPlane>,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            distance: 1.0,
            color_encoding: ColorEncoding::default(),
            icc_profile: None,
            alpha: None,
        }
    }
}

impl EncodeConfig {
    /// Convenience builder with the given butteraugli distance and otherwise
    /// default settings (sRGB primaries, linear transfer).
    pub fn with_distance(distance: f32) -> Self {
        Self {
            distance,
            ..Self::default()
        }
    }

    /// Convenience builder with quality on a libjpeg-like 0..=100 scale.
    /// See [`distance_from_quality`] for the mapping.
    pub fn with_quality(quality: f32) -> Self {
        Self::with_distance(distance_from_quality(quality))
    }

    /// Replace the color encoding (white point / primaries / transfer / intent).
    pub fn with_color_encoding(mut self, enc: ColorEncoding) -> Self {
        self.color_encoding = enc;
        self
    }

    /// Attach an ICC profile. **Panics at encode time** — see field docs.
    pub fn with_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.icc_profile = Some(icc);
        self
    }

    /// Attach an 8-bit alpha plane to be encoded losslessly via Modular.
    /// Length must equal `xsize * ysize` of the image passed to encode.
    pub fn with_alpha(mut self, alpha: AlphaPlane) -> Self {
        self.alpha = Some(alpha);
        self
    }
}

/// Convert a JPEG-style quality value (0..=100, higher = better) to a
/// butteraugli distance (smaller = better).
///
/// The mapping matches libjxl's `JxlEncoderDistanceFromQuality`:
///   * `quality == 100` → distance ≈ 0.1
///   * `quality == 90`  → distance == 1.0 (the published "visually lossless" point)
///   * `quality == 30`  → distance == 6.4 (knee of the curve)
///   * `quality == 0`   → distance == 25.0 (clamp ceiling)
///
/// For `quality >= 30` the mapping is linear: `distance = 0.1 + (100 - q) * 0.09`.
/// For `quality < 30` it transitions to a faster ramp using `2.5^((30 - q) / 5)`.
/// Out-of-range inputs are clamped (`quality > 100` becomes 100, `quality < 0`
/// becomes 0 by way of the 25.0 clamp).
///
/// `NaN` is rejected.
pub fn distance_from_quality(quality: f32) -> f32 {
    assert!(!quality.is_nan(), "quality must not be NaN");
    // Clamp at 100 from above (lossless isn't supported anyway; you'll
    // hit the MIN_DISTANCE floor below 0.03).
    let q = quality.min(100.0);
    let d = if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        6.24 + 2.5f32.powf((30.0 - q) / 5.0) / 6.25
    };
    d.min(25.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn quality_mapping_anchor_points() {
        // Matches the published libjxl mapping breakpoints.
        assert!(close(distance_from_quality(100.0), 0.1));
        assert!(close(distance_from_quality(90.0), 1.0));
        assert!(close(distance_from_quality(30.0), 6.4));
        // Below 30, the formula is quadratic-ish; q=25 must come in just above 6.4.
        let d25 = distance_from_quality(25.0);
        assert!(d25 > 6.4 && d25 < 7.0, "q=25 -> {d25}");
        // Far below 30, it climbs and clamps at 25.
        assert!(close(distance_from_quality(0.0), 25.0));
        assert!(close(distance_from_quality(-50.0), 25.0));
        // Above 100, clamped: 110 -> 100 -> 0.1.
        assert!(close(distance_from_quality(110.0), 0.1));
    }

    #[test]
    fn quality_mapping_monotone() {
        // Strictly decreasing in quality.
        let mut prev = distance_from_quality(0.0);
        for q in 1..=100 {
            let d = distance_from_quality(q as f32);
            assert!(d <= prev, "non-monotonic at q={q}: prev={prev}, d={d}");
            prev = d;
        }
    }

    #[test]
    #[should_panic(expected = "quality must not be NaN")]
    fn quality_nan_panics() {
        let _ = distance_from_quality(f32::NAN);
    }
}

/// Encode a linear-light RGB `Image3F` at the given butteraugli distance,
/// using the default color encoding (sRGB primaries, linear transfer).
///
/// Shorthand for [`encode_with_config`] with default settings.
pub fn encode_image(input: &[u8], width: usize, height: usize, distance: f32) -> Vec<u8> {
    assert!(width > 0 && height > 0, "empty image");
    assert_eq!(
        input.len(),
        width * height * 3,
        "input buffer size mismatch"
    );
    let distance = distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);
    for (y, row) in input.chunks_exact(width * 3).enumerate() {
        let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
        for (((r, g), b), src) in r_row
            .iter_mut()
            .zip(g_row.iter_mut())
            .zip(b_row.iter_mut())
            .zip(row.as_chunks::<3>().0.iter())
        {
            *r = srgb_to_linear_u8(src[0]);
            *g = srgb_to_linear_u8(src[1]);
            *b = srgb_to_linear_u8(src[2]);
        }
    }
    encode_with_config(&linear, &EncodeConfig::with_distance(distance))
}

/// Encode a linear-light RGB `Image3F` at the given butteraugli distance,
/// using the default color encoding (sRGB primaries, linear transfer).
///
/// Shorthand for [`encode_with_config`] with default settings.
pub fn encode_image_with_alpha(
    input: &[u8],
    width: usize,
    height: usize,
    distance: f32,
) -> Vec<u8> {
    assert!(width > 0 && height > 0, "empty image");
    assert_eq!(
        input.len(),
        width * height * 4,
        "input buffer size mismatch"
    );
    let distance = distance.max(MIN_DISTANCE);
    let mut linear = Image3F::new(width, height);
    let mut alpha_plane = AlphaPlane::new();
    alpha_plane.resize(width * height, 0);
    for (y, (row, alpha_row)) in input
        .chunks_exact(width * 4)
        .zip(alpha_plane.chunks_exact_mut(width))
        .enumerate()
    {
        let [r_row, g_row, b_row] = linear.all_plane_rows_mut(y);
        for ((((r, g), b), src), alpha) in r_row
            .iter_mut()
            .zip(g_row.iter_mut())
            .zip(b_row.iter_mut())
            .zip(row.as_chunks::<4>().0.iter())
            .zip(alpha_row.iter_mut())
        {
            *r = srgb_to_linear_u8(src[0]);
            *g = srgb_to_linear_u8(src[1]);
            *b = srgb_to_linear_u8(src[2]);
            *alpha = src[3];
        }
    }
    encode_with_config(
        &linear,
        &EncodeConfig::with_distance(distance).with_alpha(alpha_plane),
    )
}

/// Encode a linear-light RGB `Image3F` with the supplied configuration.
pub fn encode_with_config(input: &Image3F, config: &EncodeConfig) -> Vec<u8> {
    assert!(input.xsize() > 0 && input.ysize() > 0, "empty image");
    assert!(
        config.distance > 0.0,
        "distance must be positive (lossless not supported)"
    );
    assert!(
        config.icc_profile.is_none(),
        "ICC profile injection is not yet supported by jixel; only built-in \
         color encodings (sRGB / Display P3 / BT.2020 / etc.) are currently \
         representable. Track this as a future enhancement."
    );
    if let Some(alpha) = config.alpha.as_ref() {
        let expected = input.xsize() * input.ysize();
        assert_eq!(
            alpha.len(),
            expected,
            "alpha plane has wrong size: {} (expected {expected})",
            alpha.len()
        );
    }

    let distance = config.distance.max(MIN_DISTANCE);
    let has_alpha = config.alpha.is_some();

    let mut w = BitWriter::new();

    // Codestream signature.
    w.write(8, 0xFF);
    w.write(8, CODESTREAM_MARKER as u64);

    // Header: image dimensions.
    write_size_header(input.xsize(), input.ysize(), &mut w);

    // Header: image metadata (the bit layout is fixed by the JXL spec).
    write_image_metadata(&config.color_encoding, has_alpha, &mut w);

    encode_frame(distance, input, config.alpha.as_deref(), &mut w);

    w.into_bytes()
}

/// Write a single dimension using JXL's 4-bucket variable-length encoding.
fn write_size(size: u32, w: &mut BitWriter) {
    let size_minus_one = size - 1;
    const BUCKET_BITS: [u32; 4] = [9, 13, 18, 30];
    for (selector, &bits) in BUCKET_BITS.iter().enumerate() {
        if size_minus_one < (1 << bits) {
            w.write(2, selector as u64);
            w.write(bits as usize, size_minus_one as u64);
            return;
        }
    }
    unreachable!("dimension was bounds-checked against MAX_DIMENSION");
}

fn write_size_header(xsize: usize, ysize: usize, w: &mut BitWriter) {
    assert!(
        xsize <= MAX_DIMENSION && ysize <= MAX_DIMENSION,
        "image too large: max dimension is {MAX_DIMENSION}"
    );
    w.write(1, 0); // small = false (use full dimension fields)
    write_size(ysize as u32, w);
    w.write(3, 0); // ratio = 0 (no fixed aspect)
    write_size(xsize as u32, w);
}

// -----------------------------------------------------------------------------
// Image metadata.
// -----------------------------------------------------------------------------

/// jixel always writes XYB-encoded f32 RGB images.
/// When `has_alpha` is true, also declares one Alpha extra channel.
fn write_image_metadata(color_encoding: &ColorEncoding, has_alpha: bool, w: &mut BitWriter) {
    // ImageMetadata: not all-default, since we need to flip a few fields.
    w.write(1, 0); // all_default = false
    w.write(1, 0); // extra_fields = false (no preview / animation / extra)

    // BitDepth: U8 sample format.
    //
    // Note: with `xyb_encoded = true` (which jixel always sets), the color
    // channels are stored as quantized XYB coefficients regardless of this
    // field — it's purely metadata describing the original source. We choose
    // U8 here because:
    //
    //   1. It matches what cjxl emits for ordinary 8-bit input images, and
    //      thus exercises the most well-trodden decoder path in libjxl.
    //
    //   2. When `has_alpha = true`, the alpha ExtraChannelInfo's `bit_depth`
    //      defaults to this value. Declaring the image as F32 here caused
    //      libjxl 0.7's output stage to apply a 2× scaling to alpha pixels
    //      (the alpha pipeline indexes off `full_image.bitdepth`, which is
    //      taken from this field). Declaring U8 makes that go away.
    //
    //   3. The encoder doesn't currently round-trip bit-depth information
    //      from the input; everything goes through f32 internally, gets XYB-
    //      encoded, and quantized. So declaring U8 vs F32 makes no difference
    //      to output quality for the color channels.
    w.write(1, 0); // floating_point_sample = false
    w.write(2, 0); // bits_per_sample selector = 0 → 8

    w.write(1, 1); // modular_16_bit_buffer_sufficient = true (8-bit fits in i16)

    // num_extra_channels: u2S(0, 1, 2, Bits(4)+3) — selector "01" gives value 1.
    if has_alpha {
        w.write(2, 1); // num_extra_channels = 1

        // ExtraChannelInfo for the alpha channel — all defaults work now that
        // the image bit_depth above is U8 (defaults: ec_type=Alpha, bit_depth
        // inherited as U8, dim_shift=0, empty name, alpha_associated=false).
        w.write(1, 1); // all_default = true
    } else {
        w.write(2, 0); // num_extra_channels = 0
    }

    w.write(1, 1); // xyb_encoded = true

    // ColorEncoding: handled by the typed helper.
    write_color_encoding(color_encoding, w);

    // Extensions and transform data.
    w.write(2, 0); // extensions: none
    w.write(1, 1); // all_default transform data (default OpsinInverseMatrix etc.)

    // The codestream is byte-aligned before the frame.
    w.zero_pad_to_byte();
}
