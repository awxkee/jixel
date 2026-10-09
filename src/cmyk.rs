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
//! CMYK encoding: C, M and Y are the color planes, K is a `kBlack` extra
//! channel, and the mandatory ICC profile defines the colors. JPEG XL stores
//! every channel inverted (0 = full ink, 1 = no ink).
use crate::coder_scratch::CoderScratch;
use crate::coding::{CodingTransform, srgb_eotf};
use crate::encode_image::{
    AlphaPlane, BitsPerSample, EncodeConfigImpl, MAX_DIMENSION, MIN_DISTANCE, checked_buffer_size,
    encode_with_coded_planes, encode_with_config_loseless, lossy_context,
};
use crate::image::Image3F;
use crate::xyb::XybMatrix;
use crate::{EncodeConfig, EncodeError};

/// How CMYK samples express ink.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InkConvention {
    /// 0 is no ink, the maximum is full ink: ICC profiles, TIFF, most
    /// color-management APIs.
    Amount,
    /// 0 is full ink, the maximum is no ink: Adobe CMYK JPEGs and JPEG XL
    /// itself.
    Inverted,
}

trait CmykSample: Copy {
    fn value(self) -> u32;
}

impl CmykSample for u8 {
    #[inline]
    fn value(self) -> u32 {
        u32::from(self)
    }
}

impl CmykSample for u16 {
    #[inline]
    fn value(self) -> u32 {
        u32::from(self)
    }
}

/// Encode interleaved 8-bit CMYK samples. `icc_profile` must be a CMYK
/// profile; it is embedded and is what decoders render the image with.
pub fn encode_image_cmyk(
    input: &[u8],
    width: usize,
    height: usize,
    icc_profile: &[u8],
    convention: InkConvention,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    encode_cmyk_impl(input, width, height, 8, icc_profile, convention, config)
}

/// Encode interleaved CMYK samples of `bits` (10, 12 or 16) significant bits.
pub fn encode_image_cmyk_16bit(
    input: &[u16],
    width: usize,
    height: usize,
    bits: u8,
    icc_profile: &[u8],
    convention: InkConvention,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if !matches!(bits, 10 | 12 | 16) {
        return Err(EncodeError::Unsupported(
            "CMYK bit depth must be 10, 12 or 16",
        ));
    }
    encode_cmyk_impl(input, width, height, bits, icc_profile, convention, config)
}

/// The ICC data color space, bytes 16..20 of the header.
fn is_cmyk_profile(icc: &[u8]) -> bool {
    icc.len() >= 128 && &icc[16..20] == b"CMYK"
}

fn encode_cmyk_impl<T: CmykSample + crate::encode_image::AsSignedInt>(
    input: &[T],
    width: usize,
    height: usize,
    bits: u8,
    icc_profile: &[u8],
    convention: InkConvention,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::DimensionTooLarge { width, height });
    }
    let expected = checked_buffer_size::<T>(width, height, 4)?;
    if input.len() != expected {
        return Err(EncodeError::InputSizeMismatch {
            expected,
            actual: input.len(),
        });
    }
    if !config.distance.is_finite() || config.distance <= 0.0 {
        return Err(EncodeError::InvalidDistance(config.distance));
    }
    if !is_cmyk_profile(icc_profile) {
        return Err(EncodeError::Unsupported("CMYK needs a CMYK ICC profile"));
    }
    let max = (1u32 << bits) - 1;
    // Stored (inverted) integer value of one sample.
    let stored = |v: T| -> u32 {
        match convention {
            InkConvention::Inverted => v.value(),
            InkConvention::Amount => max - v.value().min(max),
        }
    };
    let bps = match bits {
        8 => BitsPerSample::Eight,
        10 => BitsPerSample::Ten,
        12 => BitsPerSample::Twelve,
        _ => BitsPerSample::Sixteen,
    };
    let base = EncodeConfigImpl::with_distance(config.distance)
        .with_icc_profile(Some(icc_profile.to_vec()))
        .with_black(true)
        .with_exif(config.exif.clone())
        .with_xmp(config.xmp.clone())
        .with_brotli_compression(config.brotli_compression.clone())
        .with_orientation(config.orientation)
        .with_speed(config.speed)
        .with_decoding_speed(config.decoding_speed)
        .with_num_threads(config.num_threads);

    if config.lossless {
        let stored_input: Vec<u16> = input.iter().map(|&v| stored(v) as u16).collect();
        let cfg = base
            .with_lossless(true)
            .with_progressive(config.progressive)
            .with_bits_per_sample(bps);
        return if bits == 8 {
            let bytes: Vec<u8> = stored_input.iter().map(|&v| v as u8).collect();
            encode_with_config_loseless(&bytes, width, height, true, 8, &cfg)
        } else {
            encode_with_config_loseless(&stored_input, width, height, true, bits, &cfg)
        };
    }

    let distance = config.distance.max(MIN_DISTANCE);
    let mut ctx = lossy_context(config, distance, XybMatrix::SPEC, width * height);
    // XYB would render the separations away; by default the inverted C, M
    // and Y planes are coded as YCbCr, whose luma tracks the inks' share of
    // lightness.
    let coding = match config.coding_transform {
        CodingTransform::Xyb | CodingTransform::YCbCr => CodingTransform::YCbCr,
        CodingTransform::Rgb => CodingTransform::Rgb,
    };
    ctx.set_coding(coding, distance);
    // Separations retain the default EPF scales; the Slow RGB adjustment
    // was tuned on additive RGB samples.
    ctx.epf_channel_scale = coding.epf_channel_scale();
    let inv_max = 1.0 / max as f32;
    // Coded planes, the analysis rendition that steers adaptive quantization
    // (naive ink multiply, read as sRGB), and the stored K plane.
    let mut coded = Image3F::try_new(width, height)?;
    let mut linear = Image3F::try_new(width, height)?;
    let mut k_plane_u8 = Vec::new();
    let mut k_plane_u16 = Vec::new();
    for (y, row) in input.chunks_exact(width * 4).enumerate() {
        let [c0, c1, c2] = coded.all_plane_rows_mut(y);
        let [l0, l1, l2] = linear.all_plane_rows_mut(y);
        for (x, px) in row.as_chunks::<4>().0.iter().enumerate() {
            let c = stored(px[0]) as f32 * inv_max;
            let m = stored(px[1]) as f32 * inv_max;
            let yy = stored(px[2]) as f32 * inv_max;
            let k = stored(px[3]);
            [c0[x], c1[x], c2[x]] = coding.code(c, m, yy);
            let kf = k as f32 * inv_max;
            l0[x] = srgb_eotf(c * kf);
            l1[x] = srgb_eotf(m * kf);
            l2[x] = srgb_eotf(yy * kf);
            if bits == 8 {
                k_plane_u8.push(k as u8);
            } else {
                k_plane_u16.push(k as u16);
            }
        }
    }
    let mut k_plane = if bits == 8 {
        AlphaPlane::U8(k_plane_u8)
    } else {
        AlphaPlane::U16 {
            data: k_plane_u16,
            bits,
        }
    };
    let passes = crate::encode_image::progressive_schedule(
        config.progressive,
        config.progressive_passes,
        config.progressive_shifts.as_deref(),
    );
    // K uses the same Squeeze quantizer at every pass count. Its coarse
    // levels live in the global/LF streams and its AC levels in the last pass.
    ctx.extra_squeeze_distance = BLACK_DISTANCE_SCALE * distance;
    // Learned K trees use no weighted predictor, so only the tree-free
    // decoding tier rules them out. They add ~20% encode time, so the
    // fast encode tiers keep the fixed tree.
    ctx.extra_learned_trees = config.decoding_speed != crate::DecodingSpeed::Fastest
        && !matches!(
            config.speed,
            crate::Speed::UltraFast | crate::Speed::Fastest | crate::Speed::Fast
        );
    if passes.len() > 1 && crate::squeeze::default_squeeze_steps(width, height, 1).is_empty() {
        // Tiny planes have no Squeeze levels; retain the progressive lattice
        // fallback rather than changing their K precision.
        ctx.extra_step = black_step(distance, max);
        if ctx.extra_step > 1 {
            crate::modular::snap_to_lattice(&mut k_plane, ctx.extra_step);
        }
    }
    let cfg = base
        .with_progressive_from(config)
        .with_bits_per_sample(bps)
        .with_alpha(k_plane);
    let mut scratch = Box::<CoderScratch>::default();
    encode_with_coded_planes(&linear, Some(&coded), &cfg, &ctx, &mut scratch)
}

/// K lattice step in stored units, held about as fine as luma.
fn black_step(distance: f32, max: u32) -> u32 {
    let step = distance * max as f32 / 255.0;
    (step.round() as u32).clamp(1, max / 4)
}

/// K squeeze quantizer distance relative to the frame distance, in the
/// lossy-modular arm's luma units; it leaves the K plate about as faithful
/// as C, M and Y.
const BLACK_DISTANCE_SCALE: f32 = 6.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal profile header: size, version, device class and the CMYK data
    /// color space, no tags.
    fn cmyk_icc() -> Vec<u8> {
        let mut icc = vec![0u8; 132];
        icc[0..4].copy_from_slice(&132u32.to_be_bytes());
        icc[8] = 2;
        icc[12..16].copy_from_slice(b"prtr");
        icc[16..20].copy_from_slice(b"CMYK");
        icc[20..24].copy_from_slice(b"Lab ");
        icc[36..40].copy_from_slice(b"acsp");
        icc
    }

    fn ramp(width: usize, height: usize) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| {
                let (x, y) = (i % width, i / width);
                [
                    (x * 255 / width) as u8,
                    (y * 255 / height) as u8,
                    ((x + y) % 256) as u8,
                    ((x * y) % 200) as u8,
                ]
            })
            .collect()
    }

    fn has_box(bytes: &[u8], kind: &[u8; 4]) -> bool {
        bytes.windows(4).any(|w| w == kind)
    }

    #[test]
    fn rejects_non_cmyk_profiles() {
        let mut icc = cmyk_icc();
        icc[16..20].copy_from_slice(b"RGB ");
        let px = ramp(16, 16);
        let config = EncodeConfig::default();
        for profile in [&icc[..], &icc[..64]] {
            let err = encode_image_cmyk(&px, 16, 16, profile, InkConvention::Amount, &config)
                .unwrap_err();
            assert!(matches!(err, EncodeError::Unsupported(_)), "{err:?}");
        }
    }

    #[test]
    fn rejects_unsupported_bit_depths() {
        let px = vec![0u16; 16 * 16 * 4];
        let err = encode_image_cmyk_16bit(
            &px,
            16,
            16,
            9,
            &cmyk_icc(),
            InkConvention::Amount,
            &EncodeConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, EncodeError::Unsupported(_)), "{err:?}");
    }

    /// Lossy (squeezed K across several groups, plain and progressive) and
    /// lossless encodes; every CMYK file declares level 10.
    #[test]
    fn encodes_lossy_and_lossless() {
        let (w, h) = (300, 270);
        let px = ramp(w, h);
        let icc = cmyk_icc();
        for speed in [crate::Speed::Fast, crate::Speed::Slow] {
            for config in [
                EncodeConfig::default().with_speed(speed).with_distance(1.5),
                EncodeConfig::default()
                    .with_speed(speed)
                    .with_distance(1.5)
                    .with_progressive(true),
                EncodeConfig::default()
                    .with_speed(speed)
                    .with_distance(1.5)
                    .with_coding_transform(crate::CodingTransform::Rgb),
                EncodeConfig::default()
                    .with_speed(speed)
                    .with_distance(1.5)
                    .with_lossy_modular(crate::LossyModular::Force),
                EncodeConfig::default()
                    .with_speed(speed)
                    .with_lossless(true),
            ] {
                for convention in [InkConvention::Amount, InkConvention::Inverted] {
                    let bytes = encode_image_cmyk(&px, w, h, &icc, convention, &config).unwrap();
                    assert!(has_box(&bytes, b"jxll"), "CMYK needs a level-10 box");
                }
            }
        }
        let px16: Vec<u16> = px.iter().map(|&v| u16::from(v) << 4).collect();
        let bytes = encode_image_cmyk_16bit(
            &px16,
            w,
            h,
            12,
            &icc,
            InkConvention::Amount,
            &EncodeConfig::default().with_distance(1.0),
        )
        .unwrap();
        assert!(has_box(&bytes, b"jxll"));
    }
}
