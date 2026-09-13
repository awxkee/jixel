/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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
//! ISO 21496-1 gain map metadata ("the ISO bundle").
//!
//! This is the binary metadata blob that a JPEG XL `jhgm` box carries in its
//! `gain_map_metadata` field (the same blob Ultra HDR JPEGs carry in an APP2
//! segment). The struct and the parser are ported from
//! [`gainforge`](https://github.com/awxkee/gainforge); the writer mirrors
//! libultrahdr's `encodeGainmapMetadata`, so the output parses back with
//! gainforge, libultrahdr and libjxl-based consumers.
//!
//! Every parameter is a rational number `numerator / denominator`, one per
//! color channel where the standard allows it. Single-channel metadata is
//! represented by three identical channels (the writer collapses them back
//! into the single-channel form).

use crate::EncodeError;

/// Binary ISO 21496-1 gain map metadata.
///
/// Fields are rationals: `*_n` is the numerator, `*_d` the denominator. Per-
/// channel arrays hold one value per RGB channel; when all three are equal
/// the writer emits the compact single-channel form.
///
/// Build one with [`IsoGainMap::from_floats`] (from floating-point gain map
/// parameters) or fill the fields directly; serialize with
/// [`IsoGainMap::to_metadata`] and parse with [`IsoGainMap::from_metadata`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsoGainMap {
    /// log2 of the minimum gain (per channel).
    pub gain_map_min_n: [i32; 3],
    pub gain_map_min_d: [u32; 3],
    /// log2 of the maximum gain (per channel).
    pub gain_map_max_n: [i32; 3],
    pub gain_map_max_d: [u32; 3],
    /// Gamma applied to the stored gain map samples (per channel).
    pub gain_map_gamma_n: [u32; 3],
    pub gain_map_gamma_d: [u32; 3],

    /// Offset added to the base image before applying the gain (per channel).
    pub base_offset_n: [i32; 3],
    pub base_offset_d: [u32; 3],
    /// Offset added to the alternate image before applying the gain (per channel).
    pub alternate_offset_n: [i32; 3],
    pub alternate_offset_d: [u32; 3],

    /// log2 of the base image HDR headroom.
    pub base_hdr_headroom_n: u32,
    pub base_hdr_headroom_d: u32,
    /// log2 of the alternate image HDR headroom.
    pub alternate_hdr_headroom_n: u32,
    pub alternate_hdr_headroom_d: u32,

    /// True when the gain map maps from the alternate image back to the base.
    pub backward_direction: bool,
    /// True when the gain map is applied in the base image color space
    /// (false: in the alternate image color space).
    pub use_base_color_space: bool,
}

impl Default for IsoGainMap {
    /// Identity gain map: gain 1 everywhere, gamma 1, no offsets, no headroom.
    fn default() -> Self {
        Self {
            gain_map_min_n: [0; 3],
            gain_map_min_d: [1; 3],
            gain_map_max_n: [0; 3],
            gain_map_max_d: [1; 3],
            gain_map_gamma_n: [1; 3],
            gain_map_gamma_d: [1; 3],
            base_offset_n: [0; 3],
            base_offset_d: [1; 3],
            alternate_offset_n: [0; 3],
            alternate_offset_d: [1; 3],
            base_hdr_headroom_n: 0,
            base_hdr_headroom_d: 1,
            alternate_hdr_headroom_n: 0,
            alternate_hdr_headroom_d: 1,
            backward_direction: false,
            use_base_color_space: true,
        }
    }
}

/// Floating-point gain map parameters, the convenient way to build an
/// [`IsoGainMap`]. Per-channel arrays are `[R, G, B]`; pass three identical
/// values for a single-channel (luminance) gain map.
///
/// The values follow the ISO 21496-1 / Ultra HDR conventions:
/// `min`/`max` are `log2` of the smallest/largest gain in the map,
/// `base_hdr_headroom`/`alternate_hdr_headroom` are `log2` of each rendition's
/// headroom (`0.0` for an SDR base), offsets are in normalised `0..1` units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainMapFloats {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub gamma: [f32; 3],
    pub base_offset: [f32; 3],
    pub alternate_offset: [f32; 3],
    pub base_hdr_headroom: f32,
    pub alternate_hdr_headroom: f32,
    pub backward_direction: bool,
    pub use_base_color_space: bool,
}

impl Default for GainMapFloats {
    /// Identity gain map, gamma 1, Ultra HDR default offsets (1/64).
    fn default() -> Self {
        Self {
            min: [0.0; 3],
            max: [0.0; 3],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 0.0,
            backward_direction: false,
            use_base_color_space: true,
        }
    }
}

const IS_MULTICHANNEL_MASK: u8 = 1 << 7;
const USE_BASE_COLORSPACE_MASK: u8 = 1 << 6;
const BACKWARD_DIRECTION_MASK: u8 = 1 << 2;
const USE_COMMON_DENOMINATOR_MASK: u8 = 1 << 3;

fn bad(msg: &'static str) -> EncodeError {
    EncodeError::InvalidGainMapMetadata(msg)
}

fn read_u32(arr: &[u8], pos: &mut usize) -> Result<u32, EncodeError> {
    let s = arr
        .get(*pos..*pos + 4)
        .ok_or_else(|| bad("gain map metadata truncated"))?;
    *pos += 4;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_s32(arr: &[u8], pos: &mut usize) -> Result<i32, EncodeError> {
    read_u32(arr, pos).map(|v| v as i32)
}

/// Continued-fraction approximation of `v` as `n/d` with `n <= max_numerator`
/// (ported from gainforge / libultrahdr).
fn float_to_unsigned_fraction_impl(v: f32, max_numerator: u32) -> Option<(u32, u32)> {
    if v.is_nan() || v < 0.0 || v > max_numerator as f32 {
        return None;
    }

    let max_d = if v <= 1.0 {
        u32::MAX as u64
    } else {
        (max_numerator as f64 / v.floor() as f64) as u64
    };

    let mut denominator: u32 = 1;
    let mut previous_d: u32 = 0;
    let mut current_v = v.fract() as f64;
    let max_iter = 39;

    for _ in 0..max_iter {
        let numerator_double = (denominator as f64) * (v as f64);
        if numerator_double > max_numerator as f64 {
            return None;
        }

        let numerator = numerator_double.round() as u32;
        if (numerator_double - numerator as f64).abs() == 0.0 {
            return Some((numerator, denominator));
        }

        current_v = 1.0 / current_v;
        let new_d = previous_d as u64 + (current_v.floor() as u64) * (denominator as u64);
        if new_d > max_d {
            return Some((numerator, denominator));
        }

        previous_d = denominator;
        if new_d > u32::MAX as u64 {
            return None;
        }

        denominator = new_d as u32;
        current_v -= current_v.floor();
    }

    let numerator = ((denominator as f64) * (v as f64)).round() as u32;
    Some((numerator, denominator))
}

/// Approximate `v` as a signed rational `n/d` (`|n| <= i32::MAX`).
pub(crate) fn float_to_signed_fraction(v: f32) -> Option<(i32, u32)> {
    let (numerator, denominator) = float_to_unsigned_fraction_impl(v.abs(), i32::MAX as u32)?;
    let mut n = numerator as i32;
    if v < 0.0 {
        n = -n;
    }
    Some((n, denominator))
}

/// Approximate a non-negative `v` as an unsigned rational `n/d`.
pub(crate) fn float_to_unsigned_fraction(v: f32) -> Option<(u32, u32)> {
    float_to_unsigned_fraction_impl(v, i32::MAX as u32)
}

impl IsoGainMap {
    /// Build the metadata from floating-point parameters, converting every
    /// value to a rational. Fails when a value is NaN, negative where the
    /// standard requires a non-negative number (gamma, headroom, offsets when
    /// unsigned), or too large to represent.
    pub fn from_floats(v: &GainMapFloats) -> Result<Self, EncodeError> {
        let mut m = IsoGainMap {
            backward_direction: v.backward_direction,
            use_base_color_space: v.use_base_color_space,
            ..IsoGainMap::default()
        };
        for c in 0..3 {
            (m.gain_map_min_n[c], m.gain_map_min_d[c]) =
                float_to_signed_fraction(v.min[c]).ok_or_else(|| bad("gain map min"))?;
            (m.gain_map_max_n[c], m.gain_map_max_d[c]) =
                float_to_signed_fraction(v.max[c]).ok_or_else(|| bad("gain map max"))?;
            if v.gamma[c].is_nan() || v.gamma[c] <= 0.0 {
                return Err(bad("gain map gamma must be positive"));
            }
            (m.gain_map_gamma_n[c], m.gain_map_gamma_d[c]) =
                float_to_unsigned_fraction(v.gamma[c]).ok_or_else(|| bad("gain map gamma"))?;
            (m.base_offset_n[c], m.base_offset_d[c]) =
                float_to_signed_fraction(v.base_offset[c]).ok_or_else(|| bad("base offset"))?;
            (m.alternate_offset_n[c], m.alternate_offset_d[c]) =
                float_to_signed_fraction(v.alternate_offset[c])
                    .ok_or_else(|| bad("alternate offset"))?;
        }
        (m.base_hdr_headroom_n, m.base_hdr_headroom_d) =
            float_to_unsigned_fraction(v.base_hdr_headroom)
                .ok_or_else(|| bad("base HDR headroom must be a finite non-negative number"))?;
        (m.alternate_hdr_headroom_n, m.alternate_hdr_headroom_d) =
            float_to_unsigned_fraction(v.alternate_hdr_headroom).ok_or_else(|| {
                bad("alternate HDR headroom must be a finite non-negative number")
            })?;
        Ok(m)
    }

    /// Floating-point view of the parameters.
    pub fn to_floats(&self) -> GainMapFloats {
        let f = |n: i32, d: u32| n as f32 / d as f32;
        let u = |n: u32, d: u32| n as f32 / d as f32;
        let arr = |n: &[i32; 3], d: &[u32; 3]| [f(n[0], d[0]), f(n[1], d[1]), f(n[2], d[2])];
        GainMapFloats {
            min: arr(&self.gain_map_min_n, &self.gain_map_min_d),
            max: arr(&self.gain_map_max_n, &self.gain_map_max_d),
            gamma: [
                u(self.gain_map_gamma_n[0], self.gain_map_gamma_d[0]),
                u(self.gain_map_gamma_n[1], self.gain_map_gamma_d[1]),
                u(self.gain_map_gamma_n[2], self.gain_map_gamma_d[2]),
            ],
            base_offset: arr(&self.base_offset_n, &self.base_offset_d),
            alternate_offset: arr(&self.alternate_offset_n, &self.alternate_offset_d),
            base_hdr_headroom: u(self.base_hdr_headroom_n, self.base_hdr_headroom_d),
            alternate_hdr_headroom: u(self.alternate_hdr_headroom_n, self.alternate_hdr_headroom_d),
            backward_direction: self.backward_direction,
            use_base_color_space: self.use_base_color_space,
        }
    }

    /// True when at least one per-channel parameter differs between channels,
    /// i.e. the metadata must be written in the three-channel form.
    pub fn is_multichannel(&self) -> bool {
        let same = |n: &[i32; 3], d: &[u32; 3]| {
            n[0] == n[1] && n[1] == n[2] && d[0] == d[1] && d[1] == d[2]
        };
        let same_u = |n: &[u32; 3], d: &[u32; 3]| {
            n[0] == n[1] && n[1] == n[2] && d[0] == d[1] && d[1] == d[2]
        };
        !(same(&self.gain_map_min_n, &self.gain_map_min_d)
            && same(&self.gain_map_max_n, &self.gain_map_max_d)
            && same_u(&self.gain_map_gamma_n, &self.gain_map_gamma_d)
            && same(&self.base_offset_n, &self.base_offset_d)
            && same(&self.alternate_offset_n, &self.alternate_offset_d))
    }

    /// The common denominator, if every rational in the (written) channels
    /// shares one; the writer then uses the compact common-denominator form.
    fn common_denominator(&self, channels: usize) -> Option<u32> {
        let d = self.base_hdr_headroom_d;
        if d == 0 || self.alternate_hdr_headroom_d != d {
            return None;
        }
        for c in 0..channels {
            if self.gain_map_min_d[c] != d
                || self.gain_map_max_d[c] != d
                || self.gain_map_gamma_d[c] != d
                || self.base_offset_d[c] != d
                || self.alternate_offset_d[c] != d
            {
                return None;
            }
        }
        Some(d)
    }

    /// Validate the rationals (no zero denominators).
    fn validate(&self) -> Result<(), EncodeError> {
        if self.base_hdr_headroom_d == 0 || self.alternate_hdr_headroom_d == 0 {
            return Err(bad("zero denominator in HDR headroom"));
        }
        for c in 0..3 {
            if self.gain_map_min_d[c] == 0
                || self.gain_map_max_d[c] == 0
                || self.gain_map_gamma_d[c] == 0
                || self.base_offset_d[c] == 0
                || self.alternate_offset_d[c] == 0
            {
                return Err(bad("zero denominator in per-channel gain map parameter"));
            }
            if self.gain_map_gamma_n[c] == 0 {
                return Err(bad("gain map gamma must be non-zero"));
            }
        }
        Ok(())
    }

    /// serialize to the ISO 21496-1 binary layout (big-endian):
    ///
    /// ```text
    /// u16 minimum_version = 0
    /// u16 writer_version  = 0
    /// u8  flags: bit7 multichannel, bit6 use_base_color_space,
    ///            bit2 backward_direction, bit3 common denominator
    /// [u32 common_denominator]
    /// base_hdr_headroom, alternate_hdr_headroom
    /// per channel (1 or 3): min, max, gamma, base_offset, alternate_offset
    /// ```
    ///
    /// Each rational is `n` alone in the common-denominator form, or `n, d`
    /// otherwise. Single-channel form is used when all three channels agree.
    pub fn to_metadata(&self) -> Result<Vec<u8>, EncodeError> {
        self.validate()?;
        let channels = if self.is_multichannel() { 3 } else { 1 };
        let common = self.common_denominator(channels);

        let mut flags = 0u8;
        if channels == 3 {
            flags |= IS_MULTICHANNEL_MASK;
        }
        if self.use_base_color_space {
            flags |= USE_BASE_COLORSPACE_MASK;
        }
        if self.backward_direction {
            flags |= BACKWARD_DIRECTION_MASK;
        }
        if common.is_some() {
            flags |= USE_COMMON_DENOMINATOR_MASK;
        }

        let mut out = Vec::with_capacity(5 + 4 + 16 + 40 * channels);
        out.extend_from_slice(&0u16.to_be_bytes()); // minimum_version
        out.extend_from_slice(&0u16.to_be_bytes()); // writer_version
        out.push(flags);

        let put_u32 = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_be_bytes());
        let put_s32 = |out: &mut Vec<u8>, v: i32| out.extend_from_slice(&v.to_be_bytes());

        if let Some(d) = common {
            put_u32(&mut out, d);
            put_u32(&mut out, self.base_hdr_headroom_n);
            put_u32(&mut out, self.alternate_hdr_headroom_n);
            for c in 0..channels {
                put_s32(&mut out, self.gain_map_min_n[c]);
                put_s32(&mut out, self.gain_map_max_n[c]);
                put_u32(&mut out, self.gain_map_gamma_n[c]);
                put_s32(&mut out, self.base_offset_n[c]);
                put_s32(&mut out, self.alternate_offset_n[c]);
            }
        } else {
            put_u32(&mut out, self.base_hdr_headroom_n);
            put_u32(&mut out, self.base_hdr_headroom_d);
            put_u32(&mut out, self.alternate_hdr_headroom_n);
            put_u32(&mut out, self.alternate_hdr_headroom_d);
            for c in 0..channels {
                put_s32(&mut out, self.gain_map_min_n[c]);
                put_u32(&mut out, self.gain_map_min_d[c]);
                put_s32(&mut out, self.gain_map_max_n[c]);
                put_u32(&mut out, self.gain_map_max_d[c]);
                put_u32(&mut out, self.gain_map_gamma_n[c]);
                put_u32(&mut out, self.gain_map_gamma_d[c]);
                put_s32(&mut out, self.base_offset_n[c]);
                put_u32(&mut out, self.base_offset_d[c]);
                put_s32(&mut out, self.alternate_offset_n[c]);
                put_u32(&mut out, self.alternate_offset_d[c]);
            }
        }
        Ok(out)
    }

    /// Parse the ISO 21496-1 binary layout written by [`IsoGainMap::to_metadata`]
    /// (or by libultrahdr / any Ultra HDR encoder). Single-channel metadata is
    /// expanded to three identical channels.
    pub fn from_metadata(in_data: &[u8]) -> Result<Self, EncodeError> {
        if in_data.len() < 5 {
            return Err(bad("gain map metadata too short"));
        }

        let mut pos = 0;
        let min_version = u16::from_be_bytes([in_data[0], in_data[1]]);
        pos += 2;
        if min_version != 0 {
            return Err(bad("unsupported gain map metadata minimum version"));
        }
        pos += 2; // writer_version: informational only

        let flags = in_data[pos];
        pos += 1;
        let channel_count = if (flags & IS_MULTICHANNEL_MASK) != 0 {
            3
        } else {
            1
        };

        let mut m = IsoGainMap {
            use_base_color_space: (flags & USE_BASE_COLORSPACE_MASK) != 0,
            backward_direction: (flags & BACKWARD_DIRECTION_MASK) != 0,
            ..IsoGainMap::default()
        };
        let use_common_denominator = (flags & USE_COMMON_DENOMINATOR_MASK) != 0;

        if use_common_denominator {
            let d = read_u32(in_data, &mut pos)?;
            m.base_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            m.base_hdr_headroom_d = d;
            m.alternate_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            m.alternate_hdr_headroom_d = d;

            for c in 0..channel_count {
                m.gain_map_min_n[c] = read_s32(in_data, &mut pos)?;
                m.gain_map_min_d[c] = d;
                m.gain_map_max_n[c] = read_s32(in_data, &mut pos)?;
                m.gain_map_max_d[c] = d;
                m.gain_map_gamma_n[c] = read_u32(in_data, &mut pos)?;
                m.gain_map_gamma_d[c] = d;
                m.base_offset_n[c] = read_s32(in_data, &mut pos)?;
                m.base_offset_d[c] = d;
                m.alternate_offset_n[c] = read_s32(in_data, &mut pos)?;
                m.alternate_offset_d[c] = d;
            }
        } else {
            m.base_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            m.base_hdr_headroom_d = read_u32(in_data, &mut pos)?;
            m.alternate_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            m.alternate_hdr_headroom_d = read_u32(in_data, &mut pos)?;

            for c in 0..channel_count {
                m.gain_map_min_n[c] = read_s32(in_data, &mut pos)?;
                m.gain_map_min_d[c] = read_u32(in_data, &mut pos)?;
                m.gain_map_max_n[c] = read_s32(in_data, &mut pos)?;
                m.gain_map_max_d[c] = read_u32(in_data, &mut pos)?;
                m.gain_map_gamma_n[c] = read_u32(in_data, &mut pos)?;
                m.gain_map_gamma_d[c] = read_u32(in_data, &mut pos)?;
                m.base_offset_n[c] = read_s32(in_data, &mut pos)?;
                m.base_offset_d[c] = read_u32(in_data, &mut pos)?;
                m.alternate_offset_n[c] = read_s32(in_data, &mut pos)?;
                m.alternate_offset_d[c] = read_u32(in_data, &mut pos)?;
            }
        }

        for c in channel_count..3 {
            m.gain_map_min_n[c] = m.gain_map_min_n[0];
            m.gain_map_min_d[c] = m.gain_map_min_d[0];
            m.gain_map_max_n[c] = m.gain_map_max_n[0];
            m.gain_map_max_d[c] = m.gain_map_max_d[0];
            m.gain_map_gamma_n[c] = m.gain_map_gamma_n[0];
            m.gain_map_gamma_d[c] = m.gain_map_gamma_d[0];
            m.base_offset_n[c] = m.base_offset_n[0];
            m.base_offset_d[c] = m.base_offset_d[0];
            m.alternate_offset_n[c] = m.alternate_offset_n[0];
            m.alternate_offset_d[c] = m.alternate_offset_d[0];
        }

        m.validate()?;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_floats() -> GainMapFloats {
        GainMapFloats {
            min: [-0.5, -0.25, 0.0],
            max: [2.0, 2.5, 3.0],
            gamma: [1.0, 1.2, 0.8],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.32,
            backward_direction: false,
            use_base_color_space: true,
        }
    }

    #[test]
    fn multichannel_round_trip_is_exact() {
        let m = IsoGainMap::from_floats(&sample_floats()).unwrap();
        assert!(m.is_multichannel());
        let bytes = m.to_metadata().unwrap();
        // 5 header + 4*4 headroom + 3 channels * 10 u32.
        assert_eq!(bytes.len(), 5 + 16 + 3 * 40);
        assert_eq!(bytes[4] & IS_MULTICHANNEL_MASK, IS_MULTICHANNEL_MASK);
        let back = IsoGainMap::from_metadata(&bytes).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn single_channel_uses_compact_common_denominator_form() {
        let m = IsoGainMap {
            gain_map_min_n: [-2; 3],
            gain_map_max_n: [6; 3],
            gain_map_gamma_n: [4; 3],
            base_offset_n: [0; 3],
            alternate_offset_n: [0; 3],
            gain_map_min_d: [4; 3],
            gain_map_max_d: [4; 3],
            gain_map_gamma_d: [4; 3],
            base_offset_d: [4; 3],
            alternate_offset_d: [4; 3],
            base_hdr_headroom_n: 0,
            base_hdr_headroom_d: 4,
            alternate_hdr_headroom_n: 8,
            alternate_hdr_headroom_d: 4,
            backward_direction: true,
            use_base_color_space: false,
        };
        assert!(!m.is_multichannel());
        let bytes = m.to_metadata().unwrap();
        // 5 header + common denominator + 2 headroom numerators + 5 numerators.
        assert_eq!(bytes.len(), 5 + 4 + 8 + 20);
        assert_eq!(bytes[..4], [0, 0, 0, 0]);
        assert_eq!(
            bytes[4],
            BACKWARD_DIRECTION_MASK | USE_COMMON_DENOMINATOR_MASK
        );
        let back = IsoGainMap::from_metadata(&bytes).unwrap();
        assert_eq!(back, m);
        let f = back.to_floats();
        assert_eq!(f.min, [-0.5; 3]);
        assert_eq!(f.max, [1.5; 3]);
        assert_eq!(f.alternate_hdr_headroom, 2.0);
        assert!(f.backward_direction);
        assert!(!f.use_base_color_space);
    }

    #[test]
    fn floats_survive_the_rational_round_trip() {
        let f = sample_floats();
        let back = IsoGainMap::from_floats(&f).unwrap().to_floats();
        for c in 0..3 {
            assert!((back.min[c] - f.min[c]).abs() < 1e-6);
            assert!((back.max[c] - f.max[c]).abs() < 1e-6);
            assert!((back.gamma[c] - f.gamma[c]).abs() < 1e-6);
            assert!((back.base_offset[c] - f.base_offset[c]).abs() < 1e-6);
        }
        assert!((back.alternate_hdr_headroom - f.alternate_hdr_headroom).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_floats() {
        let f = GainMapFloats {
            gamma: [1.0, 0.0, 1.0],
            ..GainMapFloats::default()
        };
        assert!(IsoGainMap::from_floats(&f).is_err());
        let f = GainMapFloats {
            base_hdr_headroom: -1.0,
            ..GainMapFloats::default()
        };
        assert!(IsoGainMap::from_floats(&f).is_err());
        let f = GainMapFloats {
            min: [f32::NAN, 0.0, 0.0],
            ..GainMapFloats::default()
        };
        assert!(IsoGainMap::from_floats(&f).is_err());
    }

    #[test]
    fn rejects_truncated_and_wrong_version() {
        let m = IsoGainMap::from_floats(&sample_floats()).unwrap();
        let bytes = m.to_metadata().unwrap();
        assert!(IsoGainMap::from_metadata(&bytes[..bytes.len() - 1]).is_err());
        let mut v = bytes.clone();
        v[1] = 1;
        assert!(IsoGainMap::from_metadata(&v).is_err());
        let mut zero = IsoGainMap::default();
        zero.gain_map_min_d[0] = 0;
        assert!(zero.to_metadata().is_err());
    }

    #[test]
    fn parses_libultrahdr_style_single_channel_blob() {
        // Hand-assembled: version 0/0, flags = use_base_color_space only,
        // explicit denominators, one channel.
        let mut b = vec![0, 0, 0, 0, USE_BASE_COLORSPACE_MASK];
        let mut u = |v: u32| b.extend_from_slice(&v.to_be_bytes());
        u(0);
        u(1); // base headroom 0/1
        u(3);
        u(1); // alt headroom 3/1
        u(0u32.wrapping_sub(1));
        u(2); // min -1/2
        u(3);
        u(1); // max 3/1
        u(1);
        u(1); // gamma 1/1
        u(1);
        u(64); // base offset 1/64
        u(1);
        u(64); // alt offset 1/64
        let m = IsoGainMap::from_metadata(&b).unwrap();
        assert_eq!(m.gain_map_min_n, [-1; 3]);
        assert_eq!(m.gain_map_min_d, [2; 3]);
        assert_eq!(m.gain_map_max_n, [3; 3]);
        assert_eq!(m.alternate_hdr_headroom_n, 3);
        assert!(m.use_base_color_space);
        assert!(!m.backward_direction);
        assert_eq!(m.to_metadata().unwrap(), b);
    }
}
