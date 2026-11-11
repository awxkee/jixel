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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orientation {
    /// 1 — 0°, no transform (the stored pixels are already upright).
    #[default]
    Normal,
    /// 2 — mirrored horizontally.
    FlipH,
    /// 3 — rotated 180°.
    Rotate180,
    /// 4 — mirrored vertically.
    FlipV,
    /// 5 — mirrored horizontally then rotated 90° clockwise (transpose).
    Transpose,
    /// 6 — rotated 90° clockwise.
    Rotate90,
    /// 7 — mirrored horizontally then rotated 90° anti-clockwise (transverse).
    Transverse,
    /// 8 — rotated 90° anti-clockwise.
    Rotate270,
}

impl Orientation {
    /// The EXIF orientation value (1..=8).
    pub fn to_exif(self) -> u8 {
        match self {
            Orientation::Normal => 1,
            Orientation::FlipH => 2,
            Orientation::Rotate180 => 3,
            Orientation::FlipV => 4,
            Orientation::Transpose => 5,
            Orientation::Rotate90 => 6,
            Orientation::Transverse => 7,
            Orientation::Rotate270 => 8,
        }
    }

    /// Build from an EXIF orientation value; returns `None` outside 1..=8.
    pub fn from_exif(value: u8) -> Option<Self> {
        Some(match value {
            1 => Orientation::Normal,
            2 => Orientation::FlipH,
            3 => Orientation::Rotate180,
            4 => Orientation::FlipV,
            5 => Orientation::Transpose,
            6 => Orientation::Rotate90,
            7 => Orientation::Transverse,
            8 => Orientation::Rotate270,
            _ => return None,
        })
    }

    /// Reads the orientation tag (0x0112) from the first directory (IFD0) of
    /// a TIFF block, the same bytes [`EncodeConfig::with_exif`] takes: a PNG
    /// `eXIf` chunk, or a HEIF/JPEG Exif payload after its offset or
    /// `Exif\0\0` prefix. Both byte orders are read; returns `None` when the
    /// tag is missing, malformed or outside 1..=8.
    ///
    /// The pixel encode path does not read this on its own: viewers follow the
    /// codestream's orientation, not the Exif tag, so pass the result to
    /// [`EncodeConfig::with_orientation`] unless the pixels were already
    /// rotated upright.
    ///
    /// [`EncodeConfig::with_exif`]: crate::EncodeConfig::with_exif
    /// [`EncodeConfig::with_orientation`]: crate::EncodeConfig::with_orientation
    pub fn from_exif_tiff(tiff: &[u8]) -> Option<Self> {
        let big_endian = match tiff.get(..4)? {
            [b'M', b'M', 0, 42] => true,
            [b'I', b'I', 42, 0] => false,
            _ => return None,
        };
        let u16_at = |at: usize| -> Option<u16> {
            let b: [u8; 2] = tiff.get(at..at.checked_add(2)?)?.try_into().ok()?;
            Some(if big_endian {
                u16::from_be_bytes(b)
            } else {
                u16::from_le_bytes(b)
            })
        };
        let u32_at = |at: usize| -> Option<u32> {
            let b: [u8; 4] = tiff.get(at..at.checked_add(4)?)?.try_into().ok()?;
            Some(if big_endian {
                u32::from_be_bytes(b)
            } else {
                u32::from_le_bytes(b)
            })
        };
        let ifd = usize::try_from(u32_at(4)?).ok()?;
        let entries = usize::from(u16_at(ifd)?);
        (0..entries).find_map(|i| {
            let entry = ifd.checked_add(2 + 12 * i)?;
            // A single SHORT, stored in the first half of the value field.
            if u16_at(entry)? != 0x0112 || u16_at(entry + 2)? != 3 || u32_at(entry + 4)? != 1 {
                return None;
            }
            Orientation::from_exif(u8::try_from(u16_at(entry + 8)?).ok()?)
        })
    }

    /// The JXL codestream payload for the orientation field, which is encoded
    /// as `1 + u(3)`. Returns the 3-bit value `to_exif() - 1` (0..=7).
    pub(crate) fn to_u3(self) -> u64 {
        (self.to_exif() - 1) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_roundtrip() {
        for v in 1u8..=8 {
            let o = Orientation::from_exif(v).unwrap();
            assert_eq!(o.to_exif(), v);
            assert_eq!(o.to_u3(), (v - 1) as u64);
        }
        assert_eq!(Orientation::from_exif(0), None);
        assert_eq!(Orientation::from_exif(9), None);
    }

    #[test]
    fn default_is_normal() {
        assert_eq!(Orientation::default(), Orientation::Normal);
        assert_eq!(Orientation::default().to_exif(), 1);
    }
}
