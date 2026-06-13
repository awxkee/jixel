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
