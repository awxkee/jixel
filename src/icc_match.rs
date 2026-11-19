/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

use crate::color::transfer_to_linear_f32;
use crate::color_encoding::{
    ColorEncoding, Primaries, RenderingIntent, TransferFunction, WhitePoint,
};

/// Colorant and white-point tolerance (XYZ units). Profiles round to
/// s15Fixed16 and use slightly different published primaries; the nearest
/// supported encodings differ by 0.03 and more.
const XYZ_TOLERANCE: f64 = 2e-3;
/// Largest tone-curve deviation (normalized units) over the sampled range.
const CURVE_TOLERANCE: f64 = 2e-3;
const CURVE_SAMPLES: usize = 256;

static D50: [f64; 3] = [0.9642, 1.0, 0.8249];
const D65_XY: (f64, f64) = (0.3127, 0.3290);

/// The rendering intent of `icc` when it describes `encoding` exactly, for a
/// `grayscale` (GRAY profile) or RGB image; `None` when it does not, or when
/// it cannot be parsed.
pub(crate) fn matching_intent(
    icc: &[u8],
    encoding: &ColorEncoding,
    grayscale: bool,
) -> Option<RenderingIntent> {
    if encoding.white_point != WhitePoint::D65
        || !matches!(
            encoding.transfer,
            TransferFunction::Srgb | TransferFunction::Linear | TransferFunction::Bt709
        )
    {
        return None;
    }
    let profile = Profile::parse(icc)?;
    let space: &[u8; 4] = if grayscale { b"GRAY" } else { b"RGB " };
    if &profile.color_space != space
        || &profile.pcs != b"XYZ "
        || !matches!(&profile.class, b"mntr" | b"scnr" | b"spac")
        || profile.has_lut()
    {
        return None;
    }
    let d65 = xy_to_xyz(D65_XY);
    let adapt = match profile.tag(b"chad") {
        Some(chad) => read_sf32_matrix(chad)?,
        None => {
            // v2 profiles store the media white; v4 (and some v2) store D50
            // and leave the adaptation implicit: Bradford, as ICC.1 asks.
            let wtpt = read_xyz(profile.tag(b"wtpt")?)?;
            if !close(&wtpt, &d65) && !close(&wtpt, &D50) {
                return None;
            }
            bradford(&d65, &D50)
        }
    };
    if !close(&mul(&adapt, &d65), &D50) {
        return None;
    }
    let curves: &[&[u8; 4]] = if grayscale {
        &[b"kTRC"]
    } else {
        let to_xyz = rgb_to_xyz(encoding.primaries)?;
        for (c, tag) in [b"rXYZ", b"gXYZ", b"bXYZ"].into_iter().enumerate() {
            let colorant = read_xyz(profile.tag(tag)?)?;
            let column = [to_xyz[0][c], to_xyz[1][c], to_xyz[2][c]];
            if !close(&colorant, &mul(&adapt, &column)) {
                return None;
            }
        }
        &[b"rTRC", b"gTRC", b"bTRC"]
    };
    for tag in curves {
        let curve = Curve::parse(profile.tag(tag)?)?;
        let worst = (0..CURVE_SAMPLES)
            .map(|i| {
                let v = i as f64 / (CURVE_SAMPLES - 1) as f64;
                let expected = f64::from(transfer_to_linear_f32(v as f32, encoding.transfer));
                (curve.eval(v) - expected).abs()
            })
            .fold(0.0f64, f64::max);
        if worst > CURVE_TOLERANCE {
            return None;
        }
    }
    Some(profile.intent)
}

struct Profile<'a> {
    bytes: &'a [u8],
    class: [u8; 4],
    color_space: [u8; 4],
    pcs: [u8; 4],
    intent: RenderingIntent,
    /// (signature, offset, size), each range checked against `bytes`.
    tags: Vec<([u8; 4], usize, usize)>,
}

impl<'a> Profile<'a> {
    fn parse(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < 132 || &bytes[36..40] != b"acsp" {
            return None;
        }
        let declared = be32(bytes, 0)? as usize;
        let bytes = bytes.get(..declared.min(bytes.len()))?;
        let intent = match be32(bytes, 64)? & 0xFFFF {
            0 => RenderingIntent::Perceptual,
            1 => RenderingIntent::Relative,
            2 => RenderingIntent::Saturation,
            3 => RenderingIntent::Absolute,
            _ => return None,
        };
        let count = be32(bytes, 128)? as usize;
        if count > (bytes.len() - 132) / 12 {
            return None;
        }
        let mut tags = Vec::with_capacity(count);
        for i in 0..count {
            let at = 132 + 12 * i;
            let signature: [u8; 4] = bytes[at..at + 4].try_into().ok()?;
            let offset = be32(bytes, at + 4)? as usize;
            let size = be32(bytes, at + 8)? as usize;
            if offset.checked_add(size)? > bytes.len() {
                return None;
            }
            tags.push((signature, offset, size));
        }
        Some(Self {
            bytes,
            class: bytes[12..16].try_into().ok()?,
            color_space: bytes[16..20].try_into().ok()?,
            pcs: bytes[20..24].try_into().ok()?,
            intent,
            tags,
        })
    }

    fn tag(&self, signature: &[u8; 4]) -> Option<&'a [u8]> {
        self.tags
            .iter()
            .find(|(s, ..)| s == signature)
            .map(|&(_, offset, size)| &self.bytes[offset..offset + size])
    }

    fn has_lut(&self) -> bool {
        self.tags.iter().any(|(s, ..)| {
            matches!(
                s,
                b"A2B0"
                    | b"A2B1"
                    | b"A2B2"
                    | b"B2A0"
                    | b"B2A1"
                    | b"B2A2"
                    | b"D2B0"
                    | b"D2B1"
                    | b"D2B2"
                    | b"D2B3"
                    | b"B2D0"
                    | b"B2D1"
                    | b"B2D2"
                    | b"B2D3"
            )
        })
    }
}

enum Curve {
    Gamma(f64),
    Table(Vec<f64>),
    /// ICC parametric function types 0-4: g, a, b, c, d, e, f.
    Parametric(u16, [f64; 7]),
}

impl Curve {
    fn parse(tag: &[u8]) -> Option<Self> {
        match tag.get(..4)? {
            b"curv" => {
                let n = be32(tag, 8)? as usize;
                match n {
                    0 => Some(Self::Gamma(1.0)),
                    1 => Some(Self::Gamma(f64::from(be16(tag, 12)?) / 256.0)),
                    _ => {
                        let table = (0..n)
                            .map(|i| be16(tag, 12 + 2 * i).map(|v| f64::from(v) / 65535.0))
                            .collect::<Option<Vec<_>>>()?;
                        Some(Self::Table(table))
                    }
                }
            }
            b"para" => {
                let kind = be16(tag, 8)?;
                let count = match kind {
                    0 => 1,
                    1 => 3,
                    2 => 4,
                    3 => 5,
                    4 => 7,
                    _ => return None,
                };
                let mut p = [0.0; 7];
                for (i, v) in p.iter_mut().take(count).enumerate() {
                    *v = s15f16(tag, 12 + 4 * i)?;
                }
                Some(Self::Parametric(kind, p))
            }
            _ => None,
        }
    }

    fn eval(&self, x: f64) -> f64 {
        match self {
            Self::Gamma(g) => x.powf(*g),
            Self::Table(t) => {
                let pos = x * (t.len() - 1) as f64;
                let i = (pos.floor() as usize).min(t.len() - 2);
                let f = pos - i as f64;
                t[i] + (t[i + 1] - t[i]) * f
            }
            &Self::Parametric(kind, [g, a, b, c, d, e, f]) => {
                let pow = |base: f64| base.max(0.0).powf(g);
                match kind {
                    0 => x.powf(g),
                    1 if x >= -b / a => pow(a * x + b),
                    1 => 0.0,
                    2 if x >= -b / a => pow(a * x + b) + c,
                    2 => c,
                    3 if x >= d => pow(a * x + b),
                    3 => c * x,
                    4 if x >= d => pow(a * x + b) + e,
                    _ => c * x + f,
                }
            }
        }
    }
}

fn be16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn s15f16(b: &[u8], at: usize) -> Option<f64> {
    Some(f64::from(be32(b, at)? as i32) / 65536.0)
}

fn read_xyz(tag: &[u8]) -> Option<[f64; 3]> {
    if tag.get(..4)? != b"XYZ " {
        return None;
    }
    Some([s15f16(tag, 8)?, s15f16(tag, 12)?, s15f16(tag, 16)?])
}

fn read_sf32_matrix(tag: &[u8]) -> Option<[[f64; 3]; 3]> {
    if tag.get(..4)? != b"sf32" {
        return None;
    }
    let mut m = [[0.0; 3]; 3];
    for (i, v) in m.iter_mut().flatten().enumerate() {
        *v = s15f16(tag, 8 + 4 * i)?;
    }
    Some(m)
}

fn close(a: &[f64; 3], b: &[f64; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= XYZ_TOLERANCE)
}

fn xy_to_xyz((x, y): (f64, f64)) -> [f64; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

fn mul(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

fn invert(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let cof =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let det = m[0][0] * cof(1, 2, 1, 2) - m[0][1] * cof(1, 2, 0, 2) + m[0][2] * cof(1, 2, 0, 1);
    if det.abs() < 1e-12 {
        return None;
    }
    let adj = [
        [cof(1, 2, 1, 2), -cof(0, 2, 1, 2), cof(0, 1, 1, 2)],
        [-cof(1, 2, 0, 2), cof(0, 2, 0, 2), -cof(0, 1, 0, 2)],
        [cof(1, 2, 0, 1), -cof(0, 2, 0, 1), cof(0, 1, 0, 1)],
    ];
    Some(adj.map(|row| row.map(|v| v / det)))
}

/// Bradford chromatic adaptation from white `from` to white `to` (XYZ).
fn bradford(from: &[f64; 3], to: &[f64; 3]) -> [[f64; 3]; 3] {
    const B: [[f64; 3]; 3] = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    let (s, d) = (mul(&B, from), mul(&B, to));
    let scale = [
        [d[0] / s[0], 0.0, 0.0],
        [0.0, d[1] / s[1], 0.0],
        [0.0, 0.0, d[2] / s[2]],
    ];
    let b_inv = invert(&B).expect("Bradford is invertible");
    mat_mul(&b_inv, &mat_mul(&scale, &B))
}

/// RGB to XYZ (D65) of the primaries jixel converts exactly; `None` for the
/// others, which it treats as sRGB and so must keep their profile.
fn rgb_to_xyz(primaries: Primaries) -> Option<[[f64; 3]; 3]> {
    let xy = match primaries {
        Primaries::Bt709 => [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)],
        Primaries::Smpte431 | Primaries::Smpte432 => {
            [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)]
        }
        Primaries::Bt2020 => [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
        _ => return None,
    };
    let columns = xy.map(xy_to_xyz);
    let m = [0, 1, 2].map(|r| [0, 1, 2].map(|c| columns[c][r]));
    let s = mul(&invert(&m)?, &xy_to_xyz(D65_XY));
    Some([0, 1, 2].map(|r| [0, 1, 2].map(|c| m[r][c] * s[c])))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A matrix/TRC profile in the layout of the v4 display profiles phones
    /// and desktops attach: D50 colorants through a Bradford `chad`, a
    /// parametric sRGB curve shared by the three channels.
    pub(crate) fn synthetic_profile(
        primaries: Primaries,
        curve: Option<&[u8]>,
        extra_tags: &[([u8; 4], Vec<u8>)],
        intent: u32,
    ) -> Vec<u8> {
        let fixed = |v: f64| ((v * 65536.0).round() as i32).to_be_bytes();
        let xyz = |v: [f64; 3]| {
            let mut t = b"XYZ \0\0\0\0".to_vec();
            v.iter().for_each(|&c| t.extend(fixed(c)));
            t
        };
        let d65 = xy_to_xyz(D65_XY);
        let adapt = bradford(&d65, &D50);
        let m = rgb_to_xyz(primaries).unwrap();
        let mut chad = b"sf32\0\0\0\0".to_vec();
        adapt.iter().flatten().for_each(|&v| chad.extend(fixed(v)));
        let srgb_para = {
            let mut t = b"para\0\0\0\0\0\x03\0\0".to_vec();
            for v in [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045] {
                t.extend(fixed(v));
            }
            t
        };
        let curve = curve.map_or(srgb_para, <[u8]>::to_vec);
        let mut tags: Vec<([u8; 4], Vec<u8>)> = vec![
            (*b"wtpt", xyz(D50)),
            (*b"chad", chad),
            (*b"rTRC", curve.clone()),
            (*b"gTRC", curve.clone()),
            (*b"bTRC", curve),
        ];
        for (c, tag) in [b"rXYZ", b"gXYZ", b"bXYZ"].into_iter().enumerate() {
            tags.push((*tag, xyz(mul(&adapt, &[m[0][c], m[1][c], m[2][c]]))));
        }
        tags.extend(extra_tags.iter().cloned());
        let mut body = Vec::new();
        let mut table = Vec::new();
        let start = 132 + 12 * tags.len();
        for (sig, data) in &tags {
            table.extend(sig);
            table.extend(((start + body.len()) as u32).to_be_bytes());
            table.extend((data.len() as u32).to_be_bytes());
            body.extend(data);
            body.resize(body.len().next_multiple_of(4), 0);
        }
        let mut icc = vec![0u8; 128];
        icc[8..12].copy_from_slice(&[4, 0x40, 0, 0]);
        icc[12..16].copy_from_slice(b"mntr");
        icc[16..20].copy_from_slice(b"RGB ");
        icc[20..24].copy_from_slice(b"XYZ ");
        icc[36..40].copy_from_slice(b"acsp");
        icc[64..68].copy_from_slice(&intent.to_be_bytes());
        icc.extend((tags.len() as u32).to_be_bytes());
        icc.extend(table);
        icc.extend(body);
        let size = icc.len() as u32;
        icc[..4].copy_from_slice(&size.to_be_bytes());
        icc
    }

    #[test]
    fn standard_profiles_match_their_encoding_only() {
        let srgb = synthetic_profile(Primaries::Bt709, None, &[], 0);
        let p3 = synthetic_profile(Primaries::Smpte431, None, &[], 1);
        assert_eq!(
            matching_intent(&srgb, &ColorEncoding::srgb(), false),
            Some(RenderingIntent::Perceptual)
        );
        assert_eq!(
            matching_intent(&p3, &ColorEncoding::display_p3(), false),
            Some(RenderingIntent::Relative)
        );
        assert_eq!(matching_intent(&p3, &ColorEncoding::srgb(), false), None);
        assert_eq!(
            matching_intent(&srgb, &ColorEncoding::display_p3(), false),
            None
        );
        assert_eq!(
            matching_intent(&srgb, &ColorEncoding::srgb_linear(), false),
            None
        );
        // An RGB profile never stands for a gray image.
        assert_eq!(matching_intent(&srgb, &ColorEncoding::srgb(), true), None);
    }

    #[test]
    fn curve_forms_are_traced_not_compared_by_shape() {
        // A 1024-entry sampled sRGB curve, as v2 sRGB profiles carry.
        let mut table = b"curv\0\0\0\0".to_vec();
        table.extend(1024u32.to_be_bytes());
        for i in 0..1024 {
            let v = transfer_to_linear_f32(i as f32 / 1023.0, TransferFunction::Srgb);
            table.extend(((f64::from(v) * 65535.0).round() as u16).to_be_bytes());
        }
        let sampled = synthetic_profile(Primaries::Bt709, Some(&table), &[], 0);
        assert!(matching_intent(&sampled, &ColorEncoding::srgb(), false).is_some());
        // Gamma 2.2 is close to sRGB but not it.
        let gamma = b"curv\0\0\0\0\0\0\0\x01\x02\x33";
        let gamma = synthetic_profile(Primaries::Bt709, Some(gamma), &[], 0);
        assert_eq!(matching_intent(&gamma, &ColorEncoding::srgb(), false), None);
        // Identity curve: linear sRGB.
        let linear = synthetic_profile(Primaries::Bt709, Some(b"curv\0\0\0\0\0\0\0\0"), &[], 0);
        assert!(matching_intent(&linear, &ColorEncoding::srgb_linear(), false).is_some());
    }

    #[test]
    fn lut_profiles_and_garbage_never_match() {
        let lut = synthetic_profile(
            Primaries::Bt709,
            None,
            &[(*b"A2B0", b"mAB \0\0\0\0".to_vec())],
            0,
        );
        assert_eq!(matching_intent(&lut, &ColorEncoding::srgb(), false), None);
        let good = synthetic_profile(Primaries::Bt709, None, &[], 0);
        for cut in [0, 64, 131, good.len() / 2] {
            assert_eq!(
                matching_intent(&good[..cut], &ColorEncoding::srgb(), false),
                None
            );
        }
        let mut cmyk = good.clone();
        cmyk[16..20].copy_from_slice(b"CMYK");
        assert_eq!(matching_intent(&cmyk, &ColorEncoding::srgb(), false), None);
        assert_eq!(
            matching_intent(&[0u8; 600], &ColorEncoding::srgb(), false),
            None
        );
    }
}
