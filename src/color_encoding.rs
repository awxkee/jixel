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

/// JXL ColorSpace enum (2 bits — direct, no U32Coder).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ColorSpace {
    Rgb = 0,
    Gray = 1,
    Xyb = 2,
    Unknown = 3,
}

/// JXL WhitePoint enum (U32Coder, 4 enumerants in the named range).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    Dci = 11,
}

/// JXL Primaries enum (U32Coder).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Primaries {
    Srgb = 1,
    Custom = 2,
    Bt2100 = 9,
    P3 = 11,
}

/// JXL TransferFunction enum (U32Coder).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TransferFunction {
    Bt709 = 1,
    Unknown = 2,
    Linear = 8,
    Srgb = 13,
    Pq = 16,
    Dci = 17,
    Hlg = 18,
}

/// JXL RenderingIntent enum (U32Coder).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum RenderingIntent {
    Perceptual = 0,
    #[default]
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

/// CICP-style color encoding: which primaries + transfer + rendering intent
/// the original image was authored in. The white point is fixed by the
/// primaries for the named enumerants (e.g. sRGB primaries imply D65), but
/// JXL still carries it as a separate field, so we expose it.
#[derive(Debug, Clone, Copy)]
pub struct ColorEncoding {
    pub white_point: WhitePoint,
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub rendering_intent: RenderingIntent,
}

impl Default for ColorEncoding {
    fn default() -> Self {
        // We declare sRGB (not linear) so that libjxl's decoder applies the
        // sRGB OETF after XYB→linear conversion and the resulting PNG bytes
        // are sRGB-encoded — viewable correctly without ICC interpretation.
        // The encoder still operates on linear-light floats internally; this
        // is purely a metadata declaration about the output color space.
        Self::srgb()
    }
}

impl ColorEncoding {
    /// sRGB primaries + D65 + linear transfer + relative intent.
    /// This is what the previous monolithic `encode_file` wrote.
    pub const fn srgb_linear() -> Self {
        Self {
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            transfer: TransferFunction::Linear,
            rendering_intent: RenderingIntent::Relative,
        }
    }

    /// Standard sRGB (sRGB primaries + D65 + sRGB transfer).
    pub const fn srgb() -> Self {
        Self {
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            transfer: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Relative,
        }
    }

    /// Display P3 (P3 primaries + D65 + sRGB transfer).
    pub const fn display_p3() -> Self {
        Self {
            white_point: WhitePoint::D65,
            primaries: Primaries::P3,
            transfer: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Relative,
        }
    }

    /// Rec. 2020 / BT.2020 with PQ transfer (HDR10-style).
    pub const fn bt2020_pq() -> Self {
        Self {
            white_point: WhitePoint::D65,
            primaries: Primaries::Bt2100,
            transfer: TransferFunction::Pq,
            rendering_intent: RenderingIntent::Relative,
        }
    }
}

fn write_jxl_enum(value: u32, w: &mut BitWriter) {
    if value == 0 {
        w.write(2, 0);
    } else if value == 1 {
        w.write(2, 1);
    } else if (2..18).contains(&value) {
        w.write(2, 2);
        w.write(4, (value - 2) as u64);
    } else if (18..82).contains(&value) {
        w.write(2, 3);
        w.write(6, (value - 18) as u64);
    } else {
        panic!("enum value {value} out of JXL U32Coder range [0, 82)");
    }
}

/// Write the JXL `ColorEncoding` field for an XYB-encoded image with no
/// ICC profile.
///
/// Field order matches `ColorEncoding` in the JXL spec:
///   1. `all_default` (1 bit) — we always write 0 (custom).
///   2. `want_icc` (1 bit) — 0; we don't support ICC injection yet.
///   3. `color_space` (2 bits) — RGB.
///   4. `white_point` (enum) — only present if `!want_icc && color_space != XYB`.
///   5. `primaries` (enum) — same condition + `color_space != Gray`.
///   6. `tf.have_gamma` (1 bit) — 0; we always select a named transfer fn.
///   7. `tf.transfer_function` (enum) — only present if `!want_icc`.
///   8. `rendering_intent` (enum) — only present if `!want_icc`.
pub(crate) fn write_color_encoding_with_icc(
    enc: &ColorEncoding,
    want_icc: bool,
    grayscale: bool,
    w: &mut BitWriter,
) {
    w.write(1, 0); // all_default = false
    w.write(1, if want_icc { 1 } else { 0 }); // want_icc
    let color_space = if grayscale {
        ColorSpace::Gray
    } else {
        ColorSpace::Rgb
    };
    w.write(2, color_space as u64); // color_space
    if want_icc {
        // Remaining fields gated on !want_icc; they're absent from the stream.
        let _ = enc;
        return;
    }
    write_jxl_enum(enc.white_point as u32, w);
    if !grayscale {
        // Primaries are only present for non-gray color spaces.
        write_jxl_enum(enc.primaries as u32, w);
    }
    w.write(1, 0); // have_gamma = false
    write_jxl_enum(enc.transfer as u32, w);
    write_jxl_enum(enc.rendering_intent as u32, w);
}
