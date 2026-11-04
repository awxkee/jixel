/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

//! Builds a JXL VarDCT frame directly from JPEG DCT coefficients.

mod ctx_opt;

use super::{DCT_BLOCK_SIZE, JpegData, JpegError, coeff_order};
use crate::ac_context::{
    K_NON_ZERO_BUCKETS, K_NUM_ORDERS, K_ZERO_DENSITY_CONTEXT_COUNT, non_zero_bucket,
    zero_density_context_8x8,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::entropy::AnsRefinement;
use crate::entropy::{
    Token, optimize_entropy_code, pack_signed, write_ans_tokens, write_entropy_code, write_token,
};
use crate::frame::{
    collect_ac_metadata_tokens, collect_dc_tokens_with_gradient, combine_sections,
    combined_sections_size, write_context_tree, write_quant_scales,
};
use crate::image::Image3B;
use crate::static_entropy_codes::K_NUM_DC_CONTEXTS;
use crate::thread_pool::ThreadPool;
use crate::util::EncodeError;
use crate::{Orientation, Speed};

const BLOCK_DIM: usize = 8;
const GROUP_DIM: usize = 256;
const GROUP_DIM_IN_BLOCKS: usize = GROUP_DIM / BLOCK_DIM;
const DC_GROUP_DIM_IN_BLOCKS: usize = 256;

/// Zig-zag order over an 8×8 block, as JXL indexes coefficients.
use crate::ac_context::K_COEFF_ORDER_8X8;

/// The JXL channel order used when interleaving tokens: Y, X, B.
static CHANNEL_ORDER: [usize; 3] = [1, 0, 2];

/// JXL channel (X, Y, B) to JPEG component (Cb, Y, Cr).
static JPEG_ORDER_YCBCR: [usize; 3] = [1, 0, 2];
/// A grayscale JPEG's single component serves every channel's quant table.
static JPEG_ORDER_GRAY: [usize; 3] = [0, 0, 0];
/// RGB JPEGs are coded without the YCbCr transform, channels in R, G, B order.
static JPEG_ORDER_RGB: [usize; 3] = [0, 1, 2];

/// Geometry of the frame, in pixels, blocks and groups.
struct Dim {
    xsize: usize,
    ysize: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    xsize_groups: usize,
    num_groups: usize,
    xsize_dc_groups: usize,
    num_dc_groups: usize,
}

impl Dim {
    fn new(xsize: usize, ysize: usize, ss: &ChannelLayout) -> Self {
        // Rounded up to a whole MCU so halving it for chroma is exact.
        let xsize_blocks = xsize.div_ceil(BLOCK_DIM << ss.max_hshift) << ss.max_hshift;
        let ysize_blocks = ysize.div_ceil(BLOCK_DIM << ss.max_vshift) << ss.max_vshift;
        let xsize_groups = xsize.div_ceil(GROUP_DIM);
        let ysize_groups = ysize.div_ceil(GROUP_DIM);
        let xsize_dc_groups = xsize_blocks.div_ceil(DC_GROUP_DIM_IN_BLOCKS);
        let ysize_dc_groups = ysize_blocks.div_ceil(DC_GROUP_DIM_IN_BLOCKS);
        Self {
            xsize,
            ysize,
            xsize_blocks,
            ysize_blocks,
            xsize_groups,
            num_groups: xsize_groups * ysize_groups,
            xsize_dc_groups,
            num_dc_groups: xsize_dc_groups * ysize_dc_groups,
        }
    }
}

/// How the JPEG's components map onto the frame's channels, expressed the way
/// the frame header wants it.
struct ChannelLayout {
    /// Per-JXL-channel horizontal and vertical shifts (0 = full resolution).
    hshift: [usize; 3],
    vshift: [usize; 3],
    /// The 2-bit mode written for each channel, in X, Y, B order.
    mode: [u64; 3],
    max_hshift: usize,
    max_vshift: usize,
    /// One component: coded as Y with all-zero chroma, as libjxl does.
    gray: bool,
    /// JXL channel to JPEG component.
    jpeg_order: [usize; 3],
    /// Whether the frame applies the YCbCr transform; false for RGB JPEGs.
    ycbcr: bool,
    /// Added to each channel's quantized DC: without the YCbCr transform the
    /// decoder subtracts `1024 / dc_quant` to recenter the samples.
    dc_offset: [i16; 3],
}

impl ChannelLayout {
    /// The JPEG component behind JXL channel `c`; `None` for the chroma of a
    /// grayscale JPEG, whose coefficients are all zero.
    #[inline]
    fn component<'a>(&self, jpg: &'a JpegData, c: usize) -> Option<&'a super::JpegComponent> {
        (!self.gray || c == 1).then(|| &jpg.components[self.jpeg_order[c]])
    }

    #[inline]
    fn luma<'a>(&self, jpg: &'a JpegData) -> &'a super::JpegComponent {
        &jpg.components[self.jpeg_order[1]]
    }
}

/// The frame header's per-channel subsampling mode for JPEG sampling factors
/// `(h, v)`; the decoder reconstructs the factors from it.
fn sampling_mode(h: usize, v: usize) -> Option<u64> {
    match (h, v) {
        (1, 1) => Some(0),
        (2, 2) => Some(1),
        (2, 1) => Some(2),
        (1, 2) => Some(3),
        _ => None,
    }
}

/// Works out the channel layout, rejecting anything JXL cannot express: only
/// chroma at full resolution or halved in either axis has a representation.
fn check_supported(jpg: &JpegData) -> Result<ChannelLayout, JpegError> {
    if jpg.components.len() == 1 {
        // A lone component is full resolution, but its sampling factors still
        // shape the MCU grid, and the decoder rebuilds them from the modes:
        // every channel carries the same one, so no shift is effective.
        let comp = &jpg.components[0];
        let mode = sampling_mode(comp.h_samp_factor, comp.v_samp_factor).ok_or(
            JpegError::UnsupportedMode("sampling factors above 2 are not representable"),
        )?;
        return Ok(ChannelLayout {
            hshift: [0; 3],
            vshift: [0; 3],
            mode: [mode; 3],
            max_hshift: comp.h_samp_factor.trailing_zeros() as usize,
            max_vshift: comp.v_samp_factor.trailing_zeros() as usize,
            gray: true,
            jpeg_order: JPEG_ORDER_GRAY,
            ycbcr: true,
            dc_offset: [0; 3],
        });
    }
    if jpg.components.len() != 3 {
        return Err(JpegError::UnsupportedMode(
            "only 1- and 3-component JPEGs are transcodable",
        ));
    }
    let max_h = jpg.max_h_samp();
    let max_v = jpg.max_v_samp();
    if is_rgb(jpg) {
        // Chroma subsampling is only signaled alongside the YCbCr transform.
        if max_h != 1 || max_v != 1 {
            return Err(JpegError::UnsupportedMode(
                "subsampled RGB JPEGs are not representable",
            ));
        }
        let dc_offset = JPEG_ORDER_RGB.map(|i| {
            let dc_quant = jpg.quant[jpg.components[i].quant_idx as usize].values[0];
            (1024 / dc_quant.max(1)) as i16
        });
        return Ok(ChannelLayout {
            hshift: [0; 3],
            vshift: [0; 3],
            mode: [0; 3],
            max_hshift: 0,
            max_vshift: 0,
            gray: false,
            jpeg_order: JPEG_ORDER_RGB,
            ycbcr: false,
            dc_offset,
        });
    }
    if !matches!(max_h, 1 | 2) || !matches!(max_v, 1 | 2) {
        return Err(JpegError::UnsupportedMode(
            "sampling factors above 2 are not representable",
        ));
    }

    // Each channel signals its own JPEG sampling factors; the decoder derives
    // the effective shifts from them and rebuilds the factors for the JPEG.
    let mut hshift = [0usize; 3];
    let mut vshift = [0usize; 3];
    let mut mode = [0u64; 3];
    for c in 0..3usize {
        let comp = &jpg.components[JPEG_ORDER_YCBCR[c]];
        mode[c] = sampling_mode(comp.h_samp_factor, comp.v_samp_factor).ok_or(
            JpegError::UnsupportedMode("sampling factors above 2 are not representable"),
        )?;
        hshift[c] = (max_h / comp.h_samp_factor).trailing_zeros() as usize;
        vshift[c] = (max_v / comp.v_samp_factor).trailing_zeros() as usize;
    }
    // Luma must be full resolution.
    if hshift[1] != 0 || vshift[1] != 0 {
        return Err(JpegError::UnsupportedMode(
            "chroma at a higher resolution than luma is not supported",
        ));
    }
    // The block grid follows the MCU, so it is rounded by the largest factor
    // even when every channel shares it.
    let max_hshift = max_h.trailing_zeros() as usize;
    let max_vshift = max_v.trailing_zeros() as usize;

    Ok(ChannelLayout {
        hshift,
        vshift,
        mode,
        max_hshift,
        max_vshift,
        gray: false,
        jpeg_order: JPEG_ORDER_YCBCR,
        ycbcr: true,
        dc_offset: [0; 3],
    })
}

/// Whether a 3-component JPEG holds RGB rather than YCbCr, decided as libjxl
/// does: a JFIF marker means YCbCr; otherwise the first Adobe APP14 marker's
/// transform flag, and failing that, component ids 'R', 'G', 'B'.
fn is_rgb(jpg: &JpegData) -> bool {
    if jpg.marker_order.contains(&0xE0) {
        return false;
    }
    let mut app = jpg.app_data.iter();
    for &marker in &jpg.marker_order {
        if marker & 0xF0 != 0xE0 {
            continue;
        }
        let Some(data) = app.next() else {
            break;
        };
        // [marker, len_hi, len_lo, "Adobe", version, flags0, flags1, transform]
        if marker == 0xEE && data.len() == 15 && &data[3..8] == b"Adobe" {
            return data[14] == 0;
        }
    }
    let ids: Vec<u32> = jpg.components.iter().map(|c| c.id).collect();
    ids == [u32::from(b'R'), u32::from(b'G'), u32::from(b'B')]
}

/// Writes an IEEE half-precision float exactly as the JXL field coder does.
fn write_f16(w: &mut BitWriter, value: f32) -> Result<(), JpegError> {
    let bits = f32::to_bits(value);
    let sign = bits >> 31;
    let biased_exp32 = (bits >> 23) & 0xFF;
    let mantissa32 = bits & 0x7F_FFFF;
    let exp = biased_exp32 as i32 - 127;

    if exp > 15 {
        return Err(JpegError::UnsupportedMode(
            "quantization value out of half-precision range",
        ));
    }
    // Anything below the smallest subnormal collapses to zero.
    if exp < -24 {
        w.write(16, 0);
        return Ok(());
    }

    let (biased_exp16, mantissa16) = if exp < -14 {
        let sub_exp = (-14 - exp) as u32;
        (0u32, (1 << (10 - sub_exp)) + (mantissa32 >> (13 + sub_exp)))
    } else {
        ((exp + 15) as u32, mantissa32 >> 13)
    };

    w.write(
        16,
        ((sign << 15) | (biased_exp16 << 10) | mantissa16) as u64,
    );
    Ok(())
}

/// Writes the `SizeHeader` bundle.
fn write_size(w: &mut BitWriter, size: usize) {
    let v = (size - 1) as u64;
    const WIDTHS: [usize; 4] = [9, 13, 18, 30];
    for (i, &bits) in WIDTHS.iter().enumerate() {
        if v < (1u64 << bits) {
            w.write(2, i as u64);
            w.write(bits, v);
            return;
        }
    }
    unreachable!("dimension exceeds the maximum representable size");
}

/// Aspect ratios the size header can encode instead of a literal width,
/// as (numerator, denominator) pairs for ratio codes 1..=7.
static ASPECT_RATIOS: [(usize, usize); 7] =
    [(1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1)];

fn write_size_header(w: &mut BitWriter, xsize: usize, ysize: usize) {
    // A height that is a multiple of 8 and at most 256 has a compact form.
    let small =
        ysize.is_multiple_of(8) && (1..=32).contains(&(ysize / 8)) && xsize.is_multiple_of(8);

    // A standard aspect ratio means the width need not be coded at all.
    let ratio = ASPECT_RATIOS
        .iter()
        .position(|&(n, d)| ysize * n / d == xsize && (ysize * n).is_multiple_of(d))
        .map(|i| i + 1)
        .unwrap_or(0);

    let small = small && (ratio != 0 || (1..=32).contains(&(xsize / 8)));

    if small {
        w.write(1, 1);
        w.write(5, (ysize / 8 - 1) as u64);
        w.write(3, ratio as u64);
        if ratio == 0 {
            w.write(5, (xsize / 8 - 1) as u64);
        }
    } else {
        w.write(1, 0);
        write_size(w, ysize);
        w.write(3, ratio as u64);
        if ratio == 0 {
            write_size(w, xsize);
        }
    }
}

/// Writes `ImageMetadata`. The key choice is `xyb_encoded = 0`: the frame
/// carries the JPEG's own channels, so the decoder must not undo XYB.
fn write_image_metadata(
    w: &mut BitWriter,
    icc: Option<&[u8]>,
    gray: bool,
    orientation: Orientation,
    scratch: &mut CoderScratch,
) {
    w.write(1, 0); // not all-default
    // extra_fields carries the orientation (and preview/animation, never set).
    let extra_fields = orientation != Orientation::Normal;
    w.write(1, u64::from(extra_fields));
    if extra_fields {
        w.write(3, orientation.to_u3());
        w.write(1, 0); // have_intrinsic_size
        w.write(1, 0); // have_preview
        w.write(1, 0); // have_animation
    }
    w.write(1, 0); // floating_point_sample = false
    w.write(2, 0); // bits_per_sample = 8
    w.write(1, 1); // modular_16bit_buffer_sufficient
    w.write(2, 0); // num_extra_channels = 0
    w.write(1, 0); // xyb_encoded = 0
    match icc {
        // The JPEG's own profile, so the decoded pixels are interpreted the
        // same way whether reconstruction data is kept.
        Some(_) => crate::color_encoding::write_color_encoding_with_icc(
            &crate::ColorEncoding::default(),
            true,
            gray,
            w,
        ),
        None if gray => crate::color_encoding::write_color_encoding_with_icc(
            &crate::ColorEncoding::default(),
            false,
            true,
            w,
        ),
        None => w.write(1, 1), // color encoding: all default (sRGB)
    }
    if extra_fields {
        w.write(1, 1); // tone mapping: all default
    }
    w.write(2, 0); // no extensions
    w.write(1, 1); // CustomTransformData: all default
    if let Some(icc) = icc {
        crate::icc_codec::write_icc_stream(icc, &mut scratch.huffman_pool, w);
    }
    w.zero_pad_to_byte();
}

/// Writes the frame header.
fn write_frame_header(w: &mut BitWriter, ss: &ChannelLayout) {
    w.write(1, 0); // not all-default
    w.write(2, 0); // regular frame
    w.write(1, 0); // encoding = VarDCT

    // flags = 128 (kSkipAdaptiveDCSmoothing): smoothing would perturb the DC
    // and break the round-trip. U64 coder, 17..=272 is selector 2 + 8 bits.
    w.write(2, 2);
    w.write(8, 128 - 17);

    // do_ycbcr (serialized because xyb_encoded = 0).
    w.write(1, u64::from(ss.ycbcr));
    if ss.ycbcr {
        // YCbCrChromaSubsampling, one 2-bit mode per channel in X, Y, B order.
        for mode in ss.mode {
            w.write(2, mode);
        }
    }

    w.write(2, 0); // upsampling = 1
    // x_qm_scale / b_qm_scale are only serialized for XYB frames.
    w.write(2, 0); // num_passes = 1
    w.write(1, 0); // no custom size or origin
    w.write(2, 0); // blending = Replace
    w.write(1, 1); // is_last
    w.write(2, 0); // no name

    // Gaborish and EPF must both be off; either would alter the samples.
    w.write(1, 0); // not default
    w.write(1, 0); // gaborish off
    w.write(2, 0); // epf_iters = 0
    w.write(2, 0); // no loop-filter extensions

    w.write(2, 0); // no frame-header extensions
}

/// Fixed-point precision of the JPEG chroma-from-luma multiply.
const CFL_PRECISION: u32 = 11;
/// CfL factors are signaled in units of 1/84 (`color_factor`).
const CFL_COLOR_FACTOR: i32 = 84;
/// Chroma-from-luma tiles are 8x8 blocks.
const CFL_TILE_DIM_IN_BLOCKS: usize = 8;

/// Integer chroma-from-luma for 4:4:4 JPEGs, exactly as the decoder's JPEG
/// reconstruction applies it: each chroma coefficient is coded as a residual
/// against a fixed-point multiple of the co-located luma coefficient.
struct JpegCfl {
    /// Luma-to-chroma quant step ratio per JPEG raster position, in
    /// `CFL_PRECISION` fixed point, per JXL channel (luma row unused).
    scaled_qtable: [[i32; DCT_BLOCK_SIZE]; 3],
    xtiles: usize,
    /// Per-tile factors for X (Cb) and B (Cr).
    maps: [Vec<i8>; 2],
}

impl JpegCfl {
    fn new(
        jpg: &JpegData,
        dim: &Dim,
        ss: &ChannelLayout,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Option<Self> {
        if ss.gray || ss.hshift.iter().chain(&ss.vshift).any(|&s| s != 0) {
            return None;
        }
        let quant =
            |c: usize| &jpg.quant[jpg.components[ss.jpeg_order[c]].quant_idx as usize].values;
        let mut scaled_qtable = [[0i32; DCT_BLOCK_SIZE]; 3];
        for c in [0usize, 2] {
            for i in 0..DCT_BLOCK_SIZE {
                let (num, den) = (i64::from(quant(1)[i]), i64::from(quant(c)[i]));
                if num <= 0 || den <= 0 {
                    return None;
                }
                let ratio = (num << CFL_PRECISION) / den;
                // libjxl's bound; beyond it the fixed-point multiply can overflow.
                if ratio > (256 << CFL_PRECISION) - 1 {
                    return None;
                }
                scaled_qtable[c][i] = ratio as i32;
            }
        }
        let xtiles = dim.xsize_blocks.div_ceil(CFL_TILE_DIM_IN_BLOCKS);
        let ytiles = dim.ysize_blocks.div_ceil(CFL_TILE_DIM_IN_BLOCKS);
        let luma = ss.luma(jpg);
        let mut maps = [Vec::new(), Vec::new()];
        for (slot, c) in [0usize, 2].into_iter().enumerate() {
            let chroma = &jpg.components[ss.jpeg_order[c]];
            let qt = &scaled_qtable[c];
            let rows = pool.steal_map(scratch, ytiles, |ty, _scratch| {
                let mut row = vec![0i8; xtiles];
                for (tx, out) in row.iter_mut().enumerate() {
                    *out = Self::fit_tile(luma, chroma, qt, dim, tx, ty);
                }
                row
            });
            maps[slot] = rows.into_iter().flatten().collect();
        }
        if maps.iter().all(|m| m.iter().all(|&v| v == 0)) {
            return None;
        }
        Some(Self {
            scaled_qtable,
            xtiles,
            maps,
        })
    }

    /// libjxl's tile search: the factor that zeroes the most chroma residuals,
    /// kept only if it beats factor 0 by more than one coefficient.
    fn fit_tile(
        luma: &super::JpegComponent,
        chroma: &super::JpegComponent,
        qt: &[i32; DCT_BLOCK_SIZE],
        dim: &Dim,
        tx: usize,
        ty: usize,
    ) -> i8 {
        // Index i is factor i - 128, so the 256 slots are exactly the i8 range.
        // libjxl offsets by 127, making its top slot a factor of 128 that wraps
        // to -128 when stored.
        const OFFSET: i32 = 128;
        let scale = CFL_COLOR_FACTOR as f32;
        let zero_thresh = scale * 0.5 * 0.9999;
        let mut d_num_zeros = [0i32; 257];
        let y0 = ty * CFL_TILE_DIM_IN_BLOCKS;
        let x0 = tx * CFL_TILE_DIM_IN_BLOCKS;
        let y1 = dim.ysize_blocks.min(y0 + CFL_TILE_DIM_IN_BLOCKS);
        let x1 = dim.xsize_blocks.min(x0 + CFL_TILE_DIM_IN_BLOCKS);
        for by in y0..y1 {
            for bx in x0..x1 {
                let m = &luma.coeffs[(by * luma.width_in_blocks + bx) * DCT_BLOCK_SIZE..]
                    [..DCT_BLOCK_SIZE];
                let s = &chroma.coeffs[(by * chroma.width_in_blocks + bx) * DCT_BLOCK_SIZE..]
                    [..DCT_BLOCK_SIZE];
                for k in 1..DCT_BLOCK_SIZE {
                    let scaled_m =
                        f32::from(m[k]) * (1.0 / (1 << CFL_PRECISION) as f32) * qt[k] as f32;
                    if scaled_m.abs() <= 1e-8 {
                        continue;
                    }
                    let scaled_s = scale * f32::from(s[k]) + OFFSET as f32 * scaled_m;
                    let (mut from, mut to) = if scaled_m > 0.0 {
                        (
                            (scaled_s - zero_thresh) / scaled_m,
                            (scaled_s + zero_thresh) / scaled_m,
                        )
                    } else {
                        (
                            (scaled_s + zero_thresh) / scaled_m,
                            (scaled_s - zero_thresh) / scaled_m,
                        )
                    };
                    from = from.max(0.0);
                    to = to.min(255.0);
                    if from <= to {
                        // Both bounds lie in [0, 256], where truncation is floor;
                        // f32::ceil/floor are libm calls on baseline x86.
                        let lo = from as usize;
                        let lo = lo + usize::from((lo as f32) < from);
                        d_num_zeros[lo] += 1;
                        d_num_zeros[(to + 1.0) as usize] -= 1;
                    }
                }
            }
        }
        let (mut best_sum, mut val) = (0i32, 0i32);
        let (mut begin, mut end) = (0usize, 0usize);
        for (i, &d) in d_num_zeros[..256].iter().enumerate() {
            val += d;
            if val > best_sum {
                best_sum = val;
                begin = i;
            }
            if val == best_sum {
                end = i;
            }
        }
        let best = ((begin + end + 1) >> 1) as i32;
        let offset_sum: i32 = d_num_zeros[..=OFFSET as usize].iter().sum();
        if best_sum > offset_sum + 1 {
            (best - OFFSET) as i8
        } else {
            0
        }
    }

    #[inline]
    fn factor(&self, c: usize, bx: usize, by: usize) -> i32 {
        let tile = (by / CFL_TILE_DIM_IN_BLOCKS) * self.xtiles + bx / CFL_TILE_DIM_IN_BLOCKS;
        i32::from(self.maps[c / 2][tile])
    }

    /// The coded chroma block (JPEG raster order) for the block at `(bx, by)`.
    #[inline]
    fn residual(
        &self,
        c: usize,
        bx: usize,
        by: usize,
        y: &[i16],
        chroma: &[i16],
    ) -> [i32; DCT_BLOCK_SIZE] {
        let scale = self.factor(c, bx, by) * (1 << CFL_PRECISION) / CFL_COLOR_FACTOR;
        let round = 1 << (CFL_PRECISION - 1);
        let qt = &self.scaled_qtable[c];
        let mut out = [0i32; DCT_BLOCK_SIZE];
        for k in 0..DCT_BLOCK_SIZE {
            let coeff_scale = (scale * qt[k] + round) >> CFL_PRECISION;
            let cfl = (i32::from(y[k]) * coeff_scale + round) >> CFL_PRECISION;
            out[k] = i32::from(chroma[k]) - cfl;
        }
        out
    }
}

/// The coefficients of one block as coded (CfL residual for 4:4:4 chroma), in
/// JPEG raster order. `(bx, by)` are in the channel's own block grid.
#[inline]
fn coded_block(
    jpg: &JpegData,
    ss: &ChannelLayout,
    cfl: Option<&JpegCfl>,
    c: usize,
    bx: usize,
    by: usize,
) -> [i32; DCT_BLOCK_SIZE] {
    let Some(comp) = ss.component(jpg, c) else {
        return [0; DCT_BLOCK_SIZE];
    };
    let src = &comp.coeffs[(by * comp.width_in_blocks + bx) * DCT_BLOCK_SIZE..][..DCT_BLOCK_SIZE];
    match cfl {
        Some(cfl) if c != 1 => {
            let luma = ss.luma(jpg);
            let y =
                &luma.coeffs[(by * luma.width_in_blocks + bx) * DCT_BLOCK_SIZE..][..DCT_BLOCK_SIZE];
            cfl.residual(c, bx, by, y, src)
        }
        _ => std::array::from_fn(|k| i32::from(src[k])),
    }
}

/// Block contexts keyed by the quantized DC values of all three channels.
/// Each plane's thresholds split its DC range into intervals; their product
/// forms the DC cells, and `ctx_map` gives every (channel, cell) a context.
struct BlockCtx {
    /// DC thresholds per JXL plane (X, Y, B); a block's interval on a plane
    /// counts the thresholds below its DC there.
    thresholds: [Vec<i32>; 3],
    num_dc_ctxs: usize,
    /// Indexed by `(row * K_NUM_ORDERS + order) * num_dc_ctxs + dc_idx`, rows
    /// in Y, X, B order.
    ctx_map: Vec<u8>,
    num_ctxs: usize,
}

impl BlockCtx {
    /// libjxl's JPEG layout: one context per luma DC bucket, chroma buckets
    /// paired up.
    fn luma_quantiles(
        jpg: &JpegData,
        dim: &Dim,
        ss: &ChannelLayout,
        qtable: &[i32; 3 * DCT_BLOCK_SIZE],
    ) -> Self {
        let luma = ss.luma(jpg);
        let mut counts = vec![0u32; 2048];
        for by in 0..dim.ysize_blocks {
            for bx in 0..dim.xsize_blocks {
                // As coded: the decoder buckets the DC image's values.
                let dc = luma.coeffs[(by * luma.width_in_blocks + bx) * DCT_BLOCK_SIZE]
                    + ss.dc_offset[1];
                counts[(i32::from(dc) + 1024).clamp(0, 2047) as usize] += 1;
            }
        }
        let total = dim.xsize_blocks * dim.ysize_blocks;
        let ceil_log2 = |v: usize| v.max(1).next_power_of_two().trailing_zeros() as i32;
        // More buckets for larger and higher-quality images.
        let qsum: i32 = qtable[1..6].iter().sum();
        let num_thresholds = (ceil_log2(total) - ceil_log2(qsum.max(1) as usize) - 7).clamp(1, 7);
        let mut luma_thresholds = Vec::new();
        let mut cumsum = 0usize;
        let mut cut = total / (num_thresholds as usize + 1);
        for (j, &n) in counts.iter().enumerate() {
            cumsum += n as usize;
            if cumsum > cut {
                luma_thresholds.push(j as i32 - 1025);
                cut = total * (luma_thresholds.len() + 1) / (num_thresholds as usize + 1);
            }
        }
        let num_dc = luma_thresholds.len() + 1;
        Self::from_cells([Vec::new(), luma_thresholds, Vec::new()], |row, i| {
            match row {
                0 => i as u8,
                // All-zero chroma shares a single context.
                _ if ss.gray => num_dc as u8,
                1 => (num_dc + i / 2) as u8,
                _ => (num_dc + (num_dc - 1) / 2 + 1 + i / 2) as u8,
            }
        })
    }

    /// Fills the DCT8 entries of every map row (Y, X, B) from `cell_ctx`.
    fn from_cells(thresholds: [Vec<i32>; 3], cell_ctx: impl Fn(usize, usize) -> u8) -> Self {
        let num_dc_ctxs = thresholds.iter().map(|t| t.len() + 1).product::<usize>();
        let mut ctx_map = vec![0u8; 3 * K_NUM_ORDERS * num_dc_ctxs];
        for row in 0..3 {
            for i in 0..num_dc_ctxs {
                ctx_map[row * K_NUM_ORDERS * num_dc_ctxs + i] = cell_ctx(row, i);
            }
        }
        let num_ctxs = *ctx_map.iter().max().unwrap() as usize + 1;
        debug_assert!(num_ctxs <= 16 && num_dc_ctxs <= 64);
        Self {
            thresholds,
            num_dc_ctxs,
            ctx_map,
            num_ctxs,
        }
    }

    fn num_ac_contexts(&self) -> usize {
        self.num_ctxs * (K_NON_ZERO_BUCKETS + K_ZERO_DENSITY_CONTEXT_COUNT)
    }

    /// DC cell of a block whose co-located quantized DC values are `dc`, in
    /// the decoder's X-major, then B, then Y nesting.
    #[inline]
    fn dc_idx(&self, dc: [i32; 3]) -> usize {
        let interval = |p: usize| self.thresholds[p].iter().filter(|&&t| dc[p] > t).count();
        (interval(0) * (self.thresholds[2].len() + 1) + interval(2))
            * (self.thresholds[1].len() + 1)
            + interval(1)
    }

    /// Block context of a DCT8 block in JXL channel `c`.
    #[inline]
    fn context(&self, c: usize, dc: [i32; 3]) -> u32 {
        let row = [1, 0, 2][c];
        u32::from(self.ctx_map[row * K_NUM_ORDERS * self.num_dc_ctxs + self.dc_idx(dc)])
    }

    #[inline]
    fn non_zero_context(&self, predicted: u32, block_ctx: u32) -> u32 {
        non_zero_bucket(predicted) * self.num_ctxs as u32 + block_ctx
    }

    #[inline]
    fn zero_density_offset(&self, block_ctx: u32) -> u32 {
        (self.num_ctxs * K_NON_ZERO_BUCKETS) as u32
            + K_ZERO_DENSITY_CONTEXT_COUNT as u32 * block_ctx
    }

    fn write(&self, scratch: &mut CoderScratch, w: &mut BitWriter) {
        w.write(1, 0); // non-default BlockCtxMap
        for thresholds in &self.thresholds {
            w.write(4, thresholds.len() as u64);
            for &t in thresholds {
                // kDCThresholdDist: U32(Bits(4), BitsOffset(8,16), BitsOffset(16,272),
                // BitsOffset(32,65808)) over PackSigned(t).
                let v = pack_signed(t);
                if v < 16 {
                    w.write(2, 0);
                    w.write(4, u64::from(v));
                } else if v < 272 {
                    w.write(2, 1);
                    w.write(8, u64::from(v - 16));
                } else if v < 65808 {
                    w.write(2, 2);
                    w.write(16, u64::from(v - 272));
                } else {
                    w.write(2, 3);
                    w.write(32, u64::from(v - 65808));
                }
            }
        }
        w.write(4, 0); // no quant-field thresholds
        let empty_codes: [crate::entropy::PrefixCode; 0] = [];
        let empty_configs: [crate::entropy::HybridUintConfig; 0] = [];
        let empty_histograms = [];
        let empty_syms: [Vec<crate::entropy::AnsEncSymbolInfo>; 0] = [];
        let empty_reverse_maps: [u16; 0] = [];
        let cm_entropy = crate::entropy::EntropyCode {
            context_map: &self.ctx_map,
            num_contexts: self.ctx_map.len(),
            prefix_codes: &empty_codes,
            hybrid_uint_configs: &empty_configs,
            num_prefix_codes: 0,
            orig_context_map: None,
            orig_num_contexts: 0,
            use_prefix_code: true,
            ans_histograms: &empty_histograms,
            ans_symbols: &empty_syms,
            ans_reverse_maps: &empty_reverse_maps,
        };
        crate::entropy::write_context_map(&cm_entropy, &mut scratch.huffman_pool, w);
    }
}

/// Writes the `LfChannelDequantization` bundle carrying the JPEG's DC quant.
fn write_dc_quant(w: &mut BitWriter, dc_quant: [f32; 3]) -> Result<(), JpegError> {
    w.write(1, 0); // not all-default
    for v in dc_quant {
        write_f16(w, v * 128.0)?;
    }
    Ok(())
}

/// Writes the color-correlation bundle, pinned to the neutral configuration
/// the decoder demands for JPEG reconstruction.
fn write_color_correlation(w: &mut BitWriter) {
    w.write(1, 0); // not all-default: base_correlation_b differs from the XYB one
    w.write(2, 0); // color_factor = 84 (the direct branch)
    w.write(16, 0); // base_correlation_x = 0.0
    w.write(16, 0); // base_correlation_b = 0.0
    w.write(8, 128); // ytox_dc = 0, offset by 128
    w.write(8, 128); // ytob_dc = 0, offset by 128
}

/// Writes the JPEG's DQT as a raw quantization matrix. Only table 0 (the 8x8
/// DCT) is used, so the rest stay on their library defaults.
fn write_dequant_matrices(
    w: &mut BitWriter,
    qtable: &[i32; 3 * DCT_BLOCK_SIZE],
    scratch: &mut CoderScratch,
) {
    w.write(1, 0); // not all-default
    for table in 0..17usize {
        if table == 0 {
            w.write(3, 7); // kQuantModeRAW
            // Fixed at 1/(8*255); the decoder checks it within 1e-8, so the
            // exact half-precision pattern matters.
            w.write(16, 0x1004);
            write_raw_quant_image(w, qtable, scratch);
        } else {
            w.write(3, 0); // kQuantModeLibrary
            // the predefined index occupies zero bits
        }
    }
}

/// Emits the 8x8x3 modular sub-image of raw quantization values, where channel
/// `c` row `y` column `x` is `qtable[c*64 + y*8 + x]`.
fn write_raw_quant_image(
    w: &mut BitWriter,
    qtable: &[i32; 3 * DCT_BLOCK_SIZE],
    scratch: &mut CoderScratch,
) {
    crate::modular::write_group_header_local_tree(w);

    let mut tokens: Vec<Token> = Vec::with_capacity(3 * DCT_BLOCK_SIZE);
    for c in 0..3usize {
        let plane = &qtable[c * DCT_BLOCK_SIZE..(c + 1) * DCT_BLOCK_SIZE];
        for y in 0..8usize {
            for x in 0..8usize {
                let at = |xx: usize, yy: usize| plane[yy * 8 + xx];
                let west = if x > 0 { at(x - 1, y) } else { 0 };
                let north = if y > 0 { at(x, y - 1) } else { 0 };
                let northwest = if x > 0 && y > 0 { at(x - 1, y - 1) } else { 0 };
                let pred = crate::modular::gradient(west, north, northwest);
                tokens.push(Token::new(0, pack_signed(at(x, y) - pred)));
            }
        }
    }

    let code = crate::modular::build_pixel_code(&tokens, scratch);
    crate::modular::write_tree_and_pixel_histograms(&code, scratch, w);
    let code_ref = code.as_ref();
    for t in &tokens {
        write_token(*t, &code_ref, w);
    }
}

/// Entropy-coding effort per speed tier; the coefficients are fixed.
#[derive(Clone, Copy)]
struct Effort {
    /// Also code the natural coefficient order and keep the cheaper one.
    try_natural_order: bool,
    /// Race the fixed DC tree against a per-image learned tree.
    learned_dc_tree: bool,
    /// Race the default DC weighted-predictor parameters against libjxl's
    /// second preset on the serialized sections.
    dc_wp_race: bool,
    /// Pick between them on the static tree's Shannon bits instead and code
    /// the pick once.
    dc_wp_proxy_pick: bool,
    /// Race an entropy-optimized block context map against the luma-quantile one.
    optimized_ctx_map: bool,
    /// Pick per-cluster HybridUint configs instead of the default.
    select_configs: bool,
    refinement: Option<AnsRefinement>,
}

impl Effort {
    fn new(speed: Speed) -> Self {
        match speed {
            Speed::UltraFast => Self {
                try_natural_order: false,
                learned_dc_tree: false,
                dc_wp_race: false,
                dc_wp_proxy_pick: false,
                optimized_ctx_map: false,
                select_configs: false,
                refinement: None,
            },
            Speed::Fastest => Self {
                try_natural_order: false,
                learned_dc_tree: false,
                dc_wp_race: false,
                dc_wp_proxy_pick: false,
                optimized_ctx_map: false,
                select_configs: false,
                refinement: Some(AnsRefinement::Fast { recluster: false }),
            },
            Speed::Fast => Self {
                try_natural_order: false,
                learned_dc_tree: false,
                dc_wp_race: false,
                dc_wp_proxy_pick: true,
                optimized_ctx_map: false,
                select_configs: true,
                refinement: Some(AnsRefinement::Fast { recluster: false }),
            },
            Speed::Medium | Speed::MediumSlow => Self {
                try_natural_order: false,
                learned_dc_tree: false,
                dc_wp_race: false,
                dc_wp_proxy_pick: true,
                optimized_ctx_map: false,
                select_configs: true,
                refinement: Some(AnsRefinement::Fast { recluster: true }),
            },
            Speed::Slow | Speed::ExtraSlow => Self {
                try_natural_order: true,
                learned_dc_tree: true,
                dc_wp_race: speed == Speed::ExtraSlow,
                dc_wp_proxy_pick: speed != Speed::ExtraSlow,
                // About −0.24 % on large photos for ×1.7 encode time.
                optimized_ctx_map: speed == Speed::ExtraSlow,
                select_configs: true,
                refinement: Some(AnsRefinement::slow_for_speed(speed)),
            },
        }
    }
}

/// Encodes `jpg` as a complete JXL codestream.
pub(crate) fn encode_jpeg_codestream(
    jpg: &JpegData,
    icc: Option<&[u8]>,
    orientation: Orientation,
    speed: Speed,
    num_threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    let ss = check_supported(jpg).map_err(|e| EncodeError::Jpeg(e.to_string()))?;
    let dim = Dim::new(jpg.width, jpg.height, &ss);
    let pool = ThreadPool::new(num_threads);
    let mut scratch = Box::<CoderScratch>::default();
    encode_jpeg_codestream_with_pool(jpg, icc, orientation, &ss, &dim, speed, &pool, &mut scratch)
}

#[allow(clippy::too_many_arguments)]
fn encode_jpeg_codestream_with_pool(
    jpg: &JpegData,
    icc: Option<&[u8]>,
    orientation: Orientation,
    ss: &ChannelLayout,
    dim: &Dim,
    speed: Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Result<Vec<u8>, EncodeError> {
    // Quantization tables, transposed into JXL's orientation
    let mut qtable = [0i32; 3 * DCT_BLOCK_SIZE];
    let mut dc_quant = [0f32; 3];
    for c in 0..3usize {
        let comp = &jpg.components[ss.jpeg_order[c]];
        let q = &jpg.quant[comp.quant_idx as usize].values;
        for (y, src) in q.as_chunks::<8>().0.iter().take(8).enumerate() {
            for (x, &src) in src.iter().enumerate() {
                // JXL transposes the DCT relative to JPEG.
                qtable[c * DCT_BLOCK_SIZE + x * 8 + y] = src;
            }
        }
        dc_quant[c] = q[0] as f32 / (255.0 * 8.0);
    }

    let effort = Effort::new(speed);
    let cfl = JpegCfl::new(jpg, dim, ss, pool, scratch);

    // DC planes. Each DC group reads a disjoint slab of coefficients.
    let dc_datas = pool.steal_map(scratch, dim.num_dc_groups, |g, _scratch| {
        let gx = g % dim.xsize_dc_groups;
        let gy = g / dim.xsize_dc_groups;
        let bx0 = gx * DC_GROUP_DIM_IN_BLOCKS;
        let by0 = gy * DC_GROUP_DIM_IN_BLOCKS;
        let bw = DC_GROUP_DIM_IN_BLOCKS.min(dim.xsize_blocks - bx0);
        let bh = DC_GROUP_DIM_IN_BLOCKS.min(dim.ysize_blocks - by0);
        let mut data = DcGroupData::new(bw, bh)?;
        // Each DC plane is the luma grid scaled by its own shift.
        let sizes = [0usize, 1, 2].map(|c| (bw >> ss.hshift[c], bh >> ss.vshift[c]));
        data.quant_dc = crate::image::Image3S::try_new_per_plane(sizes)?;
        for c in 0..3usize {
            let (cw, ch) = sizes[c];
            let Some(comp) = ss.component(jpg, c) else {
                for y in 0..ch {
                    data.quant_dc.plane_row_mut(c, y)[..cw].fill(0);
                }
                continue;
            };
            for y in 0..ch {
                let row = data.quant_dc.plane_row_mut(c, y);
                for (x, dst) in row[..cw].iter_mut().enumerate() {
                    let block = ((by0 >> ss.vshift[c]) + y) * comp.width_in_blocks
                        + (bx0 >> ss.hshift[c])
                        + x;
                    *dst = comp.coeffs[block * DCT_BLOCK_SIZE] + ss.dc_offset[c];
                }
            }
        }
        if let Some(cfl) = &cfl {
            let (tx0, ty0) = (bx0 / CFL_TILE_DIM_IN_BLOCKS, by0 / CFL_TILE_DIM_IN_BLOCKS);
            for (slot, map) in [&mut data.ytox_map, &mut data.ytob_map]
                .into_iter()
                .enumerate()
            {
                for ty in 0..map.ysize() {
                    let src = &cfl.maps[slot][(ty0 + ty) * cfl.xtiles + tx0..];
                    let row = map.row_mut(ty);
                    let n = row.len();
                    row.copy_from_slice(&src[..n]);
                }
            }
        }
        Ok::<_, EncodeError>(data)
    });
    let dc_datas: Vec<DcGroupData> = dc_datas.into_iter().collect::<Result<_, _>>()?;

    // The DC plans (fixed tree, and the learned tree where the effort races
    // one) under one set of weighted-predictor parameters.
    let plan_dc = |wp_params: crate::weighted_predictor::WpParams, scratch: &mut CoderScratch| {
        // Small images get libjxl's pruned fixed tree; its contexts are looked up
        // from each token's properties instead of the static tree's numbering.
        let dc_samples: usize = dc_datas
            .iter()
            .map(|d| {
                (0..3)
                    .map(|c| d.quant_dc.plane(c).xsize() * d.quant_dc.plane(c).ysize())
                    .sum::<usize>()
            })
            .sum();
        let small_tree = crate::dc_tree::fixed_wp_dc_tree(dim.num_dc_groups, dc_samples);

        // Fixed-tree tokens, with properties and gradient residuals only where
        // the pruned or learned tree needs them.
        let collected = pool.steal_map(scratch, dc_datas.len(), |i, _scratch| {
            let collect_props = small_tree.is_some() || effort.learned_dc_tree;
            let (mut props, mut meta_props) = (Vec::new(), Vec::new());
            let (dc, gradient) = collect_dc_tokens_with_gradient(
                &dc_datas[i],
                &crate::frame::DC_PREDICTOR_WEIGHTED,
                &mut props,
                collect_props,
                effort.learned_dc_tree,
                wp_params,
            );
            // epf_iters = 0 here, so the sharpness id is decoder-ignored;
            // any distance >= 5 keeps the stream at the historical constant 4.
            let meta = collect_ac_metadata_tokens(
                &dc_datas[i],
                &mut meta_props,
                crate::frame::epf_sharpness_id(100.0, false),
                collect_props,
            );
            (dc, gradient, meta, props, meta_props)
        });

        let build_code = |dc: &[Vec<Token>],
                          meta: &[Vec<Token>],
                          num_contexts: usize,
                          scratch: &mut CoderScratch| {
            let streams = || dc.iter().chain(meta).map(Vec::as_slice);
            let mut code = crate::entropy::optimize_entropy_code_jpeg_ac_streams(
                streams(),
                num_contexts,
                &mut scratch.huffman_pool,
                effort.select_configs,
                Some(pool),
            );
            if let Some(refinement) = effort.refinement {
                crate::entropy::refine_entropy_bundle(
                    &mut code,
                    streams(),
                    refinement,
                    pool,
                    scratch,
                );
            }
            let mut header = BitWriter::new();
            write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, &mut header);
            let bits = header.bits_written() as u64
                + crate::entropy::estimate_ac_plain_bits(streams(), &code);
            (code, bits)
        };

        let mut dc_tokens = Vec::with_capacity(collected.len());
        let mut meta_tokens = Vec::with_capacity(collected.len());
        let mut gradient_tokens = Vec::with_capacity(collected.len());
        let mut dc_props = Vec::with_capacity(collected.len());
        let mut meta_props = Vec::with_capacity(collected.len());
        for (dc, gradient, meta, props, metadata_props) in collected {
            dc_tokens.push(dc);
            gradient_tokens.push(gradient);
            meta_tokens.push(meta);
            dc_props.push(props);
            meta_props.push(metadata_props);
        }

        // Learn from the same predictor walk as the fixed-tree incumbent. The
        // final race compares serialized sections, including padding and TOC.
        let learned_dc = effort.learned_dc_tree.then(|| {
            let tree = crate::dc_tree::learn_dc_tree(
                dim.num_dc_groups,
                &dc_tokens,
                &gradient_tokens,
                &dc_props,
                &meta_tokens,
                &meta_props,
            );
            let dc = dc_tokens
                .iter()
                .zip(&gradient_tokens)
                .zip(&dc_props)
                .map(|((wp, gradient), props)| {
                    wp.iter()
                        .zip(gradient)
                        .zip(props)
                        .map(|((wp, gradient), &prop)| {
                            let context = tree.dc_context[prop as usize];
                            let value = if tree.leaf_gradient[context as usize] {
                                gradient.value
                            } else {
                                wp.value
                            };
                            Token::new(u32::from(context), value)
                        })
                        .collect()
                })
                .collect::<Vec<Vec<Token>>>();
            let meta = meta_tokens
                .iter()
                .zip(&meta_props)
                .map(|(tokens, props)| {
                    tokens
                        .iter()
                        .zip(props)
                        .map(|(token, &prop)| {
                            let slot = ((token.context as usize) << 10) | (prop & 1023) as usize;
                            Token::new(u32::from(tree.meta_context[slot]), token.value)
                        })
                        .collect()
                })
                .collect::<Vec<Vec<Token>>>();
            let (code, _) = build_code(&dc, &meta, tree.num_contexts, scratch);
            DcPlan {
                tree: Some(tree),
                dc,
                meta,
                code,
                wp_params,
            }
        });
        drop(gradient_tokens);
        let (mut dc_code, static_bits) =
            build_code(&dc_tokens, &meta_tokens, K_NUM_DC_CONTEXTS, scratch);

        // The pruned tree replaces the static one only where it is cheaper in
        // full: tree, histograms and payload.
        let mut pruned_tree = None;
        if let Some(tree) = small_tree {
            let mut dc_small = dc_tokens.clone();
            let mut meta_small = meta_tokens.clone();
            for ((dc, meta), (dc_props, meta_props)) in dc_small
                .iter_mut()
                .zip(meta_small.iter_mut())
                .zip(dc_props.iter().zip(&meta_props))
            {
                for (t, &p) in dc.iter_mut().zip(dc_props) {
                    t.context = u32::from(tree.dc_context[p as usize]);
                }
                for (t, &p) in meta.iter_mut().zip(meta_props) {
                    let slot = ((t.context as usize) << 10) | (p & 1023) as usize;
                    t.context = u32::from(tree.meta_context[slot]);
                }
            }
            let (code_small, small_bits) =
                build_code(&dc_small, &meta_small, tree.num_contexts, scratch);
            let mut static_header = BitWriter::new();
            write_context_tree(
                dim.num_dc_groups,
                &crate::frame::DC_PREDICTOR_WEIGHTED,
                &mut scratch.huffman_pool,
                &mut static_header,
            );
            let mut small_header = BitWriter::new();
            crate::frame::write_tree_tokens(
                &tree.tokens,
                &mut scratch.huffman_pool,
                &mut small_header,
            );
            if small_bits + (small_header.bits_written() as u64)
                < static_bits + (static_header.bits_written() as u64)
            {
                dc_tokens = dc_small;
                meta_tokens = meta_small;
                dc_code = code_small;
                pruned_tree = Some(tree);
            }
        }
        let dc_plan = DcPlan {
            tree: pruned_tree,
            dc: dc_tokens,
            meta: meta_tokens,
            code: dc_code,
            wp_params,
        };
        drop((dc_props, meta_props));
        (dc_plan, learned_dc)
    };
    // The default parameters against libjxl's second preset: picked on the
    // static tree's Shannon bits or, at ExtraSlow, raced on the serialized
    // sections.
    let presets = &crate::weighted_predictor::WpParams::PRESETS;
    let first = if effort.dc_wp_proxy_pick {
        let mut bits = |params| crate::frame::dc_wp_proxy_bits(pool, scratch, &dc_datas, params);
        if bits(presets[1]) < bits(presets[0]) {
            presets[1]
        } else {
            presets[0]
        }
    } else {
        presets[0]
    };
    let mut dc_candidates = vec![plan_dc(first, scratch)];
    if effort.dc_wp_race {
        dc_candidates.push(plan_dc(presets[1], scratch));
    }

    let natural_scan = {
        let mut s = [[0u8; DCT_BLOCK_SIZE]; 3];
        for row in &mut s {
            for (k, v) in row.iter_mut().enumerate() {
                *v = K_COEFF_ORDER_8X8[k];
            }
        }
        s
    };
    let orders = compute_coeff_orders(jpg, dim, ss, cfl.as_ref());
    let custom_scan = orders
        .iter()
        .any(|o| !coeff_order::is_identity(o))
        .then(|| {
            let mut custom_scan = [[0u8; DCT_BLOCK_SIZE]; 3];
            for c in 0..3 {
                for k in 0..DCT_BLOCK_SIZE {
                    custom_scan[c][k] = K_COEFF_ORDER_8X8[orders[c][k] as usize];
                }
            }
            custom_scan
        });
    // Permutation signaling: one order index (DCT8), three channels.
    let perm_tokens = || {
        let mut perm_tokens: Vec<Token> = Vec::new();
        for order in &orders {
            coeff_order::tokenize_permutation(order, 1, &mut perm_tokens);
        }
        perm_tokens
    };
    let primary_scan = custom_scan.as_ref().unwrap_or(&natural_scan);
    let plan_with = |bctx: &BlockCtx,
                     scan: &[[u8; DCT_BLOCK_SIZE]; 3],
                     perm: Option<Vec<Token>>,
                     effort: Effort,
                     scratch: &mut CoderScratch| {
        plan_ac(
            jpg,
            dim,
            ss,
            bctx,
            cfl.as_ref(),
            &qtable,
            scan,
            perm,
            effort,
            pool,
            scratch,
        )
    };
    let primary_perm = || custom_scan.is_some().then(perm_tokens);

    let mut bctx = BlockCtx::luma_quantiles(jpg, dim, ss, &qtable);
    let mut primary = None;
    // Below about a megapixel the optimized map never won on the test sweep.
    if effort.optimized_ctx_map && dim.xsize_blocks * dim.ysize_blocks >= 1 << 14 {
        let opt = ctx_opt::optimize(jpg, dim, ss, cfl.as_ref(), primary_scan, pool, scratch);
        // Raced on a Fast refinement of each arm's clustered code; the
        // winner's unrefined code then gets the full refinement. Unrefined
        // codes alone misrank near-ties.
        let race = |b: &BlockCtx, scratch: &mut CoderScratch| {
            let unrefined = Effort {
                refinement: None,
                ..effort
            };
            let plan = plan_with(b, primary_scan, primary_perm(), unrefined, scratch);
            let mut code = plan.code.clone();
            crate::entropy::refine_entropy_bundle(
                &mut code,
                plan.tokens.iter().map(Vec::as_slice),
                AnsRefinement::Fast { recluster: false },
                pool,
                scratch,
            );
            let mut header = BitWriter::new();
            b.write(scratch, &mut header);
            write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, &mut header);
            let bits = header.bits_written() as u64
                + crate::entropy::estimate_ac_plain_bits(
                    plan.tokens.iter().map(Vec::as_slice),
                    &code,
                );
            (plan, bits)
        };
        let (opt_plan, opt_bits) = race(&opt, scratch);
        let (base_plan, base_bits) = race(&bctx, scratch);
        let winner = if opt_bits < base_bits {
            bctx = opt;
            opt_plan
        } else {
            base_plan
        };
        primary = Some(winner.refine(dim, &qtable, primary_perm(), effort, pool, scratch));
    }
    let primary =
        primary.unwrap_or_else(|| plan_with(&bctx, primary_scan, primary_perm(), effort, scratch));
    let ac = if custom_scan.is_some() && effort.try_natural_order {
        let natural = plan_with(&bctx, &natural_scan, None, effort, scratch);
        if primary.estimated_bits() < natural.estimated_bits() {
            primary
        } else {
            natural
        }
    } else {
        primary
    };

    // Sections. Each is an independent BitWriter, so the per-group ones can be
    // filled in parallel and stitched together afterward.
    let mut dc_global = BitWriter::new();
    {
        let w = &mut dc_global;
        write_dc_quant(w, dc_quant).map_err(|e| EncodeError::Jpeg(e.to_string()))?;
        write_quant_scales(65536, 1, w);
        bctx.write(scratch, w);
        write_color_correlation(w);
    }
    let mut dc_section_candidates = Vec::with_capacity(2 * dc_candidates.len());
    for (dc_plan, learned_dc) in dc_candidates {
        dc_section_candidates.push(dc_plan.write_sections(&dc_global, &dc_datas, pool, scratch));
        if let Some(plan) = learned_dc {
            dc_section_candidates.push(plan.write_sections(&dc_global, &dc_datas, pool, scratch));
        }
    }

    let ac_groups = ac.write_groups(pool, scratch);
    // The smallest serialized DC coding, padding and TOC included; ties keep
    // the earlier (default-parameter, fixed-tree) candidate.
    let size = |dc: &[BitWriter]| {
        let bits: Vec<usize> = dc
            .iter()
            .chain(std::iter::once(&ac.global))
            .chain(&ac_groups)
            .map(BitWriter::bits_written)
            .collect();
        combined_sections_size(&bits)
    };
    let mut candidates = dc_section_candidates.into_iter();
    let mut dc_sections = candidates.next().expect("default DC candidate");
    let mut dc_size = size(&dc_sections);
    for candidate in candidates {
        let candidate_size = size(&candidate);
        if candidate_size < dc_size {
            dc_sections = candidate;
            dc_size = candidate_size;
        }
    }
    let mut sections = Vec::with_capacity(2 + dim.num_dc_groups + dim.num_groups);
    sections.extend(dc_sections);
    sections.push(ac.global);
    sections.extend(ac_groups);

    // Assemble
    let mut out = BitWriter::new();
    out.write(8, 0xFF);
    out.write(8, 0x0A);
    write_size_header(&mut out, dim.xsize, dim.ysize);
    write_image_metadata(&mut out, icc, ss.gray, orientation, scratch);
    write_frame_header(&mut out, ss);
    combine_sections(&mut sections, &mut out);
    Ok(out.into_bytes())
}

/// A complete DC tree and its entropy-coded DC/metadata streams.
struct DcPlan {
    tree: Option<crate::dc_tree::LearnedDcTree>,
    dc: Vec<Vec<Token>>,
    meta: Vec<Vec<Token>>,
    code: crate::entropy::OwnedEntropyCode,
    /// The DC streams' weighted-predictor parameters.
    wp_params: crate::weighted_predictor::WpParams,
}

impl DcPlan {
    fn write_sections(
        self,
        common_global: &BitWriter,
        dc_datas: &[DcGroupData],
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> Vec<BitWriter> {
        let mut global = common_global.clone();
        match &self.tree {
            Some(tree) => crate::frame::write_tree_tokens(
                &tree.tokens,
                &mut scratch.huffman_pool,
                &mut global,
            ),
            None => write_context_tree(
                dc_datas.len(),
                &crate::frame::DC_PREDICTOR_WEIGHTED,
                &mut scratch.huffman_pool,
                &mut global,
            ),
        }
        global.write(1, 0); // no lz77 for the DC histograms
        let code = self.code.as_ref();
        write_entropy_code(&code, &mut scratch.huffman_pool, &mut global);
        let groups = pool.steal_map(scratch, dc_datas.len(), |i, _scratch| {
            let data = &dc_datas[i];
            let mut section = BitWriter::new();
            section.write(2, 0); // extra_dc_precision = 0
            section.write(1, 1); // global tree
            crate::weighted_predictor::write_wp_header(self.wp_params, &mut section);
            section.write(2, 0); // no transforms
            emit_tokens(&self.dc[i], &code, &mut section);

            let num_blocks = data.ac_strategy.xsize() * data.ac_strategy.ysize();
            let nb_bits = if num_blocks <= 1 {
                0
            } else {
                usize::BITS as usize
                    - num_blocks.leading_zeros() as usize
                    - usize::from(num_blocks.is_power_of_two())
            };
            if nb_bits != 0 {
                section.write(nb_bits, (num_blocks - 1) as u64);
            }
            section.write(4, 3);
            emit_tokens(&self.meta[i], &code, &mut section);
            section
        });
        let mut sections = Vec::with_capacity(1 + groups.len());
        sections.push(global);
        sections.extend(groups);
        sections
    }
}

fn emit_tokens(tokens: &[Token], code: &crate::entropy::EntropyCode<'_>, w: &mut BitWriter) {
    if code.use_prefix_code {
        for t in tokens {
            write_token(*t, code, w);
        }
    } else {
        write_ans_tokens(
            tokens,
            code.context_map,
            code.ans_symbols,
            code.ans_reverse_maps,
            code.hybrid_uint_configs,
            w,
        );
    }
}

/// One coefficient ordering, tokenized and entropy-coded but not yet written:
/// the AC-global section is final, the groups are emitted only for the winner.
struct AcPlan {
    global: BitWriter,
    tokens: Vec<Vec<Token>>,
    code: crate::entropy::OwnedEntropyCode,
}

impl AcPlan {
    /// Applies `effort`'s refinement to a plan clustered without it.
    fn refine(
        self,
        dim: &Dim,
        qtable: &[i32; 3 * DCT_BLOCK_SIZE],
        perm: Option<Vec<Token>>,
        effort: Effort,
        pool: &ThreadPool,
        scratch: &mut CoderScratch,
    ) -> AcPlan {
        finish_ac_plan(
            self.tokens,
            self.code,
            dim,
            qtable,
            perm,
            effort.refinement,
            pool,
            scratch,
        )
    }

    /// Global section plus estimated payload, for choosing between orderings.
    fn estimated_bits(&self) -> u64 {
        self.global.bits_written() as u64
            + crate::entropy::estimate_ac_plain_bits(
                self.tokens.iter().map(Vec::as_slice),
                &self.code,
            )
    }

    fn write_groups(&self, pool: &ThreadPool, scratch: &mut CoderScratch) -> Vec<BitWriter> {
        let code = self.code.as_ref();
        pool.steal_map(scratch, self.tokens.len(), |g, _scratch| {
            let mut section = BitWriter::new();
            emit_tokens(&self.tokens[g], &code, &mut section);
            section
        })
    }
}

/// Tokenizes and entropy-codes the AC coefficients under `scan`.
///
/// `perm` carries the coefficient-order permutation tokens when `scan` is a
/// custom order; `None` signals the natural order (`used_orders = 0`).
#[allow(clippy::too_many_arguments)]
fn plan_ac(
    jpg: &JpegData,
    dim: &Dim,
    ss: &ChannelLayout,
    bctx: &BlockCtx,
    cfl: Option<&JpegCfl>,
    qtable: &[i32; 3 * DCT_BLOCK_SIZE],
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    perm: Option<Vec<Token>>,
    effort: Effort,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> AcPlan {
    let tokens = tokenize_ac(jpg, dim, ss, bctx, cfl, scan, pool, scratch);
    let code = crate::entropy::optimize_entropy_code_jpeg_ac_streams(
        tokens.iter().map(Vec::as_slice),
        bctx.num_ac_contexts(),
        &mut scratch.huffman_pool,
        effort.select_configs,
        Some(pool),
    );
    finish_ac_plan(
        tokens,
        code,
        dim,
        qtable,
        perm,
        effort.refinement,
        pool,
        scratch,
    )
}

/// Refines a clustered AC code and serializes the AC global section.
#[allow(clippy::too_many_arguments)]
fn finish_ac_plan(
    tokens: Vec<Vec<Token>>,
    mut code: crate::entropy::OwnedEntropyCode,
    dim: &Dim,
    qtable: &[i32; 3 * DCT_BLOCK_SIZE],
    perm: Option<Vec<Token>>,
    refinement: Option<AnsRefinement>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> AcPlan {
    if let Some(refinement) = refinement {
        crate::entropy::refine_entropy_bundle(
            &mut code,
            tokens.iter().map(Vec::as_slice),
            refinement,
            pool,
            scratch,
        );
    }

    let mut global = BitWriter::new();
    {
        let w = &mut global;
        write_dequant_matrices(w, qtable, scratch);
        if dim.num_groups > 1 {
            let bits = usize::BITS as usize
                - dim.num_groups.leading_zeros() as usize
                - if dim.num_groups.is_power_of_two() {
                    1
                } else {
                    0
                };
            if bits != 0 {
                w.write(bits, 0); // num_histo_bits = 0
            }
        }
        // used_orders is a 13-bit mask (one bit per order index). Only order 0
        // (DCT8) is ever used here, so the mask is 0 or 1.
        w.write(2, 3); // used_orders U32 selector 3 = raw 13 bits
        match &perm {
            Some(perm_tokens) => {
                w.write(13, 1); // custom order for DCT8
                // Its own entropy stream, then the tokens, exactly as libjxl's
                // EncodeCoeffOrders lays it out.
                let perm_code = optimize_entropy_code(
                    perm_tokens,
                    coeff_order::PERMUTATION_CONTEXTS,
                    &mut scratch.huffman_pool,
                );
                w.write(1, 0); // no lz77
                write_entropy_code(&perm_code.as_ref(), &mut scratch.huffman_pool, w);
                emit_tokens(perm_tokens, &perm_code.as_ref(), w);
            }
            None => w.write(13, 0), // natural order
        }
        w.write(1, 0); // no lz77
        write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, w);
    }

    AcPlan {
        global,
        tokens,
        code,
    }
}

/// Derives the per-channel DCT8 coefficient order from non-zero statistics.
fn compute_coeff_orders(
    jpg: &JpegData,
    dim: &Dim,
    ss: &ChannelLayout,
    cfl: Option<&JpegCfl>,
) -> [[u8; DCT_BLOCK_SIZE]; 3] {
    // Slot -> source coefficient index: the natural slot's transposed-raster
    // position mapped back into the JPEG block's own raster layout.
    let mut slot_to_src = [0usize; DCT_BLOCK_SIZE];
    for (slot, s) in slot_to_src.iter_mut().enumerate() {
        let r = K_COEFF_ORDER_8X8[slot] as usize;
        *s = (r & 7) * 8 + (r >> 3);
    }

    let mut orders: [[u8; DCT_BLOCK_SIZE]; 3] = [std::array::from_fn(|k| k as u8); 3];
    for c in 0..3usize {
        if ss.component(jpg, c).is_none() {
            continue;
        }
        let cbw = dim.xsize_blocks >> ss.hshift[c];
        let cbh = dim.ysize_blocks >> ss.vshift[c];
        let mut nonzero = [0u64; DCT_BLOCK_SIZE];
        for by in 0..cbh {
            for bx in 0..cbw {
                let coeffs = coded_block(jpg, ss, cfl, c, bx, by);
                for slot in 1..DCT_BLOCK_SIZE {
                    if coeffs[slot_to_src[slot]] != 0 {
                        nonzero[slot] += 1;
                    }
                }
            }
        }
        orders[c] = coeff_order::compute_order(&nonzero, (cbw * cbh) as u64, 1);
    }
    orders
}

/// Produces one token stream per AC group, one group per work item.
fn tokenize_ac(
    jpg: &JpegData,
    dim: &Dim,
    ss: &ChannelLayout,
    bctx: &BlockCtx,
    cfl: Option<&JpegCfl>,
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<Vec<Token>> {
    pool.steal_map(scratch, dim.num_groups, |g, _scratch| {
        let gx = g % dim.xsize_groups;
        let gy = g / dim.xsize_groups;
        tokenize_ac_group(jpg, dim, ss, bctx, cfl, scan, gx, gy)
    })
}

/// Tokenizes the AC coefficients of one group.
fn tokenize_ac_group(
    jpg: &JpegData,
    dim: &Dim,
    ss: &ChannelLayout,
    bctx: &BlockCtx,
    cfl: Option<&JpegCfl>,
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    gx: usize,
    gy: usize,
) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    visit_group_blocks(jpg, dim, ss, cfl, gx, gy, |b| {
        let block_ctx = bctx.context(b.c, b.dc);
        tokens.push(Token::new(
            bctx.non_zero_context(b.predicted, block_ctx),
            b.nzeros,
        ));
        let histo_offset = bctx.zero_density_offset(block_ctx);
        for_each_coeff(&scan[b.c], b.block, b.nzeros, |zdc, coef| {
            tokens.push(Token::new(histo_offset + zdc as u32, pack_signed(coef)));
        });
    });
    tokens
}

/// One coded DCT8 block, as the AC tokenizer sees it.
struct BlockVisit<'a> {
    /// JXL channel.
    c: usize,
    /// Co-located quantized DC of every channel, as the decoder buckets it.
    dc: [i32; 3],
    /// Non-zero count predicted from the top and left neighbors.
    predicted: u32,
    nzeros: u32,
    /// Coefficients in JXL's (transposed) raster order.
    block: &'a [i32; DCT_BLOCK_SIZE],
}

/// Visits the coded blocks of one group in bitstream order.
fn visit_group_blocks(
    jpg: &JpegData,
    dim: &Dim,
    ss: &ChannelLayout,
    cfl: Option<&JpegCfl>,
    gx: usize,
    gy: usize,
    mut visit: impl FnMut(BlockVisit<'_>),
) {
    let mut block = [0i32; DCT_BLOCK_SIZE];
    let bx0 = gx * GROUP_DIM_IN_BLOCKS;
    let by0 = gy * GROUP_DIM_IN_BLOCKS;
    let bw = GROUP_DIM_IN_BLOCKS.min(dim.xsize_blocks - bx0);
    let bh = GROUP_DIM_IN_BLOCKS.min(dim.ysize_blocks - by0);

    // Non-zero counts of already-coded neighbors, used as context.
    let mut num_nzeros = Image3B::new(GROUP_DIM_IN_BLOCKS, GROUP_DIM_IN_BLOCKS);

    for by in 0..bh {
        for bx in 0..bw {
            let abs_bx = bx0 + bx;
            let abs_by = by0 + by;
            let dc = block_dcs(jpg, ss, abs_bx, abs_by);
            for &c in &CHANNEL_ORDER {
                // One block per subsampling cell, at its corner.
                let (hs, vs) = (ss.hshift[c], ss.vshift[c]);
                if (abs_bx >> hs) << hs != abs_bx || (abs_by >> vs) << vs != abs_by {
                    continue;
                }
                let (sx, sy) = (bx >> hs, by >> vs);
                let src = coded_block(jpg, ss, cfl, c, abs_bx >> hs, abs_by >> vs);
                // JXL transposes the DCT relative to JPEG.
                for (y, src_row) in src.as_chunks::<8>().0.iter().enumerate() {
                    for (x, &src) in src_row.iter().enumerate() {
                        block[x * 8 + y] = src;
                    }
                }

                let nzeros = block[1..].iter().filter(|&&v| v != 0).count() as u32;
                num_nzeros.plane_row_mut(c, sy)[sx] = nzeros as u8;

                // Context uses the channel's own grid.
                let row_top = if sy == 0 {
                    None
                } else {
                    Some(num_nzeros.plane_row(c, sy - 1))
                };
                let row = num_nzeros.plane_row(c, sy);
                let predicted = crate::group::predict_from_top_and_left(
                    row_top,
                    row,
                    sx,
                    GROUP_DIM_IN_BLOCKS as u8,
                );
                visit(BlockVisit {
                    c,
                    dc,
                    predicted: predicted as u32,
                    nzeros,
                    block: &block,
                });
            }
        }
    }
}

/// Quantized DC of every channel at full-resolution block `(bx, by)`: each
/// channel's own block covering it, zero for grayscale chroma.
#[inline]
fn block_dcs(jpg: &JpegData, ss: &ChannelLayout, bx: usize, by: usize) -> [i32; 3] {
    std::array::from_fn(|p| {
        ss.component(jpg, p).map_or(0, |comp| {
            let block = (by >> ss.vshift[p]) * comp.width_in_blocks + (bx >> ss.hshift[p]);
            i32::from(comp.coeffs[block * DCT_BLOCK_SIZE] + ss.dc_offset[p])
        })
    })
}

/// Calls `f(zero_density_context, coefficient)` for every coefficient the
/// bitstream codes, in scan order up to the last non-zero.
#[inline]
fn for_each_coeff(
    scan: &[u8; DCT_BLOCK_SIZE],
    block: &[i32; DCT_BLOCK_SIZE],
    nzeros: u32,
    mut f: impl FnMut(usize, i32),
) {
    let mut prev = if nzeros as usize > DCT_BLOCK_SIZE / 16 {
        0
    } else {
        1
    };
    let mut remaining = nzeros;
    let mut k = 1usize;
    while k < DCT_BLOCK_SIZE && remaining != 0 {
        let coef = block[scan[k] as usize];
        f(zero_density_context_8x8(remaining as usize, k, prev), coef);
        prev = usize::from(coef != 0);
        if coef != 0 {
            remaining -= 1;
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::JpegComponent;

    fn gray(h: usize, v: usize) -> JpegData {
        JpegData {
            components: vec![JpegComponent {
                id: 1,
                h_samp_factor: h,
                v_samp_factor: v,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn grayscale_keeps_full_resolution_and_signals_its_sampling() {
        // The decoder rebuilds the component's sampling factors from the
        // per-channel modes, so they must survive even though no channel is
        // actually subsampled.
        for ((h, v), mode, max) in [
            ((1, 1), 0, (0, 0)),
            ((2, 2), 1, (1, 1)),
            ((2, 1), 2, (1, 0)),
            ((1, 2), 3, (0, 1)),
        ] {
            let ss = check_supported(&gray(h, v)).unwrap();
            assert!(ss.gray);
            assert_eq!(ss.mode, [mode; 3]);
            assert_eq!((ss.max_hshift, ss.max_vshift), max);
            assert_eq!((ss.hshift, ss.vshift), ([0; 3], [0; 3]));
        }
        assert!(check_supported(&gray(4, 1)).is_err());
    }

    #[test]
    fn grayscale_chroma_reads_as_zero() {
        let mut jpg = gray(1, 1);
        jpg.components[0].width_in_blocks = 1;
        jpg.components[0].height_in_blocks = 1;
        jpg.components[0].coeffs = (1..=64).collect();
        let ss = check_supported(&jpg).unwrap();
        assert_eq!(coded_block(&jpg, &ss, None, 1, 0, 0)[5], 6);
        assert_eq!(coded_block(&jpg, &ss, None, 0, 0, 0), [0; DCT_BLOCK_SIZE]);
        assert_eq!(coded_block(&jpg, &ss, None, 2, 0, 0), [0; DCT_BLOCK_SIZE]);
    }

    fn three(ids: [u8; 3], samp: [(usize, usize); 3], markers: &[(u8, Vec<u8>)]) -> JpegData {
        let quant = |dc: i32| super::super::JpegQuantTable {
            values: std::array::from_fn(|i| if i == 0 { dc } else { 1 }),
            precision: 0,
            index: 0,
            is_last: true,
        };
        JpegData {
            components: (0..3)
                .map(|i| JpegComponent {
                    id: u32::from(ids[i]),
                    h_samp_factor: samp[i].0,
                    v_samp_factor: samp[i].1,
                    quant_idx: i as u32,
                    ..Default::default()
                })
                .collect(),
            quant: vec![quant(3), quant(8), quant(1000)],
            marker_order: markers.iter().map(|(m, _)| *m).collect(),
            app_data: markers.iter().map(|(_, d)| d.clone()).collect(),
            ..Default::default()
        }
    }

    fn adobe(transform: u8) -> (u8, Vec<u8>) {
        let mut seg = vec![0xEE, 0, 14];
        seg.extend_from_slice(b"Adobe\x00\x64\x00\x00\x00\x00");
        seg.push(transform);
        (0xEE, seg)
    }

    #[test]
    fn rgb_detection_follows_libjxl() {
        let full = [(1, 1); 3];
        let jfif = (0xE0, vec![0xE0, 0, 16]);
        // Component ids alone, when nothing else says otherwise.
        assert!(is_rgb(&three(*b"RGB", full, &[])));
        assert!(!is_rgb(&three([1, 2, 3], full, &[])));
        // Adobe's transform flag wins over the ids...
        assert!(is_rgb(&three([1, 2, 3], full, &[adobe(0)])));
        assert!(!is_rgb(&three(*b"RGB", full, &[adobe(1)])));
        // ...and a JFIF marker means YCbCr whatever else is present.
        assert!(!is_rgb(&three(*b"RGB", full, &[jfif.clone(), adobe(0)])));
        // A later APP14 is found past other APP segments.
        let exif = (0xE1, vec![0xE1, 0, 6, b'E', b'x', b'i']);
        assert!(is_rgb(&three([1, 2, 3], full, &[exif, adobe(0)])));
    }

    #[test]
    fn each_channel_signals_its_own_sampling() {
        // Cb and Cr at different resolutions.
        let ss = check_supported(&three([1, 2, 3], [(2, 2), (2, 1), (1, 1)], &[])).unwrap();
        assert_eq!(ss.mode, [2, 1, 0]);
        assert_eq!((ss.hshift, ss.vshift), ([0, 0, 1], [1, 0, 1]));
        assert_eq!((ss.max_hshift, ss.max_vshift), (1, 1));
        // Every component at 2x1 is full resolution, but the factors must
        // survive for the JPEG and the grid still follows the 2-block MCU.
        let ss = check_supported(&three([1, 2, 3], [(2, 1); 3], &[])).unwrap();
        assert_eq!(ss.mode, [2; 3]);
        assert_eq!((ss.hshift, ss.vshift), ([0; 3], [0; 3]));
        assert_eq!((ss.max_hshift, ss.max_vshift), (1, 0));
        // Chroma sharper than luma has no layout here.
        assert!(check_supported(&three([1, 2, 3], [(1, 1), (2, 2), (2, 2)], &[])).is_err());
    }

    #[test]
    fn rgb_layout_skips_ycbcr_and_recenters_dc() {
        let ss = check_supported(&three(*b"RGB", [(1, 1); 3], &[])).unwrap();
        assert!(!ss.ycbcr && !ss.gray);
        assert_eq!(ss.jpeg_order, [0, 1, 2]);
        assert_eq!(ss.dc_offset, [1024 / 3, 1024 / 8, 1]);
        // Chroma subsampling cannot be signaled without the YCbCr transform.
        assert!(check_supported(&three(*b"RGB", [(2, 2), (1, 1), (1, 1)], &[])).is_err());

        let ycc = check_supported(&three([1, 2, 3], [(2, 2), (1, 1), (1, 1)], &[])).unwrap();
        assert!(ycc.ycbcr);
        assert_eq!(ycc.dc_offset, [0; 3]);
    }

    #[test]
    fn dc_cells_nest_like_the_decoder() {
        // Two intervals on X and Y, three on B: cell = (x * 3 + b) * 2 + y.
        let bctx = BlockCtx::from_cells([vec![0], vec![5], vec![-3, 3]], |row, i| {
            (row * 12 + i) as u8 % 16
        });
        assert_eq!(bctx.num_dc_ctxs, 12);
        assert_eq!(bctx.dc_idx([0, 5, -3]), 0);
        assert_eq!(bctx.dc_idx([0, 6, -3]), 1);
        assert_eq!(bctx.dc_idx([0, 5, -2]), 2);
        assert_eq!(bctx.dc_idx([1, 5, 4]), 10);
        assert_eq!(bctx.dc_idx([1, 6, 4]), 11);
        // Map rows are Y, X, B; channels are X, Y, B.
        let dc = [1, 6, 4];
        assert_eq!(bctx.context(1, dc), 11);
        assert_eq!(bctx.context(0, dc), (12 + 11) % 16);
        assert_eq!(bctx.context(2, dc), (24 + 11) % 16);
    }
}
