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

use super::{DCT_BLOCK_SIZE, JpegData, JpegError, coeff_order};
use crate::ac_context::{
    K_NUM_AC_CONTEXTS, block_context, non_zero_context, zero_density_context_8x8,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::entropy::{
    Token, optimize_entropy_code, optimize_entropy_code_ac, pack_signed, write_ans_tokens,
    write_entropy_code, write_token,
};
use crate::frame::{
    collect_ac_metadata_tokens, collect_dc_tokens, combine_sections, write_context_tree,
    write_quant_scales,
};
use crate::image::Image3B;
use crate::static_entropy_codes::K_NUM_DC_CONTEXTS;
use crate::thread_pool::ThreadPool;

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
    fn new(xsize: usize, ysize: usize, ss: &Subsampling) -> Self {
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

/// The subsampling layout, expressed the way the frame header wants it.
struct Subsampling {
    /// Per-JXL-channel horizontal and vertical shifts (0 = full resolution).
    hshift: [usize; 3],
    vshift: [usize; 3],
    /// The 2-bit mode written for each channel, in X, Y, B order.
    mode: [u64; 3],
    max_hshift: usize,
    max_vshift: usize,
}

/// Works out the channel layout, rejecting anything JXL cannot express: only
/// chroma at full resolution or halved in either axis has a representation.
fn check_supported(jpg: &JpegData) -> Result<Subsampling, JpegError> {
    if jpg.components.len() != 3 {
        return Err(JpegError::UnsupportedMode(
            "only 3-component JPEGs are transcodable so far",
        ));
    }
    let max_h = jpg.max_h_samp();
    let max_v = jpg.max_v_samp();
    if !matches!(max_h, 1 | 2) || !matches!(max_v, 1 | 2) {
        return Err(JpegError::UnsupportedMode(
            "sampling factors above 2 are not representable",
        ));
    }

    let mut hshift = [0usize; 3];
    let mut vshift = [0usize; 3];
    for c in 0..3usize {
        let comp = &jpg.components[JPEG_ORDER_YCBCR[c]];
        if !max_h.is_multiple_of(comp.h_samp_factor) || !max_v.is_multiple_of(comp.v_samp_factor) {
            return Err(JpegError::UnsupportedMode(
                "sampling factors are not a power-of-two ratio",
            ));
        }
        hshift[c] = (max_h / comp.h_samp_factor).trailing_zeros() as usize;
        vshift[c] = (max_v / comp.v_samp_factor).trailing_zeros() as usize;
        if (max_h / comp.h_samp_factor).count_ones() != 1
            || (max_v / comp.v_samp_factor).count_ones() != 1
        {
            return Err(JpegError::UnsupportedMode(
                "sampling factors are not a power-of-two ratio",
            ));
        }
    }
    // Luma must be full resolution and both chroma channels must agree.
    if hshift[1] != 0 || vshift[1] != 0 || hshift[0] != hshift[2] || vshift[0] != vshift[2] {
        return Err(JpegError::UnsupportedMode(
            "only standard chroma subsampling layouts are supported",
        ));
    }

    // The mode lives on the luma channel; the decoder derives each shift as
    // `max_shift - table[mode]`.
    let max_hshift = hshift[0];
    let max_vshift = vshift[0];
    let luma_mode = match (max_hshift, max_vshift) {
        (0, 0) => 0, // 4:4:4
        (1, 1) => 1, // 4:2:0
        (1, 0) => 2, // 4:2:2
        (0, 1) => 3, // 4:4:0
        _ => {
            return Err(JpegError::UnsupportedMode(
                "unsupported chroma subsampling layout",
            ));
        }
    };

    Ok(Subsampling {
        hshift,
        vshift,
        mode: [0, luma_mode, 0],
        max_hshift,
        max_vshift,
    })
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
fn write_image_metadata(w: &mut BitWriter, icc: Option<&[u8]>, scratch: &mut CoderScratch) {
    w.write(1, 0); // not all-default
    w.write(1, 0); // no extra fields (orientation, preview, animation)
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
            false,
            w,
        ),
        None => w.write(1, 1), // color encoding: all default (sRGB)
    }
    w.write(2, 0); // no extensions
    w.write(1, 1); // CustomTransformData: all default
    if let Some(icc) = icc {
        crate::icc_codec::write_icc_stream(icc, &mut scratch.huffman_pool, w);
    }
    w.zero_pad_to_byte();
}

/// Writes the frame header.
fn write_frame_header(w: &mut BitWriter, ss: &Subsampling) {
    w.write(1, 0); // not all-default
    w.write(2, 0); // regular frame
    w.write(1, 0); // encoding = VarDCT

    // flags = 128 (kSkipAdaptiveDCSmoothing): smoothing would perturb the DC
    // and break the round-trip. U64 coder, 17..=272 is selector 2 + 8 bits.
    w.write(2, 2);
    w.write(8, 128 - 17);

    w.write(1, 1); // do_ycbcr = true (serialized because xyb_encoded = 0)
    // YCbCrChromaSubsampling, one 2-bit mode per channel in X, Y, B order.
    for mode in ss.mode {
        w.write(2, mode);
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

/// Encodes `jpg` as a complete JXL codestream.
pub(crate) fn encode_jpeg_codestream(
    jpg: &JpegData,
    icc: Option<&[u8]>,
    num_threads: usize,
) -> Result<Vec<u8>, JpegError> {
    let ss = check_supported(jpg)?;
    let dim = Dim::new(jpg.width, jpg.height, &ss);
    let pool = ThreadPool::new(num_threads);
    let mut scratch = Box::<CoderScratch>::default();
    encode_jpeg_codestream_with_pool(jpg, icc, &ss, &dim, &pool, &mut scratch)
}

fn encode_jpeg_codestream_with_pool(
    jpg: &JpegData,
    icc: Option<&[u8]>,
    ss: &Subsampling,
    dim: &Dim,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Result<Vec<u8>, JpegError> {
    // Quantization tables, transposed into JXL's orientation
    let mut qtable = [0i32; 3 * DCT_BLOCK_SIZE];
    let mut dc_quant = [0f32; 3];
    for c in 0..3usize {
        let comp = &jpg.components[JPEG_ORDER_YCBCR[c]];
        let q = &jpg.quant[comp.quant_idx as usize].values;
        for (y, src) in q.as_chunks::<8>().0.iter().take(8).enumerate() {
            for (x, &src) in src.iter().enumerate() {
                // JXL transposes the DCT relative to JPEG.
                qtable[c * DCT_BLOCK_SIZE + x * 8 + y] = src;
            }
        }
        dc_quant[c] = q[0] as f32 / (255.0 * 8.0);
    }

    // DC planes. Each DC group reads a disjoint slab of coefficients.
    let dc_datas: Vec<DcGroupData> = pool.steal_map(scratch, dim.num_dc_groups, |g, _scratch| {
        let gx = g % dim.xsize_dc_groups;
        let gy = g / dim.xsize_dc_groups;
        let bx0 = gx * DC_GROUP_DIM_IN_BLOCKS;
        let by0 = gy * DC_GROUP_DIM_IN_BLOCKS;
        let bw = DC_GROUP_DIM_IN_BLOCKS.min(dim.xsize_blocks - bx0);
        let bh = DC_GROUP_DIM_IN_BLOCKS.min(dim.ysize_blocks - by0);
        let mut data = DcGroupData::new(bw, bh);
        // Each DC plane is the luma grid scaled by its own shift.
        let sizes = [0usize, 1, 2].map(|c| (bw >> ss.hshift[c], bh >> ss.vshift[c]));
        data.quant_dc = crate::image::Image3S::new_per_plane(sizes);
        for c in 0..3usize {
            let comp = &jpg.components[JPEG_ORDER_YCBCR[c]];
            let (cw, ch) = sizes[c];
            for y in 0..ch {
                let row = data.quant_dc.plane_row_mut(c, y);
                for (x, dst) in row[..cw].iter_mut().enumerate() {
                    let block = ((by0 >> ss.vshift[c]) + y) * comp.width_in_blocks
                        + (bx0 >> ss.hshift[c])
                        + x;
                    *dst = comp.coeffs[block * DCT_BLOCK_SIZE];
                }
            }
        }
        data
    });

    // Tokens
    let (dc_tokens, meta_tokens): (Vec<Vec<Token>>, Vec<Vec<Token>>) = pool
        .steal_map(scratch, dc_datas.len(), |i, _scratch| {
            (
                collect_dc_tokens(
                    &dc_datas[i],
                    &crate::frame::DC_PREDICTOR_WEIGHTED,
                    &mut Vec::new(),
                    false,
                ),
                // epf_iters = 0 here, so the sharpness id is decoder-ignored;
                // any distance >= 5 keeps the stream at the historical constant 4.
                collect_ac_metadata_tokens(&dc_datas[i], &mut Vec::new(), 100.0, false),
            )
        })
        .into_iter()
        .unzip();

    let mut all_dc: Vec<Token> = Vec::new();
    for t in &dc_tokens {
        all_dc.extend_from_slice(t);
    }
    for t in &meta_tokens {
        all_dc.extend_from_slice(t);
    }
    let dc_code = optimize_entropy_code(&all_dc, K_NUM_DC_CONTEXTS, &mut scratch.huffman_pool);
    let dc_code_ref = dc_code.as_ref();

    let natural_scan = {
        let mut s = [[0u8; DCT_BLOCK_SIZE]; 3];
        for row in &mut s {
            for (k, v) in row.iter_mut().enumerate() {
                *v = K_COEFF_ORDER_8X8[k];
            }
        }
        s
    };
    let baseline = build_ac_sections(jpg, dim, ss, &qtable, &natural_scan, None, pool, scratch);

    let orders = compute_coeff_orders(jpg, dim, ss);
    let ac = if orders.iter().any(|o| !coeff_order::is_identity(o)) {
        let mut custom_scan = [[0u8; DCT_BLOCK_SIZE]; 3];
        for c in 0..3 {
            for k in 0..DCT_BLOCK_SIZE {
                custom_scan[c][k] = K_COEFF_ORDER_8X8[orders[c][k] as usize];
            }
        }
        // Permutation signaling: one order index (DCT8), three channels.
        let mut perm_tokens: Vec<Token> = Vec::new();
        for order in &orders {
            coeff_order::tokenize_permutation(order, 1, &mut perm_tokens);
        }
        let candidate = build_ac_sections(
            jpg,
            dim,
            ss,
            &qtable,
            &custom_scan,
            Some(perm_tokens),
            pool,
            scratch,
        );
        if candidate.bytes < baseline.bytes {
            candidate
        } else {
            baseline
        }
    } else {
        baseline
    };

    // Sections. Each is an independent BitWriter, so the per-group ones can be
    // filled in parallel and stitched together afterward.
    let mut dc_global = BitWriter::new();
    {
        let w = &mut dc_global;
        write_dc_quant(w, dc_quant)?;
        write_quant_scales(65536, 1, w);
        // Written explicitly; the lossy path avoids the shortcut too.
        w.write(1, 0);
        w.write(16, 0); // no DC thresholds, no quant-field thresholds
        crate::frame::write_compact_block_context_map(&mut scratch.huffman_pool, w);
        write_color_correlation(w);
        write_context_tree(
            dim.num_dc_groups,
            &crate::frame::DC_PREDICTOR_WEIGHTED,
            &mut scratch.huffman_pool,
            w,
        );
        w.write(1, 0); // no lz77 for the DC histograms
        write_entropy_code(&dc_code_ref, &mut scratch.huffman_pool, w);
    }

    // DC groups.
    let dc_group_sections: Vec<BitWriter> =
        pool.steal_map(scratch, dim.num_dc_groups, |i, _scratch| {
            let data = &dc_datas[i];
            let mut section = BitWriter::new();
            let w = &mut section;
            w.write(2, 0); // extra_dc_precision = 0
            w.write(4, 3); // global tree, default weighted predictor, no transforms
            emit_tokens(&dc_tokens[i], &dc_code_ref, w);

            let num_blocks = data.ac_strategy.xsize() * data.ac_strategy.ysize();
            let nb_bits = if num_blocks <= 1 {
                0
            } else {
                usize::BITS as usize
                    - num_blocks.leading_zeros() as usize
                    - if num_blocks.is_power_of_two() { 1 } else { 0 }
            };
            if nb_bits != 0 {
                w.write(nb_bits, (num_blocks - 1) as u64);
            }
            w.write(4, 3);
            emit_tokens(&meta_tokens[i], &dc_code_ref, w);
            section
        });

    let mut sections = Vec::with_capacity(2 + dim.num_dc_groups + dim.num_groups);
    sections.push(dc_global);
    sections.extend(dc_group_sections);
    sections.push(ac.global);
    sections.extend(ac.groups);

    // Assemble
    let mut out = BitWriter::new();
    out.write(8, 0xFF);
    out.write(8, 0x0A);
    write_size_header(&mut out, dim.xsize, dim.ysize);
    write_image_metadata(&mut out, icc, scratch);
    write_frame_header(&mut out, ss);
    combine_sections(&mut sections, &mut out);
    Ok(out.into_bytes())
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

/// The AC-global section and per-group sections for one coefficient ordering,
/// with the total size used to choose between orderings.
struct AcSections {
    global: BitWriter,
    groups: Vec<BitWriter>,
    bytes: usize,
}

/// Tokenizes and entropy-codes the AC coefficients under `scan`, returning the
/// finished sections and their total byte size.
///
/// `perm` carries the coefficient-order permutation tokens when `scan` is a
/// custom order; `None` signals the natural order (`used_orders = 0`).
#[allow(clippy::too_many_arguments)]
fn build_ac_sections(
    jpg: &JpegData,
    dim: &Dim,
    ss: &Subsampling,
    qtable: &[i32; 3 * DCT_BLOCK_SIZE],
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    perm: Option<Vec<Token>>,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> AcSections {
    let ac_tokens = tokenize_ac(jpg, dim, ss, scan, pool, scratch);
    let mut all_ac: Vec<Token> = Vec::new();
    for t in &ac_tokens {
        all_ac.extend_from_slice(t);
    }
    let ac_code = optimize_entropy_code_ac(&all_ac, K_NUM_AC_CONTEXTS, &mut scratch.huffman_pool);
    let ac_code_ref = ac_code.as_ref();

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
        write_entropy_code(&ac_code_ref, &mut scratch.huffman_pool, w);
    }

    let groups: Vec<BitWriter> = pool.steal_map(scratch, dim.num_groups, |g, _scratch| {
        let mut section = BitWriter::new();
        emit_tokens(&ac_tokens[g], &ac_code_ref, &mut section);
        section
    });

    let bytes = global.bits_written().div_ceil(8)
        + groups
            .iter()
            .map(|s| s.bits_written().div_ceil(8))
            .sum::<usize>();
    AcSections {
        global,
        groups,
        bytes,
    }
}

/// Derives the per-channel DCT8 coefficient order from non-zero statistics.
fn compute_coeff_orders(jpg: &JpegData, dim: &Dim, ss: &Subsampling) -> [[u8; DCT_BLOCK_SIZE]; 3] {
    // Slot -> source coefficient index: the natural slot's transposed-raster
    // position mapped back into the JPEG block's own raster layout.
    let mut slot_to_src = [0usize; DCT_BLOCK_SIZE];
    for (slot, s) in slot_to_src.iter_mut().enumerate() {
        let r = K_COEFF_ORDER_8X8[slot] as usize;
        *s = (r & 7) * 8 + (r >> 3);
    }

    let mut orders = [[0u8; DCT_BLOCK_SIZE]; 3];
    for c in 0..3usize {
        let comp = &jpg.components[JPEG_ORDER_YCBCR[c]];
        let cbw = dim.xsize_blocks >> ss.hshift[c];
        let cbh = dim.ysize_blocks >> ss.vshift[c];
        let mut nonzero = [0u64; DCT_BLOCK_SIZE];
        for by in 0..cbh {
            for bx in 0..cbw {
                let base = (by * comp.width_in_blocks + bx) * DCT_BLOCK_SIZE;
                let coeffs = &comp.coeffs[base..base + DCT_BLOCK_SIZE];
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
    ss: &Subsampling,
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Vec<Vec<Token>> {
    pool.steal_map(scratch, dim.num_groups, |g, _scratch| {
        let gx = g % dim.xsize_groups;
        let gy = g / dim.xsize_groups;
        tokenize_ac_group(jpg, dim, ss, scan, gx, gy)
    })
}

/// Tokenizes the AC coefficients of one group.
fn tokenize_ac_group(
    jpg: &JpegData,
    dim: &Dim,
    ss: &Subsampling,
    scan: &[[u8; DCT_BLOCK_SIZE]; 3],
    gx: usize,
    gy: usize,
) -> Vec<Token> {
    let mut block = [0i32; DCT_BLOCK_SIZE];
    {
        {
            let bx0 = gx * GROUP_DIM_IN_BLOCKS;
            let by0 = gy * GROUP_DIM_IN_BLOCKS;
            let bw = GROUP_DIM_IN_BLOCKS.min(dim.xsize_blocks - bx0);
            let bh = GROUP_DIM_IN_BLOCKS.min(dim.ysize_blocks - by0);

            let mut tokens: Vec<Token> = Vec::new();
            // Non-zero counts of already-coded neighbors, used as context.
            let mut num_nzeros = Image3B::new(GROUP_DIM_IN_BLOCKS, GROUP_DIM_IN_BLOCKS);

            for by in 0..bh {
                for bx in 0..bw {
                    for &c in &CHANNEL_ORDER {
                        // One block per subsampling cell, at its corner.
                        let (hs, vs) = (ss.hshift[c], ss.vshift[c]);
                        let abs_bx = bx0 + bx;
                        let abs_by = by0 + by;
                        if (abs_bx >> hs) << hs != abs_bx || (abs_by >> vs) << vs != abs_by {
                            continue;
                        }
                        let (sx, sy) = (bx >> hs, by >> vs);
                        let comp = &jpg.components[JPEG_ORDER_YCBCR[c]];
                        let bi = (abs_by >> vs) * comp.width_in_blocks + (abs_bx >> hs);
                        let src = &comp.coeffs[bi * DCT_BLOCK_SIZE..(bi + 1) * DCT_BLOCK_SIZE];
                        // JXL transposes the DCT relative to JPEG.
                        for (y, src_row) in src.as_chunks::<8>().0.iter().enumerate() {
                            for (x, &src) in src_row.iter().enumerate() {
                                block[x * 8 + y] = src as i32;
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

                        // Always a plain 8x8 DCT, so strategy code 0.
                        let block_ctx = block_context(c, 0);
                        let nzero_ctx = non_zero_context(predicted as u32, block_ctx);
                        let histo_offset =
                            crate::ac_context::zero_density_contexts_offset(block_ctx);

                        tokens.push(Token::new(nzero_ctx, nzeros));

                        let mut prev = if nzeros as usize > DCT_BLOCK_SIZE / 16 {
                            0
                        } else {
                            1
                        };
                        let mut remaining = nzeros;
                        let mut k = 1usize;
                        while k < DCT_BLOCK_SIZE && remaining != 0 {
                            let coef = block[scan[c][k] as usize];
                            let ctx = histo_offset as usize
                                + zero_density_context_8x8(remaining as usize, k, prev);
                            tokens.push(Token::new(ctx as u32, pack_signed(coef)));
                            prev = if coef != 0 { 1 } else { 0 };
                            if coef != 0 {
                                remaining -= 1;
                            }
                            k += 1;
                        }
                    }
                }
            }
            tokens
        }
    }
}
