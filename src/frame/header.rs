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

//! VarDCT frame headers, global sections and quantization matrices.

use super::dc_tokens::{DcTreeChoice, write_context_tree, write_tree_tokens};
use super::extra_channels::ExtraSections;
use super::quantization::DistanceParams;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::encoding_context::FrameRole;
use crate::entropy::{EntropyCode, write_entropy_code};
use crate::patches::{PATCH_REF_ID, VarDctFrameKind};
use crate::quant_weights::quant_table_slot_of;

fn write_frame_dimension(value: usize, w: &mut BitWriter) {
    let value = value as u64;
    if value < 256 {
        w.write(2, 0);
        w.write(8, value);
    } else if value < 2304 {
        w.write(2, 1);
        w.write(11, value - 256);
    } else if value < 18688 {
        w.write(2, 2);
        w.write(14, value - 2304);
    } else {
        w.write(2, 3);
        w.write(30, value - 18688);
    }
}

pub(super) fn write_frame_header_kind(
    x_qm_scale: u32,
    b_qm_scale: u32,
    epf_iters: u32,
    epf_pass0_scale: Option<f32>,
    epf_quant_mul: Option<f32>,
    gab_enabled: bool,
    has_alpha: bool,
    coeff_shifts: &[u32],
    kind: VarDctFrameKind<'_>,
    has_splines: bool,
    has_noise: bool,
    skip_dc_smoothing: bool,
    use_dc_frame: bool,
    coding: crate::coding::CodingTransform,
    epf_channel_scale: Option<[f32; 3]>,
    role: FrameRole,
    w: &mut BitWriter,
) {
    match kind {
        VarDctFrameKind::Regular => write_frame_header(
            x_qm_scale,
            b_qm_scale,
            epf_iters,
            epf_pass0_scale,
            epf_quant_mul,
            gab_enabled,
            has_alpha,
            coeff_shifts,
            false,
            has_splines,
            has_noise,
            skip_dc_smoothing,
            use_dc_frame,
            coding,
            epf_channel_scale,
            role,
            w,
        ),
        VarDctFrameKind::Patched(_) => write_frame_header(
            x_qm_scale,
            b_qm_scale,
            epf_iters,
            epf_pass0_scale,
            epf_quant_mul,
            gab_enabled,
            has_alpha,
            coeff_shifts,
            true,
            has_splines,
            has_noise,
            skip_dc_smoothing,
            use_dc_frame,
            coding,
            epf_channel_scale,
            role,
            w,
        ),
        VarDctFrameKind::ReferenceOnly { width, height } => {
            debug_assert!(coding.is_xyb(), "patch atlases are XYB-only");
            w.write(1, 0); // not all default
            w.write(2, 0b10); // reference-only frame
            w.write(1, 0); // VarDCT
            w.write(2, 0); // flags = 0
            w.write(2, 0); // upsampling = 1
            if has_alpha {
                w.write(2, 0); // extra-channel upsampling = 1
            }
            w.write(3, x_qm_scale as u64);
            w.write(3, b_qm_scale as u64); // b_qm_scale
            // Reference-only frames omit Passes and use the implicit one pass.
            w.write(1, 1); // custom size
            write_frame_dimension(width, w);
            write_frame_dimension(height, w);
            w.write(2, PATCH_REF_ID as u64);
            w.write(1, 1); // save_before_color_transform
            w.write(2, 0); // empty name
            write_loop_filter(
                epf_iters,
                epf_pass0_scale,
                epf_quant_mul,
                gab_enabled,
                None,
                w,
            );
            w.write(2, 0); // no frame-header extensions
        }
    }
}

fn write_loop_filter(
    epf_iters: u32,
    epf_pass0_scale: Option<f32>,
    epf_quant_mul: Option<f32>,
    gab_enabled: bool,
    epf_channel_scale: Option<[f32; 3]>,
    w: &mut BitWriter,
) {
    if epf_iters == 2 && gab_enabled && epf_channel_scale.is_none() && epf_quant_mul.is_none() {
        w.write(1, 1); // default loop filter (gab=1, epf=2)
    } else {
        w.write(1, 0); // not default
        if gab_enabled {
            w.write(1, 1); // gaborish enabled
            w.write(1, 0); // gab_custom = false (use defaults)
        } else {
            w.write(1, 0); // no gaborish
        }
        w.write(2, epf_iters as u64);
        if epf_iters > 0 {
            w.write(1, 0); // default epf sharpness
            match epf_channel_scale {
                None => w.write(1, 0), // default epf weights
                Some(scale) => {
                    w.write(1, 1); // custom epf weights
                    // Channel scales, then the spec pass-1/2 zero-flush levels.
                    for v in [scale[0], scale[1], scale[2], 0.45, 0.6] {
                        w.write(16, u64::from(crate::util::f32_to_f16_bits(v)));
                    }
                }
            }
            // epf_sigma_custom: [quant_mul, pass0_scale, pass2_scale,
            // border_sad_mul], spec defaults 0.46, 0.9, 6.5, 2/3. The fitter
            // varies quant_mul and pass0; the other two stay at spec.
            if let Some(s) = epf_pass0_scale.or(epf_quant_mul.map(|_| 0.9)) {
                let c = [epf_quant_mul.unwrap_or(0.46), s, 6.5, 2.0 / 3.0];
                w.write(1, 1); // custom epf sigma
                for x in c {
                    w.write(16, crate::util::f32_to_f16_bits(x) as u64);
                }
            } else {
                w.write(1, 0); // default epf sigma
            }
        }
        w.write(2, 0); // no loop filter extensions
    }
}

fn write_frame_header(
    x_qm_scale: u32,
    b_qm_scale: u32,
    epf_iters: u32,
    epf_pass0_scale: Option<f32>,
    epf_quant_mul: Option<f32>,
    gab_enabled: bool,
    has_alpha: bool,
    coeff_shifts: &[u32],
    has_patches: bool,
    has_splines: bool,
    has_noise: bool,
    skip_dc_smoothing: bool,
    use_dc_frame: bool,
    coding: crate::coding::CodingTransform,
    epf_channel_scale: Option<[f32; 3]>,
    role: FrameRole,
    w: &mut BitWriter,
) {
    w.write(1, 0); // not all default
    w.write(2, 0); // regular frame
    w.write(1, 0); // vardct
    // Optional flags: kNoise (1), kPatches (2), kSplines (16), kUseDcFrame
    // (32) and kSkipAdaptiveDCSmoothing (128), chosen per frame by
    // `skip_dc_smoothing`. A DC frame implies no smoothing.
    let flags = if has_noise {
        crate::noise::FLAG_NOISE
    } else {
        0
    } + if has_patches { 2u64 } else { 0 }
        + if has_splines { 16 } else { 0 }
        + if use_dc_frame { 32 } else { 0 }
        + if skip_dc_smoothing && !use_dc_frame {
            128
        } else {
            0
        };
    match flags {
        0 => w.write(2, 0),
        1..=16 => {
            w.write(2, 1); // U64 selector for values 1..=16
            w.write(4, flags - 1);
        }
        _ => {
            w.write(2, 2); // U64 selector for values 17..=272
            w.write(8, flags - 17);
        }
    }
    if !coding.is_xyb() {
        let ycbcr = coding == crate::coding::CodingTransform::YCbCr;
        w.write(1, u64::from(ycbcr)); // do_YCbCr
        if ycbcr {
            for _ in 0..3 {
                w.write(2, 0); // chroma subsampling 4:4:4
            }
        }
    }
    // Absent with kUseDcFrame.
    if !use_dc_frame {
        w.write(2, 0); // no upsampling

        // Per-extra-channel upsampling. Same u2S(1,2,4,8) code, default 1:
        // selector "00" gives 1, written once per extra channel.
        if has_alpha {
            w.write(2, 0); // ec_upsampling[0] = 1
        }
    }

    // The quant scales are only serialized for XYB frames.
    if coding.is_xyb() {
        w.write(3, x_qm_scale as u64);
        w.write(3, b_qm_scale as u64); // b_qm_scale
    }
    // Passes bundle (jxl-frame header.rs:127-132):
    //   num_passes: U32(1,2,3,4+u(3))   default 1
    //   if num_passes != 1:
    //     num_ds:    U32(0,1,2,3+u(1))  -> 0 (no downsampling)
    //     shift:     Vec[u(2)] len (num_passes-1)
    //     downsample/last_pass: Vec len num_ds (empty here)
    // For VarDCT AC the per-pass coeff_shift used by the decoder is
    // passes.shift[pass]; the last pass implicitly has shift 0, so we only emit
    // shifts for passes 0..num_passes-1 and require coeff_shifts.last()==0.
    let num_passes = coeff_shifts.len();
    debug_assert!((1..=11).contains(&num_passes), "num_passes out of range");
    debug_assert!(num_passes == 1 || coeff_shifts[num_passes - 1] == 0);
    if num_passes == 1 {
        w.write(2, 0); // num_passes = 1 (U32 selector 0)
    } else {
        // U32(1,2,3,4+u(3)): selectors 0->1, 1->2, 2->3, 3->4+u(3).
        match num_passes {
            2 => w.write(2, 1),
            3 => w.write(2, 2),
            n => {
                w.write(2, 3);
                w.write(3, (n - 4) as u64);
            }
        }
        w.write(2, 0); // num_ds = 0 (U32 selector 0)
        for &s in &coeff_shifts[..num_passes - 1] {
            w.write(2, s as u64); // shift[p] = coeff_shift of pass p (u(2))
        }
    }
    w.write(1, 0); // no custom frame size or origin

    // Color-channel BlendingInfo: mode=Replace, full_frame=true means
    // source/alpha_channel/clamp are all omitted. The two zero bits select
    // BlendingMode::Replace.
    w.write(2, 0); // color blend mode = Replace

    // Per-extra-channel BlendingInfo. With num_extra_channels=1, BlendingInfo
    // for the alpha channel has:
    //   mode = Replace (2 bits = 00)
    //   alpha_channel: NOT written (mode != Blend/AWA)
    //   clamp: NOT written (mode != Blend/AWA/Mul)
    //   source: NOT written (full_frame && Replace)
    if has_alpha {
        w.write(2, 0); // ec_blending_info[0].mode = Replace
    }

    match role {
        FrameRole::Last => w.write(1, 1), // last frame
        FrameRole::SavedFirst => {
            w.write(1, 0); // not the last frame
            w.write(2, 0); // save_as_reference = 0
            w.write(1, 0); // save_before_color_transform = false
        }
    }
    w.write(2, 0); // no name
    write_loop_filter(
        epf_iters,
        epf_pass0_scale,
        epf_quant_mul,
        gab_enabled,
        epf_channel_scale,
        w,
    );
    w.write(2, 0); // no frame header extensions
}

pub(crate) fn write_quant_scales(global_scale: i32, quant_dc: i32, w: &mut BitWriter) {
    if global_scale < 2049 {
        w.write(2, 0);
        w.write(11, (global_scale - 1) as u64);
    } else if global_scale < 4097 {
        w.write(2, 1);
        w.write(11, (global_scale - 2049) as u64);
    } else if global_scale < 8193 {
        w.write(2, 2);
        w.write(12, (global_scale - 4097) as u64);
    } else {
        w.write(2, 3);
        w.write(16, (global_scale - 8193) as u64);
    }
    if quant_dc == 16 {
        w.write(2, 0);
    } else if quant_dc < 33 {
        w.write(2, 1);
        w.write(5, (quant_dc - 1) as u64);
    } else if quant_dc < 257 {
        w.write(2, 2);
        w.write(8, (quant_dc - 1) as u64);
    } else {
        w.write(2, 3);
        w.write(16, (quant_dc - 1) as u64);
    }
}

/// Serialize the per-image BlockCtxMap: no dc thresholds, the optional
/// quant-field threshold, and the ctx_map (qf inner dimension when present).
pub(super) fn write_block_ctx_map(
    plan: &crate::ac_context::AcCtxPlan,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    w.write(1, 0); // non-default BlockCtxMap
    w.write(4, 0); // dc thresholds, channel 0
    w.write(4, 0); // dc thresholds, channel 1
    w.write(4, 0); // dc thresholds, channel 2
    match plan.qf_threshold {
        None => w.write(4, 0),
        Some(t) => {
            w.write(4, 1);
            // kQFThresholdDist: U32(Bits(2), BitsOffset(3,4), BitsOffset(5,12),
            // BitsOffset(8,44)) over t - 1.
            let v = t - 1;
            if v < 4 {
                w.write(2, 0);
                w.write(2, u64::from(v));
            } else if v < 12 {
                w.write(2, 1);
                w.write(3, u64::from(v - 4));
            } else if v < 44 {
                w.write(2, 2);
                w.write(5, u64::from(v - 12));
            } else {
                w.write(2, 3);
                w.write(8, u64::from(v - 44));
            }
        }
    }
    let entries = plan.ctx_map_entries();
    let empty_codes: [crate::entropy::PrefixCode; 0] = [];
    let empty_configs: [crate::entropy::HybridUintConfig; 0] = [];
    let empty_histograms = [];
    let empty_syms: [Vec<crate::entropy::AnsEncSymbolInfo>; 0] = [];
    let empty_reverse_maps: [u16; 0] = [];
    let cm_entropy = EntropyCode {
        context_map: &entries,
        num_contexts: entries.len(),
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

pub(super) fn write_dc_global(
    distp: &DistanceParams,
    num_dc_groups: usize,
    ac_plan: &crate::ac_context::AcCtxPlan,
    dc_tree: &DcTreeChoice,
    dc_code: &EntropyCode,
    ytox_dc: i32,
    ytob_dc: i32,
    cfl: crate::color_correlation::CflFrame,
    extra: Option<&ExtraSections>,
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    if let Some(steps) = crate::color_correlation::dc_step_wire(distp.dc_step) {
        w.write(1, 0);
        for step in steps {
            w.write(16, u64::from(step));
        }
    } else {
        w.write(1, 1); // default dequant DC
    }
    write_quant_scales(distp.global_scale, distp.quant_dc, w);
    write_block_ctx_map(ac_plan, scratch, w);

    // ColorCorrelationParams. The all-default bundle pins the DC plane to the
    // XYB base correlations (X: 0, B: 1); either searched DC slope needs the
    // explicit form, which costs COLOR_CORRELATION_HEADER_BITS more.
    {
        let factor = cfl.color_factor as u32;
        if ytox_dc == 0 && ytob_dc == 0 && cfl == crate::color_correlation::CflFrame::XYB {
            w.write(1, 1); // all_default
        } else {
            w.write(1, 0); // not all-default
            // color_factor: U32(Val(84), Val(256), BitsOffset(8, 2), BitsOffset(16, 258)).
            if factor == 84 {
                w.write(2, 0);
            } else {
                w.write(2, 2);
                w.write(8, (factor - 2) as u64);
            }
            for base in [cfl.base_x, cfl.base_b] {
                w.write(16, u64::from(crate::util::f32_to_f16_bits(base)));
            }
            w.write(8, (ytox_dc + 128) as u64); // ytox_dc, offset by 128
            w.write(8, (ytob_dc + 128) as u64); // ytob_dc, offset by 128
        }
    }

    // Global tree.
    // write_context_tree emits "have_tree=1 + Histograms::decode (tree's own entropy code)
    // + tree tokens". The TREE'S PIXEL HISTOGRAMS are then written as the next two
    // bits + entropy code (DC entropy code, since it's the global tree used for DC).
    match dc_tree {
        DcTreeChoice::Static(grad) => {
            write_context_tree(num_dc_groups, grad, &mut scratch.huffman_pool, w);
        }
        DcTreeChoice::Learned(tokens) => {
            write_tree_tokens(tokens, &mut scratch.huffman_pool, w);
        }
    }
    w.write(1, 0); // no lz77 (for the global tree's pixel histograms = dc_code)

    // Then the static DC entropy code: this is the global tree's pixel histograms.
    write_entropy_code(dc_code, &mut scratch.huffman_pool, w);

    // FullModularImage::read happens HERE in the decoder. If we declared an alpha
    // extra channel, write its GroupHeader + local tree + pixel data now.
    if let Some(extra) = extra {
        w.append(&extra.global);
    }
}

/// Slots whose transform actually appears in the frame: the nine
/// `quant_table_slot_of` slots, then DCT2X2, DCT4X4 and AFV, then DCT32X8/8X32.
pub(super) fn used_quant_table_slots(dc_datas: &[DcGroupData]) -> [bool; 13] {
    use crate::dc_group_data::{
        STRATEGY_AFV0, STRATEGY_AFV1, STRATEGY_AFV2, STRATEGY_AFV3, STRATEGY_DCT2X2,
        STRATEGY_DCT4X4,
    };
    let mut used = [false; 13];
    for dc in dc_datas {
        for (_, _, strategy) in dc.ac_strategy.iter_first_blocks() {
            if let Some(slot) = quant_table_slot_of(strategy) {
                used[slot] = true;
            }
            match strategy {
                STRATEGY_DCT2X2 => used[9] = true,
                STRATEGY_DCT4X4 => used[10] = true,
                STRATEGY_AFV0 | STRATEGY_AFV1 | STRATEGY_AFV2 | STRATEGY_AFV3 => used[11] = true,
                _ => {}
            }
        }
    }
    used
}

fn write_dequant_matrices(
    matrices: &crate::quant_weights::DequantMatrices,
    used: &[bool; 13],
    w: &mut BitWriter,
) {
    use crate::util::f32_to_f16_bits;
    // A table only earns its header if the frame actually uses the transform.
    let table = |slot: usize| -> Option<&crate::quant_weights::BandOverride> {
        used[slot]
            .then(|| matrices.custom_tables[slot].as_ref())
            .flatten()
    };
    let identity = used[8].then_some(matrices.identity_weights).flatten();
    let fine = matrices.fine_tables.as_deref();
    let dct2 = fine.filter(|_| used[9]).map(|f| f.dct2);
    let dct4x4 = fine.filter(|_| used[10]).map(|f| f.dct4x4);
    let afv = fine.filter(|_| used[11]).map(|f| f.afv);
    // DCT32X8 / DCT8X32 always carry their pair-parity table.
    let t8x32 = used[12].then(|| matrices.band_8x32());
    if (0..8).all(|slot| table(slot).is_none())
        && t8x32.is_none()
        && identity.is_none()
        && dct2.is_none()
        && dct4x4.is_none()
        && afv.is_none()
    {
        w.write(1, 1); // all_default
        return;
    }
    let write_dct_params = |bands: &[[f32; 16]; 3], num_bands: usize, w: &mut BitWriter| {
        w.write(4, num_bands as u64 - 1);
        for row in bands {
            for (i, &v) in row[..num_bands].iter().enumerate() {
                let wire = if i == 0 { v / 64.0 } else { v };
                w.write(16, u64::from(f32_to_f16_bits(wire)));
            }
        }
    };
    let pad4 = |rows: &[[f32; 4]; 3]| -> [[f32; 16]; 3] {
        std::array::from_fn(|c| std::array::from_fn(|i| if i < 4 { rows[c][i] } else { 0.0 }))
    };
    const K_QUANT_MODE_ID: u64 = 1;
    const K_QUANT_MODE_DCT2: u64 = 2;
    const K_QUANT_MODE_DCT4: u64 = 3;
    const K_QUANT_MODE_AFV: u64 = 5;
    const K_NUM_QUANT_TABLES: usize = 17;
    const K_QUANT_MODE_LIBRARY: u64 = 0;
    // `kQuantModeDCT4X8` carries three F16 `dct4x8multipliers` ahead of the
    // shared DctQuantWeightParams payload; the library table uses 1.0 for all
    // three, and jixel's 4x8 matrix is built without them, so identity is
    // preserved by writing 1.0.
    const K_QUANT_MODE_DCT4X8: u64 = 4;
    const K_QUANT_MODE_DCT: u64 = 6;
    const TABLE_DCT4X8: usize = 9;
    w.write(1, 0); // all_default = false
    for idx in 0..K_NUM_QUANT_TABLES {
        let bands = match idx {
            0 => table(0),            // DCT8
            4 => table(1),            // DCT16X16
            5 => table(2),            // DCT32X32
            6 => table(3),            // DCT8X16 (= DCT16X8)
            8 => table(4),            // DCT16X32 (= DCT32X16)
            TABLE_DCT4X8 => table(5), // DCT4X8 (= DCT8X4)
            11 => table(7),           // DCT64X64
            12 => table(6),           // DCT32X64 (= DCT64X32)
            _ => None,
        };
        if idx == 1
            && let Some(id) = identity
        {
            w.write(3, K_QUANT_MODE_ID);
            for row in &id {
                for &v in row {
                    w.write(16, u64::from(f32_to_f16_bits(v / 64.0)));
                }
            }
            continue;
        }
        if idx == 2
            && let Some(dw) = dct2
        {
            w.write(3, K_QUANT_MODE_DCT2);
            for row in &dw {
                for &v in row {
                    w.write(16, u64::from(f32_to_f16_bits(v / 64.0)));
                }
            }
            continue;
        }
        if idx == 3
            && let Some(b4) = dct4x4
        {
            w.write(3, K_QUANT_MODE_DCT4);
            for _ in 0..3 {
                w.write(16, 0x3C00); // dct4multipliers[c][0] = 1.0
                w.write(16, 0x3C00); // dct4multipliers[c][1] = 1.0
            }
            write_dct_params(&pad4(&b4), 4, w);
            continue;
        }
        if idx == 10
            && let Some(aw) = afv
        {
            // AFV carries its own 4x8 and 4x4 parameters; the 4x8 ones are
            // the frame's DCT4X8 table, the 4x4 ones its DCT4X4 table.
            let o4x8 = matrices.custom_tables[5]
                .as_ref()
                .expect("custom AFV tables come with a custom DCT4X8 table");
            let b4 = fine
                .expect("custom AFV tables come with the fine tables")
                .dct4x4;
            w.write(3, K_QUANT_MODE_AFV);
            for row in &aw {
                for (i, &v) in row.iter().enumerate() {
                    let wire = if i < 6 { v / 64.0 } else { v };
                    w.write(16, u64::from(f32_to_f16_bits(wire)));
                }
            }
            write_dct_params(&o4x8.bands, 4, w);
            write_dct_params(&pad4(&b4), 4, w);
            continue;
        }
        let bands = if idx == 7 { t8x32 } else { bands };
        match bands {
            None => w.write(3, K_QUANT_MODE_LIBRARY),
            Some(o) => {
                if idx == TABLE_DCT4X8 {
                    w.write(3, K_QUANT_MODE_DCT4X8);
                    for _ in 0..3 {
                        w.write(16, 0x3C00); // dct4x8multipliers[c] = 1.0
                    }
                } else {
                    w.write(3, K_QUANT_MODE_DCT);
                }
                w.write(4, o.num_bands as u64 - 1);
                for row in &o.bands {
                    for (i, &v) in row[..o.num_bands].iter().enumerate() {
                        let wire = if i == 0 { v / 64.0 } else { v };
                        w.write(16, u64::from(f32_to_f16_bits(wire)));
                    }
                }
            }
        }
    }
}

pub(super) fn write_ac_global(
    matrices: &crate::quant_weights::DequantMatrices,
    used_quant_tables: &[bool; 13],
    coeff_orders: &crate::coeff_order::CoeffOrders,
    num_groups: usize,
    ac_codes: &[crate::entropy::OwnedEntropyCode],
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) {
    write_dequant_matrices(matrices, used_quant_tables, w);
    if num_groups > 1 {
        let bits = 32
            - (num_groups as u32).leading_zeros()
            - if num_groups.is_power_of_two() { 1 } else { 0 };
        if bits != 0 {
            w.write(bits as usize, 0);
        }
    }
    // HfGlobal parses `num_passes` HfPass blocks (jxl-frame hf_global.rs:57-59),
    // each = used_orders(U32 sel 3 + u(13)=0 -> natural order) + hf_dist entropy
    // code. Each pass gets its own code (ac_codes[p]).
    for code in ac_codes {
        crate::coeff_order::write_coeff_orders(coeff_orders, &mut scratch.huffman_pool, w);
        w.write(1, 0);
        write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, w);
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn arithmetic_matrix_header_cost_matches_writer_for_every_subset() {
        use crate::quant_weights::{DequantMatrices, MatrixHeaderCost};
        for distance in [0.5, 1.5, 2.5, 4.0, 8.0] {
            for matrices in [
                DequantMatrices::new(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_pair_b(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_saturated(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_x_heavy(distance, crate::Arbiter::Ssimulacra2),
                DequantMatrices::new_saturated_pair_b(distance, crate::Arbiter::Ssimulacra2, false),
                DequantMatrices::new_fast(distance, crate::Arbiter::Ssimulacra2),
            ] {
                let header = MatrixHeaderCost::new(matrices);
                for mask in (0..512u16).flat_map(|m| [m, m | 1 << 12]) {
                    let used = std::array::from_fn(|slot| mask & (1 << slot) != 0);
                    let mut writer = crate::bit_writer::BitWriter::new();
                    super::write_dequant_matrices(matrices, &used, &mut writer);
                    assert_eq!(
                        header.bits(mask),
                        writer.bits_written(),
                        "d={distance} mask={mask}"
                    );
                }
            }
        }
    }
}
