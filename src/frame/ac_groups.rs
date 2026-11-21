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

//! Quantize and tokenize individual AC groups, and gather their reconstructions.

use super::quantization::DistanceParams;
use super::{
    ImageDim, K_BLOCK_DIM, K_DC_GROUP_DIM, K_GROUP_DIM, K_GROUP_DIM_IN_BLOCKS, K_TILE_DIM,
    K_TILE_DIM_IN_BLOCKS,
};
use crate::coder_scratch::CoderScratch;
use crate::dc_group_data::DcGroupData;
use crate::encoding_context::EncodingContext;
use crate::entropy::{AcTokens, Token};
use crate::group::write_ac_group;
use crate::image::{Image3B, Image3F, Image3S, Rect};

/// Copies one AC group's reconstruction capture into the frame image.
pub(super) fn paste_group(target: &mut Image3F, dim: &ImageDim, group_idx: usize, image: &Image3F) {
    let x0 = (group_idx % dim.xsize_groups) * K_GROUP_DIM;
    let y0 = (group_idx / dim.xsize_groups) * K_GROUP_DIM;
    for c in 0..3 {
        for y in 0..image.ysize() {
            target.plane_row_mut(c, y0 + y)[x0..x0 + image.xsize()]
                .copy_from_slice(image.plane_row(c, y));
        }
    }
}

/// `process_ac_group` output: tokens, the group's quantized DC, the unrounded
/// source DC and order statistics when collected, and the reconstruction
/// capture with per-block token starts in loop mode.
type ProcessedAcGroup = (
    PendingAcGroup,
    Image3S,
    Option<crate::group::SourceDc>,
    Option<crate::coeff_order::OrderStats>,
    Option<(crate::image::Image3F, Vec<(u16, u16, u32)>)>,
);

/// One coded AC group of the refined pass: DC-group index, group position,
/// tokens and quantized DC.
pub(super) type RefinedAcGroup = (usize, usize, usize, PendingAcGroup, Image3S);

/// Per-AC-group buffered tokens. For progressive (multi-pass) encoding the
/// quantized AC coefficients of each block are split across passes; `tokens`
/// holds one token stream per pass, plus temporary alternative scan streams
/// in ExtraSlow's final refine pass. `group_idx` is the raster group index
/// (0..num_groups); the section for (pass, group) is
/// `2 + num_dc_groups + pass*num_groups + group_idx`.
#[derive(Clone)]
pub(super) struct PendingAcGroup {
    pub(super) group_idx: usize,
    pub(super) tokens: Vec<AcTokens>,
}

#[derive(Clone, Copy)]
pub(super) enum AcGroupCapture {
    None,
    Pixels,
    #[cfg(feature = "extraslow")]
    Loop,
}

/// Median raw quant-field value across the image (blocks weighted equally).
pub(super) fn ac_qf_threshold(dc_datas: &[DcGroupData]) -> u32 {
    let mut histogram = [0u64; 257];
    for dc_data in dc_datas {
        for y in 0..dc_data.raw_quant_field.ysize() {
            for &qf in dc_data.raw_quant_field.row(y) {
                histogram[(qf as usize).min(256)] += 1;
            }
        }
    }
    let total: u64 = histogram.iter().sum();
    let mut acc = 0u64;
    for (value, &count) in histogram.iter().enumerate() {
        acc += count;
        if acc * 2 >= total {
            return value as u32;
        }
    }
    1
}

/// Encode a single AC group: build its tile stripes, quantize and tokenize,
/// and place its DC coefficients into a returned group-local `quant_dc`
/// (origin-relative, merged by the caller). Reads `dc_data` read-only.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_ac_group(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    dim: &ImageDim,
    distp: &DistanceParams,
    dc_data: &DcGroupData,
    num_passes: usize,
    coeff_shifts: &[u32],
    dc_gx: usize,
    dc_gy: usize,
    gx: usize,
    gy: usize,
    ytob_dc: i32,
    rdoq_prices: Option<&crate::entropy::FrozenTokenPrices>,
    coeff_orders: &crate::coeff_order::CoeffOrders,
    alternative_orders: &[crate::coeff_order::CoeffOrders],
    collect_order_stats: bool,
    qf_threshold: u32,
    capture: AcGroupCapture,
    block_memo: Option<&mut Option<crate::group::AcBlockCache>>,
) -> ProcessedAcGroup {
    scratch.finish_strategy_search();
    let image_gx = dc_gx * (K_DC_GROUP_DIM / K_GROUP_DIM) + gx;
    let image_gy = dc_gy * (K_DC_GROUP_DIM / K_GROUP_DIM) + gy;
    let group_x0 = image_gx * K_GROUP_DIM;
    let group_y0 = image_gy * K_GROUP_DIM;
    let group_xsize = K_GROUP_DIM.min(dim.xsize.saturating_sub(group_x0));
    let group_ysize = K_GROUP_DIM.min(dim.ysize.saturating_sub(group_y0));
    let group_ysize_tiles = group_ysize.div_ceil(K_TILE_DIM);
    let gwb = group_xsize.div_ceil(K_BLOCK_DIM);
    let ghb = group_ysize.div_ceil(K_BLOCK_DIM);
    let qorigin_x = gx * K_GROUP_DIM_IN_BLOCKS;
    let qorigin_y = gy * K_GROUP_DIM_IN_BLOCKS;

    let capture_recon = !matches!(capture, AcGroupCapture::None);
    #[cfg(feature = "extraslow")]
    let capture_loop = matches!(capture, AcGroupCapture::Loop);
    #[cfg(not(feature = "extraslow"))]
    let capture_loop = false;
    let mut local_quant_dc = Image3S::new(gwb, ghb);
    let mut source_dc = (collect_order_stats || capture_loop).then(|| crate::group::SourceDc {
        y: crate::image::Plane::new(gwb, ghb),
        b: crate::image::Plane::new(gwb, ghb),
        x: crate::image::Plane::new(gwb, ghb),
    });
    let mut num_nzeros: Vec<Image3B> = (0..num_passes)
        .map(|_| Image3B::new(K_GROUP_DIM_IN_BLOCKS, K_GROUP_DIM_IN_BLOCKS))
        .collect();
    // Alternative streams walk the final coefficients without quantizing again.
    debug_assert!(alternative_orders.is_empty() || num_passes == 1);
    let mut tokens: Vec<Vec<Token>> = (0..num_passes + alternative_orders.len())
        .map(|_| Vec::with_capacity(K_GROUP_DIM_IN_BLOCKS * K_GROUP_DIM_IN_BLOCKS * 4))
        .collect();
    let mut order_stats = collect_order_stats.then(crate::coeff_order::OrderStats::new);
    // In memo mode the memo's reconstruction is the capture target: hits copy
    // no pixels and the caller syncs only the recomputed cells.
    debug_assert!(block_memo.is_none() || capture_recon);
    let mut memo = block_memo.map(|slot| {
        slot.get_or_insert_with(|| {
            crate::group::AcBlockCache::new(gwb, ghb, group_xsize, group_ysize)
        })
    });
    if let Some(m) = memo.as_deref_mut() {
        m.epoch += 1;
    }
    let mut recon = capture_recon.then(|| {
        let plane = match memo.as_deref_mut() {
            Some(m) => std::mem::replace(m.recon_mut(), Image3F::new(0, 0)),
            None => Image3F::new(group_xsize, group_ysize),
        };
        (plane, Vec::new())
    });

    for ty in 0..group_ysize_tiles {
        let stripe_x0 = group_x0;
        let stripe_y0 = group_y0 + ty * K_TILE_DIM;
        let stripe_xsize = group_xsize;
        let stripe_ysize = K_TILE_DIM.min(dim.ysize.saturating_sub(stripe_y0));
        let stripe_brect = Rect::new(
            qorigin_x,
            qorigin_y + ty * K_TILE_DIM_IN_BLOCKS,
            stripe_xsize.div_ceil(K_BLOCK_DIM),
            stripe_ysize.div_ceil(K_BLOCK_DIM),
        );

        // `write_ac_group` reads whole blocks straight from `opsin` and gathers
        // only footprints that cross the image edge.
        write_ac_group(
            ctx,
            &mut scratch.ac_group,
            opsin,
            (stripe_x0, stripe_y0),
            stripe_brect,
            distp.scale,
            distp.scale_dc,
            distp.dc_step,
            distp.distance,
            distp.x_qm_scale,
            dc_data,
            ytob_dc,
            &mut local_quant_dc,
            source_dc.as_mut(),
            qorigin_x,
            qorigin_y,
            &mut num_nzeros,
            coeff_shifts,
            rdoq_prices,
            coeff_orders,
            alternative_orders,
            order_stats.as_mut(),
            false,
            qf_threshold,
            recon
                .as_mut()
                .map(|(plane, starts)| (plane, ty * K_TILE_DIM, capture_loop.then_some(starts))),
            memo.as_deref_mut(),
            &mut tokens,
        );
    }
    if let (Some(m), Some((plane, _))) = (memo, recon.as_mut()) {
        *m.recon_mut() = std::mem::replace(plane, crate::image::Image3F::new(0, 0));
    }

    (
        PendingAcGroup {
            group_idx: image_gy * dim.xsize_groups + image_gx,
            tokens: tokens.into_iter().map(AcTokens::from).collect(),
        },
        local_quant_dc,
        source_dc,
        order_stats,
        recon,
    )
}
