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
/// source DC and order statistics when collected, and the luma reconstruction
/// capture (plane + per-block token starts) when requested.
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
    capture_recon: bool,
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

    let mut local_quant_dc = Image3S::new(gwb, ghb);
    let mut source_dc = (collect_order_stats || capture_recon).then(|| crate::group::SourceDc {
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
                .map(|(plane, starts)| (plane, ty * K_TILE_DIM, starts)),
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

/// Carve a stripe out of the (already-XYB-converted, gaborized) opsin image,
/// padding to whole blocks by edge-replication.
#[cfg(test)]
fn build_stripe(
    stripe: &mut Image3F,
    opsin: &Image3F,
    x0: usize,
    y0: usize,
    xsize: usize,
    ysize: usize,
    xsize_padded: usize,
    ysize_padded: usize,
) {
    stripe.resize(xsize_padded, ysize_padded);
    for c in 0..3 {
        // Copy actual content.
        for y in 0..ysize {
            let src_row = opsin.plane_row(c, y0 + y);
            let dst_row = stripe.plane_row_mut(c, y);
            let (data, padding) = dst_row.split_at_mut(xsize);
            data.copy_from_slice(&src_row[x0..x0 + xsize]);
            let last = *data.last().unwrap();
            padding[..xsize_padded - xsize].fill(last);
        }
        // Replicate bottom row.
        for y in ysize..ysize_padded {
            let (src, dst) = stripe.plane_mut(c).two_rows_mut_safe(ysize - 1, y);
            dst.copy_from_slice(src);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::quantization::compute_distance_params;

    #[test]
    fn ac_group_source_views_match_copied_stripes() {
        use crate::coeff_order::CoeffOrders;
        use crate::dc_group_data::{DcGroupData, NUM_STRATEGIES};
        use crate::encoding_context::EncodingContext;
        use crate::entropy::{FrozenTokenPrices, Token, build_entropy_code_no_cluster};
        use crate::group::{AcGroupScratch, SourceDc};
        use crate::image::{Image3B, Image3F, Image3S, Plane, Rect};

        let distance = 2.0;
        let ctx = EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            distance,
            1,
        );
        let distp = compute_distance_params(distance);
        let mut image = Image3F::new(131, 137);
        for c in 0..3 {
            for (i, value) in image.plane_mut(c).as_mut_slice().iter_mut().enumerate() {
                *value = ((i * 7919 + c * 97) % 104729) as f32 / 104729.0;
            }
        }
        let tokens: Vec<_> = (0..4096)
            .map(|i| Token::new(i % 2, if i % 3 == 0 { 0 } else { i % 17 }))
            .collect();
        let mut code = build_entropy_code_no_cluster(&tokens, 2, &mut Vec::new());
        code.context_map = (0..crate::ac_context::K_NUM_FINE_AC_CONTEXTS)
            .map(|i| (i % 2) as u8)
            .collect();
        let prices = FrozenTokenPrices::new(&code);
        let orders = CoeffOrders::natural();
        let mut dc = DcGroupData::new(16, 16).unwrap();
        for (i, q) in dc.raw_quant_field.as_mut_slice().iter_mut().enumerate() {
            *q = 3 + (i % 17) as u8;
        }
        dc.ytox_map.as_mut_slice().fill(7);
        dc.ytob_map.as_mut_slice().fill(-13);
        let mut stripe = Image3F::new(64, 64);
        let mut gathered_scratch = AcGroupScratch::default();
        let mut view_scratch = AcGroupScratch::default();
        // The 128 family does not fit this 16x16-block image.
        for strategy in
            (0..NUM_STRATEGIES as u8).filter(|&s| !crate::dc_group_data::is_128_strategy(s))
        {
            dc.ac_strategy.reset();
            dc.ac_strategy.set_first(8, 8, strategy);
            // (70, 76) crosses the image edge: the view gathers the same
            // replicated pixels a padded stripe holds.
            for origin in [(0, 0), (59, 61), (70, 76)] {
                let visible_x = (image.xsize() - origin.0).min(64);
                let visible_y = (image.ysize() - origin.1).min(64);
                let at_edge = visible_x < 64 || visible_y < 64;
                super::build_stripe(
                    &mut stripe,
                    &image,
                    origin.0,
                    origin.1,
                    visible_x,
                    visible_y,
                    64,
                    64,
                );
                for (shifts, rdoq) in [
                    (&[0u32][..], None),
                    (&[0u32][..], Some(&prices)),
                    (&[2u32, 0][..], None),
                ] {
                    // RDOQ discounts padding error only where it can see the
                    // image edge, so the padded stripe legitimately differs.
                    if at_edge && rdoq.is_some() {
                        continue;
                    }
                    let run = |input: &Image3F, origin, scratch: &mut AcGroupScratch| {
                        let mut quant_dc = Image3S::new(8, 8);
                        let mut source_dc = SourceDc {
                            y: Plane::new(8, 8),
                            b: Plane::new(8, 8),
                            x: Plane::new(8, 8),
                        };
                        let mut nzeros: Vec<_> =
                            shifts.iter().map(|_| Image3B::new(32, 32)).collect();
                        let mut tokens: Vec<_> = shifts.iter().map(|_| Vec::new()).collect();
                        let distortion = super::write_ac_group(
                            &ctx,
                            scratch,
                            input,
                            origin,
                            Rect::new(8, 8, 8, 8),
                            distp.scale,
                            distp.scale_dc,
                            distp.dc_step,
                            distance,
                            distp.x_qm_scale,
                            &dc,
                            11,
                            &mut quant_dc,
                            Some(&mut source_dc),
                            8,
                            8,
                            &mut nzeros,
                            shifts,
                            rdoq,
                            &orders,
                            &[],
                            None,
                            true,
                            10,
                            None,
                            None,
                            &mut tokens,
                        );
                        (distortion, quant_dc, source_dc, nzeros, tokens)
                    };
                    let (expected_dist, expected_dc, expected_source, expected_nz, expected) =
                        run(&stripe, (0, 0), &mut gathered_scratch);
                    let (actual_dist, actual_dc, actual_source, actual_nz, actual) =
                        run(&image, origin, &mut view_scratch);
                    assert_eq!(actual_dist.to_bits(), expected_dist.to_bits());
                    for c in 0..3 {
                        assert_eq!(actual_dc.plane_data(c), expected_dc.plane_data(c));
                        for (a, e) in actual_nz.iter().zip(&expected_nz) {
                            assert_eq!(a.plane_data(c), e.plane_data(c));
                        }
                    }
                    for (a, e) in [
                        (&actual_source.y, &expected_source.y),
                        (&actual_source.b, &expected_source.b),
                    ] {
                        assert!(
                            a.as_slice()
                                .iter()
                                .zip(e.as_slice())
                                .all(|(a, e)| a.to_bits() == e.to_bits())
                        );
                    }
                    for (a, e) in actual.iter().zip(&expected) {
                        assert_eq!(a.len(), e.len());
                        assert!(
                            a.iter()
                                .zip(e)
                                .all(|(a, e)| a.context == e.context && a.value == e.value)
                        );
                    }
                }
            }
        }
    }
}
