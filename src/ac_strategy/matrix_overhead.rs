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

use super::*;
use crate::dc_group_data::DcGroupData;
use crate::quant_weights::{MatrixHeaderCost, quant_table_slot_of};

#[derive(Clone, Copy, Debug)]
struct Alternative {
    strategy: u8,
    required: u16,
    loss_bits: f64,
}

struct Region {
    dc: usize,
    bx: usize,
    by: usize,
    width: usize,
    height: usize,
    alternatives: Vec<Alternative>,
}

fn best_alternative(alternatives: &[Alternative], allowed: u16) -> Option<&Alternative> {
    // Strict comparison preserves the selected layout on a tie.
    let mut best: Option<&Alternative> = None;
    for alternative in alternatives {
        if alternative.required & !allowed == 0
            && best.is_none_or(|b| alternative.loss_bits < b.loss_bits)
        {
            best = Some(alternative);
        }
    }
    best
}

fn choose_tables(header: &MatrixHeaderCost, used: u16, pinned: u16, regions: &[Region]) -> u16 {
    let custom = header.custom_mask();
    let mut best = used;
    let mut best_cost = header.bits(used) as f64;
    // At most nine custom tables. Unused library slots have no activation fee.
    for allowed in 0..512u16 {
        if allowed & !custom != 0 || pinned & !allowed != 0 {
            continue;
        }
        let mut cost = header.bits(allowed) as f64;
        for region in regions {
            cost += best_alternative(&region.alternatives, allowed)
                .map_or(f64::INFINITY, |a| a.loss_bits);
            if cost >= best_cost {
                break;
            }
        }
        if cost < best_cost {
            best_cost = cost;
            best = allowed;
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn account_matrix_headers(
    ctx: &EncodingContext,
    scratch: &mut CoderScratch,
    opsin: &Image3F,
    distance: f32,
    scale: f32,
    x_qm_scale: u32,
    group_coords: &[(usize, usize)],
    dc_datas: &mut [DcGroupData],
) {
    let header = MatrixHeaderCost::new(ctx.matrices());
    let custom = header.custom_mask();
    if custom & !1 == 0 || ctx.speed == crate::Speed::Fastest {
        return;
    }
    let mut used = 0;
    let mut sites: [Vec<(usize, usize, usize, u8)>; 9] = std::array::from_fn(|_| Vec::new());
    for (dc, data) in dc_datas.iter().enumerate() {
        for (bx, by, strategy) in data.ac_strategy.iter_first_blocks() {
            if let Some(slot) = quant_table_slot_of(strategy)
                && custom & (1 << slot) != 0
            {
                used |= 1 << slot;
                if slot != 0 {
                    sites[slot].push((dc, bx, by, strategy));
                }
            }
        }
    }
    // DCT8 is the terminal fallback. Existing DCT8 blocks remain fixed.
    if used & !1 == 0 {
        return;
    }
    let qm_mult_x = 1.25f32.powf(x_qm_scale as f32 - 2.0);
    let dc_ref = &*dc_datas;
    let families = ctx.thread_pool.steal_map(scratch, 8, |index, scratch| {
        let slot = index + 1;
        let mut regions = Vec::new();
        let mut unavoidable_loss = 0.0;
        for &(dc, bx, by, strategy) in &sites[slot] {
            let data = &dc_ref[dc];
            let (gx, gy) = group_coords[dc];
            let width = AcStrategyImage::covered_blocks_x_of(strategy);
            let height = AcStrategyImage::covered_blocks_y_of(strategy);
            let score = |scratch: &mut CoderScratch, s, x, y| {
                let px = gx * crate::frame::K_DC_GROUP_DIM + x * 8;
                let py = gy * crate::frame::K_DC_GROUP_DIM + y * 8;
                let qac = scale * data.raw_quant_field.row(y)[x] as f32;
                let cmap = cmap_factors(&data.ytox_map, &data.ytob_map, x, y);
                let cost = if matches!(s, STRATEGY_DCT64X64 | STRATEGY_DCT64X32 | STRATEGY_DCT32X64)
                {
                    strategy_cost64(
                        ctx, scratch, s, opsin, px, py, qac, qm_mult_x, META_R, distance, cmap,
                    )
                } else {
                    strategy_cost(
                        ctx, scratch, s, opsin, px, py, qac, qm_mult_x, META_R, distance, cmap,
                    )
                };
                f64::from(cost) / f64::from(RD_LAMBDA)
            };
            let incumbent = score(scratch, strategy, bx, by);
            let mut alternatives = vec![Alternative {
                strategy,
                required: 1 << slot,
                loss_bits: 0.0,
            }];
            for candidate in [
                STRATEGY_DCT,
                STRATEGY_DCT16X8,
                STRATEGY_DCT8X16,
                STRATEGY_DCT16X16,
                STRATEGY_DCT32X16,
                STRATEGY_DCT16X32,
                STRATEGY_DCT32X32,
                STRATEGY_DCT64X32,
                STRATEGY_DCT32X64,
            ] {
                let cx = AcStrategyImage::covered_blocks_x_of(candidate);
                let cy = AcStrategyImage::covered_blocks_y_of(candidate);
                if cx > width
                    || cy > height
                    || (cx * cy >= width * height && candidate != STRATEGY_DCT)
                    || (!SearchScope::for_speed(ctx.speed).rectangles() && cx != cy)
                {
                    continue;
                }
                let mut cost = 0.0;
                for y in (by..by + height).step_by(cy) {
                    for x in (bx..bx + width).step_by(cx) {
                        cost += score(scratch, candidate, x, y);
                    }
                }
                if cost.is_finite() && incumbent.is_finite() {
                    alternatives.push(Alternative {
                        strategy: candidate,
                        required: quant_table_slot_of(candidate).map_or(0, |s| 1 << s) & custom,
                        // Preserve earlier perceptual decisions unless a real
                        // header saving pays for changing them. Negative local
                        // model deltas cannot subsidize losses elsewhere.
                        loss_bits: (cost - incumbent).max(0.0),
                    });
                }
            }
            unavoidable_loss += alternatives[1..]
                .iter()
                .map(|a| a.loss_bits)
                .fold(f64::INFINITY, f64::min);
            // A lower bound already exceeds this family's entire possible
            // fee, including the all-default transition. No table combination
            // can profitably remove it, so stop scoring its alternatives.
            if unavoidable_loss > (header.payload[slot] + 17 * 3) as f64 {
                return (1 << slot, Vec::new());
            }
            regions.push(Region {
                dc,
                bx,
                by,
                width,
                height,
                alternatives,
            });
        }
        (0, regions)
    });
    let mut pinned = used & 1;
    let mut regions = Vec::new();
    for (fixed, candidates) in families {
        pinned |= fixed;
        regions.extend(candidates);
    }
    let allowed = choose_tables(&header, used, pinned, &regions);
    if allowed == used {
        return;
    }
    for region in regions {
        let chosen = best_alternative(&region.alternatives, allowed).unwrap();
        let cx = AcStrategyImage::covered_blocks_x_of(chosen.strategy);
        let cy = AcStrategyImage::covered_blocks_y_of(chosen.strategy);
        for y in (region.by..region.by + region.height).step_by(cy) {
            for x in (region.bx..region.bx + region.width).step_by(cx) {
                dc_datas[region.dc]
                    .ac_strategy
                    .set_first(x, y, chosen.strategy);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(slot: usize, loss: f64, fallback: u16) -> Region {
        Region {
            dc: 0,
            bx: 0,
            by: 0,
            width: 1,
            height: 1,
            alternatives: vec![
                Alternative {
                    strategy: STRATEGY_DCT16X16,
                    required: 1 << slot,
                    loss_bits: 0.0,
                },
                Alternative {
                    strategy: STRATEGY_DCT,
                    required: fallback,
                    loss_bits: loss,
                },
            ],
        }
    }

    #[test]
    fn matrix_fee_is_shared_and_scan_order_independent() {
        let header = MatrixHeaderCost {
            payload: [0, 388, 0, 0, 0, 0, 0, 0, 0],
        };
        assert_eq!(choose_tables(&header, 2, 0, &[region(1, 200.0, 0)]), 0);
        let mut regions = vec![region(1, 240.0, 0), region(1, 210.0, 0)];
        assert_eq!(choose_tables(&header, 2, 0, &regions), 2);
        regions.reverse();
        assert_eq!(choose_tables(&header, 2, 0, &regions), 2);
    }

    #[test]
    fn table_combinations_include_fallback_activation_and_shared_mode_bits() {
        let header = MatrixHeaderCost {
            payload: [292, 388, 0, 0, 0, 0, 0, 0, 0],
        };
        assert_eq!(choose_tables(&header, 2, 0, &[region(1, 95.0, 1)]), 1);
        assert_eq!(choose_tables(&header, 2, 0, &[region(1, 97.0, 1)]), 2);
        let header = MatrixHeaderCost {
            payload: [0, 388, 388, 0, 0, 0, 0, 0, 0],
        };
        // Neither table pays the shared 51 bits on its own. Removing both
        // saves it once and wins; a greedy one-family pass misses this.
        let regions = [region(1, 400.0, 0), region(2, 400.0, 0)];
        assert_eq!(choose_tables(&header, 6, 0, &regions), 0);
    }

    #[test]
    fn matrix_optimizer_matches_exhaustive_assignments() {
        let header = MatrixHeaderCost {
            payload: [100, 120, 170, 210, 0, 0, 0, 0, 0],
        };
        let mut rng = 13u32;
        for _ in 0..200 {
            let mut regions = Vec::new();
            for slot in 1..4 {
                let mut r = region(slot, 0.0, 1);
                for alternative in &mut r.alternatives[1..] {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    alternative.loss_bits = (rng % 300) as f64;
                }
                r.alternatives.push(Alternative {
                    strategy: STRATEGY_DCT,
                    required: 1 << (slot % 3 + 1),
                    loss_bits: (rng % 150) as f64,
                });
                regions.push(r);
            }
            let allowed = choose_tables(&header, 14, 0, &regions);
            let actual = header.bits(allowed) as f64
                + regions
                    .iter()
                    .map(|r| {
                        best_alternative(&r.alternatives, allowed)
                            .unwrap()
                            .loss_bits
                    })
                    .sum::<f64>();
            let mut optimum = f64::INFINITY;
            for a in &regions[0].alternatives {
                for b in &regions[1].alternatives {
                    for c in &regions[2].alternatives {
                        optimum = optimum.min(
                            header.bits(a.required | b.required | c.required) as f64
                                + a.loss_bits
                                + b.loss_bits
                                + c.loss_bits,
                        );
                    }
                }
            }
            assert_eq!(actual, optimum);
        }
    }

    #[test]
    fn matrix_pruning_covers_multiple_dc_groups_and_preserves_quantizers() {
        let distance = 3.0;
        // Include a second DC group at its real pixel origin. Constant planes
        // make the redundant transform families cost metadata but no quality.
        let opsin = Image3F::new(crate::frame::K_DC_GROUP_DIM + 64, 64);
        let run = |threads| {
            let ctx = EncodingContext::new(
                crate::Speed::Slow,
                crate::xyb::XybMatrix::SPEC,
                distance,
                threads,
            );
            let mut scratch = CoderScratch::default();
            let mut groups: Vec<_> = (0..2).map(|_| DcGroupData::new(8, 8).unwrap()).collect();
            for group in &mut groups {
                for y in 0..8 {
                    group.raw_quant_field.row_mut(y).fill(17);
                }
                group.ac_strategy.set_first(0, 0, STRATEGY_DCT32X16);
                group.ac_strategy.set_first(4, 0, STRATEGY_DCT16X32);
            }
            account_matrix_headers(
                &ctx,
                &mut scratch,
                &opsin,
                distance,
                1.0,
                2,
                &[(0, 0), (1, 0)],
                &mut groups,
            );
            let mut maps = Vec::new();
            for group in groups {
                let mut covered = [0; 64];
                for (x, y, s) in group.ac_strategy.iter_first_blocks() {
                    assert!(group.ac_strategy.can_place_strategy(x, y, s));
                    assert_ne!(quant_table_slot_of(s), Some(4));
                    for dy in 0..AcStrategyImage::covered_blocks_y_of(s) {
                        for dx in 0..AcStrategyImage::covered_blocks_x_of(s) {
                            covered[(y + dy) * 8 + x + dx] += 1;
                        }
                    }
                    maps.push((x, y, s));
                }
                assert_eq!(covered, [1; 64]);
                for y in 0..8 {
                    assert_eq!(group.raw_quant_field.row(y), &[17; 8]);
                }
            }
            maps
        };
        assert_eq!(run(1), run(3));
    }
}
