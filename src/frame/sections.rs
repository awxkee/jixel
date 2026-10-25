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

//! Section assembly, table of contents and progressive group ordering.

use super::{ImageDim, K_DC_GROUP_DIM, K_GROUP_DIM};
use crate::bit_writer::BitWriter;

/// Section sizes in codestream order, after the optional permutation
/// (`permutation[i]` = codestream slot of canonical section `i`).
fn write_toc(
    sizes: &[usize],
    permutation: Option<(&[u32], &mut Vec<crate::entropy::HuffmanNode>)>,
    w: &mut BitWriter,
) {
    match permutation {
        Some((permutation, huffman_pool)) => {
            w.write(1, 1);
            crate::coeff_order::write_permutation(permutation, huffman_pool, w);
        }
        None => w.write(1, 0),
    }
    w.zero_pad_to_byte();
    let k_bits = [10usize, 14, 22, 30];
    for &s in sizes {
        let mut offset: usize = 0;
        let mut ok = false;
        for (i, &b) in k_bits.iter().enumerate() {
            if s < offset + (1usize << b) {
                w.write(2, i as u64);
                w.write(b, (s - offset) as u64);
                ok = true;
                break;
            }
            offset += 1usize << b;
        }
        assert!(ok, "section size {} too large for TOC", s);
    }
    w.zero_pad_to_byte();
}

pub(crate) fn combine_sections(sections: &mut Vec<BitWriter>, writer: &mut BitWriter) {
    combine_sections_ordered(sections, None, writer);
}

/// Serialized section and TOC bytes, excluding the shared frame header.
/// Single-group frames concatenate all four sections before byte padding.
pub(crate) fn combined_sections_size(section_bits: &[usize]) -> usize {
    let sizes: Vec<usize> = if section_bits.len() == 4 {
        vec![section_bits.iter().sum::<usize>().div_ceil(8)]
    } else {
        section_bits.iter().map(|bits| bits.div_ceil(8)).collect()
    };
    let mut toc = BitWriter::new();
    write_toc(&sizes, None, &mut toc);
    toc.bits_written() / 8 + sizes.iter().sum::<usize>()
}

/// [`combine_sections`] storing the sections in `order` (canonical section
/// indices in codestream order), signaled through the TOC permutation.
pub(super) fn combine_sections_ordered(
    sections: &mut Vec<BitWriter>,
    order: Option<(&[usize], &mut Vec<crate::entropy::HuffmanNode>)>,
    writer: &mut BitWriter,
) {
    if sections.len() == 4 {
        // Single AC group case: concat sections 1..4 (bitwise) into section 0.
        let tail: Vec<BitWriter> = sections.drain(1..).collect();
        for s in &tail {
            sections[0].append(s);
        }
    }
    let order = order.filter(|(order, _)| {
        sections.len() > 1 && order.iter().enumerate().any(|(slot, &i)| slot != i)
    });
    let Some((order, huffman_pool)) = order else {
        let sizes: Vec<usize> = sections
            .iter()
            .map(|s| s.bits_written().div_ceil(8))
            .collect();
        write_toc(&sizes, None, writer);
        // After write_toc, writer is byte-aligned.
        writer.append_byte_aligned(sections);
        return;
    };
    debug_assert_eq!(order.len(), sections.len());
    let mut permutation = vec![0u32; order.len()];
    for (slot, &i) in order.iter().enumerate() {
        permutation[i] = slot as u32;
    }
    let mut taken: Vec<Option<BitWriter>> = sections.drain(..).map(Some).collect();
    sections.extend(order.iter().map(|&i| taken[i].take().expect("permutation")));
    let sizes: Vec<usize> = sections
        .iter()
        .map(|s| s.bits_written().div_ceil(8))
        .collect();
    write_toc(&sizes, Some((&permutation, huffman_pool)), writer);
    writer.append_byte_aligned(sections);
}

/// Canonical section indices of a VarDCT frame in `group_order`: the
/// LfGlobal and HfGlobal sections stay put, the DC groups and each pass's AC
/// groups are sorted by [`group_order_key`].
pub(super) fn group_section_order(
    dim: &ImageDim,
    num_passes: usize,
    group_order: crate::GroupOrder,
) -> Vec<usize> {
    let sorted = |xgroups: usize, ygroups: usize, group_dim: usize| -> Vec<usize> {
        let mut keys: Vec<(i64, f64, f64, usize)> = (0..xgroups * ygroups)
            .map(|g| {
                let (x0, y0) = ((g % xgroups) * group_dim, (g / xgroups) * group_dim);
                let w = group_dim.min(dim.xsize - x0);
                let h = group_dim.min(dim.ysize - y0);
                let center = (x0 as f64 + w as f64 / 2.0, y0 as f64 + h as f64 / 2.0);
                let (band, radius, angle) = group_order_key(dim, group_dim, center, group_order);
                (band, radius, angle, g)
            })
            .collect();
        keys.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.total_cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        keys.into_iter().map(|(.., g)| g).collect()
    };
    let dc = sorted(dim.xsize_dc_groups, dim.ysize_dc_groups, K_DC_GROUP_DIM);
    let ac = sorted(dim.xsize_groups, dim.ysize_groups, K_GROUP_DIM);
    let mut order = vec![0];
    order.extend(dc.iter().map(|&g| 1 + g));
    order.push(1 + dim.num_dc_groups);
    for pass in 0..num_passes {
        let base = 2 + dim.num_dc_groups + pass * dim.num_groups;
        order.extend(ac.iter().map(|&g| base + g));
    }
    order
}

/// Sort key of a group centered at `center`: (band, distance from the image
/// center, clockwise angle from the top). Center-first bands are rings one
/// group wide, each swept by angle alone; diagonal bands are half a group
/// wide around the nearer corner-to-corner diagonal and grow outward, so the
/// image draws an X that then thickens.
fn group_order_key(
    dim: &ImageDim,
    group_dim: usize,
    center: (f64, f64),
    group_order: crate::GroupOrder,
) -> (i64, f64, f64) {
    let (w, h) = (dim.xsize as f64, dim.ysize as f64);
    let (dx, dy) = (center.0 - w / 2.0, center.1 - h / 2.0);
    let radius = dx.hypot(dy);
    let angle = dx.atan2(-dy).rem_euclid(std::f64::consts::TAU);
    let g = group_dim as f64;
    match group_order {
        crate::GroupOrder::Diagonals => {
            let diagonal = w.hypot(h);
            let main = (w * center.1 - h * center.0).abs() / diagonal;
            let anti = (w * (h - center.1) - h * center.0).abs() / diagonal;
            ((main.min(anti) / (0.5 * g)).floor() as i64, radius, angle)
        }
        _ => ((radius / g).floor() as i64, 0.0, angle),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn section_size_matches_serialization_at_padding_and_toc_boundaries() {
        for sizes in [
            vec![1, 2, 3, 4],
            vec![8191, 1, 1, 1],
            vec![8192, 1, 1, 1],
            vec![1, 2, 3, 4, 5],
            vec![8184, 8185, 8193, 139256, 139257],
        ] {
            let sections: Vec<_> = sizes
                .iter()
                .map(|&bits| {
                    let mut section = crate::bit_writer::BitWriter::new();
                    for _ in 0..bits {
                        section.write(1, 1);
                    }
                    section
                })
                .collect();
            let expected = super::combined_sections_size(&sizes);
            for prefix_bits in 0..8 {
                let mut writer = crate::bit_writer::BitWriter::new();
                writer.write(prefix_bits, 0);
                super::combine_sections(&mut sections.clone(), &mut writer);
                // The first TOC bit shares the last frame-header byte.
                assert_eq!(writer.into_bytes().len(), expected);
            }
        }
    }

    #[test]
    fn center_first_order_is_rings_around_the_center() {
        // 2048x1365: 8x6 AC groups in one DC group, two passes.
        let dim = super::ImageDim::new(2048, 1365);
        let order = super::group_section_order(&dim, 2, crate::GroupOrder::CenterFirst);
        let n = dim.num_groups;
        assert_eq!(order.len(), 3 + 2 * n);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &v)| i == v));
        assert_eq!(&order[..3], &[0, 1, 2]);
        // The four groups around the center lead each pass, starting top right.
        let first: Vec<usize> = order[3..7].iter().map(|&s| s - 3).collect();
        assert_eq!(first, [20, 28, 27, 19]);
        let pass1: Vec<usize> = order[3 + n..].iter().map(|&s| s - 3 - n).collect();
        let pass0: Vec<usize> = order[3..3 + n].iter().map(|&s| s - 3).collect();
        assert_eq!(pass0, pass1);
    }

    #[test]
    fn diagonal_order_draws_an_x_first() {
        // 2048x1365: 8x6 AC groups. The first twelve form the X: the center
        // four, then both diagonals outward.
        let dim = super::ImageDim::new(2048, 1365);
        let order = super::group_section_order(&dim, 1, crate::GroupOrder::Diagonals);
        let groups: Vec<usize> = order[3..].iter().map(|&s| s - 3).collect();
        let mut x: Vec<usize> = groups[..12].to_vec();
        x.sort_unstable();
        assert_eq!(x, [9, 10, 13, 14, 19, 20, 26, 27, 28, 29, 33, 38]);
        let mut sorted = groups.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &v)| i == v));
    }
}
