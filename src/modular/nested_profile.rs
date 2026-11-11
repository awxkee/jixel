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

//! A scalar profile of an additive coordinate, using two ordinary decoder
//! transforms: `A(x,y) = P[U(x) + V(y)]`.
//!
//! The inner predictive Palette reconstructs the coordinate through Top
//! deltas; Top falls back to Left on the first image row. The outer Palette
//! maps that coordinate to the output sample. Its inverse runs after AC-group
//! stitching, so prediction does not reset at a group boundary.

use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{Token, pack_signed, write_entropy_code};
use crate::lossless::kit::{
    build_balanced_tree_tokens_mul, build_lz_pixel_code_ans, channel_to_context,
    lz77_compress_for_speed, selected_grad_pack_interior_fn, tokenize_plane_rows, write_lz_section,
    write_tree_lz77,
};
use crate::lossless::{GradientScratch, pack_gradient_row};
use std::ops::Range;

const GROUP_DIM: usize = super::GROUP_DIM;
const MAX_PIXELS: usize = crate::function_fields::MAX_PIXELS;
const MAX_COLORS: usize = 70911;
const MAX_DELTAS: usize = 66816;
const PREDICTOR_ZERO: u32 = 0;
const PREDICTOR_LEFT: u32 = 1;
const PREDICTOR_TOP: u32 = 2;
const PREDICTOR_GRADIENT: u32 = 5;

/// Encode a pre-fitted scalar model. All metadata and index streams are
/// included. The caller must still race the completed frame and check the
/// model's distortion against its incumbent reconstruction.
///
/// Invalid or excessive models are declined before coding.
/// No environment variables, sidecars, or external plans are consulted.
pub(crate) fn encode(
    pool: &crate::thread_pool::ThreadPool,
    profile: &[i32],
    u: &[i32],
    v: &[i32],
    scratch: &mut CoderScratch,
) -> Option<(BitWriter, Vec<BitWriter>)> {
    validate(profile, u, v)?;
    // The deduplicated table is smaller, but assigning regular row labels can
    // make its index plane substantially cheaper. Both are deterministic and
    // independent, so they are coded side by side.
    let (deduplicated, regular) = pool.join(
        || {
            let plan = make_plan(u, v, false)?;
            Some(encode_plan(profile, &plan, scratch))
        },
        || {
            let plan = make_plan(u, v, true)?;
            Some(encode_plan(profile, &plan, &mut CoderScratch::lossless()))
        },
    );
    let (mut best, candidate) = (deduplicated?, regular?);
    if section_bits(&candidate) < section_bits(&best) {
        best = candidate;
    }
    Some(best)
}

fn validate(profile: &[i32], u: &[i32], v: &[i32]) -> Option<()> {
    if profile.is_empty()
        || profile.len() > MAX_COLORS
        || u.is_empty()
        || v.is_empty()
        || u.len() > MAX_COLORS
        || v.len() - 1 > MAX_DELTAS
        || u.len().checked_mul(v.len())? > MAX_PIXELS
        || profile
            .iter()
            .any(|&sample| !(-65536..=65535).contains(&sample))
    {
        return None;
    }
    // Extrema prove the complete Cartesian product is within the profile.
    // i64 also permits oppositely shifted axes near the i32 boundaries.
    let lo = i64::from(*u.iter().min()?) + i64::from(*v.iter().min()?);
    let hi = i64::from(*u.iter().max()?) + i64::from(*v.iter().max()?);
    (lo >= 0 && hi < profile.len() as i64).then_some(())
}

struct Plan {
    table: Vec<i32>,
    first_row: Vec<i32>,
    later_rows: Vec<i32>,
    deltas: usize,
    absolutes: usize,
    regular: bool,
}

fn difference(&[from, to]: &[i32; 2]) -> i32 {
    // After validate(), either axis's span is less than MAX_COLORS.
    (i64::from(to) - i64::from(from)) as i32
}

fn make_plan(u: &[i32], v: &[i32], regular: bool) -> Option<Plan> {
    let (width, height) = (u.len(), v.len());
    if regular {
        let mut table = Vec::with_capacity(width + height - 1);
        table.extend(v.array_windows().map(difference));
        let deltas = height - 1;
        table.extend(
            u.iter()
                .map(|&value| (i64::from(value) + i64::from(v[0])) as i32),
        );
        return Some(Plan {
            table,
            first_row: (deltas as i32..(deltas + width) as i32).collect(),
            later_rows: (0..deltas as i32).collect(),
            deltas,
            absolutes: width,
            regular,
        });
    }

    let mut table = Vec::with_capacity(width + height - 1);
    table.extend(u.array_windows().chain(v.array_windows()).map(difference));
    table.sort_unstable();
    table.dedup();
    let deltas = table.len();
    if deltas > MAX_DELTAS {
        return None;
    }
    let index = |pair| table.binary_search(&difference(pair)).unwrap() as i32;
    let mut first_row = Vec::with_capacity(width);
    first_row.push(deltas as i32);
    first_row.extend(u.array_windows().map(index));
    let later_rows = v.array_windows().map(index).collect();
    table.push((i64::from(u[0]) + i64::from(v[0])) as i32);
    Some(Plan {
        table,
        first_row,
        later_rows,
        deltas,
        absolutes: 1,
        regular,
    })
}

impl Plan {
    fn tokenize_indices(
        &self,
        x: Range<usize>,
        y: Range<usize>,
        context: u32,
        scratch: &mut GradientScratch,
        tokens: &mut Vec<Token>,
    ) {
        let width = x.len();
        tokens.reserve(width * y.len());
        let mut row = y.start;
        let mut north = 0;
        if row == 0 {
            let first = &self.first_row[x];
            let gradient = selected_grad_pack_interior_fn();
            scratch.buf.resize(width, 0);
            pack_gradient_row(first, None, &mut scratch.buf, gradient);
            tokens.extend(scratch.buf.iter().map(|&value| Token::new(context, value)));
            row += 1;
            if row == y.end {
                return;
            }
            north = self.later_rows[0];
            scratch.cur.resize(width, north);
            scratch.cur.fill(north);
            pack_gradient_row(&scratch.cur, Some(first), &mut scratch.buf, gradient);
            tokens.extend(scratch.buf.iter().map(|&value| Token::new(context, value)));
            row += 1;
        }
        // The first sample predicts zero at a group start and north on
        // subsequent rows. Constant neighbors predict west in the interior.
        for &value in &self.later_rows[row - 1..y.end - 1] {
            tokens.push(Token::new(context, pack_signed(value - north)));
            tokens.resize(tokens.len() + width - 1, Token::new(context, 0));
            north = value;
        }
    }
}

fn section_bits(sections: &(BitWriter, Vec<BitWriter>)) -> usize {
    sections.0.bits_written()
        + sections
            .1
            .iter()
            .map(BitWriter::bits_written)
            .sum::<usize>()
}

fn encode_plan(
    profile: &[i32],
    plan: &Plan,
    scratch: &mut CoderScratch,
) -> (BitWriter, Vec<BitWriter>) {
    let (width, height) = (plan.first_row.len(), plan.later_rows.len() + 1);
    let small = width <= GROUP_DIM && height <= GROUP_DIM;
    let mut global = BitWriter::new();
    global.write(1, 0); // use_global_tree=false
    global.write(1, 1); // default weighted predictor header
    global.write(2, 2);
    global.write(4, 0); // two transforms
    write_palette(0, profile.len(), 0, PREDICTOR_ZERO, &mut global);
    write_palette(1, plan.absolutes, plan.deltas, PREDICTOR_TOP, &mut global);
    let contexts = if small { 3 } else { 2 };
    let mut tokens = Vec::with_capacity(plan.table.len() + profile.len());
    let gradient = selected_grad_pack_interior_fn();
    for (channel, table) in [plan.table.as_slice(), profile].into_iter().enumerate() {
        tokenize_plane_rows(
            channel_to_context(channel, contexts),
            |_| table,
            table.len(),
            1,
            PREDICTOR_GRADIENT,
            gradient,
            &mut scratch.gradient,
            &mut tokens,
        );
    }
    if small {
        plan.tokenize_indices(0..width, 0..height, 0, &mut scratch.gradient, &mut tokens);
        write_tokens(&tokens, contexts, scratch, &mut global);
        return (global, vec![BitWriter::new()]);
    }
    write_tokens(&tokens, contexts, scratch, &mut global);
    let mut ac: Vec<BitWriter> =
        Vec::with_capacity(width.div_ceil(GROUP_DIM) * height.div_ceil(GROUP_DIM));
    for y0 in (0..height).step_by(GROUP_DIM) {
        let first_group = ac.len();
        for x0 in (0..width).step_by(GROUP_DIM) {
            let group_width = GROUP_DIM.min(width - x0);
            // Below the first row of groups every index row is constant, so
            // full-width groups in this band have identical local streams.
            if y0 > 0 && x0 > 0 && group_width == GROUP_DIM {
                ac.push(ac[first_group].clone());
                continue;
            }
            let group_height = GROUP_DIM.min(height - y0);
            tokens.clear();
            plan.tokenize_indices(
                x0..x0 + group_width,
                y0..y0 + group_height,
                0,
                &mut scratch.gradient,
                &mut tokens,
            );
            let mut section = BitWriter::new();
            section.write(1, 0); // local tree
            section.write(1, 1); // default weighted predictor header
            section.write(2, 0); // no local transforms
            let procedural = plan.regular.then(|| section.clone());
            write_tokens(&tokens, 1, scratch, &mut section);
            if let Some(mut procedural) = procedural {
                write_regular_ma(x0, y0, plan.deltas, scratch, &mut procedural);
                if procedural.bits_written() < section.bits_written() {
                    section = procedural;
                }
            }
            ac.push(section);
        }
    }
    (global, ac)
}

fn write_palette(begin: usize, colors: usize, deltas: usize, predictor: u32, out: &mut BitWriter) {
    out.write(2, 1); // Palette transform
    out.write(2, 0);
    out.write(3, begin as u64);
    out.write(2, 0); // one channel
    if colors <= 255 {
        out.write(2, 0);
        out.write(8, colors as u64);
    } else if colors <= 1279 {
        out.write(2, 1);
        out.write(10, (colors - 256) as u64);
    } else if colors <= 5375 {
        out.write(2, 2);
        out.write(12, (colors - 1280) as u64);
    } else {
        out.write(2, 3);
        out.write(16, (colors - 5376) as u64);
    }
    if deltas == 0 {
        out.write(2, 0);
    } else if deltas <= 256 {
        out.write(2, 1);
        out.write(8, (deltas - 1) as u64);
    } else if deltas <= 1280 {
        out.write(2, 2);
        out.write(10, (deltas - 257) as u64);
    } else {
        out.write(2, 3);
        out.write(16, (deltas - 1281) as u64);
    }
    out.write(4, u64::from(predictor));
}

fn write_tokens(
    tokens: &[Token],
    contexts: usize,
    scratch: &mut CoderScratch,
    out: &mut BitWriter,
) {
    let lz = lz77_compress_for_speed(tokens, contexts as u32, crate::Speed::Slow, scratch);
    let code = build_lz_pixel_code_ans(
        false,
        std::iter::once(lz.as_slice()),
        contexts,
        64,
        true,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_tree_lz77(
        &build_balanced_tree_tokens_mul(&[(PREDICTOR_GRADIENT, 1); 3][..contexts]),
        &code,
        64,
        &mut scratch.huffman_pool,
        out,
    );
    write_lz_section(&lz, contexts as u32, &code, 64, out);
}

#[derive(Clone, Copy)]
enum Node {
    Split { property: u32, threshold: i32 },
    Leaf { predictor: u32, offset: i32 },
}

/// Breadth-first nodes, with the greater child before the lesser child.
fn regular_tree(x0: usize, y0: usize, deltas: usize) -> Vec<Node> {
    let split = |property, threshold| Node::Split {
        property,
        threshold,
    };
    let leaf = |predictor, offset| Node::Leaf { predictor, offset };
    if y0 == 0 {
        vec![
            split(2, 1),
            split(3, 0),
            split(2, 0),
            leaf(PREDICTOR_LEFT, 0),
            leaf(PREDICTOR_TOP, 1),
            leaf(PREDICTOR_ZERO, 0),
            split(3, 0),
            leaf(PREDICTOR_LEFT, 1),
            leaf(PREDICTOR_ZERO, (deltas + x0) as i32),
        ]
    } else {
        vec![
            split(2, 0),
            split(3, 0),
            leaf(PREDICTOR_ZERO, (y0 - 1) as i32),
            leaf(PREDICTOR_LEFT, 0),
            leaf(PREDICTOR_TOP, 1),
        ]
    }
}

fn write_regular_ma(
    x0: usize,
    y0: usize,
    deltas: usize,
    scratch: &mut CoderScratch,
    out: &mut BitWriter,
) {
    let tree = regular_tree(x0, y0, deltas);
    let mut tree_tokens = Vec::with_capacity(33);
    let mut contexts = 0;
    for &node in &tree {
        match node {
            Node::Split {
                property,
                threshold,
            } => tree_tokens.extend([
                Token::new(super::TREE_CTX_PROPERTY, property + 1),
                Token::new(super::TREE_CTX_SPLITVAL, pack_signed(threshold)),
            ]),
            Node::Leaf { predictor, offset } => {
                contexts += 1;
                tree_tokens.extend([
                    Token::new(super::TREE_CTX_PROPERTY, 0),
                    Token::new(super::TREE_CTX_PREDICTOR, predictor),
                    Token::new(super::TREE_CTX_OFFSET, pack_signed(offset)),
                    Token::new(super::TREE_CTX_MULTIPLIER_LOG, 0),
                    Token::new(super::TREE_CTX_MULTIPLIER_BITS, 0),
                ]);
            }
        }
    }
    crate::entropy::write_raced_token_stream(
        &tree_tokens,
        super::NUM_TREE_CONTEXTS,
        &mut scratch.huffman_pool,
        out,
    );
    // The tree predicts every label exactly. Empty histograms and histograms
    // containing only symbol zero both write a one-symbol alphabet with no payload.
    let code = super::build_pixel_code_n(&[], contexts, &mut scratch.huffman_pool);
    out.write(1, 0);
    write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inverse(plan: &Plan) -> Vec<i32> {
        let width = plan.first_row.len();
        let height = plan.later_rows.len() + 1;
        let mut samples = vec![0; width * height];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let index = if y == 0 {
                    plan.first_row[x]
                } else {
                    plan.later_rows[y - 1]
                } as usize;
                assert!(index < plan.table.len());
                let prediction = if index < plan.deltas {
                    if y > 0 {
                        samples[i - width]
                    } else if x > 0 {
                        samples[i - 1]
                    } else {
                        0
                    }
                } else {
                    0
                };
                samples[i] = plan.table[index] + prediction;
            }
        }
        samples
    }

    fn selected_leaf(tree: &[Node], x: usize, y: usize) -> Node {
        let mut index = 0;
        loop {
            match tree[index] {
                Node::Split {
                    property,
                    threshold,
                } => {
                    let preceding_splits = tree[..index]
                        .iter()
                        .filter(|node| matches!(node, Node::Split { .. }))
                        .count();
                    let coordinate = if property == 3 { x } else { y };
                    index = 2 * preceding_splits + 1 + usize::from(coordinate as i32 <= threshold);
                }
                leaf @ Node::Leaf { .. } => return leaf,
            }
        }
    }

    #[test]
    fn nested_palette_reconstructs_shifted_signed_axes_across_group_boundaries() {
        let width = 513;
        let height = 259;
        let profile: Vec<i32> = (0..2048).map(|x| (x * x / 31) % 256).collect();
        let u: Vec<i32> = (0..width).map(|x| x as i32 + i32::MIN + 1024).collect();
        let v: Vec<i32> = (0..height).map(|y| i32::MAX - 1023 + y as i32).collect();
        assert!(validate(&profile, &u, &v).is_some());
        for regular in [false, true] {
            let plan = make_plan(&u, &v, regular).unwrap();
            let reconstructed = inverse(&plan);
            for y in 0..height {
                for x in 0..width {
                    let expected = (i64::from(u[x]) + i64::from(v[y])) as usize;
                    assert_eq!(reconstructed[y * width + x], expected as i32);
                    assert_eq!(
                        profile[reconstructed[y * width + x] as usize],
                        profile[expected]
                    );
                }
            }
        }
    }

    #[test]
    fn nested_index_tokens_match_group_local_gradient_prediction() {
        let mut scratch = GradientScratch::default();
        let gradient = selected_grad_pack_interior_fn();
        for (width, height) in [(1, 513), (513, 1), (13, 2), (33, 31), (513, 259)] {
            let u: Vec<i32> = (0..width).map(|x| (x * x % 37) as i32 - 18).collect();
            let v: Vec<i32> = (0..height).map(|y| (y * y % 43) as i32 + 30).collect();
            for regular in [false, true] {
                let plan = make_plan(&u, &v, regular).unwrap();
                for y0 in (0..height).step_by(GROUP_DIM) {
                    for x0 in (0..width).step_by(GROUP_DIM) {
                        let w = GROUP_DIM.min(width - x0);
                        let h = GROUP_DIM.min(height - y0);
                        let mut indices = Vec::with_capacity(w * h);
                        for y in y0..y0 + h {
                            if y == 0 {
                                indices.extend_from_slice(&plan.first_row[x0..x0 + w]);
                            } else {
                                indices.resize(indices.len() + w, plan.later_rows[y - 1]);
                            }
                        }
                        let mut expected = Vec::new();
                        tokenize_plane_rows(
                            2,
                            |y| &indices[y * w..][..w],
                            w,
                            h,
                            PREDICTOR_GRADIENT,
                            gradient,
                            &mut scratch,
                            &mut expected,
                        );
                        let mut actual = Vec::new();
                        plan.tokenize_indices(x0..x0 + w, y0..y0 + h, 2, &mut scratch, &mut actual);
                        assert_eq!(actual.len(), expected.len());
                        assert!(
                            actual
                                .iter()
                                .zip(expected)
                                .all(|(a, b)| { a.context == b.context && a.value == b.value }),
                            "group {x0},{y0} of {width}x{height}, regular={regular}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn procedural_labels_reconstruct_with_group_local_ma_coordinates() {
        for (width, height) in [(513, 259), (1, 513), (513, 1), (257, 257)] {
            let deltas = height - 1;
            for y0 in (0..height).step_by(GROUP_DIM) {
                for x0 in (0..width).step_by(GROUP_DIM) {
                    let gw = GROUP_DIM.min(width - x0);
                    let gh = GROUP_DIM.min(height - y0);
                    let tree = regular_tree(x0, y0, deltas);
                    let mut decoded = vec![0; gw * gh];
                    for y in 0..gh {
                        for x in 0..gw {
                            let Node::Leaf { predictor, offset } = selected_leaf(&tree, x, y)
                            else {
                                panic!("tree did not terminate at a leaf")
                            };
                            let i = y * gw + x;
                            let prediction = match predictor {
                                PREDICTOR_ZERO => 0,
                                PREDICTOR_LEFT if x > 0 => decoded[i - 1],
                                PREDICTOR_TOP if y > 0 => decoded[i - gw],
                                _ => panic!("unexpected boundary predictor"),
                            };
                            decoded[i] = prediction + offset;
                            let expected = if y0 + y == 0 {
                                (deltas + x0 + x) as i32
                            } else {
                                (y0 + y - 1) as i32
                            };
                            assert_eq!(decoded[i], expected, "group {x0},{y0}, sample {x},{y}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_and_excessive_models_are_declined() {
        for (profile, u, v) in [
            (&[][..], &[0][..], &[0][..]),
            (&[0][..], &[][..], &[0][..]),
            (&[0][..], &[0][..], &[][..]),
            (&[0][..], &[1][..], &[0][..]),
            (&[0][..], &[-1][..], &[0][..]),
            (&[0][..], &[i32::MAX][..], &[i32::MAX][..]),
            (&[-65537][..], &[0][..], &[0][..]),
            (&[65536][..], &[0][..], &[0][..]),
        ] {
            assert!(
                encode(
                    &crate::thread_pool::ThreadPool::new(1),
                    profile,
                    u,
                    v,
                    &mut CoderScratch::default()
                )
                .is_none()
            );
        }
        assert!(validate(&vec![0; MAX_COLORS + 1], &[0], &[0]).is_none());
        assert!(validate(&[0], &vec![0; MAX_PIXELS / 1024 + 1], &vec![0; 1024]).is_none());
        assert!(validate(&[0], &[0], &vec![0; MAX_DELTAS + 2]).is_none());
    }
}
