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

//! The scalar profile of [`super::nested_profile`], `A(x,y) = P[U(x) + V(y)]`,
//! with one ordinary Palette instead of two.
//!
//! The nested form sends the coordinate through a second Palette whose table
//! carries explicit deltas. jxl-oxide 0.12 sizes that table from the colour
//! count alone and cannot read it. Here the coordinate plane is coded
//! directly, group by group, under a context tree that predicts it exactly
//! away from the first row and column of each group:
//!
//! * the corner is a leaf offset,
//! * the first row is coded as West differences, the first column as North
//!   differences,
//! * everywhere else one neighbor difference names the other: with the
//!   West predictor the property `NW - N` is `-(U(x) - U(x-1))`, so a leaf
//!   per distinct step with that step as its offset leaves a zero residual
//!   (or the same along the other axis with North and `W - NW`).
//!
//! The price against the nested form is the first row and column of every
//! group, which no neighboring group can predict.
//!
//! [`encode_frame`] writes the function as a modular frame of its own, which
//! may use 1024-pixel groups and has none of a VarDCT frame's overhead. That
//! frame takes either coding: the direct one above, or the nested form's
//! second Palette, under which the groups hold nothing but a tree.

use super::nested_profile::{validate, write_palette, write_tokens};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{Token, pack_signed};
use crate::lossless::kit::{
    build_lz_pixel_code_ans, channel_to_context, lz77_compress_for_speed,
    selected_grad_pack_interior_fn, tokenize_plane_rows, write_lz_section, write_toc_entry,
    write_tree_lz77,
};
use crate::thread_pool::ThreadPool;
use std::collections::VecDeque;

const GROUP_DIM: usize = super::GROUP_DIM;
const PREDICTOR_ZERO: u32 = 0;
const PREDICTOR_LEFT: u32 = 1;
const PREDICTOR_TOP: u32 = 2;
const PREDICTOR_GRADIENT: u32 = 5;
const PROPERTY_CHANNEL: u32 = 0;
const PROPERTY_Y: u32 = 2;
const PROPERTY_X: u32 = 3;
/// `W - NW`: the step of `V` into this row.
const PROPERTY_WEST_STEP: u32 = 10;
/// `NW - N`: minus the step of `U` into this column.
const PROPERTY_NORTH_STEP: u32 = 11;
/// More distinct steps than this in a group and the exact tree is not tried.
const MAX_STEPS: usize = 512;

const TAG_PROFILE: usize = 0;
const TAG_CORNER: usize = 1;
const TAG_ROW: usize = 2;
const TAG_COLUMN: usize = 3;
const TAG_ZERO: usize = 4;
const TAG_TABLE: usize = 5;
const TAG_INTERIOR: usize = 6;

/// [`super::nested_profile::encode`] for decoders without explicit delta
/// tables. Same arguments, same sections.
pub(crate) fn encode(
    pool: &ThreadPool,
    profile: &[i32],
    u: &[i32],
    v: &[i32],
    scratch: &mut CoderScratch,
) -> Option<(BitWriter, Vec<BitWriter>)> {
    validate(profile, u, v)?;
    Some(encode_streams(
        pool, profile, u, v, GROUP_DIM, 0, false, scratch,
    ))
}

/// The function as a frame of its own: a modular frame with black color and
/// the function in extra channel 0, kept in reference slot 0 and not shown.
///
/// A VarDCT frame's extra channel is cut into 256-pixel groups, and each
/// group pays for its first row and column. A modular frame may use
/// 1024-pixel groups, so up to that size the axes are coded once, as the
/// nested form codes them.
///
/// For an `xyb_encoded` image with one extra channel.
pub(crate) fn encode_frame(
    pool: &ThreadPool,
    profile: &[i32],
    u: &[i32],
    v: &[i32],
    delta: bool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<()> {
    validate(profile, u, v)?;
    let (width, height) = (u.len(), v.len());
    let shift = if width <= GROUP_DIM && height <= GROUP_DIM {
        1
    } else {
        3
    };
    let group_dim = 128usize << shift;
    writer.zero_pad_to_byte();
    writer.write(1, 0); // all_default = false
    writer.write(2, 0); // regular frame
    writer.write(1, 1); // encoding = Modular
    writer.write(2, 0); // flags
    // No do_YCbCr bit: with xyb_encoded set the color transform is XYB.
    writer.write(2, 0); // upsampling = 1
    writer.write(2, 0); // ec_upsampling[0] = 1
    writer.write(2, shift);
    writer.write(2, 0); // one pass
    writer.write(1, 0); // no crop
    writer.write(2, 0); // blending = Replace
    writer.write(2, 0); // extra-channel blending = Replace
    writer.write(1, 0); // not the last frame
    writer.write(2, 0); // save_as_reference = 0
    writer.write(1, 0); // save_before_color_transform = false
    writer.write(2, 0); // no name
    writer.write(1, 0); // loop filter not all-default
    writer.write(1, 0); // no gaborish
    writer.write(2, 0); // no EPF
    writer.write(2, 0); // no loop-filter extensions
    writer.write(2, 0); // no frame-header extensions

    let (stream, groups) = encode_streams(pool, profile, u, v, group_dim, 3, delta, scratch);
    let mut global = BitWriter::new();
    global.write(1, 1); // LfChannelDequant all_default
    global.write(1, 0); // no global tree
    global.append_bits(&stream);
    let mut sections = vec![global];
    if width > group_dim || height > group_dim {
        // LfGroups and HfGlobal carry nothing in this frame.
        let lf_groups = width.div_ceil(8 * group_dim) * height.div_ceil(8 * group_dim);
        sections.resize_with(1 + lf_groups + 1, BitWriter::new);
        sections.extend(groups);
    }
    for section in &mut sections {
        section.zero_pad_to_byte();
    }
    writer.write(1, 0); // TOC: no permutation
    writer.zero_pad_to_byte();
    for section in &sections {
        write_toc_entry(section.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    Some(())
}

/// The modular stream of the global section, from its group header on, and
/// the streams of the groups. `colors` all-zero channels precede the
/// function's channel.
///
/// With `delta` the coordinate goes through a second Palette, as in
/// [`super::nested_profile`]: its table holds the steps of `V` and then the
/// first row, every later row is one delta label, and the tree predicts all
/// of the labels. Nothing is then coded per group but the tree.
///
/// A plane that fits one group is coded whole in the global stream; the one
/// group stream is then empty.
fn encode_streams(
    pool: &ThreadPool,
    profile: &[i32],
    u: &[i32],
    v: &[i32],
    group_dim: usize,
    colors: usize,
    delta: bool,
    scratch: &mut CoderScratch,
) -> (BitWriter, Vec<BitWriter>) {
    let (width, height) = (u.len(), v.len());
    let gradient = selected_grad_pack_interior_fn();
    let deltas = height - 1;
    let table: Vec<i32> = if delta {
        v.array_windows()
            .map(step)
            .chain(u.iter().map(|&column| sum(column, v[0])))
            .collect()
    } else {
        Vec::new()
    };
    let metas = 1 + usize::from(delta);
    let mut global = BitWriter::new();
    global.write(1, 0); // use_global_tree=false
    global.write(1, 1); // default weighted predictor header
    if delta {
        global.write(2, 2);
        global.write(4, 0); // two transforms
        write_palette(colors, profile.len(), 0, PREDICTOR_ZERO, &mut global);
        write_palette(colors + 1, width, deltas, PREDICTOR_TOP, &mut global);
    } else {
        global.write(2, 1); // one transform
        write_palette(colors, profile.len(), 0, PREDICTOR_ZERO, &mut global);
    }
    let leaf = |predictor, offset, tag| Shape::Leaf {
        predictor,
        offset,
        tag,
    };
    // Channels in coding order: the tables (global stream only), the zero
    // color channels, the coordinate plane.
    let write =
        |x0: usize, y0: usize, whole: bool, scratch: &mut CoderScratch, out: &mut BitWriter| {
            let u = &u[x0..(x0 + group_dim).min(width)];
            let v = &v[y0..(y0 + group_dim).min(height)];
            let steps = if delta { None } else { Steps::new(u, v) };
            let across = |inside, edge| Shape::split(PROPERTY_X, 0, inside, edge);
            let mut shape = match &steps {
                Some(steps) => steps.shape(u, v),
                None if delta && y0 == 0 => Shape::split(
                    PROPERTY_Y,
                    1,
                    across(
                        leaf(PREDICTOR_LEFT, 0, TAG_INTERIOR),
                        leaf(PREDICTOR_TOP, 1, TAG_COLUMN),
                    ),
                    Shape::split(
                        PROPERTY_Y,
                        0,
                        across(
                            leaf(PREDICTOR_LEFT, 0, TAG_INTERIOR + 1),
                            leaf(PREDICTOR_ZERO, 0, TAG_INTERIOR + 2),
                        ),
                        across(
                            leaf(PREDICTOR_LEFT, 1, TAG_ROW),
                            leaf(PREDICTOR_ZERO, (deltas + x0) as i32, TAG_CORNER),
                        ),
                    ),
                ),
                None if delta => Shape::split(
                    PROPERTY_Y,
                    0,
                    across(
                        leaf(PREDICTOR_LEFT, 0, TAG_INTERIOR),
                        leaf(PREDICTOR_TOP, 1, TAG_COLUMN),
                    ),
                    leaf(PREDICTOR_ZERO, (y0 - 1) as i32, TAG_CORNER),
                ),
                // Too many distinct steps for a tree to name.
                None => leaf(PREDICTOR_GRADIENT, 0, TAG_CORNER),
            };
            let first = if whole { metas } else { 0 };
            if colors > 0 {
                shape = Shape::split(
                    PROPERTY_CHANNEL,
                    (first + colors - 1) as i32,
                    shape,
                    leaf(PREDICTOR_ZERO, 0, TAG_ZERO),
                );
            }
            if whole {
                let mut tables = leaf(PREDICTOR_GRADIENT, 0, TAG_PROFILE);
                if delta {
                    tables = Shape::split(
                        PROPERTY_CHANNEL,
                        0,
                        tables,
                        leaf(PREDICTOR_GRADIENT, 0, TAG_TABLE),
                    );
                }
                // The tables sit below every other channel.
                shape = match shape {
                    Shape::Split {
                        property: PROPERTY_CHANNEL,
                        threshold,
                        greater,
                        lesser,
                    } => Shape::Split {
                        property: PROPERTY_CHANNEL,
                        threshold,
                        greater,
                        lesser: Box::new(Shape::split(
                            PROPERTY_CHANNEL,
                            (metas - 1) as i32,
                            *lesser,
                            tables,
                        )),
                    },
                    plane => Shape::split(PROPERTY_CHANNEL, (metas - 1) as i32, plane, tables),
                };
            }
            let tags = TAG_INTERIOR
                + match &steps {
                    Some(steps) => steps.values.len().max(1),
                    None => 3,
                };
            let (tree, contexts, leaves) = flatten(shape, tags);
            let mut tokens = Vec::with_capacity(
                (colors + 1) * u.len() * v.len()
                    + if whole {
                        profile.len() + table.len()
                    } else {
                        0
                    },
            );
            if whole {
                for (tag, table) in [(TAG_TABLE, table.as_slice()), (TAG_PROFILE, profile)] {
                    if !table.is_empty() {
                        tokenize_plane_rows(
                            contexts[tag],
                            |_| table,
                            table.len(),
                            1,
                            PREDICTOR_GRADIENT,
                            gradient,
                            &mut scratch.gradient,
                            &mut tokens,
                        );
                    }
                }
            }
            tokens.resize(
                tokens.len() + colors * u.len() * v.len(),
                Token::new(contexts[TAG_ZERO], 0),
            );
            match &steps {
                Some(steps) => steps.tokenize(u, v, &contexts, &mut tokens),
                None if delta => {
                    // Every label is predicted; only the contexts differ.
                    let mut row = |first, rest| {
                        tokens.push(Token::new(contexts[first], 0));
                        tokens.resize(tokens.len() + u.len() - 1, Token::new(contexts[rest], 0));
                    };
                    for y in 0..v.len() {
                        match (y0, y) {
                            (0, 0) => row(TAG_CORNER, TAG_ROW),
                            (0, 1) => row(TAG_INTERIOR + 2, TAG_INTERIOR + 1),
                            (_, 0) => row(TAG_CORNER, TAG_CORNER),
                            _ => row(TAG_COLUMN, TAG_INTERIOR),
                        }
                    }
                }
                None => {
                    let crop = coordinates(u, v);
                    tokenize_plane_rows(
                        contexts[TAG_CORNER],
                        |y| &crop[y * u.len()..][..u.len()],
                        u.len(),
                        v.len(),
                        PREDICTOR_GRADIENT,
                        gradient,
                        &mut scratch.gradient,
                        &mut tokens,
                    );
                }
            }
            // Groups already run side by side.
            write_shaped(whole.then_some(pool), &tree, &tokens, leaves, scratch, out);
        };

    if width <= group_dim && height <= group_dim {
        write(0, 0, true, scratch, &mut global);
        return (global, vec![BitWriter::new()]);
    }

    let mut tokens = Vec::with_capacity(profile.len() + table.len());
    for (channel, table) in [table.as_slice(), profile]
        .into_iter()
        .filter(|table| !table.is_empty())
        .enumerate()
    {
        tokenize_plane_rows(
            channel_to_context(channel, metas),
            |_| table,
            table.len(),
            1,
            PREDICTOR_GRADIENT,
            gradient,
            &mut scratch.gradient,
            &mut tokens,
        );
    }
    write_tokens(&tokens, metas, scratch, &mut global);
    let columns = width.div_ceil(group_dim);
    let ac = pool.map(columns * height.div_ceil(group_dim), |group| {
        let mut section = BitWriter::new();
        section.write(1, 0); // local tree
        section.write(1, 1); // default weighted predictor header
        section.write(2, 0); // no local transforms
        write(
            group % columns * group_dim,
            group / columns * group_dim,
            false,
            &mut CoderScratch::lossless(),
            &mut section,
        );
        section
    });
    (global, ac)
}

fn coordinates(u: &[i32], v: &[i32]) -> Vec<i32> {
    let mut plane = Vec::with_capacity(u.len() * v.len());
    for &row in v {
        plane.extend(u.iter().map(|&column| sum(column, row)));
    }
    plane
}

fn sum(u: i32, v: i32) -> i32 {
    // validate() bounds every sum by the profile length.
    (i64::from(u) + i64::from(v)) as i32
}

fn step(&[from, to]: &[i32; 2]) -> i32 {
    (i64::from(to) - i64::from(from)) as i32
}

enum Shape {
    /// `greater` is taken where the property exceeds the threshold.
    Split {
        property: u32,
        threshold: i32,
        greater: Box<Shape>,
        lesser: Box<Shape>,
    },
    Leaf {
        predictor: u32,
        offset: i32,
        tag: usize,
    },
}

impl Shape {
    fn split(property: u32, threshold: i32, greater: Shape, lesser: Shape) -> Self {
        Shape::Split {
            property,
            threshold,
            greater: Box::new(greater),
            lesser: Box::new(lesser),
        }
    }
}

/// The tree's token stream (breadth first, greater child first) and each
/// tag's context: leaves are numbered in the order the stream names them.
fn flatten(root: Shape, tags: usize) -> (Vec<Token>, Vec<u32>, usize) {
    let mut tokens = Vec::new();
    let mut contexts = vec![0; tags];
    let mut leaves = 0;
    let mut queue = VecDeque::from([root]);
    while let Some(node) = queue.pop_front() {
        match node {
            Shape::Split {
                property,
                threshold,
                greater,
                lesser,
            } => {
                tokens.extend([
                    Token::new(super::TREE_CTX_PROPERTY, property + 1),
                    Token::new(super::TREE_CTX_SPLITVAL, pack_signed(threshold)),
                ]);
                queue.extend([*greater, *lesser]);
            }
            Shape::Leaf {
                predictor,
                offset,
                tag,
            } => {
                contexts[tag] = leaves;
                leaves += 1;
                tokens.extend([
                    Token::new(super::TREE_CTX_PROPERTY, 0),
                    Token::new(super::TREE_CTX_PREDICTOR, predictor),
                    Token::new(super::TREE_CTX_OFFSET, pack_signed(offset)),
                    Token::new(super::TREE_CTX_MULTIPLIER_LOG, 0),
                    Token::new(super::TREE_CTX_MULTIPLIER_BITS, 0),
                ]);
            }
        }
    }
    (tokens, contexts, leaves as usize)
}

/// The distinct steps of the axis that names the interior of one group.
struct Steps {
    /// Sorted distinct property values.
    values: Vec<i32>,
    /// The interior follows `U` with the West predictor; otherwise `V` with
    /// North.
    along_rows: bool,
}

impl Steps {
    fn new(u: &[i32], v: &[i32]) -> Option<Self> {
        let distinct = |axis: &[i32], sign: i32| {
            let mut values: Vec<i32> = axis.array_windows().map(|pair| sign * step(pair)).collect();
            values.sort_unstable();
            values.dedup();
            values
        };
        if u.len() < 2 || v.len() < 2 {
            return Some(Steps {
                values: Vec::new(),
                along_rows: true,
            });
        }
        // West predictions see `NW - N`, the negated step of `U`.
        let (columns, rows) = (distinct(u, -1), distinct(v, 1));
        let along_rows = columns.len() < rows.len();
        let values = if along_rows { columns } else { rows };
        (values.len() <= MAX_STEPS).then_some(Steps { values, along_rows })
    }

    fn shape(&self, u: &[i32], v: &[i32]) -> Shape {
        let leaf = |predictor, offset, tag| Shape::Leaf {
            predictor,
            offset,
            tag,
        };
        let (predictor, property, sign) = if self.along_rows {
            (PREDICTOR_LEFT, PROPERTY_NORTH_STEP, -1)
        } else {
            (PREDICTOR_TOP, PROPERTY_WEST_STEP, 1)
        };
        fn interior(
            values: &[i32],
            first: usize,
            predictor: u32,
            property: u32,
            sign: i32,
        ) -> Shape {
            match values {
                [] => Shape::Leaf {
                    predictor,
                    offset: 0,
                    tag: TAG_INTERIOR,
                },
                &[value] => Shape::Leaf {
                    predictor,
                    offset: sign * value,
                    tag: TAG_INTERIOR + first,
                },
                _ => {
                    let middle = values.len() / 2;
                    Shape::split(
                        property,
                        values[middle - 1],
                        interior(&values[middle..], first + middle, predictor, property, sign),
                        interior(&values[..middle], first, predictor, property, sign),
                    )
                }
            }
        }
        Shape::split(
            PROPERTY_Y,
            0,
            Shape::split(
                PROPERTY_X,
                0,
                interior(&self.values, 0, predictor, property, sign),
                leaf(PREDICTOR_TOP, 0, TAG_COLUMN),
            ),
            Shape::split(
                PROPERTY_X,
                0,
                leaf(PREDICTOR_LEFT, 0, TAG_ROW),
                leaf(PREDICTOR_ZERO, sum(u[0], v[0]), TAG_CORNER),
            ),
        )
    }

    /// One token per sample in raster order. Only the first row and column
    /// carry a residual.
    fn tokenize(&self, u: &[i32], v: &[i32], contexts: &[u32], tokens: &mut Vec<Token>) {
        let (width, height) = (u.len(), v.len());
        tokens.reserve(width * height);
        tokens.push(Token::new(contexts[TAG_CORNER], 0));
        tokens.extend(
            u.array_windows()
                .map(|pair| Token::new(contexts[TAG_ROW], pack_signed(step(pair)))),
        );
        let context = |value| {
            let index = self.values.binary_search(&value).unwrap();
            contexts[TAG_INTERIOR + index]
        };
        let columns: Vec<u32> = if self.along_rows && height > 1 {
            u.array_windows().map(|pair| context(-step(pair))).collect()
        } else {
            Vec::new()
        };
        for pair in v.array_windows() {
            tokens.push(Token::new(contexts[TAG_COLUMN], pack_signed(step(pair))));
            if self.along_rows {
                tokens.extend(columns.iter().map(|&context| Token::new(context, 0)));
            } else if width > 1 {
                let row = Token::new(context(step(pair)), 0);
                tokens.resize(tokens.len() + width - 1, row);
            }
        }
    }
}

/// The tree and its tokens, appended to `out` as the smaller of two codings.
///
/// Without LZ77 the exactly predicted samples sit in one-symbol contexts and
/// cost nothing. With it every run of them costs a length, but the axes of a
/// repeating tile are themselves repeats.
fn write_shaped(
    pool: Option<&ThreadPool>,
    tree: &[Token],
    tokens: &[Token],
    contexts: usize,
    scratch: &mut CoderScratch,
    out: &mut BitWriter,
) {
    let plain = || {
        let mut huffman_pool = Vec::new();
        let mut plain = BitWriter::new();
        crate::entropy::write_raced_token_stream(
            tree,
            super::NUM_TREE_CONTEXTS,
            &mut huffman_pool,
            &mut plain,
        );
        crate::entropy::write_raced_token_stream(tokens, contexts, &mut huffman_pool, &mut plain);
        plain
    };
    let repeats = |scratch: &mut CoderScratch| {
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
        let mut repeats = BitWriter::new();
        write_tree_lz77(tree, &code, 64, &mut scratch.huffman_pool, &mut repeats);
        write_lz_section(&lz, contexts as u32, &code, 64, &mut repeats);
        repeats
    };
    let (plain, repeats) = match pool {
        Some(pool) => pool.join(plain, || repeats(scratch)),
        None => (plain(), repeats(scratch)),
    };
    out.append_bits(if plain.bits_written() < repeats.bits_written() {
        &plain
    } else {
        &repeats
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flattened tree read back the way a decoder reads it.
    enum Read {
        Split(u32, i32, usize),
        Leaf(u32, i32, u32),
    }

    fn read(tokens: &[Token]) -> Vec<Read> {
        let mut nodes = Vec::new();
        let (mut splits, mut leaves, mut at) = (0, 0, 0);
        while at < tokens.len() {
            assert_eq!(tokens[at].context, crate::modular::TREE_CTX_PROPERTY);
            if tokens[at].value == 0 {
                let unpack = |value: u32| ((value >> 1) as i32) ^ -((value & 1) as i32);
                nodes.push(Read::Leaf(
                    tokens[at + 1].value,
                    unpack(tokens[at + 2].value),
                    leaves,
                ));
                leaves += 1;
                at += 5;
            } else {
                let value = tokens[at + 1].value;
                let threshold = ((value >> 1) as i32) ^ -((value & 1) as i32);
                splits += 1;
                nodes.push(Read::Split(tokens[at].value - 1, threshold, 2 * splits - 1));
                at += 2;
            }
        }
        nodes
    }

    /// Decode one group from its tree and tokens with libjxl's neighbor
    /// fallbacks, and compare with the coordinate plane.
    fn check(u: &[i32], v: &[i32]) {
        let steps = Steps::new(u, v).unwrap();
        let (tree, contexts, _) =
            flatten(steps.shape(u, v), TAG_INTERIOR + steps.values.len().max(1));
        let nodes = read(&tree);
        let mut tokens = Vec::new();
        steps.tokenize(u, v, &contexts, &mut tokens);
        let (width, height) = (u.len(), v.len());
        assert_eq!(tokens.len(), width * height);
        let mut plane = vec![0i32; width * height];
        for y in 0..height {
            for x in 0..width {
                let at = y * width + x;
                let west = if x > 0 {
                    plane[at - 1]
                } else if y > 0 {
                    plane[at - width]
                } else {
                    0
                };
                let north = if y > 0 { plane[at - width] } else { west };
                let north_west = if x > 0 && y > 0 {
                    plane[at - width - 1]
                } else {
                    west
                };
                let mut node = 0;
                let (predictor, offset, context) = loop {
                    match nodes[node] {
                        Read::Split(property, threshold, greater) => {
                            let value = match property {
                                PROPERTY_Y => y as i32,
                                PROPERTY_X => x as i32,
                                PROPERTY_WEST_STEP => west - north_west,
                                PROPERTY_NORTH_STEP => north_west - north,
                                _ => unreachable!(),
                            };
                            node = greater + usize::from(value <= threshold);
                        }
                        Read::Leaf(predictor, offset, context) => {
                            break (predictor, offset, context);
                        }
                    }
                };
                assert_eq!(tokens[at].context, context, "context at {x},{y}");
                let value = tokens[at].value;
                let residual = ((value >> 1) as i32) ^ -((value & 1) as i32);
                let prediction = match predictor {
                    PREDICTOR_ZERO => 0,
                    PREDICTOR_LEFT => west,
                    PREDICTOR_TOP => north,
                    _ => unreachable!(),
                };
                plane[at] = prediction + offset + residual;
                assert_eq!(plane[at], sum(u[x], v[y]), "sample at {x},{y}");
                if x > 0 && y > 0 {
                    assert_eq!(residual, 0, "interior residual at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn portable_tree_predicts_the_interior_exactly() {
        for (width, height) in [(1, 1), (1, 40), (40, 1), (2, 2), (37, 29), (256, 256)] {
            let u: Vec<i32> = (0..width).map(|x| (x * x % 37) as i32 + 5).collect();
            let v: Vec<i32> = (0..height).map(|y| (y * 7 % 11) as i32 + 30).collect();
            check(&u, &v);
            check(&v, &u);
        }
        // Few steps along either axis.
        let u: Vec<i32> = (0..200).map(|x| x / 3).collect();
        let v: Vec<i32> = (0..100).map(|y| 300 - y).collect();
        check(&u, &v);
        check(&v, &u);
    }

    #[test]
    fn portable_sections_cover_every_group() {
        let (width, height) = (513, 259);
        let profile: Vec<i32> = (0..2048).map(|x| (x * x / 31) % 256).collect();
        let u: Vec<i32> = (0..width).map(|x| x as i32 / 2).collect();
        let v: Vec<i32> = (0..height).map(|y| y as i32 * 3).collect();
        let mut scratch = CoderScratch::lossless();
        let pool = ThreadPool::new(2);
        let (global, ac) = encode(&pool, &profile, &u, &v, &mut scratch).unwrap();
        assert!(global.bits_written() > 0);
        assert_eq!(ac.len(), 3 * 2);
        assert!(ac.iter().all(|section| section.bits_written() > 0));
        // One 1024-pixel group: a single section, no larger than the six.
        let mut frame = BitWriter::new();
        encode_frame(&pool, &profile, &u, &v, false, &mut scratch, &mut frame).unwrap();
        let split = global.bits_written() + ac.iter().map(BitWriter::bits_written).sum::<usize>();
        assert!(frame.bits_written() % 8 == 0 && frame.bits_written() < split + 256);
        // With the second Palette the groups hold a tree and nothing else.
        let (u, v): (Vec<i32>, Vec<i32>) = (
            (0..2100).map(|x| x * x % 97).collect(),
            (0..1100).map(|y| y * 5 % 89 + 7).collect(),
        );
        let (mut direct, mut delta) = (BitWriter::new(), BitWriter::new());
        encode_frame(&pool, &profile, &u, &v, false, &mut scratch, &mut direct).unwrap();
        encode_frame(&pool, &profile, &u, &v, true, &mut scratch, &mut delta).unwrap();
        assert!(delta.bits_written() < direct.bits_written());
        let small = encode(&pool, &profile, &u[..200], &v[..100], &mut scratch).unwrap();
        assert_eq!(small.1.len(), 1);
        assert_eq!(small.1[0].bits_written(), 0);
    }
}
