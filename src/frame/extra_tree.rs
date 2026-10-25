/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! A lossless extra channel coded through the frame's global tree: its learned
//! subtree hangs under a stream-id split beside the DC tree, and its contexts
//! join the DC entropy code, so the AC-group streams carry no tree or
//! histograms of their own.

use super::dc_tokens::{DcTreeChoice, write_tree_tokens};
use super::extra_channels::{ExtraPalette, ExtraSections};
use super::{ImageDim, K_GROUP_DIM};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::encoding_context::EncodingContext;
use crate::entropy::{EntropyCode, OwnedEntropyCode, Token, pack_signed, write_entropy_code};
use std::collections::VecDeque;

const TREE_CTX_SPLITVAL: u32 = 0;
const TREE_CTX_PROPERTY: u32 = 1;
const PROP_STREAM: u32 = 1;

#[derive(Clone, Copy)]
enum Node {
    /// PROPERTY and SPLITVAL token values; `prop > val` routes to `gt`.
    Split {
        prop: u32,
        val: u32,
        gt: usize,
        le: usize,
    },
    /// PREDICTOR, OFFSET, MULTIPLIER_LOG and MULTIPLIER_BITS token values.
    Leaf([u32; 4]),
}

/// Decode a BFS tree token stream into nodes (children in BFS order).
fn parse_tree(tokens: &[Token]) -> Vec<Node> {
    let mut nodes = Vec::new();
    let mut declared = 1usize;
    let mut i = 0usize;
    while nodes.len() < declared {
        let prop = tokens[i].value;
        if prop == 0 {
            let leaf = std::array::from_fn(|k| tokens[i + 1 + k].value);
            nodes.push(Node::Leaf(leaf));
            i += 5;
        } else {
            nodes.push(Node::Split {
                prop,
                val: tokens[i + 1].value,
                gt: declared,
                le: declared + 1,
            });
            declared += 2;
            i += 2;
        }
    }
    debug_assert_eq!(i, tokens.len());
    nodes
}

/// `stream > threshold ? gt : le` over two trees, BFS-emitted. Returns the
/// tokens, each tree's leaf-context map (its own context -> merged context)
/// and the merged context count.
fn merge_trees(threshold: i32, gt: &[Node], le: &[Node]) -> (Vec<Token>, [Vec<u32>; 2], u32) {
    let trees = [gt, le];
    let mut tokens = vec![
        Token::new(TREE_CTX_PROPERTY, PROP_STREAM + 1),
        Token::new(TREE_CTX_SPLITVAL, pack_signed(threshold)),
    ];
    let mut maps: [Vec<u32>; 2] = Default::default();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::from([(0, 0), (1, 0)]);
    let mut ctx = 0u32;
    while let Some((t, i)) = queue.pop_front() {
        match trees[t][i] {
            Node::Split { prop, val, gt, le } => {
                tokens.push(Token::new(TREE_CTX_PROPERTY, prop));
                tokens.push(Token::new(TREE_CTX_SPLITVAL, val));
                queue.push_back((t, gt));
                queue.push_back((t, le));
            }
            Node::Leaf(leaf) => {
                tokens.push(Token::new(TREE_CTX_PROPERTY, 0));
                tokens.extend((0..4).map(|k| Token::new(2 + k as u32, leaf[k])));
                maps[t].push(ctx);
                ctx += 1;
            }
        }
    }
    (tokens, maps, ctx)
}

fn remap(streams: &[Vec<Token>], map: &[u32]) -> Vec<Vec<Token>> {
    streams
        .iter()
        .map(|s| {
            s.iter()
                .map(|t| Token::new(map[t.context as usize], t.value))
                .collect()
        })
        .collect()
}

fn ans_bits(tokens: &[Token], code: &EntropyCode) -> usize {
    let mut w = BitWriter::new();
    crate::entropy::write_ans_tokens(
        tokens,
        code.context_map,
        code.ans_symbols,
        code.ans_reverse_maps,
        code.hybrid_uint_configs,
        &mut w,
    );
    w.bits_written()
}

/// The global tree and its code, as `write_dc_global` writes them.
fn header_bits(tree: &[Token], code: &EntropyCode, scratch: &mut CoderScratch) -> usize {
    let mut w = BitWriter::new();
    write_tree_tokens(tree, &mut scratch.huffman_pool, &mut w);
    w.write(1, 0);
    write_entropy_code(code, &mut scratch.huffman_pool, &mut w);
    w.bits_written()
}

/// A plane stream coded through the global tree: use_global_tree, default
/// WP and, for the global stream of a palette-coded single-group plane,
/// the Palette transform over its `nb_colors` entries.
fn global_section(tokens: &[Token], code: &EntropyCode, palette: Option<u32>) -> BitWriter {
    let mut w = BitWriter::new();
    w.write(1, 1); // use_global_tree
    w.write(1, 1); // wp_header all_default
    match palette {
        Some(nb_colors) => crate::lossless::write_palette_transform(1, nb_colors, &mut w),
        None => w.write(2, 0), // no transforms
    }
    crate::entropy::write_ans_tokens(
        tokens,
        code.context_map,
        code.ans_symbols,
        code.ans_reverse_maps,
        code.hybrid_uint_configs,
        &mut w,
    );
    w
}

/// The frame's DC streams under its global tree: what a joint code is priced
/// against, and what a winning joint code replaces.
pub(super) struct GlobalStreams<'a> {
    pub(super) tree: &'a mut DcTreeChoice,
    pub(super) code: &'a mut OwnedEntropyCode,
    pub(super) dc: &'a mut Vec<Vec<Token>>,
    pub(super) meta: &'a mut Vec<Vec<Token>>,
    /// Whether the DC streams are written (not with a DC frame).
    pub(super) dc_written: bool,
}

/// A joint coding of the frame's DC, metadata and extra streams that beats
/// the local trees: the merged tree, its code, the remapped DC streams and
/// the extra channel's sections.
pub(super) struct GlobalExtraTree {
    tree: Vec<Token>,
    code: OwnedEntropyCode,
    dc: Vec<Vec<Token>>,
    meta: Vec<Vec<Token>>,
    /// The global section of a single-group plane, else one per AC group.
    sections: Vec<BitWriter>,
    single: bool,
    saving: usize,
}

impl GlobalExtraTree {
    /// Bits the joint coding saves over the local trees, DC side included.
    pub(super) fn saving(&self) -> usize {
        self.saving
    }

    /// Install the joint coding: the DC streams move onto the merged tree and
    /// each extra section keeps whichever coding is smaller.
    pub(super) fn apply(self, streams: &mut GlobalStreams<'_>, extra: &mut ExtraSections) {
        let targets: Vec<&mut BitWriter> = if self.single {
            vec![&mut extra.global]
        } else {
            extra.ac.iter_mut().collect()
        };
        for (target, section) in targets.into_iter().zip(self.sections) {
            if section.bits_written() < target.bits_written() {
                *target = section;
            }
        }
        *streams.tree = DcTreeChoice::Learned(self.tree);
        *streams.code = self.code;
        *streams.dc = self.dc;
        *streams.meta = self.meta;
    }
}

/// Learn the extra channel's subtree (over its palette indices when it has
/// one), merge it beside the DC tree and price the frame's DC, metadata and
/// extra streams under the joint code against the local codings in `extra`.
/// `None` when the local codings are smaller.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_global_extra_tree(
    ctx: &EncodingContext,
    plane: &AlphaPlane,
    palette: Option<&ExtraPalette>,
    dim: &ImageDim,
    last_pass: usize,
    streams: &GlobalStreams<'_>,
    extra: &ExtraSections,
    scratch: &mut CoderScratch,
) -> Option<GlobalExtraTree> {
    if matches!(plane, AlphaPlane::F32(_)) || streams.code.use_prefix_code {
        return None;
    }
    let single = dim.xsize <= K_GROUP_DIM && dim.ysize <= K_GROUP_DIM;
    let crops: Vec<crate::lossless::StreamCrop> = if single {
        vec![(0, 0, dim.xsize, dim.ysize, 0)]
    } else {
        (0..dim.num_groups)
            .map(|g| {
                let x0 = (g % dim.xsize_groups) * K_GROUP_DIM;
                let y0 = (g / dim.xsize_groups) * K_GROUP_DIM;
                let id = last_pass * dim.num_groups + g;
                (
                    x0,
                    y0,
                    K_GROUP_DIM.min(dim.xsize - x0),
                    K_GROUP_DIM.min(dim.ysize - y0),
                    crate::lossless::ac_stream_id(dim.num_dc_groups, id),
                )
            })
            .collect()
    };
    let coded = palette.map_or(plane, |p| &p.indices);
    let mut data: Vec<i32> = (0..coded.len()).map(|i| coded.get_i32(i)).collect();
    // A single-group plane's palette is channel 0 of its stream and the
    // indices channel 1; elsewhere the palette has the global stream to
    // itself and each group's indices are that group's channel 0.
    let (chan, palette_row) = match palette {
        Some(p) if single => (1, Some(p.entries.as_slice())),
        _ => (0, None),
    };
    let learned = crate::lossless::learn_global_plane_tree(
        &mut data,
        dim.xsize,
        chan,
        palette_row,
        &crops,
        &ctx.thread_pool,
        scratch,
    );
    drop(data);

    let dc_tokens = streams.tree.tokens(dim.num_dc_groups);
    let dc_nodes = parse_tree(&dc_tokens);
    let plane_nodes = parse_tree(&learned.tree_tokens);
    // The single-group plane is the global stream (id 0) below every DC
    // stream; AC-group streams lie above every DC and metadata stream.
    let (tree_tokens, dc_map, plane_map) = if single {
        let (tokens, [dc_map, plane_map], _) = merge_trees(0, &dc_nodes, &plane_nodes);
        (tokens, dc_map, plane_map)
    } else {
        let threshold = crate::lossless::ac_stream_id(dim.num_dc_groups, 0) - 1;
        let (tokens, [plane_map, dc_map], _) = merge_trees(threshold, &plane_nodes, &dc_nodes);
        (tokens, dc_map, plane_map)
    };
    let num_ctx = dc_map.len() + plane_map.len();
    let dc = remap(streams.dc, &dc_map);
    let meta = remap(streams.meta, &dc_map);
    let plane_streams = remap(&learned.crop_tokens, &plane_map);
    drop(learned);

    let all = || {
        dc.iter()
            .chain(&meta)
            .chain(&plane_streams)
            .map(Vec::as_slice)
    };
    let effort = ctx.speed.effort();
    let mut code = crate::entropy::optimize_entropy_code_ac_streams(
        all(),
        num_ctx,
        &mut scratch.huffman_pool,
        effort.huc_select,
        ctx.speed,
        Some(&ctx.thread_pool),
    );
    crate::entropy::refine_ans_clusters(
        &mut code,
        all(),
        crate::entropy::AnsRefinement::slow_for_speed(ctx.speed),
        &ctx.thread_pool,
        scratch,
    );

    let frame_bits = |tree: &[Token],
                      code: &EntropyCode,
                      dc: &[Vec<Token>],
                      meta: &[Vec<Token>],
                      scratch: &mut CoderScratch| {
        let dc_bits: usize = if streams.dc_written {
            dc.iter().map(|s| ans_bits(s, code)).sum()
        } else {
            0
        };
        header_bits(tree, code, scratch)
            + dc_bits
            + meta.iter().map(|s| ans_bits(s, code)).sum::<usize>()
    };
    let local_bits = frame_bits(
        &dc_tokens,
        &streams.code.as_ref(),
        streams.dc,
        streams.meta,
        scratch,
    ) + extra.bits();
    let joint = code.as_ref();
    let header_palette = palette.filter(|_| single).map(|p| p.entries.len() as u32);
    let sections: Vec<BitWriter> = plane_streams
        .iter()
        .map(|s| global_section(s, &joint, header_palette))
        .collect();
    let local_sections: Vec<&BitWriter> = if single {
        vec![&extra.global]
    } else {
        extra.ac.iter().collect()
    };
    let plane_bits: usize = sections
        .iter()
        .zip(&local_sections)
        .map(|(g, l)| g.bits_written().min(l.bits_written()))
        .sum();
    let fixed_bits = if single {
        0
    } else {
        extra.global.bits_written()
    };
    let global_bits =
        frame_bits(&tree_tokens, &joint, &dc, &meta, scratch) + plane_bits + fixed_bits;
    if global_bits >= local_bits {
        return None;
    }
    Some(GlobalExtraTree {
        tree: tree_tokens,
        code,
        dc,
        meta,
        sections,
        single,
        saving: local_bits - global_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(pred: u32) -> [Token; 5] {
        [
            Token::new(TREE_CTX_PROPERTY, 0),
            Token::new(2, pred),
            Token::new(3, 0),
            Token::new(4, 0),
            Token::new(5, 0),
        ]
    }

    #[test]
    fn merged_tree_keeps_each_subtree_and_maps_its_leaves() {
        // a: x > 3 ? (y > 1 ? L0 : L1) : L2;  b: single leaf.
        let mut a = vec![Token::new(1, 4), Token::new(0, pack_signed(3))];
        a.extend([Token::new(1, 3), Token::new(0, pack_signed(1))]);
        a.extend(leaf(5));
        a.extend(leaf(1));
        a.extend(leaf(2));
        let b: Vec<Token> = leaf(0).to_vec();
        let (na, nb) = (parse_tree(&a), parse_tree(&b));
        assert_eq!(na.len(), 5);
        let (tokens, [map_a, map_b], n) = merge_trees(7, &na, &nb);
        assert_eq!(n, 4);
        let merged = parse_tree(&tokens);
        assert_eq!(merged.len(), 7);
        let Node::Split { prop, gt, le, .. } = merged[0] else {
            panic!("root must split");
        };
        assert_eq!(prop, PROP_STREAM + 1);
        // BFS: root, a.root, b's leaf (ctx 0), a.split, then a's leaves.
        assert!(matches!(merged[le], Node::Leaf([0, ..])));
        assert!(matches!(merged[gt], Node::Split { prop: 4, .. }));
        assert_eq!(map_b, vec![0]);
        assert_eq!(map_a, vec![1, 2, 3]);
    }
}
