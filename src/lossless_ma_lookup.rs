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
use crate::coder_scratch::LZ77_MAX_CONTEXTS;
use crate::ma_tree::{LearnedTree, MaNode, NUM_MA_PREDS, NUM_MA_PROPS};

// The learner reserves one context for distance symbols, so its split count
// is below this power-of-two capacity. Masked fixed-array indices let the hot
// walk stay bounds-check-free in safe Rust. Construction validates each field.
const SPLIT_CAPACITY: usize = LZ77_MAX_CONTEXTS;
const SPLIT_MASK: u32 = SPLIT_CAPACITY as u32 - 1;
const LEAF: u32 = 1 << 31;
const PREDICTOR_BITS: u32 = 4;
const _: () = assert!(SPLIT_CAPACITY.is_power_of_two());
const _: () = assert!(NUM_MA_PREDS <= 1 << PREDICTOR_BITS);
const _: () = assert!((SPLIT_CAPACITY as u64) << PREDICTOR_BITS < LEAF as u64);

#[derive(Clone, Copy, Default)]
struct Split {
    property: u32,
    value: i32,
    // Both targets share one load: low word for <=, high word for >.
    children: u64,
}

/// An execution form of the emitted tree for one fixed channel and stream id. The
/// original tree still drives learning, estimates, and codestream emission.
/// Leaf targets embed the original context and predictor, avoiding another
/// node load and a separate context-table lookup at every pixel.
pub(super) struct MaLookup {
    splits: Box<[Split; SPLIT_CAPACITY]>,
    split_count: usize,
    root: u32,
    needs_wp: bool,
}

impl MaLookup {
    pub(super) fn new(tree: &LearnedTree, contexts: &[u32], channel: u32, stream: i32) -> Self {
        let mut lookup = Self {
            splits: Box::new([Split::default(); SPLIT_CAPACITY]),
            split_count: 0,
            root: 0,
            needs_wp: false,
        };
        lookup.root = lookup.add_node(tree, contexts, [channel as i32, stream], 0);
        lookup
    }

    fn add_node(
        &mut self,
        tree: &LearnedTree,
        contexts: &[u32],
        fixed: [i32; 2],
        index: u32,
    ) -> u32 {
        match tree.nodes[index as usize] {
            MaNode::Leaf { pred } => {
                let context = contexts[index as usize];
                assert!(pred < NUM_MA_PREDS as u32);
                assert!(context < LZ77_MAX_CONTEXTS as u32);
                self.needs_wp |= pred == 6;
                LEAF | (context << PREDICTOR_BITS) | pred
            }
            MaNode::Split { prop, val, gt, le } => {
                assert!((prop as usize) < NUM_MA_PROPS);
                // These properties are constant throughout this channel's
                // stream. Folding them preserves every possible pixel route.
                if prop <= 1 {
                    let value = fixed[prop as usize];
                    return self.add_node(tree, contexts, fixed, if value > val { gt } else { le });
                }
                let slot = self.split_count;
                assert!(slot < SPLIT_CAPACITY);
                self.split_count += 1;
                self.needs_wp |= prop == 15;
                let le = self.add_node(tree, contexts, fixed, le);
                let gt = self.add_node(tree, contexts, fixed, gt);
                self.splits[slot] = Split {
                    property: prop as u32,
                    value: val,
                    children: u64::from(le) | (u64::from(gt) << u32::BITS),
                };
                slot as u32
            }
        }
    }

    pub(super) fn needs_wp(&self) -> bool {
        self.needs_wp
    }

    /// Return the unchanged leaf context and predictor for this channel.
    #[inline]
    pub(super) fn lookup(&self, properties: &[i32; NUM_MA_PROPS]) -> (u32, u32) {
        let mut target = self.root;
        while target & LEAF == 0 {
            let split = &self.splits[(target & SPLIT_MASK) as usize];
            // `property < NUM_MA_PROPS` is asserted at construction.
            let value = properties[split.property as usize];
            // Selecting a word from one loaded value avoids a child load
            // whose address depends on the comparison result.
            let shift = u32::from(value > split.value) * u32::BITS;
            target = (split.children >> shift) as u32;
        }
        (
            (target & !LEAF) >> PREDICTOR_BITS,
            target & ((1 << PREDICTOR_BITS) - 1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(tree: &LearnedTree, contexts: &[u32], channel: u32, mut p: [i32; NUM_MA_PROPS]) {
        p[0] = channel as i32;
        for stream in [0, 21, 77, i32::MIN, i32::MAX] {
            p[1] = stream;
            let lookup = MaLookup::new(tree, contexts, channel, stream);
            let (node, pred) = tree.lookup(&p);
            assert_eq!(lookup.lookup(&p), (contexts[node as usize], pred));
        }
    }

    #[test]
    fn lookup_preserves_every_property_comparison_and_constant_branch() {
        for prop in 0..NUM_MA_PROPS as u8 {
            for threshold in [i32::MIN, -1, 0, 1, i32::MAX] {
                let tree = LearnedTree {
                    nodes: vec![
                        MaNode::Split {
                            prop,
                            val: threshold,
                            gt: 1,
                            le: 2,
                        },
                        MaNode::Leaf { pred: 13 },
                        MaNode::Leaf { pred: 6 },
                    ],
                    est_bits: 0.,
                    flat_bits: 0.,
                };
                let contexts = [u32::MAX, LZ77_MAX_CONTEXTS as u32 - 1, 7];
                for channel in [0, 1, 7, u32::MAX] {
                    for value in [
                        i32::MIN,
                        threshold.saturating_sub(1),
                        threshold,
                        threshold.saturating_add(1),
                        i32::MAX,
                    ] {
                        let mut p = [0; NUM_MA_PROPS];
                        p[prop as usize] = value;
                        check(&tree, &contexts, channel, p);
                    }
                    let lookup = MaLookup::new(&tree, &contexts, channel, 0);
                    if prop <= 1 {
                        assert_eq!(lookup.split_count, 0);
                        let value = if prop == 0 { channel as i32 } else { 0 };
                        assert_eq!(lookup.needs_wp(), value <= threshold);
                    } else {
                        assert_eq!(lookup.split_count, 1);
                        assert!(lookup.needs_wp());
                    }
                }
            }
        }
    }

    #[test]
    fn lookup_preserves_all_routes_near_the_split_capacity() {
        const DEPTH: usize = SPLIT_CAPACITY.ilog2() as usize;
        const { assert!(DEPTH + 2 <= NUM_MA_PROPS) };
        fn build(
            nodes: &mut Vec<MaNode>,
            contexts: &mut Vec<u32>,
            depth: usize,
            leaves: &mut u32,
        ) -> u32 {
            let index = nodes.len() as u32;
            nodes.push(MaNode::Leaf { pred: 0 });
            contexts.push(u32::MAX);
            if depth == 0 {
                nodes[index as usize] = MaNode::Leaf {
                    pred: *leaves % NUM_MA_PREDS as u32,
                };
                contexts[index as usize] = SPLIT_CAPACITY as u32 - 1 - *leaves;
                *leaves += 1;
            } else {
                let le = build(nodes, contexts, depth - 1, leaves);
                let gt = build(nodes, contexts, depth - 1, leaves);
                nodes[index as usize] = MaNode::Split {
                    prop: (2 + DEPTH - depth) as u8,
                    val: 0,
                    gt,
                    le,
                };
            }
            index
        }
        let mut nodes = Vec::new();
        let mut contexts = Vec::new();
        build(&mut nodes, &mut contexts, DEPTH, &mut 0);
        let tree = LearnedTree {
            nodes,
            est_bits: 0.,
            flat_bits: 0.,
        };
        let lookup = MaLookup::new(&tree, &contexts, 3, 0);
        assert_eq!(lookup.split_count, SPLIT_CAPACITY - 1);
        for route in 0..SPLIT_CAPACITY {
            let mut p = [0; NUM_MA_PROPS];
            p[0] = 3;
            for (bit, value) in p[2..2 + DEPTH].iter_mut().enumerate() {
                *value = if route & (1 << bit) == 0 { -1 } else { 1 };
            }
            let (node, pred) = tree.lookup(&p);
            assert_eq!(lookup.lookup(&p), (contexts[node as usize], pred));
        }
    }

    #[test]
    fn lookup_rejects_fields_that_would_be_truncated() {
        for (nodes, contexts) in [
            (
                vec![MaNode::Leaf {
                    pred: NUM_MA_PREDS as u32,
                }],
                vec![0],
            ),
            (
                vec![MaNode::Leaf { pred: 0 }],
                vec![LZ77_MAX_CONTEXTS as u32],
            ),
            (
                vec![MaNode::Split {
                    prop: NUM_MA_PROPS as u8,
                    val: 0,
                    gt: 0,
                    le: 0,
                }],
                vec![0],
            ),
        ] {
            let tree = LearnedTree {
                nodes,
                est_bits: 0.,
                flat_bits: 0.,
            };
            assert!(std::panic::catch_unwind(|| MaLookup::new(&tree, &contexts, 0, 0)).is_err());
        }
    }
}
