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

struct HuffmanNode {
    total_count: u32,
    index_left: i16,
    index_right_or_value: i16, // for leaves: the symbol; for internal: right child index
}

const SENTINEL: HuffmanNode = HuffmanNode {
    total_count: u32::MAX,
    index_left: -1,
    index_right_or_value: -1,
};

fn set_depth(idx: usize, pool: &[HuffmanNode], depth: &mut [u8], level: u8) {
    let node = &pool[idx];
    if node.index_left >= 0 {
        let new_level = level + 1;
        set_depth(node.index_left as usize, pool, depth, new_level);
        set_depth(node.index_right_or_value as usize, pool, depth, new_level);
    } else {
        depth[node.index_right_or_value as usize] = level;
    }
}

pub fn create_huffman_tree(data: &[u32], tree_limit: u8, depth: &mut [u8]) {
    let length = data.len();
    debug_assert!(depth.len() >= length);
    for d in depth[..length].iter_mut() {
        *d = 0;
    }

    let mut count_limit: u32 = 1;
    loop {
        let mut tree: Vec<HuffmanNode> = Vec::with_capacity(2 * length + 1);

        // Gather nonzero-count leaves in reverse order; stable sort keeps
        // tie-breaking deterministic and matching libjxl.
        for (i, &data) in data[..length].iter().enumerate().rev() {
            if data != 0 {
                let count = data.max(count_limit.saturating_sub(1));
                tree.push(HuffmanNode {
                    total_count: count,
                    index_left: -1,
                    index_right_or_value: i as i16,
                });
            }
        }

        let n = tree.len();
        if n == 1 {
            // Single symbol: placeholder depth. The simple-Huffman-tree fast
            // path in the writer handles the actual emission.
            depth[tree[0].index_right_or_value as usize] = 1;
            return;
        }

        // Stable sort by ascending count.
        tree.sort_by(|a, b| a.total_count.cmp(&b.total_count));

        tree.push(SENTINEL);
        tree.push(SENTINEL);

        let mut i = 0usize; // next leaf
        let mut j = n + 1; // next internal node

        // Build n - 1 internal nodes.
        for _ in 0..(n - 1) {
            let left;
            if tree[i].total_count <= tree[j].total_count {
                left = i;
                i += 1;
            } else {
                left = j;
                j += 1;
            }
            let right;
            if tree[i].total_count <= tree[j].total_count {
                right = i;
                i += 1;
            } else {
                right = j;
                j += 1;
            }

            // Overwrite the trailing sentinel as the new parent.
            let parent = tree.len() - 1;
            tree[parent].total_count = tree[left].total_count + tree[right].total_count;
            tree[parent].index_left = left as i16;
            tree[parent].index_right_or_value = right as i16;

            // Replace the trailing sentinel for next iteration.
            tree.push(SENTINEL);
        }
        debug_assert_eq!(tree.len(), 2 * n + 1);

        // Root is at 2n - 1.
        set_depth(2 * n - 1, &tree, depth, 0);

        let max_depth = depth[..length].iter().copied().max().unwrap_or(0);
        if max_depth <= tree_limit {
            return;
        }

        count_limit = count_limit
            .checked_mul(2)
            .expect("huffman depth limit unreachable");
    }
}
