/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use super::DCT_BLOCK_SIZE;
use crate::entropy::Token;

pub(super) const PERMUTATION_CONTEXTS: usize = 8;

fn coeff_order_context(val: u32) -> u32 {
    let token = if val == 0 {
        0
    } else {
        1 + (31 - val.leading_zeros())
    };
    token.min(PERMUTATION_CONTEXTS as u32 - 1)
}

pub(super) fn compute_order(
    nonzero: &[u64; DCT_BLOCK_SIZE],
    num_blocks: u64,
    skip: usize,
) -> [u8; DCT_BLOCK_SIZE] {
    // Bucket width mirrors libjxl: normalizing by sqrt(N) makes the number of
    // distinct buckets grow like sqrt(N), coarse enough to suppress noise.
    let inv = if num_blocks == 0 {
        0.0
    } else {
        1.0 / (num_blocks as f64).sqrt()
    };

    let mut slots: Vec<usize> = (skip..DCT_BLOCK_SIZE).collect();
    // Zero count = num_blocks - nonzero; bucket it ascending (fewest zeros, i.e.
    // most often non-zero, first). Slot index is the stable tiebreak.
    let bucket = |slot: usize| -> u64 {
        let zeros = num_blocks - nonzero[slot];
        (zeros as f64 * inv + 0.1) as u64
    };
    slots.sort_by(|&a, &b| bucket(a).cmp(&bucket(b)).then(a.cmp(&b)));

    let mut order = [0u8; DCT_BLOCK_SIZE];
    for (k, slot) in (0..skip).chain(slots).enumerate() {
        order[k] = slot as u8;
    }
    order
}

/// Whether an order is the identity (no reordering, nothing to signal).
pub(super) fn is_identity(order: &[u8; DCT_BLOCK_SIZE]) -> bool {
    order.iter().enumerate().all(|(k, &v)| k as u8 == v)
}

/// Computes the Lehmer code of a permutation via a Fenwick tree, exactly as
/// libjxl's `ComputeLehmerCode` does.
fn lehmer_code(perm: &[u8; DCT_BLOCK_SIZE]) -> [u8; DCT_BLOCK_SIZE] {
    let n = DCT_BLOCK_SIZE;
    let mut fenwick = [0u32; DCT_BLOCK_SIZE + 1];
    let mut code = [0u8; DCT_BLOCK_SIZE];
    for (idx, &s) in perm[..n].iter().enumerate() {
        let s = s as usize;
        // Prefix sum over [1, s+1]: how many smaller values already placed.
        let mut penalty = 0u32;
        let mut i = s + 1;
        while i != 0 {
            penalty += fenwick[i];
            i &= i - 1;
        }
        code[idx] = (s as u32 - penalty) as u8;
        // Point update at s+1.
        let mut i = s + 1;
        while i < n + 1 {
            fenwick[i] += 1;
            i += i & i.wrapping_neg();
        }
    }
    code
}
pub(super) fn tokenize_permutation(
    order: &[u8; DCT_BLOCK_SIZE],
    skip: usize,
    out: &mut Vec<Token>,
) {
    let lehmer = lehmer_code(order);
    // Trailing zero digits are implied; find where the meaningful tail ends.
    let mut end = skip;
    for i in skip..DCT_BLOCK_SIZE {
        if lehmer[i] != 0 {
            end = i + 1;
        }
    }
    out.push(Token::new(
        coeff_order_context(DCT_BLOCK_SIZE as u32),
        (end - skip) as u32,
    ));
    let mut last = 0u32;
    for &digit in lehmer.iter().take(end).skip(skip) {
        out.push(Token::new(coeff_order_context(last), digit as u32));
        last = digit as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstructs a permutation from its Lehmer code the naive way, to check
    /// `lehmer_code` against an independent implementation.
    fn decode_lehmer(code: &[u8; DCT_BLOCK_SIZE]) -> [u8; DCT_BLOCK_SIZE] {
        let mut avail: Vec<u8> = (0..DCT_BLOCK_SIZE as u8).collect();
        let mut perm = [0u8; DCT_BLOCK_SIZE];
        for i in 0..DCT_BLOCK_SIZE {
            perm[i] = avail.remove(code[i] as usize);
        }
        perm
    }

    #[test]
    fn lehmer_round_trips() {
        // Identity.
        let mut id = [0u8; DCT_BLOCK_SIZE];
        for (k, v) in id.iter_mut().enumerate() {
            *v = k as u8;
        }
        assert_eq!(lehmer_code(&id), [0u8; DCT_BLOCK_SIZE]);
        assert_eq!(decode_lehmer(&lehmer_code(&id)), id);

        // A deterministic shuffle.
        let mut perm = id;
        let mut state = 0x1234_5678u32;
        for i in (1..DCT_BLOCK_SIZE).rev() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let j = (state >> 8) as usize % (i + 1);
            perm.swap(i, j);
        }
        assert_eq!(decode_lehmer(&lehmer_code(&perm)), perm);
    }

    #[test]
    fn identity_order_when_uniform() {
        // Uniform statistics leave the order untouched (all in one bucket).
        let nonzero = [5u64; DCT_BLOCK_SIZE];
        let order = compute_order(&nonzero, 10, 1);
        assert!(is_identity(&order));
    }

    #[test]
    fn frequent_slots_move_forward() {
        let mut nonzero = [0u64; DCT_BLOCK_SIZE];
        // Slot 40 is non-zero in every block; nearby low slots rarely are.
        nonzero[40] = 1000;
        let order = compute_order(&nonzero, 1000, 1);
        // The always-present slot must land near the front (right after LLF).
        let pos = order.iter().position(|&v| v == 40).unwrap();
        assert_eq!(pos, 1, "hot slot should sort to the first AC position");
        assert!(!is_identity(&order));
    }

    #[test]
    fn context_matches_hybrid_uint_zero_config() {
        // min(0 or 1+floor_log2(v), 7).
        assert_eq!(coeff_order_context(0), 0);
        assert_eq!(coeff_order_context(1), 1);
        assert_eq!(coeff_order_context(2), 2);
        assert_eq!(coeff_order_context(3), 2);
        assert_eq!(coeff_order_context(4), 3);
        assert_eq!(coeff_order_context(63), 6);
        assert_eq!(coeff_order_context(64), 7);
        assert_eq!(coeff_order_context(1000), 7);
    }
}
