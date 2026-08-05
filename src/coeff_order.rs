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

use crate::entropy::{HybridUintConfig, Token, uint_encode_with_config};

/// libjxl `kPermutationContexts`.
pub(crate) const PERMUTATION_CONTEXTS: usize = 8;

/// libjxl `CoeffOrderContext`: the hybrid-uint token of `val` under the
/// `(0, 0, 0)` config, clamped to the context count. That config makes the
/// token `0` for `val == 0` and `1 + floor(log2(val))` otherwise.
#[inline]
pub(crate) fn coeff_order_context(val: u32) -> u32 {
    const CFG: HybridUintConfig = HybridUintConfig {
        split_exponent: 0,
        msb_in_token: 0,
        lsb_in_token: 0,
    };
    let (token, _, _) = uint_encode_with_config(val, CFG);
    token.min(PERMUTATION_CONTEXTS as u32 - 1)
}

/// Lehmer (inversion) code of `permutation`, via the Fenwick tree libjxl uses.
///
/// `code[i]` is how many entries after position `i` are smaller than
/// `permutation[i]`, so the identity permutation codes to all zeros.
pub(crate) fn compute_lehmer_code(permutation: &[u32], code: &mut [u32]) {
    let n = permutation.len();
    debug_assert_eq!(code.len(), n);
    let mut temp = vec![0u32; n + 1];
    for (idx, &s) in permutation.iter().enumerate() {
        let mut penalty = 0u32;
        let mut i = s + 1;
        while i != 0 {
            penalty += temp[i as usize];
            i &= i - 1; // clear lowest set bit
        }
        debug_assert!(s >= penalty, "not a permutation at {idx}");
        code[idx] = s - penalty;
        let mut i = s + 1;
        while (i as usize) < n + 1 {
            temp[i as usize] += 1;
            i += i & i.wrapping_neg(); // add lowest set bit
        }
    }
}

/// Inverse of [`compute_lehmer_code`]. Only used by tests, as the round-trip is
/// the only cheap way to prove the encoder's Lehmer code is the one the decoder
/// will invert.
#[cfg(test)]
pub(crate) fn decode_lehmer_code(code: &[u32], permutation: &mut [u32]) {
    let n = code.len();
    let mut available: Vec<u32> = (0..n as u32).collect();
    for (i, &c) in code.iter().enumerate() {
        permutation[i] = available.remove(c as usize);
    }
}

/// Emit the tokens describing `zigzag` (a permutation in natural-scan space),
/// leaving its first `skip` entries alone.
pub(crate) fn tokenize_permutation(zigzag: &[u32], skip: usize, tokens: &mut Vec<Token>) {
    let size = zigzag.len();
    let mut lehmer = vec![0u32; size];
    compute_lehmer_code(zigzag, &mut lehmer);
    let mut end = size;
    while end > skip && lehmer[end - 1] == 0 {
        end -= 1;
    }
    tokens.push(Token::new(
        coeff_order_context(size as u32),
        (end - skip) as u32,
    ));
    let mut last = 0u32;
    for &l in &lehmer[skip..end] {
        tokens.push(Token::new(coeff_order_context(last), l));
        last = l;
    }
}

/// The order groups jixel can actually produce, as
/// `(order index, llf blocks, coefficient count)`.
///
/// Indices come from `K_STRATEGY_ORDER`: 0 = DCT8, 1 = the 8x8-sized family
/// (DCT4X4/4X8/8X4), 2 = DCT16X16, 3 = DCT32X32, 4 = DCT16X8/8X16,
/// 6 = DCT32X16/16X32. DCT64X64 uses group 7 but intentionally stays on its
/// natural order, so it has no entry here and is never signaled.
pub(crate) static ORDER_SPECS: [(usize, usize, usize); 6] = [
    (0, 1, 64),
    (1, 1, 64),
    (2, 4, 256),
    (3, 16, 1024),
    (4, 2, 128),
    (6, 8, 512),
];

/// JPEG XL natural DCT64X64 order: the 8x8 LLF block first in raster order,
/// then diagonal zig-zag over the remaining coefficients.
fn natural_scan_64() -> &'static [u32] {
    static SCAN: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    SCAN.get_or_init(|| {
        const SIDE: usize = 64;
        const LLF: usize = 8;
        let mut out = Vec::with_capacity(SIDE * SIDE);
        for y in 0..LLF {
            for x in 0..LLF {
                out.push((y * SIDE + x) as u32);
            }
        }
        for sum in 0..=(2 * (SIDE - 1)) {
            let lo = sum.saturating_sub(SIDE - 1);
            let hi = sum.min(SIDE - 1);
            if sum % 2 == 0 {
                for x in lo..=hi {
                    let y = sum - x;
                    if x >= LLF || y >= LLF {
                        out.push((y * SIDE + x) as u32);
                    }
                }
            } else {
                for x in (lo..=hi).rev() {
                    let y = sum - x;
                    if x >= LLF || y >= LLF {
                        out.push((y * SIDE + x) as u32);
                    }
                }
            }
        }
        debug_assert_eq!(out.len(), SIDE * SIDE);
        out
    })
}

/// JPEG XL natural order for DCT64X32/DCT32X64 in their shared normalized
/// 32-row x 64-column coefficient layout. The first 4x8 entries are LLF.
fn natural_scan_64x32() -> &'static [u32] {
    static SCAN: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    SCAN.get_or_init(|| {
        let mut out = vec![0u32; 2048];
        let mut cur = 32usize;
        for diagonal in 0..64usize {
            for j in 0..=diagonal {
                let (mut x, mut y) = (j, diagonal - j);
                if diagonal & 1 != 0 {
                    std::mem::swap(&mut x, &mut y);
                }
                if y & 1 == 0 {
                    y >>= 1;
                    let position = if x < 8 && y < 4 {
                        y * 8 + x
                    } else {
                        let position = cur;
                        cur += 1;
                        position
                    };
                    out[position] = (y * 64 + x) as u32;
                }
            }
        }
        for ip in (1..64usize).rev() {
            let diagonal = ip - 1;
            for j in 0..=diagonal {
                let (mut x, mut y) = (63 - (diagonal - j), 63 - j);
                if diagonal & 1 != 0 {
                    std::mem::swap(&mut x, &mut y);
                }
                if y & 1 == 0 {
                    y >>= 1;
                    out[cur] = (y * 64 + x) as u32;
                    cur += 1;
                }
            }
        }
        debug_assert_eq!(cur, 2048);
        out
    })
}

/// Natural scan position of every raw coefficient index, for one order group —
/// the inverse of jixel's `K_COEFF_ORDER_*` tables (which map scan position to
/// raw index). This is the space permutations are expressed in.
pub(crate) fn natural_position_lut(size: usize) -> Vec<u32> {
    use crate::ac_context::{
        K_COEFF_ORDER_8X8, K_COEFF_ORDER_16X8, K_COEFF_ORDER_16X16, K_COEFF_ORDER_32X16,
        K_COEFF_ORDER_32X32,
    };
    let mut lut = vec![0u32; size];
    let mut set = |scan_pos: usize, raw: usize| lut[raw] = scan_pos as u32;
    match size {
        64 => K_COEFF_ORDER_8X8
            .iter()
            .enumerate()
            .for_each(|(k, &r)| set(k, r as usize)),
        128 => K_COEFF_ORDER_16X8
            .iter()
            .enumerate()
            .for_each(|(k, &r)| set(k, r as usize)),
        256 => K_COEFF_ORDER_16X16
            .iter()
            .enumerate()
            .for_each(|(k, &r)| set(k, r as usize)),
        512 => K_COEFF_ORDER_32X16
            .iter()
            .enumerate()
            .for_each(|(k, &r)| set(k, r as usize)),
        1024 => K_COEFF_ORDER_32X32
            .iter()
            .enumerate()
            .for_each(|(k, &r)| set(k, r as usize)),
        2048 => natural_scan_64x32()
            .iter()
            .enumerate()
            .for_each(|(k, &r)| set(k, r as usize)),
        4096 => natural_scan_64()
            .iter()
            .enumerate()
            .for_each(|(k, &r)| set(k, r as usize)),
        _ => unreachable!("no natural order table for size {size}"),
    }
    lut
}

/// Scan position (natural coding order) of every raw coefficient index for a
/// `width x height` coefficient block; tall orientations are the transpose of
/// the canonical wide table. Used by the rate estimators to price the token
/// walk: the coder emits a token for every scan position up to the last
/// nonzero, so late sparse coefficients drag `visited zeros` cost with them.
pub(crate) fn scan_pos_lut(width: usize, height: usize) -> &'static [u32] {
    type ShapeLuts = Vec<((usize, usize), Vec<u32>)>;
    static LUTS: std::sync::OnceLock<ShapeLuts> = std::sync::OnceLock::new();
    let luts = LUTS.get_or_init(|| {
        let shapes = [
            (8, 8),
            (16, 8),
            (8, 16),
            (16, 16),
            (32, 16),
            (16, 32),
            (32, 32),
            (64, 32),
            (32, 64),
            (64, 64),
        ];
        shapes
            .iter()
            .map(|&(w, h)| {
                let size = w * h;
                let lut = if w >= h {
                    natural_position_lut(size)
                } else {
                    let c = natural_position_lut(size);
                    let mut t = vec![0u32; size];
                    for y in 0..h {
                        for x in 0..w {
                            t[y * w + x] = c[x * h + y];
                        }
                    }
                    t
                };
                ((w, h), lut)
            })
            .collect()
    });
    &luts
        .iter()
        .find(|(shape, _)| *shape == (width, height))
        .expect("scan lut shape")
        .1
}

/// Per-frame coefficient orders. `orders[i]` holds, for the group described by
/// `ORDER_SPECS[i]`, three scan tables (one per channel) mapping scan position
/// to raw coefficient index — the same convention as `K_COEFF_ORDER_*`.
pub(crate) struct CoeffOrders {
    /// Bit `order_index` set means that group is signalled and must be used.
    pub(crate) used_mask: u16,
    pub(crate) orders: [[Vec<u32>; 3]; ORDER_SPECS.len()],
}

impl CoeffOrders {
    /// All groups on their natural order, nothing signalled.
    pub(crate) fn natural() -> Self {
        Self {
            used_mask: 0,
            orders: std::array::from_fn(|i| {
                let (_, _, size) = ORDER_SPECS[i];
                let natural: Vec<u32> = match size {
                    64 => crate::ac_context::K_COEFF_ORDER_8X8
                        .iter()
                        .map(|&r| r as u32)
                        .collect(),
                    128 => crate::ac_context::K_COEFF_ORDER_16X8
                        .iter()
                        .map(|&r| r as u32)
                        .collect(),
                    256 => crate::ac_context::K_COEFF_ORDER_16X16
                        .iter()
                        .map(|&r| r as u32)
                        .collect(),
                    512 => crate::ac_context::K_COEFF_ORDER_32X16
                        .iter()
                        .map(|&r| r as u32)
                        .collect(),
                    1024 => crate::ac_context::K_COEFF_ORDER_32X32
                        .iter()
                        .map(|&r| r as u32)
                        .collect(),
                    _ => unreachable!(),
                };
                [natural.clone(), natural.clone(), natural]
            }),
        }
    }
}

/// Write the `used_orders` field and, when any group is signalled, the shared
/// entropy code plus every group's permutation tokens.
pub(crate) fn write_coeff_orders(
    orders: &CoeffOrders,
    huffman_pool: &mut Vec<crate::entropy::HuffmanNode>,
    w: &mut crate::bit_writer::BitWriter,
) {
    // U32 selector 3 = raw u(13).
    w.write(2, 3);
    w.write(13, u64::from(orders.used_mask));
    if orders.used_mask == 0 {
        return;
    }
    let mut tokens: Vec<Token> = Vec::new();
    for (i, &(order_index, llf, size)) in ORDER_SPECS.iter().enumerate() {
        if orders.used_mask & (1 << order_index) == 0 {
            continue;
        }
        let natural_pos = natural_position_lut(size);
        for channel_order in &orders.orders[i] {
            debug_assert_eq!(channel_order.len(), size);
            // Into natural-scan space, so the default order is the identity.
            let zigzag: Vec<u32> = channel_order
                .iter()
                .map(|&raw| natural_pos[raw as usize])
                .collect();
            tokenize_permutation(&zigzag, llf, &mut tokens);
        }
    }
    let code = crate::entropy::optimize_entropy_code(&tokens, PERMUTATION_CONTEXTS, huffman_pool);
    let code_ref = code.as_ref();
    w.write(1, 0); // no lz77 for the permutation stream
    crate::entropy::write_entropy_code(&code_ref, huffman_pool, w);
    for t in &tokens {
        crate::entropy::write_token(*t, &code_ref, w);
    }
}

/// Slot in [`ORDER_SPECS`] for a strategy's order group, or `None` if the
/// strategy maps to a group jixel never emits.
#[inline]
pub(crate) fn order_slot_of(strategy_code: u8) -> Option<usize> {
    let group = crate::ac_context::K_STRATEGY_ORDER[strategy_code as usize] as usize;
    ORDER_SPECS.iter().position(|&(g, _, _)| g == group)
}

/// Per-frame tally of how often each raw coefficient index is nonzero, keyed by
/// order-group slot and channel. Accumulated by the first AC pass; the derived
/// scan puts the most-often-nonzero positions first so the coding loop's walk
/// ends sooner.
pub(crate) struct OrderStats {
    counts: Vec<[Vec<u32>; 3]>,
    blocks: Vec<u32>,
}

impl OrderStats {
    pub(crate) fn new() -> Self {
        Self {
            counts: ORDER_SPECS
                .iter()
                .map(|&(_, _, size)| [vec![0u32; size], vec![0u32; size], vec![0u32; size]])
                .collect(),
            blocks: vec![0u32; ORDER_SPECS.len()],
        }
    }

    #[inline]
    pub(crate) fn tally(&mut self, slot: usize, channel: usize, raw_index: usize) {
        self.counts[slot][channel][raw_index] += 1;
    }

    #[inline]
    pub(crate) fn tally_block(&mut self, slot: usize) {
        self.blocks[slot] += 1;
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for (dst, src) in self.counts.iter_mut().zip(other.counts.iter()) {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                for (a, &b) in d.iter_mut().zip(s.iter()) {
                    *a += b;
                }
            }
        }
        for (d, &s) in self.blocks.iter_mut().zip(other.blocks.iter()) {
            *d += s;
        }
    }

    /// Blocks seen in an order group; groups with too few are left on the
    /// natural scan, since their statistics cannot justify a permutation.
    pub(crate) fn blocks_in(&self, slot: usize) -> u32 {
        self.blocks[slot]
    }

    /// Expected coefficient tokens per block under `order`.
    ///
    /// The coding loop stops at the last nonzero, so the walk visits position `i`
    /// exactly when some nonzero lies at or after it. Summing that probability
    /// over positions gives the expected walk length. Positions are treated as
    /// independent — an approximation, but enough to rank two scans.
    fn expected_walk(&self, slot: usize, channel: usize, order: &[u32], llf: usize) -> f64 {
        let blocks = self.blocks[slot].max(1) as f64;
        let counts = &self.counts[slot][channel];
        let mut none_at_or_after = 1.0f64;
        let mut expected = 0.0f64;
        for &raw in order[llf..].iter().rev() {
            expected += 1.0 - none_at_or_after;
            let p = (counts[raw as usize] as f64 / blocks).clamp(0.0, 1.0);
            none_at_or_after *= 1.0 - p;
        }
        expected
    }

    /// Derive the scan for one group/channel: LLF entries stay put, the rest are
    /// sorted by descending nonzero frequency (ties broken by natural order, so
    /// the result is deterministic and stays close to natural).
    fn derive(&self, slot: usize, channel: usize, natural: &[u32], llf: usize) -> Vec<u32> {
        let counts = &self.counts[slot][channel];
        let mut rest: Vec<u32> = natural[llf..].to_vec();
        rest.sort_by(|&a, &b| {
            counts[b as usize]
                .cmp(&counts[a as usize])
                .then_with(|| a.cmp(&b))
        });
        let mut out = Vec::with_capacity(natural.len());
        out.extend_from_slice(&natural[..llf]);
        out.extend_from_slice(&rest);
        out
    }
}

/// Minimum blocks in an order group before its statistics are trusted.
const MIN_BLOCKS_FOR_ORDER: u32 = 64;

const BITS_PER_SAVED_TOKEN: f64 = 0.2;

/// How much the predicted saving must exceed the predicted permutation cost.
const GATE_MARGIN: f64 = 1.5;

/// Bits one permutation token costs to signal. Deliberately pessimistic: the
/// Lehmer stream is entropy-coded, so this over-charges, which is the safe
/// direction for a gate that has to protect small low-rate images.
const BITS_PER_PERMUTATION_TOKEN: f64 = 5.0;

/// Build the frame's coefficient orders from first-pass statistics.
pub(crate) fn derive_orders(stats: &OrderStats) -> CoeffOrders {
    let mut out = CoeffOrders::natural();
    for (slot, &(order_index, llf, _size)) in ORDER_SPECS.iter().enumerate() {
        let blocks = stats.blocks_in(slot);
        if blocks < MIN_BLOCKS_FOR_ORDER {
            continue;
        }
        let mut derived: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::new());
        let mut saved_bits = 0.0f64;
        let mut cost_bits = 0.0f64;
        let mut any = false;
        for channel in 0..3 {
            let natural = out.orders[slot][channel].clone();
            let candidate = stats.derive(slot, channel, &natural, llf);
            let gain = stats.expected_walk(slot, channel, &natural, llf)
                - stats.expected_walk(slot, channel, &candidate, llf);
            let mut tokens = Vec::new();
            let natural_pos = natural_position_lut(natural.len());
            let zigzag: Vec<u32> = candidate
                .iter()
                .map(|&raw| natural_pos[raw as usize])
                .collect();
            tokenize_permutation(&zigzag, llf, &mut tokens);
            let channel_saved = gain * f64::from(blocks) * BITS_PER_SAVED_TOKEN;
            let channel_cost = tokens.len() as f64 * BITS_PER_PERMUTATION_TOKEN;
            if gain > 0.0 && channel_saved > channel_cost * GATE_MARGIN {
                any = true;
                saved_bits += channel_saved;
                cost_bits += channel_cost;
                derived[channel] = candidate;
            } else {
                derived[channel] = natural;
            }
        }
        if any && saved_bits > cost_bits * GATE_MARGIN {
            out.used_mask |= 1 << order_index;
            out.orders[slot] = derived;
        }
    }
    out
}

impl CoeffOrders {
    /// Scan table (scan position -> raw coefficient index) for a strategy and
    /// channel. Hoist this out of per-coefficient loops so the lookup stays a
    /// single indexed load, exactly like the static `K_COEFF_ORDER_*` tables it
    /// replaces.
    #[inline]
    pub(crate) fn scan_for(&self, strategy_code: u8, channel: usize) -> &[u32] {
        match order_slot_of(strategy_code) {
            Some(slot) => &self.orders[slot][channel],
            None => match crate::ac_context::K_STRATEGY_ORDER[strategy_code as usize] {
                7 => natural_scan_64(),
                8 => natural_scan_64x32(),
                order => unreachable!("unsupported natural-only order {order}"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct64x32_natural_scan_is_a_permutation_with_llf_first() {
        let scan = natural_scan_64x32();
        assert_eq!(scan.len(), 2048);
        let mut sorted = scan.to_vec();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &v)| i as u32 == v));
        for y in 0..4 {
            for x in 0..8 {
                assert_eq!(scan[y * 8 + x], (y * 64 + x) as u32);
            }
        }
    }

    #[test]
    fn identity_permutation_lehmer_is_all_zeros() {
        for n in [8usize, 64, 256] {
            let perm: Vec<u32> = (0..n as u32).collect();
            let mut code = vec![0u32; n];
            compute_lehmer_code(&perm, &mut code);
            assert!(code.iter().all(|&c| c == 0), "n={n}: {code:?}");
        }
    }

    #[test]
    fn lehmer_code_round_trips() {
        // Deterministic shuffles of a few sizes.
        let mut state = 12345u32;
        for n in [4usize, 16, 64, 128] {
            let mut perm: Vec<u32> = (0..n as u32).collect();
            for i in (1..n).rev() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                perm.swap(i, (state >> 8) as usize % (i + 1));
            }
            let mut code = vec![0u32; n];
            compute_lehmer_code(&perm, &mut code);
            let mut back = vec![0u32; n];
            decode_lehmer_code(&code, &mut back);
            assert_eq!(perm, back, "n={n}");
        }
    }

    /// The natural order must cost a single token: that is what makes signaling
    /// an order group free when its statistics say the default is already best.
    #[test]
    fn natural_order_tokenizes_to_one_token() {
        let perm: Vec<u32> = (0..64).collect();
        let mut tokens = Vec::new();
        tokenize_permutation(&perm, 1, &mut tokens);
        assert_eq!(tokens.len(), 1, "{tokens:?}");
        assert_eq!(tokens[0].value, 0, "trimmed length must be 0");
    }

    /// libjxl's `HybridUintConfig(0, 0, 0)`: token 0 for 0, else
    /// `1 + floor(log2(v))`, clamped to 7.
    #[test]
    fn coeff_order_context_matches_libjxl() {
        assert_eq!(coeff_order_context(0), 0);
        assert_eq!(coeff_order_context(1), 1);
        assert_eq!(coeff_order_context(2), 2);
        assert_eq!(coeff_order_context(3), 2);
        assert_eq!(coeff_order_context(4), 3);
        assert_eq!(coeff_order_context(7), 3);
        assert_eq!(coeff_order_context(8), 4);
        assert_eq!(coeff_order_context(1 << 6), 7);
        assert_eq!(coeff_order_context(1 << 20), 7, "must clamp");
    }

    /// A one-swap permutation must produce exactly one nonzero Lehmer entry, at
    /// the earlier of the two swapped positions.
    #[test]
    fn single_swap_has_one_nonzero_lehmer_entry() {
        let mut perm: Vec<u32> = (0..16).collect();
        perm.swap(3, 4);
        let mut code = vec![0u32; 16];
        compute_lehmer_code(&perm, &mut code);
        assert_eq!(code[3], 1);
        assert!(
            code.iter().enumerate().all(|(i, &c)| i == 3 || c == 0),
            "{code:?}"
        );
    }
}
