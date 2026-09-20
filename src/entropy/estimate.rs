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
//! Bit estimate of AC token streams under a built entropy code (the plain
//! per-pass code): the pricing both the AC context-plan arms and the
//! rate-model reconciliation log rely on.

use crate::adaptive_quant::dirty_log2f;
use crate::entropy::{OwnedEntropyCode, Token};

pub(crate) fn estimate_ac_plain_bits<'a, I>(streams: I, code: &OwnedEntropyCode) -> u64
where
    I: IntoIterator<Item = &'a [Token]>,
{
    estimate_ac_plain_bits_shared(&mut streams.into_iter(), code)
}

// Dispatch once per stream; keep the cost table and token loops shared.
fn estimate_ac_plain_bits_shared(
    streams: &mut dyn Iterator<Item = &[Token]>,
    code: &OwnedEntropyCode,
) -> u64 {
    if !code.use_prefix_code {
        // rANS approaches the histogram entropy: a symbol with normalized
        // frequency `f` out of `ANS_TAB_SIZE` costs -log2(f / ANS_TAB_SIZE).
        let cost: Vec<Vec<f32>> = code
            .ans_histograms
            .iter()
            .map(|histogram| {
                histogram
                    .freqs
                    .iter()
                    .map(|&f| {
                        if f == 0 {
                            crate::entropy::ANS_LOG_TAB_SIZE as f32
                        } else {
                            crate::entropy::ANS_LOG_TAB_SIZE as f32 - dirty_log2f(f as f32)
                        }
                    })
                    .collect()
            })
            .collect();
        let mut bits = 0.0f64;
        for tokens in streams {
            for t in tokens {
                let cl = code.context_map[t.context as usize] as usize;
                let (sym, nbits, _) =
                    crate::entropy::uint_encode_with_config(t.value, code.hybrid_uint_configs[cl]);
                bits += cost[cl][sym as usize] as f64 + nbits as f64;
            }
        }
        return bits as u64;
    }
    let mut bits: u64 = 0;
    for tokens in streams {
        for t in tokens {
            let cl = code.context_map[t.context as usize] as usize;
            let (sym, nbits, _) =
                crate::entropy::uint_encode_with_config(t.value, code.hybrid_uint_configs[cl]);
            bits += code.prefix_codes[cl].depths[sym as usize] as u64 + nbits as u64;
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::estimate_ac_plain_bits;
    use crate::ac_context::K_NUM_AC_CONTEXTS;
    use crate::entropy::{Token, optimize_entropy_code_ac};

    /// A skewed AC stream is coded with rANS, which beats the Huffman floor of
    /// 1 bit/symbol. The plain estimate must reflect that, otherwise the LZ77
    /// comparison in `enc_frame` sees an inflated plain cost and switches to a
    /// prefix-coded LZ77 bundle that is actually larger.
    #[test]
    fn plain_estimate_prices_the_ans_path_below_the_huffman_floor() {
        // 99% zeros in one context: entropy is ~0.08 bit/symbol, Huffman 1.0.
        let tokens: Vec<Token> = (0..20_000)
            .map(|i| Token {
                context: 0,
                value: u32::from(i % 100 == 0),
            })
            .collect();
        let mut scratch = crate::coder_scratch::CoderScratch::default();
        let code = optimize_entropy_code_ac(&tokens, K_NUM_AC_CONTEXTS, &mut scratch.huffman_pool);
        assert!(
            !code.use_prefix_code,
            "expected the rANS path for this stream"
        );

        let estimate = estimate_ac_plain_bits(std::iter::once(tokens.as_slice()), &code);
        let huffman_floor = tokens.len() as u64;
        assert!(
            estimate < huffman_floor / 2,
            "ANS estimate {estimate} should be far below the {huffman_floor}-bit Huffman floor"
        );
    }
}
