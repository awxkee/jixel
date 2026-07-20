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
use super::ans::{ANS_LOG_TAB_SIZE, AnsEncSymbolInfo};
use super::prefix_code::{ALPHABET_SIZE, PrefixCode};
use super::token::{HybridUintConfig, Token, uint_encode_with_config};
use crate::adaptive_quant::dirty_log2f;

pub(crate) struct EntropyCode<'a> {
    pub(crate) context_map: &'a [u8],
    pub(crate) num_contexts: usize,
    pub(crate) prefix_codes: &'a [PrefixCode],
    pub(crate) hybrid_uint_configs: &'a [HybridUintConfig],
    #[allow(unused)]
    pub(crate) num_prefix_codes: usize,
    /// Original (pre-cluster) context map. None for static codes.
    pub(crate) orig_context_map: Option<&'a [u8]>,
    pub(crate) orig_num_contexts: usize,
    /// When false, the bundle is encoded with rANS using the fields below.
    pub(crate) use_prefix_code: bool,
    pub(crate) ans_freqs: &'a [Vec<u16>],
    pub(crate) ans_symbols: &'a [Vec<AnsEncSymbolInfo>],
}

/// Owned entropy code: heap-allocated context_map and prefix_codes,
/// suitable for runtime-built codes.
pub(crate) struct OwnedEntropyCode {
    pub(crate) context_map: Vec<u8>,
    pub(crate) prefix_codes: Vec<PrefixCode>,
    pub(crate) hybrid_uint_configs: Vec<HybridUintConfig>,
    /// Pre-cluster context map, if clustering was applied.
    pub(crate) orig_context_map: Option<Vec<u8>>,
    pub(crate) orig_num_contexts: usize,
    /// rANS selection + tables. Empty / true when the prefix path is used.
    pub(crate) use_prefix_code: bool,
    pub(crate) ans_freqs: Vec<Vec<u16>>,
    pub(crate) ans_symbols: Vec<Vec<AnsEncSymbolInfo>>,
}

impl OwnedEntropyCode {
    pub(crate) fn as_ref(&self) -> EntropyCode<'_> {
        EntropyCode {
            context_map: &self.context_map,
            num_contexts: self.context_map.len(),
            prefix_codes: &self.prefix_codes,
            hybrid_uint_configs: &self.hybrid_uint_configs,
            num_prefix_codes: self.prefix_codes.len(),
            orig_context_map: self.orig_context_map.as_deref(),
            orig_num_contexts: self.orig_num_contexts,
            use_prefix_code: self.use_prefix_code,
            ans_freqs: &self.ans_freqs,
            ans_symbols: &self.ans_symbols,
        }
    }
}

pub(crate) struct FrozenTokenPrices {
    context_map: Vec<u8>,
    configs: Vec<HybridUintConfig>,
    symbol_bits: Vec<[f32; ALPHABET_SIZE]>,
}

impl FrozenTokenPrices {
    pub(crate) fn new(code: &OwnedEntropyCode) -> Self {
        const UNSEEN_SYMBOL_BITS: f32 = 15.0;
        let mut symbol_bits =
            vec![[UNSEEN_SYMBOL_BITS; ALPHABET_SIZE]; code.hybrid_uint_configs.len()];
        if code.use_prefix_code {
            for (bits, prefix) in symbol_bits.iter_mut().zip(&code.prefix_codes) {
                for (price, &depth) in bits.iter_mut().zip(&prefix.depths) {
                    if depth != 0 {
                        *price = if prefix.single_symbol {
                            0.0
                        } else {
                            depth as f32
                        };
                    }
                }
            }
        } else {
            for (bits, freqs) in symbol_bits.iter_mut().zip(&code.ans_freqs) {
                for (price, &freq) in bits.iter_mut().zip(freqs) {
                    if freq != 0 {
                        *price = ANS_LOG_TAB_SIZE as f32 - dirty_log2f(freq as f32);
                    }
                }
            }
        }
        Self {
            context_map: code.context_map.clone(),
            configs: code.hybrid_uint_configs.clone(),
            symbol_bits,
        }
    }

    #[inline]
    pub(crate) fn token_bits(&self, token: Token) -> f32 {
        let cluster = self.context_map[token.context as usize] as usize;
        let (symbol, extra_bits, _) = uint_encode_with_config(token.value, self.configs[cluster]);
        self.symbol_bits[cluster][symbol as usize] + extra_bits as f32
    }
}
