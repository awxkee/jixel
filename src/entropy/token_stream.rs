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

use super::Token;
use std::ops::Range;

/// The fine AC layout needs 14 context bits. Keeping 18 value bits covers
/// ordinary quantized coefficients; whole streams fall back to wide tokens
/// for HDR or near-lossless values outside that range.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct AcToken(u32);

impl AcToken {
    const VALUE_BITS: u32 = 18;
    const MAX_VALUE: u32 = (1 << Self::VALUE_BITS) - 1;
    const MAX_CONTEXT: u32 = (1 << (32 - Self::VALUE_BITS)) - 1;

    #[inline]
    fn unpack(self) -> Token {
        Token::new(self.0 >> Self::VALUE_BITS, self.0 & Self::MAX_VALUE)
    }
}

/// Owned AC streams are compacted at the group boundary, so wide buffers are
/// bounded by active workers instead of retained for every group in the frame.
#[derive(Clone)]
pub(crate) enum AcTokens {
    Packed(Vec<AcToken>),
    Wide(Vec<Token>),
}

impl From<Vec<Token>> for AcTokens {
    fn from(mut tokens: Vec<Token>) -> Self {
        const { assert!(crate::ac_context::K_NUM_FINE_AC_CONTEXTS <= 1 << 14) };
        let (contexts, values) = tokens.iter().fold((0, 0), |(contexts, values), t| {
            (contexts | t.context, values | t.value)
        });
        if contexts <= AcToken::MAX_CONTEXT && values <= AcToken::MAX_VALUE {
            Self::Packed(
                tokens
                    .iter()
                    .map(|t| AcToken((t.context << AcToken::VALUE_BITS) | t.value))
                    .collect(),
            )
        } else {
            tokens.shrink_to_fit();
            Self::Wide(tokens)
        }
    }
}

impl AcTokens {
    pub(crate) fn as_slice(&self) -> TokenSlice<'_> {
        match self {
            Self::Packed(tokens) => TokenSlice::Packed(tokens),
            Self::Wide(tokens) => TokenSlice::Wide(tokens),
        }
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = Token> + ExactSizeIterator + '_ {
        self.as_slice().iter()
    }

    pub(crate) fn remap_contexts(&mut self, mut remap: impl FnMut(u32) -> u32) {
        match self {
            Self::Packed(tokens) => {
                for t in tokens {
                    let context = remap(t.0 >> AcToken::VALUE_BITS);
                    assert!(context <= AcToken::MAX_CONTEXT);
                    t.0 = (context << AcToken::VALUE_BITS) | (t.0 & AcToken::MAX_VALUE);
                }
            }
            Self::Wide(tokens) => {
                for t in tokens {
                    t.context = remap(t.context);
                }
            }
        }
    }
}

/// Borrow either storage format without expanding a frame's token streams.
#[derive(Clone, Copy)]
pub(crate) enum TokenSlice<'a> {
    Wide(&'a [Token]),
    Packed(&'a [AcToken]),
}

impl<'a> From<&'a [Token]> for TokenSlice<'a> {
    fn from(tokens: &'a [Token]) -> Self {
        Self::Wide(tokens)
    }
}

impl<'a> From<&'a Vec<Token>> for TokenSlice<'a> {
    fn from(tokens: &'a Vec<Token>) -> Self {
        Self::Wide(tokens)
    }
}

impl<'a, const N: usize> From<&'a [Token; N]> for TokenSlice<'a> {
    fn from(tokens: &'a [Token; N]) -> Self {
        Self::Wide(tokens)
    }
}

impl<'a> From<&TokenSlice<'a>> for TokenSlice<'a> {
    fn from(tokens: &TokenSlice<'a>) -> Self {
        *tokens
    }
}

impl<'a> TokenSlice<'a> {
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Wide(tokens) => tokens.len(),
            Self::Packed(tokens) => tokens.len(),
        }
    }

    pub(crate) fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub(crate) fn slice(self, range: Range<usize>) -> Self {
        match self {
            Self::Wide(tokens) => Self::Wide(&tokens[range]),
            Self::Packed(tokens) => Self::Packed(&tokens[range]),
        }
    }

    pub(crate) fn split_at(self, mid: usize) -> (Self, Self) {
        (self.slice(0..mid), self.slice(mid..self.len()))
    }

    pub(crate) fn iter(self) -> TokenIter<'a> {
        match self {
            Self::Wide(tokens) => TokenIter::Wide(tokens.iter()),
            Self::Packed(tokens) => TokenIter::Packed(tokens.iter()),
        }
    }
}

impl<'a> IntoIterator for TokenSlice<'a> {
    type Item = Token;
    type IntoIter = TokenIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub(crate) enum TokenIter<'a> {
    Wide(std::slice::Iter<'a, Token>),
    Packed(std::slice::Iter<'a, AcToken>),
}

impl Iterator for TokenIter<'_> {
    type Item = Token;

    #[inline]
    fn next(&mut self) -> Option<Token> {
        match self {
            Self::Wide(tokens) => tokens.next().copied(),
            Self::Packed(tokens) => tokens.next().map(|t| t.unpack()),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            Self::Wide(tokens) => tokens.len(),
            Self::Packed(tokens) => tokens.len(),
        };
        (len, Some(len))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Token> {
        match self {
            Self::Wide(tokens) => tokens.nth(n).copied(),
            Self::Packed(tokens) => tokens.nth(n).map(|t| t.unpack()),
        }
    }

    #[inline]
    fn fold<B, F: FnMut(B, Token) -> B>(self, init: B, f: F) -> B {
        // Dispatch once per stream, preserving specialized inner loops.
        match self {
            Self::Wide(tokens) => tokens.copied().fold(init, f),
            Self::Packed(tokens) => tokens.map(|t| t.unpack()).fold(init, f),
        }
    }
}

impl DoubleEndedIterator for TokenIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Token> {
        match self {
            Self::Wide(tokens) => tokens.next_back().copied(),
            Self::Packed(tokens) => tokens.next_back().map(|t| t.unpack()),
        }
    }

    #[inline]
    fn rfold<B, F: FnMut(B, Token) -> B>(self, init: B, f: F) -> B {
        match self {
            Self::Wide(tokens) => tokens.copied().rfold(init, f),
            Self::Packed(tokens) => tokens.map(|t| t.unpack()).rfold(init, f),
        }
    }
}

impl ExactSizeIterator for TokenIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(tokens: impl IntoIterator<Item = Token>) -> Vec<(u32, u32)> {
        tokens.into_iter().map(|t| (t.context, t.value)).collect()
    }

    #[test]
    fn packed_boundaries_and_wide_fallback_preserve_all_bits() {
        let source = vec![
            Token::new(0, 0),
            Token::new(AcToken::MAX_CONTEXT, AcToken::MAX_VALUE),
            Token::new(crate::ac_context::K_NUM_FINE_AC_CONTEXTS as u32 - 1, 31),
        ];
        let packed = AcTokens::from(source.clone());
        let AcTokens::Packed(storage) = &packed else {
            panic!("AC boundary values must pack");
        };
        assert_eq!(size_of::<AcToken>(), 4);
        assert_eq!(storage.capacity(), source.len());
        assert_eq!(pairs(packed.iter()), pairs(source.iter().copied()));
        for wide in [
            Token::new(0, AcToken::MAX_VALUE + 1),
            Token::new(AcToken::MAX_CONTEXT + 1, 0),
            Token::new(u32::MAX, u32::MAX),
        ] {
            let mut source = source.clone();
            source.push(wide);
            let actual = AcTokens::from(source.clone());
            assert!(matches!(actual, AcTokens::Wide(_)));
            assert_eq!(pairs(actual.iter()), pairs(source));
        }
    }

    #[test]
    fn views_preserve_ranges_reverse_iteration_and_remapped_values() {
        for wide in [false, true] {
            let source: Vec<_> = (0..97)
                .map(|i| Token::new(i % 7, if wide && i == 96 { u32::MAX } else { i * 13 }))
                .collect();
            let mut actual = AcTokens::from(source.clone());
            let slice = actual.as_slice();
            let (left, right) = slice.split_at(19);
            assert_eq!(pairs(left), pairs(source[..19].iter().copied()));
            assert_eq!(pairs(right), pairs(source[19..].iter().copied()));
            assert_eq!(
                pairs(slice.slice(17..69).iter().rev()),
                pairs(source[17..69].iter().rev().copied())
            );
            let mut iter = slice.iter();
            assert_eq!(iter.len(), source.len());
            assert_eq!(iter.next().unwrap().value, source[0].value);
            assert_eq!(iter.next_back().unwrap().value, source[96].value);
            assert_eq!(iter.nth(12).unwrap().value, source[13].value);
            assert_eq!(iter.len(), 82);
            actual.remap_contexts(|context| 15 - context);
            assert_eq!(
                pairs(actual.iter()),
                source
                    .iter()
                    .map(|t| (15 - t.context, t.value))
                    .collect::<Vec<_>>()
            );
        }
        let empty = AcTokens::from(Vec::new());
        assert!(empty.as_slice().is_empty());
        assert!(empty.iter().next().is_none());
        assert!(empty.iter().next_back().is_none());
    }

    #[test]
    fn packed_and_mixed_streams_preserve_entropy_codes_prices_and_ans_bytes() {
        use crate::bit_writer::BitWriter;
        use crate::coder_scratch::CoderScratch;
        use crate::entropy::{
            AnsRefinement, FrozenTokenPrices, estimate_ac_plain_bits,
            optimize_entropy_code_ac_streams, refine_ans_clusters, write_ans_tokens,
            write_entropy_code, write_token,
        };
        use crate::thread_pool::ThreadPool;

        for scenario in 0..3 {
            let prefix = scenario == 1;
            let source: Vec<_> = (0..140_003)
                .map(|i| {
                    Token::new(
                        (i % 31) as u32,
                        if prefix {
                            0
                        } else if scenario == 2 && i == 140_002 {
                            u32::MAX
                        } else {
                            (i * 73 % 911) as u32
                        },
                    )
                })
                .collect();
            let packed = AcTokens::from(source.clone());
            let plain = TokenSlice::Wide(&source);
            let compact = packed.as_slice();
            let packed_first = AcTokens::from(source[..65_537].to_vec());
            let layouts = [
                vec![
                    plain.slice(0..65_537),
                    TokenSlice::Wide(&[]),
                    plain.slice(65_537..source.len()),
                ],
                vec![
                    compact.slice(0..65_537),
                    TokenSlice::Wide(&[]),
                    compact.slice(65_537..source.len()),
                ],
                vec![
                    packed_first.as_slice(),
                    TokenSlice::Wide(&[]),
                    plain.slice(65_537..source.len()),
                ],
            ];
            for threads in [1, 4] {
                let pool = ThreadPool::new_lossless(threads);
                let mut scratch = CoderScratch::lossless();
                let mut expected = None;
                for streams in &layouts {
                    let mut code = optimize_entropy_code_ac_streams(
                        streams.iter().copied(),
                        31,
                        &mut scratch.huffman_pool,
                        true,
                        crate::Speed::Slow,
                        Some(&pool),
                    );
                    // Exercise prefix serialization even when the bundle
                    // selector prefers ANS for its smaller global header.
                    if prefix {
                        code.use_prefix_code = true;
                    }
                    refine_ans_clusters(
                        &mut code,
                        streams.iter().copied(),
                        AnsRefinement::Slow,
                        &pool,
                        &mut scratch,
                    );
                    assert_eq!(code.use_prefix_code, prefix);
                    let estimate = estimate_ac_plain_bits(streams.iter().copied(), &code);
                    let prices = FrozenTokenPrices::new(&code);
                    let adaptive = prices.adaptive_bits(streams.iter().copied());
                    let mut counted = prices.adaptive_rate();
                    for &stream in streams.iter().rev() {
                        counted.add(stream);
                    }
                    assert_eq!(counted.bits().to_bits(), adaptive.to_bits());
                    let mut writer = BitWriter::new();
                    write_entropy_code(&code.as_ref(), &mut scratch.huffman_pool, &mut writer);
                    for &stream in streams {
                        if prefix {
                            stream
                                .iter()
                                .for_each(|t| write_token(t, &code.as_ref(), &mut writer));
                        } else {
                            write_ans_tokens(
                                stream,
                                &code.context_map,
                                &code.ans_symbols,
                                &code.ans_reverse_maps,
                                &code.hybrid_uint_configs,
                                &mut writer,
                            );
                        }
                    }
                    let bits = writer.bits_written();
                    writer.zero_pad_to_byte();
                    let result = (
                        code.context_map,
                        code.hybrid_uint_configs,
                        estimate,
                        adaptive.to_bits(),
                        bits,
                        writer.into_bytes(),
                    );
                    if let Some(expected) = &expected {
                        assert_eq!(&result, expected);
                    } else {
                        expected = Some(result);
                    }
                }
            }
        }
    }
}
