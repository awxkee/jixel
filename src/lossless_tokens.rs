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
use crate::entropy::{CompactToken, Token};

/// Pack while tokenizing, and promote once if a context or residual is wider.
/// The fallback copies existing tokens exactly; it never reruns prediction.
pub(super) enum RawTokens {
    Compact(Vec<CompactToken>),
    Wide(Vec<Token>),
}

impl RawTokens {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self::Compact(Vec::with_capacity(capacity))
    }

    pub(super) fn extend(&mut self, row: &[Token]) {
        match self {
            Self::Compact(tokens) => {
                if CompactToken::extend_from_tokens(tokens, row) {
                    return;
                }
            }
            Self::Wide(tokens) => {
                tokens.extend_from_slice(row);
                return;
            }
        }
        self.promote();
        match self {
            Self::Wide(tokens) => tokens.extend_from_slice(row),
            Self::Compact(_) => unreachable!("promoted token storage"),
        }
    }

    #[cfg(test)]
    fn push(&mut self, token: Token) {
        self.extend(std::slice::from_ref(&token));
    }

    #[cold]
    fn promote(&mut self) {
        let previous = std::mem::replace(self, Self::Wide(Vec::new()));
        *self = Self::Wide(previous.into_wide());
    }

    fn into_wide(self) -> Vec<Token> {
        match self {
            Self::Wide(tokens) => tokens,
            Self::Compact(tokens) => {
                let mut wide = Vec::with_capacity(tokens.capacity());
                wide.extend(tokens.into_iter().map(CompactToken::unpack));
                wide
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Compact(tokens) => tokens.is_empty(),
            Self::Wide(tokens) => tokens.is_empty(),
        }
    }
}

/// Resolve representation once per frame before the monomorphized entropy
/// and matching passes. A wide stream promotes the others without re-tokenizing.
pub(super) enum RawTokenStreams {
    Compact(Vec<Vec<CompactToken>>),
    Wide(Vec<Vec<Token>>),
}

impl RawTokenStreams {
    pub(super) fn new(streams: Vec<RawTokens>) -> Self {
        if streams.iter().all(|s| matches!(s, RawTokens::Compact(_))) {
            Self::Compact(
                streams
                    .into_iter()
                    .map(|s| match s {
                        RawTokens::Compact(tokens) => tokens,
                        RawTokens::Wide(_) => unreachable!("all streams are compact"),
                    })
                    .collect(),
            )
        } else {
            Self::Wide(streams.into_iter().map(RawTokens::into_wide).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(tokens: Vec<Token>) -> Vec<(u32, u32)> {
        tokens.into_iter().map(|t| (t.context, t.value)).collect()
    }

    #[test]
    fn compact_ranges_and_wide_promotion_preserve_every_field() {
        assert_eq!(std::mem::size_of::<CompactToken>(), 4);
        for context in [0, 1, CompactToken::MAX_CONTEXT] {
            for value in [0, 1, CompactToken::MAX_VALUE] {
                let token = CompactToken::try_new(context, value).unwrap().unpack();
                assert_eq!((token.context, token.value), (context, value));
            }
        }
        for oversized in [
            Token::new(CompactToken::MAX_CONTEXT + 1, 0),
            Token::new(0, CompactToken::MAX_VALUE + 1),
            Token::new(u32::MAX, u32::MAX),
        ] {
            assert!(CompactToken::try_new(oversized.context, oversized.value).is_none());
            for at in [0, 1, 31, 64] {
                let mut actual = RawTokens::with_capacity(65);
                let expected: Vec<_> = (0..65)
                    .map(|i| {
                        if i == at {
                            oversized
                        } else {
                            Token::new(i % 7, i * 101)
                        }
                    })
                    .collect();
                for (i, &token) in expected.iter().enumerate() {
                    actual.push(token);
                    assert_eq!(matches!(actual, RawTokens::Compact(_)), i < at as usize);
                }
                assert_eq!(pairs(actual.into_wide()), pairs(expected.clone()));
                for width in [1, 2, 31, 64, 257] {
                    let mut batched = RawTokens::with_capacity(expected.len());
                    for row in expected.chunks(width) {
                        batched.extend(row);
                    }
                    assert_eq!(pairs(batched.into_wide()), pairs(expected.clone()));
                }
            }
        }
    }

    #[test]
    fn frame_promotion_preserves_empty_streams_and_stream_order() {
        for wide_at in [None, Some(1), Some(3)] {
            let expected: Vec<Vec<_>> = [0, 17, 0, 257]
                .into_iter()
                .enumerate()
                .map(|(s, n)| {
                    (0..n)
                        .map(|i| {
                            Token::new(
                                s as u32,
                                if wide_at == Some(s) && i == n / 2 {
                                    u32::MAX
                                } else {
                                    i as u32
                                },
                            )
                        })
                        .collect()
                })
                .collect();
            let streams = expected
                .iter()
                .map(|tokens| {
                    let mut out = RawTokens::with_capacity(tokens.len());
                    for &token in tokens {
                        out.push(token);
                    }
                    out
                })
                .collect();
            match RawTokenStreams::new(streams) {
                RawTokenStreams::Compact(streams) => {
                    assert!(wide_at.is_none());
                    let unpacked: Vec<_> = streams
                        .into_iter()
                        .map(|s| pairs(s.into_iter().map(CompactToken::unpack).collect()))
                        .collect();
                    assert_eq!(
                        unpacked,
                        expected.into_iter().map(pairs).collect::<Vec<_>>()
                    );
                }
                RawTokenStreams::Wide(streams) => {
                    assert!(wide_at.is_some());
                    assert_eq!(
                        streams.into_iter().map(pairs).collect::<Vec<_>>(),
                        expected.into_iter().map(pairs).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
