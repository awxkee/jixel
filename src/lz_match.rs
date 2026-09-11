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

pub(crate) type MatchKernel<T = Token> = fn(&[T], &[T]) -> usize;

#[allow(unreachable_code)]
pub(crate) fn selected() -> MatchKernel {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |a, b| unsafe { crate::neon::match_len_neon(a, b) };
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |a, b| unsafe { crate::avx::match_len_avx2(a, b) };
    }
    #[cfg(all(target_arch = "x86_64", feature = "sse"))]
    return |a, b| unsafe { crate::sse::match_len_sse2(a, b) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::match_len_wasm;
    match_len_scalar
}

#[allow(unreachable_code)]
pub(crate) fn selected_compact() -> MatchKernel<CompactToken> {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |a, b| unsafe { crate::neon::match_len_compact_neon(a, b) };
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |a, b| unsafe { crate::avx::match_len_compact_avx2(a, b) };
    }
    #[cfg(all(target_arch = "x86_64", feature = "sse"))]
    return |a, b| unsafe { crate::sse::match_len_compact_sse2(a, b) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::match_len_compact_wasm;
    match_len_compact_scalar
}

#[inline]
pub(crate) fn match_len_scalar(a: &[Token], b: &[Token]) -> usize {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.value != y.value)
        .unwrap_or(a.len().min(b.len()))
}

#[inline]
pub(crate) fn match_len_compact_scalar(a: &[CompactToken], b: &[CompactToken]) -> usize {
    a.iter()
        .zip(b)
        .position(|(x, y)| !x.same_value(*y))
        .unwrap_or(a.len().min(b.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(kernel: MatchKernel) {
        assert_eq!(std::mem::size_of::<Token>(), 8);
        for n in 0..65 {
            let a: Vec<_> = (0..n)
                .map(|i| Token::new(i as u32 % 7, i as u32 * 37))
                .collect();
            assert_eq!(kernel(&a, &a), n);
            for offset in 0..=n {
                assert_eq!(kernel(&a[offset..], &a[offset..]), n - offset);
                for mismatch in offset..n {
                    for field in 0..2 {
                        let mut b = a.clone();
                        if field == 0 {
                            b[mismatch].context ^= 1;
                        } else {
                            b[mismatch].value ^= 1;
                        }
                        // Contexts are ignored (an LZ77 copy reproduces
                        // values only); a value change ends the match.
                        let expected = if field == 0 {
                            n - offset
                        } else {
                            mismatch - offset
                        };
                        assert_eq!(kernel(&a[offset..], &b[offset..]), expected);
                    }
                }
            }
        }
        for period in 1..17 {
            let tokens: Vec<_> = (0..4097)
                .map(|i| Token::new((i % period) as u32, 17))
                .collect();
            assert_eq!(kernel(&tokens, &tokens[period..]), tokens.len() - period);
        }
    }

    #[test]
    fn compact_kernels_match_at_every_lane_tail_and_overlap() {
        for kernel in [selected_compact(), match_len_compact_scalar] {
            for n in 0..65 {
                let a: Vec<_> = (0..n)
                    .map(|i| {
                        CompactToken::try_new(
                            i as u32 % 7,
                            (i as u32 * 137) ^ CompactToken::MAX_VALUE,
                        )
                        .unwrap()
                    })
                    .collect();
                assert_eq!(kernel(&a, &a[..n / 2]), n / 2);
                assert_eq!(kernel(&a[..n / 2], &a), n / 2);
                for offset in 0..=n {
                    assert_eq!(kernel(&a[offset..], &a[offset..]), n - offset);
                    for mismatch in offset..n {
                        for field in 0..2 {
                            let mut b = a.clone();
                            let t = b[mismatch].unpack();
                            b[mismatch] = CompactToken::try_new(
                                t.context ^ u32::from(field == 0),
                                t.value ^ u32::from(field == 1),
                            )
                            .unwrap();
                            let expected = if field == 0 {
                                n - offset
                            } else {
                                mismatch - offset
                            };
                            assert_eq!(kernel(&a[offset..], &b[offset..]), expected);
                        }
                    }
                }
            }
            for period in 1..17 {
                let tokens: Vec<_> = (0..4097)
                    .map(|i| CompactToken::try_new((i % period) as u32, 17).unwrap())
                    .collect();
                assert_eq!(kernel(&tokens, &tokens[period..]), tokens.len() - period);
            }
        }
    }

    #[test]
    fn selected_matches_scalar_at_every_lane_and_tail() {
        check(selected());
    }
}

#[cfg(test)]
mod value_only_tests {
    use super::*;

    /// The SIMD kernels must agree with the scalar ones and compare values
    /// only: runs of equal contexts with differing values must stop, runs of
    /// equal values with differing contexts must continue.
    #[test]
    fn kernels_match_values_only() {
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..200 {
            let n = 1 + (next() % 70) as usize;
            let a: Vec<Token> = (0..n)
                .map(|i| {
                    Token::new(
                        next() % 3,
                        if trial % 2 == 0 {
                            i as u32 % 4
                        } else {
                            next() % 2
                        },
                    )
                })
                .collect();
            let mut b = a.clone();
            for t in b.iter_mut() {
                t.context = next() % 3;
            }
            let cut = (next() % (n as u32 + 1)) as usize;
            if cut < n {
                b[cut].value ^= 1;
            }
            let expected = a
                .iter()
                .zip(&b)
                .position(|(x, y)| x.value != y.value)
                .unwrap_or(n);
            assert_eq!(match_len_scalar(&a, &b), expected, "scalar trial {trial}");
            assert_eq!(selected()(&a, &b), expected, "simd trial {trial}");
            let ca: Vec<CompactToken> = a
                .iter()
                .map(|t| CompactToken::try_new(t.context, t.value).unwrap())
                .collect();
            let cb: Vec<CompactToken> = b
                .iter()
                .map(|t| CompactToken::try_new(t.context, t.value).unwrap())
                .collect();
            assert_eq!(
                match_len_compact_scalar(&ca, &cb),
                expected,
                "compact scalar trial {trial}"
            );
            assert_eq!(
                selected_compact()(&ca, &cb),
                expected,
                "compact simd trial {trial}"
            );
        }
    }
}
