/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! Stage 4: race the pyramid against its best exact fold.
//!
//! The fold screen ([`super::fold::select_fold`]) runs first on the coarse
//! prefix alone; only when it expects a real saving does the race code the
//! frame twice, keeping fine-channel trees and tokens in a [`TreeCache`]
//! between the two. Sizes compare whole serialized frames, header, padding
//! and TOC included.

use super::cache::TreeCache;
use super::code;
use super::fold::select_fold;
use super::pyramid::Pyramid;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::thread_pool::ThreadPool;

/// Race exact decoded representations, including the frame header, padding
/// and TOC. The original size is retained for the caller's Auto-arm decision.
pub(super) fn encode(
    mut quantized: Pyramid,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    // Screen the changed coarse prefix before allocating a fine-band cache
    // or rebuilding whole-frame entropy tables. Tiny expected savings do not
    // justify an O(pixels) trial. Fast presets and DC frames also stay direct.
    let Some(folded_steps) = select_fold(&quantized, speed, scratch) else {
        let start = writer.bits_written();
        code::write_frame(&quantized, speed, pool, scratch, writer, None);
        return Some(writer.bits_written() - start);
    };
    let prefix = writer.clone();
    let mut cache = TreeCache::new(&quantized);
    let mut best = prefix.clone();
    code::write_frame(
        &quantized,
        speed,
        pool,
        scratch,
        &mut best,
        Some(&mut cache),
    );
    let original_bits = best.bits_written() - prefix.bits_written();
    for _ in 0..folded_steps {
        if !quantized.fold_last() {
            *writer = best;
            return Some(original_bits);
        }
    }
    // Even exhaustive effort serializes at most one shortlisted candidate.
    let mut candidate = prefix.clone();
    code::write_frame(
        &quantized,
        speed,
        pool,
        scratch,
        &mut candidate,
        Some(&mut cache),
    );
    if candidate.bits_written() < best.bits_written() {
        best = candidate;
    }
    *writer = best;
    Some(original_bits)
}

#[cfg(test)]
mod tests {
    use super::super::cache::TOKEN_CACHE_BYTES;
    use super::super::pyramid::PlainInput;
    use super::*;
    use crate::coding::CodingTransform;

    #[test]
    fn race_preserves_prefix_and_reports_original_size_at_every_alignment() {
        let pool = ThreadPool::new(1);
        let mut scratch = CoderScratch::default();
        for (w, h, flat) in [
            (37usize, 29usize, false),
            (513, 17, false),
            (128, 128, true),
        ] {
            let input = PlainInput {
                color: (0..3)
                    .map(|c| {
                        (0..w * h)
                            .map(|i| {
                                if flat {
                                    (12000 + c * 8000) as i32
                                } else {
                                    ((i * 523 + (i / w) * 311 + c * 997) % 65536) as i32
                                }
                            })
                            .collect()
                    })
                    .collect(),
                extra: Vec::new(),
                max: 65535.0,
                coding: CodingTransform::Rgb,
            };
            for alignment in 0..8 {
                let quantized = Pyramid::plain(
                    &input,
                    w,
                    h,
                    6.0,
                    false,
                    &crate::thread_pool::ThreadPool::new(1),
                )
                .unwrap();
                if flat {
                    // Ensure the alignment test reaches the full trial as
                    // well as the coarse screen's direct-encoding fallback.
                    assert_eq!(
                        select_fold(&quantized, crate::Speed::Slow, &mut scratch,),
                        Some(4),
                    );
                }
                let mut prefix = BitWriter::new();
                prefix.write(8, 0xa5);
                prefix.write(alignment, (1u64 << alignment) - 1);
                let prefix_bits = prefix.bits_written();
                let mut original = prefix.clone();
                code::write_frame(
                    &quantized,
                    crate::Speed::Slow,
                    &pool,
                    &mut scratch,
                    &mut original,
                    None,
                );
                let mut raced = prefix;
                let baseline_bits = encode(
                    quantized,
                    crate::Speed::Slow,
                    &pool,
                    &mut scratch,
                    &mut raced,
                )
                .unwrap();
                assert_eq!(baseline_bits, original.bits_written() - prefix_bits);
                assert!(raced.bits_written() <= original.bits_written());
                let bytes = raced.into_bytes();
                assert_eq!(bytes[0], 0xa5);
                let mask = (1u8 << alignment) - 1;
                assert_eq!(bytes[1] & mask, mask);
            }
        }
    }

    #[test]
    fn cached_local_tokens_match_fresh_encoding_after_forest_changes() {
        let (w, h) = (513, 17);
        let input = PlainInput {
            color: (0..3)
                .map(|c| {
                    (0..w * h)
                        .map(|i| ((i * 523 + (i / w) * 311 + c * 997) % 65536) as i32)
                        .collect()
                })
                .collect(),
            extra: Vec::new(),
            max: 65535.0,
            coding: CodingTransform::Rgb,
        };
        let mut quantized = Pyramid::plain(
            &input,
            w,
            h,
            6.0,
            false,
            &crate::thread_pool::ThreadPool::new(1),
        )
        .unwrap();
        let pool = ThreadPool::new(1);
        let mut scratch = CoderScratch::default();
        let mut cache = TreeCache::new(&quantized);
        code::write_frame(
            &quantized,
            crate::Speed::Slow,
            &pool,
            &mut scratch,
            &mut BitWriter::new(),
            Some(&mut cache),
        );
        assert!(cache.token_bytes > 0 && cache.token_bytes <= TOKEN_CACHE_BYTES);
        for _ in 0..4 {
            assert!(quantized.fold_last());
        }
        let mut reused = BitWriter::new();
        code::write_frame(
            &quantized,
            crate::Speed::Slow,
            &pool,
            &mut scratch,
            &mut reused,
            Some(&mut cache),
        );
        let mut fresh = BitWriter::new();
        code::write_frame(
            &quantized,
            crate::Speed::Slow,
            &pool,
            &mut scratch,
            &mut fresh,
            None,
        );
        assert_eq!(reused.into_bytes(), fresh.into_bytes());
    }
}
