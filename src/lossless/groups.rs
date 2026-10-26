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

//! Bounded group compression scratch and parallel section writing.

use super::lz77::{
    ConstantTail, LZ77_MIN_SYMBOL, LzToken, RunLzWriter, build_lz_pixel_code,
    write_local_tree_lz77, write_lz_section, write_lz_section_tail,
};
use super::predictor::PREDICTOR_GRADIENT;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{Token, uint_encode};
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::{WpParams, write_wp_header};
use std::sync::{Condvar, Mutex};

pub(super) const SLOW_DEEP_LZ_MAX_THREADS: usize = 8;

#[inline]
pub(super) fn group_lz_threads(speed: crate::Speed, pool: &ThreadPool) -> usize {
    if speed.effort().lz_deep {
        pool.num_threads().min(SLOW_DEEP_LZ_MAX_THREADS)
    } else {
        pool.num_threads()
    }
}

/// Fixed set of reusable deep-LZ lookup tables for one group-compression
/// phase. Unlike worker-owned scratch, these buffers are dropped as soon as
/// the phase finishes and cannot spread across more than the configured lanes.
pub(super) struct DeepLzScratchPool {
    available: Mutex<Vec<Vec<u32>>>,
    activity: Condvar,
}

impl DeepLzScratchPool {
    pub(super) fn new(slots: usize) -> Self {
        Self {
            available: Mutex::new((0..slots.max(1)).map(|_| Vec::new()).collect()),
            activity: Condvar::new(),
        }
    }

    pub(super) fn with_depth<T>(&self, f: impl FnOnce(&mut Vec<u32>) -> T) -> T {
        let mut guard = self.acquire();
        f(guard.depth.as_mut().unwrap())
    }

    // Keep lock/wait setup independent of each callback and return type.
    fn acquire(&self) -> DeepLzScratchGuard<'_> {
        let depth = {
            let mut available = self.available.lock().unwrap();
            while available.is_empty() {
                available = self.activity.wait(available).unwrap();
            }
            available.pop().unwrap()
        };

        DeepLzScratchGuard {
            owner: self,
            depth: Some(depth),
        }
    }
}

struct DeepLzScratchGuard<'a> {
    owner: &'a DeepLzScratchPool,
    depth: Option<Vec<u32>>,
}

impl Drop for DeepLzScratchGuard<'_> {
    fn drop(&mut self) {
        let depth = self.depth.take().unwrap();
        self.owner.available.lock().unwrap().push(depth);
        self.owner.activity.notify_one();
    }
}

/// Encode one single-channel modular group section from precomputed
/// gradient-predictor residual tokens (token contexts are ignored): a local
/// single-leaf Gradient tree, LZ77 run collapse, and the rANS-capable pixel
/// entropy code shared with the lossless path. Used by the lossy extra-channel
/// (alpha) path, where near-constant planes otherwise pay the 1 bit/pixel
/// prefix-code floor. The length alphabet starts above the largest literal
/// symbol, as for lossless input of that depth. Returns false (writing
/// nothing) when no such start fits the alphabet (huge residuals, e.g.
/// raw-float alpha bits).
pub(crate) fn write_single_channel_lz77_section(
    tokens: &[Token],
    scratch: &mut CoderScratch,
    w: &mut BitWriter,
) -> bool {
    let max_literal = tokens
        .iter()
        .map(|t| uint_encode(t.value).0)
        .max()
        .unwrap_or(0);
    let min_symbol = LZ77_MIN_SYMBOL.max(max_literal + 1);
    if min_symbol > super::lossless_min_symbol(18) {
        return false;
    }
    let mut runs = RunLzWriter::with_capacity(tokens.len());
    for t in tokens {
        runs.push(Token::new(0, t.value));
    }
    let lz = runs.finish();
    let code = build_lz_pixel_code(
        false,
        std::iter::once(lz.as_slice()),
        1,
        min_symbol,
        true,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    // GroupHeader: use_global_tree=0, wp all_default=1, 0 transforms.
    w.write(1, 0);
    w.write(1, 1);
    w.write(2, 0);
    write_local_tree_lz77(
        &[PREDICTOR_GRADIENT],
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        w,
    );
    write_lz_section(&lz, 1, &code, min_symbol, w);
    true
}

/// Independent streams share frozen entropy tables but retain their own ANS
/// state. Bound simultaneous staging memory as well as the worker count.
/// `tails`: one `ConstantTail` per group after its tokens, or empty.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_lz_groups(
    groups: &[Vec<LzToken>],
    tails: &[ConstantTail],
    code: &crate::entropy::EntropyCode<'_>,
    distance_context: u32,
    min_symbol: u32,
    wp_params: WpParams,
    pool: &ThreadPool,
    sections: &mut [BitWriter],
) {
    write_lz_groups_with_header(
        groups,
        tails,
        code,
        distance_context,
        min_symbol,
        pool,
        sections,
        |_, section| {
            section.write(1, 1);
            write_wp_header(wp_params, section);
            section.write(2, 0);
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_lz_groups_with_header(
    groups: &[Vec<LzToken>],
    tails: &[ConstantTail],
    code: &crate::entropy::EntropyCode<'_>,
    distance_context: u32,
    min_symbol: u32,
    pool: &ThreadPool,
    sections: &mut [BitWriter],
    write_header: impl Fn(usize, &mut BitWriter) + Sync,
) {
    debug_assert_eq!(groups.len(), sections.len());
    debug_assert!(tails.is_empty() || tails.len() == groups.len());
    let max_staging = if code.use_prefix_code {
        0
    } else {
        groups
            .iter()
            .map(|g| g.len().saturating_mul(8))
            .max()
            .unwrap_or(0)
    };
    #[allow(clippy::manual_clamp)]
    let lanes = (32 * 1024 * 1024 / max_staging.max(1))
        .max(1)
        .min(SLOW_DEEP_LZ_MAX_THREADS)
        .min(pool.num_threads())
        .min(sections.len());
    let write_group = |i: usize, section: &mut BitWriter| {
        write_header(i, section);
        let tail = tails.get(i).copied().unwrap_or(ConstantTail::NONE);
        write_lz_section_tail(
            &groups[i],
            tail,
            distance_context,
            code,
            min_symbol,
            section,
        );
        section.zero_pad_to_byte();
    };
    if lanes <= 1 {
        for (i, section) in sections.iter_mut().enumerate() {
            write_group(i, section);
        }
        return;
    }
    // The caller's entropy scratch is borrowed by `code`. This lightweight
    // scratch supplies the caller lane; section writers need no worker tables.
    let mut writer_scratch = CoderScratch::lossless();
    pool.steal_for_each_mut_with_threads(&mut writer_scratch, sections, lanes, |i, section, _| {
        write_group(i, section);
    });
}
