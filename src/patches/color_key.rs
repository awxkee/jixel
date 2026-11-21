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
use std::collections::HashMap;

/// Three signed samples packed into 21-bit biased fields (planes may hold
/// negative values, e.g. after a color transform); values outside ±2^20
/// alias, which only weakens the background vote.
const GLYPH_KEY_BIAS: i64 = 1 << 20;
const GLYPH_KEY_MASK: u64 = (1 << 21) - 1;

#[inline]
pub(super) fn color_key(c0: i32, c1: i32, c2: i32) -> u64 {
    let f = |v: i32| ((v as i64 + GLYPH_KEY_BIAS) as u64) & GLYPH_KEY_MASK;
    f(c0) << 42 | f(c1) << 21 | f(c2)
}

#[inline]
pub(super) fn round_to_multiple(v: i32, k: i32) -> i32 {
    let k = k.unsigned_abs();
    let r = ((v.unsigned_abs() + k / 2) / k * k) as i32;
    if v < 0 { -r } else { r }
}

#[inline]
pub(super) fn key_color(key: u64) -> [i32; 3] {
    let f = |shift: u32| ((key >> shift) & GLYPH_KEY_MASK) as i64 - GLYPH_KEY_BIAS;
    [f(42) as i32, f(21) as i32, f(0) as i32]
}

/// Hasher for maps keyed by one `u64` (a content hash or a packed color).
/// A multiply and a fold spread the key over the table's index and tag bits;
/// SipHash on these keys was 5% of a large photo's wall time in the region
/// planner alone, which counts every pixel's color. Callers must not depend
/// on the iteration order (they look keys up or sort their groups).
#[derive(Default)]
pub(crate) struct ContentHasher(u64);

impl std::hash::Hasher for ContentHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("content hash keys are u64");
    }

    #[inline]
    fn write_u64(&mut self, key: u64) {
        let mixed = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = mixed ^ (mixed >> 32);
    }
}

pub(crate) type ContentHashMap<V> = HashMap<u64, V, std::hash::BuildHasherDefault<ContentHasher>>;

/// Ties go to the smallest key, as in the per-pixel vote.
pub(super) fn most_frequent_color(keys: &mut [u64]) -> u64 {
    keys.sort_unstable();
    let (mut best, mut best_run) = (keys[0], 0);
    for run in keys.chunk_by(|a, b| a == b) {
        if run.len() > best_run {
            best = run[0];
            best_run = run.len();
        }
    }
    best
}
