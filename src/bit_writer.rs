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

pub(crate) struct BitWriter {
    storage: Vec<u8>,
    /// Pending low-order bits not yet committed to `storage`.
    accumulator: u64,
    accumulator_bits: usize,
    bits_written: usize,
}

impl BitWriter {
    /// Upper bound on `n_bits` per `write` call. Mirrors libjxl's
    /// `kMaxBitsPerCall`; writes that cross the accumulator boundary spill once.
    pub(crate) const MAX_BITS_PER_CALL: usize = 56;

    pub(crate) fn new() -> Self {
        Self {
            storage: Vec::new(),
            accumulator: 0,
            accumulator_bits: 0,
            bits_written: 0,
        }
    }

    pub(crate) fn bits_written(&self) -> usize {
        self.bits_written
    }

    /// Write `n_bits` low-order bits of `bits`. Requires `n_bits <= 56` and
    /// upper bits of `bits` to be zero.
    pub(crate) fn write(&mut self, n_bits: usize, bits: u64) {
        debug_assert!(n_bits <= Self::MAX_BITS_PER_CALL);
        debug_assert!((n_bits == 0 && bits == 0) || (n_bits != 0 && (bits >> n_bits) == 0));
        if n_bits == 0 {
            return;
        }

        let room = 64 - self.accumulator_bits;
        self.accumulator |= bits << self.accumulator_bits;
        if n_bits < room {
            self.accumulator_bits += n_bits;
            self.bits_written += n_bits;
            return;
        }

        self.storage
            .extend_from_slice(&self.accumulator.to_le_bytes());
        let spill = n_bits - room;
        self.accumulator = if spill == 0 { 0 } else { bits >> room };
        self.accumulator_bits = spill;
        self.bits_written += n_bits;
    }

    /// Commit every complete pending byte, retaining at most seven bits.
    #[inline]
    fn flush_full_bytes(&mut self) {
        let bytes = self.accumulator_bits / 8;
        if bytes == 0 {
            return;
        }
        let encoded = self.accumulator.to_le_bytes();
        self.storage.extend_from_slice(&encoded[..bytes]);
        if bytes == 8 {
            self.accumulator = 0;
        } else {
            self.accumulator >>= bytes * 8;
        }
        self.accumulator_bits -= bytes * 8;
    }

    /// Zero-pad to the next byte boundary.
    pub(crate) fn zero_pad_to_byte(&mut self) {
        let rem = (8 - (self.bits_written % 8)) % 8;
        if rem != 0 {
            self.write(rem, 0);
        }
        self.flush_full_bytes();
        debug_assert_eq!(self.bits_written % 8, 0);
        debug_assert_eq!(self.accumulator_bits, 0);
    }

    /// Append the contents of each writer in `others` onto self, after first
    /// padding both self and each `other` to byte boundaries.
    pub(crate) fn append_byte_aligned(&mut self, others: &mut [BitWriter]) {
        let mut extra = 0usize;
        for w in others.iter() {
            extra += w.bits_written.div_ceil(8);
        }
        if extra == 0 {
            return;
        }
        debug_assert_eq!(self.bits_written % 8, 0);

        // Byte alignment does not imply that a naturally aligned accumulator
        // has already reached 64 bits.
        self.flush_full_bytes();
        debug_assert_eq!(self.accumulator_bits, 0);
        self.storage.reserve(extra);

        for w in others.iter_mut() {
            w.zero_pad_to_byte();
        }

        for w in others.iter() {
            let n = w.bits_written / 8;
            if n > 0 {
                self.storage.extend_from_slice(&w.storage[..n]);
            }
        }
        self.bits_written += extra * 8;
    }

    /// Append `other`'s bits onto self, bit-by-bit (general, not byte-aligned).
    pub(crate) fn append(&mut self, other: &BitWriter) {
        for &data in &other.storage {
            self.write(8, data as u64);
        }

        let first = other.accumulator_bits.min(Self::MAX_BITS_PER_CALL);
        if first != 0 {
            let mask = (1u64 << first) - 1;
            self.write(first, other.accumulator & mask);
        }
        let rest = other.accumulator_bits - first;
        if rest != 0 {
            self.write(rest, other.accumulator >> first);
        }
    }

    /// Consume self and return the byte buffer. Self must be byte-aligned.
    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        debug_assert_eq!(self.bits_written % 8, 0);
        self.flush_full_bytes();
        debug_assert_eq!(self.accumulator_bits, 0);
        debug_assert_eq!(self.storage.len(), self.bits_written / 8);
        self.storage
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::BitWriter;

    #[derive(Clone, Default)]
    struct ReferenceWriter {
        bytes: Vec<u8>,
        bits_written: usize,
    }

    impl ReferenceWriter {
        fn write(&mut self, n_bits: usize, bits: u64) {
            for bit in 0..n_bits {
                let byte = self.bits_written / 8;
                let shift = self.bits_written % 8;
                if byte == self.bytes.len() {
                    self.bytes.push(0);
                }
                self.bytes[byte] |= (((bits >> bit) & 1) as u8) << shift;
                self.bits_written += 1;
            }
        }

        fn zero_pad_to_byte(&mut self) {
            let padding = (8 - self.bits_written % 8) % 8;
            self.write(padding, 0);
        }

        fn append(&mut self, other: &Self) {
            for bit in 0..other.bits_written {
                self.write(1, ((other.bytes[bit / 8] >> (bit % 8)) & 1) as u64);
            }
        }
    }

    fn snapshot(writer: &BitWriter) -> Vec<u8> {
        let mut bytes = writer.storage.clone();
        let pending = writer.accumulator_bits.div_ceil(8);
        bytes.extend_from_slice(&writer.accumulator.to_le_bytes()[..pending]);
        bytes.truncate(writer.bits_written.div_ceil(8));
        bytes
    }

    fn assert_matches(writer: &BitWriter, reference: &ReferenceWriter) {
        assert_eq!(writer.bits_written(), reference.bits_written);
        assert_eq!(snapshot(writer), reference.bytes);
    }

    fn next_random(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    fn random_writes(writer: &mut BitWriter, reference: &mut ReferenceWriter, state: &mut u64) {
        for i in 0..500 {
            let n_bits = (next_random(state) % 57) as usize;
            let raw = next_random(state);
            let bits = if n_bits == 0 {
                0
            } else {
                raw & ((1u64 << n_bits) - 1)
            };
            writer.write(n_bits, bits);
            reference.write(n_bits, bits);
            if i % 17 == 0 {
                assert_matches(writer, reference);
            }
        }
    }

    #[test]
    fn accumulator_matches_bit_reference_across_boundaries() {
        let mut state = 0x5eed_f00d_dead_beefu64;
        for _ in 0..32 {
            let mut writer = BitWriter::new();
            let mut reference = ReferenceWriter::default();
            random_writes(&mut writer, &mut reference, &mut state);
            assert_matches(&writer, &reference);

            writer.zero_pad_to_byte();
            reference.zero_pad_to_byte();
            assert_matches(&writer, &reference);
            assert_eq!(writer.into_bytes(), reference.bytes);
        }
    }

    #[test]
    fn into_bytes_flushes_naturally_aligned_pending_bits() {
        let mut writer = BitWriter::new();
        writer.write(24, 0x00ab_cdef);
        assert_eq!(writer.accumulator_bits, 24);
        assert_eq!(writer.into_bytes(), vec![0xef, 0xcd, 0xab]);
    }

    #[test]
    fn append_matches_bit_reference_when_unaligned() {
        let mut state = 0xa55a_1234_9876_0ff0u64;
        let mut writer = BitWriter::new();
        let mut reference = ReferenceWriter::default();
        let mut other = BitWriter::new();
        let mut other_reference = ReferenceWriter::default();
        random_writes(&mut writer, &mut reference, &mut state);
        random_writes(&mut other, &mut other_reference, &mut state);

        writer.append(&other);
        reference.append(&other_reference);
        assert_matches(&writer, &reference);
    }

    #[test]
    fn byte_aligned_append_matches_reference_and_pads_inputs() {
        let mut state = 0x1357_9bdf_2468_ace0u64;
        let mut writer = BitWriter::new();
        let mut reference = ReferenceWriter::default();
        random_writes(&mut writer, &mut reference, &mut state);
        writer.zero_pad_to_byte();
        reference.zero_pad_to_byte();

        let mut others: Vec<BitWriter> = (0..7).map(|_| BitWriter::new()).collect();
        let mut other_references = vec![ReferenceWriter::default(); others.len()];
        for (other, reference) in others.iter_mut().zip(other_references.iter_mut()) {
            random_writes(other, reference, &mut state);
        }

        writer.append_byte_aligned(&mut others);
        let mut expected = reference.clone();
        for other in &mut other_references {
            other.zero_pad_to_byte();
            expected.append(other);
        }
        assert_matches(&writer, &expected);
        for (other, reference) in others.iter().zip(other_references.iter()) {
            assert_matches(other, reference);
        }
    }
}
