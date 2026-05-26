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
    bits_written: usize,
}

impl BitWriter {
    /// Upper bound on `n_bits` per `write` call. Mirrors libjxl's
    /// kMaxBitsPerCall — the running byte (up to 7 valid bits) plus the new
    /// bits must fit in 64 bits with room for the next zero-init byte.
    pub(crate) const MAX_BITS_PER_CALL: usize = 56;

    pub(crate) fn new() -> Self {
        Self {
            storage: Vec::new(),
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
        debug_assert!(n_bits == 64 || (bits >> n_bits) == 0);

        let byte_pos = self.bits_written / 8;
        let bit_in_byte = self.bits_written % 8;

        // Ensure room for an 8-byte read-modify-write at byte_pos, plus the
        // invariant that the byte at logical-end is zero (so the partial byte
        // mid-stream has zero in its unused high bits).
        let needed = byte_pos + 8;
        if self.storage.len() < needed {
            self.storage.resize(needed, 0);
        }

        let shifted = bits << bit_in_byte;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.storage[byte_pos..byte_pos + 8]);
        let mut v = u64::from_le_bytes(buf);
        debug_assert!(
            v >> bit_in_byte == 0,
            "BitWriter invariant violated at byte {}",
            byte_pos
        );
        v |= shifted;
        self.storage[byte_pos..byte_pos + 8].copy_from_slice(&v.to_le_bytes());

        self.bits_written += n_bits;
    }

    /// Zero-pad to the next byte boundary.
    pub(crate) fn zero_pad_to_byte(&mut self) {
        let rem = (8 - (self.bits_written % 8)) % 8;
        if rem != 0 {
            self.write(rem, 0);
        }
        debug_assert_eq!(self.bits_written % 8, 0);
    }

    /// Append the contents of each writer in `others` onto self, after first
    /// padding both self and each `other` to byte boundaries.
    pub(crate) fn append_byte_aligned(&mut self, others: &mut [BitWriter]) {
        let mut extra = 0usize;
        for w in others.iter() {
            extra += (w.bits_written + 7) / 8;
        }
        if extra == 0 {
            return;
        }
        debug_assert_eq!(self.bits_written % 8, 0);

        for w in others.iter_mut() {
            w.zero_pad_to_byte();
        }

        let mut pos = self.bits_written / 8;
        // +1 trailing zero to keep "byte past logical end is zero" invariant.
        let new_len = pos + extra + 1;
        if self.storage.len() < new_len {
            self.storage.resize(new_len, 0);
        }
        for w in others.iter() {
            let n = w.bits_written / 8;
            if n > 0 {
                self.storage[pos..pos + n].copy_from_slice(&w.storage[..n]);
                pos += n;
            }
        }
        self.bits_written += extra * 8;
    }

    /// Append `other`'s bits onto self, bit-by-bit (general, not byte-aligned).
    pub(crate) fn append(&mut self, other: &BitWriter) {
        let full_bytes = other.bits_written / 8;
        let trailing_bits = other.bits_written % 8;
        for i in 0..full_bytes {
            self.write(8, other.storage[i] as u64);
        }
        if trailing_bits > 0 {
            let last = other.storage[full_bytes] as u64;
            let mask = (1u64 << trailing_bits) - 1;
            self.write(trailing_bits, last & mask);
        }
    }

    /// Consume self and return the byte buffer. Self must be byte-aligned.
    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        debug_assert_eq!(self.bits_written % 8, 0);
        self.storage.truncate(self.bits_written / 8);
        self.storage
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}
