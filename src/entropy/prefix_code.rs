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

pub(crate) const ALPHABET_SIZE: usize = 128;

#[derive(Clone, Copy)]
pub(crate) struct PrefixCode {
    pub(crate) depths: [u8; ALPHABET_SIZE],
    pub(crate) bits: [u16; ALPHABET_SIZE],
    pub(crate) single_symbol: bool,
}

impl PrefixCode {
    pub(crate) const fn zero() -> Self {
        Self {
            depths: [0; ALPHABET_SIZE],
            bits: [0; ALPHABET_SIZE],
            single_symbol: false,
        }
    }

    /// Recompute `single_symbol` from the current depths (one nonzero entry).
    pub(crate) fn update_single_symbol(&mut self) {
        self.single_symbol = self.depths.iter().filter(|&&d| d != 0).count() == 1;
    }
}

impl Default for PrefixCode {
    fn default() -> Self {
        Self::zero()
    }
}

/// Reverse the low `num_bits` of `bits`. JXL stores Huffman codes
/// "LSB-first emission of MSB-first canonical codes", which means after
/// building canonical codes in the standard way we have to bit-reverse.
fn reverse_bits(num_bits: u32, bits: u16) -> u16 {
    // Pre-reversed nibble LUT.
    const LUT: [u16; 16] = [
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
    ];
    let mut retval: u16 = LUT[(bits & 0xf) as usize];
    let mut bits = bits >> 4;
    let mut i = 4;
    while i < num_bits {
        retval <<= 4;
        retval |= LUT[(bits & 0xf) as usize];
        bits >>= 4;
        i += 4;
    }
    // Right-shift by (-num_bits & 0x3) to discard the padding-up-to-nibble bits.
    retval >> ((4 - (num_bits & 3)) & 3)
}

/// Compute canonical bit patterns from given bit depths.
///   depth[i] = 0 means symbol absent; depth[i] > 0 is the code length.
/// Output: bits[i] = the (bit-reversed) prefix code for symbol i.
pub(crate) fn convert_bit_depths_to_symbols(depth: &[u8], bits: &mut [u16]) {
    const MAX_BITS: usize = 16;
    let mut bl_count = [0u16; MAX_BITS];
    for &d in depth {
        bl_count[d as usize] += 1;
    }
    bl_count[0] = 0;

    let mut next_code = [0u16; MAX_BITS];
    let mut code: u32 = 0;
    for i in 1..MAX_BITS {
        code = (code + bl_count[i - 1] as u32) << 1;
        next_code[i] = code as u16;
    }

    for (bits, &depth) in bits.iter_mut().zip(depth.iter()) {
        if depth != 0 {
            let d = depth as u32;
            let raw = next_code[d as usize];
            *bits = reverse_bits(d, raw);
            next_code[d as usize] += 1;
        } else {
            *bits = 0;
        }
    }
}
