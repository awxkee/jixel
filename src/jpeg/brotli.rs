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

//! A minimal Brotli *encoder* that only ever emits stored (uncompressed)
//! meta-blocks.

use crate::bit_writer::BitWriter;

const MAX_META_BLOCK: usize = 1 << 24;

/// Wraps `data` in a Brotli stream built entirely from stored meta-blocks.
pub(crate) fn brotli_store(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();

    // Stream header (§9.1): a single zero bit selects WBITS = 16. The window
    // size is irrelevant for stored blocks since no backward references exist.
    w.write(1, 0);

    let mut offset = 0usize;
    while offset < data.len() {
        let len = (data.len() - offset).min(MAX_META_BLOCK);
        let chunk = &data[offset..offset + len];

        // ISLAST = 0: a final empty meta-block terminates the stream instead,
        // which keeps the loop uniform.
        w.write(1, 0);

        // MNIBBLES (§9.2): 0/1/2 select 4/5/6 nibbles of MLEN-1.
        let mlen_minus_1 = (len - 1) as u64;
        let nibbles: u32 = if mlen_minus_1 < (1 << 16) {
            4
        } else if mlen_minus_1 < (1 << 20) {
            5
        } else {
            6
        };
        w.write(2, (nibbles - 4) as u64);
        w.write(nibbles as usize * 4, mlen_minus_1);

        // ISUNCOMPRESSED. Only present when ISLAST is 0, which is always here.
        w.write(1, 1);

        // The literal bytes are byte-aligned.
        w.zero_pad_to_byte();
        for &b in chunk {
            w.write(8, b as u64);
        }

        offset += len;
    }

    // Terminating empty meta-block: ISLAST = 1, ISLASTEMPTY = 1.
    w.write(1, 1);
    w.write(1, 1);
    w.zero_pad_to_byte();

    w.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes through the system `brotli` binary, the only way to be sure the
    /// framing is conformant rather than merely self-consistent. Skipped when
    /// `brotli` is not installed.
    fn roundtrip(data: &[u8]) {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let encoded = brotli_store(data);
        let Ok(mut child) = Command::new("brotli")
            .args(["-d", "-c"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        else {
            return;
        };
        child.stdin.take().unwrap().write_all(&encoded).unwrap();
        let Ok(out) = child.wait_with_output() else {
            return;
        };
        assert!(
            out.status.success(),
            "brotli rejected our stream ({} bytes in): {}",
            data.len(),
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            out.stdout, data,
            "brotli decoded our stream to different bytes"
        );
    }

    #[test]
    fn stores_empty() {
        roundtrip(&[]);
    }

    #[test]
    fn stores_short() {
        roundtrip(b"Exif\0\0MM\0*");
    }

    #[test]
    fn stores_incompressible() {
        // A simple LCG stands in for random data without pulling in a crate.
        let mut state = 0x12345678u32;
        let data: Vec<u8> = (0..70_000)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect();
        roundtrip(&data);
    }

    #[test]
    fn stores_across_nibble_widths() {
        // Exercises the 4- and 5-nibble MLEN encodings.
        for len in [1usize, 0xFFFF, 0x1_0000, 0x1_0001] {
            roundtrip(&vec![0xABu8; len]);
        }
    }
}
