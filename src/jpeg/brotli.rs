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

//! A small Brotli *encoder* for JPEG reconstruction data and metadata boxes:
//! greedy LZ77 with one literal, command and distance prefix code per
//! meta-block, falling back to stored meta-blocks when that is smaller.

use crate::bit_writer::BitWriter;
use crate::entropy::{
    HuffmanNode, convert_bit_depths_to_symbols, create_huffman_tree, write_brotli_prefix_code,
};
use crate::util::EncodeError;

const MAX_META_BLOCK: usize = 1 << 24;

/// Optional Brotli encoder used for metadata and JPEG reconstruction data.
pub trait BrotliCompression: Send + Sync {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, EncodeError>;
}

impl std::fmt::Debug for dyn BrotliCompression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn BrotliCompression")
    }
}

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

pub(crate) fn compress(
    data: &[u8],
    compressor: Option<&dyn BrotliCompression>,
) -> Result<Vec<u8>, EncodeError> {
    match compressor {
        Some(compressor) => compressor.compress(data),
        None => Ok(brotli_compress(data)),
    }
}

/// log2 of the window: distances stay below `(1 << WINDOW_BITS) - 16`.
const WINDOW_BITS: u32 = 22;
const MAX_DISTANCE: usize = (1 << WINDOW_BITS) - 16;
const MIN_MATCH: usize = 4;
const HASH_BITS: u32 = 16;
const MAX_CHAIN: usize = 32;

const NUM_LITERALS: usize = 256;
const NUM_COMMANDS: usize = 704;
/// 16 short codes plus 48 long ones (NPOSTFIX = 0, NDIRECT = 0).
const NUM_DISTANCES: usize = 64;

/// `(extra bits, base)` per insert-length code.
static INSERT_CODES: [(u32, u32); 24] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 6),
    (1, 8),
    (2, 10),
    (2, 14),
    (3, 18),
    (3, 26),
    (4, 34),
    (4, 50),
    (5, 66),
    (5, 98),
    (6, 130),
    (7, 194),
    (8, 322),
    (9, 578),
    (10, 1090),
    (12, 2114),
    (14, 6210),
    (24, 22594),
];
/// `(extra bits, base)` per copy-length code.
static COPY_CODES: [(u32, u32); 24] = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (1, 10),
    (1, 12),
    (2, 14),
    (2, 18),
    (3, 22),
    (3, 30),
    (4, 38),
    (4, 54),
    (5, 70),
    (5, 102),
    (6, 134),
    (7, 198),
    (8, 326),
    (9, 582),
    (10, 1094),
    (24, 2118),
];

fn length_code(table: &[(u32, u32); 24], len: u32) -> usize {
    table.partition_point(|&(_, base)| base <= len) - 1
}

/// One command: `insert` literals, then (unless it ends the meta-block) a copy
/// of `copy` bytes from `distance` back.
#[derive(Clone, Copy)]
struct Command {
    insert: u32,
    copy: u32,
    distance: u32,
}

impl Command {
    /// The insert-and-copy symbol, always with an explicit distance (§5).
    fn symbol(&self) -> usize {
        let ins = length_code(&INSERT_CODES, self.insert);
        let cop = length_code(&COPY_CODES, self.copy.max(2));
        // Cells of 64 symbols by (insert range, copy range), explicit distance.
        const CELL: [[usize; 3]; 3] = [[128, 192, 384], [256, 320, 512], [448, 576, 640]];
        CELL[ins >> 3][cop >> 3] + ((ins & 7) << 3) + (cop & 7)
    }

    /// Distance symbol and its extra bits (count, value).
    fn distance_code(&self) -> (usize, u32, u32) {
        let dd = self.distance + 3;
        let nbits = 31 - dd.leading_zeros() - 1;
        let prefix = (dd >> nbits) & 1;
        let code = 16 + 2 * (nbits - 1) + prefix;
        (code as usize, nbits, dd - ((2 + prefix) << nbits))
    }
}

/// Greedy LZ77 with one step of lazy matching over `data[start..end]`; the
/// hash chains index the whole buffer so matches may reach earlier blocks.
fn find_commands(
    data: &[u8],
    start: usize,
    end: usize,
    head: &mut [u32],
    prev: &mut [u32],
) -> Vec<Command> {
    let hash = |i: usize| -> usize {
        let v = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        (v.wrapping_mul(0x1E35_A7BD) >> (32 - HASH_BITS)) as usize
    };
    // `prev` is a ring over positions; its length is a power of two at least
    // as long as any distance taken.
    let mask = prev.len() - 1;
    let insert_hash = |i: usize, head: &mut [u32], prev: &mut [u32]| {
        if i + MIN_MATCH <= data.len() {
            let h = hash(i);
            prev[i & mask] = head[h];
            head[h] = i as u32 + 1;
        }
    };
    let longest = |i: usize, head: &[u32], prev: &[u32]| -> (usize, usize) {
        if i + MIN_MATCH > end {
            return (0, 0);
        }
        let mut best = (0usize, 0usize);
        let mut cand = head[hash(i)] as usize;
        let mut chain = 0;
        while cand != 0 && chain < MAX_CHAIN {
            let pos = cand - 1;
            let distance = i - pos;
            if distance > MAX_DISTANCE || distance > mask {
                break;
            }
            let len = data[pos..]
                .iter()
                .zip(&data[i..end])
                .take_while(|(a, b)| a == b)
                .count();
            if len > best.0 {
                best = (len, distance);
            }
            cand = prev[pos & mask] as usize;
            chain += 1;
        }
        if best.0 >= MIN_MATCH { best } else { (0, 0) }
    };

    let mut commands = Vec::new();
    let mut literal_start = start;
    let mut i = start;
    while i < end {
        let (len, distance) = longest(i, head, prev);
        if len == 0 {
            insert_hash(i, head, prev);
            i += 1;
            continue;
        }
        // Lazy step: a longer match one byte later wins.
        insert_hash(i, head, prev);
        let (next_len, _) = longest(i + 1, head, prev);
        if next_len > len + 1 {
            i += 1;
            continue;
        }
        commands.push(Command {
            insert: (i - literal_start) as u32,
            copy: len as u32,
            distance: distance as u32,
        });
        for j in i + 1..i + len {
            insert_hash(j, head, prev);
        }
        i += len;
        literal_start = i;
    }
    if literal_start < end {
        commands.push(Command {
            insert: (end - literal_start) as u32,
            copy: 0,
            distance: 0,
        });
    }
    commands
}

/// A prefix code over one alphabet: depths as signaled, and the per-symbol
/// bit counts and codes as emitted. They differ for a lone symbol, which is
/// signaled as a one-symbol code and then costs no bits at all.
struct Code {
    depths: Vec<u8>,
    lengths: Vec<u8>,
    bits: Vec<u16>,
}

impl Code {
    fn new(counts: &[u32], pool: &mut Vec<HuffmanNode>) -> Self {
        let mut depths = vec![0u8; counts.len()];
        let used: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] != 0).collect();
        if let [only] = used[..] {
            depths[only] = 1;
            return Self {
                depths,
                lengths: vec![0; counts.len()],
                bits: vec![0; counts.len()],
            };
        }
        if used.len() > 1 {
            create_huffman_tree(counts, 15, &mut depths, pool);
        }
        let mut bits = vec![0u16; counts.len()];
        convert_bit_depths_to_symbols(&depths, &mut bits);
        Self {
            lengths: depths.clone(),
            depths,
            bits,
        }
    }

    #[inline]
    fn emit(&self, symbol: usize, w: &mut BitWriter) {
        w.write(
            usize::from(self.lengths[symbol]),
            u64::from(self.bits[symbol]),
        );
    }
}

/// Brotli-compresses `data`, keeping the stored form when that is smaller.
pub(crate) fn brotli_compress(data: &[u8]) -> Vec<u8> {
    let stored = brotli_store(data);
    if data.len() < 64 {
        return stored;
    }
    let mut w = BitWriter::new();
    // Stream header: WBITS = 17 + 5.
    w.write(1, 1);
    w.write(3, u64::from(WINDOW_BITS - 17));

    let mut head = vec![0u32; 1 << HASH_BITS];
    let mut prev = vec![0u32; data.len().next_power_of_two().min(1 << WINDOW_BITS)];
    let mut pool = Vec::new();

    let mut offset = 0usize;
    while offset < data.len() {
        let len = (data.len() - offset).min(MAX_META_BLOCK);
        let commands = find_commands(data, offset, offset + len, &mut head, &mut prev);

        let mut literal_counts = [0u32; NUM_LITERALS];
        let mut command_counts = [0u32; NUM_COMMANDS];
        let mut distance_counts = [0u32; NUM_DISTANCES];
        let mut pos = offset;
        for c in &commands {
            for &b in &data[pos..pos + c.insert as usize] {
                literal_counts[usize::from(b)] += 1;
            }
            command_counts[c.symbol()] += 1;
            if c.copy != 0 {
                distance_counts[c.distance_code().0] += 1;
            }
            pos += (c.insert + c.copy) as usize;
        }
        let literal_code = Code::new(&literal_counts, &mut pool);
        let command_code = Code::new(&command_counts, &mut pool);
        let distance_code = Code::new(&distance_counts, &mut pool);

        // Meta-block header: not last, MLEN, compressed.
        w.write(1, 0);
        let mlen_minus_1 = (len - 1) as u64;
        let nibbles: u32 = if mlen_minus_1 < (1 << 16) {
            4
        } else if mlen_minus_1 < (1 << 20) {
            5
        } else {
            6
        };
        w.write(2, u64::from(nibbles - 4));
        w.write(nibbles as usize * 4, mlen_minus_1);
        w.write(1, 0); // ISUNCOMPRESSED
        w.write(1, 0); // NBLTYPESL = 1
        w.write(1, 0); // NBLTYPESI = 1
        w.write(1, 0); // NBLTYPESD = 1
        w.write(2, 0); // NPOSTFIX
        w.write(4, 0); // NDIRECT
        w.write(2, 0); // literal context mode (one block type)
        w.write(1, 0); // NTREESL = 1
        w.write(1, 0); // NTREESD = 1
        write_brotli_prefix_code(&literal_code.depths, 8, &mut pool, &mut w);
        write_brotli_prefix_code(&command_code.depths, 10, &mut pool, &mut w);
        write_brotli_prefix_code(&distance_code.depths, 6, &mut pool, &mut w);

        let mut pos = offset;
        for c in &commands {
            command_code.emit(c.symbol(), &mut w);
            let (ins_bits, ins_base) = INSERT_CODES[length_code(&INSERT_CODES, c.insert)];
            w.write(ins_bits as usize, u64::from(c.insert - ins_base));
            let (copy_bits, copy_base) = COPY_CODES[length_code(&COPY_CODES, c.copy.max(2))];
            w.write(copy_bits as usize, u64::from(c.copy.max(2) - copy_base));
            for &b in &data[pos..pos + c.insert as usize] {
                literal_code.emit(usize::from(b), &mut w);
            }
            // The meta-block ends after the final literals; no distance follows.
            if c.copy != 0 {
                let (code, nbits, extra) = c.distance_code();
                distance_code.emit(code, &mut w);
                w.write(nbits as usize, u64::from(extra));
            }
            pos += (c.insert + c.copy) as usize;
        }
        offset += len;
    }
    // Terminating empty meta-block.
    w.write(1, 1);
    w.write(1, 1);
    w.zero_pad_to_byte();
    let compressed = w.into_bytes();
    if compressed.len() < stored.len() {
        compressed
    } else {
        stored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes through the system `brotli` binary, the only way to be sure the
    /// framing is conformant rather than merely self-consistent. Skipped when
    /// `brotli` is not installed.
    fn roundtrip(data: &[u8]) {
        decodes_to(&brotli_store(data), data);
        decodes_to(&brotli_compress(data), data);
    }

    fn decodes_to(encoded: &[u8], data: &[u8]) {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let Ok(mut child) = Command::new("brotli")
            .args(["-d", "-c"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        else {
            return;
        };
        // Fed from another thread: large streams would otherwise fill both
        // pipes and deadlock against the decoder's output.
        let mut stdin = child.stdin.take().unwrap();
        let input = encoded.to_vec();
        let feeder = std::thread::spawn(move || stdin.write_all(&input));
        let Ok(out) = child.wait_with_output() else {
            return;
        };
        feeder.join().unwrap().unwrap();
        assert!(
            out.status.success(),
            "brotli rejected our stream ({} bytes in): {}",
            data.len(),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.stdout == data,
            "brotli decoded our stream to different bytes ({} in, {} out)",
            data.len(),
            out.stdout.len()
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

    /// Deterministic text-like data: words from a small vocabulary.
    fn words(len: usize, seed: u32) -> Vec<u8> {
        const VOCAB: [&str; 12] = [
            "<rdf:li>",
            "exif:",
            "Sony ",
            "ILCE-6700",
            "2026:07:12 ",
            "\n   ",
            "xmp",
            "=\"",
            "</rdf:Seq>",
            "photoshop",
            "0",
            "tiff:Orientation",
        ];
        let mut state = seed;
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            out.extend_from_slice(VOCAB[(state >> 24) as usize % VOCAB.len()].as_bytes());
        }
        out.truncate(len);
        out
    }

    #[test]
    fn compresses_text_and_stays_decodable() {
        let data = words(200_000, 7);
        let compressed = brotli_compress(&data);
        assert!(
            compressed.len() * 4 < data.len(),
            "{} bytes",
            compressed.len()
        );
        roundtrip(&data);
    }

    #[test]
    fn compresses_runs_and_overlapping_copies() {
        let mut data = vec![0u8; 50_000];
        data.extend((0..50_000u32).map(|i| (i % 3) as u8));
        data.extend(b"abcabcabcabcabcabcabcabc".iter().cycle().take(10_000));
        roundtrip(&data);
    }

    #[test]
    fn compresses_every_small_length() {
        for len in 0..300 {
            roundtrip(&words(len, len as u32));
        }
    }

    #[test]
    fn compresses_across_meta_blocks() {
        // Past one 16 MiB meta-block, with matches reaching back across it.
        let mut data = words(MAX_META_BLOCK + 100_000, 3);
        let tail = data[..50_000].to_vec();
        data.extend_from_slice(&tail);
        roundtrip(&data);
    }

    #[test]
    fn falls_back_to_stored_for_noise() {
        let mut state = 0x9E37_79B9u32;
        let data: Vec<u8> = (0..10_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect();
        assert_eq!(brotli_compress(&data), brotli_store(&data));
    }

    #[test]
    fn uses_the_supplied_compressor() {
        struct Prefix;
        impl BrotliCompression for Prefix {
            fn compress(&self, data: &[u8]) -> Result<Vec<u8>, EncodeError> {
                let mut out = b"custom:".to_vec();
                out.extend_from_slice(data);
                Ok(out)
            }
        }

        assert_eq!(
            compress(b"payload", Some(&Prefix)).unwrap(),
            b"custom:payload"
        );
    }
}
