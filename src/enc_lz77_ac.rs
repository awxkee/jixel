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

use crate::bit_writer::BitWriter;
use crate::entropy::{
    Histogram, OwnedEntropyCode, Token, build_huffman_codes, cluster_histograms, uint_encode,
    write_entropy_code,
};

pub(crate) const LZ77_MIN_SYMBOL: u32 = 64;
pub(crate) const LZ77_MIN_LENGTH: u32 = 3;
/// special_distance index 1 == (dx=1, dy=0) == "1 token back".
const LZ77_DIST_VALUE: u32 = 1;

#[inline]
fn lz77_length_encode(length_value: u32) -> (u32, u32, u32) {
    if length_value < 16 {
        (length_value, 0, 0)
    } else {
        let n = 31 - length_value.leading_zeros();
        let token = 16 + n - 4;
        let nbits = n;
        let bits = length_value - (1 << n);
        (token, nbits, bits)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum AcLz {
    /// A literal AC token, emitted via the normal uint_encode path.
    Lit { context: u32, value: u32 },
    /// Back-reference: copy `length_value + LZ77_MIN_LENGTH` previous tokens.
    /// The length symbol is coded on `context` (the repeated token's context),
    /// then a distance symbol on the distance context.
    Copy { context: u32, length_value: u32 },
}

/// Collapse runs of identical (context, value) AC tokens into back-references.
pub(crate) fn lz77_compress_ac(tokens: &[Token]) -> Vec<AcLz> {
    // Collapse runs of identical AC tokens into a literal + distance-1 back-ref.
    //
    // Runs are grouped by BOTH context and value (not value alone). This is the
    // conservative choice: a run never spans a point where the per-coefficient
    // context changes, so the LZ77 length symbol — which the decoder reads at the
    // position right after the run's literal head — is always read with the same
    // context it was coded on, and a run never crosses a block boundary (the
    // nonzero-count token carries a different context than the coefficients).
    //
    // Zero-valued runs are deliberately NOT collapsed. A dense 32x32 block emits
    // long stretches of zero coefficients in the high-frequency tail (its
    // context buckets are 16-wide, `k >> 4`, versus 4-wide for 16x16), and
    // back-referencing those zero runs desynced the libjxl decoder. Coding the
    // zeros as plain literals (still efficiently entropy-coded by the
    // zero-density context model) avoids that without affecting the far more
    // valuable nonzero runs.
    let n = tokens.len();
    let mut out: Vec<AcLz> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let t = tokens[i];
        out.push(AcLz::Lit {
            context: t.context,
            value: t.value,
        });
        let mut j = i + 1;
        while j < n && tokens[j].context == t.context && tokens[j].value == t.value {
            j += 1;
        }
        let run_extra = (j - i - 1) as u32; // copies after the first literal
        if run_extra >= LZ77_MIN_LENGTH && t.value != 0 {
            out.push(AcLz::Copy {
                context: t.context,
                length_value: run_extra - LZ77_MIN_LENGTH,
            });
            i = j;
        } else {
            i += 1;
        }
    }
    out
}
#[allow(dead_code)] // kept for a future (correct) LZ77 alpha path; see modular::write_ac_group_alpha
pub(crate) fn build_lz_code_no_cluster(
    streams: &[Vec<AcLz>],
    num_contexts: usize,
) -> OwnedEntropyCode {
    let distance_context = (num_contexts - 1) as u32;
    let mut histograms = vec![Histogram::new(); num_contexts];
    for stream in streams {
        for t in stream {
            match *t {
                AcLz::Lit { context, value } => {
                    let (sym, _, _) = uint_encode(value);
                    histograms[context as usize].add(sym);
                }
                AcLz::Copy {
                    context,
                    length_value,
                } => {
                    let (len_tok, _, _) = lz77_length_encode(length_value);
                    histograms[context as usize].add(LZ77_MIN_SYMBOL + len_tok);
                    histograms[distance_context as usize].add(LZ77_DIST_VALUE);
                }
            }
        }
    }
    let context_map: Vec<u8> = (0..num_contexts as u8).collect();
    let mut prefix_codes = build_huffman_codes(&histograms);

    for pc in &mut prefix_codes {
        let mut nonzero = 0usize;
        let mut idx = 0usize;
        for (i, &d) in pc.depths.iter().enumerate() {
            if d != 0 {
                nonzero += 1;
                idx = i;
                if nonzero > 1 {
                    break;
                }
            }
        }
        if nonzero == 1 {
            if idx == 0 {
                pc.depths[0] = 0;
                pc.bits[0] = 0;
            } else {
                pc.depths[0] = 1;
                pc.bits[0] = 0;
                pc.depths[idx] = 1;
                pc.bits[idx] = 1;
            }
        }
        pc.update_single_symbol();
    }
    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![crate::entropy::HybridUintConfig::DEFAULT; num_contexts],
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: true,
        ans_freqs: Vec::new(),
        ans_symbols: Vec::new(),
    }
}

/// Build per-context histograms over the LZ-compressed stream, then cluster
/// them (libjxl-tiny clusters to <= 8). `num_contexts` already includes the
/// trailing distance context. Returns an `OwnedEntropyCode` whose
/// `orig_context_map` has `num_contexts` entries (the distance context is the
/// last), so `write_entropy_code` signals it correctly.
pub(crate) fn build_ac_lz_code(streams: &[Vec<AcLz>], num_contexts: usize) -> OwnedEntropyCode {
    let distance_context = (num_contexts - 1) as u32;
    let mut histograms = vec![Histogram::new(); num_contexts];
    for stream in streams {
        for t in stream {
            match *t {
                AcLz::Lit { context, value } => {
                    let (sym, _, _) = uint_encode(value);
                    histograms[context as usize].add(sym);
                }
                AcLz::Copy {
                    context,
                    length_value,
                } => {
                    let (len_tok, _, _) = lz77_length_encode(length_value);
                    histograms[context as usize].add(LZ77_MIN_SYMBOL + len_tok);
                    histograms[distance_context as usize].add(LZ77_DIST_VALUE);
                }
            }
        }
    }

    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map);
    let prefix_codes = build_huffman_codes(&histograms);

    OwnedEntropyCode {
        context_map,
        prefix_codes,
        hybrid_uint_configs: vec![crate::entropy::HybridUintConfig::DEFAULT; histograms.len()],
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: true,
        ans_freqs: Vec::new(),
        ans_symbols: Vec::new(),
    }
}

/// Estimate the encoded size in bits of an LZ stream under `code`, including
/// the literal/length prefix-code bits and the hybrid-uint extra bits and the
/// distance symbols. Used to decide whether LZ77 is worth enabling.
pub(crate) fn estimate_ac_lz_bits(
    streams: &[Vec<AcLz>],
    code: &OwnedEntropyCode,
    num_contexts: usize,
) -> u64 {
    let distance_context = num_contexts - 1;
    let mut bits: u64 = 0;
    for stream in streams {
        for t in stream {
            match *t {
                AcLz::Lit { context, value } => {
                    let (sym, nbits, _) = uint_encode(value);
                    let cl = code.context_map[context as usize] as usize;
                    let pc = &code.prefix_codes[cl];
                    let d = if pc.single_symbol {
                        0
                    } else {
                        pc.depths[sym as usize] as u64
                    };
                    bits += d + nbits as u64;
                }
                AcLz::Copy {
                    context,
                    length_value,
                } => {
                    let (len_tok, len_nbits, _) = lz77_length_encode(length_value);
                    let sym = LZ77_MIN_SYMBOL + len_tok;
                    let cl = code.context_map[context as usize] as usize;
                    let pc = &code.prefix_codes[cl];
                    let d = if pc.single_symbol {
                        0
                    } else {
                        pc.depths[sym as usize] as u64
                    };
                    bits += d + len_nbits as u64;
                    let (dsym, dnbits, _) = uint_encode(LZ77_DIST_VALUE);
                    let dcl = code.context_map[distance_context] as usize;
                    let dpc = &code.prefix_codes[dcl];
                    let dd = if dpc.single_symbol {
                        0
                    } else {
                        dpc.depths[dsym as usize] as u64
                    };
                    bits += dd + dnbits as u64;
                }
            }
        }
    }
    bits
}

/// Estimate the encoded size in bits of the original (non-LZ) token stream
/// under a plain code built over `K_NUM_AC_CONTEXTS` contexts.
pub(crate) fn estimate_ac_plain_bits(tokens: &[Token], code: &OwnedEntropyCode) -> u64 {
    let mut bits: u64 = 0;
    for t in tokens {
        let (sym, nbits, _) = uint_encode(t.value);
        let cl = code.context_map[t.context as usize] as usize;
        bits += code.prefix_codes[cl].depths[sym as usize] as u64 + nbits as u64;
    }
    bits
}
#[inline]
pub(crate) fn write_ac_lz(
    t: AcLz,
    code: &OwnedEntropyCode,
    num_contexts: usize,
    w: &mut BitWriter,
) {
    let distance_context = num_contexts - 1;
    match t {
        AcLz::Lit { context, value } => {
            let (sym, nbits, bits) = uint_encode(value);
            let cluster = code.context_map[context as usize] as usize;
            let pc = &code.prefix_codes[cluster];
            if pc.single_symbol {
                w.write(nbits as usize, bits as u64);
            } else {
                let d = pc.depths[sym as usize] as usize;
                let data = (pc.bits[sym as usize] as u64) | ((bits as u64) << d);
                w.write(d + nbits as usize, data);
            }
        }
        AcLz::Copy {
            context,
            length_value,
        } => {
            let (len_tok, len_nbits, len_bits) = lz77_length_encode(length_value);
            let sym = LZ77_MIN_SYMBOL + len_tok;
            let pcluster = code.context_map[context as usize] as usize;
            let pc = &code.prefix_codes[pcluster];
            if pc.single_symbol {
                w.write(len_nbits as usize, len_bits as u64);
            } else {
                let d = pc.depths[sym as usize] as usize;
                debug_assert!(d > 0, "LZ77 AC length symbol {} unrepresented", sym);
                let data = (pc.bits[sym as usize] as u64) | ((len_bits as u64) << d);
                w.write(d + len_nbits as usize, data);
            }

            // Distance symbol: value LZ77_DIST_VALUE, coded on the distance ctx.
            let (dsym, dnbits, dbits) = uint_encode(LZ77_DIST_VALUE);
            let dcluster = code.context_map[distance_context] as usize;
            let dpc = &code.prefix_codes[dcluster];
            if dpc.single_symbol {
                w.write(dnbits as usize, dbits as u64);
            } else {
                let dd = dpc.depths[dsym as usize] as usize;
                let ddata = (dpc.bits[dsym as usize] as u64) | ((dbits as u64) << dd);
                w.write(dd + dnbits as usize, ddata);
            }
        }
    }
}

/// Write the AC LZ77 sub-bundle header (mirrors `write_lz77_header` in the
/// lossless path): enabled bit + min_symbol (64) + min_length (3) +
/// length_uint_config(split=4, msb=0, lsb=0). Then the entropy code.
pub(crate) fn write_ac_lz_header_and_code(code: &OwnedEntropyCode, w: &mut BitWriter) {
    w.write(1, 1); // lz77 enabled
    // min_symbol: U32(Val(224), Val(512), Val(4096), BitsOffset(15,8)).
    // For 64: selector 3 -> "11" + 15 bits of (64 - 8) = 56.
    w.write(2, 0b11);
    w.write(15, (LZ77_MIN_SYMBOL - 8) as u64);
    // min_length: U32(Val(3), ...). For 3: selector 0.
    w.write(2, 0b00);
    // length_uint_config: split_exp=4 (4 bits), msb=0 (3 bits), lsb=0 (3 bits).
    w.write(4, 4);
    w.write(3, 0);
    w.write(3, 0);
    // The entropy code: its context map already carries the trailing distance
    // context as the last (orig_num_contexts-th) entry.
    write_entropy_code(&code.as_ref(), w);
}
