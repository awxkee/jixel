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

//! Lossy modular coding: an image, or one extra channel, as a quantized
//! Squeeze pyramid in modular streams instead of VarDCT.
//!
//! A lossy modular frame runs these stages, one module each:
//!
//! 1. [`calibrate`]: map the caller's distance to this arm's quantizer
//!    distance, so the modular and VarDCT arms land near the same quality.
//! 2. [`pyramid`]: put the source planes on the frame's integer [`layout`]
//!    (the XYB lattice, or the samples themselves through YCoCg), run the
//!    forward Squeeze, and divide every channel by its [`quantize`] step.
//!    At ExtraSlow, `polish` then revisits the XYB frame's luma and chroma residuals
//!    against the exact SSIMULACRA2 and ColorVideoVDP arbiters.
//! 3. [`fold`]: screen folding the pyramid's exact coarse tail back into
//!    full-scale averages: the same decoded image, another entropy
//!    representation.
//! 4. [`race`]: code the pyramid, and the screen's fold when it found one;
//!    keep the smaller frame.
//! 5. [`code`]: code one pyramid as a complete frame. A single-group frame is
//!    one section ([`single`]). Otherwise [`tags`] assigns every channel crop
//!    to a stream slot, [`trees`] learns one MA subtree per tag, [`forest`]
//!    merges them under the global tree, [`tokens`] tokenizes every stream
//!    and [`write`] emits the sections and TOC. [`cache`] carries unchanged
//!    trees and tokens from one race candidate to the next.
//!
//! [`alpha`] codes a lossy extra channel inside a VarDCT frame with the same
//! squeeze, quantizer and forest.

mod alpha;
#[cfg(test)]
mod alpha_tests;
mod cache;
mod calibrate;
mod closed_loop;
mod code;
mod fold;
mod forest;
mod layout;
#[cfg(feature = "extraslow")]
mod polish;
mod pyramid;
#[cfg(test)]
mod pyramid_tests;
mod quantize;
mod race;
mod single;
mod tags;
mod tokens;
mod trees;
mod write;

pub(crate) use alpha::ExtraSqueeze;
pub(crate) use calibrate::{calibrated_distance, plain_distance};
pub(crate) use pyramid::PlainInput;

use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::entropy::f_log2;
use crate::image::Image3F;
use crate::thread_pool::ThreadPool;
use pyramid::Pyramid;

/// Lossy modular frame of an XYB codestream. `distance` is the arm's
/// calibrated quantizer distance ([`calibrated_distance`] and the Auto gate)
/// and `screen` whether the image is screen content (its step rows);
/// `patches` are dictionary entries over an atlas frame the caller has
/// already written.
/// Returns the frame size in bits before the fold race, for the Auto gate,
/// or None when the image is too small to squeeze.
pub(crate) fn encode_xyb_frame(
    xyb: &Image3F,
    distance: f32,
    screen: bool,
    patches: &[crate::patches::PatchReference],
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    let mut pyramid = Pyramid::xyb(xyb, distance, screen, pool)?;
    pyramid.patches = patches.to_vec();
    pyramid.group_shift = group_shift(speed);
    #[cfg(feature = "extraslow")]
    if speed.effort().lm_residual_polish() {
        polish::polish_frame(&mut pyramid, xyb, distance, screen, pool, scratch);
    }
    race::encode(pyramid, speed, pool, scratch, writer)
}

/// Lossy modular frame of a non-XYB codestream: the integer planes in
/// `input` at the plain-layout schedule ([`plain_distance`]). Returns the
/// frame size in bits before the fold race, for the Auto gate, or None when
/// the image is too small to squeeze. `black_plate` says the extra channel is
/// CMYK's K, stored inverted, whose no-ink value a lossy palette must keep.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_plain_frame(
    input: &PlainInput,
    xsize: usize,
    ysize: usize,
    distance: f32,
    black_plate: bool,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> Option<usize> {
    // A lossless extra channel (alpha, say) codes the same samples squeezed
    // or not: unsqueezed wins on cutouts and stickers (to -27 %), the squeeze
    // on smooth fades (one vignette +7..+14 % without it). The color pyramid
    // is the same either way, so from Slow the extras alone decide, coded
    // under an all-zero gray plane at Fastest (crops and tile samples break
    // the fades and pick wrong); faster tiers take unsqueezed, -9.7 % over
    // eight RGBA images and quicker to code.
    let unsqueeze = input.extra.last().is_some_and(|&(_, d)| d == 0.0)
        && (speed < crate::Speed::Slow || {
            let probe = PlainInput {
                color: vec![vec![0; xsize * ysize]],
                extra: input.extra.clone(),
                max: input.max,
                coding: input.coding,
            };
            let bits = |unsqueeze: bool, scratch: &mut CoderScratch| {
                let pyramid = Pyramid::plain(&probe, xsize, ysize, distance, unsqueeze, pool)?;
                let mut w = BitWriter::new();
                code::write_frame(&pyramid, crate::Speed::Fastest, pool, scratch, &mut w, None);
                Some(w.bits_written())
            };
            match (bits(false, scratch), bits(true, scratch)) {
                (Some(squeezed), Some(plain)) => plain < squeezed,
                _ => false,
            }
        });
    let mut pyramid = Pyramid::plain(input, xsize, ysize, distance, unsqueeze, pool)?;
    pyramid.group_shift = group_shift(speed);
    // From Slow, a lossy extra channel (alpha) also races as a lossy
    // palette, decided by estimated rates: order-0 context entropy of the
    // squeezed channel and of the palette indices. Calibrated against the
    // full double encode over 32 RGBA images x 5 distances: 8/160 choices
    // differ, all near ties (+0.05 % bytes); probe encodes of the extras
    // cost 1.69x the arm-free frame, the estimate 1.18x.
    let Some(levels) = (speed >= crate::Speed::Slow)
        .then(|| lossy_palette_levels(input, &pyramid, xsize, ysize, black_plate))
        .flatten()
    else {
        return race::encode(pyramid, speed, pool, scratch, writer);
    };
    let palette_bits = palette_index_bits(&levels, xsize, ysize);
    let take =
        palette_bits < PALETTE_RATE_MARGIN * squeezed_extra_bits(&pyramid, input.color.len());
    let tiled = repeated_share(&input.extra[0].0, xsize) > TILED_SHARE;
    if !take && !tiled {
        return race::encode(pyramid, speed, pool, scratch, writer);
    }
    let levels = PlainInput {
        color: input.color.clone(),
        extra: vec![(levels, 0.0)],
        max: input.max,
        coding: input.coding,
    };
    let Some(mut palette) = Pyramid::plain_palette(&levels, xsize, ysize, distance, pool) else {
        return race::encode(pyramid, speed, pool, scratch, writer);
    };
    palette.group_shift = pyramid.group_shift;
    if !tiled {
        return race::encode(palette, speed, pool, scratch, writer);
    }
    let mut squeezed = writer.clone();
    let squeezed_bits = race::encode(pyramid, speed, pool, scratch, &mut squeezed);
    let palette_bits = race::encode(palette, speed, pool, scratch, writer);
    if squeezed.bits_written() < writer.bits_written() {
        *writer = squeezed;
        return squeezed_bits;
    }
    palette_bits
}

/// Tiled alpha races both codings exactly instead: LZ77 copies its repeats,
/// which neither order-0 estimate sees, so either coding can win by 2-4x (a
/// tiled background at 8 to 16 bits, both ways). Over 18 deep and 18 8-bit
/// RGBA images the tile's [`repeated_share`] is 0.79-0.93 and the next 0.71;
/// a race taken in vain only costs time.
const TILED_SHARE: f64 = 0.75;

/// Share of the plane's samples in non-constant 16-sample windows (raster
/// order) that LZ77 could copy from more than two rows back in runs of at
/// least 128. Nearer repeats (a straight edge continuing) and short ones
/// (antialiased edges alike) are the estimates' contexts' to see.
fn repeated_share(plane: &[i32], xsize: usize) -> f64 {
    const WINDOW: usize = 16;
    const MIN_RUN: usize = 128;
    const HASH_BITS: u32 = 18;
    let hash = |w: &[i32]| {
        let h = w.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| {
            (h ^ u64::from(v as u32)).wrapping_mul(0x0100_0000_01b3)
        });
        (h >> (64 - HASH_BITS)) as usize
    };
    let mut last = vec![u32::MAX; 1 << HASH_BITS];
    let (mut varied, mut repeated) = (0usize, 0usize);
    let mut i = 0;
    while i + WINDOW <= plane.len() {
        let w = &plane[i..i + WINDOW];
        if w.iter().all(|&v| v == w[0]) {
            i += 1;
            continue;
        }
        let h = hash(w);
        let j = last[h] as usize;
        last[h] = i as u32;
        if j < i && plane[j..j + WINDOW] == *w {
            let mut run = WINDOW;
            while i + run < plane.len() && plane[j + run] == plane[i + run] {
                run += 1;
            }
            varied += run;
            if run >= MIN_RUN && i - j > 2 * xsize {
                repeated += run;
            }
            i += run;
        } else {
            varied += 1;
            i += 1;
        }
    }
    if varied == 0 {
        0.0
    } else {
        repeated as f64 / varied as f64
    }
}

/// The palette is taken while its estimated index bits stay under this many
/// times the squeezed channel's: the estimate rates the indices about 1.45x
/// their learned-tree coding relative to the squeeze's.
const PALETTE_RATE_MARGIN: f64 = 1.45;

/// Hybrid-uint token and raw bits of a packed residual (split exponent 4,
/// one mantissa bit), as the frame's entropy code splits it.
fn residual_token(e: u32) -> (u32, u32) {
    if e < 16 {
        (e, 0)
    } else {
        let n = 31 - e.leading_zeros();
        (16 + (n - 4) * 2 + ((e >> (n - 1)) & 1), n - 1)
    }
}

/// Order-0 bits of `(context, residual)` samples: per-context token entropy,
/// raw bits and a small table cost per context.
fn context_entropy_bits(samples: impl Iterator<Item = (usize, i32)>, contexts: usize) -> f64 {
    let mut hist = vec![std::collections::HashMap::<u32, u32>::new(); contexts];
    let mut bits = 0f64;
    for (c, v) in samples {
        let (token, raw) = residual_token(crate::entropy::pack_signed(v));
        *hist[c].entry(token).or_default() += 1;
        bits += f64::from(raw);
    }
    for h in &hist {
        let n: u32 = h.values().sum();
        if n == 0 {
            continue;
        }
        bits += 12.0 + 6.0 * h.len() as f64;
        for &k in h.values() {
            bits -= f64::from(k) * f_log2(f64::from(k) / f64::from(n));
        }
    }
    bits
}

/// The decoder's left, top and top-left neighbours of `(x, y)` in a plane.
fn neighbours(data: &[i32], w: usize, x: usize, y: usize) -> (i32, i32, i32) {
    let left = if x > 0 {
        data[y * w + x - 1]
    } else if y > 0 {
        data[(y - 1) * w + x]
    } else {
        0
    };
    let top = if y > 0 { data[(y - 1) * w + x] } else { left };
    let top_left = if x > 0 && y > 0 {
        data[(y - 1) * w + x - 1]
    } else {
        left
    };
    (left, top, top_left)
}

/// Estimated bits of a plain pyramid's one squeezed extra channel (the last
/// squeezed plane): its coarsest channel through the clamped gradient, its
/// residual channels as stored under |W| > 0 / |N| > 0 contexts, like the
/// fixed extra-channel context tree.
fn squeezed_extra_bits(p: &Pyramid, num_color: usize) -> f64 {
    let planes = num_color + 1;
    let n_pre: usize = p
        .steps
        .iter()
        .filter(|s| !s.in_place)
        .map(|s| s.num_c)
        .sum();
    let channels = &p.channels[..p.channels.len() - n_pre];
    let samples = channels
        .iter()
        .enumerate()
        .filter(|&(i, _)| i % planes == num_color)
        .flat_map(|(i, ch)| {
            (0..ch.w * ch.h).map(move |k| {
                let (x, y) = (k % ch.w, k / ch.w);
                let (left, top, top_left) = neighbours(&ch.data, ch.w, x, y);
                let v = ch.data[k];
                if i < planes {
                    let pred = (left + top - top_left).clamp(left.min(top), left.max(top));
                    (4, v - pred)
                } else {
                    (usize::from(left != 0) * 2 + usize::from(top != 0), v)
                }
            })
        });
    context_entropy_bits(samples, 5)
}

/// Estimated bits of the palette indices of `levels`: each index against
/// its left neighbour under W == N / N == NW contexts.
pub(crate) fn palette_index_bits(levels: &[i32], xsize: usize, ysize: usize) -> f64 {
    let mut entries = levels.to_vec();
    entries.sort_unstable();
    entries.dedup();
    let indices: Vec<i32> = levels
        .iter()
        .map(|v| entries.binary_search(v).unwrap_or(0) as i32)
        .collect();
    let samples = (0..xsize * ysize).map(|k| {
        let (left, top, top_left) = neighbours(&indices, xsize, k % xsize, k / xsize);
        (
            usize::from(left == top) * 2 + usize::from(top == top_left),
            indices[k] - left,
        )
    });
    context_entropy_bits(samples, 4)
}

/// The lossy palette levels of a plain frame's one lossy extra channel
/// (alpha): the fewest levels within part of the squeezed channel's weighted
/// error ([`crate::frame::extra_levels`]), to code losslessly through a
/// Palette transform. None without such a channel, or when every value is
/// needed. A `black_plate` keeps its no-ink value (the full one) exact.
fn lossy_palette_levels(
    input: &PlainInput,
    squeezed: &Pyramid,
    xsize: usize,
    ysize: usize,
    black_plate: bool,
) -> Option<Vec<i32>> {
    use crate::encode_image::AlphaPlane;
    use crate::frame::extra_levels::{LOSSY_PALETTE_ERROR, error_weights, fewest_levels};
    let [(plane, d)] = input.extra.as_slice() else {
        return None;
    };
    if *d <= 0.0 || plane.iter().any(|&v| !(0..=65535).contains(&v)) {
        return None;
    }
    let source = AlphaPlane::U16 {
        data: plane.iter().map(|&v| v as u16).collect(),
        bits: 16,
    };
    let weights = error_weights(&source, xsize, ysize);
    let error: f64 = squeezed
        .decoded_extra(0)
        .iter()
        .zip(plane)
        .zip(&weights)
        .map(|((&v, &s), &w)| {
            let e = f64::from(v.clamp(0, 65535) - s);
            f64::from(w) * e * e
        })
        .sum();
    let AlphaPlane::U16 { data, .. } =
        fewest_levels(&source, &weights, LOSSY_PALETTE_ERROR * error, black_plate)?
    else {
        unreachable!("levels keep the plane's type")
    };
    Some(data.into_iter().map(i32::from).collect())
}

/// The group_size_shift of a regular frame. From Slow, 1024-pixel groups:
/// every group edge costs its first row and column their neighbors, so
/// larger groups code the same pixels in -1.3 % (Slow) / -1.5 % (Fast) over
/// 54 forced encodes, to -4.6 %. They cost 18-24 % encode time at 8
/// threads and the decoder's group parallelism, so faster tiers keep 256.
fn group_shift(speed: crate::Speed) -> u32 {
    if speed >= crate::Speed::Slow {
        crate::lossless::kit::GroupLayout::LARGE.shift
    } else {
        crate::lossless::kit::GroupLayout::DEFAULT.shift
    }
}

/// Progressive DC: the 1:8 DC image of a VarDCT frame as its squeezed DC
/// frame (frame type kDCFrame, dc_level 1), which the VarDCT frame then
/// references instead of coding its own DC. Full-resolution chroma, no loop
/// filters. `levels` keeps that many of the finest squeeze levels (clamped
/// to at least one; `None` squeezes down to the 8x8 preview). Returns false
/// when the image is too small to squeeze.
pub(crate) fn encode_dc_frame(
    dc: &Image3F,
    vardct_steps: [f32; 3],
    levels: Option<u32>,
    speed: crate::Speed,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    match Pyramid::dc_frame(dc, vardct_steps, levels, pool) {
        Some(pyramid) => race::encode(pyramid, speed, pool, scratch, writer).is_some(),
        None => false,
    }
}

/// Whether [`encode_dc_frame`] can code a DC image of this size.
pub(crate) fn dc_frame_supported(xsize: usize, ysize: usize) -> bool {
    !crate::squeeze::default_squeeze_steps(xsize, ysize, 3).is_empty()
}
