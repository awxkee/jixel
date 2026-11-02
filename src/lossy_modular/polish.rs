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

//! ExtraSlow residual polish of lossy-modular XYB frames.
//!
//! The luma squeeze residuals of every level are revisited after the
//! closed-loop quantizer, coarsest level first: step a level toward
//! zero, or restore a zeroed one whose exact residual reaches a quarter step.
//! Each proposal is decoded exactly: the inverse squeeze's tendency reads the
//! previous output pixel, so a line is recomputed until it agrees with the old
//! output again, and the changed outputs are the next finer level's changed
//! averages, down to full-resolution luma. Proposals are priced under a frozen
//! context model of their residual channel and kept only when SSIMULACRA2 and
//! ColorVideoVDP both improve at their own exchange rates (deficit per bit).

use super::layout::XYB_SCALE;
use super::pyramid::Pyramid;
use crate::adaptive_quant::dirty_log2f;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::cvvdp_arbiter::{CvvdpArbiter, CvvdpCheckpoint};
use crate::image::Image3F;
use crate::squeeze::{Channel, apply_step_inverse, smooth_tendency};
use crate::ssimulacra2::{Rect, Ssimulacra2Checkpoint, Ssimulacra2Reference};
use crate::thread_pool::ThreadPool;
use crate::util::round_i32;
use cvvdp_lite::PreparedReference;

/// Proposals verified per tile and kind (toward zero / restore).
const PER_TILE: usize = 32;
/// Residual pairs per tile along the line, and lines per tile.
const TILE_PAIRS: usize = 16;
const TILE_LINES: usize = 16;
/// A zeroed residual is a restore candidate from 1 / this many steps.
const RESTORE_DENOMINATOR: u64 = 4;
/// Memory model of a band's frame, pyramid and metric states.
const POLISH_STATE_BYTES_PER_PIXEL: usize = 256;
const POLISH_MEMORY_BUDGET: usize = 1 << 30;
const POLISH_MAX_BANDS: usize = 8;
const MIN_BAND_ROWS: usize = 64;
/// Context buckets of |left| + |top| over the divided residuals.
const BUCKETS: usize = 6;
/// Values coded by the histogram; larger magnitudes pay an escape.
const VMAX: i32 = 15;

/// Polish `pyramid`, the arm's quantized `xyb` at `distance`: the exchange
/// rates come from Slow-coded pyramids at 0.9x and 1.1x the distance (deficit
/// per coded bit). Without a ColorVideoVDP slope (near transparency the
/// neighbours do not separate) SSIMULACRA2 decides alone.
pub(super) fn polish_frame(
    pyramid: &mut Pyramid,
    xyb: &Image3F,
    distance: f32,
    screen: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) {
    let Some(reference) = Ssimulacra2Reference::new(xyb) else {
        return;
    };
    let prepared = CvvdpArbiter::prepare(xyb);
    let mut arbiter = prepared
        .clone()
        .and_then(|prepared| CvvdpArbiter::new_shared(xyb, prepared));
    let dc = zero_dc(xyb.xsize(), xyb.ysize());
    let mut point = |d: f32, scratch: &mut CoderScratch| {
        let mut p = Pyramid::xyb(xyb, d, screen, pool)?;
        p.group_shift = pyramid.group_shift;
        let mut w = BitWriter::new();
        super::code::write_frame(&p, crate::Speed::Slow, pool, scratch, &mut w, None);
        let decoded = decode_xyb(&p);
        let cv = arbiter.as_mut().map(|a| a.deficit_of(&decoded, &dc));
        Some((reference.deficit(&decoded), cv, w.bits_written() as f32))
    };
    let (Some((d_lo, cv_lo, b_lo)), Some((d_hi, cv_hi, b_hi))) = (
        point(distance * 0.9, scratch),
        point(distance * 1.1, scratch),
    ) else {
        return;
    };
    if b_lo <= b_hi || d_hi <= d_lo {
        return;
    }
    let cv = match (prepared, cv_lo, cv_hi) {
        (Some(prepared), Some(lo), Some(hi)) if hi > lo => {
            Some((prepared, (hi - lo) / (b_lo - b_hi)))
        }
        _ => None,
    };
    polish(
        pyramid,
        xyb,
        &reference,
        (d_hi - d_lo) / (b_lo - b_hi),
        cv,
        pool,
    );
}

/// Decoded XYB of a pyramid (no loop filters).
pub(super) fn decode_xyb(p: &Pyramid) -> Image3F {
    let mut channels = dequantized(p);
    for s in p.steps.iter().rev() {
        apply_step_inverse(&mut channels, s);
    }
    planes_to_xyb(
        &channels[0].data,
        &channels[1].data,
        &channels[2].data,
        p.xsize,
        p.ysize,
    )
}

fn dequantized(p: &Pyramid) -> Vec<Channel> {
    p.channels
        .iter()
        .zip(&p.quants)
        .map(|(ch, &q)| Channel {
            data: ch.data.iter().map(|&v| v * q as i32).collect(),
            ..ch.clone()
        })
        .collect()
}

fn planes_to_xyb(y: &[i32], x: &[i32], bmy: &[i32], w: usize, h: usize) -> Image3F {
    let mut out = Image3F::new(w, h);
    let len = w * h;
    assert!(x.len() >= len && y.len() >= len && bmy.len() >= len);

    let [px, py, pb] = out.all_planes_mut();
    let [sx, sy, sb] = XYB_SCALE;

    let outputs = px
        .as_mut_slice()
        .iter_mut()
        .zip(py.as_mut_slice())
        .zip(pb.as_mut_slice());

    let inputs = x.iter().zip(y).zip(bmy);

    for (((dx, dy), db), ((&x, &y), &bmy)) in outputs.zip(inputs) {
        *dx = x as f32 / sx;
        *dy = y as f32 / sy;
        *db = (bmy + y) as f32 / sb;
    }

    out
}

/// Frozen context model of one divided residual channel.
#[derive(Clone)]
struct Prices {
    cost: [[f32; (2 * VMAX + 2) as usize]; BUCKETS],
}

impl Prices {
    fn new(res: &[i32], w: usize, h: usize) -> Self {
        let mut counts = [[0u32; (2 * VMAX + 2) as usize]; BUCKETS];
        for y in 0..h {
            for x in 0..w {
                counts[bucket(res, w, x, y)][slot(res[y * w + x])] += 1;
            }
        }
        let mut cost = [[0f32; (2 * VMAX + 2) as usize]; BUCKETS];
        for (c, n) in cost.iter_mut().zip(&counts) {
            let total: f32 = n.iter().map(|&v| v as f32 + 0.5).sum();
            for (c, &k) in c.iter_mut().zip(n) {
                *c = -dirty_log2f((k as f32 + 0.5) / total);
            }
        }
        Self { cost }
    }

    fn of(&self, b: usize, v: i32) -> f32 {
        self.cost[b][slot(v)] + Self::escape(v)
    }

    fn escape(v: i32) -> f32 {
        if v.abs() > VMAX {
            2.0 * ((v.unsigned_abs() - VMAX as u32) as f32 + 1.0).log2() + 1.0
        } else {
            0.0
        }
    }

    /// Bits of the value at (x, y) and of the two neighbours whose context it
    /// sets, for the channel as it stands.
    #[cfg(test)]
    fn local(&self, res: &[i32], w: usize, h: usize, x: usize, y: usize) -> f32 {
        let mut bits = self.of(bucket(res, w, x, y), res[y * w + x]);
        if x + 1 < w {
            bits += self.of(bucket(res, w, x + 1, y), res[y * w + x + 1]);
        }
        if y + 1 < h {
            bits += self.of(bucket(res, w, x, y + 1), res[(y + 1) * w + x]);
        }
        bits
    }
}

fn slot(v: i32) -> usize {
    if v.abs() > VMAX {
        (2 * VMAX + 1) as usize
    } else {
        (v + VMAX) as usize
    }
}

fn bucket(res: &[i32], w: usize, x: usize, y: usize) -> usize {
    let left = if x > 0 {
        res[y * w + x - 1].unsigned_abs()
    } else {
        0
    };
    let top = if y > 0 {
        res[(y - 1) * w + x].unsigned_abs()
    } else {
        0
    };
    bucket_of(left, top)
}

fn bucket_of(left: u32, top: u32) -> usize {
    match left + top {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        _ => 5,
    }
}

/// One squeeze step seen as lines of (average, residual) pairs.
#[derive(Clone)]
struct Lines {
    horizontal: bool,
    /// Output width of the step.
    w: usize,
    /// Pairs per line (residuals along the line) and line count.
    pairs: usize,
    lines: usize,
    /// Averages per line (pairs, plus one when the output length is odd).
    avgs: usize,
}

impl Lines {
    fn avg_index(&self, line: usize, i: usize) -> usize {
        if self.horizontal {
            line * self.avgs + i
        } else {
            i * self.w + line
        }
    }
    /// (line, position) of an average.
    fn avg_pos(&self, idx: usize) -> (usize, usize) {
        if self.horizontal {
            (idx / self.avgs, idx % self.avgs)
        } else {
            (idx % self.w, idx / self.w)
        }
    }
    fn res_index(&self, line: usize, i: usize) -> usize {
        if self.horizontal {
            line * self.pairs + i
        } else {
            i * self.w + line
        }
    }
    /// Output sample of element `k` (0 = first, 1 = second) of pair `i`;
    /// `i == pairs` with `k == 0` is an odd line's copied tail.
    fn out_index(&self, line: usize, i: usize, k: usize) -> usize {
        if self.horizontal {
            line * self.w + 2 * i + k
        } else {
            (2 * i + k) * self.w + line
        }
    }
    /// The residual channel's (x, y) of pair `i` of `line`.
    fn res_xy(&self, line: usize, i: usize) -> (usize, usize) {
        if self.horizontal {
            (i, line)
        } else {
            (line, i)
        }
    }
}

/// One polished luma level: its step, residual channel and models.
#[derive(Clone)]
struct Level {
    l: Lines,
    stored: usize,
    q: i64,
    div: Vec<i32>,
    res: Vec<i64>,
    res_w: usize,
    res_h: usize,
    /// Rows of the full image per residual row.
    vshift: i32,
    prices: Prices,
    /// Source luma at this level's output resolution.
    exact: Vec<i32>,
}

/// Recompute pairs `from..` of `line` (always through `until`, then until a
/// pair decodes unchanged), logging (level, index, old).
#[allow(clippy::too_many_arguments)]
fn relayout(
    lv: &Level,
    level: usize,
    avg: &[i32],
    out: &mut [i32],
    line: usize,
    from: usize,
    until: usize,
    log: &mut Vec<(usize, usize, i32)>,
) {
    let l = &lv.l;
    let stride = if l.horizontal { 1 } else { l.w };
    let mut ai = l.avg_index(line, from);
    let mut ri = l.res_index(line, from);
    let mut ia = l.out_index(line, from, 0);
    let mut left = if from > 0 {
        i64::from(out[ia - stride])
    } else {
        i64::from(avg[ai])
    };
    for i in from..l.pairs {
        let av = i64::from(avg[ai]);
        let next = if i + 1 < l.avgs {
            i64::from(avg[ai + stride])
        } else {
            av
        };
        let diff = lv.res[ri] + smooth_tendency(left, av, next);
        let first = av + diff / 2;
        let (a, b) = (first as i32, (first - diff) as i32);
        let ib = ia + stride;
        if i > until && out[ia] == a && out[ib] == b {
            break;
        }
        for (idx, v) in [(ia, a), (ib, b)] {
            if out[idx] != v {
                log.push((level, idx, out[idx]));
                out[idx] = v;
            }
        }
        left = i64::from(b);
        ai += stride;
        ri += stride;
        ia += 2 * stride;
    }
}

/// Apply level `k`'s residual change at pair `i` of `line` (already written
/// into `levels[k].res`) through every finer level. `outs[j]` is level j's
/// output and `outs[j + 1]` its averages. Returns the start of the changed
/// full-resolution luma entries in `log`, which also undoes everything.
fn propagate(
    levels: &[Level],
    outs: &mut [Vec<i32>],
    k: usize,
    line: usize,
    i: usize,
    log: &mut Vec<(usize, usize, i32)>,
) -> usize {
    // The proposed residual affects just one line at its own level. Avoid
    // building a one-entry tree, particularly at the finest level.
    let mut start = log.len();
    let (lo, hi) = outs.split_at_mut(k + 1);
    relayout(&levels[k], k, &hi[0], &mut lo[k], line, i, i, log);
    let mut pending = std::collections::BTreeMap::new();
    for j in (0..k).rev() {
        // The changed outputs are level j's changed averages: their own
        // pair and the pair before it (its `next`) recompute.
        pending.clear();
        let below = &levels[j].l;
        for &(_, idx, _) in &log[start..] {
            let (line, pos) = below.avg_pos(idx);
            let e = pending.entry(line).or_insert((usize::MAX, 0));
            e.0 = e.0.min(pos.saturating_sub(1));
            e.1 = e.1.max(pos);
        }
        start = log.len();
        let lv = &levels[j];
        let (lo, hi) = outs.split_at_mut(j + 1);
        let (out, avg) = (&mut lo[j], &hi[0]);
        for (&line, &(from, until)) in &pending {
            if until >= lv.l.pairs {
                // An odd line's last average is copied to its last output.
                let (ia, ix) = (
                    lv.l.avg_index(line, lv.l.pairs),
                    lv.l.out_index(line, lv.l.pairs, 0),
                );
                if out[ix] != avg[ia] {
                    log.push((j, ix, out[ix]));
                    out[ix] = avg[ia];
                }
            }
            if from < lv.l.pairs {
                relayout(lv, j, avg, out, line, from, until.min(lv.l.pairs - 1), log);
            }
        }
    }
    start
}

fn undo(outs: &mut [Vec<i32>], log: &[(usize, usize, i32)]) {
    for &(j, idx, old) in log.iter().rev() {
        outs[j][idx] = old;
    }
}

/// Per stored channel: `Some((base channel, step))` for a residual.
fn residual_tags(p: &Pyramid, num_c: usize) -> Vec<Option<(usize, usize)>> {
    let mut tags: Vec<Option<(usize, usize)>> = vec![None; num_c];
    for (k, s) in p.steps.iter().enumerate() {
        let endc = s.begin_c + s.num_c - 1;
        let offset = if s.in_place { endc + 1 } else { tags.len() };
        for i in 0..s.num_c {
            tags.insert(offset + i, Some((s.begin_c + i, k)));
        }
    }
    tags
}

/// A second arbiter: ColorVideoVDP with its own exchange rate. A move the
/// SSIMULACRA2 rule accepts must also lower its deficit plus rate.
struct CvJudge {
    arbiter: CvvdpArbiter,
    lambda_bits: f32,
}

/// The frame's zero DC error at 1:8 (modular frames have no VarDCT DC).
fn zero_dc(w: usize, h: usize) -> Image3F {
    Image3F::new(w.div_ceil(8), h.div_ceil(8))
}

/// Polish the pyramid's luma residuals in place. `lambda_bits` is the
/// SSIMULACRA2 deficit one bit buys at this frame's operating point; `cv`
/// carries the analysed source and exchange rate of the ColorVideoVDP judge.
fn polish(
    p: &mut Pyramid,
    source: &Image3F,
    reference: &Ssimulacra2Reference,
    lambda_bits: f32,
    cv: Option<(std::sync::Arc<PreparedReference>, f32)>,
    pool: &ThreadPool,
) -> Option<()> {
    let base = Baseline::new(p, source)?;
    let (w, h) = (base.w, base.h);
    // Horizontal bands, each against its own copy of the pyramid, the frame
    // and the metric states built from the same baseline, so the result
    // depends on neither the thread count nor the scheduling. The band count
    // is bounded by the states' memory.
    let bands = (POLISH_MEMORY_BUDGET / (POLISH_STATE_BYTES_PER_PIXEL * w * h).max(1))
        .clamp(1, POLISH_MAX_BANDS)
        .min(h.div_ceil(MIN_BAND_ROWS))
        .max(1);
    let results = pool.map(bands, |b| {
        let rows = (b * h / bands, (b + 1) * h / bands);
        let cv = cv.as_ref().and_then(|(prepared, lambda_bits)| {
            Some(CvJudge {
                arbiter: CvvdpArbiter::new_shared(source, prepared.clone())?,
                lambda_bits: *lambda_bits,
            })
        });
        polish_band(&base, reference, cv, lambda_bits, rows)
    });
    let mut lv = base.lv;
    for (j, ri, target) in results.into_iter().flatten() {
        lv[j].div[ri] = target;
    }
    for level in lv {
        p.channels[level.stored].data = level.div;
    }
    Some(())
}

/// The decoded pyramid every band starts from.
struct Baseline {
    lv: Vec<Level>,
    outs: Vec<Vec<i32>>,
    recon: Image3F,
    bmy: Vec<i32>,
    src_luma: Vec<i32>,
    w: usize,
    h: usize,
}

impl Baseline {
    /// The luma levels of `p`'s shared squeeze steps, decoded, with the
    /// source's exact averages; None without a three-channel shared step.
    fn new(p: &Pyramid, source: &Image3F) -> Option<Self> {
        let shared = p.steps.iter().position(|s| s.in_place && s.begin_c == 0)?;
        if p.steps[shared].num_c < 3 {
            return None;
        }
        let top = p.steps.len() - 1;
        let tags = residual_tags(p, 3);
        debug_assert_eq!(tags.len(), p.channels.len());
        // outs[j] = output of level j (step shared + j); outs[n] = the coarsest
        // luma averages, which stay fixed.
        let n = top - shared + 1;
        let mut walk = dequantized(p);
        let mut outs: Vec<Vec<i32>> = vec![Vec::new(); n + 1];
        outs[n] = walk[0].data.clone();
        let mut geometry = vec![(0usize, 0usize); n + 1];
        geometry[n] = (walk[0].w, walk[0].h);
        for k in (shared..=top).rev() {
            apply_step_inverse(&mut walk, &p.steps[k]);
            outs[k - shared] = walk[0].data.clone();
            geometry[k - shared] = (walk[0].w, walk[0].h);
        }
        for s in p.steps[..shared].iter().rev() {
            apply_step_inverse(&mut walk, s);
        }
        let (w, h) = (p.xsize, p.ysize);
        debug_assert_eq!(geometry[0], (w, h));
        let x_plane = walk[1].data.clone();
        let bmy = walk[2].data.clone();
        let recon = planes_to_xyb(&outs[0], &x_plane, &bmy, w, h);
        // Source luma on the lattice, and its exact averages per level.
        let src_luma: Vec<i32> = (0..h)
            .flat_map(|y| {
                source.plane_row(1, y)[..w]
                    .iter()
                    .map(|&v| round_i32(v * XYB_SCALE[1]))
            })
            .collect();
        let mut exact_ch = Channel {
            data: src_luma.clone(),
            ..Channel::new(w, h)
        };
        let mut exact = vec![src_luma.clone()];
        for k in shared..top {
            let (avg, _) = if p.steps[k].horizontal {
                crate::squeeze::fwd_h_squeeze(&exact_ch)
            } else {
                crate::squeeze::fwd_v_squeeze(&exact_ch)
            };
            exact.push(avg.data.clone());
            exact_ch = avg;
        }
        let mut lv: Vec<Level> = Vec::with_capacity(n);
        for j in 0..n {
            let k = shared + j;
            let stored = tags.iter().position(|&t| t == Some((0, k)))?;
            let ch = &p.channels[stored];
            let horizontal = p.steps[k].horizontal;
            let (ow, _) = geometry[j];
            let (aw, ah) = geometry[j + 1];
            let q = i64::from(p.quants[stored]);
            lv.push(Level {
                l: Lines {
                    horizontal,
                    w: ow,
                    pairs: if horizontal { ch.w } else { ch.h },
                    lines: if horizontal { ch.h } else { ch.w },
                    avgs: if horizontal { aw } else { ah },
                },
                stored,
                q,
                div: ch.data.clone(),
                res: ch.data.iter().map(|&v| i64::from(v) * q).collect(),
                res_w: ch.w,
                res_h: ch.h,
                vshift: ch.vshift,
                prices: Prices::new(&ch.data, ch.w, ch.h),
                exact: std::mem::take(&mut exact[j]),
            });
        }
        Some(Self {
            lv,
            outs,
            recon,
            bmy,
            src_luma,
            w,
            h,
        })
    }
}

/// Polish the tiles whose first full-resolution row lies in `rows`. Returns
/// the accepted moves as (level, residual index, divided value).
fn polish_band(
    base: &Baseline,
    reference: &Ssimulacra2Reference,
    mut cv: Option<CvJudge>,
    lambda_bits: f32,
    rows: (usize, usize),
) -> Vec<(usize, usize, i32)> {
    let (w, h) = (base.w, base.h);
    let mut lv = base.lv.clone();
    let mut outs = base.outs.clone();
    let mut recon = base.recon.clone();
    let (bmy, src_luma) = (&base.bmy, &base.src_luma);
    let n = lv.len();
    let mut moves = Vec::new();
    let mut state = reference.state(&recon);
    let dc = zero_dc(w, h);
    if let Some(cv) = cv.as_mut() {
        cv.arbiter.reset(&recon, &dc);
    }
    let mut cv_cp = CvvdpCheckpoint::default();
    let mut cp = Ssimulacra2Checkpoint::default();
    let mut log: Vec<(usize, usize, i32)> = Vec::new();
    // Reuse the two tile-sized candidate buffers throughout this band.
    let mut toward: Vec<(f64, usize, usize, i32)> = Vec::new();
    let mut restore: Vec<(f64, usize, usize, i32)> = Vec::new();
    for j in (0..n).rev() {
        let (lines, pairs) = (lv[j].l.lines, lv[j].l.pairs);
        for tile_line in (0..lines).step_by(TILE_LINES) {
            for tile_pair in (0..pairs).step_by(TILE_PAIRS) {
                let (_, ry) = lv[j].l.res_xy(tile_line, tile_pair);
                if !(rows.0..rows.1).contains(&(ry << lv[j].vshift)) {
                    continue;
                }
                // (proxy, line, pair, new divided value)
                toward.clear();
                restore.clear();
                for line in tile_line..(tile_line + TILE_LINES).min(lines) {
                    for i in tile_pair..(tile_pair + TILE_PAIRS).min(pairs) {
                        let level = &lv[j];
                        let (cx, cy) = level.l.res_xy(line, i);
                        let ri = cy * level.res_w + cx;
                        let v = level.div[ri];
                        let target = if v != 0 {
                            v - v.signum()
                        } else {
                            // The exact residual against the decoded neighbours.
                            let l = &level.l;
                            let avg = &outs[j + 1];
                            let av = i64::from(avg[l.avg_index(line, i)]);
                            let next = if i + 1 < l.avgs {
                                i64::from(avg[l.avg_index(line, i + 1)])
                            } else {
                                av
                            };
                            let left = if i > 0 {
                                i64::from(outs[j][l.out_index(line, i - 1, 1)])
                            } else {
                                av
                            };
                            let d = i64::from(level.exact[l.out_index(line, i, 0)])
                                - i64::from(level.exact[l.out_index(line, i, 1)]);
                            let e = d - smooth_tendency(left, av, next);
                            // |e / q| < 1/4, exactly, without float division.
                            if e.unsigned_abs() < (level.q as u64).div_ceil(RESTORE_DENOMINATOR) {
                                continue;
                            }
                            e.signum() as i32
                        };
                        let dbits = move_bits(&lv[j], cx, cy, target);
                        let level = &mut lv[j];
                        level.res[ri] = i64::from(target) * level.q;
                        log.clear();
                        let changed = propagate(&lv, &mut outs, j, line, i, &mut log);
                        let mut dsse = 0f64;
                        for &(_, idx, old) in &log[changed..] {
                            let s = f64::from(src_luma[idx]);
                            dsse += (f64::from(outs[0][idx]) - s).powi(2)
                                - (f64::from(old) - s).powi(2);
                        }
                        undo(&mut outs, &log);
                        let level = &mut lv[j];
                        level.res[ri] = i64::from(v) * level.q;
                        if v != 0 {
                            toward.push((-f64::from(dbits) / dsse.max(1.0), line, i, target));
                        } else {
                            restore.push((-dsse / f64::from(dbits).max(0.05), line, i, target));
                        }
                    }
                }
                for list in [&mut toward, &mut restore] {
                    // Generation order is (line, pair). Keep that tie order
                    // explicitly, allowing the allocation-free sort.
                    list.sort_unstable_by(|a, b| {
                        b.0.total_cmp(&a.0)
                            .then_with(|| a.1.cmp(&b.1))
                            .then_with(|| a.2.cmp(&b.2))
                    });
                    for &(_, line, i, target) in list.iter().take(PER_TILE) {
                        let (cx, cy) = lv[j].l.res_xy(line, i);
                        let ri = cy * lv[j].res_w + cx;
                        let v = lv[j].div[ri];
                        // The tile's earlier commits may have moved this element.
                        let dbits = move_bits(&lv[j], cx, cy, target);
                        lv[j].div[ri] = target;
                        lv[j].res[ri] = i64::from(target) * lv[j].q;
                        log.clear();
                        let changed = propagate(&lv, &mut outs, j, line, i, &mut log);
                        if changed == log.len() {
                            undo(&mut outs, &log);
                            lv[j].div[ri] = v;
                            lv[j].res[ri] = i64::from(v) * lv[j].q;
                            continue;
                        }
                        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
                        let [_, py, pb] = recon.all_planes_mut();
                        let (py, pb) = (py.as_mut_slice(), pb.as_mut_slice());
                        for &(_, idx, _) in &log[changed..] {
                            let (x, y) = (idx % w, idx / w);
                            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                            py[idx] = outs[0][idx] as f32 / XYB_SCALE[1];
                            pb[idx] = (bmy[idx] + outs[0][idx]) as f32 / XYB_SCALE[2];
                        }
                        let base = state.deficit();
                        let rect = Rect::new(x0, y0, x1, y1);
                        state.update_checkpointed(&recon, rect, &mut cp);
                        if state.deficit() - base + lambda_bits * dbits < 0.0 {
                            let keep = match cv.as_mut() {
                                None => true,
                                Some(cv) => {
                                    let base = cv.arbiter.deficit();
                                    let after = cv
                                        .arbiter
                                        .update_checkpointed(&recon, &dc, rect, &mut cv_cp);
                                    let keep = after - base + cv.lambda_bits * dbits < 0.0;
                                    if !keep {
                                        cv.arbiter.rollback(&cv_cp);
                                    }
                                    keep
                                }
                            };
                            if keep {
                                moves.push((j, ri, target));
                                continue;
                            }
                        }
                        state.rollback(&cp);
                        undo(&mut outs, &log);
                        lv[j].div[ri] = v;
                        lv[j].res[ri] = i64::from(v) * lv[j].q;

                        let [_, py, pb] = recon.all_planes_mut();
                        let py = py.as_mut_slice();
                        let pb = pb.as_mut_slice();
                        let y_out = &outs[0];
                        let [_, sy, sb] = XYB_SCALE;

                        for &(_, idx, _) in &log[changed..] {
                            let y = y_out[idx];
                            py[idx] = y as f32 / sy;
                            pb[idx] = (bmy[idx] + y) as f32 / sb;
                        }
                    }
                }
            }
        }
    }
    moves
}

/// Bit change of setting the divided residual at (x, y) to `target`: the
/// value's own token and the two whose context it sets.
fn move_bits(level: &Level, x: usize, y: usize, target: i32) -> f32 {
    let (w, h) = (level.res_w, level.res_h);
    let i = y * w + x;
    let res = &level.div;
    let prices = &level.prices;
    let v = res[i];
    let context = bucket(res, w, x, y);
    let mut before = prices.of(context, v);
    let mut after = prices.of(context, target);
    // Only the right and bottom contexts change. Their values (and escape
    // costs) stay fixed, so calculate each escape just once. Preserve the
    // original local sums' addition order for exactly the same bit delta.
    let mut neighbour = |idx: usize, other: u32| {
        let value = res[idx];
        let slot = slot(value);
        let extra = Prices::escape(value);
        before += prices.cost[bucket_of(v.unsigned_abs(), other)][slot] + extra;
        after += prices.cost[bucket_of(target.unsigned_abs(), other)][slot] + extra;
    };
    if x + 1 < w {
        let top = if y > 0 {
            res[i + 1 - w].unsigned_abs()
        } else {
            0
        };
        neighbour(i + 1, top);
    }
    if y + 1 < h {
        let left = if x > 0 {
            res[i + w - 1].unsigned_abs()
        } else {
            0
        };
        neighbour(i + w, left);
    }
    after - before
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// Gradients, an edge and noise in plausible XYB ranges.
    fn source(w: usize, h: usize, seed: &mut u64) -> Image3F {
        let mut img = Image3F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let noise = (next(seed) % 1000) as f32 / 1000.0 - 0.5;
                let edge = if x * 3 > w * 2 { 0.15 } else { 0.0 };
                let luma = 0.25 + 0.3 * x as f32 / w as f32 + edge + 0.02 * noise;
                img.plane_row_mut(0, y)[x] = 0.01 * (y as f32 / h as f32 - 0.5);
                img.plane_row_mut(1, y)[x] = luma;
                img.plane_row_mut(2, y)[x] = luma * 0.9 + 0.02 * noise;
            }
        }
        img
    }

    fn decoded_luma(p: &Pyramid) -> Vec<i32> {
        let mut channels = dequantized(p);
        for s in p.steps.iter().rev() {
            apply_step_inverse(&mut channels, s);
        }
        channels.swap_remove(0).data
    }

    #[test]
    fn move_prices_preserve_local_sum_bits() {
        let mut seed = 17;
        let pool = ThreadPool::new(1);
        for (w, h) in [(67, 45), (41, 90)] {
            let src = source(w, h, &mut seed);
            let p = Pyramid::xyb(&src, 3.0, false, &pool).unwrap();
            let mut base = Baseline::new(&p, &src).unwrap();
            for level in &mut base.lv {
                // Include all context boundaries, both signs and escapes.
                let values = [-1000, -16, -15, -8, -4, -2, -1, 0, 1, 2, 4, 8, 15, 16, 1000];
                for value in &mut level.div {
                    *value = values[next(&mut seed) as usize % values.len()];
                }
                let (w, h) = (level.res_w, level.res_h);
                level.prices = Prices::new(&level.div, w, h);
                for y in 0..h {
                    for x in 0..w {
                        let idx = y * w + x;
                        let old = level.div[idx];
                        for target in [-16, -1, 0, 1, 16] {
                            let actual = move_bits(level, x, y, target);
                            let before = level.prices.local(&level.div, w, h, x, y);
                            level.div[idx] = target;
                            let after = level.prices.local(&level.div, w, h, x, y);
                            level.div[idx] = old;
                            assert_eq!(actual.to_bits(), (after - before).to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn propagated_moves_decode_like_the_pyramid() {
        let mut seed = 5u64;
        // Wide (H first) and tall (V first) scripts, odd sizes.
        for (w, h) in [(67, 45), (41, 90), (130, 64)] {
            let src = source(w, h, &mut seed);
            let pool = ThreadPool::new(1);
            let mut p = Pyramid::xyb(&src, 3.0, false, &pool).unwrap();
            let base = Baseline::new(&p, &src).unwrap();
            assert_eq!(base.outs[0], decoded_luma(&p));
            let (mut lv, mut outs) = (base.lv.clone(), base.outs.clone());
            let mut log = Vec::new();
            for _ in 0..300 {
                let j = next(&mut seed) as usize % lv.len();
                let (lines, pairs) = (lv[j].l.lines, lv[j].l.pairs);
                if lines == 0 || pairs == 0 {
                    continue;
                }
                let line = next(&mut seed) as usize % lines;
                let i = next(&mut seed) as usize % pairs;
                let (cx, cy) = lv[j].l.res_xy(line, i);
                let ri = cy * lv[j].res_w + cx;
                let step = if next(&mut seed) % 2 == 0 { 1 } else { -1 };
                lv[j].div[ri] += step;
                lv[j].res[ri] = i64::from(lv[j].div[ri]) * lv[j].q;
                log.clear();
                propagate(&lv, &mut outs, j, line, i, &mut log);
            }
            for level in &lv {
                p.channels[level.stored].data = level.div.clone();
            }
            assert_eq!(outs[0], decoded_luma(&p), "{w}x{h}");
        }
    }

    #[test]
    fn undo_restores_every_level() {
        let mut seed = 9u64;
        let src = source(96, 70, &mut seed);
        let pool = ThreadPool::new(1);
        let p = Pyramid::xyb(&src, 2.0, false, &pool).unwrap();
        let base = Baseline::new(&p, &src).unwrap();
        let (mut lv, mut outs) = (base.lv.clone(), base.outs.clone());
        let mut log = Vec::new();
        for j in 0..lv.len() {
            let (cx, cy) = lv[j].l.res_xy(0, 0);
            let ri = cy * lv[j].res_w + cx;
            lv[j].res[ri] += 5 * lv[j].q;
            propagate(&lv, &mut outs, j, 0, 0, &mut log);
            lv[j].res[ri] -= 5 * lv[j].q;
        }
        undo(&mut outs, &log);
        assert_eq!(outs, base.outs);
    }

    #[test]
    fn polish_is_independent_of_the_thread_count() {
        let mut seed = 3u64;
        let src = source(150, 260, &mut seed);
        let reference = Ssimulacra2Reference::new(&src).unwrap();
        let polished = |threads: usize| {
            let pool = ThreadPool::new(threads);
            let mut p = Pyramid::xyb(&src, 3.0, false, &pool).unwrap();
            let before: Vec<Vec<i32>> = p.channels.iter().map(|c| c.data.clone()).collect();
            polish(&mut p, &src, &reference, 1e-4, None, &pool).unwrap();
            let after: Vec<Vec<i32>> = p.channels.iter().map(|c| c.data.clone()).collect();
            (before, after)
        };
        let (before, one) = polished(1);
        let (_, four) = polished(4);
        assert_eq!(one, four);
        assert_ne!(before, one, "the polish moved nothing");
    }
}
