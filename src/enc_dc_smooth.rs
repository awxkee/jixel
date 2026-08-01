/*
 * // Copyright (c) Radzivon Bartoshyk 8/2026. All rights reserved.
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

//! Closed-loop DC rounding against decoder-side adaptive DC smoothing.

use crate::coder_scratch::CoderScratch;
use crate::image::{Image3F, Image3S};
use crate::thread_pool::ThreadPool;

// libjxl compressed_dc.cc kernel weights.
const W1: f32 = 0.203_451_4;
const W2: f32 = 0.033_482_92;
const W0: f32 = 1.0 - 4.0 * (W1 + W2);

const ACTIVE_GAP: f32 = 1.7;

const MAX_SWEEPS: usize = 6;

/// Raw-pointer view of the descent state, shared across band workers.
///
/// # Safety
/// All pointers come from live, never-reallocated buffers owned by
/// [`optimize_dc_rounding`], and every access goes through raw-pointer
/// reads/writes (no `&mut` slices are formed), so aliasing rules are not
/// violated. Concurrent use is restricted to the two-phase band schedule:
/// within a phase each running band writes `recon`/`quant`/`active` only in
/// its own rows and `cache` in its own rows ±1, while all rows it reads
/// outside its own range belong to bands frozen in that phase. Bands are ≥2
/// rows tall, which keeps the ±1 cache spills of two running bands disjoint.
struct Shared<'a> {
    w: usize,
    h: usize,
    steps: [f32; 3],
    inv_step2: [f32; 3],
    r_b: f32,
    target: [&'a [f32]; 3],
    recon: [*mut f32; 3],
    quant: [*mut i16; 3],
    cache: *mut f64,
    active: *mut bool,
}

unsafe impl Sync for Shared<'_> {}

/// Decoder DC output error at (x, y): weighted SSE of the smoothed (interior)
/// or raw (border) recon vs the float target, summed over channels and
/// normalized to quant-step units.
///
/// # Safety
/// (x, y) must be in bounds and the [`Shared`] access discipline upheld.
#[inline]
unsafe fn output_err(sh: &Shared, x: usize, y: usize) -> f64 {
    // SAFETY: contract stated on the function.
    unsafe {
        let w = sh.w;
        let i = y * w + x;
        if x == 0 || y == 0 || x + 1 >= w || y + 1 >= sh.h {
            let mut sum = 0.0f64;
            for c in 0..3 {
                let e = *sh.recon[c].add(i) - sh.target[c][i];
                sum += (e * e * sh.inv_step2[c]) as f64;
            }
            return sum;
        }
        let mut mc = [0.0f32; 3];
        let mut sm = [0.0f32; 3];
        let mut gap = 0.5f32;
        for c in 0..3 {
            let p = sh.recon[c];
            let corner =
                *p.add(i - w - 1) + *p.add(i - w + 1) + *p.add(i + w - 1) + *p.add(i + w + 1);
            let side = *p.add(i - 1) + *p.add(i + 1) + *p.add(i - w) + *p.add(i + w);
            let m = *p.add(i);
            let s = corner * W2 + side * W1 + m * W0;
            mc[c] = m;
            sm[c] = s;
            gap = gap.max((m - s).abs() / sh.steps[c]);
        }
        let factor = (3.0 - 4.0 * gap).max(0.0);
        let mut sum = 0.0f64;
        for c in 0..3 {
            let out = mc[c] + (sm[c] - mc[c]) * factor;
            let e = out - sh.target[c][i];
            sum += (e * e * sh.inv_step2[c]) as f64;
        }
        sum
    }
}

/// Freshly computed error over the 3×3 output window of (x, y) — the exact
/// set of outputs a move at (x, y) can affect.
///
/// # Safety
/// See [`output_err`].
unsafe fn window_err(sh: &Shared, x: usize, y: usize) -> f64 {
    // SAFETY: contract stated on the function.
    unsafe {
        let mut sum = 0.0f64;
        for ny in y.saturating_sub(1)..(y + 2).min(sh.h) {
            for nx in x.saturating_sub(1)..(x + 2).min(sh.w) {
                sum += output_err(sh, nx, ny);
            }
        }
        sum
    }
}

/// Same window, but summed from the per-pixel error cache.
///
/// # Safety
/// See [`output_err`]; the cache must be current for the window.
unsafe fn window_cache_sum(sh: &Shared, x: usize, y: usize) -> f64 {
    // SAFETY: contract stated on the function.
    unsafe {
        let mut sum = 0.0f64;
        for ny in y.saturating_sub(1)..(y + 2).min(sh.h) {
            for nx in x.saturating_sub(1)..(x + 2).min(sh.w) {
                sum += *sh.cache.add(ny * sh.w + nx);
            }
        }
        sum
    }
}

/// Shift the recon for a `delta`-step move on channel `c` at (x, y),
/// propagating a Y move into B through the CfL slope.
///
/// # Safety
/// See [`output_err`].
#[inline]
unsafe fn apply_move(sh: &Shared, c: usize, x: usize, y: usize, delta: f32) {
    // SAFETY: contract stated on the function.
    unsafe {
        let i = y * sh.w + x;
        match c {
            0 => *sh.recon[0].add(i) += delta * sh.steps[0],
            1 => {
                *sh.recon[1].add(i) += delta * sh.steps[1];
                *sh.recon[2].add(i) += delta * sh.steps[1] * sh.r_b;
            }
            _ => *sh.recon[2].add(i) += delta * sh.steps[2],
        }
    }
}

/// Worst-channel deviation from the 3×3 kernel average in quant-step units
/// (the smoothing-gate quantity). Interior pixels only.
///
/// # Safety
/// See [`output_err`]; (x, y) must be interior.
#[inline]
unsafe fn max_gap_at(sh: &Shared, x: usize, y: usize) -> f32 {
    // SAFETY: contract stated on the function.
    unsafe {
        let w = sh.w;
        let i = y * w + x;
        let mut gap = 0.0f32;
        for c in 0..3 {
            let p = sh.recon[c];
            let corner =
                *p.add(i - w - 1) + *p.add(i - w + 1) + *p.add(i + w - 1) + *p.add(i + w + 1);
            let side = *p.add(i - 1) + *p.add(i + 1) + *p.add(i - w) + *p.add(i + w);
            let s = corner * W2 + side * W1 + *p.add(i) * W0;
            gap = gap.max((*p.add(i) - s).abs() / sh.steps[c]);
        }
        gap
    }
}

/// One coordinate-descent sweep over the band's rows. Returns the number of
/// accepted moves and records their positions in `touched`.
///
/// # Safety
/// Callers must uphold the two-phase band schedule described on [`Shared`]:
/// `[y0, y1)` is this worker's exclusive row range in the current phase, and
/// all adjacent bands are frozen.
unsafe fn sweep_band(sh: &Shared, y0: usize, y1: usize, touched: &mut Vec<u32>) -> u32 {
    // SAFETY: contract stated on the function.
    unsafe {
        let w = sh.w;
        let mut changed = 0u32;
        for y in y0..y1 {
            for x in 0..w {
                if !*sh.active.add(y * w + x) {
                    continue;
                }
                let mut before = window_cache_sum(sh, x, y);
                for c in 0..3 {
                    let qp = sh.quant[c].add(y * w + x);
                    // Try both directions; keep the better if it improves.
                    let mut best_delta = 0i16;
                    let mut best = before;
                    for delta in [-1i16, 1] {
                        let q = *qp;
                        if q.checked_add(delta).is_none() {
                            continue;
                        }
                        apply_move(sh, c, x, y, delta as f32);
                        let after = window_err(sh, x, y);
                        apply_move(sh, c, x, y, -delta as f32);
                        if after < best - 1e-9 {
                            best = after;
                            best_delta = delta;
                        }
                    }
                    if best_delta != 0 {
                        *qp += best_delta;
                        apply_move(sh, c, x, y, best_delta as f32);
                        // Refresh the error cache over the affected window; the
                        // one-row spill outside the band is covered by the
                        // two-phase schedule.
                        for ny in y.saturating_sub(1)..(y + 2).min(sh.h) {
                            for nx in x.saturating_sub(1)..(x + 2).min(w) {
                                *sh.cache.add(ny * w + nx) = output_err(sh, nx, ny);
                            }
                        }
                        before = window_cache_sum(sh, x, y);
                        changed += 1;
                        touched.push((y * w + x) as u32);
                    }
                }
            }
        }
        changed
    }
}

struct Band {
    y0: usize,
    y1: usize,
    touched: Vec<u32>,
    changed: u32,
}

/// Re-round the quantized DC plane so the *smoothed* decoder recon best
/// matches the unquantized DC. Coordinate descent over per-pixel ±1 moves on
/// each channel; a Y move shifts the B recon through the CfL slope, which the
/// model prices exactly. Only valid when `quant_img` spans the whole image
/// (the smoothing filter runs on the full DC plane, so a partial view would
/// treat interior seams as unsmoothed borders). Pass `parallel` to run the
/// row bands on the encoder pool; `None` runs serially (same result).
pub(crate) fn optimize_dc_rounding(
    quant_img: &mut Image3S,
    target: &Image3F,
    steps: [f32; 3],
    r_b: f32,
    mut parallel: Option<(&ThreadPool, &mut CoderScratch)>,
) -> u32 {
    let w = quant_img.xsize();
    let h = quant_img.ysize();
    if w <= 2 || h <= 2 {
        return 0;
    }
    let inv_step2 = [
        1.0 / (steps[0] * steps[0]),
        1.0 / (steps[1] * steps[1]),
        1.0 / (steps[2] * steps[2]),
    ];

    // Contiguous working copies; the Vecs are never reallocated, so the raw
    // pointers in `Shared` stay valid until write-back.
    let mut quant: [Vec<i16>; 3] = [vec![0; w * h], vec![0; w * h], vec![0; w * h]];
    let mut recon: [Vec<f32>; 3] = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
    for y in 0..h {
        let qx = quant_img.plane_row(0, y);
        let qy = quant_img.plane_row(1, y);
        let qb = quant_img.plane_row(2, y);
        for x in 0..w {
            let i = y * w + x;
            quant[0][i] = qx[x];
            quant[1][i] = qy[x];
            quant[2][i] = qb[x];
            let yv = qy[x] as f32 * steps[1];
            recon[0][i] = qx[x] as f32 * steps[0];
            recon[1][i] = yv;
            recon[2][i] = qb[x] as f32 * steps[2] + r_b * yv;
        }
    }
    let mut cache = vec![0.0f64; w * h];
    let mut active = vec![false; w * h];

    let sh = Shared {
        w,
        h,
        steps,
        inv_step2,
        r_b,
        target: [
            target.plane_data(0),
            target.plane_data(1),
            target.plane_data(2),
        ],
        recon: [
            recon[0].as_mut_ptr(),
            recon[1].as_mut_ptr(),
            recon[2].as_mut_ptr(),
        ],
        quant: [
            quant[0].as_mut_ptr(),
            quant[1].as_mut_ptr(),
            quant[2].as_mut_ptr(),
        ],
        cache: cache.as_mut_ptr(),
        active: active.as_mut_ptr(),
    };

    // SAFETY: single-threaded initialization through the freshly created
    // pointers; nothing else accesses the buffers.
    unsafe {
        for y in 0..h {
            for x in 0..w {
                *sh.cache.add(y * w + x) = output_err(&sh, x, y);
            }
        }
        // Initial active set: pixels whose 3×3 neighborhood touches an
        // openable smoothing gate.
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if max_gap_at(&sh, x, y) < ACTIVE_GAP {
                    for ny in y - 1..=y + 1 {
                        *sh.active.add(ny * w + x - 1) = true;
                        *sh.active.add(ny * w + x) = true;
                        *sh.active.add(ny * w + x + 1) = true;
                    }
                }
            }
        }
    }

    // Row bands, ≥2 rows each (required by the cache-spill disjointness
    // argument), alternating between the two phases. The band height is a
    // constant so the partition — and therefore the descent result — is
    // identical for every thread count, including the serial path.
    const BAND_H: usize = 16;
    let band_h = BAND_H;
    let mut bounds: Vec<(usize, usize)> = (0..h.div_ceil(band_h))
        .map(|i| (i * band_h, ((i + 1) * band_h).min(h)))
        .collect();
    if let [.., prev, last] = bounds.as_mut_slice() {
        if last.1 - last.0 < 2 {
            prev.1 = last.1;
            bounds.pop();
        }
    }
    let mut phases: [Vec<Band>; 2] = [Vec::new(), Vec::new()];
    for (i, &(y0, y1)) in bounds.iter().enumerate() {
        phases[i % 2].push(Band {
            y0,
            y1,
            touched: Vec::new(),
            changed: 0,
        });
    }

    let mut changed_total = 0u32;
    for _sweep in 0..MAX_SWEEPS {
        for phase in &mut phases {
            match (&mut parallel, phase.len() > 1) {
                (Some((pool, scratch)), true) => {
                    pool.steal_for_each_mut(scratch, phase, |_, band, _| {
                        // SAFETY: bands within a phase are separated by a
                        // frozen ≥2-row band — the schedule on `Shared`.
                        band.changed =
                            unsafe { sweep_band(&sh, band.y0, band.y1, &mut band.touched) };
                    });
                }
                _ => {
                    for band in phase.iter_mut() {
                        // SAFETY: serial execution; no concurrent access.
                        band.changed =
                            unsafe { sweep_band(&sh, band.y0, band.y1, &mut band.touched) };
                    }
                }
            }
        }
        let changed: u32 = phases.iter().flatten().map(|b| b.changed).sum();
        changed_total += changed;
        if changed == 0 {
            break;
        }
        // Next sweep only revisits the neighborhoods of accepted moves.
        // SAFETY: single-threaded between phases/sweeps.
        unsafe {
            for i in 0..w * h {
                *sh.active.add(i) = false;
            }
            for band in phases.iter_mut().flatten() {
                for &i in &band.touched {
                    let (x, y) = (i as usize % w, i as usize / w);
                    for ny in y.saturating_sub(1)..(y + 2).min(h) {
                        for nx in x.saturating_sub(1)..(x + 2).min(w) {
                            *sh.active.add(ny * w + nx) = true;
                        }
                    }
                }
                band.touched.clear();
                band.changed = 0;
            }
        }
    }

    if changed_total > 0 {
        for c in 0..3 {
            for y in 0..h {
                quant_img.plane_row_mut(c, y)[..w].copy_from_slice(&quant[c][y * w..y * w + w]);
            }
        }
    }
    changed_total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_err(quant: &Image3S, target: &Image3F, steps: [f32; 3], r_b: f32) -> f64 {
        let (w, h) = (quant.xsize(), quant.ysize());
        let mut recon: [Vec<f32>; 3] = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
        for y in 0..h {
            for x in 0..w {
                let yv = quant.plane_row(1, y)[x] as f32 * steps[1];
                recon[0][y * w + x] = quant.plane_row(0, y)[x] as f32 * steps[0];
                recon[1][y * w + x] = yv;
                recon[2][y * w + x] = quant.plane_row(2, y)[x] as f32 * steps[2] + r_b * yv;
            }
        }
        let inv_step2 = [
            1.0 / (steps[0] * steps[0]),
            1.0 / (steps[1] * steps[1]),
            1.0 / (steps[2] * steps[2]),
        ];
        let sh = Shared {
            w,
            h,
            steps,
            inv_step2,
            r_b,
            target: [
                target.plane_data(0),
                target.plane_data(1),
                target.plane_data(2),
            ],
            recon: [
                recon[0].as_mut_ptr(),
                recon[1].as_mut_ptr(),
                recon[2].as_mut_ptr(),
            ],
            quant: [std::ptr::null_mut(); 3],
            cache: std::ptr::null_mut(),
            active: std::ptr::null_mut(),
        };
        let mut sum = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                // SAFETY: recon/target pointers are live; quant/cache/active
                // are unused by output_err.
                sum += unsafe { output_err(&sh, x, y) };
            }
        }
        sum
    }

    /// Quantize a smooth ramp, optimize, and check the smoothed-recon error
    /// strictly drops while raw nearest-rounding error would have been left
    /// untouched by construction.
    #[test]
    fn ramp_error_improves_and_terminates() {
        let (w, h) = (24, 16);
        let steps = [0.6f32, 0.25, 1.0];
        let mut target = Image3F::new(w, h);
        let mut quant = Image3S::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    // Gentle diagonal ramp: ~1 step per 7 px.
                    let v = (x as f32 + 0.5 * y as f32) * steps[c] / 7.0;
                    target.plane_row_mut(c, y)[x] = v;
                }
            }
        }
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = target.plane_row(c, y)[x];
                    quant.plane_row_mut(c, y)[x] = (v / steps[c]).round() as i16;
                }
            }
        }
        // r_b = 1 couples B to Y; store B as residual like the encoder does.
        let r_b = 1.0f32;
        for y in 0..h {
            for x in 0..w {
                let yv = quant.plane_row(1, y)[x] as f32 * steps[1];
                let res = (target.plane_row(2, y)[x] - r_b * yv) / steps[2];
                quant.plane_row_mut(2, y)[x] = res.round() as i16;
            }
        }

        let before = total_err(&quant, &target, steps, r_b);
        let changed = optimize_dc_rounding(&mut quant, &target, steps, r_b, None);
        let after = total_err(&quant, &target, steps, r_b);
        assert!(changed > 0, "optimizer found no moves on a ramp");
        assert!(
            after < before * 0.9,
            "smoothed-recon error should drop ≥10%: before {before} after {after}"
        );

        // Idempotence: a second run finds nothing.
        let again = optimize_dc_rounding(&mut quant, &target, steps, r_b, None);
        assert_eq!(again, 0, "descent did not reach a fixed point");
    }

    /// A constant plane quantizes exactly: sm == mc everywhere, and the pass
    /// must find nothing to move.
    #[test]
    fn constant_plane_is_left_alone() {
        let (w, h) = (16, 12);
        let steps = [0.5f32, 0.5, 0.5];
        let mut target = Image3F::new(w, h);
        let mut quant = Image3S::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    target.plane_row_mut(c, y)[x] = 3.0 * steps[c];
                    quant.plane_row_mut(c, y)[x] = 3;
                }
            }
        }
        let changed = optimize_dc_rounding(&mut quant, &target, steps, 0.0, None);
        assert_eq!(changed, 0, "constant plane should already be optimal");
    }

    /// Parallel banding must produce the identical plane as the serial path:
    /// the band partition is thread-count-dependent but the two-phase
    /// schedule within it is deterministic, and a mismatch against the
    /// single-band serial result beyond tiny order effects would indicate a
    /// race. Uses a target mixing gradient and noise so both accept and
    /// reject paths run.
    #[test]
    fn parallel_matches_serial_objective() {
        let (w, h) = (40, 33);
        let steps = [0.6f32, 0.25, 1.0];
        let mut target = Image3F::new(w, h);
        let mut state = 0x1234_5678u32;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = ((state >> 16) as f32 / 65536.0 - 0.5) * 0.4;
                    let v = (x as f32 * 0.4 + y as f32 * 0.2 + noise) * steps[c] / 3.0;
                    target.plane_row_mut(c, y)[x] = v;
                }
            }
        }
        let mut quant_serial = Image3S::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    quant_serial.plane_row_mut(c, y)[x] =
                        (target.plane_row(c, y)[x] / steps[c]).round() as i16;
                }
            }
        }
        let mut quant_parallel = Image3S::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                quant_parallel
                    .plane_row_mut(c, y)
                    .copy_from_slice(quant_serial.plane_row(c, y));
            }
        }

        let base = total_err(&quant_serial, &target, steps, 0.3);
        optimize_dc_rounding(&mut quant_serial, &target, steps, 0.3, None);
        let serial = total_err(&quant_serial, &target, steps, 0.3);

        let pool = ThreadPool::new(4);
        let mut scratch = CoderScratch::default();
        optimize_dc_rounding(
            &mut quant_parallel,
            &target,
            steps,
            0.3,
            Some((&pool, &mut scratch)),
        );

        assert!(serial < base, "serial descent did not improve");
        // The band partition is thread-count independent and bands within a
        // phase have disjoint state footprints, so the parallel result must
        // be bit-identical to the serial one; any divergence is a race or a
        // halo bug.
        for c in 0..3 {
            for y in 0..h {
                assert_eq!(
                    quant_serial.plane_row(c, y),
                    quant_parallel.plane_row(c, y),
                    "parallel/serial mismatch at c={c} y={y}"
                );
            }
        }
    }
}
