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

//! Arithmetic-coded JPEG (SOF9 / SOF10) support.
//!
//! The QM decoder and coefficient models are ported from IJG/libjpeg-turbo's
//! `jdarith.c` (developed 1997-2015 by Guido Vollbeding; libjpeg-turbo
//! modifications by D. R. Commander), with the Table D.2 layout shared with
//! the jhm17 arithmetic encoder. This software is based in part on the work
//! of the Independent JPEG Group.
//!
//! JPEG XL's reconstruction data can only describe Huffman-coded files, so
//! the original arithmetic bytes cannot be rebuilt. When reconstruction is
//! requested, [`synthesize_huffman`] rewrites the frame as the equivalent
//! Huffman JPEG with the same quantized coefficients and quantization values.
//! Huffman tables and restart markers are regenerated for `jbrd`.

use super::parse::JpegError;
use super::{
    DCT_BLOCK_SIZE, HUFF_ALPHABET_SIZE, HUFF_MAX_BIT_LENGTH, JpegComponentScanInfo, JpegData,
    JpegHuffmanCode, JpegScanInfo, MAX_COMPONENTS, NATURAL_ORDER,
};

/// Table D.2: Qe in bits 16..31, the next MPS state in bits 8..15, and the
/// next LPS state with the MPS-switch flag in bits 0..7 (libjpeg's
/// `jpeg_aritab` layout).
static QM: [u32; 114] = [
    0x5a1d0181, 0x2586020e, 0x11140310, 0x080b0412, 0x03d80514, 0x01da0617, 0x00e50719, 0x006f081c,
    0x0036091e, 0x001a0a21, 0x000d0b23, 0x00060c09, 0x00030d0a, 0x00010d0c, 0x5a7f0f8f, 0x3f251024,
    0x2cf21126, 0x207c1227, 0x17b91328, 0x1182142a, 0x0cef152b, 0x09a1162d, 0x072f172e, 0x055c1830,
    0x04061931, 0x03031a33, 0x02401b34, 0x01b11c36, 0x01441d38, 0x00f51e39, 0x00b71f3b, 0x008a203c,
    0x0068213e, 0x004e223f, 0x003b2320, 0x002c0921, 0x5ae125a5, 0x484c2640, 0x3a0d2741, 0x2ef12843,
    0x261f2944, 0x1f332a45, 0x19a82b46, 0x15182c48, 0x11772d49, 0x0e742e4a, 0x0bfb2f4b, 0x09f8304d,
    0x0861314e, 0x0706324f, 0x05cd3330, 0x04de3432, 0x040f3532, 0x03633633, 0x02d43734, 0x025c3835,
    0x01f83936, 0x01a43a37, 0x01603b38, 0x01253c39, 0x00f63d3a, 0x00cb3e3b, 0x00ab3f3d, 0x008f203d,
    0x5b1241c1, 0x4d044250, 0x412c4351, 0x37d84452, 0x2fe84553, 0x293c4654, 0x23794756, 0x1edf4857,
    0x1aa94957, 0x174e4a48, 0x14244b48, 0x119c4c4a, 0x0f6b4d4a, 0x0d514e4b, 0x0bb64f4d, 0x0a40304d,
    0x583251d0, 0x4d1c5258, 0x438e5359, 0x3bdd545a, 0x34ee555b, 0x2eae565c, 0x299a575d, 0x25164756,
    0x557059d8, 0x4ca95a5f, 0x44d95b60, 0x3e225c61, 0x38245d63, 0x32b45e63, 0x2e17565d, 0x56a860df,
    0x4f466165, 0x47e56266, 0x41cf6367, 0x3c3d6468, 0x375e5d63, 0x52316669, 0x4c0f676a, 0x4639686b,
    0x415e6367, 0x56276ae9, 0x50e76b6c, 0x4b85676d, 0x55976d6e, 0x504f6b6f, 0x5a106fee, 0x55226d70,
    0x59eb6ff0, 0x5a1d7171,
];

const DC_STAT_BINS: usize = 64;
const AC_STAT_BINS: usize = 256;
/// Statistics bin index of the fixed 0.5 probability estimate.
const FIXED_STATE: u8 = 113;
/// Largest quantized coefficient of an 8-bit DCT image (T.81 F.1.1.4).
const MAX_COEFF: i32 = 1023;
const MIN_DC_COEFF: i32 = -1024;

/// Conditioning parameters set by DAC segments (defaults per T.81 F.1.4.4).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArithConditioning {
    pub(crate) dc_l: [u8; 4],
    pub(crate) dc_u: [u8; 4],
    pub(crate) ac_k: [u8; 4],
}

impl Default for ArithConditioning {
    fn default() -> Self {
        Self {
            dc_l: [0; 4],
            dc_u: [1; 4],
            ac_k: [5; 4],
        }
    }
}

/// Parses a DAC segment into `cond`.
pub(crate) fn process_dac(
    data: &[u8],
    pos: usize,
    cond: &mut ArithConditioning,
) -> Result<usize, JpegError> {
    const M_DAC: u8 = 0xCC;
    if pos + 1 >= data.len() {
        return Err(JpegError::Truncated);
    }
    let len = ((data[pos] as usize) << 8) | data[pos + 1] as usize;
    let end = pos + len;
    if len < 4 || (len - 2) % 2 != 0 || end > data.len() {
        return Err(JpegError::BadSegmentLength(M_DAC));
    }
    let mut p = pos + 2;
    while p < end {
        let index = data[p];
        let val = data[p + 1];
        p += 2;
        let tb = (index & 0x0F) as usize;
        if tb > 3 {
            return Err(JpegError::BadTable("arithmetic conditioning"));
        }
        match index >> 4 {
            0 => {
                let (l, u) = (val & 0x0F, val >> 4);
                if l > u {
                    return Err(JpegError::BadTable("arithmetic conditioning"));
                }
                cond.dc_l[tb] = l;
                cond.dc_u[tb] = u;
            }
            1 if (1..=63).contains(&val) => cond.ac_k[tb] = val,
            _ => return Err(JpegError::BadTable("arithmetic conditioning")),
        }
    }
    Ok(end)
}

/// The QM-coder register state plus its input position (T.81 Annex D,
/// with libjpeg's floating C-register layout).
struct QmDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    c: i64,
    a: i64,
    ct: i32,
    /// Set once a marker was reached; zeros are supplied from then on.
    hit_marker: bool,
    /// Position of the `FF` that introduces the marker that was reached.
    marker_pos: usize,
}

impl<'a> QmDecoder<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            c: 0,
            a: 0,
            // Forces two initial bytes into C before the first decision.
            ct: -16,
            hit_marker: false,
            marker_pos: pos,
        }
    }

    fn next_byte(&mut self) -> i64 {
        if self.hit_marker {
            return 0;
        }
        let Some(&b) = self.data.get(self.pos) else {
            self.hit_marker = true;
            self.marker_pos = self.data.len();
            return 0;
        };
        self.pos += 1;
        if b != 0xFF {
            return b as i64;
        }
        // Swallow fill bytes; a stuffed zero stands for a literal 0xFF.
        loop {
            match self.data.get(self.pos) {
                Some(0xFF) => self.pos += 1,
                Some(0x00) => {
                    self.pos += 1;
                    return 0xFF;
                }
                // Hitting a marker is legal in arithmetic coding: zeros are
                // supplied until decoding is complete.
                Some(_) => {
                    self.hit_marker = true;
                    self.marker_pos = self.pos - 1;
                    return 0;
                }
                None => {
                    self.hit_marker = true;
                    self.marker_pos = self.data.len();
                    return 0;
                }
            }
        }
    }

    /// Decodes one binary decision with the adaptive estimate in `st`.
    #[inline]
    fn decode(&mut self, st: &mut u8) -> bool {
        // Renormalization and data input per D.2.6.
        while self.a < 0x8000 {
            self.ct -= 1;
            if self.ct < 0 {
                let data = self.next_byte();
                self.c = (self.c << 8) | data;
                self.ct += 8;
                if self.ct < 0 {
                    self.ct += 1;
                    if self.ct == 0 {
                        // Two initial bytes read: A becomes 0x10000 below.
                        self.a = 0x8000;
                    }
                }
            }
            self.a <<= 1;
        }

        let sv = *st;
        let entry = QM[(sv & 0x7F) as usize];
        let nl = (entry & 0xFF) as u8;
        let nm = ((entry >> 8) & 0xFF) as u8;
        let qe = (entry >> 16) as i64;

        // Decoding and estimation per D.2.4 and D.2.5.
        let temp = self.a - qe;
        self.a = temp;
        let temp = temp << self.ct;
        let mut sv = sv;
        if self.c >= temp {
            self.c -= temp;
            // Conditional LPS exchange.
            if self.a < qe {
                self.a = qe;
                *st = (sv & 0x80) ^ nm;
            } else {
                self.a = qe;
                *st = (sv & 0x80) ^ nl;
                sv ^= 0x80;
            }
        } else if self.a < 0x8000 {
            // Conditional MPS exchange.
            if self.a < qe {
                *st = (sv & 0x80) ^ nl;
                sv ^= 0x80;
            } else {
                *st = (sv & 0x80) ^ nm;
            }
        }
        sv >> 7 != 0
    }

    /// Position of the next marker's `FF`, skipping whatever entropy-coded
    /// bytes were left unread.
    fn marker_position(&self) -> usize {
        if self.hit_marker {
            return self.marker_pos;
        }
        let mut p = self.pos;
        while p + 1 < self.data.len() {
            if self.data[p] == 0xFF && self.data[p + 1] != 0x00 && self.data[p + 1] != 0xFF {
                return p;
            }
            p += 1;
        }
        self.data.len()
    }
}

/// Adaptive statistics for one scan, indexed by conditioning table.
struct ArithStats {
    dc: [[u8; DC_STAT_BINS]; 4],
    ac: [[u8; AC_STAT_BINS]; 4],
    fixed: u8,
    last_dc: [i32; MAX_COMPONENTS],
    dc_context: [usize; MAX_COMPONENTS],
}

impl ArithStats {
    fn new() -> Self {
        Self {
            dc: [[0; DC_STAT_BINS]; 4],
            ac: [[0; AC_STAT_BINS]; 4],
            fixed: FIXED_STATE,
            last_dc: [0; MAX_COMPONENTS],
            dc_context: [0; MAX_COMPONENTS],
        }
    }

    /// Zeroes the statistics the scan uses, as at scan start and after each
    /// restart marker.
    fn reset(&mut self, scan: &JpegScanInfo, progressive: bool) {
        for ci in 0..scan.num_components as usize {
            let csi = scan.components[ci];
            if !progressive || (scan.ss == 0 && scan.ah == 0) {
                self.dc[csi.dc_tbl_idx as usize] = [0; DC_STAT_BINS];
                self.last_dc[ci] = 0;
                self.dc_context[ci] = 0;
            }
            if !progressive || scan.ss != 0 {
                self.ac[csi.ac_tbl_idx as usize] = [0; AC_STAT_BINS];
            }
        }
    }
}

/// A block visited in scan order.
#[derive(Clone, Copy)]
pub(crate) struct ScanBlock {
    /// Position of the component inside the scan header.
    pub(crate) scan_comp: usize,
    pub(crate) comp_idx: usize,
    /// Block index within the component's coefficient plane.
    pub(crate) block: usize,
}

/// What [`walk_scan`] visits, in coding order.
#[derive(Clone, Copy)]
pub(crate) enum ScanEvent {
    /// A restart marker precedes the next restart interval.
    Restart,
    Block(ScanBlock),
}

/// Walks a scan in coding order: every block, with a restart event before
/// each restart interval but the first.
pub(crate) fn walk_scan<E>(
    jpg: &JpegData,
    scan: &JpegScanInfo,
    restart_interval: u32,
    mut visit: impl FnMut(ScanEvent) -> Result<(), E>,
) -> Result<(), E> {
    let ns = scan.num_components as usize;
    let max_h = jpg.max_h_samp();
    let max_v = jpg.max_v_samp();
    let (units_x, units_y) = if ns == 1 {
        let c = &jpg.components[scan.components[0].comp_idx as usize];
        (
            (jpg.width * c.h_samp_factor).div_ceil(8 * max_h),
            (jpg.height * c.v_samp_factor).div_ceil(8 * max_v),
        )
    } else {
        (
            jpg.width.div_ceil(max_h * 8),
            jpg.height.div_ceil(max_v * 8),
        )
    };
    let mut to_go = restart_interval;
    for uy in 0..units_y {
        for ux in 0..units_x {
            if restart_interval > 0 {
                if to_go == 0 {
                    visit(ScanEvent::Restart)?;
                    to_go = restart_interval;
                }
                to_go -= 1;
            }
            if ns == 1 {
                let comp_idx = scan.components[0].comp_idx as usize;
                let bw = jpg.components[comp_idx].width_in_blocks;
                visit(ScanEvent::Block(ScanBlock {
                    scan_comp: 0,
                    comp_idx,
                    block: uy * bw + ux,
                }))?;
            } else {
                for i in 0..ns {
                    let comp_idx = scan.components[i].comp_idx as usize;
                    let c = &jpg.components[comp_idx];
                    let (h, v, bw) = (c.h_samp_factor, c.v_samp_factor, c.width_in_blocks);
                    for by in 0..v {
                        for bx in 0..h {
                            visit(ScanEvent::Block(ScanBlock {
                                scan_comp: i,
                                comp_idx,
                                block: (uy * v + by) * bw + ux * h + bx,
                            }))?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Decodes one arithmetic-coded scan starting at `start`, returning the
/// position of the marker that follows it.
pub(crate) fn decode_scan(
    data: &[u8],
    start: usize,
    jpg: &mut JpegData,
    scan: &JpegScanInfo,
    cond: &ArithConditioning,
) -> Result<usize, JpegError> {
    let progressive = jpg.is_progressive;
    let mut stats = ArithStats::new();
    stats.reset(scan, progressive);

    // The walk only reads the frame layout; coefficients are written through
    // a separate borrow of the planes.
    let mut planes: Vec<Vec<i16>> = jpg
        .components
        .iter_mut()
        .map(|c| core::mem::take(&mut c.coeffs))
        .collect();
    let mut dec = QmDecoder::new(data, start);
    let mut restart_index = 0;

    let result = walk_scan(jpg, scan, jpg.restart_interval, |event| {
        let b = match event {
            ScanEvent::Restart => {
                let p = dec.marker_position();
                if data.get(p + 1) != Some(&(0xD0 + restart_index)) {
                    return Err(JpegError::BadEntropyData);
                }
                restart_index = (restart_index + 1) % 8;
                dec = QmDecoder::new(data, p + 2);
                stats.reset(scan, progressive);
                return Ok(());
            }
            ScanEvent::Block(b) => b,
        };
        let base = b.block * DCT_BLOCK_SIZE;
        let coeffs = planes[b.comp_idx]
            .get_mut(base..base + DCT_BLOCK_SIZE)
            .ok_or(JpegError::BadEntropyData)?;
        let csi = scan.components[b.scan_comp];
        if !progressive {
            decode_dc_first(&mut dec, &mut stats, cond, csi, b.scan_comp, 0, coeffs)?;
            decode_ac_first(&mut dec, &mut stats, cond, csi, 1, 63, 0, coeffs)
        } else if scan.ah == 0 {
            if scan.ss == 0 {
                decode_dc_first(
                    &mut dec,
                    &mut stats,
                    cond,
                    csi,
                    b.scan_comp,
                    scan.al,
                    coeffs,
                )
            } else {
                let (ss, se, al) = (scan.ss as usize, scan.se as usize, scan.al);
                decode_ac_first(&mut dec, &mut stats, cond, csi, ss, se, al, coeffs)
            }
        } else if scan.ss == 0 {
            let bit = i32::from(dec.decode(&mut stats.fixed)) << scan.al;
            coeffs[0] = checked_dc_coefficient(i32::from(coeffs[0]) | bit, scan.al)?;
            Ok(())
        } else {
            let (ss, se, al) = (scan.ss as usize, scan.se as usize, scan.al);
            decode_ac_refine(&mut dec, &mut stats, csi, ss, se, al, coeffs)
        }
    });

    for (c, plane) in jpg.components.iter_mut().zip(planes) {
        c.coeffs = plane;
    }
    result?;
    let p = dec.marker_position();
    match data.get(p + 1) {
        None => Err(JpegError::Truncated),
        Some(0xD0..=0xD7) => Err(JpegError::BadEntropyData),
        Some(_) => Ok(p),
    }
}

/// Decodes the magnitude category and bit pattern shared by DC and AC values
/// (Figures F.23 and F.24), with `st` at the first magnitude bin and `x1` the
/// first bin of the category ladder. Returns `|v| - 1`.
#[inline]
fn decode_magnitude(
    dec: &mut QmDecoder<'_>,
    bins: &mut [u8],
    mut st: usize,
    x1: usize,
    ac: bool,
) -> Result<(i32, usize), JpegError> {
    let mut m: i32 = dec.decode(&mut bins[st]) as i32;
    if m != 0 {
        let ladder = if ac {
            // AC re-tests the same bin before switching to the X ladder.
            if dec.decode(&mut bins[st]) {
                m <<= 1;
                st = x1;
                true
            } else {
                false
            }
        } else {
            st = x1;
            true
        };
        if ladder {
            while dec.decode(bins.get_mut(st).ok_or(JpegError::BadEntropyData)?) {
                m <<= 1;
                if m == 0x8000 {
                    return Err(JpegError::BadEntropyData);
                }
                st += 1;
            }
        }
    }
    let mut v = m;
    st += 14;
    let mut bit = m >> 1;
    while bit != 0 {
        if dec.decode(bins.get_mut(st).ok_or(JpegError::BadEntropyData)?) {
            v |= bit;
        }
        bit >>= 1;
    }
    Ok((v, m as usize))
}

fn checked_dc_coefficient(value: i32, al: u32) -> Result<i16, JpegError> {
    // DC successive approximation rounds negative values down to a multiple
    // of 2^Al, so the lower bound can be below -1024 at coarse precision.
    let min = (MIN_DC_COEFF >> al) << al;
    if !(min..=MAX_COEFF).contains(&value) {
        return Err(JpegError::BadEntropyData);
    }
    Ok(value as i16)
}

fn checked_ac_coefficient(value: i32) -> Result<i16, JpegError> {
    // AC successive approximation truncates the magnitude (T.81 A.4).
    if !(-MAX_COEFF..=MAX_COEFF).contains(&value) {
        return Err(JpegError::BadEntropyData);
    }
    Ok(value as i16)
}

/// DC first pass (sequential DC, or progressive DC with Ah = 0).
fn decode_dc_first(
    dec: &mut QmDecoder<'_>,
    stats: &mut ArithStats,
    cond: &ArithConditioning,
    csi: JpegComponentScanInfo,
    ci: usize,
    al: u32,
    coeffs: &mut [i16],
) -> Result<(), JpegError> {
    let tbl = csi.dc_tbl_idx as usize;
    let bins = &mut stats.dc[tbl];
    let s0 = stats.dc_context[ci];
    if !dec.decode(&mut bins[s0]) {
        stats.dc_context[ci] = 0;
    } else {
        let sign = dec.decode(&mut bins[s0 + 1]);
        let st = s0 + 2 + sign as usize;
        let (v, m) = decode_magnitude(dec, bins, st, 20, false)?;
        // F.1.4.4.1.2: conditioning category for the next difference.
        let s = sign as usize;
        stats.dc_context[ci] = if m < (1usize << cond.dc_l[tbl]) >> 1 {
            0
        } else if m > (1usize << cond.dc_u[tbl]) >> 1 {
            12 + s * 4
        } else {
            4 + s * 4
        };
        let v = if sign { -(v + 1) } else { v + 1 };
        stats.last_dc[ci] += v;
    }
    // decode_magnitude caps |v| at 2^15 and scan validation caps Al at 13,
    // so the scaled predictor fits i32 before checking the coefficient range.
    coeffs[0] = checked_dc_coefficient(stats.last_dc[ci] << al, al)?;
    Ok(())
}

/// AC first pass (sequential AC, or progressive AC with Ah = 0).
#[allow(clippy::too_many_arguments)]
fn decode_ac_first(
    dec: &mut QmDecoder<'_>,
    stats: &mut ArithStats,
    cond: &ArithConditioning,
    csi: JpegComponentScanInfo,
    ss: usize,
    se: usize,
    al: u32,
    coeffs: &mut [i16],
) -> Result<(), JpegError> {
    let tbl = csi.ac_tbl_idx as usize;
    let k_thresh = cond.ac_k[tbl] as usize;
    let mut k = ss;
    while k <= se {
        let mut st = 3 * (k - 1);
        if dec.decode(&mut stats.ac[tbl][st]) {
            break; // EOB
        }
        while !dec.decode(&mut stats.ac[tbl][st + 1]) {
            st += 3;
            k += 1;
            if k > se {
                return Err(JpegError::BadEntropyData);
            }
        }
        let sign = dec.decode(&mut stats.fixed);
        let x1 = if k <= k_thresh { 189 } else { 217 };
        let (v, _) = decode_magnitude(dec, &mut stats.ac[tbl], st + 2, x1, true)?;
        let v = if sign { -(v + 1) } else { v + 1 };
        coeffs[NATURAL_ORDER[k]] = checked_ac_coefficient(v << al)?;
        k += 1;
    }
    Ok(())
}

/// AC successive-approximation refinement.
fn decode_ac_refine(
    dec: &mut QmDecoder<'_>,
    stats: &mut ArithStats,
    csi: JpegComponentScanInfo,
    ss: usize,
    se: usize,
    al: u32,
    coeffs: &mut [i16],
) -> Result<(), JpegError> {
    let tbl = csi.ac_tbl_idx as usize;
    let bit = 1i32 << al;
    // EOBx: the previous stage's end of block.
    let mut kex = se;
    while kex > 0 && coeffs[NATURAL_ORDER[kex]] == 0 {
        kex -= 1;
    }
    let mut k = ss;
    while k <= se {
        let mut st = 3 * (k - 1);
        if k > kex && dec.decode(&mut stats.ac[tbl][st]) {
            break; // EOB
        }
        loop {
            let idx = NATURAL_ORDER[k];
            let cur = coeffs[idx];
            if cur != 0 {
                if dec.decode(&mut stats.ac[tbl][st + 2]) {
                    let delta = if cur < 0 { -bit } else { bit };
                    coeffs[idx] = checked_ac_coefficient(i32::from(cur) + delta)?;
                }
                break;
            }
            if dec.decode(&mut stats.ac[tbl][st + 1]) {
                let sign = dec.decode(&mut stats.fixed);
                coeffs[idx] = checked_ac_coefficient(if sign { -bit } else { bit })?;
                break;
            }
            st += 3;
            k += 1;
            if k > se {
                return Err(JpegError::BadEntropyData);
            }
        }
        k += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Huffman synthesis
// ---------------------------------------------------------------------------

/// Huffman table slot for the `i`-th component of a scan: the first gets
/// slot 0 and the rest share slot 1, the usual luma/chroma split, which keeps
/// sequential files baseline-compatible.
fn slot_for(i: usize) -> u32 {
    (i > 0) as u32
}

/// Number of bits needed for `|v|`.
#[inline]
fn nbits(v: u32) -> u32 {
    32 - v.leading_zeros()
}

/// Symbol counts per table slot for one scan.
struct ScanHistograms {
    dc: [[u64; HUFF_ALPHABET_SIZE]; 2],
    ac: [[u64; HUFF_ALPHABET_SIZE]; 2],
}

impl ScanHistograms {
    const EMPTY: Self = Self {
        dc: [[0; HUFF_ALPHABET_SIZE]; 2],
        ac: [[0; HUFF_ALPHABET_SIZE]; 2],
    };

    fn add(&mut self, other: Self) {
        for (target, source) in self.dc.iter_mut().flatten().zip(other.dc.iter().flatten()) {
            *target += source;
        }
        for (target, source) in self.ac.iter_mut().flatten().zip(other.ac.iter().flatten()) {
            *target += source;
        }
    }
}

/// Mirror of the reconstruction writer's end-of-band state
/// (libjxl `DCTCodingState`): only the symbols matter here.
struct EobRun {
    run: u32,
    slot: usize,
}

impl EobRun {
    fn flush(&mut self, h: &mut ScanHistograms) {
        if self.run > 0 {
            h.ac[self.slot][(nbits(self.run) as usize - 1) << 4] += 1;
            self.run = 0;
        }
    }

    fn buffer(&mut self, slot: usize, h: &mut ScanHistograms) {
        if self.run == 0 {
            self.slot = slot;
        }
        self.run += 1;
        if self.run == 0x7FFF {
            self.flush(h);
        }
    }
}

/// Counts the symbols libjxl's JPEG writer will emit for `scan` (see
/// `EncodeDCTBlockSequential`, `EncodeDCTBlockProgressive` and
/// `EncodeRefinementBits` in `dec_jpeg_data_writer.cc`).
fn scan_histograms(
    jpg: &JpegData,
    scan: &JpegScanInfo,
    restart_interval: u32,
) -> Result<ScanHistograms, JpegError> {
    let mut h = ScanHistograms::EMPTY;
    let progressive = jpg.is_progressive;
    let (ss, se, al) = if progressive {
        (scan.ss as usize, scan.se as usize, scan.al)
    } else {
        (0, 63, 0)
    };
    let refine = progressive && scan.ah != 0;
    let mut last_dc = [0i16; MAX_COMPONENTS];
    let mut eob = EobRun { run: 0, slot: 0 };

    walk_scan(jpg, scan, restart_interval, |event| {
        let b = match event {
            ScanEvent::Restart => {
                eob.flush(&mut h);
                last_dc = [0; MAX_COMPONENTS];
                return Ok(());
            }
            ScanEvent::Block(b) => b,
        };
        let h = &mut h;
        let slot = slot_for(b.scan_comp) as usize;
        let base = b.block * DCT_BLOCK_SIZE;
        let coeffs = &jpg.components[b.comp_idx].coeffs[base..base + DCT_BLOCK_SIZE];
        let last = &mut last_dc[b.comp_idx];
        if !progressive {
            let diff = coeffs[0].wrapping_sub(*last);
            *last = coeffs[0];
            let n = nbits(diff.unsigned_abs() as u32);
            if n > 11 {
                return Err(JpegError::UnsupportedMode("DC difference beyond 11 bits"));
            }
            h.dc[slot][n as usize] += 1;
            let mut r = 0u32;
            for k in 1..64 {
                let v = coeffs[NATURAL_ORDER[k]];
                if v == 0 {
                    r += 1;
                    continue;
                }
                while r > 15 {
                    h.ac[slot][0xF0] += 1;
                    r -= 16;
                }
                let n = nbits(v.unsigned_abs() as u32);
                if n > 15 {
                    return Err(JpegError::UnsupportedMode("AC value beyond 15 bits"));
                }
                h.ac[slot][((r << 4) + n) as usize] += 1;
                r = 0;
            }
            if r > 0 {
                h.ac[slot][0] += 1;
            }
            return Ok(());
        }

        if !refine {
            let mut k0 = ss;
            if ss == 0 {
                let v = coeffs[0] >> al;
                let diff = v.wrapping_sub(*last);
                *last = v;
                let n = nbits(diff.unsigned_abs() as u32);
                if n > 11 {
                    return Err(JpegError::UnsupportedMode("DC difference beyond 11 bits"));
                }
                h.dc[slot][n as usize] += 1;
                k0 = 1;
            }
            if k0 > se {
                return Ok(());
            }
            let mut r = 0u32;
            for k in k0..=se {
                let a = (coeffs[NATURAL_ORDER[k]].unsigned_abs() as u32) >> al;
                if a == 0 {
                    r += 1;
                    continue;
                }
                eob.flush(h);
                while r > 15 {
                    h.ac[slot][0xF0] += 1;
                    r -= 16;
                }
                let n = nbits(a);
                if n > 15 {
                    return Err(JpegError::UnsupportedMode("AC value beyond 15 bits"));
                }
                h.ac[slot][((r << 4) + n) as usize] += 1;
                r = 0;
            }
            if r > 0 {
                eob.buffer(slot, h);
                if ss == 0 {
                    eob.flush(h);
                }
            }
            return Ok(());
        }

        // Refinement: DC bits carry no symbols.
        let k0 = ss.max(1);
        if k0 > se {
            return Ok(());
        }
        let mut abs = [0u32; DCT_BLOCK_SIZE];
        let mut last_one = 0usize;
        for k in k0..=se {
            abs[k] = (coeffs[NATURAL_ORDER[k]].unsigned_abs() as u32) >> al;
            if abs[k] == 1 {
                last_one = k;
            }
        }
        let mut r = 0u32;
        let mut pending_bits = 0u32;
        for k in k0..=se {
            if abs[k] == 0 {
                r += 1;
                continue;
            }
            while r > 15 && k <= last_one {
                eob.flush(h);
                h.ac[slot][0xF0] += 1;
                r -= 16;
                pending_bits = 0;
            }
            if abs[k] > 1 {
                pending_bits += 1;
                continue;
            }
            eob.flush(h);
            h.ac[slot][((r << 4) + 1) as usize] += 1;
            pending_bits = 0;
            r = 0;
        }
        if r > 0 || pending_bits > 0 {
            eob.buffer(slot, h);
            if ss == 0 {
                eob.flush(h);
            }
        }
        Ok(())
    })?;
    eob.flush(&mut h);
    Ok(h)
}

/// Builds a length-limited optimal Huffman code (T.81 Annex K.2, as
/// libjpeg's `jpeg_gen_optimal_table`), returning `(bits, values)` exactly as
/// a DHT segment would carry them.
fn optimal_table(freq_in: &[u64; HUFF_ALPHABET_SIZE]) -> ([u32; HUFF_MAX_BIT_LENGTH + 1], Vec<u8>) {
    const N: usize = HUFF_ALPHABET_SIZE + 1;
    let mut freq = [0u64; N];
    freq[..HUFF_ALPHABET_SIZE].copy_from_slice(freq_in);
    // Reserve one code point so no real symbol gets the all-ones code.
    freq[HUFF_ALPHABET_SIZE] = 1;
    let mut codesize = [0usize; N];
    let mut others = [usize::MAX; N];
    loop {
        // Smallest nonzero frequency, ties to the largest symbol.
        let mut c1 = usize::MAX;
        let mut v = u64::MAX;
        for i in 0..N {
            if freq[i] != 0 && freq[i] <= v {
                v = freq[i];
                c1 = i;
            }
        }
        let mut c2 = usize::MAX;
        v = u64::MAX;
        for i in 0..N {
            if freq[i] != 0 && freq[i] <= v && i != c1 {
                v = freq[i];
                c2 = i;
            }
        }
        if c2 == usize::MAX {
            break;
        }
        freq[c1] += freq[c2];
        freq[c2] = 0;
        let mut c = c1;
        codesize[c] += 1;
        while others[c] != usize::MAX {
            c = others[c];
            codesize[c] += 1;
        }
        others[c] = c2;
        let mut c = c2;
        codesize[c] += 1;
        while others[c] != usize::MAX {
            c = others[c];
            codesize[c] += 1;
        }
    }

    let mut bits = [0u32; 64];
    for &s in &codesize {
        if s != 0 {
            bits[s] += 1;
        }
    }
    // Limit code lengths to 16 bits (Figure K.3).
    for i in (HUFF_MAX_BIT_LENGTH + 1..64).rev() {
        while bits[i] > 0 {
            let mut j = i - 2;
            while bits[j] == 0 {
                j -= 1;
            }
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
    }
    // Drop the reserved code point from the longest length.
    let mut i = HUFF_MAX_BIT_LENGTH;
    while bits[i] == 0 {
        i -= 1;
    }
    bits[i] -= 1;

    let mut values = Vec::new();
    for len in 1..64 {
        for (sym, &s) in codesize.iter().enumerate().take(HUFF_ALPHABET_SIZE) {
            if s == len {
                values.push(sym as u8);
            }
        }
    }
    let mut out = [0u32; HUFF_MAX_BIT_LENGTH + 1];
    out[1..].copy_from_slice(&bits[1..=HUFF_MAX_BIT_LENGTH]);
    (out, values)
}

/// Wraps DHT contents in the reconstruction convention: a sentinel symbol
/// appended at the deepest populated length (see `process_dht`).
pub(crate) fn huffman_code_with_sentinel(
    slot_id: u32,
    counts: &[u32; HUFF_MAX_BIT_LENGTH + 1],
    values: &[u8],
) -> Result<JpegHuffmanCode, JpegError> {
    let mut code = JpegHuffmanCode {
        slot_id,
        is_last: false,
        counts: *counts,
        ..Default::default()
    };
    for (dst, &v) in code.values.iter_mut().zip(values) {
        *dst = v as u32;
    }
    let max_depth = (1..=HUFF_MAX_BIT_LENGTH)
        .rev()
        .find(|&l| code.counts[l] != 0)
        .ok_or(JpegError::BadTable("Huffman"))?;
    code.counts[max_depth] += 1;
    code.values[values.len()] = HUFF_ALPHABET_SIZE as u32;
    Ok(code)
}

fn coded_classes(progressive: bool, scan: &JpegScanInfo) -> [bool; 2] {
    [
        !progressive || (scan.ss == 0 && scan.ah == 0),
        !progressive || scan.ss != 0,
    ]
}

fn append_huffman_tables(
    h: ScanHistograms,
    classes: [bool; 2],
    slots: usize,
    tables: &mut Vec<JpegHuffmanCode>,
) -> Result<(), JpegError> {
    for (class, (hist, used)) in [h.dc, h.ac].into_iter().zip(classes).enumerate() {
        if !used {
            continue;
        }
        for (slot, mut freq) in hist.into_iter().take(slots).enumerate() {
            if freq.iter().all(|&f| f == 0) {
                freq[0] = 1;
            }
            let (bits, values) = optimal_table(&freq);
            let slot_id = ((class as u32) << 4) | slot as u32;
            tables.push(huffman_code_with_sentinel(slot_id, &bits, &values)?);
        }
    }
    tables
        .last_mut()
        .ok_or(JpegError::BadTable("Huffman"))?
        .is_last = true;
    Ok(())
}

/// Rewrites a fully decoded arithmetic-coded frame as the equivalent
/// Huffman-coded one, so `jbrd` can reconstruct a standard JPEG with the
/// same coefficients.
pub(crate) fn synthesize_huffman(jpg: &mut JpegData) -> Result<(), JpegError> {
    const M_DHT: u8 = 0xC4;
    const M_SOF9: u8 = 0xC9;
    const M_SOF10: u8 = 0xCA;
    const M_DAC: u8 = 0xCC;
    const M_SOS: u8 = 0xDA;
    const M_DRI: u8 = 0xDD;

    let sequential_sof = if jpg.quant.iter().any(|q| q.precision != 0) {
        0xC1
    } else {
        0xC0
    };
    // The reconstruction format stores just one interval. A JPEG can
    // redefine it between scans, so regenerate those files without restarts.
    let multiple_dri = jpg
        .marker_order
        .iter()
        .filter(|&&m| m == M_DRI)
        .nth(1)
        .is_some();
    let mut restart_interval = 0;
    let num_tables: usize = jpg
        .scan_info
        .iter()
        .map(|scan| {
            let slots = 1 + usize::from(scan.num_components > 1);
            coded_classes(jpg.is_progressive, scan)
                .into_iter()
                .filter(|&used| used)
                .count()
                * slots
        })
        .sum();
    if num_tables == 0 {
        return Err(JpegError::BadTable("Huffman"));
    }
    // jbrd represents 2 through 89 tables. Share tables for long scan
    // scripts and supply a spare AC table for a grayscale DC-only image.
    let share_tables = !(2..=89).contains(&num_tables);
    let mut shared = ScanHistograms::EMPTY;
    let mut shared_slots = 1;
    let mut marker_order = Vec::with_capacity(jpg.marker_order.len() + jpg.scan_info.len());
    let mut huffman_code = Vec::new();
    let mut scan_index = 0usize;
    for &m in &jpg.marker_order {
        match m {
            M_DAC | M_DHT => {}
            M_DRI if multiple_dri => {}
            M_DRI => {
                restart_interval = jpg.restart_interval;
                marker_order.push(m);
            }
            M_SOF9 => marker_order.push(sequential_sof),
            M_SOF10 => marker_order.push(0xC2),
            M_SOS => {
                let scan = jpg
                    .scan_info
                    .get(scan_index)
                    .ok_or(JpegError::BadScanHeader)?;
                let ns = scan.num_components as usize;
                let classes = coded_classes(jpg.is_progressive, scan);
                let [codes_dc, codes_ac] = classes;
                if share_tables && scan_index == 0 {
                    marker_order.push(M_DHT);
                }
                if codes_dc || codes_ac {
                    let h = scan_histograms(jpg, scan, restart_interval)?;
                    let slots = 1 + usize::from(ns > 1);
                    if share_tables {
                        shared.add(h);
                        shared_slots = shared_slots.max(slots);
                    } else {
                        append_huffman_tables(h, classes, slots, &mut huffman_code)?;
                        marker_order.push(M_DHT);
                    }
                }
                marker_order.push(M_SOS);
                let scan = &mut jpg.scan_info[scan_index];
                for (i, c) in scan.components.iter_mut().take(ns).enumerate() {
                    let slot = slot_for(i);
                    c.dc_tbl_idx = if codes_dc { slot } else { 0 };
                    c.ac_tbl_idx = if codes_ac { slot } else { 0 };
                }
                scan_index += 1;
            }
            other => marker_order.push(other),
        }
    }
    if share_tables {
        append_huffman_tables(shared, [true, true], shared_slots, &mut huffman_code)?;
    }
    jpg.restart_interval = restart_interval;
    jpg.marker_order = marker_order;
    jpg.huffman_code = huffman_code;
    jpg.padding_bits.clear();
    jpg.has_zero_padding_bit = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// tiny_huff.jpg: 40x24 4:2:0, `jpegtran -copy none` from one source.
    #[rustfmt::skip]
    static TINY_HUFFMAN: &[u8] = &[
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
        0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07,
        0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d, 0x0e, 0x12, 0x10, 0x0d,
        0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
        0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x03, 0x04,
        0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0d, 0x0b, 0x0d, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xff, 0xc0,
        0x00, 0x11, 0x08, 0x00, 0x18, 0x00, 0x28, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11,
        0x01, 0xff, 0xc4, 0x00, 0x1a, 0x00, 0x01, 0x00, 0x02, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x05, 0x01, 0x03, 0x06, 0x07, 0xff, 0xc4, 0x00,
        0x2f, 0x10, 0x00, 0x01, 0x02, 0x03, 0x05, 0x04, 0x0a, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x06, 0x11, 0x04, 0x05, 0x07, 0x21, 0x61, 0x12, 0x51, 0x71,
        0xd1, 0x15, 0x16, 0x17, 0x31, 0x33, 0x34, 0x41, 0x42, 0x44, 0xd2, 0x84, 0x93, 0xc2, 0xff, 0xc4,
        0x00, 0x1a, 0x01, 0x00, 0x02, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x05, 0x06, 0x08, 0xff, 0xc4, 0x00, 0x24, 0x11, 0x00,
        0x01, 0x03, 0x02, 0x05, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x02, 0x91, 0x03, 0x52, 0x11, 0x13, 0x14, 0x22, 0x42, 0x04, 0x12, 0x31, 0x51, 0x53, 0x62,
        0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00, 0xe7, 0x4c,
        0x36, 0xb6, 0x19, 0x71, 0x0e, 0xd9, 0x20, 0xd5, 0x9b, 0x8a, 0xd0, 0xd6, 0x95, 0x75, 0xd4, 0xcb,
        0xec, 0x83, 0x58, 0x90, 0x00, 0xd5, 0xed, 0xe6, 0xb0, 0x24, 0x6b, 0xf6, 0xb9, 0x3a, 0xce, 0x46,
        0x8e, 0x1c, 0xd6, 0xad, 0xda, 0xef, 0x4b, 0xab, 0x9b, 0xd4, 0x51, 0x1c, 0xc2, 0xa9, 0x6e, 0x4b,
        0x39, 0xea, 0xad, 0x4c, 0x8f, 0x30, 0xb4, 0x64, 0xd8, 0x44, 0xe8, 0x1a, 0x7f, 0xa5, 0x07, 0xc9,
        0x93, 0x14, 0x33, 0xe0, 0xc3, 0x70, 0xa6, 0xf6, 0x7d, 0xd4, 0x60, 0xe1, 0xe4, 0x26, 0x6a, 0x68,
        0x5e, 0x25, 0x57, 0x07, 0x11, 0xea, 0x8b, 0x73, 0xe5, 0x69, 0x91, 0xbf, 0x08, 0x9f, 0xd7, 0xf7,
        0x44, 0xbc, 0x4d, 0xa5, 0x34, 0x56, 0xa3, 0xf4, 0x12, 0xbc, 0x50, 0x62, 0xac, 0xd6, 0xd0, 0x40,
        0xbf, 0x2d, 0xa4, 0x6b, 0x18, 0x94, 0xed, 0x56, 0x6c, 0xa5, 0x3a, 0x6e, 0xd8, 0x38, 0x45, 0x28,
        0x8b, 0x19, 0x8b, 0xee, 0x32, 0x95, 0x9e, 0x2c, 0x6c, 0x20, 0xc5, 0x49, 0xb9, 0xb9, 0x8b, 0xf6,
        0xde, 0x38, 0xc7, 0x71, 0x50, 0x7e, 0x2a, 0x4d, 0x8e, 0x7e, 0xd1, 0xbf, 0x6d, 0xf5, 0xee, 0xf3,
        0x0e, 0xe6, 0x88, 0x8d, 0xc7, 0x91, 0x94, 0x67, 0x0f, 0x9b, 0x61, 0x44, 0xe2, 0x6c, 0xd3, 0xee,
        0xbe, 0xad, 0xce, 0xfc, 0x82, 0x88, 0x8a, 0x37, 0x5c, 0x65, 0x57, 0x53, 0xf8, 0x6c, 0x2f, 0xff,
        0xd9,
    ];
    /// tiny_ar_seq.jpg: 40x24 4:2:0, `jpegtran -copy none` from one source.
    #[rustfmt::skip]
    static TINY_ARITH_SEQ: &[u8] = &[
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
        0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07,
        0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d, 0x0e, 0x12, 0x10, 0x0d,
        0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
        0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x03, 0x04,
        0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0d, 0x0b, 0x0d, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xff, 0xc9,
        0x00, 0x11, 0x08, 0x00, 0x18, 0x00, 0x28, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11,
        0x01, 0xff, 0xcc, 0x00, 0x0a, 0x00, 0x10, 0x10, 0x05, 0x01, 0x10, 0x11, 0x05, 0xff, 0xda, 0x00,
        0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00, 0xff, 0x00, 0x63, 0x50, 0x5d,
        0x0a, 0x8d, 0xc1, 0x5b, 0xb7, 0x88, 0x19, 0x7c, 0x09, 0xd9, 0x57, 0x5a, 0x20, 0x1e, 0x3c, 0x68,
        0x34, 0x2a, 0xf6, 0x59, 0x7c, 0xbc, 0x4a, 0xf9, 0xe9, 0xee, 0x71, 0x23, 0xe2, 0x72, 0x69, 0x90,
        0xca, 0x8b, 0x62, 0x94, 0x75, 0x9b, 0x7f, 0x6c, 0x3e, 0x02, 0x81, 0x44, 0x59, 0xe0, 0xdb, 0xe8,
        0x6d, 0x5b, 0xbb, 0xea, 0x00, 0xa7, 0x3a, 0x92, 0xfb, 0x55, 0xfb, 0xde, 0xe2, 0x34, 0xd3, 0x91,
        0x18, 0xad, 0x79, 0x84, 0xc0, 0xa8, 0xb9, 0x5f, 0xa6, 0x5b, 0xc8, 0x92, 0x8e, 0x87, 0x20, 0x4e,
        0x8c, 0xd0, 0xee, 0x6a, 0x31, 0x0e, 0x99, 0x87, 0x80, 0xd9, 0xc4, 0x5c, 0x0f, 0x17, 0xa5, 0x36,
        0x34, 0x4e, 0x69, 0x48, 0xdb, 0x1e, 0x9e, 0xf6, 0xa7, 0x6a, 0x72, 0xe5, 0x9a, 0xdf, 0x4d, 0xdb,
        0xb9, 0xbc, 0x7c, 0x84, 0x0e, 0xaf, 0xeb, 0x66, 0xb1, 0xc8, 0xa5, 0xaa, 0x18, 0xa5, 0xef, 0x5d,
        0x93, 0x26, 0x0f, 0x83, 0xa7, 0xbc, 0xb3, 0x18, 0xad, 0x30, 0xa3, 0x33, 0xc0, 0x57, 0xb2, 0x29,
        0x5b, 0x00, 0x6d, 0xfd, 0x4d, 0xb7, 0xc2, 0x67, 0xb4, 0xd0, 0xc2, 0x68, 0x61, 0xb4, 0xe2, 0x94,
        0xe7, 0x23, 0x90, 0xd3, 0xec, 0x3b, 0x48, 0x93, 0xcc, 0x2f, 0x36, 0x98, 0xa1, 0x8b, 0x8c, 0xb4,
        0xaf, 0xbb, 0xf5, 0x6b, 0xe7, 0x9d, 0xc3, 0x56, 0x42, 0x4b, 0x30, 0x2f, 0xa7, 0x3f, 0x11, 0x00,
        0xc9, 0xe0, 0xff, 0xd9,
    ];
    /// tiny_ar_prog_rst.jpg: 40x24 4:2:0, `jpegtran -copy none` from one source.
    #[rustfmt::skip]
    static TINY_ARITH_PROG_RST: &[u8] = &[
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
        0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07,
        0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d, 0x0e, 0x12, 0x10, 0x0d,
        0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
        0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x03, 0x04,
        0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0d, 0x0b, 0x0d, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xff, 0xca,
        0x00, 0x11, 0x08, 0x00, 0x18, 0x00, 0x28, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11,
        0x01, 0xff, 0xcc, 0x00, 0x06, 0x00, 0x10, 0x01, 0x10, 0xff, 0xdd, 0x00, 0x04, 0x00, 0x03, 0xff,
        0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x10, 0x03, 0x10, 0x00, 0x00, 0x01, 0xfe, 0xc5, 0xda,
        0x96, 0x87, 0xff, 0x00, 0x1b, 0x1e, 0x9c, 0x98, 0x1b, 0xff, 0x00, 0xcd, 0x3c, 0x5f, 0xff, 0xd0,
        0xff, 0x00, 0xb2, 0x83, 0x0a, 0xe8, 0x8f, 0x65, 0x74, 0x18, 0xff, 0xcc, 0x00, 0x04, 0x10, 0x05,
        0xff, 0xdd, 0x00, 0x04, 0x00, 0x05, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x05, 0x02,
        0x90, 0x88, 0xff, 0xd0, 0x9c, 0xc7, 0xf7, 0x3c, 0x24, 0xdc, 0xaf, 0x62, 0x66, 0x09, 0x90, 0xff,
        0xd1, 0x86, 0xd4, 0xcd, 0x97, 0x71, 0x77, 0x8c, 0x9b, 0xd0, 0xa3, 0xff, 0xcc, 0x00, 0x04, 0x11,
        0x05, 0xff, 0xdd, 0x00, 0x04, 0x00, 0x03, 0xff, 0xda, 0x00, 0x08, 0x01, 0x03, 0x01, 0x01, 0x3f,
        0x01, 0x10, 0x28, 0x2c, 0xab, 0x33, 0x25, 0x24, 0xff, 0xd0, 0x87, 0x66, 0x12, 0xa2, 0x76, 0x9c,
        0xff, 0xcc, 0x00, 0x04, 0x11, 0x05, 0xff, 0xda, 0x00, 0x08, 0x01, 0x02, 0x01, 0x01, 0x3f, 0x01,
        0x92, 0xa2, 0x9a, 0xa0, 0xff, 0xd0, 0x62, 0x5a, 0x15, 0xd0, 0xff, 0xcc, 0x00, 0x04, 0x10, 0x05,
        0xff, 0xdd, 0x00, 0x04, 0x00, 0x05, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x06, 0x3f, 0x02,
        0xc0, 0xff, 0xd0, 0x58, 0x33, 0x91, 0x0b, 0xeb, 0x82, 0x7b, 0xff, 0xd1, 0x5a, 0x51, 0x22, 0x81,
        0xf3, 0x94, 0x03, 0x57, 0x2b, 0x56, 0x5b, 0x80, 0xff, 0xcc, 0x00, 0x04, 0x10, 0x05, 0xff, 0xda,
        0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3f, 0x21, 0xa0, 0x9e, 0x61, 0xd1, 0x7d, 0x91, 0xff, 0xd0,
        0x08, 0x8a, 0x6a, 0xe6, 0x23, 0x35, 0x23, 0x5a, 0xb6, 0x1c, 0xe2, 0xbb, 0xa2, 0x9b, 0xe2, 0xf0,
        0xff, 0xd1, 0x9a, 0x9f, 0x73, 0x06, 0xa5, 0x8b, 0x30, 0x63, 0x39, 0xe5, 0x3f, 0xec, 0x68, 0xff,
        0xdd, 0x00, 0x04, 0x00, 0x03, 0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00,
        0x00, 0x00, 0x10, 0x04, 0x51, 0x80, 0xff, 0xd0, 0xfb, 0xdd, 0x40, 0xff, 0xcc, 0x00, 0x04, 0x11,
        0x05, 0xff, 0xda, 0x00, 0x08, 0x01, 0x03, 0x01, 0x01, 0x3f, 0x10, 0x6e, 0x15, 0x32, 0xe2, 0x51,
        0x2e, 0xe0, 0xff, 0xd0, 0x13, 0xa0, 0xf3, 0xbe, 0x9d, 0xc5, 0xff, 0xcc, 0x00, 0x04, 0x11, 0x05,
        0xff, 0xda, 0x00, 0x08, 0x01, 0x02, 0x01, 0x01, 0x3f, 0x10, 0x4c, 0x6d, 0x87, 0x1d, 0x1e, 0xff,
        0xd0, 0x35, 0x55, 0x49, 0x2f, 0x45, 0x74, 0x30, 0xff, 0xcc, 0x00, 0x04, 0x10, 0x05, 0xff, 0xdd,
        0x00, 0x04, 0x00, 0x05, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3f, 0x10, 0xa7, 0x89,
        0xea, 0x5b, 0x8e, 0xa5, 0x4e, 0x27, 0x40, 0xb7, 0xa8, 0x80, 0xff, 0xd0, 0x18, 0x7d, 0x72, 0x72,
        0x24, 0x61, 0x1d, 0xd9, 0x79, 0x31, 0x06, 0xa4, 0x7c, 0x10, 0x1a, 0x5f, 0x1f, 0x3b, 0x85, 0xf1,
        0x2e, 0xff, 0xd1, 0xfa, 0x00, 0xd1, 0xa4, 0x25, 0x26, 0x50, 0xe6, 0x0b, 0x25, 0x44, 0x87, 0x7c,
        0x57, 0xd8, 0x2b, 0x1f, 0x27, 0x6b, 0x7c, 0xff, 0xd9,
    ];

    fn coeffs(jpg: &JpegData) -> Vec<Vec<i16>> {
        jpg.components.iter().map(|c| c.coeffs.clone()).collect()
    }

    fn tables(jpg: &JpegData) -> Vec<(u32, Vec<u32>, Vec<u32>)> {
        let mut t: Vec<_> = jpg
            .huffman_code
            .iter()
            .map(|h| {
                let n: u32 = h.counts.iter().sum();
                (
                    h.slot_id,
                    h.counts.to_vec(),
                    h.values[..n as usize].to_vec(),
                )
            })
            .collect();
        t.sort();
        t
    }

    #[test]
    fn arithmetic_files_decode_to_the_huffman_coefficients() {
        let huff = super::super::parse_jpeg(TINY_HUFFMAN).unwrap();
        assert!(!huff.is_arithmetic);
        for file in [TINY_ARITH_SEQ, TINY_ARITH_PROG_RST] {
            let mut jpg = super::super::parse_jpeg(file).unwrap();
            assert!(jpg.is_arithmetic);
            assert_eq!(coeffs(&jpg), coeffs(&huff));
            assert!(jpg.huffman_code.is_empty());
            synthesize_huffman(&mut jpg).unwrap();
            // Reconstruction sees only Huffman-coded markers.
            assert!(
                !jpg.marker_order
                    .iter()
                    .any(|m| matches!(m, 0xC9 | 0xCA | 0xCC))
            );
            // Every coded scan is preceded by its own tables.
            let sos = jpg.marker_order.iter().filter(|&&m| m == 0xDA).count();
            assert_eq!(sos, jpg.scan_info.len());
            assert!(jpg.huffman_code.iter().any(|h| h.is_last));
        }
    }

    #[test]
    fn sequential_tables_match_jpegtran_optimize() {
        // `jpegtran -optimize` builds per-slot optimal tables with the same
        // luma/chroma split, so the synthesized tables come out identical.
        let huff = super::super::parse_jpeg(TINY_HUFFMAN).unwrap();
        let mut arith = super::super::parse_jpeg(TINY_ARITH_SEQ).unwrap();
        synthesize_huffman(&mut arith).unwrap();
        assert_eq!(tables(&arith), tables(&huff));
        assert_eq!(arith.marker_order.iter().filter(|&&m| m == 0xC0).count(), 1);
    }

    #[test]
    fn multiple_restart_intervals_are_normalized_for_reconstruction() {
        let mut jpg = super::super::parse_jpeg(TINY_ARITH_PROG_RST).unwrap();
        assert!(jpg.marker_order.iter().filter(|&&m| m == 0xDD).count() > 1);
        synthesize_huffman(&mut jpg).unwrap();
        assert_eq!(jpg.restart_interval, 0);
        assert!(!jpg.marker_order.contains(&0xDD));
    }

    #[test]
    fn late_restart_interval_does_not_change_earlier_scan_histogram() {
        let mut coeffs = vec![0; 2 * DCT_BLOCK_SIZE];
        coeffs[0] = 400;
        coeffs[DCT_BLOCK_SIZE] = 400;
        let mut jpg = JpegData {
            width: 16,
            height: 8,
            restart_interval: 1,
            is_progressive: true,
            marker_order: vec![0xCA, 0xDA, 0xDD, 0xDA, 0xD9],
            components: vec![super::super::JpegComponent {
                h_samp_factor: 1,
                v_samp_factor: 1,
                width_in_blocks: 2,
                height_in_blocks: 1,
                coeffs,
                ..Default::default()
            }],
            scan_info: vec![
                JpegScanInfo {
                    num_components: 1,
                    ..Default::default()
                },
                JpegScanInfo {
                    num_components: 1,
                    ss: 1,
                    se: 63,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        synthesize_huffman(&mut jpg).unwrap();
        assert_eq!(jpg.restart_interval, 1);
        assert_eq!(jpg.marker_order, [0xC2, 0xC4, 0xDA, 0xDD, 0xC4, 0xDA, 0xD9]);
        assert!(jpg.huffman_code[0].values[..2].contains(&0)); // second DC difference
        assert_eq!(jpg.huffman_code[1].counts[1], 2); // one AC symbol plus sentinel
        assert_eq!(jpg.huffman_code[1].values[0], 0); // one-block EOB runs
    }

    #[test]
    fn sequential_reconstruction_with_wide_quantization_uses_extended_sof() {
        let mut jpg = super::super::parse_jpeg(TINY_ARITH_SEQ).unwrap();
        jpg.quant[0].precision = 1;
        synthesize_huffman(&mut jpg).unwrap();
        assert!(jpg.marker_order.contains(&0xC1));
        assert!(!jpg.marker_order.contains(&0xC0));
    }

    #[test]
    fn long_progressive_scripts_share_huffman_tables() {
        let mut jpg = super::super::parse_jpeg(TINY_ARITH_PROG_RST).unwrap();
        let mut dc = jpg.scan_info[0].clone();
        dc.al = 0;
        jpg.scan_info = vec![dc];
        for comp_idx in 0..3 {
            for k in 1..64 {
                let mut scan = JpegScanInfo {
                    num_components: 1,
                    ss: k,
                    se: k,
                    ..Default::default()
                };
                scan.components[0].comp_idx = comp_idx;
                jpg.scan_info.push(scan);
            }
        }
        jpg.marker_order
            .retain(|&m| !matches!(m, 0xCC | 0xDD | 0xDA | 0xD9));
        jpg.marker_order.extend(jpg.scan_info.iter().map(|_| 0xDA));
        jpg.marker_order.push(0xD9);
        synthesize_huffman(&mut jpg).unwrap();
        assert_eq!(jpg.marker_order.iter().filter(|&&m| m == 0xC4).count(), 1);
        assert_eq!(jpg.huffman_code.len(), 4);
        assert!(super::super::jbrd::encode_jbrd(&jpg, None, None, &[], None).is_ok());
    }

    #[test]
    fn grayscale_dc_only_scan_has_a_representable_table_count() {
        // 17x17 grayscale cjpeg -arithmetic with the scan script
        // `0: 0 0 0 0;`.
        #[rustfmt::skip]
        let dc_only = [
            0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06,
            0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0a, 0x0c, 0x14, 0x0d,
            0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12, 0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d,
            0x1a, 0x1c, 0x1c, 0x20, 0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c, 0x23, 0x1c, 0x1c, 0x28,
            0x37, 0x29, 0x2c, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32,
            0x3c, 0x2e, 0x33, 0x34, 0x32, 0xff, 0xca, 0x00, 0x0b, 0x08, 0x00, 0x11, 0x00, 0x11,
            0x01, 0x01, 0x11, 0x00, 0xff, 0xcc, 0x00, 0x04, 0x00, 0x10, 0xff, 0xda, 0x00, 0x08,
            0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0xc7, 0x7a, 0xd1, 0x99, 0x5c, 0x59, 0xef, 0xd2,
            0xf6, 0x94, 0x98, 0xff, 0xd9,
        ];
        let out = super::super::encode_jpeg_lossless(&dc_only).unwrap();
        assert!(out.windows(4).any(|w| w == b"jbrd"));
    }

    #[test]
    fn arithmetic_files_transcode_with_and_without_reconstruction() {
        let config = super::super::JpegTranscodeConfig::default().with_jpeg_reconstruction(false);
        let huff = super::super::encode_jpeg_lossless_with_config(TINY_HUFFMAN, &config).unwrap();
        for file in [TINY_ARITH_SEQ, TINY_ARITH_PROG_RST] {
            let out = super::super::encode_jpeg_lossless(file).unwrap();
            assert!(out.windows(4).any(|w| w == b"jbrd"));
            let codestream = super::super::encode_jpeg_lossless_with_config(file, &config).unwrap();
            assert_eq!(codestream, huff);
        }
    }

    #[test]
    fn ultra_hdr_with_arithmetic_jpeg_preserves_the_gain_map() {
        let metadata = crate::IsoGainMap {
            gain_map_max_n: [1; 3],
            alternate_hdr_headroom_n: 1,
            ..Default::default()
        }
        .to_metadata()
        .unwrap();
        let append_iso = |jpeg: &mut Vec<u8>, metadata: &[u8]| {
            let tag = b"urn:iso:std:iso:ts:21496:-1\0";
            let len = u16::try_from(2 + tag.len() + metadata.len()).unwrap();
            jpeg.extend_from_slice(&[0xFF, 0xE2]);
            jpeg.extend_from_slice(&len.to_be_bytes());
            jpeg.extend_from_slice(tag);
            jpeg.extend_from_slice(metadata);
        };
        for (base, gain, reconstruct) in [
            (TINY_ARITH_SEQ, TINY_HUFFMAN, false),
            (TINY_HUFFMAN, TINY_ARITH_SEQ, true),
        ] {
            let mut primary = base[..2].to_vec();
            append_iso(&mut primary, &[0; 4]);
            let mut secondary = gain[..2].to_vec();
            append_iso(&mut secondary, &metadata);
            secondary.extend_from_slice(&gain[2..]);

            // Two-entry MP Index IFD, with sizes measured through each EOI
            // and the secondary offset measured from the TIFF header.
            let mut mpf = *b"\xff\xe2\x00\x58MPF\0MM\x00\x2a\x00\x00\x00\x08\x00\x03\
                \xb0\x00\x00\x07\x00\x00\x00\x040100\
                \xb0\x01\x00\x04\x00\x00\x00\x01\x00\x00\x00\x02\
                \xb0\x02\x00\x07\x00\x00\x00\x20\x00\x00\x00\x32\x00\x00\x00\x00\
                \x00\x03\x00\x00\0\0\0\0\0\0\0\0\0\0\0\0\
                \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let primary_size = (primary.len() + mpf.len() + base.len() - 2) as u32;
            let secondary_offset = primary_size - primary.len() as u32 - 8;
            mpf[62..66].copy_from_slice(&primary_size.to_be_bytes());
            mpf[78..82].copy_from_slice(&(secondary.len() as u32).to_be_bytes());
            mpf[82..86].copy_from_slice(&secondary_offset.to_be_bytes());
            primary.extend_from_slice(&mpf);
            primary.extend_from_slice(&base[2..]);
            primary.extend_from_slice(&secondary);

            let out = super::super::encode_jpeg_lossless(&primary).unwrap();
            let mut boxes = out.as_slice();
            let mut has_jbrd = false;
            let mut jhgm = None;
            while !boxes.is_empty() {
                let size = u32::from_be_bytes(boxes[..4].try_into().unwrap()) as usize;
                let (chunk, rest) = boxes.split_at(size);
                match &chunk[4..8] {
                    b"jbrd" => has_jbrd = true,
                    b"jhgm" => jhgm = crate::gain_map::parse_jhgm_bundle(&chunk[8..]),
                    _ => {}
                }
                boxes = rest;
            }
            assert_eq!(has_jbrd, reconstruct);
            let jhgm = jhgm.unwrap();
            assert_eq!(jhgm.metadata, metadata);
            assert!(jhgm.codestream.starts_with(&[0xFF, 0x0A]));
        }
    }

    #[test]
    fn arithmetic_rejects_duplicate_first_scan() {
        let mut data = TINY_ARITH_PROG_RST.to_vec();
        let start = data.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        let end = start
            + data[start..]
                .windows(2)
                .position(|w| w == [0xFF, 0xCC])
                .unwrap();
        let first_scan = data[start..end].to_vec();
        data.splice(end..end, first_scan);
        assert!(matches!(
            super::super::parse_jpeg(&data),
            Err(JpegError::BadScanHeader)
        ));
    }

    #[test]
    fn arithmetic_uses_the_quantization_table_defined_at_first_scan() {
        let mut data = TINY_ARITH_SEQ.to_vec();
        let start = data.windows(2).position(|w| w == [0xFF, 0xDB]).unwrap();
        let end = start + 2 + u16::from_be_bytes([data[start + 2], data[start + 3]]) as usize;
        let mut replacement = data[start..end].to_vec();
        replacement[5] += 1;
        let expected_dc_quant = i32::from(replacement[5]);
        data.splice(end..end, replacement);

        let mut jpg = super::super::parse_jpeg(&data).unwrap();
        assert_eq!(jpg.quant.len(), 2);
        assert_eq!(
            jpg.quant[jpg.components[0].quant_idx as usize].values[0],
            expected_dc_quant
        );
        synthesize_huffman(&mut jpg).unwrap();
        assert_eq!(jpg.marker_order.iter().filter(|&&m| m == 0xDB).count(), 1);
    }

    #[test]
    fn huffman_reconstruction_retains_unused_dac_bytes() {
        let dac = [0xFF, 0xCC, 0, 4, 0, 0x10];
        let mut data = TINY_HUFFMAN.to_vec();
        let start = data.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        data.splice(start..start, dac);
        data.splice(2..2, dac);
        let jpg = super::super::parse_jpeg(&data).unwrap();
        assert_eq!(jpg.inter_marker_data, [dac, dac]);
        assert!(!jpg.marker_order.contains(&0xCC));
        assert!(super::super::encode_jpeg_lossless(&data).is_ok());
    }

    /// Canonical code lengths assigned to each symbol by `bits`/`values`.
    fn lengths(bits: &[u32; HUFF_MAX_BIT_LENGTH + 1], values: &[u8]) -> Vec<(u8, usize)> {
        let mut out = Vec::new();
        let mut it = values.iter();
        for (len, &n) in bits.iter().enumerate() {
            for _ in 0..n {
                out.push((*it.next().unwrap(), len));
            }
        }
        out
    }

    #[test]
    fn optimal_table_reserves_all_ones_and_limits_length() {
        let mut freq = [0u64; HUFF_ALPHABET_SIZE];
        freq[0] = 10;
        let (bits, values) = optimal_table(&freq);
        assert_eq!(values, vec![0]);
        assert_eq!(lengths(&bits, &values), vec![(0, 1)]);

        // Fibonacci frequencies force depths far beyond 16 before limiting.
        let mut freq = [0u64; HUFF_ALPHABET_SIZE];
        let (mut a, mut b) = (1u64, 1u64);
        for f in freq.iter_mut().take(30) {
            *f = a;
            (a, b) = (b, a + b);
        }
        let (bits, values) = optimal_table(&freq);
        assert_eq!(values.len(), 30);
        assert!(bits[1..].iter().sum::<u32>() == 30);
        // Kraft sum strictly below one: the all-ones code stays free.
        let kraft: f64 = (1..=16).map(|l| bits[l] as f64 / (1u64 << l) as f64).sum();
        assert!(kraft < 1.0);
    }

    #[test]
    fn sentinel_lands_at_deepest_length() {
        let mut bits = [0u32; HUFF_MAX_BIT_LENGTH + 1];
        bits[2] = 3;
        let code = huffman_code_with_sentinel(0x10, &bits, &[1, 2, 3]).unwrap();
        assert_eq!(code.counts[2], 4);
        assert_eq!(code.values[3], HUFF_ALPHABET_SIZE as u32);
    }

    #[test]
    fn rejects_empty_dac_and_reserved_table_class() {
        let mut cond = ArithConditioning::default();
        assert!(matches!(
            process_dac(&[0, 2], 0, &mut cond),
            Err(JpegError::BadSegmentLength(0xCC))
        ));
        assert!(matches!(
            process_dac(&[0, 4, 0x20, 5], 0, &mut cond),
            Err(JpegError::BadTable("arithmetic conditioning"))
        ));
    }

    #[test]
    fn arithmetic_scans_use_default_conditioning_and_accept_marker_fill() {
        let original = super::super::parse_jpeg(TINY_ARITH_SEQ).unwrap();
        let mut defaults = TINY_ARITH_SEQ.to_vec();
        let dac = defaults.windows(2).position(|w| w == [0xFF, 0xCC]).unwrap();
        let len = u16::from_be_bytes([defaults[dac + 2], defaults[dac + 3]]) as usize;
        defaults.drain(dac..dac + 2 + len);
        defaults.insert(defaults.len() - 2, 0xFF);
        let decoded = super::super::parse_jpeg(&defaults).unwrap();
        assert_eq!(coeffs(&decoded), coeffs(&original));
    }

    #[test]
    fn arithmetic_restart_markers_are_sequenced_and_only_between_intervals() {
        let mut wrong_sequence = TINY_ARITH_PROG_RST.to_vec();
        let rst = wrong_sequence
            .windows(2)
            .position(|w| w == [0xFF, 0xD0])
            .unwrap();
        wrong_sequence[rst + 1] = 0xD1;
        assert!(matches!(
            super::super::parse_jpeg(&wrong_sequence),
            Err(JpegError::BadEntropyData)
        ));

        let mut unexpected_restart = TINY_ARITH_SEQ.to_vec();
        let eoi = unexpected_restart.len() - 2;
        unexpected_restart.splice(eoi..eoi, [0xFF, 0xD0]);
        assert!(matches!(
            super::super::parse_jpeg(&unexpected_restart),
            Err(JpegError::BadEntropyData)
        ));
    }

    #[test]
    fn coefficient_ranges_account_for_negative_dc_point_transform() {
        assert_eq!(checked_dc_coefficient(-8192, 13).unwrap(), -8192);
        assert_eq!(checked_dc_coefficient(-1024, 0).unwrap(), -1024);
        assert!(checked_dc_coefficient(-2048, 10).is_err());
        assert!(checked_dc_coefficient(1024, 0).is_err());
        assert!(checked_dc_coefficient(65536, 4).is_err());
        assert_eq!(checked_ac_coefficient(-1023).unwrap(), -1023);
        assert!(checked_ac_coefficient(-1024).is_err());
        assert!(checked_ac_coefficient(1024).is_err());
    }
}
