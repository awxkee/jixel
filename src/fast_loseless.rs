/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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
#![allow(clippy::explicit_counter_loop)]

use crate::{
    ColorSpace, EncodeError, Orientation, Primaries, RenderingIntent, TransferFunction, WhitePoint,
};

const K_NUM_RAW_SYMBOLS: usize = 19;
const K_NUM_LZ77: usize = 33;
const K_LZ77_MIN_LENGTH: usize = 7;
const KPAD: usize = 32;
const KCHUNK: usize = 8;
const NPX: usize = 320;
static GROUP_SIZE_OFFSET: [usize; 4] = [0, 1024, 17408, 4211712];
static TOC_BITS: [u32; 4] = [12, 16, 24, 32];

/// Color / display metadata for the fast-lossless encoders.
///
/// Bit depth is intentionally NOT here: [`encode_fast_lossless`] implies 8 bits,
/// and [`encode_fast_lossless_u16`] takes `bits` explicitly. Endianness is also
/// unnecessary, since the high-bit-depth entry consumes native `u16` samples.
#[derive(Debug, Clone)]
pub struct FlMeta {
    /// Whether color channels are premultiplied by alpha.
    pub alpha_premultiplied: bool,
    /// Image orientation (default [`Orientation::Normal`]).
    pub orientation: Orientation,
    pub white_point: WhitePoint,
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub rendering_intent: RenderingIntent,
    /// Optional intrinsic (display) size hint as (width, height).
    pub intrinsic_size: Option<(usize, usize)>,
    /// Embedded ICC profile (compressed via `icc_codec`).
    pub icc: Option<Vec<u8>>,
    /// EXIF/TIFF metadata, embedded via an `Exif` container box. Setting this
    /// forces the output into the JXL container form. Raw TIFF bytes.
    pub exif: Option<Vec<u8>>,
    /// XMP packet, embedded unchanged via an uncompressed `xml ` container box.
    pub xmp: Option<Vec<u8>>,
}
impl Default for FlMeta {
    fn default() -> Self {
        Self {
            alpha_premultiplied: false,
            orientation: Orientation::Normal,
            white_point: WhitePoint::D65,
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            rendering_intent: RenderingIntent::Relative,
            intrinsic_size: None,
            icc: None,
            exif: None,
            xmp: None,
        }
    }
}
impl FlMeta {
    /// sRGB (the default).
    pub fn srgb() -> Self {
        Self::default()
    }
    /// HDR10-style: Rec.2100 primaries + PQ transfer (use with `encode_fast_lossless_u16`).
    pub fn rec2100_pq() -> Self {
        Self {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Smpte2084,
            ..Self::default()
        }
    }
    /// HLG HDR: Rec.2100 primaries + HLG transfer.
    pub fn rec2100_hlg() -> Self {
        Self {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Hlg,
            ..Self::default()
        }
    }
    /// Linear-light, sRGB gamut.
    pub fn linear() -> Self {
        Self {
            transfer: TransferFunction::Linear,
            ..Self::default()
        }
    }
    /// Display-P3 (P3 primaries, sRGB transfer).
    pub fn display_p3() -> Self {
        Self {
            primaries: Primaries::Smpte431,
            ..Self::default()
        }
    }
}

#[inline]
fn store_le64(t: &mut [u8], d: u64) {
    t[..8].copy_from_slice(&d.to_le_bytes());
}
#[inline]
fn add_bits(c: u32, b: u64, buf: &mut [u8], bib: &mut usize, bb: &mut u64) -> usize {
    *bb |= b << *bib;
    *bib += c as usize;
    store_le64(buf, *bb);
    let by = *bib / 8;
    *bib -= by * 8;
    *bb >>= (by * 8) as u32;
    by
}
struct BitWriter {
    data: Vec<u8>,
    bytes_written: usize,
    bits_in_buffer: usize,
    buffer: u64,
}
impl BitWriter {
    fn allocate(mb: usize) -> Self {
        Self {
            data: vec![0u8; mb / 8 + 64],
            bytes_written: 0,
            bits_in_buffer: 0,
            buffer: 0,
        }
    }
    #[inline]
    fn write(&mut self, c: u32, b: u64) {
        let bw = self.bytes_written;
        let n = add_bits(
            c,
            b,
            &mut self.data[bw..],
            &mut self.bits_in_buffer,
            &mut self.buffer,
        );
        self.bytes_written += n;
    }
    fn zero_pad_to_byte(&mut self) {
        if self.bits_in_buffer != 0 {
            self.write(8 - self.bits_in_buffer as u32, 0);
        }
    }
}
fn enc_hyb000(v: u32) -> (u32, u32, u32) {
    if v == 0 {
        return (0, 0, 0);
    }
    let n = 31 - v.leading_zeros();
    (n + 1, n, v - (1 << n))
}
fn enc_hyb_lz77(v: u32) -> (u32, u32, u32) {
    let n = if v == 0 { 0 } else { 31 - v.leading_zeros() };
    if v < 16 {
        (v, 0, 0)
    } else {
        (16 + n - 4, n, v - (1 << n))
    }
}
const KMAXSYM: usize = if K_NUM_RAW_SYMBOLS + 1 < K_NUM_LZ77 {
    K_NUM_LZ77
} else {
    K_NUM_RAW_SYMBOLS + 1
};
fn bit_reverse(nb: usize, bits: u16) -> u16 {
    const N: [u16; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
    let r = (N[(bits & 0xF) as usize] << 12)
        | (N[((bits >> 4) & 0xF) as usize] << 8)
        | (N[((bits >> 8) & 0xF) as usize] << 4)
        | N[(bits >> 12) as usize];
    r >> (16 - nb)
}
fn canon(c1n: &[u8], c1b: &mut [u16], c2n: &[u8], c2b: &mut [u16]) {
    const KMAX: usize = 15;
    let mut cnt = [0u32; KMAX + 1];
    for &n in c1n {
        cnt[n as usize] += 1;
    }
    for &n in c2n {
        cnt[n as usize] += 1;
    }
    let mut next = [0u32; KMAX + 1];
    let mut code = 0u32;
    for i in 1..KMAX + 1 {
        code = (code + cnt[i - 1]) << 1;
        next[i] = code;
    }
    for i in 0..c1n.len() {
        let n = c1n[i] as usize;
        if n > 0 {
            c1b[i] = bit_reverse(n, next[n] as u16);
            next[n] += 1;
        }
    }
    for i in 0..c2n.len() {
        let n = c2n[i] as usize;
        if n > 0 {
            c2b[i] = bit_reverse(n, next[n] as u16);
            next[n] += 1;
        }
    }
}
fn ccl_nz_impl(f: &[u64], n: usize, prec: usize, mn: &[u8], mx: &[u8], nb: &mut [u8]) {
    let inf = u64::MAX / 4;
    let w = (1usize << prec) + 1;
    let mut d = vec![inf; w * (n + 1)];
    let id = |s: usize, o: usize| s * w + o;
    d[id(0, 0)] = 0;
    for s in 0..n {
        for bits in mn[s]..=mx[s] {
            let od = 1usize << (prec - bits as usize);
            let mut o = 0;
            while o + od <= (1usize << prec) {
                let cand = d[id(s, o)].saturating_add(f[s] * bits as u64);
                let t = id(s + 1, o + od);
                if cand < d[t] {
                    d[t] = cand;
                }
                o += 1;
            }
        }
    }
    let mut s = n;
    let mut o = 1usize << prec;
    while s > 0 {
        s -= 1;
        for bits in mn[s]..=mx[s] {
            let od = 1usize << (prec - bits as usize);
            if od <= o && d[id(s + 1, o)] == d[id(s, o - od)] + f[s] * bits as u64 {
                o -= od;
                nb[s] = bits;
                break;
            }
        }
    }
}
fn ccl_nz(f: &[u64], n: usize, mn: &mut [u8], mx: &[u8], nb: &mut [u8]) {
    let mut prec = 0usize;
    let mut sh = 255usize;
    for i in 0..n {
        if mn[i] < 1 {
            mn[i] = 1;
        }
        prec = prec.max(mx[i] as usize);
        sh = sh.min(mn[i] as usize);
    }
    prec -= sh - 1;
    ccl_nz_impl(f, n, prec, mn, mx, nb);
}
fn ccl(f: &[u64], n: usize, mni: &[u8], mxi: &[u8], nb: &mut [u8]) {
    let mut cf = [0u64; KMAXSYM];
    let mut mn = [0u8; KMAXSYM];
    let mut mx = [0u8; KMAXSYM];
    let mut ni = 0;
    for i in 0..n {
        if f[i] != 0 {
            cf[ni] = f[i];
            mn[ni] = mni[i];
            mx[ni] = mxi[i];
            ni += 1;
        }
    }
    let mut numbits = [0u8; KMAXSYM];
    ccl_nz(&cf, ni, &mut mn, &mx, &mut numbits);
    ni = 0;
    for i in 0..n {
        nb[i] = 0;
        if f[i] != 0 {
            nb[i] = numbits[ni];
            ni += 1;
        }
    }
}
struct PrefixCode {
    raw_nbits: [u8; 19],
    raw_bits: [u16; 19],
    lz77_nbits: [u8; 33],
    lz77_bits: [u16; 33],
    cache_nbits: [u8; 32],
    cache_bits: [u64; 32],
}
impl PrefixCode {
    fn new(rc: &[u64; 19], lc: &[u64; 33], mn: &[u8], mx: &[u8]) -> Self {
        let mut l1 = [0u64; 20];
        l1[..19].copy_from_slice(rc);
        let mut numraw = 19;
        while numraw > 0 && l1[numraw - 1] == 0 {
            numraw -= 1;
        }
        l1[numraw] = 0;
        for i in 0..33 {
            l1[numraw] += lc[i];
        }
        let mut l1n = [0u8; 20];
        ccl(&l1, numraw + 1, mn, mx, &mut l1n);
        let mut numlz = 33;
        while numlz > 0 && lc[numlz - 1] == 0 {
            numlz -= 1;
        }
        let ll = 15 - l1n[numraw];
        let minl = [0u8; 33];
        let maxl = [ll; 33];
        let mut l2n = [0u8; 33];
        ccl(lc, numlz, &minl, &maxl, &mut l2n);
        let mut raw_nbits = [0u8; 19];
        let mut raw_bits = [0u16; 19];
        let mut lz_nbits = [0u8; 33];
        let mut lz_bits = [0u16; 33];
        raw_nbits[..numraw].copy_from_slice(&l1n[..numraw]);
        for i in 0..numlz {
            lz_nbits[i] = if l2n[i] != 0 { l1n[numraw] + l2n[i] } else { 0 };
        }
        canon(
            &raw_nbits[..numraw],
            &mut raw_bits[..numraw],
            &lz_nbits,
            &mut lz_bits,
        );
        let mut cn = [0u8; 32];
        let mut cb = [0u64; 32];
        for count in 0..32 {
            let (t, nb, bits) = enc_hyb_lz77(count as u32);
            let t = t as usize;
            cn[count] = lz_nbits[t] + nb as u8 + raw_nbits[0];
            let mut wb = ((bits as u64) << lz_nbits[t]) | lz_bits[t] as u64;
            wb = (wb << raw_nbits[0]) | raw_bits[0] as u64;
            cb[count] = wb;
        }
        Self {
            raw_nbits,
            raw_bits,
            lz77_nbits: lz_nbits,
            lz77_bits: lz_bits,
            cache_nbits: cn,
            cache_bits: cb,
        }
    }
    fn write_to(&self, w: &mut BitWriter) {
        let mut clc = [0u64; 18];
        clc[17] = 3 + 2 * (K_NUM_LZ77 as u64 - 1);
        for &n in &self.raw_nbits {
            clc[n as usize] += 1;
        }
        for &n in &self.lz77_nbits {
            clc[n as usize] += 1;
        }
        let mut cln = [0u8; 18];
        let cmin = [0u8; 18];
        let cmax = [5u8; 18];
        ccl(&clc, 18, &cmin, &cmax, &mut cln);
        w.write(2, 0b00);
        let order = [
            1usize, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];
        let clln = [2u32, 4, 3, 2, 2, 4];
        let cllb = [0u64, 7, 3, 2, 1, 15];
        let mut ncl = 18;
        while cln[order[ncl - 1]] == 0 {
            ncl -= 1;
        }
        for i in 0..ncl {
            let s = cln[order[i]] as usize;
            w.write(clln[s], cllb[s]);
        }
        let mut clb = [0u16; 18];
        let e: [u8; 0] = [];
        let mut eb: [u16; 0] = [];
        canon(&cln, &mut clb, &e, &mut eb);
        for &n in &self.raw_nbits {
            w.write(cln[n as usize] as u32, clb[n as usize] as u64);
        }
        let mut numlz = 33;
        while self.lz77_nbits[numlz - 1] == 0 {
            numlz -= 1;
        }
        w.write(cln[17] as u32, clb[17] as u64);
        w.write(3, 0b010);
        w.write(cln[17] as u32, clb[17] as u64);
        w.write(3, 0b000);
        w.write(cln[17] as u32, clb[17] as u64);
        w.write(3, 0b010);
        for i in 0..numlz {
            let n = self.lz77_nbits[i] as usize;
            w.write(cln[n] as u32, clb[n] as u64);
        }
    }
}

fn ycocg(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    (y, co, cg)
}
fn build_planes_from<F: Fn(usize) -> i32>(w: usize, h: usize, nb: usize, get: F) -> Vec<Vec<i32>> {
    let n = w * h;
    let mut p = vec![vec![0i32; n]; nb];
    match nb {
        1 => {
            for i in 0..n {
                p[0][i] = get(i);
            }
        }
        2 => {
            for i in 0..n {
                p[0][i] = get(2 * i);
                p[1][i] = get(2 * i + 1);
            }
        }
        3 => {
            for i in 0..n {
                let (y, co, cg) = ycocg(get(3 * i), get(3 * i + 1), get(3 * i + 2));
                p[0][i] = y;
                p[1][i] = co;
                p[2][i] = cg;
            }
        }
        4 => {
            for i in 0..n {
                let (y, co, cg) = ycocg(get(4 * i), get(4 * i + 1), get(4 * i + 2));
                p[0][i] = y;
                p[1][i] = co;
                p[2][i] = cg;
                p[3][i] = get(4 * i + 3);
            }
        }
        _ => unreachable!(),
    }
    p
}
fn build_planes_u8(img: &[u8], w: usize, h: usize, nb: usize) -> Vec<Vec<i32>> {
    build_planes_from(w, h, nb, |i| img[i] as i32)
}
fn build_planes_u16(img: &[u16], w: usize, h: usize, nb: usize) -> Vec<Vec<i32>> {
    build_planes_from(w, h, nb, |i| img[i] as i32)
}
#[inline]
fn pack_signed(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

trait Sink {
    fn chunk(&mut self, run: usize, res: &[u32; 8], skip: usize, n: usize);
    fn finalize(&mut self, run: usize);
}
struct Collector<'a> {
    raw: &'a mut [u64; 19],
    lz: &'a mut [u64; 33],
}
impl<'a> Collector<'a> {
    fn rle(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        self.raw[0] += 1;
        let c = count - (K_LZ77_MIN_LENGTH + 1);
        let (t, _, _) = enc_hyb_lz77(c as u32);
        self.lz[t as usize] += 1;
    }
}
impl<'a> Sink for Collector<'a> {
    fn chunk(&mut self, run: usize, res: &[u32; 8], skip: usize, n: usize) {
        self.rle(run);
        for ix in skip..n {
            let (t, _, _) = enc_hyb000(res[ix]);
            self.raw[t as usize] += 1;
        }
    }
    fn finalize(&mut self, _: usize) {}
}
struct Encoder<'a> {
    code: &'a PrefixCode,
    out: &'a mut BitWriter,
}
impl<'a> Encoder<'a> {
    fn rle(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        let c = count - (K_LZ77_MIN_LENGTH + 1);
        if c < 32 {
            self.out
                .write(self.code.cache_nbits[c] as u32, self.code.cache_bits[c]);
        } else {
            let (t, nb, bits) = enc_hyb_lz77(c as u32);
            let t = t as usize;
            let mut wb = ((bits as u64) << self.code.lz77_nbits[t]) | self.code.lz77_bits[t] as u64;
            wb = (wb << self.code.raw_nbits[0]) | self.code.raw_bits[0] as u64;
            self.out.write(
                self.code.lz77_nbits[t] as u32 + nb + self.code.raw_nbits[0] as u32,
                wb,
            );
        }
    }
}
impl<'a> Sink for Encoder<'a> {
    fn chunk(&mut self, run: usize, res: &[u32; 8], skip: usize, n: usize) {
        self.rle(run);
        for ix in skip..n {
            let (t, nb, bits) = enc_hyb000(res[ix]);
            let t = t as usize;
            self.out.write(
                self.code.raw_nbits[t] as u32 + nb,
                self.code.raw_bits[t] as u64 | (bits as u64) << self.code.raw_nbits[t],
            );
        }
    }
    fn finalize(&mut self, run: usize) {
        self.rle(run);
    }
}
fn process_chunk(
    sink: &mut dyn Sink,
    run: &mut usize,
    cur: &[i32],
    prev: &[i32],
    first: bool,
    x: usize,
    n: usize,
) {
    let mut res = [0u32; 8];
    let mut ps = 0usize;
    let mut req = 0usize;
    for ix in 0..8 {
        let bi = KPAD + x + ix;
        let px = cur[bi];
        let left = cur[bi - 1];
        let (top, topleft) = if first {
            (cur[bi - 1], cur[bi - 1])
        } else {
            (prev[bi], prev[bi - 1])
        };
        let ac = left.wrapping_sub(topleft);
        let bc = top.wrapping_sub(topleft);
        let grad = ac.wrapping_add(top);
        let clamp = if (left.wrapping_sub(top) ^ bc) < 0 {
            top
        } else {
            left
        };
        let pred = if (ac ^ bc) < 0 { grad } else { clamp };
        let p = pack_signed(px.wrapping_sub(pred));
        res[ix] = p;
        ps = if ps == req {
            ps + (p == 0) as usize
        } else {
            ps
        };
        req += 1;
    }
    ps = n.min(ps);
    let km = K_LZ77_MIN_LENGTH;
    if ps == n && (*run > 0 || ps > km) {
        *run += ps;
    } else if ps + *run > km {
        sink.chunk(*run + ps, &res, ps, n);
        *run = 0;
    } else {
        sink.chunk(0, &res, 0, n);
    }
}
fn process_plane(
    sink: &mut dyn Sink,
    plane: &[i32],
    pstride: usize,
    x0: usize,
    y0: usize,
    xs: usize,
    yskip: usize,
    ys: usize,
) {
    let mut cur = vec![0i32; NPX];
    let mut prev = vec![0i32; NPX];
    let mut run = 0usize;
    for y in 0..ys {
        std::mem::swap(&mut cur, &mut prev);
        for x in 0..xs {
            cur[KPAD + x] = plane[(y0 + y) * pstride + x0 + x];
        }
        cur[KPAD - 1] = if y > 0 { prev[KPAD] } else { 0 };
        if y > 0 {
            prev[KPAD - 1] = prev[KPAD];
        }
        if y < yskip {
            continue;
        }
        let first = y == 0;
        let mut x = 0;
        while x < xs {
            let n = KCHUNK.min(xs - x);
            process_chunk(sink, &mut run, &cur, &prev, first, x, n);
            x += KCHUNK;
        }
    }
    sink.finalize(run);
}
fn toc_bucket(gs: usize) -> usize {
    let mut b = 0;
    while b < 3 && gs >= GROUP_SIZE_OFFSET[b + 1] {
        b += 1;
    }
    b
}
fn wsz(o: &mut BitWriter, size: usize) {
    let s = (size - 1) as u64;
    if s < (1 << 9) {
        o.write(2, 0);
        o.write(9, s);
    } else if s < (1 << 13) {
        o.write(2, 1);
        o.write(13, s);
    } else if s < (1 << 18) {
        o.write(2, 2);
        o.write(18, s);
    } else {
        o.write(2, 3);
        o.write(30, s);
    }
}
fn dc_global_common(_single: bool, _w: usize, _h: usize, code: &[PrefixCode], o: &mut BitWriter) {
    o.write(1, 1);
    o.write(1, 1);
    o.write(1, 0);
    o.write(1, 1);
    o.write(2, 0);
    o.write(1, 1);
    o.write(4, 0);
    o.write(6, 0b100011);
    o.write(2, 1);
    o.write(2, 3);
    o.write(2, 0);
    o.write(2, 1);
    o.write(2, 2);
    o.write(2, 3);
    o.write(1, 0);
    let sb = [0b00u64, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    let sn = [2u32, 2, 3, 3, 4, 4];
    for &v in &[
        1usize, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ] {
        o.write(sn[v], sb[v]);
    }
    o.write(1, 1);
    o.write(2, 0);
    o.write(4, 0b1010);
    o.write(4, 4);
    o.write(3, 0);
    o.write(3, 0);
    o.write(1, 1);
    o.write(2, 3);
    o.write(3, 4);
    o.write(3, 3);
    o.write(3, 2);
    o.write(3, 1);
    o.write(3, 0);
    o.write(1, 1);
    o.write(4, 0);
    for _ in 0..4 {
        o.write(4, 0);
    }
    o.write(5, 0b00001);
    for _ in 0..4 {
        o.write(1, 1);
        o.write(4, 8);
        o.write(8, 256);
    }
    o.write(2, 1);
    o.write(2, 0);
    o.write(1, 1);
    for i in 0..4 {
        code[i].write_to(o);
    }
    o.write(1, 1);
    o.write(1, 1);
}
fn write_enum(o: &mut BitWriter, v: u32) {
    if v == 0 {
        o.write(2, 0);
    } else if v == 1 {
        o.write(2, 1);
    } else if v <= 17 {
        o.write(2, 2);
        o.write(4, (v - 2) as u64);
    } else {
        o.write(2, 3);
        o.write(6, (v - 18) as u64);
    }
}
fn write_bits_per_sample(o: &mut BitWriter, bits: u32) {
    match bits {
        8 => o.write(2, 0),
        10 => o.write(2, 1),
        12 => o.write(2, 2),
        b => {
            o.write(2, 3);
            o.write(6, (b - 1) as u64);
        }
    }
}
fn append_icc_stream(
    o: &mut BitWriter,
    icc: &[u8],
    scratch: &mut crate::coder_scratch::CoderScratch,
) {
    let mut tmp = crate::bit_writer::BitWriter::new();
    crate::icc_codec::write_icc_stream(icc, &mut scratch.huffman_pool, &mut tmp);
    let nbits = tmp.bits_written(); // exact bit count, captured before padding
    tmp.zero_pad_to_byte(); // into_bytes() requires byte alignment
    let bytes = tmp.into_bytes(); // ceil(nbits/8) bytes; pad bits are in the high bits of the last byte
    let full = nbits / 8;
    for &byte in &bytes[..full] {
        o.write(8, byte as u64);
    }
    let rem = nbits % 8;
    if rem != 0 {
        // copy only the `rem` real bits; the zero-pad bits above them are discarded
        o.write(rem as u32, (bytes[full] as u64) & ((1u64 << rem) - 1));
    }
}

fn write_color(o: &mut BitWriter, cs: ColorSpace, m: &FlMeta) {
    o.write(1, 0); // color_encoding all_default = 0
    if m.icc.is_some() {
        o.write(1, 1); // want_icc = 1
        write_enum(o, cs as u32); // color_space (still present)
        return; // white_point/primaries/transfer/intent are gated on !want_icc
    }
    o.write(1, 0); // want_icc = 0
    write_enum(o, cs as u32); // color_space (Rgb=0, Gray=1; matches JXL enum)
    write_enum(o, m.white_point as u32); // white_point
    if cs != ColorSpace::Gray {
        write_enum(o, m.primaries as u32);
    } // primaries (cond color_space != Grey)
    o.write(1, 0); // transfer fn: have_gamma = 0
    write_enum(o, m.transfer as u32); // transfer function enum
    write_enum(o, m.rendering_intent as u32); // rendering_intent
}
fn prepare_header(
    o: &mut BitWriter,
    w: usize,
    h: usize,
    nb: usize,
    cs: ColorSpace,
    bits: u32,
    m: &FlMeta,
    gsizes: &[usize],
    scratch: &mut crate::coder_scratch::CoderScratch,
) {
    let have_alpha = nb == 2 || nb == 4;
    let extra_fields = m.orientation != Orientation::Normal || m.intrinsic_size.is_some();
    o.write(16, 0x0AFF);
    o.write(1, 0);
    wsz(o, h);
    o.write(3, 0);
    wsz(o, w);
    o.write(1, 0); // ImageMetadata all_default = 0
    if extra_fields {
        o.write(1, 1); // extra_fields = 1
        o.write(3, m.orientation.to_u3()); // orientation = 1 + u(3)
        if let Some((iw, ih)) = m.intrinsic_size {
            o.write(1, 1); // have_intr_size
            o.write(1, 0);
            wsz(o, ih);
            o.write(3, 0);
            wsz(o, iw); // intrinsic SizeHeader
        } else {
            o.write(1, 0);
        }
        o.write(1, 0); // have_preview
        o.write(1, 0); // have_animation
    } else {
        o.write(1, 0);
    } // extra_fields = 0
    o.write(1, 0); // bit_depth.floating_point_sample = 0
    write_bits_per_sample(o, bits);
    o.write(1, if bits <= 14 { 1 } else { 0 }); // modular_16bit_buffers
    if have_alpha {
        o.write(2, 0b01); // num_extra = 1
        if bits == 8 && !m.alpha_premultiplied {
            o.write(1, 1); // ExtraChannelInfo all_default (8-bit alpha)
        } else {
            o.write(1, 0); // not all_default
            write_enum(o, 0); // type = kAlpha
            o.write(1, 0); // alpha bit_depth float_sample = 0
            write_bits_per_sample(o, bits);
            o.write(2, 0); // dim_shift = 0
            o.write(2, 0); // name length = 0
            o.write(1, m.alpha_premultiplied as u64); // alpha_associated
        }
    } else {
        o.write(2, 0);
    } // num_extra = 0
    o.write(1, 0); // xyb_encoded = 0
    write_color(o, cs, m);
    if extra_fields {
        o.write(1, 1);
    } // tone_mapping all_default (cond extra_fields)
    o.write(2, 0); // extensions = 0
    o.write(1, 1); // default_m = 1
    if let Some(icc) = &m.icc {
        append_icc_stream(o, icc, scratch); // ICC stream after CustomTransformData, before pad
    }
    o.zero_pad_to_byte();
    // ---- frame header + TOC (bit-depth/color independent) ----
    o.write(1, 0);
    o.write(2, 0);
    o.write(1, 1);
    o.write(2, 0);
    o.write(1, 0);
    o.write(2, 0);
    if have_alpha {
        o.write(2, 0);
    }
    o.write(2, 0b01);
    o.write(2, 0);
    o.write(1, 0);
    o.write(2, 0);
    if have_alpha {
        o.write(2, 0);
    }
    o.write(1, 1);
    o.write(2, 0);
    o.write(1, 0);
    o.write(1, 0);
    o.write(2, 0);
    o.write(2, 0);
    o.write(2, 0);
    o.write(1, 0);
    o.zero_pad_to_byte();
    for &gs in gsizes {
        let b = toc_bucket(gs);
        o.write(2, b as u64);
        o.write(TOC_BITS[b] - 2, (gs - GROUP_SIZE_OFFSET[b]) as u64);
    }
    o.zero_pad_to_byte();
}
fn bd_limits(bits: u32, ycocg: bool) -> (usize, [u8; 20], [u8; 20]) {
    let ns = (bits as usize + if ycocg { 3 } else { 2 }).min(19);
    let mn = [0u8; 20];
    let mut mx = [0u8; 20];
    if bits <= 8 {
        for i in 0..20 {
            mx[i] = 7;
        }
        mx[11] = 10;
    } else if bits <= 13 {
        for i in 0..20 {
            mx[i] = 8;
        }
        mx[16] = 10;
    } else if bits == 14 {
        for i in 0..20 {
            mx[i] = 7;
        }
        mx[15] = 8;
        mx[16] = 8;
        mx[17] = 10;
    } else {
        for i in 0..20 {
            mx[i] = 7;
        }
        for i in 13..19 {
            mx[i] = 8;
        }
        mx[19] = 10;
    }
    (ns, mn, mx)
}
fn build_codes(
    planes: &[Vec<i32>],
    w: usize,
    h: usize,
    nb: usize,
    ngx: usize,
    ngy: usize,
    bits: u32,
) -> Vec<PrefixCode> {
    let effort = 2;
    let (num_symbols, mn, mx) = bd_limits(bits, nb > 2);
    let mut rc = [[0u64; 19]; 4];
    let mut lc = [[0u64; 33]; 4];
    for g in 0..ngx * ngy {
        let xg = g % ngx;
        let yg = g / ngx;
        let x0 = xg * 256;
        let y0 = yg * 256;
        let xs = (w - x0).min(256);
        let ys = (h - y0).min(256);
        let num_rows = 2 * effort * ys / 256;
        let y_begin = ((ys as isize - num_rows as isize).max(0) as usize) / 2;
        let y_count = num_rows.min(ys.saturating_sub(y_begin + 1));
        let x_max = xs / 8 * 8;
        if y_count > 0 && x_max > 0 {
            for c in 0..nb {
                let mut coll = Collector {
                    raw: &mut rc[c],
                    lz: &mut lc[c],
                };
                process_plane(
                    &mut coll,
                    &planes[c],
                    w,
                    x0,
                    y0 + y_begin,
                    x_max,
                    1,
                    1 + y_count,
                );
            }
        }
    }
    let mut base_raw = [
        3843u64, 852, 1270, 1214, 1014, 727, 481, 300, 159, 51, 5, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    for i in num_symbols..19 {
        base_raw[i] = 0;
    }
    for ci in 0..4 {
        for i in 0..19 {
            rc[ci][i] = (rc[ci][i] << 8) + base_raw[i];
        }
    }
    let base_lz = [
        29u64, 27, 25, 23, 21, 21, 19, 18, 21, 17, 16, 15, 15, 14, 13, 13, 137, 98, 61, 34, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    ];
    for ci in 0..4 {
        for i in 0..33 {
            lc[ci][i] = (lc[ci][i] << 8) + base_lz[i];
        }
    }
    (0..4)
        .map(|i| PrefixCode::new(&rc[i], &lc[i], &mn, &mx))
        .collect()
}

/// Number of color channels for the supported color spaces.
/// `Gray` -> 1, `Rgb` -> 3; `Xyb`/`Unknown` are not supported by this path.
fn color_channels(cs: ColorSpace) -> Result<usize, EncodeError> {
    match cs {
        ColorSpace::Gray => Ok(1),
        ColorSpace::Rgb => Ok(3),
        other => Err(EncodeError::UnsupportedColorSpace(other)),
    }
}

/// Shared validation. Returns the sample count (`w * h * nb`) on success.
fn validate(w: usize, h: usize, nb: usize, _meta: &FlMeta) -> Result<usize, EncodeError> {
    if w == 0 || h == 0 {
        return Err(EncodeError::EmptyImage);
    }
    if !(1..=4).contains(&nb) {
        return Err(EncodeError::BadChannelCount(nb));
    }
    const MAX_DIM: usize = 1 << 30; // JXL spec per-axis limit
    if w > MAX_DIM || h > MAX_DIM {
        return Err(EncodeError::DimensionTooLarge {
            width: w,
            height: h,
        });
    }
    let npix = w.checked_mul(h).ok_or(EncodeError::SizeOverflow)?;
    npix.checked_mul(nb).ok_or(EncodeError::SizeOverflow)
}

/// Encode an 8-bit image.
/// Wrap the bare codestream in a JXL container when EXIF or XMP metadata must
/// be carried; otherwise return it untouched.
fn finalize_fl(
    codestream: Vec<u8>,
    bits: u32,
    alpha: bool,
    meta: &FlMeta,
) -> Result<Vec<u8>, EncodeError> {
    if meta.exif.is_some() || meta.xmp.is_some() {
        let alpha_bits = if alpha { bits } else { 0 };
        let need_l10 = crate::encode_image::needs_level_10(bits, true, alpha_bits);
        crate::encode_image::wrap_jxl_container(
            codestream,
            if need_l10 { 10 } else { 5 },
            meta.exif.as_deref(),
            meta.xmp.as_deref(),
            None,
        )
    } else {
        Ok(codestream)
    }
}

pub fn encode_fast_lossless(
    img: &[u8],
    w: usize,
    h: usize,
    color_space: ColorSpace,
    alpha: bool,
    meta: &FlMeta,
) -> Result<Vec<u8>, EncodeError> {
    let nb = color_channels(color_space)? + alpha as usize;
    let nsamp = validate(w, h, nb, meta)?;
    if img.len() != nsamp {
        return Err(EncodeError::InputLength {
            expected: nsamp,
            got: img.len(),
        });
    }
    finalize_fl(
        encode_planes(
            &build_planes_u8(img, w, h, nb),
            w,
            h,
            nb,
            color_space,
            8,
            meta,
        ),
        8,
        alpha,
        meta,
    )
}

/// Encode a high-bit-depth image (9..=16 bits).
pub fn encode_fast_lossless_u16(
    img: &[u16],
    w: usize,
    h: usize,
    color_space: ColorSpace,
    alpha: bool,
    bits: u32,
    meta: &FlMeta,
) -> Result<Vec<u8>, EncodeError> {
    if !(9..=16).contains(&bits) {
        return Err(EncodeError::BadBitDepth(bits));
    }
    let nb = color_channels(color_space)? + alpha as usize;
    let nsamp = validate(w, h, nb, meta)?;
    if img.len() != nsamp {
        return Err(EncodeError::InputLength {
            expected: nsamp,
            got: img.len(),
        });
    }
    finalize_fl(
        encode_planes(
            &build_planes_u16(img, w, h, nb),
            w,
            h,
            nb,
            color_space,
            bits,
            meta,
        ),
        bits,
        alpha,
        meta,
    )
}

/// Assemble the JXL codestream from pre-built i32 planes. Shared by both entries.
fn encode_planes(
    planes: &[Vec<i32>],
    w: usize,
    h: usize,
    nb: usize,
    cs: ColorSpace,
    bits: u32,
    meta: &FlMeta,
) -> Vec<u8> {
    let mut scratch = Box::<crate::coder_scratch::CoderScratch>::default();
    let ngx = w.div_ceil(256);
    let ngy = h.div_ceil(256);
    let onegroup = ngx == 1 && ngy == 1;
    let ndg = w.div_ceil(2048) * h.div_ceil(2048);
    let num_ac = ngx * ngy;
    let hcode = build_codes(planes, w, h, nb, ngx, ngy, bits);
    let mut lfg = BitWriter::allocate(100000 + if onegroup { w * h * 32 * nb } else { 0 });
    dc_global_common(onegroup, w, h, &hcode, &mut lfg);
    if nb > 2 {
        lfg.write(2, 0b01);
        lfg.write(2, 0b00);
        lfg.write(5, 0b00000);
        lfg.write(2, 0b00);
    } else {
        lfg.write(2, 0b00);
    }
    if onegroup {
        for c in 0..nb {
            let mut enc = Encoder {
                code: &hcode[c],
                out: &mut lfg,
            };
            process_plane(&mut enc, &planes[c], w, 0, 0, w, 0, h);
        }
        let gsize = (lfg.bytes_written * 8 + lfg.bits_in_buffer).div_ceil(8);
        let mut hdr = BitWriter::allocate(4000);
        prepare_header(&mut hdr, w, h, nb, cs, bits, meta, &[gsize], &mut scratch);
        let mut out = Vec::new();
        out.extend_from_slice(&hdr.data[..hdr.bytes_written]);
        out.extend_from_slice(&lfg.data[..lfg.bytes_written]);
        if lfg.bits_in_buffer > 0 {
            out.push((lfg.buffer & 0xFF) as u8);
        }
        return out;
    }
    lfg.zero_pad_to_byte();
    let num_groups = 2 + ndg + num_ac;
    let mut gsizes = vec![0usize; num_groups];
    gsizes[0] = lfg.bytes_written;
    let mut ac_sections: Vec<Vec<u8>> = Vec::with_capacity(num_ac);
    for g in 0..num_ac {
        let xg = g % ngx;
        let yg = g / ngx;
        let x0 = xg * 256;
        let y0 = yg * 256;
        let xs = (w - x0).min(256);
        let ys = (h - y0).min(256);
        let mut gw = BitWriter::allocate(xs * ys * nb * 40 + 100000);
        gw.write(1, 1);
        gw.write(1, 1);
        gw.write(2, 0b00);
        for c in 0..nb {
            let mut enc = Encoder {
                code: &hcode[c],
                out: &mut gw,
            };
            process_plane(&mut enc, &planes[c], w, x0, y0, xs, 0, ys);
        }
        gw.zero_pad_to_byte();
        gsizes[2 + ndg + g] = gw.bytes_written;
        ac_sections.push(gw.data[..gw.bytes_written].to_vec());
    }
    let mut hdr = BitWriter::allocate(64 + num_groups * 40);
    prepare_header(&mut hdr, w, h, nb, cs, bits, meta, &gsizes, &mut scratch);
    let mut out = Vec::new();
    out.extend_from_slice(&hdr.data[..hdr.bytes_written]);
    out.extend_from_slice(&lfg.data[..lfg.bytes_written]);
    for sec in &ac_sections {
        out.extend_from_slice(sec);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColorSpace::{Gray, Rgb};

    fn ramp8(w: usize, h: usize, nb: usize) -> Vec<u8> {
        let mut img = vec![0u8; w * h * nb];
        for i in 0..img.len() {
            img[i] = ((i * 131 + 7) & 0xff) as u8;
        }
        img
    }
    fn ramp16(w: usize, h: usize, nb: usize, bits: u32) -> Vec<u16> {
        let maxv = ((1u32 << bits) - 1) as u16;
        let mut img = vec![0u16; w * h * nb];
        for i in 0..img.len() {
            img[i] = (((i * 2654435761usize) as u16) & maxv).min(maxv);
        }
        img
    }

    #[test]
    fn rejects_zero_dims() {
        let img = ramp8(10, 10, 3);
        assert_eq!(
            encode_fast_lossless(&img, 0, 10, Rgb, false, &FlMeta::srgb()),
            Err(EncodeError::EmptyImage)
        );
        assert_eq!(
            encode_fast_lossless(&img, 10, 0, Rgb, false, &FlMeta::srgb()),
            Err(EncodeError::EmptyImage)
        );
    }
    #[test]
    fn rejects_unsupported_color_space() {
        let img = ramp8(10, 10, 3);
        assert!(matches!(
            encode_fast_lossless(&img, 10, 10, ColorSpace::Xyb, false, &FlMeta::srgb()),
            Err(EncodeError::UnsupportedColorSpace(ColorSpace::Xyb))
        ));
        assert!(matches!(
            encode_fast_lossless(&img, 10, 10, ColorSpace::Unknown, false, &FlMeta::srgb()),
            Err(EncodeError::UnsupportedColorSpace(ColorSpace::Unknown))
        ));
    }

    #[test]
    fn accepts_embedded_icc() {
        let img = ramp8(10, 10, 3);
        let mut m = FlMeta::srgb();
        // A minimal non-empty ICC blob; the encoder compresses it via icc_codec.
        m.icc = Some(vec![0u8; 128]);
        let out = encode_fast_lossless(&img, 10, 10, Rgb, false, &m).expect("ICC encode");
        // ICC-bearing stream must be larger than the equivalent no-ICC stream.
        let base = encode_fast_lossless(&img, 10, 10, Rgb, false, &FlMeta::srgb()).unwrap();
        assert!(out.len() > base.len());
    }

    #[test]
    fn accepts_xmp() {
        let img = ramp8(10, 10, 3);
        let xmp = b"<x:xmpmeta/>".to_vec();
        let mut meta = FlMeta::srgb();
        meta.xmp = Some(xmp.clone());
        let out = encode_fast_lossless(&img, 10, 10, Rgb, false, &meta).unwrap();
        let at = out
            .windows(4)
            .position(|window| window == b"xml ")
            .expect("xml box");
        assert_eq!(&out[at + 4..], xmp.as_slice());
    }
    #[test]
    fn rejects_wrong_length() {
        let img = ramp8(10, 10, 3);
        // claim Gray (1 channel) for 3-channel data -> length mismatch
        assert!(matches!(
            encode_fast_lossless(&img, 10, 10, Gray, false, &FlMeta::srgb()),
            Err(EncodeError::InputLength { .. })
        ));
        let img16 = ramp16(10, 10, 3, 16);
        assert!(matches!(
            encode_fast_lossless_u16(&img16, 10, 10, Gray, false, 16, &FlMeta::srgb()),
            Err(EncodeError::InputLength { .. })
        ));
    }
    #[test]
    fn rejects_bad_bit_depth() {
        let img16 = ramp16(10, 10, 3, 16);
        assert!(matches!(
            encode_fast_lossless_u16(&img16, 10, 10, Rgb, false, 8, &FlMeta::srgb()),
            Err(EncodeError::BadBitDepth(8))
        ));
        assert!(matches!(
            encode_fast_lossless_u16(&img16, 10, 10, Rgb, false, 17, &FlMeta::srgb()),
            Err(EncodeError::BadBitDepth(17))
        ));
    }
    #[test]
    fn rejects_oversized() {
        let img = ramp8(4, 4, 3);
        assert!(matches!(
            encode_fast_lossless(&img, (1usize << 30) + 1, 4, Rgb, false, &FlMeta::srgb()),
            Err(EncodeError::DimensionTooLarge { .. })
        ));
        assert!(
            encode_fast_lossless(
                &img,
                1usize << 30,
                1usize << 30,
                Rgb,
                false,
                &FlMeta::srgb()
            )
            .is_err()
        );
    }
    #[test]
    fn encodes_8bit_metadata_variants() {
        let img = ramp8(48, 48, 3);
        let mut metas = vec![FlMeta::srgb(), FlMeta::linear(), FlMeta::display_p3()];
        for o in [
            Orientation::Normal,
            Orientation::FlipH,
            Orientation::Rotate180,
            Orientation::FlipV,
            Orientation::Transpose,
            Orientation::Rotate90,
            Orientation::Transverse,
            Orientation::Rotate270,
        ] {
            let mut m = FlMeta::srgb();
            m.orientation = o;
            metas.push(m);
        }
        let mut m = FlMeta::srgb();
        m.intrinsic_size = Some((96, 96));
        metas.push(m);
        for m in &metas {
            let out = encode_fast_lossless(&img, 48, 48, Rgb, false, m).expect("encode");
            assert!(out.len() > 8 && out[0] == 0xFF && out[1] == 0x0A);
        }
    }
    #[test]
    fn encodes_high_bit_depth_hdr() {
        for &(cs, alpha, bits) in &[
            (Gray, false, 10u32),
            (Rgb, false, 12),
            (Rgb, false, 16),
            (Rgb, true, 16),
        ] {
            let nb = (if cs == Gray { 1 } else { 3 }) + alpha as usize;
            let img = ramp16(70, 50, nb, bits);
            let out =
                encode_fast_lossless_u16(&img, 70, 50, cs, alpha, bits, &FlMeta::rec2100_pq())
                    .expect("u16 encode");
            assert!(out.len() > 8 && out[0] == 0xFF && out[1] == 0x0A);
        }
        let img = ramp16(48, 48, 3, 10);
        assert!(
            encode_fast_lossless_u16(&img, 48, 48, Rgb, false, 10, &FlMeta::rec2100_hlg()).is_ok()
        );
    }
    #[test]
    fn encodes_edge_dimensions() {
        for &(w, h) in &[
            (1usize, 1usize),
            (1, 300),
            (300, 1),
            (255, 255),
            (256, 256),
            (257, 257),
            (256, 257),
            (513, 2),
        ] {
            for &(cs, alpha) in &[(Gray, false), (Rgb, false), (Rgb, true)] {
                let nb = (if cs == Gray { 1 } else { 3 }) + alpha as usize;
                let img = ramp8(w, h, nb);
                assert!(
                    encode_fast_lossless(&img, w, h, cs, alpha, &FlMeta::srgb()).is_ok(),
                    "{}x{} cs={:?} a={}",
                    w,
                    h,
                    cs,
                    alpha
                );
                let img16 = ramp16(w, h, nb, 16);
                assert!(
                    encode_fast_lossless_u16(&img16, w, h, cs, alpha, 16, &FlMeta::srgb()).is_ok(),
                    "u16 {}x{} cs={:?} a={}",
                    w,
                    h,
                    cs,
                    alpha
                );
            }
        }
    }
}
