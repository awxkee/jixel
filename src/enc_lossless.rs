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

const TREE_CTX_SPLIT_VAL: u32 = 0;
const TREE_CTX_PROPERTY: u32 = 1;
const TREE_CTX_PREDICTOR: u32 = 2;
const TREE_CTX_OFFSET: u32 = 3;
const TREE_CTX_MULTIPLIER_LOG: u32 = 4;
const TREE_CTX_MULTIPLIER_BITS: u32 = 5;
const NUM_TREE_CONTEXTS: usize = 6;
#[allow(dead_code)]
const PREDICTOR_GRADIENT: u32 = 5;
const PREDICTOR_WEIGHTED: u32 = 6;

// ---------------------------------------------------------------------------
// Self-correcting Weighted Predictor (WP), bit-faithful port of libjxl's
// `weighted::State` (context_predict.h) with the *default* header
// (wp_default=1): p1C=16, p2C=10, p3Ca/b/c=7, p3Cd/e=0, w=[0xd,0xc,0xc,0xc].
// Used for all modular channels in the lossless path; the decoder reconstructs
// each pixel as `prediction + residual` using the identical state machine.
// ---------------------------------------------------------------------------
const WP_EXTRA_BITS: i64 = 3;
const WP_PRED_ROUND: i64 = ((1 << WP_EXTRA_BITS) >> 1) - 1; // = 3
const WP_W: [u32; 4] = [0xd, 0xc, 0xc, 0xc];
const WP_P1C: i64 = 16;
const WP_P2C: i64 = 10;
const WP_P3CA: i64 = 7;
const WP_P3CB: i64 = 7;
const WP_P3CC: i64 = 7;
const WP_P3CD: i64 = 0;
const WP_P3CE: i64 = 0;
// divlookup[i] = (1<<24)/(i+1)
const WP_DIV: [u32; 64] = [
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201, 1398101, 1290555, 1198372, 1118481, 1048576, 986895, 932067, 883011, 838860, 798915,
    762600, 729444, 699050, 671088, 645277, 621378, 599186, 578524, 559240, 541200, 524288, 508400,
    493447, 479349, 466033, 453438, 441505, 430185, 419430, 409200, 399457, 390167, 381300, 372827,
    364722, 356962, 349525, 342392, 335544, 328965, 322638, 316551, 310689, 305040, 299593, 294337,
    289262, 284359, 279620, 275036, 270600, 266305, 262144,
];

struct WpState {
    xsize: usize,
    pred_errors: [Vec<u32>; 4],
    error: Vec<i64>,
    prediction: [i64; 4],
    pred: i64,
    /// libjxl property kWPProp (p[15]): the signed neighbour WP-error with the
    /// largest absolute value among {W, N, NW, NE}. Set on each `predict`.
    wp_prop: i64,
}

impl WpState {
    fn new(xsize: usize) -> Self {
        let n = (xsize + 2) * 2;
        WpState {
            xsize,
            pred_errors: [vec![0u32; n], vec![0u32; n], vec![0u32; n], vec![0u32; n]],
            error: vec![0i64; n],
            prediction: [0; 4],
            pred: 0,
            wp_prop: 0,
        }
    }
    #[inline]
    fn add_bits(x: i64) -> i64 {
        x << WP_EXTRA_BITS
    }
    #[inline]
    fn floor_log2(x: u64) -> u32 {
        debug_assert!(x >= 1);
        63 - x.leading_zeros()
    }
    #[inline]
    fn error_weight(x: u64, maxweight: u32) -> u32 {
        let mut shift = Self::floor_log2(x + 1) as i32 - 5;
        if shift < 0 {
            shift = 0;
        }
        4 + (((maxweight as u64 * WP_DIV[(x >> shift) as usize] as u64) >> shift) as u32)
    }
    #[inline]
    fn weighted_average(pred: &[i64; 4], w_in: &[u32; 4]) -> i64 {
        let mut weight_sum: u32 = w_in.iter().sum();
        let log_weight = Self::floor_log2(weight_sum as u64);
        let mut w = [0u32; 4];
        weight_sum = 0;
        for i in 0..4 {
            w[i] = w_in[i] >> (log_weight - 4);
            weight_sum += w[i];
        }
        let mut sum: i64 = (weight_sum as i64 >> 1) - 1;
        for i in 0..4 {
            sum += pred[i] * w[i] as i64;
        }
        (sum * WP_DIV[(weight_sum - 1) as usize] as i64) >> 24
    }
    /// Predict pixel (x,y). Neighbors are in *normal* (un-shifted) value space;
    /// AddBits is applied internally, matching libjxl's `State::Predict`.
    #[inline]
    fn predict(&mut self, x: usize, y: usize, n: i64, w: i64, ne: i64, nw: i64, nn: i64) -> i64 {
        let xsize = self.xsize;
        let cur_row = if y & 1 == 1 { 0 } else { xsize + 2 };
        let prev_row = if y & 1 == 1 { xsize + 2 } else { 0 };
        let pos_n = prev_row + x;
        let pos_ne = if x < xsize - 1 { pos_n + 1 } else { pos_n };
        let pos_nw = if x > 0 { pos_n - 1 } else { pos_n };
        let mut weights = [0u32; 4];
        for i in 0..4 {
            let s = self.pred_errors[i][pos_n] as u64
                + self.pred_errors[i][pos_ne] as u64
                + self.pred_errors[i][pos_nw] as u64;
            weights[i] = Self::error_weight(s, WP_W[i]);
        }
        let an = Self::add_bits(n);
        let aw = Self::add_bits(w);
        let ane = Self::add_bits(ne);
        let anw = Self::add_bits(nw);
        let ann = Self::add_bits(nn);
        let te_w = if x == 0 {
            0
        } else {
            self.error[cur_row + x - 1]
        };
        let te_n = self.error[pos_n];
        let te_nw = self.error[pos_nw];
        let te_ne = self.error[pos_ne];
        // kWPProp (p[15]): signed neighbour error with max abs value.
        let mut wpp = te_w;
        if te_n.abs() > wpp.abs() {
            wpp = te_n;
        }
        if te_nw.abs() > wpp.abs() {
            wpp = te_nw;
        }
        if te_ne.abs() > wpp.abs() {
            wpp = te_ne;
        }
        self.wp_prop = wpp;
        let s_wn = te_n + te_w;
        self.prediction[0] = aw + ane - an;
        self.prediction[1] = an - (((s_wn + te_ne) * WP_P1C) >> 5);
        self.prediction[2] = aw - (((s_wn + te_nw) * WP_P2C) >> 5);
        self.prediction[3] = an
            - ((te_nw * WP_P3CA
                + te_n * WP_P3CB
                + te_ne * WP_P3CC
                + (ann - an) * WP_P3CD
                + (anw - aw) * WP_P3CE)
                >> 5);
        let pred = Self::weighted_average(&self.prediction, &weights);
        if ((te_n ^ te_w) | (te_n ^ te_nw)) > 0 {
            self.pred = pred;
            (pred + WP_PRED_ROUND) >> WP_EXTRA_BITS
        } else {
            let mx = aw.max(ane).max(an);
            let mn = aw.min(ane).min(an);
            let predc = pred.max(mn).min(mx);
            // libjxl reassigns the member `pred` to the clamped value here, so
            // UpdateErrors uses the clamped prediction in this branch.
            self.pred = predc;
            (predc + WP_PRED_ROUND) >> WP_EXTRA_BITS
        }
    }
    /// Update error state with the true value `val` (normal space).
    #[inline]
    fn update(&mut self, val: i64, x: usize, y: usize) {
        let xsize = self.xsize;
        let cur_row = if y & 1 == 1 { 0 } else { xsize + 2 };
        let prev_row = if y & 1 == 1 { xsize + 2 } else { 0 };
        let valb = Self::add_bits(val);
        self.error[cur_row + x] = self.pred - valb;
        for i in 0..4 {
            let e = ((self.prediction[i] - valb).abs() + WP_PRED_ROUND) >> WP_EXTRA_BITS;
            self.pred_errors[i][cur_row + x] = e as u32;
            self.pred_errors[i][prev_row + x + 1] += e as u32;
        }
    }
}

const GROUP_DIM: usize = 256;
const LF_GROUP_DIM: usize = 2048;

const LZ77_MIN_SYMBOL: u32 = 64;
const LZ77_MIN_LENGTH: u32 = 3;
// Distance value we emit for run-length encoding (distance = 1 → previous token).
const LZ77_DIST_VALUE: u32 = 1; // special_distance[1] = (dx=1, dy=0) → 1 token back

pub(crate) fn encode_frame_lossless(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    num_color: usize,
    writer: &mut BitWriter,
) {
    // The hybrid-uint token of a modular residual can reach ~4*B+11 for B-bit
    // input. For B<=13 this stays below 64, so keep the tight default; for higher
    // bit depths raise LZ77_MIN_SYMBOL above the max literal token so the decoder
    // never mistakes a large residual for an LZ77 back-reference (the cause of
    // the old ~12-bit lossless ceiling). Stays under the 128-symbol prefix-code
    // alphabet through ~18-bit.
    let min_symbol: u32 = if max_bits <= 13 {
        LZ77_MIN_SYMBOL
    } else {
        4 * max_bits + 24
    };
    let xsize = linear.xsize();
    let ysize = linear.ysize();
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };

    let xsize_groups = xsize.div_ceil(GROUP_DIM);
    let ysize_groups = ysize.div_ceil(GROUP_DIM);
    let num_ac_groups = xsize_groups * ysize_groups;
    let xsize_dc_groups = xsize.div_ceil(LF_GROUP_DIM);
    let ysize_dc_groups = ysize.div_ceil(LF_GROUP_DIM);
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
    let single_group = num_ac_groups == 1;

    // Palette (v1): single-group, RGB (no alpha), <=256 distinct colors. Encodes
    // the image as a small palette meta-channel + an index channel, which is a
    // large win for low-color/graphic content. Falls through to the normal
    // RCT+predictor path when it doesn't apply.
    if single_group
        && num_color == 3
        && alpha.is_none()
        && try_encode_palette_single_group(linear, xsize, ysize, min_symbol, writer)
    {
        return;
    }

    // Stage-2 progressive lossless (opt-in): Squeeze pyramid (RGB + optional alpha).
    if single_group
        && num_color == 3
        && (progressive || std::env::var("JIXEL_SQUEEZE").as_deref() == Ok("1"))
    {
        encode_squeeze_single_group(linear, alpha, xsize, ysize, min_symbol, writer);
        return;
    }

    // Progressive multi-group (Stage A: all squeezed channels fit the global
    // stream). Falls through to the non-progressive path if a channel is still
    // larger than a group (Stage B).
    if !single_group
        && num_color == 3
        && (progressive || std::env::var("JIXEL_SQUEEZE").as_deref() == Ok("1"))
        && encode_squeeze_multigroup(
            linear,
            alpha,
            xsize,
            ysize,
            min_symbol,
            xsize_groups,
            ysize_groups,
            xsize_dc_groups,
            ysize_dc_groups,
            num_dc_groups,
            num_ac_groups,
            writer,
        )
    {
        return;
    }

    // Pick Gradient(5) or Weighted(6) per channel by estimated cost (one global
    // choice, used for the tree and every group). Never worse than all-WP.
    let predictors = choose_predictors(linear, alpha, xsize, ysize);
    // Contiguous per-modular-channel predictors: the `num_color` color channels
    // (Y for gray; Y/Co/Cg for color) followed by alpha. For 3-color this is just
    // predictors[..nb_chans]; for gray it is [Y_pred, (alpha_pred)].
    let chan_preds: Vec<u32> = {
        let mut v: Vec<u32> = (0..num_color).map(|c| predictors[c]).collect();
        if alpha.is_some() {
            v.push(predictors[3]);
        }
        v
    };

    // Context tree (v1): single-group. Splits each channel's entropy context on
    // the WP activity property; a big win on smooth+edge content. Falls through
    // to the flat path when it isn't estimated to help.
    if num_color == 3 {
        if single_group {
            if try_encode_context_tree_single_group(
                linear,
                alpha,
                xsize,
                ysize,
                &predictors,
                min_symbol,
                writer,
            ) {
                return;
            }
        } else if try_encode_context_tree_multi_group(
            linear,
            alpha,
            xsize,
            ysize,
            &predictors,
            xsize_groups,
            ysize_groups,
            num_dc_groups,
            min_symbol,
            writer,
        ) {
            return;
        }
    }

    write_frame_header_modular(alpha.is_some(), writer);

    if single_group {
        // Single section: GroupHeader + local tree + pixel histograms + pixels.
        let mut section = BitWriter::new();
        // 1 bit: dc_quant all_default = 1
        section.write(1, 1);
        // 1 bit: has_tree = 0  (no global tree; the local tree lives in the GroupHeader).
        section.write(1, 0);
        // GroupHeader: use_global_tree=0, wp_default=1, RCT transform on R/G/B.
        section.write(1, 0);
        section.write(1, 1);
        write_modular_transforms(nb_chans, &mut section);

        // Tokenize all channels (post-YCoCg, per-channel contexts).
        let tokens = tokenize_all(
            linear,
            alpha,
            xsize,
            ysize,
            0,
            0,
            xsize,
            ysize,
            num_color,
            &chan_preds,
        );

        // LZ77 layer: collapse runs of identical tokens into back-references.
        // The distance context is the (nb_chans)-th context, appended after the
        // per-channel ones.
        let distance_ctx = nb_chans as u32;
        let lz_tokens = lz77_compress(&tokens, distance_ctx);

        // Per-cluster prefix codes (nb_chans + 1 contexts), balanced N-leaf tree.
        let code = build_lz_pixel_code(&lz_tokens, nb_chans, min_symbol);
        write_local_tree_lz77(&chan_preds, &code, min_symbol, &mut section);

        // Emit the LZ77'd token stream.
        for t in &lz_tokens {
            write_lz_token(*t, &code, min_symbol, &mut section);
        }
        section.zero_pad_to_byte();

        // TOC.
        writer.write(1, 0); // no permutation
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append(&section);
        writer.zero_pad_to_byte();
    } else {
        // Multi-group: a single global tree + histograms in DC global, then each
        // AC group emits its tokens against those codes.
        let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
        let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

        // Tokenize each AC group (sub-image-local neighbours, matching what
        // we'll emit below) and run LZ77 over each group's stream separately so
        // back-references stay within a group's modular sub-image.  Pool the
        // resulting LzToken streams to build a single global prefix code so
        // every per-group emission is guaranteed to be representable.
        let distance_ctx = nb_chans as u32;
        let mut group_lz_tokens: Vec<Vec<LzToken>> = Vec::with_capacity(num_ac_groups);
        let mut all_lz: Vec<LzToken> = Vec::new();
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);
                let toks = tokenize_all(
                    linear,
                    alpha,
                    xsize,
                    ysize,
                    x0,
                    y0,
                    gw,
                    gh,
                    num_color,
                    &chan_preds,
                );
                let lz = lz77_compress(&toks, distance_ctx);
                all_lz.extend_from_slice(&lz);
                group_lz_tokens.push(lz);
            }
        }
        let code = build_lz_pixel_code(&all_lz, nb_chans, min_symbol);

        // ----- Section 0: DC global -----
        sections[0].write(1, 1); // dc_quant all_default = 1
        sections[0].write(1, 1); // has_tree = 1
        write_local_tree_lz77(&chan_preds, &code, min_symbol, &mut sections[0]);
        // GroupHeader for the global modular image: use_global_tree=1, wp=1, RCT transform.
        sections[0].write(1, 1);
        sections[0].write(1, 1);
        write_modular_transforms(nb_chans, &mut sections[0]);
        sections[0].zero_pad_to_byte();

        // ----- DC groups: empty GroupHeader only -----
        for i in 0..num_dc_groups {
            sections[1 + i].write(1, 1); // use_global_tree
            sections[1 + i].write(1, 1); // wp_default
            sections[1 + i].write(2, 0); // 0 transforms
            sections[1 + i].zero_pad_to_byte();
        }

        // ----- AC global: trivial (all_default flags) -----
        let ac_global_idx = 1 + num_dc_groups;
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].zero_pad_to_byte();

        // ----- AC groups: pixel data per group -----
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let group_index = gy * xsize_groups + gx;
                let section_idx = 2 + num_dc_groups + group_index;

                // GroupHeader: use_global_tree=1, wp=1, 0 transforms (the global
                // header already declared the RCT for the whole image).
                sections[section_idx].write(1, 1);
                sections[section_idx].write(1, 1);
                sections[section_idx].write(2, 0);

                for t in &group_lz_tokens[group_index] {
                    write_lz_token(*t, &code, min_symbol, &mut sections[section_idx]);
                }
                sections[section_idx].zero_pad_to_byte();
            }
        }

        // TOC.
        writer.write(1, 0);
        writer.zero_pad_to_byte();
        for s in &sections {
            write_toc_entry(s.bits_written() / 8, writer);
        }
        writer.zero_pad_to_byte();
        for s in &sections {
            writer.append(s);
            writer.zero_pad_to_byte();
        }
    }
}

fn write_frame_header_modular(has_alpha: bool, w: &mut BitWriter) {
    w.write(1, 0); // all_default = false
    w.write(2, 0b00); // regular frame
    w.write(1, 1); // encoding = Modular
    w.write(2, 0b00); // flags = u64(0)
    w.write(1, 0); // do_ycbcr = false   (xyb_encoded=0 so this is serialized)
    w.write(2, 0b00); // upsampling = 1
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(2, 0b01); // group_size_shift = 1 (256-pixel groups)
    w.write(2, 0b00); // num_passes = 1
    w.write(1, 0); // have_crop = false
    w.write(2, 0b00); // blending = Replace
    if has_alpha {
        w.write(2, 0b00);
    }
    w.write(1, 1); // is_last
    w.write(2, 0b00); // name length = 0
    w.write(1, 0); // loop_filter NOT all_default
    w.write(1, 0); // no gaborish
    w.write(2, 0); // 0 EPF iters
    w.write(2, 0b00); // no LF extensions
    w.write(2, 0b00); // no FH extensions
}

fn write_modular_transforms(nb_chans: usize, w: &mut BitWriter) {
    if nb_chans >= 3 {
        // transforms count u2S(0, 1, Bits(4)+2, Bits(8)+18): selector 1 = Val(1) → 1 transform.
        w.write(2, 0b01);
        // Transform[0]:
        //   id Bits(2)            = 0 (RCT)
        //   begin_channel u2S(Bits(3), ...): selector 0 = Bits(3) → value 0 = 5 bits "00000"
        //   rct_type u2S(6, ...):  selector 0 = Val(6) → 2 bits "00"
        w.write(2, 0b00); // id = RCT (Bits(2))
        w.write(2, 0b00); // begin_channel selector 0
        w.write(3, 0); // begin_channel value (Bits(3)) = 0
        w.write(2, 0b00); // rct_type selector 0 = Val(6) = YCoCg
    } else {
        w.write(2, 0b00); // 0 transforms
    }
}

/// Transform list for RCT + a Squeeze with the given step sequence.
/// Byte-exact to libjxl Transform/SqueezeParams field encoding.
fn write_modular_transforms_rct_squeeze(steps: &[crate::squeeze::SqueezeStep], w: &mut BitWriter) {
    // transforms count = 2 -> selector 2 + Bits(4)=0.
    w.write(2, 0b10);
    w.write(4, 0);
    // Transform[0] = RCT (YCoCg), begin_c=0.
    w.write(2, 0b00);
    w.write(2, 0b00);
    w.write(3, 0);
    w.write(2, 0b00);
    // Transform[1] = Squeeze.
    w.write(2, 0b10); // id = kSqueeze
    // num_squeezes: U32(Val0, BitsOffset(4,1), BitsOffset(6,9), BitsOffset(8,41)).
    let n = steps.len() as u32;
    if n == 0 {
        w.write(2, 0b00);
    } else if n <= 16 {
        w.write(2, 0b01);
        w.write(4, (n - 1) as u64);
    } else if n <= 72 {
        w.write(2, 0b10);
        w.write(6, (n - 9) as u64);
    } else {
        w.write(2, 0b11);
        w.write(8, (n - 41) as u64);
    }
    for s in steps {
        w.write(1, if s.horizontal { 1 } else { 0 });
        w.write(1, if s.in_place { 1 } else { 0 });
        // begin_c: U32(Bits3, ...): for small values use selector 0 = Bits(3).
        debug_assert!(s.begin_c < 8);
        w.write(2, 0b00);
        w.write(3, s.begin_c as u64);
        // num_c: U32(Val1,Val2,Val3,BitsOffset(4,4)).
        match s.num_c {
            1 => w.write(2, 0b00),
            2 => w.write(2, 0b01),
            3 => w.write(2, 0b10),
            n => {
                w.write(2, 0b11);
                w.write(4, (n - 4) as u64);
            }
        }
    }
}

/// Stage-2 progressive-lossless path (opt-in, JIXEL_SQUEEZE=1): single-group,
/// no alpha. Applies one horizontal in-place Squeeze to the RCT'd channels,
/// producing [avgY,avgCo,avgCg,resY,resCo,resCg], then tokenizes the 6 channels
/// with the Gradient predictor. Lossless: djxl reconstructs bit-exact after
/// inverse-Squeeze + inverse-RCT; the 3 half-width avg channels form the preview.
fn encode_squeeze_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    writer: &mut BitWriter,
) {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
    // Lift the 3 RCT'd planes (and alpha, if present) into channels.
    let num_c = if alpha.is_some() { 4 } else { 3 };
    let mut channels: Vec<Channel> = Vec::with_capacity(num_c);
    for c in 0..3usize {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            let row = linear.plane_row(c, y);
            ch.data[y * xsize..y * xsize + xsize].copy_from_slice(&row[..xsize]);
        }
        channels.push(ch);
    }
    if let Some(a) = alpha {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            for x in 0..xsize {
                ch.data[y * xsize + x] = a.get_i32(y * xsize + x);
            }
        }
        channels.push(ch);
    }
    // Apply the alternating H/V pyramid over all channels (explicit sequence).
    let steps = default_squeeze_steps(xsize, ysize, num_c);
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();

    // Per-channel predictor selection (Gradient vs Weighted) by estimated cost,
    // mirroring the main path — recovers the size squeeze otherwise leaves on
    // the table (residual channels often prefer a different predictor).
    let predictors: Vec<u32> = channels
        .iter()
        .map(|ch| {
            let data = &ch.data;
            let w = ch.w;
            let get = move |gx: usize, gy: usize| data[gy * w + gx];
            let bg = estimate_channel_bits(&get, ch.w, ch.h, PREDICTOR_GRADIENT);
            let bw = estimate_channel_bits(&get, ch.w, ch.h, PREDICTOR_WEIGHTED);
            if bw <= bg {
                PREDICTOR_WEIGHTED
            } else {
                PREDICTOR_GRADIENT
            }
        })
        .collect();
    let mut tokens: Vec<Token> = Vec::new();
    for (c, ch) in channels.iter().enumerate() {
        let ctx = channel_to_context(c, nb);
        let data = &ch.data;
        let w = ch.w;
        let get = move |gx: usize, gy: usize| data[gy * w + gx];
        tokenize_plane(ctx, &get, ch.w, ch.h, predictors[c], &mut tokens);
    }

    write_frame_header_modular(alpha.is_some(), writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut section);

    let distance_ctx = nb as u32;
    let lz_tokens = lz77_compress(&tokens, distance_ctx);
    let code = build_lz_pixel_code(&lz_tokens, nb, min_symbol);
    write_local_tree_lz77(&predictors, &code, min_symbol, &mut section);
    for t in &lz_tokens {
        write_lz_token(*t, &code, min_symbol, &mut section);
    }
    section.zero_pad_to_byte();

    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append(&section);
    writer.zero_pad_to_byte();
}

/// Stage-2 progressive lossless, multi-group (image > one AC group).
///
/// Applies the Squeeze pyramid to the whole frame, then splits channels per the
/// JXL rule: channels that fit a group form the global modular image (LfGlobal /
/// section 0); larger channels are partitioned into AC groups, each carrying the
/// `frame_rect >> shift` crop of every large channel. A single global tree (chain
/// on the channel property) serves all streams; the within-group channel index is
/// what the tree sees, so group crops share contexts with the leading globals.
#[allow(clippy::too_many_arguments)]
fn encode_squeeze_multigroup(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    xsize_groups: usize,
    ysize_groups: usize,
    xsize_dc_groups: usize,
    ysize_dc_groups: usize,
    num_dc_groups: usize,
    num_ac_groups: usize,
    writer: &mut BitWriter,
) -> bool {
    use crate::squeeze::{Channel, apply_step_forward, default_squeeze_steps};
    let num_c = if alpha.is_some() { 4 } else { 3 };
    let mut channels: Vec<Channel> = Vec::with_capacity(num_c);
    for c in 0..3usize {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            let row = linear.plane_row(c, y);
            ch.data[y * xsize..y * xsize + xsize].copy_from_slice(&row[..xsize]);
        }
        channels.push(ch);
    }
    if let Some(a) = alpha {
        let mut ch = Channel::new(xsize, ysize);
        for y in 0..ysize {
            for x in 0..xsize {
                ch.data[y * xsize + x] = a.get_i32(y * xsize + x);
            }
        }
        channels.push(ch);
    }
    let steps = default_squeeze_steps(xsize, ysize, num_c);
    for s in &steps {
        apply_step_forward(&mut channels, s);
    }
    let nb = channels.len();

    // Channel -> stream split (JXL rule): the leading run of channels that fit a
    // group is the global modular image; the first channel still larger than a
    // group starts the suffix that is partitioned into AC groups.
    let split = channels
        .iter()
        .position(|c| c.w > GROUP_DIM || c.h > GROUP_DIM)
        .unwrap_or(nb);

    let predictors: Vec<u32> = channels
        .iter()
        .map(|ch| {
            let data = &ch.data;
            let w = ch.w;
            let get = move |gx: usize, gy: usize| data[gy * w + gx];
            let bg = estimate_channel_bits(&get, ch.w, ch.h, PREDICTOR_GRADIENT);
            let bw = estimate_channel_bits(&get, ch.w, ch.h, PREDICTOR_WEIGHTED);
            if bw <= bg {
                PREDICTOR_WEIGHTED
            } else {
                PREDICTOR_GRADIENT
            }
        })
        .collect();

    let distance_ctx = nb as u32;

    // Global stream: the small channels [0, split), whole.
    let mut global_tokens: Vec<Token> = Vec::new();
    for c in 0..split {
        let ch = &channels[c];
        let ctx = channel_to_context(c, nb);
        let data = &ch.data;
        let w = ch.w;
        let get = move |gx: usize, gy: usize| data[gy * w + gx];
        tokenize_plane(ctx, &get, ch.w, ch.h, predictors[c], &mut global_tokens);
    }
    let global_lz = lz77_compress(&global_tokens, distance_ctx);
    let mut all_lz: Vec<LzToken> = global_lz.clone();

    // One group's worth of cropped large-channel tokens. `gdim` is the group's
    // frame-space size (GROUP_DIM for AC, LF_GROUP_DIM for DC); a channel is
    // included when min(hshift,vshift) is inside [minsh,maxsh], and contributes
    // its `(group_rect >> shift)` crop. The decoder rebuilds the same scan, so
    // the within-group index (sequential over non-empty crops) is exactly the
    // `chan` property the global tree keys on.
    let crop_group = |gdim: usize, gx: usize, gy: usize, minsh: i32, maxsh: i32| -> Vec<LzToken> {
        let mut gtok: Vec<Token> = Vec::new();
        let mut within = 0usize;
        for c in split..nb {
            let ch = &channels[c];
            let msh = ch.hshift.min(ch.vshift);
            if msh < minsh || msh > maxsh {
                continue;
            }
            let hs = ch.hshift as usize;
            let vs = ch.vshift as usize;
            let rx0 = (gx * gdim) >> hs;
            let ry0 = (gy * gdim) >> vs;
            if rx0 >= ch.w || ry0 >= ch.h {
                continue;
            }
            let rw = (gdim >> hs).min(ch.w - rx0);
            let rh = (gdim >> vs).min(ch.h - ry0);
            if rw == 0 || rh == 0 {
                continue;
            }
            let ctx = channel_to_context(within, nb);
            let pred = predictors[within];
            let data = &ch.data;
            let w = ch.w;
            let get = move |lx: usize, ly: usize| data[(ry0 + ly) * w + (rx0 + lx)];
            tokenize_plane(ctx, &get, rw, rh, pred, &mut gtok);
            within += 1;
        }
        lz77_compress(&gtok, distance_ctx)
    };

    // DC (LF) groups carry the deeply-squeezed large channels (min shift >= 3),
    // partitioned into LF_GROUP_DIM rects. Empty unless the image is large enough
    // that a >=3x-squeezed channel still exceeds a group (dimension > ~2048).
    let mut dc_group_lz: Vec<Vec<LzToken>> = Vec::with_capacity(num_dc_groups);
    for rgy in 0..ysize_dc_groups {
        for rgx in 0..xsize_dc_groups {
            let glz = crop_group(LF_GROUP_DIM, rgx, rgy, 3, 1000);
            all_lz.extend_from_slice(&glz);
            dc_group_lz.push(glz);
        }
    }

    // AC groups carry the shallow large channels (min shift <= 2), in GROUP_DIM rects.
    let mut ac_group_lz: Vec<Vec<LzToken>> = Vec::with_capacity(num_ac_groups);
    for gy in 0..ysize_groups {
        for gx in 0..xsize_groups {
            let glz = crop_group(GROUP_DIM, gx, gy, 0, 2);
            all_lz.extend_from_slice(&glz);
            ac_group_lz.push(glz);
        }
    }

    let code = build_lz_pixel_code(&all_lz, nb, min_symbol);

    write_frame_header_modular(alpha.is_some(), writer);

    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    // ----- Section 0: LfGlobal = global tree + global modular image -----
    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_local_tree_lz77(&predictors, &code, min_symbol, &mut sections[0]);
    sections[0].write(1, 1); // use_global_tree = 1
    sections[0].write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut sections[0]);
    for t in &global_lz {
        write_lz_token(*t, &code, min_symbol, &mut sections[0]);
    }
    sections[0].zero_pad_to_byte();

    // ----- DC groups: GroupHeader + any min-shift>=3 large-channel crops -----
    for i in 0..num_dc_groups {
        sections[1 + i].write(1, 1); // use_global_tree
        sections[1 + i].write(1, 1); // wp_default
        sections[1 + i].write(2, 0); // 0 transforms (declared globally)
        for t in &dc_group_lz[i] {
            write_lz_token(*t, &code, min_symbol, &mut sections[1 + i]);
        }
        sections[1 + i].zero_pad_to_byte();
    }

    // ----- AC global: trivial -----
    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    // ----- AC groups: GroupHeader + the cropped large-channel tokens -----
    for g in 0..num_ac_groups {
        let idx = 2 + num_dc_groups + g;
        sections[idx].write(1, 1); // use_global_tree
        sections[idx].write(1, 1); // wp_default
        sections[idx].write(2, 0); // 0 transforms (declared globally)
        for t in &ac_group_lz[g] {
            write_lz_token(*t, &code, min_symbol, &mut sections[idx]);
        }
        sections[idx].zero_pad_to_byte();
    }

    // TOC + sections.
    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    for s in &sections {
        writer.append(s);
        writer.zero_pad_to_byte();
    }
    true
}

/// Serialize a single Palette transform (mirrors `Transform::VisitFields` for
/// `TransformId::kPalette`). v1: begin_c=0, num_c=3, nb_deltas=0, predictor=Zero.
fn write_palette_transform(num_c: u32, nb_colors: u32, w: &mut BitWriter) {
    debug_assert_eq!(num_c, 3);
    // transforms count U32(Val(0),Val(1),...): selector 1 = Val(1) = 1 transform.
    w.write(2, 0b01);
    // id U32(Val(RCT=0),Val(Palette=1),...): selector 1 = Palette.
    w.write(2, 0b01);
    // begin_c U32(Bits(3),...): selector 0 = Bits(3), value 0.
    w.write(2, 0b00);
    w.write(3, 0);
    // num_c U32(Val(1),Val(3),Val(4),BitsOffset(13,1)): num_c=3 -> selector 1 = Val(3).
    w.write(2, 0b01);
    // nb_colors U32(BitsOffset(8,0),BitsOffset(10,256),...).
    if nb_colors <= 255 {
        w.write(2, 0b00); // selector 0 = BitsOffset(8, 0)
        w.write(8, nb_colors as u64);
    } else {
        // nb_colors == 256 (our cap): selector 1 = BitsOffset(10, 256).
        w.write(2, 0b01);
        w.write(10, (nb_colors - 256) as u64);
    }
    // nb_deltas U32(Val(0),...): selector 0 = Val(0) = 0.
    w.write(2, 0b00);
    // predictor Bits(4): Predictor::Zero = 0.
    w.write(4, 0);
}

/// v1 palette path: single-group, RGB (no alpha), <=256 distinct colors.
/// Encodes a palette meta-channel (h=num_c=3, w=nb_colors) + an index channel
/// (h=ysize, w=xsize), declaring a Palette transform so the decoder's
/// `InvPalette` reconstructs RGB directly (no RCT). Returns true and writes the
/// full frame if palette applies; false (writing nothing) otherwise.
fn try_encode_palette_single_group(
    linear: &Image3Si,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    writer: &mut BitWriter,
) -> bool {
    use std::collections::HashMap;
    let npx = xsize * ysize;

    // 1) Reconstruct RGB from YCoCg, collect distinct colors (bail past 256).
    let mut color_of: Vec<(i32, i32, i32)> = Vec::with_capacity(npx);
    let mut seen: HashMap<(i32, i32, i32), ()> = HashMap::new();
    for gy in 0..ysize {
        let yr = linear.plane_row(0, gy);
        let cor = linear.plane_row(1, gy);
        let cgr = linear.plane_row(2, gy);
        for gx in 0..xsize {
            let c = inverse_ycocg(yr[gx], cor[gx], cgr[gx]);
            color_of.push(c);
            seen.entry(c).or_insert(());
            if seen.len() > 256 {
                return false;
            }
        }
    }
    let nb_colors = seen.len();
    if nb_colors == 0 {
        return false;
    }

    // 2) Sorted palette + color->index map.
    let mut colors: Vec<(i32, i32, i32)> = seen.keys().copied().collect();
    colors.sort_unstable();
    let mut idx_of: HashMap<(i32, i32, i32), u32> = HashMap::with_capacity(nb_colors);
    for (i, c) in colors.iter().enumerate() {
        idx_of.insert(*c, i as u32);
    }

    // 3) Palette meta-channel (row c = component c of each color) + index channel.
    let mut palette_ch = vec![0i32; 3 * nb_colors];
    for (i, &(r, g, b)) in colors.iter().enumerate() {
        palette_ch[i] = r;
        palette_ch[nb_colors + i] = g;
        palette_ch[2 * nb_colors + i] = b;
    }
    let index_img: Vec<i32> = color_of.iter().map(|c| idx_of[c] as i32).collect();

    // 4) Channel accessors (channel 0 = palette, channel 1 = index).
    let pget = |gx: usize, gy: usize| palette_ch[gy * nb_colors + gx];
    let iget = |gx: usize, gy: usize| index_img[gy * xsize + gx];

    // 5) Per-channel predictor selection (gradient vs WP).
    let pick = |get: &dyn Fn(usize, usize) -> i32, w: usize, h: usize| -> u32 {
        let bg = estimate_channel_bits(get, w, h, PREDICTOR_GRADIENT);
        let bw = estimate_channel_bits(get, w, h, PREDICTOR_WEIGHTED);
        if bw <= bg {
            PREDICTOR_WEIGHTED
        } else {
            PREDICTOR_GRADIENT
        }
    };
    let preds = [
        pick(&pget, nb_colors, 3), // channel 0 = palette
        pick(&iget, xsize, ysize), // channel 1 = index
    ];

    // 6) Frame header + single section (mirrors the RGB single-group layout).
    write_frame_header_modular(false, writer);

    let nb_chans = 2usize;
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_palette_transform(3, nb_colors as u32, &mut section);

    let mut tokens: Vec<Token> = Vec::with_capacity(3 * nb_colors + npx);
    tokenize_plane(
        channel_to_context(0, nb_chans),
        &pget,
        nb_colors,
        3,
        preds[0],
        &mut tokens,
    );
    tokenize_plane(
        channel_to_context(1, nb_chans),
        &iget,
        xsize,
        ysize,
        preds[1],
        &mut tokens,
    );

    let distance_ctx = nb_chans as u32;
    let lz_tokens = lz77_compress(&tokens, distance_ctx);
    let code = build_lz_pixel_code(&lz_tokens, nb_chans, min_symbol);
    write_local_tree_lz77(&preds, &code, min_symbol, &mut section);
    for t in &lz_tokens {
        write_lz_token(*t, &code, min_symbol, &mut section);
    }
    section.zero_pad_to_byte();

    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append(&section);
    writer.zero_pad_to_byte();
    true
}

fn write_toc_entry(byte_len: usize, w: &mut BitWriter) {
    const OFFSETS: [usize; 4] = [0, 1024, 17_408, 4_211_712];
    const BITS: [usize; 4] = [10, 14, 22, 30];
    let mut bucket = 0usize;
    while bucket < 3 && byte_len >= OFFSETS[bucket + 1] {
        bucket += 1;
    }
    w.write(2, bucket as u64);
    w.write(BITS[bucket], (byte_len - OFFSETS[bucket]) as u64);
}

// ---------------------------------------------------------------------------
// Forward reversible YCoCg (RCT type 6, matches libjxl's InvRCTRow<6>).
//
// Encoder:
//   co  = r - b
//   tmp = b + (co >> 1)
//   cg  = g - tmp
//   y   = tmp + (cg >> 1)
//
// Decoder undoes this with the exact same shift sequence:
//   tmp = y - (cg >> 1);  g = cg + tmp;
//   y'  = tmp - (co >> 1); r = y' + co;  b = y'
//
// Reversible because every operation is invertible without rounding.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn forward_ycocg(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    (y, co, cg)
}

/// Exact integer inverse of `forward_ycocg` (undoes the steps in reverse).
#[inline]
pub(crate) fn inverse_ycocg(y: i32, co: i32, cg: i32) -> (i32, i32, i32) {
    let tmp = y - (cg >> 1);
    let g = cg + tmp;
    let b = tmp - (co >> 1);
    let r = b + co;
    (r, g, b)
}

#[inline]
fn channel_to_context(chan: usize, nb_chans: usize) -> u32 {
    (nb_chans - 1 - chan) as u32
}

// ---------------------------------------------------------------------------
// LZ77 hybrid-uint encoding (length values: split_exp=4, msb=0, lsb=0).
// ---------------------------------------------------------------------------

/// Hybrid encode of `length_value` (= run_length - min_length).
/// Returns (alphabet_token, nbits, payload).  The actual alphabet symbol is
/// `LZ77_MIN_SYMBOL + alphabet_token`.
#[inline]
fn lz77_length_encode(length_value: u32) -> (u32, u32, u32) {
    // hybrid uint with split_exponent = 4, msb_in_token = 0, lsb_in_token = 0
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

// ---------------------------------------------------------------------------
// LZ77 layer.
// ---------------------------------------------------------------------------

/// A unit of emission in the LZ77'd token stream.
#[derive(Clone, Copy)]
enum LzToken {
    /// A regular residual token, written via the normal `uint_encode` path.
    Pixel { context: u32, value: u32 },
    /// LZ77 back-reference: copy `length_value + LZ77_MIN_LENGTH` previous tokens.
    /// Emitted as a length symbol on the `pixel_context` channel, then a
    /// distance symbol on the distance context.
    Lz77 {
        pixel_context: u32,
        distance_context: u32,
        length_value: u32,
    },
}

/// Compact a sequence of `Token`s by back-referencing runs of identical
/// (context, value) pairs.  Returns the LZ77'd stream.
fn lz77_compress(tokens: &[Token], distance_context: u32) -> Vec<LzToken> {
    let mut out: Vec<LzToken> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        out.push(LzToken::Pixel {
            context: t.context,
            value: t.value,
        });
        // Find how many subsequent tokens are identical to `t`.
        let mut j = i + 1;
        while j < tokens.len() && tokens[j].context == t.context && tokens[j].value == t.value {
            j += 1;
        }
        let run_extra = (j - i - 1) as u32; // copies after the first one
        if run_extra >= LZ77_MIN_LENGTH {
            out.push(LzToken::Lz77 {
                pixel_context: t.context,
                distance_context,
                length_value: run_extra - LZ77_MIN_LENGTH,
            });
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn tokenize_all(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    _ysize: usize,
    x0: usize,
    y0: usize,
    gw: usize,
    gh: usize,
    num_color: usize,
    predictors: &[u32],
) -> Vec<Token> {
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };
    let mut out = Vec::with_capacity(gw * gh * nb_chans);

    // Color channels: plane 0 (Y/luma) for gray, planes 0..3 (Y/Co/Cg) for color.
    for chan in 0..num_color {
        let ctx = channel_to_context(chan, nb_chans);
        let get = |gx: usize, gy: usize| linear.plane_row(chan, y0 + gy)[x0 + gx];
        tokenize_plane(ctx, &get, gw, gh, predictors[chan], &mut out);
    }

    // Alpha (untransformed) is the modular channel right after the color ones.
    if let Some(a) = alpha {
        let ctx = channel_to_context(num_color, nb_chans);
        let get = |gx: usize, gy: usize| a.get_i32((y0 + gy) * xsize + (x0 + gx));
        tokenize_plane(ctx, &get, gw, gh, predictors[num_color], &mut out);
    }

    out
}

/// Tokenize one channel's group-local rectangle with the chosen predictor
/// (`PREDICTOR_GRADIENT` or `PREDICTOR_WEIGHTED`). Neighbors use libjxl's exact
/// border conventions (see weighted::State / Predict()):
///   left = x>0 ? W : (y>0 ? N : 0); top = y>0 ? N : left;
///   topleft = (x&&y) ? NW : left; topright = (x+1<w && y) ? NE : top;
///   toptop  = y>1 ? NN : top.
fn tokenize_plane(
    ctx: u32,
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
    out: &mut Vec<Token>,
) {
    let mut wp = WpState::new(gw);
    for gy in 0..gh {
        for gx in 0..gw {
            let v = get(gx, gy) as i64;
            let w_ = if gx > 0 {
                get(gx - 1, gy) as i64
            } else if gy > 0 {
                get(gx, gy - 1) as i64
            } else {
                0
            };
            let n_ = if gy > 0 { get(gx, gy - 1) as i64 } else { w_ };
            let nw_ = if gx > 0 && gy > 0 {
                get(gx - 1, gy - 1) as i64
            } else {
                w_
            };
            let ne_ = if gx + 1 < gw && gy > 0 {
                get(gx + 1, gy - 1) as i64
            } else {
                n_
            };
            let nn_ = if gy > 1 { get(gx, gy - 2) as i64 } else { n_ };
            // The WP state must be updated for every pixel whenever the channel
            // uses the Weighted predictor (matches the decoder, which updates wp
            // for every pixel when the tree references predictor 6). For the
            // Gradient channels the WP state is unused, so we skip it.
            let pred = if pred_id == PREDICTOR_WEIGHTED {
                let p = wp.predict(gx, gy, n_, w_, ne_, nw_, nn_);
                wp.update(v, gx, gy);
                p
            } else {
                // Gradient with the same border convention (== libjxl ClampedGradient).
                let lo = w_.min(n_);
                let hi = w_.max(n_);
                (w_ + n_ - nw_).clamp(lo, hi)
            };
            out.push(Token::new(ctx, pack_signed((v - pred) as i32)));
        }
    }
}

/// Estimate the coded cost (order-0 residual entropy, in bits) of one channel
/// over the whole image under a given predictor. Used only to pick the cheaper
/// predictor per channel; the choice never affects decodability (the tree tells
/// the decoder which predictor each channel uses).
fn estimate_channel_bits(
    get: impl Fn(usize, usize) -> i32,
    w: usize,
    h: usize,
    pred_id: u32,
) -> f64 {
    let mut toks: Vec<Token> = Vec::with_capacity(w * h);
    tokenize_plane(0, get, w, h, pred_id, &mut toks);
    if toks.is_empty() {
        return 0.0;
    }
    // order-0 entropy of the packed residual values via a direct-indexed
    // histogram (deterministic order, no hashing).
    let max = toks.iter().map(|t| t.value).max().unwrap_or(0) as usize;
    let mut hist = vec![0u64; max + 1];
    for t in &toks {
        hist[t.value as usize] += 1;
    }
    let total = toks.len() as f64;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c != 0 {
            let p = c as f64 / total;
            bits -= c as f64 * p.log2();
        }
    }
    bits
}

/// Choose Gradient(5) or Weighted(6) per channel by estimated cost over the
/// whole image. Guarantees the lossless stream is never worse than all-Gradient
/// or all-WP. Returns one predictor id per channel (indices 0..nb_chans).
/// Compute the order-0 entropy estimate for BOTH the Gradient and Weighted
/// predictors in a single traversal. Equivalent to calling
/// `estimate_channel_bits` twice (Gradient then Weighted) — same neighbour
/// convention, same WP state evolution, same packed residuals — but the pixel
/// fetches and loop overhead happen once instead of twice. Returns
/// `(gradient_bits, weighted_bits)`.
fn estimate_grad_and_wp_bits(get: impl Fn(usize, usize) -> i32, w: usize, h: usize) -> (f64, f64) {
    if w == 0 || h == 0 {
        return (0.0, 0.0);
    }
    let mut wp = WpState::new(w);
    let mut grad: Vec<u32> = Vec::with_capacity(w * h);
    let mut wpr: Vec<u32> = Vec::with_capacity(w * h);
    for gy in 0..h {
        for gx in 0..w {
            let v = get(gx, gy) as i64;
            let w_ = if gx > 0 {
                get(gx - 1, gy) as i64
            } else if gy > 0 {
                get(gx, gy - 1) as i64
            } else {
                0
            };
            let n_ = if gy > 0 { get(gx, gy - 1) as i64 } else { w_ };
            let nw_ = if gx > 0 && gy > 0 {
                get(gx - 1, gy - 1) as i64
            } else {
                w_
            };
            let ne_ = if gx + 1 < w && gy > 0 {
                get(gx + 1, gy - 1) as i64
            } else {
                n_
            };
            let nn_ = if gy > 1 { get(gx, gy - 2) as i64 } else { n_ };
            // Weighted (advances WP state exactly as the WP-only pass does).
            let wp_pred = wp.predict(gx, gy, n_, w_, ne_, nw_, nn_);
            wp.update(v, gx, gy);
            wpr.push(pack_signed((v - wp_pred) as i32));
            // Gradient (== libjxl ClampedGradient, same border convention).
            let lo = w_.min(n_);
            let hi = w_.max(n_);
            let g_pred = (w_ + n_ - nw_).clamp(lo, hi);
            grad.push(pack_signed((v - g_pred) as i32));
        }
    }
    (order0_entropy(&grad), order0_entropy(&wpr))
}

fn choose_predictors(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
) -> [u32; 4] {
    let mut preds = [PREDICTOR_WEIGHTED; 4];
    for chan in 0..3usize {
        let pd = linear.plane_data(chan);
        let get = |gx: usize, gy: usize| pd[gy * xsize + gx];
        let (bg, bw) = estimate_grad_and_wp_bits(&get, xsize, ysize);
        preds[chan] = if bw <= bg {
            PREDICTOR_WEIGHTED
        } else {
            PREDICTOR_GRADIENT
        };
    }
    if let Some(a) = alpha {
        let get = |gx: usize, gy: usize| a.get_i32(gy * xsize + gx);
        let (bg, bw) = estimate_grad_and_wp_bits(&get, xsize, ysize);
        preds[3] = if bw <= bg {
            PREDICTOR_WEIGHTED
        } else {
            PREDICTOR_GRADIENT
        };
    }
    preds
}

// ---------------------------------------------------------------------------
// Context tree (MA tree): split each channel's entropy context on the WP error
// property kWPProp (p[15]) into 3 activity buckets. The decoder must run the WP
// state for every pixel of a channel whose subtree references p[15], so we run
// WP for all channels here (the leaf predictor still chooses gradient vs WP).
// ---------------------------------------------------------------------------

const PROP_WP: u32 = 15; // kNumStaticProperties(2) + 13

/// Encoder-side tree: Split(property, splitval, gt-branch, le-branch) routes
/// `prop > splitval` to the gt-branch. Leaf carries (predictor, tag) where
/// tag = channel*3 + bucket.
enum CtTree {
    Split(u32, i32, Box<CtTree>, Box<CtTree>),
    Leaf(u32, u32),
}

/// BFS-emit the tree (matches libjxl's FIFO tree decode) and return the context
/// id assigned to each leaf tag. Context id == leaf's order of appearance.
fn emit_ct_tree(root: &CtTree, out: &mut Vec<Token>) -> std::collections::HashMap<u32, u32> {
    use std::collections::{HashMap, VecDeque};
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut q: VecDeque<&CtTree> = VecDeque::new();
    q.push_back(root);
    let mut ctx = 0u32;
    while let Some(node) = q.pop_front() {
        match node {
            CtTree::Split(prop, val, gt, le) => {
                push_split(out, *prop, *val);
                q.push_back(gt);
                q.push_back(le);
            }
            CtTree::Leaf(pred, tag) => {
                push_leaf(out, *pred);
                map.insert(*tag, ctx);
                ctx += 1;
            }
        }
    }
    map
}

#[inline]
fn bucket_of(prop: i64, t: i64) -> u32 {
    if prop > t {
        2
    } else if prop > -t - 1 {
        1
    } else {
        0
    }
}

/// 3-leaf activity subtree for channel `c` with predictor `pred` and threshold `t`.
fn act_sub(c: u32, pred: u32, t: i32) -> CtTree {
    CtTree::Split(
        PROP_WP,
        t,
        Box::new(CtTree::Leaf(pred, c * 3 + 2)),
        Box::new(CtTree::Split(
            PROP_WP,
            -t - 1,
            Box::new(CtTree::Leaf(pred, c * 3 + 1)),
            Box::new(CtTree::Leaf(pred, c * 3)),
        )),
    )
}

/// Channel-split tree (same shape as build_balanced_tree_tokens) with each
/// channel-leaf replaced by its activity subtree.
fn build_context_tree(nb_chans: usize, preds: &[u32], t: &[i32]) -> CtTree {
    let a = |c: usize| act_sub(c as u32, preds[c], t[c]);
    match nb_chans {
        1 => a(0),
        2 => CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0))),
        3 => CtTree::Split(
            0,
            1,
            Box::new(a(2)),
            Box::new(CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0)))),
        ),
        4 => CtTree::Split(
            0,
            1,
            Box::new(CtTree::Split(0, 2, Box::new(a(3)), Box::new(a(2)))),
            Box::new(CtTree::Split(0, 0, Box::new(a(1)), Box::new(a(0)))),
        ),
        _ => unreachable!("context tree supports 1..=4 channels"),
    }
}

#[inline]
fn clamped_gradient(w: i64, n: i64, nw: i64) -> i64 {
    let lo = w.min(n);
    let hi = w.max(n);
    (w + n - nw).clamp(lo, hi)
}

fn order0_entropy(vals: &[u32]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    // Direct-indexed frequency histogram (residual symbols are small-range), in
    // place of a HashMap: no hashing, and a deterministic accumulation order.
    let max = vals.iter().copied().max().unwrap_or(0) as usize;
    let mut hist = vec![0u64; max + 1];
    for &v in vals {
        hist[v as usize] += 1;
    }
    let total = vals.len() as f64;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c != 0 {
            let p = c as f64 / total;
            bits -= c as f64 * p.log2();
        }
    }
    bits
}

/// Run WP over one channel's group rectangle, returning per-pixel
/// (packed residual under `pred_id`, WP property p[15]) in row-major order.
fn collect_channel(
    get: impl Fn(usize, usize) -> i32,
    gw: usize,
    gh: usize,
    pred_id: u32,
) -> (Vec<u32>, Vec<i64>) {
    let mut wp = WpState::new(gw);
    let mut res: Vec<u32> = Vec::with_capacity(gw * gh);
    let mut prp = Vec::with_capacity(gw * gh);
    for gy in 0..gh {
        for gx in 0..gw {
            let v = get(gx, gy) as i64;
            let w_ = if gx > 0 {
                get(gx - 1, gy) as i64
            } else if gy > 0 {
                get(gx, gy - 1) as i64
            } else {
                0
            };
            let n_ = if gy > 0 { get(gx, gy - 1) as i64 } else { w_ };
            let nw_ = if gx > 0 && gy > 0 {
                get(gx - 1, gy - 1) as i64
            } else {
                w_
            };
            let ne_ = if gx + 1 < gw && gy > 0 {
                get(gx + 1, gy - 1) as i64
            } else {
                n_
            };
            let nn_ = if gy > 1 { get(gx, gy - 2) as i64 } else { n_ };
            let wp_pred = wp.predict(gx, gy, n_, w_, ne_, nw_, nn_);
            prp.push(wp.wp_prop);
            let pred = if pred_id == PREDICTOR_WEIGHTED {
                wp_pred
            } else {
                clamped_gradient(w_, n_, nw_)
            };
            res.push(pack_signed((v - pred) as i32));
            wp.update(v, gx, gy);
        }
    }
    (res, prp)
}

/// Pick the best activity threshold for a channel among candidates, returning
/// (best_t, best_bucketed_bits, flat_bits).
fn entropy_of_hist(hist: &[u64], total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c != 0 {
            let p = c as f64 / t;
            bits -= c as f64 * p.log2();
        }
    }
    bits
}

fn pick_threshold(res: &[u32], prp: &[i64]) -> (i32, f64, f64) {
    let flat = order0_entropy(res);
    let max = res.iter().copied().max().unwrap_or(0) as usize;
    // Reusable per-bucket histograms; for each candidate threshold we fill them
    // in a single pass over (res, prp) instead of materializing three bucket
    // Vecs and re-scanning each. Same frequencies, same entropy, same choice.
    let mut h0 = vec![0u64; max + 1];
    let mut h1 = vec![0u64; max + 1];
    let mut h2 = vec![0u64; max + 1];
    let mut best_t = 0i32;
    let mut best_bits = f64::INFINITY;
    for &t in &[8i64, 16, 24, 32, 48, 64, 96] {
        h0.fill(0);
        h1.fill(0);
        h2.fill(0);
        let (mut n0, mut n1, mut n2) = (0u64, 0u64, 0u64);
        for (&r, &p) in res.iter().zip(prp.iter()) {
            match bucket_of(p, t) {
                0 => {
                    h0[r as usize] += 1;
                    n0 += 1;
                }
                1 => {
                    h1[r as usize] += 1;
                    n1 += 1;
                }
                _ => {
                    h2[r as usize] += 1;
                    n2 += 1;
                }
            }
        }
        let bits = entropy_of_hist(&h0, n0) + entropy_of_hist(&h1, n1) + entropy_of_hist(&h2, n2);
        if bits < best_bits {
            best_bits = bits;
            best_t = t as i32;
        }
    }
    (best_t, best_bits, flat)
}

/// v1 context tree: single-group only. Returns true (and writes the full frame)
/// if the context tree is estimated to help; false otherwise (caller falls
/// through to the flat path, having written nothing).
fn try_encode_context_tree_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    predictors: &[u32],
    min_symbol: u32,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };

    // Collect residuals + WP property per channel (WP runs for every channel).
    let mut chan_res: Vec<Vec<u32>> = Vec::with_capacity(nb_chans);
    let mut chan_prp: Vec<Vec<i64>> = Vec::with_capacity(nb_chans);
    for chan in 0..3usize {
        let pd = linear.plane_data(chan);
        let get = |gx: usize, gy: usize| pd[gy * xsize + gx];
        let (r, p) = collect_channel(&get, xsize, ysize, predictors[chan]);
        chan_res.push(r);
        chan_prp.push(p);
    }
    if let Some(a) = alpha {
        let get = |gx: usize, gy: usize| a.get_i32(gy * xsize + gx);
        let (r, p) = collect_channel(&get, xsize, ysize, predictors[3]);
        chan_res.push(r);
        chan_prp.push(p);
    }

    // Per-channel threshold + cost comparison.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    for chan in 0..nb_chans {
        let (t, cb, fb) = pick_threshold(&chan_res[chan], &chan_prp[chan]);
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    // Guard: require the context tree to beat the flat path by more than the
    // extra-context header overhead (~64 bytes per added context, conservative).
    let overhead_bits = (2 * nb_chans) as f64 * 64.0 * 8.0;
    if ctx_bits + overhead_bits >= flat_bits {
        return false;
    }

    // Build tree + context map.
    let tree = build_context_tree(nb_chans, predictors, &ts);
    let mut tree_tokens: Vec<Token> = Vec::new();
    let ctx_map = emit_ct_tree(&tree, &mut tree_tokens);
    // Flat lookup over the dense (chan*3+bucket) property space, replacing a
    // per-pixel HashMap probe in the tokenize loop.
    let ctx_lut: Vec<u32> = (0..(nb_chans as u32) * 3).map(|k| ctx_map[&k]).collect();
    let num_pixel_ctx = nb_chans * 3;

    // Tokenize: each pixel routed to context (channel,bucket).
    let mut tokens: Vec<Token> = Vec::with_capacity(xsize * ysize * nb_chans);
    for chan in 0..nb_chans {
        let res = &chan_res[chan];
        let prp = &chan_prp[chan];
        let t = ts[chan] as i64;
        for (&prp, &res) in prp[..res.len()].iter().zip(res.iter()) {
            let bucket = bucket_of(prp, t);
            let ctx = ctx_lut[chan * 3 + bucket as usize];
            tokens.push(Token::new(ctx, res));
        }
    }

    // Frame header + single section.
    write_frame_header_modular(alpha.is_some(), writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_modular_transforms(nb_chans, &mut section);

    let distance_ctx = num_pixel_ctx as u32;
    let lz_tokens = lz77_compress(&tokens, distance_ctx);
    let code = build_lz_pixel_code(&lz_tokens, num_pixel_ctx, min_symbol);
    write_tree_lz77(&tree_tokens, &code, min_symbol, &mut section);
    for t in &lz_tokens {
        write_lz_token(*t, &code, min_symbol, &mut section);
    }
    section.zero_pad_to_byte();

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append(&section);
    writer.zero_pad_to_byte();
    true
}

/// v1 context tree, multi-group. Global context tree in LfGlobal; each AC group
/// routes its group-local pixels through it (fresh WP per group, matching the
/// decoder). Returns true (and writes the full frame) if it helps; else false.
fn try_encode_context_tree_multi_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    predictors: &[u32],
    xsize_groups: usize,
    ysize_groups: usize,
    num_dc_groups: usize,
    min_symbol: u32,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };
    let num_ac_groups = xsize_groups * ysize_groups;

    // 1) Collect (residual, WP property) per group per channel (group-local WP).
    let mut groups: Vec<Vec<(Vec<u32>, Vec<i64>)>> = Vec::with_capacity(num_ac_groups);
    for gy in 0..ysize_groups {
        for gx in 0..xsize_groups {
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let gw = GROUP_DIM.min(xsize - x0);
            let gh = GROUP_DIM.min(ysize - y0);
            let mut chans: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(nb_chans);
            for chan in 0..3usize {
                let pd = linear.plane_data(chan);
                let get = |lx: usize, ly: usize| pd[(y0 + ly) * xsize + (x0 + lx)];
                chans.push(collect_channel(&get, gw, gh, predictors[chan]));
            }
            if let Some(a) = alpha {
                let get = |lx: usize, ly: usize| a.get_i32((y0 + ly) * xsize + (x0 + lx));
                chans.push(collect_channel(&get, gw, gh, predictors[3]));
            }
            groups.push(chans);
        }
    }

    // 2) Global per-channel threshold from aggregated stats.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    for chan in 0..nb_chans {
        let mut res: Vec<u32> = Vec::new();
        let mut prp: Vec<i64> = Vec::new();
        for g in &groups {
            res.extend_from_slice(&g[chan].0);
            prp.extend_from_slice(&g[chan].1);
        }
        let (t, cb, fb) = pick_threshold(&res, &prp);
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    let overhead_bits = (2 * nb_chans) as f64 * 64.0 * 8.0;
    if ctx_bits + overhead_bits >= flat_bits {
        return false;
    }

    // 3) Global context tree + map.
    let tree = build_context_tree(nb_chans, predictors, &ts);
    let mut tree_tokens: Vec<Token> = Vec::new();
    let ctx_map = emit_ct_tree(&tree, &mut tree_tokens);
    // Flat lookup over the dense (chan*3+bucket) property space, replacing a
    // per-pixel HashMap probe in the tokenize loop.
    let ctx_lut: Vec<u32> = (0..(nb_chans as u32) * 3).map(|k| ctx_map[&k]).collect();
    let num_pixel_ctx = nb_chans * 3;
    let distance_ctx = num_pixel_ctx as u32;

    // 4) Per-group tokens (reusing collected res/prop) + per-group LZ77.
    let mut group_lz_tokens: Vec<Vec<LzToken>> = Vec::with_capacity(num_ac_groups);
    let mut all_lz: Vec<LzToken> = Vec::new();
    for g in &groups {
        let mut toks: Vec<Token> = Vec::new();
        for chan in 0..nb_chans {
            let (res, prp) = &g[chan];
            let t = ts[chan] as i64;
            for i in 0..res.len() {
                let bucket = bucket_of(prp[i], t);
                let ctx = ctx_lut[chan * 3 + bucket as usize];
                toks.push(Token::new(ctx, res[i]));
            }
        }
        let lz = lz77_compress(&toks, distance_ctx);
        all_lz.extend_from_slice(&lz);
        group_lz_tokens.push(lz);
    }
    let code = build_lz_pixel_code(&all_lz, num_pixel_ctx, min_symbol);

    // 5) Sections (same layout as the flat multi-group path).
    write_frame_header_modular(alpha.is_some(), writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default = 1
    sections[0].write(1, 1); // has_tree = 1
    write_tree_lz77(&tree_tokens, &code, min_symbol, &mut sections[0]);
    sections[0].write(1, 1); // use_global_tree
    sections[0].write(1, 1); // wp_default
    write_modular_transforms(nb_chans, &mut sections[0]);
    sections[0].zero_pad_to_byte();

    for i in 0..num_dc_groups {
        sections[1 + i].write(1, 1);
        sections[1 + i].write(1, 1);
        sections[1 + i].write(2, 0);
        sections[1 + i].zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    for group_index in 0..num_ac_groups {
        let section_idx = 2 + num_dc_groups + group_index;
        sections[section_idx].write(1, 1);
        sections[section_idx].write(1, 1);
        sections[section_idx].write(2, 0);
        for t in &group_lz_tokens[group_index] {
            write_lz_token(*t, &code, min_symbol, &mut sections[section_idx]);
        }
        sections[section_idx].zero_pad_to_byte();
    }

    writer.write(1, 0);
    writer.zero_pad_to_byte();
    for s in &sections {
        write_toc_entry(s.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    for s in &sections {
        writer.append(s);
        writer.zero_pad_to_byte();
    }
    true
}

use crate::bit_writer::BitWriter;
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    Histogram, OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code,
    write_token,
};
use crate::image::Image3Si;

/// Build a frequency histogram for each cluster from an `LzToken` stream.
fn lz_build_histograms(
    toks: &[LzToken],
    context_map: &[u8],
    num_clusters: usize,
    min_symbol: u32,
) -> Vec<Histogram> {
    let mut hs = vec![Histogram::new(); num_clusters];
    for t in toks {
        match *t {
            LzToken::Pixel { context, value } => {
                let (sym, _, _) = crate::entropy::uint_encode(value);
                let cluster = context_map[context as usize] as usize;
                hs[cluster].add(sym);
            }
            LzToken::Lz77 {
                pixel_context,
                distance_context,
                length_value,
            } => {
                let (len_tok, _, _) = lz77_length_encode(length_value);
                let pixel_cluster = context_map[pixel_context as usize] as usize;
                hs[pixel_cluster].add(min_symbol + len_tok);

                let dist_cluster = context_map[distance_context as usize] as usize;
                hs[dist_cluster].add(LZ77_DIST_VALUE);
            }
        }
    }
    hs
}

/// Build per-cluster prefix codes from an `LzToken` stream.
/// `nb_chans + 1` contexts: `nb_chans` channel leaves + 1 distance context.
fn build_lz_pixel_code(toks: &[LzToken], nb_chans: usize, min_symbol: u32) -> OwnedEntropyCode {
    use crate::entropy::build_huffman_codes;
    use crate::entropy::cluster_histograms;

    let num_contexts = nb_chans + 1;
    let context_map_initial: Vec<u8> = (0..num_contexts).map(|i| i as u8).collect();
    let mut histograms = lz_build_histograms(toks, &context_map_initial, num_contexts, min_symbol);

    let mut context_map: Vec<u8> = Vec::new();
    cluster_histograms(&mut histograms, &mut context_map);

    let mut code = OwnedEntropyCode {
        context_map,
        prefix_codes: build_huffman_codes(&histograms),
        orig_context_map: None,
        orig_num_contexts: num_contexts,
        use_prefix_code: true,
        ans_freqs: Vec::new(),
        ans_symbols: Vec::new(),
    };

    // Apply the single-symbol patch (mirrors build_pixel_code) per cluster so
    // that contexts with one unique symbol still emit a parseable code.
    for pc in &mut code.prefix_codes {
        let mut nonzero = 0;
        let mut idx = 0;
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
                pc.depths[idx] = 0;
                pc.bits[idx] = 0;
            } else {
                pc.depths[0] = 1;
                pc.bits[0] = 0;
                pc.depths[idx] = 1;
                pc.bits[idx] = 1;
            }
        }
    }
    code
}

/// Emit one `LzToken` into the bitstream.
#[inline]
fn write_lz_token(t: LzToken, code: &OwnedEntropyCode, min_symbol: u32, w: &mut BitWriter) {
    match t {
        LzToken::Pixel { context, value } => {
            let (sym, nbits, bits) = crate::entropy::uint_encode(value);
            let cluster = code.context_map[context as usize] as usize;
            let pc = &code.prefix_codes[cluster];
            let d = pc.depths[sym as usize] as usize;
            let data = (pc.bits[sym as usize] as u64) | ((bits as u64) << d);
            w.write(d + nbits as usize, data);
        }
        LzToken::Lz77 {
            pixel_context,
            distance_context,
            length_value,
        } => {
            let (len_tok, len_nbits, len_bits) = lz77_length_encode(length_value);
            let sym = min_symbol + len_tok;
            let pcluster = code.context_map[pixel_context as usize] as usize;
            let pc = &code.prefix_codes[pcluster];
            let d = pc.depths[sym as usize] as usize;
            debug_assert!(
                d > 0,
                "LZ77 length symbol {} unrepresented in histogram",
                sym
            );
            let data = (pc.bits[sym as usize] as u64) | ((len_bits as u64) << d);
            w.write(d + len_nbits as usize, data);

            // Distance symbol: value LZ77_DIST_VALUE = 0, no extra bits.
            let dcluster = code.context_map[distance_context as usize] as usize;
            let dc = &code.prefix_codes[dcluster];
            let dd = dc.depths[LZ77_DIST_VALUE as usize] as usize;
            // (Could be 0 if it's the only symbol in a single-symbol histogram.)
            if dd > 0 {
                w.write(dd, dc.bits[LZ77_DIST_VALUE as usize] as u64);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tree writing (balanced N-leaf, Gradient predictor).
//
// libjxl reads modular trees in BFS order: each node is one PROPERTY token
// (=0 for leaf, =prop+1 for split) optionally followed by a SPLIT_VAL token
// or 4 leaf-data tokens (predictor, offset, multiplier-log, multiplier-bits).
// All channels share the Gradient predictor with offset 0 and multiplier 1.
// ---------------------------------------------------------------------------

fn push_split(out: &mut Vec<Token>, property: u32, split_val: i32) {
    out.push(Token::new(TREE_CTX_PROPERTY, property + 1));
    out.push(Token::new(TREE_CTX_SPLIT_VAL, pack_signed(split_val)));
}

fn push_leaf(out: &mut Vec<Token>, predictor: u32) {
    out.push(Token::new(TREE_CTX_PROPERTY, 0));
    out.push(Token::new(TREE_CTX_PREDICTOR, predictor));
    out.push(Token::new(TREE_CTX_OFFSET, pack_signed(0)));
    out.push(Token::new(TREE_CTX_MULTIPLIER_LOG, 0));
    out.push(Token::new(TREE_CTX_MULTIPLIER_BITS, 0));
}

/// Build a balanced binary tree over `n_leaves` leaves splitting on property 0
/// (the channel index after RCT).  BFS leaf order is chan N-1, ..., chan 0.
fn build_balanced_tree_tokens(predictors: &[u32]) -> Vec<Token> {
    let n_leaves = predictors.len();
    let mut t = Vec::new();
    // Leaves are emitted in BFS order chan(N-1), ..., chan0 (matches fjxl), so
    // index them with predictors[chan].
    match n_leaves {
        1 => push_leaf(&mut t, predictors[0]),
        2 => {
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        3 => {
            push_split(&mut t, 0, 1);
            push_leaf(&mut t, predictors[2]); // chan 2
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        4 => {
            push_split(&mut t, 0, 1);
            push_split(&mut t, 0, 2);
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[3]); // chan 3
            push_leaf(&mut t, predictors[2]); // chan 2
            push_leaf(&mut t, predictors[1]); // chan 1
            push_leaf(&mut t, predictors[0]); // chan 0
        }
        _ => {
            // N>=5: right-leaning chain. BFS emission yields leaves in order
            // chan(N-1)..chan0 -> contexts 0..N-1, matching channel_to_context.
            // Peels off the highest channel at each split: split(0,k) sends
            // channel>k (== k+1) to a leaf, else continue.
            for k in (1..=(n_leaves - 2)).rev() {
                push_split(&mut t, 0, k as i32);
                push_leaf(&mut t, predictors[k + 1]);
            }
            push_split(&mut t, 0, 0);
            push_leaf(&mut t, predictors[1]);
            push_leaf(&mut t, predictors[0]);
        }
    }
    t
}

/// Write the LZ77 sub-bundle (matches `LZ77Params::VisitFields` and `DecodeUintConfig`):
///   1 bit:  enabled = 1
///   U32 min_symbol:  U32(Val(224), Val(512), Val(4096), BitsOffset(15, 8))
///                    For LZ77_MIN_SYMBOL = 64: selector 3 → "11" + 15 bits (64 - 8) = 56
///   U32 min_length:  U32(Val(3), Val(4), BitsOffset(2, 5), BitsOffset(8, 9))
///                    For LZ77_MIN_LENGTH = 3: selector 0 → "00"
///   length_uint_config: DecodeUintConfig(log_alpha_size = 8).
///                       split_exp = 4 (in CeilLog2(8+1) = 4 bits = "0010" LSB-first),
///                       msb_in_token = 0 (in CeilLog2(4+1) = 3 bits = "000"),
///                       lsb_in_token = 0 (in CeilLog2(4-0+1) = 3 bits = "000").
fn write_lz77_header(min_symbol: u32, w: &mut BitWriter) {
    w.write(1, 1); // enabled
    // min_symbol: selector 3 (Bits(15) + 8), value 64 → payload = 56
    w.write(2, 0b11);
    w.write(15, (min_symbol - 8) as u64);
    // min_length: selector 0 (Val(3))
    w.write(2, 0b00);
    // length_uint_config (split=4, msb=0, lsb=0):
    w.write(4, 4);
    w.write(3, 0);
    w.write(3, 0);
}

/// Write the local tree + LZ77-enabled pixel histograms, then return.
/// The pixel `code` must have `nb_chans + 1` contexts (last = distance).
fn write_local_tree_lz77(
    predictors: &[u32],
    pixel_code: &OwnedEntropyCode,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    let tree_tokens = build_balanced_tree_tokens(predictors);
    write_tree_lz77(&tree_tokens, pixel_code, min_symbol, w);
}

/// Write a pre-built MA tree (token stream) + the LZ77 pixel code header.
fn write_tree_lz77(
    tree_tokens: &[Token],
    pixel_code: &OwnedEntropyCode,
    min_symbol: u32,
    w: &mut BitWriter,
) {
    let tree_code = optimize_entropy_code(tree_tokens, NUM_TREE_CONTEXTS);
    let tree_code_ref = tree_code.as_ref();

    // Tree's entropy code: no LZ77 in the tree itself.
    w.write(1, 0);
    write_entropy_code(&tree_code_ref, w);
    for tok in tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }

    // Pixel entropy code: LZ77 ENABLED for the main bitstream.
    write_lz77_header(min_symbol, w);
    // The decoder appends an extra context (distance) when LZ77 is on, so the
    // context map we write must already include it as its last entry.
    write_entropy_code(&pixel_code.as_ref(), w);
}

// ---------------------------------------------------------------------------
// f32 lossless (v1): non-negative float, RGB, no alpha. The 32-bit float bits
// are reinterpreted as int32 channels (matching libjxl float_to_int for
// bits==32: a raw memcpy), then coded as a modular image with NO RCT and NO
// LZ77. LZ77 must be off because float residual tokens can reach 127, leaving
// no room below the 128-symbol alphabet for LZ77 length symbols. Per-channel
// gradient/WP prediction and the channel-split tree are reused unchanged.
// ---------------------------------------------------------------------------

/// Write a channel-split MA tree (no LZ77 in the tree) followed by the pixel
/// entropy code header with LZ77 DISABLED. Mirrors the tree-writing convention
/// but flips the pixel stream's LZ77 flag off.
fn write_tree_and_pixel_code_nolz(
    tree_tokens: &[Token],
    pixel_code: &OwnedEntropyCode,
    w: &mut BitWriter,
) {
    let tree_code = optimize_entropy_code(tree_tokens, NUM_TREE_CONTEXTS);
    let tree_code_ref = tree_code.as_ref();
    w.write(1, 0); // tree entropy code: no LZ77
    write_entropy_code(&tree_code_ref, w);
    for tok in tree_tokens {
        write_token(*tok, &tree_code_ref, w);
    }
    // Pixel entropy code: LZ77 DISABLED (1 bit = 0), then the code itself.
    w.write(1, 0);
    write_entropy_code(&pixel_code.as_ref(), w);
}

pub(crate) fn encode_frame_lossless_float(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    writer: &mut BitWriter,
) {
    let xsize = linear.xsize();
    let ysize = linear.ysize();
    let nb_chans = 3usize + if alpha.is_some() { 1 } else { 0 };

    let xsize_groups = xsize.div_ceil(GROUP_DIM);
    let ysize_groups = ysize.div_ceil(GROUP_DIM);
    let num_ac_groups = xsize_groups * ysize_groups;
    let xsize_dc_groups = xsize.div_ceil(LF_GROUP_DIM);
    let ysize_dc_groups = ysize.div_ceil(LF_GROUP_DIM);
    let num_dc_groups = xsize_dc_groups * ysize_dc_groups;
    let single_group = num_ac_groups == 1;

    // Float-bit values reach ~2^30, where the Weighted Predictor's internal
    // weight/divlookup arithmetic is not bit-verified against libjxl. Gradient
    // (clamp(W+N-NW), computed in i64) is exact at any magnitude and the
    // decoder reconstructs it identically, so the float path uses gradient on
    // every channel. (Predictor id 5 = gradient.)
    const GRADIENT_PRED: u32 = 5;
    let predictors = [GRADIENT_PRED; 4];
    let tree_tokens = build_balanced_tree_tokens(&predictors[..nb_chans]);

    write_frame_header_modular(alpha.is_some(), writer);

    if single_group {
        let mut section = BitWriter::new();
        section.write(1, 1); // dc_quant all_default = 1
        section.write(1, 0); // has_tree = 0
        section.write(1, 0); // use_global_tree = 0
        section.write(1, 1); // wp_default = 1
        section.write(2, 0b00); // 0 transforms (no RCT for float bits)

        let tokens = tokenize_all(
            linear,
            alpha,
            xsize,
            ysize,
            0,
            0,
            xsize,
            ysize,
            3,
            &predictors,
        );
        let code = optimize_entropy_code(&tokens, nb_chans);
        write_tree_and_pixel_code_nolz(&tree_tokens, &code, &mut section);
        for t in &tokens {
            write_token(*t, &code.as_ref(), &mut section);
        }
        section.zero_pad_to_byte();

        writer.write(1, 0);
        writer.zero_pad_to_byte();
        write_toc_entry(section.bits_written() / 8, writer);
        writer.zero_pad_to_byte();
        writer.append(&section);
        writer.zero_pad_to_byte();
    } else {
        let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
        let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

        // Tokenize each AC group (group-local) and pool for one global code.
        let mut group_tokens: Vec<Vec<Token>> = Vec::with_capacity(num_ac_groups);
        let mut all_tokens: Vec<Token> = Vec::new();
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);
                let toks =
                    tokenize_all(linear, alpha, xsize, ysize, x0, y0, gw, gh, 3, &predictors);
                all_tokens.extend_from_slice(&toks);
                group_tokens.push(toks);
            }
        }
        let code = optimize_entropy_code(&all_tokens, nb_chans);

        // Section 0: DC global (tree + pixel code, both no-LZ77) + GroupHeader.
        sections[0].write(1, 1); // dc_quant
        sections[0].write(1, 1); // has_tree = 1
        write_tree_and_pixel_code_nolz(&tree_tokens, &code, &mut sections[0]);
        sections[0].write(1, 1); // use_global_tree
        sections[0].write(1, 1); // wp_default
        sections[0].write(2, 0b00); // 0 transforms
        sections[0].zero_pad_to_byte();

        for i in 0..num_dc_groups {
            sections[1 + i].write(1, 1);
            sections[1 + i].write(1, 1);
            sections[1 + i].write(2, 0);
            sections[1 + i].zero_pad_to_byte();
        }

        let ac_global_idx = 1 + num_dc_groups;
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].write(1, 1);
        sections[ac_global_idx].zero_pad_to_byte();

        for group_index in 0..num_ac_groups {
            let section_idx = 2 + num_dc_groups + group_index;
            sections[section_idx].write(1, 1);
            sections[section_idx].write(1, 1);
            sections[section_idx].write(2, 0);
            for t in &group_tokens[group_index] {
                write_token(*t, &code.as_ref(), &mut sections[section_idx]);
            }
            sections[section_idx].zero_pad_to_byte();
        }

        writer.write(1, 0);
        writer.zero_pad_to_byte();
        for s in &sections {
            write_toc_entry(s.bits_written() / 8, writer);
        }
        writer.zero_pad_to_byte();
        for s in &sections {
            writer.append(s);
            writer.zero_pad_to_byte();
        }
    }
}
