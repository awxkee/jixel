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
const PREDICTOR_LEFT: u32 = 1;
const PREDICTOR_TOP: u32 = 2;
const PREDICTOR_SELECT: u32 = 4;
const PREDICTOR_GRADIENT: u32 = 5;
const PREDICTOR_WEIGHTED: u32 = 6;
const PREDICTOR_AVERAGE4: u32 = 13;
static SLOW_PREDICTORS: [u32; 6] = [
    PREDICTOR_WEIGHTED,
    PREDICTOR_GRADIENT,
    PREDICTOR_AVERAGE4,
    PREDICTOR_SELECT,
    PREDICTOR_LEFT,
    PREDICTOR_TOP,
];

#[derive(Clone, Copy)]
struct PredictorNeighbors {
    left: i64,
    top: i64,
    top_left: i64,
    top_right: i64,
    left_left: i64,
    top_top: i64,
    top_right_right: i64,
}

#[inline]
fn predictor_neighbors<F: Fn(usize, usize) -> i32 + ?Sized>(
    get: &F,
    x: usize,
    y: usize,
    width: usize,
) -> PredictorNeighbors {
    let left = if x > 0 {
        get(x - 1, y) as i64
    } else if y > 0 {
        get(x, y - 1) as i64
    } else {
        0
    };
    let top = if y > 0 { get(x, y - 1) as i64 } else { left };
    let top_left = if x > 0 && y > 0 {
        get(x - 1, y - 1) as i64
    } else {
        left
    };
    let top_right = if x + 1 < width && y > 0 {
        get(x + 1, y - 1) as i64
    } else {
        top
    };
    PredictorNeighbors {
        left,
        top,
        top_left,
        top_right,
        left_left: if x > 1 { get(x - 2, y) as i64 } else { left },
        top_top: if y > 1 { get(x, y - 2) as i64 } else { top },
        top_right_right: if x + 2 < width && y > 0 {
            get(x + 2, y - 1) as i64
        } else {
            top_right
        },
    }
}

#[inline]
fn predictor_value(pred_id: u32, n: PredictorNeighbors, weighted: i64) -> i64 {
    match pred_id {
        PREDICTOR_LEFT => n.left,
        PREDICTOR_TOP => n.top,
        PREDICTOR_SELECT => {
            let projected = n.left + n.top - n.top_left;
            if (projected - n.left).abs() < (projected - n.top).abs() {
                n.left
            } else {
                n.top
            }
        }
        PREDICTOR_GRADIENT => clamped_gradient(n.left, n.top, n.top_left),
        PREDICTOR_WEIGHTED => weighted,
        PREDICTOR_AVERAGE4 => {
            (6 * n.top - 2 * n.top_top
                + 7 * n.left
                + n.left_left
                + n.top_right_right
                + 3 * n.top_right
                + 8)
                / 16
        }
        _ => unreachable!("unsupported modular predictor {pred_id}"),
    }
}

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
static WP_DIV: [u32; 64] = [
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201, 1398101, 1290555, 1198372, 1118481, 1048576, 986895, 932067, 883011, 838860, 798915,
    762600, 729444, 699050, 671088, 645277, 621378, 599186, 578524, 559240, 541200, 524288, 508400,
    493447, 479349, 466033, 453438, 441505, 430185, 419430, 409200, 399457, 390167, 381300, 372827,
    364722, 356962, 349525, 342392, 335544, 328965, 322638, 316551, 310689, 305040, 299593, 294337,
    289262, 284359, 279620, 275036, 270600, 266305, 262144,
];

pub(crate) struct WpState {
    xsize: usize,
    pred_errors: [Vec<u32>; 4],
    error: Vec<i64>,
    prediction: [i64; 4],
    pred: i64,
    /// libjxl property kWPProp (p[15]): the signed neighbour WP-error with the
    /// largest absolute value among {W, N, NW, NE}. Set on each `predict`.
    pub(crate) wp_prop: i64,
}

impl WpState {
    pub(crate) fn new(xsize: usize) -> Self {
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
    pub(crate) fn predict(
        &mut self,
        x: usize,
        y: usize,
        n: i64,
        w: i64,
        ne: i64,
        nw: i64,
        nn: i64,
    ) -> i64 {
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
    pub(crate) fn update(&mut self, val: i64, x: usize, y: usize) {
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
const LZ77_NUM_SPECIAL_DISTANCES: u32 = 120;

pub(crate) fn encode_frame_lossless(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    max_bits: u32,
    progressive: bool,
    num_color: usize,
    speed: crate::Speed,
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let num_threads = num_threads.max(1);
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
    let grad_pack_fn = selected_grad_pack_interior_fn();

    // Palette (v2): single-group, RGB/RGBA, <=256 distinct colors. Encodes
    // the image as a small palette meta-channel + an index channel, which is a
    // large win for low-color/graphic content. Falls through to the normal
    // RCT+predictor path when it doesn't apply.
    if single_group
        && num_color == 3
        && try_encode_palette_single_group(
            linear,
            alpha,
            xsize,
            ysize,
            min_symbol,
            grad_pack_fn,
            speed,
            writer,
        )
    {
        return;
    }

    // Large graphics often exceed 256 colors globally while each 256x256 group
    // remains low-color. Encode qualifying RGB/RGBA groups through an exact
    // local Palette transform and leave high-color groups on the normal path.
    if !single_group
        && num_color == 3
        && try_encode_local_palette_multi_group(
            linear,
            alpha,
            xsize,
            ysize,
            xsize_groups,
            ysize_groups,
            num_dc_groups,
            min_symbol,
            grad_pack_fn,
            speed,
            num_threads,
            writer,
        )
    {
        return;
    }

    // Stage-2 progressive lossless (opt-in): Squeeze pyramid (RGB + optional alpha).
    if single_group && num_color == 3 && progressive {
        encode_squeeze_single_group(
            linear,
            alpha,
            xsize,
            ysize,
            min_symbol,
            grad_pack_fn,
            speed,
            num_threads,
            writer,
        );
        return;
    }

    // Progressive multi-group (Stage A: all squeezed channels fit the global
    // stream). Falls through to the non-progressive path if a channel is still
    // larger than a group (Stage B).
    if !single_group
        && num_color == 3
        && progressive
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
            grad_pack_fn,
            speed,
            num_threads,
            writer,
        )
    {
        return;
    }

    // Fast uses fixed Weighted prediction and skips all adaptive analysis. Slow
    // searches the supported Slow predictor set per channel by estimated cost
    // (one global choice, used for the tree and every group).
    let adaptive_search = speed == crate::Speed::Slow;
    let predictors = if adaptive_search {
        choose_predictors(linear, alpha, xsize, ysize, num_threads)
    } else {
        [PREDICTOR_WEIGHTED; 4]
    };
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
    if adaptive_search && num_color == 3 {
        if single_group {
            if try_encode_context_tree_single_group(
                linear,
                alpha,
                xsize,
                ysize,
                &predictors,
                min_symbol,
                num_threads,
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
            num_threads,
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
            grad_pack_fn,
            num_threads,
        );

        // LZ77 layer: collapse runs of identical tokens into back-references.
        // The distance context is the (nb_chans)-th context, appended after the
        // per-channel ones.
        let distance_ctx = nb_chans as u32;
        let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed);

        // Per-cluster prefix codes (nb_chans + 1 contexts), balanced N-leaf tree.
        let code = build_lz_pixel_code(
            &lz_tokens,
            nb_chans,
            min_symbol,
            speed == crate::Speed::Slow,
        );
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
        let group_lz_tokens: Vec<Vec<LzToken>> =
            crate::thread_pool::steal_map(num_ac_groups, num_threads, |group_index| {
                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
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
                    grad_pack_fn,
                    1,
                );
                lz77_compress_for_speed(&toks, distance_ctx, speed)
            });
        let mut all_lz: Vec<LzToken> = Vec::new();
        for lz in &group_lz_tokens {
            all_lz.extend_from_slice(lz);
        }
        let code = build_lz_pixel_code(&all_lz, nb_chans, min_symbol, speed == crate::Speed::Slow);

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

/// Progressive-lossless single-group path. Applies the alternating Squeeze
/// pyramid to the RCT'd channels, then tokenizes the resulting channels with
/// fixed Weighted prediction in Fast mode or adaptive prediction in Slow mode.
/// The decoder reconstructs the input exactly through inverse-Squeeze + inverse-RCT.
fn encode_squeeze_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    num_threads: usize,
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
        match a {
            AlphaPlane::U8(data) => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    for (dst, &src) in row[..xsize].iter_mut().zip(src_row.iter()) {
                        *dst = src as i32;
                    }
                }
            }
            AlphaPlane::U16 { data, bits: _ } => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    for (dst, &src) in row[..xsize].iter_mut().zip(src_row.iter()) {
                        *dst = src as i32;
                    }
                }
            }
            AlphaPlane::F32(data) => {
                for (row, src_row) in ch
                    .data
                    .chunks_exact_mut(xsize)
                    .zip(data.chunks_exact(xsize))
                {
                    for (dst, &src) in row[..xsize].iter_mut().zip(src_row.iter()) {
                        *dst = src;
                    }
                }
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

    // Slow searches the useful general and directional predictors per channel;
    // Fast avoids the analysis pass and uses fixed Weighted prediction.
    let predictors: Vec<u32> = if speed == crate::Speed::Slow {
        crate::thread_pool::steal_map(channels.len(), num_threads, |c| {
            let ch = &channels[c];
            let data = &ch.data;
            let w = ch.w;
            let get = move |gx: usize, gy: usize| data[gy * w + gx];
            choose_predictor_for_plane(get, ch.w, ch.h)
        })
    } else {
        vec![PREDICTOR_WEIGHTED; nb]
    };
    let channel_tokens = crate::thread_pool::steal_map(nb, num_threads, |c| {
        let ch = &channels[c];
        let mut tokens = Vec::with_capacity(ch.w * ch.h);
        tokenize_plane(
            channel_to_context(c, nb),
            |x, y| ch.data[y * ch.w + x],
            ch.w,
            ch.h,
            predictors[c],
            grad_pack_fn,
            &mut tokens,
        );
        tokens
    });
    let mut tokens: Vec<Token> = Vec::new();
    for channel in channel_tokens {
        tokens.extend(channel);
    }

    write_frame_header_modular(alpha.is_some(), writer);
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0 (local tree in GroupHeader)
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_modular_transforms_rct_squeeze(&steps, &mut section);

    let distance_ctx = nb as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed);
    let code = build_lz_pixel_code(&lz_tokens, nb, min_symbol, speed == crate::Speed::Slow);
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
#[derive(Default)]
struct SqueezePredictorCost {
    costs: PredictorCosts,
}

impl SqueezePredictorCost {
    /// Add one independently predicted modular crop. Weighted prediction state
    /// resets here because the decoder resets it for every modular sub-image.
    fn add_crop(&mut self, get: impl Fn(usize, usize) -> i32, w: usize, h: usize) {
        if w == 0 || h == 0 {
            return;
        }
        let mut wp = WpState::new(w);
        for y in 0..h {
            for x in 0..w {
                let value = get(x, y) as i64;
                let neighbors = predictor_neighbors(&get, x, y, w);
                let weighted = wp.predict(
                    x,
                    y,
                    neighbors.top,
                    neighbors.left,
                    neighbors.top_right,
                    neighbors.top_left,
                    neighbors.top_top,
                );
                self.costs.add(value, neighbors, weighted);
                wp.update(value, x, y);
            }
        }
    }

    fn predictor(&self) -> u32 {
        self.costs.best_predictor()
    }
}

#[allow(clippy::too_many_arguments)]
fn for_each_squeeze_group_crop(
    channels: &[crate::squeeze::Channel],
    split: usize,
    gdim: usize,
    gx: usize,
    gy: usize,
    minsh: i32,
    maxsh: i32,
    mut visit: impl FnMut(usize, usize, usize, usize, usize, usize),
) {
    let mut within = 0usize;
    for c in split..channels.len() {
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
        visit(within, c, rx0, ry0, rw, rh);
        within += 1;
    }
}

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
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    num_threads: usize,
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

    let predictors: Vec<u32> = if speed == crate::Speed::Slow {
        // The MA tree sees a channel's position inside each modular sub-image,
        // not its index in `channels`. Pool costs by that decoder-visible slot.
        let mut costs: Vec<SqueezePredictorCost> =
            (0..nb).map(|_| SqueezePredictorCost::default()).collect();
        for c in 0..split {
            let ch = &channels[c];
            costs[c].add_crop(|x, y| ch.data[y * ch.w + x], ch.w, ch.h);
        }
        for gy in 0..ysize_dc_groups {
            for gx in 0..xsize_dc_groups {
                for_each_squeeze_group_crop(
                    &channels,
                    split,
                    LF_GROUP_DIM,
                    gx,
                    gy,
                    3,
                    1000,
                    |within, c, x0, y0, w, h| {
                        let ch = &channels[c];
                        costs[within].add_crop(|x, y| ch.data[(y0 + y) * ch.w + x0 + x], w, h);
                    },
                );
            }
        }
        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                for_each_squeeze_group_crop(
                    &channels,
                    split,
                    GROUP_DIM,
                    gx,
                    gy,
                    0,
                    2,
                    |within, c, x0, y0, w, h| {
                        let ch = &channels[c];
                        costs[within].add_crop(|x, y| ch.data[(y0 + y) * ch.w + x0 + x], w, h);
                    },
                );
            }
        }
        costs.iter().map(SqueezePredictorCost::predictor).collect()
    } else {
        vec![PREDICTOR_WEIGHTED; nb]
    };

    let distance_ctx = nb as u32;

    // Global stream: the small channels [0, split), whole.
    let global_channel_tokens = crate::thread_pool::steal_map(split, num_threads, |c| {
        let ch = &channels[c];
        let ctx = channel_to_context(c, nb);
        let data = &ch.data;
        let w = ch.w;
        let get = move |gx: usize, gy: usize| data[gy * w + gx];
        let mut tokens = Vec::with_capacity(ch.w * ch.h);
        tokenize_plane(
            ctx,
            get,
            ch.w,
            ch.h,
            predictors[c],
            grad_pack_fn,
            &mut tokens,
        );
        tokens
    });
    let mut global_tokens: Vec<Token> = Vec::new();
    for channel in global_channel_tokens {
        global_tokens.extend(channel);
    }
    let global_lz = lz77_compress_for_speed(&global_tokens, distance_ctx, speed);
    let mut all_lz: Vec<LzToken> = global_lz.clone();

    // One group's worth of cropped large-channel tokens. `gdim` is the group's
    // frame-space size (GROUP_DIM for AC, LF_GROUP_DIM for DC); a channel is
    // included when min(hshift,vshift) is inside [minsh,maxsh], and contributes
    // its `(group_rect >> shift)` crop. The decoder rebuilds the same scan, so
    // the within-group index (sequential over non-empty crops) is exactly the
    // `chan` property the global tree keys on.
    let crop_group = |gdim: usize, gx: usize, gy: usize, minsh: i32, maxsh: i32| -> Vec<LzToken> {
        let mut gtok: Vec<Token> = Vec::new();
        for_each_squeeze_group_crop(
            &channels,
            split,
            gdim,
            gx,
            gy,
            minsh,
            maxsh,
            |within, c, rx0, ry0, rw, rh| {
                let ch = &channels[c];
                let ctx = channel_to_context(within, nb);
                let pred = predictors[within];
                let data = &ch.data;
                let w = ch.w;
                let get = move |lx: usize, ly: usize| data[(ry0 + ly) * w + (rx0 + lx)];
                tokenize_plane(ctx, get, rw, rh, pred, grad_pack_fn, &mut gtok);
            },
        );
        lz77_compress_for_speed(&gtok, distance_ctx, speed)
    };

    // DC (LF) groups carry the deeply-squeezed large channels (min shift >= 3),
    // partitioned into LF_GROUP_DIM rects. Empty unless the image is large enough
    // that a >=3x-squeezed channel still exceeds a group (dimension > ~2048).
    let dc_group_lz = crate::thread_pool::steal_map(num_dc_groups, num_threads, |group_index| {
        let gx = group_index % xsize_dc_groups;
        let gy = group_index / xsize_dc_groups;
        crop_group(LF_GROUP_DIM, gx, gy, 3, 1000)
    });
    for glz in &dc_group_lz {
        all_lz.extend_from_slice(glz);
    }

    // AC groups carry the shallow large channels (min shift <= 2), in GROUP_DIM rects.
    let ac_group_lz = crate::thread_pool::steal_map(num_ac_groups, num_threads, |group_index| {
        let gx = group_index % xsize_groups;
        let gy = group_index / xsize_groups;
        crop_group(GROUP_DIM, gx, gy, 0, 2)
    });
    for glz in &ac_group_lz {
        all_lz.extend_from_slice(glz);
    }

    let code = build_lz_pixel_code(&all_lz, nb, min_symbol, speed == crate::Speed::Slow);

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
    debug_assert!(matches!(num_c, 3 | 4));
    // transforms count U32(Val(0),Val(1),...): selector 1 = Val(1) = 1 transform.
    w.write(2, 0b01);
    // id U32(Val(RCT=0),Val(Palette=1),...): selector 1 = Palette.
    w.write(2, 0b01);
    // begin_c U32(Bits(3),...): selector 0 = Bits(3), value 0.
    w.write(2, 0b00);
    w.write(3, 0);
    // num_c U32(Val(1),Val(3),Val(4),BitsOffset(13,1)).
    w.write(2, if num_c == 3 { 0b01 } else { 0b10 });
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

/// Single-group RGB/RGBA exact palette path. Encodes a palette meta-channel plus
/// an index channel, declaring a Palette transform so `InvPalette` reconstructs
/// the original channels directly (no RCT).
fn try_encode_palette_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    writer: &mut BitWriter,
) -> bool {
    use std::collections::HashMap;
    let npx = xsize * ysize;
    let num_c = 3 + usize::from(alpha.is_some());

    // 1) Reconstruct RGB from YCoCg, collect distinct tuples (bail past 256).
    let plane0 = linear.plane_data(0);
    let plane1 = linear.plane_data(1);
    let plane2 = linear.plane_data(2);
    let color_at = |gx: usize, gy: usize| {
        let (r, g, b) = inverse_ycocg(
            plane0[gy * xsize + gx],
            plane1[gy * xsize + gx],
            plane2[gy * xsize + gx],
        );
        [r, g, b, alpha.map_or(0, |a| a.get_i32(gy * xsize + gx))]
    };
    let mut seen: HashMap<[i32; 4], ()> = HashMap::with_capacity(257);
    for gy in 0..ysize {
        for gx in 0..xsize {
            seen.entry(color_at(gx, gy)).or_insert(());
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
    let mut colors: Vec<[i32; 4]> = seen.keys().copied().collect();
    colors.sort_unstable();
    let mut idx_of: HashMap<[i32; 4], u32> = HashMap::with_capacity(nb_colors);
    for (i, c) in colors.iter().enumerate() {
        idx_of.insert(*c, i as u32);
    }

    // 3) Palette meta-channel (row c = component c of each color) + index channel.
    let mut palette_ch = vec![0i32; num_c * nb_colors];
    for (i, color) in colors.iter().enumerate() {
        for c in 0..num_c {
            palette_ch[c * nb_colors + i] = color[c];
        }
    }
    let mut index_img = Vec::with_capacity(npx);
    for gy in 0..ysize {
        for gx in 0..xsize {
            index_img.push(idx_of[&color_at(gx, gy)] as i32);
        }
    }

    // 4) Channel accessors (channel 0 = palette, channel 1 = index).
    let pget = |gx: usize, gy: usize| palette_ch[gy * nb_colors + gx];
    let iget = |gx: usize, gy: usize| index_img[gy * xsize + gx];

    // 5) Slow searches predictors per channel; Fast stays fixed Weighted.
    let preds = if speed == crate::Speed::Slow {
        [
            choose_predictor_for_plane(pget, nb_colors, num_c),
            choose_predictor_for_plane(iget, xsize, ysize),
        ]
    } else {
        [PREDICTOR_WEIGHTED; 2]
    };

    // 6) Frame header + single section (mirrors the RGB single-group layout).
    write_frame_header_modular(alpha.is_some(), writer);

    let nb_chans = 2usize;
    let mut section = BitWriter::new();
    section.write(1, 1); // dc_quant all_default = 1
    section.write(1, 0); // has_tree = 0
    section.write(1, 0); // use_global_tree = 0
    section.write(1, 1); // wp_default = 1
    write_palette_transform(num_c as u32, nb_colors as u32, &mut section);

    let mut tokens: Vec<Token> = Vec::with_capacity(num_c * nb_colors + npx);
    tokenize_plane(
        channel_to_context(0, nb_chans),
        pget,
        nb_colors,
        num_c,
        preds[0],
        grad_pack_fn,
        &mut tokens,
    );
    tokenize_plane(
        channel_to_context(1, nb_chans),
        iget,
        xsize,
        ysize,
        preds[1],
        grad_pack_fn,
        &mut tokens,
    );

    let distance_ctx = nb_chans as u32;
    let lz_tokens = lz77_compress_for_speed(&tokens, distance_ctx, speed);
    let code = build_lz_pixel_code(
        &lz_tokens,
        nb_chans,
        min_symbol,
        speed == crate::Speed::Slow,
    );
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

struct LocalPaletteGroup {
    palette: Vec<i32>,
    indices: Vec<i32>,
    nb_colors: usize,
    w: usize,
    h: usize,
}

fn estimated_local_stream_bits(
    tokens: &[Token],
    predictors: &[u32],
    num_contexts: usize,
    min_symbol: u32,
    speed: crate::Speed,
) -> usize {
    let lz = lz77_compress_runs(tokens, num_contexts as u32);
    let code = build_lz_pixel_code(&lz, num_contexts, min_symbol, speed == crate::Speed::Slow);
    let mut writer = BitWriter::new();
    write_local_tree_lz77(predictors, &code, min_symbol, &mut writer);
    for token in &lz {
        write_lz_token(*token, &code, min_symbol, &mut writer);
    }
    writer.bits_written()
}

#[allow(clippy::too_many_arguments)]
fn local_palette_is_better(
    palette: &LocalPaletteGroup,
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
) -> bool {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let palette_predictors = if speed == crate::Speed::Slow {
        [
            choose_predictor_for_plane(
                |x, y| palette.palette[y * palette.nb_colors + x],
                palette.nb_colors,
                nb_chans,
            ),
            choose_predictor_for_plane(
                |x, y| palette.indices[y * palette.w + x],
                palette.w,
                palette.h,
            ),
        ]
    } else {
        [PREDICTOR_WEIGHTED; 2]
    };
    let mut palette_tokens =
        Vec::with_capacity(nb_chans * palette.nb_colors + palette.w * palette.h);
    tokenize_plane(
        channel_to_context(0, 2),
        |x, y| palette.palette[y * palette.nb_colors + x],
        palette.nb_colors,
        nb_chans,
        palette_predictors[0],
        grad_pack_fn,
        &mut palette_tokens,
    );
    tokenize_plane(
        channel_to_context(1, 2),
        |x, y| palette.indices[y * palette.w + x],
        palette.w,
        palette.h,
        palette_predictors[1],
        grad_pack_fn,
        &mut palette_tokens,
    );
    let mut palette_transform = BitWriter::new();
    write_palette_transform(
        nb_chans as u32,
        palette.nb_colors as u32,
        &mut palette_transform,
    );
    let palette_bits =
        estimated_local_stream_bits(&palette_tokens, &palette_predictors, 2, min_symbol, speed)
            + palette_transform.bits_written();

    let plain_predictors: Vec<u32> = if speed == crate::Speed::Slow {
        (0..nb_chans)
            .map(|chan| {
                if chan < 3 {
                    let plane = linear.plane_data(chan);
                    choose_predictor_for_plane(
                        |x, y| plane[(y0 + y) * xsize + x0 + x],
                        palette.w,
                        palette.h,
                    )
                } else {
                    let alpha = alpha.expect("alpha channel must exist");
                    choose_predictor_for_plane(
                        |x, y| alpha.get_i32((y0 + y) * xsize + x0 + x),
                        palette.w,
                        palette.h,
                    )
                }
            })
            .collect()
    } else {
        vec![PREDICTOR_WEIGHTED; nb_chans]
    };
    let plain_tokens = tokenize_all(
        linear,
        alpha,
        xsize,
        linear.ysize(),
        x0,
        y0,
        palette.w,
        palette.h,
        3,
        &plain_predictors,
        grad_pack_fn,
        1,
    );
    let plain_bits = estimated_local_stream_bits(
        &plain_tokens,
        &plain_predictors,
        nb_chans,
        min_symbol,
        speed,
    ) + 2; // zero-transform count

    palette_bits < plain_bits
}

fn build_local_palette_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Option<LocalPaletteGroup> {
    use std::collections::HashMap;

    let num_c = 3 + usize::from(alpha.is_some());
    let num_pixels = w * h;
    let plane0 = linear.plane_data(0);
    let plane1 = linear.plane_data(1);
    let plane2 = linear.plane_data(2);
    let tuple_at = |gx: usize, gy: usize| {
        [
            plane0[gy * xsize + gx],
            plane1[gy * xsize + gx],
            plane2[gy * xsize + gx],
            alpha.map_or(0, |a| a.get_i32(gy * xsize + gx)),
        ]
    };

    // Most photographic groups exceed the palette limit almost immediately.
    // Probe only the distinct set first, avoiding a full-group tuple allocation
    // on that overwhelmingly common rejection path.
    let mut seen = HashMap::<[i32; 4], ()>::with_capacity(257);
    for y in 0..h {
        for x in 0..w {
            let gx = x0 + x;
            let gy = y0 + y;
            seen.entry(tuple_at(gx, gy)).or_insert(());
            if seen.len() > 256 {
                return None;
            }
        }
    }

    let nb_colors = seen.len();
    // Palette coding replaces `num_c * num_pixels` samples with one index per
    // pixel plus `num_c * nb_colors` palette samples. Leave a small margin for
    // the transform header and altered entropy statistics.
    let palette_samples = num_pixels + num_c * nb_colors + 16;
    if nb_colors == 0 || palette_samples >= num_c * num_pixels {
        return None;
    }

    let mut colors: Vec<[i32; 4]> = seen.keys().copied().collect();
    colors.sort_unstable();
    let mut index_of = HashMap::<[i32; 4], i32>::with_capacity(nb_colors);
    let mut palette = vec![0i32; num_c * nb_colors];
    for (index, color) in colors.iter().enumerate() {
        index_of.insert(*color, index as i32);
        for c in 0..num_c {
            palette[c * nb_colors + index] = color[c];
        }
    }
    let mut indices = Vec::with_capacity(num_pixels);
    for y in 0..h {
        for x in 0..w {
            indices.push(index_of[&tuple_at(x0 + x, y0 + y)]);
        }
    }

    Some(LocalPaletteGroup {
        palette,
        indices,
        nb_colors,
        w,
        h,
    })
}

#[allow(clippy::too_many_arguments)]
fn try_encode_local_palette_multi_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    xsize_groups: usize,
    ysize_groups: usize,
    num_dc_groups: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    num_threads: usize,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let num_ac_groups = xsize_groups * ysize_groups;
    let palettes = crate::thread_pool::steal_map(num_ac_groups, num_threads, |group_index| {
        let gx = group_index % xsize_groups;
        let gy = group_index / xsize_groups;
        let x0 = gx * GROUP_DIM;
        let y0 = gy * GROUP_DIM;
        let w = GROUP_DIM.min(xsize - x0);
        let h = GROUP_DIM.min(ysize - y0);
        let palette = build_local_palette_group(linear, alpha, xsize, x0, y0, w, h)?;
        local_palette_is_better(
            &palette,
            linear,
            alpha,
            xsize,
            x0,
            y0,
            min_symbol,
            grad_pack_fn,
            speed,
        )
        .then_some(palette)
    });
    if !palettes.iter().any(Option::is_some) {
        return false;
    }

    // The global MA tree sees group-local channel slots. Pool predictor costs
    // for palette/index channels and ordinary YCoCg(A) channels by those slots.
    let predictors: Vec<u32> = if speed == crate::Speed::Slow {
        crate::thread_pool::steal_map(nb_chans, num_threads, |slot| {
            let mut cost = SqueezePredictorCost::default();
            for (group_index, palette) in palettes.iter().enumerate() {
                if let Some(palette) = palette {
                    if slot == 0 {
                        cost.add_crop(
                            |x, y| palette.palette[y * palette.nb_colors + x],
                            palette.nb_colors,
                            nb_chans,
                        );
                    } else if slot == 1 {
                        cost.add_crop(
                            |x, y| palette.indices[y * palette.w + x],
                            palette.w,
                            palette.h,
                        );
                    }
                    continue;
                }

                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let w = GROUP_DIM.min(xsize - x0);
                let h = GROUP_DIM.min(ysize - y0);
                if slot < 3 {
                    let plane = linear.plane_data(slot);
                    cost.add_crop(|x, y| plane[(y0 + y) * xsize + x0 + x], w, h);
                } else {
                    let alpha = alpha.expect("alpha slot requires alpha channel");
                    cost.add_crop(|x, y| alpha.get_i32((y0 + y) * xsize + x0 + x), w, h);
                }
            }
            cost.predictor()
        })
    } else {
        vec![PREDICTOR_WEIGHTED; nb_chans]
    };

    let distance_ctx = nb_chans as u32;
    let group_lz_tokens: Vec<Vec<LzToken>> =
        crate::thread_pool::steal_map(num_ac_groups, num_threads, |group_index| {
            let mut tokens = if let Some(palette) = &palettes[group_index] {
                Vec::with_capacity(nb_chans * palette.nb_colors + palette.w * palette.h)
            } else {
                Vec::new()
            };
            if let Some(palette) = &palettes[group_index] {
                tokenize_plane(
                    channel_to_context(0, nb_chans),
                    |x, y| palette.palette[y * palette.nb_colors + x],
                    palette.nb_colors,
                    nb_chans,
                    predictors[0],
                    grad_pack_fn,
                    &mut tokens,
                );
                tokenize_plane(
                    channel_to_context(1, nb_chans),
                    |x, y| palette.indices[y * palette.w + x],
                    palette.w,
                    palette.h,
                    predictors[1],
                    grad_pack_fn,
                    &mut tokens,
                );
            } else {
                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let w = GROUP_DIM.min(xsize - x0);
                let h = GROUP_DIM.min(ysize - y0);
                tokens = tokenize_all(
                    linear,
                    alpha,
                    xsize,
                    ysize,
                    x0,
                    y0,
                    w,
                    h,
                    3,
                    &predictors,
                    grad_pack_fn,
                    1,
                );
            }
            lz77_compress_for_speed(&tokens, distance_ctx, speed)
        });

    let mut all_lz = Vec::with_capacity(group_lz_tokens.iter().map(Vec::len).sum());
    for tokens in &group_lz_tokens {
        all_lz.extend_from_slice(tokens);
    }
    let code = build_lz_pixel_code(&all_lz, nb_chans, min_symbol, speed == crate::Speed::Slow);

    write_frame_header_modular(alpha.is_some(), writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default
    sections[0].write(1, 1); // has global tree
    write_local_tree_lz77(&predictors, &code, min_symbol, &mut sections[0]);
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
        sections[section_idx].write(1, 1); // use_global_tree
        sections[section_idx].write(1, 1); // wp_default
        if let Some(palette) = &palettes[group_index] {
            write_palette_transform(
                nb_chans as u32,
                palette.nb_colors as u32,
                &mut sections[section_idx],
            );
        } else {
            sections[section_idx].write(2, 0); // no local transforms
        }
        for token in &group_lz_tokens[group_index] {
            write_lz_token(*token, &code, min_symbol, &mut sections[section_idx]);
        }
        sections[section_idx].zero_pad_to_byte();
    }

    writer.write(1, 0); // no TOC permutation
    writer.zero_pad_to_byte();
    for section in &sections {
        write_toc_entry(section.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    for section in &sections {
        writer.append(section);
        writer.zero_pad_to_byte();
    }
    true
}

fn write_toc_entry(byte_len: usize, w: &mut BitWriter) {
    static OFFSETS: [usize; 4] = [0, 1024, 17_408, 4_211_712];
    static BITS: [usize; 4] = [10, 14, 22, 30];
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
        distance_value: u32,
    },
}

#[inline]
fn lz_fingerprint(tokens: &[Token], pos: usize) -> u32 {
    let mut h = 0x9e37_79b9u32;
    for token in &tokens[pos..tokens.len().min(pos + 3)] {
        h ^= token.value.wrapping_mul(0x85eb_ca6b).rotate_left(13);
        h = h.wrapping_mul(0xc2b2_ae35) ^ token.context;
    }
    h
}

#[inline]
fn lz_hash(tokens: &[Token], pos: usize) -> usize {
    const HASH_BITS: usize = 18;
    (lz_fingerprint(tokens, pos) as usize) & ((1 << HASH_BITS) - 1)
}

fn lz_has_repetition(tokens: &[Token]) -> bool {
    const MAX_SAMPLES: usize = 8_192;
    const SAMPLE_TABLE_SIZE: usize = 1 << 14;
    if tokens.len() < 256 {
        return true;
    }
    let stride = tokens.len().div_ceil(MAX_SAMPLES).max(1);
    let mut table = vec![0u32; SAMPLE_TABLE_SIZE];
    let mut samples = 0usize;
    let mut repeats = 0usize;
    for pos in (0..tokens.len().saturating_sub(2)).step_by(stride) {
        let fingerprint = lz_fingerprint(tokens, pos) | 1;
        let slot = fingerprint as usize & (SAMPLE_TABLE_SIZE - 1);
        repeats += usize::from(table[slot] == fingerprint);
        table[slot] = fingerprint;
        samples += 1;
    }
    // Only enable the expensive chain search for strongly repetitive streams.
    // Sparse matches are better handled by the allocation-free run coder.
    repeats * 5 >= samples
}

fn lz_match_len(tokens: &[Token], a: usize, b: usize) -> usize {
    let mut len = 0usize;
    const MAX_MATCH: usize = 1 << 20;
    let max_len = (tokens.len() - b).min(MAX_MATCH);
    while len < max_len {
        let x = tokens[a + len];
        let y = tokens[b + len];
        if x.context != y.context || x.value != y.value {
            break;
        }
        len += 1;
        // For overlap, compare against the already-known periodic source.
        if a + len == b {
            let period = b - a;
            while len < max_len {
                let x = tokens[b + len - period];
                let y = tokens[b + len];
                if x.context != y.context || x.value != y.value {
                    break;
                }
                len += 1;
            }
            break;
        }
    }
    len
}

fn lz_find_match(
    tokens: &[Token],
    pos: usize,
    head: &[u32],
    prev: &[u32],
    max_probes: usize,
) -> (usize, usize) {
    if pos + LZ77_MIN_LENGTH as usize > tokens.len() {
        return (0, 0);
    }
    let mut candidate = head[lz_hash(tokens, pos)];
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    let mut probes = 0;
    while candidate != u32::MAX && probes < max_probes {
        let candidate_pos = candidate as usize;
        let distance = pos - candidate_pos;
        if distance <= u32::MAX as usize {
            let len = lz_match_len(tokens, candidate_pos, pos);
            if len > best_len || (len == best_len && distance < best_dist) {
                best_len = len;
                best_dist = distance;
            }
        }
        candidate = prev[candidate_pos];
        probes += 1;
    }
    (best_len, best_dist)
}

/// Compact a sequence of `Token`s with a bounded hash-chain LZ77 search.
#[cfg(test)]
fn lz77_compress(tokens: &[Token], distance_context: u32) -> Vec<LzToken> {
    lz77_compress_with_depth(tokens, distance_context, 8)
}

#[inline]
fn lz77_compress_for_speed(
    tokens: &[Token],
    distance_context: u32,
    speed: crate::Speed,
) -> Vec<LzToken> {
    if speed == crate::Speed::Fast || !lz_has_repetition(tokens) {
        return lz77_compress_runs(tokens, distance_context);
    }
    let deep = lz77_compress_with_depth(tokens, distance_context, 8);
    let run_token_count = lz77_run_token_count(tokens);
    if deep.len() * 100 > run_token_count * 90 {
        return lz77_compress_runs(tokens, distance_context);
    }
    let runs = lz77_compress_runs(tokens, distance_context);
    if estimate_lz_payload_bits(&deep, distance_context) * 100
        <= estimate_lz_payload_bits(&runs, distance_context) * 90
    {
        deep
    } else {
        runs
    }
}

fn estimate_lz_payload_bits(tokens: &[LzToken], distance_context: u32) -> u64 {
    let num_contexts = distance_context as usize + 1;
    let context_map: Vec<u8> = (0..num_contexts as u8).collect();
    let histograms = lz_build_histograms(tokens, &context_map, num_contexts, LZ77_MIN_SYMBOL);
    let codes = crate::entropy::build_huffman_codes(&histograms);
    let mut bits = 0u64;
    for &token in tokens {
        match token {
            LzToken::Pixel { context, value } => {
                let (symbol, nbits, _) = crate::entropy::uint_encode(value);
                bits += codes[context as usize].depths[symbol as usize] as u64 + nbits as u64;
            }
            LzToken::Lz77 {
                pixel_context,
                length_value,
                distance_value,
                ..
            } => {
                let (symbol, nbits, _) = lz77_length_encode(length_value);
                bits += codes[pixel_context as usize].depths[(LZ77_MIN_SYMBOL + symbol) as usize]
                    as u64
                    + nbits as u64;
                let (symbol, nbits, _) = crate::entropy::uint_encode(distance_value);
                bits +=
                    codes[distance_context as usize].depths[symbol as usize] as u64 + nbits as u64;
            }
        }
    }
    bits
}

fn lz77_run_token_count(tokens: &[Token]) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        count += 1;
        let token = tokens[i];
        let mut end = i + 1;
        while end < tokens.len()
            && tokens[end].context == token.context
            && tokens[end].value == token.value
        {
            end += 1;
        }
        if end - i > LZ77_MIN_LENGTH as usize {
            count += 1;
            i = end;
        } else {
            i += 1;
        }
    }
    count
}

fn lz77_compress_runs(tokens: &[Token], distance_context: u32) -> Vec<LzToken> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0usize;
    while i < tokens.len() {
        let token = tokens[i];
        out.push(LzToken::Pixel {
            context: token.context,
            value: token.value,
        });
        let mut end = i + 1;
        while end < tokens.len()
            && tokens[end].context == token.context
            && tokens[end].value == token.value
        {
            end += 1;
        }
        let copied = end - i - 1;
        if copied >= LZ77_MIN_LENGTH as usize {
            out.push(LzToken::Lz77 {
                pixel_context: token.context,
                distance_context,
                length_value: copied as u32 - LZ77_MIN_LENGTH,
                distance_value: LZ77_DIST_VALUE,
            });
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

fn lz77_compress_with_depth(
    tokens: &[Token],
    distance_context: u32,
    max_probes: usize,
) -> Vec<LzToken> {
    let mut out: Vec<LzToken> = Vec::with_capacity(tokens.len());
    let mut head = vec![u32::MAX; 1 << 18];
    let mut prev = vec![u32::MAX; tokens.len()];
    let mut i = 0;
    while i < tokens.len() {
        let (match_len, distance) = lz_find_match(tokens, i, &head, &prev, max_probes);
        // Four literals are the conservative break-even after the length and
        // distance symbols. Longer distances need one more copied token.
        let threshold = if distance <= 16 { 4 } else { 5 };
        if match_len >= threshold {
            // One-token lazy matching: do not consume a merely adequate match
            // when the next position opens a materially longer one.
            let hash = lz_hash(tokens, i);
            let old_head = head[hash];
            prev[i] = old_head;
            head[hash] = i as u32;
            let (next_len, _) = if i + 1 < tokens.len() {
                lz_find_match(tokens, i + 1, &head, &prev, max_probes)
            } else {
                (0, 0)
            };
            if next_len > match_len + 1 {
                let t = tokens[i];
                out.push(LzToken::Pixel {
                    context: t.context,
                    value: t.value,
                });
                i += 1;
                continue;
            }
            head[hash] = old_head;
            prev[i] = u32::MAX;

            let distance_value = if distance == 1 {
                LZ77_DIST_VALUE
            } else {
                LZ77_NUM_SPECIAL_DISTANCES + distance as u32 - 1
            };
            out.push(LzToken::Lz77 {
                pixel_context: tokens[i].context,
                distance_context,
                length_value: match_len as u32 - LZ77_MIN_LENGTH,
                distance_value,
            });
            for p in i..i + match_len {
                let hash = lz_hash(tokens, p);
                prev[p] = head[hash];
                head[hash] = p as u32;
            }
            i += match_len;
        } else {
            let t = tokens[i];
            out.push(LzToken::Pixel {
                context: t.context,
                value: t.value,
            });
            let hash = lz_hash(tokens, i);
            prev[i] = head[hash];
            head[hash] = i as u32;
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
    grad_pack_fn: GradPackInteriorFn,
    num_threads: usize,
) -> Vec<Token> {
    let nb_chans = num_color + if alpha.is_some() { 1 } else { 0 };
    let channel_tokens = crate::thread_pool::steal_map(nb_chans, num_threads, |chan| {
        let mut out = Vec::with_capacity(gw * gh);
        let ctx = channel_to_context(chan, nb_chans);
        if chan < num_color {
            let get = |gx: usize, gy: usize| linear.plane_row(chan, y0 + gy)[x0 + gx];
            tokenize_plane(ctx, get, gw, gh, predictors[chan], grad_pack_fn, &mut out);
        } else {
            let a = alpha.expect("alpha channel must exist");
            let get = |gx: usize, gy: usize| a.get_i32((y0 + gy) * xsize + (x0 + gx));
            tokenize_plane(ctx, get, gw, gh, predictors[chan], grad_pack_fn, &mut out);
        }
        out
    });
    let mut out = Vec::with_capacity(gw * gh * nb_chans);
    for channel in channel_tokens {
        out.extend(channel);
    }
    out
}

#[derive(Default)]
struct GradientScratch {
    cur: Vec<i32>,
    prev: Vec<i32>,
    buf: Vec<u32>,
}

thread_local! {
    static GRADIENT_SCRATCH: RefCell<GradientScratch> =
        RefCell::new(GradientScratch::default());
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
    grad_pack_fn: GradPackInteriorFn,
    out: &mut Vec<Token>,
) {
    if pred_id == PREDICTOR_WEIGHTED {
        // Weighted predictor: sequential (per-pixel error feedback), unchanged.
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
                let p = wp.predict(gx, gy, n_, w_, ne_, nw_, nn_);
                wp.update(v, gx, gy);
                out.push(Token::new(ctx, pack_signed((v - p) as i32)));
            }
        }
    } else if pred_id == PREDICTOR_GRADIENT {
        // Gradient (ClampedGradient): per-pixel independent, pure integer ->
        // vectorized over the interior of each row.
        GRADIENT_SCRATCH.with_borrow_mut(|scratch| {
            if scratch.buf.len() < gw {
                scratch.buf.resize(gw, 0);
            }
            if scratch.cur.len() < gw {
                scratch.cur.resize(gw, 0);
            }
            if scratch.prev.len() < gw {
                scratch.prev.resize(gw, 0);
            }
            let mut cur = &mut scratch.cur[..gw];
            let mut prev = &mut scratch.prev[..gw];
            let buf = &mut scratch.buf[..gw];
            for gy in 0..gh {
                std::mem::swap(&mut cur, &mut prev); // prev = last row's cur
                for (gx, c) in cur.iter_mut().enumerate() {
                    *c = get(gx, gy);
                }
                if gy == 0 {
                    buf[0] = pack_signed(cur[0]); // gx 0: pred = 0
                    for gx in 1..gw {
                        buf[gx] = pack_signed(cur[gx].wrapping_sub(cur[gx - 1]));
                        // pred = W
                    }
                } else {
                    buf[0] = pack_signed(cur[0].wrapping_sub(prev[0])); // gx 0: pred = N
                    grad_pack_fn(cur, prev, buf, gw); // gx in 1..gw
                }
                for &b in buf.iter().take(gw) {
                    out.push(Token::new(ctx, b));
                }
            }
        });
    } else {
        debug_assert!(matches!(
            pred_id,
            PREDICTOR_AVERAGE4 | PREDICTOR_SELECT | PREDICTOR_LEFT | PREDICTOR_TOP
        ));
        for gy in 0..gh {
            for gx in 0..gw {
                let value = get(gx, gy) as i64;
                let neighbors = predictor_neighbors(&get, gx, gy, gw);
                let pred = predictor_value(pred_id, neighbors, 0);
                out.push(Token::new(ctx, pack_signed((value - pred) as i32)));
            }
        }
    }
}
type GradPackInteriorFn = fn(&[i32], &[i32], &mut [u32], usize);
fn select_grad_pack_interior_fn() -> GradPackInteriorFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if is_x86_feature_detected!("avx2") {
        return |c, p, o, g| unsafe { crate::avx::grad_pack_interior(c, p, o, g) };
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
    if is_x86_feature_detected!("sse4.1") {
        return |c, p, o, g| unsafe { crate::sse::grad_pack_interior(c, p, o, g) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |c, p, o, g| unsafe { crate::neon::grad_pack_interior(c, p, o, g) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
    {
        crate::wasm::grad_pack_interior
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
    )))]
    {
        grad_pack_interior_scalar
    }
}

static GRAD_PACK_INTERIOR_FN: OnceLock<GradPackInteriorFn> = OnceLock::new();

#[inline]
fn selected_grad_pack_interior_fn() -> GradPackInteriorFn {
    *GRAD_PACK_INTERIOR_FN.get_or_init(select_grad_pack_interior_fn)
}

#[allow(unused)]
fn grad_pack_interior_scalar(cur: &[i32], prev: &[i32], out: &mut [u32], gw: usize) {
    for gx in 1..gw {
        let w = cur[gx - 1];
        let n = prev[gx];
        let nw = prev[gx - 1];
        let ac = w.wrapping_sub(nw);
        let bc = n.wrapping_sub(nw);
        let grad = ac.wrapping_add(n);
        let clamp = if (w.wrapping_sub(n) ^ bc) < 0 { n } else { w };
        let pred = if (ac ^ bc) < 0 { grad } else { clamp };
        out[gx] = pack_signed(cur[gx].wrapping_sub(pred));
    }
}

#[derive(Default)]
struct PredictorCosts {
    histograms: [Vec<u64>; SLOW_PREDICTORS.len()],
    total: u64,
}

impl PredictorCosts {
    #[inline]
    fn add(&mut self, value: i64, neighbors: PredictorNeighbors, weighted: i64) {
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate() {
            let pred = predictor_value(pred_id, neighbors, weighted);
            let symbol = pack_signed((value - pred) as i32) as usize;
            let hist = &mut self.histograms[candidate];
            if hist.len() <= symbol {
                hist.resize(symbol + 1, 0);
            }
            hist[symbol] += 1;
        }
        self.total += 1;
    }

    fn best_predictor(&self) -> u32 {
        let mut best_id = SLOW_PREDICTORS[0];
        let mut best_bits = entropy_of_hist(&self.histograms[0], self.total);
        for (candidate, &pred_id) in SLOW_PREDICTORS.iter().enumerate().skip(1) {
            let bits = entropy_of_hist(&self.histograms[candidate], self.total);
            if bits < best_bits {
                best_bits = bits;
                best_id = pred_id;
            }
        }
        best_id
    }
}

/// Evaluate all Slow-mode predictors in one traversal and choose the lowest
/// order-0 residual entropy. Weighted remains the deterministic tie-breaker.
fn choose_predictor_for_plane(get: impl Fn(usize, usize) -> i32, w: usize, h: usize) -> u32 {
    if w == 0 || h == 0 {
        return PREDICTOR_WEIGHTED;
    }
    let mut wp = WpState::new(w);
    let mut costs = PredictorCosts::default();
    for gy in 0..h {
        for gx in 0..w {
            let value = get(gx, gy) as i64;
            let neighbors = predictor_neighbors(&get, gx, gy, w);
            let weighted = wp.predict(
                gx,
                gy,
                neighbors.top,
                neighbors.left,
                neighbors.top_right,
                neighbors.top_left,
                neighbors.top_top,
            );
            costs.add(value, neighbors, weighted);
            wp.update(value, gx, gy);
        }
    }
    costs.best_predictor()
}

fn choose_predictors(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    num_threads: usize,
) -> [u32; 4] {
    let mut preds = [PREDICTOR_WEIGHTED; 4];
    let num_channels = 3 + usize::from(alpha.is_some());
    let selected = crate::thread_pool::steal_map(num_channels, num_threads, |chan| {
        if chan < 3 {
            let pd = linear.plane_data(chan);
            choose_predictor_for_plane(|x, y| pd[y * xsize + x], xsize, ysize)
        } else {
            let a = alpha.expect("alpha channel must exist");
            choose_predictor_for_plane(|x, y| a.get_i32(y * xsize + x), xsize, ysize)
        }
    });
    preds[..num_channels].copy_from_slice(&selected);
    if alpha.is_none() {
        preds[3] = PREDICTOR_WEIGHTED;
    }
    preds
}

// ---------------------------------------------------------------------------
// Context tree (MA tree): split each channel's entropy context on the WP error
// property kWPProp (p[15]) into 3 activity buckets. The decoder must run the WP
// state for every pixel of a channel whose subtree references p[15], so we run
// WP for all channels here regardless of the selected leaf predictor.
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

thread_local! {
    static ORDER0_ENTROPY_SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(vec![]) };
}

fn order0_entropy(vals: &[u32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    ORDER0_ENTROPY_SCRATCH.with_borrow_mut(|cell| {
        // Direct-indexed frequency histogram (residual symbols are small-range), in
        // place of a HashMap: no hashing, and a deterministic accumulation order.
        let max = vals.iter().copied().max().unwrap_or(0) as usize;
        if cell.len() < max + 1 {
            cell.resize(max + 1, 0);
        }
        let hist = &mut cell[..max + 1];
        hist.fill(0);
        for &v in vals {
            hist[v as usize] += 1;
        }
        let total = vals.len() as f32;
        let mut bits = 0.0;
        for &c in hist.iter() {
            if c != 0 {
                let p = c as f32 / total;
                bits -= c as f32 * dirty_log2f(p);
            }
        }
        bits
    })
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
            let neighbors = predictor_neighbors(&get, gx, gy, gw);
            let wp_pred = wp.predict(
                gx,
                gy,
                neighbors.top,
                neighbors.left,
                neighbors.top_right,
                neighbors.top_left,
                neighbors.top_top,
            );
            prp.push(wp.wp_prop);
            let pred = predictor_value(pred_id, neighbors, wp_pred);
            res.push(pack_signed((v - pred) as i32));
            wp.update(v, gx, gy);
        }
    }
    (res, prp)
}

/// Pick the best activity threshold for a channel among candidates, returning
/// (best_t, best_bucketed_bits, flat_bits).
fn entropy_of_hist(hist: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f32;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c != 0 {
            let p = c as f32 / t;
            bits -= c as f32 * p.log2();
        }
    }
    bits
}

struct PickThresholdScratch {
    hist_scratch: Vec<u64>,
}

impl PickThresholdScratch {
    fn make_scratches(&mut self, size: usize) -> (&mut [u64], &mut [u64], &mut [u64]) {
        let bucket = size + 1;
        if self.hist_scratch.len() < bucket * 3 {
            self.hist_scratch.resize(bucket * 3, 0);
        }
        let (b0, r0) = self.hist_scratch.split_at_mut(bucket);
        let (b1, b2) = r0.split_at_mut(bucket);
        (b0, b1, &mut b2[..bucket])
    }
}

thread_local! {
    static THRESHOLD_SCRATCH: RefCell<PickThresholdScratch> = const { RefCell::new(PickThresholdScratch {
        hist_scratch: Vec::new(),
    }) }
}

fn pick_threshold(res: &[u32], prp: &[i64]) -> (i32, f32, f32) {
    THRESHOLD_SCRATCH.with_borrow_mut(|cell| {
        let flat = order0_entropy(res);
        let max = res.iter().copied().max().unwrap_or(0) as usize;
        let (h0, h1, h2) = cell.make_scratches(max + 1);
        let mut best_t = 0i32;
        let mut best_bits = f32::INFINITY;
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
            let bits = entropy_of_hist(h0, n0) + entropy_of_hist(h1, n1) + entropy_of_hist(h2, n2);
            if bits < best_bits {
                best_bits = bits;
                best_t = t as i32;
            }
        }
        (best_t, best_bits, flat)
    })
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
    num_threads: usize,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };

    // Collect residuals + WP property per channel (WP runs for every channel).
    let collected = crate::thread_pool::steal_map(nb_chans, num_threads, |chan| {
        if chan < 3 {
            let pd = linear.plane_data(chan);
            collect_channel(|x, y| pd[y * xsize + x], xsize, ysize, predictors[chan])
        } else {
            let a = alpha.expect("alpha channel must exist");
            collect_channel(
                |x, y| a.get_i32(y * xsize + x),
                xsize,
                ysize,
                predictors[chan],
            )
        }
    });
    let (chan_res, chan_prp): (Vec<Vec<u32>>, Vec<Vec<i64>>) = collected.into_iter().unzip();

    // Per-channel threshold + cost comparison.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    let threshold_costs = crate::thread_pool::steal_map(nb_chans, num_threads, |chan| {
        pick_threshold(&chan_res[chan], &chan_prp[chan])
    });
    for (chan, (t, cb, fb)) in threshold_costs.into_iter().enumerate() {
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    // Guard: require the context tree to beat the flat path by more than the
    // extra-context header overhead (~64 bytes per added context, conservative).
    let overhead_bits = (2 * nb_chans) as f32 * 64.0 * 8.0;
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
    let channel_tokens = crate::thread_pool::steal_map(nb_chans, num_threads, |chan| {
        let mut tokens = Vec::with_capacity(xsize * ysize);
        let res = &chan_res[chan];
        let prp = &chan_prp[chan];
        let t = ts[chan] as i64;
        for (&prp, &res) in prp[..res.len()].iter().zip(res.iter()) {
            let bucket = bucket_of(prp, t);
            let ctx = ctx_lut[chan * 3 + bucket as usize];
            tokens.push(Token::new(ctx, res));
        }
        tokens
    });
    let mut tokens: Vec<Token> = Vec::with_capacity(xsize * ysize * nb_chans);
    for channel in channel_tokens {
        tokens.extend(channel);
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
    let lz_tokens = lz77_compress_runs(&tokens, distance_ctx);
    let code = build_lz_pixel_code(&lz_tokens, num_pixel_ctx, min_symbol, true);
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
    num_threads: usize,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + if alpha.is_some() { 1 } else { 0 };
    let num_ac_groups = xsize_groups * ysize_groups;

    // 1) Collect (residual, WP property) per group per channel (group-local WP).
    let groups: Vec<Vec<(Vec<u32>, Vec<i64>)>> =
        crate::thread_pool::steal_map(num_ac_groups, num_threads, |group_index| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let gw = GROUP_DIM.min(xsize - x0);
            let gh = GROUP_DIM.min(ysize - y0);
            let mut chans: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(nb_chans);
            for chan in 0..3usize {
                let pd = linear.plane_data(chan);
                let get = |lx: usize, ly: usize| pd[(y0 + ly) * xsize + (x0 + lx)];
                chans.push(collect_channel(get, gw, gh, predictors[chan]));
            }
            if let Some(a) = alpha {
                let get = |lx: usize, ly: usize| a.get_i32((y0 + ly) * xsize + (x0 + lx));
                chans.push(collect_channel(get, gw, gh, predictors[3]));
            }
            chans
        });

    // 2) Global per-channel threshold from aggregated stats.
    let mut ts = [0i32; 4];
    let mut ctx_bits = 0.0;
    let mut flat_bits = 0.0;
    let threshold_costs = crate::thread_pool::steal_map(nb_chans, num_threads, |chan| {
        let mut res: Vec<u32> = Vec::new();
        let mut prp: Vec<i64> = Vec::new();
        for g in &groups {
            res.extend_from_slice(&g[chan].0);
            prp.extend_from_slice(&g[chan].1);
        }
        pick_threshold(&res, &prp)
    });
    for (chan, (t, cb, fb)) in threshold_costs.into_iter().enumerate() {
        ts[chan] = t;
        ctx_bits += cb;
        flat_bits += fb;
    }
    let overhead_bits = (2 * nb_chans) as f32 * 64.0 * 8.0;
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
    let group_lz_tokens: Vec<Vec<LzToken>> =
        crate::thread_pool::steal_map(num_ac_groups, num_threads, |group_index| {
            let g = &groups[group_index];
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
            lz77_compress_runs(&toks, distance_ctx)
        });
    let mut all_lz: Vec<LzToken> = Vec::new();
    for lz in &group_lz_tokens {
        all_lz.extend_from_slice(lz);
    }
    let code = build_lz_pixel_code(&all_lz, num_pixel_ctx, min_symbol, true);

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

use crate::adaptive_quant::dirty_log2f;
use crate::bit_writer::BitWriter;
use crate::encode_image::AlphaPlane;
use crate::entropy::{
    Histogram, OwnedEntropyCode, Token, optimize_entropy_code, pack_signed, write_entropy_code,
    write_token,
};
use crate::image::Image3Si;
use std::cell::RefCell;
use std::sync::OnceLock;

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
                distance_value,
            } => {
                let (len_tok, _, _) = lz77_length_encode(length_value);
                let pixel_cluster = context_map[pixel_context as usize] as usize;
                hs[pixel_cluster].add(min_symbol + len_tok);

                let dist_cluster = context_map[distance_context as usize] as usize;
                let (symbol, _, _) = crate::entropy::uint_encode(distance_value);
                hs[dist_cluster].add(symbol);
            }
        }
    }
    hs
}

/// Build per-cluster prefix codes from an `LzToken` stream.
/// `nb_chans + 1` contexts: `nb_chans` channel leaves + 1 distance context.
fn build_lz_pixel_code(
    toks: &[LzToken],
    nb_chans: usize,
    min_symbol: u32,
    refined: bool,
) -> OwnedEntropyCode {
    let refined = refined
        && toks.iter().any(|token| {
            matches!(
                token,
                LzToken::Lz77 { distance_value, .. }
                    if *distance_value != LZ77_DIST_VALUE
            )
        });
    use crate::entropy::build_huffman_codes;
    use crate::entropy::cluster_histograms;

    let num_contexts = nb_chans + 1;
    let context_map_initial: Vec<u8> = (0..num_contexts).map(|i| i as u8).collect();
    let mut histograms = lz_build_histograms(toks, &context_map_initial, num_contexts, min_symbol);

    let mut context_map: Vec<u8> = Vec::new();
    if refined {
        crate::entropy::cluster_histograms_refined(&mut histograms, &mut context_map);
    } else {
        cluster_histograms(&mut histograms, &mut context_map);
    }

    let hybrid_uint_configs = if refined {
        let mut raw_values = vec![Vec::<u32>::new(); histograms.len()];
        let mut literal_values = vec![Vec::<u32>::new(); histograms.len()];
        for &tok in toks {
            match tok {
                LzToken::Pixel { context, value } => {
                    let cluster = context_map[context as usize] as usize;
                    raw_values[cluster].push(value);
                    literal_values[cluster].push(value);
                }
                LzToken::Lz77 {
                    distance_context,
                    distance_value,
                    ..
                } => {
                    raw_values[context_map[distance_context as usize] as usize]
                        .push(distance_value);
                }
            }
        }
        let configs: Vec<_> = raw_values
            .iter()
            .enumerate()
            .map(|(cluster, values)| {
                let selected = crate::entropy::select_hybrid_config(values);
                if literal_values[cluster].iter().all(|&value| {
                    crate::entropy::uint_encode_with_config(value, selected).0 < min_symbol
                }) {
                    selected
                } else {
                    crate::entropy::HybridUintConfig::DEFAULT
                }
            })
            .collect();
        histograms = vec![Histogram::new(); configs.len()];
        for &tok in toks {
            match tok {
                LzToken::Pixel { context, value } => {
                    let cluster = context_map[context as usize] as usize;
                    let (symbol, _, _) =
                        crate::entropy::uint_encode_with_config(value, configs[cluster]);
                    histograms[cluster].add(symbol);
                }
                LzToken::Lz77 {
                    pixel_context,
                    distance_context,
                    length_value,
                    distance_value,
                } => {
                    let (len_tok, _, _) = lz77_length_encode(length_value);
                    histograms[context_map[pixel_context as usize] as usize]
                        .add(min_symbol + len_tok);
                    let cluster = context_map[distance_context as usize] as usize;
                    let (symbol, _, _) =
                        crate::entropy::uint_encode_with_config(distance_value, configs[cluster]);
                    histograms[cluster].add(symbol);
                }
            }
        }
        configs
    } else {
        vec![crate::entropy::HybridUintConfig::DEFAULT; histograms.len()]
    };

    let mut code = OwnedEntropyCode {
        context_map,
        prefix_codes: build_huffman_codes(&histograms),
        hybrid_uint_configs,
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
            let cluster = code.context_map[context as usize] as usize;
            let (sym, nbits, bits) =
                crate::entropy::uint_encode_with_config(value, code.hybrid_uint_configs[cluster]);
            let pc = &code.prefix_codes[cluster];
            let d = pc.depths[sym as usize] as usize;
            let data = (pc.bits[sym as usize] as u64) | ((bits as u64) << d);
            w.write(d + nbits as usize, data);
        }
        LzToken::Lz77 {
            pixel_context,
            distance_context,
            length_value,
            distance_value,
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
            let (dist_symbol, dist_nbits, dist_bits) = crate::entropy::uint_encode_with_config(
                distance_value,
                code.hybrid_uint_configs[dcluster],
            );
            let dd = dc.depths[dist_symbol as usize] as usize;
            // (Could be 0 if it's the only symbol in a single-symbol histogram.)
            if dd > 0 {
                let data = dc.bits[dist_symbol as usize] as u64 | ((dist_bits as u64) << dd);
                w.write(dd + dist_nbits as usize, data);
            } else if dist_nbits != 0 {
                w.write(dist_nbits as usize, dist_bits as u64);
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
    num_threads: usize,
    writer: &mut BitWriter,
) {
    let num_threads = num_threads.max(1);
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
    let grad_pack_fn = selected_grad_pack_interior_fn();

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
            grad_pack_fn,
            num_threads,
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
        let group_tokens: Vec<Vec<Token>> =
            crate::thread_pool::steal_map(num_ac_groups, num_threads, |group_index| {
                let gx = group_index % xsize_groups;
                let gy = group_index / xsize_groups;
                let x0 = gx * GROUP_DIM;
                let y0 = gy * GROUP_DIM;
                let gw = GROUP_DIM.min(xsize - x0);
                let gh = GROUP_DIM.min(ysize - y0);
                tokenize_all(
                    linear,
                    alpha,
                    xsize,
                    ysize,
                    x0,
                    y0,
                    gw,
                    gh,
                    3,
                    &predictors,
                    grad_pack_fn,
                    1,
                )
            });
        let mut all_tokens: Vec<Token> = Vec::new();
        for tokens in &group_tokens {
            all_tokens.extend_from_slice(tokens);
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

#[cfg(test)]
mod predictor_tests {
    use super::*;

    #[test]
    fn added_predictors_match_jxl_formulas() {
        let n = PredictorNeighbors {
            left: 11,
            top: 7,
            top_left: 5,
            top_right: 13,
            left_left: 3,
            top_top: 2,
            top_right_right: 17,
        };
        assert_eq!(predictor_value(PREDICTOR_LEFT, n, 99), 11);
        assert_eq!(predictor_value(PREDICTOR_TOP, n, 99), 7);
        assert_eq!(predictor_value(PREDICTOR_SELECT, n, 99), 11);
        assert_eq!(predictor_value(PREDICTOR_AVERAGE4, n, 99), 11);

        // Select resolves equal distances toward Top, matching libjxl's
        // `pa < pb ? left : top`.
        let tie = PredictorNeighbors {
            left: 9,
            top: 3,
            top_left: 6,
            ..n
        };
        assert_eq!(predictor_value(PREDICTOR_SELECT, tie, 99), 3);
    }

    #[test]
    fn slow_search_selects_directional_predictors() {
        const W: usize = 64;
        const H: usize = 64;
        let mut horizontal = vec![0i32; W * H];
        for y in 0..H {
            horizontal[y * W] = ((y * 73) & 255) as i32;
            for x in 1..W {
                let delta = (((x * 17 + y * 31) ^ (x * y * 3)) % 7) as i32 - 3;
                horizontal[y * W + x] = horizontal[y * W + x - 1] + delta;
            }
        }
        let mut vertical = vec![0i32; W * H];
        for x in 0..W {
            vertical[x] = ((x * 73) & 255) as i32;
            for y in 1..H {
                let delta = (((y * 17 + x * 31) ^ (x * y * 3)) % 7) as i32 - 3;
                vertical[y * W + x] = vertical[(y - 1) * W + x] + delta;
            }
        }

        assert_eq!(
            choose_predictor_for_plane(|x, y| horizontal[y * W + x], W, H),
            PREDICTOR_SELECT
        );
        assert_eq!(
            choose_predictor_for_plane(|x, y| vertical[y * W + x], W, H),
            PREDICTOR_SELECT
        );
    }

    #[test]
    fn slow_search_selects_average4_for_its_recurrence() {
        const W: usize = 64;
        const H: usize = 64;
        let mut plane = vec![0i32; W * H];
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                if y < 2 || x < 2 {
                    plane[i] = (((x * 97 + y * 53) ^ (x * y * 11)) & 1023) as i32 - 512;
                    continue;
                }
                let top_right = if x + 1 < W {
                    plane[(y - 1) * W + x + 1]
                } else {
                    plane[(y - 1) * W + x]
                };
                let top_right_right = if x + 2 < W {
                    plane[(y - 1) * W + x + 2]
                } else {
                    top_right
                };
                plane[i] = (6 * plane[(y - 1) * W + x] - 2 * plane[(y - 2) * W + x]
                    + 7 * plane[i - 1]
                    + plane[i - 2]
                    + top_right_right
                    + 3 * top_right
                    + 8)
                    / 16;
            }
        }

        assert_eq!(
            choose_predictor_for_plane(|x, y| plane[y * W + x], W, H),
            PREDICTOR_AVERAGE4
        );
    }
}

#[cfg(test)]
mod lz77_tests {
    use super::*;

    fn expand(stream: &[LzToken]) -> Vec<Token> {
        let mut out = Vec::new();
        for &token in stream {
            match token {
                LzToken::Pixel { context, value } => out.push(Token::new(context, value)),
                LzToken::Lz77 {
                    length_value,
                    distance_value,
                    ..
                } => {
                    let distance = if distance_value == LZ77_DIST_VALUE {
                        1
                    } else {
                        (distance_value - LZ77_NUM_SPECIAL_DISTANCES + 1) as usize
                    };
                    for _ in 0..length_value + LZ77_MIN_LENGTH {
                        let source = out.len() - distance;
                        out.push(out[source]);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn hash_chain_finds_non_run_matches_and_round_trips() {
        let pattern: Vec<Token> = (0..37)
            .map(|i| Token::new((i % 3) as u32, ((i * 29 + i * i) % 251) as u32))
            .collect();
        let mut input = pattern.clone();
        input.extend((0..11).map(|i| Token::new(1, 900 + i)));
        input.extend_from_slice(&pattern);
        input.extend_from_slice(&pattern);

        let compressed = lz77_compress(&input, 3);
        assert!(compressed.iter().any(|token| matches!(
            token,
            LzToken::Lz77 { distance_value, .. }
                if *distance_value >= LZ77_NUM_SPECIAL_DISTANCES
        )));
        let decoded = expand(&compressed);
        assert_eq!(decoded.len(), input.len());
        assert!(
            decoded
                .iter()
                .zip(input.iter())
                .all(|(a, b)| a.context == b.context && a.value == b.value)
        );
    }

    #[test]
    fn hash_chain_uses_compact_distance_for_runs() {
        let input = vec![Token::new(0, 7); 128];
        let compressed = lz77_compress(&input, 1);
        assert!(compressed.iter().any(|token| matches!(
            token,
            LzToken::Lz77 {
                distance_value: LZ77_DIST_VALUE,
                ..
            }
        )));
        assert_eq!(expand(&compressed).len(), input.len());
    }

    #[test]
    fn speed_policy_keeps_fast_run_only_and_slow_structured_search() {
        let pattern: Vec<Token> = (0..64)
            .map(|i| Token::new((i % 3) as u32, ((i * 37 + 11) % 257) as u32))
            .collect();
        let mut input = pattern.clone();
        for _ in 0..7 {
            input.extend_from_slice(&pattern);
        }

        let fast = lz77_compress_for_speed(&input, 3, crate::Speed::Fast);
        assert!(!fast.iter().any(|token| matches!(
            token,
            LzToken::Lz77 { distance_value, .. }
                if *distance_value != LZ77_DIST_VALUE
        )));

        let slow = lz77_compress_for_speed(&input, 3, crate::Speed::Slow);
        assert!(slow.iter().any(|token| matches!(
            token,
            LzToken::Lz77 { distance_value, .. }
                if *distance_value != LZ77_DIST_VALUE
        )));
        assert_eq!(expand(&slow).len(), input.len());
    }
}
