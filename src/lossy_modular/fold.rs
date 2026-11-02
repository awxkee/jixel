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

//! Stage 3: folding the pyramid's exact coarse tail.
//!
//! Once the coarse steps reach unit quantizers, the decoder's averages at
//! that depth are exact integers. Folding such a step back (merging its
//! average and residual into the full-scale average) leaves the decoded image
//! unchanged but codes the coarse prefix differently, sometimes in fewer
//! bytes. [`select_fold`] screens the candidates on the coarse prefix alone;
//! [`super::race`] then serializes the frame both ways.

use super::pyramid::Pyramid;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{ALPHABET_SIZE, Token, f_log2, pack_signed, uint_encode};
use crate::lossless::kit::{
    PREDICTOR_GRADIENT, selected_grad_pack_interior_fn, tokenize_plane_rows,
};
use crate::squeeze::{Channel, fold_quantized_step};

const MODEL_BITS_PER_CHANNEL: f64 = 48.0;
// Valid coarse prefixes are normally only a few thousand samples. This also
// bounds work and allocation if an unexpected pyramid reaches the screen.
const MAX_COARSE_SAMPLES: usize = 1 << 18;
const MAX_COARSE_CHANNELS: usize = 256;

struct SymbolCost {
    counts: [u32; ALPHABET_SIZE],
    raw_bits: u64,
    total: u32,
}

impl Default for SymbolCost {
    fn default() -> Self {
        Self {
            counts: [0; ALPHABET_SIZE],
            raw_bits: 0,
            total: 0,
        }
    }
}

impl SymbolCost {
    fn add(&mut self, value: u32) {
        let (symbol, bits, _) = uint_encode(value);
        self.counts[symbol as usize] += 1;
        self.raw_bits += u64::from(bits);
        self.total += 1;
    }

    fn bits(&self) -> f64 {
        let total = f64::from(self.total);
        self.raw_bits as f64
            + self
                .counts
                .iter()
                .filter(|&&count| count != 0)
                .map(|&count| f64::from(count) * f_log2(total / f64::from(count)))
                .sum::<f64>()
    }
}

fn channel_bits(ch: &Channel, tokens: &mut Vec<Token>, scratch: &mut CoderScratch) -> f64 {
    let mut zero = SymbolCost::default();
    for &value in &ch.data {
        zero.add(pack_signed(value));
    }
    tokens.clear();
    tokenize_plane_rows(
        0,
        |y| &ch.data[y * ch.w..][..ch.w],
        ch.w,
        ch.h,
        PREDICTOR_GRADIENT,
        selected_grad_pack_interior_fn(),
        &mut scratch.gradient,
        tokens,
    );
    let mut gradient = SymbolCost::default();
    for token in tokens.iter() {
        gradient.add(token.value);
    }
    zero.bits().min(gradient.bits()) + MODEL_BITS_PER_CHANNEL
}

fn prefix_bits(channels: &[Channel], scratch: &mut CoderScratch) -> f64 {
    let mut tokens = Vec::new();
    channels
        .iter()
        .map(|channel| channel_bits(channel, &mut tokens, scratch))
        .sum()
}

impl Pyramid {
    /// Fold one coarse step using the integers reconstructed by the decoder.
    /// Retained bands keep their original multipliers; the merged averages
    /// become full-scale integer channels. Layout, EPF and DC scaling stay put.
    pub(super) fn fold_last(&mut self) -> bool {
        if self.steps.len() <= 1 {
            return false;
        }
        let step = *self.steps.last().unwrap();
        // Keep the chroma pre-squeeze and at least one explicit step. The
        // shared coarse suffix always splits all original channels in place.
        if !step.in_place || step.begin_c != 0 {
            return false;
        }
        let offset = step.num_c;
        if offset == 0 || offset > self.channels.len() / 2 {
            return false;
        }
        for c in 0..step.num_c {
            let avg = &self.channels[c];
            let res = &self.channels[offset + c];
            let dims = if step.horizontal {
                avg.w.checked_add(res.w).map(|w| (w, avg.h))
            } else {
                avg.h.checked_add(res.h).map(|h| (avg.w, h))
            };
            // The current global forest requires global channels. Keeping
            // every changed average global also leaves all group crops stable.
            if dims.is_none_or(|(w, h)| w > self.group_dim() || h > self.group_dim()) {
                return false;
            }
        }
        if !fold_quantized_step(&mut self.channels, &mut self.quants, &step) {
            return false;
        }
        self.steps.pop();
        true
    }
}

/// Pick one promising fold using only the affected coarse prefix. This is a
/// heuristic work filter, not a coded-size bound: the caller still races the
/// complete serialized frame before accepting any representation.
pub(super) fn select_fold(
    q: &Pyramid,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> Option<usize> {
    // A palette's meta channel shifts every channel off the squeeze script's
    // indices; such frames keep their tail.
    if q.dc_frame
        || q.palette.is_some()
        || !speed.effort().predictor_search
        || q.channels.len() != q.quants.len()
    {
        return None;
    }
    let last = *q.steps.last()?;
    if last.num_c == 0 || last.begin_c != 0 || !last.in_place {
        return None;
    }
    let exhaustive = speed.effort().ans_exhaustive();
    let cuts: &[usize] = if exhaustive { &[2, 4, 6] } else { &[4] };
    let shared_steps = q
        .steps
        .iter()
        .rev()
        .take_while(|step| step.in_place && step.begin_c == 0 && step.num_c == last.num_c)
        .count();
    let max_fold = cuts
        .iter()
        .copied()
        .filter(|&count| count <= shared_steps && count < q.steps.len())
        .max()?;
    let prefix = (max_fold + 1).checked_mul(last.num_c)?;
    if prefix > MAX_COARSE_CHANNELS {
        return None;
    }
    let original = q.channels.get(..prefix)?;
    let quants = q.quants.get(..prefix)?;
    let mut samples = 0usize;
    for (ch, &quant) in original.iter().zip(quants) {
        samples = samples.checked_add(ch.data.len())?;
        if quant == 0
            || ch.w.checked_mul(ch.h)? != ch.data.len()
            || ch.w > q.group_dim()
            || ch.h > q.group_dim()
            || samples > MAX_COARSE_SAMPLES
        {
            return None;
        }
    }
    let baseline = prefix_bits(original, scratch);
    let mut channels = original.to_vec();
    let mut quants = quants.to_vec();
    let mut best = None;
    let pixels = q.xsize.checked_mul(q.ysize)?;
    let mut best_gain = 8.0 * (32.0 + pixels as f64 / 2048.0);
    for folded in 1..=max_fold {
        let step = &q.steps[q.steps.len() - folded];
        let fits = (0..step.num_c).all(|c| {
            let avg = &channels[c];
            let residual = &channels[step.num_c + c];
            let dims = if step.horizontal {
                avg.w.checked_add(residual.w).map(|w| (w, avg.h))
            } else {
                avg.h.checked_add(residual.h).map(|h| (avg.w, h))
            };
            dims.is_some_and(|(w, h)| w <= q.group_dim() && h <= q.group_dim())
        });
        if !fits {
            break;
        }
        if !fold_quantized_step(&mut channels, &mut quants, step) {
            return None;
        }
        if cuts.contains(&folded) {
            let gain = baseline - prefix_bits(&channels, scratch);
            if gain > best_gain {
                best = Some(folded);
                best_gain = gain;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::super::pyramid::PlainInput;
    use super::*;
    use crate::coding::CodingTransform;

    fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> i32) -> Pyramid {
        let input = PlainInput {
            color: vec![(0..w * h).map(|i| f(i % w, i / w)).collect()],
            extra: Vec::new(),
            max: 65535.0,
            coding: CodingTransform::Rgb,
        };
        Pyramid::plain(
            &input,
            w,
            h,
            0.00001,
            false,
            &crate::thread_pool::ThreadPool::new(1),
        )
        .unwrap()
    }

    #[test]
    fn flat_prefix_is_not_worth_a_full_frame_trial() {
        let q = plane(257, 65, |_, _| 12000);
        assert_eq!(
            select_fold(&q, crate::Speed::Slow, &mut CoderScratch::default()),
            None
        );
    }

    #[test]
    fn screen_does_not_modify_the_quantized_pyramid() {
        let q = plane(129, 65, |x, y| 20000 + x as i32 * 113 - y as i32 * 59);
        let data: Vec<_> = q.channels.iter().map(|ch| ch.data.clone()).collect();
        let quants = q.quants.clone();
        let steps = q
            .steps
            .iter()
            .map(|s| (s.horizontal, s.in_place, s.begin_c, s.num_c))
            .collect::<Vec<_>>();
        let _ = select_fold(&q, crate::Speed::Slow, &mut CoderScratch::default());
        assert!(
            q.channels
                .iter()
                .zip(data)
                .all(|(ch, data)| ch.data == data)
        );
        assert_eq!(q.quants, quants);
        assert_eq!(
            q.steps
                .iter()
                .map(|s| (s.horizontal, s.in_place, s.begin_c, s.num_c))
                .collect::<Vec<_>>(),
            steps
        );
    }

    #[test]
    fn malformed_and_overflowing_prefixes_are_rejected() {
        let mut q = plane(65, 65, |x, y| 12000 + (x * y) as i32);
        q.quants[0] = u32::MAX;
        assert_eq!(
            select_fold(&q, crate::Speed::Slow, &mut CoderScratch::default()),
            None
        );
        q.quants[0] = 1;
        q.channels[0].data.pop();
        assert_eq!(
            select_fold(&q, crate::Speed::Slow, &mut CoderScratch::default()),
            None
        );
    }

    #[test]
    fn screen_can_select_a_profitable_coarse_representation() {
        // For a small three-channel image, eliminating the model overhead
        // itself can justify a trial. Larger images require a larger gain.
        let input = PlainInput {
            color: (0..3).map(|c| vec![12000 + c * 1300; 128 * 128]).collect(),
            extra: Vec::new(),
            max: 65535.0,
            coding: CodingTransform::Rgb,
        };
        let q = Pyramid::plain(
            &input,
            128,
            128,
            0.00001,
            false,
            &crate::thread_pool::ThreadPool::new(1),
        )
        .unwrap();
        assert_eq!(
            select_fold(&q, crate::Speed::Slow, &mut CoderScratch::default()),
            Some(4),
        );
    }

    #[test]
    fn signed_extremes_use_a_bounded_symbol_histogram() {
        let ch = Channel {
            data: vec![i32::MIN, i32::MAX, 0, -1, 1, i32::MIN],
            w: 3,
            h: 2,
            hshift: 0,
            vshift: 0,
        };
        let bits = channel_bits(&ch, &mut Vec::new(), &mut CoderScratch::default());
        assert!(bits.is_finite() && bits >= MODEL_BITS_PER_CHANNEL);
        assert!(bits <= MODEL_BITS_PER_CHANNEL + 40.0 * ch.data.len() as f64);
    }
}
