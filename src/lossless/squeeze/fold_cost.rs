/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Bounded coarse-only screening for exact Squeeze representation trials.

use super::super::predictor::{
    PREDICTOR_GRADIENT, selected_grad_pack_interior_fn, tokenize_plane_rows,
};
use super::quantize::LmQuantized;
use crate::coder_scratch::CoderScratch;
use crate::entropy::{ALPHABET_SIZE, Token, f_log2, pack_signed, uint_encode};
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

/// Pick one promising fold using only the affected coarse prefix. This is a
/// heuristic work filter, not a coded-size bound: the caller still races the
/// complete serialized frame before accepting any representation.
pub(super) fn select_fold(
    q: &LmQuantized,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> Option<usize> {
    if q.dc_frame || !speed.effort().predictor_search || q.channels.len() != q.quants.len() {
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
            || ch.w > super::GROUP_DIM
            || ch.h > super::GROUP_DIM
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
            dims.is_some_and(|(w, h)| w <= super::GROUP_DIM && h <= super::GROUP_DIM)
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
    use super::super::quantize::{LmPlainInput, lm_quantize_plain};
    use super::*;
    use crate::coding::CodingTransform;

    fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> i32) -> LmQuantized {
        let input = LmPlainInput {
            color: vec![(0..w * h).map(|i| f(i % w, i / w)).collect()],
            extra: Vec::new(),
            max: 65535.0,
            coding: CodingTransform::Rgb,
        };
        lm_quantize_plain(&input, w, h, 0.00001).unwrap()
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
        let input = LmPlainInput {
            color: (0..3).map(|c| vec![12000 + c * 1300; 128 * 128]).collect(),
            extra: Vec::new(),
            max: 65535.0,
            coding: CodingTransform::Rgb,
        };
        let q = lm_quantize_plain(&input, 128, 128, 0.00001).unwrap();
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
