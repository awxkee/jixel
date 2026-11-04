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

//! Runtime dispatch for lossless histogram entropy estimates.

use std::sync::OnceLock;

pub(crate) type EntropyOfHistFn = fn(&[u64], u64) -> f32;

fn select_entropy_of_hist_fn() -> EntropyOfHistFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |hist, total| unsafe { crate::avx::entropy_of_hist_avx2(hist, total) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        |hist, total| unsafe { crate::neon::entropy_of_hist_neon(hist, total) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        entropy_of_hist_scalar
    }
}

/// Resolve CPU dispatch once, then retain the pointer in worker scratch or
/// predictor-cost storage so histogram loops never check the static guard.
pub(crate) fn selected_entropy_of_hist_fn() -> EntropyOfHistFn {
    static ENTROPY_OF_HIST: OnceLock<EntropyOfHistFn> = OnceLock::new();
    *ENTROPY_OF_HIST.get_or_init(select_entropy_of_hist_fn)
}

#[cfg(any(test, not(all(target_arch = "aarch64", feature = "neon"))))]
pub(super) fn entropy_of_hist_scalar(hist: &[u64], total: u64) -> f32 {
    use crate::adaptive_quant::dirty_log2f;

    if total == 0 {
        return 0.0;
    }
    // One reciprocal per histogram instead of a division per bin.
    let inv_t = 1.0 / total as f32;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c != 0 {
            let p = c as f32 * inv_t;
            bits -= c as f32 * dirty_log2f(p);
        }
    }
    bits
}
