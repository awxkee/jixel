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

//! Bounded candidate search for the rectangle-search tiers.

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CandidateSearch {
    pub fine_runner_up: bool,
    pub medium_upgrades: bool,
    pub merge32_upgrades: bool,
}

impl CandidateSearch {
    pub(super) fn for_speed(speed: crate::Speed, arbiter: crate::Arbiter) -> Self {
        let effort = speed.effort();
        // Without the full reconstruction rerank, the rectangle search keeps
        // a fine runner-up and verifies a few merge upgrades instead.
        let substitute = effort.rectangles && !effort.full_rerank;
        Self {
            fine_runner_up: substitute,
            medium_upgrades: substitute,
            // This search improves CVVDP, with a small SSIMULACRA2 tradeoff.
            merge32_upgrades: effort.full_rerank
                && !effort.layout_loop()
                && arbiter == crate::Arbiter::Cvvdp,
        }
    }
}

/// Coefficient-domain bounds and reconstruction-evaluation budgets.
pub(super) const FINE_RUNNER_UP_RATIO: f32 = 1.05;
pub(super) const MERGE32_SHORTLIST_RATIO: f32 = 1.03;
pub(super) const MERGE32_CANDIDATES_PER_REGION: usize = 3;
pub(super) const MEDIUM_CANDIDATES_PER_REGION: usize = 2;
