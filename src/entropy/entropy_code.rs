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

use super::prefix_code::PrefixCode;

pub struct EntropyCode<'a> {
    pub context_map: &'a [u8],
    pub num_contexts: usize,
    pub prefix_codes: &'a [PrefixCode],
    #[allow(unused)]
    pub num_prefix_codes: usize,
    /// Original (pre-cluster) context map. None for static codes.
    pub orig_context_map: Option<&'a [u8]>,
    pub orig_num_contexts: usize,
}

impl<'a> EntropyCode<'a> {
    /// Static entropy code: no clustering was applied.
    pub(crate) fn new(context_map: &'a [u8], prefix_codes: &'a [PrefixCode]) -> Self {
        Self {
            context_map,
            num_contexts: context_map.len(),
            prefix_codes,
            num_prefix_codes: prefix_codes.len(),
            orig_context_map: None,
            orig_num_contexts: 0,
        }
    }
}

/// Owned entropy code: heap-allocated context_map and prefix_codes,
/// suitable for runtime-built codes.
pub struct OwnedEntropyCode {
    pub context_map: Vec<u8>,
    pub prefix_codes: Vec<PrefixCode>,
    /// Pre-cluster context map, if clustering was applied.
    pub orig_context_map: Option<Vec<u8>>,
    pub orig_num_contexts: usize,
}

impl OwnedEntropyCode {
    pub fn as_ref(&self) -> EntropyCode<'_> {
        EntropyCode {
            context_map: &self.context_map,
            num_contexts: self.context_map.len(),
            prefix_codes: &self.prefix_codes,
            num_prefix_codes: self.prefix_codes.len(),
            orig_context_map: self.orig_context_map.as_deref(),
            orig_num_contexts: self.orig_num_contexts,
        }
    }
}
