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

mod ans;
mod cluster;
mod dlog2;
mod entropy_code;
mod fast_div_u16;
mod histogram;
mod huffman_tree;
mod prefix_code;
mod token;
mod write;

pub(crate) use ans::{ANS_LOG_TAB_SIZE, AnsEncSymbolInfo, write_ans_tokens};
pub(crate) use cluster::{
    CLUSTERS_LIMIT, FixedClusterScratch, cluster_histograms, cluster_histograms_fixed,
};
pub(crate) use dlog2::f_log2;
pub(crate) use entropy_code::{EntropyCode, FrozenTokenPrices, OwnedEntropyCode};
pub(crate) use histogram::Histogram;
pub(crate) use huffman_tree::HuffmanNode;
pub(crate) use prefix_code::{ALPHABET_SIZE, PrefixCode};
pub(crate) use token::{
    HybridUintConfig, Token, pack_signed, uint_encode, uint_encode_with_config,
};
pub(crate) use write::{
    build_entropy_code_no_cluster, build_huffman_codes, build_huffman_codes_into,
    optimize_entropy_code, optimize_entropy_code_ac, optimize_entropy_code_ac_streams,
    optimize_entropy_code_ac_streams_fast, select_hybrid_config, write_context_map,
    write_entropy_code, write_prefix_codes, write_token,
};
