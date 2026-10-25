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
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::approx_constant
)]
#![deny(unreachable_pub)]
mod ac_context;
mod ac_strategy;
mod adaptive_quant;
mod afv;
mod arbiter;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod bit_writer;
mod cell_layer;
mod clip_continuation;
mod cmyk;
mod coder_scratch;
mod coding;
#[cfg(feature = "extraslow")]
mod coef_polish;
mod coeff_order;
mod color;
mod color_correlation;
mod color_encoding;
#[cfg(feature = "extraslow")]
mod cvvdp_arbiter;
mod dc_group_data;
mod dc_ops;
mod dc_tree;
mod dct;
mod dct128;
mod effort;
mod encode_image;
mod encoding_context;
mod entropy;
mod epf_filter;
mod epf_proxy;
mod fast_loseless;
mod frame;
mod gaborish;
mod gain_map;
mod group;
mod haar;
mod icc_codec;
mod image;
mod inflated_cost;
mod ink_layer;
mod invisible;
mod iso_gain_map;
mod jpeg;
mod lattice_patches;
mod lossless;
mod lz77_ac;
mod lz_match;
mod ma_tree;
mod modular;
mod mosaic_seam;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod noise;
mod orientation;
mod patches;
mod path_order;
mod quant_weights;
mod region_frames;
#[cfg(feature = "splines")]
mod splines;
mod squeeze;
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
mod sse;
#[cfg(feature = "extraslow")]
mod ssimulacra2;
mod static_entropy_codes;
mod thread_pool;
mod upsampling;
mod util;
mod vector_layers;
#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
mod wasm;
mod weighted_predictor;
mod xyb;
mod yellow_opsin;

pub use cmyk::{InkConvention, encode_image_cmyk, encode_image_cmyk_16bit};
pub use coding::CodingTransform;
pub use color_encoding::{
    ColorEncoding, ColorSpace, Primaries, RenderingIntent, TransferFunction, WhitePoint,
};
pub use encode_image::{
    Arbiter, DecodingSpeed, EncodeConfig, GroupOrder, LossyModular, Speed, distance_from_quality,
    encode_image, encode_image_10bit, encode_image_12bit, encode_image_16bit, encode_image_f16,
    encode_image_f32, encode_image_gray, encode_image_gray_10bit, encode_image_gray_12bit,
    encode_image_gray_16bit, encode_image_gray_alpha, encode_image_gray_alpha_10bit,
    encode_image_gray_alpha_12bit, encode_image_gray_alpha_16bit, encode_image_gray_f16,
    encode_image_gray_f32, encode_image_with_alpha, encode_image_with_alpha_10bit,
    encode_image_with_alpha_12bit, encode_image_with_alpha_16bit, encode_image_with_alpha_f16,
    encode_image_with_alpha_f32,
};
pub use fast_loseless::{FlMeta, encode_fast_lossless, encode_fast_lossless_u16};
pub use gain_map::{GainMap, GainMapPixels};
pub use iso_gain_map::{GainMapFloats, IsoGainMap};
pub use jpeg::{
    BrotliCompression, JpegTranscodeConfig, encode_jpeg_lossless, encode_jpeg_lossless_with_config,
};
pub use noise::Grain;
pub use orientation::Orientation;
pub use util::EncodeError;
