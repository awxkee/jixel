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
mod ac_context;
mod adaptive_quant;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod bit_writer;
mod color;
mod color_encoding;
mod dc_group_data;
mod dct;
mod enc_ac_strategy;
mod enc_color_correlation;
mod enc_frame;
mod enc_group;
mod enc_lossless;
mod enc_lz77_ac;
mod enc_xyb;
mod encode_image;
mod entropy;
mod err;
mod gaborish;
mod icc_codec;
mod image;
mod modular;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod quant_weights;
mod static_entropy_codes;

pub use color_encoding::{
    ColorEncoding, ColorSpace, Primaries, RenderingIntent, TransferFunction, WhitePoint,
};
pub use encode_image::{
    EncodeConfig, distance_from_quality, encode_image, encode_image_10bit, encode_image_12bit,
    encode_image_gray, encode_image_gray_alpha, encode_image_with_alpha,
    encode_image_with_alpha_10bit, encode_image_with_alpha_12bit,
};
pub use err::EncodeError;
