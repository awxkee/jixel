/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

//! Single-group palette encoding and palette transform serialization.

mod global;
mod local;

pub(crate) type PaletteRowFn = fn(&[i32], &[i32], &mut [[i32; 4]]);

#[allow(unreachable_code)]
pub(crate) fn selected_palette_row_fn() -> PaletteRowFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |a, b, c| unsafe { crate::avx::palette_row_avx2(a, b, c) };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        return |a, b, c| unsafe { crate::sse::palette_row_sse41(a, b, c) };
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    return |a, b, c| unsafe { crate::neon::palette_row_neon(a, b, c) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return crate::wasm::palette_row_wasm;
    palette_row_scalar
}

#[allow(dead_code)]
pub(crate) fn palette_row_scalar(above: &[i32], row: &[i32], out: &mut [[i32; 4]]) {
    for (x, out) in out.iter_mut().enumerate() {
        let (ll, l, v) = (row[x], row[x + 1], row[x + 2]);
        let (tl, t, tr) = (above[x + 1], above[x + 2], above[x + 3]);
        let sign = |v: i32| v.signum() + 1;
        let ctx = sign(l - tl) * 27 + sign(t - tl) * 9 + sign(t - tr) * 3 + sign(l - ll);
        let gradient =
            ((l as i64 + t as i64 - tl as i64).clamp(l.min(t) as i64, l.max(t) as i64)) as i32;
        *out = [ctx, v - l, v - t, v - gradient];
    }
}

#[cfg(test)]
mod tests;

pub(super) use global::{PALETTE_COARSE_MARGIN, PALETTE_FINAL_MARGIN, build_global_palette};
#[cfg(test)]
pub(super) use local::local_palette_coverage_is_sufficient;
pub(super) use local::{
    LOCAL_PALETTE_MAX_COLORS, LocalPaletteGroup, build_local_palette_group_limited,
    try_encode_local_palette_multi_group,
};

use super::GroupLayout;
use super::header::{write_frame_header_modular_kind, write_toc_entry};
use super::lz77::{
    RunLzWriter, build_lz_pixel_code, lz77_compress_for_speed, write_local_tree_lz77,
    write_lz_section,
};
use super::predictor::{GradPackInteriorFn, choose_predictor_for_rows, fixed_predictor};
use super::rct::inverse_ycocg;
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::patches::ModularFrameKind;
use global::ColorMap;

/// Serialize a single Palette transform (mirrors `Transform::VisitFields` for
/// `TransformId::kPalette`). v1: begin_c=0, num_c=3, nb_deltas=0, predictor=Zero.
pub(crate) fn write_palette_transform(num_c: u32, nb_colors: u32, w: &mut BitWriter) {
    debug_assert!((1..=4).contains(&num_c));
    // transforms count U32(Val(0),Val(1),...): selector 1 = Val(1) = 1 transform.
    w.write(2, 0b01);
    // id U32(Val(RCT=0),Val(Palette=1),...): selector 1 = Palette.
    w.write(2, 0b01);
    // begin_c U32(Bits(3),...): selector 0 = Bits(3), value 0.
    w.write(2, 0b00);
    w.write(3, 0);
    // num_c U32(Val(1),Val(3),Val(4),BitsOffset(13,1)).
    match num_c {
        1 => w.write(2, 0b00),
        3 => w.write(2, 0b01),
        4 => w.write(2, 0b10),
        _ => {
            w.write(2, 0b11);
            w.write(13, u64::from(num_c - 1));
        }
    }
    // nb_colors U32(BitsOffset(8,0), BitsOffset(10,256), BitsOffset(12,1280),
    // BitsOffset(16,5376)).
    if nb_colors <= 255 {
        w.write(2, 0b00);
        w.write(8, nb_colors as u64);
    } else if nb_colors <= 256 + 1023 {
        w.write(2, 0b01);
        w.write(10, (nb_colors - 256) as u64);
    } else if nb_colors <= 1280 + 4095 {
        w.write(2, 0b10);
        w.write(12, (nb_colors - 1280) as u64);
    } else {
        debug_assert!(nb_colors <= 5376 + 65535);
        w.write(2, 0b11);
        w.write(16, (nb_colors - 5376) as u64);
    }
    // nb_deltas U32(Val(0),...): selector 0 = Val(0) = 0.
    w.write(2, 0b00);
    // predictor Bits(4): Predictor::Zero = 0.
    w.write(4, 0);
}

/// Visit a crop in raster order, stopping as soon as its palette is rejected.
/// Row slicing checks the plane bounds once; alpha dispatch also stays outside
/// the pixel loop. Global palettes use RGB, group-local palettes keep YCoCg.
#[inline]
fn visit_palette_colors<const RGB: bool>(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    stride: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    mut visit: impl FnMut([i32; 4]) -> bool,
) -> bool {
    if w == 0 || h == 0 {
        return true;
    }
    let plane0 = linear.plane_data(0);
    let plane1 = linear.plane_data(1);
    let plane2 = linear.plane_data(2);
    for y in y0..y0 + h {
        let start = y * stride + x0;
        let row0 = &plane0[start..][..w];
        let row1 = &plane1[start..][..w];
        let row2 = &plane2[start..][..w];
        let mut colors = row0.iter().zip(row1).zip(row2).map(|((&a, &b), &c)| {
            let (a, b, c) = if RGB {
                inverse_ycocg(a, b, c)
            } else {
                (a, b, c)
            };
            [a, b, c, 0]
        });
        let complete = match alpha {
            None => colors.all(&mut visit),
            Some(AlphaPlane::U8(data)) => colors.zip(&data[start..][..w]).all(|(mut c, &a)| {
                c[3] = a as i32;
                visit(c)
            }),
            Some(AlphaPlane::U16 { data, .. }) => {
                colors.zip(&data[start..][..w]).all(|(mut c, &a)| {
                    c[3] = a as i32;
                    visit(c)
                })
            }
            Some(AlphaPlane::F32(data)) => colors.zip(&data[start..][..w]).all(|(mut c, &a)| {
                c[3] = a;
                visit(c)
            }),
        };
        if !complete {
            return false;
        }
    }
    true
}

/// Visit every (value, alpha) tuple of a gray plane as `[v, v, v, alpha]`,
/// so the luma, axis and chain orders rank gray levels by value.
fn visit_gray_colors(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    mut visit: impl FnMut([i32; 4]) -> bool,
) -> bool {
    let plane = &linear.plane_data(0)[..xsize * ysize];
    match alpha {
        None => plane.iter().all(|&v| visit([v, v, v, 0])),
        Some(AlphaPlane::U8(data)) => plane
            .iter()
            .zip(data)
            .all(|(&v, &a)| visit([v, v, v, i32::from(a)])),
        Some(AlphaPlane::U16 { data, .. }) => plane
            .iter()
            .zip(data)
            .all(|(&v, &a)| visit([v, v, v, i32::from(a)])),
        Some(AlphaPlane::F32(data)) => plane.iter().zip(data).all(|(&v, &a)| visit([v, v, v, a])),
    }
}

/// Single-group RGB/RGBA exact palette path. Encodes a palette meta-channel plus
/// an index channel, declaring a Palette transform so `InvPalette` reconstructs
/// the original channels directly (no RCT).
pub(super) fn try_encode_palette_single_group(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    min_symbol: u32,
    grad_pack_fn: GradPackInteriorFn,
    speed: crate::Speed,
    use_wp: bool,
    frame_kind: ModularFrameKind<'_>,
    layout: GroupLayout,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    let npx = xsize * ysize;
    let num_c = 3 + usize::from(alpha.is_some());

    let nb_chans = 2usize;
    let distance_ctx = nb_chans as u32;
    let (preds, mut section, lz_tokens) = {
        // 1) Reconstruct RGB from YCoCg, collect distinct tuples (bail past 256).
        let mut seen = ColorMap::<()>::with_capacity_and_hasher(
            LOCAL_PALETTE_MAX_COLORS + 1,
            Default::default(),
        );
        let mut previous = None;
        if !visit_palette_colors::<true>(linear, alpha, xsize, 0, 0, xsize, ysize, |color| {
            if previous != Some(color) {
                seen.entry(color).or_insert(());
                previous = Some(color);
            }
            seen.len() <= LOCAL_PALETTE_MAX_COLORS
        }) {
            return false;
        }
        let nb_colors = seen.len();
        if nb_colors == 0 {
            return false;
        }

        // 2) Sorted palette + color->index map.
        let mut colors: Vec<[i32; 4]> = seen.keys().copied().collect();
        colors.sort_unstable();
        let mut idx_of = ColorMap::<i32>::with_capacity_and_hasher(nb_colors, Default::default());
        for (i, c) in colors.iter().enumerate() {
            idx_of.insert(*c, i as i32);
        }

        // 3) Palette meta-channel (row c = component c of each color) + index channel.
        let mut palette_ch = vec![0i32; num_c * nb_colors];
        for (c, row) in palette_ch.chunks_exact_mut(nb_colors).enumerate() {
            for (value, color) in row.iter_mut().zip(&colors) {
                *value = color[c];
            }
        }
        let mut index_img = Vec::with_capacity(npx);
        let mut previous = ([0; 4], -1);
        visit_palette_colors::<true>(linear, alpha, xsize, 0, 0, xsize, ysize, |color| {
            if previous.1 < 0 || previous.0 != color {
                previous = (color, idx_of[&color]);
            }
            index_img.push(previous.1 as u16);
            true
        });

        let palette = LocalPaletteGroup {
            palette: palette_ch,
            indices: index_img,
            nb_colors,
            w: xsize,
            h: ysize,
        };
        // 4) Channel accessors (channel 0 = palette, channel 1 = index).
        let pget = |y: usize| &palette.palette[y * nb_colors..][..nb_colors];
        let iget = |y: usize| &palette.indices[y * xsize..][..xsize];

        // 5) Slow searches predictors per channel; Fast stays fixed Weighted.
        let preds = if speed.effort().predictor_search {
            [
                choose_predictor_for_rows(pget, nb_colors, num_c, use_wp),
                choose_predictor_for_rows(iget, xsize, ysize, use_wp),
            ]
        } else {
            [fixed_predictor(use_wp); 2]
        };

        // 6) Frame header + single section (mirrors the RGB single-group layout).
        write_frame_header_modular_kind(alpha.is_some(), frame_kind, layout, writer);

        let mut section = BitWriter::new();
        section.write(1, 1); // dc_quant all_default = 1
        section.write(1, 0); // has_tree = 0
        section.write(1, 0); // use_global_tree = 0
        section.write(1, 1); // wp_default = 1
        write_palette_transform(num_c as u32, nb_colors as u32, &mut section);

        let lz_tokens = if speed.effort().lz_deep {
            let mut tokens = Vec::with_capacity(num_c * nb_colors + npx);
            palette.tokenize(
                nb_chans,
                &preds,
                grad_pack_fn,
                &mut scratch.gradient,
                &mut tokens,
            );
            lz77_compress_for_speed(&tokens, distance_ctx, speed, scratch)
        } else {
            let mut runs = RunLzWriter::with_capacity((num_c * nb_colors + npx).min(16 * 1024));
            palette.tokenize(
                nb_chans,
                &preds,
                grad_pack_fn,
                &mut scratch.gradient,
                &mut runs,
            );
            runs.finish()
        };
        (preds, section, lz_tokens)
    };
    let code = build_lz_pixel_code(
        speed.effort().lz_exhaustive_entropy(),
        std::iter::once(lz_tokens.as_slice()),
        nb_chans,
        min_symbol,
        speed.effort().lz_refined_entropy,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    write_local_tree_lz77(
        &preds,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut section,
    );
    write_lz_section(&lz_tokens, distance_ctx, &code, min_symbol, &mut section);
    section.zero_pad_to_byte();

    writer.write(1, 0); // no permutation
    writer.zero_pad_to_byte();
    write_toc_entry(section.bits_written() / 8, writer);
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(std::slice::from_mut(&mut section));
    writer.zero_pad_to_byte();
    true
}

#[cfg(test)]
mod search_row_tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Rows {
        palette: PaletteRowFn,
    }

    fn backends() -> Vec<Rows> {
        #[allow(unused_mut)]
        let mut rows = vec![Rows {
            palette: palette_row_scalar,
        }];
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            rows.push(Rows {
                palette: |a, b, c| unsafe { crate::avx::palette_row_avx2(a, b, c) },
            });
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            rows.push(Rows {
                palette: |a, b, c| unsafe { crate::sse::palette_row_sse41(a, b, c) },
            });
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            rows.push(Rows {
                palette: |a, b, c| unsafe { crate::neon::palette_row_neon(a, b, c) },
            });
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
        {
            rows.push(Rows {
                palette: crate::wasm::palette_row_wasm,
            });
        }
        rows.push(Rows {
            palette: selected_palette_row_fn(),
        });
        rows
    }

    #[test]
    fn palette_contexts_and_residuals_match_scalar() {
        for len in 0..133 {
            for pattern in 0..6 {
                let row: Vec<i32> = (0..len + 3)
                    .map(|x| ((x * 79 + pattern * 17) % 1024) as i32)
                    .collect();
                let above: Vec<i32> = (0..len + 3)
                    .map(|x| {
                        if pattern == 0 {
                            row[x]
                        } else {
                            ((x * 13 + pattern * 327) % 1024) as i32
                        }
                    })
                    .collect();
                let mut expected = vec![[0; 4]; len];
                palette_row_scalar(&above, &row, &mut expected);
                for backend in backends() {
                    let mut actual = vec![[0; 4]; len];
                    (backend.palette)(&above, &row, &mut actual);
                    assert_eq!(actual, expected, "len={len}, pattern={pattern}");
                }
            }
        }
    }
}
