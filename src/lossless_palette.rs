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

//! Global and group-local palette construction, selection, and encoding.

use super::lz77::{
    LzToken, RunLzWriter, build_lz_pixel_code, lz77_compress_channels_for_speed_with_depth,
    lz77_compress_for_speed, lz77_compress_for_speed_with_depth, write_local_tree_lz77,
    write_lz_section,
};
use super::predictor::{
    GradPackInteriorFn, GradientScratch, SqueezePredictorCost, TokenSink, channel_to_context,
    choose_predictor_for_rows, fixed_predictor, tokenize_channels_with_wp, tokenize_plane_rows,
    tokenize_runs_with_wp, tokenize_sample_rows,
};
use super::rct::{inverse_ycocg, rct_gradient_row};
use super::{
    DeepLzScratchPool, GROUP_DIM, GroupLayout, MaChannel, MaPixels, MaSource, MaTransform,
    SLOW_DEEP_LZ_MAX_THREADS, entropy_of_hist, group_lz_threads, write_frame_header_modular,
    write_lz_groups_with_header, write_modular_transforms, write_toc_entry,
};
use crate::bit_writer::BitWriter;
use crate::coder_scratch::CoderScratch;
use crate::encode_image::AlphaPlane;
use crate::image::Image3Si;
use crate::thread_pool::ThreadPool;
use crate::weighted_predictor::WpParams;

/// A sparse set of local-palette groups forces Slow mode to build a complete
/// alternate frame while most groups lose the learned-tree model. Require a
/// useful frame footprint before paying for that exact candidate.
const LOCAL_PALETTE_MIN_COVERAGE_NUM: usize = 1;
const LOCAL_PALETTE_MIN_COVERAGE_DEN: usize = 4;

#[inline]
pub(super) fn local_palette_coverage_is_sufficient(
    palette_pixels: usize,
    total_pixels: usize,
) -> bool {
    palette_pixels.saturating_mul(LOCAL_PALETTE_MIN_COVERAGE_DEN)
        >= total_pixels.saturating_mul(LOCAL_PALETTE_MIN_COVERAGE_NUM)
}

/// Serialize a single Palette transform (mirrors `Transform::VisitFields` for
/// `TransformId::kPalette`). v1: begin_c=0, num_c=3, nb_deltas=0, predictor=Zero.
pub(super) fn write_palette_transform(num_c: u32, nb_colors: u32, w: &mut BitWriter) {
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
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    let npx = xsize * ysize;
    let num_c = 3 + usize::from(alpha.is_some());

    let nb_chans = 2usize;
    let distance_ctx = nb_chans as u32;
    let (preds, mut section, lz_tokens) = {
        // 1) Reconstruct RGB from YCoCg, collect distinct tuples (bail past 256).
        let mut seen = ColorMap::<()>::with_capacity_and_hasher(257, Default::default());
        let mut previous = None;
        if !visit_palette_colors::<true>(linear, alpha, xsize, 0, 0, xsize, ysize, |color| {
            if previous != Some(color) {
                seen.entry(color).or_insert(());
                previous = Some(color);
            }
            seen.len() <= 256
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
        for (i, color) in colors.iter().enumerate() {
            for c in 0..num_c {
                palette_ch[c * nb_colors + i] = color[c];
            }
        }
        let mut index_img = Vec::with_capacity(npx);
        let mut previous = ([0; 4], -1);
        visit_palette_colors::<true>(linear, alpha, xsize, 0, 0, xsize, ysize, |color| {
            if previous.1 < 0 || previous.0 != color {
                previous = (color, idx_of[&color]);
            }
            index_img.push(previous.1 as u8);
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
        let preds = if speed == crate::Speed::Slow {
            [
                choose_predictor_for_rows(pget, nb_colors, num_c, use_wp),
                choose_predictor_for_rows(iget, xsize, ysize, use_wp),
            ]
        } else {
            [fixed_predictor(use_wp); 2]
        };

        // 6) Frame header + single section (mirrors the RGB single-group layout).
        write_frame_header_modular(alpha.is_some(), GroupLayout::DEFAULT, writer);

        let mut section = BitWriter::new();
        section.write(1, 1); // dc_quant all_default = 1
        section.write(1, 0); // has_tree = 0
        section.write(1, 0); // use_global_tree = 0
        section.write(1, 1); // wp_default = 1
        write_palette_transform(num_c as u32, nb_colors as u32, &mut section);

        let lz_tokens = if speed == crate::Speed::Slow {
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
        std::iter::once(lz_tokens.as_slice()),
        nb_chans,
        min_symbol,
        speed == crate::Speed::Slow,
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

struct LocalPaletteGroup {
    palette: Vec<i32>,
    indices: Vec<u8>,
    nb_colors: usize,
    w: usize,
    h: usize,
}

impl LocalPaletteGroup {
    fn tokenize(
        &self,
        num_contexts: usize,
        predictors: &[u32],
        grad_pack_fn: GradPackInteriorFn,
        scratch: &mut GradientScratch,
        out: &mut impl TokenSink,
    ) {
        tokenize_plane_rows(
            channel_to_context(0, num_contexts),
            |y| &self.palette[y * self.nb_colors..][..self.nb_colors],
            self.nb_colors,
            self.palette.len() / self.nb_colors,
            predictors[0],
            grad_pack_fn,
            scratch,
            out,
        );
        tokenize_sample_rows(
            channel_to_context(1, num_contexts),
            |y| &self.indices[y * self.w..][..self.w],
            self.w,
            self.h,
            predictors[1],
            grad_pack_fn,
            scratch,
            out,
        );
    }
}

fn add_alpha_predictor_cost(
    cost: &mut SqueezePredictorCost,
    alpha: &AlphaPlane,
    stride: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    use_wp: bool,
) {
    match alpha {
        AlphaPlane::U8(data) => {
            cost.add_rows(|y| &data[(y0 + y) * stride + x0..][..w], w, h, use_wp)
        }
        AlphaPlane::U16 { data, .. } => {
            cost.add_rows(|y| &data[(y0 + y) * stride + x0..][..w], w, h, use_wp)
        }
        AlphaPlane::F32(data) => {
            cost.add_rows(|y| &data[(y0 + y) * stride + x0..][..w], w, h, use_wp)
        }
    }
}

fn estimated_local_lz_stream_bits(
    lz: &[LzToken],
    predictors: &[u32],
    num_contexts: usize,
    min_symbol: u32,
    speed: crate::Speed,
    scratch: &mut CoderScratch,
) -> usize {
    let code = build_lz_pixel_code(
        std::iter::once(lz),
        num_contexts,
        min_symbol,
        speed == crate::Speed::Slow,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );
    let mut writer = BitWriter::new();
    write_local_tree_lz77(
        predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut writer,
    );
    write_lz_section(lz, num_contexts as u32, &code, min_symbol, &mut writer);
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
    use_wp: bool,
    scratch: &mut CoderScratch,
) -> (bool, SqueezePredictorCost) {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let mut index_cost = SqueezePredictorCost::default();
    let palette_predictors = if speed == crate::Speed::Slow {
        [
            choose_predictor_for_rows(
                |y| &palette.palette[y * palette.nb_colors..][..palette.nb_colors],
                palette.nb_colors,
                nb_chans,
                use_wp,
            ),
            {
                index_cost.add_rows(
                    |y| &palette.indices[y * palette.w..][..palette.w],
                    palette.w,
                    palette.h,
                    use_wp,
                );
                index_cost.predictor(use_wp)
            },
        ]
    } else {
        [fixed_predictor(use_wp); 2]
    };
    let mut palette_tokens = RunLzWriter::with_capacity(
        (nb_chans * palette.nb_colors + palette.w * palette.h).min(16 * 1024),
    );
    palette.tokenize(
        2,
        &palette_predictors,
        grad_pack_fn,
        &mut scratch.gradient,
        &mut palette_tokens,
    );
    let mut palette_transform = BitWriter::new();
    write_palette_transform(
        nb_chans as u32,
        palette.nb_colors as u32,
        &mut palette_transform,
    );
    let palette_bits = {
        let palette_lz = palette_tokens.finish();
        estimated_local_lz_stream_bits(
            &palette_lz,
            &palette_predictors,
            2,
            min_symbol,
            speed,
            scratch,
        ) + palette_transform.bits_written()
    };

    let plain_predictors: Vec<u32> = if speed == crate::Speed::Slow {
        (0..nb_chans)
            .map(|chan| {
                if chan < 3 {
                    let plane = linear.plane_data(chan);
                    choose_predictor_for_rows(
                        |y| &plane[(y0 + y) * xsize + x0..][..palette.w],
                        palette.w,
                        palette.h,
                        use_wp,
                    )
                } else {
                    let alpha = alpha.expect("alpha channel must exist");
                    let mut cost = SqueezePredictorCost::default();
                    add_alpha_predictor_cost(
                        &mut cost, alpha, xsize, x0, y0, palette.w, palette.h, use_wp,
                    );
                    cost.predictor(use_wp)
                }
            })
            .collect()
    } else {
        vec![fixed_predictor(use_wp); nb_chans]
    };
    let plain_lz = tokenize_runs_with_wp(
        linear,
        alpha,
        xsize,
        x0,
        y0,
        palette.w,
        palette.h,
        3,
        &plain_predictors,
        grad_pack_fn,
        scratch,
        WpParams::DEFAULT,
    );
    let plain_bits = estimated_local_lz_stream_bits(
        &plain_lz,
        &plain_predictors,
        nb_chans,
        min_symbol,
        speed,
        scratch,
    ) + 2; // zero-transform count

    (palette_bits < plain_bits, index_cost)
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
    let num_c = 3 + usize::from(alpha.is_some());
    let num_pixels = w * h;
    // Most photographic groups exceed the palette limit almost immediately.
    // Probe only the distinct set first, avoiding a full-group tuple allocation
    // on that overwhelmingly common rejection path.
    // Palette indices come from the sorted colors below, independently of
    // hash-table iteration order. Reuse the global palette's cheap hasher.
    let mut seen = ColorMap::<()>::with_capacity_and_hasher(257, Default::default());
    let mut previous = None;
    if !visit_palette_colors::<false>(linear, alpha, xsize, x0, y0, w, h, |color| {
        if previous != Some(color) {
            seen.entry(color).or_insert(());
            previous = Some(color);
        }
        seen.len() <= 256
    }) {
        return None;
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
    let mut index_of = ColorMap::<i32>::with_capacity_and_hasher(nb_colors, Default::default());
    let mut palette = vec![0i32; num_c * nb_colors];
    for (index, color) in colors.iter().enumerate() {
        index_of.insert(*color, index as i32);
        for c in 0..num_c {
            palette[c * nb_colors + index] = color[c];
        }
    }
    let mut indices = Vec::with_capacity(num_pixels);
    let mut previous = ([0; 4], -1);
    visit_palette_colors::<false>(linear, alpha, xsize, x0, y0, w, h, |color| {
        if previous.1 < 0 || previous.0 != color {
            previous = (color, index_of[&color]);
        }
        indices.push(previous.1 as u8);
        true
    });

    Some(LocalPaletteGroup {
        palette,
        indices,
        nb_colors,
        w,
        h,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_encode_local_palette_multi_group(
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
    use_wp: bool,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
    writer: &mut BitWriter,
) -> bool {
    let nb_chans = 3 + usize::from(alpha.is_some());
    let num_ac_groups = xsize_groups * ysize_groups;
    // Index samples are bytes, so these residual histograms stay small.
    // Merge only accepted palettes while their group costs are still available;
    // do not retain one set of histograms per group or serialize group scoring.
    let pooled_index_cost = std::sync::Mutex::new(SqueezePredictorCost::default());
    let palettes = if speed == crate::Speed::Slow {
        let mut palettes = pool.steal_map(scratch, num_ac_groups, |group_index, _scratch| {
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let x0 = gx * GROUP_DIM;
            let y0 = gy * GROUP_DIM;
            let w = GROUP_DIM.min(xsize - x0);
            let h = GROUP_DIM.min(ysize - y0);
            build_local_palette_group(linear, alpha, xsize, x0, y0, w, h)
        });
        let palette_pixels: usize = palettes
            .iter()
            .flatten()
            .map(|palette| palette.w * palette.h)
            .sum();
        if !local_palette_coverage_is_sufficient(palette_pixels, xsize * ysize) {
            return false;
        }

        let palette_is_better = pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
            let Some(palette) = &palettes[group_index] else {
                return false;
            };
            let gx = group_index % xsize_groups;
            let gy = group_index / xsize_groups;
            let (keep, index_cost) = local_palette_is_better(
                palette,
                linear,
                alpha,
                xsize,
                gx * GROUP_DIM,
                gy * GROUP_DIM,
                min_symbol,
                grad_pack_fn,
                speed,
                use_wp,
                scratch,
            );
            if keep {
                pooled_index_cost
                    .lock()
                    .unwrap()
                    .costs
                    .merge(&index_cost.costs);
            }
            keep
        });
        for (palette, keep) in palettes.iter_mut().zip(palette_is_better) {
            if !keep {
                *palette = None;
            }
        }
        palettes
    } else {
        pool.steal_map(scratch, num_ac_groups, |group_index, scratch| {
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
                use_wp,
                scratch,
            )
            .0
            .then_some(palette)
        })
    };
    if !palettes.iter().any(Option::is_some) {
        return false;
    }

    // The global MA tree sees group-local channel slots. Pool predictor costs
    // for palette/index channels and ordinary YCoCg(A) channels by those slots.
    let predictors: Vec<u32> = if speed == crate::Speed::Slow {
        // Split the remaining palette-meta and plain-channel work by group.
        // Each crop still resets WP; integer histogram merging leaves the
        // final costs and predictor tie-breaking unchanged.
        let tasks = pool
            .num_threads()
            .min(SLOW_DEEP_LZ_MAX_THREADS)
            .min(num_ac_groups)
            .max(1);
        let groups_per_task = num_ac_groups.div_ceil(tasks);
        let parts = pool.steal_map(scratch, tasks, |task, _scratch| {
            let mut costs: [SqueezePredictorCost; 4] =
                std::array::from_fn(|_| SqueezePredictorCost::default());
            let begin = task * groups_per_task;
            let end = (begin + groups_per_task).min(num_ac_groups);
            for group_index in begin..end {
                let palette = &palettes[group_index];
                for (slot, cost) in costs[..nb_chans].iter_mut().enumerate() {
                    if let Some(palette) = palette {
                        if slot == 0 {
                            cost.add_rows(
                                |y| &palette.palette[y * palette.nb_colors..][..palette.nb_colors],
                                palette.nb_colors,
                                nb_chans,
                                use_wp,
                            );
                        }
                        // Slot 1 already includes accepted palette indices.
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
                        cost.add_rows(|y| &plane[(y0 + y) * xsize + x0..][..w], w, h, use_wp);
                    } else {
                        let alpha = alpha.expect("alpha slot requires alpha channel");
                        add_alpha_predictor_cost(cost, alpha, xsize, x0, y0, w, h, use_wp);
                    }
                }
            }
            costs
        });
        let mut costs: [SqueezePredictorCost; 4] =
            std::array::from_fn(|_| SqueezePredictorCost::default());
        costs[1] = pooled_index_cost.into_inner().unwrap();
        for part in parts {
            for (cost, part) in costs.iter_mut().zip(part) {
                cost.costs.merge(&part.costs);
            }
        }
        costs[..nb_chans]
            .iter()
            .map(|cost| cost.predictor(use_wp))
            .collect()
    } else {
        vec![fixed_predictor(use_wp); nb_chans]
    };

    let distance_ctx = nb_chans as u32;
    let group_lz_tokens: Vec<Vec<LzToken>> = {
        let deep_lz = (speed == crate::Speed::Slow)
            .then(|| DeepLzScratchPool::new(group_lz_threads(speed, pool)));
        pool.steal_map_with_threads(
            scratch,
            num_ac_groups,
            group_lz_threads(speed, pool),
            |group_index, scratch| {
                if let Some(palette) = &palettes[group_index] {
                    let capacity = nb_chans * palette.nb_colors + palette.w * palette.h;
                    if let Some(deep_lz) = &deep_lz {
                        let mut tokens = Vec::with_capacity(capacity);
                        palette.tokenize(
                            nb_chans,
                            &predictors,
                            grad_pack_fn,
                            &mut scratch.gradient,
                            &mut tokens,
                        );
                        deep_lz.with_depth(|depth| {
                            lz77_compress_for_speed_with_depth(
                                &tokens,
                                distance_ctx,
                                speed,
                                depth,
                                None,
                                scratch,
                            )
                        })
                    } else {
                        let mut runs = RunLzWriter::with_capacity(capacity.min(16 * 1024));
                        palette.tokenize(
                            nb_chans,
                            &predictors,
                            grad_pack_fn,
                            &mut scratch.gradient,
                            &mut runs,
                        );
                        runs.finish()
                    }
                } else {
                    let gx = group_index % xsize_groups;
                    let gy = group_index / xsize_groups;
                    let x0 = gx * GROUP_DIM;
                    let y0 = gy * GROUP_DIM;
                    let w = GROUP_DIM.min(xsize - x0);
                    let h = GROUP_DIM.min(ysize - y0);
                    if speed == crate::Speed::Slow {
                        let channel_tokens = tokenize_channels_with_wp(
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
                            pool,
                            scratch,
                            WpParams::DEFAULT,
                        );
                        deep_lz.as_ref().unwrap().with_depth(|depth| {
                            lz77_compress_channels_for_speed_with_depth(
                                channel_tokens,
                                distance_ctx,
                                speed,
                                depth,
                                scratch,
                            )
                        })
                    } else {
                        tokenize_runs_with_wp(
                            linear,
                            alpha,
                            xsize,
                            x0,
                            y0,
                            w,
                            h,
                            3,
                            &predictors,
                            grad_pack_fn,
                            scratch,
                            WpParams::DEFAULT,
                        )
                    }
                }
            },
        )
    };
    // Only transform metadata is needed after tokenization. Release palette
    // planes before allocating entropy tables and serialized group sections.
    let palette_sizes: Vec<_> = palettes
        .into_iter()
        .map(|palette| palette.map(|palette| palette.nb_colors))
        .collect();

    let code = build_lz_pixel_code(
        group_lz_tokens.iter().map(Vec::as_slice),
        nb_chans,
        min_symbol,
        speed == crate::Speed::Slow,
        &mut scratch.lz_entropy,
        &mut scratch.huffman_pool,
    );

    write_frame_header_modular(alpha.is_some(), GroupLayout::DEFAULT, writer);
    let num_sections = 1 + num_dc_groups + 1 + num_ac_groups;
    let mut sections: Vec<BitWriter> = (0..num_sections).map(|_| BitWriter::new()).collect();

    sections[0].write(1, 1); // dc_quant all_default
    sections[0].write(1, 1); // has global tree
    write_local_tree_lz77(
        &predictors,
        &code,
        min_symbol,
        &mut scratch.huffman_pool,
        &mut sections[0],
    );
    sections[0].write(1, 1); // use_global_tree
    sections[0].write(1, 1); // wp_default
    write_modular_transforms(nb_chans, 6, &mut sections[0]);
    sections[0].zero_pad_to_byte();

    for section in sections[1..num_dc_groups + 1].iter_mut() {
        section.write(1, 1);
        section.write(1, 1);
        section.write(2, 0);
        section.zero_pad_to_byte();
    }

    let ac_global_idx = 1 + num_dc_groups;
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].write(1, 1);
    sections[ac_global_idx].zero_pad_to_byte();

    write_lz_groups_with_header(
        &group_lz_tokens,
        &code,
        distance_ctx,
        min_symbol,
        pool,
        &mut sections[2 + num_dc_groups..],
        |group_index, section| {
            section.write(1, 1); // use_global_tree
            section.write(1, 1); // wp_default
            if let Some(nb_colors) = palette_sizes[group_index] {
                write_palette_transform(nb_chans as u32, nb_colors as u32, section);
            } else {
                section.write(2, 0); // no local transforms
            }
        },
    );

    writer.write(1, 0); // no TOC permutation
    writer.zero_pad_to_byte();
    for section in &sections {
        write_toc_entry(section.bits_written() / 8, writer);
    }
    writer.zero_pad_to_byte();
    writer.append_byte_aligned(&mut sections);
    writer.zero_pad_to_byte();
    true
}

/// Largest global palette the learned-tree path tries (libjxl's default).
const GLOBAL_PALETTE_MAX_COLORS: usize = 4096;
const _: () = assert!(GLOBAL_PALETTE_MAX_COLORS <= u16::MAX as usize + 1);
/// The palette candidate is only deepened and written when its first-stage
/// estimate is within this factor of the RGB candidate's.
pub(super) const PALETTE_COARSE_MARGIN: f64 = 1.10;
/// The palette frame is only written when its final learned estimate is
/// within this factor of the RGB frame's (the estimate is ~0.5% noisy).
pub(super) const PALETTE_FINAL_MARGIN: f64 = 1.03;

/// A global palette: RGB(A) entries in the index order that scored best in
/// `build_global_palette` (luma, PCA, a color-space chain or the spatial
/// co-occurrence chain — index neighbors must be spatial neighbors for the
/// predictors and the tree to exploit them) and one index per pixel.
pub(super) struct GlobalPalette {
    palette: Vec<i32>,
    // At most 1024 colors: retain two bytes per pixel instead of four.
    indices: Vec<u16>,
    nb_colors: usize,
    num_c: usize,
}

impl GlobalPalette {
    pub(super) fn source(&self, xsize: usize, ysize: usize) -> MaSource<'_> {
        let nb_colors = self.nb_colors;
        MaSource {
            channels: vec![
                MaChannel {
                    w: nb_colors,
                    h: self.num_c,
                    meta: true,
                    pixels: MaPixels::I32(&self.palette),
                },
                MaChannel {
                    w: xsize,
                    h: ysize,
                    meta: false,
                    pixels: MaPixels::U16(&self.indices),
                },
            ],
            transform: MaTransform::Palette {
                num_c: self.num_c as u32,
                nb_colors: nb_colors as u32,
            },
        }
    }
}

fn luma_key(c: &[i32; 4]) -> i64 {
    299 * c[0] as i64 + 587 * c[1] as i64 + 114 * c[2] as i64
}

/// Candidate index orders for a palette, each as `order[canonical] = index`
/// over the luma-sorted canonical list: luma itself, the spatial
/// co-occurrence chain, the principal axis of the color cloud, and a greedy
/// nearest-neighbor chain from the darkest color. Gradient-style predictors on the index image only work when
/// spatially adjacent colors get adjacent indices, and which order does
/// that is content-dependent.
fn palette_order_candidates(colors: &[[i32; 4]], cooc: &[u32]) -> Vec<Vec<i32>> {
    let n = colors.len();
    let identity: Vec<i32> = (0..n as i32).collect();
    let mut orders = vec![identity];
    if n < 3 {
        return orders;
    }
    orders.push(cooccurrence_chain(n, cooc));
    // Single precision is ample for the axis (the color cloud spans 16
    // bits; the projection only ranks colors, ties broken by index) and the
    // chain distances are exact integers.
    let comp = |c: &[i32; 4]| [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32];
    let mean = colors.iter().fold([0.0f32; 4], |mut m, c| {
        for (m, v) in m.iter_mut().zip(comp(c)) {
            *m += v / n as f32;
        }
        m
    });
    // Principal axis by power iteration on the 4x4 covariance.
    let mut cov = [[0.0f32; 4]; 4];
    for c in colors {
        let d: [f32; 4] = std::array::from_fn(|i| comp(c)[i] - mean[i]);
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }
    let mut axis = [0.5f32, 0.5, 0.5, 0.1];
    for _ in 0..32 {
        let mut next = [0.0f32; 4];
        for (i, row) in cov.iter().enumerate() {
            next[i] = row.iter().zip(axis).map(|(c, a)| c * a).sum();
        }
        let norm = next.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm < 1e-9 {
            break;
        }
        axis = next.map(|v| v / norm);
    }
    let projection: Vec<f32> = colors
        .iter()
        .map(|c| comp(c).iter().zip(axis).map(|(v, x)| v * x).sum())
        .collect();
    let mut by_axis: Vec<usize> = (0..n).collect();
    by_axis.sort_by(|&a, &b| projection[a].total_cmp(&projection[b]).then(a.cmp(&b)));
    let mut pca = vec![0i32; n];
    for (index, &canonical) in by_axis.iter().enumerate() {
        pca[canonical] = index as i32;
    }
    orders.push(pca);

    // Greedy nearest-neighbor chain (O(n^2), n <= 1024) on exact squared
    // distances.
    let mut used = vec![false; n];
    let mut chain = vec![0i32; n];
    let mut current = 0usize; // darkest color in canonical (luma) order
    used[0] = true;
    for index in 1..n {
        let cur = colors[current];
        let mut best = usize::MAX;
        let mut best_d = i64::MAX;
        for (k, &u) in used.iter().enumerate() {
            if u {
                continue;
            }
            let d: i64 = colors[k]
                .iter()
                .zip(cur)
                .map(|(&v, c)| {
                    let d = (v - c) as i64;
                    d * d
                })
                .sum();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        used[best] = true;
        chain[best] = index as i32;
        current = best;
    }
    orders.push(chain);
    orders
}

/// Counts of horizontally and vertically adjacent distinct color pairs, as a
/// symmetric dense `n * n` matrix over canonical indices (n <= 1024).
fn index_cooccurrence(indices: &[u16], xsize: usize, ysize: usize, n: usize) -> Vec<u32> {
    let mut cooc = vec![0u32; n * n];
    let mut bump = |a: u16, b: u16| {
        if a != b {
            cooc[a as usize * n + b as usize] += 1;
            cooc[b as usize * n + a as usize] += 1;
        }
    };
    for y in 0..ysize {
        let row = &indices[y * xsize..(y + 1) * xsize];
        for pair in row.array_windows::<2>() {
            bump(pair[0], pair[1]);
        }
        if y > 0 {
            let above = &indices[(y - 1) * xsize..y * xsize];
            for (&a, &b) in above.iter().zip(row) {
                bump(a, b);
            }
        }
    }
    cooc
}

/// Order colors by spatial adjacency rather than by color: grow a chain from
/// the most-connected color, at each step attaching the unused color with the
/// most co-occurrences to whichever chain end it touches more.
fn cooccurrence_chain(n: usize, cooc: &[u32]) -> Vec<i32> {
    let total: Vec<u64> = (0..n)
        .map(|i| cooc[i * n..(i + 1) * n].iter().map(|&v| v as u64).sum())
        .collect();
    let most_connected = |used: &[bool]| {
        (0..n)
            .filter(|&i| !used[i])
            .max_by_key(|&i| (total[i], std::cmp::Reverse(i)))
            .expect("an unused color remains")
    };
    let mut used = vec![false; n];
    let start = most_connected(&used);
    used[start] = true;
    let mut chain = std::collections::VecDeque::with_capacity(n);
    chain.push_back(start);
    let (mut left, mut right) = (start, start);
    let best_from = |end: usize, used: &[bool]| {
        let row = &cooc[end * n..(end + 1) * n];
        (0..n)
            .filter(|&i| !used[i])
            .map(|i| (row[i], i))
            .max_by_key(|&(v, i)| (v, std::cmp::Reverse(i)))
            .unwrap_or((0, usize::MAX))
    };
    for _ in 1..n {
        let (lv, li) = best_from(left, &used);
        let (rv, ri) = best_from(right, &used);
        if lv >= rv && lv > 0 {
            chain.push_front(li);
            used[li] = true;
            left = li;
        } else if rv > 0 {
            chain.push_back(ri);
            used[ri] = true;
            right = ri;
        } else {
            // Disconnected component: continue from its most-connected color.
            let next = most_connected(&used);
            chain.push_back(next);
            used[next] = true;
            right = next;
        }
    }
    let mut order = vec![0i32; n];
    for (index, &canonical) in chain.iter().enumerate() {
        order[canonical] = index as i32;
    }
    order
}

/// Gradient-predictor residual entropy of the index image under a candidate
/// order, over the same activity contexts as the RCT estimator (rows are
/// scored in parallel; the histogram sums make the split irrelevant).
fn estimate_index_plane_cost(
    indices: &[u16],
    xsize: usize,
    ysize: usize,
    order: &[i32],
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> f32 {
    const NCTX: usize = 18;
    const ALPHA: usize = 64;
    let threads = pool.num_threads().min(ysize.max(1));
    let rows_per = ysize.div_ceil(threads).max(1);
    let parts = pool.steal_map(scratch, ysize.div_ceil(rows_per), |part, scratch| {
        let y0 = part * rows_per;
        index_plane_cost_rows(
            indices,
            xsize,
            y0,
            (y0 + rows_per).min(ysize),
            order,
            &mut scratch.gradient,
        )
    });
    let mut hist = [0u64; NCTX * ALPHA];
    let mut extra_bits = 0u64;
    for (h, e) in parts {
        for (a, b) in hist.iter_mut().zip(h) {
            *a += b;
        }
        extra_bits += e;
    }
    let mut cost = extra_bits as f32;
    for h in hist.as_chunks::<ALPHA>().0 {
        let total: u64 = h.iter().sum();
        cost += entropy_of_hist(h, total);
    }
    cost
}

fn index_plane_cost_rows(
    indices: &[u16],
    xsize: usize,
    y_begin: usize,
    y_end: usize,
    order: &[i32],
    scratch: &mut GradientScratch,
) -> ([u64; 18 * 64], u64) {
    const NCTX: usize = 18;
    const ALPHA: usize = 64;
    let mut hist = [0u64; NCTX * ALPHA];
    let mut extra_bits = 0u64;
    if xsize == 0 || y_begin == y_end {
        return (hist, extra_bits);
    }
    scratch.cur.resize(xsize, 0);
    scratch.prev.resize(xsize, 0);
    let mut current = scratch.cur.as_mut_slice();
    let mut north = scratch.prev.as_mut_slice();
    let remap = |y: usize, out: &mut [i32]| {
        let row = &indices[y * xsize..][..xsize];
        for (out, &index) in out.iter_mut().zip(row) {
            *out = order[index as usize];
        }
    };
    // Each parallel band starts with the preceding image row, not a crop edge.
    if y_begin > 0 {
        remap(y_begin - 1, north);
    }
    for y in y_begin..y_end {
        remap(y, current);
        extra_bits += rct_gradient_row(current, north, y == 0, &mut hist);
        std::mem::swap(&mut current, &mut north);
    }
    (hist, extra_bits)
}

/// Multiplicative hasher for packed color keys (the default SipHash costs
/// more than the rest of the palette scan).
#[derive(Default)]
struct ColorHasher(u64);

impl std::hash::Hasher for ColorHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut v = [0u8; 8];
            v[..chunk.len()].copy_from_slice(chunk);
            self.0 = (self.0 ^ u64::from_le_bytes(v))
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(29);
        }
    }
}

type ColorMap<V> =
    std::collections::HashMap<[i32; 4], V, std::hash::BuildHasherDefault<ColorHasher>>;

pub(super) fn build_global_palette(
    linear: &Image3Si,
    alpha: Option<&AlphaPlane>,
    xsize: usize,
    ysize: usize,
    pool: &ThreadPool,
    scratch: &mut CoderScratch,
) -> Option<GlobalPalette> {
    let num_c = 3 + usize::from(alpha.is_some());
    let npx = xsize * ysize;
    let (colors, canonical_of_id, mut indices) = {
        let (first_seen, indices) = {
            // One pass: first-seen ids per pixel, remapped to the final order below.
            let mut seen: ColorMap<i32> = ColorMap::default();
            seen.reserve(GLOBAL_PALETTE_MAX_COLORS + 1);
            let mut first_seen: Vec<[i32; 4]> = Vec::with_capacity(GLOBAL_PALETTE_MAX_COLORS + 1);
            let mut indices: Vec<u16> = Vec::with_capacity(npx);
            let mut previous = ([0; 4], -1);
            // YCoCg is reversible, so distinct input tuples identify exactly the same
            // colors. Reconstruct RGB only when adding a new palette entry.
            if !visit_palette_colors::<false>(linear, alpha, xsize, 0, 0, xsize, ysize, |color| {
                if previous.1 < 0 || previous.0 != color {
                    let next = first_seen.len() as i32;
                    let id = *seen.entry(color).or_insert_with(|| {
                        let (r, g, b) = inverse_ycocg(color[0], color[1], color[2]);
                        first_seen.push([r, g, b, color[3]]);
                        next
                    });
                    if first_seen.len() > GLOBAL_PALETTE_MAX_COLORS {
                        return false;
                    }
                    previous = (color, id);
                }
                indices.push(previous.1 as u16);
                true
            }) {
                return None;
            }
            let nb_colors = first_seen.len();
            if nb_colors == 0 || npx + num_c * nb_colors >= num_c * npx {
                return None;
            }
            (first_seen, indices)
        };
        let nb_colors = first_seen.len();
        // Canonical order (luma) gives one index image; every other candidate
        // order is a permutation of it, scored in place.
        let mut canonical_ids: Vec<usize> = (0..nb_colors).collect();
        canonical_ids.sort_unstable_by_key(|&id| {
            let c = &first_seen[id];
            (c[3], luma_key(c), c[0], c[1], c[2])
        });
        let colors: Vec<[i32; 4]> = canonical_ids.iter().map(|&id| first_seen[id]).collect();
        let mut canonical_of_id = vec![0i32; nb_colors];
        for (index, &id) in canonical_ids.iter().enumerate() {
            canonical_of_id[id] = index as i32;
        }
        (colors, canonical_of_id, indices)
    };
    for v in indices.iter_mut() {
        *v = canonical_of_id[*v as usize] as u16;
    }
    let nb_colors = colors.len();
    let orders = {
        let cooc = index_cooccurrence(&indices, xsize, ysize, nb_colors);
        palette_order_candidates(&colors, &cooc)
    };
    let mut best_order = 0usize;
    let mut best_cost = f32::INFINITY;
    for (k, order) in orders.iter().enumerate() {
        let cost = estimate_index_plane_cost(&indices, xsize, ysize, order, pool, scratch);
        if cost < best_cost {
            best_cost = cost;
            best_order = k;
        }
    }
    let order = &orders[best_order];
    // `order[canonical] = final index`.
    for v in indices.iter_mut() {
        *v = order[*v as usize] as u16;
    }
    let mut palette = vec![0i32; num_c * nb_colors];
    for (canonical, color) in colors.iter().enumerate() {
        let index = order[canonical] as usize;
        for c in 0..num_c {
            palette[c * nb_colors + index] = color[c];
        }
    }
    Some(GlobalPalette {
        palette,
        indices,
        nb_colors,
        num_c,
    })
}

#[cfg(test)]
mod tests {
    use super::super::predictor::clamped_gradient;
    use super::super::rct::rct_context;
    use super::*;
    use crate::entropy::{pack_signed, uint_encode};

    #[test]
    fn global_palette_reconstructs_rgb_and_alpha_from_input_tuple_ids() {
        use super::super::rct::forward_ycocg;

        let (w, h) = (37, 19);
        let pool = ThreadPool::new_lossless(3);
        let mut scratch = CoderScratch::lossless();
        for alpha_only in [false, true] {
            let colors: Vec<[i32; 4]> = (0..17)
                .map(|i| {
                    if alpha_only {
                        [11, 22, 33, i * 3855]
                    } else {
                        [i * 3855, 65535 - i * 1997, i * 977, i * 3855]
                    }
                })
                .collect();
            let mut linear = Image3Si::new(w, h);
            let mut alpha = Vec::with_capacity(w * h);
            let mut expected = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let color = colors[(x / 3 + y * 7) % colors.len()];
                    let (yy, co, cg) = forward_ycocg(color[0], color[1], color[2]);
                    for (c, value) in [yy, co, cg].into_iter().enumerate() {
                        linear.plane_row_mut(c, y)[x] = value;
                    }
                    alpha.push(color[3] as u16);
                    expected.push(color);
                }
            }
            let alpha = AlphaPlane::U16 {
                data: alpha,
                bits: 16,
            };
            let palette =
                build_global_palette(&linear, Some(&alpha), w, h, &pool, &mut scratch).unwrap();
            assert_eq!(palette.nb_colors, colors.len());
            for (&index, color) in palette.indices.iter().zip(expected) {
                for (c, value) in color.into_iter().enumerate() {
                    assert_eq!(
                        palette.palette[c * palette.nb_colors + index as usize],
                        value
                    );
                }
            }
        }
    }

    fn reference_index_cost_rows(
        indices: &[u16],
        xsize: usize,
        y_begin: usize,
        y_end: usize,
        order: &[i32],
    ) -> (Vec<u64>, u64) {
        const NCTX: usize = 18;
        const ALPHA: usize = 64;
        let mut hist = vec![0u64; NCTX * ALPHA];
        let mut extra_bits = 0u64;
        let at = |x: usize, y: usize| order[indices[y * xsize + x] as usize];
        for y in y_begin..y_end {
            for x in 0..xsize {
                let left = if x > 0 {
                    at(x - 1, y)
                } else if y > 0 {
                    at(x, y - 1)
                } else {
                    0
                };
                let top = if y > 0 { at(x, y - 1) } else { left };
                let topleft = if x > 0 && y > 0 {
                    at(x - 1, y - 1)
                } else {
                    left
                };
                let mx = left.max(top).max(topleft);
                let mn = left.min(top).min(topleft);
                let ctx = rct_context((mx - mn) as u32);
                let res =
                    at(x, y) as i64 - clamped_gradient(left as i64, top as i64, topleft as i64);
                let (tok, nb, _) = uint_encode(pack_signed(res as i32));
                hist[ctx * ALPHA + (tok as usize).min(ALPHA - 1)] += 1;
                extra_bits += nb as u64;
            }
        }
        (hist, extra_bits)
    }

    #[test]
    fn pooled_index_cost_matches_serial_histograms() {
        const COLORS: usize = 17;
        for (w, h) in [(0, 0), (0, 5), (1, 1), (1, 17), (3, 5), (257, 259)] {
            let indices: Vec<u16> = (0..w * h)
                .map(|i| ((i * 7 + i / w.max(1) * 11) % COLORS) as u16)
                .collect();
            for order in [
                (0..COLORS as i32).collect::<Vec<_>>(),
                (0..COLORS as i32).rev().collect(),
                (0..COLORS).map(|i| ((i * 7) % COLORS) as i32).collect(),
            ] {
                let (hist, extra_bits) = reference_index_cost_rows(&indices, w, 0, h, &order);
                let mut rows = GradientScratch::default();
                for begin in [0, h / 2, h] {
                    let expected = reference_index_cost_rows(&indices, w, begin, h, &order);
                    let actual = index_plane_cost_rows(&indices, w, begin, h, &order, &mut rows);
                    assert_eq!(actual.0.as_slice(), expected.0, "{w}x{h}, begin={begin}");
                    assert_eq!(actual.1, expected.1);
                }
                let mut reference = extra_bits as f32;
                for context in hist.as_chunks::<64>().0 {
                    reference += entropy_of_hist(context, context.iter().sum());
                }
                for threads in [1, 2, 8] {
                    let pool = ThreadPool::new_lossless(threads);
                    let mut scratch = CoderScratch::lossless();
                    let cost =
                        estimate_index_plane_cost(&indices, w, h, &order, &pool, &mut scratch);
                    assert_eq!(
                        cost.to_bits(),
                        reference.to_bits(),
                        "{w}x{h}, threads={threads}"
                    );
                }
            }
        }
    }

    #[test]
    fn cooccurrence_chain_recovers_a_scrambled_ramp() {
        // A horizontal ramp of 40 levels whose canonical ids are a random
        // permutation of the ramp position: the chain must put the levels
        // back in ramp order (either direction), which luma order cannot.
        let (w, h, n) = (200usize, 8usize, 40usize);
        let mut perm: Vec<u16> = (0..n as u16).collect();
        let mut state = 0x2545_F491u32;
        for i in (1..n).rev() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            perm.swap(i, state as usize % (i + 1));
        }
        let indices: Vec<u16> = (0..h)
            .flat_map(|_| (0..w).map(|x| perm[x * n / w]))
            .collect();
        let cooc = index_cooccurrence(&indices, w, h, n);
        let order = cooccurrence_chain(n, &cooc);
        let ramp: Vec<i32> = (0..n).map(|level| order[perm[level] as usize]).collect();
        let ascending = ramp.windows(2).all(|p| p[1] == p[0] + 1);
        let descending = ramp.windows(2).all(|p| p[1] == p[0] - 1);
        assert!(ascending || descending, "chain order {ramp:?}");
    }

    #[test]
    fn palette_scan_matches_indexed_crops_and_early_rejection() {
        for stride in [1, 3, 33, 257] {
            let height = 5;
            let mut linear = Image3Si::new(stride, height);
            for c in 0..3 {
                for y in 0..height {
                    for (x, value) in linear.plane_row_mut(c, y).iter_mut().enumerate() {
                        *value = ((x * 977 + y * 619 + c * 71) & 0xffff) as i32 - 32768;
                    }
                }
            }
            let n = stride * height;
            let alphas = [
                None,
                Some(AlphaPlane::U8((0..n).map(|i| i as u8).collect())),
                Some(AlphaPlane::U16 {
                    data: (0..n).map(|i| (i * 71) as u16).collect(),
                    bits: 16,
                }),
                Some(AlphaPlane::F32((0..n).map(|i| !(i as i32)).collect())),
            ];
            for alpha in &alphas {
                for rgb in [false, true] {
                    for (x0, y0, w, h) in [
                        (0, 0, stride, height),
                        (stride / 2, 2, stride - stride / 2, 2),
                        (0, 0, 0, height),
                        (0, 0, stride, 0),
                    ] {
                        let mut expected = Vec::new();
                        for y in y0..y0 + h {
                            for x in x0..x0 + w {
                                let i = y * stride + x;
                                let (a, b, c) = (
                                    linear.plane_data(0)[i],
                                    linear.plane_data(1)[i],
                                    linear.plane_data(2)[i],
                                );
                                let (a, b, c) = if rgb {
                                    inverse_ycocg(a, b, c)
                                } else {
                                    (a, b, c)
                                };
                                expected.push([
                                    a,
                                    b,
                                    c,
                                    alpha.as_ref().map_or(0, |a| a.get_i32(i)),
                                ]);
                            }
                        }
                        for stop in [1, stride + 1, usize::MAX] {
                            let mut actual = Vec::new();
                            let visit = |color| {
                                actual.push(color);
                                actual.len() < stop
                            };
                            let complete = if rgb {
                                visit_palette_colors::<true>(
                                    &linear,
                                    alpha.as_ref(),
                                    stride,
                                    x0,
                                    y0,
                                    w,
                                    h,
                                    visit,
                                )
                            } else {
                                visit_palette_colors::<false>(
                                    &linear,
                                    alpha.as_ref(),
                                    stride,
                                    x0,
                                    y0,
                                    w,
                                    h,
                                    visit,
                                )
                            };
                            assert_eq!(actual, expected[..expected.len().min(stop)]);
                            assert_eq!(complete, expected.len() < stop);
                        }
                    }
                }
            }
        }
    }
}
