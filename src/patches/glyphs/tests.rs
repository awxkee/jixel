/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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
use super::exact::GLYPH_ATLAS_STEP;
use super::*;
use crate::patches::MODULAR_PATCH_REF_ID;
use crate::patches::color_key::round_to_multiple;

/// Six distinct 16x16 glyphs, each repeated exactly six times (comfortably
/// past MIN_PATCH_OCCURRENCES), so every group has the same count and only
/// the tie-break can order them.
fn glyph_ctx() -> EncodingContext {
    EncodingContext::new(
        crate::Speed::Slow,
        crate::Arbiter::Ssimulacra2,
        crate::xyb::XybMatrix::SPEC,
        1.0,
        1,
    )
}

fn glyph_params(min_occurrences: usize) -> GlyphParams {
    GlyphParams {
        templates: false,
        min_occurrences,
        min_repeat_pixels: 0,
        coarse: 1,
        min_contrast: 0,
        singles_min_cover: 0.0,
    }
}

/// A page of two flat backgrounds with two antialiased marks stamped at
/// unaligned positions on both, plus one unrepeated mark. Every value sits
/// on the XYB lattice so a mark differs between backgrounds only by place.
fn glyph_page() -> Image3F {
    let (w, h) = (192usize, 128usize);
    let mut img = Image3F::new(w, h);
    for y in 0..h {
        let bg = if y < 64 {
            [4.0f32 / 4096.0, 307.0 / 512.0, 140.0 / 256.0]
        } else {
            [-8.0 / 4096.0, 102.0 / 512.0, 64.0 / 256.0]
        };
        for (c, &value) in bg.iter().enumerate() {
            img.plane_row_mut(c, y)[..w].fill(value);
        }
    }
    let mut stamp = |x0: usize, y0: usize, w: usize, h: usize, unit: f32| {
        for dy in 0..h {
            for dx in 0..w {
                let ink = ((dx * 3 + dy * 5 + w) % 4) as f32 * unit;
                img.plane_row_mut(1, y0 + dy)[x0 + dx] -= ink * 16.0 / 512.0;
                img.plane_row_mut(2, y0 + dy)[x0 + dx] -= ink * 4.0 / 256.0;
                img.plane_row_mut(0, y0 + dy)[x0 + dx] += ink * 2.0 / 4096.0;
            }
        }
    };
    for &(x, y) in &[(3, 5), (41, 9), (77, 22), (130, 37), (11, 75), (99, 101)] {
        stamp(x, y, 5, 7, 2.0);
    }
    for &(x, y) in &[(20, 40), (160, 12), (60, 90), (140, 110)] {
        stamp(x, y, 4, 4, 2.0);
    }
    stamp(150, 80, 6, 6, 1.0);
    img
}

/// kAdd semantics: base + decoded patches must give the image back up to
/// float rounding, wherever the patches land.
fn assert_glyph_plan_reconstructs(img: &Image3F, plan: &LossyGlyphPatches<Image3F>) {
    use crate::quant_weights::INV_DC_QUANT;
    let mut recon = plan.base.clone();
    let [ay, ax, ab] = &plan.atlas;
    for r in &plan.references {
        assert!(r.add);
        for &(px, py) in &r.positions {
            for dy in 0..r.height {
                for dx in 0..r.width {
                    let at = (r.atlas_y + dy) * plan.atlas_width + r.atlas_x + dx;
                    recon.plane_row_mut(1, py + dy)[px + dx] += ay[at] as f32 / INV_DC_QUANT[1];
                    recon.plane_row_mut(0, py + dy)[px + dx] += ax[at] as f32 / INV_DC_QUANT[0];
                    recon.plane_row_mut(2, py + dy)[px + dx] +=
                        (ab[at] + ay[at]) as f32 / INV_DC_QUANT[2];
                }
            }
        }
    }
    for c in 0..3 {
        for y in 0..img.ysize() {
            for (a, b) in img.plane_row(c, y)[..img.xsize()]
                .iter()
                .zip(&recon.plane_row(c, y)[..img.xsize()])
            {
                assert!((a - b).abs() < 1e-5, "plane {c} row {y}: {a} vs {b}");
            }
        }
    }
}

fn find_lossy_glyph_patches(
    ctx: &EncodingContext,
    xyb: &Image3F,
    params: &GlyphParams,
) -> Option<LossyGlyphPatches<Image3F>> {
    plan_lossy_glyph_patches(ctx, xyb, params).map(|plan| plan.materialize(xyb))
}

#[test]
fn glyph_base_subtracts_overlapping_placements_before_sorting() {
    let mut img = Image3F::new(3, 1);
    img.plane_row_mut(1, 0)[1] = 0.1;
    let plan = LossyGlyphPatches {
        atlas: [vec![-6, 120], vec![0; 2], vec![6, -120]],
        atlas_width: 2,
        atlas_height: 1,
        base: None,
        references: vec![PatchReference {
            atlas_x: 0,
            atlas_y: 0,
            width: 2,
            height: 1,
            ref_frame: MODULAR_PATCH_REF_ID,
            alpha_only: false,
            add: true,
            mul: false,
            positions: vec![(1, 0), (0, 0)],
        }],
        alpha_atlas: None,
    }
    .materialize(&img);
    let expected = (0.1f32 - (-6.0 / 512.0)) - 120.0 / 512.0;
    let reversed = (0.1f32 - 120.0 / 512.0) - (-6.0 / 512.0);
    assert_ne!(expected.to_bits(), reversed.to_bits());
    assert_eq!(plan.base.plane_row(1, 0)[1].to_bits(), expected.to_bits());
    assert_eq!(plan.references[0].positions, [(0, 0), (1, 0)]);
}

#[test]
fn lattice_rounding_matches_float_round() {
    let values: Vec<i32> = (-4099..=4099).chain([i32::MIN / 4, i32::MAX / 4]).collect();
    for shift in 1..=4u32 {
        let k = 1i32 << shift;
        let kernels: Vec<(&str, crate::xyb::RoundLatticePow2Fn)> = vec![
            ("dispatch", crate::xyb::selected_round_lattice_pow2_fn()),
            ("scalar", crate::xyb::round_lattice_pow2_scalar),
        ];
        #[cfg(any(
            all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"),
            all(target_arch = "x86_64", feature = "avx"),
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm")
        ))]
        let mut kernels = kernels;
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        if std::is_x86_feature_detected!("sse2") {
            kernels.push(("sse2", crate::sse::round_lattice_pow2_sse2));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", crate::avx::round_lattice_pow2_avx2));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        kernels.push(("neon", crate::neon::round_lattice_pow2_neon));
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "wasm"))]
        kernels.push(("wasm", crate::wasm::round_lattice_pow2_wasm));
        for (name, f) in kernels {
            let mut out = values.clone();
            unsafe { f(&mut out, shift) };
            for (&v, &got) in values.iter().zip(&out) {
                let want = (v as f64 / k as f64).round() as i32 * k;
                assert_eq!(got, want, "{name}: v={v} k={k}");
            }
        }
        for &v in &values {
            let want = (v as f64 / k as f64).round() as i32 * k;
            assert_eq!(round_to_multiple(v, k), want, "v={v} k={k}");
        }
    }
    for v in -100..=100 {
        assert_eq!(
            round_to_multiple(v, 6),
            (v as f64 / 6.0).round() as i32 * 6,
            "v={v}"
        );
    }
}

/// One dark ink alpha-blended over two different flat backgrounds: the
/// same glyph differs between them in every antialiased pixel, so exact
/// patches need one atlas entry per background.
fn blended_glyph_page() -> Image3F {
    let (w, h) = (192usize, 128usize);
    let mut img = Image3F::new(w, h);
    let backgrounds = [
        [4.0f32 / 4096.0, 300.0 / 512.0, 140.0 / 256.0],
        [-12.0 / 4096.0, 260.0 / 512.0, 200.0 / 256.0],
    ];
    let ink = [2.0f32 / 4096.0, 40.0 / 512.0, 30.0 / 256.0];
    for y in 0..h {
        let bg = backgrounds[usize::from(y >= 64)];
        for (c, &value) in bg.iter().enumerate() {
            img.plane_row_mut(c, y)[..w].fill(value);
        }
    }
    for (k, &(x0, y0)) in [
        (3, 5),
        (41, 9),
        (77, 22),
        (130, 37),
        (11, 75),
        (99, 101),
        (150, 80),
    ]
    .iter()
    .enumerate()
    {
        let bg = backgrounds[usize::from(y0 >= 64)];
        for dy in 0..7 {
            for dx in 0..5 {
                // Coverage 0.25..1 everywhere in the box: one component.
                let a = 0.25 + ((dx * 3 + dy * 5 + k % 2) % 4) as f32 * 0.25;
                let a = if k % 2 == 0 { a } else { 1.25 - a };
                for c in 0..3 {
                    img.plane_row_mut(c, y0 + dy)[x0 + dx] = bg[c] * (1.0 - a) + ink[c] * a;
                }
            }
        }
    }
    img
}

#[test]
fn glyph_templates_share_blended_glyphs_across_backgrounds() {
    use crate::quant_weights::INV_DC_QUANT;
    let img = blended_glyph_page();
    let exact = find_lossy_glyph_patches(&glyph_ctx(), &img, &glyph_params(2)).expect("plan");
    assert!(exact.alpha_atlas.is_none());
    // Glyph A twice on each background, glyph B twice on the first and
    // once on the second: three exact entries (B's single is dropped).
    assert_eq!(exact.references.len(), 3);

    let mut ctx = glyph_ctx();
    let probe = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    ctx.template_probe = Some(std::sync::Arc::clone(&probe));
    let params = GlyphParams {
        templates: true,
        ..glyph_params(2)
    };
    let plan = find_lossy_glyph_patches(&ctx, &img, &params).expect("plan");
    // Each glyph's two exact groups become one template: one 5x7 tile saved each.
    assert_eq!(probe.load(std::sync::atomic::Ordering::Relaxed), 2 * 35);
    let alpha = plan.alpha_atlas.as_ref().expect("template alpha");
    assert_eq!(alpha.len(), plan.atlas_width * plan.atlas_height);
    // One blend-above template per glyph, each placed on both backgrounds.
    assert_eq!(plan.references.len(), 2);
    assert!(plan.references.iter().all(|r| r.mul && !r.add));
    assert!(plan.references.iter().all(|r| {
        r.positions.iter().any(|p| p.1 < 64) && r.positions.iter().any(|p| p.1 >= 64)
    }));

    // kBlendAbove over an opaque base: base * (1 - a) + color * a.
    let mut recon = plan.base.clone();
    let [ay, ax, ab] = &plan.atlas;
    for r in &plan.references {
        for &(px, py) in &r.positions {
            for dy in 0..r.height {
                for dx in 0..r.width {
                    let at = (r.atlas_y + dy) * plan.atlas_width + r.atlas_x + dx;
                    let a = f32::from(alpha[at]) / 255.0;
                    let color = [
                        ax[at] as f32 / INV_DC_QUANT[0],
                        ay[at] as f32 / INV_DC_QUANT[1],
                        (ab[at] + ay[at]) as f32 / INV_DC_QUANT[2],
                    ];
                    for (c, &color) in color.iter().enumerate() {
                        let v = &mut recon.plane_row_mut(c, py + dy)[px + dx];
                        *v = *v * (1.0 - a) + color * a;
                    }
                }
            }
        }
    }
    for c in 0..3 {
        let tolerance = 4.0 / INV_DC_QUANT[c];
        for y in 0..img.ysize() {
            for (x, (a, b)) in img.plane_row(c, y)[..img.xsize()]
                .iter()
                .zip(&recon.plane_row(c, y)[..img.xsize()])
                .enumerate()
            {
                assert!(
                    (a - b).abs() <= tolerance,
                    "plane {c} ({x}, {y}): {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn lossy_glyph_patches_reconstruct_and_flatten_the_base() {
    let img = glyph_page();
    let repeated = find_lossy_glyph_patches(&glyph_ctx(), &img, &glyph_params(2)).expect("plan");
    // The mark differs between the two backgrounds only by where it sits.
    assert_eq!(repeated.references.len(), 2);
    let mut counts: Vec<usize> = repeated
        .references
        .iter()
        .map(|r| r.positions.len())
        .collect();
    counts.shared_sort();
    assert_eq!(counts, [4, 6]);
    assert_glyph_plan_reconstructs(&img, &repeated);
    // Under the repeated marks the base is the background again, to within
    // the lattice rounding of the patch.
    let step = GLYPH_ATLAS_STEP as f32 / 2.0 / crate::quant_weights::INV_DC_QUANT[1];
    assert!((repeated.base.plane_row(1, 8)[5] - 307.0 / 512.0).abs() <= step);

    let all = find_lossy_glyph_patches(&glyph_ctx(), &img, &glyph_params(1)).expect("plan");
    assert_eq!(all.references.len(), 3);
    assert_glyph_plan_reconstructs(&img, &all);
}

/// A ramp of 12-pixel steps (two luma lattice units each) with one bright
/// antialiased mark per step; every other mark has a dark rule against its
/// left edge and is one pixel wider for it. Returns the page and the box
/// corners.
fn ramp_glyph_page() -> (Image3F, Vec<(usize, usize)>) {
    let (w, h) = (240usize, 96usize);
    let ground = |x: usize| {
        let step = (x / 12) as f32;
        [
            2.0f32 / 4096.0,
            (120.0 + 2.0 * step) / 512.0,
            (60.0 + step) / 256.0,
        ]
    };
    let ink = [0.0f32, 420.0 / 512.0, 210.0 / 256.0];
    let mut img = Image3F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            for (c, value) in ground(x).into_iter().enumerate() {
                img.plane_row_mut(c, y)[x] = value;
            }
        }
    }
    let mut marks = Vec::new();
    for k in 0..18 {
        let (x0, y0) = (12 * k + 15, 10 + (k % 3) * 27);
        marks.push((x0 - usize::from(k % 2 == 0), y0));
        for dy in 0..7 {
            for dx in 0..5 {
                let a = 0.25 + ((dx * 3 + dy * 5) % 4) as f32 * 0.25;
                let bg = ground(x0 + dx);
                for c in 0..3 {
                    img.plane_row_mut(c, y0 + dy)[x0 + dx] = bg[c] * (1.0 - a) + ink[c] * a;
                }
            }
        }
        if k % 2 == 0 {
            // A rule in another color against the mark's left edge.
            for dy in 0..7 {
                img.plane_row_mut(1, y0 + dy)[x0 - 1] = 20.0 / 512.0;
            }
        }
    }
    (img, marks)
}

#[test]
fn smooth_ground_goes_back_to_the_background() {
    let (img, marks) = ramp_glyph_page();
    let lattice = Lattice::new(&img, 1);
    let keys = ink::color_keys(&glyph_ctx(), &lattice);
    let (w, h) = (img.xsize(), img.ysize());
    let mut mask = ink::ink_mask(&keys, w, h);
    // The tile vote keeps a few steps of the ramp and calls the rest ink.
    let flat_ink = mask.iter().filter(|&&p| p).count();
    assert!(flat_ink > w * h / 4);
    let flat_mask = mask.clone();
    let ground = ink::absorb_smooth_ground(&lattice, &mut mask).expect("a ramp");
    // A frame that does not use its ground gets the flat mask back.
    let mut restored = mask.clone();
    ground.give_back(&mut restored);
    assert_eq!(restored, flat_mask);
    // What is left is the marks and their rules, nothing of the ramp.
    let expected = marks.len() * 35 + marks.len().div_ceil(2) * 7;
    assert_eq!(mask.iter().filter(|&&p| p).count(), expected);
    for &(x0, y0) in &marks {
        assert!(mask[y0 * w + x0] && !ground.holds(x0, y0));
        assert!(!mask[(y0 - 1) * w + x0]);
    }
    let held = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| ground.holds(x, y))
        .count();
    assert_eq!(held, flat_ink - expected);
    // A step the tile vote kept flat continues the ramp around it.
    let (x, y) = (0..w)
        .map(|x| (x, 50))
        .find(|&(x, y)| !mask[y * w + x] && !ground.holds(x, y))
        .expect("a flat step");
    let color = std::array::from_fn(|c| lattice.planes[c][y * w + x]);
    assert!(ground.continues(x, y, color));
    assert!(!ground.continues(x, y, [color[0] + 200, color[1], color[2]]));
}

#[test]
fn glyphs_on_a_ramp_share_entries_and_reconstruct() {
    let (img, marks) = ramp_glyph_page();
    // Additive entries need identical occurrences, and on a ramp every
    // step shifts the mark: nothing repeats.
    assert!(find_lossy_glyph_patches(&glyph_ctx(), &img, &glyph_params(2)).is_none());

    // One blend-above template serves every step of the ramp: one for the
    // plain mark and one for the mark with its rule.
    let params = GlyphParams {
        templates: true,
        ..glyph_params(2)
    };
    let plan = find_lossy_glyph_patches(&glyph_ctx(), &img, &params).expect("plan");
    assert_eq!(plan.references.len(), 2);
    assert!(plan.references.iter().all(|r| r.mul && !r.add));
    let mut widths: Vec<usize> = plan.references.iter().map(|r| r.width).collect();
    widths.shared_sort();
    assert_eq!(widths, [5, 6]);
    let mut positions: Vec<(usize, usize)> = plan
        .references
        .iter()
        .flat_map(|r| r.positions.iter().copied())
        .collect();
    positions.shared_sort();
    let mut expected = marks.clone();
    expected.shared_sort();
    assert_eq!(positions, expected);
    // Under a quarter-covered pixel the base carries the difference and is
    // the ramp again.
    let (x0, y0) = marks[1];
    let ground = (120.0 + 2.0 * ((x0 / 12) as f32)) / 512.0;
    assert!((plan.base.plane_row(1, y0)[x0] - ground).abs() <= 4.0 / 512.0);
}
