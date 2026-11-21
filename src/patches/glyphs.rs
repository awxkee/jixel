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
use crate::encoding_context::EncodingContext;
use crate::image::Image3F;
use crate::patches::PatchReference;
use crate::shared_sort::SharedSort;

mod components;
mod exact;
mod grid;
mod ink;
mod templates;

/// Largest glyph box side the glyph detector keeps.
pub(super) const GLYPH_MAX_SIDE: usize = 64;
/// Background is decided per tile of this size (the tile's most frequent
/// color, or that of its eight neighbors, so boxes and headers whose
/// edges cross a tile still count as background on both sides).
pub(super) const GLYPH_BG_TILE: usize = 32;
/// A shape must repeat: (occurrences - 1) * box pixels at or above these pays
/// for its atlas entry and per-occurrence position tokens.
pub(super) const GLYPH_MIN_REPEAT_PIXELS: usize = 48;
/// Widest atlas the shapes are shelf-packed into.
pub(super) const GLYPH_ATLAS_MAX_WIDTH: usize = 1024;

/// Glyph patches for the lossy path, libjxl style: connected ink components
/// over a flat background, stored as the difference from that background on
/// the modular XYB lattice and blended with kAdd. The base keeps the image
/// minus the decoded patch, i.e. the flat background plus lattice rounding.
pub(crate) struct LossyGlyphPatches<Base> {
    /// Lattice channels Y, X, B-Y of the atlas.
    pub(crate) atlas: [Vec<i32>; 3],
    pub(crate) atlas_width: usize,
    pub(crate) atlas_height: usize,
    pub(crate) base: Base,
    pub(crate) references: Vec<PatchReference>,
    /// 8-bit atlas alpha: template coverage, zero under the kAdd entries.
    /// `None` when no template was placed.
    pub(crate) alpha_atlas: Option<Vec<u8>>,
}

impl LossyGlyphPatches<Option<Image3F>> {
    /// Materialize the additive residual after the atlas has been coded.
    /// The template base, if present, already has its blend-above patches
    /// removed. Additive placements keep their original order until the
    /// subtraction finishes, since their bounding boxes can overlap.
    pub(crate) fn materialize(self, xyb: &Image3F) -> LossyGlyphPatches<Image3F> {
        use crate::quant_weights::INV_DC_QUANT;

        let Self {
            atlas,
            atlas_width,
            atlas_height,
            base,
            mut references,
            alpha_atlas,
        } = self;
        let mut base = base.unwrap_or_else(|| xyb.clone());
        // Atlas order is Y, X, B-Y, while the image planes are X, Y, B.
        let image_planes = [1usize, 0, 2];
        let steps = image_planes.map(|c| 1.0 / INV_DC_QUANT[c]);
        for r in references.iter_mut().filter(|r| r.add) {
            for &(x, y) in &r.positions {
                for c in 0..3 {
                    for dy in 0..r.height {
                        let row = base.plane_row_mut(image_planes[c], y + dy);
                        let start = (r.atlas_y + dy) * atlas_width + r.atlas_x;
                        let pixels = row[x..][..r.width]
                            .iter_mut()
                            .zip(&atlas[c][start..][..r.width]);
                        for (dx, (pixel, &sample)) in pixels.enumerate() {
                            let d = if c == 2 {
                                sample + atlas[0][start + dx]
                            } else {
                                sample
                            };
                            if d != 0 {
                                *pixel -= d as f32 * steps[c];
                            }
                        }
                    }
                }
            }
            r.positions.shared_sort_unstable_by_key(|&(x, y)| (y, x));
        }
        LossyGlyphPatches {
            atlas,
            atlas_width,
            atlas_height,
            base,
            references,
            alpha_atlas,
        }
    }
}

pub(crate) struct GlyphParams {
    /// Share glyphs recurring over different backgrounds as blend-above
    /// templates; the frame must carry an 8-bit alpha channel.
    pub(crate) templates: bool,
    pub(crate) min_occurrences: usize,
    /// (occurrences - 1) * box pixels a group must reach.
    pub(crate) min_repeat_pixels: usize,
    /// Lattice values are rounded to multiples of this.
    pub(crate) coarse: i32,
    /// A shape whose largest lattice difference from its background is
    /// below this is left to VarDCT.
    pub(crate) min_contrast: i32,
    /// Unrepeated groups are kept only when together they cover this share
    /// of the frame (0 keeps them unconditionally).
    pub(crate) singles_min_cover: f64,
}

struct GlyphShape {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    label: u32,
    bg: [i32; 3],
    hash: u64,
}

/// The frame on the modular XYB lattice: planes Y, X, B (the rounded B
/// itself, not B-Y) at full precision. Glyphs are matched on these values
/// rounded to multiples of `1 << shift`.
struct Lattice {
    planes: [Vec<i32>; 3],
    width: usize,
    height: usize,
    shift: u32,
}

impl Lattice {
    fn new(xyb: &Image3F, coarse: i32) -> Self {
        debug_assert!(coarse > 0 && (coarse as u32).is_power_of_two());
        let [qy, qx, mut qb] = crate::xyb::quantize_xyb_channels(xyb, 1);
        // Work with the rounded B itself; the atlas goes back to B-Y at the end.
        for (b, &y) in qb.iter_mut().zip(&qy) {
            *b += y;
        }
        Self {
            planes: [qy, qx, qb],
            width: xyb.xsize(),
            height: xyb.ysize(),
            shift: coarse.trailing_zeros(),
        }
    }

    #[inline]
    fn coarse(&self, value: i32) -> i32 {
        crate::xyb::round_lattice_pow2_value(value, self.shift)
    }

    /// Coarse difference of a shape's pixel from its background; zero where
    /// the pixel belongs to another component.
    #[inline]
    fn diff(&self, label: &[u32], s: &GlyphShape, c: usize, dx: usize, dy: usize) -> i32 {
        let i = (s.y0 + dy) * self.width + s.x0 + dx;
        if label[i] == s.label {
            self.coarse(self.planes[c][i]) - s.bg[c]
        } else {
            0
        }
    }

    /// The shape of the pixels labeled `owner` within an inclusive box,
    /// unless it is too faint to be worth an atlas entry.
    fn shape_of(
        &self,
        label: &[u32],
        (x0, x1, y0, y1): PixelBox,
        owner: u32,
        bg: [i32; 3],
        min_contrast: i32,
    ) -> Option<GlyphShape> {
        let width = self.width;
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        let mut hash: u64 = 0x9e37_79b9_7f4a_7c15 ^ ((w as u64) << 32 | h as u64);
        let mut contrast: i32 = 0;
        let rows = y0 * width..(y1 + 1) * width;
        for (plane, &bg) in self.planes.iter().zip(&bg) {
            let plane_rows = plane[rows.clone()].chunks_exact(width);
            let label_rows = label[rows.clone()].chunks_exact(width);
            for (values, owners) in plane_rows.zip(label_rows) {
                for (&value, &pixel_owner) in values[x0..=x1].iter().zip(&owners[x0..=x1]) {
                    let v = if pixel_owner == owner {
                        self.coarse(value) - bg
                    } else {
                        0
                    };
                    contrast = contrast.max(v.abs());
                    hash = (hash ^ (v as u32 as u64)).wrapping_mul(0xff51_afd7_ed55_8ccd);
                    hash ^= hash >> 29;
                }
            }
        }
        (contrast >= min_contrast).then_some(GlyphShape {
            x0,
            y0,
            w,
            h,
            label: owner,
            bg,
            hash,
        })
    }
}

/// Inclusive pixel box: x0, x1, y0, y1.
type PixelBox = (usize, usize, usize, usize);

/// Plans the glyph patches of a frame: ink over flat backgrounds is cut into
/// connected components (touching monospace characters into grid cells),
/// glyphs shared across backgrounds become blend-above templates, and the
/// repeating rest goes to the exact additive atlas.
pub(crate) fn plan_lossy_glyph_patches(
    ctx: &EncodingContext,
    xyb: &Image3F,
    params: &GlyphParams,
) -> Option<LossyGlyphPatches<Option<Image3F>>> {
    let (width, height) = (xyb.xsize(), xyb.ysize());
    if width < GLYPH_BG_TILE || height < GLYPH_BG_TILE || ink::sample_is_mostly_ink(xyb, params) {
        return None;
    }
    let lattice = Lattice::new(xyb, params.coarse);
    let keys = ink::color_keys(ctx, &lattice);
    let ink = ink::ink_mask(&keys, width, height)?;
    let mut components = components::label_components(&lattice, &ink, &keys, params.min_contrast);
    let grid = grid::detect_glyph_grid(&ink, width, height, &components.whole_shapes());
    drop((ink, keys));
    let splits = match &grid {
        Some(grid) => grid::split_touching_glyphs(grid, &lattice, &mut components, params),
        None => Vec::new(),
    };
    let (mut shapes, label) = components.into_shapes(splits);
    if shapes.len() < 4 {
        return None;
    }
    let templates =
        templates::take_glyph_templates(ctx, params, xyb, &lattice, &label, &mut shapes);
    exact::plan_exact_glyphs(&lattice, &label, &shapes, params, templates)
}

#[cfg(test)]
mod tests;
