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

//! Independent coordinate-based scalar oracle; it has no row caches or in-place writes.
use super::*;

fn sample(image: &Image3F, c: usize, x: isize, y: isize) -> f32 {
    image.plane_row(c, mirror(y, image.ysize() as isize))[mirror(x, image.xsize() as isize)]
}
fn difference(image: &Image3F, lf: &LoopFilter, x: isize, y: isize, dx: isize, dy: isize) -> f32 {
    let mut result = 0.0;
    for c in 0..3 {
        result +=
            (sample(image, c, x, y) - sample(image, c, x + dx, y + dy)).abs() * lf.channel_scale[c];
    }
    result
}
fn reference_stage(
    image: &Image3F,
    blocks: &FrameBlocks,
    lf: &LoopFilter,
    sharpness: i32,
    stage: &Stage,
    sm: f32,
) -> Image3F {
    let (w, h) = (image.xsize(), image.ysize());
    let mut out = Image3F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let quant = blocks.sigma_quant[(y / 8) * blocks.xsize + x / 8];
            let s = 1.0 / (lf.quant_mul / quant * (sharpness as f32 / 7.0)).min(-1e-4);
            let keep = if s >= MIN_SIGMA { 1.0 } else { 0.0 };
            let sm = if y % 8 == 0 || y % 8 == 7 || x % 8 == 0 || x % 8 == 7 {
                sm * lf.border_sad_mul
            } else {
                sm
            };
            let inv = if s >= MIN_SIGMA { s * sm } else { 0.0 };
            let mut sum = 1.0;
            let mut acc = std::array::from_fn::<_, 3, _>(|c| image.plane_row(c, y)[x]);
            for &(dir, along) in stage.neighbors {
                let (dx, dy) = stage.dirs[dir];
                let (sx, sy) = if along {
                    (x as isize, y as isize)
                } else {
                    (x as isize - dx, y as isize - dy)
                };
                let diff = |x, y| difference(image, lf, x, y, dx, dy);
                let sad = if stage.patch {
                    diff(sx, sy)
                        + (diff(sx - 1, sy) + diff(sx + 1, sy))
                        + (diff(sx, sy - 1) + diff(sx, sy + 1))
                } else {
                    diff(sx, sy)
                };
                let weight = (sad * inv + 1.0).max(0.0) * keep;
                sum += weight;
                let (nx, ny) = if along {
                    (x as isize + dx, y as isize + dy)
                } else {
                    (sx, sy)
                };
                for c in 0..3 {
                    acc[c] += weight * sample(image, c, nx, ny);
                }
            }
            let inv_sum = 1.0 / sum;
            for c in 0..3 {
                out.plane_row_mut(c, y)[x] = acc[c] * inv_sum;
            }
        }
    }
    out
}
fn reference(image: &Image3F, blocks: &FrameBlocks, lf: &LoopFilter, sharpness: i32) -> Image3F {
    if lf.epf_iters == 0 || sharpness == 0 {
        return image.clone();
    }
    let mut out = image.clone();
    if lf.epf_iters >= 3 {
        out = reference_stage(
            &out,
            blocks,
            lf,
            sharpness,
            &STAGE0,
            SAD_MUL * lf.pass0_sigma_scale,
        );
    }
    out = reference_stage(&out, blocks, lf, sharpness, &STAGE1, SAD_MUL);
    if lf.epf_iters >= 2 {
        out = reference_stage(
            &out,
            blocks,
            lf,
            sharpness,
            &STAGE2,
            SAD_MUL * lf.pass2_sigma_scale,
        );
    }
    out
}
fn image(w: usize, h: usize) -> Image3F {
    let mut image = Image3F::new(w, h);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                image.plane_row_mut(c, y)[x] =
                    0.1 + ((x * 1877 + y * 6719 + c * 941) % 3253) as f32 * 0.0001;
            }
        }
    }
    image
}
fn assert_image_eq(actual: &Image3F, expected: &Image3F, context: &str) {
    for c in 0..3 {
        for y in 0..actual.ysize() {
            for x in 0..actual.xsize() {
                assert_eq!(
                    actual.plane_row(c, y)[x].to_bits(),
                    expected.plane_row(c, y)[x].to_bits(),
                    "{context}, channel={c}, ({x},{y})"
                );
            }
        }
    }
}
#[test]
fn streamed_stages_match_scalar_oracle_with_mixed_blocks_all_tails_and_workers() {
    for (w, h) in (1..=17).map(|w| (w, 19 - w)).chain([(35, 27), (65, 49)]) {
        let image = image(w, h);
        let blocks = FrameBlocks {
            xsize: w.div_ceil(8),
            ysize: h.div_ceil(8),
            sigma_quant: (0..w.div_ceil(8) * h.div_ceil(8))
                .map(|i| [-0.1, -1.0, -10.0, 0.0][i % 4])
                .collect(),
        };
        let candidates: Vec<_> = [
            (3, 0.46, 7, 0.9),
            (1, 1.84, 3, 0.9),
            (2, 0.46, 7, 0.9),
            (3, 1.84, 7, 3.6),
            (2, 0.12, 7, 0.9),
            (0, 0.46, 7, 0.9),
            (1, 0.46, 0, 0.9),
        ]
        .map(|(iters, q, sharp, pass0)| {
            let lf = LoopFilter::new(iters, Some(pass0), Some(q), Some([31.0, 6.0, 4.5]));
            let expected = reference(&image, &blocks, &lf, sharp);
            (lf, sharp, expected)
        })
        .into();
        for threads in [1, 2, 4] {
            let pool = ThreadPool::new(threads);
            let mut scratch = CoderScratch::default();
            for (name, kernels) in kernels::available() {
                let mut epf = Epf::new(&pool, &image, &blocks);
                epf.kernels = kernels;
                for (lf, sharp, expected) in &candidates {
                    let actual = epf.apply(&pool, &mut scratch, lf, *sharp);
                    assert_image_eq(
                        actual,
                        expected,
                        &format!(
                            "{name}, {w}x{h}, workers={threads}, iters={}, sharpness={sharp}",
                            lf.epf_iters
                        ),
                    );
                }
            }
        }
    }
}
#[test]
fn row_scratch_depends_on_width_and_workers_and_noop_does_not_allocate_output() {
    let mut size = None;
    let pool = ThreadPool::new(2);
    let mut scratch = CoderScratch::default();
    for h in [33, 129, 513] {
        let w = 35;
        let image = image(w, h);
        let blocks = FrameBlocks {
            xsize: w.div_ceil(8),
            ysize: h.div_ceil(8),
            sigma_quant: vec![-1.0; w.div_ceil(8) * h.div_ceil(8)],
        };
        let lf = LoopFilter::new(3, None, None, None);
        let mut epf = Epf::new(&pool, &image, &blocks);
        assert!(std::ptr::eq(epf.apply(&pool, &mut scratch, &lf, 0), &image));
        assert!(epf.output.is_none());
        assert_eq!(
            epf.work.iter().map(stream::Scratch::floats).sum::<usize>(),
            0
        );
        epf.apply(&pool, &mut scratch, &lf, 7);
        let floats = epf.work.iter().map(stream::Scratch::floats).sum::<usize>();
        assert!(floats <= 80 * (w + 6) * pool.num_threads(), "{floats}");
        if let Some(size) = size {
            assert_eq!(floats, size);
        } else {
            size = Some(floats);
        }
    }
}
#[test]
fn gaborish_matches_coordinate_oracle_without_padding() {
    const W1: f32 = 1.1 * 0.104_699_57;
    const W2: f32 = 1.1 * 0.055_680_54;
    let norm = 1.0 / (1.0 + 4.0 * (W1 + W2));
    let (w0, w1, w2) = (norm, W1 * norm, W2 * norm);
    for w in 1..=19 {
        let image = image(w, 7);
        let mut expected = Image3F::new(w, 7);
        for c in 0..3 {
            for y in 0usize..7 {
                for x in 0..w {
                    let get = |dx, dy| sample(&image, c, x as isize + dx, y as isize + dy);
                    let side = (get(-1, 0) + get(1, 0)) + (get(0, -1) + get(0, 1));
                    let corner = (get(-1, -1) + get(1, -1)) + (get(-1, 1) + get(1, 1));
                    expected.plane_row_mut(c, y)[x] = corner * w2 + (side * w1 + get(0, 0) * w0);
                }
            }
        }
        assert_image_eq(&gaborish(&image), &expected, "gaborish selected");
        let mut in_place = image.clone();
        gaborish_in_place(&mut in_place);
        assert_image_eq(&in_place, &expected, "gaborish in place");
        for (name, k) in kernels::available() {
            let mut actual = Image3F::new(w, 7);
            for c in 0..3 {
                for y in 0usize..7 {
                    (k.gaborish)(
                        [
                            image.plane_row(c, y.saturating_sub(1)),
                            image.plane_row(c, y),
                            image.plane_row(c, (y + 1).min(6)),
                        ],
                        [w0, w1, w2],
                        actual.plane_row_mut(c, y),
                    );
                }
            }
            assert_image_eq(&actual, &expected, name);
        }
    }
}
#[test]
fn sigma_and_dc_kernels_match_scalar_bits() {
    let scalar = kernels::scalar();
    for (name, k) in kernels::available() {
        for n in 1..=33 {
            let src: Vec<_> = (0..n)
                .map(|i| [0.0, -0.0, -0.1, -1.0, -10.0, f32::NAN, f32::NEG_INFINITY][i % 7])
                .collect();
            for sharp in [0., 3. / 7., 1.] {
                let mut expected = vec![0.; n];
                let mut actual = vec![0.; n];
                (scalar.sigma)(&src, sharp, 0.46, &mut expected);
                (k.sigma)(&src, sharp, 0.46, &mut actual);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{name}, sigma"
                );
            }
            let src: Vec<_> = (0..n).map(|i| i as f32 * 0.003 - 0.1).collect();
            let mut a = src.clone();
            let mut b = src.clone();
            (scalar.add)(&mut a, &src);
            (k.add)(&mut b, &src);
            assert_eq!(a, b, "{name}, add");
            (scalar.add_constant)(&mut a, 0.023);
            (k.add_constant)(&mut b, 0.023);
            assert_eq!(a, b, "{name}, add constant");
        }
        for (n, cx) in [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 2),
            (5, 5),
            (8, 4),
            (16, 4),
            (64, 8),
            (256, 16),
        ] {
            let matrix: Vec<_> = (0..n * n).map(|i| (i % 31) as f32 * 0.007 - 0.13).collect();
            let delta: Vec<_> = (0..n).map(|i| (i % 17) as f32 * 0.003 - 0.04).collect();
            let mut a = vec![0.; n * 64];
            let mut b = vec![0.; n * 64];
            (scalar.project)(&matrix, &delta, cx, &mut a);
            (k.project)(&matrix, &delta, cx, &mut b);
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}, projection n={n}"
            );
        }
    }
}
