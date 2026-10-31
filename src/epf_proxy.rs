//! Six-term EPF ranking proxy. Weights frozen before Kodak evaluation.
//! Half/quarter-resolution luma structure and full-resolution adjacent gradients.
//! No perceptual metric dependency; input is the encoder's raw XYB Y plane.

use crate::thread_pool::ThreadPool;

#[path = "epf_proxy/box_filter.rs"]
mod box_filter;
#[path = "epf_proxy/row_kernels.rs"]
mod row_kernels;

fn resize(v: &mut Vec<f32>, n: usize) {
    v.resize(n, 0.);
}

#[inline]
fn mirror(i: isize, n: usize) -> usize {
    let n = n as isize;
    if i >= 0 && i < n {
        return i as usize;
    }
    let i = i.rem_euclid(2 * n);
    if i < n {
        i as usize
    } else {
        (2 * n - i - 1) as usize
    }
}

fn half(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, k: row_kernels::Kernels) {
    let (ow, oh) = (w / 2, h / 2);
    resize(dst, ow * oh);
    for y in 0..oh {
        (k.half)(
            [
                &src[2 * y * w..(2 * y + 1) * w],
                &src[(2 * y + 1) * w..(2 * y + 2) * w],
            ],
            &mut dst[y * ow..(y + 1) * ow],
        );
    }
}

// Keep the full slice length so the next candidate needs no resize/zero-fill.
fn half_in_place(values: &mut [f32], w: usize, h: usize, k: row_kernels::Kernels) -> &[f32] {
    (k.half_in_place)(values, w, h);
    &values[..(w / 2) * (h / 2)]
}

fn half_y(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, k: row_kernels::Kernels) {
    let (ow, oh) = (w / 2, h / 2);
    resize(dst, ow * oh);
    for y in 0..oh {
        (k.half_y)(
            [
                &src[2 * y * w..(2 * y + 1) * w],
                &src[(2 * y + 1) * w..(2 * y + 2) * w],
            ],
            &mut dst[y * ow..(y + 1) * ow],
        );
    }
}

struct Scale {
    values: Vec<f32>,
    mean: Vec<f32>,
    variance: Vec<f32>,
    width: usize,
    height: usize,
}

impl Scale {
    fn new(
        values: Vec<f32>,
        width: usize,
        height: usize,
        kernels: box_filter::Kernels,
        work: &mut box_filter::Scratch,
        rows: row_kernels::Kernels,
    ) -> Self {
        let mut mean = vec![0.; values.len()];
        let mut variance = vec![0.; values.len()];
        work.run::<false>(kernels, &values, &[], width, height, |y, [m, square, _]| {
            let start = y * width;
            mean[start..start + width].copy_from_slice(m);
            (rows.variance)(m, square, &mut variance[start..start + width]);
        });
        Self {
            values,
            mean,
            variance,
            width,
            height,
        }
    }

    fn compare(
        &self,
        test: &[f32],
        work: &mut box_filter::Scratch,
        kernels: box_filter::Kernels,
        rows: row_kernels::Kernels,
    ) -> [f32; 2] {
        let (w, h) = (self.width, self.height);
        let mut pooled = [0.0; 2];
        work.run::<true>(
            kernels,
            test,
            &self.values,
            w,
            h,
            |y, [mean, square, cross]| {
                let start = y * w;
                pooled = (rows.pool)(
                    [
                        mean,
                        square,
                        cross,
                        &self.mean[start..start + w],
                        &self.variance[start..start + w],
                    ],
                    pooled,
                );
            },
        );
        let [sum, fourth] = pooled;
        let n = test.len() as f32;
        [sum / n, (fourth / n).sqrt().sqrt()]
    }
}

#[derive(Default)]
pub(crate) struct Scratch {
    half: Vec<f32>,
    moments: box_filter::Scratch,
}

pub(crate) struct LumaProxy<'a> {
    source: &'a [f32],
    pool: &'a ThreadPool,
    half: Scale,
    quarter: Scale,
    width: usize,
    height: usize,
    kernels: box_filter::Kernels,
    rows: row_kernels::Kernels,
}

impl<'a> LumaProxy<'a> {
    /// Score scales and gradients concurrently on the shared pool.
    pub(crate) fn new(y: &'a [f32], width: usize, height: usize, pool: &'a ThreadPool) -> Self {
        assert!(width >= 4 && height >= 4);
        assert_eq!(y.len(), width * height);
        let mut h = Vec::new();
        let mut q = Vec::new();
        let rows = row_kernels::selected();
        half_y(y, width, height, &mut h, rows);
        half(&h, width / 2, height / 2, &mut q, rows);
        let kernels = box_filter::selected_kernels();
        let mut work = box_filter::Scratch::default();
        Self {
            source: y,
            pool,
            half: Scale::new(h, width / 2, height / 2, kernels, &mut work, rows),
            quarter: Scale::new(q, width / 4, height / 4, kernels, &mut work, rows),
            width,
            height,
            kernels,
            rows,
        }
    }

    pub(crate) fn compare(&self, y: &[f32], work: &mut Scratch) -> (f32, [f32; 6]) {
        let (w, h) = (self.width, self.height);
        assert_eq!(y.len(), w * h);
        let scales = |work: &mut Scratch| {
            half_y(y, w, h, &mut work.half, self.rows);
            let hs = self
                .half
                .compare(&work.half, &mut work.moments, self.kernels, self.rows);
            let quarter = half_in_place(&mut work.half, w / 2, h / 2, self.rows);
            let qs = self
                .quarter
                .compare(quarter, &mut work.moments, self.kernels, self.rows);
            (hs, qs)
        };
        let ((hs, qs), [error, excess, loss4]) =
            self.pool.join(|| scales(work), || self.gradients(y));
        let n = (h * (w - 1) + (h - 1) * w) as f32;
        let terms = [
            qs[0],
            hs[0],
            hs[1],
            error / n,
            excess / n,
            (loss4 / n).sqrt().sqrt(),
        ];
        static WEIGHTS: [f32; 6] = [
            109.4163228369586,
            4.685925102200196,
            2.491781270704224,
            2.167275675983448,
            4.55227651921194,
            2.847714678987977,
        ];
        (
            terms
                .iter()
                .zip(WEIGHTS)
                .map(|(&v, weight)| v * weight)
                .sum(),
            terms,
        )
    }

    /// Error, excess and fourth-power pools of the horizontal, then the
    /// vertical neighbor differences against the source.
    fn gradients(&self, y: &[f32]) -> [f32; 3] {
        let (w, h) = (self.width, self.height);
        let mut pooled = [0.0; 3];
        for row in 0..h {
            let i = row * w;
            pooled = (self.rows.gradients)(
                [
                    &self.source[i..i + w - 1],
                    &self.source[i + 1..i + w],
                    &y[i..i + w - 1],
                    &y[i + 1..i + w],
                ],
                pooled,
            );
        }
        for row in 0..h - 1 {
            let i = row * w;
            pooled = (self.rows.gradients)(
                [
                    &self.source[i..i + w],
                    &self.source[i + w..i + 2 * w],
                    &y[i..i + w],
                    &y[i + w..i + 2 * w],
                ],
                pooled,
            );
        }
        pooled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_place_pyramid_matches_separate_downsampling_on_odd_sizes() {
        for (w, h) in [(2, 2), (3, 7), (16, 17), (33, 19)] {
            let mut values: Vec<_> = (0..w * h).map(|i| i as f32 * 0.007).collect();
            let mut expected = Vec::new();
            half(&values, w, h, &mut expected, row_kernels::selected());
            assert_eq!(
                half_in_place(&mut values, w, h, row_kernels::selected()),
                expected
            );
            assert_eq!(values.len(), w * h);
        }
    }

    #[test]
    fn flat_fields_keep_identity_zero_without_double_precision() {
        let (w, h) = (257, 129);
        for value in [0., 0.07, 0.21, 0.53, 0.91] {
            let y = vec![value; w * h];
            let pool = ThreadPool::new(1);
            let reference = LumaProxy::new(&y, w, h, &pool);
            assert_eq!(reference.source.as_ptr(), y.as_ptr());
            let mut work = Scratch::default();
            let identity = reference.compare(&y, &mut work).0;
            assert!(identity < 1e-5, "value={value}, identity={identity}");
            let shifted = vec![value + 0.02; w * h];
            let score = reference.compare(&shifted, &mut work).0;
            assert!(score.is_finite() && score > identity);
        }
    }

    #[test]
    fn identity_and_blurred_detail_on_odd_sizes() {
        for (w, h) in [(4, 4), (19, 27), (128, 65)] {
            let y: Vec<_> = (0..w * h).map(|i| 0.2 + (i % 17) as f32 * 0.01).collect();
            let pool = ThreadPool::new(1);
            let reference = LumaProxy::new(&y, w, h, &pool);
            let mut work = Scratch::default();
            let identity = reference.compare(&y, &mut work).0;
            assert!(identity < 1e-5, "{identity}");
            let flat = vec![0.28; w * h];
            assert!(reference.compare(&flat, &mut work).0 > identity + 0.01);
        }
    }

    #[test]
    fn pooled_scores_match_serial_scores() {
        let (w, h) = (131, 67);
        let source: Vec<_> = (0..w * h).map(|i| (i % 47) as f32 * 0.01).collect();
        let serial_pool = ThreadPool::new(1);
        let parallel_pool = ThreadPool::new(4);
        let serial = LumaProxy::new(&source, w, h, &serial_pool);
        let parallel = LumaProxy::new(&source, w, h, &parallel_pool);
        let mut serial_work = Scratch::default();
        let mut parallel_work = Scratch::default();
        for shift in [0.0, 0.01, -0.03] {
            let test: Vec<_> = source.iter().map(|&v| v + shift).collect();
            assert_eq!(
                serial.compare(&test, &mut serial_work),
                parallel.compare(&test, &mut parallel_work),
            );
        }
    }
}
