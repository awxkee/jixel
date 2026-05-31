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

#[derive(Clone)]
pub(crate) struct Plane<T> {
    data: Vec<T>,
    xsize: usize,
    ysize: usize,
}

impl<T: Copy + Default> Plane<T> {
    pub(crate) fn new(xsize: usize, ysize: usize) -> Self {
        Self {
            data: vec![T::default(); xsize * ysize],
            xsize,
            ysize,
        }
    }

    pub(crate) fn new_fill(xsize: usize, ysize: usize, val: T) -> Self {
        Self {
            data: vec![val; xsize * ysize],
            xsize,
            ysize,
        }
    }

    #[inline]
    pub(crate) fn xsize(&self) -> usize {
        self.xsize
    }
    #[inline]
    pub(crate) fn ysize(&self) -> usize {
        self.ysize
    }

    #[inline]
    pub(crate) fn row(&self, y: usize) -> &[T] {
        let w = self.xsize;
        &self.data[y * w..(y + 1) * w]
    }

    #[inline]
    pub(crate) fn row_mut(&mut self, y: usize) -> &mut [T] {
        let w = self.xsize;
        &mut self.data[y * w..(y + 1) * w]
    }

    /// Borrow one row immutably and a different one mutably. Useful for
    /// replicating a row to fill padding.
    #[inline]
    pub(crate) fn two_rows_mut_safe(&mut self, y_src: usize, y_dst: usize) -> (&[T], &mut [T]) {
        let w = self.xsize;
        debug_assert!(y_src != y_dst);
        if y_src < y_dst {
            let (a, b) = self.data.split_at_mut(y_dst * w);
            (&a[y_src * w..(y_src + 1) * w], &mut b[..w])
        } else {
            let (a, b) = self.data.split_at_mut(y_src * w);
            (&b[..w], &mut a[y_dst * w..(y_dst + 1) * w])
        }
    }
}

pub(crate) type ImageB = Plane<u8>;
pub(crate) type ImageSB = Plane<i8>;

#[derive(Clone)]
pub(crate) struct Image3<T> {
    planes: [Plane<T>; 3],
}

impl<T: Copy + Default> Image3<T> {
    pub(crate) fn new(xsize: usize, ysize: usize) -> Self {
        Self {
            planes: [
                Plane::new(xsize, ysize),
                Plane::new(xsize, ysize),
                Plane::new(xsize, ysize),
            ],
        }
    }

    #[inline]
    pub(crate) fn xsize(&self) -> usize {
        self.planes[0].xsize
    }

    #[inline]
    pub(crate) fn ysize(&self) -> usize {
        self.planes[0].ysize
    }

    #[inline]
    pub(crate) fn plane(&self, c: usize) -> &Plane<T> {
        &self.planes[c]
    }
    #[inline]
    pub(crate) fn plane_mut(&mut self, c: usize) -> &mut Plane<T> {
        &mut self.planes[c]
    }

    #[inline]
    pub(crate) fn plane_row(&self, c: usize, y: usize) -> &[T] {
        self.planes[c].row(y)
    }

    /// Whole plane as a contiguous row-major slice (length `xsize*ysize`).
    /// Lets hot inner loops index `data[y*w + x]` directly instead of paying a
    /// row-slice creation per access through `plane_row`.
    pub(crate) fn plane_data(&self, c: usize) -> &[T] {
        &self.planes[c].data
    }

    #[inline]
    pub(crate) fn plane_row_mut(&mut self, c: usize, y: usize) -> &mut [T] {
        self.planes[c].row_mut(y)
    }

    /// Borrow one row from each plane simultaneously (mutable).
    pub(crate) fn all_plane_rows_mut(&mut self, y: usize) -> [&mut [T]; 3] {
        let [p0, p1, p2] = &mut self.planes;
        [p0.row_mut(y), p1.row_mut(y), p2.row_mut(y)]
    }
}

pub(crate) type Image3B = Image3<u8>;
pub(crate) type Image3S = Image3<i16>;
pub(crate) type Image3Si = Image3<i32>;
pub(crate) type Image3F = Image3<f32>;

/// Axis-aligned rectangle within an image (coordinates in units of the image:
/// pixels, blocks, or tiles depending on context).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rect {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) xsize: usize,
    pub(crate) ysize: usize,
}

impl Rect {
    pub(crate) const fn new(x0: usize, y0: usize, xsize: usize, ysize: usize) -> Self {
        Self {
            x0,
            y0,
            xsize,
            ysize,
        }
    }
}
