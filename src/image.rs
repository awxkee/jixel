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

pub struct Plane<T> {
    data: Vec<T>,
    xsize: usize,
    ysize: usize,
}

impl<T: Copy + Default> Plane<T> {
    pub fn new(xsize: usize, ysize: usize) -> Self {
        Self {
            data: vec![T::default(); xsize * ysize],
            xsize,
            ysize,
        }
    }

    pub fn new_fill(xsize: usize, ysize: usize, val: T) -> Self {
        Self {
            data: vec![val; xsize * ysize],
            xsize,
            ysize,
        }
    }

    #[inline]
    pub fn xsize(&self) -> usize {
        self.xsize
    }
    #[inline]
    pub fn ysize(&self) -> usize {
        self.ysize
    }
    /// Stride in elements (== xsize in jixel since rows have no padding).

    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        let w = self.xsize;
        &self.data[y * w..(y + 1) * w]
    }

    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        let w = self.xsize;
        &mut self.data[y * w..(y + 1) * w]
    }

    /// Borrow one row immutably and a different one mutably. Useful for
    /// replicating a row to fill padding.
    #[inline]
    pub fn two_rows_mut_safe(&mut self, y_src: usize, y_dst: usize) -> (&[T], &mut [T]) {
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

pub type ImageB = Plane<u8>;
pub type ImageSB = Plane<i8>;

pub struct Image3<T> {
    planes: [Plane<T>; 3],
}

impl<T: Copy + Default> Image3<T> {
    pub fn new(xsize: usize, ysize: usize) -> Self {
        Self {
            planes: [
                Plane::new(xsize, ysize),
                Plane::new(xsize, ysize),
                Plane::new(xsize, ysize),
            ],
        }
    }

    #[inline]
    pub fn xsize(&self) -> usize {
        self.planes[0].xsize
    }

    #[inline]
    pub fn ysize(&self) -> usize {
        self.planes[0].ysize
    }

    #[inline]
    pub fn plane(&self, c: usize) -> &Plane<T> {
        &self.planes[c]
    }
    #[inline]
    pub fn plane_mut(&mut self, c: usize) -> &mut Plane<T> {
        &mut self.planes[c]
    }

    #[inline]
    pub fn plane_row(&self, c: usize, y: usize) -> &[T] {
        self.planes[c].row(y)
    }

    #[inline]
    pub fn plane_row_mut(&mut self, c: usize, y: usize) -> &mut [T] {
        self.planes[c].row_mut(y)
    }

    /// Borrow one row from each plane simultaneously (mutable).
    pub fn all_plane_rows_mut(&mut self, y: usize) -> [&mut [T]; 3] {
        let [p0, p1, p2] = &mut self.planes;
        [p0.row_mut(y), p1.row_mut(y), p2.row_mut(y)]
    }
}

pub type Image3B = Image3<u8>;
pub type Image3S = Image3<i16>;
pub type Image3Si = Image3<i32>;
pub type Image3F = Image3<f32>;

/// Axis-aligned rectangle within an image (coordinates in units of the image:
/// pixels, blocks, or tiles depending on context).
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub xsize: usize,
    pub ysize: usize,
}

impl Rect {
    pub const fn new(x0: usize, y0: usize, xsize: usize, ysize: usize) -> Self {
        Self {
            x0,
            y0,
            xsize,
            ysize,
        }
    }
}
