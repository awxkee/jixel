/*
 * // Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

use std::cmp::Ordering;

#[inline]
fn ordering(less: bool) -> Ordering {
    if less {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// Stable order of `0..len` under `less`, the one stable sort body.
#[inline(never)]
fn stable_order(len: usize, less: &mut dyn FnMut(usize, usize) -> bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| ordering(less(a, b)));
    order
}

/// Rearranges `order.len()` elements of `size` bytes so that slot `i`
/// receives the element that was at `order[i]`.
///
/// # Safety
/// `base` must point to `order.len()` elements of `size` bytes and `order`
/// must be a permutation of their indices.
#[inline(never)]
unsafe fn permute(base: *mut u8, size: usize, order: &[usize]) {
    let bytes = order.len() * size;
    let mut moved = Vec::<u8>::with_capacity(bytes);
    let out = moved.as_mut_ptr();
    for (slot, &from) in order.iter().enumerate() {
        debug_assert!(from < order.len());
        unsafe { std::ptr::copy_nonoverlapping(base.add(from * size), out.add(slot * size), size) };
    }
    unsafe { std::ptr::copy_nonoverlapping(out, base, bytes) };
}

#[inline(never)]
fn unstable_by_dyn<T>(items: &mut [T], less: &mut dyn FnMut(&T, &T) -> bool) {
    items.sort_unstable_by(|a, b| ordering(less(a, b)));
}

fn stable_by_less<T>(items: &mut [T], mut less: impl FnMut(&T, &T) -> bool) {
    if items.len() < 2 || size_of::<T>() == 0 {
        return;
    }
    let order = {
        let view = &*items;
        // SAFETY: the sort only hands back indices below `view.len()`.
        stable_order(view.len(), &mut |a, b| unsafe {
            less(view.get_unchecked(a), view.get_unchecked(b))
        })
    };
    // SAFETY: `order` is a permutation of the slice's indices, so every
    // element is moved exactly once; nothing is dropped or duplicated.
    unsafe { permute(items.as_mut_ptr().cast(), size_of::<T>(), &order) };
}

pub(crate) trait SharedSort<T> {
    /// `sort_by` with a shared body.
    fn shared_sort_by(&mut self, compare: impl FnMut(&T, &T) -> Ordering);
    /// `sort_by_key` with a shared body.
    fn shared_sort_by_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K);
    /// `sort` with a shared body. It also replaces `sort_unstable` wherever
    /// equal elements are indistinguishable (integers, tuples and arrays of
    /// them), since both then produce the same slice.
    fn shared_sort(&mut self)
    where
        T: Ord;
    /// `sort_unstable_by` with one body per element type.
    fn shared_sort_unstable_by(&mut self, compare: impl FnMut(&T, &T) -> Ordering);
    /// `sort_unstable_by_key` with one body per element type.
    fn shared_sort_unstable_by_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K);
}

impl<T> SharedSort<T> for [T] {
    #[inline]
    fn shared_sort_by(&mut self, mut compare: impl FnMut(&T, &T) -> Ordering) {
        stable_by_less(self, |a, b| compare(a, b) == Ordering::Less);
    }

    #[inline]
    fn shared_sort_by_key<K: Ord>(&mut self, mut key: impl FnMut(&T) -> K) {
        stable_by_less(self, |a, b| key(a).lt(&key(b)));
    }

    #[inline]
    fn shared_sort(&mut self)
    where
        T: Ord,
    {
        stable_by_less(self, |a, b| a.lt(b));
    }

    #[inline]
    fn shared_sort_unstable_by(&mut self, mut compare: impl FnMut(&T, &T) -> Ordering) {
        unstable_by_dyn(self, &mut |a, b| compare(a, b) == Ordering::Less);
    }

    #[inline]
    fn shared_sort_unstable_by_key<K: Ord>(&mut self, mut key: impl FnMut(&T) -> K) {
        unstable_by_dyn(self, &mut |a, b| key(a).lt(&key(b)));
    }
}

#[cfg(test)]
mod tests {
    use super::SharedSort;

    fn noise(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn shared_sorts_match_the_standard_sorts() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for len in [0usize, 1, 2, 3, 7, 20, 21, 33, 64, 257, 1000, 5000] {
            for keys in [2u64, 16, 1 << 40] {
                let items: Vec<(u64, String, usize)> = (0..len)
                    .map(|i| (noise(&mut seed) % keys, format!("item{i}"), i))
                    .collect();

                let (mut expected, mut actual) = (items.clone(), items.clone());
                expected.sort_by_key(|a| std::cmp::Reverse(a.0));
                actual.shared_sort_by(|a, b| b.0.cmp(&a.0));
                assert_eq!(actual, expected, "sort_by len {len}");

                let (mut expected, mut actual) = (items.clone(), items.clone());
                expected.sort_by_key(|item| item.0);
                actual.shared_sort_by_key(|item| item.0);
                assert_eq!(actual, expected, "sort_by_key len {len}");

                let (mut expected, mut actual) = (items.clone(), items.clone());
                expected.sort();
                actual.shared_sort();
                assert_eq!(actual, expected, "sort len {len}");

                let raw: Vec<[u16; 3]> = items.iter().map(|item| [item.0 as u16, 7, 1]).collect();
                let (mut expected, mut actual) = (raw.clone(), raw);
                expected.sort_unstable();
                actual.shared_sort();
                assert_eq!(actual, expected, "sort_unstable len {len}");

                let order: Vec<usize> = (0..len).collect();
                let (mut expected, mut actual) = (order.clone(), order.clone());
                expected.sort_unstable_by(|&a, &b| items[a].0.cmp(&items[b].0));
                actual.shared_sort_unstable_by(|&a, &b| items[a].0.cmp(&items[b].0));
                assert_eq!(actual, expected, "sort_unstable_by len {len}");

                let (mut expected, mut actual) = (order.clone(), order);
                expected.sort_unstable_by_key(|&i| items[i].0);
                actual.shared_sort_unstable_by_key(|&i| items[i].0);
                assert_eq!(actual, expected, "sort_unstable_by_key len {len}");
            }
        }
    }
}
