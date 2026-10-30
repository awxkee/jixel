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

//! Checked storage for retained MA sample properties. Arithmetic and split
//! thresholds remain i32; a sample set widens when any property needs it.

use super::NUM_MA_PROPS;
use std::ops::Range;

pub(crate) trait MaProperty: Copy + Into<i32> + Send + Sync {
    fn checked_row(row: [i32; NUM_MA_PROPS]) -> Option<[Self; NUM_MA_PROPS]>;
}

impl MaProperty for i16 {
    #[inline]
    fn checked_row(row: [i32; NUM_MA_PROPS]) -> Option<[Self; NUM_MA_PROPS]> {
        // Adding 32768 maps precisely the signed-16 range to 0..=65535.
        // The OR reduction checks every property and can vectorize.
        let range = row
            .iter()
            .fold(0u32, |bits, &v| bits | (v as u32).wrapping_add(1 << 15));
        (range <= u16::MAX as u32).then(|| row.map(|v| v as i16))
    }
}

impl MaProperty for i32 {
    #[inline]
    fn checked_row(row: [i32; NUM_MA_PROPS]) -> Option<[Self; NUM_MA_PROPS]> {
        Some(row)
    }
}

/// Write directly into a typed sampling range. A failed narrow write leaves
/// the row untouched; the producer must retry the sample set in wide storage.
#[inline]
pub(crate) fn write_property_row<T: MaProperty>(
    out: &mut [T; NUM_MA_PROPS],
    row: [i32; NUM_MA_PROPS],
) -> bool {
    if let Some(row) = T::checked_row(row) {
        *out = row;
        true
    } else {
        false
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MaProperties {
    Narrow(Vec<[i16; NUM_MA_PROPS]>),
    Wide(Vec<[i32; NUM_MA_PROPS]>),
}

#[derive(Clone, Copy)]
pub(crate) enum MaPropertySlice<'a> {
    Narrow(&'a [[i16; NUM_MA_PROPS]]),
    Wide(&'a [[i32; NUM_MA_PROPS]]),
}

impl MaProperties {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::Narrow(Vec::with_capacity(capacity))
    }

    pub(crate) fn zeroed(len: usize) -> Self {
        Self::Narrow(vec![[0; NUM_MA_PROPS]; len])
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Narrow(rows) => rows.len(),
            Self::Wide(rows) => rows.len(),
        }
    }

    pub(crate) fn get(&self, index: usize) -> [i32; NUM_MA_PROPS] {
        match self {
            Self::Narrow(rows) => rows[index].map(i32::from),
            Self::Wide(rows) => rows[index],
        }
    }

    pub(crate) fn slice(&self, range: Range<usize>) -> MaPropertySlice<'_> {
        match self {
            Self::Narrow(rows) => MaPropertySlice::Narrow(&rows[range]),
            Self::Wide(rows) => MaPropertySlice::Wide(&rows[range]),
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = [i32; NUM_MA_PROPS]> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }

    fn widen(&mut self) {
        if let Self::Narrow(rows) = self {
            let mut wide = Vec::with_capacity(rows.capacity());
            wide.extend(rows.iter().map(|row| row.map(i32::from)));
            *self = Self::Wide(wide);
        }
    }

    pub(crate) fn push(&mut self, row: [i32; NUM_MA_PROPS]) {
        if let Self::Narrow(rows) = self {
            if let Some(row) = i16::checked_row(row) {
                rows.push(row);
                return;
            }
            self.widen();
        }
        let Self::Wide(rows) = self else {
            unreachable!()
        };
        rows.push(row);
    }

    /// Mutations such as lossy-modular property rescaling are checked too.
    pub(crate) fn map_rows(&mut self, mut f: impl FnMut(&mut [i32; NUM_MA_PROPS])) {
        for i in 0..self.len() {
            let mut row = self.get(i);
            f(&mut row);
            if let Self::Narrow(rows) = self {
                if write_property_row(&mut rows[i], row) {
                    continue;
                }
                self.widen();
            }
            let Self::Wide(rows) = self else {
                unreachable!()
            };
            rows[i] = row;
        }
    }

    pub(crate) fn select(&self, indices: &[u32]) -> Self {
        match self {
            Self::Narrow(rows) => Self::Narrow(indices.iter().map(|&i| rows[i as usize]).collect()),
            Self::Wide(rows) => Self::Wide(indices.iter().map(|&i| rows[i as usize]).collect()),
        }
    }

    pub(crate) fn sampled(&self, step: usize) -> Self {
        match self {
            Self::Narrow(rows) => Self::Narrow(rows.iter().step_by(step).copied().collect()),
            Self::Wide(rows) => Self::Wide(rows.iter().step_by(step).copied().collect()),
        }
    }

    pub(crate) fn copy_range(&self, range: Range<usize>) -> Self {
        match self {
            Self::Narrow(rows) => Self::Narrow(rows[range].to_vec()),
            Self::Wide(rows) => Self::Wide(rows[range].to_vec()),
        }
    }

    #[cfg(test)]
    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        match self {
            Self::Narrow(rows) => rows.swap(a, b),
            Self::Wide(rows) => rows.swap(a, b),
        }
    }

    #[cfg(test)]
    pub(crate) fn reverse(&mut self) {
        match self {
            Self::Narrow(rows) => rows.reverse(),
            Self::Wide(rows) => rows.reverse(),
        }
    }

    #[cfg(test)]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Narrow(rows) => rows.as_ptr().cast(),
            Self::Wide(rows) => rows.as_ptr().cast(),
        }
    }
}

impl FromIterator<[i32; NUM_MA_PROPS]> for MaProperties {
    fn from_iter<I: IntoIterator<Item = [i32; NUM_MA_PROPS]>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut rows = Self::with_capacity(iter.size_hint().0);
        for row in iter {
            rows.push(row);
        }
        rows
    }
}

// Resolve storage once per operation; hot scan/partition kernels are typed.
macro_rules! with_ma_properties {
    ($props:expr, |$rows:ident| $body:expr) => {
        match $props {
            $crate::ma_tree::MaProperties::Narrow($rows) => $body,
            $crate::ma_tree::MaProperties::Wide($rows) => $body,
        }
    };
}

pub(crate) use with_ma_properties;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_rows_cover_every_property_and_signed_boundary() {
        for column in 0..NUM_MA_PROPS {
            for value in [
                i32::MIN,
                -65536,
                -32769,
                -32768,
                -1,
                0,
                32767,
                32768,
                65535,
                i32::MAX,
            ] {
                let mut row = [0; NUM_MA_PROPS];
                row[column] = value;
                let mut out = [123i16; NUM_MA_PROPS];
                let fits = i16::try_from(value).is_ok();
                assert_eq!(write_property_row(&mut out, row), fits);
                if fits {
                    assert_eq!(out.map(i32::from), row);
                } else {
                    assert_eq!(out, [123; NUM_MA_PROPS]);
                }
                let mut wide = [0; NUM_MA_PROPS];
                assert!(write_property_row(&mut wide, row));
                assert_eq!(wide, row);
            }
        }
    }

    #[test]
    fn push_and_mutation_widen_without_changing_earlier_rows() {
        let mut rows = MaProperties::with_capacity(8);
        rows.push([i16::MIN as i32; NUM_MA_PROPS]);
        rows.push([i16::MAX as i32; NUM_MA_PROPS]);
        assert!(matches!(&rows, MaProperties::Narrow(_)));
        rows.push([32768; NUM_MA_PROPS]);
        assert!(matches!(&rows, MaProperties::Wide(_)));
        assert_eq!(rows.get(0), [i16::MIN as i32; NUM_MA_PROPS]);
        assert_eq!(rows.get(1), [i16::MAX as i32; NUM_MA_PROPS]);
        assert_eq!(rows.get(2), [32768; NUM_MA_PROPS]);
        rows.push([i32::MIN; NUM_MA_PROPS]);
        rows.push([i32::MAX; NUM_MA_PROPS]);
        assert_eq!(rows.get(3), [i32::MIN; NUM_MA_PROPS]);
        assert_eq!(rows.get(4), [i32::MAX; NUM_MA_PROPS]);

        let mut rows: MaProperties = [[1; NUM_MA_PROPS], [20000; NUM_MA_PROPS], [-7; NUM_MA_PROPS]]
            .into_iter()
            .collect();
        rows.map_rows(|row| row[23] *= 2);
        assert!(matches!(&rows, MaProperties::Wide(_)));
        assert_eq!(rows.get(0)[23], 2);
        assert_eq!(rows.get(1)[23], 40000);
        assert_eq!(rows.get(2)[23], -14);
        assert_eq!(rows.get(1)[22], 20000);
        assert_eq!(
            rows.select(&[2, 0, 2]).iter().collect::<Vec<_>>(),
            [rows.get(2), rows.get(0), rows.get(2)]
        );
    }
}
