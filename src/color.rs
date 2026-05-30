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

fn srgb_to_linear_u8_ref(v: u8) -> f32 {
    let v = v as f64 * (1. / 255.0);
    (if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }) as f32
}

use std::sync::OnceLock;

fn lut() -> &'static [f32; 256] {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut tab = [0.0f32; 256];
        for (i, slot) in tab.iter_mut().enumerate() {
            *slot = srgb_to_linear_u8_ref(i as u8);
        }
        tab
    })
}

#[inline]
pub(crate) fn srgb_to_linear_u8(v: u8) -> f32 {
    lut()[v as usize]
}

pub(crate) struct LutHighBit {
    pub(crate) table: Box<[f32; 65536]>,
    _max: f32,
}

impl LutHighBit {
    fn new(bits: u8) -> Self {
        let size = 1usize << bits;
        let max = (size - 1) as f32;
        let mut table = Box::new([0f32; 65536]);
        for (i, dst) in table[..size].iter_mut().enumerate() {
            *dst = srgb_to_linear_u16(i as u16, max);
        }
        Self { table, _max: max }
    }
}

#[inline]
pub(crate) fn srgb_to_linear_f32(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB transfer function: u16 (0..=max) → linear-light f32.
///
/// `max` is `(1 << bits) - 1`, e.g. 1023 for 10-bit, 4095 for 12-bit.
/// No LUT — these depths are rarer and a LUT would be large.
#[inline]
pub(crate) fn srgb_to_linear_u16(v: u16, max: f32) -> f32 {
    let v = v as f32 / max;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn lut_high_bit(bits: u8) -> &'static LutHighBit {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<u8, &'static LutHighBit>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    map.entry(bits)
        .or_insert_with(|| Box::leak(Box::new(LutHighBit::new(bits))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_reference() {
        for i in 0..=255u8 {
            let r = srgb_to_linear_u8_ref(i);
            let l = srgb_to_linear_u8(i);
            assert_eq!(
                r.to_bits(),
                l.to_bits(),
                "srgb LUT differs from reference at {i}: ref={r} lut={l}"
            );
        }
    }
}
