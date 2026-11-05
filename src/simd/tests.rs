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

#[allow(unused_macros)]
macro_rules! integer_contract {
    ($backend:ident, $feature:literal, $lanes:literal) => {
        mod $backend {
            use crate::simd::$backend::i32 as v;

            #[allow(unused_unsafe)]
            #[target_feature(enable = $feature)]
            pub(super) fn check() {
                let values = [i32::MIN, i32::MAX, -1, 0, 1, -65535, 65535, 0x2aaa_aaab];
                for offset in 0..values.len() {
                    let a: [i32; $lanes] = std::array::from_fn(|i| values[i % values.len()]);
                    let b: [i32; $lanes] =
                        std::array::from_fn(|i| values[(offset + i) % values.len()]);
                    // Each multiplier lane differs: the high-half product
                    // must work beyond squeeze's broadcast divisor constant.
                    let [av, bv] = [a, b].map(|v| unsafe { v::load(v.as_ptr()) });
                    let mut actual = [0; $lanes];
                    unsafe { v::store(actual.as_mut_ptr(), v::mulhi(av, bv)) };
                    assert_eq!(
                        actual,
                        std::array::from_fn(|i| {
                            ((i64::from(a[i]) * i64::from(b[i])) >> 32) as i32
                        })
                    );
                    unsafe {
                        v::store(
                            actual.as_mut_ptr(),
                            v::select(v::lt_unsigned(av, bv), av, bv),
                        )
                    };
                    assert_eq!(
                        actual,
                        std::array::from_fn(|i| {
                            if (a[i] as u32) < (b[i] as u32) {
                                a[i]
                            } else {
                                b[i]
                            }
                        })
                    );
                    assert_eq!(
                        v::bits_clear(av, bv),
                        a.into_iter().zip(b).all(|(a, b)| a & b == 0)
                    );
                }

                let pairs: [i32; 2 * $lanes] = std::array::from_fn(|i| values[i % values.len()]);
                let (a, b) = unsafe { v::load_pairs(pairs.as_ptr()) };
                let mut even = [0; $lanes];
                let mut odd = [0; $lanes];
                unsafe {
                    v::store(even.as_mut_ptr(), a);
                    v::store(odd.as_mut_ptr(), b);
                }
                assert_eq!(even, std::array::from_fn(|i| pairs[2 * i]));
                assert_eq!(odd, std::array::from_fn(|i| pairs[2 * i + 1]));
            }
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
integer_contract!(neon, "neon", 4);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
integer_contract!(sse, "sse4.1", 4);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
integer_contract!(avx, "avx2", 8);
#[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
integer_contract!(wasm, "simd128", 4);

#[test]
#[allow(unused_unsafe)]
fn integer_lanes_match_widened_scalar_math() {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    unsafe {
        neon::check()
    };
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::is_x86_feature_detected!("sse4.1") {
        unsafe { sse::check() };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { avx::check() };
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    unsafe {
        wasm::check()
    };
}
