/*
 * // Copyright (c) Radzivon Bartoshyk 8/2025. All rights reserved.
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

use super::common::f_fmla;

#[inline(always)]
pub(crate) fn f_polyeval3(x: f64, a0: f64, a1: f64, a2: f64) -> f64 {
    f_fmla(x, f_fmla(x, a2, a1), a0)
}

#[inline(always)]
pub(crate) fn f_polyeval4(x: f64, a0: f64, a1: f64, a2: f64, a3: f64) -> f64 {
    let t2 = f_fmla(x, a3, a2);
    let t5 = f_fmla(x, t2, a1);
    f_fmla(x, t5, a0)
}

#[inline(always)]
pub(crate) fn f_polyeval5(x: f64, a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64 {
    let mut acc = a4;
    acc = f_fmla(x, acc, a3);
    acc = f_fmla(x, acc, a2);
    acc = f_fmla(x, acc, a1);
    f_fmla(x, acc, a0)
}

#[inline(always)]
pub(crate) fn f_estrin_polyeval5(x: f64, a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64 {
    let x2 = x * x;
    let p01 = f_fmla(x, a1, a0);
    let p23 = f_fmla(x, a3, a2);
    let t = f_fmla(x2, a4, p23);
    f_fmla(x2, t, p01)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn d_polyeval3(x: f64, a0: f64, a1: f64, a2: f64) -> f64 {
    f64::mul_add(x, f64::mul_add(x, a2, a1), a0)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn d_polyeval4(x: f64, a0: f64, a1: f64, a2: f64, a3: f64) -> f64 {
    let t2 = f64::mul_add(x, a3, a2);
    let t5 = f64::mul_add(x, t2, a1);
    f64::mul_add(x, t5, a0)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn d_polyeval5(x: f64, a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64 {
    let mut acc = a4;
    acc = f64::mul_add(x, acc, a3);
    acc = f64::mul_add(x, acc, a2);
    acc = f64::mul_add(x, acc, a1);
    f64::mul_add(x, acc, a0)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn d_estrin_polyeval5(x: f64, a0: f64, a1: f64, a2: f64, a3: f64, a4: f64) -> f64 {
    let x2 = x * x;
    let p01 = f64::mul_add(x, a1, a0);
    let p23 = f64::mul_add(x, a3, a2);
    let t = f64::mul_add(x2, a4, p23);
    f64::mul_add(x2, t, p01)
}
