/*
 * // Copyright (c) Radzivon Bartoshyk 7/2025. All rights reserved.
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

const M: [f64; 2] = [0., 1.];
const PI: f64 = f64::from_bits(0x400921fb54442d18);
const PI2: f64 = f64::from_bits(0x3ff921fb54442d18);
const PI2L: f64 = f64::from_bits(0x3c91a62633145c07);
static OFF: [f64; 8] = [0.0, PI2, PI, PI2, -0.0, -PI2, -PI, -PI2];
static OFFL: [f64; 8] = [0.0, PI2L, 2. * PI2L, PI2L, -0.0, -PI2L, -2. * PI2L, -PI2L];
static SGN: [f64; 2] = [1., -1.];

#[inline(always)]
fn atan2f_gen_impl<Q: Fn(f64, f64, f64) -> f64>(y: f32, x: f32, fma: Q) -> f32 {
    let tx = x.to_bits();
    let ty = y.to_bits();
    let ux = tx;
    let uy = ty;
    let ax = ux & 0x7fffffff;
    let ay = uy & 0x7fffffff;
    if ay >= (0xff << 23) || ax >= (0xff << 23) {
        // x or y is nan or inf
        /* we use x+y below so that the invalid exception is set
        for (x,y) = (qnan,snan) or (snan,qnan) */
        if ay > (0xff << 23) {
            return x + y;
        } // case y nan
        if ax > (0xff << 23) {
            return x + y;
        } // case x nan
        let yinf = ay == (0xff << 23);
        let xinf = ax == (0xff << 23);
        if yinf & xinf {
            return if (ux >> 31) != 0 {
                (f64::from_bits(0x4002d97c7f3321d2) * SGN[(uy >> 31) as usize]) as f32 // +/-3pi/4
            } else {
                (f64::from_bits(0x3fe921fb54442d18) * SGN[(uy >> 31) as usize]) as f32 // +/-pi/4
            };
        }
        if xinf {
            return if (ux >> 31) != 0 {
                (PI * SGN[(uy >> 31) as usize]) as f32
            } else {
                (0.0 * SGN[(uy >> 31) as usize]) as f32
            };
        }
        if yinf {
            return (PI2 * SGN[(uy >> 31) as usize]) as f32;
        }
    }
    if ay == 0 {
        if ax == 0 {
            let i = (uy >> 31)
                .wrapping_mul(4)
                .wrapping_add((ux >> 31).wrapping_mul(2));
            return if (ux >> 31) != 0 {
                (OFF[i as usize] + OFFL[i as usize]) as f32
            } else {
                OFF[i as usize] as f32
            };
        }
        if (ux >> 31) == 0 {
            return (0.0 * SGN[(uy >> 31) as usize]) as f32;
        }
    }
    let gt = (ay > ax) as usize;
    let i = (uy >> 31)
        .wrapping_mul(4)
        .wrapping_add((ux >> 31).wrapping_mul(2))
        .wrapping_add(gt as u32);

    let zx = x as f64;
    let zy = y as f64;
    let mut z = fma(M[gt], zx, M[1usize.wrapping_sub(gt)] * zy)
        / fma(M[gt], zy, M[1usize.wrapping_sub(gt)] * zx);
    // z = x/y if |y| > |x|, and z = y/x otherwise
    let mut r;

    let d = ax as i32 - ay as i32;
    if d < (27 << 23) && d > (-(27 << 23)) {
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        /* z2 cannot underflow, since for |y|=0x1p-149 and |x|=0x1.fffffep+127
        we get |z| > 2^-277 thus z2 > 2^-554, but z4 and z8 might underflow,
        which might give spurious underflow exceptions. */

        const CN: [u64; 7] = [
            0x3ff0000000000000,
            0x40040e0698f94c35,
            0x400248c5da347f0d,
            0x3fed873386572976,
            0x3fc46fa40b20f1d0,
            0x3f833f5e041eed0f,
            0x3f1546bbf28667c5,
        ];
        const CD: [u64; 7] = [
            0x3ff0000000000000,
            0x4006b8b143a3f6da,
            0x4008421201d18ed5,
            0x3ff8221d086914eb,
            0x3fd670657e3a07ba,
            0x3fa0f4951fd1e72d,
            0x3f4b3874b8798286,
        ];

        let mut cn0 = fma(z2, f64::from_bits(CN[1]), f64::from_bits(CN[0]));
        let cn2 = fma(z2, f64::from_bits(CN[3]), f64::from_bits(CN[2]));
        let mut cn4 = fma(z2, f64::from_bits(CN[5]), f64::from_bits(CN[4]));
        let cn6 = f64::from_bits(CN[6]);
        cn0 = fma(z4, cn2, cn0);
        cn4 = fma(z4, cn6, cn4);
        cn0 = fma(z8, cn4, cn0);
        let mut cd0 = fma(z2, f64::from_bits(CD[1]), f64::from_bits(CD[0]));
        let cd2 = fma(z2, f64::from_bits(CD[3]), f64::from_bits(CD[2]));
        let mut cd4 = fma(z2, f64::from_bits(CD[5]), f64::from_bits(CD[4]));
        let cd6 = f64::from_bits(CD[6]);
        cd0 = fma(z4, cd2, cd0);
        cd4 = fma(z4, cd6, cd4);
        cd0 = fma(z8, cd4, cd0);
        r = cn0 / cd0;
    } else {
        r = 1.;
    }
    z *= SGN[gt];
    r = fma(z, r, OFF[i as usize]);
    r as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn atan2f_fma_impl(y: f32, x: f32) -> f32 {
    atan2f_gen_impl(y, x, f64::mul_add)
}

/// Computes atan2
///
/// Max found ULP 0.49999842
#[inline]
pub(crate) fn f_atan2f(y: f32, x: f32) -> f32 {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        atan2f_gen_impl(y, x, f_fmla)
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use std::sync::OnceLock;
        static EXECUTOR: OnceLock<unsafe fn(f32, f32) -> f32> = OnceLock::new();
        let q = EXECUTOR.get_or_init(|| {
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("fma")
            {
                atan2f_fma_impl
            } else {
                fn def_atan2f(y: f32, x: f32) -> f32 {
                    atan2f_gen_impl(y, x, f_fmla)
                }
                def_atan2f
            }
        });
        // SAFETY: the AVX/FMA implementation is selected only after checking
        // both features; the fallback requires no additional CPU features.
        unsafe { q(y, x) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_atan2_test() {
        assert_eq!(
            f_atan2f(
                0.000000000000000000000000000000000000017907922,
                170141180000000000000000000000000000000.
            ),
            0.
        );
        assert_eq!(f_atan2f(-3590000000., -15437000.), -1.5750962);
        assert_eq!(f_atan2f(-5., 2.), -1.19029);
    }
}
