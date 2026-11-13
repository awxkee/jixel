//! Contiguous `f32` rows with a single scale per source row.

use std::sync::OnceLock;

pub(crate) type AddScaledFn = fn(&mut [f32], &[f32], f32);

/// Select a row kernel with separate multiply and add rounding.
pub(crate) fn selected_add_scaled_fn() -> AddScaledFn {
    static KERNEL: OnceLock<AddScaledFn> = OnceLock::new();
    *KERNEL.get_or_init(select_add_scaled_fn)
}

#[allow(unreachable_code)]
fn select_add_scaled_fn() -> AddScaledFn {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return |dst, src, scale| unsafe { super::avx::add_scaled_avx2(dst, src, scale) };
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        return |dst, src, scale| unsafe { super::sse::add_scaled_sse41(dst, src, scale) };
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the supported AArch64 target baseline.
    return |dst, src, scale| unsafe { super::neon_add_scaled::add_scaled_neon(dst, src, scale) };
    #[cfg(all(target_arch = "wasm32", feature = "wasm", target_feature = "simd128"))]
    return super::wasm::add_scaled_wasm;
    add_scaled_scalar
}

#[inline]
pub(crate) fn add_scaled_scalar(dst: &mut [f32], src: &[f32], scale: f32) {
    let src = &src[..dst.len()];
    for (out, &value) in dst.iter_mut().zip(src) {
        *out += scale * value;
    }
}
