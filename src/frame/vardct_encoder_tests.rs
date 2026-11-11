//! Repeated-alpha frame serialization and fallback regression coverage.
use super::*;

#[test]
fn repeated_alpha_race_preserves_every_incoming_bit_offset() {
    let (width, height) = (512, 256);
    let mut tile = [0u8; 32 * 32];
    let mut state = 0x81c9_e7abu32;
    for sample in &mut tile {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *sample = state as u8;
    }
    let alpha = AlphaPlane::U8(
        (0..height)
            .flat_map(|y| {
                let tile = &tile;
                (0..width).map(move |x| tile[(y % 32) * 32 + x % 32])
            })
            .collect(),
    );
    let mut image = Image3F::new(width, height);
    for y in 0..height {
        image.plane_row_mut(1, y).fill(0.4);
        image.plane_row_mut(2, y).fill(0.2);
    }
    for offset in 0..8 {
        let mut ctx = EncodingContext::new(
            crate::Speed::Slow,
            crate::Arbiter::Ssimulacra2,
            crate::xyb::XybMatrix::SPEC,
            1.5,
            1,
        );
        ctx.alpha_patches = true;
        ctx.extra_plane_tree = Some(crate::lossless::PlaneTreeBudget::FULL);
        ctx.extra_squeeze_distance = 6.0;
        ctx.extra_learned_trees = true;
        ctx.extra_context_trees = true;
        let tag = (1u64 << offset) - 1;
        let mut prefix = BitWriter::new();
        prefix.write(offset, tag);
        let mut baseline = prefix.clone();
        encode_frame_core_plain(
            &ctx,
            &mut CoderScratch::default(),
            1.5,
            image.clone(),
            Some(&alpha),
            &[0],
            VarDctFrameKind::Regular,
            #[cfg(feature = "splines")]
            None,
            #[cfg(feature = "splines")]
            false,
            GrainPass::Inherit,
            &mut baseline,
        )
        .unwrap();
        let mut candidate = prefix;
        encode_frame_core(
            &ctx,
            &mut CoderScratch::default(),
            1.5,
            image.clone(),
            Some(&alpha),
            &[0],
            VarDctFrameKind::Regular,
            #[cfg(feature = "splines")]
            None,
            #[cfg(feature = "splines")]
            false,
            GrainPass::Inherit,
            &mut candidate,
        )
        .unwrap();
        assert!(
            candidate.bits_written() <= baseline.bits_written(),
            "offset {offset}"
        );
        assert_eq!(candidate.bits_written() % 8, 0);
        let bytes = candidate.into_bytes();
        assert_eq!(u64::from(bytes[0]) & tag, tag, "offset {offset}");
    }
}

#[test]
fn function_field_race_preserves_offsets_and_exact_fallback() {
    let (width, height) = (513, 257);
    let alpha = AlphaPlane::U8(
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let x = x as f64 / (width - 1) as f64;
                    let y = y as f64 / (height - 1) as f64;
                    let s = ((x - 0.51) / 0.49).powi(2) + 0.55 * ((y - 0.47) / 0.53).powi(2);
                    (255.0 * (-3.367 * s).exp()).round_ties_even() as u8
                })
            })
            .collect(),
    );
    assert!(!crate::function_fields::fit(&alpha, width, height).is_empty());
    let mut saw_smaller_frame = false;
    for offset in 0..8 {
        let config = crate::EncodeConfig::default()
            .with_distance(1.5)
            .with_speed(crate::Speed::Slow)
            .with_function_fields(true)
            .with_num_threads(1);
        let ctx = crate::encode_image::lossy_context(
            &config,
            1.5,
            crate::xyb::XybMatrix::SPEC,
            width * height,
        );
        assert!(ctx.function_fields);
        let mut image = Image3F::new(width, height);
        let (x, y, b) = crate::xyb::rgb_to_xyb_pixel_f32(&ctx.xyb, 0.25, 0.5, 0.75);
        for row in 0..height {
            image.plane_row_mut(0, row).fill(x);
            image.plane_row_mut(1, row).fill(y);
            image.plane_row_mut(2, row).fill(b);
        }
        let tag = (1u64 << offset) - 1;
        let mut prefix = BitWriter::new();
        prefix.write(offset, tag);
        let mut baseline = prefix.clone();
        encode_frame_core_plain(
            &ctx,
            &mut CoderScratch::default(),
            1.5,
            image.clone(),
            Some(&alpha),
            &[0],
            VarDctFrameKind::Regular,
            #[cfg(feature = "splines")]
            None,
            #[cfg(feature = "splines")]
            false,
            GrainPass::Inherit,
            &mut baseline,
        )
        .unwrap();
        let mut candidate = prefix.clone();
        encode_frame_core(
            &ctx,
            &mut CoderScratch::default(),
            1.5,
            image.clone(),
            Some(&alpha),
            &[0],
            VarDctFrameKind::Regular,
            #[cfg(feature = "splines")]
            None,
            #[cfg(feature = "splines")]
            false,
            GrainPass::Inherit,
            &mut candidate,
        )
        .unwrap();
        assert!(
            candidate.bits_written() <= baseline.bits_written(),
            "offset {offset}"
        );
        saw_smaller_frame |= candidate.bits_written() < baseline.bits_written();
        assert_eq!(candidate.bits_written() % 8, 0);
        let bytes = candidate.into_bytes();
        assert_eq!(u64::from(bytes[0]) & tag, tag, "offset {offset}");

        // An exact-alpha request cannot substitute an approximate function,
        // even when the function would make a substantially smaller file.
        if offset == 0 || offset == 7 {
            let mut ctx = ctx;
            ctx.extra_squeeze_distance = 0.0;
            let mut exact = prefix.clone();
            encode_frame_core_plain(
                &ctx,
                &mut CoderScratch::default(),
                1.5,
                image.clone(),
                Some(&alpha),
                &[0],
                VarDctFrameKind::Regular,
                #[cfg(feature = "splines")]
                None,
                #[cfg(feature = "splines")]
                false,
                GrainPass::Inherit,
                &mut exact,
            )
            .unwrap();
            let mut candidate = prefix;
            encode_frame_core(
                &ctx,
                &mut CoderScratch::default(),
                1.5,
                image,
                Some(&alpha),
                &[0],
                VarDctFrameKind::Regular,
                #[cfg(feature = "splines")]
                None,
                #[cfg(feature = "splines")]
                false,
                GrainPass::Inherit,
                &mut candidate,
            )
            .unwrap();
            assert_eq!(
                candidate.into_bytes(),
                exact.into_bytes(),
                "exact offset {offset}"
            );
        }
    }
    assert!(saw_smaller_frame, "function candidate was never selected");
}
