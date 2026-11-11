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
