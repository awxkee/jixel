use jixel::{JpegTranscodeConfig, encode_jpeg_lossless_with_config};

const MAX_FUZZ_PIXELS: usize = 1 << 18;

fn declared_area(data: &[u8]) -> Option<usize> {
    let mut i = 2usize;
    while i + 9 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        // SOF0/1/2 are the only frame headers the transcoder accepts.
        if matches!(marker, 0xC0..=0xC2) {
            let height = usize::from(data[i + 5]) << 8 | usize::from(data[i + 6]);
            let width = usize::from(data[i + 7]) << 8 | usize::from(data[i + 8]);
            return Some(width.saturating_mul(height));
        }
        i += 2;
    }
    None
}

fn check_container(out: &[u8]) {
    const SIGNATURE: &[u8] = &[0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A];
    assert!(
        out.starts_with(SIGNATURE),
        "output does not begin with the JXL signature box"
    );

    let mut pos = 0usize;
    let mut seen_jbrd = false;
    let mut seen_codestream = false;

    while pos < out.len() {
        assert!(
            pos + 8 <= out.len(),
            "box header at {pos} runs past the end of the buffer"
        );
        let mut size = u32::from_be_bytes(out[pos..pos + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = out[pos + 4..pos + 8].try_into().unwrap();
        let mut header = 8usize;

        if size == 1 {
            assert!(pos + 16 <= out.len(), "largesize box header is truncated");
            size = u64::from_be_bytes(out[pos + 8..pos + 16].try_into().unwrap()) as usize;
            header = 16;
        }

        assert!(
            size >= header,
            "box {:?} declares size {size}, smaller than its own header",
            core::str::from_utf8(&kind)
        );
        assert!(
            pos + size <= out.len(),
            "box {:?} at {pos} declares size {size}, overrunning the {} byte buffer",
            core::str::from_utf8(&kind),
            out.len()
        );

        match &kind {
            b"jbrd" => seen_jbrd = true,
            b"jxlc" | b"jxlp" => seen_codestream = true,
            _ => {}
        }
        pos += size;
    }

    assert_eq!(pos, out.len(), "boxes do not tile the buffer exactly");
    assert!(seen_jbrd, "no jbrd box: the JPEG could not be reconstructed");
    assert!(seen_codestream, "no codestream box");
}

fn fuzz_transcode(data: &[u8]) {
    // Cheap rejections first, so the fuzzer spends its time in the parser
    // rather than in the allocator.
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return;
    }
    if declared_area(data).is_some_and(|area| area > MAX_FUZZ_PIXELS) {
        return;
    }

    // Both modes share a parser and a frame encoder but differ in whether
    // reconstruction data and a container are produced. Rejecting a malformed
    // or unsupported JPEG is a valid outcome.
    let config = JpegTranscodeConfig::default();
    if let Ok(out) = encode_jpeg_lossless_with_config(data, &config) {
        check_container(&out);
    }

    let bare = config.with_jpeg_reconstruction(false);
    if let Ok(out) = encode_jpeg_lossless_with_config(data, &bare) {
        // Without a container the output is a raw codestream.
        assert!(
            out.starts_with(&[0xFF, 0x0A]),
            "bare output does not begin with the JXL codestream signature"
        );
    }
}

fn main() {
    afl::fuzz!(|data: &[u8]| {
        fuzz_transcode(data);
    });
}
