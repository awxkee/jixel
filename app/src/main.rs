//! CLI front-end: reads a PPM (P6) file and writes a JPEG XL.
//!
//! ```text
//! jxl-mini input.ppm output.jxl
//! ```

#![forbid(unsafe_code)]

use image::imageops::FilterType;
use jixel::{ColorEncoding, EncodeConfig};
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() {
    let output = "encoded_lossy_b.jxl";
    let display_p3 = fs::read("./assets/Display P3.icc").unwrap();
    let image = image::open(Path::new("./assets/digital_art_portrait.jpg")).unwrap();
    let rgb_img = image.to_rgb8();
    let rgba_img = image.to_rgba8();
    // let src_rgb = rgb_img.as_raw();
    // for i in 0..10 {
    //     let instant = Instant::now();
    //     let d_bytes = jixel::encode_image_with_alpha(
    //         &rgba_img,
    //         image.width() as usize,
    //         image.height() as usize,
    //         &EncodeConfig::default()
    //             .with_lossless(true)
    //             .with_quality(99.)
    //             .with_icc_profile(display_p3.to_vec()),
    //     );
    //     println!("Encoded in {}ms", instant.elapsed().as_millis());
    // }
    // let img10 = image.to_rgb16().iter().map(|x| x >> 6).collect::<Vec<_>>();
    let bytes = jixel::encode_image(
        &rgb_img,
        image.width() as usize,
        image.height() as usize,
        &EncodeConfig::default()
            .with_lossless(false)
            .with_quality(15.)
            .with_color_encoding(ColorEncoding::srgb()),
    )
    .unwrap();
    std::fs::write(&output, &bytes).expect("failed to write output");
}
