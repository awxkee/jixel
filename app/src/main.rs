#![forbid(unsafe_code)]

// use image::imageops::FilterType;
use jixel::{ColorEncoding, DecodingSpeed, EncodeConfig, Speed};
use std::num::NonZero;
use std::path::Path;
use std::thread::available_parallelism;
use std::time::Instant;

fn main() {
    let output = "encoded_lossy_b.jxl";
    // let display_p3 = fs::read("./assets/Display P3.icc").unwrap();
    let image = image::open(Path::new("./assets/sun/pexels-anatoleos-35404883.jpg")).unwrap();
    let rgb_img = image.to_rgb8();
    // let rgba_img = image.to_rgba8();
    // let gray_img = image.to_luma8();
    // let src_rgb = rgb_img.as_raw();
    for _ in 0..4 {
        let instant = Instant::now();
        let _d_bytes = jixel::encode_image(
            &rgb_img,
            image.width() as usize,
            image.height() as usize,
            // ColorSpace::Rgb,
            // false,
            // &FlMeta::srgb(),
            &EncodeConfig::default()
                .with_lossless(false)
                .with_quality(90.)
                .with_progressive(false)
                .with_patches(false)
                .with_speed(Speed::Slow)
                .with_num_threads(
                    available_parallelism()
                        .unwrap_or(NonZero::new(1).unwrap())
                        .get(),
                ),
            // .with_icc_profile(display_p3.to_vec()),
        )
        .unwrap();
        println!("Encoded in {}ms", instant.elapsed().as_millis());
    }
    let width = image.width() as usize;
    let height = image.height() as usize;
    let bytes = jixel::encode_image(
        &rgb_img,
        width,
        height,
        &EncodeConfig::default()
            .with_lossless(false)
            .with_distance(1.0)
            .with_speed(Speed::Slow)
            .with_progressive(false)
            .with_decoding_speed(DecodingSpeed::Fast)
            .with_patches(false)
            .with_color_encoding(ColorEncoding::srgb()),
    )
    .unwrap();
    std::fs::write(output, &bytes).expect("failed to write output");
    // let width = 2000;
    // let height = 1000;
    // let img10 = vec![0u8; width * height * 3];
    // let bytes = jixel::encode_image(
    //     &img10,
    //     width,
    //     height,
    //     &EncodeConfig::default()
    //         .with_lossless(false)
    //         .with_quality(90.)
    //         .with_color_encoding(ColorEncoding::srgb()),
    // )
    // .unwrap();
    // std::fs::write(&output, &bytes).expect("failed to write output");
}
