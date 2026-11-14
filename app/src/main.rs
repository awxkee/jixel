#![forbid(unsafe_code)]

use jixel::{CodingTransform, ColorEncoding, DecodingSpeed, EncodeConfig, Speed};
use std::num::NonZero;
use std::path::Path;
use std::thread::available_parallelism;

fn main() {
    let output = "encoded_lossy_b.jxl";
    let image = image::open(Path::new("./assets/cerys_no_alpha.png")).unwrap();
    let rgb_img = image.to_rgb8();
    let distance = 1.64;
    let width = image.width() as usize;
    let height = image.height() as usize;
    let cfg = EncodeConfig::default()
        .with_lossless(false)
        .with_distance(distance)
        .with_speed(Speed::Slow)
        .with_progressive(false)
        .with_decoding_speed(DecodingSpeed::Slow)
        .with_upsampling(false)
        .with_patches(true)
        .with_color_encoding(ColorEncoding::srgb())
        .with_coding_transform(CodingTransform::Xyb)
        .with_num_threads(
            available_parallelism()
                .unwrap_or(NonZero::new(1).unwrap())
                .get(),
        )
        .with_ink_layer(true)
        .with_vector_layers(true)
        .with_splines(false)
        .with_dots(true);
    let bytes = jixel::encode_image(&rgb_img, width, height, &cfg).unwrap();
    std::fs::write(output, &bytes).expect("failed to write output");
}
