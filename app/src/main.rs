//! CLI front-end: reads a PPM (P6) file and writes a JPEG XL.
//!
//! ```text
//! jxl-mini input.ppm output.jxl
//! ```

#![forbid(unsafe_code)]

use image::codecs::png::PngEncoder;
use jixel::srgb_to_linear_u8;
use jxl_encoder::quality_to_distance;
use std::path::Path;
use std::time::Instant;

fn main() {
    let output = "encoded_lossy2.jxl";
    let image = image::open(Path::new("./assets/abstract_alpha.png")).unwrap();
    // for i in 0..10 {
    //     let instant = Instant::now();
    //     let bytes = jixel::encode_image(
    //         image.to_rgb8().as_raw(),
    //         image.width() as usize,
    //         image.height() as usize,
    //         0.01,
    //     );
    //     println!("Encoded in {}ms", instant.elapsed().as_millis());
    // }
    let bytes = jixel::encode_image_with_alpha(
        image.to_rgba8().as_raw(),
        image.width() as usize,
        image.height() as usize,
        quality_to_distance(99.),
    );
    std::fs::write(&output, &bytes).expect("failed to write output");
}
