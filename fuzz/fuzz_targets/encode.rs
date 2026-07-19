use jixel::{
    EncodeConfig, Speed, encode_image, encode_image_gray, encode_image_gray_alpha,
    encode_image_with_alpha,
};

const HEADER_LEN: usize = 4;
const MAX_DIMENSION: usize = 128;

fn exact_size_pixels(payload: &[u8], len: usize) -> Vec<u8> {
    let mut pixels = vec![0; len];
    if payload.is_empty() {
        return pixels;
    }

    for (index, pixel) in pixels.iter_mut().enumerate() {
        *pixel = payload[index % payload.len()];
    }
    pixels
}

fn fuzz_encode(data: &[u8]) {
    let Some(header) = data.get(..HEADER_LEN) else {
        return;
    };

    let format = header[0] & 3;
    let width = usize::from(header[1]) % MAX_DIMENSION + 1;
    let height = usize::from(header[2]) % MAX_DIMENSION + 1;
    let control = header[3];
    let channels = match format {
        0 => 3, // RGB
        1 => 4, // RGBA
        2 => 1, // grayscale
        _ => 2, // grayscale + alpha
    };

    println!("width {} height {}", width, height);

    let pixels = exact_size_pixels(&data[HEADER_LEN..], width * height * channels);
    let quality = f32::from(control >> 3) * (100.0 / 31.0);
    let config = EncodeConfig::default()
        .with_quality(quality)
        .with_lossless(control & 1 != 0)
        .with_progressive(control & 2 != 0)
        .with_speed(if control & 4 != 0 {
            Speed::Slow
        } else {
            Speed::Fast
        })
        .with_num_threads(1);

    let result = match format {
        0 => encode_image(&pixels, width, height, &config),
        1 => encode_image_with_alpha(&pixels, width, height, &config),
        2 => encode_image_gray(&pixels, width, height, &config),
        _ => encode_image_gray_alpha(&pixels, width, height, &config),
    };

    // Encoder errors are valid outcomes. Panics, aborts, and sanitizer findings
    // are left for AFL to report as crashes.
    _ = std::hint::black_box(result);
}

fn main() {
    afl::fuzz!(|data: &[u8]| {
        fuzz_encode(data);
    });
}
