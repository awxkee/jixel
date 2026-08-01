#![forbid(unsafe_code)]

use std::hint::black_box;
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::thread::available_parallelism;
use std::time::{Duration, Instant};

use jixel::{EncodeConfig, Speed};
use jpegxl_rs::ThreadsRunner;
use jpegxl_rs::encode::{EncoderResult, EncoderSpeed};

const DEFAULT_IMAGE: &str = "./assets/digital_art_portrait.jpg";
const DEFAULT_ITERATIONS: usize = 7;
const QUALITY: f32 = 75.0;

struct Measurement {
    elapsed: Duration,
    bytes: Vec<u8>,
}

fn encode_jixel(rgb: &[u8], width: usize, height: usize, threads: usize) -> Measurement {
    let config = EncodeConfig::default()
        .with_lossless(false)
        .with_quality(QUALITY)
        .with_progressive(false)
        .with_patches(false)
        .with_speed(Speed::Slow)
        .with_num_threads(threads);
    let start = Instant::now();
    let bytes = jixel::encode_image(black_box(rgb), width, height, &config)
        .expect("jixel failed to encode");
    Measurement {
        elapsed: start.elapsed(),
        bytes,
    }
}

fn encode_libjxl(rgb: &[u8], width: u32, height: u32, threads: usize) -> Measurement {
    // Construct the runner and encoder inside the timed operation because
    // jixel also constructs its worker pool and encoder context per call.
    let start = Instant::now();
    let runner =
        ThreadsRunner::new(None, Some(threads)).expect("libjxl failed to create its thread runner");
    let mut encoder = jpegxl_rs::encoder_builder()
        .lossless(false)
        .jpeg_quality(QUALITY)
        // libjxl effort 3: its fastest generally useful VarDCT preset.
        .speed(EncoderSpeed::Squirrel)
        .parallel_runner(&runner)
        .build()
        .expect("libjxl failed to create its encoder");
    let encoded: EncoderResult<u8> = encoder
        .encode::<u8, u8>(black_box(rgb), width, height)
        .expect("libjxl failed to encode");
    Measurement {
        elapsed: start.elapsed(),
        bytes: encoded.data,
    }
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn mean(samples: &[Duration]) -> Duration {
    let total = samples.iter().map(Duration::as_secs_f64).sum::<f64>();
    Duration::from_secs_f64(total / samples.len() as f64)
}

fn print_summary(name: &str, samples: &[Duration], output_size: usize, pixels: usize) {
    let median = median(samples);
    let mean = mean(samples);
    let min = *samples.iter().min().unwrap();
    let megapixels_per_second = pixels as f64 / 1_000_000.0 / median.as_secs_f64();
    println!(
        "{name:<8} median {:>8.2} ms  min {:>8.2} ms  mean {:>8.2} ms  \
         {:>7.1} MP/s  {:>10} bytes",
        median.as_secs_f64() * 1_000.0,
        min.as_secs_f64() * 1_000.0,
        mean.as_secs_f64() * 1_000.0,
        megapixels_per_second,
        output_size,
    );
}

fn parse_args() -> (PathBuf, usize) {
    let mut args = std::env::args().skip(1);
    let image = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_IMAGE));
    let iterations = args
        .next()
        .map(|arg| {
            arg.parse::<usize>()
                .expect("iterations must be a positive integer")
        })
        .unwrap_or(DEFAULT_ITERATIONS);
    assert!(iterations != 0, "iterations must be greater than zero");
    assert!(
        args.next().is_none(),
        "usage: cargo run -p app --release -- [image] [iterations]"
    );
    (image, iterations)
}

fn main() {
    let (image_path, iterations) = parse_args();
    println!("image path: {:?}", image_path);
    let image = image::open(Path::new(&image_path)).expect("failed to open benchmark image");
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = width as usize * height as usize;
    let threads = available_parallelism()
        .unwrap_or(NonZero::new(1).unwrap())
        .get();

    println!(
        "{}: {}x{}, quality {}, {} threads, {} measured iterations",
        image_path.display(),
        width,
        height,
        QUALITY,
        threads,
        iterations,
    );
    println!("jixel: Fast; libjxl: Falcon/effort 3");

    // Warm up dispatch, allocators, thread creation, and both codec libraries.
    black_box(encode_jixel(
        rgb.as_raw(),
        width as usize,
        height as usize,
        threads,
    ));
    black_box(encode_libjxl(rgb.as_raw(), width, height, threads));

    let mut jixel_times = Vec::with_capacity(iterations);
    let mut libjxl_times = Vec::with_capacity(iterations);
    let mut jixel_output = Vec::new();
    let mut libjxl_output = Vec::new();

    // Alternate order to reduce systematic thermal/frequency bias.
    for iteration in 0..iterations {
        let (jixel, libjxl) = if iteration % 2 == 0 {
            let jixel = encode_jixel(rgb.as_raw(), width as usize, height as usize, threads);
            let libjxl = encode_libjxl(rgb.as_raw(), width, height, threads);
            (jixel, libjxl)
        } else {
            let libjxl = encode_libjxl(rgb.as_raw(), width, height, threads);
            let jixel = encode_jixel(rgb.as_raw(), width as usize, height as usize, threads);
            (jixel, libjxl)
        };
        jixel_times.push(jixel.elapsed);
        libjxl_times.push(libjxl.elapsed);
        jixel_output = jixel.bytes;
        libjxl_output = libjxl.bytes;
    }

    println!();
    print_summary("jixel", &jixel_times, jixel_output.len(), pixels);
    print_summary("libjxl", &libjxl_times, libjxl_output.len(), pixels);
    let ratio = median(&libjxl_times).as_secs_f64() / median(&jixel_times).as_secs_f64();
    println!(
        "speed: jixel is {:.2}x {} than libjxl",
        if ratio >= 1.0 { ratio } else { 1.0 / ratio },
        if ratio >= 1.0 { "faster" } else { "slower" },
    );

    std::fs::write("benchmark_jixel.jxl", &jixel_output)
        .expect("failed to write benchmark_jixel.jxl");
    std::fs::write("benchmark_libjxl.jxl", &libjxl_output)
        .expect("failed to write benchmark_libjxl.jxl");
}
