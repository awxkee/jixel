//! `jixel-tuner` — a minimal encode-only CLI used by the offline parameter
//! optimizer in `parameters_fit/`.
//!
//! It loads an image, encodes it with jixel at a given butteraugli distance,
//! writes the `.jxl`, and prints one line of JSON describing the result:
//!
//! ```text
//! {"bytes":12345,"encode_ms":18.3,"width":768,"height":768,"pixels":589824,"bpp":0.1675}
//! ```
//!
//! Tuning parameters are supplied out-of-band via the `JIXEL_TUNING_JSON`
//! environment variable (read by the jixel crate itself), so this binary stays
//! trivial and never needs to know the parameter schema. Decoding and quality
//! scoring (djxl + ssimulacra2) are done by the Python harness.
//!
//! Usage:
//! ```text
//! jixel-tuner <input-image> <output.jxl> --distance <d> [--threads <n>]
//! ```

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: jixel-tuner <input-image> <output.jxl> --distance <d> [--threads <n>] [--boost <cfg>]\n\
         (--boost: `1`/`on` for the Dark-AQ preset, or the BoostCfg CSV; omitted = off)"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let mut distance: Option<f32> = None;
    let mut threads: usize = 1;
    let mut boost: Option<jixel::DarkAqConfig> = None;
    let mut speed = jixel::Speed::Fast;
    let mut lossless = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--distance" | "-d" => {
                let v = args.get(i + 1).unwrap_or_else(|| usage());
                distance = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("invalid --distance: {v}");
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--boost" => {
                let v = args.get(i + 1).unwrap_or_else(|| usage());
                boost = Some(jixel::DarkAqConfig::parse(v).unwrap_or_else(|| {
                    eprintln!("invalid --boost config: {v}");
                    std::process::exit(2);
                }));
                i += 2;
            }
            "--threads" | "-t" => {
                let v = args.get(i + 1).unwrap_or_else(|| usage());
                threads = v.parse().unwrap_or_else(|_| {
                    eprintln!("invalid --threads: {v}");
                    std::process::exit(2);
                });
                i += 2;
            }
            "--speed" => {
                let v = args.get(i + 1).unwrap_or_else(|| usage());
                speed = match v.as_str() {
                    "slow" => jixel::Speed::Slow,
                    "fast" => jixel::Speed::Fast,
                    _ => {
                        eprintln!("invalid --speed: {v}");
                        std::process::exit(2);
                    }
                };
                i += 2;
            }
            "--lossless" => {
                lossless = true;
                i += 1;
            }
            "-h" | "--help" => usage(),
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    if positional.len() != 2 {
        usage();
    }
    let input = &positional[0];
    let output = &positional[1];
    let distance = if lossless {
        // Ignored by the lossless path, but config validation wants it valid.
        distance.unwrap_or(1.0)
    } else {
        distance.unwrap_or_else(|| usage())
    };

    let image = match image::open(Path::new(input)) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("failed to open {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let width = image.width() as usize;
    let height = image.height() as usize;
    let rgb = image.to_rgb8();

    let mut cfg = jixel::EncodeConfig::default()
        .with_lossless(lossless)
        .with_distance(distance)
        .with_num_threads(threads.max(1))
        .with_speed(speed);
    if let Some(b) = boost {
        cfg = cfg.with_dark_aq_config(b);
    }

    let start = Instant::now();
    let data = match jixel::encode_image(rgb.as_raw(), width, height, &cfg) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("jixel encode failed: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    if let Err(e) = std::fs::write(output, &data) {
        eprintln!("failed to write {output}: {e}");
        return ExitCode::FAILURE;
    }

    let bytes = data.len() as u64;
    let pixels = (width * height) as f64;
    let bpp = (bytes as f64 * 8.0) / pixels;
    // Single flat JSON line on stdout — easy for the Python harness to parse.
    println!(
        "{{\"bytes\":{bytes},\"encode_ms\":{encode_ms:.3},\"width\":{width},\
         \"height\":{height},\"pixels\":{},\"bpp\":{bpp:.6}}}",
        width * height
    );
    ExitCode::SUCCESS
}
