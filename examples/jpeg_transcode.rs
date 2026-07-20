//! Losslessly transcodes a JPEG file to JPEG XL.
//!
//! The output is a JXL container whose codestream carries the JPEG's own DCT
//! coefficients, plus a `jbrd` box describing everything else about the source
//! file. Decoding it with `djxl out.jxl back.jpg` reproduces the original JPEG
//! byte for byte.
//!
//! ```text
//! cargo run --release --example jpeg_transcode -- photo.jpg photo.jxl
//! ```

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: jpeg_transcode <input.jpg> <output.jxl>");
        std::process::exit(2);
    }
    let (input, output) = (Path::new(&args[1]), Path::new(&args[2]));

    let jpeg = match std::fs::read(input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("could not read {}: {e}", input.display());
            std::process::exit(1);
        }
    };

    let jxl = match jixel::encode_jpeg_lossless(&jpeg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::write(output, &jxl) {
        eprintln!("could not write {}: {e}", output.display());
        std::process::exit(1);
    }

    let saved = 100.0 - (jxl.len() as f64 / jpeg.len() as f64) * 100.0;
    println!(
        "{} bytes -> {} bytes ({saved:.1}% smaller)",
        jpeg.len(),
        jxl.len()
    );
}
