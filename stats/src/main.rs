/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use anyhow::{Context, Result, bail};
use jixel::Speed;
use plotters::prelude::*;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::available_parallelism;

const FONT: &[u8] = include_bytes!("../../assets/DejaVuSans.ttf");

// --- AV1 reference: SYSTEM libavif (aom) for both encode and decode (on PATH). ---
const SYS_AVIFENC: &str = "avifenc";
const SYS_AVIFDEC: &str = "avifdec";
// --- maroontree AV2: the `maroontree-cli` binary for encode (`-e av2`) ...
const MAROONTREE_CLI: &str =
    "/Users/radzivon/RustroverProjects/maroontree/target/release/maroontree-cli";
const AVM_AVIFDEC: &str = "avifdec"; //"/Users/radzivon/RustroverProjects/maroontree/libavif-main/build/avifdec";
const AVM_AVIFENC: &str = "/Users/radzivon/RustroverProjects/maroontree/libavif-main/build/avifenc";

fn register_fonts() {
    use plotters::style::FontStyle;
    for style in [
        FontStyle::Normal,
        FontStyle::Bold,
        FontStyle::Italic,
        FontStyle::Oblique,
    ] {
        let _ = plotters::style::register_font("sans-serif", style, FONT);
    }
}

/// One measured (rate, quality) point.
#[derive(Clone)]
struct Point {
    bpp: f64,
    bytes: u64,
    ss2: f64,
    /// Per-point annotation drawn on the chart (e.g. "-d 1" or "q90").
    note: String,
}

/// A labeled series of points (one encoder, or one cjxl effort).
struct Series {
    label: String,
    color: RGBColor,
    points: Vec<Point>,
}

/// External AVIF tool configuration. Three independent pipelines:
///   * aom (AV1)  : system `avifenc` encode  -> system `avifdec` decode
///   * maroontree : `avif -e av2` encode     -> avm-capable `avifdec` decode
///   * avm enc    : avm `avifenc -c avm`     -> avm-capable `avifdec` decode
struct AvifTools {
    // AV1 reference (system libavif + aom)
    aom_enc: String,
    aom_dec: String,
    aom_speed: String, // avifenc -s (0 slow .. 10 fast)
    // maroontree AV2
    mt_cli: String,
    avm_dec: String,
    av2_speed: String, // maroontree -s (slow|medium|fast)
    // avm avifenc AV2 (libavif built against libaom2/avm)
    avm_enc: String,
    avm_enc_speed: String, // avifenc -s (default "9")
    // image crate AVIF (ravif/rav1e, in-process)
    image_avif_speed: u8, // rav1e speed (1 slow .. 10 fast)
    // shared chroma (avifenc -y / maroontree -c)
    // match jixel/cjxl; switch to 420 for the typical web config.
    yuv: String,
}

#[allow(clippy::too_many_arguments)]
fn bench_maroontree_vvc(
    img: &Path,
    d: f32,
    orig: &[u8],
    w: usize,
    h: usize,
    tmp: &Path,
    stem: &str,
    npx: f64,
    t: &AvifTools,
    nthreads: usize,
) -> Result<Point> {
    let q = distance_to_quality(d);
    let out = tmp.join(format!("{stem}_vvc_q{q}.heif"));
    let dec_png = tmp.join(format!("{stem}_vvc_q{q}_dec.png"));
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&dec_png);

    // encode
    let output = Command::new(&t.mt_cli)
        .arg("-e")
        .arg("vvc")
        .arg("-q")
        .arg(q.to_string())
        .arg("-c")
        .arg(&t.yuv)
        .arg("-t")
        .arg(nthreads.to_string())
        .arg(img)
        .arg(&out)
        .output()
        .context("running maroontree (vvc encode)")?;
    if !output.status.success() || !out.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "maroontree vvc encode failed for {} d={d} q={q}\nstdout: {stdout}\nstderr: {stderr}",
            img.display()
        );
    }

    // decode
    let output = Command::new(&t.mt_cli)
        .arg("-e")
        .arg("vvc")
        .arg(&out)
        .arg(&dec_png)
        .output()
        .context("running maroontree (vvc decode)")?;
    if !output.status.success() || !dec_png.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "maroontree vvc decode failed for {}\nstdout: {stdout}\nstderr: {stderr}",
            out.display()
        );
    }

    let bytes = std::fs::metadata(&out)?.len();
    let (dec_rgb, dw, dh) = load_rgb(&dec_png)?;
    if dw != w || dh != h {
        bail!("vvc decoded size {dw}x{dh} != {w}x{h}");
    }
    let ss2 = score(orig, &dec_rgb, w, h)?;
    Ok(Point {
        bpp: bytes as f64 * 8.0 / npx,
        bytes,
        ss2,
        note: format!("q{q}"),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut images: Vec<PathBuf> = Vec::new();
    let mut out_dir = PathBuf::from("bench_out");
    let mut distances = vec![0.5f32, 1.0, 2.0, 3.0];
    let mut efforts = vec![7u32, 9];
    let mut tools = AvifTools {
        aom_enc: SYS_AVIFENC.to_string(),
        aom_dec: SYS_AVIFDEC.to_string(),
        aom_speed: "6".to_string(),
        mt_cli: MAROONTREE_CLI.to_string(),
        avm_dec: AVM_AVIFDEC.to_string(),
        av2_speed: "slow".to_string(),
        avm_enc: AVM_AVIFENC.to_string(),
        avm_enc_speed: "6".to_string(),
        image_avif_speed: 6,
        yuv: "444".to_string(),
    };
    let mut with_aom = true;
    let mut with_av2 = true;
    let mut with_vvc = true;
    let mut with_avm_enc = true;
    let mut with_image_avif = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--distances" => {
                distances = parse_f32_list(&args[i + 1])?;
                i += 2;
            }
            "--efforts" => {
                efforts = parse_u32_list(&args[i + 1])?;
                i += 2;
            }
            "--avifenc" => {
                tools.aom_enc = args[i + 1].clone();
                i += 2;
            }
            "--avifdec" => {
                tools.aom_dec = args[i + 1].clone();
                i += 2;
            }
            "--maroontree" => {
                tools.mt_cli = args[i + 1].clone();
                i += 2;
            }
            "--avm-avifdec" => {
                tools.avm_dec = args[i + 1].clone();
                i += 2;
            }
            "--avm-avifenc" => {
                tools.avm_enc = args[i + 1].clone();
                i += 2;
            }
            "--aom-speed" => {
                tools.aom_speed = args[i + 1].clone();
                i += 2;
            }
            "--av2-speed" => {
                tools.av2_speed = args[i + 1].clone();
                i += 2;
            }
            "--avm-enc-speed" => {
                tools.avm_enc_speed = args[i + 1].clone();
                i += 2;
            }
            "--image-avif-speed" => {
                tools.image_avif_speed = args[i + 1]
                    .parse()
                    .context("bad --image-avif-speed (expected 1..=10)")?;
                i += 2;
            }
            "--avif-yuv" => {
                tools.yuv = args[i + 1].clone();
                i += 2;
            }
            "--no-aom" => {
                with_aom = false;
                i += 1;
            }
            "--no-av2" => {
                with_av2 = false;
                i += 1;
            }
            "--no-vvc" => {
                with_vvc = false;
                i += 1;
            }
            "--no-avm-enc" => {
                with_avm_enc = false;
                i += 1;
            }
            "--no-image-avif" => {
                with_image_avif = false;
                i += 1;
            }
            "--no-avif" => {
                with_aom = false;
                with_av2 = false;
                with_avm_enc = false;
                with_image_avif = false;
                i += 1;
            }
            other => {
                images.push(PathBuf::from(other));
                i += 1;
            }
        }
    }
    if images.is_empty() {
        bail!(
            "usage: stats IMAGE.png [more.png ...] [--out DIR] [--distances 0.5,1,2] \
             [--efforts 7,9]\n  AVIF (aom AV1 reference, system libavif): \
             [--avifenc PATH] [--avifdec PATH] [--aom-speed 6] [--no-aom]\n  \
             AVIF (maroontree AV2): [--maroontree PATH] [--avm-avifdec PATH] \
             [--av2-speed slow|medium|fast] [--no-av2]\n  \
             AVIF (avm avifenc AV2): [--avm-avifenc PATH] [--avm-avifdec PATH] \
             [--avm-enc-speed 9] [--no-avm-enc]\n  \
             AVIF (image crate, ravif/rav1e): [--image-avif-speed 6] [--no-image-avif]\n  \
             shared: [--avif-yuv 444|420] [--no-avif]"
        );
    }
    register_fonts();
    std::fs::create_dir_all(&out_dir)?;
    check_tool("cjxl")?;
    check_tool("djxl")?;

    // Probe the optional AVIF tools; warn + skip a series rather than aborting the run.
    if with_aom && !(available(&tools.aom_enc) && available(&tools.aom_dec)) {
        eprintln!(
            "warning: system avifenc/avifdec not found ({} / {}); skipping the aom (AV1) series. \
             Override with --avifenc/--avifdec or pass --no-aom.",
            tools.aom_enc, tools.aom_dec
        );
        with_aom = false;
    }
    if with_av2 && !(available(&tools.mt_cli) && available(&tools.avm_dec)) {
        eprintln!(
            "warning: maroontree CLI or avm avifdec not found ({} / {}); skipping the AV2 series. \
             Override with --maroontree/--avm-avifdec or pass --no-av2.",
            tools.mt_cli, tools.avm_dec
        );
        with_av2 = false;
    }
    if with_vvc && !available(&tools.mt_cli) {
        eprintln!(
            "warning: maroontree CLI not found ({}); skipping VVC series. \
         Override with --maroontree or pass --no-vvc.",
            tools.mt_cli
        );
        with_vvc = false;
    }
    if with_image_avif && !available(&tools.aom_dec) {
        eprintln!(
            "warning: system avifdec not found ({}); skipping the image crate (ravif) series. \
             Override with --avifdec or pass --no-image-avif.",
            tools.aom_dec
        );
        with_image_avif = false;
    }
    if with_avm_enc && !(available(&tools.avm_enc) && available(&tools.avm_dec)) {
        eprintln!(
            "warning: avm avifenc or avm avifdec not found ({} / {}); skipping the avm-enc (AV2) series. \
             Override with --avm-avifenc/--avm-avifdec or pass --no-avm-enc.",
            tools.avm_enc, tools.avm_dec
        );
        with_avm_enc = false;
    }

    let nthreads = available_parallelism()
        .unwrap_or(NonZero::new(1).unwrap())
        .get();

    let tmp = out_dir.join("_tmp");
    std::fs::create_dir_all(&tmp)?;

    for img in &images {
        let stem = img.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        println!("\n=== {} ===", img.display());
        let (orig_rgb, w, h) = load_rgb(img)?;
        let npx = (w * h) as f64;

        // jixel series.
        let mut jixel = Series {
            label: "jixel".into(),
            color: RGBColor(0xE5, 0x3E, 0x3E),
            points: vec![],
        };
        for &d in &distances {
            let p = bench_jixel(&orig_rgb, w, h, d, &orig_rgb, &tmp, stem, npx, nthreads)?;
            println!(
                "  jixel            d={d:<4} {:>9} B  {:.3} bpp  SS2 {:.3}",
                p.bytes, p.bpp, p.ss2
            );
            jixel.points.push(p);
        }

        // cjxl series, one per effort.
        fn cjxl_color(k: usize, n: usize) -> RGBColor {
            let (r0, g0, b0) = (0x6B_i32, 0xAE, 0xD6); // medium-light blue
            let (r1, g1, b1) = (0x08_i32, 0x30, 0x6B); // deep navy
            let t = if n <= 1 {
                0.0
            } else {
                k as f64 / (n as f64 - 1.0)
            };
            let lerp = |a: i32, b: i32| (a as f64 + (b - a) as f64 * t).round() as u8;
            RGBColor(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
        }
        let mut cjxl_series: Vec<Series> = Vec::new();
        let n_eff = efforts.len();
        for (k, &e) in efforts.iter().enumerate() {
            let mut s = Series {
                label: format!("cjxl -e{e}"),
                color: cjxl_color(k, n_eff),
                points: vec![],
            };
            for &d in &distances {
                let p = bench_cjxl(img, e, d, &orig_rgb, w, h, &tmp, stem, npx)?;
                println!(
                    "  cjxl -e{e}         d={d:<4} {:>9} B  {:.3} bpp  SS2 {:.3}",
                    p.bytes, p.bpp, p.ss2
                );
                s.points.push(p);
            }
            cjxl_series.push(s);
        }

        let mut all = vec![jixel];
        all.extend(cjxl_series);

        // AV1 reference (system libavif + aom): encode `avifenc`, decode `avifdec`.
        if with_aom {
            let mut s = Series {
                label: "libavif aom (AV1)".into(),
                color: RGBColor(0xFF, 0x7F, 0x00),
                points: vec![],
            };
            for &d in &distances {
                let q = distance_to_quality(d);
                let p = bench_avif_aom(img, d, &orig_rgb, w, h, &tmp, stem, npx, &tools)?;
                println!(
                    "  libavif aom      d={d:<4}->q{q:<3} {:>9} B  {:.3} bpp  SS2 {:.3}",
                    p.bytes, p.bpp, p.ss2
                );
                s.points.push(p);
            }
            all.push(s);
        }

        // maroontree AV2: encode `avif -e av2`, decode avm-capable `avifdec`.
        if with_av2 {
            let mut s = Series {
                label: "maroontree (av1)".into(),
                color: RGBColor(0x98, 0x4E, 0xA3),
                points: vec![],
            };
            for &d in &distances {
                let q = distance_to_quality(d);
                let p =
                    bench_maroontree(img, d, &orig_rgb, w, h, &tmp, stem, npx, &tools, nthreads)?;
                println!(
                    "  maroontree av2   d={d:<4}->q{q:<3} {:>9} B  {:.3} bpp  SS2 {:.3}",
                    p.bytes, p.bpp, p.ss2
                );
                s.points.push(p);
            }
            all.push(s);
        }

        // image crate AVIF (ravif/rav1e): encode in-process, decode with system avifdec.
        if with_image_avif {
            let mut s = Series {
                label: "image crate ravif (AV1)".into(),
                color: RGBColor(0x4D, 0xAF, 0x4A),
                points: vec![],
            };
            for &d in &distances {
                let q = distance_to_quality(d);
                let p = bench_avif_image_crate(
                    &orig_rgb, w, h, d, &orig_rgb, &tmp, stem, npx, &tools, nthreads,
                )?;
                println!(
                    "  image ravif      d={d:<4}->q{q:<3} {:>9} B  {:.3} bpp  SS2 {:.3}",
                    p.bytes, p.bpp, p.ss2
                );
                s.points.push(p);
            }
            all.push(s);
        }

        // if with_vvc {
        //     let mut s = Series {
        //         label: "maroontree (VVC)".into(),
        //         color: RGBColor(0x4D, 0xAF, 0x4A), // green
        //         points: vec![],
        //     };
        //     for &d in &distances {
        //         let q = distance_to_quality(d);
        //         let p = bench_maroontree_vvc(
        //             img, d, &orig_rgb, w, h, &tmp, stem, npx, &tools, nthreads,
        //         )?;
        //         println!(
        //             "  maroontree vvc   d={d:<4}->q{q:<3} {:>9} B  {:.3} bpp  SS2 {:.3}",
        //             p.bytes, p.bpp, p.ss2
        //         );
        //         s.points.push(p);
        //     }
        //     all.push(s);
        // }

        // avm avifenc AV2: encode with avm-capable avifenc -c avm, decode with avm avifdec.
        // if with_avm_enc {
        //     let mut s = Series {
        //         label: "libavif avm (AV2)".into(),
        //         color: RGBColor(0xA6, 0x56, 0x28),
        //         points: vec![],
        //     };
        //     for &d in &distances {
        //         let q = distance_to_quality(d);
        //         let p = bench_avif_avm(img, d, &orig_rgb, w, h, &tmp, stem, npx, &tools)?;
        //         println!(
        //             "  libavif avm      d={d:<4}->q{q:<3} {:>9} B  {:.3} bpp  SS2 {:.3}",
        //             p.bytes, p.bpp, p.ss2
        //         );
        //         s.points.push(p);
        //     }
        //     all.push(s);
        // }

        let chart_path = out_dir.join(format!("{stem}_rd.png"));
        draw_chart(&chart_path, &format!("{stem} — SSIMULACRA2 vs rate"), &all)?;
        println!("  chart -> {}", chart_path.display());
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("\nDone. Charts in {}", out_dir.display());
    Ok(())
}

fn distance_to_quality(d: f32) -> u8 {
    const Q0: f32 = 100.0; // quality at distance 0
    const SLOPE: f32 = 10.0; // quality lost per unit of distance
    (Q0 - SLOPE * d).round().clamp(1.0, 100.0) as u8
}

/// Chart annotation for a distance-driven point ("-d 1", "-d 0.5").
fn dist_note(d: f32) -> String {
    if d.fract() == 0.0 {
        format!("-d {}", d as u32)
    } else {
        let t = format!("-d {:.2}", d);
        t.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Encode with jixel, decode with djxl, score.
#[allow(clippy::too_many_arguments)]
fn bench_jixel(
    rgb: &[u8],
    w: usize,
    h: usize,
    d: f32,
    orig: &[u8],
    tmp: &Path,
    stem: &str,
    npx: f64,
    nthreads: usize,
) -> Result<Point> {
    let cfg = jixel::EncodeConfig::default()
        .with_distance(d)
        .with_num_threads(nthreads)
        .with_speed(Speed::Slow);
    let data = jixel::encode_image(rgb, w, h, &cfg)
        .map_err(|e| anyhow::anyhow!("jixel encode failed: {e:?}"))?;
    let jxl = tmp.join(format!("{stem}_jixel_{d}.jxl"));
    std::fs::write(&jxl, &data)?;
    let bytes = data.len() as u64;
    let dec = decode_to_rgb(&jxl, tmp, w, h)?;
    let ss2 = score(orig, &dec, w, h)?;
    Ok(Point {
        bpp: bytes as f64 * 8.0 / npx,
        bytes,
        ss2,
        note: dist_note(d),
    })
}

/// Encode with cjxl at (effort, distance), decode with djxl, score.
fn bench_cjxl(
    img: &Path,
    effort: u32,
    d: f32,
    orig: &[u8],
    w: usize,
    h: usize,
    tmp: &Path,
    stem: &str,
    npx: f64,
) -> Result<Point> {
    let jxl = tmp.join(format!("{stem}_cjxl_e{effort}_{d}.jxl"));
    let ext = img.extension().and_then(|e| e.to_str()).unwrap_or("png");
    let _ = std::fs::remove_file(&jxl);
    let mut cmd = Command::new("cjxl");
    cmd.arg(img)
        .arg(&jxl)
        .arg("-d")
        .arg(d.to_string())
        .arg("-e")
        .arg(effort.to_string())
        .arg("--quiet");

    if ext != "png" {
        cmd.arg("--lossless_jpeg=0");
    }

    let output = cmd.output().context("running cjxl")?;
    if !output.status.success() || !jxl.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "cjxl failed for {} d={d} e={effort}\nstdout: {stdout}\nstderr: {stderr}",
            img.display()
        );
    }
    let bytes = std::fs::metadata(&jxl)?.len();
    let dec = decode_to_rgb(&jxl, tmp, w, h)?;
    let ss2 = score(orig, &dec, w, h)?;
    Ok(Point {
        bpp: bytes as f64 * 8.0 / npx,
        bytes,
        ss2,
        note: dist_note(d),
    })
}

/// AV1 reference: encode with system `avifenc -c aom -q <quality>` (mapped from
/// `d`), decode with system `avifdec`, score.
#[allow(clippy::too_many_arguments)]
fn bench_avif_aom(
    img: &Path,
    d: f32,
    orig: &[u8],
    w: usize,
    h: usize,
    tmp: &Path,
    stem: &str,
    npx: f64,
    t: &AvifTools,
) -> Result<Point> {
    let q = distance_to_quality(d);
    let out = tmp.join(format!("{stem}_aom_q{q}.avif"));
    let _ = std::fs::remove_file(&out);

    let output = Command::new(&t.aom_enc)
        .arg("-c")
        .arg("aom")
        .arg("-q")
        .arg(q.to_string())
        .arg("-y")
        .arg(&t.yuv)
        .arg("-s")
        .arg(&t.aom_speed)
        .arg("-j")
        .arg("all")
        .arg(img)
        .arg(&out)
        .output()
        .context("running avifenc (aom)")?;
    if !output.status.success() || !out.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "avifenc (aom) failed for {} d={d} q={q}\nstdout: {stdout}\nstderr: {stderr}",
            img.display()
        );
    }
    let bytes = std::fs::metadata(&out)?.len();
    let dec = decode_avif(&out, tmp, w, h, &t.aom_dec)?;
    let ss2 = score(orig, &dec, w, h)?;
    Ok(Point {
        bpp: bytes as f64 * 8.0 / npx,
        bytes,
        ss2,
        note: format!("q{q}"),
    })
}

/// maroontree AV2: encode with the `avif -e av2 -q <quality>` CLI (quality mapped
/// from `d`), decode with the avm-capable `avifdec`, score.
#[allow(clippy::too_many_arguments)]
fn bench_maroontree(
    img: &Path,
    d: f32,
    orig: &[u8],
    w: usize,
    h: usize,
    tmp: &Path,
    stem: &str,
    npx: f64,
    t: &AvifTools,
    nthreads: usize,
) -> Result<Point> {
    let q = distance_to_quality(d);
    let out = tmp.join(format!("{stem}_av2_q{q}.avif"));
    let _ = std::fs::remove_file(&out);

    let output = Command::new(&t.mt_cli)
        .arg("-e")
        .arg("av1")
        .arg("-q")
        .arg(q.to_string())
        .arg("-c")
        .arg(&t.yuv)
        .arg("-s")
        .arg(&t.av2_speed)
        .arg("-t")
        .arg(nthreads.to_string())
        .arg(img)
        .arg(&out)
        .output()
        .context("running maroontree (avif -e av2)")?;
    if !output.status.success() || !out.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "maroontree av2 failed for {} d={d} q={q}\nstdout: {stdout}\nstderr: {stderr}",
            img.display()
        );
    }
    let bytes = std::fs::metadata(&out)?.len();
    let dec = decode_avif(&out, tmp, w, h, &t.avm_dec)?;
    let ss2 = score(orig, &dec, w, h)?;
    Ok(Point {
        bpp: bytes as f64 * 8.0 / npx,
        bytes,
        ss2,
        note: format!("q{q}"),
    })
}

/// AV2 via libavif avm backend: encode with `avifenc -c avm -q <quality>` at
/// speed 9, decode with the avm-capable `avifdec`, score.
#[allow(clippy::too_many_arguments)]
fn bench_avif_avm(
    img: &Path,
    d: f32,
    orig: &[u8],
    w: usize,
    h: usize,
    tmp: &Path,
    stem: &str,
    npx: f64,
    t: &AvifTools,
) -> Result<Point> {
    let q = distance_to_quality(d);
    let out = tmp.join(format!("{stem}_avm_q{q}.avif"));
    let _ = std::fs::remove_file(&out);

    let output = Command::new(&t.avm_enc)
        .arg("-c")
        .arg("avm")
        .arg("-q")
        .arg(q.to_string())
        .arg("-y")
        .arg(&t.yuv)
        .arg("-s")
        .arg(&t.avm_enc_speed)
        .arg("-j")
        .arg("all")
        .arg(img)
        .arg(&out)
        .output()
        .context("running avifenc (avm)")?;
    if !output.status.success() || !out.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "avifenc (avm) failed for {} d={d} q={q}\nstdout: {stdout}\nstderr: {stderr}",
            img.display()
        );
    }
    let bytes = std::fs::metadata(&out)?.len();
    let dec = decode_avif(&out, tmp, w, h, &t.avm_dec)?;
    let ss2 = score(orig, &dec, w, h)?;
    Ok(Point {
        bpp: bytes as f64 * 8.0 / npx,
        bytes,
        ss2,
        note: format!("q{q}"),
    })
}

/// AVIF via the image crate (ravif/rav1e backend): encode RGB8 in-process with
/// `AvifEncoder` at (speed, quality mapped from `d`), decode with the system
/// `avifdec`, score. Note ravif does its own RGB->YUV conversion internally, so
/// the shared `--avif-yuv` setting does not apply to this series.
#[allow(clippy::too_many_arguments)]
fn bench_avif_image_crate(
    rgb: &[u8],
    w: usize,
    h: usize,
    d: f32,
    orig: &[u8],
    tmp: &Path,
    stem: &str,
    npx: f64,
    t: &AvifTools,
    nthreads: usize,
) -> Result<Point> {
    use image::ImageEncoder;
    use image::codecs::avif::AvifEncoder;

    let q = distance_to_quality(d);
    let out = tmp.join(format!("{stem}_imgravif_q{q}.avif"));
    let _ = std::fs::remove_file(&out);

    let mut buf = Vec::new();
    AvifEncoder::new_with_speed_quality(&mut buf, t.image_avif_speed, q)
        .with_num_threads(Some(nthreads))
        .write_image(rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8)
        .context("image crate avif encode")?;
    std::fs::write(&out, &buf)?;

    let bytes = buf.len() as u64;
    let dec = decode_avif(&out, tmp, w, h, &t.aom_dec)?;
    let ss2 = score(orig, &dec, w, h)?;
    Ok(Point {
        bpp: bytes as f64 * 8.0 / npx,
        bytes,
        ss2,
        note: format!("q{q}"),
    })
}

/// Decode a .jxl to interleaved RGB8 via djxl (through a temp PNG).
fn decode_to_rgb(jxl: &Path, tmp: &Path, w: usize, h: usize) -> Result<Vec<u8>> {
    let png = tmp.join(format!(
        "{}_dec.png",
        jxl.file_stem().unwrap().to_str().unwrap()
    ));
    let _ = std::fs::remove_file(&png);
    let output = Command::new("djxl")
        .arg(jxl)
        .arg(&png)
        .output()
        .context("running djxl")?;
    if !output.status.success() || !png.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "djxl failed for {}\nstdout: {stdout}\nstderr: {stderr}",
            jxl.display()
        );
    }
    let (rgb, dw, dh) = load_rgb(&png)?;
    if dw != w || dh != h {
        bail!("decoded size {dw}x{dh} != {w}x{h}");
    }
    Ok(rgb)
}

/// Decode an .avif to interleaved RGB8 via the given avifdec binary (through a
/// temp PNG). Pass the system avifdec for AV1, the avm-capable one for AV2.
fn decode_avif(avif: &Path, tmp: &Path, w: usize, h: usize, avifdec: &str) -> Result<Vec<u8>> {
    let png = tmp.join(format!(
        "{}_dec.png",
        avif.file_stem().unwrap().to_str().unwrap()
    ));
    let _ = std::fs::remove_file(&png);
    let output = Command::new(avifdec)
        .arg(avif)
        .arg(&png)
        .output()
        .with_context(|| format!("running avifdec ({avifdec})"))?;
    if !output.status.success() || !png.exists() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "avifdec ({avifdec}) failed for {}\nstdout: {stdout}\nstderr: {stderr}",
            avif.display()
        );
    }
    let (rgb, dw, dh) = load_rgb(&png)?;
    if dw != w || dh != h {
        bail!("decoded size {dw}x{dh} != {w}x{h}");
    }
    Ok(rgb)
}

/// SSIMULACRA2 between two interleaved RGB8 buffers.
fn score(orig: &[u8], dist: &[u8], w: usize, h: usize) -> Result<f64> {
    let to_rgb = |b: &[u8]| -> Result<Rgb> {
        let data: Vec<[f32; 3]> = b
            .chunks_exact(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect();
        Rgb::new(
            data,
            w,
            h,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .map_err(|e| anyhow::anyhow!("rgb build: {e}"))
    };
    compute_frame_ssimulacra2(to_rgb(orig)?, to_rgb(dist)?).context("ssimulacra2")
}

/// Load a PNG (or anything `image` reads) as interleaved RGB8.
fn load_rgb(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.into_raw(), w, h))
}

fn draw_chart(path: &Path, title: &str, series: &[Series]) -> Result<()> {
    let root = BitMapBackend::new(path, (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;
    let (xmin, xmax, ymin, ymax) = bounds(series);
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 26))
        .margin(16)
        .x_label_area_size(48)
        .y_label_area_size(56)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;
    chart
        .configure_mesh()
        .x_desc("rate (bits / pixel)")
        .y_desc("SSIMULACRA2 (higher = better)")
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 14))
        .draw()?;
    for s in series {
        let mut pts: Vec<(f64, f64)> = s.points.iter().map(|p| (p.bpp, p.ss2)).collect();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        chart
            .draw_series(LineSeries::new(pts.clone(), s.color.stroke_width(2)))?
            .label(&s.label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 22, y)], s.color.stroke_width(3))
            });
        chart.draw_series(
            pts.iter()
                .map(|&(x, y)| Circle::new((x, y), 4, s.color.filled())),
        )?;
        // Per-point annotation above each dot ("-d 1" for distance encoders,
        // "q90" for the quality-driven AVIF encoders), offset so it doesn't
        // overlap the circle.
        let series_color = s.color;
        chart.draw_series(s.points.iter().map(|pt| {
            let style = ("sans-serif", 14).into_font().color(&series_color);
            Text::new(pt.note.clone(), (pt.bpp + 0.01, pt.ss2 + 0.4), style)
        }))?;
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.3))
        .label_font(("sans-serif", 15))
        .draw()?;
    root.present()?;
    Ok(())
}

fn bounds(series: &[Series]) -> (f64, f64, f64, f64) {
    let (mut xmn, mut xmx, mut ymn, mut ymx) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for s in series {
        for p in &s.points {
            xmn = xmn.min(p.bpp);
            xmx = xmx.max(p.bpp);
            ymn = ymn.min(p.ss2);
            ymx = ymx.max(p.ss2);
        }
    }
    let xpad = (xmx - xmn) * 0.05 + 1e-6;
    let ypad = (ymx - ymn) * 0.08 + 1e-6;
    (
        xmn - xpad,
        xmx + xpad,
        (ymn - ypad).max(0.0),
        (ymx + ypad).min(100.0),
    )
}

fn parse_f32_list(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|x| x.trim().parse::<f32>().context("bad distance"))
        .collect()
}
fn parse_u32_list(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(|x| x.trim().parse::<u32>().context("bad effort"))
        .collect()
}

fn check_tool(name: &str) -> Result<()> {
    Command::new(name)
        .arg("--version")
        .output()
        .with_context(|| format!("`{name}` not found on PATH (needed for benchmarking)"))?;
    Ok(())
}

/// True if `bin` can be spawned (found on PATH or at the given path). Probes with
/// `--help`, which all three external tools accept; only spawn failure (binary
/// missing) matters, not the exit code.
fn available(bin: &str) -> bool {
    Command::new(bin).arg("--help").output().is_ok()
}
