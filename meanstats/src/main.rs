/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

//! `meanstats` — aggregate encoder quality over a *folder* of images.
//!
//! Where `stats` produces an R/D chart per single image, `meanstats` runs the
//! same encoders — jixel, `cjxl` at each effort, and (optionally) the libavif
//! aom AV1 reference — over every image in a folder at each requested
//! butteraugli distance, decodes, scores SSIMULACRA2, and reports the **folder
//! mean** rate (bits/pixel) and SSIMULACRA2 per series. It prints the per-series
//! means and draws one aggregate R/D chart (mean SS2 vs mean bpp), one line per
//! series — the corpus-level analogue of the per-image chart in `stats`.
//!
//! Usage:
//! ```text
//! meanstats FOLDER [--distances 0.5,1,2,3] [--efforts 7,9] [--threads N] [--out DIR]
//!                  [--avifenc PATH] [--avifdec PATH] [--aom-speed 6] [--avif-yuv 420]
//!                  [--no-aom] [--no-cjxl]
//! ```

use anyhow::{Context, Result, bail};
use plotters::prelude::*;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::available_parallelism;

const FONT: &[u8] = include_bytes!("../../assets/DejaVuSans.ttf");

/// Extensions we treat as input images (matched case-insensitively).
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg"];

// AV1 reference: system libavif (aom) for both encode and decode (on PATH).
const SYS_AVIFENC: &str = "avifenc";
const SYS_AVIFDEC: &str = "avifdec";

/// One measured (rate, quality) pair for a single image.
struct Sample {
    bpp: f64,
    ss2: f64,
}

/// A folder-mean R/D point for one series at one distance.
struct Point {
    bpp: f64,
    ss2: f64,
    /// Chart annotation ("-d 1", "q90"-style — here the butteraugli distance).
    note: String,
}

/// A labeled series (one encoder / cjxl effort), accumulated across distances.
struct Series {
    label: String,
    color: RGBColor,
    points: Vec<Point>,
}

/// Which encoder a series drives.
enum Kind {
    Jixel,
    Cjxl(u32),
    Aom,
}

/// System libavif (aom / AV1) reference tool configuration.
struct AvifTools {
    enc: String,
    dec: String,
    speed: String, // avifenc -s (0 slow .. 10 fast)
    yuv: String,   // avifenc -y (444 | 420)
}

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut folder: Option<PathBuf> = None;
    let mut distances = vec![0.5f32, 1.0, 2.0, 3.0];
    let mut efforts = vec![7u32, 9];
    let mut out_dir = PathBuf::from("meanstats_out");
    let mut threads = available_parallelism()
        .unwrap_or(NonZero::new(1).unwrap())
        .get();
    let mut tools = AvifTools {
        enc: SYS_AVIFENC.to_string(),
        dec: SYS_AVIFDEC.to_string(),
        speed: "6".to_string(),
        yuv: "420".to_string(),
    };
    let mut with_cjxl = true;
    let mut with_aom = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--distances" | "-d" => {
                distances = parse_f32_list(arg(&args, i + 1)?)?;
                i += 2;
            }
            "--efforts" => {
                efforts = parse_u32_list(arg(&args, i + 1)?)?;
                i += 2;
            }
            "--threads" | "-t" => {
                threads = arg(&args, i + 1)?.parse().context("bad --threads")?;
                i += 2;
            }
            "--out" => {
                out_dir = PathBuf::from(arg(&args, i + 1)?);
                i += 2;
            }
            "--avifenc" => {
                tools.enc = arg(&args, i + 1)?.to_string();
                i += 2;
            }
            "--avifdec" => {
                tools.dec = arg(&args, i + 1)?.to_string();
                i += 2;
            }
            "--aom-speed" => {
                tools.speed = arg(&args, i + 1)?.to_string();
                i += 2;
            }
            "--avif-yuv" => {
                tools.yuv = arg(&args, i + 1)?.to_string();
                i += 2;
            }
            "--no-cjxl" => {
                with_cjxl = false;
                i += 1;
            }
            "--no-aom" => {
                with_aom = false;
                i += 1;
            }
            "-h" | "--help" => usage(),
            other => {
                if folder.is_some() {
                    bail!("unexpected extra argument: {other}");
                }
                folder = Some(PathBuf::from(other));
                i += 1;
            }
        }
    }

    let folder = folder.unwrap_or_else(|| usage());
    if !folder.is_dir() {
        bail!("{} is not a directory", folder.display());
    }
    let threads = threads.max(1);

    check_tool("djxl")?;
    if with_cjxl {
        check_tool("cjxl")?;
    }
    // Probe the optional aom AVIF tools; warn + skip rather than aborting.
    if with_aom && !(available(&tools.enc) && available(&tools.dec)) {
        eprintln!(
            "warning: system avifenc/avifdec not found ({} / {}); skipping the aom (AV1) series. \
             Override with --avifenc/--avifdec or pass --no-aom.",
            tools.enc, tools.dec
        );
        with_aom = false;
    }

    let images = collect_images(&folder)?;
    if images.is_empty() {
        bail!(
            "no images ({}) found in {}",
            IMAGE_EXTS.join("/"),
            folder.display()
        );
    }
    register_fonts();
    std::fs::create_dir_all(&out_dir)?;
    let tmp = out_dir.join("_tmp");
    std::fs::create_dir_all(&tmp)?;

    println!(
        "Folder {}: {} image(s), {} thread(s)\n",
        folder.display(),
        images.len(),
        threads
    );

    // Build the series list once; colours mirror `stats`.
    let mut series: Vec<Series> = vec![Series {
        label: "jixel".into(),
        color: RGBColor(0xE5, 0x3E, 0x3E),
        points: vec![],
    }];
    let mut kinds: Vec<Kind> = vec![Kind::Jixel];
    if with_cjxl {
        let n_eff = efforts.len();
        for (k, &e) in efforts.iter().enumerate() {
            series.push(Series {
                label: format!("cjxl -e{e}"),
                color: cjxl_color(k, n_eff),
                points: vec![],
            });
            kinds.push(Kind::Cjxl(e));
        }
    }
    if with_aom {
        series.push(Series {
            label: "libavif aom (AV1)".into(),
            color: RGBColor(0xFF, 0x7F, 0x00),
            points: vec![],
        });
        kinds.push(Kind::Aom);
    }

    for &d in &distances {
        println!("=== distance d={d} ===");
        // Per-series sample buckets for this distance.
        let mut buckets: Vec<Vec<Sample>> = (0..series.len()).map(|_| Vec::new()).collect();

        for img in &images {
            let (rgb, w, h) = match load_rgb(img) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("  {:<32} SKIP (load): {e:#}", short_name(img));
                    continue;
                }
            };
            let npx = (w * h) as f64;
            let stem = img.file_stem().and_then(|s| s.to_str()).unwrap_or("image");

            for (idx, kind) in kinds.iter().enumerate() {
                let res = match kind {
                    Kind::Jixel => bench_jixel(&rgb, w, h, d, &tmp, stem, npx, threads),
                    Kind::Cjxl(e) => bench_cjxl(img, *e, d, &rgb, w, h, &tmp, stem, npx),
                    Kind::Aom => bench_avif_aom(img, d, &rgb, w, h, &tmp, stem, npx, &tools),
                };
                match res {
                    Ok(s) => buckets[idx].push(s),
                    Err(e) => eprintln!(
                        "  {:<32} {:<18} SKIP: {e:#}",
                        short_name(img),
                        series[idx].label
                    ),
                }
            }
        }

        // Aggregate each series for this distance: print the means and push the
        // R/D point (mean bpp, mean SS2) onto the chart series.
        for (idx, s) in series.iter_mut().enumerate() {
            let bucket = &buckets[idx];
            if bucket.is_empty() {
                println!("  {:<18} d={d}  no images scored", s.label);
                continue;
            }
            let bpp: Vec<f64> = bucket.iter().map(|p| p.bpp).collect();
            let ss2: Vec<f64> = bucket.iter().map(|p| p.ss2).collect();
            let mean_bpp = arith_mean(&bpp);
            let mean_ss2 = arith_mean(&ss2);
            println!(
                "  {:<18} n={:<3}  bpp[arith {:.4} geo {}]  SS2 {:.4}",
                s.label,
                bucket.len(),
                mean_bpp,
                fmt_geo(geo_mean(&bpp)),
                mean_ss2,
            );
            s.points.push(Point {
                bpp: mean_bpp,
                ss2: mean_ss2,
                note: dist_note(d),
            });
        }
        println!();
    }

    let name = folder
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("folder");
    let chart_path = out_dir.join(format!("{name}_mean_rd.png"));
    draw_chart(
        &chart_path,
        &format!(
            "{name} — mean SSIMULACRA2 vs rate ({} images)",
            images.len()
        ),
        &series,
    )?;
    println!("chart -> {}", chart_path.display());

    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

/// cjxl blue gradient (medium-light -> deep navy), matching `stats`.
fn cjxl_color(k: usize, n: usize) -> RGBColor {
    let (r0, g0, b0) = (0x6B_i32, 0xAE, 0xD6);
    let (r1, g1, b1) = (0x08_i32, 0x30, 0x6B);
    let t = if n <= 1 {
        0.0
    } else {
        k as f64 / (n as f64 - 1.0)
    };
    let lerp = |a: i32, b: i32| (a as f64 + (b - a) as f64 * t).round() as u8;
    RGBColor(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
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
    tmp: &Path,
    stem: &str,
    npx: f64,
    threads: usize,
) -> Result<Sample> {
    let cfg = jixel::EncodeConfig::default()
        .with_lossless(false)
        .with_distance(d)
        .with_num_threads(threads);
    let data = jixel::encode_image(rgb, w, h, &cfg)
        .map_err(|e| anyhow::anyhow!("jixel encode failed: {e:?}"))?;
    let jxl = tmp.join(format!("{stem}_jixel_{d}.jxl"));
    std::fs::write(&jxl, &data)?;
    let bytes = data.len() as u64;
    let dec = decode_to_rgb(&jxl, tmp, w, h)?;
    let ss2 = score(rgb, &dec, w, h)?;
    Ok(Sample {
        bpp: bytes as f64 * 8.0 / npx,
        ss2,
    })
}

/// Encode with cjxl at (effort, distance), decode with djxl, score.
#[allow(clippy::too_many_arguments)]
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
) -> Result<Sample> {
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
    Ok(Sample {
        bpp: bytes as f64 * 8.0 / npx,
        ss2,
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
) -> Result<Sample> {
    let q = distance_to_quality(d);
    let out = tmp.join(format!("{stem}_aom_q{q}.avif"));
    let _ = std::fs::remove_file(&out);

    let output = Command::new(&t.enc)
        .arg("-c")
        .arg("aom")
        .arg("-q")
        .arg(q.to_string())
        .arg("-y")
        .arg(&t.yuv)
        .arg("-s")
        .arg(&t.speed)
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
    let dec = decode_avif(&out, tmp, w, h, &t.dec)?;
    let ss2 = score(orig, &dec, w, h)?;
    Ok(Sample {
        bpp: bytes as f64 * 8.0 / npx,
        ss2,
    })
}

/// Map a JPEG XL butteraugli `distance` to an approximate AVIF quality (0–100,
/// higher = better). Rough monotonic heuristic; see the equivalent in `stats`.
fn distance_to_quality(d: f32) -> u8 {
    const Q0: f32 = 100.0;
    const SLOPE: f32 = 10.0;
    (Q0 - SLOPE * d).round().clamp(1.0, 100.0) as u8
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

/// Decode an .avif to interleaved RGB8 via the given avifdec binary (temp PNG).
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

/// Load a PNG/JPEG (or anything `image` reads) as interleaved RGB8.
fn load_rgb(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.into_raw(), w, h))
}

/// Non-recursively collect image files from `folder`, sorted by path.
fn collect_images(folder: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(folder).with_context(|| format!("reading {}", folder.display()))?
    {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false);
        if ext_ok {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Aggregate R/D chart: one line per series, folder-mean SS2 vs folder-mean bpp.
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
        .x_desc("mean rate (bits / pixel)")
        .y_desc("mean SSIMULACRA2 (higher = better)")
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
    // Guard against an all-empty chart (every series skipped).
    if xmn > xmx {
        return (0.0, 1.0, 0.0, 100.0);
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

/// Arithmetic mean of a non-empty slice.
fn arith_mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Geometric mean, `exp(mean(ln x))`. Requires strictly positive values; returns
/// `None` if any value is `<= 0`. Used for bpp only (always positive); SS2 is not
/// geometric-averaged since SSIMULACRA2 can be negative.
fn geo_mean(xs: &[f64]) -> Option<f64> {
    if xs.iter().any(|&x| x <= 0.0) {
        return None;
    }
    Some((xs.iter().map(|x| x.ln()).sum::<f64>() / xs.len() as f64).exp())
}

fn fmt_geo(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.4}"),
        None => "n/a".to_string(),
    }
}

fn short_name(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string()
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

fn arg(args: &[String], i: usize) -> Result<&str> {
    args.get(i)
        .map(|s| s.as_str())
        .context("missing argument value")
}

fn check_tool(name: &str) -> Result<()> {
    Command::new(name)
        .arg("--version")
        .output()
        .with_context(|| format!("`{name}` not found on PATH"))?;
    Ok(())
}

/// True if `bin` can be spawned (found on PATH or at the given path).
fn available(bin: &str) -> bool {
    Command::new(bin).arg("--help").output().is_ok()
}

fn usage() -> ! {
    eprintln!(
        "usage: meanstats FOLDER [--distances 0.5,1,2,3] [--efforts 7,9] [--threads N] [--out DIR]\n\
         \x20                [--avifenc PATH] [--avifdec PATH] [--aom-speed 6] [--avif-yuv 420]\n\
         \x20                [--no-aom] [--no-cjxl]\n\
         \n  Runs jixel, cjxl (per effort) and the libavif aom AV1 reference over every\n  \
         image in FOLDER at each distance, decodes, scores SSIMULACRA2, prints the folder\n  \
         mean bpp/SS2 per series, and writes an aggregate R/D chart to DIR."
    );
    std::process::exit(2);
}
