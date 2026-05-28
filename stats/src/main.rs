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
use plotters::prelude::*;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const FONT: &[u8] = include_bytes!("../../assets/DejaVuSans.ttf");

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
#[derive(Clone, Copy)]
struct Point {
    bpp: f64,
    bytes: u64,
    ss2: f64,
    distance: f32,
}

/// A labelled series of points (one encoder, or one cjxl effort).
struct Series {
    label: String,
    color: RGBColor,
    points: Vec<Point>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut images: Vec<PathBuf> = Vec::new();
    let mut out_dir = PathBuf::from("bench_out");
    let mut distances = vec![0.5f32, 1.0, 2.0, 3.0];
    let mut efforts = vec![7u32, 9];

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
            other => {
                images.push(PathBuf::from(other));
                i += 1;
            }
        }
    }
    if images.is_empty() {
        bail!(
            "usage: jixbench IMAGE.png [more.png ...] [--out DIR] [--distances 0.5,1,2] [--efforts 7,9]"
        );
    }
    register_fonts();
    std::fs::create_dir_all(&out_dir)?;
    check_tool("cjxl")?;
    check_tool("djxl")?;

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
            let p = bench_jixel(&orig_rgb, w, h, d, &orig_rgb, &tmp, stem, npx)?;
            println!(
                "  jixel       d={d:<4} {:>9} B  {:.3} bpp  SS2 {:.3}",
                p.bytes, p.bpp, p.ss2
            );
            jixel.points.push(p);
        }

        // cjxl series, one per effort.
        let palette = [
            RGBColor(0x2E, 0x7D, 0xE5),
            RGBColor(0x2E, 0xA8, 0x4E),
            RGBColor(0x9B, 0x5D, 0xE5),
            RGBColor(0xE5, 0x8A, 0x2E),
        ];
        let mut cjxl_series: Vec<Series> = Vec::new();
        for (k, &e) in efforts.iter().enumerate() {
            let mut s = Series {
                label: format!("cjxl -e{e}"),
                color: palette[k % palette.len()],
                points: vec![],
            };
            for &d in &distances {
                let p = bench_cjxl(img, e, d, &orig_rgb, w, h, &tmp, stem, npx)?;
                println!(
                    "  cjxl -e{e}    d={d:<4} {:>9} B  {:.3} bpp  SS2 {:.3}",
                    p.bytes, p.bpp, p.ss2
                );
                s.points.push(p);
            }
            cjxl_series.push(s);
        }

        let mut all = vec![jixel];
        all.extend(cjxl_series);
        let chart_path = out_dir.join(format!("{stem}_rd.png"));
        draw_chart(&chart_path, &format!("{stem} — SSIMULACRA2 vs rate"), &all)?;
        println!("  chart -> {}", chart_path.display());
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("\nDone. Charts in {}", out_dir.display());
    Ok(())
}

/// Encode with jixel, decode with djxl, score.
fn bench_jixel(
    rgb: &[u8],
    w: usize,
    h: usize,
    d: f32,
    orig: &[u8],
    tmp: &Path,
    stem: &str,
    npx: f64,
) -> Result<Point> {
    let cfg = jixel::EncodeConfig::default().with_distance(d);
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
        distance: d,
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
    let _ = std::fs::remove_file(&jxl);
    let output = Command::new("cjxl")
        .arg(img)
        .arg(&jxl)
        .arg("-d")
        .arg(d.to_string())
        .arg("-e")
        .arg(effort.to_string())
        .arg("--lossless_jpeg=0")
        .arg("--quiet")
        .output()
        .context("running cjxl")?;

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
        distance: d,
    })
}

/// Decode a .jxl to interleaved RGB8 via djxl (through a temp PNG).
fn decode_to_rgb(jxl: &Path, tmp: &Path, w: usize, h: usize) -> Result<Vec<u8>> {
    let png = tmp.join(format!(
        "{}_dec.png",
        jxl.file_stem().unwrap().to_str().unwrap()
    ));
    let _ = std::fs::remove_file(&png);
    let status = Command::new("djxl")
        .arg(jxl)
        .arg(&png)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("running djxl")?;
    if !status.success() || !png.exists() {
        bail!("djxl failed for {}", jxl.display());
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
        // Distance label above each dot, offset slightly so it doesn't overlap the circle.
        let series_color = s.color;
        chart.draw_series(s.points.iter().map(|pt| {
            let label = if pt.distance.fract() == 0.0 {
                format!("-d {}", pt.distance as u32)
            } else {
                // Trim trailing zeros: 0.50 → "0.5"
                let tmp = format!("-d {:.2}", pt.distance);
                tmp.trim_end_matches('0').trim_end_matches('.').to_string()
            };
            let style = ("sans-serif", 14).into_font().color(&series_color);
            Text::new(label, (pt.bpp + 0.01, pt.ss2 + 0.4), style)
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
