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
//! same encoders — jixel, `cjxl` at each effort, (optionally) the libavif
//! aom AV1 reference, and (optionally, `--jpeg`) a classic JPEG reference
//! (cjpegli when available, else cjpeg/libjpeg-turbo, else the image crate)
//! — over every image in a folder at each requested
//! butteraugli distance, decodes, scores SSIMULACRA2 (plus the butteraugli
//! 3-norm whenever `butteraugli_main` is on PATH), and reports the **folder
//! mean** rate (bits/pixel) and each metric per series. It prints the per-series
//! means and draws one aggregate R/D chart (mean SS2 vs mean bpp), one line per
//! series — the corpus-level analogue of the per-image chart in `stats`.
//!
//! Usage:
//! ```text
//! meanstats FOLDER [--distances 0.5,1,2,3] [--efforts 7,9] [--threads N] [--out DIR]
//!                  [--avifenc PATH] [--avifdec PATH] [--aom-speed 6] [--avif-yuv 444]
//!                  [--no-aom] [--no-cjxl] [--patches] [--jpeg] [--cjpegli PATH]
//!                  [--no-butteraugli] [--butteraugli-bin PATH]
//! ```

use anyhow::{Context, Result, bail};
use jixel::Speed;
use plotters::prelude::*;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread::available_parallelism;

const FONT: &[u8] = include_bytes!("../../assets/DejaVuSans.ttf");

/// Extensions we treat as input images (matched case-insensitively).
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg"];

// AV1 reference: system libavif (aom) for both encode and decode (on PATH).
const SYS_AVIFENC: &str = "avifenc";
const SYS_AVIFDEC: &str = "avifdec";

// --- ColorVideoVDP: the `cvvdp` CLI from gfxdisp/ColorVideoVDP (pip package
// `pycvvdp`, installed from GitHub). Opt-in via `--cvvdp`. One `cvvdp -i`
// process stays alive for the whole run (torch starts once) and scores every
// series of an image in a single request against one reference. Inputs are
// 8-bit PPMs written from the very buffers SSIMULACRA2 scores: pycvvdp reads
// PNGs through imageio's FreeImage plugin, whose bundled dylib is x86_64-only,
// while any other extension goes through Pillow.
const CVVDP_BIN: &str = "cvvdp";
/// Fallback when `cvvdp` is not on PATH: the parameters_fit venv of this repo.
const CVVDP_VENV_BIN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../parameters_fit/.venv/bin/cvvdp"
);
/// Display model passed as `--display` (photometry + geometry the metric assumes).
const CVVDP_DISPLAY: &str = "standard_4k";

// --- butteraugli: libjxl's `butteraugli_main` (on PATH), run with --pnorm 3.
// On by default when the tool is there (it is cheap next to the encodes);
// disable with `--no-butteraugli`. It reads the same PPMs cvvdp is fed, so
// every metric scores exactly the buffers SSIMULACRA2 scored.
const BUTTERAUGLI_BIN: &str = "butteraugli_main";
/// p of the butteraugli p-norm reported alongside the max distance.
const BUTTERAUGLI_PNORM: u32 = 3;

/// One measured (rate, quality) pair for a single image.
struct Sample {
    bpp: f64,
    ss2: f64,
    /// butteraugli p-norm distance (lower = better). `None` when butteraugli
    /// scoring is off or failed for this sample.
    ba: Option<f64>,
    /// butteraugli max distance (worst region), lower = better.
    ba_max: Option<f64>,
    /// ColorVideoVDP quality in JOD (10 = identical, lower = worse). `None`
    /// when cvvdp scoring is off or failed for this sample.
    cvvdp: Option<f64>,
}

/// A folder-mean R/D point for one series at one distance.
struct Point {
    bpp: f64,
    ss2: f64,
    /// Folder-mean butteraugli p-norm / max distance; `None` when no sample of
    /// the bucket had a score.
    ba: Option<f64>,
    ba_max: Option<f64>,
    /// Folder-mean CVVDP JOD; `None` when no sample of the bucket had a score.
    cvvdp: Option<f64>,
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
    Jpeg,
}

/// How the optional JPEG reference series encodes (decode is always the
/// in-process `image` crate).
enum JpegTool {
    /// `cjpegli` (libjxl tools): butteraugli-distance native via `-d`.
    Cjpegli(String),
    /// libjpeg-turbo `cjpeg -quality N -optimize` fed through a temp PPM.
    Cjpeg,
    /// In-process `image`-crate baseline encoder (last resort).
    Builtin,
}

/// ColorVideoVDP scorer: a lazily spawned, persistent `cvvdp --interactive`
/// child that takes one argument line per request on stdin and answers with
/// one bare JOD per test file (`--quiet`). A child that dies mid-run is
/// dropped and respawned on the next request.
struct Cvvdp {
    bin: String,
    display: String,
    device: Option<String>,
    child: Option<CvvdpChild>,
}

struct CvvdpChild {
    proc: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Cvvdp {
    fn spawn_child(&self) -> Result<CvvdpChild> {
        let mut proc = Command::new(&self.bin)
            .arg("--interactive")
            // Python block-buffers stdout on a pipe; we need each answer now.
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning cvvdp ({})", self.bin))?;
        let stdin = proc.stdin.take().context("cvvdp stdin")?;
        let stdout = BufReader::new(proc.stdout.take().context("cvvdp stdout")?);
        Ok(CvvdpChild {
            proc,
            stdin,
            stdout,
        })
    }

    /// JOD of each `tests` image against `reference`, in order.
    fn score(&mut self, reference: &Path, tests: &[PathBuf]) -> Result<Vec<f64>> {
        if self.child.is_none() {
            self.child = Some(self.spawn_child()?);
        }
        let res = self.request(reference, tests);
        if res.is_err() {
            // Whatever happened, the stream is out of sync: start over next time.
            if let Some(mut c) = self.child.take() {
                let _ = c.proc.kill();
                let _ = c.proc.wait();
            }
        }
        res
    }

    fn request(&mut self, reference: &Path, tests: &[PathBuf]) -> Result<Vec<f64>> {
        let child = self.child.as_mut().context("cvvdp child missing")?;
        // The child splits the line with shlex: single-quote every path.
        let mut line = String::from("--test");
        for t in tests {
            line.push(' ');
            line.push_str(&shell_quote(t));
        }
        line.push_str(" --ref ");
        line.push_str(&shell_quote(reference));
        line.push_str(" --display ");
        line.push_str(&shell_quote(Path::new(&self.display)));
        if let Some(dev) = &self.device {
            line.push_str(" --device ");
            line.push_str(&shell_quote(Path::new(dev)));
        }
        line.push_str(" --quiet\n");
        child
            .stdin
            .write_all(line.as_bytes())
            .and_then(|_| child.stdin.flush())
            .context("writing to cvvdp stdin (process gone?)")?;

        let mut scores = Vec::with_capacity(tests.len());
        let mut buf = String::new();
        while scores.len() < tests.len() {
            buf.clear();
            let n = child
                .stdout
                .read_line(&mut buf)
                .context("reading cvvdp stdout")?;
            if n == 0 {
                let status = child.proc.wait().ok();
                bail!(
                    "cvvdp exited after {} of {} score(s) (status {status:?})",
                    scores.len(),
                    tests.len()
                );
            }
            match parse_cvvdp_line(buf.trim()) {
                Some(v) => scores.push(v),
                None => bail!("unexpected cvvdp output line: {:?}", buf.trim()),
            }
        }
        Ok(scores)
    }
}

impl Drop for Cvvdp {
    fn drop(&mut self) {
        if let Some(c) = self.child.take() {
            // Closing stdin ends the interactive loop; then reap.
            let CvvdpChild {
                mut proc, stdin, ..
            } = c;
            drop(stdin);
            let _ = proc.wait();
        }
    }
}

/// POSIX single-quote `p` for shlex.
fn shell_quote(p: &Path) -> String {
    format!("'{}'", p.to_string_lossy().replace('\'', "'\\''"))
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
        yuv: "444".to_string(),
    };
    let mut patches = false;
    let mut with_cjxl = true;
    let mut with_aom = true;
    let mut with_jpeg = false;
    let mut cjpegli = "cjpegli".to_string();
    let mut with_cvvdp = false;
    let mut cvvdp_bin: Option<String> = None;
    let mut with_butteraugli = true;
    let mut butteraugli_bin = BUTTERAUGLI_BIN.to_string();
    let mut cvvdp_display = CVVDP_DISPLAY.to_string();
    let mut cvvdp_device: Option<String> = None;

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
            "--patches" => {
                patches = true;
                i += 1;
            }
            "--no-cjxl" => {
                with_cjxl = false;
                i += 1;
            }
            "--no-aom" => {
                with_aom = false;
                i += 1;
            }
            "--jpeg" => {
                with_jpeg = true;
                i += 1;
            }
            "--cjpegli" => {
                cjpegli = arg(&args, i + 1)?.to_string();
                i += 2;
            }
            "--cvvdp" => {
                with_cvvdp = true;
                i += 1;
            }
            "--no-cvvdp" => {
                with_cvvdp = false;
                i += 1;
            }
            "--cvvdp-bin" => {
                cvvdp_bin = Some(arg(&args, i + 1)?.to_string());
                with_cvvdp = true;
                i += 2;
            }
            "--cvvdp-display" => {
                cvvdp_display = arg(&args, i + 1)?.to_string();
                i += 2;
            }
            "--cvvdp-device" => {
                cvvdp_device = Some(arg(&args, i + 1)?.to_string());
                i += 2;
            }
            "--butteraugli" => {
                with_butteraugli = true;
                i += 1;
            }
            "--no-butteraugli" => {
                with_butteraugli = false;
                i += 1;
            }
            "--butteraugli-bin" => {
                butteraugli_bin = arg(&args, i + 1)?.to_string();
                with_butteraugli = true;
                i += 2;
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
    // JPEG reference: prefer cjpegli (distance-native), then libjpeg-turbo
    // cjpeg, then the in-process image-crate encoder.
    let jpeg_tool = if available(&cjpegli) {
        JpegTool::Cjpegli(cjpegli.clone())
    } else if available("cjpeg") {
        JpegTool::Cjpeg
    } else {
        JpegTool::Builtin
    };
    // butteraugli 3-norm: on whenever libjxl's `butteraugli_main` is reachable.
    if with_butteraugli && !available(&butteraugli_bin) {
        eprintln!(
            "warning: butteraugli_main not found ({butteraugli_bin}); skipping the butteraugli \
             {BUTTERAUGLI_PNORM}-norm. Override with --butteraugli-bin or pass --no-butteraugli."
        );
        with_butteraugli = false;
    }
    let butteraugli = with_butteraugli.then_some(butteraugli_bin);

    // ColorVideoVDP: explicit --cvvdp-bin, else `cvvdp` on PATH, else the
    // parameters_fit venv. Warn + skip rather than aborting when none works.
    let mut cvvdp: Option<Cvvdp> = if with_cvvdp {
        let candidates: Vec<String> = match &cvvdp_bin {
            Some(b) => vec![b.clone()],
            None => vec![CVVDP_BIN.to_string(), CVVDP_VENV_BIN.to_string()],
        };
        match candidates.into_iter().find(|b| available(b)) {
            Some(bin) => {
                println!(
                    "cvvdp: {bin} (--display {cvvdp_display}{})",
                    cvvdp_device
                        .as_deref()
                        .map(|d| format!(" --device {d}"))
                        .unwrap_or_default()
                );
                Some(Cvvdp {
                    bin,
                    display: cvvdp_display.clone(),
                    device: cvvdp_device.clone(),
                    child: None,
                })
            }
            None => {
                eprintln!(
                    "warning: cvvdp not found (tried {}); skipping ColorVideoVDP. \
                     Install with `pip install git+https://github.com/gfxdisp/ColorVideoVDP.git` \
                     or pass --cvvdp-bin PATH.",
                    match &cvvdp_bin {
                        Some(b) => b.clone(),
                        None => format!("{CVVDP_BIN}, {CVVDP_VENV_BIN}"),
                    }
                );
                None
            }
        }
    } else {
        None
    };

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

    // Build the series list once; colors mirror `stats`.
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
    if with_jpeg {
        let label = match &jpeg_tool {
            JpegTool::Cjpegli(_) => "jpeg (cjpegli)",
            JpegTool::Cjpeg => "jpeg (cjpeg -optimize)",
            JpegTool::Builtin => "jpeg (image-rs)",
        };
        series.push(Series {
            label: label.into(),
            color: RGBColor(0x60, 0x60, 0x60),
            points: vec![],
        });
        kinds.push(Kind::Jpeg);
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

            // Encode/decode/SS2 every series first, then score all decoded
            // buffers of this image with a single cvvdp request.
            let mut done: Vec<(usize, Sample, Vec<u8>)> = Vec::with_capacity(kinds.len());
            for (idx, kind) in kinds.iter().enumerate() {
                let res = match kind {
                    Kind::Jixel => bench_jixel(&rgb, w, h, d, &tmp, stem, npx, threads, patches),
                    Kind::Cjxl(e) => bench_cjxl(img, *e, d, &rgb, w, h, &tmp, stem, npx),
                    Kind::Aom => bench_avif_aom(img, d, &rgb, w, h, &tmp, stem, npx, &tools),
                    Kind::Jpeg => bench_jpeg(img, d, &rgb, w, h, &tmp, stem, npx, &jpeg_tool),
                };
                match res {
                    Ok((s, dec)) => done.push((idx, s, dec)),
                    Err(e) => eprintln!(
                        "  {:<32} {:<18} SKIP: {e:#}",
                        short_name(img),
                        series[idx].label
                    ),
                }
            }
            // The file-fed metrics (butteraugli, cvvdp) share one set of PPMs:
            // the reference plus one decode per series, written once per image.
            if (butteraugli.is_some() || cvvdp.is_some()) && !done.is_empty() {
                let written = (|| -> Result<(PathBuf, Vec<PathBuf>)> {
                    let ref_ppm = tmp.join(format!("{stem}_ref.ppm"));
                    write_ppm(&ref_ppm, &rgb, w, h)?;
                    let mut tests = Vec::with_capacity(done.len());
                    for (idx, _, dec) in &done {
                        let t = tmp.join(format!("{stem}_{idx}_d{d}_dec.ppm"));
                        write_ppm(&t, dec, w, h)?;
                        tests.push(t);
                    }
                    Ok((ref_ppm, tests))
                })();
                match written {
                    Ok((ref_ppm, tests)) => {
                        if let Some(bin) = &butteraugli {
                            for ((idx, s, _), test) in done.iter_mut().zip(&tests) {
                                match score_butteraugli(bin, &ref_ppm, test) {
                                    Ok((norm, max)) => {
                                        s.ba = Some(norm);
                                        s.ba_max = Some(max);
                                    }
                                    Err(e) => eprintln!(
                                        "  {:<32} {:<18} butteraugli SKIP: {e:#}",
                                        short_name(img),
                                        series[*idx].label
                                    ),
                                }
                            }
                        }
                        if let Some(tool) = cvvdp.as_mut() {
                            match tool.score(&ref_ppm, &tests) {
                                Ok(jods) => {
                                    for ((_, s, _), jod) in done.iter_mut().zip(jods) {
                                        s.cvvdp = Some(jod);
                                    }
                                }
                                Err(e) => eprintln!(
                                    "  {:<32} cvvdp SKIP (all series): {e:#}",
                                    short_name(img)
                                ),
                            }
                        }
                        let _ = std::fs::remove_file(&ref_ppm);
                        for t in &tests {
                            let _ = std::fs::remove_file(t);
                        }
                    }
                    Err(e) => eprintln!("  {:<32} metrics SKIP (ppm): {e:#}", short_name(img)),
                }
            }
            for (idx, s, _) in done {
                buckets[idx].push(s);
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
            let jod: Vec<f64> = bucket.iter().filter_map(|p| p.cvvdp).collect();
            let ba: Vec<f64> = bucket.iter().filter_map(|p| p.ba).collect();
            let ba_max: Vec<f64> = bucket.iter().filter_map(|p| p.ba_max).collect();
            let mean_bpp = arith_mean(&bpp);
            let mean_ss2 = arith_mean(&ss2);
            let mean_cvvdp = (!jod.is_empty()).then(|| arith_mean(&jod));
            let mean_ba = (!ba.is_empty()).then(|| arith_mean(&ba));
            let mean_ba_max = (!ba_max.is_empty()).then(|| arith_mean(&ba_max));
            let ba_col = match mean_ba {
                Some(v) => {
                    let n = if ba.len() == bucket.len() {
                        String::new()
                    } else {
                        format!(" (n={})", ba.len())
                    };
                    match mean_ba_max {
                        Some(m) => format!("  BA{BUTTERAUGLI_PNORM} {v:.4} (max {m:.4}){n}"),
                        None => format!("  BA{BUTTERAUGLI_PNORM} {v:.4}{n}"),
                    }
                }
                None if butteraugli.is_some() => format!("  BA{BUTTERAUGLI_PNORM} n/a"),
                None => String::new(),
            };
            let cvvdp_col = match mean_cvvdp {
                Some(v) if jod.len() == bucket.len() => format!("  CVVDP {v:.4}"),
                Some(v) => format!("  CVVDP {v:.4} (n={})", jod.len()),
                None if cvvdp.is_some() => "  CVVDP n/a".to_string(),
                None => String::new(),
            };
            println!(
                "  {:<18} n={:<3}  bpp[arith {:.4} geo {}]  SS2 {:.4}{ba_col}{cvvdp_col}",
                s.label,
                bucket.len(),
                mean_bpp,
                fmt_geo(geo_mean(&bpp)),
                mean_ss2,
            );
            s.points.push(Point {
                bpp: mean_bpp,
                ss2: mean_ss2,
                ba: mean_ba,
                ba_max: mean_ba_max,
                cvvdp: mean_cvvdp,
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
        &YAxis {
            desc: "mean SSIMULACRA2 (higher = better)".into(),
            range: (0.0, 100.0),
            label_dy: 0.4,
            get: |p| Some(p.ss2),
        },
    )?;
    println!("chart -> {}", chart_path.display());
    if butteraugli.is_some() {
        let chart_path = out_dir.join(format!("{name}_mean_rd_butteraugli.png"));
        draw_chart(
            &chart_path,
            &format!(
                "{name} — mean butteraugli {BUTTERAUGLI_PNORM}-norm vs rate ({} images)",
                images.len()
            ),
            &series,
            &YAxis {
                desc: format!(
                    "mean butteraugli {BUTTERAUGLI_PNORM}-norm distance (lower = better)"
                ),
                range: (0.0, f64::INFINITY),
                label_dy: 0.02,
                get: |p| p.ba,
            },
        )?;
        println!("chart -> {}", chart_path.display());
        let chart_path = out_dir.join(format!("{name}_mean_rd_butteraugli_max.png"));
        draw_chart(
            &chart_path,
            &format!(
                "{name} — mean butteraugli max distance vs rate ({} images)",
                images.len()
            ),
            &series,
            &YAxis {
                desc: "mean butteraugli max distance (lower = better)".into(),
                range: (0.0, f64::INFINITY),
                label_dy: 0.05,
                get: |p| p.ba_max,
            },
        )?;
        println!("chart -> {}", chart_path.display());
    }
    if cvvdp.is_some() {
        let chart_path = out_dir.join(format!("{name}_mean_rd_cvvdp.png"));
        draw_chart(
            &chart_path,
            &format!("{name} — mean CVVDP vs rate ({} images)", images.len()),
            &series,
            &YAxis {
                desc: "mean CVVDP (JOD, 10 = identical, higher = better)".into(),
                range: (0.0, 10.0),
                label_dy: 0.04,
                get: |p| p.cvvdp,
            },
        )?;
        println!("chart -> {}", chart_path.display());
    }

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
    patches: bool,
) -> Result<(Sample, Vec<u8>)> {
    let cfg = jixel::EncodeConfig::default()
        .with_lossless(false)
        .with_distance(d)
        .with_num_threads(threads)
        .with_patches(patches)
        .with_speed(Speed::Slow);
    #[cfg(feature = "splines")]
    let cfg = cfg.with_splines(true);
    let data = jixel::encode_image(rgb, w, h, &cfg)
        .map_err(|e| anyhow::anyhow!("jixel encode failed: {e:?}"))?;
    let jxl = tmp.join(format!("{stem}_jixel_{d}.jxl"));
    std::fs::write(&jxl, &data)?;
    let bytes = data.len() as u64;
    let dec = decode_to_rgb(&jxl, tmp, w, h)?;
    let ss2 = score(rgb, &dec, w, h)?;
    Ok((
        Sample {
            bpp: bytes as f64 * 8.0 / npx,
            ss2,
            ba: None,
            ba_max: None,
            cvvdp: None,
        },
        dec,
    ))
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
) -> Result<(Sample, Vec<u8>)> {
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
    Ok((
        Sample {
            bpp: bytes as f64 * 8.0 / npx,
            ss2,
            ba: None,
            ba_max: None,
            cvvdp: None,
        },
        dec,
    ))
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
) -> Result<(Sample, Vec<u8>)> {
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
    Ok((
        Sample {
            bpp: bytes as f64 * 8.0 / npx,
            ss2,
            ba: None,
            ba_max: None,
            cvvdp: None,
        },
        dec,
    ))
}

#[allow(clippy::too_many_arguments)]
fn bench_jpeg(
    img: &Path,
    d: f32,
    orig: &[u8],
    w: usize,
    h: usize,
    tmp: &Path,
    stem: &str,
    npx: f64,
    tool: &JpegTool,
) -> Result<(Sample, Vec<u8>)> {
    let out = tmp.join(format!("{stem}_d{d}.jpg"));
    let _ = std::fs::remove_file(&out);
    let jpg: Vec<u8> = match tool {
        JpegTool::Cjpegli(bin) => {
            let output = Command::new(bin)
                .arg(img)
                .arg(&out)
                .arg("-d")
                .arg(d.to_string())
                .output()
                .context("running cjpegli")?;
            if !output.status.success() || !out.exists() {
                bail!(
                    "cjpegli failed for {} d={d}\nstderr: {}",
                    img.display(),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::fs::read(&out)?
        }
        JpegTool::Cjpeg => {
            // cjpeg reads PPM, not PNG: hand it the raw RGB as P6.
            let q = jpeg_quality_from_distance(d);
            let ppm = tmp.join(format!("{stem}.ppm"));
            let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
            data.extend_from_slice(orig);
            std::fs::write(&ppm, data)?;
            let output = Command::new("cjpeg")
                .arg("-quality")
                .arg(q.to_string())
                .arg("-optimize")
                .arg("-outfile")
                .arg(&out)
                .arg(&ppm)
                .output()
                .context("running cjpeg")?;
            let _ = std::fs::remove_file(&ppm);
            if !output.status.success() || !out.exists() {
                bail!(
                    "cjpeg failed for {} q={q}\nstderr: {}",
                    img.display(),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::fs::read(&out)?
        }
        JpegTool::Builtin => {
            let q = jpeg_quality_from_distance(d);
            let mut buf = Vec::new();
            let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
                std::io::Cursor::new(&mut buf),
                q,
            );
            image::ImageEncoder::write_image(
                enc,
                orig,
                w as u32,
                h as u32,
                image::ExtendedColorType::Rgb8,
            )
            .context("image-rs jpeg encode")?;
            buf
        }
    };
    let dec = image::load_from_memory_with_format(&jpg, image::ImageFormat::Jpeg)
        .context("decoding jpeg")?
        .to_rgb8();
    if (dec.width() as usize, dec.height() as usize) != (w, h) {
        bail!("jpeg decode size mismatch");
    }
    let ss2 = score(orig, dec.as_raw(), w, h)?;
    Ok((
        Sample {
            bpp: jpg.len() as f64 * 8.0 / npx,
            ss2,
            ba: None,
            ba_max: None,
            cvvdp: None,
        },
        dec.into_raw(),
    ))
}

/// Inverse of jixel's `distance_from_quality` (piecewise): the JPEG quality
/// whose jixel-equivalent distance is `d`, for the quality-based encoders.
fn jpeg_quality_from_distance(d: f32) -> u8 {
    let q = if d <= 0.05 {
        100.0
    } else if d <= 0.1 {
        100.0 - (d / 0.05).log2()
    } else if d <= 1.0 {
        99.0 - 9.0 * (1.0 + d.log10())
    } else if d <= 6.4 {
        100.0 - (d - 0.1) / 0.09
    } else {
        30.0 - 5.0 * ((d - 6.24) * 6.25).ln() / 2.5f32.ln()
    };
    q.round().clamp(1.0, 100.0) as u8
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

/// Write interleaved RGB8 as a binary PPM (P6): the cvvdp input format, so the
/// metric sees exactly the buffer SSIMULACRA2 scored.
fn write_ppm(path: &Path, rgb: &[u8], w: usize, h: usize) -> Result<()> {
    let mut f = BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
    );
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    f.flush()?;
    Ok(())
}

/// One cvvdp stdout line -> JOD. Accepts the quiet form (`8.1234`) and the
/// verbose one (`cvvdp=8.1234 [JOD]`); anything else (warnings, blank) is skipped.
fn parse_cvvdp_line(line: &str) -> Option<f64> {
    let body = match line.split_once('=') {
        Some((k, v)) if k.trim().eq_ignore_ascii_case("cvvdp") => v,
        Some(_) => return None,
        None => line,
    };
    body.split_whitespace().next()?.parse::<f64>().ok()
}

/// butteraugli distance between two images via libjxl's `butteraugli_main`.
/// Returns `(p-norm, max)`; both lower = better. Output shape is the max
/// distance on the first line, then "`<p>`-norm: `<value>`".
fn score_butteraugli(bin: &str, reference: &Path, dist: &Path) -> Result<(f64, f64)> {
    let output = Command::new(bin)
        .arg(reference)
        .arg(dist)
        .arg("--pnorm")
        .arg(BUTTERAUGLI_PNORM.to_string())
        .output()
        .with_context(|| format!("running butteraugli ({bin})"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "butteraugli ({bin}) failed for {}\nstdout: {stdout}\nstderr: {stderr}",
            dist.display()
        );
    }
    let tag = format!("{BUTTERAUGLI_PNORM}-norm:");
    let mut max = None;
    let mut norm = None;
    for line in stdout.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&tag) {
            norm = rest.trim().parse::<f64>().ok();
        } else if max.is_none() {
            max = line.parse::<f64>().ok();
        }
    }
    match (norm, max) {
        (Some(n), Some(m)) => Ok((n, m)),
        _ => bail!("could not parse butteraugli output:\n{stdout}"),
    }
}

/// SSIMULACRA2 between two interleaved RGB8 buffers.
fn score(orig: &[u8], dist: &[u8], w: usize, h: usize) -> Result<f64> {
    let to_rgb = |b: &[u8]| -> Result<Rgb> {
        let data: Vec<[f32; 3]> = b
            .as_chunks::<3>()
            .0
            .iter()
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

/// Which quality metric a chart plots on its y axis.
struct YAxis<F: Fn(&Point) -> Option<f64>> {
    desc: String,
    /// Hard clamp of the padded y range (metric's natural bounds).
    range: (f64, f64),
    /// Vertical offset of the per-point annotation, in axis units.
    label_dy: f64,
    /// Metric accessor; points returning `None` are left off the chart.
    get: F,
}

/// Aggregate R/D chart: one line per series, folder-mean metric vs folder-mean bpp.
fn draw_chart<F: Fn(&Point) -> Option<f64>>(
    path: &Path,
    title: &str,
    series: &[Series],
    y: &YAxis<F>,
) -> Result<()> {
    let root = BitMapBackend::new(path, (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;
    let (xmin, xmax, ymin, ymax) = bounds(series, y);
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 26))
        .margin(16)
        .x_label_area_size(48)
        .y_label_area_size(56)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;
    chart
        .configure_mesh()
        .x_desc("mean rate (bits / pixel)")
        .y_desc(y.desc.as_str())
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 14))
        .draw()?;
    for s in series {
        let plotted: Vec<(&Point, f64)> = s
            .points
            .iter()
            .filter_map(|p| (y.get)(p).map(|v| (p, v)))
            .collect();
        let mut pts: Vec<(f64, f64)> = plotted.iter().map(|(p, v)| (p.bpp, *v)).collect();
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
        chart.draw_series(plotted.iter().map(|(pt, v)| {
            let style = ("sans-serif", 14).into_font().color(&series_color);
            Text::new(pt.note.clone(), (pt.bpp + 0.01, v + y.label_dy), style)
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

fn bounds<F: Fn(&Point) -> Option<f64>>(series: &[Series], y: &YAxis<F>) -> (f64, f64, f64, f64) {
    let (mut xmn, mut xmx, mut ymn, mut ymx) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for s in series {
        for p in &s.points {
            let Some(v) = (y.get)(p) else { continue };
            xmn = xmn.min(p.bpp);
            xmx = xmx.max(p.bpp);
            ymn = ymn.min(v);
            ymx = ymx.max(v);
        }
    }
    // Guard against an all-empty chart (every series skipped).
    if xmn > xmx {
        return (0.0, 1.0, y.range.0, y.range.1);
    }
    let xpad = (xmx - xmn) * 0.05 + 1e-6;
    let ypad = (ymx - ymn) * 0.08 + 1e-6;
    (
        xmn - xpad,
        xmx + xpad,
        (ymn - ypad).max(y.range.0),
        (ymx + ypad).min(y.range.1),
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
         \x20                [--avifenc PATH] [--avifdec PATH] [--aom-speed 6] [--avif-yuv 444]\n\
         \x20                [--no-aom] [--no-cjxl] [--jpeg] [--cjpegli PATH]\n\
         \x20                [--cvvdp] [--cvvdp-bin PATH] [--cvvdp-display standard_4k] [--cvvdp-device mps|cpu]\n\
         \x20                [--no-cvvdp] [--no-butteraugli] [--butteraugli-bin PATH]\n\
         \n  Runs jixel, cjxl (per effort) and the libavif aom AV1 reference over every\n  \
         image in FOLDER at each distance, decodes, scores SSIMULACRA2 (the butteraugli\n  \
         3-norm too whenever butteraugli_main is on PATH, and ColorVideoVDP JOD with\n  \
         --cvvdp), prints the folder mean bpp/SS2[/BA3][/CVVDP] per series, and writes\n  \
         an aggregate R/D chart per metric to DIR."
    );
    std::process::exit(2);
}
