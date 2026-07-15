"""Paths and global settings for the jixel offline parameter study.

Everything is anchored to the repository root so the harness can be run from
anywhere (``python -m parameters_fit.optimize`` from the repo, or the scripts in
this folder).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# parameters_fit/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# The encode-only Rust CLI (built by `cargo build --release -p tuner`).
ENCODER = REPO_ROOT / "target" / "release" / "jixel-tuner"

# External measurement tools. Resolved on PATH (Homebrew installs both).
DJXL = shutil.which("djxl") or "djxl"
SSIMULACRA2 = shutil.which("ssimulacra2") or "ssimulacra2"

# Source corpus. Each entry is (directory, max_images). ``None`` uses every
# image in the directory; an int samples that many, spread evenly across the
# sorted listing (deterministic — no RNG). DIV2K has 800 images, far more than a
# search needs, so we take an evenly-spaced subset for content diversity.
SOURCE_DIRS: list[tuple[Path, int | None]] = [
    (REPO_ROOT / "assets" / "train0", None),            # all 16
    (REPO_ROOT / "assets" / "DIV2K_train_HR", 24),      # 24 of 800
]

# Back-compat alias (the first source dir).
SOURCE_DIR = SOURCE_DIRS[0][0]

# Working directories (all git-ignored via parameters_fit/.gitignore).
WORK_DIR = Path(__file__).resolve().parent / "work"
CROP_DIR = WORK_DIR / "crops"           # prepared source crops (PNG)
CACHE_DIR = WORK_DIR / "cache"          # baseline RD curves (JSON)
TMP_DIR = WORK_DIR / "tmp"              # per-encode scratch (.jxl/.png)
STUDY_DIR = WORK_DIR / "studies"        # Optuna SQLite databases + best params

# --- Corpus preparation -----------------------------------------------------
# One aligned square crop per source image. 768 is a multiple of 256, so it is
# aligned to the codec's largest (32x32 block = 256px group) boundaries.
CROP_SIZE = 768

# Deterministic train/holdout split. Every N-th image (0-based) is held out and
# never participates in optimization.
HOLDOUT_EVERY = 4  # -> 25% holdout

# --- Quality points ---------------------------------------------------------
# Search across the three anchor bands from the Level-2 plan (HQ ~0.1-0.7,
# mid ~1-3, low ~4-5) so a single global config is optimized not to regress
# anywhere — the earlier HQ-only study overfit d<=0.7 and hurt d>=1.
SEARCH_DISTANCES = [0.10, 0.30, 0.70, 1.50, 3.00, 5.00]

# Denser sweep used to build the per-crop baseline rate/quality curve, so a
# candidate encode at any rate can be compared against a matched baseline.
# Must span (with margin) both the search and validation distances.
BASELINE_DISTANCES = [
    0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70,
    1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 6.00,
]

# Final-validation sweep (run only on the best few configs).
VALIDATION_DISTANCES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00, 5.00]

# Encoder threads per encode. Each encode uses all cores; the study then runs
# ONE trial at a time (n_jobs=1) so encodes never contend, which keeps the
# measured time_ratio penalty meaningful and reproducible.
ENCODE_THREADS = os.cpu_count() or 1


def ensure_dirs() -> None:
    for d in (WORK_DIR, CROP_DIR, CACHE_DIR, TMP_DIR, STUDY_DIR):
        d.mkdir(parents=True, exist_ok=True)
